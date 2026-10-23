#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <math.h>

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
#include "libavutil/crc.h"
#include "libavutil/bswap.h"
#include "libavutil/samplefmt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/channel_layout.h"
#include "libavutil/audio_fifo.h"
#include "libavutil/threadmessage.h"
#include "libavutil/hwcontext.h"
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/filters.h"   /* 1.0.1-pre11: FilterLink.current_pts_us — resampler-boundary slip probe (in-tree build) */
#include "libswresample/swresample.h"

#include "ptvencoder.h"

/* 1.0.1-pre8 (d) SELF-MADE-GAP LOG HONESTY: when the demux/decode shed packets in the recent
 * window (video head/tail QSHED, audio drop-oldest — g_shed_wall/g_shed_cnt), the downstream
 * AGLUE/ASTEP lines must say so, so our own drops are never again misread as source
 * burstiness (the "bursty channel" taxonomy was self-portraiture — owner mandate). Returns ""
 * outside a shed window (all pinned log-line shapes byte-identical when nothing was shed);
 * inside the 5s window returns " [self: N pkts shed]" with N = sheds since this track's
 * window opened (per-track mark refreshed while quiet). 1.0.1-pre16: reads THIS TRACK'S
 * INPUT's shed stamp (Input.shed_wall/shed_cnt — identical to the globals on single input),
 * so on mv slot B's shed no longer annotates slot A's AGLUE/ASTEP lines. */
static const char *ptv_self_shed_note(AudioState *a, char *buf, size_t sz)
{
    int64_t w = a->shed_wall ? atomic_load_explicit(a->shed_wall, memory_order_relaxed)
                             : atomic_load_explicit(&g_shed_wall, memory_order_relaxed);
    int64_t c = a->shed_cnt  ? atomic_load_explicit(a->shed_cnt, memory_order_relaxed)
                             : atomic_load_explicit(&g_shed_cnt, memory_order_relaxed);
    if (!w || av_gettime_relative() - w > 5000000) {
        a->shed_mark = c;
        buf[0] = 0;
        return buf;
    }
    snprintf(buf, sz, " [self: %lld pkts shed]", (long long)(c - a->shed_mark));
    return buf;
}

#define PTV_REANCH_WIN_US   10000000   /* rebuild discontinuity window (µs) */
#define PTV_REANCH_MIN_FR   10         /* emitted frames before the (c) step may fire (~0.2s settle) */
#define PTV_REANCH_CAP_US    5000000   /* (c) step authority (µs) */
#define PTV_REANCH_BAND_US     80000   /* 2.0.2-pre3: the re-anchor paths keep the 2.0.1 80 ms band (a walk
                                          * ends inside it, a rebuild step fires outside it); only the
                                          * corrector's engage band moved to 40 ms (g_rscorr_engage_us) */

/* ---- audio path (decode -> resample 48k stereo -> AAC -> mux) ---- */

/* encode the SAME loudness-processed frame into each rung's own AAC encoder (so
 * per-rung -b:a is honored), routing each rung's packets to its muxer. The frame
 * is only ref'd by avcodec_send_frame, so the one frame feeds all N encoders.
 * frame=NULL flushes every encoder. (ffmpeg's filter -> asplit -> N encoders.) */
static int audio_encode_push(AudioState *a, AVFrame *frame)
{
    int i, j, ret;
    for (i = 0; i < a->n_out; i++) {
        if (a->enc_owner[i] != i)
            continue;                       /* this rung rides another rung's encoder */
        ret = avcodec_send_frame(a->enc[i], frame);
        if (ret < 0)
            return ret;
        for (;;) {
            AVPacket *pkt = av_packet_alloc();
            if (!pkt)
                return AVERROR(ENOMEM);
            ret = avcodec_receive_packet(a->enc[i], pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
            if (ret < 0) { av_packet_free(&pkt); return ret; }
            /* fan this encode out to every rung sharing it. Each still gets its OWN rescale,
             * stream_index and delivery gate — only the encode itself is shared, and the
             * bytes are identical by construction (same encoder, frames and settings). */
            for (j = 0; j < a->n_out; j++) {
                AVPacket *out;
                if (a->enc_owner[j] != i)
                    continue;
                out = av_packet_clone(pkt);
                if (!out) { av_packet_free(&pkt); return AVERROR(ENOMEM); }
                av_packet_rescale_ts(out, a->enc[i]->time_base, a->ost[j]->time_base);
                out->stream_index = a->ost[j]->index;
                if (a->gate[j] && (out->dts != AV_NOPTS_VALUE || out->pts != AV_NOPTS_VALUE)) {
                    /* §7.5a: hold until this rung's video front reaches it (block=1 → back-pressure) */
                    int64_t ts = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
                    dlv_enqueue(a->gate[j], out, av_rescale_q(ts, a->ost[j]->time_base, AV_TIME_BASE_Q), 1);
                } else if (av_thread_message_queue_send(a->mux_q[j], &out, 0) < 0)   /* blocking */
                    av_packet_free(&out);
            }
            av_packet_free(&pkt);
        }
    }
    return 0;
}

static int audio_drain_fifo(AudioState *a)
{
    int ret = 0;
    while (av_audio_fifo_size(a->fifo) >= a->frame_size) {
        AVFrame *f = av_frame_alloc();
        if (!f) return AVERROR(ENOMEM);
        f->nb_samples  = a->frame_size;
        f->format      = a->out_sfmt;
        f->sample_rate = a->out_rate;
        av_channel_layout_copy(&f->ch_layout, &a->out_chl);
        if ((ret = av_frame_get_buffer(f, 0)) < 0) { av_frame_free(&f); return ret; }
        av_audio_fifo_read(a->fifo, (void **)f->data, a->frame_size);
        f->pts = a->next_pts;
        a->next_pts += a->frame_size;
        ret = audio_encode_push(a, f);
        a->out_frames++;
        av_frame_free(&f);
        if (ret < 0) return ret;
    }
    return 0;
}

/* ===================== 1.0.1-pre14 — residual-sync CORRECTOR =====================
 * Component 2 of the residual-sync supervisor; NORMATIVE DESIGN =
 * analysis/ptvencoder-corrector-design.md (§ refs below are its sections). Everything here
 * runs on the owning track's audio thread, once per EMITTED audio frame, immediately after
 * the pre9 sensor publishes that frame's m_a — so R is always current when consumed.
 *
 * SENSOR CONTRACT (§2): the corrector consumes EXACTLY ONE signal — the per-track residual
 *   R = (m_v + E_v) − (m_a + E_a)   (µs; + = audio EARLY, the external oracle's convention)
 * read from the g_rsx atomics through rsync_track_R() below. It must NEVER read `avlag=`
 * ([PTV-AVSYNC]), `[PTV-LIPSYNC] err=`, `[PTV-CHAIN] introduced=`, `async_pad`, or any other
 * internal estimate — those are control-domain proxies with their own (opposite) sign
 * conventions; the pre13 avlag rename exists because that collision already caused one
 * oracle-analysis error. Actuation sign law: injecting +corr at the graph door delays audio
 * content → raises m_a → dR/dcorr = −1.
 *
 * ACTUATOR (§4/§5): the pre3 steer bus at the audio_feed graph door — corr_us joins
 * glue_off + house_skew + af_steer_us as a fourth linearly-summed term, realized by
 * aresample=async as bounded SOFT compensation (per-frame steps ~43µs at 21.3ms frames,
 * three orders under ACOMP's 25ms click threshold). Renounced actuators, permanently:
 * output drop/pad (B4), output label re-stamping (pre3/TRACKUP warble), per-stream packet
 * offsets (standing owner ban). Anti-windup is structural (§4): (a) corr_us enters the
 * sensor's inj term so R feeds back only the REALIZED trim, (b) integration freezes while
 * the slip probe reads nonzero (resampler behind = pushing harder is windup by definition),
 * (c) any engage-condition loss freezes integration the same frame; corr_us is the only
 * state and it never holds more than realized-or-in-flight trim. */

/* §2 accessor (MV-NORMATIVE §8.3): track k's residual, pairing its audio terms with its
 * INPUT SLOT's video terms. 1.0.1-pre16: g_rsx carries PER-SLOT video ledgers and the slot
 * hard-wire died here — mv_wall/mv_ema/ev_us are read at a->dbg_in (identically slot 0 on
 * single input). Nothing else in the corrector changed — exactly the §8.3 promise. */
typedef struct RsyncTrackR { int valid; int64_t R_us, L_us, mv_wall, ma_wall; } RsyncTrackR;  /* L_us: the ledger part ev − ea (2.0.2) */
/* TEST-ONLY (PTV_RSCORR_TESTWALK, µs/s; PTV_PLL_TESTNOISE_MS precedent): add a linearly
 * walking offset to the R the CORRECTOR reads (the sensor/stats stay untouched) — the
 * "maximally lying-but-plausible sensor" of §6's damage bound, the only way to exercise the
 * per-engagement authority cap in bounded wall time (a real R never walks: label-faithful
 * drift reads R=0 by design). Never set in production. */
static int64_t g_rscorr_testwalk_us_s = 0, g_rscorr_testwalk_t0 = 0;
/* pre17 fix round (F2): PTV_RSCORR_TESTWALK_CAP_MS saturates the walk — without it the
 * linear-forever walk is cancelled by the steer at equal rate, R never re-enters the park
 * band, and mv ENGAGE→PARK is structurally unreachable via TESTWALK (reviewer F2). A capped
 * walk = a synthetic SETTLED bake the corrector can steer to 0 and PARK on. TEST-ONLY.
 * pre29: the cap saturates by MAGNITUDE (a negative walk saturates at −cap — a settled
 * NEGATIVE bake); positive-walk behavior unchanged. PTV_RSCORR_TESTWALK_DECAY_AT_S zeroes
 * the walk t seconds after its onset (a TRANSIENT). pre30: PTV_RSCORR_TESTWALK_PAUSE_AT_S +
 * PTV_RSCORR_TESTWALK_PAUSE_S (and the _PAUSE2_ pair) zero the walk inside
 * [onset+pause_at, onset+pause_at+pause_len). Built as fixtures for the RESYNC engine
 * (deleted in 2.1). TEST-ONLY. */
static int64_t g_rscorr_testwalk_cap_us = 0, g_rscorr_testwalk_decay_us = 0;
static int64_t g_rscorr_testwalk_pause_at_us = 0, g_rscorr_testwalk_pause_len_us = 0;
static int64_t g_rscorr_testwalk_pause2_at_us = 0, g_rscorr_testwalk_pause2_len_us = 0;
static RsyncTrackR rsync_track_R(const AudioState *a, int64_t now)
{
    RsyncTrackR r = { 0, 0, 0, 0, 0 };
    int k = a->dbg_k, in = a->dbg_in;
    if (k < 0 || k >= PTV_MAX_AUDIO || in < 0 || in >= PTV_MAX_INPUT || !a->rs_ma_seed)
        return r;
    r.mv_wall = atomic_load_explicit(&g_rsx.mv_wall[in], memory_order_relaxed);
    r.ma_wall = atomic_load_explicit(&g_rsx.ma_wall[k], memory_order_relaxed);
    /* freshness (§2): a side that has not flowed for 3s is stale ("--" on the stats line)
     * — no reading, the corrector holds/disarms. Per slot: audio flowing over a slated
     * cell = video-side stale = no reading (the single-input stale-anchor rule). */
    if (!r.mv_wall || !r.ma_wall || now - r.mv_wall > 3000000 || now - r.ma_wall > 3000000)
        return r;
    r.L_us = atomic_load_explicit(&g_rsx.ev_us[in], memory_order_relaxed)
           - atomic_load_explicit(&g_rsx.ea_us[k],  memory_order_relaxed);
    r.R_us = atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed) - a->rs_ma_ema + r.L_us;
    if (g_rscorr_testwalk_us_s == 0) {
        const char *tw = getenv("PTV_RSCORR_TESTWALK");
        const char *ta = getenv("PTV_RSCORR_TESTWALK_AT_S");
        const char *tc = getenv("PTV_RSCORR_TESTWALK_CAP_MS");
        const char *td = getenv("PTV_RSCORR_TESTWALK_DECAY_AT_S");
        const char *tp = getenv("PTV_RSCORR_TESTWALK_PAUSE_AT_S");
        const char *tl = getenv("PTV_RSCORR_TESTWALK_PAUSE_S");
        const char *tp2 = getenv("PTV_RSCORR_TESTWALK_PAUSE2_AT_S");
        const char *tl2 = getenv("PTV_RSCORR_TESTWALK_PAUSE2_S");
        g_rscorr_testwalk_us_s = (tw && atoi(tw)) ? atoi(tw) : -1;   /* -1 = parsed, off */
        g_rscorr_testwalk_cap_us = (tc && atoi(tc) > 0) ? (int64_t)atoi(tc) * 1000 : 0;
        g_rscorr_testwalk_decay_us = (td && atoi(td) > 0) ? (int64_t)atoi(td) * 1000000 : 0;
        g_rscorr_testwalk_pause_at_us  = (tp && atoi(tp) > 0) ? (int64_t)atoi(tp) * 1000000 : 0;
        g_rscorr_testwalk_pause_len_us = (tl && atoi(tl) > 0) ? (int64_t)atoi(tl) * 1000000 : 0;
        g_rscorr_testwalk_pause2_at_us  = (tp2 && atoi(tp2) > 0) ? (int64_t)atoi(tp2) * 1000000 : 0;
        g_rscorr_testwalk_pause2_len_us = (tl2 && atoi(tl2) > 0) ? (int64_t)atoi(tl2) * 1000000 : 0;
        /* PTV_RSCORR_TESTWALK_AT_S delays the walk's onset — the cap is only reachable
         * through MID-ENGAGEMENT drift (§6): a walk from birth is correctly rejected by the
         * dwell stability bound and can never engage (verified, first fixture round). */
        g_rscorr_testwalk_t0 = now + (ta ? (int64_t)atoi(ta) * 1000000 : 0);
    }
    if (g_rscorr_testwalk_us_s != 0 && g_rscorr_testwalk_us_s != -1 &&
        now > g_rscorr_testwalk_t0) {
        /* pre29: negative rates are now legal (audio-LATE fixtures) — av_rescale requires
         * b >= 0 (returns INT64_MIN otherwise), so rescale the magnitude and re-sign. */
        int64_t w = av_rescale(now - g_rscorr_testwalk_t0,
                               llabs(g_rscorr_testwalk_us_s), 1000000);
        if (g_rscorr_testwalk_us_s < 0)
            w = -w;
        if (g_rscorr_testwalk_cap_us > 0 && llabs(w) > g_rscorr_testwalk_cap_us)
            w = w > 0 ? g_rscorr_testwalk_cap_us   /* F2: saturated walk = a steerable settled bake */
                      : -g_rscorr_testwalk_cap_us; /* pre29: magnitude cap (negative bake) */
        if (g_rscorr_testwalk_decay_us > 0 &&
            now - g_rscorr_testwalk_t0 > g_rscorr_testwalk_decay_us)
            w = 0;                                 /* pre29: transient fixture — the "source recovered" */
        if (g_rscorr_testwalk_pause_at_us > 0 &&
            now - g_rscorr_testwalk_t0 >  g_rscorr_testwalk_pause_at_us &&
            now - g_rscorr_testwalk_t0 <= g_rscorr_testwalk_pause_at_us +
                                          g_rscorr_testwalk_pause_len_us)
            w = 0;                                 /* pre30 FX-I: designed fire gap (TEST-ONLY) */
        if (g_rscorr_testwalk_pause2_at_us > 0 &&
            now - g_rscorr_testwalk_t0 >  g_rscorr_testwalk_pause2_at_us &&
            now - g_rscorr_testwalk_t0 <= g_rscorr_testwalk_pause2_at_us +
                                          g_rscorr_testwalk_pause2_len_us)
            w = 0;                                 /* pre30 FX-I: second designed gap (TEST-ONLY) */
        r.R_us += w;
        r.L_us += w;   /* the walk stands in for a ledger bake (the ceiling's flatness reads it) */
    }
    r.valid = 1;
    return r;
}

/* §3 delivery-liveness gate: actuation requires the wire to be provably moving on EVERY
 * rung AND the steered track's input to be flowing — one dead rung holds the corrector for
 * the whole channel (the steer is upstream of the per-rung fan-out, and a dead rung is
 * exactly the state in which R has been shown to lie: Newsmax2 read a serene +174ms with
 * rung 6000 dead). Signals per rung: g_mux_sent_wc (the wire itself — PRIMARY, §9.3
 * owner-approved) + DlvGate a_hi/v_hi_change_wc watermarks + mux_q depth below half
 * capacity (a backed-up mux_q = the muxer/socket is not draining). Gate NULL
 * (PTV_NO_DELIVERY / no-audio) → the watermark + depth checks remain. Per-input
 * (1.0.1-pre17, the §3 per-input wiring): the steered track's OWN input's arrival
 * watermark (Input.v_arrive_wc; the g_v_arrive_wc any-input aggregate is the unwired
 * fallback, identical on single input) — on mv a dead sibling must not read as "flowing"
 * and, conversely, THIS input dying must hold this track even while siblings flow. The
 * rung-wire watermarks stay per-RUNG deliberately (§3 gate rule: one dead rung anywhere
 * holds the corrector for the whole channel — the steer is upstream of the rung fan-out).
 * The track's own ma_wall freshness is already in the §2 sensor validity. */
static const char *rscorr_delivery_dead(AudioState *a, int64_t now)
{
    int i;
    int64_t arr = a->v_arrive_wc
                ? atomic_load_explicit(a->v_arrive_wc, memory_order_relaxed)
                : atomic_load_explicit(&g_v_arrive_wc, memory_order_relaxed);
    if (!arr || now - arr > 5000000)
        return "input not flowing";
    for (i = 0; i < a->n_out; i++) {
        int64_t ms = atomic_load_explicit(&g_mux_sent_wc[i], memory_order_relaxed);
        if (!ms || now - ms > 5000000)
            return "rung wire stale";
        if (a->mux_q[i] &&
            av_thread_message_queue_nb_elems(a->mux_q[i]) > PTV_QDEPTH / 2)
            return "mux_q backed up";
        if (a->gate[i]) {
            int64_t ah = atomic_load_explicit(&a->gate[i]->a_hi_change_wc, memory_order_relaxed);
            int64_t vh = atomic_load_explicit(&a->gate[i]->v_hi_change_wc, memory_order_relaxed);
            if (!ah || now - ah > 5000000) return "audio delivery stale";
            if (!vh || now - vh > 5000000) return "video encode stale";
        }
    }
    return NULL;
}

/* TEST-ONLY (1.0.1-pre18; PTV_RSCORR_TESTWALK precedent): PTV_RSCORR_TESTHS="amp_ms:period_s"
 * superimposes a TRIANGLE STAIRCASE (0→amp→2amp→amp→0→…, one ±amp step per period) on the
 * house_skew value the CORRECTOR's event detector reads — the bounded-wall-time stand-in
 * for pulldown/decim 1-tick hs churn (the #51a live class: hs WALKS in tick steps, so the
 * cumulative pre18 rule fires every ~2 steps while each individual step stays ≤1 tick —
 * a square wave cannot model that: no amplitude both crosses the old 50ms cumulative edge
 * AND stays under the 1.25-tick per-step filter). Exercises the hs-tick filter and the
 * #51b starvation ceiling without a production channel. The REAL house_skew (actuation,
 * sensors, AVLOCK) is untouched. Never set in production. */
static int64_t g_rscorr_tesths_amp = -1, g_rscorr_tesths_per = 0;
static int64_t rscorr_hs_read(AudioState *a)
{
    int64_t v = a->house_skew ? *a->house_skew : 0;
    if (g_rscorr_tesths_amp < 0) {
        const char *s = getenv("PTV_RSCORR_TESTHS");
        int amp_ms = 0, per_s = 15;
        if (s && sscanf(s, "%d:%d", &amp_ms, &per_s) >= 1 && amp_ms > 0 && per_s > 0) {
            g_rscorr_tesths_amp = (int64_t)amp_ms * 1000;
            g_rscorr_tesths_per = (int64_t)per_s * 1000000;
        } else
            g_rscorr_tesths_amp = 0;   /* parsed, off */
    }
    if (g_rscorr_tesths_amp > 0) {
        /* 6-level walk 0→1→2→3→2→1: per-step ±amp (1 tick, filtered by #51a), but the
         * RANGE spans 3 amps so the pre18 cumulative-50ms rule fires from ANY snapshot
         * level within ≤2 periods (a 4-level 0,1,2,1 triangle never fires when the dwell
         * snapshots at the mid level — measured, first battery round). */
        static const int tri[6] = { 0, 1, 2, 3, 2, 1 };
        v += g_rscorr_tesths_amp *
             tri[(av_gettime_relative() / g_rscorr_tesths_per) % 6];
    }
    return v;
}

/* §4.4 event-EDGE detector: any pipeline event that touches this track's label lineage,
 * detected as a snapshot delta and CONSUMED (snapshots advance) so one event fires once.
 * Returns the reason string, or NULL. */
static const char *rscorr_event_edge(AudioState *a)
{
    CorrState *c = &a->corr;
    int64_t v;
    v = a->corr_epoch ? atomic_load_explicit(a->corr_epoch, memory_order_relaxed) : 0;
    if (v != c->epoch_snap) { c->epoch_snap = v; return "disturb_epoch"; }
    if (a->glue_events != c->glue_snap) { c->glue_snap = a->glue_events; return "aglue"; }
    if (a->afmt_rebuilds != c->afmt_snap) { c->afmt_snap = a->afmt_rebuilds; return "afmt-rebuild"; }
    if (a->dec_reopens != c->reopen_snap) { c->reopen_snap = a->dec_reopens; return "adecwd-reopen"; }
    v = atomic_load_explicit(&g_rsx.ev_us[a->dbg_in], memory_order_relaxed);   /* pre16: own slot's ledger */
    if (v != c->ev_snap) { c->ev_snap = v; return "E_v ledger"; }
    v = atomic_load_explicit(&g_rsx.ea_us[a->dbg_k], memory_order_relaxed);
    if (v != c->ea_snap) { c->ea_snap = v; return "E_a ledger"; }
    v = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
    if (v != c->bank_snap) { c->bank_snap = v; return "bank change"; }
    if (a->house_skew) {
        v = rscorr_hs_read(a);
        if (g_hstick_filter) {
            /* 1.0.1-pre18 #51a (AWE dwell starvation, live 2026-07-19): on pulldown/decim
             * channels hs ticks ±1 video tick every 10-17s FOREVER (a benign cadence
             * artifact — R measured flat +2380..+2384 through every tick), and the old
             * cumulative-50ms rule fired every couple of ticks → the dwell never
             * accumulated (or, with bursty ticking, nibbled ~60ms per lucky quiet window
             * between 10min storm holdoffs) and the corrector never engaged on a
             * +2.38s bake. Magnitude-filter the EDGE: a step ≤ 1 tick + ¼ tick is NOT an
             * event — absorb it into the snapshot silently (no dwell reset, no storm
             * count). ≥2 ticks stays an event (the threshold is 1.25 ticks < 2 ticks).
             * The §4.3 continuous R-stability guard remains the actuation safety for
             * anything a tick-sized hs step actually does to R. All other named events
             * keep full event status. PTV_NO_HSTICK_FILTER=1 reverts to the cumulative
             * 50ms rule. */
            int64_t tol = a->tick_dur_us > 0 ? a->tick_dur_us + a->tick_dur_us / 4 : 50000;
            if (llabs(v - c->hs_snap) > tol) { c->hs_snap = v; return "hs step"; }
            c->hs_snap = v;   /* tick-sized churn: rebase the reference, never accumulate */
        } else if (llabs(v - c->hs_snap) >= 50000) { c->hs_snap = v; return "hs step"; }
    }
    return NULL;
}

/* §4.4 event-ACTIVE detector: conditions that must be INACTIVE across the whole quiet
 * window (returned every call while they hold; no state consumed). */
static const char *rscorr_event_active(AudioState *a, int64_t now)
{
    int64_t v;
    if (a->nbs_fill_active)   /* pre15 §5 rule 2: R is synthetic-flat on a filled track — never engage */
        return "nbs silence-fill";
    /* 2.0.0-pre4 (T-056 §5.5): while the source is held, R grows by the hold length and is no desync;
     * after the rejoin the readings re-settle — wait out a full quiet window before engaging. */
    if (ptv_src_holding())
        return "source hold";
    if (g_src_hold_act) {
        int64_t rj = atomic_load_explicit(&g_src_rejoin_wall, memory_order_relaxed);
        if (rj && now - rj < g_rscorr_quiet_us)
            return "recent rejoin";
    }
    /* 1.0.1-pre17 sibling-slate condition (finding 1, grid soak 2026-07-19): while ANY mv
     * slot is black-slated, NO track on the mosaic may engage — a sibling outage disturbs
     * the shared compositor pacing, and the soak measured the healthy slots' readings
     * drifting ~−150ms under exactly this condition (an artifact, fixed in the pre17
     * sensor; this freeze is the §4.4 belt over that fix). The slated slot's own tracks
     * are already held by `--` freshness; this extends the hold mosaic-wide. Clearing the
     * slate re-runs the 3min quiet window before any engage. */
    if (a->multiview && atomic_load_explicit(&g_mv_slate_mask, memory_order_relaxed))
        return "sibling slate";
    /* pre16: per-input feeds — THIS track's input's governor flag + shed stamp (identical to
     * the globals on single input; on mv slot B's shed/governor no longer freezes slot A). */
    if (a->gov_on ? atomic_load_explicit(a->gov_on, memory_order_relaxed)
                  : atomic_load_explicit(&g_gov_on, memory_order_relaxed))
        return "catch-up governor";
    v = a->shed_wall ? atomic_load_explicit(a->shed_wall, memory_order_relaxed)
                     : atomic_load_explicit(&g_shed_wall, memory_order_relaxed);
    if (v && now - v < g_rscorr_quiet_us)
        return "recent self-shed";
    return NULL;
}

/* re-anchor the dwell (fresh 5min) and re-snapshot every event feed */
static void rscorr_dwell_reset(AudioState *a, int64_t now, int64_t R)
{
    CorrState *c = &a->corr;
    c->dwell_start_wc = now;
    c->dwell_r0       = R;
    c->epoch_snap  = a->corr_epoch ? atomic_load_explicit(a->corr_epoch, memory_order_relaxed) : 0;
    c->glue_snap   = a->glue_events;
    c->afmt_snap   = a->afmt_rebuilds;
    c->reopen_snap = a->dec_reopens;
    c->ev_snap     = atomic_load_explicit(&g_rsx.ev_us[a->dbg_in], memory_order_relaxed);   /* pre16: own slot */
    c->ea_snap     = atomic_load_explicit(&g_rsx.ea_us[a->dbg_k], memory_order_relaxed);
    c->bank_snap   = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
    c->hs_snap     = rscorr_hs_read(a);
}

/* §6 event-storm accounting: ≥3 COUNTED dwell resets within 10min → disarm ("stop
 * oscillating on a churny channel"). A reset is counted only when ≥10s of dwell had
 * accumulated — continuous churn never accumulates dwell (so it neither engages NOR
 * trips the storm; the dwell alone already holds it), while a channel that repeatedly
 * builds real quiet stretches and loses them to fresh events is exactly the oscillation
 * the storm disarm exists for. (Deviation note: the doc does not define the counting
 * granularity; without this floor every churn frame would count and any churny channel
 * would disarm instantly and permanently.) Returns 1 when the storm threshold tripped. */
static int rscorr_storm_count(AudioState *a, int64_t now, int64_t accum_us)
{
    CorrState *c = &a->corr;
    if (accum_us < 10000000)
        return 0;
    if (!c->rst_win_wc || now - c->rst_win_wc > 600000000) {
        c->rst_win_wc = now;
        c->rst_cnt    = 1;
        return 0;
    }
    return ++c->rst_cnt >= 3;
}

static void rscorr_disarm(AudioState *a, int64_t now, const char *why, int is_error)
{
    CorrState *c = &a->corr;
    av_log(NULL, is_error ? AV_LOG_ERROR : AV_LOG_WARNING,
           "[PTV-RSCORR] a%d(in%d) DISARM (%s) corr held %+"PRId64"ms  [+ = audio early]\n",
           a->dbg_k, a->dbg_in, why, c->corr_us / 1000);
    c->state    = PTV_CORR_DISARMED;
    c->park_wc  = 0;
    c->slip_bad_wc = 0;
    /* every disarm carries a re-arm holdoff (60s; the storm path raises it to 10min at its
     * call site) — without one, a persisting disarm condition (e.g. implausible R) flapped
     * DISARM→re-arm per frame (first fixture round). The holdoff also makes the re-arm line
     * mean something: "recovered" = the condition stayed clear for the whole holdoff. */
    if (c->holdoff_wc < now + 60000000)
        c->holdoff_wc = now + 60000000;
}

/* pre14 corrector stale-track watchdog (1.0.1-pre17: moved here from the ptvencoder_clock.c
 * master block, body unchanged, so BOTH cadence owners can run it — the mv passthrough rung
 * loop returns before the single-input master block, so on mv it never ran; the compositor
 * now calls it once per second, port-doc §6's named arming prerequisite). A DWELL/ENGAGED
 * track whose audio thread stopped emitting (ma_wall stale >5s — source-audio death, the
 * pre12 W3 class) can neither steer NOR log its own disarm (its thread is blocked in recv).
 * Integration is inherently frozen (no frames), so this is purely the §6 one-line disarm +
 * published-state sync: CAS the published state to DISARMED (exactly one line even if the
 * track resumes mid-check) and hand the owning thread a silent disarm_req to consume.
 * Callers rate-limit to ~1s. */
void rscorr_stale_watchdog(int64_t now)
{
    int kc;
    if (!g_rsync_corr)
        return;
    for (kc = 0; kc < g_rsx.n_a; kc++) {
        int st = atomic_load_explicit(&g_corr_state_pub[kc], memory_order_relaxed);
        if (st == PTV_CORR_DWELL || st == PTV_CORR_ENGAGED) {
            int64_t maw = atomic_load_explicit(&g_rsx.ma_wall[kc], memory_order_relaxed);
            if (maw && now - maw > 5000000 &&
                atomic_compare_exchange_strong_explicit(&g_corr_state_pub[kc], &st,
                        PTV_CORR_DISARMED, memory_order_relaxed, memory_order_relaxed)) {
                atomic_store_explicit(&g_corr_disarm_req[kc], 1, memory_order_relaxed);
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-RSCORR] a%d DISARM (track stopped emitting — sensor stale) "
                       "corr held %+"PRId64"ms  [+ = audio early]\n",
                       kc, atomic_load_explicit(&g_corr_pub[kc], memory_order_relaxed) / 1000);
            }
        }
    }
}

/* §4 control law + state machine, one call per EMITTED audio frame (f_us = its duration).
 * R/valid come from rsync_track_R() computed by the caller right after the sensor publish. */
static void rscorr_update(AudioState *a, RsyncTrackR *r, int64_t f_us, int64_t now)
{
    CorrState *c = &a->corr;
    const char *dead, *ev;
    int64_t R = r->R_us;

    /* 1.0.1-pre17: the pre16 `if (a->multiview) return;` sensor-first hold is REMOVED — the
     * mv corrector is ARMED (mv-sensor-port §6's named removal site, after the observation
     * soak + the finding-1 sibling-slate sensor artifact fix). The arming prerequisites all
     * landed with this pre: per-INPUT liveness (rscorr_delivery_dead reads THIS track's
     * input's arrival watermark), the sibling-slate freeze (rscorr_event_active,
     * g_mv_slate_mask), and the stale-track watchdog re-homed so it runs under the mv
     * cadence owner too (rscorr_stale_watchdog, called by compositor + master rung).
     * NOTE: the SHIP decision for this arm is soak-gated — merge is the owner's call. */
    if (!g_rsync_corr)
        return;

    /* master stale-track watchdog handoff (ptvencoder_clock.c): the one disarm the owning
     * thread cannot log itself (it was blocked — no frames). Sync silently; the line is out. */
    if (atomic_exchange_explicit(&g_corr_disarm_req[a->dbg_k], 0, memory_order_relaxed) &&
        (c->state == PTV_CORR_DWELL || c->state == PTV_CORR_ENGAGED)) {
        c->state = PTV_CORR_DISARMED;
        c->holdoff_wc = now + 60000000;   /* same universal re-arm holdoff as rscorr_disarm() */
    }

    if (c->state == PTV_CORR_OFF) {
        c->state = PTV_CORR_ARMED;
        av_log(NULL, AV_LOG_INFO,
               "[PTV-RSCORR] a%d(in%d) armed (engage band %"PRId64"ms, dwell %"PRId64"s + %"PRId64"s quiet, "
               "slew %"PRId64"ms/s)  [+ = audio early]\n",
               a->dbg_k, a->dbg_in, g_rscorr_engage_us / 1000, g_rscorr_dwell_us / 1000000,
               g_rscorr_quiet_us / 1000000, g_rscorr_slew_us_s / 1000);
    }

    /* §6 implausibility tracker (log-only territory per the supervisor: an R this large is
     * a broken sensor, not a residual) */
    if (r->valid && llabs(R) > 5000000) {
        if (!c->implaus_wc) c->implaus_wc = now;
    } else
        c->implaus_wc = 0;

    dead = rscorr_delivery_dead(a, now);

    /* 1.0.1-pre18 #51b ANTI-STARVATION CEILING (the legacy-0007 PLL_HARD_CEILING 60min +
     * PLL_STUCK |baseline|>2s & drift<50ms pattern, sized to the certified sensor): a channel
     * whose R has stayed LARGE and FLAT for ≥ the ceiling while the corrector could never
     * complete a dwell (event resets and storm holdoffs included — AWE live 2026-07-19: R
     * flat +2374..+2385 over 28min of tick churn) ENGAGES anyway with one WARNING. The
     * FLATNESS requirement is load-bearing: the span restarts whenever R moves beyond the
     * §4.3 bound vs the span's own reference, so a genuinely churning R can never
     * ceiling-engage. Sensor-invalid, delivery-dead and implausible R all CLOSE the span
     * (they must keep blocking); the storm-disarm holdoff deliberately does NOT (that is
     * the point of the ceiling). Runs in ARMED/DWELL/DISARMED; steering/parked states
     * clear the span. PTV_NO_RSCORR_CEIL=1 reverts; PTV_RSCORR_CEIL_MIN tunes (15min).
     * 2.0.2: flatness is judged on the LEDGER part of R (ev − ea), not on the dm term the PLL
     * moves with its own pads / drops: live-transcoder MV_2x2_RAV slot 2 2026-10-19 sat at
     * R ≈ +1.1 s (ev +1092 ms, flat) for 3.7 h while each PLL ACQUIRE swung dm ±150–430 ms
     * for 10–80 s and restarted the span (72 of 102 restarts within 40 s of an ACQUIRE).
     * PTV_NO_CEIL_LEDGER=1 reverts to the whole R. */
    if (g_rscorr_ceil &&
        (c->state == PTV_CORR_ARMED || c->state == PTV_CORR_DWELL ||
         c->state == PTV_CORR_DISARMED)) {
        if (!r->valid || dead || c->implaus_wc || llabs(R) <= g_rscorr_engage_us ||
            a->nbs_fill_active) {   /* 2.0.2: R on a filled track is synthetic (rule above) — live-transcoder Grid_2x2 slot 2
                                     * 2026-10-18 ceiling-ENGAGED at a flat −237 ms after its source dropped the audio PID */
            c->starve_wc = 0;
        } else if (!c->starve_wc ||
                   llabs((g_ceil_ledger ? r->L_us : R) - c->starve_r0) >= FFMAX(40000, llabs(R) / 4)) {
            c->starve_wc = now;      /* open — or churning R: restart from here */
            c->starve_r0 = g_ceil_ledger ? r->L_us : R;
        } else if (now - c->starve_wc >= g_rscorr_ceil_us) {
            /* re-snapshot every event feed first — entering ENGAGED from a long
             * DISARM/holdoff would otherwise read the accumulated feed deltas as fresh
             * events on the first evaluation (spurious freeze + storm credit). */
            rscorr_dwell_reset(a, now, R);
            c->state         = PTV_CORR_ENGAGED;
            c->engage_r0     = R;
            c->engaged_corr0 = c->corr_us;
            c->engage_wc     = now;
            c->park_wc       = 0;
            c->slip_bad_wc   = 0;
            c->holdoff_wc    = 0;
            c->starve_wc     = 0;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-RSCORR] a%d(in%d) ENGAGE (starvation ceiling %"PRId64"min: R large+flat "
                   "the whole span, dwell never completed) R=%+"PRId64"ms → steering  "
                   "[+ = audio early]\n",
                   a->dbg_k, a->dbg_in, g_rscorr_ceil_us / 60000000, R / 1000);
        }
    } else if (c->state == PTV_CORR_ENGAGED || c->state == PTV_CORR_PARKED)
        c->starve_wc = 0;

    switch (c->state) {
    case PTV_CORR_ARMED:
        if (r->valid && !dead && llabs(R) > g_rscorr_engage_us) {
            c->state = PTV_CORR_DWELL;
            rscorr_dwell_reset(a, now, R);
            c->last_event_wc = 0;
        }
        break;

    case PTV_CORR_DWELL: {
        int64_t accum = now - c->dwell_start_wc;
        if (!r->valid || dead) {
            /* liveness/sensor loss is an event for the CONTINUOUS-across-dwell rule (§3);
             * fall back to ARMED — the dwell restarts from scratch when conditions return. */
            if (rscorr_storm_count(a, now, accum)) { rscorr_disarm(a, now, "event storm", 0); c->holdoff_wc = now + 600000000; break; }
            if (now - c->log_wc >= 10000000) {
                c->log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-RSCORR] a%d(in%d) HOLD (event: %s) dwell reset  [+ = audio early]\n",
                       a->dbg_k, a->dbg_in, dead ? dead : "sensor stale");
            }
            c->state = PTV_CORR_ARMED;
            break;
        }
        if (llabs(R) <= g_rscorr_engage_us) { c->state = PTV_CORR_ARMED; break; }   /* condition lapsed, silent */
        if (c->implaus_wc && now - c->implaus_wc >= 5000000) { rscorr_disarm(a, now, "R implausible (>5s)", 0); break; }
        ev = rscorr_event_edge(a);
        if (!ev) ev = rscorr_event_active(a, now);
        if (ev) {
            if (rscorr_storm_count(a, now, accum)) { rscorr_disarm(a, now, "event storm", 0); c->holdoff_wc = now + 600000000; break; }
            rscorr_dwell_reset(a, now, R);
            c->last_event_wc = now;
            if (now - c->log_wc >= 10000000) {
                c->log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-RSCORR] a%d(in%d) HOLD (event: %s) dwell reset  [+ = audio early]\n",
                       a->dbg_k, a->dbg_in, ev);
            }
            break;
        }
        /* §4.3 stability, enforced continuously: a decaying EMA transient moves >4x this
         * bound inside the window (STOP/CONT −1150→−15, wedge −810→−58) and keeps
         * re-anchoring the dwell — transients are CORRECT readings, never actuated. */
        if (llabs(R - c->dwell_r0) >= FFMAX(40000, llabs(R) / 4)) {
            c->dwell_start_wc = now;   /* silent re-anchor (not an event: no HOLD, no storm count) */
            c->dwell_r0 = R;
            break;
        }
        if (accum >= g_rscorr_dwell_us &&
            (!c->last_event_wc || now - c->last_event_wc >= g_rscorr_quiet_us)) {
            c->state         = PTV_CORR_ENGAGED;
            c->engage_r0     = R;
            c->engaged_corr0 = c->corr_us;
            c->engage_wc     = now;
            c->park_wc       = 0;
            c->slip_bad_wc   = 0;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-RSCORR] a%d(in%d) ENGAGE R=%+"PRId64"ms (dwell %"PRId64"s quiet) → steering  [+ = audio early]\n",
                   a->dbg_k, a->dbg_in, R / 1000, accum / 1000000);
        }
        break;
    }

    case PTV_CORR_ENGAGED: {
        int frozen = 0;
        if (!r->valid) { rscorr_disarm(a, now, "sensor stale", 0); break; }
        if (dead)      { rscorr_disarm(a, now, dead, 0); break; }
        if (c->implaus_wc && now - c->implaus_wc >= 5000000) { rscorr_disarm(a, now, "R implausible (>5s)", 0); break; }
        ev = rscorr_event_edge(a);
        if (ev) {
            /* §4 anti-windup (c): an event freezes integration the same frame; resume only
             * after 60s of re-quiet. Storm accounting applies (an engaged track being hit
             * by repeated events is the same oscillation risk). */
            c->last_event_wc = now;
            if (rscorr_storm_count(a, now, 60000000)) { rscorr_disarm(a, now, "event storm", 0); c->holdoff_wc = now + 600000000; break; }
            if (now - c->log_wc >= 10000000) {
                c->log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-RSCORR] a%d(in%d) HOLD (event: %s) steer frozen  [+ = audio early]\n",
                       a->dbg_k, a->dbg_in, ev);
            }
        } else if (rscorr_event_active(a, now))
            c->last_event_wc = now;
        if (c->last_event_wc && now - c->last_event_wc < 60000000)
            frozen = 1;
        /* §4 anti-windup (b): the resampler is behind (slip parked outside its dead band) —
         * pushing harder is windup by definition; >60s of it engaged = not realizing → disarm. */
        if (a->rs_slip_us != 0) {
            if (!c->slip_bad_wc) c->slip_bad_wc = now;
            else if (now - c->slip_bad_wc > 60000000) { rscorr_disarm(a, now, "resampler slip parked ≠0", 0); break; }
            frozen = 1;
        } else
            c->slip_bad_wc = 0;
        if (!frozen) {
            /* §4 steer law: proportional R/τ (τ=30s), slew-clamped, per emitted frame.
             * dR/dcorr = −1: +R (audio early) → +step → audio content delayed → R → 0.
             * pre29 #69 adaptive slew: above 150ms |R| the clamp rises to PTV_RSCORR_SLEW_FAST
             * (default 5ms/s = 0.5% rate, below the pitch JND — inaudible); back inside 150ms
             * the base clamp applies. Cuts a 300ms steer from ~2.5min to ~1min.
             * PTV_RSCORR_SLEW_FAST=0 disables (the base clamp everywhere, pre28 behavior). */
            int64_t slew = (g_rscorr_slew_fast > 0 && llabs(R) > 150000)
                         ? g_rscorr_slew_fast : g_rscorr_slew_us_s;
            int64_t rate = R / 30;                                   /* µs of trim per second */
            int64_t step;
            if (rate >  slew) rate =  slew;
            if (rate < -slew) rate = -slew;
            step = rate * f_us / 1000000;
            c->corr_us += step;
            if (llabs(c->corr_us - c->engaged_corr0) > 5000000) {    /* §6 per-engagement authority */
                c->corr_us = c->engaged_corr0 + (c->corr_us > c->engaged_corr0 ? 5000000 : -5000000);
                rscorr_disarm(a, now, "per-engagement authority cap (5s)", 1);
                break;
            }
            if (llabs(c->corr_us) > 10000000) {                      /* §6 lifetime authority */
                c->corr_us = c->corr_us > 0 ? 10000000 : -10000000;
                rscorr_disarm(a, now, "lifetime authority cap (10s)", 1);
                /* 1.0.1-pre21 #24 (owner-approved shape): a lifetime-cap disarm means 10
                 * whole seconds of trim did not close R — re-arming would only walk the
                 * same doomed staircase again (each round = another 5s per-engagement of
                 * content trim on air). Stay DISARMED for the life of the process. */
                c->perm_disarm = 1;
                av_log(NULL, AV_LOG_ERROR,
                       "[PTV-RSCORR] a%d(in%d) lifetime authority cap — corrector stays "
                       "DISARMED for this process (restart to reset)\n",
                       a->dbg_k, a->dbg_in);
                break;
            }
        }
        /* §4 convergence: |R| ≤ 20ms sustained 60s → PARK, corr_us RETAINED (it is an
         * accumulated content alignment; decaying it would re-open the offset). */
        if (llabs(R) <= 20000) {
            if (!c->park_wc) c->park_wc = now;
            else if (now - c->park_wc >= 60000000) {
                c->state = PTV_CORR_PARKED;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-RSCORR] a%d(in%d) PARK R=%+"PRId64"→%+"PRId64"ms corr=%+"PRId64"ms in %"PRId64"s  [+ = audio early]\n",
                       a->dbg_k, a->dbg_in, c->engage_r0 / 1000, R / 1000,
                       (c->corr_us - c->engaged_corr0) / 1000, (now - c->engage_wc) / 1000000);
                break;
            }
        } else
            c->park_wc = 0;
        if (g_diag && now - c->diag_wc >= 30000000) {
            c->diag_wc = now;
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-RSCORR] a%d(in%d) steering R=%+"PRId64"ms corr=%+"PRId64"ms slip=%+"PRId64"ms%s  [+ = audio early]\n",
                   a->dbg_k, a->dbg_in, R / 1000, c->corr_us / 1000, a->rs_slip_us / 1000,
                   frozen ? " (frozen)" : "");
        }
        break;
    }

    case PTV_CORR_PARKED:
        /* re-engagement requires a FRESH full dwell (§4) */
        if (r->valid && !dead && llabs(R) > g_rscorr_engage_us) {
            c->state = PTV_CORR_DWELL;
            rscorr_dwell_reset(a, now, R);
            c->last_event_wc = 0;
        }
        break;

    case PTV_CORR_DISARMED:
        if (c->perm_disarm)         /* #24: lifetime-cap disarm is final (logged once above) */
            break;
        if (c->holdoff_wc && now < c->holdoff_wc)
            break;
        if (c->implaus_wc)          /* R still implausible = the disarm condition persists: */
            break;                  /* stay silently disarmed (no re-arm flap) */
        if (r->valid && !dead) {
            c->state = PTV_CORR_ARMED;
            c->holdoff_wc = 0;
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-RSCORR] a%d(in%d) re-armed (recovered; fresh full dwell required)  [+ = audio early]\n",
                   a->dbg_k, a->dbg_in);
        }
        break;
    }

    atomic_store_explicit(&g_corr_pub[a->dbg_k], c->corr_us, memory_order_relaxed);
    atomic_store_explicit(&g_corr_state_pub[a->dbg_k], c->state, memory_order_relaxed);
}

/* one deliberate door-label step every observing engine treats as expected (the [PTV-ANCHOR] algebra: glue_off
 * += step retires exactly +step) — the T-395 relabel undo */
static void ptv_label_step(AudioState *a, int64_t step, int64_t now)
{
    a->glue_off_us  += step;            /* the [PTV-ANCHOR] algebra: retires exactly +step */
    a->rs_ma_seed    = 0;               /* base redefined — sensor re-seeds */
    a->pend_comp_us += step;            /* §2.4 tripwire composes with any pending verdict */
    a->pend_comp_wc  = now;
    if (step < 0) {                     /* backward door labels — arm the mux monotonic guard */
        a->reanch_mono = 1;
        a->reanch_mono_dropped = 0;
    }
    a->glue_events++;                   /* corrector event edge: it re-dwells fresh */
    a->acomp_exp_us  = AV_NOPTS_VALUE;  /* deliberate step, not a click risk */
}
/* ======================================================================================= */

/* -af path: drain the filtergraph's (fixed-size) output frames straight to the
 * per-rung encoders, stamping each with ITS OWN filter PTS rebased onto the house
 * anchor h0 (in out_rate sample units). This HONORS aresample=async's correction
 * — discarding it for a free sample counter is what let audio drift behind video.
 * Frames whose rebased pts precedes the video anchor (<0) are dropped. */
static int audio_drain_fg(AudioState *a)
{
    AVFrame *filt = av_frame_alloc();
    AVRational sink_tb;
    int64_t h0, h0_samp;
    int ret = 0;
    if (!filt) return AVERROR(ENOMEM);
    pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
    h0_samp = (h0 == AV_NOPTS_VALUE) ? 0 : av_rescale(h0, a->out_rate, 1000000);
    sink_tb = av_buffersink_get_time_base(a->afsink);
    while ((ret = av_buffersink_get_frame(a->afsink, filt)) >= 0) {
        if (filt->pts != AV_NOPTS_VALUE && a->afmt_draining && a->afmt_drain_end_us != AV_NOPTS_VALUE) {
            /* 2.0.0-pre9.6: a flushed buffering filter (loudnorm's 100 ms blocks) pads its last block past the real
             * content end — cut the drain there, the rebuilt path continues from that point */
            int64_t st_us = av_rescale_q(filt->pts, sink_tb, AV_TIME_BASE_Q);
            int64_t keep  = av_rescale(a->afmt_drain_end_us - st_us, filt->sample_rate > 0 ? filt->sample_rate : 48000, 1000000);
            if (keep <= 0) { av_frame_unref(filt); continue; }
            if (keep < filt->nb_samples) filt->nb_samples = (int)keep;
        }
        if (filt->pts != AV_NOPTS_VALUE) {
            int64_t src_abs_us = av_rescale_q(filt->pts, sink_tb, AV_TIME_BASE_Q);  /* A/V probe: this frame's (post-async) source content time (us), before pts is rebased */
            if (a->dbg_k == 0)   /* [PTV-CHAIN] primary-audio source-content being emitted (us) */
                atomic_store_explicit(&g_ch_aout_src, src_abs_us, memory_order_relaxed);
            if (g_diag) {   /* [PTV-ASTEP]/[PTV-AFLOW]: post-graph step detector + content-flow ledger */
                if (a->dbg_sink_us && llabs(src_abs_us - (a->dbg_sink_us + a->dbg_sink_dur_us)) > 5000) {
                    char sn[48];
                    av_log(NULL, AV_LOG_WARNING, "[PTV-ASTEP] sink-pts step %+lldms (sink=%lldus expect=%lldus)%s\n",
                           (long long)((src_abs_us - a->dbg_sink_us - a->dbg_sink_dur_us) / 1000),
                           (long long)src_abs_us, (long long)(a->dbg_sink_us + a->dbg_sink_dur_us),
                           ptv_self_shed_note(a, sn, sizeof sn));
                }
                a->dbg_sink_us = src_abs_us;
                a->dbg_sink_dur_us = av_rescale(filt->nb_samples, 1000000, a->out_rate);
                a->dbg_out_samp += filt->nb_samples;
                int64_t nowf = av_gettime_relative();
                if (nowf - a->dbg_flow_last_us >= FFMAX(5000000, g_stats_period_us)) {   /* 2.0.0: per stats period */
                    a->dbg_flow_last_us = nowf;
                    av_log(NULL, AV_LOG_INFO,
                           "[PTV-AFLOW] a%d in_samp=%lld out_samp=%lld imbalance=%+lldms in_pts=%lldms sink_pts=%lldms\n",
                           a->dbg_k, (long long)a->dbg_in_samp, (long long)a->dbg_out_samp,
                           (long long)((a->dbg_in_samp - a->dbg_out_samp) * 1000 / (a->out_rate > 0 ? a->out_rate : 48000)),
                           (long long)(a->dbg_in_us / 1000), (long long)(src_abs_us / 1000));
                }
            }
            /* Content-anchored output PTS = source content − h0. This rides the source clock and is
             * discontinuity-absorbed by demux_unwrap (so it survives real splices). */
            int64_t opts = av_rescale_q(filt->pts, sink_tb, (AVRational){1, a->out_rate}) - h0_samp;
            if (opts < 0) { av_frame_unref(filt); continue; }   /* precedes video anchor */
            filt->pts = opts;
            /* AUDIO-FOLLOW (Option A, multiview only): apply the compositor's latched per-slot
             * offset as a ONE-TIME deterministic correction — emit on a CONTINUOUS output counter
             * (gapless, monotonic), DROPPING content when the audio is behind the video (advance)
             * or PADDING silence when ahead (delay). Single-input / PTV_NO_AUDIO_FOLLOW keep the
             * content-stamped opts path untouched. */
            if (a->multiview && a->tl_lag && a->tl_at) {
                /* 2.1 T-409 (design §5.4a): the slot is a frame synchronizer — its picture shows source time S at
                 * S - h0 + L (mv_tl_pick). The sound goes to the same place, by construction: no measurement, no
                 * PLL. Emitted gapless; the error to that place (the labels + L; the corrector's walk is in the
                 * labels) is taken exactly, to the whole frame, once it reaches half a frame (silence padded / late
                 * frames dropped, like a frame sync's audio); the residue stays within half a frame. L is the picture's
                 * REALIZED lag (tick rounding and decimation included), published by the compositor */
                int     nb = filt->nb_samples;
                /* a lag step applies from its content boundary on (mv_tl_step): the same content as the picture's */
                int64_t at = atomic_load_explicit(a->tl_at, memory_order_acquire);
                int64_t L  = src_abs_us >= at ? atomic_load_explicit(a->tl_nlag, memory_order_relaxed)
                                              : atomic_load_explicit(a->tl_lag, memory_order_relaxed);
                int64_t exact = opts + av_rescale(L, a->out_rate, 1000000), cont, err;
                /* T-411/T-414: an overflow step — the picture skipped from its old due line to the boundary (those
                 * pictures are gone); this sound, not placed yet, goes with them (placed, it played Δ late for that
                 * long, then the sound skipped what the picture showed) */
                if (src_abs_us < at && src_abs_us >= atomic_load_explicit(a->tl_skip, memory_order_relaxed)) {
                    a->rs_ma_seed = 0;
                    av_frame_unref(filt);
                    continue;
                }
                if (!a->af_out_set) {
                    if (exact < 0) { av_frame_unref(filt); continue; }
                    a->af_last_out = exact - nb;
                    a->af_out_set  = 1;
                }
                cont = a->af_last_out + nb;
                err  = exact - cont;                           /* samples: + = the sound belongs later */
                if (err <= -nb / 2) {
                    a->rs_ma_seed = 0;                         /* the sensor's audio side takes the step with it */
                    av_frame_unref(filt);                      /* late: dropped (the next frames meet the place) */
                    continue;
                }
                if (err >= nb / 2) {
                    int64_t n = (err + nb / 2) / nb;
                    a->rs_ma_seed = 0;
                    while (n-- > 0) {                          /* a hole: silence where the sound is missing */
                        AVFrame *s = av_frame_alloc();
                        if (s) {
                            s->nb_samples = nb; s->format = filt->format; s->sample_rate = a->out_rate;
                            av_channel_layout_copy(&s->ch_layout, &filt->ch_layout);
                            if (av_frame_get_buffer(s, 0) >= 0) {
                                av_samples_set_silence(s->data, 0, nb, s->ch_layout.nb_channels, s->format);
                                s->pts = a->af_last_out + nb; a->af_last_out = s->pts;
                                audio_encode_push(a, s); a->out_frames++;
                            }
                            av_frame_free(&s);
                        }
                    }
                    cont = a->af_last_out + nb;
                }
                a->af_last_out = cont;
                filt->pts = cont;
                {                                              /* the newest content placed (the next step's boundary
                                                                * goes past it), the slot's tracks' maximum */
                    int64_t hd = src_abs_us + (int64_t)nb * 1000000 / a->out_rate;
                    int64_t cur = atomic_load_explicit(a->tl_ahead, memory_order_relaxed);
                    while (hd > cur && !atomic_compare_exchange_weak_explicit(a->tl_ahead, &cur, hd,
                                                                             memory_order_relaxed, memory_order_relaxed))
                        ;
                }
            }
            /* pre20 fix round (F1): MONOTONIC EMISSION GUARD. A re-anchor that retired a
             * positive carried offset (or a negative (c) step) moved the door labels BACKWARD;
             * until the source's own label advance covers the retired amount, every drained
             * frame's output label precedes the last emitted one — emitting it is a backward
             * DTS into the muxer = dead wire (reviewer blocker, 3 sites). DROP those frames
             * (the birth opts<0 rule, mirrored); release on the first label that EXCEEDS the
             * last emitted (bounded: the retired amount itself, in content ms, inside the
             * already-audible rebuild). Non-follow path only — the mv follow path's af_last_out
             * clamp already forbids backward emission. Dropped frames do not reach the sensor
             * or the (c) settle counter (they were never emitted). */
            if (a->reanch_mono && !(a->multiview && g_audio_follow)) {
                if (a->out_pts_set && filt->pts <= a->out_last_pts) {
                    a->reanch_mono_dropped++;
                    av_frame_unref(filt);
                    continue;
                }
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-ANCHOR] a%d(in%d) monotonic guard released — %"PRId64" frames "
                       "(~%"PRId64"ms) dropped while the re-based labels caught up to the last "
                       "emitted DTS (the retired offset, realized as a content skip)\n",
                       a->dbg_k, a->dbg_in, a->reanch_mono_dropped,
                       a->out_rate > 0 && filt->nb_samples > 0
                           ? a->reanch_mono_dropped * av_rescale(filt->nb_samples, 1000, a->out_rate)
                           : 0);
                a->reanch_mono = 0;
            }
            /* ====================================================================================
             * 1.0.1-pre9 residual sensor (PASSIVE), audio side. 1.0.1-pre16: live on MULTIVIEW
             * too (the mv gate is gone) — the compositor publishes per-slot video lineage, so
             * every (slot,track) gets the same certified sensor. filt->pts here is the emitted
             * output label: the content-anchored opts on single input, the re-stamped `want` on
             * the mv follow path — so out carries the B3 acquire drops/pads (af_applied_us),
             * and inj mirrors the PATH-DEPENDENT graph-door bus (below), leaving R the RESIDUAL.
             *   m_a = out − (sink_src − inj) − slip
             * inj recovers the RAW post-demux label from the sink label by removing what this
             * thread itself injected at the graph door: AGLUE's cumulative relabel-erase
             * (glue_off — a relabel-erased REAL step surfaces here as persistent R, the trap-1
             * requirement) + AVLOCK's house_skew (whose REALIZED retiming is matched by the
             * video side's measured dup ratchet → cancels in R = shared latency, not desync —
             * the wedge/Lindel class where span accounting read −2986ms on a +24ms channel).
             * slip = the resampler's UN-REALIZED correction, scoped (1.0.1-pre11) to the async
             * aresample FILTER's own boundary: (label head at its input link) − (label head at
             * its output link) − (swr sample backlog). ≈0 when aresample=async has converged
             * content onto its labels; parks at the un-taken correction when hard-comp wedges
             * (the [PTV-SWRDELAY] class label math alone is blind to). The pre9 probe spanned
             * the WHOLE graph (door acomp_exp_us − sink head), so any buffering -af filter
             * (loudnorm ~3s analysis window) parked its hold in slip = a constant false
             * audio-early bias (+2914ms fixture/live class, Defect 1); a passive filter's hold
             * preserves labels — it is shared latency, not desync — and must not appear here.
             * 50ms dead band swallows the structural residue (frame-start vs head labels, rate
             * rounding) so steady state is exactly label-based. Pads/drops that fill GENUINE
             * label gaps never appear in any term — label-referenced mapping is edit-neutral
             * for them by construction (trap 2).
             * ==================================================================================== */
            if (g_rsync_sense && a->use_fg &&
                a->dbg_k < PTV_MAX_AUDIO && a->out_rate > 0) {
                int64_t out_us = av_rescale(filt->pts, 1000000, a->out_rate);
                int64_t hs     = (!(a->multiview && g_audio_follow) && g_avlock && a->house_skew)
                               ? *a->house_skew : 0;
                /* pre14 (§5 bus rule 3): corr_us joins inj — the sensor recovers the RAW
                 * post-demux label by removing the FULL bus sum this thread injected at the
                 * graph door, so R moves label-deterministically with the trim (the AVLOCK
                 * accounting pattern; NOT self-blinding — the corrector is supposed to see
                 * its own correction land, and the external oracle stays the ground truth).
                 * pre16: inj mirrors the PATH-DEPENDENT door bus exactly (the AVSYNC2
                 * subtraction, §5 bus rule 2 site): single/non-follow injects house_skew
                 * (AVLOCK, audio_feed) — the mv follow path injects af_steer_us instead
                 * (pre3 steer); glue_off + corr are path-independent. Sign law preserved on
                 * both paths: +corr at the door delays audio content ⇒ raises m_a ⇒
                 * dR/dcorr = −1 (§2). */
                int64_t inj    = a->glue_off_us + a->corr.corr_us + hs;
                int64_t slip   = 0, m, nowr;
                if (a->fg_swr && a->fg_swr_flt &&
                    a->fg_swr_flt->nb_inputs > 0 && a->fg_swr_flt->nb_outputs > 0) {
                    /* pre11: label HEADS at the aresample filter's OWN links (updated by the
                     * generic consume path on every frame crossing them) — filters before or
                     * after it (loudnorm etc.) hold content with labels intact, outside this
                     * boundary, so their hold can no longer read as slip. current_pts_us is
                     * the START of the last chunk CONSUMED off the link; reconstruct true
                     * heads symmetrically (fixture-measured, both matter vs the 50ms band):
                     *  in  head = start + avg consumed-chunk duration (counters are exact for
                     *             steady chunking: mp2 1152; loudnorm's 100ms consume quantum)
                     *  out head = start + avg chunk + QUEUED duration (produced-but-unconsumed
                     *             frames sit in the link fifo with labels dense above the
                     *             consumed head — sample_count_in − sample_count_out) */
                    const FilterLink *fli = ff_filter_link(a->fg_swr_flt->inputs[0]);
                    const FilterLink *flo = ff_filter_link(a->fg_swr_flt->outputs[0]);
                    int irate = a->fg_swr_flt->inputs[0]->sample_rate;
                    int orate = a->fg_swr_flt->outputs[0]->sample_rate;
                    if (fli->current_pts_us != AV_NOPTS_VALUE &&
                        flo->current_pts_us != AV_NOPTS_VALUE &&
                        irate > 0 && orate > 0 &&
                        fli->frame_count_out > 0 && flo->frame_count_out > 0) {
                        int64_t ihead = fli->current_pts_us
                                      + av_rescale(fli->sample_count_out / fli->frame_count_out,
                                                   1000000, irate);
                        int64_t ohead = flo->current_pts_us
                                      + av_rescale(flo->sample_count_out / flo->frame_count_out,
                                                   1000000, orate)
                                      + av_rescale(flo->sample_count_in - flo->sample_count_out,
                                                   1000000, orate);
                        slip = ihead - ohead - swr_get_delay(a->fg_swr, 1000000);
                        if (slip > -50000 && slip < 50000) slip = 0;      /* dead band */
                        else slip += slip > 0 ? -50000 : 50000;
                    }
                }
                a->rs_slip_us = slip;
                /* 1.0.1-pre15 §2.4 realization tripwire (#33): a GAP/FLUSH-APPLY verdict armed
                 * pend_comp_us; hard comp (min_hard_comp=0.03 in the prod chain) realizes it
                 * INSTANTLY, so a slip still parked near the verdict size 2s later means the
                 * resampler did NOT take the pad/drop — synthesize the parked remainder at the
                 * swr boundary ourselves (the resampler's own compensation primitives, not a
                 * second actuator) instead of shipping on faith. Expected NEVER to fire on
                 * forward steps (PATRIOT-proven); it is the G7 witness for backward >1s drops. */
                if (g_glueclass && a->pend_comp_us) {
                    if (llabs(slip) <= llabs(a->pend_comp_us) / 10) {
                        a->pend_comp_us = 0;                       /* realized */
                    } else if (av_gettime_relative() - a->pend_comp_wc > 2000000) {
                        /* pre16 #47-B (Fashion 2026-07-18): AUTHORITY CLAMP. The tripwire had
                         * none — a +11594s verdict parked slip at +8798s and the synthesis
                         * injected ~2800s at the swr, pinning the channel at async for hours.
                         * A parked slip beyond PTV_GLUE_TW_CAP_US (2s — the tripwire deadline
                         * itself; every legitimate hard-comp verdict realizes instantly and
                         * the largest real one ever seen is PATRIOT's 30.8s, which realized)
                         * is evidence of upstream chaos, NOT something to actuate: NO
                         * synthesis, one ERROR line, corrector freeze, verdict retired. */
                        if (llabs(slip) > PTV_GLUE_TW_CAP_US) {
                            a->glue_events++;                      /* corrector freeze (§5) */
                            av_log(NULL, AV_LOG_ERROR,
                                   "[PTV-AGLUE] a%d(in%d) verdict %+"PRId64"ms NOT realized (slip parked "
                                   "%+"PRId64"ms) — BEYOND the %ds tripwire authority: refusing to "
                                   "synthesize (upstream label chaos; residual to sensor/corrector)\n",
                                   a->dbg_k, a->dbg_in, a->pend_comp_us / 1000, slip / 1000,
                                   (int)(PTV_GLUE_TW_CAP_US / AV_TIME_BASE));
                            a->pend_comp_us = 0;
                            /* 1.0.1-pre17 fix round (R3c, review-recommended): AGLUE PLAUSIBILITY
                             * CEILING. A step the tripwire just REFUSED is one the resampler is
                             * still PURSUING — aresample=async grinds at its maximum rate against
                             * a label step that has no content reality (Fashion live: the +11594s
                             * pursuit pinned the channel at async −73Mppm for hours after the
                             * one-sided release). Stop the chase: RELABEL-erase the parked
                             * remainder into glue_off (the butt-joint semantic — content
                             * continues, the door labels rejoin the output timeline), one loud
                             * line. glue_off is part of the sensor's inj, so R accounting stays
                             * label-deterministic (identical to an AGLUE relabel verdict); the
                             * erased residual is exactly what the sensor/corrector own from here.
                             * PTV_NO_AGLUE_CEIL=1 reverts to the pursue-forever posture. */
                            if (g_aglue_ceil) {
                                a->glue_off_us -= slip;
                                av_log(NULL, AV_LOG_ERROR,
                                       "[PTV-AGLUE] a%d(in%d) plausibility ceiling: erased the refused "
                                       "%+"PRId64"ms label pursuit at the graph door (glue_off now %+"PRId64"ms) "
                                       "— channel stays watchable; residual owned by sensor/corrector "
                                       "(PTV_NO_AGLUE_CEIL reverts)\n",
                                       a->dbg_k, a->dbg_in, slip / 1000, a->glue_off_us / 1000);
                            }
                        } else {
                        int irate = a->fg_swr_flt ? a->fg_swr_flt->inputs[0]->sample_rate : 0;
                        int orate = a->fg_swr_flt ? a->fg_swr_flt->outputs[0]->sample_rate : 0;
                        if (a->fg_swr && slip > 0 && irate > 0)
                            swr_inject_silence(a->fg_swr, (int)av_rescale(slip, irate, 1000000));
                        else if (a->fg_swr && slip < 0 && orate > 0)
                            swr_drop_output(a->fg_swr, (int)av_rescale(-slip, orate, 1000000));
                        a->tw_synth_cnt++;
                        a->glue_events++;                          /* corrector freeze (§5 rule 4) */
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) verdict %+"PRId64"ms NOT realized by the resampler "
                               "within 2s (slip parked %+"PRId64"ms) — synthesized at the swr boundary (#%"PRId64")\n",
                               a->dbg_k, a->dbg_in, a->pend_comp_us / 1000, slip / 1000,
                               a->tw_synth_cnt);
                        a->pend_comp_us = 0;
                        }
                    }
                }
                m = out_us - (src_abs_us - inj) - slip;
                /* 2.0.0-pre9.6 (T-075): the EMA smooths only the MEASURED part of m_a (slip, rounding);
                 * the label edits this thread made (glue_off relabels/folds, the door's house_skew) are
                 * steps by nature and join outside it, as the demux ledger E does. Smoothed, a hold's
                 * house_skew step or a fold slewed for minutes while the video side and E had moved at
                 * once: lipsync= −1.8 s on sync_flapping, +1.6 s on sync_gap_20, −12.7 s on the canary
                 * NTD drop, while the instantaneous R read the ruler's +192 ms / 0 ms exactly. corr_us
                 * stays inside (the corrector's realized-trim feedback, §4). */
                {
                    int64_t step_us = a->glue_off_us + hs;
                    if (!a->rs_ma_seed) { a->rs_ma_res = m - step_us; a->rs_ma_seed = 1; }
                    else {
                        int64_t dv = a->frame_size > 0                    /* EMA ≈ 30s of audio frames */
                                   ? 30LL * a->out_rate / a->frame_size : 1406;
                        if (dv < 8) dv = 8;
                        a->rs_ma_res += (m - step_us - a->rs_ma_res) / dv;
                    }
                    a->rs_ma_ema = a->rs_ma_res + step_us - a->rl_neutral_us;   /* T-087: relabels judged by content */
                }
                atomic_store_explicit(&g_rsx.ma_ema[a->dbg_k], a->rs_ma_ema, memory_order_relaxed);
                nowr = av_gettime_relative();
                atomic_store_explicit(&g_rsx.ma_wall[a->dbg_k], nowr, memory_order_relaxed);
                /* pre14 corrector: one evaluation per emitted frame, immediately after this
                 * frame's m_a publish (R is current, and rs_slip_us above is this frame's). */
                {
                    RsyncTrackR rr = rsync_track_R(a, nowr);
                    rscorr_update(a, &rr,
                                  av_rescale(filt->nb_samples, 1000000, a->out_rate), nowr);
                    /* 1.0.1-pre20 (c) rebuild residual step (see ptv_rebuild_reanchor): inside
                     * the 10s rebuild discontinuity window ONLY, after ≥10 emitted frames have
                     * re-seeded the sensor, a residual still outside the engage band is applied
                     * as ONE immediate base step (cap ±5s) — the rebuild is already an audible
                     * discontinuity, so the step is free; the corrector trims leftovers from a
                     * fresh dwell (glue_events edge below resets it). dR/d(glue_off) = −1 (the
                     * same door algebra as corr), so `glue_off += R` retires exactly +R. */
                    /* (c) runs on the single-input / non-follow path only: on the mv
                     * audio-follow path the closed-loop PLL owns content alignment (it
                     * measures out_v−out_a directly and would re-track a base step within
                     * an acquire — two actuators on one displacement); there the pre17
                     * ACQUIRE + (b)'s re-seed already restore honesty within ~12s. */
                    if (g_rebuild_reanchor && a->reanch_wc &&
                        (a->multiview && g_audio_follow)) {
                        a->reanch_wc = 0;                         /* mv follow: PLL owns it (a format switch's splice
                                                                   * edit is returned exactly at the switch: T-352) */
                    } else if (g_rebuild_reanchor && a->reanch_wc) {
                        if (nowr - a->reanch_wc > PTV_REANCH_WIN_US) {
                            a->reanch_wc = 0;                     /* window closed — never fires late */
                        } else if (llabs(a->reanch_step_us) < PTV_REANCH_CAP_US &&
                                   ++a->reanch_frames >= PTV_REANCH_MIN_FR &&
                                   rr.valid && llabs(rr.R_us) > PTV_REANCH_BAND_US) {
                            /* fix round (F1 follow-up): MULTI-STEP inside the one window — a seam's
                             * video-side ledger edit can land AFTER the first step (measured: the
                             * one-shot fired 4ms post-rebuild, the video [PTV-DISCONT] absorb landed
                             * ~500ms later and re-opened R by the ledger delta); each fire re-requires
                             * a fresh ≥10-emitted-frame settle, total authority ±5s across the window,
                             * all inside the same audible rebuild. */
                            int64_t stepR = rr.R_us;
                            int64_t left  = PTV_REANCH_CAP_US - llabs(a->reanch_step_us);
                            if (stepR >  left) stepR =  left;
                            if (stepR < -left) stepR = -left;
                            a->glue_off_us   += stepR;
                            a->rs_ma_seed     = 0;                /* base redefined again — re-seed */
                            a->pend_comp_us  += stepR;            /* §2.4 tripwire: COMPOSE with any pending
                                                                   * verdict (F1 minor — was clobbered) */
                            a->pend_comp_wc   = nowr;
                            if (stepR < 0) {                      /* F1: a backward step moves the door
                                                                   * labels BACK — arm the monotonic
                                                                   * emission guard so the mux can never
                                                                   * see a backward DTS (absolute) */
                                a->reanch_mono = 1;
                                a->reanch_mono_dropped = 0;
                            }
                            a->glue_events++;                     /* corrector event edge: fresh dwell */
                            a->acomp_exp_us   = AV_NOPTS_VALUE;   /* deliberate step, not a click risk */
                            a->reanch_step_us += stepR;           /* window budget (±5s total) */
                            a->reanch_frames  = 0;                /* next fire needs a fresh settle */
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-ANCHOR] a%d(in%d) rebuild residual %+"PRId64"ms stepped out at "
                                   "the base (%"PRId64"ms after rebuild; window total %+"PRId64"ms, "
                                   "budget %ds) — corrector re-dwells on the remainder\n",
                                   a->dbg_k, a->dbg_in, stepR / 1000,
                                   (nowr - a->reanch_wc) / 1000, a->reanch_step_us / 1000,
                                   (int)(PTV_REANCH_CAP_US / 1000000));
                        }
                    }
                }
                /* pre24 #63 cross-stream conservation diag (measurement only, no control):
                 * the running A−V difference of UNEVIDENCED real-content deletions IS the
                 * predicted permanent on-air offset pressure — say so when it moves. */
                {
                    int64_t cua = atomic_load_explicit(&g_rsx.uea_us[a->dbg_k], memory_order_relaxed)
                                + a->u_door_us;
                    int64_t cuv = atomic_load_explicit(&g_rsx.uev_us[a->dbg_in], memory_order_relaxed);
                    int64_t imb = cua - cuv;
                    if (llabs(imb - a->wev_cons_last_us) > 500000 &&
                        (a->wev_cons_log_wc == 0 || nowr - a->wev_cons_log_wc >= 10000000)) {
                        a->wev_cons_last_us = imb;
                        a->wev_cons_log_wc  = nowr;
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-WALLEV] a%d(in%d) unevidenced real-content deletion imbalance "
                               "A−V = %+"PRId64"ms (A=%"PRId64"ms V=%"PRId64"ms) — permanent-offset "
                               "pressure (measurement only)  [+ = audio early]\n",
                               a->dbg_k, a->dbg_in, imb / 1000, cua / 1000, cuv / 1000);
                    }
                }
                if (g_diag && (a->rs_log_last == 0 ||
                               nowr - a->rs_log_last >= g_stats_period_us)) {
                    /* [PTV-RSYNC] components. dm = m_v − m_a (the shared −h0 cancels); the raw
                     * EMAs are h0-offset and unreadable alone. R = dm + E_v − E_a. Not printed
                     * until the video side has published (dm would be raw −h0-scale garbage). */
                    int64_t mvw = atomic_load_explicit(&g_rsx.mv_wall[a->dbg_in], memory_order_relaxed);
                    if (mvw) {
                        int64_t mv = atomic_load_explicit(&g_rsx.mv_ema[a->dbg_in], memory_order_relaxed);
                        int64_t ev = atomic_load_explicit(&g_rsx.ev_us[a->dbg_in],  memory_order_relaxed);
                        int64_t ea = atomic_load_explicit(&g_rsx.ea_us[a->dbg_k], memory_order_relaxed);
                        int64_t dm = mv - a->rs_ma_ema;
                        a->rs_log_last = nowr;
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-RSYNC] a%d(in%d) R=%+"PRId64"ms%s dm=%+"PRId64"ms ev=%+"PRId64"ms ea=%+"PRId64"ms "
                            "glue=%+"PRId64"ms hs=%+"PRId64"ms slip=%+"PRId64"ms  [+ = audio early; passive]\n",
                            a->dbg_k, a->dbg_in, (dm + ev - ea) / 1000,
                            (nowr - mvw < 3000000) ? "" : "(video-stale)",
                            dm / 1000, ev / 1000, ea / 1000,
                            a->glue_off_us / 1000, hs / 1000, slip / 1000);
                    }
                }
            }
            /* ====================================================================================
             * A/V offset sensor (A/V PLL redesign Phase A, analysis/ptvencoder-avsync-pll-redesign-plan.md
             *   §3; its verbose [PTV-AVSYNC2] print was removed in 2.0.0). Measures the REAL per-track
             *   lip-sync offset, NOT a proxy: for the source content C this emitted audio frame
             *   carries, look up the output time the VIDEO showed that SAME content (the per-input
             *   ring, written by the compositor / single-input output thread) and compare:
             *       offset = out_v(C) − out_a(C)     (− = picture ahead of audio = video leads)
             *   with the video_lag / audio_lag split (§3.2a — which side moved) and the content
             *   pairing residual (§3.2b). out_a is the ACTUAL emitted pts (the af counter+nudge in
             *   multiview, opts in single-input), so it is faithful where async_pad/house_skew were
             *   confounded. No actuator — this only reports. M-b cross-check = "offset ≈ 0 on a clean
             *   synced source" (validated on the local clean run), not a separate metric (adjacent-DTS
             *   at the mux reads ≈0 regardless = the av_off trap).
             * ==================================================================================== */
            if (a->vring) {
                int64_t out_a_us = av_rescale(filt->pts, 1000000, a->out_rate);   /* emitted output time (us) */
                int64_t h0_us    = (h0 == AV_NOPTS_VALUE) ? 0 : h0;
                int64_t content  = src_abs_us;                                    /* abs source content of this audio frame */
                int64_t out_v, msrc;
                /* single-input injects house_skew into the graph INPUT → the buffersink pts carries it;
                 * remove it to recover the true source content for the video pairing. Multiview
                 * audio-follow (pre3) injects af_steer_us the same way — remove that too, so the
                 * measurement reads the TRUE post-steer offset instead of pairing against a content
                 * value already shifted by the actuator.
                 * LOOP SIGN (pre3 steer): out_a(true content C) = (C + steer_realized) − h0 + applied,
                 * so alag = steer_realized + applied and offset = vlag − alag ⇒ d(offset)/d(steer) = −1
                 * (same convention as the old label `applied`: steer += (+offset·frame/τ) drives the
                 * measured offset → 0). During a steer transient the sink labels lag the commanded
                 * value by the pending (unrealized) part — bounded by the 10ms/s rate clamp vs the
                 * resampler's 20.8ms/s authority, so the pairing error decays within a second. */
                /* pre14 (§5 bus rule 2): corr_us joins the actuator-term subtraction in every
                 * measurement loop that reads sink labels (glue_off stays un-subtracted, as
                 * today: glue is the label-truth judgment, not an actuator). */
                if (!(a->multiview && g_audio_follow) && a->house_skew)
                    content -= *a->house_skew + a->corr.corr_us;
                else if (a->multiview && g_audio_follow)
                    content -= a->corr.corr_us;
                if (!(a->multiview && a->mvf_hi_us > a->mvf_lo_us &&             /* T-093: fill phase frames carry no */
                      src_abs_us >= a->mvf_lo_us && src_abs_us < a->mvf_hi_us) &&  /* content to pair */
                    vring_lookup(a->vring, content, &out_v, &msrc) == 0) {
                    int64_t vlag   = out_v    - (msrc    - h0_us);   /* video realized output − content (at msrc) */
                    int64_t alag   = out_a_us - (content - h0_us);   /* audio realized output − content (at content) */
                    int64_t offset = vlag - alag;                    /* the content-referenced (via shared h0) lip-sync the
                                                                       * PLAYER sees — not the raw out_v−out_a, which carries
                                                                       * the pairing residual msrc−content (deep video prime →
                                                                       * composition lags the audio drain in WALL time). */
                    /* Latch the latest measurement for the [PTV-AVSYNC] line (PTV_DIAG-only since v0.9.2). */
                    a->av_offset_us = offset; a->av_vlag_us = vlag; a->av_alag_us = alag; a->av_off_valid = 1;
                }
            }
            /* v0.9.2 — aresample WORK RATE (always-on, primary track only). The honest measure is the
             * RATE of change of the audio's realized output-vs-content span: d(outspan − content)/d(wall)
             * in ppm — NOT a raw in/out sample ratio (which reads the nominal 44.1k→48k conversion as a
             * huge constant). A rate also washes out the slowly-varying house_skew DC term. + = adding
             * samples (stretch/pad), − = dropping/compressing; ~0 = idle. Latched for the progress line. */
            if (g_stats && a->dbg_k == 0) {
                int64_t nowa = av_gettime_relative();
                int64_t bal  = (a->out_frames * (int64_t)a->frame_size * 1000000 / a->out_rate)
                             - av_rescale_q(a->dbg_last_src - a->dbg_first_src, a->ist_tb, AV_TIME_BASE_Q);
                if (a->async_stat_last == 0) { a->async_stat_last = nowa; a->async_prev_bal = bal; }
                else if (nowa - a->async_stat_last >= g_stats_period_us) {
                    int64_t dw = nowa - a->async_stat_last;
                    int64_t r  = dw > 0 ? (bal - a->async_prev_bal) * 1000000 / dw : 0;
                    int64_t cur = atomic_load_explicit(&g_async_ppm, memory_order_relaxed);
                    /* EMA (÷8): the per-interval balance is quantized to ~one audio frame (~21ms ⇒
                     * ~1000ppm noise @10s), so smooth to the NET rate — idle ≈ 0, a sustained sign = real work. */
                    atomic_store_explicit(&g_async_ppm, cur + ((r - cur) >> 3), memory_order_relaxed);
                    a->async_stat_last = nowa; a->async_prev_bal = bal;
                }
            }
            /* [PTV-AVSYNC] / [PTV-SWRDELAY] / [PTV-CHAIN] — internal A/V CONTROLLER telemetry. These are
             * control-domain ESTIMATES (offset / house_skew / outA-V) that diverge from the wire (they
             * read +11.7s while the wire was ±80ms), so as of v0.9.2 they are DEBUG-only (PTV_DIAG).
             * Reports what the per-slot actuator is correcting (lag), how much it applied, the residual
             * (err), and acquire drop/pad. Multiview audio-follow only; absolute lip-sync is NOT self-
             * reported — it is measured externally by the wire oracle (drift-continuous.py). */
            if (g_diag) {
                int64_t nowp = av_gettime_relative();
                if (a->avsync_stat_last == 0) a->avsync_stat_last = nowp;
                else if (nowp - a->avsync_stat_last >= g_stats_period_us) {
                    int mv = a->multiview && g_audio_follow;
                    int64_t lag = a->house_skew ? *a->house_skew : 0;
                    /* avlag (1.0.1-pre13, was "lipsync=" — renamed so the lipsync= token appears ONLY
                     * on the -stats progress line, whose sign convention is OPPOSITE; the collision
                     * caused an oracle-analysis sign error 2026-07-16): the pipeline-introduced A/V
                     * lag ESTIMATE on this DIAG line — the AUDIO's realized output-vs-content lag
                     * (async_pad = outspan − content_span) minus the VIDEO's TRUE lag (lag_true).
                     * CONVENTION HERE: avlag > 0 = audio LATE (stats-line lipsync= is + = audio
                     * EARLY). (`offset` below is the independent vring-paired cross-check;
                     * avlag>0 ≈ offset<0.) */
                    int64_t content_us = av_rescale_q(a->dbg_last_src - a->dbg_first_src, a->ist_tb, AV_TIME_BASE_Q);
                    int64_t outspan_us = a->out_frames * (int64_t)a->frame_size * 1000000 / a->out_rate;
                    int64_t lag_true   = a->house_lag_true ? *a->house_lag_true : lag;
                    int64_t lserr      = (outspan_us - content_us) - lag_true;   /* async_pad − lag_true */
                    /* v0.6.19: the async_pad span estimate (lserr) does NOT include the PLL's content
                     * drop/pad retiming (af_applied_us), so on a CONVERGED PLL slot it kept reporting the
                     * bank the acquire already removed (avlag ≈ applied) — reading "off" while the
                     * faithful vring-paired offset was ~0. Headline the faithful measured offset when it
                     * has paired (− because offset<0 = audio late ≡ avlag>0 = audio late); fall back to
                     * the span estimate only before the vring pairs (offset = --). */
                    int64_t lshead     = a->av_off_valid ? -a->av_offset_us : lserr;
                    char m[24];
                    if (a->av_off_valid) snprintf(m, sizeof m, "%+"PRId64"ms", a->av_offset_us / 1000);
                    else                 snprintf(m, sizeof m, "--");
                    if (mv)              /* multiview: lip-sync + measured offset (the slot's sound is placed by its frame sync) */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) avlag=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "acomp=%"PRId64"  [offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            a->acomp_cnt);
                    else                 /* single-input: lip-sync + measured offset + house-clock lock state */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) avlag=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "house_skew=%+"PRId64"ms  [avlag>0 / offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000, lag / 1000);
                    /* FAITHFUL resampler-slip sensor (the one signal offset=/sync_check-D can't see).
                     * If swr_delay grows unbounded → the hard-comp (min_hard_comp) is NOT firing (AVLOCK
                     * may be masking delta); if it stays bounded → the slip is being corrected. */
                    if (a->fg_swr) {
                        int64_t dms   = swr_get_delay(a->fg_swr, 1000);
                        int64_t dsamp = swr_get_delay(a->fg_swr, a->out_rate);  /* output samples — shows sub-ms delay the ms field rounds to 0 */
                        if (dms > a->fg_swr_delay_max_ms) a->fg_swr_delay_max_ms = dms;
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-SWRDELAY] a%d(in%d) swr_delay=%"PRId64"ms (%"PRId64" samp, max %"PRId64"ms)\n",
                            a->dbg_k, a->dbg_in, dms, dsamp, a->fg_swr_delay_max_ms);
                    }
                    if (a->dbg_k == 0) {   /* [PTV-CHAIN] data-driven A/V trace, primary track only */
                        int64_t vs  = atomic_load_explicit(&g_ch_vsrc, memory_order_relaxed);
                        int64_t as  = atomic_load_explicit(&g_ch_asrc, memory_order_relaxed);
                        int64_t vsr = atomic_load_explicit(&g_ch_vsrc_raw, memory_order_relaxed);
                        int64_t asr = atomic_load_explicit(&g_ch_asrc_raw, memory_order_relaxed);
                        int64_t vo  = atomic_load_explicit(&g_ch_vout_src, memory_order_relaxed);
                        int64_t ao  = atomic_load_explicit(&g_ch_aout_src, memory_order_relaxed);
                        /* rawA-V grows → source-inherent A/V drift (→ §5.B genlock); rawA-V flat but
                         * unwrap_inj grows → demux_unwrap injects it (→ §5.A program rebase). */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-CHAIN] demux rawA-V=%+"PRId64"ms srcA-V=%+"PRId64"ms (unwrap_inj=%+"PRId64"ms) | outA-V=%+"PRId64"ms | introduced=%+"PRId64"ms\n",
                            (asr-vsr)/1000, (as-vs)/1000, ((as-vs)-(asr-vsr))/1000, (ao-vo)/1000, ((ao-vo)-(as-vs))/1000);
                    }
                    a->avsync_stat_last = nowp;
                }
            }
            /* PTV_DIAG per-slot lip-sync probe (box-usable, NO markers needed). The faithful
             * pipeline-introduced lip-sync error for this slot is:
             *     err = async_pad − house_skew
             * where house_skew (compositor-measured) = output_time − video_content_time = the
             * video's realized output-vs-content lag, and async_pad = outspan − content_span =
             * the AUDIO's realized output-vs-content lag (it already INCLUDES the house_skew we
             * inject into the audio input pts, plus any EXTRA async over/under-production). When
             * the audio actuator tracks the video retiming exactly, async_pad ≈ house_skew and
             * err ≈ 0 = in sync. + = audio late (async over-produced beyond the commanded skew —
             * the failure mode the ADR feared); − = audio early. This is the HONEST replacement
             * for av_off, which compared the two PRODUCTION THREADS' progress at one wall instant
             * and so read production-buffer lead (+3.5s offline / −0.2s live for the SAME in-sync
             * content), not playback sync. NOTE: this isolates error INTRODUCED BY ptvencoder; a
             * source feed that is itself A/V-misaligned passes through with err≈0 (compare to a
             * single-input output of the same feed to attribute source-side offset).
             * 2.0.0: single input only, once per stats period. On a mosaic slot err is dominated by
             * async_pad (−0.2 … −5.4 s read on an in-sync slot) — not lip sync; the slot's sensor is the
             * inK:lipsync on the mv stats line and [PTV-RSYNC]. */
            if (g_diag && !a->multiview) {
                int64_t now = av_gettime_relative();
                if (a->dbg_first_out == AV_NOPTS_VALUE) a->dbg_first_out = opts;
                if (now - a->dbg_diag_last >= FFMAX(1000000, g_stats_period_us)) {
                    int64_t content_us   = av_rescale_q(a->dbg_last_src - a->dbg_first_src, a->ist_tb, AV_TIME_BASE_Q);
                    int64_t outspan_us   = a->out_frames * (int64_t)a->frame_size * 1000000 / a->out_rate;
                    int64_t async_pad_us = outspan_us - content_us;                  /* audio realized output−content */
                    int64_t hs_us        = a->house_skew ? *a->house_skew : 0;       /* commanded skew (capped/floored) */
                    int64_t lag_us       = a->house_lag_true ? *a->house_lag_true    /* TRUE video lag (multiview) … */
                                                             : hs_us;               /* … or house_skew (single-input, uncapped) */
                    a->dbg_diag_last = now;
                    /* err = audio's realized lag − video's TRUE lag. Using the TRUE (uncapped) video lag, not the
                     * commanded house_skew, so a slot pinned at the 250ms cap or floored at 0 (video racing/dragging
                     * beyond the correctable range) still surfaces. lag≠house_skew ⇒ the actuator is saturated. */
                    av_log(NULL, AV_LOG_INFO,
                        "[PTV-LIPSYNC] a%d(in%d) err=%+"PRId64"ms  (lag_true=%+"PRId64"ms house_skew=%+"PRId64"ms async_pad=%+"PRId64"ms first_out=%"PRId64"ms)  [+ = audio late]\n",
                        a->dbg_k, a->dbg_in, (async_pad_us - lag_us) / 1000,
                        lag_us / 1000, hs_us / 1000, async_pad_us / 1000, a->dbg_first_out * 1000 / a->out_rate);
                }
            }
            /* 1.0.1-pre26 D3 instrumentation (PTV_MUXDIAG=1, temporary/gated): the emission-point
             * backward detector — the closest upstream twin of [PTV-MUXGUARD]. Fires when THIS
             * emission's label precedes the last emitted one (the exact packet the muxer would
             * EINVAL), dumping the composition state so the preceding loud label-edit lines
             * (AGLUE/CONV/WALLEV/ANCHOR/flush) name the emitter. */
            if (g_muxdiag && a->out_pts_set && filt->pts <= a->out_last_pts) {
                int64_t now_md = av_gettime_relative();
                if (now_md - a->md_warn_wc >= 500000) {
                    a->md_warn_wc = now_md;
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-MUXDIAG] a%d(in%d) BACKWARD EMISSION %+"PRId64"ms (out %"PRId64" <= last %"PRId64" smp) "
                           "glue_off=%+"PRId64"ms pend_comp=%+"PRId64"ms corr=%+"PRId64"ms reanch_win=%s "
                           "afmt_rebuilds=%d nbs_fill=%d reanch_mono=%d conv_bl=%+"PRId64"ms wev_out=%+"PRId64"ms\n",
                           a->dbg_k, a->dbg_in,
                           av_rescale(filt->pts - a->out_last_pts, 1000, a->out_rate > 0 ? a->out_rate : 48000),
                           filt->pts, a->out_last_pts,
                           a->glue_off_us / 1000, a->pend_comp_us / 1000, a->corr.corr_us / 1000,
                           (a->reanch_wc && now_md - a->reanch_wc <= PTV_REANCH_WIN_US) ? "IN" : "out",
                           a->afmt_rebuilds, a->nbs_fill_active, a->reanch_mono,
                           a->conv_bl_us / 1000, a->wev_out_us / 1000);
                }
            }
            a->out_last_pts = filt->pts;   /* F1: the guard's monotonic reference (all paths) */
            a->out_pts_set  = 1;
        }
        if (a->afmt_draining && a->frame_size > 0 && filt->nb_samples < a->frame_size) {
            /* 2.0.0-pre9.6: the drained old graph's last PARTIAL frame — the encoder would pad it to a full frame
             * (overlap with the rebuilt path, then the monotonic guard's hole); its span is re-covered by the
             * rebuild's silence bridge instead */
            a->afmt_drop_us += av_rescale(filt->nb_samples, 1000000, a->out_rate > 0 ? a->out_rate : 48000);
            av_frame_unref(filt);
            continue;
        }
        ret = audio_encode_push(a, filt);
        a->out_frames++;
        av_frame_unref(filt);
        if (ret < 0) break;
    }
    av_frame_free(&filt);
    return (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ? 0 : ret;
}

/* Require this many consecutive frames at new audio params before rebuilding, so a transient or
 * corrupt single-frame flip at a splice boundary doesn't trigger a spurious reconfig (legacy 0003). */
#define PTV_AFMT_HYSTERESIS 5

/* memcap F1: AFMT rebuild-storm circuit-breaker master switch (-1 = parse PTV_NO_AFMT_BREAKER
 * on first confirmed rebuild candidate; audio threads race benignly — same getenv result). */
static int g_afmt_breaker = -1;

/* 1.0.1-pre15 #33 (g_glueclass): publish this track's NEWEST still-open pad-ledger entry so
 * the demux §5.A.2 absorber can decline to erase a matching backward return leg at the packet
 * layer (advisory — AGLUE's own ledger scan is the authoritative match). Audio thread only. */
static void ptv_pad_pub(AudioState *a)
{
    int pi, best = -1;
    if (a->dbg_k < 0 || a->dbg_k >= PTV_MAX_AUDIO)
        return;
    for (pi = 0; pi < PTV_GLUE_PAD_LED; pi++)
        if (a->pad_led_us[pi] > 0 &&
            (best < 0 || a->pad_led_wc[pi] > a->pad_led_wc[best]))
            best = pi;
    atomic_store_explicit(&g_pad_pub_wc[a->dbg_k],
                          best >= 0 ? a->pad_led_wc[best] : 0, memory_order_relaxed);
    atomic_store_explicit(&g_pad_pub_step[a->dbg_k],
                          best >= 0 ? a->pad_led_us[best] : 0, memory_order_release);
}


/* [PTV-ANCHOR] house mapping, factored (1.0.1-pre20): a content position's house-anchored
 * output base in out_rate samples — THE single copy of the birth anchoring formula, shared
 * by the birth anchor (audio_anchor_and_feed) and the rebuild re-anchor below so the two
 * can never diverge. */
static int64_t ptv_anchor_next_pts(const AudioState *a, int64_t house_us)
{
    return av_rescale(house_us, a->out_rate, 1000000);
}

/* ===================== 1.0.1-pre20 REBUILD RE-ANCHOR (g_rebuild_reanchor) =====================
 * Called at [PTV-AFMT] rebuild COMPLETION (audio_feed, after the path is rebuilt for the
 * confirmed params — the ACHOP full rebuild completes through the same site, so it takes the
 * same path). TODAY'S DEFECT: the rebuilt path inherited its audio base from carried state —
 * glue_off_us polluted by broken-phase relabel-erasures, the swr-fallback next_pts free
 * counter frozen across the outage, a live corr trim sized against the dead mapping — so the
 * outage/transition duration leaked in as a ~1s residual the corrector then walked for ~10min
 * (Azorse live 2026-07-20 10:54: rebuild → R +1021ms → 599s walk). The source is A/V-synced
 * at the change; the residual is OURS. Re-derive the base BIRTH-EQUIVALENT from the current
 * house/video mapping: output = content − h0 with glue_off_us = 0 (exactly the [PTV-ANCHOR]
 * birth state; AVLOCK's house_skew stays a dynamic door term on both paths), via the SAME
 * factored formula birth uses. Pieces:
 *  (a) base: glue_off_us → 0, next_pts re-derived from the confirming frame's content,
 *      fresh AGLUE label baseline (no step classification across the rebuild), pad ledger +
 *      pending tripwire cleared — carried ledgers describe the dead mapping;
 *  (corr) carried corr_us is RETIRED to 0 (logged): the new base already includes reality,
 *      so keeping the old trim would DOUBLE-apply it; a steering/parked corrector falls back
 *      to ARMED (fresh full dwell required — §4's re-engagement rule);
 *  (b) m_a EMA re-seed (rs_ma_seed=0): a rebuild REDEFINES this track's mapping baseline —
 *      the pre17 rule (REANCHOR2 / ACQUIRE / slate-recovery already re-seed). m_v is NOT
 *      touched: an audio-path rebuild does not move the slot's video mapping;
 *  (c) armed here, applied in audio_drain_fg: within the first 10s post-rebuild (the rebuild
 *      DISCONTINUITY WINDOW — the rebuild is already audible, so a step is free), once ≥10
 *      frames have emitted and the re-seeded sensor still reads |R| > the engage band, the
 *      residual is applied as ONE immediate base step (cap ±5s, one WARNING); outside the
 *      window the step can never fire — the corrector owns anything later.
 * Birth is untouched (this runs only with pts_set, and only from the AFMT rebuild site);
 * REANCHOR2/ACQUIRE mv re-seed paths are untouched. PTV_NO_REBUILD_REANCHOR=1 reverts all
 * three pieces. */

static void ptv_rebuild_reanchor(AudioState *a, const AVFrame *frame)
{
    int64_t h0, content_us, house_us, old_glue, old_corr;
    int pi;
    if (!g_rebuild_reanchor || !a->pts_set)
        return;
    pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
    if (h0 == AV_NOPTS_VALUE || frame->best_effort_timestamp == AV_NOPTS_VALUE)
        return;
    content_us = av_rescale_q(frame->best_effort_timestamp, a->ist_tb, AV_TIME_BASE_Q);
    house_us   = content_us - h0;
    if (house_us < 0)
        return;                        /* content precedes h0: the birth rule would drop it too */
    old_glue = a->glue_off_us;
    old_corr = a->corr.corr_us;
    /* (a) base re-derivation — LABEL-NEUTRAL at the door (pre20 fix round F1, reviewer
     * blocker): the first cut zeroed glue_off and corr outright, which stepped the door
     * labels BACKWARD by the retired sum → backward DTS → mpegts muxer death on the first
     * write (dead wire, zombie process). And zeroing glue was WRONG in the balanced case:
     * the sensor pairs glue against the demux edit ledgers (R = dm + E_v − E_a), so a glue
     * that mirrors ledgered video edits is label-TRUTH, not pollution — discarding it
     * re-opened R by the ledger difference (fixture: dm −261/ev +821 → R +560). The
     * retirement is therefore a BOOKKEEPING TRANSFER: corr folds INTO glue_off (inj sum
     * unchanged ⇒ door labels perfectly continuous ⇒ the blocker is mechanically
     * impossible at retirement), and whatever part of the carried sum does NOT match
     * reality shows up as R, which the (c) residual step(s) below retire with an explicit,
     * forward-or-guarded label edit. The corrector's own trim is still retired (corr = 0,
     * state → ARMED) — it never double-applies. */
    a->glue_off_us      = old_glue + old_corr;
    {   /* F1: the swr-fallback free counter must never step backward either (same mux
         * invariant); a backward re-derivation is clamped to the current counter — the
         * fallback path realizes forward re-anchors only, the residual stays with the
         * sensor/corrector. */
        int64_t np = ptv_anchor_next_pts(a, house_us);
        if (np < a->next_pts)
            np = a->next_pts;
        a->next_pts = np;
    }
    a->glue_raw_last_us = AV_NOPTS_VALUE;      /* fresh label baseline: the first post-rebuild
                                                * frame is a reference, never a "step" */
    for (pi = 0; pi < PTV_GLUE_PAD_LED; pi++)
        a->pad_led_us[pi] = 0;
    ptv_pad_pub(a);
    a->pend_comp_us = 0;
    a->acomp_exp_us = AV_NOPTS_VALUE;          /* the base reset is deliberate — not a click risk
                                                * (the resampler was just rebuilt, no old baseline) */
    /* (corr) retire the carried trim — the fresh base includes reality */
    if (old_corr || a->corr.state == PTV_CORR_DWELL ||
        a->corr.state == PTV_CORR_ENGAGED || a->corr.state == PTV_CORR_PARKED) {
        a->corr.corr_us       = 0;
        a->corr.engaged_corr0 = 0;
        if (a->corr.state == PTV_CORR_DWELL || a->corr.state == PTV_CORR_ENGAGED ||
            a->corr.state == PTV_CORR_PARKED)
            a->corr.state = PTV_CORR_ARMED;    /* re-engagement requires a fresh full dwell */
        if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO) {
            atomic_store_explicit(&g_corr_pub[a->dbg_k], 0, memory_order_relaxed);
            atomic_store_explicit(&g_corr_state_pub[a->dbg_k], a->corr.state, memory_order_relaxed);
        }
    }
    /* (b) sensor m_a re-seed — the rebuild is a baseline redefinition (pre17 rule) */
    a->rs_ma_seed = 0;
    /* (c) arm the residual-step window */
    a->reanch_wc      = av_gettime_relative();
    a->reanch_frames  = 0;
    a->reanch_step_us = 0;
    av_log(NULL, AV_LOG_WARNING,
           "[PTV-ANCHOR] a%d(in%d) re-anchored (afmt-rebuild): base=content-h0 at %+"PRId64"ms, "
           "corr %+"PRId64"ms retired into the base ledger (door labels continuous; carried glue "
           "%+"PRId64"ms), sensor re-seeded, residual window %ds (PTV_NO_REBUILD_REANCHOR reverts)\n",
           a->dbg_k, a->dbg_in, house_us / 1000, old_corr / 1000, old_glue / 1000,
           (int)(PTV_REANCH_WIN_US / 1000000));
}

/* Feed one h0-anchored decoded audio frame into the -af graph (or swr fallback). */
/* 2.0.0-pre8.1: may the real frames buffered after a filled hold go to the door now? */
static int rj_release_ok(AudioState *a)
{
    int64_t now = av_gettime_relative();
    int64_t rj  = atomic_load_explicit(&g_src_rejoin_wall, memory_order_relaxed);
    if (a->rj_t0 && now - a->rj_t0 > 4000000)
        return 1;                                            /* bounded: never hold audio longer */
    if (!ptv_src_holding() && rj <= a->hold_fill_wc0)       /* a short repeat run, no hold */
        /* 2.1 T-409: the timeline's labels carry a kept hole and house_skew keeps its pre-gap value (T-321): nothing
         * to wait for. Waiting for a fresh frame raced the video's own freeze — a fill begun late, as the last pre-gap
         * picture was still showing, held the return to the post-gap picture (sync_flapping: audio dark 1.77 s) */
        return 1;
    return ptv_rj_settled(a->rj_cls_seq0, now);   /* hold over, rejoin classified, its house_skew shown (NEW DOMAIN:
                                                   * 200 ms — 1 s of holding cost a 0.9 s dark wire for nothing) */
}

static void rj_flush(AudioState *a);

/* 2.0.0 T-321: the house_skew a real frame meets at the graph door. A short source gap (no hold) ends at the door
 * before it ends on air: the returning audio arrives while the video still repeats frames through the gap, so
 * house_skew carries that dup run (2 s flaps: +600 ms); ~1.6 s later the video shows the post-gap content, house_skew
 * falls back and aresample drops what it padded — the sound played late, then lost its first ~0.6 s (measured: every
 * return of sync_flapping, the ruler's +600 ms shrinking one frame per cycle). The returning frames keep the pre-gap
 * house_skew until the video is back on it (bounded 4 s; then the current value, as before). A hold (STALLED →
 * rejoin) is not touched: its return has its own buffer and mapping. PTV_NO_HS_GAPKEEP=1 = off. */
static int64_t hs_door(AudioState *a, const AVFrame *frame, int64_t hs)
{
    static int off = -1;
    int64_t now = av_gettime_relative(), src = AV_NOPTS_VALUE, out = hs;
    int64_t tol = a->tick_dur_us > 0 ? a->tick_dur_us + a->tick_dur_us / 4 : 50000;
    int gap, hold;
    if (off < 0)
        off = !!getenv("PTV_NO_HS_GAPKEEP");
    if (off)
        return hs;
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
        src = av_rescale_q(frame->best_effort_timestamp, a->ist_tb, AV_TIME_BASE_Q);
    gap = a->hsg_seen && src != AV_NOPTS_VALUE && a->hsg_src != AV_NOPTS_VALUE &&
          src - a->hsg_src - a->hsg_dur > 500000;
    hold = ptv_src_holding() || atomic_load_explicit(&g_src_rejoin_wall, memory_order_relaxed) > a->hsg_wc;
    if (hold)
        a->hsg_on = 0;      /* a hold took over: its return has its own buffer and mapping */
    /* flapping: the video often returns while the audio is already in its NEXT gap, so no frame sees house_skew back
     * on the kept value; the next gap's return then still keeps it — the bound counts from the latest gap */
    if (a->hsg_on && gap)
        a->hsg_t0 = now;
    if (a->hsg_on) {
        if (llabs(hs - a->hsg_val) <= tol || now - a->hsg_t0 > 4000000) {
            if (a->hsg_n <= 3)
                av_log(NULL, AV_LOG_INFO, "[PTV-AVLOCK] a%d(in%d) short gap: house_skew %+"PRId64" ms again after "
                       "%.2f s (%s)\n", a->dbg_k, a->dbg_in, hs / 1000, (now - a->hsg_t0) / 1e6,
                       llabs(hs - a->hsg_val) <= tol ? "the video is back" : "bounded");
            a->hsg_on = 0;
        } else
            out = a->hsg_val;
    } else if (gap && hs - a->hsg_hs > tol && !hold) {
        a->hsg_on  = 1;
        a->hsg_val = out = a->hsg_hs;
        a->hsg_t0  = now;
        if (++a->hsg_n <= 3)
            av_log(NULL, AV_LOG_INFO, "[PTV-AVLOCK] a%d(in%d) short gap (%+"PRId64" ms, no hold): the returning audio "
                   "keeps house_skew %+"PRId64" ms (now %+"PRId64" ms) until the video shows it%s\n", a->dbg_k,
                   a->dbg_in, (src - a->hsg_src - a->hsg_dur) / 1000, out / 1000, hs / 1000,
                   a->hsg_n == 3 ? " (further gaps not logged)" : "");
    }
    a->hsg_seen = 1;
    a->hsg_src  = src;
    a->hsg_dur  = frame->sample_rate > 0 ? av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0;
    a->hsg_hs   = out;
    a->hsg_wc   = now;
    return out;
}

/* 2.0.0-pre9.6: frames that confirm a source format change are held for the rebuilt path (were dropped) */
static void afmt_q_reset(AudioState *a)
{
    while (a->afmt_qn > 0)
        av_frame_free(&a->afmt_q[--a->afmt_qn]);
}
static void afmt_q_add(AudioState *a, const AVFrame *frame)
{
    if (a->afmt_qn < PTV_AFMT_QMAX) {
        AVFrame *c = av_frame_clone(frame);
        if (c)
            a->afmt_q[a->afmt_qn++] = c;
    }
}

/* 2.0.0-pre17 T-347/T-349: the source-format converter in front of the fixed-input graph (see cv_swr) */
static int fmt_is(const AVFrame *f, int rate, enum AVSampleFormat fmt, const AVChannelLayout *chl)
{
    return f->sample_rate == rate && f->format == fmt && !av_channel_layout_compare(&f->ch_layout, chl);
}

/* the graph's fixed input params: 48 kHz float planar in the track's output layout */
void ptv_graph_in_params(AudioState *a)
{
    a->fg_in_rate = 48000;
    a->fg_in_fmt  = AV_SAMPLE_FMT_FLTP;
    av_channel_layout_uninit(&a->fg_in_chl);
    av_channel_layout_copy(&a->fg_in_chl, &a->out_chl);
}

/* (re)form the converter for a source format; rate <= 0 = not known yet (the first confirmed frames form it) */
int ptv_cv_setup(AudioState *a, int rate, enum AVSampleFormat fmt, const AVChannelLayout *chl)
{
    swr_free(&a->cv_swr);
    av_channel_layout_uninit(&a->cv_chl);
    a->cv_rate = -1; a->cv_fmt = AV_SAMPLE_FMT_NONE;
    if (rate <= 0 || fmt == AV_SAMPLE_FMT_NONE || !chl || chl->nb_channels <= 0)
        return 0;
    if (rate != a->fg_in_rate || fmt != a->fg_in_fmt || av_channel_layout_compare(chl, &a->fg_in_chl)) {
        /* rematrix_maxval 1: a downmix is scaled to stay within full scale, as swr does for integer output — for
         * float output it is NOT by default: a 5.1 ruler downmixed +7.8 dB hot (measured), and loudnorm then rode a
         * level jump the source never had (the integer-output graph of pre16 got the scaled one) */
        if (swr_alloc_set_opts2(&a->cv_swr, &a->fg_in_chl, a->fg_in_fmt, a->fg_in_rate,
                                chl, fmt, rate, 0, NULL) < 0 ||
            av_opt_set_double(a->cv_swr, "rematrix_maxval", 1.0, 0) < 0 || swr_init(a->cv_swr) < 0) {
            swr_free(&a->cv_swr);
            return AVERROR(EINVAL);
        }
    }
    a->cv_rate = rate; a->cv_fmt = fmt;
    av_channel_layout_copy(&a->cv_chl, chl);
    return 0;
}

static int audio_feed(AudioState *a, AVFrame *frame);

/* 2.1 T-409 (P1, design §3.3/§5.2): the audio is placed where the VIDEO showed the same content (its record, vmap), not
 * with the house_skew of the moment the frame reaches the door — the video shows a label ~1 s after its sound arrives, and
 * around an event that skew moves in between (sync_storm: holes in quick succession inside one hold put the returning
 * sound +15 / -15 / +24 s off its fill). Single input, the record door: active within 30 s of a program event and while
 * the source holds. Real frames wait until their content is on air (at least 3 s, and while the video still shows new
 * pictures; fill sentinels drain the queue by time); content never shown as such (its pictures cut, or a video-only
 * outage) continues right after the previous placed frame, or takes the house_skew placement when there is none
 * within 1 s. The silence fill follows where the video is on air. PTV_TL_NO_VMDOOR=1 = the house_skew placement. */
static int vm_mode(const AudioState *a)
{
    static int on = -1;
    int64_t pw;
    if (on < 0)
        on = !getenv("PTV_TL_NO_VMDOOR");  /* the default under the timeline (sync_storm, bursty, flapping);
                                            * PTV_TL_NO_VMDOOR=1 = the house_skew placement */
    if (!on || !g_vmap || a->multiview || !g_avlock || !a->house_skew)
        return 0;
    if (a->vm_n)
        return 1;                                   /* drain in order */
    pw = atomic_load_explicit(&g_tl_prog_wc, memory_order_relaxed);
    return ptv_src_holding() || (pw && av_gettime_relative() - pw < 30000000);
}

static int vm_drain(AudioState *a, int force)
{
    int64_t now = av_gettime_relative(), tick = atomic_load_explicit(&g_house_tick_us, memory_order_relaxed);
    int ret = 0;
    while (a->vm_n && ret >= 0) {
        AVFrame *f = a->vm_q[0];
        int64_t o = AV_NOPTS_VALUE, gap = 0, lab = av_rescale_q(f->best_effort_timestamp, a->ist_tb, AV_TIME_BASE_Q);
        int r = ptv_vmap_lookup(lab, &o, &gap);
        if (r == 0 && !force && (now - a->vm_wc[0] < 3000000 ||
            now - atomic_load_explicit(&g_src_fresh_wc, memory_order_relaxed) < 500000))
            break;                                  /* not on air yet: wait while the video still shows new pictures (a burst
                                                     * backlog waits its whole bank delay), at least 3 s */
        a->vm_out_us = r == 1 && gap <= FFMAX(2 * tick, 50000) ? o : AV_NOPTS_VALUE;
        /* content never shown as such (its pictures cut at the burst door, or still held when the video went stale):
         * it continues where the previous frame was placed — the house_skew placement belongs to another domain
         * (sync_bursty_20: 4 frames at the end of a cut stepped the door input −15 s, audio PID dark 1.6 s) */
        if (a->vm_out_us == AV_NOPTS_VALUE && a->vm_last_out != AV_NOPTS_VALUE &&
            lab > a->vm_last_lab && lab - a->vm_last_lab < 1000000)
            a->vm_out_us = a->vm_last_out + (lab - a->vm_last_lab);
        a->vm_last_lab = lab;
        a->vm_last_out = a->vm_out_us;
        memmove(a->vm_q, a->vm_q + 1, (a->vm_n - 1) * sizeof(*a->vm_q));
        memmove(a->vm_wc, a->vm_wc + 1, (a->vm_n - 1) * sizeof(*a->vm_wc));
        if (!--a->vm_n)
            atomic_fetch_sub_explicit(&g_vm_waiting, 1, memory_order_relaxed);
        a->vm_feeding = 1;
        ret = audio_feed(a, f);
        a->vm_feeding = 0;
        av_frame_free(&f);
    }
    return ret;
}

/* convert one source frame to the graph's input and feed it. Its label = the source label minus what the converter
 * still holds (the output starts that much earlier in source time). */
static int cv_feed(AudioState *a, const AVFrame *in)
{
    AVFrame *cv;
    int64_t held_us;
    int n, ret;
    cv = av_frame_alloc();
    if (!cv)
        return AVERROR(ENOMEM);
    av_frame_copy_props(cv, in);
    cv->format      = a->fg_in_fmt;
    cv->sample_rate = a->fg_in_rate;
    av_channel_layout_copy(&cv->ch_layout, &a->fg_in_chl);
    cv->nb_samples  = swr_get_out_samples(a->cv_swr, in->nb_samples);
    if (cv->nb_samples <= 0 || av_frame_get_buffer(cv, 0) < 0) {
        av_frame_free(&cv);
        return 0;
    }
    held_us = swr_get_delay(a->cv_swr, 1000000);
    n = swr_convert(a->cv_swr, cv->extended_data, cv->nb_samples, (const uint8_t **)in->extended_data, in->nb_samples);
    if (n <= 0) {
        av_frame_free(&cv);
        return n;
    }
    cv->nb_samples = n;
    if (in->pts != AV_NOPTS_VALUE)
        cv->pts = in->pts - av_rescale_q(held_us, AV_TIME_BASE_Q, a->ist_tb);
    if (in->best_effort_timestamp != AV_NOPTS_VALUE)
        cv->best_effort_timestamp = in->best_effort_timestamp - av_rescale_q(held_us, AV_TIME_BASE_Q, a->ist_tb);
    cv->duration = av_rescale_q(n, (AVRational){ 1, a->fg_in_rate }, a->ist_tb);
    a->cv_feeding = 1;
    ret = audio_feed(a, cv);
    a->cv_feeding = 0;
    av_frame_free(&cv);
    return ret;
}

static int audio_feed(AudioState *a, AVFrame *frame)
{
    uint8_t **out = NULL;
    int out_max, got, ret = 0;
    if (a->cv_feeding)
        goto graph_in;                             /* a converted frame: vetted by its source frame's pass */
    if (a->multiview && g_audio_follow && a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO) {   /* T-352 ledger history */
        int64_t hw = av_gettime_relative();
        if (hw - a->ea_hist_wc[(a->ea_hist_i + PTV_EA_HIST - 1) % PTV_EA_HIST] >= 500000) {
            a->ea_hist_wc[a->ea_hist_i] = hw;
            a->ea_hist_v[a->ea_hist_i]  = atomic_load_explicit(&g_rsx.ea_us[a->dbg_k], memory_order_relaxed);
            a->ea_hist_i = (a->ea_hist_i + 1) % PTV_EA_HIST;
        }
    }
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
        a->dbg_last_src = frame->best_effort_timestamp;   /* probe: latest fed source pts */
    if (!a->vm_feeding && !a->hold_feeding && !a->rj_releasing && frame->best_effort_timestamp != AV_NOPTS_VALUE &&
        vm_mode(a)) {
        AVFrame *c;
        if (a->hold_fill_us && ptv_src_holding() && !ptv_rj_burst_since(a->rj_cls_seq0))
            return 0;                               /* a fragment during the hold: the fill covers it (T-083) */
        if (a->vm_n == PTV_VMQ && (ret = vm_drain(a, 1)) < 0)
            return ret;
        if (!(c = av_frame_clone(frame)))
            return AVERROR(ENOMEM);
        a->vm_wc[a->vm_n]  = av_gettime_relative();
        if (!a->vm_n)
            atomic_fetch_add_explicit(&g_vm_waiting, 1, memory_order_relaxed);
        a->vm_q[a->vm_n++] = c;
        return vm_drain(a, 0);
    }
    /* 2.0.0-pre6/pre8.1: real frames after a filled hold must meet the POST-rejoin house_skew. It is written on the
     * tick that shows the post-gap content; fed earlier, a LIVE LOSS rejoin counts the hole twice at the door (labels
     * +W and the hold-grown house_skew +W): aresample pads ~W and drops it again at the snap — the program's audio
     * replaced by silence for about the outage length (live-transcoder 2026-10-04: GB_News 45 s outage → ~47 s of
     * silence after the source returned). pre6 BLOCKED until "a fresh frame since the fill began", which the canary
     * showed is not enough: the fill can start while the frame queue still drains (B), and the first fresh frame
     * after a hold can be the pre-gap PES tail while the post-gap frames are dropped until the next IDR (A). Now the
     * frames are BUFFERED (the audio thread keeps serving fill sentinels) until rj_release_ok(). */
    if (!a->hold_feeding && !a->rj_releasing && !a->vm_feeding && a->hold_fill_us && !a->rj_released) {
        /* 2.0.0-pre9.4 (T-083): while the source still HOLDS, a real frame is a fragment that survived the damage (the
         * video is dead, bars are on air). Fed — at once or from this buffer after its 4 s cap — it met the hold-grown
         * house_skew AND its own label jump (Cinestar 50 % loss: one AAC frame 88 s after the last, +95 s of
         * house_skew → aresample padded ~1000 frames ahead of the video, the delivery FIFO filled and the copied AC-3
         * was dropped for 30 s); kept for after the rejoin it is stale (−29 s re-label). The fill already covers its
         * time: drop it; the return is buffered below once the hold has ended, as before. */
        /* 2.0.0 T-315 (was T-103): only while the input is still gone. Once it has come back as a BURST the frames are
         * the returning content — a sender that catches up delivers the paused stretch in one burst, before the video
         * has ended the hold — and dropping them put the pictures on air with no sound (sync_stop_20: 21 flashes over
         * silence, also stop_5: 5). They go to the rejoin buffer below like any return. Other classes: see
         * ptv_rj_burst_since(). */
        if (ptv_src_holding() && !ptv_rj_burst_since(a->rj_cls_seq0)) {
            if (!a->hold_drop_n++)
                av_log(NULL, AV_LOG_INFO, "[PTV-SRC] a%d(in%d) real frames during the hold are dropped — the fill "
                       "covers the track until the rejoin\n", a->dbg_k, a->dbg_in);
            return 0;
        }
        if (!rj_release_ok(a) && a->rj_n < PTV_RJ_BUF) {
            AVFrame *c = av_frame_clone(frame);
            if (c) {
                if (!a->rj_n) {
                    a->rj_t0 = av_gettime_relative();
                    atomic_fetch_add_explicit(&g_rj_buffering, 1, memory_order_relaxed);
                }

                a->rj_buf[a->rj_n++] = c;
                return 0;
            }
        }
        rj_flush(a);
    }

    /* 2.0.0-pre17 T-347/T-349: with a graph, a source format switch only re-forms the converter in front of it (same
     * 5-frame hysteresis: a lone corrupt frame is held, not converted into a noise burst). The graph and its state
     * (loudnorm, aresample history, the anchor) carry on. The full rebuild below remains for a torn-down / failed
     * path (ACHOP, init failure → plain swr). */
    if (a->use_fg && a->afg) {
        if (a->hold_feeding || a->nbs_feeding)
            goto graph_in;                         /* fills are synthesized in the graph's own format */
        /* the source's CURRENT format is cv_* (also when it is the graph's own: no converter then). A change — a
         * return to the graph's format included — is a switch: confirmed, re-anchored (measured: the 5.1 -> stereo
         * return taken as plain frames left its splice -107 ms absorbed, +128 ms late for good) */
        {
            int same_pending;
            if (a->cv_rate > 0 && fmt_is(frame, a->cv_rate, a->cv_fmt, &a->cv_chl)) {
                if (a->afmt_stable) {              /* a candidate that did not confirm: transient */
                    a->afmt_stable = 0;
                    afmt_q_reset(a);
                    av_channel_layout_uninit(&a->afmt_pending_chl);
                    a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
                }
                if (a->cv_swr)
                    return cv_feed(a, frame);
                goto graph_in;
            }
            same_pending = a->afmt_stable && fmt_is(frame, a->afmt_pending_rate, a->afmt_pending_fmt, &a->afmt_pending_chl);
            if (!same_pending) {
                a->afmt_pending_rate = frame->sample_rate;
                a->afmt_pending_fmt  = frame->format;
                av_channel_layout_uninit(&a->afmt_pending_chl);
                av_channel_layout_copy(&a->afmt_pending_chl, &frame->ch_layout);
                a->afmt_stable = 1;
                afmt_q_reset(a);
                afmt_q_add(a, frame);
                return 0;
            }
            if (++a->afmt_stable < PTV_AFMT_HYSTERESIS) {
                afmt_q_add(a, frame);
                return 0;
            }
            {
                char ochl[64], nchl[64], gchl[64];
                int qi, qn = a->afmt_qn, qret = 0;
                AVFrame *q[PTV_AFMT_QMAX];
                if (a->cv_rate > 0)
                    av_channel_layout_describe(&a->cv_chl, ochl, sizeof ochl);
                else
                    snprintf(ochl, sizeof ochl, "unknown");
                av_channel_layout_describe(&frame->ch_layout, nchl, sizeof nchl);
                av_channel_layout_describe(&a->fg_in_chl, gchl, sizeof gchl);
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-AFMT] a%d(in%d) audio input changed %dHz %s %s -> %dHz %s %s (confirmed %d frames) — "
                       "%s %s/%dHz in front of the unchanged audio path\n", a->dbg_k, a->dbg_in,
                       a->cv_rate > 0 ? a->cv_rate : a->fg_in_rate,
                       a->cv_rate > 0 ? av_get_sample_fmt_name(a->cv_fmt) : av_get_sample_fmt_name(a->fg_in_fmt), ochl,
                       frame->sample_rate, av_get_sample_fmt_name(frame->format), nchl, a->afmt_stable,
                       fmt_is(frame, a->fg_in_rate, a->fg_in_fmt, &a->fg_in_chl) ? "no conversion: it is the" : "converted to",
                       gchl, a->fg_in_rate);
                a->afmt_stable = 0;
                av_channel_layout_uninit(&a->afmt_pending_chl);
                a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
                memcpy(q, a->afmt_q, qn * sizeof(*q));
                a->afmt_qn = 0;
                if (ptv_cv_setup(a, frame->sample_rate, frame->format, &frame->ch_layout) < 0) {
                    av_log(NULL, AV_LOG_ERROR, "[PTV-AFMT] a%d(in%d) converter for the new format failed — frames "
                           "dropped until the next format\n", a->dbg_k, a->dbg_in);
                    for (qi = 0; qi < qn; qi++)
                        av_frame_free(&q[qi]);
                    return 0;
                }
                a->cv_switches++;
                /* a format switch is a splice: the source's own labels are its A/V truth from here on, as at a
                 * rebuild (1.0.1-pre20) — measured: the splice's audio steps back 171 ms (overlap), the demux absorbs
                 * it as continuous, and without the re-anchor the new stretch stayed +192 ms late for good. The graph
                 * carries on, so the overlap must not reach it: aresample=async drops an overlap only above
                 * min_hard_comp (30 ms); a smaller one — an AAC encoder's priming frame at the splice, 21.3 ms measured —
                 * passes as a backward step onto the wire (stock aresample, reproduced with ffmpeg; loudnorm's
                 * re-stamping hid it, any low-latency chain shows it). Single input: trim it at the door (below). */
                if (!a->multiview && a->acomp_exp_us != AV_NOPTS_VALUE) {
                    a->cv_trim_left   = qn + 1;   /* the replay: after a rate switch its first frames still carry
                                                   * lavf's old-rate timing (half-length steps) until the demux
                                                   * relabel has 3 stable headers — measured −10.7 ms twice at 96→48 */
                    a->cv_trim_exp_us = a->acomp_exp_us;
                }
                a->afmt_rebuilds++;                /* corrector event (§4.4) */
                ptv_rebuild_reanchor(a, qn ? q[0] : frame);
                if (a->multiview && g_audio_follow && a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO) {
                    /* T-352: the demux absorbed the splice's audio step as continuous (a label edit, posted to the ea
                     * ledger); the slot's PLL reads the label-domain pairing and cannot see it (MV sync_aswitch_1:
                     * +348 ms for good). At a format switch the source's labels are the truth: return exactly the
                     * ledger change of the last ~4 s (the demux reads ahead of the decoder) to the labels. The PLL
                     * keeps running (measured: it does not re-track the step; a forced stale-mark instead locked in the
                     * moment's reading: +68 ms for good on a switch WITHOUT an edit, sync_aswitch_96). */
                    int64_t nw = av_gettime_relative(), ea_now, ea_then = AV_NOPTS_VALUE, best = INT64_MAX;
                    int hi;
                    ea_now = atomic_load_explicit(&g_rsx.ea_us[a->dbg_k], memory_order_relaxed);
                    for (hi = 0; hi < PTV_EA_HIST; hi++) {
                        int64_t age = nw - a->ea_hist_wc[hi];
                        if (a->ea_hist_wc[hi] && age >= 4000000 && age < best) {
                            best = age;
                            ea_then = a->ea_hist_v[hi];
                        }
                    }
                    if (ea_then != AV_NOPTS_VALUE && ea_now != ea_then) {
                        a->glue_off_us -= ea_now - ea_then;
                        a->acomp_exp_us = AV_NOPTS_VALUE;     /* deliberate, not a click risk */
                        av_log(NULL, AV_LOG_WARNING, "[PTV-AFMT] a%d(in%d) splice edit %+"PRId64"ms returned to the "
                               "labels at the switch (mosaic slot: the source's labels are its A/V truth)\n",
                               a->dbg_k, a->dbg_in, (ea_now - ea_then) / 1000);
                    }
                }
                for (qi = 0; qi < qn; qi++) {
                    if (qret >= 0)
                        qret = audio_feed(a, q[qi]);
                    av_frame_free(&q[qi]);
                }
                return qret < 0 ? qret : audio_feed(a, frame);
            }
        }
    }

    /* Source audio format change (stereo↔mono, sample-rate, fmt) at a splice: the graph/swr was
     * configured for the prior params and abuffersrc rejects the changed frame → the audio path
     * wedges permanently (observed on TruBLU 2ch→1ch). Detect it, apply hysteresis to ignore
     * transient/corrupt flips, then rebuild the path for the new INPUT params. OUTPUT stays pinned
     * to out_chl/48k (the trailing aformat), so the AAC encoders keep getting continuous stereo —
     * mono is upmixed. Ported from legacy 0003 (fftools/ffmpeg_filter.c). */
    if (frame->sample_rate != a->fg_in_rate ||
        frame->format      != a->fg_in_fmt  ||
        av_channel_layout_compare(&frame->ch_layout, &a->fg_in_chl)) {
        int same_pending = (a->afmt_pending_rate == frame->sample_rate &&
                            a->afmt_pending_fmt  == frame->format &&
                            !av_channel_layout_compare(&a->afmt_pending_chl, &frame->ch_layout));
        if (!same_pending) {                       /* new candidate → start hysteresis, hold this frame */
            a->afmt_pending_rate = frame->sample_rate;
            a->afmt_pending_fmt  = frame->format;
            av_channel_layout_uninit(&a->afmt_pending_chl);
            av_channel_layout_copy(&a->afmt_pending_chl, &frame->ch_layout);
            a->afmt_stable = 1;
            afmt_q_reset(a);
            afmt_q_add(a, frame);
            return 0;
        }
        if (++a->afmt_stable < PTV_AFMT_HYSTERESIS) {
            afmt_q_add(a, frame);                  /* still settling — held (downstream can't take the change yet) */
            return 0;
        }
        /* memcap F1: rebuild-storm CIRCUIT-BREAKER. A flapping origin (CORELINK class flaps
         * 44.1<->48kHz across discontinuities, sometimes continuously) confirms a "new" format
         * over and over; every rebuild churns the whole -af graph (measured ~2MB RSS/rebuild of
         * allocator fragmentation with a FLAT live heap — the growth is unbounded only in
         * rebuild COUNT = the HTTV 26GB RSS-runaway class, and pre26's crash fix means flap
         * storms now survive long enough for it to matter). Enforce an ESCALATING minimum
         * interval between rebuilds: 2nd rebuild within 120s of the previous starts the backoff
         * at 2s, doubling to a 60s cap; 120s of rebuild quiet resets it. While the breaker
         * holds, changed-format frames keep being dropped (the existing settling posture —
         * during a flap storm the audio is upstream-garbage anyway, the Azorse broken-phase
         * rule); the flap that is still present when the interval expires rebuilds normally.
         * Bounds the churn to ~1 rebuild/min worst case. PTV_NO_AFMT_BREAKER=1 disables. */
        if (g_afmt_breaker < 0) {
            const char *bs = getenv("PTV_NO_AFMT_BREAKER");
            g_afmt_breaker = !(bs && atoi(bs));
        }
        if (g_afmt_breaker && a->afmt_next_ok_us) {
            int64_t bnow = av_gettime_relative();
            if (bnow < a->afmt_next_ok_us) {
                if (a->afmt_stable > PTV_AFMT_HYSTERESIS)
                    a->afmt_stable = PTV_AFMT_HYSTERESIS;   /* stay "confirmed", don't overflow */
                a->afmt_bk_dropped++;
                afmt_q_reset(a);
                if (bnow - a->afmt_bk_log_us >= 5000000) {
                    a->afmt_bk_log_us = bnow;
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-AFMT] a%d(in%d) rebuild BREAKER holding %.1fs more (backoff %.0fs, "
                           "flap storm; %"PRId64" frames dropped) — origin keeps flapping audio params "
                           "(PTV_NO_AFMT_BREAKER=1 disables)\n",
                           a->dbg_k, a->dbg_in, (a->afmt_next_ok_us - bnow) / 1e6,
                           a->afmt_backoff_us / 1e6, a->afmt_bk_dropped);
                }
                return 0;
            }
        }
        {   /* confirmed: rebuild for the new params (a->dec already reflects them) */
            char ochl[64], nchl[64], tchl[64];
            int64_t raw_end0 = a->glue_raw_last_us != AV_NOPTS_VALUE   /* the re-anchor below resets it */
                             ? a->glue_raw_last_us + a->glue_raw_dur_us : AV_NOPTS_VALUE;
            av_channel_layout_describe(&a->fg_in_chl, ochl, sizeof ochl);
            av_channel_layout_describe(&frame->ch_layout, nchl, sizeof nchl);
            av_channel_layout_describe(&a->out_chl, tchl, sizeof tchl);
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-AFMT] audio input changed %dHz %s %s -> %dHz %s %s "
                   "(confirmed %d frames) — rebuilding audio path; output stays %s/48k\n",
                   a->fg_in_rate, av_get_sample_fmt_name(a->fg_in_fmt), ochl,
                   frame->sample_rate, av_get_sample_fmt_name(frame->format), nchl,
                   a->afmt_stable, tchl);
            a->afmt_stable = 0;
            /* 2.0.0-pre9.6 (T-083): drain the old graph before it goes — what it still holds (the sink's partial
             * encoder frame, resampler and limiter tails) was lost at every rebuild: with the dropped confirming
             * frames a 0.23 s AAC hole at each source format switch (audio_format_switch, since pre9.3). */
            a->afmt_drop_us = 0;
            a->afmt_drain_end_us = a->acomp_exp_us;   /* the door's next expected label = the real content end */
            if (a->afg && a->afsrc && a->use_fg) {
                a->afmt_draining = 1;
                if (av_buffersrc_add_frame_flags(a->afsrc, NULL, 0) >= 0)
                    audio_drain_fg(a);
                a->afmt_draining = 0;
                if (raw_end0 != AV_NOPTS_VALUE)
                    raw_end0 -= a->afmt_drop_us;   /* the bridge re-covers the withheld partial tail */
            }
            if (a->afg) { avfilter_graph_free(&a->afg); a->afsrc = a->afsink = NULL; a->fg_swr = NULL; a->fg_swr_flt = NULL; }
            a->use_fg = 0;
            /* v0.9.17.1: rebuild the -af graph whenever the chain exists — INCLUDING for a track
             * whose startup init failed outright (Azorse: source in an undecodable-AAC phase at
             * open → probe/dec params garbage → graph AND swr init failed → the old code skipped
             * the track FOREVER, even after the source healed; the old `was_fg` gate then also
             * refused to build a graph here because the track never had one). a->dec reflects the
             * confirmed good frames now, so this build has real params. */
            if (a->fg_af &&
                build_audio_filter(a, a->dec, a->ist_tb, a->fg_af, a->out_sfmt) < 0) {
                avfilter_graph_free(&a->afg); a->use_fg = 0;
            }
            if (!a->use_fg) {                      /* originally swr, or graph rebuild failed → plain swr */
                swr_free(&a->swr);
                swr_alloc_set_opts2(&a->swr, &a->out_chl, a->out_sfmt, a->out_rate,
                                    &a->dec->ch_layout, a->dec->sample_fmt, a->dec->sample_rate, 0, NULL);
                if (a->swr) swr_init(a->swr);
            }
            a->afmt_rebuilds++;                    /* pre14: an AFMT rebuild is a corrector event (§4.4) */
            /* memcap F1: escalate the breaker interval — a rebuild within 120s of the previous
             * doubles the backoff (2s start, 60s cap); 120s of quiet resets it. Covers ACHOP
             * too (its teardown re-forms through this same confirm site). */
            {
                int64_t brb = av_gettime_relative();
                if (a->afmt_last_rb_us && brb - a->afmt_last_rb_us < 120000000)
                    a->afmt_backoff_us = a->afmt_backoff_us ? FFMIN(a->afmt_backoff_us * 2, 60000000)
                                                            : 2000000;
                else
                    a->afmt_backoff_us = 0;
                a->afmt_last_rb_us = brb;
                a->afmt_next_ok_us = a->afmt_backoff_us ? brb + a->afmt_backoff_us : 0;
                a->afmt_bk_dropped = 0;
                if (a->afmt_backoff_us)
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-AFMT] a%d(in%d) rebuild #%d within 120s of the last — breaker "
                           "backoff now %.0fs (flap-storm churn bound; PTV_NO_AFMT_BREAKER=1 "
                           "disables)\n",
                           a->dbg_k, a->dbg_in, a->afmt_rebuilds, a->afmt_backoff_us / 1e6);
            }
            if (a->use_fg) {                       /* pre17: the rebuilt graph takes the fixed input, the converter the source */
                ptv_graph_in_params(a);
                if (ptv_cv_setup(a, frame->sample_rate, frame->format, &frame->ch_layout) < 0)
                    av_log(NULL, AV_LOG_ERROR, "[PTV-AFMT] a%d(in%d) converter for the rebuilt path failed\n",
                           a->dbg_k, a->dbg_in);
            } else {
                a->fg_in_rate = a->dec->sample_rate;   /* the path is now configured for these */
                a->fg_in_fmt  = a->dec->sample_fmt;
                av_channel_layout_uninit(&a->fg_in_chl);
                av_channel_layout_copy(&a->fg_in_chl, &a->dec->ch_layout);
            }
            av_channel_layout_uninit(&a->afmt_pending_chl);
            a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
            /* 1.0.1-pre20: the rebuilt path gets a BIRTH-EQUIVALENT base derived from the
             * current house mapping instead of the carried one (see ptv_rebuild_reanchor).
             * Runs for ACHOP-triggered rebuilds too — they complete through this same site. */
            ptv_rebuild_reanchor(a, a->afmt_qn ? a->afmt_q[0] : frame);   /* pre9.6: the first held frame starts the new path */
            /* 1.0.1-pre26 ROOT-CAUSE FIX (the NBS/CORELINK backward-label mux killer,
             * live 2026-07-25, 8+ crashes/24h; fixture-reproduced): a BACKWARD door step
             * left in the labels (a flush-routed negative mismatch / pad round-trip — the
             * exp_hit "aresample drops content" remedies) opens a window of |step| seconds
             * where the OLD graph's emitted labels lead its input labels while swr realizes
             * the drop. A rebuild landing INSIDE that window destroys the swr state that
             * carried the high-water mark: the NEW graph re-seeds its output at the door
             * labels — up to |step| BEHIND the last emitted label — and the next emission
             * is a backward DTS into the muxer (measured: door −2773ms → rebuild → emission
             * −2667ms → libfdk "Queue input is backward in time" → mpegts EINVAL → all
             * rungs die). ptv_rebuild_reanchor keeps the DOOR labels continuous (correct)
             * but never protected the EMISSION side. Arm the existing pre20 monotonic
             * emission guard at EVERY rebuild (unconditionally — also when the re-anchor
             * itself gated off): it drops post-rebuild frames until the first label that
             * exceeds the last emitted one, realizing the interrupted drop's REMAINDER as
             * the same content skip the old graph was mid-way through — label-only, no
             * re-stamp, releases with the existing [PTV-ANCHOR] guard line. The mv follow
             * path is excluded inside the guard itself (its af_last_out clamp owns this). */
            a->reanch_mono = 1;
            a->reanch_mono_dropped = 0;
            /* 1.0.1-pre19 #42 hardening: the rebuild above configured the path from a->dec,
             * but THIS frame (the confirming one) falls through to be fed below. During a
             * broken-AAC phase the decoder context can diverge from the frames it emitted
             * earlier (per-frame reconfig / queued pre-h0 replay), and swr_convert reads as
             * many planes as ITS configured input layout has — a frame with fewer channels
             * than a->dec's layout means reads through nonexistent plane pointers. If the
             * frame no longer matches the just-configured input params, drop it; the next
             * frame re-enters the normal AFMT detection. */
            if (!a->use_fg &&                      /* pre17: a graph takes it through the converter */
                (frame->sample_rate != a->fg_in_rate ||
                 frame->format      != a->fg_in_fmt  ||
                 av_channel_layout_compare(&frame->ch_layout, &a->fg_in_chl))) {
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-AFMT] a%d(in%d) decoder context diverged from the confirmed frame "
                       "params during rebuild — frame dropped, detection re-arms\n",
                       a->dbg_k, a->dbg_in);
                afmt_q_reset(a);
                return 0;
            }
            /* 2.0.0-pre9.6: the confirming frames were held, not dropped — feed them first, in order (they match the
             * new params by construction: same_pending) */
            if (a->afmt_qn) {
                int qi, qn = a->afmt_qn, qret = 0;
                AVFrame *q[PTV_AFMT_QMAX];
                memcpy(q, a->afmt_q, qn * sizeof(*q));
                a->afmt_qn = 0;
                /* a source gap AT the switch (audio_format_switch: 0.15 s of missing audio at the splice) is real missing
                 * content, but the rebuilt graph has no history for aresample to pad it from: fill it with silence on the
                 * source timeline (raw labels, ≤ 2 s), so the output stays continuous (T-083) */
                if (q[0]->pts != AV_NOPTS_VALUE && q[0]->sample_rate > 0 && raw_end0 != AV_NOPTS_VALUE) {
                    int64_t end = raw_end0;
                    int64_t gap = av_rescale_q(q[0]->pts, a->ist_tb, AV_TIME_BASE_Q) - end;
                    int64_t fdur = av_rescale(1024, 1000000, q[0]->sample_rate);
                    if (gap > fdur / 2 && gap <= 2000000) {
                        int64_t left = av_rescale(gap, q[0]->sample_rate, 1000000), at = end;
                        av_log(NULL, AV_LOG_WARNING, "[PTV-AFMT] a%d(in%d) source gap of %"PRId64" ms at the switch — "
                               "filled with silence (the rebuilt path cannot pad it)\n", a->dbg_k, a->dbg_in, gap / 1000);
                        while (left > 0 && qret >= 0) {
                            AVFrame *z = av_frame_alloc();
                            int n = left > 1024 ? 1024 : (int)left;
                            if (!z)
                                break;
                            z->nb_samples  = n;
                            z->format      = q[0]->format;
                            z->sample_rate = q[0]->sample_rate;
                            av_channel_layout_copy(&z->ch_layout, &q[0]->ch_layout);
                            if (av_frame_get_buffer(z, 0) >= 0) {
                                av_samples_set_silence(z->extended_data, 0, n, z->ch_layout.nb_channels, z->format);
                                z->pts = z->best_effort_timestamp = av_rescale_q(at, AV_TIME_BASE_Q, a->ist_tb);
                                qret = audio_feed(a, z);
                            }
                            av_frame_free(&z);
                            at   += av_rescale(n, 1000000, q[0]->sample_rate);
                            left -= n;
                        }
                    }
                }
                for (qi = 0; qi < qn; qi++) {
                    if (qret >= 0)
                        qret = audio_feed(a, q[qi]);
                    av_frame_free(&q[qi]);
                }
                if (qret < 0)
                    return qret;
            }
            if (a->use_fg)                         /* pre17: this frame, too, goes through the converter */
                return audio_feed(a, frame);
        }
    } else if (a->afmt_stable) {                   /* params returned to normal → transient filtered out */
        a->afmt_stable = 0;
        afmt_q_reset(a);
        av_channel_layout_uninit(&a->afmt_pending_chl);
        a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
    }

graph_in:
    if (a->use_fg) {
        /* -af: feed the graph; aresample async + loudness emit fixed-size frames
         * whose PTS already carries async's A/V correction — drain them straight
         * to the encoders. Common-mode A/V lock: add the video's house-vs-content
         * skew so aresample=async targets the HOUSE clock instead of the source. */
        /* Single-input (and PTV_NO_AUDIO_FOLLOW): nudge the graph input pts by house_skew so
         * aresample=async targets the house clock. Multiview audio-follow does NOT do this — it
         * feeds content-aligned input and applies the offset deterministically in the drain. */
        /* WUCR (W1 CORRECTED 2026-06-29): AVLOCK is KEPT ON under WUCR. Tracing the W1 failure showed the
         * house clock MUST dup-fill on source stall/corruption (never-stop), so video content lags by
         * house_skew and that lag RATCHETS (the decoder hands back sequential frames, never skipping to
         * catch up). That lag is a DIFFERENTIAL audio cannot ignore — AVLOCK (audio follows house_skew)
         * is the necessary coupling that keeps A/V matched through dups. AVLOCK was never the disease;
         * UNBOUNDED house_skew (free house clock) was — and W0's ρ servo bounds it, making AVLOCK
         * harmless + correct. Removing AVLOCK (the original W1) caused the −900ms desync on TruBLU. */
        /* [PTV-AGLUE] Audio label-step glue (v0.9.16.3, verdict rule corrected v0.9.16.4).
         * Measured disease (3-act step fixture + [PTV-ASTEP]/[PTV-AFLOW], 2026-07-05): VIDEO
         * label steps are structurally ERASED by the house clock (output is stamped by frame
         * count, so input video label jumps are invisible), but AUDIO label steps were silently
         * FOLLOWED by aresample=async — a forward step pads silence (audio-late), a BACKWARD
         * step drops content (audio-early: the AWE −9.5ms/h accumulator's sign; A/B-measured
         * −308ms permanent from one −300ms wire step). ZERO log lines either way.
         * Verdict rule, DIRECTION-ASYMMETRIC on purpose:
         *   BACKWARD step → RELABEL, erase into glue_off_us. Content cannot be negatively
         *   missing, so a backward label step is always a relabel; erasing matches the video
         *   side's structural erasure and closes the audio-early accumulator.
         *   FORWARD step → GAP, always: keep labels, aresample pads (the pre-glue behavior).
         *   v0.9.16.3 tried to erase wall-continuous forward steps as relabels — LIVE FAILURE
         *   within the hour (AWE 2026-07-05): its audio gaps are cut UPSTREAM, so the stream
         *   arrives flowing with no wall pause at our end; 4 events erased +983ms in 16s and
         *   put audio visibly in front of video. Wall-clock continuity is NOT usable evidence
         *   for forward steps; padding is faithful for real gaps and merely latency-neutral
         *   for a true forward relabel (audio-late by the step, which the source's own return
         *   step later cancels).
         * Steps above g_aglue_max_ms belong to the >1s discontinuity layer (demux_unwrap/
         * LAYERA) — log and stand aside. Detection runs on RAW labels BEFORE the AVLOCK
         * house_skew injection below, so LAYERA flushes / house_skew actuation never
         * masquerade as source steps.
         * 1.0.1-pre5 EXCEPTION (D1): a step the demux shared flush REGISTERED for this track
         * (ptv_pair_expect) is not a source relabel — it is the A-vs-V jump difference the
         * flush deliberately routed here. Backward-and-registered is APPLIED (labels kept,
         * aresample drops content to converge onto the source's post-event alignment), never
         * erased; unregistered steps keep the rules above unchanged. */
        {   /* 1.0.1-pre8 (d): refresh the self-shed window mark once per frame while quiet, so
             * the first annotated line after a shed counts only THIS window's sheds (two
             * relaxed loads; output discarded). */
            char snr[48];
            ptv_self_shed_note(a, snr, sizeof snr);
        }
        if (g_aglue_ms > 0 && frame->pts != AV_NOPTS_VALUE && !a->hold_feeding) {   /* pre6: fill is stamped at the door */
            int64_t raw_us = av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            int64_t now_wc = av_gettime_relative();
            /* rr23 (F5): EAGER seam-park expiry — checked per fed frame (this thread owns the
             * state, race-free), not only at the next above-cap step: if seams stop during the
             * park, the entry line still gets its matching expiry line. The conv= P suffix is
             * timestamp-derived by the builder, so the display never depended on this. */
            if (g_convcap && a->seam_park_until && now_wc >= a->seam_park_until) {
                a->seam_park_until = 0;
                if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO)
                    atomic_store_explicit(&g_conv_park_pub[a->dbg_k], 0, memory_order_relaxed);
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-CONV] a%d(in%d) seam-park expired — converging re-enabled\n",
                       a->dbg_k, a->dbg_in);
            }
            if (g_glueclass && a->nbs_fill_active && !a->nbs_feeding) {
                a->nbs_fill_active = 0;
                if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO && !a->hold_fill_us)
                    atomic_store_explicit(&g_afill_us[a->dbg_k], 0, memory_order_relaxed);
                a->nbs_last_wall_us = 0;
                a->nbs_carry_us     = 0;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-ADISC] a%d(in%d) real frames resumed — silence-fill released after "
                       "%d quanta; resume step classified below\n",
                       a->dbg_k, a->dbg_in, a->nbs_fills);
            }
            /* 2.1 T-409: the labels arrive decided (the timeline), no step verdicts here; still track the arrival
             * cadence — the silence fill's patience. At 0 every PES-burst pause after a return (150-300 ms) started a
             * fill and the rejoin buffer held the real frames ~1 s for a fresh picture (sync_flapping: 3-5 flashes
             * without their beep). */
            if (g_glueclass && a->glue_raw_last_us != AV_NOPTS_VALUE &&
                llabs(raw_us - (a->glue_raw_last_us + a->glue_raw_dur_us)) <= (int64_t)g_aglue_ms * 1000) {
                int64_t wg = now_wc - a->glue_wall_last_us;
                if (wg > 5000 && wg < 2000000)
                    a->glue_cad_us = a->glue_cad_us ? a->glue_cad_us + (wg - a->glue_cad_us) / 8 : wg;
            }
            if (a->hold_fill_us &&   /* 2.0.0-pre6: 2 s after video resumed, the rejoin's step verdicts are in */
                atomic_load_explicit(&g_src_fresh_wc, memory_order_relaxed) >= a->hold_fill_wc0) {
                if (!a->hold_fresh_wc)
                    a->hold_fresh_wc = now_wc;
                else if (now_wc - a->hold_fresh_wc > 2000000) {
                    if (a->hold_fill_us >= 3000000)
                        av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] a%d(in%d) hold fill ended: %.1f s of silence\n",
                               a->dbg_k, a->dbg_in, a->hold_fill_us / 1e6);
                    a->hold_fill_us  = 0;
                    a->hold_fresh_wc = 0;
                    if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO)
                        atomic_store_explicit(&g_afill_us[a->dbg_k], 0, memory_order_relaxed);
                }
            }
            a->glue_raw_last_us  = raw_us;
            a->glue_raw_dur_us   = frame->sample_rate > 0 ?
                av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0;
            a->glue_wall_last_us = now_wc;
            a->hf_hs_real  = (g_avlock && a->house_skew) ? *a->house_skew : 0;
            a->hf_dup_real = atomic_load_explicit(&g_dup_out_us, memory_order_relaxed);
            if (a->glue_off_us)
                frame->pts += av_rescale_q(a->glue_off_us, AV_TIME_BASE_Q, a->ist_tb);
        }
        if (a->vm_feeding && a->vm_out_us != AV_NOPTS_VALUE && frame->pts != AV_NOPTS_VALUE) {
            int64_t h0;
            pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
            if (h0 != AV_NOPTS_VALUE)               /* output = door - h0: exactly where the video put this content */
                frame->pts = av_rescale_q(a->vm_out_us + h0, AV_TIME_BASE_Q, a->ist_tb);
        } else if (g_avlock && a->house_skew && !(a->multiview && g_audio_follow) && frame->pts != AV_NOPTS_VALUE &&
            !a->hold_feeding) {
            int64_t sk = hs_door(a, frame, *a->house_skew);
            if (sk) frame->pts += av_rescale_q(sk, AV_TIME_BASE_Q, a->ist_tb);
        }
        /* 1.0.1-pre14 steer bus (§5 rule 1): the corrector's cumulative trim, injected at the
         * SAME single graph-door site as glue_off/house_skew — aresample=async
         * realizes it as bounded SOFT compensation (rate-clamped 2ms/s upstream in
         * rscorr_update; per-frame deltas ~43µs, three orders under min_hard_comp — the
         * ACOMP proxy below monitors the summed stream and is the click tripwire for any
         * mis-sized bus term). corr_us==0 (default-off / parked-at-zero) skips = byte-inert. */
        if (a->corr.corr_us && frame->pts != AV_NOPTS_VALUE && !a->hold_feeding)
            frame->pts += av_rescale_q(a->corr.corr_us, AV_TIME_BASE_Q, a->ist_tb);
        /* 2.0.0 T-093 part 2: the door span of a multiview fill phase — its silence plus the pad up to the first real
         * frame. The PLL's pairing skips frames of that span when they leave the graph (loudnorm holds ~3 s, so they
         * leave after the slot is back): paired against the returning picture's ring they read +19 s, the noise floor
         * rose to 4.5 s and the real −1 s offset at the return was never acquired (sync_gap_20). */
        if (a->multiview && g_audio_follow && frame->pts != AV_NOPTS_VALUE) {
            int64_t door_us = av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            if (a->nbs_feeding) {
                if (!a->mvf_open) { a->mvf_open = 1; a->mvf_lo_us = door_us; }
                a->mvf_hi_us = door_us + (frame->sample_rate > 0 ?
                               av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0);
            } else if (a->mvf_open) {
                a->mvf_open  = 0;
                a->mvf_hi_us = door_us;
            }
        }
        /* pre18: the overlap of a splice the converter took (armed at the switch for its replay) — a frame's head
         * that lies before the door's expected next label (the end of the frame fed before it) is dropped; the
         * rest keeps its labels (what aresample's hard compensation does above 30 ms, for any size up to 1 s).
         * A frame wholly inside the overlap is dropped. */
        if (a->cv_trim_left > 0 && frame->pts != AV_NOPTS_VALUE && frame->sample_rate > 0) {
            int64_t exp = a->acomp_exp_us != AV_NOPTS_VALUE ? a->acomp_exp_us : a->cv_trim_exp_us;
            int64_t ov  = exp - av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            int64_t k   = ov > 0 && ov <= 1000000 ? av_rescale(ov, frame->sample_rate, 1000000) : 0;
            a->cv_trim_left--;
            if (k >= frame->nb_samples) {
                a->cv_trim_us += av_rescale(frame->nb_samples, 1000000, frame->sample_rate);
                return 0;
            }
            if (k > 0) {
                if ((ret = av_frame_make_writable(frame)) < 0)
                    return ret;
                av_samples_copy(frame->extended_data, frame->extended_data, 0, (int)k,
                                frame->nb_samples - (int)k, frame->ch_layout.nb_channels, frame->format);
                frame->nb_samples -= (int)k;
                frame->pts += av_rescale_q(k, (AVRational){ 1, frame->sample_rate }, a->ist_tb);
                a->cv_trim_us += av_rescale(k, 1000000, frame->sample_rate);
                av_log(NULL, AV_LOG_INFO, "[PTV-AFMT] a%d(in%d) splice overlap %.1f ms trimmed at the switch "
                       "(labels kept; total %.1f ms)\n", a->dbg_k, a->dbg_in, ov / 1000.0, a->cv_trim_us / 1000.0);
            }
        }
        /* 2.0.0 T-356: the graph input never steps BACK unless a deliberate relabel reset the expectation (those paths
         * set acomp_exp_us = NOPTS). A video-only outage that reaches LOST: audio kept its own clock, then the rejoin
         * map moved house_skew −2.1 s and the audio's input stepped back by that — swr hard compensation and, at 50 /
         * 59.94, frames backward into the encoder: 8–54 AAC overlaps (vnull_40; loudnorm's lookahead had hidden it,
         * loudlive does not). What lies before the expected next label is dropped (labels kept: the audio follows the
         * video's new mapping, one forward cut instead of a replay); a step beyond 5 s is left as before */
        if (frame->pts != AV_NOPTS_VALUE && frame->sample_rate > 0 && a->acomp_exp_us != AV_NOPTS_VALUE) {
            int64_t ov = a->acomp_exp_us - av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            if (a->bk_trim_us && ov <= 0) {      /* the overlap ended on a frame boundary */
                av_log(NULL, AV_LOG_INFO, "[PTV-ATRIM] a%d(in%d) %.1f ms dropped in all\n",
                       a->dbg_k, a->dbg_in, a->bk_trim_us / 1000.0);
                a->bk_trim_us = 0;
            }
            /* a trim under way takes every remaining overlap (29.97: the last 15 ms reached the encoder).
             * 2.0.0 T-374: any overlap over 1 ms (label rounding stays below it), not only > 25 ms — the first real
             * frame after a hold fill landed 24 ms behind the fill's end on some runs (which tick the fill stops on is
             * a wall-clock phase), aresample carried it as an 18 ms step back and the encoder wrote one AAC overlap */
            if (ov > (a->bk_trim_us ? 0 : 1000) && ov <= 5000000) {
                int64_t k = av_rescale(ov, frame->sample_rate, 1000000);
                if (!a->bk_trim_us)
                    av_log(NULL, AV_LOG_WARNING, "[PTV-ATRIM] a%d(in%d) input stepped back %.0f ms — the overlap is "
                           "dropped at the graph door (labels kept)\n", a->dbg_k, a->dbg_in, ov / 1000.0);
                if (k >= frame->nb_samples) {
                    a->bk_trim_us += av_rescale(frame->nb_samples, 1000000, frame->sample_rate);
                    return 0;
                }
                if ((ret = av_frame_make_writable(frame)) < 0)
                    return ret;
                av_samples_copy(frame->extended_data, frame->extended_data, 0, (int)k,
                                frame->nb_samples - (int)k, frame->ch_layout.nb_channels, frame->format);
                frame->nb_samples -= (int)k;
                frame->pts += av_rescale_q(k, (AVRational){ 1, frame->sample_rate }, a->ist_tb);
                a->bk_trim_us += av_rescale(k, 1000000, frame->sample_rate);
                av_log(NULL, AV_LOG_INFO, "[PTV-ATRIM] a%d(in%d) %.1f ms dropped in all\n",
                       a->dbg_k, a->dbg_in, a->bk_trim_us / 1000.0);
                a->bk_trim_us = 0;
            }
        }
        /* 1.0.1-pre3 [PTV-ACOMP] — swr hard-compensation proxy (always-on, log rate-limited to
         * ~1/10s per track). aresample=async realizes a graph-input pts step beyond
         * min_hard_comp (~30ms in the production chain) as an INSTANTANEOUS sample insert/drop
         * — a click risk with zero log lines of its own. Detect it at the graph door: track the
         * expected next input pts (last pts + frame duration) and flag any ~25ms+ instantaneous
         * deviation of the stream the resampler actually sees (post-AGLUE, post-AVLOCK/steer —
         * a house_skew or LAYERA step that reaches swr counts, by design). */
        if (frame->pts != AV_NOPTS_VALUE) {
            int64_t inus = av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            if (a->acomp_exp_us != AV_NOPTS_VALUE && llabs(inus - a->acomp_exp_us) > 25000) {
                int64_t nowc = av_gettime_relative();
                a->acomp_cnt++;
                if (a->acomp_log_us == 0 || nowc - a->acomp_log_us >= 10000000) {
                    a->acomp_log_us = nowc;
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-ACOMP] a%d(in%d) input pts step %+"PRId64"ms — swr hard compensation likely (click risk) (total %"PRId64")\n",
                           a->dbg_k, a->dbg_in, (inus - a->acomp_exp_us) / 1000, a->acomp_cnt);
                }
            }
            a->acomp_exp_us = inus + (frame->sample_rate > 0 ?
                av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0);
        }
        if (g_diag && frame->pts != AV_NOPTS_VALUE) {   /* [PTV-ASTEP] pre-graph label-step detector */
            int64_t inus = av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            if (a->dbg_in_us && llabs(inus - (a->dbg_in_us + a->dbg_in_dur_us)) > 5000) {
                char sn[48];
                av_log(NULL, AV_LOG_WARNING, "[PTV-ASTEP] in-pts step %+lldms (in=%lldus expect=%lldus)%s\n",
                       (long long)((inus - a->dbg_in_us - a->dbg_in_dur_us) / 1000),
                       (long long)inus, (long long)(a->dbg_in_us + a->dbg_in_dur_us),
                       ptv_self_shed_note(a, sn, sizeof sn));
            }
            a->dbg_in_us = inus;
            a->dbg_in_dur_us = frame->sample_rate > 0 ?
                av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0;
            a->dbg_in_samp += frame->nb_samples;
        }
        if ((ret = av_buffersrc_add_frame(a->afsrc, frame)) < 0)
            return ret;
        return audio_drain_fg(a);
    }
    if (!a->swr)                       /* v0.9.17.1: path dead (init failed, AFMT retry pending) — drop, never deref NULL */
        return 0;
    out_max = av_rescale_rnd(swr_get_delay(a->swr, frame->sample_rate) + frame->nb_samples,
                             a->out_rate, frame->sample_rate, AV_ROUND_UP);
    if ((ret = av_samples_alloc_array_and_samples(&out, NULL, a->out_chl.nb_channels,
                                                  out_max, a->out_sfmt, 0)) < 0)
        return ret;
    got = swr_convert(a->swr, out, out_max,
                      (const uint8_t **)frame->extended_data, frame->nb_samples);
    if (got > 0)
        av_audio_fifo_write(a->fifo, (void **)out, got);
    if (out) { av_freep(&out[0]); av_freep(&out); }
    if (got < 0) return got;
    return audio_drain_fifo(a);
}

/* Anchor (on the first kept frame) then feed `frame`, given a known h0. Drops frames
 * whose content precedes h0. Used both for live frames and the replayed pre-h0 buffer. */
/* 2.0.2 T-386: a mosaic slot not live at the start — its track exists from the first tick and carries silence on the
 * house axis (sample 0 = the first tick, where a live slot's audio is anchored too) up to the compositor's current
 * tick, straight into the encoders (no source, so no graph). Without it the PID would be declared but empty: sync_check
 * DEAD_AUDIO_TRACK = a restart loop. At most 2 s per call (a late first call catches up over several). */
static int slot_pump(AudioState *a)
{
    int64_t target, lim;
    int ret = 0;

    if (!a->slot_deferred || a->pts_set)
        return 0;
    target = av_rescale(atomic_load_explicit(&g_mv_house_us, memory_order_relaxed), a->out_rate, 1000000);
    lim    = a->pump_next + 2 * a->out_rate;
    while (ret >= 0 && a->pump_next + a->frame_size <= FFMIN(target, lim)) {
        AVFrame *f = av_frame_alloc();
        if (!f)
            return AVERROR(ENOMEM);
        f->nb_samples  = a->frame_size;
        f->format      = a->out_sfmt;
        f->sample_rate = a->out_rate;
        av_channel_layout_copy(&f->ch_layout, &a->out_chl);
        if ((ret = av_frame_get_buffer(f, 0)) >= 0) {
            av_samples_set_silence(f->extended_data, 0, f->nb_samples, f->ch_layout.nb_channels, f->format);
            f->pts = a->pump_next;
            a->pump_next += a->frame_size;
            ret = audio_encode_push(a, f);
            a->out_frames++;
        }
        av_frame_free(&f);
    }
    return ret;
}

/* 2.0.2 T-386: the slot bound — its real decoder replaces the stand-in (the converter in front of the fixed graph
 * input re-forms from the first clean frames if the source is not 48 kHz stereo: [PTV-AFMT]) */
static void slot_bind_audio(AudioState *a, AVPacket *pkt)
{
    PtvSlotBind *b = pkt->opaque;
    const AVCodec *c;
    AVCodecContext *nc = NULL;
    int ret;

    pkt->opaque = NULL;
    if (!b)
        return;
    if (!(c = avcodec_find_decoder(b->par->codec_id)) || !(nc = avcodec_alloc_context3(c)) ||
        avcodec_parameters_to_context(nc, b->par) < 0) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-BIND] a%d(in%d): no decoder for %s — the track stays silent\n",
               a->dbg_k, a->dbg_in, avcodec_get_name(b->par->codec_id));
        goto done;
    }
    nc->pkt_timebase = b->tb;
    ptv_adec_opts(nc);
    if ((ret = avcodec_open2(nc, c, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-BIND] a%d(in%d): open %s: %s — the track stays silent\n",
               a->dbg_k, a->dbg_in, c->name, av_err2str(ret));
        goto done;
    }
    avcodec_free_context(&a->dec);
    a->dec = nc;
    a->ist_tb = a->ist_pkt_tb = b->tb;
    if (a->ist_par)
        avcodec_parameters_copy(a->ist_par, b->par);
    av_log(NULL, AV_LOG_INFO, "[PTV-BIND] a%d(in%d) decoder %s %d Hz %dch open — silence until its first frame anchors\n",
           a->dbg_k, a->dbg_in, c->name, b->par->sample_rate, b->par->ch_layout.nb_channels);
    nc = NULL;
done:
    avcodec_free_context(&nc);
    avcodec_parameters_free(&b->par);
    av_free(b);
}

static int audio_anchor_and_feed(AudioState *a, AVFrame *frame, int64_t h0)
{
    int64_t ts = frame->best_effort_timestamp;
    if (ts == AV_NOPTS_VALUE) return 0;
    if (!a->pts_set) {
        int64_t house_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q) - h0;
        int64_t fill_us  = 0;
        if (house_us < 0) { a->anchor_drop_pre++; return 0; }   /* audio precedes video anchor: drop */
        if (a->slot_deferred) {
            /* 2.0.2 T-386: a joining slot — the silence covered house 0 .. pump_next; real audio continues from there:
             * a frame before it is dropped, a gap after it is filled (from the silence's end, not from house 0) */
            int64_t pump_us = av_rescale(a->pump_next, 1000000, a->out_rate);
            if (house_us < pump_us) { a->anchor_drop_pre++; return 0; }
            a->slot_deferred = 0;
            if (house_us > pump_us && frame->sample_rate > 0 && frame->nb_samples > 0) {
                /* continuous from the silence's end, the real audio at its own house time: ONE silence frame exactly
                 * the gap long (any length — the encoder FIFO repackages; whole AAC frames left the audio up to one
                 * frame early, measured 16 ms) and no timestamp step (output linearity, GOALS.md #1) */
                int64_t gap_us = house_us - pump_us, src_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q);
                int     ns     = (int)av_rescale(gap_us, frame->sample_rate, 1000000), ret = 0;
                a->next_pts = a->pump_next;
                a->pts_set  = 1;
                a->dbg_first_src = ts;
                av_log(NULL, AV_LOG_WARNING, "[PTV-ANCHOR] a%d(in%d) joined at house %.3f s (silence to %.3f s, gap %.1f ms "
                       "filled)\n", a->dbg_k, a->dbg_in, house_us / 1e6, pump_us / 1e6, gap_us / 1e3);
                if (ns > 0) {
                    AVFrame *sf = av_frame_alloc();
                    if (!sf) return AVERROR(ENOMEM);
                    sf->nb_samples = ns; sf->format = frame->format; sf->sample_rate = frame->sample_rate;
                    av_channel_layout_copy(&sf->ch_layout, &frame->ch_layout);
                    if ((ret = av_frame_get_buffer(sf, 0)) >= 0) {
                        av_samples_set_silence(sf->data, 0, sf->nb_samples, sf->ch_layout.nb_channels, sf->format);
                        sf->pts = sf->best_effort_timestamp = av_rescale_q(src_us - gap_us, AV_TIME_BASE_Q, a->ist_tb);
                        ret = audio_feed(a, sf);
                    }
                    av_frame_free(&sf);
                    if (ret < 0) return ret;
                }
                return audio_feed(a, frame);
            }
            a->next_pts = a->pump_next;
            a->pts_set  = 1;
            a->dbg_first_src = ts;
            av_log(NULL, AV_LOG_WARNING, "[PTV-ANCHOR] a%d(in%d) joined at house %.3f s (silence to %.3f s)\n",
                   a->dbg_k, a->dbg_in, house_us / 1e6, pump_us / 1e6);
            return audio_feed(a, frame);
        }
        /* 1.0.1 ANCHOR HEAD-FILL (PTV_NO_ANCHOR_HEADFILL=1 reverts): when the source's audio
         * HEAD is missing (first kept audio starts >200ms after h0, or the pre-h0 ring
         * overflowed so the kept head is not the true head), the track's first packet used
         * to sit at PTS = first_audio − h0 — PTS-coherent, but first-packet-MISALIGNED for
         * naive consumers (RAV mv 2026-07-07: +2058ms — the suspected app-visible
         * audio-early). Synthesize silence covering house 0 → first_audio−h0 IN THE SOURCE
         * DOMAIN (the first kept frame's rate/layout/format, labels stepping seamlessly into
         * the real head) and push it through the normal feed path, so the encoder emits
         * audio from ~PTS 0 and every downstream layer (graph, PLL, gates) sees an ordinary
         * continuous track. Capped at the pre-h0 ring's own time span (~5.5s default) — the
         * same bound the buffered-head path already lives under. */
        if (g_anchor_headfill && (a->anchor_drop_ring > 0 || house_us > 200000) &&
            frame->sample_rate > 0 && frame->nb_samples > 0) {
            int64_t dur_us = av_rescale(frame->nb_samples, 1000000, frame->sample_rate);
            int64_t cap_us = (int64_t)g_cp.aq_prehold * dur_us;   /* ring capacity in time */
            fill_us = FFMIN(house_us, cap_us);
        }
        a->next_pts = ptv_anchor_next_pts(a, house_us - fill_us);   /* pre20: shared birth formula */
        a->pts_set  = 1;
        a->dbg_first_src = ts - av_rescale_q(fill_us, AV_TIME_BASE_Q, a->ist_tb);
        /* [PTV-ANCHOR] (v0.9.16.3, always-on) — the birth A/V relationship this track is built on.
         * house_us = first kept audio content − h0 (first video frame): the input-side head skew
         * the whole run inherits. A large value here (with clean internals after) is the
         * Zimbo-class startup-structural offset — visible at birth, invisible to every drift
         * sensor. ring_dropped>0 means the pre-h0 buffer overflowed (audio led video by more
         * than the ring; the kept head is NOT the true source head). */
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ANCHOR] a%d(in%d) anchored: first_audio-h0=%+"PRId64"ms (h0=%"PRId64"ms) "
               "dropped_pre_h0=%d ring_dropped=%d\n",
               a->dbg_k, a->dbg_in, house_us / 1000, h0 / 1000,
               a->anchor_drop_pre, a->anchor_drop_ring);
        if (fill_us > 0) {
            int64_t dur_us = av_rescale(frame->nb_samples, 1000000, frame->sample_rate);
            int     nfill  = (int)((fill_us + dur_us - 1) / dur_us), i;
            int64_t src_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q);
            int     ret    = 0;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-ANCHOR] a%d(in%d) headfill %"PRId64"ms silence (house 0 → first kept audio; "
                   "%d frames, cap %"PRId64"ms)\n",
                   a->dbg_k, a->dbg_in, fill_us / 1000, nfill,
                   (int64_t)g_cp.aq_prehold * dur_us / 1000);
            for (i = nfill; i > 0 && ret >= 0; i--) {
                AVFrame *s = av_frame_alloc();
                if (!s) break;
                s->nb_samples  = frame->nb_samples;
                s->format      = frame->format;
                s->sample_rate = frame->sample_rate;
                av_channel_layout_copy(&s->ch_layout, &frame->ch_layout);
                if (av_frame_get_buffer(s, 0) >= 0) {
                    av_samples_set_silence(s->data, 0, s->nb_samples,
                                           s->ch_layout.nb_channels, s->format);
                    s->pts = s->best_effort_timestamp =
                        av_rescale_q(src_us - i * dur_us, AV_TIME_BASE_Q, a->ist_tb);
                    ret = audio_feed(a, s);
                }
                av_frame_free(&s);
            }
            if (ret < 0) return ret;
        }
    }
    return audio_feed(a, frame);
}

/* 2.0.0-pre9.6: peak |sample| of a decoded float frame (−1 = not a float format); non-finite → +inf */
static double afrm_peak(const AVFrame *f)
{
    int planar = av_sample_fmt_is_planar(f->format), ch = f->ch_layout.nb_channels;
    int np = planar ? ch : 1, ns = planar ? f->nb_samples : f->nb_samples * ch, p, i;
    enum AVSampleFormat pk = av_get_packed_sample_fmt(f->format);
    double peak = 0;
    if (pk != AV_SAMPLE_FMT_FLT && pk != AV_SAMPLE_FMT_DBL)
        return -1;
    for (p = 0; p < np && f->extended_data[p]; p++)
        for (i = 0; i < ns; i++) {
            double x = pk == AV_SAMPLE_FMT_FLT ? ((const float *)f->extended_data[p])[i]
                                               : ((const double *)f->extended_data[p])[i];
            if (!isfinite(x)) return INFINITY;
            if (fabs(x) > peak) peak = fabs(x);
        }
    return peak;
}

/* 2.0.1 T-087 relabel check. A backward audio label step is erased (the 0.9.16.4 rule: content cannot run backwards)
 * — at the demux absorber or at the graph door (AGLUE). The sensor reads A/V in the SOURCE's label domain (so it can
 * see an erase that broke the source's A/V, T-088), which makes a pure relabel (content continuous, labels lied) read
 * as a desync: measured −300 ms on sync_arelabel (the corrector then moved synced audio −292 ms for good) and −65 ms
 * on TruBLU on air (2026-10-16, content vs source: in sync). The audio itself tells them apart: after a real repeat the
 * S ms before the seam come again, after a relabel they do not. 10 ms RMS envelope of channel 0: the first S after the
 * seam (80 ms .. 1 s) against the S before it; the same window at other lags must not match as well, and a window
 * without structure (silence, a steady tone) gives no verdict. Relabel → the sensor follows the erase; repeat or no verdict → as before. A video edit
 * around the seam (a both-stream step, T-088) keeps the source-domain reading. PTV_NO_RLCHECK=1 reverts. */
static double rl_corr(const float *env, int64_t a0, int64_t b0, int n)
{
    double ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
    int i;
    for (i = 0; i < n; i++) {
        ma += env[(a0 + i) % PTV_RL_ENV];
        mb += env[(b0 + i) % PTV_RL_ENV];
    }
    ma /= n; mb /= n;
    for (i = 0; i < n; i++) {
        double x = env[(a0 + i) % PTV_RL_ENV] - ma, y = env[(b0 + i) % PTV_RL_ENV] - mb;
        sab += x * y; saa += x * x; sbb += y * y;
    }
    return saa > 0 && sbb > 0 ? sab / sqrt(saa * sbb) : 0;
}

static void rl_eval(AudioState *a, int64_t now)
{
    int W = 100;                                        /* up to 1 s of 10 ms bins, see below */
    int64_t sb, lo, base = a->rl_n - PTV_RL_ENV;
    double c_rep = -1, c_oth = -1, mean = 0, var = 0;
    const char *why = NULL;
    int i, d;
    static const int oth[] = { -50, -25, 25, 50 };

    if (!a->rl_pend || a->rl_n - a->rl_pos < W || now - a->rl_wc < 1200000)
        return;
    if (a->rl_pend == 1 && a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO &&
        atomic_load_explicit(&g_rl_undone[a->dbg_k], memory_order_relaxed) == a->rl_seq) {
        a->rl_pend = 0;                                 /* the demux reverted the erase (one corrupt timestamp) */
        return;
    }
    if (a->rl_ev_chg_wc && a->rl_ev_chg_wc >= a->rl_wc - (a->rl_pend == 2 ? 3000000 : 1200000))
        why = "the video was edited around the seam (both streams stepped)";
    sb = (a->rl_S_us + 5000) / 10000;
    /* a repeat is S long: only the first S after the seam can match the S before it (measured: a 341 ms repeat
     * against a 1 s window read corr 0.32 = "relabel" — wrong, the corrector then never fixed the +342 ms) */
    W  = av_clip(sb, 8, 100);
    lo = a->rl_pos - sb - 50;                           /* oldest bin the comparisons need */
    if (!why && (sb < 3 || lo < FFMAX(base, 0)))
        why = "step outside the check's range";
    /* 2.0.1 gate (vt-as/sync_aswitch_1): a stereo -> mono switch at the seam made channel 0's envelope before it unlike
     * the mono after it (corr 0.30) — read as a relabel, the sensor followed a −192 ms erase that broke A/V and nothing
     * corrected it (+192 ms for good; pre22k slewed it back). Across a format change the audio cannot vouch either way */
    if (!why && a->rl_fmt_have && a->rl_fmt_n >= lo && a->rl_fmt_n <= a->rl_pos + W)
        why = "the audio format changed at the seam (a splice)";
    if (!why) {
        for (i = 0; i < W; i++)
            mean += a->rl_env[(a->rl_pos + i) % PTV_RL_ENV];
        mean /= W;
        for (i = 0; i < W; i++) {
            double x = a->rl_env[(a->rl_pos + i) % PTV_RL_ENV] - mean;
            var += x * x;
        }
        if (mean < 1e-4)
            why = "no structure in the audio (silence)";
        else if (sqrt(var / W) < 0.05 * mean)
            why = "no structure in the audio (a steady tone)";
    }
    if (!why) {
        for (d = -2; d <= 2; d++)
            c_rep = FFMAX(c_rep, rl_corr(a->rl_env, a->rl_pos, a->rl_pos - sb + d, W));
        for (i = 0; i < FF_ARRAY_ELEMS(oth); i++)
            if (a->rl_pos - sb + oth[i] >= lo && a->rl_pos - sb + oth[i] + W <= a->rl_pos)
                c_oth = FFMAX(c_oth, rl_corr(a->rl_env, a->rl_pos, a->rl_pos - sb + oth[i], W));
    }
    /* 2.0.2 T-395: an audio-only erase over SILENCE has no evidence either way, and the no-verdict path ends at the
     * source's A/V anyway (the corrector steers back to it after its dwell). On a restless input that dwell never
     * completes: glo-1 Atlântico_TV 2026-10-19 22:18, a −76 ms audio-only absorb after a rejoin, sensor −94…−131 ms
     * for 11 h (640 disarm/re-arm cycles); Fé_TV 2026-10-18 the same shape. Undo the erase instead — S of audio is
     * dropped at the door (a whole-step skip), only while the audio there is silent too (inaudible), within 10 s;
     * single input (the live cases; a mosaic slot's door is shared). A steady tone keeps the old path (owner
     * 2026-10-20: silence only). A both-stream step never gets here (its 'video was edited' reason comes first). */
    if (why && g_rlundo && !a->multiview && a->rl_S_us > 0 && !strcmp(why, "no structure in the audio (silence)")) {
        a->rl_undo_us = a->rl_S_us;
        a->rl_undo_dl = now + 10000000;
        av_log(NULL, AV_LOG_INFO, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase: no verdict — %s; the erase will be "
               "undone at the next silent audio (PTV_NO_RLUNDO reverts)\n", a->dbg_k, a->dbg_in, a->rl_S_us / 1000, why);
    } else if (why)
        av_log(NULL, AV_LOG_INFO, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase: no verdict — %s; the sensor keeps the "
               "source's A/V\n", a->dbg_k, a->dbg_in, a->rl_S_us / 1000, why);
    /* 2.0.2 T-391: a RELABEL verdict blinds the sensor to the erase for good, so it needs clear evidence. A real
     * ~100 ms repeat (glo-1 Multivisão, HLS segment overlap re-encoded: content cross-correlation proved the erase left
     * the audio +100 ms late) scored 0.47 / 0.51 / 0.58 on its 9-10 envelope bins and was called a relabel at < 0.5 —
     * uncorrected for hours, ~+400 ms on air after several. Clear relabels score <= 0.21 (synthetic -0.17, fleet
     * -0.53 .. 0.21); between 0.3 and 0.8 = no verdict: the sensor keeps the source's A/V and the corrector decides. */
    else if (c_rep < g_rl_relabel_max) {
        a->rl_neutral_us += a->rl_S_us;
        av_log(NULL, AV_LOG_WARNING, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase was a RELABEL: the audio after the seam "
               "does not repeat the %"PRId64" ms before it (corr %.2f) — the sensor follows the erase "
               "(PTV_NO_RLCHECK reverts)\n", a->dbg_k, a->dbg_in, a->rl_S_us / 1000, a->rl_S_us / 1000, c_rep);
    } else if (c_rep >= 0.8 && c_rep - c_oth >= 0.3)
        av_log(NULL, AV_LOG_WARNING, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase: the audio REPEATS (corr %.2f, other "
               "lags %.2f) — a real backward step; the sensor keeps the source's A/V for the corrector\n",
               a->dbg_k, a->dbg_in, a->rl_S_us / 1000, c_rep, c_oth);
    else
        av_log(NULL, AV_LOG_INFO, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase: no verdict (corr %.2f, other lags %.2f); "
               "the sensor keeps the source's A/V\n", a->dbg_k, a->dbg_in, a->rl_S_us / 1000, c_rep, c_oth);
    a->rl_pend = 0;
}

/* T-087: feed one decoded frame into the envelope ring; arm a seam the demux flagged; evaluate a pending one */
static void rl_feed(AudioState *a, const AVFrame *f)
{
    int64_t now = av_gettime_relative(), ev;
    int i, n = f->nb_samples, ch = f->ch_layout.nb_channels, planar;
    enum AVSampleFormat fmt = f->format;

    if (!g_rlcheck || n <= 0 || ch <= 0 || f->sample_rate <= 0)
        return;
    planar = av_sample_fmt_is_planar(fmt);
    fmt = av_get_packed_sample_fmt(fmt);
    if (fmt != AV_SAMPLE_FMT_FLT && fmt != AV_SAMPLE_FMT_S16 && fmt != AV_SAMPLE_FMT_S32)
        return;
    if (!a->rl_bin || a->rl_bin != f->sample_rate / 100) {
        a->rl_bin   = f->sample_rate / 100;
        a->rl_undo_us = 0;
        a->rl_acc   = 0;
        a->rl_acc_n = 0;
        a->rl_pend  = 0;
        a->rl_ev_last = a->dbg_in >= 0 && a->dbg_in < PTV_MAX_INPUT ?
                        atomic_load_explicit(&g_rsx.ev_us[a->dbg_in], memory_order_relaxed) : 0;
    }
    ev = a->dbg_in >= 0 && a->dbg_in < PTV_MAX_INPUT ?
         atomic_load_explicit(&g_rsx.ev_us[a->dbg_in], memory_order_relaxed) : 0;
    if (ev != a->rl_ev_last) {
        a->rl_ev_last   = ev;
        a->rl_ev_chg_wc = now;
    }
    if (a->rl_armed && !a->rl_pend && a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO) {
        a->rl_pend  = 1;
        a->rl_pos   = a->rl_n;
        a->rl_S_us  = atomic_load_explicit(&g_rl_step_us[a->dbg_k], memory_order_relaxed);
        a->rl_wc    = atomic_load_explicit(&g_rl_wc[a->dbg_k], memory_order_relaxed);
        a->rl_seq   = atomic_load_explicit(&g_rl_seq[a->dbg_k], memory_order_relaxed);
    }
    a->rl_armed = 0;
    if (a->rl_ch && ch != a->rl_ch) {                   /* a format switch (splice): its envelopes do not compare */
        a->rl_fmt_n    = a->rl_n;
        a->rl_fmt_have = 1;
    }
    a->rl_ch = ch;
    for (i = 0; i < n; i++) {
        double v;
        if (fmt == AV_SAMPLE_FMT_FLT)
            v = planar ? ((const float *)f->extended_data[0])[i] : ((const float *)f->data[0])[i * ch];
        else if (fmt == AV_SAMPLE_FMT_S16)
            v = (planar ? ((const int16_t *)f->extended_data[0])[i] : ((const int16_t *)f->data[0])[i * ch]) / 32768.0;
        else
            v = (planar ? ((const int32_t *)f->extended_data[0])[i] : ((const int32_t *)f->data[0])[i * ch]) / 2147483648.0;
        a->rl_acc += v * v;
        if (++a->rl_acc_n >= a->rl_bin) {
            a->rl_env[a->rl_n % PTV_RL_ENV] = sqrt(a->rl_acc / a->rl_acc_n);
            a->rl_n++;
            a->rl_acc = 0;
            a->rl_acc_n = 0;
        }
    }
    rl_eval(a, now);
    if (a->rl_undo_us) {                                /* 2.0.2 T-395: apply a scheduled undo at silent audio */
        int64_t k, kb = av_clip((a->rl_undo_us + 5000) / 10000 + 5, 8, 100);
        double m = 0;
        if (now > a->rl_undo_dl) {
            av_log(NULL, AV_LOG_INFO, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase stays: no silent audio within 10 s — "
                   "the sensor keeps the source's A/V\n", a->dbg_k, a->dbg_in, a->rl_undo_us / 1000);
            a->rl_undo_us = 0;
        } else if (a->rl_n >= kb) {
            for (k = a->rl_n - kb; k < a->rl_n; k++)
                m += a->rl_env[k % PTV_RL_ENV];
            if (m / kb < 1e-4) {
                ptv_label_step(a, -a->rl_undo_us, now);
                av_log(NULL, AV_LOG_WARNING, "[PTV-RELABEL] a%d(in%d) -%"PRId64"ms erase UNDONE: silent audio at the seam and "
                       "now, no evidence for the erase — %"PRId64" ms of silence dropped, the audio keeps the source's A/V "
                       "(glue %+"PRId64"ms; PTV_NO_RLUNDO reverts)\n", a->dbg_k, a->dbg_in, a->rl_undo_us / 1000,
                       a->rl_undo_us / 1000, a->glue_off_us / 1000);
                a->rl_undo_us = 0;
            }
        }
    }
}

/* 2.1 T-411/T-414 (mosaic): the slot's sound is placed up to its bound — tl_alim_us, the shown picture + the primed
 * margin + 1 s while the cell advances (the compositor publishes it; INT64_MAX = none: single input, a held or dead
 * cell) — and the decoded frames past it PARK, released in order as the picture catches up. Placed up to the decode
 * front, the sound put every step's boundary past the pictures waiting (a decoder catching up after the compose, a
 * burst: the slot re-locked and never committed, T-411) and a step moved picture and sound apart by what was placed
 * ahead. Bounded: 30 s of content or PTV_APARK frames release the oldest regardless. */
static int apark_out(AudioState *a, AVFrame *f)     /* the frame goes to its place: anchored, then fed */
{
    int64_t h0;
    if (a->pts_set) {
        if (f->best_effort_timestamp == AV_NOPTS_VALUE) return 0;   /* 1.0.1: un-stamped = cannot be anchored */
        a->dbg_last_src = f->best_effort_timestamp;                 /* probe: latest source audio pts */
        return audio_feed(a, f);
    }
    pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
    if (h0 == AV_NOPTS_VALUE) return 0;
    return audio_anchor_and_feed(a, f, h0);
}

static int apark_release(AudioState *a, int all)
{
    int ret = 0;
    while (a->apark_n > 0) {
        AVFrame *f = a->apark[a->apark_h];
        if (!all) {
            AVFrame *t   = a->apark[(a->apark_h + a->apark_n - 1) % PTV_APARK];
            int64_t  lim = atomic_load_explicit(a->tl_alim, memory_order_relaxed);
            int64_t  src = av_rescale_q(f->best_effort_timestamp, a->ist_tb, AV_TIME_BASE_Q);
            int64_t  spn = av_rescale_q(t->best_effort_timestamp, a->ist_tb, AV_TIME_BASE_Q) - src;
            if (src > lim && spn < 30000000 && a->apark_n < PTV_APARK)
                break;
        }
        a->apark_h = (a->apark_h + 1) % PTV_APARK;
        a->apark_n--;
        if (ret >= 0) ret = apark_out(a, f);
        av_frame_free(&f);
    }
    return ret;
}

static int apark_feed(AudioState *a, AVFrame *f)
{
    int64_t ts = f->best_effort_timestamp;
    if (a->tl_alim && ts != AV_NOPTS_VALUE &&
        (a->apark_n > 0 ||                           /* frames wait already: in order behind them */
         av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q) > atomic_load_explicit(a->tl_alim, memory_order_relaxed))) {
        AVFrame *c = av_frame_clone(f);
        int ret;
        if (!c) return AVERROR(ENOMEM);
        if (a->apark_n >= PTV_APARK) {               /* full: the oldest goes out */
            AVFrame *o = a->apark[a->apark_h];
            a->apark_h = (a->apark_h + 1) % PTV_APARK;
            a->apark_n--;
            ret = apark_out(a, o);
            av_frame_free(&o);
            if (ret < 0) { av_frame_free(&c); return ret; }
        }
        a->apark[(a->apark_h + a->apark_n) % PTV_APARK] = c;
        a->apark_n++;
        return apark_release(a, 0);
    }
    return apark_out(a, f);
}

static int audio_push(AudioState *a, AVFrame *frame)
{
    int64_t ts = frame->best_effort_timestamp;
    int ret = 0, i;

    a->in_frames++;

    /* 2.0.0-pre9.6: a frame with IMPOSSIBLE samples is damage, not content. Bit flips inside a valid AAC
     * packet can decode without an error into samples tens of thousands of times full scale (+93 dBFS
     * measured, sync_ac3_bitflip_100 s1); loudnorm's live gain follows the INTEGRATED loudness, which one such
     * frame raises for the rest of the programme → the track stayed silent (−103 dB) until restart — the
     * fleet's default -af (plain ffmpeg + loudnorm does the same). Above +18 dBFS (8.0; real decoded audio
     * overshoots full scale by a few dB at most) or non-finite: mute the frame (timing kept, the damage shows
     * as silence) and count it as damaged audio for the glue rules. */
    if (frame->nb_samples > 0) {
        double pk = afrm_peak(frame);
        if (pk > 8.0) {
            int64_t now = av_gettime_relative();
            av_samples_set_silence(frame->extended_data, 0, frame->nb_samples,
                                   frame->ch_layout.nb_channels, frame->format);
            a->dmg_wc = now;
            a->imposs_n++;
            if (!a->imposs_log_wc || now - a->imposs_log_wc > 10000000) {
                a->imposs_log_wc = now;
                av_log(NULL, AV_LOG_WARNING, "[PTV-ADEC] a%d(in%d) decoded frame with impossible samples (peak %s%.1f dBFS) "
                       "— muted as damage, kept out of the -af chain (total %"PRId64")\n", a->dbg_k, a->dbg_in,
                       isfinite(pk) ? "+" : "", isfinite(pk) ? 20 * log10(pk) : INFINITY, a->imposs_n);
            }
        }
    }

    /* Audio anchors to the FIRST VIDEO frame (h0, set by the video decode thread) so
     * A/V share one origin. While h0 is unset (the slot's video is still acquiring its
     * first frame) the audio is BUFFERED, not dropped: dropping it made that slot's
     * audio start late by the whole h0-acquire delay (up to ~1s on the box) — the
     * per-slot "audio delayed" desync. Once h0 is known we replay the buffer, keeping
     * content >= h0 and dropping the lead. Bounded ring so a never-arriving video can't
     * grow it unboundedly. */
    if (!a->pts_set) {
        int64_t h0;
        pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
        if (h0 == AV_NOPTS_VALUE) {
            if (ts != AV_NOPTS_VALUE) {
                AVFrame *c = av_frame_clone(frame);
                if (c) {
                    if (a->aq_npending >= g_cp.aq_prehold) {       /* ring (aq_prehold; 256 default = byte-identical, PTV_AQ_PREROLL deep): drop oldest */
                        a->anchor_drop_ring++;
                        av_frame_free(&a->aq_pending[0]);
                        memmove(a->aq_pending, a->aq_pending + 1,
                                (g_cp.aq_prehold - 1) * sizeof(*a->aq_pending));
                        a->aq_npending = g_cp.aq_prehold - 1;
                    }
                    a->aq_pending[a->aq_npending++] = c;
                }
            }
            return 0;
        }
        for (i = 0; i < a->aq_npending; i++) {        /* h0 known: replay buffered head */
            if (ret >= 0) ret = apark_feed(a, a->aq_pending[i]);
            av_frame_free(&a->aq_pending[i]);
        }
        a->aq_npending = 0;
        if (ret < 0) return ret;
        return apark_feed(a, frame);
    }
    /* 1.0.1: an un-stamped frame (e.g. a garbage-tail decode next to a tolerated [PTV-ADEC] error) cannot be
     * content-anchored; encoding it hands the muxer a timestamp-less packet = EINVAL mux wedge. Dropped (apark_out) —
     * same rule the pre-anchor path applies. */
    return apark_feed(a, frame);
}

/* 1.0.1 [PTV-ADEC] — tolerate a hard audio decode error: count it, log a rate-limited
 * WARNING (AGLUE-style: 10 lines / 10s window, then one summary as the window rolls),
 * drop the undecodable data and keep the thread alive. Before this, a hard error from
 * avcodec_receive_frame was `goto done` = SILENT PERMANENT track death (Pure Flix
 * 2026-07-08: one corrupt-PCE AAC event killed the track for 14h; video survives
 * identical storms via concealment). send_packet hard errors were already swallowed —
 * they now share this counter/log so a decode-error storm is visible either way. */
static void adec_error(AudioState *a, int err)
{
    int64_t now = av_gettime_relative();
    a->dec_errs++;
    a->dmg_wc = now;                   /* 2.0.0-pre9.3 (T-080 A2): damaged audio */
    a->dec_ts_carry = AV_NOPTS_VALUE;  /* pre19.1 [PTV-ASTAMP]: never extrapolate across a decode error —
                                        * a garbage-tail NOPTS frame next to a tolerated error keeps being
                                        * dropped (the rule the audio_push NOPTS guard was added for) */
    if (now - a->decerr_win_us >= 10000000) {
        if (a->decerr_supp)
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-ADEC] a%d(in%d) %d more decode errors suppressed in last 10s (total %"PRId64")\n",
                   a->dbg_k, a->dbg_in, a->decerr_supp, a->dec_errs);
        a->decerr_win_us = now;
        a->decerr_win_n  = 0;
        a->decerr_supp   = 0;
    }
    if (a->decerr_win_n < 10) {
        a->decerr_win_n++;
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ADEC] a%d(in%d) decode error (%s) — dropped, track continues (total %"PRId64")\n",
               a->dbg_k, a->dbg_in, av_err2str(err), a->dec_errs);
    } else
        a->decerr_supp++;
}

#define PTV_ADECWD_US 45000000   /* decode-death watchdog: packets arriving but zero frames for 45s → reopen */

/* 1.0.1 [PTV-ADECWD] — decode-death watchdog: the decoder has produced NOTHING for 45s
 * while packets kept arriving = it is wedged in a state error tolerance alone cannot
 * clear (Pure Flix corrupt-PCE class). Reconstruct it exactly as transcode() setup did
 * (codecpar → context, pkt_timebase, open) and swap it in. The anchor/pts_set state is
 * deliberately PRESERVED — this is mid-run recovery, the track's timeline continues and
 * aresample absorbs the dead span as it does for a source gap. If the reopened decoder
 * emits frames at different params than the configured path, the [PTV-AFMT]
 * hysteresis+rebuild in audio_feed re-configures downstream (same machinery as a source
 * format change). Swap only on successful open, so a failed reopen leaves the old
 * context in place (never a NULL a->dec) and retries a window later. */
/* 1.0.1-pre19 #38: enable the lavc AAC tolerant channel-element allocation BY NAME on the
 * decoder's private context — no compile-time dependency: a build against a lavc without
 * the option (patch 0004 not applied) simply doesn't find it and decodes strictly, the
 * pre-#38 behavior. With it, broken broadcast phases (Azorse: 7.1-signalled AAC carrying a
 * CPE-first element sequence, strict lavc rejects every frame with "channel element 1.0 is
 * not allocated" while VLC/FAAD play it) decode into real PCM. Harmless on non-AAC audio
 * decoders (option not found). Call before avcodec_open2 (read per-frame either way). */
void ptv_adec_opts(AVCodecContext *dec)
{
    if (!g_tolerant_dec)               /* 1.0.1-pre19.1: PTV_NO_TOLERANT_DEC=1 = strict lavc (pre-#38) */
        return;
    if (dec && dec->priv_data)
        av_opt_set(dec->priv_data, "tolerant_ch_alloc", "1", 0);
}

/* 1.0.1-pre19.1 — avformat_find_stream_info with the #38 tolerance on the AUDIO probe
 * decoders. lavf probes streams with its OWN internal decoders, which ptv_adec_opts never
 * touched: on a source opened DURING a broken-AAC phase (Azorse 7.1) the strict probe
 * decoder rejected every frame ("channel element 1.0 is not allocated" ×294 at open) →
 * "Could not find codec parameters" → sample_rate 0 → avformat could not stamp the
 * parser-split ADTS frames 2..6 of each PES (no duration to extrapolate cur_dts with) →
 * 5 of 6 decoded frames arrived NOPTS and were dropped = the 107ms/128ms ticking. With the
 * probe as tolerant as the runtime decoder, params resolve and every split frame is stamped
 * upstream — the [PTV-ASTAMP] extrapolation in audio_thread remains as the backstop for any
 * residual un-stamped class.
 * The option goes into EVERY stream's dict, not just audio: lavf's try_decode_frame is handed
 * options[i] with i = the (stale) "first stream still missing parameters" index, NOT the
 * packet's stream index (demux.c:2937) — measured on the Azorse fixture: the aac probe
 * decoder opened with stream 0's (video's) dict and stayed strict when the option was set
 * audio-only. Setting it everywhere is safe: option lookups that miss just leave the entry in
 * the dict (no open failure, non-AAC decoders unaffected; on a stock lavc the entry is simply
 * never consumed). Streams discovered DURING find_stream_info keep the strict default. */
/* 2.0.1 T-358: the input is opened with lavf's fix_teletext_pts off (DVB subtitles keep their own PES timestamps
 * through a video-only outage, see the open). lavf's comment says some sources' TELETEXT timestamps cannot be
 * trusted, and production copies source teletext: an input that carries a teletext PID keeps lavf's fix. */
static void ptv_teletext_fix(AVFormatContext *ic)
{
    unsigned i;
    for (i = 0; i < ic->nb_streams; i++)
        if (ic->streams[i]->codecpar->codec_id == AV_CODEC_ID_DVB_TELETEXT) {
            av_opt_set_int(ic, "fix_teletext_pts", 1, AV_OPT_SEARCH_CHILDREN);
            return;
        }
}

int ptv_find_stream_info(AVFormatContext *ic)
{
    AVDictionary **opts;
    unsigned i, n = ic->nb_streams;
    int r;

    if (!g_adts_split || !g_tolerant_dec || !n) {
        r = avformat_find_stream_info(ic, NULL);
        ptv_teletext_fix(ic);
        return r;
    }
    opts = av_calloc(n, sizeof(*opts));
    if (!opts) {
        r = avformat_find_stream_info(ic, NULL);
        ptv_teletext_fix(ic);
        return r;
    }
    for (i = 0; i < n; i++)
        av_dict_set(&opts[i], "tolerant_ch_alloc", "1", 0);
    r = avformat_find_stream_info(ic, opts);
    for (i = 0; i < n; i++)            /* nb_streams may have grown; only n dicts exist */
        av_dict_free(&opts[i]);
    av_freep(&opts);
    ptv_teletext_fix(ic);
    return r;
}

static int adec_swap(AudioState *a)
{
    const AVCodec *codec = a->dec ? a->dec->codec : NULL;
    AVCodecContext *nd;

    if (!codec || !a->ist_par)
        return AVERROR(EINVAL);
    nd = avcodec_alloc_context3(codec);
    if (!nd)
        return AVERROR(ENOMEM);
    /* pre17 R1: owned codecpar/timebase copies — the mv demux reopen may have closed the
     * AVFormatContext an AVStream* would have pointed into. */
    avcodec_parameters_to_context(nd, a->ist_par);
    nd->pkt_timebase = a->ist_pkt_tb;
    ptv_adec_opts(nd);                         /* #38: tolerant AAC decode (by name, optional) */
    if (avcodec_open2(nd, codec, NULL) < 0) {
        avcodec_free_context(&nd);
        return AVERROR(EIO);
    }
    avcodec_free_context(&a->dec);
    a->dec = nd;
    a->dec_reopens++;
    a->dec_ts_carry = AV_NOPTS_VALUE;  /* pre19.1 [PTV-ASTAMP]: fresh decoder, no extrapolation base */
    return 0;
}

static void adec_reopen(AudioState *a)
{
    a->wd_frame_us = av_gettime_relative();   /* re-arm the window whatever happens below */
    a->wd_pkts     = 0;
    if (adec_swap(a) < 0) {
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ADECWD] a%d(in%d) decoder reopen FAILED — keeping the old context, retrying in %ds\n",
               a->dbg_k, a->dbg_in, (int)(PTV_ADECWD_US / 1000000));
        return;
    }
    av_log(NULL, AV_LOG_WARNING,
           "[PTV-ADECWD] a%d(in%d) no decoded frames for %ds with packets arriving — decoder reopened "
           "(#%d, errs=%"PRId64"); anchor preserved, aresample absorbs the gap\n",
           a->dbg_k, a->dbg_in, (int)(PTV_ADECWD_US / 1000000), a->dec_reopens, a->dec_errs);
}

/* 1.0.1-pre19 #46 [PTV-ACHOP] — sustained-chop stuck-state escape. The Azorse post-storm
 * variant where the corruption NEVER ends: repeated self-shed/erase-class events sustain
 * indefinitely and the track warbles/chops until a process restart (the self-recovering
 * variant — storm ends, corrector walks the bake out — is already covered by pre14+).
 * pre18's #49 backoff suppressed the ACQUIRE churn; the remaining hole is the stuck LATCH
 * in the decode→graph→swr path itself. Escape = the FULL audio-path rebuild the 0.9.17.1
 * init-failure path proved out: swap the decoder, tear down graph+swr, seed impossible
 * input params so the next confirmed-clean frames rebuild everything through the normal
 * [PTV-AFMT] hysteresis. Anchor/pts state is PRESERVED (mid-run recovery, same as ADECWD).
 * Rate-limited per track: a permanently-broken source cycles quietly instead of chopping.
 * Detection runs on 10s windows in audio_thread: chop = decode-error rate OR self-shed
 * rate at/above the per-minute floors, sustained g_achop_sust_min minutes. */
static void achop_rebuild(AudioState *a)
{
    int swap = adec_swap(a);
    a->achop_rebuilds++;
    if (a->afg) { avfilter_graph_free(&a->afg); a->afsrc = a->afsink = NULL; a->fg_swr = NULL; a->fg_swr_flt = NULL; }
    a->use_fg = 0;
    swr_free(&a->swr);
    /* 0.9.17.1 impossible-seed: every next frame differs from these, so the path re-forms
     * from real, 5-frame-confirmed params via the [PTV-AFMT] rebuild in audio_feed. */
    a->fg_in_rate = -1;
    a->fg_in_fmt  = AV_SAMPLE_FMT_NONE;
    av_channel_layout_uninit(&a->fg_in_chl);
    av_channel_layout_uninit(&a->afmt_pending_chl);
    a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
    a->afmt_stable = 0;
    afmt_q_reset(a);
    a->afmt_rebuilds++;                        /* corrector event (§4.4), same as an AFMT rebuild */
    av_log(NULL, AV_LOG_WARNING,
           "[PTV-ACHOP] a%d(in%d) sustained audio chop (>=%d min above floor; errs=%"PRId64") — "
           "full audio-path rebuild #%d: decoder %s, graph/swr torn down; re-forms from the next "
           "clean frames [PTV-AFMT]; next attempt in >=%ds (PTV_NO_ACHOP_REBUILD=1 disables)\n",
           a->dbg_k, a->dbg_in, g_achop_sust_min, a->dec_errs, a->achop_rebuilds,
           swap < 0 ? "reopen FAILED (old context kept)" : "reopened",
           (int)(g_achop_relimit_us / 1000000));
}

static void achop_tick(AudioState *a)
{
    int64_t now, errs, sheds, shed_now;
    int chop;
    if (!g_achop)
        return;
    now = av_gettime_relative();
    shed_now = a->shed_cnt ? atomic_load_explicit(a->shed_cnt, memory_order_relaxed)
                           : atomic_load_explicit(&g_shed_cnt, memory_order_relaxed);
    if (!a->achop_win_wc) {
        a->achop_win_wc    = now;
        a->achop_win_errs  = a->dec_errs;
        a->achop_win_sheds = shed_now;
        return;
    }
    if (now - a->achop_win_wc < 10000000)
        return;
    errs  = a->dec_errs - a->achop_win_errs;
    sheds = shed_now    - a->achop_win_sheds;
    chop  = errs * 6 >= g_achop_errs_min || sheds * 6 >= g_achop_sheds_min;  /* per-minute rates */
    a->achop_sust      = chop ? a->achop_sust + 1 : 0;
    a->achop_win_wc    = now;
    a->achop_win_errs  = a->dec_errs;
    a->achop_win_sheds = shed_now;
    if (a->achop_sust >= g_achop_sust_min * 6 &&
        (!a->achop_last_wc || now - a->achop_last_wc >= g_achop_relimit_us)) {
        a->achop_last_wc = now;
        a->achop_sust    = 0;
        achop_rebuild(a);
    }
}

/* 1.0.1-pre15 #33 §3 (g_glueclass + PTV_NBS_FILL, opt-in): synthesize ONE quantum of stamped
 * silence at this track's expected next graph-door position. Triggered by a FILL sentinel from
 * the demux thread — the only thread alive during a corrupt-discard starvation phase (this
 * thread is otherwise blocked on recv). Frames are built in the ACTIVE input configuration
 * (fg_in_*) and pushed through the normal audio_feed path, so AGLUE sees a dense continuation,
 * AFMT sees unchanged params, output labels stay dense and house-aligned, encoders/mux/gates
 * stay alive and the sensor keeps a valid ≈0 reading (instead of the −8..−13.5s label-walk).
 * Pre-anchor / graph-less / unstamped tracks skip (a track BORN into a broken phase stays dead
 * until real frames arrive — the 0.9.17.1 AFMT retry owns that; upstream problem). */
static void nbs_fill_quantum(AudioState *a)
{
    int64_t dur_us, base, now, want_us;
    int n, i, ret = 0;

    if (!g_glueclass || !a->pts_set || !a->use_fg ||
        a->glue_raw_last_us == AV_NOPTS_VALUE || a->fg_in_rate <= 0)
        return;
    if (!a->nbs_fill_active) {
        a->nbs_fill_active = 1;
        a->nbs_last_wall_us = 0;
        a->nbs_carry_us     = 0;
        a->nbs_engage_wc    = av_gettime_relative();
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ADISC] a%d(in%d) silence-fill ENGAGED — nothing decoding on this track while "
               "video flows (packets corrupt-discarded, or none arriving for 2 s); synthesizing "
               "dense silence until real frames resume\n",
               a->dbg_k, a->dbg_in);
    }
    dur_us = av_rescale(1024, 1000000, a->fg_in_rate);
    if (dur_us <= 0)
        return;
    /* rr15 F9: synthesize the WALL time actually elapsed since the previous quantum,
     * carrying the sub-frame remainder — int(quantum/frame) per sentinel under-filled
     * 30-37% (sentinel cadence = first corrupt pkt ≥quantum, not exactly quantum; mp2
     * PES 120ms vs the 100ms quantum), leaving a +14.5s resume step after a 40s phase.
     * Clamped so a sentinel drought can never dump a burst. */
    now     = av_gettime_relative();
    a->dmg_wc = now;                 /* 2.0.0-pre9.3 (T-080 A2): a silence-fill phase is damaged audio */
    want_us = (a->nbs_last_wall_us ? now - a->nbs_last_wall_us : g_nbs_quantum_us)
            + a->nbs_carry_us;
    a->nbs_last_wall_us = now;
    if (want_us > 2000000)
        want_us = 2000000;
    n = (int)(want_us / dur_us);
    a->nbs_carry_us = want_us - (int64_t)n * dur_us;
    if (n < 1) {
        a->nbs_fills++;
        return;                      /* remainder carried to the next sentinel */
    }
    base = a->glue_raw_last_us + a->glue_raw_dur_us;
    a->nbs_feeding = 1;
    for (i = 0; i < n && ret >= 0; i++) {
        AVFrame *s = av_frame_alloc();
        if (!s)
            break;
        s->nb_samples  = 1024;
        s->format      = a->fg_in_fmt;
        s->sample_rate = a->fg_in_rate;
        av_channel_layout_copy(&s->ch_layout, &a->fg_in_chl);
        if (av_frame_get_buffer(s, 0) >= 0) {
            av_samples_set_silence(s->extended_data, 0, s->nb_samples,
                                   s->ch_layout.nb_channels, s->format);
            s->pts = s->best_effort_timestamp =
                av_rescale_q(base + i * dur_us, AV_TIME_BASE_Q, a->ist_tb);
            ret = audio_feed(a, s);
        }
        av_frame_free(&s);
    }
    a->nbs_feeding = 0;
    a->nbs_fills++;
    if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO)   /* 2.0.0-pre8: afill= (NBS run so far) */
        atomic_store_explicit(&g_afill_us[a->dbg_k], av_gettime_relative() - a->nbs_engage_wc, memory_order_relaxed);
}

/* 2.0.0-pre6 (T-056 §5.1/§5.2): silence while input 0 holds. This track's door position is
 * label + glue_off + corr + house_skew; in a hold the labels stop and house_skew grows by one tick per held
 * video tick (house clock minus the frozen content), so filling the door up to the last label's end +
 * glue_off + corr + house_skew(now) keeps audio's output position on the held video. Frames are stamped AT
 * the door (hold_feeding: AGLUE and the door additions are skipped) and raw labels stay untouched, so a
 * LIVE LOSS rejoin (labels +W, house_skew back to ~0) lands where the fill ended and a BURST rejoin (labels
 * contiguous, house_skew kept) continues it. Only while the track itself is silent (no real frame for
 * max(150 ms, 3 x its arrival cadence)):
 * a PSI-only hold keeps its real audio, and real frames arriving before the hold ends stop the fill.
 * Driven while video only repeats (master: no fresh frame for 1 s, or a hold), not only in a declared hold. */
/* 2.0.0-pre8.1: release the frames buffered for the post-rejoin mapping (before the frame that ends the wait) */
static void rj_flush(AudioState *a)
{
    a->rj_released  = 1;
    a->rj_releasing = 1;
    if (a->rj_n) {
        atomic_fetch_sub_explicit(&g_rj_buffering, 1, memory_order_relaxed);
        av_log(NULL, AV_LOG_INFO, "[PTV-SRC] a%d(in%d) %d frames held %.2f s for the post-rejoin mapping\n",
               a->dbg_k, a->dbg_in, a->rj_n, (av_gettime_relative() - a->rj_t0) / 1e6);
    }
    for (int i = 0; i < a->rj_n; i++) {
        audio_feed(a, a->rj_buf[i]);
        av_frame_free(&a->rj_buf[i]);
    }
    a->rj_n = 0;
    a->hold_drop_n = 0;
    a->rj_t0 = 0;        /* 2.0.0-pre8.4: the 4 s cap times THIS buffering — a stale start released the next outage's
                          * frames at once (canary NTD 2026-10-05: 2nd outage → audio ~12 s late after the return) */
    a->rj_releasing = 0;
}

static void hold_fill_quantum(AudioState *a)
{
    int64_t dur_us, door, target, now = av_gettime_relative();
    int n, i, ret = 0;

    if (a->vm_n)
        vm_drain(a, 0);                             /* 2.1 T-409: frames whose content went on air since */

    /* 2.0.0-pre8.1: the wait can end between real frames — a post-gap burst is often all buffered before the video
     * shows the jump, and the next frame came 265 ms later (sync_gap_20: a 0.32 s dark wire, video waiting for audio
     * at the delivery gate). The sentinels keep coming while a track buffers, so look again on each. */
    if (a->rj_n && !a->rj_released && rj_release_ok(a))
        rj_flush(a);
    /* 2.1 T-409 (the record door): with the audio placed by the video's record, the fill follows WHERE THE VIDEO IS ON AIR
     * (house output + h0, the placement's own domain) 200 ms behind, instead of time / dup counts since the last real
     * frame: queued sound belongs to content not shown yet, so it lands at or after that point — the fill can never cover
     * it, and the returning frames meet the fill within the margin (aresample pads the rest) */
    if (vm_mode(a) && a->pts_set && a->use_fg && a->fg_in_rate > 0 && a->acomp_exp_us != AV_NOPTS_VALUE) {
        int64_t h0, pos = atomic_load_explicit(&g_house_out_us, memory_order_relaxed);
        pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
        if (h0 == AV_NOPTS_VALUE || pos <= 0)
            return;
        dur_us = av_rescale(1024, 1000000, a->fg_in_rate);
        target = pos + h0 + a->corr.corr_us - 200000;
        door   = a->acomp_exp_us;
        if (target - door > 2000000)
            target = door + 2000000;
        n = dur_us > 0 ? (int)((target - door) / dur_us) : 0;
        if (n < 1)
            return;
        goto fill;
    }
    if (!a->pts_set || !a->use_fg || a->multiview || a->fg_in_rate <= 0 ||
        a->glue_raw_last_us == AV_NOPTS_VALUE || a->acomp_exp_us == AV_NOPTS_VALUE ||
        now - a->glue_wall_last_us < FFMAX(150000, 3 * a->glue_cad_us))   /* the track's own arrival cadence */
        return;
    dur_us = av_rescale(1024, 1000000, a->fg_in_rate);
    /* 2.0.0-pre9.4 (T-083): ride the output time on ticks whose content did not advance (dups, hold pictures) since this
     * track's last real frame (g_dup_out_us), not the house_skew. Both grow alike; but a LIVE LOSS rejoin map pulls house_skew back by the outage
     * while this track's last label stays where it was if no real audio came back (Cinestar 50 % loss: all AAC
     * corrupt-discarded) — the target fell 67 s behind the door and the track was absent from the wire for 75 s. The
     * clock never steps back. (Not the whole output position: that also counts fresh ticks, and every routine
     * 120 ms video stall then filled the time since the last real frame over the real frames about to arrive.) */
    target = a->glue_raw_last_us + a->glue_raw_dur_us + a->glue_off_us + a->corr.corr_us + a->hf_hs_real +
             (atomic_load_explicit(&g_dup_out_us, memory_order_relaxed) - a->hf_dup_real);
    door   = a->acomp_exp_us;
    if (target - door > 2000000)
        target = door + 2000000;                 /* catch up at most 2 s per sentinel (100 ms apart) */
    n = dur_us > 0 ? (int)((target - door) / dur_us) : 0;
    if (n < 1)
        return;
fill:
    if (!a->hold_fill_us) {
        a->hold_fill_wc0 = now;
        a->hold_fresh_wc = 0;
        a->rj_released   = 0;                                /* 2.0.0-pre8.1: this fill's real frames get buffered */
        a->rj_t0         = 0;                                /* 2.0.0-pre8.4: ... timed from their own first frame */
        a->rj_cls_seq0   = atomic_load_explicit(&g_rj_cls_seq, memory_order_acquire);
    }
    a->hold_feeding = 1;
    for (i = 0; i < n && ret >= 0; i++) {
        AVFrame *s = av_frame_alloc();
        if (!s)
            break;
        s->nb_samples  = 1024;
        s->format      = a->fg_in_fmt;
        s->sample_rate = a->fg_in_rate;
        av_channel_layout_copy(&s->ch_layout, &a->fg_in_chl);
        if (av_frame_get_buffer(s, 0) >= 0) {
            av_samples_set_silence(s->extended_data, 0, s->nb_samples,
                                   s->ch_layout.nb_channels, s->format);
            s->pts = s->best_effort_timestamp =
                av_rescale_q(door + i * dur_us, AV_TIME_BASE_Q, a->ist_tb);
            ret = audio_feed(a, s);
            if (a->hold_fill_us < 3000000 && a->hold_fill_us + dur_us >= 3000000)   /* short repeat runs stay quiet */
                av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] a%d(in%d) hold fill: silence aligned to the held video "
                       "(PTV_NO_SRC_FILL=1 disables)\n", a->dbg_k, a->dbg_in);
            a->hold_fill_us += dur_us;
        }
        av_frame_free(&s);
    }
    a->hold_feeding = 0;
    if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO)   /* 2.0.0-pre8: afill= */
        atomic_store_explicit(&g_afill_us[a->dbg_k], a->hold_fill_us, memory_order_relaxed);
}

/* 2.0.0-pre18 T-316: replay the pre-h0 ring as soon as h0 is known. It used to be replayed only by the NEXT
 * decoded audio frame — 21 ms later on a steady source, but on a burst source the whole first burst sits in the
 * ring and the next frame comes with the next burst: no audio on the output for 8 s (sync_bursty_8: video released
 * alone after 6 s, the first segments without audio, a default-probing HLS segmenter could not open the output). */
static int aq_replay_if_anchored(AudioState *a)
{
    int64_t h0;
    int i, ret = 0;
    if (a->pts_set || !a->aq_npending)
        return 0;
    pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
    if (h0 == AV_NOPTS_VALUE)
        return 0;
    for (i = 0; i < a->aq_npending; i++) {
        if (ret >= 0) ret = apark_feed(a, a->aq_pending[i]);
        av_frame_free(&a->aq_pending[i]);
    }
    a->aq_npending = 0;
    return ret;
}

void *audio_thread(void *arg)
{
    AudioState *a = arg;
    AVPacket *pkt;
    AVFrame  *frame = av_frame_alloc();
    int ret = 0;

    if (!frame)
        goto done;
    for (;;) {
        if ((!a->pts_set && a->aq_npending) ||     /* T-316: frames wait for h0 — poll, replay the moment it is known */
            a->apark_n > 0) {                      /* T-411: frames wait for the picture — poll, release as it catches up */
            ret = av_thread_message_queue_recv(a->audio_q, &pkt, AV_THREAD_MESSAGE_NONBLOCK);
            if (ret == AVERROR(EAGAIN)) {
                if (a->apark_n > 0 && (ret = apark_release(a, 0)) < 0)
                    break;
                if (!a->pts_set && a->aq_npending && (ret = aq_replay_if_anchored(a)) < 0)
                    break;
                if (!a->pts_set || a->apark_n > 0)
                    av_usleep(5000);
                continue;
            }
        } else
            ret = av_thread_message_queue_recv(a->audio_q, &pkt, 0);
        if (ret < 0) break;
        if (pkt && (pkt->flags & (PTV_PKT_FLAG_NBS_FILL | PTV_PKT_FLAG_HOLD_FILL)) && a->apark_n > 0) {
            av_packet_free(&pkt);                /* T-411: sound waits for the picture — no hole to fill */
            continue;
        }
        if (pkt && (pkt->flags & PTV_PKT_FLAG_NBS_FILL)) {
            /* pre15 §3 FILL sentinel — NOT a packet: must not feed the decoder and must not
             * advance wd_pkts (ADECWD would read "packets arriving, nothing decodes" and churn
             * a reopen every 45s of a fill phase). */
            av_packet_free(&pkt);
            nbs_fill_quantum(a);
            continue;
        }
        if (pkt && (pkt->flags & PTV_PKT_FLAG_SLOT_BIND)) {   /* 2.0.2 T-386: the slot bound — open its real decoder */
            slot_bind_audio(a, pkt);
            av_packet_free(&pkt);
            continue;
        }
        if (pkt && (pkt->flags & PTV_PKT_FLAG_SLOT_FILL)) {   /* 2.0.2 T-386: a deferred slot's silence, not a packet */
            av_packet_free(&pkt);
            if ((ret = slot_pump(a)) < 0)
                break;
            continue;
        }
        if (pkt && (pkt->flags & PTV_PKT_FLAG_HOLD_FILL)) {   /* 2.0.0-pre6 hold-fill sentinel, not a packet */
            av_packet_free(&pkt);
            hold_fill_quantum(a);
            continue;
        }
        a->wd_pkts++;
        if (!a->wd_frame_us) a->wd_frame_us = av_gettime_relative();   /* seed at first packet */
        if (pkt && (pkt->flags & PTV_PKT_FLAG_RELABEL))
            a->rl_armed = 1;                     /* 2.0.1 T-087: the next decoded frame starts the seam */
        ret = avcodec_send_packet(a->dec, pkt);
        av_packet_free(&pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN))   /* 1.0.1: decode errors surface here too (eager decode) — count/log them */
            adec_error(a, ret);
        while (ret >= 0) {
            ret = avcodec_receive_frame(a->dec, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) { adec_error(a, ret); ret = 0; break; }   /* 1.0.1: drop + continue (was: silent thread death) */
            a->wd_frame_us = av_gettime_relative();
            a->wd_pkts     = 0;
            if (a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO)   /* pre15 E6: demux-visible decode watermark (NBS discriminator) */
                atomic_store_explicit(&g_adec_frame_wc[a->dbg_k], a->wd_frame_us,
                                      memory_order_relaxed);
            /* 1.0.1-pre19.1 [PTV-ASTAMP] — stamp un-stamped decoded frames by sample-count
             * extrapolation from the previous frame (see the AudioState field comment: the
             * probe-failed broken-phase route leaves ADTS frames 2..6 of each PES with no pts;
             * audio_push would drop them = ticking). Carry only ever spans contiguous clean
             * decode: adec_error()/adec_swap() invalidate it. */
            if (g_adts_split && frame->nb_samples > 0 && frame->sample_rate > 0) {
                if (a->dbg_in >= 0 && a->dbg_in < PTV_MAX_INPUT) {   /* rider (a), rr191 F3: a demux
                                                                      * REOPEN is a fresh join — the
                                                                      * extrapolation carry must not
                                                                      * span it */
                    int64_t rwc = atomic_load_explicit(&g_reopen_wc[a->dbg_in], memory_order_relaxed);
                    if (rwc && rwc != a->reopen_seen_wc) {
                        a->reopen_seen_wc = rwc;
                        a->dec_ts_carry   = AV_NOPTS_VALUE;
                    }
                }
                if (frame->best_effort_timestamp == AV_NOPTS_VALUE &&
                    a->dec_ts_carry != AV_NOPTS_VALUE) {
                    frame->pts = frame->best_effort_timestamp = a->dec_ts_carry;
                    if (!a->astamp_logged) {
                        a->astamp_logged = 1;
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-ASTAMP] a%d(in%d) un-stamped decoded frames — stamping by "
                               "sample-count extrapolation (probe-failed source phase; "
                               "PTV_NO_ADTS_SPLIT=1 reverts)\n", a->dbg_k, a->dbg_in);
                    }
                }
                if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
                    a->dec_ts_carry = frame->best_effort_timestamp +
                        av_rescale_q(frame->nb_samples,
                                     (AVRational){1, frame->sample_rate}, a->ist_tb);
                else
                    a->dec_ts_carry = AV_NOPTS_VALUE;
            }
            rl_feed(a, frame);                   /* 2.0.1 T-087 */
            ret = audio_push(a, frame);
            av_frame_unref(frame);
            if (ret < 0) goto done;
        }
        if (g_adecwd && a->wd_pkts > 0 &&
            av_gettime_relative() - a->wd_frame_us > PTV_ADECWD_US)
            adec_reopen(a);
        achop_tick(a);   /* 1.0.1-pre19 #46: sustained-chop stuck-state escape (10s windows) */
    }
    /* flush decoder -> resampler/filtergraph -> encoder */
    avcodec_send_packet(a->dec, NULL);
    while (avcodec_receive_frame(a->dec, frame) >= 0) { audio_push(a, frame); av_frame_unref(frame); }
    apark_release(a, 1);                                   /* T-411: the parked sound goes out */
    if (a->use_fg) {
        if (av_buffersrc_add_frame(a->afsrc, NULL) >= 0)   /* signal EOF to the graph */
            audio_drain_fg(a);
        audio_encode_push(a, NULL);                        /* flush encoders */
    } else {
        uint8_t **out = NULL; int got, out_max = 4096;
        /* 1.0.1-pre19 #42: a dead audio path (init failed on an undecodable source phase,
         * AFMT retry pending — v0.9.17.1) reaches EOF/death with a->swr == NULL. The
         * v0.9.17.1 NULL guard covered audio_feed only, so this flush dereferenced NULL
         * inside swr_convert (SIGSEGV at swr_is_initialized, fault addr = offsetof
         * in_buffer.ch_count — the pre11 awe/Azorse broken-AAC capture crash). */
        if (a->swr &&
            av_samples_alloc_array_and_samples(&out, NULL, a->out_chl.nb_channels,
                                               out_max, a->out_sfmt, 0) >= 0) {
            while ((got = swr_convert(a->swr, out, out_max, NULL, 0)) > 0)
                av_audio_fifo_write(a->fifo, (void **)out, got);
            av_freep(&out[0]); av_freep(&out);
        }
        audio_drain_fifo(a);
        audio_encode_push(a, NULL);
    }
done:
    av_frame_free(&frame);
    { int i; for (i = 0; i < a->aq_npending; i++) av_frame_free(&a->aq_pending[i]); a->aq_npending = 0; }
    while (a->apark_n > 0) { av_frame_free(&a->apark[a->apark_h]); a->apark_h = (a->apark_h + 1) % PTV_APARK; a->apark_n--; }
    afmt_q_reset(a);
    av_thread_message_queue_set_err_send(a->audio_q, AVERROR_EOF);   /* unblock demux (a SENDER) */
    { int i; for (i = 0; i < a->n_out; i++) {        /* EOF marker to each muxer */
        AVPacket *eof = NULL; av_thread_message_queue_send(a->mux_q[i], &eof, 0); } }
    return NULL;
}

/* Build the audio filtergraph: abuffer -> [user -af chain] -> aformat -> abuffersink.
 * Mirrors build_video_filter for audio. The trailing aformat pins the sink to the
 * encoder's format (48k stereo + enc sample_fmt) so the graph auto-inserts any
 * needed aresample even when -af omits one; the -af chain (aresample=async,
 * acompressor, alimiter, ...) runs first. Sets a->use_fg on success. */
int build_audio_filter(AudioState *a, AVCodecContext *adec, AVRational tb,
                              const char *af, enum AVSampleFormat out_fmt)
{
    char args[256], chain[512], chl[64], outchl[64];
    const AVFilter *bsrc  = avfilter_get_by_name("abuffer");
    const AVFilter *bsink = avfilter_get_by_name("abuffersink");
    AVFilterInOut *ins = avfilter_inout_alloc(), *outs = avfilter_inout_alloc();
    int ret;

    if (!bsrc || !bsink || !ins || !outs) { ret = AVERROR(ENOMEM); goto end; }
    a->afg = avfilter_graph_alloc();
    if (!a->afg) { ret = AVERROR(ENOMEM); goto end; }

    /* 2.0.0-pre17 T-347/T-349: the input is FIXED (48 kHz fltp, the track's output layout) — the source format is
     * converted in front of it (cv_swr), so the graph is the same whatever the source carries, now or later */
    (void)adec;
    ptv_graph_in_params(a);
    av_channel_layout_describe(&a->fg_in_chl, chl, sizeof chl);
    snprintf(args, sizeof args,
             "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
             tb.num, tb.den, a->fg_in_rate,
             av_get_sample_fmt_name(a->fg_in_fmt), chl);
    if ((ret = avfilter_graph_create_filter(&a->afsrc, bsrc, "in", args, NULL, a->afg)) < 0) goto end;
    if ((ret = avfilter_graph_create_filter(&a->afsink, bsink, "out", NULL, NULL, a->afg)) < 0) goto end;

    av_channel_layout_describe(&a->out_chl, outchl, sizeof outchl);   /* -ac:a:N target layout */
    snprintf(chain, sizeof chain,
             "%s%saformat=sample_fmts=%s:sample_rates=48000:channel_layouts=%s",
             af ? af : "", af ? "," : "", av_get_sample_fmt_name(out_fmt), outchl);

    outs->name = av_strdup("in");  outs->filter_ctx = a->afsrc;  outs->pad_idx = 0; outs->next = NULL;
    ins->name  = av_strdup("out"); ins->filter_ctx  = a->afsink; ins->pad_idx  = 0; ins->next  = NULL;
    if ((ret = avfilter_graph_parse_ptr(a->afg, chain, &ins, &outs, NULL)) < 0) goto end;
    if ((ret = avfilter_graph_config(a->afg, NULL)) < 0) goto end;

    /* Grab the async aresample filter's internal SwrContext. swr_get_delay() on it is the
     * FAITHFUL resampler-slip sensor: the PTS-based metrics (offset=/house_skew/sync_check D)
     * are structurally blind to a sub-resampler slip, so this is the one number that sees it.
     * Prefer the aresample whose swr has async set (the explicit -af one), fall back to the first. */
    a->fg_swr = NULL;
    a->fg_swr_flt = NULL;
    for (unsigned i = 0; i < a->afg->nb_filters; i++) {
        AVFilterContext *fc = a->afg->filters[i];
        if (fc && fc->filter && !strcmp(fc->filter->name, "aresample") && fc->priv) {
            SwrContext *cand = av_opt_child_next(fc->priv, NULL);
            if (cand) {
                int64_t as = 0;
                av_opt_get_int(cand, "async", 0, &as);
                if (!a->fg_swr || as) { a->fg_swr = cand; a->fg_swr_flt = fc; }
                if (as) break;
            }
        }
    }
    /* 2.0.1 T-357: start the async resampler's compensation stage NOW, while it holds no timestamps. A 48 -> 48 kHz
     * swr has no resampler until its first SOFT compensation; swr_set_compensation() then creates it by re-running
     * swr_init(), which resets firstpts, and the next swr_next_pts() re-seeds outpts from the input label — the step
     * it was asked to stretch over passes through as a label step. Measured (std/sync_ac3_bitflip_100, prod graph):
     * the first undecodable AAC frame = in-pts +21.3 ms -> lazy swr_init -> aresample out +21.2 ms -> alimiter
     * (latency=1) re-anchors on the step and loses its 5 ms latency offset -> sink +26.2 ms = an AAC hole at 20.5 s
     * (48 ms on older media: two frames); every later step was soft-compensated, as designed. A zero compensation
     * creates the resampler here instead (what the first compensation does anyway, minus the reset). */
    if (a->fg_swr && !getenv("PTV_NO_SWR_UPFRONT")) {   /* PTV_NO_SWR_UPFRONT=1 reverts */
        int64_t as = 0;
        av_opt_get_int(a->fg_swr, "async", 0, &as);
        if (as && (ret = swr_set_compensation(a->fg_swr, 0, 0)) < 0)
            av_log(NULL, AV_LOG_WARNING, "ptvencoder: could not start the aresample compensation stage (%s)\n",
                   av_err2str(ret));
    }

    /* deliver encoder-sized frames so we can feed them straight to the AAC
     * encoders carrying their own (async-corrected) PTS — no FIFO repackaging. */
    if (a->frame_size > 0)
        av_buffersink_set_frame_size(a->afsink, a->frame_size);

    av_log(NULL, AV_LOG_INFO, "ptvencoder: audio filter [%s]\n", chain);
    a->use_fg = 1;
    ret = 0;
end:
    avfilter_inout_free(&ins);
    avfilter_inout_free(&outs);
    return ret;
}

