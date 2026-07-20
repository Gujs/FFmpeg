#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

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

static int     g_aglue_max_ms = 1000;     /* [PTV-AGLUE] cap (ms): steps above this are the >1s discontinuity layer's job
                                           * (demux_unwrap/LAYERA) — the glue logs and stands aside. Internalized 0.9.18.7 (was PTV_AGLUE_MAX_MS).
                                           * v0.9.17: 900→1000 to meet LAYERA's threshold exactly — the 900ms-1s sliver
                                           * previously had NO owner (aglue stood aside, LAYERA blind, aresample followed). */
static int     g_af_acquire_us = 100000;   /* gap above this → discrete drop/pad; at/below → smooth nudge (internalized 0.9.18.7; was PTV_AF_ACQUIRE_MS) */
static int     g_af_rate_us = 10000;       /* smooth follow/nudge rate ceiling, us per second. B2 (2026-06-21):
                                            * raised 2000→10000 so B1's content-anchored offset tracks a
                                            * source-slowness dup ramp in vlag (~6–8 ms/s) in near-real-time
                                            * instead of lagging ~100 s at 2 ms/s. 10 ms/s ≈ 1% (under the ~2%
                                            * audible budget) and only engages transiently while converging —
                                            * steady-state step=gap is tiny. Internalized 0.9.18.7 (was PTV_AF_RATE_MS_S). */
static int     g_pll_ema_shift = 7;          /* EMA smoothing of the measured offset (τ≈2.7s @ ~47 afps). Raised 5→7 (v0.6.21): on jittery NTSC legs the measured offset has ±100–200ms noise that a τ≈0.7s EMA tracked → the acquire chased it; τ≈2.7s averages the zero-mean noise below the threshold so only the DC startup bank triggers an acquire. Internalized 0.9.18.7 (was PTV_PLL_EMA_SHIFT). */
static int64_t g_pll_tau_us = 5000000;       /* integral track time-constant (us): step = ema*frame_us/τ (internalized 0.9.18.7; was PTV_PLL_TAU_MS) */
static int     g_pll_acquire_us = 40000;     /* |ema| above this = "large" → ACQUIRE one-shot; else TRACK. 40ms ≈ 2 audio frames: shrinks the dead band [gate 25ms, threshold] so a stable sub-100ms residual (TRACK is guard-limited on jittery sources) is snapped in by a whole-frame acquire instead of stranded. The flatness debounce (threshold/4 = 10ms) still rejects jitter. Internalized 0.9.18.7 (was PTV_PLL_ACQUIRE_MS). */
static int     g_pll_acquire_n = 32;         /* debounce: N stable (large AND flat) readings before acquire; also the refractory (internalized 0.9.18.7; was PTV_PLL_ACQUIRE_N) */
static int64_t g_pll_refractory_us = 12000000; /* v0.6.21: HARD refractory after an acquire (12s) — the backstop that breaks the self-excited limit cycle on jittery legs (the acquire's own drop/pad perturbs the next measurement → re-triggers; box: a2 thrashed ~1 acquire/7s, acq=92). Must exceed the thrash period; bounds acquires to ≤1/12s regardless of the noise spectrum. Was conflated with g_pll_acquire_n (32 frames ≈0.68s — far too short). Internalized 0.9.18.7 (was PTV_PLL_REFRACTORY_MS). */
static int     g_pll_noise_k = 3;            /* v0.6.22: NOISE-ADAPTIVE acquire threshold = max(g_pll_acquire_us, k·pll_dev). Clean legs (dev≈0) keep the 40ms; jittery legs raise the bar above their own offset jitter so steady-state noise can't re-fire the acquire (the 0.6.20/0.6.21 limit cycle). 0 disables (fixed threshold). Internalized 0.9.18.7 (was PTV_PLL_NOISE_K). */
static int     g_pll_dev_shift = 9;          /* v0.6.22: EMA shift for pll_dev (τ≈11s) — slow so dev ramps AFTER the big startup bank is caught (dev≈0 → thr=40ms at t0 → bank acquires), then rises to the noise floor → steady-state quiet. Internalized 0.9.18.7 (was PTV_PLL_DEV_SHIFT). */
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

/* ---- audio path (decode -> resample 48k stereo -> AAC -> mux) ---- */

/* encode the SAME loudness-processed frame into each rung's own AAC encoder (so
 * per-rung -b:a is honored), routing each rung's packets to its muxer. The frame
 * is only ref'd by avcodec_send_frame, so the one frame feeds all N encoders.
 * frame=NULL flushes every encoder. (ffmpeg's filter -> asplit -> N encoders.) */
static int audio_encode_push(AudioState *a, AVFrame *frame)
{
    int i, ret;
    for (i = 0; i < a->n_out; i++) {
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
            av_packet_rescale_ts(pkt, a->enc[i]->time_base, a->ost[i]->time_base);
            pkt->stream_index = a->ost[i]->index;
            if (a->gate[i] && (pkt->dts != AV_NOPTS_VALUE || pkt->pts != AV_NOPTS_VALUE)) {
                /* §7.5a: hold until this rung's video front reaches it (block=1 → back-pressure) */
                int64_t ts = pkt->dts != AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
                dlv_enqueue(a->gate[i], pkt, av_rescale_q(ts, a->ost[i]->time_base, AV_TIME_BASE_Q), 1);
            } else if (av_thread_message_queue_send(a->mux_q[i], &pkt, 0) < 0)   /* blocking */
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
typedef struct RsyncTrackR { int valid; int64_t R_us, mv_wall, ma_wall; } RsyncTrackR;
/* TEST-ONLY (PTV_RSCORR_TESTWALK, µs/s; PTV_PLL_TESTNOISE_MS precedent): add a linearly
 * walking offset to the R the CORRECTOR reads (the sensor/stats stay untouched) — the
 * "maximally lying-but-plausible sensor" of §6's damage bound, the only way to exercise the
 * per-engagement authority cap in bounded wall time (a real R never walks: label-faithful
 * drift reads R=0 by design). Never set in production. */
static int64_t g_rscorr_testwalk_us_s = 0, g_rscorr_testwalk_t0 = 0;
/* pre17 fix round (F2): PTV_RSCORR_TESTWALK_CAP_MS saturates the walk — without it the
 * linear-forever walk is cancelled by the steer at equal rate, R never re-enters the park
 * band, and mv ENGAGE→PARK is structurally unreachable via TESTWALK (reviewer F2). A capped
 * walk = a synthetic SETTLED bake the corrector can steer to 0 and PARK on. TEST-ONLY. */
static int64_t g_rscorr_testwalk_cap_us = 0;
static RsyncTrackR rsync_track_R(const AudioState *a, int64_t now)
{
    RsyncTrackR r = { 0, 0, 0, 0 };
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
    r.R_us = (atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed)
            + atomic_load_explicit(&g_rsx.ev_us[in],  memory_order_relaxed))
           - (a->rs_ma_ema
            + atomic_load_explicit(&g_rsx.ea_us[k], memory_order_relaxed));
    if (g_rscorr_testwalk_us_s == 0) {
        const char *tw = getenv("PTV_RSCORR_TESTWALK");
        const char *ta = getenv("PTV_RSCORR_TESTWALK_AT_S");
        const char *tc = getenv("PTV_RSCORR_TESTWALK_CAP_MS");
        g_rscorr_testwalk_us_s = (tw && atoi(tw)) ? atoi(tw) : -1;   /* -1 = parsed, off */
        g_rscorr_testwalk_cap_us = (tc && atoi(tc) > 0) ? (int64_t)atoi(tc) * 1000 : 0;
        /* PTV_RSCORR_TESTWALK_AT_S delays the walk's onset — the cap is only reachable
         * through MID-ENGAGEMENT drift (§6): a walk from birth is correctly rejected by the
         * dwell stability bound and can never engage (verified, first fixture round). */
        g_rscorr_testwalk_t0 = now + (ta ? (int64_t)atoi(ta) * 1000000 : 0);
    }
    if (g_rscorr_testwalk_us_s > 0 && now > g_rscorr_testwalk_t0) {
        int64_t w = av_rescale(now - g_rscorr_testwalk_t0, g_rscorr_testwalk_us_s, 1000000);
        if (g_rscorr_testwalk_cap_us > 0 && w > g_rscorr_testwalk_cap_us)
            w = g_rscorr_testwalk_cap_us;   /* F2: saturated walk = a steerable settled bake */
        r.R_us += w;
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
    if (a->pll_acq_count != c->acq_snap) { c->acq_snap = a->pll_acq_count; return "pll-acquire"; }
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
    if (a->glue_exp_dl) {
        /* outstanding = registered AND not yet expired. The slot's deadline is only zeroed
         * by the AGLUE consume/expire path when a matching step ARRIVES; a registration
         * whose step never arrives sits expired in the slot indefinitely (fixture F8 round
         * 1: it wedged the corrector in permanent HOLD). Expiry is a wall-clock compare —
         * no store, the AGLUE path stays the only writer. */
        int64_t dl = atomic_load_explicit(a->glue_exp_dl, memory_order_relaxed);
        if (dl != 0 && now <= dl)
            return "pair-expect outstanding";
    }
    if (a->corr_layera_active && *a->corr_layera_active)   /* advisory read (demux-written int) */
        return "layera buffering";
    if (a->nbs_fill_active)   /* pre15 §5 rule 2: R is synthetic-flat on a filled track — never engage */
        return "nbs silence-fill";
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
    c->acq_snap    = a->pll_acq_count;
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
     * clear the span. PTV_NO_RSCORR_CEIL=1 reverts; PTV_RSCORR_CEIL_MIN tunes (15min). */
    if (g_rscorr_ceil &&
        (c->state == PTV_CORR_ARMED || c->state == PTV_CORR_DWELL ||
         c->state == PTV_CORR_DISARMED)) {
        if (!r->valid || dead || c->implaus_wc || llabs(R) <= g_rscorr_engage_us) {
            c->starve_wc = 0;
        } else if (!c->starve_wc ||
                   llabs(R - c->starve_r0) >= FFMAX(40000, llabs(R) / 4)) {
            c->starve_wc = now;      /* open — or churning R: restart from here */
            c->starve_r0 = R;
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
             * dR/dcorr = −1: +R (audio early) → +step → audio content delayed → R → 0. */
            int64_t rate = R / 30;                                   /* µs of trim per second */
            int64_t step;
            if (rate >  g_rscorr_slew_us_s) rate =  g_rscorr_slew_us_s;
            if (rate < -g_rscorr_slew_us_s) rate = -g_rscorr_slew_us_s;
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
                if (nowf - a->dbg_flow_last_us >= 5000000) {
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
            if (a->multiview && g_audio_follow && g_avsync_pll) {
                /* B3 (Phase B3) — CLOSED-LOOP two-regime controller on the MEASURED av_offset_us.
                 * Emits on this SAME content-anchored base (want = opts + applied) so ACQUIRE and TRACK
                 * share one base: the acquire's content-drop (opts jumps +Δq) and applied += Δq CANCEL
                 * in want → continuous, the monotonic guard never sees a backward step. ACQUIRE snaps the
                 * frozen startup bank out in one tune-in skip; TRACK is a type-1 integral trim. Drives
                 * the measured offset → 0. Sign: d(offset)/d(applied) < 0, so applied += (+offset). */
                int nb = filt->nb_samples;
                int64_t frame_us = (int64_t)nb * 1000000 / a->out_rate;
                int64_t want;
                if (frame_us < 1) frame_us = 1;
                if (a->pll_drop > 0) {                       /* consume a pending one-shot DROP (advance) */
                    a->pll_drop--; av_frame_unref(filt); continue;
                }
                if (!a->af_started) { a->af_applied_us = 0; a->af_started = 1; }  /* seed 0 (house_skew is wrong-sign in the banked regime) */
                if (a->av_off_valid) {
                    int64_t off = a->av_offset_us;           /* the FAITHFUL measured offset (vlag − alag) */
                    if (g_pll_testnoise_us)                  /* TEST-ONLY: ±N square wave (default ~7s flip, matches the box thrash period; holds long enough to defeat the debounce like the real noise) to reproduce the box limit cycle locally. 1.0.1-pre18 (#49 gate): PTV_PLL_TESTNOISE_P sets the half-period in FRAMES — the erase-class storm is a FLAT step flipping SLOWER than pll_dev's τ (~11s), so the 7s default lets the noise-adaptive threshold tame it and the storm control never forms; ~30s flips model the live class. */
                        off += ((a->out_frames / g_pll_testnoise_frames) & 1) ? g_pll_testnoise_us : -g_pll_testnoise_us;
                    if (!a->pll_seed) { a->pll_ema = off; a->pll_dbnc_ref = off; a->pll_seed = 1; }
                    else a->pll_ema += (off - a->pll_ema) >> g_pll_ema_shift;   /* smooth ±vlag jitter (N6: toward −∞, sub-ms, negligible) */
                    /* v0.6.22: NOISE-ADAPTIVE acquire threshold. Track the leg's offset jitter (slow EMA of
                     * |off−ema|, seeded 0 so it ramps AFTER the startup bank is caught) and raise the threshold
                     * above it: thr = max(g_pll_acquire_us, k·dev), capped. Clean legs (dev≈0) keep the tight
                     * 40ms; jittery legs (dev ~150ms) get thr ~450ms so steady-state noise can't re-fire the
                     * acquire (the 0.6.20/0.6.21 limit cycle) — while the big DC startup bank still acquires
                     * while dev is still low. */
                    a->pll_dev += (FFABS(off - a->pll_ema) - a->pll_dev) >> g_pll_dev_shift;
                    int64_t thr = (int64_t)g_pll_acquire_us;
                    /* 1.0.1: TICK-QUANTIZATION dead-band. vlag (the video half of the measured
                     * offset) is quantized to the house video tick — the vring records first-
                     * display output times on tick boundaries — so the measurement has an
                     * irreducible ±1-tick quantum (40ms @25fps ≥ the 40ms base threshold): the
                     * PLL hard-snapped on its own quantization noise (live grids: pad/drop
                     * ~42ms alternating every 12-60s per slot, ~939-1511 ACQUIREs / 22h).
                     * Floor the threshold at 1.5 ticks so a ±1-tick reading can never clear
                     * it. tick_dur_us = the HOUSE tick, wired at setup — one clock for every
                     * slot, the same axis the compositor measures each slot's vlag on. */
                    if (a->tick_dur_us > 0 && 3 * a->tick_dur_us / 2 > thr)
                        thr = 3 * a->tick_dur_us / 2;
                    if ((int64_t)g_pll_noise_k * a->pll_dev > thr) thr = (int64_t)g_pll_noise_k * a->pll_dev;
                    if (thr > 1500000) thr = 1500000;            /* cap the adaptive rise */
                    /* 1.0.1-pre18 #49 REPEATED-ACQUIRE BACKOFF (g_acq_backoff): on audio-erase-
                     * class corruption the measured offset is a FLAT ±step that flips at each
                     * erase — flat defeats the noise-adaptive dev (it only sees jitter) AND the
                     * 3-window sustain (the step is genuinely stable for >2s), so each flip
                     * re-acquired at the refractory rate forever (mv live: ±277ms every ~12s,
                     * slot warbles until restart). Each ACQUIRE inside a 60s window DOUBLES the
                     * threshold (decays one level per acquire-free 60s, cap ×32 / 1.5s abs):
                     * after 2-3 storm acquires the bar outgrows the corruption step and the
                     * storm converges; a legitimate isolated acquire (startup bank, splice)
                     * pays at most one doubling that decays within a minute. TRACK, the
                     * refractory and the tick floor are untouched. PTV_NO_ACQ_BACKOFF reverts. */
                    if (g_acq_backoff && a->acq_backoff > 0) {
                        int lvl = a->acq_backoff -
                                  (int)((av_gettime_relative() - a->acq_last_wc) / 60000000);
                        if (lvl < 0) lvl = 0;
                        a->acq_backoff = lvl;        /* decay applied at read */
                        if (lvl > 0) {
                            thr <<= lvl;
                            if (thr > 1500000) thr = 1500000;
                        }
                    }
                    /* N7: stability-debounce — fire only when the EMA is LARGE *and* FLAT, so Δ is sized
                     * to the FROZEN bank, not one still forming. */
                    if (FFABS(a->pll_ema) > thr &&
                        FFABS(a->pll_ema - a->pll_dbnc_ref) < thr / 4)
                        a->pll_dbnc++;
                    else { a->pll_dbnc = 0; a->pll_dbnc_ref = a->pll_ema; }
                    /* v0.6.18 — acquire on ANY stable large offset, throttled only by the refractory; NOT
                     * gated to the startup window / a disturbance event. tmtg RAV box A/B (v0.6.17) showed
                     * the 5s startup gate left a SLOW-FORMING bank (a slot's +1.1s bank that stabilized
                     * after the window) permanently uncorrected at −1.1s while the three fast-forming slots
                     * acquired+converged. The stability-debounce already rejects noise; the sub-threshold
                     * residuals on the converged slots stay put (won't re-fire); a frozen bank converges in
                     * 1–2 refractory-throttled acquires regardless of WHEN it forms. */
                    int may_acq = a->pll_refractory <= 0 &&
                                  FFABS(a->pll_ema) > thr &&
                                  a->pll_dbnc >= g_pll_acquire_n;
                    /* 1.0.1: SUSTAINED-OFFSET requirement (PTV_ACQ_INSTANT=1 reverts to the old
                     * single-window fire). One completed debounce window (32 large-AND-flat
                     * readings ≈0.7s) is short enough that slow-moving quantization/EMA noise
                     * can hold it once; require the offset to survive N=3 CONSECUTIVE windows
                     * (≈2s continuously large) before snapping. The window counter resets the
                     * moment |ema| falls back under the threshold, so noise excursions never
                     * accumulate credit; the TRACK path below is untouched. */
                    if (may_acq && !g_acq_instant && ++a->pll_acq_win < 3) {
                        a->pll_dbnc = 0; a->pll_dbnc_ref = a->pll_ema;   /* start the next evaluation window */
                        may_acq = 0;
                    }
                    if (FFABS(a->pll_ema) <= thr)
                        a->pll_acq_win = 0;                               /* condition lapsed → windows no longer consecutive */
                    if (may_acq) {
                        a->pll_acq_win = 0;
                        int64_t half = frame_us / 2;                        /* round to NEAREST whole frame (half away from 0) */
                        int64_t dq = ((a->pll_ema + (a->pll_ema < 0 ? -half : half)) / frame_us) * frame_us; /* → residual ≤ ½ frame (≈11ms), vs ≤1 frame for truncation */
                        if (dq != 0) {
                            a->af_applied_us += dq;          /* UNIFORM both directions — cancels the content jump in want */
                            a->pll_ema       -= dq;          /* N1: bumpless credit → no re-fire on the next reading */
                            if (dq < 0) { a->pll_drop = (int)(-dq / frame_us); a->af_acq_drop_us += -dq; }  /* advance: drop content */
                            else        { a->pll_pad  = (int)( dq / frame_us); a->af_acq_pad_us  +=  dq; }   /* delay: pad silence */
                            a->pll_acq_count++;
                            /* 1.0.1-pre17: an ACQUIRE REDEFINES this track's content→label
                             * mapping in one step (af_applied jump + content drop/pad) — the
                             * pre-acquire m_a EMA samples describe a dead mapping, and blending
                             * them in makes R decay for ~2-3min after reality is already fixed
                             * (the birth-C fixture read −2.2s stale). Re-seed from the next
                             * emitted frame — the mirror of the compositor's REANCHOR2 m_v
                             * re-seed; a persistent post-acquire residual (#24's shape) shows
                             * IMMEDIATELY instead of after the EMA settles. The acquire remains
                             * a corrector event (pll_acq_count edge → dwell reset). */
                            a->rs_ma_seed = 0;
                            /* 0.9.18.7: promoted PTV_DIAG→always-on WARNING. An ACQUIRE is a discrete
                             * audio drop/pad (a bank snap, not a TRACK bleed) — rare in normal
                             * operation (startup bank + real disturbances) and already hard
                             * rate-limited by the 12s post-acquire refractory (≤1/12s per track by
                             * construction), so no extra rate limit is added. */
                            /* pre18 #49: an acquire inside the 60s window raises the backoff
                             * (the read above already applied the decay for this frame). */
                            if (g_acq_backoff) {
                                if (a->acq_backoff < 5) a->acq_backoff++;
                                a->acq_last_wc = av_gettime_relative();
                            }
                            av_log(NULL, AV_LOG_WARNING, "[PTV-PLL] a%d(in%d) ACQUIRE %s %"PRId64"ms (ema→%"PRId64"ms applied=%"PRId64"ms #%d backoff=%d)\n",
                                   a->dbg_k, a->dbg_in, dq < 0 ? "drop" : "pad", FFABS(dq) / 1000,
                                   a->pll_ema / 1000, a->af_applied_us / 1000, a->pll_acq_count,
                                   a->acq_backoff);
                        }
                        a->pll_refractory = (int)(g_pll_refractory_us / frame_us);  /* v0.6.21: HARD ~12s refractory (was g_pll_acquire_n ≈0.68s) — breaks the self-excited limit cycle */
                        a->pll_dbnc = 0; a->pll_dbnc_ref = a->pll_ema;
                        if (a->pll_drop > 0) { a->pll_drop--; av_frame_unref(filt); continue; }  /* drop the current frame too */
                    } else if (g_pll_trackup) {
                        /* TRACK — 1.0.1-pre3: RESAMPLER STEER, never labels. pre2's TRACK actuated by
                         * re-stamping output labels (af_applied_us moved `want` every frame): the PCM
                         * stayed byte-clean but the output AAC pts spacing stretched up to +158ms/min
                         * during integration episodes, and PTS-honoring players chased that with their
                         * own rate correctors = audible warble (owner-confirmed at the exact drift-
                         * episode timestamps; rc builds with no TRACK measure EXACTLY 21.333ms spacing
                         * forever). Label re-stamping is a forbidden actuator. Instead accumulate the
                         * same rate-clamped integral trim into af_steer_us, which audio_feed adds to
                         * the pts of frames FED INTO the -af graph (the single-input AVLOCK injection
                         * style): aresample=async realizes it as bounded sample insert/drop (async=1000
                         * → ≤20.8ms/s; our clamp is 10ms/s, safely below, and per-frame steps ~213us
                         * stay far under min_hard_comp so only SOFT compensation runs — inaudible).
                         * Output labels stay perfectly dense between ACQUIREs (af_applied_us now
                         * changes ONLY there — discrete, logged, rare). The pre2 [PTV-TRACKUP]
                         * anti-windup machinery is retired with the label actuator: TRACK no longer
                         * touches `want`, so the monotonic-guard pin cannot eat the integrator.
                         * PTV_NO_PLL_TRACKUP=1 now disables TRACK entirely (acquire-only — the
                         * operators' current production mute keeps its meaning: labels flat, no steer). */
                        int64_t step = a->pll_ema * frame_us / g_pll_tau_us;        /* integral: move ema/τ per frame */
                        int64_t lim  = (int64_t)g_af_rate_us * nb / a->out_rate;    /* rate clamp (us) */
                        if (lim < 1) lim = 1;
                        if (step >  lim) step =  lim;
                        if (step < -lim) step = -lim;
                        a->af_steer_us += step;
                    }
                    if (a->pll_refractory > 0) a->pll_refractory--;   /* ticks on every EMITTED (non-dropped) frame */
                    while (a->pll_pad > 0) {                  /* PAD: emit pending one-shot silence on THIS base before the real frame */
                        AVFrame *s = av_frame_alloc();
                        if (s) {
                            s->nb_samples = nb; s->format = filt->format; s->sample_rate = a->out_rate;
                            av_channel_layout_copy(&s->ch_layout, &filt->ch_layout);
                            if (av_frame_get_buffer(s, 0) >= 0) {
                                int64_t sp = a->af_out_set ? a->af_last_out + nb
                                                           : opts + av_rescale(a->af_applied_us, a->out_rate, 1000000);
                                av_samples_set_silence(s->data, 0, nb, s->ch_layout.nb_channels, s->format);
                                s->pts = sp; a->af_last_out = sp; a->af_out_set = 1;
                                audio_encode_push(a, s); a->out_frames++;
                            }
                            av_frame_free(&s);
                        }
                        a->pll_pad--;
                    }
                }
                want = opts + av_rescale(a->af_applied_us, a->out_rate, 1000000);
                if (a->af_out_set && want < a->af_last_out + nb) { want = a->af_last_out + nb; a->pll_guard_fires++; }
                a->af_last_out = want; a->af_out_set = 1;
                filt->pts = want;
            } else if (a->multiview && g_audio_follow && g_af_anchor) {
                /* B1 (Phase B) — CONTENT-ANCHORED follow. out = opts (async's self-correcting content
                 * target, so async startup over-production is NOT banked — the Phase A root cause) + a
                 * smooth rate-limited offset that tracks the compositor's per-slot lag, so the audio
                 * follows the video DISPLAY. Seeded to the current lag at the first frame (no glitch —
                 * nothing emitted yet); thereafter ≤g_af_rate_us/s so out stays monotonic (opts advances
                 * ~nb/frame ≫ the per-frame offset change). No free counter, no drop/pad/silence.
                 * Converges multiview audio onto the single-input mechanism (both content-anchored). */
                int nb = filt->nb_samples;
                int64_t off = a->house_skew ? *a->house_skew : 0;             /* per-slot lag to follow (us) */
                int64_t gap = off - a->af_applied_us;
                int64_t want;
                if (!a->af_started) { a->af_applied_us = off; a->af_started = 1; }   /* seed at first frame */
                else {
                    int64_t lim = (int64_t)g_af_rate_us * nb / a->out_rate;   /* per-frame ceiling (us) */
                    if (lim < 1) lim = 1;
                    if (gap >  lim) gap =  lim;
                    if (gap < -lim) gap = -lim;
                    a->af_applied_us += gap;
                }
                want = opts + av_rescale(a->af_applied_us, a->out_rate, 1000000);  /* content + smooth follow offset */
                /* MONOTONIC GUARD (B1-fix, v0.6.8) — opts is the async/buffersink output pts; it steps
                 * BACKWARD when h0 is re-anchored forward (P2: opts = buffersink − h0_samp, larger h0 →
                 * smaller opts) or at a source PTS discontinuity. The pre-B1 free counter was monotonic
                 * by construction; content-anchoring lost that → backward out → libfdk_aac "Queue input
                 * is backward in time" + mpegts non-monotonic-DTS → that audio stream stalls and the
                 * interleaver wedges (no output — box-observed). Keep out_a monotonic + frame-spaced; on
                 * a backward step it advances at nb (dense, like the old counter) until opts recovers. */
                if (a->af_out_set && want < a->af_last_out + nb) want = a->af_last_out + nb;
                a->af_last_out = want; a->af_out_set = 1;
                filt->pts = want;
            } else if (a->multiview && g_audio_follow) {
                /* PRE-B1 free-running counter + discrete acquire/drop/pad (A/B via PTV_AF_NO_ANCHOR).
                 * Banks aresample=async's startup over-production → permanent audio-late (Phase A). */
                int nb = filt->nb_samples;
                int64_t ns;
                if (!a->af_started) { a->af_next_pts = opts; a->af_started = 1; }
                {
                    int64_t off = a->house_skew ? *a->house_skew : 0;          /* target correction = per-slot lag (us) */
                    int64_t gap = off - a->af_applied_us;                      /* remaining correction to apply */
                    if (!g_af_pll || FFABS(gap) > g_af_acquire_us) {
                        int64_t d = av_rescale(gap, a->out_rate, 1000000);    /* signed samples */
                        a->af_applied_us = off;
                        if (d < 0) { a->af_drop += -d; a->af_acq_drop_us += -gap; }  /* video ahead → advance → drop */
                        else       { a->af_pad  +=  d; a->af_acq_pad_us  +=  gap; }   /* video behind → delay → pad */
                        if (g_diag && d)
                            av_log(NULL, AV_LOG_INFO, "[PTV-AFOLLOW] a%d(in%d) off=%+"PRId64"ms → acquire %s %"PRId64"ms\n",
                                   a->dbg_k, a->dbg_in, off/1000, d<0?"drop":"pad", FFABS(d)*1000/a->out_rate);
                    } else if (gap != 0) {
                        int64_t lim = (int64_t)g_af_rate_us * nb / a->out_rate;   /* per-frame ceiling from us/s */
                        int64_t step = gap;
                        if (lim < 1) lim = 1;
                        if (step >  lim) step =  lim;
                        if (step < -lim) step = -lim;
                        a->af_nudge_us   += step;
                        a->af_applied_us += step;
                    }
                }
                ns = av_rescale(a->af_nudge_us, a->out_rate, 1000000);         /* smooth nudge → output samples */
                while (a->af_pad >= nb) {                                      /* delay: insert silence */
                    AVFrame *s = av_frame_alloc();
                    if (s) {
                        s->nb_samples = nb; s->format = filt->format; s->sample_rate = a->out_rate;
                        av_channel_layout_copy(&s->ch_layout, &filt->ch_layout);
                        if (av_frame_get_buffer(s, 0) >= 0) {
                            av_samples_set_silence(s->data, 0, nb, s->ch_layout.nb_channels, s->format);
                            s->pts = a->af_next_pts + ns; a->af_next_pts += nb;
                            audio_encode_push(a, s); a->out_frames++;
                        }
                        av_frame_free(&s);
                    }
                    a->af_pad -= nb;
                }
                if (a->af_drop >= nb) { a->af_drop -= nb; av_frame_unref(filt); continue; }  /* advance: skip content */
                filt->pts = a->af_next_pts + ns; a->af_next_pts += nb;        /* continuous + smooth nudge */
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
                int64_t inj    = a->glue_off_us + a->corr.corr_us + hs
                               + ((a->multiview && g_audio_follow && g_avsync_pll)
                                  ? a->af_steer_us : 0);
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
                if (!a->rs_ma_seed) { a->rs_ma_ema = m; a->rs_ma_seed = 1; }
                else {
                    int64_t dv = a->frame_size > 0                        /* EMA ≈ 30s of audio frames */
                               ? 30LL * a->out_rate / a->frame_size : 1406;
                    if (dv < 8) dv = 8;
                    a->rs_ma_ema += (m - a->rs_ma_ema) / dv;
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
             * [PTV-AVSYNC2] — A/V PLL redesign Phase A READ-ONLY measurement probe
             *   (analysis/ptvencoder-avsync-pll-redesign-plan.md §3). Measures the REAL per-track
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
                else if (a->multiview && g_audio_follow && g_avsync_pll)
                    content -= a->af_steer_us + a->corr.corr_us;
                if (vring_lookup(a->vring, content, &out_v, &msrc) == 0) {
                    int64_t vlag   = out_v    - (msrc    - h0_us);   /* video realized output − content (at msrc) */
                    int64_t alag   = out_a_us - (content - h0_us);   /* audio realized output − content (at content) */
                    int64_t paird  = msrc - content;                 /* pairing residual: msrc and content differ when the
                                                                       * video ring hasn't yet composited the audio's content
                                                                       * (deep video prime → composition lags the audio drain
                                                                       * in WALL time; invisible to the player). */
                    int64_t ring   = out_v - out_a_us;               /* raw direct out_v−out_a — CONTAMINATED by paird */
                    int64_t offset = vlag - alag;                    /* = ring − paird: the content-referenced (via shared h0),
                                                                       * pairδ-corrected lip-sync the PLAYER sees. PRIMARY. */
                    if (!a->av_seed) { a->av_vlag_ema = vlag; a->av_alag_ema = alag; a->av_seed = 1; }
                    else { a->av_vlag_ema += (vlag - a->av_vlag_ema) >> 8;   /* slow baseline (~5s @47fps) */
                           a->av_alag_ema += (alag - a->av_alag_ema) >> 8; }
                    /* Latch the latest measurement for the always-on [PTV-AVSYNC] status line (§8). */
                    a->av_offset_us = offset; a->av_vlag_us = vlag; a->av_alag_us = alag; a->av_off_valid = 1;
                    if (g_avsync_probe) {        /* verbose probe (PTV_AVSYNC_PROBE): the full §3.2 decomposition */
                        int64_t per = g_stats_period_us > 0 ? g_stats_period_us : 5000000;
                        int64_t nowp = av_gettime_relative();
                        if (a->av_probe_last == 0) a->av_probe_last = nowp;
                        else if (nowp - a->av_probe_last >= per) {
                            av_log(NULL, AV_LOG_INFO,
                                "[PTV-AVSYNC2] a%d(in%d) offset=%+"PRId64"ms | "
                                "vlag=%+"PRId64"ms(base%+"PRId64" dev%+"PRId64") "
                                "alag=%+"PRId64"ms(base%+"PRId64" dev%+"PRId64") | ring=%+"PRId64"ms pairδ=%+"PRId64"ms"
                                "  [offset<0 = picture ahead of audio; |pairδ| large ⇒ trust offset(=vlag−alag), not ring]\n",
                                a->dbg_k, a->dbg_in, offset / 1000,
                                vlag / 1000, a->av_vlag_ema / 1000, (vlag - a->av_vlag_ema) / 1000,
                                alag / 1000, a->av_alag_ema / 1000, (alag - a->av_alag_ema) / 1000,
                                ring / 1000, paird / 1000);
                            a->av_probe_last = nowp;
                        }
                    }
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
                    if (mv && g_avsync_pll)  /* B3 closed loop: measured offset + integrator state + acquire/guard counts */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) avlag=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "pll[ema=%+"PRId64"ms dev=%"PRId64"ms applied=%+"PRId64"ms steer=%+"PRId64"ms acq=%d guard=%"PRId64" drop=%"PRId64"ms pad=%"PRId64"ms acomp=%"PRId64"]"
                            "  [offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            a->pll_ema / 1000, a->pll_dev / 1000, a->af_applied_us / 1000, a->af_steer_us / 1000,
                            a->pll_acq_count, a->pll_guard_fires,
                            a->af_acq_drop_us / 1000, a->af_acq_pad_us / 1000, a->acomp_cnt);
                    else if (mv)         /* multiview (open-loop B1): lip-sync + measured offset + the per-slot actuator state */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) avlag=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "house_skew=%+"PRId64"ms applied=%+"PRId64"ms trk=%+"PRId64"ms nudge=%+"PRId64"ms "
                            "acq[drop=%"PRId64"ms pad=%"PRId64"ms]  [avlag>0 / offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            lag / 1000, a->af_applied_us / 1000, (lag - a->af_applied_us) / 1000,
                            a->af_nudge_us / 1000, a->af_acq_drop_us / 1000, a->af_acq_pad_us / 1000);
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
             * single-input output of the same feed to attribute source-side offset). */
            if (g_diag) {
                int64_t now = av_gettime_relative();
                if (a->dbg_first_out == AV_NOPTS_VALUE) a->dbg_first_out = opts;
                if (now - a->dbg_diag_last >= 1000000) {
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


/* Feed one h0-anchored decoded audio frame into the -af graph (or swr fallback). */
static int audio_feed(AudioState *a, AVFrame *frame)
{
    uint8_t **out = NULL;
    int out_max, got, ret = 0;
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
        a->dbg_last_src = frame->best_effort_timestamp;   /* probe: latest fed source pts */

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
        if (!same_pending) {                       /* new candidate → start hysteresis, drop this frame */
            a->afmt_pending_rate = frame->sample_rate;
            a->afmt_pending_fmt  = frame->format;
            av_channel_layout_uninit(&a->afmt_pending_chl);
            av_channel_layout_copy(&a->afmt_pending_chl, &frame->ch_layout);
            a->afmt_stable = 1;
            return 0;
        }
        if (++a->afmt_stable < PTV_AFMT_HYSTERESIS)
            return 0;                              /* still settling — drop (downstream can't take the change) */
        {   /* confirmed: rebuild for the new params (a->dec already reflects them) */
            char ochl[64], nchl[64], tchl[64];
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
            a->fg_in_rate = a->dec->sample_rate;   /* the path is now configured for these */
            a->fg_in_fmt  = a->dec->sample_fmt;
            av_channel_layout_uninit(&a->fg_in_chl);
            av_channel_layout_copy(&a->fg_in_chl, &a->dec->ch_layout);
            av_channel_layout_uninit(&a->afmt_pending_chl);
            a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
            /* 1.0.1-pre19 #42 hardening: the rebuild above configured the path from a->dec,
             * but THIS frame (the confirming one) falls through to be fed below. During a
             * broken-AAC phase the decoder context can diverge from the frames it emitted
             * earlier (per-frame reconfig / queued pre-h0 replay), and swr_convert reads as
             * many planes as ITS configured input layout has — a frame with fewer channels
             * than a->dec's layout means reads through nonexistent plane pointers. If the
             * frame no longer matches the just-configured input params, drop it; the next
             * frame re-enters the normal AFMT detection. */
            if (frame->sample_rate != a->fg_in_rate ||
                frame->format      != a->fg_in_fmt  ||
                av_channel_layout_compare(&frame->ch_layout, &a->fg_in_chl)) {
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-AFMT] a%d(in%d) decoder context diverged from the confirmed frame "
                       "params during rebuild — frame dropped, detection re-arms\n",
                       a->dbg_k, a->dbg_in);
                return 0;
            }
        }
    } else if (a->afmt_stable) {                   /* params returned to normal → transient filtered out */
        a->afmt_stable = 0;
        av_channel_layout_uninit(&a->afmt_pending_chl);
        a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
    }

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
        if (g_aglue_ms > 0 && frame->pts != AV_NOPTS_VALUE) {
            int64_t raw_us = av_rescale_q(frame->pts, a->ist_tb, AV_TIME_BASE_Q);
            int64_t now_wc = av_gettime_relative();
            int fill_resumed = 0;   /* 1.0.1-pre15 §3: first REAL frame after an NBS fill phase */
            if (g_glueclass && a->nbs_fill_active && !a->nbs_feeding) {
                a->nbs_fill_active = 0;
                a->nbs_last_wall_us = 0;
                a->nbs_carry_us     = 0;
                fill_resumed = 1;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-ADISC] a%d(in%d) real frames resumed — silence-fill released after "
                       "%d quanta; resume step classified below\n",
                       a->dbg_k, a->dbg_in, a->nbs_fills);
            }
            /* 1.0.1-pre18 #50 (E5 net, g_glueveto): a LAYERA-flush relabel is INVISIBLE here —
             * the labels arrive already-shifted, so a GAP-pad's RETURN leg consumed by a flush
             * erase left the pad's inserted silence permanently baked (pad + erase = double
             * application; the ledger only ever saw arriving label steps). The demux publishes
             * each flush's per-track label shift; a shift matching an open pad IS that return
             * leg, erased at the packet layer — counter-apply it at the graph door instead:
             * glue_off_us -= pad steps the door labels back by the pad, aresample=async
             * hard-drops the inserted silence, and the sensor's inj accounting rides glue_off
             * exactly as for an arriving relabel verdict. A young publish that matches no open
             * pad stays PENDING (the pad may still be in flight through audio_q) and re-scans
             * per frame until the pad TTL retires it. */
            if (g_glueveto && g_glueclass && a->dbg_k >= 0 && a->dbg_k < PTV_MAX_AUDIO) {
                int64_t fwc = atomic_load_explicit(&g_flush_relab_wc[a->dbg_k], memory_order_acquire);
                if (fwc && fwc != a->flush_relab_seen_wc) {
                    if (now_wc - fwc > PTV_GLUE_PAD_TTL_US)
                        a->flush_relab_seen_wc = fwc;          /* stale: retire unmatched */
                    else {
                        int64_t fstep = atomic_load_explicit(&g_flush_relab_step[a->dbg_k],
                                                             memory_order_relaxed);
                        int pf;
                        for (pf = 0; pf < PTV_GLUE_PAD_LED; pf++) {
                            int64_t pp = a->pad_led_us[pf];
                            if (pp > 0 && now_wc - a->pad_led_wc[pf] <= PTV_GLUE_PAD_TTL_US &&
                                llabs(fstep - pp) <= FFMAX(80000, pp / 4)) {
                                a->glue_off_us -= pp;
                                a->pad_led_us[pf] = 0;         /* consumed */
                                ptv_pad_pub(a);
                                a->glue_events++;
                                a->flush_relab_seen_wc = fwc;
                                a->pend_comp_us = -pp;         /* §2.4 tripwire: the drop must realize */
                                a->pend_comp_wc = now_wc;
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-AGLUE] a%d(in%d) LAYERA flush relabel %+"PRId64"ms matches an "
                                       "open GAP-pad +%"PRId64"ms — the pad's return leg was erased at the "
                                       "packet layer; counter-applied at the graph door (glue total "
                                       "%+"PRId64"ms) so the pad's silence is dropped (PTV_NO_GLUEVETO "
                                       "reverts)\n",
                                       a->dbg_k, a->dbg_in, fstep / 1000, pp / 1000,
                                       a->glue_off_us / 1000);
                                break;
                            }
                        }
                    }
                }
            }
            if (a->glue_raw_last_us != AV_NOPTS_VALUE) {
                int64_t step = raw_us - (a->glue_raw_last_us + a->glue_raw_dur_us);
                if (llabs(step) > (int64_t)g_aglue_ms * 1000) {
                    int64_t wall_gap = now_wc - a->glue_wall_last_us;
                    /* 1.0.1-pre5 (D1) shared-flush expected-step handshake: the demux flush
                     * registered the A-vs-V mismatch it routed into THIS track's labels
                     * (ptv_pair_expect). A matching arriving step is a REAL alignment step —
                     * it must be APPLIED (aresample converges content onto the new labels),
                     * never relabel-ERASED: erasing it re-bakes the mismatch the shared offset
                     * existed to avoid (the fx-mir2 class, backward steps in (-1000,-500)ms).
                     * Match = within [-PTV_PAIR_EXPECT_LO_US, +PTV_PAIR_EXPECT_HI_US] of the
                     * registered value before the deadline (see ptvencoder.h for the window
                     * rationale); consumed one-shot. Plain source steps (no registration, or
                     * value/deadline miss) keep every pre-existing rule byte-identical. */
                    int exp_hit = 0, exp_late = 0;
                    int64_t exp_step = 0;
                    if (a->glue_exp_dl) {
                        int64_t dl = atomic_load_explicit(a->glue_exp_dl, memory_order_acquire);
                        if (dl) {
                            int vmatch;
                            exp_step = atomic_load_explicit(a->glue_exp_step, memory_order_relaxed);
                            vmatch = step - exp_step >= -PTV_PAIR_EXPECT_LO_US &&
                                     step - exp_step <=  PTV_PAIR_EXPECT_HI_US;
                            if (now_wc <= dl && vmatch) {
                                atomic_store_explicit(a->glue_exp_dl, 0, memory_order_relaxed);  /* consumed */
                                exp_hit = 1;
                            } else if (now_wc > dl) {
                                /* 1.0.1-pre15 (#33 b3): a VALUE match after the TTL is still the
                                 * flush-routed step — a deep bank can legally hold it past any
                                 * fixed TTL (review-2 F1), and falling back to the relabel-erase
                                 * re-bakes the mismatch (the D1 defect, flushless at the point of
                                 * damage). The value window (±[-250,+500]ms of a >500ms step) is
                                 * the real collision guard; consume late, say so. */
                                if (g_glueclass && vmatch) {
                                    atomic_store_explicit(a->glue_exp_dl, 0, memory_order_relaxed);
                                    exp_hit = exp_late = 1;
                                } else
                                    atomic_store_explicit(a->glue_exp_dl, 0, memory_order_relaxed);  /* expired */
                            }
                        }
                    }
                    /* 1.0.1-pre15 §2.2 rule 3a (#33, closes b1): an UNREGISTERED backward step
                     * matching a recent still-open GAP-pad is the pad's RETURN leg — APPLYING it
                     * (aresample drops = unwinds the pad's inserted silence) is the round-trip
                     * cancel; erasing it bakes the pad (rr14 A3: +150/−150 = REAL −150ms). A
                     * coincidental independent relabel of matching size converges to the same
                     * content end-state through the drop (doc §2.6 risk note; G11). */
                    int pad_cancel = 0;
                    int64_t pc_pad = 0;
                    if (g_glueclass && !exp_hit && step < 0 && !fill_resumed) {
                        int pi;
                        for (pi = 0; pi < PTV_GLUE_PAD_LED; pi++) {
                            int64_t p = a->pad_led_us[pi];
                            if (p > 0 && now_wc - a->pad_led_wc[pi] <= PTV_GLUE_PAD_TTL_US &&
                                llabs(step + p) <= FFMAX(80000, p / 4)) {
                                a->pad_led_us[pi] = 0;   /* consumed */
                                ptv_pad_pub(a);
                                pad_cancel = 1;
                                pc_pad = p;
                                break;
                            }
                        }
                    }
                    char sn[48];   /* 1.0.1-pre8 (d): self-shed honesty note ("" when nothing shed) */
                    if (llabs(step) > (int64_t)g_aglue_max_ms * 1000) {
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms above %dms cap — left to the discontinuity layer%s%s%s%s\n",
                               a->dbg_k, a->dbg_in, step / 1000, g_aglue_max_ms,
                               exp_hit ? " (matches the shared-flush expected step — aresample converges it)" : "",
                               exp_late ? " [late match — TTL had expired]" : "",
                               pad_cancel ? " (cancels an open GAP-pad — round-trip; aresample drops)" : "",
                               ptv_self_shed_note(a, sn, sizeof sn));
                    } else if (exp_hit) {
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms matches the shared-flush expected step %+"PRId64"ms "
                               "— REAL A-vs-V alignment step, APPLIED (aresample converges; not erased)%s%s\n",
                               a->dbg_k, a->dbg_in, step / 1000, exp_step / 1000,
                               exp_late ? " [late match — TTL had expired]" : "",
                               ptv_self_shed_note(a, sn, sizeof sn));
                    } else if (pad_cancel) {
                        a->glue_events++;   /* corrector freeze set (§5 rule 3) */
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms — pad round-trip cancelled "
                               "(open GAP-pad +%"PRId64"ms): APPLIED, aresample drops the pad's inserted "
                               "silence (not erased)%s\n",
                               a->dbg_k, a->dbg_in, step / 1000, pc_pad / 1000,
                               ptv_self_shed_note(a, sn, sizeof sn));
                    } else if (fill_resumed && step < 0) {
                        /* §3 resume-anchor: the overlap is OUR OWN synthesized silence — dropping
                         * it is free; erasing would relabel real content onto the fill. */
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) fill-resume overlap %+"PRId64"ms — synthesized "
                               "region dropped (aresample), not erased%s\n",
                               a->dbg_k, a->dbg_in, step / 1000,
                               ptv_self_shed_note(a, sn, sizeof sn));
                    } else {
                        /* 0.9.18: verdict-LOG rate limit. An Azorse-class label flood (source labels
                         * striding ~6x content = one verdict per frame, ~8 lines/s indefinitely) must
                         * not drown the log. Verdicts still APPLY to every frame; only the per-event
                         * lines are capped: 10 per 10s window, then one summary as the window rolls. */
                        int allow;
                        if (now_wc - a->glue_log_win_us >= 10000000) {
                            if (a->glue_supp_n)
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-AGLUE] a%d(in%d) %d more label steps (net %+"PRId64"ms) suppressed in last 10s — source label flood, verdicts still applied\n",
                                       a->dbg_k, a->dbg_in, a->glue_supp_n, a->glue_supp_net_us / 1000);
                            a->glue_log_win_us = now_wc;
                            a->glue_log_win_n  = 0;
                            a->glue_supp_n     = 0;
                            a->glue_supp_net_us = 0;
                        }
                        allow = a->glue_log_win_n < 10;
                        if (allow) a->glue_log_win_n++;
                        else     { a->glue_supp_n++; a->glue_supp_net_us += step; }
                        if (step > 0) {
                            if (allow)
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms (wall gap %"PRId64"ms) — GAP; aresample pads%s\n",
                                       a->dbg_k, a->dbg_in, step / 1000, wall_gap / 1000,
                                       ptv_self_shed_note(a, sn, sizeof sn));
                        } else {
                            a->glue_off_us -= step;
                            a->glue_events++;
                            if (allow)
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms (wall gap %"PRId64"ms) — backward RELABEL erased (glue total %+"PRId64"ms, event %d)%s\n",
                                       a->dbg_k, a->dbg_in, step / 1000, wall_gap / 1000,
                                       a->glue_off_us / 1000, a->glue_events,
                                       ptv_self_shed_note(a, sn, sizeof sn));
                        }
                    }
                    /* 1.0.1-pre15 #33 verdict bookkeeping (every branch above except the
                     * backward RELABEL erase left the step IN the labels for aresample to
                     * converge — GAP-pad, FLUSH-APPLY, pad-cancel, above-cap stand-aside): */
                    if (g_glueclass) {
                        int erased_here = step < 0 && !exp_hit && !pad_cancel && !fill_resumed &&
                                          llabs(step) <= (int64_t)g_aglue_max_ms * 1000;
                        if (!erased_here) {
                            /* §2.4 realization tripwire: hard comp is instantaneous by design —
                             * checked against the pre11 slip probe ~2s from now (audio_drain_fg). */
                            a->pend_comp_us = step;
                            a->pend_comp_wc = now_wc;
                        }
                        if (step > 0 && !exp_hit &&
                            wall_gap < step / 2 + FFMAX(a->glue_cad_us, 40000)) {
                            /* E5 pad ledger: an open GAP-pad awaiting a possible return leg (3a).
                             * Registered (flush-routed) forward steps are alignment, not gaps —
                             * they never enter the ledger. rr15 R2: neither do E3-CORROBORATED
                             * REAL-gap pads — silence that filled genuinely missing time shifted
                             * nothing, so there is no round trip to unwind, and cancelling against
                             * such a pad deletes real content (fx-rr15-a3: a real 400ms gap's pad +
                             * a coincidentally-sized In-Touch both-stream backward relabel deleted
                             * 405ms = re-opened the 0.9.16.4 audio-early accumulator). Real gap ⇒
                             * this frame was wall-ABSENT ≈ the step on top of the normal arrival
                             * cadence (PES-burst period, EMA'd below); a relabel/flood pad arrives
                             * FLOWING (wall_gap ≈ one cadence ≪ step/2 + cadence). Only the
                             * flowing (splice-suspect, b1/Azorse-class) pads are candidates. */
                            int slot = a->pad_led_n % PTV_GLUE_PAD_LED;
                            a->pad_led_us[slot] = step;
                            a->pad_led_wc[slot] = now_wc;
                            a->pad_led_n++;
                            ptv_pad_pub(a);
                        }
                        if (!erased_here && llabs(step) > 10 * AV_TIME_BASE)
                            /* owner call 2026-07-18 (Q5): >10s convergences on healthy sources stay
                             * UNBOUNDED (invariant mandate — PATRIOT 30.8s is locked in), with an
                             * operator alert as the only escalation. */
                            av_log(NULL, AV_LOG_ERROR,
                                   "[PTV-AGLUE] a%d(in%d) %+"PRId64"s audio content convergence in flight "
                                   "(>10s) — unbounded by mandate; verify source alignment if unexpected\n",
                                   a->dbg_k, a->dbg_in, step / AV_TIME_BASE);
                    }
                } else if (g_glueclass) {
                    /* rr15 R2: quiet frame — track the normal ARRIVAL CADENCE (EMA of nonzero
                     * wall gaps between fed frames = the PES-burst period; intra-burst frames
                     * arrive within µs and are skipped so the EMA reads the burst period, not
                     * the frame duration). This is the E3 baseline the pad-ledger gate above
                     * subtracts: a real gap's wall absence rides ON TOP of one cadence. */
                    int64_t wg = now_wc - a->glue_wall_last_us;
                    if (wg > 5000 && wg < 2000000) {
                        if (!a->glue_cad_us) a->glue_cad_us = wg;
                        else                 a->glue_cad_us += (wg - a->glue_cad_us) / 8;
                    }
                }
            }
            a->glue_raw_last_us  = raw_us;
            a->glue_raw_dur_us   = frame->sample_rate > 0 ?
                av_rescale(frame->nb_samples, 1000000, frame->sample_rate) : 0;
            a->glue_wall_last_us = now_wc;
            if (a->glue_off_us)
                frame->pts += av_rescale_q(a->glue_off_us, AV_TIME_BASE_Q, a->ist_tb);
        }
        if (g_avlock && a->house_skew && !(a->multiview && g_audio_follow) && frame->pts != AV_NOPTS_VALUE) {
            int64_t sk = *a->house_skew;
            if (sk) frame->pts += av_rescale_q(sk, AV_TIME_BASE_Q, a->ist_tb);
        }
        /* 1.0.1-pre3: multiview audio-follow PLL — inject TRACK's accumulated steer into the
         * graph-input pts (the AVLOCK style above), so aresample=async realizes the trim as
         * bounded content stretch/squeeze while output labels stay dense. Written by the same
         * thread in audio_drain_fg (≤ ~213us/frame, rate-clamped), so the input pts stream
         * stays strictly monotonic (frame spacing ~21333us dwarfs any per-frame steer delta)
         * and never trips min_hard_comp on its own. */
        if (a->multiview && g_audio_follow && g_avsync_pll && a->af_steer_us &&
            frame->pts != AV_NOPTS_VALUE)
            frame->pts += av_rescale_q(a->af_steer_us, AV_TIME_BASE_Q, a->ist_tb);
        /* 1.0.1-pre14 steer bus (§5 rule 1): the corrector's cumulative trim, injected at the
         * SAME single graph-door site as glue_off/house_skew/af_steer_us — aresample=async
         * realizes it as bounded SOFT compensation (rate-clamped 2ms/s upstream in
         * rscorr_update; per-frame deltas ~43µs, three orders under min_hard_comp — the
         * ACOMP proxy below monitors the summed stream and is the click tripwire for any
         * mis-sized bus term). corr_us==0 (default-off / parked-at-zero) skips = byte-inert. */
        if (a->corr.corr_us && frame->pts != AV_NOPTS_VALUE)
            frame->pts += av_rescale_q(a->corr.corr_us, AV_TIME_BASE_Q, a->ist_tb);
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
static int audio_anchor_and_feed(AudioState *a, AVFrame *frame, int64_t h0)
{
    int64_t ts = frame->best_effort_timestamp;
    if (ts == AV_NOPTS_VALUE) return 0;
    if (!a->pts_set) {
        int64_t house_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q) - h0;
        int64_t fill_us  = 0;
        if (house_us < 0) { a->anchor_drop_pre++; return 0; }   /* audio precedes video anchor: drop */
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
        a->next_pts = av_rescale(house_us - fill_us, a->out_rate, 1000000);
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

static int audio_push(AudioState *a, AVFrame *frame)
{
    int64_t ts = frame->best_effort_timestamp;
    int ret = 0, i;

    a->in_frames++;

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
            if (ret >= 0) ret = audio_anchor_and_feed(a, a->aq_pending[i], h0);
            av_frame_free(&a->aq_pending[i]);
        }
        a->aq_npending = 0;
        if (ret < 0) return ret;
        return audio_anchor_and_feed(a, frame, h0);
    }
    if (ts == AV_NOPTS_VALUE) return 0;   /* 1.0.1: un-stamped frame (e.g. a garbage-tail decode next to a
                                           * tolerated [PTV-ADEC] error) cannot be content-anchored; encoding
                                           * it hands the muxer a timestamp-less packet = EINVAL mux wedge.
                                           * Drop it — same rule the pre-anchor path applies. */
    a->dbg_last_src = ts;                 /* probe: latest source audio pts */
    return audio_feed(a, frame);
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
static void adec_reopen(AudioState *a)
{
    const AVCodec *codec = a->dec ? a->dec->codec : NULL;
    AVCodecContext *nd;

    a->wd_frame_us = av_gettime_relative();   /* re-arm the window whatever happens below */
    a->wd_pkts     = 0;
    if (!codec || !a->ist_par)
        return;
    nd = avcodec_alloc_context3(codec);
    if (!nd)
        return;
    /* pre17 R1: owned codecpar/timebase copies — the mv demux reopen may have closed the
     * AVFormatContext an AVStream* would have pointed into. */
    avcodec_parameters_to_context(nd, a->ist_par);
    nd->pkt_timebase = a->ist_pkt_tb;
    if (avcodec_open2(nd, codec, NULL) < 0) {
        avcodec_free_context(&nd);
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ADECWD] a%d(in%d) decoder reopen FAILED — keeping the old context, retrying in %ds\n",
               a->dbg_k, a->dbg_in, (int)(PTV_ADECWD_US / 1000000));
        return;
    }
    avcodec_free_context(&a->dec);
    a->dec = nd;
    a->dec_reopens++;
    av_log(NULL, AV_LOG_WARNING,
           "[PTV-ADECWD] a%d(in%d) no decoded frames for %ds with packets arriving — decoder reopened "
           "(#%d, errs=%"PRId64"); anchor preserved, aresample absorbs the gap\n",
           a->dbg_k, a->dbg_in, (int)(PTV_ADECWD_US / 1000000), a->dec_reopens, a->dec_errs);
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
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-ADISC] a%d(in%d) silence-fill ENGAGED — demux is corrupt-discarding this "
               "track's packets with nothing decoding; synthesizing dense silence until real "
               "frames resume\n",
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
        ret = av_thread_message_queue_recv(a->audio_q, &pkt, 0);
        if (ret < 0) break;
        if (pkt && (pkt->flags & PTV_PKT_FLAG_NBS_FILL)) {
            /* pre15 §3 FILL sentinel — NOT a packet: must not feed the decoder and must not
             * advance wd_pkts (ADECWD would read "packets arriving, nothing decodes" and churn
             * a reopen every 45s of a fill phase). */
            av_packet_free(&pkt);
            nbs_fill_quantum(a);
            continue;
        }
        a->wd_pkts++;
        if (!a->wd_frame_us) a->wd_frame_us = av_gettime_relative();   /* seed at first packet */
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
            ret = audio_push(a, frame);
            av_frame_unref(frame);
            if (ret < 0) goto done;
        }
        if (g_adecwd && a->wd_pkts > 0 &&
            av_gettime_relative() - a->wd_frame_us > PTV_ADECWD_US)
            adec_reopen(a);
    }
    /* flush decoder -> resampler/filtergraph -> encoder */
    avcodec_send_packet(a->dec, NULL);
    while (avcodec_receive_frame(a->dec, frame) >= 0) { audio_push(a, frame); av_frame_unref(frame); }
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

    av_channel_layout_describe(&adec->ch_layout, chl, sizeof chl);
    snprintf(args, sizeof args,
             "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
             tb.num, tb.den, adec->sample_rate,
             av_get_sample_fmt_name(adec->sample_fmt), chl);
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
    if (a->fg_swr)
        av_log(NULL, AV_LOG_INFO, "[PTV-SWRDELAY] sensor armed (aresample SwrContext found)\n");

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

