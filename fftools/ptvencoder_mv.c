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
#include "libswresample/swresample.h"

#include "ptvencoder.h"

static int     g_h0_reanchor_ms = 120;   /* trigger (ms); internalized 0.9.18.7 (was PTV_H0_REANCHOR_MS) */
/* PTV_DIAG: compositor publishes its current VIDEO output time (us) so the audio probe can
 * log a synchronized per-track audio-minus-video offset. Temporary diagnostic. */
static _Atomic int64_t g_vout_us;
/* fill a planar-YUV / RGB frame with black (held cell for a not-yet-arrived or
 * stale multiview slot): luma 16, chroma 128, RGB 0. */
static void ptv_fill_black(AVFrame *f)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(f->format);
    int p;
    if (!d) return;
    for (p = 0; p < AV_NUM_DATA_POINTERS && f->data[p]; p++) {
        int rgb = !!(d->flags & AV_PIX_FMT_FLAG_RGB);
        int chroma = !rgb && (p == 1 || p == 2);
        int val = rgb ? 0 : (chroma ? 128 : 16);
        int h = (chroma) ? AV_CEIL_RSHIFT(f->height, d->log2_chroma_h) : f->height;
        memset(f->data[p], val, (size_t)f->linesize[p] * h);
    }
}

static AVFrame *make_black_frame(AVCodecContext *vdec)
{
    AVFrame *f = av_frame_alloc();
    if (!f) return NULL;
    f->format = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
    f->width  = vdec->width;
    f->height = vdec->height;
    if (av_frame_get_buffer(f, 0) < 0) { av_frame_free(&f); return NULL; }
    ptv_fill_black(f);
    return f;
}

/* Multiview compositor thread — the video house clock. Each wall-paced tick:
 * sample-and-hold each input's latest decoded frame (dup-hold a stale one;
 * black-slate a missing/long-stale one), feed all N buffersrcs with pts = tick
 * (so the mosaic's xstack/overlay framesync pairs them and never waits), pull
 * each rung's composited+scaled frame, stamp it consecutive-CFR, and hand it to
 * that rung's frame_q. Publishes per-input house_skew for the slot's audio lock.
 * A late/dead/frozen slot NEVER stalls the mosaic (R-MV-6). */
/* v0.9.12: house tick T on the EXACT measurement axis. The composited video PTS is a tick
 * counter interpreted at av_inv_q(out_fps) — exact — but all output-time MEASUREMENTS used
 * T x integer tick_dur_us, +10ppm fast at NTSC rates → the slot audio followers regulated
 * audio onto the wrong axis (enforced drift). Fixed by the exact-rational axis below. */
static int64_t mv_tick_us(CompositorCtx *c, int64_t t)
{
    if (g_mv_exacttick && c->out_fps.num > 0)
        return av_rescale(t, 1000000LL * c->out_fps.den, c->out_fps.num);   /* exact-rational tick */
    return t * c->tick_dur_us;            /* fallback: integer tick = faithful old behavior */
}

void *compositor_thread(void *arg)
{
    CompositorCtx *c = arg;
    int n = c->n_input, R = c->n_rung, k, r;
    AVFrame *blackf[PTV_MAX_INPUT] = {0};
    AVFrame *last[PTV_MAX_INPUT] = {0};       /* last frame popped per input (dup source) */
    AVFrame *pending[PTV_MAX_INPUT] = {0};    /* content-clamp: a frame popped but held back because its
                                               * content-time leads the house clock (gap); shown when out catches up */
    int64_t last_fresh_us[PTV_MAX_INPUT] = {0};
    int64_t skew_us[PTV_MAX_INPUT] = {0};     /* per-slot audio skew = accumulated dup-hold ticks */
    int64_t lag_true_us[PTV_MAX_INPUT] = {0}; /* PTV_DIAG: TRUE uncapped signed video lag (output−content); when
                                               * this >> skew_us the 250ms cap is saturating = audio can't follow */
    int      slated[PTV_MAX_INPUT] = {0};     /* slot is/was black-slated (outage) since last fresh frame */
    int64_t res_due_us[PTV_MAX_INPUT] = {0};  /* v0.9.13 residence: house-time the next pop is allowed (0 = immediate) */
    int64_t res_ema_us[PTV_MAX_INPUT];        /* v0.9.13: EMA of content deltas = the slot's smoothed cadence */
    int64_t res_src_us[PTV_MAX_INPUT];        /* v0.9.13: content time (us) of the last popped frame */
    int64_t pd_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: residence holds (correct cadence, not starvation) */
    int64_t sv_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: genuine starvation dups (due but buffer empty) */
    int64_t md_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: decimation drops (multiple frames due in one tick) */
    int     res_occ_tgt = 4;                  /* v0.9.13: occupancy-servo target (set from the primed depth) */
    /* AUDIO-FOLLOW (Option A) latch: average the per-slot lag over a startup window (past the
     * lossy join) and latch a STABLE signed offset, published to house_skew for the audio's
     * one-time deterministic correction. Re-latched on outage return. */
    int64_t  af_off[PTV_MAX_INPUT] = {0};     /* audio-follow: continuously-smoothed per-slot lag (EMA, us) */
    int64_t  af_t0[PTV_MAX_INPUT]  = {0};     /* tick+1 of this slot's first real frame (0 = unseeded) */
    int      h0_logged[PTV_MAX_INPUT] = {0};  /* P0 diag: one-shot PTV-H0 per slot at first display */
    int64_t  r2_win_us[PTV_MAX_INPUT] = {0};  /* 0.9.18.7 [PTV-REANCHOR2] log rate-limit: window start (wall) */
    int      r2_win_n[PTV_MAX_INPUT] = {0};   /*   lines printed this window */
    int      r2_supp[PTV_MAX_INPUT] = {0};    /*   events suppressed this window */
    int64_t  r2_supp_us[PTV_MAX_INPUT] = {0}; /*   net h0 shift suppressed this window (us) */
    int64_t  r2_skring[PTV_MAX_INPUT][5] = {{0}};  /* 1.0.1 REANCHOR2 debounce: last 5 evaluated sk samples per slot */
    unsigned r2_cond[PTV_MAX_INPUT] = {0};    /*   bitmask: which of the last 5 evaluated ticks held sk < -thr */
    int      r2_pos[PTV_MAX_INPUT] = {0};     /*   ring write position */
    int      done_in[PTV_MAX_INPUT] = {0};
    int64_t rung_pts[PTV_MAX_RUNG] = {0};
    /* 1.0.1-pre16 mv sensor port: per-slot video-side EMA state (the compositor is the ONE
     * writer of g_rsx.mv_ema/mv_wall[*] on mv — the single-input master-rung mirror). τ ≈ 30s
     * of ticks, divisor verbatim from the single-input sensor (ptvencoder_clock.c). */
    int64_t rs_mv_ema[PTV_MAX_INPUT] = {0};
    int     rs_mv_seed[PTV_MAX_INPUT] = {0};
    int64_t rs_mv_div;
    AVFrame *filt = av_frame_alloc();
    int64_t tick = 0, wall0 = 0;
    /* v0.9.18.2 M4: use the RESOLVED preroll (CushionPlan) — this getenv re-read with a 350ms
     * fallback predated v0.9.1's genlock default (1000ms) and was never updated, so plain mv
     * invocations primed with roughly a THIRD of the intended startup cushion (plan §3.6).
     * resolve_cushions() runs in transcode() before this thread starts; an explicit
     * PTV_PREROLL_MS still wins exactly as before (it feeds the resolved value). */
    int preroll_ms = g_cp.preroll_ms;
    int n_prime = (preroll_ms > 0 && c->tick_dur_us > 0) ? (int)((int64_t)preroll_ms * 1000 / c->tick_dur_us) : 0;
    int64_t diag_t0 = av_gettime_relative(), diag_last = diag_t0;
    int64_t stat_last = diag_t0, stat_prev = 0;

    if (!filt) goto done;
    rs_mv_div = c->tick_dur_us > 0 ? 30000000 / c->tick_dur_us : 750;
    if (rs_mv_div < 8) rs_mv_div = 8;
    if (n_prime > g_frameq_cap - 8) n_prime = g_frameq_cap - 8;
    if (n_prime < 0) n_prime = 0;
    res_occ_tgt = n_prime > 4 ? n_prime : 4;   /* v0.9.13 servo target = the primed jitter depth,
                                                * clamped below the pressure valve (a deep PTV_PREROLL_MS
                                                * would otherwise put the target INSIDE the bypass zone
                                                * and the two would fight) */
    if (res_occ_tgt > g_frameq_cap - g_frameq_cap / 4 - 8)
        res_occ_tgt = g_frameq_cap - g_frameq_cap / 4 - 8;
    for (k = 0; k < n; k++) blackf[k] = make_black_frame(c->inputs[k].vdec);
    for (k = 0; k < n; k++) {                 /* v0.9.13: seed cadence at the house rate; EMA adapts in ~16 frames.
                                               * Known limit: a source FASTER than ~2x house sits outside the
                                               * acceptance band and keeps the house-rate estimate — the pressure
                                               * valve then bounds it to today's overflow-decimation behavior. */
        res_ema_us[k] = mv_tick_us(c, 1);
        res_src_us[k] = AV_NOPTS_VALUE;
    }

    /* preroll: prime every input's jitter buffer to ~PTV_PREROLL_MS so bursty
     * decode delivery has a cushion (no startup dup storm) and the mosaic starts
     * with every cell live. A never-arriving input is left to its black cell. */
    {
        int64_t t0 = av_gettime_relative();
        for (;;) {
            int ready = 0, eofall = 1;
            for (k = 0; k < n; k++) {
                pthread_mutex_lock(&c->inputs[k].hold.lock);
                if (!c->inputs[k].hold.eof) eofall = 0;
                pthread_mutex_unlock(&c->inputs[k].hold.lock);
                if (av_thread_message_queue_nb_elems(c->inputs[k].hold.q) >= (n_prime > 0 ? n_prime : 1)) ready++;
            }
            if (ready == n || eofall) break;
            if (av_gettime_relative() - t0 > 3000000) break;   /* 3s: start with what's there */
            av_usleep(5000);
        }
        /* v0.9.13: trim any startup BACKLOG down to the primed depth (live+residence only). A join
         * can dump seconds of banked frames at once (deep UDP socket buffer read in one burst,
         * inputs that filled while the preroll waited on a slower sibling) — a jitter buffer must
         * ACQUIRE at its target depth; the excess is stale latency that pinned the queue at the
         * pressure valve and ratcheted the residence schedule (measured: occ=119 at tick 0). The
         * legacy path (PTV_NO_RESIDENCE) keeps the old catch-up-by-consumption behavior. */
        if (c->live && g_mv_residence) {
            for (k = 0; k < n; k++) {
                AVFrame *tf; int trimmed = 0;
                while (av_thread_message_queue_nb_elems(c->inputs[k].hold.q) > res_occ_tgt &&
                       av_thread_message_queue_recv(c->inputs[k].hold.q, &tf, AV_THREAD_MESSAGE_NONBLOCK) >= 0) {
                    av_frame_free(&tf); trimmed++;
                }
                if (trimmed)
                    av_log(NULL, AV_LOG_INFO, "[PTV-RES] in%d startup backlog trimmed %d frames (keep %d)\n",
                           k, trimmed, res_occ_tgt);
            }
        }
        wall0 = av_gettime_relative();
    }

    for (;;) {
        int all_eof = 1, any_fresh = 0;
        int64_t now_us;
        {                                            /* wall-pace the house tick (also offline:
                                                      * inputs have independent clocks, so the
                                                      * mosaic cadence is the house rate, not media) */
            int64_t target = wall0 + mv_tick_us(c, tick);
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
        }
        now_us = av_gettime_relative();

        for (k = 0; k < n; k++) {                    /* pop ONE frame from this input's jitter
                                                      * buffer (FIFO) -> feed buffersrc k; dup-hold
                                                      * its last frame on underrun, black when stale */
            VideoHold *h = &c->inputs[k].hold;
            AVFrame *f = NULL, *st; int stale, fresh = 0;
            /* Take a candidate: a frame held back last tick (pending) or a fresh pop. Two hold
             * gates may keep it back (frame stays in pending, skew/EMA freeze via pending[k]):
             *  - CONTENT CLAMP (opt-in, g_mv_clamp): content-age leads the house clock.
             *  - CADENCE RESIDENCE (v0.9.13, default): the previous frame's content-projected
             *    residence hasn't elapsed — the slot is consumed at its SOURCE rate, so a
             *    rate-mismatched slot holds a regular cadence (5:6 for 25-in-29.97) and a
             *    burst of late frames NEVER fast-forwards (the due re-base turns a starvation
             *    deficit into constant slot latency). Multiple due frames in one tick (a
             *    59.94 slot in a 29.97 house) pop through and the newest displays. */
            AVFrame *cand = NULL;
            {
                int   resid = g_mv_residence && c->tick_dur_us > 0;
                int64_t hnow = mv_tick_us(c, tick);
                int64_t half = c->tick_dur_us / 2;
                int   qhot  = av_thread_message_queue_nb_elems(h->q) >= g_frameq_cap - g_frameq_cap / 4;
                int   pops;
                for (pops = 0; pops < 4; pops++) {
                    AVFrame *nx = pending[k];
                    pending[k] = NULL;
                    if (!nx) {
                        int rr = av_thread_message_queue_recv(h->q, &nx, AV_THREAD_MESSAGE_NONBLOCK);
                        if (rr == AVERROR_EOF) { done_in[k] = 1; break; }
                        if (rr < 0) break;                       /* jitter buffer empty */
                    }
                    if (g_mv_clamp && c->tick_dur_us > 0 && nx->pts != AV_NOPTS_VALUE) {
                        int64_t h0c; pthread_mutex_lock(&c->inputs[k].h0_lock); h0c = c->inputs[k].h0; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                        if (h0c != AV_NOPTS_VALUE &&
                            av_rescale_q(nx->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - h0c > mv_tick_us(c, tick + 1)) {
                            pending[k] = nx;                     /* content leads the clock -> hold (exact tick+1 since v0.9.12) */
                            break;
                        }
                    }
                    if (resid && !qhot && hnow + half < res_due_us[k]) {
                        pending[k] = nx;                         /* residence hold: not due yet (deliberate pacing) */
                        if (!cand) pd_cnt[k]++;
                        break;
                    }
                    if (cand) { av_frame_free(&cand); md_cnt[k]++; }   /* superseded within one tick (decimation) */
                    cand = nx;
                    if (resid) {
                        int64_t src = nx->pts != AV_NOPTS_VALUE ?
                                      av_rescale_q(nx->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) : AV_NOPTS_VALUE;
                        if (src != AV_NOPTS_VALUE && res_src_us[k] != AV_NOPTS_VALUE) {
                            /* Cadence estimate: EMA over deltas ACCEPTED only inside a band around the
                             * current estimate ([ema/2, 2*ema]) — a delta spanning skipped frames
                             * (drop-oldest, decode drops, corrupt skips) would inflate the estimate,
                             * which over-holds, which overflows the buffer, which drops more frames:
                             * a runaway (v1 of this gate ratcheted sk to +23s exactly this way). */
                            int64_t d = src - res_src_us[k];
                            if (d > res_ema_us[k] / 2 && d < res_ema_us[k] * 2)
                                res_ema_us[k] += (d - res_ema_us[k]) / 16;
                            /* hard bounds: 3x house tick covers 10fps..90fps sources in a 29.97 house */
                            res_ema_us[k] = av_clip64(res_ema_us[k], c->tick_dur_us / 3, c->tick_dur_us * 3);
                        }
                        if (src != AV_NOPTS_VALUE) res_src_us[k] = src;
                        if (qhot && hnow + half < res_due_us[k]) {
                            /* valve-FORCED pop (gate wanted to hold): not a cadence event — re-base
                             * instead of accumulating, else a hot queue ratchets the schedule ahead
                             * (measured +1.0-1.6s duephase in the first second of a startup-backlog
                             * run, which the servo then needed ~90s to bleed off). */
                            res_due_us[k] = hnow - half + res_ema_us[k];
                        } else {
                            /* Occupancy servo (the single-input WUCR lesson: PROPORTIONAL, small,
                             * capped): trim the residence toward the primed jitter-buffer depth so
                             * long-term consumption always equals arrival even if the cadence
                             * estimate is biased. +-2% authority: enough to null estimator bias,
                             * far too weak to disturb the 5:6 cadence pattern. */
                            int64_t corr = ((int64_t)av_thread_message_queue_nb_elems(h->q) - res_occ_tgt) * 1000;
                            corr = av_clip64(corr, -20000, 20000);
                            res_due_us[k] = FFMAX(res_due_us[k], hnow - half)
                                            + res_ema_us[k] - res_ema_us[k] * corr / 1000000;
                        }
                    } else break;                                /* residence off: exactly one pop per tick (legacy) */
                }
                if (!cand && !pending[k] && resid && !done_in[k] && last[k] &&
                    hnow + half >= res_due_us[k])
                    sv_cnt[k]++;                                 /* due but nothing arrived = genuine starvation dup */
            }
            fresh = (cand != NULL);
            if (fresh) {
                f = cand;                                 /* for the re-anchor diag below */
                if (last[k]) av_frame_free(&last[k]);
                last[k] = cand; any_fresh = 1; last_fresh_us[k] = now_us;
                /* Option F (coarse half) — RE-ANCHOR on return from an outage. When a slot
                 * comes back after having been black-slated, clear its accumulated dup skew
                 * so the returning audio is NOT delayed by the stale dup-hold total. For a
                 * continuous-PTS source (the common network-blip case) the source's own PTS
                 * already advanced across the gap, so once the stale skew is gone the audio
                 * lands at the current output time = its returning video -> back IN SYNC,
                 * exactly what a hardware frame-synchronizer does on re-acquire. This reset
                 * is a DECREASE but it cannot stall async: the outage (>= slate timeout)
                 * always exceeds the cleared skew, so the returning audio's input pts is
                 * still forward of the last pre-outage one. Only fires after a real slate,
                 * never on routine dup jitter (which the non-decreasing fine skew handles). */
                if (slated[k]) {
                    if (g_diag) {
                        int64_t h0d; pthread_mutex_lock(&c->inputs[k].h0_lock); h0d = c->inputs[k].h0; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                        int64_t dd = (f && f->pts != AV_NOPTS_VALUE) ? av_rescale_q(f->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) : -1;
                        av_log(NULL, AV_LOG_INFO, "[PTV-REANCHOR] slot %d return tick=%"PRId64": prev_skew=%"PRId64"ms disp=%"PRId64"ms h0=%"PRId64"ms out=%"PRId64"ms reanchor=%d\n",
                               k, tick, skew_us[k]/1000, dd/1000, h0d/1000, (tick*c->tick_dur_us)/1000, g_reanchor);
                    }
                    if (g_reanchor) { skew_us[k] = 0; c->inputs[k].house_skew = 0; } slated[k] = 0;
                    atomic_fetch_add_explicit(&c->inputs[k].house_disturb, 1, memory_order_relaxed);  /* B3: arm PLL mid-run re-acquire */
                    /* audio-follow re-tracks continuously via its EMA — no per-outage reset needed */
                }
            }
            if (!done_in[k]) all_eof = 0;
            stale = (c->slate_after_us > 0 && last_fresh_us[k] > 0 && now_us - last_fresh_us[k] > c->slate_after_us);
            if (stale) slated[k] = 1;                 /* mark the outage so the next fresh frame re-anchors */
            /* Option F (fine half) — per-slot audio skew = the MEASURED output-vs-content
             * offset of the frame this cell actually displays: skew = out_time -
             * (displayed_src - h0). = single-input's house_skew (output - content) but
             * measured per slot at the mosaic join, so the slot's audio rides exactly the
             * retiming the compositor applied to its video (dup-hold -> skew grows). Reduces
             * to ~0 on a clean 1:1 FIFO (no regression on healthy inputs).
             *
             * NON-DECREASING + capped: this value is added to the audio's INPUT pts and fed
             * through aresample=async, which REQUIRES a monotonic input — a decreasing skew
             * steps the input pts backward, async stalls, the mux waits and the output
             * freezes (proven: a 20fps-into-25fps input oscillates skew negative and stalled
             * F-v1). So the async path carries only the rising dup drift; the one legitimate
             * decrease (return-from-outage) is the re-anchor reset above, which is safe
             * because the outage gap exceeds the cleared skew. Updated only while a real
             * frame is shown; frozen during black-slate (no audio then anyway). */
            if (last[k] && !stale && c->tick_dur_us > 0) {
                int64_t h0k;
                pthread_mutex_lock(&c->inputs[k].h0_lock); h0k = c->inputs[k].h0; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                /* FIRST-DISPLAY anchor (g_h0_at_display, multiview): if h0 is not yet set, anchor it to
                 * the frame being displayed NOW so that content maps to the current house output time →
                 * sk=0. Replaces the decode-thread anchor (first DECODED frame) which, under a deep
                 * startup prime, is an earlier/different content → the displayed video leaps ahead at
                 * tick 0 → P2 re-anchors h0 → the transcoded audio banks (monotonic guard) and a copied
                 * track's DTS jumps backward (clamp/freeze, historically an EINVAL no-data outage). With
                 * h0 anchored here, the audio + copied tracks anchor to the SAME h0 from the start → no
                 * leap, no P2, no bank, no clamp. */
                if (h0k == AV_NOPTS_VALUE && g_h0_at_display && last[k]->pts != AV_NOPTS_VALUE) {
                    h0k = av_rescale_q(last[k]->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - mv_tick_us(c, tick);
                    pthread_mutex_lock(&c->inputs[k].h0_lock); c->inputs[k].h0 = h0k; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                }
                if (h0k != AV_NOPTS_VALUE && last[k]->pts != AV_NOPTS_VALUE) {
                    int64_t disp_src = av_rescale_q(last[k]->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q);
                    int64_t sk = mv_tick_us(c, tick) - (disp_src - h0k);
                    /* 1.0.1-pre16 residual sensor (PASSIVE), per-slot video side: m_v[slot] =
                     * out − disp_src per house TICK — dup-holds and residence holds included
                     * (a dup presents old content later: disp_src frozen, out advances, m_v
                     * grows — REAL presentation shift, the single-input dups-included rule).
                     * out on the exact-rational axis (mv_tick_us — integer tick would re-import
                     * the ~10ppm EXACTTICK drift). h0-free in FORM but ≈ −h0 at anchor time —
                     * the audio side carries −h0 structurally, so the shared −h0 cancels in R
                     * exactly as single-input. Published only while last[k] && !stale (the
                     * enclosing block): a slated slot stops publishing → its tracks read `--`.
                     * Measurement site = the DISPLAY site, not the pop site (a residence-held
                     * frame is still the displayed content this tick). */
                    if (g_rsync_sense) {
                        int64_t m = mv_tick_us(c, tick) - disp_src;
                        if (!rs_mv_seed[k]) { rs_mv_ema[k] = m; rs_mv_seed[k] = 1; }
                        else rs_mv_ema[k] += (m - rs_mv_ema[k]) / rs_mv_div;
                        atomic_store_explicit(&g_rsx.mv_ema[k], rs_mv_ema[k], memory_order_relaxed);
                        atomic_store_explicit(&g_rsx.mv_wall[k], now_us, memory_order_relaxed);
                    }
                    /* P2 — floor the per-slot lag to ≥0 by re-anchoring h0. A cell that leaps AHEAD of
                     * the house clock (sk very negative: −560ms on a 2x1, up to −2.5s on a 4-up, from an
                     * anomalous first decoded frame and/or a deep startup buffer prime) is physically
                     * wrong and is UNCORRECTABLE on a COPIED audio track (a copy can only be delayed,
                     * not advanced — backward DTS hits the monotonic clamp). Re-anchor h0 forward so the
                     * lag lands at +1 tick (slot reads slightly BEHIND, the normal buffered state): the
                     * video display is unchanged, transcoded audio rides the same h0+house_skew so it
                     * stays locked, and copied audio now only needs to DELAY → correctable. Fires only on
                     * a real video-ahead excursion (sk < −g_h0_reanchor_ms); gradual positive drift never
                     * triggers. MULTIVIEW ONLY; g_h0_reanchor gates it. */
                    if (n > 1 && g_h0_reanchor) {
                        int cond = sk < -(int64_t)g_h0_reanchor_ms * 1000;
                        int64_t sk_used = sk;
                        int fire;
                        if (g_reanchor2_instant)
                            fire = cond;                          /* pre-1.0.1 single-sample fire */
                        else {
                            /* 1.0.1 DEBOUNCE: shift = −sk + tick was computed from ONE instantaneous
                             * displayed-frame label, so a single corrupt PTS (one frame, DTS intact —
                             * passes the demux layer) inflated the shift by its full excursion and
                             * displaced the whole slot (transient audio-early until the PLL healed).
                             * Keep the last 5 evaluated sk samples per slot; fire only when ≥3 of
                             * them (including the current tick) held sk < −thr, and size the shift
                             * from the MEDIAN of the qualifying samples — a one-tick corrupt label
                             * is 1-of-5 (ignored), a real video-ahead excursion persists across
                             * ticks and still re-anchors within 5 ticks. PTV_REANCHOR2_INSTANT=1
                             * reverts. */
                            int idx = r2_pos[k];
                            r2_skring[k][idx] = sk;
                            r2_cond[k] = ((r2_cond[k] << 1) | (cond ? 1 : 0)) & 0x1f;
                            r2_pos[k] = (idx + 1) % 5;
                            fire = cond && av_popcount(r2_cond[k]) >= 3;
                            if (fire) {
                                int64_t v[5];
                                int m = 0, b, x, y;
                                for (b = 0; b < 5; b++)
                                    if (r2_cond[k] & (1u << b))
                                        v[m++] = r2_skring[k][(idx - b + 5) % 5];
                                for (x = 1; x < m; x++) {          /* insertion sort, m ≤ 5 */
                                    int64_t tv = v[x];
                                    for (y = x; y > 0 && v[y - 1] > tv; y--) v[y] = v[y - 1];
                                    v[y] = tv;
                                }
                                sk_used = v[m / 2];               /* median of the qualifying samples */
                                r2_cond[k] = 0;                   /* re-debounce after firing */
                            }
                        }
                        if (fire) {
                            int64_t shift = -sk_used + c->tick_dur_us; /* bring sk from negative to +1 tick */
                            pthread_mutex_lock(&c->inputs[k].h0_lock);
                            c->inputs[k].h0 += shift; h0k = c->inputs[k].h0;
                            pthread_mutex_unlock(&c->inputs[k].h0_lock);
                            sk = mv_tick_us(c, tick) - (disp_src - h0k);   /* now ≈ +1 tick (median-sized) */
                            af_off[k] = sk;                            /* snap the audio-follow EMA to the floored lag */
                            /* 1.0.1-pre16: a REANCHOR2 h0 shift is a label-lineage disturbance for
                             * this slot's tracks — bump the slot's disturbance epoch so the corrector
                             * dwell resets (feed wired now, consumed by the ARMING pre). REUSES the
                             * existing per-input house_disturb epoch (owner Q3, 2026-07-18): safe
                             * because house_disturb is consumed ONLY by corrector snapshots
                             * (rscorr_event_edge/dwell_reset) — the PLL acquire has been
                             * event-ungated since v0.6.18 — and the mv corrector is HELD OFF this
                             * pre, so the bump changes nothing until arming. */
                            atomic_fetch_add_explicit(&c->inputs[k].house_disturb, 1, memory_order_relaxed);
                            /* 0.9.18.7: promoted PTV_DIAG→always-on WARNING, with the AGLUE-style log
                             * rate limit. In LIVE mv a re-anchor is rare (a real video-ahead excursion),
                             * but on an unpaced source (file mv, decoder outrunning the clock) it can
                             * refire EVERY tick (~30/s measured) — the re-anchors still APPLY; only the
                             * lines are capped: 4 per 10s window per slot, then one summary as it rolls. */
                            {
                                int64_t now_r2 = av_gettime_relative();
                                if (now_r2 - r2_win_us[k] >= 10000000) {
                                    if (r2_supp[k])
                                        av_log(NULL, AV_LOG_WARNING,
                                            "[PTV-REANCHOR2] in%d %d more re-anchors (net h0 +%"PRId64"ms) suppressed in last 10s — unpaced/racing source, re-anchors still applied\n",
                                            k, r2_supp[k], r2_supp_us[k] / 1000);
                                    r2_win_us[k] = now_r2; r2_win_n[k] = 0;
                                    r2_supp[k] = 0; r2_supp_us[k] = 0;
                                }
                                if (r2_win_n[k] < 4) {
                                    r2_win_n[k]++;
                                    av_log(NULL, AV_LOG_WARNING,
                                        "[PTV-REANCHOR2] in%d tick=%"PRId64" video-ahead → h0 +%"PRId64"ms, lag→%"PRId64"ms\n",
                                        k, tick, shift / 1000, sk / 1000);
                                } else {
                                    r2_supp[k]++; r2_supp_us[k] += shift;
                                }
                            }
                        }
                    }
                    if (g_diag && !h0_logged[k]) {   /* P0: one-shot per slot at its first displayed frame */
                        h0_logged[k] = 1;
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-H0] in%d FIRST-DISPLAY tick=%"PRId64" h0=%"PRId64"ms first_disp_src=%"PRId64"ms (disp-h0=%"PRId64"ms) out=%"PRId64"ms lag0=%"PRId64"ms qd=%d dec=%"PRId64"\n",
                            k, tick, h0k / 1000, disp_src / 1000, (disp_src - h0k) / 1000,
                            (mv_tick_us(c, tick)) / 1000, sk / 1000,
                            av_thread_message_queue_nb_elems(c->inputs[k].hold.q), c->inputs[k].dc.dec_frames);
                    }
                    /* startup/ramp trace: per-tick for the first ~3s, then 1/s out to ~60s to capture the
                     * full per-slot lag RAMP (P0: see whether disp-h0 advances slower/faster than out). */
                    if (g_diag && (tick < 75 || (tick < 1500 && tick % 25 == 0)))
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-START] t=%"PRId64" in%d age=%"PRId64"ms out=%"PRId64"ms h0=%"PRId64"ms srcpts=%"PRId64" lag=%"PRId64"ms fresh=%d qd=%d\n",
                            tick, k, (disp_src - h0k) / 1000, (mv_tick_us(c, tick)) / 1000, h0k / 1000,
                            last[k]->pts, sk / 1000, fresh, av_thread_message_queue_nb_elems(c->inputs[k].hold.q));
                    lag_true_us[k] = sk;                                  /* PTV_DIAG: capture BEFORE clamp = lip-sync truth */
                    c->inputs[k].house_lag_true = sk;                     /* publish for the per-slot audio lip-sync probe */
                    /* A/V probe (read-only): record this slot's distinct displayed content → its
                     * first-display output time, so the slot's audio can pair against it (§3.2b). */
                    if (fresh)
                        vring_put(&c->inputs[k].vring, disp_src, mv_tick_us(c, tick));
                    /* Don't ratchet the audio skew during a CONTENT-CLAMP hold: that freeze is
                     * deliberate pacing (a future frame is pending, video waits for the clock),
                     * NOT a dup-underrun the audio should follow. Letting skew grow here would
                     * just move the desync from video-ahead to audio-late. The audio keeps
                     * playing and meets the resumed video. Genuine dup-holds (no pending frame)
                     * still grow skew so audio follows a real stall. */
                    if (g_audio_follow) {
                        /* AUDIO-FOLLOW (Option A) — CONTINUOUS re-tracking. Maintain a slow EMA of this
                         * slot's measured lag and publish it every tick; the drain applies an incremental
                         * deterministic drop/pad whenever the target moves >40ms. This re-tracks the per-slot
                         * video lag as it RAMPS IN at startup (it can take ~30s to settle — e.g. in1 ramped
                         * 0→+320ms over 30s) and any later drift, instead of latching one early value and
                         * freezing it (the old one-shot latched ~0 at t≈1s, missed the ramp, and left that
                         * slot's audio permanently ~the steady lag ahead — ~1s on the box). The EMA (~1.3s)
                         * smooths the ±100ms interlaced-PTS jitter so steady state stays put (the >40ms drain
                         * threshold = hysteresis, and the measured settled lag is stable to <1ms → no churn).
                         * Both signs are handled by the drain: lag>0 → pad/delay audio, lag<0 → drop/advance. */
                        if (!pending[k]) {
                            if (af_t0[k] == 0) { af_t0[k] = tick + 1; af_off[k] = sk; }  /* seed on first real frame */
                            else af_off[k] += (sk - af_off[k]) / 32;                     /* slow EMA (~1.3s) */
                        }
                        /* brief warmup so the EMA settles past the join transient before the drain acts */
                        c->inputs[k].house_skew = (af_t0[k] && tick - (af_t0[k] - 1) >= 25) ? af_off[k] : 0;
                    } else if (!pending[k]) {
                        if (sk < skew_us[k]) sk = skew_us[k];                 /* non-decreasing: async-safe */
                        if (sk > PTV_MV_SKEW_CAP_US) sk = PTV_MV_SKEW_CAP_US; /* bound to the async budget */
                        skew_us[k] = sk;
                        c->inputs[k].house_skew = sk;
                    }
                }
            }
            if (last[k] && !stale) st = av_frame_clone(last[k]);
            else                   st = blackf[k] ? av_frame_clone(blackf[k]) : NULL;
            if (!st) continue;
            st->pts = tick; st->pkt_dts = AV_NOPTS_VALUE;
            /* v0.9.16: add_frame consumes only the REFERENCE (frame reset, struct still ours) —
             * the clone shell must be freed on success too, else 448B leak per slot per tick
             * (~193MB/h on a 2x2 grid; the cor-2 mosaic RSS leak, masked for weeks by daily
             * sync_check restarts). On failure the frame is untouched; free releases the ref. */
            { int fr = av_buffersrc_add_frame(c->fsrc[k], st); (void)fr; }
            av_frame_free(&st);
        }

        for (r = 0; r < R; r++) {                    /* pull each rung's composited frame */
            while (av_buffersink_get_frame(c->fsink[r], filt) >= 0) {
                AVFrame *out = av_frame_alloc();
                if (out) {
                    av_frame_move_ref(out, filt);
                    out->pts = rung_pts[r]++;        /* per-rung consecutive CFR */
                    out->pkt_dts = AV_NOPTS_VALUE; out->duration = 0;
                    push_frame_q(c->frame_q[r], c->live, &c->framedrop[r], out);
                    if (r == 0) { c->emitted++; if (!any_fresh) c->dup++; }
                } else av_frame_unref(filt);
            }
        }
        tick++;
        g_vout_us = mv_tick_us(c, c->emitted);   /* PTV_DIAG: video output time for the audio probe */

        if (g_diag) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                char rb[448]; int rp = 0;    /* v0.9.13 residence internals: occ vs target, ema, due phase */
                for (k = 0; k < n && rp < (int)sizeof rb - 64; k++)
                    rp += snprintf(rb + rp, sizeof rb - rp, " in%d:occ=%d/ema=%.2fms/duephase=%+.1fms/pd=%"PRId64"/md=%"PRId64,
                                   k, av_thread_message_queue_nb_elems(c->inputs[k].hold.q),
                                   res_ema_us[k] / 1000.0, (res_due_us[k] - mv_tick_us(c, tick)) / 1000.0,
                                   pd_cnt[k], md_cnt[k]);
                av_log(NULL, AV_LOG_INFO, "[PTV-RES] t=%"PRId64" tgt=%d%s\n", tick, res_occ_tgt, rb);
                char db[512]; int dp = 0;
                for (k = 0; k < n && dp < (int)sizeof db - 88; k++)
                    dp += snprintf(db + dp, sizeof db - dp, " in%d:dec=%"PRId64"/skew=%dms/lag=%dms/holddrop=%"PRId64"/md=%"PRId64"/gpps=%d/%d/gov=%d",
                                   k, c->inputs[k].dc.dec_frames, (int)(skew_us[k] / 1000), (int)(lag_true_us[k] / 1000),
                                   c->inputs[k].hold.framedrop, md_cnt[k],   /* drop-oldest count: startup overflow = video-lead cause; md = residence decimation */
                                   /* pre16 rr13 blindness fix: the catch-up governor RAN on mv but said
                                    * nothing — per-slot measured/declared pps + engagement (the single-
                                    * input DIAG t= gpps=/gov= tokens, per slot) */
                                   atomic_load_explicit(&c->inputs[k].gov_gpps, memory_order_relaxed),
                                   atomic_load_explicit(&c->inputs[k].gov_decl, memory_order_relaxed),
                                   atomic_load_explicit(&c->inputs[k].gov_on,   memory_order_relaxed));
                av_log(NULL, AV_LOG_INFO,
                    "[PTV-DIAG] mv t=%.1fs emitted=%"PRId64" dup=%"PRId64" muxed=%"PRId64" frameq0=%d%s\n",
                    (nowd - diag_t0) / 1000000.0, c->emitted, c->dup, g_muxed,
                    av_thread_message_queue_nb_elems(c->frame_q[0]), db);
                diag_last = nowd;
            }
        }
        if (g_stats) {
            int64_t nows = av_gettime_relative();
            if (nows - stat_last >= g_stats_period_us) {
                double dt    = (nows - stat_last) / 1000000.0;
                double fps   = (c->emitted - stat_prev) / (dt > 0 ? dt : 1);   /* instantaneous, like single-input */
                double secs  = mv_tick_us(c, c->emitted) / 1000000.0;
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                char dlv[64] = "";                   /* §7.5a delivery gate readout (mv gated since v0.9.12.1) */
                if (c->gate0)
                    snprintf(dlv, sizeof dlv, " dlvhold=%"PRId64"ms dlvforced=%"PRId64,
                             atomic_load_explicit(&c->gate0->st_hold_us, memory_order_relaxed) / 1000,
                             atomic_load_explicit(&c->gate0->st_forced, memory_order_relaxed));
                char aco[24] = "";                   /* pre15/pre16 #33: corrupt-discarded AUDIO pkts, GLOBAL sum
                                                      * (owner Q — per-track detail stays on the [PTV-ADISC]/NBS
                                                      * lines; absent while zero, clean line unchanged) */
                {
                    int64_t ac = atomic_load_explicit(&g_acorrupt, memory_order_relaxed);
                    if (ac > 0)
                        snprintf(aco, sizeof aco, " acor=%lld", (long long)ac);
                }
                char rsl[24 + PTV_MAX_AUDIO * 16];   /* pre16: per-slot residual sensor lipsync= — ALWAYS-ON on mv
                                                      * (aK: prefix forced; `--` on a slated slot IS the outage
                                                      * signal; the observation soak is the deliverable) */
                ptv_stats_lipsync(rsl, sizeof rsl, nows, 1);
                /* corr= deliberately NOT printed: the mv corrector is HELD OFF this pre
                 * (rscorr_update hold) — the arming pre adds ptv_stats_corr here. */
                char ls[448]; int lp = 0;   /* per-slot: qdrop=input-q overflow, corrupt=demux+decode,
                                             * pd=cadence holds (NORMAL for a rate-mismatched slot),
                                             * sv=starvation dups, sk=published audio skew,
                                             * skres=LAYERA erase-residue ledger (0.9.18.7, same
                                             * accounting as single-input hsres= — the slot's sk
                                             * measurement rides the same erased label stream) */
                for (k = 0; k < n && lp < (int)sizeof ls - 88; k++)
                    lp += snprintf(ls + lp, sizeof ls - lp,
                                   " in%d:qdrop=%"PRId64"/corrupt=%"PRId64"/pd=%"PRId64"/sv=%"PRId64"/sk=%+dms/skres=%+dms",
                                   k, c->inputs[k].da.vdrop, c->inputs[k].da.vcorrupt + c->inputs[k].dc.vcorrupt,
                                   pd_cnt[k], sv_cnt[k], (int)(c->inputs[k].house_skew / 1000),
                                   (int)(c->inputs[k].da.disc_resid_us / 1000));
                av_log(NULL, AV_LOG_INFO,   /* v0.9.13 parity: size/bitrate/speed/genlock dropped (v0.9.10 single-input rationale) */
                    "frame=%6"PRId64" fps=%4.1f time=%02d:%02d:%05.2f dup=%"PRId64" drop=%"PRId64"%s%s%s%s\n",
                    c->emitted, fps, hh, mm, ss, c->dup, c->framedrop[0], dlv, aco, rsl, ls);
                stat_last = nows; stat_prev = c->emitted;
            }
        }
        if (g_slow) av_usleep(g_slow);
        if (all_eof) break;                          /* every input terminated -> tear down */
    }

    for (k = 0; k < n; k++) { int fr = av_buffersrc_add_frame(c->fsrc[k], NULL); (void)fr; }   /* flush graph */
    for (r = 0; r < R; r++)
        while (av_buffersink_get_frame(c->fsink[r], filt) >= 0) {
            AVFrame *out = av_frame_alloc();
            if (out) { av_frame_move_ref(out, filt); out->pts = rung_pts[r]++; out->pkt_dts = AV_NOPTS_VALUE; out->duration = 0;
                       push_frame_q(c->frame_q[r], c->live, &c->framedrop[r], out); }
            else     { av_frame_unref(filt); }
        }
done:
    av_frame_free(&filt);
    for (k = 0; k < n; k++) { if (blackf[k]) av_frame_free(&blackf[k]); if (last[k]) av_frame_free(&last[k]); if (pending[k]) av_frame_free(&pending[k]); }
    for (r = 0; r < R; r++) av_thread_message_queue_set_err_recv(c->frame_q[r], AVERROR_EOF);
    return NULL;
}

