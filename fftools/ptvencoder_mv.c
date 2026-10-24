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
#include "libswscale/swscale.h"
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
static int64_t mv_tick_us(CompositorCtx *c, int64_t t);

/* 2.1 T-409 (mosaic, design §5.4a): a slot is a frame synchronizer. Its picture is shown by CONTENT TIME at a lag L:
 * on each tick the newest picture whose source time (on the slot's h0 axis) is due at output - L, older due ones
 * skipped, none due = the last one repeated. A lost picture is then a repeat and a late one a skip, and L does not move:
 * the slot's sound, placed at its source time + L, needs no step for either (the count-paced residence let losses pull
 * the picture forward and re-primes push it back, and the PLL chased both ~3.6 s later: cor-3 Curiosity 2026-10-22).
 * L moves only by
 *  - a re-lock: pictures more than 0.5 s late for 1 s without catching up (a delayed source, not a backlog) re-lock
 *    behind them with the primed jitter-buffer depth as margin (grows only); content more than 10 s past the margin
 *    ahead with nothing shown for 3 s re-locks onto it;
 *  - the buffer's minimum depth: a start-up backlog it never drains below is shed in one step; after the slot's first
 *    20 s (60 s of evidence after any step): what the last 60 s never went below, past the primed depth + 2 ticks, is
 *    shed in one step; below a quarter of it in every 10 s = a source clock slower than the house, one tick more.
 *    Bursty or jittery arrival and a lone source hole move nothing; a backlog catching up is skipped, not kept.
 * Every step takes effect at a content boundary for the picture and the sound (mv_tl_step). */
typedef struct MvTlSlot {
    int64_t lag, cad, last_src, shown_wc, pop_wc, relock_wc;
    int64_t cad_min, cad_wc;                        /* the period: the shortest picture step of the current 2 s window */
    int     cad_n;                                  /* ... and how many steps it saw */
    int64_t real;                                   /* the realized lag: the shown pictures' output - source, averaged */
    int64_t shown_src;                              /* source time of the last picture shown */
    int64_t first_wc;                               /* the slot's first picture (wall) */
    int     reseed;                                 /* the lag moved: re-seed the sensor's video side at the next
                                                     * picture shown (the move shows from then on) */
    int     tails;                                  /* stale pictures dropped since the input ran dry */
    int64_t late_wc, late_ref;                      /* pictures more than 0.5 s late: since when, how late then */
    int64_t wmin, win_wc, mins[6];                  /* the buffer's minimum depth: this window, its start, the last 6 */
    int     nmin;
    int     stepping;                               /* a lag step is scheduled: ... */
    int64_t nlag, nat;                              /* ... the lag from content nat (h0 axis) on */
    int64_t ovf_seen, ovf_wc;                       /* the queue's drop count at the last pop; the last overflow step */
    int     gapstep;                                /* the scheduled step is an overflow's: the content before its
                                                     * boundary, from skip_from on, is gone (pictures there are dropped) */
    int64_t skip_from;
    int     pend_gap;                               /* the pending picture: the full queue dropped pictures before it */
} MvTlSlot;

/* A lag step takes effect at a CONTENT boundary, for the picture and the sound alike. The sound for the content in flight
 * is already placed with the old lag (it waits in the delivery gate with its timestamps): a step at a moment in time
 * moved the picture while that sound stayed, and they skipped different content (mv04 sync_flapping, a -2.4 s re-lock:
 * pictures without their sound for 8 s). The boundary is just past the newest content the sound has been placed for:
 * before it both keep the old lag, from it on both the new one — a step down skips the same content on both, a step up
 * holds the picture and pads the sound at the same point. */
static void mv_tl_step(CompositorCtx *c, int k, MvTlSlot *s, int64_t lag, int64_t h0, int64_t now_us, int64_t at)
{
    int64_t ah = atomic_load_explicit(&c->inputs[k].tl_ahead_us, memory_order_relaxed);
    if (at != AV_NOPTS_VALUE) {                     /* a boundary given: the picture waiting there (an overflow left
                                                     * the pictures before it behind — nothing to wait for) */
        s->nat       = s->stepping ? FFMIN(s->nat, at) : at;
        s->stepping  = 1;
        s->gapstep   = 1;
        s->skip_from = s->nat - (s->lag - lag);     /* the old due line: shown up to here, nothing until the boundary */
    } else if (!s->stepping) {
        s->nat = s->shown_src + s->cad;
        if (ah && h0 != AV_NOPTS_VALUE)
            s->nat = FFMAX(s->nat, ah - h0 + 200000);   /* + what the sound may place before it sees the step */
        s->stepping = 1;
    }
    s->nlag      = lag;
    s->relock_wc = now_us;
    s->nmin      = 0;                               /* the depth history read the old lag: 60 s of fresh evidence
                                                     * before the next drift step (one 0.7 s source hole stepped
                                                     * +40 ms every 10 s to +240 ms: Curiosity replay) */
    atomic_store_explicit(&c->inputs[k].tl_nlag_us, s->real + (lag - s->lag), memory_order_relaxed);
    /* an overflow step: the picture skipped from the old due line (the boundary less the step) to the boundary */
    atomic_store_explicit(&c->inputs[k].tl_skip_us, s->gapstep && h0 != AV_NOPTS_VALUE ? s->skip_from + h0 : INT64_MAX,
                          memory_order_relaxed);
    atomic_store_explicit(&c->inputs[k].tl_at_us, h0 != AV_NOPTS_VALUE ? s->nat + h0 : INT64_MIN, memory_order_release);
}

/* the first picture at or past the boundary is shown: the step is the slot's lag from now on */
static void mv_tl_commit(CompositorCtx *c, int k, MvTlSlot *s)
{
    s->real    += s->nlag - s->lag;
    s->lag      = s->nlag;
    s->stepping = 0;
    s->gapstep  = 0;
    s->reseed   = 1;                                /* the sensor's video side takes the step with it (it read the
                                                     * step as lip sync, decaying over a minute: mv02 -1027 ms) */
    atomic_store_explicit(&c->inputs[k].tl_lag_us, s->real, memory_order_relaxed);
    atomic_store_explicit(&c->inputs[k].tl_skip_us, INT64_MAX, memory_order_relaxed);   /* no boundary: nothing skipped */
    atomic_store_explicit(&c->inputs[k].tl_at_us, INT64_MAX, memory_order_release);
}

static int64_t mv_tl_lag(const MvTlSlot *s, int64_t fs)
{
    return s->stepping && fs >= s->nat ? s->nlag : s->lag;
}

static AVFrame *mv_tl_pick(CompositorCtx *c, int k, int64_t tick, int64_t now_us, AVFrame **pending, int *done,
                           MvTlSlot *s, int occ_tgt, int *mv_seed)
{
    VideoHold *h = &c->inputs[k].hold;
    /* the primed depth in CONTENT TIME (the preroll, house ticks) — and the depth below too: depth was a picture count
     * times the learned period, so a 59.94 slot got half the margin (0.48 s), and a queue pinned at its capacity by a
     * decoder catching up read 'full' whatever was shed (the 60 s rule shed -4.4 s three times, the slot 9.7 s ahead
     * and starving: mass-restart replay) */
    int64_t out = mv_tick_us(c, tick), half = c->tick_dur_us / 2, margin = (int64_t)occ_tgt * c->tick_dur_us;
    int64_t h0, src = AV_NOPTS_VALUE, relock = AV_NOPTS_VALUE;
    int dry, occ, gap = 0;
    AVFrame *cand = NULL, *nx;
    pthread_mutex_lock(&c->inputs[k].h0_lock); h0 = c->inputs[k].h0; pthread_mutex_unlock(&c->inputs[k].h0_lock);
    dry = s->pop_wc && now_us - s->pop_wc >= 500000;
    for (;;) {
        int64_t fs;
        nx = *pending;
        *pending = NULL;
        gap = s->pend_gap;
        if (!nx) {
            int rr = av_thread_message_queue_recv(h->q, &nx, AV_THREAD_MESSAGE_NONBLOCK);
            int64_t ov;
            if (rr == AVERROR_EOF) { *done = 1; break; }
            if (rr < 0) break;
            /* the full queue dropped pictures since the last pop: the content before this picture is gone */
            ov  = atomic_load_explicit(&h->ovf, memory_order_acquire);
            gap = ov != s->ovf_seen;
            s->ovf_seen = ov;
        }
        if (h0 == AV_NOPTS_VALUE || nx->pts == AV_NOPTS_VALUE) {   /* the slot's first picture anchors h0 (lag 0) */
            if (cand) { *pending = nx; break; }
            cand = nx;
            if (h0 == AV_NOPTS_VALUE) break;
            continue;
        }
        fs = av_rescale_q(nx->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - h0;
        /* the slot's cadence (every picture is popped, decimated ones too) = the SHORTEST step of the last 2 s, as the
         * single-input period (T-011): a lost picture lengthens a step, and so does one the full queue dropped. The 1/8
         * average learned long steps, and past 4x the period it ignored the real ones for good: cor-3 GB News slot 0
         * (40 ms) read 204 ms in its first 10 s after the mass restart — the depth and the primed margin read 5x
         * ('primed 5907 ms'), shed -17.8 s, re-locked +17.1 s, bars, again every minute. Seeded at the house tick; a
         * 59.94 slot in a 29.97 mosaic learns 16.7 ms (mv01-t397) */
        if (s->last_src != AV_NOPTS_VALUE && fs - s->last_src >= 1000 && (!s->cad_min || fs - s->last_src < s->cad_min))
            s->cad_min = fs - s->last_src;
        if (s->last_src != AV_NOPTS_VALUE && fs - s->last_src >= 1000)
            s->cad_n++;
        /* a hole in the content (more than 0.5 s missing): the depth history is the hole's, not the clock's — 60 s of
         * fresh windows before any drift step. The dry rule below sees only a hole the buffer did not bridge: a source
         * flapping 2 s off / 2 s on read as a slow clock, +80 ms per minute (mv04 sync_flapping) */
        if (s->last_src != AV_NOPTS_VALUE && fs - s->last_src > 500000) {
            s->nmin   = 0;
            s->win_wc = 0;
        }
        s->last_src = fs;
        /* the picture nearest the tick: due half a picture ahead of it — half a tick, or half the slot's own period
         * when it is faster than the house. A 59.94 slot in 29.97 put every second picture ON the half-tick line,
         * and a 1/90000 s stamp difference moved it across: 1-3-1-3 pictures a tick instead of 2 (T-341) */
        if (s->stepping && s->gapstep && fs >= s->skip_from && fs < s->nat) {   /* inside an overflow step's skip */
            av_frame_free(&nx);
            continue;
        }
        if (fs > out - mv_tl_lag(s, fs) + FFMIN(half, s->cad / 2)) {
            /* 2.1 T-411/T-414: the full queue dropped the pictures before this one (a decoder catching up after the
             * compose, a burst past the queue's capacity): waiting for its time shows nothing meanwhile (up to the
             * queue's 5.3 s: bars every 5.3 s on a mass restart's slots, a bursty slot's re-lock never due). An
             * overflow is a step to the newest content queued less the primed margin — the latency the shed rules
             * would remove anyway, in one step — from the old due line on, pictures and sound alike (not the sound's
             * placed front: that boundary kept the picture waiting). One a second while the decoder keeps overflowing
             * (the picture holds at most 1 s between). Whenever it happens (pre6: one rule, no start-up window — a
             * mass restart is just a backlog): the buffer holds what it holds, the rest is not played late */
            if (!cand && gap && s->shown_wc && now_us - s->ovf_wc >= 1000000) {
                int64_t np = atomic_load_explicit(&h->newest_pts, memory_order_relaxed), tgt = fs;
                if (np != AV_NOPTS_VALUE) {
                    int64_t nf = av_rescale_q(np, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - h0;
                    if (nf - margin > fs && nf - fs < 60000000)
                        tgt = nf - margin;
                }
                av_log(NULL, AV_LOG_INFO, "[PTV-MV] in%d re-locked: picture %+"PRId64" ms (the buffer overflowed: the "
                       "pictures before this one are gone, %"PRId64" dropped so far)\n", k, (out - tgt - s->lag) / 1000,
                       s->ovf_seen);
                mv_tl_step(c, k, s, out - tgt, h0, now_us, tgt);
                s->ovf_wc = now_us;
                gap = 0;
                if (fs < tgt) {                     /* this one is inside the skip too */
                    av_frame_free(&nx);
                    continue;
                }
            }
        }
        if (fs > out - mv_tl_lag(s, fs) + FFMIN(half, s->cad / 2)) {
            /* content far ahead of its time (10 s past the margin) with nothing shown for 3 s — labels the timeline
             * left ahead: re-lock behind it with the primed margin. Content a burst brings ahead of its time is
             * normal for a bursty source and waits (re-locking on 1 s churned an HLS-burst slot every few seconds) */
            if (!cand && fs - (out - mv_tl_lag(s, fs)) > margin + 10000000 && now_us - s->shown_wc > 3000000 &&
                now_us - s->relock_wc > 3000000) {
                av_log(NULL, AV_LOG_INFO, "[PTV-MV] in%d re-locked: picture %+"PRId64" ms (content ahead)\n",
                       k, (out - fs + margin - s->lag) / 1000);
                mv_tl_step(c, k, s, out - fs + margin, h0, now_us, AV_NOPTS_VALUE);
            }
            *pending = nx;                          /* not due yet */
            s->pend_gap = gap;
            break;
        }
        /* after a dry spell, the first picture can be the pre-gap PES tail (T-350: the cut PES completed with post-gap
         * bytes, flushed with its OLD label) — locked onto, the slot re-locked +20 s and back (mv04 gap_20). A picture
         * more than 1 s late whose source time is still near the last one shown is dropped, up to 3 per return; a
         * source that really resumes where it stopped (late by the pause) re-locks after them */
        if (dry && s->tails < 3 && s->shown_wc && out - mv_tl_lag(s, fs) - fs > 1000000 &&
            fs - s->shown_src < (now_us - s->shown_wc) / 2) {
            s->tails++;
            av_frame_free(&nx);
            continue;
        }
        if (cand)
            av_frame_free(&cand);                   /* superseded within the tick (a late picture / decimation) */
        cand = nx;
        src = fs;
    }
    occ = av_thread_message_queue_nb_elems(h->q) + (*pending != NULL);
    if (cand) {
        if (s->stepping && src != AV_NOPTS_VALUE && src >= s->nat)
            mv_tl_commit(c, k, s);
        /* late = skip while the slot catches up; re-lock only behind pictures that STAY late. A backlog (this
         * process or its decode stalled) arrives faster than real time and each picture is less late than the one
         * before — re-locking on its first picture kept it as latency (+1..2.5 s per slot after a 6 s stall) and as
         * memory; a delayed source stays late at real-time pace. The buffer refills by itself after an outage when
         * the source comes back on time: no re-prime (it added the margin to the lag at every outage) */
        if (src != AV_NOPTS_VALUE && !s->stepping) {
            int64_t late = out - s->lag - src;
            if (late <= 500000)
                s->late_wc = 0;
            else if (!s->late_wc || late < s->late_ref - 2 * c->tick_dur_us) {
                s->late_wc  = now_us;               /* late, or catching up: (re)start the second */
                s->late_ref = late;
            } else if (now_us - s->late_wc >= 1000000)
                relock = out - src + margin;        /* late for 1 s at real-time pace: hold for the primed margin */
        }
        if (relock != AV_NOPTS_VALUE && relock - s->lag >= c->tick_dur_us) {   /* grows only */
            av_log(NULL, AV_LOG_INFO, "[PTV-MV] in%d re-locked: picture %+"PRId64" ms (late for 1 s)\n", k,
                   (relock - s->lag) / 1000);
            s->late_wc = 0;
            mv_tl_step(c, k, s, relock, h0, now_us, AV_NOPTS_VALUE);
        } else
            relock = AV_NOPTS_VALUE;
        if (src != AV_NOPTS_VALUE) {
            /* pictures land on ticks: the lag they realize is L rounded by the slot's phase (a 25p slot in a 25p mosaic:
             * L 21 ms showed every picture 40 ms late, its sound placed at L read 19 ms early); a rate-converted slot's
             * cadence pattern averages out. The sound follows this */
            /* ... from the picture SHOWN: a decimated slot's skipped picture is not shown at all, and the sound belongs with
             * the one that is (a frame sync leaves the audio continuous and drops whole pictures). 2.1.0-pre2 took the mean
             * of the pictures a tick stands for — tuned against the flash ruler, which reads by the shown picture's parity:
             * by content (avxcorr on the cell, parity-free) it put a 59.94 slot's sound 8.3 ms late (T-420) */
            int64_t smp = out - src;
            {   /* TEST-ONLY (PTV_TEST_MVTRACE): the slot's lags once a second */
                static int tr = -1;
                static int64_t twc[PTV_MAX_INPUT];
                if (tr < 0) tr = getenv("PTV_TEST_MVTRACE") != NULL;
                if (tr && k < PTV_MAX_INPUT && (now_us - twc[k] >= 1000000 || smp - s->lag > c->tick_dur_us)) {
                    twc[k] = now_us;
                    av_log(NULL, AV_LOG_INFO, "[PTV-TEST] in%d out %.3f shown %.3f lag %.1f real %.1f smp %.1f occ %d%s\n", k,
                           out / 1e6, src / 1e6, s->lag / 1000.0, s->real / 1000.0, smp / 1000.0, occ, s->stepping ? " stepping" : "");
                }
            }
            if (!s->shown_wc)
                s->real = s->lag;
            /* ... by at most a picture per sample: two source pictures labelled 29 pictures old (RAV slot 1, a 29.97i
             * feed, at 36 s) moved it 30 ms, the sound chased it there and back in half-frame steps and kept -7.5 ms */
            {
                int64_t w = FFMAX(c->tick_dur_us, s->cad);
                s->real += av_clip64(smp - s->real, -w, w) / 32;
            }
        }
        if (src != AV_NOPTS_VALUE)
            s->shown_src = src;
        if (s->reseed && !s->stepping) {
            s->reseed = 0;
            *mv_seed  = 0;
        }
        if (!s->first_wc)
            s->first_wc = now_us;
        s->shown_wc = now_us;
        s->pop_wc   = now_us;
        s->tails    = 0;
        if (dry)                                    /* back after a dry spell: the depth history is the outage's, not
                                                     * the clock's — 60 s of fresh windows before any drift step */
            s->nmin = 0;
    } else if (occ)
        s->pop_wc = now_us;                         /* pictures waiting: not dry */
    /* the buffer's MINIMUM depth over a window says what the lag must do; its average does not — bursty or jittery
     * arrival swings the depth by design (cor-3 Curiosity slot 1: 0.7..1.3 s every minute, re-locked +-300 ms each time;
     * an HLS-burst slot re-locked every few seconds). 10 s windows, the last 60 s: never below the primed
     * depth + 2 ticks = latency nobody uses, shed in one step (a backlog past the buffer is cut at once above: the
     * overflow step, the decoder's catch-up); below half of it in EVERY window = a source clock
     * slower than the house, one tick more, two while the newest window is below a quarter (one hole drains the buffer
     * once: content lost, not drift); after a shed or a re-lock 60 s of fresh windows. Pictures that stay late re-lock
     * above. */
    if (h0 != AV_NOPTS_VALUE && s->first_wc) {
        int64_t depth = (int64_t)occ * s->cad, wl = 10000000;
        int64_t np = atomic_load_explicit(&h->newest_pts, memory_order_relaxed);
        if (np != AV_NOPTS_VALUE) {   /* the depth in content time: the newest picture queued past the due line */
            int64_t nf = av_rescale_q(np, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - h0;
            if (nf - s->shown_src >= -1000000 && nf - s->shown_src <= 60000000)   /* not across a wrap / label jump */
                depth = FFMAX(0, nf - (out - s->lag));
        }
        if (!s->win_wc) {
            s->win_wc = now_us;
            s->wmin   = depth;
        }
        s->wmin = FFMIN(s->wmin, depth);
        if (now_us - s->win_wc >= wl) {
            int64_t wm = s->wmin;
            s->mins[s->nmin++ % 6] = wm;
            s->win_wc = now_us;
            s->wmin   = depth;
            if (!s->stepping && now_us - s->relock_wc >= wl) {   /* after a step, 10 s before the next */
                if (s->nmin >= 6) {
                    int64_t lo = s->mins[0], hi = s->mins[0];
                    int x;
                    for (x = 1; x < 6; x++) {
                        lo = FFMIN(lo, s->mins[x]);
                        hi = FFMAX(hi, s->mins[x]);
                    }
                    if (lo > 2 * margin) {   /* unused latency: never needed in 60 s, more than the primed depth on
                                              * top of it — a backlog left over, not a clock: one cut. Up to there drift
                                              * goes a picture per window even after damage paused the evidence (a
                                              * +1000 ppm slot on a damaged capture: 6 ticks as the limit cut -266 ms) */
                        av_log(NULL, AV_LOG_INFO, "[PTV-MV] in%d re-locked: picture %+"PRId64" ms (the buffer never "
                               "below %"PRId64" ms in 60 s, primed %"PRId64" ms)\n", k, -(lo - margin) / 1000,
                               lo / 1000, margin / 1000);
                        mv_tl_step(c, k, s, s->lag - (lo - margin), h0, now_us, AV_NOPTS_VALUE);
                    } else if (lo > margin + c->tick_dur_us) {
                        /* 2.1 T-422: a source clock faster than the house — one picture dropped per window, as a frame sync
                         * does, the history kept (read shallower by the step). The backlog rule above cut it in one step
                         * once 60 s never went below margin + 2 ticks, then waited 60 s of fresh windows: cor-3 RAV slot 0
                         * (~140 ppm fast) cut -83..-116 ms of picture and sound every 10-15 min */
                        int nm = s->nmin;
                        mv_tl_step(c, k, s, s->lag - c->tick_dur_us, h0, now_us, AV_NOPTS_VALUE);
                        s->nmin = nm;
                        for (x = 0; x < 6; x++)
                            s->mins[x] -= c->tick_dur_us;
                    }
                    else if (hi < margin / 2 && now_us - s->shown_wc < wl) {   /* running low — a live slot only:
                                                                                 * an outage's empty windows are no clock
                                                                                 * evidence */
                        int nm = s->nmin;
                        int64_t st = (wm < margin / 4 ? 2 : 1) * c->tick_dur_us;
                        mv_tl_step(c, k, s, s->lag + st, h0, now_us, AV_NOPTS_VALUE);
                        /* the history stays, read deeper by the step from here. With 60 s of fresh windows per tick the
                         * rule followed 550 ppm at most: a source 1100 ppm slow (RAV in0, PCR -1076 ppm against arrival)
                         * drained its buffer, ran late and re-locked +1 s every 10 min (1 s of silence each). A tick per
                         * 10 s window from a quarter of the depth still sat at empty: RAV in0's content runs 3000 ppm
                         * slow for minutes (2400 ppm at its start), a step a window followed it but never refilled, and
                         * every late burst repeated a picture (sv +60 per 10 s). From half the depth, two ticks while
                         * the newest window is below a quarter: up to 6600 ppm, the buffer kept off the floor */
                        s->nmin = nm;
                        for (x = 0; x < 6; x++)
                            s->mins[x] += st;
                    }
                }
            }
        }
    }
    /* the window's shortest step is the period — from 8 steps on: a slot of 10 pictures a second shows 20. A burst into a
     * full queue drops the oldest while the next picture waits for its tick, and the slot pops one picture every few
     * seconds: one step of 4.6 s read as the period gave a primed depth of 133 s, shed -577 s and re-locked +567 s
     * (a replay of GB News slot 0 with a 20 s backlog at 2.5x). Fewer steps = unchanged */
    if (now_us - s->cad_wc >= 2000000) {
        if (s->cad_min && s->cad_n >= 8)
            s->cad = s->cad_min;
        s->cad_min = 0;
        s->cad_n   = 0;
        s->cad_wc  = now_us;
    }
    atomic_store_explicit(&c->inputs[k].tl_lag_us, s->real, memory_order_relaxed);
    return cand;
}

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
    void    *cellpic[PTV_MAX_INPUT] = {0};    /* 2.0.0: per-slot no-signal picture cache (ptv_hold_cell_pic) */
    int64_t  cell_dead_us[PTV_MAX_INPUT] = {0};   /* when the cell went to its placeholder (0 = showing the slot) */
    AVFrame *last[PTV_MAX_INPUT] = {0};       /* last frame popped per input (dup source) */
    AVFrame *pending[PTV_MAX_INPUT] = {0};    /* content-clamp: a frame popped but held back because its
                                               * content-time leads the house clock (gap); shown when out catches up */
    int64_t last_fresh_us[PTV_MAX_INPUT] = {0};
    int64_t skew_us[PTV_MAX_INPUT] = {0};     /* per-slot audio skew = accumulated dup-hold ticks */
    int64_t lag_true_us[PTV_MAX_INPUT] = {0}; /* PTV_DIAG: TRUE uncapped signed video lag (output−content); when
                                               * this >> skew_us the 250ms cap is saturating = audio can't follow */
    int      slated[PTV_MAX_INPUT] = {0};     /* slot is/was black-slated (outage) since last fresh frame */
    int64_t  mv_fill_us[PTV_MAX_INPUT][PTV_MAX_AUDIO] = {{0}};   /* T-093 part 2: last fill sentinel per slot track */
    int64_t  mv_pump_us[PTV_MAX_INPUT] = {0};   /* 2.0.2 T-386: last silence sentinel per deferred slot */
    struct SwsContext *norm_sws[PTV_MAX_INPUT] = {0};   /* 2.0.2 T-386: frames in another geometry than the cell's */
    int      norm_logged[PTV_MAX_INPUT] = {0};
    int64_t res_due_us[PTV_MAX_INPUT] = {0};  /* v0.9.13 residence: house-time the next pop is allowed (0 = immediate) */
    int64_t res_ema_us[PTV_MAX_INPUT];        /* v0.9.13: EMA of content deltas = the slot's smoothed cadence */
    int64_t pd_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: residence holds (correct cadence, not starvation) */
    int64_t sv_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: genuine starvation dups (due but buffer empty) */
    int64_t md_cnt[PTV_MAX_INPUT] = {0};      /* v0.9.13: decimation drops (multiple frames due in one tick) */
    MvTlSlot tl_slot[PTV_MAX_INPUT];          /* 2.1 T-409: the slot as a frame synchronizer */
    int     res_occ_tgt = 4;                  /* v0.9.13: occupancy-servo target (set from the primed depth) */
    /* AUDIO-FOLLOW (Option A) latch: average the per-slot lag over a startup window (past the
     * lossy join) and latch a STABLE signed offset, published to house_skew for the audio's
     * one-time deterministic correction. Re-latched on outage return. */
    int64_t  af_off[PTV_MAX_INPUT] = {0};     /* audio-follow: continuously-smoothed per-slot lag (EMA, us) */
    int64_t  af_t0[PTV_MAX_INPUT]  = {0};     /* tick+1 of this slot's first real frame (0 = unseeded) */
    int      h0_logged[PTV_MAX_INPUT] = {0};  /* P0 diag: one-shot PTV-H0 per slot at first display */
    int      done_in[PTV_MAX_INPUT] = {0};
    int64_t rung_pts[PTV_MAX_RUNG] = {0};
    /* 1.0.1-pre16 mv sensor port: per-slot video-side EMA state (the compositor is the ONE
     * writer of g_rsx.mv_ema/mv_wall[*] on mv — the single-input master-rung mirror). τ ≈ 30s
     * of ticks, divisor verbatim from the single-input sensor (ptvencoder_clock.c). */
    int64_t rs_mv_ema[PTV_MAX_INPUT] = {0};
    int     rs_mv_seed[PTV_MAX_INPUT] = {0};
    int64_t rs_mv_div;
    int64_t rs_mv_src[PTV_MAX_INPUT];         /* 2.0.1 T-362: the slot's last displayed source pts */
    int64_t rs_mv_adv[PTV_MAX_INPUT] = {0};   /* 2.0.1 T-362: the tick its displayed picture last changed */
    int64_t rs_mv_rel[PTV_MAX_INPUT] = {0};   /* 2.0.1 T-362: -1 in a hold, else the tick it ended (0 = none) */
    int     born[PTV_MAX_INPUT] = {0};   /* pre17: slot has displayed its first fresh frame
                                          * (clears the slot's birth slate-mask bit below) */
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
    int64_t corr_wd_last = diag_t0;   /* 1.0.1-pre17: stale-track corrector watchdog rate limit (1s) */
    int64_t rsync_in_last = 0;        /* fix round (V6): [PTV-RSYNC] inK summary throttle (30s) */

    if (!filt) goto done;
    /* 1.0.1-pre17: BIRTH slate-mask — every slot starts "slated" for the corrector (its cell
     * is black until its first display; a NEVER-arriving slot — the finding-1 restart shape —
     * previously never set the mask because `stale` requires last_fresh_us > 0). Cleared per
     * slot at its first fresh display. No mv corrector engagement until every cell is live. */
    for (k = 0; k < n; k++)
        if (!c->inputs[k].deferred)   /* 2.0.2 T-386: a slot not live at the start must not freeze the live slots'
                                       * correction until it joins (GAFamily: 3 days without correction) */
            atomic_fetch_or_explicit(&g_mv_slate_mask, 1 << k, memory_order_relaxed);
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
        memset(&tl_slot[k], 0, sizeof(tl_slot[k]));
        tl_slot[k].cad      = mv_tick_us(c, 1);
        tl_slot[k].last_src = AV_NOPTS_VALUE;
        /* 2.1 T-414: until the first compose, a slot keeps twice its primed depth — a mass restart composed ~10 s after
         * the slots went live, each banked to the 160-picture capacity (5.3 s), the preroll trimmed it to the primed
         * depth, and the decoders' buffer pools kept the high-water for the process life (cor-3 RAV 2x2 2416 MB vs
         * 1027 MB). Lifted at the compose (a burst mid-run keeps the full capacity) */
        atomic_store_explicit(&c->inputs[k].hold.depth_cap, FFMAX(2 * res_occ_tgt, 8), memory_order_relaxed);
    }

    /* preroll: prime every input's jitter buffer to ~PTV_PREROLL_MS so bursty
     * decode delivery has a cushion (no startup dup storm) and the mosaic starts
     * with every cell live. A never-arriving input is left to its black cell. */
    {
        int64_t t0;
        const char *ps = getenv("PTV_TEST_MV_PRESTALL_S");   /* TEST-ONLY: the compositor starts N s late while the slot
                                                              * decoders bank (a mass restart: cor-3 composed ~10 s after
                                                              * its slots went live) */
        if (ps && atoi(ps) > 0)
            av_usleep(atoi(ps) * 1000000U);
        t0 = av_gettime_relative();
        for (;;) {
            int ready = 0, eofall = 1;
            for (k = 0; k < n; k++) {
                pthread_mutex_lock(&c->inputs[k].hold.lock);
                if (!c->inputs[k].hold.eof) eofall = 0;
                pthread_mutex_unlock(&c->inputs[k].hold.lock);
                if (c->inputs[k].deferred ||                    /* 2.0.2 T-386: nothing to prime yet */
                    av_thread_message_queue_nb_elems(c->inputs[k].hold.q) >= (n_prime > 0 ? n_prime : 1)) ready++;
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
        for (k = 0; k < n; k++)                       /* T-414: the full capacity from the compose on */
            atomic_store_explicit(&c->inputs[k].hold.depth_cap, 0, memory_order_relaxed);
        wall0 = av_gettime_relative();
    }

    for (;;) {
        int all_eof = 1, any_fresh = 0;
        int64_t now_us;
        PTV_HB_OUT(PTV_HB_OUT_LOOP);   /* 1.0.1-pre21 heartbeat: on mv the compositor IS the
                                        * house clock / stats owner — it stamps the OUT beat
                                        * (single-input: the master output thread stamps it) */
        {                                            /* wall-pace the house tick (also offline:
                                                      * inputs have independent clocks, so the
                                                      * mosaic cadence is the house rate, not media) */
            int64_t target = wall0 + mv_tick_us(c, tick);
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
        }
        now_us = av_gettime_relative();
        /* 2.0.2 T-386: a slot not live at the start has an audio track from the first tick: its audio thread encodes
         * silence up to this tick's output time (the same axis a live slot's audio is anchored on), every 100 ms */
        atomic_store_explicit(&g_mv_house_us, mv_tick_us(c, tick), memory_order_relaxed);
        for (k = 0; k < n; k++) {
            DemuxArgs *da = &c->inputs[k].da;
            int j;
            if (!c->inputs[k].deferred || now_us - mv_pump_us[k] < 100000)
                continue;
            mv_pump_us[k] = now_us;
            for (j = 0; j < da->n_audio; j++) {
                AVPacket *fs = av_packet_alloc();
                if (!fs) continue;
                fs->flags |= PTV_PKT_FLAG_SLOT_FILL;
                if (av_thread_message_queue_send(da->audio_q[j], &fs, AV_THREAD_MESSAGE_NONBLOCK) < 0)
                    av_packet_free(&fs);
            }
        }

        for (k = 0; k < n; k++) {                    /* pop ONE frame from this input's jitter
                                                      * buffer (FIFO) -> feed buffersrc k; dup-hold
                                                      * its last frame on underrun, black when stale */
            VideoHold *h = &c->inputs[k].hold;
            AVFrame *f = NULL, *st; int stale, fresh = 0;
            /* 2.1 T-409: the slot is a frame synchronizer (mv_tl_pick): the picture due at its content time and
             * lag, a lost one repeated, a late one skipped. */
            AVFrame *cand = mv_tl_pick(c, k, tick, now_us, &pending[k], &done_in[k], &tl_slot[k], res_occ_tgt,
                                       &rs_mv_seed[k]);
            if (!cand && last[k]) {             /* the stats line's pd= / sv=: a repeat by cadence (the next
                                                 * picture waits for its time) or by starvation (none queued) */
                if (pending[k] || av_thread_message_queue_nb_elems(h->q))
                    pd_cnt[k]++;
                else
                    sv_cnt[k]++;
            }
            fresh = (cand != NULL);
            if (fresh) {
                f = cand;                                 /* for the re-anchor diag below */
                if (last[k]) av_frame_free(&last[k]);
                last[k] = cand; any_fresh = 1; last_fresh_us[k] = now_us;
                if (c->inputs[k].deferred) {             /* 2.0.2 T-386: the slot's first picture — it joins here */
                    c->inputs[k].deferred = 0;
                    av_log(NULL, AV_LOG_WARNING, "[PTV-MV] in%d joined: first picture at tick %"PRId64" (%.3f s)\n",
                           k, tick, mv_tick_us(c, tick) / 1e6);
                }
                if (!born[k]) {                           /* pre17: first display — the slot is live;
                                                           * clear its BIRTH slate-mask bit */
                    born[k] = 1;
                    atomic_fetch_and_explicit(&g_mv_slate_mask, ~(1 << k), memory_order_relaxed);
                }
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
                    /* 1.0.1-pre17: slate RECOVERY edge — clear this slot's bit; the corrector's
                     * sibling-slate freeze lifts and its 3min quiet window re-runs from here.
                     * Also re-seed the slot's m_v EMA: the outage gap + skew reset changed the
                     * mapping baseline; pre-outage samples must not blend into the fresh one
                     * (the soak's "readings snap back on recovery" made exact). */
                    atomic_fetch_and_explicit(&g_mv_slate_mask, ~(1 << k), memory_order_relaxed);
                    rs_mv_seed[k] = 0;
                    atomic_fetch_add_explicit(&c->inputs[k].house_disturb, 1, memory_order_relaxed);  /* B3: arm PLL mid-run re-acquire */
                    /* audio-follow re-tracks continuously via its EMA — no per-outage reset needed */
                }
            }
            if (!done_in[k]) all_eof = 0;
            stale = (c->slate_after_us > 0 && last_fresh_us[k] > 0 && now_us - last_fresh_us[k] > c->slate_after_us);
            if (stale && !slated[k]) {                /* slate ONSET edge: mark the outage (next fresh frame
                                                       * re-anchors) + set this slot's bit in the sibling-slate
                                                       * mask (1.0.1-pre17) — no mv corrector may engage while
                                                       * any slot is slated (finding 1's condition). */
                slated[k] = 1;
                atomic_fetch_or_explicit(&g_mv_slate_mask, 1 << k, memory_order_relaxed);
            }
            /* 2.0.0 T-093 part 2: a slot whose source stops altogether (audio AND video) — its demux thread is
             * blocked in the read, so the per-slot silence fill of part 1 (sent on video packet arrival) never
             * runs and the slot's audio PID went dark on the wire for the whole outage, then the PLL padded the
             * hole in one burst at the return (fixture kill_return_20: 21 s dark, ACQUIRE pad 19.3 s). The
             * compositor keeps ticking: once the cell has shown no fresh frame for 1 s and the track has decoded
             * nothing for 2 s, it sends the same fill sentinels (the quantum fills the wall time elapsed, so a
             * sentinel from the demux as well never doubles the silence). PTV_NO_SRC_FILL=1 off. */
            /* 2.0.1 T-099: ... and once the slot's input has read nothing at all for 1 s (its demux heartbeat), after 1 s
             * of no decoded audio. The cell plays out its jitter buffer (~1 s) before it shows no fresh frame, so both
             * waits added up to 2 s past the source stop while the delivered audio covered only ~1 s of it: MV-04
             * gap_20 slot 0 audio PID dark 1.27 s, and the composite video 0.97 s (the video delivery gate keys on the
             * slowest live track, which this one stayed for 2 s). */
            int64_t hbw = c->inputs[k].da.rsync_slot >= 0 && c->inputs[k].da.rsync_slot < PTV_MAX_INPUT ?
                          atomic_load_explicit(&g_hb_demux_wall[c->inputs[k].da.rsync_slot], memory_order_relaxed) : 0;
            int quiet = hbw && now_us - hbw > 1000000;
            if (g_hold_fill && g_glueclass && last_fresh_us[k] > 0 && (quiet || now_us - last_fresh_us[k] > 1000000)) {
                DemuxArgs *da = &c->inputs[k].da;
                int64_t nw = av_gettime_relative();
                int j;
                for (j = 0; j < da->n_audio; j++) {
                    int g = da->aglobal[j];
                    int64_t fw = g >= 0 && g < PTV_MAX_AUDIO ? atomic_load_explicit(&g_adec_frame_wc[g], memory_order_relaxed) : 0;
                    AVPacket *fs;
                    if (!fw || nw - fw < (quiet ? 1000000 : 2000000) || nw - mv_fill_us[k][j] < g_nbs_quantum_us)
                        continue;
                    mv_fill_us[k][j] = nw;
                    if (!(fs = av_packet_alloc())) continue;
                    fs->stream_index = da->astream[j];
                    fs->flags |= PTV_PKT_FLAG_NBS_FILL;
                    if (av_thread_message_queue_send(da->audio_q[j], &fs, AV_THREAD_MESSAGE_NONBLOCK) < 0)
                        av_packet_free(&fs);
                }
            }
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
                        /* 2.0.1 T-362 (measured, mv04 sync_gap_20): through a slot's outage the cell keeps its last
                         * picture, so m grew 1 s per second and the EMA reached +18.5 s by the return; m was back at
                         * its pre-outage value on the first fresh picture, but lipsync= read +19.9 s decaying over
                         * minutes (sync_check's pin rule fires on 3 readings over 2 s). A picture held 0.5 s or more
                         * is no content to measure: the slot stops publishing (reads `--`, as a slated slot) and the
                         * EMA reseeds 1 s after the pictures move again — the first one after a gap can be the
                         * pre-gap tail (lavf PES tail), shown 20 s late, which seeded +19.9 s once. Short cadence /
                         * residence holds still count. */
                        if (disp_src != rs_mv_src[k] || !rs_mv_adv[k]) {
                            rs_mv_src[k] = disp_src;
                            rs_mv_adv[k] = tick;
                        }
                        if (mv_tick_us(c, tick) - mv_tick_us(c, rs_mv_adv[k]) >= 500000) {
                            rs_mv_seed[k] = 0;
                            rs_mv_rel[k]  = -1;
                        } else if (rs_mv_rel[k] == -1)
                            rs_mv_rel[k] = tick;
                        else if (rs_mv_rel[k] > 0 && mv_tick_us(c, tick) - mv_tick_us(c, rs_mv_rel[k]) < 1000000)
                            ;
                        else {
                            rs_mv_rel[k] = 0;
                            if (!rs_mv_seed[k]) { rs_mv_ema[k] = m; rs_mv_seed[k] = 1; }
                            else rs_mv_ema[k] += (m - rs_mv_ema[k]) / rs_mv_div;
                            atomic_store_explicit(&g_rsx.mv_ema[k], rs_mv_ema[k], memory_order_relaxed);
                            atomic_store_explicit(&g_rsx.mv_wall[k], now_us, memory_order_relaxed);
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
                    if (fresh) {
                        /* 2.0.0 T-089: one continuous content segment per ring. A slot returning from an
                         * outage on a new timeline (or any >2 s label step) starts a fresh ring: the audio
                         * pairing is nearest-by-content and extrapolates by the content distance, so stale
                         * pre-gap entries paired the returning audio across the hole and the PLL padded the
                         * outage a second time (live RAV: pad 54.5 s, then drop 55.9 s of real audio). */
                        VOutRing *vr = &c->inputs[k].vring;
                        pthread_mutex_lock(&vr->lock);
                        if (vr->n && llabs(disp_src - vr->src[(vr->n - 1) % PTV_VRING]) > 2000000) {
                            vr->n = 0;
                            vr->seg_wc = av_gettime_relative();
                        }
                        pthread_mutex_unlock(&vr->lock);
                        vring_put(vr, disp_src, mv_tick_us(c, tick));
                        if (g_vmap)          /* 2.0.0 T-342: release the slot's copies this frame has caught up to */
                            ptv_copy_vwait(&c->inputs[k].da);
                    }
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
            /* 2.1 T-411/T-414: the slot's sound may be placed up to the shown picture + the primed margin + 1 s while
             * the cell advances (ptvencoder_audio.c parks the rest: a sound placed up to the decode front put every
             * step's boundary past the pictures waiting — a catch-up or a burst re-locked and never committed, and a
             * step moved picture and sound apart by what was placed ahead). A cell with no fresh picture for 1.5 s
             * (held, dead, a video stall) lifts the bound: the sound plays on as before */
            {
                int64_t alim = INT64_MAX;
                if (last[k] && !stale && last[k]->pts != AV_NOPTS_VALUE && now_us - last_fresh_us[k] <= 1500000)
                    alim = av_rescale_q(last[k]->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) +
                           (int64_t)res_occ_tgt * c->tick_dur_us + 1000000;
                atomic_store_explicit(&c->inputs[k].tl_alim_us, alim, memory_order_relaxed);
            }
            if (last[k] && !stale) {
                st = av_frame_clone(last[k]);
                cell_dead_us[k] = 0;
            } else {
                /* 2.0.0 (owner 2026-10-13, Cinestar outage on Grid_2x2: a dead cell was black — the same as a live
                 * channel sending black): a dead cell (no fresh frame for slate_after_us, or none at all that long
                 * after the start) shows the no-signal picture single input shows (-hold: bars by default), in the
                 * black frame's shape so the slot's buffersrc sees no parameter change; black if it cannot be built. */
                int64_t nwc = av_gettime_relative();
                st = NULL;
                if (last[k] || c->inputs[k].deferred ||          /* 2.0.2 T-386: placeholder from the first tick */
                    (c->slate_after_us > 0 && nwc - diag_t0 > c->slate_after_us)) {
                    if (!cell_dead_us[k]) {
                        cell_dead_us[k] = nwc;
                        av_log(NULL, AV_LOG_WARNING, "[PTV-MV] in%d no signal \xe2\x86\x92 %s in its cell\n", k,
                               g_hold_mode == PTV_HOLD_BARS ? "bars" : g_hold_mode == PTV_HOLD_SLATE ? "slate" : "black");
                    }
                    AVFrame *bp = blackf[k] ? ptv_hold_cell_pic(&cellpic[k], blackf[k], nwc - cell_dead_us[k]) : NULL;
                    /* the black frame's own properties, only the pixels replaced: the slot's buffersrc must see
                     * exactly the frame it was configured for */
                    if (bp && (st = av_frame_clone(blackf[k])) &&
                        (av_frame_make_writable(st) < 0 || av_frame_copy(st, bp) < 0))
                        av_frame_free(&st);
                    av_frame_free(&bp);
                }
                if (!st)
                    st = blackf[k] ? av_frame_clone(blackf[k]) : NULL;
            }
            if (!st) continue;
            /* 2.0.2 T-386: the slot's buffersrc was configured for blackf's geometry (its own probe, or the first live
             * sibling's for a slot that joined later). A frame in another size passes buffersrc with only a warning
             * and bwdif / yadif then read it with the configured size — memory-unsafe (also a slot that reopened on a
             * reshaped source). Such a frame is scaled to the cell's geometry. */
            if (blackf[k] && (st->width != blackf[k]->width || st->height != blackf[k]->height ||
                              st->format != blackf[k]->format)) {
                AVFrame *nf = av_frame_alloc();
                norm_sws[k] = sws_getCachedContext(norm_sws[k], st->width, st->height, st->format,
                                                   blackf[k]->width, blackf[k]->height, blackf[k]->format,
                                                   SWS_BICUBIC, NULL, NULL, NULL);
                if (!norm_logged[k]) {
                    norm_logged[k] = 1;
                    av_log(NULL, AV_LOG_WARNING, "[PTV-MV] in%d pictures are %dx%d %s, its cell %dx%d %s — scaled to the "
                           "cell\n", k, st->width, st->height, av_get_pix_fmt_name(st->format) ? av_get_pix_fmt_name(st->format) : "?",
                           blackf[k]->width, blackf[k]->height,
                           av_get_pix_fmt_name(blackf[k]->format) ? av_get_pix_fmt_name(blackf[k]->format) : "?");
                }
                if (nf && norm_sws[k]) {
                    nf->width = blackf[k]->width; nf->height = blackf[k]->height; nf->format = blackf[k]->format;
                    if (av_frame_get_buffer(nf, 0) >= 0 && av_frame_copy_props(nf, st) >= 0 &&
                        sws_scale(norm_sws[k], (const uint8_t * const *)st->data, st->linesize, 0, st->height,
                                  nf->data, nf->linesize) > 0) {
                        av_frame_free(&st);
                        st = nf; nf = NULL;
                    }
                }
                av_frame_free(&nf);
                if (st->width != blackf[k]->width || st->height != blackf[k]->height || st->format != blackf[k]->format) {
                    av_frame_free(&st);                  /* could not normalise: never feed it */
                    st = av_frame_clone(blackf[k]);
                    if (!st) continue;
                }
            }
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
        if (g_t_us > 0 && mv_tick_us(c, c->emitted) >= g_t_us &&
            !atomic_load_explicit(&g_t_stop, memory_order_relaxed))
            atomic_store_explicit(&g_t_stop, 1, memory_order_relaxed);   /* rider (b): -t (mv cadence owner) */

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
                char dlv[112] = "";                  /* §7.5a delivery gate readout (mv gated since v0.9.12.1) */
                if (c->gate0) {
                    int dn = snprintf(dlv, sizeof dlv, " dlvhold=%"PRId64"ms dlvforced=%"PRId64,
                             atomic_load_explicit(&c->gate0->st_hold_us, memory_order_relaxed) / 1000,
                             atomic_load_explicit(&c->gate0->st_forced, memory_order_relaxed));
                    if (c->gate0->v_on && dn > 0 && dn < (int)sizeof dlv) {
                        /* §7.5b symmetric gate on mv (task #48): EARLY-VIDEO hold ≈ the audio
                         * path's wall latency (loudnorm ~3s fill); vdlvforced only when the
                         * backstop released */
                        int64_t vf = atomic_load_explicit(&c->gate0->st_vforced, memory_order_relaxed);
                        dn += snprintf(dlv + dn, sizeof dlv - dn, " vdlvhold=%"PRId64"ms",
                                 atomic_load_explicit(&c->gate0->st_vhold_us, memory_order_relaxed) / 1000);
                        if (vf > 0 && dn > 0 && dn < (int)sizeof dlv)
                            snprintf(dlv + dn, sizeof dlv - dn, " vdlvforced=%"PRId64, vf);
                    }
                }
                char aco[24] = "";                   /* pre15/pre16 #33: corrupt-discarded AUDIO pkts, GLOBAL sum
                                                      * (owner Q — per-track detail stays on the [PTV-ADISC]/NBS
                                                      * lines; absent while zero, clean line unchanged) */
                {
                    int64_t ac = atomic_load_explicit(&g_acorrupt, memory_order_relaxed);
                    if (ac > 0)
                        snprintf(aco, sizeof aco, " acor=%lld", (long long)ac);
                }
                /* pre16.1 (owner-directed): the per-slot sensor reading rides INSIDE each inN:
                 * group below (/lipsync=), not as a separate combined token — the mv line
                 * already keys per input. `--` on a slated slot IS the outage signal.
                 * pre17: corr= joins the line (the mv corrector is ARMED) — the shared pre14
                 * builder with the aK: prefix forced; absent while every track is quiet. */
                char crs[10 + PTV_MAX_AUDIO * 20];
                ptv_stats_corr(crs, sizeof crs, 1);
                char cvs[10 + PTV_MAX_AUDIO * 16];   /* pre23 rr23: conv= (aK: forced, as corr=);
                                                      * absent while every track's fold total is 0 */
                ptv_stats_conv(cvs, sizeof cvs, nows, 1);
                char ls[768]; int lp = 0;   /* 4 slots x ~122 chars once the counters grow: at 448
                                             * the 4th slot's whole token silently vanished — the
                                             * one the per-slot cc= exists to make visible */   /* per-slot: qdrop=input-q overflow, corrupt=demux+decode,
                                             * pd=cadence holds (NORMAL for a rate-mismatched slot),
                                             * sv=starvation dups, sk=published audio skew,
                                             * skres=LAYERA erase-residue ledger (0.9.18.7, same
                                             * accounting as single-input hsres= — the slot's sk
                                             * measurement rides the same erased label stream) */
                for (k = 0; k < n && lp < (int)sizeof ls - 144; k++) {
                    char lsin[40], lst[56] = "", cct[32] = "";
                    ptv_stats_lipsync_in(lsin, sizeof lsin, nows, k);
                    if (lsin[0])
                        snprintf(lst, sizeof lst, "/lipsync=%s", lsin);
                    /* -cc_extract: THIS slot's captions/a53. The fleet-wide cc= of the
                     * single-input line cannot say which slot of a mosaic went dark, which is
                     * the only CC failure that matters here. Present only on slots the
                     * extraction actually runs on (-cc_slots), so the line is unchanged when
                     * the feature is off. */
                    if (c->inputs[k].dc.cc.on)
                        snprintf(cct, sizeof cct, "/cc=%lld/%lld",
                                 (long long)atomic_load_explicit(&g_cc_caps_in[k], memory_order_relaxed),
                                 (long long)atomic_load_explicit(&g_cc_a53_in[k],  memory_order_relaxed));
                    lp += snprintf(ls + lp, sizeof ls - lp,
                                   " in%d:qdrop=%"PRId64"/corrupt=%"PRId64"/pd=%"PRId64"/sv=%"PRId64"/sk=%+dms/skres=%+dms%s%s",
                                   k, c->inputs[k].da.vdrop, c->inputs[k].da.vcorrupt + c->inputs[k].dc.vcorrupt,
                                   pd_cnt[k], sv_cnt[k],
                                   (int)(atomic_load_explicit(&c->inputs[k].tl_lag_us, memory_order_relaxed) / 1000),
                                   (int)(c->inputs[k].da.disc_resid_us / 1000), lst, cct);
                }
                /* GLOBAL ccmux= — teletext packets our own mux backward-dts guard dropped.
                 * Its own buffer rather than an append to ls[]: that loop's bound only
                 * guarantees headroom for one more slot token, and this line already lost a
                 * whole token to a too-small buffer once. Not per-slot: the guard sits in the
                 * mux thread, downstream of the fan-out, so the count is not attributable to a
                 * slot — and it is per rung per packet (see the log legend). Printed only when
                 * the extraction runs at all, so an mv without -cc_extract is unchanged. */
                char ccg[32] = "";
                {   /* shown only when NONZERO, exactly like the single-input line and what the
                     * log legend promises — a clean run's line stays byte-for-byte unchanged. */
                    int64_t cx = atomic_load_explicit(&g_cc_mux, memory_order_relaxed);
                    if (cx > 0)
                        snprintf(ccg, sizeof ccg, " ccmux=%lld", (long long)cx);
                }
                av_log(NULL, AV_LOG_INFO,   /* v0.9.13 parity: size/bitrate/speed/genlock dropped (v0.9.10 single-input rationale) */
                    "frame=%6"PRId64" fps=%4.1f time=%02d:%02d:%05.2f dup=%"PRId64" drop=%"PRId64"%s%s%s%s%s%s\n",
                    c->emitted, fps, hh, mm, ss, c->dup, c->framedrop[0], dlv, aco, ls, ccg, crs, cvs);
                /* 1.0.1-pre17 (owner-floated): per-INPUT always-on [PTV-RSYNC] summary, ONE
                 * compact line per input — the soak forensics that used to need PTV_DIAG.
                 * R = this input's track reading(s) (same builder as the stats token), ev =
                 * the slot's demux edit ledger, sk = published follow skew, occ = jitter-
                 * buffer depth, SLATED while the cell is black. Fix round (reviewer V6):
                 * THROTTLED to one round per 30s (~11.5k lines/day on a 4-up, vs ~35k
                 * unthrottled at a 10s stats period) — readability kept, log budget bounded. */
                if (g_diag && g_rsync_sense && nows - rsync_in_last >= 30000000) {   /* 2.0.0: DIAG — the
                     * default log is the stats line + events (owner); R is the stats line's inK:lipsync */
                    rsync_in_last = nows;
                    for (k = 0; k < n; k++) {
                        char rin[40];
                        ptv_stats_lipsync_in(rin, sizeof rin, nows, k);
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-RSYNC] in%d R=%s ev=%+lldms sk=%+lldms occ=%d%s  [+ = audio early]\n",
                            k, rin[0] ? rin : "n/a",
                            (long long)(atomic_load_explicit(&g_rsx.ev_us[k], memory_order_relaxed) / 1000),
                            (long long)(c->inputs[k].house_skew / 1000),
                            av_thread_message_queue_nb_elems(c->inputs[k].hold.q),
                            slated[k] ? " SLATED" : "");
                    }
                }
                stat_last = nows; stat_prev = c->emitted;
            }
        }
        {   /* 1.0.1-pre17: the pre14 stale-track corrector watchdog runs under the MV cadence
             * owner too (the passthrough rung loops return before the single-input master
             * block, so it never ran on mv — port-doc §6's named arming prerequisite). */
            int64_t nww = av_gettime_relative();
            if (g_rsync_corr && nww - corr_wd_last >= 1000000) {
                corr_wd_last = nww;
                rscorr_stale_watchdog(nww);
            }
        }
        if (g_slow) av_usleep(g_slow);
        if (all_eof) break;                          /* every input terminated -> tear down */
    }

    for (k = 0; k < n; k++) { int fr = av_buffersrc_add_frame(c->fsrc[k], NULL); (void)fr; }   /* flush graph */
    for (k = 0; k < n; k++) sws_freeContext(norm_sws[k]);   /* 2.0.2 T-386 */
    for (r = 0; r < R; r++)
        while (av_buffersink_get_frame(c->fsink[r], filt) >= 0) {
            AVFrame *out = av_frame_alloc();
            if (out) { av_frame_move_ref(out, filt); out->pts = rung_pts[r]++; out->pkt_dts = AV_NOPTS_VALUE; out->duration = 0;
                       push_frame_q(c->frame_q[r], c->live, &c->framedrop[r], out); }
            else     { av_frame_unref(filt); }
        }
done:
    av_frame_free(&filt);
    for (k = 0; k < n; k++) ptv_hold_cell_free(&cellpic[k]);
    for (k = 0; k < n; k++) { if (blackf[k]) av_frame_free(&blackf[k]); if (last[k]) av_frame_free(&last[k]); if (pending[k]) av_frame_free(&pending[k]); }
    for (r = 0; r < R; r++) av_thread_message_queue_set_err_recv(c->frame_q[r], AVERROR_EOF);
    return NULL;
}

