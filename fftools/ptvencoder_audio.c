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
 * window opened (per-track mark refreshed while quiet). */
static const char *ptv_self_shed_note(AudioState *a, char *buf, size_t sz)
{
    int64_t w = atomic_load_explicit(&g_shed_wall, memory_order_relaxed);
    int64_t c = atomic_load_explicit(&g_shed_cnt, memory_order_relaxed);
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
                    if (g_pll_testnoise_us)                  /* TEST-ONLY: ±N square wave (~7s flip, matches the box thrash period; holds long enough to defeat the debounce like the real noise) to reproduce the box limit cycle locally */
                        off += ((a->out_frames / 330) & 1) ? g_pll_testnoise_us : -g_pll_testnoise_us;
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
                            /* 0.9.18.7: promoted PTV_DIAG→always-on WARNING. An ACQUIRE is a discrete
                             * audio drop/pad (a bank snap, not a TRACK bleed) — rare in normal
                             * operation (startup bank + real disturbances) and already hard
                             * rate-limited by the 12s post-acquire refractory (≤1/12s per track by
                             * construction), so no extra rate limit is added. */
                            av_log(NULL, AV_LOG_WARNING, "[PTV-PLL] a%d(in%d) ACQUIRE %s %"PRId64"ms (ema→%"PRId64"ms applied=%"PRId64"ms #%d)\n",
                                   a->dbg_k, a->dbg_in, dq < 0 ? "drop" : "pad", FFABS(dq) / 1000,
                                   a->pll_ema / 1000, a->af_applied_us / 1000, a->pll_acq_count);
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
                if (!(a->multiview && g_audio_follow) && a->house_skew)
                    content -= *a->house_skew;
                else if (a->multiview && g_audio_follow && g_avsync_pll)
                    content -= a->af_steer_us;
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
                    /* lipsync = the [PTV-LIPSYNC] err folded into the always-on line (the operator's
                     * headline A/V number; no PTV_DIAG needed). It is the faithful pipeline-introduced
                     * lip-sync error: the AUDIO's realized output-vs-content lag (async_pad = outspan −
                     * content_span) minus the VIDEO's TRUE lag (lag_true). + = audio late. (`offset`
                     * below is the independent vring-paired cross-check; lipsync>0 ≈ offset<0.) */
                    int64_t content_us = av_rescale_q(a->dbg_last_src - a->dbg_first_src, a->ist_tb, AV_TIME_BASE_Q);
                    int64_t outspan_us = a->out_frames * (int64_t)a->frame_size * 1000000 / a->out_rate;
                    int64_t lag_true   = a->house_lag_true ? *a->house_lag_true : lag;
                    int64_t lserr      = (outspan_us - content_us) - lag_true;   /* async_pad − lag_true */
                    /* v0.6.19: the async_pad span estimate (lserr) does NOT include the PLL's content
                     * drop/pad retiming (af_applied_us), so on a CONVERGED PLL slot it kept reporting the
                     * bank the acquire already removed (lipsync ≈ applied) — reading "off" while the
                     * faithful vring-paired offset was ~0. Headline the faithful measured offset when it
                     * has paired (− because offset<0 = audio late ≡ lipsync>0 = audio late); fall back to
                     * the span estimate only before the vring pairs (offset = --). */
                    int64_t lshead     = a->av_off_valid ? -a->av_offset_us : lserr;
                    char m[24];
                    if (a->av_off_valid) snprintf(m, sizeof m, "%+"PRId64"ms", a->av_offset_us / 1000);
                    else                 snprintf(m, sizeof m, "--");
                    if (mv && g_avsync_pll)  /* B3 closed loop: measured offset + integrator state + acquire/guard counts */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) lipsync=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "pll[ema=%+"PRId64"ms dev=%"PRId64"ms applied=%+"PRId64"ms steer=%+"PRId64"ms acq=%d guard=%"PRId64" drop=%"PRId64"ms pad=%"PRId64"ms acomp=%"PRId64"]"
                            "  [offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            a->pll_ema / 1000, a->pll_dev / 1000, a->af_applied_us / 1000, a->af_steer_us / 1000,
                            a->pll_acq_count, a->pll_guard_fires,
                            a->af_acq_drop_us / 1000, a->af_acq_pad_us / 1000, a->acomp_cnt);
                    else if (mv)         /* multiview (open-loop B1): lip-sync + measured offset + the per-slot actuator state */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) lipsync=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "house_skew=%+"PRId64"ms applied=%+"PRId64"ms trk=%+"PRId64"ms nudge=%+"PRId64"ms "
                            "acq[drop=%"PRId64"ms pad=%"PRId64"ms]  [lipsync>0 / offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            lag / 1000, a->af_applied_us / 1000, (lag - a->af_applied_us) / 1000,
                            a->af_nudge_us / 1000, a->af_acq_drop_us / 1000, a->af_acq_pad_us / 1000);
                    else                 /* single-input: lip-sync + measured offset + house-clock lock state */
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-AVSYNC] a%d(in%d) lipsync=%+"PRId64"ms | offset=%s (vlag=%+"PRId64"ms alag=%+"PRId64"ms) "
                            "house_skew=%+"PRId64"ms  [lipsync>0 / offset<0 = audio late]\n",
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
            if (a->afg) { avfilter_graph_free(&a->afg); a->afsrc = a->afsink = NULL; a->fg_swr = NULL; }
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
            a->fg_in_rate = a->dec->sample_rate;   /* the path is now configured for these */
            a->fg_in_fmt  = a->dec->sample_fmt;
            av_channel_layout_uninit(&a->fg_in_chl);
            av_channel_layout_copy(&a->fg_in_chl, &a->dec->ch_layout);
            av_channel_layout_uninit(&a->afmt_pending_chl);
            a->afmt_pending_rate = 0; a->afmt_pending_fmt = AV_SAMPLE_FMT_NONE;
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
                    int exp_hit = 0;
                    int64_t exp_step = 0;
                    if (a->glue_exp_dl) {
                        int64_t dl = atomic_load_explicit(a->glue_exp_dl, memory_order_acquire);
                        if (dl) {
                            exp_step = atomic_load_explicit(a->glue_exp_step, memory_order_relaxed);
                            if (now_wc <= dl &&
                                step - exp_step >= -PTV_PAIR_EXPECT_LO_US &&
                                step - exp_step <=  PTV_PAIR_EXPECT_HI_US) {
                                atomic_store_explicit(a->glue_exp_dl, 0, memory_order_relaxed);  /* consumed */
                                exp_hit = 1;
                            } else if (now_wc > dl) {
                                atomic_store_explicit(a->glue_exp_dl, 0, memory_order_relaxed);  /* expired */
                            }
                        }
                    }
                    char sn[48];   /* 1.0.1-pre8 (d): self-shed honesty note ("" when nothing shed) */
                    if (llabs(step) > (int64_t)g_aglue_max_ms * 1000) {
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms above %dms cap — left to the discontinuity layer%s%s\n",
                               a->dbg_k, a->dbg_in, step / 1000, g_aglue_max_ms,
                               exp_hit ? " (matches the shared-flush expected step — aresample converges it)" : "",
                               ptv_self_shed_note(a, sn, sizeof sn));
                    } else if (exp_hit) {
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-AGLUE] a%d(in%d) label step %+"PRId64"ms matches the shared-flush expected step %+"PRId64"ms "
                               "— REAL A-vs-V alignment step, APPLIED (aresample converges; not erased)%s\n",
                               a->dbg_k, a->dbg_in, step / 1000, exp_step / 1000,
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
    if (!codec || !a->ist)
        return;
    nd = avcodec_alloc_context3(codec);
    if (!nd)
        return;
    avcodec_parameters_to_context(nd, a->ist->codecpar);
    nd->pkt_timebase = a->ist->time_base;
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
        if (av_samples_alloc_array_and_samples(&out, NULL, a->out_chl.nb_channels,
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
    for (unsigned i = 0; i < a->afg->nb_filters; i++) {
        AVFilterContext *fc = a->afg->filters[i];
        if (fc && fc->filter && !strcmp(fc->filter->name, "aresample") && fc->priv) {
            SwrContext *cand = av_opt_child_next(fc->priv, NULL);
            if (cand) {
                int64_t as = 0;
                av_opt_get_int(cand, "async", 0, &as);
                if (!a->fg_swr || as) a->fg_swr = cand;
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

