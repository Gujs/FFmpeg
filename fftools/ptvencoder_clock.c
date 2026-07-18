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

/* PTV-EMPTY: edge-triggered per-queue starvation logger. Logs a refill line with the empty duration
 * for episodes >=200ms, plus (only when hb != NULL) a 5s chronic heartbeat while empty. Returns the
 * episode duration (us) when a >=200ms episode just ended, else 0 — the adaptive cushion feeds on
 * frame_q episodes. v0.9.10: no heartbeat for video_q/mux_q (empty is their NORMAL state — the
 * consumer drains them instantly; a healthy channel reads "empty" at every sample. Only frame_q
 * empty means real starvation → it keeps the heartbeat). */
static int64_t ptv_empty_watch(const char *name, int depth, int64_t now,
                               int64_t *since, int64_t *hb, int64_t log_thresh_us)
{
    if (depth == 0) {
        if (*since == 0) { *since = now; if (hb) *hb = now; }  /* enter empty silently (normal 0-crossings are noise) */
        else if (hb && now - *hb >= 5000000) {                 /* chronic: heartbeat every 5s so "empty now" is visible */
            *hb = now;
            av_log(NULL, AV_LOG_INFO, "[PTV-EMPTY] %s still empty %lldms\n",
                   name, (long long)((now - *since) / 1000));
        }
    } else if (*since) {
        int64_t dur = now - *since;
        *since = 0;
        if (dur >= 200000) {                                   /* episodes >=200ms are real starvation, not tick jitter */
            if (dur >= log_thresh_us)                          /* 0.9.10.1: per-episode line only above the caller's threshold
                                                                * (frame_q passes 2s — sub-2s episodes go to the 60s SUMMARY;
                                                                * a 23.976-film segment produced one line every few seconds) */
                av_log(NULL, AV_LOG_INFO, "[PTV-EMPTY] %s refilled after %lldms empty\n",
                       name, (long long)(dur / 1000));
            return dur;
        }
    }
    return 0;
}
/* PTV_NVENC_SERIALIZE (2026-07-06 scale incident, opt-in): serialize all rung threads' video
 * encoder calls behind ONE process-wide mutex. Rationale: each avcodec_send/receive on NVENC
 * enters the NVIDIA Resource-Manager rwlock via ioctl; 6 rung threads x N processes contending
 * it collapses the driver lock into osq_lock spinning (measured 32% of box CPU at 56 channels;
 * ffmpeg drives the same encoders from ONE thread per process at sys=5%). Serializing cuts this
 * process's concurrent RM callers 6 -> 1. Costs sub-tick wall jitter only: pacing sleeps, PTS
 * math and the delivery gate are untouched (the gate drain runs OUTSIDE the lock so a full
 * mux_q can never stall sibling rungs behind the mutex). Default OFF until soaked. */
int             g_nvenc_serialize = 0;
static pthread_mutex_t g_enc_serial_lock = PTHREAD_MUTEX_INITIALIZER;

/* Drain an encoder, pushing packets to the mux queue. frame=NULL flushes. When `gate` is set, the
 * video front (the newest emitted DTS) is published and the held audio/copy is released in lockstep
 * (§7.5a). Video packets ALWAYS go straight to mux_q — they are the gating front, never held. */
static int encode_push_inner(AVThreadMessageQueue *mux_q, AVCodecContext *enc,
                             AVStream *ost, AVFrame *frame, DlvGate *gate, int *need_drain)
{
    int ret;
    /* Let the ENCODER choose the GOP: clear the decoder's leftover I/P/B
     * classification. Otherwise mpeg2video (and any pict_type-honoring encoder)
     * tries to replicate the source's frame types — h264's long B-runs trip
     * "too many B-frames in a row" and stall; NVENC's forced-IDR GOP can misalign. */
    if (frame)
        frame->pict_type = AV_PICTURE_TYPE_NONE;
    ret = avcodec_send_frame(enc, frame);
    if (ret < 0)
        return ret;
    for (;;) {
        AVPacket *pkt = av_packet_alloc();
        if (!pkt)
            return AVERROR(ENOMEM);
        ret = avcodec_receive_packet(enc, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            if (gate) *need_drain = 1;    /* video front advanced this call → release caught-up audio/copy (outside the serialize lock) */
            return 0;
        }
        if (ret < 0) {
            av_packet_free(&pkt);
            return ret;
        }
        av_packet_rescale_ts(pkt, enc->time_base, ost->time_base);
        pkt->stream_index = ost->index;
        {
            int64_t dts_us = gate && pkt->dts != AV_NOPTS_VALUE
                           ? av_rescale_q(pkt->dts, ost->time_base, AV_TIME_BASE_Q) : AV_NOPTS_VALUE;
            /* §7.5b invariant: publish the encoder front BEFORE the packet can be held — the
             * audio gate's release key (v_enc_dts_hi) must never depend on video DELIVERY
             * (see the symmetric-gate header in ptvencoder_gate.c). Timing-invisible when the
             * video hold is off: the only v_enc_dts_hi reader is dlv_drain, which runs after
             * this whole call. */
            if (gate && dts_us != AV_NOPTS_VALUE) dlv_publish_video(gate, dts_us);
            if (gate && gate->v_on && dts_us != AV_NOPTS_VALUE) {
                ret = dlv_video_deliver(gate, pkt, dts_us);       /* §7.5b: send now, or hold EARLY video */
                if (ret < 0)
                    return ret;                                   /* mux gone */
            } else {
                ret = av_thread_message_queue_send(mux_q, &pkt, 0);   /* blocking; video bypasses the gate */
                if (ret < 0) {
                    av_packet_free(&pkt);
                    return ret;                                   /* mux gone */
                }
            }
        }
    }
}

static int encode_push(AVThreadMessageQueue *mux_q, AVCodecContext *enc,
                       AVStream *ost, AVFrame *frame, DlvGate *gate)
{
    int need_drain = 0, ret;
    if (g_nvenc_serialize) pthread_mutex_lock(&g_enc_serial_lock);
    ret = encode_push_inner(mux_q, enc, ost, frame, gate, &need_drain);
    if (g_nvenc_serialize) pthread_mutex_unlock(&g_enc_serial_lock);
    if (need_drain && gate) {
        dlv_drain(gate);
        dlv_video_drain(gate);   /* §7.5b: the drain above advanced a_dlv_dts_hi — release caught-up video */
    }
    return ret;
}

/* Content index of a source pts on the house grid — THE single copy of the stamping arithmetic
 * (EXACTTICK exact-rational, integer-tick fallback). The v0.9.11 pulldown lookahead uses the SAME
 * function for its hold decision so lookahead and stamp can never disagree (a diverging second
 * copy would reintroduce the monotonic-guard ratchet). Returns -1 when not computable. */
static int64_t content_index(VideoCtx *v, int64_t src_pts)
{
    int64_t house_us;
    if (src_pts == AV_NOPTS_VALUE || *v->h0 == AV_NOPTS_VALUE) return -1;
    house_us = av_rescale_q(src_pts, v->out_tb, AV_TIME_BASE_Q) - *v->h0;
    if (house_us < 0) house_us = 0;
    if (g_exacttick && v->out_fps.num > 0)
        return av_rescale_rnd(house_us, v->out_fps.num, 1000000LL * v->out_fps.den, AV_ROUND_NEAR_INF);
    return (house_us + v->tick_dur_us / 2) / v->tick_dur_us;
}

/* 0.9.18 R2 (map §2.4): the house-rate correction LADDER, extracted verbatim from
 * output_thread's master block. Returns the correction (ppm; positive = slow the house)
 * the master publishes and every rung applies. Priority order, unchanged:
 *   P-servo → reprime override → +1.5% sustained cap → hard ±6% clamp →
 *   gentle zone ±0.6% → bank top-up floor → clock-follow subtraction.
 * Inputs are the parameters plus the published bank atomics; the servo EMA, reprime
 * state machine and clock-follow latch live in *hr, the published estimator rates in
 * *est (0.9.18 R3/R4). Master thread only — non-reentrant. */
static int64_t house_rate_corr_ppm(HouseRateState *hr, const RateEstimator *est,
                                   int occ, int sp, int base_sp, int64_t tick_dur_us)
{
    /* PROPORTIONAL occupancy servo (v0.9.6). The earlier INTEGRAL servo (corr += K·err)
     * oscillated: the buffer is itself an integrator (ρ → consume-rate → ∫ → occupancy),
     * so an integrating CONTROLLER on top makes a type-2 loop that limit-cycles (the
     * ±7-13k ppm ρ wobble), and a series EMA only added phase lag = worse. A PROPORTIONAL
     * controller of an integrator plant is unconditionally stable — no windup, no limit
     * cycle. ρ = Kp·(setpoint − occ): buffer filling → ρ<0 → house faster → drains it.
     * Steady state parks at err = −mismatch/Kp (sub-frame for any real crystal) and
     * ρ ≈ −(source rate offset) — i.e. ρ now READS the true per-source rate, smooth and
     * non-hunting. Wide ±6% clamp keeps drain authority (the old ±150ppm proportional try
     * SATURATED → couldn't drain → crept to 42f → dlvforced; ±6% never saturates on a real
     * source). A light EMA gives fractional occ so ρ doesn't step on integer-frame
     * quantization; with P-control its lag is harmless. */
    int repriming = 0;
    if (!hr->occ_ema_seeded) { hr->occ_ema_milli = (int64_t)occ * 1000; hr->occ_ema_seeded = 1; }
    hr->occ_ema_milli += ((int64_t)occ * 1000 - hr->occ_ema_milli) / 16;   /* EMA N=16 → smooth fractional occ */
    int64_t err_milli = (int64_t)sp * 1000 - hr->occ_ema_milli;          /* setpoint − occ (milli-frames) */
    int64_t corr = (err_milli * 500) / 1000;                           /* ρ = Kp·err, Kp=500 ppm/frame (PROPORTIONAL, no accumulation) */
    int64_t hi = 60000;                                            /* +6% normal positive clamp */
    if (g_reprime && occ <= (base_sp + 1) / 2) {                   /* RE-PRIME: drained below half the BASE floor (true starvation —
                                                                    * NOT the adaptive raised target). Slow the house HARD to refill fast.
                                                                    * 0.9.10.1 state machine: an engagement lasts AT MOST 10s, then a 300s
                                                                    * cooldown applies unconditionally — occupancy oscillating around the
                                                                    * trigger must NOT re-arm (the 0.9.10 flaw: AWE 23.976-film segments
                                                                    * kept occ at the threshold → reprime pinned the house at 0.77x for
                                                                    * the whole segment → downstream underrun). */
        int64_t nw2 = av_gettime_relative();
        if (hr->reprime_start == 0 &&
            (hr->reprime_last_end == 0 || nw2 - hr->reprime_last_end > 300LL * 1000000))
            hr->reprime_start = nw2;                               /* begin a new engagement (cooldown clear) */
        if (hr->reprime_start && nw2 - hr->reprime_start <= 10LL * 1000000) {
            corr = 300000; hi = 300000;                            /* ≈ house 0.77x, bounded to 10s */
            repriming = 1;
        } else if (hr->reprime_start) {
            hr->reprime_last_end = nw2; hr->reprime_start = 0;     /* cap hit → end + cooldown */
        }
    } else if (hr->reprime_start) {
        hr->reprime_last_end = av_gettime_relative(); hr->reprime_start = 0;   /* occ recovered → end + cooldown */
    }
    if (!repriming && corr > 15000) corr = 15000;                  /* 0.9.10.1: sustained positive (slow-down) authority capped at 1.5%
                                                                    * — the proven pre-0.9.10 level (Kp x base_sp). The servo must never
                                                                    * RATE-MATCH a sustained content-rate deficit (23.976 film in a 29.97
                                                                    * container = 24 AU/s is LEGITIMATE; dups there are 3:2 pulldown).
                                                                    * Real source-clock offsets are ppm-scale; 1.5% covers any crystal. */
    if (corr >  hi)     corr =  hi;
    if (corr < -60000)  corr = -60000;
    /* v0.9.10 gentle zone: above the BASE safety floor the servo only nudges (±0.6%) —
     * an adaptive GROW fills lazily from the source's natural catch-up bursts and a
     * SHRINK drains at ppm scale, so tier transitions never jerk downstream delivery.
     * Full authority below the floor (real starvation) and under re-prime. */
    if (g_adapt_cushion && !repriming && hr->occ_ema_milli >= (int64_t)base_sp * 1000) {
        if (corr >  6000) corr =  6000;
        if (corr < -6000) corr = -6000;
    }
    /* v0.9.14 AUTO-BANK top-up: deficit retention alone converges to BREAK-EVEN only
     * (each cycle starves by gap−coverage and retains exactly that; the 1.5x margin
     * never builds — measured: dup trickle persists at the boundary). Fill the margin
     * actively: above the safety floor, bias the gentle zone to slow-fill (+0.6%,
     * imperceptible) until video_q holds the bank target; normal servo resumes there. */
    {
        int64_t bt = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
        /* margin = TOTAL buffered content: compressed video_q + decoded frame_q
         * (counting video_q alone would overshoot by frame_q's depth, ~5s) */
        int64_t have = ((int64_t)atomic_load_explicit(&g_vq_elems, memory_order_relaxed) + occ)
                       * tick_dur_us;
        if (bt > 0 && !repriming && hr->occ_ema_milli >= (int64_t)base_sp * 1000 &&
            have < bt && corr < 6000)
            corr = 6000;
    }
    /* v0.9.15 CLOCK-FOLLOW: a locked coarse-FLL offset beyond the arm threshold is a
     * real source-clock fault — follow it beyond the gentle zone, else the servo pegs
     * at +-0.6%, buffers pin and aresample churns forever. Hysteresis latch (arm
     * >5000, release <2000); capped +-2%. The tick pacing (and thus output PCR) runs
     * at the source's true rate — receivers slave to PCR, so the chain simply runs at
     * source pace, as with any PCR-locked feed. v0.9.15.5: arm 3000->5000 — NewsNation's
     * clock WANDERS -700..-3400ppm and chattered arm/release across 3000 (~15/day);
     * sub-5000 offsets are handled fine unfollowed (WUCR + decimation, proven live),
     * so follow engages only for the genuinely-broken-clock class it was built for. */
    if (g_clockfollow && atomic_load_explicit(&est->cf_locked, memory_order_relaxed)) {
        int64_t cf_ppm = ((atomic_load_explicit(&est->cf_rate_q20, memory_order_relaxed)
                           - (1 << 20)) * 1000000) >> 20;
        /* v0.9.15.3: a BURSTY-classified channel (auto-bank armed) violates the coarse
         * estimator's smooth-delivery assumption — its clump windows alias into a bogus
         * offset (Unique TV latched cf=+28450ppm -> followed +2% fast -> drained the very
         * bank that absorbs the clumps). Never follow while the bank is armed. */
        if (atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0) {
            if (hr->cf_following) {
                hr->cf_following = 0;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-CLOCK] BURSTY channel (bank armed) — follow released, estimator untrusted\n");
            }
        } else
        if (!hr->cf_following && llabs(cf_ppm) > 5000) {
            hr->cf_following = 1;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-CLOCK] source clock runs %+lldppm off realtime — FOLLOWING it "
                   "(output+PCR pace at the source's true rate; PTV_NO_CLOCKFOLLOW reverts)\n",
                   (long long)cf_ppm);
        } else if (hr->cf_following && llabs(cf_ppm) < 2000) {
            hr->cf_following = 0;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-CLOCK] source clock back within normal range (%+lldppm) — released\n",
                   (long long)cf_ppm);
        }
        if (hr->cf_following)
            corr -= av_clip64(cf_ppm, -20000, 20000);
    }
    return corr;
}

void *output_thread(void *arg)
{
    VideoCtx *v = arg;
    AVFrame *held = av_frame_alloc();
    AVFrame *f;
    int have = 0, ret = 0;
    int64_t tick = 0, wall0 = 0, last_vpts = -1, gl_phase = 0;   /* gl_phase: v0.9.0 genlock-scaled cumulative wall span */
    int64_t last_content_vpts = -1;  /* v0.9.15.3: content index of the last REAL frame emitted. Decimation must
                                      * compare against PLAYED CONTENT, not last_vpts: each dup bumps last_vpts one
                                      * tick past content (monotonic guard), so after a delivery stall last_vpts sits
                                      * N ticks ahead and the refill clump all reads as "surplus" -> decimation eats
                                      * the very latency AUTO-BANK retains -> dup/decim oscillation (Unique TV,
                                      * dup=615K = decim=613K over 10h45m, 6s pause/fast-forward cycle). */
    int64_t held_src_pts = AV_NOPTS_VALUE;   /* ORIGINAL source pts of held frame (held->pts gets
                                                overwritten to vpts on emit; dups must not re-read it) */
    int64_t diag_t0 = av_gettime_relative(), diag_last = diag_t0;
    int64_t stat_last = diag_t0, stat_prev = 0;

    if (!held)
        goto done;

    if (!v->live) {
        /* offline: media clock — encode every decoded frame 1:1, no pacing/dup */
        for (;;) {
            ret = av_thread_message_queue_recv(v->frame_q, &f, 0);
            if (ret < 0) break;
            f->pts = tick++; f->pkt_dts = AV_NOPTS_VALUE; f->duration = 0;
            ret = encode_push(v->mux_q, v->venc, v->ost, f, NULL);   /* offline: no delivery gate */
            v->emitted++; v->last_emit_us = av_gettime_relative();
            av_frame_free(&f);
            if (ret < 0) break;
        }
        encode_push(v->mux_q, v->venc, v->ost, NULL, NULL);
        goto done;
    }

    if (v->passthrough) {
        /* multiview: the compositor IS the house clock — it paced this frame and
         * already stamped pts (in venc tb). Encode 1:1, no re-pace / dup / skew
         * (the compositor owns all of that, and the stats/diag line). */
        for (;;) {
            ret = av_thread_message_queue_recv(v->frame_q, &f, 0);
            if (ret < 0) break;
            f->pkt_dts = AV_NOPTS_VALUE; f->duration = 0;
            ret = encode_push(v->mux_q, v->venc, v->ost, f, v->gate);   /* gate slot audio/copy to this composite video */
            v->emitted++; v->last_emit_us = av_gettime_relative();
            av_frame_free(&f);
            if (ret < 0) break;
        }
        encode_push(v->mux_q, v->venc, v->ost, NULL, v->gate);
        goto done;
    }

    /* live: free-running master clock at the house rate. Pop ONE frame per tick;
     * the frame_q is a jitter buffer that absorbs decoder delivery bursts, so at
     * matched rates this is a smooth 1:1 (CFR). A genuine source gap -> dup; a
     * genuine overflow (source faster / output stalled) -> drop-oldest at decode.
     *
     * Pre-roll: decode delivery is bursty (OS scheduling, network read batching)
     * even when the source cadence is perfectly steady, while the master clock
     * consumes at a matched average rate. With no cushion the buffer sits near
     * empty, so any momentary decode gap starves a tick -> a repeated frame (dup)
     * -> visible micro-stutter. Priming frame_q to ~PTV_PREROLL_MS worth before
     * starting the clock gives the gaps something to draw down instead. The video
     * PTS stays content-anchored to h0, so the cushion only shifts WHEN frames
     * emit, never their timestamps -> A/V sync is unchanged. */
    {
        int preroll_ms = g_cp.preroll_ms;   /* v0.9.1: single-input frame_q cushion tracks the resolved prime (genlock default ~1s); was a separate getenv→350 read */
        int n_prime = (preroll_ms > 0 && v->tick_dur_us > 0)
                          ? (int)((int64_t)preroll_ms * 1000 / v->tick_dur_us) : 0;
        int primed;
        int64_t eq_vq_s = 0, eq_fq_s = 0, eq_fq_h = 0, eq_mq_s = 0;  /* PTV-EMPTY per-queue empty-since (+ frame_q heartbeat) state */
        int64_t corr_wd_last = 0;   /* pre14: stale-track corrector watchdog rate limit (1s) */
        int64_t ep_agg_cnt = 0, ep_agg_min = 0, ep_agg_max = 0, ep_agg_t0 = 0;  /* 0.9.10.1: frame_q sub-2s episode aggregator (60s summary) */
        if (n_prime > g_cp.frameq_cap - 8) n_prime = g_cp.frameq_cap - 8;
        if (n_prime < 0) n_prime = 0;
        primed = (n_prime == 0);
        /* v0.9.10 adaptive cushion (master only): two discrete frame_q targets, lazy transitions. */
        int base_sp   = n_prime > 2 ? n_prime : 4;                    /* safety floor = the resolved preroll */
        int raised_sp = (v->tick_dur_us > 0) ? (int)(g_cp.cushion_raised_us / v->tick_dur_us) : base_sp;
        int64_t ep_last_us = 0, ep_prev_us = 0;                       /* starvation-episode wall times (grow gate) */
        int64_t rr_starve_since = 0, rr_last_rel = 0;                 /* 1.0.1-pre8 (b): ratchet-release detector state */
        int64_t sh_starve_since = 0, sh_last = 0;                     /* 1.0.1-pre8 (c): self-heal detector state */
        int64_t cr_starve_since = 0, cr_ok_since = 0, cr_last_rel = 0;/* 1.0.1-pre10 (e): cushion-release detector state */
        int64_t heal_refire_us = av_rescale(300000000, g_jit_milli, 1000); /* 1.0.1-pre10 (g): jittered SELFHEAL re-fire (4-6min per PID) */
        if (raised_sp > g_cp.frameq_cap - 8) raised_sp = g_cp.frameq_cap - 8;
        if (raised_sp < base_sp) raised_sp = base_sp;                 /* explicit deep preroll >= cushion -> adaptive no-op */
        if (v->is_master) {                                           /* 0.9.18 M3: register the adaptive tier with the
                                                                       * escalation runtime — cushion_escalate() owns
                                                                       * cur_sp from here (mutated only by the master's
                                                                       * own GROW/SHRINK calls, so the unlocked reads
                                                                       * below are same-thread-ordered) */
            g_curt.base_sp = base_sp; g_curt.raised_sp = raised_sp;
            g_curt.cur_sp  = base_sp;
        }
        /* 1.0.1-pre9 residual sensor: video-side EMA state (master rung; τ ≈ 30s of ticks) */
        int64_t rs_mv_ema = 0, rs_mv_div = v->tick_dur_us > 0 ? 30000000 / v->tick_dur_us : 750;
        int     rs_mv_seed = 0;
        if (rs_mv_div < 8) rs_mv_div = 8;
        /* v0.9.11 pulldown state: 1-frame lookahead + film-mode detector (see g_pulldown comment) */
        AVFrame *nextf = NULL;
        int next_have = 0, film_arm = 0, held_extra = 0;
        unsigned rff_bits = 0;
        int64_t cad_ema_us   = v->tick_dur_us;      /* M7: fresh-frame source-spacing EMA (tau ~8f); seeded real-time */
        int64_t cad_prev_src = AV_NOPTS_VALUE;      /* previous fresh frame's SOURCE pts (held_src_pts domain) */
        int     cad_dropouts = 0, cad_in_drop = 0;  /* flag-dropout EVENTS ridden this engagement + in-a-dropout flag */

    for (;;) {
        int fresh = 0, cadence_hold = 0;
        if (g_pulldown && film_arm && have) {       /* film cadence: pop via content-projected lookahead */
            if (!next_have) {
                ret = av_thread_message_queue_recv(v->frame_q, &f, AV_THREAD_MESSAGE_NONBLOCK);
                if (ret >= 0) { nextf = f; next_have = 1; }
                else if (ret == AVERROR_EOF) break; /* pending nextf was already promoted (recv only when empty) */
            }
            if (next_have) {
                int64_t nc = content_index(v, nextf->pts);
                if (nc < 0 || nc <= last_vpts + 1 || held_extra >= 1) {   /* due, unstampable, or residence CAP hit */
                    av_frame_unref(held); av_frame_move_ref(held, nextf); av_frame_free(&nextf);
                    next_have = 0; fresh = 1; held_src_pts = held->pts; held_extra = 0;
                } else
                    cadence_hold = 1;               /* held frame legitimately occupies this tick (3-field residence) */
            }
            /* queue empty + no lookahead: fall through = dup-on-empty exactly as today */
        } else if (next_have) {
            /* v0.9.16.2 (defensive): drain a PARKED pulldown lookahead first. If cadence ever
             * disarms with a frame still in nextf, that frame would sit orphaned until the next
             * arm promoted it STALE (out-of-order emission + a one-tick house-skew spike the
             * audio path samples via AVLOCK). In the normal flow disarm lands on promote ticks
             * (nextf just consumed), so this is a rare-path guard, NOT a lip-sync fix: a 46-flap
             * flash+beep A/B (synthetic soft-telecine flapping, the AWE profile) measured
             * byte-identical A/V alignment with and without it — flap transitions are A/V-neutral.
             * The promoted frame is the NEXT content frame, so no decim check needed (mirrors the
             * film path's own promotion). */
            av_frame_unref(held); av_frame_move_ref(held, nextf); av_frame_free(&nextf);
            next_have = 0; have = 1; fresh = 1; held_src_pts = held->pts; held_extra = 0;
        } else {
            /* v0.9.15.2 CADENCE DECIMATION (single-input mirror of the 0.9.13 mosaic multi-pop):
             * a frame whose content index does NOT advance past the last emitted tick is SURPLUS —
             * a stream delivering more real frames than its declared rate (NewsNation: ~25.3-25.5
             * real fps stamped truly, declared 25/1; cadence WANDERS so no fixed house rate fits).
             * Take the next frame instead and display only the newest due one: the output samples
             * the source's own timeline at the house rate, so lip-sync stays exact and frame_q
             * stays level (before: +0.45f/s surplus pinned it at 160 -> bursty drop-oldest +
             * async churn). Never fires for <=house-rate content (indices always advance) — film
             * pulldown, exact-rate and slow sources are untouched. Bounded 3 pops/tick (+8%).
             * PTV_NO_DECIMATE reverts. */
            int pops = 0, got_eof = 0;
            for (;;) {
                ret = av_thread_message_queue_recv(v->frame_q, &f, AV_THREAD_MESSAGE_NONBLOCK);
                if (ret >= 0) {
                    av_frame_unref(held); av_frame_move_ref(held, f); av_frame_free(&f);
                    have = 1; fresh = 1; held_src_pts = held->pts;   /* capture before emit overwrites it */
                    held_extra = 0;
                    pops++;
                    if (g_decimate && pops < 3) {
                        int64_t hc = content_index(v, held->pts);
                        /* v0.9.15.3: surplus = maps to already-PLAYED content (last_content_vpts), NOT to the
                         * dup-advanced output cursor (last_vpts) — post-stall refill frames are new content and
                         * must play at 1x with the latency retained (the AUTO-BANK posture); only a genuinely
                         * >house-rate cadence decimates. Catch-up fast-forward is gone by construction. */
                        if (hc >= 0 && hc <= last_content_vpts) { v->decim++; continue; }   /* surplus: take a fresher one */
                    }
                    break;
                } else if (ret == AVERROR_EOF) {
                    if (!fresh) got_eof = 1;        /* terminal only if nothing taken this tick */
                    break;
                } else
                    break;                          /* queue empty */
            }
            if (got_eof)
                break;                              /* decode finished, queue drained */
        }
        if (!have) { av_usleep(2000); continue; }   /* await first frame (no startup dups) */

        if (fresh && g_pulldown) {                  /* film-mode detector: progressive frames with rff==1 only
                                                     * (==1 excludes doubling/tripling 2/4 — the bogus pic_struct=7
                                                     * class; interlaced-flagged rff never arms) */
            if (cad_prev_src != AV_NOPTS_VALUE && held_src_pts != AV_NOPTS_VALUE) {
                int64_t dt = av_rescale_q(held_src_pts - cad_prev_src, v->out_tb, AV_TIME_BASE_Q);
                if (dt > 0 && dt < 200000)          /* skip splices/jumps/wraps; keep the EMA honest */
                    cad_ema_us += (dt - cad_ema_us) / 8;
            }
            cad_prev_src = held_src_pts;
            rff_bits = (rff_bits << 1) |
                       (held->repeat_pict == 1 && !(held->flags & AV_FRAME_FLAG_INTERLACED));
            int rn = av_popcount(rff_bits & 0xffu);
            if (!film_arm && rn >= 3) {
                film_arm = 1; cad_dropouts = 0; cad_in_drop = 0;
                if (v->is_master)
                    av_log(NULL, AV_LOG_INFO, "[PTV-PULLDOWN] armed (telecine cadence detected: %d/8 rff frames)\n", rn);
            } else if (film_arm && rn == 0) {
                /* v0.9.18.1 M7: disarm needs CONTENT-RATE evidence, not just absent flags (see
                 * g_cad_disarm). Ride flag dropouts while spacing stays film-paced; a real
                 * film->video transition brings spacing to ~tick within ~10-15 frames (EMA
                 * tau 8) and disarms then — the few extra pd holds at the boundary are benign. */
                if (!g_cad_disarm || cad_ema_us <= v->tick_dur_us + v->tick_dur_us / 8) {
                    film_arm = 0;
                    if (v->is_master)
                        av_log(NULL, AV_LOG_INFO, "[PTV-PULLDOWN] disarmed (cadence ended; %"PRId64" holds, %d flag dropouts ridden)\n",
                               v->pd, cad_dropouts);
                } else if (!cad_in_drop) {
                    cad_in_drop = 1;
                    if (!cad_dropouts++ && v->is_master)
                        av_log(NULL, AV_LOG_INFO, "[PTV-PULLDOWN] rff flags dropped out at film pacing (%.1fms/frame) — staying armed\n",
                               cad_ema_us / 1000.0);
                }
            } else if (film_arm && rn > 0)
                cad_in_drop = 0;                    /* flags returned — dropout event closed */
        }

        if (!primed) {                              /* one-time jitter-buffer pre-roll */
            int64_t pt0 = av_gettime_relative();
            while (av_thread_message_queue_nb_elems(v->frame_q) < n_prime &&
                   av_gettime_relative() - pt0 < (int64_t)preroll_ms * 3000)
                av_usleep(2000);
            primed = 1;
            if (g_diag)
                av_log(NULL, AV_LOG_INFO,
                       "[PTV-DIAG] preroll: primed frame_q to %d frames (~%dms target)\n",
                       av_thread_message_queue_nb_elems(v->frame_q), preroll_ms);
        }

        if (v->emitted == 0) wall0 = av_gettime_relative();
        {
            /* v0.9.0 genlock: pace off a phase accumulator (not tick*tick_dur) so a rate change never
             * teleports the target. per_tick scales by the recovered source rate (ALL single-input rungs,
             * once locked); otherwise == tick_dur_us → gl_phase == tick*tick_dur → byte-identical free-run. */
            if (v->is_master) {  /* DIAG: publish frame_q depth so the demux-thread discontinuity logs can show the drain */
                int64_t nw = av_gettime_relative();
                int64_t ep;
                atomic_store_explicit(&g_frameq_depth, av_thread_message_queue_nb_elems(v->frame_q), memory_order_relaxed);
                if (g_diag) {   /* video_q/mux_q: log only >=2s episodes — empty is their NORMAL state (consumer
                                 * drains instantly; sub-2s "refills" are sampling noise — the Cinestar lesson).
                                 * A >=2s video_q episode = a real input stall, still event-worthy. */
                    if (v->dbg_video_q) ptv_empty_watch("video_q", av_thread_message_queue_nb_elems(v->dbg_video_q), nw, &eq_vq_s, NULL, 2000000);
                    if (v->mux_q) ptv_empty_watch("mux_q", av_thread_message_queue_nb_elems(v->mux_q), nw, &eq_mq_s, NULL, 2000000);
                }
                /* frame_q starvation watch runs UNGATED — it feeds the adaptive cushion. Per-episode
                 * lines only >=2s; sub-2s episodes aggregate into one summary line per minute. */
                ep = ptv_empty_watch("frame_q", av_thread_message_queue_nb_elems(v->frame_q), nw, &eq_fq_s, &eq_fq_h, 2000000);
                if (ep > 0 && ep < 2000000) {
                    if (!ep_agg_cnt) { ep_agg_min = ep_agg_max = ep; ep_agg_t0 = ep_agg_t0 ? ep_agg_t0 : nw; }
                    else { if (ep < ep_agg_min) ep_agg_min = ep; if (ep > ep_agg_max) ep_agg_max = ep; }
                    ep_agg_cnt++;
                    if (!ep_agg_t0) ep_agg_t0 = nw;
                }
                if (ep_agg_cnt && nw - ep_agg_t0 >= 60LL * 1000000) {
                    av_log(NULL, AV_LOG_INFO, "[PTV-EMPTY] frame_q: %lld episodes in %llds (%lld-%lldms)\n",
                           (long long)ep_agg_cnt, (long long)((nw - ep_agg_t0) / 1000000),
                           (long long)(ep_agg_min / 1000), (long long)(ep_agg_max / 1000));
                    ep_agg_cnt = 0; ep_agg_t0 = nw;
                }
                /* pre14 corrector stale-track watchdog: a DWELL/ENGAGED track whose audio
                 * thread stopped emitting (ma_wall stale >5s — source-audio death, the pre12
                 * W3 class) can neither steer NOR log its own disarm (its thread is blocked
                 * in recv). Integration is inherently frozen (no frames), so this is purely
                 * the §6 one-line disarm + published-state sync: CAS the published state to
                 * DISARMED (the CAS guarantees exactly one line even if the track resumes
                 * mid-check) and hand the owning thread a silent disarm_req to consume. */
                if (g_rsync_corr && nw - corr_wd_last >= 1000000) {
                    int kc;
                    corr_wd_last = nw;
                    for (kc = 0; kc < g_rsx.n_a; kc++) {
                        int st = atomic_load_explicit(&g_corr_state_pub[kc], memory_order_relaxed);
                        if (st == PTV_CORR_DWELL || st == PTV_CORR_ENGAGED) {
                            int64_t maw = atomic_load_explicit(&g_rsx.ma_wall[kc], memory_order_relaxed);
                            if (maw && nw - maw > 5000000 &&
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
                if (g_adapt_cushion && !v->passthrough) {
                    /* 0.9.18 M3: trigger conditions stay here; the write bodies (tier store,
                     * cap delta, maxq, log) moved verbatim to cushion_escalate(). */
                    if (ep > 0) {                                     /* a >=200ms starvation episode just ended */
                        ep_prev_us = ep_last_us; ep_last_us = nw;
                        /* 1.0.1-pre10 (e): 10min GROW suppression after a CUSHION_RELEASE —
                         * under a persistent decode deficit the very next episode pair would
                         * re-GROW seconds after the release and the pair would flap once a
                         * minute; while the contradiction persists the tier belongs at base.
                         * cr_last_rel stays 0 when PTV_NO_CUSHREL (condition unchanged). */
                        if (g_curt.cur_sp < raised_sp && ep_prev_us && nw - ep_prev_us < 3600LL * 1000000 &&
                            (!cr_last_rel || nw - cr_last_rel >= 600LL * 1000000))
                            cushion_escalate(CUSHION_GROW, nw - ep_prev_us, ep);
                    } else if (g_curt.cur_sp > base_sp && ep_last_us && nw - ep_last_us > 6LL * 3600 * 1000000) {
                        cushion_escalate(CUSHION_SHRINK, 0, 0);
                    }
                }
                /* 1.0.1-pre8 (b)+(c) — the #32 wedge starvation-contradiction detectors.
                 * "Starved while input flows" is the contradiction state: frame_q pinned ≤2
                 * frames while the demux keeps receiving video (clean wire). Normal deep-bank
                 * operation never looks like this (a delivery gap means input is NOT flowing;
                 * a catch-up refill means frame_q is NOT starved), so banks working as
                 * designed are untouched.
                 *   (b) ≥5s of it with an ARMED bank → release the ratchet (BANK_RELEASE)
                 *       instead of the 6h decay; 60s re-fire limit.
                 *   (c) ≥30s of it regardless of bank → the decode path is wedged on stale/
                 *       undecodable backlog: request the in-process re-prime (the decode
                 *       thread flushes video_q + decoder and resumes at the next IDR — what a
                 *       supervisor restart achieves without the restart); one attempt per 5min. */
                if ((g_ratchrel || g_selfheal || g_cushrel) && v->live && !v->passthrough) {
                    int fqd = av_thread_message_queue_nb_elems(v->frame_q);
                    int64_t arr = atomic_load_explicit(&g_v_arrive_wc, memory_order_relaxed);
                    int flowing = arr && (nw - arr) < 2000000;
                    /* 1.0.1-pre10 (e) CUSHION RELEASE — the (b)/(c) contradiction applied to the
                     * adaptive TIER: it armed at birth (2 episodes/60min — birth under contention
                     * trips it in ~6s) and its only release is 6h of ZERO starvation episodes,
                     * which the churn itself makes unreachable (Phase A: cushion=2535ms + fqhw=160
                     * + grown caps still held 12min post-recovery; in production that pins the
                     * frame pool + NVENC registration set at maximum forever). >=60s of the
                     * contradiction with the tier raised -> step it back to base (CUSHION_RELEASE,
                     * same stores as SHRINK). Unlike (b)/(c) this timer FORGIVES <=5s refill
                     * blips: the ~6s shed cycle refills frame_q for ~1-2s every cycle (starved
                     * fraction 0.88-0.97 measured), so a hard reset would make 60s continuous
                     * unreachable under exactly the symptom this targets. A genuine outage
                     * (input NOT flowing) hard-resets — a real stall keeps its cushion. */
                    if (g_cushrel) {
                        if (fqd <= 2 && flowing) {
                            if (!cr_starve_since) cr_starve_since = nw;
                            cr_ok_since = 0;
                            if (nw - cr_starve_since >= 60LL * 1000000 &&
                                g_curt.cur_sp > base_sp &&
                                (!cr_last_rel || nw - cr_last_rel >= 60LL * 1000000)) {
                                cr_last_rel = nw;
                                cushion_escalate(CUSHION_RELEASE, nw - cr_starve_since, 0);
                                cr_starve_since = 0;        /* a further step needs a fresh 60s */
                            }
                        } else if (!flowing) {
                            cr_starve_since = 0; cr_ok_since = 0;   /* outage: keep the cushion */
                        } else if (cr_starve_since) {       /* flowing + refilled: 5s blip forgiveness */
                            if (!cr_ok_since) cr_ok_since = nw;
                            else if (nw - cr_ok_since > 5000000) { cr_starve_since = 0; cr_ok_since = 0; }
                        }
                    }
                    if (fqd <= 2 && flowing) {
                        if (g_ratchrel) {
                            if (!rr_starve_since) rr_starve_since = nw;
                            else if (nw - rr_starve_since >= 5000000 &&
                                     (rr_last_rel == 0 || nw - rr_last_rel >= 60000000) &&
                                     atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0) {
                                rr_last_rel = nw;
                                cushion_escalate(BANK_RELEASE, nw - rr_starve_since, 0);
                            }
                        }
                        if (g_selfheal) {
                            if (!sh_starve_since) sh_starve_since = nw;
                            else if (nw - sh_starve_since >= 30000000 &&
                                     (sh_last == 0 || nw - sh_last >= heal_refire_us)) {   /* 1.0.1-pre10 (g): per-PID
                                                                                            * jittered 5min re-fire —
                                                                                            * co-located heal/refill
                                                                                            * bursts de-phase */
                                int64_t starved_s = (nw - sh_starve_since) / 1000000;
                                sh_last = nw;
                                sh_starve_since = nw;   /* a re-fire needs a fresh 30s of starvation */
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-SELFHEAL] frame_q starved %llds with input flowing — requesting "
                                       "internal re-prime (flush video_q + decoder, resume at next IDR; "
                                       "PTV_NO_SELFHEAL disables)\n", (long long)starved_s);
                                atomic_store_explicit(&g_selfheal_req, 1, memory_order_relaxed);
                            }
                        }
                    } else {
                        rr_starve_since = 0;
                        sh_starve_since = 0;
                    }
                }
            }
            int64_t per_tick = v->tick_dur_us;
            if (g_wucr) {
                /* WUCR ρ (W0): PROPORTIONAL occupancy controller. corr = Kp·(setpoint − occ_ema).
                 * occ above setpoint = buffer filling = source faster than house consumes → corr<0 →
                 * house faster (rate-match). NO integrator → no windup, no pegging: ρ self-settles at the
                 * source rate with the buffer floating a frame or two off setpoint (steady-state offset =
                 * rate/Kp, e.g. +15ppm → ~+1.9 frames). A slow EMA smooths jitter AND delivery bursts (a
                 * burst moves occ_ema gradually, never spikes ρ). ±150ppm hard clamp = physical crystal
                 * bound → runaway impossible. Zero-steady-offset (PI + anti-windup) is the W1 refinement.
                 * Master computes; all rungs apply hr->rho_corr_ppm identically. */
                if (v->is_master) {
                    /* 0.9.18 R2: the ladder body moved verbatim to house_rate_corr_ppm() above;
                     * g_curt.cur_sp = adaptive tier target (base preroll unless grown; M3 moved it
                     * into the escalation runtime). Master computes; all rungs apply the published
                     * hr->rho_corr_ppm identically. */
                    int occ = av_thread_message_queue_nb_elems(v->frame_q);
                    atomic_store_explicit(&v->hr->rho_corr_ppm,
                        house_rate_corr_ppm(v->hr, v->est, occ, g_curt.cur_sp, base_sp, v->tick_dur_us),
                        memory_order_relaxed);
                }
                int64_t corr = atomic_load_explicit(&v->hr->rho_corr_ppm, memory_order_relaxed);
                per_tick = av_rescale(per_tick, 1000000, 1000000 - corr);
            } else if (g_genlock &&
                atomic_load_explicit(&v->est->src_rate_locked, memory_order_relaxed)) {
                int64_t rate = atomic_load_explicit(&v->est->src_rate_q20, memory_order_relaxed);
                if (rate > 0) per_tick = av_rescale(v->tick_dur_us, 1 << 20, rate);  /* source faster (rate>nominal) → shorter span → consume faster */
            }
            int64_t target = wall0 + gl_phase;
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
            gl_phase += per_tick;
        }
        /* Stamp output PTS from the frame's SOURCE time on the shared house
         * anchor (h0) — the SAME mapping audio uses — so dropped/duped frames
         * never skew the timeline and A/V stays locked. (A pure tick counter
         * drifts by the number of startup/stall-dropped frames -> A/V skew.)
         * Pacing still rides the wall clock via `tick`; PTS rides content. */
        {
            int64_t vpts;
            int64_t src_ts = held_src_pts;   /* ORIGINAL source pts (out_tb); survives dups */
            /* EXACTTICK (v0.9.9) content index, via the shared helper (v0.9.11): exact-rational
             * round-nearest — the integer-us divisor (33367 vs 33366.667 at 30000/1001) compressed
             * the mapping ~10ppm -> the chronic audio-behind drift. */
            int64_t content_vpts = content_index(v, src_ts);
            vpts = (content_vpts >= 0) ? content_vpts : last_vpts + 1;
            if (vpts <= last_vpts) vpts = last_vpts + 1;   /* monotonic CFR; dup/hold -> next slot */
            held->pts = vpts; held->pkt_dts = AV_NOPTS_VALUE; held->duration = 0;
            last_vpts = vpts;
            if (content_vpts >= 0)
                last_content_vpts = content_vpts;   /* v0.9.15.3 decimation cursor: real content played
                                                     * (held_src_pts survives dups -> idempotent on dup/hold) */
            /* Publish how far the house clock now runs AHEAD of source content
             * (vpts - content_vpts, in ticks). Each dup bumps vpts past content via
             * the monotonic guard, so this grows by one tick per dup and persists.
             * The audio path adds it so audio rides the same house clock instead of
             * staying source-locked (which is what drifts ~40ms per dup).
             * v0.9.11: a cadence HOLD is content-legitimate residence, NOT skew — subtract
             * held_extra so hs stays 0 through film (the 0<->33ms sawtooth that caused
             * aresample hard-comps = the audible clicks is gone at the SENSOR, not masked).
             * A genuine starvation dup after a hold still measures +1 tick. */
            if (cadence_hold) held_extra++;
            if (v->is_master && v->house_skew && content_vpts >= 0)
                *v->house_skew = (vpts - content_vpts - held_extra) * v->tick_dur_us;
            if (src_ts != AV_NOPTS_VALUE)   /* [PTV-CHAIN] video source-content being emitted (us); any rung (same content) */
                atomic_store_explicit(&g_ch_vout_src, av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q), memory_order_relaxed);
            /* A/V probe (read-only): record this distinct content's first-display output time so the
             * audio drain can pair against it (single-input master rung only; multiview → compositor). */
            if (v->vring && fresh && content_vpts >= 0)
                vring_put(v->vring, av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q), vpts * v->tick_dur_us);
            /* 1.0.1-pre9 residual sensor (PASSIVE), video side: m_v = out − src per EMITTED
             * frame — dups included (a dup presents old content later: that lateness is REAL
             * and must be measured, not read back from house_skew, a control variable). out on
             * the exact-rational axis (the mux pts axis; the integer tick would re-import the
             * ~10ppm EXACTTICK drift into the sensor). EMA ≈ 30s of ticks. Single-input master
             * rung only; multiview (passthrough) never reaches this block. */
            if (g_rsync_sense && v->is_master && src_ts != AV_NOPTS_VALUE) {
                int64_t out_us = v->out_fps.num > 0
                    ? av_rescale(vpts, 1000000LL * v->out_fps.den, v->out_fps.num)
                    : vpts * v->tick_dur_us;
                int64_t m = out_us - av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q);
                if (!rs_mv_seed) { rs_mv_ema = m; rs_mv_seed = 1; }
                else rs_mv_ema += (m - rs_mv_ema) / rs_mv_div;
                /* pre16: slot 0 of the per-slot arrays — single input IS slot 0 (multiview
                 * never reaches this block: passthrough rungs return above; the compositor
                 * owns per-slot publication there). */
                atomic_store_explicit(&g_rsx.mv_ema[0], rs_mv_ema, memory_order_relaxed);
                atomic_store_explicit(&g_rsx.mv_wall[0], av_gettime_relative(), memory_order_relaxed);
            }
        }
        ret = encode_push(v->mux_q, v->venc, v->ost, held, v->gate);   /* §7.5a: publish video front + release caught-up audio/copy */
        v->last_emit_us = av_gettime_relative();
        tick++; v->emitted++;
        if (!fresh) { if (cadence_hold) v->pd++; else v->dup++; }   /* pd = intentional cadence residence; dup stays the health alarm */
        if (g_slow) av_usleep(g_slow);
        if (ret < 0) break;

        if (g_diag && v->is_master) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                /* 1.0.1-pre13: gpps=measured/declared + gov= engagement on the headline DIAG —
                 * the Newsmax2 wedge (dec ≪ fps with vq pinned) was blind without them; a
                 * `gov=1` with dec far below gpps*1.25 is the governor-misbehaving signature.
                 * govslip= (printed when >0) counts oversleep strikes (actuator overshoot). */
                int64_t gslip = atomic_load_explicit(&g_gov_slip, memory_order_relaxed);
                char gv[40] = "";
                if (gslip > 0)
                    snprintf(gv, sizeof gv, " govslip=%"PRId64, gslip);
                av_log(NULL, AV_LOG_INFO,
                    "[PTV-DIAG] t=%.1fs dec=%"PRId64" vcorrupt=%"PRId64" emitted=%"PRId64
                    " muxed=%"PRId64" dup=%"PRId64" pd=%"PRId64" framedrop=%"PRId64" vq=%d frameq=%d muxq=%d gpps=%d/%d gov=%d%s genlock=%d rate=%+.0fppm wucr_rho=%+.0fppm cf=%+.0fppm/%d\n",
                    (nowd - diag_t0) / 1000000.0, *v->dbg_dec_frames, *v->dbg_vcorrupt, v->emitted,
                    g_muxed, v->dup, v->pd, v->framedrop,
                    av_thread_message_queue_nb_elems(v->dbg_video_q),
                    av_thread_message_queue_nb_elems(v->frame_q),
                    av_thread_message_queue_nb_elems(v->mux_q),
                    atomic_load_explicit(&g_gov_gpps, memory_order_relaxed),
                    atomic_load_explicit(&g_gov_decl, memory_order_relaxed),
                    atomic_load_explicit(&g_gov_on, memory_order_relaxed), gv,
                    atomic_load_explicit(&v->est->src_rate_locked, memory_order_relaxed),
                    (atomic_load_explicit(&v->est->src_rate_q20, memory_order_relaxed) - (1 << 20)) * 1e6 / (1 << 20),
                    (double)(-atomic_load_explicit(&v->hr->rho_corr_ppm, memory_order_relaxed)),
                    (atomic_load_explicit(&v->est->cf_rate_q20, memory_order_relaxed) - (1 << 20)) * 1e6 / (1 << 20),
                    atomic_load_explicit(&v->est->cf_locked, memory_order_relaxed));
                diag_last = nowd;
            }
        }

        if (g_stats && v->is_master) {          /* ffmpeg-style progress line */
            int64_t nows = av_gettime_relative();
            if (nows - stat_last >= g_stats_period_us) {
                double dt    = (nows - stat_last) / 1000000.0;
                double fps   = (v->emitted - stat_prev) / (dt > 0 ? dt : 1);   /* INSTANTANEOUS emit rate — the
                                                                                * "alive right now" signal (cumulative
                                                                                * speed= froze at 1.00x after hours and
                                                                                * hid current wedges; removed v0.9.10) */
                double secs  = v->emitted * v->tick_dur_us / 1000000.0;   /* CFR output time */
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                int64_t cr = (v->dbg_pcorrupt ? *v->dbg_pcorrupt : 0) + (v->dbg_vcorrupt ? *v->dbg_vcorrupt : 0);  /* corrupt: demux + decode */
                char dlv[112] = "";                                  /* §7.5a delivery gate: max hold + cap-forced releases */
                if (v->gate) {
                    int dn = snprintf(dlv, sizeof dlv, " dlvhold=%"PRId64"ms dlvforced=%"PRId64,
                             atomic_load_explicit(&v->gate->st_hold_us, memory_order_relaxed) / 1000,
                             atomic_load_explicit(&v->gate->st_forced, memory_order_relaxed));
                    if (v->gate->v_on && dn > 0 && dn < (int)sizeof dlv) {
                        /* §7.5b symmetric gate: EARLY-VIDEO hold (≈ the audio path's wall
                         * latency, e.g. loudnorm ~3s fill); vdlvforced only when the backstop
                         * released (audio flowing but behind, or FIFO overflow) */
                        int64_t vf = atomic_load_explicit(&v->gate->st_vforced, memory_order_relaxed);
                        dn += snprintf(dlv + dn, sizeof dlv - dn, " vdlvhold=%"PRId64"ms",
                                 atomic_load_explicit(&v->gate->st_vhold_us, memory_order_relaxed) / 1000);
                        if (vf > 0 && dn > 0 && dn < (int)sizeof dlv)
                            snprintf(dlv + dn, sizeof dlv - dn, " vdlvforced=%"PRId64, vf);
                    }
                }
                int64_t aw = atomic_load_explicit(&g_async_ppm, memory_order_relaxed);  /* aresample work (ppm) */
                /* v0.9.10 cleanup: genlock=/srcppm= dropped — WUCR (default) never paces via the FLL and
                 * wucr_rho IS the source-ppm readout. size=/bitrate= dropped (CBR is configured; cumulative
                 * averages carry no signal). speed= (cumulative) replaced by instantaneous fps=. qdrop=
                 * dropped (PTV_DIAG demux line still carries it). */
                char wu[144] = "";                                   /* WUCR readout: buffer depth + recovered ρ (go/no-go vs srcppm) */
                if (g_wucr) {
                    int occ = av_thread_message_queue_nb_elems(v->frame_q);
                    int64_t corr = atomic_load_explicit(&v->hr->rho_corr_ppm, memory_order_relaxed);
                    int64_t hs   = v->house_skew ? *v->house_skew : 0;   /* W1 check: must stay ≈0 (ρ genlock → dups→0 → AVLOCK has nothing to inject) */
                    int64_t hsr  = v->dbg_disc_resid ? *v->dbg_disc_resid : 0;   /* 0.9.18.7: LAYERA erase-residue ledger (reporting only) */
                    snprintf(wu, sizeof wu, " wucr_buf=%df/%lldms wucr_rho=%+lldppm hs=%+lldms hsres=%+lldms cushion=%dms fqhw=%d",
                             occ, (long long)(occ * v->tick_dur_us / 1000), (long long)(-corr), (long long)(hs / 1000),
                             (long long)(hsr / 1000),
                             (int)((int64_t)g_curt.cur_sp * v->tick_dur_us / 1000),  /* -corr = recovered source dev (+=faster); cushion = adaptive tier target */
                             atomic_load_explicit(&g_fq_hw, memory_order_relaxed));
                }
                char bk[48] = "";                                    /* v0.9.14 AUTO-BANK: actual/target — actual = TOTAL
                                                                      * buffered margin (compressed video_q + decoded frame_q) */
                {
                    int64_t bt = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
                    if (bt > 0)
                        snprintf(bk, sizeof bk, " bank=%lld/%lldms",
                                 (long long)(((int64_t)atomic_load_explicit(&g_vq_elems, memory_order_relaxed)
                                              + av_thread_message_queue_nb_elems(v->frame_q)) * v->tick_dur_us / 1000),
                                 (long long)(bt / 1000));
                }
                char cfs[56] = "";                                   /* v0.9.15.1 CLOCK-FOLLOW readout (shown when notable) */
                {
                    int64_t cq = atomic_load_explicit(&v->est->cf_rate_q20, memory_order_relaxed);
                    int64_t cp = ((cq - (1 << 20)) * 1000000) >> 20;
                    int     lk = atomic_load_explicit(&v->est->cf_locked, memory_order_relaxed);
                    int     n = 0;
                    if (lk || llabs(cp) > 500)
                        n = snprintf(cfs, sizeof cfs, " cf=%+lldppm%s", (long long)cp, lk ? "" : "?");
                    if (v->decim > 0)                                /* v0.9.15.2: surplus-cadence decimation count */
                        snprintf(cfs + n, sizeof cfs - n, " decim=%"PRId64, v->decim);
                }
                char rsl[24 + PTV_MAX_AUDIO * 16];                   /* pre9 residual sensor: lipsync= (+ = audio early);
                                                                      * pre16: shared per-slot builder (ptvencoder_legend.c) */
                ptv_stats_lipsync(rsl, sizeof rsl, nows, 0);
                char aco[24] = "";                                   /* pre15 #33: corrupt-discarded AUDIO pkts (NBS phase
                                                                      * visibility; absent while zero — clean line unchanged) */
                {
                    int64_t ac = atomic_load_explicit(&g_acorrupt, memory_order_relaxed);
                    if (ac > 0)
                        snprintf(aco, sizeof aco, " acor=%lld", (long long)ac);
                }
                char crs[10 + PTV_MAX_AUDIO * 20];                   /* pre14 corrector: corr= (cumulative trim; * = integrating);
                                                                      * pre16: shared builder — the mv printer gains it by
                                                                      * calling this when the mv corrector hold is lifted */
                ptv_stats_corr(crs, sizeof crs);
                av_log(NULL, AV_LOG_INFO,
                    "frame=%6"PRId64" fps=%4.1f time=%02d:%02d:%05.2f "
                    "dup=%"PRId64" pd=%"PRId64" drop=%"PRId64" corrupt=%"PRId64" "
                    "async=%+"PRId64"ppm%s%s%s%s%s%s%s\n",
                    v->emitted, fps, hh, mm, ss,
                    v->dup, v->pd, v->framedrop, cr, aw, dlv, wu, bk, cfs, aco, rsl, crs);
                stat_last = nows; stat_prev = v->emitted;
            }
        }
    }
    encode_push(v->mux_q, v->venc, v->ost, NULL, v->gate);
    av_frame_free(&nextf);   /* v0.9.11: pending pulldown lookahead (normally promoted before EOF) */
    }
done:
    av_frame_free(&held);
    /* release everything still held + close the gate (no held audio/copy lost at shutdown, and
     * any blocked enqueuer wakes to send direct) — BEFORE the video EOF marker so the muxer sees
     * the tail audio/copy first. No-op when there is no gate (offline). */
    if (v->gate) dlv_flush_all(v->gate);
    v->output_done = 1;
    { AVPacket *eof = NULL; av_thread_message_queue_send(v->mux_q, &eof, 0); }
    return NULL;
}

