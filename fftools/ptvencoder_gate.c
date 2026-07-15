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

#define PTV_WD_DEADLINE_US (2 * (int64_t)AV_TIME_BASE)   /* watchdog stall threshold */
/* demux→decode video queue depth. Raised 48→256 (Phase B #1): at startup the decoder's init window
 * (finding a keyframe, building up) produces nothing while the realtime source keeps filling video_q,
 * so the old 48-deep queue overflowed and dropped ~30 frames → a content GAP → the position-anchored
 * composite video LEAPS to the newest content → audio left behind = per-slot picture-ahead. A deeper
 * queue ABSORBS the one-time init backlog (the decoder then drains it faster than realtime and catches
 * up — multicast/live is realtime steady-state, so video_q sits near-empty after). drop-newest remains
 * the backstop for genuine SUSTAINED overload. PTV_VIDEOQ overrides. */
static int     g_videoq = 512;       /* v0.9.14: default 256->512 — headroom for the runtime AUTO-BANK
                                      * (compressed packets, ~KBs each; 512 covers a 12s bank at 30fps
                                      * with margin, and removes the manual PTV_VIDEOQ for bursty channels) */
static int     g_preroll_set;        /* PTV_PREROLL_MS set explicitly → suppresses the v0.9.0 genlock ~1-GOP default */
static int     g_aq_cap = 256;       /* §13: effective pre-h0 audio ring depth (<= PTV_AQ_PREROLL). Default 256 = historical (byte-identical); raised to PTV_AQ_PREROLL only for a deep prime so audio buffers through the long video-decode delay. */
static _Atomic int64_t g_delivery_cap_us = 3000000;   /* PTV_DELIVERY_CAP_MS: force-release ceiling (≥ max encoder latency).
                                                * v0.9.16.2: _Atomic — runtime writer is the master output thread
                                                * (cushion GROW/SHRINK), readers are demux (bank stores) + init.
                                                * 3s (v0.7.3): the real steady-state hold under production load is ~2s
                                                * (box: TruBLU on cor-1 dlvhold=2055ms — A0's 845ms underestimated it),
                                                * so the old 2s default cap-saturated (dlvforced climbing); 3s lets the
                                                * precise DTS-match win (dlvforced→0) with margin, harmless on low-hold
                                                * channels (match wins well before 3s), still bounds a stuck encoder to 3s. */
static int     g_delivery_maxq = 1024;        /* PTV_DELIVERY_MAXQ: hold-FIFO size backstop (total-stall back-pressure point).
                                               * v0.9.14.1: default 512->1024 and auto-sized for the AUTO-BANK ceiling like
                                               * the deep-preroll path sizes itself (Unique_TV: a 10s bank holds ~470 audio
                                               * pkts steady + a 7s clump surge ~800 peak > the old 512 -> back-pressure ->
                                               * demux audio drops -> resampler comps = clicks WITH a perfect video bank).
                                               * Nodes are per-enqueue, not preallocated - a generous backstop is free. */
static int             g_cushion_ms = 4000;          /* RAISED tier (ms of frames); PTV_CUSHION_MS overrides, [1000,10000] */
CushionPlan     g_cp;                          /* resolved once in transcode() setup, before threads start */
void dlv_init(DlvGate *g, AVThreadMessageQueue *mux_q, int64_t cap_us, int maxq)
{
    pthread_mutex_init(&g->lock, NULL);
    pthread_cond_init(&g->space, NULL);
    g->head = g->tail = NULL;
    g->count = 0; g->closed = 0; g->inited = 1;
    g->mux_q = mux_q;
    atomic_store_explicit(&g->cap_us, cap_us > 0 ? cap_us : 2000000, memory_order_relaxed);
    g->maxq = maxq > 0 ? maxq : 512;
    atomic_store(&g->v_enc_dts_hi, INT64_MIN);
    atomic_store(&g->v_hi_change_wc, av_gettime_relative());   /* "advanced just now" so startup isn't seen as a stall */
    atomic_store(&g->st_hold_us, 0);
    atomic_store(&g->st_forced, 0);
    atomic_store(&g->st_dropped, 0);
}

/* the video encode_push calls this after handing each video packet downstream */
void dlv_publish_video(DlvGate *g, int64_t dts_us)
{
    int64_t cur = atomic_load_explicit(&g->v_enc_dts_hi, memory_order_relaxed);
    int advanced = 0;
    while (dts_us > cur) {
        if (atomic_compare_exchange_weak_explicit(&g->v_enc_dts_hi, &cur, dts_us,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            advanced = 1;
            break;
        }
        /* cur reloaded on failure */
    }
    if (advanced)   /* high-water mark moved up → video is alive and progressing (resets the stall timer) */
        atomic_store_explicit(&g->v_hi_change_wc, av_gettime_relative(), memory_order_relaxed);
}

/* enqueue a dense audio/copy packet; takes ownership of *pkt. block=1 → back-pressure (the audio
 * thread blocks when the FIFO is full, so audio stalls WITH a stalled video and stays locked);
 * block=0 → drop-on-full (the shared demux/copy thread must never stall the whole input). */
void dlv_enqueue(DlvGate *g, AVPacket *pkt, int64_t dts_us, int block)
{
    DlvNode *n;
    pthread_mutex_lock(&g->lock);
    while (!g->closed && g->count >= g->maxq) {
        if (!block) {                       /* full + non-blocking → drop this copy packet */
            atomic_fetch_add_explicit(&g->st_dropped, 1, memory_order_relaxed);
            pthread_mutex_unlock(&g->lock);
            av_packet_free(&pkt);
            return;
        }
        pthread_cond_wait(&g->space, &g->lock);   /* total video stall → block until drain or close */
    }
    if (g->closed) {                        /* video gone → send direct (no drainer left) */
        pthread_mutex_unlock(&g->lock);
        if (av_thread_message_queue_send(g->mux_q, &pkt, 0) < 0)
            av_packet_free(&pkt);
        return;
    }
    n = av_mallocz(sizeof(*n));
    if (!n) { pthread_mutex_unlock(&g->lock); av_packet_free(&pkt); return; }  /* OOM → drop */
    n->pkt = pkt; n->dts_us = dts_us; n->enq_us = av_gettime_relative(); n->next = NULL;
    if (g->tail) g->tail->next = n; else g->head = n;
    g->tail = n; g->count++;
    pthread_mutex_unlock(&g->lock);
}

/* release every held packet the video has caught up to (dts ≤ v_enc_dts_hi) OR — only when video has
 * STALLED (v_enc_dts_hi not advanced within cap_us) — every aged packet (forced: video dead/blocked,
 * degrade to "audio ahead", keep flowing). A long but healthy steady hold (video still advancing) is
 * NOT forced. Called by the rung's video output thread on each emit. Collect under lock, send after
 * (never hold the lock across a blocking mux_q send). */
void dlv_drain(DlvGate *g)
{
    int64_t hi  = atomic_load_explicit(&g->v_enc_dts_hi, memory_order_relaxed);
    int64_t now = av_gettime_relative();
    int64_t adv = atomic_load_explicit(&g->v_hi_change_wc, memory_order_relaxed);
    /* Force-release ONLY when video is genuinely stuck — its emitted DTS hasn't advanced within cap_us.
     * A healthy pipeline carries a long-but-steady hold (preroll + audio-ahead + encoder latency); video
     * keeps advancing, so packets drain naturally as it reaches them and we must NOT force them out early
     * (the old absolute packet-age cap did, on every jitter excursion past the steady hold → dlvforced
     * storms that also fed "audio-way-ahead" bursts into the downstream PLL). */
    int64_t cap_now = atomic_load_explicit(&g->cap_us, memory_order_relaxed);
    int video_stalled = (now - adv) > cap_now;
    DlvNode *out_head = NULL, *out_tail = NULL;   /* released, FIFO order */
    DlvNode *p, *prev, *nx;
    int64_t oldest = 0;
    int freed = 0, forced = 0;

    pthread_mutex_lock(&g->lock);
    prev = NULL; p = g->head;
    while (p) {
        int reached = (hi != INT64_MIN && p->dts_us <= hi);
        int cap     = video_stalled && (now - p->enq_us) > cap_now;
        nx = p->next;
        if (reached || cap) {
            if (prev) prev->next = nx; else g->head = nx;
            if (g->tail == p) g->tail = prev;
            g->count--; freed++;
            if (cap && !reached) forced++;
            p->next = NULL;
            if (out_tail) out_tail->next = p; else out_head = p;
            out_tail = p;
            p = nx;
        } else {
            int64_t age = now - p->enq_us;
            if (age > oldest) oldest = age;
            prev = p; p = nx;
        }
    }
    atomic_store_explicit(&g->st_hold_us, oldest, memory_order_relaxed);
    if (forced) atomic_fetch_add_explicit(&g->st_forced, forced, memory_order_relaxed);
    if (freed)  pthread_cond_broadcast(&g->space);    /* wake blocked enqueuers */
    pthread_mutex_unlock(&g->lock);

    for (p = out_head; p; ) {
        DlvNode *q = p->next;
        if (av_thread_message_queue_send(g->mux_q, &p->pkt, 0) < 0)
            av_packet_free(&p->pkt);
        av_free(p);
        p = q;
    }
}

/* video thread is done: release EVERYTHING still held (shutdown — don't drop the tail), then mark
 * closed + wake any blocked enqueuer so it falls through to a direct send (no hang). */
void dlv_flush_all(DlvGate *g)
{
    DlvNode *out_head, *p;
    pthread_mutex_lock(&g->lock);
    out_head = g->head;
    g->head = g->tail = NULL; g->count = 0;
    g->closed = 1;
    pthread_cond_broadcast(&g->space);
    pthread_mutex_unlock(&g->lock);
    for (p = out_head; p; ) {
        DlvNode *q = p->next;
        if (av_thread_message_queue_send(g->mux_q, &p->pkt, 0) < 0)
            av_packet_free(&p->pkt);
        av_free(p);
        p = q;
    }
}

void dlv_destroy(DlvGate *g)
{
    DlvNode *p;
    if (!g->inited) return;
    for (p = g->head; p; ) { DlvNode *q = p->next; av_packet_free(&p->pkt); av_free(p); p = q; }
    g->head = g->tail = NULL; g->count = 0;
    pthread_cond_destroy(&g->space);
    pthread_mutex_destroy(&g->lock);
    g->inited = 0;
}

CushionRt g_curt = { .lock = PTHREAD_MUTEX_INITIALIZER };

/* 0.9.18 M3 — cushion_escalate(): the single entry point for all four cushion-escalation
 * writers (pure code motion: same stores in the same order, same log lines, now under one
 * mutex). Callers keep their TRIGGER logic in place — the master output thread fires
 * CUSHION_GROW/SHRINK (and, 1.0.1-pre8, BANK_RELEASE on the starvation contradiction),
 * the demux thread fires BANK_ESCALATE/RETIRE; two threads, hence the lock. Both call sites hold no other lock, and this body takes none besides
 * rt->lock (relaxed atomics + gate cap_us atomic stores + av_log only). Per-event args:
 *   CUSHION_GROW:   a0 = wall gap since the previous starvation episode (us, log only),
 *                   a1 = the just-ended episode duration (us, log only)
 *   CUSHION_SHRINK: args unused
 *   BANK_ESCALATE:  a0 = worst observed stall (us), a1 = now (wall us, advisory limiter)
 *   BANK_RETIRE:    args unused
 *   BANK_RELEASE:   a0 = starvation-contradiction duration (us, log only)
 * NOTE (map §3.5, deliberate): GROW/SHRINK do NOT touch the live gate->cap_us — exactly
 * as before this motion (only the BANK arms rewrite the gates). Closing that gap is M5,
 * a behavior change with its own fixture; when it lands it is the same rt->gate loop the
 * BANK arms already run. */
void cushion_escalate(CushionEvent ev, int64_t a0, int64_t a1)
{
    CushionRt *rt = &g_curt;
    pthread_mutex_lock(&rt->lock);
    switch (ev) {
    case CUSHION_GROW: {
        rt->cur_sp = rt->raised_sp;                        /* GROW: 2nd episode within 60min; fill is lazy (gentle zone) */
        int64_t add = (int64_t)(rt->raised_sp - rt->base_sp) * rt->tick_dur_us;  /* audio gate rides the deeper video hold */
        int64_t nc  = atomic_fetch_add_explicit(&g_delivery_cap_us, add, memory_order_relaxed) + add;
        g_delivery_maxq = FFMAX(g_delivery_maxq, (int)(nc / 1000000 * 256));
        /* v0.9.18.3 M5 (plan §3.5): the raised cap must reach the LIVE gates — before this,
         * it reached a gate only if a later BANK event happened to rewrite cap_us, so a
         * RAISED-no-bank channel force-released held audio at the LEAN cap on a real video
         * wedge (~3s early). Same write the BANK arms do: live base + armed bank margin. */
        {
            int64_t bw = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
            int r3;
            for (r3 = 0; r3 < rt->n_gate; r3++)
                if (rt->gate[r3])
                    atomic_store_explicit(&rt->gate[r3]->cap_us, nc + bw, memory_order_relaxed);
            if (g_diag)
                av_log(NULL, AV_LOG_INFO, "[PTV-GATE] caps -> %.1fs (base %.1fs + bank %.1fs) on %d gates (GROW)\n",
                       (nc + bw) / 1e6, nc / 1e6, bw / 1e6, rt->n_gate);
        }
        av_log(NULL, AV_LOG_INFO,
               "[PTV-CUSHION] target %d->%d frames (~%dms): 2 starvations within %lldmin (last %lldms)\n",
               rt->base_sp, rt->raised_sp, (int)((int64_t)rt->raised_sp * rt->tick_dur_us / 1000),
               (long long)(a0 / 60000000), (long long)(a1 / 1000));
        break;
    }
    case CUSHION_SHRINK: {
        rt->cur_sp = rt->base_sp;                          /* SHRINK: 6h with zero starvations; drains at ppm scale */
        /* v0.9.16.2: symmetric restore of the gate base — GROW added exactly this much;
         * without it, daily grow/shrink cycles RATCHET the stall force-release ceiling
         * ~+(raised−base) ticks per cycle FOREVER (months → minutes of held audio on a
         * real video wedge). maxq stays as a high-water backstop (RAM materializes only
         * while actually holding, and the restored cap bounds that duration). */
        int64_t nc2 = atomic_fetch_sub_explicit(&g_delivery_cap_us,
            (int64_t)(rt->raised_sp - rt->base_sp) * rt->tick_dur_us, memory_order_relaxed)
            - (int64_t)(rt->raised_sp - rt->base_sp) * rt->tick_dur_us;
        {   /* v0.9.18.3 M5: symmetric — the restored (lean) cap reaches the live gates too */
            int64_t bw2 = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
            int r4;
            for (r4 = 0; r4 < rt->n_gate; r4++)
                if (rt->gate[r4])
                    atomic_store_explicit(&rt->gate[r4]->cap_us, nc2 + bw2, memory_order_relaxed);
            if (g_diag)
                av_log(NULL, AV_LOG_INFO, "[PTV-GATE] caps -> %.1fs (base %.1fs + bank %.1fs) on %d gates (SHRINK)\n",
                       (nc2 + bw2) / 1e6, nc2 / 1e6, bw2 / 1e6, rt->n_gate);
        }
        av_log(NULL, AV_LOG_INFO, "[PTV-CUSHION] target back to %d frames (quiet 6h)\n", rt->base_sp);
        break;
    }
    case BANK_ESCALATE: {
        int64_t worst_us = a0, now = a1;
        int64_t want_us = worst_us * 3 / 2;
        int64_t ceil_us = g_cp.bank_ceil_us;
        int64_t cur     = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
        int     pkts, r;
        if (want_us > ceil_us) {
            want_us = ceil_us;
            /* at the ceiling AND still short: the one case left for a human — surface it, rate-limited */
            if (worst_us * 12 / 10 > ceil_us && now - rt->bank_advise_us > 60000000) {
                rt->bank_advise_us = now;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-CUSHION] BANK at its %llds ceiling but the worst stall is %.1fs — stalls this size are "
                       "an upstream incident; raising PTV_CUSHION_MAX_MS is a conscious latency trade.\n",
                       (long long)(g_cushion_max_ms / 1000), worst_us / 1e6);
            }
        }
        if (want_us <= cur + 500000)                    /* hysteresis: gap jitter must not re-log identical escalations */
            break;
        pkts = (int)(want_us / 1000000 * g_cp.vid_pps) + 64;  /* v0.9.18.4 M6: video pkt/s from out_fps (was 35/s
                                                               * ~= 29.97+margin — undersized the bank ~30-42%
                                                               * on 50/59.94fps channels, so video_q clipped the
                                                               * real bank below its us target) + margin */
        if (pkts > g_cp.videoq_pkts - 32) pkts = g_cp.videoq_pkts - 32;
        atomic_store_explicit(&g_bank_us, want_us, memory_order_relaxed);
        atomic_store_explicit(&g_bank_pkts, pkts, memory_order_relaxed);
        for (r = 0; r < rt->n_gate; r++)                /* audio gate rides the deeper hold (waits out long stalls) */
            if (rt->gate[r])
                atomic_store_explicit(&rt->gate[r]->cap_us, g_delivery_cap_us + want_us, memory_order_relaxed);
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-CUSHION] BANK escalated to %.1fs (worst stall %.1fs x1.5, ceiling %llds): stall latency is now "
               "RETAINED as a compressed video_q bank — self-heals within a stall cycle, no restart needed\n",
               want_us / 1e6, worst_us / 1e6, (long long)(g_cushion_max_ms / 1000));
        break;
    }
    case BANK_RETIRE: {
        int r2;
        atomic_store_explicit(&g_bank_us, 0, memory_order_relaxed);
        atomic_store_explicit(&g_bank_pkts, 0, memory_order_relaxed);
        for (r2 = 0; r2 < rt->n_gate; r2++)
            if (rt->gate[r2])
                atomic_store_explicit(&rt->gate[r2]->cap_us, g_delivery_cap_us, memory_order_relaxed);
        av_log(NULL, AV_LOG_INFO,
               "[PTV-CUSHION] BANK retired after %llds without qualifying stalls; banked latency drains via catch-up\n",
               (long long)(g_bank_decay_us / 1000000));
        break;
    }
    case BANK_RELEASE: {
        /* 1.0.1-pre8 (b): starvation contradiction — the master rung measured frame_q starved
         * (a0 = duration, us) with input FLOWING while the bank held latency and the gates held
         * audio for it. Holding latency for a buffer that is empty is a contradiction (the #32
         * wedge/aftermath shape: dlvhold ratcheted 12-25s, wucr_buf ~0, clean wire) — release
         * NOW instead of the 6h decay: same stores as BANK_RETIRE, honest log. The master
         * rung's blocking push disarms with g_bank_pkts, so the retained latency drains via
         * the normal catch-up path. Normal deep-bank operation (buffers full / input absent)
         * never reaches here. */
        int64_t held = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
        int r5;
        atomic_store_explicit(&g_bank_us, 0, memory_order_relaxed);
        atomic_store_explicit(&g_bank_pkts, 0, memory_order_relaxed);
        for (r5 = 0; r5 < rt->n_gate; r5++)
            if (rt->gate[r5])
                atomic_store_explicit(&rt->gate[r5]->cap_us, g_delivery_cap_us, memory_order_relaxed);
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-CUSHION] BANK released (was %.1fs): frame_q starved %.1fs with input flowing while "
               "latency was held — contradiction; gate caps back to %.1fs, retained latency drains via "
               "catch-up (PTV_NO_RATCHREL disables)\n",
               held / 1e6, a0 / 1e6, g_delivery_cap_us / 1e6);
        break;
    }
    }
    pthread_mutex_unlock(&rt->lock);
}

/* hand a frame to one rung's jitter buffer; drop-oldest in live so a stalled
 * encoder never blocks the shared decode (same behaviour as before, per rung). */
void push_frame_q(AVThreadMessageQueue *q, int live, int64_t *framedrop, AVFrame *out)
{
    if (!live) {                                  /* offline: lossless back-pressure */
        if (av_thread_message_queue_send(q, &out, 0) < 0)
            av_frame_free(&out);
    } else {
        int ret = av_thread_message_queue_send(q, &out, AV_THREAD_MESSAGE_NONBLOCK);
        if (ret == AVERROR(EAGAIN)) {                /* full -> drop oldest, keep newest */
            AVFrame *old;
            if (av_thread_message_queue_recv(q, &old, AV_THREAD_MESSAGE_NONBLOCK) >= 0) {
                av_frame_free(&old);
                (*framedrop)++;                      /* v0.9.10: count drop-oldest — this is real skipped content
                                                      * (catch-up overflow = the latency-drain meter); it was
                                                      * silent before, leaving stats drop= misleadingly at 0 */
            }
            if (av_thread_message_queue_send(q, &out, AV_THREAD_MESSAGE_NONBLOCK) < 0) {
                av_frame_free(&out);
                (*framedrop)++;
            }
        } else if (ret < 0) {
            av_frame_free(&out);
        }
    }
    {   /* 0.9.18 #19: track the deepest any queue has ever been (fqhw=). Runs for the
         * blocking path too — deep-prime/bank-armed masters are exactly the queues
         * that pin deepest (the original early return blinded fqhw there). */
        int n  = av_thread_message_queue_nb_elems(q);
        int hw = atomic_load_explicit(&g_fq_hw, memory_order_relaxed);
        while (n > hw && !atomic_compare_exchange_weak(&g_fq_hw, &hw, n));
    }
}

/* watchdog: flag (does not yet auto-recover) a stalled output/encoder so a hung
 * NVENC session is visible. A hung in-process session can't be safely torn down
 * from another thread, so auto-reinit needs process isolation — a follow-up. */
void *watchdog_thread(void *arg)
{
    VideoCtx *v = arg;
    while (!v->output_done) {
        av_usleep(500000);
        int64_t le = v->last_emit_us;
        if (v->emitted > 0 && le > 0) {
            int64_t age = av_gettime_relative() - le;
            if (age > PTV_WD_DEADLINE_US) {
                if (!v->stalled) {
                    av_log(NULL, AV_LOG_WARNING,
                        "[PTV-WATCHDOG] output stalled %.1fs — encoder not advancing (input keeps draining)\n",
                        age / 1000000.0);
                    v->stalled = 1;
                }
            } else {
                v->stalled = 0;
            }
        }
    }
    return NULL;
}

/* 0.9.18 M1 — resolve ALL cushion/queue sizing in ONE place: the env parses, the genlock
 * preroll default, the deep-prime side-cars (env-override-wins ordering preserved: side-cars
 * BEFORE the explicit PTV_DELIVERY_CAP_MS/PTV_DELIVERY_MAXQ reads), the per-track audio_q
 * depth rule and the deep-prime derivation — all moved VERBATIM from main()'s env block and
 * transcode()'s alloc sites. Writes the same g_* globals as before (they stay the source for
 * everything not yet repointed) and mirrors the results into *cp. Called once in transcode()
 * setup, after out_fps/n_audio/live are known and before the first consuming allocation —
 * i.e. before any thread starts, so cp is immutable once running.
 * NOTE: PTV_FRAMEQ (+ its NVENC registration-cache warning) stays in main(): g_frameq_cap is
 * consumed by the multiview hold.q alloc BEFORE this runs; mirrored into cp->frameq_cap here. */
void resolve_cushions(CushionPlan *cp, int live, int multiview,
                             AVRational out_fps, int n_audio)
{
    /* ONCE-ONLY: the deep-prime side-cars below ACCUMULATE onto g_delivery_cap_us/g_videoq
     * (+=/FFMAX) — a second call in one process would compound the sizing. transcode() runs
     * once per process today; keep it that way or make these idempotent first. */
    { const char *cm = getenv("PTV_CUSHION_MS"); if (cm && atoi(cm) > 0) { int x = atoi(cm); if (x < 1000) x = 1000; if (x > 10000) x = 10000; g_cushion_ms = x; } }   /* adaptive RAISED tier */
    { const char *pe = getenv("PTV_PREROLL_MS"); if (pe) { int v = atoi(pe); if (v < 0) v = 0; if (v > 30000) v = 30000; g_preroll_ms = v; g_preroll_set = 1; } }  /* §13: startup cushion target (ms), bounded 0-30s */
    { const char *vq = getenv("PTV_VIDEOQ"); if (vq && atoi(vq) > 0) g_videoq = atoi(vq); }   /* video_q depth (startup-burst absorb) */
    { const char *cm = getenv("PTV_CUSHION_MAX_MS"); if (cm && atoi(cm) > 0) g_cushion_max_ms = atoi(cm); }  /* v0.9.14: AUTO-BANK ceiling (default 12000) */
    { const char *bd = getenv("PTV_BANK_DECAY_S"); if (bd && atoi(bd) > 0) g_bank_decay_us = (int64_t)atoi(bd) * 1000000; }  /* v0.9.14: quiet time before bank retires (test hook; default 6h) */
    if (g_genlock && !g_preroll_set) g_preroll_ms = 1000;  /* v0.9.1: default the single-input prime to ~1s (frame_q cushion) — smooths decode-rate dips while video+gate-hold stays under the 3s gate cap (cap scaling stays dormant). Deep video_q prime + cap-scale remain available for explicit high PTV_PREROLL_MS (bursty Fintech-class). PTV_PREROLL_MS overrides, PTV_NO_GENLOCK reverts to 350. */
    if (g_preroll_ms > 1600) g_delivery_cap_us += (int64_t)g_preroll_ms * 1000;  /* v0.9.0: the deep input prime delays VIDEO ~g_preroll_ms; the §7.5a gate holds audio+copy to match (it IS the audio-side of the whole-stream delay), so size its cap to the prime — else it force-releases and audio leaks ahead (TruBLU dlvforced). Explicit PTV_DELIVERY_CAP_MS (below) overrides. */
    if (g_preroll_ms > 1600) g_delivery_maxq = FFMAX(g_delivery_maxq, (int)(g_delivery_cap_us / 1000000 * 256));  /* v0.9.0: the deeper hold needs more FIFO nodes (≤ cap_s × Σ stream pkt-rates); without this a multi-audio channel (2 transcoded + copied AC-3) hits the maxq backstop and back-pressure-stalls before the cap. Explicit PTV_DELIVERY_MAXQ (below) overrides. */
    /* §13: a cushion deeper than frame_q (~1.6s) is carried by video_q -> size it to hold the
     * backlog (packets ~= preroll_ms x out_fps + margin), bounded. Default 350ms -> no change. */
    if (g_preroll_ms > 1600) { int pps = out_fps.num > 0 && out_fps.den > 0 ? (int)((out_fps.num + out_fps.den - 1) / out_fps.den) : 60;  /* v0.9.18.4 M6: exact video pkt/s (was pessimistic 60/s) — mutually consistent with deep_prime_pkts' out_fps sizing */
        int need = (int)((int64_t)g_preroll_ms * pps / 1000) + 64; if (need > 2048) need = 2048; if (g_videoq < need) g_videoq = need; g_aq_cap = PTV_AQ_PREROLL; }  /* deep prime: also raise the pre-h0 audio ring (default stays 256 = byte-identical) */
    { const char *dc = getenv("PTV_DELIVERY_CAP_MS"); if (dc && atoi(dc) > 0) g_delivery_cap_us = (int64_t)atoi(dc) * 1000; }  /* force-release ceiling (A0 ≈1.5–2s) */
    { const char *dq = getenv("PTV_DELIVERY_MAXQ");   if (dq && atoi(dq) > 0) g_delivery_maxq = atoi(dq); }                    /* hold-FIFO size backstop */
    /* v0.9.18.4 M6: video_q must be able to HOLD the auto-bank ceiling. Measured (59.94 fixture,
     * A/B): the default capacity (512 pkts = 8.5s @59.94 + ~2.7s frame_q ~= 11.2s) covers most
     * escalations — the exposed corner is a 59.94fps channel banking at the FULL 12s ceiling
     * (11.2 < 12; NTSC holds ~17s, never bound). Close the corner: live channels (the only ones
     * that can bank) get capacity for ceiling x out_fps. Slots are pointers; memory materializes
     * only while a bank actually holds (~5MB compressed at 3Mbps x 12s). */
    if (live) {
        int pps2 = out_fps.num > 0 && out_fps.den > 0 ? (int)((out_fps.num + out_fps.den - 1) / out_fps.den) : 60;
        int bank_need = (int)(g_cushion_max_ms / 1000 * pps2) + 64;
        if (bank_need > 2048) bank_need = 2048;
        if (g_videoq < bank_need) g_videoq = bank_need;
    }

    {
        /* §13: deep prime delays video ~preroll_ms, so audio must buffer that long without the
         * demux dropping on a full audio_q during bursts. Size audio_q to the cushion (~50 audio
         * frames/s + margin), bounded; default 350ms -> PTV_QDEPTH unchanged. */
        int aqd = PTV_QDEPTH;
        /* v0.9.14.2 AUTO-BANK: clumped delivery slams a whole burst of audio into this queue at
         * once (Unique_TV live: 7s clump = ~330 pkts into the old 48-deep queue -> adrop ~25/s =
         * HALF the audio dropped at the demux door = the clicks, with a perfect video bank and an
         * idle gate). Size it for the bank ceiling with the SAME formula the manual deep preroll
         * uses — the last of its three sizing side-cars (video_q, gate FIFO, audio_q) applied
         * automatically. Live only; a pointer array, so capacity is free. */
        if (live) { int need = (int)(g_cushion_max_ms * 50 / 1000) + 48; if (need > 2048) need = 2048; if (aqd < need) aqd = need; }
        if (g_preroll_ms > 1600) { int need = (int)((int64_t)g_preroll_ms * 50 / 1000) + 48; if (need > 2048) need = 2048; if (aqd < need) aqd = need; }
        cp->audioq_pkts = aqd;
    }

    /* §13: deep startup cushion target (packets ~= preroll_ms x fps). Single-input + multiview inputs
     * (both can be bursty, v0.9.0), and only when the cushion exceeds frame_q (~1.6s) -> then decode_thread delays its start
     * until video_q banks this much. Default 350ms -> 0 -> no delay (byte-identical).
     * NOTE: out_fps approximates the INPUT packet rate (exact when in==out fps, i.e. no -r
     * conversion / not field-rate); bursty single-input channels have in==out so it holds.
     * CLAMP to g_videoq-32 so the prime-wait is always satisfiable (video_q is the cap the
     * banked packets sit in; a target above it could never be reached -> always time out). */
    cp->deep_prime_pkts = 0;
    if (!multiview && out_fps.num > 0) {         /* v0.9.1: deep video_q prime is single-input only; multiview relies on the compositor's hold.q (already a paced per-input de-jitter buffer) */
        int tgt = (int)((int64_t)g_preroll_ms * out_fps.num / (1000LL * out_fps.den));
        if (tgt > g_videoq - 32) tgt = g_videoq - 32;
        if (tgt > g_frameq_cap - 8) cp->deep_prime_pkts = tgt;
    }
    cp->deep_prime_budget_us = (int64_t)g_preroll_ms * 2000;   /* 2x preroll_ms, in us (decode start-delay budget) */
    cp->vid_pps = out_fps.num > 0 && out_fps.den > 0
                ? (int)((out_fps.num + out_fps.den - 1) / out_fps.den) : 60;  /* v0.9.18.4 M6; unknown rate -> pessimistic 60 */

    cp->preroll_ms        = g_preroll_ms;
    cp->videoq_pkts       = g_videoq;
    cp->frameq_cap        = g_frameq_cap;
    cp->aq_prehold        = g_aq_cap;
    cp->delivery_cap_us   = g_delivery_cap_us;
    cp->delivery_maxq     = g_delivery_maxq;
    cp->cushion_raised_us = (int64_t)g_cushion_ms * 1000;
    cp->bank_ceil_us      = g_cushion_max_ms * 1000;
    cp->bank_decay_us     = g_bank_decay_us;
    (void)n_audio;   /* reserved: M6 folds the per-track pkt-rate sizing (35/50/60) in here */
}

