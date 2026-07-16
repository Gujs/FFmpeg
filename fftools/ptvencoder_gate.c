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
    /* §7.5b video side: OFF until dlv_video_cfg() arms it (single-input live + gated audio) */
    g->v_on = 0;
    g->vhead = g->vtail = NULL; g->vcount = 0; g->vmaxq = 512;
    g->v_band_us = PTV_VDLV_BAND_US; g->v_cap_us = 6000000;
    g->v_disarmed = 0; g->v_disarm_wc = 0;
    atomic_store(&g->a_dlv_dts_hi, INT64_MIN);
    atomic_store(&g->a_hi_change_wc, av_gettime_relative());   /* "delivered just now": birth isn't an audio death */
    atomic_store(&g->st_vhold_us, 0);
    atomic_store(&g->st_vforced, 0);
}

/* ===================== §7.5b (1.0.1-pre12) — SYMMETRIC delivery gate: the VIDEO side ==========
 * The §7.5a gate above holds EARLY AUDIO for the video encoder front. Its mirror problem
 * (owner-demonstrated live, AWE_Plus on cor-3, 2026-07-16): a buffering -af (the fleet-wide
 * loudnorm chain, ~3s analysis fill) delays AUDIO ~3s in WALL time while preserving its labels,
 * so at any wall instant the mux writes video PTS t next to audio PTS t−3s — ffprobe of the
 * output read video start 1134.03 vs audio start 1131.69 (audio content ~2.3s OLDER than the
 * concurrently-emitted video). Players are fine (PTS-aligned), but the WIRE is skewed:
 * sync_check-class monitors (video_last − audio_last) trip, and downstream buffers must cover
 * the gap. The audio-side gate has nothing to hold there — LATE audio is not queued anywhere;
 * video simply leaves first.
 * FIX: hold EARLY VIDEO, keyed on the AUDIO DELIVERED-DTS high-water (a_dlv_dts_hi — advanced
 * whenever dlv_drain/dlv_flush_all actually hands an audio/copy packet to mux_q): a video
 * packet whose DTS leads it by more than v_band_us queues here (FIFO, same DlvNode) until the
 * audio catches up. MEASURED, not assumed: on a channel whose audio is NOT wall-late, the audio
 * sits in the §7.5a gate AHEAD of the video front, so a_dlv_dts_hi tracks the front within a
 * tick and video always passes the band check inline — zero added latency, byte-identical wire.
 *
 * DEADLOCK INVARIANT (the two gates cannot close a cycle):
 *   - the AUDIO gate's release key is v_enc_dts_hi, published at ENCODE time
 *     (dlv_publish_video runs BEFORE the video packet can be held) — audio release never
 *     depends on video DELIVERY;
 *   - the VIDEO hold's release key is a_dlv_dts_hi, which advances whenever audio is DELIVERED
 *     — i.e. it depends only on v_enc_dts_hi + audio arrival, never on held video;
 *   - the video hold NEVER blocks its thread (append + return; a vmaxq overflow force-releases
 *     the OLDEST held packet — video is never dropped and never sleeps), so encoding always
 *     progresses and v_enc_dts_hi keeps advancing while frames flow.
 *   Chain: video encodes → v_enc_dts_hi advances → §7.5a delivers audio → a_dlv_dts_hi
 *   advances → held video releases. If the shape ever changes, the priority is explicit:
 *   the AUDIO gate yields — it must never gain a dependency on DELIVERED video.
 *
 * AUDIO-DEATH SAFETY (the make-or-break property): if a_dlv_dts_hi stops advancing for
 * v_cap_us (default 6s ≈ 2× the loudnorm class; PTV_VDELIVERY_CAP_MS) while video is held,
 * ALL held video flushes and the hold DISARMS (one WARNING) — an audio outage (dead track,
 * Azorse-class undecodable phase, source lost audio) degrades to the pre-pre12 wire, never a
 * frozen channel. It RE-ARMS (one INFO) when audio delivery advances again, and the hold then
 * re-forms so the skew re-closes. A per-packet age backstop (enqueue age > v_cap_us, counted
 * st_vforced) additionally releases through a flowing-but-permanently-behind audio path
 * (JLTV-class label spread), clamping the added latency at ~v_cap_us instead of pinning the
 * FIFO. All video-side state is single-threaded (the rung's video output thread) — no lock.
 * Single-input only: multiview slots' audio share one gate per rung, so the high-water would
 * key the hold to the LEAST-delayed slot — gated off at setup with a startup note.
 * PTV_NO_VDELIVERY=1 kills the video side everywhere (pre11 wire behavior). */
void dlv_video_cfg(DlvGate *g, int64_t band_us, int64_t cap_us, int vmaxq)
{
    g->v_band_us = band_us > 0 ? band_us : PTV_VDLV_BAND_US;
    g->v_cap_us  = cap_us  > 0 ? cap_us  : 6000000;
    g->vmaxq     = vmaxq   > 0 ? vmaxq   : 512;
    g->v_on      = 1;
}

/* deliver ONE audio/copy packet to mux_q and advance the audio delivered high-water (§7.5b's
 * release key). Takes ownership of *pkt. All callers run on the rung's video output thread
 * (dlv_drain + dlv_flush_all), so a_dlv_dts_hi is single-writer; atomics only for the stats
 * reader and style-consistency with v_enc_dts_hi. */
static void dlv_deliver_audio(DlvGate *g, AVPacket *pkt, int64_t dts_us)
{
    if (av_thread_message_queue_send(g->mux_q, &pkt, 0) < 0)
        av_packet_free(&pkt);
    /* the high-water advances either way — content left the gate (muxer-gone is terminal) */
    if (dts_us > atomic_load_explicit(&g->a_dlv_dts_hi, memory_order_relaxed)) {
        atomic_store_explicit(&g->a_dlv_dts_hi, dts_us, memory_order_relaxed);
        atomic_store_explicit(&g->a_hi_change_wc, av_gettime_relative(), memory_order_relaxed);
    }
}

/* send one held/passing video packet to mux_q. Returns the send error (mux gone) so the inline
 * caller can terminate the rung exactly as the pre-pre12 direct send did. */
static int dlv_video_send(DlvGate *g, AVPacket *pkt)
{
    int ret = av_thread_message_queue_send(g->mux_q, &pkt, 0);
    if (ret < 0)
        av_packet_free(&pkt);
    return ret;
}

/* §7.5b: route one just-encoded video packet — send now, or hold it as EARLY video. Takes
 * ownership of pkt. Video output thread only. Returns <0 only on a dead mux_q (rung exit). */
int dlv_video_deliver(DlvGate *g, AVPacket *pkt, int64_t dts_us)
{
    int64_t a_hi = atomic_load_explicit(&g->a_dlv_dts_hi, memory_order_relaxed);
    DlvNode *n;

    if (g->v_disarmed) {
        /* audio-death escape latched: video flows direct (pre-pre12 wire). Re-arm only when
         * audio DELIVERY has advanced again — a delivery that never advances the high-water
         * (frozen labels) must not re-arm, or the hold/escape pair would flap every v_cap_us. */
        if (atomic_load_explicit(&g->a_hi_change_wc, memory_order_relaxed) > g->v_disarm_wc) {
            g->v_disarmed = 0;
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-VDLV] audio delivery resumed — early-video hold re-armed\n");
        } else
            return dlv_video_send(g, pkt);
    }

    if (!g->vcount) {                       /* FIFO order: with anything held, a newer packet must queue */
        int due;
        if (a_hi != INT64_MIN)
            due = dts_us <= a_hi + g->v_band_us;
        else {
            /* Nothing delivered yet (birth). If audio is already WAITING in the §7.5a gate
             * (count > 0 — the normal birth: near-zero-latency audio held for this very video
             * front), video must flow: it IS the release key, and the wire is not skewed.
             * Only a still-silent audio path (loudnorm analysis fill / dead track) holds. */
            pthread_mutex_lock(&g->lock);
            due = g->count > 0;
            pthread_mutex_unlock(&g->lock);
        }
        if (due)
            return dlv_video_send(g, pkt);
    }

    n = av_mallocz(sizeof(*n));
    if (!n)                                 /* OOM → degrade to unheld; never lose video */
        return dlv_video_send(g, pkt);
    n->pkt = pkt; n->dts_us = dts_us; n->enq_us = av_gettime_relative(); n->next = NULL;
    if (g->vtail) g->vtail->next = n; else g->vhead = n;
    g->vtail = n; g->vcount++;
    if (g->vcount > g->vmaxq) {             /* backstop: force-release the OLDEST (never block/drop) */
        DlvNode *h = g->vhead;
        g->vhead = h->next;
        if (!g->vhead) g->vtail = NULL;
        g->vcount--;
        atomic_fetch_add_explicit(&g->st_vforced, 1, memory_order_relaxed);
        dlv_video_send(g, h->pkt);          /* send error is non-terminal here (drain semantics) */
        av_free(h);
    }
    dlv_video_drain(g);                     /* audio may already cover part of the hold */
    return 0;
}

/* §7.5b: release held video the audio delivery has caught up to (dts ≤ a_dlv_dts_hi + band),
 * plus the aged backstop; run the audio-death escape. Called by the rung's video output thread
 * after every dlv_drain (a_hi may have advanced) and from dlv_video_deliver. */
void dlv_video_drain(DlvGate *g)
{
    int64_t now, a_hi, adv;
    DlvNode *p;

    if (!g->v_on || g->v_disarmed)
        return;
    if (!g->vcount) {
        atomic_store_explicit(&g->st_vhold_us, 0, memory_order_relaxed);
        return;
    }
    now  = av_gettime_relative();
    a_hi = atomic_load_explicit(&g->a_dlv_dts_hi,   memory_order_relaxed);
    adv  = atomic_load_explicit(&g->a_hi_change_wc, memory_order_relaxed);

    if (now - adv > g->v_cap_us) {
        /* AUDIO-DEATH ESCAPE: nothing delivered for v_cap_us while video is held and flowing.
         * Release everything + disarm (one line) — degrade to the pre-pre12 wire, never a
         * frozen channel. Re-arms in dlv_video_deliver when delivery advances again. */
        int freed = g->vcount;
        while ((p = g->vhead)) {
            g->vhead = p->next;
            dlv_video_send(g, p->pkt);
            av_free(p);
        }
        g->vtail = NULL; g->vcount = 0;
        g->v_disarmed = 1; g->v_disarm_wc = now;
        atomic_store_explicit(&g->st_vhold_us, 0, memory_order_relaxed);
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-VDLV] no audio delivered for %.1fs with video flowing — released %d held "
               "video pkts, early-video hold DISARMED until audio delivery resumes "
               "(PTV_NO_VDELIVERY=1 disables the hold entirely)\n",
               (now - adv) / 1e6, freed);
        return;
    }

    while ((p = g->vhead)) {
        int due  = a_hi != INT64_MIN && p->dts_us <= a_hi + g->v_band_us;
        int aged = (now - p->enq_us) > g->v_cap_us;   /* audio flowing but permanently behind (label spread):
                                                       * clamp the added latency at ~v_cap_us */
        if (!due && !aged)
            break;
        if (aged && !due)
            atomic_fetch_add_explicit(&g->st_vforced, 1, memory_order_relaxed);
        g->vhead = p->next;
        if (!g->vhead) g->vtail = NULL;
        g->vcount--;
        dlv_video_send(g, p->pkt);
        av_free(p);
    }
    atomic_store_explicit(&g->st_vhold_us,
                          g->vhead ? now - g->vhead->enq_us : 0, memory_order_relaxed);
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
        dlv_deliver_audio(g, p->pkt, p->dts_us);   /* §7.5b: each delivery advances a_dlv_dts_hi */
        av_free(p);
        p = q;
    }
}

/* video thread is done: release EVERYTHING still held (shutdown — don't drop the tail), then mark
 * closed + wake any blocked enqueuer so it falls through to a direct send (no hang). */
void dlv_flush_all(DlvGate *g)
{
    DlvNode *out_head, *p;
    /* §7.5b first: the held EARLY video (this runs on the video thread — the only vhead
     * toucher); the muxer's interleaver reorders it against the audio flushed below by DTS. */
    while ((p = g->vhead)) {
        g->vhead = p->next;
        if (av_thread_message_queue_send(g->mux_q, &p->pkt, 0) < 0)
            av_packet_free(&p->pkt);
        av_free(p);
    }
    g->vtail = NULL; g->vcount = 0;
    pthread_mutex_lock(&g->lock);
    out_head = g->head;
    g->head = g->tail = NULL; g->count = 0;
    g->closed = 1;
    pthread_cond_broadcast(&g->space);
    pthread_mutex_unlock(&g->lock);
    for (p = out_head; p; ) {
        DlvNode *q = p->next;
        dlv_deliver_audio(g, p->pkt, p->dts_us);
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
    for (p = g->vhead; p; ) { DlvNode *q = p->next; av_packet_free(&p->pkt); av_free(p); p = q; }   /* §7.5b */
    g->vhead = g->vtail = NULL; g->vcount = 0;
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
 *   CUSHION_RELEASE: a0 = starvation-contradiction duration (us, log only)
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
    case CUSHION_RELEASE: {
        /* 1.0.1-pre10 (e): the starvation contradiction (frame_q starved with input FLOWING,
         * a0 = held duration) persisted >=60s while the tier was raised — holding a deeper
         * fill target for a buffer the decode deficit can never fill is the same contradiction
         * BANK_RELEASE answers, and the SHRINK release (6h of ZERO starvation episodes) is
         * unreachable while churning (every episode resets its clock). Same stores as SHRINK
         * (tier back to base + symmetric gate-cap restore), honest log. The caller guarantees
         * input was flowing (a genuine stall/outage keeps its cushion) and applies a 10min
         * post-release GROW suppression so the pair cannot flap once a minute under a
         * persistent deficit. g_delivery_maxq stays (design backstop, RAM-only high-water). */
        int64_t nc3, bw3;
        int r6;
        if (rt->cur_sp <= rt->base_sp)                     /* belt-and-braces: never double-restore */
            break;
        nc3 = atomic_fetch_sub_explicit(&g_delivery_cap_us,
            (int64_t)(rt->raised_sp - rt->base_sp) * rt->tick_dur_us, memory_order_relaxed)
            - (int64_t)(rt->raised_sp - rt->base_sp) * rt->tick_dur_us;
        bw3 = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
        rt->cur_sp = rt->base_sp;
        for (r6 = 0; r6 < rt->n_gate; r6++)
            if (rt->gate[r6])
                atomic_store_explicit(&rt->gate[r6]->cap_us, nc3 + bw3, memory_order_relaxed);
        if (g_diag)
            av_log(NULL, AV_LOG_INFO, "[PTV-GATE] caps -> %.1fs (base %.1fs + bank %.1fs) on %d gates (RELEASE)\n",
                   (nc3 + bw3) / 1e6, nc3 / 1e6, bw3 / 1e6, rt->n_gate);
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-CUSHREL] cushion released %d->%d frames: frame_q starved %.1fs with input "
               "flowing while the raised tier held — contradiction; gate caps restored, next "
               "GROW suppressed 10min (PTV_NO_CUSHREL disables)\n",
               rt->raised_sp, rt->base_sp, a0 / 1e6);
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

    /* §7.5b (1.0.1-pre12) video-side hold sizing. Cap default 6s = the loudnorm ~3s audio-latency
     * class with margin (mirrors PTV_DELIVERY_CAP_MS practice: precise release must win before the
     * backstop). FIFO backstop = cap × video pkt/s + margin, bounded — nodes are per-enqueue, so
     * capacity is free; the packets themselves bound RSS at ~cap × rung bitrate (6s × 3.8Mbps top
     * rung ≈ 2.9MB, ≈ 7MB across the 6-rung ladder at the ladder's summed ~9.7Mbps). */
    cp->vdlv_cap_us = 6000000;
    { const char *vc = getenv("PTV_VDELIVERY_CAP_MS"); if (vc && atoi(vc) > 0) cp->vdlv_cap_us = (int64_t)atoi(vc) * 1000; }
    cp->vdlv_maxq = (int)(cp->vdlv_cap_us / 1000000 * cp->vid_pps) + 64;
    if (cp->vdlv_maxq < 512)  cp->vdlv_maxq = 512;
    if (cp->vdlv_maxq > 2048) cp->vdlv_maxq = 2048;

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

