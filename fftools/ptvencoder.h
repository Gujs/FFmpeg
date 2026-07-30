#ifndef FFTOOLS_PTVENCODER_H
#define FFTOOLS_PTVENCODER_H

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

#define PTV_QDEPTH      48     /* demux->decode packet queue (~1s jitter) */
/* 0.9.18 R1+R3/R4 (map: analysis/ptvencoder-0918-implementation-map.md §2): demux-side
 * rate-sensing state, hoisted verbatim from function-local statics in demux_dispatch (R1),
 * with the published rate atomics folded in (R3). One per INPUT — the instance lives in
 * the Input struct (R4), wired to the demux (writer) and output threads (readers) at
 * transcode() setup like h0/house_skew. Non-zero initializers are set in the per-input
 * init loop (former statics'/globals' initializers, verbatim). */
typedef struct RateEstimator {
    int64_t c0, w0, ema_q20;        /* tight-FLL sub-window anchor + rate EMA (Q20) */
    int_least64_t ep_prev;          /* disturbance epoch of the current sub-window */
    int     chunks;                 /* clean FLL chunks (lock at 8) */
    int64_t cf_ema_q20;             /* coarse clock-follow rate EMA (Q20) */
    int     cf_chunks, cf_wins, cf_la_acc, cf_la_tot, cf_frozen;
    /* published outputs (demux thread writes, output threads read; relaxed atomics) */
    _Atomic int64_t cf_rate_q20;    /* coarse source rate (content-us/wall-us, Q20) */
    _Atomic int     cf_locked;      /* 20 clean chunks (~60s) before the servo may follow */
    _Atomic int64_t src_rate_q20;   /* recovered source rate (content-µs/wall-µs), Q20 */
    _Atomic int     src_rate_locked;/* 0 until the FLL trusts the estimate */
} RateEstimator;
/* 0.9.18 R1+R3/R4: master-output-side house-rate actuation state (the ladder's memory).
 * One per HOUSE CLOCK — the instance is a transcode()-scope struct shared by the rung set
 * (master writes, all rungs apply rho_corr_ppm), wired via VideoCtx like h0/house_skew. */
typedef struct HouseRateState {
    int cf_following;               /* clock-follow hysteresis latch (arm >5000, release <2000 ppm) */
    int64_t occ_ema_milli;          /* WUCR: EMA-filtered master frame_q occupancy (milli-frames); master-thread only, non-atomic */
    int     occ_ema_seeded;         /* WUCR: seed the EMA to the first occupancy sample so there is no startup-ramp transient */
    int64_t reprime_start;          /* 0.9.10.1: wall time the CURRENT reprime engagement began (0 = idle). An
                                     * engagement is hard-capped at 10s, then a 300s cooldown applies UNCONDITIONALLY
                                     * — the 0.9.10 "continuing" clause let occupancy oscillating around the trigger
                                     * re-arm forever (observed live: AWE film segments pinned the house at 0.77x). */
    int64_t reprime_last_end;       /* wall time the last engagement ended (cooldown reference) */
    _Atomic int64_t rho_corr_ppm;   /* WUCR ρ: applied house-rate correction (ppm, corr>0 = house slower); ±6% clamp (±30% under re-prime) */
} HouseRateState;
/* 0.9.18 M1 — CushionPlan: ALL cushion/queue sizing derived in ONE place (resolve_cushions()),
 * in the units each value is consumed in. The g_* globals above stay authoritative for every
 * consumer not yet repointed (mv compositor, BANK escalate/decay writes, adaptive GROW/SHRINK
 * runtime stores, stats/log lines); resolve_cushions() assigns them AND mirrors them here. */
typedef struct CushionPlan {
    /* startup-resolved (env + genlock defaulting + deep-prime side-cars) */
    int      preroll_ms;           /* resolved: env | genlock 1000 | 350   (replaces g_preroll_ms reads) */
    int      videoq_pkts;          /* video_q capacity                     (g_videoq) */
    int      frameq_cap;           /* per-rung frame_q capacity            (g_frameq_cap; parse stays in main()) */
    int      audioq_pkts;          /* per-track audio_q capacity (live/deep rules folded in) */
    int      aq_prehold;           /* pre-h0 audio ring depth              (g_aq_cap) */
    int      deep_prime_pkts;      /* video_q startup bank; 0 = off        (single-input, from out_fps) */
    int64_t  deep_prime_budget_us; /* decode start-delay budget (2x preroll) */
    int64_t  delivery_cap_us;      /* gate force-release BASE (pre-bank)   (g_delivery_cap_us at init) */
    int      delivery_maxq;        /* gate FIFO backstop                   (g_delivery_maxq) */
    int64_t  cushion_raised_us;    /* adaptive RAISED tier                 (g_cushion_ms*1000) */
    int64_t  bank_ceil_us;         /* AUTO-BANK ceiling                    (g_cushion_max_ms*1000) */
    int64_t  bank_decay_us;        /* quiet time before the bank retires   (g_bank_decay_us) */
    int      vid_pps;              /* v0.9.18.4 M6: VIDEO packets/second = ceil(out_fps) — the ONE
                                    * seconds->video_q-packets rate (was three constants: bank 35/s,
                                    * side-car 60/s, deep-prime out_fps; a 50/59.94fps channel's bank
                                    * was sized ~30-42% under its us target). AUDIO sizing keeps its
                                    * own 50/s: that is an AAC frame rate (48kHz/1024 ~= 47/s), not
                                    * a video rate — intentionally NOT unified. */
    int64_t  vdlv_cap_us;          /* §7.5b video-hold escape/age backstop (PTV_VDELIVERY_CAP_MS, default 6s
                                    * = the loudnorm ~3s audio-latency class with margin) */
    int      vdlv_maxq;            /* §7.5b video hold FIFO backstop (vdlv_cap × vid_pps + margin) */
} CushionPlan;
/* ===================== §7.5a delivery-alignment gate (P1) — per output rung =====================
 * Dense near-zero-latency streams (transcoded audio + copied AC-3/MP2) are HELD here instead of
 * going straight to mux_q; the rung's VIDEO encode_push publishes its newest emitted DTS and DRAINS
 * every held packet whose DTS the video has reached — so audio/copy reach the muxer in lockstep with
 * the (≈1s later) video for the SAME content, instead of ~1s ahead on the wire. PTS are NEVER
 * modified — only WHEN a packet reaches the muxer. One drainer (the rung's video output thread),
 * many enqueuers (N audio threads + the demux/copy thread); a small mutex guards the list. NO control
 * loop — a deterministic release gate (NFR-SIMPLE). */
typedef struct DlvNode {
    AVPacket       *pkt;
    int64_t         dts_us;     /* packet DTS on the (content − h0) µs axis (shared with video) */
    int64_t         enq_us;     /* monotonic wall time enqueued (for the cap_us age release) */
    struct DlvNode *next;
} DlvNode;

typedef struct DlvGate {
    AVThreadMessageQueue *mux_q;        /* release target (this rung's muxer queue) */
    pthread_mutex_t lock;
    pthread_cond_t  space;              /* signalled when the drain frees a slot (blocking enqueuers wait) */
    DlvNode        *head, *tail;
    int             count, maxq;
    int             closed;             /* video thread done → enqueuers fall through to a direct send */
    int             inited;
    _Atomic int64_t cap_us;     /* v0.9.14: runtime-adjustable (AUTO-BANK extends it so audio waits out long stalls) */
    _Atomic int64_t v_enc_dts_hi;       /* newest video DTS the encoder has emitted (µs); INT64_MIN = none yet */
    _Atomic int64_t v_hi_change_wc;     /* monotonic wall time v_enc_dts_hi last ADVANCED — stall detector:
                                         * the cap force-release fires only when video is GENUINELY stuck
                                         * (this hasn't moved within cap_us), NOT when the pipeline merely
                                         * carries a long-but-healthy steady hold (preroll + audio-ahead +
                                         * encoder latency ≈ a few s). Absolute-age cap mis-fired there. */
    /* stats (NFR-OBS) */
    _Atomic int64_t st_hold_us;         /* age of the oldest still-held packet at the last drain */
    _Atomic int64_t st_forced;          /* cap_us-forced releases (encoder latency > cap) */
    _Atomic int64_t st_dropped;         /* non-blocking copy drops on a full FIFO */
    /* ---- §7.5b (1.0.1-pre12) SYMMETRIC gate: the video-side hold (see ptvencoder_gate.c
     * header for the model + the deadlock invariant). vhead/vtail/vcount are touched ONLY by
     * the rung's video output thread (deliver + drain + flush all run there) — no lock. ---- */
    int             v_on;               /* video hold armed (single-input live + gated audio present) */
    DlvNode        *vhead, *vtail;      /* held EARLY video, FIFO (video output thread only) */
    int             vcount, vmaxq;      /* vmaxq: force-release-oldest backstop (never blocks, never drops) */
    int64_t         v_band_us;          /* tolerance: video may run this far past a_dlv_dts_hi (PTV_VDLV_BAND_US) */
    int64_t         v_cap_us;           /* audio-death escape + per-packet age backstop (PTV_VDELIVERY_CAP_MS) */
    int             v_birth_flow;       /* task #48 (mv): video FLOWS at birth until the first audio delivery —
                                         * mv audio anchors at first DISPLAY, so at the first video packet the
                                         * §7.5a FIFO is still empty (count==0) and the single-input birth rule
                                         * would hold video, unbounding the encoder's startup race past mux_q
                                         * backpressure and flipping healthy channels' equilibrium to
                                         * dlvhold=0/vdlvhold≈1s (measured). Flowing at birth keeps healthy mv
                                         * stock-identical; a wall-late audio path still converges once its
                                         * first delivery lands (one-time ≤fill transient). */
    int             v_disarmed;         /* audio-death escape tripped → video flows direct until audio resumes */
    int64_t         v_disarm_wc;        /* wall time of the disarm (re-arm when a_hi_change_wc passes it) */
    /* 1.0.1-pre17 (owner mandate: af-independence must never need per-channel config): the
     * escape/age cap AUTO-SIZES to the MEASURED audio-chain lateness. v_cap_base = the
     * configured floor (default 6s / PTV_VDELIVERY_CAP_MS, now an override-floor, not a
     * tuning knob); v_lat_ema = EMA of (v_enc_dts_hi − a_dlv_dts_hi) sampled per video
     * drain WHILE audio is actively delivering (a_hi_change_wc fresh <2s — an audio
     * OUTAGE must not inflate the escape timer); v_cap_us = clamp(base, 1.5×ema,
     * PTV_VDLV_CEIL_US) recomputed there. A >4s -af chain (the class the legend used to
     * say "raise the env for") now sizes its own hold; the 12s ceiling matches the
     * CUSHION_MAX posture (beyond it = an upstream incident, not something to buffer).
     * All fields video-output-thread-only (like the rest of the v-side). */
    int64_t         v_cap_base;         /* configured floor (env or 6s default) */
    int64_t         v_lat_ema;          /* EMA of the enc-vs-delivered content skew (µs) */
    int             v_lat_seed;         /* EMA seeded */
    int64_t         v_cap_log_wc;       /* rate limit for the cap-resize INFO line */
    _Atomic int64_t a_dlv_dts_hi;       /* newest audio/copy DTS DELIVERED to mux_q (µs); INT64_MIN = none yet */
    _Atomic int64_t a_hi_change_wc;     /* wall time a_dlv_dts_hi last ADVANCED (escape/re-arm detector) */
    _Atomic int64_t st_vhold_us;        /* stats vdlvhold=: age of the oldest held video at the last video drain */
    _Atomic int64_t st_vforced;         /* backstop releases (vmaxq overflow + aged-while-audio-flowing) */
    /* ---- 1.0.1-pre18 PER-STREAM delivered watermarks (g_perstream_wm; the pre17 (B)
     * KNOWN OPEN, owner-approved): the shared a_dlv_dts_hi is the LEAST-delayed stream's
     * high-water, so on a MIXED rung (loudnorm'd AAC ~3s late + copied AC-3 ~0s) the
     * slower stream's wire alignment depended on latency coincidence. The video hold now
     * keys on the MINIMUM watermark across currently-LIVE gated audio streams (per output
     * stream_index); a stream with no delivery for >2s is stale and EXCLUDED, so a dead
     * track can never hold video (the §7.5b disarm's audio-outage class stays keyed on
     * the aggregate a_hi_change_wc). All fields video-output-thread-only (delivery and
     * the video hold both run there) — no lock, plain fields. */
#define PTV_DLV_MAX_AS 16
    int             as_n;                   /* registered gated audio streams (first-delivery) */
    int             as_idx[PTV_DLV_MAX_AS]; /* output stream_index per slot */
    int64_t         as_hi[PTV_DLV_MAX_AS];  /* per-stream delivered-DTS high-water (µs) */
    int64_t         as_wc[PTV_DLV_MAX_AS];  /* wall µs of that stream's last delivery (staleness) */
} DlvGate;

/* §7.5b auto-size ceiling (1.0.1-pre17): the measured-lateness cap growth stops here — an
 * audio chain claiming to buffer more than this is an upstream incident to surface, not
 * latency to add (the CUSHION_MAX philosophy). An explicit PTV_VDELIVERY_CAP_MS above the
 * ceiling still wins (the floor is never clipped). */
#define PTV_VDLV_CEIL_US 12000000
/* §7.5b tolerance band: a video packet may lead the audio DELIVERED high-water by this much
 * before it is held. Must exceed one video tick + one audio frame (~60ms) so a healthy channel
 * (audio waiting in the gate, a_hi tracking the video front within a tick) NEVER holds — and it
 * is the steady wire skew a held (loudnorm-class) channel converges to, so keep it well under
 * the ~2s sync_check class; 300ms also clears the muxer's 200ms max_interleave_delta. */
#define PTV_VDLV_BAND_US 300000

#define PTV_MAX_RUNG 8
#define PTV_MAX_AUDIO 8    /* max transcoded audio output tracks (multi-language, multiview slots) */
#define PTV_AQ_PREROLL 1024 /* per-track pre-h0 audio buffer (frames, ~21s @47fps): preserve a slot's audio head
                            * while its video decodes its first frame (sets h0), instead of dropping
                            * it (which made that slot's audio start ~h0-acquire-delay late). Bounded
                            * ring (drop-oldest) so a never-arriving video can't grow it unboundedly. */
#define PTV_MAX_INPUT 4    /* max composited inputs (multiview): 1 / 2 / 4 */
#define PTV_MV_SKEW_CAP_US 250000   /* multiview per-slot audio skew cap (async budget) */

/* ===================== 1.0.1-pre15 — glue classification (#33) =====================
 * NORMATIVE DESIGN = analysis/ptvencoder-33-glue-classification.md (§refs below are its
 * sections). Classification/routing fixes on the pre4–pre7 glue machinery — the flush tree,
 * AGLUE verdicts, pair-expect handshake and absorber all stay; what changes is which OWNER
 * each event class routes to. One revert switch: PTV_NO_GLUECLASS=1 (g_glueclass).
 *   E5 pad ledger: per track, the last N applied GAP-pads {step_us, wall_us} not yet
 *   cancelled (written at the forward GAP verdict, consumed by §2.2 rule 3a — a backward
 *   step matching a recent pad is the pad's RETURN leg and must be APPLIED (aresample drops
 *   the pad's inserted silence), never relabel-erased: the rr14-A3 b1 round-trip bake).
 *   The NEWEST open entry is also published per track (atomics below) so the demux §5.A.2
 *   absorber can decline to erase the return leg at the packet layer (where sub-1s backward
 *   steps are normally absorbed before AGLUE ever sees them). Match rule (both sites):
 *   |step + pad| <= max(80ms, pad/4), pad younger than the TTL. */
#define PTV_GLUE_PAD_LED     4
#define PTV_GLUE_PAD_TTL_US  (120LL * AV_TIME_BASE)
/* 1.0.1-pre16 #47 (Fashion 2026-07-18): sanity bounds the two live incidents demanded.
 * MAX_ROUTE: a flush mismatch beyond this is REFUSED regardless of label health — source
 * A/V misalignment beyond ~2min has no content-alignment reality (largest REAL routed
 * mismatch ever observed: PATRIOT +30.8s; Fashion's ±11594s opposite-jump pair computed a
 * +23188s "mismatch" and pre15 routed it to the resampler). TW_CAP: the §2.4 tripwire may
 * synthesize at most this much at the swr boundary; a parked slip beyond it is upstream
 * chaos, not something to actuate (Fashion: tripwire synthesized ~2800s and pinned the
 * channel at async for hours). */
#define PTV_GLUE_MAX_ROUTE_US (120LL * AV_TIME_BASE)
#define PTV_GLUE_TW_CAP_US    (2LL * AV_TIME_BASE)
/* 1.0.1-pre23 #60/#61 BOUNDED CONVERGENCE (g_convcap; PTV_NO_CONVCAP=1 reverts A+B+C as one).
 * The Avivando OOM class (2026-07-23, 183GB anon-RSS → whole-box outage): a channel with
 * RECURRING label-jump seams on a continuously flowing feed hands each above-cap AGLUE step
 * to "aresample converges it"; the convergence never completes before the next seam, the
 * re-measured step stacks the unconverged backlog (each order ~2× the last: +1.2s→+702s in
 * 23min), and every order is an UNCAPPED swr_inject_silence allocation. Three layers:
 *   A admission cap  — a single accepted order above PTV_CONV_CAP_S (default 60s) is never
 *     handed to the resampler: folded label-neutrally instead (glue_off, the pre20 fold —
 *     door labels continuous, nothing reaches swr). The 2026-07-18 Q5 mandate is PRESERVED
 *     for in-cap orders (PATRIOT 30.8s passes through; latency stays unbounded while a
 *     convergence SHRINKS).
 *   B ladder escape  — a NEW above-cap step arriving while an order is in flight, whose
 *     magnitude (≈ the re-measured total backlog — the shared-flush handshake re-measures
 *     A-vs-V including the unconverged remainder) is NOT SHRINKING vs the in-flight order,
 *     is the ladder signature: fold the entire backlog, don't stack a bigger order. A
 *     genuinely draining convergence (strictly shrinking between seams) is left alone.
 *   C seam-park      — ≥3 fold-escapes (A or B) on one track within a rolling hour puts the
 *     track in SEAM-PARK for the next hour: every above-cap step folds immediately (the
 *     rc1-like parking posture that survived the same sources, but label-neutral and loud).
 * ESC ring: escape wall-times kept for the rolling recurrence window. */
#define PTV_CONV_ESC          8
/* 1.0.1-pre24 #63 WALL-EVIDENCE SPLIT (g_wallev; PTV_NO_WALLEV=1 reverts to the pre23
 * behavior at every touched site — the storm-diag counterfactual). Root cause (storm-diag
 * 2026-07-24, arithmetic closed against a flash+beep ruler): REAL content holes
 * (corrupt-discarded frames) were classified as label lies and ERASED (butt-joint / relabel /
 * fold) instead of padded, by three per-stream engines with no cross-stream conservation —
 * the A−V difference of erased totals is the permanent on-air offset (storm1: audio erased
 * 17452ms − video erased 12534ms = +4918 ≡ ruler ≡ ledger R). Principle: split every forward
 * label step J into W = wall-absence-evidenced portion (real missing content — PAD: labels
 * keep the hole, aresample pads audio / the house clock dups video) and J−W = flowing/relabel
 * portion (a lie — ERASE, exactly as today). W = clamp(delivery wall gap − cadence EMA, 0, J);
 * evidence floor g_gap_min_us (700ms); W forced 0 (fall back to today) when delivery is
 * bursty/banked (g_bank_us armed / bursty advisor) — wall absence is then not content
 * evidence. HARDCAP: the conv-fold exemption for wall-evidenced gaps is bounded by an
 * outstanding-pad backlog ledger — above it the fold happens anyway (allocation safety trumps
 * sync, one loud admission): the #60 OOM protection is never regressed. */
#define PTV_WALLEV_HARDCAP_US (120LL * AV_TIME_BASE)
/* NBS silence-fill sentinel (§3): a zero-size AVPacket carrying this PRIVATE flag bit on a
 * track's audio_q tells the audio thread to synthesize one quantum of silence at the track's
 * expected next graph-door pts (demux-side corrupt-discard starvation — the thread itself is
 * blocked on recv and cannot self-detect). Bit chosen above libavcodec's AV_PKT_FLAG_* space;
 * never leaves the process. */
#define PTV_PKT_FLAG_NBS_FILL (1 << 16)

/* A/V PLL redesign Phase A probe (PTV_AVSYNC_PROBE): per-input ring recording, for each DISTINCT
 * video content the cell displayed, the output time it went out at — (abs source pts → out_v, both
 * us). The audio drain pairs its emitted frame's source content against this ring to read the output
 * time the VIDEO showed that SAME content (§3.2b), giving the real lip-sync offset out_v(C)−out_a(C).
 * Written by the compositor (multiview) and the single-input output thread; read by audio_drain_fg.
 * Single-producer/single-consumer; the small lock keeps a torn read out of the diagnostic. */
#define PTV_VRING 512      /* distinct video contents kept (~10s @50fps; spans the V↔A content offset) */
typedef struct VOutRing {
    int64_t          src[PTV_VRING];   /* absolute source pts of the displayed content (us) */
    int64_t          out[PTV_VRING];   /* output time that content was emitted at (us, output-PTS axis) */
    int64_t          n;                /* total writes (monotonic); newest index = (n-1) % PTV_VRING */
    pthread_mutex_t  lock;
} VOutRing;
/* ===================== 1.0.1-pre9 — PASSIVE residual lip-sync SENSOR =====================
 * Component 1 of the residual-sync supervisor (analysis/ptvencoder-residual-sync-supervisor.md).
 * NON-BLIND by construction: each stream's SOURCE→OUTPUT content mapping is measured
 * independently against the post-demux (post-glue) label domain, PLUS a per-stream ledger of
 * every label EDIT the pipeline itself made, so any edit that shifts ONE stream's mapping
 * shows up while shared source discontinuities cancel:
 *   m_v = EMA[out_v − src_v]            (output thread, per emitted frame incl. dups —
 *                                        the dup ratchet and label-followed jumps are REAL
 *                                        presentation shifts and must be measured, not read
 *                                        from house_skew, which is a control variable)
 *   m_a = EMA[out_a − (sink_src − inj) − slip]
 *                                       (audio drain, per emitted frame; inj = the label
 *                                        offsets the audio thread itself injected at the graph
 *                                        door = AGLUE glue_off + corr + the PATH term (AVLOCK
 *                                        house_skew on single/non-follow, af_steer_us on the
 *                                        mv follow path — pre16 mirrors the door bus), so the
 *                                        reference recovers the RAW post-demux label; slip =
 *                                        the resampler's UN-REALIZED correction beyond a 50ms
 *                                        dead band — the parked-slip class [PTV-SWRDELAY]
 *                                        exists for, which label math alone cannot see.
 *                                        1.0.1-pre11: slip is scoped to the async-aresample
 *                                        FILTER boundary (its own in/out link label heads −
 *                                        swr_get_delay), NOT the whole -af graph — a buffering
 *                                        filter's hold (loudnorm ~3s) preserves labels, so it
 *                                        is latency, not slip; the pre9 whole-graph probe read
 *                                        it as a constant false audio-early = the hold)
 *   E_s = per-stream demux label-edit ledger (µs): discontinuity self-rebase (§5.A.2 absorbs),
 *         LAYERA flush applied_offset persists, pre5 retro-corrections. Pure 2^33 wraps are
 *         EXCLUDED (always genuine, always shared; including them would spike R by 26.5h
 *         during every A-before-V wrap straddle).
 *   R = (m_v + E_v) − (m_a + E_a)       (+ = video presented later = audio EARLY, the
 *                                        external oracle's convention; R = −(disc-oracle
 *                                        "ADDED" which prints + = audio made later))
 * Both m terms contain −h0 and both realized house retimings (AVLOCK) — those cancel in R:
 * shared latency is NOT desync (the wedge/AUTO-BANK posture). What does NOT cancel: AGLUE
 * relabel-erases (glue_off), per-stream-unequal demux rebases (E_v−E_a — the wrong-glue /
 * partner-step-bake class), parked resampler slip, and any label-followed single-stream jump.
 * PASSIVE: nothing consumes R — it feeds the stats line (`lipsync=`) and the [PTV-RSYNC]
 * DIAG line only. The corrector consumes R via rsync_track_R() (pre14; armed on mv since
 * pre17). PTV_RSYNC_SENSE=0 disables (sensor + slip probe + the pre15
 * realization tripwire, both modes).
 * 1.0.1-pre16 MV SENSOR PORT (analysis/ptvencoder-mv-sensor-port.md): the video half is
 * PER-SLOT — mv_ema/mv_wall/ev_us are arrays keyed by input slot. Single writer per field,
 * relaxed atomics, no locks: mv_ema/mv_wall[s] — ONE writer thread (the compositor on mv at
 * the sk-measurement/display site, per house tick incl. dup/residence holds; the master
 * output thread = slot 0 on single input); ev_us[s] — slot s's demux thread; ma_ema, ma_wall
 * and ea_us[k] — track k's audio/demux threads (unchanged, GLOBAL track index). A slated slot stops
 * publishing → mv_wall[s] stales past 3s → that slot's tracks read `--` (the single-input
 * stale-anchor rule, per slot). R pairs track k's audio terms with its OWN slot's video
 * terms inside rsync_track_R() — per-(slot,track) residuals, cross-slot independent. */
typedef struct RsyncSense {
    _Atomic int64_t mv_ema[PTV_MAX_INPUT];  /* per-slot video EMA(out−src) µs (compositor / master rung writes) */
    _Atomic int64_t mv_wall[PTV_MAX_INPUT]; /* wall µs of the slot's last video sample (freshness) */
    _Atomic int64_t ev_us[PTV_MAX_INPUT];   /* per-slot video label-edit ledger E_v (µs, slot's demux writes) */
    _Atomic int64_t ma_ema[PTV_MAX_AUDIO];  /* per-track audio EMA (audio threads write) */
    _Atomic int64_t ma_wall[PTV_MAX_AUDIO]; /* wall µs of the last audio sample */
    _Atomic int64_t ea_us[PTV_MAX_AUDIO];   /* per-track audio stream ledger E_a (µs) */
    /* 1.0.1-pre24 #63 wall-evidence PROVENANCE ledgers (always measured, even under
     * PTV_NO_WALLEV — they feed the re-anchor corroboration, never control directly):
     * cumulative UNEVIDENCED real-content deletions (µs, ≥0) — label erases whose event-time
     * wall evidence said the content was genuinely ABSENT (a real hole erased = a real on-air
     * desync of that amount on that stream). Erases of wall-FLOWING relabels (lies — correct
     * on air) are deliberately NOT counted: aseam-class pinned-R false positives read 0 here.
     * Demux thread posts (fetch_add; single logical writer per slot/track). */
    _Atomic int64_t uev_us[PTV_MAX_INPUT];  /* per-slot VIDEO unevidenced deletions */
    _Atomic int64_t uea_us[PTV_MAX_AUDIO];  /* per-track AUDIO unevidenced deletions (demux side;
                                             * the door adds its own AudioState.u_door_us) */
    _Atomic int64_t hh_q10[PTV_MAX_AUDIO];  /* pre24: the audio stream's demux label-health H
                                             * (Q10 EMA — see PtvDiscStreamState) published for
                                             * the re-anchor health gate (0 = no reading yet) */
    int             n_a;                    /* transcoded tracks wired (pre16: = n_audio ALWAYS, mv included) */
    int             a_in[PTV_MAX_AUDIO];    /* track → input slot map (stats printers; plain int,
                                             * set in transcode() before threads spawn) */
} RsyncSense;

/* ===================== 1.0.1-pre14 — residual-sync CORRECTOR =====================
 * Component 2 of the residual-sync supervisor (analysis/ptvencoder-corrector-design.md —
 * the normative spec; §ref in comments below are ITS sections). The actuation half of the
 * pre9/pre11 sensor: when input is healthy and the per-track residual R (the stats-line
 * `lipsync=`, + = audio EARLY) dwells outside a dead band, steer it to zero through the
 * resampler — the pre3 graph-door steer-bus actuator (§5), rate-clamped to 2ms/s. A trim
 * loop, NOT a controller: authority is deliberately too small to paper over a structural
 * glue failure (§6 damage bound, owner-approved caps: 5s/engagement, 10s lifetime).
 * MV-NORMATIVE (§8): one CorrState per (input-slot, audio-track), embedded in AudioState;
 * the corrector consumes R only through the rsync_track_R() accessor (pre16: per-slot video
 * ledgers keyed by AudioState.dbg_in — the slot hard-wire died inside the accessor).
 * 1.0.1-pre17: ARMED ON MULTIVIEW (the pre16 sensor-first hold removed): liveness is
 * per-RUNG (g_mux_sent_wc + gate watermarks, §3 — one dead rung holds the whole channel by
 * design) AND per-INPUT (the steered track's own input's arrival watermark,
 * Input.v_arrive_wc); event freeze is per-slot (all pre16 feeds resolve through
 * dbg_in/per-track state) + the sibling-slate condition (g_mv_slate_mask — while ANY slot
 * is black-slated, no track may engage: a sibling outage perturbs the shared compositor
 * pacing, so R is held un-actionable for the whole mosaic until slate + quiet clear).
 * DEFAULT ON single-input (owner-directed 2026-07-17); PTV_NO_RSYNC_CORR=1 kills it
 * outright (§6/§7).
 * All state below is owned by that track's audio thread; cross-thread visibility goes
 * through the g_corr_* published atomics only. */
enum {
    PTV_CORR_OFF = 0,      /* not opted in (or killed): no state machine, corr_us stays 0 */
    PTV_CORR_ARMED,        /* enabled, |R| inside the engage band (or no valid reading yet) */
    PTV_CORR_DWELL,        /* |R| > engage band — accumulating the 5min stable dwell (§4.3) */
    PTV_CORR_ENGAGED,      /* steering: proportional R/30s, slew-clamped 2ms/s (§4) */
    PTV_CORR_PARKED,       /* converged (|R|≤20ms held 60s): corr_us RETAINED, integration off */
    PTV_CORR_DISARMED      /* auto-disarm (§6): re-arm only via a fresh full dwell */
};
typedef struct CorrState {          /* all owned by that track's audio thread (§4) */
    int64_t corr_us;                /* cumulative applied trim (µs) — the steer-bus term */
    int     state;                  /* PTV_CORR_* */
    int64_t dwell_start_wc;         /* wall µs |R| first exceeded the engage band */
    int64_t dwell_r0;               /* R at dwell start (stability reference, §4.3) */
    int64_t ev_snap, ea_snap;       /* ledger snapshots at dwell start (event detect, §4.4) */
    int64_t epoch_snap;             /* input disturb_epoch snapshot */
    int     glue_snap, acq_snap;    /* glue_events / pll_acq_count snapshots */
    int     afmt_snap, reopen_snap; /* AFMT rebuilds / ADECWD reopens snapshots */
    int64_t bank_snap;              /* g_bank_us snapshot (armed-and-filling detect) */
    int64_t hs_snap;                /* house_skew at dwell start (|Δhs|<50ms freeze) */
    int64_t engage_r0;              /* R at engage (overshoot + convergence logging) */
    int64_t engaged_corr0;          /* corr_us at engage (per-engagement 5s authority) */
    int64_t engage_wc;              /* wall µs of the engage (PARK line duration) */
    int64_t park_wc;                /* wall µs |R| first inside the 20ms target band */
    int64_t implaus_wc;             /* wall µs R first read implausible (>5s) */
    int64_t last_event_wc;          /* wall µs of the last event hit (3min quiet + engaged 60s re-quiet) */
    int64_t slip_bad_wc;            /* wall µs rs_slip_us went nonzero while ENGAGED (60s → disarm) */
    int64_t rst_win_wc;             /* event-storm window start (≥3 counted dwell resets / 10min) */
    int     rst_cnt;                /* counted dwell resets in the window */
    int64_t holdoff_wc;             /* re-arm holdoff after a storm disarm (wall µs) */
    int64_t log_wc;                 /* HOLD-line rate limit (state-change lines are not limited) */
    int64_t diag_wc;                /* PTV_DIAG 30s progress-line rate limit while engaged */
    /* 1.0.1-pre18 #51b anti-starvation ceiling (the legacy-0007 PLL_HARD_CEILING/PLL_STUCK
     * pattern): the starvation SPAN — how long R has stayed large (>engage band) AND flat
     * (the §4.3 criterion vs starve_r0) with the sensor valid and delivery live, while the
     * corrector could not complete a dwell (resets, storm holdoffs included — the span runs
     * through ARMED/DWELL/DISARMED). Span ≥ the ceiling → ENGAGE anyway (rscorr_update). */
    int64_t starve_wc;              /* span start (wall µs); 0 = no open span */
    int64_t starve_r0;              /* R at span start (the whole-span flatness reference) */
    int     perm_disarm;            /* 1.0.1-pre21 #24: lifetime-cap disarm is FINAL for this
                                     * process (10s of trim failed to close R — re-arming would
                                     * walk the same doomed staircase); logged once at the cap */
} CorrState;

/* 0.9.18 M3 — CushionRt: the runtime COORDINATION home for cushion escalation (map §1.3).
 * Holds the per-run wiring every escalation event needs (the per-rung delivery gates, the
 * master tick, the adaptive tier targets) plus the mutex that makes cushion_escalate() the
 * single serialized writer of every escalation-dependent store. The escalation DATA itself
 * (g_bank_us / g_bank_pkts / g_delivery_cap_us / g_delivery_maxq) stays in the globals
 * above — the hot-path readers (master blocking-push arming, dlv_drain cap loads, the
 * stats lines, the WUCR bank top-up floor) keep reading the same relaxed atomics
 * unchanged; moving the data home is a later step.
 * Registration: gate[]/n_gate/tick_dur_us are set in transcode() setup right after
 * dlv_init(), before any thread starts. base_sp/raised_sp/cur_sp are set by the MASTER
 * output thread at startup (they are per-run locals derived from the resolved preroll +
 * tick there); cur_sp is thereafter mutated ONLY inside cushion_escalate() on the
 * master's own GROW/SHRINK calls, so the master's unlocked trigger/stats reads are
 * same-thread-ordered and race-free (the demux-side BANK events never touch the tier
 * fields). */
typedef enum { CUSHION_GROW, CUSHION_SHRINK, BANK_ESCALATE, BANK_RETIRE,
               BANK_RELEASE,   /* 1.0.1-pre8 (b): starvation-contradiction fast release */
               CUSHION_RELEASE /* 1.0.1-pre10 (e): tier release on the same contradiction —
                                * the 6h zero-starvation SHRINK is unreachable while churning */ } CushionEvent;
typedef struct CushionRt {
    DlvGate        *gate[PTV_MAX_RUNG]; /* per-rung delivery gate (NULL = delivery off) */
    int             n_gate;
    int64_t         tick_dur_us;        /* master video tick (us) */
    int             base_sp;            /* adaptive tier: BASE frame_q target (ticks) */
    int             raised_sp;          /* adaptive tier: RAISED frame_q target (ticks) */
    int             cur_sp;             /* adaptive tier: current target (escalate-owned) */
    int64_t         bank_advise_us;     /* v0.9.14: rate-limit for the at-ceiling advisory */
    pthread_mutex_t lock;               /* cold path: events are seconds-to-hours apart */
} CushionRt;
/* Multiview per-input jitter buffer: decode pushes each frame onto `q`; the
 * compositor pops ONE per house tick (absorbing bursty decode delivery, exactly
 * like the single-input frame_q) and dup-holds its last frame when `q` underruns,
 * so a late/dead slot never stalls the mosaic. A depth-1 "latest only" hold
 * instead would discard intra-burst frames -> massive dup/judder. */
typedef struct VideoHold {
    AVThreadMessageQueue *q;   /* decode -> compositor (AVFrame*) */
    int64_t         framedrop; /* drop-oldest count when q overflows (live) */
    pthread_mutex_t lock;      /* guards wall_us + eof */
    int64_t         wall_us;   /* when a frame was last pushed (staleness -> slate) */
    int             eof;       /* decode terminated for this input (terminal) */
} VideoHold;

/* Shared decode side of the ABR ladder (the ffmpeg model: decode the source
 * ONCE, run it through one filter graph — a -filter_complex `split`, a single
 * -filter:v chain, or none — and hand each rung its own frames via that rung's
 * frame_q). One decoder + one graph feeding N independent outputs.
 *
 * Multiview: `hold` is set and `filtering` is 0 — the decode thread stages each
 * frame into the per-input hold instead of running the graph; the compositor
 * owns the (N-input) graph and the frame_q fan. */
typedef struct DecodeCtx {
    AVThreadMessageQueue *video_q;            /* demux -> decode (AVPacket*) */
    AVCodecContext  *vdec;
    AVRational       ist_tb;                  /* decoder pkt time_base */
    int64_t         *h0;                      /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    int              live;
    VideoHold       *hold;                    /* multiview: stage frames here (NULL = filter inline) */
    /* filter graph: filtering -> N buffersinks (one per rung); else clone decode */
    int              filtering;
    AVFilterGraph   *fg;
    AVFilterContext *fsrc;
    int              n_rung;
    AVFilterContext *fsink[PTV_MAX_RUNG];
    AVThreadMessageQueue *frame_q[PTV_MAX_RUNG];
    int64_t          framedrop[PTV_MAX_RUNG];
    /* shared counters (the master output thread reports them) */
    int64_t          dec_frames, vcorrupt;
    int              deep_prime_packets;   /* §13: if >0, delay decode start until video_q banks this many packets (deep bursty-input cushion; single-input only) */
    /* 1.0.1-pre8 (a) GOP-coherent overflow: the decode side executes the head-GOP shed the
     * demux requests on video_q overflow (only the consumer can pop the queue head). */
    _Atomic int     *vq_shed_req;          /* -> Input.vq_shed_req (demux writes, decode consumes) */
    int64_t          shed_pkts;            /* cumulative head-shed packets */
    int64_t          shed_log_us;          /* [PTV-QSHED] decode-side log rate limit */
    int              heal_dropkf;          /* 1.0.1-pre8 (c): post-re-prime drop-until-IDR */
    int64_t          heal_arm_us;          /* (c): wall time the drop armed — 5s escape, the
                                            * Session-109 kf-gate rule (long-GOP sources must
                                            * not freeze waiting for an IDR that never comes) */
    /* 1.0.1-pre10 review fix (rr10 D1): the catch-up governor's rate currency is THIS
     * INPUT's real rate — each decode thread (mv slot or single) paces by its own input. */
    _Atomic int     *vin_pps;              /* -> DemuxArgs.vin_pps (demux-measured arrival pkts/s) */
    _Atomic int64_t *vin_pps_wall;         /* -> DemuxArgs.vin_pps_wall (wall time of the last publish —
                                            * 1.0.1-pre13: a stale value must never keep governing) */
    int              hb_slot;              /* 1.0.1-pre21: this input's slot — keys the g_hb_vdec_* heartbeats */
    int              in_pps_decl;          /* declared input rate, ceil(avg_frame_rate) — the
                                            * pre13 trust floor: a measured rate BELOW this is broken,
                                            * not a slow source (Newsmax2 live defect); 0 = unknown */
    /* 1.0.1-pre16: per-input published telemetry (→ Input fields). Governor gpps/decl/on
     * were single globals gated !hold, so on mv the governor RAN but said nothing (rr13
     * blindness); the globals remain as the input-0 alias for the single-input DIAG t= line
     * (log-string-identical). Self-shed wall/cnt mirror g_shed_wall/g_shed_cnt per input so
     * slot B's shed neither annotates slot A's AGLUE lines nor freezes slot A's corrector. */
    _Atomic int     *gov_gpps;             /* -> Input.gov_gpps */
    _Atomic int     *gov_decl;             /* -> Input.gov_decl */
    _Atomic int     *gov_on;               /* -> Input.gov_on */
    _Atomic int64_t *shed_wall;            /* -> Input.shed_wall */
    _Atomic int64_t *shed_cnt;             /* -> Input.shed_cnt */
    /* 1.0.1-pre30 #69 (item B): GOP-span estimator — key-to-key dts spacing EMA measured at
     * the pop site (published to g_vgop_est_us / g_vgop_key_wall; the resync fire-side
     * viability gate reads them). Measurement only; gated on g_resync so the kill switch
     * keeps the whole path inert. */
    int64_t          vgop_last_key_dts;    /* dts of the last KEY packet popped */
    int              vgop_key_seen;        /* first key seen (0 until then) */
} DecodeCtx;

/* Per-rung output side: pop this rung's frame_q on the house clock, stamp the
 * content-anchored PTS, encode, hand to this rung's mux_q. One per output. */
typedef struct VideoCtx {
    AVThreadMessageQueue *frame_q;   /* decode -> output  (AVFrame*)  */
    AVThreadMessageQueue *mux_q;     /* output -> mux     (AVPacket*) */
    DlvGate         *gate;           /* §7.5a delivery-alignment gate for this rung (NULL = disabled) */
    AVRational       out_tb;         /* time_base of frames at this rung's sink (or ist_tb) */
    int64_t         *h0;             /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    int64_t         *house_skew;     /* master publishes house-vs-content skew (us) here */
    RateEstimator   *est;            /* input 0's rate sensor (genlock fallback + cf/diag/stats reads) */
    HouseRateState  *hr;             /* per-house actuation state: master computes+publishes rho, all rungs apply it */
    VOutRing        *vring;          /* A/V probe: single-input video output ring (PTV_AVSYNC_PROBE) */
    AVCodecContext  *venc;
    AVStream        *ost;
    int64_t          tick_dur_us;
    AVRational       out_fps;        /* exact output rate — EXACTTICK content-index stamping (v0.9.9) */
    int              live;
    int              passthrough;    /* multiview: compositor already paced+stamped; encode 1:1 */
    int              is_master;      /* only the master rung prints stats/diag */
    /* shared decode counters + queue, for the master's diag line */
    AVThreadMessageQueue *dbg_video_q;
    int64_t         *dbg_dec_frames, *dbg_vcorrupt;
    int64_t         *dbg_vdrop, *dbg_pcorrupt;       /* single-input stats: demux video_q drops + corrupt-pkt count */
    int64_t         *dbg_disc_resid;                 /* 0.9.18.7: input-0 LAYERA hs-residue ledger (hsres= on the stats line) */
    /* counters */
    int64_t          framedrop, emitted, dup, pd;   /* pd = intentional cadence holds (telecine residence), split from dup (health alarm) */
    int64_t          decim;          /* v0.9.15.2: surplus frames decimated by content mapping (>house-rate source) */
    /* watchdog */
    int64_t          last_emit_us;
    volatile int     output_done;
    int              stalled;
} VideoCtx;

typedef struct AudioState {
    AVThreadMessageQueue *audio_q;
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    DlvGate              *gate[PTV_MAX_RUNG];    /* §7.5a: per-rung delivery gate (NULL = send direct) */
    AVStream             *ost[PTV_MAX_RUNG];     /* audio out stream in each muxer */
    int              n_out;
    AVCodecContext  *dec;
    AVCodecContext  *enc[PTV_MAX_RUNG];          /* one AAC encoder per rung (per-rung -b:a) */
    AVRational       ist_tb;
    SwrContext      *swr;                         /* no -af: plain resample to 48k stereo */
    SwrContext      *fg_swr;                       /* -af path: the (async) aresample filter's internal SwrContext — swr_get_delay() = faithful resampler-slip sensor (PTS metrics are blind to it) */
    AVFilterContext *fg_swr_flt;                   /* 1.0.1-pre11: the aresample AVFilterContext owning fg_swr — its own in/out links scope the slip probe to the RESAMPLER boundary (a buffering -af's hold must not read as slip) */
    int64_t          fg_swr_delay_max_ms;          /* running peak of swr_get_delay for observability */
    AVFilterGraph   *afg;                         /* -af present: abuffer -> chain -> abuffersink */
    AVFilterContext *afsrc, *afsink;
    int              use_fg;
    /* Audio input-format reconfig (ported from legacy 0003 fftools/ffmpeg_filter.c). The -af graph /
     * swr is built once for the source's initial params; if the source then changes channel layout /
     * rate / fmt mid-stream (e.g. stereo→mono at an ad-splice) abuffersrc rejects the frame
     * ("Changing audio frame properties not supported") and the audio path wedges. We track the
     * configured input params and rebuild for the new ones (output stays pinned to out_chl/48k →
     * encoder keeps getting continuous stereo). Hysteresis ignores transient/corrupt single-frame
     * flips at the splice boundary. */
    const char      *fg_af;                /* the -af chain string (program-lifetime) for rebuilds */
    int              fg_in_rate;           /* input params the active path is configured for */
    enum AVSampleFormat fg_in_fmt;
    AVChannelLayout  fg_in_chl;
    int              afmt_pending_rate;    /* hysteresis: candidate new params */
    enum AVSampleFormat afmt_pending_fmt;
    AVChannelLayout  afmt_pending_chl;
    int              afmt_stable;          /* consecutive frames seen at the candidate params */
    /* memcap F1: AFMT rebuild-storm circuit-breaker (flapping origins — CORELINK 44.1<->48kHz —
     * fire rebuilds continuously; each graph free/rebuild cycle churns ~2MB of allocator
     * fragmentation (measured, flat live-heap), unbounded ONLY in rebuild count = the HTTV
     * RSS-runaway class). Escalating minimum interval between rebuilds; see audio_feed. */
    int64_t          afmt_last_rb_us;      /* wall time of the last rebuild (0 = none yet) */
    int64_t          afmt_backoff_us;      /* current escalated min-interval (0 = no backoff) */
    int64_t          afmt_next_ok_us;      /* earliest wall time the next rebuild may run */
    int64_t          afmt_bk_dropped;      /* frames dropped while the breaker held */
    int64_t          afmt_bk_log_us;       /* rate limit for the breaker WARNING line */
    AVAudioFifo     *fifo;
    int              frame_size;
    int              out_rate;
    enum AVSampleFormat out_sfmt;
    AVChannelLayout  out_chl;
    int64_t         *h0;
    pthread_mutex_t *h0_lock;
    int64_t         *house_skew;    /* video's house-vs-content skew (us); -af audio rides it */
    int64_t         *house_lag_true;/* PTV_DIAG: uncapped true video lag for the lip-sync err (NULL single-input → use house_skew) */
    int              pts_set;
    int64_t          next_pts;
    int64_t          in_frames, out_frames;
    /* PTV_DIAG audio-side probe (temporary): identify per-track A/V offset on real feeds */
    int              dbg_k, dbg_in;
    int64_t          dbg_first_out, dbg_diag_last;
    int64_t          dbg_first_src, dbg_last_src;   /* source audio content span (us) for async-pad probe */
    AVFrame         *aq_pending[PTV_AQ_PREROLL];     /* pre-h0 audio buffer (preserve head until video anchors) */
    int              aq_npending;
    /* Multiview AUDIO-FOLLOW (Option A): apply the compositor's per-slot offset deterministically.
     * multiview=1 enables it (n_input>1). af_applied_us = the offset already applied; when the
     * compositor's published offset changes by more than a frame, the delta becomes a one-time
     * drop (advance audio: skip content) or pad (delay audio: insert silence), in output samples. */
    int              multiview;
    int64_t          af_applied_us;
    int64_t          af_drop, af_pad;                /* pending one-time correction, in out_rate samples */
    int              af_started;                     /* follow path: continuous output counter initialized */
    int64_t          af_next_pts;                    /* follow path: continuous output pts (out_rate samples) */
    int64_t          af_nudge_us;                    /* P1: smooth rate-limited PTS nudge (us), tracks residual+drift glitch-free */
    int64_t          af_last_out;                    /* B1: last emitted output pts (samples) — monotonic guard vs backward opts */
    int              af_out_set;                      /* B1: af_last_out valid */
    int64_t          avsync_stat_last;               /* [PTV-AVSYNC] status: last print time (us) */
    int64_t          async_stat_last, async_prev_bal; /* v0.9.2: aresample-work rate (g_async_ppm) state (primary track) */
    int64_t          af_acq_drop_us, af_acq_pad_us;  /* cumulative discrete acquire work (us dropped / padded) */
    /* A/V PLL redesign Phase A probe (PTV_AVSYNC_PROBE, read-only): real per-track A/V offset. */
    VOutRing        *vring;                           /* video output ring for this track's source input */
    int64_t          av_vlag_ema, av_alag_ema;        /* slow baselines of video_lag / audio_lag (us) */
    int              av_seed;                         /* baselines seeded */
    int64_t          av_probe_last;                   /* [PTV-AVSYNC2]: last print time (us) */
    int64_t          av_offset_us, av_vlag_us, av_alag_us;  /* latest MEASURED A/V offset (always computed, for the always-on [PTV-AVSYNC] line) */
    int              av_off_valid;                    /* a measurement has paired (else the status line prints offset=--) */
    /* A/V PLL redesign Phase B3 — closed-loop two-regime controller on the measured av_offset_us (g_avsync_pll). */
    int64_t          pll_ema;                         /* EMA of the measured offset (us) */
    int64_t          pll_dev;                          /* v0.6.22: slow EMA of |off−ema| = the leg's offset jitter; raises the acquire threshold above the noise floor */
    int              pll_seed;                        /* pll_ema seeded at the first valid measurement */
    int              pll_dbnc;                        /* stability-debounce: consecutive large-AND-flat readings */
    int64_t          pll_dbnc_ref;                    /* ema value when the debounce window started (flatness reference) */
    int              pll_refractory;                  /* frames remaining before acquire may re-arm (bumpless-credit backstop) */
    int              pll_acq_win;                     /* 1.0.1: consecutive completed above-threshold debounce windows (fire at 3; PTV_ACQ_INSTANT reverts) */
    int              pll_yielded;                     /* 1.0.1-pre21 #24: PLL actuators currently yielded to the steering corrector (transition-logged) */
    int64_t          pll_yield_bias_us;               /* #24 bumpless resume: alignment reference adopted at yield-resume — the PLL measures RELATIVE to the corrector's realized walk (never undoes it) */
    int              pll_bias_stale;                  /* rr21 A1: bias calibration voided by a label-baseline event — actuators suspended until the settled-window re-adoption */
    int              pll_bias_settle;                 /* rr21 A1: frames since the stale mark (adopt at ≥128 flat / 1280 hard cap) */
    int              pll_watch_seeded;                /* rr21 A1: ea/glue watch initialized */
    int64_t          pll_ea_watch;                    /* rr21 A1: last-seen demux label-edit ledger (g_rsx.ea_us[track]) */
    int64_t          pll_glue_watch;                  /* rr21 A1: last-seen AGLUE glue_off_us */
    int              acq_backoff;                     /* 1.0.1-pre18 #49: repeated-ACQUIRE backoff level (thr <<= level; +1 per
                                                       * acquire, −1 per acquire-free 60s) — an erase-class corruption phase
                                                       * re-anchors ±1 flat step per refractory forever (mv slot warble); the
                                                       * doubled threshold outgrows the step and the storm converges */
    int64_t          acq_last_wc;                     /* wall µs of the last ACQUIRE (backoff decay reference) */
    int64_t          tick_dur_us;                     /* 1.0.1: house video tick (us) — the vlag measurement quantum; floors the ACQUIRE threshold at 1.5 ticks */
    int              pll_acq_count;                   /* acquires fired this run (startup-k cap + gate assertion) */
    int              pll_drop, pll_pad;               /* pending one-shot acquire: frames to drop (advance) / pad (delay), on the B1 base */
    int64_t          pll_guard_fires;                 /* monotonic-guard activations (windup observability) */
    /* 1.0.1-pre3 — TRACK steers through the RESAMPLER, never labels. af_steer_us is the
     * accumulated integral trim, added to the pts of frames FED INTO the -af graph (the
     * single-input AVLOCK injection style): aresample=async realizes it as bounded content
     * stretch/squeeze (≤ async samples/s) while the OUTPUT label stream stays perfectly
     * dense — label re-stamping is a forbidden actuator (production 2026-07-13: pre2's
     * label-TRACK stretched output AAC pts spacing up to +158ms/min during integration
     * episodes; PTS-honoring players chased the drift with their own rate correctors =
     * audible warble). Written and read on the audio thread only (drain writes, feed reads). */
    int64_t          af_steer_us;                     /* cumulative TRACK trim injected into graph-input pts (us) */
    /* 1.0.1-pre3 [PTV-ACOMP] — app-layer proxy for swr hard-compensation triggers: a graph-
     * input pts step beyond ~min_hard_comp is realized by aresample as an instantaneous
     * sample insert/drop (click risk), invisibly. Track the expected next input pts and log
     * (rate-limited) when the actual deviates. */
    int64_t          acomp_exp_us;                    /* expected next graph-input pts (us); NOPTS until first frame */
    int64_t          acomp_cnt;                       /* input pts steps >25ms seen (hard-comp proxy count) */
    int64_t          acomp_log_us;                    /* log rate limit: last [PTV-ACOMP] line (wall) */
    /* v0.9.16.x lip-sync instrumentation (PTV_DIAG): where do audio label steps get eaten?
     * [PTV-ASTEP] fires on any in-pts (pre-graph) or sink-pts (post-graph) discontinuity;
     * [PTV-AFLOW] cumulative in/out sample counters — a graph CONTENT drop/pad shows as a
     * step in (in−out), a TIMELINE adoption shows sink-pts absorbing the step with (in−out)
     * unchanged. The pair discriminates what black-box runs could not. */
    int64_t          dbg_in_us, dbg_in_dur_us;        /* last in-pts (us) + expected frame span */
    int64_t          dbg_sink_us, dbg_sink_dur_us;    /* last sink-pts (us) + expected span */
    int64_t          dbg_in_samp, dbg_out_samp;       /* cumulative samples fed / drained */
    int64_t          dbg_flow_last_us;                /* [PTV-AFLOW] cadence */
    /* v0.9.16.3 [PTV-AGLUE] — symmetric audio label-step glue (see audio_feed). State is in the
     * RAW label domain (pre-glue, pre-AVLOCK) so LAYERA/house_skew actuation never looks like a
     * source step. glue_off_us accumulates erased relabels and is added to every graph-input pts. */
    int64_t          glue_off_us;                     /* cumulative relabel offset applied to input labels (us) */
    int64_t          glue_raw_last_us;                /* last RAW in-pts (us); NOPTS until first frame */
    int64_t          glue_raw_dur_us;                 /* its frame span (us) */
    int64_t          glue_wall_last_us;               /* monotonic wall time of the previous fed frame */
    int              glue_events;                     /* RELABEL verdicts this run */
    int64_t          glue_log_win_us;                 /* 0.9.18: verdict-log rate-limit window start (wall) */
    int              glue_log_win_n;                  /* verdict lines emitted this window */
    int              glue_supp_n;                     /* verdicts suppressed this window (still applied) */
    int64_t          glue_supp_net_us;                /* net label movement of the suppressed verdicts */
    /* 1.0.1-pre5 shared-flush expected-step handshake (D1): registration slot this track reads.
     * Demux thread writes (value first, deadline last, release); audio thread acquires the
     * deadline then reads the value, and clears the deadline on consume/expiry. NULL = unwired.
     * See PTV_PAIR_EXPECT_* in this header for semantics. */
    _Atomic int_least64_t *glue_exp_step;             /* registered expected label step (us) */
    _Atomic int_least64_t *glue_exp_dl;               /* wall-us deadline; 0 = no registration */
    /* v0.9.16.3 [PTV-ANCHOR] — birth-relationship observability (Zimbo-class startup offsets). */
    int              anchor_drop_pre;                 /* frames dropped because content preceded h0 */
    int              anchor_drop_ring;                /* pre-h0 ring overflow drops (oldest evicted) */
    /* 1.0.1 [PTV-ADEC] decode-death tolerance + watchdog: hard decode errors are dropped
     * (rate-limited WARNING) instead of killing the thread; if packets keep arriving but
     * nothing decodes for 45s wall the decoder is reopened from ist->codecpar (anchor/pts
     * state preserved — mid-run recovery, aresample absorbs the gap). PTV_NO_ADECWD gates
     * only the reopen; the error tolerance is unconditional. */
    /* 1.0.1-pre17 fix round (R1): OWNED COPIES of the source stream's decode params — the
     * decode-death watchdog reopen must not dereference an AVStream* into an AVFormatContext
     * the demux reopen-retry may have closed (the old ctx must be CLOSED, not leaked: its udp
     * socket would otherwise keep draining the port the fresh open re-binds). */
    AVCodecParameters *ist_par;                       /* copied codecpar (watchdog reopen source) */
    AVRational       ist_pkt_tb;                      /* copied packet time_base */
    int64_t          dec_errs;                        /* hard decode errors tolerated (dropped) */
    int              dec_reopens;                     /* watchdog decoder reopens this run */
    int64_t          decerr_win_us;                   /* [PTV-ADEC] log rate-limit window start (wall) */
    int              decerr_win_n;                    /* lines emitted this window */
    int              decerr_supp;                     /* errors suppressed this window (still counted) */
    int64_t          wd_frame_us;                     /* wall time of the last decoded frame (seeded at first packet) */
    int64_t          wd_pkts;                         /* packets received since the last decoded frame */
    /* 1.0.1-pre19.1 [PTV-ASTAMP] — un-stamped decoded-frame extrapolation (g_adts_split).
     * When the source is opened DURING a broken-AAC phase (Azorse 7.1: lavf's strict probe
     * decoder rejects every frame → "Could not find codec parameters" → sample_rate 0 → the
     * mpegts AAC parser splits the 6-frames-per-PES payloads but avformat cannot compute
     * per-frame durations, so frames 2..6 of each PES carry NO pts), the NOPTS drop rule in
     * audio_push discarded 5 of 6 frames = 21ms audio + 107ms hole per 128ms "ticking".
     * Fix: stamp a NOPTS decoded frame from the previous frame's ts + nb_samples. The carry
     * is INVALIDATED on any decode error (the garbage-tail class the drop rule was added
     * for) and on a decoder swap, so only contiguous clean decode is ever extrapolated. */
    int64_t          dec_ts_carry;                    /* next expected decoded-frame ts (ist_tb); AV_NOPTS_VALUE = no valid base */
    int              astamp_logged;                   /* one-shot engage log */
    /* 1.0.1-pre19 #46 [PTV-ACHOP] — post-storm stuck-chop escape (g_achop). A track in
     * SUSTAINED chop (decode-error or self-shed rate above the floor for g_achop_sust_min
     * minutes — the Azorse variant where the corruption never ends and the slot warbles
     * until a process restart) escapes via a FULL audio-path rebuild: decoder swap +
     * graph/swr teardown + the 0.9.17.1 AFMT impossible-seed, so the path re-forms from
     * the next confirmed-clean frames. Rate-limited (g_achop_relimit_us per track) so a
     * permanently-broken source cycles quietly instead of chopping. */
    int64_t          achop_win_wc;                    /* 10s sample window start (wall µs) */
    int64_t          achop_win_errs;                  /* dec_errs snapshot at window start */
    int64_t          achop_win_sheds;                 /* shed count snapshot at window start */
    int              achop_sust;                      /* consecutive windows at/above the chop floor */
    int64_t          achop_last_wc;                   /* wall µs of the last escape (rate limit) */
    int              achop_rebuilds;                  /* escapes this run */
    int64_t          shed_mark;                       /* 1.0.1-pre8 (d): g_shed_cnt snapshot while no shed window is
                                                       * open; " [self: N pkts shed]" = cnt_now − mark on AGLUE/ASTEP
                                                       * lines within 5s of a self-inflicted queue drop */
    /* 1.0.1-pre9 residual sensor (PASSIVE — see RsyncSense): audio-side content mapping. */
    int64_t          rs_ma_ema;                       /* EMA of m_a = out − (sink_src − inj) − slip (µs) */
    int              rs_ma_seed;                      /* EMA seeded at first sample */
    int64_t          rs_slip_us;                      /* latest net (dead-banded) resampler slip (DIAG) */
    int64_t          rs_log_last;                     /* [PTV-RSYNC] DIAG rate limit (wall µs) */
    /* 1.0.1-pre14 residual-sync corrector (see CorrState above; design doc §4/§8). The two
     * pointers wire this track's EVENT feeds (owned by its input slot): the disturbance epoch
     * (demux/compositor bump it) and the LAYERA buffer-active flag (plain int, demux-thread
     * written — read here ADVISORY only: a torn/late read merely delays an engage by one
     * evaluation, never corrupts state). afmt_rebuilds counts [PTV-AFMT] path rebuilds — a
     * rebuild is an event (§4.4). */
    CorrState        corr;
    _Atomic int_least64_t *corr_epoch;                /* -> Input.house_disturb (event feed) */
    int             *corr_layera_active;              /* -> Input.disc.active (g_layera only; NULL = off) */
    int              afmt_rebuilds;                   /* [PTV-AFMT] rebuild count (corrector event feed) */
    /* 1.0.1-pre16: per-input event feeds (this track's OWN input slot; NULL → global
     * fallback). Cross-slot independence: another slot's shed/governor must not annotate
     * this track's AGLUE lines or freeze this track's corrector quiet window. */
    _Atomic int64_t *shed_wall;                       /* -> Input.shed_wall (self-shed window) */
    _Atomic int64_t *shed_cnt;                        /* -> Input.shed_cnt */
    _Atomic int     *gov_on;                          /* -> Input.gov_on (catch-up governor engaged) */
    _Atomic int64_t *v_arrive_wc;                     /* 1.0.1-pre17 (§3 per-input liveness): -> Input.v_arrive_wc —
                                                       * THIS track's input's video-arrival watermark. NULL → the
                                                       * g_v_arrive_wc any-input fallback (identical on single input);
                                                       * on mv a dead sibling input no longer reads as "flowing". */
    /* 1.0.1-pre15 glue classification (#33; g_glueclass) — all owned by this track's audio
     * thread. E5 pad ledger (see the PTV_GLUE_PAD_* block): open GAP-pads awaiting a possible
     * return leg; 0 = consumed/empty slot. */
    int64_t          pad_led_us[PTV_GLUE_PAD_LED];    /* pad size (us, >0) per slot */
    int64_t          pad_led_wc[PTV_GLUE_PAD_LED];    /* wall us the pad was verdicted */
    int              pad_led_n;                       /* ring cursor (monotonic) */
    /* §2.4 realization tripwire: the last GAP/FLUSH-APPLY verdict's step, awaiting the
     * resampler's hard comp (instantaneous by design). Checked against the pre11 slip probe
     * ~2s after arming; a parked slip means the pad/drop was NOT realized — synthesize the
     * remainder at the swr boundary (inject_silence/drop_output) instead of shipping on faith. */
    int64_t          pend_comp_us;                    /* outstanding verdict step (us); 0 = none */
    int64_t          pend_comp_wc;                    /* wall us the verdict armed */
    int64_t          tw_synth_cnt;                    /* tripwire syntheses fired (observability) */
    /* §3 NBS starvation fill: track is living on synthesized silence while the demux discards
     * an undecodable source phase (opt-in PTV_NBS_FILL=1). nbs_feeding marks frames audio_feed
     * receives FROM the fill itself (they must not consume the resume classification). */
    int              nbs_fill_active;                 /* fill phase open */
    int              nbs_feeding;                     /* inside nbs_fill_quantum's feed loop */
    int              nbs_fills;                       /* quanta synthesized this run (observability) */
    int64_t          nbs_last_wall_us;                /* rr15 F9: wall of the previous quantum (elapsed base) */
    int64_t          nbs_carry_us;                    /* rr15 F9: sub-frame remainder carried between quanta */
    int64_t          glue_cad_us;                     /* rr15 R2: EMA of nonzero fed-frame wall gaps (PES-burst
                                                       * period) — the E3 cadence baseline for the pad-ledger gate */
    /* 1.0.1-pre20 REBUILD RE-ANCHOR (g_rebuild_reanchor; PTV_NO_REBUILD_REANCHOR=1 reverts).
     * An [PTV-AFMT] rebuild used to inherit the track's audio base from carried state
     * (glue_off_us polluted by broken-phase erasures; the swr-fallback next_pts counter
     * frozen across the outage) — the transition duration leaked in as a ~1s residual the
     * corrector then walked for ~10min (Azorse live 2026-07-20 10:54: R +1021ms, 599s walk).
     * The source is A/V-synced at the change, so the residual is OURS: at rebuild completion
     * the base is re-derived BIRTH-EQUIVALENT from the current house mapping (content − h0,
     * glue_off 0 — the [PTV-ANCHOR] birth formula, factored), carried corr is RETIRED, the
     * m_a EMA re-seeds (the pre17 baseline-redefinition rule), and for the first 10s a
     * one-shot bounded base step retires any residual the settled sensor still reads. */
    int64_t          reanch_wc;                       /* rebuild-completion wall µs; 0 = no open window */
    int              reanch_frames;                   /* emitted frames since the rebuild / last step (settle) */
    int64_t          reanch_step_us;                  /* cumulative (c) steps this window (±5s budget —
                                                       * fix round: MULTI-step, a seam's video ledger edit
                                                       * can land after the first fire) */
    /* pre20 fix round (F1, reviewer blocker): the retirement is LABEL-NEUTRAL (corr folds into
     * glue_off — inj sum unchanged, door labels continuous), so no backward step can arise there.
     * A NEGATIVE (c) residual step still moves door labels backward; this guard makes the mux
     * invariant absolute: post-step frames whose output label does not EXCEED the last emitted
     * one are DROPPED (the birth opts<0 rule, mirrored) until labels catch up naturally —
     * bounded by the step size, inside the already-audible rebuild. The mux must NEVER see a
     * backward audio DTS. Non-follow path only (mv audio-follow is clamped by af_last_out);
     * disarms ONLY on catch-up (a wall expiry would re-open the backward window). */
    int              reanch_mono;                     /* guard armed */
    int64_t          reanch_mono_dropped;             /* frames dropped by the guard (logged at release) */
    int64_t          out_last_pts;                    /* last emitted output label (out_rate samples) */
    int              out_pts_set;                     /* out_last_pts valid */
    int64_t          flush_relab_seen_wc;             /* 1.0.1-pre18 #50 (E5 net): last consumed/retired
                                                       * g_flush_relab_wc value for this track (young
                                                       * unmatched publishes stay pending and re-scan) */
    int64_t          reopen_seen_wc;                  /* 1.0.1-pre20 rider (a): last consumed g_reopen_wc
                                                       * value for this track's input (ASTAMP carry inval) */
    /* 1.0.1-pre23 #60/#61 bounded convergence (g_convcap — see the PTV_CONV_ESC block).
     * All owned by this track's audio thread. */
    int64_t          conv_bl_us;                      /* in-flight accepted-convergence order magnitude (us); 0 = none */
    int64_t          conv_bl_wc;                      /* wall us the order was accepted */
    int64_t          conv_bl_dl;                      /* in-flight deadline (wall us): accept + 2×order + 60s —
                                                       * injected silence plays out ~1× realtime on a live-paced
                                                       * channel; past this the order is considered played out.
                                                       * The TTL can only MISS an escape (a slower drain reads as
                                                       * fresh), never fold a healthy independent step early —
                                                       * a recurring ladder re-trips B on its own re-measurement. */
    int64_t          conv_esc_wc[PTV_CONV_ESC];       /* fold-escape wall times (rolling recurrence window) */
    int              conv_esc_n;                      /* escape ring cursor (monotonic) */
    int64_t          seam_park_until;                 /* wall us; nonzero+future = SEAM-PARK: every above-cap
                                                       * step folds immediately (entry/expiry logged) */
    int64_t          conv_total_us;                   /* rr23: cumulative folded label motion (us) — the net
                                                       * label divergence the folds refused; published to
                                                       * g_conv_pub for the stats conv= token */
    /* 1.0.1-pre24 #63 WALL-EVIDENCE (g_wallev; PTV_NO_WALLEV=1 reverts the ACTION — the
     * measurement below stays on to feed the re-anchor corroboration). Door side. */
    int64_t          u_door_us;                       /* Σ unevidenced REAL-content deletions this door made
                                                       * (µs, ≥0): folds/erases of steps whose own wall
                                                       * evidence (wall_gap ≥ step/2 + cadence) said the
                                                       * content was genuinely absent — the #63 leak term,
                                                       * measured. Feeds the re-anchor corroboration. */
    int64_t          wev_out_us;                      /* outstanding wall-evidenced pad backlog (µs) — the
                                                       * conv-fold-exemption OOM backstop ledger; drains at
                                                       * ~1× realtime (decayed by wall elapsed) */
    int64_t          wev_out_wc;                      /* wall µs of the last wev_out update (decay ref) */
    int64_t          wev_fb_log_wc;                   /* rr24 F4: door-side bursty-fallback log rate limit */
    int64_t          md_warn_wc;                      /* pre26 D3: [PTV-MUXDIAG] emission-point rate limit */
    int64_t          wev_cons_last_us;                /* conservation diag: last logged A−V unevidenced
                                                       * deletion imbalance (µs) */
    int64_t          wev_cons_log_wc;                 /* conservation diag log rate limit (10s) */
    /* 1.0.1-pre24 #63 CORROBORATED RECOVERY RE-ANCHOR (g_recanchor; PTV_NO_RECANCHOR=1).
     * One-shot, health-gated recovery for a channel left with a stable content↔label offset
     * after input recovers (owner mandate 2026-07-24: "perfect lip sync once input is OK
     * again"). Engages ONLY when R is large+stable through a quiet settle window AND the
     * unevidenced-deletion ledgers CORROBORATE it (aseam-class pinned-R false positives have
     * no deletion imbalance and must never engage). Actuation = the [PTV-ANCHOR] base-step
     * algebra (glue_off += step retires +R), slewed in bounded chunks. All owned by this
     * track's audio thread. */
    int64_t          ra_r0;                           /* R stability reference (µs) */
    int64_t          ra_r0_wc;                        /* wall µs R last moved beyond ±100ms of ra_r0 */
    int64_t          ra_ev_wc;                        /* wall µs of the last observed event (own snapshots) */
    int64_t          ra_cooldown_until;               /* wall µs before which no engage (0 = none) */
    int              ra_active;                       /* mid-walk (chunked slew in progress) */
    int64_t          ra_budget_us;                    /* remaining authority this engagement (|R0|×1.2) */
    int64_t          ra_step_wc;                      /* wall µs of the last applied chunk */
    int64_t          ra_ua_snap, ra_uv_snap;          /* unevidenced-deletion ledger snapshots (corroborate
                                                       * only NEW deletions since the last completion) */
    int              ra_test_steps;                   /* TEST ONLY (rr24 F2 gate): steps this engagement */
    int              ra_test_aborted;                 /* TEST ONLY: forced abort already fired this run */
    int64_t          ra_applied_us;                   /* rr24 F2: Σ signed steps applied since the last
                                                       * COMPLETE — each step retired exactly +step of R,
                                                       * so after a mid-walk ABORT the corroboration must
                                                       * predict the REMAINDER: rp = ΔU − Δcorr − Σapplied
                                                       * (without this, |R−rp| = Σapplied wedged re-engage
                                                       * forever → permanently half-recovered channel) */
    int64_t          ra_corr_snap;                    /* corr_us snapshot (trim already applied against the
                                                       * same bake must not double-count in R_pred) */
    int64_t          ra_log_wc;                       /* refusal-line rate limit */
    /* own event-edge snapshots (the corrector's CorrState snapshots are CONSUMED by its own
     * edge detector — the re-anchor must never share them) */
    int64_t          ra_epoch_snap, ra_ev_led_snap, ra_ea_led_snap, ra_bank_snap, ra_hs_snap;
    int              ra_glue_snap, ra_acq_snap, ra_afmt_snap, ra_reopen_snap, ra_esc_snap;
    int              ra_seeded;                       /* snapshots initialized */
    /* 1.0.1-pre29 #69 RESYNC (g_resync; PTV_RESYNC=1 enables, default OFF). The second
     * engage path inside ptv_recanchor: a confirmed-timer HARD RESET for the >PTV_RESYNC_MS
     * dead zone (owner-verified 4/4: large stable R is a real on-air desync; today it is
     * refused by the corroboration gate and disarms the corrector as implausible). Shares
     * the walk/actuation state above (ra_cooldown_until, ra_applied_us, the step recipe)
     * so mutual exclusion with the corroborated path is structural. */
    int64_t          rsn_wc;                          /* wall µs the >threshold timer opened (0 = closed) */
    int64_t          rsn_r0;                          /* R at timer open (|r0|>2s selects the big confirm) */
    int              rsn_active;                      /* chunked audio-early (R>0) walk in progress */
    int64_t          rsn_budget_us;                   /* remaining chunk authority (R0 + R0/5) */
    int64_t          rsn_step_wc;                     /* wall µs of the last applied chunk */
    int64_t          rsn_cooldown_until;              /* wall µs before which no fire — set ONLY by a
                                                       * mid-walk abort (settle) or the armed breaker's
                                                       * backoff; NEVER by a routine COMPLETE (owner:
                                                       * seams are cheaper than desync-time) */
    int              rsn_breaker;                     /* circuit breaker armed (pathological fire rate) */
    int64_t          rsn_backoff_us;                  /* armed-breaker backoff (120s ×2 → 600s cap) */
    /* 1.0.1-pre30 (item C): SLIDING breaker window — ring of the last N fire wall-times.
     * Replaces the pre29 tumbling first-fire-anchored window (N fires straddling a window
     * boundary never armed it). Arm rule: ts[newest] − ts[newest−(N−1)] ≤ BREAKER_WIN. */
#define PTV_RSN_RING 16
    int64_t          rsn_fire_ring[PTV_RSN_RING];     /* fire wall µs, newest at (head−1) mod ring */
    int              rsn_fire_head;                   /* next write slot */
    int              rsn_fire_cnt;                    /* valid entries (≤ PTV_RSN_RING) */
    int64_t          rsn_last_over_wc;                /* wall µs |R| last exceeded the band (quiet disarm) */
    int64_t          rsn_count;                       /* total resync fires (stats rsn= token) */
    /* 1.0.1-pre30 (item A): post-seam sensor hold. Live evidence (first two production
     * fires, 2026-07-29): the reading ~2-4ms after a seam is garbage (i24: chunk +2000 on
     * R=+2081 → expected +81, instant read −1035 → false ABORT; settled +125 ~1-2min later,
     * decay ~12ms/s). rsn_step_wc doubles as the last-seam stamp; R-based walk decisions
     * (COMPLETE / band-crossing / next seam) are suspended for PTV_RESYNC_SEAM_HOLD_S after
     * it, then need 2 consecutive ≥5s-spaced SAME-SIDE samples moving <50ms (a still-decaying
     * sensor never reads stable, so the walk waits it out instead of aborting falsely). */
    int              rsn_rd_side;                     /* last post-hold sample's side (in-band/cross/over) */
    int64_t          rsn_rd_R;                        /* last post-hold sample's R (stability check) */
    int64_t          rsn_rd_wc;                       /* wall µs of that sample (≥5s spacing) */
    int              rsn_rd_n;                        /* consecutive same-side stable samples */
    /* 1.0.1-pre30 (item B): audio-EARLY default actuator = video IDR-skip (jump cut) via the
     * decode-thread executor (g_vskip_* handshake); PTV_RESYNC_SILENCE=1 reverts to pre29
     * silence-pad chunks, which also remain the in-walk fallback when no IDR is viable. */
    int              rsn_vskip;                       /* this walk actuates via video IDR-skip */
    int64_t          rsn_vskip_deadline;              /* wall µs: executor-unresponsive fallback */
    /* 1.0.1-pre30.1: hs-step SELF-ATTRIBUTION for vskip walks. Live (i24/cor-3 3x
     * 2026-07-30, reproduced): the skip consumes buffered video; the next delivery gap
     * then starves frame_q → a dup run raises house_skew one tick per dup while the
     * audio thread idles the same gap → its next evaluation sees ONE >1.25-tick hs jump
     * → the walk's event-edge check aborted its own walk ("new event: hs-step") ~2s
     * after every seam. Attribution is directional + budgeted, NOT a blanket window: a
     * POSITIVE hs-only step is absorbed while cumulative growth since arming stays
     * within Σ(posted skip spans)+slack — the most starvation our own skip can cause.
     * Foreign disturbances still abort: any other edge type, any negative step, or
     * growth beyond the budget (e.g. a genuine input stall's dup run busts it). */
    int              rsn_hs_absorb;                   /* armed: vskip span posted this walk */
    int64_t          rsn_hs0;                         /* house_skew at arming (growth base) */
    int64_t          rsn_hs_budget_us;                /* max self-attributable hs growth */
    int64_t          rsn_hs_req_us;                   /* last posted span (refund on REFUSED/recall;
                                                       * shrink by req−achieved at DONE/ESCAPE) */
    int64_t          rsn_walk_wc;                     /* rr30 (T1): walk ENGAGE stamp — with item A,
                                                       * a sensor that never corroborates (2 stable
                                                       * samples) would otherwise pin rsn_active
                                                       * forever (and the corrector defers via
                                                       * resync_owns): the liveness ceiling aborts
                                                       * a walk with no seam/complete progress for
                                                       * PTV_RESYNC_WALK_CEIL_S */
} AudioState;

/* ---- demux + mux ---- */

#define PTV_MAX_PASS 16
typedef struct PassStream {
    int        input;                 /* source input index (multiview); 0 single-input */
    int        in_index;              /* input stream index being copied 1:1 (within that input) */
    AVStream  *ost[PTV_MAX_RUNG];     /* output stream in each muxer (fan-out) */
    AVRational in_tb;                 /* input/output time_base (copy: identical) */
    int64_t    last_dts;              /* last emitted dts (monotonic guard; NOPTS until first) */
    int        gated;                 /* §7.5a: dense copied AUDIO (AC-3/MP2) → route via the delivery
                                       * gate; sparse subs/data/SCTE-35 bypass (their wire-arrival lead
                                       * is a feature) */
} PassStream;

/* ---- legacy-0004 TS-discontinuity buffer (g_layera / PTV_LAYERA, default OFF) ----
 * Faithful port of patches/legacy/0004-ts-discontinuity-buffering. At a content
 * glue point the source interleaves OLD-timeline and NEW-timeline packets across
 * the straddle. The default ptvencoder path (g_layera==0) self-rebases each dense
 * stream's wrap_off at its own crossing and lets mis-glued OLD packets pass through.
 * When g_layera is on we instead: detect the jump, BUFFER the dense V/A packets while
 * both timelines are in flight, classify each held packet OLD/NEW, KEEP NEW / DISCARD
 * OLD, compute one AUDIO-derived offset, apply it to every kept packet, and release
 * in order. Sparse SUBTITLE/DATA (DVB-sub, SCTE-35) are NEVER buffered — they keep
 * the existing prog_off path in demux_unwrap. All timestamps here are AV_TIME_BASE
 * (us); the held packet already has demux_unwrap's 33-bit wrap correction applied. */
#define PTV_DISC_CAPACITY        512   /* pre17 fix round (R3a): 256→512 — the bounded
                                        * hold-until-drain may legally hold up to the 5s
                                        * pairing window (Fashion ~77pkt/s ⇒ ~385 pkts;
                                        * 256 would ENOSPC-force-flush at ~3.3s). Slots
                                        * are pointers; RSS = held compressed packets,
                                        * bounded by window × input bitrate. */
#define PTV_DISC_THRESHOLD_US    (1 * AV_TIME_BASE)   /* 1s   jump threshold */
#define PTV_DISC_TIMEOUT_US      (500 * 1000)         /* 500ms forced-flush timeout */
#define PTV_DISC_TOL_US          (100 * 1000)         /* 100ms timeline classification tolerance */
/* 1.0.1-pre4 shared-flush pairing window (wall us): dense flushes closer together than this
 * belong to ONE source event, and every dense stream in the event shares the event's
 * VIDEO-derived offset (see the decision tree in ptv_disc_flush). Sized for the live evidence
 * (Curiosity 2026-07-13: video and audio crossed 0.6s apart -> two partial flushes ~0.6s apart
 * after the 500ms buffer timeout) with wide margin, yet far below the spacing of independent
 * jumps (few/hour worst case). THE ACTUAL FALSE-PAIRING GUARANTEE (pre5 — the original "benign
 * by construction" claim was falsified by review: an inherited offset costs an aresample
 * convergence PROPORTIONAL to the inherited-vs-own disagreement, i.e. UNBOUNDED, and the
 * fx-dbl re-inherit destroyed ~17s of audio): each stream can inherit/apply the event's offset
 * AT MOST ONCE PER FLUSH-CYCLE CROSSING SET (pair_has is checked per flush at 3a: a flush
 * inherits if ANY crossing stream is unapplied and then applies to ALL crossing streams — an
 * already-applied stream that crosses in the SAME 500ms cycle as a late leg can re-apply;
 * review-2 F2, rare multi-audio coincidence, registered + aresample-converged when it happens)
 * and the window CLOSES as soon as every flowing dense audio stream has applied it — so an
 * independent audio wobble AFTER the event completes can never re-inherit a stale offset. What remains inherently ambiguous is a
 * video-only crossing followed by an INDEPENDENT first audio crossing inside the window: that
 * is indistinguishable from the genuine Curiosity ordering by construction, and the inherit
 * then costs an audible aresample convergence of the disagreement. The window/eps/pair_has
 * checks bound WHEN a false pairing can happen (first application per stream per window),
 * not its size. */
#define PTV_PAIR_WINDOW_US       (5 * AV_TIME_BASE)
/* Shared-flush equality band: a V-vs-A offset disagreement at or below this is flush BOOKKEEPING
 * (duration-estimate overhang ~1 frame, trailing-OLD discard holes ~100-400ms of interleave), NOT
 * a source A-vs-V jump difference — for those the production-proven audio-preferred butt-joint is
 * kept BYTE-IDENTICAL (the TruBLU symmetric-rewind mandate: equal deltas ⇒ per-stream-identical
 * behavior, log-line equality). The invariant holds in the band regardless — ONE offset is applied
 * to all dense streams either way; the band only decides WHICH content machinery absorbs the
 * bookkeeping residual (audio butt-joint, as today). Above the band the disagreement is a real
 * asymmetric event (live cases: 1.0s, 30.8s): VIDEO defines the timeline and the difference goes
 * to the audio content path. Half the LAYERA jump threshold, so a genuine >1s-delta pairing whose
 * streams disagree by more than this always engages. */
#define PTV_PAIR_EPS_US          (500 * 1000)
/* 1.0.1-pre5 demux->audio expected-step handshake (the D1 fix): when a shared flush routes an
 * A-vs-V mismatch to the audio content path, the demux thread REGISTERS the step it just put
 * into that track's label stream (value + wall deadline, atomics — demux writes, audio thread
 * reads). AGLUE consumes a matching arriving step as a REAL alignment step: it must be APPLIED
 * (aresample converges content to the new labels), never relabel-ERASED — the erase is exactly
 * what re-broke the invariant for backward mismatches in (-1000ms,-500ms) (mirror-signed
 * events: video jumping further forward than audio). Plain source backward steps carry no
 * registration and keep the 0.9.16.4 relabel-erase rule.
 *   TTL: must cover the audio pipeline's WORST-CASE demux->AGLUE residency, not the typical
 *   1-2s packet-queue latency — under a deep prime / auto-bank the audio_q legally holds the
 *   full bank ceiling (g_cushion_max_ms, 12s default, operator-raisable) plus the delivery-gate
 *   hold, and the post-outage resume carrying the discontinuity is EXACTLY the joint-event case
 *   this handshake exists for (review-2 F1: a 10s TTL expired mid-bank and fell back to the
 *   relabel-erase, re-breaking the invariant). Sized like the delivery-maxq formula
 *   (ptvencoder.c: delivery_cap + cushion_max + margin): 30s. The value-match collision window
 *   this buys is still negligible (a collision needs an unrelated backward step of near-equal
 *   >500ms magnitude on the same track inside the TTL). A stale registration expires silently.
 *   Match window is ASYMMETRIC around the registered value: flush-borne steps (same-cycle /
 *   inherit) arrive EXACT to within duration-estimate noise (measured -32..0ms), but a
 *   retro-corrected step rides the first normal-path packet after the flush and MERGES with the
 *   flush's own trailing-OLD discard hole, which is always FORWARD (measured +456ms on the
 *   ordering fixture) — hence [-LO, +HI] = [-250ms, +500ms] of the registered value. */
#define PTV_PAIR_EXPECT_TTL_US   (30 * AV_TIME_BASE)
#define PTV_PAIR_EXPECT_LO_US    (250 * 1000)
#define PTV_PAIR_EXPECT_HI_US    (500 * 1000)
/* 1.0.1-pre18 #50 (g_glueveto) — the one-remedy match keys. A gap verdict and a LAYERA
 * cycle/flush describe the SAME source event when (a) they land within the WIN of each
 * other (live: verdict 42ms after the arm, flush 520ms after — the arm-vs-verdict and
 * flush-vs-verdict gaps are both bounded by the 500ms disc-buffer timeout + interleave),
 * (b) the jump magnitudes agree within EPS (live: +2.421s jump vs +2.413s gap = 8ms), and
 * (c) the jump's from/to positions lie in the SAME timeline domain as the gap's bracketing
 * labels (DOM ≈ program interleave depth) — magnitude alone must not match a wrapped/
 * corrupt-dts jump living in a different domain (the Fashion class). Two genuinely
 * independent same-program events could still pass (a)+(c); the (b) conjunction inside a
 * 500ms window is the discriminator there (independent similar-magnitude events within
 * 500ms have no observed precedent; a wrong veto degrades to the pad-only posture, a
 * wrong non-veto is today's double remedy). */
#define PTV_GVETO_WIN_US   (500 * 1000)
#define PTV_GVETO_EPS_US   (150 * 1000)
#define PTV_GVETO_DOM_US   (2 * AV_TIME_BASE)


typedef struct PtvDiscPacket {
    AVPacket *pkt;
    int       stream_idx;
    int64_t   raw_dts;     /* DTS before applied_offset (us); already 33-bit-unwrapped */
    int       timeline;    /* 0=old, 1=new, 2=continuing (own timeline, no offset), -1=unknown */
    int       own_cont;    /* 1 = at arrival this packet was CONTINUOUS with its stream's own
                            * last_dts_us (|delta| <= PTV_DISC_THRESHOLD_US) — 1.0.1-pre7: a
                            * stream that never crossed this cycle keeps such packets on its
                            * OWN timeline instead of borrowing the partner's bases */
} PtvDiscPacket;

typedef struct PtvDiscStreamState {
    int64_t cumulative_ts_offset;  /* per-stream offset (us), diagnostic; persists across cycles */
    int64_t last_sent_dts;         /* end of last packet sent for THIS stream (us); NOPTS until first */
    int64_t last_dts_us;           /* last post-unwrap DTS seen (us) — jump-detection ref; persists */
    int64_t old_timeline_base;     /* this stream's last DTS before jump (per-cycle) */
    int64_t new_timeline_base;     /* this stream's first DTS in new timeline (per-cycle) */
    int     has_old_base;
    int     has_new_base;
    /* 1.0.1-pre4 shared flush (semantics tightened pre5 — the D2 re-inherit fix): what offset
     * this (audio) stream applied within the current pairing window, and HOW:
     *   pair_has  = 1 once the stream applied the event's FINAL offset (video-defined, or
     *               band-equal to it). Consulted at 3a: a stream that already applied can NOT
     *               inherit again — its next crossing is a new independent event (fx-dbl: the
     *               -2s wobble 2s after a mirror event re-inherited -14.98s and destroyed ~17s
     *               of audio). Also drives the completion close of the window.
     *   pair_prov = 1 while the stream holds a PROVISIONAL own offset (audio crossed before
     *               video defined the timeline) — the only state the video-crossing flush (2d)
     *               retro-corrects. Cleared (-> pair_has) by the retro-correct.
     * Persist across cycles; cleared when the pairing window closes. */
    int64_t pair_applied_us;
    int     pair_has;
    int     pair_prov;
    int     aanch;                 /* 1.0.1-pre22 audio-anchor (a-anchor): this transcoded audio
                                    * stream was seeded a PROVISIONAL zero-offset leg by a
                                    * lone-VIDEO-jump flush (pair_prov=1, pair_applied_us=0 —
                                    * "audio didn't jump"). NOT retro-corrected by the
                                    * video-crossing flush's 2d (that flush is the SAME one that
                                    * seeded it); instead FIRED at pairing-window expiry
                                    * (ptv_aanch_fire) — the absence of the audio leg is only
                                    * provable then. Cleared by ptv_disc_pair_reset and by any
                                    * real audio crossing in the window (the cancel). */
    /* 1.0.1-pre15 E4 label health H (#33 §2.1; g_glueclass): demux-side windowed EMA of
     * Δdts/Δwall over the trailing ~30s, EXCLUDING buffered/flush windows and per-packet
     * jumps (those are events, not rate evidence). Healthy ≈ 1.0 ±5% (Q10: 1024 = 1.0);
     * a label-flood source (Azorse: labels stride ~6x content) reads H >> 1. Consulted by
     * the §2.3 refuse gate before a flush routes a mismatch to the content path — evidence
     * QUALITY, not magnitude (PATRIOT's 30.8s was real; Azorse's 31.078s was noise). */
    int64_t h_prev_wall;           /* wall us of this stream's previous dense packet */
    int64_t h_dts_acc, h_wall_acc; /* open sub-window accumulators (us) */
    int64_t h_ema_q10;             /* windowed-rate EMA, Q10 (0 until first window) */
    int     h_wins;                /* closed windows folded into the EMA */
    int64_t h_wild_wc;             /* wall us of the last WILD window (|r-1| > 50%) — the
                                    * flood-recency signal for the <3-packet base rule */
    /* 1.0.1-pre24 #63 (g_wallev): the wall-absence-evidenced portion W of this stream's
     * jump (µs, ∈ [0, J]; 0 = flowing/unmeasurable), captured at base-record time from the
     * jump packet's pre-update delivery wall gap. The flush's butt-joint preserves this
     * much of the label hole (erase only J−W, the flowing/relabel part) so each stream's
     * content path pads exactly its REAL hole — the storm cross-stream conservation fix.
     * Per-cycle: cleared with the bases. */
    int64_t jump_wallev_us;
} PtvDiscStreamState;

typedef struct PtvDiscBuf {
    PtvDiscPacket     **packets;
    int                 nb_packets;
    int                 capacity;
    int                 active;             /* 1 while buffering across a straddle */
    int                 flushing;           /* 1 while flushing (suppress re-detection) */
    int64_t             buffer_start_time;  /* wall clock (us) when buffering started */
    int                 jump_detected;      /* 1 once any stream recorded bases this cycle */
    uint8_t            *stream_transitioned;/* per-stream: 1 once it reached the new timeline */
    int                 nb_streams;
    PtvDiscStreamState *stream_state;
    int64_t             applied_offset;     /* single offset (us, audio-derived) applied to ALL kept packets */
    /* v0.9.13 [PTV-GLUE] running stats — the LAYERA-retirement decision data: vid_err = how much
     * source A/V mis-mux each glue carried (what LAYERA corrects and the plain absorber would
     * leak into audio). |err| persistently ~0 => the simpler absorber suffices. */
    int64_t             glue_cnt;           /* glues with BOTH media measured */
    int64_t             glue_partial;       /* flushes where only one media type crossed in the window */
    int64_t             err_abs_sum_us;     /* Σ|vid_err| */
    int64_t             err_abs_max_us;     /* max|vid_err| */
    int64_t             err_gt100_cnt;      /* glues with |vid_err| > 100ms */
    /* 1.0.1-pre4 shared flush: pairing-window state (persists ACROSS flush cycles — that is the
     * point: the live failure was two partial flushes 0.6s apart, each erasing its own stream). */
    int64_t             pair_start_us;      /* wall us of the event's first dense flush; 0 = no open event */
    int                 pair_vid_defined;   /* 1 once VIDEO's crossing defined this event's timeline */
    int64_t             pair_vid_off_us;    /* the video-defined shared offset (us) */
    int                 pair_anchored;      /* rr-d1 R3: this event's video "crossing" is a lone-audio
                                             * ANCHOR (offset 0, no real video leg) — keep the pair
                                             * state through the end-of-flush close until the window
                                             * expires, so a staggered matching VIDEO leg finds the
                                             * audio pair state (suppresses the #47-C sibling hold;
                                             * it then flushes immediately as 2a/new event) */
    int                 aanch_pend;         /* 1.0.1-pre22 audio-anchor (a-anchor, the rr-d1 R3 mirror):
                                             * a lone-VIDEO forward jump seeded provisional zero-offset
                                             * audio leg(s) (stream_state[].aanch) and now awaits the
                                             * pairing-window verdict — the end-of-flush close must not
                                             * pair_reset the event away while pending. Fired (audio
                                             * re-based onto the video timeline + expected step
                                             * registered) at window EXPIRY on the quiet path; cleared
                                             * by ptv_disc_pair_reset or by a real audio crossing. */
    int                 cycle_trigger;      /* 1.0.1-pre7: stream whose detect ARMED this buffer cycle
                                             * (-1 = none); scopes the continuing-stream keep (see
                                             * ptv_disc_flush) to transcoded-triggered cycles */
    int                 partial_ext;        /* 1.0.1-pre16 #47-C / pre17 fix round (R3a): partial-hold
                                             * extensions granted this cycle (was one-shot; now repeats
                                             * while the sibling leg is missing, capped by the pairing
                                             * window from cycle open + a capacity guard) */
    int64_t             cycle_open_us;      /* pre17 R3a: wall µs this buffer cycle armed (extension cap base) */
    /* 1.0.1-pre18 #50 (g_glueveto) inverse-guard evidence: the last flush that applied a
     * nonzero offset to AUDIO labels — a gap verdict arriving just after it (matching
     * magnitude + window) is the SAME source event already remedied, so the verdict is
     * suppressed instead of double-applying (pad on top of the flush's relabel). */
    int64_t             fl_wall;            /* wall µs of that flush (0 = never) */
    int64_t             fl_delta_us;        /* the trigger stream's jump delta at that flush (µs) */
} PtvDiscBuf;

typedef struct DemuxArgs {
    AVFormatContext      *ifmt;
    AVThreadMessageQueue *video_q;
    AVThreadMessageQueue *audio_q[PTV_MAX_AUDIO]; /* one per transcoded audio track */
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    DlvGate              *gate[PTV_MAX_RUNG];    /* §7.5a: per-rung delivery gate (NULL = send direct) */
    int                   n_out;
    int                   vstream;
    int                   astream[PTV_MAX_AUDIO]; /* input stream feeding each audio_q */
    int                   n_audio;
    /* 1.0.1-pre5 shared-flush expected-step handshake (D1): per transcoded track, the slot the
     * flush registers into when it routes an A-vs-V mismatch to that track's content path
     * (storage lives in Input; AudioState holds the read side). Indexed like audio_q/astream. */
    _Atomic int_least64_t *aglue_exp_step[PTV_MAX_AUDIO];
    _Atomic int_least64_t *aglue_exp_dl[PTV_MAX_AUDIO];
    int                   aglobal[PTV_MAX_AUDIO];     /* 1.0.1-pre15: GLOBAL track index (dbg_k) of
                                                       * local track j — keys the per-track published
                                                       * atomics (pad ledger, decode watermark, fill) */
    /* 1.0.1-pre15 §3 (NBS) manifestation (c): the corrupt-discard site counted per TRACK
     * (video already had vcorrupt). Unconditional observability — even under
     * PTV_NO_GLUECLASS. [PTV-ADISC] log window is per track. */
    int64_t               acorrupt[PTV_MAX_AUDIO];    /* corrupt-flagged audio pkts discarded */
    int64_t               adisc_win_us[PTV_MAX_AUDIO];/* [PTV-ADISC] 10s log window start */
    int64_t               adisc_win_n[PTV_MAX_AUDIO]; /* discards in the open window */
    int64_t               nbs_last_fill_us[PTV_MAX_AUDIO]; /* last FILL sentinel sent (quantum pace) */
    int64_t               glue_refuse_cnt;            /* §2.3 F2 refuse ledger (per input) */
    /* 1.0.1-pre16 #47-A: per-stream snapshot of d->vpkt at this stream's last packet — at a
     * gap verdict, (d->vpkt − gap_vsnap[stream]) = video packets that arrived DURING the gap.
     * The R1 freshness guard measured video's AGE at verdict time, which cannot distinguish
     * "video flowed through the gap" from "video resumed 2ms earlier" (TWN incident). */
    int64_t              *gap_vsnap;
    /* 1.0.1-pre16 #47-C: last KNOWN jump per media type (0=video 1=audio): magnitude + wall.
     * Stamped at LAYERA jump detects and audio GAP verdicts; consulted before a PARTIAL flush
     * releases one-sided (the sibling's matching jump = evidence the event is two-legged). */
    int64_t               sib_jump_us[2];
    int64_t               sib_jump_wall[2];
    int                   drop;          /* non-blocking + drop on full (network input) */
    PassStream           *pass;          /* copy-passthrough: extra audio, subs, data */
    int                   n_pass;
    int64_t              *h0;             /* house origin (us); copy ts rebased onto it */
    pthread_mutex_t      *h0_lock;
    int64_t              *house_skew;     /* video's house-vs-content skew (us); copy rides it */
    _Atomic int_least64_t *disturb_epoch; /* B3: bump this input's disturbance epoch when the discont absorber fires */
    RateEstimator        *est;            /* this input's rate sensor (0.9.18 R4; demux thread is the sole feeder) */
    int64_t              *wrap_off;       /* per input stream: cumulative 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;      /* per input stream: last RAW ts seen (wrap detection) */
    int64_t              *wrap_wall_last; /* per input stream: wall-clock (us) of this stream's last packet — gap-vs-splice discriminator */
    /* 1.0.1-pre24 #63 wall-evidence (g_wallev): per-stream delivery-cadence EMA (µs between
     * good dense packets, quiet gaps only) + the CURRENT packet's pre-update wall gap —
     * demux_unwrap updates wrap_wall_last before the LAYERA loop runs, so jump detection
     * needs the gap preserved separately. W = clamp(pkt_wall_gap − cadence, 0, J). */
    int64_t              *wall_cad_us;
    int64_t              *pkt_wall_gap_us;
    int64_t               wallev_fb_log_us; /* fallback (bank/bursty) advisory rate limit */
    int64_t              *edit_us;        /* 1.0.1-pre9 sensor: per-stream label-EDIT ledger (µs) — the
                                           * non-wrap share of wrap_off (splice absorbs, LAYERA persists,
                                           * retro-corrections); demux thread only, published via g_rsx */
    int                   rsync_slot;     /* pre16: this input's slot index — keys g_rsx.ev_us[] (video
                                           * ledger publish; audio publishes by aglobal[j]). ALL inputs
                                           * publish now (was rsync_pub, single-input input 0 only) */
    _Atomic int64_t      *shed_wall;      /* pre16: -> Input.shed_wall (per-input self-shed stamp) */
    _Atomic int64_t      *shed_cnt;       /* pre16: -> Input.shed_cnt */
    _Atomic int64_t      *v_arrive_wc;    /* pre17: -> Input.v_arrive_wc (per-input arrival watermark) */
    /* 1.0.1-pre17 fix round (R1): live mv inputs RE-OPEN on any av_read_frame error instead of
     * EOF-latching the slot forever (rw_timeout expiry = the >=30min-outage class; owner
     * mandate: everything auto-resumes after ANY outage). The demux thread owns the swap:
     * close the dead ctx (frees its udp socket so the re-bind actually receives), open url
     * with a COPY of the original format opts, validate the stream layout (same count +
     * consumed-stream codec types), publish the new ctx to *ifmt_home for teardown. Bounded
     * backoff, retries forever; downstream sees the outage exactly like a <rw_timeout gap
     * (slate -> recovery machinery, V10-proven). Single-input keeps EOF = channel end. */
    const char           *url;            /* input url (reopen target) */
    AVDictionary         *reopen_opts;    /* copy of the ORIGINAL open format_opts */
    int                   reopen;         /* live && multiview && net input */
    AVFormatContext     **ifmt_home;      /* -> Input.ifmt (teardown's close target follows the swap) */
    int64_t               reopen_cnt;     /* successful reopens (log) */
    PtvDiscBuf           *disc;           /* legacy-0004 buffer-classify-discard (g_layera only; NULL otherwise) */
    int64_t               video_fwd_us;   /* wall-clock (us) of the last VIDEO forward-discontinuity crossing (whole-program-splice indicator) */
    int64_t               prog_off;       /* P2 (§7.1): program-level discontinuity offset (90kHz, detected on the
                                           * DENSE video reference) applied to SPARSE copied streams (sub/data/SCTE)
                                           * which don't self-rebase — keeps them aligned to video across an ad-break
                                           * PTS jump instead of orphaned/vanishing. V/A keep per-stream self-rebase.
                                           * §5.A.2 (g_progoff_av): dense V/A self-rebase by the SHARED first-crosser amount. */
    int64_t               splice_adj;       /* §5.A.2: the first-crosser's discontinuity adj for the current splice */
    int64_t               splice_adj_us;    /* §5.A.2: wall-clock when splice_adj was set (debounce; 0 = never) */
    int                   drop_until_kf;  /* P2 2b: armed on a video discontinuity → drop video until the next IDR */
    int64_t               kf_arm_us;      /* P2 2b: wall time the drop was armed (first-arm-only escape deadline) */
    int64_t               kf_arm_vdrop;   /* DIAG: vdrop count when DUKF armed (→ per-event drop count at resume) */
    /* 1.0.1-pre8 (a) GOP-coherent video overflow (QSHED): on a full video_q the demux
     * requests a head-GOP flush from the decoder and TAIL-drops the arriving stream to the
     * next IDR, so the queue never carries a headless GOP (the #32 wedge fragmenter). */
    _Atomic int          *vq_shed_req;    /* -> Input.vq_shed_req (decode consumes) */
    int                   vq_tail_drop;   /* in tail drop-until-IDR mode */
    int64_t               qshed_tail_arm_us; /* wall time the tail-drop armed — Session-109 time
                                            * escape (g_dukf_escape_us): never wait forever for
                                            * an IDR that never comes (intra-refresh sources) */
    int64_t               qshed_tail_n;   /* pkts dropped in the current tail episode */
    int64_t               qshed_tail_tot; /* cumulative tail-dropped pkts (log accounting) */
    int64_t               qshed_log_us;   /* [PTV-QSHED] demux-side log rate limit */
    /* 1.0.1-pre10 (h) DEGRADED MODE (opt-in PTV_DEGRADED=1): >=3min of persistent QSHED
     * full-cycles -> DEMAND-DRIVEN GOP admission at the live edge (admit an arriving GOP
     * only when video_q <= ~1s — the queue depth IS the throughput measurement; retained
     * latency self-scales with the deficit instead of accumulating, which is what kept
     * audio alive: A/V ride the same delay and the door buffers hold ~15s, not 60s);
     * entry flushes the stale backlog (selfheal re-prime); release after 60s of continuous
     * decode headroom (frame_q un-starved with vq shallow). */
    int                   deg_active;       /* currently degrading admission */
    int                   deg_admit;        /* current GOP is admitted (decided at its IDR) */
    int64_t               deg_train_us;     /* start of the current full-cycle train (0 = none) */
    int64_t               deg_last_full_us; /* last full-cycle arm (train continuity, 30s gap) */
    int64_t               deg_head_ok_us;   /* headroom continuously since (0 = not in headroom) */
    int64_t               deg_dropped;      /* total video pkts dropped by degraded admission */
    int64_t               deg_log_us;       /* [PTV-DEGRADED] status rate limit */
    /* 1.0.1-pre10 review fix (rr10 D1): per-input MEASURED video arrival rate — the catch-up
     * governor's rate currency. A 4s window over demux_dispatch video-packet arrivals; any
     * >1s arrival gap restarts the window (an outage boundary must never dilute the rate and
     * under-cap the governed drain). Published as pkts/s (0 until the first clean window
     * completes). 1.0.1-pre13: the governor now TRUSTS a publish only when it is ≥ the
     * declared rate AND fresh (<30s old, vin_pps_wall) — otherwise it fails OPEN. */
    int64_t               vin_win_us;       /* measurement window start (0 = unstarted) */
    int64_t               vin_last_us;      /* previous video-pkt arrival (gap guard) */
    int                   vin_win_cnt;      /* video pkts since the window-start packet */
    _Atomic int           vin_pps;          /* published measured arrival rate (pkts/s) */
    _Atomic int64_t       vin_pps_wall;     /* wall time of that publish (0 = never) — staleness gate */
    int64_t               vpkt, apkt, ppkt, vdrop, adrop, pdrop;
    int64_t               disc_resid_us;   /* 0.9.18.7 hs-residue ledger (REPORTING ONLY, never read by control):
                                            * Σ(−applied_offset) over LAYERA glue erases that shifted the VIDEO
                                            * label stream. Every erase shifts all subsequent content labels by
                                            * applied_offset, i.e. shifts the hs/sk reading by −applied_offset vs
                                            * the raw source labels — a jump-to-live erase (applied_offset<0)
                                            * parks the stall's dup-ratcheted skew in hs permanently. hs growing
                                            * IN STEP with this ledger = erased-discontinuity bookkeeping, not
                                            * retained buffer latency; hs growing with this flat = real hold. */
    /* v0.9.12 [PTV-BURSTY] advisor: detect HLS-burst-over-SRT delivery (video arrives in clumps
     * separated by multi-second stalls) and log a once-per-minute WARNING with the SIZED env
     * recipe (deep §13 packet prime). Detection = >=3 completed arrival gaps >=1.5s within 60s
     * (a periodic burst pattern — a single outage is 1 gap and never trips it). Silent when
     * PTV_PREROLL_MS already covers the observed gap (correctly configured channel). */
    int64_t               by_last_v_wall;   /* wall time of the previous video packet */
    int64_t               by_win_start;     /* rolling 60s window start */
    int64_t               by_max_gap;       /* max arrival gap in the window (us) */
    int                   by_gap_cnt;       /* completed gaps >=1.5s in the window */
    int                   autobank;         /* v0.9.14: runtime bank escalation armed for this input (single-input live) */
    int64_t               by_bank_last_q;   /* v0.9.14: wall time of the last QUALIFYING stall (decay reference) */
                                            /* (by_bank_advise_us moved to CushionRt.bank_advise_us — 0.9.18 M3) */
    int64_t               vcorrupt;       /* video packets flagged AV_PKT_FLAG_CORRUPT (discarded if g_discardcorrupt) */
} DemuxArgs;

/* One source input. Single-input uses inputs[0]; multiview uses 1/2/4. Each has
 * its own demuxer + video decoder + clock anchor (h0) + wrap state; multiview
 * decode stages into `hold` for the compositor instead of filtering inline. */
typedef struct Input {
    const char           *url;
    AVFormatContext      *ifmt;
    int                   vstream;
    const AVCodec        *vdecoder;
    AVCodecContext       *vdec;
    AVStream             *vist;
    AVRational            ist_tb;
    AVThreadMessageQueue *video_q;           /* demux -> decode */
    int64_t               h0;                /* this input's A/V anchor (us); decode sets it */
    pthread_mutex_t       h0_lock;
    int64_t               house_skew;        /* compositor publishes; this input's audio/copy ride it */
    int64_t               house_lag_true;     /* PTV_DIAG: compositor publishes the UNCAPPED signed video lag
                                               * (output−content) for the lip-sync probe; = house_skew unless the
                                               * 250ms cap / non-decreasing floor clips it (multiview only) */
    VOutRing              vring;             /* A/V probe: this input's (displayed content → out_v) ring */
    _Atomic int_least64_t house_disturb;     /* B3: per-input disturbance epoch — bumped on slate-return (compositor) AND
                                              * discont absorb (demux); TWO writer threads → atomic. The PLL's mid-run
                                              * acquire arms only when this advances (never on bare vlag noise). */
    RateEstimator         est;               /* 0.9.18 R4: this input's demux-side rate sensor (FLL genlock +
                                              * coarse clock-follow); non-zero fields seeded in the init loop */
    VideoHold             hold;              /* multiview: latest decoded frame for the compositor */
    int64_t              *wrap_off;          /* per stream: 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;         /* per stream: last RAW ts (wrap detection) */
    int64_t              *wrap_wall_last;    /* per stream: wall-clock (us) of last packet (gap-vs-splice discriminator) */
    int64_t              *edit_us;           /* pre9 sensor: per-stream label-edit ledger storage (µs) */
    int64_t              *gap_vsnap;         /* 1.0.1-pre16 #47-A: per-stream d->vpkt snapshot storage */
    int64_t              *wall_cad_us;       /* 1.0.1-pre24 #63: per-stream delivery-cadence EMA storage */
    int64_t              *pkt_wall_gap_us;   /* 1.0.1-pre24 #63: per-stream current-pkt wall-gap storage */
    PtvDiscBuf            disc;              /* legacy-0004 buffer-classify-discard state (used only when g_layera) */
    /* 1.0.1-pre5 shared-flush expected-step handshake storage (D1) — one slot per GLOBAL
     * transcoded track index (only the tracks sourced from this input are wired). Demux thread
     * writes, that track's audio thread reads (the house_skew/disturb_epoch publish idiom,
     * hardened with atomics because value+deadline are a pair). Zero-init = no registration. */
    _Atomic int_least64_t aglue_exp_step[PTV_MAX_AUDIO];
    _Atomic int_least64_t aglue_exp_dl[PTV_MAX_AUDIO];
    _Atomic int           vq_shed_req;       /* 1.0.1-pre8 (a): per-input head-GOP shed request */
    /* 1.0.1-pre16 per-input published telemetry (the globals g_shed_wall/g_shed_cnt and
     * g_gov_* stay stamped as the any-input aggregate / input-0 single-input alias; these
     * carry the SLOT-scoped copy for the per-track readers — AGLUE self-shed notes, the
     * corrector's quiet window and governor feed, the mv DIAG per-slot gpps/gov segment). */
    _Atomic int64_t       shed_wall;         /* wall µs of this input's last self-inflicted queue drop */
    _Atomic int64_t       shed_cnt;          /* cumulative self-shed pkts on this input */
    _Atomic int64_t       v_arrive_wc;       /* 1.0.1-pre17: wall µs of THIS input's last video pkt at its demux
                                              * (per-input twin of g_v_arrive_wc — the corrector's §3 per-input
                                              * liveness; the global stays as the any-input aggregate) */
    _Atomic int           gov_gpps;          /* catch-up governor: last measured arrival pps it saw */
    _Atomic int           gov_decl;          /* declared input pps (trust floor) */
    _Atomic int           gov_on;            /* 1 = governing this input's decode */
    DecodeCtx             dc;
    DemuxArgs             da;
    pthread_t             th_demux, th_decode;
    int                   started_demux, started_decode;
    int                   open_ret;          /* parallel-open result */
} Input;

/* Multiview compositor = the video house clock. Samples each input's hold at
 * each tick, feeds the N buffersrcs, pulls each rung's composited frame, and
 * publishes per-input house_skew. (Single-input never uses this — decode feeds
 * the graph inline and the per-rung output_thread is the house clock.) */
typedef struct CompositorCtx {
    Input                *inputs;
    int                   n_input;
    AVFilterGraph        *fg;
    AVFilterContext      *fsrc[PTV_MAX_INPUT];
    int                   n_rung;
    AVFilterContext      *fsink[PTV_MAX_RUNG];
    AVThreadMessageQueue *frame_q[PTV_MAX_RUNG];
    int64_t               framedrop[PTV_MAX_RUNG];
    int64_t               tick_dur_us;       /* 1/out_fps in us (INTEGER — pacing fallback only since v0.9.12) */
    AVRational            out_fps;           /* exact output rate — MV-EXACTTICK measurement axis (v0.9.12) */
    int                   live;
    int64_t               slate_after_us;    /* stale hold -> black cell (0 = never) */
    /* stats (compositor is the cadence owner in multiview) */
    int64_t               emitted, dup;
    struct DlvGate       *gate0;              /* rung-0 delivery gate for the stats readout (NULL = ungated) */
} CompositorCtx;


/* ==== cross-file globals (defined in ptvencoder.c unless noted) ==== */
extern int     g_diag;
extern int     g_avlock;
extern int     g_reanchor;
extern int     g_mv_clamp;
extern int     g_mv_residence;
extern int     g_mv_birthtrim;
extern int     g_aglue_ceil;
extern int     g_discont;
extern int     g_gapdiscrim;
extern int     g_adecwd;
extern int     g_adts_split;                              /* 1.0.1-pre19.1 broken-phase probe/stamp fix (PTV_NO_ADTS_SPLIT=1 reverts to pre19) */
extern int     g_tolerant_dec;                            /* 1.0.1-pre19.1 kill for the #38 tolerant AAC decode (PTV_NO_TOLERANT_DEC=1 = strict pre-#38) */
extern int     g_achop;                                   /* 1.0.1-pre19 #46 stuck-chop escape (see ptvencoder.c) */
extern int     g_achop_errs_min, g_achop_sheds_min, g_achop_sust_min;
extern int64_t g_achop_relimit_us;
extern int     g_anchor_headfill;
extern int64_t g_wrap_guard_us;
extern int     g_aglue_ms;
extern int     g_convcap;                /* 1.0.1-pre23 bounded convergence (A+B+C); PTV_NO_CONVCAP=1 reverts */
extern int64_t g_conv_cap_us;            /* A: single-order admission cap (PTV_CONV_CAP_S, default 60s) */
extern int64_t g_seam_park_us;           /* C: recurrence window AND park duration (3600s;
                                          * PTV_SEAM_PARK_S TEST-ONLY override for the G4 gate) */
extern int64_t g_novideo_exit_us;        /* 1.0.1-pre23 startup sanity: packets flowing but no video frame
                                          * decoded for this long since start → FATAL exit
                                          * (PTV_NOVIDEO_EXIT_S, default 300s; 0 disables) */
extern int     g_wallev;                 /* 1.0.1-pre24 #63 wall-evidence split (PTV_NO_WALLEV=1 reverts
                                          * the action; provenance measurement stays on) */
extern int     g_recanchor;              /* 1.0.1-pre24 #63 corroborated recovery re-anchor
                                          * (PTV_NO_RECANCHOR=1 disables) */
extern int64_t g_recanchor_settle_us;    /* quiet+stable window before an engage (PTV_RECANCHOR_SETTLE_S, 300s) */
extern int64_t g_recanchor_cooldown_us;  /* re-engage holdoff after a completed re-anchor
                                          * (PTV_RECANCHOR_COOLDOWN_S, 1800s) */
extern int     g_recanchor_test_abort_n; /* TEST ONLY (rr24 F2 gate): force one mid-walk abort
                                          * after N steps, no event injected (0 = off, default) */
extern int     g_resync;                 /* 1.0.1-pre29 #69 RESYNC hard-reset path for the
                                          * >PTV_RESYNC_MS dead zone (PTV_RESYNC=1 enables, default OFF) */
extern int64_t g_resync_ms_us;           /* resync band threshold (PTV_RESYNC_MS, 350ms) */
extern int64_t g_resync_ok_us;           /* OK band — timer closes / walk completes (PTV_RESYNC_OK_MS, 150ms) */
extern int64_t g_resync_confirm_us;      /* confirm timer before a fire (PTV_RESYNC_CONFIRM_S, 120s) */
extern int64_t g_resync_confirm_big_us;  /* confirm when |R0| > 2s (PTV_RESYNC_CONFIRM_BIG_S, 60s) */
extern int     g_resync_breaker_n;       /* circuit breaker: fires within the window that ARM
                                          * (PTV_RESYNC_BREAKER_N, 4) — NO routine cooldown otherwise */
extern int64_t g_resync_breaker_win_us;  /* breaker fire-count window (PTV_RESYNC_BREAKER_WIN_S, 900s) */
extern int64_t g_resync_quiet_us;        /* below-band span that DISARMS the breaker + clears the
                                          * history (PTV_RESYNC_QUIET_S, 1800s) */
extern int64_t g_resync_chunk_us;        /* audio-early chunk size (PTV_RESYNC_CHUNK_MS, 2000ms) */
extern int64_t g_resync_chunk_gap_us;    /* gap between chunks (PTV_RESYNC_CHUNK_GAP_S, 5s) */
/* ---- 1.0.1-pre30 #69 refinements ---- */
extern int64_t g_resync_seam_hold_us;    /* item A: post-seam sensor hold before any R-based walk
                                          * decision (PTV_RESYNC_SEAM_HOLD_S, 20s; 0 disables).
                                          * The effective inter-seam gap is max(chunk_gap, hold). */
extern int     g_resync_vskip;           /* item B: audio-EARLY default actuator = video IDR-skip
                                          * (PTV_RESYNC_SILENCE=1 reverts to pre29 silence chunks) */
extern int64_t g_resync_idr_wait_us;     /* item B: executor bound waiting for a usable IDR
                                          * (PTV_RESYNC_IDR_WAIT_S, 5s ≈ 2× typical GOP) */
extern int64_t g_resync_vskip_tol_us;    /* item B: whole-GOP overshoot tolerance — skip the
                                          * largest whole-GOP total ≤ R + tol
                                          * (PTV_RESYNC_VSKIP_TOL_MS, 250ms) */
extern int64_t g_resync_walk_ceil_us;    /* rr30 (T1): walk liveness ceiling — abort a walk
                                          * whose sensor never corroborates, measured from the
                                          * last seam (or ENGAGE) (PTV_RESYNC_WALK_CEIL_S,
                                          * 600s; 0 disables) */
/* item B cross-thread state (single-input only; the audio thread requests, the decode thread
 * executes at the video_q pop site — upstream of the rung split, so every rung skips the same
 * content coherently; the output threads read the label mapping in content_index()). */
enum {                                   /* g_vskip_state */
    PTV_VSKIP_IDLE = 0,
    PTV_VSKIP_PENDING,                   /* request posted, executor working */
    PTV_VSKIP_DONE,                      /* clean IDR stop — achieved span in g_vskip_done_us */
    PTV_VSKIP_ESCAPE,                    /* deadline escape (mid-GOP resume) — partial span */
    PTV_VSKIP_REFUSED                    /* nothing skippable inside the horizon */
};
extern _Atomic int64_t g_vskip_req_us;   /* requested target span µs (audio posts, decode consumes) */
extern _Atomic int     g_vskip_state;    /* PTV_VSKIP_* (audio arms PENDING, decode completes) */
extern _Atomic int64_t g_vskip_done_us;  /* achieved span µs (valid at DONE/ESCAPE) */
extern _Atomic int     g_vskip_done_gops;/* whole GOPs (key packets) dropped */
/* label mapping: the skip is label-neutral — content advances, output labels continue
 * monotonically. content_index() subtracts off_total for src ≥ from (post-skip content) and
 * off_before for src < from (frames of earlier epochs still in flight through frame_q).
 * Write order (decode thread): off_before=old_total → from=new → off_total=new_total. */
extern _Atomic int64_t g_vskip_off_total;  /* cumulative skipped span (µs) for src ≥ from */
extern _Atomic int64_t g_vskip_off_before; /* cumulative span before the latest skip */
extern _Atomic int64_t g_vskip_from_us;    /* latest skip boundary on the source-time axis (µs) */
extern _Atomic int     g_vskip_epoch;      /* bumped per applied skip → master rung re-seeds m_v EMA
                                            * at the first post-boundary emit */
extern _Atomic int64_t g_vgop_est_us;      /* decode-measured key-to-key span EMA (0 = unknown) */
extern _Atomic int64_t g_vgop_key_wall;    /* wall µs a KEY packet last passed the pop site */
extern int64_t g_rscorr_slew_fast;       /* pre29 adaptive corrector slew above 150ms |R|, µs/s
                                          * (PTV_RSCORR_SLEW_FAST, 5000; 0 = always the base clamp);
                                          * active regardless of PTV_RESYNC (0.5%% rate < pitch JND) */
extern int     g_muxguard;               /* 1.0.1-pre26 survive-first backward-dts mux backstop
                                          * (PTV_NO_MUXGUARD=1 disables) */
extern int64_t g_muxtest_back_at_us;     /* TEST ONLY (pre26 gates): one injected backward audio
                                          * dts at the mux feed, t µs after mux start (0 = off) */
extern int64_t g_muxtest_back_ms;        /* TEST ONLY: injected backward magnitude (ms) */
extern int     g_muxdiag;                /* pre26 D3: gated emission-point backward-label instrumentation */
extern int     g_prog_off;
extern int     g_progoff_av;
extern int     g_layera;
extern int     g_layera_fullskip;
extern int     g_shared_flush;
extern int     g_vanchor;                /* d1-fix lone-audio video-anchor; PTV_NO_VANCHOR=1 reverts */
extern int     g_aanchor;                /* 1.0.1-pre22 lone-video audio-anchor (the D1 mirror); PTV_NO_AANCHOR=1 reverts */
extern int     g_drop_until_kf;
extern int     g_audio_follow;
extern int     g_pll_yield;              /* 1.0.1-pre21 #24 PLL/corrector arbitration; PTV_NO_PLL_YIELD=1 reverts */

/* 1.0.1-pre21 thread-position HEARTBEATS (live Azorse wedge 2026-07-21 10:43:58: the vdec
 * thread froze in a futex with video_q FULL and frame_q EMPTY — nothing in the log said
 * WHERE; this is how the next occurrence self-identifies). The vdec thread stamps its step
 * position + wall before every potentially-blocking stage; the master output thread (the
 * -stats printer) stamps its loop points; the demux stamps a per-input arrival watermark
 * (the "input flows" evidence). The existing per-rung watchdog_thread (a DIFFERENT thread,
 * 500ms cadence, always-on) checks the stamps: vdec stamp >5s old while the demux stamp is
 * fresh → one rate-limited [PTV-STALL] line naming the position. Stamps are relaxed atomics
 * — reporting only, no control consumer. */
enum {
    PTV_HB_NONE = 0,
    PTV_HB_VDEC_QRECV,       /* polling video_q for the next packet */
    PTV_HB_VDEC_BANK,        /* deep-prime bank wait / governed catch-up sleep */
    PTV_HB_VDEC_SENDPKT,     /* avcodec_send_packet */
    PTV_HB_VDEC_RECVFRAME,   /* avcodec_receive_frame */
    PTV_HB_VDEC_HWUP,        /* filter-graph push/pull (hwupload lives in the graph) */
    PTV_HB_VDEC_FQSEND,      /* frame_q / mv hold.q push */
    PTV_HB_OUT_LOOP,         /* output/stats thread loop top (tick pacing) */
    PTV_HB_OUT_ENC,          /* encode_push (encoder + delivery gate) */
    PTV_HB_OUT_STATS         /* stats/diag print section */
};
extern _Atomic int     g_hb_vdec_pos[PTV_MAX_INPUT];
extern _Atomic int64_t g_hb_vdec_wall[PTV_MAX_INPUT];
extern _Atomic int64_t g_hb_vdec_dec[PTV_MAX_INPUT];    /* dec_frames snapshot at last stamp */
extern _Atomic int64_t g_hb_demux_wall[PTV_MAX_INPUT];  /* input-flow evidence (demux read loop) */
extern _Atomic int     g_hb_vq[PTV_MAX_INPUT];          /* video_q depth (demux-published) */
extern _Atomic int     g_hb_out_pos;
extern _Atomic int64_t g_hb_out_wall;
const char *ptv_hb_name(int pos);
#define PTV_HB_VDEC(slot, p) do { if ((slot) >= 0 && (slot) < PTV_MAX_INPUT) { \
    atomic_store_explicit(&g_hb_vdec_pos[slot], (p), memory_order_relaxed); \
    atomic_store_explicit(&g_hb_vdec_wall[slot], av_gettime_relative(), memory_order_relaxed); } } while (0)
#define PTV_HB_OUT(p) do { \
    atomic_store_explicit(&g_hb_out_pos, (p), memory_order_relaxed); \
    atomic_store_explicit(&g_hb_out_wall, av_gettime_relative(), memory_order_relaxed); } while (0)
extern int     g_h0_reanchor;
extern int     g_reanchor2_instant;
extern int     g_h0_at_display;
extern int     g_avsync_probe;
extern int     g_af_pll;
extern int     g_af_anchor;
extern int     g_avsync_pll;
extern int     g_acq_instant;
extern int     g_pll_trackup;
extern int64_t g_pll_testnoise_us;
extern int     g_pll_testnoise_frames;   /* TEST-ONLY (#49 gate): square-wave half-period, frames (PTV_PLL_TESTNOISE_P) */
extern int64_t g_cushion_max_ms;
extern int64_t g_bank_decay_us;
extern _Atomic int     g_bank_pkts;
extern _Atomic int64_t g_bank_us;
extern _Atomic int     g_vq_elems;
extern _Atomic int     g_fq_hw;
extern int     g_exacttick;
extern int     g_mv_exacttick;
extern int     g_decimate;
extern int     g_pulldown;
extern int     g_cad_disarm;
extern int     g_frameq_cap;
extern int     g_preroll_ms;
extern int     g_discardcorrupt;
extern _Atomic int64_t g_muxed;
extern int     g_stats;
extern _Atomic int64_t g_ch_vsrc, g_ch_asrc, g_ch_vout_src, g_ch_aout_src;
extern _Atomic int64_t g_ch_vsrc_raw, g_ch_asrc_raw;
extern int     g_genlock;
extern int     g_genlock_ok;
extern int     g_clockfollow;
extern int     g_wucr;
extern int     g_reprime;
extern int     g_adapt_cushion;
extern CushionPlan g_cp;                 /* defined in ptvencoder_gate.c */
extern _Atomic int     g_frameq_depth;
extern int     g_genlock_guard;
extern _Atomic int64_t g_async_ppm;
extern int64_t g_stats_period_us;
extern int     g_slow;
/* 1.0.1-pre8 #32 wedge fixes (defined in ptvencoder.c) */
extern int     g_qshed;                  /* (a) GOP-coherent video_q overflow (PTV_NO_QSHED reverts) */
extern int     g_ratchrel;               /* (b) ratchet release on starvation contradiction (PTV_NO_RATCHREL) */
extern int     g_selfheal;               /* (c) self-heal re-prime backstop (PTV_NO_SELFHEAL) */
extern _Atomic int     g_selfheal_req;   /* (c) master output thread -> decode thread */
extern _Atomic int64_t g_v_arrive_wc;    /* wall us of the last video pkt at the demux (input-flowing signal) */
extern _Atomic int64_t g_shed_wall;      /* (d) wall us of the last self-inflicted queue drop (ANY input —
                                          * pre16: per-track readers use Input.shed_wall; this aggregate
                                          * stays for the catch-up governor + unwired fallback) */
extern _Atomic int64_t g_shed_cnt;       /* (d) cumulative self-shed pkts (video head+tail, audio drop-oldest) */
extern int     g_nvenc_serialize;        /* defined in ptvencoder_clock.c */
extern CushionRt g_curt;                 /* defined in ptvencoder_gate.c */
/* 1.0.1-pre10 birth-armed churn fixes (defined in ptvencoder.c) */
extern int     g_cushrel;                /* (e) cushion-tier release on starvation contradiction (PTV_NO_CUSHREL) */
extern int     g_catchgov;               /* (f) governed deficit-recovery decode, 1.25x realtime (PTV_NO_CATCHGOV) */
/* 1.0.1-pre13 catch-up governor observability (single-input decode publishes; DIAG t= line reads).
 * The Newsmax2 wedge was undiagnosable from logs — dec ≪ fps with vq pinned had NO gpps trace.
 * pre16: these globals are the INPUT-0 SINGLE-INPUT ALIAS (log strings unchanged); the per-input
 * truth lives in Input.gov_* (mv DIAG per-slot segment + the corrector's per-track feed). */
extern _Atomic int     g_gov_gpps;       /* last measured arrival pps the governor saw (0 = none) */
extern _Atomic int     g_gov_decl;       /* declared input pps (trust floor) */
extern _Atomic int     g_gov_on;         /* 1 = governing (sleeps active), 0 = disengaged/fail-open */
extern _Atomic int64_t g_gov_slip;       /* cumulative oversleep strikes (actuator overshooting) */
extern int     g_jit_milli;              /* (g) per-PID shed/heal phase jitter x1000 (800..1200; 1000 = PTV_NO_PHASEJIT) */
extern int     g_degraded;               /* (h) opt-in sustained-deficit every-Kth-GOP admission (PTV_DEGRADED=1) */
/* 1.0.1-pre9 residual sensor (defined in ptvencoder.c) */
extern int        g_rsync_sense;         /* PASSIVE sensor on (PTV_RSYNC_SENSE=0 disables) */
extern RsyncSense g_rsx;                 /* published sensor state (single-input, input 0) */
/* 1.0.1-pre14 residual-sync corrector (defined in ptvencoder.c; design doc §6/§7) */
extern int     g_rsync_corr;             /* DEFAULT ON; PTV_NO_RSYNC_CORR=1 kills */
extern int64_t g_rscorr_engage_us;       /* engage dead band (80ms default; PTV_RSCORR_ENGAGE_MS) */
extern int64_t g_rscorr_dwell_us;        /* stable-dwell length (300s; PTV_RSCORR_DWELL_S — TEST ONLY) */
extern int64_t g_rscorr_quiet_us;        /* trailing event-free window (180s; PTV_RSCORR_QUIET_S — TEST ONLY) */
extern int64_t g_rscorr_slew_us_s;       /* slew clamp, µs of trim per wall second (2000; PTV_RSCORR_SLEW — TEST ONLY) */
/* published cross-thread state (audio thread writes; stats line + master stale-track watchdog
 * read; the watchdog may CAS ENGAGED/DWELL→DISARMED when the track itself stopped emitting —
 * the one transition the owning thread cannot log, see rscorr_* in ptvencoder_audio.c) */
extern _Atomic int64_t g_corr_pub[PTV_MAX_AUDIO];        /* cumulative corr_us per track */
extern _Atomic int     g_corr_state_pub[PTV_MAX_AUDIO];  /* PTV_CORR_* per track */
extern _Atomic int64_t g_conv_pub[PTV_MAX_AUDIO];        /* rr23: cumulative folded label motion (us) */
extern _Atomic int64_t g_conv_park_pub[PTV_MAX_AUDIO];   /* rr23: seam_park_until wall us (0/past = unparked) */
extern _Atomic int64_t g_rsn_pub[PTV_MAX_AUDIO];         /* pre29: total resync fires (stats rsn= token) */
extern _Atomic int     g_corr_disarm_req[PTV_MAX_AUDIO]; /* master watchdog → audio thread (silent sync) */
/* 1.0.1-pre17 sibling-slate condition (compositor writes, corrector reads): bit k set while
 * input slot k is black-slated. While ANY bit is set no mv track may engage — a sibling
 * outage perturbs the shared compositor pacing (finding 1, grid soak 2026-07-19), so R is
 * un-actionable mosaic-wide until the slate clears + the quiet window re-runs. Belt over
 * the pre17 sensor fix (the fixed sensor reads true through a sibling slate; this freeze
 * is the §4.4 defense-in-depth for whatever the fix did not anticipate). */
extern _Atomic int g_mv_slate_mask;
/* per-rung wire-send watermark (§3, owner-approved "build it"): wall µs of the last SUCCESSFUL
 * av_interleaved_write_frame on that rung — the only liveness signal that is actually the wire
 * (the Newsmax2 dead rung read calm on every label-domain signal). One relaxed store per packet. */
extern _Atomic int64_t g_mux_sent_wc[PTV_MAX_RUNG];
/* 1.0.1-pre15 glue classification #33 (defined in ptvencoder.c; design doc
 * analysis/ptvencoder-33-glue-classification.md) */
extern int     g_glueclass;              /* the whole classifier; PTV_NO_GLUECLASS=1 reverts wholesale */
extern int     g_nbs_fill;               /* §3 starvation silence-fill — OPT-IN (PTV_NBS_FILL=1; owner Q2) */
extern int     g_glue_htol;              /* §2.3 label-health tolerance, percent (5; PTV_GLUE_HTOL_PCT — TEST/tuning) */
extern int64_t g_pair_ttl_us;            /* pair-expect TTL (30s; PTV_PAIR_EXPECT_TTL_US — TEST ONLY, G6) */
extern int64_t g_nbs_quantum_us;         /* fill quantum (100ms; PTV_NBS_QUANTUM_MS — TEST ONLY, G8) */
extern _Atomic int64_t g_acorrupt;                        /* total corrupt-discarded AUDIO pkts (acor= stats) */
extern _Atomic int64_t g_adec_frame_wc[PTV_MAX_AUDIO];    /* wall µs of track k's last DECODED frame (audio
                                                           * thread stamps; demux reads = the E6 starvation
                                                           * discriminator: packets arrive, nothing decodes) */
extern _Atomic int64_t g_pad_pub_step[PTV_MAX_AUDIO];     /* newest OPEN pad-ledger entry per track (audio thread */
extern _Atomic int64_t g_pad_pub_wc[PTV_MAX_AUDIO];       /* publishes; demux absorber reads — advisory) */
/* 1.0.1-pre18 #50 (defined in ptvencoder.c): gap-verdict/LAYERA one-remedy invariant */
extern int     g_glueveto;               /* veto + inverse guard + E5 flush-relabel net; PTV_NO_GLUEVETO=1 reverts */
extern int     g_hstick_filter;          /* #51a: ≤1-tick hs steps are not corrector events; PTV_NO_HSTICK_FILTER=1 reverts */
extern int     g_rscorr_ceil;            /* #51b: starvation-ceiling engage; PTV_NO_RSCORR_CEIL=1 reverts */
extern int64_t g_rscorr_ceil_us;         /* #51b ceiling span (15min default; PTV_RSCORR_CEIL_MIN minutes) */
extern int     g_perstream_wm;           /* §7.5b per-stream watermarks (slowest-live key); PTV_NO_PERSTREAM_WM=1 reverts */
extern int     g_rebuild_reanchor;       /* 1.0.1-pre20: AFMT-rebuild birth-equivalent re-anchor; PTV_NO_REBUILD_REANCHOR=1 reverts */
extern int     g_acq_backoff;            /* #49: repeated-ACQUIRE threshold backoff; PTV_NO_ACQ_BACKOFF=1 reverts */
extern _Atomic int64_t g_flush_relab_step[PTV_MAX_AUDIO]; /* last LAYERA-flush label shift per track (demux writes, */
extern _Atomic int64_t g_flush_relab_wc[PTV_MAX_AUDIO];   /* step first / wall last-release; audio thread reads) */
/* 1.0.1-pre20 rider (a) — rr191 F3: per-input demux-REOPEN stamp (wall µs; demux thread
 * stores at a successful [PTV-REOPEN], audio threads consume). The [PTV-ASTAMP] extrapolation
 * carry must never span a demux reopen: the fresh ctx is a fresh join and a first-frame NOPTS
 * extrapolated from the pre-outage carry would stamp it pre-outage-continuous. */
extern _Atomic int64_t g_reopen_wc[PTV_MAX_INPUT];
/* 1.0.1-pre20 rider (b) — proper `-t` (house-clock output media time; was parsed-and-ignored).
 * The cadence owner (single-input master rung / offline rung 0 / mv compositor) sets
 * g_t_stop once emitted output media time reaches g_t_us; every demux thread treats it as
 * EOF (stop pulling), so the whole pipeline flushes and exits through the proven file-EOF
 * teardown. The flushed pipeline residue means the final duration is >= -t by up to the
 * buffered cushion — -t bounds the INPUT pull, exactly what test cells need. 0 = no limit. */
extern int64_t g_t_us;
extern _Atomic int g_t_stop;

/* ==== cross-file functions ==== */
/* ptvencoder_gate.c */
void dlv_init(DlvGate *g, AVThreadMessageQueue *mux_q, int64_t cap_us, int maxq);
void dlv_publish_video(DlvGate *g, int64_t dts_us);
void dlv_enqueue(DlvGate *g, AVPacket *pkt, int64_t dts_us, int block);
void dlv_drain(DlvGate *g);
void dlv_flush_all(DlvGate *g);
void dlv_destroy(DlvGate *g);
void dlv_video_cfg(DlvGate *g, int64_t band_us, int64_t cap_us, int vmaxq);   /* §7.5b: arm the video-side hold */
int  dlv_video_deliver(DlvGate *g, AVPacket *pkt, int64_t dts_us);            /* §7.5b: video output thread only */
void dlv_video_drain(DlvGate *g);                                             /* §7.5b: video output thread only */
void cushion_escalate(CushionEvent ev, int64_t a0, int64_t a1);
void push_frame_q(AVThreadMessageQueue *q, int live, int64_t *framedrop, AVFrame *out);
void resolve_cushions(CushionPlan *cp, int live, int multiview,
                      AVRational out_fps, int n_audio);
void *watchdog_thread(void *arg);
/* ptvencoder_clock.c */
void *output_thread(void *arg);
/* ptvencoder_audio.c */
void *audio_thread(void *arg);
void ptv_adec_opts(AVCodecContext *dec);   /* 1.0.1-pre19 #38: opt-in tolerant AAC decode (by name, missing-option tolerant) */
int  ptv_find_stream_info(AVFormatContext *ic);   /* 1.0.1-pre19.1: find_stream_info with the #38 tolerance on the AUDIO probe decoders */
int build_audio_filter(AudioState *a, AVCodecContext *adec, AVRational tb,
                       const char *af, enum AVSampleFormat out_fmt);
/* ptvencoder_demux.c */
void *demux_thread(void *arg);
int ptv_disc_init(PtvDiscBuf *b, int capacity, int nb_streams);
void ptv_disc_free(PtvDiscBuf *b);
/* ptvencoder_mv.c */
void *compositor_thread(void *arg);
/* ptvencoder_legend.c */
void ptv_print_log_legend(int full);
/* 1.0.1-pre16 shared stats-line builders (single-input master printer in ptvencoder_clock.c
 * + the mv compositor printer in ptvencoder_mv.c). Byte-identical extraction of the pre9
 * lipsync= / pre14 corr= builders; the only semantic change is the per-slot video term
 * (g_rsx.mv_ema/mv_wall/ev_us keyed by g_rsx.a_in[ki] — identity 0 on single input). force_idx=1 (mv)
 * always prints the aK: prefix; 0 keeps the pre9 n_a>1 rule (single-input token-identical).
 * Both write "" when the field is absent. */
void ptv_stats_lipsync(char *buf, size_t size, int64_t now_us, int force_idx);
void ptv_stats_lipsync_in(char *buf, size_t size, int64_t now_us, int in);
void ptv_stats_corr(char *buf, size_t size, int force_idx);
void ptv_stats_conv(char *buf, size_t size, int64_t now_us, int force_idx);
void ptv_stats_rsn(char *buf, size_t size, int force_idx);
/* 1.0.1-pre17: the pre14 stale-track corrector watchdog, shared by both cadence owners —
 * the single-input master rung (ptvencoder_clock.c) and the mv compositor (the passthrough
 * rung loop returns before the master block, so mv needed its own call site). Callers
 * rate-limit (~1s). Body unchanged from pre14 (CAS to DISARMED + silent disarm_req). */
void rscorr_stale_watchdog(int64_t now);
/* ptvencoder.c */
void vring_put(VOutRing *r, int64_t src_us, int64_t out_us);
int vring_lookup(VOutRing *r, int64_t want_src, int64_t *out_v, int64_t *matched_src);

#endif /* FFTOOLS_PTVENCODER_H */
