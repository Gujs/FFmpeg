/*
 * ptvencoder — purpose-built live MPEG-TS re-encoder on a house-clock timing engine.
 * Architecture & design: analysis/ptvencoder-functional-spec.md
 * Version history / release notes: fftools/ptvencoder-changelog.md
 *
 * This file is licensed under the same terms as FFmpeg (GPL, --enable-gpl).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>   /* getpid() — 1.0.1-pre10 (g) per-PID phase jitter */

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

#include "cmdutils.h"
#include "ptvencoder.h"

const char program_name[] = "ptvencoder";
const int  program_birth_year = 2026;

#define PTVENCODER_VERSION "1.0.1-pre15"   /* bump per release; notes go in ptvencoder-changelog.md */
#define PTV_FRAME_QDEPTH 48    /* decode->output jitter buffer (frames); holds the pre-roll cushion */
int     g_diag;
/* A/V common-mode lock: the video frame-synchronizer's dup/drop makes the house
 * clock run ahead of source content; that skew is published by the master output
 * thread and added to the audio resampler's target so audio rides the SAME house
 * clock as video (else audio stays source-locked and drifts ~40ms per video dup).
 * On by default; PTV_NO_AVLOCK=1 reverts to the old source-locked audio. */
int     g_avlock = 1;
/* multiview coarse re-anchor: clear a slot's accumulated dup skew when it returns from a
 * black-slate outage, so its audio re-syncs (not delayed by the stale dup total). On by
 * default; PTV_NO_REANCHOR=1 keeps the stale skew across outages (for A/B comparison). */
int     g_reanchor = 1;
/* Multiview house-clock CONTENT CLAMP (DEFAULT OFF — superseded by the discontinuity absorber
 * below). It held an ahead-of-clock frame one tick at a time; on real feeds whose source PTS is
 * jittery that holds repeatedly = visible STUTTER (regressed in0/in2 on the box). Kept as an
 * opt-in experiment: PTV_MV_CLAMP=1 enables it. The real cause (a source PTS discontinuity, not
 * a content gap) is fixed at the source by g_discont, which keeps video smooth. */
int     g_mv_clamp = 0;
/* v0.9.13 per-slot CADENCE RESIDENCE (multiview): pop a slot's next frame only when its
 * content-projected residence has elapsed on the house axis, so the slot is CONSUMED at the
 * source rate (a 25fps slot in a 29.97 mosaic pops 5-per-6 ticks) instead of one-pop-per-tick.
 * Without it a rate-mismatched slot lives at the buffer boundary: clumped (SRT/HLS) arrival
 * turns into freeze-then-1.2x-fast batching (drain at house rate, starve, catch up at house
 * rate). The gate keys on an EMA of content deltas — NEVER raw per-frame deltas, which are
 * jittery on real interlaced feeds and caused the old content clamp's stutter regression.
 * On display the due time re-bases with FFMAX(due, now-half_tick): a starvation deficit
 * becomes constant slot latency (like a matched-rate slot's jitter buffer), never fast-motion
 * catch-up. High-rate slots (59.94-in-29.97) pop multiple due frames per tick and display the
 * newest (clean 2:1 decimation instead of hold-queue overflow). A ≥75%-full hold queue
 * bypasses the gate (pressure valve vs a wrong rate estimate). PTV_NO_RESIDENCE reverts. */
int     g_mv_residence = 1;
/* Per-input source PTS-discontinuity absorber (THE multiview audio-late fix). Real live feeds
 * (the 1080i50 Grid SRT inputs) throw a forward PTS jump of a few hundred ms at the join while
 * FRAMES stay continuous (one per tick, buffer full) — a timestamp glitch, not lost frames. Left
 * raw it shifts that slot's content→output mapping so the cell's video leaps ahead of its
 * (continuous) audio = per-slot "audio behind video". This absorbs it the SAME way demux_unwrap
 * absorbs the 33-bit wrap (a per-stream offset that keeps the effective timeline continuous),
 * applied to video+audio+copy uniformly so they stay aligned and video stays smooth. On by
 * default; PTV_NO_DISCONT=1 reverts; detect threshold internalized 0.9.18.7 (was PTV_DISCONT_MS; forward DTS jump
 * beyond it = a glitch to absorb). */
int     g_discont = 1;
int     g_gapdiscrim = 1;          /* gap-fix (2026-06-26): on a FORWARD audio jump, discriminate an audio-only source GAP
                                           * (video did NOT also cross + this stream was wall-absent ~the jump) from a whole-program
                                           * SPLICE. A GAP is NOT absorbed → aresample=async hard-pads silence → audio stays aligned
                                           * with the continuous video (fixes the AWE audio-gap → permanent A/V step). A SPLICE absorbs
                                           * as before. PTV_NO_GAPDISCRIM=1 reverts to unconditional forward absorb (old behaviour). */
int64_t g_wrap_guard_us = 0;       /* v0.9.16.1: sparse-PID wrap-guard threshold override (us); 0 = half the wrap period (13.26h @90kHz). PTV_WRAP_GUARD_S, TEST ONLY */
int     g_aglue_ms = 60;           /* v0.9.16.3 [PTV-AGLUE]: audio label-step glue threshold (ms); a decoded-audio in-pts step
                                           * beyond this (either direction) gets an explicit RELABEL-vs-GAP verdict at audio_feed —
                                           * the sub-1s band where the demux absorber (LAYERA-disabled) and LAYERA (>1s) are both
                                           * blind and aresample=async silently followed audio labels while video labels are
                                           * structurally erased by the house clock (the AWE-class lip-sync accumulator).
                                           * PTV_AGLUE_MS overrides; 0 disables (reverts to silent label-following). */
int     g_anchor_headfill = 1;     /* 1.0.1 anchor head-fill (PTV_NO_ANCHOR_HEADFILL reverts): when the source's
                                           * audio head is missing at birth (first kept audio >200ms after h0, or the
                                           * pre-h0 ring overflowed), synthesize silence covering house 0 → first kept
                                           * audio so the track's first packet sits at ~PTS 0 instead of first_audio−h0
                                           * (RAV mv 2026-07-07: +2058ms first packet = app-visible audio-early on naive
                                           * consumers). Capped at the pre-h0 ring's time span (~5.5s default). */
int     g_adecwd = 1;              /* 1.0.1 audio decode-death watchdog (PTV_NO_ADECWD reverts): if audio packets
                                           * keep arriving but the decoder yields ZERO frames for 45s wall, reopen the
                                           * decoder from the stream's codecpar (Pure Flix 2026-07-08: one corrupt-PCE
                                           * AAC event wedged the decoder and killed the track for 14h while video
                                           * survived identical storms via concealment). Anchor/pts state is PRESERVED —
                                           * mid-run recovery, aresample absorbs the gap like a source gap. The hard-
                                           * decode-error TOLERANCE in audio_thread (drop + [PTV-ADEC] + continue,
                                           * instead of silent thread death) is unconditional — only the reopen is
                                           * gated here. */
/* P2 §7.1 (hybrid): apply the program-level discontinuity offset (tracked from the dense VIDEO reference)
 * to the SPARSE copied streams (DVB-sub/teletext, data, SCTE-35) that can't self-rebase — so an ad-break
 * PTS jump shifts them WITH the video instead of orphaning/vanishing them. Dense V/A (incl. copied AC-3)
 * keep their own per-stream rebase (g_discont) untouched. Separate from g_discont so the sparse-program-
 * offset can be A/B'd against plain v0.6.23 sparse behaviour without disabling the whole absorber.
 * Default ON; PTV_NO_PROG_OFF=1 reverts (sparse get 33-bit wrap only, = v0.6.23 → orphaned across a jump). */
int     g_prog_off = 1;
/* §5.A.2 (v0.7.8 corrected): make DENSE video+audio absorb the SAME discontinuity amount so they don't
 * diverge. The old per-stream self-rebase used each stream's OWN adj (video frame-dur ≠ audio frame-dur,
 * different packet position across the interleaved splice) → a same-sign per-splice A/V residual that
 * accumulates (~+150ms/hr on TruBLU). Fix: each dense stream STILL self-rebases its own wrap_off at its
 * OWN crossing (the v0.6.23-proven path — never offsets a not-yet-crossed stream, so the compositor
 * h0/skew math sees no premature leap), but it uses the SHARED first-crosser adj (splice_adj, debounced
 * on wall-clock g_progoff_debounce_us so the 2nd stream crossing the SAME splice adopts the amount) →
 * V and A land on the same offset → zero divergence (= legacy 0004's single audio-derived offset; for
 * TruBLU's audio-led splices audio sets it, video adopts).
 *   ⚠ v0.7.7 FIRST TRY (apply prog_off to ALL packets immediately) was WRONG — during the V/A straddle it
 *   offset the not-yet-crossed stream → house_skew/aresample blew up a full splice (~1372s) live. This
 *   version touches ONLY the rebase amount; the apply path is the unchanged proven one.
 * DEFAULT ON (v0.7.10 — live-validated on TruBLU 13 ad-breaks eye-confirmed + Cinestar AC-3 channel 1h51m);
 * PTV_NO_PROGOFF_AV=1 disables it (per-channel A/B / rollback). ⚠ assumes a threshold-crossing jump is
 * program-wide; a video-only BACKWARD jump 80ms-1s would shift audio spuriously (§5.B-reserved asymmetric
 * case; none on fleet — watch unwrap_inj per source). */
int     g_progoff_av = 1;
/* Layer A (proven legacy 0004), PTV_LAYERA=1, default OFF. At a content glue (discontinuity) use ONE
 * AUDIO-derived offset for ALL dense streams: audio rebases seamlessly, video (and copy) ADOPT the
 * audio offset so video absorbs the residual via CFR dup/drop. This CORRECTS a source glue A/V
 * mis-alignment (g_progoff_av merely shares the first-crosser amount → preserves the source A/V,
 * unwrap_inj≈0 → fast channels that glue mis-muxed content accumulate the error). Supersedes
 * g_progoff_av when set. Re-aligns video if it crossed before audio within the debounce window. */
int     g_layera = 1;   /* v0.9.10: DEFAULT ON (proven production posture); PTV_NO_LAYERA reverts */
/* 0.9.18.5 (In-Touch audio-late accumulator, analysis/ptvencoder-intouch-desync-analysis.md §4b):
 * under g_layera the demux_unwrap absorber used to be skipped for ALL super-threshold jumps, but
 * LAYERA itself only claims jumps >1s — the sub-1s band (80ms..1s backward) had NO packet-layer
 * owner. A BOTH-STREAM backward step there was converted into house_skew ratchet/decimation on the
 * video side while AGLUE RELABEL-erased the audio side, and AVLOCK re-injected the video conversion
 * into audio → the same source event actuated TWICE on audio = audio permanently LATE by ~the step
 * per event (measured +620ms/3h, +1477ms/26h on In-Touch_+; F1 fixture: +301ms/event staircase).
 * The skip is now scoped to LAYERA's own band (>PTV_DISC_THRESHOLD_US, matching
 * ptv_disc_detect_jump); sub-1s steps fall through to the proven §5.A.2 shared-amount absorber.
 * PTV_LAYERA_FULLSKIP=1 restores the old full-skip posture (A/B / rollback). */
int     g_layera_fullskip = 0;
/* 1.0.1-pre4 SHARED FLUSH (the LAYERA asymmetric-event invariant fix). THE INVARIANT
 * (owner-mandated): after any input event, the output's A/V alignment must equal the source's
 * post-event alignment — latency may be retained; relative A/V offset may never be. LAYERA's
 * per-flush erase preserved that only when V and A jumped by the SAME amount in the SAME flush
 * cycle; the Curiosity/PATRIOT 2026-07-13 provider playout jumps were ASYMMETRIC and crossed
 * 0.6s apart (past the 500ms buffer timeout), so each stream's flush applied its OWN offset
 * (vid −14.148s, aud −15.155s) and the ~1s A-vs-V jump difference was FROZEN into the output
 * for 8.5h with every counter clean. Now dense flushes within PTV_PAIR_WINDOW_US share ONE
 * offset — VIDEO's delta defines the timeline (video is the house-clock anchor; prog_off/SCTE
 * ride the video timeline) — and the A-vs-V jump difference is NOT erased: it surfaces as an
 * audio label step that the CONTENT machinery converges (pre5: the flush REGISTERS the step
 * per track — ptv_pair_expect — so AGLUE APPLIES it in every direction: forward = aresample
 * pads, backward = aresample drops, above the AGLUE cap = aresample=async hard pad/drop —
 * bounded, and exactly what a stateless player shows; AGLUE's backward relabel-ERASE would
 * otherwise re-bake sub-1s backward mismatches, the pre4 D1 defect). REDUCES to
 * per-stream-identical (byte-identical logs) behavior when the deltas
 * are equal: offset disagreement at or below PTV_PAIR_EPS_US (500ms) is flush bookkeeping
 * (duration-estimate overhang, trailing-OLD discard holes), and there the production-proven
 * audio-preferred butt-joint is kept exactly (TruBLU symmetric rewinds unchanged — the
 * invariant holds in the band regardless, since ONE offset is applied either way).
 * Decision tree at ptv_disc_flush. PTV_NO_SHARED_FLUSH=1 reverts to the per-stream erase. */
int     g_shared_flush = 1;
/* P2 §7.1 / stage 2b: after a detected source discontinuity, DROP video packets until the next keyframe
 * (IDR) before they reach the decoder — a splice starts a NEW timeline mid-GOP, so the P/B frames that
 * reference the missing IDR decode as a corruption burst (greyed/torn frames) that the house clock would
 * then sample. Dropping them lets the house clock dup-hold the last good frame across the splice = a clean
 * cut instead of a corruption burst. Bounded by a wall-clock ESCAPE (g_dukf_escape_us) so a stream that
 * never sends an IDR can't freeze the cell (the session-109 28h-freeze lesson), and armed FIRST-ARM-ONLY
 * (never re-stamp the escape deadline while already armed — the re-arm slide). Default ON; PTV_NO_DUKF=1
 * reverts (decode the post-splice burst, = v0.6.23). MULTI/SINGLE both (per-input demux state). */
int     g_drop_until_kf = 1;
/* Multiview per-slot AUDIO-FOLLOW (Option A) — the per-slot A/V-sync fix. A mosaic's composite
 * video is forced onto a house-clock POSITION timeline (rung_pts; one shared frame can't be
 * content-stamped per cell), while the audio is CONTENT-anchored (src-h0). At the join they sit
 * on different origins → a stable per-slot offset → "audio behind video". Single-input has no
 * split (its video IS content-stamped) and is untouched. Fix: the compositor measures each
 * slot's stable offset (smoothed past the interlaced ±100ms PTS jitter, latched after a warmup)
 * and the slot's audio applies it as a ONE-TIME deterministic correction — DROP |offset| of
 * audio content if it's behind the video, PAD silence if ahead — landing the audio on the video's
 * displayed-content clock. Deterministic because aresample=async is far too slow (~20ms/s) and
 * can't advance audio for a sub-second offset. MULTIVIEW ONLY (n_input>1); PTV_NO_AUDIO_FOLLOW
 * reverts to the old floored/capped async-skew path for A/B. */
int     g_audio_follow = 1;
/* Multiview per-slot h0 RE-ANCHOR — floor each slot's lag to ≥0 so a cell is never displayed
 * AHEAD of the house clock. A slot's video can leap far ahead (measured: −560ms on a 2x1, up to
 * −2.5s on a 4-up) when h0 is anchored to an anomalous first decoded frame and/or the input
 * primed a deep startup backlog (the open-join barrier lets fast decoders over-fill their jitter
 * buffer while waiting for the slowest input). Video-ahead (negative lag) is (a) physically wrong
 * for a frame-synchronizer and (b) UNCORRECTABLE on a COPIED audio track — a copy can only have
 * its timestamps shifted LATER (delay); advancing it means a backward DTS, which the copy path's
 * monotonic-DTS clamp rejects. So when a slot's lag drops below −g_h0_reanchor_ms, re-anchor its
 * h0 forward (h0 += deficit) so the lag lands at a small POSITIVE value: the video display is
 * unchanged (same frame shown), the transcoded audio rides the same h0+house_skew so it stays
 * locked, and the copied audio now only needs to DELAY → correctable. MULTIVIEW ONLY (n_input>1);
 * PTV_NO_H0_REANCHOR=1 reverts. */
int     g_h0_reanchor = 1;
int     g_reanchor2_instant = 0;   /* 1.0.1 (PTV_REANCHOR2_INSTANT=1 reverts): REANCHOR2 fires only when ≥3 of the
                                           * last 5 evaluated ticks held sk < −threshold, with the h0 shift sized from the
                                           * MEDIAN qualifying sk — a single corrupt-PTS frame (DTS intact, so it passes
                                           * the demux discontinuity layer) no longer inflates the shift by its full
                                           * excursion (transient audio-early until the PLL healed). */
/* h0-AT-DISPLAY (multiview): anchor each slot's h0 to the first frame the COMPOSITOR actually DISPLAYS,
 * not the first frame the decoder produces. Under a deep startup prime the first-decoded frame is an
 * earlier/different content than the first-displayed one, so the old decode-thread anchor left the
 * displayed video leaping ahead of h0 at tick 0 → P2 re-anchored h0 forward → the transcoded audio
 * banked (monotonic guard) and a copied audio track's DTS jumped backward (clamp/freeze, historically
 * an EINVAL no-data outage). Anchoring at first display makes sk=0 from the start so P2 never fires —
 * no bank, no clamp, no outage. MULTIVIEW ONLY; single-input keeps the decode-thread anchor (BYTE-
 * IDENTICAL). PTV_NO_H0_AT_DISPLAY=1 reverts to the decode-thread anchor (A/B). */
int     g_h0_at_display = 1;
/* A/V PLL redesign — Phase A READ-ONLY measurement probe (analysis/ptvencoder-avsync-pll-redesign-plan.md).
 * Off by default; PTV_AVSYNC_PROBE=1 enables the [PTV-AVSYNC2] per-track real A/V offset measurement
 * (out_v(C) − out_a(C), content-paired, video_lag/audio_lag split). Measures only — no actuator. */
int     g_avsync_probe = 0;
/* Multiview audio-follow ACTUATOR (P1) — a per-slot two-mode controller for glitch-free A/V tracking.
 * The v0.6.2/0.6.3 audio-follow corrected the per-slot lag only with whole-frame drop/pad fired on a
 * 40ms threshold: it tracked the slow per-slot drift (source-vs-house-clock, e.g. +360ms over 35min)
 * in discrete ~40ms steps — each step a momentary A/V hop ("sometimes off, then OK again"), and it
 * left a ±frame residual ("audio slightly behind"). P1 keeps the fast discrete drop/pad ONLY to
 * ACQUIRE a large gap (startup ramp / big jump the smooth rate can't catch), and otherwise TRACKS the
 * residual + slow drift with a SMOOTH, rate-limited (≤g_af_rate_us/s, imperceptible) sub-sample PTS
 * nudge — no silence inserts, no content drops, no hops. Reuses the legacy 0007 PLL's rate-limited
 * actuator idea, per slot, driven by the directly-measured lag (no jump_comp). MULTIVIEW ONLY;
 * PTV_AF_NO_PLL=1 reverts to pure discrete. */
int     g_af_pll = 1;
/* Phase B (B1) — CONTENT-ANCHORED multiview audio. The pre-B1 multiview audio-follow emitted on a
 * FREE-RUNNING sample counter (af_next_pts), which faithfully banked aresample=async's STARTUP
 * over-production into a permanent per-slot audio-late offset (Phase A root cause: alag +400..+1252ms,
 * the dominant desync term). B1 instead anchors the output to opts (async's self-correcting content
 * target — exactly what single-input does, which is why single-input reads offset≈0) plus a smooth
 * rate-limited offset that tracks the compositor's per-slot lag, so the audio follows the video DISPLAY
 * without banking over-production. Default ON; PTV_AF_NO_ANCHOR=1 reverts to the free-counter path (A/B). */
int     g_af_anchor = 1;
/* A/V PLL redesign Phase B3 — CLOSED-LOOP two-regime controller on the MEASURED offset.
 * The open-loop audio-follow (B1) steers af_applied_us to track house_skew (the VIDEO's lag) and is
 * structurally blind to the AUDIO's own startup bank: aresample=async over-produces over the startup
 * gap and the v0.6.8 monotonic guard FREEZES it into a permanent per-slot audio-late offset (measured
 * fleet-wide on tmtg: alag a frozen +0..+2100ms STEP, never drifting; the open-loop applied=house_skew,
 * trk≈0, cannot see it). B3 closes the loop on the faithful measured av_offset_us (= vlag − alag): a
 * fast ACQUIRE (one-shot drop/pad sized to the frozen bank — the ~99%-step disturbance) snaps it out in
 * one tune-in skip, then a type-1 integral TRACK trims the residual + any slow drift. Both regimes emit
 * on the B1 content-anchored base (want = opts + applied) so there is ONE base and the guard never sees
 * a backward step (the acquire's content-drop and applied step CANCEL in want). MULTIVIEW transcoded
 * audio ONLY; single-input + copy paths untouched. Default OFF for box A/B; PTV_AVSYNC_PLL=1 enables.
 * Sign proven: d(offset)/d(applied) < 0 ⇒ to raise a negative offset (audio late), advance (drop). */
int     g_avsync_pll = 1;             /* B3 closed-loop A/V controller DEFAULT-ON (v0.6.20, box-validated on cor-2 RAV + live-transcoder grids). PTV_NO_AVSYNC_PLL reverts to the open-loop B1 follow. Multiview transcoded audio only; single-input + copy paths byte-identical regardless. */
int     g_acq_instant = 0;            /* 1.0.1 (PTV_ACQ_INSTANT=1 reverts): ACQUIRE needs the |EMA offset| above threshold for 3 CONSECUTIVE debounce windows (and the threshold is floored at 1.5 house ticks) — the vlag measurement is tick-quantized, so the single-window fire snapped on its own quantization noise (live grids: ~939-1511 ACQUIREs/22h alternating ±42ms pad/drop). */
int     g_pll_trackup = 1;            /* 1.0.1-pre3 (PTV_NO_PLL_TRACKUP=1 disables TRACK entirely = acquire-only, labels flat — the operators' production mute keeps its meaning): TRACK now steers through the RESAMPLER (af_steer_us into the graph-input pts, AVLOCK-style) instead of re-stamping output labels. pre2's label-TRACK stretched output AAC pts spacing up to +158ms/min during integration episodes → PTS-honoring players rate-chased it = audible warble (production 2026-07-13). The pre2 [PTV-TRACKUP] direction-aware anti-windup is retired with the label actuator. */
int64_t g_pll_testnoise_us = 0;       /* TEST-ONLY (default off): inject a ±N ms square wave (flips ~every 3.2s) into the measured offset to REPRODUCE the box limit cycle locally (local sources are clean). PTV_PLL_TESTNOISE_MS sets it; never set in production. */
/* v0.9.14 AUTO-BANK (the owner-agreed auto-cushion escalation, cap 12s). The 0.9.10 adaptive
 * cushion lives in frame_q (DECODED frames) and maxes at ~4s — structurally too shallow for
 * HLS-burst channels with 6-8s gaps (Unique_TV/ZOE class), whose fix has been a MANUAL deep
 * startup prime (PTV_PREROLL_MS/PTV_VIDEOQ). This closes the loop at runtime: when the demux
 * BURSTY detector qualifies a channel (>=3 stalls >=1.5s per 60s, or a single stall >=3s), the
 * bank target becomes 1.5x the worst observed stall, capped at PTV_CUSHION_MAX_MS (12s). Arming
 * flips the master rung to the deep-prime BLOCKING push (frame_q gates decode -> the excess
 * banks in video_q as COMPRESSED packets — cheap), so each stall's own latency is RETAINED as
 * the bank instead of being drained: the channel self-heals within a stall cycle or two, no
 * restart, no viewer-visible fill. Growth is self-limiting (once bank >= gap, stalls stop
 * starving so latency stops growing). The per-rung delivery-gate cap rides the bank so audio
 * waits out long stalls with video. Decays after 6h without qualifying stalls (the drained
 * latency then bleeds via the normal catch-up path). PTV_NO_AUTOBANK reverts to advisor-only;
 * an explicit PTV_PREROLL_MS deep prime keeps working and pre-fills at startup as before.
 * Single-input live only (mosaic slots ride cadence-residence starvation-slip; mv bank later). */
static int     g_autobank = 1;
int64_t g_cushion_max_ms = 12000;      /* PTV_CUSHION_MAX_MS: bank ceiling (owner: >12s stalls are
                                               * upstream INCIDENTS to surface, not absorb; also = startup
                                               * darkness after every restart if baked into a preroll) */
int64_t g_bank_decay_us = 6 * 3600 * 1000000LL;   /* PTV_BANK_DECAY_S: quiet time before the bank drops */
_Atomic int     g_bank_pkts;           /* >0 = deep-bank semantics armed (master rung blocking push) */
_Atomic int64_t g_bank_us;             /* current bank TARGET (us) — stats/log readout */
_Atomic int     g_vq_elems;            /* video_q depth (pkts), demux-updated — the bank ACTUAL readout */
_Atomic int     g_fq_hw;               /* 0.9.18 #19: worst frame-queue depth ever seen (any rung; mv incl.
                                               * hold.q) — one catch-up burst fills a rung to cap and the CUDA frame
                                               * pool keeps that high-water forever, so this ≈ the per-process VRAM
                                               * footprint driver (stats fqhw=) */
int     g_exacttick = 1;      /* v0.9.9 EXACTTICK (PTV_NO_EXACTTICK reverts): compute the video content
                                      * index with the EXACT rational frame duration instead of dividing by the
                                      * integer-us tick_dur_us. The integer tick (e.g. 33367 vs true 33366.667us
                                      * at 30000/1001) compresses video's content->PTS mapping by ~10ppm while
                                      * audio is stamped sample-exact -> audio drifts BEHIND at ~36ms/h on every
                                      * NTSC-rate channel (TruBLU/AWE), zero at 25/50fps (Fintech/Cinestar) --
                                      * invisible to hs/dup by construction. Root cause of the chronic lip-sync
                                      * drift (4-agent audit 2026-07-02; oracle-measured +42..52ms/h vs +36 predicted). */
int     g_mv_exacttick = 1;   /* v0.9.12 MV-EXACTTICK (PTV_NO_MV_EXACTTICK reverts): compositor MEASUREMENT
                                      * axes (pacing target, per-slot sk/house_skew, h0 anchor, the B3 PLL's vring
                                      * sensor, clamp, vout, stats) in EXACT-rational tick-us instead of
                                      * tick x integer tick_dur_us — the integer axis ran +10ppm fast vs the muxed
                                      * video axis at 30000/1001, so the per-slot audio followers ENFORCED
                                      * ~36ms/h audio-late onto the wire while every internal offset read bounded.
                                      * See analysis/ptvencoder-0911-multiview-tick-audit.md. */
int     g_decimate = 1;       /* v0.9.15.2 cadence decimation (PTV_NO_DECIMATE reverts): a frame whose
                                      * content index does not advance past the last emitted tick is surplus
                                      * (source delivers MORE real frames than its declared rate — NewsNation
                                      * ~25.3-25.5 real fps declared 25/1, cadence wandering) — replace it with
                                      * the next and display only the newest due frame. The output samples the
                                      * source's own timeline at the house rate: lip-sync exact, frame_q level. */
int     g_pulldown = 1;       /* v0.9.11 telecine-aware emit (PTV_NO_PULLDOWN reverts): during 23.976-film-
                                      * in-29.97 segments (2:3 soft pulldown, repeat_pict flags — AWE movies) a
                                      * flagged frame legitimately OCCUPIES extra emission ticks. repeat_pict only
                                      * ARMS the mode (>=3 progressive-rff frames of the last 8); the hold decision
                                      * is CONTENT-PROJECTED via a 1-frame lookahead using the SAME content-index
                                      * arithmetic as the stamps — a blind rff half-tick accumulator was REJECTED
                                      * (uniform-stamped telecine would ratchet hs +0.5s/min; see
                                      * analysis/ptvencoder-0911-repeat-pict-design.md). Result during film:
                                      * consumption matches supply -> no starvation dups (irregular stutter), no
                                      * hs sawtooth -> no aresample hard-comps (the audio clicks), proper 2:3
                                      * cadence on the wire. Inert unless armed; non-film pop path is verbatim. */
int     g_cad_disarm = 1;     /* v0.9.18.1 M7 (PTV_NO_CADDISARM reverts): pulldown DISARM additionally
                                      * requires CONTENT-RATE evidence (frame spacing back at ~house tick).
                                      * AWE's encoder drops RFF flags for >=8-frame runs mid-film (measured
                                      * 2026-07-06, test-results/pulldown-trap/): flag-only disarm then drains
                                      * frame_q ~6f/s (29.97 house vs ~24 AU/s film — no cushion offsets a
                                      * rate deficit) until the next flag run re-arms -> dup bursts + aresample
                                      * hard-comps = the audible clicking. Frame spacing is evidence flags
                                      * can't fake: film-in-NTSC arrives ~41.7ms/frame, real-time ~= tick. */
int     g_frameq_cap = 160;  /* decode->output jitter-buffer CAPACITY (frames; slots are pointers — memory
                                     * is used only by FILLED depth). v0.9.10 default 160 = headroom for the 4s
                                     * adaptive tier (~120f @30fps) + catch-up bursts, while bounding the worst-case
                                     * transient bank (~1GB VRAM ladder-wide on CUDA channels vs ~2GB at 320).
                                     * PTV_FRAMEQ overrides [48,1024] for explicit deep setups (e.g. 8s + preroll). */
int     g_preroll_ms = 350;   /* §13: house-clock startup cushion (ms). >~1.6s (frame_q cap) → single-input decode delays its start until video_q banks this much (deep bursty-input prime). Default 350 → 0 deep packets → byte-identical. Bounded [0,30000]; PTV_PREROLL_MS sets it. v0.9.0: genlock defaults this to ~1 GOP (2000) unless set explicitly. */
/* Discard video packets the demuxer flags AV_PKT_FLAG_CORRUPT (= -fflags +discardcorrupt), at the demux
 * before they reach the decoder, and COUNT them (DemuxArgs.vcorrupt) so frame loss is visible in stats.
 * A corrupt frame, like a dropped one, becomes a content GAP that the position-anchored composite video
 * leaps across → desync; discarding early + counting makes it observable. Default ON; PTV_KEEP_CORRUPT=1
 * keeps corrupt packets (lets the decoder try to use them — the prior behavior). */
int     g_discardcorrupt = 1;
/* §7.5a DELIVERY ALIGNMENT (P1) — the post-encode A/V wire-alignment gate. The NVENC video
 * encoder holds a frame ~1s (B-frames + CBR bufsize + GPU) while the audio/copied-AC-3 encoder
 * is near-zero-latency, so audio reaches the muxer ~1s AHEAD of the video for the SAME content
 * → audio-ahead-of-video on the wire → the downstream sync_check (video_last − audio_last) trips
 * → restart (measured fleet-wide on the nvenc ladder; A0 = +0.85–0.9s, encoder-caused). This is
 * the fftools "sync queue" the greenfield mux dropped. The gate re-creates the interleave-wait for
 * the DENSE near-zero-latency streams (transcoded audio + copied AC-3/MP2): hold each until the
 * VIDEO encoder's emitted DTS has reached that packet's DTS, then release in lockstep. PTS are
 * NEVER modified — only WHEN a packet reaches the muxer. Sparse SCTE-35/subs BYPASS (their wire-
 * arrival lead is a feature). Default ON for LIVE single-input AND multiview (v0.9.12.1). The
 * wire skew equals the CURRENT video in-process hold: the interleaver orders by DTS but only
 * waits max_interleave_delta (200ms) for the late stream, so when video reaches the muxer
 * held back by frame_q occupancy + the HW encoder (~1s NVENC steady, several s during
 * post-stall catch-up on bursty slots), audio passes through and video lands out-of-order
 * behind it. cor-2's downstream sync_check measured D = video-audio = -0.6..-5.9s on ungated
 * mosaics and restart-looped them (|D|>2s => restart). PTV_NO_DELIVERY=1 kills the gate
 * everywhere (audio sent direct = byte-identical to v0.6.23); PTV_NO_DELIVERY_MV=1 keeps only
 * multiview ungated (pre-0.9.12.1 wire staging). Offline (file out) always bypasses. */
static int     g_delivery = 1;
static int     g_delivery_mv = 1;             /* v0.9.12.1: gate multiview too (PTV_NO_DELIVERY_MV reverts) */
/* §7.5b (1.0.1-pre12) SYMMETRIC delivery gate — the VIDEO side: hold EARLY VIDEO on the audio
 * DELIVERED-DTS high-water, closing the mirror skew the §7.5a gate cannot touch (a buffering -af
 * — the fleet loudnorm, ~3s analysis fill — delays audio in WALL time, so video leaves the mux
 * seconds of content AHEAD; AWE_Plus live: video start 1134.03 vs audio 1131.69). Model, deadlock
 * invariant and audio-death safety: see the §7.5b header in ptvencoder_gate.c. Single-input live
 * with gated audio only; PTV_NO_VDELIVERY=1 reverts to the pre-pre12 wire. */
static int     g_vdelivery = 1;
/* Muxed-packet stats counters. Written by N mux threads (6-rung ABR) -> atomic to avoid a
 * data race / lost updates in the bitrate=/size= stat line. Stats-only; not on any hot path. */
_Atomic int64_t g_muxed;
/* ffmpeg-style progress line (frame=/fps=/bitrate=/speed=); on unless -nostats. */
int     g_stats = 1;
static _Atomic int64_t g_muxed_bytes;
/* [PTV-CHAIN] data-driven A/V trace (diagnostic): latest SOURCE-CONTENT time (us, post-unwrap)
 * at the demux and at output emission, for video + primary audio. THREE-WAY split:
 * rawA-V (PRE-demux_unwrap, source-native) vs srcA-V (POST-unwrap) vs outA-V (output) →
 * separates source-inherent A/V drift (raw grows) from demux_unwrap per-stream rebase
 * divergence (unwrap_inj grows) from ptvencoder restamp (introduced). Coarse 10s, relaxed atomics. */
_Atomic int64_t g_ch_vsrc, g_ch_asrc, g_ch_vout_src, g_ch_aout_src;
_Atomic int64_t g_ch_vsrc_raw, g_ch_asrc_raw;   /* [PTV-CHAIN] PRE-unwrap raw source ts (us) */
/* v0.9.0 source-clock genlock: slave the single-input master output cadence to the recovered source
 * frame rate so the house clock stops drifting vs the channel (no growing output-slower lag), house_skew
 * → 0, and aresample is freed for honest A/V trim. The estimator (demux thread, post-unwrap video DTS vs
 * wall clock) publishes RateEstimator.src_rate_q20 = content-µs per wall-µs in Q20 (1<<20 == declared
 * nominal); the master pacer scales its per-tick wall span by it. g_genlock_ok is true only for
 * single-input live (the multiview compositor is unaffected). PTV_NO_GENLOCK reverts to byte-identical
 * free-run. */
int             g_genlock = 1;
int             g_genlock_ok;                  /* runtime: single-input live (set at setup) */
/* v0.9.15 CLOCK-FOLLOW: some real sources run their transport clock PERCENT-scale fast/slow
 * (NewsNation measured +12200ppm: a relay/playout fault the provider won't fix). The tight FLL
 * above correctly rejects that as insane for crystal-drift sensing (±300ppm guard), so a
 * PARALLEL coarse estimator (same unbiased sub-window rate, ±3% envelope, own outlier reject,
 * 60s lock) feeds the WUCR servo, which then FOLLOWS the verified offset beyond its ±0.6%
 * gentle zone (cap ±2%): output pacing (and PCR) run at the source's true rate — receivers
 * slave to PCR, buffers stay level, and aresample drops from churning hard-comps to a steady
 * soft ratio. Film-in-NTSC can never arm this (its DTS advance is realtime; the estimator
 * reads ~0). Single-input live only. PTV_NO_CLOCKFOLLOW reverts. */
int             g_clockfollow = 1;
/* WUCR (W0): PTV_WUCR enables the occupancy-recovered house rate ρ — a type-2 PI loop on frame_q
 * fill, ±150ppm HARD clamp (physical crystal bound → runaway structurally impossible), burst-freeze
 * on super-physical fill slope. W0 drives ONLY the video pacer (per_tick) and surfaces buf/rho in
 * -stats next to srcppm for the go/no-go (ρ flat where the DTS-vs-wall FLL ran away). The FLL still
 * runs (srcppm computed for comparison) but does not pace. Audio coupling to ρ is W1.
 * Genlock remains the fallback pacer. HouseRateState.rho_corr_ppm = the APPLIED correction (I+P),
 * one producer (master), all rungs apply it identically. */
int             g_wucr = 1;                   /* v0.9.10: DEFAULT ON (proven production posture); PTV_NO_WUCR reverts */
int             g_reprime = 1;                /* PTV_REPRIME: when a glue drains frame_q below half the BASE floor, slow the house HARD (≈0.77x)
                                                      * to refill fast → dups stop, house_skew stays bounded (the ±6% refill let it run to 7-15s overnight).
                                                      * Composes with WUCR (occupancy servo) + AVLOCK. v0.9.10: DEFAULT ON, rate-limited to one
                                                      * engagement per 5min (PTV_NO_REPRIME reverts). Engagement state lives in HouseRateState. */
/* v0.9.10 ADAPTIVE CUSHION (PTV_NO_ADAPTIVE reverts to a fixed preroll target). Two discrete frame_q
 * targets, no continuous wander: BASE = the resolved preroll (~1s) and RAISED = g_cushion_ms (~4s).
 * GROW: two starvation episodes (frame_q empty >=200ms) within 60min -> RAISED (one-off glitches never
 * grow it). Fill is LAZY: no house slow-down — the servo's gentle zone (+/-0.6% above the base floor)
 * lets natural source catch-up bursts fill the deeper target, so downstream delivery never jerks.
 * SHRINK: 6h with zero starvations -> back to BASE (drains at ppm scale; GPU frames free as it drains).
 * State is quantized + hysteretic (grow needs recurrence in 1h, shrink needs 6h silence) -> no
 * oscillation, no per-channel tuning; transitions log [PTV-CUSHION] and depth shows in -stats. */
int             g_adapt_cushion = 1;
_Atomic int     g_frameq_depth;               /* DIAG: master video frame_q occupancy (frames), published each tick for the discontinuity logs */
/* v0.9.4 genlock GUARD (PTV_NO_GENLOCK_GUARD reverts to exact v0.9.x behavior). TruBLU-class jittery/
 * bursty sources alias the 3s FLL window → noisy sub-window rates that the loose ±1% gate folded in,
 * driving a slew-limited ±1000ppm limit cycle + an UNBOUNDED house_skew runaway (cor-1: 8.6→28s over
 * 16h; the audio is then padded to chase a clock that's running away → visible desync after hours).
 * Guard = (A) a hard ABSOLUTE bound on the applied rate (±g_gl_max_q20) so a fooled estimate can never
 * pace the house clock past a physical envelope, and (B) RELATIVE outlier rejection (skip a sub-window
 * whose rate deviates from the running estimate by > g_gl_reject_q20 — the burst-alias spikes). Clean
 * sources (Cinestar ±45ppm, AWE ±300ppm) sit inside both bounds → unaffected. */
int             g_genlock_guard = 1;
/* v0.9.2 logging cleanup — HONEST always-on aresample-work metric, latched by the audio drain and
 * read by the master video thread for the progress line (relaxed atomic, like the g_ch_* chain:
 * no lock, no clock read on the hot path).
 *   g_async_ppm  : aresample compensation RATE = d(out_span − content_span)/d(wall), ppm. + =
 *                  stretching/adding samples, − = compressing. ~0 = idle (genlock removed the
 *                  structural drift); a sustained sign = the resampler is doing net work. A rate,
 *                  so the slowly-varying house_skew DC term washes out (unlike the confounded
 *                  async_pad span).
 * (An egress emitted-PES A/V skew metric `emitA-V` was built here and REJECTED after wire-oracle
 *  validation: a +200ms injected content shift moved the oracle by +200ms but emitA-V by 0 — it is
 *  dominated by encoder B-frame reorder and blind to the content↔PTS mapping that IS lip-sync. So
 *  ptvencoder does NOT self-report lip-sync; it is measured externally by drift-continuous.py.) */
_Atomic int64_t g_async_ppm;
/* -stats_period: interval (us) between progress lines. Default 1s; raise for
 * production (e.g. -stats_period 10 -> every 10s) to keep logs quiet. */
int64_t g_stats_period_us = 1000000;
/* PTV_SLOW_US: inject N us of extra per-emitted-frame consumer cost, to model a
 * slow/blocking encoder on a box that has none. Stress knob, gated. */
int     g_slow;
/* PTV_SLOW_DEC_US (1.0.1-pre8): inject N us of extra per-video-packet DECODE cost — the
 * faithful stand-in for a slow/contended NVDEC (the #32 wedge entry: decode falls behind →
 * video_q fills → overflow policy engages). PTV_SLOW_US slows the OUTPUT thread, which the
 * frame_q drop-oldest decouples from video_q, so it cannot exercise the demux overflow path.
 * Optional PTV_SLOW_DEC_FROM_S / PTV_SLOW_DEC_FOR_S window the slowdown (seconds since
 * process start) so a mid-run consumer-slowdown + release is reproducible without signals.
 * Stress knob, gated; default off. */
static int     g_slow_dec;
static int64_t g_slow_dec_on_us, g_slow_dec_off_us;   /* absolute monotonic window; off=0 → forever */
/* ==== 1.0.1-pre8 — the #32 WEDGE fixes (live-proven mechanism, cor-3 2026-07-15): slow NVDEC
 * (mass-restart GPU contention) → video_q fills to cap → the demux TAIL-DROPPED arriving video
 * PER-PACKET, MID-GOP (~70%) → the decoder received GOP fragments and ran at ~11% of realtime →
 * the queue never drained → the drop policy fragmented its own input FOREVER on a clean wire.
 * Self-sustaining; also entered MID-RUN whenever the consumer fell behind long enough. ==== */
/* (a) GOP-COHERENT VIDEO OVERFLOW (QSHED): when video_q must shed load, never drop random tail
 * packets. The decoder head-sheds WHOLE GOPs (oldest first — drop from the head to the next
 * keyframe, which then decodes) on the demux's overflow request, and the demux tail-drops the
 * arriving stream to the next IDR while full — so the decoder ALWAYS receives contiguous,
 * decodable GOPs and runs at full speed the instant it can consume: the fragmentation feedback
 * loop is structurally impossible. Audio overflow sheds whole frames OLDEST-first (frames are
 * independent; keeping the freshest drains latency instead of pinning it). PTV_NO_QSHED reverts. */
int     g_qshed = 1;
/* (b) RATCHET RELEASE ON STARVATION: a starved frame_q (≤2 frames for ≥5s) with an ARMED BANK
 * while input IS flowing is a contradiction — holding gate latency for a buffer that is empty.
 * Release the bank + delivery caps immediately (BANK_RELEASE) instead of the 6h decay; the
 * retained latency then drains via the normal catch-up path (blocking push disarms). Normal
 * deep-bank operation (bursty delivery, buffers full or input absent) is untouched — banks
 * exist for that. PTV_NO_RATCHREL reverts. */
int     g_ratchrel = 1;
/* (c) SELF-HEAL RE-PRIME BACKSTOP: sustained frame_q starvation (≥30s) while input IS flowing
 * means the decode path is wedged on stale/undecodable backlog — the decoder flushes video_q +
 * its own state and resumes at the next IDR (what a supervisor restart achieves, in-process).
 * Rate-limited to one attempt per 5min; loudly logged [PTV-SELFHEAL]. PTV_NO_SELFHEAL reverts. */
int     g_selfheal = 1;
int     g_vindbg;    /* TEMP pre13 diagnosis: PTV_VINDBG=1 traces the vin_pps window + governor */
_Atomic int     g_selfheal_req;
_Atomic int64_t g_v_arrive_wc;
/* (d) SELF-MADE-GAP LOG HONESTY: every self-inflicted queue drop (video head/tail shed, audio
 * drop-oldest) stamps these; AGLUE/ASTEP lines within 5s carry " [self: N pkts shed]" so our
 * own drops are never again misread as source burstiness. */
_Atomic int64_t g_shed_wall;
_Atomic int64_t g_shed_cnt;
/* ==== 1.0.1-pre10 (7h) — BIRTH-ARMED CHURN: mode release + consumption rate-shape. Phase-A
 * localization (pre10 verdicts): the live ~6s QSHED full-cycle churn is a capacity-deficit
 * limit cycle whose armed states release either never (g_delivery_maxq) or only on conditions
 * the churn itself makes unreachable (cushion tier: 6h of ZERO starvation — every cycle resets
 * the clock), and whose post-shed catch-up decode runs UNGOVERNED (measured p95 2.2x realtime
 * bursts; N co-located instances burst-feed a shared device in phase). ==== */
/* (e) CUSHION RELEASE ON STARVATION-CONTRADICTION: the tier that armed at birth (2 starvation
 * episodes in 60min — birth under contention trips it in ~6s) raises the frame_q target
 * 59->152 and the gate caps +2.5s, and its only release is 6h with zero starvation episodes —
 * unreachable while churning, so in production it pins the frame pool + NVENC registration
 * set at maximum forever. Symmetric with pre8's BANK_RELEASE: when the starvation
 * contradiction (frame_q <=2 with input FLOWING) has held >=60s and the tier is raised,
 * step it back to base + restore the grown gate caps (CUSHION_RELEASE). Never fires when
 * input is NOT flowing (a genuine stall/outage keeps its cushion). PTV_NO_CUSHREL reverts. */
int     g_cushrel = 1;
/* (f) GOVERNED CATCH-UP: the decode->frame_q submission path has no rate governance — after
 * every shed/heal/starvation dip it runs at device max (frame_q pushes are drop-oldest
 * NONBLOCK in live, so backpressure vanishes exactly when it matters). Cap deficit-recovery
 * decode at 1.25x realtime (per-frame budget = 4/5 of the master tick — the same currency
 * WUCR governs the emit side with). Engaged ONLY while a self-shed/heal happened within the
 * last 10min AND video_q holds >1s of backlog: normal steady-state decode (1.0x by supply)
 * and clean channels (g_shed_wall never stamped) are structurally untouched.
 * PTV_NO_CATCHGOV reverts. */
int     g_catchgov = 1;
/* 1.0.1-pre13 governor observability (Newsmax2 live defect, 2026-07-16 cor-3): the wedge —
 * dec=6.6/s on a clean 59.94pps wire, vq pinned 725-784, dup 45/s, kill-switch A/B instant
 * 60/s — was undiagnosable from logs because the DIAG t= line carried no gpps/engagement.
 * Single-input decode publishes; the master DIAG t= line prints gpps=meas/decl gov=. */
_Atomic int     g_gov_gpps;
_Atomic int     g_gov_decl;
_Atomic int     g_gov_on;
_Atomic int64_t g_gov_slip;
/* (g) PHASE JITTER: deterministic per-PID +/-20% jitter on the shed/heal cycle timings (the
 * head-shed depth-gate margin + the SELFHEAL 5min re-fire) so N co-located instances cannot
 * phase-lock their ~6s burst cycles on one shared device. 1000 = no jitter (PTV_NO_PHASEJIT). */
int     g_jit_milli = 1000;
/* (h) SUSTAINED-DEFICIT DEGRADED MODE (opt-in, PTV_DEGRADED=1 — the cor-3 experiment lever;
 * the local repro cannot prove the production self-sustainment it targets): after >=3min of
 * persistent QSHED full-cycles the demux flushes the stale backlog and goes DEMAND-DRIVEN:
 * an arriving GOP is admitted only when video_q <= ~1s (the queue depth IS the throughput
 * measurement) so consumption <= capacity, video_q stops cap-cycling, and retained latency
 * self-scales with the deficit instead of accumulating (fixture-measured: modulus-K
 * admission held 60s of decode-time depth and audio, A/V-locked to it, died at the demux
 * door; demand admission keeps the delay in the ~15s class the defaults churn already
 * proves audio rides). Releases after 60s of demonstrated decode headroom. Default OFF —
 * a no-op (byte-identical) when the env is unset. */
int     g_degraded = 0;
/* 1.0.1-pre9 (7g) — PASSIVE residual lip-sync sensor (component 1 of the residual-sync
 * supervisor; full model at the RsyncSense declaration in ptvencoder.h). Writers: master
 * output thread (m_v), audio threads (m_a), demux thread (E ledgers). Readers: the stats
 * line (`lipsync=`) + [PTV-RSYNC] DIAG — NOTHING ELSE (no actuation; the corrector is a
 * later round, gated on a live sensor-vs-oracle soak). PTV_RSYNC_SENSE=0 disables. */
int        g_rsync_sense = 1;
RsyncSense g_rsx;
/* 1.0.1-pre14 — residual-sync CORRECTOR (component 2; analysis/ptvencoder-corrector-design.md).
 * The actuation half: a per-track, dwell-gated, slew-clamped (2ms/s) resampler steer on the
 * certified pre9/pre11 sensor, injected at the graph door on the steer bus (§5). DEFAULT ON
 * (owner-directed 2026-07-17: every channel runs it, parked and byte-inert when healthy);
 * PTV_NO_RSYNC_CORR=1 is the permanent kill switch; sensor off implies corrector off. State
 * machine + control law live in ptvencoder_audio.c (rscorr_*); the stats line prints corr= and
 * the master output thread runs the stale-track disarm watchdog (ptvencoder_clock.c). */
int     g_rsync_corr      = 1;
int64_t g_rscorr_engage_us = 80000;      /* §4.2: 80ms engage dead band (supervisor start value) */
int64_t g_rscorr_dwell_us  = 300000000;  /* §4.3: 5min stable dwell */
int64_t g_rscorr_quiet_us  = 180000000;  /* §4.4: 3min trailing event-free window */
int64_t g_rscorr_slew_us_s = 2000;       /* §4: 2ms/s slew clamp (1/5 of pre3's TRACK clamp) */
_Atomic int64_t g_corr_pub[PTV_MAX_AUDIO];
_Atomic int     g_corr_state_pub[PTV_MAX_AUDIO];
_Atomic int     g_corr_disarm_req[PTV_MAX_AUDIO];
_Atomic int64_t g_mux_sent_wc[PTV_MAX_RUNG];
/* 1.0.1-pre15 — glue classification #33 (analysis/ptvencoder-33-glue-classification.md).
 * One revert switch for the whole classifier: PTV_NO_GLUECLASS=1 restores pre14 WIRE
 * behavior wholesale (§2.2 pad-cancel + tripwire, §2.3 refuse, §2.5 gap-verdict propagation,
 * late pair-expect matching, and the §3 fill owner) — NOT full log parity: the DUKF
 * resume/escape promotion and the (c) acorrupt counter/[PTV-ADISC]/acor= observability stay
 * UNCONDITIONAL (owner-sanctioned, byte-inert; rr15 F7). The §3 silence-fill is additionally
 * OPT-IN (PTV_NBS_FILL=1 — owner call 2026-07-18: observability first fleet-wide, fill
 * second; NBS phases are currently restart-cured, not silent-failing). */
int     g_glueclass = 1;
int     g_nbs_fill  = 0;
int     g_glue_htol = 5;                       /* §2.3 |H−1| tolerance, % (fixture-tuned, G4) */
int64_t g_pair_ttl_us = PTV_PAIR_EXPECT_TTL_US;
int64_t g_nbs_quantum_us = 100000;             /* fill quantum: 100ms of silence per sentinel */
_Atomic int64_t g_acorrupt;
_Atomic int64_t g_adec_frame_wc[PTV_MAX_AUDIO];
_Atomic int64_t g_pad_pub_step[PTV_MAX_AUDIO];
_Atomic int64_t g_pad_pub_wc[PTV_MAX_AUDIO];

/* PTV_LOG_TS=1: prefix every log line with a local wall-clock timestamp
 * [YYYY-MM-DD HH:MM:SS.mmm], so production logs are self-dated natively
 * (replaces piping through `ts`). Wraps libav's line formatter; serialized so
 * lines from the demux/decode/encode/mux threads don't interleave. */
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;
static void ptv_log_ts_callback(void *avcl, int level, const char *fmt, va_list vl)
{
    static int print_prefix = 1, at_line_start = 1;
    char buf[2048];
    int n, start;

    if (level > av_log_get_level())
        return;
    pthread_mutex_lock(&g_log_mtx);
    av_log_format_line2(avcl, level, fmt, vl, buf, sizeof buf, &print_prefix);
    n = (int)strlen(buf);
    for (start = 0; start < n; ) {
        const char *nl = memchr(buf + start, '\n', n - start);
        int end = nl ? (int)(nl - buf) + 1 : n;
        if (at_line_start) {
            int64_t now = av_gettime();
            time_t s = (time_t)(now / 1000000);
            struct tm tm; char d[24];
            localtime_r(&s, &tm);
            strftime(d, sizeof d, "%Y-%m-%d %H:%M:%S", &tm);
            fprintf(stderr, "[%s.%03d] ", d, (int)((now % 1000000) / 1000));
            at_line_start = 0;
        }
        fwrite(buf + start, 1, (size_t)(end - start), stderr);
        if (nl) at_line_start = 1;
        start = end;
    }
    pthread_mutex_unlock(&g_log_mtx);
}


/* Free function for AVThreadMessageQueue elements (AVPacket* / AVFrame*; a NULL
 * element is an end-of-stream marker on mux_q). */
static void free_pkt_msg(void *msg)   { av_packet_free(msg); }
static void free_frame_msg(void *msg) { av_frame_free(msg); }

/* ---- video: decode (free-run) + output (master clock, sample-and-hold) ---- */


void vring_put(VOutRing *r, int64_t src_us, int64_t out_us)
{
    pthread_mutex_lock(&r->lock);
    int i = (int)(r->n % PTV_VRING);
    r->src[i] = src_us; r->out[i] = out_us; r->n++;
    pthread_mutex_unlock(&r->lock);
}

/* nearest-by-content lookup: of all kept entries, return the out_v and matched src of the one
 * whose src is closest to want_src. 0 = found (ring non-empty), -1 = empty. */
int vring_lookup(VOutRing *r, int64_t want_src, int64_t *out_v, int64_t *matched_src)
{
    int64_t best = INT64_MAX, bo = 0, bs = 0;
    int found = 0, cnt, i;
    pthread_mutex_lock(&r->lock);
    cnt = r->n < PTV_VRING ? (int)r->n : PTV_VRING;
    for (i = 0; i < cnt; i++) {
        int idx = (int)((r->n - 1 - i) % PTV_VRING);
        int64_t d = r->src[idx] - want_src; if (d < 0) d = -d;
        if (d < best) { best = d; bo = r->out[idx]; bs = r->src[idx]; found = 1; }
    }
    pthread_mutex_unlock(&r->lock);
    if (found) { *out_v = bo; *matched_src = bs; }
    return found ? 0 : -1;
}

/* Build the optional video filter graph: buffer -> [deint][,scale] -> buffersink.
 * CPU backend: bwdif + scale + format=yuv420p. CUDA backend: hwupload_cuda +
 * bwdif_cuda + scale_cuda (output stays on GPU, fed straight to NVENC).
 * On success sets v->filtering and returns the output w/h/pixfmt (+ hw_frames_ctx
 * for the CUDA path) so the encoder can be configured to match. */
static int build_video_filter(DecodeCtx *d, AVCodecContext *vdec, AVRational tb,
                              const char *vf, int do_deint, int sw, int sh, int hw_cuda,
                              AVBufferRef *hw_device,
                              int *out_w, int *out_h, int *out_pixfmt,
                              AVBufferRef **out_hwfr)
{
    char args[256], desc[256];
    const char *chain;
    const AVFilter *bsrc  = avfilter_get_by_name("buffer");
    const AVFilter *bsink = avfilter_get_by_name("buffersink");
    AVFilterInOut *ins = avfilter_inout_alloc(), *outs = avfilter_inout_alloc();
    AVRational sar = vdec->sample_aspect_ratio.num ? vdec->sample_aspect_ratio : (AVRational){1, 1};
    int ret;

    if (!bsrc || !bsink || !ins || !outs) { ret = AVERROR(ENOMEM); goto end; }
    d->fg = avfilter_graph_alloc();
    if (!d->fg) { ret = AVERROR(ENOMEM); goto end; }

    snprintf(args, sizeof(args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
             vdec->width, vdec->height, vdec->pix_fmt, tb.num, tb.den, sar.num, sar.den);
    if ((ret = avfilter_graph_create_filter(&d->fsrc, bsrc, "in", args, NULL, d->fg)) < 0) goto end;
    if ((ret = avfilter_graph_create_filter(&d->fsink[0], bsink, "out", NULL, NULL, d->fg)) < 0) goto end;

    if (vf) {
        chain = vf;                                  /* raw ffmpeg-dialect chain */
    } else {                                         /* convenience flags -> chain */
        char *p = desc; int rem = sizeof(desc), n = 0;
#define APPEND(...) do { int k = snprintf(p, rem, "%s", n++ ? "," : ""); p += k; rem -= k; \
                         k = snprintf(p, rem, __VA_ARGS__); p += k; rem -= k; } while (0)
        if (hw_cuda) {
            APPEND("hwupload_cuda");
            if (do_deint) APPEND("bwdif_cuda=mode=send_frame");   /* non-doubling, match -r (like CPU --deint) */
            if (sw > 0)   APPEND("scale_cuda=%d:%d", sw, sh);
        } else {
            if (do_deint) APPEND("bwdif=mode=send_frame:deint=all");
            if (sw > 0)   APPEND("scale=%d:%d:flags=bicubic", sw, sh);
            APPEND("format=yuv420p");
        }
#undef APPEND
        chain = desc;
    }

    /* outputs = the buffer src feeding the parsed chain; inputs = the sink it feeds */
    outs->name = av_strdup("in");  outs->filter_ctx = d->fsrc;     outs->pad_idx = 0; outs->next = NULL;
    ins->name  = av_strdup("out"); ins->filter_ctx  = d->fsink[0]; ins->pad_idx  = 0; ins->next  = NULL;
    if ((ret = avfilter_graph_parse_ptr(d->fg, chain, &ins, &outs, NULL)) < 0) goto end;

    if (hw_cuda && hw_device)
        for (unsigned i = 0; i < d->fg->nb_filters; i++)
            d->fg->filters[i]->hw_device_ctx = av_buffer_ref(hw_device);

    if ((ret = avfilter_graph_config(d->fg, NULL)) < 0) goto end;

    *out_w      = av_buffersink_get_w(d->fsink[0]);
    *out_h      = av_buffersink_get_h(d->fsink[0]);
    *out_pixfmt = av_buffersink_get_format(d->fsink[0]);
    if (out_hwfr) {
        AVBufferRef *hf = av_buffersink_get_hw_frames_ctx(d->fsink[0]);
        *out_hwfr = hf ? av_buffer_ref(hf) : NULL;
    }
    av_log(NULL, AV_LOG_INFO, "ptvencoder: filter [%s] -> %dx%d\n", chain, *out_w, *out_h);
    d->filtering = 1;
    ret = 0;
end:
    avfilter_inout_free(&ins);
    avfilter_inout_free(&outs);
    return ret;
}

/* Build a SHARED filter_complex graph: N video buffersrcs (one per input, bound
 * to the graph's `[k:v]` labels) -> the user graph (single-input ABR split, or a
 * multiview xstack/hstack/overlay mosaic + split) -> N buffersinks (one per rung
 * label). Mirrors ffmpeg's -filter_complex. n_inputs==1 is the single-input
 * ladder (unchanged); n_inputs>1 is multiview, fed by the compositor.
 *
 * vdecs[k] supplies input k's width/height/pix_fmt/sar; src_tb is the time_base
 * stamped on every buffersrc (single-input: the source stream tb, frames carry
 * source pts; multiview: 1/out_fps, the compositor stamps pts = house tick).
 * labels[i] is rung i's bare output label; sinks[i] receives that branch. */
static int build_filter_complex(const char *graph_str, AVCodecContext **vdecs,
                                int n_inputs, AVRational src_tb, AVBufferRef *hw_device,
                                const char *const *labels, int n_labels,
                                AVFilterGraph **out_fg, AVFilterContext **srcs,
                                AVFilterContext **sinks)
{
    char args[256], name[16];
    AVFilterGraph        *fg    = avfilter_graph_alloc();
    AVFilterGraphSegment *seg   = NULL;
    const AVFilter       *bsrc  = avfilter_get_by_name("buffer");
    const AVFilter       *bsink = avfilter_get_by_name("buffersink");
    AVFilterInOut        *gin = NULL, *gout = NULL, *io;
    int ret, i, k, linked = 0;

    if (!fg || !bsrc || !bsink) { ret = AVERROR(ENOMEM); goto fail; }

    /* Build via the segment API so the hw device is assigned to every filter
     * BEFORE it is initialised. Plain `hwupload` (unlike `hwupload_cuda`)
     * hard-requires avctx->hw_device_ctx in its init(); a one-shot
     * avfilter_graph_parse2() inits filters during the parse, too early to set
     * it. Sequence: parse -> create_filters -> SET DEVICE -> apply_opts -> init
     * -> link. After link, gin = the unconnected inputs ([k:v]) and gout = the
     * unconnected outputs (each rung label), wired to our buffersrcs / sinks. */
    if ((ret = avfilter_graph_segment_parse(fg, graph_str, 0, &seg)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex parse: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_create_filters(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex create: %s\n", av_err2str(ret)); goto fail;
    }
    if (hw_device)                                   /* must precede init() (hwupload) */
        for (unsigned f = 0; f < fg->nb_filters; f++)
            fg->filters[f]->hw_device_ctx = av_buffer_ref(hw_device);
    if ((ret = avfilter_graph_segment_apply_opts(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex opts: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_init(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex init: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_link(seg, 0, &gin, &gout)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex link: %s\n", av_err2str(ret)); goto fail;
    }

    /* one buffersrc per input, params from that input's decoder */
    for (k = 0; k < n_inputs; k++) {
        AVRational sar = vdecs[k]->sample_aspect_ratio.num ? vdecs[k]->sample_aspect_ratio : (AVRational){1, 1};
        snprintf(name, sizeof name, "in%d", k);
        snprintf(args, sizeof args,
                 "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
                 vdecs[k]->width, vdecs[k]->height, vdecs[k]->pix_fmt, src_tb.num, src_tb.den, sar.num, sar.den);
        if ((ret = avfilter_graph_create_filter(&srcs[k], bsrc, name, args, NULL, fg)) < 0) goto fail;
    }
    /* link each unconnected graph input [K:v] to buffersrc K (single input: [0:v]->src0) */
    for (io = gin; io; io = io->next) {
        int idx = 0;
        if (io->name) { const char *c = io->name; if (*c >= '0' && *c <= '9') idx = atoi(c); }
        if (idx < 0 || idx >= n_inputs) {
            av_log(NULL, AV_LOG_ERROR, "filter_complex input [%s] out of range (n_inputs=%d)\n",
                   io->name ? io->name : "?", n_inputs);
            ret = AVERROR(EINVAL); goto fail;
        }
        if ((ret = avfilter_link(srcs[idx], 0, io->filter_ctx, io->pad_idx)) < 0) goto fail;
        linked++;
    }
    if (linked != n_inputs) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex: %d input(s) linked, expected %d ([0:v]..[%d:v])\n",
               linked, n_inputs, n_inputs - 1);
        ret = AVERROR(EINVAL); goto fail;
    }

    for (i = 0; i < n_labels; i++) {                 /* one buffersink per rung label */
        AVFilterContext *sink = NULL;
        for (io = gout; io; io = io->next)
            if (io->name && !strcmp(io->name, labels[i])) break;
        if (!io) {
            av_log(NULL, AV_LOG_ERROR, "filter_complex has no output labelled [%s]\n", labels[i]);
            ret = AVERROR(EINVAL); goto fail;
        }
        if ((ret = avfilter_graph_create_filter(&sink, bsink, labels[i], NULL, NULL, fg)) < 0) goto fail;
        if ((ret = avfilter_link(io->filter_ctx, io->pad_idx, sink, 0)) < 0) goto fail;
        sinks[i] = sink;
    }

    if ((ret = avfilter_graph_config(fg, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex config: %s\n", av_err2str(ret)); goto fail;
    }
    *out_fg = fg;
    avfilter_graph_segment_free(&seg);
    avfilter_inout_free(&gin); avfilter_inout_free(&gout);
    return 0;
fail:
    avfilter_graph_segment_free(&seg);
    avfilter_inout_free(&gin); avfilter_inout_free(&gout);
    avfilter_graph_free(&fg);
    return ret;
}

/* Hand a decoded frame downstream: straight to the jitter buffer, or through the
 * filter graph first. Source PTS is preserved (frame->pts) so the output thread's
 * content-PTS A/V anchoring still holds across the filter.
 *
 * §13 deep-prime channels (deep_prime_packets>0): rung 0 (the master/house-clock rung) pushes
 * LOSSLESS/BLOCKING for the WHOLE run (not just startup), so the decoder back-pressures on
 * frame_q[0] -> stays paced to the house clock -> the deep cushion stays parked in video_q
 * (a fast decoder can't race it into the 48-frame frame_q and drop it). Consequences, intended:
 *   - the drop-newest backstop moves from frame_q[0] to the (deeper) video_q cap on sustained overload;
 *   - back-pressure chains rung0-output -> decoder -> video_q -> demux -> input, so a rung-0
 *     output/mux/gate stall becomes INPUT packet loss on these channels (right trade for bursty inputs).
 * Non-master rungs keep drop-newest (stall-isolated). Default (deep_prime_packets==0) = d->live everywhere. */
static void emit_video(DecodeCtx *d, AVFrame *frame, AVFrame *filt)
{
    int i;
    if (!d->filtering) {                 /* no graph: clone the decoded frame to each rung */
        if (frame->best_effort_timestamp != AV_NOPTS_VALUE)   /* source time in ist_tb (== out_tb) */
            frame->pts = frame->best_effort_timestamp;
        for (i = 0; i < d->n_rung; i++) {
            AVFrame *out;
            if (i == d->n_rung - 1) { out = av_frame_alloc(); if (out) av_frame_move_ref(out, frame); }
            else                    { out = av_frame_clone(frame); }
            if (out) push_frame_q(d->frame_q[i], ((d->deep_prime_packets > 0 || atomic_load_explicit(&g_bank_pkts, memory_order_relaxed) > 0) && i == 0) ? 0 : d->live, &d->framedrop[i], out);
            else if (i == d->n_rung - 1) av_frame_unref(frame);
        }
        return;
    }
    frame->pts = frame->best_effort_timestamp;   /* carry source time through the graph */
    if (av_buffersrc_add_frame(d->fsrc, frame) < 0)   /* consumes frame */
        return;
    for (i = 0; i < d->n_rung; i++) {                 /* split branch -> each rung's frame_q */
        while (av_buffersink_get_frame(d->fsink[i], filt) >= 0) {
            AVFrame *out = av_frame_alloc();
            if (out) { av_frame_move_ref(out, filt); push_frame_q(d->frame_q[i], ((d->deep_prime_packets > 0 || atomic_load_explicit(&g_bank_pkts, memory_order_relaxed) > 0) && i == 0) ? 0 : d->live, &d->framedrop[i], out); }
            else     { av_frame_unref(filt); }
        }
    }
}

/* Multiview: push a decoded frame onto this input's jitter buffer for the
 * compositor (FIFO, one per tick). Carries source pts on frame->pts so the
 * compositor can compute this slot's house skew. Takes ownership of the frame. */
static void stage_hold(VideoHold *h, int live, AVFrame *frame)
{
    AVFrame *nf = av_frame_alloc();
    if (!nf) { av_frame_unref(frame); return; }
    av_frame_move_ref(nf, frame);
    if (nf->best_effort_timestamp != AV_NOPTS_VALUE) nf->pts = nf->best_effort_timestamp;
    push_frame_q(h->q, live, &h->framedrop, nf);    /* drop-oldest in live; consumes nf */
    pthread_mutex_lock(&h->lock);
    h->wall_us = av_gettime_relative();
    pthread_mutex_unlock(&h->lock);
}

static void *decode_thread(void *arg)
{
    DecodeCtx *d = arg;
    AVPacket *pkt;
    AVFrame  *frame = av_frame_alloc();
    AVFrame  *filt  = av_frame_alloc();
    int ret = 0, i;
    int64_t gov_next_us = 0;   /* 1.0.1-pre10 (f): governed catch-up pacing anchor (0 = disengaged) */
    int64_t gov_strike_win_us = 0, gov_holdoff_until_us = 0;   /* 1.0.1-pre13: oversleep strikes + fail-open holdoff */
    int     gov_strikes = 0;

    if (!frame || !filt)
        goto done;
    /* §13 deep bursty-input prime: wait for the demux to bank ≥ deep_prime_packets in video_q
     * BEFORE decoding, so the realtime-limited decoder has a multi-segment buffer to ride
     * HLS-segment delivery gaps (~6s segment = 1.3s burst + 4.7s gap). Without this, the decoder
     * keeps video_q drained to ~one burst and the gaps starve the house clock -> monotonic
     * house_skew runaway. Demux fills video_q while we sleep; bounded by 3x the target time. */
    if (d->deep_prime_packets > 0) {
        int64_t t0 = av_gettime_relative();
        /* ⚠ STARTUP BLACKOUT: while we bank the cushion the decoder emits no frames, so the output
         * thread emits nothing — for up to `budget`. Normal (≈realtime source) fill takes ≈preroll_ms;
         * the 2x budget caps the worst case at ~2x preroll_ms (e.g. ~16s at PTV_PREROLL_MS=8000). This
         * is the deep buffer's latency, paid once at channel start (and on each crash-loop restart). */
        int64_t budget = g_cp.deep_prime_budget_us;   /* 2x preroll_ms, in us (resolve_cushions) */
        while (av_thread_message_queue_nb_elems(d->video_q) < d->deep_prime_packets
               && av_gettime_relative() - t0 < budget)
            av_usleep(5000);
        if (g_diag)
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-DIAG] deep prime: video_q banked %d/%d packets in %.1fs before decode\n",
                   av_thread_message_queue_nb_elems(d->video_q), d->deep_prime_packets,
                   (av_gettime_relative() - t0) / 1000000.0);
    }
    for (;;) {
        pkt = NULL;
        /* 1.0.1-pre8 (a) HEAD-GOP SHED: the demux flagged a video_q overflow. Only the consumer
         * can pop the queue head, so the shed executes here: drop packets from the HEAD up to
         * (not including) the next keyframe — the oldest whole GOP (or the un-decoded remainder
         * of the GOP currently in progress) — and DECODE that keyframe. The decoder's input
         * stays contiguous whole GOPs, so it runs at full speed the moment it can consume;
         * repeated overflows re-arm the request one GOP at a time. Entry is depth-gated so a
         * stale request after the pressure passed can never shed a healthy queue. */
        if (g_qshed && d->live && d->vq_shed_req &&
            atomic_load_explicit(d->vq_shed_req, memory_order_relaxed)) {
            atomic_store_explicit(d->vq_shed_req, 0, memory_order_relaxed);
            /* 1.0.1-pre10 (g): the 128-pkt stale-request margin carries the per-PID +/-20%
             * jitter — co-located instances re-enter the shed at different depths, so their
             * ~6s full-cycle trains cannot phase-lock on a shared device. */
            if (av_thread_message_queue_nb_elems(d->video_q) >=
                FFMAX(g_cp.videoq_pkts / 2, g_cp.videoq_pkts - 128 * g_jit_milli / 1000)) {
                int shed_n = 0;
                int64_t d0 = AV_NOPTS_VALUE, d1 = AV_NOPTS_VALUE;
                for (;;) {
                    AVPacket *sp;
                    if (av_thread_message_queue_recv(d->video_q, &sp,
                                                     AV_THREAD_MESSAGE_NONBLOCK) < 0)
                        break;                                   /* drained — stop */
                    if ((sp->flags & AV_PKT_FLAG_KEY) && shed_n > 0) {
                        pkt = sp;                                /* GOP boundary: this key DECODES */
                        break;
                    }
                    if (sp->dts != AV_NOPTS_VALUE) {
                        if (d0 == AV_NOPTS_VALUE) d0 = sp->dts;
                        d1 = sp->dts;
                    }
                    shed_n++;
                    av_packet_free(&sp);
                }
                if (shed_n > 0) {
                    int64_t nw = av_gettime_relative();
                    int64_t span_ms = (d0 != AV_NOPTS_VALUE && d1 != AV_NOPTS_VALUE)
                        ? av_rescale_q(d1 - d0, d->ist_tb, (AVRational){1, 1000}) : 0;
                    d->shed_pkts += shed_n;
                    atomic_store_explicit(&g_shed_wall, nw, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g_shed_cnt, shed_n, memory_order_relaxed);
                    if (nw - d->shed_log_us >= 1000000) {        /* rate-limit; totals keep it honest */
                        d->shed_log_us = nw;
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-QSHED] video_q overflow: dropped GOP %d pkts (%"PRId64"ms) from the head — "
                               "resuming at the next keyframe (total head-shed %"PRId64" pkts)\n",
                               shed_n, span_ms, d->shed_pkts);
                    }
                }
            }
        }
        if (!pkt) {
            /* Timed recv (rr8 review defect 1, heal reachability): the heal executor lives in
             * THIS thread, so a pending g_selfheal_req must be servable even while video_q is
             * empty — a blocking recv made the heal unreachable exactly when it was needed
             * (demux wedged in tail-drop, queue drained → thread parked forever). Poll the
             * queue and serve the request on the empty path; the resume-at-IDR (with its
             * Session-109 escape) then runs instead of waiting for a packet that never
             * comes. */
            for (;;) {
                ret = av_thread_message_queue_recv(d->video_q, &pkt, AV_THREAD_MESSAGE_NONBLOCK);
                if (ret != AVERROR(EAGAIN)) break;              /* got a pkt, or queue closed */
                if (g_selfheal && d->live && !d->hold &&
                    atomic_exchange_explicit(&g_selfheal_req, 0, memory_order_relaxed)) {
                    d->heal_dropkf = 1;
                    d->heal_arm_us = av_gettime_relative();
                    atomic_store_explicit(&g_shed_wall, d->heal_arm_us, memory_order_relaxed);
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-SELFHEAL] re-prime (video_q empty): decoder resets at the "
                           "next IDR\n");
                }
                av_usleep(5000);
            }
            if (ret < 0) break;
        }
        /* 1.0.1-pre8 (c) SELF-HEAL RE-PRIME: the master output thread measured sustained
         * frame_q starvation with input flowing — the decode path is wedged on stale or
         * undecodable backlog. Do what a supervisor restart achieves, in-process: drop the
         * whole queued backlog and resume clean at the next IDR (the decoder reset is
         * deferred to that IDR — see the heal_dropkf block). Anchors (h0) and downstream
         * state are preserved; aresample absorbs the content gap. */
        if (g_selfheal && d->live && !d->hold &&
            atomic_exchange_explicit(&g_selfheal_req, 0, memory_order_relaxed)) {
            AVPacket *sp;
            int flushed = 1;                                     /* the packet in hand goes too */
            av_packet_free(&pkt);
            while (av_thread_message_queue_recv(d->video_q, &sp, AV_THREAD_MESSAGE_NONBLOCK) >= 0) {
                av_packet_free(&sp);
                flushed++;
            }
            d->heal_dropkf = 1;
            d->heal_arm_us = av_gettime_relative();
            atomic_store_explicit(&g_shed_wall, d->heal_arm_us, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_shed_cnt, flushed, memory_order_relaxed);
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-SELFHEAL] re-prime: flushed %d queued video pkts; decoder resets at "
                   "the next IDR\n", flushed);
            continue;
        }
        if (d->heal_dropkf) {                                    /* (c): clean resume boundary */
            if (pkt->flags & AV_PKT_FLAG_KEY) {
                /* rr8 review defect 1: the decoder reset is DEFERRED to here — flushing at
                 * heal-request time destroyed h264 sync, and on a source with no IDR /
                 * recovery point the decoder then consumed packets forever without emitting
                 * a frame (permanent freeze the heal itself caused). Resetting right before
                 * the IDR decodes is observably identical on IDR-rich sources (no packet
                 * touches the decoder in between) and keeps sync when the escape fires. */
                avcodec_flush_buffers(d->vdec);
                d->heal_dropkf = 0;
            } else if (av_gettime_relative() - d->heal_arm_us > 5000000) {
                d->heal_dropkf = 0;                              /* Session-109 escape: no IDR within 5s
                                                                  * → decode anyway, never freeze; the
                                                                  * decoder is deliberately NOT flushed
                                                                  * (its established sync is the only
                                                                  * one a no-IDR source will ever have) */
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-SELFHEAL] no IDR within 5s of the re-prime — resuming mid-GOP "
                       "(decoder kept unflushed)\n");
            } else {
                av_packet_free(&pkt);
                atomic_fetch_add_explicit(&g_shed_cnt, 1, memory_order_relaxed);
                continue;
            }
        }
        if (g_slow_dec) {   /* 1.0.1-pre8 stress knob: model a slow/contended NVDEC (windowed) */
            int64_t nws = av_gettime_relative();
            if (nws >= g_slow_dec_on_us && (!g_slow_dec_off_us || nws < g_slow_dec_off_us))
                av_usleep(g_slow_dec);
        }
        ret = avcodec_send_packet(d->vdec, pkt);
        av_packet_free(&pkt);
        while (ret >= 0) {
            ret = avcodec_receive_frame(d->vdec, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) goto done;
            if (frame->flags & AV_FRAME_FLAG_CORRUPT) { d->vcorrupt++; av_frame_unref(frame); continue; }
            int64_t ts = frame->best_effort_timestamp;
            if (ts != AV_NOPTS_VALUE) {
                int unset;
                pthread_mutex_lock(d->h0_lock); unset = (*d->h0 == AV_NOPTS_VALUE); pthread_mutex_unlock(d->h0_lock);
                /* Multiview anchors h0 at the compositor's FIRST DISPLAY (g_h0_at_display), not here at
                 * first DECODE — a deep prime makes those different contents → startup leap → P2 → audio
                 * bank. Single-input (no d->hold) keeps the decode-thread anchor (BYTE-IDENTICAL). */
                if (unset && (!d->hold || !g_h0_at_display)) {
                    pthread_mutex_lock(d->h0_lock);
                    if (*d->h0 == AV_NOPTS_VALUE) {
                        *d->h0 = av_rescale_q(ts, d->ist_tb, AV_TIME_BASE_Q);
                        /* [PTV-ANCHOR] (v0.9.16.3, always-on): the video half of the birth pair —
                         * each audio track logs its first_audio-h0 offset against this. */
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-ANCHOR] h0 anchored at %"PRId64"ms (first decoded video frame)\n",
                               *d->h0 / 1000);
                    }
                    pthread_mutex_unlock(d->h0_lock);
                }
            }
            /* 1.0.1-pre10 (f) GOVERNED CATCH-UP: pace deficit-recovery decode at 1.25x realtime
             * (per-frame floor = 4/5 of THIS INPUT's tick). Engaged ONLY while (i) a
             * self-shed/heal happened within the last 10min (g_shed_wall — never stamped on a
             * clean channel, so this path is structurally inert there) AND (ii) video_q still
             * holds >1s of INPUT backlog: the shed stays (it is correct load-shedding), only
             * its consumption aftermath stops arriving as a device-max burst (Phase-A measured
             * p95 2.2x; N co-located churners burst-feed a shared device in phase). Under
             * contention the floor never binds (decode is slower than it); the max sleep per
             * frame is <=0.8 of the input tick at any fps (13ms at 59.94, 32ms at 25), so the
             * heal executor (it runs at packet boundaries) stays reachable. Disengages when
             * the backlog is drained below 1s — the 10min window cannot expire mid-backlog
             * into an ungoverned tail burst. PTV_NO_CATCHGOV reverts.
             * rr10 review fix (D1): the rate currency is the INPUT's real rate, NOT the master
             * output tick — an input whose pps exceeds 1.25x out-fps (mixed-fps mv slots,
             * decimated single inputs at -r below source) was governed BELOW its own arrival
             * rate, so vq re-capped, QSHED re-stamped g_shed_wall, and the governor itself
             * manufactured a permanent shed cycle (rr10b-da fixture: 65 sheds still firing at
             * t=300 vs pre9's 14 ceased at t=134; WUCR railed -14400ppm). Each decode thread
             * (mv slot or single) paces by ITS OWN input: prefer the demux-measured arrival
             * pps (4s gap-guarded window), clamp to the declared header rate while the
             * measurement warms up. If NEITHER is available (VFR source at startup) fail
             * OPEN — an ungoverned catch-up burst is transient and survivable; a governed
             * wedge is permanent. */
            if (g_catchgov && d->live) {
                int64_t gnw = av_gettime_relative();
                int64_t gsw = atomic_load_explicit(&g_shed_wall, memory_order_relaxed);
                int gpps  = d->vin_pps ? atomic_load_explicit(d->vin_pps, memory_order_relaxed) : 0;
                int64_t gpw = d->vin_pps_wall ? atomic_load_explicit(d->vin_pps_wall, memory_order_relaxed) : 0;
                /* 1.0.1-pre13 TRUST GATE (Newsmax2 live defect): govern ONLY on a measurement that is
                 *   (a) >= the declared header rate — a measured rate BELOW declared is a BROKEN
                 *       measurement, not a slow source (the wire is the ground truth the demux rides;
                 *       a clean 59.94pps channel braked to 6.6 dec/s live because a wrong-but-nonzero
                 *       gpps won over the declared clamp — the rr10 re-review A-1 residual). An FFMAX
                 *       floor is NOT enough: declared itself can under-state the wire (29.97-with-
                 *       fields on a 59.94 stream = a brake at 37.5pps). Below declared => DO NOT
                 *       GOVERN: an ungoverned catch-up burst is transient and recoverable, a brake
                 *       that under-paces the wire pins vq full and self-sustains (shed -> engage ->
                 *       starve -> shed) until a human pulls the kill switch;
                 *   (b) FRESH (<30s) — a droughty dispatch pattern stops publishes and a frozen value
                 *       must not keep pacing;
                 * and (c) not in the oversleep holdoff — if the pacing sleeps themselves overshoot
                 *       (throttled/quantized wakeups), the realized brake is stronger than designed
                 *       by an unbounded factor; 3 strikes in 10s fail OPEN for 60s. Warm-up
                 *       (measured==0) no longer clamps to declared: it fails open too — the birth
                 *       backlog burst is exactly the recoverable kind (live 04:21 flag-run birth). */
                int trusted = gpps > 0 && gpps >= d->in_pps_decl &&
                              gpw && gnw - gpw < 30LL * 1000000 &&
                              gnw >= gov_holdoff_until_us;
                if (!d->hold) {                        /* single-input: DIAG t= line telemetry */
                    atomic_store_explicit(&g_gov_gpps, gpps, memory_order_relaxed);
                    atomic_store_explicit(&g_gov_decl, d->in_pps_decl, memory_order_relaxed);
                }
                if (trusted && gsw && gnw - gsw < 600LL * 1000000 &&
                    av_thread_message_queue_nb_elems(d->video_q) > gpps) {
                    int64_t step = 800000 / gpps;       /* 4/5 input tick = 1.25x INPUT realtime */
                    if (g_vindbg && !gov_next_us)
                        av_log(NULL, AV_LOG_INFO, "[PTV-VINDBG] gov ENGAGE gpps=%d decl=%d step=%"PRId64"us vq=%d\n",
                               gpps, d->in_pps_decl, step,
                               av_thread_message_queue_nb_elems(d->video_q));
                    if (!d->hold) atomic_store_explicit(&g_gov_on, 1, memory_order_relaxed);
                    if (gov_next_us > gnw) {
                        int64_t want = gov_next_us - gnw, real;
                        av_usleep((unsigned)want);
                        real = av_gettime_relative();
                        /* pre13 ACTUATOR SELF-CHECK: pacing is only as gentle as the sleeps are
                         * honest. A strike = a sleep that WAS the pacing floor (want >= step/2 —
                         * a tiny residual sleep while decode runs behind schedule is not a brake,
                         * and its overshoot self-corrects via the FFMAX resync) waking >50ms late;
                         * 3 strikes in a rolling 10s => the environment is stretching the brake
                         * (cgroup throttle / scheduler starvation) — fail open 60s rather than
                         * under-pace the wire. */
                        if (want >= step / 2 && real - gnw > want + 50000) {
                            if (real - gov_strike_win_us > 10LL * 1000000) {
                                gov_strike_win_us = real;
                                gov_strikes = 0;
                            }
                            atomic_fetch_add_explicit(&g_gov_slip, 1, memory_order_relaxed);
                            if (++gov_strikes >= 3) {
                                /* 60s, not longer: strikes only occur when decode is FAST but
                                 * wakeups are slow (under real compute contention the floor never
                                 * binds, so there is no sleep to strike on) — that is precisely
                                 * the pathological throttled-wakeup case. If it persists, 3 fresh
                                 * strikes re-arm within ~10s of re-engagement; if it passed, a
                                 * healthy actuator must not stay ungoverned for long. */
                                gov_holdoff_until_us = real + 60LL * 1000000;
                                gov_strikes = 0;
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-CATCHGOV] pacing sleeps overshooting (wanted %"PRId64"ms, got %"PRId64"ms, "
                                       "3 strikes in 10s) — governor fails OPEN for 60s (catch-up unpaced; "
                                       "a stretched brake under-paces the wire and wedges)\n",
                                       want / 1000, (real - gnw) / 1000);
                            }
                        }
                        gnw = gov_next_us;              /* pace on the INTENDED schedule (rate-exact) */
                    }
                    gov_next_us = FFMAX(gnw, gov_next_us) + step;
                } else {
                    if (g_vindbg && gov_next_us)
                        av_log(NULL, AV_LOG_INFO, "[PTV-VINDBG] gov DISENGAGE gpps=%d trusted=%d vq=%d\n",
                               gpps, trusted, av_thread_message_queue_nb_elems(d->video_q));
                    if (!d->hold) atomic_store_explicit(&g_gov_on, 0, memory_order_relaxed);
                    gov_next_us = 0;                    /* incl. untrusted rate: fail open, never wedge */
                }
            }
            d->dec_frames++;
            if (d->hold) stage_hold(d->hold, d->live, frame);   /* multiview: compositor samples this */
            else         emit_video(d, frame, filt);
        }
    }
    /* flush decoder */
    avcodec_send_packet(d->vdec, NULL);
    while (avcodec_receive_frame(d->vdec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        d->dec_frames++;
        if (d->hold) stage_hold(d->hold, d->live, frame);
        else         emit_video(d, frame, filt);
    }
    /* flush filter graph: push EOF into the src, drain every rung's sink */
    if (d->filtering) {
        int fr = av_buffersrc_add_frame(d->fsrc, NULL); (void)fr;
        for (i = 0; i < d->n_rung; i++)
            while (av_buffersink_get_frame(d->fsink[i], filt) >= 0) {
                AVFrame *out = av_frame_alloc();
                if (out) { av_frame_move_ref(out, filt); push_frame_q(d->frame_q[i], ((d->deep_prime_packets > 0 || atomic_load_explicit(&g_bank_pkts, memory_order_relaxed) > 0) && i == 0) ? 0 : d->live, &d->framedrop[i], out); }
                else     { av_frame_unref(filt); }
            }
    }
done:
    av_frame_free(&filt);
    av_frame_free(&frame);
    if (d->hold) {                          /* multiview: signal terminal EOF to the compositor */
        pthread_mutex_lock(&d->hold->lock);
        d->hold->eof = 1;
        pthread_mutex_unlock(&d->hold->lock);
        av_thread_message_queue_set_err_recv(d->hold->q, AVERROR_EOF);   /* drain then EOF the jitter buffer */
    }
    av_thread_message_queue_set_err_send(d->video_q, AVERROR_EOF);   /* unblock demux (a SENDER) */
    for (i = 0; i < d->n_rung; i++)
        av_thread_message_queue_set_err_recv(d->frame_q[i], AVERROR_EOF);  /* EOF to each output (RECEIVER) */
    return NULL;
}

typedef struct MuxArgs {
    AVFormatContext      *ofmt;
    AVThreadMessageQueue *mux_q;
    int                   n_producers;
    int                   err;
    int                   is_master;                  /* Φ1′: rung 0 — compute the wire-DTS sensor here only */
    int                   rung;                       /* pre14: index into g_mux_sent_wc (wire-send watermark) */
} MuxArgs;

static void *mux_thread(void *arg)
{
    MuxArgs *m = arg;
    AVPacket *pkt;
    int done = 0, ret;

    for (;;) {
        ret = av_thread_message_queue_recv(m->mux_q, &pkt, 0);
        if (ret < 0)
            break;
        if (!pkt) {                                  /* end-of-stream marker */
            if (++done >= m->n_producers)
                break;
            continue;
        }
        {
            int64_t wt0 = g_diag ? av_gettime_relative() : 0;
            g_muxed_bytes += pkt->size;
            ret = av_interleaved_write_frame(m->ofmt, pkt);
            if (g_diag) {
                int64_t dlt = av_gettime_relative() - wt0;
                if (dlt > 800000)
                    av_log(NULL, AV_LOG_WARNING, "[PTV-DIAG] write blocked %"PRId64" ms\n", dlt / 1000);
            }
        }
        av_packet_free(&pkt);
        if (ret < 0) { m->err = ret; break; }
        /* pre14 (§3, owner call 3): per-rung wire-send watermark — stamped ONLY after a
         * SUCCESSFUL interleaved write, so a stalled/backed-up muxer (the Newsmax2 dead
         * rung, invisible to every label-domain signal) goes stale within seconds. The
         * corrector's delivery-liveness gate treats this as the primary signal. */
        atomic_store_explicit(&g_mux_sent_wc[m->rung], av_gettime_relative(), memory_order_relaxed);
        g_muxed++;
    }
    av_thread_message_queue_set_err_send(m->mux_q, AVERROR_EOF);   /* unblock producers (SENDERS) */
    return NULL;
}

/* resolved per-output selection from an ffmpeg-style command (see resolve_plan) */
/* One TRANSCODED audio output track (copy audio rides the copy[] passthrough list,
 * NOT this, so it keeps demux_pass's wrap unwrap + monotonic-DTS clamp + SCTE rebase). */
typedef struct AOutSpec {
    int            input;                 /* source input index (multiview -map K:a:N); 0 single-input */
    int            stream;                /* input audio stream index within that input */
    const AVCodec *adec;
    const char    *aenc;                  /* -c:a:N encoder name (NULL = aac) */
    const char    *abr;                   /* -b:a:N */
    const char    *filter;                /* -filter:a:N (NULL = global -af) */
    int            ac;                    /* -ac:a:N output channels (0 = default stereo) */
    const char    *lang;                  /* source language (override -metadata later) */
} AOutSpec;

typedef struct Sel {
    int            have;                  /* 1 if -map present (explicit plan) */
    int            vstream;               /* input video stream to transcode (-1 none) */
    const AVCodec *vdec;
    const char    *venc;                  /* -c:v encoder name (NULL = default) */
    const char    *vf;                    /* -filter:v / -vf */
    const char    *vbr;                   /* -b:v */
    AOutSpec       aout[PTV_MAX_AUDIO];   /* transcoded audio output tracks */
    int            n_aout;
    int            copy[PTV_MAX_PASS];    /* ALL copy: audio (5.1/2ch) + sub + data + scte */
    int            copy_input[PTV_MAX_PASS]; /* source input index per copy stream (multiview) */
    int            n_copy;
} Sel;
static const char *og_get(OptionGroup *g, const char *key);
static void apply_stream_meta(OptionGroup *g, char t, int idx, AVStream *ost);
struct Input;
static int resolve_plan(struct Input *inputs, int n_input, OptionGroup *outg, Sel *s);

/* One ABR ladder rung = one output: its own muxer, video encoder, queues and
 * threads. Audio + passthrough are shared (decoded/copied once, fanned out). */
typedef struct Rung {
    AVFormatContext *ofmt;
    AVCodecContext  *venc;
    AVThreadMessageQueue *frame_q, *mux_q;
    DlvGate          gate;                       /* §7.5a delivery-alignment FIFO (per rung) */
    VideoCtx         vc;
    MuxArgs          ma;
    AVBufferRef     *fhwfr;
    int              fw, fh, fpix;
    char             vlabel[64];                 /* filter output label, ladder only */
    pthread_t        th_output, th_mux, th_wd;
    int              started_output, started_mux, started_wd, hdr_written;
} Rung;

/* open one input on its own thread (parallel open: a dead/slow slot must not
 * delay the others, and serial open would block on its long rw_timeout). */
typedef struct OpenArg { Input *in; AVDictionary **opts; } OpenArg;
static void *open_input_thread(void *arg)
{
    OpenArg *o = arg; Input *in = o->in;
    in->open_ret = avformat_open_input(&in->ifmt, in->url, NULL, o->opts);
    if (in->open_ret >= 0) in->open_ret = avformat_find_stream_info(in->ifmt, NULL);
    return NULL;
}

static int is_net_url(const char *u)
{
    return u && (!strncmp(u, "udp://", 6) || !strncmp(u, "rtp://", 6) || !strncmp(u, "srt://", 6));
}

/* transcode: ins = parsed input group list (1/2/4 inputs; >1 = multiview);
 * outs = the list of output groups (one per ABR rung); fcomplex = the shared
 * -filter_complex. The ffmpeg model — decode each input once, one filter graph
 * (single-input split, or N-input mosaic+split) feeds each rung's independent
 * muxer/encoder; audio + subs/data decoded/copied once and fanned out.
 * Selection (transcode vs copy) per group comes from its -map/-c. */
static int transcode(OptionGroupList *ins, OptionGroupList *outs, const char *fcomplex,
                     const char *hwdev, int mode)
{
    int n_input = ins->nb_groups;
    int n_rung = outs->nb_groups;
    int multiview;
    int delivery_on = 0;                          /* §7.5a delivery gate active for this run */
    Input            inputs[PTV_MAX_INPUT];
    AVCodecContext  *vdecs[PTV_MAX_INPUT];
    AVFilterContext *fsrc[PTV_MAX_INPUT] = {0};
    AVThreadMessageQueue *audio_q[PTV_MAX_AUDIO] = {0};
    AVFilterGraph   *fg = NULL;                 /* the shared graph (single or multiview) */
    AVFilterContext **vsink = NULL;            /* per-rung buffersinks (single dc / multi comp) */
    int              filtering = 0;
    AVBufferRef     *hw_device = NULL;
    AudioState       as[PTV_MAX_AUDIO];        /* one per transcoded audio track */
    int              asrc[PTV_MAX_AUDIO];      /* input-local stream feeding each as[] */
    int              asrc_in[PTV_MAX_AUDIO];   /* source input index feeding each as[] */
    int              n_audio = 0;
    Rung             rung[PTV_MAX_RUNG];
    Sel              sel[PTV_MAX_RUNG];
    CompositorCtx    comp;
    HouseRateState   house_rate;               /* 0.9.18 R4: one per house clock, shared by the rung set (via VideoCtx.hr) */
    pthread_t        th_compositor, th_audio[PTV_MAX_AUDIO];
    int              started_compositor = 0;
    int              started_audio[PTV_MAX_AUDIO] = {0};
    int ret = 0, live, net_input, have_audio = 0, hw_cuda = 0;
    int aborted = 0, r, si, k, kk, n_copy_inputs = 0;
    AVRational out_fps;
    PassStream pass[PTV_MAX_PASS]; int n_pass = 0;
    /* aliases to input 0 (the shared single-input setup code works on it) */
    AVCodecContext *vdec; AVStream *vist; int vstream; const AVCodec *vdecoder;

    if (n_input < 1) { av_log(NULL, AV_LOG_ERROR, "no input\n"); return AVERROR(EINVAL); }
    if (n_input > PTV_MAX_INPUT || n_input == 3) {
        av_log(NULL, AV_LOG_ERROR, "multiview supports 1, 2 or 4 inputs (got %d)\n", n_input);
        return AVERROR(EINVAL);
    }
    multiview = n_input > 1;
    if (multiview && !fcomplex) {
        av_log(NULL, AV_LOG_ERROR, "multiview (%d inputs) requires -filter_complex (mosaic graph)\n", n_input);
        return AVERROR(EINVAL);
    }
    if (g_degraded && multiview) {
        /* rr10 review fix (D2): PTV_DEGRADED is SINGLE-INPUT ONLY. On multiview the entry's
         * backlog flush (g_selfheal_req) has NO consumer — mv slot decode threads run with
         * d->hold and never service the re-prime (decode_thread guards it with !d->hold), so
         * the flag would sit armed forever and degraded admission would ride the stale
         * backlog it was designed to shed. And the release headroom reads g_frameq_depth =
         * the COMPOSITE frame_q, which the compositor keeps fed from held frames regardless
         * of this slot's decode health — the wrong signal for slot decode headroom. Hard
         * disable with a loud log; making mv-degraded work is a separate feature. */
        g_degraded = 0;
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-DEGRADED] PTV_DEGRADED is single-input only — disabled on multiview "
               "(%d inputs): the entry flush has no decode-side consumer on mv slots and the "
               "release signal is the composite frame_q\n", n_input);
    }
    if (n_rung > PTV_MAX_RUNG) {
        av_log(NULL, AV_LOG_WARNING, "%d outputs > max %d; using the first %d\n", n_rung, PTV_MAX_RUNG, PTV_MAX_RUNG);
        n_rung = PTV_MAX_RUNG;
    }
    memset(inputs, 0, sizeof inputs); memset(as, 0, sizeof as); memset(rung, 0, sizeof rung);
    memset(&comp, 0, sizeof comp); memset(&house_rate, 0, sizeof house_rate);
    for (k = 0; k < n_input; k++) {
        inputs[k].url = ins->groups[k].arg;
        inputs[k].h0  = AV_NOPTS_VALUE;
        /* rate estimator: the former singleton's initializers, verbatim (rest is zero) */
        inputs[k].est.c0 = AV_NOPTS_VALUE;
        inputs[k].est.ema_q20 = 1 << 20;
        inputs[k].est.ep_prev = -1;
        inputs[k].est.cf_ema_q20 = 1 << 20;
        inputs[k].est.src_rate_q20 = 1 << 20;
        inputs[k].est.cf_rate_q20 = 1 << 20;
        pthread_mutex_init(&inputs[k].h0_lock, NULL);
        pthread_mutex_init(&inputs[k].hold.lock, NULL);
        pthread_mutex_init(&inputs[k].vring.lock, NULL);
        if (multiview) {                         /* per-input jitter buffer for the compositor */
            if ((ret = av_thread_message_queue_alloc(&inputs[k].hold.q, g_frameq_cap, sizeof(AVFrame *))) < 0) goto end;
            av_thread_message_queue_set_free_func(inputs[k].hold.q, free_frame_msg);
        }
        /* Take raw 33-bit timestamps; demux_unwrap extends them (libav's
         * correct_ts_overflow extends inconsistently across the B-frame reorder). */
        av_dict_set(&ins->groups[k].format_opts, "correct_ts_overflow", "0", AV_DICT_DONT_OVERWRITE);
    }

    /* open ALL inputs in parallel: a dead/slow slot must not delay the others,
     * and a serial open would block on its (long, multiview) rw_timeout. */
    {
        OpenArg oa[PTV_MAX_INPUT];
        pthread_t th[PTV_MAX_INPUT];
        int started[PTV_MAX_INPUT] = {0};
        for (k = 0; k < n_input; k++) {
            oa[k].in = &inputs[k]; oa[k].opts = &ins->groups[k].format_opts;
            if (pthread_create(&th[k], NULL, open_input_thread, &oa[k]) == 0) started[k] = 1;
            else { inputs[k].open_ret = AVERROR(errno); }
        }
        for (k = 0; k < n_input; k++) if (started[k]) pthread_join(th[k], NULL);
        for (k = 0; k < n_input; k++) {
            if (inputs[k].open_ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "cannot open input %d '%s': %s\n",
                       k, inputs[k].url, av_err2str(inputs[k].open_ret));
                ret = inputs[k].open_ret; goto end;
            }
            av_dump_format(inputs[k].ifmt, k, inputs[k].url, 0);
        }
    }

    /* resolve each output group's -map/-c into its transcode/copy selection */
    for (r = 0; r < n_rung; r++)
        if ((ret = resolve_plan(inputs, n_input, &outs->groups[r], &sel[r])) < 0) goto end;

    /* per-input video decoder. Ladder/mosaic rungs map filter labels [vN]; each
     * input's source video is the best video stream (single-input may also map an
     * input video directly via -map). */
    for (k = 0; k < n_input; k++) {
        int vs; const AVCodec *vd = NULL;
        if (k == 0 && sel[0].vstream >= 0) { vs = sel[0].vstream; vd = sel[0].vdec; }
        else vs = av_find_best_stream(inputs[k].ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &vd, 0);
        if (vs < 0 || !vd) { av_log(NULL, AV_LOG_ERROR, "no video stream in input %d\n", k); ret = AVERROR(EINVAL); goto end; }
        inputs[k].vstream  = vs;
        inputs[k].vdecoder = vd;
        inputs[k].vist     = inputs[k].ifmt->streams[vs];
        inputs[k].ist_tb   = inputs[k].vist->time_base;
        inputs[k].vdec = avcodec_alloc_context3(vd);
        if (!inputs[k].vdec) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_to_context(inputs[k].vdec, inputs[k].vist->codecpar);
        inputs[k].vdec->pkt_timebase = inputs[k].ist_tb;
        /* NOTE: keep single-threaded decode — frame-threaded hangs offline at EOF. */
        if ((ret = avcodec_open2(inputs[k].vdec, vd, NULL)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "open video decoder (input %d): %s\n", k, av_err2str(ret)); goto end;
        }
        vdecs[k] = inputs[k].vdec;
        inputs[k].wrap_off  = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_off));
        inputs[k].wrap_last = av_malloc_array(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_last));
        inputs[k].wrap_wall_last = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_wall_last)); /* 0 = no prev packet yet */
        inputs[k].edit_us   = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].edit_us));   /* pre9 sensor label-edit ledger */
        if (!inputs[k].wrap_off || !inputs[k].wrap_last || !inputs[k].wrap_wall_last || !inputs[k].edit_us) { ret = AVERROR(ENOMEM); goto end; }
        for (si = 0; si < (int)inputs[k].ifmt->nb_streams; si++) inputs[k].wrap_last[si] = AV_NOPTS_VALUE;
        if (g_layera) {   /* legacy-0004 buffer-classify-discard state (only when enabled) */
            if ((ret = ptv_disc_init(&inputs[k].disc, PTV_DISC_CAPACITY,
                                     inputs[k].ifmt->nb_streams)) < 0) goto end;
        }
    }
    vdec = inputs[0].vdec; vist = inputs[0].vist;
    vstream = inputs[0].vstream; vdecoder = inputs[0].vdecoder; (void)vstream;

    /* house rate: -r on the first output, else preserve the source's actual FRAME
     * rate. Prefer avg_frame_rate over r_frame_rate: for an interlaced source
     * r_frame_rate is the FIELD rate (e.g. 1080i25 -> 50), but the decoder/
     * deinterlacer emits one frame per coded frame (25), so a 50-fps house clock
     * would tick twice per delivered frame -> ~50% duplicates (judder). Force a
     * specific rate (incl. field-doubling deint) with -r. */
    {
        const char *rate_str = og_get(&outs->groups[0], "r");
        if (rate_str) {
            if (av_parse_video_rate(&out_fps, rate_str) < 0 || out_fps.num <= 0) {
                av_log(NULL, AV_LOG_ERROR, "bad -r '%s'\n", rate_str); ret = AVERROR(EINVAL); goto end;
            }
        } else {
            out_fps = vist->avg_frame_rate.num ? vist->avg_frame_rate
                    : vist->r_frame_rate.num ? vist->r_frame_rate : (AVRational){25, 1};
        }
    }

    /* CUDA backend when the filter graph targets it (filter_complex or single -vf).
     * ONE device is created and set on every hw filter (hwupload/bwdif/scale_cuda);
     * NVENC inherits it via the filtered frames' hw_frames_ctx — so a single GPU
     * ordinal drives the whole chain. Selected ffmpeg-style with
     * `-init_hw_device cuda=cuda:N` (the device part after the last ':'); default 0. */
    hw_cuda = (fcomplex && (strstr(fcomplex, "_cuda") || strstr(fcomplex, "hwupload_cuda"))) ||
              (sel[0].vf && (strstr(sel[0].vf, "_cuda") || strstr(sel[0].vf, "hwupload_cuda")));
    if (hw_cuda) {
        const char *cuda_ord = NULL; char ordbuf[64];
        if (hwdev) {
            const char *c = strrchr(hwdev, ':');         /* cuda=cuda:N / cuda:N -> "N" */
            if (c && c[1]) {
                char *comma;
                snprintf(ordbuf, sizeof ordbuf, "%s", c + 1);
                if ((comma = strchr(ordbuf, ','))) *comma = 0;   /* drop trailing ,opts */
                if (ordbuf[0]) cuda_ord = ordbuf;
            }
        }
        if ((ret = av_hwdevice_ctx_create(&hw_device, AV_HWDEVICE_TYPE_CUDA, cuda_ord, NULL, 0)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "cannot create CUDA device '%s': %s\n",
                   cuda_ord ? cuda_ord : "0", av_err2str(ret)); goto end;
        }
        av_log(NULL, AV_LOG_INFO,
               "ptvencoder: CUDA device %s (hwupload + deint + scale + nvenc all share it)\n",
               cuda_ord ? cuda_ord : "0");
    }

    /* shared filter graph: -filter_complex (N video inputs -> split/mosaic ->
     * N sinks), a single -filter:v chain (single-input N==1), or none (clone the
     * decoded frame to each rung). For multiview the compositor owns the graph
     * and feeds N buffersrcs at pts = house tick (src_tb = 1/out_fps); for
     * single-input the decode thread feeds the one buffersrc with source pts. */
    inputs[0].dc.n_rung = n_rung;
    if (fcomplex) {
        const char *labels[PTV_MAX_RUNG];
        AVRational src_tb = multiview ? av_inv_q(out_fps) : inputs[0].ist_tb;
        AVFilterGraph **pfg = multiview ? &fg : &inputs[0].dc.fg;
        AVFilterContext **psinks = multiview ? comp.fsink : inputs[0].dc.fsink;
        for (r = 0; r < n_rung; r++) {
            OptionGroup *g = &outs->groups[r];
            const char *lab = NULL; int o; size_t L;
            for (o = 0; o < g->nb_opts; o++)
                if (!strcmp(g->opts[o].key, "map") && g->opts[o].val[0] == '[') { lab = g->opts[o].val; break; }
            if (!lab) { av_log(NULL, AV_LOG_ERROR, "output %d has no -map [label] for filter_complex\n", r);
                        ret = AVERROR(EINVAL); goto end; }
            snprintf(rung[r].vlabel, sizeof rung[r].vlabel, "%s", lab + 1);   /* drop '[' */
            L = strlen(rung[r].vlabel);
            if (L && rung[r].vlabel[L-1] == ']') rung[r].vlabel[L-1] = 0;      /* drop ']' */
            labels[r] = rung[r].vlabel;
        }
        if ((ret = build_filter_complex(fcomplex, vdecs, n_input, src_tb, hw_device,
                                        labels, n_rung, pfg, fsrc, psinks)) < 0) goto end;
        filtering = 1;
        if (multiview) { vsink = comp.fsink; }
        else { inputs[0].dc.fsrc = fsrc[0]; inputs[0].dc.filtering = 1; vsink = inputs[0].dc.fsink; }
    } else if (n_rung == 1 && sel[0].vf) {
        int fw = 0, fh = 0, fpix = AV_PIX_FMT_NONE; AVBufferRef *hf = NULL;
        if ((ret = build_video_filter(&inputs[0].dc, vdec, inputs[0].ist_tb, sel[0].vf, 0, 0, 0,
                                      hw_cuda, hw_device, &fw, &fh, &fpix, &hf)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "build video filter: %s\n", av_err2str(ret)); goto end;
        }
        av_buffer_unref(&hf);
        filtering = inputs[0].dc.filtering; vsink = inputs[0].dc.fsink;
    }   /* else: filtering stays 0 -> clone the decoded frame to each rung */

    /* per-rung video encoder, sized from this rung's sink (or the decoder) */
    for (r = 0; r < n_rung; r++) {
        OptionGroup *g = &outs->groups[r];
        const char *out_url = g->arg, *out_fmt = og_get(g, "f");
        const char *venc_name = sel[r].venc ? sel[r].venc : "h264_videotoolbox";
        const AVCodec *vencoder;

        if (filtering) {
            rung[r].fw   = av_buffersink_get_w(vsink[r]);
            rung[r].fh   = av_buffersink_get_h(vsink[r]);
            rung[r].fpix = av_buffersink_get_format(vsink[r]);
            { AVBufferRef *hf = av_buffersink_get_hw_frames_ctx(vsink[r]);
              rung[r].fhwfr = hf ? av_buffer_ref(hf) : NULL; }
        } else {
            rung[r].fw = vdec->width; rung[r].fh = vdec->height;
            rung[r].fpix = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
        }

        ret = avformat_alloc_output_context2(&rung[r].ofmt, NULL, out_fmt, out_url);
        if (ret < 0 && !out_fmt)   /* udp://, srt://: no extension to guess from */
            ret = avformat_alloc_output_context2(&rung[r].ofmt, NULL, "mpegts", out_url);
        if (ret < 0) { av_log(NULL, AV_LOG_ERROR, "output ctx '%s': %s (try -f mpegts)\n", out_url, av_err2str(ret)); goto end; }

        vencoder = avcodec_find_encoder_by_name(venc_name);
        if (!vencoder) { av_log(NULL, AV_LOG_WARNING, "encoder '%s' not found, using mpeg2video\n", venc_name);
                         vencoder = avcodec_find_encoder_by_name("mpeg2video"); }
        if (!vencoder) { ret = AVERROR_ENCODER_NOT_FOUND; goto end; }

        rung[r].venc = avcodec_alloc_context3(vencoder);
        if (!rung[r].venc) { ret = AVERROR(ENOMEM); goto end; }
        rung[r].venc->width = rung[r].fw; rung[r].venc->height = rung[r].fh; rung[r].venc->pix_fmt = rung[r].fpix;
        if (rung[r].fhwfr) rung[r].venc->hw_frames_ctx = av_buffer_ref(rung[r].fhwfr);   /* CUDA frames -> NVENC */
        rung[r].venc->time_base = av_inv_q(out_fps); rung[r].venc->framerate = out_fps;
        rung[r].venc->bit_rate = 3000000;
        rung[r].venc->gop_size = 2 * (out_fps.num / FFMAX(out_fps.den, 1));
        if (rung[r].ofmt->oformat->flags & AVFMT_GLOBALHEADER) rung[r].venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        {   /* -b:v + forwarded encoder opts (-preset/-rc/-maxrate/-g/-s12m_tc/...) */
            AVDictionary *vopts = NULL;
            av_dict_copy(&vopts, g->codec_opts, 0);
            if (sel[r].vbr) av_dict_set(&vopts, "b", sel[r].vbr, 0);
            ret = avcodec_open2(rung[r].venc, vencoder, &vopts);
            av_dict_free(&vopts);
            if (ret < 0) { av_log(NULL, AV_LOG_ERROR, "open video encoder '%s' (output %d): %s\n", vencoder->name, r, av_err2str(ret)); goto end; }
        }
        rung[r].vc.ost = avformat_new_stream(rung[r].ofmt, NULL);
        if (!rung[r].vc.ost) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_from_context(rung[r].vc.ost->codecpar, rung[r].venc);
        rung[r].vc.ost->time_base = rung[r].venc->time_base;
    }

    /* shared audio: for EACH transcoded audio track (sel[0].aout[]), decode +
     * loudness-filter ONCE, then encode PER RUNG (per-rung -b:a) into an output
     * audio stream in EACH muxer. Source stream/decoder/codec from the first
     * rung's plan; per-rung -b:a from each rung. Audio COPY (AC-3 5.1, 2ch) rides
     * the passthrough list below (keeping demux_pass's wrap unwrap + DTS clamp). */
    for (k = 0; k < sel[0].n_aout && n_audio < PTV_MAX_AUDIO; k++) {
        AOutSpec      *spec = &sel[0].aout[k];
        AVStream      *kist = inputs[spec->input].ifmt->streams[spec->stream];
        const AVCodec *kdecoder = spec->adec ? spec->adec
                                : avcodec_find_decoder(kist->codecpar->codec_id);
        AVCodecContext *kdec, *encs[PTV_MAX_RUNG] = {0};
        AudioState    *a;
        enum AVSampleFormat sfmt;
        const char    *af;
        AVChannelLayout ochl;
        int            nch = spec->ac > 0 ? spec->ac : 2;   /* -ac:a:N output channels (default stereo) */
        int            eok = 1;

        if (!kdecoder) { av_log(NULL, AV_LOG_WARNING, "audio track %d (stream %d): no decoder; skipped\n", k, spec->stream); continue; }
        av_channel_layout_default(&ochl, nch);              /* 2->stereo, 6->5.1, 1->mono */
        kdec = avcodec_alloc_context3(kdecoder);
        if (!kdec) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_to_context(kdec, kist->codecpar);
        kdec->pkt_timebase = kist->time_base;
        if (avcodec_open2(kdec, kdecoder, NULL) < 0) {
            av_log(NULL, AV_LOG_WARNING, "audio track %d decoder failed; skipped\n", k);
            avcodec_free_context(&kdec); continue;
        }
        /* open all rung encoders into a temp array FIRST — so a mid-rung failure
         * leaves no orphan output streams in the earlier muxers. */
        for (r = 0; r < n_rung; r++) {
            const AVCodec *aenc = avcodec_find_encoder_by_name(spec->aenc ? spec->aenc : "aac");
            const char *abr = (k < sel[r].n_aout) ? sel[r].aout[k].abr : NULL;
            AVDictionary *aopts = NULL; AVCodecContext *e;
            if (!aenc) aenc = avcodec_find_encoder_by_name("aac");
            e = avcodec_alloc_context3(aenc);
            if (!e) { ret = AVERROR(ENOMEM); avcodec_free_context(&kdec); for (si = 0; si < r; si++) avcodec_free_context(&encs[si]); goto end; }
            e->sample_rate = 48000;
            e->ch_layout   = ochl;                          /* from -ac:a:N (stereo / 5.1 / …) */
            {   /* pick the encoder's first supported sample format. Use avcodec_get_supported_config()
                 * (the AVCodec.sample_fmts field is deprecated and REMOVED on current upstream — the
                 * build box clones fresh upstream, so the old field breaks the static Linux build). */
                const enum AVSampleFormat *sfmts = NULL;
                e->sample_fmt = (avcodec_get_supported_config(NULL, aenc, AV_CODEC_CONFIG_SAMPLE_FORMAT,
                                                              0, (const void **)&sfmts, NULL) >= 0
                                 && sfmts && sfmts[0] != AV_SAMPLE_FMT_NONE)
                              ? sfmts[0] : AV_SAMPLE_FMT_FLTP;
            }
            e->bit_rate    = 160000;
            e->time_base   = (AVRational){1, 48000};
            if (rung[r].ofmt->oformat->flags & AVFMT_GLOBALHEADER) e->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            if (abr) av_dict_set(&aopts, "b", abr, 0);
            if (avcodec_open2(e, aenc, &aopts) < 0) {
                av_log(NULL, AV_LOG_WARNING, "audio track %d encoder (rung %d) failed; track skipped\n", k, r);
                avcodec_free_context(&e); av_dict_free(&aopts); eok = 0; break;
            }
            av_dict_free(&aopts);
            encs[r] = e;
        }
        if (!eok) { for (r = 0; r < n_rung; r++) avcodec_free_context(&encs[r]); avcodec_free_context(&kdec); continue; }

        /* commit: this is audio track n_audio (dense). Add an output stream per rung. */
        a = &as[n_audio];
        a->dec = kdec;
        for (r = 0; r < n_rung; r++) {
            AVStream *aos; AVDictionaryEntry *klang;
            a->enc[r] = encs[r];
            aos = avformat_new_stream(rung[r].ofmt, NULL);
            if (!aos) { ret = AVERROR(ENOMEM); goto end; }
            avcodec_parameters_from_context(aos->codecpar, encs[r]);
            aos->time_base = encs[r]->time_base;
            if ((klang = av_dict_get(kist->metadata, "language", NULL, 0)))
                av_dict_set(&aos->metadata, "language", klang->value, 0);
            apply_stream_meta(&outs->groups[r], 'a', k, aos);   /* CLI -metadata:s:a:N / -disposition:a:N (G5) */
            a->ost[r] = aos;
        }
        sfmt = a->enc[0]->sample_fmt;
        af   = spec->filter;                              /* -filter:a:N (per-track); else global -af */
        if (!af) { af = og_get(&outs->groups[0], "af"); if (!af) af = og_get(&outs->groups[0], "filter:a"); }
        /* No -af: still route through aresample=async so the audio rides the HOUSE
         * clock (raw swr is identity 48k->48k with no resampler to stretch -> drifts).
         * The common-mode house-skew in audio_push then keeps A/V locked. */
        if (!af) af = "aresample=async=1000";
        a->out_chl    = ochl;                             /* from -ac:a:N (stereo / 5.1 / …) */
        a->out_rate   = 48000;
        a->out_sfmt   = sfmt;
        a->n_out      = n_rung;
        a->frame_size = a->enc[0]->frame_size > 0 ? a->enc[0]->frame_size : 1024;
        a->fifo       = av_audio_fifo_alloc(sfmt, 2, a->frame_size);
        if (!a->fifo) { ret = AVERROR(ENOMEM); goto end; }   /* used by the swr fallback path */
        a->ist_tb     = kist->time_base;
        a->ist        = kist;          /* 1.0.1: codecpar source for the decode-death watchdog reopen */
        if (af && build_audio_filter(a, a->dec, kist->time_base, af, sfmt) < 0) {
            av_log(NULL, AV_LOG_WARNING, "audio track %d filtergraph failed; plain resample\n", k);
            avfilter_graph_free(&a->afg); a->use_fg = 0;
        }
        if (!a->use_fg) {              /* no -af (or graph failed): plain resample */
            swr_alloc_set_opts2(&a->swr, &a->out_chl, sfmt, 48000,
                                &a->dec->ch_layout, a->dec->sample_fmt, a->dec->sample_rate, 0, NULL);
            if (!a->swr || swr_init(a->swr) < 0) {
                if (a->swr) swr_free(&a->swr);
                av_log(NULL, AV_LOG_WARNING,
                       "audio track %d path init failed (source audio undecodable/params unknown at open?) — "
                       "will rebuild from the first cleanly decoded frames [PTV-AFMT]\n", k);
            }
        }
        a->fg_af      = af;            /* remember the chain so audio_feed can rebuild on a source format change */
        /* v0.9.17.1: seed the tracked input params — but if the path FAILED to init (dead source
         * phase at open: Azorse's broken 7.1-signaled AAC decoded nothing → dec params garbage),
         * seed an IMPOSSIBLE rate instead so the first cleanly decoded frame is guaranteed to
         * differ and route through the [PTV-AFMT] hysteresis+rebuild — the skip is self-healing
         * once the source recovers (ffmpeg-parity behavior) instead of permanent. */
        if (a->use_fg || a->swr) {
            a->fg_in_rate = a->dec->sample_rate;   /* live path: seed real params (no false trigger on frame 1) */
            a->fg_in_fmt  = a->dec->sample_fmt;
            av_channel_layout_copy(&a->fg_in_chl, &a->dec->ch_layout);
        } else {
            a->fg_in_rate = -1;                    /* dead path: force AFMT rebuild on first good frame */
            a->fg_in_fmt  = AV_SAMPLE_FMT_NONE;
        }
        asrc[n_audio]    = spec->stream;
        asrc_in[n_audio] = spec->input;
        n_audio++;
    }
    have_audio = n_audio > 0;

    /* shared passthrough (copy): each non-transcoded input stream — extra audio
     * (AC-3 5.1), DVB subtitles, data/SCTE-35 — gets an output stream in EVERY
     * muxer; the copied packets fan out. Built GROUPED BY INPUT so each input's
     * demux gets a contiguous pass[] slice (inputs[kk].da.pass). Created before
     * the headers are written. */
    /* per-type output stream-index counters for the CLI -metadata:s:<t>:N / -disposition:<t>:N
     * specifiers on COPY streams. Seeded past the transcoded streams of each type so the index
     * matches the muxer's per-type numbering: each output has 1 composite video (v:0) + n_audio
     * transcoded audio (a:0..n_audio-1); ptvencoder never transcodes subtitles/data, so those
     * start at 0. Incremented in stream-CREATION order (= FFmpeg's -metadata:s:s:N order). */
    int copy_vidx = 1, copy_aidx = n_audio, copy_sidx = 0, copy_didx = 0;
    for (kk = 0; kk < n_input; kk++) {
        inputs[kk].da.pass = &pass[n_pass];          /* this input's contiguous slice */
        inputs[kk].da.n_pass = 0;
        for (si = 0; si < sel[0].n_copy && n_pass < PTV_MAX_PASS; si++) {
            int sidx, tidx; char tlet;
            AVStream *ist; AVDictionaryEntry *lang;
            if (sel[0].copy_input[si] != kk) continue;
            sidx = sel[0].copy[si];
            ist  = inputs[kk].ifmt->streams[sidx];
            lang = av_dict_get(ist->metadata, "language", NULL, 0);
            switch (ist->codecpar->codec_type) {     /* type specifier + per-type output index */
                case AVMEDIA_TYPE_AUDIO:    tlet = 'a'; tidx = copy_aidx++; break;
                case AVMEDIA_TYPE_SUBTITLE: tlet = 's'; tidx = copy_sidx++; break;
                case AVMEDIA_TYPE_VIDEO:    tlet = 'v'; tidx = copy_vidx++; break;
                default:                    tlet = 'd'; tidx = copy_didx++; break;
            }
            pass[n_pass].input    = kk;
            pass[n_pass].in_index = sidx;
            pass[n_pass].in_tb    = ist->time_base;
            pass[n_pass].last_dts = AV_NOPTS_VALUE;
            pass[n_pass].gated    = (ist->codecpar->codec_type == AVMEDIA_TYPE_AUDIO);  /* §7.5a: dense AC-3/MP2 ride the gate; sparse subs/data/SCTE-35 bypass */
            for (r = 0; r < n_rung; r++) {
                AVStream *os = avformat_new_stream(rung[r].ofmt, NULL);
                if (!os) { ret = AVERROR(ENOMEM); goto end; }
                if ((ret = avcodec_parameters_copy(os->codecpar, ist->codecpar)) < 0) goto end;
                os->codecpar->codec_tag = 0;
                os->time_base   = ist->time_base;
                os->disposition = ist->disposition;
                if (lang) av_dict_set(&os->metadata, "language", lang->value, 0);
                apply_stream_meta(&outs->groups[r], tlet, tidx, os);  /* CLI -metadata:s:<t>:N / -disposition (G5) — copy streams (subs/data/extra-audio) */
                pass[n_pass].ost[r] = os;
            }
            n_pass++;
            inputs[kk].da.n_pass++;
        }
        if (inputs[kk].da.n_pass > 0) n_copy_inputs++;
    }
    if (n_pass)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: passthrough %d stream(s) per output (copy), %d input(s)\n",
               n_pass, n_copy_inputs);

    /* per-rung: open the output, bound the interleave (sparse-sub smoothing),
     * apply file -metadata, write the header. */
    for (r = 0; r < n_rung; r++) {
        OptionGroup *g = &outs->groups[r];
        const char *out_url = g->arg;
        if (!(rung[r].ofmt->oformat->flags & AVFMT_NOFILE))
            if ((ret = avio_open(&rung[r].ofmt->pb, out_url, AVIO_FLAG_WRITE)) < 0) {
                av_log(NULL, AV_LOG_ERROR, "open output '%s': %s\n", out_url, av_err2str(ret)); goto end;
            }
        rung[r].ofmt->max_interleave_delta = 200000;   /* 200 ms */
        {   /* forwarded muxer opts (-mpegts_flags/-pat_period/-pcr_period/...) + file -metadata */
            AVDictionary *mopts = NULL; int mi;
            av_dict_copy(&mopts, g->format_opts, 0);
            for (mi = 0; mi < g->nb_opts; mi++) {
                char kv[256], *eq;            /* -metadata service_name=CineStar (file-level) */
                if (strcmp(g->opts[mi].key, "metadata")) continue;
                snprintf(kv, sizeof kv, "%s", g->opts[mi].val);
                if ((eq = strchr(kv, '='))) { *eq = 0; av_dict_set(&rung[r].ofmt->metadata, kv, eq + 1, 0); }
            }
            ret = avformat_write_header(rung[r].ofmt, &mopts);
            av_dict_free(&mopts);
        }
        if (ret < 0) { av_log(NULL, AV_LOG_ERROR, "write header (output %d): %s\n", r, av_err2str(ret)); goto end; }
        rung[r].hdr_written = 1;
        av_dump_format(rung[r].ofmt, r, out_url, 1);   /* ffmpeg-style "Output #r ..." */
    }

    net_input = is_net_url(inputs[0].url);
    live = mode < 0 ? net_input : mode;

    /* 0.9.18 M1: resolve ALL cushion/queue sizing in one place (env parses + genlock default +
     * deep-prime side-cars + per-track audio depth + deep-prime target). Writes the same g_*
     * globals as before — runtime consumers (BANK escalate/decay, adaptive GROW/SHRINK, the mv
     * compositor, stats/logs) still read them — and mirrors everything into g_cp. Must run
     * before the first consuming allocation below and before any thread starts. */
    resolve_cushions(&g_cp, live, multiview, out_fps, n_audio);

    /* queues: per-input video_q, one audio_q per transcoded track, per-rung frame_q + mux_q */
    for (k = 0; k < n_input; k++) {
        if ((ret = av_thread_message_queue_alloc(&inputs[k].video_q, g_cp.videoq_pkts, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(inputs[k].video_q, free_pkt_msg);
    }
    for (k = 0; k < n_audio; k++) {
        /* per-track depth rule (§13 deep-prime + v0.9.14.2 bank-ceiling sizing) moved to
         * resolve_cushions() (0.9.18 M1) */
        if ((ret = av_thread_message_queue_alloc(&audio_q[k], g_cp.audioq_pkts, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(audio_q[k], free_pkt_msg);
    }
    for (r = 0; r < n_rung; r++) {
        if ((ret = av_thread_message_queue_alloc(&rung[r].frame_q, g_cp.frameq_cap, sizeof(AVFrame *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].frame_q, free_frame_msg);
        if ((ret = av_thread_message_queue_alloc(&rung[r].mux_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].mux_q, free_pkt_msg);
    }
    /* §7.5a delivery-alignment gate: default ON for LIVE, single-input AND (since v0.9.12.1)
     * multiview (PTV_NO_DELIVERY / PTV_NO_DELIVERY_MV revert). Offline always bypasses →
     * byte-identical. Every transcoded audio track fans into EVERY rung's gate, so size the
     * hold-FIFO backstop by the track count (a 2x2 mosaic = 4 AAC x ~47pkt/s x 3s cap ≈ 570 >
     * the 512 single-input default; FFMAX only raises, nodes are per-enqueue not preallocated;
     * an explicit PTV_DELIVERY_MAXQ still wins). */
    delivery_on = live && g_delivery && (!multiview || g_delivery_mv);
    if (delivery_on) {
        int maxq = g_cp.delivery_maxq;
        if (!getenv("PTV_DELIVERY_MAXQ"))
            maxq = FFMAX(maxq, n_audio * 50 *
                         (int)(g_cp.delivery_cap_us / 1000000 + g_cushion_max_ms / 1000 + 5));
            /* per gated track: ~50 pkt/s x (gate cap + bank ceiling + clump surge headroom) —
             * the same sizing the deep-preroll path applies, computed automatically (v0.9.14.1) */
        for (r = 0; r < n_rung; r++)
            dlv_init(&rung[r].gate, rung[r].mux_q, g_cp.delivery_cap_us, maxq);
        /* §7.5b (1.0.1-pre12) symmetric gate: arm the EARLY-VIDEO hold, keyed on the audio
         * delivered high-water. Single-input only — multiview slots' audio share ONE gate per
         * rung, so the high-water would key the hold to the LEAST-delayed slot (untested
         * per-slot death semantics on top) — and only when the run HAS gated audio to key on
         * (a transcoded track or dense copied AC-3/MP2): a no-audio channel must not pay the
         * audio-death escape timeout at birth. PTV_NO_VDELIVERY=1 reverts. */
        if (g_vdelivery) {
            int have_gated_audio = n_audio > 0, gp;
            for (gp = 0; gp < n_pass && !have_gated_audio; gp++)
                if (pass[gp].gated) have_gated_audio = 1;
            if (multiview)
                av_log(NULL, AV_LOG_INFO,
                       "[PTV-VDLV] early-video hold is single-input only — disabled on this mosaic "
                       "(per-slot audio skews diverge on the shared per-rung gate)\n");
            else if (!have_gated_audio)
                av_log(NULL, AV_LOG_INFO,
                       "[PTV-VDLV] no gated audio on this channel — early-video hold disabled\n");
            else
                for (r = 0; r < n_rung; r++)
                    dlv_video_cfg(&rung[r].gate, PTV_VDLV_BAND_US, g_cp.vdlv_cap_us, g_cp.vdlv_maxq);
        }
    }
    /* 0.9.18 M3: register the per-rung gates + master tick with the escalation runtime —
     * cushion_escalate() is now the single writer of every gate->cap_us rewrite. Before any
     * thread starts; NULL gates when delivery is off, so the BANK arms skip them exactly as
     * the demux-side d->gate wiring below does. */
    g_curt.n_gate = n_rung;
    for (r = 0; r < n_rung; r++)
        g_curt.gate[r] = delivery_on ? &rung[r].gate : NULL;
    g_curt.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);

    /* per-input decode side. single-input: inputs[0].dc already holds the graph
     * (fg/fsrc/fsink/filtering) + feeds the rung frame_q inline. multiview: each
     * decode stages into hold; the compositor owns the graph + frame_q fan. */
    for (k = 0; k < n_input; k++) {
        DecodeCtx *d = &inputs[k].dc;
        d->video_q = inputs[k].video_q; d->vdec = inputs[k].vdec; d->ist_tb = inputs[k].ist_tb;
        d->h0 = &inputs[k].h0; d->h0_lock = &inputs[k].h0_lock; d->live = live;
        if (multiview) { d->hold = &inputs[k].hold; d->filtering = 0; d->n_rung = 0; }
        else { d->n_rung = n_rung; for (r = 0; r < n_rung; r++) d->frame_q[r] = rung[r].frame_q; }
        /* §13 deep startup cushion — target derivation moved to resolve_cushions() (0.9.18 M1) */
        d->deep_prime_packets = g_cp.deep_prime_pkts;
        d->vq_shed_req = &inputs[k].vq_shed_req;   /* 1.0.1-pre8 (a): head-GOP shed request slot */
        /* rr10 review fix (D1): the catch-up governor's INPUT-rate currency — measured
         * arrival pps (demux publishes) + the declared header rate.
         * Prefer avg_frame_rate. CAUTION (pre13 semantics): declared is now the TRUST
         * FLOOR (govern only when measured >= declared), so an OVERSTATED declared is a
         * standing trust veto — r_frame_rate's FIELD rate (2x packets, when
         * avg_frame_rate is missing on interlaced) silently disables governance on such
         * channels (DIAG shows gpps=M/D gov=0 with M ~= D/2). Fail-open = safe. */
        d->vin_pps = &inputs[k].da.vin_pps;
        d->vin_pps_wall = &inputs[k].da.vin_pps_wall;   /* pre13: publish-freshness gate */
        {
            AVRational ifr = inputs[k].vist && inputs[k].vist->avg_frame_rate.num
                           ? inputs[k].vist->avg_frame_rate
                           : inputs[k].vist ? inputs[k].vist->r_frame_rate : (AVRational){0, 0};
            d->in_pps_decl = (ifr.num > 0 && ifr.den > 0) ? (ifr.num + ifr.den - 1) / ifr.den : 0;
        }
    }

    g_genlock_ok = (live && !multiview);             /* v0.9.0: genlock applies to single-input live only */

    if (multiview) {                                 /* compositor = the video house clock */
        comp.inputs = inputs; comp.n_input = n_input; comp.fg = fg; comp.n_rung = n_rung;
        comp.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        comp.out_fps = out_fps;                      /* v0.9.12 MV-EXACTTICK: exact measurement axis */
        comp.gate0 = delivery_on ? &rung[0].gate : NULL;   /* v0.9.13: dlvhold=/dlvforced= in the mv stats line */
        comp.live = live;
        comp.slate_after_us = live ? 5 * (int64_t)AV_TIME_BASE : 0;   /* stale cell -> black after 5s */
        for (k = 0; k < n_input; k++) comp.fsrc[k] = fsrc[k];
        for (r = 0; r < n_rung; r++) { comp.fsink[r] = vsink[r]; comp.frame_q[r] = rung[r].frame_q; }
    }

    /* per-rung output side */
    for (r = 0; r < n_rung; r++) {
        VideoCtx *vc = &rung[r].vc;
        vc->frame_q = rung[r].frame_q; vc->mux_q = rung[r].mux_q; vc->venc = rung[r].venc;
        vc->gate = delivery_on ? &rung[r].gate : NULL;   /* §7.5a: this rung's delivery-alignment FIFO */
        vc->out_tb = filtering ? av_buffersink_get_time_base(vsink[r]) : inputs[0].ist_tb;
        vc->tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        vc->out_fps = out_fps;                       /* EXACTTICK: exact rational for content-index stamping */
        vc->live = live; vc->passthrough = multiview;
        vc->h0 = &inputs[0].h0; vc->h0_lock = &inputs[0].h0_lock;
        vc->house_skew = &inputs[0].house_skew;
        vc->est = &inputs[0].est;                /* R4: rate sensor rides input 0's clock, like h0/house_skew */
        vc->hr  = &house_rate;                   /* R4: house-rate actuation state, shared by the rung set */
        vc->vring = (!multiview && r == 0) ? &inputs[0].vring : NULL;  /* single-input: master rung feeds the A/V probe ring (multiview: compositor does) */
        vc->is_master = (r == 0);
        vc->dbg_video_q = inputs[0].video_q; vc->dbg_dec_frames = &inputs[0].dc.dec_frames; vc->dbg_vcorrupt = &inputs[0].dc.vcorrupt;
        vc->dbg_vdrop = &inputs[0].da.vdrop; vc->dbg_pcorrupt = &inputs[0].da.vcorrupt;   /* stats: demux video_q drops + corrupt-pkt */
        vc->dbg_disc_resid = &inputs[0].da.disc_resid_us;   /* 0.9.18.7: hsres= (LAYERA erase-residue ledger) */
        rung[r].ma.ofmt = rung[r].ofmt; rung[r].ma.mux_q = rung[r].mux_q;
        rung[r].ma.is_master = (r == 0);                        /* Φ1′: wire-DTS sensor on rung 0 only */
        rung[r].ma.rung = r;                                    /* pre14: g_mux_sent_wc slot (wire watermark) */
        rung[r].ma.n_producers = 1 + n_audio + n_copy_inputs;   /* video out + N audio + per-input copy fan */
    }
    for (k = 0; k < n_audio; k++) {              /* per-track audio: source from its input's clock */
        as[k].audio_q = audio_q[k];
        as[k].h0 = &inputs[asrc_in[k]].h0; as[k].h0_lock = &inputs[asrc_in[k]].h0_lock;
        as[k].house_skew = &inputs[asrc_in[k]].house_skew;
        as[k].house_lag_true = (n_input > 1) ? &inputs[asrc_in[k]].house_lag_true : NULL;  /* multiview: true lag; single: NULL→house_skew */
        as[k].vring = &inputs[asrc_in[k]].vring;         /* A/V probe: pair this track's audio against its input's video ring */
        as[k].multiview = (n_input > 1);                 /* multiview-only: enable deterministic audio-follow */
        as[k].af_applied_us = 0;
        as[k].dbg_k = k; as[k].dbg_in = asrc_in[k]; as[k].dbg_first_out = AV_NOPTS_VALUE;
        as[k].glue_raw_last_us = AV_NOPTS_VALUE;         /* [PTV-AGLUE] no continuity reference yet */
        as[k].glue_exp_step = &inputs[asrc_in[k]].aglue_exp_step[k];   /* pre5 (D1): shared-flush expected-step slot */
        as[k].glue_exp_dl   = &inputs[asrc_in[k]].aglue_exp_dl[k];
        as[k].corr_epoch    = &inputs[asrc_in[k]].house_disturb;       /* pre14: corrector event feeds (its input slot) */
        as[k].corr_layera_active = g_layera ? &inputs[asrc_in[k]].disc.active : NULL;
        as[k].acomp_exp_us = AV_NOPTS_VALUE;             /* [PTV-ACOMP] no expected-pts reference yet */
        as[k].tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);  /* 1.0.1: house tick = the PLL's vlag quantum (one house clock for all slots) */
        for (r = 0; r < n_rung; r++) {
            as[k].mux_q[r] = rung[r].mux_q;
            as[k].gate[r]  = delivery_on ? &rung[r].gate : NULL;   /* §7.5a: hold transcoded audio for the video front */
        }
    }
    for (kk = 0; kk < n_input; kk++) {           /* per-input demux args (pass/n_pass set in copy loop) */
        DemuxArgs *d = &inputs[kk].da;
        d->ifmt = inputs[kk].ifmt; d->video_q = inputs[kk].video_q;
        d->vstream = inputs[kk].vstream; d->drop = is_net_url(inputs[kk].url); d->n_out = n_rung;
        d->h0 = &inputs[kk].h0; d->h0_lock = &inputs[kk].h0_lock; d->house_skew = &inputs[kk].house_skew;
        d->disturb_epoch = &inputs[kk].house_disturb;   /* B3: discont absorber arms the PLL mid-run re-acquire */
        d->est = &inputs[kk].est;                       /* R4: this input's rate sensor (demux thread feeds it) */
        d->wrap_off = inputs[kk].wrap_off; d->wrap_last = inputs[kk].wrap_last;
        d->wrap_wall_last = inputs[kk].wrap_wall_last; d->video_fwd_us = 0;
        d->edit_us = inputs[kk].edit_us;                /* pre9 sensor: per-stream label-edit ledger */
        d->rsync_pub = (!multiview && kk == 0);         /* single-input: publish to g_rsx */
        d->disc = g_layera ? &inputs[kk].disc : NULL;   /* legacy-0004 buffer (NULL when off) */
        d->vq_shed_req = &inputs[kk].vq_shed_req;       /* 1.0.1-pre8 (a): overflow -> request head-GOP shed */
        d->autobank = g_autobank && !multiview && live && is_net_url(inputs[kk].url);   /* v0.9.14: single-input live only */
        for (r = 0; r < n_rung; r++) {
            d->mux_q[r] = rung[r].mux_q;
            d->gate[r]  = delivery_on ? &rung[r].gate : NULL;   /* §7.5a: dense copied audio rides the gate */
        }
        d->n_audio = 0;
        for (k = 0; k < n_audio; k++)
            if (asrc_in[k] == kk) {
                d->audio_q[d->n_audio] = audio_q[k]; d->astream[d->n_audio] = asrc[k];
                d->aglue_exp_step[d->n_audio] = &inputs[kk].aglue_exp_step[k];   /* pre5 (D1): write side of the */
                d->aglue_exp_dl[d->n_audio]   = &inputs[kk].aglue_exp_dl[k];     /* expected-step handshake      */
                d->aglobal[d->n_audio] = k;   /* pre15: global track index — keys the per-track
                                               * published atomics (pad ledger, decode watermark, fill) */
                d->n_audio++;
            }
    }

    /* pre9 sensor: wire the track count for the stats-line lipsync= field. Single-input only —
     * multiview leaves n_a = 0, so the mv stats path prints no lipsync= (explicit, not garbage). */
    g_rsx.n_a = multiview ? 0 : n_audio;

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %s %d input(s) %d rung(s)  house %d/%d fps (%s)  v:%s->enc  a:%s  in:%s  pull-pipeline\n",
        multiview ? "MULTIVIEW" : "single", n_input, n_rung, out_fps.num, out_fps.den,
        live ? "live" : "offline", vdecoder->name,
        have_audio ? "aac" : "none", net_input ? "net(drop)" : "file(block)");
    for (r = 0; r < n_rung; r++)
        av_log(NULL, AV_LOG_INFO, "  rung%d: %dx%d -> %s [%s]\n",
               r, rung[r].fw, rung[r].fh, outs->groups[r].arg, rung[r].ofmt->oformat->name);

    /* spawn: N mux + N output + N watchdog + (per input) decode + demux + 1 audio
     * per track + (multiview) 1 compositor. */
    for (r = 0; r < n_rung; r++) {
        int pe = pthread_create(&rung[r].th_mux, NULL, mux_thread, &rung[r].ma);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        rung[r].started_mux = 1;
    }
    for (r = 0; r < n_rung; r++) {
        int pe = pthread_create(&rung[r].th_output, NULL, output_thread, &rung[r].vc);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        rung[r].started_output = 1;
    }
    for (r = 0; r < n_rung; r++)
        if (!pthread_create(&rung[r].th_wd, NULL, watchdog_thread, &rung[r].vc)) rung[r].started_wd = 1;
    if (multiview) {
        int pe = pthread_create(&th_compositor, NULL, compositor_thread, &comp);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        started_compositor = 1;
    }
    for (k = 0; k < n_input; k++) {
        int pe = pthread_create(&inputs[k].th_decode, NULL, decode_thread, &inputs[k].dc);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        inputs[k].started_decode = 1;
    }
    for (k = 0; k < n_audio; k++) {
        if (!pthread_create(&th_audio[k], NULL, audio_thread, &as[k])) started_audio[k] = 1;
        else {                                      /* this track produces nothing: send its mux EOF */
            av_log(NULL, AV_LOG_WARNING, "audio thread %d create failed\n", k);
            for (r = 0; r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
        }
    }
    for (k = 0; k < n_input; k++) {
        int pe = pthread_create(&inputs[k].th_demux, NULL, demux_thread, &inputs[k].da);
        if (pe) {                                   /* couldn't start this demux: EOF its consumers */
            av_thread_message_queue_set_err_recv(inputs[k].video_q, AVERROR_EOF);
            { int t; for (t = 0; t < inputs[k].da.n_audio; t++) av_thread_message_queue_set_err_recv(inputs[k].da.audio_q[t], AVERROR_EOF); }
            for (r = 0; inputs[k].da.n_pass > 0 && r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
            ret = AVERROR(pe);
        } else inputs[k].started_demux = 1;
    }
    for (k = 0; k < n_input; k++) if (inputs[k].started_demux) pthread_join(inputs[k].th_demux, NULL);

shutdown:
    if (aborted) {                                  /* force the pipeline to unwind: release ANY
                                                     * thread blocked in send OR recv on any queue */
        for (k = 0; k < n_input; k++) if (inputs[k].video_q) {
            av_thread_message_queue_set_err_send(inputs[k].video_q, AVERROR_EOF);
            av_thread_message_queue_set_err_recv(inputs[k].video_q, AVERROR_EOF);
        }
        for (k = 0; k < n_audio; k++) if (audio_q[k]) {
            av_thread_message_queue_set_err_send(audio_q[k], AVERROR_EOF);
            av_thread_message_queue_set_err_recv(audio_q[k], AVERROR_EOF);
        }
        for (r = 0; r < n_rung; r++) {
            rung[r].vc.output_done = 1;             /* stop the watchdog */
            if (rung[r].frame_q) {
                av_thread_message_queue_set_err_send(rung[r].frame_q, AVERROR_EOF);
                av_thread_message_queue_set_err_recv(rung[r].frame_q, AVERROR_EOF);
            }
            if (rung[r].mux_q) {
                av_thread_message_queue_set_err_send(rung[r].mux_q, AVERROR_EOF);
                av_thread_message_queue_set_err_recv(rung[r].mux_q, AVERROR_EOF);
            }
        }
    }
    for (k = 0; k < n_input; k++) if (inputs[k].started_decode) pthread_join(inputs[k].th_decode, NULL);
    if (started_compositor) pthread_join(th_compositor, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_output) pthread_join(rung[r].th_output, NULL);
    for (k = 0; k < n_audio; k++) if (started_audio[k]) pthread_join(th_audio[k], NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_wd) pthread_join(rung[r].th_wd, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_mux) {
        if (!ret && rung[r].ma.err < 0) ret = rung[r].ma.err;
        pthread_join(rung[r].th_mux, NULL);
    }

    {
        int64_t dec_sum = 0, vpkt = 0, ppkt = 0;
        int64_t m_emit = multiview ? comp.emitted : rung[0].vc.emitted;
        int64_t m_dup  = multiview ? comp.dup     : rung[0].vc.dup;
        for (k = 0; k < n_input; k++) { dec_sum += inputs[k].dc.dec_frames; vpkt += inputs[k].da.vpkt; ppkt += inputs[k].da.ppkt; }
        av_log(NULL, AV_LOG_INFO,
            "ptvencoder: done — %d input(s) %d rung(s); video dec %"PRId64" out %"PRId64
            " (dup %"PRId64")  demux v:%"PRId64" p:%"PRId64"%s\n",
            n_input, n_rung, dec_sum, m_emit, m_dup, vpkt, ppkt, have_audio ? "" : "  [no audio]");
    }
    if (have_audio) {
        int64_t ain = 0, aout = 0, apkt = 0;
        for (k = 0; k < n_audio; k++) { ain += as[k].in_frames; aout += as[k].out_frames; }
        for (k = 0; k < n_input; k++) apkt += inputs[k].da.apkt;
        av_log(NULL, AV_LOG_INFO, "ptvencoder: audio %d track(s), in %"PRId64" frames, out %"PRId64" aac (demux a:%"PRId64")\n",
               n_audio, ain, aout, apkt);
    }
    if (ret > 0) ret = 0;

end:
    for (r = 0; r < n_rung; r++) {
        if (rung[r].hdr_written) av_write_trailer(rung[r].ofmt);
        if (rung[r].ofmt && !(rung[r].ofmt->oformat->flags & AVFMT_NOFILE) && rung[r].ofmt->pb)
            avio_closep(&rung[r].ofmt->pb);
    }
    for (k = 0; k < n_audio; k++) av_thread_message_queue_free(&audio_q[k]);
    for (r = 0; r < n_rung; r++) {
        dlv_destroy(&rung[r].gate);              /* §7.5a: free any held packets + the gate's mutex/cond */
        av_thread_message_queue_free(&rung[r].frame_q);
        av_thread_message_queue_free(&rung[r].mux_q);
        avcodec_free_context(&rung[r].venc);
        av_buffer_unref(&rung[r].fhwfr);
        if (rung[r].ofmt) avformat_free_context(rung[r].ofmt);
    }
    for (k = 0; k < n_audio; k++) {
        if (as[k].swr)  swr_free(&as[k].swr);
        if (as[k].fifo) av_audio_fifo_free(as[k].fifo);
        avfilter_graph_free(&as[k].afg);
        for (r = 0; r < n_rung; r++) avcodec_free_context(&as[k].enc[r]);
        avcodec_free_context(&as[k].dec);
    }
    avfilter_graph_free(&fg);
    for (k = 0; k < n_input; k++) {
        avfilter_graph_free(&inputs[k].dc.fg);       /* single-input graph (multiview: NULL) */
        avcodec_free_context(&inputs[k].vdec);
        av_thread_message_queue_free(&inputs[k].video_q);
        av_thread_message_queue_free(&inputs[k].hold.q);
        av_freep(&inputs[k].wrap_off);
        av_freep(&inputs[k].wrap_last);
        av_freep(&inputs[k].wrap_wall_last);
        av_freep(&inputs[k].edit_us);
        ptv_disc_free(&inputs[k].disc);   /* legacy-0004 buffer (no-op if never inited) */
        if (inputs[k].ifmt) avformat_close_input(&inputs[k].ifmt);
        pthread_mutex_destroy(&inputs[k].h0_lock);
        pthread_mutex_destroy(&inputs[k].hold.lock);
        pthread_mutex_destroy(&inputs[k].vring.lock);
    }
    av_buffer_unref(&hw_device);
    return ret;
}

/* ---- ffmpeg-style command parsing (reuses cmdutils split_commandline) ----
 * We only name the STRUCTURAL options; unknown encoder/mux options (-preset,
 * -rc, -mpegts_flags, -pat_period, ...) fall through to opt_default and land in
 * each group's codec_opts/format_opts dicts, forwarded verbatim to libav. A few
 * ffmpeg-CLI-only opts are listed as recognized/no-op so real commands parse. */
static const OptionGroupDef ptv_groups[] = {
    { "output url", NULL, OPT_OUTPUT },   /* no separator: ended by a bare URL; must be first */
    { "input url",  "i",  OPT_INPUT  },
};

static const OptionDef ptv_options[] = {
    /* recognized-but-passive ffmpeg-CLI globals (so production commands parse) */
    { "v",                OPT_TYPE_STRING, 0,                        { .off = 0 }, "log level", "level" },
    { "loglevel",         OPT_TYPE_STRING, 0,                        { .off = 0 }, "log level", "level" },
    { "stats",            OPT_TYPE_BOOL,   0,                        { .off = 0 }, "print stats" },
    { "nostats",          OPT_TYPE_BOOL,   0,                        { .off = 0 }, "disable stats" },
    { "stats_period",     OPT_TYPE_STRING, 0,                        { .off = 0 }, "stats period", "t" },
    { "y",                OPT_TYPE_BOOL,   0,                        { .off = 0 }, "overwrite output" },
    { "n",                OPT_TYPE_BOOL,   0,                        { .off = 0 }, "never overwrite" },
    { "hide_banner",      OPT_TYPE_BOOL,   0,                        { .off = 0 }, "suppress startup banner" },
    { "init_hw_device",   OPT_TYPE_STRING, 0,                        { .off = 0 }, "init hw device", "args" },
    { "filter_hw_device", OPT_TYPE_STRING, 0,                        { .off = 0 }, "filter hw device", "name" },
    { "filter_complex",   OPT_TYPE_STRING, 0,                        { .off = 0 }, "filtergraph", "graph" },
    { "abort_on",         OPT_TYPE_STRING, 0,                        { .off = 0 }, "abort conditions", "flags" },
    /* per-output structural options (walked from g->opts[]) */
    { "map",              OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "stream map", "spec" },
    { "c",                OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "codec", "codec" },
    { "codec",            OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "codec", "codec" },
    { "b",                OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "bitrate", "rate" },
    { "metadata",         OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "metadata", "key=val" },
    { "disposition",      OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "disposition", "flags" },
    { "filter",           OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "stream filtergraph", "graph" },
    { "vf",               OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "video filtergraph", "graph" },
    { "af",               OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "audio filtergraph", "graph" },
    { "r",                OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "frame rate", "fps" },
    { "ar",               OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "audio rate", "hz" },
    { "ac",               OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "audio channels", "n" },
    { "fps_mode",         OPT_TYPE_STRING, OPT_SPEC | OPT_OUTPUT,    { .off = 0 }, "fps mode", "mode" },
    { "avoid_negative_ts",OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "avoid negative ts", "mode" },
    { "max_muxing_queue_size", OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "max muxing queue", "n" },
    { "t",                OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "duration", "sec" },
    { "an",               OPT_TYPE_BOOL,   OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "no audio" },
    { "f",                OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "force format", "fmt" },
    { NULL },
};

static void ptv_dump_group(const char *kind, OptionGroup *grp)
{
    const AVDictionaryEntry *e = NULL;
    int o;
    av_log(NULL, AV_LOG_INFO, "=== %s: %s ===\n", kind, grp->arg && *grp->arg ? grp->arg : "(global)");
    for (o = 0; o < grp->nb_opts; o++)
        av_log(NULL, AV_LOG_INFO, "    opt        %-12s = %s\n", grp->opts[o].key, grp->opts[o].val);
    while ((e = av_dict_iterate(grp->codec_opts, e)))
        av_log(NULL, AV_LOG_INFO, "    codec_opt  %-12s = %s\n", e->key, e->value);
    e = NULL;
    while ((e = av_dict_iterate(grp->format_opts, e)))
        av_log(NULL, AV_LOG_INFO, "    fmt_opt    %-12s = %s\n", e->key, e->value);
}

/* PTV_PARSE_DEBUG=1: parse an ffmpeg-style command and print the resolved plan,
 * then exit. Validates the cmdutils reuse before the pipeline is rewired. */
static int ptv_parse_and_print(int argc, char **argv)
{
    OptionParseContext octx;
    int g, gi, ret;
    ret = split_commandline(&octx, argc, argv, ptv_options, ptv_groups,
                            sizeof(ptv_groups) / sizeof(ptv_groups[0]));
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "split_commandline failed: %s\n", av_err2str(ret));
        return ret;
    }
    ptv_dump_group("global", &octx.global_opts);
    for (g = 0; g < octx.nb_groups; g++) {
        OptionGroupList *l = &octx.groups[g];
        for (gi = 0; gi < l->nb_groups; gi++)
            ptv_dump_group(l->group_def->name, &l->groups[gi]);
    }
    uninit_parse_context(&octx);
    return 0;
}

/* ---- ffmpeg-style plan resolution (stage 1: resolve -map/-c, dry-run print) ---- */

/* last value of an exact-key option in a group (e.g. "f", "filter:v", "r") */
static const char *og_get(OptionGroup *g, const char *key)
{
    const char *v = NULL; int i;
    for (i = 0; i < g->nb_opts; i++)
        if (!strcmp(g->opts[i].key, key)) v = g->opts[i].val;
    return v;
}

/* most-specific per-stream string option: <p>:<t>:<idx>  >  <p>:<t>  >  <p>.
 * p = "c"/"b"/... , t = 'v'/'a'/'s'/'d', idx = output type-index. */
static const char *og_spec(OptionGroup *g, const char *p, char t, int idx)
{
    /* Buffers must hold the longest option name ("disposition", 11) plus the ":t" and
     * ":t:idx" suffixes; undersized buffers silently truncated -disposition / -disposition:a
     * so only the fully-indexed form matched. Sized with headroom for any future option. */
    char k0[24], k1[28], k2[32]; const char *best = NULL; int i, rank = -1;
    snprintf(k0, sizeof k0, "%s", p);
    snprintf(k1, sizeof k1, "%s:%c", p, t);
    snprintf(k2, sizeof k2, "%s:%c:%d", p, t, idx);
    for (i = 0; i < g->nb_opts; i++) {
        const char *k = g->opts[i].key; int r = -1;
        if (!strcmp(k, k0) || (!strcmp(p, "c") && !strcmp(k, "codec"))) r = 0;
        if (!strcmp(k, k1)) r = 1;
        if (!strcmp(k, k2)) r = 2;
        if (r > rank) { rank = r; best = g->opts[i].val; }
    }
    return best;
}

/* Apply CLI -metadata:s:<t>:idx (possibly several) and -disposition:<t>:idx onto
 * an output stream (G5). metadata overrides the source value set earlier;
 * disposition takes ffmpeg's "flag", "+flag+flag" or "0"/"none" forms. */
static void apply_stream_meta(OptionGroup *g, char t, int idx, AVStream *ost)
{
    char key[24]; int i;
    const char *disp = og_spec(g, "disposition", t, idx);
    if (disp) {
        if (!strcmp(disp, "0") || !strcmp(disp, "none")) {
            ost->disposition = 0;
        } else {
            char buf[128], *tok, *sp = NULL; int d = 0;
            snprintf(buf, sizeof buf, "%s", disp);
            for (tok = strtok_r(buf, "+", &sp); tok; tok = strtok_r(NULL, "+", &sp)) {
                int f = av_disposition_from_string(tok);
                if (f > 0) d |= f;
            }
            ost->disposition = d;
        }
    }
    snprintf(key, sizeof key, "metadata:s:%c:%d", t, idx);     /* -metadata:s:a:N key=val (repeatable) */
    for (i = 0; i < g->nb_opts; i++) {
        char kv[256], *eq;
        if (strcmp(g->opts[i].key, key)) continue;
        snprintf(kv, sizeof kv, "%s", g->opts[i].val);
        if ((eq = strchr(kv, '='))) { *eq = 0; av_dict_set(&ost->metadata, kv, eq + 1, 0); }
    }
}

/* strip a leading "<digits>:" file index and a trailing '?' from a -map value */
static const char *map_spec(const char *v, char *buf, size_t bufsz, int *optional, int *file_idx)
{
    const char *colon = strchr(v, ':'), *s = v; size_t L;
    if (file_idx) *file_idx = 0;
    if (colon && colon > v) {
        const char *p; int alldig = 1;
        for (p = v; p < colon; p++) if (*p < '0' || *p > '9') { alldig = 0; break; }
        if (alldig) { if (file_idx) *file_idx = atoi(v); s = colon + 1; }
    }
    snprintf(buf, bufsz, "%s", s);
    L = strlen(buf);
    *optional = (L && buf[L-1] == '?');
    if (*optional) buf[L-1] = 0;
    return buf;
}

/* PTV_PLAN_DEBUG=1: parse an ffmpeg-style command, open the input, resolve each
 * -map to an input stream + its copy/encode decision (and applied opts), print. */
/* Resolve an output group's -map/-c into a Sel (transcode vs copy decision).
 * No -map -> auto (best video + first <=2ch audio + copy the rest), back-compat
 * (single-input only). Explicit -map K:... selects from input K (multiview). */
static int resolve_plan(Input *inputs, int n_input, OptionGroup *outg, Sel *s)
{
    AVFormatContext *ifmt = inputs[0].ifmt;
    int o, si, tcnt[5] = {0}, nmap = 0, astream = -1;
    int no_audio = og_get(outg, "an") != NULL;   /* -an: suppress auto-selected audio */
    memset(s, 0, sizeof *s);
    s->vstream = -1;
    s->vf = og_get(outg, "filter:v"); if (!s->vf) s->vf = og_get(outg, "vf");
    for (o = 0; o < outg->nb_opts; o++) if (!strcmp(outg->opts[o].key, "map")) nmap++;
    s->have = nmap > 0;

    if (!nmap) {                       /* no -map: auto-select on input 0 (back-compat) */
        s->vstream = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &s->vdec, 0);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            AVCodecParameters *cp = ifmt->streams[si]->codecpar;
            if (cp->codec_type == AVMEDIA_TYPE_AUDIO &&
                cp->ch_layout.nb_channels > 0 && cp->ch_layout.nb_channels <= 2) { astream = si; break; }
        }
        if (no_audio) astream = -1;    /* -an: no transcoded audio and no audio copy */
        else if (astream < 0) astream = av_find_best_stream(ifmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        if (astream >= 0) {            /* one transcoded audio track (back-compat) */
            s->aout[0].input  = 0;
            s->aout[0].stream = astream;
            s->aout[0].adec   = avcodec_find_decoder(ifmt->streams[astream]->codecpar->codec_id);
            s->n_aout = 1;
        }
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            enum AVMediaType mt = ifmt->streams[si]->codecpar->codec_type;
            if (si == s->vstream || si == astream) continue;
            if (mt == AVMEDIA_TYPE_AUDIO && no_audio) continue;   /* -an: drop audio copy too */
            if ((mt == AVMEDIA_TYPE_AUDIO || mt == AVMEDIA_TYPE_SUBTITLE || mt == AVMEDIA_TYPE_DATA)
                && s->n_copy < PTV_MAX_PASS) { s->copy_input[s->n_copy] = 0; s->copy[s->n_copy++] = si; }
        }
        s->venc = og_get(outg, "c:v"); if (!s->venc) s->venc = og_get(outg, "c");
        s->vbr  = og_get(outg, "b:v"); if (!s->vbr)  s->vbr  = og_get(outg, "b");
        return 0;
    }

    for (o = 0; o < outg->nb_opts; o++) {              /* explicit -map plan */
        char buf[64]; int optional, fidx; const char *spec, *mv = outg->opts[o].val;
        AVFormatContext *kfmt;
        if (strcmp(outg->opts[o].key, "map")) continue;
        if (mv[0] == '[') {                            /* filter-output label = this rung's video */
            int idx = tcnt[0]++;                       /* video output index */
            s->venc = og_spec(outg, "c", 'v', idx);    /* -c:v / -c:v:0 (NULL -> default encoder) */
            s->vbr  = og_spec(outg, "b", 'v', idx);    /* -b:v / -b:v:0 */
            continue;                                  /* video comes from the graph, not an input stream */
        }
        spec = map_spec(mv, buf, sizeof buf, &optional, &fidx);
        if (fidx < 0 || fidx >= n_input) {
            av_log(NULL, AV_LOG_ERROR, "-map %s: input %d out of range (%d input(s))\n", mv, fidx, n_input);
            return AVERROR(EINVAL);
        }
        kfmt = inputs[fidx].ifmt;
        for (si = 0; si < (int)kfmt->nb_streams; si++) {
            enum AVMediaType mt; char t; int ti, idx; const char *codec;
            if (avformat_match_stream_specifier(kfmt, kfmt->streams[si], spec) <= 0) continue;
            mt  = kfmt->streams[si]->codecpar->codec_type;
            t   = mt==AVMEDIA_TYPE_VIDEO?'v':mt==AVMEDIA_TYPE_AUDIO?'a':
                  mt==AVMEDIA_TYPE_SUBTITLE?'s':mt==AVMEDIA_TYPE_DATA?'d':'?';
            ti  = mt==AVMEDIA_TYPE_VIDEO?0:mt==AVMEDIA_TYPE_AUDIO?1:
                  mt==AVMEDIA_TYPE_SUBTITLE?2:mt==AVMEDIA_TYPE_DATA?3:4;
            idx = tcnt[ti]++;
            codec = og_spec(outg, "c", t, idx);
            if (codec && !strcmp(codec, "copy")) {
                if (s->n_copy < PTV_MAX_PASS) { s->copy_input[s->n_copy] = fidx; s->copy[s->n_copy++] = si; }
            } else if (mt == AVMEDIA_TYPE_VIDEO && s->vstream < 0 && fidx == 0) {
                s->vstream = si; s->venc = codec; s->vbr = og_spec(outg, "b", t, idx);
                s->vdec = avcodec_find_decoder(kfmt->streams[si]->codecpar->codec_id);
            } else if (mt == AVMEDIA_TYPE_AUDIO && s->n_aout < PTV_MAX_AUDIO) {
                AOutSpec *a = &s->aout[s->n_aout++];    /* one transcoded audio track per -map */
                const char *acs = og_spec(outg, "ac", t, idx);   /* -ac:a:N output channels */
                a->input = fidx; a->stream = si; a->aenc = codec; a->abr = og_spec(outg, "b", t, idx);
                a->filter = og_spec(outg, "filter", t, idx);     /* -filter:a:N (else global -af) */
                a->ac = acs ? atoi(acs) : 0;                      /* 0 = default stereo */
                a->adec = avcodec_find_decoder(kfmt->streams[si]->codecpar->codec_id);
            } else if (s->n_copy < PTV_MAX_PASS) {
                s->copy_input[s->n_copy] = fidx; s->copy[s->n_copy++] = si;   /* over the cap -> copy */
            }
        }
    }
    return 0;
}

static int plan_resolve_and_print(int argc, char **argv)
{
    OptionParseContext octx;
    AVFormatContext *ifmt = NULL;
    OptionGroupList *outs, *ins;
    OptionGroup *outg, *ing;
    const AVDictionaryEntry *e;
    int ret, o, si, tcnt[5] = {0};

    if ((ret = split_commandline(&octx, argc, argv, ptv_options, ptv_groups,
                                 sizeof(ptv_groups)/sizeof(ptv_groups[0]))) < 0)
        return ret;
    ins = &octx.groups[1]; outs = &octx.groups[0];
    if (ins->nb_groups < 1 || outs->nb_groups < 1) {
        av_log(NULL, AV_LOG_ERROR, "need -i <input> and an output url\n");
        uninit_parse_context(&octx); return AVERROR(EINVAL);
    }
    ing = &ins->groups[0]; outg = &outs->groups[0];
    if ((ret = avformat_open_input(&ifmt, ing->arg, NULL, &ing->format_opts)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "open input '%s': %s\n", ing->arg, av_err2str(ret));
        uninit_parse_context(&octx); return ret;
    }
    avformat_find_stream_info(ifmt, NULL);
    av_log(NULL, AV_LOG_INFO, "PLAN  in=%s  out=%s  fmt=%s\n",
           ing->arg, outg->arg, og_get(outg, "f") ? og_get(outg, "f") : "(guess)");
    for (o = 0; o < outg->nb_opts; o++) {
        char buf[64]; int optional, matched = 0; const char *spec;
        if (strcmp(outg->opts[o].key, "map")) continue;
        if (outg->opts[o].val[0] == '[') {
            av_log(NULL, AV_LOG_INFO, "  map %-9s -> filter output (ladder phase, not yet wired)\n",
                   outg->opts[o].val);
            continue;
        }
        spec = map_spec(outg->opts[o].val, buf, sizeof buf, &optional, NULL);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            enum AVMediaType mt; char t; int ti, idx; const char *codec, *br;
            if (avformat_match_stream_specifier(ifmt, ifmt->streams[si], spec) <= 0) continue;
            matched++;
            mt = ifmt->streams[si]->codecpar->codec_type;
            t  = mt==AVMEDIA_TYPE_VIDEO?'v':mt==AVMEDIA_TYPE_AUDIO?'a':
                 mt==AVMEDIA_TYPE_SUBTITLE?'s':mt==AVMEDIA_TYPE_DATA?'d':'?';
            ti = mt==AVMEDIA_TYPE_VIDEO?0:mt==AVMEDIA_TYPE_AUDIO?1:
                 mt==AVMEDIA_TYPE_SUBTITLE?2:mt==AVMEDIA_TYPE_DATA?3:4;
            idx = tcnt[ti]++;
            codec = og_spec(outg, "c", t, idx);
            br    = og_spec(outg, "b", t, idx);
            if (codec && !strcmp(codec, "copy"))
                av_log(NULL, AV_LOG_INFO, "  map %-9s -> in#%d %-9s : COPY (passthrough)\n",
                       outg->opts[o].val, si, av_get_media_type_string(mt));
            else
                av_log(NULL, AV_LOG_INFO, "  map %-9s -> in#%d %-9s : ENCODE %s%s%s\n",
                       outg->opts[o].val, si, av_get_media_type_string(mt),
                       codec ? codec : "(default)", br ? " @" : "", br ? br : "");
        }
        if (!matched)
            av_log(NULL, optional ? AV_LOG_INFO : AV_LOG_WARNING,
                   "  map %-9s -> no match%s\n", outg->opts[o].val, optional ? " (optional)" : " (REQUIRED!)");
    }
    av_log(NULL, AV_LOG_INFO, "  video filter: %s\n",
           og_get(outg,"filter:v") ? og_get(outg,"filter:v") :
           og_get(outg,"vf") ? og_get(outg,"vf") : "(none)");
    e = NULL; while ((e = av_dict_iterate(outg->codec_opts, e)))
        av_log(NULL, AV_LOG_INFO, "  enc-opt  %s=%s\n", e->key, e->value);
    e = NULL; while ((e = av_dict_iterate(outg->format_opts, e)))
        av_log(NULL, AV_LOG_INFO, "  mux-opt  %s=%s\n", e->key, e->value);
    avformat_close_input(&ifmt);
    uninit_parse_context(&octx);
    return 0;
}

/* ffmpeg-style startup banner: product name + FFmpeg build id + lib versions.
 * Emitted via av_log (so the [timestamp] prefix and -loglevel gating apply, like
 * the rest of the output); suppressed by -hide_banner. */
static void ptv_show_banner(void)
{
    av_log(NULL, AV_LOG_INFO, "Perception TV Encoder (ptvencoder) %s  FFmpeg %s\n", PTVENCODER_VERSION, av_version_info());
    av_log(NULL, AV_LOG_INFO, "  libavutil      %u.%u.%u\n",
           AV_VERSION_MAJOR(avutil_version()), AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
    av_log(NULL, AV_LOG_INFO, "  libavcodec     %u.%u.%u\n",
           AV_VERSION_MAJOR(avcodec_version()), AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
    av_log(NULL, AV_LOG_INFO, "  libavformat    %u.%u.%u\n",
           AV_VERSION_MAJOR(avformat_version()), AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
    av_log(NULL, AV_LOG_INFO, "  libavfilter    %u.%u.%u\n",
           AV_VERSION_MAJOR(avfilter_version()), AV_VERSION_MINOR(avfilter_version()), AV_VERSION_MICRO(avfilter_version()));
    av_log(NULL, AV_LOG_INFO, "  libswresample  %u.%u.%u\n",
           AV_VERSION_MAJOR(swresample_version()), AV_VERSION_MINOR(swresample_version()), AV_VERSION_MICRO(swresample_version()));
}

int main(int argc, char **argv)
{
    OptionParseContext octx;
    OptionGroupList *ins;
    OptionGroupList *outs;
    const char *fcomplex = NULL, *hwdev = NULL;
    int mode = -1, ret, gi, hide_banner = 0;

    init_dynload();
    av_log_set_level(AV_LOG_INFO);
    g_diag = !!getenv("PTV_DIAG");
    g_vindbg = !!getenv("PTV_VINDBG");   /* TEMP pre13 diagnosis probe */
    if (getenv("PTV_NO_AVLOCK")) g_avlock = 0;   /* revert to source-locked audio (drifts on dup) */
    if (getenv("PTV_LAYERA")) g_layera = 1;       /* legacy 0004: audio-derived common offset at glue points (corrects source A/V mis-mux) */
    if (getenv("PTV_REPRIME")) g_reprime = 1;     /* fast buffer re-prime after a glue (default ON since v0.9.10; kept for compat) */
    if (getenv("PTV_NO_REPRIME")) g_reprime = 0;
    if (getenv("PTV_NO_LAYERA"))  g_layera  = 0;  /* v0.9.10: WUCR/LAYERA/REPRIME are default-on; NO_* revert */
    if (getenv("PTV_LAYERA_FULLSKIP")) g_layera_fullskip = 1;  /* 0.9.18.5 revert: LAYERA skips the demux absorber for ALL
                                                                * super-threshold jumps again (restores the sub-1s no-owner
                                                                * band = the In-Touch audio-late accumulator; A/B only) */
    if (getenv("PTV_NO_SHARED_FLUSH")) g_shared_flush = 0;     /* 1.0.1-pre4 revert: LAYERA flushes erase per-stream again
                                                                * (bakes the A-vs-V jump difference into the output on
                                                                * asymmetric events — A/B / rollback only) */
    if (getenv("PTV_NO_ADAPTIVE")) g_adapt_cushion = 0;   /* fixed preroll target (pre-0.9.10 behavior) */
    /* PTV_CUSHION_MS parse moved to resolve_cushions() (0.9.18 M1) */
    if (getenv("PTV_NO_GENLOCK")) g_genlock = 0; /* v0.9.0: revert to free-run nominal pacing (+ old 350ms prime) = byte-identical */
    if (getenv("PTV_WUCR")) {                    /* occupancy-ρ video pacing (EMA-filtered + ±1f deadband, gain-6/±6%) + buf/rho readout. AVLOCK KEPT ON: ρ bounds house_skew so AVLOCK's audio-follow is harmless in steady state and keeps A/V matched through unavoidable dups. FLL still computes srcppm for comparison. */
        g_wucr = 1;
        av_log(NULL, AV_LOG_INFO, "ptvencoder: [WUCR] active — PROPORTIONAL occupancy-ρ pacing (ρ=500·err, EMA N=16, ±6%% clamp) + AVLOCK ON "
               "(ρ bounds house_skew; audio follows it so A/V stays matched through dups). Expect: ρ smooth, parks near −(source ppm offset) with no wobble, dlvforced≈0, dup low, speed=1.00x.\n");
    }
    if (getenv("PTV_NO_WUCR")) g_wucr = 0;       /* v0.9.10: WUCR default-on; revert to genlock/free-run */
    if (getenv("PTV_NO_GENLOCK_GUARD")) g_genlock_guard = 0;  /* v0.9.4: revert to the unbounded ±1%-gate FLL (A/B the runaway) */
    /* 0.9.18.7: PTV_GENLOCK_MAX_PPM / PTV_GENLOCK_REJECT_PPM / PTV_GENLOCK_WINDOW_MS /
     * PTV_GENLOCK_EMA_SHIFT internalized at the production defaults (300ppm / 700ppm / 3000ms / 6)
     * — see the g_gl_* declarations. The reject >= 2*max invariant holds statically (734 >= 2*314). */
    if (getenv("PTV_NO_REANCHOR")) g_reanchor = 0;   /* keep stale dup skew across outages (A/B) */
    if (getenv("PTV_MV_CLAMP")) g_mv_clamp = 1;      /* opt-in: re-enable the (stutter-prone) content clamp */
    if (getenv("PTV_NO_DISCONT")) g_discont = 0;     /* A/B: don't absorb source PTS discontinuities */
    if (getenv("PTV_NO_GAPDISCRIM")) g_gapdiscrim = 0;   /* gap-fix A/B: revert to unconditional forward absorb (old desync-on-audio-gap behaviour) */
    if (getenv("PTV_NO_ADECWD")) g_adecwd = 0;       /* 1.0.1: disable the audio decode-death watchdog (error TOLERANCE stays on) */
    if (getenv("PTV_NO_ANCHOR_HEADFILL")) g_anchor_headfill = 0;   /* 1.0.1: revert to first-packet-at-first_audio−h0 birth */
    /* 0.9.18.7: PTV_GAP_MIN_MS internalized (700ms — g_gap_min_us) */
    { const char *wg = getenv("PTV_WRAP_GUARD_S"); if (wg && atoi(wg) > 0) g_wrap_guard_us = (int64_t)atoi(wg) * 1000000; }  /* v0.9.16.1 wrap-guard threshold override (TEST ONLY) */
    if (getenv("PTV_NVENC_SERIALIZE")) g_nvenc_serialize = 1;  /* v0.9.16.5 scale fix B2 (opt-in): one process-wide mutex around video encoder calls — cuts concurrent NVIDIA RM-lock callers 6->1 per process */
    { const char *ag = getenv("PTV_AGLUE_MS");     if (ag) g_aglue_ms = atoi(ag); }          /* v0.9.16.3 label-step glue threshold; 0 disables */
    /* 0.9.18.7: PTV_AGLUE_MAX_MS (1000ms) / PTV_DISCONT_MS (1000ms) / PTV_DISCONT_BACK_MS (80ms)
     * internalized — see g_aglue_max_ms / g_discont_ms / g_discont_back_ms */
    if (getenv("PTV_NO_PROG_OFF")) g_prog_off = 0;   /* P2: A/B — sparse copied streams get 33-bit wrap only (v0.6.23) */
    if (getenv("PTV_PROGOFF_AV")) g_progoff_av = 1;     /* §5.A.2: explicit enable (redundant — default ON) */
    if (getenv("PTV_NO_PROGOFF_AV")) g_progoff_av = 0;  /* §5.A.2: A/B disable → legacy per-stream self-rebase */
    /* 0.9.18.7: PTV_PROGOFF_DEBOUNCE_MS internalized (1000ms — g_progoff_debounce_us) */
    if (getenv("PTV_NO_DUKF")) g_drop_until_kf = 0;  /* P2 2b: A/B — decode the post-splice corruption burst (v0.6.23) */
    /* 0.9.18.7: PTV_DUKF_ESCAPE_MS (5000ms) / PTV_DUKF_MIN_MS (1000ms) internalized —
     * see g_dukf_escape_us / g_dukf_min_ms */
    if (getenv("PTV_NO_AUDIO_FOLLOW")) g_audio_follow = 0;  /* A/B: multiview audio uses old floored/capped async skew */
    if (getenv("PTV_NO_H0_REANCHOR")) g_h0_reanchor = 0;    /* A/B: don't floor per-slot lag (allow video-ahead) */
    if (getenv("PTV_REANCHOR2_INSTANT")) g_reanchor2_instant = 1;  /* 1.0.1: revert REANCHOR2 to single-sample fire (no 3-of-5 median debounce) */
    if (getenv("PTV_NO_H0_AT_DISPLAY")) g_h0_at_display = 0; /* A/B: multiview anchors h0 at first DECODE, not first DISPLAY */
    /* 0.9.18.7: PTV_H0_REANCHOR_MS internalized (120ms — g_h0_reanchor_ms) */
    if (getenv("PTV_AF_NO_PLL")) g_af_pll = 0;              /* A/B: pure discrete drop/pad (no smooth nudge) */
    if (getenv("PTV_AF_NO_ANCHOR")) g_af_anchor = 0;        /* A/B: revert B1 → pre-B1 free-running counter */
    /* PTV_PREROLL_MS / PTV_VIDEOQ / PTV_CUSHION_MAX_MS / PTV_BANK_DECAY_S parses moved to resolve_cushions() (0.9.18 M1) */
    if (getenv("PTV_NO_AUTOBANK")) g_autobank = 0;   /* v0.9.14: revert to advisor-only (manual PTV_PREROLL_MS recipe) */
    if (getenv("PTV_NO_CLOCKFOLLOW")) g_clockfollow = 0;   /* v0.9.15: never follow a large source-clock offset (buffers pin + resampler churns on such sources) */
    if (getenv("PTV_NO_DECIMATE")) g_decimate = 0;         /* v0.9.15.2: keep pop-per-tick even for >house-rate sources (frame_q pins on surplus) */
    /* PTV_FRAMEQ stays HERE (not resolve_cushions()): the multiview hold.q alloc consumes
     * g_frameq_cap before resolve_cushions() runs in transcode() setup (0.9.18 M1). */
    { const char *fq = getenv("PTV_FRAMEQ"); if (fq && atoi(fq) > 0) { int v = atoi(fq); if (v < PTV_FRAME_QDEPTH) v = PTV_FRAME_QDEPTH; if (v > 1024) v = 1024; g_frameq_cap = v; } }   /* frame_q (decode->output) capacity; raise + deep PTV_PREROLL_MS to absorb an ad-break decode-rate dip (AWE) */
    {   /* 0.9.16.5 postmortem guard: a frame_q deeper than the NVENC registration cache makes every
         * frame past the cache evict+re-register = 2 RM WRITE ioctls/frame/rung — at fleet scale that
         * was the RM rwlock spiral. Cache is 512 (v2 patch 0003) unless PTV_NVENC_REG_CAP lowers it;
         * an unpatched libavcodec is 64 and cannot be detected from here — patch 0003 is assumed. */
        int reg = 512; const char *rc = getenv("PTV_NVENC_REG_CAP");
        if (rc && atoi(rc) > 0) { reg = atoi(rc); if (reg < 64) reg = 64; if (reg > 512) reg = 512; }
        if (g_frameq_cap + 32 > reg)
            av_log(NULL, AV_LOG_WARNING,
                   "PTV_FRAMEQ %d (+in-flight margin) exceeds the NVENC registration cache (%d) — expect per-frame registration thrash; lower PTV_FRAMEQ or raise PTV_NVENC_REG_CAP\n",
                   g_frameq_cap, reg);
    }
    if (getenv("PTV_NO_EXACTTICK")) g_exacttick = 0;   /* revert to integer-us tick content index (the ~10ppm NTSC drift) */
    if (getenv("PTV_NO_PULLDOWN"))  g_pulldown = 0;    /* v0.9.11: revert telecine-aware emit (film segments back to dup-fill) */
    if (getenv("PTV_NO_CADDISARM")) g_cad_disarm = 0;  /* v0.9.18.1 M7: revert to flag-only pulldown disarm (dropouts drain frame_q again) */
    if (getenv("PTV_NO_MV_EXACTTICK")) g_mv_exacttick = 0;  /* v0.9.12: revert mv measurement axes to integer tick (re-enables the enforced ~10ppm mosaic drift) */
    if (getenv("PTV_NO_RESIDENCE")) g_mv_residence = 0;     /* v0.9.13: revert to one-pop-per-tick (rate-mismatched slots batch dups + fast-forward after starvation) */
    /* genlock preroll default + the three deep-prime side-cars moved to resolve_cushions() (0.9.18 M1) */
    if (getenv("PTV_KEEP_CORRUPT")) g_discardcorrupt = 0;   /* keep AV_PKT_FLAG_CORRUPT video packets (don't +discardcorrupt) */
    if (getenv("PTV_NO_DELIVERY")) g_delivery = 0;          /* §7.5a: disable the A/V delivery-alignment gate (audio sent direct = v0.6.23) */
    if (getenv("PTV_NO_VDELIVERY")) g_vdelivery = 0;        /* §7.5b (pre12): disable the symmetric EARLY-VIDEO hold (video sent direct = pre11 wire) */
    if (getenv("PTV_DELIVERY_MV")) g_delivery_mv = 1;       /* pre-0.9.12.1 opt-in — now the default; kept as a harmless no-op for existing configs */
    if (getenv("PTV_NO_DELIVERY_MV")) g_delivery_mv = 0;    /* v0.9.12.1: revert multiview to ungated wire staging (sync_check-visible audio lead) */
    /* PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ override parses moved to resolve_cushions() (0.9.18 M1) */
    if (getenv("PTV_AVSYNC_PROBE")) g_avsync_probe = 1;    /* Phase A: read-only [PTV-AVSYNC2] real A/V offset */
    /* 0.9.18.7: PTV_AF_ACQUIRE_MS (100ms) / PTV_AF_RATE_MS_S (10) internalized —
     * see g_af_acquire_us / g_af_rate_us */
    if (getenv("PTV_NO_AVSYNC_PLL")) g_avsync_pll = 0;     /* B3 closed-loop is DEFAULT-ON (v0.6.20); this reverts to the open-loop B1 content-anchored follow. (PTV_AVSYNC_PLL=1 still honored implicitly = the default.) */
    if (getenv("PTV_ACQ_INSTANT")) g_acq_instant = 1;      /* 1.0.1: revert ACQUIRE to single-window fire (no 3-consecutive-window sustain; the tick floor stays) */
    if (getenv("PTV_NO_PLL_TRACKUP")) g_pll_trackup = 0;   /* 1.0.1-pre3: disable the steer-TRACK entirely (acquire-only; labels flat, no steer — the production mute) */
    /* 0.9.18.7: PTV_PLL_EMA_SHIFT (7) / PTV_PLL_TAU_MS (5000) / PTV_PLL_ACQUIRE_MS (40) /
     * PTV_PLL_ACQUIRE_N (32) / PTV_PLL_REFRACTORY_MS (12000) / PTV_PLL_NOISE_K (3) /
     * PTV_PLL_DEV_SHIFT (9) internalized — see the g_pll_* declarations */
    { const char *tn = getenv("PTV_PLL_TESTNOISE_MS");  if (tn && atoi(tn) > 0) g_pll_testnoise_us  = (int64_t)atoi(tn) * 1000; }  /* TEST-ONLY: inject ±N ms offset square wave */
    if (getenv("PTV_NO_QSHED"))    g_qshed    = 0;   /* 1.0.1-pre8 (a): revert to per-packet tail-drop on video_q overflow (the #32 fragmenter; A/B only) */
    if (getenv("PTV_NO_RATCHREL")) g_ratchrel = 0;   /* 1.0.1-pre8 (b): keep the 6h bank decay even under the starvation contradiction */
    if (getenv("PTV_NO_SELFHEAL")) g_selfheal = 0;   /* 1.0.1-pre8 (c): no internal re-prime on sustained starvation */
    { const char *rs = getenv("PTV_RSYNC_SENSE"); if (rs && !atoi(rs)) g_rsync_sense = 0; }  /* 1.0.1-pre9: passive residual sensor default ON; =0 disables */
    /* 1.0.1-pre14 residual-sync corrector: DEFAULT ON (owner-directed 2026-07-17 — parked and
     * byte-inert on a healthy channel, so every channel runs it unmodified); PTV_NO_RSYNC_CORR=1
     * is the permanent kill switch; sensor off implies corrector off (it is the only input). */
    if (getenv("PTV_NO_RSYNC_CORR") || !g_rsync_sense) g_rsync_corr = 0;
    /* 1.0.1-pre15 glue classification (#33): one revert for the whole classifier. The §3
     * silence-fill is OPT-IN on top (observability-first rollout, owner call); the acorrupt
     * counters/[PTV-ADISC] stay on under every switch (unconditional observability). */
    if (getenv("PTV_NO_GLUECLASS")) g_glueclass = 0;
    if (getenv("PTV_NBS_FILL") && g_glueclass) g_nbs_fill = 1;
    { const char *s = getenv("PTV_GLUE_HTOL_PCT");     if (s && atoi(s) > 0) g_glue_htol = atoi(s); }             /* tuning knob (G4) */
    { const char *s = getenv("PTV_PAIR_EXPECT_TTL_US");if (s && atoll(s) > 0) g_pair_ttl_us = atoll(s); }          /* TEST ONLY (G6) */
    { const char *s = getenv("PTV_NBS_QUANTUM_MS");    if (s && atoi(s) > 0) g_nbs_quantum_us = (int64_t)atoi(s) * 1000; }  /* TEST ONLY (G8) */
    /* TEST-ONLY overrides (PTV_WRAP_GUARD_S precedent): shorten the dwell/quiet windows or move
     * the band/slew so the F-gate fixtures can exercise storm/authority paths in bounded wall
     * time. NEVER set in production — the 5min/3min/80ms/2ms/s defaults are the owner-approved
     * §4/§6 numbers. */
    { const char *s = getenv("PTV_RSCORR_ENGAGE_MS"); if (s && atoi(s) > 0) g_rscorr_engage_us = (int64_t)atoi(s) * 1000; }
    { const char *s = getenv("PTV_RSCORR_DWELL_S");   if (s && atoi(s) > 0) g_rscorr_dwell_us  = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RSCORR_QUIET_S");   if (s && atoi(s) > 0) g_rscorr_quiet_us  = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RSCORR_SLEW");      if (s && atoi(s) > 0) g_rscorr_slew_us_s = atoi(s); }
    if (getenv("PTV_NO_CUSHREL"))  g_cushrel  = 0;   /* 1.0.1-pre10 (e): keep the 6h zero-starvation cushion decay even under the contradiction */
    if (getenv("PTV_NO_CATCHGOV")) g_catchgov = 0;   /* 1.0.1-pre10 (f): deficit-recovery decode back to device max (the 2.2x catch-up bursts) */
    if (!getenv("PTV_NO_PHASEJIT")) {                /* 1.0.1-pre10 (g): per-PID +/-20% shed/heal cycle jitter (deterministic per PID) */
        unsigned jh = (unsigned)getpid() * 2654435761u;   /* Knuth multiplicative hash — spreads adjacent PIDs */
        g_jit_milli = 800 + (int)(jh % 401u);             /* 800..1200 = x0.8..x1.2 */
        /* rr10 advisory 3: multiview slots within ONE process share this jitter value BY
         * DESIGN — the target is de-phasing co-located INSTANCES on a shared device; slots
         * inside one instance already de-phase through their independent input timing. */
    }
    if (getenv("PTV_DEGRADED"))    g_degraded = 1;   /* 1.0.1-pre10 (h): opt-in sustained-deficit every-Kth-GOP admission */
    { const char *s = getenv("PTV_SLOW_US"); g_slow = s ? atoi(s) : 0; }
    { const char *s = getenv("PTV_SLOW_DEC_US"); g_slow_dec = s ? atoi(s) : 0;   /* 1.0.1-pre8 stress knob (slow-NVDEC stand-in) */
      if (g_slow_dec) {
          const char *fr = getenv("PTV_SLOW_DEC_FROM_S"), *fo = getenv("PTV_SLOW_DEC_FOR_S");
          int64_t nw0 = av_gettime_relative();
          g_slow_dec_on_us  = nw0 + (fr ? (int64_t)atoi(fr) * 1000000 : 0);
          g_slow_dec_off_us = fo ? g_slow_dec_on_us + (int64_t)atoi(fo) * 1000000 : 0;
      } }
    if (getenv("PTV_LOG_TS") && atoi(getenv("PTV_LOG_TS")))   /* native [timestamp] log prefix */
        av_log_set_callback(ptv_log_ts_callback);

    if (argc >= 2 && (!strcmp(argv[1], "-version") || !strcmp(argv[1], "--version"))) {
        printf("Perception TV Encoder (ptvencoder) %s  FFmpeg %s\n", PTVENCODER_VERSION, av_version_info());
        printf("  libavutil      %u.%u.%u\n", AV_VERSION_MAJOR(avutil_version()),
               AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
        printf("  libavcodec     %u.%u.%u\n", AV_VERSION_MAJOR(avcodec_version()),
               AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
        printf("  libavformat    %u.%u.%u\n", AV_VERSION_MAJOR(avformat_version()),
               AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
        printf("  libavfilter    %u.%u.%u\n", AV_VERSION_MAJOR(avfilter_version()),
               AV_VERSION_MINOR(avfilter_version()), AV_VERSION_MICRO(avfilter_version()));
        printf("  libswresample  %u.%u.%u\n", AV_VERSION_MAJOR(swresample_version()),
               AV_VERSION_MINOR(swresample_version()), AV_VERSION_MICRO(swresample_version()));
        return 0;
    }
    if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        show_help_default(NULL, NULL); return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "-log-legend")) {   /* full description of every log field/line */
        ptv_print_log_legend(1); return 0;
    }
    if (getenv("PTV_PARSE_DEBUG"))   /* validate the ffmpeg-style parser, then exit */
        return ptv_parse_and_print(argc, argv) < 0 ? 1 : 0;
    if (getenv("PTV_PLAN_DEBUG"))    /* resolve -map/-c against the input, print plan, exit */
        return plan_resolve_and_print(argc, argv) < 0 ? 1 : 0;

    /* ffmpeg-style: split argv into the input (-i) group + output (url) group(s) */
    if (split_commandline(&octx, argc, argv, ptv_options, ptv_groups,
                           sizeof(ptv_groups)/sizeof(ptv_groups[0])) < 0) {
        av_log(NULL, AV_LOG_ERROR, "command parse failed\n"); return 1;
    }
    for (gi = 0; gi < octx.global_opts.nb_opts; gi++) {
        if (!strcmp(octx.global_opts.opts[gi].key, "nostats")) g_stats = 0;        /* honor -nostats */
        if (!strcmp(octx.global_opts.opts[gi].key, "filter_complex"))              /* shared split graph */
            fcomplex = octx.global_opts.opts[gi].val;
        if (!strcmp(octx.global_opts.opts[gi].key, "init_hw_device"))              /* cuda=cuda:N -> GPU N */
            hwdev = octx.global_opts.opts[gi].val;
        if (!strcmp(octx.global_opts.opts[gi].key, "stats_period")) {              /* progress-line interval */
            int64_t p; if (av_parse_time(&p, octx.global_opts.opts[gi].val, 1) >= 0 && p > 0) g_stats_period_us = p;
        }
        if (!strcmp(octx.global_opts.opts[gi].key, "hide_banner")) hide_banner = 1; /* suppress startup banner */
    }
    if (!hide_banner) {
        ptv_show_banner();
        ptv_print_log_legend(0);   /* v0.9.2: compact field legend at the top of every channel log */
    }
    if (octx.groups[1].nb_groups < 1 || octx.groups[0].nb_groups < 1) {
        av_log(NULL, AV_LOG_ERROR,
               "usage: ptvencoder [opts] -i <input> [-filter_complex ..] "
               "[-map .. -c:TYPE .. -b:TYPE ..] <output> [<output> ...]\n");
        uninit_parse_context(&octx); return 1;
    }
    ins  = &octx.groups[1];             /* all input groups (one per -i; 1/2/4 = multiview) */
    outs = &octx.groups[0];             /* all output groups (one per ABR rung) */
    ret  = transcode(ins, outs, fcomplex, hwdev, mode);
    uninit_parse_context(&octx);
    return ret < 0 ? 1 : 0;
}
