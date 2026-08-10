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

#define PTVENCODER_VERSION "1.2.0-pre1"   /* bump per release; notes go in ptvencoder-changelog.md */
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
int     g_mv_birthtrim = 1;
int     g_aglue_ceil = 1;     /* 1.0.1-pre17 fix round (R3c): erase a tripwire-REFUSED label pursuit at the
                               * graph door (butt-joint) instead of letting aresample grind at max rate
                               * forever against an implausible step. PTV_NO_AGLUE_CEIL=1 reverts. */   /* 1.0.1-pre17 (finding 1): mv birth-trim WINDOW — post-preroll backlog is
                               * dropped-oldest silently for the first ~20s instead of being displayed as
                               * a servo-paced catch-up slide the audio follow cannot track (the restart-
                               * with-dead-slot 150-240ms audio-late transient). PTV_NO_MV_BIRTHTRIM=1 reverts. */
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
int     g_convcap = 1;             /* 1.0.1-pre23 #60/#61 BOUNDED CONVERGENCE (the Avivando 183GB OOM class): A single-order
                                           * admission cap + B doubling-ladder escape + C seam-park recurrence guard, one switch.
                                           * PTV_NO_CONVCAP=1 reverts all three to the pre22 hand-everything-to-aresample posture.
                                           * See the PTV_CONV_ESC block in ptvencoder.h for the full pathology + design. */
int64_t g_conv_cap_us = 60LL * 1000000;      /* A: single accepted-convergence order cap (PTV_CONV_CAP_S; default 60s —
                                           * PATRIOT's 30.8s, the largest REAL convergence ever observed, passes with 2× margin;
                                           * a 60s order is ≤ ~23MB of injected silence = allocation-bounded). */
int64_t g_seam_park_us = 3600LL * 1000000;   /* C: rolling recurrence window and park duration (1h).
                                           * PTV_SEAM_PARK_S TEST-ONLY override (G4 gate); never set in production. */
int64_t g_novideo_exit_us = 300LL * 1000000; /* 1.0.1-pre23 startup sanity rider: input packets flowing but no video frame
                                           * decoded for this long since start → FATAL exit (a supervised restart beats a
                                           * wedged forever-startup — the probe-OK-never-decodes silent state, #60 arm D).
                                           * PTV_NOVIDEO_EXIT_S overrides; 0 disables. */
int     g_wallev = 1;              /* 1.0.1-pre24 #63 WALL-EVIDENCE SPLIT (the corrupt-storm desync class): every
                                           * erase engine splits a forward label step J into W = wall-absence-evidenced
                                           * portion (real missing content → PAD) and J−W = flowing/relabel portion
                                           * (label lie → ERASE, as today). PTV_NO_WALLEV=1 reverts the ACTION at every
                                           * touched site (the storm-diag counterfactual); the W measurement itself
                                           * stays on — it feeds the re-anchor corroboration provenance. See the
                                           * PTV_WALLEV_HARDCAP_US block in ptvencoder.h. */
int     g_recanchor = 1;           /* 1.0.1-pre24 #63 CORROBORATED RECOVERY RE-ANCHOR (owner mandate 2026-07-24:
                                           * "perfect lip sync once input is OK again"): one-shot, health-gated, slewed
                                           * base re-anchor of a large stable R — ONLY when the unevidenced-deletion
                                           * ledgers corroborate a real deletion imbalance ≈ R (aseam-class pinned-R
                                           * false positives read 0 there and must never engage — a naive trust of
                                           * large stable R would CREATE a desync on relabel channels).
                                           * PTV_NO_RECANCHOR=1 disables. */
int64_t g_recanchor_settle_us   = 300LL  * 1000000; /* PTV_RECANCHOR_SETTLE_S */
int64_t g_recanchor_cooldown_us = 1800LL * 1000000; /* PTV_RECANCHOR_COOLDOWN_S */
int     g_recanchor_test_abort_n = 0;  /* TEST ONLY (rr24 F2 gate): force ONE mid-walk abort after
                                        * N applied steps, no content event injected — so the clean
                                        * re-engage-on-remainder path can be gated. 0 = off (default,
                                        * byte-identical). PTV_RECANCHOR_TEST_ABORT_N. */
int     g_resync = 1;              /* 1.0.1-pre29 #69 RESYNC (default ON since pre29.1 per the project
                                           * convention — new engines arm by default on the pre train,
                                           * PTV_NO_RESYNC=1 disables; owner 2026-07-29): the
                                           * hard-reset second engage path inside ptv_recanchor for the
                                           * >PTV_RESYNC_MS dead zone. lipsync= R is owner-verified correct
                                           * (4/4 by eye, both signs, up to 20s) — a large STABLE R is a real
                                           * on-air desync, but today the implausibility disarm plus the
                                           * RECANCHOR corroboration refusal leave |R|>5s unowned. A confirmed
                                           * timer (no corroboration — the confirm span + health gates are the
                                           * evidence) fires ONE whole backward step (audio LATE: skip seam)
                                           * or a chunked forward walk (audio EARLY: silence-pad seams), then
                                           * runs the RECANCHOR ledger amnesty. PTV_NO_RESYNC=1 =
                                           * byte-identical pre28 (kill switch). */
int64_t g_resync_ms_us          = 350000;          /* PTV_RESYNC_MS (350ms band threshold) */
int64_t g_resync_ok_us          = 150000;          /* PTV_RESYNC_OK_MS (timer close / walk done) */
int64_t g_resync_confirm_us     = 120LL * 1000000; /* PTV_RESYNC_CONFIRM_S */
int64_t g_resync_confirm_big_us = 60LL  * 1000000; /* PTV_RESYNC_CONFIRM_BIG_S (|R0| > 2s) */
/* NO routine cooldown (owner decision 2026-07-28: a seam costs <1s, a desync costs the whole
 * wait — the fresh confirm window IS the seam-spacing floor, ~1 per 60-120s max). The
 * CIRCUIT BREAKER below exists solely to surface+contain pathological loops (a resync
 * self-oscillation, an extreme source thrash-storm) at ERROR level, with an escalating
 * backoff (#49 ACQUIRE / pre27 AFMT-breaker pattern, fixed 120s base ×2 → 600s cap). */
int     g_resync_breaker_n      = 4;               /* PTV_RESYNC_BREAKER_N: fires within the window that ARM */
int64_t g_resync_breaker_win_us = 900LL  * 1000000;/* PTV_RESYNC_BREAKER_WIN_S */
int64_t g_resync_quiet_us       = 1800LL * 1000000;/* PTV_RESYNC_QUIET_S: R below the band this long
                                                    * DISARMS the breaker and clears the fire history */
int64_t g_resync_chunk_us       = 2000000;         /* PTV_RESYNC_CHUNK_MS (audio-early chunk) */
int64_t g_resync_chunk_gap_us   = 5LL   * 1000000; /* PTV_RESYNC_CHUNK_GAP_S */
/* ---- 1.0.1-pre30 #69 refinements ---- */
int64_t g_resync_seam_hold_us   = 20LL  * 1000000; /* PTV_RESYNC_SEAM_HOLD_S (item A): post-seam
                                                    * sensor hold. Live evidence (i24/Law&Crime
                                                    * 2026-07-29): the instant post-seam reading is
                                                    * garbage-negative and decays ~12ms/s, so the
                                                    * hold buys re-convergence headroom, and the
                                                    * 2-consecutive-stable-samples rule after it
                                                    * (5s spacing, <50ms motion) refuses a reading
                                                    * that is still decaying. 0 disables. */
int     g_resync_vskip          = 1;               /* item B: audio-EARLY default actuator = video
                                                    * IDR-skip jump cut (owner mandate: "drop video,
                                                    * not insert silence"). PTV_RESYNC_SILENCE=1
                                                    * reverts to the pre29 silence-pad chunks; that
                                                    * path also remains the in-walk fallback when no
                                                    * IDR is viable inside the horizon. */
int64_t g_resync_idr_wait_us    = 5LL   * 1000000; /* PTV_RESYNC_IDR_WAIT_S (item B): executor bound
                                                    * waiting for a usable IDR (~2× typical GOP) */
int64_t g_resync_vskip_tol_us   = 250000;          /* PTV_RESYNC_VSKIP_TOL_MS (item B): whole-GOP
                                                    * overshoot tolerance (skip ≤ R + tol) */
int64_t g_resync_walk_ceil_us   = 600LL * 1000000; /* PTV_RESYNC_WALK_CEIL_S (rr30 T1): walk
                                                    * liveness ceiling — a never-stable sensor
                                                    * (readings moving ≥50ms every sample) never
                                                    * satisfies item A's corroboration, so a walk
                                                    * would otherwise pin rsn_active FOREVER
                                                    * (blocking the corrector via resync_owns);
                                                    * abort + settle when no seam/complete lands
                                                    * for this long. 10x the worst measured
                                                    * post-seam settle (~1-2min). 0 disables. */
/* item B cross-thread state — see ptvencoder.h for the handshake/mapping contract */
_Atomic int64_t g_vskip_req_us;
_Atomic int     g_vskip_state;
_Atomic int64_t g_vskip_done_us;
_Atomic int     g_vskip_done_gops;
_Atomic int64_t g_vskip_off_total;
_Atomic int64_t g_vskip_off_before;
_Atomic int64_t g_vskip_from_us;
_Atomic int     g_vskip_epoch;
_Atomic int64_t g_vgop_est_us;
_Atomic int64_t g_vgop_key_wall;
int64_t g_rscorr_slew_fast      = 5000;            /* pre29 adaptive corrector slew above 150ms |R|
                                                    * (PTV_RSCORR_SLEW_FAST µs/s; 0 = always the base
                                                    * clamp). Active regardless of PTV_RESYNC — 0.5%
                                                    * rate is below the pitch JND, inaudible. */
_Atomic int64_t g_rsn_pub[PTV_MAX_AUDIO];          /* pre29: resync fire count (stats rsn= token) */
int     g_muxguard = 1;            /* 1.0.1-pre26 [PTV-MUXGUARD] survive-first mux backstop: drop (never
                                           * re-stamp) any packet whose DTS would EINVAL lavf's per-stream
                                           * monotonic feed check, instead of letting one leaked backward label
                                           * kill all rungs at once (NBS/CORELINK live crash class 2026-07-25).
                                           * PTV_NO_MUXGUARD=1 disables (pre24 EINVAL-exit behavior). */
int64_t g_muxtest_back_at_us = 0;  /* TEST ONLY (pre26 W1/W2 gates): PTV_MUXTEST_BACK_AT_S — inject ONE
                                        * artificial backward audio dts at the mux feed t s after mux start.
                                        * 0 = off (default, byte-identical). */
int64_t g_muxtest_back_ms = 2795;  /* TEST ONLY: PTV_MUXTEST_BACK_MS (default ≈ the live −2794.7ms). */
int64_t g_muxtest_back_hold_us = 0; /* TEST ONLY (pre27 #62 GF gate): PTV_MUXTEST_BACK_HOLD_S — with >0,
                                        * the backward-dts injection above applies to EVERY pkt of the
                                        * target type in [back_at, back_at+hold) instead of once, so
                                        * [PTV-MUXGUARD] drops continuously (exercises the 60s drop-span
                                        * ceiling). 0 = off. */
int     g_muxtest_back_type = AVMEDIA_TYPE_AUDIO; /* TEST ONLY (pre27 #62 GH gate):
                                        * PTV_MUXTEST_BACK_TYPE=a|s|d — stream type the injection
                                        * targets (audio default; s/d = sparse guarded streams, to
                                        * verify the ceiling does NOT fire there). */
int     g_muxdiag = 0;             /* pre26 D3 instrumentation (PTV_MUXDIAG=1): emission-point backward-label
                                        * detector + state dump in the audio thread. Diagnostic only. */
int     g_muxtol = 1;              /* 1.0.1-pre27 #62 [PTV-MUXTOL] egress-pressure errno filter (Praise_TV
                                        * glo-2 death loop 2026-07-24: bitrate=-paced udp egress with a ~6.4s
                                        * fifo_size — every respawn's startup catch-up burst filled the fifo
                                        * ~6-7s after anchor, udp.c returned ENOMEM, the pre26 always-fatal
                                        * path killed the channel again; 7 deaths, channel dark). ENOMEM/
                                        * EAGAIN from the mux write = transient egress pressure: DROP the pkt
                                        * + count + rate-limited warn and keep muxing; 60s after the FIRST
                                        * failure of a failing run with no success in between escalates to
                                        * the fatal path (a dark channel needs the respawn anyway — never a
                                        * forever-throttled zombie). Everything else (EINVAL label corruption = the pre26
                                        * crash class, EPIPE, EIO, ...) stays immediately fatal.
                                        * PTV_NO_MUXTOL=1 reverts (any write error = fatal, pre26 behavior). */
int     g_muxfail_sim_err = 0;     /* TEST ONLY (pre27 #62 gates): PTV_MUXFAIL_SIM="<errno>:<start_s>:<dur_s>"
                                        * (errno = enomem|eagain|einval) — the write path pretends
                                        * av_interleaved_write_frame returned that error (pkt NOT written)
                                        * during the window, measured from mux start. 0 = off (unset =
                                        * byte-identical). */
int64_t g_muxfail_sim_from_us = 0, g_muxfail_sim_to_us = 0;
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
int     g_adts_split = 1;          /* 1.0.1-pre19.1 broken-phase audio "ticking" fix (PTV_NO_ADTS_SPLIT=1 reverts to
                                           * pre19): Azorse opened DURING a broken-7.1 phase → lavf's STRICT probe decoder
                                           * rejects every frame → find_stream_info fails (sample_rate 0) → the parser-split
                                           * ADTS frames 2..6 of each 6-frame PES get NO pts (no duration to extrapolate
                                           * with) → the audio_push NOPTS drop rule discarded 5/6 frames = 107ms hole per
                                           * 128ms. Gates BOTH halves of the fix: (1) tolerant probe (ptv_find_stream_info)
                                           * and (2) the [PTV-ASTAMP] decoded-frame extrapolation backstop. */
int     g_tolerant_dec = 1;        /* 1.0.1-pre19.1 kill for the pre19 #38 tolerant AAC channel-element allocation
                                           * (PTV_NO_TOLERANT_DEC=1 = strict lavc everywhere, the pre-#38 behavior:
                                           * broken-7.1 phases flood decode errors and stay silent-but-alive). Per-channel
                                           * control, owner request. Also gates the probe-side tolerance. */
int     g_achop = 1;               /* 1.0.1-pre19 #46 [PTV-ACHOP] stuck-chop escape (PTV_NO_ACHOP_REBUILD=1 kills):
                                           * sustained per-track chop (decode-error/self-shed rate above the floors for
                                           * g_achop_sust_min minutes — the Azorse never-ending-corruption variant, slot
                                           * warbles until restart) triggers a FULL audio-path rebuild (decoder swap +
                                           * graph/swr teardown + AFMT impossible-seed), rate-limited per track. */
int     g_achop_errs_min  = 60;    /* PTV_ACHOP_ERRS_MIN: chop floor, decode errors per minute */
int     g_achop_sheds_min = 120;   /* PTV_ACHOP_SHEDS_MIN: chop floor, self-shed packets per minute */
int     g_achop_sust_min  = 3;     /* PTV_ACHOP_SUST_MIN: minutes the rate must hold before the escape */
int64_t g_achop_relimit_us = 600000000; /* PTV_ACHOP_RELIMIT_S: min wall between escapes per track (default 10min) */
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
/* d1-fix: lone-audio LAYERA event whose flowing video already sits ON the jump-target
 * timeline anchors the event on video at offset 0 (the audio label step stays on the wire,
 * registered for the content path — the D1 handshake) instead of the 3b provisional
 * butt-joint erase + flowing-video discard the pre7 false-crossing gate left this shape
 * with (live Grid_2x2 2026-07-21: audio 1.32s early + video-ahead re-anchor).
 * PTV_NO_VANCHOR=1 reverts. */
int     g_vanchor = 1;
/* 1.0.1-pre22 (a-anchor): the role-swapped mirror of the d1-fix above — a lone VIDEO forward
 * jump whose audio partner never crosses inside the pairing window (Fashion 2026-07-22 13:21:
 * +5.520s video-only, one-sided relabel-erase, ev −5.480s, R pinned, corrector DISARMed, wire
 * desynced by the jump if the video content really skipped). The lone-video flush seeds a
 * provisional ZERO-offset audio leg ("audio didn't jump") and, if the window EXPIRES with no
 * real audio crossing, the audio is re-based onto the video-defined timeline and the full
 * A-vs-V mismatch (= the video delta) is REGISTERED for the content path (ptv_pair_expect) —
 * the D1 handshake, deferred to expiry because the absence of the audio leg is only provable
 * then (an immediate shift would destroy the genuine staggered pair the 3a INHERIT handles).
 * PTV_NO_AANCHOR=1 reverts. */
int     g_aanchor = 1;
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
/* 1.0.1-pre21 #24: the mv audio-follow PLL yields (actuators frozen) while the SAME track's
 * residual corrector is STEERING — the two loops share the graph-door actuator and the PLL's
 * output-alignment objective cancels the corrector 1:1 (p24 gate-1), making mv ENGAGE→PARK
 * structurally unreachable. PTV_NO_PLL_YIELD=1 reverts the arbitration only. */
int     g_pll_yield = 1;
/* 1.0.1-pre21 heartbeats (see ptvencoder.h for the design note). */
_Atomic int     g_hb_vdec_pos[PTV_MAX_INPUT];
_Atomic int64_t g_hb_vdec_wall[PTV_MAX_INPUT];
_Atomic int64_t g_hb_vdec_dec[PTV_MAX_INPUT];
_Atomic int64_t g_hb_demux_wall[PTV_MAX_INPUT];
_Atomic int     g_hb_vq[PTV_MAX_INPUT];
_Atomic int     g_hb_out_pos;
_Atomic int64_t g_hb_out_wall;
int64_t g_test_vdec_stall_us;   /* TEST-ONLY (PTV_TEST_VDEC_STALL_S, precedent PTV_SLOW_US):
                                 * sleep the vdec loop once after 60s of runtime so the
                                 * [PTV-STALL] path live-fires in a gate — never ship an
                                 * unfired diagnostic (the pre20 silent-zombie lesson). */
int64_t g_test_vdec_stall_at_us = 60LL * 1000000;   /* TEST-ONLY (1.0.1-pre23, PTV_TEST_VDEC_STALL_AT_S):
                                 * the stall trigger time — 0 lands the stall BEFORE the first
                                 * decode (the startup-wedge shape the [PTV-NOVIDEO] gate needs). */
const char *ptv_hb_name(int pos)
{
    switch (pos) {
    case PTV_HB_VDEC_QRECV:     return "q_recv";
    case PTV_HB_VDEC_BANK:      return "bank";
    case PTV_HB_VDEC_SENDPKT:   return "send_pkt";
    case PTV_HB_VDEC_RECVFRAME: return "recv_frame";
    case PTV_HB_VDEC_HWUP:      return "hw_upload";
    case PTV_HB_VDEC_FQSEND:    return "fq_send";
    case PTV_HB_OUT_LOOP:       return "loop";
    case PTV_HB_OUT_ENC:        return "encode_push";
    case PTV_HB_OUT_STATS:      return "stats";
    default:                    return "unknown";
    }
}
int     g_acq_instant = 0;            /* 1.0.1 (PTV_ACQ_INSTANT=1 reverts): ACQUIRE needs the |EMA offset| above threshold for 3 CONSECUTIVE debounce windows (and the threshold is floored at 1.5 house ticks) — the vlag measurement is tick-quantized, so the single-window fire snapped on its own quantization noise (live grids: ~939-1511 ACQUIREs/22h alternating ±42ms pad/drop). */
int     g_pll_trackup = 1;            /* 1.0.1-pre3 (PTV_NO_PLL_TRACKUP=1 disables TRACK entirely = acquire-only, labels flat — the operators' production mute keeps its meaning): TRACK now steers through the RESAMPLER (af_steer_us into the graph-input pts, AVLOCK-style) instead of re-stamping output labels. pre2's label-TRACK stretched output AAC pts spacing up to +158ms/min during integration episodes → PTS-honoring players rate-chased it = audible warble (production 2026-07-13). The pre2 [PTV-TRACKUP] direction-aware anti-windup is retired with the label actuator. */
int64_t g_pll_testnoise_us = 0;       /* TEST-ONLY (default off): inject a ±N ms square wave (flips ~every 3.2s) into the measured offset to REPRODUCE the box limit cycle locally (local sources are clean). PTV_PLL_TESTNOISE_MS sets it; never set in production. */
int     g_pll_testnoise_frames = 330; /* TEST-ONLY (1.0.1-pre18, #49 gate): half-period in frames (PTV_PLL_TESTNOISE_P) — slow flips (~30s ≫ pll_dev τ) model the erase-class FLAT-step storm the 7s default cannot (dev adapts to fast flips) */
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
_Atomic int     g_selfheal_req;
_Atomic int64_t g_v_arrive_wc;
/* 1.0.1-pre17: sibling-slate mask (bit k = input slot k black-slated; compositor writes,
 * rscorr_event_active reads) — no mv corrector engagement while any slot is slated. */
_Atomic int     g_mv_slate_mask;
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
_Atomic int64_t g_conv_pub[PTV_MAX_AUDIO];       /* 1.0.1-pre23 rr23: cumulative folded label motion (us);
                                                  * nonzero => the stats line grows a conv= token — on a
                                                  * fold/park channel the lipsync=/async= stats include the
                                                  * folded label divergence BY DESIGN (read the wire +
                                                  * [PTV-CONV] lines); conv= quantifies it for monitors. */
_Atomic int64_t g_conv_park_pub[PTV_MAX_AUDIO];  /* rr23: seam_park_until (wall us; 0/past = not parked) —
                                                  * the builder derives the P suffix by timestamp so the
                                                  * display can never show a stale park. */
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
/* 1.0.1-pre18 #50 — GAP-VERDICT vs LAYERA one-remedy invariant (the AWE_Plus +2.38s live
 * defect 2026-07-19 14:44): one source event must get exactly ONE remedy. A gap verdict
 * landing while a matching LAYERA cycle is armed-but-unflushed DISBANDS the cycle (labels
 * carry the jump on every stream; AGLUE/aresample pads — the content really is missing);
 * a verdict landing just AFTER a matching flush is SUPPRESSED (the flush already remedied
 * the event; the jump goes to the discontinuity layer so both streams get the same remedy).
 * Plus the E5 net: LAYERA-flush relabels are published per track (atomics below) so the
 * pad round-trip ledger can cancel a pad whose return leg was erased at the packet layer.
 * PTV_NO_GLUEVETO=1 reverts all three. */
int     g_glueveto = 1;
_Atomic int64_t g_flush_relab_step[PTV_MAX_AUDIO];  /* last LAYERA-flush label shift per track (µs) */
_Atomic int64_t g_flush_relab_wc[PTV_MAX_AUDIO];    /* wall µs it was persisted (0 = never) */
_Atomic int64_t g_reopen_wc[PTV_MAX_INPUT];         /* 1.0.1-pre20 rider (a): per-input demux-reopen stamp (ASTAMP carry inval) */
int64_t g_t_us = 0;                                 /* 1.0.1-pre20 rider (b): -t output-media-time bound (0 = none) */
_Atomic int g_t_stop;                               /* set by the cadence owner at -t; demux threads treat it as EOF */
/* 1.0.1-pre18 #51a — corrector hs-tick event filter (AWE dwell starvation, live 2026-07-19):
 * a ±1-video-tick house_skew step is pulldown/decim cadence noise, not a lineage event —
 * it must neither reset the dwell nor count toward the storm (rscorr_event_edge).
 * PTV_NO_HSTICK_FILTER=1 reverts to counting every cumulative-50ms hs move. */
int     g_hstick_filter = 1;
/* 1.0.1-pre18 #51b — corrector anti-starvation ceiling (legacy-0007 PLL_HARD_CEILING/
 * PLL_STUCK lineage): R large + flat for ≥ the ceiling with the dwell never completing
 * (event resets + storm holdoffs included) → ENGAGE anyway, one WARNING. Flatness is
 * load-bearing (churning R never ceiling-engages); sensor-invalid/delivery-dead still
 * block. PTV_NO_RSCORR_CEIL=1 reverts; PTV_RSCORR_CEIL_MIN (minutes, default 15) tunes. */
int     g_rscorr_ceil = 1;
int64_t g_rscorr_ceil_us = 900000000;
/* 1.0.1-pre18 — per-stream delivery watermarks (pre17 (B) KNOWN OPEN, owner-approved; owner
 * mandate: af-independent transport on single input AND mv): the §7.5b video hold keys on
 * the SLOWEST currently-LIVE gated audio stream's delivered watermark, staleness-excluded
 * (>2s silent = out), instead of the least-delayed aggregate high-water. Mixed rungs
 * (loudnorm AAC + copied AC-3) stop depending on latency coincidence.
 * PTV_NO_PERSTREAM_WM=1 reverts to the aggregate key. */
int     g_perstream_wm = 1;
/* 1.0.1-pre20 REBUILD RE-ANCHOR (default ON): at [PTV-AFMT] rebuild completion the track's
 * audio base is re-derived birth-equivalent from the current house mapping instead of
 * carrying pre-rebuild state (see the AudioState.reanch_* comment in ptvencoder.h).
 * PTV_NO_REBUILD_REANCHOR=1 reverts to the pre19.1 carried-base posture. */
int     g_rebuild_reanchor = 1;
/* 1.0.1-pre18 #49 — mv PLL repeated-ACQUIRE backoff: erase-class corruption re-anchors a
 * slot ±one flat step per 12s refractory forever (audible warble until restart) — the flat
 * step defeats both the noise-adaptive threshold (it measures jitter, not flat flips) and
 * the 3-window sustain (the step IS stable). Each ACQUIRE within 60s doubles the acquire
 * threshold (decays back one level per acquire-free 60s). PTV_NO_ACQ_BACKOFF=1 reverts. */
int     g_acq_backoff = 1;
/* EIA-608 -> DVB-teletext closed-caption extraction. OPT-IN per output (-cc_extract, with
 * -cc_lang for the G0 national subset + stream language); PTV_NO_CC=1 is the runtime kill
 * switch that keeps the whole path inert even when the CLI asks for it. Without
 * -cc_extract nothing is allocated, no stream is created and no packet is produced — the
 * output is byte-identical to a build without this feature. */
int     g_aenc_share = 1;   /* share one audio encoder between rungs with identical settings */
int     g_cc = 1;
/* atomic because the "emitter thread create failed" path clears it AFTER the output threads
 * (which read it for the stats line) are already running */
_Atomic int g_cc_on;
_Atomic int64_t g_cc_a53, g_cc_caps, g_cc_erase, g_cc_keep, g_cc_err, g_cc_dropped,
                g_cc_bump, g_cc_reset;
_Atomic int64_t g_cc_caps_in[PTV_MAX_INPUT], g_cc_a53_in[PTV_MAX_INPUT];
_Atomic int64_t g_cc_eskip;   /* erases refused for the minimum-display rule */

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

/* Map an ISO 639-2 language code to the teletext G0 national option subset (ETS 300 706
 * table 33). Ported verbatim from legacy patch 0005; unknown languages get 0 (English). */
static int lang_to_g0_subset(const char *lang)
{
    if (!lang || !lang[0])
        return 0;
    if (!strcmp(lang, "deu") || !strcmp(lang, "ger"))
        return 1;
    if (!strcmp(lang, "swe") || !strcmp(lang, "fin") || !strcmp(lang, "hun"))
        return 2;
    if (!strcmp(lang, "ita"))
        return 3;
    if (!strcmp(lang, "fra") || !strcmp(lang, "fre"))
        return 4;
    if (!strcmp(lang, "spa") || !strcmp(lang, "por"))
        return 5;
    if (!strcmp(lang, "cze") || !strcmp(lang, "ces") ||
        !strcmp(lang, "slo") || !strcmp(lang, "slk"))
        return 6;
    return 0;
}

/* free func for the CC event queue (the queue owns the ASS string) */
static void free_cc_msg(void *msg) { CcEvent *e = msg; av_freep(&e->ass); }

/* Does this cc_dec ASS dialog line carry anything displayable? The line is
 * "<readorder>,<layer>,<style>,<speaker>,0,0,0,,<text>" (ff_ass_get_dialog), so the text
 * begins after the 8th comma; {...} override blocks, the \N/\n/\h escapes and whitespace
 * are markup, not content — the same test the dvb_teletext encoder reaches after its ASS
 * split. An EIA-608 EDM/EOC (the source cleared its caption memory) arrives as a rect whose
 * text is empty by this test: that, NOT a zero-rect subtitle, is what a source erase looks
 * like coming out of cc_dec (it sets got_sub = num_rects > 0, so it never emits one).
 * This only CLASSIFIES the event — for caps= vs ccera= and for the QUIET watchdog. The
 * encoder stays the authority for what goes on the wire, and gets the rect either way. */
static int cc_ass_has_text(const char *ass)
{
    const unsigned char *p = (const unsigned char *)ass;
    int commas = 0;

    for (; *p && commas < 8; p++)
        if (*p == ',')
            commas++;
    if (commas < 8)
        return 1;              /* not a shape we understand — treat as content, never erase */
    for (; *p; p++) {
        /* "{\" — and ONLY "{\" — opens an override block, exactly as
         * ff_ass_split_override_codes tests it. A bare '{' is literal text there, and
         * cc_dec really can emit one (the extended Portuguese/German/Danish charset maps
         * 0x29/0x2a to '{'/'}' and does no ASS escaping), so skipping every '{' would
         * classify a caption reading "{Music}" as an erase. */
        if (p[0] == '{' && p[1] == '\\') {
            while (*p && *p != '}')
                p++;
            if (!*p)
                break;
            continue;
        }
        if (*p == '\\' && (p[1] == 'N' || p[1] == 'n' || p[1] == 'h')) {
            p++;
            continue;
        }
        /* Codepoints the teletext G0 mapping blanks are not content either: NBSP, the full
         * block real EIA-608 encoders use for cleared rows, and the box-drawing corners.
         * The encoder erases the page on a row made only of these, so calling them a
         * caption would leave the QUIET watchdog asleep through a real blanking. UTF-8
         * byte forms; keep in step with utf8_to_teletext_g0() in dvbteletextenc.c. The
         * p[1]-before-p[2] ordering is load-bearing — short-circuit stops the read at the
         * terminating NUL. */
        if (p[0] == 0xC2 && p[1] == 0xA0) {                        /* U+00A0 NBSP */
            p++;
            continue;
        }
        if (p[0] == 0xE2 &&
            ((p[1] == 0x96 && p[2] == 0x88) ||                     /* U+2588 █ */
             (p[1] == 0x94 && (p[2] == 0x8C || p[2] == 0x90 ||     /* U+250C ┌  U+2510 ┐ */
                               p[2] == 0x94 || p[2] == 0x98)))) {  /* U+2514 └  U+2518 ┘ */
            p += 2;
            continue;
        }
        if (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n')
            return 1;
    }
    return 0;
}

/* DECODE-THREAD CC tap (see the CcTap header in ptvencoder.h). Feeds this frame's A53 CC
 * bytes to cc_dec and hands the resulting caption — or the 1 Hz zero-rect keepalive — to
 * the emitter together with the frame's SOURCE pts (the emitter, not us, does the
 * house-clock stamping). Two invariants:
 *   - the A53 side data is ALWAYS stripped, on every path, so the video encoder cannot
 *     re-inject the captions into its own SEI (they would then be duplicated, and stamped
 *     on the source timeline);
 *   - we NEVER block. The push is NONBLOCK and a full queue drops the newest event: a
 *     wedged/slow emitter must not cost video frames.
 * The keepalive is NOT coalescing (the CDN owns that): it is what makes the encoder's 10s
 * auto-erase reachable and what keeps a sparse SUBTITLE stream from gating lavf's
 * interleaver (mux.c counts SUBTITLE in nb_interleaved_streams; ptvencoder only bounds
 * that with max_interleave_delta). */
/* decode-side twin of cc_tag(): "" single-input, " in2" on a mosaic */
static const char *cc_tap_tag(CcTap *t, char *buf, size_t n)
{
    if (!t->multi) { buf[0] = 0; return buf; }
    snprintf(buf, n, " in%d", t->slot);
    return buf;
}

/* Push one event to the emitter. NEVER blocks: a full queue drops the NEWEST and the
 * counter carries it — a wedged emitter must not cost a video frame. Takes ownership of
 * ev->ass either way. */
static void cc_tap_send(CcTap *t, CcEvent *ev, char *ttag)
{
    if (av_thread_message_queue_send(t->q, ev, AV_THREAD_MESSAGE_NONBLOCK) < 0) {
        av_freep(&ev->ass);
        atomic_fetch_add_explicit(&g_cc_dropped, 1, memory_order_relaxed);
        if (!t->lost_warned) {
            t->lost_warned = 1;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-CC]%s event queue full or closed — caption events are being DROPPED "
                   "(video is unaffected; see ccdrop= on the stats line)\n",
                   cc_tap_tag(t, ttag, sizeof(char[16])));
        }
    } else
        t->last_evt_us = ev->src_us;
}

static void cc_tap_frame(CcTap *t, AVFrame *frame, AVRational ist_tb)
{
    AVFrameSideData *sd;
    char ttag[16];
    AVSubtitle sub = { 0 };
    CcEvent ev = { NULL, AV_NOPTS_VALUE, 0, PTV_CC_KEEPALIVE };
    int64_t cur_us;
    int got = 0;

    if (!t->dec || !t->q || frame->best_effort_timestamp == AV_NOPTS_VALUE)
        goto strip;
    cur_us = av_rescale_q(frame->best_effort_timestamp, ist_tb, AV_TIME_BASE_Q);

    /* A backward source step this large is a rebased/glued timeline (LAYERA, wrap edge,
     * splice): the half-assembled caption inside cc_dec belongs to the old one, and the
     * keepalive phase is meaningless. Drop both. */
    if (t->last_src_us != AV_NOPTS_VALUE && cur_us < t->last_src_us - PTV_CC_BACKSTEP_US) {
        avcodec_flush_buffers(t->dec);
        t->last_evt_us = AV_NOPTS_VALUE;
        atomic_fetch_add_explicit(&g_cc_reset, 1, memory_order_relaxed);
        av_log(NULL, AV_LOG_INFO,
               "[PTV-CC]%s source pts stepped back %"PRId64"ms — cc_dec state reset\n",
               cc_tap_tag(t, ttag, sizeof ttag), (t->last_src_us - cur_us) / 1000);
    }
    t->last_src_us = cur_us;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_A53_CC);
    if (sd && sd->size >= 3) {
        AVPacket *cc_pkt = av_packet_alloc();
        atomic_fetch_add_explicit(&g_cc_a53, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_cc_a53_in[t->slot], 1, memory_order_relaxed);
        if (cc_pkt && av_new_packet(cc_pkt, sd->size) >= 0) {
            int ret;
            memcpy(cc_pkt->data, sd->data, sd->size);
            cc_pkt->pts = cur_us;                 /* cc_dec pkt_timebase is AV_TIME_BASE_Q */
            sub.pts     = cur_us;
            ret = avcodec_decode_subtitle2(t->dec, &sub, &got, cc_pkt);
            if (ret < 0) {
                /* counted and reported, NOT swallowed — a source whose 608 pairs are
                 * consistently rejected is a real defect, and cc= going flat with ccerr=
                 * climbing is what says so. */
                int64_t nw = av_gettime_relative();
                got = 0;
                atomic_fetch_add_explicit(&g_cc_err, 1, memory_order_relaxed);
                if (nw - t->err_log_us > 10000000) {       /* one line per 10s */
                    t->err_log_us = nw;
                    av_log(NULL, AV_LOG_WARNING, "[PTV-CC]%s EIA-608 decode error: %s\n",
                           cc_tap_tag(t, ttag, sizeof ttag), av_err2str(ret));
                }
            }
        }
        av_packet_free(&cc_pkt);
    }
    if (got && sub.num_rects > 0 && sub.rects[0]->ass) {
        /* cc_dec emits a single ASS rect per subtitle; rect 0 is the caption. Forward it
         * verbatim EITHER WAY — a text-bearing caption and a source erase (an empty text
         * field, see cc_ass_has_text) differ only in how we count them here. The encoder
         * reads the same rect and answers with a page update or a page erase; sending a
         * zero-rect subtitle in the erase case instead would be WRONG, because zero rects
         * is the keepalive contract and only erases after the encoder's 10s timeout. */
        const char *ass = sub.rects[0]->ass;
        int end_ms = (sub.end_display_time == UINT32_MAX ||
                      sub.end_display_time > PTV_CC_MAX_DISPLAY_MS)
                     ? PTV_CC_MAX_DISPLAY_MS : sub.end_display_time;
        if (cc_ass_has_text(ass)) {
            /* DEBOUNCE (see PTV_CC_DEBOUNCE_US): buffer it. Only a CHANGE restarts the
             * silence timer — an identical retransmission must not keep the caption
             * pending forever. */
            if (!t->pend_ass || strcmp(t->pend_ass, ass)) {
                if (!t->pend_ass)
                    t->pend_first_us = cur_us;        /* new pending cycle */
                av_freep(&t->pend_ass);
                t->pend_ass = av_strdup(ass);
                t->pend_changed_us = cur_us;
                t->pend_end_ms = end_ms;
            }
        } else {
            /* Source erase (EDM/EOC). REFUSE it if the page has not had its minimum time on
             * screen: a roll-up clears rows constantly, and obeying every clear reduced real
             * captions to 0.3-0.6s of display. Dropping the erase is safe — the next caption
             * replaces the page, and the encoder's 10s stale-content timeout still clears an
             * abandoned one. */
            if (t->shown_since_us != AV_NOPTS_VALUE &&
                cur_us - t->shown_since_us < PTV_CC_MIN_DISPLAY_US) {
                avsubtitle_free(&sub);
                atomic_fetch_add_explicit(&g_cc_eskip, 1, memory_order_relaxed);
                goto strip;
            }
            /* Whatever is pending was on screen up to now, so it must go out BEFORE the
             * clear, otherwise a caption that ends the moment it completes is never seen. */
            if (t->pend_ass) {
                CcEvent pe = { t->pend_ass, cur_us, t->pend_end_ms, PTV_CC_CAPTION };
                t->pend_ass = NULL;
                cc_tap_send(t, &pe, ttag);
            }
            ev.ass    = av_strdup(ass);
            ev.kind   = PTV_CC_ERASE;
            ev.end_ms = end_ms;
            ev.src_us = cur_us;
            avsubtitle_free(&sub);
            cc_tap_send(t, &ev, ttag);
            goto strip;
        }
        avsubtitle_free(&sub);
    } else {
        /* Belt-and-braces. avcodec_decode_subtitle2 already frees on its own error paths
         * (and avsubtitle_free memsets, so this is a no-op, never a double free); what it
         * genuinely covers is got != 0 with no usable rect 0, which the branch above skips.
         * Cheap insurance on a path that runs for months. */
        avsubtitle_free(&sub);
    }

    /* --- the two timers --- */
    if (t->pend_ass &&
        (cur_us - t->pend_changed_us >= PTV_CC_DEBOUNCE_US ||     /* text went quiet */
         cur_us - t->pend_first_us   >= PTV_CC_DEADLINE_US)) {    /* roll-up deadline */
        CcEvent pe = { t->pend_ass, cur_us, t->pend_end_ms, PTV_CC_CAPTION };
        t->pend_ass = NULL;
        cc_tap_send(t, &pe, ttag);
        t->shown_since_us = cur_us;               /* min-display clock starts now */
        goto strip;
    }
    if (t->last_evt_us == AV_NOPTS_VALUE) {
        t->last_evt_us = cur_us;                  /* phase the cadence on the first frame */
        goto strip;
    }
    if (t->pend_ass || cur_us - t->last_evt_us < PTV_CC_KEEPALIVE_US)
        goto strip;                               /* don't keepalive over a pending caption */
    ev.src_us = cur_us;                           /* ass == NULL => zero-rect keepalive */
    cc_tap_send(t, &ev, ttag);

strip:
    /* ALWAYS, on every path: both h264_nvenc (a53_cc, default on) and h264_videotoolbox
     * re-inject this side data as SEI, which would duplicate the captions on the output
     * AND stamp them on the source timeline. */
    av_frame_remove_side_data(frame, AV_FRAME_DATA_A53_CC);
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
    int64_t hb_t0 = av_gettime_relative();   /* 1.0.1-pre21: heartbeat + test-stall runtime anchor */
    int     hb_test_done = 0;

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
               && av_gettime_relative() - t0 < budget) {
            PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_BANK);   /* pre21: banking is residence, not a stall */
            av_usleep(5000);
        }
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
                    if (d->shed_wall) atomic_store_explicit(d->shed_wall, nw, memory_order_relaxed);
                    if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, shed_n, memory_order_relaxed);
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
        /* 1.0.1-pre30 #69 (item B) VIDEO IDR-SKIP EXECUTOR: the resync audio-EARLY actuator.
         * The audio thread posted a target span; skip that much video content FORWARD here at
         * the video_q pop site — upstream of decode and the rung split, so every rung skips
         * the same content coherently — by dropping whole GOPs from the head, exactly the
         * pre8 QSHED shape (the stop KEY decodes; the decoder input stays contiguous whole
         * GOPs). Greedy whole-GOP rule: keep dropping while (span-to-this-key + gop_est) ≤
         * target + tol — the largest whole-GOP total ≤ R + tol; the sub-GOP residual goes to
         * the corrector (the same +125ms handoff as a chunked walk). Label-neutrality: the
         * achieved span is published as a src-keyed offset that content_index() subtracts for
         * post-boundary content, so output labels continue monotonically while content jumps
         * (the video mirror of the audio-late reanch_mono skip seam); the m_v EMA re-seeds at
         * the first post-boundary emit (g_vskip_epoch). If no usable IDR appears within
         * PTV_RESYNC_IDR_WAIT_S, escape: resume decoding mid-GOP UNFLUSHED (Session-109
         * posture — rr30 honesty: on SW h264 the frame_num-gap concealment decodes those
         * frames against stale/dummy refs with NO corrupt flag, so expect smear, not a
         * clean freeze, until the next key; flushing instead would permanently freeze a
         * no-IDR source) and report the partial span so the walk falls back to silence
         * chunks. */
        if (!pkt && g_resync && g_resync_vskip && d->live && !d->hold) {
            int64_t tgt = atomic_exchange_explicit(&g_vskip_req_us, 0, memory_order_relaxed);
            if (tgt > 0) {
                int64_t deadline = av_gettime_relative() + g_resync_idr_wait_us;
                int64_t gop_est  = atomic_load_explicit(&g_vgop_est_us, memory_order_relaxed);
                int64_t d0 = AV_NOPTS_VALUE, dl = AV_NOPTS_VALUE, lkey = AV_NOPTS_VALUE;
                int64_t achieved = 0, boundary = AV_NOPTS_VALUE;
                int     gops = 0, ndrop = 0, escape = 0;
                if (gop_est <= 0) gop_est = 1000000;             /* fire-side gate vetted it; belt+braces */
                for (;;) {
                    AVPacket *sp;
                    int rc = av_thread_message_queue_recv(d->video_q, &sp,
                                                          AV_THREAD_MESSAGE_NONBLOCK);
                    if (rc == AVERROR(EAGAIN)) {
                        if (av_gettime_relative() > deadline) { escape = 1; break; }
                        PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_QRECV);   /* waiting = residence, not a stall */
                        av_usleep(5000);
                        continue;
                    }
                    if (rc < 0) { escape = 1; break; }           /* queue closed/EOF: stop */
                    if ((sp->flags & AV_PKT_FLAG_KEY) && d0 != AV_NOPTS_VALUE) {
                        int64_t kts  = sp->dts != AV_NOPTS_VALUE ? sp->dts : sp->pts;
                        int64_t span = kts != AV_NOPTS_VALUE
                            ? av_rescale_q(kts - d0, d->ist_tb, AV_TIME_BASE_Q) : 0;
                        if (lkey != AV_NOPTS_VALUE && kts != AV_NOPTS_VALUE && kts > lkey) {
                            int64_t g = av_rescale_q(kts - lkey, d->ist_tb, AV_TIME_BASE_Q);
                            if (g > 0 && g < 30000000) gop_est = g;   /* live update while walking */
                        }
                        if (span + gop_est > tgt + g_resync_vskip_tol_us || span >= tgt) {
                            pkt      = sp;                       /* STOP: this key DECODES */
                            achieved = span;
                            boundary = sp->pts != AV_NOPTS_VALUE
                                ? av_rescale_q(sp->pts, d->ist_tb, AV_TIME_BASE_Q)
                                : av_rescale_q(kts, d->ist_tb, AV_TIME_BASE_Q);
                            break;
                        }
                        lkey = kts;                              /* this whole GOP goes too */
                        gops++;
                    } else if ((sp->flags & AV_PKT_FLAG_KEY) && d0 == AV_NOPTS_VALUE) {
                        lkey = sp->dts != AV_NOPTS_VALUE ? sp->dts : sp->pts;
                        gops++;                                  /* head IS a key: GOP 1 starts the skip */
                    }
                    if (sp->dts != AV_NOPTS_VALUE) {
                        if (d0 == AV_NOPTS_VALUE) d0 = sp->dts;
                        dl = sp->dts;
                    }
                    ndrop++;
                    av_packet_free(&sp);
                    /* rr30 (T4): RUNNING SPAN BOUND — the stop rule only evaluates at KEYS,
                     * so a GOP that turns out longer than the estimate (film→ad break,
                     * 1s→4s) would otherwise drop its WHOLE real length before the next key
                     * (worst case: audio flips LATE by new-GOP − R, a bigger desync than
                     * the one being fixed, then hold+abort+settle before recovery). Bound
                     * the dropped span at target+tol at PACKET granularity: past it, escape
                     * mid-GOP exactly like the deadline path (partial span, silence chunks
                     * own the remainder). */
                    if (d0 != AV_NOPTS_VALUE && dl != AV_NOPTS_VALUE && dl > d0 &&
                        av_rescale_q(dl - d0, d->ist_tb, AV_TIME_BASE_Q) >
                            tgt + g_resync_vskip_tol_us) {
                        escape = 1;
                        break;
                    }
                }
                if (escape && d0 != AV_NOPTS_VALUE && dl != AV_NOPTS_VALUE && dl > d0) {
                    achieved = av_rescale_q(dl - d0, d->ist_tb, AV_TIME_BASE_Q);
                    boundary = av_rescale_q(dl, d->ist_tb, AV_TIME_BASE_Q) + 1000;   /* just past the last drop */
                }
                if (achieved > 0 && boundary != AV_NOPTS_VALUE) {
                    int64_t nw  = av_gettime_relative();
                    int64_t old = atomic_load_explicit(&g_vskip_off_total, memory_order_relaxed);
                    /* rr30 (T2b): clean-IDR resume FLUSHES the decoder first — the selfheal
                     * executor's deferred-reset recipe. Without it a non-IDR stop key
                     * (recovery-point SEI sets AV_PKT_FLAG_KEY too) leaves pre-skip refs in
                     * the DPB, and h264's frame_num-gap concealment then papers stale
                     * content into the skipped range with NO corrupt flag (the CORRUPT
                     * check never sees it) — smeared frames on air under clean labels.
                     * Flushed, everything before the recovery point is suppressed by the
                     * decoder itself. The deadline ESCAPE stays deliberately UNFLUSHED
                     * (Session-109 posture: a no-IDR source's established sync is the only
                     * one it will ever have). */
                    if (!escape)
                        avcodec_flush_buffers(d->vdec);
                    /* mapping publish — WRITE ORDER is the readers' consistency contract:
                     * off_before (old total) → from (new boundary) → off_total (new total,
                     * RELEASE: content_index() acquire-loads off_total first, so seeing the
                     * new total guarantees it sees the new boundary — rr30 (T2a), relaxed
                     * stores alone let the compiler/CPU tear the pair). In-flight frames
                     * older than the new boundary keep the old mapping; post-boundary
                     * frames (not yet decoded) pick up the new one. */
                    atomic_store_explicit(&g_vskip_off_before, old, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_from_us, boundary, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_off_total, old + achieved, memory_order_release);
                    atomic_fetch_add_explicit(&g_vskip_epoch, 1, memory_order_relaxed);
                    /* self-made-gap honesty + catch-up governor engagement (the QSHED stamps) */
                    d->shed_pkts += ndrop;
                    atomic_store_explicit(&g_shed_wall, nw, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g_shed_cnt, ndrop, memory_order_relaxed);
                    if (d->shed_wall) atomic_store_explicit(d->shed_wall, nw, memory_order_relaxed);
                    if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, ndrop, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_done_us, achieved, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_done_gops, gops, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_state,
                                          escape ? PTV_VSKIP_ESCAPE : PTV_VSKIP_DONE,
                                          memory_order_release);
                    av_log(NULL, escape ? AV_LOG_ERROR : AV_LOG_WARNING,
                           "[PTV-VSKIP] skipped %"PRId64"ms video content (%d GOPs, %d pkts) at the "
                           "video_q head%s — labels continue (offset %+"PRId64"ms from src %"PRId64"ms)\n",
                           achieved / 1000, gops, ndrop,
                           escape ? "; DEADLINE ESCAPE, resuming mid-GOP unflushed (decoder "
                                    "conceals until the next key)" : ", resuming at the stop IDR",
                           (old + achieved) / 1000, boundary / 1000);
                } else {
                    atomic_store_explicit(&g_vskip_done_us, 0, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_done_gops, 0, memory_order_relaxed);
                    atomic_store_explicit(&g_vskip_state, PTV_VSKIP_REFUSED, memory_order_release);
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-VSKIP] no skippable GOP inside the %"PRId64"s horizon (target "
                           "%"PRId64"ms) — refusing; the walk falls back to silence chunks\n",
                           g_resync_idr_wait_us / 1000000, tgt / 1000);
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
                PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_QRECV);     /* pre21: idle-wait keeps the stamp fresh */
                atomic_store_explicit(&g_hb_vdec_dec[d->hb_slot], d->dec_frames, memory_order_relaxed);
                ret = av_thread_message_queue_recv(d->video_q, &pkt, AV_THREAD_MESSAGE_NONBLOCK);
                if (ret != AVERROR(EAGAIN)) break;              /* got a pkt, or queue closed */
                if (g_selfheal && d->live && !d->hold &&
                    atomic_exchange_explicit(&g_selfheal_req, 0, memory_order_relaxed)) {
                    d->heal_dropkf = 1;
                    d->heal_arm_us = av_gettime_relative();
                    atomic_store_explicit(&g_shed_wall, d->heal_arm_us, memory_order_relaxed);
                    if (d->shed_wall) atomic_store_explicit(d->shed_wall, d->heal_arm_us, memory_order_relaxed);
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-SELFHEAL] re-prime (video_q empty): decoder resets at the "
                           "next IDR\n");
                }
                av_usleep(5000);
            }
            if (ret < 0) break;
        }
        /* 1.0.1-pre21 TEST-ONLY stall injection (PTV_TEST_VDEC_STALL_S): once, after 60s of
         * runtime (1.0.1-pre23: PTV_TEST_VDEC_STALL_AT_S overrides the trigger — AT 0 the
         * stall lands BEFORE the first decode, the startup-wedge shape the [PTV-NOVIDEO]
         * gate needs), park this thread for N seconds at the q_recv position — the gate
         * proves the [PTV-STALL] watchdog fires with the right position while the wire
         * stays alive (output dups), then the pipeline recovers. */
        if (g_test_vdec_stall_us > 0 && !hb_test_done &&
            av_gettime_relative() - hb_t0 >= g_test_vdec_stall_at_us) {
            hb_test_done = 1;
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-TEST] vdec stall injection: sleeping %"PRId64"s at q_recv (in%d)\n",
                   g_test_vdec_stall_us / 1000000, d->hb_slot);
            PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_QRECV);
            av_usleep((unsigned int)g_test_vdec_stall_us);
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
            if (d->shed_wall) atomic_store_explicit(d->shed_wall, d->heal_arm_us, memory_order_relaxed);
            if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, flushed, memory_order_relaxed);
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
                if (d->shed_cnt) atomic_fetch_add_explicit(d->shed_cnt, 1, memory_order_relaxed);
                continue;
            }
        }
        if (g_slow_dec) {   /* 1.0.1-pre8 stress knob: model a slow/contended NVDEC (windowed) */
            int64_t nws = av_gettime_relative();
            if (nws >= g_slow_dec_on_us && (!g_slow_dec_off_us || nws < g_slow_dec_off_us))
                av_usleep(g_slow_dec);
        }
        /* 1.0.1-pre30 #69 (item B): GOP-span estimator — key-to-key dts spacing EMA at the pop
         * site, published for the resync fire-side viability gate (vskip is only chosen when a
         * whole GOP fits inside R + tol and a key passed recently). Measurement only; gated on
         * g_resync so PTV_NO_RESYNC keeps the path inert. */
        if (g_resync && d->live && !d->hold && (pkt->flags & AV_PKT_FLAG_KEY)) {
            int64_t kdts = pkt->dts != AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
            if (kdts != AV_NOPTS_VALUE) {
                if (d->vgop_key_seen && kdts > d->vgop_last_key_dts) {
                    int64_t g = av_rescale_q(kdts - d->vgop_last_key_dts, d->ist_tb, AV_TIME_BASE_Q);
                    if (g > 0 && g < 30000000) {                 /* splice/wrap-guarded */
                        int64_t e = atomic_load_explicit(&g_vgop_est_us, memory_order_relaxed);
                        atomic_store_explicit(&g_vgop_est_us, e > 0 ? e + (g - e) / 4 : g,
                                              memory_order_relaxed);
                    }
                }
                d->vgop_last_key_dts = kdts;
                d->vgop_key_seen     = 1;
                atomic_store_explicit(&g_vgop_key_wall, av_gettime_relative(), memory_order_relaxed);
            }
        }
        PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_SENDPKT);
        ret = avcodec_send_packet(d->vdec, pkt);
        av_packet_free(&pkt);
        while (ret >= 0) {
            PTV_HB_VDEC(d->hb_slot, PTV_HB_VDEC_RECVFRAME);
            ret = avcodec_receive_frame(d->vdec, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) goto done;
            if (frame->flags & AV_FRAME_FLAG_CORRUPT) { d->vcorrupt++; av_frame_unref(frame); continue; }
            /* -cc_extract: EIA-608 tap. HERE — before the h0 anchor and before emit_video —
             * because cc_dec must see each frame's 608 pair exactly once in order, and this
             * is the last point upstream of the frame_q drop-oldest / dup / QSHED paths.
             * Always strips the A53 side data (so the video encoder cannot re-inject it). */
            if (d->cc.on || d->cc.strip) cc_tap_frame(&d->cc, frame, d->ist_tb);
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
                if (!d->hold) {                        /* single-input: DIAG t= line telemetry (input-0 global alias, pre16) */
                    atomic_store_explicit(&g_gov_gpps, gpps, memory_order_relaxed);
                    atomic_store_explicit(&g_gov_decl, d->in_pps_decl, memory_order_relaxed);
                }
                /* pre16: per-input publish, ALL inputs — the governor RAN on mv but said
                 * nothing (rr13 blindness); the mv DIAG per-slot segment reads these. */
                if (d->gov_gpps) atomic_store_explicit(d->gov_gpps, gpps, memory_order_relaxed);
                if (d->gov_decl) atomic_store_explicit(d->gov_decl, d->in_pps_decl, memory_order_relaxed);
                if (trusted && gsw && gnw - gsw < 600LL * 1000000 &&
                    av_thread_message_queue_nb_elems(d->video_q) > gpps) {
                    int64_t step = 800000 / gpps;       /* 4/5 input tick = 1.25x INPUT realtime */
                    if (!d->hold) atomic_store_explicit(&g_gov_on, 1, memory_order_relaxed);
                    if (d->gov_on) atomic_store_explicit(d->gov_on, 1, memory_order_relaxed);   /* pre16: per-input */
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
                    if (!d->hold) atomic_store_explicit(&g_gov_on, 0, memory_order_relaxed);
                    if (d->gov_on) atomic_store_explicit(d->gov_on, 0, memory_order_relaxed);   /* pre16: per-input */
                    gov_next_us = 0;                    /* incl. untrusted rate: fail open, never wedge */
                }
            }
            d->dec_frames++;
            PTV_HB_VDEC(d->hb_slot, d->hold ? PTV_HB_VDEC_FQSEND : PTV_HB_VDEC_HWUP);
            if (d->hold) stage_hold(d->hold, d->live, frame);   /* multiview: compositor samples this */
            else         emit_video(d, frame, filt);
        }
    }
    /* flush decoder */
    avcodec_send_packet(d->vdec, NULL);
    while (avcodec_receive_frame(d->vdec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        if (d->cc.on || d->cc.strip) cc_tap_frame(&d->cc, frame, d->ist_tb);
        d->dec_frames++;
        PTV_HB_VDEC(d->hb_slot, d->hold ? PTV_HB_VDEC_FQSEND : PTV_HB_VDEC_HWUP);
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
    if (d->cc.q)                                                     /* -cc_extract: drain, then EOF the emitter */
        av_thread_message_queue_set_err_recv(d->cc.q, AVERROR_EOF);
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
    /* 1.0.1-pre26 [PTV-MUXGUARD] survive-first backward-DTS backstop (PTV_NO_MUXGUARD=1
     * disables). Live crash class (NBS/CORELINK 2026-07-25, 8+ in 24h): a remedy-leaked
     * BACKWARD audio label reaches the muxer → lavf mux.c EINVALs the feed ("non
     * monotonically increasing dts") → every rung dies at once (single audio encode fans
     * out) → supervised respawn (and, before pre26, a wedged exit). A dropped packet is a
     * ~24ms content blip; a dead mux is a dead channel. So: mirror lavf's per-stream
     * monotonic check HERE (it fires at FEED time, compute_muxer_pkt_fields — so a
     * last-fed-DTS mirror is exactly aligned with what EINVALs) and DROP the offending
     * packet with a loud rate-limited diag + counter instead of feeding it to its death.
     * LABEL-ONLY: never re-stamps. The EINVAL fatal path below REMAINS as final backstop.
     * No interaction with reanch_mono (that guard drops FRAMES pre-encode; whatever it
     * drops never becomes a packet — this one catches only what actually leaks through). */
    int       mg_n = m->ofmt->nb_streams;
    int64_t  *mg_last = av_malloc_array(mg_n > 0 ? mg_n : 1, sizeof(*mg_last));
    int64_t   mg_dropped = 0, mg_warn_last = 0;
    int64_t   mt0 = av_gettime_relative();
    int       mg_test_injected = 0;
    /* 1.0.1-pre27 #62 (pre26 review rider): [PTV-MUXGUARD] drop-span ceiling — per-stream
     * wallclock of the first drop of the current drop run (0 = not dropping; reset when a
     * pkt on that stream is accepted by the guard). STRICT (audio/video) streams only: a
     * 60s drop span with no accept there means the label stream is dead, not leaking —
     * escalate to the fatal path instead of silently discarding that stream forever.
     * Sparse guarded streams (SCTE-35/teletext/DVB-sub) keep the pre26 drop-silently-
     * survive posture: two isolated drops minutes apart (e.g. the sparse-PID wrap-aliasing
     * class, where one forward-aliased accept ratchets mg_last hours ahead) must never
     * kill the channel. Governed by g_muxguard (no guard, no ceiling). */
    int64_t  *mg_dropsince = av_calloc(mg_n > 0 ? mg_n : 1, sizeof(*mg_dropsince));
    /* 1.0.1-pre27 #62 [PTV-MUXTOL] per-rung tolerated-write state (thread-local; one
     * mux_thread per rung): running count, 10s warn rate limit, and the wallclock of the
     * FIRST failure of the current failing run (0 = not failing; cleared by any
     * successful write) — the 60s dead-egress ceiling base. First-failure-based, not
     * last-success-based, so a >60s upstream stall followed by one transient catch-up
     * ENOMEM still gets the full 60s of tolerance (pre27 review finding 3). */
    int64_t   tol_count = 0, tol_warn_last = 0, tol_fail_since = 0;
    if (mg_last)
        for (int i = 0; i < mg_n; i++) mg_last[i] = AV_NOPTS_VALUE;

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
            int stream_index = pkt->stream_index;
            /* TEST ONLY (pre26 W1/W2 gates, PTV_RECANCHOR_TEST_ABORT_N precedent):
             * PTV_MUXTEST_BACK_AT_S=<t> re-stamps the first AUDIO packet ≥t s after mux
             * start backward by PTV_MUXTEST_BACK_MS (default 2795 ≈ the live −2794.7ms),
             * once per rung — the live shape: one fanned-out backward audio packet hits
             * every rung at the same content position. 0/unset = off, byte-identical.
             * 1.0.1-pre27 #62 GF gate: PTV_MUXTEST_BACK_HOLD_S=<dur> widens the injection
             * to EVERY audio pkt in [back_at, back_at+hold) — with a back_ms larger than
             * the hold, [PTV-MUXGUARD] then drops continuously for the whole window
             * (exercises the 60s drop-span ceiling below). */
            if (g_muxtest_back_at_us > 0 &&
                pkt->dts != AV_NOPTS_VALUE && stream_index < mg_n &&
                m->ofmt->streams[stream_index]->codecpar->codec_type == g_muxtest_back_type &&
                av_gettime_relative() - mt0 >= g_muxtest_back_at_us &&
                (g_muxtest_back_hold_us > 0
                     ? av_gettime_relative() - mt0 < g_muxtest_back_at_us + g_muxtest_back_hold_us
                     : !mg_test_injected)) {
                int64_t back = av_rescale_q(g_muxtest_back_ms * 1000, AV_TIME_BASE_Q,
                                            m->ofmt->streams[stream_index]->time_base);
                pkt->dts -= back;
                if (pkt->pts != AV_NOPTS_VALUE) pkt->pts -= back;
                if (!mg_test_injected)
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-MUXTEST] rung %d: injected backward dts (-%"PRId64"ms) on stream %d "
                           "(gate fixture, PTV_MUXTEST_BACK_AT_S)\n",
                           m->rung, g_muxtest_back_ms, stream_index);
                mg_test_injected = 1;
            }
            if (g_muxguard && mg_last && pkt->dts != AV_NOPTS_VALUE && stream_index < mg_n) {
                enum AVMediaType mgt = m->ofmt->streams[stream_index]->codecpar->codec_type;
                int     strict = mgt != AVMEDIA_TYPE_SUBTITLE && mgt != AVMEDIA_TYPE_DATA;
                int64_t last   = mg_last[stream_index];
                /* `last != 0` mirrors lavf's `sti->cur_dts &&` quirk: fire iff the muxer
                 * would EINVAL, never stricter. */
                if (last != AV_NOPTS_VALUE && last != 0 &&
                    (strict ? pkt->dts <= last : pkt->dts < last)) {
                    int64_t now = av_gettime_relative();
                    int64_t back_ms = av_rescale_q(last - pkt->dts,
                                                   m->ofmt->streams[stream_index]->time_base,
                                                   (AVRational){ 1, 1000 });
                    mg_dropped++;
                    /* 1.0.1-pre27 #62 rider: STRICT (audio/video) streams only — a 60s
                     * drop span with no accepted pkt = a dead label stream, not a
                     * transient leak — exit for a supervised respawn instead of
                     * discarding that stream forever (the motivating class is the
                     * fanned-out backward audio). Sparse streams (SCTE-35/teletext/
                     * DVB-sub) stay on the pre26 drop-silently-survive posture — two
                     * isolated drops >=60s apart there must never kill the channel. */
                    if (strict && mg_dropsince) {
                        if (!mg_dropsince[stream_index]) {
                            mg_dropsince[stream_index] = now;
                        } else if (now - mg_dropsince[stream_index] >= 60000000) {
                            av_log(NULL, AV_LOG_FATAL,
                                   "[PTV-MUX] rung %d stream %d: MUXGUARD drop span 60s with "
                                   "no accepted pkt on this A/V stream (%"PRId64" dropped on "
                                   "this rung) — label stream dead; exiting for supervised "
                                   "respawn\n", m->rung, stream_index, mg_dropped);
                            fflush(NULL);
                            _exit(1);
                        }
                    }
                    if (now - mg_warn_last >= 1000000) {
                        mg_warn_last = now;
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-MUXGUARD] rung %d stream %d: dropped pkt with backward dts "
                               "(-%"PRId64"ms vs last) — remedy leak, channel survives "
                               "(%"PRId64" dropped on this rung so far)\n",
                               m->rung, stream_index, back_ms, mg_dropped);
                    }
                    av_packet_free(&pkt);
                    continue;
                }
                mg_last[stream_index] = pkt->dts;
                if (mg_dropsince) mg_dropsince[stream_index] = 0;   /* accepted — reset the drop span */
            }
            g_muxed_bytes += pkt->size;
            if (g_muxfail_sim_err) {
                /* TEST ONLY (pre27 #62 gates): pretend the write failed inside the window —
                 * the pkt is NOT written (a failed write loses it), matching the real shape. */
                int64_t el = av_gettime_relative() - mt0;
                if (el >= g_muxfail_sim_from_us && el < g_muxfail_sim_to_us)
                    ret = g_muxfail_sim_err;
                else
                    ret = av_interleaved_write_frame(m->ofmt, pkt);
            } else
                ret = av_interleaved_write_frame(m->ofmt, pkt);
            if (g_diag) {
                int64_t dlt = av_gettime_relative() - wt0;
                if (dlt > 800000)
                    av_log(NULL, AV_LOG_WARNING, "[PTV-DIAG] write blocked %"PRId64" ms\n", dlt / 1000);
            }
            av_packet_free(&pkt);
            if (ret < 0 && g_muxtol &&
                (ret == AVERROR(ENOMEM) || ret == AVERROR(EAGAIN))) {
                /* 1.0.1-pre27 #62 [PTV-MUXTOL]: ENOMEM/EAGAIN = transient egress pressure
                 * (the paced-udp fifo is full — Praise_TV glo-2 respawn death loop
                 * 2026-07-24: every restart's catch-up burst overran the bitrate=-paced
                 * fifo_size egress ~6s after anchor, udp.c returned ENOMEM, the fatal path
                 * below killed the channel again). A full fifo drains; a dropped pkt is a
                 * blip. Tolerate: drop + count + rate-limited warn. But 60s since the
                 * FIRST failure of the current failing run with no successful write in
                 * between is a dead egress, not pressure — a dark channel needs the
                 * respawn anyway, so escalate rather than idle as a forever-throttled
                 * zombie. Everything else (EINVAL label corruption = the pre26 crash
                 * class, EPIPE, EIO, ...) stays immediately fatal below.
                 * PTV_NO_MUXTOL=1 reverts. */
                int64_t now = av_gettime_relative();
                tol_count++;
                if (!tol_fail_since)
                    tol_fail_since = now;
                else if (now - tol_fail_since >= 60000000) {
                    av_log(NULL, AV_LOG_FATAL,
                           "[PTV-MUX] rung %d egress dead 60s (tolerated %"PRId64" writes) — "
                           "exiting for supervised respawn\n", m->rung, tol_count);
                    fflush(NULL);
                    _exit(1);
                }
                if (now - tol_warn_last >= 10000000) {
                    tol_warn_last = now;
                    av_log(NULL, AV_LOG_WARNING,
                           "[PTV-MUXTOL] rung %d stream %d: %s from mux write — pkt dropped, "
                           "channel survives (%"PRId64" tolerated on this rung so far)\n",
                           m->rung, stream_index,
                           ret == AVERROR(ENOMEM) ? "ENOMEM" : "EAGAIN", tol_count);
                }
                continue;
            }
            if (ret < 0) {
                /* 1.0.1-pre23 #54: a mux write error used to break out of this thread with
                 * ZERO log lines — the wire went dead while the process lived on as a silent
                 * zombie (and, #60: the closed delivery gate then freed every audio packet
                 * into a dead mux_q, removing ALL backpressure from the audio thread — the
                 * sustained-allocation enabler). A dead mux IS a dead channel either way:
                 * die loud so supervisord respawns a working one. No env gate — this is
                 * defense-in-depth, not behavior anyone can want to keep.
                 * 1.0.1-pre26: _exit(1), NOT exit(1). exit() runs atexit/cleanup handlers
                 * from THIS (non-main) thread while udp-rx/CUDA/audio threads run and hold
                 * locks — live-captured wedge (NBS cor-3 / CORELINK glo-2 2026-07-25): the
                 * exiting thread parked in futex_do_wait inside exit cleanup, process
                 * zombied with a dead wire until sync_check bounced it minutes later.
                 * _exit() skips all handlers (fflush(NULL) first so the FATAL line lands). */
                m->err = ret;
                av_log(NULL, AV_LOG_FATAL,
                       "[PTV-MUX] rung %d write failed on stream %d: %s — a dead mux is a dead "
                       "channel; exiting for supervised respawn\n",
                       m->rung, stream_index, av_err2str(ret));
                fflush(NULL);
                _exit(1);
            }
        }
        /* pre14 (§3, owner call 3): per-rung wire-send watermark — stamped ONLY after a
         * SUCCESSFUL interleaved write, so a stalled/backed-up muxer (the Newsmax2 dead
         * rung, invisible to every label-domain signal) goes stale within seconds. The
         * corrector's delivery-liveness gate treats this as the primary signal. */
        tol_fail_since = 0;                      /* pre27 #62: success ends the failing run */
        atomic_store_explicit(&g_mux_sent_wc[m->rung], av_gettime_relative(), memory_order_relaxed);
        g_muxed++;
    }
    if (mg_dropped)
        av_log(NULL, AV_LOG_WARNING, "[PTV-MUXGUARD] rung %d: %"PRId64" backward-dts pkts "
               "dropped in total this run\n", m->rung, mg_dropped);
    if (tol_count)
        av_log(NULL, AV_LOG_WARNING, "[PTV-MUXTOL] rung %d: %"PRId64" egress-pressure write "
               "errors tolerated in total this run\n", m->rung, tol_count);
    av_freep(&mg_last);
    av_freep(&mg_dropsince);
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

/* Build the whole -cc_extract path in one place: cc_dec on the video input's decode thread,
 * ONE dvb_teletext encoder, one synthetic subtitle stream per rung, and the decode->emitter
 * queue. (The legacy fftools implementation had to split this into a "mark the video stream
 * now, resolve it later" two-phase dance across translation units to satisfy the old CLI's
 * scheduler graph; ptvencoder resolves its whole plan in transcode(), so one function does.)
 *
 * CALL ORDERING IS LOAD-BEARING, from both sides:
 *  - AFTER the decoders exist and after the copy-passthrough streams are created: cc_dec's
 *    ASS subtitle_header is the encoder's mandatory init input, and the copied DVB-sub
 *    streams must already have taken their -metadata:s:s:N indices;
 *  - BEFORE avformat_write_header: mpegtsenc builds the PMT teletext descriptor (0x56) out
 *    of codecpar->extradata, and the encoder only produces those 2 bytes at avcodec_open2.
 *
 * `vin` is the AVFormatContext of THE INPUT THIS EXTRACTION BELONGS TO — passed in
 * explicitly rather than assumed, because the legacy code's bug was exactly this: it walked
 * the filtergraph and then unconditionally took the first video stream of the first input
 * file, so on a mosaic it always bound input 0 (which in the reported incident had no
 * captions at all). Multiview now calls this ONCE PER SLOT, each with its own input, its own
 * cc_dec/encoder/queue/thread and its own subtitle track — no slot is ever guessed.
 *
 * `slot` is the input index (0 for single-input), `sidx` the LOGICAL subtitle index this
 * track takes for -metadata:s:s:N, and `multi` marks multiview for the log lines. */
static int cc_setup(CcCtx *cc, CcTap *tap, AVThreadMessageQueue **cc_q,
                    const AVCodec *cdec, const AVCodec *cenc,
                    AVFormatContext *vin, Rung *rung, int n_rung, OptionGroupList *outs,
                    int slot, int sidx, int multi)
{
    /* the cc_* options configure the extraction for the WHOLE ladder, so they are read from
     * the first output group (like -cc_extract itself). On multiview they apply to every
     * slot: -cc_lang pins the language for all of them, and the page walks per slot below. */
    const char *cclang = og_get(&outs->groups[0], "cc_lang");
    const char *ccpage = og_get(&outs->groups[0], "cc_page");
    const char *ccmag  = og_get(&outs->groups[0], "cc_magazine");
    AVCodecContext *ce;
    int subset, page, magazine, r, k, ret;

    /* --- cc_dec, on the video input's decode thread (see the CcTap header) --- */
    tap->dec = avcodec_alloc_context3(cdec);
    if (!tap->dec)
        return AVERROR(ENOMEM);
    tap->dec->pkt_timebase = AV_TIME_BASE_Q;      /* the tap stamps its packets in us */
    /* real_time: emit on every CC buffer change instead of waiting for a mode switch — a
     * live channel has no end of stream to flush against. */
    av_opt_set_int(tap->dec, "real_time", 1, AV_OPT_SEARCH_CHILDREN);
    /* FIELD 1 explicitly. cc_dec's data_field defaults to -1 = "pick the first field that
     * appears" and then IGNORES the other one for the rest of the run. Field 2 carries
     * secondary services (CC3/CC4) and XDS, so on a source whose first A53 pair happens to be
     * field 2 the extraction locks onto XDS and emits two-character fragments that accumulate
     * into garbage — measured on Newsmax2: "HDPAHDPAHDHDPAHD..." on air while stock ffmpeg
     * read real captions off the same source. CC1 (field 1) is the primary caption service. */
    av_opt_set_int(tap->dec, "data_field", 0, AV_OPT_SEARCH_CHILDREN);
    if ((ret = avcodec_open2(tap->dec, cdec, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-CC] open cc_dec: %s\n", av_err2str(ret));
        return ret;
    }
    tap->last_src_us = AV_NOPTS_VALUE;
    tap->last_evt_us = AV_NOPTS_VALUE;
    tap->shown_since_us = AV_NOPTS_VALUE;
    tap->slot        = slot;
    tap->multi       = multi;

    /* --- language: -cc_lang, else the first non-"und" audio language on the video's own
     * input, else English. Drives both the stream metadata (which mpegtsenc writes into the
     * 0x56 descriptor) and the G0 national option subset. --- */
    if (!cclang || !cclang[0]) {
        for (k = 0; k < (int)vin->nb_streams; k++) {
            AVStream *ist = vin->streams[k];
            AVDictionaryEntry *le;
            if (ist->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
            le = av_dict_get(ist->metadata, "language", NULL, 0);
            if (le && le->value[0] && strcmp(le->value, "und")) { cclang = le->value; break; }
        }
    }
    if (!cclang || !cclang[0]) cclang = "eng";
    subset = lang_to_g0_subset(cclang);

    /* --- the dvb_teletext encoder. magazine/page are operator-settable (base-0 parse, so
     * both 0x88 and 136 work); the defaults are the encoder's own 8 / 0x88 = page 888. --- */
    magazine = ccmag  ? (int)strtol(ccmag,  NULL, 0) : 8;
    page     = ccpage ? (int)strtol(ccpage, NULL, 0) : 0x88;
    if (magazine < 1 || magazine > 8) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-CC] -cc_magazine %d out of range (1-8)\n", magazine);
        return AVERROR(EINVAL);
    }
    if (page < 0 || page > 0xFF) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-CC] -cc_page 0x%X out of range (0x00-0xFF)\n", page);
        return AVERROR(EINVAL);
    }
    /* Multiview: each slot gets its OWN page, walked in BCD from the base (888, 889, 890,
     * 891) so a receiver browsing teletext sees four distinct pages rather than four things
     * all claiming 888. Separate PIDs would make identical pages legal, but distinct ones
     * cost nothing and are unambiguous everywhere. Nibbles above 9 are skipped — 88A is not
     * a page a normal receiver will show. */
    for (k = 0; k < slot; k++) {
        page = (page & 0x0F) < 9 ? page + 1 : (page & 0xF0) + 0x10;
        /* both nibbles must stay decimal: 0x99 -> 0xA0 is "8A0", not a page any receiver
         * keypad can reach, and > 0xFF alone would let it through silently */
        if (page > 0x99 || (page & 0x0F) > 9) {
            av_log(NULL, AV_LOG_ERROR,
                   "[PTV-CC] -cc_page base 0x%X leaves no room for %d slots\n",
                   ccpage ? (int)strtol(ccpage, NULL, 0) : 0x88, slot + 1);
            return AVERROR(EINVAL);
        }
    }
    ce = avcodec_alloc_context3(cenc);
    if (!ce)
        return AVERROR(ENOMEM);
    cc->enc = ce;
    ce->time_base = AV_TIME_BASE_Q;               /* the emitter stamps in us */
    av_opt_set_int(ce->priv_data, "magazine",  magazine, 0);
    av_opt_set_int(ce->priv_data, "page",      page,     0);
    av_opt_set_int(ce->priv_data, "g0_subset", subset,   0);
    /* MANDATORY: the encoder inits its ASS splitter from subtitle_header and
     * ff_ass_split(NULL) fails — so hand it cc_dec's header, NUL-terminated (the splitter
     * parses it as a C string). */
    if (tap->dec->subtitle_header_size > 0) {
        int hs = tap->dec->subtitle_header_size;
        ce->subtitle_header = av_mallocz(hs + 1);
        if (!ce->subtitle_header)
            return AVERROR(ENOMEM);
        memcpy(ce->subtitle_header, tap->dec->subtitle_header, hs);
        ce->subtitle_header_size = hs;
    }
    if ((ret = avcodec_open2(ce, cenc, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "[PTV-CC] open dvb_teletext: %s\n", av_err2str(ret));
        return ret;
    }

    /* --- one synthetic subtitle stream per rung --- */
    for (r = 0; r < n_rung; r++) {
        AVStream *os = avformat_new_stream(rung[r].ofmt, NULL);
        if (!os)
            return AVERROR(ENOMEM);
        if ((ret = avcodec_parameters_from_context(os->codecpar, ce)) < 0)
            return ret;
        os->time_base = AV_TIME_BASE_Q;
        av_dict_set(&os->metadata, "language", cclang, 0);   /* the 0x56 descriptor reads this */
        /* -metadata:s:s:N lands on THIS track (index assignment is at the copy fan),
         * which is how the multiview convention (language=mva/mvb/…, title="View N") is
         * applied — and it deliberately runs AFTER the source-derived language above, so the
         * operator's virtual code wins on the wire while the G0 subset stays derived from the
         * REAL language (a slot tagged "mva" must still render its Spanish correctly). */
        apply_stream_meta(&outs->groups[r], 's', sidx, os);
        cc->ost[r] = os;
    }

    if ((ret = av_thread_message_queue_alloc(cc_q, PTV_CC_QDEPTH, sizeof(CcEvent))) < 0)
        return ret;
    av_thread_message_queue_set_free_func(*cc_q, free_cc_msg);
    cc->last_dts        = AV_NOPTS_VALUE;
    cc->last_caption_us = AV_NOPTS_VALUE;
    cc->slot            = slot;
    cc->multi           = multi;

    /* Two DIFFERENT things, and conflating them is the trap: the source language chooses the
     * teletext G0 national character set, while the tag a player lists is whatever ends up in
     * the stream metadata — which -metadata:s:s:N overrides (multiview sets mva/mvb/mvc/mvd).
     * Keeping them separate is what lets a slot be listed as "mvc" and still render its
     * Spanish accents; deriving the subset from the virtual code would silently give it the
     * English repertoire. Say both, so the log cannot be read as "the output is tagged eng". */
    if (multi)
        av_log(NULL, AV_LOG_INFO,
               "ptvencoder: CC extraction ON for input %d — EIA-608 -> dvb_teletext page "
               "%d%02X, subtitle s:%d in every output; source lang \"%s\" -> G0 subset %d "
               "(character set only — the tag on the wire is s:%d's metadata, e.g. "
               "-metadata:s:s:%d language=mva)\n",
               slot, magazine, page, sidx, cclang, subset, sidx, sidx);
    else
        av_log(NULL, AV_LOG_INFO,
               "ptvencoder: CC extraction ON — EIA-608 -> dvb_teletext page %d%02X, "
               "1 subtitle stream per output; source lang \"%s\" -> G0 subset %d, tagged "
               "\"%s\" unless -metadata:s:s:0 language=... overrides it\n",
               magazine, page, cclang, subset, cclang);
    return 0;
}

/* open one input on its own thread (parallel open: a dead/slow slot must not
 * delay the others, and serial open would block on its long rw_timeout). */
typedef struct OpenArg { Input *in; AVDictionary **opts; } OpenArg;
static void *open_input_thread(void *arg)
{
    OpenArg *o = arg; Input *in = o->in;
    in->open_ret = avformat_open_input(&in->ifmt, in->url, NULL, o->opts);
    if (in->open_ret >= 0) in->open_ret = ptv_find_stream_info(in->ifmt);   /* pre19.1: tolerant AUDIO probe */
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
    {   /* pre20 fix round (F5): -t placed BEFORE -i lands in the INPUT group and is not
         * consumed (ptvencoder's -t is an OUTPUT option) — was a truly silent ignore, and
         * ffmpeg users expect input-side -t to work. One loud line per offending input. */
        int gi5;
        for (gi5 = 0; gi5 < ins->nb_groups; gi5++)
            if (og_get(&ins->groups[gi5], "t"))
                av_log(NULL, AV_LOG_WARNING,
                       "ptvencoder: -t before -i is ignored (-t is an OUTPUT option — place it "
                       "after -i with the output options)\n");
    }
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
    int eowner[PTV_MAX_RUNG];                  /* audio encoder sharing: owning rung per rung */
    int ret = 0, live, net_input, have_audio = 0, hw_cuda = 0;
    int aborted = 0, r, si, k, kk, n_copy_inputs = 0;
    AVRational out_fps;
    PassStream pass[PTV_MAX_PASS]; int n_pass = 0;
    /* -cc_extract: EIA-608 -> DVB-teletext (all inert unless the option is given). One
     * INDEPENDENT extraction per participating input — single-input is simply n_cc == 1. */
    CcCtx            cc[PTV_MAX_INPUT];
    AVThreadMessageQueue *cc_q[PTV_MAX_INPUT] = {0};
    const AVCodec   *cc_dec_codec = NULL, *cc_enc_codec = NULL;
    pthread_t        th_cc[PTV_MAX_INPUT];
    int              cc_slot[PTV_MAX_INPUT];      /* input index of each extraction */
    int              started_cc[PTV_MAX_INPUT] = {0};
    int              cc_want[PTV_MAX_INPUT] = {0};/* per-input: extract from this slot? */
    int              cc_on = 0, n_cc = 0;
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
    /* -cc_extract resolution, up front: it can REFUSE the whole run (multiview), and nothing
     * is allocated yet. Codec availability is settled here too so that the copy fan below
     * can seed its subtitle index knowing whether a CC stream will exist. */
    {
        const char *ccv = og_get(&outs->groups[0], "cc_extract");
        int gi;
        cc_on = g_cc && ccv && atoi(ccv);
        /* The whole ladder shares ONE extraction (one cc_dec, one encoder, fanned out to
         * every rung), so the cc_* options are read from output group 0 only. Say so rather
         * than letting a rung-1 -cc_extract look like it did something. */
        for (gi = 1; gi < outs->nb_groups; gi++) {
            if (og_get(&outs->groups[gi], "cc_extract")   ||
                og_get(&outs->groups[gi], "cc_slots")     ||
                og_get(&outs->groups[gi], "cc_lang")      ||
                og_get(&outs->groups[gi], "cc_page")      ||
                og_get(&outs->groups[gi], "cc_magazine")) {
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-CC] -cc_extract/-cc_slots/-cc_lang/-cc_page/-cc_magazine on output %d are "
                       "IGNORED — the extraction is shared by the whole ladder and is "
                       "configured on the first output only%s\n", gi,
                       cc_on ? "" : " (and the first output does not enable it, so no CC "
                                    "stream will exist at all)");
            }
        }
        /* WHICH inputs to extract from. Single-input: the one input. Multiview: every slot
         * by default, or the explicit list in -cc_slots. The list exists because the caller
         * usually knows more than we can: a slot that carries real DVB subtitles wants those
         * copied, not a second, redundant caption track synthesised beside them (that is
         * exactly the per-slot dvb/cc/none choice the production wrapper already makes).
         * There is no guessing left — a slot is either named or it is not extracted. */
        if (!cc_on && og_get(&outs->groups[0], "cc_slots"))
            av_log(NULL, AV_LOG_WARNING,
                   "[PTV-CC] -cc_slots is ignored without -cc_extract\n");
    {   /* -af / -filter:a are resolved PER AUDIO TRACK from output group 0 only (one filter
         * graph per AudioState — see the af lookup at the audio setup), so a DIFFERENT chain
         * on a later output is silently ignored: that rung gets group 0's audio. Identical
         * copies are what the production wrapper emits and are harmless, so say nothing for
         * those; warn only on a real divergence, which is otherwise invisible in the output. */
        const char *af0 = og_get(&outs->groups[0], "af");
        int gi;
        if (!af0) af0 = og_get(&outs->groups[0], "filter:a");
        for (gi = 1; gi < outs->nb_groups; gi++) {
            const char *afn = og_get(&outs->groups[gi], "af");
            if (!afn) afn = og_get(&outs->groups[gi], "filter:a");
            if (afn && (!af0 || strcmp(afn, af0)))
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-AF] -af/-filter:a on output %d DIFFERS from output 0 and is IGNORED "
                       "— the audio filter graph is built once per TRACK from output 0, so this "
                       "output receives output 0's audio. Use -filter:a:N for a per-track chain; "
                       "per-RUNG audio filtering is not supported.\n", gi);
        }
    }
        if (cc_on) {
            const char *ccsl = og_get(&outs->groups[0], "cc_slots");
            if (ccsl && ccsl[0]) {
                const char *s = ccsl;
                while (*s) {
                    char *end;
                    long v = strtol(s, &end, 10);
                    if (end == s) {
                        av_log(NULL, AV_LOG_ERROR,
                               "[PTV-CC] -cc_slots '%s': expected a comma-separated list of "
                               "input indices (e.g. 0,2)\n", ccsl);
                        return AVERROR(EINVAL);
                    }
                    if (v < 0 || v >= n_input) {
                        av_log(NULL, AV_LOG_ERROR,
                               "[PTV-CC] -cc_slots names input %ld but this run has %d input(s)\n",
                               v, n_input);
                        return AVERROR(EINVAL);
                    }
                    cc_want[v] = 1;
                    s = end;
                    while (*s == ',' || *s == ' ') s++;
                }
            } else
                for (k = 0; k < n_input; k++) cc_want[k] = 1;
            for (k = 0; k < n_input; k++) if (cc_want[k]) n_cc++;
            if (!n_cc) {
                av_log(NULL, AV_LOG_ERROR, "[PTV-CC] -cc_slots selected no input\n");
                return AVERROR(EINVAL);
            }
        }
        if (cc_on) {
            cc_dec_codec = avcodec_find_decoder(AV_CODEC_ID_EIA_608);
            cc_enc_codec = avcodec_find_encoder(AV_CODEC_ID_DVB_TELETEXT);
            if (!cc_dec_codec || !cc_enc_codec) {
                av_log(NULL, AV_LOG_ERROR,
                       "[PTV-CC] -cc_extract requested but %s is missing from this build\n",
                       !cc_dec_codec ? "the EIA-608 decoder (cc_dec)"
                                     : "the dvb_teletext encoder");
                return AVERROR_ENCODER_NOT_FOUND;
            }
        }
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
    memset(cc, 0, sizeof cc);
    for (k = 0; k < PTV_MAX_INPUT; k++) { cc[k].last_dts = AV_NOPTS_VALUE; cc_slot[k] = 0; }
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
            /* pre17 R1: keep a pristine copy of the open options — avformat_open_input
             * consumes recognized entries, and the mv demux reopen-retry must reopen with
             * the ORIGINAL set (rw_timeout, fifo, overrun_nonfatal, ...). */
            av_dict_copy(&inputs[k].da.reopen_opts, ins->groups[k].format_opts, 0);
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
        /* Decode threading is left to libavcodec, which resolves to ONE thread for our
         * streams (measured: "threads=1 type=none"). That is what we want on a box running
         * 40+ channels — each needs 1x realtime, not minimum latency — and it matches the
         * long-standing intent here, which was never actually enforced in code. The log below
         * makes it visible, so a libavcodec default change that starts spawning a pool per
         * channel cannot slip in unnoticed. */
        if ((ret = avcodec_open2(inputs[k].vdec, vd, NULL)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "open video decoder (input %d): %s\n", k, av_err2str(ret)); goto end;
        }
        /* what libavcodec ACTUALLY resolved — auto-detect is invisible otherwise, and the
         * thread count per channel is the whole point of the knob */
        av_log(NULL, AV_LOG_INFO,
               "ptvencoder: input %d video decoder %s — threads=%d type=%s%s\n", k, vd->name,
               inputs[k].vdec->thread_count,
               inputs[k].vdec->active_thread_type & FF_THREAD_FRAME ? "frame" :
               inputs[k].vdec->active_thread_type & FF_THREAD_SLICE ? "slice" : "none", "");
        vdecs[k] = inputs[k].vdec;
        inputs[k].wrap_off  = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_off));
        inputs[k].wrap_last = av_malloc_array(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_last));
        inputs[k].wrap_wall_last = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_wall_last)); /* 0 = no prev packet yet */
        inputs[k].edit_us   = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].edit_us));   /* pre9 sensor label-edit ledger */
        inputs[k].gap_vsnap = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].gap_vsnap)); /* pre16 #47-A: vpkt snapshots */
        inputs[k].wall_cad_us     = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wall_cad_us));     /* pre24 #63: cadence EMA */
        inputs[k].pkt_wall_gap_us = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].pkt_wall_gap_us)); /* pre24 #63: current-pkt gap */
        if (!inputs[k].wrap_off || !inputs[k].wrap_last || !inputs[k].wrap_wall_last || !inputs[k].edit_us || !inputs[k].gap_vsnap ||
            !inputs[k].wall_cad_us || !inputs[k].pkt_wall_gap_us) { ret = AVERROR(ENOMEM); goto end; }
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
        const char *t_str = og_get(g, "t");   /* 1.0.1-pre20 rider (b): -t now CONSUMED (house-clock
                                               * output media time; was parsed-and-ignored). Any rung's
                                               * -t arms the shared bound; the largest wins. */
        if (t_str) {
            int64_t tus = 0;
            if (av_parse_time(&tus, t_str, 1) >= 0 && tus > 0) {
                if (tus > g_t_us) g_t_us = tus;
                av_log(NULL, AV_LOG_INFO, "ptvencoder: -t %s -> stop pulling input at %.1fs of output media time (flush + clean exit)\n",
                       t_str, g_t_us / 1e6);
            } else
                av_log(NULL, AV_LOG_ERROR, "ptvencoder: invalid -t '%s' (ignored)\n", t_str);
        }
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
        ptv_adec_opts(kdec);               /* #38: tolerant AAC decode (by name, optional) */
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
            /* SHARE an encoder with an earlier rung whose settings are identical. Within one
             * audio track every other input to the encoder is fixed (48k, ochl, sample_fmt,
             * the same codec), so the key is exactly (codec, bitrate) — a 6-rung ladder over
             * 2 distinct bitrates opens 2 encoders, not 6, and the bytes are unchanged.
             * ⚠ IF A PER-RUNG AUDIO OPTION IS EVER ADDED (profile, cutoff, afterburner…) IT
             * MUST JOIN THIS KEY, or rungs would silently receive another rung's stream. */
            eowner[r] = r;
            if (g_aenc_share) {
                for (si = 0; si < r; si++) {
                    const char *sbr = (k < sel[si].n_aout) ? sel[si].aout[k].abr : NULL;
                    if (eowner[si] == si && encs[si] && encs[si]->codec == aenc &&
                        ((!sbr && !abr) || (sbr && abr && !strcmp(sbr, abr)))) {
                        eowner[r] = si; break;
                    }
                }
                if (eowner[r] != r) { encs[r] = NULL; continue; }
            }
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
        a->dec_ts_carry = AV_NOPTS_VALUE;  /* pre19.1 [PTV-ASTAMP]: no extrapolation base yet (0 would be a valid pts) */
        for (r = 0; r < n_rung; r++) {
            AVStream *aos; AVDictionaryEntry *klang;
            a->enc[r]       = encs[eowner[r]];        /* aliased when shared */
            a->enc_owner[r] = eowner[r];
            aos = avformat_new_stream(rung[r].ofmt, NULL);
            if (!aos) { ret = AVERROR(ENOMEM); goto end; }
            avcodec_parameters_from_context(aos->codecpar, a->enc[r]);
            aos->time_base = a->enc[r]->time_base;
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
        /* pre17 R1: OWNED copies — the demux reopen closes the old AVFormatContext, so an
         * AVStream* would dangle by the time the ADECWD watchdog reopen fires. */
        a->ist_par    = avcodec_parameters_alloc();
        if (a->ist_par) avcodec_parameters_copy(a->ist_par, kist->codecpar);
        a->ist_pkt_tb = kist->time_base;
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
    /* cc_on was resolved at entry; the extracted CC stream owns the LOGICAL subtitle index
     * s:0 (its -metadata:s:s:0 / -disposition:s:0), so copied DVB-sub streams start at 1
     * instead of colliding with it. */
    /* Extracted CC tracks own the LOGICAL subtitle indices s:0 .. s:(n_cc-1) — one per
     * participating input, in input order — so copied DVB-subs start after them. NOTE the
     * PHYSICAL muxer order differs: the CC streams are created below, after this fan, because
     * their extradata must exist before write_header. So ffprobe lists the copied subs first
     * while -metadata:s:s:N / -c:s:N count the CC tracks first. The N used by the wrapper is
     * the logical one. */
    int copy_vidx = 1, copy_aidx = n_audio, copy_sidx = n_cc, copy_didx = 0;
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

    /* -cc_extract: build the EIA-608 -> DVB-teletext path (ordering rationale in cc_setup),
     * ONE INDEPENDENT EXTRACTION PER PARTICIPATING INPUT — its own cc_dec on that input's own
     * decode thread, its own encoder, its own queue and thread, its own subtitle track. Each
     * binding is explicit; nothing is guessed. Single-input is just n_cc == 1 on input 0.
     * A hard failure here is fatal: the operator asked for captions. */
    if (cc_on) {
        int ci = 0;
        for (kk = 0; kk < n_input; kk++) {
            if (!cc_want[kk]) continue;
            cc_slot[ci] = kk;
            ret = cc_setup(&cc[ci], &inputs[kk].dc.cc, &cc_q[ci], cc_dec_codec, cc_enc_codec,
                           inputs[kk].ifmt, rung, n_rung, outs, kk, ci, multiview);
            if (ret < 0) goto end;
            ci++;
        }
        g_cc_on = 1;
    }

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
         * delivered high-water. Multiview too since task #48 (2026-07-19): the 2026-07-13
         * fleet loudnorm rollout (~3s one-pass analysis fill) made EVERY mv audio track
         * wall-late, so the §7.5a gate reads dlvhold=0 (nothing early to hold) and the grids'
         * wire carried video ~2-3s ahead of audio with labels intact — the exact single-input
         * pre12 class, and mv had no closing mechanism. Slots share ONE gate per rung;
         * 1.0.1-pre18 (g_perstream_wm): the hold keys on the SLOWEST currently-LIVE stream's
         * per-stream delivered watermark (dlv_a_hi_key) instead of the least-delayed
         * aggregate — a mixed rung (loudnorm AAC + AC-3 copy) aligns to its slowest track,
         * and a stream silent >2s is excluded so a single dead slot/track still cannot wedge
         * video (the 6s audio-death escape keeps needing ALL tracks silent). Armed only when the
         * run HAS gated audio to key on (a transcoded track or dense copied AC-3/MP2): a
         * no-audio channel must not pay the audio-death escape timeout at birth.
         * PTV_NO_VDELIVERY=1 reverts. */
        if (g_vdelivery) {
            int have_gated_audio = n_audio > 0, gp;
            for (gp = 0; gp < n_pass && !have_gated_audio; gp++)
                if (pass[gp].gated) have_gated_audio = 1;
            if (!have_gated_audio)
                av_log(NULL, AV_LOG_INFO,
                       "[PTV-VDLV] no gated audio on this channel — early-video hold disabled\n");
            else
                for (r = 0; r < n_rung; r++) {
                    dlv_video_cfg(&rung[r].gate, PTV_VDLV_BAND_US, g_cp.vdlv_cap_us, g_cp.vdlv_maxq);
                    /* mv: flow at birth (audio anchors at first DISPLAY, so the §7.5a FIFO
                     * is legitimately empty at the first video packet — see ptvencoder.h) */
                    rung[r].gate.v_birth_flow = multiview;
                }
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
        d->hb_slot = k;                            /* 1.0.1-pre21: heartbeat slot key */
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
        d->gov_gpps  = &inputs[k].gov_gpps;   /* pre16: per-input governor telemetry (mv DIAG + */
        d->gov_decl  = &inputs[k].gov_decl;   /* corrector feed; globals stay the input-0 alias) */
        d->gov_on    = &inputs[k].gov_on;
        d->shed_wall = &inputs[k].shed_wall;  /* pre16: per-input self-shed stamp (row 14) */
        d->shed_cnt  = &inputs[k].shed_cnt;
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
        rung[r].ma.n_producers = 1 + n_audio + n_copy_inputs + n_cc;   /* video out + N audio + per-input copy fan + one CC emitter per extraction */
    }
    for (k = 0; k < n_cc; k++) {                 /* one emitter per extraction */
        int in = cc_slot[k];
        /* The stamping domain. Every rung shares out_tb/out_fps/tick_dur_us (one cadence for
         * the whole ladder), so those come from rung 0 either way; only h0 is per-slot. On
         * single-input that is input 0 = exactly what the rungs use, so the result is
         * unchanged. On multiview it is THIS SLOT's h0 — the same anchor its audio rides
         * (as[].h0), which is what puts a slot's captions on its own dialogue rather than on
         * whichever input happens to lead the mosaic. */
        cc[k].clk            = rung[0].vc;       /* copy the cadence fields... */
        cc[k].clk.h0         = &inputs[in].h0;   /* ...then re-point the anchor at this slot */
        cc[k].clk.h0_lock    = &inputs[in].h0_lock;
        cc[k].vc             = &cc[k].clk;
        cc[k].q              = cc_q[k];
        cc[k].n_out          = n_rung;
        for (r = 0; r < n_rung; r++) cc[k].mux_q[r] = rung[r].mux_q;
        inputs[in].dc.cc.q  = cc_q[k];           /* decode-thread tap -> this slot's emitter */
        inputs[in].dc.cc.on = 1;
    }
    /* Strip A53 side data on EVERY input, including ones -cc_slots excluded. xstack/hstack
     * allocate a fresh frame and drop side data, but overlay-type filters forward the main
     * frame intact — so an excluded slot's raw 608 could otherwise reach h264_nvenc (a53cc
     * defaults ON) and be re-injected as SEI on the composite, source-timeline-stamped,
     * alongside the teletext tracks. An unarmed tap has no decoder and does nothing else. */
    if (cc_on)
        for (k = 0; k < n_input; k++)
            if (!inputs[k].dc.cc.on) inputs[k].dc.cc.strip = 1;
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
        as[k].shed_wall = &inputs[asrc_in[k]].shed_wall;   /* pre16: per-input self-shed window (AGLUE notes + */
        as[k].shed_cnt  = &inputs[asrc_in[k]].shed_cnt;    /* corrector quiet) — slot B never smears slot A */
        as[k].gov_on    = &inputs[asrc_in[k]].gov_on;      /* pre16: this track's OWN input's governor flag */
        as[k].v_arrive_wc = &inputs[asrc_in[k]].v_arrive_wc;  /* pre17: §3 per-input liveness — the corrector
                                                               * reads THIS input's arrival watermark */
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
        d->gap_vsnap = inputs[kk].gap_vsnap;            /* pre16 #47-A: per-stream vpkt snapshots */
        d->wall_cad_us     = inputs[kk].wall_cad_us;     /* pre24 #63: delivery-cadence EMA */
        d->pkt_wall_gap_us = inputs[kk].pkt_wall_gap_us; /* pre24 #63: current-pkt wall gap */
        d->rsync_slot = kk;                             /* pre16: EVERY input publishes its ledgers to g_rsx,
                                                         * video keyed by this slot (was single-input-only) */
        d->shed_wall = &inputs[kk].shed_wall;           /* pre16: per-input self-shed stamp */
        d->shed_cnt  = &inputs[kk].shed_cnt;
        d->v_arrive_wc = &inputs[kk].v_arrive_wc;       /* pre17: per-input arrival watermark (corrector §3) */
        /* pre17 R1: mv live net inputs reopen-retry on read error (single-input keeps
         * EOF = channel end, supervisor-owned; file inputs keep EOF = media end). */
        d->url       = inputs[kk].url;
        d->reopen    = live && multiview && is_net_url(inputs[kk].url);
        d->ifmt_home = &inputs[kk].ifmt;
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

    /* pre9 sensor track count / pre16 mv port: wired for BOTH modes now — the compositor
     * publishes per-slot video lineage, so mv lipsync= is real per-(slot,track) data, not
     * garbage. a_in[] is the track→slot map the shared stats builder keys the video term by
     * (identically 0 on single input); plain ints, set before any thread spawns. */
    g_rsx.n_a  = n_audio;
    for (k = 0; k < n_audio; k++)
        g_rsx.a_in[k] = asrc_in[k];
    if (multiview && g_rsync_sense && n_audio > 0) {
        /* startup track→slot map (mv only — trivial on single input, whose log stays
         * byte-identical to pre15): makes the always-on per-slot lipsync= self-describing. */
        char tm[16 + PTV_MAX_AUDIO * 12];
        int  tn = snprintf(tm, sizeof tm, "[PTV-RSYNC] tracks:");
        for (k = 0; k < n_audio && tn > 0 && tn < (int)sizeof tm - 12; k++)
            tn += snprintf(tm + tn, sizeof tm - tn, " a%d→in%d", k, asrc_in[k]);
        av_log(NULL, AV_LOG_INFO, "%s\n", tm);
    }

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
    for (k = 0; k < n_cc; k++) {                 /* BEFORE the decode threads: on a create
                                                  * failure the tap is disarmed with nothing
                                                  * yet pushing into the queue (no race) */
        if (!pthread_create(&th_cc[k], NULL, cc_thread, &cc[k])) { started_cc[k] = 1; continue; }
        av_log(NULL, AV_LOG_WARNING,
               "[PTV-CC] emitter thread create failed for input %d — no CC output from that "
               "input (the other extractions are unaffected)\n", cc_slot[k]);
        inputs[cc_slot[k]].dc.cc.on = 0;
        av_thread_message_queue_set_err_send(cc_q[k], AVERROR_EOF);
        /* this emitter's producer slot is retired by hand, or every muxer would wait for an
         * EOF marker that no thread will ever send */
        for (r = 0; r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
    }
    if (multiview) {                             /* AFTER the CC emitters: the compositor's
                                                  * stats line reads dc.cc.on, which the
                                                  * create-failure path above clears */
        int pe = pthread_create(&th_compositor, NULL, compositor_thread, &comp);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        started_compositor = 1;
    }
    {   /* the stats line's cc= is meaningful only while at least one extraction is alive */
        int alive = 0;
        for (k = 0; k < n_cc; k++) alive += started_cc[k];
        if (cc_on && !alive) g_cc_on = 0;
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
        for (k = 0; k < n_cc; k++) if (cc_q[k]) {
            av_thread_message_queue_set_err_send(cc_q[k], AVERROR_EOF);
            av_thread_message_queue_set_err_recv(cc_q[k], AVERROR_EOF);
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
    for (k = 0; k < n_cc; k++)                   /* each decode thread EOF'd its own cc_q at exit */
        if (started_cc[k]) pthread_join(th_cc[k], NULL);
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
    for (k = 0; k < n_cc; k++)                   /* the free func releases any queued ASS lines */
        av_thread_message_queue_free(&cc_q[k]);
    for (k = 0; k < n_cc; k++) avcodec_free_context(&cc[k].enc);
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
        for (r = 0; r < n_rung; r++)             /* shared encoders are aliased: free once */
            if (as[k].enc_owner[r] == r) avcodec_free_context(&as[k].enc[r]);
        avcodec_free_context(&as[k].dec);
        avcodec_parameters_free(&as[k].ist_par);   /* pre17 R1: owned copy */
    }
    avfilter_graph_free(&fg);
    for (k = 0; k < n_input; k++) {
        avfilter_graph_free(&inputs[k].dc.fg);       /* single-input graph (multiview: NULL) */
        avcodec_free_context(&inputs[k].dc.cc.dec);  /* -cc_extract (NULL when off) */
        av_freep(&inputs[k].dc.cc.pend_ass);         /* a caption still inside the debounce */
        avcodec_free_context(&inputs[k].vdec);
        av_thread_message_queue_free(&inputs[k].video_q);
        av_thread_message_queue_free(&inputs[k].hold.q);
        av_freep(&inputs[k].wrap_off);
        av_freep(&inputs[k].wrap_last);
        av_freep(&inputs[k].wrap_wall_last);
        av_freep(&inputs[k].edit_us);
        av_freep(&inputs[k].gap_vsnap);
        av_freep(&inputs[k].wall_cad_us);       /* pre24 #63 */
        av_freep(&inputs[k].pkt_wall_gap_us);   /* pre24 #63 */
        av_dict_free(&inputs[k].da.reopen_opts);   /* pre17 R1 */
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
    { "cc_extract",       OPT_TYPE_BOOL,   OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "extract EIA-608 closed captions to a DVB-teletext subtitle stream" },
    { "cc_slots",         OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "multiview: which inputs to extract CC from (default: all)", "list" },
    { "cc_lang",          OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "language of the extracted CC stream (default: from audio)", "iso639-2" },
    { "cc_page",          OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "teletext page for the extracted CC (default 0x88)", "page" },
    { "cc_magazine",      OPT_TYPE_STRING, OPT_PERFILE | OPT_OUTPUT, { .off = 0 }, "teletext magazine for the extracted CC (1-8, default 8)", "n" },
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
    if (getenv("PTV_NO_VANCHOR")) g_vanchor = 0;               /* d1-fix revert: lone-audio flush falls back to the 3b
                                                                * provisional butt-joint (erases the step, discards the
                                                                * flowing video — the pre7..pre20 regression shape) */
    if (getenv("PTV_NO_AANCHOR")) g_aanchor = 0;               /* 1.0.1-pre22 revert: a lone-video jump keeps the
                                                                * one-sided relabel-erase (video glued, audio untouched,
                                                                * no registration — the pre21 Fashion shape) */
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
    if (getenv("PTV_NO_ADTS_SPLIT")) g_adts_split = 0;     /* 1.0.1-pre19.1: revert the broken-phase probe/stamp fix (pre19 ticking) */
    if (getenv("PTV_NO_TOLERANT_DEC")) g_tolerant_dec = 0; /* 1.0.1-pre19.1: strict lavc AAC everywhere (pre-#38) */
    if (getenv("PTV_NO_ACHOP_REBUILD")) g_achop = 0; /* 1.0.1-pre19 #46: disable the sustained-chop full-path rebuild escape */
    { const char *e = getenv("PTV_ACHOP_ERRS_MIN");  if (e && atoi(e) > 0) g_achop_errs_min  = atoi(e); }
    { const char *e = getenv("PTV_ACHOP_SHEDS_MIN"); if (e && atoi(e) > 0) g_achop_sheds_min = atoi(e); }
    { const char *e = getenv("PTV_ACHOP_SUST_MIN");  if (e && atoi(e) > 0) g_achop_sust_min  = atoi(e); }
    { const char *e = getenv("PTV_ACHOP_RELIMIT_S"); if (e && atoi(e) > 0) g_achop_relimit_us = (int64_t)atoi(e) * 1000000; }
    if (getenv("PTV_NO_ANCHOR_HEADFILL")) g_anchor_headfill = 0;   /* 1.0.1: revert to first-packet-at-first_audio−h0 birth */
    /* 0.9.18.7: PTV_GAP_MIN_MS internalized (700ms — g_gap_min_us) */
    { const char *wg = getenv("PTV_WRAP_GUARD_S"); if (wg && atoi(wg) > 0) g_wrap_guard_us = (int64_t)atoi(wg) * 1000000; }  /* v0.9.16.1 wrap-guard threshold override (TEST ONLY) */
    if (getenv("PTV_NVENC_SERIALIZE")) g_nvenc_serialize = 1;  /* v0.9.16.5 scale fix B2 (opt-in): one process-wide mutex around video encoder calls — cuts concurrent NVIDIA RM-lock callers 6->1 per process */
    { const char *ag = getenv("PTV_AGLUE_MS");     if (ag) g_aglue_ms = atoi(ag); }          /* v0.9.16.3 label-step glue threshold; 0 disables */
    if (getenv("PTV_NO_CONVCAP")) g_convcap = 0;   /* 1.0.1-pre23 revert: above-cap steps all hand to aresample again
                                                    * (the #60 unbounded swr_inject_silence ladder — A/B only) */
    { const char *s = getenv("PTV_CONV_CAP_S");  if (s && atoi(s) > 0) g_conv_cap_us  = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_SEAM_PARK_S"); if (s && atoi(s) > 0) g_seam_park_us = (int64_t)atoi(s) * 1000000; }  /* TEST ONLY (G4) */
    if (getenv("PTV_NO_WALLEV")) g_wallev = 0;     /* 1.0.1-pre24 #63 revert: erase engines back to the pre23
                                                    * whole-step remedies (butt-joint every >1s hole; the
                                                    * corrupt-storm desync counterfactual — A/B only) */
    if (getenv("PTV_NO_RECANCHOR")) g_recanchor = 0;  /* 1.0.1-pre24 #63: no recovery re-anchor (a baked
                                                       * post-storm offset stays until restart) */
    { const char *s = getenv("PTV_RECANCHOR_SETTLE_S");   if (s && atoi(s) > 0) g_recanchor_settle_us   = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RECANCHOR_COOLDOWN_S"); if (s && atoi(s) > 0) g_recanchor_cooldown_us = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RECANCHOR_TEST_ABORT_N"); if (s && atoi(s) > 0) g_recanchor_test_abort_n = atoi(s); }  /* TEST ONLY (rr24 F2 gate) */
    if (getenv("PTV_NO_RESYNC")) g_resync = 0;   /* 1.0.1-pre29.1 #69: >350ms hard-reset path default ON; kill switch reverts to pre28 */
    { const char *s = getenv("PTV_RESYNC_MS");            if (s && atoi(s) > 0) g_resync_ms_us          = (int64_t)atoi(s) * 1000; }
    { const char *s = getenv("PTV_RESYNC_OK_MS");         if (s && atoi(s) > 0) g_resync_ok_us          = (int64_t)atoi(s) * 1000; }
    { const char *s = getenv("PTV_RESYNC_CONFIRM_S");     if (s && atoi(s) > 0) g_resync_confirm_us     = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RESYNC_CONFIRM_BIG_S"); if (s && atoi(s) > 0) g_resync_confirm_big_us = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RESYNC_BREAKER_N");     if (s && atoi(s) > 0) g_resync_breaker_n      = atoi(s); }
    { const char *s = getenv("PTV_RESYNC_BREAKER_WIN_S"); if (s && atoi(s) > 0) g_resync_breaker_win_us = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RESYNC_QUIET_S");       if (s && atoi(s) > 0) g_resync_quiet_us       = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RESYNC_CHUNK_MS");      if (s && atoi(s) > 0) g_resync_chunk_us       = (int64_t)atoi(s) * 1000; }
    { const char *s = getenv("PTV_RESYNC_CHUNK_GAP_S");   if (s && atoi(s) > 0) g_resync_chunk_gap_us   = (int64_t)atoi(s) * 1000000; }
    /* 1.0.1-pre30 #69 refinements */
    { const char *s = getenv("PTV_RESYNC_SEAM_HOLD_S");   if (s && atoi(s) >= 0) g_resync_seam_hold_us  = (int64_t)atoi(s) * 1000000; }   /* 0 explicitly disables the hold */
    if (getenv("PTV_RESYNC_SILENCE")) g_resync_vskip = 0; /* item B kill: pre29 silence-pad actuator */
    { const char *s = getenv("PTV_RESYNC_IDR_WAIT_S");    if (s && atoi(s) > 0) g_resync_idr_wait_us    = (int64_t)atoi(s) * 1000000; }
    { const char *s = getenv("PTV_RESYNC_VSKIP_TOL_MS");  if (s && atoi(s) > 0) g_resync_vskip_tol_us   = (int64_t)atoi(s) * 1000; }
    { const char *s = getenv("PTV_RESYNC_WALK_CEIL_S");   if (s && atoi(s) >= 0) g_resync_walk_ceil_us  = (int64_t)atoi(s) * 1000000; }   /* rr30 T1; 0 disables */
    /* item C: the ring is fixed-size — clamp N into [2, PTV_RSN_RING] whatever the env says */
    if (g_resync_breaker_n < 2)            g_resync_breaker_n = 2;
    if (g_resync_breaker_n > PTV_RSN_RING) g_resync_breaker_n = PTV_RSN_RING;
    { const char *s = getenv("PTV_RSCORR_SLEW_FAST");     if (s) g_rscorr_slew_fast = atoi(s) > 0 ? atoi(s) : 0; }  /* 0 = disabled (always the base clamp) */
    if (getenv("PTV_NO_MUXGUARD")) g_muxguard = 0;   /* 1.0.1-pre26: disable the survive-first backward-dts mux backstop (pre24 EINVAL-exit behavior) */
    { const char *s = getenv("PTV_MUXTEST_BACK_AT_S"); if (s && atoi(s) > 0) g_muxtest_back_at_us = (int64_t)atoi(s) * 1000000; }  /* TEST ONLY (pre26 gates) */
    { const char *s = getenv("PTV_MUXTEST_BACK_MS");   if (s && atoi(s) > 0) g_muxtest_back_ms = atoi(s); }                        /* TEST ONLY (pre26 gates) */
    { const char *s = getenv("PTV_MUXTEST_BACK_HOLD_S"); if (s && atoi(s) > 0) g_muxtest_back_hold_us = (int64_t)atoi(s) * 1000000; }  /* TEST ONLY (pre27 #62 GF gate) */
    { const char *s = getenv("PTV_MUXTEST_BACK_TYPE");                                                                                 /* TEST ONLY (pre27 #62 GH gate) */
      if (s) g_muxtest_back_type = s[0] == 's' ? AVMEDIA_TYPE_SUBTITLE :
                                   s[0] == 'd' ? AVMEDIA_TYPE_DATA : AVMEDIA_TYPE_AUDIO; }
    if (getenv("PTV_MUXDIAG")) g_muxdiag = 1;        /* pre26 D3: gated emission-point backward-label instrumentation */
    if (getenv("PTV_NO_MUXTOL")) g_muxtol = 0;       /* 1.0.1-pre27 #62: any mux write error = fatal (pre26 behavior) */
    { const char *s = getenv("PTV_MUXFAIL_SIM");     /* TEST ONLY (pre27 #62 gates): <errno>:<start_s>:<dur_s> */
      if (s) {
          char name[16] = ""; int st = 0, du = 0;
          if (sscanf(s, "%15[a-z]:%d:%d", name, &st, &du) == 3 && st >= 0 && du > 0)
              g_muxfail_sim_err = !strcmp(name, "enomem") ? AVERROR(ENOMEM) :
                                  !strcmp(name, "eagain") ? AVERROR(EAGAIN) :
                                  !strcmp(name, "einval") ? AVERROR(EINVAL) : 0;
          if (g_muxfail_sim_err) {
              g_muxfail_sim_from_us = (int64_t)st * 1000000;
              g_muxfail_sim_to_us   = g_muxfail_sim_from_us + (int64_t)du * 1000000;
          } else
              av_log(NULL, AV_LOG_WARNING, "[PTV-MUXTOL] bad PTV_MUXFAIL_SIM '%s' "
                     "(want enomem|eagain|einval:<start_s>:<dur_s>) — ignored\n", s);
      } }
    { const char *s = getenv("PTV_NOVIDEO_EXIT_S"); if (s) g_novideo_exit_us = (int64_t)atoll(s) * 1000000; }  /* 1.0.1-pre23 startup sanity; 0 disables */
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
    if (getenv("PTV_NO_MV_BIRTHTRIM")) g_mv_birthtrim = 0;  /* pre17: one-shot trim only (re-enables the birth catch-up slide) */
    if (getenv("PTV_NO_AGLUE_CEIL"))   g_aglue_ceil = 0;     /* pre17 fix round: unbounded refused-step pursuit (pre-Fashion posture) */
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
    if (getenv("PTV_NO_PLL_YIELD")) g_pll_yield = 0;       /* 1.0.1-pre21 #24 revert: PLL keeps actuating while the corrector steers (the 1:1 cancel — A/B only) */
    { const char *ts = getenv("PTV_TEST_VDEC_STALL_S"); if (ts && atoi(ts) > 0) g_test_vdec_stall_us = (int64_t)atoi(ts) * 1000000; }   /* TEST-ONLY: [PTV-STALL] live-fire */
    { const char *ts = getenv("PTV_TEST_VDEC_STALL_AT_S"); if (ts && atoi(ts) >= 0) g_test_vdec_stall_at_us = (int64_t)atoi(ts) * 1000000; }   /* TEST-ONLY (pre23): stall trigger time; 0 = startup wedge */
    if (getenv("PTV_ACQ_INSTANT")) g_acq_instant = 1;      /* 1.0.1: revert ACQUIRE to single-window fire (no 3-consecutive-window sustain; the tick floor stays) */
    if (getenv("PTV_NO_PLL_TRACKUP")) g_pll_trackup = 0;   /* 1.0.1-pre3: disable the steer-TRACK entirely (acquire-only; labels flat, no steer — the production mute) */
    /* 0.9.18.7: PTV_PLL_EMA_SHIFT (7) / PTV_PLL_TAU_MS (5000) / PTV_PLL_ACQUIRE_MS (40) /
     * PTV_PLL_ACQUIRE_N (32) / PTV_PLL_REFRACTORY_MS (12000) / PTV_PLL_NOISE_K (3) /
     * PTV_PLL_DEV_SHIFT (9) internalized — see the g_pll_* declarations */
    { const char *tn = getenv("PTV_PLL_TESTNOISE_MS");  if (tn && atoi(tn) > 0) g_pll_testnoise_us  = (int64_t)atoi(tn) * 1000; }  /* TEST-ONLY: inject ±N ms offset square wave */
    { const char *tp = getenv("PTV_PLL_TESTNOISE_P");   if (tp && atoi(tp) > 0) g_pll_testnoise_frames = atoi(tp); }              /* TEST-ONLY (#49 gate): square-wave half-period, frames */
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
    if (getenv("PTV_NO_GLUEVETO")) g_glueveto = 0;   /* 1.0.1-pre18 #50: gap-verdict-vs-LAYERA one-remedy invariant off */
    if (getenv("PTV_NO_HSTICK_FILTER")) g_hstick_filter = 0;   /* 1.0.1-pre18 #51a: hs-tick steps count as corrector events again */
    if (getenv("PTV_NO_RSCORR_CEIL")) g_rscorr_ceil = 0;       /* 1.0.1-pre18 #51b: no starvation-ceiling engage */
    { const char *s = getenv("PTV_RSCORR_CEIL_MIN"); if (s && atoi(s) > 0) g_rscorr_ceil_us = (int64_t)atoi(s) * 60000000; }  /* #51b ceiling, minutes */
    if (getenv("PTV_NO_PERSTREAM_WM")) g_perstream_wm = 0;     /* 1.0.1-pre18: §7.5b back to the aggregate (least-delayed) key */
    if (getenv("PTV_NO_REBUILD_REANCHOR")) g_rebuild_reanchor = 0;  /* 1.0.1-pre20: AFMT rebuild carries the old base again (residual + corrector walk) */
    if (getenv("PTV_NO_ACQ_BACKOFF")) g_acq_backoff = 0;       /* 1.0.1-pre18 #49: no repeated-ACQUIRE threshold backoff */
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
    if (getenv("PTV_NO_AENC_SHARE")) g_aenc_share = 0;  /* one audio encoder per rung (pre-1.2.0 behaviour) */
    if (getenv("PTV_NO_CC"))       g_cc       = 0;   /* CC->teletext kill switch: ignore -cc_extract entirely (byte-inert output) */
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
