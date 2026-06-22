/*
 * ptvencoder — purpose-built live MPEG-TS re-encoder.
 *
 * A sibling fftools program (alongside ffmpeg/ffprobe/ffplay) that links the
 * same patched libav* libraries but runs on its own house-clock timing engine.
 * See analysis/ptvencoder-functional-spec.md.
 *
 * Phase 1, increment 5: pull-based multi-stage pipeline (the professional model).
 *
 *   demux ─video_q─▶ decode (free-run) ─frame_q─▶ output(master clock, sample
 *         ─audio_q─▶ audio (decode▶resample▶AAC) ──────────────────┐  & hold)
 *                                                          mux_q ◀──┴──▶ mux
 *
 *  - A free-running output clock is the master: a wall-paced timer in the output
 *    thread emits at the house rate no matter what upstream does.
 *  - The frame synchronizer is sample-and-hold: decode runs free in its own
 *    thread and keeps the latest decoded frame current (frame_q, drop-oldest);
 *    each output tick samples it — repeat if decode is behind, drop intermediate
 *    frames if it is ahead. Source PTS is advisory; video output PTS is the tick
 *    counter, so source wrap/jump/gap is invisible to the output (no re-anchor
 *    needed — the pull model dissolves it).
 *  - The encoder is downstream and never allowed to block the clock from
 *    draining input: if it stalls (e.g. NVENC blocking the caller under GPU
 *    load — the failure this design exists to survive), decode keeps running,
 *    the demuxer keeps draining the socket (dropping on a full queue), and a
 *    watchdog flags the stall. (Auto-reinit of a hung in-process session is a
 *    follow-up; this build contains + detects.)
 *  - The mux is clock-locked and keeps emitting (dup-fill) so the TS never stops.
 *
 * Video and audio map onto one shared input anchor (h0) for A/V start offset.
 *
 * This file is licensed under the same terms as FFmpeg (GPL, --enable-gpl).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
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

const char program_name[] = "ptvencoder";
const int  program_birth_year = 2026;

/* ptvencoder's OWN version, independent of the FFmpeg git-describe string (which,
 * on the BtbN box build, reflects fresh-upstream + `git apply` and so does NOT
 * encode which patch revision is applied). Bump by hand on each meaningful fix/
 * feature so a deployed binary self-identifies via the banner / -version. */
#define PTVENCODER_VERSION "0.6.19"  /* 0.6.19: (1) FIX the always-on [PTV-AVSYNC] `lipsync=` headline for the B3 PLL path — it was the async_pad span estimate (outspan−content−lag_true), which does NOT account for the PLL's content drop/pad retiming (af_applied_us), so on a CONVERGED slot it kept reporting the bank the acquire had already removed (lipsync ≈ applied → e.g. RAV box read lipsync −258/−812/−1665/−504ms while the faithful vring-paired `offset` was −8/−85/+3/+4ms and the eye was clean). Operators (and any monitoring/sync-check parsing the field) read "off" on a synced channel. FIX: headline the FAITHFUL measured offset (−av_offset_us; sign: offset<0 = audio late ≡ lipsync>0 = audio late) whenever the vring has paired, fall back to the span estimate only before it pairs (offset=--). Logging-only, BYTE-IDENTICAL. (2) TIGHTEN the PLL dead band: lower g_pll_acquire_us 100→40ms so a STABLE sub-100ms residual gets snapped in by a whole-frame acquire instead of stranded (TRACK is guard-limited on jittery NTSC sources → can't trim it; box a1 sat at −84ms: 84<100 threshold so acquire refused, TRACK too weak). The flatness debounce (threshold/4 = 10ms) still rejects jitter so converged (|ema|<40) and wandering slots don't churn. Plus acquire quantization now rounds to the NEAREST whole frame (was truncate-toward-0) → residual ≤½ frame (~11ms) not ≤1 frame; a1's −84ms → one acquire → ~−1ms (well inside the ±25ms gate). MULTIVIEW-PLL only; PTV_PLL_ACQUIRE_MS overrides. 0.6.18: B3 PLL fix — acquire on ANY stable large offset (drop the startup-window/disturbance-event gate). tmtg RAV box A/B of 0.6.17 (PTV_DIAG): the three FAST-forming banks acquired in the first ~3s and converged (offset −32/−42/−89ms), but a slot's SLOW-forming +1.1s bank stabilized AFTER the 5s startup window with no disturbance epoch → the gate refused to acquire it → stuck at −1100ms. ROOT: big banks accumulate slower than the 5s window; the startup/event gate (v0.6.17 N4) is too restrictive. FIX: `may_acq` drops the `((in_startup && acq<k) || armed)` clause — fire whenever |pll_ema|>threshold AND stable (the stability-debounce already rejects noise; the refractory throttles; converged slots' sub-threshold residuals won't re-fire). A frozen bank now converges in 1–2 refractory-throttled acquires regardless of when it forms. The startup-window/disturbance-epoch machinery (g_pll_startup_us, g_pll_acquire_k, the atomic per-input epoch + its compositor/demux writers) is now DORMANT (left in place, harmless). KNOWN residual (deferred): the ~30–90ms post-acquire residual on the fast slots is TRACK-limited — the monotonic guard fires on most frames of these jittery NTSC sources so the conditional-integration anti-windup skips the integral; acquire (content drop) is the reliable lever, TRACK (PTS nudge) has weak authority here. 0.6.17: A/V PLL redesign Phase B3 — CLOSED-LOOP two-regime A/V controller for MULTIVIEW transcoded audio (PTV_AVSYNC_PLL, default OFF for box A/B). ROOT (measured fleet-wide on tmtg-cor-transcoder-2 via the always-on [PTV-AVSYNC] vlag/alag split): every multiview slot banks a CONSTANT per-slot audio-late offset at startup (alag a frozen +0..+2100ms STEP — GAFamily +0, PureFlix +499, RAV +2034, GBNews +2100 — NEVER drifting; vlag never drifts ⇒ audio-side, not a video regression). aresample=async over-produces over the startup gap and the v0.6.8 monotonic guard FREEZES it; the open-loop audio-follow steers applied→house_skew (the VIDEO's lag) and is structurally blind to the bank (applied=house_skew, trk≈0, yet offset −61..−2163ms). A pure legacy-style slow integral is the WRONG tool for a 2s STEP (210s to unwind at 10ms/s). FIX: close the loop on the faithful measured av_offset_us (=vlag−alag) with a two-regime controller, BOTH on the B1 content-anchored base (want=opts+applied) so the acquire's content-drop and applied step CANCEL in want (guard never sees a backward step): (1) ACQUIRE — one-shot drop(advance)/pad(delay) sized to the FROZEN bank, snaps it out in one tune-in skip; stability-debounced (large AND flat, so Δ sizes the frozen — not still-forming — bank), startup fires ≤k times then mid-run only on a disturbance epoch (atomic, bumped by slate-return in compositor_thread + discont absorber in demux_unwrap — two writers); bumpless EMA credit (pll_ema−=Δ) + refractory so it never re-fires/over-drops. (2) TRACK — type-1 integral trim (step=ema·frame/τ, rate-clamped to g_af_rate_us, no dead zone), conditional-integration anti-windup vs the guard; type-1 suffices since the disturbance is a STEP (no ramp ⇒ no r·τ residual). Drives offset→0; gate is the EMA-SMOOTHED offset (instantaneous = vlag jitter, which the loop rides at the mean, not chases). MULTIVIEW transcoded audio ONLY; single-input + copy paths BYTE-IDENTICAL/unchanged (flag-gated; copy stays on the ≥0 floor + monotonic clamp). Knobs: PTV_PLL_{EMA_SHIFT,TAU_MS,ACQUIRE_MS,ACQUIRE_N,STARTUP_MS,ACQUIRE_K}. [PTV-AVSYNC] gains a pll[ema/applied/acq/guard/drop/pad] view; [PTV-PLL] logs each acquire (PTV_DIAG). 0.6.16: the always-on [PTV-AVSYNC] line now leads with `lipsync=±Nms` — the faithful pipeline-introduced lip-sync error (audio's realized output-vs-content lag async_pad − video's TRUE lag lag_true; + = audio late), i.e. the PTV_DIAG [PTV-LIPSYNC] `err` folded into the no-flag stats line (operator's headline A/V number). `offset` (vring-paired) stays as the independent cross-check (lipsync>0 ≈ offset<0). The old multiview actuator-residual field `err=` is renamed `trk=` to avoid two "err"s. Logging-only, byte-identical. 0.6.15: FIX — multiview per-slot audio banked 0.3–2.5s LATE at startup (tmtg-cor-transcoder-2; "sound completely off"). ROOT (reproduced local from real tmtg RAV captures via tsp UDP, traced with PTV_ATRACE): h0 was anchored in the DECODE thread to the first DECODED frame, but under a deep startup jitter-buffer prime the compositor's first DISPLAYED frame is a different/later content → the displayed video leaps ~prime-depth AHEAD of h0 at tick 0 → P2 (v0.6.3 h0 re-anchor) shoves h0 forward → the transcoded audio's opts steps backward → the v0.6.8 monotonic guard advances out_a and FREEZES a permanent +Δ bank; a COPIED audio track's DTS would jump backward into the clamp (freeze; historically the EINVAL no-data outage — d2460f4180/v0.2.3, 9b1f3843f9/v0.2.1). FIX: anchor each slot's h0 at the COMPOSITOR'S FIRST DISPLAY (h0 = disp_src − tick·tick_dur) so sk=0 from the start and P2 never fires — audio + copied tracks anchor to the SAME h0, nothing banks or clamps. MULTIVIEW ONLY; single-input keeps the decode-thread anchor (BYTE-IDENTICAL). PTV_NO_H0_AT_DISPLAY=1 reverts (A/B). Doesn't touch buffer depth/cushion/latency (B2 "deep prime helps" intact) — distinct from the v0.5.9 per-frame display clamp that stuttered. Verified local: tmtg4 alag→0 all slots, copied AC-3 monotonic (no clamp), grid4 + single-input + single-input-AC-3-copy unchanged. PTV_ATRACE = temp per-audio-frame B1 trace (default off, byte-identical). 0.6.14: FIX — copied SPARSE streams (DVB subtitles, SCTE-35 data) were thrown OUT OF SYNC by the v0.6.0 discontinuity-absorber. demux_unwrap's forward-DTS-jump absorber (g_discont, re-base a >PTV_DISCONT_MS=80ms jump to last+nominal) ran on EVERY stream. Continuous video/audio are ~1 frame apart so only a real glitch trips it — correct. But sparse subtitle/data streams have NATURAL multi-second inter-packet gaps (DVB-sub events seconds apart, SCTE-35 ad markers minutes apart), so essentially EVERY packet's gap exceeds 80ms and got "absorbed" → the sparse timeline COLLAPSED (subtitle inter-packet deltas crushed to the 20ms nominal, whole stream shifted) → subs drift out of sync / never paint; ad-marker positions would shift too. ROOT-CAUSED by A/B on cinestar_src_5min.ts (4 DVB subs): default ON crushed deltas to 0.02s; PTV_NO_DISCONT=1 preserved the source deltas exactly. FIX: gate the forward-jump absorber to ct==VIDEO||ct==AUDIO; SUBTITLE/DATA skip it. The 33-bit WRAP branches (delta<-half / >half) STILL apply to every stream (copied AC-3/SCTE-35 across the 2^33 roll — the v0.2.1 reason). Verified: default-ON output now reproduces source subtitle deltas bit-for-bit, video/audio discont-absorb unchanged. 0.6.13: FIX — CLI -metadata:s:s:N / -disposition:s:N on COPY streams (subtitles, extra-audio, data) was IGNORED. The transcoded-audio path applied per-stream CLI metadata/disposition (apply_stream_meta, type 'a'), but the copy/passthrough loop only carried the SOURCE language+disposition and never called apply_stream_meta → -metadata:s:s:N language=/title= and -disposition:s:N silently did nothing (multiview operators could name audio views but not subtitle views). FIX: the copy loop now calls apply_stream_meta(tlet, tidx) per copy stream, with the type letter from the source codec_type (a/s/v/d) and a per-type output index seeded past the transcoded streams (1 composite video + n_audio transcoded audio; subs/data start at 0), incremented in stream-creation order = FFmpeg's -metadata:s:<t>:N numbering. So copied subtitles now honor -metadata:s:s:N (title/language override) + -disposition:s:N, exactly like audio. Source language/disposition stays the default when no CLI override is given. 0.6.12: A/V-SYNC STATUS in STATS (always-on, single-input incl., §8). The MEASURED A/V offset (offset=vlag−alag, − = picture ahead of audio) is now computed on EVERY channel (the [PTV-AVSYNC2] video-output ring + the audio-drain pairing are no longer gated by PTV_AVSYNC_PROBE — they were proven negligible-cost on the box, "probe on ALL channels", speed=1.00x) and printed on the -stats_period cadence at AV_LOG_INFO via the [PTV-AVSYNC] line, so operators running `-stats_period 10` at info level SEE the real lip-sync number live without any env flag. The line now prints for SINGLE-INPUT too (was multiview-only): single-input shows offset + vlag/alag + house_skew (its house-clock lock state); multiview additionally shows the per-slot actuator state (applied/err/nudge/acq). offset=-- until the first audio frame pairs against the video ring. The verbose [PTV-AVSYNC2] decomposition (vlag/alag base/dev + ring + pairδ) stays opt-in behind PTV_AVSYNC_PROBE for deep diagnosis. Fulfills the plan §8 commitment ("the [PTV-AVSYNC] status line gains the measured offset ... single-input included"). Output BYTE-IDENTICAL (logging-only). 0.6.11: built-in discardcorrupt now covers ALL streams (was video-only) so dropping the CLI -fflags +discardcorrupt is a zero-regression swap — and unlike the CLI flag (which makes libavformat discard SILENTLY inside av_read_frame, hiding the count), the built-in COUNTS video corrupt drops in the `corrupt=` stat. RECOMMENDATION: remove -fflags +discardcorrupt from the command line and rely on the built-in (default ON; PTV_KEEP_CORRUPT=1 reverts). 0.6.10: FRAME-LOSS in STATS + -fflags +discardcorrupt. Surface the two frame-loss sources that cause the multiview leap, on the -stats_period cadence (so operators SEE it): per-input qdrop (video_q overflow = vdrop) + corrupt (demux AV_PKT_FLAG_CORRUPT discards + decode AV_FRAME_FLAG_CORRUPT drops). Multiview: appended per-input to the compositor stats line (" in0:qdrop=N/corrupt=M …"); single-input: "qdrop=N corrupt=M" on the progress line. Plus g_discardcorrupt (default ON, = -fflags +discardcorrupt): the demux now COUNTS and DROPS corrupt video packets before decode (DemuxArgs.vcorrupt) — a corrupt frame, like a dropped one, becomes a content gap the position-anchored composite leaps across; discarding early + counting makes it observable. PTV_KEEP_CORRUPT=1 keeps them (prior behavior). 0.6.9: Phase B #1 — deeper video_q (48→256) to STOP startup frame loss = the multiview leap ROOT. Box (multicast, realtime source) dropped ~30 video frames in the first ~1s (vdrop spiked then went flat; holddrop=0; [PTV-DISCONT] NEVER on the video stream → NOT a PTS discontinuity, the absorber can't help) because the decoder's init window produces nothing while the realtime source fills the 48-deep video_q → overflow drop → a content GAP → the position-anchored composite video LEAPS to the newest content → audio left behind = ~600ms-1s per-slot PICTURE-AHEAD (the P2/REANCHOR2 + audio-late residual all trace to this). Single-input is immune: content-anchored video turns a dropped frame into a harmless output PTS gap (A/V stay aligned) — which is ALSO why the legacy single-input PLL never saw this. FIX: video_q 48→256 absorbs the one-time decoder-init backlog (decoder then drains it faster than realtime and catches up; steady-state near-empty); drop-newest stays the backstop for sustained overload. PTV_VIDEOQ overrides. Validate on the relay: vdrop→0, no content leap, [PTV-AVSYNC2] offset stays ~0 (no P2 storm, no audio-late). 0.6.8: B1 MONOTONIC GUARD (fix box stall). B1 set multiview audio out = opts + offset, where opts (async/buffersink output pts) STEPS BACKWARD when h0 is re-anchored forward (P2: opts = buffersink − h0_samp, larger h0 → smaller opts) or at a source PTS discontinuity (box live feeds throw +hundreds-of-ms jumps). The pre-B1 free counter was monotonic BY CONSTRUCTION; content-anchoring lost that → backward out_a → libfdk_aac "Queue input is backward in time" + mpegts "non monotonically increasing dts" → that audio stream stalled and the mux interleaver WEDGED (muxed frozen, frame_q full → grids sent NO data; box-observed at startup, ~−785ms backward step). FIX: keep out_a monotonic + frame-spaced (af_last_out guard); on a backward step it advances at nb (dense, like the old counter) until opts recovers. Local missed it (built-in aac not libfdk_aac; file not UDP; P2 non-deterministic). 0.6.7: A/V PLL redesign Phase B (B2) — track vlag faster. B1's content-anchored offset tracked the per-slot lag at only 2 ms/s, so a slot whose source is momentarily slow at startup (its cell dups → vlag ramps ~6–8 ms/s and settles at e.g. +200 ms) had the audio converge over ~100 s (≤~140 ms audio-ahead meanwhile). REVISED ADR: this residual is SOURCE-SLOWNESS dups, NOT the deep prime — and post-B1 a deep prime HELPS (its cushion absorbs slowness, keeping vlag=0 longer; a shallow prime underruns sooner). So "bound the prime" was backwards. FIX (B2): raise the follow-rate ceiling g_af_rate_us 2→10 ms/s so the smooth content-anchored offset tracks the dup ramp in near-real-time; 10 ms/s ≈ 1% (under the ~2% audible budget), engaged only transiently during convergence (steady-state step=gap is tiny). PTV_AF_RATE_MS_S overrides. 0.6.6: A/V PLL redesign Phase B (B1) — CONTENT-ANCHORED multiview audio. Phase A's [PTV-AVSYNC2] probe localized the per-slot desync to the AUDIO side: the multiview audio-follow emitted on a FREE-RUNNING sample counter (af_next_pts) that banked aresample=async's STARTUP over-production into a permanent audio-late offset (alag +400..+1252ms, the dominant term; vlag≈0). Single-input was immune because content-anchored (out=opts, async's self-correcting target). FIX (B1): multiview audio now also CONTENT-ANCHORS — out = opts + a smooth rate-limited offset (≤PTV_AF_RATE_MS_S) that tracks the compositor's per-slot lag (house_skew) so audio follows the video DISPLAY, seeded to the current lag at the first frame (no glitch). No free counter, no drop/pad/silence/acquire. Makes alag→house_skew=vlag ⇒ measured offset→0. Converges multiview audio onto the single-input mechanism (unification down-payment). MULTIVIEW ONLY; PTV_AF_NO_ANCHOR=1 reverts to the pre-B1 free-counter path (A/B); single-input BYTE-IDENTICAL. Validate via [PTV-AVSYNC2] on the SRT relay: offset→~0 on the deep-prime slots, clean slots unchanged. (B2 = bound the deep video prime; B3 = thin closed-loop trim on the measured offset — to come.) 0.6.5: A/V PLL redesign Phase A — READ-ONLY measurement probe [PTV-AVSYNC2] (PTV_AVSYNC_PROBE=1; analysis/ptvencoder-avsync-pll-redesign-plan.md). Measures the REAL per-track lip-sync offset out_v(C)−out_a(C) by pairing each emitted audio frame's source content C against the output time the VIDEO showed that SAME content (a per-input ring of (displayed abs-src → out_v), written by the compositor in multiview / the master output thread in single-input). Reports offset + the video_lag/audio_lag split (§3.2a: which side moved, with slow EMA baselines + deviation) + the content pairing residual (§3.2b). Faithful where the old proxies (async_pad/house_skew/lip-sync err) were confounded for the audio-follow path, because out_a is the ACTUAL emitted pts (af counter+nudge in multiview, opts in single-input). NO actuator — measures only; gated PTV_AVSYNC_PROBE, prints on the -stats_period cadence. Single-input subtracts house_skew from the audio content key (single-input injects it at the graph input). M-b cross-check = "offset ≈ 0 on a clean synced source" (local clean run), not separate code. Purpose: validate the measurement against the box eye (mvb ~1s, Fé drift) BEFORE building the closed-loop controller (Phase B). Zero behavior change when off; BYTE-IDENTICAL output. 0.6.4: multiview audio-follow ACTUATOR upgrade (P1) — glitch-free smooth tracking. The v0.6.2/0.6.3 audio-follow corrected the per-slot lag only with whole-frame drop/pad on a 40ms threshold → it tracked the slow per-slot drift (source-vs-house clock, measured +360ms/35min on a live slot) in discrete ~40ms hops (box symptom: "A/V sometimes off then OK again") and left a ±frame residual ("audio slightly behind" on re-anchored slots). FIX: two-mode per-slot controller in audio_drain_fg — keep the fast discrete drop/pad ONLY to ACQUIRE a large gap (>PTV_AF_ACQUIRE_MS, default 100; startup ramp / big jump the smooth rate can't catch), and otherwise TRACK the residual+drift with a SMOOTH rate-limited (≤PTV_AF_RATE_MS_S, default 2ms/s — imperceptible) sub-sample PTS nudge (af_nudge_us added to the continuous output counter; no silence inserts, no content drops, monotonic since rate≪frame). Reuses the legacy 0007 PLL's rate-limited actuator idea, per slot, driven by the directly-measured lag (NO jump_comp). MULTIVIEW ONLY; PTV_AF_NO_PLL reverts to pure discrete; single-input BYTE-IDENTICAL. Verified local SRT relay (2x1+4-up): zero discrete hops after startup (smooth nudge tracks the drift), transcode↔copy xcorr +48ms→+21ms (q=0.994), 0 clamp, all healthy. Also adds the always-on [PTV-AVSYNC] per-slot status line on the -stats_period cadence (obeys -nostats, NOT gated by PTV_DIAG): lag (cell's measured video-vs-house offset) / applied (audio retiming in effect) / err (lag−applied residual, ~0=tracking) / nudge / cumulative acquire work. 0.6.3: multiview per-slot h0 RE-ANCHOR — floor each slot's lag to ≥0 so a cell is never displayed AHEAD of the house clock, which (a) is physically wrong for a frame-synchronizer and (b) is UNCORRECTABLE on a COPIED audio track (a copy can only be delayed, not advanced — a backward DTS hits the monotonic clamp). ROOT (P0, local SRT relay): a slot's video leaps far ahead (−560ms on a 2x1, up to −2.5s on a 4-up) from an anomalous first decoded frame and/or a deep startup buffer prime (the open-join barrier lets fast decoders over-fill their jitter buffer while waiting for the slowest input). FIX: when a slot's lag drops below −PTV_H0_REANCHOR_MS (default 120), re-anchor its h0 forward (h0 += deficit) so the lag lands at +1 tick (slot reads slightly BEHIND = the normal buffered state): video display unchanged, transcoded audio rides the same h0+house_skew (stays locked), copied audio now only needs to DELAY → correctable. MULTIVIEW ONLY (n>1); PTV_NO_H0_REANCHOR reverts; PTV_H0_REANCHOR_MS tunes. VERIFIED local SRT relay (real POP TV + HD History + the 4 Grid feeds): 4-up lag −2500ms→≥~0; transcode-path(drop/pad) vs copy-path(h0-rebase) of the SAME source agree to +48ms (xcorr q=0.987) ⇒ copy tracks video as well as the box-confirmed transcode; no muxer clamp storm; single-input BYTE-IDENTICAL. (PTV-LIPSYNC err is confounded for the audio-follow path on jumpy sources — the xcorr path-consistency check + box eye-check are the oracles.) 0.6.2: AUDIO-FOLLOW now CONTINUOUSLY RE-TRACKS (was one-shot). ROOT (box+local-SRT-relay confirmed via PTV-LIPSYNC trajectory): a slot's per-slot video lag RAMPS IN over ~30s (in1: 0→+320ms then rock-stable) but the one-time latch fired at t≈1s while lag was still ~0, latched 0, never re-tracked → that slot's audio left permanently ~the steady lag AHEAD of video (~1s on the box; mvb/HD-History symptom). FIX: compositor keeps a slow EMA (~1.3s, smooths ±100ms interlaced-PTS jitter) of each slot's measured lag and publishes it every tick instead of freezing one early value; the drain's existing >40ms-threshold deterministic drop/pad re-tracks it (lag>0 pad/delay, lag<0 drop/advance) through the startup ramp and any later drift, and stays put once settled (no churn). Multiview-only, gated n_input>1 && g_audio_follow; single-input BYTE-IDENTICAL. PTV_NO_AUDIO_FOLLOW reverts. 0.6.1: multiview per-slot A/V FIX = AUDIO-FOLLOW (Option A). ROOT (confirmed: single-input synced on the SAME live feed, multiview not): composite video is POSITION-anchored (rung_pts; one shared frame can't be content-stamped per cell) while audio is CONTENT-anchored (src−h0); at the join they sit on different origins → stable per-slot offset = "audio behind video". Single-input has no split (video=(src−h0)/tick, same h0 as audio) → untouched. FIX (multiview only, n_input>1): compositor latches each slot's STABLE signed offset (avg past the lossy join, re-latch on outage); the slot's audio applies it as a ONE-TIME deterministic correction — DROP content (advance) or PAD silence (delay) on a continuous gapless output counter — landing audio on the video's displayed-content clock. Deterministic because aresample=async is far too slow (~20ms/s) and can't advance. PTV_NO_AUDIO_FOLLOW reverts. 0.6.0 discont-absorber: default ON (helps copy/sparse streams); 0.5.9 clamp: DEFAULT OFF (stuttered; opt-in PTV_MV_CLAMP). 0.6.0: FIX multiview per-slot audio-late = per-input source PTS-DISCONTINUITY ABSORBER. ROOT CAUSE (live-repro confirmed via SRT relay of the real 1080i50 Grid feeds): the source throws a forward PTS jump of a few hundred ms to ~1s at the live join (in1 +480 / in3 +960) while FRAMES stay continuous (one per tick, buffer full, holddrop=0) — a timestamp glitch, NOT lost frames. Raw, it shifts that slot's content→output mapping so the cell video leaps ahead of its (continuous) audio = per-slot "audio behind video", stable for the whole run. FIX: demux_unwrap now also absorbs an arbitrary forward DTS jump (>PTV_DISCONT_MS, default 80) the same way it absorbs the 33-bit wrap — a per-stream offset re-basing to last+nominal, keeping video+audio+copy on one continuous timeline → lag→~0, video stays SMOOTH. The 0.5.9 content CLAMP is now DEFAULT OFF (it stuttered on jittery sources; opt-in via PTV_MV_CLAMP). PTV_NO_DISCONT reverts. 0.5.9: (superseded) house-clock content clamp. ROOT CAUSE (box+local confirmed): a startup/source PTS gap (skipped/corrupt frames during decode startup) makes a cell's video content leap AHEAD of the house clock (lag_true −480..−900ms, audio innocent); the compositor showed the jumped frame immediately → video raced ahead of its (continuous) audio → per-slot "audio behind video". FIX: compositor displays a frame only once the house clock reaches its content-time (disp−h0 <= out+1tick); an ahead-of-clock frame is held one tick at a time → the cell's video freezes across the lost-frame gap (correct for lost video) while audio continues → A/V stays locked, lag→~0. Compositor-only; audio path untouched. Validated locally vs the flash+beep ruler (startup-gap repro: slot +354ms→~baseline). PTV_NO_MV_CLAMP=1 reverts. 0.5.8: [PTV-START] per-tick startup trace (first ~3s, per slot: h0/srcpts/content-age/output/qd) to SEE how the per-slot video-lead onsets — box showed holddrop=0 (NOT jitter-buffer drops) yet video leads the house clock, so the cause is a startup PTS irregularity, not a buffer. 0.5.7: compositor mv line adds per-slot /holddrop= (per-input jitter-buffer drop-oldest count) to confirm the multiview audio-late ROOT CAUSE measured on the box: per-slot VIDEO leads the house clock by a fixed startup offset (lag_true −640/−800/−900ms, mvb 0; audio innocent: house_skew=0 async_pad=0). Hypothesis: early-decoding feeds overflow the 48-frame hold at startup → drop-oldest skips that cell's content ahead → displayed video leads. holddrop>0 on the leading slots at startup confirms it. 0.5.6: PTV-LIPSYNC diag — faithful per-slot pipeline-introduced lip-sync error err=async_pad−house_skew (REPLACES the misleading av_off, which was production-thread buffer lead, not playback sync: +3.5s offline / −0.2s live for identical IN-SYNC content). Compositor mv line adds per-slot /lag= (TRUE uncapped video lag) so 250ms-cap saturation is visible (lag>>skew = audio can't follow). Cross-validated locally vs the flash+beep ruler (grid4-sync.sh): healthy err≈0 == ruler in-sync. Gated PTV_DIAG. 0.5.5: FIX per-slot audio-late in multiview — audio arriving before a slot's video sets h0 was DROPPED, so the slot whose video is slowest to acquire its first frame lost the head of its audio (first_out up to ~1s = the per-slot "audio delayed"; box-confirmed on Grid_2x2). Now buffer pre-h0 audio (bounded ring) and replay it once h0 is known, keeping content>=h0 -> first_out~0 all slots. ADIAG probe retained to verify. 0.5.4-adiag (diag, temporary): PTV-ADIAG per-track audio-vs-video offset (av_off) + g_vout_us — MP2 tracks land ~1s audio-late from aresample=async startup over-production (first_out~16ms aligned; av_off ~+1s for high source-V-A MP2 inputs). Gated PTV_DIAG; remove after fix. 0.5.3: code-review fixes — og_spec buffers sized for "disposition" (was silently truncating -disposition/-disposition:a); -an now honored (suppresses auto-selected audio + audio copy in the no-map path); accurate -h (dropped never-wired -s/--deint/--hw/--mode); fifo alloc NULL-checked; g_muxed/g_muxed_bytes atomic (stats data race). 0.5.2: Option F COARSE half — re-anchor on return-from-outage: clear a slot's accumulated dup skew when it comes back after a black-slate, so a continuous-PTS feed re-syncs A/V on return (the source's advanced PTS lands at current output once stale skew is gone). Safe (outage>=slate exceeds cleared skew -> async input still forward). PTV_NO_REANCHOR=1 for A/B. 0.5.1: Option F skew made NON-DECREASING+capped (async input requires monotonic; a decreasing skew stalled the mux — 20fps-into-25fps repro). Fine-half only: rising dup drift rides async; the DECREASING re-anchor-on-return (full A/V re-sync after outage) needs a separate direct output-offset path (coarse half, TODO). 0.5.0: multiview A/V sync = software frame-synchronizer (Option F). Compositor publishes the MEASURED per-slot output-vs-content skew = out_time-(displayed_src-h0) of the frame each cell actually shows; the slot's audio rides it (existing async path), so A/V stays locked through jitter/drop/dup/interruption and RE-SYNCS on input return (shows current content). Reduces to ~0 on healthy 1:1 inputs (no regression). Replaces the dup-event counter (0.4.1), which missed drops/returns. Copy path protected by its monotonic-DTS clamp. 0.4.1: multiview per-slot audio skew = dup-event counter (was arithmetic+non-decreasing, which locked startup jitter -> later-priming slots' audio over-delayed); per-slot skew in PTV_DIAG. 0.4.0: MULTIVIEW (1/2/4-input mosaic — house-clock compositor, per-input jitter buffer + clock, per-slot audio/sub, parallel open); 0.3.0: multiple transcoded audio tracks + per-track -ac/-filter:a/-metadata + source fan-out; 0.2.3: monotonic-DTS clamp on copy path; 0.2.2: no -r preserves source FRAME rate (avg); 0.2.1: 33-bit PTS-wrap on copy-passthrough */

#define PTV_QDEPTH      48     /* demux->decode packet queue (~1s jitter) */
#define PTV_FRAME_QDEPTH 48    /* decode->output jitter buffer (frames); holds the pre-roll cushion */
#define PTV_WD_DEADLINE_US (2 * (int64_t)AV_TIME_BASE)   /* watchdog stall threshold */

/* Diagnostics (env PTV_DIAG=1): per-second stage counters + slow-call
 * breadcrumbs to localize a stall. Temporary, gated, low-overhead (Rule 0). */
static int     g_diag;
/* A/V common-mode lock: the video frame-synchronizer's dup/drop makes the house
 * clock run ahead of source content; that skew is published by the master output
 * thread and added to the audio resampler's target so audio rides the SAME house
 * clock as video (else audio stays source-locked and drifts ~40ms per video dup).
 * On by default; PTV_NO_AVLOCK=1 reverts to the old source-locked audio. */
static int     g_avlock = 1;
/* multiview coarse re-anchor: clear a slot's accumulated dup skew when it returns from a
 * black-slate outage, so its audio re-syncs (not delayed by the stale dup total). On by
 * default; PTV_NO_REANCHOR=1 keeps the stale skew across outages (for A/B comparison). */
static int     g_reanchor = 1;
/* Multiview house-clock CONTENT CLAMP (DEFAULT OFF — superseded by the discontinuity absorber
 * below). It held an ahead-of-clock frame one tick at a time; on real feeds whose source PTS is
 * jittery that holds repeatedly = visible STUTTER (regressed in0/in2 on the box). Kept as an
 * opt-in experiment: PTV_MV_CLAMP=1 enables it. The real cause (a source PTS discontinuity, not
 * a content gap) is fixed at the source by g_discont, which keeps video smooth. */
static int     g_mv_clamp = 0;
/* Per-input source PTS-discontinuity absorber (THE multiview audio-late fix). Real live feeds
 * (the 1080i50 Grid SRT inputs) throw a forward PTS jump of a few hundred ms at the join while
 * FRAMES stay continuous (one per tick, buffer full) — a timestamp glitch, not lost frames. Left
 * raw it shifts that slot's content→output mapping so the cell's video leaps ahead of its
 * (continuous) audio = per-slot "audio behind video". This absorbs it the SAME way demux_unwrap
 * absorbs the 33-bit wrap (a per-stream offset that keeps the effective timeline continuous),
 * applied to video+audio+copy uniformly so they stay aligned and video stays smooth. On by
 * default; PTV_NO_DISCONT=1 reverts; PTV_DISCONT_MS sets the detect threshold (forward DTS jump
 * beyond it = a glitch to absorb). */
static int     g_discont = 1;
static int     g_discont_ms = 80;
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
static int     g_audio_follow = 1;
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
 * PTV_NO_H0_REANCHOR=1 reverts; PTV_H0_REANCHOR_MS sets the trigger (default 120ms). */
static int     g_h0_reanchor = 1;
static int     g_h0_reanchor_ms = 120;
/* h0-AT-DISPLAY (multiview): anchor each slot's h0 to the first frame the COMPOSITOR actually DISPLAYS,
 * not the first frame the decoder produces. Under a deep startup prime the first-decoded frame is an
 * earlier/different content than the first-displayed one, so the old decode-thread anchor left the
 * displayed video leaping ahead of h0 at tick 0 → P2 re-anchored h0 forward → the transcoded audio
 * banked (monotonic guard) and a copied audio track's DTS jumped backward (clamp/freeze, historically
 * an EINVAL no-data outage). Anchoring at first display makes sk=0 from the start so P2 never fires —
 * no bank, no clamp, no outage. MULTIVIEW ONLY; single-input keeps the decode-thread anchor (BYTE-
 * IDENTICAL). PTV_NO_H0_AT_DISPLAY=1 reverts to the decode-thread anchor (A/B). */
static int     g_h0_at_display = 1;
/* A/V PLL redesign — Phase A READ-ONLY measurement probe (analysis/ptvencoder-avsync-pll-redesign-plan.md).
 * Off by default; PTV_AVSYNC_PROBE=1 enables the [PTV-AVSYNC2] per-track real A/V offset measurement
 * (out_v(C) − out_a(C), content-paired, video_lag/audio_lag split). Measures only — no actuator. */
static int     g_avsync_probe = 0;
static int     g_atrace = 0;       /* PTV_ATRACE: temp per-audio-frame startup trace (opts/applied/guard) to localize the bank */
/* Multiview audio-follow ACTUATOR (P1) — a per-slot two-mode controller for glitch-free A/V tracking.
 * The v0.6.2/0.6.3 audio-follow corrected the per-slot lag only with whole-frame drop/pad fired on a
 * 40ms threshold: it tracked the slow per-slot drift (source-vs-house-clock, e.g. +360ms over 35min)
 * in discrete ~40ms steps — each step a momentary A/V hop ("sometimes off, then OK again"), and it
 * left a ±frame residual ("audio slightly behind"). P1 keeps the fast discrete drop/pad ONLY to
 * ACQUIRE a large gap (startup ramp / big jump the smooth rate can't catch), and otherwise TRACKS the
 * residual + slow drift with a SMOOTH, rate-limited (≤g_af_rate_us/s, imperceptible) sub-sample PTS
 * nudge — no silence inserts, no content drops, no hops. Reuses the legacy 0007 PLL's rate-limited
 * actuator idea, per slot, driven by the directly-measured lag (no jump_comp). MULTIVIEW ONLY;
 * PTV_AF_NO_PLL=1 reverts to pure discrete; PTV_AF_ACQUIRE_MS / PTV_AF_RATE_MS_S tune. */
static int     g_af_pll = 1;
static int     g_af_acquire_us = 100000;   /* gap above this → discrete drop/pad; at/below → smooth nudge */
static int     g_af_rate_us = 10000;       /* smooth follow/nudge rate ceiling, us per second. B2 (2026-06-21):
                                            * raised 2000→10000 so B1's content-anchored offset tracks a
                                            * source-slowness dup ramp in vlag (~6–8 ms/s) in near-real-time
                                            * instead of lagging ~100 s at 2 ms/s. 10 ms/s ≈ 1% (under the ~2%
                                            * audible budget) and only engages transiently while converging —
                                            * steady-state step=gap is tiny. PTV_AF_RATE_MS_S overrides. */
/* Phase B (B1) — CONTENT-ANCHORED multiview audio. The pre-B1 multiview audio-follow emitted on a
 * FREE-RUNNING sample counter (af_next_pts), which faithfully banked aresample=async's STARTUP
 * over-production into a permanent per-slot audio-late offset (Phase A root cause: alag +400..+1252ms,
 * the dominant desync term). B1 instead anchors the output to opts (async's self-correcting content
 * target — exactly what single-input does, which is why single-input reads offset≈0) plus a smooth
 * rate-limited offset that tracks the compositor's per-slot lag, so the audio follows the video DISPLAY
 * without banking over-production. Default ON; PTV_AF_NO_ANCHOR=1 reverts to the free-counter path (A/B). */
static int     g_af_anchor = 1;
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
static int     g_avsync_pll = 0;
static int     g_pll_ema_shift = 5;          /* EMA smoothing of the measured offset (τ≈0.7s @ ~47 afps) */
static int64_t g_pll_tau_us = 5000000;       /* integral track time-constant (us): step = ema*frame_us/τ */
static int     g_pll_acquire_us = 40000;     /* |ema| above this = "large" → ACQUIRE one-shot; else TRACK. 40ms ≈ 2 audio frames: shrinks the dead band [gate 25ms, threshold] so a stable sub-100ms residual (TRACK is guard-limited on jittery sources) is snapped in by a whole-frame acquire instead of stranded. The flatness debounce (threshold/4 = 10ms) still rejects jitter. PTV_PLL_ACQUIRE_MS overrides. */
static int     g_pll_acquire_n = 32;         /* debounce: N stable (large AND flat) readings before acquire; also the refractory */
static int64_t g_pll_startup_us = 5000000;   /* startup window: acquire may fire ≤k times; also the mid-run re-acquire arm window */
static int     g_pll_acquire_k = 4;          /* max startup acquires (a stepwise-forming bank can need a top-up) */
/* demux→decode video queue depth. Raised 48→256 (Phase B #1): at startup the decoder's init window
 * (finding a keyframe, building up) produces nothing while the realtime source keeps filling video_q,
 * so the old 48-deep queue overflowed and dropped ~30 frames → a content GAP → the position-anchored
 * composite video LEAPS to the newest content → audio left behind = per-slot picture-ahead. A deeper
 * queue ABSORBS the one-time init backlog (the decoder then drains it faster than realtime and catches
 * up — multicast/live is realtime steady-state, so video_q sits near-empty after). drop-newest remains
 * the backstop for genuine SUSTAINED overload. PTV_VIDEOQ overrides. */
static int     g_videoq = 256;
/* Discard video packets the demuxer flags AV_PKT_FLAG_CORRUPT (= -fflags +discardcorrupt), at the demux
 * before they reach the decoder, and COUNT them (DemuxArgs.vcorrupt) so frame loss is visible in stats.
 * A corrupt frame, like a dropped one, becomes a content GAP that the position-anchored composite video
 * leaps across → desync; discarding early + counting makes it observable. Default ON; PTV_KEEP_CORRUPT=1
 * keeps corrupt packets (lets the decoder try to use them — the prior behavior). */
static int     g_discardcorrupt = 1;
/* Muxed-packet stats counters. Written by N mux threads (6-rung ABR) -> atomic to avoid a
 * data race / lost updates in the bitrate=/size= stat line. Stats-only; not on any hot path. */
static _Atomic int64_t g_muxed;
/* ffmpeg-style progress line (frame=/fps=/bitrate=/speed=); on unless -nostats. */
static int     g_stats = 1;
static _Atomic int64_t g_muxed_bytes;
/* PTV_DIAG: compositor publishes its current VIDEO output time (us) so the audio probe can
 * log a synchronized per-track audio-minus-video offset. Temporary diagnostic. */
static _Atomic int64_t g_vout_us;
/* -stats_period: interval (us) between progress lines. Default 1s; raise for
 * production (e.g. -stats_period 10 -> every 10s) to keep logs quiet. */
static int64_t g_stats_period_us = 1000000;
/* PTV_SLOW_US: inject N us of extra per-emitted-frame consumer cost, to model a
 * slow/blocking encoder on a box that has none. Stress knob, gated. */
static int     g_slow;

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

void show_help_default(const char *opt, const char *arg)
{
    av_log(NULL, AV_LOG_INFO,
        "usage: ptvencoder [options] -i <input> [-i <input> ...] [out-opts] <output> [[out-opts] <output> ...]\n"
        "\n"
        "  inputs:\n"
        "    -i <url>            input (file or udp://...). 1 input = single transcode;\n"
        "                        2 or 4 inputs = multiview mosaic (requires -filter_complex).\n"
        "  video:\n"
        "    -vf <chain>         libavfilter chain for the (single) input, e.g. \"bwdif,scale=1280:720\"\n"
        "    -filter_complex <g> mosaic graph for multiview ([0:v][1:v]hstack...[vN]) and/or ABR split\n"
        "    -map [vN]|K:v       select this output's video (filter label or input stream)\n"
        "    -c:v <name>         video encoder (default: h264_videotoolbox, fallback mpeg2video)\n"
        "    -b:v / -r           video bitrate / output (house-clock) frame rate; default rate = source\n"
        "  audio (per output stream N):\n"
        "    -map K:a:n          add a transcoded audio track from input K\n"
        "    -af / -filter:a:N   audio filtergraph (default aresample=async=1000)\n"
        "    -c:a:N -b:a:N -ac:a:N   encoder / bitrate / channels per track\n"
        "    -an                 no audio (suppresses auto-selected audio when no -map is given)\n"
        "  passthrough / output:\n"
        "    -map K:s|K:d -c copy   copy subtitle / data (incl. SCTE-35) streams through\n"
        "    -metadata:s:<t>:N k=v   -disposition:<t>:N flags   per-stream metadata/disposition\n"
        "    -f <mux>            output format (default: guessed; mpegts for udp://...)\n"
        "    -stats_period <s>   progress-line interval (default 1)\n"
        "    -version, -h\n"
        "\n"
        "  Pacing is automatic: live (wall-clock) for net inputs, media-clock for files.\n");
}

/* Free function for AVThreadMessageQueue elements (AVPacket* / AVFrame*; a NULL
 * element is an end-of-stream marker on mux_q). */
static void free_pkt_msg(void *msg)   { av_packet_free(msg); }
static void free_frame_msg(void *msg) { av_frame_free(msg); }

/* Drain an encoder, pushing packets to the mux queue. frame=NULL flushes. */
static int encode_push(AVThreadMessageQueue *mux_q, AVCodecContext *enc,
                       AVStream *ost, AVFrame *frame)
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
            return 0;
        }
        if (ret < 0) {
            av_packet_free(&pkt);
            return ret;
        }
        av_packet_rescale_ts(pkt, enc->time_base, ost->time_base);
        pkt->stream_index = ost->index;
        ret = av_thread_message_queue_send(mux_q, &pkt, 0);   /* blocking */
        if (ret < 0) {
            av_packet_free(&pkt);
            return ret;                                       /* mux gone */
        }
    }
}

/* ---- video: decode (free-run) + output (master clock, sample-and-hold) ---- */

#define PTV_MAX_RUNG 8
#define PTV_MAX_AUDIO 8    /* max transcoded audio output tracks (multi-language, multiview slots) */
#define PTV_AQ_PREROLL 256 /* per-track pre-h0 audio buffer (frames): preserve a slot's audio head
                            * while its video decodes its first frame (sets h0), instead of dropping
                            * it (which made that slot's audio start ~h0-acquire-delay late). Bounded
                            * ring (drop-oldest) so a never-arriving video can't grow it unboundedly. */
#define PTV_MAX_INPUT 4    /* max composited inputs (multiview): 1 / 2 / 4 */
#define PTV_MV_SKEW_CAP_US 250000   /* multiview per-slot audio skew cap (async budget) */

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

static void vring_put(VOutRing *r, int64_t src_us, int64_t out_us)
{
    pthread_mutex_lock(&r->lock);
    int i = (int)(r->n % PTV_VRING);
    r->src[i] = src_us; r->out[i] = out_us; r->n++;
    pthread_mutex_unlock(&r->lock);
}

/* nearest-by-content lookup: of all kept entries, return the out_v and matched src of the one whose
 * src is closest to want_src. 0 = found (ring non-empty), -1 = empty. */
static int vring_lookup(VOutRing *r, int64_t want_src, int64_t *out_v, int64_t *matched_src)
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
} DecodeCtx;

/* Per-rung output side: pop this rung's frame_q on the house clock, stamp the
 * content-anchored PTS, encode, hand to this rung's mux_q. One per output. */
typedef struct VideoCtx {
    AVThreadMessageQueue *frame_q;   /* decode -> output  (AVFrame*)  */
    AVThreadMessageQueue *mux_q;     /* output -> mux     (AVPacket*) */
    AVRational       out_tb;         /* time_base of frames at this rung's sink (or ist_tb) */
    int64_t         *h0;             /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    int64_t         *house_skew;     /* master publishes house-vs-content skew (us) here */
    VOutRing        *vring;          /* A/V probe: single-input video output ring (PTV_AVSYNC_PROBE) */
    AVCodecContext  *venc;
    AVStream        *ost;
    int64_t          tick_dur_us;
    int              live;
    int              passthrough;    /* multiview: compositor already paced+stamped; encode 1:1 */
    int              is_master;      /* only the master rung prints stats/diag */
    /* shared decode counters + queue, for the master's diag line */
    AVThreadMessageQueue *dbg_video_q;
    int64_t         *dbg_dec_frames, *dbg_vcorrupt;
    int64_t         *dbg_vdrop, *dbg_pcorrupt;       /* single-input stats: demux video_q drops + corrupt-pkt count */
    /* counters */
    int64_t          framedrop, emitted, dup;
    /* watchdog */
    int64_t          last_emit_us;
    volatile int     output_done;
    int              stalled;
} VideoCtx;

/* hand a frame to one rung's jitter buffer; drop-oldest in live so a stalled
 * encoder never blocks the shared decode (same behaviour as before, per rung). */
static void push_frame_q(AVThreadMessageQueue *q, int live, int64_t *framedrop, AVFrame *out)
{
    if (!live) {                                  /* offline: lossless back-pressure */
        if (av_thread_message_queue_send(q, &out, 0) < 0)
            av_frame_free(&out);
        return;
    }
    int ret = av_thread_message_queue_send(q, &out, AV_THREAD_MESSAGE_NONBLOCK);
    if (ret == AVERROR(EAGAIN)) {                    /* full -> drop oldest, keep newest */
        AVFrame *old;
        if (av_thread_message_queue_recv(q, &old, AV_THREAD_MESSAGE_NONBLOCK) >= 0)
            av_frame_free(&old);
        if (av_thread_message_queue_send(q, &out, AV_THREAD_MESSAGE_NONBLOCK) < 0) {
            av_frame_free(&out);
            (*framedrop)++;
        }
    } else if (ret < 0) {
        av_frame_free(&out);
    }
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
 * content-PTS A/V anchoring still holds across the filter. */
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
            if (out) push_frame_q(d->frame_q[i], d->live, &d->framedrop[i], out);
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
            if (out) { av_frame_move_ref(out, filt); push_frame_q(d->frame_q[i], d->live, &d->framedrop[i], out); }
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

    if (!frame || !filt)
        goto done;
    for (;;) {
        ret = av_thread_message_queue_recv(d->video_q, &pkt, 0);
        if (ret < 0) break;
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
                    const char *hd = getenv("PTV_H0_DELAY_MS");   /* TEST ONLY: simulate slow first-frame acquire */
                    if (hd && atoi(hd) > 0) av_usleep((unsigned)atoi(hd) * 1000);
                    pthread_mutex_lock(d->h0_lock);
                    if (*d->h0 == AV_NOPTS_VALUE) *d->h0 = av_rescale_q(ts, d->ist_tb, AV_TIME_BASE_Q);
                    pthread_mutex_unlock(d->h0_lock);
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
                if (out) { av_frame_move_ref(out, filt); push_frame_q(d->frame_q[i], d->live, &d->framedrop[i], out); }
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

static void *output_thread(void *arg)
{
    VideoCtx *v = arg;
    AVFrame *held = av_frame_alloc();
    AVFrame *f;
    int have = 0, ret = 0;
    int64_t tick = 0, wall0 = 0, last_vpts = -1;
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
            ret = encode_push(v->mux_q, v->venc, v->ost, f);
            v->emitted++; v->last_emit_us = av_gettime_relative();
            av_frame_free(&f);
            if (ret < 0) break;
        }
        encode_push(v->mux_q, v->venc, v->ost, NULL);
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
            ret = encode_push(v->mux_q, v->venc, v->ost, f);
            v->emitted++; v->last_emit_us = av_gettime_relative();
            av_frame_free(&f);
            if (ret < 0) break;
        }
        encode_push(v->mux_q, v->venc, v->ost, NULL);
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
        const char *pe = getenv("PTV_PREROLL_MS");
        int preroll_ms = pe ? atoi(pe) : 350;
        int n_prime = (preroll_ms > 0 && v->tick_dur_us > 0)
                          ? (int)((int64_t)preroll_ms * 1000 / v->tick_dur_us) : 0;
        int primed;
        if (n_prime > PTV_FRAME_QDEPTH - 8) n_prime = PTV_FRAME_QDEPTH - 8;
        if (n_prime < 0) n_prime = 0;
        primed = (n_prime == 0);

    for (;;) {
        int fresh = 0;
        ret = av_thread_message_queue_recv(v->frame_q, &f, AV_THREAD_MESSAGE_NONBLOCK);
        if (ret >= 0) {
            av_frame_unref(held); av_frame_move_ref(held, f); av_frame_free(&f);
            have = 1; fresh = 1; held_src_pts = held->pts;   /* capture before emit overwrites it */
        } else if (ret == AVERROR_EOF) {
            break;                                  /* decode finished, queue drained */
        }
        if (!have) { av_usleep(2000); continue; }   /* await first frame (no startup dups) */

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
            int64_t target = wall0 + tick * v->tick_dur_us;
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
        }
        /* Stamp output PTS from the frame's SOURCE time on the shared house
         * anchor (h0) — the SAME mapping audio uses — so dropped/duped frames
         * never skew the timeline and A/V stays locked. (A pure tick counter
         * drifts by the number of startup/stall-dropped frames -> A/V skew.)
         * Pacing still rides the wall clock via `tick`; PTS rides content. */
        {
            int64_t vpts, content_vpts = -1;
            int64_t src_ts = held_src_pts;   /* ORIGINAL source pts (out_tb); survives dups */
            if (src_ts != AV_NOPTS_VALUE && *v->h0 != AV_NOPTS_VALUE) {
                int64_t house_us = av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q) - *v->h0;
                if (house_us < 0) house_us = 0;
                content_vpts = (house_us + v->tick_dur_us / 2) / v->tick_dur_us;
                vpts = content_vpts;
            } else {
                vpts = last_vpts + 1;
            }
            if (vpts <= last_vpts) vpts = last_vpts + 1;   /* monotonic CFR; dup -> next slot */
            held->pts = vpts; held->pkt_dts = AV_NOPTS_VALUE; held->duration = 0;
            last_vpts = vpts;
            /* Publish how far the house clock now runs AHEAD of source content
             * (vpts - content_vpts, in ticks). Each dup bumps vpts past content via
             * the monotonic guard, so this grows by one tick per dup and persists.
             * The audio path adds it so audio rides the same house clock instead of
             * staying source-locked (which is what drifts ~40ms per dup). */
            if (v->is_master && v->house_skew && content_vpts >= 0)
                *v->house_skew = (vpts - content_vpts) * v->tick_dur_us;
            /* A/V probe (read-only): record this distinct content's first-display output time so the
             * audio drain can pair against it (single-input master rung only; multiview → compositor). */
            if (v->vring && fresh && content_vpts >= 0)
                vring_put(v->vring, av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q), vpts * v->tick_dur_us);
        }
        ret = encode_push(v->mux_q, v->venc, v->ost, held);
        v->last_emit_us = av_gettime_relative();
        tick++; v->emitted++;
        if (!fresh) v->dup++;
        if (g_slow) av_usleep(g_slow);
        if (ret < 0) break;

        if (g_diag && v->is_master) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                av_log(NULL, AV_LOG_INFO,
                    "[PTV-DIAG] t=%.1fs dec=%"PRId64" vcorrupt=%"PRId64" emitted=%"PRId64
                    " muxed=%"PRId64" dup=%"PRId64" framedrop=%"PRId64" vq=%d frameq=%d muxq=%d\n",
                    (nowd - diag_t0) / 1000000.0, *v->dbg_dec_frames, *v->dbg_vcorrupt, v->emitted,
                    g_muxed, v->dup, v->framedrop,
                    av_thread_message_queue_nb_elems(v->dbg_video_q),
                    av_thread_message_queue_nb_elems(v->frame_q),
                    av_thread_message_queue_nb_elems(v->mux_q));
                diag_last = nowd;
            }
        }

        if (g_stats && v->is_master) {          /* ffmpeg-style progress line */
            int64_t nows = av_gettime_relative();
            if (nows - stat_last >= g_stats_period_us) {
                double dt    = (nows - stat_last) / 1000000.0;
                double fps   = (v->emitted - stat_prev) / (dt > 0 ? dt : 1);
                double secs  = v->emitted * v->tick_dur_us / 1000000.0;   /* CFR output time */
                double wall  = (nows - wall0) / 1000000.0;
                double speed = wall > 0 ? secs / wall : 0;
                double kbps  = secs > 0 ? g_muxed_bytes * 8.0 / secs / 1000.0 : 0;
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                int64_t qd = v->dbg_vdrop ? *v->dbg_vdrop : 0;       /* video_q overflow drops */
                int64_t cr = (v->dbg_pcorrupt ? *v->dbg_pcorrupt : 0) + (v->dbg_vcorrupt ? *v->dbg_vcorrupt : 0);  /* corrupt: demux + decode */
                av_log(NULL, AV_LOG_INFO,
                    "frame=%6"PRId64" fps=%3.0f size=%8"PRId64"KiB time=%02d:%02d:%05.2f "
                    "bitrate=%7.1fkbits/s dup=%"PRId64" drop=%"PRId64" qdrop=%"PRId64" corrupt=%"PRId64" speed=%4.2fx\n",
                    v->emitted, fps, g_muxed_bytes / 1024, hh, mm, ss, kbps,
                    v->dup, v->framedrop, qd, cr, speed);
                stat_last = nows; stat_prev = v->emitted;
            }
        }
    }
    encode_push(v->mux_q, v->venc, v->ost, NULL);
    }
done:
    av_frame_free(&held);
    v->output_done = 1;
    { AVPacket *eof = NULL; av_thread_message_queue_send(v->mux_q, &eof, 0); }
    return NULL;
}

/* watchdog: flag (does not yet auto-recover) a stalled output/encoder so a hung
 * NVENC session is visible. A hung in-process session can't be safely torn down
 * from another thread, so auto-reinit needs process isolation — a follow-up. */
static void *watchdog_thread(void *arg)
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

/* ---- audio path (decode -> resample 48k stereo -> AAC -> mux) ---- */

typedef struct AudioState {
    AVThreadMessageQueue *audio_q;
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    AVStream             *ost[PTV_MAX_RUNG];     /* audio out stream in each muxer */
    int              n_out;
    AVCodecContext  *dec;
    AVCodecContext  *enc[PTV_MAX_RUNG];          /* one AAC encoder per rung (per-rung -b:a) */
    AVRational       ist_tb;
    SwrContext      *swr;                         /* no -af: plain resample to 48k stereo */
    AVFilterGraph   *afg;                         /* -af present: abuffer -> chain -> abuffersink */
    AVFilterContext *afsrc, *afsink;
    int              use_fg;
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
    int              pll_seed;                        /* pll_ema seeded at the first valid measurement */
    int              pll_dbnc;                        /* stability-debounce: consecutive large-AND-flat readings */
    int64_t          pll_dbnc_ref;                    /* ema value when the debounce window started (flatness reference) */
    int              pll_refractory;                  /* frames remaining before acquire may re-arm (bumpless-credit backstop) */
    int              pll_acq_count;                   /* acquires fired this run (startup-k cap + gate assertion) */
    int64_t          pll_t0_us;                       /* first-measurement wallclock (startup window) */
    int64_t          pll_arm_until_us;                /* mid-run: acquire armed (one-shot) until this wallclock, set on a disturbance epoch advance */
    int64_t          pll_disturb_seen;                /* last disturbance epoch observed */
    int              pll_drop, pll_pad;               /* pending one-shot acquire: frames to drop (advance) / pad (delay), on the B1 base */
    int64_t          pll_guard_fires;                 /* monotonic-guard activations (windup observability) */
    _Atomic int_least64_t *disturb_epoch;             /* compositor/demux publish this input's disturbance epoch (slate-return / discont) */
} AudioState;

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
            if (av_thread_message_queue_send(a->mux_q[i], &pkt, 0) < 0)   /* blocking */
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
                    if (!a->pll_seed) { a->pll_ema = off; a->pll_dbnc_ref = off; a->pll_seed = 1; }
                    else a->pll_ema += (off - a->pll_ema) >> g_pll_ema_shift;   /* smooth ±vlag jitter (N6: toward −∞, sub-ms, negligible) */
                    /* N7: stability-debounce — fire only when the EMA is LARGE *and* FLAT, so Δ is sized
                     * to the FROZEN bank, not one still forming. */
                    if (FFABS(a->pll_ema) > (int64_t)g_pll_acquire_us &&
                        FFABS(a->pll_ema - a->pll_dbnc_ref) < (int64_t)g_pll_acquire_us / 4)
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
                                  FFABS(a->pll_ema) > (int64_t)g_pll_acquire_us &&
                                  a->pll_dbnc >= g_pll_acquire_n;
                    if (may_acq) {
                        int64_t half = frame_us / 2;                        /* round to NEAREST whole frame (half away from 0) */
                        int64_t dq = ((a->pll_ema + (a->pll_ema < 0 ? -half : half)) / frame_us) * frame_us; /* → residual ≤ ½ frame (≈11ms), vs ≤1 frame for truncation */
                        if (dq != 0) {
                            a->af_applied_us += dq;          /* UNIFORM both directions — cancels the content jump in want */
                            a->pll_ema       -= dq;          /* N1: bumpless credit → no re-fire on the next reading */
                            if (dq < 0) { a->pll_drop = (int)(-dq / frame_us); a->af_acq_drop_us += -dq; }  /* advance: drop content */
                            else        { a->pll_pad  = (int)( dq / frame_us); a->af_acq_pad_us  +=  dq; }   /* delay: pad silence */
                            a->pll_acq_count++;
                            if (g_diag)
                                av_log(NULL, AV_LOG_INFO, "[PTV-PLL] a%d(in%d) ACQUIRE %s %"PRId64"ms (ema→%"PRId64"ms applied=%"PRId64"ms #%d)\n",
                                       a->dbg_k, a->dbg_in, dq < 0 ? "drop" : "pad", FFABS(dq) / 1000,
                                       a->pll_ema / 1000, a->af_applied_us / 1000, a->pll_acq_count);
                        }
                        a->pll_refractory = g_pll_acquire_n; a->pll_dbnc = 0; a->pll_dbnc_ref = a->pll_ema;
                        if (a->pll_drop > 0) { a->pll_drop--; av_frame_unref(filt); continue; }  /* drop the current frame too */
                    } else {
                        /* TRACK — type-1 integral trim, rate-clamped, NO dead zone. Conditional-integration
                         * anti-windup: don't integrate while the monotonic guard would saturate (N3). */
                        int64_t pre = opts + av_rescale(a->af_applied_us, a->out_rate, 1000000);
                        if (!(a->af_out_set && pre < a->af_last_out + nb)) {
                            int64_t step = a->pll_ema * frame_us / g_pll_tau_us;        /* integral: move ema/τ per frame */
                            int64_t lim  = (int64_t)g_af_rate_us * nb / a->out_rate;    /* rate clamp (us) */
                            if (lim < 1) lim = 1;
                            if (step >  lim) step =  lim;
                            if (step < -lim) step = -lim;
                            a->af_applied_us += step;
                        }
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
                int64_t want_raw = want;          /* ATRACE: value before the monotonic guard */
                /* MONOTONIC GUARD (B1-fix, v0.6.8) — opts is the async/buffersink output pts; it steps
                 * BACKWARD when h0 is re-anchored forward (P2: opts = buffersink − h0_samp, larger h0 →
                 * smaller opts) or at a source PTS discontinuity. The pre-B1 free counter was monotonic
                 * by construction; content-anchoring lost that → backward out → libfdk_aac "Queue input
                 * is backward in time" + mpegts non-monotonic-DTS → that audio stream stalls and the
                 * interleaver wedges (no output — box-observed). Keep out_a monotonic + frame-spaced; on
                 * a backward step it advances at nb (dense, like the old counter) until opts recovers. */
                if (a->af_out_set && want < a->af_last_out + nb) want = a->af_last_out + nb;
                if (g_atrace && a->out_frames < 220)
                    av_log(NULL, AV_LOG_INFO,
                        "[PTV-ATRACE] a%d f=%"PRId64" opts=%+"PRId64"ms applied=%+"PRId64"ms want_raw=%+"PRId64"ms want=%+"PRId64"ms guard=%+"PRId64"ms srcabs=%"PRId64"ms\n",
                        a->dbg_k, a->out_frames, opts * 1000 / a->out_rate, a->af_applied_us / 1000,
                        want_raw * 1000 / a->out_rate, want * 1000 / a->out_rate,
                        (want - want_raw) * 1000 / a->out_rate, src_abs_us / 1000);
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
                 * audio-follow feeds content-aligned input (no injection) → use src_abs_us directly. */
                if (!(a->multiview && g_audio_follow) && a->house_skew)
                    content -= *a->house_skew;
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
            /* [PTV-AVSYNC] production A/V-sync status — per slot, on the -stats_period cadence (obeys
             * -nostats / -stats_period like the progress line, NOT PTV_DIAG). Reports what the
             * per-slot actuator is correcting (lag = the cell's measured video-vs-house-clock offset),
             * how much it has applied (applied = the audio retiming in effect: smooth nudge + the
             * cumulative discrete acquire), the residual it has not yet absorbed (err = lag − applied;
             * ~0 = tracking, large = catching up after a jump), and total work done (acquire drop/pad).
             * Multiview audio-follow only. (Absolute lip-sync is the box eye-check; this is the
             * actuator's own tracking state — the honest, always-available health signal.) */
            if (g_stats) {
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
                            "pll[ema=%+"PRId64"ms applied=%+"PRId64"ms acq=%d guard=%"PRId64" drop=%"PRId64"ms pad=%"PRId64"ms]"
                            "  [offset<0 = audio late]\n",
                            a->dbg_k, a->dbg_in, lshead / 1000, m, a->av_vlag_us / 1000, a->av_alag_us / 1000,
                            a->pll_ema / 1000, a->af_applied_us / 1000, a->pll_acq_count, a->pll_guard_fires,
                            a->af_acq_drop_us / 1000, a->af_acq_pad_us / 1000);
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

/* Feed one h0-anchored decoded audio frame into the -af graph (or swr fallback). */
static int audio_feed(AudioState *a, AVFrame *frame)
{
    uint8_t **out = NULL;
    int out_max, got, ret = 0;
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
        a->dbg_last_src = frame->best_effort_timestamp;   /* probe: latest fed source pts */
    if (a->use_fg) {
        /* -af: feed the graph; aresample async + loudness emit fixed-size frames
         * whose PTS already carries async's A/V correction — drain them straight
         * to the encoders. Common-mode A/V lock: add the video's house-vs-content
         * skew so aresample=async targets the HOUSE clock instead of the source. */
        /* Single-input (and PTV_NO_AUDIO_FOLLOW): nudge the graph input pts by house_skew so
         * aresample=async targets the house clock. Multiview audio-follow does NOT do this — it
         * feeds content-aligned input and applies the offset deterministically in the drain. */
        if (g_avlock && a->house_skew && !(a->multiview && g_audio_follow) && frame->pts != AV_NOPTS_VALUE) {
            int64_t sk = *a->house_skew;
            if (sk) frame->pts += av_rescale_q(sk, AV_TIME_BASE_Q, a->ist_tb);
        }
        if ((ret = av_buffersrc_add_frame(a->afsrc, frame)) < 0)
            return ret;
        return audio_drain_fg(a);
    }
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
        if (house_us < 0) return 0;                  /* audio precedes video anchor: drop */
        a->next_pts = av_rescale(house_us, a->out_rate, 1000000);
        a->pts_set  = 1;
        a->dbg_first_src = ts;
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
                    if (a->aq_npending >= PTV_AQ_PREROLL) {       /* ring: drop oldest */
                        av_frame_free(&a->aq_pending[0]);
                        memmove(a->aq_pending, a->aq_pending + 1,
                                (PTV_AQ_PREROLL - 1) * sizeof(*a->aq_pending));
                        a->aq_npending = PTV_AQ_PREROLL - 1;
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
    if (ts != AV_NOPTS_VALUE) a->dbg_last_src = ts;   /* probe: latest source audio pts */
    return audio_feed(a, frame);
}

static void *audio_thread(void *arg)
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
        ret = avcodec_send_packet(a->dec, pkt);
        av_packet_free(&pkt);
        while (ret >= 0) {
            ret = avcodec_receive_frame(a->dec, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) goto done;
            ret = audio_push(a, frame);
            av_frame_unref(frame);
            if (ret < 0) goto done;
        }
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
static int build_audio_filter(AudioState *a, AVCodecContext *adec, AVRational tb,
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

/* ---- demux + mux ---- */

#define PTV_MAX_PASS 16
typedef struct PassStream {
    int        input;                 /* source input index (multiview); 0 single-input */
    int        in_index;              /* input stream index being copied 1:1 (within that input) */
    AVStream  *ost[PTV_MAX_RUNG];     /* output stream in each muxer (fan-out) */
    AVRational in_tb;                 /* input/output time_base (copy: identical) */
    int64_t    last_dts;              /* last emitted dts (monotonic guard; NOPTS until first) */
} PassStream;

typedef struct DemuxArgs {
    AVFormatContext      *ifmt;
    AVThreadMessageQueue *video_q;
    AVThreadMessageQueue *audio_q[PTV_MAX_AUDIO]; /* one per transcoded audio track */
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    int                   n_out;
    int                   vstream;
    int                   astream[PTV_MAX_AUDIO]; /* input stream feeding each audio_q */
    int                   n_audio;
    int                   drop;          /* non-blocking + drop on full (network input) */
    PassStream           *pass;          /* copy-passthrough: extra audio, subs, data */
    int                   n_pass;
    int64_t              *h0;             /* house origin (us); copy ts rebased onto it */
    pthread_mutex_t      *h0_lock;
    int64_t              *house_skew;     /* video's house-vs-content skew (us); copy rides it */
    _Atomic int_least64_t *disturb_epoch; /* B3: bump this input's disturbance epoch when the discont absorber fires */
    int64_t              *wrap_off;       /* per input stream: cumulative 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;      /* per input stream: last RAW ts seen (wrap detection) */
    int64_t               vpkt, apkt, ppkt, vdrop, adrop, pdrop;
    int64_t               vcorrupt;       /* video packets flagged AV_PKT_FLAG_CORRUPT (discarded if g_discardcorrupt) */
} DemuxArgs;

typedef struct MuxArgs {
    AVFormatContext      *ofmt;
    AVThreadMessageQueue *mux_q;
    int                   n_producers;
    int                   err;
} MuxArgs;

static int demux_send(AVThreadMessageQueue *q, AVPacket *pkt, int drop, int64_t *drops)
{
    int ret = av_thread_message_queue_send(q, &pkt, drop ? AV_THREAD_MESSAGE_NONBLOCK : 0);
    if (drop && ret == AVERROR(EAGAIN)) {   /* full -> drop */
        av_packet_free(&pkt);
        (*drops)++;
        return 0;
    }
    if (ret < 0)
        av_packet_free(&pkt);               /* queue closed */
    return ret;
}

/* Copy-passthrough: route an input packet we don't transcode (extra audio, DVB
 * subtitle, data/SCTE-35) straight to the muxer, rebased onto the same h0 house
 * timeline the encoded streams use so everything stays in sync. Packets that
 * precede the anchor are dropped (exactly like audio_push). */
static int demux_pass(DemuxArgs *d, AVPacket *out)
{
    int pi, i;
    for (pi = 0; pi < d->n_pass; pi++) {
        int64_t h0, h0_tb, ref;
        if (out->stream_index != d->pass[pi].in_index)
            continue;
        pthread_mutex_lock(d->h0_lock); h0 = *d->h0; pthread_mutex_unlock(d->h0_lock);
        if (h0 == AV_NOPTS_VALUE) { av_packet_free(&out); return 0; }  /* video not anchored yet */
        h0_tb = av_rescale_q(h0, AV_TIME_BASE_Q, d->pass[pi].in_tb);
        /* Ride the house clock: subtract h0, then ADD the video's house-vs-content
         * skew so copied streams (AC-3, subs, data) stay aligned with the dup-shifted
         * video instead of source-locked. Copied audio can't be resampled, so this is
         * a step (the skew grows in ~40ms tick increments) -> a small periodic A/V hop
         * on dense audio; sparse subs/data ride it invisibly. (Smooth would need rate-
         * discipline so dups -> 0; see option 2.) */
        if (g_avlock && d->house_skew)
            h0_tb -= av_rescale_q(*d->house_skew, AV_TIME_BASE_Q, d->pass[pi].in_tb);
        if (out->pts != AV_NOPTS_VALUE) out->pts -= h0_tb;
        if (out->dts != AV_NOPTS_VALUE) out->dts -= h0_tb;
        ref = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
        if (ref != AV_NOPTS_VALUE && ref < 0) { av_packet_free(&out); return 0; }  /* precedes anchor */
        /* Monotonic-DTS guard for the copy path. The house-skew rebase above (and
         * source quirks / wrap edges) can nudge a copied stream's dts backward
         * between packets; the mpegts muxer rejects a backward dts with EINVAL and
         * the rung dies — this froze channels under the 50fps field-rate dup-storm,
         * and a rare dup-induced skew dip could trip it even at the right rate.
         * Clamp strictly increasing per copied stream, shifting pts by the same
         * amount so pts>=dts still holds. (On-wire SCTE-35 splice timing rides
         * pts_adjustment, not container dts, so a sub-ms nudge is invisible there.) */
        if (out->dts != AV_NOPTS_VALUE) {
            if (d->pass[pi].last_dts != AV_NOPTS_VALUE && out->dts <= d->pass[pi].last_dts) {
                int64_t bump = d->pass[pi].last_dts + 1 - out->dts;
                out->dts += bump;
                if (out->pts != AV_NOPTS_VALUE) out->pts += bump;
            }
            d->pass[pi].last_dts = out->dts;
        }
        out->pos = -1;
        d->ppkt++;
        for (i = 0; i < d->n_out; i++) {            /* fan the copy out to every muxer */
            AVPacket *c = av_packet_clone(out);
            if (!c) continue;
            c->stream_index = d->pass[pi].ost[i]->index;   /* ts already rebased onto h0 */
            demux_send(d->mux_q[i], c, d->drop, &d->pdrop);
        }
        av_packet_free(&out);
        return 0;
    }
    av_packet_free(&out);
    return 0;
}

/* Unwrap a packet's MPEG-TS PTS/DTS into a monotonic, extended timeline.
 * The source counter rolls every 2^pts_wrap_bits ticks (33 bits = ~26.5h at
 * 90kHz). The house clock makes that roll invisible to the re-encoded video, but
 * the copy-passthrough streams carry the source timestamps through: at the roll a
 * copied stream's DTS leaps backward a full cycle, which the muxer rejects as
 * non-monotonic — fatal for audio (mux.c exempts only SUBTITLE/DATA), so every
 * rung dies and the pipeline wedges. The roll also collapses the video
 * house-vs-content skew. We run the input with correct_ts_overflow=0 (raw 33-bit
 * in, predictable — libav's own extension is inconsistent across the B-frame
 * reorder boundary, which is what produced the leap) and add a per-stream
 * multiple-of-2^bits offset so every downstream consumer sees one continuous
 * timeline. Keyed on DTS (decode order, monotonic); the SAME offset is added to
 * PTS so A/V copy stays aligned. mpegtsenc masks back to 33 bits on the wire. */
static void demux_unwrap(DemuxArgs *d, AVPacket *pkt)
{
    AVStream *st = d->ifmt->streams[pkt->stream_index];
    int bits = st->pts_wrap_bits;
    int64_t mask, half, raw, off;

    if (bits <= 0 || bits >= 63)              /* only meaningful for a real TS wrap */
        return;
    mask = 1LL << bits;
    half = mask >> 1;
    raw  = pkt->dts != AV_NOPTS_VALUE ? pkt->dts : pkt->pts;   /* DTS = decode order = monotonic (B-frame-safe) */
    if (raw != AV_NOPTS_VALUE) {
        int64_t last = d->wrap_last[pkt->stream_index];
        if (last != AV_NOPTS_VALUE) {
            int64_t delta = raw - last;
            int ct = st->codecpar->codec_type;
            if (delta < -half)      d->wrap_off[pkt->stream_index] += mask;  /* 33-bit wrap: rolled forward */
            else if (delta >  half) d->wrap_off[pkt->stream_index] -= mask;  /* late pre-roll pkt */
            else if (g_discont && (ct == AVMEDIA_TYPE_VIDEO || ct == AVMEDIA_TYPE_AUDIO)) {
                /* Source PTS discontinuity (forward DTS jump, NOT a wrap). Frames are continuous
                 * (one per tick) but the timestamp leaps; absorb the excess so the effective
                 * timeline stays continuous (re-base to last + one nominal frame), exactly like
                 * the wrap branch. Detected on DTS (monotonic), so B-frame PTS reorder never
                 * false-triggers. Keeps video/audio/copy aligned across the glitch.
                 * CONTINUOUS streams ONLY: sparse SUBTITLE/DATA (DVB-sub, SCTE-35) have natural
                 * multi-second inter-packet gaps that ALL exceed the threshold — absorbing them
                 * collapses the sparse timeline (subs drift out of sync / vanish; ad markers shift).
                 * The 33-bit wrap branches above still apply to every stream (copied AC-3/SCTE-35
                 * across the roll). */
                int64_t thresh = av_rescale(g_discont_ms, st->time_base.den, (int64_t)st->time_base.num * 1000);
                if (thresh > 0 && delta > thresh) {
                    int64_t nominal = pkt->duration > 0 ? pkt->duration : thresh / 4;
                    d->wrap_off[pkt->stream_index] -= (delta - nominal);
                    if (d->disturb_epoch)   /* B3: a real content discontinuity → arm the PLL's mid-run re-acquire */
                        atomic_fetch_add_explicit(d->disturb_epoch, 1, memory_order_relaxed);
                    if (g_diag)
                        av_log(NULL, AV_LOG_INFO, "[PTV-DISCONT] stream %d: +%"PRId64"ms PTS jump absorbed (re-based to continuous)\n",
                               pkt->stream_index, av_rescale_q(delta, st->time_base, (AVRational){1,1000}));
                }
            }
        }
        d->wrap_last[pkt->stream_index] = raw;
    }
    off = d->wrap_off[pkt->stream_index];
    if (off) {
        if (pkt->pts != AV_NOPTS_VALUE) pkt->pts += off;
        if (pkt->dts != AV_NOPTS_VALUE) pkt->dts += off;
    }
}

static void *demux_thread(void *arg)
{
    DemuxArgs *d = arg;
    AVPacket *pkt = av_packet_alloc();
    int64_t diag_last = av_gettime_relative();
    int ret = 0;

    if (!pkt)
        goto end;
    while (av_read_frame(d->ifmt, pkt) >= 0) {
        if (g_diag) {
            int64_t now = av_gettime_relative();
            if (now - diag_last >= 1000000) {
                av_log(NULL, AV_LOG_INFO, "[PTV-DIAG] demux vpkt=%"PRId64" vdrop=%"PRId64" vcorrupt=%"PRId64
                       " apkt=%"PRId64" adrop=%"PRId64" ppkt=%"PRId64" pdrop=%"PRId64"\n",
                       d->vpkt, d->vdrop, d->vcorrupt, d->apkt, d->adrop, d->ppkt, d->pdrop);
                diag_last = now;
            }
        }
        AVPacket *out = av_packet_alloc();
        if (!out) { av_packet_unref(pkt); break; }
        av_packet_move_ref(out, pkt);
        demux_unwrap(d, out);               /* 33-bit source wrap -> monotonic extended ts (ONCE) */
        if ((out->flags & AV_PKT_FLAG_CORRUPT) && g_discardcorrupt) {
            /* = -fflags +discardcorrupt, ALL streams — but COUNTED (video) so frame loss shows in stats
             * (libavformat's own flag discards silently, hiding the count). Drop before decode: a corrupt
             * frame, like a dropped one, becomes a content gap the position-anchored composite leaps
             * across → desync. PTV_KEEP_CORRUPT=1 disables (lets the decoder try to use them). */
            if (out->stream_index == d->vstream) d->vcorrupt++;
            av_packet_free(&out);
            continue;
        }
        if (out->stream_index == d->vstream) {
            d->vpkt++;
            ret = demux_send(d->video_q, out, d->drop, &d->vdrop);
        } else {
            /* Fan one source PID to every transcoded audio track on it (a clone
             * each), then hand the original to demux_pass (copy-passthrough; it
             * frees it, whether or not it's a copy stream). demux_unwrap ran ONCE
             * above, so every clone carries the same unwrapped ts (load-bearing:
             * never unwrap per-clone — the per-stream wrap state is stateful). */
            int k;
            for (k = 0; k < d->n_audio; k++) {
                AVPacket *c;
                if (d->astream[k] != out->stream_index) continue;
                if (!(c = av_packet_clone(out))) continue;
                d->apkt++;
                demux_send(d->audio_q[k], c, d->drop, &d->adrop);
            }
            ret = demux_pass(d, out);       /* copy fan + monotonic-DTS clamp; frees out */
        }
        if (ret < 0)
            break;
    }
end:
    /* producer done → signal CONSUMERS to drain then get EOF. recv() returns
     * err_recv (set_err_send is invisible to receivers), so this MUST be
     * set_err_recv or decode/audio block forever (the offline-EOF deadlock). */
    av_thread_message_queue_set_err_recv(d->video_q, AVERROR_EOF);
    { int k; for (k = 0; k < d->n_audio; k++)
        av_thread_message_queue_set_err_recv(d->audio_q[k], AVERROR_EOF); }
    if (d->n_pass > 0) {                     /* copy-passthrough producer EOF, per muxer */
        int i;
        for (i = 0; i < d->n_out; i++) {
            AVPacket *eof = NULL;
            av_thread_message_queue_send(d->mux_q[i], &eof, 0);
        }
    }
    av_packet_free(&pkt);
    return NULL;
}

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
    VideoCtx         vc;
    MuxArgs          ma;
    AVBufferRef     *fhwfr;
    int              fw, fh, fpix;
    char             vlabel[64];                 /* filter output label, ladder only */
    pthread_t        th_output, th_mux, th_wd;
    int              started_output, started_mux, started_wd, hdr_written;
} Rung;

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
    VideoHold             hold;              /* multiview: latest decoded frame for the compositor */
    int64_t              *wrap_off;          /* per stream: 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;         /* per stream: last RAW ts (wrap detection) */
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
    int64_t               tick_dur_us;       /* 1/out_fps in us */
    int                   live;
    int64_t               slate_after_us;    /* stale hold -> black cell (0 = never) */
    /* stats (compositor is the cadence owner in multiview) */
    int64_t               emitted, dup;
    int64_t              *dbg_dec_sum;        /* optional: sum of per-input dec_frames */
} CompositorCtx;

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
static void *compositor_thread(void *arg)
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
    /* AUDIO-FOLLOW (Option A) latch: average the per-slot lag over a startup window (past the
     * lossy join) and latch a STABLE signed offset, published to house_skew for the audio's
     * one-time deterministic correction. Re-latched on outage return. */
    int64_t  af_off[PTV_MAX_INPUT] = {0};     /* audio-follow: continuously-smoothed per-slot lag (EMA, us) */
    int64_t  af_t0[PTV_MAX_INPUT]  = {0};     /* tick+1 of this slot's first real frame (0 = unseeded) */
    int      h0_logged[PTV_MAX_INPUT] = {0};  /* P0 diag: one-shot PTV-H0 per slot at first display */
    int      done_in[PTV_MAX_INPUT] = {0};
    int64_t rung_pts[PTV_MAX_RUNG] = {0};
    AVFrame *filt = av_frame_alloc();
    int64_t tick = 0, wall0 = 0;
    const char *pe = getenv("PTV_PREROLL_MS");
    int preroll_ms = pe ? atoi(pe) : 350;
    int n_prime = (preroll_ms > 0 && c->tick_dur_us > 0) ? (int)((int64_t)preroll_ms * 1000 / c->tick_dur_us) : 0;
    int64_t diag_t0 = av_gettime_relative(), diag_last = diag_t0;
    int64_t stat_last = diag_t0, stat_prev = 0;

    if (!filt) goto done;
    if (n_prime > PTV_FRAME_QDEPTH - 8) n_prime = PTV_FRAME_QDEPTH - 8;
    if (n_prime < 0) n_prime = 0;
    for (k = 0; k < n; k++) blackf[k] = make_black_frame(c->inputs[k].vdec);

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
        wall0 = av_gettime_relative();
    }

    for (;;) {
        int all_eof = 1, any_fresh = 0;
        int64_t now_us;
        {                                            /* wall-pace the house tick (also offline:
                                                      * inputs have independent clocks, so the
                                                      * mosaic cadence is the house rate, not media) */
            int64_t target = wall0 + tick * c->tick_dur_us;
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
        }
        now_us = av_gettime_relative();

        for (k = 0; k < n; k++) {                    /* pop ONE frame from this input's jitter
                                                      * buffer (FIFO) -> feed buffersrc k; dup-hold
                                                      * its last frame on underrun, black when stale */
            VideoHold *h = &c->inputs[k].hold;
            AVFrame *f = NULL, *st; int stale, fresh = 0;
            /* Take a candidate: a frame held back last tick (pending) or a fresh pop. The
             * CONTENT CLAMP then only DISPLAYS it once the house clock reaches its content-time;
             * a frame whose content leaps ahead of the clock (a startup/source PTS gap from
             * skipped/corrupt frames) is held — the cell's video freezes across the gap while
             * audio continues, instead of the video racing ahead (the per-slot audio-late). */
            AVFrame *cand = pending[k];
            pending[k] = NULL;
            if (!cand) {
                int rr = av_thread_message_queue_recv(h->q, &f, AV_THREAD_MESSAGE_NONBLOCK);
                if (rr >= 0)                cand = f;
                else if (rr == AVERROR_EOF) done_in[k] = 1;
            }
            if (cand && g_mv_clamp && c->tick_dur_us > 0 && cand->pts != AV_NOPTS_VALUE) {
                int64_t h0c; pthread_mutex_lock(&c->inputs[k].h0_lock); h0c = c->inputs[k].h0; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                if (h0c != AV_NOPTS_VALUE) {
                    int64_t cand_age = av_rescale_q(cand->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - h0c;
                    if (cand_age > tick * c->tick_dur_us + c->tick_dur_us) {  /* content leads the clock -> hold */
                        pending[k] = cand; cand = NULL;
                    }
                }
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
                    h0k = av_rescale_q(last[k]->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q) - tick * c->tick_dur_us;
                    pthread_mutex_lock(&c->inputs[k].h0_lock); c->inputs[k].h0 = h0k; pthread_mutex_unlock(&c->inputs[k].h0_lock);
                }
                if (h0k != AV_NOPTS_VALUE && last[k]->pts != AV_NOPTS_VALUE) {
                    int64_t disp_src = av_rescale_q(last[k]->pts, c->inputs[k].ist_tb, AV_TIME_BASE_Q);
                    int64_t sk = tick * c->tick_dur_us - (disp_src - h0k);
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
                    if (n > 1 && g_h0_reanchor &&
                        sk < -(int64_t)g_h0_reanchor_ms * 1000) {
                        int64_t shift = -sk + c->tick_dur_us;     /* bring sk from negative to +1 tick */
                        pthread_mutex_lock(&c->inputs[k].h0_lock);
                        c->inputs[k].h0 += shift; h0k = c->inputs[k].h0;
                        pthread_mutex_unlock(&c->inputs[k].h0_lock);
                        sk = tick * c->tick_dur_us - (disp_src - h0k);   /* now ≈ +1 tick */
                        af_off[k] = sk;                            /* snap the audio-follow EMA to the floored lag */
                        if (g_diag)
                            av_log(NULL, AV_LOG_INFO,
                                "[PTV-REANCHOR2] in%d tick=%"PRId64" video-ahead → h0 +%"PRId64"ms, lag→%"PRId64"ms\n",
                                k, tick, shift / 1000, sk / 1000);
                    }
                    if (g_diag && !h0_logged[k]) {   /* P0: one-shot per slot at its first displayed frame */
                        h0_logged[k] = 1;
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-H0] in%d FIRST-DISPLAY tick=%"PRId64" h0=%"PRId64"ms first_disp_src=%"PRId64"ms (disp-h0=%"PRId64"ms) out=%"PRId64"ms lag0=%"PRId64"ms qd=%d dec=%"PRId64"\n",
                            k, tick, h0k / 1000, disp_src / 1000, (disp_src - h0k) / 1000,
                            (tick * c->tick_dur_us) / 1000, sk / 1000,
                            av_thread_message_queue_nb_elems(c->inputs[k].hold.q), c->inputs[k].dc.dec_frames);
                    }
                    /* startup/ramp trace: per-tick for the first ~3s, then 1/s out to ~60s to capture the
                     * full per-slot lag RAMP (P0: see whether disp-h0 advances slower/faster than out). */
                    if (g_diag && (tick < 75 || (tick < 1500 && tick % 25 == 0)))
                        av_log(NULL, AV_LOG_INFO,
                            "[PTV-START] t=%"PRId64" in%d age=%"PRId64"ms out=%"PRId64"ms h0=%"PRId64"ms srcpts=%"PRId64" lag=%"PRId64"ms fresh=%d qd=%d\n",
                            tick, k, (disp_src - h0k) / 1000, (tick * c->tick_dur_us) / 1000, h0k / 1000,
                            last[k]->pts, sk / 1000, fresh, av_thread_message_queue_nb_elems(c->inputs[k].hold.q));
                    lag_true_us[k] = sk;                                  /* PTV_DIAG: capture BEFORE clamp = lip-sync truth */
                    c->inputs[k].house_lag_true = sk;                     /* publish for the per-slot audio lip-sync probe */
                    /* A/V probe (read-only): record this slot's distinct displayed content → its
                     * first-display output time, so the slot's audio can pair against it (§3.2b). */
                    if (fresh)
                        vring_put(&c->inputs[k].vring, disp_src, tick * c->tick_dur_us);
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
            if (av_buffersrc_add_frame(c->fsrc[k], st) < 0) av_frame_free(&st);
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
        g_vout_us = c->emitted * c->tick_dur_us;   /* PTV_DIAG: video output time for the audio probe */

        if (g_diag) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                char db[448]; int dp = 0;
                for (k = 0; k < n && dp < (int)sizeof db - 56; k++)
                    dp += snprintf(db + dp, sizeof db - dp, " in%d:dec=%"PRId64"/skew=%dms/lag=%dms/holddrop=%"PRId64,
                                   k, c->inputs[k].dc.dec_frames, (int)(skew_us[k] / 1000), (int)(lag_true_us[k] / 1000),
                                   c->inputs[k].hold.framedrop);   /* drop-oldest count: startup overflow = video-lead cause */
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
                double fps   = (c->emitted - stat_prev) / (dt > 0 ? dt : 1);
                double secs  = c->emitted * c->tick_dur_us / 1000000.0;
                double wall  = (nows - wall0) / 1000000.0;
                double speed = wall > 0 ? secs / wall : 0;
                double kbps  = secs > 0 ? g_muxed_bytes * 8.0 / secs / 1000.0 : 0;
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                char ls[320]; int lp = 0;   /* per-input frame loss: qdrop = video_q overflow, corrupt = demux+decode */
                for (k = 0; k < n && lp < (int)sizeof ls - 48; k++)
                    lp += snprintf(ls + lp, sizeof ls - lp, " in%d:qdrop=%"PRId64"/corrupt=%"PRId64,
                                   k, c->inputs[k].da.vdrop, c->inputs[k].da.vcorrupt + c->inputs[k].dc.vcorrupt);
                av_log(NULL, AV_LOG_INFO,
                    "frame=%6"PRId64" fps=%3.0f size=%8"PRId64"KiB time=%02d:%02d:%05.2f "
                    "bitrate=%7.1fkbits/s dup=%"PRId64" drop=%"PRId64" speed=%4.2fx%s\n",
                    c->emitted, fps, g_muxed_bytes / 1024, hh, mm, ss, kbps,
                    c->dup, c->framedrop[0], speed, ls);
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
    if (n_rung > PTV_MAX_RUNG) {
        av_log(NULL, AV_LOG_WARNING, "%d outputs > max %d; using the first %d\n", n_rung, PTV_MAX_RUNG, PTV_MAX_RUNG);
        n_rung = PTV_MAX_RUNG;
    }
    memset(inputs, 0, sizeof inputs); memset(as, 0, sizeof as); memset(rung, 0, sizeof rung);
    memset(&comp, 0, sizeof comp);
    for (k = 0; k < n_input; k++) {
        inputs[k].url = ins->groups[k].arg;
        inputs[k].h0  = AV_NOPTS_VALUE;
        pthread_mutex_init(&inputs[k].h0_lock, NULL);
        pthread_mutex_init(&inputs[k].hold.lock, NULL);
        pthread_mutex_init(&inputs[k].vring.lock, NULL);
        if (multiview) {                         /* per-input jitter buffer for the compositor */
            if ((ret = av_thread_message_queue_alloc(&inputs[k].hold.q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
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
        if (!inputs[k].wrap_off || !inputs[k].wrap_last) { ret = AVERROR(ENOMEM); goto end; }
        for (si = 0; si < (int)inputs[k].ifmt->nb_streams; si++) inputs[k].wrap_last[si] = AV_NOPTS_VALUE;
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
            e->sample_fmt  = aenc->sample_fmts ? aenc->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
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
        if (af && build_audio_filter(a, a->dec, kist->time_base, af, sfmt) < 0) {
            av_log(NULL, AV_LOG_WARNING, "audio track %d filtergraph failed; plain resample\n", k);
            avfilter_graph_free(&a->afg); a->use_fg = 0;
        }
        if (!a->use_fg) {              /* no -af (or graph failed): plain resample */
            swr_alloc_set_opts2(&a->swr, &a->out_chl, sfmt, 48000,
                                &a->dec->ch_layout, a->dec->sample_fmt, a->dec->sample_rate, 0, NULL);
            if (!a->swr || swr_init(a->swr) < 0) { av_log(NULL, AV_LOG_WARNING, "audio track %d swr init failed; skipped\n", k); }
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

    /* queues: per-input video_q, one audio_q per transcoded track, per-rung frame_q + mux_q */
    for (k = 0; k < n_input; k++) {
        if ((ret = av_thread_message_queue_alloc(&inputs[k].video_q, g_videoq, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(inputs[k].video_q, free_pkt_msg);
    }
    for (k = 0; k < n_audio; k++) {
        if ((ret = av_thread_message_queue_alloc(&audio_q[k], PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(audio_q[k], free_pkt_msg);
    }
    for (r = 0; r < n_rung; r++) {
        if ((ret = av_thread_message_queue_alloc(&rung[r].frame_q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].frame_q, free_frame_msg);
        if ((ret = av_thread_message_queue_alloc(&rung[r].mux_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].mux_q, free_pkt_msg);
    }

    /* per-input decode side. single-input: inputs[0].dc already holds the graph
     * (fg/fsrc/fsink/filtering) + feeds the rung frame_q inline. multiview: each
     * decode stages into hold; the compositor owns the graph + frame_q fan. */
    for (k = 0; k < n_input; k++) {
        DecodeCtx *d = &inputs[k].dc;
        d->video_q = inputs[k].video_q; d->vdec = inputs[k].vdec; d->ist_tb = inputs[k].ist_tb;
        d->h0 = &inputs[k].h0; d->h0_lock = &inputs[k].h0_lock; d->live = live;
        if (multiview) { d->hold = &inputs[k].hold; d->filtering = 0; d->n_rung = 0; }
        else { d->n_rung = n_rung; for (r = 0; r < n_rung; r++) d->frame_q[r] = rung[r].frame_q; }
    }

    if (multiview) {                                 /* compositor = the video house clock */
        comp.inputs = inputs; comp.n_input = n_input; comp.fg = fg; comp.n_rung = n_rung;
        comp.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        comp.live = live;
        comp.slate_after_us = live ? 5 * (int64_t)AV_TIME_BASE : 0;   /* stale cell -> black after 5s */
        for (k = 0; k < n_input; k++) comp.fsrc[k] = fsrc[k];
        for (r = 0; r < n_rung; r++) { comp.fsink[r] = vsink[r]; comp.frame_q[r] = rung[r].frame_q; }
    }

    /* per-rung output side */
    for (r = 0; r < n_rung; r++) {
        VideoCtx *vc = &rung[r].vc;
        vc->frame_q = rung[r].frame_q; vc->mux_q = rung[r].mux_q; vc->venc = rung[r].venc;
        vc->out_tb = filtering ? av_buffersink_get_time_base(vsink[r]) : inputs[0].ist_tb;
        vc->tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        vc->live = live; vc->passthrough = multiview;
        vc->h0 = &inputs[0].h0; vc->h0_lock = &inputs[0].h0_lock;
        vc->house_skew = &inputs[0].house_skew;
        vc->vring = (!multiview && r == 0) ? &inputs[0].vring : NULL;  /* single-input: master rung feeds the A/V probe ring (multiview: compositor does) */
        vc->is_master = (r == 0);
        vc->dbg_video_q = inputs[0].video_q; vc->dbg_dec_frames = &inputs[0].dc.dec_frames; vc->dbg_vcorrupt = &inputs[0].dc.vcorrupt;
        vc->dbg_vdrop = &inputs[0].da.vdrop; vc->dbg_pcorrupt = &inputs[0].da.vcorrupt;   /* stats: demux video_q drops + corrupt-pkt */
        rung[r].ma.ofmt = rung[r].ofmt; rung[r].ma.mux_q = rung[r].mux_q;
        rung[r].ma.n_producers = 1 + n_audio + n_copy_inputs;   /* video out + N audio + per-input copy fan */
    }
    for (k = 0; k < n_audio; k++) {              /* per-track audio: source from its input's clock */
        as[k].audio_q = audio_q[k];
        as[k].h0 = &inputs[asrc_in[k]].h0; as[k].h0_lock = &inputs[asrc_in[k]].h0_lock;
        as[k].house_skew = &inputs[asrc_in[k]].house_skew;
        as[k].house_lag_true = (n_input > 1) ? &inputs[asrc_in[k]].house_lag_true : NULL;  /* multiview: true lag; single: NULL→house_skew */
        as[k].vring = &inputs[asrc_in[k]].vring;         /* A/V probe: pair this track's audio against its input's video ring */
        as[k].disturb_epoch = &inputs[asrc_in[k]].house_disturb;  /* B3: PLL mid-run re-acquire trigger (slate-return/discont) */
        as[k].multiview = (n_input > 1);                 /* multiview-only: enable deterministic audio-follow */
        as[k].af_applied_us = 0;
        as[k].dbg_k = k; as[k].dbg_in = asrc_in[k]; as[k].dbg_first_out = AV_NOPTS_VALUE;
        for (r = 0; r < n_rung; r++) as[k].mux_q[r] = rung[r].mux_q;
    }
    for (kk = 0; kk < n_input; kk++) {           /* per-input demux args (pass/n_pass set in copy loop) */
        DemuxArgs *d = &inputs[kk].da;
        d->ifmt = inputs[kk].ifmt; d->video_q = inputs[kk].video_q;
        d->vstream = inputs[kk].vstream; d->drop = is_net_url(inputs[kk].url); d->n_out = n_rung;
        d->h0 = &inputs[kk].h0; d->h0_lock = &inputs[kk].h0_lock; d->house_skew = &inputs[kk].house_skew;
        d->disturb_epoch = &inputs[kk].house_disturb;   /* B3: discont absorber arms the PLL mid-run re-acquire */
        d->wrap_off = inputs[kk].wrap_off; d->wrap_last = inputs[kk].wrap_last;
        for (r = 0; r < n_rung; r++) d->mux_q[r] = rung[r].mux_q;
        d->n_audio = 0;
        for (k = 0; k < n_audio; k++)
            if (asrc_in[k] == kk) { d->audio_q[d->n_audio] = audio_q[k]; d->astream[d->n_audio] = asrc[k]; d->n_audio++; }
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
    if (getenv("PTV_NO_AVLOCK")) g_avlock = 0;   /* revert to source-locked audio (drifts on dup) */
    if (getenv("PTV_NO_REANCHOR")) g_reanchor = 0;   /* keep stale dup skew across outages (A/B) */
    if (getenv("PTV_MV_CLAMP")) g_mv_clamp = 1;      /* opt-in: re-enable the (stutter-prone) content clamp */
    if (getenv("PTV_NO_DISCONT")) g_discont = 0;     /* A/B: don't absorb source PTS discontinuities */
    { const char *dm = getenv("PTV_DISCONT_MS"); if (dm && atoi(dm) > 0) g_discont_ms = atoi(dm); }
    if (getenv("PTV_NO_AUDIO_FOLLOW")) g_audio_follow = 0;  /* A/B: multiview audio uses old floored/capped async skew */
    if (getenv("PTV_NO_H0_REANCHOR")) g_h0_reanchor = 0;    /* A/B: don't floor per-slot lag (allow video-ahead) */
    if (getenv("PTV_NO_H0_AT_DISPLAY")) g_h0_at_display = 0; /* A/B: multiview anchors h0 at first DECODE, not first DISPLAY */
    { const char *rm = getenv("PTV_H0_REANCHOR_MS"); if (rm && atoi(rm) > 0) g_h0_reanchor_ms = atoi(rm); }
    if (getenv("PTV_AF_NO_PLL")) g_af_pll = 0;              /* A/B: pure discrete drop/pad (no smooth nudge) */
    if (getenv("PTV_AF_NO_ANCHOR")) g_af_anchor = 0;        /* A/B: revert B1 → pre-B1 free-running counter */
    { const char *vq = getenv("PTV_VIDEOQ"); if (vq && atoi(vq) > 0) g_videoq = atoi(vq); }   /* video_q depth (startup-burst absorb) */
    if (getenv("PTV_KEEP_CORRUPT")) g_discardcorrupt = 0;   /* keep AV_PKT_FLAG_CORRUPT video packets (don't +discardcorrupt) */
    if (getenv("PTV_AVSYNC_PROBE")) g_avsync_probe = 1;    /* Phase A: read-only [PTV-AVSYNC2] real A/V offset */
    if (getenv("PTV_ATRACE")) g_atrace = 1;                /* temp: per-audio-frame startup trace to localize the bank */
    { const char *am = getenv("PTV_AF_ACQUIRE_MS"); if (am && atoi(am) > 0) g_af_acquire_us = atoi(am) * 1000; }
    { const char *rr = getenv("PTV_AF_RATE_MS_S");  if (rr && atoi(rr) > 0) g_af_rate_us = atoi(rr) * 1000; }
    if (getenv("PTV_AVSYNC_PLL")) g_avsync_pll = 1;        /* B3: closed-loop two-regime A/V controller on the measured offset (multiview transcoded audio) */
    { const char *es = getenv("PTV_PLL_EMA_SHIFT");  if (es && atoi(es) >= 0) g_pll_ema_shift = atoi(es); }
    { const char *tu = getenv("PTV_PLL_TAU_MS");     if (tu && atoi(tu) > 0) g_pll_tau_us = (int64_t)atoi(tu) * 1000; }
    { const char *aq = getenv("PTV_PLL_ACQUIRE_MS"); if (aq && atoi(aq) > 0) g_pll_acquire_us = atoi(aq) * 1000; }
    { const char *an = getenv("PTV_PLL_ACQUIRE_N");  if (an && atoi(an) > 0) g_pll_acquire_n = atoi(an); }
    { const char *su = getenv("PTV_PLL_STARTUP_MS"); if (su && atoi(su) > 0) g_pll_startup_us = (int64_t)atoi(su) * 1000; }
    { const char *ak = getenv("PTV_PLL_ACQUIRE_K");  if (ak && atoi(ak) > 0) g_pll_acquire_k = atoi(ak); }
    { const char *s = getenv("PTV_SLOW_US"); g_slow = s ? atoi(s) : 0; }
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
    if (!hide_banner)
        ptv_show_banner();
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
