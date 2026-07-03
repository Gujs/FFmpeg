# ptvencoder version history

Per-release notes, extracted verbatim from the `ptvencoder.c` header on 2026-07-03
(the in-code block had grown to ~190 comment lines). **Add new release notes HERE,**
keep only the current `PTVENCODER_VERSION` define in the source. This file is part of
the v2 `0001` patch (additive, travels with the source to the build box).

## 0.9.15.2 (2026-07-03)

- **CADENCE DECIMATION: surplus real-frame sources** (`PTV_NO_DECIMATE` reverts; `decim=`
  counter in stats). The 0.9.15.1 `cf=` readout on live NewsNation showed **-823ppm locked** —
  the transport clock is TRUE, so clock-follow (correctly) stayed dark and the real fault
  surfaced: wall-timed measurement (509 frames / 20.0s wall, DTS span 19.92s) proves the source
  delivers a wandering **~25.3-25.5 REAL frames per second, consistently stamped, while
  declaring 25/1**. (The original "+1.2% fast clock" wire read was stamp SPACING over a packet
  count — it cannot distinguish clock offset from surplus cadence; `cf=` can, and did.) Surplus
  cadence can't be rate-followed (output would wander with it) — the frames must be decimated
  by CONTENT: in the normal emit pop path, a popped frame whose `content_index` does not
  advance past the last emitted tick is surplus — pop through to a fresher one (<=3 pops/tick,
  bounded). The output samples the source's own timeline at exactly house rate, so lip-sync is
  exact by construction; frame_q stays level (consumption self-adapts to the true delivered
  cadence), hs stays flat, aresample goes quiet. The single-input mirror of the 0.9.13 mosaic
  per-slot multi-pop. Can never fire on <=house-rate sources (their content indices always
  advance). Validated on a +1.8% surplus fixture (setts ts x1000/1018 sent plain `-re`; SPS
  still declares 25 — needs the new cmd-single.sh `FPS=25` override because avg_frame_rate
  probes 305/12): house 25/1, `decim=` accruing evenly ~27/min (= the surplus, exactly),
  dup=0, hs=+0ms flat, cf stayed <+2100ppm (clock-follow dark, as it must be).

## 0.9.15.1 (2026-07-03)

- **CLOCK-FOLLOW convergence + observability fixes** (first NewsNation deploy didn't arm in
  4min and was silent about why): (1) the post-lock outlier-reject band was ±3000ppm, but the
  EMA sits at only ~72% of a true offset at lock — every honest window of a clean +12000ppm
  source then exceeded the band and the estimate deadlocked below truth; band now ±8000ppm
  (still rejects burst-alias spikes; max honest gap for the ±2% cap is ~5500). (2) The stats
  line now shows ` cf=<ppm>` whenever notable (`?` suffix = not yet locked), and an always-on
  breadcrumb reports "N/M windows accepted, ema X" every ~3min while unlocked — estimator
  starvation is never silent again. Re-validated on the +1.2% fixture: armed at lock,
  converged past the old deadlock point, health flat.

## 0.9.15 (2026-07-03)

- **CLOCK-FOLLOW: follow a large verified source-clock offset** (`PTV_NO_CLOCKFOLLOW` reverts;
  single-input live only — a mosaic has one output clock and N source clocks, and its slots
  already absorb fast sources via 0.9.13 per-slot residence decimation). NewsNation measured at
  the wire: PTS advance 25.305 fps from a stream declaring 25.0 = a +12,200ppm transport clock
  (relay/playout fault, provider won't fix). The WUCR servo pegged at its +-0.6% gentle limit ->
  frame_q pinned full forever (also the 2.2GB VRAM high-water), hs climbing, aresample churning
  at +-1% (audible). Now a PARALLEL coarse FLL (same unbiased sub-window rate as the v0.9.0
  estimator, but +-3% envelope, own +-3000ppm outlier reject, 60s lock — the tight FLL keeps its
  +-300ppm crystal guards for genlock pacing) feeds the servo, which FOLLOWS offsets beyond
  +-3000ppm (hysteresis release <2000, cap +-2%): output pacing and PCR run at the source's true
  rate — receivers slave to PCR as with any PCR-locked feed — buffers stay level and audio drops
  to a steady soft resample. Film-in-NTSC can never arm it (film DTS advance is realtime; the
  estimator reads ~0 — the 0.9.10.1 content-vs-clock discrimination, by construction).
  `[PTV-CLOCK]` logs arm/release; PTV_DIAG line gains `cf=<ppm>/<locked>`. Validated against a
  clean +1.2%-clock fixture (setts-retimed -re + wire-restore bsf; note: burst_send's byte-budget
  pacing is NOT a clock-offset fixture on VBR files): armed at lock, fps 30.5 = source pace,
  dup=0, hs flat, async=+0ppm.

## 0.9.14.2 (2026-07-03)

- **AUTO-BANK: audio_q sized like the deep preroll (the THIRD and last sizing side-car).**
  Second live deploy (Unique_TV, 0.9.14.1): video stayed perfect but audio still clicked —
  PTV_DIAG showed `adrop` ~25/s: a 7s delivery clump slams ~330 audio packets into the
  demux->audio queue, which was still PTV_QDEPTH=48 deep unless the manual PTV_PREROLL_MS
  path resized it. Now sized for the bank ceiling with the manual path's own formula
  (~50 pkt/s x ceiling + margin => 648 @ 12s), live runs only. Root lesson recorded: the
  manual PTV_PREROLL_MS env never worked by the prime alone — it also resizes video_q, the
  delivery-gate FIFO, and audio_q at startup; the runtime bank had to replicate ALL THREE
  (0.9.14 video_q, 0.9.14.1 gate, 0.9.14.2 audio_q). Validated under 7s clumps with diag:
  adrop=0, async=+0ppm, bank at target. Acceptance metric set for bursty channels is now:
  dup + drop + async + ADROP (diag).

## 0.9.14.1 (2026-07-03)

- **AUTO-BANK audio-path sizing** — the deep-preroll path's own sizing rules, applied
  automatically (owner: "same system as PTV_PREROLL_MS, just automatic sizing"). First live
  deploy (Unique_TV) healed the VIDEO completely (dup 397/9min -> 0, hs=+0) but audio kept
  clicking (async swinging +-1000ppm): with a 10s bank the delivery gate holds ~470 audio
  packets steady and a burst clump surges it to ~800 — past the old maxq=512 backstop ->
  back-pressure -> demux audio drops -> resampler compensation. The hold-FIFO backstop is now
  1024 by default and auto-sized per gated track for gate-cap + bank-ceiling + clump surge
  (~50 pkt/s x 20s), exactly as the manual deep-preroll sized itself; nodes are per-enqueue so
  the generous backstop costs nothing. Validated at the 12s ceiling bank under 8s clumps:
  async=+0ppm at every sample (the live channel swung +-1000ppm).


## 0.9.14 (2026-07-03)

- **AUTO-BANK: the runtime cushion escalation for bursty channels** (`PTV_NO_AUTOBANK` reverts;
  ceiling `PTV_CUSHION_MAX_MS`, default 12000 — owner-set: >12s stalls are upstream incidents to
  surface, not absorb). The 0.9.10 adaptive cushion lives in frame_q (DECODED frames) and maxes
  at ~4s — structurally too shallow for HLS-burst channels with 6-8s gaps (Unique_TV/ZOE class),
  which until now needed the manual `PTV_PREROLL_MS/PTV_VIDEOQ` recipe. Now the demux BURSTY
  detector escalates at runtime: on >=3 stalls >=1.5s per 60s, or a single stall >=3s, the bank
  target becomes 1.5x the worst observed stall (capped). Arming flips the master rung to the
  deep-prime BLOCKING push, so each stall's own latency is RETAINED as a compressed-packet bank
  in video_q (~KBs) instead of draining — the channel self-heals within a stall cycle or two,
  no restart, no fill artifacts. Growth is self-limiting: once bank >= gap, stalls stop starving.
  The per-rung delivery-gate cap rides the bank (audio waits out long stalls with video; gate
  cap_us is now runtime-atomic). Retires after 6h without qualifying stalls (`PTV_BANK_DECAY_S`
  test hook). Stats gain ` bank=actual/target ms` while armed. The [PTV-BURSTY] env-recipe
  advisory now fires only in the one case left for a human: bank at its ceiling and still short.
  `video_q` default 256 -> 512 (bank headroom; compressed packets, negligible memory). Explicit
  `PTV_PREROLL_MS` deep primes keep working and pre-fill at startup as before. Single-input live
  only (mosaic slots ride 0.9.13 cadence-residence starvation-slip; a per-slot mv bank is future
  work if ever needed).

## 0.9.13 (2026-07-03)

- **Per-slot CADENCE RESIDENCE in the compositor** (`PTV_NO_RESIDENCE` reverts). A slot is now
  consumed at its SOURCE rate: the next frame pops only when the previous frame's
  content-projected residence has elapsed on the house axis (EMA-smoothed content deltas —
  never raw per-frame deltas, which are jittery on real interlaced feeds and caused the old
  content clamp's stutter regression; the clamp stays opt-in and unchanged). Effects:
  - A rate-mismatched slot (25fps in a 29.97 mosaic — GB News class) holds a regular 5:6
    cadence instead of draining its buffer at house rate and living at the empty boundary,
    so clumped SRT/HLS arrival is absorbed by the buffer instead of showing as
    freeze-then-1.2x-fast batching. On display the due time re-bases with
    max(due, now − half-tick): a starvation deficit becomes constant slot latency (like a
    matched-rate slot's jitter buffer), NEVER fast-motion catch-up.
  - A high-rate slot (59.94 in a 29.97 house) pops multiple due frames per tick (≤4) and
    displays the newest — clean 2:1 decimation instead of hold-queue overflow.
  - Matched-rate slots are untouched by construction (due advances exactly one tick per frame).
  - Occupancy servo: residence trimmed proportionally (max ±2%) toward the primed jitter depth,
    so long-term consumption equals arrival even if the cadence estimate is biased (the
    single-input WUCR lesson: proportional, small, capped). Target clamps below the valve.
  - Pressure valve: a ≥75%-full hold queue bypasses the gate (self-corrects a wrong estimate);
    a valve-forced pop RE-BASES the schedule instead of advancing it (accumulating ratcheted
    the schedule +1.0-1.6s inside the first second of a backlogged start).
  - Startup backlog trim (live only): after preroll each slot's hold queue is trimmed to the
    primed depth — a join can dump seconds of banked frames at once (deep UDP socket buffer
    read in one burst), and a jitter buffer must ACQUIRE at its target depth; the excess is
    stale latency (measured: occ pinned at the valve from tick 0 without this).
  - Residence holds park the frame in `pending`, so the existing "deliberate pacing must not
    ratchet audio skew" rule applies unchanged.
  - Validated (local 2x2, VT mirror; slot0=25fps CFR, slot2=29.97 CFR, YDIF repeat analysis
    on decoded quadrants vs PTV_NO_RESIDENCE controls): repeat-gap regularity 4-6x better
    (gap_sd 2.8-3.8 vs 14.6 in every control), matched slot pd=0 and repeats at content
    baseline, per-slot sk bounded (+0 smooth; burst deficit converts to CONSTANT latency,
    +2.3s stable, vs unbounded ratchet pre-servo), pd=4.4/s on 25-in-29.97 slots = the exact
    5:6 conversion rate; with a 4.5s prime the bursty slot runs sv=0 after acquisition.
- **Discontinuity events log ALWAYS-ON** (were PTV_DIAG-only — on single-input AND multiview,
  so production logs were silent no matter how many glues fired; TruBLU's ~15-min splices have
  been absorbed invisibly all along). `[PTV-LAYERA] jump/flush` and `[PTV-DISCONT] absorbed /
  audio GAP` now print unconditionally at INFO — they are rare (few/hour worst case) and
  operationally meaningful. The verbose [PTV-QSNAP] queue dumps stay diag-gated. Verified with
  the send-jump harness: a +30s injected splice logs jump+flush (applied_offset=-29.98s) with
  no PTV_DIAG; behavior itself is unchanged (identical outcome counters diag vs not).
  Each flush also emits `[PTV-GLUE]` — running per-input source A/V mis-mux stats
  (per-glue vid_err, mean/max |err|, >100ms count) — the LAYERA-retirement decision data:
  if real glues show |err| ~0 over days, the simpler per-stream absorber suffices and the
  LAYERA state machine can go; persistent 100ms+ errors mean LAYERA is actively correcting
  mis-muxed splices the absorber would leak into audible audio steps.
- **Multiview stats-line parity**: the mosaic line now matches single-input
  (`frame= fps= time= dup= drop=` + `dlvhold=/dlvforced=` gate readout; size/bitrate/speed/
  genlock dropped per the v0.9.10 rationale) plus per-slot
  `inK:qdrop/corrupt/pd/sv/sk` — `pd` = cadence holds (NORMAL for a rate-mismatched slot),
  `sv` = genuine starvation dups, `sk` = the published per-slot audio skew (ms).
  `-log-legend` documents the mv line.

## 0.9.12.1 (2026-07-03)

- **Delivery gate default-ON for multiview** (`PTV_NO_DELIVERY_MV` reverts; `PTV_DELIVERY_MV`, the
  old opt-in, is now a harmless no-op). The wire skew on an ungated output equals the CURRENT
  video in-process hold: the muxer interleaves by DTS but waits only max_interleave_delta (200ms)
  for the late stream, so video held in-process (frame_q occupancy + ~1s NVENC steady, several
  seconds during post-stall catch-up on bursty slots) lands on the wire out-of-order behind
  already-written audio. cor-2's downstream sync_check (first-50-packets
  `D = video_last_pts - audio_last_pts`, restart at |D| > 2s) measured D = -0.6..-5.9s on the
  ungated mosaics and restart-looped them 12-67x/day. Single-input never showed it because the
  gate ships there since v0.7.0. With the gate, audio holds until the encoder's video front
  reaches its DTS → wire D ≈ -0.1s. Player lip-sync was never affected (PTS mapping correct);
  this fixes wire STAGING. Local 2x2 A/B (VideoToolbox mirror, sync_check's exact probe):
  gate-on D = -0.05..-0.08s flat across normal, 3s-primed, and 4s-burst-slot runs, fps=30 dup≈0
  drop=0 (no pipeline disturbance); the [PTV-BURSTY] advisor fires per-input on mosaics.
  (Local controls can't reproduce the production magnitude — the VT video path holds only
  ~100ms; the -0.6..-5.9s figures are the production measurement itself.) The hold-FIFO
  backstop auto-sizes by audio-track count (each track fans into every rung's gate: a 2x2 mosaic =
  4 AAC x ~47pkt/s x 3s cap ≈ 570 > the 512 single-input default; explicit PTV_DELIVERY_MAXQ wins).

## 0.9.12 (2026-07-03)

- **MV-EXACTTICK** (`PTV_NO_MV_EXACTTICK` reverts): the compositor's MEASUREMENT axes (B3 PLL vring
  sensor, per-slot sk/house_skew, h0 anchor, wall pacing, content clamp, vout probe, stats) now use
  exact-rational tick-us via `mv_tick_us()` instead of tick x integer `tick_dur_us`. The integer axis
  ran +10ppm fast at 30000/1001 (-20ppm at 59.94), so the per-slot audio followers ENFORCED ~36ms/h
  audio-late (~72ms/h audio-early at 59.94) onto the wire between PLL re-acquires — observed live as
  the Salem-slot sawtooth on the 59.94 mosaic (accumulates through long splice-free program blocks,
  snaps to zero at each break). Splice-heavy slots reset too often to show it. The PLL itself is
  correct and stays; this fixes its reference ruler. Audit:
  analysis/ptvencoder-0911-multiview-tick-audit.md. NOTE: the PTV_TICK_ADJ_US diag accelerator now
  skews PACING ONLY (an accelerator inside the measurement axis would re-create the bug).
- **[PTV-BURSTY] advisor**: demux-side detection of HLS-burst-over-SRT delivery (>=3 completed
  video-arrival stalls >=1.5s per 60s = periodic pattern; single outages never trip). Logs one
  WARNING per minute with the sized, copy-pasteable channel-environment recipe
  (PTV_PREROLL_MS="N",PTV_VIDEOQ="M"; 1.5x worst observed gap, capped 30s) and stays SILENT when
  PTV_PREROLL_MS already covers the observed gaps. Validated: fires under pulsed droughts, silent
  when configured.
- Docs: version history extracted to this file; --help gains the environment-variables section;
  stats legend documents pd=.

## Pre-0.9.12 history (extracted verbatim from the source header)

```
#define PTVENCODER_VERSION "0.9.12-dev" /* 0.9.11 (telecine-aware emit, 2026-07-03): honor repeat_pict — during
                                    23.976-film-in-29.97 (2:3 soft pulldown; AWE movies) a flagged frame OCCUPIES
                                    its content-projected extra ticks via a 1-frame lookahead sharing the stamping
                                    arithmetic (content_index()); repeat_pict only ARMS the mode (>=3 progressive-
                                    rff of last 8). Film segments: consumption==supply -> no starvation dups
                                    (irregular stutter GONE), hs pinned 0 (aresample hard-comp clicks GONE),
                                    proper 2:3 wire cadence, servo/cushion quiet. New pd= counter (cadence holds)
                                    split from dup= (health). PTV_NO_PULLDOWN reverts. Design:
                                    analysis/ptvencoder-0911-repeat-pict-design.md. */
                                    /* 0.9.10.1 (film-cadence hotfix, 2026-07-02 late): (1) REPRIME state machine —
                                    an engagement is hard-capped at 10s with an UNCONDITIONAL 300s cooldown (the
                                    0.9.10 "continuing" clause let occupancy oscillating around the trigger re-arm
                                    forever: AWE 23.976-film segments [2:3 pulldown, ~24 AU/s — CONFIRMED by live
                                    telecine dissection, repeat_pict on 51% of frames] pinned the house at 0.77x
                                    for whole movie blocks -> downstream underrun). (2) Sustained positive servo
                                    authority capped +1.5% (pre-0.9.10 proven level): NEVER rate-match a content-
                                    rate deficit — film-in-NTSC dups ARE 3:2 pulldown; real clock offsets are
                                    ppm-scale. (3) PTV-EMPTY frame_q: per-episode lines only >=2s; sub-2s episodes
                                    -> one 60s summary line (film segments spammed a line every few seconds).
                                    0.9.11 TODO: honor repeat_pict in the emit (flagged frame = 1.5 ticks) so film
                                    consumption matches supply — no starvation/dups/spam at all. */
                                    /* 0.9.10 (adaptive cushion + proven-base defaults, 2026-07-02): (1) ADAPTIVE
                                    frame_q cushion — two discrete targets (BASE=resolved preroll ~1s, RAISED=
                                    PTV_CUSHION_MS default 4s); grows on 2 starvation episodes (>=200ms) within
                                    60min, shrinks after 6h quiet; fill/drain are LAZY via the servo gentle zone
                                    (+/-0.6% above the base floor) so transitions never jerk delivery; logs
                                    [PTV-CUSHION], -stats shows cushion=. PTV_NO_ADAPTIVE reverts. (2) WUCR +
                                    LAYERA + REPRIME now DEFAULT-ON (proven production posture; PTV_NO_WUCR/
                                    PTV_NO_LAYERA/PTV_NO_REPRIME revert); reprime rate-limited (1 per 5min) and
                                    triggered on the BASE floor, not the raised tier. (3) frame_q capacity default
                                    160 (slots only; PTV_FRAMEQ overrides). (4) PTV-EMPTY polish: video_q/mux_q
                                    lose the misleading chronic heartbeat (empty is their healthy state); frame_q
                                    watch runs ungated and feeds the adaptive cushion. LAYERA logic UNTOUCHED. */
                                    /* 0.9.9 (EXACTTICK, 2026-07-02): video content index stamped at the EXACT
                                    rational frame rate (was: divide by integer-us tick_dur_us -> ~10ppm mapping
                                    compression at 30000/1001 -> chronic audio-behind lip-sync drift +36ms/h on
                                    NTSC channels, zero at 25/50fps; found by 4-agent audit, confirmed by 6h
                                    oracle regression +42..52ms/h vs +36 predicted). PTV_NO_EXACTTICK reverts;
                                    PTV_TICK_ADJ_US=+N accelerates the old bug for falsification. */
                                    /* 0.9.8 (deep frame_q cushion, 2026-07-01): PTV_FRAMEQ raises the decode->output jitter-buffer cap (default 48) so a deep PTV_PREROLL_MS can hold a ~3-4s post-decode cushion — rides out an AWE ad-break decode-rate DIP without frame_q draining -> no house dup-fill -> no house_skew step. Default 48 = byte-identical. */
                                    /* 0.9.7 (audio filtergraph reconfig, 2026-06-29): rebuild the -af graph/swr when the SOURCE audio changes channel-layout/rate/fmt mid-stream (e.g. stereo→mono at an ad-splice) — fixes permanent audio loss; hysteresis ignores transient splice flips. Ported from legacy 0003. Output stays pinned stereo/48k. */
                                    /* 0.9.6 (WUCR proportional ρ servo, 2026-06-29): occupancy-locked house rate via a PROPORTIONAL controller (ρ=Kp·err) — kills the integral servo's ±7-13k ppm limit cycle; ρ parks at −(source ppm) so it doubles as a per-source rate readout. PTV_WUCR=1. */
                                    /* 0.9.5 (genlock phase-2a — LONG-BASELINE rate, 2026-06-27): the real cure
                                    * for the runaway the 0.9.4 guard only CAPPED. The 3s FLL window aliases bursty
                                    * delivery → ±1000ppm noise the EMA walks → house_skew accumulates. A longer
                                    * measurement window (PTV_GENLOCK_WINDOW_MS) averages the bursts out → the
                                    * recovered rate ≈ the TRUE source rate → the house clock matches the source →
                                    * house_skew stays ~0 (accumulation stops at the source, not just capped).
                                    * Env-tunable WINDOW_MS + EMA_SHIFT, slew scales with the window; DEFAULTS = the
                                    * old 3s/shift-6 path → byte-identical until raised (sandbox A/B turns it on, then
                                    * promote). Guard (0.9.4) stays as the backstop. phase-2b (a true phase-lock that
                                    * drives house_skew→0 via a g_house_skew_us atomic) is the follow-up IF a residual
                                    * bias survives the long baseline. Validate via the EXTERNAL oracle / house_skew on
                                    * a multi-hour sandbox run. ---- 0.9.4 (genlock STABILITY guard, 2026-06-27): fixes the TruBLU A/V root
                                    * cause found via the 0.9.3 PROBE + cor-1 16h diag — NOT an audio drift but a
                                    * genlock RATE-RUNAWAY: jittery/bursty sources alias the 3s FLL window → noisy
                                    * sub-window rates that the loose ±1% gate folded in → a ±1000ppm slew-limited
                                    * limit cycle + unbounded house_skew runaway (8.6→28s/16h), which async masks
                                    * until it frays → visible desync after hours. Every internal A/V metric is BLIND
                                    * to it (proven: 28s skew under a flat err; the live wire moved +1647→+127ms
                                    * while wall/dts/span read ~+10ms) → the fix is the CLOCK, not the audio. Guard:
                                    * (A) hard absolute bound on the applied rate (PTV_GENLOCK_MAX_PPM, default 300)
                                    * + (B) relative outlier rejection of burst-aliased windows (PTV_GENLOCK_REJECT_PPM,
                                    * default 700, ≥2×MAX). Default-on, PTV_NO_GENLOCK_GUARD reverts; clean sources
                                    * (Cinestar ±45, AWE ±271 measured) sit inside both → unaffected. NOTE: this is a
                                    * SAFETY FLOOR — it caps the runaway SLOPE (~2→~1s/hr) but a biased source pinned at
                                    * the bound still accumulates house_skew; the full cure (longer-baseline/median rate
                                    * estimator that removes the burst aliasing) is the planned next iteration. The
                                    * PTV_AVTRIM audio
                                    * actuator is RETIRED (all 3 candidate signals proved blind); PROBE kept as the
                                    * diagnostic that exposed this. Validate via the EXTERNAL oracle on a multi-hour run.
                                    * ---- 0.9.3 (single-input A/V drift-null PTV_AVTRIM — PROBE, 2026-06-27): the slow
                                    * audio-late drift (root cause of TruBLU "fine at 3:00, broken at 4:30"; genlock
                                    * A/B-proven innocent) is invisible to the legacy patch-0007 closed-loop signal
                                    * here, because that signal needs video+audio on DIFFERENT clocks and ptvencoder's
                                    * house clock + AVLOCK put them on ONE (the Session-83 blindness 0007 escaped) —
                                    * the drift moved from the timestamp domain into the CONTENT domain. So this step
                                    * LOGS THREE candidate drift signals via [PTV-AVTRIM] (PTV_AVTRIM_PROBE) and the box
                                    * picks which tracks the wire oracle (Rule-0, don't assume): wall = wall_a(C)−wall_v(C)
                                    * production timing (vring now carries the video mux-handoff wall time); dts = the
                                    * legacy timestamp offset (expected flat = masked); span = async sample-vs-source-content
                                    * slip (content domain). PTV_AVTRIM reserved for the actuator (NOT built — built on the
                                    * validated signal at the resampler input, legacy-0007 control law). Single-input only;
                                    * both default OFF → byte-identical. Multiview (B4) is a separate later change.
                                    * ---- 0.9.2 (diagnostics/logging cleanup, 2026-06-27): the always-on -stats
                                    * progress line is now an OPERATOR-TRUSTWORTHY line — genlock state +srcppm
                                    * (promoted from PTV_DIAG) and a NEW `async` aresample-work rate (ppm). The
                                    * MISLEADING internal A/V estimates ([PTV-AVSYNC] offset/house_skew,
                                    * [PTV-SWRDELAY], [PTV-CHAIN] outA-V) drop behind PTV_DIAG. A `-log-legend` flag
                                    * (+ a compact legend at startup) documents every field. An egress emitted-PES
                                    * lip-sync `emitA-V` was built and REJECTED by wire-oracle validation (a +200ms
                                    * content shift moved the oracle +200ms but emitA-V 0ms — it tracks encoder
                                    * reorder, not the content↔PTS offset that is lip-sync), so lip-sync stays an
                                    * EXTERNAL measurement (drift-continuous.py). Logging-only; no timing path touched.
                                    * ---- 0.9.1 (genlock + shallow input prime, 2026-06-27): the single-input output
                                    * cadence (ALL rungs) is SLAVED to the recovered SOURCE frame rate. A sliding-window FLL in the
                                    * demux thread (per-~3s UNBIASED Σdc/Σdw of post-unwrap video DTS vs wall clock → EMA τ≈4-5min,
                                    * slew-clamped, wild-chunk reject; re-anchored across disturbance epochs but KEEPING the learned
                                    * rate) publishes a Q20 rate ratio, and each rung's pacer scales its per-tick wall span by it via a
                                    * phase accumulator (a rate change never teleports the target). This stops the house clock drifting
                                    * vs the channel (the eye-observed ~1s/30min output-slower drift), so house_skew→0 and aresample is
                                    * freed for honest A/V trim (the rate slave is single-input only; the multiview compositor clock is
                                    * unchanged). PTV_NO_GENLOCK reverts to byte-identical free-run. PAIRED with a SHALLOW input prime
                                    * (v0.9.1): the single-input frame_q cushion defaults to ~1s (g_preroll_ms=1000) — smooths the bursty
                                    * decode-rate dips while video+gate-hold stays UNDER the §7.5a gate's normal 3s cap (no cap scaling).
                                    * Multiview reverts to the compositor's hold.q (already a paced per-input de-jitter buffer). The deep
                                    * video_q prime + the gate-cap auto-scale stay available for an explicit deep PTV_PREROLL_MS (bursty
                                    * Fintech-class), dormant by default. (v0.9.0 defaulted a 2s deep prime for single+MV; it over-delayed
                                    * video → §7.5a force-release on TruBLU (dlvforced=11614) + grid startup-black; reverted after a 3-agent
                                    * review showed a paced whole-stream input buffer would feed back into the genlock estimator.)
                                    * See analysis/ptvencoder-avsync-genlock-design.md.
                                    * ---- prior 0.8.2 (gap-vs-splice fix, 2026-06-26): the discontinuity absorber no longer
                                    * "re-bases to continuous" a FORWARD jump on a dense AUDIO stream when it is an
                                    * audio-only SOURCE GAP — i.e. the VIDEO stream did NOT also forward-cross recently
                                    * (content signal: a real ad-splice jumps video too) AND this stream's packets were
                                    * genuinely wall-absent ~the jump. Such a gap is left un-absorbed so aresample=async
                                    * hard-pads silence and audio stays aligned with the house-clock-continuous video
                                    * (copied AC-3 keeps the real forward gap). Whole-program SPLICEs (video crosses) and
                                    * audio relabels with packets still flowing absorb exactly as before; BACKWARD jumps
                                    * unchanged (anti-stall). Fixes the AWE audio-dropout → permanent ~2.4s A/V step
                                    * (audio ahead) that internal PTS-domain metrics (lipsync=/dlvhold) could not see —
                                    * only the external source-vs-output oracle + [PTV-CHAIN] outA-V caught it.
                                    * PTV_NO_GAPDISCRIM=1 reverts; PTV_GAP_MIN_MS tunes the gap floor (default 700ms).
                                    * analysis/ptvencoder-avsync-gap-vs-splice-fix.md. ---- prior 0.8.1 (§13 hardening, review): startup-blackout budget 3x->2x preroll (worst-case ~16s not ~24s, documented); clamp deep_prime_packets to video_q-32 so the prime-wait is always satisfiable at high fps (>~68fps could exceed the queue cap ->永 time out); pre-h0 audio ring gated by g_aq_cap (256 default = byte-identical, PTV_AQ_PREROLL only when deep) so a slow-h0 NON-bursty channel stays bit-identical; documented the whole-session rung-0 lossless block + its decoder->video_q->demux->input back-pressure chain. No behavior change vs 0.8.0 on a deep-prime channel; default path now strictly byte-identical. ⚠ VALIDATION: the residual-rate-deficit failure mode drains the cushion over HOURS (8s cushion / 0.1% deficit ~= 2.2h) — a short box test FALSELY passes; soak Fintech >=3-4h to tell complete-fix (flat past ~3h) from stopgap (climbs again, envelope ~ cushion size -> needs clock recovery P3/B4).
                                    * 0.8.0 (§13 DEEP BURSTY-INPUT PRIME, opt-in per-channel): PTV_PREROLL_MS now sizes a deep startup cushion carried by video_q (not the ~1.6s frame_q cap) so HLS-segment bursty delivery (Fintech_247, Unique_TV: ~6s segment as a 1.3s burst + 4.7s gap) no longer starves the house clock into a monotonic house_skew runaway. Mechanism: the realtime-limited decoder delays its start until video_q banks ≥ PTV_PREROLL_MS worth of packets (demux fills it while waiting), and video_q is auto-sized to hold it (bounded). Cost: ~+PTV_PREROLL_MS latency on opted-in channels only. DEFAULT 350ms → 0 deep packets → BYTE-IDENTICAL (no decode delay; only bursty channels set PTV_PREROLL_MS≈segment+margin, e.g. 8000 for 6s segments, 12000 for 10s). Open risk: if a channel still climbs with a deep prime → true rate deficit (needs clock recovery, P3/B4), not a buffer.
                                    * 0.7.10 (§5.A.2 DEFAULT-ON): the shared-adj-at-own-crossing A/V-drift fix is now default ON (g_progoff_av=1) after live validation — TruBLU 13 ad-breaks eye-confirmed lip-synced (unwrap_inj flat, house_skew ≤33ms, no blowup) + Cinestar AC-3 channel 1h51m clean. PTV_NO_PROGOFF_AV=1 disables (A/B/rollback). With §5.A.1 directional threshold = the legacy-0004 single-input A/V-drift restore, complete.
                                    * 0.7.9 (§5.A.2 FIX — v0.7.8 had a live straddle blowup): dense V/A absorb the SHARED first-crosser discontinuity amount, but each still self-rebases its OWN wrap_off AT ITS OWN crossing (proven path) → no premature offset on un-crossed packets → no house_skew/aresample blowup. v0.7.8 applied prog_off to ALL packets immediately → during the V/A straddle the not-yet-crossed stream got the offset → house_skew blew up ~1372s on TruBLU live (audio destroyed). This version touches ONLY the rebase amount (shared vs own); apply path unchanged. Still zeroes V/A divergence. DEFAULT OFF; PTV_PROGOFF_AV=1, PTV_PROGOFF_DEBOUNCE_MS tunes.
                                    * 0.7.8 (§5.A.2, WITHDRAWN — straddle blowup): dense V/A shared prog_off applied to all packets; broke live. Superseded by 0.7.9.
                                    * 0.7.7 (§5.A.1 A/V-drift fix): DIRECTIONAL discontinuity-absorber threshold — FORWARD jumps default 1000ms (was 80ms), BACKWARD keep 80ms. Box-confirmed (TruBLU): the +90ms forward video-only frame-drops were being absorbed → video timeline compressed ~57ms each → audio behind ~+150ms/hr; at forward-1000 they flow through (player holds last frame, A/V aligned). Backward stays 80ms so a backward jump still absorbs → no aresample stall (v0.6.23). Knobs PTV_DISCONT_MS (forward) / PTV_DISCONT_BACK_MS (backward).
                                    * 0.7.6 (diagnostic, logging-only/byte-identical): [PTV-CHAIN] adds rawA-V (PRE-demux_unwrap source-native A/V) + unwrap_inj (= srcA-V − rawA-V) → separates source-inherent A/V drift (rawA-V grows) from demux_unwrap per-stream rebase divergence (unwrap_inj grows). The number that decides §5.A (program rebase) vs §5.B (genlock).
                                    * 0.7.5 (diagnostic): [PTV-CHAIN] traces source-content time (us) at the demux + at output emission for video/primary-audio every 10s — srcAV (input) vs outAV (output) localizes WHERE the A/V relationship diverges (source drift vs ptvencoder restamp). Logging-only/byte-identical.
                                    * 0.7.4 (diagnostic, logging-only/byte-identical): [PTV-SWRDELAY] logs the -af aresample filter's internal swr_get_delay() on the -stats cadence — the FAITHFUL resampler-slip sensor the PTS-based offset=/house_skew/sync_check-D are structurally blind to; min_hard_comp (set in ptvencoder.sh) should bound it.
                                    * 0.7.3 (P2 box-tuning from cor-1 A16 real-HW validation): (1) DELIVERY CAP default
 * 2s→3s — the real production-load hold is ~2s (TruBLU cor-1 dlvhold=2055ms; A0's 845ms underestimated), so the 2s
 * default cap-saturated (dlvforced climbing); 3s lets the precise DTS-match win (box-confirmed dlvforced 38033→0,
 * wire D −0.078→−0.057s), harmless on low-hold channels. (2) DROP-UNTIL-KEYFRAME now arms only on a LARGE jump
 * (PTV_DUKF_MIN_MS=1000) not the 80ms absorber threshold — a +90ms VIDEO-ONLY jitter blip on TruBLU spuriously
 * dropped a GOP; real splices were ≥120s so ≥1s cleanly separates them. Validated real-HW: TruBLU rode 4 real
 * ad-break jumps (+630/−120/−1461/−750s) audio-continuous no-stall (app-confirmed); AWE_Plus SCTE ad break
 * CDN-detected (clean splice). Spec: analysis/ptvencoder-p2-discontinuity-normalizer.md.
 *
 * 0.7.2 (A/V-sync redesign P2, stage 2b part 1 — DROP-UNTIL-KEYFRAME): after a detected
 * source discontinuity (ad-splice), the new timeline starts mid-GOP, so its pre-IDR P/B frames decode as a corruption
 * burst (greyed/torn) that the house clock then samples. Now drop video packets until the next IDR (AV_PKT_FLAG_KEY) in
 * demux_thread → the house clock dup-holds the last good frame across the splice = a CLEAN CUT instead of a burst.
 * Bounded by a wall-clock ESCAPE (PTV_DUKF_ESCAPE_MS, default 5s) so a no-IDR stream can't freeze the cell (the
 * session-109 28h-freeze lesson), armed FIRST-ARM-ONLY (never re-stamp the escape while armed — the re-arm slide).
 * Default ON; PTV_NO_DUKF=1 reverts. Mechanism local-validated (arms on disc / resumes at IDR / never arms on clean);
 * the real mid-GOP burst-suppression is a box property (the injected-concat boundary is keyframe-aligned). REMAINING
 * 2b: buffer-classify-keep-NEW (the interleaved-straddle whipsaw fix) — deferred pending box measurement of the
 * residual straddle glitch after 2a+dukf. Spec: analysis/ptvencoder-p2-discontinuity-normalizer.md §3.5.
 *
 * 0.7.1 (A/V-sync redesign P2, stage 2a — hybrid sparse program-offset): an ad-break
 * PTS jump used to ORPHAN the sparse copied streams (DVB-sub/teletext, data, SCTE-35) — they skip the per-stream
 * discontinuity absorber (their multi-second gaps would false-trigger it), so they got NO discontinuity offset →
 * desync by the jump, and on a BACKWARD jump (TruBlu −500s) their rebased ts<h0 → demux_pass dropped them → subs/SCTE
 * VANISHED for minutes. FIX (hybrid): track a program-level offset `prog_off` from the DENSE video reference's
 * discontinuity (the same −(delta−nominal) the video absorber applies to itself) and add it to the SPARSE streams in
 * demux_unwrap's apply step — a uniform constant shift (preserves their sparse inter-packet deltas, so NOT the v0.6.14
 * collapse) that moves them WITH the video across the splice. Dense V/A (incl. copied AC-3) keep their validated
 * per-stream self-rebase UNTOUCHED (applying prog_off to them too would double-count). SCTE-35 also rides prog_off in
 * its `pts_adjustment` rebase (0002) so the on-wire splice marker lands at the right content PTS after a jump. STAGE 2a
 * = clean-splice fix ONLY: prog_off mirrors the video absorber so it whipsaws on an INTERLEAVED straddle (trailing-OLD
 * after the first NEW pkt), and the copied-AC-3 whipsaw is untouched — both need 2b's buffer-classify (drops trailing-
 * OLD) + drop-until-keyframe. Default ON; PTV_NO_PROG_OFF=1 reverts sparse to v0.6.23 (A/B vs the whole g_discont
 * absorber). Spec: analysis/ptvencoder-p2-discontinuity-normalizer.md. Validation (gate B-copy) gated on the P1 soak.
 *
 * 0.7.0 (A/V-sync redesign P1): §7.5a POST-ENCODE DELIVERY-ALIGNMENT GATE — the
 * fftools "sync queue" the greenfield mux dropped. NVENC holds video ~0.85–0.9s (B-frames + CBR bufsize + GPU; A0
 * measured fleet-wide, encoder-caused) while transcoded audio + copied AC-3/MP2 are near-zero-latency → audio reaches
 * the muxer ~1s AHEAD of the video for the SAME content → audio-ahead-of-video on the wire → the downstream sync_check
 * (video_last − audio_last) trips → restart. The per-rung DlvGate HOLDS the dense near-zero-latency streams (transcoded
 * audio + copied AC-3/MP2) until that rung's VIDEO encode_push has emitted a DTS ≥ the held packet's DTS, then releases
 * in lockstep — A/V aligned on the wire. PTS are NEVER modified (only WHEN a packet reaches the muxer). Sparse
 * SCTE-35/subs BYPASS (their wire-arrival lead is a feature). One drainer (the rung's video output thread), many
 * enqueuers (audio threads block = back-pressure; the shared demux/copy thread drops-on-full so it never stalls the
 * input). cap_us force-release (default 2s) degrades to "audio ahead" rather than stalling on a blocked encoder; a total
 * video stall back-pressures audio (stays locked) + the watchdog owns the hang. Default ON for LIVE single-input;
 * PTV_NO_DELIVERY reverts to byte-identical-to-0.6.23 (audio direct); PTV_DELIVERY_MV gates multiview (default OFF in P1,
 * reworked in P3); PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ tune. Offline (file out) always bypasses → byte-identical.
 * Stats: dlvhold=<max-hold>ms dlvforced=<cap-releases>. Spec: analysis/ptvencoder-avsync-redesign-spec.md §7.5a + App B.
 *
 * 0.6.23: FIX multiview per-slot AUDIO STALL on a leg's source PTS DISCONTINUITY (task#23, TruBlue 2x2 cor-2: "mva a0 sound not playing" → then "whole multiview not playing"). ROOT CAUSE (reproduced + measured locally from a faithful read-only tsp capture of all 4 TruBlue legs, test-scripts/repro/trublu-disc-repro.sh + PTV_DBG_VDTS frame-in/enc-out/mux-in trace): TruBlue's ad-splice drops the program DTS BACKWARD hundreds of seconds (e.g. 523.9s→10s = −513.9s, NOT a 33-bit wrap since |Δ|<half). The composite VIDEO survives (the compositor re-stamps output to the house clock, rung_pts → immune to ANY source jump), but the source-content-anchored TRANSCODED AUDIO does not: aresample=async needs a monotonic input, so the backward leap made that slot's audio drain STALL (in0/a0 emitted 64 packets ≈1.3s then went silent for the whole run while video + the other 3 audios ran full-length). The stalled a0 then degrades/freezes the mux interleaver = the whole-grid outage. ⚠ The incident's earlier "composite VIDEO DTS corrupted by the mpegts MUXER" attribution was WRONG: the `[mpegts] Packet corrupt / DTS out of order / Invalid timestamps stream=0` lines are INPUT DEMUXER warnings (demux.c:589/1006/1468 — the [mpegts] ctx is the in0 INPUT demuxer), benign (the corrupt splice packet is dropped); the composite video reaching the muxer is provably clean (rung_pts, monotonic, verified at frame-in/enc-out/mux-in). FIX: demux_unwrap's discontinuity absorber (v0.6.0, which re-bases a >g_discont_ms forward jump to last+nominal for continuous V/A streams) now also absorbs BACKWARD jumps (delta < -thresh, still NOT a full wrap which delta<-half catches) — the same re-base formula (wrap_off -= delta-nominal) maps the new ts to last+nominal for either sign. So the audio resampler, the compositor h0/skew math (also kills a catastrophic lag=513880ms blowup), and any copy stream all stay on one continuous monotonic timeline across an ad-splice. demux_unwrap is per-source-packet (runs once, before the video/audio split + copy fan-out) so the fix covers every consumer. VALIDATED local (Rule 0, symptom moved): trublu fast (backward disc) a0 64→1821 pkts continuous; trublu full 8min real-timing (in2 wrap @24s + in0 forward disc @7.5min + backward jumps) all audio continuous @1.00x; clean grid4 0 spurious absorbs; single-input byte-identical (0 absorbs; the path I changed never executes on a clean source — proven by a stashed-binary A/B); copy-AC3 mv 0 backward DTS / 0 EINVAL. SUBTITLE/DATA still skip the absorber (sparse, v0.6.14); the 33-bit wrap branches unchanged. ptvencoder.c-only → folds into v2 patch 0001. 0.6.22: NOISE-ADAPTIVE acquire threshold — eliminate the residual ~4/min micro-acquires that v0.6.21's 12s refractory bounded but didn't zero on very-jittery-source legs (box: GBNews/Curiosity/PureFlix/REVn carry vlag ±300–560ms source-PTS jitter; the fixed 40ms threshold sits below their offset noise floor → an acquire every refractory window forever). FIX: track each leg's offset jitter (pll_dev = slow EMA, shift 9 τ≈11s, of |off−pll_ema|, seeded 0) and set the effective acquire threshold thr = max(g_pll_acquire_us, g_pll_noise_k·pll_dev) capped 1.5s. dev≈0 at startup → thr=40ms → the big DC bank acquires immediately; then dev ramps to the leg's noise → thr rises above it (e.g. ~450ms for ±150ms jitter) → steady-state noise can't re-fire. Clean legs (dev≈0) keep the exact 40ms tightness (RAV/dead-band win preserved). Knobs PTV_PLL_{NOISE_K=3 (0 disables),DEV_SHIFT=9}; [PTV-AVSYNC] pll[…] gains dev=Nms (per-leg noise, visible without PTV_DIAG). Local A/B (PTV_PLL_TESTNOISE_MS): pure ±150ms noise → acquires collapse to ~startup-only then 0 (vs 0.6.21's 4/45s); induced 533ms bank still acquires once → offset 0. MULTIVIEW-PLL only; clean/single/copy untouched. 0.6.21: FIX the PLL acquire SELF-EXCITED LIMIT CYCLE on jittery NTSC legs (v0.6.20 fleet rollout regression — TruBlue/PureFlix grids "didn't play"; the video pipeline was PROVABLY CLEAN dup=0/qdrop=0/1.00x, so it was an audio-loop fault). ROOT (proven by box PTV_DIAG + a PTV_NO_AVSYNC_PLL A/B): on legs whose measured offset carries ±100–200ms noise, the v0.6.19 40ms threshold tripped the acquire ~every 7s, ALTERNATING drop↔pad (acq=92–167, ~9s drop + ~9s pad), because each acquire's own drop/pad perturbs the next vring measurement → re-triggers = a limit cycle. The A/B nailed it: with the PLL OFF the same leg's offset jitter was only ±35ms (vs ±165ms with it on → self-excited ~5×), AND the slots had REAL frozen startup banks (a0 −1334ms, a1 −967ms audio-late) that open-loop B1 leaves uncorrected → the acquire is NEEDED, it just must not chase the residual. FIX (damp, keep the acquire): (1) HARD refractory g_pll_refractory_us=12s after any acquire (was conflated with g_pll_acquire_n ≈0.68s — far shorter than the ~7s thrash period) → bounds acquires to ≤1/12s regardless of noise spectrum; (2) slower offset EMA shift 5→7 (τ≈2.7s) → averages the zero-mean ±150ms noise below the 40ms threshold so only the DC bank triggers. Net: snaps a real startup bank ONCE, then leaves the ±35ms residual to the glitch-free TRACK. MULTIVIEW-PLL only; PTV_PLL_{REFRACTORY_MS,EMA_SHIFT} override; PTV_PLL_TESTNOISE_MS injects a ±N square wave to reproduce the limit cycle locally (local sources are clean). Bigger INPUT buffer RULED OUT: not bursty starvation (no dup/drop/adrop/skew-runaway); the banks are async startup over-production and the jitter is a control-loop instability. 0.6.20: FLIP the B3 closed-loop A/V PLL to DEFAULT-ON (g_avsync_pll 0→1) after the v0.6.19 box A/B PASSED on BOTH boxes: cor-2 RAV 2x2 (banked tmtg) all 4 slots converged within ±17ms (a1's frozen bank acquired drop 533ms→offset +0), `lipsync==−offset` faithful, 1–2 acquires/slot no thrash, 1.00x, 0 backward/EINVAL; live-transcoder grids (clean multiview, hours) NO-REGRESSION — offset ±3ms, TRACK glitch-free follows small vlag drift, acq≈1 over hours, 1.00x, 0 errors. Revert env added: PTV_NO_AVSYNC_PLL=1 → open-loop B1 follow (the env parse had NO disable path before; PTV_AVSYNC_PLL=1 stays an implicit no-op = the new default). MULTIVIEW transcoded audio ONLY; single-input + copy/sparse (DVB-sub/SCTE-35/copied-AC-3) BYTE-IDENTICAL regardless of the flag. The always-on [PTV-AVSYNC] health line (lipsync/offset/vlag/alag) is gated on -stats, NOT PTV_DIAG, so operators see the faithful A/V number on the -stats_period cadence without any diag env; only the verbose [PTV-PLL]/[PTV-DIAG] lines need PTV_DIAG. ⚠ point any A/V-health monitoring (sync_check.sh) at the `offset=` field. 0.6.19: (1) FIX the always-on [PTV-AVSYNC] `lipsync=` headline for the B3 PLL path — it was the async_pad span estimate (outspan−content−lag_true), which does NOT account for the PLL's content drop/pad retiming (af_applied_us), so on a CONVERGED slot it kept reporting the bank the acquire had already removed (lipsync ≈ applied → e.g. RAV box read lipsync −258/−812/−1665/−504ms while the faithful vring-paired `offset` was −8/−85/+3/+4ms and the eye was clean). Operators (and any monitoring/sync-check parsing the field) read "off" on a synced channel. FIX: headline the FAITHFUL measured offset (−av_offset_us; sign: offset<0 = audio late ≡ lipsync>0 = audio late) whenever the vring has paired, fall back to the span estimate only before it pairs (offset=--). Logging-only, BYTE-IDENTICAL. (2) TIGHTEN the PLL dead band: lower g_pll_acquire_us 100→40ms so a STABLE sub-100ms residual gets snapped in by a whole-frame acquire instead of stranded (TRACK is guard-limited on jittery NTSC sources → can't trim it; box a1 sat at −84ms: 84<100 threshold so acquire refused, TRACK too weak). The flatness debounce (threshold/4 = 10ms) still rejects jitter so converged (|ema|<40) and wandering slots don't churn. Plus acquire quantization now rounds to the NEAREST whole frame (was truncate-toward-0) → residual ≤½ frame (~11ms) not ≤1 frame; a1's −84ms → one acquire → ~−1ms (well inside the ±25ms gate). MULTIVIEW-PLL only; PTV_PLL_ACQUIRE_MS overrides. 0.6.18: B3 PLL fix — acquire on ANY stable large offset (drop the startup-window/disturbance-event gate). tmtg RAV box A/B of 0.6.17 (PTV_DIAG): the three FAST-forming banks acquired in the first ~3s and converged (offset −32/−42/−89ms), but a slot's SLOW-forming +1.1s bank stabilized AFTER the 5s startup window with no disturbance epoch → the gate refused to acquire it → stuck at −1100ms. ROOT: big banks accumulate slower than the 5s window; the startup/event gate (v0.6.17 N4) is too restrictive. FIX: `may_acq` drops the `((in_startup && acq<k) || armed)` clause — fire whenever |pll_ema|>threshold AND stable (the stability-debounce already rejects noise; the refractory throttles; converged slots' sub-threshold residuals won't re-fire). A frozen bank now converges in 1–2 refractory-throttled acquires regardless of when it forms. The startup-window/disturbance-epoch machinery (g_pll_startup_us, g_pll_acquire_k, the atomic per-input epoch + its compositor/demux writers) is now DORMANT (left in place, harmless). KNOWN residual (deferred): the ~30–90ms post-acquire residual on the fast slots is TRACK-limited — the monotonic guard fires on most frames of these jittery NTSC sources so the conditional-integration anti-windup skips the integral; acquire (content drop) is the reliable lever, TRACK (PTS nudge) has weak authority here. 0.6.17: A/V PLL redesign Phase B3 — CLOSED-LOOP two-regime A/V controller for MULTIVIEW transcoded audio (PTV_AVSYNC_PLL, default OFF for box A/B). ROOT (measured fleet-wide on tmtg-cor-transcoder-2 via the always-on [PTV-AVSYNC] vlag/alag split): every multiview slot banks a CONSTANT per-slot audio-late offset at startup (alag a frozen +0..+2100ms STEP — GAFamily +0, PureFlix +499, RAV +2034, GBNews +2100 — NEVER drifting; vlag never drifts ⇒ audio-side, not a video regression). aresample=async over-produces over the startup gap and the v0.6.8 monotonic guard FREEZES it; the open-loop audio-follow steers applied→house_skew (the VIDEO's lag) and is structurally blind to the bank (applied=house_skew, trk≈0, yet offset −61..−2163ms). A pure legacy-style slow integral is the WRONG tool for a 2s STEP (210s to unwind at 10ms/s). FIX: close the loop on the faithful measured av_offset_us (=vlag−alag) with a two-regime controller, BOTH on the B1 content-anchored base (want=opts+applied) so the acquire's content-drop and applied step CANCEL in want (guard never sees a backward step): (1) ACQUIRE — one-shot drop(advance)/pad(delay) sized to the FROZEN bank, snaps it out in one tune-in skip; stability-debounced (large AND flat, so Δ sizes the frozen — not still-forming — bank), startup fires ≤k times then mid-run only on a disturbance epoch (atomic, bumped by slate-return in compositor_thread + discont absorber in demux_unwrap — two writers); bumpless EMA credit (pll_ema−=Δ) + refractory so it never re-fires/over-drops. (2) TRACK — type-1 integral trim (step=ema·frame/τ, rate-clamped to g_af_rate_us, no dead zone), conditional-integration anti-windup vs the guard; type-1 suffices since the disturbance is a STEP (no ramp ⇒ no r·τ residual). Drives offset→0; gate is the EMA-SMOOTHED offset (instantaneous = vlag jitter, which the loop rides at the mean, not chases). MULTIVIEW transcoded audio ONLY; single-input + copy paths BYTE-IDENTICAL/unchanged (flag-gated; copy stays on the ≥0 floor + monotonic clamp). Knobs: PTV_PLL_{EMA_SHIFT,TAU_MS,ACQUIRE_MS,ACQUIRE_N,STARTUP_MS,ACQUIRE_K}. [PTV-AVSYNC] gains a pll[ema/applied/acq/guard/drop/pad] view; [PTV-PLL] logs each acquire (PTV_DIAG). 0.6.16: the always-on [PTV-AVSYNC] line now leads with `lipsync=±Nms` — the faithful pipeline-introduced lip-sync error (audio's realized output-vs-content lag async_pad − video's TRUE lag lag_true; + = audio late), i.e. the PTV_DIAG [PTV-LIPSYNC] `err` folded into the no-flag stats line (operator's headline A/V number). `offset` (vring-paired) stays as the independent cross-check (lipsync>0 ≈ offset<0). The old multiview actuator-residual field `err=` is renamed `trk=` to avoid two "err"s. Logging-only, byte-identical. 0.6.15: FIX — multiview per-slot audio banked 0.3–2.5s LATE at startup (tmtg-cor-transcoder-2; "sound completely off"). ROOT (reproduced local from real tmtg RAV captures via tsp UDP, traced with PTV_ATRACE): h0 was anchored in the DECODE thread to the first DECODED frame, but under a deep startup jitter-buffer prime the compositor's first DISPLAYED frame is a different/later content → the displayed video leaps ~prime-depth AHEAD of h0 at tick 0 → P2 (v0.6.3 h0 re-anchor) shoves h0 forward → the transcoded audio's opts steps backward → the v0.6.8 monotonic guard advances out_a and FREEZES a permanent +Δ bank; a COPIED audio track's DTS would jump backward into the clamp (freeze; historically the EINVAL no-data outage — d2460f4180/v0.2.3, 9b1f3843f9/v0.2.1). FIX: anchor each slot's h0 at the COMPOSITOR'S FIRST DISPLAY (h0 = disp_src − tick·tick_dur) so sk=0 from the start and P2 never fires — audio + copied tracks anchor to the SAME h0, nothing banks or clamps. MULTIVIEW ONLY; single-input keeps the decode-thread anchor (BYTE-IDENTICAL). PTV_NO_H0_AT_DISPLAY=1 reverts (A/B). Doesn't touch buffer depth/cushion/latency (B2 "deep prime helps" intact) — distinct from the v0.5.9 per-frame display clamp that stuttered. Verified local: tmtg4 alag→0 all slots, copied AC-3 monotonic (no clamp), grid4 + single-input + single-input-AC-3-copy unchanged. PTV_ATRACE = temp per-audio-frame B1 trace (default off, byte-identical). 0.6.14: FIX — copied SPARSE streams (DVB subtitles, SCTE-35 data) were thrown OUT OF SYNC by the v0.6.0 discontinuity-absorber. demux_unwrap's forward-DTS-jump absorber (g_discont, re-base a >PTV_DISCONT_MS=80ms jump to last+nominal) ran on EVERY stream. Continuous video/audio are ~1 frame apart so only a real glitch trips it — correct. But sparse subtitle/data streams have NATURAL multi-second inter-packet gaps (DVB-sub events seconds apart, SCTE-35 ad markers minutes apart), so essentially EVERY packet's gap exceeds 80ms and got "absorbed" → the sparse timeline COLLAPSED (subtitle inter-packet deltas crushed to the 20ms nominal, whole stream shifted) → subs drift out of sync / never paint; ad-marker positions would shift too. ROOT-CAUSED by A/B on cinestar_src_5min.ts (4 DVB subs): default ON crushed deltas to 0.02s; PTV_NO_DISCONT=1 preserved the source deltas exactly. FIX: gate the forward-jump absorber to ct==VIDEO||ct==AUDIO; SUBTITLE/DATA skip it. The 33-bit WRAP branches (delta<-half / >half) STILL apply to every stream (copied AC-3/SCTE-35 across the 2^33 roll — the v0.2.1 reason). Verified: default-ON output now reproduces source subtitle deltas bit-for-bit, video/audio discont-absorb unchanged. 0.6.13: FIX — CLI -metadata:s:s:N / -disposition:s:N on COPY streams (subtitles, extra-audio, data) was IGNORED. The transcoded-audio path applied per-stream CLI metadata/disposition (apply_stream_meta, type 'a'), but the copy/passthrough loop only carried the SOURCE language+disposition and never called apply_stream_meta → -metadata:s:s:N language=/title= and -disposition:s:N silently did nothing (multiview operators could name audio views but not subtitle views). FIX: the copy loop now calls apply_stream_meta(tlet, tidx) per copy stream, with the type letter from the source codec_type (a/s/v/d) and a per-type output index seeded past the transcoded streams (1 composite video + n_audio transcoded audio; subs/data start at 0), incremented in stream-creation order = FFmpeg's -metadata:s:<t>:N numbering. So copied subtitles now honor -metadata:s:s:N (title/language override) + -disposition:s:N, exactly like audio. Source language/disposition stays the default when no CLI override is given. 0.6.12: A/V-SYNC STATUS in STATS (always-on, single-input incl., §8). The MEASURED A/V offset (offset=vlag−alag, − = picture ahead of audio) is now computed on EVERY channel (the [PTV-AVSYNC2] video-output ring + the audio-drain pairing are no longer gated by PTV_AVSYNC_PROBE — they were proven negligible-cost on the box, "probe on ALL channels", speed=1.00x) and printed on the -stats_period cadence at AV_LOG_INFO via the [PTV-AVSYNC] line, so operators running `-stats_period 10` at info level SEE the real lip-sync number live without any env flag. The line now prints for SINGLE-INPUT too (was multiview-only): single-input shows offset + vlag/alag + house_skew (its house-clock lock state); multiview additionally shows the per-slot actuator state (applied/err/nudge/acq). offset=-- until the first audio frame pairs against the video ring. The verbose [PTV-AVSYNC2] decomposition (vlag/alag base/dev + ring + pairδ) stays opt-in behind PTV_AVSYNC_PROBE for deep diagnosis. Fulfills the plan §8 commitment ("the [PTV-AVSYNC] status line gains the measured offset ... single-input included"). Output BYTE-IDENTICAL (logging-only). 0.6.11: built-in discardcorrupt now covers ALL streams (was video-only) so dropping the CLI -fflags +discardcorrupt is a zero-regression swap — and unlike the CLI flag (which makes libavformat discard SILENTLY inside av_read_frame, hiding the count), the built-in COUNTS video corrupt drops in the `corrupt=` stat. RECOMMENDATION: remove -fflags +discardcorrupt from the command line and rely on the built-in (default ON; PTV_KEEP_CORRUPT=1 reverts). 0.6.10: FRAME-LOSS in STATS + -fflags +discardcorrupt. Surface the two frame-loss sources that cause the multiview leap, on the -stats_period cadence (so operators SEE it): per-input qdrop (video_q overflow = vdrop) + corrupt (demux AV_PKT_FLAG_CORRUPT discards + decode AV_FRAME_FLAG_CORRUPT drops). Multiview: appended per-input to the compositor stats line (" in0:qdrop=N/corrupt=M …"); single-input: "qdrop=N corrupt=M" on the progress line. Plus g_discardcorrupt (default ON, = -fflags +discardcorrupt): the demux now COUNTS and DROPS corrupt video packets before decode (DemuxArgs.vcorrupt) — a corrupt frame, like a dropped one, becomes a content gap the position-anchored composite leaps across; discarding early + counting makes it observable. PTV_KEEP_CORRUPT=1 keeps them (prior behavior). 0.6.9: Phase B #1 — deeper video_q (48→256) to STOP startup frame loss = the multiview leap ROOT. Box (multicast, realtime source) dropped ~30 video frames in the first ~1s (vdrop spiked then went flat; holddrop=0; [PTV-DISCONT] NEVER on the video stream → NOT a PTS discontinuity, the absorber can't help) because the decoder's init window produces nothing while the realtime source fills the 48-deep video_q → overflow drop → a content GAP → the position-anchored composite video LEAPS to the newest content → audio left behind = ~600ms-1s per-slot PICTURE-AHEAD (the P2/REANCHOR2 + audio-late residual all trace to this). Single-input is immune: content-anchored video turns a dropped frame into a harmless output PTS gap (A/V stay aligned) — which is ALSO why the legacy single-input PLL never saw this. FIX: video_q 48→256 absorbs the one-time decoder-init backlog (decoder then drains it faster than realtime and catches up; steady-state near-empty); drop-newest stays the backstop for sustained overload. PTV_VIDEOQ overrides. Validate on the relay: vdrop→0, no content leap, [PTV-AVSYNC2] offset stays ~0 (no P2 storm, no audio-late). 0.6.8: B1 MONOTONIC GUARD (fix box stall). B1 set multiview audio out = opts + offset, where opts (async/buffersink output pts) STEPS BACKWARD when h0 is re-anchored forward (P2: opts = buffersink − h0_samp, larger h0 → smaller opts) or at a source PTS discontinuity (box live feeds throw +hundreds-of-ms jumps). The pre-B1 free counter was monotonic BY CONSTRUCTION; content-anchoring lost that → backward out_a → libfdk_aac "Queue input is backward in time" + mpegts "non monotonically increasing dts" → that audio stream stalled and the mux interleaver WEDGED (muxed frozen, frame_q full → grids sent NO data; box-observed at startup, ~−785ms backward step). FIX: keep out_a monotonic + frame-spaced (af_last_out guard); on a backward step it advances at nb (dense, like the old counter) until opts recovers. Local missed it (built-in aac not libfdk_aac; file not UDP; P2 non-deterministic). 0.6.7: A/V PLL redesign Phase B (B2) — track vlag faster. B1's content-anchored offset tracked the per-slot lag at only 2 ms/s, so a slot whose source is momentarily slow at startup (its cell dups → vlag ramps ~6–8 ms/s and settles at e.g. +200 ms) had the audio converge over ~100 s (≤~140 ms audio-ahead meanwhile). REVISED ADR: this residual is SOURCE-SLOWNESS dups, NOT the deep prime — and post-B1 a deep prime HELPS (its cushion absorbs slowness, keeping vlag=0 longer; a shallow prime underruns sooner). So "bound the prime" was backwards. FIX (B2): raise the follow-rate ceiling g_af_rate_us 2→10 ms/s so the smooth content-anchored offset tracks the dup ramp in near-real-time; 10 ms/s ≈ 1% (under the ~2% audible budget), engaged only transiently during convergence (steady-state step=gap is tiny). PTV_AF_RATE_MS_S overrides. 0.6.6: A/V PLL redesign Phase B (B1) — CONTENT-ANCHORED multiview audio. Phase A's [PTV-AVSYNC2] probe localized the per-slot desync to the AUDIO side: the multiview audio-follow emitted on a FREE-RUNNING sample counter (af_next_pts) that banked aresample=async's STARTUP over-production into a permanent audio-late offset (alag +400..+1252ms, the dominant term; vlag≈0). Single-input was immune because content-anchored (out=opts, async's self-correcting target). FIX (B1): multiview audio now also CONTENT-ANCHORS — out = opts + a smooth rate-limited offset (≤PTV_AF_RATE_MS_S) that tracks the compositor's per-slot lag (house_skew) so audio follows the video DISPLAY, seeded to the current lag at the first frame (no glitch). No free counter, no drop/pad/silence/acquire. Makes alag→house_skew=vlag ⇒ measured offset→0. Converges multiview audio onto the single-input mechanism (unification down-payment). MULTIVIEW ONLY; PTV_AF_NO_ANCHOR=1 reverts to the pre-B1 free-counter path (A/B); single-input BYTE-IDENTICAL. Validate via [PTV-AVSYNC2] on the SRT relay: offset→~0 on the deep-prime slots, clean slots unchanged. (B2 = bound the deep video prime; B3 = thin closed-loop trim on the measured offset — to come.) 0.6.5: A/V PLL redesign Phase A — READ-ONLY measurement probe [PTV-AVSYNC2] (PTV_AVSYNC_PROBE=1; analysis/ptvencoder-avsync-pll-redesign-plan.md). Measures the REAL per-track lip-sync offset out_v(C)−out_a(C) by pairing each emitted audio frame's source content C against the output time the VIDEO showed that SAME content (a per-input ring of (displayed abs-src → out_v), written by the compositor in multiview / the master output thread in single-input). Reports offset + the video_lag/audio_lag split (§3.2a: which side moved, with slow EMA baselines + deviation) + the content pairing residual (§3.2b). Faithful where the old proxies (async_pad/house_skew/lip-sync err) were confounded for the audio-follow path, because out_a is the ACTUAL emitted pts (af counter+nudge in multiview, opts in single-input). NO actuator — measures only; gated PTV_AVSYNC_PROBE, prints on the -stats_period cadence. Single-input subtracts house_skew from the audio content key (single-input injects it at the graph input). M-b cross-check = "offset ≈ 0 on a clean synced source" (local clean run), not separate code. Purpose: validate the measurement against the box eye (mvb ~1s, Fé drift) BEFORE building the closed-loop controller (Phase B). Zero behavior change when off; BYTE-IDENTICAL output. 0.6.4: multiview audio-follow ACTUATOR upgrade (P1) — glitch-free smooth tracking. The v0.6.2/0.6.3 audio-follow corrected the per-slot lag only with whole-frame drop/pad on a 40ms threshold → it tracked the slow per-slot drift (source-vs-house clock, measured +360ms/35min on a live slot) in discrete ~40ms hops (box symptom: "A/V sometimes off then OK again") and left a ±frame residual ("audio slightly behind" on re-anchored slots). FIX: two-mode per-slot controller in audio_drain_fg — keep the fast discrete drop/pad ONLY to ACQUIRE a large gap (>PTV_AF_ACQUIRE_MS, default 100; startup ramp / big jump the smooth rate can't catch), and otherwise TRACK the residual+drift with a SMOOTH rate-limited (≤PTV_AF_RATE_MS_S, default 2ms/s — imperceptible) sub-sample PTS nudge (af_nudge_us added to the continuous output counter; no silence inserts, no content drops, monotonic since rate≪frame). Reuses the legacy 0007 PLL's rate-limited actuator idea, per slot, driven by the directly-measured lag (NO jump_comp). MULTIVIEW ONLY; PTV_AF_NO_PLL reverts to pure discrete; single-input BYTE-IDENTICAL. Verified local SRT relay (2x1+4-up): zero discrete hops after startup (smooth nudge tracks the drift), transcode↔copy xcorr +48ms→+21ms (q=0.994), 0 clamp, all healthy. Also adds the always-on [PTV-AVSYNC] per-slot status line on the -stats_period cadence (obeys -nostats, NOT gated by PTV_DIAG): lag (cell's measured video-vs-house offset) / applied (audio retiming in effect) / err (lag−applied residual, ~0=tracking) / nudge / cumulative acquire work. 0.6.3: multiview per-slot h0 RE-ANCHOR — floor each slot's lag to ≥0 so a cell is never displayed AHEAD of the house clock, which (a) is physically wrong for a frame-synchronizer and (b) is UNCORRECTABLE on a COPIED audio track (a copy can only be delayed, not advanced — a backward DTS hits the monotonic clamp). ROOT (P0, local SRT relay): a slot's video leaps far ahead (−560ms on a 2x1, up to −2.5s on a 4-up) from an anomalous first decoded frame and/or a deep startup buffer prime (the open-join barrier lets fast decoders over-fill their jitter buffer while waiting for the slowest input). FIX: when a slot's lag drops below −PTV_H0_REANCHOR_MS (default 120), re-anchor its h0 forward (h0 += deficit) so the lag lands at +1 tick (slot reads slightly BEHIND = the normal buffered state): video display unchanged, transcoded audio rides the same h0+house_skew (stays locked), copied audio now only needs to DELAY → correctable. MULTIVIEW ONLY (n>1); PTV_NO_H0_REANCHOR reverts; PTV_H0_REANCHOR_MS tunes. VERIFIED local SRT relay (real POP TV + HD History + the 4 Grid feeds): 4-up lag −2500ms→≥~0; transcode-path(drop/pad) vs copy-path(h0-rebase) of the SAME source agree to +48ms (xcorr q=0.987) ⇒ copy tracks video as well as the box-confirmed transcode; no muxer clamp storm; single-input BYTE-IDENTICAL. (PTV-LIPSYNC err is confounded for the audio-follow path on jumpy sources — the xcorr path-consistency check + box eye-check are the oracles.) 0.6.2: AUDIO-FOLLOW now CONTINUOUSLY RE-TRACKS (was one-shot). ROOT (box+local-SRT-relay confirmed via PTV-LIPSYNC trajectory): a slot's per-slot video lag RAMPS IN over ~30s (in1: 0→+320ms then rock-stable) but the one-time latch fired at t≈1s while lag was still ~0, latched 0, never re-tracked → that slot's audio left permanently ~the steady lag AHEAD of video (~1s on the box; mvb/HD-History symptom). FIX: compositor keeps a slow EMA (~1.3s, smooths ±100ms interlaced-PTS jitter) of each slot's measured lag and publishes it every tick instead of freezing one early value; the drain's existing >40ms-threshold deterministic drop/pad re-tracks it (lag>0 pad/delay, lag<0 drop/advance) through the startup ramp and any later drift, and stays put once settled (no churn). Multiview-only, gated n_input>1 && g_audio_follow; single-input BYTE-IDENTICAL. PTV_NO_AUDIO_FOLLOW reverts. 0.6.1: multiview per-slot A/V FIX = AUDIO-FOLLOW (Option A). ROOT (confirmed: single-input synced on the SAME live feed, multiview not): composite video is POSITION-anchored (rung_pts; one shared frame can't be content-stamped per cell) while audio is CONTENT-anchored (src−h0); at the join they sit on different origins → stable per-slot offset = "audio behind video". Single-input has no split (video=(src−h0)/tick, same h0 as audio) → untouched. FIX (multiview only, n_input>1): compositor latches each slot's STABLE signed offset (avg past the lossy join, re-latch on outage); the slot's audio applies it as a ONE-TIME deterministic correction — DROP content (advance) or PAD silence (delay) on a continuous gapless output counter — landing audio on the video's displayed-content clock. Deterministic because aresample=async is far too slow (~20ms/s) and can't advance. PTV_NO_AUDIO_FOLLOW reverts. 0.6.0 discont-absorber: default ON (helps copy/sparse streams); 0.5.9 clamp: DEFAULT OFF (stuttered; opt-in PTV_MV_CLAMP). 0.6.0: FIX multiview per-slot audio-late = per-input source PTS-DISCONTINUITY ABSORBER. ROOT CAUSE (live-repro confirmed via SRT relay of the real 1080i50 Grid feeds): the source throws a forward PTS jump of a few hundred ms to ~1s at the live join (in1 +480 / in3 +960) while FRAMES stay continuous (one per tick, buffer full, holddrop=0) — a timestamp glitch, NOT lost frames. Raw, it shifts that slot's content→output mapping so the cell video leaps ahead of its (continuous) audio = per-slot "audio behind video", stable for the whole run. FIX: demux_unwrap now also absorbs an arbitrary forward DTS jump (>PTV_DISCONT_MS, default 80) the same way it absorbs the 33-bit wrap — a per-stream offset re-basing to last+nominal, keeping video+audio+copy on one continuous timeline → lag→~0, video stays SMOOTH. The 0.5.9 content CLAMP is now DEFAULT OFF (it stuttered on jittery sources; opt-in via PTV_MV_CLAMP). PTV_NO_DISCONT reverts. 0.5.9: (superseded) house-clock content clamp. ROOT CAUSE (box+local confirmed): a startup/source PTS gap (skipped/corrupt frames during decode startup) makes a cell's video content leap AHEAD of the house clock (lag_true −480..−900ms, audio innocent); the compositor showed the jumped frame immediately → video raced ahead of its (continuous) audio → per-slot "audio behind video". FIX: compositor displays a frame only once the house clock reaches its content-time (disp−h0 <= out+1tick); an ahead-of-clock frame is held one tick at a time → the cell's video freezes across the lost-frame gap (correct for lost video) while audio continues → A/V stays locked, lag→~0. Compositor-only; audio path untouched. Validated locally vs the flash+beep ruler (startup-gap repro: slot +354ms→~baseline). PTV_NO_MV_CLAMP=1 reverts. 0.5.8: [PTV-START] per-tick startup trace (first ~3s, per slot: h0/srcpts/content-age/output/qd) to SEE how the per-slot video-lead onsets — box showed holddrop=0 (NOT jitter-buffer drops) yet video leads the house clock, so the cause is a startup PTS irregularity, not a buffer. 0.5.7: compositor mv line adds per-slot /holddrop= (per-input jitter-buffer drop-oldest count) to confirm the multiview audio-late ROOT CAUSE measured on the box: per-slot VIDEO leads the house clock by a fixed startup offset (lag_true −640/−800/−900ms, mvb 0; audio innocent: house_skew=0 async_pad=0). Hypothesis: early-decoding feeds overflow the 48-frame hold at startup → drop-oldest skips that cell's content ahead → displayed video leads. holddrop>0 on the leading slots at startup confirms it. 0.5.6: PTV-LIPSYNC diag — faithful per-slot pipeline-introduced lip-sync error err=async_pad−house_skew (REPLACES the misleading av_off, which was production-thread buffer lead, not playback sync: +3.5s offline / −0.2s live for identical IN-SYNC content). Compositor mv line adds per-slot /lag= (TRUE uncapped video lag) so 250ms-cap saturation is visible (lag>>skew = audio can't follow). Cross-validated locally vs the flash+beep ruler (grid4-sync.sh): healthy err≈0 == ruler in-sync. Gated PTV_DIAG. 0.5.5: FIX per-slot audio-late in multiview — audio arriving before a slot's video sets h0 was DROPPED, so the slot whose video is slowest to acquire its first frame lost the head of its audio (first_out up to ~1s = the per-slot "audio delayed"; box-confirmed on Grid_2x2). Now buffer pre-h0 audio (bounded ring) and replay it once h0 is known, keeping content>=h0 -> first_out~0 all slots. ADIAG probe retained to verify. 0.5.4-adiag (diag, temporary): PTV-ADIAG per-track audio-vs-video offset (av_off) + g_vout_us — MP2 tracks land ~1s audio-late from aresample=async startup over-production (first_out~16ms aligned; av_off ~+1s for high source-V-A MP2 inputs). Gated PTV_DIAG; remove after fix. 0.5.3: code-review fixes — og_spec buffers sized for "disposition" (was silently truncating -disposition/-disposition:a); -an now honored (suppresses auto-selected audio + audio copy in the no-map path); accurate -h (dropped never-wired -s/--deint/--hw/--mode); fifo alloc NULL-checked; g_muxed/g_muxed_bytes atomic (stats data race). 0.5.2: Option F COARSE half — re-anchor on return-from-outage: clear a slot's accumulated dup skew when it comes back after a black-slate, so a continuous-PTS feed re-syncs A/V on return (the source's advanced PTS lands at current output once stale skew is gone). Safe (outage>=slate exceeds cleared skew -> async input still forward). PTV_NO_REANCHOR=1 for A/B. 0.5.1: Option F skew made NON-DECREASING+capped (async input requires monotonic; a decreasing skew stalled the mux — 20fps-into-25fps repro). Fine-half only: rising dup drift rides async; the DECREASING re-anchor-on-return (full A/V re-sync after outage) needs a separate direct output-offset path (coarse half, TODO). 0.5.0: multiview A/V sync = software frame-synchronizer (Option F). Compositor publishes the MEASURED per-slot output-vs-content skew = out_time-(displayed_src-h0) of the frame each cell actually shows; the slot's audio rides it (existing async path), so A/V stays locked through jitter/drop/dup/interruption and RE-SYNCS on input return (shows current content). Reduces to ~0 on healthy 1:1 inputs (no regression). Replaces the dup-event counter (0.4.1), which missed drops/returns. Copy path protected by its monotonic-DTS clamp. 0.4.1: multiview per-slot audio skew = dup-event counter (was arithmetic+non-decreasing, which locked startup jitter -> later-priming slots' audio over-delayed); per-slot skew in PTV_DIAG. 0.4.0: MULTIVIEW (1/2/4-input mosaic — house-clock compositor, per-input jitter buffer + clock, per-slot audio/sub, parallel open); 0.3.0: multiple transcoded audio tracks + per-track -ac/-filter:a/-metadata + source fan-out; 0.2.3: monotonic-DTS clamp on copy path; 0.2.2: no -r preserves source FRAME rate (avg); 0.2.1: 33-bit PTS-wrap on copy-passthrough */


/* Diagnostics (env PTV_DIAG=1): per-second stage counters + slow-call
 * breadcrumbs to localize a stall. Temporary, gated, low-overhead (Rule 0). */
```

## Appendix — architecture sketch & version-string rationale (moved from the ptvencoder.c header, 2026-07-03)

Pull-based multi-stage pipeline (the professional model):

```
  demux ─video_q─▶ decode (free-run) ─frame_q─▶ output(master clock, sample
        ─audio_q─▶ audio (decode▶resample▶AAC) ──────────────────┐  & hold)
                                                         mux_q ◀──┴──▶ mux
```

- A free-running output clock is the master: a wall-paced timer in the output thread emits at
  the house rate no matter what upstream does.
- The frame synchronizer is sample-and-hold: decode runs free in its own thread and keeps the
  latest decoded frame current (frame_q, drop-oldest); each output tick samples it — repeat if
  decode is behind, drop intermediate frames if it is ahead. Source PTS is advisory; video
  output PTS is the tick counter, so source wrap/jump/gap is invisible to the output (no
  re-anchor needed — the pull model dissolves it).
- The encoder is downstream and never allowed to block the clock from draining input: if it
  stalls (e.g. NVENC blocking the caller under GPU load — the failure this design exists to
  survive), decode keeps running, the demuxer keeps draining the socket (dropping on a full
  queue), and a watchdog flags the stall.
- The mux is clock-locked and keeps emitting (dup-fill) so the TS never stops.
- Video and audio map onto one shared input anchor (h0) for the A/V start offset.

`PTVENCODER_VERSION` is ptvencoder's OWN version, independent of the FFmpeg git-describe string
(which, on the BtbN box build, reflects fresh-upstream + `git apply` and so does NOT encode which
patch revision is applied). Bump it by hand on each release so a deployed binary self-identifies
via the banner / `-version`.
