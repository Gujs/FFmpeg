# ptvencoder version history

Per-release notes, extracted verbatim from the `ptvencoder.c` header on 2026-07-03
(the in-code block had grown to ~190 comment lines). **Add new release notes HERE,**
keep only the current `PTVENCODER_VERSION` define in the source. This file is part of
the v2 `0001` patch (additive, travels with the source to the build box).

## 1.2.0-pre1 (2026-08-08) — CC (EIA-608) → DVB-teletext

**New capability, so a new minor line.** Ported from legacy patch 0005 **minus coalescing** (the
CDN owns that) and extended to multiview.

- **`-cc_extract`** (+ `-cc_slots` / `-cc_lang` / `-cc_page` / `-cc_magazine`; `PTV_NO_CC=1` kill
  switch). OFF by default — output is byte-identical to a build without the feature when the
  option is absent.
- **One INDEPENDENT extraction per participating input.** Single-input is just N=1; on multiview
  each slot gets its own `cc_dec` (on that input's own decode thread), its own `dvb_teletext`
  encoder, its own subtitle track per rung, and its own page (BCD-walked from the base: 888, 889,
  890, 891). Each slot's captions are stamped on **that slot's own `h0`** — the same anchor its
  audio rides — which is what keeps them on their own dialogue instead of the mosaic's leader.
  The legacy implementation guessed here and always bound input 0.
- **Per-slot naming:** CC tracks own the LOGICAL subtitle indices `s:0..s:(n_cc-1)` in INPUT
  order (copies numbered after), so `-metadata:s:s:N language=mva` applies the multiview
  convention. The tag on the wire is the metadata; the G0 national subset still follows the REAL
  source language, so a slot tagged `mvc` still renders its Spanish.
- **Observability:** `cc=caps/erase/keep/a53` on the single-input stats line, per-slot
  `in<k>:.../cc=<caps>/<a53>` on the multiview one, `[PTV-CC]` first-caption / QUIET / RESUMED /
  reset lines all carrying their slot.

**⚠ A binary reporting 1.1.0 may or may not have this** — three builds carried that banner
(released 1.1.0 without CC; an intermediate that got the teletext ENCODER via patch 0006 but not
the option, because patch 0001 was not regenerated; and the intended one). Check the option, not
the banner: `ptvencoder -h 2>&1 | grep cc_extract`.

**Verified:** 9/9 deterministic encoder fixtures; four live SRT feeds x 200s (err=0 drop=0); 2- and
4-slot mosaics with distinct per-slot captions proven by an independent libzvbi decode; a killed
slot erases its page and keeps its PID alive; mixed dvb+cc numbering; byte-inert when off.

## 🏁 1.1.0 (2026-08-08) — RELEASED. Content-identical to `v1.1.0-rc1` (banner bump only).

**Release gate PASSED — 4-day fleet soak, 2026-08-05 → 08-08, ~243 channels on 8 hosts** (the rc
content ran on the canary tier from 07-30, so effectively 9 days on the code):
- **Zero FATAL, zero SIGSEGV, zero `[PTV-STALL]`** fleet-wide for the whole window.
- **Accumulator classes exercised and clean:** the 26.5h 33-bit PTS wrap was crossed by ~200
  channels within the same morning (48/56 readable on cor-1 alone past 27h) with **zero muxer
  EINVAL / MUXGUARD** — the class that froze Cinestar in July. 43 channels passed 3 days uptime
  with `lipsync=` flat and RSS bounded.
- **Lip-sync at gate time: 236/243 in band (≤50ms), nothing above 1s** fleet-wide.
- **Engines converging:** 140 RESYNC engages → 56 completes, 167 corrector parks, **0 circuit
  breaker ERROR-latches** in the final 2-day window.
- **Every desync intervention traced to a broken INPUT, never to healthy content:** 67 PIN-driven
  restarts over 16 channels in 5 days (160 flags → 122 confirmed → 67 restarts; 93 flags cleared by
  recheck, 4 channels flagged but never restarted). 37 of the 67 were one channel (K-TV) whose
  source mux was discarding 4.7% of video packets; after the provider was given the TSDuck evidence
  its rate went 37 → 3 → 0/day. **Zero PIN restarts on a healthy feed in the entire soak.**
- **First unattended save of a previously-unownable class:** a multiview 2×2 slot pinned at +2.7s
  with its corrector storm-disarmed and no mv-RESYNC to take over → detected, confirmed, restarted,
  +13ms (2026-08-05). That class cost 30 hours of on-air desync in July (NewsNation) because nothing
  could see it.

Known limitations carry forward UNCHANGED from the rc entry below (F7 bank-blind bake / >5s
gate-silence / T5 confirm-on-flipped-reading / no mv-RESYNC / ZENLIFE dying-feed runaway) — each with
containment, each with a named 1.1.x fix vehicle; see
`analysis/ptvencoder-f7-silence-fix-plan.md` and #58 v0.5 §9–§10. Release patch set:
`patches/v2/release-1.1.0/` (0001–0005, chain-verified on clean upstream `master fc4b523596`).
**v1.0.0 remains the rollback anchor.**

## 1.1.0-rc1 (2026-08-05) — CONTENT FREEZE, tag `v1.1.0-rc1` = pre30.1 `c43094bbf9`

Release candidate for v1.1.0 (the fleet-wide milestone, roadmap #43). The tag sits on the EXACT
commit the whole fleet runs (build `N-125856-g2ae2413488`, banner still reads `1.0.1-pre30.1`) —
tagging the running content instead of a banner-bump build preserves the banked soak uptime
(canary tier since 2026-07-30; fleet-wide incl. cor-1/cor-2 since 2026-08-05, ~100% adoption
after the restart waves). At release: one banner-bump commit → tag `v1.1.0` → one build.
**Content freeze on `audio-batch`: docs-only commits until v1.1.0.**

**Soak gate (7 days, through ~2026-08-12), pass criteria stated up front:** zero
FATAL/segv/[PTV-STALL] fleet-wide; RSS bounded — MEMCAP kills confined to the known leak channels
at a stable ~1-2/day; no silent >5s desync that outlives PIN_DESYNC detection; engines converging
(completes/parks accruing, circuit breaker never ERROR-latching). These are the v1.0.0 gates plus
the sensor-backed criteria that build could not express.

**What 1.1.0 adds over 1.0.0 (the pre9→pre30.1 line, headline items):** the passive residual
lip-sync SENSOR (`lipsync=`, owner-certified against the wire oracle) → the residual-sync
CORRECTOR (default-ON, band ≤350ms) → RESYNC hard-reset engine for the >350ms band (default-ON,
audio-late whole-step jump cut + audio-early video IDR-skip walks with hs self-attribution) →
recovery re-anchor + wall-evidence split (#63 storm fix) → glue classifier (#33) → broken-AAC
tolerance + rebuild escapes (#38/#42/#46) → AFMT rebuild re-anchor → MEMCAP/MUXTOL/MUXGUARD
guardrails → PPS-dedup (patch 0005, the 40GB storm-heap fix) → mv sensor port + mv corrector.
Wedge classes fixed: #32 (pre8), EINVAL zombie (pre26), egress-pressure death loop (pre27).

**Known limitations (all documented, all with containment — none is a 1.0.0 regression):**
1. **F7 bank-blind whole-step bake** — an armed bank disables the wall-evidence split; on
   chronically bursty feeds (63 channels at the 12s ceiling) an asymmetric event can erase a real
   hole and bake the difference (Avivando −5.9s, 2026-08-03). Contained: sync_check PIN_DESYNC
   auto-restarts. Fix = #58 §9/A4 per-event disturbance test (1.1.x).
2. **>5s gate-silence** — above the corrector's implausibility bound, RESYNC's confirm gates can
   decline with zero log output. Contained: PIN_DESYNC. Fix = decline-reason logging (1.1.1).
3. **T5 confirm-on-flipped-reading** — a RESYNC confirm timer can fire on a reading whose truth
   flipped during the window; measured live at ±30s (AlrightTV 2026-08-04); the engine walks its
   own overcorrection back (self-correcting double excursion). Fix needs the FX-I redesign (1.1.x).
4. **No mv RESYNC** — multiview slots have corrector+PLL only; a slot pinned >5s has no in-process
   owner. Contained: PIN_DESYNC (proven live: mv 2×2 slot +2.7s → restart → +13ms, 2026-08-05).
   Fix = mv-RESYNC with #24 (1.1.x).
5. **Dying-feed async runaway (ZENLIFE class)** — video trickle drives unbounded aresample pad;
   bounded by MEMCAP (~10min respawn cycles). Fix = #53 + async-pad rate bound (1.1.x).

External containment shipped alongside (transcoder repo): sync_check.sh PIN_DESYNC — content-domain
pinned-desync detection from the `lipsync=` sensor with engine-activity awareness (`96647ea` +
multi-audio token fix `fb08f66`); day-1 in production: 16 flags → 9 confirmed → 5 channels
restarted, incl. the first unattended save of a class no in-process engine could own.

## 1.0.1 (pending) — mv-audio robustness batch

(8d) PRE30.1 — RESYNC vskip walk survives its own post-seam starvation dups
(2026-07-30, driven by the first live vskip fires).

LIVE EVIDENCE (i24/cor-3, three identical arcs 2026-07-30): every vskip seam was followed
~2s later by "walk ABORTED (new event)", turning a tens-of-seconds walk into a 5.5-minute
two-stage arc — 14:26 CDT: ENGAGE +2081ms → vskip 1560ms (1 whole GOP) → abort at R=+619ms
→ settle 300s → re-ENGAGE +543 → chunk → COMPLETE −1ms; 14:51 CDT: same shape through the
PARTIAL-GOP ESCAPE seam (vskip +1320ms, 0 GOPs). Control: glo-2/RNTV (same binary, calm
pipeline) completed two multi-seam vskip walks the same day with NO abort — the abort is
environment-coupled.

MEASURED ROOT CAUSE (reproduced locally: bursty 2s-burst delivery + a cushion the skip
mostly consumes; edge-naming instrumentation): the walk's own vskip consumes buffered
video; the next inter-burst delivery gap then starves frame_q and the output dups — one
house_skew tick per dup — while the audio thread idles the same gap. Its next evaluation
sees ONE >1.25-tick hs jump, ptv_recanchor's event-edge snapshot fires, and the walk
aborts its own seam's aftermath as an external disturbance ("new event" = the hs-step
edge — verified by naming the edge; NOT the g_shed_wall annotation, which feeds only the
corrector's event-ACTIVE hold and never the walk's abort check; the pll-acquire edge is
multiview-only and vskip walks are single-input-only).

FIX — hs-step SELF-ATTRIBUTION, directional + budgeted (not a blanket window):
- Arming: when the walk posts a vskip span to the executor (ENGAGE and every in-walk
  re-request), the span extends an hs "self-budget" (Σ posted spans + 2-tick slack),
  based at the house_skew value at first arming. Covers the PENDING phase too (a
  shallow-cushion executor starves frame_q while waiting for the stop IDR — dups begin
  before the seam is harvested) and both seam shapes (DONE and partial-GOP ESCAPE).
- Absorb rule (walk block, before the abort check): an event that is hs-ONLY, with a
  POSITIVE step (starvation dups only ever raise hs), while cumulative growth since
  arming ≤ the self-budget → absorbed with a WARNING line
  ("hs step +Nms absorbed mid-walk (own vskip's starvation dups; ...)"), walk continues.
- Foreign disturbances still abort: any other edge type (AFMT/rebuild/reopen/ledger/
  bank/epoch/glue — never gated), any negative hs step, or growth beyond the budget
  (a genuine input stall's dup run busts it within one gap; a ≥5s stall additionally
  trips the delivery-dead watermarks unchanged). Budget honesty: the posted span is
  refunded on REFUSED / executor-unresponsive recall, and shrunk by (requested −
  achieved) at DONE/ESCAPE, so the foreign-hiding room never exceeds what our skip
  could actually starve.
- Chunk-only walks (PTV_RESYNC_SILENCE / fallback) arm nothing — byte-identical abort
  behavior there. Corrector HOLD-on-recent-self-shed, dwell resets, and every
  RECANCHOR/corrector event path untouched.
- The walk's abort line now NAMES the edge ("walk ABORTED (new event: hs step)") —
  the live arcs' bare "new event" cost a full diagnosis round; grep-compatible prefix.
No new knobs (the budget is derived from the walk's own posted spans). Fixtures: FX-K
(multi-seam vskip walk under bursty delivery + shallow cushion survives its own
starvation dups to COMPLETE; the pre-fix binary aborts on the same harness), FX-K2
(foreign-disturbance control: a mid-walk sender stall must still abort the walk).
DEVIATION NOTE vs the task brief: the brief attributed the abort to the executor's
g_shed_wall/g_shed_cnt stamp and suggested tagging it; measurement showed the shed
stamp has NO path into the walk's abort check — the real carrier is the hs-step
snapshot edge, so the attribution tag lives on the hs axis (directional budget from
posted spans) instead. Verdict on the coordinator's "should walks yield in churning
environments?": yes for genuinely-foreign churn (everything outside the budget still
aborts), no for the walk's own seam aftermath (that self-abort is the live defect).

(8d-rr301) PRE30.1 ADVERSARIAL-REVIEW FIX (2026-07-30, on top of 8d):
- T1 OVERSHOOT CREDIT: the DONE/ESCAPE budget re-true only handled ach < req; the
  executor's stop rule and the rr30-T4 packet bound both land up to tol + 1 pkt
  (~290ms at the 250ms default) PAST the requested span — real self-caused starvation
  with only the 80ms 2-tick slack to cover it, so our own post-seam dups could bust
  the budget and re-open the false abort. The adjustment is now symmetric
  (budget += ach − req, clamped ≥0) — attribution stays bounded by what the skip
  actually consumed. Review verdicts: shed-no-path claim VERIFIED in code (abort
  inputs = 9 snapshot edges + delivery watermarks only; shed reaches only the
  corrector's event-ACTIVE hold + log annotation); budget clears on every walk exit
  (absorb check is inside the rsn_active block, fields reset at ENGAGE before re-arm);
  pre-fix pre30 binary FAILS FX-K 3/3 (aborts in the PENDING phase at R=+4502, zero
  seams) — non-vacuity proven against a 2d71edeede worktree build. FX-K live log:
  budget 4580 = 4500 posted + 2×40ms ticks, +360ms PENDING-phase absorb, COMPLETE.
  Harness: FX-K/K2's no-ENGAGE path now consumes a retry instead of failing the
  fixture (measured flake: a try with zero resync activity and healthy buffers).

(8c-rr30) PRE30 ADVERSARIAL-REVIEW FIXES (2026-07-29, on top of 8c):
- T1 WALK LIVENESS CEILING: item A's corroboration waits a moving sensor out "however
  long it takes" — with NO bound, a never-stable sensor (storm, jittery source) pinned
  rsn_active forever, and a pinned walk defers ALL other engines (corrector via
  resync_owns, RECANCHOR via its rsn_active gate) — a new wedge class. A walk with no
  seam progress for PTV_RESYNC_WALK_CEIL_S (600s ≈ 10x the worst measured post-seam
  settle; 0 disables) now aborts + settles, same posture as budget exhaustion. FX-J.
- T4 EXECUTOR RUNNING SPAN BOUND: the vskip stop rule evaluated only at KEYS, so a GOP
  longer than the fire-time estimate (film→ad, 1s→4s) dropped its whole REAL length —
  audio flipped LATE by the overshoot (a bigger desync than the one being fixed, plus
  hold+abort+settle before recovery). The drop loop now escapes mid-GOP past
  target+tol at PACKET granularity (overshoot ≤ tol + 1 pkt).
- T2b DONE-PATH DECODER FLUSH: a non-IDR stop key (recovery-point SEI also sets
  AV_PKT_FLAG_KEY) left pre-skip refs in the DPB; h264 frame_num-gap concealment then
  smears stale content under clean labels with NO corrupt flag — the claimed CORRUPT-
  check coverage does not exist on the SW h264 path. Clean-IDR resume now flushes
  (the selfheal deferred-reset recipe); the deadline ESCAPE stays unflushed (Session-109
  no-IDR posture) with its log/comment stating the real concealment behavior.
- T2a OFFSET-PUBLISH ORDERING: off_total store is release / content_index load is
  acquire — the relaxed pair let compiler/CPU reordering tear boundary vs total for
  in-flight pre-boundary frames.
- T3 STALE-REQUEST RECALL: after the walk gave up on an unresponsive executor (or an
  event abort), an un-picked-up g_vskip_req_us stayed posted — a decode thread
  unwedging minutes later would land a stale skip on a walk long gone (double actuation
  with the silence chunks that took over). Both paths atomically recall an unstarted
  request; a started one is left to land (real content edit; the sensor owns the truth).
- T9 SELF-PAD DELIVERY-DEAD GUARD: the walk's own 2s silence pad transits mux_q above
  the half-depth dead line while the wire drains it (~94 AAC frames vs PTV_QDEPTH/2=24)
  — pre29 never lived long enough to see it (the crossing check aborted first); item A
  unmasks it. Exactly the "mux_q backed up" reason is ignored inside the post-seam
  window; the 5s wire watermarks still abort a really dead wire.
Knob added: PTV_RESYNC_WALK_CEIL_S=600 (0 disables). Fixture added: FX-J (ceiling).
Full battery (A..J) re-run green on the fixed binary.
KNOWN + DEFERRED (documented, no code change): the confirm timer can fire on a reading
that decayed/flipped sign during the window (i24 08:58 −327ms on r0=+2081 — pre30 does
not change this); a same-side/flatness-at-expiry condition is recommended as a follow-up
but NOT hot-patched here because FX-I's designed fire train deliberately exploits the
flip (residue-pinned reopen fires LATE after a positive-r0 confirm) — the guard needs a
new test knob and an FX-I redesign together. The ESCAPE path's mid-GOP smear (see T2b)
is accepted, bounded by one GOP.

(8c) PRE30 — #69 RESYNC REFINEMENTS: post-seam sensor hold + video IDR-skip actuator +
sliding breaker (owner-approved 2026-07-29, driven by the first two production fires).

A. POST-SEAM SENSOR HOLD (the priority — live evidence 2026-07-29). The sensor reading
taken ~2-4ms after a seam is garbage: i24 applied chunk +2000 on R=+2081 (expected residual
+81) and the instant reading was −1035 — the mid-walk band-crossing check falsely ABORTED
the walk; the true value settled +125 flat only ~1-2min later (observed decay ~12ms/s).
Law&Crime's chunk +1708 on R=+1708 read −110 — in-band, so it COMPLETEd, luckily correct.
Root: the seam re-seeds the content-anchor EMA (rs_ma_seed=0) and readings during
re-convergence are biased. Fix, gating R-BASED WALK DECISIONS ONLY (event-edge aborts —
AFMT/rebuild/reopen/delivery-death — stay live on any reading):
  - After every seam (audio-early chunk, vskip, and the LATE whole-step fire) NO R is
    consumed for the COMPLETE / band-crossing / next-seam decisions for
    PTV_RESYNC_SEAM_HOLD_S (default 20s; 0 disables = pre29 walk decisions).
  - After the hold, acting on a reading additionally requires 2 consecutive samples ≥5s
    apart on the SAME side (in-band / crossed / still-over) moving <50ms between them.
    Justification for hold=20s despite the ~90-120s observed settle: a still-decaying
    sensor moves ~60ms per 5s at the observed 12ms/s — it NEVER reads stable — so the
    stability rule makes the walk wait out the transient however long it takes, while a
    settled sensor passes within ~10s of the hold expiring. The hold alone would have
    needed ~90s+ to make i24's reading trustworthy; hold+stability gets the right answer
    without hard-coding the decay time. (Deliberate strengthening beyond "2 same-side
    readings": same-side alone is defeated by a slow decay that parks on the wrong side
    for a minute — i24's would have double-aborted at any hold ≤60s.)
  - The inter-seam gap is now max(PTV_RESYNC_CHUNK_GAP_S, PTV_RESYNC_SEAM_HOLD_S) — a seam
    never fires on an unheld reading.
  - A reading inside the hold cannot OPEN a confirm timer (the LATE seam stamps the hold
    too) — an opened-on-garbage timer with |r0|>2s could otherwise confirm a
    wrong-direction second fire on the BIG window.
  - Confirm-window logic BEFORE an engage is untouched (no seam yet — the sensor is fine).
    Budget/abort/settle semantics unchanged.

B. VIDEO IDR-SKIP ACTUATOR for audio-EARLY walks (owner mandate: "drop video, not insert
silence"). The pre29 audio-early actuator mutes audio up to 2s per chunk. Default is now a
video jump cut: skip video content FORWARD at GOP granularity using the pre8 QSHED
machinery — the audio thread posts a target span, the DECODE THREAD executes it at the
video_q pop site (upstream of decode and the rung split, so every rung skips the same
content coherently), dropping whole GOPs from the head and decoding the stop IDR (the
QSHED shape: decoder input stays contiguous whole GOPs). Greedy whole-GOP rule: largest
whole-GOP total ≤ R + PTV_RESYNC_VSKIP_TOL_MS (250ms); the sub-GOP residual goes to the
corrector exactly like today's post-walk handoff. Label-neutrality: the achieved span is
published as a src-keyed offset subtracted in content_index() — output labels CONTINUE
monotonically while content jumps (the video mirror of the audio-late reanch_mono skip
seam; two-tier src-keyed so frames of earlier epochs still in frame_q keep their mapping) —
and the m_v EMA re-seeds at the first post-boundary emit (the rs_ma_seed=0 mirror). The
sensor sees the skip directly through m_v (out−src drops by the achieved span), so the walk
measures its own progress; ra_applied_us accounts it for the corroborated path exactly like
a door step. Audio is UNTOUCHED — no silence, no glue edit, no corrector event edge needed.
  - Default ON for audio-early fires; PTV_RESYNC_SILENCE=1 reverts to the pre29
    silence-chunk actuator (code path intact — it remains the fallback whenever vskip is
    not viable: multiview, no GOP estimate yet, GOP longer than R+tol, stale keys, executor
    REFUSED/ESCAPE/unresponsive).
  - Viability gate at fire/next-seam time: decode-thread-measured key-to-key span EMA
    (g_vgop_est_us) must fit R+tol AND a key must have passed within max(2×GOP,
    PTV_RESYNC_IDR_WAIT_S) — never start dropping into a stream that has stopped
    delivering IDRs.
  - If no usable IDR appears within PTV_RESYNC_IDR_WAIT_S (default 5s ≈ 2× typical GOP)
    the executor DEADLINE-ESCAPES: stops dropping, resumes decoding mid-GOP UNFLUSHED
    (the Session-109 posture; corrupt frames are dropped by the existing CORRUPT check
    until the next key), reports the partial span, and the walk falls back to silence
    chunks for the remainder (logged).
  - A/V ordering: the skip realizes as one jump cut; the walk is fire → item-A hold →
    assess residual → possibly another skip after max(gap, hold). Chunk budget/abort
    semantics carry over (achieved spans decrement the same budget).
  - Log lines: `[PTV-RESYNC] ... vskip <N>ms (<K> GOPs) applied (IDR jump-cut seam; ...)`
    (walk side) + `[PTV-VSKIP] skipped <N>ms ...` (executor side).
  - HONEST LIMITS (flagged for review): (1) the skip consumes BUFFERED content — on a
    shallow pipeline the IDR wait rides the frame_q cushion; if the cushion empties
    mid-wait the output dups and the dup-advanced label cursor eats part of the relabel
    (AVLOCK then realizes that part as audio delay = pad, the very thing being avoided) —
    bounded by the wait cap and self-correcting because the walk re-measures real R.
    (2) A sub-GOP residual >OK-band re-enters the walk as silence chunks (bounded < 1 GOP)
    — strictly better than pre29's full-R mute but not silence-free on every phase.
    (3) The whole-GOP overshoot beyond R+tol is possible when the head partial-GOP
    remainder alone exceeds the target (entry-gated by GOP-est ≤ R+tol, so bounded ~1 GOP;
    a resulting small negative R is owned by the crossed-band abort → corrector/LATE path).
    (4) SCTE-35/sub copied streams keep their source→h0 mapping (no orphaning, gates
    unaffected) but a cue's lead vs the SKIPPED video content shifts by the skipped span —
    same class as any content deletion upstream of the splice point.

C. SLIDING BREAKER WINDOW. The pre29 tumbling first-fire-anchored window reset its count
at expiry, so N fires straddling one boundary (e.g. 6 fires across it, 3+3) never armed
the breaker. Now a ring of the last PTV_RSN_RING(16) fire wall-times arms when
ts[newest] − ts[newest−(N−1)] ≤ PTV_RESYNC_BREAKER_WIN_S. PTV_RESYNC_BREAKER_N is clamped
into [2,16]. Escalating backoff / quiet-disarm (clears the ring) / no-dead-zone semantics
unchanged.

Knobs added (all pre30): PTV_RESYNC_SEAM_HOLD_S=20 (0 disables) · PTV_RESYNC_SILENCE=1
reverts item B · PTV_RESYNC_IDR_WAIT_S=5 · PTV_RESYNC_VSKIP_TOL_MS=250. Test-only:
PTV_RSCORR_TESTWALK_PAUSE_AT_S/_PAUSE_S and _PAUSE2_* (zero the walk inside a window —
the FX-I long-short-short fire train). Fixtures: battery re-based on pre29.1 (d3b10d2d37,
PTV_NO_RESYNC=1 kill-switch inertness gate); FX-C/F pinned to the silence path + short
seam-hold; new FX-G (item A), FX-H/H2/H3 (item B + control + escape), FX-I (item C, with
a measured straddle self-check). DEVIATION NOTES: the silence ENGAGE line now carries a
reason suffix (e.g. "(PTV_RESYNC_SILENCE)", or the measured key-age vs freshness limit) —
log-signature-identical tags, not byte-identical lines, vs pre29; FX-H tolerates a bounded
silence-chunk fallback for sub-GOP residuals (hard-gates zero-silence only on the
zero-chunk path) — see item B honest limits; FX-F1 runs 275s (the item-A hold + 2-sample
corroboration stretch the pre29 fire/walk cadence ~+20s against the 40s quiet-disarm
deadline). Fixture-round harness findings (test-scripts/repro/resync-fx.sh comments carry
the detail): ptvencoder's -t is a PER-OUTPUT-GROUP option — placed before -i it lands in
the input group and is silently ignored (the 2-rung runner hit it); a sparse-key source
joined mid-GOP fails the input probe (video_size 0x0), so explicit-src fixtures start the
receiver before the sender; force_key_frames expr needs the isnan(prev_forced_t) seed.

(8b) PRE29.1 — #69 RESYNC DEFAULT ON (owner 2026-07-29). pre29 shipped the engine opt-in
(PTV_RESYNC=1) as a deploy-safety stance, but that contradicts the project convention — new
engines arm by default on the pre train (RECANCHOR/MEMCAP/MUXTOL all shipped default-on with
a PTV_NO_* kill switch) — and the fleet deploy mechanics set no per-channel env, so an opt-in
flag would simply never arm. Flip: g_resync defaults to 1; the PTV_RESYNC enable knob is
REMOVED; **PTV_NO_RESYNC=1 is the kill switch** and restores byte-identical pre28 behavior
(the review-proven inert path). All other knobs/defaults unchanged. Fixture battery updated
(FX-A's flag-off state = PTV_NO_RESYNC=1; armed fixtures rely on the default). Safety stance
unchanged in substance: the 120s/60s confirm windows, health/stability gates, and the circuit
breaker are the protection — the flag was never the safety mechanism.

(8a) PRE29 — #69 RESYNC: THE LARGE-STABLE-R DEAD ZONE (owner verification 2026-07-28: the
lipsync= R checked 4/4 by eye against real channels — both signs, offsets up to 20s — the
sensor is CORRECT; a large STABLE R is a real on-air desync). Root cause of the dead zone:
above ~5s the corrector disarms R as implausible, and RECANCHOR's deletion-ledger
corroboration (rightly — the aseam pinned-R guard) refuses the uncorroborated class, so a
channel like RNTV (+14.4s) had NO engine allowed to act until restart. pre29 gives every
band an owner:
    |R| ≤ 50ms          nothing (the existing 20ms park / 80ms engage behavior, unchanged)
    50–350ms            corrector as today + ADAPTIVE SLEW: above 150ms |R| the 2ms/s clamp
                        rises to PTV_RSCORR_SLEW_FAST (default 5000 µs/s = 0.5% rate, below
                        the pitch JND — inaudible; 0 disables). Active regardless of
                        PTV_RESYNC; cuts a 300ms steer from ~2.5min to ~1min.
    >350ms (PTV_RESYNC=1) the RESYNC path — a SECOND ENGAGE PATH inside ptv_recanchor
                        (one engine, shared walk state = structural mutual exclusion with
                        the corroborated path). Corroboration is traded for TIME: a confirm
                        timer (PTV_RESYNC_CONFIRM_S=120; PTV_RESYNC_CONFIRM_BIG_S=60 when
                        |R0|>2s) during which R must stay over PTV_RESYNC_OK_MS=150, plus
                        every RECANCHOR engage-side health gate (delivery live, slip
                        parked, label health, no event within 10s), is the evidence. Two
                        seam types, both the proven [PTV-ANCHOR] door algebra + the exact
                        RECANCHOR self-edit recipe: audio LATE (R<0) = ONE whole backward
                        step, realized by the monotonic emission guard as a bounded audio
                        content skip (skip seam); audio EARLY (R>0) = chunked forward steps
                        (PTV_RESYNC_CHUNK_MS=2000 per PTV_RESYNC_CHUNK_GAP_S=5, budget
                        R+R/5) as aresample hard-comp silence (silence-pad seams). At fire
                        time any carried corrector trim folds INTO glue_off label-neutrally
                        (the ptv_rebuild_reanchor bookkeeping transfer) and the RECANCHOR
                        ledger AMNESTY runs (factored helper, existing path byte-identical);
                        the corrector's perm_disarm / lifetime authority accounting is NOT
                        touched. While the corrector defers to resync it keeps any ALREADY
                        ENGAGED steer running.
NO routine cooldown (owner decision: a seam costs <1s on air, a desync costs the whole wait
— throttling normal operation is wrong). After a COMPLETE the only gate before the next
reset is the trigger sequence itself — R over the band again + a fresh confirm window (the
natural seam-spacing floor, ~1 seam per 60-120s max). A CIRCUIT BREAKER exists solely to
surface+contain pathology (resync self-oscillation, extreme thrash-storm) at ERROR level:
PTV_RESYNC_BREAKER_N=4 fires within PTV_RESYNC_BREAKER_WIN_S=900 ARM it (one ERROR line);
armed, each further reset additionally waits an escalating backoff (fixed 120s ×2 → 600s
cap — the #49 ACQUIRE / pre27 AFMT-breaker escalating-interval pattern) and logs the
throttle at WARNING; PTV_RESYNC_QUIET_S=1800 below the band disarms it and clears the
history. NO-DEAD-ZONE invariant (owner): under no combination of flags, timers, or states
is an over-band R left with no engine allowed to act — while the armed breaker's backoff
(or an abort settle) blocks a re-fire, the corrector's deferral lifts and it engages even
above 350ms (fixture-asserted). BROKEN-SENSOR CEILING: |R| > 600s never fires (closes the
timer / aborts the walk) — the first fixture round live-fired the need: a sensor artifact
published R ≈ INT64_MIN with valid=1 and an unguarded fire stepped glue_off by −9.2e18µs;
the owner-verified real range is tens of seconds, beyond 600s is a broken sensor. Adaptive
fast slew carries ~rate×sensor-lag proportional overshoot (~50ms at 5ms/s, settles at the
base slew inside the park band); end-to-end park time for ~300ms offsets stays
settle-dominated — the fast rate's end-to-end win is at larger offsets and in the
no-dead-zone nibbling role. Stats: fired resets append rsn=N (absent-when-zero — the
healthy-channel line is byte-identical). Explicitly OUT of this pre: the video-side
IDR-skip actuator (QSHED-based, pre30 — resync only actuates the audio door); corrector
lifetime accounting unchanged; mv untouched (ptv_recanchor already runs only on the
single-input / non-follow path; the follow PLL owns content alignment). PTV_RESYNC
defaults OFF = pre28 behavior everywhere EXCEPT the adaptive slew, which is live by
default and changes pre28 behavior whenever an engaged corrector steers |R|>150ms
(PTV_RSCORR_SLEW_FAST=0 restores the exact pre28 clamp; every other code path is inert
with the flag off). Review fixes (r2): the corrector deferral lifts when resync's confirm
timer cannot run (label health out of band / unpublished — closes the timer every
evaluation, so deferring there recreated the dead zone; measured), and the 600s ceiling
also catches the exact INT64_MIN sentinel (llabs UB bypassed it). Test-only: the TESTWALK cap
now saturates by magnitude (negative bakes) and PTV_RSCORR_TESTWALK_DECAY_AT_S zeroes the
walk after t seconds (the transient/no-fire and breaker fixtures).

(7z) PRE28 — #67 THE STORM-STATE PPS-CHURN RUNAWAY (ENT-CORELINK-HTTV glo-2 2026-07-26:
    40.1GB RSS in 37min, run born into an active discontinuity storm; ~625 fully-dirty
    ~64MiB glibc heaps). Root cause is in libavcodec, so the fix ships as v2 patch
    0005-h264-pps-dedup (libavcodec/h264_ps.c): ff_h264_decode_picture_parameter_set
    allocated a fresh ~174KB refstruct for EVERY PPS NAL; the SPS path has an
    identical-content shortcut, PPS did not. Calm channels reuse the freed chunk (flat
    RSS); sustained storms interleave long-lived disc-buffer allocations into the same
    arenas so every churned PPS lands in fresh space and the arenas only grow. Fix =
    PPS mirror of the SPS shortcut (length+raw-RBSP memcmp vs pps_list[id] + SPS
    refstruct pointer-identity check -> keep existing object). Verified bit-exact vs
    stock ffmpeg 8.1.1 on identical-resend, one-bit-changed, and byte-identical-PPS-
    across-SPS-change fixtures; 174KB alloc rate under storm fixture -> 0. This
    fftools tree carries NO code change for pre28 (version bump only) — the content
    is entirely patch 0005. Known residual grower (bounded by PTV_RSS_CAP_MB, next
    release): aresample hard-comp gap-silence buffer churn from the non-converging
    storm-birth control loop (hsres climbing forever, vdlvhold pinned) — convergence
    fix tracked as the #63-adjacent control-loop item.

(7y) PRE27 (same release) — #62 THE EGRESS-PRESSURE MUX-DEATH CLASS (Praise_TV glo-2
2026-07-24: output udp uses bitrate= pacing + fifo_size=28672 ≈ 6.4s at the cap; every
supervised respawn's startup prime/catch-up burst filled the paced egress fifo ~6-7s after
anchor, udp.c returned ENOMEM from the write, and the pre26 always-fatal [PTV-MUX] path
killed the channel again — 7 deaths in a row, channel dark). The fatal path treated ALL
write errors alike; transient egress pressure is not label corruption.
1) [PTV-MUXTOL] errno filter at the mux write (mux_thread, per-rung thread-local state):
   ENOMEM/EAGAIN = tolerate — drop that pkt, count it, rate-limited WARNING (1/10s per
   rung) with rung/stream/errno/count, keep muxing. Everything else (EINVAL = the pre26
   backward-label crash class, EPIPE, EIO, ...) stays immediately fatal, unchanged.
2) Dead-egress ceiling: 60s after the FIRST tolerated failure of a failing run with no
   successful write in between escalates to the fatal path with a distinct message
   ("egress dead 60s") — a dark channel needs the respawn anyway; never a silent
   forever-throttled zombie. (Review hardening: first-failure-based, not last-success-
   based, so a >60s upstream stall + one transient catch-up error still gets 60s.)
3) [PTV-MUXGUARD] drop-span ceiling (pre26 review rider), STRICT (audio/video) streams
   only: a 60s guard-drop span on one A/V stream with no accepted pkt there likewise
   escalates to the fatal path (span resets on any accepted pkt on that stream);
   governed by PTV_NO_MUXGUARD as before. Sparse guarded streams (SCTE-35/teletext/
   DVB-sub) keep the pre26 drop-silently-survive posture — two isolated drops minutes
   apart (the sparse-PID wrap-aliasing class) must never kill the channel.
Kills: PTV_NO_MUXTOL=1 reverts 1+2 (any write error = fatal, pre26 behavior). Test-only:
PTV_MUXFAIL_SIM="<enomem|eagain|einval>:<start_s>:<dur_s>" makes the write path pretend
av_interleaved_write_frame failed during the window (unset = byte-identical);
PTV_MUXTEST_BACK_HOLD_S widens the pre26 one-shot backward-dts injection to a window
(sustains MUXGUARD drops to exercise ceiling 3); PTV_MUXTEST_BACK_TYPE=a|s|d targets
the injection at audio (default) or a sparse subtitle/data stream (verifies the
ceiling's strict-only scope).

(7x) PRE27 — THE MEMORY-RUNAWAY CLASS (ENT-CORELINK-HTTV glo-2 2026-07-26: 26.2GB RSS in
~46min on pre24, neighbors ENOMEM-killed by slice pressure; normal RSS 0.4GB). Hunt verdict
(memhunt, local pre24 fixtures + live Praise_TV thread/queue captures): the WEDGE does NOT
grow (measured flat 17min with input flowing, default AND deep-prime; every queue is capped —
video_q ≤2048 deep-prime, frame_q ≤PTV_FRAMEQ, mux_q 48, gate lists bounded, disc buffer
per-cycle-freed, udp fifo stock-bounded; Praise's vq=2048 was the CAP, not growth); storm
events step RSS once via AUTO-BANK/cushion escalation filling frame_q to cap (bounded
pool-peak ratchet, +600MB local). The UNBOUNDED axis is AFMT REBUILD COUNT: param-flapping
origins (CORELINK flaps 44.1<->48kHz across discontinuities — Riff_TV [PTV-AFMT] 2.9s before
its EINVAL death) confirm audio-path rebuilds forever; each rebuild churns the whole -af
graph (~2MB/cycle transient, live heap FLAT = allocator churn, which glibc's per-thread
arenas retain far worse than macOS); a 6s-flap fixture also reproduced pre24's full EINVAL
crash chain at rebuild #3 (60s) — and with pre26's crash fix the channel now SURVIVES the
storm, so the churn runs indefinitely. Two bounds, both new:
1) [PTV-AFMT] REBUILD-STORM CIRCUIT-BREAKER (audio_feed confirm block): an escalating
   minimum interval between rebuilds — 2nd rebuild within 120s starts at 2s, doubles to a
   60s cap, 120s of quiet resets; while held, changed-format frames keep dropping (the
   existing settling posture; flap-storm audio is upstream-garbage anyway) and a rate-limited
   WARNING names the hold. Bounds worst-case churn to ~1 rebuild/min. Covers ACHOP re-forms
   (same confirm site). Measured A/B (6s flap, 6-rung, ~6min): pre26 57 rebuilds vs memcap
   10 rebuilds + breaker lines. PTV_NO_AFMT_BREAKER=1 disables.
2) [PTV-MEMCAP] RSS CAP WATCHDOG (master watchdog thread, 10s samples): PTV_RSS_CAP_MB
   (default 8192, 0 disables) — WARNING at 75% (with a capture-now hint), FATAL + fflush +
   _exit(1) on two consecutive samples >= cap: one supervised respawn at a bounded size
   instead of a 26GB box-killer taking out neighbors. Linux /proc/self/statm; macOS mach
   task_info. Live-fired in gate (900MB test cap: warn + fatal + death <2s).
   Parse hardened (review finding 4): non-numeric PTV_RSS_CAP_MB = OFF with a warning
   (was: silently off), 0 < value < 1024 warns and clamps to 1024 MB (was: "8G" parsed
   as an 8MB cap = a ~20s respawn loop); one [PTV-MEMCAP] line states the effective cap.
   Ops note: MALLOC_ARENA_MAX=2 in the channel wrapper is a zero-code live A/B lever for
   the glibc-arena-retention half of the runaway.

(7w) PRE26 — THE BACKWARD-LABEL MUX-DEATH CLASS (NBS/CORELINK live 2026-07-25, 8+ crashes in
24h): wedge-free fatal exits + survive-first mux backstop + the emitter root-cause fix.
Kills: PTV_NO_MUXGUARD=1 disables the mux backstop (pre24 EINVAL-exit behavior). Diagnostics:
PTV_MUXDIAG=1 (gated, default off) enables an emission-point backward-label detector with a
composition-state dump. Test-only: PTV_MUXTEST_BACK_AT_S / PTV_MUXTEST_BACK_MS inject one
backward audio dts at the mux feed (gate fixture; unset = byte-identical).
SYMPTOM (live, canary+glo): relabel-flood provider channels (HTTV/Non-Profit/Riff/NBS class)
died with "[mpegts] non monotonically increasing dts in stream 1" → EINVAL → [PTV-MUX] exit,
which then WEDGED (exit() cleanup parked in futex_do_wait while udp-rx/CUDA threads ran) =
silent zombie until sync_check bounced it. Backward magnitudes −2794.7ms / −20778ms; libfdk
"Queue input is backward in time" in the same logs; splits/folds precede by 1-10min.
1) WEDGE FIX (measured: pre24 + a blocking-atexit shim stayed alive ≥30s after [PTV-MUX];
   fixed binary gone <2s, rc=1): the two non-main-thread fatal sites — [PTV-MUX] mux_thread
   and [PTV-NOVIDEO] clock thread, the only exit() calls in ptvencoder*.c — now _exit(1)
   after fflush(NULL): no atexit/cleanup handlers run from a worker thread.
2) [PTV-MUXGUARD] survive-first backstop in mux_thread, immediately before
   av_interleaved_write_frame: mirrors lavf's per-stream monotonic feed check exactly (strict
   for A/V, non-strict for sub/data, incl. the cur_dts!=0 quirk — fires iff the muxer would
   EINVAL) and DROPS the offending packet with a rate-limited WARNING + per-rung counter.
   Label-only (never re-stamps); the EINVAL fatal path remains as final backstop. Measured:
   the injected live-shape backward dts (−2795ms on the fanned-out audio) EINVAL-killed the
   guard-off binary at t=46s and was dropped guard-on with the wire at the +40ms harness
   baseline before and after (240s cell). It also caught the ROOT-CAUSE leak below (−2667ms)
   live in a fixture cell — channel survived where pre24 died.
3) ROOT CAUSE (fixture-reproduced on unmodified pre24 with the exact live signature — libfdk
   backward-input line + mpegts EINVAL at −2667ms, the live magnitude class): a BACKWARD door
   step left in the labels (flush-routed negative A-vs-V mismatch exp_hit / pad round-trip —
   the "aresample drops content" remedies) opens a window of |step| seconds during which the
   graph's emitted labels lead its door labels while swr realizes the drop (swr outpts is
   monotonic by construction — a LIVE graph can never emit backward; lavfi-verified, 0
   non-monotonic out of a −2.6s input step). An [PTV-AFMT]/ACHOP audio-path REBUILD landing
   inside that window destroys the swr state carrying the high-water mark: the new graph
   re-seeds output at the door labels, up to |step| behind the last emitted label, and the
   next emission is a backward DTS fanned to every rung. ptv_rebuild_reanchor (pre20) keeps
   the DOOR labels continuous but never protected the EMISSION side.
   FIX: arm the existing pre20 monotonic emission guard (reanch_mono) unconditionally at the
   rebuild-completion site in audio_feed (covers AFMT, ACHOP, and re-anchor-gated-off paths):
   post-rebuild frames drop until the first label exceeding the last emitted one — the
   interrupted drop's remainder realizes as the same content skip the old graph was mid-way
   through; releases with the existing [PTV-ANCHOR] guard line. Measured: the crash fixture
   on the fix = "monotonic guard released — 126 frames (~2646ms) dropped", 0 backward at mux;
   a 21-min flood with 15 alternating AFMT rebuilds = 10 real windows contained (2.2-2.6s
   each), 0 backward labels, 0 EINVALs — the same fixture killed pre24 at its second event.
   Invariants re-held on the fix binary: storm1 PRE/POST +40ms 0 erases; aseam +40ms pre+post
   (R-pinned relabel class untouched); agapseam ruler3 +320ms (the pre24 composite bound).
   NOTE: the owner-supplied canonical HTTV origin trace (double discontinuity + 16s dark +
   destroyed segments) does NOT crash pre24 by itself in fixture — its resume is near-
   symmetric (mism −31ms); the live chain additionally requires a backward-mismatch window ×
   a rebuild. Falsifiable prediction for the live logs: an [PTV-AFMT]/[PTV-ACHOP]/rebuild
   line sits between each crash's event cluster and its EINVAL.

(7w) PRE24 — #63 CORRUPT-STORM DESYNC: WALL-EVIDENCE SPLIT + CORROBORATED RECOVERY RE-ANCHOR.
Kills: PTV_NO_WALLEV=1 reverts Part 1's action at every touched site (provenance measurement
stays on); PTV_NO_RECANCHOR=1 disables Part 2. Tunables: PTV_RECANCHOR_SETTLE_S (300),
PTV_RECANCHOR_COOLDOWN_S (1800). v1.1.0 freeze gate; owner mandate 2026-07-24: "perfect a/v
lip sync once input is OK again, no matter what happened on input."
SYMPTOM (storm-diag 2026-07-24, arithmetic closed against a flash+beep ruler; live: OAN_Plus
+8.8s ear-confirmed, Avivando +4.9s on a +9.812s step, TV_Mundial +3.4s): a TEI corrupt storm
leaves a PERMANENT post-storm audio-early offset. storm1 repro: ruler +40ms → −4880ms baked;
ledger R=+4917 TRUE. Closed arithmetic: audio deleted (|ea|+|glue|) 17452ms − video deleted
(|ev|) 12534ms = +4918 ≡ ruler ≡ R.
ROOT CAUSE: REAL content holes (corrupt-discarded frames) were classified as label lies and
ERASED instead of padded, by three per-stream engines with no cross-stream conservation —
the A−V difference of erased totals IS the on-air offset:
  E1v LAYERA butt-joints every >1s VIDEO hole (video never had a gap discriminator);
  E1a audio erased whenever the gap verdict was unreachable (vcrossed — usually true
      mid-storm — or wall_gap < max(700ms, J/2); the J/2 boundary = Avivando's half-step bake);
  E2  pre23 conv rule B compared a NEW event's magnitude against a DIFFERENT event's
      bureaucratically-alive backlog (dl=2×mag+60s) → folded REAL storm gaps; 3 escapes →
      seam-park folded everything for 60min (−13418 of storm1's −17452);
  E3  (opposite sign) composite events (label step J containing a real gap W) got a
      WHOLE-step pad → audio LATE by J−W, sensor-blind (agapseam1: +5840ms, R wrong sign).
PART 1 — WALL-EVIDENCE SPLIT (g_wallev, [PTV-WALLEV]): split every forward label step J into
W = wall-absence-evidenced portion (clamp(delivery wall gap − cadence EMA, 0, J); floor
700ms; W:=0 under bank/bursty delivery — fall back to today, never guess) → PAD (labels keep
the hole; aresample pads audio, the house clock's starvation dups cover video) and J−W =
flowing/relabel portion → ERASE (as today). Sites: demux_unwrap gains a VIDEO gap/composite
discriminator (mirror of the v0.8.2 audio one; full-gap video does NOT stamp video_fwd_us,
so vcrossed becomes a truthful splice signal mid-storm) + the audio discriminator now splits
composites and holds full verdicts by per-stream evidence even when vcrossed; wallev-qualified
verdicts propagate to the disc buffer unconditionally (per-stream W conservation makes
one-sided handling safe); the LAYERA flush butt-joint preserves each crossing stream's W
(cumulative_ts_offset + W; rr24 F1: when the shared-flush tree overrides that offset the door
step becomes mism + W and THAT is what the flush registers — padded content = W exactly when
mism ≥ 0, and a negative mism collapses part of the hole out of the labels, posted as
unevidenced provenance so the recovery re-anchor corroborates and walks it back — the
whole-splice-with-audio-absence corner, mism ≈ −W, is recovered by Part 2 by design); the
conv classifier EXEMPTS wall-evidenced gaps (door wall_gap ≥ step/2 + cadence; rr24 F4: the
exemption AND its measurement respect the same bank/deep-prime fallback gate as the demux
side — bursty door wall gaps are delivery jitter) from fold_park/cap/ladder — bounded by
a 120s outstanding-pad HARDCAP that folds anyway with a loud admission (allocation safety
trumps sync; #60 never regressed) — and rule B's deadline is the order's REALISTIC playout
(mag + 15s, injection is instantaneous): an order that provably played out cannot be "the
same non-converging backlog" (the cross-event aliasing fix). PATRIOT-class real 30.8s steps
unchanged (in-cap, wall-evidenced → pad); HTTV pure-relabel floods still fold label-neutrally.
PART 2 — CORROBORATED RECOVERY RE-ANCHOR (g_recanchor, [PTV-RECANCHOR]): channels left with a
large STABLE R after input recovers (the corrector disarms >5s as implausible; ≤5s takes
~40min at 2ms/s) get a one-shot, health-gated, chunked base re-anchor (the [PTV-ANCHOR]
algebra: glue_off += step, ≤1s per 10s, budget |R0|×1.2, abort on any event) — ONLY when the
new UNEVIDENCED-DELETION provenance ledgers (U_a/U_v, measured at every erase engine from
event-time wall evidence, live even under PTV_NO_WALLEV) CORROBORATE it: R_pred = ΔU_a − ΔU_v
− Δcorr ≈ R within max(500ms, 10%). THE MANDATORY GUARD (aseam counterexample): a lone flowing
relabel pins R at the step forever while the wire is PERFECT — its erase was flow-evidenced,
U_a=0, R_pred≈0 ⇒ REFUSED (a naive "trust large stable R" would CREATE a desync). Engage
gates: |R|>1s, R stable ±100ms + NO events through a 300s settle window, slip=0, delivery
live, label health H (now published per track, g_rsx.hh_q10) within ±15%, 1800s cooldown.
rr24 F2: applied slew chunks are accounted across mid-walk ABORTS (Σapplied subtracted in
R_pred, reset only at COMPLETE) — a post-abort re-engage corroborates and walks only the
REMAINDER instead of being refused forever with a half-recovered channel. (Gate-only hook
PTV_RECANCHOR_TEST_ABORT_N forces one mid-walk abort after N steps, no event injected;
0 = off = byte-identical.)
Cross-stream conservation diag: one [PTV-WALLEV] line whenever the running A−V unevidenced
deletion imbalance moves >500ms (measurement only).
DOCUMENTED LIMITATION (F7): bank-armed / deep-prime channels (Fintech-class bursty
delivery) have wall evidence disabled BY DESIGN at every site — Part 1 is inert and Part 2
has no provenance to corroborate there (nothing measurable to act on): the deep-prime class
is out of scope for pre24.
GATES — MEASURED (flash+beep ruler, x264+fdk 1-rung, xcorr ground truth; +40ms = harness
baseline): smoke +40ms, 0 fires. storm1 (12 TEI bursts, 11min hands-off post window) POST
+40ms BASELINE via Part 1 alone — 0 erases (ev=ea=glue=0), 9 video + 13 audio gap verdicts.
Counterfactual (both kills): +8530ms baked (the disease; this run took the no-fold leg —
video-erase dominant, matching storm-diag's storm2nc prediction of −8.5s audio-late; the
historical eb2937cfff run took the conv-fold leg to −4.9s early — same disease, leg-dependent
sign). Part-2-only arm (PTV_NO_WALLEV=1, re-anchor live): bake −8398ms → 325s settle →
ENGAGE corroborated (pred −8057 = ΔU_a +4415 − ΔU_v +12473, tol 840) → 9 slewed steps →
ruler +40ms BASELINE (the owner-mandate arc, both actuator directions proven: pad and drop).
aseam1: wire +40ms throughout, R pinned +9800 forever, re-anchor REFUSED with pred=+0 (the
mandatory guard, live-fired). agap1 +40ms. PATRIOT 35.2s measured gap: verdict + door
acceptance + >10s mandate alert + 0 folds + POST +40ms. HTTV flood (42×90s flowing relabels):
all butt-jointed label-neutrally at LAYERA, ZERO misclassified as gaps, wire +40ms at 21min,
RSS 763→783MB over 22min (flat). mv 2×2 (4 UDP rulers): offsets steady +3..+6ms all 4 tracks,
0 corrector errors, 0 wallev fires. COMPOSITE ACCURACY BOUND (agapseam J=9.8/W=4.0 and the
Avivando J/2-boundary W=4.6): split arithmetic exact (erased+padded=J; splits logged
4291/5519 and 4950/5113), residual +320ms / +120ms = the SOURCE's audio PES-interleave
spread (this fixture's lavf mux: 245-350ms bursts; the pre-fix bakes were +5840ms /
≈half-step seconds) — production CBR TS with tight audio interleave bounds the residual
correspondingly; the leftover is sensor-blind (edit-neutral), same class as E3 but bounded
by interleave noise instead of J−W.
THREE IN-BATTERY FIXES (first storm battery caught all three live): (1) the bursty fallback
gated on d->autobank = an ELIGIBILITY flag (true on every single-input live channel) — split
was disabled everywhere; now g_bank_us>0 or PTV_PREROLL_MS≥4000. (2) corrupt-discarded and
DUKF-dropped video packets now stamp the delivery-arrival tracker — a TEI storm read as
delivery stalls, auto-bank escalated on a flowing transport and its armed bank disabled the
split mid-storm (rider benefit: corrupt storms no longer falsely arm AUTO-BANK). (3) W
subtracts cadence/2 not full cadence (unbiased under bursty PES delivery; full-cad
under-padded composites by half a burst period and under-measured U_v enough to refuse a
legitimate re-anchor corroboration at margin).
(#54 mux-death loud-fatal, startup sanity). Kill: PTV_NO_CONVCAP=1 reverts A+B+C as one;
riders: #54 has no gate (defense-in-depth), startup sanity PTV_NOVIDEO_EXIT_S=0 disables.
SYMPTOM (perception-glo-transcoder-1 2026-07-23 01:07 UTC): TV_Avivando_Nações — a
channel with recurring label-jump seams on a continuously flowing HLS→SRT→UDP relay —
grew 183GB anon-RSS in 52min and the kernel OOM kill took the WHOLE BOX down (systemd
OOMPolicy SIGKILLed the supervisord cgroup, 51 channels dark 8.4h). On-box ladder:
AGLUE steps +1.2s→2.5→5.0→10→21→42→84→172→347→+702s over 23min, every order
"left to the discontinuity layer (matches the shared-flush expected step — aresample
converges it)" = an UNCAPPED contiguous swr_inject_silence allocation (label-gap ×
48000 samples; a single 2^31-byte heap block was measured in the #60 repro). Verbatim
ladders on 3 boxes under pre21/pre22 (Rsbn cor-3 +2.6→145s survivable on 503GB;
Praise_TV glo-2 →328s); 22 channels show precursors, 4 are OOM-class. rc1 on identical
sources PARKED the step in the hs/hsres ledger (stable, ugly) — the pre15 #33 classifier
turned that parking into unbounded convergence pursuit.
MECHANISM: each seam's A-vs-V mismatch is routed to the audio content path by the
shared-flush handshake (registered expected step); the convergence never completes
before the next seam; the re-measured mismatch stacks the unconverged backlog into
each new order (~2× the last).
FIX — three layers in the AGLUE above-cap branch, ONE remedy (the pre20 LABEL-NEUTRAL
FOLD: glue_off_us -= step — door labels continuous, nothing reaches the resampler, the
sensor/corrector own any real residual), one loud [PTV-CONV] line each:
  A ADMISSION CAP (PTV_CONV_CAP_S, default 60s): a single accepted order above the cap
    is folded, never handed to aresample. THE 2026-07-18 Q5 MANDATE IS PRESERVED for
    in-cap orders: PATRIOT's 30.8s (the largest REAL convergence ever observed) passes
    with 2× margin, and an accepted convergence stays latency-UNBOUNDED while it
    SHRINKS — the mandate governed healthy-source convergences, which all realize
    instantly (hard comp); only the pathological ladder is refused.
  B LADDER ESCAPE: a NEW above-cap step arriving while an order is in flight (within
    2×order+60s of acceptance — injected silence plays out ~1× realtime through the
    delivery gate) whose magnitude is NOT SHRINKING vs the in-flight order is the
    ladder signature: the ENTIRE backlog folds, no bigger order is stacked. A recurring
    equal-size seam train folds too — by design: recurrence IS the chunk-seam channel,
    even when individual orders drain (one-shot events never see a second step).
  C SEAM-PARK: ≥3 fold-escapes (A or B) on one track within a rolling hour → every
    above-cap step folds immediately for the next hour (rc1-like parking, label-neutral
    and loud; entry/expiry logged; park folds don't re-count so the park self-expires).
FOLDED steps are ERASED at the door: no §2.4 pend_comp tripwire, no E5 pad-ledger entry
(a later matching backward step takes the normal RELABEL erase and nets the fold to
zero — treating it as a pad round-trip would drop content that was never padded), no
>10s mandate alert. NOT folded (exempt): pad_cancel and backward fill-resume overlaps —
they DROP our own inserted silence (no allocation; folding would bake it). exp_hit does
NOT exempt — the ladder's own orders carry the expected-step suffix.
RIDER #54 (mux-death loud-fatal, no env gate): a mux write error used to end mux_thread
with ZERO log lines — wire dead, process alive (the silent zombie), and the closed
delivery gate then freed every audio packet into the dead mux_q, removing all
backpressure (the #60 sustained-allocation enabler). Now: one [PTV-MUX] AV_LOG_FATAL
line (rung, stream, error) + exit(1) for supervised respawn. Measured (G5, connected-UDP
wire broken at t=25): FATAL + rc=1 at t=28; pre22 control = stats frozen, alive 75s+,
zero lines.
RIDER STARTUP SANITY (PTV_NOVIDEO_EXIT_S, default 300s, 0 disables): input packets
flowing (demux heartbeat fresh) but no video frame decoded since start parks the master
output thread BEFORE the stats block forever (zero log lines — the wedged-startup
shape). Now: [PTV-NOVIDEO] FATAL + exit(1) at the deadline. Master rung, single-input
live path only (the mv compositor never parks there). A dead source stays rw_timeout /
[PTV-REOPEN]'s job (the flowing check). NOTE (measured): an audio-only/no-video-PID or
garbage-video source already exits LOUD at open (~5-10s, buffersrc 0x0) — the state this
rider covers is probe-OK-then-never-decodes (startup vdec wedge, the Azorse futex class;
gate fixture = PTV_TEST_VDEC_STALL_AT_S=0, a new TEST-ONLY trigger-time override for the
pre21 stall injector).
GATES (fixture = seam_relay chunk-seam UDP relay; the `asym` mode — video +J, audio
+J+D per seam — reproduces the EXACT production entry lines: "paired flush: expected
audio label step +Ds registered … content path will APPLY it" → above-cap AGLUE with
the expected-step suffix → ">10s convergence in flight"; every gate carries a
PTV_NO_CONVCAP=1 / env-disabled / pre22-binary broken arm as the liveness proof):
G1 Avivando shape (routed +90s per 45s seam, 16min): CONTROL grew the PRODUCTION
LADDER (+90→+141.6→+411.7→+1221.4s re-measured orders, RSS 1.1MB→970MB and climbing,
async −1.9Mppm, dlvhold 21.8s) — FIX: 3 A-cap folds → seam-park at escape 3 → in-park
folds, RSS 181MB FLAT 15min, wire A/V dts spread ≤40ms to the end. G1b (routed +40s
per 30s): B ladder-escape fired AND the production ×2 re-measure reproduced (+40s
accepted → next order arrived +80s [1668 pkts shed] → A-cap fold); RSS 201MB flat.
G2 PATRIOT (+31s one-shot): pre23 vs pre22 LINE-FOR-LINE identical (accepted, mandate
alert, 0 [PTV-CONV], lipsync +13ms both) — the mandate is preserved. G3 (+45s
one-shot): the initial order ACCEPTED (mandate honored); this 1-rung cell then
self-degenerates (QSHED shed-gaps = secondary above-cap steps on BOTH binaries — the
pre22 control handed 57 of them to aresample) and pre23 correctly parked the train;
wire spread 25ms. G4 recurrence (PTV_SEAM_PARK_S=600 TEST override): entry at escape
3 → per-seam in-park folds → "seam-park expired — converging re-enabled" — full
lifecycle, RSS 187MB flat 20min. G5 #54: FATAL + exit rc=1 within 3s of the wire
break; pre22 control = frozen stats, alive 75s+, zero lines. G6 startup wedge
(PTV_TEST_VDEC_STALL_AT_S=0): [PTV-NOVIDEO] FATAL rc=1 at t=33 (30s deadline);
disabled control parks silently forever. G7: healthy = log-vocabulary identical +
per-5s output byte counts IDENTICAL to pre22; bursty stop/go = [PTV-BURSTY]/bank
engagement identical (bank target Δ4ms); p25 anchor fixtures F1/F2R re-pass (F1
lipsync decays to +3ms as the pre22 reference, F2R supersede path byte-same).
MEASUREMENT BOUND (reviewer-confirmed, rr23 F1): on fold/park channels read the WIRE +
the [PTV-CONV] lines — lipsync= includes the folded label divergence BY DESIGN (the
fold refuses the source's label motion; reviewer measured lipsync= +1,828,749ms at a
wire A/V dts spread of 80ms) and async= reads Mppm for the same reason. The rr23
`conv=` stats token quantifies it: appended only when a track's cumulative folded
label motion is nonzero (healthy lines stay byte-identical), integer seconds of net
folded motion, `P` suffix while SEAM-PARKED (timestamp-derived, never stale), e.g.
`conv=+141sP`. rr23 F5: seam-park expiry is logged eagerly (per fed frame, the owning
audio thread) so every park-entry line gets its matching expiry line even when the
seams stop mid-park.

(7v) PRE22 — lone-VIDEO-jump audio-anchor (the role-swapped D1 mirror; kill
PTV_NO_AANCHOR=1).
SYMPTOM (Fashion live 2026-07-22 13:21:42): a lone VIDEO label jump +5.520s whose
audio partner never crossed inside the pairing window was one-sided relabel-erased
(flush 41 pkts old=26 new=15 applied=−5.480 vid_err=−5.480, "partial flush (only
video crossed)") — video labels glued, audio untouched, NO registration, NO
expected-step handshake: the ev ledger takes the full −5480ms, R pins there, the
corrector correctly DISARMs (>5s implausible), and if the video content really
skipped, the wire is desynced by the jump amount with no remedy. ~0.5s of flowing
audio (old=26) also deleted. The pre21 D1 anchor only covers lone TRANSCODED-AUDIO
jumps; the video-trigger mirror fell through to the old one-sided path.
FIX (deferred, two-phase — the one deliberate design deviation from a flush-time
mirror): D1's discriminator (video position classifies NEW) is INSTANTANEOUS
evidence; the mirror's evidence — "the audio partner never arrives" — is only
provable at PAIRING-WINDOW EXPIRY, and an immediate shift+registration at the video
flush would destroy the genuine staggered pair the 3a INHERIT machinery handles
(Fashion 2026-07-20 10:14 class). So:
  SEED (at the lone-video flush): transcoded-video trigger + no audio crossing or
  pair state this window + audio flowed own-continuous + trigger jump FORWARD (R2
  mirror; backward lone flips — the ±11594s class — keep the doctrine-correct
  erase, wraps excluded) + trigger SETTLED at NEW (an intra-cycle jump+return never
  seeds) → each flowing transcoded audio stream gets a PROVISIONAL zero-offset leg
  (pair_prov=1, applied=0 — "audio didn't jump"): cont_eligible now KEEPS the
  flowing audio (the old=26 discard is gone), the 2d retro-correct skips seeded
  streams, aanch_pend (R3 mirror) keeps the pair state through the end-of-flush
  close, and R1-mirror stale-window expiry is the seed block's first act.
  VERDICT: a real audio leg crossing in-window CANCELS the seed (free — nothing was
  shifted) and the pairing machinery owns the event byte-identically; window EXPIRY
  un-crossed FIRES the anchor on the quiet path (ptv_aanch_fire, before
  demux_unwrap so the firing packet rides the shifted refs): audio re-based onto
  the video-defined timeline (corr = pair_vid_off = the FULL video delta) +
  ptv_pair_expect registration — the D1 handshake; the content path (#33/AGLUE +
  the #47 caps) judges the step, the demux never hard-commits to "real splice".
  The 2d refuse gates apply unchanged at the fire (120s route cap + label-health
  H): a refused anchor logs loudly and falls back to the pre21 one-sided erase.
MAGNITUDE BOUND (measured, F1 gate, +5.24s fixture): the registered backward step
is realized by aresample=async as BOUNDED SOFT compensation — ~2.1%/s (the
async=1000 slew), ≈ +22900ppm decaying to ~+1000ppm over ~3.5min for a 5.2s step;
no output hole, no swr starvation, slip stays 0, the #47-B tripwire never arms.
Content alignment (flash+beep ruler): PRE +32ms → POST +32ms on the fix arm vs
POST +5232ms on the PTV_NO_AANCHOR control. ev=ea after the fire → R decays
+229→0ms instead of pinning at −5199; lipsync stat +0ms (control −5199 pinned).
KNOWN BOUNDS (documented for review): (1) a LABEL-ONLY lone video restamp with no
return is INDISTINGUISHABLE from a real video-content skip at every layer — the
anchor converges content onto the labels, which for the label-only shape means
~step of real audio dropped (measured F2 gate: ruler POST −5168ms audio-early,
R reads +0 — label-blind; the pre21 control keeps the wire at +32ms with R pinned
−5199). An in-window return (≤5s label flip round-trip, the realistic transient
restamp) is SAFE — the return's 2a cancels the pending seed with zero side
effects (measured F2R gate: 0 anchor/0 expect lines, ev round-trips to 0, ruler
+32ms); a return AFTER the fired anchor leaves the created step standing until
operator action — sensor-visible at the return (R steps to ev−ea = +step). (2) a genuinely-paired event whose audio
leg arrives AFTER the window (>5s stagger): pre21 self-healed at the leg's own
erase; pre22's fired anchor makes that a ≤step residual — sensor-VISIBLE (R steps
at the leg, ev≠ea) and one-log-read attributable (anchor line → later audio 3b).
(3) copy-only audio (AC-3 passthrough) is never seeded (no content machinery to
apply the step) — its labels stay untouched, exactly the pre21 shape for that
track (the standing D1 copy-audio bound, unchanged). (4) gap-verdict compound
(rr22 finding 3, traced benign): a WALL-ABSENT staggered audio leg routed by the
§2.5 gap-verdict propagation bypasses the seed cancel (no flush, no has_aud —
the vcrossed debounce is 1s vs the 5s window), so the anchor still fires on top
of the AGLUE gap-pad. For a real splice the two are COMPLEMENTARY (the pad
restores label continuity, the anchor realigns content; pre21 left a permanent
step); for a label-only restamp it degenerates to the adjudicated bound (1).
(5) mv-only reopen residual: an anchor firing after demux_reopen_once rides the
preserved pair state — bounded ≤ the step; the rejoin glue folds or refuses it.

(7u) PRE21 — D1 lone-audio anchor fold + #24 PLL/corrector arbitration + [PTV-STALL]
heartbeats. Four commits: d1-fix merge (3cb74d8b4a + da218b1425, adversarial-reviewed
MERGE-READY), #24 (c98de627d1), heartbeats (6824e9b4e8), this docs/version commit.

ITEM 1 — D1 REGRESSION FIX (lone-audio video-anchor; kill PTV_NO_VANCHOR=1).
SYMPTOM (Grid_2x2 in3 live 2026-07-21 14:08, owner ear-confirmed): a lone AUDIO label
jump +1.344s flushed as a one-sided butt-joint (old=22 new=5 applied=−1.320, NO
"expected audio label step" line) → audio content 1.32s early permanent (R=+1322,
ea=−1320) + 22 flowing video pkts DELETED ([PTV-REANCHOR2] video-ahead +860ms).
Verified working on pre5 (2026-07-14: old=0 vid=+0.020, step +1.700 registered).
ROOT CAUSE: pre7 (77e7410e61) closed the borrowed-base false crossing that pre5 was
ACCIDENTALLY relying on for this shape — post-pre7 the flush fell to 3b (provisional,
no registration, video discarded via cont_eligible's pair_vid_defined gate).
FIX: deliberate video-anchor at flush time — lone transcoded-audio trigger + video
flowing own-continuous + video position classifying NEW against the trigger's bases
(+ rr-d1 fix round: R1 stale-window expiry first, R2 FORWARD-only trigger jumps, R3
pair_anchored survives the end-of-flush close ≤5s so a staggered video leg finds the
pair state, declines the #47-C hold, and 2a-flushes immediately).
KNOWN BOUND (reviewer-accepted, ARM-C): a genuinely-paired event whose video leg
arrives >500ms after the anchored audio flush bakes a pad-vs-erase residual ≤ the
step, sensor-visible. ONE-LOG-READ SIGNATURE: anchor line → video 2a flush → R step
with ev=−step. Closure of that residual rides ITEM 2 (#24). pair_anchored persists
≤5s (window expiry / 2a) — an independent same-window audio re-cross takes 3c
(butt-joint) exactly as before. Gates: 7-arm battery (ARM-A/B/C adversarial + main
fixture + PTV_NO_VANCHOR parity + clean-source + fx50 veto) all PASS, all wire-alive.

ITEM 2 — #24 MV PLL/CORRECTOR ARBITRATION (kill PTV_NO_PLL_YIELD=1).
MEASURED DISEASE (p24 gate-1): the mv audio-follow PLL and the residual corrector
share the graph-door actuator with opposing objectives — corr walked +655ms at full
slew (slip=0) while ΔR=−1ms and the PLL steer series read exactly −2ms/s from the
engage: the PLL trims back every µs the corrector realizes, 1:1 → mv ENGAGE→PARK
structurally unreachable, every engagement ends in an authority-cap DISARM.
FIX: the PLL YIELDS (af_steer_us integration + ACQUIRE drop/pad frozen; measurement
chain live) while THIS track's corrector is ENGAGED; resumes on PARK/DISARM; one
"[PTV-RSCORR] aN PLL yields/resumes" line per transition. BUMPLESS RESUME (8d6f4de4bb):
the PLL measures relative to a bias adopted at each yield-resume (bias += ema,
smoothing re-seeded) — without it the first post-PARK resume reads the corrector's
DELIBERATE walk as fresh misalignment and ACQUIREs it back (a structural ~12min
sawtooth exchanging the full step per cycle); when the PLL's pairing measure agrees
with R (real-content-gap case) the resume ema is ~0 and the bias is a no-op. ALSO
(owner-approved shape): a lifetime-authority-cap DISARM is now FINAL for the process
(perm_disarm, logged once) — 10s of failed trim must not re-walk the staircase
forever. GATE (b-yield, broken-shape fixture): ENGAGE R=+1391 → walk slip=0 → PARK
R=+1391→−17ms corr=+1452ms in 811s → "PLL resumes ... bumpless: adopted −1380ms";
post-PARK R flat −16ms, zero re-engage/ACQUIRE churn. CONTROL (+PTV_NO_PLL_YIELD=1):
the old disease byte-exact — corr walks +476ms while R stays +1358 flat (dR/dcorr≈0).
rr21 A1 (review fix round): the bias is a CALIBRATION of the PLL's label-domain
pairing miscalibration and any label-baseline redefinition strands it — the concrete
class: a TruBLU splice-RETURN backward jump relabel-erases and CANCELS the
miscalibration an erase-class walk encoded → av_offset returns ~0 while the bias
holds −walk → +walk PHANTOM the PLL would ACQUIRE within seconds (permanent
mis-alignment after a lifetime-cap perm_disarm). FIX: STALE-MARK + settled-window
RE-ADOPT (never blind-zero): ptv_pll_bias_mark at (i) ptv_rebuild_reanchor, (ii) a
per-frame watch on the demux label-edit ledger (g_rsx.ea_us — LAYERA persists,
absorber self-rebase, retro-correct) + AGLUE glue_off changes, (iii) the resume path
itself (adoption is now ALWAYS deferred — a DISARM-path resume can no longer adopt a
mid-transient sample, closing the mid-engagement-rebuild hazard). While stale the
PLL's actuators (acquire+TRACK) suspend, measurement runs; adoption at ≥128 flat
frames (~2.7s) or the 1280-frame hard cap (~27s). bias==0 un-forced marks no-op
(never-engaged tracks byte-identical).

ITEM 3 — THREAD-POSITION HEARTBEATS + [PTV-STALL] (always-on, reporting only).
CONTEXT (Azorse live wedge 2026-07-21 10:43:58; cor-1 1.0.0 survived the same source
event): vdec froze in a futex with video_q FULL and frame_q EMPTY — the log never
said WHERE. Now: vdec stamps q_recv/bank/send_pkt/recv_frame/hw_upload/fq_send +
wall; the stats owner (single-input master output thread / mv compositor) stamps its
loop points; demux stamps arrival+vq depth. The per-rung watchdog thread (different
thread, 500ms) reports "[PTV-STALL] vdec thread (inN) stalled Ns at <position>
(dec=, vq=, frameq=) — input flowing" when a stamp is >5s stale while the demux
stamp is fresh (<2s); rate-limited 60s/slot. LIVE-FIRED via TEST-ONLY
PTV_TEST_VDEC_STALL_S (the pre20 silent-zombie rule: no unfired diagnostics).
Reviewer notes carried (rr21, non-blocking): (1) on mv BOTH the compositor and the
master rung's output thread write the single OUT stamp — an output-thread stall can
be masked by a live compositor (UNDER-reporting only, never noise); give the
compositor sole OUT ownership (or a second slot) in a later pre. (2) a genuine >5s
graph/hwupload block under GPU load WILL print [PTV-STALL] — that is a TRUE POSITIVE
by intent (the NVENC-block class), ERROR-level, rate-limited.
Gate tables: analysis + per-cell logs in the session scratchpad (pre21-progress.md);
summary in PROGRESS.md.

(7t) PRE20 — REBUILD RE-ANCHOR (headline) + behavior-inert cleanup sweep + riders.
The last content pre before the v1.1.0 freeze. Three code commits, in order:
cleanup (parity gates bind to it alone), re-anchor, riders.

ITEM 1 — AFMT REBUILD RE-ANCHOR (kill PTV_NO_REBUILD_REANCHOR=1).
SYMPTOM (Azorse live 2026-07-20 10:54): an [PTV-AFMT] rebuild (format flap /
post-outage resume / ACHOP escape) inherited its audio base from carried state
— the transition duration leaked in as a ~1s residual (R +1021ms) the corrector
then walked for ~10min (599s). The source is A/V-synced at the change; the
residual is OURS. ROOT CAUSE (carried state enumerated): glue_off_us polluted
by broken-phase relabel-erasures; the swr-fallback next_pts free counter frozen
across the outage; a live corr trim sized against the dead mapping; a stale m_a
EMA blending pre-rebuild samples. FIX (three pieces):
 (a) ptv_rebuild_reanchor() at rebuild COMPLETION in audio_feed (ACHOP rebuilds
 complete through the same site): fresh AGLUE label baseline (no step
 classification across the rebuild), pad ledger + pending tripwire cleared,
 swr-fallback counter re-derived via ptv_anchor_next_pts() — the factored
 single copy of the [PTV-ANCHOR] birth formula (birth calls it too; the two
 can never diverge) — clamped FORWARD-only (mux invariant). CORR RETIREMENT
 DECISION (fix-round rr20 F1 corrected): retirement is a LABEL-NEUTRAL
 BOOKKEEPING TRANSFER — corr folds INTO glue_off (inj sum unchanged ⇒ door
 labels perfectly continuous ⇒ no backward step can arise at retirement),
 logged with the amount; a DWELL/ENGAGED/PARKED corrector falls back to ARMED
 (fresh full dwell, the §4 re-engagement rule) and the trim never
 double-applies because the (c) step below is sized by the MEASURED
 post-transfer R. THE FIRST CUT zeroed glue_off + corr outright — CONFIRMED
 BLOCKER (rr20, 3 sites): retiring a positive sum stepped the door labels
 BACKWARD by it → backward audio DTS → mpegts "non monotonically increasing
 dts" → mux_thread exits on first write error → DEAD WIRE, zombie process, no
 watchdog (flap222 died at 54.09s on rebuild-2, walk at 264.57s retiring corr
 +228ms, gap90 at 205.2s — ANY channel rebuilding twice died at the second
 rebuild). Zeroing glue was also WRONG in the balanced case: the sensor pairs
 glue against the demux edit ledgers (R = dm + E_v − E_a), so glue that
 mirrors ledgered video edits is label-TRUTH, not pollution — discarding it
 re-opened R by the ledger difference (measured: dm −261 / ev +821 → R +560
 baked).
 (b) m_a EMA re-seed — the rebuild joins the pre17 baseline-redefinition
 re-seed list (REANCHOR2 / ACQUIRE / slate-recovery); m_v deliberately NOT
 re-seeded (an audio-path rebuild does not move the slot's video mapping).
 (c) bounded residual step(s) INSIDE the rebuild discontinuity window only
 (defined precisely: 10s wall from rebuild completion; each fire needs a fresh
 ≥10-emitted-frame settle; MULTI-step with a ±5s TOTAL budget per window —
 fix round: a seam's video-side [PTV-DISCONT] ledger edit can land ~500ms
 AFTER a first one-shot fire and re-open R, measured on the flap fixture):
 |R| > engage band → glue_off += R (one WARNING per fire carrying the window
 total; dR/d(glue_off) = −1, the same door algebra as corr — verified against
 the sensor inj identity m_a = inj − h0 − slip on the fg path; pend_comp
 COMPOSED, not clobbered — rr20 minor). The rebuild is already an audible
 discontinuity, so the steps are free; glue_events++ hands the corrector a
 fresh dwell for leftovers. Outside the window no step can EVER fire
 (reanch_wc zeroed at expiry). A NEGATIVE step moves the door labels BACKWARD
 — it arms the MONOTONIC EMISSION GUARD (the mux must NEVER see a backward
 audio DTS, absolute invariant): frames whose output label does not exceed
 the last emitted one are DROPPED (the birth opts<0 rule, mirrored) until
 the labels catch up naturally — bounded by the step size, released with one
 WARNING carrying the dropped count; disarm is catch-up-only (a wall expiry
 would re-open the backward window). (c) is single-input/non-follow ONLY: on
 the mv audio-follow path the closed-loop PLL owns content alignment (it
 measures out_v−out_a directly and would re-track a base step — two actuators
 on one displacement); there (a)+(b) still apply and the pre17 ACQUIRE
 re-seed restores honesty within ~an acquire; mv-follow emission is already
 monotonic by the af_last_out clamp.
GATES (fix-round rr20 F2 battery — every cell WIRE-LIVENESS-asserted: output
still being written at kill time, ffprobe duration == cell duration − startup
(floor runs−15s), stats lines to the end, zero muxer non-monotonic errors;
8/8 cells PASS, controls included; fresh 70xx ports, pid-scoped kills;
pre20fix-item1.sh):
 (i) synth flap p20_flap222 (30s stereo → 5.1 → stereo, continuous ts; 5.1 not
 7.1 — ADTS cannot carry channelConfiguration>7, tick-in covers the real 7.1
 class): pre20 = 2 rebuilds, label-neutral retirements, residual steps +261ms
 (rb1) and +560ms (rb2 — the MULTI-step design earning its keep: rb2's window
 total includes the seam's late video ledger edit), lipsync=+0ms on ALL 48
 samples, wire alive 96s (the pre-fix cut died HERE at 54.09s on rebuild-2);
 pre19.1 control AND kill control: lipsync +0→+261ms→+821ms compounding
 residual (IDENTICAL histograms 12/15/21 samples — kill parity exact), both
 liveness-PASS (the no-reanchor path never had the blocker).
 (ii) outage-resume p20_gap90 (60s stereo / 90s NO audio packets / 60s 5.1
 resume): pre20 = sensor `--` through the outage, resume rebuild + ONE +821ms
 step 4ms after it, 59×+0ms (2 in-window transient samples), the fixture-loop
 seam's SECOND rebuild folds the carried +821ms with NO step needed, wire
 alive 211s (pre-fix died at 205.2s on that second rebuild); pre19.1 control
 = +821ms FLAT (34 samples; the corrector would walk it ~10min — the live
 Azorse class at magnitude). NEGATIVE controls (recorded): p20_dead90
 (all-XOR) and p20_chop90 (every-3rd XOR) produce ZERO decode errors on this
 source class (demux discard / parser drop) — no rebuild fires, both builds
 +1ms after heal; ACHOP-triggered rebuilds covered BY CONSTRUCTION (same AFMT
 completion site; pre19 #46 gate stands) — an ACHOP-reaching local fixture
 for this class is an open fixture gap, noted.
 (iii) birth parity: [PTV-ANCHOR] birth line identical (first_audio-h0=+8ms,
 h0=5781ms) across pre20/ref/kill; the item-2 single parity cell's audio ES
 byte-identical.
 (iv) corrector interaction (p20_walkflap, test-scaled dwell 60s/quiet 30s,
 TESTWALK 5000µs/s capped 300ms at t=30s, FRESH ports — the first run was
 port-collision-contaminated): ENGAGE R=+300ms → corr steered to +229ms at
 the flap; rebuild logs "corr +229ms retired into the base ledger (door
 labels continuous)" — the fold keeps the realized trim (real content
 alignment), corr → 0, corrector re-dwells, NO second engage; ONE +251ms
 step (the measured post-fold residual); wire alive 326s (the pre-fix cut
 died HERE at 264.57s — the AWE parked-corr class); post-rebuild stats
 lipsync flat −298/−299ms = the capped synthetic walk turned into a real
 offset by the step (the documented TESTWALK artifact: corrector-read
 R = −299 + walk(+300) ≈ 0, quiet).
 (v) kill PTV_NO_REBUILD_REANCHOR=1: zero re-anchor lines, lipsync histogram
 IDENTICAL to the pre19.1 control sample-for-sample, liveness PASS.
 (i-tick) tick-in 100s: no AFMT fires on the combined 0004-tolerant build
 (capture decodes continuously as 7.1); lipsync +0ms all 48 samples BOTH
 builds, liveness PASS both — broken-phase parity clean.
ITEM 2 — CLEANUP SWEEP (behavior-inert; separate commit, gates bound to it).
Dead code removed (each provably unreachable or write-only): g_vout_us (mv,
write-only since the probe era), RsyncSense.n_in (assigned once, never read),
DemuxArgs.splice_ref_v (declared, zero uses), AudioState.nopts_stamped
(write-only pre19.1 counter), ptv_qsnap() + its two PTV_DIAG call sites
(QSNAP-era probe of the CLOSED v0.9.8 feed-drop investigation), g_vindbg +
[PTV-VINDBG] lines + the PTV_VINDBG env (self-described TEMP pre13 probe;
investigation closed with the governor trust gate). Demoted: "[PTV-SWRDELAY]
sensor armed" INFO→PTV_DIAG (slip-probe investigation closed pre11; the
periodic readout was already DIAG-only). KEPT deliberately: DlvGate.st_dropped
(write-only but marks REAL non-blocking copy drops — future stats-token
candidate), [PTV-HDBG] (env-gated test hook), [PTV-START]/[PTV-H0]
(DIAG-gated), estimator [PTV-CLOCK] lifecycle lines (legend-documented). No
TESTWALK/TESTHS/TESTNOISE chatter exists outside their envs (verified).
STATS-LINE AUDIT: every single-input token (frame/fps/time/dup/pd/drop/
corrupt/async/dlvhold/dlvforced/vdlvhold/vdlvforced/wucr_buf/wucr_rho/hs/
hsres/cushion/fqhw/bank/cf/decim/acor/lipsync/corr) and every mv token
(dup/drop/dlv*/acor/inK:qdrop/corrupt/pd/sv/sk/skres/lipsync + corr) verified
live-wired; NO always-zero/dead tokens found, nothing removed. corr= present
when armed on both printers; acor= conditional >0; ACQUIRE-line backoff=
visibility is the rider-(c) fix. Full format (every token, units, when
present) documented in analysis/ptvencoder-usage.md §"Reading the stats line"
— parseable by sync_check.
GATES (pre19.1-ref vs cleanup-only binary): single 65s cinestar UDP cell —
audio ES BYTE-IDENTICAL across ref/ref/cleanup, video md5 differs even A/A
(x264 nondeterminism, the pre18 control), event-sequence diff == exactly the
demoted SWRDELAY line (A/A floor 0); mv 2x2 60s — A/B event diff == A/A floor
(birth-trim count digits) + the 4 demoted SWRDELAY lines, dup=0, stats shape
identical; tick-in broken-phase 40s — audio ES BYTE-IDENTICAL, silence-hole
count identical (0/0 over the first 35s), AFMT/ASTAMP/ADEC counts identical.

ITEM 3 — RIDERS (one batch commit).
(a) rr191 F3: [PTV-ASTAMP] extrapolation carry INVALIDATED across a demux
 REOPEN — demux_reopen_once stamps per-input g_reopen_wc; the audio thread
 consumes the edge and drops dec_ts_carry (a fresh join must never be stamped
 pre-outage-continuous).
(b) -t IMPLEMENTED (house-clock output media time; was parsed-and-ignored).
 The cadence owner (single master rung / offline rung 0 / mv compositor) sets
 g_t_stop at -t of emitted output; every demux (incl. the mv reopen-retry
 loop) treats it as EOF → stop pulling → the proven file-EOF flush/teardown →
 clean rc-0 exit. Final duration ≥ -t by the buffered residue (-t bounds the
 INPUT pull). PLACEMENT: -t is an OUTPUT option (OPT_PERFILE|OPT_OUTPUT) — it
 must come AFTER -i, with the output options; a -t found BEFORE -i logs one
 WARNING ("-t before -i is ignored") — fix round rr20 F5, was a truly silent
 ignore and ffmpeg users expect input-side -t. The parse line confirms
 consumption of the output-side form. Smoke: -t 20
 file cell exits clean in 3.7s wall, 30.4s output (offline queues hold ~10s);
 -t 30 LIVE UDP cell exits rc 0 at 31.3s output (live residue = the ~1.3s
 cushion). Every future gate cell can now self-terminate.
(c) rr19 finding 3 (cosmetic): ACQUIRE line prints backoff= only while
 g_acq_backoff is enabled (disabled builds read "backoff=0" as armed-at-0);
 line bytes unchanged when on.
(d) vist-null hygiene + C2 partial-hold simplification: NOTE-AND-SKIP — the
 partial hold is already factored (ptv_disc_partial_hold, pre17 R3a, window
 cap + capacity guard), vist reads are null-guarded, no "simplify later"
 markers remain. Nothing genuinely simple left.
(e) ceiling-mirror gate cell added to the gates/ collection (p20-ceilmv.sh):
 the #51b anti-starvation ceiling exercised on MULTIVIEW (2-up, TESTHS
 staircase + PTV_NO_HSTICK_FILTER churn + capped TESTWALK, CEIL_MIN=2
 test-scaled): PASS — ceiling ENGAGE on BOTH tracks (a0 R=+299ms, a1 +304ms
 "starvation ceiling 2min ... dwell never completed"), ZERO plain dwell
 engages (the churn starves the dwell, as designed), event-storm disarms
 between — the single-input (C) shape reproduced on mv.

(7s) HOTFIX pre19.1 — broken-phase audio "ticking" (Azorse open-during-broken-
phase; the #38 follow-up). SYMPTOM: on the Azorse broken-7.1 capture the pre19
combined build (0004 lavc + #38 hook) played 21ms of audio + ~107ms of silence
per 128ms — a metronome tick (live capture 200 holes/min, local repro 450/min,
both ~1 hole per 128ms audio PES). ROOT CAUSE (measured): the defect window is
OPEN-during-broken-phase. lavf's find_stream_info probes with its OWN decoder,
which the #38 by-name hook never touched → strict probe rejects every frame
("channel element 1.0 is not allocated" ×294 at open = the probe, not runtime)
→ "Could not find codec parameters"/sample_rate=0 → the mpegts AAC parser
still SPLITS the 6-frames-per-PES payloads, but with no sample_rate avformat
cannot compute parsed-frame durations → frames 2..6 of each PES arrive at the
decoder with NO pts → the audio_push NOPTS drop rule (1.0.1, mux-wedge guard)
discarded 5 of 6 decoded frames. 1-in-6 surviving 21ms frames + aresample
silence fill = the tick, exactly. FIX (fftools-only, g_adts_split default ON):
 1. ptv_find_stream_info(): per-stream {tolerant_ch_alloc=1} dicts into
 avformat_find_stream_info at BOTH open sites (open_input_thread +
 demux_reopen_once) — the probe gains the runtime's tolerance, params resolve,
 avformat stamps every split frame. The option goes into EVERY stream's dict,
 not audio-only: lavf's try_decode_frame indexes options[] by the stale
 "first stream still missing params" loop variable, NOT pkt->stream_index
 (demux.c:2937) — measured on the fixture: audio-only dicts left the aac probe
 strict because it was opened with the VIDEO stream's dict. Unknown-option
 safe (dict leftovers are ignored; stock lavc = strict probe as today).
 2. [PTV-ASTAMP] backstop in audio_thread: a decoded frame with no
 best_effort_timestamp is stamped by sample-count extrapolation from the
 previous stamped frame; the carry is INVALIDATED on any decode error (the
 garbage-tail class the NOPTS drop rule protects against) and on decoder swap,
 so only contiguous clean decode is extrapolated. One-shot WARNING on engage.
KILLS: PTV_NO_ADTS_SPLIT=1 reverts both pieces (= pre19 ticking) — the name
is historical (from the pre-diagnosis brief): the parser SPLITS fine either
way; the env gates the tolerant-probe + ASTAMP fix, not any splitting.
PTV_NO_TOLERANT_DEC=1 gates the #38 runtime hook AND the probe opts (= full
strict pre-#38: flood + silent-but-alive track; rr191 note: ASTAMP stays
armed under it — moot, strict broken-phase decode yields no clean NOPTS
frames and healthy streams never produce them).
GATES (tick-in.ts = tsp capture of the live broken phase; 5ms-block −45dBFS
hole detector, ≥20ms holes, first 3s/last 1s excluded): FIX = probe flood 0
(was 294), params found, ONE tolerant qualification line per decoder instance
(probe + runtime), ticking GONE — 53 scattered content-shaped holes/60s
(20–105ms, no periodicity) vs the source's OWN content floor of 108/60s and
rms −19dBFS (the fixture carries corrupt-packet bursts and real silence; the
literal ≤5/60s target predated measuring the source floor). ASTAMP backstop
covers the residual un-stamped class (engaged only while the probe fix is
absent/killed). KILL cell PTV_NO_ADTS_SPLIT=1 = 294 flood + 453 periodic
~100ms holes/60s (pre19 reproduced); PTV_NO_TOLERANT_DEC=1 = 3328 strict
errors, audio dead, video flowing (pre-#38 reproduced). HEALTHY parity
(cinestar 65s UDP cell, fix vs both-kills): output SIZE-IDENTICAL with
IDENTICAL hole map (113 content-silence holes at identical positions) and
zero probe/tolerant/ASTAMP lines both — the healthy probe path is untouched.
fx51 corrector cell re-run green (see LOCKED SET note below).

(7r) BROKEN-AAC BATCH — pre19: three items (#42 / #46 / #38), one commit each.
(#42) swr_convert SIGSEGV on a dead audio path at EOF. SYMPTOM: process crash
on the broken-AAC awe/Azorse chaos capture (pre11 crash report 2026-07-16:
SIGSEGV KERN_INVALID_ADDRESS at 0x3ab8 in swr_is_initialized inlined in
swr_convert, swresample.c:722/:732). ROOT CAUSE (measured, fixture-reproduced
byte-for-byte): fault address == offsetof(SwrContext, in_buffer.ch_count)
dereferenced through NULL — the v0.9.17.1 dead-path NULL guard covered
audio_feed only; the audio_thread EOF/death flush still called
swr_convert(a->swr, ...) unguarded, and a track whose path init FAILED at open
(undecodable broken-AAC phase: use_fg==0, swr==NULL, AFMT retry pending)
crashed the moment the stream hit EOF or died. FIX: guard the flush; plus AFMT
rebuild hardening — the rebuild configures swr/graph from a->dec while the
CONFIRMING frame is what gets fed next, so a decoder-context/frame divergence
(broken-AAC per-frame reconfig) could feed a frame with fewer channels than
the configured input layout into swr_convert (reads through nonexistent plane
pointers); mismatched frames are now dropped and detection re-arms.
GATE: all-XOR-corrupt-audio Azorse capture as file input — pre18 exit 139
(fresh crash report, identical signature), pre19 exit 0 clean EOF; also
re-proven on the strict-lavc broken-71 TS run (init-failed track to EOF).
(#46) [PTV-ACHOP] post-storm audio-chop stuck-state escape (g_achop,
default ON). SYMPTOM (Azorse class, sensor-soak byproduct): a corruption
phase that NEVER ends sustains repeated self-shed/erase events and the slot
warbles/chops until a process restart; pre18 #49 backoff tamed the ACQUIRE
churn but not the stuck decode->graph->swr LATCH. FIX: per-track 10s windows
in audio_thread; chop = decode-error rate >= PTV_ACHOP_ERRS_MIN/min (60) OR
self-shed rate >= PTV_ACHOP_SHEDS_MIN/min (120), sustained >=
PTV_ACHOP_SUST_MIN minutes (3) -> FULL audio-path rebuild (decoder swap via
the new adec_swap — factored from the ADECWD reopen —, graph+swr teardown,
0.9.17.1 AFMT impossible-seed so the path re-forms from 5-frame-confirmed
clean params; anchor/pts preserved; counts as an afmt-rebuild corrector
event). One WARNING line per attempt; rate-limited PTV_ACHOP_RELIMIT_S (600)
per track — a permanently-broken source cycles quietly instead of chopping.
PTV_NO_ACHOP_REBUILD=1 kills (byte-inert; detection not even sampled).
GATES (deadaudio recipe, every-3rd audio pkt XOR 0x55, looped tsp/UDP):
pre18 = indefinite chop, 208 [PTV-ADEC] errors/180s, zero escapes; pre19
(test-scaled SUST_MIN=1, RELIMIT_S=90) = escapes at t=50s and t=140s (exactly
the rate limit), and on a mid-run switch to the clean feed the errors stop
instantly, the impossible-seed [PTV-AFMT] rebuild fires (-1Hz (null) 0ch ->
48kHz fltp 7.1) and output audio resumes (RMS -23dB, zero >=0.5s holes).
(#38) TOLERANT AAC DECODE for the Azorse broken-7.1 class — lavc half in the
NEW v2 patch 0004-aac-tolerant (branch aac-tolerant off master; libav*
features are SEPARATE patches per the pre19 owner directive), app half here.
SYMPTOM: broken phases emit 7.1-signalled AAC-LC with a CPE-first element
sequence; strict lavc rejects every frame ("channel element 1.0 is not
allocated" — 467/467 pkts on the 2026-07-15 capture) while VLC/FAAD play it.
LAVC (0004): decoder private option tolerant_ch_alloc (default OFF =
byte-identical): an element the positional mapper cannot place is mapped by
its own type/id (existing allocation for that tag, else allocated on demand,
bounds-checked SCE/CPE/CCE/LFE + MAX_ELEM_ID). HARDENED per owner directive
(the same error fires on plain corruption/parser desync, which must never be
decoded as audio): PERSISTENCE-BEFORE-TRUST — the SAME unexpected element
pattern (ordered (type,id) additions) must repeat over 3 consecutive
fully-parsed frames before any tolerated frame is released; qualifying frames
decode (windowing state stays continuous) but are discarded as errors exactly
like the strict path; any pattern change restarts qualification; tolerance
covers ALLOCATION only (payload parse errors keep failing frames through the
existing paths); output planes of absent elements are silenced (no
uninitialized-buffer leak into a downmix). GA path only; ER/USAC untouched.
APP (this patch): ptv_adec_opts() sets the option BY NAME via av_opt_set on
the decoder priv context at both audio open sites (initial + adec_swap) — no
compile-time dependency; against a stock lavc the option is simply absent and
decode stays strict (verified: strict flood, dead track, clean exit).
GATES: corruption fixture (per-frame VARYING bogus element ids, 7046
target-site rejections) = tolerant releases NOTHING, identical to strict;
one-off bogus element (LFE[9], exact "not allocated" site) = rejected
identically both modes (md5-equal outputs); positive fixture
azorse-broken71-ts.ts (GENUINE Azorse audio re-encoded stereo, ADTS
channel_configuration precisely rewritten 2->7 via PES-aware ES walk —
reproduces BOTH live log signatures) = released from frame 3, front-pair PCM
SAMPLE-EXACT (md5) vs the true stereo decode from the first released frame;
clean streams md5-identical option ON vs OFF and patched vs unpatched
(option OFF); ptvencoder end-to-end on the broken TS (combined build):
init-fail -> "pattern persisted 3 frames" -> [PTV-AFMT] -1Hz->48k/7.1 rebuild
-> output audio RMS -33.7dB, zero silent holes, clean exit. The 2026-07-15
VLC reference WAV did not survive the scratchpad wipe; the sample-exact
front-pair identity is the (stronger) replacement gate. NOTE: broken phases
only produce audio on builds carrying patch 0004; without it behavior is
pre-#38 (silent track, AFMT self-heal on source recovery).
LOCKED SET (final stock-lavc pre19 binary vs pre18-ref, all new envs at
defaults): mir2 = full CONTENT-EXACT escmp MATCH (p18 a1 == p19 b2 — the
pre18 gate's own verdict class); tb30 = no 3x3 escmp match (pre18 A/A
precedent 0/10) but component-exact cross pairs (video ES+pts EXACT in one
pair, audio ES+pts EXACT in two) and evseq differences 10-13 lines vs the
SAME-BINARY A/A floor of 11 (all startup scatter: ANCHOR drop counts, LAYERA
flush 63vs64 pkts, EMPTY wall digits); g2 = audio ES+pts EXACT in EVERY pair
incl. A/A, video differs identically in A/A, evseq cross 12 == A/A 12; clean
TruBLU (trublu-20260313-fresh, 30min file cell) audio ES BYTE-IDENTICAL
across p18/p18-rerun/p19 (89c9d0bc...) with video md5 differing even
p18-vs-p18 (x264 file-cell nondeterminism — the control that isolates it);
fx51 corrector cell (TESTHS 40:15 + capped 300ms TESTWALK, dwell 60s/quiet
30s) line-identical: armed -> ENGAGE R=+300ms -> PARK +300->-17ms corr=+313ms
in 232s, 0 hs-step holds BOTH binaries (pre18 published +300->-16/+312/230s);
kill cell PTV_NO_ACHOP_REBUILD=1 on the corrupt feed 150s = ACHOP 0 with 186
ADEC errors flowing; zero [PTV-ACHOP]/divergence lines in every clean cell.
fx50's crafted two-PID jump+gap sender did not survive as an artifact — its
behavior class is covered by mir2's full MATCH (the #50/#33 flush machinery
path) + the pre18-identical event sequences above; noted for the reviewer.

(7q) LIVE-DEFECT BATCH — pre18: five items, one commit each, kill-switch each.
(A) GAP-VERDICT vs LAYERA ONE-REMEDY INVARIANT (task #50; the AWE_Plus +2.38s live
defect 2026-07-19 14:44). SYMPTOM: lipsync stepped to +2385ms flat at a source
audio cut. ROOT CAUSE (live log sequence): stream-2's +2.421s jump armed a LAYERA
cycle at t; stream-1's +2413ms audio GAP verdict landed at t+42ms (aresample pads,
verdict propagated); the armed cycle flushed at t+520ms with applied_offset
aud=−2.389/vid=0.000 — ONE source event (an upstream cut: flowing labels on one
PID, wall absence on the other, the AWE shape) got BOTH remedies: the verdict's
pad AND the flush's relabel-erase. FIX (g_glueveto, three parts):
 1. VETO: a gap verdict landing while a matching cycle is ARMED-but-unflushed
 (magnitude within 150ms + armed within 500ms + the jump's from/to domain
 brackets the gap's labels within 2s — a wrapped/corrupt-dts jump in a different
 domain never matches, the Fashion class) DISBANDS the cycle: ptv_disc_cancel()
 releases every buffered packet unrebased in DTS order (last_sent_dts advances
 like the normal path), the jump stays in the labels on every stream, and each
 stream's own content machinery pads it (the gap is the evidence the content is
 genuinely missing — the pad is the right remedy everywhere). The gap stream's
 own continuity ref advances too, so neither stream re-arms a cycle.
 2. INVERSE guard: a verdict landing within 500ms AFTER a matching flush is
 SUPPRESSED (PtvDiscBuf.fl_wall/fl_delta_us stamped at every audio-moving flush)
 — the step falls to the discontinuity layer, which erases it consistently with
 the already-flushed sibling. One event, one remedy, in either ordering.
 3. E5 NET (second net for orderings the veto keys miss): LAYERA-flush label
 shifts are published per track (g_flush_relab_step/wc); a shift matching an
 OPEN pad-ledger entry is that pad's RETURN leg erased at the packet layer
 (invisible to AGLUE — the labels arrive already-shifted) → counter-applied at
 the graph door (glue_off_us −= pad; async hard-drops the inserted silence;
 §2.4 tripwire armed on the drop; corrector glue_events freeze).
KNOWN BOUND: a THIRD dense audio stream whose matching jump arrives flowing
after the veto window gets its own cycle/erase (3-audio channels; no live
precedent — noted for the record). PTV_NO_GLUEVETO=1 reverts all three.
(B) CORRECTOR HS-TICK EVENT FILTER (task #51a; AWE dwell starvation, live
2026-07-19). SYMPTOM: with a +2380ms flat bake standing, the corrector never
completed a dwell — on pulldown/decim channels house_skew ticks ±1 video tick
(~33ms) every 10-17s forever (bursty: tick trains for ~1-2min, then 1-5min
quiet), each crossing of the cumulative-50ms edge reset the dwell and fed the
storm counter → re-arm/storm-disarm limit cycle; the live shape was "nibble
+60ms per lucky quiet window, 10min holdoff between" (15:07 window: ENGAGE,
corr +59ms in 30s, storm-disarm at 15:12:59). R measured flat +2380..+2384
through every tick = the ticks are benign cadence noise, not lineage events.
FIX: magnitude-filter the hs event EDGE — a step ≤ 1 tick + ¼ tick (~41ms at
29.97, 50ms at 25fps; always < the 2-tick event bar) is absorbed into the
snapshot silently (no dwell reset, no storm count); larger steps stay events.
(Not purely a relaxation: at 50fps the 1.25-tick bar = 25ms, so a 2-tick 40ms
step is now an event where the old cumulative rule needed 50ms — intent-
consistent, ≥2 ticks is always an event.)
All other named events (LAYERA, verdicts, AGLUE, reopen, AFMT, glue, ledgers,
bank) keep full event status; the §4.3 continuous R-stability re-anchor
remains the actuation safety, untouched. PTV_NO_HSTICK_FILTER=1 reverts.
(C) ANTI-STARVATION CEILING (task #51b; the legacy-0007 PLL_HARD_CEILING 60min
+ PLL_STUCK |baseline|>2s & drift<50ms pattern, sized to the certified
sensor). Recovery belt under (B): whatever event class churns, a channel whose
R has stayed LARGE (>engage band) and FLAT (the §4.3 criterion against the
span's own reference — live AWE: R +2374..+2385 over 28min qualifies) for
≥15min total while the dwell never completed (resets and storm holdoffs
INCLUDED — the span runs through ARMED/DWELL/DISARMED; the 10min storm holdoff
deliberately does not block it) ENGAGES anyway with one WARNING line. The
flatness requirement is load-bearing: any R move beyond max(40ms, R/4)
restarts the span, so a genuinely churning R can never ceiling-engage.
Sensor-invalid, delivery-dead and implausible R close the span (they still
block). Event feeds are re-snapshotted at the ceiling engage (a stale-snapshot
entry from a long DISARM would read accumulated deltas as instant events).
Authority/park/disarm semantics of the ENGAGED state are unchanged — a
ceiling engage steers under the same 5s/10s caps and freezes on the same
events. PTV_NO_RSCORR_CEIL=1 reverts; PTV_RSCORR_CEIL_MIN (minutes) tunes.
(D) PER-STREAM DELIVERY WATERMARKS (pre17 (B) KNOWN OPEN, owner-approved;
owner mandate: af-independent transport, single input AND mv, auto-sized).
SYMPTOM: on a MIXED rung (loudnorm'd transcoded AAC ~3s late + copied AC-3
~0s) the §7.5b hold and its auto-size were keyed to the shared a_dlv_dts_hi
= the LEAST-delayed stream, so the slower track rode the wire late by its
whole chain latency (measured +843ms single-loudnorm, +6.6s triple; live
Cinestar was clean only because video pipeline ≈ loudnorm latency —
coincidence, not design). FIX: DlvGate tracks a per-output-stream delivered
watermark (registered at first delivery, staleness-stamped per delivery);
the video hold's release key (dlv_a_hi_key) is the MINIMUM across streams
delivering within the last 2s — video waits for the SLOWEST LIVE audio
stream, on single input and mv alike (mv slots share one gate per rung: the
key becomes the slowest live slot). A stream silent >2s is EXCLUDED, so a
dead AC-3 track (or dead mv slot) can never hold video beyond that window —
the §7.5b audio-death escape/disarm stays keyed on the AGGREGATE
a_hi_change_wc (fires only when ALL tracks are silent, unchanged). The
pre17 (B2) cap auto-size now measures against the same slowest-live key, so
the escape cap sizes to the slowest chain. SINGLE-audio channels read a
min over one stream == the old aggregate (identical code path; review-verified
via mir2 content-exact). Multi-track UNIFORM channels change key max→min —
bounded by the interleave phase (~one audio frame), benign but not byte-equal.
>PTV_DLV_MAX_AS (16) gated audio streams: extras stay unkeyed = pre17 posture
(fail-open). Gate (cinestar mixed rung, single
loudnorm, local x264 cell): pre17 wire = AAC median +1080ms late / AC-3 +16ms;
pre18 = AAC +240ms (in band) / AC-3 −760ms — the copy now LEADS by the
latency spread, which is structural: §7.5a releases audio on the ENCODE front
(a delivery-keyed audio hold would close the forbidden deadlock cycle), so
the fastest stream leads once video waits for the slowest. Early copy is
bufferable/benign; the defect was the late transcoded stream. Audio-death
gate: AC-3 PID killed mid-run → no video stall (fps flat 25.x, zero
watchdog), stale exclusion took it out of the key within 2s.
(E) MV PLL ACQUIRE LIMIT CYCLE (task #49, pre-existing). SYMPTOM: on
audio-erase-class corruption an mv slot re-anchors ±277ms every ~12s forever
(audible warble until restart). ROOT CAUSE (code-confirmed): an erase-class
phase presents the PLL a FLAT ±step offset that FLIPS at each erase — flat
defeats the v0.6.22 noise-adaptive threshold (pll_dev is an EMA of |off−ema|
jitter; a flat step reads dev≈0) AND the 1.0.1 3-consecutive-window sustain
(the step is genuinely stable >2s), so every flip is a textbook "stable large
offset" and acquires at exactly the 12s refractory rate; each acquire's
drop/pad is itself the next flip's perturbation. FIX: repeated-ACQUIRE
BACKOFF — each ACQUIRE within a 60s window doubles the acquire threshold
(level +1 per acquire, −1 per acquire-free 60s, cap ×32 under the existing
1.5s absolute cap); after 2-3 storm acquires the bar outgrows the corruption
step and the storm converges (TRACK, the refractory and the 1.5-tick floor
untouched; a legitimate isolated acquire pays one doubling that decays within
a minute). ACQUIRE line gains backoff=N. Mechanism fixture-gated (synthetic
±300ms square-wave storm via PTV_PLL_TESTNOISE_MS on an mv pair) + clean-mv
no-regression; LIVE ACCEPTANCE = the next Azorse-class erase event (a
faithful local repro of the corruption class is impractical — declared per
the brief). PTV_NO_ACQ_BACKOFF=1 reverts.
(gates, 2026-07-19/20, all local x264/tsp cells) —
(A) fx50-gapjump (flowing +2.421s jump arming LAYERA + wall-absent +2.421s
gap 200ms later, the live ordering): pre17 = one-sided flush aud=−2.400 →
lipsync a0:+2400ms flat (live signature reproduced); pre18 = cycle DISBANDED
(armed 205ms ago), both steps padded, lipsync a0:+0ms a1:+0ms;
PTV_NO_GLUEVETO=1 reproduces pre17 line-for-line (kill parity).
(B) fx51-clean + TESTHS 1-tick walk + capped 300ms TESTWALK (test-scaled
dwell 60s/quiet 30s): filter ON = zero hs-step holds, ENGAGE → PARK
R +300→−16ms (corr +312ms in 230s) THROUGH the churn; kill-control = 3
hs-holds → event-storm DISARM, corr +0ms, no engage in 480s (the live
starvation shape); 2-tick (80ms) steps still reset the dwell (4 holds).
(C) ceiling (CEIL_MIN=2 test-scaled) under unfiltered churn: fired at ~2min
three times across the run (storm-disarms between; corr nibbled
0→+60→+181ms, one WARNING each — under the production-default (B) filter
this cycling does not arise, (B) parks instead); churning-R control
(uncapped 50ms/s walk) NEVER ceiling-engaged (ended implausible-R disarm).
(E) mv 2-up, ±300ms flat-step storm flipping every ~30s (TESTNOISE_P=1400):
19 acquires (PTV_NO_ACQ_BACKOFF control) vs 12 (backoff, oscillating levels
1-2) over 240s×2 tracks; clean mv pair 0 acquires on BOTH builds.
LOCKED SET: fashion dj4/5/6 verdicts identical to the rr17 baselines
(wall-jitter digits only); fx33 mir2 = full CONTENT-EXACT escmp MATCH
(pre17 a1 == pre18 b3); tb30/g2/trublu-300s = content-exact (event
sequences line-identical, per-half ES exact in cross pairs) — pre17 A/A
itself is start-phase byte-nondeterministic on these replay fixtures
(0/10 same-binary escmp matches on tb30), so byte-equality is judged only
where A/A supports it, per the rr16/rr17 review precedent.
TEST HOOKS added this pre (never set in production): PTV_RSCORR_TESTHS
(triangle-staircase hs churn read only by the corrector's event detector)
and PTV_PLL_TESTNOISE_P (TESTNOISE square-wave half-period, frames).

(7p) MV COMPLETION PRE — pre17: sibling-slate sensor artifact fix + af-independent
mv transport (task #48) + MV CORRECTOR ARM. Three work items; the corrector-arm
SHIP decision is soak-gated (grid soak verdict) — implemented and fixture-gated
here, merge is the owner's call.
(A) "SIBLING-SLATE SENSOR ARTIFACT" (grid soak finding 1, 2026-07-19,
arm-blocker) — ROOT-CAUSED FROM THE BOX LOGS (Grid_2x2 00:25-01:10, PTV_DIAG
was on) as something else entirely: the readings were TRUE and the slate was
innocent. Measured sequence: (1) mid-slate readings were FLAT ±6ms (00:37-
00:42 — a live 4-up with slot2 slated reads true); (2) sync_check then put the
grid in a RESTART LOOP (00:42:31, 00:48:51, 01:00:58, 01:07:18 — ~6min apart);
(3) EVERY restart-with-a-dead-slot birth banks the LIVE slots' backlog (the
compositor preroll wait runs its full 3s timeout; the one-shot v0.9.13 trim
fires, but the video_q compressed backlog re-decodes into hold.q right after —
box: occ 25→110 within 10s of tick 0), the occupancy servo then drains the
excess at its ±2% authority = a ~20ms/s content slide for ~3min (md decimation
0.5/s, REANCHOR2 birth storm h0 +~1s in ticks 19-33), and the per-slot audio
follow CANNOT track a 20ms/s slide (PLL acquires throttled 12s-refractory/
~400ms steps + TRACK ≤10ms/s — box: acq #1..#12, applied −853→−2773ms over
3min) → every live slot ran a REAL 150-240ms audio-late transient per restart
([PTV-AVSYNC] offset −160ms concurs; the 00:46 oracle +69ms sampled AFTER
convergence). The lipsync= token faithfully measured it; the restart loop made
it look like a stable slate-correlated drift.
FIX (three parts, at the cause + at the instrument):
 1. mv BIRTH-TRIM WINDOW — for the first ~20s the startup trim is a window,
 not a one-shot: excess over the primed target is dropped-oldest SILENTLY
 (never servo-slid at 20ms/s for minutes; the backlog is disposed of within
 the join seconds). mv-only (single-input deep-prime/AUTO-BANK retention
 untouched); PTV_NO_MV_BIRTHTRIM=1 reverts.
 2. m_v EMA RE-SEED on REANCHOR2 and on slate-recovery REANCHOR — an h0
 re-anchor REDEFINES the slot's mapping baseline; blending pre-shift EMA
 samples in is a stale reading (birth fixture: −185ms EMA-diluted ghost),
 and re-seeding makes a GENUINE post-shift displacement (#24's shape) show
 immediately, not masked.
 3. m_a EMA RE-SEED on a PLL ACQUIRE (the audio-side mirror: af_applied jump
 + content drop/pad = a one-step mapping redefinition; without it R decayed
 a stale −2.2s for ~2-3min after the acquire had already fixed reality).
 The acquire remains a corrector dwell-reset event.
Fixture (late-joining-slot 2x2 = the local restart-with-dead-slot): pre16.1
reproduces the box signature exactly (REANCHOR2 storm, acquire cascade, R peak
−139ms decaying ~6min, player-visible offset −100..−166ms STILL desynced at
+6min); pre17: backlog disposed in the 20s window, R honest and |R|<50ms
within ~60s on the resident slots, player-visible offset ±5..30ms by ~2min.
Mid-run slate cells (STOP 600s) read flat ±8ms on BOTH builds — per-slot
independence was never broken; the soak's mid-slate readings were true.
BELT (corrector, part of (C)): birth slate-mask — every slot starts masked
until its first display (a NEVER-arrived slot previously never set the mask:
`stale` requires last_fresh_us>0), so the finding-1 shape can never engage the
corrector even unfixed. Mid-run slate cells (STOP 600s) read flat ±8ms on BOTH
pre16.1 and pre17 — per-slot independence was never broken.
(B) AF-INDEPENDENT MV TRANSPORT (task #48; owner mandate 2026-07-19: the wire's
A/V interleave alignment must be invariant to any -af latency, on single AND
multi input, auto-sized, never per-channel config): the §7.5b early-video hold
is ARMED ON MULTIVIEW (fold of the t48-mv-vdlv fix branch). Root cause of the
grids' dlvhold=0-since-07-13: the fleet loudnorm rollout (~3s one-pass fill)
made every mv audio track wall-LATE — the §7.5a gate holds EARLY audio only, so
it read dlvhold=0 (starved, working as designed) and the wire carried video
~2-3s ahead of audio with labels intact; mv had no closing mechanism. Slots
share ONE gate per rung: a_dlv_dts_hi keys the hold to the LEAST-delayed slot's
audio (fleet-uniform loudness chain ⇒ small per-slot spread bounds the residual
skew; a single dead slot cannot wedge video — any flowing track advances the
high-water, the audio-death escape needs ALL tracks silent). v_birth_flow lets
mv video flow at birth (mv audio anchors at first DISPLAY, so the §7.5a
count==0 birth rule misfires there). mv stats line gains vdlvhold=/vdlvforced=.
ACCEPTED METRIC FLIP (owner-decided): a buffering-af mv channel's buffering
moves from the audio gate to the video hold — dlvhold reads 0 and vdlvhold
~1-3s; wire, latency and RAM equivalent, but fleet dlvhold health dashboards
change meaning on mv.
(B2) VDLV CAP AUTO-SIZE (owner addition): PTV_VDELIVERY_CAP_MS (6s) was a
per-channel tuning knob for >4s audio chains — mandate violation. The
escape/age cap now AUTO-SIZES to 1.5× the measured audio-chain lateness
(EMA of v_enc_dts_hi − a_dlv_dts_hi per video drain, sampled only while audio
is actively delivering so an outage can never inflate the escape timer),
floor = the env/6s default, ceiling 12s (PTV_VDLV_CEIL_US, the CUSHION_MAX
philosophy). Env kept as override-floor only. [PTV-VDLV] logs resizes.
Gate (uniform-audio, triple-loudnorm ~6.9s lateness, single input): pre16.1
fixed cap = vdlvhold pinned 6.0s + vdlvforced storming + wire +635ms; pre17 =
one birth escape/re-arm, cap auto-sized 6.0→10.3s, wire −40ms.
KNOWN OPEN (surfaced by the mandate's own gate, PRE-EXISTING — pre16.1 reads
identically): a rung with MIXED per-track lateness (loudnorm'd transcoded AAC
+ un-normalized copied AC-3) is STRUCTURALLY invisible to the §7.5b hold and
to this auto-size — the shared per-rung a_dlv_dts_hi high-water is the
LEAST-delayed stream (the AC-3 copy, +8ms aligned), so the AAC track rides
the wire late by its net lateness (measured: +843ms single-loudnorm, +6.6s
triple; identical on pre16.1). Closing it needs per-STREAM delivered
high-waters keyed on the MINIMUM with staleness exclusion (else one dead
track wedges video — the exact trade the mv arm avoids) — an owner design
decision, deliberately NOT half-shipped here.
(C) MV CORRECTOR ARM: the pre16 `if (a->multiview) return;` hold in
rscorr_update is removed. Arming prerequisites, all landed here:
 - per-INPUT delivery liveness (§3): rscorr_delivery_dead reads the steered
   track's OWN input's arrival watermark (new Input.v_arrive_wc, stamped by
   that input's demux; g_v_arrive_wc stays as the any-input aggregate for the
   single-input starvation detectors). Rung-wire watermarks stay per-RUNG
   deliberately — one dead rung anywhere holds the whole channel (§3 gate
   rule), that is not an any-input smear.
 - SIBLING-SLATE FREEZE (finding-1 defense-in-depth over the (A) sensor fix):
   compositor maintains g_mv_slate_mask (bit per slot, set at slate onset /
   cleared at recovery); rscorr_event_active returns "sibling slate" while any
   bit is set — no mv track may engage during the exact condition (A) showed
   can contaminate readings; recovery re-runs the 3min quiet window.
 - stale-track watchdog re-homed: the pre14 body moved to
   rscorr_stale_watchdog() (ptvencoder_audio.c), called by BOTH cadence owners
   — single-input master rung (unchanged strings) and the mv compositor (1s),
   where it never ran (passthrough rung loops return early; port-doc §6's
   named prerequisite).
 - mv stats line gains corr= (shared pre14 builder, aK: prefix forced; absent
   while quiet); per-slot event edges already resolve per-slot since pre16
   (epoch/shed/gov/LAYERA/pair-expect by dbg_in; REANCHOR2 bumps its slot's
   epoch — now live-consumed).
(rider) per-input always-on `[PTV-RSYNC] inK R= ev= sk= occ= [SLATED]` summary,
one compact line per input on mv (owner-floated; the soak forensics that used
to need PTV_DIAG). Fix round: throttled to one round per 30s (reviewer V6 —
~11.5k lines/day on a 4-up instead of ~35k).

(7p FIX ROUND, rr17 review 2026-07-19 — same banner, second commit):
R1 MV INPUT EOF PERMANENCE (confirmed defect, owner-rejected behavior): any
av_read_frame error (rw_timeout expiry = the >=30min-outage class) exited the
demux thread permanently — slot slated forever, g_mv_slate_mask bit latched,
corrector held mosaic-wide forever. Fix: live mv net inputs REOPEN-RETRY
forever (bounded 1..5s backoff): close the dead ctx (its udp socket must
release the port — a leaked socket's circular-buffer reader would steal the
datagrams from the re-bind), reopen with a COPY of the original format opts,
validate the stream layout (count + consumed-stream codec types) before
swapping; resume rides the proven <rw_timeout recovery machinery (slate →
recovery edge, re-seed, fresh dwell — V10). [PTV-REOPEN] lines. AudioState's
AVStream* replaced by owned codecpar/timebase copies (the ADECWD reopen must
not dereference a closed ctx). Single-input and file inputs keep EOF = end.
SEMANTIC NOTE: a live mv NET input now NEVER exits on read error — an mv
channel ends only by signal (the production posture: supervisord SIGTERM) or
by file/single-input EOF. Fixture cells that relied on sender-EOF teardown of
mv runs must timer-kill (the gates harness was updated accordingly).
R3 FASHION CLASS closed (three pieces):
 (b) THE LOAD-BEARING FIX — corrupt-packet LAYERA poisoning: the corrupt
 discard lived only in demux_dispatch, DOWNSTREAM of the LAYERA machinery, so
 a corrupt packet's garbage dts fed jump detection, FALSE base recording,
 sib_jump stamps and continuity refs before being thrown away. Fashion live
 (20:05:45.733): a corrupt audio packet with a wrapped dts (7869470123 ≈
 87438s → post-unwrap −8005.7s = exactly the video's post-jump domain)
 classified NEW against the video's fresh bases, transitioned the un-crossed
 audio, and the 72ms flush rebased it +11594s — the one-sided cycle B, the
 permanent avoff and the async grind all descend from that poisoned
 classification. Fix: corrupt-flagged packets are discarded (counted, via the
 existing dispatch path) BEFORE demux_unwrap/LAYERA. Replay fixture
 fashion_dj4 (opposite ±11594.4s legs + TEI'd wrapped-dts poison packets)
 reproduces the live signature on the unfixed build (poisoned cycle-A
 "shared" flush aud=+11594.4 on un-crossed audio, avoff parked +72254s) and
 reads clean post-fix.
 (a) BOUNDED HOLD-UNTIL-DRAIN (owner rule): the pre16 one-shot 500ms partial
 extension repeats while the matching sibling jump is known and its leg
 absent, capped by the 5s pairing window from cycle open + a capacity guard;
 PTV_DISC_CAPACITY 256→512 so the window is reachable (256 would ENOSPC-flush
 at ~3.3s at Fashion's ~77pkt/s). Gate fashion_dj6 (return legs 1.3s apart):
 pre-fix releases one-sided at the single extension; fixed build collects the
 sibling leg into the SAME cycle → SHARED flush, avoff→~0.
 (c) AGLUE PLAUSIBILITY CEILING (review-recommended): a label step the
 tripwire REFUSED (parked slip > its 2s authority) is no longer pursued —
 the parked remainder is relabel-erased into glue_off (butt-joint; sensor inj
 accounting automatic), one ERROR line; the channel stays watchable instead
 of grinding aresample at max rate forever (Fashion's −73Mppm for hours).
 PTV_NO_AGLUE_CEIL=1 reverts.
F2 PTV_RSCORR_TESTWALK_CAP_MS (TEST-ONLY): the walk saturates at the cap, so
the corrector can steer the synthetic bake to 0 and PARK — the mv
ENGAGE→steer→PARK happy path becomes demonstrable (TESTWALK alone is
cancelled by the steer at equal rate and never re-enters the park band).
ADVISORY closes: [PTV-RSYNC] inK throttled (above); cor-2 dlvhold=1361ms
datum RECONCILED — cor-2's deployed /opt/scripts/ptvencoder.sh contains no
loudnorm (rollout never reached it): cor-2 mosaics run the legacy chain
(audio early → dlvhold ~1.4s), live-transcoder grids run loudnorm (audio
wall-late → dlvhold 0). Both consistent with (B)'s mechanism; finding-2's
"pre4/5/6 regression" theory stays falsified.

(7o) GLUE CLASSIFICATION LIVE-INCIDENT FIXES — pre16, task #47. Two deployed-
pre15 incidents (The_Word_Network/cor-3, Fashion/live-transcoder 2026-07-18)
exposed three defects in (7m)'s classifier; all three fixed, incident shapes
fixture-reproduced and flipped, glueclass + pre16 gates re-run.
DEFECT A (TWN, ordinary/high-incidence): the rr15 §2.5 propagation guard
tested video's packet AGE at verdict time (≤ min(2s, wall_gap/2)) — after a
whole-program 10s outage where video resumed 2ms BEFORE audio, video's own
resume packet satisfied it → GAP mis-verdict propagated → audio LAYERA leg
killed → video's +10s jump flushed PARTIAL one-sided → R=−10s baked flat
(corrector correctly disarmed implausible). Fixed: propagation now requires
video to have PROGRESSED THROUGH the gap — ≥ one video packet per 80ms of
gap (min 3) arrived since the audio stream's last packet (new per-stream
gap_vsnap of the demux vpkt counter), plus the 2s liveness belt. A
whole-program outage yields ~0-5 resume packets and falls through to the
shared LAYERA flush (the pre14 path, which handles it perfectly), in EITHER
resume order.
DEFECT B (Fashion, exotic/catastrophic): the §2.4 tripwire had NO authority
clamp — an insane routed verdict (+11594s, see C) parked slip at +8798s and
the tripwire synthesized ~2800s at the swr boundary, pinning the channel at
async for hours. Fixed: synthesis authority = PTV_GLUE_TW_CAP_US (2s — every
legitimate hard-comp verdict realizes instantly; the largest real one ever
routed, PATRIOT 30.8s, realized); beyond it: NO synthesis, one ERROR line,
corrector freeze, verdict retired.
DEFECT C (common root): the PARTIAL flush release applied a one-sided
re-base with no classification — manifestation (a)'s partial=1 path, never
reached by (7m)'s rules. Two-part fix, both evidence-based and fail-safe to
pre14:
 C1 ROUTE CAP: a flush mismatch beyond PTV_GLUE_MAX_ROUTE_US (120s) is
 REFUSED (per-stream butt-joint + loud line) regardless of label health —
 Fashion's opposite ±11594s jumps computed a +23188s "mismatch"; magnitude
 IS decisive out there (no source A/V misalignment reality exceeds minutes;
 4x headroom over PATRIOT). Applied at the 2b/3a routing pre-scan and the
 2d retro leg.
 C2 PARTIAL HOLD (brief option (i), bounded): when a flush would release
 with only one media type crossed, offset > EPS, and the MISSING sibling
 type shows a KNOWN jump of matching magnitude within the pairing window
 (new per-input LAYERA-detect + gap-verdict stamps) that has NOT yet
 participated in the event window, hold the cycle ONE extra 500ms so the
 sibling leg can cross into the same cycle and the flush runs SHARED. If it
 still doesn't cross: release as pre14 + loud WARNING. Option (ii)
 (retro-apply to the sibling) was REJECTED as unsafe — the sibling's jumped
 packets may already have dispatched, so a retroactive wrap_off shift
 injects a second opposite step at the graph door; option (iii) reduces to
 today's release for a leg that never crossed. The hold is inert on every
 pinned partial fixture (gate-1/tb30: the sibling already participated —
 pair_vid_defined / pair_has short-circuits).
Gates: FX-TWN (10s whole-program outage, video-first resume by ~25ms) now
shared-flushes identically to pre14, no propagation, no partial one-sided
re-base; FX-FASHION (opposite ±11594s double jump, video to negative dts)
refuses the route (cap), no tripwire synthesis, both legs butt-jointed,
oracle tail ≈0; G1/G2 genuine gaps still propagate; glueclass G9 locked set
+ G10 byte gates re-run green vs pre16.

(pre16.1) mv stats readability, owner-directed 2026-07-18: the per-slot sensor
reading moved INSIDE each inK: group (`inK:.../lipsync=+10ms`) instead of the
separate combined `lipsync=a0:...` token — the mv line already keys per input,
so the reading belongs with the slot's other fields. New shared builder
ptv_stats_lipsync_in() (same R arithmetic; multi-track slots joined '|';
absent when the input has no sensed track). Single-input line untouched.
Log-format-only change; no behavior delta.

(7n) MULTIVIEW SENSOR PORT — pre16, task #45 items 1+2. NORMATIVE DESIGN =
analysis/ptvencoder-mv-sensor-port.md (owner-approved 2026-07-18 with all
five §11 questions resolved to the doc's recommendations). Every
(input-slot, audio-track) on a 2/4-up mosaic gets the SAME certified sensor
single-input has — mv desync becomes measurable (#24 RAV/RSBN audio-early
birth and #27 audio-only-outage resume misanchor become instrument-visible;
this pre MEASURES them, fixes are later rounds). The one structural insight:
the sensor's audio half was already per-(slot,track); only the video half was
scalar and every mv block was an explicit `!multiview` gate. The port:
(a) RsyncSense re-shape: mv_ema/mv_wall/ev_us become per-slot arrays
(PTV_MAX_INPUT), + n_in + a_in[] track→slot map; single-writer-per-field
relaxed atomics unchanged (compositor owns mv_*[k] on mv, master rung = slot
0 on single input; slot's demux owns ev_us[s]). g_rsx.n_a = n_audio ALWAYS
(was 0 on mv).
(b) Compositor per-slot video publish at the sk-measurement/DISPLAY site:
m_v[slot] = EMA[mv_tick_us(tick) − disp_src] per house tick, dup/residence
holds included (single-input dups-included rule), exact-rational axis, τ≈30s
divisor verbatim; published only while `last[k] && !stale` → a slated slot's
tracks read `--` (freshness = the outage signal). REANCHOR2 excursions are
MEASURED into m_v then cancelled through the h0-shifted audio side —
residual mismatch is exactly what R must show (#24's suspected shape).
(c) mv gates deleted: the audio sensor block + pre11 slip probe + pre15
realization tripwire run on mv (one `!a->multiview` removed); `inj` now
mirrors the PATH-DEPENDENT graph-door bus (single/non-follow: house_skew;
mv follow: af_steer_us; glue+corr both paths) so R reads the residual and
dR/dcorr=−1 holds per track on both paths. The tripwire is the sensor
block's only non-passive resident on mv (deliberate — closes the row-22
silent hole); PTV_RSYNC_SENSE=0 remains the kill for sensor+tripwire, both
modes.
(d) LATENT INDEX BUG FIX: demux ea_us publish keyed by DEMUX-LOCAL j
(identity on single input; on mv input 1's first track posted into ea_us[0])
→ g_rsx.ea_us[d->aglobal[j]]; every input now publishes (rsync_pub →
rsync_slot).
(e) CORRECTOR HELD OFF ON MV (mandatory, owner Q1): one-line
`if (a->multiview) return;` at the top of rscorr_update() — before the port
mv was inert only by accident (rs_ma_seed never set); without the hold the
fleet-default-ON corrector would actuate on grids with no soak. CorrState
stays OFF, corr_us stays 0 ⇒ bus term + inj term never fire ⇒ mv
byte-inertness. Removal site = the mv corrector-arm pre, which must also
re-home the clock.c stale-track watchdog to the compositor (unreachable on
mv — passthrough rungs return early) and wire per-input liveness
(g_v_arrive_wc + rung watermarks).
(f) Stats/DIAG parity: lipsync=/corr= builders extracted to shared helpers
(ptvencoder_legend.c; single-input line TOKEN-IDENTICAL — the video term
indexes a_in[ki]≡0 there). mv stats line gains ALWAYS-ON per-slot
`lipsync=a0:+3ms,a1:--,...` (aK: prefix forced; owner Q2) + `acor=` global
sum (per-track detail stays on [PTV-ADISC]/NBS lines; owner Q5); corr=
deliberately absent while the hold stands. Governor telemetry re-homed
per-input (Input.gov_gpps/decl/on; globals stay as the input-0 alias —
single-input DIAG t= string unchanged, owner Q4); mv [PTV-DIAG] per-slot
segment gains /gpps=M/D/gov=G (the governor ran blind on mv, rr13).
Startup `[PTV-RSYNC] tracks: a0→in0 …` map line (mv only).
(g) Per-slot event feeds wired for the ARMING pre (consumed then, exercised
by the sensor soak now): REANCHOR2 fires bump the slot's house_disturb epoch
(REUSED, not a new atomic — owner Q3: house_disturb is consumed only by
corrector snapshots, PLL acquire event-ungated since v0.6.18, so the bump is
inert while the hold stands); g_shed_wall/g_shed_cnt re-homed per-input
(Input.shed_wall/shed_cnt; globals stay stamped as the any-input aggregate
for the catch-up governor) — AGLUE self-shed notes + the corrector's
quiet-window/governor feeds read the track's OWN input, so slot B's
shed/governor no longer annotates or freezes slot A. Corrector snapshots/
event-edge read ev_us[dbg_in]. Remaining any-input smears (g_v_arrive_wc,
rung wire watermarks) are explicitly the arming pre's §3 scope.
Known floor: a 25-in-29.97 slot's residence sawtooth EMA-averages to a small
stable offset (≤ ~half tick) — documented noise, far under the 80ms engage
band; soak-verification item, not a defect. Gates run: MG-B1 single-input
byte-identical to pre15 (audio ES+apts content-exact); MG-B2 mv rung outputs
byte-identical PTV_RSYNC_SENSE=0 vs 1 on verdict-free fixtures; MG-B3
corrector-hold proof under injected per-slot R (readings move, zero
[PTV-RSCORR] activity); MG-1/2/3 per-slot independence + storm/glue
isolation; MG-4 stats-line grammar (single-input token-identical); MG-5
per-slot flash+beep oracle agreement; MG-6 #24/#27 shaped fixtures RENDERED
by the instrument (measurement, no fix expected).

(7m) GLUE CLASSIFICATION — pre15, task #33. NORMATIVE DESIGN =
analysis/ptvencoder-33-glue-classification.md (owner-approved 2026-07-18 with
all five §7 questions resolved to the doc's recommendations). A
classification/ROUTING fix on the existing pre4–pre7 glue machinery — the
flush decision tree, AGLUE verdicts, pair-expect handshake and absorber all
stay; what changes is which OWNER each one-sided audio-glue event class routes
to. One revert: PTV_NO_GLUECLASS=1 (wholesale). The five closures:
(d) GAP-VERDICT PROPAGATION (§2.5, demux_unwrap): the gap discriminator's
verdict was invisible to LAYERA — the disc buffer saw the same >1s step,
armed, and its flush BUTT-JOINTED the gap the discriminator just preserved
(fixture: 8s audio-PID-null → oracle +7986ms audio-EARLY; rr14-A4's 4.6s bake
= same chain). Now an audio-only, wall-absent, VIDEO-FRESH gap also advances
the disc buffer's continuity ref: no cycle, no flush — labels carry the gap
to AGLUE, which pads (PATRIOT-proven path). Video-fresh guard: after a
whole-program outage whose first resumed packet is audio, video's wall ref is
stale too → no propagation → today's LAYERA shape (fx-att-u900/b80 pinned).
(b1) PAD ROUND-TRIP CANCEL (§2.2 rule 3a, E5 pad ledger): every unregistered
forward GAP verdict opens a ledger entry {step,wall} (4 slots, 120s TTL); a
backward step matching an open pad (|step+pad| ≤ max(80ms, pad/4)) is the
pad's RETURN leg → APPLIED (aresample drops the pad's inserted silence),
never relabel-erased (rr14 A3: +150/−150 pair left a REAL −150ms bake). The
newest open entry is published per track so the demux §5.A.2 absorber (which
otherwise erases sub-1s backward steps before AGLUE sees them — the actual A3
erase site) declines the return leg and lets it flow to AGLUE's match.
(b3) LATE PAIR-EXPECT MATCH: a registered step whose TTL expired still
consumes on a VALUE match (±[-250,+500]ms of a >500ms step is the real
collision guard; a deep bank legally out-lives any fixed TTL — review-2 F1).
PTV_PAIR_EXPECT_TTL_US env override (TEST ONLY, G6).
(a) EVIDENCE-QUALITY REFUSE (§2.3, ptv_disc_flush): NEW per-dense-stream
label-health H = windowed EMA of Δdts/Δwall over ~30s (quiet path only;
healthy ≈ 1.0 ±5%, PTV_GLUE_HTOL_PCT tunes). Before a flush routes a >500ms
A-vs-V mismatch to the content path (2b/3a stamp + 2d retro), unhealthy H (or
a <3-packet new base during a fresh wild window) → REFUSE: per-stream OWN
butt-joints (the pre-pre4 posture, owner call Q1), loud [PTV-GLUE] REFUSED
line, refuse ledger + disturb_epoch bump (corrector freeze). Magnitude cannot
discriminate — PATRIOT's 30.8s was REAL (healthy H routes it unchanged);
Azorse's +31.078s was flood noise faithfully executed as a ~31s pad.
§2.4 REALIZATION TRIPWIRE: every non-erase verdict (GAP-pad / FLUSH-APPLY /
pad-cancel / above-cap stand-aside) arms pend_comp; if the pre11 slip probe
still parks near the verdict size 2s later (hard comp is instantaneous by
design — expected NEVER on forward, the G7 witness for backward >1s drops),
synthesize the parked remainder at the swr boundary (swr_inject_silence /
swr_drop_output — the resampler's own primitives, not a second actuator) +
WARNING + glue_events bump (corrector freeze). Steps >10s get an operator
ALERT line but stay UNBOUNDED (invariant mandate, owner call Q5).
(c) NBS STARVATION — §3: the demux corrupt-discard dropped every
corrupt-flagged packet on ALL streams but counted video only; a corrupted
audio phase (~46 pkt/s, Azorse broken-7.1-AAC class) starved the track
SILENTLY upstream of audio_q (thread blocked in recv; ADECWD structurally
blind — wd_pkts never advances; restart-only). Part 1 UNCONDITIONAL (even
under the kill): per-track acorrupt counter + rate-limited [PTV-ADISC] line +
acor= stats field. Part 2 OPT-IN (PTV_NBS_FILL=1, owner call Q2 —
observability first): while the track's packets arrive-and-discard with
nothing decoded >2s and video alive, the demux sends a FILL sentinel per
quantum (zero-size flagged pkt on audio_q — never counted as a packet, so
ADECWD cannot churn); the audio thread synthesizes stamped silence at the
expected next graph-door pts (labels dense, delivery alive, sensor valid ≈0,
corrector held off via the fill_active freeze). First real frame = resume
anchor: forward remainder GAP-pads; backward overlap is dropped (our own
synthesized silence — never erased). Tracks BORN into a broken phase stay
dead until real frames arrive (0.9.17.1 AFMT retry owns that).
(b4) DUKF observability (owner call Q3: accept + log): the resume/escape
drop-count lines promoted PTV_DIAG → always-on (arms only on ≥1s video jumps
— rare); no compensation this round.
CORRECTOR HANDOFF (§5): refuse → disturb_epoch; fill_active → event-active
"nbs silence-fill" (R is synthetic-flat on a filled track); pad-cancel and
tripwire synthesis → glue_events. Structural ownership of gap-shaped events,
pad round-trips, healthy-source routed mismatches and starvation phases moves
to the glue; the corrector keeps external/unknowable residuals only.
DEVIATION from the doc (flagged): §5 rule 4 asks the tripwire synthesis to
post an E_a ledger edit; implemented as a glue_events freeze WITHOUT the R
shift — the synthesis completes the resampler's own label-declared
compensation (slip→0 makes the sensor self-consistent), and a ledger post
would permanently offset R by the synthesized amount on a path expected never
to fire forward. Revisit if G7-class syntheses ever fire in production.
(Adversarial review rr15 BLESSED this and the other three declared
deviations.)
rr15 FIX ROUND (same pre15, second commit) — three CONFIRMED reject-grade
findings, all wire-reproduced by the reviewer, fixed and re-gated:
R1: the §2.5 video-fresh guard was a flat ≤2s bound — a whole-program
relabel delivered with a 1.3-2s stall and an AUDIO-first resume passed it
(video's last packet is exactly stall-old) → one-sided split, −2.6s bake
(fx-rr15-a2). Fixed: video must have flowed DURING the audio's absence —
video_last ≤ min(2s, wall_gap/2).
R2: the E5 pad ledger appended EVERY unregistered forward pad, including
E3-corroborated REAL-gap pads, which have nothing to unwind — a
coincidentally-sized both-stream backward relabel (In-Touch shape) then
"cancelled" against one and deleted 405ms of real content (fx-rr15-a3, the
AWE fleet class). Fixed: only FLOWING (splice-suspect) pads enter the ledger
— gate on wall_gap < step/2 + arrival-cadence EMA (new glue_cad_us: EMA of
nonzero fed-frame wall gaps = the PES-burst period; a real gap's absence
rides on top of one cadence).
R3: the F2 secondary refuse rule (<3-packet base during a wild window)
fired on HEALTHY channels — a benign sub-2s delivery stall closes one H
window at r≈0.47 = WILD, and a genuine Curiosity-ordering event 30-40s
later was REFUSED (−747ms baked, fx-rr15-a1b2). Fixed: WILD is DIRECTIONAL
— only flood-direction wildness (r > 1.5) arms the flood-recency window;
stalls read r < 1 and no longer count. (The primary EMA rule was verified
SOUND at the margin — 3.5-4% wobble still routes.)
F9: the NBS fill under-filled 30-37% (int-truncated 4-frame quantum vs the
≥100ms+jitter sentinel cadence) → +14.5s resume step after a 40s phase.
Fixed: synthesize the wall time actually elapsed since the previous quantum
with sub-frame remainder carry (clamped 2s/quantum).
Cosmetics folded: dead nbs_resume field and the never-read g_nbs_fill_st
atomics removed; the PTV_NO_GLUECLASS wire-vs-log-parity caveat documented
(F7). rr15 advisories F4 (tripwire realized-check is optimistic; fails open
to pre14) and F5 (~−2%/event H bias at flood cadence only) accepted as
known bounds, no code change this round.
supervisor (task #44, the owner's PLL lineage). NORMATIVE DESIGN =
analysis/ptvencoder-corrector-design.md (owner-approved 2026-07-16 with §9
resolutions folded in). Gated on the sensor soak CERTIFICATION (2026-07-16,
cor-3): oracle agreement Δ12–21ms on real excursions, human-verified in both
signs, NTSC 24h flatness 0/0/0.
WHAT: when input is healthy and the certified pre9/pre11 sensor's per-track R
(`lipsync=`, + = audio early) dwells outside an 80ms dead band for 5min stable
+ 3min event-free with delivery provably live, steer R→0 through the resampler
— the pre3 graph-door steer bus gains a fourth term (corr_us), proportional
R/30s slew-clamped to 2ms/s, PARK at |R|≤20ms held 60s (trim retained). A trim
safety net under the fast event path (the AGLUE/LAYERA escape-bake class:
JLTV +42.9s, Azorse +467ms), NOT a controller — authority is deliberately too
small to paper over a glue failure (per-engagement 5s, lifetime 10s → hard
DISARM+ERROR; a leaked >authority bake grinding at 2ms/s and logging loudly IS
the escalation signal).
STRUCTURE (MV-NORMATIVE): CorrState per (input-slot, audio-track) in
AudioState; R consumed only through the rsync_track_R() accessor (slot 0
hard-wired today); corr_us joins the sensor's inj term and every sink-label
measurement subtraction (anti-windup is structural — R feeds back only
realized trim, integration freezes while the slip probe reads ≠0).
DELIVERY-LIVENESS (§3): NEW per-rung g_mux_sent_wc wire-send watermark (one
relaxed store per successful av_interleaved_write_frame — the Newsmax2
dead-rung answer, owner-approved "build it") + DlvGate a_hi/v_hi watermarks +
mux_q depth; ALL rungs AND the input must be live across the whole dwell.
CONTAINMENT: DEFAULT ON (owner-directed 2026-07-17: every channel runs it
unmodified — parked and byte-inert when healthy, "it should work if channel
needs it or not"); kill PTV_NO_RSYNC_CORR=1 (kept forever); sensor off implies
corrector off.
Auto-disarm on sensor stale (incl. a master-side stale-track watchdog for the
one disarm the blocked audio thread cannot log), delivery death, event storm
(≥3 counted dwell resets/10min → 10min holdoff), implausible R (>5s sustained
5s), parked slip ≠0 >60s engaged, authority caps. One [PTV-RSCORR] line per
state change; stats field corr=±Nms (absent on a quiet channel).
GATES (fixture, this session): F1 +300ms-class bake converges (oracle-
confirmed on the wire, zero clicks on a real quiet-passage fixture,
pts-spacing flat); F2 mirror sign; F3 byte-inert armed-parked (healthy
content: audio ES byte-identical corrector on/off; per-track isolation —
clean sibling track untouched at byte level); F4 kill-switch parity (full ES
identity incl. video on the event fixture); F5 dwell immunity (STOP/CONT +
rewind seam + loudnorm); F6 delivery-death disarm/re-arm; F7 authority
disarm; F8 event-storm disarm; F9 WUCR/cadence/gate-skew non-coupling.
Independently reproduced + extended by the adversarial review (13 cells,
MERGE-READY): direction flip, pad-then-erase pair, 4.6s butt-jointed gap
(chase CORRECT — oracle +33ms tail), 9s event train byte-identical, 905s
clean channel silent under load. Numbers in the pre14 session notes.

(7k) CATCH-UP GOVERNOR FAILS OPEN ON AN UNTRUSTED RATE MEASUREMENT —
the Newsmax2 live defect (pre13; ptvencoder.c decode_thread governor +
ptvencoder_demux.c vin publish stamp + DIAG t= line).
WHY (live-proven, Newsmax2/cor-3 2026-07-16): pre11 restart-looped 4
births 01:50→04:07; every run wedged within seconds — dec=6.6/s on a
wire measured CLEAN (5.4Mb/s CBR 59.94pps, demux vpkt 60/s smooth all
run, vdrop/vcorrupt 0 at onset), vq pinned 725-784, dup 45/s, QSHED
churn + SELFHEAL every 5min re-stamping g_shed_wall so the governor
never released. Kill-switch A/B decisive: PTV_NO_CATCHGOV=1 added to
the channel env 04:20:13, the 04:21:11 restart of the SAME pre11
binary was instantly clean (dec 60/s, dup=0, vq=0). The realized brake
was ~160ms/frame ≈ 800000/gpps with gpps≈5 — a wrong-but-nonzero
measurement outvoting the declared 60 (the rr10 re-review A-1
residual, materialized), and NOTHING in the logs could show it: the
DIAG line carried no gpps.
FIX (three prongs, all fail-open):
 (a) TRUST GATE: govern only when measured >= declared AND the publish
     is FRESH (<30s, new vin_pps_wall stamp). Measured below declared
     is a broken measurement, not a slow source — declared itself can
     under-state the wire (29.97-with-fields on 59.94 = a 37.5pps
     brake), so an FFMAX floor is NOT sufficient: below declared means
     DO NOT GOVERN. An ungoverned catch-up burst is transient and
     recoverable; a brake that under-paces the wire pins vq full and
     self-sustains (shed → engage → starve → shed) until a human pulls
     the kill switch. Warm-up (measured==0) now also fails open — the
     birth backlog burst is exactly the recoverable kind.
 (b) ACTUATOR SELF-CHECK: each governor sleep is measured; waking
     >50ms late 3x in a rolling 10s (throttled/quantized wakeups =
     brake stretched by an unbounded factor) fails open for 60s with
     one [PTV-CATCHGOV] WARNING.
 (c) OBSERVABILITY (the N5 gap): DIAG t= line now prints
     gpps=measured/declared gov=engagement (+ govslip= strike count
     when >0) — `dec ≪ gpps*1.25 with vq pinned and gov=1` is now a
     log-diagnosable signature. PTV_VINDBG=1 traces the measurement
     window (RESET/PUBLISH) and governor transitions for field debug.
GATES: N1 3-phase fixture (declared=60 probe → genuine 8pps phase
publishes 8 → 60pps duty-cycled so >1s gaps freeze the window + churn):
pre12 reproduces the live signature (dec ~10/s vs 60pps wire, vq
pinned, no recovery), pre13 fails open and recovers to input rate ±2%;
N2 pre10 churn cell still binds (governance intact where measurement
is healthy); N3 rr10b-da/mv fixtures unchanged; N4 byte gate vs pre12
quiet-channel.
ALSO (7k, logging): [PTV-AVSYNC] DIAG estimate renamed lipsync= →
avlag= (avlag>0 = audio LATE) and [PTV-RSYNC] already prints R= — the
lipsync= token now appears ONLY on the -stats progress line (pre11
sensor, + = audio EARLY); the opposite-sign collision caused an
oracle-analysis error 2026-07-16.

(7j) SYMMETRIC DELIVERY GATE — hold EARLY VIDEO on the audio delivered
high-water (pre12; ptvencoder_gate.c §7.5b + encode_push_inner hook).
WHY (owner-demonstrated LIVE, AWE_Plus on cor-3, 2026-07-16): ffprobe of
the output read video start 1134.03 vs audio start 1131.69 — the mux
emits audio content ~2.3s OLDER than the concurrently-emitted video.
Cause: the fleet-wide loudnorm -af holds ~3s of audio inside its
analysis buffer, so audio content exits the pipeline seconds after the
same-PTS video already left. Players are fine (PTS-aligned) but the
WIRE is skewed: sync_check-class monitors (video_last − audio_last)
trip, and downstream buffers must cover the gap. The v0.7.0 gate is
one-directional — it holds audio that is EARLY; late audio has nothing
held for it, video simply leaves first.
FIX: per rung, a video packet whose DTS leads a_dlv_dts_hi (the newest
audio/copy DTS actually DELIVERED to that rung's mux_q — advanced by
the §7.5a drain/flush send loops; copied AC-3/MP2 counts, sparse
subs/SCTE-35 bypass as today) by more than 300ms (PTV_VDLV_BAND_US) is
queued FIFO until audio catches up. MEASURED, not assumed: on a channel
whose audio is not wall-late the audio waits in the §7.5a gate AHEAD of
the video front, so a_hi tracks the front within a tick and video is
never held — zero cost, byte-identical (fixture-proven). DEADLOCK
INVARIANT (comment in ptvencoder_gate.c): audio releases on
v_enc_dts_hi published at ENCODE time BEFORE the video packet can be
held; video releases on audio DELIVERY; the video hold never blocks its
thread (vmaxq overflow force-releases the OLDEST) — the cycle cannot
close; if the shape ever changes, the audio gate yields.
AUDIO-DEATH SAFETY (make-or-break, fixture-proven): if no audio is
delivered for PTV_VDELIVERY_CAP_MS (6s ≈ 2× the loudnorm class) while
video is held, ALL held video flushes and the hold DISARMS (one
[PTV-VDLV] WARNING) — an audio outage degrades to the pre12-less wire,
never a frozen channel; re-arms (one INFO) when delivery advances
again. A per-packet age backstop (same 6s, counted vdlvforced=)
releases through a flowing-but-permanently-behind audio path
(JLTV-class label spread) so pre12 never chases an upstream label bug
with unbounded latency. Sizing: hold FIFO = cap × out-fps + margin
[512..2048] slots; RSS bound ≈ cap × rung bitrate (~2.9MB @3.8Mbps top
rung). Single-input live with gated audio only: multiview slots share
one gate per rung (the high-water would key to the least-delayed slot)
— disabled with a startup note (pre10 D2 pattern); no-audio channels
disabled (must not pay the escape timeout at birth). Stats: vdlvhold=
(+ vdlvforced= when >0) + legend; lipsync=/lineage untouched (the hold
changes delivery timing, not labels; fps=/watchdog count emitted-to-
gate, so a birth hold shows normal fps).
GATES (local, cinestar full-mux tsp replay + x264 rung): W1 loudnorm
chain wire skew v−a median +1229ms (pre11) → −40ms (pre12), lipsync=
−1ms both, dup=0, vdlvhold≈1.2s steady = the measured audio hold; W2
plain chain 22MB output BYTE-IDENTICAL pre11↔pre12 (deterministic-join
live UDP); W3 45s both-PIDs audio-death: escape fired once at +6s,
fps never left 24.9–25.0 (video never froze), disarmed window clean,
re-arm on resume — the post-resume +45s audio-early label spread is the
KNOWN pre-existing #33 audio-only butt-joint (pre11 control identical:
lipsync=+45s, wire +47s; pre12 clamps the wire to +41s = its 6s bound);
W5 rewind seam (−305s shared backward jump, TruBLU class): LAYERA
jump/flush/AGLUE lines line-for-line identical pre11↔pre12, lipsync=
−6ms identical, and the gate re-closed the post-seam wire (+2059ms →
−43ms); SCTE-35 fixture: demux/copy path identical pre11↔pre12 —
NOTE the audio-batch branch itself cannot mux SCTE-35 (v2 0002 is not
an ancestor here; stream lands as bin_data with no packets on BOTH
binaries — pre-existing branch gap, unchanged by pre12, restored when
0002 stacks on); W6 held ~150 pkts ≪ 512 maxq at the 6s bound, RSS
pre12 127.6MB vs pre11 128.3MB. KNOWN SEMANTICS (deliberate, spec'd): a
channel carrying a dense COPIED audio track anchors a_hi at the video
front (the copy is delivered in lockstep), so the wire always has
fresh audio beside video, but the -af-delayed TRANSCODED track still
trails by its hold on such channels — same as pre11 (w1b: AC-3 copy
+8ms, AAC +1.9s). PTV_NO_VDELIVERY=1 reverts.

(7i) RESAMPLER-SCOPED SLIP PROBE — buffering -af hold no longer biases
lipsync= (pre11; ptvencoder_audio.c sensor + build_audio_filter).
WHY (pre9 review Defect 1, shipped as known-limitation, FALSIFIED LIVE):
the review accepted the bias because no production channel ran a
buffering -af — then the owner re-enabled loudnorm FLEET-WIDE 2026-07-13
(chain `aresample=async=1000:min_hard_comp=0.03,loudnorm=...,
aresample=48000`) and genuine pre9 channels on cor-3 (Daystar, Mysteria,
Newsmax2) read lipsync=+2913..+2919ms — exactly the fixture bias
(+2914ms) — biasing the entire sensor soak.
ROOT CAUSE: the pre9 slip term spanned the WHOLE -af graph (door label
head `acomp_exp_us` − sink head − swr_get_delay), so any filter that
BUFFERS content (loudnorm's ~3s analysis window) parked its hold in
slip. But a passive filter's hold preserves labels — content comes out
carrying the labels it went in with, the content→label mapping m_a is
untouched — it is shared latency, not desync.
FIX: scope the label-head pair to the async-aresample FILTER's own
links: slip = (input-link label head) − (output-link label head) −
swr_get_delay, via FilterLink.current_pts_us + the link frame/sample
counters (libavfilter/filters.h, in-tree; all maintained by the generic
consume path on every frame crossing a link). current_pts_us is the
START of the last chunk CONSUMED off a link, so true heads are
reconstructed symmetrically: head = start + avg consumed-chunk duration
(sample_count_out/frame_count_out — exact for steady chunking: mp2
1152-sample frames in, loudnorm's 100ms consume quantum out), and the
output head additionally adds the QUEUED duration (sample_count_in −
sample_count_out: produced-but-unconsumed frames sitting in the link
fifo carry labels dense above the consumed head). Fixture-measured,
both reconstructions matter vs the 50ms dead band: consumed-start
labels alone left a false +26..+40ms residual through loudnorm (its
100ms quantum vs the 24ms input frames) — with reconstruction the full
production chain reads slip=+0, R=−1ms. The aresample AVFilterContext
is captured alongside fg_swr at graph build (and cleared/re-found
across AFMT rebuilds). Filters before OR after the resampler are now
outside the probe, whatever the chain order; the boundary still sees
exactly the parked-compensation class (labels diverging from target
while the swr backlog stays flat = [PTV-SWRDELAY]'s reason to exist).
50ms dead band unchanged.
SCOPE NOTE (for the record, unchanged semantics): a rate-CHANGING -af
(atempo) genuinely alters the content mapping; the sensor treats the
user's -af chain as label-faithful and never measured that class — pre9
read it as a bogus constant (its hold), pre11 reads 0. Out of scope by
design, same as source-label lies (TRACKUP): R = pipeline-ADDED desync.
GATES (dg-run/tsp harness, fresh build): L1 EXACT production loudnorm
chain on a clean channel ≥10min: lipsync settles ±40ms of 0 (was
+2914); L2 injected −300ms label shift THROUGH loudnorm: R reads the
injection ±40ms (the fix must not eat real desync); L3 aresample-only
rr9 core gates (a300b/v200b/b70) identical to pre9/pre10 ±5ms; L4
fx-wcl350 byte gate IDENTICAL (sensor stays passive); L5 stats/DIAG
format unchanged; L6 CPU unchanged. Legend caveat replaced by the
exclusion note; (7g)'s KNOWN LIMITATION is RESOLVED here.

(7h) BIRTH-ARMED CHURN — MODE RELEASE + CONSUMPTION RATE-SHAPE (pre10;
ptvencoder_clock.c detector / ptvencoder_gate.c CUSHION_RELEASE /
decode_thread governor / ptvencoder_demux.c degraded admission).
WHY (pre10 Phase-A verdicts, local 4-instance birth-contention cells,
p8deep = live depth): the live churner is a CAPACITY-DEFICIT LIMIT CYCLE
armed at birth — video_q fills to exactly the live cap 784, [PTV-QSHED]
full-cycles with median period 6.2s (live ~6s), frame_q starved 88-97% of
samples, dup 52/s at depth. Every armed state releases either never
(g_delivery_maxq, by design) or only on conditions the churn itself makes
unreachable: the cushion tier arms at birth (~6s under contention) and its
only release is 6h with ZERO starvation episodes — each cycle resets that
clock, so post-recovery residue held cushion=2535ms + fqhw=160 + grown
gate caps 12min after full recovery (in production: frame pool + NVENC
surface registrations pinned at maximum forever, WUCR railed −15000ppm
filling a target the deficit can never fill). And the post-shed catch-up
decode path is UNGOVERNED (frame_q pushes are drop-oldest NONBLOCK, so
backpressure vanishes exactly when it matters): measured catch-up dec p95
2.2x realtime (133/s on a 59.94 channel) — N co-located churners
burst-feed a shared device in phase (the owner's "feeding it with bursts"
observation). Provenance: rc enters the SAME stuck state at identical
aggregate cost but expresses it as per-packet mid-GOP shredding that at
live depth also kills audio at the demux door (adrop 29.4/s = ~58% of
audio packets, garbled slow-motion output) — the DO-NOT-REGRESS-TO
baseline; pre8's whole-GOP expression keeps adrop 0 and honest frames.
FOUR CHANGES (kill-switch each; ALL structurally inert on a healthy
channel — byte-gate proven):
(e) CUSHION RELEASE [PTV_NO_CUSHREL]: the pre8 (b)/(c) starvation
contradiction (frame_q <=2 while input FLOWS, g_v_arrive_wc) applied to
the adaptive TIER — holding a deeper fill target for a buffer the deficit
can never fill. Held >=60s with the tier raised ->
cushion_escalate(CUSHION_RELEASE): tier back to base + symmetric gate-cap
restore (same stores as SHRINK), loud [PTV-CUSHREL], one release per
firing with a 60s re-fire floor + 10min post-release GROW suppression (a
persistent deficit would otherwise re-GROW seconds later and flap the
pair once a minute). The timer forgives <=5s refill blips — the ~6s cycle
refills frame_q ~1-2s per cycle, so demanding 60s CONTINUOUS starvation
would be unreachable under exactly the symptom (starve fraction measured
0.88-0.97). Input NOT flowing hard-resets the timer: a genuine
stall/outage KEEPS its cushion (STOP/CONT gate).
(f) GOVERNED CATCH-UP [PTV_NO_CATCHGOV]: deficit-recovery decode capped
at 1.25x THE INPUT's realtime — per-frame floor 4/5 of the INPUT tick,
each decode thread (mv slot or single) pacing by ITS OWN input's rate:
the demux-measured arrival pps (4s gap-guarded window, published per
input), clamped to the declared header rate while the measurement warms
up; with NEITHER available (VFR at startup) the governor FAILS OPEN — an
ungoverned burst is transient, a governed wedge is permanent. [rr10
review fix D1: the first cut paced by the MASTER OUTPUT tick — any input
whose pps exceeds 1.25x out-fps (mixed-fps mv slots, single inputs
decimated by -r) was governed below its own arrival rate, vq re-capped,
QSHED re-stamped g_shed_wall, and the governor manufactured a PERMANENT
shed cycle (fixture: 65 sheds still firing at t=300 + WUCR railed
-14400ppm vs pre9's 14 ceased at t=134).] Engaged ONLY while a
self-shed/heal happened within 10min (g_shed_wall, never stamped on a
clean channel) AND video_q holds >1s of INPUT backlog; disengage is
by BACKLOG DRAIN, not the time window, so the window cannot expire
mid-backlog into an ungoverned tail burst. Under contention the floor
never binds (decode is slower than it); normal steady-state decode runs
1.0x by supply and never sees a sleep. Max sleep per frame = 0.8 of the
input tick at any fps (13ms at 59.94, 32ms at 25) at packet boundaries
keeps the pre8 heal executor reachable (rr8 defect-1 history).
The shed itself stays — correct load-shedding; only its aftermath stops
arriving as a device-max burst.
(g) PHASE JITTER [PTV_NO_PHASEJIT]: deterministic per-PID +/-20%
(g_jit_milli 800..1200, Knuth hash of getpid()) on the head-shed
depth-gate margin (128 -> 102..153 pkts) and the SELFHEAL 5min re-fire
(4-6min) — co-located instances cannot phase-lock their burst cycles.
Thresholds are consulted only inside shed/heal paths: inert when nothing
sheds.
(h) DEGRADED MODE [opt-in PTV_DEGRADED=1, DEFAULT OFF]: >=3min of
persistent QSHED full-cycles (train = tail-arms <=30s apart; median cycle
6.2s) -> flush the stale backlog (selfheal re-prime) and go DEMAND-DRIVEN
at the live edge: an arriving GOP is admitted only when video_q <= ~1s —
the queue depth IS the decode-throughput measurement (no estimator), all
decisions at IDR boundaries only (admission never flips mid-GOP), and
retained latency self-scales with the deficit (~2s content/utilization)
instead of accumulating. DESIGN ITERATION (fixture-measured, two rounds):
modulus-K admission stopped the full-cycles but thereby stopped their
latency-SHEDDING — vq parked at 8s of content = 60s of DECODE TIME on a
12% box, hs grew +59s monotonic, and audio (A/V-locked to that delay)
died at the demux door at ~40/s (the rc class); the defaults churn keeps
hs bounded 8-14s by head-shedding old content and audio demonstrably
rides that. Demand admission reproduces the churn's live-edge property
without its cycle. 60s of CONTINUOUS decode headroom (frame_q un-starved
with vq shallow) releases back to full admission at that IDR (re-entry
needs a fresh 3min train — hysteresis). Loud [PTV-DEGRADED] enter/status/
release. Default-off because the local repro cannot prove the production
self-sustainment it targets (state-dependent NVENC/RM/VRAM per-frame
cost) — it ships as the cor-3 experiment lever; byte-identical no-op with
the env unset.
DEGRADED SCOPE + KNOWN LIMITS (rr10 review round): SINGLE-INPUT ONLY
[rr10 D2] — on multiview the entry's backlog flush (g_selfheal_req) has
no decode-side consumer (mv slot decode runs with d->hold) and the
release headroom reads the COMPOSITE frame_q, the wrong signal for a
slot; PTV_DEGRADED=1 on a multiview invocation is hard-disabled with a
loud [PTV-DEGRADED] startup WARNING. With PTV_NO_SELFHEAL set, the entry
backlog flush is SKIPPED [rr10 advisory 4] — degraded admission then
converges to the live edge only as demand admission drains the stale
backlog (slower, no wedge). Entry is CALIBRATED TO ~6s GOP-shed cycles
[rr10 advisory 1]: full-cycles more than 30s apart never form a train,
so very-long-GOP channels (e.g. 24s-GOP mpeg2: cycles every 50-67s)
never enter — the lever is a silent no-op there (scaling the train gap
with GOP length is a possible later behavior change, not taken). Known
pre-existing (not pre10): a mid-stream UDP join on long-GOP mpeg2 can
take ~24s to the first sequence header at graph build (probe behavior).
GATES (pre10-cell.sh/pre10-sum.py fixtures + dg-run byte gate; numbers in
the session report): G1 p8deep-equivalent churn cell — dup <=52/s, catch-up
dec p95 <=1.3x realtime (was 2.2x), adrop 0, no sustained ~6s full-cycle
train under contention beyond isolated sheds; G2 post-lift recovery <2min
unchanged (dec settles 59.94, dup slope 0, vq drains); G3 release
correctness — [PTV-CUSHREL] fires under a sustained contradiction
(PTV_SLOW_DEC_US cell) and does NOT fire across a 90s sender STOP/CONT
(cushion retained through a genuine stall); G4 clean-channel byte gate —
fx-wcl350 pre9-vs-pre10 output BYTE-IDENTICAL (all four features
structurally inert without a shed/starvation episode); G5 churn-cell
demux adrop stays 0 (the rc 29/s door-kill class must never return).
rr10 FIX-ROUND RE-VALIDATION (input-rate currency + mv DEGRADED gate):
R1 rr10b-da cell (59.94p in, -r 30000/1001 out, 60s slow-decode window)
— fixed: 7 full-cycles all in-window, tail ZERO sheds, head-shed 987
pkts, dup frozen 257, no WUCR rail (rejected build: 65 sheds churning at
t=300, railed -14400ppm; pre9: 14 ceased t=134); attribution leg
PTV_NO_CATCHGOV=1 statistically identical (6/942/253). R2 production-
mirror 4-input mosaic, slot0=59.94p: 21 full-cycles / head-shed 1305 vs
pre9 26/1327 (rejected: 51/6233 churning). R3 churn cell reproduced (dup
52.1-52.3, catch-up decP95 75-76, adrop 0, CUSHREL at 60.0s x4). R4 byte
gate IDENTICAL (62652416 B). R5 90s STOP/CONT zero CUSHREL, cushion
retained. R6 degraded single-input lifecycle intact + 2-input mv with
PTV_DEGRADED=1 log-proven disabled at startup.

(7g) RESIDUAL LIP-SYNC SENSOR, PASSIVE (pre9; component 1 of the
residual-sync supervisor — analysis/ptvencoder-residual-sync-supervisor.md).
WHY: every desync class so far needed its own detector+glue, and one missed
event bakes a permanent offset the internals cannot see (blind-sensor
history: house-stamped-audio vs house-stamped-video cancels by
construction; [PTV-LIPSYNC]'s async_pad accounting read −2986ms on a
channel the oracle measured +24ms, Lindel 2026-07-15 — resampler ACTIVITY
is not desync). This round ships the NON-BLIND sensor only — NO ACTUATION
(grep-proven: every g_rsx read terminates in av_log/snprintf); the
corrector is a later round, gated on this sensor matching the external
oracle in a live soak.
DESIGN (full model at the RsyncSense declaration in ptvencoder.h): each
stream's source→output content mapping measured independently against the
post-demux (post-glue) label domain + a per-stream ledger of every label
EDIT the pipeline itself made:
  m_v = EMA[out−src] per EMITTED frame (dups included — the dup ratchet is
        a real presentation shift, MEASURED, not read back from house_skew,
        a control variable); out on the exact-rational axis (the integer
        tick would re-import the ~10ppm NTSC drift into the sensor);
  m_a = EMA[out − (sink_src − inj) − slip]: inj = AGLUE glue_off + AVLOCK
        house_skew (recovers the RAW post-demux label at the sink); slip =
        the resampler's UN-REALIZED correction (door-label head − sink-
        label head − swr_get_delay, 50ms dead band) — the parked-slip
        class that label math alone is blind to ([PTV-SWRDELAY]'s reason);
  E_s = per-stream demux label-edit ledger (discontinuity self-rebase,
        LAYERA flush persists, pre5 retro-corrections; pure 2^33 wraps
        EXCLUDED — always genuine+shared, and posting them would spike R
        by the wrap period during every A-before-V wrap straddle);
  R = (m_v+E_v) − (m_a+E_a), + = audio EARLY (= −(disc-oracle "ADDED",
      which prints + = audio made later)).
Shared source discontinuities and AVLOCK-realized retiming cancel (a glue
that moves BOTH ledgers equally is latency, not desync — the wedge/
AUTO-BANK posture); what shows: AGLUE relabel-erases (glue_off),
per-stream-UNEQUAL demux rebases (the JLTV/Azorse wrong-glue bake class),
parked resampler slip, and any label-followed single-stream jump.
Aresample pads/drops that fill genuine label gaps are mapping-neutral by
construction (label-referenced accounting, not span accounting).
SURFACE: `lipsync=±Nms` on the stats line (a0:/a1: per track when
multi-track; `--` when a side has not flowed for 3s — no stale anchors) +
rate-limited [PTV-RSYNC] DIAG line (R + dm/ev/ea/glue/hs/slip components)
under PTV_DIAG. Single-input only: multiview publishes NOTHING (n_a=0, mv
stats line unchanged) rather than garbage — per-slot lineage belongs to
the compositor, a later round. PTV_RSYNC_SENSE=0 disables.
GATES (local dg-run/tsp harness; oracle = disc-oracle differential
POST−PRE, instrument-constant-corrected; agreement bands ±20ms clean /
±40ms events; detailed numbers in the session report):
  (1) clean fx-smoke steady state: R vs oracle within ±20ms (including the
      sample's own −13.98s LAYERA glue: ev=ea=−13980ms cancel exactly);
  (2) baked offset: audio −300ms label shift (backward absorber erases →
      E_a=+300ms) — R = −300ms-class, agrees with the oracle incl. sign;
      +300ms forward control (AGLUE GAP-pads, faithful): both ~0;
  (3) event neutrality fx-wclose / fx-splitband: R back within ±40ms after
      the glue, no spurious drift during LAYERA buffering;
  (4) wedge (PTV_SLOW_DEC_US window): R bounded and near 0 post recovery —
      hs-injected pads are matched by the measured video ratchet;
  (5) fx-wcl350 full-file BYTE gate pre8-vs-pre9 IDENTICAL (passive proof)
      + battery spots line-identical modulo the new fields; 15-min soak
      lipsync= flat near 0, CPU unchanged;
  (6) sign proof: video −200ms label shift (absorber erases → E_v=+200ms →
      video presented later) reads lipsync POSITIVE (+ = audio early).
      (The forward +200ms variant is NOT a sign fixture: it passes the 1s
      forward threshold unabsorbed = faithfully realized = added 0.)
LIVE SOAK MUST VALIDATE before the corrector round (the design doc's hard
gate): sensor-vs-oracle SIGN and SLOPE agreement on (a) a clean channel,
(b) a TruBLU rewind session, (c) a provoked escape/garbage episode. A
sensor that disagrees with the oracle is discarded, not tuned around.
KNOWN LIMITATION — RESOLVED in pre11 (7i) (adversarial review Defect 1,
fixture-proven, then falsified live by the 2026-07-13 fleet-wide loudnorm
re-enable: Daystar/Mysteria/Newsmax2 read +2913..+2919ms): the pre9 slip
probe (audio.c drain) measured door-label head − sink head − swr_delay =
the WHOLE -af graph's hold, so a buffering filter (loudnorm: +2914ms
constant false audio-early; long lookaheads) biased R by its hold
forever. pre11 scopes the probe to the async-aresample filter boundary —
see (7i). Soak interpretation notes from review: (a) AWE-class sources whose
labels lie read R=0 BY DESIGN (R = pipeline-ADDED desync only — the
pts-spacing flatness gate stays mandatory, TRACKUP class invisible here);
(b) EMA τ30s → alert only on multi-minute dwell; (c) parked slip <50ms is
under the dead band (floor).

(7f) #32 WEDGE — GOP-COHERENT VIDEO OVERFLOW + STARVATION-CONTRADICTION
RECOVERY (pre8; demux_dispatch/decode_thread/output_thread/cushion_escalate).
THE DEFECT (live-proven on cor-3 2026-07-15, owner-tcpdump-verified clean
wire; the most severe known): slow NVDEC init (mass-restart GPU contention)
→ video_q fills to cap (observed vq=664 pinned) → the demux TAIL-DROPPED
arriving video PER-PACKET, MID-GOP (~70% dropped) → the decoder received GOP
fragments and produced ~11% of realtime (5.6 pkt/s on a 50fps channel) → the
queue never drained → the drop policy fragmented its own input FOREVER on a
continuous wire. Self-sustaining; entry also mid-run whenever the consumer
fell behind long enough (Newsmax2, 20min after a clean single restart).
Escape only on a load lull, and even then the AUTO-BANK/delivery ratchet
kept the channel up to ~92s behind live permanently (Daystar_esp). Version-
independent — the tail-drop policy is ancient. Wedge signature: frameq=1
pinned + vq at cap + rho railed −15000 + dlvhold ratcheted + vdrop climbing
+ tick-multiple ASTEP flood, all with clean input.
LOCAL RULE-0 FIXTURE (new stress knob PTV_SLOW_DEC_US windowed per-packet
decode cost — the faithful slow-NVDEC stand-in; PTV_SLOW_US slows the OUTPUT
thread, which frame_q drop-oldest decouples from video_q): ntsc60 59.94fps
g60 looping UDP, 40ms/pkt (32% consumption) for t=60..150s. PRE7 reproduced
the full signature: vq pinned 782-784(cap), frameq 0-1 for one continuous
87.3s starvation, vdrop 2893 climbing per-packet mid-GOP, dup ~40/s,
rho −15000 railed, async +489k ppm, dlvhold 3.3→22.2s ratchet, hs +35.8s
monotonic, 1463 ASTEP + 1047 ACOMP, ~556 in-window h264 fragment-decode
errors; recovery only by a 48s content leap.
FOUR FIXES (each with a kill-switch):
(a) GOP-COHERENT OVERFLOW [PTV_NO_QSHED]: when video_q must shed, never
drop random tail packets. Demux overflow → requests a HEAD flush from the
decoder (only the consumer can pop the head): decode drops whole oldest
GOPs — head to the next keyframe, which then decodes — one GOP per request,
depth-gated against stale requests; meanwhile the demux TAIL-drops the
arriving stream to the next IDR that fits, so the queue never holds a
headless GOP. The decoder ALWAYS receives contiguous decodable GOPs → runs
at full speed the instant it can consume → the fragmentation feedback loop
is structurally impossible. Head-shed also drops OLDEST content first =
overflow now DRAINS latency instead of pinning the stale head. Audio
overflow sheds whole frames oldest-first (was drop-newest). All sheds are
accounted: "[PTV-QSHED] video_q overflow: dropped GOP <n> pkts (<ms>ms)"
(rate-limited, honest totals).
(b) RATCHET RELEASE ON STARVATION [PTV_NO_RATCHREL]: frame_q ≤2 frames for
≥5s while input IS flowing (new g_v_arrive_wc demux stamp) with an ARMED
bank = holding gate latency for a buffer that is empty — a contradiction
(the wedge/aftermath shape). cushion_escalate(BANK_RELEASE): bank + gate
caps back to base immediately instead of the 6h decay; blocking push
disarms, retained latency drains via the normal catch-up path. Normal
deep-bank operation is untouched (a delivery gap = input NOT flowing; a
catch-up refill = frame_q NOT starved).
(c) SELF-HEAL RE-PRIME BACKSTOP [PTV_NO_SELFHEAL]: the same contradiction
sustained ≥30s → the decode thread flushes video_q + its own state
(avcodec_flush_buffers) and resumes at the next IDR — what a supervisor
restart achieves, in-process, anchors preserved. Rate-limited 1/5min,
loudly logged [PTV-SELFHEAL].
(d) SELF-MADE-GAP LOG HONESTY: every self-inflicted drop stamps
g_shed_wall/g_shed_cnt; AGLUE/ASTEP lines within 5s of one carry
" [self: N pkts shed]" so our own drops are never again misread as source
burstiness (the "bursty channel" taxonomy was self-portraiture — owner
mandate). Log shapes byte-identical when nothing was shed.
GATES (pre7 = 77e7410e61 + knob vs pre8, same fixture): in-window h264
fragment errors 556→2 (only the identical 154-line tune-in burst remains);
decoder input whole GOPs (QSHED head 1765 pkts + tail 384, fully accounted
vs blind vdrop=2893); hs bounded sawtooth ≤13.4s vs +35.8s monotonic;
dlvhold ≤14.9s sawtooth vs 22.2s pinned; ASTEP 655 vs 1463; adrop 0 vs 361.
RECOVERY after release: dup slope→0 and frame_q refilled 157/160 by +7s,
dlvhold 14.9→2.99s by +17s (→ baseline ~1.9s), retained latency +1.63s
(cushion-scale) vs pre7's 48s content leap — full recovery ≤60s: PASS.
BIRTH cell (bo birth, frozen 20s at spawn, 25fps): pre8 comes up clean with
BOUNDED retained latency — one accounted head-GOP shed at CONT (61 pkts/
1200ms), dlvhold 3.6→2.4s (baseline) by +5min vs pre7 5.2-5.9s still
draining (~10min more to go). DBL cell (STOP 20/5/10 — the AUTO-BANK
ratchet): pre7/pre8 SHAPE-IDENTICAL by design (bank 12s, dlvhold ~10.5s
pinned, buffers FULL = no contradiction, (b) correctly silent) — normal
deep-bank operation untouched. (b) cell (dbl-armed bank + slow-consumer
window): pre8 "[PTV-CUSHION] BANK released (was 12.0s): frame_q starved
5.0s with input flowing" at exactly 5.0s, caps back to 3.0s, dlvhold then
DRAINS 10.5→5.5s and falling; pre7 REBUILDS the ratchet after the window
(bank climbing back toward 12s, dlvhold rising — 6h decay the only out).
(c)-alone cell (PTV_NO_QSHED+PTV_NO_RATCHREL, the slowdown fixture): the
wedge re-forms exactly as pre7 (vdrop 2000, 495 fragment errors, hs to
+34.3s); [PTV-SELFHEAL] fires at 30s, flushes 783 pkts + decoder, resumes
at IDR — hs collapses +12.0→+1.25s at the heal; full recovery post-window
(dlvhold 22→2.9s). Rate limit 1/5min held. The heal is what a supervisor
restart achieves, in-process. (Session-109 rule applied: the post-heal
drop-until-IDR carries a 5s escape so long-GOP sources can never freeze.)
REGRESSION BATTERY (14 fixtures vs the pre7 baselines, dg-run live-UDP +
tsp regulate): wclose/sym/aonly/b300/tb900 (tb900 vs the valid rr7-p7
baseline — the dg-tb900-p7 file was a truncated pre-event run) event-line
IDENTICAL; splitband/mir2/dbl/tb30/wcl400/wc520/att-u900/att-b80 identical
applied offsets, verdicts, vid_err values and line shapes with only
buffer-cycle count jitter (flush pkts ±1-6; the mir2 GAP hole +456→+552ms
= exactly 4 extra OLD mp2 frames × 24ms) — same class and magnitude as the
pre7 A/A control (p7 vs rr7-p7: 64→70, +456→+576). fx-wcl350 full-file
byte gate: fresh QUIET-BOX serial pre7-vs-pre8 A/B FULL-FILE BYTE-IDENTICAL (62,652,416 bytes; a first contended-box attempt was root-caused to the pre7 leg running concurrent with the soak — fps=1.7 stall, invalid A/B).
20-min clean soak: dup=0, hs=+0, rho ppm-scale, zero QSHED/SELFHEAL/BANK
events (3 identical +824ms source-label AGLUE GAPs = a known sample
artifact, present on pre7 too; post-EOF starvation tail = input-ended
artifact). NOTHING regressed — the new machinery is inert without
overflow/starvation, and log shapes are byte-identical when nothing shed.
FOLLOW-UP note: retained-latency drain after a release rides the normal
catch-up path (ppm-scale + drop-oldest), minutes not seconds — deliberate
(no viewer-visible fast-forward).
FOLLOW-UP (out of scope this round, noted): the WUCR ±15000ppm sustained
slow-down authority cap is structurally too small vs a consumer running
≥5% slow (rails during any wedge); revisit with its own fixture.
REVIEW DEFECT 1 (rr8 adversarial review, fixture-proven; FIXED in the
follow-up commit): the new vq_tail_drop mode exited ONLY on a KEY packet
that fit — on a no-IDR source (intra-refresh / encoder-restart feeds, or
any GOP longer than the ~13s queue) no IDR ever came: the demux never
resumed sending, the decode thread blocked forever on the emptied video_q,
and the [PTV-SELFHEAL] request was NEVER SERVED (the heal executor lives in
that blocked thread) → permanent freeze-frame with live audio (noidr2
fixture: pre8 dec frozen 214s, dup climbing 60/s; pre7 fully recovered).
The Session-109 rule was in DUKF and the heal resume but not here. FIX:
(1) vq_tail_drop carries the same g_dukf_escape_us time escape — no IDR
within 5s of arming → resume sending non-IDR ("[PTV-QSHED] tail escape",
deadline re-armed if the queue is genuinely full again, so freeze episodes
are bounded by the escape, never a full GOP); (2) decode_thread recv is now
a timed poll that serves a pending g_selfheal_req on the empty-queue path,
so the heal stays reachable even when no packet ever arrives; (3) the
heal's avcodec_flush_buffers is DEFERRED from request time to IDR-arrival
(fixture-forced: flushing at request time destroyed h264 sync, and with no
recovery point in the stream the decoder then consumed packets forever
without emitting a frame — a permanent freeze the heal itself caused; the
mid-GOP escape now resumes with the decoder deliberately unflushed, its
established sync being the only one a no-IDR source will ever have.
Observably identical on IDR-rich sources — no packet touches the decoder
between request and IDR).

(7e) LAYERA CONTINUING-STREAM KEEP — cycle-2 video deletion on split events +
borrowed-base false crossing (pre7; ptv_disc_flush + demux loop). On every
split (two-cycle) discontinuity event, the second cycle's buffer holds ~500ms
of packets from the OTHER, already-glued stream; ptv_disc_classify's
borrowed-base fallback matched them against the jumping stream's bases, they
landed OLD, and the flush DELETED them — measured (PTV_DISCDBG localization,
fx-wcl400/fx-wc520 live-UDP): flush-2 discarded 25/31 CONTINUING VIDEO packets
(0.5-0.6s), and because the deletion breaks the decoder reference chain the
on-air picture loss ran to the next IDR — content-coverage instrument: 65 ref
frames (2.6s) of post-seam content missing on wcl400, 52 (2.08s) on wc520 —
a visible skip per split event (live: Azorse 2026-07-14 05:57 flush "160
pkts: old=22 new=138", the 22 = continuing video). A/V sync was unharmed
(house dup-fill + AVLOCK compensate) — picture continuity was the defect.
FIX (a): at flush, an own-continuous packet (|delta vs own last_dts_us| <=
1s at arrival, tracked per packet) of a stream with NO own bases this cycle
is never classified against borrowed bases. It is KEPT ON ITS OWN TIMELINE
AT OFFSET 0 (timeline 2 — exactly what the normal path would have dispatched)
when the stream is already on the event's continuous timeline (video:
pair_vid_defined; audio: pair_has/pair_prov) AND the cycle's trigger is a
transcoded dense stream (vstream/astream); otherwise it keeps the legacy
discard shape (timeline 0 — where legacy classification put it in every
pinned fixture). Kept-continuing packets advance last_sent_dts without the
applied offset; a separate "[PTV-LAYERA] flush kept N continuing pkt(s)"
line reports them (the main flush-line format is pinned). Both narrowing
gates are deliberate: no-pair-state streams keep the first-cycle video-only
discard byte-identically (fx-att-u900/b80, the GAP-pad shape), and
copy/unconsumed-triggered cycles (TruBLU rewinds: the AC-3 leg crossing
~0.6s after video+audio already glued, fx-tb30 flush-2 old=47) keep
production-proven behavior byte-identically per the TruBLU mandate —
INCLUDING their continuing-stream discard, a known, deliberately retained
cost this round.
FIX (b), same mechanism (pre6 review residual 4): the demux-loop transition
record (classify==1 -> record own bases + transitioned) is gated on
!own_cont — a NON-jumping stream whose labels coincidentally land within the
100ms tolerance of the partner's new base no longer "crosses" with fake own
bases and off≈0, which had routed the full partner offset as an expected
step (fx-att-rpt event 2 — a mirror pair: video +3.020 with a REAL mp2 jump
−2.856 arriving ~380ms later; pre6's fake early all-transitioned flush is
gone, the cycle runs to timeout and the audio's real jump joins it → pre7 =
one both-crossed flush, applied −2.980, expect −5.548 registered and
APPLIED; landmarks within 20ms of pre6, oracle within 1 bin — the fake
crossing removed, the real one handled; adversarial-review-corrected
description). A genuine crosser's first stepped packet has own_cont=0
(>1s own delta, dts-based), so every real transition records exactly as
before. SCOPE (review-corrected): the keep applies to ANY transcoded-
triggered cycle inside the 5s pair window — including independent wobbles
while another PID holds the window open (fx-dbl 3c wobble: old 39->12,
kept 26) — not only strict two-cycle splits; verified strictly better at
pre6 oracle values everywhere. Both fixes only under g_shared_flush
(PTV_NO_SHARED_FLUSH reverts wholesale).
Gates (pre6 = 3973c97cab vs pre7, live-UDP dg-run + tsp regulate):
fx-wcl400/fx-wc520 — flush-2 "old=43 new=24" -> "old=18/12 new=24" + "kept
25/31 continuing", applied offsets, +596/+452ms expect registration and
AGLUE APPLIED lines byte-same as pre6; content coverage across the seam 65
missing ref frames -> 0 (wcl400; review re-measured 57->0), 52 -> 40
(wc520; the 40 = the source's own mid-GOP label-jump decode loss —
review-verified a STRICT PREFIX of pre6's hole, no video deleted by pre7);
oracle differential 0ms both (pre6 values). fx-att-rpt event 2 as above.
Byte gates (pre6-vs-pre7): fx-wcl350 full-file BYTE-IDENTICAL; fx-tb30/
fx-att-u900/fx-att-b80 gate on log-line/flush-shape equality (their A/A
controls byte-differ per run — live house-clock anchoring; the pre6-review-
established criterion). Regression battery
wclose/splitband/mir2/dbl/sym/tb900/aonly/b300 at pre6 oracle numbers and
line shapes (split-event flush lines gain the kept-continuing line and
smaller old= counts — documented per fixture in the session notes).
20.4-min clean-source soak: zero events, stats flat.

(7d) SHARED-FLUSH 3a INHERITS IN EVERY BAND — split-flush in-band bake
(pre6; ptv_disc_flush). Live (Azorse_TV 2026-07-14 05:57): video-only flush
applied −6.857; the audio-only flush 3s later applied its OWN −6.961 because
the 104ms disagreement sat inside PTV_PAIR_EPS_US (3a band kept aud_off) —
the split baked 104ms of relative A/V desync per event. In a SAME-cycle full
flush a sub-band disagreement is bookkeeping (ONE offset either way,
alignment preserved); in a SPLIT flush the audio-only cycle applies a SECOND
offset, permanently erasing the trailing-OLD discard hole + the sub-band
A-vs-V jump difference. FIX = 3a inherits pair_vid_off_us in EVERY band when
a crossing stream has not yet applied the event's offset; the sub-band
mismatch is REGISTERED for the content path (ptv_pair_expect) down to the
AGLUE engagement floor (g_aglue_ms — below it the glue never examines the
step and aresample=async soft-converges it unregistered; a registered
micro-step could mis-consume an unrelated later step within the TTL).
Same-cycle 2b band path untouched — both-crossed single-flush in-band
behavior stays byte-identical (the TruBLU mandate); 3b/3c/2d unchanged.
Gates (A/B pre-fix vs pre6, live-UDP dg-run + silencedetect landmark +
xcorr oracle): fx-splitband (V+6.02@115 / A+6.22@117.5, offsets differ
220ms in-band, cycles 2.4s apart) — pre5 baked −200ms (landmark: the
source's +200ms audio-later shift erased), pre6 +20ms (inherit −6.000,
+220ms registered, AGLUE "matches the shared-flush expected step —
APPLIED", landmark shows audio +220ms); fx-wc520 (interleave-lag split,
452ms in-band) — pre5 −440ms added, pre6 +40ms; tb900/tb30 TruBLU split
rewinds — vid/aud offsets agree exactly, applied values and log lines
identical pre-fix vs pre6; fx-sym flush 1 (same-cycle band) byte-identical,
flush 2 (split leg, copy-only AC-3) now carries the event offset −15.144;
regression battery mir2/dbl/sym/tb900/tb30/aonly/b300 all at pre5 oracle
numbers; 20.5-min clean-source soak zero flushes, stats flat.

(7c) LAYERA PARTNER-CROSSING DETECT — the escaped-step defect (pre6; demux
loop). Live evidence (cor-3, 2026-07-14): JLTV — video +6.033s jump detected,
video-only partial flush (applied −6.000); the partner AUDIO step (+6.5s)
arriving at the window close NEVER produced a jump line — it escaped the
discontinuity layer entirely, fell through AGLUE's >1000ms cap ("left to the
discontinuity layer", which never took it) and hit aresample as a raw ~8s
input-pts hard pad; measured output video−audio label spread +42.9s.
Azorse_TV same day, same shape (V −39.4s back / A +7.1s fwd → +467ms
audio-early). Mechanism, fixture-reproduced (fx-wclose = V−6.02s@115 /
A+6.02s@115.35 over live UDP — the exact live signature: one jump line,
"partial flush (only video crossed)", "[PTV-AGLUE] +6596ms above 1000ms cap",
no second jump line, −6.0s added desync by silencedetect landmark
accounting): a >1s jump on a SECOND stream arriving while the disc buffer is
ACTIVE gets no detection (the !b->active gate) and is left to classification,
which borrows the first stream's bases; on a mirror/asymmetric event the
partner's post-jump position lands NEARER THE OLD base → its stepped packets
are misclassified OLD and DELETED, and the per-packet continuity-ref update
(last_dts_us) has already advanced onto the stepped timeline → post-flush the
step is invisible to detection FOREVER. FIX = while the buffer is active, a
dense stream with no new base whose OWN delta exceeds the 1s threshold AND
whose borrowed classification would NOT tag it NEW gets its own bases
recorded via ptv_disc_detect_jump — its stepped packets then classify NEW,
the stream transitions, and the flush handles the event as both-crossed
(tree 2b: video defines the timeline, the mismatch registered for the audio
content path). Gated on the would-be misclassification, so every ordering
classification already handles (TruBLU symmetric rewinds, forward-forward
pairs — δ-swept 350/400/420/435/450/520ms incl. AAC-audio variants, all
detected) keeps byte-identical behavior AND log lines. fx-wclose after the
fix: jump-on-stream-1 line, single both-crossed flush (+6.040), expected
step +12.492s registered and AGLUE-matched, silencedetect-verified 12.492s
pad, added desync −20ms (source post-event shift S=+12.04s carried exactly).

(7) LAYERA SHARED FLUSH — asymmetric-event invariant fix (pre4;
ptv_disc_flush). THE INVARIANT (owner-mandated): "after any input event, the
output's A/V alignment must equal the source's post-event alignment. Latency
may be retained; relative A/V offset may never be." A stateless player gets
this for free; LAYERA's stateful per-flush erase preserved it only when video
and audio jumped by the same amount in the same flush cycle. Live evidence
(2026-07-13 01:01, Curiosity provider playout jump, two boxes at the same
minute): Curiosity_Now/cor-1 — video jumped +14.181s, flush
applied_offset=-14.148s (vid only); 0.6s later audio jumped +15.176s (past
the 500ms buffer timeout, so the events did NOT pair), second flush
applied_offset=-15.155s (aud only) — each stream erased its OWN jump and the
~1.0s A-vs-V jump difference was FROZEN into the output for 8.5h,
viewer-visible desync with every counter clean.
PATRIOT-Curiosity_Channel/cor-2, same minute: audio +15.864s FORWARD, video
−15.811s BACKWARD; flushes applied aud −15.832 / vid 0.000. Design: dense
flushes within PTV_PAIR_WINDOW_US (5s) are ONE source event and every dense
stream gets the event's VIDEO-derived offset (video is the house-clock
anchor; prog_off/SCTE ride the video timeline) — one offset can never change
relative A/V alignment. The A-vs-V jump difference is NOT erased: it
surfaces as an audio label step routed to the existing CONTENT machinery
(pre5 correction — the pre4 text claimed "AGLUE gap-pad within its 1000ms
cap", which was WRONG for backward steps: AGLUE relabel-ERASED those, the D1
defect below; since pre5 the flush registers the step so AGLUE APPLIES it in
every direction — forward = aresample pads, backward = aresample drops,
above the AGLUE cap = aresample=async hard pad/drop — bounded convergent,
never another per-stream erase). Pairing
covers all three orderings via a decision tree (documented at
ptv_disc_flush): video-first (Curiosity — audio-only flush INHERITS the
event's video offset), audio-first (PATRIOT class — audio flushes with a
provisional own offset and is RETRO-CORRECTED onto the video timeline when
video's flush arrives in the window), and same-cycle full flushes (audio
shares video's applied offset directly). New [PTV-GLUE] "paired flush"
ledger line logs the mismatch the shared offset avoided baking in
(shared_offset= / av_mismatch= / -> audio content path), emitted exactly when
the shared offset overrode a stream's own butt-joint. TruBLU-equivalence
guarantee: offset disagreement ≤ PTV_PAIR_EPS_US (500ms) is flush BOOKKEEPING
(duration-estimate overhang ~20ms, trailing-OLD discard holes ~100-400ms of
interleave), not a source A-vs-V jump difference — in that band the
production-proven audio-preferred butt-joint is kept byte-identical (equal
deltas ⇒ per-stream-identical behavior, log-line equality; the invariant
holds in the band regardless, since ONE offset is applied to all dense
streams either way — the band only picks which content machinery absorbs the
bookkeeping residual). Above the band (live cases: 1.0s, 30.8s) video defines
the timeline. Sparse SUBTITLE/DATA/SCTE-35 keep the prog_off path untouched;
scte35_rebase_pts_adjustment byte-identical. PTV_NO_SHARED_FLUSH=1 reverts
to the per-stream erase. Gates (A/B vs pre3): invariant fixture (V+14s/A+15s
0.6s apart, live-UDP single-input encode across the jump, pts-origin-corrected
xcorr oracle) — pre3 bakes ≈+1s, pre4 ≤±60ms added desync; PATRIOT variant
(A+15s fwd / V−15.8s back) same bar; audio-first ordering same bar; symmetric
+15s and TruBLU −900s/−30s rewinds — LAYERA jump/flush lines and hs/hsres
log-equivalent to pre3, added desync ≈0 both binaries; pts-spacing flatness
<1µs/frame outside the event bucket; item2 count+genuine at pre3 numbers;
tier-1 = the known legacy fails only.

(7b) SHARED-FLUSH REVIEW-ROUND FIXES (pre5; ptv_disc_flush + audio_feed) —
an adversarial review of pre4 CONFIRMED three defects empirically; all three
fixed and re-gated A/B vs pre3:
  D1 — mirror-signed mismatch erased by AGLUE (invariant still broken for
  av_mismatch in (−1000ms,−500ms), video jumping FURTHER forward than audio):
  the routed backward label step hit AGLUE's 0.9.16.4 relabel-ERASE and landed
  audio on the pre3 butt-joint while the "paired flush" ledger claimed
  success (fx-mir2: V+15.0s@125.3 / A+14.2s@126.1 → pre4 == pre3 == +760ms
  baked). FIX = a demux→audio EXPECTED-STEP handshake: when a shared flush
  routes a mismatch it REGISTERS the per-track step (value + 10s wall
  deadline; _Atomic pair in Input, demux writes value-then-deadline release,
  audio thread acquires — the house_skew/disturb_epoch publish idiom) via
  ptv_pair_expect(); AGLUE consumes a matching arriving step (asymmetric
  window [−250ms,+500ms]: flush-borne steps arrive exact to −32ms, a
  retro-corrected step merges with the flush's forward discard hole, +456ms
  measured) as a REAL alignment step — APPLIED, logged
  "matches the shared-flush expected step ... APPLIED", never erased.
  Unregistered (plain source) backward steps keep the erase rule
  byte-identical; forward-path behavior unchanged (annotation only).
  D2 — re-inherit hole, WORSE than pre3: decision 3a inherited
  pair_vid_off_us even when the stream had ALREADY applied the event's offset
  (pair_has recorded but never consulted; same hole reachable through 2d).
  fx-dbl (mirror event + independent audio −2s wobble 2s later): pre4
  inherited −14.980s for the wobble → −14.8s desync, ~17s of audio destroyed
  (pre3: clean +2.0s butt-joint). FIX = pair_has is now CONSULTED: a crossing
  set whose streams all already applied is a NEW INDEPENDENT event (3c —
  plain butt-joint, pair state untouched); 2d retro-corrects only
  pair_prov (provisional) streams, never finalized ones; and the pairing
  window CLOSES as soon as video has defined the offset and every flowing
  dense audio stream has applied it (completion close). The header's "false
  pairing is benign by construction" claim was falsified and rewritten to the
  actual guarantee: each stream applies an event's offset AT MOST ONCE per
  window and a completed event cannot be paired with; the inherently
  ambiguous case (video-only crossing + a first independent audio crossing
  inside the 5s window) remains indistinguishable from the genuine Curiosity
  ordering, and a false inherit there costs an audible aresample convergence
  proportional to the disagreement.
  D3 — prog_off double-bump on split-cycle events (pre-existing, exposed by
  the split orderings): the flush persist added applied_offset to prog_off on
  EVERY nonzero flush, including the audio-only inherit flush → SCTE-35/
  DVB-sub timing moved ~2× the offset on split events. FIX = prog_off is
  gated on has_vid (same gate as disc_resid_us): sparse rides the VIDEO/
  program timeline and moves exactly once, at video's own flush; an
  audio-only UNPAIRED event moves no video labels and now moves no sparse
  timing either (the pre-pre4 audio-only bump was the same latent bug —
  deliberately fixed unconditionally, i.e. also under PTV_NO_SHARED_FLUSH).
  KNOWN BOUND (copy-only audio): a copy-passthrough audio stream (e.g. AC-3
  -c copy) has no content machinery — a large BACKWARD routed correction on
  such a stream compresses into the demux_pass monotonic-DTS clamp until real
  time catches up (labels held, content squeezed against the clamp). Dense
  transcoded tracks (the production posture for the primary audio) are
  unaffected; documented as a bound, not fixed.
  GATE-2b EQUIVALENCE, stated precisely: for symmetric events (TruBLU +15s /
  −30s / −900s classes) pre5's APPLIED OFFSETS are byte-equal to pre3's
  ([PTV-LAYERA] flush applied_offset= values); buffered-PACKET COUNTS in the
  flush lines may differ by a few packets (wall-clock window jitter between
  runs), which is run noise, not behavior. Re-gate results (A/B vs pre3
  79e941d313, committed pre5 binary): see the commit message of the pre5
  commit for the full matrix (fx-mir2 D1, fx-dbl D2, SCTE split-event D3,
  gate-1/PATRIOT reproduction, audio-only event, plain −300ms backward step
  erase intact, spacing/item2/tier-1).

(6) TRACK STEERS THROUGH THE RESAMPLER, NEVER LABELS (pre3; audio_drain_fg +
audio_feed). Production proof (2026-07-13, live grids): the mv audio-follow
PLL's TRACK integrator actuated by RE-STAMPING output labels (af_applied_us
moved `want = opts + applied` every frame). The PCM stayed byte-clean, but the
output AAC pts spacing stretched by +19..+158 ms/min during integration
episodes — and PTS-honoring players chase that drift with their own rate
correctors = clearly audible warble ("guitar-effect distortion",
owner-confirmed at the exact drift-episode timestamps). rc builds (no TRACK)
measure EXACTLY 21.333ms spacing forever; pre2 with PTV_NO_PLL_TRACKUP=1
(TRACK dead) is also flat and owner-confirmed clean. Conclusion: label
re-stamping is a FORBIDDEN actuator. pre3 ports the single-input AVLOCK
mechanism to the mv TRACK: the rate-clamped integral trim accumulates into a
per-track af_steer_us added to the pts of frames FED INTO the -af graph, so
aresample=async realizes it as bounded sample insert/drop (10ms/s clamp, well
under async=1000's ~20.8ms/s soft authority and far under min_hard_comp —
inaudible), while output stamping stays `want = opts + af_applied_us` with
af_applied_us changing ONLY at ACQUIREs (discrete, logged, rare): between
acquires the label stream is perfectly uniform. The measurement loop removes
the injected steer from the sink pts before vring pairing (like single-input
removes house_skew), so it reads the TRUE post-steer offset;
d(offset)/d(steer) = −1 (documented at the pairing site). The pre2
[PTV-TRACKUP] direction-aware anti-windup is retired with the label actuator
(TRACK no longer touches `want`, so the monotonic-guard pin cannot eat the
integrator). PTV_NO_PLL_TRACKUP=1 now disables the steer-TRACK entirely
(acquire-only, labels flat, zero steer) — the grids' current production mute
keeps its exact meaning. (B) vlag bias audit — NO centering change shipped: the
vring writer records first-display at the tick boundary (compositor
`vring_put(..., mv_tick_us(c, tick))`), but the residence pop gate already
centers display quantization around the due schedule (a frame pops at the
first tick within HALF a tick of its due time: `hnow + half < res_due` holds
it back, so display−due ∈ [−T/2, +T/2)), and h0-at-display zeroes the anchor
frame's vlag exactly. Empirically (3 identical TRACK-muted clean 2x1 runs):
the static offset is run-dependent (+16 / +50 / +60-70 ms), not a constant
+half-tick — it is anchor-phase + buffer-depth DC, i.e. REAL display latency
the audio should (and now does) follow, plus slow ±1-tick quantization wander
that the ACQUIRE tick floor (1.5 ticks) rejects and the steer trims as a
bounded, sub-perceptual rate. (C)
[PTV-ACOMP] visibility: always-on app-layer proxy for swr hard-compensation
triggers — a graph-input pts step >25ms (post-AGLUE, post-AVLOCK/steer: the
stream the resampler actually sees) logs `[PTV-ACOMP] aN(inN) input pts step
+Xms — swr hard compensation likely (click risk)`, rate-limited ~1/10s per
track, with a cumulative per-track counter surfaced as acomp= on the PLL diag
line. Gates (A/B vs pre2, 2x1 jitter grid): pts-spacing flatness 20min — pre3
worst 30s-bucket mean dev +0.0005us/frame (startup bucket +0.95us from one
+1.4ms anchor-settle step, present in both builds) with the steer demonstrably
active (a1 steer ramped to +1.2s following the stall-jitter vlag); pre2 =
18/40 buckets on a0 and 40/40 on a1 over the 5us bar, worst +131us/frame ≈
+369ms/min stretch — the production warble signature reproduced and
eliminated. Shifted-source (+300ms): converges via ACQUIRE + steer trim,
output labels flat (worst bucket +0.95us), 0 clicks in the quiet-block scan
(pre2 same fixture: 4 alternating pad/drop ACQUIREs + label stretch to
+69us/frame). item2 count 300s: 0 ACQUIREs (bar ≤9, no alternation). item2
genuine: 8s-outage bank captured in 2 ACQUIREs, flashbeep tail median +0.3ms.
tier-1 = the 4 known legacy fails only. ACOMP fixture (+50ms mid-stream audio
pts step): fires exactly once at the step, silent on the clean sibling track
and on clean smooth runs (the only organic firings were real AGLUE GAP
loop-seam pads — correct).

(5) [PTV-TRACKUP] DIRECTION-AWARE TRACK ANTI-WINDUP (audio_drain_fg, B3 TRACK
branch): the 1.0.1-pre1 grid soak showed item (2)'s dead-band killed the 42ms
quantization-snap class but NOT the acquire rate (~41-44/h, strict pad/drop
alternation at ±63ms, `applied` toggling one step forever). Local repro (2x1
grid, PTV_AVSYNC_PROBE) localized the real cause upstream of the dead-band:
the TRACK integrator's symmetric N3 anti-windup made the monotonic-guard pin
an ABSORBING state — the first advance-direction step leaves `want` a
sub-frame amount under the dense line af_last_out+nb, the deficit never decays
(both sides advance at the same rate; measured guard +1/frame, 51k fires in
20min), and TRACK is dead for the rest of the run, so every correction fell to
whole-frame ACQUIREs whose ≤½-frame residual ±1-tick vlag quantization
re-crossed the threshold from the opposite side = the limit cycle. Fix: allow
POSITIVE (delay-direction) steps while pinned — they repay the label deficit
and re-close the loop; negative steps stay blocked (true windup; advancing
audio = content drops = ACQUIRE's job). PTV_NO_PLL_TRACKUP=1 reverts. Gates:
jitter fixture 51→8 ACQUIREs/20min (falsification: fix binary + revert env →
54, causal); item2 count suite 9→0; shifted-source flashbeep tail
+1848→+954ms; tier-1 = the 4 known legacy fails only. Soak note: acquire
counts on disturbance-heavy channels may legitimately RISE (acquires now do
real work); watch the alternation signature and guard growth (+1/frame = the
bug, bounded bursts + plateaus = healthy). Known asymmetry by design:
audio-early errors converge ~0 via TRACKUP; audio-late residuals can park in
(−60ms, 0] (below the acquire tick floor, advance-TRACK blocked by the pin) —
a stable −40ms on [PTV-AVSYNC] is the tradeoff, not a regression.

(4) REANCHOR2 DEBOUNCE (compositor_thread): the P2 h0-re-anchor's shift
(−sk + tick) was computed from ONE instantaneous displayed-frame label, so a
single corrupt PTS — one frame with a mangled PTS but intact DTS sails through
the demux discontinuity layer (which watches DTS) — inflated the shift by its
full excursion and displaced the whole slot's h0 (transient audio-early until
the PLL healed). Now each slot keeps a 5-deep ring of its evaluated sk samples:
fire only when ≥3 of the last 5 evaluated ticks (including the current one)
held sk < −threshold, and size the shift from the MEDIAN of the qualifying
samples; the ring re-debounces after each fire. A one-tick corrupt label is
1-of-5 → ignored; a genuine video-ahead excursion persists across ticks and
still re-anchors within 5 ticks. PTV_REANCHOR2_INSTANT=1 reverts to the
single-sample fire. Gate: 2x1 live mv with one +2s-PTS-corrupted video frame
(DTS intact) in in0 — rc fires "[PTV-REANCHOR2] in0 tick=971 video-ahead → h0
+2000ms" on the single frame; fixed ignores it (0 fires); genuine class
(file-input unpaced mv, decoder outruns the house clock) still fires on both
builds (rc 32+6-suppressed, fixed re-anchors within 5 ticks of each
excursion); tier-1 identical.

(3) ANCHOR HEAD-FILL (audio_anchor_and_feed): when the source's audio HEAD is
missing at birth — first kept audio starts >200ms after h0, or the pre-h0 ring
overflowed (kept head ≠ true head) — the output track's first packet sat at
PTS = first_audio−h0: PTS-coherent but first-packet-MISALIGNED for naive
consumers (RAV mv 2026-07-07: +2058ms, the suspected app-visible audio-early).
Now the anchor synthesizes silence covering house 0 → first_audio−h0 in the
SOURCE domain (first kept frame's rate/layout/format, labels stepping
seamlessly into the real head) and pushes it through the normal feed path, so
the encoder emits audio from ~PTS 0 and the graph/PLL/gates see an ordinary
continuous track. Capped at the pre-h0 ring's own time span (aq_prehold ×
frame-dur, ~5.5s default); one [PTV-ANCHOR] headfill line. next_pts and
dbg_first_src are seeded at the FILL start so the swr path and the async-pad
span accounting stay coherent. PTV_NO_ANCHOR_HEADFILL=1 reverts. Gate: source
with the audio PID stripped for its first 3s — rc: output first audio packet
+2.699s after first video (ANCHOR first_audio-h0=+2720ms); fixed: first audio
pkt 0.000 vs first video 0.011 ("headfill 2720ms silence, 128 frames, cap
5461ms" logged) and the REAL content still starts at 2.732s (silencedetect) vs
rc's 2.699s first packet = content↔PTS mapping unchanged within one frame
quantum; PTV_NO_ANCHOR_HEADFILL=1 reproduces the rc birth exactly (+2.699s
head gap, no headfill line). Tier-1 identical.

(2) MV PLL ACQUIRE TICK-QUANTIZATION DEAD-BAND (audio_drain_fg B3 controller):
the vlag half of the measured offset is quantized to the house video tick (the
vring records first-display output times on tick boundaries), so at 25fps the
measurement quantum (40ms) EQUALS the base acquire threshold — the PLL
hard-snapped on its own quantization noise, alternating ±42ms pad/drop every
12-60s per slot (live grids on the rc run: ~939-1511 ACQUIREs per 22h). Fix:
(a) acquire threshold floored at 1.5× the house tick (AudioState.tick_dur_us,
wired at setup from out_fps — the house tick is global across slots, the same
axis the compositor measures every slot's vlag on), so a ±1-tick reading can
never clear it; (b) SUSTAINED-OFFSET requirement — fire only after the |EMA|
stayed above threshold for 3 CONSECUTIVE completed debounce windows (~2s
continuously large; the window counter resets the moment |ema| falls under
the threshold). TRACK path untouched. PTV_ACQ_INSTANT=1 reverts to the
single-window fire (the tick floor stays). Gate: 2x1 mv fixture, in1 through
stall_send jitter (0.7s stall / 5s), 600s — see commit/report for rc-vs-fixed
ACQUIRE counts; genuine-offset check (in0 sender killed 8s mid-run, real bank
forms) still converges and the flash+beep tail reads aligned; tier-1 identical.

(1) AUDIO DECODE-DEATH TOLERANCE + WATCHDOG [PTV-ADEC]/[PTV-ADECWD] (branch
audio-batch): a hard `avcodec_receive_frame` error in audio_thread was `goto done`
= silent permanent track death (Pure Flix 2026-07-08: ONE corrupt-PCE AAC event
killed the track for 14h; video survives identical storms via concealment).
Now: (a) hard decode errors are TOLERATED unconditionally — dropped + counted
(dec_errs) + AGLUE-style rate-limited WARNING (10 lines/10s + rolling summary);
send_packet hard errors (where eager decode surfaces most single-frame AAC
errors — previously swallowed silently) share the same counter/log. (b) decode-
death WATCHDOG: packets arriving but ZERO decoded frames for 45s wall → reopen
the decoder from ist->codecpar exactly as transcode() setup built it (swap only
on successful open, retry next window on failure). Anchor/pts_set state is
PRESERVED — mid-run recovery, the track's timeline continues, aresample absorbs
the dead span like a source gap; if the reopened decoder emits different params
the [PTV-AFMT] hysteresis+rebuild reconfigures downstream. PTV_NO_ADECWD=1
disables the watchdog only. Also: audio_push now DROPS a decoded frame with no
timestamp (garbage-tail decode next to a tolerated error) — un-stamped frames
cannot be content-anchored and previously rode through the graph into a
timestamp-less packet = mpegts non-monotonic-DTS EINVAL = mux-thread death =
whole-rung wedge (measured on the first fixed-build gate run; rc never hit it
only because the thread died at the first error). Gates (local UDP fixtures,
cinestar-AAC 100s loop): (1) death: 6s glue-corrupt ADTS window (valid frame +
garbage tail per packet → errors surface at receive_frame) — rc: audio 469
pkts/10s until t≈26 then 0 FOREVER, zero log lines; fixed: 468-469 pkts/10s
through the window and to end-of-run, 10 [PTV-ADEC] lines (rate-limit cap).
(2) watchdog: 55s all-frames-corrupt window (errors at send, zero frames) —
exactly one "[PTV-ADECWD] no decoded frames for 45s with packets arriving —
decoder reopened (#1, errs=2114)", audio continuous; with PTV_NO_ADECWD=1 the
same run logs ZERO ADECWD lines (revert verified). (3) AGLUE regression:
audio-only −300ms label step @32s — buckets/verdicts identical rc vs fixed
(the step is absorbed at the demux layer per 0.9.18.5; no AGLUE verdict on
either build). Tier-1 fail set identical to baseline.

## 1.0-rc1 (2026-07-10) — file split (movement-only decomposition of ptvencoder.c)

No behavior change — pure code movement (v1-cleanup-plan §7). The ~6.5k-line
`fftools/ptvencoder.c` monolith is decomposed by thread-ownership domain into:
`ptvencoder.h` (shared types + extern decls + cross-file prototypes),
`ptvencoder_clock.c` (house clock / output_thread ladder, encode_push, the
house-rate-correction ladder, content_index, ptv_empty_watch),
`ptvencoder_demux.c` (demux thread, demux_unwrap, LAYERA ptv_disc_*, DUKF,
SCTE-35 rebase, rate_estimator_feed, demux_dispatch), `ptvencoder_audio.c`
(audio thread: feed/anchor/drain, AGLUE, AFMT rebuild, audio-follow/B3 PLL,
build_audio_filter), `ptvencoder_mv.c` (compositor thread + slate/black
helpers), `ptvencoder_gate.c` (delivery gate dlv_*, cushion_escalate/CushionRt,
resolve_cushions, push_frame_q, watchdog_thread), `ptvencoder_legend.c` (help
text + log legend); `ptvencoder.c` keeps main()/env parsing, decode_thread +
filter-graph builders, transcode() wiring, mux_thread, plan resolution and the
vring probe helpers. Every line moved verbatim; the only code deltas are
`static ` keyword removals on symbols that became cross-file (extern'd in
ptvencoder.h) and the removal of the now-redundant in-file build_audio_filter
forward declaration. Build wires the new objects via `OBJS-ptvencoder` in
fftools/Makefile. Pre-split state is tagged `v0.9.18.7-monolith`.

## 0.9.18.7 (2026-07-09) — env tiering (21 internalized) + hs/hsres split + log promotions

Logging/config-surface only — byte-identical control behavior is the gate; no control
path reads anything new.

(1) ENV TIERING (implementation-map PART 3): the 21 single-read debug envs that never
moved off their defaults in production are internalized — getenv deleted, value frozen
at the production default, static kept with an "internalized 0.9.18.7 (was PTV_X)"
comment. Frozen: GENLOCK_MAX_PPM=300ppm(Q20 314) / GENLOCK_REJECT_PPM=700ppm(Q20 734;
the reject>=2*max invariant now holds statically, 734>=628) / GENLOCK_WINDOW_MS=3000 /
GENLOCK_EMA_SHIFT=6 / GAP_MIN_MS=700 / AGLUE_MAX_MS=1000 / DISCONT_MS=1000 /
DISCONT_BACK_MS=80 / PROGOFF_DEBOUNCE_MS=1000 / DUKF_ESCAPE_MS=5000 / DUKF_MIN_MS=1000 /
H0_REANCHOR_MS=120 / AF_ACQUIRE_MS=100 / AF_RATE_MS_S=10 / PLL_EMA_SHIFT=7 /
PLL_TAU_MS=5000 / PLL_ACQUIRE_MS=40 / PLL_ACQUIRE_N=32 / PLL_REFRACTORY_MS=12000 /
PLL_NOISE_K=3 / PLL_DEV_SHIFT=9. Setting any of them is now a silent no-op (channel
configs that still export them are harmless by construction). NOT touched: all PTV_NO_*
reverts, PTV_DIAG/PTV_LOG_TS, the 9 tuning knobs, compat no-ops, injectors
(PTV_PLL_TESTNOISE_MS/PTV_WRAP_GUARD_S/PTV_SLOW_US/PTV_BANK_DECAY_S/PTV_AVSYNC_PROBE),
PTV_AGLUE_MS, PTV_NVENC_SERIALIZE/REG_CAP, and the post-map reverts
PTV_LAYERA_FULLSKIP / PTV_NO_CADDISARM.

(2) hs/hsres SPLIT (reporting only): hs= conflated real retained latency with
erased-discontinuity bookkeeping — a LAYERA jump-to-live erase makes the post-glue
labels continuous, so the stall's dup-ratcheted skew never re-syncs and parks in hs
permanently (In-Touch analysis). New ledger DemuxArgs.disc_resid_us =
Σ(−applied_offset) over ptv_disc_flush erases that shifted the VIDEO label stream
(has_vid only; audio-only partial glues excluded). Printed as an ADJACENT field
(`hs=+Xms hsres=+Yms`), not as an hs sub-component: every erase shifts the hs reading
by exactly −applied_offset vs raw labels, but "hs−hsres = real hold" is only valid for
wall-tracking label streams (a label-repeating source, e.g. a loop seam, moves hsres
without any hold change), so presenting Y inside hs would mislead. Read it by TREND:
hs growing in step with hsres = erased-discontinuity residue; hs growing with hsres
flat = real retained latency. hs itself is UNTOUCHED (same expression, same consumers —
AVLOCK/copy-path still read *house_skew directly). Multiview gets the same treatment:
per-slot sk= rides the same erased label stream (sk = mv_tick − (disp_src − h0)), so
the slot line gains /skres= from the same per-input ledger.

(3) LOG PROMOTIONS (rare events, PTV_DIAG→always-on WARNING): [PTV-REANCHOR2] (mv h0
re-anchor) and [PTV-PLL] ACQUIRE (bank snap drop/pad — already hard rate-limited by
the 12s post-acquire refractory, ≤1/12s per track by construction; no extra limit
added). Message text unchanged. REANCHOR2 got the AGLUE-style rate limit (4 lines/10s
per slot + suppressed-count summary): MEASURED on a file-source (unpaced) mv smoke it
refires every tick (~30 lines/s, decoder outruns the house clock and each +1-tick
landing is re-passed immediately) — live mv is paced so it stays rare there, but the
promotion must not be able to flood; re-anchors still apply, only lines are capped.

Legend: hsres=/skres= entries + internalized-envs note added; Reverts list unchanged
(nothing on it was internalized). Docs: analysis/ptvencoder-usage.md env section notes
the internalization.

Gate: build clean, -version 0.9.18.7 @54b760ba5e; tier-1 fail set identical to
baseline {t0_smoke, t1_audio_discont, t1_cc_extraction, t1_hw_reconfig}; 120s live UDP
mirror A/B vs 0.9.18.6 (cinestar loop, x264 1-rung) — trajectories equivalent (async
EMA decay −132→−38 new vs −131→−37 old-with-envs, dup=0, rho zero-mean, dlvhold
~2.6s band), hsres= present; env-freeze A/B (old binary WITH PTV_DISCONT_MS=1000
PTV_GENLOCK_WINDOW_MS=3000 PTV_PLL_TAU_MS=5000 vs new with no env) — identical
trajectories; seam fixture (tsp --infinite native-PTS 30s sample → −29.984s
backward jump per seam; NOTE: ffmpeg -stream_loop adds file duration per loop =
CONTINUOUS ts, so send.sh alone produces NO seams) — new binary: 5 glues, hsres steps
exactly −30016ms per glue (= −applied_offset), hs=+0 flat, dup=0; old binary same
feed: hs=+0 flat = accounting untouched. mv smoke: skres= prints per slot; REANCHOR2
limiter caps the file-mv flood (71 lines vs ~30/s unbounded, "146 more suppressed"
summaries roll correctly).

## 0.9.18.6 (2026-07-08) — R3+R4: estimator/house-rate state into per-input/per-house structs

Pure-motion refactor (no behavior change intended; this bump exists so the soak build is
banner-distinguishable from 0.9.18.5). RateEstimator (incl. the former g_src_rate_*/g_cf_*
atomics) now lives per-input in the Input struct; HouseRateState (rho servo, occ EMA,
reprime state) is owned at transcode() scope and shared by the rung set via VideoCtx.hr —
the same pattern as h0/house_skew. house_rate_corr_ppm() gains const RateEstimator*.
All _Atomic types, orderings, seeds, and formulas moved verbatim; multiview semantics
unchanged (estimator is only fed in single-input live mode). Closes the last file-split
prerequisite.

Gate: tier-1 fail-set identical; WUCR A/B rho trajectory + cf-lock schedule identical;
escalation fixture event-sequence identical; deep-preroll fqhw plateau identical;
adversarial review COMMIT-READY (byte-identical substantiated at every access site).
Live gate = 24h cor-3/live-transcoder soak (cross-thread atomics moved — the one class
where fixtures are weakest; watch wucr_rho/cf/async parity vs 0.9.18.5 baselines).

## 0.9.18.5 (2026-07-07) — sub-1s "no-owner band": shared-amount absorber owns it under LAYERA

Fixes the In-Touch_+ audio-late accumulator (+620ms/3h, +1477ms/26h measured live; full
root-cause: analysis/ptvencoder-intouch-desync-analysis.md). Under g_layera the demux
absorber was skipped for ALL super-threshold (>80ms) jumps, but LAYERA itself only claims
>1s — so a both-stream backward step in the 80ms..1s band had NO packet-layer owner:
video became house_skew ratchet/decimation while AGLUE RELABEL-erased audio, and AVLOCK
re-injected the video conversion into audio → the same source event actuated TWICE on
audio = audio permanently late ~step-size per event. The skip is now scoped to jumps
LAYERA will actually claim (>PTV_DISC_THRESHOLD_US, same comparison as
ptv_disc_detect_jump); sub-1s falls through to the proven §5.A.2 shared-amount absorber,
which erases the step identically on both streams before house clock/AGLUE/aresample see
it. PTV_LAYERA_FULLSKIP=1 restores the old posture (A/B / rollback).

Fold-ins: (a) ptv_disc_flush shifts stream_state[].last_dts_us by applied_offset — kills
the phantom re-detect/flush (applied_offset=-0.000s + ~50ms hold) after every glue;
(b) video_fwd_us stamped before the LAYERA skip so the audio gap discriminator's
vcrossed signal is truthful under g_layera. Adversarial-review disclosures: (b) is
outcome-invariant coupling, not zero coupling — when video crosses first in a
whole-program splice, a later >1s audio forward jump's is_gap flips 1->0, but both
routes end at absorb_done unabsorbed and LAYERA claims the glue either way (residual =
log line + disturb_epoch bump source). PTV_LAYERA_FULLSKIP is NOT byte-equivalent to
0.9.18.4: both fold-ins stay active under it (hygiene-only; the A/B arm restores the
accumulator mechanism faithfully, but do not expect the phantom -0.000s flush lines to
reappear). And sub-1s backward events now bump disturb_epoch (PLL re-acquire) via the
restored absorber body — the proven g_layera=0 posture, previously silent under LAYERA.

Gate: F1 fixture (both-stream −300ms steps, flash+beep ruler) — HEAD shows the exact
+301ms/event staircase, fixed build flat [−32,+30]ms; F4 (3× ~6s real wall stalls) —
≥1s LAYERA path byte-identical per-event deltas vs HEAD, one flush per event (HEAD: 2,
phantom); AGLUE regression (audio-only −300ms) — absorber owns it on the new default,
RELABEL fires unchanged under FULLSKIP; regression tier-1 set identical.

## 0.9.18.4 (2026-07-07) — M6: one video packet rate, derived from out_fps

The seconds->video_q-packets conversions used three constants (bank 35/s, deep-preroll
side-car 60/s, deep-prime out_fps). Now ONE rate: CushionPlan.vid_pps = ceil(out_fps).
Audio sizing keeps its own 50/s — that is an AAC frame rate (48kHz/1024 ~= 47/s), not a
video rate, intentionally not unified.

MEASURED CORRECTION to the plan's premise (59.94fps fixture, A/B vs pre-M6): the old bank
packet count is consumed as a BOOLEAN (arms blocking-push), so the 35/s value was inert;
and default capacity (videoq 512 + frame_q) already holds ~11.2s @59.94 — both builds
retained a 9.2s bank fine. The real fixes here: (a) the [PTV-BURSTY] advisor recipe now
recommends a correct PTV_VIDEOQ for fast channels (was ~42% under at 59.94); (b) live
channels' video_q is auto-sized to hold the FULL bank ceiling (closes the one exposed
corner: 59.94fps at the 12s ceiling needed ~720+ pkts vs 512+frame_q ~= 11.2s); (c) the
deep-preroll side-car uses the exact rate (was pessimistic 60/s — now mutually consistent
with deep_prime's own sizing).

Gate: 59.94 escalation fixture — new build retains 12.1s actual >= 9.2s target with the
ceiling-sized queue, overshoot-retention drains at ppm scale as designed; regression
tier-1 set identical; 29.97/25fps sizing arithmetic unchanged by construction
(ceil(out_fps) reproduces the old effective numbers there).

## 0.9.18.3 (2026-07-07) — M5: cushion tier changes reach the live delivery gates

The adaptive GROW/SHRINK arms raised/restored only the GLOBAL delivery cap; the per-rung
gates latched their cap at init, so a RAISED-no-bank channel (5 of 56 on cor-3 at audit
time) kept its gate at the LEAN cap — on a real video-encoder wedge it force-released held
audio ~3s early (plan §3.5; the last of the "one queue forgot" class). GROW/SHRINK now run
the same live gate-cap write the BANK arms have always done (live base + armed bank margin),
inside cushion_escalate()'s mutex. New PTV_DIAG line [PTV-GATE] logs every gate-cap rewrite
with its arithmetic.

Gate: fixture shows the write composing correctly under GROW-with-armed-BANK
(caps -> 15.3s = base 6.0 + bank 9.3 on the live gate); escalation-sequence fixture
byte-identical to M3; the cap-respecting RELEASE path is unchanged code production-proven
by every BANK escalation. A true encoder-only wedge is not reproducible host-side
(process freeze ages all clocks together and the resumed encoder clears the stall before
a drain observes it) — release-timing delta is expected to show as absent early
dlvforced on wedges of RAISED channels in the fleet.

## 0.9.18.2 (2026-07-07) — M4: multiview primes with the resolved preroll

The compositor re-read PTV_PREROLL_MS with a 350ms fallback that predated v0.9.1's genlock
default (1000ms) — plain mv invocations primed with a third of the intended cushion (plan
§3.6, the last un-resolved cushion consumer). Now reads CushionPlan's resolved value;
explicit PTV_PREROLL_MS wins exactly as before.

Fixture A/B (2x2 grid, 4 live UDP 25fps inputs, x264): old = one slot 5 startup starvation
dups + slots born misaligned (sk -74/-53ms, permanent, and per-slot audio follows it);
new = sv=0 all slots, sk=+0 all slots, dlvhold ~1.6->2.4s (deeper hold, by design).
Composite dup= reads higher purely because aligned slots hold on the SAME tick (uniform
cadence repeat) instead of staggered — slot-level content is identical-or-better.

## 0.9.18.1 (2026-07-06) — M7: cadence-evidence pulldown disarm (AWE clicking)

Measured (test-results/pulldown-trap/ + cor-1/cor-3 log correlation): AWE's encoder emits
unreliable RFF flags — solid-bogus segments (rff on EVERY frame of 29.97 real-time content;
contained by content-projected residence, unchanged) AND >=8-frame flag DROPOUTS mid-film.
Flag-only disarm (window==0) then drained frame_q ~6f/s (29.97 house vs ~24 AU/s film — no
cushion offsets a rate deficit) until the next flag run re-armed: dup bursts + aresample
hard-comps = the audible clicking (cor-1 AWE 14:00-15:00: queue pinned 1-3f, +1900 dups,
async +-1900ppm).

Fix: disarm additionally requires CONTENT-RATE evidence — fresh-frame source-spacing EMA
(tau ~8f) at <= tick*9/8. Film pacing (~41.7ms) rides dropouts armed ("[PTV-PULLDOWN] rff
flags dropped out at film pacing — staying armed" + dropouts-ridden count at disarm); a real
film->video transition disarms within ~10-15 frames (few extra pd holds, benign). Arm logic
unchanged. PTV_NO_CADDISARM reverts.

Fixture A/B (synthetic soft-telecine + 12s flag dropout + real-time tail, x264 --pulldown 32):
old = disarm at every dropout, dup=100; new = armed through dropouts, disarm on real-time
evidence, dup=0.

## 0.9.18 (in progress) — consolidation toward 1.0-rc

Implementation map: `analysis/ptvencoder-0918-implementation-map.md`. Step 0 (this commit),
observability only — zero behavior change:

- **`fqhw=` stats field** (closes #19): deepest any frame queue has ever been. The 2026-07-06
  fleet measurement + code trace showed per-process VRAM is set by the frame_q CUDA pool
  HIGH-WATER (one catch-up burst fills a rung to `g_frameq_cap` and the pool keeps it forever),
  NOT by the cushion tier — three live escalations moved GPU/RSS memory by zero. AUTO-BANK
  banks compressed CPU packets by design; no structural fix needed or taken.
- **[PTV-AGLUE] flood rate-limit**: an Azorse-class label flood (source audio labels striding
  ~6× content = one verdict/frame ≈ 8 lines/s indefinitely) now logs 10 detail lines per 10 s
  window + one suppression summary (net ms). Verdicts still apply to every frame.
- **NVENC registration-cache startup guard** (0.9.16.5 postmortem): warn when
  `PTV_FRAMEQ` + in-flight margin exceeds the registration cache (512 with v2 patch 0003,
  or `PTV_NVENC_REG_CAP`) — the config that reproduces the fleet-wide RM rwlock spiral.

Steps R1+R2 (pure code motion, zero behavior change; gated on identical regression set +
live ρ/cf traces):

- **R1** — the demux estimator's function-local statics hoisted into a `RateEstimator`
  singleton; the clock-follow latch into `HouseRateState`.
- **R2** — the house-rate correction ladder extracted as `house_rate_corr_ppm()` (its
  signature declares the ladder's complete inputs) and both demux-side rate sensors as
  `rate_estimator_feed()`. Prerequisites for per-input instances (R3/R4) and the 1.0-rc
  file split.

## 0.9.17.1 (2026-07-06) — audio path self-heals when an undecodable source recovers

Azorse TV (live incident): the source intermittently emits broken 7.1-signaled AAC that NO
decoder setting can decode (verified: default AND -strict 1 both fail; ffmpeg's log shows the
same errors). When ptvencoder STARTED during a broken phase, it built the audio path from the
never-initialized decoder params (rate 0) → graph AND swr init failed → the track was skipped
PERMANENTLY — audio never returned even after the source healed (ffmpeg recovers per-packet).
The known "audio-init race" backlog item, observed live.

Fix (app-layer): init failure is now retry-pending, not terminal — the tracked input params are
seeded with an impossible rate (-1) so the FIRST cleanly decoded frames route through the
existing [PTV-AFMT] hysteresis (5 stable frames) and rebuild the full -af path with real params;
the AFMT rebuild also drops its old `was_fg` gate so a track that never had a working graph
still gets one; a NULL-swr guard makes the dead-path window crash-proof. Validated on a
dead-then-healing fixture (audio payloads corrupted for 60s then clean): init-fail logged,
[PTV-AFMT] -1Hz→48kHz rebuild at heal, output audio present from the heal point.

Note: while the source is IN a broken phase there is still no audio — that part is upstream's
to fix; the encoder now rides through it and recovers on its own.

## 0.9.17 (2026-07-06) — dead-code removal + logging cleanup (1.0 train, behavior-neutral)

~330 lines of closed-investigation scaffolding deleted (1.0 review §4; every removed path was
default-off env-gated → output-neutral by construction; build-verified per family):
- **Φ suite** (~170): PTV_PHI1/PHIAV/PLL/PHI2/DRIFTPROBE/PHI1_RAMP/FREECNT — sensor block, mux
  wire-PLL hooks, globals, env; AVLOCK gate simplified. EXACTTICK closed this investigation.
- **PTV_AVTRIM** (~75): the actuator was never built (all 3 candidate signals proved blind);
  the vring A/V probe STAYS but loses its AVTRIM-only wall[] column (vring_put/lookup simplified).
- **PTV_RATE_LOCK** integral servo (~26, proven-bad WUCR ancestor), **PTV_ATRACE**,
  **PTV_TICK_ADJ_US** (falsification-only; 4 pacing expressions simplified), **PTV_H0_DELAY_MS**.
- Dormant B3 fields (g_pll_startup_us/g_pll_acquire_k set-never-read; AudioState pll_t0_us/
  pll_arm_until_us/pll_disturb_seen/disturb_epoch — demux-side disturb_epoch/house_disturb STAY,
  they feed the estimator freeze), dead singles (dbg_dec_sum, PTV_DISC_DROP_KF_TO_US, cf_skips),
  repo-root stale test binaries.

Logging (owner-flagged):
- **Stripped the stray "ptvencoder: " prefix** from the remaining [PTV-*] event lines — the tag
  is the identifier; grep patterns should match the bare tag.
- **[PTV-CLOCK] estimator acquire-progress chatter → PTV_DIAG** ("X/N windows accepted"; on
  chronically wandering sources it never durably locks by design and the line was permanent
  noise ~15-25/h). FOLLOW/release + unlatch events stay always-on.
- **PTV_AGLUE_MAX_MS default 900→1000**: closes the 900ms-1s orphan band (aglue stood aside,
  LAYERA blind, aresample silently followed).

Deferred to 0.9.18 (consolidation): env-surface tiering (~35 knobs → constants) — same lines as
the statics→structs work, keeps this release small and provable.

## 0.9.16.5 (2026-07-06) — SCALE FIX: NVIDIA RM-lock contention (the full-migration overload)

Full cor-1 migration (~56 channels) overloaded the box while the same channels ran fine under
ffmpeg (load 25 vs 270+, per-channel CPU 1.4×, ALL excess in sys time). Root cause measured on
cor-3 (perf: osq_lock = 32% of box CPU; callgraph ioctl→rwsem_down_write_slowpath; live kernel
stacks of the six per-rung threads all in os_acquire_rwlock_* [nvidia]): contention on the
NVIDIA driver Resource-Manager rwlock, in a feedback spiral —

  RM contention slows encode → frame_q backs up to its 160-frame cap (measured 56/56 channels
  pinned) → the working set of CUDA buffers cycling through NVENC exceeds libavcodec's
  64-entry registration cache → EVERY frame evicts+re-registers = 2 RM WRITE-lock ioctls per
  frame per rung (~20k/s box-wide) → contention worsens. Invisible at 3 channels, cliff at 56.

Two independent, env-gated fixes:
- **B1 (libavcodec, rides the new v2 0003-nvenc patch): registration cache 64 → 512**
  (MAX_REGISTERED_FRAMES, nvenc.h) so the deepest cushion (PTV_FRAMEQ cap 160 + in-flight)
  re-registers nothing in steady state. PTV_NVENC_REG_CAP=64 restores byte-identical upstream
  eviction for A/B. Registrations map already-allocated pool VRAM; entries are small records.
- **B2 (ptvencoder.c, opt-in): PTV_NVENC_SERIALIZE=1** — one process-wide mutex around the
  rung threads' video-encoder calls (encode_push), cutting this process's concurrent RM-lock
  callers 6→1 (ffmpeg's single encode thread at sys=5% is the existence proof). Pacing, PTS
  math and the delivery gate are untouched; the gate drain runs OUTSIDE the lock so a full
  mux_q can never stall sibling rungs. Default OFF until soaked.

Verify (same numbers that exposed it): perf osq_lock share, box sys% (target ≈5%), per-channel
%CPU ≈ ffmpeg's ~50%, wucr_buf returning from 160f to the ~30f target, graduated ramp on cor-3
(3→10→20→40→56). Secondary (NOT the gap, deferred): ptvencoder-added futex ~500-900/s from
queue hops; the "mystery" ~660 nanosleeps/s are NVIDIA-driver backoff inside encode calls and
should collapse with the contention.

## 0.9.16.4 (2026-07-05) — HOTFIX: [PTV-AGLUE] forward steps are GAPS, not relabels

- **Live failure of the 0.9.16.3 verdict rule within the hour of deploy (AWE Plus):** its audio
  gaps are cut UPSTREAM, so the stream arrives flowing continuously — no wall pause at our end —
  and the wall-continuity discriminator classified 4 forward gap steps (+87/+384/+85/+426ms,
  splice pairs) as RELABELs, erasing +983ms in 16 s and putting audio visibly in front of video.
  Owner caught it live; the event log made the wrong verdict diagnosable in minutes (these
  events were fully invisible before 0.9.16.3).
- **Corrected rule, direction-asymmetric:** BACKWARD steps → RELABEL erased (unambiguous —
  content cannot be negatively missing; this is the real −9.5 ms/h audio-early accumulator fix,
  A/B-validated: −308 ms permanent without glue → −10 ms with). FORWARD steps → GAP always:
  labels kept, aresample pads — the pre-glue behavior that handled AWE's gaps correctly for
  weeks. Wall-clock continuity is NOT usable evidence for forward steps; padding is faithful for
  real gaps and latency-neutral for a genuine forward relabel (the source's own return step
  cancels it).
- Deploy note: any `PTV_AGLUE_MS=0` emergency override set for 0.9.16.3 should be REMOVED with
  this version — =0 also disables the backward (accumulator) fix.

## 0.9.16.3 (2026-07-05) — the lip-sync accumulator fix ([PTV-AGLUE]) + birth log ([PTV-ANCHOR])

- **[PTV-AGLUE] symmetric audio label-step glue — the AWE-class slow lip-sync accumulator,
  mechanism MEASURED and closed.** Instrumented 3-act step-fixture runs ([PTV-ASTEP] in-pts
  camera + [PTV-AFLOW] sample ledger) plus raw-wire pts scans (`ffprobe -fflags nofillin`)
  proved the disease is an ARCHITECTURAL inconsistency, not any single component: **video
  label steps are structurally erased** by the house clock (output is stamped by frame count —
  fixture boundary 1, a +467 ms video pts step: dup=0, output unchanged), while **audio label
  steps were silently followed** by `aresample=async` (fixture boundary 2, +465 ms audio pts
  step: ~450 ms of silence padded, flash+beep residual +471 ms PERMANENT, zero log lines).
  Any source event that relabels tracks asymmetrically — or out-and-back — therefore banks a
  permanent A/V offset equal to the audio step, invisible to every internal metric (avoff=0,
  async ~0, the sub-1s band where the demux absorber is LAYERA-disabled and LAYERA is blind).
  Fix: at `audio_feed`, a raw-label step beyond `PTV_AGLUE_MS` (default 60 ms, 0 disables) gets
  an explicit logged verdict — **RELABEL** (wall-clock delivery kept flowing → erase the step
  into a per-track glue offset, matching the video side) vs **GAP** (delivery gapped ~the step →
  real missing content, labels kept, aresample pads — same logic as the v0.8.2 gap
  discriminator). Backward steps are always relabels; steps above `PTV_AGLUE_MAX_MS` (900 ms)
  are logged and left to the >1 s discontinuity layer. Detection runs on PRE-AVLOCK labels so
  LAYERA/house_skew actuation can never masquerade as a source step.
- **[PTV-ANCHOR] birth-relationship log (always-on, owner-requested):** h0 anchor (first decoded
  video frame) + per-audio-track `first_audio-h0` offset with pre-anchor drop counts
  (`dropped_pre_h0`, `ring_dropped`). A Zimbo-class startup-structural offset (fresh restart
  already ~−1.1 s, internals flat forever after) is visible in THIS line, not in any drift
  sensor.
- Kept: the PTV_DIAG-gated [PTV-ASTEP]/[PTV-AFLOW] instrumentation that localized the mechanism
  (in-pts/sink-pts step cameras + cumulative in/out sample ledger with 5 s cadence).

## 0.9.16.2 (2026-07-05)

- **Defensive: drain a parked pulldown lookahead in the normal pop path.** If cadence ever
  disarms with a frame still in `nextf`, it previously sat orphaned until the next arm promoted
  it stale (out-of-order emission + a one-tick house-skew spike the audio path samples via
  AVLOCK). Rare-path guard, NOT a lip-sync fix: a 46-flap flash+beep A/B on a synthetic
  soft-telecine flapping fixture (x264 --pulldown 32 segments alternating with native 29.97 —
  the AWE flag-dropout profile) measured **byte-identical A/V alignment with and without it**
  (+42.8/+41.9 ms pre/post in both arms, 318 events each) — **flap transitions are A/V-neutral,
  disproving pulldown flapping as the AWE lip-sync cause**. The AWE audio-early offset
  (~−408 ms measured at 46 h uptime; owner: visible after ~24 h) is under investigation as a
  slow accumulator — time-series tool: `test-scripts/repro/awe-drift-point.sh`. Fixture
  generator: scratchpad make-flapfix.sh (soft-telecine SEI via x264 CLI; flash+beep ruler).

## 0.9.16.1 (2026-07-04) — stability release, part 2

- **Sparse-PID wrap guard** (1.0 review §3.2): a PID silent for more than HALF the 33-bit wrap
  period (13.26 h @90 kHz) aliased the ±half unwrap heuristic BOTH ways — a no-wrap gap >13.26 h
  read as "late pre-roll" (−2^33 → the PID lands 26.5 h in the past → `demux_pass` drops every
  later packet FOREVER), and ≥1 wraps crossed during the silence read as small deltas (the +2^33
  silently missed → same landing). SCTE-35 quiet overnight/weekend = the realistic victim on
  months-scale uptimes. Fix: past the half-period wall threshold, RE-ANCHOR by wall projection —
  choose the wrap count landing the packet nearest its wall-expected position (a live mux stamps
  a resuming PID with the current STC, so projection is exact to clock-ppm ≪ half; handles any
  number of missed wraps). Below the threshold the delta branches are provably always correct —
  zero change to normal operation. Always-on `[PTV-DISCONT] re-anchored after N.Nh silence
  (±k wraps)` line; `PTV_WRAP_GUARD_S` overrides the threshold (TEST ONLY). k-arithmetic
  unit-checked (0/±1/±2 wraps, ±30 min projection error); live-path A/B with a 10 s test
  threshold on DVB-sub gaps: fires with k=+0 (no-op), health + sub passthrough identical to
  control.
- **Delivery-cap ratchet fixed** (1.0 review §3.3, found independently by 3 of 4 review agents):
  adaptive-cushion GROW added `(raised−base)×tick` to `g_delivery_cap_us` (+ maxq) but SHRINK
  never restored it — daily grow/shrink cycles ratcheted the gate's video-stall force-release
  ceiling ~+3 s per cycle FOREVER (months → minutes of held audio + tens of MB per rung when a
  real wedge finally happens). SHRINK now subtracts the exact GROW amount; `g_delivery_cap_us`
  is now `_Atomic` (was a plain cross-thread int64: output-thread writer, demux readers). maxq
  intentionally stays as a high-water backstop (RAM materializes only while holding, and the
  restored cap bounds that duration).

## 0.9.16 (2026-07-04) — stability release, part 1

- **MULTIVIEW MEMORY LEAK FIXED — one AVFrame shell (448 B) leaked per slot per tick.** The
  compositor's per-slot graph feed cloned a frame shell each tick and relied on
  `av_buffersrc_add_frame()` to dispose of it — but that call consumes only the *reference*
  (struct reset, still caller-owned), so on every success the shell leaked: 4-slot grid ×
  30 ticks/s × 448 B ≈ **193 MB/h**, 2-slot ≈ 97 MB/h. Exactly matches the cor-2 fleet
  measurement (big mosaics 175–240 MB/h, small 85–125, 2:1 class ratio = slot count;
  single-input unaffected — its feed reuses a long-lived struct correctly). **The leak was
  historically masked by sync_check restarting mosaics daily; 0.9.12.1 stopped those restarts,
  making this the #1 uptime bound (~4 GB/day/mosaic).** Confirmed with macOS `leaks` on a live
  local 2×2: 11,835 leaked 448-byte allocations growing at ~120/s (= 4 × 29.97). Fix: free the
  shell after the add (reference consumed on success; frame untouched on error). Found during
  the 1.0 stability review (analysis/ptvencoder-v1-cleanup-plan.md §3.1).
- Help text: escaped the `%` in the PTV_NO_CLOCKFOLLOW line (invalid printf conversion) and
  updated its stale 0.3% figure to the 0.9.15.5 threshold (0.5%).

## 0.9.15.5 (2026-07-04)

- **Clock-follow arm threshold 3000 → 5000ppm (owner-approved).** NewsNation's transport clock
  WANDERS −700..−3400ppm and chattered arm/release across the old 3000 line (~15 events/day,
  ±0.3% pace steps — harmless but noisy, and follow demonstrably isn't load-bearing there:
  the channel ran equally perfect unfollowed on WUCR + decimation). Follow now engages only
  for the genuinely-broken-clock class it was built for (the +12000ppm relay-fault class);
  release stays <2000 (wider hysteresis). Estimator, freeze, and unlatch logic unchanged.

## 0.9.15.4 (2026-07-04)

- **Log legend caught up to 0.9.14/0.9.15 (owner-requested; doc-only, no behavior change).**
  New `-stats` fields documented: `cf=` (coarse source-clock estimate, `?` unlocked, frozen on
  BURSTY) and `decim=` (surplus-cadence decimation — steady even accrual is CORRECT and
  perceived speed is always 1x). New "health events" section: [PTV-BURSTY] per-minute stall
  status, [PTV-CUSHION] tier/BANK moves, [PTV-CLOCK] arm/release + estimator lifecycle,
  [PTV-EMPTY] starvation episodes. Defaults block updated v0.9.10→v0.9.15 (adds delivery gate
  mv, residence, auto-bank, clock-follow, decimation + their PTV_NO_* reverts); tuning adds
  PTV_CUSHION_MAX_MS.

## 0.9.15.3 (2026-07-04)

- **Unique TV pause/fast-forward REGRESSION fixed — decimation cursor + estimator bursty-freeze.**
  First bursty-channel encounter of 0.9.15.x (glo-1 Unique TV, 10h45m): `dup=615288` ≈
  `decim=613220` = 53% of ALL output, a ~6s pause(+dups)/fast-forward(3x pops) cycle, with
  `cf=+28450ppm` LATCHED and clock-follow pacing +2% fast. Three-link chain: (1) the coarse
  estimator locked onto burst-aliased windows (HLS-clump DTS advance) and the 0.9.15.1-widened
  reject band then held it there forever; (2) clock-follow armed on the bogus offset — a
  structural +2% over-consumption that drained the auto-bank 0.9.14.2 had validated on this
  very channel; (3) each starvation dup bumps `last_vpts` one tick past content (the monotonic
  guard, by design), so the refill clump all read as "surplus" to the decimation check —
  decimation ate the very latency the bank retains, and the pair oscillated forever. Fixes:
  **(a) decimation compares against `last_content_vpts`** — the content index of the last REAL
  frame emitted (dups don't advance it): post-stall refills are new content, play at 1x, latency
  retained (the AUTO-BANK posture); catch-up fast-forward is gone BY CONSTRUCTION — decimation
  can only ever skip frames mapping to already-displayed content, so perceived speed is always
  exactly 1x. The <=3 pops/tick stays as a loop bound only (a 50fps feed into a 25 house = every
  other frame skipped = correct conversion, still 1x). **(b) The coarse estimator FREEZES +
  RESETS while the auto-bank is armed** (the BURSTY classifier is precisely the "these windows
  are not a clock measurement" signal), and clock-follow force-releases — re-acquires from
  scratch only if the bank decays away. **(c) Stuck-latch unlock**: post-lock, if >75% of ~2min
  of windows are rejected, the LOCKED estimate is what's wrong (an honest source accepts nearly
  all post-0.9.15.1) — unlock and re-acquire. NewsNation (smooth surplus, never banks, never
  dups) is untouched by all three. Interim channel mitigation until deploy:
  `PTV_NO_CLOCKFOLLOW=1,PTV_NO_DECIMATE=1` = exact 0.9.14.2 behavior.

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
