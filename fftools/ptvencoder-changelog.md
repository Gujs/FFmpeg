# ptvencoder version history

Per-release notes, extracted verbatim from the `ptvencoder.c` header on 2026-07-03
(the in-code block had grown to ~190 comment lines). **Add new release notes HERE,**
keep only the current `PTVENCODER_VERSION` define in the source. This file is part of
the v2 `0001` patch (additive, travels with the source to the build box).

## 1.0.1 (pending) — mv-audio robustness batch

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
KNOWN LIMITATION (adversarial review Defect 1, fixture-proven): the slip
probe (audio.c drain) measures door-label head − sink head − swr_delay =
the WHOLE -af graph's hold, so a buffering filter (loudnorm: +2914ms
constant false audio-early; atempo, long lookaheads) biases R by its hold
forever. Exposure today none — the exact production single-input chain
aresample+acompressor+alimiter fixture-reads +0ms, and multiview (where
grids run loudnorm) publishes no lipsync=. Legend carries the caveat; the
structural fix (resampler-scoped slip probe) lands with the corrector
round. Soak interpretation notes from review: (a) AWE-class sources whose
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
