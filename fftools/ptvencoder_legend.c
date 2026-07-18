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

#include "cmdutils.h"
#include "ptvencoder.h"

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
        "    -log-legend         describe every log field/line (also printed once at startup), then exit\n"
        "    -version, -h\n"
        "\n"
        "  Pacing is automatic: live (wall-clock) for net inputs, media-clock for files.\n"
        "\n"
        "  environment variables (the production posture is DEFAULT-ON — a plain invocation is correct):\n"
        "   tuning:\n"
        "    PTV_PREROLL_MS=N    startup cushion / adaptive base tier (default ~1000). Set DEEP (e.g.\n"
        "                        12000) for HLS-burst-over-SRT sources — the [PTV-BURSTY] log warning\n"
        "                        computes the right value per channel. >1600 also deep-primes video_q\n"
        "                        packets + auto-sizes the audio delivery gate. Adds ~N ms latency.\n"
        "    PTV_VIDEOQ=N        demux->decode packet-queue depth (default 256; raise with deep preroll)\n"
        "    PTV_FRAMEQ=N        decode->output frame-buffer capacity (default 160, slots only)\n"
        "    PTV_CUSHION_MS=N    adaptive cushion RAISED tier (default 4000, grows on repeated starvation)\n"
        "    PTV_CUSHION_MAX_MS  AUTO-BANK ceiling (default 12000): bursty channels self-escalate a\n"
        "                        compressed video_q bank to 1.5x their worst stall — no env needed\n"
        "    PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ   audio delivery-gate sizing (auto-sized normally)\n"
        "    PTV_VDELIVERY_CAP_MS  early-video hold bound + audio-death escape (default 6000; raise for\n"
        "                        an audio chain buffering >~4s — the hold itself is measured, not sized)\n"
        "   reverts (each disables one default-on mechanism; for A/B and rollback only):\n"
        "    PTV_NO_WUCR         occupancy-servo house pacing      PTV_NO_LAYERA   glue/discontinuity buffer\n"
        "    PTV_NO_REPRIME      fast post-glue buffer refill      PTV_NO_ADAPTIVE fixed (non-adaptive) cushion\n"
        "    PTV_NO_EXACTTICK    exact-rational video stamping     PTV_NO_MV_EXACTTICK  mosaic measurement axes\n"
        "    PTV_NO_PULLDOWN     telecine-aware film emit          PTV_NO_AVLOCK   audio house-lock\n"
        "    PTV_NO_GENLOCK      source-rate estimator             PTV_NO_GAPDISCRIM audio gap-vs-splice\n"
        "    PTV_NO_DELIVERY     audio delivery-alignment gate     PTV_NO_DELIVERY_MV ungated mosaics (pre-0.9.12.1)\n"
        "    PTV_NO_VDELIVERY    symmetric EARLY-VIDEO hold (pre12; back to video-ahead wire under a buffering -af)\n"
        "    PTV_NO_RESIDENCE    mosaic per-slot source-rate cadence (pre-0.9.13 pop-per-tick)\n"
        "    PTV_NO_AUTOBANK     runtime bursty-channel bank escalation (back to advisor-only)\n"
        "    PTV_NO_CLOCKFOLLOW  following a large verified source-clock offset (>0.5%%; e.g. a fast relay)\n"
        "    PTV_NO_QSHED        GOP-coherent video_q overflow shed (back to per-pkt tail-drop = the #32 wedge)\n"
        "    PTV_NO_RATCHREL     bank/dlvhold release on the starved-while-flowing contradiction\n"
        "    PTV_NO_SELFHEAL     internal re-prime backstop on sustained frame_q starvation\n"
        "   logging: PTV_DIAG=1 debug lines · PTV_LOG_TS=1 timestamp prefix · see -log-legend for probes\n");
}

/* v0.9.2 self-documenting log legend. full=0 (compact) describes the always-on `-stats` progress
 * line and is printed once at startup below the banner, so every channel log explains itself;
 * full=1 (via `-log-legend`, exits after) also documents the PTV_DIAG debug lines + env switches.
 * Split into <1KiB av_log calls (the default log callback truncates a single line at 1024). */
void ptv_print_log_legend(int full)
{
    av_log(NULL, AV_LOG_INFO,
        "log legend — the always-on `-stats` progress line (one per -stats_period, default 1s):\n"
        "  frame      output frames emitted so far (CFR count)\n"
        "  fps        INSTANTANEOUS emit rate over the last interval — the 'alive right now' check\n"
        "             (must sit at the output rate, e.g. 29.97; a current wedge shows here immediately)\n"
        "  time       output media time HH:MM:SS.ss\n");
    av_log(NULL, AV_LOG_INFO,
        "  dup        frames REPEATED because content was missing or late (feed-health meter):\n"
        "             steady trickle = source dropping frames upstream; bursts = delivery droughts\n"
        "  pd         intentional cadence holds (v0.9.11 telecine-aware emit): during 23.976-film-in-\n"
        "             29.97 segments a repeat_pict frame occupies its 2:3 residence — ~6/s during a\n"
        "             movie is CORRECT pulldown, not a fault ([PTV-PULLDOWN] brackets film segments)\n"
        "  drop       frames SKIPPED at frame_q overflow — the latency-drain meter: after a stall,\n"
        "             drop= ticks up while hs= bleeds down (the debt repaid in skipped content)\n"
        "  corrupt    corrupt packets discarded (demux + decode)\n"
        "  async      aresample compensation RATE (ppm); ~0 = idle/healthy, large = resampler fighting\n");
    av_log(NULL, AV_LOG_INFO,
        "  dlvhold    (delivery gate) ms of audio HELD waiting for matching video (≈ encoder latency +\n"
        "             cushion); normal ~1-2s, scales with the cushion\n"
        "  dlvforced  (gate) packets force-released because video STALLED — MUST stay ~0\n"
        "  vdlvhold   (§7.5b symmetric gate, pre12) ms of EARLY VIDEO held for audio delivery to catch\n"
        "             up ≈ the audio path's WALL latency (loudnorm ~3s fill); 0 on channels whose audio\n"
        "             is not late. [PTV-VDLV] logs the audio-death escape (video released + hold\n"
        "             disarmed until audio resumes — an audio outage never freezes video)\n"
        "  vdlvforced (shown when >0) video released by the backstop: audio flowing but permanently\n"
        "             behind (label spread) or hold-FIFO overflow — added latency clamped at ~6s\n"
        "  wucr_buf   frame_q occupancy (frames/ms) — the jitter cushion fill vs cushion= target\n"
        "  fqhw       deepest any frame queue has ever been (frames) — the CUDA pool high-water;\n"
        "             this, not the cushion tier, sets the per-process VRAM footprint\n"
        "  wucr_rho   applied house-rate offset (ppm) = recovered source-clock deviation (+ = source\n"
        "             faster); pegged ±6000 = gentle-zone fill/drain in progress\n");
    av_log(NULL, AV_LOG_INFO,
        "  hs         house_skew: latency debt vs baseline (ms) — a feed stall steps it up, catch-up\n"
        "             drops bleed it back; A/V stays LOCKED throughout (this is delay, not lip-sync)\n"
        "  hsres      (0.9.18.7) LAYERA erase-residue ledger (ms): cumulative label offset the glue\n"
        "             erases injected into the hs reading. hs growing IN STEP with hsres = erased-\n"
        "             discontinuity bookkeeping (e.g. jump-to-live parked the stall's dups), NOT\n"
        "             held content; hs growing while hsres is flat = real retained latency\n"
        "  cushion    adaptive frame_q target: ~1s lean tier or ~4s raised tier ([PTV-CUSHION] logs\n"
        "             each transition: grows on 2 starvations/60min, shrinks after 6h quiet;\n"
        "             1.0.1-pre10: [PTV-CUSHREL] releases a raised tier held against a starved-\n"
        "             while-flowing contradiction, and deficit-recovery decode is GOVERNED to\n"
        "             1.25x realtime while a shed episode's backlog drains — catch-up arrives as\n"
        "             a paced trickle, not a device-max burst. 1.0.1-pre13: the governor FAILS\n"
        "             OPEN unless its measured input rate is TRUSTED (>= declared AND fresh <30s)\n"
        "             and its own sleeps are honest ([PTV-CATCHGOV] logs a 60s fail-open when\n"
        "             wakeups overshoot 3x in 10s) — a wrong brake wedges (Newsmax2 2026-07-16:\n"
        "             6.6 dec/s on a clean 59.94pps wire), an unpaced burst only spikes)\n"
        "  bank       (v0.9.14, shown when armed) AUTO-BANK actual/target ms: a bursty channel's\n"
        "             self-escalated compressed video_q cushion (1.5x worst stall, cap 12s); fills\n"
        "             from the stalls' own retained latency, retires after 6h quiet\n");
    av_log(NULL, AV_LOG_INFO,
        "  cf         (v0.9.15, shown when notable) coarse source-CLOCK estimate (ppm vs realtime,\n"
        "             `?` = not yet locked ~60s). |cf|>5000 locked => output+PCR FOLLOW the source's\n"
        "             true rate ([PTV-CLOCK] logs arm/release); frozen while BURSTY/bank is armed\n"
        "             (clump delivery is not a clock measurement)\n"
        "  decim      (v0.9.15.2, shown when >0) SURPLUS frames decimated: source delivers more real\n"
        "             frames than its declared rate (e.g. ~25.4 fps declaring 25) — each skip is a\n"
        "             content position already displayed, so perceived speed is always EXACTLY 1x;\n"
        "             steady even accrual = correct (equals the surplus); never fires <=house-rate\n"
        "  lipsync    (1.0.1-pre9; per-slot on multiview since pre16) PASSIVE residual-sync sensor R\n"
        "             (ms, + = audio EARLY): per-stream source→output content mapping difference\n"
        "             (video EMA(out−src) + demux edit ledger vs audio ledger m_a) — sees\n"
        "             relabel-erases, wrong glues and parked resampler slip; shared latency (hs)\n"
        "             cancels. `--` = a side not flowing. On mv each track pairs against ITS OWN\n"
        "             slot's video (aK: prefix always; track→slot map = the startup [PTV-RSYNC]\n"
        "             tracks: line). 1.0.1-pre11: the slip probe is scoped to the async-aresample\n"
        "             boundary — a buffering -af's hold (loudnorm ~3s) is label-preserving latency,\n"
        "             EXCLUDED from R (the pre9 whole-graph probe read it as false audio-early).\n"
        "             Soak-CERTIFIED vs the external oracle 2026-07-16 (Δ12-21ms on real excursions,\n"
        "             both signs human-verified, NTSC 24h flat) — the corrector's gate condition.\n"
        "             PTV_RSYNC_SENSE=0 disables the sensor AND the pre15 realization tripwire\n"
        "             (both modes). Components: [PTV-RSYNC] under PTV_DIAG.\n"
        "  corr       (1.0.1-pre14, only when nonzero/engaged) residual-sync\n"
        "             CORRECTOR cumulative resampler trim (ms; `*` = actively integrating). The\n"
        "             actuation half of the supervisor: when the sensor's R dwells outside ±80ms\n"
        "             for 5min stable + 3min event-free with ALL rungs' wire provably moving, it\n"
        "             steers R→0 through aresample (≤2ms/s; park |R|≤20ms; authority 5s/engagement,\n"
        "             10s lifetime → hard disarm). [PTV-RSCORR] logs every state change\n"
        "             (arm/ENGAGE/PARK/HOLD/DISARM). analysis/ptvencoder-corrector-design.md.\n");
    av_log(NULL, AV_LOG_INFO,
        "discontinuity events (always-on since v0.9.13; were PTV_DIAG-only):\n"
        "  [PTV-LAYERA]   jump = a >1s splice detected (buffering starts); flush = the glue applied\n"
        "                 (vid_err = source A/V mis-mux this glue corrected)\n"
        "  [PTV-GLUE]     running per-input mis-mux stats — the LAYERA-retirement decision line:\n"
        "                 mean/max|err| ~0 over days => the simpler per-stream absorber suffices.\n"
        "                 REFUSED (1.0.1-pre15 #33): a flush mismatch computed from UNHEALTHY\n"
        "                 labels (health H far from 1.0 — label-flood source) is NOT routed to\n"
        "                 the content path; per-stream butt-joint instead (pre-pre4 posture),\n"
        "                 residual left to sensor/corrector\n"
        "  [PTV-DISCONT]  per-stream PTS jump absorbed / audio GAP left to aresample padding\n"
        "  [PTV-AGLUE]    (v0.9.16.4) sub-1s audio label step verdict, by direction: BACKWARD\n"
        "                 => RELABEL erased (content can't be negatively missing; closes the\n"
        "                 audio-early accumulator where aresample silently dropped content);\n"
        "                 FORWARD => GAP, aresample pads (upstream-cut gaps arrive flowing, so\n"
        "                 wall-clock is no evidence — the v0.9.16.3 lesson). PTV_AGLUE_MS=0 off.\n"
        "                 1.0.1-pre15 (#33): a backward step matching a recent open GAP-pad is\n"
        "                 the pad's RETURN leg — APPLIED (round-trip cancelled), never erased;\n"
        "                 every non-erase verdict is realization-checked (tripwire synthesizes\n"
        "                 at the swr boundary if hard comp did not fire). PTV_NO_GLUECLASS=1 off\n"
        "  [PTV-ADISC]    (1.0.1-pre15 #33, unconditional) corrupt-flagged AUDIO discarded at the\n"
        "                 demux (rate-limited count + acor= stats) — the NBS undecodable-source\n"
        "                 phase, previously silent/restart-only; PTV_NBS_FILL=1 additionally\n"
        "                 synthesizes dense silence while the phase lasts (labels stay valid,\n"
        "                 corrector held off, first real frame classified as a resume-anchor)\n"
        "  [PTV-ANCHOR]   (v0.9.16.3) birth A/V relationship: h0 (first video frame) + each\n"
        "                 audio track's first_audio-h0 offset and pre-anchor drop counts — a\n"
        "                 startup-structural lip-sync offset is visible HERE, not in drift\n"
        "  [PTV-ACOMP]    (1.0.1-pre3) graph-input audio pts step >25ms — aresample will hard-\n"
        "                 compensate (instantaneous sample insert/drop, click risk); rate-limited\n"
        "                 ~1/10s per track, cumulative count on the PLL diag line (acomp=)\n");
    av_log(NULL, AV_LOG_INFO,
        "health events (always-on):\n"
        "  [PTV-BURSTY]   per-minute delivery-stall status (count + worst gap + bank state) while a\n"
        "                 channel is bursty-classified; also the auto-bank escalation advisor\n"
        "  [PTV-CUSHION]  adaptive cushion tier moves + BANK escalations (target, sizing rationale)\n"
        "  [PTV-CUSHREL]  (1.0.1-pre10) raised cushion tier released: frame_q starved >=60s with\n"
        "                 input FLOWING while the tier held — the 6h zero-starvation release is\n"
        "                 unreachable under churn; tier back to base, gate caps restored\n"
        "                 (PTV_NO_CUSHREL disables; an input outage never triggers this)\n"
        "  [PTV-CLOCK]    clock-follow arm/release (source clock offset FOLLOWED/back-in-range) +\n"
        "                 estimator lifecycle (frozen on BURSTY, stuck-latch re-acquire, lock progress)\n"
        "  [PTV-DEGRADED] (1.0.1-pre10, opt-in PTV_DEGRADED=1, SINGLE-INPUT ONLY — hard-disabled\n"
        "                 with a startup WARNING on multiview) sustained-deficit demand admission\n"
        "                 enter/status/release; entry needs a >=3min train of QSHED full-cycles\n"
        "                 <=30s apart (very-long-GOP channels never enter)\n"
        "  [PTV-EMPTY]    frame_q starvation episodes >=2s (refill time; sub-2s aggregate per 60s)\n");
    av_log(NULL, AV_LOG_INFO,
        "multiview stats line — same head (frame/fps/time/dup/drop/dlvhold/dlvforced) + per slot:\n"
        "  inK:lipsync  (1.0.1-pre16.1, ALWAYS-ON, inside each inK: group) that slot's sensor R\n"
        "               (same sensor/sign as single-input: + = audio EARLY; multi-track slots\n"
        "               joined '|'); `--` = the slot slated / not flowing — itself the outage\n"
        "               signal; absent = the input has no sensed audio track. corr= absent: the\n"
        "               mv corrector is HELD OFF this pre (sensor-first observation soak)\n"
        "  acor         (when >0) corrupt-discarded audio pkts, GLOBAL sum — per-track detail on\n"
        "               the [PTV-ADISC]/NBS log lines\n"
        "  inK:qdrop    input-K video queue overflow drops (demux side)\n"
        "  inK:corrupt  input-K corrupt packets (demux + decode)\n"
        "  inK:pd       cadence-residence holds (v0.9.13) — a 25fps slot in a 29.97 mosaic holds\n"
        "               every 6th tick BY DESIGN (~5/s is correct rate conversion, not a fault)\n"
        "  inK:sv       genuine starvation dups (frame was DUE but the jitter buffer was empty)\n"
        "  inK:sk       published per-slot audio skew (ms) the slot's audio follows\n"
        "  inK:skres    (0.9.18.7) slot LAYERA erase-residue ledger (ms) — read like hsres= vs sk=\n");
    if (!full)
        return;
    av_log(NULL, AV_LOG_INFO,
        "\ndebug lines — set PTV_DIAG=1 to enable. These are internal CONTROLLER estimates: useful for\n"
        "debugging the pipeline, but they do NOT track on-wire lip-sync (measure that with the oracle):\n"
        "  [PTV-DIAG]     per-second engine state: dec/emitted/muxed, dup/framedrop, queue depths\n"
        "                 vq (demux→decode) frameq (decode→output jitter) muxq (encode→mux), genlock+rate.\n"
        "                 1.0.1-pre13: gpps=measured/declared input pps + gov= catch-up governor\n"
        "                 engagement (+ govslip= oversleep strikes when >0) — dec ≪ gpps*1.25 with\n"
        "                 vq pinned and gov=1 is the governor-misbehaving signature, diagnosable\n"
        "                 from logs alone. 1.0.1-pre16: per-input — the mv [PTV-DIAG] mv per-slot\n"
        "                 segment carries inK:.../gpps=M/D/gov=G (the governor ran blind on mv)\n"
        "  [PTV-AVSYNC]   per-track A/V controller telemetry: offset/avlag estimate, vlag/alag,\n"
        "                 house_skew, and (multiview) the A/V PLL integrator state. 1.0.1-pre13:\n"
        "                 the estimate is printed as avlag= (was lipsync= — SIGN IS OPPOSITE to\n"
        "                 the stats-line lipsync= sensor: avlag>0 = audio LATE; lipsync>0 = audio\n"
        "                 EARLY. The token now appears only on the -stats progress line)\n"
        "  [PTV-SWRDELAY] aresample internal buffer occupancy (a latency LEVEL; `async` is the RATE)\n"
        "  [PTV-RSYNC]    (1.0.1-pre9) residual-sensor components per track: R + dm(m_v−m_a) +\n"
        "                 ev/ea (demux label-edit ledgers) + glue/hs (audio-injected offsets) +\n"
        "                 slip (un-realized resampler correction). PASSIVE — nothing consumes R\n");
    av_log(NULL, AV_LOG_INFO,
        "  [PTV-CHAIN]    A/V trace demux→output (rawA-V / srcA-V / unwrap_inj / outA-V) to localize\n"
        "                 where an A/V offset enters\n"
        "  [PTV-LIPSYNC]  per-track err = async_pad − video lag (internal estimate)\n"
        "  [PTV-WATCHDOG] (always-on WARNING) the encoder stalled and stopped advancing\n"
        "defaults (v0.9.15): WUCR occupancy pacing + LAYERA glue handling + REPRIME fast refill +\n"
        "  ADAPTIVE cushion + delivery gate (single & mosaic) + cadence residence + AUTO-BANK +\n"
        "  clock-follow + cadence decimation are all ON — no env needed for the production posture.\n"
        "  Reverts: PTV_NO_WUCR · PTV_NO_LAYERA · PTV_NO_REPRIME · PTV_NO_ADAPTIVE · PTV_NO_AVLOCK ·\n"
        "  PTV_NO_DELIVERY_MV · PTV_NO_RESIDENCE · PTV_NO_AUTOBANK · PTV_NO_CLOCKFOLLOW ·\n"
        "  PTV_NO_DECIMATE · PTV_LAYERA_FULLSKIP (LAYERA skips the demux absorber for sub-1s steps\n"
        "  again — restores the In-Touch audio-late accumulator; A/B only) ·\n"
        "  PTV_NO_EXACTTICK (re-enables the integer-tick ~10ppm NTSC lip-sync drift; A/B only) ·\n"
        "  PTV_NO_PULLDOWN (revert telecine-aware emit: film segments back to dup-fill + hs sawtooth) ·\n"
        "  PTV_NO_RSYNC_CORR (residual-sync corrector off; 1.0.1-pre14, DEFAULT ON — parked and\n"
        "  byte-inert on a healthy channel; kill switch kept forever)\n");
    av_log(NULL, AV_LOG_INFO,
        "tuning: PTV_CUSHION_MS=N adaptive raised tier (default 4000, [1000,10000]) · PTV_CUSHION_MAX_MS=N\n"
        "  auto-bank ceiling (default 12000; beyond it = an upstream incident to surface) · PTV_FRAMEQ=N\n"
        "  frame_q capacity (default 160, [48,1024]) · PTV_PREROLL_MS=N startup cushion / base tier ·\n"
        "  PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ delivery-gate sizing\n"
        "probes: PTV_DIAG=1 debug lines above · PTV_LOG_TS=1 prepend [timestamp] ·\n"
        "  PTV_AVSYNC_PROBE=1 [PTV-AVSYNC2] decomposition of the live A/V controller\n"
        "internalized (0.9.18.7): 21 debug envs frozen at their production defaults and no longer\n"
        "  read (GENLOCK_MAX_PPM/REJECT_PPM/WINDOW_MS/EMA_SHIFT, GAP_MIN_MS, AGLUE_MAX_MS,\n"
        "  DISCONT_MS/_BACK_MS, PROGOFF_DEBOUNCE_MS, DUKF_ESCAPE_MS/_MIN_MS, H0_REANCHOR_MS,\n"
        "  AF_ACQUIRE_MS/AF_RATE_MS_S, PLL_EMA_SHIFT/TAU_MS/ACQUIRE_MS/ACQUIRE_N/REFRACTORY_MS/\n"
        "  NOISE_K/DEV_SHIFT) — setting them is now a silent no-op; see ptvencoder-changelog.md\n");
}

/* ===================== 1.0.1-pre16 shared stats-line builders =====================
 * Used by BOTH stats printers: the single-input master rung (ptvencoder_clock.c) and the mv
 * compositor (ptvencoder_mv.c). Byte-identical extraction of the pre9 lipsync= and pre14
 * corr= builders — the only semantic change is the PER-SLOT video term: mv_ema/mv_wall/ev_us
 * are read at the track's own input slot (g_rsx.a_in[ki]), which is identically 0 on single
 * input (token-for-token the pre15 line). force_idx=1 (mv) always prints the aK: prefix —
 * slots matter even with one track; 0 keeps the pre9 n_a>1 rule. Freshness is PER SIDE and
 * per slot: a slated slot's mv_wall stales past 3s → its tracks read `--` (the outage
 * signal), independent of every other slot. */
void ptv_stats_lipsync(char *buf, size_t size, int64_t now_us, int force_idx)
{
    int nn, ki;
    buf[0] = 0;
    if (!g_rsync_sense || g_rsx.n_a <= 0)
        return;
    nn = snprintf(buf, size, " lipsync=");
    for (ki = 0; ki < g_rsx.n_a && nn < (int)size - 14; ki++) {
        int     in  = (g_rsx.a_in[ki] >= 0 && g_rsx.a_in[ki] < PTV_MAX_INPUT) ? g_rsx.a_in[ki] : 0;
        int64_t mvw = atomic_load_explicit(&g_rsx.mv_wall[in], memory_order_relaxed);
        int64_t maw = atomic_load_explicit(&g_rsx.ma_wall[ki], memory_order_relaxed);
        int fresh = mvw && maw && now_us - mvw < 3000000 && now_us - maw < 3000000;
        if (force_idx || g_rsx.n_a > 1)
            nn += snprintf(buf + nn, size - nn, "%sa%d:", ki ? "," : "", ki);
        if (fresh) {                                 /* R = (m_v+E_v) − (m_a+E_a); stale side → -- (no stale anchors) */
            int64_t mv = atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ev_us[in],  memory_order_relaxed);
            int64_t ma = atomic_load_explicit(&g_rsx.ma_ema[ki], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ea_us[ki],  memory_order_relaxed);
            nn += snprintf(buf + nn, size - nn, "%+lldms", (long long)((mv - ma) / 1000));
        } else
            nn += snprintf(buf + nn, size - nn, "--");
    }
}

/* 1.0.1-pre16.1 (owner-directed): per-INPUT lipsync fragment for the mv line's inN: groups —
 * the mv stats line already carries per-input segments, so the sensor reading belongs inside
 * them rather than in a separate combined token. Emits this input's track reading(s) (same R
 * arithmetic as ptv_stats_lipsync; multi-track slots joined with '|'), `--` when a side is
 * stale (the outage signal), empty when the input has no sensed audio track. */
void ptv_stats_lipsync_in(char *buf, size_t size, int64_t now_us, int in)
{
    int nn = 0, ki, first = 1;
    buf[0] = 0;
    if (!g_rsync_sense || g_rsx.n_a <= 0 || in < 0 || in >= PTV_MAX_INPUT)
        return;
    for (ki = 0; ki < g_rsx.n_a && nn < (int)size - 12; ki++) {
        int     kin = (g_rsx.a_in[ki] >= 0 && g_rsx.a_in[ki] < PTV_MAX_INPUT) ? g_rsx.a_in[ki] : 0;
        int64_t mvw, maw;
        int     fresh;
        if (kin != in)
            continue;
        mvw   = atomic_load_explicit(&g_rsx.mv_wall[in], memory_order_relaxed);
        maw   = atomic_load_explicit(&g_rsx.ma_wall[ki], memory_order_relaxed);
        fresh = mvw && maw && now_us - mvw < 3000000 && now_us - maw < 3000000;
        if (!first)
            nn += snprintf(buf + nn, size - nn, "|");
        first = 0;
        if (fresh) {                                 /* R = (m_v+E_v) − (m_a+E_a), as ptv_stats_lipsync */
            int64_t mv = atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ev_us[in],  memory_order_relaxed);
            int64_t ma = atomic_load_explicit(&g_rsx.ma_ema[ki], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ea_us[ki],  memory_order_relaxed);
            nn += snprintf(buf + nn, size - nn, "%+lldms", (long long)((mv - ma) / 1000));
        } else
            nn += snprintf(buf + nn, size - nn, "--");
    }
}

/* corr= builder (pre14, moved verbatim). NOT called by the mv printer this pre — the mv
 * corrector is HELD OFF (see rscorr_update); the arming pre gets the mv corr= column by
 * adding the call. Absent while every track sits at corr==0 un-engaged — the quiet-channel
 * stats line is unchanged (§6). */
void ptv_stats_corr(char *buf, size_t size)
{
    int any = 0, ki, nn;
    buf[0] = 0;
    if (!g_rsync_corr || g_rsx.n_a <= 0)
        return;
    for (ki = 0; ki < g_rsx.n_a; ki++)
        if (atomic_load_explicit(&g_corr_pub[ki], memory_order_relaxed) != 0 ||
            atomic_load_explicit(&g_corr_state_pub[ki], memory_order_relaxed) == PTV_CORR_ENGAGED)
            any = 1;
    if (!any)
        return;
    nn = snprintf(buf, size, " corr=");
    for (ki = 0; ki < g_rsx.n_a && nn > 0 && nn < (int)size - 18; ki++) {
        int64_t cu = atomic_load_explicit(&g_corr_pub[ki], memory_order_relaxed);
        int     st = atomic_load_explicit(&g_corr_state_pub[ki], memory_order_relaxed);
        if (g_rsx.n_a > 1)
            nn += snprintf(buf + nn, size - nn, "%sa%d:", ki ? "," : "", ki);
        nn += snprintf(buf + nn, size - nn, "%+lldms%s",
                       (long long)(cu / 1000), st == PTV_CORR_ENGAGED ? "*" : "");
    }
}
