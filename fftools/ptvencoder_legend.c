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
        "    -cc_extract         extract EIA-608 closed captions from the video into a DVB-teletext\n"
        "                        subtitle stream in every output — OFF by default. On MULTIVIEW it\n"
        "                        runs once per input: each slot gets its OWN track, stamped on that\n"
        "                        slot's own clock (so captions stay on their own dialogue) and its\n"
        "                        own page, walked in BCD from the base (888, 889, 890, 891).\n"
        "                        -cc_slots 0,2 limits it to named inputs (default: all) — use it\n");
    av_log(NULL, AV_LOG_INFO,
        "                        when a slot already carries real DVB subtitles to copy.\n"
        "                        -cc_lang <iso639-2> sets the language + G0 national subset for\n"
        "                        every extraction (default: that input's first non-und audio\n"
        "                        language); -cc_magazine 1-8 / -cc_page 0x00-0xFF place the base\n"
        "                        page (default 8 / 0x88 = 888); PTV_NO_CC=1 kills the whole path.\n"
        "                        -cc_style faithful|block (default FAITHFUL; env PTV_CC_STYLE\n"
        "                        OVERRIDES this flag, so a box can be flipped without touching\n"
        "                        its channel config) picks WHEN a roll-up line is transmitted;\n"
        "                        the text is never rewritten under either style. faithful\n"
        "                        transmits cc_dec's snapshots verbatim, so the teletext page\n"
        "                        mirrors the CC update for update (partial words type out and\n"
        "                        scroll exactly as on the source channel).\n"
        "                        block (opt-in) transmits only lines the source has FINISHED — the row\n"
        "                        being typed never goes on the wire — and holds the source's own\n"
        "                        window depth minus that row (RU3 => 2 lines, RU4 => 3, RU2 => 1),\n"
        "                        scrolling up as the source does, cleared 5s after speech stops.\n"
        "                        It engages ONLY after a roll-up carriage return has actually been\n"
        "                        seen, so a source that never rolls up is passed through byte for\n"
        "                        byte under either style. A pop-on caption FLIPPED IN over live\n"
        "                        roll-up content (an ad break on a news channel) is finished text\n");
    av_log(NULL, AV_LOG_INFO,
        "                        by definition and goes out whole, at once.\n"
        "                        Block advances are PACED: a completed line joins a queue and the\n"
        "                        window advances one line at a time, no sooner than 1500 ms after\n"
        "                        the last displayed change (env PTV_CC_BLOCK_HOLD_MS, 0 = off),\n"
        "                        so a captioner flushing a burst of finished lines cannot repaint\n"
        "                        the page faster than it can be read; the queue drains faster the\n"
        "                        deeper it gets, and a line is never dropped to pace it — only a\n"
        "                        timeline rebase (which flushes the whole assembly) discards one.\n"
        "                        CC tracks own subtitle indices s:0..s:N-1 in input order, so\n"
        "                        -metadata:s:s:N language=mva applies the multiview naming (it\n"
        "                        overrides the tag on the wire; the G0 subset still follows the\n"
        "                        REAL source language). Copied DVB-subs are numbered after them.\n"
        "    -metadata:s:<t>:N k=v   -disposition:<t>:N flags   per-stream metadata/disposition\n"
        "    -pid_plan <plan>    DETERMINISTIC OUTPUT PIDs, ON BY DEFAULT (env PTV_PID_PLAN wins; read\n"
        "                        from output 0 for the whole ladder). v2 (DEFAULT) = content-keyed static\n"
        "                        map: a track's PID is a function of its own identity (class, variant,\n"
        "                        SOURCE language), never of which other tracks are present — video 200,\n"
        "                        data 500+ (scte_35 500), stereo audio 1000+rank(language), multichannel\n"
        "                        audio 2000+, DVB subs 3000+, audio description 4000+, hearing-impaired\n"
        "                        subs 5000+, copied teletext 6000+, CC teletext 7000+ (a mosaic keys\n"
        "                        audio / subs / teletext by SLOT, not language). A -metadata relabel never\n"
        "                        moves a PID; same-identity duplicates go to the block's overflow area.\n"
        "                        v1[,<class>=<base>,...] = the 1.2.0 packed plan (video 200, audio 300+,\n"
        "                        subtitle 400+, data 500+; rollback only); off = source-order PIDs.\n"
        "                        The startup [PTV-PID] lines print the whole plan.\n");
    av_log(NULL, AV_LOG_INFO,
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
        "    PTV_VIDEOQ=N        demux->decode packet-queue depth (default 512; raise with deep preroll)\n"
        "    PTV_FRAMEQ=N        decode->output frame-buffer capacity (default 160, slots only)\n"
        "    PTV_CUSHION_MS=N    adaptive cushion RAISED tier (default 4000, grows on repeated starvation)\n"
        "    PTV_CUSHION_MAX_MS  AUTO-BANK ceiling (default 12000): bursty channels self-escalate a\n"
        "                        compressed video_q bank to 1.5x their worst stall — no env needed\n"
        "    PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ   audio delivery-gate sizing (auto-sized normally)\n"
        "    PTV_RSS_CAP_MB=N    process RSS ceiling (default 8192; 0 disables): [PTV-MEMCAP] warn at\n");
    av_log(NULL, AV_LOG_INFO,
        "                        75%%, supervised-respawn exit at cap — bounds the flap-storm memory\n"
        "                        runaway class so one channel can never ENOMEM its neighbors\n"
        "    PTV_NO_AFMT_BREAKER=1  disable the [PTV-AFMT] rebuild-storm breaker (escalating 2..60s\n"
        "                        backoff between audio-path rebuilds on param-flapping origins)\n"
        "    PTV_VDELIVERY_CAP_MS  early-video hold FLOOR + audio-death escape base (default 6000).\n"
        "                        1.0.1-pre17: the live cap AUTO-SIZES to 1.5x the measured audio-chain\n"
        "                        lateness (ceiling 12s) — no per-channel tuning needed; env = floor only\n"
        "   timing (2.1 timeline — always on): one classifier per input decides every gap, relabel, restart,\n"
        "   burst cut and corrupt timestamp ONCE for all its streams ([PTV-TL] events); single input places the\n"
        "   sound where its picture was shown around events (the record door); mosaic slots are frame\n"
        "   synchronizers (picture by content time at a lag, sound placed with it; [PTV-MV] re-locked).\n"
        "    PTV_TL_OPEN_MS=N    path tolerance that opens an event window (default 500)\n"
        "    PTV_TL_NO_VMDOOR=1  A/B only: single-input sound by house_skew instead of the record door\n"
        "   reverts (each disables one default-on mechanism; for A/B and rollback only):\n"
        "    PTV_NO_WUCR         occupancy-servo house pacing      PTV_NO_ADAPTIVE fixed (non-adaptive) cushion\n"
        "    PTV_NO_REPRIME      fast post-glue buffer refill      PTV_NO_RUNG_FOLLOW  a clock per rung (2.2.1)\n"
        "    PTV_NO_EXACTTICK    exact-rational video stamping     PTV_NO_MV_EXACTTICK  mosaic measurement axes\n"
        "    PTV_NO_PULLDOWN     telecine-aware film emit          PTV_NO_AVLOCK   audio house-lock\n"
        "    PTV_NO_GENLOCK      source-rate estimator\n");
    av_log(NULL, AV_LOG_INFO,
        "    PTV_NO_DELIVERY     audio delivery-alignment gate     PTV_NO_DELIVERY_MV ungated mosaics (pre-0.9.12.1)\n"
        "    PTV_NO_VDELIVERY    symmetric EARLY-VIDEO hold (pre12; back to video-ahead wire under a buffering -af)\n"
        "    PTV_NO_AUTOBANK     runtime bursty-channel bank escalation (back to advisor-only)\n"
        "    PTV_NO_CLOCKFOLLOW  following a large verified source-clock offset (>0.5%%; e.g. a fast relay)\n"
        "    PTV_NO_QSHED        GOP-coherent video_q overflow shed (back to per-pkt tail-drop = the #32 wedge)\n"
        "    PTV_NO_RATCHREL     bank/dlvhold release on the starved-while-flowing contradiction\n"
        "    PTV_NO_SELFHEAL     internal re-prime backstop on sustained frame_q starvation\n"
        "    PTV_NO_MUXTOL       [PTV-MUXTOL] egress-pressure tolerance (mux-write ENOMEM/EAGAIN =\n"
        "                        drop pkt + 60s dead-rung ceiling) back to any-write-error-is-fatal\n"
        "                        (test-only: PTV_MUXFAIL_SIM=enomem|eagain|einval:<start_s>:<dur_s>\n"
        "                        pretends the mux write failed during the window)\n"
        "   logging: PTV_DIAG=1 debug lines · PTV_LOG_TS=1 timestamp prefix · see -log-legend for probes\n");
}

/* v0.9.2 self-documenting log legend. full=0 (compact) describes the always-on `-stats` progress
 * line and is printed once at startup below the banner, so every channel log explains itself;
 * full=1 (via `-log-legend`, exits after) also documents the PTV_DIAG debug lines + env switches.
 * Every av_log call stays under 2 KiB: the PTV_LOG_TS callback formats one call into a 2048-byte buffer
 * (ptvencoder.c) — a longer chunk lost its tail AND its newline, gluing the next log line onto the legend. */
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
        "             movie is CORRECT pulldown, not a fault (PTV_DIAG: [PTV-PULLDOWN] brackets film segments)\n"
        "  drop       frames SKIPPED at frame_q overflow ([PTV-FRAMEQ]) — 0 in normal running (2.1+:\n"
        "             an outage's hold is kept or erased by the timeline, not drained by drops)\n"
        "  corrupt    corrupt packets discarded (demux + decode)\n"
        "  async      aresample compensation RATE (ppm, EMA); ~0 = idle/healthy, a sustained large value =\n"
        "             resampler fighting (2.2.1: a step — hold, fill, label step — is not averaged in)\n");
    av_log(NULL, AV_LOG_INFO,
        "  dlvhold    (delivery gate) ms of audio HELD waiting for matching video (≈ encoder latency +\n"
        "             cushion); normal ~1-2s, scales with the cushion\n"
        "  dlvforced  (gate) packets force-released because video STALLED — MUST stay ~0\n"
        "  dlvdrop    (gate, single input, shown when >0) copied AC-3/MP2 packets dropped because the gate\n"
        "             was full — MUST be 0\n"
        "  vdlvhold   (§7.5b symmetric gate, pre12; armed on MULTIVIEW too since pre17 — task #48:\n"
        "             the fleet loudnorm chain made every mv audio track wall-late, video left ~2-3s\n"
        "             early on the wire with labels intact) ms of EARLY VIDEO held for audio delivery\n"
        "             to catch up ≈ the audio path's WALL latency (loudnorm ~3s fill); 0 on channels\n"
        "             whose audio is not late. NOTE the metric flip on a buffering-af mv channel:\n"
        "             dlvhold reads 0 (nothing early to hold) and vdlvhold ~1-3s — EXPECTED, the\n"
        "             buffering moved to the video hold. [PTV-VDLV] logs the audio-death escape\n"
        "             (video released + hold disarmed until audio resumes — an audio outage never\n"
        "             freezes video) and the pre17 cap auto-size (1.5x measured lateness, ceil 12s)\n"
        "  vdlvforced (shown when >0) video released by the backstop: audio flowing but permanently\n"
        "             behind (label spread) or hold-FIFO overflow — added latency clamped at the\n"
        "             auto-sized cap (floor 6s / PTV_VDELIVERY_CAP_MS, ceiling 12s)\n"
        "  wucr_buf   frame_q occupancy (frames/ms) — the jitter cushion fill vs cushion= target\n"
        "  fqhw       deepest any frame queue has ever been (frames) — the CUDA pool high-water;\n"
        "             this, not the cushion tier, sets the per-process VRAM footprint\n"
        "  wucr_rho   applied house-rate offset (ppm) = recovered source-clock deviation (+ = source\n");
    av_log(NULL, AV_LOG_INFO,
        "             faster); pegged ±6000 = gentle-zone fill/drain in progress\n");
    av_log(NULL, AV_LOG_INFO,
        "  hs         house_skew (ms): output time minus source label time since the start. Grows by every\n"
        "             kept hole, hold, erased relabel and skipped catch-up and does not come back — it is\n"
        "             NOT latency (latency = wucr_buf + bank + dlvhold) and not lip sync (that is lipsync=)\n"
        "  hsres      always 0 since 2.1 (the LAYERA erase-residue ledger is not run); kept for parsers\n"
        "  vph        (2.2.2 T-499, single input) the picture-phase target (ms): how far after (+) or before (-)\n"
        "             their labels the pictures are on air inside one tick; each transcoded track's sound moves\n"
        "             to it at 2-4 ms/s. 0.0 = not followed (film cadence, a frame-rate conversion, no pictures)\n"
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
        "             from the stalls' own retained latency, retires after 1 h quiet; a lone\n"
        "             >=3 s stall's bank (pre20) holds only what came and retires after 10 min\n");
    av_log(NULL, AV_LOG_INFO,
        "  cf         (v0.9.15, single input) the source CLOCK's rate (ppm vs realtime, + = fast), shown\n"
        "             while locked or beyond 500 ppm. 2.2.1: a regulated input = measured on its PCR by the\n"
        "             regulator (earliest arrival per 2 s, 16-60 s fit) — shown on every line while that\n"
        "             measure is valid — and followed at once when it clears 5000 ppm by 2 standard errors\n"
        "             of its fit (a measure older than 30 s counts as none); otherwise the coarse video\n"
        "             estimate (`?` = not locked, ~60 s; followed after the servo was pinned 2 min).\n"
        "             [PTV-CLOCK] logs follow/release; never followed while BURSTY/bank is armed\n"
        "  decim      (v0.9.15.2, shown when >0) SURPLUS frames decimated: source delivers more real\n"
        "             frames than its declared rate (e.g. ~25.4 fps declaring 25) — each skip is a\n"
        "             content position already displayed, so perceived speed is always EXACTLY 1x;\n"
        "             steady even accrual = correct (equals the surplus); never fires <=house-rate\n"
        "  lipsync    (1.0.1-pre9; per-slot on multiview since pre16) PASSIVE residual-sync sensor R\n"
        "             (ms, + = audio EARLY): per-stream source→output content mapping difference\n"
        "             (video EMA(out−src) + demux edit ledger vs audio ledger m_a) — sees\n"
        "             relabel-erases, wrong glues and parked resampler slip; shared latency (hs)\n"
        "             cancels; an ADOPTED clock of its own ([PTV-TL] offset adopted) is excluded. `--` = a side not flowing, the track filled with silence (afill=)\n"
        "             or (2.0.2, single input) the source STALLED — no content to compare.\n"
        "             On mv each track pairs against ITS OWN\n"
        "             slot's video (aK: prefix always; track→slot map = the startup [PTV-RSYNC]\n"
        "             tracks: line). 1.0.1-pre11: the slip probe is scoped to the async-aresample\n"
        "             boundary — a buffering -af's hold (loudnorm ~3s) is label-preserving latency,\n");
    av_log(NULL, AV_LOG_INFO,
        "             EXCLUDED from R (the pre9 whole-graph probe read it as false audio-early).\n"
        "             Soak-CERTIFIED vs the external oracle 2026-07-16 (Δ12-21ms on real excursions,\n"
        "             both signs human-verified, NTSC 24h flat) — the corrector's gate condition.\n"
        "             PTV_RSYNC_SENSE=0 disables the sensor AND the pre15 realization tripwire\n"
        "             (both modes). Components: [PTV-RSYNC] under PTV_DIAG.\n"
        "  corr       (1.0.1-pre14, only when nonzero/engaged) residual-sync\n"
        "             CORRECTOR cumulative resampler trim (ms; `*` = actively integrating). The\n"
        "             actuation half of the supervisor: when the sensor's R dwells outside ±40ms\n"
        "             for 5min stable + 3min event-free with ALL rungs' wire provably moving, it\n"
        "             steers R→0 through aresample (≤2ms/s; park |R|≤20ms; authority 5s/engagement,\n"
        "             10s lifetime → hard disarm). [PTV-RSCORR] logs every state change\n"
        "             (arm/ENGAGE/PARK/HOLD/DISARM). analysis/ptvencoder-corrector-design.md.\n");
    av_log(NULL, AV_LOG_INFO,
        "  cc         (only with -cc_extract) EIA-608->DVB-teletext, as captions/erases/keepalives/\n"
        "             a53: pages of TEXT emitted / page ERASES emitted (the source cleared its\n"
        "             caption memory) / 1 Hz stuffing packets / decoded frames that CARRIED A53 CC\n"
        "             side data. Always printed while the feature is on, because a SUBTITLE stream\n"
        "             is exempt from the [PTV-MUXGUARD] fatal escalation — a dead CC path would\n"
        "             otherwise be completely silent.\n"
        "             Read the quad: a53=0 => the SOURCE carries no captions (nothing to do);\n"
        "             a53 climbing with captions=0 => the extraction produces nothing (fault, or a\n"
        "             genuinely silent stretch — [PTV-CC] QUIET says so after 60s); erases without\n"
        "             captions => decoding runs but every page comes out blank; keepalives NOT\n"
        "             advancing at ~1/s => the tap or the emitter is wedged.\n"
        "  ccheld     (shown when >0) DISTINCT source clears DEFERRED because the page had not had\n"
        "             its minimum on-screen time (1.2s). Roll-up sources clear rows constantly, so\n"
        "             obeying every clear showed real captions for only 0.3-0.6s. Counted once per\n"
        "             clear, not per repeat/re-arm, so cclate <= ccheld always holds\n"
        "  cclate     (shown when >0) of those, the ones the teletext encoder actually put on the\n"
        "             wire once the 1.2s elapsed (counted after the encode, like caps/erase/keep).\n"
        "             The difference is clears superseded by a new caption before they were due,\n"
        "             plus clears the encoder had nothing to do (page already blank).\n"
        "             ccheld climbing with cclate flat => clears are being LOST, not delayed, and\n"
        "             pages will outlive the source's (that was the pre-fix behaviour)\n");
    av_log(NULL, AV_LOG_INFO,
        "  ccmux      (shown when >0) teletext packets OUR OWN mux backward-dts guard dropped\n"
        "             ([PTV-MUXGUARD], stream = the teletext PID). Every other counter here is\n"
        "             taken where the page was produced, so this is the only one that speaks for\n"
        "             the WIRE: caps= can climb while ccmux= eats the packets and no receiver ever\n"
        "             sees them. Any nonzero value means captions were lost after the encoder.\n"
        "             Counted PER RUNG PER PACKET: one caption lost on a 6-rung ABR ladder reads\n"
        "             ccmux=6, so divide by the rung count before comparing it with caps=. It\n"
        "             covers ONLY this guard — packets lavf itself declines are not counted here.\n"
        "             Same name on the stats line, the [PTV-CC] done line and the MULTIVIEW stats\n"
        "             line (where it is global, not per slot — the guard runs downstream of the\n"
        "             per-slot fan-out and cannot be attributed to one).\n"
        "  mrect      (shown when >0, [PTV-CC] done line) cc_dec subtitles that carried MORE THAN\n"
        "             ONE rect for a single frame — an EDM followed by the next screen's\n"
        "             characters is the reachable case. Every rect is processed, in order; the\n"
        "             counter exists because reading only rect 0 used to classify the whole packet\n"
        "             by the ERASE and throw the text away. Nonzero is NORMAL for such a source,\n"
        "             not a fault. One [PTV-CC] INFO line names it the first time it happens.\n"
        "             (2.1 T-326) [PTV-CC] inK captions of a corrupt-discarded picture kept: the picture is\n"
        "             dropped before decode, its EIA-608 pair still reaches the tap in display order;\n"
        "             [PTV-CC] inK the source carries EIA-608 captions but -cc_extract is not set for it\n");
    av_log(NULL, AV_LOG_INFO,
        "             (2.1 T-400, once, after 10 real pairs): pin SV_CC_EXTRACT=true for this channel\n"
        "  unreg      (single input, shown when >0, 2.2.0-pre12) inputs the regulator passes as they arrive now\n"
        "             (their PCR is no clock: [PTV-REG] 'no clock to pace by' / 'regulated again')\n"
        "  ccerr      (shown when >0) cc_dec + teletext encoder errors — rate-limited [PTV-CC] lines\n"
        "  ccdrop     (shown when >0) caption events LOST: event queue full (emitter slower than\n"
        "             decode — never blocks video), captions arriving before the h0 anchor exists,\n"
        "             an encoder error, or a mux queue already closed. A steady climb is a real\n"
        "             fault.\n"
        "             [PTV-CC] also logs the first caption, QUIET/RESUMED, timeline resets and a\n"
        "             final a53/caps/erase/keep/err/drop/bump/reset/eheld/elate/mrect/ccmux tally.\n"
        "  src        (2.0.0, single input) source state: live | stalled (+ hold=<s> since the hold\n"
        "             began) | waiting (+ hold=<s>, before the first picture, on a frame=0 heartbeat line —\n"
        "             a mosaic prints this heartbeat too while it waits for its slots; its running line has no src=).\n"
        "             sync_check alerts instead of restarting while it is not live.\n"
        "  afill      (2.0.0, shown while filling) silence-fill run so far, per track: aN = transcoded\n"
        "             track N (source gone, or its packets absent 2 s with video flowing), cN = copied\n"
        "             AC-3/E-AC-3 at input stream N (silent frames of its own codec).\n");
    av_log(NULL, AV_LOG_INFO,
        "timing events (2.1 timeline, always on — one decision per source event, for every stream of the input):\n"
        "  [PTV-TL]       one line per event, numbered, naming the stream that opened it and the decision —\n"
        "                 J = the step in the labels, wall = the input's silence before it, kept / erased = what\n"
        "                 the output keeps as a hole / joins up (whole pictures), k = the input's program offset\n"
        "                 after it. Decisions: hole kept | hole, flowing\n"
        "                 part erased (labels jumped further than the wall time: the rest erased) | relabel erased\n"
        "                 (a jump with no wall time) | step back erased | new domain, gap kept (a restart) |\n"
        "                 outlier window (corrupt labels dropped) | single-stream step (one stream only, labels\n"
        "                 as sent; RELABEL may repair it). WARNING when the program moved, INFO otherwise.\n"
        "                 2.2.1: an audio whose clock leaves the program clock while pictures run is MUTED\n"
        "                 (silence, like a frame synchronizer) until it is back on the program clock; still\n"
        "                 off after 10 s its offset is ADOPTED (its own timing: lipsync= excludes it). Also:\n"
        "                 'did not cross / crossed after the video's step', 'its own step ... erased on the\n"
        "                 stream' (a joint backward step erased per stream by its own step, rc7)\n"
        "  [PTV-SYNCCHK]  (2.2.1) once per track and event, after 3 s of clean source and 2 s of agreeing\n"
        "                 readings: the A/V offset (ledger + vring, + = audio early) — in sync (within 25 ms)\n"
        "                 | STEPPED (25 ms .. 5 s stepped out of the audio labels once) | would step ... not\n"
        "                 stepped (an erase without a verdict, a mosaic slot, PTV_NO_B2STEP=1) | beyond the 5 s\n");
    av_log(NULL, AV_LOG_INFO,
        "                 cap | no steady reading. Prints the lipsync= sensor's value to compare.\n"
        "                 A mosaic slot (2.2.3 T-515): vring = the slot's realized picture lag against the place\n"
        "                 its sound went to, its delay line counted (the paired picture's own lag only when it\n"
        "                 is off the realized lag by more than 100 ms)\n"
        "                 STEPPED moves the sound on air at the graph door (2.2.1-rc10, T-488). Single input,\n"
        "                 also 'the averaged sensor (lipsync=, the corrector) read X ms where the check reads\n"
        "                 Y ms: calibrated A -> B ms' — lipsync= and the slow corrector are set to agree with the\n"
        "                 verdict, the corrector acts on drift only (PTV_NO_RSCAL=1 off)\n"
        "  [PTV-SRC]      (2.0, single input) the source state machine, one line per change: in0 LIVE (start),\n"
        "                 LIVE \xe2\x86\x92 STALLED (no picture for -stall_min: the last picture is held, then bars / black\n"
        "                 after -freeze_max; the sound is filled with silence: 'hold fill' / 'hold fill ended'),\n"
        "                 STALLED \xe2\x86\x92 LOST (no packets: the input is reopened, 'reopen attempt #N failed' every 30 s),\n"
        "                 LOST \xe2\x86\x92 reopened, STALLED \xe2\x86\x92 LIVE after N s (N = the hold's length), 'still STALLED'\n"
        "                 (1/min, after 10 min every 10 min). 'rejoin class BURST | LIVE LOSS | NEW DOMAIN' = how the\n"
        "                 source came back (its backlog / the live content / a restarted clock); 'skipped the\n"
        "                 catch-up' = more than the max delay would be kept, the output resumes at the live content.\n"
        "                 'came back reshaped ... rejoined in-process' (2.2.1: more streams, a new picture size);\n"
        "                 'came back RESHAPED' + 'exiting 4' = a mapped stream is gone or changed: the wrapper\n"
        "                 re-probes and restarts (normal after a source change, a fault when it repeats).\n"
        "                 Other returns: 'came back with N of its M streams on new PIDs, the layout unchanged —\n"
        "                 resumed' (a renumber) | 'came back incomplete ... retrying for up to 10 s before calling\n"
        "                 it reshaped' | 'came back without its declared SCTE-35 — the PID stays, empty until the\n"
        "                 source sends it again'. 'damaged input: N frames discarded until the video played\n"
        "                 cleanly' = the hold of a damaged source ends. Exits: 'video changed to WxH ... for 2 s —\n"
        "                 exiting 4' | 'live for N s under the fallback start \xe2\x86\x92 exiting 5' (-fallback_rebind:\n"
        "                 the wrapper re-probes and starts the full plan) | 'PROBING for N s with packets flowing\n"
        "                 but no usable video — exiting' | 'gave up after N s (-wait_input)' (exit 1)\n"
        "                 inK WAITING / PROBING (start, with the frame=0 heartbeat line), inK WAITING \xe2\x86\x92 LIVE (mosaic slot)\n"
        "  [PTV-SRC] in0 burst: our own depth cut — 'the cut is erased at the door' (source live: no jump)\n"
        "                 or 'stays as a hole' (the source paused inside it)\n"
        "  [PTV-SRC] inK PMT version a \xe2\x86\x92 b: the source's program changed (PIDs added / gone); inK audio\n"
        "                 track N ... silent for 10 s while the video flows / flowing again (2.1, T-003);\n"
        "                 inK pid 0xP (codec): 'the source's PMT says LANG (was OLD) — the output follows' (a copied\n"
        "                 subtitle track takes its language / page data from the source's current PMT; 2.2.1, T-493);\n");
    av_log(NULL, AV_LOG_INFO,
        "                 in0 source frame rate changed a \xe2\x86\x92 b fps (held 30 s; the output rate stays, the\n"
        "                 pictures of a slower source are shown on their due ticks — 2.1, T-011)\n"
        "  [PTV-SRC] PID / codec moves (2.2, taken in-process): 'video moved to pid B (pid A silent N s) —\n"
        "                 taken over in-process, the picture resumes at its next keyframe' (or '— reopening');\n"
        "                 'audio moved to pid B (pid A silent N s) — taken over in-process on the same track'\n"
        "                 (2.2.1: only with evidence of a move — the video flowing, the new stream born after\n"
        "                 the old one's last packet, the same language when both carry one); 'audio pid A flows\n"
        "                 again — the take-over of pid B ends'; 'video codec X \xe2\x86\x92 Y on pid P — decoder swapped';\n"
        "                 'SCTE-35 appeared on pid P — carried on the declared output PID' | '... after the start\n"
        "                 — NOT on the output' (no SCTE-35 at the start: cues lost until a restart; set\n"
        "                 SV_SCTE35=declared for the channel)\n"
        "  [PTV-VMAP]     copies / captions placed where the video showed their content (counts)\n"
        "  [PTV-MV]       (re-lock) a mosaic slot's frame sync moved its lag by <step> ms (pictures late for\n"
        "                 1 s at real-time pace, a start-up backlog, depth the buffer never used in 60 s,\n"
        "                 content far ahead); the slot's sound takes the same step at the same content. A\n"
        "                 repeated / skipped picture (lost, late, a backlog catching up, rate conversion)\n"
        "                 moves nothing and is not logged\n"
        "                 Reasons as printed: late for 1 s | content ahead | the buffer overflowed (the\n"
        "                 pictures before it are gone) | the buffer never below N ms in 60 s. Other lines:\n"
        "                 inK joined: first picture (a late slot starts) | inK catch-up: the decoder had N\n"
        "                 packets banked ... the slot cuts to the live edge (after a return) | aK sound\n"
        "                 re-placed (lag step / drift: the sound follows its picture) | aK sound more than N s\n"
        "                 ahead of its picture: dropped | inK pictures are WxH ... scaled to the cell (the\n"
        "                 source changed size) | inK no signal \xe2\x86\x92 bars / slate / black in its cell\n"
        "  [PTV-RES]      inK startup backlog trimmed N frames (keep M): a slot's start-up burst cut to its buffer\n"
        "                 depth — normal at every mosaic start (PTV_DIAG adds the per-second [PTV-RES] t= depths)\n"
        "  [PTV-DISCONT]  one picture's corrupt timestamp restamped between its neighbours / a wrap step\n");
    av_log(NULL, AV_LOG_INFO,
        "                 reverted (a single bad label, not a timeline event)\n"
        "  [PTV-LAYERA]   a video PTS far from its DTS (a corrupt label): the PTS dropped, the frame keeps its DTS\n"
        "  (2.0 lines no longer written in 2.1: [PTV-GLUE], the [PTV-AGLUE] step verdicts, the mosaic\n"
        "   [PTV-PLL] ACQUIRE / FROZEN, [PTV-RESFOLLOW], [PTV-REANCHOR2], [PTV-RESYNC], [PTV-RECANCHOR])\n"
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
        "                 ~1/10s per track, total in the line; acomp= on the mosaic [PTV-AVSYNC] DIAG line\n"
        "  [PTV-AF]       (WARNING, start) -af/-filter:a on output N differs from output 0 and is IGNORED: the\n"
        "                 audio graph is built once per track from output 0 (-filter:a:N = per track)\n"
        "  [PTV-CONV]     aK(inN) seam-park expired — converging re-enabled (rare, informational)\n"
        "  [PTV-PID]      (1.2.0-pre8, startup) the deterministic output PID plan and where it came\n"
        "                 from (default / -pid_plan / env PTV_PID_PLAN), then one line per output\n"
        "                 stream: pid=<n> <class>/<codec>/<source language> <- <source>. When a\n"
        "                 -metadata override puts a DIFFERENT tag on the wire the column reads\n"
        "                 src(wire:tag) — the sort used the source tag, never the override. The same\n");
    av_log(NULL, AV_LOG_INFO,
        "                 plan is written to every rung, so the lines are printed once. Absent when\n"
        "                 -pid_plan off restores the legacy source-order PIDs\n");
    av_log(NULL, AV_LOG_INFO,
        "audio path events (always-on, aK(inN) = output track K from input N):\n"
        "  [PTV-BIND]     which source stream each output track / the video is bound to (start: '-map rule \xe2\x86\x92\n"
        "                 inN 0xPID codec ch lang'), decoder open lines, and mid-run: 'pid gone ... silent until\n"
        "                 one comes' / 're-bound: pid A \xe2\x86\x92 B' (same language + role) / 'no source track qualifies'.\n"
        "                 'the track stays silent' (no decoder / open failed) is a fault to look at\n"
        "  [PTV-AFMT]     the audio INPUT format changed (rate / layout / sample format, confirmed over several\n"
        "                 frames): a converter is put in front of the unchanged path, or the path is rebuilt; the\n"
        "                 output format never changes. 'rebuild BREAKER' = the source flaps its format, rebuilds\n"
        "                 are spaced 2..60 s (frames dropped meanwhile). 'splice overlap trimmed' / 'source gap ...\n"
        "                 filled with silence' = what the switch itself cost. Normal at ad splices; a storm is not\n"
        "  [PTV-ATRIM]    'input stepped back N ms — the overlap is dropped at the graph door (labels kept)': the\n"
        "                 sound's labels went back up to 5 s (after a rejoin map or a hold fill); what was already\n"
        "                 sent is not sent twice. Followed by 'N ms dropped in all'. Normal after an event\n"
        "  [PTV-ADEC]     'decode error ... dropped, track continues' (first one, then a 10 s count) = damaged\n"
        "                 audio packets; 'impossible samples ... muted as damage' = a decoder output far over full\n"
        "                 scale replaced by silence. A few at a loss are normal, a steady stream is a bad source\n"
        "  [PTV-ADECWD]   no decoded audio for 45 s although its packets arrive: the decoder is reopened (the\n");
    av_log(NULL, AV_LOG_INFO,
        "                 anchor is kept). Rare; repeated = the source's audio is undecodable\n"
        "  [PTV-ACHOP]    the audio stayed chopped (errors above the floor) for minutes: the whole audio path of\n"
        "                 the track is rebuilt once ([PTV-AFMT] follows). PTV_NO_ACHOP_REBUILD=1 disables\n"
        "  [PTV-ASTAMP]   decoded audio frames without a timestamp (a probe-failed source phase): stamped by\n"
        "                 counting samples from the last label\n"
        "  [PTV-AVLOCK]   (single input) 'short gap (+N ms, no hold)': the sound returned after a gap too short\n"
        "                 for a hold; it keeps the pre-gap house_skew until the video shows the same content\n"
        "                 ('house_skew ... again after N s'). First 3 gaps per track only. Informational\n"
        "  [PTV-RELABEL]  the content check after an audio-only step back that the timeline repaired: 'was a\n"
        "                 RELABEL' (the sound did not repeat: the repair stands) | 'the audio REPEATS' (a real step\n"
        "                 back: left to the sync check / corrector) | 'no verdict' (tone, silence, a splice) |\n"
        "                 'erase UNDONE' (silent on both sides, no evidence). Decides what the lipsync sensor reads after it\n"
        "  [PTV-WALLEV]   measurement only: more real audio than video (or the reverse) was deleted without wall-\n"
        "                 time evidence, by more than 0.5 s — pressure towards a lasting offset; watch the lipsync sensor\n");
    av_log(NULL, AV_LOG_INFO,
        "start-up lines (once per start):\n"
        "  [PTV-START]    'first picture on air N s after start' with the split (source live / set-up done / first\n"
        "                 picture decoded) and the content queued behind it; 'start-up window closed after 30 s'\n"
        "                 (longer when a start-up backlog was kept); 'start-up backlog drained'. PTV_DIAG adds the\n"
        "                 mosaic per-tick trace [PTV-START] t=\n"
        "  [PTV-RATE]     the output frame rate and why (-r source | auto | <rate> | max:<rate> | field; the source's\n"
        "                 rate and scan). A wrong output rate is read here first\n"
        "  [PTV-OUT]      'GOP N ticks on the output timeline' (every rung's keyframes on the same pictures);\n"
        "                 mosaic: 'all N slots live after X s' or 'slot_wait expired: starting with K of N slots\n"
        "                 live'; '-map ...: slot K not live — silent until it joins / skipped'; 'no slot live' (ERROR)\n"
        "  [PTV-RUNG]     (2+ rungs) every rung shows the picture rung 0 shows, on the same tick. 'rung N fell behind\n"
        "                 rung 0 by more than its frame queue — K tick(s) skipped' (that rung's encoder is too slow; at\n"
        "                 most one line per 10 s) | 'rung N found none of rung 0's pictures in its queue for 2 s'\n"
        "                 (it starts over from the next picture). PTV_DIAG adds 'rung N: K tick(s) in 60 s without\n"
        "                 rung 0's picture' (its queue dropped them; its last picture repeated). PTV_NO_RUNG_FOLLOW=1\n"
        "                 gives every rung its own clock again (2.2.1)\n"
        "  [PTV-LADDER]   (-ladder) one line per rung (size, video / audio bitrate, destination), the -max_res\n"
        "                 cap when it trimmed rungs, and the generated filter graph\n"
        "  [PTV-REPORT]   (-bind_report) the planned PIDs and bindings, nothing encoded\n"
        "  [PTV-MEMCAP]   'RSS cap watchdog armed: N MB' at start; WARNING at 75%% of the cap (memory growth:\n"
        "                 capture the [PTV-AFMT] / [PTV-CUSHION] lines), FATAL exit for a supervised respawn after\n"
        "                 two 10 s samples at the cap. PTV_RSS_CAP_MB tunes it, 0 disables\n");
    av_log(NULL, AV_LOG_INFO,
        "health events (always-on):\n"
        "  [PTV-BURSTY]   per-minute delivery-stall status (count + worst gap + bank state) while a\n"
        "                 channel is bursty-classified; also the auto-bank escalation advisor.\n"
        "                 A -regulate input reports it as 'inK ... at the input — regulated: A-B s\n"
        "                 held (D d s)', measured where the source arrives (its demuxer sees a paced\n"
        "                 stream); A near 0 = the delay barely covers the gaps. A mosaic slot\n"
        "                 reports it as 'inK ... — mosaic slot' (no bank; -regulate absorbs bursts)\n"
        "  [PTV-REG]      (2.2) the input regulator: every udp:// input is read at its PCR time + D\n"
        "                 (-regulate D; default 0.5 s since 2.2.0-pre10, 0 = off). Start line per\n"
        "                 input; 'hole X s' = the source stopped as long as its PCR jumped (kept on\n"
        "                 its schedule); 'PCR jump' = schedule continued; 'N of M PCR packets later\n"
        "                 than D' (per minute) = arrival jitter beyond D, those passed as they came\n"
        "                 (as unregulated) — a channel that says it every minute wants a larger D;\n"
        "                 'more than D + 10 s held' = the oldest dropped. The schedule follows the\n"
        "                 source's delivery (the earliest packet of each 10 s waits D; PTV_DIAG adds\n"
        "                 [PTV-REGDIAG] per 10 s: that wait, the rate correction, the depth);\n"
        "                 'no clock to pace by' (2.2.0-pre11) = the input's PCR runs ahead of its\n"
        "                 arrivals or comes less than once a second: the input passes as it arrives\n"
        "                 (not regulated); 'regulated again' (2.2.0-pre12) = its PCR was a clock for\n"
        "                 10 min, the hold builds back up to D over ~1 min; 'PCR moved: pid A -> B' = the\n"
        "                 source's PCR changed PID, the schedule follows it; 'no PCR in 5 s — passing the\n"
        "                 input through' = no PCR in the first 5 s: the input passes delayed by D, not regulated\n");
    av_log(NULL, AV_LOG_INFO,
        "  [PTV-CUSHION]  adaptive cushion tier moves + BANK escalations (target, sizing rationale)\n"
        "  [PTV-CUSHREL]  (1.0.1-pre10) raised cushion tier released: frame_q starved >=60s with\n"
        "                 input FLOWING while the tier held — the 6h zero-starvation release is\n"
        "                 unreachable under churn; tier back to base, gate caps restored\n"
        "                 (PTV_NO_CUSHREL disables; an input outage never triggers this)\n"
        "  [PTV-CLOCK]    clock-follow arm/release (source clock offset FOLLOWED — '(measured on its PCR)' for a\n"
        "                 regulated input, '(estimated from its video)' otherwise — back in range / measure lost) +\n"
        "                 estimator frozen on BURSTY (PTV_DIAG adds re-acquire + lock progress; [PTV-REGDIAG]\n"
        "                 carries the regulator's clock measure)\n"
        "  [PTV-DEGRADED] (1.0.1-pre10, opt-in PTV_DEGRADED=1, SINGLE-INPUT ONLY — hard-disabled\n"
        "                 with a startup WARNING on multiview) sustained-deficit demand admission\n"
        "                 enter/status/release; entry needs a >=3min train of QSHED full-cycles\n"
        "                 <=30s apart (very-long-GOP channels never enter)\n"
        "  [PTV-EMPTY]    frame_q starvation episodes >=2s (refill time; PTV_DIAG: sub-2s aggregate\n"
        "                 per 60s + a 5s still-empty heartbeat)\n"
        "  [PTV-WATCHDOG] (WARNING) the encoder stalled and stopped advancing\n");
    av_log(NULL, AV_LOG_INFO,
        "  [PTV-STALL]    (ERROR, at most 1/min) which thread stopped and where: 'vdec thread (inK) stalled N s at\n"
        "                 <position>' (packets arrive, nothing decoded for 5 s) or 'output/stats thread stalled'.\n"
        "                 With [PTV-WATCHDOG] (no picture emitted for 2 s) the line to send with a wedge report\n"
        "  [PTV-NOVIDEO]  (FATAL) packets flow but no picture decoded since the start (PTV_NOVIDEO_EXIT_S, default\n"
        "                 300 s): undecodable video, the process exits for a supervised respawn\n"
        "  [PTV-VFILL]    N pictures repeated into content gaps (source pictures lost or corrupt), at most one\n"
        "                 line per 10 s; the output timestamps stay continuous. Counts as dup= too\n"
        "                 (2.2.2 T-501, PTV_DIAG only) 'N picture repeat(s) into source timing gaps in 60 s' = forward\n"
        "                 label steps < 200 ms with no frame lost (a splice re-phase, a label walk) filled with whole\n"
        "                 pictures instead of a PTS hole; they count as pd= (residence), not dup= (PTV_NO_VGAP_SRC=1 disables)\n"
        "  [PTV-VPHASE]   (2.2.2 T-499, single input) 'the pictures' place on the house grid moved +N ms ...: the\n"
        "                 sound follows over S s' = the source's picture labels stepped by part of a frame (a splice);\n"
        "                 the transcoded sound moves to the pictures again at 2 ms/s (4 ms/s for the part beyond 20 ms\n"
        "                 and next to the rounding line), copied tracks do not. 'the picture phase is not followed:\n"
        "                 <why> — the sound returns to its label time' = no steady phase for 5 s (film cadence, a rate\n"
        "                 conversion, no pictures) or whole-picture slips too close together (a frame-rate conversion;\n"
        "                 followed again 5 min after the last). Each at most once per 60 s for moves of 5 ms or more;\n"
        "                 smaller moves ('target a -> b') and 'house_skew +N ms with the opposite picture slip' (a slip\n"
        "                 shown as a repeat on an empty queue: no step in the sound; counted only while the phase is\n"
        "                 followed, and 'house_skew +N ms: the phase term is ... from its target already, the sound takes\n"
        "                 the step' when a second one comes before the first is back, T-516) under PTV_DIAG.\n"
        "                 PTV_NO_VPHASE=1 off\n"
        "  [PTV-SRCPER]   (2.2.2 T-512, PTV_DIAG) 'source picture period a -> b ms (N steps in 2 s: median, shortest)':\n"
        "                 the source's picture period as learned over the last 2 s of content (the median label step;\n"
        "                 it places the rounding line of a source faster than the house and is the period of\n"
        "                 [PTV-VPHASE]). Printed when it moved by 1 %% of a tick or the shortest step differs from the\n"
        "                 median (a label step: a segment join of an HLS-origin feed). PTV_SRCPER_MIN=1 = the shortest\n"
        "                 step, as before 2.2.2\n"
        "  [PTV-DUKF]     after a jump of 1 s or more in the video the decoder input waits for the next keyframe:\n"
        "                 'resume at keyframe — dropped N video frames' (normal, at most one GOP) | 'escape — no IDR\n"
        "                 within 5000ms' (resumed without one: a few damaged pictures may show)\n"
        "  [PTV-VDEC]     'the source changed to WxH ... scaled to fit' (a mid-stream size switch, taken in-process,\n"
        "                 each new size once) | 'N damaged picture(s) in another size dropped' | 'decoder error on\n"
        "                 output ... frame lost' | 'scaling ... failed' (ERROR). The first is normal on ABR sources\n"
        "  [PTV-VGRAPH]   (ERROR) the video filter graph took pictures and returned none: a filter failed, the\n"
        "                 process exits 4 for a re-probe\n"
        "  [PTV-QSHED]    the decoder cannot keep up: 'video_q full — shedding whole GOPs' / 'overflow: dropped GOP\n"
        "                 N pkts ... from the head' / 'tail-dropped ... to the IDR boundary' / 'tail escape ... without\n");
    av_log(NULL, AV_LOG_INFO,
        "                 IDR'. Pictures are lost each time: look at the GPU / CPU load of the box\n"
        "  [PTV-SELFHEAL] the frame queue stayed empty for tens of seconds although input flows: video_q and the decoder are\n"
        "                 flushed and decoding resumes at the next keyframe ('no IDR within 5s' = resumed mid-GOP).\n"
        "                 A decoder that lost its references; repeated = report it. PTV_NO_SELFHEAL disables\n");
    av_log(NULL, AV_LOG_INFO,
        "  [PTV-FRAMEQ]   'frame queue full (N pictures): dropping the oldest decoded pictures — a timestamp hole\n"
        "                 on the wire' then 'stopped dropping: N pictures dropped over X s'. The output did not\n"
        "                 take pictures (encoder / egress slow)\n"
        "  [PTV-FQBP]     the output took no picture for 2 s with pictures queued: the decoder's back-pressure\n"
        "                 is released until the queue drains (a guard against a decoder / output deadlock)\n"
        "  [PTV-DLV]      'delivery gate full (N packets): copied packet dropped' = a copied (passthrough)\n"
        "                 packet waited for its video longer than the gate holds (dlvdrop= on the stats line) —\n"
        "                 must stay 0; the line says how far the video is behind\n"
        "  [PTV-MUX]      WARNING 'dropped a packet stamped before the start' (start-up only, harmless). FATAL,\n"
        "                 each exits for a supervised respawn: 'MUXGUARD drop span 60s with no accepted pkt' (a\n"
        "                 stream's labels are dead) | 'egress dead 60s' | 'write failed' (the output is gone).\n"
        "                 WARNING 'output PMT version a \xe2\x86\x92 b: N copied track(s) take the source's language / page\n"
        "                 data' (after a [PTV-SRC] '... the output follows'; receivers re-read the PMT; 2.2.1, T-493)\n"
        "  [PTV-MUXGUARD] a packet with a backward dts dropped before the muxer ('remedy leak, channel survives')\n"
        "                 + a total per rung at exit. Any line is a label bug to report with the lines before it\n"
        "  [PTV-MUXTOL]   a mux write returned ENOMEM / EAGAIN (egress pressure): the packet is dropped, the\n"
        "                 channel survives; 60 s of it ends in the [PTV-MUX] egress-dead exit. PTV_NO_MUXTOL reverts\n"
        "  [PTV-DEMUX]    'stream #N (pid, type) appeared after open — ignored' (a PID born mid-run that is not\n"
        "                 mapped; normal) | 'AAC now N Hz ... labels inside a PES re-derived from the ADTS header'\n"
        "                 (the source switched its AAC sample rate)\n");
    av_log(NULL, AV_LOG_INFO,
        "multiview stats line — same head (frame/fps/time/dup/drop/dlvhold/dlvforced) + per slot:\n"
        "  inK:lipsync  (1.0.1-pre16.1, ALWAYS-ON, inside each inK: group) that slot's sensor R\n"
        "               (same sensor/sign as single-input: + = audio EARLY; multi-track slots\n"
        "               joined '|'); `--` = the slot slated / not flowing — itself the outage\n"
        "               signal; absent = the input has no sensed audio track. 2.2.3 (T-515): the video\n"
        "               side takes each picture once, where it is first shown (seeded from the slot's\n"
        "               realized lag), the sound side counts the sound waiting in the slot's delay line,\n"
        "               and a lag step is read when the picture takes it (the sound takes it first)\n"
        "  corr         (1.0.1-pre17: the mv corrector is ARMED) per-track cumulative trim,\n"
        "               aK:-prefixed (track→slot map = the startup [PTV-RSYNC] tracks: line);\n"
        "               absent while quiet. While ANY slot is black-slated no track may engage\n"
        "               ([PTV-RSCORR] HOLD 'sibling slate') — finding-1 defense-in-depth\n"
        "  [PTV-RSYNC] inK  (pre17; PTV_DIAG since 2.0.0, one round per 30s) per-slot soak summary:\n"
        "               R (as inK:lipsync), ev= slot edit ledger, sk= the sound's follow offset (not the\n"
        "               stats line's inK:sk picture lag), occ=\n"
        "               jitter depth, SLATED while black\n"
        "  [PTV-REOPEN] (pre17 fix round) live mv net input re-opened after a read error\n"
        "               (rw_timeout expiry included) — the slot auto-resumes as a fresh join;\n"
        "               retries forever with 1..5s backoff while the source is down ('still down' /\n"
        "               'came back with another stream layout, not taken'). A single input logs its\n"
        "               reopen as [PTV-SRC] in0 LOST ... (-reopen_backoff, default 1,2,5,10,30 s) and\n"
        "               [PTV-REOPEN] 're-opened after read failure (#N)' / 'read failed ... retrying' too\n"
        "  acor         (when >0) corrupt-discarded audio pkts, GLOBAL sum — per-track detail on\n"
        "               the [PTV-ADISC]/NBS log lines\n"
        "  inK:qdrop    input-K video queue overflow drops (demux side)\n"
        "  inK:corrupt  input-K corrupt packets (demux + decode)\n");
    av_log(NULL, AV_LOG_INFO,
        "  inK:pd       cadence repeats: the cell showed its picture again because the next one was not\n"
        "               due yet — a 25fps slot in a 29.97 mosaic repeats every 6th tick BY DESIGN (~5/s\n"
        "               is correct rate conversion, not a fault)\n"
        "  inK:sv       starvation repeats (no picture queued: the slot's input is late or gone)\n"
        "  inK:sk       (2.1) the slot's realized picture lag (ms, output - source on the slot's h0 axis);\n"
        "               its sound is placed with it. Moves only at a [PTV-MV] re-lock or a drift step.\n"
        "               NOT the slot's delay: it sums the content dropped (or repeated) so far, so a source\n"
        "               running fast or slow against the box clock walks it one picture per drift step\n"
        "               (+130 ppm = -0.47 s/h, by design). The delay is the slot's queue depth: PTV_DIAG\n"
        "               [PTV-RES] inK:occ= counts PICTURES, its tgt= house TICKS (a 59.94 slot reads ~2x tgt)\n"
        "  inK:skres    (0.9.18.7) slot LAYERA erase-residue ledger (ms) — 2.1: always 0; kept for parsers\n"
        "  inK:cc       (only on slots with -cc_extract) that slot's captions/a53 (as the single-input cc=)\n"
        "  not on the mosaic line: src= (except the start-up waiting heartbeat), hs=, cf=, unreg=, dlvdrop=\n");
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
        "  [PTV-AVSYNC]   per-track A/V controller telemetry: offset/avlag estimate, vlag/alag (a mosaic\n"
        "                 slot, 2.2.3 T-515: vlag = the slot's realized picture lag, alag = where its sound\n"
        "                 went, the delay line counted — offset is the sound's follow error),\n"
        "                 house_skew (2.1: a mosaic slot has no PLL — its sound follows its frame sync). 1.0.1-pre13:\n"
        "                 the estimate is printed as avlag= (was lipsync= — SIGN IS OPPOSITE to\n"
        "                 the stats-line lipsync= sensor: avlag>0 = audio LATE; lipsync>0 = audio\n"
        "                 EARLY. The token now appears only on the -stats progress line)\n"
        "  [PTV-SWRDELAY] aresample internal buffer occupancy (a latency LEVEL; `async` is the RATE)\n"
        "  [PTV-RSYNC]    (1.0.1-pre9) residual-sensor components per track: R + dm(m_v−m_a) +\n"
        "                 ev/ea (demux label-edit ledgers) + glue/hs (audio-injected offsets) +\n"
        "                 slip (un-realized resampler correction) − cal (the calibration set at the\n"
        "                 last sync check verdict, rc10). PASSIVE — nothing consumes R\n"
        "  [PTV-VMDOOR]   (PTV_DIAG, rc10) the record door places this track's sound where the video\n"
        "                 showed its content: ON / OFF, frames waiting, the sync check's steps so far\n");
    av_log(NULL, AV_LOG_INFO,
        "  [PTV-CHAIN]    A/V trace demux→output (rawA-V / srcA-V / unwrap_inj / outA-V) to localize\n"
        "                 where an A/V offset enters\n"
        "  [PTV-LIPSYNC]  (single input) per-track err = the ring-paired offset (picture vs sound, + = audio late);\n"
        "                 a mosaic slot's lip sync is its inK:lipsync on the mv stats line\n"
        "defaults (2.1): the program timeline (one classifier per input, the record door, mosaic frame\n"
        "  synchronizers) + WUCR occupancy pacing + REPRIME fast refill + ADAPTIVE cushion + delivery\n"
        "  gate (single & mosaic) + AUTO-BANK + clock-follow + cadence decimation are all ON — no env\n"
        "  needed for the production posture.\n"
        "  Reverts: PTV_NO_WUCR · PTV_NO_REPRIME · PTV_NO_ADAPTIVE · PTV_NO_AVLOCK ·\n"
        "  PTV_NO_DELIVERY_MV · PTV_NO_AUTOBANK · PTV_NO_CLOCKFOLLOW · PTV_NO_DECIMATE ·\n"
        "  PTV_NO_EXACTTICK (re-enables the integer-tick ~10ppm NTSC lip-sync drift; A/B only) ·\n"
        "  PTV_NO_PULLDOWN (revert telecine-aware emit: film segments back to dup-fill + hs sawtooth) ·\n"
        "  PTV_NO_RSYNC_CORR (residual-sync corrector off; 1.0.1-pre14, DEFAULT ON — parked and\n"
        "  byte-inert on a healthy channel; kill switch kept forever)\n");
    av_log(NULL, AV_LOG_INFO,
        "tuning: PTV_CUSHION_MS=N adaptive raised tier (default 4000, [1000,10000]) · PTV_CUSHION_MAX_MS=N\n"
        "  auto-bank ceiling (default 12000; beyond it = an upstream incident to surface) · PTV_FRAMEQ=N\n"
        "  frame_q capacity (default 160, [48,1024]) · PTV_PREROLL_MS=N startup cushion / base tier ·\n"
        "  PTV_DELIVERY_CAP_MS / PTV_DELIVERY_MAXQ delivery-gate sizing\n"
        "probes: PTV_DIAG=1 debug lines above · PTV_LOG_TS=1 prepend [timestamp]\n"
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
        /* 2.0.2 T-389: a single input whose source is STALLED has no content on either side — the reading is the
         * hold's own drift (Atlantico 2026-10-18: −0.4 … −2.9 s while stalled, sync_check PIN restarts on it); a
         * track being FILLED (afill=: no source audio, silence) has no audio content (live-transcoder Grid_2x2
         * slot 2 2026-10-18: the source dropped its audio PID, the filled track read a flat −237 ms) */
        if ((g_src_watch && atomic_load_explicit(&g_src_state, memory_order_relaxed)) ||
            atomic_load_explicit(&g_afill_us[ki], memory_order_relaxed) > 0)
            fresh = 0;
        if (force_idx || g_rsx.n_a > 1)
            nn += snprintf(buf + nn, size - nn, "%sa%d:", ki ? "," : "", ki);
        if (fresh) {                                 /* R = (m_v+E_v) − (m_a+E_a); stale side → -- (no stale anchors) */
            int64_t mv = atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ev_us[in],  memory_order_relaxed);
            int64_t ma = atomic_load_explicit(&g_rsx.ma_ema[ki], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ea_us[ki],  memory_order_relaxed)
                       - atomic_load_explicit(&g_rsx.eo_us[ki],  memory_order_relaxed);   /* T-480/T-486: an adopted
                                                                    * clock of its own is the stream's timing, not desync
                                                                    * (the corrector's R_us; rc5/rc6 read +10313001 ms
                                                                    * in sync on aflip60 — sync_check's PIN would restart) */
            ma += atomic_load_explicit(&g_rsx.cal_us[ki], memory_order_relaxed);   /* T-488: as the corrector reads it */
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
        fresh = mvw && maw && now_us - mvw < 3000000 && now_us - maw < 3000000 &&
                atomic_load_explicit(&g_afill_us[ki], memory_order_relaxed) <= 0;   /* 2.0.2: a filled track (above) */
        if (!first)
            nn += snprintf(buf + nn, size - nn, "|");
        first = 0;
        if (fresh) {                                 /* R = (m_v+E_v) − (m_a+E_a), as ptv_stats_lipsync */
            int64_t mv = atomic_load_explicit(&g_rsx.mv_ema[in], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ev_us[in],  memory_order_relaxed);
            int64_t ma = atomic_load_explicit(&g_rsx.ma_ema[ki], memory_order_relaxed)
                       + atomic_load_explicit(&g_rsx.ea_us[ki],  memory_order_relaxed)
                       - atomic_load_explicit(&g_rsx.eo_us[ki],  memory_order_relaxed);   /* T-480/T-486: an adopted
                                                                    * clock of its own is the stream's timing, not desync
                                                                    * (the corrector's R_us; rc5/rc6 read +10313001 ms
                                                                    * in sync on aflip60 — sync_check's PIN would restart) */
            nn += snprintf(buf + nn, size - nn, "%+lldms", (long long)((mv - ma) / 1000));
        } else
            nn += snprintf(buf + nn, size - nn, "--");
    }
}

/* corr= builder (pre14, moved verbatim). Called by BOTH printers since 1.0.1-pre17 (the mv
 * corrector is armed): force_idx=1 (mv) always prints the aK: prefix — the track→slot map
 * (startup [PTV-RSYNC] tracks: line) resolves aK to its input. Absent while every track
 * sits at corr==0 un-engaged — the quiet-channel stats line is unchanged (§6). */
/* conv= builder (1.0.1-pre23 rr23). On a fold/park channel the pairing lipsync= and the
 * v0.9.2 async= stats include the FOLDED label divergence by design (the fold refuses the
 * source's label motion; the wire is the ground truth) — a monitor banding on lipsync=
 * would alarm on exactly the channels the bounded-convergence fix rescues. This token makes
 * the state self-describing: cumulative net folded label motion in integer seconds, with a
 * `P` suffix while the track is SEAM-PARKED (timestamp-derived — never stale). Absent while
 * every track's fold total is 0, so the healthy-channel stats line stays byte-identical. */
void ptv_stats_conv(char *buf, size_t size, int64_t now_us, int force_idx)
{
    int any = 0, ki, nn;
    buf[0] = 0;
    if (!g_convcap || g_rsx.n_a <= 0)
        return;
    for (ki = 0; ki < g_rsx.n_a; ki++)
        if (atomic_load_explicit(&g_conv_pub[ki], memory_order_relaxed) != 0)
            any = 1;
    if (!any)
        return;
    nn = snprintf(buf, size, " conv=");
    for (ki = 0; ki < g_rsx.n_a && nn > 0 && nn < (int)size - 18; ki++) {
        int64_t cv = atomic_load_explicit(&g_conv_pub[ki], memory_order_relaxed);
        int64_t pu = atomic_load_explicit(&g_conv_park_pub[ki], memory_order_relaxed);
        if (force_idx || g_rsx.n_a > 1)
            nn += snprintf(buf + nn, size - nn, "%sa%d:", ki ? "," : "", ki);
        nn += snprintf(buf + nn, size - nn, "%+llds%s",
                       (long long)(cv / AV_TIME_BASE), (pu && now_us < pu) ? "P" : "");
    }
}

void ptv_stats_corr(char *buf, size_t size, int force_idx)
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
        if (force_idx || g_rsx.n_a > 1)
            nn += snprintf(buf + nn, size - nn, "%sa%d:", ki ? "," : "", ki);
        nn += snprintf(buf + nn, size - nn, "%+lldms%s",
                       (long long)(cu / 1000), st == PTV_CORR_ENGAGED ? "*" : "");
    }
}
