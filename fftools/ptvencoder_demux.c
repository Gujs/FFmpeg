#include <unistd.h>   /* _exit() — 2.0.0-pre5 reshaped-source exit, the pre26 wedge-free fatal path */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#include "libavutil/avutil.h"
#include "libavutil/avstring.h"
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

int64_t g_gap_min_us = 700000;            /* min wall-absence (us) to call a forward audio jump a real GAP when video did not also cross (internalized 0.9.18.7; was PTV_GAP_MIN_MS) */
static int64_t g_dukf_escape_us = 5000000;   /* force-resume if no IDR within this (internalized 0.9.18.7; was PTV_DUKF_ESCAPE_MS) */
/* v0.9.5 phase-2a — LONG-BASELINE rate. The guard (below) only CAPS a fooled estimate; the actual cure
 * is to make the estimate ACCURATE so house_skew never accumulates. The 3s sub-window aliases bursty
 * delivery (→ ±1000ppm noise the EMA walks). A longer window averages the bursts out → the recovered
 * rate ≈ the true source rate → the house clock matches the source → house_skew stays ~0. Env-tunable;
 * DEFAULTS = the old 3s/shift-6 path (byte-identical) so this only engages if the (internalized 0.9.18.7) window constant is
 * raised (the sandbox A/B turns it on; promote the default once validated). The slew clamp scales with
 * the window so the ppm/s slew rate is preserved. */
static int64_t         g_gl_window_us  = 3000000;    /* internalized 0.9.18.7 (was PTV_GENLOCK_WINDOW_MS): 3000ms */
static int             g_gl_ema_shift  = 6;          /* internalized 0.9.18.7 (was PTV_GENLOCK_EMA_SHIFT): 6 (α≈1/64) */
static int64_t         g_gl_max_q20    = 314;        /* internalized 0.9.18.7 (was PTV_GENLOCK_MAX_PPM): 300ppm (≈314 in Q20) */
static int64_t         g_gl_reject_q20 = 734;        /* internalized 0.9.18.7 (was PTV_GENLOCK_REJECT_PPM): 700ppm (≈734 in Q20). KEEP ≥ 2×MAX:
                                                      * the reject is RELATIVE to the (bounded) estimate, so the band must span
                                                      * the full ±MAX envelope twice — else `ema` pinned at one bound could
                                                      * reject the windows pulling it to the opposite bound (a stuck zone). */
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

/* Shift a SCTE-35 splice_info_section's pts_adjustment by offset_us (the same
 * offset demux_pass applies to the packet timestamps), so the wire
 * effective_splice_time = (splice_time + pts_adjustment) mod 2^33 rides the
 * house clock and a downstream splicer matches it against our output video PTS.
 * The muxer (mpegtsenc) writes the section verbatim, assuming this rebase ran. */
static void scte35_rebase_pts_adjustment(AVPacket *pkt, int64_t offset_us)
{
    const uint8_t *buf;
    uint8_t *wbuf;
    int len, section_length, ret;
    int64_t old_pts_adjustment, new_pts_adjustment, offset_90k;
    uint32_t crc;

    if (!offset_us || !pkt->data || pkt->size < 17)
        return; /* need at least header + 4-byte CRC */

    buf = pkt->data;
    len = pkt->size;

    if (buf[0] != 0xFC)         /* table_id must be splice_info_section */
        return;

    section_length = ((buf[1] & 0x0F) << 8) | buf[2];
    if (section_length + 3 > len)
        return;

    /* Read 33-bit pts_adjustment: byte 4 bit 0 (MSB) + bytes 5-8 (BE32). */
    old_pts_adjustment = ((int64_t)(buf[4] & 0x01) << 32) |
                         ((int64_t)buf[5] << 24) |
                         ((int64_t)buf[6] << 16) |
                         ((int64_t)buf[7] << 8)  |
                         buf[8];

    offset_90k = av_rescale(offset_us, 90000, AV_TIME_BASE);
    new_pts_adjustment = (old_pts_adjustment + offset_90k) & 0x1FFFFFFFFLL;

    ret = av_packet_make_writable(pkt);
    if (ret < 0)
        return;
    wbuf = pkt->data;

    /* Write 33-bit pts_adjustment back; preserve byte 4's top 7 bits
     * (encrypted_packet flag + encryption_algorithm). */
    wbuf[4] = (wbuf[4] & 0xFE) | ((new_pts_adjustment >> 32) & 0x01);
    wbuf[5] = (new_pts_adjustment >> 24) & 0xFF;
    wbuf[6] = (new_pts_adjustment >> 16) & 0xFF;
    wbuf[7] = (new_pts_adjustment >>  8) & 0xFF;
    wbuf[8] =  new_pts_adjustment        & 0xFF;

    /* Recompute the trailing CRC32 over the whole section. */
    crc = av_bswap32(av_crc(av_crc_get_table(AV_CRC_32_IEEE), -1,
                            wbuf, len - 4));
    wbuf[len - 4] = (crc >> 24) & 0xFF;
    wbuf[len - 3] = (crc >> 16) & 0xFF;
    wbuf[len - 2] = (crc >>  8) & 0xFF;
    wbuf[len - 1] =  crc        & 0xFF;

    av_log(NULL, AV_LOG_DEBUG,
           "ptvencoder: SCTE-35 pts_adjustment %"PRId64" -> %"PRId64" (%+.3fs)\n",
           old_pts_adjustment, new_pts_adjustment, (double)offset_90k / 90000.0);
}

/* demux_dispatch routes a post-unwrap packet to its queue; the timeline releases through it (ptv_demux_dispatch). */
static int demux_dispatch(DemuxArgs *d, AVPacket *out);

/* 1.0.1-pre9 residual-sensor label-edit ledger (PASSIVE — see RsyncSense in ptvencoder.h):
 * post every NON-WRAP label edit the demux makes to stream `s` (µs, the amount ADDED to the
 * stream's labels). Callers: the discontinuity self-rebase (−adj), the LAYERA flush persist
 * (+applied_offset), the pre5 retro-correct (+corr). Pure 2^33 wraps are deliberately NOT
 * posted (always genuine + shared; posting them would spike R by the wrap period during every
 * A-before-V wrap straddle). Demux thread only; published to g_rsx (pre16: EVERY input —
 * video keyed by this input's slot, audio by the GLOBAL track index) so the sensor's R sees
 * per-stream-UNEQUAL edits (the wrong-glue bake class) while shared edits cancel, per slot,
 * independently of every other slot. Ledger + publish are reporting-only — no control path
 * reads them. */
static void rsync_post_edit(DemuxArgs *d, int s, int64_t delta_us)
{
    int j;
    if (!g_rsync_sense || !d->edit_us || s < 0 || s >= (int)d->ifmt->nb_streams)
        return;
    d->edit_us[s] += delta_us;
    if (s == d->vstream) {
        if (d->rsync_slot >= 0 && d->rsync_slot < PTV_MAX_INPUT)
            atomic_store_explicit(&g_rsx.ev_us[d->rsync_slot], d->edit_us[s], memory_order_relaxed);
    } else
        /* pre16 latent-bug fix: key by the GLOBAL track index d->aglobal[j], not the
         * demux-LOCAL j (identity on single input; on mv input 1's first track would have
         * posted into ea_us[0] and corrupted another slot's reading). */
        for (j = 0; j < d->n_audio && j < PTV_MAX_AUDIO; j++)
            if (d->astream[j] == s &&
                d->aglobal[j] >= 0 && d->aglobal[j] < PTV_MAX_AUDIO)
                atomic_store_explicit(&g_rsx.ea_us[d->aglobal[j]], d->edit_us[s], memory_order_relaxed);
}

/* One silent frame of a copied AC-3 / E-AC-3 stream at ts (in_tb), routed like a real copy (delivery gate or mux_q).
 * Caller holds pass_lock and keeps ts above ps->last_dts. */
static void copy_emit_sil(DemuxArgs *d, PassStream *ps, int64_t ts)
{
    int64_t dropped = 0;
    int i;
    for (i = 0; i < d->n_out; i++) {
        AVPacket *c = av_packet_clone(ps->sil);
        if (!c) continue;
        c->pts = c->dts = ts;
        c->stream_index = ps->ost[i]->index;
        if (d->gate[i])
            dlv_enqueue(d->gate[i], c, av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q), 0);
        else
            demux_send(d->mux_q[i], c, d->drop, &dropped);
    }
    ps->last_dts = ts;
}

/* Copy-passthrough: route an input packet we don't transcode (extra audio, DVB
 * subtitle, data/SCTE-35) straight to the muxer, rebased onto the same h0 house
 * timeline the encoded streams use so everything stays in sync. Packets that
 * precede the anchor are dropped (exactly like audio_push). */
/* 2.1 T-409: vwait_drain() releases dense copies with pass_lock already held (its callers take it) — demux_pass_one() then
 * must not take it again (non-recursive: the master deadlocked on the first picture). Per thread: the demux thread and
 * the master both drain the queue. */
static _Thread_local int t_pass_held;
static void pass_lock_take(DemuxArgs *d) { if (!t_pass_held) pthread_mutex_lock(&d->pass_lock); }
static void pass_lock_give(DemuxArgs *d) { if (!t_pass_held) pthread_mutex_unlock(&d->pass_lock); }

static int demux_pass_one(DemuxArgs *d, AVPacket *out, int64_t vm_out_us)
{
    int pi, i;
    for (pi = 0; pi < d->n_pass; pi++) {
        int64_t h0, h0_tb, ref, hs_used = 0;
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
        if (g_avlock && d->house_skew) { /* WUCR-corrected: AVLOCK kept (copied streams ride house_skew so they stay matched to the dup-lagged video); ρ bounds house_skew so the dup-hop stays small */
            hs_used = *d->house_skew;
            h0_tb -= av_rescale_q(hs_used, AV_TIME_BASE_Q, d->pass[pi].in_tb);
        }
        if (vm_out_us != AV_NOPTS_VALUE) {
            /* 2.0.0 T-100: stamped where the video put this content (the video's record), not h0/house_skew */
            int64_t ts = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
            h0_tb = -av_rescale_q(vm_out_us - av_rescale_q(ts, d->pass[pi].in_tb, AV_TIME_BASE_Q),
                                  AV_TIME_BASE_Q, d->pass[pi].in_tb);   /* the net offset, as the old path's -h0_tb */
        }
        if (out->pts != AV_NOPTS_VALUE) out->pts -= h0_tb;
        if (out->dts != AV_NOPTS_VALUE) out->dts -= h0_tb;
        ref = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
        if (ref != AV_NOPTS_VALUE && ref < 0) { av_packet_free(&out); return 0; }  /* precedes anchor */
        /* SCTE-35: rebase the in-section pts_adjustment by the SAME net offset applied to this packet's
         * pts/dts so effective_splice_time = (splice_time + pts_adjustment) rides the OUTPUT timeline and a
         * downstream splicer matches it to the output video PTS. That net offset is (prog_off − h0): −h0_tb
         * here in demux_pass, PLUS the program-level discontinuity offset prog_off already added to this
         * DATA packet's container ts in demux_unwrap (P2 §7.1). Without the prog_off term the marker would
         * arrive at the right container time but point at the WRONG content PTS after an ad-break jump (the
         * 33-bit wrap is handled by the function's mod-2^33). */
        if (d->pass[pi].is_scte) {   /* pre20: not d->ifmt — vwait_drain reaches here from the master thread */
            int64_t adj_us = -av_rescale_q(h0_tb, d->pass[pi].in_tb, AV_TIME_BASE_Q);
            if (g_discont && g_prog_off)
                adj_us += av_rescale_q(vm_out_us != AV_NOPTS_VALUE ? d->vm_po : d->prog_off, d->pass[pi].in_tb, AV_TIME_BASE_Q);
            scte35_rebase_pts_adjustment(out, adj_us);
        }
        /* Monotonic-DTS guard for the copy path (final, after the SCTE rebase). The
         * house-skew rebase above (and source quirks / wrap edges) can nudge a copied
         * stream's dts backward between packets; the mpegts muxer rejects a backward
         * dts with EINVAL and the rung dies — this froze channels under the 50fps
         * field-rate dup-storm, and a rare dup-induced skew dip could trip it even at
         * the right rate. Clamp strictly increasing per copied stream, shifting pts by
         * the same amount so pts>=dts still holds. (On-wire SCTE-35 splice timing rides
         * pts_adjustment, not container dts, so a sub-ms nudge is invisible there.) */
        if (d->pass[pi].sil)                 /* 2.0.0-pre6 D17: the master's AC-3 fill shares this state */
            pass_lock_take(d);
        if (d->pass[pi].sil && out->dts != AV_NOPTS_VALUE) {
            PassStream *ps = &d->pass[pi];
            int64_t exp = ps->real_end;              /* where this track's next frame belongs */
            if (ps->start_fill && ps->fill_end != AV_NOPTS_VALUE && out->dts > ps->fill_end &&
                out->dts - ps->fill_end <= av_rescale_q(60000000, AV_TIME_BASE_Q, ps->in_tb)) {
                /* 2.0.0-pre9.6: the first real packet of a track absent since the start comes in ahead of the video
                 * output position the start fill follows (a start-up backlog: 6.5 s on the T-084 media) — the gap up
                 * to it is still the start fill's, bridge it */
                int64_t cur = ps->fill_end;
                for (; cur + ps->sil_dur / 2 < out->dts; cur += ps->sil_dur)
                    copy_emit_sil(d, ps, cur);
                ps->fill_end = cur;
            }
            if (ps->fill_end != AV_NOPTS_VALUE) {
                /* hand-back: real packets below the end of the replayed silence are dropped (a sub-frame gap
                 * is left), never re-stamped — bounded to 2 s of drops */
                if (out->dts < ps->fill_end &&
                    (int64_t)ps->drop_n * ps->sil_dur < av_rescale_q(2000000, AV_TIME_BASE_Q, ps->in_tb)) {
                    ps->drop_n++;
                    pass_lock_give(d);
                    av_packet_free(&out);
                    return 0;
                }
                if (ps->fill_n * ps->sil_dur >= av_rescale_q(3000000, AV_TIME_BASE_Q, ps->in_tb))
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] copy #%d (%s) hold fill ended: %d silent frames, "
                           "%d real packets dropped at hand-back\n", ps->in_index,
                           ps->cname,
                           ps->fill_n, ps->drop_n);
                exp = ps->fill_end;
                ps->fill_end = AV_NOPTS_VALUE;
                ps->fill_n = ps->drop_n = 0;
                if (pi < PTV_MAX_PASS)
                    atomic_store_explicit(&g_cfill_us[pi], 0, memory_order_relaxed);
            }
            /* 2.0.0-pre9.4 (T-083): the copied track stays continuous through damage. A real frame up to 1 s
             * BEHIND where the track already is (the overlap a damaged PES leaves: the parser cut frames from
             * corrupt data and ran ahead) is dropped — the transcoded track's pre9.3 rule; it used to be bumped
             * 1 tick past the previous one, stacking frames on one instant. A frame up to 3 s AHEAD (frames lost
             * or corrupt-discarded, or a house_skew dup step) gets the gap filled with silent frames first —
             * nothing real is ever dropped for it. Longer gaps: the hold fill / copy_gap_watch. Both windows widen
             * by the house_skew change since the last real frame: the copy rides it, so a house_skew drop (the
             * video leapt over content: 1.08 s → 0 measured under loss) is an overlap of exactly the content the
             * video skipped, and a rise is time the video spent repeating frames. */
            if (g_copy_gapfill && exp != AV_NOPTS_VALUE) {
                int64_t gap = out->dts - exp, half = ps->sil_dur / 2, now = av_gettime_relative();
                int64_t hsd = av_rescale_q(hs_used - ps->hs_real, AV_TIME_BASE_Q, ps->in_tb);
                /* ... and a frame more than half a frame behind what is already on the wire (last_dts: real or
                 * silent), up to 5 s: the monotonic guard below would stack it on one instant (Cinestar 50 % loss:
                 * a parked run released behind the fill — 127 frames on one pts) */
                int64_t wire_behind = ps->last_dts != AV_NOPTS_VALUE ? ps->last_dts + ps->sil_dur - out->dts : 0;
                if ((gap <= -half && gap >= -av_rescale_q(1000000, AV_TIME_BASE_Q, ps->in_tb) + FFMIN(hsd, 0)) ||
                    (wire_behind > half && wire_behind <= av_rescale_q(5000000, AV_TIME_BASE_Q, ps->in_tb))) {
                    ps->ovl_n++;
                    pass_lock_give(d);
                    av_packet_free(&out);
                    return 0;
                }
                if (gap > half && gap <= av_rescale_q(3000000, AV_TIME_BASE_Q, ps->in_tb) + FFMAX(hsd, 0)) {
                    int64_t cur = exp;
                    if (ps->last_dts != AV_NOPTS_VALUE && cur < ps->last_dts + ps->sil_dur)
                        cur = ps->last_dts + ps->sil_dur;   /* after the frame on the wire (2.0.0-pre9.4) */
                    for (; cur + half < out->dts; cur += ps->sil_dur, ps->gap_n++)
                        copy_emit_sil(d, ps, cur);
                }
                if ((ps->gap_n || ps->ovl_n) && now - ps->dmg_log_wc >= 10000000) {
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] copy #%d (%s) kept continuous: %d silent frames put into "
                           "gaps, %d overlapping frames dropped (PTV_NO_COPY_GAPFILL=1 disables)\n", ps->in_index,
                           ps->cname, ps->gap_n, ps->ovl_n);
                    ps->gap_n = ps->ovl_n = 0;
                    ps->dmg_log_wc = now;
                }
            }
            ps->arr_wc   = av_gettime_relative();
            ps->real_end = out->dts + (out->duration > 0 ? out->duration : ps->sil_dur);
            ps->start_fill = 0;
            ps->hs_real  = hs_used;
            ps->dup_real = atomic_load_explicit(&g_dup_out_us, memory_order_relaxed);
        }
        if (out->dts != AV_NOPTS_VALUE) {
            if (d->pass[pi].last_dts != AV_NOPTS_VALUE && out->dts <= d->pass[pi].last_dts && vm_out_us != AV_NOPTS_VALUE &&
                !d->pass[pi].is_scte) {
                /* SCTE-35 is always sent (T-335: two cues released on the same frame shared its stamp) */
                d->vm_late++;                    /* 2.0.0 T-100: stamped behind what is on the wire: its moment passed */
                if (d->pass[pi].sil)
                    pass_lock_give(d);
                av_packet_free(&out);
                return 0;
            }
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
            if (d->pass[pi].gated && d->gate[i] && (c->dts != AV_NOPTS_VALUE || c->pts != AV_NOPTS_VALUE)) {
                /* §7.5a: dense copied AC-3/MP2 → hold for the video front. block=0: the shared demux
                 * thread must never stall the whole input (preserves copy's drop-on-full for net). */
                int64_t ts = c->dts != AV_NOPTS_VALUE ? c->dts : c->pts;
                dlv_enqueue(d->gate[i], c, av_rescale_q(ts, d->pass[pi].in_tb, AV_TIME_BASE_Q), 0);
            } else
                demux_send(d->mux_q[i], c, d->drop, &d->pdrop);   /* sparse subs/data/SCTE-35 bypass */
        }
        if (d->pass[pi].sil)
            pass_lock_give(d);
        av_packet_free(&out);
        return 0;
    }
    av_packet_free(&out);
    return 0;
}

/* 2.0.0-pre6: stamp the parked runs whose wait is over, or 4 s, or force. A copy that was filled through a hold waits
 * until house_skew has its post-rejoin value (2.0.0-pre8.5, ptv_rj_settled — the transcoded rule); a plain pause with
 * video flowing still releases at the next fresh frame. */
static void park_drain(DemuxArgs *d, int force)
{
    int64_t fw = atomic_load_explicit(&g_src_fresh_wc, memory_order_acquire), now = av_gettime_relative();
    int pi, j;
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        int ready;
        if (!ps->npark)
            continue;
        pthread_mutex_lock(&d->pass_lock);
        if (ps->fill_end == AV_NOPTS_VALUE ||                                      /* plain pause, no fill */
            (!ptv_src_holding() &&                                                 /* filled, but no rejoin since */
             atomic_load_explicit(&g_src_rejoin_wall, memory_order_relaxed) <= ps->fill_wc0))
            ready = fw > ps->park_since;
        else {
            ready = ptv_rj_settled(ps->fill_cls_seq0, now);                        /* 2.0.0-pre8.5 */
            /* 2.0.0 T-324: ... and the video has put this run's content on air. Under heavy loss the rejoin is
             * classified by fragments mid-hold, so "settled" was already true at the return, 200 ms before the
             * master wrote the post-rejoin house_skew (it does so on the frame it records here): the first AC-3
             * packets rode the hold's skew, 21 s ahead of the video, and the real track sat behind them (Cinestar
             * 50 % loss: AC-3 holes of 9-22 s, 3 of 9 seeds) */
            if (ready && g_vmap) {
                AVPacket *p0 = ps->park[0];
                int64_t ts = p0->dts != AV_NOPTS_VALUE ? p0->dts : p0->pts, o, gap;
                ready = ts == AV_NOPTS_VALUE ||
                        ptv_vmap_lookup(av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q), &o, &gap) != 0;
            }
        }
        pthread_mutex_unlock(&d->pass_lock);
        if (!(force || ready || now - ps->park_since > 4000000))
            continue;
        for (j = 0; j < ps->npark; j++)
            demux_pass_one(d, ps->park[j], AV_NOPTS_VALUE);
        ps->npark = 0;
    }
}

/* Copy fan-out. 2.0.0-pre6: a dense copied audio stream that resumes after >=300 ms of silence waits (parked,
 * the demux thread keeps reading — it carries the video that ends the hold) until the master has emitted a
 * fresh frame after that resume; in steady flow that is the next tick. Single input. */
/* 2.0.0 T-100: release the sparse copies the video has shown, in arrival order per stream, stamped where the video
 * put their content. Called under pass_lock, by the demux thread on every packet and by the master output thread on
 * every fresh frame (the demux thread sleeps in the read during an outage). A subtitle / data packet whose content was
 * never shown (skipped, erased, lost in the outage) or whose moment on the output passed more than 1 s ago is dropped:
 * shown now it would sit off its picture. SCTE-35 is always sent (its splice point lies ahead). force = the queue is
 * full: what the video has not reached yet takes the old stamp. */
/* where the video showed content c_us: the single-input record, or (2.0.0 T-342) this multiview slot's own ring */
static int copy_shown(DemuxArgs *d, int64_t c_us, int64_t *out_us, int64_t *gap_us)
{
    return d->mv_vring ? vring_shown(d->mv_vring, c_us, out_us, gap_us) : ptv_vmap_lookup(c_us, out_us, gap_us);
}

static void vwait_drain(DemuxArgs *d, int force)
{
    int pi;
    t_pass_held = 1;                   /* every caller holds pass_lock */
    int64_t now = av_gettime_relative(), pos = atomic_load_explicit(&g_house_out_us, memory_order_relaxed);
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        int scte = ps->is_scte;   /* pre20: never d->ifmt here — the master thread calls this during a LOST reopen */
        int k = 0;
        while (k < ps->nvwait) {
            AVPacket *p = ps->vwait[k];
            int64_t ts = p->dts != AV_NOPTS_VALUE ? p->dts : p->pts, o = AV_NOPTS_VALUE, gap = 0;
            int r = ts == AV_NOPTS_VALUE ? -1 : copy_shown(d, av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q), &o, &gap);
            if (r == 0 && !force && !scte && !d->mv_vring && !ps->gated) {   /* (dense audio runs on through a video outage) */
                /* 2.0.0 T-378: a copy stamped far past the newest video the demux has READ is a corrupt timestamp — one
                 * bit-flipped DVB sub PTS (+1491 s, rich50b bitflip) waited for content that never came, and every
                 * subtitle behind it waited too (54 subs + 17 clears lost to the end of the run). A real copy is muxed
                 * with its video (a few seconds either way); the picture ON AIR is no reference: a 20 s burst puts real
                 * subtitles 29 s ahead of it (the first T-378 bound dropped 8 of them). Dropped; the stream flows on */
                if (d->vread_set && av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q) > d->vread_us + 10000000) {
                    if (now - d->vm_far_wc >= 60000000) {
                        d->vm_far_wc = now;
                        av_log(NULL, AV_LOG_WARNING, "[PTV-VMAP] a copied packet stamped %.1f s past the newest video "
                               "read — a corrupt timestamp, dropped (it would hold the stream's later packets)\n",
                               (av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q) - d->vread_us) / 1e6);
                    }
                    av_packet_free(&p);
                    d->vm_late++;
                    k++;
                    continue;
                }
            }
            if (r == 0 && !force && !(ps->gated && now - ps->vwait_wc[k] >= 1500000))
                break;                          /* its content is not on the output yet */
            if (ps->gated && r != 1) {
                /* a dense copy is never dropped for timing: not shown within 1.5 s (a video-only outage, a dead video
                 * PID) or older than the record — stamped the old way (house_skew), the track stays continuous */
                demux_pass_one(d, p, AV_NOPTS_VALUE);
                d->vm_fb++;
                k++;
                continue;
            }
            if (r < 0 && scte && !force && ts != AV_NOPTS_VALUE) {
                /* 2.0.0 T-316: copies now wait from before the anchor (a start-up burst); a cue whose content precedes
                 * the start was never on the output — dropped, as it was when it arrived before the anchor */
                int64_t h0;
                pthread_mutex_lock(d->h0_lock); h0 = *d->h0; pthread_mutex_unlock(d->h0_lock);
                if (h0 != AV_NOPTS_VALUE && av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q) < h0) {
                    av_packet_free(&p);
                    d->vm_late++;
                    k++;
                    continue;
                }
            }
            if (r < 0 && scte && !force) {
                /* 2.0.0 T-335: a cue with no usable time of its own (the PCR rides the video PID: none while only the
                 * video is gone, or a stale one) took the read-time stamp, +8.96 s off its splice picture at a
                 * video-only outage's return. Its splice point lies ahead: it goes out with the offset the video runs
                 * on once it is back, up to 12 s (the output delay pin), then the old way */
                int64_t ns, no;
                if (!d->mv_vring && ptv_vmap_newest(ps->vwait_wc[k], &ns, &no)) {
                    p->pts = p->dts = av_rescale_q(ns, AV_TIME_BASE_Q, ps->in_tb);
                    d->vm_po = d->prog_off;     /* the video's content is unwrapped with the current prog_off */
                    demux_pass_one(d, p, no);
                    d->vm_ok++;
                    k++;
                    continue;
                }
                if (now - ps->vwait_wc[k] < 12000000)
                    break;
            }
            if (r == 1 && ps->gated && gap > 1000000) {
                demux_pass_one(d, p, AV_NOPTS_VALUE);   /* sound the video did not show as content (its own outage):
                                                         * kept, stamped the old way */
                d->vm_fb++;
            } else if (r == 1 && !scte && !ps->gated && (gap > 1000000 || o < pos - 1000000)) {
                av_packet_free(&p);
                d->vm_late++;
            } else if (r == 1) {
                d->vm_po = ps->vwait_po[k];     /* the prog_off this packet was unwrapped with */
                demux_pass_one(d, p, o);
                d->vm_ok++;
            } else if (r < 0 && !scte) {
                av_packet_free(&p);             /* older than anything the video still records */
                d->vm_late++;
            } else {
                demux_pass_one(d, p, AV_NOPTS_VALUE);
                d->vm_fb++;
            }
            k++;
        }
        if (k) {
            memmove(ps->vwait, ps->vwait + k, (ps->nvwait - k) * sizeof(*ps->vwait));
            memmove(ps->vwait_wc, ps->vwait_wc + k, (ps->nvwait - k) * sizeof(*ps->vwait_wc));
            memmove(ps->vwait_po, ps->vwait_po + k, (ps->nvwait - k) * sizeof(*ps->vwait_po));
            memmove(ps->vwait_vn, ps->vwait_vn + k, (ps->nvwait - k) * sizeof(*ps->vwait_vn));
            ps->nvwait -= k;
        }
    }
    t_pass_held = 0;
    if ((d->vm_late || d->vm_fb) && now - d->vm_log_wc >= 60000000) {
        av_log(NULL, AV_LOG_WARNING, "[PTV-VMAP] %s: %"PRId64" stamped where the video showed their content, "
               "%"PRId64" dropped (content never shown, or its moment had passed), %"PRId64" stamped the old way (queue "
               "full, or a dense copy not shown in time) — PTV_NO_VMAP=1 disables\n",
               "copies", d->vm_ok, d->vm_late, d->vm_fb);
        d->vm_ok = d->vm_late = d->vm_fb = 0;
        d->vm_log_wc = now;
    }
}

void ptv_copy_vwait(DemuxArgs *d)
{
    pthread_mutex_lock(&d->pass_lock);
    vwait_drain(d, 0);
    pthread_mutex_unlock(&d->pass_lock);
}

static int demux_pass(DemuxArgs *d, AVPacket *out)
{
    int pi;
    park_drain(d, 0);
    if ((d->single || d->mv_vring) && g_vmap) {   /* 2.0.0 T-100: sparse copies wait for the video to show their content
                                                   * (T-342: on a multiview slot, its own cell) */
        pthread_mutex_lock(&d->pass_lock);
        vwait_drain(d, 0);
        for (pi = 0; pi < d->n_pass; pi++) {
            PassStream *ps = &d->pass[pi];
            /* 2.1 T-409: with the timeline the dense copies (AC-3 / MP2) wait too — the labels carry every kept hole
             * and erase, so house_skew at READ time is no longer the offset the video will show them with (a source
             * restart: +14 s while holding, -1.6 s after the rejoin map — the AC-3 landed behind its own hold fill and
             * was dropped for 15 s, gate sync_ac3_kill_return_20). See vwait_drain for their release rules */
            if ((ps->gated && !(ps->nvwait || ptv_tl_settling(d))) || out->stream_index != ps->in_index)
                continue;
            /* 2.0.0 T-316: queued from before the anchor too. A bursty source delivers its first 6-10 s at once,
             * before the first frame is decoded (h0): every subtitle / cue of that burst was dropped here while its
             * video was buffered and shown (rich bursty_6: DVB sub 4-6, SCTE-35 5, DVB clear 3 lost). Waiting, they
             * go out on their pictures; content before the start is still dropped (the record never shows it). */
            if (ps->nvwait == PTV_VWAIT_MAX)
                vwait_drain(d, 1);
            if (d->ifmt->streams[ps->in_index]->codecpar->codec_id == AV_CODEC_ID_SCTE_35 &&
                out->dts != AV_NOPTS_VALUE && d->n_audio > 0 && d->wrap_last[d->astream[0]] != AV_NOPTS_VALUE) {
                /* 2.0.0 T-345: a cue's time comes from the PCR, which rides the video PID: while only the video is gone
                 * it stays at the last PCR (every cue of a 20 s video outage read 30.700 s), so the cue was placed with
                 * the offset the video had before the outage — 2.07 s off its splice once the video came back on
                 * another one (rich sync_vnull_20, marker 50). More than 1 s behind the audio, which flows on: no
                 * time of its own — it waits for the video's offset like a cue without one (T-335) */
                const AVStream *as = d->ifmt->streams[d->astream[0]];
                int64_t at = av_rescale_q(d->wrap_last[d->astream[0]] + d->wrap_off[d->astream[0]], as->time_base, AV_TIME_BASE_Q);
                if (av_rescale_q(out->dts, ps->in_tb, AV_TIME_BASE_Q) < at - 1000000)
                    out->dts = out->pts = AV_NOPTS_VALUE;
            }
            ps->vwait_wc[ps->nvwait] = av_gettime_relative();
            ps->vwait_po[ps->nvwait] = d->prog_off;
            ps->vwait_vn[ps->nvwait] = d->vunwrap_n;
            ps->vwait[ps->nvwait++]  = out;
            pthread_mutex_unlock(&d->pass_lock);
            return 0;
        }
        pthread_mutex_unlock(&d->pass_lock);
    }
    if (d->single && g_src_watch) {
        for (pi = 0; pi < d->n_pass; pi++) {
            PassStream *ps = &d->pass[pi];
            int64_t now, prev;
            int unanchored, holding;
            if (!ps->gated || out->stream_index != ps->in_index)
                continue;
            now = av_gettime_relative();
            /* 2.0.0-pre9.4 (T-083): also park while the video is not anchored yet — demux_pass_one drops a copy
             * that arrives before h0 exists, and h0 is set at the first DECODED frame, ~1 s of input later: the
             * copied AC-3 started 0.92 s after the video (the transcoded track has its ring). The drain releases
             * at the first fresh frame; demux_pass_one then keeps what is at or after the anchor. */
            pthread_mutex_lock(d->h0_lock);
            unanchored = *d->h0 == AV_NOPTS_VALUE;
            pthread_mutex_unlock(d->h0_lock);
            /* 2.0.0-pre9.4 (T-083): while the source HOLDS and this copy is being filled, a real packet is a fragment
             * that survived the damage (the video is dead): stamped now it rode the hold's house_skew — 108 s ahead of
             * the output at a 107 s hold (Cinestar 50 % loss); the track's timeline stayed there and every real frame
             * after the rejoin sat behind it (bumped, held by the delivery gate: AC-3 absent for good). The fill
             * covers its time: drop it (the transcoded track's rule, audio_feed). */
            holding = ptv_src_holding();
            if (holding) {
                pthread_mutex_lock(&d->pass_lock);
                holding = ps->fill_end != AV_NOPTS_VALUE &&         /* ... and this copy is being filled */
                          /* 2.0.0 T-315: ... and the input has not come back as a BURST (a catch-up is real content) */
                          !ptv_rj_burst_since(ps->fill_cls_seq0);
                pthread_mutex_unlock(&d->pass_lock);
            }
            if (holding && !ps->npark) {   /* before in_wc: the copy's return must still read as a resume and park */
                av_packet_free(&out);
                return 0;
            }
            prev = ps->in_wc;
            ps->in_wc = now;
            if (ps->npark || (prev && now - prev >= 300000) || unanchored) {
                if (!ps->npark)
                    ps->park_since = now;
                if (ps->npark < PTV_PARK_MAX) {
                    ps->park[ps->npark++] = out;
                    return 0;
                }
                park_drain(d, 1);
            }
            break;
        }
    }
    return demux_pass_one(d, out, AV_NOPTS_VALUE);
}

/* 2.0.0-pre6 (T-056 §5.1, D17), called by the master output thread while video repeats: a copied AC-3 /
 * E-AC-3 track whose PID has been silent 300 ms gets its pre-encoded silent frame, one per frame duration,
 * from the end of its last real packet up to that end + the output time on ticks whose content did not advance since
 * that packet was stamped (pre9.4; was the house_skew grown since — the same rule as the transcoded fill), so the silence
 * stays on the held video (never beyond the video front plus the lead it had). Routed like a real copy (delivery gate or mux_q). */
void ptv_copy_fill(DemuxArgs *d, int64_t now)
{
    int pi;

    pthread_mutex_lock(&d->pass_lock);
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        int64_t target, cur;
        int n = 0;
        if (!ps->sil || ps->real_end == AV_NOPTS_VALUE || now - ps->arr_wc < 300000)
            continue;
        /* 2.0.0-pre9.4 (T-083): the dup clock, not house_skew — see hold_fill_quantum() */
        target = ps->real_end + av_rescale_q(atomic_load_explicit(&g_dup_out_us, memory_order_relaxed) -
                                             ps->dup_real, AV_TIME_BASE_Q, ps->in_tb);
        cur    = ps->fill_end != AV_NOPTS_VALUE ? ps->fill_end : ps->real_end;
        if (ps->last_dts != AV_NOPTS_VALUE && cur < ps->last_dts + ps->sil_dur)
            cur = ps->last_dts + ps->sil_dur;   /* 2.0.0-pre9.4: after the frame on the wire, not 1 tick into it */
        while (cur + ps->sil_dur <= target && n < 100) {   /* at most 100 frames (~3 s) per call */
            copy_emit_sil(d, ps, cur);
            cur += ps->sil_dur;
            n++;
        }
        if (n) {
            if (!ps->fill_n) {   /* 2.0.0-pre8.5: the rejoin that ends this fill is the next classification */
                ps->fill_cls_seq0 = atomic_load_explicit(&g_rj_cls_seq, memory_order_acquire);
                ps->fill_wc0      = now;
            }
            if (ps->fill_n * ps->sil_dur < av_rescale_q(3000000, AV_TIME_BASE_Q, ps->in_tb) &&
                (ps->fill_n + n) * ps->sil_dur >= av_rescale_q(3000000, AV_TIME_BASE_Q, ps->in_tb))
                av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] copy #%d (%s) hold fill: silent frames aligned to the "
                       "held video (PTV_NO_SRC_FILL=1 disables)\n", ps->in_index,
                       ps->cname);
            ps->fill_n  += n;
            ps->fill_end = cur;
            atomic_store_explicit(&g_cfill_idx[pi], ps->in_index, memory_order_relaxed);   /* 2.0.0-pre8: afill= */
            atomic_store_explicit(&g_cfill_us[pi], av_rescale_q((int64_t)ps->fill_n * ps->sil_dur, ps->in_tb,
                                                                AV_TIME_BASE_Q), memory_order_relaxed);
        }
    }
    pthread_mutex_unlock(&d->pass_lock);
}

/* 2.0.0-pre9.4 (T-083), demux thread on a video arrival: the source is delivering but a copied AC-3 / E-AC-3 PID has
 * delivered nothing usable (lost or corrupt-discarded) for 2 s of a gap-free video run — the transcoded NBS rule and
 * threshold (1-2 s interleave blips are normal). Silent frames follow the wall clock from the track's last real end,
 * 200 ms short of now so a resuming real frame meets the existing hand-back, which drops real frames below the fill.
 * A bursty source stalls every PID together, so it never fills here. Long holes only: shorter ones are filled when
 * the next real frame arrives (demux_pass_one). */
static void copy_gap_watch(DemuxArgs *d, int64_t now)
{
    int pi;
    int64_t h0;
    pthread_mutex_lock(d->h0_lock); h0 = *d->h0; pthread_mutex_unlock(d->h0_lock);
    pthread_mutex_lock(&d->pass_lock);
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        int64_t target, cur;
        int n = 0;
        /* 2.0.0-pre9.6 (T-084 rest / T-083): a copied track that has delivered NOTHING since the start (T-084: its
         * PID absent on a damaged source — 5 of 8 segments without audio, and the CDN-side segmenter could not even
         * read the stream's parameters) gets silent frames from the video's start, under the same 2 s rule; its
         * first real packet takes over through the existing hand-back. */
        if (ps->sil && ps->real_end == AV_NOPTS_VALUE && h0 != AV_NOPTS_VALUE) {
            if (!ps->absent_wc)
                ps->absent_wc = now;
            else if (now - ps->absent_wc >= 2000000) {
                ps->real_end   = 0;            /* output timeline: 0 = the video anchor */
                ps->arr_wc     = ps->absent_wc;
                ps->start_fill = 1;
                ps->hs_real  = 0;
                ps->dup_real = atomic_load_explicit(&g_dup_out_us, memory_order_relaxed);
                av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] copy #%d (%s) absent since the start — silent frames from the "
                       "video's start until its first packet\n", ps->in_index,
                       ps->cname);
            }
        }
        if (!ps->sil || ps->real_end == AV_NOPTS_VALUE || now - FFMAX(ps->arr_wc, d->v_flow_since_us) < 1000000)
            continue;
        /* 2.0.1 T-099: 1 s while the input is DAMAGED (corrupt video, or a corrupt-discarded packet on a transcoded
         * track, within 5 s), the T-379 rule of the transcoded track; clean sources keep 2 s. Measured (Cinestar
         * capture, 10 % loss, VT prod graph): the copied AC-3 PID dark 1.28 s at the onset while the AAC (1 s rule)
         * was 0.47 s and the video 0.35 s. */
        if (now - FFMAX(ps->arr_wc, d->v_flow_since_us) < 2000000) {
            int64_t dw = atomic_load_explicit(&g_vcorrupt_wc, memory_order_relaxed);
            int k;
            for (k = 0; k < d->n_audio; k++)
                if (d->aglobal[k] >= 0 && d->aglobal[k] < PTV_MAX_AUDIO)
                    dw = FFMAX(dw, atomic_load_explicit(&g_adamage_wc[d->aglobal[k]], memory_order_relaxed));
            if (!dw || now - dw >= 5000000)
                continue;
        }
        target = ps->real_end + av_rescale_q(now - ps->arr_wc - 200000, AV_TIME_BASE_Q, ps->in_tb);
        if (ps->start_fill)   /* the output timeline, not the wall: a long start-up probe leaves a backlog that drains
                               * ahead of the wall (T-084 media: 7.4 s) — the video's output position, 200 ms short */
            target = av_rescale_q(atomic_load_explicit(&g_house_out_us, memory_order_relaxed) - 200000,
                                  AV_TIME_BASE_Q, ps->in_tb);
        cur    = ps->fill_end != AV_NOPTS_VALUE ? ps->fill_end : ps->real_end;
        if (ps->last_dts != AV_NOPTS_VALUE && cur < ps->last_dts + ps->sil_dur)
            cur = ps->last_dts + ps->sil_dur;   /* 2.0.0-pre9.4: after the frame on the wire, not 1 tick into it */
        for (; cur + ps->sil_dur <= target && n < 100; cur += ps->sil_dur, n++)
            copy_emit_sil(d, ps, cur);
        if (!n)
            continue;
        if (!ps->fill_n) {
            ps->fill_cls_seq0 = atomic_load_explicit(&g_rj_cls_seq, memory_order_acquire);
            ps->fill_wc0      = now;
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] copy #%d (%s) gap fill: no usable packets for %.1f s while video "
                   "flows — silent frames (PTV_NO_COPY_GAPFILL=1 disables)\n", ps->in_index,
                   ps->cname, (now - FFMAX(ps->arr_wc, d->v_flow_since_us)) / 1e6);
        }
        ps->fill_n  += n;
        ps->fill_end = cur;
        atomic_store_explicit(&g_cfill_idx[pi], ps->in_index, memory_order_relaxed);
        atomic_store_explicit(&g_cfill_us[pi], av_rescale_q((int64_t)ps->fill_n * ps->sil_dur, ps->in_tb,
                                                            AV_TIME_BASE_Q), memory_order_relaxed);
    }
    pthread_mutex_unlock(&d->pass_lock);
}

/* 2.0.1 T-332: a video packet comes out of lavf two packets after its PES started (the PES completes when the next one
 * starts, the h264 parser then holds one frame more), so a subtitle / cue that follows a video PES in the mux is read
 * BEFORE that video packet. When that video packet moves prog_off (a source step absorbed on it), the copies read in
 * between were unwrapped with the old prog_off although they belong to the new timeline. Measured (rich2997b
 * sync_bstep, step at content 45 s): the stepped video PES starts at TS packet 59885; the DVB sub at 59913 and the
 * SCTE-35 cue next to it were read before it (and before the old-timeline frame at 59840) and took prog_off 10800
 * instead of 21600: both −120 ms on the output. The copies still waiting for their picture that were read since the
 * video packet before the previous one and lie after this PES in the mux take the step too (a section has no pos:
 * it is read at its own packet, inside that window). */
/* 2.1 T-409: a SCTE-35 cue's effective splice time (splice_time + pts_adjustment, 90 kHz, 33 bit) — time_signal and
 * program splice_insert with a specified time; -1 = none (immediate, component mode, cancel, malformed) */
static int64_t scte35_splice_90k(const AVPacket *p)
{
    const uint8_t *b = p->data;
    int64_t adj, t;
    int type, o;
    if (!b || p->size < 20 || b[0] != 0xFC || (b[4] & 0x80))   /* encrypted: unreadable */
        return -1;
    adj  = ((int64_t)(b[4] & 1) << 32) | ((int64_t)b[5] << 24) | (b[6] << 16) | (b[7] << 8) | b[8];
    type = b[13];
    if (type == 0x06)
        o = 14;                                       /* time_signal: splice_time() */
    else if (type == 0x05) {
        if (p->size < 21 || (b[18] & 0x80))            /* splice_event_cancel_indicator */
            return -1;
        if (!(b[19] & 0x40) || (b[19] & 0x10))         /* component mode / splice_immediate */
            return -1;
        o = 20;
    } else
        return -1;
    if (p->size < o + 5 || !(b[o] & 0x80))             /* time_specified_flag */
        return -1;
    t = ((int64_t)(b[o] & 1) << 32) | ((int64_t)b[o + 1] << 24) | (b[o + 2] << 16) | (b[o + 3] << 8) | b[o + 4];
    return (t + adj) & 0x1FFFFFFFFLL;
}

/* 2.1 T-409: the timeline moved the program offset by dpo (90 kHz) at a picture boundary of source time from_us. Copies
 * queued for the video whose SOURCE time (label minus the prog_off they were unwrapped with) lies at or after it belong
 * to the new content — subtitles and cues are muxed ahead of their pictures, so they were read with the old offset
 * (gate sync_bstep: SCTE-35 / DVB subs 120 ms early after a -120 ms step erase). */
void ptv_vwait_rebase(DemuxArgs *d, int64_t from_us, int64_t old_end_us, int64_t open_wc, int64_t dpo)
{
    int pi, k;
    if (!dpo)
        return;
    pthread_mutex_lock(&d->pass_lock);
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        for (k = 0; k < ps->nvwait; k++) {
            AVPacket *p = ps->vwait[k];
            int64_t ts = p->dts != AV_NOPTS_VALUE ? p->dts : p->pts, src;
            if (ts == AV_NOPTS_VALUE)
                continue;
            src = av_rescale_q(ts, ps->in_tb, AV_TIME_BASE_Q) -
                  av_rescale_q(ps->vwait_po[k], (AVRational){ 1, 90000 }, AV_TIME_BASE_Q);
            if (ps->is_scte) {
                /* a cue belongs where it splices, not where its packet sits: cues lead their splice (the pre-roll),
                 * so one read before the step can splice into the new content (sync_bstep: cues -120 ms) */
                int64_t sp = scte35_splice_90k(p), raw90 = av_rescale_q(src, AV_TIME_BASE_Q, (AVRational){ 1, 90000 });
                if (sp >= 0) {
                    int64_t dlt = (sp - (raw90 & 0x1FFFFFFFFLL)) & 0x1FFFFFFFFLL;
                    if (dlt >= (1LL << 32))
                        dlt -= 1LL << 33;               /* nearest: the splice within +-13 h of its packet */
                    src += av_rescale_q(dlt, (AVRational){ 1, 90000 }, AV_TIME_BASE_Q);
                }
            }
            /* a step back overlaps the old content's source time [from, old_end): there the read order decides — read
             * after the step began (the window opened) = the new content (sync_bstep: DVB subs -120 ms) */
            if (src < from_us || (src < old_end_us && ps->vwait_wc[k] < open_wc))
                continue;
            if (p->pts != AV_NOPTS_VALUE) p->pts += av_rescale_q(dpo, (AVRational){ 1, 90000 }, ps->in_tb);
            if (p->dts != AV_NOPTS_VALUE) p->dts += av_rescale_q(dpo, (AVRational){ 1, 90000 }, ps->in_tb);
            ps->vwait_po[k] += dpo;
        }
    }
    pthread_mutex_unlock(&d->pass_lock);
}

static void vwait_follow_video(DemuxArgs *d, int64_t vpos, int64_t dpo)
{
    int pi, k;
    pthread_mutex_lock(&d->pass_lock);
    for (pi = 0; pi < d->n_pass; pi++) {
        PassStream *ps = &d->pass[pi];
        for (k = ps->nvwait - 1; k >= 0 && ps->vwait_vn[k] >= d->vunwrap_n - 1; k--) {
            AVPacket *p = ps->vwait[k];
            if (p->pos >= 0 && vpos >= 0 && p->pos < vpos)
                continue;
            if (p->pts != AV_NOPTS_VALUE) p->pts += dpo;
            if (p->dts != AV_NOPTS_VALUE) p->dts += dpo;
            ps->vwait_po[k] += dpo;
        }
    }
    pthread_mutex_unlock(&d->pass_lock);
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
    int64_t po0 = d->prog_off;               /* 2.0.1 T-332: what this packet moves prog_off by */
    int dmg = 0;

    if (bits <= 0 || bits >= 63)              /* only meaningful for a real TS wrap */
        return;
    mask = 1LL << bits;
    half = mask >> 1;
    raw  = pkt->dts != AV_NOPTS_VALUE ? pkt->dts : pkt->pts;   /* DTS = decode order = monotonic (B-frame-safe) */
    if (raw != AV_NOPTS_VALUE) {
        int64_t wall_now = av_gettime_relative();
        int64_t last = d->wrap_last[pkt->stream_index];
        /* 2.0.2 T-402: the stream is in a damage phase (a corrupt packet of it discarded in the last 3 s, the T-350 window):
         * its delivered packets come out of a mangled PES stream — no "normal step" / delivery-cadence learning from them
         * and no small label edit on their evidence (see the step EMA and the video absorber below) */
        if (g_dmgstat && d->ts_outl && pkt->stream_index < d->nb_streams_open) {
            int64_t dw = d->ts_outl[pkt->stream_index].dmg_wc;
            dmg = dw && wall_now - dw < 3000000;
        }
        if (d->any_wall_last && wall_now - d->any_wall_last >= g_gap_min_us)
            d->any_stall_wc = wall_now;         /* 2.0.0 T-345: the whole input paused (delivery, not one PID) */
        d->any_wall_last = wall_now;
        /* pre24 #63: preserve THIS packet's pre-update delivery wall gap and fold quiet gaps
         * into the per-stream cadence EMA (burst-period reading — the rr15 R2 door pattern).
         * The LAYERA loop runs AFTER wrap_wall_last below has advanced to this packet, so the
         * wall-evidence sites there read the preserved copy. Dense streams only. */
        if (d->pkt_wall_gap_us &&
            (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO ||
             st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)) {
            int64_t pwl = d->wrap_wall_last[pkt->stream_index];
            int64_t pg  = pwl > 0 ? wall_now - pwl : 0;
            d->pkt_wall_gap_us[pkt->stream_index] = pg;
            /* T-327: a video pause (> 500 ms with no video packet) is damage evidence for the output clock: a content gap
             * behind it is content lost in the pause, filled at tick rate like a loss gap, not leapt. The post-gap frames
             * come out of the decoder right behind the pre-gap tail (lavf PES tail, decoder delay), so a pause measured
             * at the decoder output misses them; measured here it covers them (emit_video tags frames for 5 s). Flapping
             * at 50 fps: the one return whose cut carried no corrupt PES was leapt — a 1.68 s video PTS hole. Not after
             * a hold (STALLED): that return has its own rejoin mapping (kill_return_20 at 50 fps read 0.86 live share
             * on 3 of 3 runs with it, the control 0.86 / 0.89 / PASS). */
            if (pg > 500000 && st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && !ptv_src_holding())
                atomic_store_explicit(&g_vdmg_wc, wall_now, memory_order_relaxed);
            /* 2.0.0-pre4c PES-TAIL carry (measured 2026-10-03): lavf completes a video PES only when
             * the NEXT one starts (unbounded PES length), so after a real outage the first video
             * packet out is the pre-gap TAIL frame — it carries the wall gap but no content jump —
             * and the content jump arrives one packet later with a ~1-frame wall gap. The E1v video
             * wall-evidence rule then saw a "flowing" jump and LAYERA erased the hole while the
             * audio leg (bounded PES: jump and wall gap on the same packet) was padded: the output
             * stayed audio-late by the whole outage. The packet right after a gapped video packet
             * inherits that gap for the wall-evidence test (pkt_wall_gap_us only; cadence unchanged).
             * PTV_NO_PESTAIL=1 reverts. */
            /* Measured on audio too: a cut that splits an audio PES makes lavf complete it with the
             * first post-gap bytes, so the same tail carries the gap there (fixture cut at 20.000 s:
             * video kept its hole, audio erased → audio 10 s early). Per dense stream. */
            if (g_pestail && d->tail_gap_us && pkt->stream_index < d->nb_streams_open) {
                int64_t *tg = &d->tail_gap_us[pkt->stream_index];
                /* 2.0.1 T-381 (measured, 50 % loss): only a FRESH tail (< 1 s, the T-089 bound) carries. A tail whose
                 * next packet came after its own 3.6 s gap passed it the old 4.9 s and was cleared, so this packet's
                 * gap was never recorded as a tail and the content jump right behind it (same read) read 0 ms:
                 * W = 0, LAYERA erased a real 1.45 s audio hole — sound 1.43 s early for good. */
                if (*tg && pg < *tg &&
                    (!d->tail_wc_us || wall_now - d->tail_wc_us[pkt->stream_index] < 1000000)) {
                    d->pkt_wall_gap_us[pkt->stream_index] = *tg + pg;
                    /* 2.0.0 T-089 (measured, MV slot rejoin): a cut can leave TWO pre-gap tail frames — the second
                     * one took the carry and the jump packet after it read a 60 ms gap: video W = 0, the flush
                     * refused, video erased the hole while audio kept it (audio 51 s late). Packets still on the
                     * old timeline (forward, under the 1 s jump bar) within 1 s of the gap pass the carry on. */
                    if (d->tail_wc_us && wall_now - d->tail_wc_us[pkt->stream_index] < 1000000 &&
                        last != AV_NOPTS_VALUE && raw > last &&
                        av_rescale_q(raw - last, st->time_base, AV_TIME_BASE_Q) <= PTV_DISC_THRESHOLD_US)
                        *tg += pg;
                    else
                        *tg = 0;
                } else {
                    *tg = pg >= g_gap_min_us ? pg : 0;
                    if (*tg && d->tail_wc_us)
                        d->tail_wc_us[pkt->stream_index] = wall_now;
                }
            }
            if (d->wall_cad_us && pg > 5000 && pg < 2000000 && !dmg) {
                if (!d->wall_cad_us[pkt->stream_index])
                    d->wall_cad_us[pkt->stream_index] = pg;
                else
                    d->wall_cad_us[pkt->stream_index] +=
                        (pg - d->wall_cad_us[pkt->stream_index]) / 8;
            }
        }
        if (last != AV_NOPTS_VALUE) {
            int64_t delta = raw - last;
            /* 2.0.0-pre9.2 (T-080 B): a one-packet timestamp outlier. PES headers carry no CRC, so one upstream bit
             * error makes ONE packet's DTS lie. Measured (sync_bitflip_100): a video DTS 142 ms behind its predecessor was
             * absorbed below as a backward source step (+182 ms on every later packet); the next packet was back on the
             * old timeline (+2 frames after the one before the outlier), read +182 ms "ahead", under the 1 s forward
             * threshold — nothing undid it: A/V −200 ms for good. A real step continues on its new timeline. */
            PtvTsOutlier *ol = d->ts_outl && pkt->stream_index < d->nb_streams_open ? &d->ts_outl[pkt->stream_index] : NULL;
            /* 2.0.1 T-329: the check above ran on the NEXT packet only, but lavf splits one PES into all of its frames
             * and gives every later one the corrupt label + n frames — an AAC PES carries 8 (170 ms). Measured
             * (rich2997ru, one audio PES PTS −91 ms = one flipped bit 13): −70 ms absorbed, the second frame read
             * −48 ms from the pre-step label and disarmed the revert, the next PES (back on the old timeline) stepped
             * +91 ms under the 1 s forward bar, [PTV-AGLUE] GAP-padded it: audio +91 ms for good. Stay armed while the
             * stream keeps flowing on the outlier's timeline (16 packets, the T-088 window); revert when a packet lands
             * on the pre-step timeline projected to it, within one frame. */
            if (ol && ol->pend_last != AV_NOPTS_VALUE) {
                int64_t from_pre = raw - ol->pend_last;
                int keep = 0;
                ol->pend_n++;
                if (ol->nom > 0 && from_pre > 0 &&
                    (ol->pend_n == 1 ? from_pre <= 3 * ol->nom
                                     : llabs(from_pre - (ol->pend_n + 1) * ol->nom) <= ol->nom)) {   /* back on the pre-step timeline */
                    d->wrap_off[pkt->stream_index] -= ol->pend_dw;
                    d->prog_off -= ol->pend_dprog;
                    d->splice_adj_us = 0;                    /* no sibling may adopt the bogus amount */
                    rsync_post_edit(d, pkt->stream_index, -av_rescale_q(ol->pend_dw, st->time_base, AV_TIME_BASE_Q));
                    {   /* 2.0.1 T-087: a pending relabel check of this erase is void */
                        int j;
                        for (j = 0; j < d->n_audio; j++)
                            if (d->astream[j] == pkt->stream_index && d->aglobal[j] >= 0 && d->aglobal[j] < PTV_MAX_AUDIO)
                                atomic_store_explicit(&g_rl_undone[d->aglobal[j]],
                                    atomic_load_explicit(&g_rl_seq[d->aglobal[j]], memory_order_relaxed), memory_order_relaxed);
                    }
                    av_log(NULL, AV_LOG_WARNING, "[PTV-DISCONT] stream %d: the absorbed %+"PRId64"ms step was ONE corrupt "
                           "timestamp (the next packet is back on the old timeline) — absorption reverted\n",
                           pkt->stream_index, -av_rescale_q(ol->pend_dw, st->time_base, (AVRational){1,1000}));
                } else if (ol->nom > 0 && ol->pend_n < 16 && raw - last > 0 && raw - last <= 2 * ol->nom)
                    keep = 1;                                /* the outlier PES's next frame */
                if (!keep)
                    ol->pend_last = AV_NOPTS_VALUE;
            }
            /* 2.0.2 T-402: not across corrupt-discarded packets. In a corruption storm (live RAV slot 2, 2026-10-19: every
             * video PES corrupt for 50 min, the few intact ones 2-3 frames apart) the steps between delivered packets
             * spanned the discarded frames and walked the "normal step" from 33 to ~87 ms (each step up to twice it is
             * learned) — then a repeated label (a PES lavf emits twice) read as an 87 ms backward step, over the 80 ms bar,
             * and was absorbed: +87 ms on every later video label, ~21 times, while the all-discarded audio kept its
             * labels; the delivery cadence (above) walked to ~850 ms the same way and a 2168 ms hole lost 792 ms as
             * "flowing". Net video +1092 ms vs audio 0 = the slot's sound 1.1 s early after the storm on both canaries
             * (the edit ledgers ev / ea, content -1130 ms). PTV_NO_DMGSTAT=1 reverts. */
            if (ol && delta > 0 && delta <= av_rescale(250, st->time_base.den, (int64_t)st->time_base.num * 1000) &&
                (!ol->nom || delta <= 2 * ol->nom) && !dmg)   /* T-088: an excursion is not a normal step */
                ol->nom = ol->nom ? ol->nom + (delta - ol->nom) / 8 : delta;   /* the normal step */
            /* 2.0.0 T-088: remember a forward excursion under the absorber's 1 s bar (it flows), for 16 packets — a
             * corrupt PES header shifts every frame lavf splits out of that PES (AC-3: three), so the return to the
             * old timeline comes several packets later, not on the next one */
            if (ol && ol->nom > 0 && delta > 2 * ol->nom &&
                av_rescale_q(delta, st->time_base, AV_TIME_BASE_Q) <= PTV_DISC_THRESHOLD_US) {
                ol->fx_base = last;
                ol->fx_n    = 1;
            } else if (ol && ol->fx_n)
                ol->fx_n = ol->fx_n < 16 ? ol->fx_n + 1 : 0;
            /* v0.9.16.1 sparse-PID wrap guard: past HALF the wrap period (13.26h @90kHz) of wall
             * silence, the ±half delta heuristic ALIASES both ways — a no-wrap gap >13.26h reads
             * as "late pre-roll" (−2^33 → the PID lands 26.5h in the past and demux_pass drops it
             * FOREVER), and ≥1 wraps crossed during the silence read as small deltas (the +2^33
             * is missed → same landing). SCTE-35 quiet overnight/weekend is the real-world case.
             * Below the threshold the delta branches are provably always right (gap G<13.26h ⇒
             * a genuine wrap gives delta<−half, no wrap gives |delta|<half), so nothing changes
             * for normal operation. Fix: re-anchor by WALL PROJECTION — choose the wrap count
             * that lands the packet nearest its wall-expected position (a live mux stamps a
             * resuming PID with the CURRENT STC, so projection is exact up to clock ppm ≪ half).
             * PTV_WRAP_GUARD_S overrides the threshold (test only). */
            int64_t wl        = d->wrap_wall_last[pkt->stream_index];
            int64_t period_us = av_rescale(mask, (int64_t)st->time_base.num * 1000000, st->time_base.den);
            int64_t guard_us  = g_wrap_guard_us > 0 ? g_wrap_guard_us : period_us / 2;
            /* 2.0.1 T-320: the wrap branches below read the delta to `last`, so a one-packet outlier near HALF the wrap
             * takes a wrap step (its delta just past half) that its return does not undo (that delta just inside half).
             * Measured (rich25b, one video PES PTS bit 32 flipped = +2^32 ticks + one frame from its predecessor): the
             * outlier took the late-pre-roll step (wrap_off −2^33), the good frames after it all landed 2^33 low, LAYERA
             * saw 13 of them on its "new" timeline, kept NEW and applied +47721.859 s: picture and sound lost for good.
             * A packet one normal step on from the packet BEFORE the outlier is back on its timeline: undo the outlier's
             * wrap step, then LAYERA's outlier rule drops the one corrupt packet. */
            int64_t wrap_dw = ol ? ol->wrap_dw : 0;
            if (ol)
                ol->wrap_dw = 0;
            if (wrap_dw && ol->nom > 0 && ol->last2 != AV_NOPTS_VALUE &&
                raw - ol->last2 > 0 && raw - ol->last2 <= 3 * ol->nom) {
                d->wrap_off[pkt->stream_index] -= wrap_dw;
                av_log(NULL, AV_LOG_WARNING, "[PTV-DISCONT] stream %d: the %+.3fs wrap step was taken on ONE corrupt "
                       "timestamp (the next packet is back on the old timeline) — wrap step reverted\n",
                       pkt->stream_index, wrap_dw * av_q2d(st->time_base));
            } else
            if (wl > 0 && wall_now - wl > guard_us) {
                int64_t expect = last + av_rescale(wall_now - wl, st->time_base.den,
                                                   (int64_t)st->time_base.num * 1000000);
                int64_t diff = expect - raw;
                int64_t k = diff >= 0 ? (diff + half) / mask : -((-diff + half) / mask);
                d->wrap_off[pkt->stream_index] += k * mask;
                av_log(NULL, AV_LOG_INFO,
                       "[PTV-DISCONT] stream %d: re-anchored after %.1fh silence (%+"PRId64" wraps; "
                       "delta heuristic aliases past %.1fh)\n",
                       pkt->stream_index, (wall_now - wl) / 3600.0e6, k, guard_us / 3600.0e6);
            }
            else if (delta < -half) {
                d->wrap_off[pkt->stream_index] += mask;  /* 33-bit wrap: rolled forward */
                if (ol) ol->wrap_dw = mask;              /* T-320: undone if the next packet says "outlier" */
            } else if (delta >  half) {
                d->wrap_off[pkt->stream_index] -= mask;  /* late pre-roll pkt */
                if (ol) ol->wrap_dw = -mask;
            }
        }
        if (d->ts_outl && pkt->stream_index < d->nb_streams_open)
            d->ts_outl[pkt->stream_index].last2 = d->wrap_last[pkt->stream_index];   /* 2.0.0-pre9.2 */
        d->wrap_last[pkt->stream_index] = raw;
        d->wrap_wall_last[pkt->stream_index] = wall_now;   /* gap-fix: per-stream packet arrival wall-clock */
        if (d->gap_vsnap)                                  /* pre16 #47-A: video-progress ref for the gap guard */
            d->gap_vsnap[pkt->stream_index] = d->vpkt;
    }
    off = d->wrap_off[pkt->stream_index];   /* 33-bit mask + per-stream discontinuity self-rebase (dense V/A) */
    /* P2 §7.1: sparse copied streams (DVB-sub/teletext, data, SCTE-35) skip the per-stream absorber (their
     * multi-second gaps would false-trigger it) and instead ride the program offset. Dense V/A carry their
     * own per-stream rebase in wrap_off above (§5.A.2 makes that rebase use the SHARED amount, but it's still
     * applied per-stream at each stream's own crossing — NOT added to prog_off here). */
    if (g_discont && g_prog_off && (st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE ||
                                    st->codecpar->codec_type == AVMEDIA_TYPE_DATA))
        off += d->prog_off;
    if (off) {
        if (pkt->pts != AV_NOPTS_VALUE) pkt->pts += off;
        if (pkt->dts != AV_NOPTS_VALUE) pkt->dts += off;
    }
    if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        if (d->prog_off != po0 && g_discont && g_prog_off)
            vwait_follow_video(d, pkt->pos, d->prog_off - po0);
        d->vunwrap_n++;
    }
}

/* 1.0.1-pre17 fix round (R1): reopen a live mv net input after a read error. Closes the dead
 * context FIRST (its udp socket must release the port before the re-bind, or the stale
 * circular-buffer reader keeps draining the datagrams), then open/find with a COPY of the
 * original format opts, then validate the stream layout — the demux-side per-stream arrays
 * (wrap/edit/disc) and every consumed index (vstream/astream/pass) were sized and resolved
 * against the original layout, so a source that comes back RESHAPED must keep retrying, not
 * dispatch mis-indexed packets. Returns 0 on success (d->ifmt swapped + published), <0 on a
 * failed attempt (caller backs off and retries forever — owner mandate: everything
 * auto-resumes after ANY outage). */
/* 2.0.0-pre5: what a reopen must find again. Mosaic slots compare count + types (as since pre17);
 * a single input also compares PID, language and the video codec / coded size — a same-shape PID
 * swap would otherwise mis-bind tracks, and a size change would break the filtergraph. */
typedef struct ReopenShape {
    uint8_t type;
    int     id, codec_id, w, h;
    char    lang[8];
} ReopenShape;

#define PTV_REOPEN_RESHAPED AVERROR(EPROTO)   /* single input: the source came back different */

static void reopen_shape_fill(ReopenShape *sh, const AVStream *st)
{
    const AVDictionaryEntry *lg = av_dict_get(st->metadata, "language", NULL, 0);
    sh->type     = (uint8_t)st->codecpar->codec_type;
    sh->id       = st->id;
    sh->codec_id = st->codecpar->codec_id;
    sh->w        = st->codecpar->width;
    sh->h        = st->codecpar->height;
    snprintf(sh->lang, sizeof sh->lang, "%s", lg ? lg->value : "");
}

/* 2.0.0-pre5: AVIOInterruptCB for the single input. A source that keeps sending PSI / nulls but no
 * video never returns from av_read_frame and never trips the read timeout (bytes keep arriving):
 * cut the blocked read once no video packet was read for -lost_after. Armed only while the demux
 * thread runs (never during the start-up probe); the reopen loop re-stamps the read clock so each
 * new open gets a full -lost_after. Must be set BEFORE avformat_open_input (the URLContext copies it). */
/* 2.0.0-pre7 (T-056 §4/§15): while a single input waits for its source at start there is no stats line, so a channel
 * waiting on purpose looked like a wedged process to sync_check (it trusts a src= reading newer than 2 x the stats
 * period). Print a stats-format heartbeat every -stats_period while acquiring. The open blocks for up to the read
 * timeout, so this runs from the I/O interrupt callback lavf polls during it, and between attempts. */
_Atomic int64_t g_acq_since;
static _Atomic int64_t g_acq_hb_last;
void ptv_acq_heartbeat(void)
{
    int64_t t0 = atomic_load_explicit(&g_acq_since, memory_order_relaxed), now, last;
    if (!t0 || !g_stats)
        return;
    now  = av_gettime_relative();
    last = atomic_load_explicit(&g_acq_hb_last, memory_order_relaxed);
    if (last && now - last < g_stats_period_us)
        return;
    if (!atomic_compare_exchange_strong(&g_acq_hb_last, &last, now))
        return;
    av_log(NULL, AV_LOG_INFO, "frame=    0 fps=0.0 src=waiting hold=%.1f\n", (now - t0) / 1e6);
}

/* 2.0.1 T-360: the single input's demux thread, for the copy release below */
static DemuxArgs *icb_da;
static pthread_t  icb_tid;

int ptv_src_interrupt(void *opaque)
{
    int64_t vr;
    (void)opaque;
    ptv_acq_heartbeat();
    /* 2.0.1 T-360: while the output holds (no fresh frame for 100 ms) the demux thread sits in the read and lavf calls
     * this every <= 100 ms: release the sparse copies on the last picture before the hold here (ptv_vmap_lookup's held
     * rule), on the demux thread itself. The master output thread did it on held ticks in c32558cbca and crashed every
     * LOST reopen (304c6227e5); here the reopen runs on the same thread and the drain never reads d->ifmt */
    if (icb_da && pthread_equal(pthread_self(), icb_tid) &&
        av_gettime_relative() - atomic_load_explicit(&g_src_fresh_wc, memory_order_relaxed) >= 100000)
        ptv_copy_vwait(icb_da);
    if (!atomic_load_explicit(&g_src_icb_armed, memory_order_relaxed))
        return 0;
    vr = atomic_load_explicit(&g_src_vread_wc, memory_order_relaxed);
    return vr && av_gettime_relative() - vr > g_lost_after_us;
}

/* 1.0.1-pre17 fix round (R1): reopen a live mv net input after a read error (2.0.0-pre5: and a single
 * input). Closes the dead context FIRST (its udp socket must release the port before the re-bind, or the
 * stale circular-buffer reader keeps draining the datagrams), then open/find with a COPY of the original
 * format opts, then validate the stream layout against the layout AT OPEN (streams born later — T-064 — are
 * not part of it): the demux-side per-stream arrays and every consumed index were sized and resolved
 * against it. Returns 0 on success (d->ifmt swapped + published), PTV_REOPEN_RESHAPED when a single input
 * came back different, other <0 on a failed attempt (caller backs off and retries). */
static int demux_reopen_once(DemuxArgs *d, unsigned old_nb, const ReopenShape *old)
{
    AVFormatContext *nf = NULL;
    AVDictionary *o2 = NULL;
    int r;
    unsigned i, nb;
    int renum = 0;
    if (!(nf = avformat_alloc_context()))
        return AVERROR(ENOMEM);
    if (d->single)
        nf->interrupt_callback = (AVIOInterruptCB){ ptv_src_interrupt, NULL };
    av_dict_copy(&o2, d->reopen_opts, 0);
    r = avformat_open_input(&nf, d->url, NULL, &o2);
    av_dict_free(&o2);
    if (r < 0)
        return r;
    r = ptv_find_stream_info(nf);   /* pre19.1: tolerant AUDIO probe (same as the initial open) */
    if (r < 0)
        goto bad;
    nf->max_probe_packets = 1;      /* 1.2.2 T-073: no codec probe for streams born after the open */
    r = d->single ? PTV_REOPEN_RESHAPED : AVERROR(EINVAL);
    /* lavf adds a synthetic EPG stream (PID 0x12, in no program) when an EIT section arrives within the probe: RAV
     * (cor-3, 2026-10-13) came back from a 57 s video outage with video / audio / SCTE-35 as before plus that stream,
     * read as RESHAPED and restarted (~25 s dark). Trailing EPG streams beyond the open layout are ignored like any
     * stream born after the open (T-064: their packets are dropped before anything indexes them) */
    nb = nf->nb_streams;
    while (nb > old_nb && nf->streams[nb - 1]->codecpar->codec_id == AV_CODEC_ID_EPG)
        nb--;
    if (nb > old_nb || (nb < old_nb && !d->single))
        goto bad;
    /* 2.2 (spec §6 / E9): a PID renumber with the layout unchanged — the same streams in the same order, the same
     * languages, the same video codec and size — is the same source (the video moved to a new PID, a provider's mux
     * renumbered): resumed like an outage, the -map indices still point at the same content (the output PIDs are
     * content-keyed). A reordered or changed layout stays RESHAPED (exit 4 for a re-probe). Was: any PID change = exit 4 */
    for (i = 0; i < nb; i++) {
        ReopenShape now;
        reopen_shape_fill(&now, nf->streams[i]);
        if (now.type != old[i].type)
            goto bad;
        if (d->single && now.id != old[i].id)
            renum++;
        if (d->single &&
            (strcmp(now.lang, old[i].lang) ||
             (now.type == AVMEDIA_TYPE_VIDEO &&
              (now.codec_id != old[i].codec_id ||
               (now.w && old[i].w && (now.w != old[i].w || now.h != old[i].h))))))
            goto bad;
    }
    /* 2.1 T-424: fewer streams, the ones there as before = the source is not complete yet, not reshaped. A provider's
     * encoder restart came back with video + audio and its SCTE-35 PID a moment later (cor-3 Fintech_247, ~1/day): read
     * as RESHAPED, exit 4, the wrapper's re-probe found the same three streams — ~40 s dark for nothing. Retried as a
     * failed attempt; still short after 10 s = a real change (exit 4 as before) */
    if (nb < old_nb) {
        int64_t now = av_gettime_relative();
        if (!d->short_wc) {
            d->short_wc = now;
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 came back incomplete (%u of %u streams, the ones there as "
                   "before) — retrying for up to 10 s before calling it reshaped\n", nb, old_nb);
        }
        if (now - d->short_wc < 10000000)
            r = AVERROR(EAGAIN);
        goto bad;
    }
    d->short_wc = 0;
    if (renum)
        av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 came back with %d of its %u streams on new PIDs, the layout unchanged — "
               "resumed (the same source renumbered)\n", renum, nb);
    d->ifmt = nf;
    if (d->ifmt_home) *d->ifmt_home = nf;      /* teardown close target follows the swap */
    d->pmt_seen = 0;                           /* T-003: the new open's PMT is the reference */
    d->cc_raw_dts = AV_NOPTS_VALUE;            /* T-326: a new open's labels are a new domain */
    d->reopen_cnt++;
    if (d->rsync_slot >= 0 && d->rsync_slot < PTV_MAX_INPUT)   /* rider (a), rr191 F3: the ASTAMP
                                                                * extrapolation carry must not span the reopen */
        atomic_store_explicit(&g_reopen_wc[d->rsync_slot], av_gettime_relative(), memory_order_relaxed);
    av_log(NULL, AV_LOG_WARNING,
           "[PTV-REOPEN] input '%s' re-opened after read failure (#%"PRId64") — resuming as a "
           "fresh join (slate/recovery machinery downstream)\n", d->url, d->reopen_cnt);
    return 0;
bad:
    if (r == PTV_REOPEN_RESHAPED)
        av_log(NULL, AV_LOG_ERROR, "[PTV-SRC] in0 came back RESHAPED (%u streams, was %u; or a PID, language, "
               "video codec or size changed) on %s\n", nf->nb_streams, old_nb, d->url);
    avformat_close_input(&nf);
    return r;
}

static void by_count_gap(DemuxArgs *d, int64_t gap, int64_t now);   /* 2.0.0-pre4, defined below */
static void door_arrived(DemuxArgs *d, int si, int64_t nw);          /* 2.0.0-pre20 T-364, defined below */
static void door_erase(DemuxArgs *d, const AVStream *vst, int64_t vb);   /* 2.0.2 T-398, defined below */

/* 2.0.0 T-345: raw (a video packet after a gap) is still on the pre-gap timeline: 0..500 ms past ref (33-bit wrap) */
static int src_tail_more(const AVStream *vst, int64_t raw, int64_t ref)
{
    int64_t delta = raw - ref;
    if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {
        int64_t m = 1LL << vst->pts_wrap_bits;
        delta &= m - 1;
        if (delta >= m >> 1) delta -= m;
    }
    delta = av_rescale_q(delta, vst->time_base, AV_TIME_BASE_Q);
    return delta >= 0 && delta < 500000;
}

/* 2.0.2 T-386 (spec §9 late join): a deferred mosaic slot's source arrived — set up what the start-up does for a live
 * slot (per-stream state, video stream, LAYERA buffer), resolve each of its tracks' -map specifier, and hand the
 * decoders over to the decode / audio threads through their own queues (BIND sentinels: every AudioState / DecodeCtx
 * write stays on its owner thread). A later outage of the slot takes the normal reopen. */
static int audio_bind_send(DemuxArgs *d, int j, int si);

static int slot_bind(DemuxArgs *d)
{
    struct Input *in = d->in;
    AVFormatContext *f = in->ifmt;
    int nb = f->nb_streams, si, j, vs;

    in->wrap_off        = av_calloc(nb, sizeof(*in->wrap_off));
    in->wrap_last       = av_malloc_array(nb, sizeof(*in->wrap_last));
    in->wrap_wall_last  = av_calloc(nb, sizeof(*in->wrap_wall_last));
    in->edit_us         = av_calloc(nb, sizeof(*in->edit_us));
    in->gap_vsnap       = av_calloc(nb, sizeof(*in->gap_vsnap));
    in->wall_cad_us     = av_calloc(nb, sizeof(*in->wall_cad_us));
    in->pkt_wall_gap_us = av_calloc(nb, sizeof(*in->pkt_wall_gap_us));
    in->tail_gap_us     = av_calloc(nb, sizeof(*in->tail_gap_us));
    in->tail_wc_us      = av_calloc(nb, sizeof(*in->tail_wc_us));
    in->ts_outl         = av_calloc(nb, sizeof(*in->ts_outl));
    if (!in->wrap_off || !in->wrap_last || !in->wrap_wall_last || !in->edit_us || !in->gap_vsnap ||
        !in->wall_cad_us || !in->pkt_wall_gap_us || !in->tail_gap_us || !in->tail_wc_us || !in->ts_outl)
        return AVERROR(ENOMEM);
    for (si = 0; si < nb; si++) {
        in->wrap_last[si] = AV_NOPTS_VALUE;
        in->ts_outl[si].last2 = in->ts_outl[si].pend_last = AV_NOPTS_VALUE;
    }
    vs = av_find_best_stream(f, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs < 0)
        return AVERROR_STREAM_NOT_FOUND;
    in->vstream = d->vstream = vs;
    in->vist    = f->streams[vs];
    d->wrap_off = in->wrap_off; d->wrap_last = in->wrap_last; d->wrap_wall_last = in->wrap_wall_last;
    d->ts_outl = in->ts_outl; d->edit_us = in->edit_us; d->gap_vsnap = in->gap_vsnap;
    d->wall_cad_us = in->wall_cad_us; d->pkt_wall_gap_us = in->pkt_wall_gap_us;
    d->tail_gap_us = in->tail_gap_us; d->tail_wc_us = in->tail_wc_us;
    d->nb_streams_open = nb;
    d->late_stream_max = -1;
    d->cc_raw_dts = AV_NOPTS_VALUE;
    /* audio: each deferred track's -map specifier against the arrived layout; a track whose source has no match keeps
     * its silence (logged) */
    for (j = 0; j < d->n_audio; j++) {
        PtvBindRule br;
        PtvBindPick pk;
        int r;
        if (d->astream[j] >= 0 || !d->abind[j][0])
            continue;
        if (ptv_bind_parse(d->abind[j], &br) > 0)        /* 2.2: an audio rule (spec §7) picks by content */
            si = ptv_bind_audio(f, d->rsync_slot, &br, 1, &pk, 1, 0) == 1 ? pk.stream : nb;
        else
            for (si = 0; si < nb; si++)
                if (f->streams[si]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
                    avformat_match_stream_specifier(f, f->streams[si], d->abind[j]) > 0)
                    break;
        if (si == nb) {
            av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] %d:%s → no such audio in in%d — the track stays silent\n",
                   d->rsync_slot, d->abind[j], d->rsync_slot);
            continue;
        }
        if ((r = audio_bind_send(d, j, si)) == AVERROR(ENOMEM))
            return r;
        if (r < 0)
            continue;
        {
            AVDictionaryEntry *lg = av_dict_get(f->streams[si]->metadata, "language", NULL, 0);
            av_log(NULL, AV_LOG_INFO, "[PTV-BIND] %d:%s → in%d pid 0x%x %s %dch %s (slot %d)\n", d->rsync_slot,
                   d->abind[j], d->rsync_slot, f->streams[si]->id, avcodec_get_name(f->streams[si]->codecpar->codec_id),
                   f->streams[si]->codecpar->ch_layout.nb_channels, lg ? lg->value : "und", d->rsync_slot);
        }
    }
    {   /* video last: the decoder opens before the first packet of the slot reaches it */
        AVPacket *bp = av_packet_alloc();
        PtvSlotBind *b = av_mallocz(sizeof(*b));
        if (!bp || !b || !(b->par = avcodec_parameters_alloc()) ||
            avcodec_parameters_copy(b->par, f->streams[vs]->codecpar) < 0) {
            if (b) avcodec_parameters_free(&b->par);
            av_free(b); av_packet_free(&bp);
            return AVERROR(ENOMEM);
        }
        b->tb  = f->streams[vs]->time_base;
        b->avg = f->streams[vs]->avg_frame_rate;
        bp->opaque = b;
        bp->flags |= PTV_PKT_FLAG_SLOT_BIND;
        if (av_thread_message_queue_send(d->video_q, &bp, 0) < 0) {
            avcodec_parameters_free(&b->par); av_free(b); av_packet_free(&bp);
            return AVERROR_EXIT;
        }
    }
    d->ifmt = f;                                     /* *d->ifmt_home == in->ifmt already (the acquire set it) */
    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in%d WAITING → LIVE: slot %d joins the mosaic\n", d->rsync_slot, d->rsync_slot);
    return 0;
}

/* 2.1 T-326: the EIA-608 data of an H.264 access unit (SEI user_data_registered, ATSC A/53 "GA94" cc_data) — the bytes the
 * decoder would export as AV_FRAME_DATA_A53_CC: cc_count triplets. Checked as the decoder checks them (T.35 country 0xB5,
 * provider 0x0031, GA94, type 3, process_cc_data_flag, the 0xFF marker after the triplets). Returns the byte count. */
static int ptv_a53_from_h264(const uint8_t *buf, int size, uint8_t *out, int cap)
{
    uint8_t rbsp[1024];
    int i = 0, n = 0;
    while (i + 3 < size) {
        int s, e, r = 0, z = 0, k;
        if (buf[i] || buf[i + 1] || buf[i + 2] != 1) {
            i++;
            continue;
        }
        s = i + 3;
        for (e = s; e + 2 < size && (buf[e] || buf[e + 1] || buf[e + 2] > 1); e++) ;
        if (e + 2 >= size)
            e = size;
        i = e;
        if ((buf[s] & 0x1f) != 6)                   /* SEI */
            continue;
        for (k = s + 1; k < e && r < (int)sizeof(rbsp); k++) {   /* drop the emulation prevention bytes */
            if (z >= 2 && buf[k] == 3) {
                z = 0;
                continue;
            }
            z = buf[k] ? 0 : z + 1;
            rbsp[r++] = buf[k];
        }
        for (k = 0; k + 2 < r && rbsp[k] != 0x80; ) {
            int type = 0, len = 0;
            while (k < r && rbsp[k] == 0xff) type += rbsp[k++];
            if (k >= r) break;
            type += rbsp[k++];
            while (k < r && rbsp[k] == 0xff) len += rbsp[k++];
            if (k >= r) break;
            len += rbsp[k++];
            if (k + len > r)
                break;
            if (type == 4 && len >= 11 && rbsp[k] == 0xb5 && rbsp[k + 1] == 0x00 && rbsp[k + 2] == 0x31 &&
                !memcmp(rbsp + k + 3, "GA94", 4) && rbsp[k + 7] == 3 && (rbsp[k + 8] & 0x40)) {
                int cnt = rbsp[k + 8] & 0x1f;
                if (10 + 3 * cnt < len && rbsp[k + 10 + 3 * cnt] == 0xff && n + 3 * cnt <= cap) {
                    memcpy(out + n, rbsp + k + 10, 3 * cnt);
                    n += 3 * cnt;
                }
            }
            k += len;
        }
    }
    return n;
}

/* 2.1 T-326: a corrupt video packet is discarded before decode, and the EIA-608 pair its SEI carried went with it ("oice"
 * for "choice" on TruBLU). The SEI leads the access unit and usually survives a damaged tail: send its captions on as a
 * caption-only packet. Its timestamps here are raw (unwrap and timeline come later), so it carries them relative to the
 * previous video packet — the decode thread places it on that packet's dts. */
static void ptv_cc_salvage(DemuxArgs *d, AVPacket *pkt)
{
    const AVStream *st = d->ifmt->streams[pkt->stream_index];
    int64_t prev = d->cc_raw_dts;
    d->cc_raw_dts = pkt->dts;
    if (!(pkt->flags & AV_PKT_FLAG_CORRUPT) || !g_discardcorrupt || st->codecpar->codec_id != AV_CODEC_ID_H264 ||
        prev == AV_NOPTS_VALUE || pkt->dts == AV_NOPTS_VALUE || pkt->pts == AV_NOPTS_VALUE) {
        return;
    } else {
        uint8_t cc[PTV_CC_PEND_BYTES];
        int n = ptv_a53_from_h264(pkt->data, pkt->size, cc, sizeof(cc));
        int64_t m = st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63 ? 1LL << st->pts_wrap_bits : 0;
        int64_t step = pkt->dts - prev, reord = pkt->pts - pkt->dts;
        AVPacket *cp;
        uint8_t *sd;
        if (m) {                                    /* nearest 33-bit wrap */
            step &= m - 1; if (step >= m >> 1) step -= m;
            reord &= m - 1; if (reord >= m >> 1) reord -= m;
        }
        if (!n || step <= 0 || av_rescale_q(step, st->time_base, AV_TIME_BASE_Q) > 1000000 ||
            llabs(av_rescale_q(reord, st->time_base, AV_TIME_BASE_Q)) > 1000000 || !(cp = av_packet_alloc()))
            return;                                 /* no captions, or labels no neighbour can place */
        if (!(sd = av_packet_new_side_data(cp, AV_PKT_DATA_A53_CC, n))) {
            av_packet_free(&cp);
            return;
        }
        memcpy(sd, cc, n);
        cp->stream_index = pkt->stream_index;
        cp->flags        = PTV_PKT_FLAG_CC_ONLY;
        cp->dts          = step;
        cp->pts          = reord;
        demux_send(d->video_q, cp, d->drop, &(int64_t){ 0 });   /* a full queue drops it uncounted: not a picture */
    }
}

/* the audio track j's decoder for stream si (the audio thread swaps it in: a deferred slot's first bind, or a re-bind) */
static int audio_bind_send(DemuxArgs *d, int j, int si)
{
    AVFormatContext *f = d->ifmt;
    AVPacket *bp;
    PtvSlotBind *b;
    if (!(bp = av_packet_alloc()) || !(b = av_mallocz(sizeof(*b))) || !(b->par = avcodec_parameters_alloc()) ||
        avcodec_parameters_copy(b->par, f->streams[si]->codecpar) < 0) {
        if (bp && bp->opaque) av_free(bp->opaque);
        av_packet_free(&bp);
        return AVERROR(ENOMEM);
    }
    b->tb = f->streams[si]->time_base;
    bp->opaque = b;
    bp->flags |= PTV_PKT_FLAG_SLOT_BIND;
    if (av_thread_message_queue_send(d->audio_q[j], &bp, 0) < 0) {
        avcodec_parameters_free(&b->par); av_free(b); av_packet_free(&bp);
        return AVERROR(EAGAIN);
    }
    d->astream[j] = si;
    return 0;
}

/* 2.2 T-427 (spec §7.3, D36): a rule's track follows its source inside its language + role. Its stream gone from the PMT
 * or silent past -lost_after → the rule runs again on what is there now, limited to the track's identity; the original
 * back and ranking first again → back to it. Nothing in the group → the track stays silent (strict across languages).
 * Another stream counts only while in the PMT with a packet in the last second; the bound one until -lost_after. Pins
 * (a:pid / a:index) and copied tracks never move. Demux thread, once a second, while the video flows. */
static void rule_rebind(DemuxArgs *d, int64_t now)
{
    AVFormatContext *f = d->ifmt;
    int slot = d->rsync_slot > 0 ? d->rsync_slot : 0, j, si, b;
    for (j = 0; j < d->n_audio && j < 32; j++) {
        PtvBindRule r;
        PtvBindPick pk;
        uint64_t present = 0;
        int cur = d->astream[j];
        if (cur < 0 || !d->abind[j][0] || ptv_bind_parse(d->abind[j], &r) <= 0 ||
            r.kind == PTV_BIND_PID || r.kind == PTV_BIND_INDEX)
            continue;
        if (!(d->aid_set & (1u << j))) {
            if (ptv_bind_ident(f, slot, cur, d->aid_lang[j], &d->aid_role[j]) < 0)
                continue;
            d->aid_set |= 1u << j;
        }
        for (si = 0; si < d->nb_streams_open && si < 64; si++) {
            int64_t last = d->wrap_wall_last ? d->wrap_wall_last[si] : 0;
            for (b = 0; b < d->pmt_n && d->pmt_pid[b] != f->streams[si]->id; b++) ;
            if (b < d->pmt_n && (si == cur ? !last || now - FFMAX(last, d->v_flow_wc) < g_lost_after_us
                                           : last && now - last < 1000000))
                present |= 1ULL << si;
        }
        if (ptv_bind_audio_in(f, slot, &r, !d->single, present, d->aid_lang[j], d->aid_role[j],
                              &pk, 1, 0) != 1) {
            if (!(present >> cur & 1) && !(d->a_orphan & (1u << j))) {
                d->a_orphan |= 1u << j;
                av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] in%d a:%d %s: pid 0x%x gone and no other %s track — silent until "
                       "one comes (the output track stays)\n", slot, j, d->abind[j], f->streams[cur]->id, d->aid_lang[j]);
            }
            continue;
        }
        d->a_orphan &= ~(1u << j);
        if (pk.stream == cur || audio_bind_send(d, j, pk.stream) < 0)
            continue;
        av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] in%d a:%d %s re-bound: pid 0x%x → 0x%x %s %dch %s (%s)\n", slot, j,
               d->abind[j], f->streams[cur]->id, f->streams[pk.stream]->id,
               avcodec_get_name(f->streams[pk.stream]->codecpar->codec_id),
               f->streams[pk.stream]->codecpar->ch_layout.nb_channels, d->aid_lang[j],
               present >> cur & 1 ? "ranks first again" : "its stream is gone; same language + role");
    }
}

/* the followed program's PMT has a new version (checked per packet: a re-bind follows it at once, T-427) */
static int pmt_moved(const DemuxArgs *d)
{
    unsigned i;
    if (!d->pmt_seen)
        return 0;
    for (i = 0; i < d->ifmt->nb_programs; i++)
        if (d->ifmt->programs[i]->id == d->pmt_prog)
            return d->ifmt->programs[i]->pmt_version != d->pmt_ver;
    return 0;
}

/* 2.1 T-003 step 1: what changed in the source, logged once — the PMT's version (with the PIDs it added and dropped) and
 * a MAPPED audio stream that goes silent while the video flows (sync_check restarts such a channel as DEAD_AUDIO_TRACK
 * with nothing in our log to say why). Called from the demux loop at most once a second. */
static void ptv_src_check(DemuxArgs *d, int64_t now)
{
    AVFormatContext *f = d->ifmt;
    int slot = d->rsync_slot > 0 ? d->rsync_slot : 0, i, j;
    d->src_chk_wc = now;
    for (i = 0; i < (int)f->nb_programs; i++) {
        const AVProgram *p = f->programs[i];
        int has_v = 0, n = 0, pid[PTV_SRC_PIDS];
        for (j = 0; j < (int)p->nb_stream_indexes; j++)
            has_v |= (int)p->stream_index[j] == d->vstream;
        if (d->pmt_seen ? p->id != d->pmt_prog : !has_v)
            continue;                               /* followed by id: a PMT update rebuilds its stream list */
        for (j = 0; j < (int)p->nb_stream_indexes && n < PTV_SRC_PIDS; j++)
            pid[n++] = f->streams[p->stream_index[j]]->id;
        if (d->pmt_seen && p->pmt_version != d->pmt_ver) {
            char add[160] = "", del[160] = "";
            int a, b;
            for (a = 0; a < n; a++) {
                for (b = 0; b < d->pmt_n && d->pmt_pid[b] != pid[a]; b++) ;
                if (b == d->pmt_n)
                    av_strlcatf(add, sizeof(add), " 0x%x", pid[a]);
            }
            for (b = 0; b < d->pmt_n; b++) {
                for (a = 0; a < n && pid[a] != d->pmt_pid[b]; a++) ;
                if (a == n)
                    av_strlcatf(del, sizeof(del), " 0x%x", d->pmt_pid[b]);
            }
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in%d PMT version %d \xe2\x86\x92 %d (program %d): %d streams; added:%s; "
                   "gone:%s\n", slot, d->pmt_ver, p->pmt_version, p->id, n, *add ? add : " -", *del ? del : " -");
        }
        d->pmt_seen = 1;
        d->pmt_prog = p->id;
        d->pmt_ver  = p->pmt_version;
        d->pmt_n    = n;
        memcpy(d->pmt_pid, pid, n * sizeof(*pid));
        break;
    }
    for (j = 0; j < d->n_audio && j < 32 && now - d->v_last_wc < 1000000; j++) {
        int s = d->astream[j];
        const AVDictionaryEntry *lang;
        int64_t last;
        if (s < 0 || s >= d->nb_streams_open || !d->wrap_wall_last || !(last = d->wrap_wall_last[s]))
            continue;                               /* not mapped to this source / never flowed */
        last = FFMAX(last, d->v_flow_wc);           /* a whole-source outage is not the audio's silence */
        lang = av_dict_get(f->streams[s]->metadata, "language", NULL, 0);
        if (!(d->a_silent & (1u << j)) && now - last >= 10000000) {
            d->a_silent |= 1u << j;
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in%d audio track %d (stream #%d, pid 0x%x, %s) silent for %.0f s "
                   "while the video flows\n", slot, j, s, f->streams[s]->id, lang ? lang->value : "und",
                   (now - last) / 1e6);
        } else if ((d->a_silent & (1u << j)) && now - last < 1000000) {
            d->a_silent &= ~(1u << j);
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in%d audio track %d (stream #%d, pid 0x%x) flowing again\n",
                   slot, j, s, f->streams[s]->id);
        }
    }
    if (d->pmt_seen && now - d->v_last_wc < 1000000)   /* 2.2 T-427: not during a whole-source outage (the hold's) */
        rule_rebind(d, now);
}

void *demux_thread(void *arg)
{
    DemuxArgs *d = arg;
    AVPacket *pkt = av_packet_alloc();
    int64_t diag_last = av_gettime_relative();
    int ret = 0;
    int rd_ret = 0, rd_force = 0;   /* 2.0.0-pre5: read result / forced reopen */

    if (d->reopen && d->single)     /* 2.0.0-pre5: the no-video read interrupt goes live with the thread */
        atomic_store_explicit(&g_src_icb_armed, 1, memory_order_relaxed);
    if (d->single && g_vmap) {      /* 2.0.1 T-360: the read interrupt releases held copies on this thread */
        icb_tid = pthread_self();
        icb_da  = d;
    }

    if (!pkt)
        goto end;
    for (;;) {
        if (g_t_us > 0 && atomic_load_explicit(&g_t_stop, memory_order_relaxed)) {
            av_log(NULL, AV_LOG_INFO, "ptvencoder: -t reached — input '%s' stops pulling (flush to clean exit)\n", d->url);
            break;                                 /* rider (b): ride the file-EOF teardown */
        }
        if (!d->ifmt) {   /* 2.0.2 T-386: a mosaic slot not live at the start — acquire it here (no heartbeat while
                           * waiting: a slot without a source is not a wedged decoder), then bind it */
            if (!d->in || ptv_slot_acquire(d->in, d->rsync_slot) < 0 || slot_bind(d) < 0) {
                if (d->in && d->in->ifmt && !d->ifmt)
                    avformat_close_input(&d->in->ifmt);
                if (atomic_load_explicit(&g_t_stop, memory_order_relaxed))
                    break;
                av_usleep(1000000);
            }
            continue;
        }
        /* 1.0.1-pre21 heartbeat: input-flow evidence + video_q depth for the [PTV-STALL]
         * watchdog (a vdec stall only reports while THIS keeps advancing — a dead input is
         * starvation, not a wedge). */
        if (d->rsync_slot >= 0 && d->rsync_slot < PTV_MAX_INPUT) {
            atomic_store_explicit(&g_hb_demux_wall[d->rsync_slot],
                                  av_gettime_relative(), memory_order_relaxed);
            atomic_store_explicit(&g_hb_vq[d->rsync_slot],
                                  av_thread_message_queue_nb_elems(d->video_q),
                                  memory_order_relaxed);
        }
        {   /* 2.0.0-pre5: video gone while other packets (audio) keep flowing — reopen after 2 x -lost_after */
            int64_t vr = atomic_load_explicit(&g_src_vread_wc, memory_order_relaxed);
            rd_force = (d->reopen && d->single && vr &&
                        av_gettime_relative() - vr > 2 * g_lost_after_us) || d->vmove_force;
        }
        if (rd_force || (rd_ret = av_read_frame(d->ifmt, pkt)) < 0) {
            /* pre17 R1: a live mv net input must NEVER EOF-latch its slot (rw_timeout expiry
             * on a >=30min outage previously slated the cell forever AND held the corrector
             * mosaic-wide via the slate mask). Reopen-retry with bounded backoff, forever;
             * validation failures (source reshaped) keep retrying too. The old ctx keeps
             * carrying the reads until a validated replacement lands. */
            if (d->reopen) {
                /* the layout AT OPEN: streams lavf added later (T-064) are not part of it — with the
                 * live count a clean source never matched again after one corrupt PID */
                unsigned old_nb = d->nb_streams_open, oi;
                ReopenShape old_shape[64];
                int64_t backoff = 1000000, rlog = 0, lost_t0 = av_gettime_relative();
                int attempt = 0;
                if (old_nb > 64) old_nb = 64;      /* validation window (mpegts programs are small) */
                for (oi = 0; oi < old_nb; oi++)
                    reopen_shape_fill(&old_shape[oi], d->ifmt->streams[oi]);
                if (d->single) {
                    int64_t vr = atomic_load_explicit(&g_src_vread_wc, memory_order_relaxed);
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 LOST: %s (no video for %.1f s, %d flushed "
                           "PES tails dropped) — reopening\n",
                           d->vmove_force ? "the video moved to a new PID" :
                           rd_force ? "video gone while other packets flow" : av_err2str(rd_ret),
                           vr ? (lost_t0 - vr) / 1e6 : 0.0, d->tail_flushed);
                    d->tail_flushed = 0;
                    atomic_store_explicit(&g_src_icb_armed, 0, memory_order_relaxed);
                }
                rd_force = 0;
                d->vmove_force = 0; d->vmove_n = 0; d->v_last_wc = 0;
                /* Close the dead ctx BEFORE retrying: its udp socket (with the fifo reader
                 * thread) stays bound otherwise and STEALS the datagrams from the re-bind —
                 * measured: the reopen loop failed forever after the source resumed (r1 gate,
                 * first round). We are in an outage; there is nothing left to read from it. */
                avformat_close_input(&d->ifmt);
                if (d->ifmt_home) *d->ifmt_home = NULL;
                for (;;) {
                    if (g_t_us > 0 && atomic_load_explicit(&g_t_stop, memory_order_relaxed))
                        goto end;                  /* rider (b): -t ends the retry-forever loop too */
                    int rr;
                    int64_t at0;
                    if (d->single) {   /* each attempt gets a full -lost_after before the no-video cut */
                        atomic_store_explicit(&g_src_vread_wc, av_gettime_relative(), memory_order_relaxed);
                        atomic_store_explicit(&g_src_icb_armed, 1, memory_order_relaxed);
                    }
                    attempt++;
                    at0 = av_gettime_relative();
                    rr = demux_reopen_once(d, old_nb, old_shape);
                    if (rr >= 0)
                        break;
                    if (rr == PTV_REOPEN_RESHAPED) {
                        /* 2.0.0 exits for a supervised respawn: the wrapper re-probes and builds the new
                         * plan (in-process re-binding is 2.1). Exit 4 (3 = the wrapper's duplicate refusal). */
                        av_log(NULL, AV_LOG_ERROR, "[PTV-SRC] in0 reshaped source — exiting 4 for a re-probe\n");
                        fflush(NULL);
                        _exit(4);
                    }
                    if (d->single) {
                        backoff = (int64_t)g_backoff_s[FFMIN(attempt, g_backoff_n) - 1] * 1000000;
                        if (ptv_url_passive(d->url))   /* the attempt was the wait; keep 1 s between starts
                                                          * (measured: stop_300 rejoined 19 s after the source) */
                            backoff = FFMAX(0, 1000000 - (av_gettime_relative() - at0));
                        atomic_store_explicit(&g_src_icb_armed, 0, memory_order_relaxed);
                    }
                    if (!rlog || av_gettime_relative() - rlog > 30000000) {
                        rlog = av_gettime_relative();
                        if (d->single)
                            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 LOST %.0f s on %s — reopen attempt #%d "
                                   "failed (%s), next in %.0f s\n", (av_gettime_relative() - lost_t0) / 1e6,
                                   d->url, attempt, av_err2str(rr), backoff / 1e6);
                        else
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-REOPEN] input '%s' still down — retrying (backoff %.0fs)\n",
                                   d->url, backoff / 1e6);
                    }
                    if (backoff > 0)
                        av_usleep((unsigned)backoff);
                    if (!d->single && backoff < 5000000) backoff += 1000000;
                }
                if (d->single)
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 LOST \xe2\x86\x92 reopened after %.1f s (attempt #%d)\n",
                           (av_gettime_relative() - lost_t0) / 1e6, attempt);
                continue;
            }
            break;
        }
        /* 2.0.0-pre5: a failed read makes mpegts hand back its pending PES (one per call) BEFORE the
         * error; pb->error is already set. These are pre-gap tails arriving a read timeout late:
         * classified as a rejoin they armed a false BURST (BANK to 12 s) and re-stamped video arrival.
         * The reopen follows, so drop them (a lone stale frame/audio frame under a hold). */
        if (d->reopen && d->single && d->ifmt->pb && d->ifmt->pb->error) {
            d->tail_flushed++;
            av_packet_unref(pkt);
            continue;
        }
        /* 1.2.2 T-064: lavf adds an AVStream mid-run for every new PMT PID and, with auto_guess,
         * for any unknown PID that starts a PES — one corrupt TS header after an outage is enough.
         * The per-stream arrays are sized to the stream count at open, and a late-born stream is
         * routed nowhere (dispatch matches only the streams bound at open): drop its packets here,
         * before any bookkeeping indexes them (was a heap overflow in demux_unwrap). */
        if (pkt->stream_index >= d->nb_streams_open) {
            /* 2.0.2 T-387: the source moved its video to a new PID (PMT update; cor-3 2026-10-08 Curiosity Now: the
             * audio flowed on, the slot's picture stayed dead until a restart 200 s later; a single input waited out
             * 2 x -lost_after, then re-probed). A late-born VIDEO stream that delivers 1 s of packets while the bound
             * video is silent >= 2 s = the move: reopen now — a mosaic slot resumes on the new PID as after an
             * outage (same stream count + types), a single input comes back RESHAPED and exits 4 for a re-probe. */
            if (d->reopen && !d->vmove_force && d->vstream >= 0 &&
                d->ifmt->streams[pkt->stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                int64_t now = av_gettime_relative();
                if (!d->v_last_wc || now - d->v_last_wc < 2000000)
                    d->vmove_n = 0;
                else if (++d->vmove_n >= 25) {
                    d->vmove_force = 1;
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] input '%s': video moved to pid 0x%x (pid 0x%x silent "
                           "%.1f s) — reopening\n", d->url, d->ifmt->streams[pkt->stream_index]->id,
                           d->ifmt->streams[d->vstream]->id, (now - d->v_last_wc) / 1e6);
                }
            }
            if (pkt->stream_index > d->late_stream_max) {
                const AVStream *ls = d->ifmt->streams[pkt->stream_index];
                d->late_stream_max = pkt->stream_index;
                av_log(NULL, AV_LOG_WARNING,
                       "[PTV-DEMUX] input '%s': stream #%d (pid 0x%x, %s) appeared after open "
                       "(%d at open) — ignored\n", d->url, pkt->stream_index, ls->id,
                       av_get_media_type_string(ls->codecpar->codec_type) ?
                       av_get_media_type_string(ls->codecpar->codec_type) : "unknown",
                       d->nb_streams_open);
            }
            av_packet_unref(pkt);
            continue;
        }
        {
            int64_t pnow = av_gettime_relative();
            if (pkt->stream_index == d->vstream) {
                if (pnow - d->v_last_wc > 1000000)
                    d->v_flow_wc = pnow;             /* T-003: silence counts only while the video flows */
                d->v_last_wc = pnow;                 /* 2.0.2 T-387 */
            }
            if (pnow - d->src_chk_wc >= 1000000 || pmt_moved(d))   /* T-003: any stream — a PMT update can move the video
                                                                  * itself; 2.2 T-427: a new PMT at once (a re-bind) */
                ptv_src_check(d, pnow);
        }
        /* 2.0.0-pre17 T-348 (see aac_next): a frame without its own PES label (pos < 0: lavf split it out of the
         * PES and interpolated its label) gets the previous label + the previous frame's ADTS duration */
        if (pkt->stream_index < PTV_AAC_LBL && pkt->size >= 7 &&
            d->ifmt->streams[pkt->stream_index]->codecpar->codec_id == AV_CODEC_ID_AAC &&
            pkt->data[0] == 0xFF && (pkt->data[1] & 0xF6) == 0xF0) {
            static const int adts_rate[16] = { 96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
                                               16000, 12000, 11025, 8000, 7350 };
            const AVStream *ast = d->ifmt->streams[pkt->stream_index];
            int si = pkt->stream_index, rate = adts_rate[(pkt->data[2] >> 2) & 0xF];
            /* only a STABLE header rate (3 frames in a row) counts, and only while it differs from the rate lavf times
             * the frames with: one damaged header (bit flips, loss) must never re-time a PES — it did (+405 ms for
             * good after sync_bitflip_100, -555 ms after loss_100: a forward re-time padded, the next PES label then
             * read as a backward relabel and was erased). An unswitched stream is left exactly as lavf labels it. */
            if (rate == d->aac_rate_cand[si]) {
                if (d->aac_rate_n[si] < 3 && ++d->aac_rate_n[si] == 3)
                    d->aac_rate_stable[si] = rate;
            } else {
                d->aac_rate_cand[si] = rate;
                d->aac_rate_n[si]    = 1;
            }
            rate = d->aac_rate_stable[si];
            if (rate > 0 && ast->codecpar->sample_rate > 0 &&     /* lavf: frame_size / sample_rate (2048 for HE-AAC) */
                av_rescale_q(ast->codecpar->frame_size > 0 ? ast->codecpar->frame_size : 1024,
                             (AVRational){ 1, ast->codecpar->sample_rate }, ast->time_base) !=
                av_rescale_q(1024, (AVRational){ 1, rate }, ast->time_base)) {
                int64_t dur = av_rescale_q(1024, (AVRational){ 1, rate }, ast->time_base);
                if (pkt->pos < 0 && d->aac_ok[si] && pkt->pts != AV_NOPTS_VALUE && pkt->pts != d->aac_next[si] &&
                    !(pkt->flags & AV_PKT_FLAG_CORRUPT)) {
                    if (rate != d->aac_rate[si]) {
                        d->aac_rate[si] = rate;
                        av_log(NULL, AV_LOG_WARNING, "[PTV-DEMUX] input '%s': stream #%d (pid 0x%x) AAC now %d Hz "
                               "(lavf times its frames at %d Hz) — labels inside a PES re-derived from the ADTS "
                               "header\n", d->url, si, ast->id, rate, ast->codecpar->sample_rate);
                    }
                    pkt->pts = pkt->dts = d->aac_next[si];
                    d->aac_fixed++;
                }
                if (pkt->pts != AV_NOPTS_VALUE) {
                    d->aac_next[si] = pkt->pts + dur;
                    d->aac_ok[si]   = 1;
                    pkt->duration   = dur;
                }
            } else
                d->aac_ok[si] = 0;                 /* lavf's timing is right: nothing to chain */
        }
        if (g_diag) {
            int64_t now = av_gettime_relative();
            if (now - diag_last >= 1000000) {
                /* Layer-A probe: the APPLIED A/V wrap_off differential (audio − video), rescaled to ms.
                 * For A/V sync the CHANGE in this across a splice must be ~0 (both rebased equally). A
                 * persistent jump that accumulates over splices is the per-stream-rebase divergence we're
                 * chasing; a value that returns to baseline after a straddle is only a transient. */
                int64_t avoff_ms = 0;
                if (d->n_audio > 0) {
                    int va = d->vstream, aa = d->astream[0];
                    avoff_ms = av_rescale_q(d->wrap_off[aa], d->ifmt->streams[aa]->time_base, (AVRational){1,1000})
                             - av_rescale_q(d->wrap_off[va], d->ifmt->streams[va]->time_base, (AVRational){1,1000});
                }
                av_log(NULL, AV_LOG_INFO, "[PTV-DIAG] demux vpkt=%"PRId64" vdrop=%"PRId64" vcorrupt=%"PRId64
                       " apkt=%"PRId64" adrop=%"PRId64" ppkt=%"PRId64" pdrop=%"PRId64" avoff=%"PRId64"ms\n",
                       d->vpkt, d->vdrop, d->vcorrupt, d->apkt, d->adrop, d->ppkt, d->pdrop, avoff_ms);
                diag_last = now;
            }
        }
        AVPacket *out = av_packet_alloc();
        if (!out) { av_packet_unref(pkt); break; }
        av_packet_move_ref(out, pkt);
        /* [PTV-CHAIN] RAW source-content tap — BEFORE demux_unwrap, so rawA-V reflects the
         * source's native A/V relationship. Compared against the post-unwrap srcA-V below, this
         * separates source-inherent A/V drift (rawA-V grows) from demux_unwrap per-stream rebase
         * divergence (rawA-V flat, srcA-V grows = unwrap_inj) — the number that decides §5.A vs §5.B. */
        if (out->pts != AV_NOPTS_VALUE) {
            if (out->stream_index == d->vstream)
                atomic_store_explicit(&g_ch_vsrc_raw, av_rescale_q(out->pts, d->ifmt->streams[d->vstream]->time_base, AV_TIME_BASE_Q), memory_order_relaxed);
            else if (d->n_audio > 0 && out->stream_index == d->astream[0])
                atomic_store_explicit(&g_ch_asrc_raw, av_rescale_q(out->pts, d->ifmt->streams[out->stream_index]->time_base, AV_TIME_BASE_Q), memory_order_relaxed);
        }
        /* 1.0.1-pre17 fix round (R3b, THE Fashion load-bearing fix): discard corrupt-flagged
         * packets BEFORE demux_unwrap and the LAYERA machinery. The corrupt check used to
         * live only in demux_dispatch — DOWNSTREAM of LAYERA — so a corrupt packet's garbage
         * dts fed jump DETECTION, false BASE recording, sib_jump stamps, last_dts continuity
         * refs and the wrap state before being thrown away. Fashion live (2026-07-18
         * 20:05:45.733): a corrupt audio packet with a wrapped dts (7869470123 ≈ 87438s →
         * post-unwrap −8005.7s = EXACTLY the video's post-jump domain) classified NEW against
         * the video's fresh bases, recorded FALSE audio bases + transitioned the stream, and
         * the 72ms-later "shared" flush rebased the UN-crossed audio +11594s — every
         * downstream disaster (one-sided cycle B, permanent avoff +11594s, the −73Mppm
         * pursuit) descends from that one poisoned classification. Bookkeeping must never
         * learn from a packet the pipeline is about to discard. The demux_dispatch check
         * stays as the belt for re-injected/other paths. */
        /* 2.0.0 T-320: a video packet whose PTS lies more than PTV_DISC_THRESHOLD_US from its own DTS carries a corrupt
         * PTS (one bit error in a PES header — no CRC). Measured on the 29.97 bitflip_100 timing media: PTS 82.826 s with
         * DTS 36.190 s (lavf interpolated the DTS across the damage, the header gave only the bad PTS); LAYERA watches DTS
         * and saw nothing, the decoder stamped the frame 46.6 s ahead and the master followed it — a 46.5 s output PTS
         * hole, and subtitles / SCTE-35 lost behind it for 30 s at 25 and 59.94 fps. Real reordering is a few frames.
         * The frame is kept; its PTS is dropped and the decoder falls back to the (continuous) DTS. */
        if (out->stream_index == d->vstream && out->pts != AV_NOPTS_VALUE && out->dts != AV_NOPTS_VALUE) {
            const AVStream *vst2 = d->ifmt->streams[out->stream_index];
            int64_t pd = out->pts - out->dts;
            if (vst2->pts_wrap_bits > 0 && vst2->pts_wrap_bits < 63) {      /* nearest 33-bit wrap */
                int64_t m = 1LL << vst2->pts_wrap_bits;
                pd &= m - 1;
                if (pd >= m >> 1) pd -= m;
            }
            if (llabs(av_rescale_q(pd, vst2->time_base, AV_TIME_BASE_Q)) > PTV_DISC_THRESHOLD_US) {
                if (!d->pts_outlier_n++ || !(d->pts_outlier_n % 100))
                    av_log(NULL, AV_LOG_WARNING, "[PTV-LAYERA] video PTS %+.3f s away from its DTS — a corrupt label, "
                           "PTS dropped, the frame keeps its DTS (%"PRId64" so far)\n",
                           av_rescale_q(pd, vst2->time_base, AV_TIME_BASE_Q) / 1e6, d->pts_outlier_n);
                out->pts = AV_NOPTS_VALUE;
            }
        }
        if (out->stream_index == d->vstream)
            ptv_cc_salvage(d, out);         /* 2.1 T-326: a corrupt picture's captions go on without it */
        if ((out->flags & AV_PKT_FLAG_CORRUPT) && g_discardcorrupt) {
            ret = demux_dispatch(d, out);   /* dispatch = the existing counted discard path */
            if (ret < 0) break;
            continue;
        }
        /* 2.0.0-pre3 (T-056 §3/§5.4), at READ time — before demux_unwrap (which already self-rebases
         * per-stream discontinuities) and before the LAYERA buffer (which can hold packets while input
         * flows): (1) stamp video arrival for the STALLED test; (2) classify the rejoin at the first video
         * packet after a hold on the RAW 33-bit DTS: A = content advance, W = wall gap. BURST = contiguous
         * content (delivery paused); LIVE LOSS = the source kept running (A ~ W); NEW DOMAIN = anything
         * else (restart, PTS reset). OBSERVE-ONLY: logged, nothing acts on it yet (pre4). */
        if (g_src_watch && out->stream_index == d->vstream) {
            int64_t now = av_gettime_relative();
            int64_t raw = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
            atomic_store_explicit(&g_src_vread_wc, now, memory_order_relaxed);
            if (raw != AV_NOPTS_VALUE) {
                const AVStream *vst = d->ifmt->streams[d->vstream];
                /* lavf completes a video PES only when the NEXT one starts (unbounded PES length), so the
                 * first video packet out after a gap is the pre-gap TAIL frame (contiguous content, by
                 * construction). The first genuinely post-gap frame is the SECOND packet: classify on it,
                 * against the last packet before the gap (A), with the wall gap of the first (W). */
                /* 2.0.0-pre8.5: a classification in progress finishes even if the hold ended between its two
                 * packets (copied AC-3 + mid-GOP cut: the state flipped back to LIVE after the tail, the class was
                 * never set, the audio waiting for it ran into its 4 s cap and video waited at the gate: 1.9 s dark) */
                if (atomic_load_explicit(&g_src_state, memory_order_relaxed) || d->src_pend) {
                    if (!d->src_classified && d->src_last_vwall) {
                        if (!d->src_pend) {                 /* first packet after the gap: the held tail */
                            d->src_pend     = 1;
                            d->src_pend_w   = now - d->src_last_vwall;
                            d->src_pend_raw = d->src_last_vdts_raw;   /* last packet BEFORE the gap */
                        } else if (d->src_pend < 3 && src_tail_more(vst, raw, d->src_pend_raw)) {
                            /* 2.0.0 T-345: a cut can leave TWO pre-gap tail frames (T-089): the second was classified,
                             * A = +0.04 s → BURST after a 20 s video-only outage (rich50b sync_vnull_20; the jump to the
                             * live content came one packet later). BURST counted the gap toward AUTO-BANK, the armed
                             * bank disabled the wall evidence and the real 20 s hole was erased as a splice: 18 s of
                             * silence, A/V +20 s. Up to two more packets still on the old timeline are tail too; a real
                             * burst stays contiguous and is classified BURST on the next one */
                            d->src_pend++;
                        } else {                            /* second: the first post-gap content */
                            int64_t delta = raw - d->src_pend_raw, A, W = d->src_pend_w;
                            if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {   /* nearest 33-bit wrap */
                                int64_t m = 1LL << vst->pts_wrap_bits;
                                delta &= m - 1;
                                if (delta >= m >> 1) delta -= m;
                            }
                            A = av_rescale_q(delta, vst->time_base, AV_TIME_BASE_Q);
                            int burst = A >= 0 && A < 500000;
                            /* 2.0.0 T-103: the delay this pause would keep (house_skew has grown through the hold) */
                            int64_t kept = (d->house_skew ? *d->house_skew : 0) - d->rj_hs_base;
                            int64_t maxd = g_max_delay_us >= 0 ? g_max_delay_us : g_cushion_max_ms * 1000;
                            /* 2.0.0 T-316 (bursts longer than the ceiling): the delay this return would carry is at least
                             * W — the hold came after the queues ran dry, so the first returning content is W behind the
                             * source. house_skew's growth alone missed the bank the output had played out before the hold
                             * (sync_bursty_20: kept 5.7 s, W 19.9 s): no skip, the burst overflowed the queues and video and
                             * audio each shed different content (59.94: video GOPs, audio +1…+2.4 s late; 25 fps: audio
                             * oldest-first, −6…−9.7 s). A channel with the bank armed keeps the bank's worth of the burst
                             * (the next pause needs it): the cut leaves bank (≤ max delay) of delay, the same content on
                             * every stream; otherwise the skip goes to the live content, as before. */
                            int64_t bank = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
                            int64_t keep = bank > 0 ? FFMIN(bank, maxd) : 0;
                            int skip = burst && d->single && maxd > 0 && FFMAX(kept, W) > maxd;
                            if (skip) {
                                d->rj_skip        = 1;
                                d->rj_skip_target = d->src_pend_raw + av_rescale_q(W - keep, AV_TIME_BASE_Q, vst->time_base);
                                d->rj_skip_wc     = now;
                                d->rj_skip_n      = 0;
                                av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 rejoin: keeping this pause would hold %.1f s of "
                                       "delay (max %.0f s) — skipping the catch-up to %s (PTV_MAX_DELAY_MS=0 keeps it)\n",
                                       FFMAX(kept, W) / 1e6, maxd / 1e6, keep > 0 ? "the bank's depth" : "the live content");
                            }
                            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 rejoin class %s (A=%+.2f s, W=%.2f s)%s\n",
                                   burst ? "BURST" :
                                   llabs(A - W) <= FFMAX(1000000, W / 10) ? "LIVE LOSS" : "NEW DOMAIN",
                                   A / 1e6, W / 1e6, g_src_hold_act ? "" : " (observe-only)");
                            if (g_src_hold_act) {           /* 2.0.0-pre4: settle the hold's arrival gap */
                                if (d->src_gap_pending) {
                                    if (burst)
                                        by_count_gap(d, d->src_gap_pending, now);
                                    d->src_gap_pending = 0;
                                } else
                                    d->src_gap_class = burst ? 1 : 2;   /* the gap packet is still to be dispatched */
                            }
                            d->src_classified = 1;
                            if (!skip) {                    /* a skip publishes its class at the cut */
                                atomic_store_explicit(&g_rj_cls, burst ? 1 :                       /* 2.0.0-pre8.1 */
                                                      llabs(A - W) <= FFMAX(1000000, W / 10) ? 2 : 3, memory_order_relaxed);
                                atomic_fetch_add_explicit(&g_rj_cls_seq, 1, memory_order_release);
                            }
                            d->src_pend       = 0;
                        }
                    }
                } else {
                    d->src_classified = 0;
                    d->src_pend       = 0;
                }
                d->src_last_vdts_raw = raw;
                d->src_last_vwall    = now;   /* timestamped packets only: a broken PES without PTS (corrupt
                                               * resume) must not reset the gap the next packet measures */
            }
        }
        /* 2.0.0 T-103: skipping a pause's catch-up (decided at the rejoin class above). Every stream drops until a video
         * keyframe at the live point (the content the source would be at had it not paused); then the other streams drop
         * what lies before that keyframe, so picture and sound resume at the same content. A source that does not catch
         * up within 2 s (content arriving in real time) resumes where it is: its pause stays as delay, skipping more
         * would only starve the output. The class goes out at the cut, as BURST: the returning content settles when the
         * hold ends. */
        if (d->rj_skip) {
            const AVStream *vst = d->ifmt->streams[d->vstream], *st = d->ifmt->streams[out->stream_index];
            int64_t raw = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts, nw = av_gettime_relative();
            int isv = out->stream_index == d->vstream;
            if (d->rj_skip == 1) {
                int at_live = 0;
                if (isv && raw != AV_NOPTS_VALUE) {
                    int64_t dl = raw - d->rj_skip_target;
                    if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {
                        int64_t m = 1LL << vst->pts_wrap_bits;
                        dl &= m - 1;
                        if (dl >= m >> 1) dl -= m;
                    }
                    at_live = dl >= 0;
                }
                if (isv && raw != AV_NOPTS_VALUE && (out->flags & AV_PKT_FLAG_KEY) && (at_live || nw - d->rj_skip_wc > 2000000)) {
                    d->rj_skip     = 2;
                    d->rj_skip_cut = raw;
                    d->rj_skip_wc  = nw;
                    d->door_v0      = 0;          /* T-316: the held output resumes here — the door measures from it */
                    d->door_v0_arm  = 1;
                    d->door_vs_hold = atomic_load_explicit(&g_ch_vsrc, memory_order_relaxed);
                    if (at_live && d->house_skew)
                        d->rj_hs_base = *d->house_skew;   /* the hold's repeats are not delay any more */
                    /* settle when the hold ends (BURST's rule): the content lands on the house position — nothing for
                     * the rejoin map to move, no jump for LIVE LOSS's settle to wait for (t103a: audio held 4.1 s, dark) */
                    atomic_store_explicit(&g_rj_cls, 1, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g_rj_cls_seq, 1, memory_order_release);
                    av_log(NULL, AV_LOG_WARNING, at_live ?
                           "[PTV-SRC] in0 rejoin: skipped the catch-up (%"PRId64" packets) — resuming at the live content, "
                           "picture and sound together\n" :
                           "[PTV-SRC] in0 rejoin: the source did not catch up within 2 s (%"PRId64" packets skipped) — resuming "
                           "where it is; its pause stays as delay\n", d->rj_skip_n);
                } else {
                    d->rj_skip_n++;
                    av_packet_free(&out);
                    continue;
                }
            } else if (!isv && raw != AV_NOPTS_VALUE) {
                int64_t dl = raw - av_rescale_q(d->rj_skip_cut, vst->time_base, st->time_base);
                if (st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63) {
                    int64_t m = 1LL << st->pts_wrap_bits;
                    dl &= m - 1;
                    if (dl >= m >> 1) dl -= m;
                }
                if (dl < 0) {
                    av_packet_free(&out);
                    continue;
                }
            }
            if (d->rj_skip == 2 && nw - d->rj_skip_wc > 3000000)
                d->rj_skip = 0;
        }
        /* 2.0.0 T-316: the depth door. A burst longer than the max delay arriving while LIVE overflowed the queues, and each
         * queue shed on its own: video whole GOPs at its head and tail, audio its oldest frames — different content per
         * stream, so the output lost lip sync by seconds (sync_bursty_20: 25 fps audio −2.5…−6.5 s, 59.94 video +2.9…+6.4 s).
         * Content already past the queues cannot be taken back consistently (audio runs seconds ahead of video into the
         * delivery gate), so the cut is made here, at arrival: once the content held between this door and the output
         * (the newest video read minus the video on air) passes the max delay, every stream drops what arrives; video
         * resumes at a keyframe once the depth is back 1 s under it, the other streams at that keyframe's content (what
         * arrives late for content before the door closed still passes: a subtitle trails its picture in the mux). The door
         * stands aside only while a rejoin skip (T-103 above) drops to its cut; after the cut it holds the returning burst to
         * the same max (sync_bursty_20 59.94: 16 s came in behind a skip while the hold ended, and the queues shed on their
         * own again). It keeps dropping while the rejoin classifies, so no stray packets carry a second content step. */
        if (d->single && g_src_watch && g_src_hold_act) {
            const AVStream *vst = d->ifmt->streams[d->vstream], *st = d->ifmt->streams[out->stream_index];
            int64_t raw = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts, nw = av_gettime_relative();
            int64_t maxd = g_max_delay_us >= 0 ? g_max_delay_us : g_cushion_max_ms * 1000;
            int isv = out->stream_index == d->vstream;
            int active = d->rj_skip != 1;
            int64_t vs = atomic_load_explicit(&g_ch_vsrc, memory_order_relaxed);
            int64_t vo = atomic_load_explicit(&g_ch_vout_src, memory_order_relaxed), ref;
            int64_t tick = atomic_load_explicit(&g_house_tick_us, memory_order_relaxed);
            int fresh;
            if (tick <= 0 && vst->avg_frame_rate.num > 0)   /* before the first output tick (a start-up burst) */
                tick = av_rescale(1000000, vst->avg_frame_rate.den, vst->avg_frame_rate.num);
            if (vo != d->door_vo) {
                d->door_vo    = vo;
                d->door_vo_wc = nw;
            }
            /* the output's content: what is on air while it plays new content; while it is held (nothing new on air for
             * 0.5 s, or nothing yet at the start), the content it resumes at — the first video let in after a cut made
             * during the hold (a rejoin skip's or this door's), or at the start. A hold without a cut (a real loss: the
             * first packet back is the pre-gap PES tail, then content far ahead) gives no reading: the door stays open,
             * and one already closed reopens at the next keyframe. */
            fresh = vo && nw - d->door_vo_wc < 500000;
            if (fresh)
                d->door_hold = 0;
            else if (!d->door_hold) {
                d->door_hold    = 1;
                d->door_vs_hold = vs;
                d->door_v0      = 0;
                d->door_v0_arm  = !vo;
            }
            if (d->door_v0_arm && vs != d->door_vs_hold) {
                d->door_v0     = vs;
                d->door_v0_arm = 0;
            }
            ref = fresh ? vo : d->door_v0;
            /* T-364: a cut is erased only while the source stays LIVE: a source pause inside it is repeated through,
             * and its wall evidence must stand (sync_bursty_20: the first burst's cut lasted into the silence) */
            if (d->door_startup && (atomic_load_explicit(&g_src_state, memory_order_relaxed) || d->src_pend || d->rj_skip))
                d->door_startup = 0;
            /* T-316 diag. 2.0.0: event-driven — on a state change, and 1/s only while the door is closed or the depth is
             * within 2 s of the max (it was 1/s always: 3 500 lines/h for a door idle 99.9 % of the time) */
            if (g_diag && isv && ((d->door * 8 + active * 4 + d->door_hold * 2 + fresh) != d->dbg_door_st ||
                                  ((d->door || (ref && vs - ref > maxd - 2000000)) && nw - d->dbg_door_wc >= 1000000))) {
                d->dbg_door_wc = nw;
                d->dbg_door_st = d->door * 8 + active * 4 + d->door_hold * 2 + fresh;
                av_log(NULL, AV_LOG_INFO, "[PTV-DOORDBG] door=%d active=%d fresh=%d hold=%d vs=%.3f vo=%.3f v0=%.3f ref=%.3f "
                       "depth=%.3f st=%d rj=%d pend=%d vq=%d\n", d->door, active, fresh, d->door_hold, vs / 1e6, vo / 1e6,
                       d->door_v0 / 1e6, ref / 1e6, ref ? (vs - ref) / 1e6 : -1.0,
                       atomic_load_explicit(&g_src_state, memory_order_relaxed), d->rj_skip, d->src_pend,
                       av_thread_message_queue_nb_elems(d->video_q));
            }
            if (!active || maxd <= 0) {
                d->door = 0;
            } else if (d->door != 1 && isv && raw != AV_NOPTS_VALUE && vs && ref && vs - ref > maxd &&
                       (int64_t)(av_thread_message_queue_nb_elems(d->video_q) +
                                 atomic_load_explicit(&g_frameq_depth, memory_order_relaxed)) * tick >= maxd - 2000000) {
                /* also right after a resume: a whole burst can arrive while the other streams finish that cut
                 * (59.94: it did, and the video queue shed on its own again). The labels alone are not enough: the
                 * frames actually buffered must hold most of the max delay too — a label jump with nothing queued is
                 * not a burst (pre19 gate: a LIVE LOSS rejoin's lone pre-gap tail on air read 20.1 s, a bit-flipped
                 * PTS 93.9 s; the door cut live content: a 0.8 s video PTS hole, lip sync −3.4 s) */
                d->door     = 1;
                d->door_startup = 1;                /* T-364: the source is LIVE — cleared if it pauses in the cut */
                d->door_raw0 = out->pts != AV_NOPTS_VALUE ? out->pts : raw;   /* PTS: the clock vs and the copies use */
                d->door_us0  = vs;
                d->door_wc   = nw;
                d->door_n    = 0;
                av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 burst: %.1f s of content held (max %.0f s) — dropping the "
                       "arriving content on every stream until it is back under\n", (vs - ref) / 1e6, maxd / 1e6);
            } else if (d->door == 1) {
                int resume = 0;
                if (isv && raw != AV_NOPTS_VALUE && (out->flags & AV_PKT_FLAG_KEY)) {
                    int64_t dl = (out->pts != AV_NOPTS_VALUE ? out->pts : raw) - d->door_raw0;
                    if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {
                        int64_t m = 1LL << vst->pts_wrap_bits;
                        dl &= m - 1;
                        if (dl >= m >> 1) dl -= m;
                    }
                    /* a held output that has played out everything let in resumes at this keyframe */
                    resume = !ref || d->door_us0 + av_rescale_q(dl, vst->time_base, AV_TIME_BASE_Q) - ref <= maxd - 1000000;
                }
                if (resume) {
                    d->door     = 2;
                    d->door_cut = raw;
                    d->door_wc  = nw;
                    d->door_v0      = 0;
                    d->door_v0_arm  = 1;
                    d->door_vs_hold = vs;
                    d->door_erased  = 0;
                    if (d->door_startup && d->door_vhi_ok && out->pts != AV_NOPTS_VALUE &&
                        vst->avg_frame_rate.num > 0 && vst->avg_frame_rate.den > 0)
                        door_erase(d, vst, out->pts);
                    /* 2.0.0: say which way the cut goes (T-364/T-368) — only the kept case used to show (WALLEV) */
                    av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 burst: back under the max delay — resuming at a keyframe, "
                           "picture and sound together (%"PRId64" packets dropped; %s)\n", d->door_n, d->door_startup ?
                           "the source stayed live: the cut is erased, no timestamp jump" :
                           "the source paused in the cut: it stays as a hole the output repeats through");
                } else {
                    int early = 0;
                    if (!isv && raw != AV_NOPTS_VALUE) {
                        int64_t dl = raw - av_rescale_q(d->door_raw0, vst->time_base, st->time_base);
                        if (st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63) {
                            int64_t m = 1LL << st->pts_wrap_bits;
                            dl &= m - 1;
                            if (dl >= m >> 1) dl -= m;
                        }
                        early = dl <= 0;      /* content up to the picture that closed the door is on air */
                        if (d->door_vhi_ok && vst->avg_frame_rate.num > 0) {
                            /* T-398: ... up to the end of the newest picture let through, where the erase joins */
                            int64_t va = d->door_vhi + av_rescale_q(1, av_inv_q(vst->avg_frame_rate), vst->time_base);
                            dl = raw - av_rescale_q(va, vst->time_base, st->time_base);
                            if (st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63) {
                                int64_t m = 1LL << st->pts_wrap_bits;
                                dl &= m - 1;
                                if (dl >= m >> 1) dl -= m;
                            }
                            early = dl < 0;
                        }
                    }
                    if (!early) {
                        /* a dropped picture still arrived (as the DUKF / corrupt drops): without the stamp the door's
                         * own drop span read as one stall and escalated the bank to 12 s for hours (cor-3 restart
                         * 2026-10-13: a 13.8 s "stall" that was this door, then BURSTY churn on clean sources) */
                        if (isv)
                            d->by_last_v_wall = nw;
                        if (d->door_startup)
                            door_arrived(d, out->stream_index, nw);
                        d->door_n++;
                        av_packet_free(&out);
                        continue;
                    }
                }
            } else if (d->door == 2 && d->door_erased && raw != AV_NOPTS_VALUE &&
                       (isv || st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)) {
                /* T-398: an erased cut joins at the keyframe's PTS — what lies before it (the keyframe's leading pictures,
                 * the sound before its picture) belongs to the erased span */
                int64_t t  = out->pts != AV_NOPTS_VALUE ? out->pts : raw;
                int64_t dl = t - av_rescale_q(d->door_vb, vst->time_base, st->time_base);
                if (st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63) {
                    int64_t m = 1LL << st->pts_wrap_bits;
                    dl &= m - 1;
                    if (dl >= m >> 1) dl -= m;
                }
                if (dl < 0) {
                    door_arrived(d, out->stream_index, nw);
                    av_packet_free(&out);
                    continue;
                }
            } else if (d->door == 2 && !isv && raw != AV_NOPTS_VALUE) {
                int64_t dl = raw - av_rescale_q(d->door_cut, vst->time_base, st->time_base);
                if (st->pts_wrap_bits > 0 && st->pts_wrap_bits < 63) {
                    int64_t m = 1LL << st->pts_wrap_bits;
                    dl &= m - 1;
                    if (dl >= m >> 1) dl -= m;
                }
                if (dl < 0) {
                    if (d->door_startup)
                        door_arrived(d, out->stream_index, nw);
                    av_packet_free(&out);
                    continue;
                }
            }
            if (d->door == 2 && nw - d->door_wc > 3000000)
                d->door = 0;
            if (isv && out->pts != AV_NOPTS_VALUE) {   /* T-398: the newest picture let through (wrap-aware) */
                int64_t dl = out->pts - d->door_vhi;
                if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {
                    int64_t m = 1LL << vst->pts_wrap_bits;
                    dl &= m - 1;
                    if (dl >= m >> 1) dl -= m;
                }
                if (!d->door_vhi_ok || dl > 0 || dl < -(int64_t)av_rescale_q(10 * AV_TIME_BASE, AV_TIME_BASE_Q, vst->time_base)) {
                    d->door_vhi    = out->pts;
                    d->door_vhi_ok = 1;
                }
            }
        }
        demux_unwrap(d, out);               /* 33-bit source wrap -> monotonic extended ts (ONCE) */
        /* 2.1 T-409: one classifier, one remedy for every stream of the input (ptv_tl_ingest dispatches) */
        if (!d->tl && !(d->tl = ptv_tl_alloc(d->nb_streams_open))) {
            av_packet_free(&out);
            ret = AVERROR(ENOMEM);
            break;
        }
        ret = ptv_tl_ingest(d, out);
        if (ret < 0)
            break;
    }
end:
    ptv_tl_free(&d->tl);                     /* 2.1 T-409: held packets of an open window are dropped at the end */
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
    if (icb_da == d)
        icb_da = NULL;
    av_packet_free(&pkt);
    return NULL;
}

/* Route one post-unwrap input packet to its queue: corrupt-discard, then video
 * (DUKF + genlock + CHAIN tap + video_q) or audio (fan to audio_q[] + copy via
 * demux_pass). Extracted verbatim from demux_thread so the disc-buffer flush can
 * re-inject kept NEW packets through the identical path. Takes ownership of out
 * (frees it). Returns 0, or <0 on a queue-closed error. */
/* v0.9.14 AUTO-BANK escalation: raise the runtime bank target to 1.5x the worst observed stall,
 * capped at PTV_CUSHION_MAX_MS. Arming flips the master rung to blocking pushes (deep-prime
 * semantics), so each subsequent stall's own latency is RETAINED as a compressed-packet bank in
 * video_q instead of draining — the channel self-heals within a stall cycle or two. Idempotent:
 * the target only ever rises (decay handles the way down); every qualifying stall refreshes the
 * decay reference. Runs in the demux thread; readers are the decode pushes (g_bank_pkts) and the
 * gate drains (cap_us) — all relaxed atomics, no ordering needed (advisory values).
 * 0.9.18 M3: the write body (target computation + ceiling advisory + stores + gate rewrite +
 * log) moved verbatim to cushion_escalate(BANK_ESCALATE); the demux-side decay bookkeeping
 * stays here. */
static void ptv_autobank_escalate(DemuxArgs *d, int64_t worst_us, int64_t now)
{
    d->by_bank_last_q = now;
    cushion_escalate(BANK_ESCALATE, worst_us, now);
}

/* 2.0.2 T-398: the depth door erases its own cut. Left to the paths downstream, a cut was erased by whichever one its
 * label step reached — LAYERA with the audio's offset (whole AAC frames: the pictures moved off the house grid), or
 * AGLUE padding the audio by its own label step while the house took whole ticks back (ATRIM) — each leaving a sub-frame
 * remainder as a permanent content A/V step: GYF's production replay walked −4 → +39 ms in an hour of 12 s burst cuts
 * (one step at every cut, 20 ms the largest). The door knows the cut: from the end of the newest picture let through to
 * the keyframe it resumes at, whole pictures. 2.1 T-409 (the timeline): the cut is erased from the program offset (the
 * timeline moves every stream alike); the audio is let through on the same boundaries, its sub-frame rest is the
 * audio's own continuity (padded / trimmed by its labels). */
static void door_erase(DemuxArgs *d, const AVStream *vst, int64_t vb)
{
    int64_t fd = av_rescale_q(1, av_inv_q(vst->avg_frame_rate), vst->time_base);
    int64_t cut = vb - (d->door_vhi + fd), cut_us;
    if (vst->pts_wrap_bits > 0 && vst->pts_wrap_bits < 63) {
        int64_t m = 1LL << vst->pts_wrap_bits;
        cut &= m - 1;
        if (cut >= m >> 1) cut -= m;
    }
    cut_us = av_rescale_q(cut, vst->time_base, AV_TIME_BASE_Q);
    if (cut <= 0 || cut_us > 60 * (int64_t)AV_TIME_BASE || !d->tl)
        return;
    ptv_tl_cut(d, cut_us);                   /* 2.1 T-409: the program offset, every stream alike */
    d->door_erased = 1;
    d->door_vb     = vb;
    av_log(NULL, AV_LOG_INFO, "[PTV-SRC] in0 burst: the cut is erased at the door — %.3f s (%"PRId64" pictures) on every "
           "stream, joined on the pictures' boundaries\n", cut_us / 1e6, cut / fd);
}

/* 2.0.0-pre20 T-364: a packet the depth door drops while the source is LIVE still ARRIVED. The wall-evidence split reads
 * a content jump with no arrivals behind it as a real hole the house repeats through; while the source flows the output
 * plays its queue and repeats nothing, so the cut must read as flowing (erased) — kept, a mass restart's start-up backlog
 * cut left a 13.3 s video PTS hole (audio padded instead) and the ffmpeg HLS segmenter logged ~34 000 timestamp
 * discontinuities. Once the source pauses inside the cut, its wall evidence stands: a burst source's silence is repeated
 * through (stamping there erased 8 s of a sync_bursty_20 return on video only — audio whole-step under the armed bank:
 * lip sync +8 s). pre21: pre20 required the cut to begin before the first picture; cor-3's pre20 restart had Azorse_TV
 * start with 11.6 s queued, cadence holds took it over 12 s once on air → the same 12.9 s hole */
static void door_arrived(DemuxArgs *d, int si, int64_t nw)
{
    d->any_wall_last = nw;
    if (d->wrap_wall_last && si >= 0 && (unsigned)si < d->nb_streams_open)
        d->wrap_wall_last[si] = nw;
}

/* count one completed >=1.5 s video arrival stall toward [PTV-BURSTY] / AUTO-BANK */
static void by_count_gap(DemuxArgs *d, int64_t gap, int64_t now)
{
    d->by_gap_cnt++;
    if (gap > d->by_max_gap) d->by_max_gap = gap;
    if (d->autobank && gap >= 3000000) {    /* BIG stalls (sparse-glitch class) */
        /* 2.0.0-pre20 (owner 2026-10-14): one stall is an incident, a second within 30 min a pattern. A lone stall used
         * to arm the bank for 6 h and fill it: the channel then held 9-12 s of latency (fixture: one 6 s pause, 1.3 ->
         * 9.6 s), or under real delivery jitter churned repeats (cor-3 restart 2026-10-13: 25-490 repeated frames an
         * hour on six clean sources, 0 on cor-1). The bank still arms for the stall itself — it is what keeps picture
         * and sound together while the catch-up burst is absorbed (not arming shed 24 video frames from the full frame
         * queue: sound +801 ms for 5 s) — but a LONE stall's bank does not top up and retires after 10 min of quiet */
        int recent = d->by_big_wc && now - d->by_big_wc <= 1800000000LL;
        int armed  = atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0;
        if (recent || (armed && !atomic_load_explicit(&g_bank_lone, memory_order_relaxed))) {
            if (atomic_exchange_explicit(&g_bank_lone, 0, memory_order_relaxed))
                av_log(NULL, AV_LOG_INFO, "[PTV-BURSTY] a second %.1f s stall within 30 min — a pattern: the bank stays "
                       "and fills, retires after %llds of quiet\n", gap / 1e6, (long long)(g_cp.bank_decay_us / 1000000));
            ptv_autobank_escalate(d, recent ? FFMAX(gap, d->by_big_gap) : gap, now);
        } else {
            atomic_store_explicit(&g_bank_lone, 1, memory_order_relaxed);
            ptv_autobank_escalate(d, gap, now);
            av_log(NULL, AV_LOG_INFO, "[PTV-BURSTY] one %.1f s stall — banked for this one only: no top-up, retires after "
                   "10 min of quiet unless another comes within 30 min\n", gap / 1e6);
        }
        d->by_big_wc  = now;
        d->by_big_gap = gap;
    }
}

/* v0.9.0 genlock estimator: recover the source frame rate as a SLIDING-window FLL. Each ~4s
 * sub-window contributes an UNBIASED rate (Σdc/Σdw over the window — averages out the bursty
 * per-packet delivery jitter that a per-packet Δc/Δw cannot: UDP delivers video in clumps).
 * Each sub-rate folds into an EMA (τ≈64 chunks×4s≈4-5min) that TRACKS slow drift (NOT a
 * latch-forever cumulative mean — crystal drift over a day must be followed), with a per-chunk
 * slew clamp (bounds d(rate)/dt, PCR-friendly) and a wild-chunk reject (±1%, so a glitched
 * window can't bias the rate). Single-input live only; published to ALL rungs via
 * e->src_rate_q20; locks after ~8 chunks (~24s). A disturbance epoch bump (splice/wrap/gap)
 * re-anchors the CURRENT sub-window (discards the partial, can't skew Σdc) but KEEPS the learned
 * rate+lock (the source's physical clock is continuous across a content splice). */
/* 0.9.18 R2 (map §2.4): extracted verbatim from demux_dispatch. Feeds BOTH sensors —
 * the tight FLL (genlock) and the coarse clock-follow estimator — one video packet's
 * (content c_now, wall w_now, disturbance-epoch ep) sample at a time. Publishes via the
 * e->src_rate/cf atomics exactly as before (R3 folded them into the struct). Demux
 * thread only — non-reentrant. */
static void rate_estimator_feed(RateEstimator *e, int64_t c_now, int64_t w_now,
                                int_least64_t ep)
{
    if (e->c0 == AV_NOPTS_VALUE || ep != e->ep_prev) {   /* (re)anchor the current sub-window; keep ema+lock */
        e->ep_prev = ep; e->c0 = c_now; e->w0 = w_now;
    } else {
        int64_t win_w = w_now - e->w0, win_c = c_now - e->c0;
        if (win_w >= g_gl_window_us) {                          /* close a sub-window (default 3s; longer = less aliased) */
            if (win_c > 0) {
                int64_t r  = av_rescale(win_c, 1 << 20, win_w); /* unbiased sub-window rate, Q20 */
                int64_t lo = (1 << 20) - ((1 << 20) / 100);     /* coarse sane gate (±1%) */
                int64_t hi = (1 << 20) + ((1 << 20) / 100);
                /* v0.9.15 CLOCK-FOLLOW coarse estimator: the same unbiased rate, WIDE
                 * envelope (±3%) with its own outlier reject (±3000ppm vs own EMA after
                 * lock) and a slower lock (20 clean chunks ≈ 60s). Sensor-only for the
                 * WUCR follow; the tight FLL below keeps its crystal-scale guards. */
                {
                    int64_t cf_env = ((int64_t)(1 << 20)) * 3 / 100;    /* ±3% */
                    /* v0.9.15.1: reject band 3000→8000ppm — at lock the EMA sits at only
                     * ~72% of a true offset (alpha 1/16 x 20 chunks), so a ±3000 band
                     * rejected every honest window of a clean +12000ppm source and the
                     * estimate deadlocked below truth (NewsNation first deploy). 8000
                     * still rejects burst-alias spikes; max honest gap at lock for the
                     * ±2% follow cap is ~5500. */
                    int64_t cf_rej = ((int64_t)(1 << 20)) * 8 / 1000;   /* ±8000ppm */
                    /* v0.9.15.3: FREEZE + RESET while the auto-bank is armed — clump
                     * delivery aliases the sub-window rates (DTS advances in bursts), and a
                     * burst-poisoned ema must not survive into follow (Unique TV latched
                     * +28450ppm). The BURSTY classifier is exactly the "windows are garbage
                     * here" signal. Re-acquires from scratch if the bank ever decays away. */
                    if (atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0) {
                        if (!e->cf_frozen) {
                            e->cf_frozen = 1;
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-CLOCK] estimator frozen+reset — BURSTY channel (bank armed), "
                                   "clump windows are not a clock measurement\n");
                        }
                        e->cf_ema_q20 = (1 << 20); e->cf_chunks = 0; e->cf_la_acc = e->cf_la_tot = 0;
                        atomic_store_explicit(&e->cf_rate_q20, (int64_t)(1 << 20), memory_order_relaxed);
                        atomic_store_explicit(&e->cf_locked, 0, memory_order_relaxed);
                    } else {
                    e->cf_frozen = 0;
                    e->cf_wins++;
                    if (r > (1 << 20) - cf_env && r < (1 << 20) + cf_env &&
                        (e->cf_chunks < 20 || (r - e->cf_ema_q20 < cf_rej && e->cf_ema_q20 - r < cf_rej))) {
                        e->cf_ema_q20 += (r - e->cf_ema_q20) >> 4;      /* alpha 1/16, tau ~50s */
                        if (e->cf_chunks < 100000) e->cf_chunks++;
                        atomic_store_explicit(&e->cf_rate_q20, e->cf_ema_q20, memory_order_relaxed);
                        if (e->cf_chunks >= 20)
                            atomic_store_explicit(&e->cf_locked, 1, memory_order_relaxed);
                        if (e->cf_chunks > 20) e->cf_la_acc++;
                    }
                    /* v0.9.15.3 stuck-latch unlock: post-lock, if the reject band throws away
                     * most windows for ~2min, the LOCKED estimate is what's wrong (an honest
                     * source accepts nearly every window post-0.9.15.1) — unlock and
                     * re-acquire instead of holding a value reality keeps contradicting. */
                    if (e->cf_chunks > 20 && ++e->cf_la_tot >= 40) {
                        if (e->cf_la_acc < 10) {
                            /* 2.0.0: DIAG — an internal re-acquire, every few minutes on a noisy (healthy) source clock */
                            av_log(NULL, g_diag ? AV_LOG_WARNING : AV_LOG_VERBOSE,
                                   "[PTV-CLOCK] estimator unlatched — locked ema %+lldppm rejected %d/%d "
                                   "recent windows; re-acquiring\n",
                                   (long long)(((e->cf_ema_q20 - (1 << 20)) * 1000000) >> 20),
                                   e->cf_la_tot - e->cf_la_acc, e->cf_la_tot);
                            e->cf_ema_q20 = (1 << 20); e->cf_chunks = 0;
                            atomic_store_explicit(&e->cf_rate_q20, (int64_t)(1 << 20), memory_order_relaxed);
                            atomic_store_explicit(&e->cf_locked, 0, memory_order_relaxed);
                        }
                        e->cf_la_acc = e->cf_la_tot = 0;
                    }
                    /* v0.9.15.1 breadcrumb: if the estimator can't lock, say why (once per
                     * ~3min of windows). v0.9.17: PTV_DIAG-gated — on chronically wandering
                     * sources (AWE-class) the estimator NEVER durably locks by design and
                     * this line becomes permanent chatter (~15-25/h, owner-flagged); the
                     * FOLLOW/release + unlatch events stay always-on and tell the real story. */
                    if (g_diag &&
                        !atomic_load_explicit(&e->cf_locked, memory_order_relaxed) &&
                        e->cf_wins % 60 == 0)
                        av_log(NULL, AV_LOG_INFO,
                               "[PTV-CLOCK] estimator: %d/%d windows accepted, ema %+lldppm (lock needs 20)\n",
                               e->cf_chunks, e->cf_wins,
                               (long long)(((e->cf_ema_q20 - (1 << 20)) * 1000000) >> 20));
                    }
                }
                /* GUARD-B: relative outlier reject — a burst-aliased window that jumps far from the
                 * running estimate is jitter, not a real rate change; skip it (slide the window
                 * anyway). Anchored by GUARD-A's bound so `ema` can't itself wander far. ONLY after
                 * lock (chunks>=8): pre-lock we must ACQUIRE freely, else a jittery source whose
                 * sub-window rates straddle the band would never accumulate the 8 chunks to lock
                 * (genlock would silently disable → revert to the old drift). The runaway is a
                 * post-lock phenomenon, so post-lock rejection is exactly what's needed. */
                int reject = g_genlock_guard && e->chunks >= 8 &&
                             (r - e->ema_q20 > g_gl_reject_q20 || e->ema_q20 - r > g_gl_reject_q20);
                if (r >= lo && r <= hi && !reject) {
                    int64_t step = (r - e->ema_q20) >> g_gl_ema_shift; /* EMA (default α≈1/64 → τ≈4-5min) */
                    int64_t dmax = av_rescale((1 << 20) / 100000, g_gl_window_us, 3000000); /* slew ≈10ppm per 3s-window, scaled with the window → constant ppm/s */
                    if (step > dmax) step = dmax;
                    else if (step < -dmax) step = -dmax;
                    e->ema_q20 += step;
                    /* GUARD-A: hard absolute bound — the applied house-clock rate can never exceed a
                     * physical envelope, so a fooled estimate cannot drive the house_skew runaway. */
                    if (g_genlock_guard) {
                        int64_t emin = (1 << 20) - g_gl_max_q20, emax = (1 << 20) + g_gl_max_q20;
                        if (e->ema_q20 > emax) e->ema_q20 = emax;
                        else if (e->ema_q20 < emin) e->ema_q20 = emin;
                    }
                    atomic_store_explicit(&e->src_rate_q20, e->ema_q20, memory_order_relaxed);
                    if (e->chunks < 100000) e->chunks++;
                    if (e->chunks >= 8)                         /* ~24s+ of clean chunks → trust + apply */
                        atomic_store_explicit(&e->src_rate_locked, 1, memory_order_relaxed);
                }
            }
            e->c0 = c_now; e->w0 = w_now;                      /* slide to the next sub-window */
        }
    }
}

int ptv_demux_dispatch(DemuxArgs *d, AVPacket *pkt)   /* 2.1 T-409: the timeline releases through here */
{
    return demux_dispatch(d, pkt);
}

void ptv_rsync_post_edit(DemuxArgs *d, int s, int64_t delta_us)
{
    rsync_post_edit(d, s, delta_us);
}

static int demux_dispatch(DemuxArgs *d, AVPacket *out)
{
    int ret = 0;
    if ((out->flags & AV_PKT_FLAG_CORRUPT) && g_discardcorrupt) {
        if (d->ts_outl && out->stream_index < d->nb_streams_open)   /* 2.0.2 T-402: the stream's damage phase (demux_unwrap) */
            d->ts_outl[out->stream_index].dmg_wc = av_gettime_relative();
        /* = -fflags +discardcorrupt, ALL streams — but COUNTED (video) so frame loss shows in stats
         * (libavformat's own flag discards silently, hiding the count). Drop before decode: a corrupt
         * frame, like a dropped one, becomes a content gap the position-anchored composite leaps
         * across → desync. PTV_KEEP_CORRUPT=1 disables (lets the decoder try to use them). */
        if (out->stream_index == d->vstream) {
            d->vcorrupt++;
            atomic_store_explicit(&g_vdmg_wc, av_gettime_relative(), memory_order_relaxed);   /* pre9.4 */
            atomic_store_explicit(&g_vcorrupt_wc, av_gettime_relative(), memory_order_relaxed);   /* 2.0.0 T-324 */
            atomic_fetch_add_explicit(&g_vcorrupt_n, 1, memory_order_relaxed);
            /* pre24 #63: a corrupt-discarded video packet IS a delivery arrival — the
             * [PTV-BURSTY]/AUTO-BANK advisor measures DELIVERY stalls (HLS-burst clumping),
             * and leaving these unstamped made every TEI corrupt storm read as arrival
             * stalls: auto-bank escalated on a channel whose transport was flowing
             * perfectly, and the armed bank then disabled the wall-evidence split exactly
             * when it was needed (storm1 fixture, first battery). Content absence is the
             * wall-evidence machinery's business; the bank is for delivery. */
            d->by_last_v_wall = av_gettime_relative();
        } else {
            /* 1.0.1-pre15 #33 §3 (NBS manifestation (c)): audio was discarded here SILENTLY,
             * UNCOUNTED — the track's thread blocks on recv (no recovery logic can run), ADECWD
             * is structurally blind (wd_pkts never advances), the sensor label-walks and the
             * corrector refuses: restart-only. Part 1 (UNCONDITIONAL, even under
             * PTV_NO_GLUECLASS): count per track + rate-limited [PTV-ADISC] + acor= stats.
             * Part 2 (g_glueclass + OPT-IN PTV_NBS_FILL): while the track's packets keep
             * arriving-and-discarding but nothing has decoded for >2s and video is alive, send
             * one FILL sentinel per quantum — the audio thread synthesizes stamped silence at
             * the expected next graph-door pts, so labels stay dense, delivery stays alive and
             * the sensor keeps a valid ≈0 reading instead of a label-walk. */
            int k;
            for (k = 0; k < d->n_audio; k++) {
                int64_t nw;
                if (d->astream[k] != out->stream_index)
                    continue;
                nw = av_gettime_relative();
                d->acorrupt[k]++;
                atomic_fetch_add_explicit(&g_acorrupt, 1, memory_order_relaxed);
                if (d->aglobal[k] >= 0 && d->aglobal[k] < PTV_MAX_AUDIO)   /* 2.0.0-pre9.3 (T-080 A2) */
                    atomic_store_explicit(&g_adamage_wc[d->aglobal[k]], nw, memory_order_relaxed);
                if (nw - d->adisc_win_us[k] >= 10000000) {
                    if (d->adisc_win_n[k])
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-ADISC] a%d corrupt-discard %"PRId64" pkts in last 10s — source "
                               "undecodable phase (total %"PRId64")\n",
                               d->aglobal[k], d->adisc_win_n[k], d->acorrupt[k]);
                    d->adisc_win_us[k] = nw;
                    d->adisc_win_n[k]  = 0;
                }
                d->adisc_win_n[k]++;
                if (g_glueclass && g_nbs_fill &&
                    nw - d->nbs_last_fill_us[k] >= g_nbs_quantum_us) {
                    int64_t fw = atomic_load_explicit(&g_adec_frame_wc[d->aglobal[k]],
                                                      memory_order_relaxed);
                    int64_t va = atomic_load_explicit(&g_v_arrive_wc, memory_order_relaxed);
                    if (fw && nw - fw > 2000000 && va && nw - va < 5000000) {
                        AVPacket *fs = av_packet_alloc();
                        if (fs) {
                            fs->stream_index = out->stream_index;
                            fs->flags |= PTV_PKT_FLAG_NBS_FILL;
                            if (av_thread_message_queue_send(d->audio_q[k], &fs,
                                                             AV_THREAD_MESSAGE_NONBLOCK) < 0)
                                av_packet_free(&fs);   /* queue full/closed — no fill needed/possible */
                        }
                        d->nbs_last_fill_us[k] = nw;
                    }
                }
            }
        }
        av_packet_free(&out);
        return 0;
    }
        if (out->stream_index == d->vstream) {
            if (d->drop_until_kf) {   /* P2 2b: post-splice → drop video until the next IDR (bounded by the escape) */
                if (out->flags & AV_PKT_FLAG_KEY) {
                    d->drop_until_kf = 0;                                  /* IDR → clean resume; send it */
                    /* 1.0.1-pre15 #33 (b4, owner call 2026-07-18): DUKF's post-splice content
                     * drop is ACCEPTED as a bounded (≤GOP) known cost, but it gets always-on
                     * one-line observability — it was g_diag-only, a candidate contributor to
                     * small flushless bakes that never showed in fleet logs. Rare by
                     * construction (arms only on ≥1s video jumps). */
                    av_log(NULL, AV_LOG_INFO,
                        "[PTV-DUKF] resume at keyframe — dropped %"PRId64" video frames over %"PRId64"ms; frame_q now %d (it drained while video was paused)\n",
                        d->vdrop - d->kf_arm_vdrop, (av_gettime_relative() - d->kf_arm_us)/1000,
                        atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
                } else if (av_gettime_relative() - d->kf_arm_us > g_dukf_escape_us) {
                    d->drop_until_kf = 0;                                  /* escape: no IDR in time → don't freeze */
                    av_log(NULL, AV_LOG_WARNING,   /* pre15: always-on (see the resume line above) */
                        "[PTV-DUKF] escape — no IDR within %"PRId64"ms; dropped %"PRId64" video frames; frame_q now %d\n",
                        g_dukf_escape_us/1000, d->vdrop - d->kf_arm_vdrop,
                        atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
                } else {
                    /* pre24 #63: a DUKF-dropped packet is also a delivery arrival (see the
                     * corrupt-branch note) — without the stamp, hole + DUKF span read as one
                     * ≥3s stall and single-stall auto-bank escalation could fire post-hole. */
                    d->by_last_v_wall = av_gettime_relative();
                    d->vdrop++; av_packet_free(&out); return 0;           /* drop the mid-GOP new-timeline burst */
                }
            }
            d->vpkt++;
            {   /* [PTV-BURSTY]/AUTO-BANK — burst detection drives the runtime cushion escalation
                 * (v0.9.14) on qualifying inputs, and remains a log-only advisor otherwise. */
                int64_t bw = av_gettime_relative();
                if (d->by_last_v_wall) {
                    int64_t gap = bw - d->by_last_v_wall;
                    if (gap >= 1500000) {                       /* a completed >=1.5s arrival stall */
                        /* 2.0.0-pre4 (T-056 §3): a gap that was a source HOLD counts toward AUTO-BANK
                         * only if its rejoin class is BURST (delivery paused, content kept); a LIVE
                         * LOSS / NEW DOMAIN is an outage, not burstiness, and must not bank 12 s of
                         * latency for hours. The class is decided on the 2nd post-gap packet, which
                         * may come after this one is dispatched: then the gap waits for it. */
                        if (g_src_hold_act && (atomic_load_explicit(&g_src_state, memory_order_relaxed) ||
                                               d->src_pend || d->src_gap_class)) {
                            if (d->src_gap_class == 1)
                                by_count_gap(d, gap, bw);
                            else if (!d->src_gap_class)
                                d->src_gap_pending = gap;
                            d->src_gap_class = 0;
                        } else
                            by_count_gap(d, gap, bw);
                    }
                }
                d->by_last_v_wall = bw;
                if (!d->by_win_start) d->by_win_start = bw;
                if (bw - d->by_win_start >= 60000000) {         /* 60s window closes */
                    /* per-minute burst visibility (owner-requested): any window with stalls logs
                     * count + worst gap + how the bank is handling it; quiet windows stay silent.
                     * (Current fill level is on the stats line as bank=actual/target.) */
                    if (d->autobank && d->by_gap_cnt > 0) {
                        int64_t bt = atomic_load_explicit(&g_bank_us, memory_order_relaxed);
                        if (bt > 0)
                            av_log(NULL, AV_LOG_INFO,
                                   "[PTV-BURSTY] %d stall%s >=1.5s in the last 60s (worst %.1fs) — auto-bank target %.1fs\n",
                                   d->by_gap_cnt, d->by_gap_cnt == 1 ? "" : "s", d->by_max_gap / 1e6, bt / 1e6);
                        else
                            av_log(NULL, AV_LOG_INFO,
                                   "[PTV-BURSTY] %d stall%s >=1.5s in the last 60s (worst %.1fs) — below auto-bank threshold\n",
                                   d->by_gap_cnt, d->by_gap_cnt == 1 ? "" : "s", d->by_max_gap / 1e6);
                    }
                    if (d->by_gap_cnt >= 3) {                   /* HLS-burst cadence */
                        if (d->autobank) {
                            if (atomic_exchange_explicit(&g_bank_lone, 0, memory_order_relaxed))   /* a cadence: a pattern */
                                av_log(NULL, AV_LOG_INFO, "[PTV-BURSTY] stalls in a cadence — a pattern: the bank stays and "
                                       "fills, retires after %llds of quiet\n", (long long)(g_cp.bank_decay_us / 1000000));
                            ptv_autobank_escalate(d, d->by_max_gap, bw);
                        } else if ((int64_t)g_preroll_ms < (d->by_max_gap * 12 / 10) / 1000) {
                            int64_t need_ms = (d->by_max_gap * 3 / 2) / 1000;   /* 1.5x worst gap */
                            need_ms = ((need_ms + 999) / 1000) * 1000;
                            if (need_ms > 30000) need_ms = 30000;
                            int vq = (int)(need_ms / 1000 * g_cp.vid_pps) + 64; /* v0.9.18.4 M6: video pkt/s from out_fps */
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-BURSTY] video arrives in bursts: %d stalls >=1.5s in the last 60s "
                                   "(worst %.1fs). The frame cushion cannot ride gaps this size — this looks "
                                   "like HLS-segment delivery over SRT. Add to the channel environment and "
                                   "restart: PTV_PREROLL_MS=\"%lld\",PTV_VIDEOQ=\"%d\" "
                                   "(deep packet prime; adds ~%llds latency). See -log-legend.\n",
                                   d->by_gap_cnt, d->by_max_gap / 1e6,
                                   (long long)need_ms, vq, (long long)(need_ms / 1000));
                        }
                    }
                    /* v0.9.14 decay: a long quiet spell retires the bank (latency then bleeds via
                     * the normal catch-up path once the master rung returns to drop-oldest). */
                    if (g_diag && atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0)
                        av_log(NULL, AV_LOG_INFO, "[PTV-BANK] window close: quiet=%.1fs decay_after=%llds gap_cnt=%d\n",
                               d->by_bank_last_q ? (bw - d->by_bank_last_q) / 1e6 : -1.0,
                               (long long)(g_bank_decay_us / 1000000), d->by_gap_cnt);
                    if (d->autobank && d->by_bank_last_q &&
                        atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0 &&
                        bw - d->by_bank_last_q > (atomic_load_explicit(&g_bank_lone, memory_order_relaxed) ?
                                                  600000000LL : g_cp.bank_decay_us))
                        cushion_escalate(BANK_RETIRE, atomic_load_explicit(&g_bank_lone, memory_order_relaxed) ?
                                         600000000LL : g_cp.bank_decay_us, 0);   /* 0.9.18 M3: write body moved */
                    d->by_win_start = bw; d->by_gap_cnt = 0; d->by_max_gap = 0;
                }
                atomic_store_explicit(&g_vq_elems, av_thread_message_queue_nb_elems(d->video_q), memory_order_relaxed);
            }
            if (out->pts != AV_NOPTS_VALUE)   /* [PTV-CHAIN] video source-content at demux (us) */
                atomic_store_explicit(&g_ch_vsrc, av_rescale_q(out->pts, d->ifmt->streams[d->vstream]->time_base, AV_TIME_BASE_Q), memory_order_relaxed);
            /* v0.9.0 genlock estimator + v0.9.15 coarse clock-follow — body moved verbatim
             * to rate_estimator_feed() above (0.9.18 R2). Single-input live only. */
            if (g_genlock && g_genlock_ok && out->dts != AV_NOPTS_VALUE)
                rate_estimator_feed(d->est,
                    av_rescale_q(out->dts, d->ifmt->streams[d->vstream]->time_base, AV_TIME_BASE_Q),
                    av_gettime_relative(),
                    d->disturb_epoch ? atomic_load_explicit(d->disturb_epoch, memory_order_relaxed) : 0);
            /* 1.0.1-pre8: input-flowing signal for the (b)/(c) starvation-contradiction detectors
             * — a video packet reached the demux dispatch just now (clean-wire evidence).
             * 1.0.1-pre17: ALSO stamped per input (Input.v_arrive_wc) — the corrector's §3
             * per-input liveness; the global stays as the any-input aggregate for the
             * single-input detectors above (identical there: one input). */
            atomic_store_explicit(&g_v_arrive_wc, av_gettime_relative(), memory_order_relaxed);
            if (d->v_arrive_wc)
                atomic_store_explicit(d->v_arrive_wc, av_gettime_relative(), memory_order_relaxed);
            /* 2.0.0-pre6 (T-056 §5.1, D24/D25): video flows but a transcoded track's packets stopped for
             * 2 s (measured: above the 1-2 s interleave blips, below the 3 s+ real holes) — the NBS
             * silence fill, default-on: its thread is blocked on an empty audio_q, so this (the live
             * demux thread) sends the sentinels; labels advance with wall time (house_skew is not
             * moving while video flows). PTV_NO_SRC_FILL=1 off. Only silence DURING a
             * gap-free video run counts: after an input gap the track's last packet is as old as the gap,
             * and filling there (pre-gap label continuation) stepped the door back by the gap at a rejoin.
             * 2.0.0 T-093: multiview slots too (each slot is its own input) — Grid_2x2 live 2026-10-09: a slot
             * source stopped its audio while its video flowed, the slot's audio PID went dead on the wire 4.5 min
             * and sync_check restarted the mosaic (DEAD_AUDIO_TRACK). The copied-track watch stays single input. */
            if (g_hold_fill && g_glueclass) {
                int64_t nw = av_gettime_relative();
                int k;
                if (!d->v_last_arr_us || nw - d->v_last_arr_us > 1000000)
                    d->v_flow_since_us = nw;
                d->v_last_arr_us = nw;
                for (k = 0; k < d->n_audio; k++) {
                    AVPacket *fs;
                    /* 2.0.0 (T-099 class): 1 s while the input is DAMAGED (corrupt video or a corrupt-discarded packet on
                     * this track within 5 s) — the 1-2 s interleave blips the 2 s threshold rides over are clean-source
                     * behaviour; under loss a missing audio stretch is lost audio, and waiting 2 s left the track's PID
                     * dark 1.8 s on the wire (AWE capture, 10 % loss: source audio absent 2.56 s while video flowed) */
                    int64_t dw = d->aglobal[k] >= 0 && d->aglobal[k] < PTV_MAX_AUDIO ?
                                 atomic_load_explicit(&g_adamage_wc[d->aglobal[k]], memory_order_relaxed) : 0;
                    if (d->single)
                        dw = FFMAX(dw, atomic_load_explicit(&g_vcorrupt_wc, memory_order_relaxed));
                    if (!d->a_arr_us[k] || nw - FFMAX(d->a_arr_us[k], d->v_flow_since_us) <
                                           (dw && nw - dw < 5000000 ? 1000000 : 2000000) ||
                        nw - d->nbs_last_fill_us[k] < g_nbs_quantum_us)
                        continue;
                    d->nbs_last_fill_us[k] = nw;
                    if (!(fs = av_packet_alloc())) continue;
                    fs->stream_index = d->astream[k];
                    fs->flags |= PTV_PKT_FLAG_NBS_FILL;
                    if (av_thread_message_queue_send(d->audio_q[k], &fs, AV_THREAD_MESSAGE_NONBLOCK) < 0)
                        av_packet_free(&fs);
                }
                if (g_copy_gapfill && d->single && d->n_pass && !ptv_src_holding())   /* 2.0.0-pre9.4 (T-083) */
                    copy_gap_watch(d, nw);
            }
            /* 1.0.1-pre10 review fix (rr10 D1): measure THIS input's video arrival rate — the
             * catch-up governor's pacing currency (the master OUT tick was the wrong currency:
             * any input whose pps exceeds 1.25x out-fps was governed below its own arrival
             * rate and the governor manufactured a permanent shed cycle). 4s windows; a >1s
             * arrival gap restarts the window so an outage can never dilute the measured rate
             * (under-measuring is the dangerous direction — it would under-cap the drain). */
            {
                int64_t anw = av_gettime_relative();
                if (!d->vin_win_us || anw - d->vin_last_us > 1000000) {
                    d->vin_win_us = anw; d->vin_win_cnt = 0;   /* startup or a gap: fresh window */
                } else {
                    d->vin_win_cnt++;
                    if (anw - d->vin_win_us >= 4000000) {
                        atomic_store_explicit(&d->vin_pps,          /* round to nearest: a floor would
                                                                     * under-cap low-fps inputs */
                            (int)((d->vin_win_cnt * 1000000LL + (anw - d->vin_win_us) / 2) /
                                  (anw - d->vin_win_us)),
                            memory_order_relaxed);
                        atomic_store_explicit(&d->vin_pps_wall, anw, memory_order_relaxed);
                        /* pre13: publish age gates trust — a droughty dispatch pattern (>1s gaps
                         * more often than every 4s, e.g. a stalled/clumped source) stops publishes
                         * entirely, and the governor must not keep pacing on a frozen value. */
                        d->vin_win_us = anw; d->vin_win_cnt = 0;
                    }
                }
                d->vin_last_us = anw;
            }
            /* 1.0.1-pre10 (h) DEGRADED MODE (opt-in PTV_DEGRADED=1 — no-op otherwise): while a
             * sustained decode deficit keeps QSHED full-cycling (entry armed below at the
             * tail-arm site after a >=3min train), admission goes DEMAND-DRIVEN at the live
             * edge: an arriving GOP is admitted only when the queued backlog is nearly
             * consumed (video_q <= ~1s) — the queue depth IS the decode-throughput
             * measurement, no estimator needed, and retained latency self-scales with the
             * deficit (~2s of content / utilization) instead of accumulating. (The first
             * modulus-K cut kept vq at 8s of content = 60s of DECODE TIME at a 12% box —
             * hs grew +59s and audio, A/V-locked to that delay, overflowed at the demux door
             * at ~40/s; the defaults churn keeps hs bounded 8-14s by head-shedding old
             * content, which audio demonstrably rides.) All decisions happen at IDR
             * boundaries only (admission never flips mid-GOP — the queue stays GOP-coherent);
             * release = 60s of CONTINUOUS decode headroom (frame_q un-starved with vq
             * shallow); re-entry needs a fresh 3min full-cycle train (hysteresis). */
            if (g_degraded && d->drop && d->deg_active) {
                int64_t dnw = av_gettime_relative();
                if (out->flags & AV_PKT_FLAG_KEY) {
                    int vqe = av_thread_message_queue_nb_elems(d->video_q);
                    d->deg_admit = vqe <= g_cp.vid_pps;
                    if (vqe < g_cp.videoq_pkts / 4 &&
                        atomic_load_explicit(&g_frameq_depth, memory_order_relaxed) > 2) {
                        if (!d->deg_head_ok_us) d->deg_head_ok_us = dnw;
                        else if (dnw - d->deg_head_ok_us >= 60LL * 1000000) {
                            d->deg_active = 0; d->deg_train_us = 0; d->deg_admit = 1;
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-DEGRADED] released: 60s of decode headroom — full admission "
                                   "resumes at this IDR (%"PRId64" pkts dropped while degraded)\n",
                                   d->deg_dropped);
                        }
                    } else
                        d->deg_head_ok_us = 0;
                }
                if (d->deg_active && !d->deg_admit) {
                    d->deg_dropped++;
                    if (dnw - d->deg_log_us >= 30000000) {
                        d->deg_log_us = dnw;
                        av_log(NULL, AV_LOG_WARNING,
                               "[PTV-DEGRADED] demand admission at the live edge (vq %d/%d; "
                               "%"PRId64" pkts dropped total)\n",
                               av_thread_message_queue_nb_elems(d->video_q),
                               g_cp.videoq_pkts, d->deg_dropped);
                    }
                    av_packet_free(&out);
                    return 0;
                }
            }
            /* 2.0.0 T-378: the newest video content read (the copies' timeline) — a copy far past it is a corrupt timestamp */
            if (out->dts != AV_NOPTS_VALUE || out->pts != AV_NOPTS_VALUE) {
                d->vread_us  = av_rescale_q(out->dts != AV_NOPTS_VALUE ? out->dts : out->pts,
                                            d->ifmt->streams[d->vstream]->time_base, AV_TIME_BASE_Q);
                d->vread_set = 1;
            }
            /* 1.0.1-pre8 (a) GOP-COHERENT VIDEO OVERFLOW (the #32 wedge core fix). The old
             * policy tail-dropped the ARRIVING packet per-packet, MID-GOP, so under sustained
             * overflow ~70% of packets vanished at random and the decoder received GOP
             * fragments — it then ran at ~11% of realtime and the queue never drained: the
             * drop policy fragmented its own input forever on a clean wire (live-proven,
             * cor-3 2026-07-15). New policy, two coherent halves:
             *   HEAD: request the decoder to flush the queue head to the next keyframe
             *         boundary (whole oldest GOPs — only the consumer can pop the head);
             *   TAIL: while full, drop the arriving stream to the next IDR that fits, so the
             *         queue never holds a headless GOP.
             * Every enqueued GOP stays contiguous and decodable. PTV_NO_QSHED reverts. */
            if (g_qshed && d->drop) {
                int64_t nw = av_gettime_relative();
                if (d->vq_tail_drop) {
                    /* Session-109 time escape (rr8 review defect 1): an intra-refresh / no-IDR
                     * source (or GOP longer than the queue) never delivers the KEY this mode
                     * waits for — without a deadline the demux stops sending FOREVER: permanent
                     * freeze-frame with live audio. After g_dukf_escape_us without an IDR,
                     * resume sending non-IDR packets (the decoder re-syncs on what arrives —
                     * strictly better than freezing; identical to DUKF's escape semantics). */
                    int esc = (nw - d->qshed_tail_arm_us > g_dukf_escape_us);
                    int r3 = AVERROR(EAGAIN);   /* non-key: never attempt (stay GOP-coherent) */
                    if ((out->flags & AV_PKT_FLAG_KEY) || esc)
                        r3 = av_thread_message_queue_send(d->video_q, &out,
                                                          AV_THREAD_MESSAGE_NONBLOCK);
                    if (r3 < 0 && r3 != AVERROR(EAGAIN)) {   /* queue closed → terminate like demux_send */
                        av_packet_free(&out);
                        return r3;
                    }
                    if (r3 >= 0) {              /* IDR + space → clean GOP boundary re-entry */
                        if (esc && !(out->flags & AV_PKT_FLAG_KEY))
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-QSHED] tail escape after %"PRId64"s without IDR — resuming "
                                   "non-IDR (dropped %"PRId64" pkts this episode; total tail %"PRId64" pkts)\n",
                                   g_dukf_escape_us / 1000000, d->qshed_tail_n, d->qshed_tail_tot);
                        else if (nw - d->qshed_log_us >= 5000000) {
                            d->qshed_log_us = nw;
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-QSHED] video_q overflow: tail-dropped %"PRId64" pkts to the IDR "
                                   "boundary — enqueue resumed GOP-coherent (total tail %"PRId64" pkts)\n",
                                   d->qshed_tail_n, d->qshed_tail_tot);
                        }
                        d->vq_tail_drop = 0;
                        d->qshed_tail_n = 0;
                        ret = 0;
                    } else {
                        if (esc)                /* escaped but the queue is genuinely full again:
                                                 * re-arm the deadline so every freeze episode
                                                 * stays bounded by the escape, never a full GOP */
                            d->qshed_tail_arm_us = nw;
                        if (d->vq_shed_req)
                            atomic_store_explicit(d->vq_shed_req, 1, memory_order_relaxed);
                        d->vdrop++; d->qshed_tail_n++; d->qshed_tail_tot++;
                        atomic_store_explicit(&g_shed_wall, nw, memory_order_relaxed);
                        atomic_fetch_add_explicit(&g_shed_cnt, 1, memory_order_relaxed);
                        if (d->shed_wall) atomic_store_explicit(d->shed_wall, nw, memory_order_relaxed);
                        if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, 1, memory_order_relaxed);
                        av_packet_free(&out);
                        ret = 0;
                    }
                } else {
                    ret = av_thread_message_queue_send(d->video_q, &out, AV_THREAD_MESSAGE_NONBLOCK);
                    if (ret == AVERROR(EAGAIN)) {   /* full → GOP-coherent shed engages */
                        if (d->vq_shed_req)
                            atomic_store_explicit(d->vq_shed_req, 1, memory_order_relaxed);
                        d->vq_tail_drop = 1;
                        d->qshed_tail_arm_us = nw;   /* arm the Session-109 escape deadline */
                        d->qshed_tail_n = 1; d->qshed_tail_tot++;
                        d->vdrop++;
                        /* 1.0.1-pre10 (h): full-cycle train tracking for DEGRADED entry. Each
                         * tail-arm is one full-cycle; cycles <=30s apart (measured median 6.2s)
                         * form a train, and a >=3min train is a sustained capacity deficit the
                         * shed/refill cycle cannot exit — enter every-Kth-GOP admission. A
                         * full-cycle occurring WHILE degraded means K is still too generous. */
                        if (g_degraded) {
                            if (!d->deg_train_us || nw - d->deg_last_full_us > 30000000)
                                d->deg_train_us = nw;
                            d->deg_last_full_us = nw;
                            if (!d->deg_active && nw - d->deg_train_us >= 180LL * 1000000) {
                                d->deg_active = 1;
                                d->deg_admit = 1; d->deg_head_ok_us = 0;
                                /* live-edge start: flush the accumulated stale backlog via the
                                 * pre8 re-prime (decode thread drops video_q + resumes at the
                                 * next IDR = our next admitted GOP) — degraded playout must
                                 * ride fresh content, not a 13s-old queue.
                                 * rr10 advisory 4: with PTV_NO_SELFHEAL this flush is SKIPPED —
                                 * degraded admission then rides the stale backlog until demand
                                 * admission drains it (slower live-edge convergence, no wedge).
                                 * Documented interaction, changelog (7h). Only the single-input
                                 * decode thread consumes this request — PTV_DEGRADED is
                                 * single-input only (rr10 D2, gated at transcode() setup). */
                                if (g_selfheal)
                                    atomic_store_explicit(&g_selfheal_req, 1, memory_order_relaxed);
                                av_log(NULL, AV_LOG_WARNING,
                                       "[PTV-DEGRADED] entering: QSHED full-cycles persisted %"PRId64"s — "
                                       "demand admission at the live edge until 60s of decode headroom; "
                                       "backlog flush requested (PTV_DEGRADED=1 experiment lever)\n",
                                       (nw - d->deg_train_us) / 1000000);
                            } else if (d->deg_active)
                                d->deg_head_ok_us = 0;   /* a full-cycle while degraded = no headroom */
                        }
                        atomic_store_explicit(&g_shed_wall, nw, memory_order_relaxed);
                        atomic_fetch_add_explicit(&g_shed_cnt, 1, memory_order_relaxed);
                        if (d->shed_wall) atomic_store_explicit(d->shed_wall, nw, memory_order_relaxed);
                        if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, 1, memory_order_relaxed);
                        if (nw - d->qshed_log_us >= 5000000) {
                            d->qshed_log_us = nw;
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-QSHED] video_q full (%d pkts) — shedding whole GOPs: head flush "
                                   "requested from the decoder, arriving pkts dropped to the next IDR\n",
                                   av_thread_message_queue_nb_elems(d->video_q));
                        }
                        av_packet_free(&out);
                        ret = 0;
                    } else if (ret < 0) {
                        av_packet_free(&out);       /* queue closed */
                        if (d->reopen)              /* 2.0.2 T-390: a live input never ends silently */
                            av_log(NULL, AV_LOG_ERROR, "[PTV-SRC] in%d the video decoder stopped (%s) — "
                                   "the input ends\n", d->rsync_slot, av_err2str(ret));
                    }
                }
            } else
                ret = demux_send(d->video_q, out, d->drop, &d->vdrop);
        } else {
            /* Fan one source PID to every transcoded audio track on it (a clone
             * each), then hand the original to demux_pass (copy-passthrough; it
             * frees it, whether or not it's a copy stream). demux_unwrap ran ONCE
             * above, so every clone carries the same unwrapped ts (load-bearing:
             * never unwrap per-clone — the per-stream wrap state is stateful). */
            if (d->n_audio > 0 && out->stream_index == d->astream[0] && out->pts != AV_NOPTS_VALUE)  /* [PTV-CHAIN] primary-audio source-content at demux (us) */
                atomic_store_explicit(&g_ch_asrc, av_rescale_q(out->pts, d->ifmt->streams[out->stream_index]->time_base, AV_TIME_BASE_Q), memory_order_relaxed);
            if (d->n_audio > 0 && out->stream_index == d->astream[0] && out->pts != AV_NOPTS_VALUE) {
                /* 2.0.0 T-378: the read head follows the audio too — a video-only outage stops the video reads while the
                 * subtitles flow on with the audio */
                int64_t au = av_rescale_q(out->pts, d->ifmt->streams[out->stream_index]->time_base, AV_TIME_BASE_Q);
                d->vread_us  = d->vread_set ? FFMAX(d->vread_us, au) : au;
                d->vread_set = 1;
            }
            int k;
            for (k = 0; k < d->n_audio; k++) {
                AVPacket *c;
                if (d->astream[k] != out->stream_index) continue;
                if (!(c = av_packet_clone(out))) continue;
                d->apkt++;
                d->a_arr_us[k] = av_gettime_relative();   /* 2.0.0-pre6: absence trigger */
                if (d->single)
                    atomic_store_explicit(&g_src_aread_wc, d->a_arr_us[k], memory_order_relaxed);
                /* 1.0.1-pre8 (a): audio overflow sheds whole frames OLDEST-first (audio frames
                 * are independent — no GOP structure). The old drop-NEWEST pinned the stalest
                 * content in the queue; keeping the freshest drains latency instead. */
                if (g_qshed && d->drop) {
                    int r2 = av_thread_message_queue_send(d->audio_q[k], &c, AV_THREAD_MESSAGE_NONBLOCK);
                    if (r2 == AVERROR(EAGAIN)) {
                        AVPacket *oldp;
                        if (av_thread_message_queue_recv(d->audio_q[k], &oldp,
                                                         AV_THREAD_MESSAGE_NONBLOCK) >= 0) {
                            av_packet_free(&oldp);
                            d->adrop++;
                            atomic_store_explicit(&g_shed_wall, av_gettime_relative(), memory_order_relaxed);
                            atomic_fetch_add_explicit(&g_shed_cnt, 1, memory_order_relaxed);
                            if (d->shed_wall) atomic_store_explicit(d->shed_wall, av_gettime_relative(), memory_order_relaxed);
                            if (d->shed_cnt)  atomic_fetch_add_explicit(d->shed_cnt, 1, memory_order_relaxed);
                        }
                        if (av_thread_message_queue_send(d->audio_q[k], &c,
                                                         AV_THREAD_MESSAGE_NONBLOCK) < 0) {
                            av_packet_free(&c);
                            d->adrop++;
                        }
                    } else if (r2 < 0)
                        av_packet_free(&c);
                } else
                    demux_send(d->audio_q[k], c, d->drop, &d->adrop);
            }
            ret = demux_pass(d, out);       /* copy fan + monotonic-DTS clamp; frees out */
        }
        return ret;
}

