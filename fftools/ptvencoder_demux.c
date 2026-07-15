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

#include "ptvencoder.h"

/* DIRECTIONAL discontinuity-absorber thresholds (v0.7.7, §5.A.1). The absorber re-bases a source
 * DTS jump to a continuous timeline. FORWARD jumps default to 1000ms (was 80ms): a recurring small
 * FORWARD video-only frame-drop (observed +90ms on TruBLU, no audio pair) was being absorbed →
 * compressing video's timeline ~57ms each → audio progressively BEHIND (measured ~+150ms/hr drift).
 * At 1000ms those flow through unabsorbed (the player holds the last frame across the pts gap, on
 * the true timeline → A/V stays aligned). BACKWARD jumps keep 80ms: a backward jump unabsorbed would
 * step aresample=async's input backward → audio STALL (the v0.6.23 / task#23 whole-channel outage) —
 * so backward MUST still absorb. (Real ad-splices are seconds, far above either threshold, and still
 * absorb in both directions.) thresholds internalized 0.9.18.7 (were PTV_DISCONT_MS forward / PTV_DISCONT_BACK_MS backward). */
static int     g_discont_ms = 1000;       /* forward jump threshold (internalized 0.9.18.7; was PTV_DISCONT_MS) */
static int     g_discont_back_ms = 80;    /* backward jump threshold (keep small — anti-stall) (internalized 0.9.18.7; was PTV_DISCONT_BACK_MS) */
static int64_t g_gap_min_us = 700000;     /* min wall-absence (us) to call a forward audio jump a real GAP when video did not also cross (internalized 0.9.18.7; was PTV_GAP_MIN_MS) */
static int64_t g_progoff_debounce_us = 1000000;   /* coalesce a V/A straddle into one bump (internalized 0.9.18.7; was PTV_PROGOFF_DEBOUNCE_MS) */
static int64_t g_disc_viderr_sum;    /* PTV-FLUSHAV: running total of per-flush vid_err (source A/V misalignment absorbed at glues, us) — correlate its growth vs the oracle drift to test whether flushes leak into audio-behind */
static int64_t g_dukf_escape_us = 5000000;   /* force-resume if no IDR within this (internalized 0.9.18.7; was PTV_DUKF_ESCAPE_MS) */
/* P2 2b (v0.7.3): arm drop-until-keyframe only on a LARGE jump (a real ad-splice), not on sub-second
 * jitter. (v0.7.7: the forward absorber threshold g_discont_ms is now 1000ms = g_dukf_min_ms, so the
 * +90ms forward jitter below no longer absorbs OR arms DUKF — it flows through. This comment predates
 * that; backward jumps still absorb at g_discont_back_ms=80ms.) The absorber re-base is fine for small,
 * harmless timeline re-base — but DUKF *drops video to the next IDR*, so a sub-second blip (observed:
 * a +90ms VIDEO-ONLY jitter event on TruBLU, no audio pair → not a real splice) would needlessly drop
 * up to a GOP. Gate the video-drop on a separate, higher threshold (real splices on the box were
 * ≥120s; anything ≥~1s is a genuine timeline change). Internalized 0.9.18.7 (was PTV_DUKF_MIN_MS). */
static int     g_dukf_min_ms = 1000;
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

/* ========== legacy-0004 TS-discontinuity buffer (g_layera only) ========== */

/* demux_dispatch routes a post-unwrap dense packet (video/audio) to its queue,
 * exactly as the demux_thread normal path does. The disc-buffer flush re-injects
 * each kept NEW packet through it, mirroring 0004's flush→demux_send. Forward-
 * declared because the flush below calls it; defined just before demux_thread. */
static int demux_dispatch(DemuxArgs *d, AVPacket *out);

/* Estimate a packet's duration in us from codec params, falling back to
 * pkt->duration. Used to advance last_sent_dts to the END of a packet so NEW
 * content starts after OLD content ends (faithful to 0004's helper). */
static int64_t ptv_disc_pkt_duration(AVStream *st, AVPacket *pkt)
{
    AVCodecParameters *par = st->codecpar;
    int64_t dur = 0;

    if (par->codec_type == AVMEDIA_TYPE_AUDIO && par->sample_rate && par->frame_size)
        dur = ((int64_t)AV_TIME_BASE * par->frame_size) / par->sample_rate;
    else if (par->codec_type == AVMEDIA_TYPE_VIDEO) {
        if (st->avg_frame_rate.num)
            dur = av_rescale_q(1, av_inv_q(st->avg_frame_rate), AV_TIME_BASE_Q);
        else if (st->r_frame_rate.num)
            dur = av_rescale_q(1, av_inv_q(st->r_frame_rate), AV_TIME_BASE_Q);
    }
    if (dur == 0 && pkt->duration > 0)
        dur = av_rescale_q(pkt->duration, st->time_base, AV_TIME_BASE_Q);
    return dur;
}

int ptv_disc_init(PtvDiscBuf *b, int capacity, int nb_streams)
{
    int i;
    memset(b, 0, sizeof(*b));
    b->packets = av_calloc(capacity, sizeof(*b->packets));
    if (!b->packets)
        return AVERROR(ENOMEM);
    b->stream_transitioned = av_calloc(nb_streams, sizeof(*b->stream_transitioned));
    b->stream_state        = av_calloc(nb_streams, sizeof(*b->stream_state));
    if (!b->stream_transitioned || !b->stream_state) {
        av_freep(&b->packets);
        av_freep(&b->stream_transitioned);
        av_freep(&b->stream_state);
        return AVERROR(ENOMEM);
    }
    b->capacity   = capacity;
    b->nb_streams = nb_streams;
    b->cycle_trigger = -1;
    for (i = 0; i < nb_streams; i++) {
        b->stream_state[i].cumulative_ts_offset = 0;
        b->stream_state[i].last_sent_dts        = AV_NOPTS_VALUE;
        b->stream_state[i].last_dts_us          = AV_NOPTS_VALUE;
        b->stream_state[i].old_timeline_base    = AV_NOPTS_VALUE;
        b->stream_state[i].new_timeline_base    = AV_NOPTS_VALUE;
    }
    return 0;
}

/* Reset per-cycle state after a flush WITHOUT clearing cumulative_ts_offset /
 * last_sent_dts (those persist across discontinuity cycles). */
static void ptv_disc_reset(PtvDiscBuf *b)
{
    int i;
    for (i = 0; i < b->nb_packets; i++) {
        if (b->packets[i]) {
            av_packet_free(&b->packets[i]->pkt);
            av_freep(&b->packets[i]);
        }
    }
    b->nb_packets = 0;
    if (b->stream_transitioned)
        memset(b->stream_transitioned, 0, b->nb_streams * sizeof(*b->stream_transitioned));
    for (i = 0; i < b->nb_streams; i++) {
        b->stream_state[i].old_timeline_base = AV_NOPTS_VALUE;
        b->stream_state[i].new_timeline_base = AV_NOPTS_VALUE;
        b->stream_state[i].has_old_base      = 0;
        b->stream_state[i].has_new_base      = 0;
    }
    b->active            = 0;
    b->jump_detected     = 0;
    b->buffer_start_time = 0;
    b->applied_offset    = 0;
    b->cycle_trigger     = -1;
}

/* 1.0.1-pre4 shared flush: close the pairing window — forget the event's video-defined offset
 * and every stream's provisional applied offset. Called when the window expires or when a new
 * video crossing starts a new event. (NOT part of ptv_disc_reset: pair state must survive the
 * per-cycle reset — the whole point is pairing ACROSS flush cycles.) */
static void ptv_disc_pair_reset(PtvDiscBuf *b)
{
    int i;
    b->pair_start_us    = 0;
    b->pair_vid_defined = 0;
    b->pair_vid_off_us  = 0;
    for (i = 0; i < b->nb_streams; i++) {
        b->stream_state[i].pair_applied_us = 0;
        b->stream_state[i].pair_has        = 0;
        b->stream_state[i].pair_prov       = 0;
    }
}

/* 1.0.1-pre5 (D1): register the audio label step a shared flush just routed to `stream_idx`'s
 * content path, so AGLUE treats the arriving step as a REAL A-vs-V alignment step (APPLY —
 * aresample converges content) instead of relabel-ERASING it (the 0.9.16.4 backward rule,
 * correct for plain source steps, re-broke the invariant for routed backward mismatches in
 * (-1000ms,-500ms): the fx-mir2 class, video jumping further forward than audio). Value is
 * stored first, deadline last (release) — the audio thread acquires the deadline before
 * reading the value. Only transcoded tracks are wired; copy-only audio (AC-3 passthrough) has
 * no content machinery — see the changelog's known-bound note. */
static void ptv_pair_expect(DemuxArgs *d, int stream_idx, int64_t step_us)
{
    int j;
    for (j = 0; j < d->n_audio; j++) {
        if (d->astream[j] != stream_idx || !d->aglue_exp_step[j] || !d->aglue_exp_dl[j])
            continue;
        atomic_store_explicit(d->aglue_exp_step[j], step_us, memory_order_relaxed);
        atomic_store_explicit(d->aglue_exp_dl[j],
                              av_gettime_relative() + PTV_PAIR_EXPECT_TTL_US,
                              memory_order_release);
        av_log(NULL, AV_LOG_INFO,
               "[PTV-GLUE] paired flush: expected audio label step %+.3fs registered for a%d "
               "(stream %d) — content path will APPLY it\n",
               (double)step_us / AV_TIME_BASE, j, stream_idx);
    }
}

void ptv_disc_free(PtvDiscBuf *b)
{
    if (!b || !b->packets)
        return;
    ptv_disc_reset(b);
    av_freep(&b->packets);
    av_freep(&b->stream_transitioned);
    av_freep(&b->stream_state);
    b->capacity = b->nb_streams = 0;
}

/* Clone pkt into the buffer with its raw (post-unwrap) DTS in us. own_cont = the
 * packet was CONTINUOUS with its stream's own last_dts_us at arrival (pre7). */
static int ptv_disc_add(PtvDiscBuf *b, AVPacket *pkt, int stream_idx, int64_t raw_dts,
                        int own_cont)
{
    PtvDiscPacket *dp;
    if (b->nb_packets >= b->capacity)
        return AVERROR(ENOSPC);
    dp = av_mallocz(sizeof(*dp));
    if (!dp)
        return AVERROR(ENOMEM);
    dp->pkt = av_packet_clone(pkt);
    if (!dp->pkt) {
        av_freep(&dp);
        return AVERROR(ENOMEM);
    }
    dp->stream_idx = stream_idx;
    dp->raw_dts    = raw_dts;
    dp->timeline   = -1;
    dp->own_cont   = own_cont;
    b->packets[b->nb_packets++] = dp;
    return 0;
}

/* Classify raw_dts to OLD(0)/NEW(1) by nearest base within tolerance, preferring
 * this stream's own bases and BORROWING any stream's bases otherwise. -1 if no
 * bases recorded yet. Faithful to 0004's discont_classify_timeline. */
static int ptv_disc_classify(PtvDiscBuf *b, int stream_idx, int64_t raw_dts)
{
    int64_t old_base = AV_NOPTS_VALUE, new_base = AV_NOPTS_VALUE;
    int64_t dist_old, dist_new;
    int i;

    if (!b->jump_detected)
        return -1;
    if (stream_idx >= 0 && stream_idx < b->nb_streams) {
        PtvDiscStreamState *ss = &b->stream_state[stream_idx];
        if (ss->has_old_base && ss->has_new_base) {
            old_base = ss->old_timeline_base;
            new_base = ss->new_timeline_base;
        }
    }
    if (old_base == AV_NOPTS_VALUE || new_base == AV_NOPTS_VALUE) {
        for (i = 0; i < b->nb_streams; i++) {
            PtvDiscStreamState *ss = &b->stream_state[i];
            if (ss->has_old_base && ss->has_new_base) {
                old_base = ss->old_timeline_base;
                new_base = ss->new_timeline_base;
                break;
            }
        }
    }
    if (old_base == AV_NOPTS_VALUE || new_base == AV_NOPTS_VALUE)
        return -1;
    dist_old = llabs(raw_dts - old_base);
    dist_new = llabs(raw_dts - new_base);
    if (dist_old < dist_new && dist_old < PTV_DISC_TOL_US)
        return 0;
    else if (dist_new < PTV_DISC_TOL_US)
        return 1;
    else if (dist_old < dist_new)
        return 0;
    else
        return 1;
}

/* All dense (V/A) streams that this input transcodes/copies have transitioned.
 * ptvencoder has no per-stream discard/finished flags, so check the streams we
 * actually consume: the video stream and the transcoded audio streams. */
static int ptv_disc_all_transitioned(DemuxArgs *d, PtvDiscBuf *b)
{
    int k;
    if (d->vstream >= 0 && d->vstream < b->nb_streams && !b->stream_transitioned[d->vstream])
        return 0;
    for (k = 0; k < d->n_audio; k++) {
        int s = d->astream[k];
        if (s >= 0 && s < b->nb_streams && !b->stream_transitioned[s])
            return 0;
    }
    return 1;
}

static int ptv_disc_timeout(PtvDiscBuf *b)
{
    if (b->buffer_start_time == 0)
        return 0;
    return (av_gettime_relative() - b->buffer_start_time) > PTV_DISC_TIMEOUT_US;
}

static int ptv_disc_compare(const void *a, const void *bb)
{
    const PtvDiscPacket *pa = *(const PtvDiscPacket **)a;
    const PtvDiscPacket *pb = *(const PtvDiscPacket **)bb;
    if (pa->raw_dts < pb->raw_dts) return -1;
    if (pa->raw_dts > pb->raw_dts) return  1;
    return pa->stream_idx - pb->stream_idx;
}

/* Record this stream's old/new bases on a detected jump and arm jump_detected.
 * Detection runs against the per-stream wrap_last_us we keep in the disc buffer
 * (post-unwrap DTS in us), NOT the demux_unwrap wrap_last (stream-tb raw). */
/* PTV-QSNAP: one-line depth snapshot of EVERY queue in the pipeline — video_q (demux->decode
 * feed), frame_q cushion (via the published g_frameq_depth atomic), each transcoded audio_q,
 * each per-rung mux_q, and the LAYERA disc buffer. Logged at buffer-start and at flush so the
 * pair shows whether the rebase window starves the decode feed (video_q/frame_q drain during
 * the straddle -> house dup-fill). g_diag-gated, alongside the [PTV-LAYERA] logs. */
static void ptv_qsnap(DemuxArgs *d, PtvDiscBuf *b, const char *tag)
{
    char qs[320]; int p = 0, k;
    p += snprintf(qs + p, sizeof(qs) - p, "vq=%d frameq=%d aq=[",
                  av_thread_message_queue_nb_elems(d->video_q),
                  (int)atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
    for (k = 0; k < d->n_audio && p < (int)sizeof(qs); k++)
        p += snprintf(qs + p, sizeof(qs) - p, "%s%d", k ? "," : "",
                      av_thread_message_queue_nb_elems(d->audio_q[k]));
    p += snprintf(qs + p, sizeof(qs) - p, "] muxq=[");
    for (k = 0; k < d->n_out && p < (int)sizeof(qs); k++)
        p += snprintf(qs + p, sizeof(qs) - p, "%s%d", k ? "," : "",
                      av_thread_message_queue_nb_elems(d->mux_q[k]));
    snprintf(qs + p, sizeof(qs) - p, "] disc=%d", b->nb_packets);
    av_log(NULL, AV_LOG_INFO, "[PTV-QSNAP] %s: %s\n", tag, qs);
}

static int ptv_disc_detect_jump(DemuxArgs *d, PtvDiscBuf *b, int stream_idx,
                                int64_t raw_dts, int64_t last_dts)
{
    int64_t delta;
    if (b->flushing || last_dts == AV_NOPTS_VALUE)
        return 0;
    delta = raw_dts - last_dts;
    if (llabs(delta) <= PTV_DISC_THRESHOLD_US)
        return 0;
    if (stream_idx >= 0 && stream_idx < b->nb_streams) {
        PtvDiscStreamState *ss = &b->stream_state[stream_idx];
        ss->old_timeline_base = last_dts;
        ss->new_timeline_base = raw_dts;
        ss->has_old_base = 1;
        ss->has_new_base = 1;
    }
    b->jump_detected = 1;
    av_log(NULL, AV_LOG_INFO,   /* v0.9.13: always-on — a glue is rare (few/hour worst case) and operators need it in the log */
           "[PTV-LAYERA] jump on stream %d: %.3fs -> %.3fs (delta=%.3fs) — buffering\n",
           stream_idx, (double)last_dts / AV_TIME_BASE,
           (double)raw_dts / AV_TIME_BASE, (double)delta / AV_TIME_BASE);
    if (g_diag) ptv_qsnap(d, b, "buffer-start");
    return 1;
}

/* 1.0.1-pre7 continuing-stream keep: is `sidx` (own-continuous, no own bases this
 * cycle) ALREADY on the event's continuous/flushed output timeline, in a cycle
 * whose trigger we transcode? Then its buffered packets are KEPT AT OFFSET 0
 * (timeline 2) instead of being classified against the trigger's borrowed bases —
 * which is what deleted ~0.5s of continuing VIDEO per split (two-cycle) event
 * (flush-2 of fx-wcl400/fx-wc520: 25 video pkts 67474.79..67475.27 tagged OLD =
 * a visible picture skip; live: Azorse 2026-07-14 05:57 old=22/160). Deliberately
 * NARROW, two gates:
 *   - pair-state evidence only (video: the event's video crossing already applied
 *     its offset — pair_vid_defined; audio: this stream applied/holds the event's
 *     offset — pair_has/pair_prov). A stream with NO pair participation keeps the
 *     legacy classify-and-discard shape, so a first-cycle video-only flush still
 *     discards the not-yet-jumped audio tail byte-identically (fx-att-u900/b80,
 *     the pinned GAP-pad shape).
 *   - transcoded trigger only (vstream/astream). A cycle triggered by a
 *     copy/unconsumed dense stream (TruBLU rewinds: the AC-3 leg crossing ~0.6s
 *     after video+audio already glued — fx-tb30 flush-2) keeps today's
 *     production-proven behavior byte-identically (the TruBLU mandate),
 *     INCLUDING its continuing-stream discard — a known, deliberately retained
 *     cost this round. */
static int ptv_disc_cont_eligible(DemuxArgs *d, PtvDiscBuf *b, int sidx)
{
    int k, trig_transcoded;
    PtvDiscStreamState *ss;
    if (b->cycle_trigger < 0 || sidx >= (int)d->ifmt->nb_streams)
        return 0;
    trig_transcoded = b->cycle_trigger == d->vstream;
    for (k = 0; k < d->n_audio && !trig_transcoded; k++)
        trig_transcoded = d->astream[k] == b->cycle_trigger;
    if (!trig_transcoded)
        return 0;
    if (d->ifmt->streams[sidx]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        return b->pair_vid_defined;
    ss = &b->stream_state[sidx];
    return ss->pair_has || ss->pair_prov;
}

/* Flush: classify each held packet, KEEP NEW / DISCARD OLD, compute one
 * audio-derived applied_offset, apply it to all kept packets' pts+dts, and
 * release them in DTS order through demux_dispatch. Faithful to 0004's
 * discont_buffer_flush (always-keep-NEW, audio-offset preference). */
static int ptv_disc_flush(DemuxArgs *d, PtvDiscBuf *b)
{
    int i, ret = 0;
    int old_count = 0, new_count = 0, keep_timeline;
    int cont_count = 0;          /* 1.0.1-pre7: continuing-stream packets kept at offset 0 */
    int any_started = 0;
    int64_t vid_off = 0, aud_off = 0;
    int has_vid = 0, has_aud = 0;
    int pair_first_vid = 0;      /* 1.0.1-pre4: this flush is the event's first VIDEO crossing */
    int pair_inherit   = 0;      /* 1.0.1-pre4: audio-only flush adopted the event's video offset */

    if (b->nb_packets == 0) {
        ptv_disc_reset(b);
        return 0;
    }
    b->flushing = 1;

    for (i = 0; i < b->nb_packets; i++) {
        PtvDiscPacket *dp = b->packets[i];
        /* 1.0.1-pre7: an own-continuous packet of a stream that never crossed this cycle
         * (no own bases) must NEVER be classified against another stream's borrowed bases
         * — that is the split-event deletion defect AND (when its labels coincidentally
         * land within the tolerance of the trigger's NEW base) the false-crossing defect.
         * Eligible streams (already on the event's continuous timeline, transcoded-
         * triggered cycle) are KEPT on their own timeline at offset 0 (timeline 2); the
         * rest keep the legacy discard shape (timeline 0 — where legacy classification
         * put them anyway in every pinned fixture). Only under the shared-flush pairing
         * model (PTV_NO_SHARED_FLUSH reverts wholesale). */
        if (g_shared_flush && dp->timeline < 0 && dp->own_cont &&
            dp->stream_idx >= 0 && dp->stream_idx < b->nb_streams &&
            !b->stream_state[dp->stream_idx].has_new_base)
            dp->timeline = ptv_disc_cont_eligible(d, b, dp->stream_idx) ? 2 : 0;
        if (dp->timeline < 0)
            dp->timeline = ptv_disc_classify(b, dp->stream_idx, dp->raw_dts);
        if (dp->timeline == 0) old_count++;
        else if (dp->timeline == 1) new_count++;
        else if (dp->timeline == 2) cont_count++;
    }

    qsort(b->packets, b->nb_packets, sizeof(PtvDiscPacket *), ptv_disc_compare);

    for (i = 0; i < b->nb_streams; i++)
        if (b->stream_state[i].last_sent_dts != AV_NOPTS_VALUE) { any_started = 1; break; }

    /* Always keep NEW when both timelines have packets (the continued content);
     * if only one timeline buffered, keep that one. */
    if (old_count == 0 && new_count > 0)
        keep_timeline = 1;
    else if (new_count == 0 && old_count > 0)
        keep_timeline = 0;
    else
        keep_timeline = 1;

    /* Per-stream offset = where this stream left off minus where NEW resumes,
     * so the kept content butts against the previously-sent timeline. */
    if (keep_timeline == 1 && any_started) {
        for (i = 0; i < b->nb_streams; i++) {
            PtvDiscStreamState *ss = &b->stream_state[i];
            if (!ss->has_new_base)
                continue;
            if (ss->last_sent_dts != AV_NOPTS_VALUE)
                ss->cumulative_ts_offset = ss->last_sent_dts - ss->new_timeline_base;
            else if (ss->has_old_base)
                ss->cumulative_ts_offset = ss->old_timeline_base - ss->new_timeline_base;
        }
    }

    /* Gather each media type's own offset. ONE of them is applied to ALL streams (one offset
     * preserves relative A/V alignment): under the 1.0.1-pre4 shared flush VIDEO's offset
     * defines the timeline (decision tree below); the pre-pre4 path preferred audio (audio
     * seamless, video absorbs the residual via CFR dup/drop). */
    for (i = 0; i < b->nb_streams; i++) {
        AVStream *st;
        if (!b->stream_state[i].has_new_base)
            continue;
        st = d->ifmt->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vid_off = b->stream_state[i].cumulative_ts_offset; has_vid = 1;
        } else if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            aud_off = b->stream_state[i].cumulative_ts_offset; has_aud = 1;
        }
    }
    /* 1.0.1-pre4 SHARED FLUSH decision tree (g_shared_flush; see the invariant note at its
     * definition). Dense flushes within PTV_PAIR_WINDOW_US are ONE source event; every dense
     * stream in the event gets the event's VIDEO-derived offset, so the output's post-event A/V
     * alignment equals the source's (one offset never changes relative alignment). The A-vs-V
     * jump difference is NOT erased — it surfaces as an audio label step which the flush
     * REGISTERS for the track (ptv_pair_expect, pre5) so the CONTENT machinery APPLIES it in
     * every direction: aresample pads (forward) or drops (backward) to converge, and above the
     * AGLUE cap aresample=async hard pad/drop — bounded convergent, never another per-stream
     * erase. (pre4 relied on AGLUE's default rules here, and its backward relabel-ERASE
     * re-baked sub-1s backward mismatches — the D1 defect.) Tree, per flush:
     *   1. window expired (now − pair_start > PAIR_WIN)      → close it (independent events).
     *   2. VIDEO crossed in this flush:
     *      2a. video already defined an open event           → NEW event (close the old one).
     *      2b. audio also crossed and |vid_off − aud_off| > PTV_PAIR_EPS_US (a REAL A-vs-V jump
     *          difference, not bookkeeping — see the define) → applied = vid_off: video IS the
     *          timeline (the house clock anchors on video, prog_off/SCTE ride the video
     *          timeline, and a video label step would leak into audio via house_skew/AVLOCK —
     *          the In-Touch double-actuation class); the mismatch surfaces as an audio label
     *          step for the content machinery. Within the band (TruBLU symmetric rewinds, ad
     *          breaks: equal deltas ⇒ the offsets agree to bookkeeping noise) → applied =
     *          aud_off, BYTE-IDENTICAL to the per-stream path (the reduction mandate).
     *      2c. video-only flush → applied = vid_off (same as per-stream).
     *      2d. audio flushed EARLIER in this event with a provisional own offset (audio
     *          crossed first — the PATRIOT ordering) → retro-correct it below (after the
     *          persist block): shift its wrap_off/continuity refs by (event offset −
     *          provisional) when they disagree beyond the band, putting it on the
     *          video-defined timeline from now on; the shift itself is the audio label step
     *          the content machinery converges.
     *   3. AUDIO-only flush (video did not cross in this cycle):
     *      3a. event offset already defined by video (video crossed first — the Curiosity
     *          ordering, V and A 0.6s apart, past the 500ms buffer timeout) → INHERIT it when
     *          some crossing stream has NOT yet applied this event's offset (pair_has, the pre5
     *          D2 fix) — pre6: in EVERY band. A split flush that kept aud_off for a sub-band
     *          disagreement applied a second offset and BAKED the (vid−aud) residual as
     *          permanent relative desync (Azorse 104ms/event); the sub-band mismatch is
     *          registered for the content path instead (aglue floor — see the stamp block).
     *      3b. no video crossing yet → applied = aud_off (provisional pre-pre4 behavior; an
     *          unpaired audio-only jump keeps today's semantics, and a paired one is
     *          retro-corrected at video's flush, 2d).
     *      3c. (pre5, the D2 re-inherit fix) beyond the band but every crossing stream ALREADY
     *          applied this event's offset → a NEW INDEPENDENT audio event, NOT a late leg of
     *          the paired one: applied = aud_off (the plain butt-joint), and its pair state is
     *          left alone (fx-dbl: a -2s wobble 2s after a mirror event re-inherited -14.98s
     *          and destroyed ~17s of audio; each stream applies an event's offset AT MOST once).
     * Every audio stream crossing in a flush records what it applied (pair_applied_us) and how
     * (pair_has = final / pair_prov = provisional, awaiting 2d). Once video has defined the
     * event and every flowing dense audio stream has applied it, the window CLOSES (end of this
     * function) so later independent flushes cannot pair with a completed event.
     * PTV_NO_SHARED_FLUSH=1 restores the plain audio-preferred per-stream choice below. */
    if (g_shared_flush && keep_timeline == 1) {
        int64_t pnow = av_gettime_relative();
        int aud_unapplied = 0;   /* some audio stream crossing NOW has not yet applied this event's offset */
        int pair_stamp = 0;      /* what to persist for crossing audio streams: 0=nothing, 1=final, 2=provisional */
        if (b->pair_start_us && pnow - b->pair_start_us > PTV_PAIR_WINDOW_US)
            ptv_disc_pair_reset(b);                       /* 1: window expired */
        for (i = 0; i < b->nb_streams && i < (int)d->ifmt->nb_streams; i++)
            if (b->stream_state[i].has_new_base && !b->stream_state[i].pair_has &&
                d->ifmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
                aud_unapplied = 1;
        if (has_vid) {
            if (b->pair_vid_defined)
                ptv_disc_pair_reset(b);                   /* 2a: a second video crossing = a new event */
            if (!b->pair_start_us)
                b->pair_start_us = pnow;
            if (has_aud && llabs(vid_off - aud_off) <= PTV_PAIR_EPS_US)
                b->applied_offset = aud_off;              /* 2b band: bookkeeping-equal — per-stream-identical */
            else
                b->applied_offset = vid_off;              /* 2b/2c: video defines the timeline */
            b->pair_vid_defined = 1;
            b->pair_vid_off_us  = b->applied_offset;      /* the event's shared offset */
            pair_first_vid      = 1;                      /* 2d runs after the persist block */
            pair_stamp          = 1;
        } else if (has_aud) {
            if (!b->pair_start_us)
                b->pair_start_us = pnow;
            if (b->pair_vid_defined && aud_unapplied) {
                /* 3a: inherit the video-defined offset — 1.0.1-pre6: EVEN WITHIN the band. In a
                 * SAME-cycle full flush a sub-band disagreement is bookkeeping and either offset
                 * preserves alignment (ONE offset for all streams); but a SPLIT flush applies a
                 * SECOND offset in this cycle, so keeping aud_off baked the (vid−aud) residual
                 * (trailing-OLD discard hole + sub-band A-vs-V jump difference) into the output
                 * as permanent relative desync (live: Azorse 104ms/event 2026-07-14; fixtures
                 * fx-wc520 −440ms, fx-splitband −200ms). The sub-band mismatch is registered
                 * below (aglue floor) so the audio content path converges it instead. */
                b->applied_offset = b->pair_vid_off_us;
                pair_inherit      = 1;
                pair_stamp        = 1;
            } else if (b->pair_vid_defined &&
                       llabs(b->pair_vid_off_us - aud_off) > PTV_PAIR_EPS_US) {
                b->applied_offset = aud_off;          /* 3c: independent event — butt-joint, no pair stamp */
            } else {
                b->applied_offset = aud_off;              /* 3a band re-cross / 3b: per-stream-identical */
                pair_stamp = b->pair_vid_defined ? 1 : 2; /* band re-cross = final; no video yet = provisional */
            }
        }
        if (pair_stamp)
            for (i = 0; i < b->nb_streams && i < (int)d->ifmt->nb_streams; i++) {
                PtvDiscStreamState *ss = &b->stream_state[i];
                if (!ss->has_new_base ||
                    d->ifmt->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO)
                    continue;
                if (pair_stamp == 1) {
                    /* pre5 (D1): the shared offset overrode this stream's own butt-joint by more
                     * than the band → its label stream now carries that step; register it so the
                     * audio content path APPLIES it (aresample converges) instead of erasing.
                     * pre6: a split-flush inherit (pair_inherit) registers SUB-band mismatches
                     * too, down to the AGLUE engagement floor — below g_aglue_ms the glue never
                     * examines the step and aresample=async soft-converges it unregistered
                     * (registering there would only arm a stale expect that could mis-consume an
                     * unrelated later step). Same-cycle full flushes (2b) keep the >EPS gate —
                     * their sub-band residual stays on the byte-identical band path. */
                    int64_t mism = b->applied_offset - ss->cumulative_ts_offset;
                    if (has_aud && (llabs(mism) > PTV_PAIR_EPS_US ||
                                    (pair_inherit && llabs(mism) > (int64_t)g_aglue_ms * 1000)))
                        ptv_pair_expect(d, i, mism);
                    ss->pair_applied_us = b->applied_offset;
                    ss->pair_has        = 1;
                    ss->pair_prov       = 0;
                } else {
                    ss->pair_applied_us = b->applied_offset;   /* 3b: provisional, 2d may retro-correct */
                    ss->pair_prov       = 1;
                }
            }
    } else {
        if (has_aud)      b->applied_offset = aud_off;
        else if (has_vid) b->applied_offset = vid_off;
    }
    if (has_vid && has_aud) g_disc_viderr_sum += (vid_off - aud_off);   /* PTV-FLUSHAV: running total of source A/V misalignment absorbed at glues */

    av_log(NULL, AV_LOG_INFO,   /* v0.9.13: always-on (paired with the jump line above) */
           "[PTV-LAYERA] flush %d pkts: old=%d new=%d keep=%s applied_offset=%.3fs (vid=%.3fs aud=%.3fs vid_err=%.3fs cum_vid_err=%.3fs)\n",
           b->nb_packets, old_count, new_count, keep_timeline ? "NEW" : "OLD",
           (double)b->applied_offset / AV_TIME_BASE,
           (double)vid_off / AV_TIME_BASE, (double)aud_off / AV_TIME_BASE,
           (double)(vid_off - aud_off) / AV_TIME_BASE,
           (double)g_disc_viderr_sum / AV_TIME_BASE);
    if (cont_count > 0)   /* 1.0.1-pre7: separate line — the main flush line format is pinned */
        av_log(NULL, AV_LOG_INFO,
               "[PTV-LAYERA] flush kept %d continuing pkt(s) on their own timeline (offset 0) "
               "— streams already glued to this event, not classified against borrowed bases\n",
               cont_count);
    /* v0.9.13 [PTV-GLUE] — the LAYERA-vs-absorber decision line: running per-input stats of the
     * source A/V mis-mux at glues. The LAST such line in a log answers "does LAYERA earn its
     * keep on this channel" at a glance (mean/max ~0 => plain absorber suffices). */
    if (has_vid && has_aud) {
        int64_t e = vid_off - aud_off, ae = llabs(e);
        b->glue_cnt++;
        b->err_abs_sum_us += ae;
        if (ae > b->err_abs_max_us) b->err_abs_max_us = ae;
        if (ae > 100000) b->err_gt100_cnt++;
        av_log(NULL, AV_LOG_INFO,
               "[PTV-GLUE] #%"PRId64" vid_err=%+.3fs — run: mean|err|=%.3fs max=%.3fs >100ms=%"PRId64"/%"PRId64" partial=%"PRId64"\n",
               b->glue_cnt, (double)e / AV_TIME_BASE,
               (double)(b->err_abs_sum_us / b->glue_cnt) / AV_TIME_BASE,
               (double)b->err_abs_max_us / AV_TIME_BASE,
               b->err_gt100_cnt, b->glue_cnt, b->glue_partial);
    } else {
        b->glue_partial++;
        av_log(NULL, AV_LOG_INFO,
               "[PTV-GLUE] partial flush (only %s crossed in the window) — mis-mux not measurable this glue (partial=%"PRId64")\n",
               has_vid ? "video" : "audio", b->glue_partial);
    }
    /* 1.0.1-pre4 [PTV-GLUE] paired-flush ledger: the A-vs-V mismatch the shared offset AVOIDED
     * baking into the output (it routes to the audio content path instead — AGLUE/aresample).
     * Logged exactly when the shared offset actually OVERRODE the stream's own butt-joint
     * (disagreement above PTV_PAIR_EPS_US) — within the band behavior is per-stream-identical
     * and the log stays identical too (the TruBLU log-equality mandate). */
    if (g_shared_flush && keep_timeline == 1) {
        int64_t mm = 0, so = 0;
        int have = 0;
        if (has_vid && has_aud)      { so = b->applied_offset; mm = b->applied_offset - aud_off; have = 1; }
        else if (pair_inherit)       { so = b->applied_offset; mm = b->applied_offset - aud_off; have = 1; }
        if (have && llabs(mm) > PTV_PAIR_EPS_US)
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-GLUE] paired flush: shared_offset=%.3fs av_mismatch=%+.3fs -> audio content path\n",
                   (double)so / AV_TIME_BASE, (double)mm / AV_TIME_BASE);
    }
    if (g_diag) ptv_qsnap(d, b, "at-flush");

    for (i = 0; i < b->nb_packets; i++) {
        PtvDiscPacket *dp = b->packets[i];
        AVStream *st;
        int64_t pkt_off, eff_off;

        if (dp->timeline != keep_timeline && (dp->timeline == 0 || dp->timeline == 1)) {
            av_packet_free(&dp->pkt);
            av_freep(&b->packets[i]);
            continue;
        }
        if (dp->stream_idx < 0 || dp->stream_idx >= (int)d->ifmt->nb_streams) {
            av_packet_free(&dp->pkt);
            av_freep(&b->packets[i]);
            continue;
        }
        st = d->ifmt->streams[dp->stream_idx];

        /* Apply the single (audio-derived) offset to ALL kept packets in stream tb.
         * pre7: continuing-stream packets (timeline 2) are already on the output
         * timeline — they ride at offset 0, exactly as the normal path would have
         * dispatched them had the buffer not been active. */
        eff_off = dp->timeline == 2 ? 0 : b->applied_offset;
        if (eff_off != 0) {
            pkt_off = av_rescale_q(eff_off, AV_TIME_BASE_Q, st->time_base);
            if (dp->pkt->dts != AV_NOPTS_VALUE) dp->pkt->dts += pkt_off;
            if (dp->pkt->pts != AV_NOPTS_VALUE) dp->pkt->pts += pkt_off;
        }

        /* Advance last_sent_dts to the END of this kept packet (output domain) so
         * the next discontinuity cycle butts new content after it. */
        if (dp->stream_idx < b->nb_streams) {
            PtvDiscStreamState *ss = &b->stream_state[dp->stream_idx];
            int64_t out_end = dp->raw_dts + eff_off + ptv_disc_pkt_duration(st, dp->pkt);
            if (ss->last_sent_dts == AV_NOPTS_VALUE || out_end > ss->last_sent_dts)
                ss->last_sent_dts = out_end;
        }

        ret = demux_dispatch(d, dp->pkt);   /* re-inject through the normal send path; frees pkt */
        dp->pkt = NULL;
        av_freep(&b->packets[i]);
        if (ret < 0)
            break;
    }

    /* PERSIST the applied offset so SUBSEQUENT normal-path packets continue on the same
     * rebased timeline as the flushed NEW packets. demux_unwrap rides wrap_off (dense V/A)
     * and prog_off (sparse sub/data/SCTE); under g_layera it does `goto absorb_done` and
     * never updates them, so without this the next packets jump back to the raw new-timeline
     * ts (e.g. audio +1175s rebased, then the very next packet at the raw 10s) → a huge
     * BACKWARD jump → aresample chokes → audio dies (sync_check "no audio"). Faithful to
     * 0004, which applies its per-stream offset to every subsequent packet, not just buffered. */
    if (b->applied_offset != 0) {
        int s;
        for (s = 0; s < b->nb_streams && s < (int)d->ifmt->nb_streams; s++)
            if (b->stream_state[s].has_new_base) {
                d->wrap_off[s] += av_rescale_q(b->applied_offset, AV_TIME_BASE_Q,
                                               d->ifmt->streams[s]->time_base);
                /* 0.9.18.5: shift the stale jump-detection continuity ref too. last_dts_us is
                 * stored in the demux loop from the PRE-offset raw DTS of the last buffered
                 * packet; once wrap_off above carries the applied offset, the next normal-path
                 * packet arrives already-erased, so an unshifted ref re-triggered a phantom
                 * detect→flush cycle (applied_offset=-0.000s + a ~50ms hold) after every glue.
                 * See analysis/ptvencoder-intouch-desync-analysis.md §1.6 / §5 hygiene fix 1. */
                if (b->stream_state[s].last_dts_us != AV_NOPTS_VALUE)
                    b->stream_state[s].last_dts_us += b->applied_offset;
            }
        /* pre5 (D3): prog_off moves ONLY when this flush moved the VIDEO labels (has_vid, the
         * same gate as disc_resid_us below). Sparse sub/data/SCTE-35 ride the VIDEO/program
         * timeline; bumping prog_off on EVERY nonzero flush double-moved it on split-cycle
         * events (video flush + the audio-only inherit flush each added the shared offset →
         * SCTE-35/DVB-sub timing moved ~2x the offset). An audio-only UNPAIRED event moves no
         * video labels at all, so sparse timing must not move either — the pre-pre4 bump on
         * audio-only flushes was the same latent bug, merely never split-exposed. */
        if (has_vid)
            d->prog_off += av_rescale_q(b->applied_offset, AV_TIME_BASE_Q, (AVRational){1, 90000});
        /* 0.9.18.7 hs-residue ledger (logging only, no control consumer): this glue shifted the
         * video label stream by applied_offset, so the hs/sk reading moves by −applied_offset
         * relative to the raw source labels from here on. Counted only when VIDEO crossed
         * (has_vid) — an audio-only partial glue leaves the video timeline (and hs) untouched. */
        if (has_vid)
            d->disc_resid_us += -b->applied_offset;
    }

    /* 1.0.1-pre4 shared flush, tree 2d (pre5: PROVISIONAL streams only — pair_prov; a stream
     * that already applied the event's FINAL offset is never re-corrected, the D2 rule): this
     * flush is the pairing window's FIRST video crossing and some audio stream(s) already
     * flushed earlier in the window with a provisional audio-derived offset (audio's jump
     * crossed the demuxer first). Re-base them onto the video-defined timeline: shift wrap_off
     * by (vid_off − provisional) so every SUBSEQUENT packet rides the shared offset, and shift
     * the LAYERA continuity refs by the same amount so the step cannot re-trigger a phantom
     * detect→flush cycle (the 0.9.18.5 hygiene rule). The step this introduces in the audio
     * label stream IS the A-vs-V mismatch, delivered to the content machinery via the
     * registered expected step (ptv_pair_expect: AGLUE applies it; above its cap aresample
     * hard-converges) instead of being erased. prog_off is deliberately NOT touched here: it
     * moves only under the has_vid gate above — exactly once per event, at video's own flush. */
    if (pair_first_vid) {
        int s;
        for (s = 0; s < b->nb_streams && s < (int)d->ifmt->nb_streams; s++) {
            PtvDiscStreamState *ss = &b->stream_state[s];
            int64_t corr;
            if (!ss->pair_prov || ss->has_new_base)
                continue;                        /* only PROVISIONAL streams from EARLIER flushes of this event */
            corr = b->pair_vid_off_us - ss->pair_applied_us;
            ss->pair_applied_us = b->pair_vid_off_us;
            ss->pair_prov       = 0;
            ss->pair_has        = 1;             /* now on the event's final offset */
            if (llabs(corr) <= PTV_PAIR_EPS_US)
                continue;                        /* within the equality band — no step worth taking */
            d->wrap_off[s] += av_rescale_q(corr, AV_TIME_BASE_Q, d->ifmt->streams[s]->time_base);
            if (ss->last_dts_us   != AV_NOPTS_VALUE) ss->last_dts_us   += corr;
            if (ss->last_sent_dts != AV_NOPTS_VALUE) ss->last_sent_dts += corr;
            ptv_pair_expect(d, s, corr);         /* pre5 (D1): the content path must APPLY this step */
            av_log(NULL, AV_LOG_INFO,
                   "[PTV-GLUE] paired flush (retro): stream %d re-based onto the video timeline: "
                   "shared_offset=%.3fs av_mismatch=%+.3fs -> audio content path\n",
                   s, (double)b->pair_vid_off_us / AV_TIME_BASE, (double)corr / AV_TIME_BASE);
        }
    }

    /* pre5 (D2): CLOSE the pairing window once the event is COMPLETE — video has defined the
     * offset and every flowing dense audio stream has applied it (pair_has). From here any
     * further flush is a new, independent event; without this close, an independent audio
     * wobble inside the remaining window re-entered 3a and inherited a stale event offset
     * (belt for the 3a pair_has check — that check alone already stops the fx-dbl re-inherit,
     * this also stops pairing with streams that never cross, once the crossing set is done).
     * Streams that flow but never crossed keep the window open until it expires (5s) — the
     * genuine Curiosity ordering needs exactly that grace. */
    if (g_shared_flush && b->pair_vid_defined) {
        int open_aud = 0;
        for (i = 0; i < b->nb_streams && i < (int)d->ifmt->nb_streams; i++)
            if (d->ifmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
                b->stream_state[i].last_dts_us != AV_NOPTS_VALUE &&
                !b->stream_state[i].pair_has)
                open_aud = 1;
        if (!open_aud)
            ptv_disc_pair_reset(b);
    }

    b->flushing = 0;
    ptv_disc_reset(b);
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
        if (g_avlock && d->house_skew)   /* WUCR-corrected: AVLOCK kept (copied streams ride house_skew so they stay matched to the dup-lagged video); ρ bounds house_skew so the dup-hop stays small */
            h0_tb -= av_rescale_q(*d->house_skew, AV_TIME_BASE_Q, d->pass[pi].in_tb);
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
        if (d->ifmt->streams[d->pass[pi].in_index]->codecpar->codec_id == AV_CODEC_ID_SCTE_35) {
            int64_t adj_us = -av_rescale_q(h0_tb, d->pass[pi].in_tb, AV_TIME_BASE_Q);
            if (g_discont && g_prog_off)
                adj_us += av_rescale_q(d->prog_off, d->pass[pi].in_tb, AV_TIME_BASE_Q);
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
            if (d->pass[pi].gated && d->gate[i] && (c->dts != AV_NOPTS_VALUE || c->pts != AV_NOPTS_VALUE)) {
                /* §7.5a: dense copied AC-3/MP2 → hold for the video front. block=0: the shared demux
                 * thread must never stall the whole input (preserves copy's drop-on-full for net). */
                int64_t ts = c->dts != AV_NOPTS_VALUE ? c->dts : c->pts;
                dlv_enqueue(d->gate[i], c, av_rescale_q(ts, d->pass[pi].in_tb, AV_TIME_BASE_Q), 0);
            } else
                demux_send(d->mux_q[i], c, d->drop, &d->pdrop);   /* sparse subs/data/SCTE-35 bypass */
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
        int64_t wall_now = av_gettime_relative();
        int64_t last = d->wrap_last[pkt->stream_index];
        if (last != AV_NOPTS_VALUE) {
            int64_t delta = raw - last;
            int ct = st->codecpar->codec_type;
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
            else if (delta < -half) d->wrap_off[pkt->stream_index] += mask;  /* 33-bit wrap: rolled forward */
            else if (delta >  half) d->wrap_off[pkt->stream_index] -= mask;  /* late pre-roll pkt */
            else if (g_discont && (ct == AVMEDIA_TYPE_VIDEO || ct == AVMEDIA_TYPE_AUDIO)) {
                /* Source PTS discontinuity (a DTS jump, NOT a 33-bit wrap), either direction.
                 * Frames are continuous (one per tick) but the timestamp leaps; absorb the excess
                 * so the effective timeline stays continuous (re-base to last + one nominal frame),
                 * exactly like the wrap branch. Detected on DTS (monotonic), so B-frame PTS reorder
                 * never false-triggers. Keeps video/audio/copy aligned across the glitch.
                 *   BACKWARD jumps matter as much as forward (task#23, TruBlue): an ad-splice drops
                 *   the program DTS back hundreds of seconds (e.g. 523.9s -> 10s = -513.9s, not a
                 *   wrap since |Δ| < half). VIDEO survives unabsorbed (the compositor re-stamps output
                 *   to the house clock) but the source-content-anchored TRANSCODED AUDIO does not:
                 *   aresample=async needs a monotonic input, so a backward leap STALLS that slot's
                 *   audio drain (a0 went silent, then the mosaic). Re-basing here keeps the resampler,
                 *   the compositor h0/skew math, and any copy stream all on one continuous timeline.
                 *   The re-base formula (wrap_off -= delta-nominal) maps the new ts to last+nominal
                 *   for either sign (nominal is the small forward step).
                 * CONTINUOUS streams ONLY: sparse SUBTITLE/DATA (DVB-sub, SCTE-35) have natural
                 * multi-second inter-packet gaps that ALL exceed the threshold — absorbing them
                 * collapses the sparse timeline (subs drift out of sync / vanish; ad markers shift).
                 * The 33-bit wrap branches above still apply to every stream (copied AC-3/SCTE-35
                 * across the roll). */
                int64_t fwd_thresh  = av_rescale(g_discont_ms,      st->time_base.den, (int64_t)st->time_base.num * 1000);
                int64_t back_thresh = av_rescale(g_discont_back_ms, st->time_base.den, (int64_t)st->time_base.num * 1000);
                /* DIRECTIONAL (§5.A.1): forward jump must exceed the (large) forward threshold — small
                 * forward frame-drops flow through unabsorbed; backward jump must exceed the (small)
                 * backward threshold — backward jumps still absorb to protect aresample from a stall. */
                if ((fwd_thresh > 0 && delta > fwd_thresh) || (back_thresh > 0 && delta < -back_thresh)) {
                    int64_t thresh  = delta > 0 ? fwd_thresh : back_thresh;
                    int64_t nominal = pkt->duration > 0 ? pkt->duration : thresh / 4;
                    int64_t adj = delta - nominal;
                    int is_gap = 0;
                    /* gap-fix (2026-06-26): a FORWARD jump on a dense AUDIO stream is an audio-only SOURCE GAP
                     * (not a whole-program splice) when (a) the VIDEO stream did NOT also forward-cross recently
                     * (content signal — a real splice jumps video too) AND (b) this stream's packets were
                     * genuinely ABSENT for ~the jump in wall time. Absorbing it would delete the gap from the
                     * audio timeline → permanent A/V step (audio ahead of the house-clock-continuous video — the
                     * AWE bug). Instead do NOT absorb: aresample=async hard-pads silence (copied AC-3 keeps the
                     * real forward gap) → audio stays aligned with video. A whole-program splice (video crosses)
                     * or an audio relabel with packets still flowing (wall_gap≈0) is absorbed as before. See
                     * analysis/ptvencoder-avsync-gap-vs-splice-fix.md; PTV_NO_GAPDISCRIM reverts. */
                    if (g_gapdiscrim && delta > 0 && ct == AVMEDIA_TYPE_AUDIO) {
                        int64_t jump_us  = av_rescale_q(delta, st->time_base, AV_TIME_BASE_Q);
                        int64_t wl       = d->wrap_wall_last[pkt->stream_index];
                        int64_t wall_gap = wl ? wall_now - wl : 0;   /* 0 = no prior packet (sentinel) → treat as flowing */
                        int vcrossed = d->video_fwd_us && (wall_now - d->video_fwd_us <= g_progoff_debounce_us);
                        if (!vcrossed && wall_gap >= FFMAX(g_gap_min_us, jump_us / 2)) {
                            is_gap = 1;
                            if (d->disturb_epoch)   /* audio dropout is a disturbance (freeze rate-recovery / arm re-acquire) */
                                atomic_fetch_add_explicit(d->disturb_epoch, 1, memory_order_relaxed);
                            av_log(NULL, AV_LOG_INFO,   /* v0.9.13: always-on — a real source audio dropout, rare + meaningful */
                                   "[PTV-DISCONT] stream %d: %+"PRId64"ms audio GAP — NOT absorbed (aresample pads; wall_gap=%"PRId64"ms)\n",
                                   pkt->stream_index, av_rescale_q(delta, st->time_base, (AVRational){1,1000}), wall_gap / 1000);
                        }
                    }
                    if (is_gap) goto absorb_done;
                    int64_t nowb = av_gettime_relative();
                    /* 0.9.18.5 fold-in (log-truth only, no behavior coupling): record video forward
                     * crossings BEFORE the LAYERA skip so the audio gap discriminator's `vcrossed`
                     * signal works under g_layera too — it was only set inside the absorber body the
                     * skip bypassed, so a whole-program stall was always classified (and logged) as
                     * "audio GAP — NOT absorbed" even when LAYERA was about to erase it.
                     * gap-fix: video forward crossing = whole-program-splice signal for the audio
                     * gap discriminator. */
                    if (ct == AVMEDIA_TYPE_VIDEO && delta > 0)
                        d->video_fwd_us = wall_now;
                    if (g_layera &&
                        (g_layera_fullskip ||
                         llabs(av_rescale_q(delta, st->time_base, AV_TIME_BASE_Q)) > PTV_DISC_THRESHOLD_US)) {
                        /* Layer A = the legacy-0004 buffer-classify-discard mechanism, which lives in
                         * demux_thread (ptv_disc_*): it BUFFERS dense V/A across the straddle, discards
                         * OLD-timeline packets, computes ONE audio-derived offset, and applies it at flush.
                         * So demux_unwrap must NOT also absorb this crossing into wrap_off/prog_off — that
                         * would double-rebase the kept packets. Skip the per-stream absorber entirely (the
                         * 33-bit wrap branches above still ran, which the buffer relies on); leave DUKF and
                         * the disturb bump to the buffer/normal-path. (g_layera==0 path is unchanged.)
                         * 0.9.18.5: the skip is scoped to jumps LAYERA will actually claim
                         * (>PTV_DISC_THRESHOLD_US = 1s, same comparison as ptv_disc_detect_jump). Sub-1s
                         * steps fall through to the proven §5.A.2 shared-amount absorber below, so a
                         * both-stream backward step in the 80ms..1s "no-owner band" is erased identically
                         * on BOTH streams at the packet layer and house_skew/decimation/AGLUE/aresample
                         * never see it (the In-Touch audio-late accumulator — see
                         * analysis/ptvencoder-intouch-desync-analysis.md §4b/§5).
                         * PTV_LAYERA_FULLSKIP=1 restores the unconditional skip. */
                        goto absorb_done;
                    } else if (g_progoff_av) {
                        /* §5.A.2 (adopt-on-crossing, SHARED first-crosser amount — PRESERVES source A/V, see g_layera). */
                        if (nowb - d->splice_adj_us <= g_progoff_debounce_us)
                            adj = d->splice_adj;                                   /* same splice → adopt shared amount */
                        else { d->splice_adj = adj; d->splice_adj_us = nowb; }     /* first crosser sets the shared amount */
                    }
                    d->wrap_off[pkt->stream_index] -= adj;   /* per-stream rebase AT OWN CROSSING (audio-derived common offset when g_layera) */
                    if (ct == AVMEDIA_TYPE_VIDEO) {
                        d->prog_off -= adj;                  /* P2: sparse sub/data/SCTE ride this */
                        /* (video_fwd_us for the gap discriminator is stamped above, before the LAYERA skip — 0.9.18.5) */
                        /* P2 2b: arm drop-until-keyframe on VIDEO's own crossing (first-arm-only), ONLY on a LARGE jump. */
                        int64_t dukf_thresh = av_rescale(g_dukf_min_ms, st->time_base.den, (int64_t)st->time_base.num * 1000);
                        if (g_drop_until_kf && !d->drop_until_kf &&
                            (delta > dukf_thresh || delta < -dukf_thresh)) {
                            d->drop_until_kf = 1;
                            d->kf_arm_us = av_gettime_relative();
                            d->kf_arm_vdrop = d->vdrop;   /* DIAG: baseline for the per-event drop count */
                        }
                    }
                    if (d->disturb_epoch)   /* B3: a real content discontinuity → arm the PLL's mid-run re-acquire */
                        atomic_fetch_add_explicit(d->disturb_epoch, 1, memory_order_relaxed);
                    av_log(NULL, AV_LOG_INFO,   /* v0.9.13: always-on — one line per stream per crossing, events are rare */
                           "[PTV-DISCONT] stream %d: %+"PRId64"ms PTS jump absorbed (re-based to continuous)%s; frame_q=%d at jump\n",
                           pkt->stream_index, av_rescale_q(delta, st->time_base, (AVRational){1,1000}),
                           (ct == AVMEDIA_TYPE_VIDEO && d->drop_until_kf) ? " [DUKF armed → video drops until IDR]" : "",
                           atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
                    absorb_done: ;   /* gap-fix: a non-absorbed audio GAP jumps here — wrap_off left untouched, aresample pads */
                }
            }
        }
        d->wrap_last[pkt->stream_index] = raw;
        d->wrap_wall_last[pkt->stream_index] = wall_now;   /* gap-fix: per-stream packet arrival wall-clock */
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
}

void *demux_thread(void *arg)
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
        demux_unwrap(d, out);               /* 33-bit source wrap -> monotonic extended ts (ONCE) */

        /* legacy-0004 TS-discontinuity buffer (g_layera only). Dense V/A get
         * detect→buffer→classify→discard-OLD→single-offset; sparse SUBTITLE/DATA
         * (incl. SCTE-35) are NEVER buffered — they fall straight through to
         * demux_dispatch and keep the prog_off path. */
        if (g_layera && d->disc && out->dts != AV_NOPTS_VALUE) {
            AVStream *st = d->ifmt->streams[out->stream_index];
            int ct = st->codecpar->codec_type;
            if (ct == AVMEDIA_TYPE_VIDEO || ct == AVMEDIA_TYPE_AUDIO) {
                PtvDiscBuf *b = d->disc;
                int sidx = out->stream_index;
                int64_t raw_dts = av_rescale_q_rnd(out->dts, st->time_base, AV_TIME_BASE_Q,
                                                   AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
                int64_t last_dts = (sidx < b->nb_streams) ? b->stream_state[sidx].last_dts_us
                                                          : AV_NOPTS_VALUE;
                /* 1.0.1-pre7: continuity of this packet with its OWN stream's timeline, judged
                 * at arrival (before the last_dts_us update below). A stream whose packets stay
                 * own-continuous through a buffer cycle never genuinely crossed — the flush must
                 * not classify it against another stream's borrowed bases (see ptv_disc_flush). */
                int own_cont = last_dts != AV_NOPTS_VALUE &&
                               llabs(raw_dts - last_dts) <= PTV_DISC_THRESHOLD_US;
                /* Arm buffering on a fresh jump. */
                if (!b->active && ptv_disc_detect_jump(d, b, sidx, raw_dts, last_dts)) {
                    b->active = 1;
                    b->cycle_trigger = sidx;
                    b->buffer_start_time = av_gettime_relative();
                    if (d->disturb_epoch)   /* a real content discontinuity → arm the PLL re-acquire */
                        atomic_fetch_add_explicit(d->disturb_epoch, 1, memory_order_relaxed);
                } else if (b->active && sidx < b->nb_streams &&
                           !b->stream_state[sidx].has_new_base &&
                           ptv_disc_classify(b, sidx, raw_dts) != 1 &&
                           ptv_disc_detect_jump(d, b, sidx, raw_dts, last_dts)) {
                    /* 1.0.1-pre6 partner-crossing detect (the JLTV/Azorse escape): a >1s jump on a
                     * SECOND stream arriving while the buffer is active was left to classification,
                     * which borrows the FIRST stream's bases — on a mirror/asymmetric event the
                     * partner's post-jump position lands nearer the OLD base, so its stepped packets
                     * were tagged OLD and DELETED, the flush stayed video-only ("partial"), and the
                     * last_dts_us update below had already advanced this stream's continuity ref onto
                     * the stepped timeline — after the flush the step was INVISIBLE to detection
                     * forever and hit AGLUE/aresample as a raw multi-second input-pts step (measured
                     * -6.0s added desync on the wclose fixture). Detecting it here records the
                     * stream's OWN bases (detect_jump), so its stepped packets classify NEW against
                     * them, the stream transitions, and the flush treats the event as both-crossed
                     * (tree 2b: video defines the timeline, the A-vs-V difference is registered for
                     * the audio content path). Gated on the borrowed classification NOT already
                     * returning NEW, so orderings classification handles today (TruBLU symmetric,
                     * same-direction pairs) keep byte-identical behavior and log lines. */
                }
                if (sidx < b->nb_streams)
                    b->stream_state[sidx].last_dts_us = raw_dts;

                if (b->active) {
                    int timeline;
                    ret = ptv_disc_add(b, out, sidx, raw_dts, own_cont);
                    if (ret == AVERROR(ENOSPC)) {       /* buffer full → force flush, then re-add */
                        ret = ptv_disc_flush(d, b);
                        if (ret >= 0)
                            ret = ptv_disc_add(b, out, sidx, raw_dts, own_cont);
                    }
                    av_packet_free(&out);               /* buffer holds its own clone */
                    if (ret < 0) break;
                    /* Mark transitioned + record this stream's bases if it didn't
                     * trigger detection itself (faithful to 0004).
                     * 1.0.1-pre7 borrowed-base false-crossing gate: a stream whose packet is
                     * CONTINUOUS with its own last_dts_us never genuinely transitioned — a
                     * borrowed-base classify==1 here is a coincidence (labels within the 100ms
                     * tolerance of ANOTHER stream's new base) that recorded fake own bases with
                     * offset ≈ 0 and routed the full partner offset as an expected step
                     * (fx-att-rpt event 2, pre6 review residual 4). A genuine crosser's FIRST
                     * stepped packet has own_cont=0 (>1s own delta), so real transitions —
                     * detect-armed, rescue-detected, or borrowed-NEW — record exactly as before,
                     * and its later stepped packets find has_new_base already set. Its buffered
                     * own-continuous packets instead take the flush's own-continuity path
                     * (kept at offset 0 when eligible, legacy discard otherwise). */
                    timeline = ptv_disc_classify(b, sidx, raw_dts);
                    if (timeline == 1 && sidx < b->nb_streams && !(g_shared_flush && own_cont)) {
                        PtvDiscStreamState *ss = &b->stream_state[sidx];
                        b->stream_transitioned[sidx] = 1;
                        if (!ss->has_new_base) {
                            ss->new_timeline_base = raw_dts;
                            ss->has_new_base = 1;
                            if (!ss->has_old_base && last_dts != AV_NOPTS_VALUE) {
                                ss->old_timeline_base = last_dts;
                                ss->has_old_base = 1;
                            }
                        }
                    }
                    if (ptv_disc_all_transitioned(d, b) || ptv_disc_timeout(b)) {
                        ret = ptv_disc_flush(d, b);
                        if (ret < 0) break;
                    }
                    continue;                           /* don't dispatch now — held/released by flush */
                }
            }
        }

        /* Layer A: track last-sent DTS (output domain, +duration) per dense stream on the NORMAL
         * path too. The flush computes its rebase as (last_sent_dts − new_timeline_base); without
         * this the first glue saw last_sent_dts==NOPTS → offset 0 → the jump was NOT corrected.
         * Normal-path packets carry no disc offset, so their dispatched DTS IS the output DTS. */
        if (g_layera && d->disc && out->dts != AV_NOPTS_VALUE) {
            int sx = out->stream_index;
            if (sx >= 0 && sx < d->disc->nb_streams) {
                AVStream *st2 = d->ifmt->streams[sx];
                int ct2 = st2->codecpar->codec_type;
                if (ct2 == AVMEDIA_TYPE_VIDEO || ct2 == AVMEDIA_TYPE_AUDIO)
                    d->disc->stream_state[sx].last_sent_dts =
                        av_rescale_q(out->dts, st2->time_base, AV_TIME_BASE_Q)
                        + ptv_disc_pkt_duration(st2, out);
            }
        }

        ret = demux_dispatch(d, out);
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
                            av_log(NULL, AV_LOG_WARNING,
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

static int demux_dispatch(DemuxArgs *d, AVPacket *out)
{
    int ret = 0;
    if ((out->flags & AV_PKT_FLAG_CORRUPT) && g_discardcorrupt) {
        /* = -fflags +discardcorrupt, ALL streams — but COUNTED (video) so frame loss shows in stats
         * (libavformat's own flag discards silently, hiding the count). Drop before decode: a corrupt
         * frame, like a dropped one, becomes a content gap the position-anchored composite leaps
         * across → desync. PTV_KEEP_CORRUPT=1 disables (lets the decoder try to use them). */
        if (out->stream_index == d->vstream) d->vcorrupt++;
        av_packet_free(&out);
        return 0;
    }
        if (out->stream_index == d->vstream) {
            if (d->drop_until_kf) {   /* P2 2b: post-splice → drop video until the next IDR (bounded by the escape) */
                if (out->flags & AV_PKT_FLAG_KEY) {
                    d->drop_until_kf = 0;                                  /* IDR → clean resume; send it */
                    if (g_diag) av_log(NULL, AV_LOG_INFO,
                        "[PTV-DUKF] resume at keyframe — dropped %"PRId64" video frames over %"PRId64"ms; frame_q now %d (it drained while video was paused)\n",
                        d->vdrop - d->kf_arm_vdrop, (av_gettime_relative() - d->kf_arm_us)/1000,
                        atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
                } else if (av_gettime_relative() - d->kf_arm_us > g_dukf_escape_us) {
                    d->drop_until_kf = 0;                                  /* escape: no IDR in time → don't freeze */
                    if (g_diag) av_log(NULL, AV_LOG_WARNING,
                        "[PTV-DUKF] escape — no IDR within %"PRId64"ms; dropped %"PRId64" video frames; frame_q now %d\n",
                        g_dukf_escape_us/1000, d->vdrop - d->kf_arm_vdrop,
                        atomic_load_explicit(&g_frameq_depth, memory_order_relaxed));
                } else {
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
                        d->by_gap_cnt++;
                        if (gap > d->by_max_gap) d->by_max_gap = gap;
                        if (d->autobank && gap >= 3000000)      /* one BIG stall qualifies immediately (sparse-glitch class) */
                            ptv_autobank_escalate(d, gap, bw);
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
                        bw - d->by_bank_last_q > g_cp.bank_decay_us)
                        cushion_escalate(BANK_RETIRE, 0, 0);   /* 0.9.18 M3: write body moved */
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
             * — a video packet reached the demux dispatch just now (clean-wire evidence). */
            atomic_store_explicit(&g_v_arrive_wc, av_gettime_relative(), memory_order_relaxed);
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
                        atomic_store_explicit(&g_shed_wall, nw, memory_order_relaxed);
                        atomic_fetch_add_explicit(&g_shed_cnt, 1, memory_order_relaxed);
                        if (nw - d->qshed_log_us >= 5000000) {
                            d->qshed_log_us = nw;
                            av_log(NULL, AV_LOG_WARNING,
                                   "[PTV-QSHED] video_q full (%d pkts) — shedding whole GOPs: head flush "
                                   "requested from the decoder, arriving pkts dropped to the next IDR\n",
                                   av_thread_message_queue_nb_elems(d->video_q));
                        }
                        av_packet_free(&out);
                        ret = 0;
                    } else if (ret < 0)
                        av_packet_free(&out);       /* queue closed */
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
            int k;
            for (k = 0; k < d->n_audio; k++) {
                AVPacket *c;
                if (d->astream[k] != out->stream_index) continue;
                if (!(c = av_packet_clone(out))) continue;
                d->apkt++;
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

