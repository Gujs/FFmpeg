/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <float.h>
#include <stdint.h>

#include "ffmpeg.h"
#include "ffmpeg_sched.h"
#include "ffmpeg_utils.h"

#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/display.h"
#include "libavutil/error.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/parseutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/time.h"
#include "libavutil/timestamp.h"

#include "libavcodec/bsf.h"
#include "libavcodec/packet.h"

#include "libavformat/avformat.h"

/* Discontinuity buffer structures for handling interleaved packets
 * at discontinuity boundaries (e.g., ad splices without discontinuity_indicator) */

#define DISCONT_BUFFER_DEFAULT_SIZE   256
#define DISCONT_THRESHOLD_US          (1 * AV_TIME_BASE)   /* 1 second */
#define DISCONT_TIMEOUT_US            (500 * 1000)         /* 500ms */
#define DISCONT_TIMELINE_TOLERANCE_US (100 * 1000)         /* 100ms */

typedef struct DiscontinuityPacket {
    AVPacket *pkt;
    int stream_idx;
    int64_t raw_dts;         /* Original DTS before adjustment (AV_TIME_BASE units) */
    int timeline;            /* 0=old, 1=new, -1=unknown */
} DiscontinuityPacket;

typedef struct DiscontinuityBuffer {
    DiscontinuityPacket **packets;
    int nb_packets;
    int capacity;

    int64_t old_timeline_base;   /* DTS of last packet before discontinuity (AV_TIME_BASE units) */
    int64_t new_timeline_base;   /* DTS of first packet in new timeline (AV_TIME_BASE units) */
    int64_t timeline_delta;      /* Adjustment to apply: new_base - old_base */
    int timeline_established;    /* 1 if both old and new bases are known */

    int active;                  /* 1 if currently buffering packets */
    int flushing;                /* 1 if currently flushing (skip detection) */
    int64_t buffer_start_time;   /* Wall clock time when buffering started (us) */

    uint8_t *stream_transitioned; /* Per-stream flag: 1 if stream has transitioned to new timeline */
    int nb_streams;

    /* Cumulative offset tracking for multiple discontinuities */
    int64_t cumulative_ts_offset; /* Total offset applied to all packets (AV_TIME_BASE units) */
    int64_t last_sent_dts;        /* Last DTS sent downstream after all adjustments (AV_TIME_BASE) */
} DiscontinuityBuffer;

typedef struct DemuxStream {
    InputStream              ist;

    // name used for logging
    char                     log_name[32];

    int                      sch_idx_stream;
    int                      sch_idx_dec;

    double                   ts_scale;

    /* non zero if the packets must be decoded in 'raw_fifo', see DECODING_FOR_* */
    int                      decoding_needed;
#define DECODING_FOR_OST    1
#define DECODING_FOR_FILTER 2

    /* true if stream data should be discarded */
    int                      discard;

    // scheduler returned EOF for this stream
    int                      finished;

    int                      streamcopy_needed;
    int                      have_sub2video;
    int                      reinit_filters;
    int                      autorotate;
    int                      apply_cropping;
    int                      force_display_matrix;
    int                      drop_changed;


    int                      wrap_correction_done;
    int                      saw_first_ts;
    /// dts of the first packet read for this stream (in AV_TIME_BASE units)
    int64_t                  first_dts;

    /* predicted dts of the next packet read for this stream or (when there are
     * several frames in a packet) of the next frame in current packet (in AV_TIME_BASE units) */
    int64_t                  next_dts;
    /// dts of the last packet read for this stream (in AV_TIME_BASE units)
    int64_t                  dts;

    const AVCodecDescriptor *codec_desc;

    AVDictionary            *decoder_opts;
    DecoderOpts              dec_opts;
    char                     dec_name[16];
    // decoded media properties, as estimated by opening the decoder
    AVFrame                 *decoded_params;

    AVBSFContext            *bsf;

    /* number of packets successfully read for this stream */
    uint64_t                 nb_packets;
    // combined size of all the packets read
    uint64_t                 data_size;
    // latest wallclock time at which packet reading resumed after a stall - used for readrate
    int64_t                  resume_wc;
    // timestamp of first packet sent after the latest stall - used for readrate
    int64_t                  resume_pts;
    // measure of how far behind packet reading is against spceified readrate
    int64_t                  lag;

    /* Per-stream discontinuity handling */
    int64_t                  ts_offset_discont;      /* Per-stream timestamp offset */
    int                      discontinuity_pending;  /* Flag when discontinuity just detected */
    int64_t                  last_raw_dts;           /* Last DTS before any adjustment (AV_TIME_BASE) */

    /* Drop non-keyframe video packets after a discontinuity boundary.
     * The source sends corrupt packets at splice points that poison the
     * decoder's reference frames. All subsequent P/B frames decode as
     * garbage until the next IDR keyframe resets the decoder. By dropping
     * non-keyframe packets, the decoder produces no output during this
     * period and cfr mode duplicates the last good frame (clean freeze)
     * instead of outputting 10s of visual corruption. */
    int                      discont_drop_until_keyframe;

    /* Per-stream PTS wrap correction (AV_TIME_BASE units).
     * MPEG-TS uses a 33-bit counter that wraps every ~26.5 hours. Different
     * streams may wrap at slightly different times. This tracks the cumulative
     * wrap correction for each stream independently so that a video wrap
     * doesn't corrupt audio timestamps (and vice versa). */
    int64_t                  pts_wrap_correction;
} DemuxStream;

typedef struct DemuxStreamGroup {
    InputStreamGroup         istg;

    // name used for logging
    char                     log_name[32];
} DemuxStreamGroup;

typedef struct Demuxer {
    InputFile             f;

    // name used for logging
    char                  log_name[32];

    int64_t               wallclock_start;

    /**
     * Extra timestamp offset added by discontinuity handling.
     */
    int64_t               ts_offset_discont;
    int64_t               last_ts;

    int64_t               recording_time;
    int                   accurate_seek;

    /* number of times input stream should be looped */
    int                   loop;
    int                   have_audio_dec;
    /* duration of the looped segment of the input file */
    Timestamp             duration;
    /* pts with the smallest/largest values ever seen */
    Timestamp             min_pts;
    Timestamp             max_pts;

    /* number of streams that the user was warned of */
    int                   nb_streams_warn;

    float                 readrate;
    double                readrate_initial_burst;
    float                 readrate_catchup;

    Scheduler            *sch;

    AVPacket             *pkt_heartbeat;

    int                   read_started;
    int                   nb_streams_used;
    int                   nb_streams_finished;

    /* Discontinuity packet buffer for handling interleaved packets */
    DiscontinuityBuffer   discont_buf;
    int64_t               discont_threshold;     /* Jump threshold (default: 1 second) */
    int                   discont_buffer_size;   /* Max packets to buffer (default: 256) */
    int64_t               discont_timeout_us;    /* Timeout before forced flush (default: 500ms) */

    /* Diagnostic packet flow counters (per-second summary) */
    struct {
        int64_t interval_start;    /* Wall clock start of current 1s interval */
        int     pkts_read;         /* Packets read from av_read_frame */
        int     pkts_discarded;    /* Packets on wrong/finished streams */
        int     pkts_buffered;     /* Packets held in discont buffer */
        int     pkts_buf_flushed;  /* Packets released from buffer flush */
        int     pkts_dropped_kf;   /* Packets dropped by drop-until-keyframe */
        int     pkts_dropped_corrupt; /* Packets with AV_PKT_FLAG_CORRUPT */
        int     pkts_sent;         /* Packets sent to decoder (normal path) */
        int     vid_pkts_sent;     /* Video packets sent to decoder */
        int     aud_pkts_sent;     /* Audio packets sent to decoder */
        int     eagain_count;      /* EAGAIN returns from av_read_frame */
    } diag;
} Demuxer;

typedef struct DemuxThreadContext {
    // packet used for reading from the demuxer
    AVPacket *pkt_demux;
    // packet for reading from BSFs
    AVPacket *pkt_bsf;
} DemuxThreadContext;

static DemuxStream *ds_from_ist(InputStream *ist)
{
    return (DemuxStream*)ist;
}

static Demuxer *demuxer_from_ifile(InputFile *f)
{
    return (Demuxer*)f;
}

InputStream *ist_find_unused(enum AVMediaType type)
{
    for (InputStream *ist = ist_iter(NULL); ist; ist = ist_iter(ist)) {
        DemuxStream *ds = ds_from_ist(ist);
        if (ist->par->codec_type == type && ds->discard &&
            ist->user_set_discard != AVDISCARD_ALL)
            return ist;
    }
    return NULL;
}

static void report_new_stream(Demuxer *d, const AVPacket *pkt)
{
    const AVStream *st = d->f.ctx->streams[pkt->stream_index];

    if (pkt->stream_index < d->nb_streams_warn)
        return;
    av_log(d, AV_LOG_WARNING,
           "New %s stream with index %d at pos:%"PRId64" and DTS:%ss\n",
           av_get_media_type_string(st->codecpar->codec_type),
           pkt->stream_index, pkt->pos, av_ts2timestr(pkt->dts, &st->time_base));
    d->nb_streams_warn = pkt->stream_index + 1;
}

static int seek_to_start(Demuxer *d, Timestamp end_pts)
{
    InputFile    *ifile = &d->f;
    AVFormatContext *is = ifile->ctx;
    int ret;

    ret = avformat_seek_file(is, -1, INT64_MIN, is->start_time, is->start_time, 0);
    if (ret < 0)
        return ret;

    if (end_pts.ts != AV_NOPTS_VALUE &&
        (d->max_pts.ts == AV_NOPTS_VALUE ||
         av_compare_ts(d->max_pts.ts, d->max_pts.tb, end_pts.ts, end_pts.tb) < 0))
        d->max_pts = end_pts;

    if (d->max_pts.ts != AV_NOPTS_VALUE) {
        int64_t min_pts = d->min_pts.ts == AV_NOPTS_VALUE ? 0 : d->min_pts.ts;
        d->duration.ts = d->max_pts.ts - av_rescale_q(min_pts, d->min_pts.tb, d->max_pts.tb);
    }
    d->duration.tb = d->max_pts.tb;

    if (d->loop > 0)
        d->loop--;

    return ret;
}

/* ========== Discontinuity Buffer Functions ========== */

/**
 * Estimate packet duration in AV_TIME_BASE units from codec parameters.
 * Used for tracking last sent position (end of packet) for cumulative
 * offset calculation across discontinuities.
 */
static int64_t discont_estimate_pkt_duration(InputStream *ist, AVPacket *pkt)
{
    int64_t pkt_duration = 0;

    if (ist->par->codec_type == AVMEDIA_TYPE_AUDIO && ist->par->sample_rate) {
        pkt_duration = ((int64_t)AV_TIME_BASE * ist->par->frame_size) / ist->par->sample_rate;
    } else if (ist->par->codec_type == AVMEDIA_TYPE_VIDEO) {
        if (ist->framerate.num)
            pkt_duration = av_rescale_q(1, av_inv_q(ist->framerate), AV_TIME_BASE_Q);
        else if (ist->par->framerate.num)
            pkt_duration = av_rescale_q(1, av_inv_q(ist->par->framerate), AV_TIME_BASE_Q);
    }
    if (pkt_duration == 0 && pkt->duration > 0)
        pkt_duration = av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

    return pkt_duration;
}

/**
 * Initialize the discontinuity buffer for the given number of streams.
 * @param buf     The buffer to initialize
 * @param capacity Maximum number of packets to buffer
 * @param nb_streams Number of streams in the demuxer
 * @return 0 on success, negative AVERROR on failure
 */
static int discont_buffer_init(DiscontinuityBuffer *buf, int capacity, int nb_streams)
{
    memset(buf, 0, sizeof(*buf));

    buf->packets = av_calloc(capacity, sizeof(*buf->packets));
    if (!buf->packets)
        return AVERROR(ENOMEM);

    buf->stream_transitioned = av_calloc(nb_streams, sizeof(*buf->stream_transitioned));
    if (!buf->stream_transitioned) {
        av_freep(&buf->packets);
        return AVERROR(ENOMEM);
    }

    buf->capacity = capacity;
    buf->nb_streams = nb_streams;
    buf->nb_packets = 0;
    buf->active = 0;
    buf->timeline_established = 0;
    buf->old_timeline_base = AV_NOPTS_VALUE;
    buf->new_timeline_base = AV_NOPTS_VALUE;
    buf->timeline_delta = 0;
    buf->buffer_start_time = 0;
    buf->cumulative_ts_offset = 0;
    buf->last_sent_dts = AV_NOPTS_VALUE;

    return 0;
}

/**
 * Reset the discontinuity buffer state without freeing memory.
 * Called after a successful flush.
 */
static void discont_buffer_reset(DiscontinuityBuffer *buf)
{
    /* Free any remaining packets */
    for (int i = 0; i < buf->nb_packets; i++) {
        if (buf->packets[i]) {
            av_packet_free(&buf->packets[i]->pkt);
            av_freep(&buf->packets[i]);
        }
    }
    buf->nb_packets = 0;

    /* Reset per-stream transition flags */
    if (buf->stream_transitioned)
        memset(buf->stream_transitioned, 0, buf->nb_streams * sizeof(*buf->stream_transitioned));

    buf->active = 0;
    buf->timeline_established = 0;
    buf->old_timeline_base = AV_NOPTS_VALUE;
    buf->new_timeline_base = AV_NOPTS_VALUE;
    buf->timeline_delta = 0;
    buf->buffer_start_time = 0;
}

/**
 * Free all resources associated with the discontinuity buffer.
 */
static void discont_buffer_free(DiscontinuityBuffer *buf)
{
    if (!buf)
        return;

    discont_buffer_reset(buf);
    av_freep(&buf->packets);
    av_freep(&buf->stream_transitioned);
    buf->capacity = 0;
    buf->nb_streams = 0;
}

/**
 * Add a packet to the discontinuity buffer.
 * @param buf         The discontinuity buffer
 * @param pkt         The packet to add (will be cloned)
 * @param stream_idx  Stream index of the packet
 * @param raw_dts     Original DTS in AV_TIME_BASE units before any adjustment
 * @return 0 on success, negative AVERROR on failure
 */
static int discont_buffer_add(DiscontinuityBuffer *buf, AVPacket *pkt,
                              int stream_idx, int64_t raw_dts)
{
    DiscontinuityPacket *dp;

    if (buf->nb_packets >= buf->capacity)
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
    dp->raw_dts = raw_dts;
    dp->timeline = -1;  /* Unknown until classified */

    buf->packets[buf->nb_packets++] = dp;

    return 0;
}

/**
 * Classify a packet to OLD (0) or NEW (1) timeline.
 * @param buf      The discontinuity buffer
 * @param raw_dts  The packet's raw DTS in AV_TIME_BASE units
 * @return 0 for old timeline, 1 for new timeline, -1 if unknown
 */
static int discont_classify_timeline(DiscontinuityBuffer *buf, int64_t raw_dts)
{
    int64_t dist_to_old, dist_to_new;

    if (!buf->timeline_established)
        return -1;  /* Can't classify yet */

    dist_to_old = llabs(raw_dts - buf->old_timeline_base);
    dist_to_new = llabs(raw_dts - buf->new_timeline_base);

    /* Use tolerance for classification */
    if (dist_to_old < dist_to_new && dist_to_old < DISCONT_TIMELINE_TOLERANCE_US)
        return 0;  /* OLD timeline */
    else if (dist_to_new < DISCONT_TIMELINE_TOLERANCE_US)
        return 1;  /* NEW timeline */
    else if (dist_to_old < dist_to_new)
        return 0;  /* Closer to old */
    else
        return 1;  /* Closer to new */
}

/**
 * Check if all streams have transitioned to the new timeline.
 * @param buf        The discontinuity buffer
 * @param f          The input file (to check which streams are active)
 * @return 1 if all active streams have transitioned, 0 otherwise
 */
static int discont_all_streams_transitioned(DiscontinuityBuffer *buf, InputFile *f)
{
    for (int i = 0; i < f->nb_streams && i < buf->nb_streams; i++) {
        InputStream *ist = f->streams[i];
        DemuxStream *ds = ds_from_ist(ist);

        /* Skip discarded/finished streams */
        if (ds->discard || ds->finished)
            continue;

        /* Only check video and audio streams (not data/subtitle) */
        if (ist->par->codec_type != AVMEDIA_TYPE_VIDEO &&
            ist->par->codec_type != AVMEDIA_TYPE_AUDIO)
            continue;

        if (!buf->stream_transitioned[i])
            return 0;
    }
    return 1;
}

/**
 * Check if the discontinuity buffer has timed out.
 * @param buf  The discontinuity buffer
 * @return 1 if timeout exceeded, 0 otherwise
 */
static int discont_buffer_timeout(DiscontinuityBuffer *buf, int64_t timeout_us)
{
    int64_t now = av_gettime_relative();

    if (buf->buffer_start_time == 0)
        return 0;

    return (now - buf->buffer_start_time) > timeout_us;
}

/**
 * Compare function for sorting packets by adjusted DTS.
 */
static int discont_packet_compare(const void *a, const void *b)
{
    const DiscontinuityPacket *pa = *(const DiscontinuityPacket **)a;
    const DiscontinuityPacket *pb = *(const DiscontinuityPacket **)b;
    int64_t dts_a, dts_b;

    /* Use adjusted DTS for comparison */
    dts_a = pa->raw_dts;
    dts_b = pb->raw_dts;

    /* OLD timeline packets need adjustment */
    /* (Note: timeline_delta is applied during flush, not here) */

    if (dts_a < dts_b)
        return -1;
    else if (dts_a > dts_b)
        return 1;
    else
        return pa->stream_idx - pb->stream_idx;  /* Stable sort by stream */
}

/* Forward declarations */
static int demux_send(Demuxer *d, DemuxThreadContext *dt, DemuxStream *ds,
                      AVPacket *pkt, unsigned flags);
static int input_packet_process(Demuxer *d, AVPacket *pkt, unsigned *send_flags);

/**
 * Flush the discontinuity buffer, applying timestamp corrections and
 * reordering packets by adjusted DTS.
 * @param d   The demuxer
 * @param dt  The demux thread context
 * @return 0 on success, negative AVERROR on failure
 */
static int discont_buffer_flush(Demuxer *d, DemuxThreadContext *dt)
{
    DiscontinuityBuffer *buf = &d->discont_buf;
    InputFile *f = &d->f;
    int ret = 0;

    if (buf->nb_packets == 0) {
        discont_buffer_reset(buf);
        return 0;
    }

    /* Set flushing flag to prevent re-detection during processing */
    buf->flushing = 1;

    /* Classify all packets */
    {
        int old_count = 0, new_count = 0;
        for (int i = 0; i < buf->nb_packets; i++) {
            DiscontinuityPacket *dp = buf->packets[i];
            if (dp->timeline < 0)
                dp->timeline = discont_classify_timeline(buf, dp->raw_dts);
            if (dp->timeline == 0) old_count++;
            else if (dp->timeline == 1) new_count++;
        }
        av_log(d, AV_LOG_VERBOSE,
               "[DISCONT-BUF] Flushing %d buffered packets (old=%d, new=%d, delta=%.3fs)\n",
               buf->nb_packets, old_count, new_count,
               (double)buf->timeline_delta / AV_TIME_BASE);
    }

    /* Sort new-timeline packets by DTS for proper ordering */
    qsort(buf->packets, buf->nb_packets, sizeof(DiscontinuityPacket *),
          discont_packet_compare);

    /* Determine which timeline to keep:
     * 1. If only one timeline has packets, keep those
     * 2. If both timelines have packets, always keep NEW (the continued content)
     */
    {
        int sent_count = 0, discarded_count = 0;
        int old_count = 0, new_count = 0;
        int keep_timeline;

        /* Count packets in each timeline */
        for (int i = 0; i < buf->nb_packets; i++) {
            if (buf->packets[i]->timeline == 0) old_count++;
            else if (buf->packets[i]->timeline == 1) new_count++;
        }

        /* Decide which timeline to keep and calculate rebase offset.
         *
         * Key insight: We need to maintain continuous output timestamps across
         * multiple discontinuities. Use cumulative_ts_offset to track the total
         * adjustment needed.
         *
         * For NEW timeline packets:
         *   output_dts = raw_dts + cumulative_ts_offset
         *   We want output to continue from last_sent_dts
         *   So: cumulative_ts_offset = last_sent_dts - first_new_raw_dts + small_delta
         *
         * Simplified: add -timeline_delta to cumulative offset each time we keep NEW
         */
        int64_t rebase_offset = 0;
        int is_stream_start = (buf->last_sent_dts == AV_NOPTS_VALUE);

        if (old_count == 0 && new_count > 0) {
            keep_timeline = 1;  /* No old packets buffered, keep new */

            if (is_stream_start && buf->old_timeline_base < AV_TIME_BASE) {
                /* True stream start - no rebasing needed */
                av_log(d, AV_LOG_VERBOSE, "[DISCONT-BUF] Stream start, keeping all new (%d), no rebase\n", new_count);
            } else {
                /* Mid-stream jump - calculate offset to continue from last output position
                 *
                 * rebase_offset = last_sent_dts - new_timeline_base
                 * This makes new packets continue from where we left off
                 */
                if (buf->last_sent_dts != AV_NOPTS_VALUE) {
                    /* rebase_offset = offset needed to make new_base map to last_sent_dts
                     * This IS the total cumulative offset needed, not an increment */
                    rebase_offset = buf->last_sent_dts - buf->new_timeline_base;
                } else {
                    /* Fallback: use old base as reference */
                    rebase_offset = buf->old_timeline_base - buf->new_timeline_base;
                }
                buf->cumulative_ts_offset = rebase_offset;  /* SET, not increment */

                av_log(d, AV_LOG_VERBOSE, "[DISCONT-BUF] %s jump (delta=%.3fs), keeping new (%d), "
                       "cumulative=%.3fs (last_sent=%.3fs, new_base=%.3fs)\n",
                       buf->timeline_delta > 0 ? "Forward" : "Backward",
                       (double)buf->timeline_delta / AV_TIME_BASE, new_count,
                       (double)buf->cumulative_ts_offset / AV_TIME_BASE,
                       (double)buf->last_sent_dts / AV_TIME_BASE,
                       (double)buf->new_timeline_base / AV_TIME_BASE);
            }
        } else if (new_count == 0 && old_count > 0) {
            keep_timeline = 0;  /* No new packets, keep old */
            /* Old packets already have cumulative offset applied via last_sent_dts tracking */
            av_log(d, AV_LOG_VERBOSE, "[DISCONT-BUF] No new-timeline packets, keeping all old (%d)\n", old_count);
        } else {
            /* Both timelines have packets - always keep NEW.
             *
             * The NEW timeline represents where the source is going next,
             * regardless of whether the jump is forward or backward.
             * The OLD side typically has just 1 packet (the last before the
             * jump was detected), while NEW has the continued content.
             */
            keep_timeline = 1;

            /* Calculate offset to continue from old position */
            if (buf->last_sent_dts != AV_NOPTS_VALUE) {
                rebase_offset = buf->last_sent_dts - buf->new_timeline_base;
            } else {
                rebase_offset = buf->old_timeline_base - buf->new_timeline_base;
            }
            buf->cumulative_ts_offset = rebase_offset;  /* SET, not increment */

            av_log(d, AV_LOG_VERBOSE, "[DISCONT-BUF] Jump %s (delta=%.3fs), keeping new (old=%d, new=%d), "
                   "cumulative=%.3fs\n",
                   buf->timeline_delta > 0 ? "forward" : "backward",
                   (double)buf->timeline_delta / AV_TIME_BASE,
                   old_count, new_count,
                   (double)buf->cumulative_ts_offset / AV_TIME_BASE);
        }

        /* Set drop-until-keyframe for video streams when keeping NEW timeline.
         * This prevents corrupt splice boundary packets from poisoning the
         * decoder's reference frames, which would cause ~10s of garbage output
         * until the next IDR keyframe arrives. */
        if (keep_timeline == 1 && (old_count > 0 || !is_stream_start)) {
            for (int s = 0; s < f->nb_streams; s++) {
                InputStream *ist_s = f->streams[s];
                if (ist_s->par->codec_type == AVMEDIA_TYPE_VIDEO) {
                    DemuxStream *ds_vid = ds_from_ist(ist_s);
                    ds_vid->discont_drop_until_keyframe = 1;
                    av_log(d, AV_LOG_VERBOSE,
                           "[DISCONT-BUF] Dropping non-keyframe video packets on stream %d until keyframe\n", s);
                }
            }
        }

        for (int i = 0; i < buf->nb_packets; i++) {
            DiscontinuityPacket *dp = buf->packets[i];
            DemuxStream *ds;
            unsigned send_flags = 0;

            /* Discard packets from the timeline we don't want */
            if (dp->timeline != keep_timeline && dp->timeline >= 0) {
                av_log(d, AV_LOG_DEBUG, "[DISCONT-BUF] Discarding %s-timeline packet %d stream=%d dts=%"PRId64"\n",
                       dp->timeline == 0 ? "old" : "new", i + 1, dp->stream_idx, dp->raw_dts);
                discarded_count++;
                av_packet_free(&dp->pkt);
                av_freep(&buf->packets[i]);
                continue;
            }

            av_log(d, AV_LOG_DEBUG, "[DISCONT-BUF] Processing %s-timeline packet %d/%d stream=%d dts=%"PRId64"\n",
                   dp->timeline == 0 ? "old" : "new", i + 1, buf->nb_packets, dp->stream_idx, dp->raw_dts);

            /* Drop non-keyframe video packets after discontinuity to prevent
             * corrupt splice boundary data from reaching the decoder */
            if (dp->stream_idx < f->nb_streams) {
                InputStream *ist_chk = f->streams[dp->stream_idx];
                DemuxStream *ds_chk = ds_from_ist(ist_chk);
                if (ds_chk->discont_drop_until_keyframe &&
                    ist_chk->par->codec_type == AVMEDIA_TYPE_VIDEO) {
                    if (dp->pkt->flags & AV_PKT_FLAG_KEY) {
                        ds_chk->discont_drop_until_keyframe = 0;
                        av_log(d, AV_LOG_VERBOSE,
                               "[DISCONT-BUF] Keyframe found in buffer for stream %d, resuming video\n",
                               dp->stream_idx);
                    } else {
                        av_log(d, AV_LOG_DEBUG,
                               "[DISCONT-BUF] Dropping non-keyframe video packet from buffer: stream=%d dts=%"PRId64"\n",
                               dp->stream_idx, dp->raw_dts);
                        discarded_count++;
                        av_packet_free(&dp->pkt);
                        av_freep(&buf->packets[i]);
                        continue;
                    }
                }
            }

            if (dp->stream_idx >= f->nb_streams) {
                av_log(d, AV_LOG_WARNING, "[DISCONT-BUF] Invalid stream index %d\n",
                       dp->stream_idx);
                av_packet_free(&dp->pkt);
                av_freep(&buf->packets[i]);
                continue;
            }

            ds = ds_from_ist(f->streams[dp->stream_idx]);

            /* Apply timestamp rebasing using cumulative_ts_offset.
             * Use the stream's time_base since packets don't have their own time_base set */
            if (buf->cumulative_ts_offset != 0) {
                InputStream *ist_pkt = f->streams[dp->stream_idx];
                AVRational time_base = ist_pkt->st->time_base;
                int64_t pkt_offset = av_rescale_q(buf->cumulative_ts_offset, AV_TIME_BASE_Q, time_base);
                av_log(d, AV_LOG_DEBUG, "[DISCONT-BUF] Rebasing: cumulative=%"PRId64" time_base=%d/%d pkt_offset=%"PRId64" orig_dts=%"PRId64"\n",
                       buf->cumulative_ts_offset, time_base.num, time_base.den, pkt_offset, dp->pkt->dts);
                if (dp->pkt->dts != AV_NOPTS_VALUE)
                    dp->pkt->dts += pkt_offset;
                if (dp->pkt->pts != AV_NOPTS_VALUE)
                    dp->pkt->pts += pkt_offset;
                av_log(d, AV_LOG_DEBUG, "[DISCONT-BUF] Rebased packet: new_dts=%"PRId64"\n", dp->pkt->dts);
            }

            /* Mark flushed packets with discontinuity flag only if they were rebased.
             * Stream start packets (cumulative_ts_offset == 0) need normal
             * ts_discontinuity_process handling to establish proper offsets. */
            if (buf->cumulative_ts_offset != 0) {
                dp->pkt->flags |= AV_PKT_FLAG_DISCONTINUITY;
            }

            /* Process through normal timestamp fixup path */
            ret = input_packet_process(d, dp->pkt, &send_flags);
            if (ret < 0) {
                av_log(d, AV_LOG_ERROR, "[DISCONT-BUF] Failed to process packet: %s\n",
                       av_err2str(ret));
                av_packet_free(&dp->pkt);
                av_freep(&buf->packets[i]);
                break;
            }

            ret = demux_send(d, dt, ds, dp->pkt, send_flags);
            if (ret < 0) {
                av_log(d, AV_LOG_ERROR, "[DISCONT-BUF] Failed to send packet: %s\n",
                       av_err2str(ret));
                av_packet_free(&dp->pkt);
                av_freep(&buf->packets[i]);
                break;
            }

            /* Update ds->last_raw_dts so subsequent packets don't trigger new
             * discontinuity detection. Use the RAW (unrebasedT) DTS so the next
             * packet from the demuxer has a small delta, not a huge one. */
            if (dp->raw_dts != AV_NOPTS_VALUE) {
                ds->last_raw_dts = dp->raw_dts;
            }

            /* Track last sent position (END of packet, not start) for cumulative offset
             * calculation in future flushes. Using end time ensures new content starts
             * AFTER old content ends, not at the same time (which causes audio overlap).
             * Duration is estimated from codec parameters since pkt->duration may be 0. */
            if (dp->raw_dts != AV_NOPTS_VALUE) {
                InputStream *ist_pkt = f->streams[dp->stream_idx];
                int64_t pkt_duration = discont_estimate_pkt_duration(ist_pkt, dp->pkt);
                int64_t output_dts = dp->raw_dts + buf->cumulative_ts_offset;
                int64_t output_end = output_dts + pkt_duration;
                if (output_end > buf->last_sent_dts || buf->last_sent_dts == AV_NOPTS_VALUE) {
                    buf->last_sent_dts = output_end;
                    av_log(d, AV_LOG_DEBUG, "[DISCONT-BUF] Updated last_sent_dts to %.3fs (end of packet, start=%.3fs, dur=%.3fs)\n",
                           (double)buf->last_sent_dts / AV_TIME_BASE,
                           (double)output_dts / AV_TIME_BASE,
                           (double)pkt_duration / AV_TIME_BASE);
                }
            }

            sent_count++;
            d->diag.pkts_buf_flushed++;

            /* Clean up this packet entry */
            av_packet_free(&dp->pkt);
            av_freep(&buf->packets[i]);
        }

        av_log(d, AV_LOG_VERBOSE, "[DISCONT-BUF] Flush complete: %d sent, %d discarded\n",
               sent_count, discarded_count);
    }

    /* Clear flushing flag and reset buffer */
    buf->flushing = 0;
    discont_buffer_reset(buf);

    return ret;
}

/**
 * Log diagnostic packet flow counters once per second.
 * Only logs when something looks abnormal (no video sent, buffer active,
 * packets dropped, or excessive EAGAIN). Zero overhead during normal operation.
 */
static void discont_diag_log(Demuxer *d)
{
    int64_t now = av_gettime_relative();
    int64_t elapsed_us = now - d->diag.interval_start;

    if (elapsed_us < 1000000)  /* Not yet 1 second */
        return;

    /* Only log when something looks wrong:
     * - Zero video packets sent to decoder
     * - Buffer was active (packets held)
     * - Packets were dropped
     * - av_read_frame returned EAGAIN many times */
    if (d->diag.vid_pkts_sent == 0 ||
        d->diag.pkts_buffered > 0 ||
        d->diag.pkts_dropped_kf > 0 ||
        d->diag.eagain_count > 10) {

        av_log(d, AV_LOG_VERBOSE,
               "[DISCONT-DIAG] 1s: read=%d discard=%d buffered=%d "
               "flushed=%d drop_kf=%d corrupt=%d sent=%d "
               "(vid=%d aud=%d) eagain=%d\n",
               d->diag.pkts_read,
               d->diag.pkts_discarded,
               d->diag.pkts_buffered,
               d->diag.pkts_buf_flushed,
               d->diag.pkts_dropped_kf,
               d->diag.pkts_dropped_corrupt,
               d->diag.pkts_sent,
               d->diag.vid_pkts_sent,
               d->diag.aud_pkts_sent,
               d->diag.eagain_count);
    }

    /* Reset for next interval */
    memset(&d->diag, 0, sizeof(d->diag));
    d->diag.interval_start = now;
}

/**
 * Detect if a packet triggers a discontinuity and should start buffering.
 * @param d    The demuxer
 * @param ist  The input stream
 * @param pkt  The packet being processed
 * @param raw_dts  The packet's raw DTS in AV_TIME_BASE units
 * @return 1 if discontinuity detected and buffering should start, 0 otherwise
 */
static int discont_detect_jump(Demuxer *d, InputStream *ist, AVPacket *pkt,
                               int64_t raw_dts)
{
    DemuxStream *ds = ds_from_ist(ist);
    DiscontinuityBuffer *buf = &d->discont_buf;
    int64_t delta;

    /* Skip detection if we're currently flushing the buffer */
    if (buf->flushing)
        return 0;

    /* Only detect on video/audio streams */
    if (ist->par->codec_type != AVMEDIA_TYPE_VIDEO &&
        ist->par->codec_type != AVMEDIA_TYPE_AUDIO)
        return 0;

    /* Need previous DTS to compare */
    if (ds->last_raw_dts == AV_NOPTS_VALUE)
        return 0;

    /* Calculate delta */
    delta = raw_dts - ds->last_raw_dts;

    /* Check for PTS wrap (33-bit counter wraps every ~26.5 hours).
     * A wrap looks like a huge jump whose magnitude is close to the
     * wrap value (2^pts_wrap_bits in stream timebase, rescaled).
     * We detect this BEFORE treating it as a real discontinuity. */
    if (ist->st->pts_wrap_bits > 0 && ist->st->pts_wrap_bits < 64) {
        int64_t wrap_value = av_rescale_q(1LL << ist->st->pts_wrap_bits,
                                          ist->st->time_base, AV_TIME_BASE_Q);
        if (llabs(delta) > d->discont_threshold &&
            llabs(llabs(delta) - wrap_value) < 2 * AV_TIME_BASE) {
            /* This is a counter wrap, not a real discontinuity.
             * Record per-stream correction so the caller can adjust
             * this packet and all future packets on this stream. */
            ds->pts_wrap_correction -= delta;
            av_log(ist, AV_LOG_WARNING,
                   "[DISCONT-BUF] Timestamp jump %.3fs on stream %d is a PTS wrap "
                   "(wrap=%.3fs), per-stream correction now %.3fs\n",
                   (double)delta / AV_TIME_BASE, ist->index,
                   (double)wrap_value / AV_TIME_BASE,
                   (double)ds->pts_wrap_correction / AV_TIME_BASE);
            return 2;  /* wrap detected, caller applies correction */
        }
    }

    /* Check for significant jump (forward or backward) */
    if (llabs(delta) > d->discont_threshold) {
        av_log(ist, AV_LOG_WARNING,
               "[DISCONT-BUF] Detected timestamp jump on stream %d: "
               "delta=%.3fs (threshold=%.3fs)\n",
               ist->index, (double)delta / AV_TIME_BASE,
               (double)d->discont_threshold / AV_TIME_BASE);

        /* Set up timeline bases */
        if (!buf->timeline_established) {
            buf->old_timeline_base = ds->last_raw_dts;
            buf->new_timeline_base = raw_dts;
            buf->timeline_delta = raw_dts - ds->last_raw_dts;
            buf->timeline_established = 1;

            av_log(d, AV_LOG_VERBOSE,
                   "[DISCONT-BUF] Established timelines: old=%.3fs, new=%.3fs, delta=%.3fs\n",
                   (double)buf->old_timeline_base / AV_TIME_BASE,
                   (double)buf->new_timeline_base / AV_TIME_BASE,
                   (double)buf->timeline_delta / AV_TIME_BASE);
        }

        return 1;
    }

    return 0;
}

static void ts_discontinuity_detect(Demuxer *d, InputStream *ist,
                                    AVPacket *pkt)
{
    InputFile *ifile = &d->f;
    DemuxStream *ds = ds_from_ist(ist);
    const int fmt_is_discont = ifile->ctx->iformat->flags & AVFMT_TS_DISCONT;
    int disable_discontinuity_correction = copy_ts;
    int64_t pkt_dts = av_rescale_q_rnd(pkt->dts, pkt->time_base, AV_TIME_BASE_Q,
                                       AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);

    /* Skip standard discontinuity correction for packets that have already been
     * processed by the discontinuity buffer. The buffer has already rebased
     * timestamps to maintain continuity. */
    if (pkt->flags & AV_PKT_FLAG_DISCONTINUITY) {
        av_log(ist, AV_LOG_DEBUG,
               "[DISCONT-BUF] Skipping standard discontinuity correction (already handled)\n");
        /* Update tracking to avoid triggering on subsequent packets.
         * Set next_dts to AV_NOPTS_VALUE so it gets reinitialized from this packet.
         * Update last_ts so inter-stream detection doesn't trigger. */
        ds->next_dts = AV_NOPTS_VALUE;
        d->last_ts = pkt_dts;
        return;
    }

    if (copy_ts && ds->next_dts != AV_NOPTS_VALUE &&
        fmt_is_discont && ist->st->pts_wrap_bits < 60) {
        int64_t wrap_dts = av_rescale_q_rnd(pkt->dts + (1LL<<ist->st->pts_wrap_bits),
                                            pkt->time_base, AV_TIME_BASE_Q,
                                            AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
        if (FFABS(wrap_dts - ds->next_dts) < FFABS(pkt_dts - ds->next_dts)/10)
            disable_discontinuity_correction = 0;
    }

    if (ds->next_dts != AV_NOPTS_VALUE && !disable_discontinuity_correction) {
        int64_t delta = pkt_dts - ds->next_dts;
        if (fmt_is_discont) {
            if (FFABS(delta) > 1LL * dts_delta_threshold * AV_TIME_BASE ||
                pkt_dts + AV_TIME_BASE/10 < ds->dts) {
                d->ts_offset_discont -= delta;
                av_log(ist, AV_LOG_WARNING,
                       "timestamp discontinuity "
                       "(stream id=%d): %"PRId64", new offset= %"PRId64"\n",
                       ist->st->id, delta, d->ts_offset_discont);
                pkt->dts -= av_rescale_q(delta, AV_TIME_BASE_Q, pkt->time_base);
                if (pkt->pts != AV_NOPTS_VALUE)
                    pkt->pts -= av_rescale_q(delta, AV_TIME_BASE_Q, pkt->time_base);
            }
        } else {
            if (FFABS(delta) > 1LL * dts_error_threshold * AV_TIME_BASE) {
                av_log(ist, AV_LOG_WARNING,
                       "DTS %"PRId64", next:%"PRId64" st:%d invalid dropping\n",
                       pkt->dts, ds->next_dts, pkt->stream_index);
                pkt->dts = AV_NOPTS_VALUE;
            }
            if (pkt->pts != AV_NOPTS_VALUE){
                int64_t pkt_pts = av_rescale_q(pkt->pts, pkt->time_base, AV_TIME_BASE_Q);
                delta = pkt_pts - ds->next_dts;
                if (FFABS(delta) > 1LL * dts_error_threshold * AV_TIME_BASE) {
                    av_log(ist, AV_LOG_WARNING,
                           "PTS %"PRId64", next:%"PRId64" invalid dropping st:%d\n",
                           pkt->pts, ds->next_dts, pkt->stream_index);
                    pkt->pts = AV_NOPTS_VALUE;
                }
            }
        }
    } else if (ds->next_dts == AV_NOPTS_VALUE && !copy_ts &&
               fmt_is_discont && d->last_ts != AV_NOPTS_VALUE) {
        int64_t delta = pkt_dts - d->last_ts;
        if (FFABS(delta) > 1LL * dts_delta_threshold * AV_TIME_BASE) {
            d->ts_offset_discont -= delta;
            av_log(ist, AV_LOG_DEBUG,
                   "Inter stream timestamp discontinuity %"PRId64", new offset= %"PRId64"\n",
                   delta, d->ts_offset_discont);
            pkt->dts -= av_rescale_q(delta, AV_TIME_BASE_Q, pkt->time_base);
            if (pkt->pts != AV_NOPTS_VALUE)
                pkt->pts -= av_rescale_q(delta, AV_TIME_BASE_Q, pkt->time_base);
        }
    }

    d->last_ts = av_rescale_q(pkt->dts, pkt->time_base, AV_TIME_BASE_Q);
}

static void ts_discontinuity_process(Demuxer *d, InputStream *ist,
                                     AVPacket *pkt)
{
    int64_t offset = av_rescale_q(d->ts_offset_discont, AV_TIME_BASE_Q,
                                  pkt->time_base);

    /* For packets that have been rebased by the discontinuity buffer
     * (marked with AV_PKT_FLAG_DISCONTINUITY), we still apply the existing
     * d->ts_offset_discont (which includes start_time offset), but skip
     * the discontinuity detection that might add additional corrections. */
    if (pkt->flags & AV_PKT_FLAG_DISCONTINUITY) {
        av_log(ist, AV_LOG_DEBUG,
               "[DISCONT-BUF] Applying existing offset (%.3fs) to rebased packet, skipping detection\n",
               (double)d->ts_offset_discont / AV_TIME_BASE);

        // apply the existing offset
        if (pkt->dts != AV_NOPTS_VALUE)
            pkt->dts += offset;
        if (pkt->pts != AV_NOPTS_VALUE)
            pkt->pts += offset;

        /* Update d->last_ts so inter-stream detection doesn't trigger
         * on subsequent non-rebased packets */
        d->last_ts = av_rescale_q(pkt->dts, pkt->time_base, AV_TIME_BASE_Q);
        return;
    }

    // apply previously-detected timestamp-discontinuity offset
    // (to all streams, not just audio/video)
    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += offset;
    if (pkt->pts != AV_NOPTS_VALUE)
        pkt->pts += offset;

    // detect timestamp discontinuities for audio/video
    if ((ist->par->codec_type == AVMEDIA_TYPE_VIDEO ||
         ist->par->codec_type == AVMEDIA_TYPE_AUDIO) &&
        pkt->dts != AV_NOPTS_VALUE)
        ts_discontinuity_detect(d, ist, pkt);
}

static int ist_dts_update(DemuxStream *ds, AVPacket *pkt, FrameData *fd)
{
    InputStream *ist = &ds->ist;
    const AVCodecParameters *par = ist->par;

    if (!ds->saw_first_ts) {
        ds->first_dts =
        ds->dts = ist->st->avg_frame_rate.num ? - ist->par->video_delay * AV_TIME_BASE / av_q2d(ist->st->avg_frame_rate) : 0;
        if (pkt->pts != AV_NOPTS_VALUE) {
            ds->first_dts =
            ds->dts += av_rescale_q(pkt->pts, pkt->time_base, AV_TIME_BASE_Q);
        }
        ds->saw_first_ts = 1;
    }

    if (ds->next_dts == AV_NOPTS_VALUE)
        ds->next_dts = ds->dts;

    if (pkt->dts != AV_NOPTS_VALUE)
        ds->next_dts = ds->dts = av_rescale_q(pkt->dts, pkt->time_base, AV_TIME_BASE_Q);

    ds->dts = ds->next_dts;
    switch (par->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
        av_assert1(pkt->duration >= 0);
        if (par->sample_rate) {
            ds->next_dts += ((int64_t)AV_TIME_BASE * par->frame_size) /
                              par->sample_rate;
        } else {
            ds->next_dts += av_rescale_q(pkt->duration, pkt->time_base, AV_TIME_BASE_Q);
        }
        break;
    case AVMEDIA_TYPE_VIDEO:
        if (ist->framerate.num) {
            // TODO: Remove work-around for c99-to-c89 issue 7
            AVRational time_base_q = AV_TIME_BASE_Q;
            int64_t next_dts = av_rescale_q(ds->next_dts, time_base_q, av_inv_q(ist->framerate));
            ds->next_dts = av_rescale_q(next_dts + 1, av_inv_q(ist->framerate), time_base_q);
        } else if (pkt->duration) {
            ds->next_dts += av_rescale_q(pkt->duration, pkt->time_base, AV_TIME_BASE_Q);
        } else if (ist->par->framerate.num != 0) {
            AVRational field_rate = av_mul_q(ist->par->framerate,
                                             (AVRational){ 2, 1 });
            int fields = 2;

            if (ds->codec_desc                                 &&
                (ds->codec_desc->props & AV_CODEC_PROP_FIELDS) &&
                av_stream_get_parser(ist->st))
                fields = 1 + av_stream_get_parser(ist->st)->repeat_pict;

            ds->next_dts += av_rescale_q(fields, av_inv_q(field_rate), AV_TIME_BASE_Q);
        }
        break;
    }

    fd->dts_est = ds->dts;

    return 0;
}

static int ts_fixup(Demuxer *d, AVPacket *pkt, FrameData *fd)
{
    InputFile *ifile = &d->f;
    InputStream *ist = ifile->streams[pkt->stream_index];
    DemuxStream  *ds = ds_from_ist(ist);
    const int64_t start_time = ifile->start_time_effective;
    int64_t duration;
    int ret;

    pkt->time_base = ist->st->time_base;

#define SHOW_TS_DEBUG(tag_)                                             \
    if (debug_ts) {                                                     \
        av_log(ist, AV_LOG_INFO, "%s -> ist_index:%d:%d type:%s "       \
               "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s duration:%s duration_time:%s\n", \
               tag_, ifile->index, pkt->stream_index,                   \
               av_get_media_type_string(ist->st->codecpar->codec_type), \
               av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &pkt->time_base), \
               av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &pkt->time_base), \
               av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, &pkt->time_base)); \
    }

    SHOW_TS_DEBUG("demuxer");

    if (!ds->wrap_correction_done && start_time != AV_NOPTS_VALUE &&
        ist->st->pts_wrap_bits < 64) {
        int64_t stime, stime2;

        stime = av_rescale_q(start_time, AV_TIME_BASE_Q, pkt->time_base);
        stime2= stime + (1ULL<<ist->st->pts_wrap_bits);
        ds->wrap_correction_done = 1;

        if(stime2 > stime && pkt->dts != AV_NOPTS_VALUE && pkt->dts > stime + (1LL<<(ist->st->pts_wrap_bits-1))) {
            pkt->dts -= 1ULL<<ist->st->pts_wrap_bits;
            ds->wrap_correction_done = 0;
        }
        if(stime2 > stime && pkt->pts != AV_NOPTS_VALUE && pkt->pts > stime + (1LL<<(ist->st->pts_wrap_bits-1))) {
            pkt->pts -= 1ULL<<ist->st->pts_wrap_bits;
            ds->wrap_correction_done = 0;
        }
    }

    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, pkt->time_base);
    if (pkt->pts != AV_NOPTS_VALUE)
        pkt->pts += av_rescale_q(ifile->ts_offset, AV_TIME_BASE_Q, pkt->time_base);

    if (pkt->pts != AV_NOPTS_VALUE)
        pkt->pts *= ds->ts_scale;
    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts *= ds->ts_scale;

    duration = av_rescale_q(d->duration.ts, d->duration.tb, pkt->time_base);
    if (pkt->pts != AV_NOPTS_VALUE) {
        // audio decoders take precedence for estimating total file duration
        int64_t pkt_duration = d->have_audio_dec ? 0 : pkt->duration;

        pkt->pts += duration;

        // update max/min pts that will be used to compute total file duration
        // when using -stream_loop
        if (d->max_pts.ts == AV_NOPTS_VALUE ||
            av_compare_ts(d->max_pts.ts, d->max_pts.tb,
                          pkt->pts + pkt_duration, pkt->time_base) < 0) {
            d->max_pts = (Timestamp){ .ts = pkt->pts + pkt_duration,
                                      .tb = pkt->time_base };
        }
        if (d->min_pts.ts == AV_NOPTS_VALUE ||
            av_compare_ts(d->min_pts.ts, d->min_pts.tb,
                          pkt->pts, pkt->time_base) > 0) {
            d->min_pts = (Timestamp){ .ts = pkt->pts,
                                      .tb = pkt->time_base };
        }
    }

    if (pkt->dts != AV_NOPTS_VALUE)
        pkt->dts += duration;

    SHOW_TS_DEBUG("demuxer+tsfixup");

    // detect and try to correct for timestamp discontinuities
    ts_discontinuity_process(d, ist, pkt);

    // update estimated/predicted dts
    ret = ist_dts_update(ds, pkt, fd);
    if (ret < 0)
        return ret;

    return 0;
}

static int input_packet_process(Demuxer *d, AVPacket *pkt, unsigned *send_flags)
{
    InputFile     *f = &d->f;
    InputStream *ist = f->streams[pkt->stream_index];
    DemuxStream  *ds = ds_from_ist(ist);
    FrameData *fd;
    int ret = 0;

    fd = packet_data(pkt);
    if (!fd)
        return AVERROR(ENOMEM);

    ret = ts_fixup(d, pkt, fd);
    if (ret < 0)
        return ret;

    if (d->recording_time != INT64_MAX) {
        int64_t start_time = 0;
        if (copy_ts) {
            start_time += f->start_time != AV_NOPTS_VALUE ? f->start_time : 0;
            start_time += start_at_zero ? 0 : f->start_time_effective;
        }
        if (ds->dts >= d->recording_time + start_time)
            *send_flags |= DEMUX_SEND_STREAMCOPY_EOF;
    }

    ds->data_size += pkt->size;
    ds->nb_packets++;

    fd->wallclock[LATENCY_PROBE_DEMUX] = av_gettime_relative();

    if (debug_ts) {
        av_log(ist, AV_LOG_INFO, "demuxer+ffmpeg -> ist_index:%d:%d type:%s pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s duration:%s duration_time:%s off:%s off_time:%s\n",
               f->index, pkt->stream_index,
               av_get_media_type_string(ist->par->codec_type),
               av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &pkt->time_base),
               av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &pkt->time_base),
               av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, &pkt->time_base),
               av_ts2str(f->ts_offset),  av_ts2timestr(f->ts_offset, &AV_TIME_BASE_Q));
    }

    return 0;
}

static void readrate_sleep(Demuxer *d)
{
    InputFile *f = &d->f;
    int64_t file_start = copy_ts * (
                          (f->start_time_effective != AV_NOPTS_VALUE ? f->start_time_effective * !start_at_zero : 0) +
                          (f->start_time != AV_NOPTS_VALUE ? f->start_time : 0)
                         );
    int64_t initial_burst = AV_TIME_BASE * d->readrate_initial_burst;
    int resume_warn = 0;

    for (int i = 0; i < f->nb_streams; i++) {
        InputStream *ist = f->streams[i];
        DemuxStream  *ds = ds_from_ist(ist);
        int64_t stream_ts_offset, pts, now, wc_elapsed, elapsed, lag, max_pts, limit_pts;

        if (ds->discard) continue;

        stream_ts_offset = FFMAX(ds->first_dts != AV_NOPTS_VALUE ? ds->first_dts : 0, file_start);
        pts = av_rescale(ds->dts, 1000000, AV_TIME_BASE);
        now = av_gettime_relative();
        wc_elapsed = now - d->wallclock_start;

        if (pts <= stream_ts_offset + initial_burst) continue;

        max_pts = stream_ts_offset + initial_burst + (int64_t)(wc_elapsed * d->readrate);
        lag = FFMAX(max_pts - pts, 0);
        if ( (!ds->lag && lag > 0.3 * AV_TIME_BASE) || ( lag > ds->lag + 0.3 * AV_TIME_BASE) ) {
            ds->lag = lag;
            ds->resume_wc = now;
            ds->resume_pts = pts;
            av_log_once(ds, AV_LOG_WARNING, AV_LOG_DEBUG, &resume_warn,
                        "Resumed reading at pts %0.3f with rate %0.3f after a lag of %0.3fs\n",
                        (float)pts/AV_TIME_BASE, d->readrate_catchup, (float)lag/AV_TIME_BASE);
        }
        if (ds->lag && !lag)
            ds->lag = ds->resume_wc = ds->resume_pts = 0;
        if (ds->resume_wc) {
            elapsed = now - ds->resume_wc;
            limit_pts = ds->resume_pts + (int64_t)(elapsed * d->readrate_catchup);
        } else {
            elapsed = wc_elapsed;
            limit_pts = max_pts;
        }

        if (pts > limit_pts)
            av_usleep(pts - limit_pts);
    }
}

static int do_send(Demuxer *d, DemuxStream *ds, AVPacket *pkt, unsigned flags,
                   const char *pkt_desc)
{
    int ret;

    pkt->stream_index = ds->sch_idx_stream;

    ret = sch_demux_send(d->sch, d->f.index, pkt, flags);
    if (ret == AVERROR_EOF) {
        av_packet_unref(pkt);

        av_log(ds, AV_LOG_VERBOSE, "All consumers of this stream are done\n");
        ds->finished = 1;

        if (++d->nb_streams_finished == d->nb_streams_used) {
            av_log(d, AV_LOG_VERBOSE, "All consumers are done\n");
            return AVERROR_EOF;
        }
    } else if (ret < 0) {
        if (ret != AVERROR_EXIT)
            av_log(d, AV_LOG_ERROR,
                   "Unable to send %s packet to consumers: %s\n",
                   pkt_desc, av_err2str(ret));
        return ret;
    }

    return 0;
}

static int demux_send(Demuxer *d, DemuxThreadContext *dt, DemuxStream *ds,
                      AVPacket *pkt, unsigned flags)
{
    InputFile  *f = &d->f;
    int ret;

    // pkt can be NULL only when flushing BSFs
    av_assert0(ds->bsf || pkt);

    // send heartbeat for sub2video streams
    if (d->pkt_heartbeat && pkt && pkt->pts != AV_NOPTS_VALUE) {
        for (int i = 0; i < f->nb_streams; i++) {
            DemuxStream *ds1 = ds_from_ist(f->streams[i]);

            if (ds1->finished || !ds1->have_sub2video)
                continue;

            d->pkt_heartbeat->pts          = pkt->pts;
            d->pkt_heartbeat->time_base    = pkt->time_base;
            d->pkt_heartbeat->opaque       = (void*)(intptr_t)PKT_OPAQUE_SUB_HEARTBEAT;

            ret = do_send(d, ds1, d->pkt_heartbeat, 0, "heartbeat");
            if (ret < 0)
                return ret;
        }
    }

    if (ds->bsf) {
        if (pkt)
            av_packet_rescale_ts(pkt, pkt->time_base, ds->bsf->time_base_in);

        ret = av_bsf_send_packet(ds->bsf, pkt);
        if (ret < 0) {
            if (pkt)
                av_packet_unref(pkt);
            av_log(ds, AV_LOG_ERROR, "Error submitting a packet for filtering: %s\n",
                   av_err2str(ret));
            return ret;
        }

        while (1) {
            ret = av_bsf_receive_packet(ds->bsf, dt->pkt_bsf);
            if (ret == AVERROR(EAGAIN))
                return 0;
            else if (ret < 0) {
                if (ret != AVERROR_EOF)
                    av_log(ds, AV_LOG_ERROR,
                           "Error applying bitstream filters to a packet: %s\n",
                           av_err2str(ret));
                return ret;
            }

            dt->pkt_bsf->time_base = ds->bsf->time_base_out;

            ret = do_send(d, ds, dt->pkt_bsf, 0, "filtered");
            if (ret < 0) {
                av_packet_unref(dt->pkt_bsf);
                return ret;
            }
        }
    } else {
        ret = do_send(d, ds, pkt, flags, "demuxed");
        if (ret < 0)
            return ret;
    }

    return 0;
}

static int demux_bsf_flush(Demuxer *d, DemuxThreadContext *dt)
{
    InputFile *f = &d->f;
    int ret;

    for (unsigned i = 0; i < f->nb_streams; i++) {
        DemuxStream *ds = ds_from_ist(f->streams[i]);

        if (!ds->bsf)
            continue;

        ret = demux_send(d, dt, ds, NULL, 0);
        ret = (ret == AVERROR_EOF) ? 0 : (ret < 0) ? ret : AVERROR_BUG;
        if (ret < 0) {
            av_log(ds, AV_LOG_ERROR, "Error flushing BSFs: %s\n",
                   av_err2str(ret));
            return ret;
        }

        av_bsf_flush(ds->bsf);
    }

    return 0;
}

static void discard_unused_programs(InputFile *ifile)
{
    for (int j = 0; j < ifile->ctx->nb_programs; j++) {
        AVProgram *p = ifile->ctx->programs[j];
        int discard  = AVDISCARD_ALL;

        for (int k = 0; k < p->nb_stream_indexes; k++) {
            DemuxStream *ds = ds_from_ist(ifile->streams[p->stream_index[k]]);

            if (!ds->discard) {
                discard = AVDISCARD_DEFAULT;
                break;
            }
        }
        p->discard = discard;
    }
}

static void thread_set_name(InputFile *f)
{
    char name[16];
    snprintf(name, sizeof(name), "dmx%d:%s", f->index, f->ctx->iformat->name);
    ff_thread_setname(name);
}

static void demux_thread_uninit(DemuxThreadContext *dt)
{
    av_packet_free(&dt->pkt_demux);
    av_packet_free(&dt->pkt_bsf);

    memset(dt, 0, sizeof(*dt));
}

static int demux_thread_init(DemuxThreadContext *dt)
{
    memset(dt, 0, sizeof(*dt));

    dt->pkt_demux = av_packet_alloc();
    if (!dt->pkt_demux)
        return AVERROR(ENOMEM);

    dt->pkt_bsf = av_packet_alloc();
    if (!dt->pkt_bsf)
        return AVERROR(ENOMEM);

    return 0;
}

static int input_thread(void *arg)
{
    Demuxer   *d = arg;
    InputFile *f = &d->f;

    DemuxThreadContext dt;

    int ret = 0;

    ret = demux_thread_init(&dt);
    if (ret < 0)
        goto finish;

    thread_set_name(f);

    discard_unused_programs(f);

    d->read_started    = 1;
    d->wallclock_start = av_gettime_relative();
    d->diag.interval_start = av_gettime_relative();

    /* Lazy init: if discontinuity buffer wasn't initialized in ifile_open
     * (e.g., when using parallel input opening via ifile_open_from_io),
     * initialize it now before the demux loop starts. */
    if (d->discont_threshold == 0) {
        d->discont_threshold = DISCONT_THRESHOLD_US;
        d->discont_buffer_size = DISCONT_BUFFER_DEFAULT_SIZE;
        d->discont_timeout_us = DISCONT_TIMEOUT_US;
        ret = discont_buffer_init(&d->discont_buf, d->discont_buffer_size, f->nb_streams);
        if (ret < 0)
            goto finish;
    }

    while (1) {
        DemuxStream *ds;
        unsigned send_flags = 0;
        int64_t read_start, read_elapsed;

        /* Flush diagnostics before potentially blocking in av_read_frame */
        discont_diag_log(d);

        read_start = av_gettime_relative();
        ret = av_read_frame(f->ctx, dt.pkt_demux);
        read_elapsed = av_gettime_relative() - read_start;

        if (read_elapsed > 1000000) {
            av_log(d, AV_LOG_WARNING,
                   "[DISCONT-DIAG] av_read_frame blocked for %.3fs (ret=%s)\n",
                   (double)read_elapsed / AV_TIME_BASE,
                   av_err2str(ret));
        }

        if (ret == AVERROR(EAGAIN)) {
            d->diag.eagain_count++;
            av_usleep(10000);
            continue;
        }
        if (ret < 0) {
            int ret_bsf;

            if (ret == AVERROR_EOF)
                av_log(d, AV_LOG_VERBOSE, "EOF while reading input\n");
            else {
                av_log(d, AV_LOG_ERROR, "Error during demuxing: %s\n",
                       av_err2str(ret));
                ret = exit_on_error ? ret : 0;
            }

            ret_bsf = demux_bsf_flush(d, &dt);
            ret = err_merge(ret == AVERROR_EOF ? 0 : ret, ret_bsf);

            if (d->loop) {
                /* signal looping to our consumers */
                dt.pkt_demux->stream_index = -1;
                ret = sch_demux_send(d->sch, f->index, dt.pkt_demux, 0);
                if (ret >= 0)
                    ret = seek_to_start(d, (Timestamp){ .ts = dt.pkt_demux->pts,
                                                        .tb = dt.pkt_demux->time_base });
                if (ret >= 0)
                    continue;

                /* fallthrough to the error path */
            }

            break;
        }

        d->diag.pkts_read++;

        if (do_pkt_dump) {
            av_pkt_dump_log2(NULL, AV_LOG_INFO, dt.pkt_demux, do_hex_dump,
                             f->ctx->streams[dt.pkt_demux->stream_index]);
        }

        /* the following test is needed in case new streams appear
           dynamically in stream : we ignore them */
        ds = dt.pkt_demux->stream_index < f->nb_streams ?
             ds_from_ist(f->streams[dt.pkt_demux->stream_index]) : NULL;
        if (!ds || ds->discard || ds->finished) {
            d->diag.pkts_discarded++;
            report_new_stream(d, dt.pkt_demux);
            av_packet_unref(dt.pkt_demux);
            continue;
        }

        if (dt.pkt_demux->flags & AV_PKT_FLAG_CORRUPT) {
            d->diag.pkts_dropped_corrupt++;
            av_log(d, exit_on_error ? AV_LOG_FATAL : AV_LOG_WARNING,
                   "corrupt input packet in stream %d\n",
                   dt.pkt_demux->stream_index);
            if (exit_on_error) {
                av_packet_unref(dt.pkt_demux);
                ret = AVERROR_INVALIDDATA;
                break;
            }
        }

        /* Capture raw DTS before ts_fixup for discontinuity detection */
        {
            InputStream *ist = f->streams[dt.pkt_demux->stream_index];
            DemuxStream *ds_wrap = ds_from_ist(ist);
            int64_t raw_dts = AV_NOPTS_VALUE;

            /* Apply per-stream PTS wrap correction BEFORE computing raw_dts.
             * This corrects all future packets after a wrap has been detected
             * on this stream. */
            if (ds_wrap->pts_wrap_correction != 0 &&
                dt.pkt_demux->dts != AV_NOPTS_VALUE) {
                int64_t corr_tb = av_rescale_q(ds_wrap->pts_wrap_correction,
                                               AV_TIME_BASE_Q, ist->st->time_base);
                dt.pkt_demux->dts += corr_tb;
                if (dt.pkt_demux->pts != AV_NOPTS_VALUE)
                    dt.pkt_demux->pts += corr_tb;
            }

            if (dt.pkt_demux->dts != AV_NOPTS_VALUE) {
                raw_dts = av_rescale_q_rnd(dt.pkt_demux->dts,
                                           ist->st->time_base, AV_TIME_BASE_Q,
                                           AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
            }

            /* Check if this packet triggers discontinuity buffering */
            if (!d->discont_buf.active && raw_dts != AV_NOPTS_VALUE) {
                int64_t old_wrap_corr = ds_wrap->pts_wrap_correction;
                int jump_ret = discont_detect_jump(d, ist, dt.pkt_demux, raw_dts);
                if (jump_ret == 2) {
                    /* PTS wrap detected - apply only the NEWLY ADDED correction
                     * to this packet. The previous correction was already applied
                     * at line 1614 above; applying the full accumulated value
                     * would double-count the old portion. */
                    int64_t new_delta = ds_wrap->pts_wrap_correction - old_wrap_corr;
                    int64_t corr_tb = av_rescale_q(new_delta,
                                                   AV_TIME_BASE_Q, ist->st->time_base);
                    dt.pkt_demux->dts += corr_tb;
                    if (dt.pkt_demux->pts != AV_NOPTS_VALUE)
                        dt.pkt_demux->pts += corr_tb;
                    raw_dts = av_rescale_q_rnd(dt.pkt_demux->dts,
                                               ist->st->time_base, AV_TIME_BASE_Q,
                                               AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
                } else if (jump_ret == 1) {
                    d->discont_buf.active = 1;
                    d->discont_buf.buffer_start_time = av_gettime_relative();
                    av_log(d, AV_LOG_VERBOSE,
                           "[DISCONT-BUF] Starting packet buffering (%d stream capacity)\n",
                           d->discont_buf.nb_streams);
                }
            }

            /* If buffering is active, add packet to buffer */
            if (d->discont_buf.active) {
                int timeline;

                d->diag.pkts_buffered++;
                ret = discont_buffer_add(&d->discont_buf, dt.pkt_demux,
                                         dt.pkt_demux->stream_index, raw_dts);
                if (ret == AVERROR(ENOSPC)) {
                    /* Buffer full, force flush */
                    av_log(d, AV_LOG_WARNING,
                           "[DISCONT-BUF] Buffer full (%d packets), forcing flush\n",
                           d->discont_buf.nb_packets);
                    ret = discont_buffer_flush(d, &dt);
                    if (ret < 0)
                        break;

                    /* Re-add this packet now that buffer is cleared */
                    ret = discont_buffer_add(&d->discont_buf, dt.pkt_demux,
                                             dt.pkt_demux->stream_index, raw_dts);
                }
                if (ret < 0) {
                    av_log(d, AV_LOG_ERROR,
                           "[DISCONT-BUF] Failed to add packet to buffer: %s\n",
                           av_err2str(ret));
                    break;
                }

                /* Mark this stream as transitioned if packet is on new timeline */
                timeline = discont_classify_timeline(&d->discont_buf, raw_dts);
                if (timeline == 1 && dt.pkt_demux->stream_index < d->discont_buf.nb_streams) {
                    d->discont_buf.stream_transitioned[dt.pkt_demux->stream_index] = 1;
                }

                /* Check if we should flush */
                if (discont_all_streams_transitioned(&d->discont_buf, f) ||
                    discont_buffer_timeout(&d->discont_buf, d->discont_timeout_us)) {

                    if (discont_buffer_timeout(&d->discont_buf, d->discont_timeout_us)) {
                        av_log(d, AV_LOG_WARNING,
                               "[DISCONT-BUF] Timeout reached, flushing buffer\n");
                    } else {
                        av_log(d, AV_LOG_VERBOSE,
                               "[DISCONT-BUF] All streams transitioned, flushing buffer\n");
                    }

                    ret = discont_buffer_flush(d, &dt);
                    if (ret < 0)
                        break;
                }

                /* Update last_raw_dts for this stream */
                ds->last_raw_dts = raw_dts;

                av_packet_unref(dt.pkt_demux);
                continue;  /* Don't process this packet through normal path */
            }

            /* Detect "poisoned" packets at splice boundaries where DTS
             * continues monotonically from the old timeline but PTS has
             * already jumped to the new timeline. discont_detect_jump only
             * checks DTS, so these slip through with a catastrophically
             * wrong PTS that causes massive frame drops in the CFR filter.
             * Clamp PTS to DTS when the gap exceeds a reasonable B-frame
             * reordering delay. */
            if (raw_dts != AV_NOPTS_VALUE &&
                dt.pkt_demux->pts != AV_NOPTS_VALUE) {
                int64_t raw_pts = av_rescale_q_rnd(dt.pkt_demux->pts,
                                                    ist->st->time_base,
                                                    AV_TIME_BASE_Q,
                                                    AV_ROUND_NEAR_INF |
                                                    AV_ROUND_PASS_MINMAX);
                int64_t pts_dts_diff = llabs(raw_pts - raw_dts);
                if (pts_dts_diff > d->discont_threshold) {
                    av_log(ist, AV_LOG_WARNING,
                           "[DISCONT-BUF] Poisoned packet detected on stream %d: "
                           "PTS=%.3fs DTS=%.3fs diff=%.3fs - clamping PTS to DTS\n",
                           ist->index,
                           (double)raw_pts / AV_TIME_BASE,
                           (double)raw_dts / AV_TIME_BASE,
                           (double)pts_dts_diff / AV_TIME_BASE);
                    dt.pkt_demux->pts = dt.pkt_demux->dts;
                }
            }

            /* Update last_raw_dts for discontinuity detection */
            ds->last_raw_dts = raw_dts;
        }

        /* Apply cumulative timestamp offset from discontinuity handling.
         * This offset is calculated by the discontinuity buffer to maintain
         * continuous timestamps across source discontinuities. All packets
         * (not just flushed ones) need this adjustment after a discontinuity
         * has been detected and handled. */
        if (d->discont_buf.cumulative_ts_offset != 0 && dt.pkt_demux->dts != AV_NOPTS_VALUE) {
            InputStream *ist_pkt = f->streams[dt.pkt_demux->stream_index];
            AVRational time_base = ist_pkt->st->time_base;
            int64_t pkt_offset = av_rescale_q(d->discont_buf.cumulative_ts_offset,
                                              AV_TIME_BASE_Q, time_base);

            av_log(ist_pkt, AV_LOG_DEBUG,
                   "[DISCONT-BUF] Applying cumulative offset to normal packet: "
                   "stream=%d raw_dts=%"PRId64" offset=%"PRId64" new_dts=%"PRId64"\n",
                   dt.pkt_demux->stream_index, dt.pkt_demux->dts,
                   pkt_offset, dt.pkt_demux->dts + pkt_offset);

            dt.pkt_demux->dts += pkt_offset;
            if (dt.pkt_demux->pts != AV_NOPTS_VALUE)
                dt.pkt_demux->pts += pkt_offset;

            /* Mark packet as having discontinuity correction applied so
             * ts_discontinuity_process doesn't double-adjust it */
            dt.pkt_demux->flags |= AV_PKT_FLAG_DISCONTINUITY;
        }

        /* Drop non-keyframe video packets after discontinuity boundary.
         * The source's new content stream is joined mid-GOP, so non-IDR
         * packets reference frames our decoder has never seen. They would
         * either decode as garbage or fail entirely for ~10s until the next
         * IDR arrives. Dropping them lets cfr duplicate the last good frame. */
        if (ds->discont_drop_until_keyframe &&
            ds->ist.par->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (dt.pkt_demux->flags & AV_PKT_FLAG_KEY) {
                ds->discont_drop_until_keyframe = 0;
                av_log(&ds->ist, AV_LOG_VERBOSE,
                       "[DISCONT-BUF] Keyframe arrived on stream %d, resuming video decode\n",
                       dt.pkt_demux->stream_index);
            } else {
                d->diag.pkts_dropped_kf++;
                av_log(&ds->ist, AV_LOG_DEBUG,
                       "[DISCONT-BUF] Dropping non-keyframe video packet: stream=%d dts=%"PRId64"\n",
                       dt.pkt_demux->stream_index, dt.pkt_demux->dts);
                av_packet_unref(dt.pkt_demux);
                continue;
            }
        }

        ret = input_packet_process(d, dt.pkt_demux, &send_flags);
        if (ret < 0)
            break;

        if (d->readrate)
            readrate_sleep(d);

        ret = demux_send(d, &dt, ds, dt.pkt_demux, send_flags);
        if (ret < 0)
            break;

        d->diag.pkts_sent++;
        if (ds->ist.par->codec_type == AVMEDIA_TYPE_VIDEO)
            d->diag.vid_pkts_sent++;
        else if (ds->ist.par->codec_type == AVMEDIA_TYPE_AUDIO)
            d->diag.aud_pkts_sent++;

        /* Track last sent position (END of packet, not start) for discontinuity buffer
         * cumulative offset calculation. Using end time ensures new content starts
         * AFTER old content ends, not at the same time (which causes audio overlap).
         * Duration is estimated from codec parameters since pkt->duration may be 0. */
        if (ds->last_raw_dts != AV_NOPTS_VALUE && d->discont_buf.capacity > 0) {
            int64_t pkt_duration = discont_estimate_pkt_duration(&ds->ist, dt.pkt_demux);
            int64_t output_dts = ds->last_raw_dts + d->discont_buf.cumulative_ts_offset;
            int64_t output_end = output_dts + pkt_duration;
            if (output_end > d->discont_buf.last_sent_dts || d->discont_buf.last_sent_dts == AV_NOPTS_VALUE)
                d->discont_buf.last_sent_dts = output_end;
        }

        discont_diag_log(d);
    }

    // EOF/EXIT is normal termination
    if (ret == AVERROR_EOF || ret == AVERROR_EXIT)
        ret = 0;

finish:
    demux_thread_uninit(&dt);

    return ret;
}

static void demux_final_stats(Demuxer *d)
{
    InputFile *f = &d->f;
    uint64_t total_packets = 0, total_size = 0;

    av_log(f, AV_LOG_VERBOSE, "Input file #%d (%s):\n",
           f->index, f->ctx->url);

    for (int j = 0; j < f->nb_streams; j++) {
        InputStream *ist = f->streams[j];
        DemuxStream  *ds = ds_from_ist(ist);
        enum AVMediaType type = ist->par->codec_type;

        if (ds->discard || type == AVMEDIA_TYPE_ATTACHMENT)
            continue;

        total_size    += ds->data_size;
        total_packets += ds->nb_packets;

        av_log(f, AV_LOG_VERBOSE, "  Input stream #%d:%d (%s): ",
               f->index, j, av_get_media_type_string(type));
        av_log(f, AV_LOG_VERBOSE, "%"PRIu64" packets read (%"PRIu64" bytes); ",
               ds->nb_packets, ds->data_size);

        if (ds->decoding_needed) {
            av_log(f, AV_LOG_VERBOSE,
                   "%"PRIu64" frames decoded; %"PRIu64" decode errors",
                   ist->decoder->frames_decoded, ist->decoder->decode_errors);
            if (type == AVMEDIA_TYPE_AUDIO)
                av_log(f, AV_LOG_VERBOSE, " (%"PRIu64" samples)", ist->decoder->samples_decoded);
            av_log(f, AV_LOG_VERBOSE, "; ");
        }

        av_log(f, AV_LOG_VERBOSE, "\n");
    }

    av_log(f, AV_LOG_VERBOSE, "  Total: %"PRIu64" packets (%"PRIu64" bytes) demuxed\n",
           total_packets, total_size);
}

static void ist_free(InputStream **pist)
{
    InputStream *ist = *pist;
    DemuxStream *ds;

    if (!ist)
        return;
    ds = ds_from_ist(ist);

    dec_free(&ist->decoder);

    av_dict_free(&ds->decoder_opts);
    av_freep(&ist->filters);
    av_freep(&ds->dec_opts.hwaccel_device);

    avcodec_parameters_free(&ist->par);

    av_frame_free(&ds->decoded_params);

    av_bsf_free(&ds->bsf);

    av_freep(pist);
}

static void istg_free(InputStreamGroup **pistg)
{
    InputStreamGroup *istg = *pistg;

    if (!istg)
        return;

    av_freep(pistg);
}

void ifile_close(InputFile **pf)
{
    InputFile *f = *pf;
    Demuxer   *d = demuxer_from_ifile(f);

    if (!f)
        return;

    if (d->read_started)
        demux_final_stats(d);

    for (int i = 0; i < f->nb_streams; i++)
        ist_free(&f->streams[i]);
    av_freep(&f->streams);

    for (int i = 0; i < f->nb_stream_groups; i++)
        istg_free(&f->stream_groups[i]);
    av_freep(&f->stream_groups);

    avformat_close_input(&f->ctx);

    av_packet_free(&d->pkt_heartbeat);

    /* Free discontinuity buffer */
    discont_buffer_free(&d->discont_buf);

    av_freep(pf);
}

int ist_use(InputStream *ist, int decoding_needed,
            const ViewSpecifier *vs, SchedulerNode *src)
{
    Demuxer      *d = demuxer_from_ifile(ist->file);
    DemuxStream *ds = ds_from_ist(ist);
    int ret;

    if (ist->user_set_discard == AVDISCARD_ALL) {
        av_log(ist, AV_LOG_ERROR, "Cannot %s a disabled input stream\n",
               decoding_needed ? "decode" : "streamcopy");
        return AVERROR(EINVAL);
    }

    if (decoding_needed && !ist->dec) {
        av_log(ist, AV_LOG_ERROR,
               "Decoding requested, but no decoder found for: %s\n",
                avcodec_get_name(ist->par->codec_id));
        return AVERROR(EINVAL);
    }

    if (ds->sch_idx_stream < 0) {
        ret = sch_add_demux_stream(d->sch, d->f.index);
        if (ret < 0)
            return ret;
        ds->sch_idx_stream = ret;
    }

    if (ds->discard) {
        ds->discard = 0;
        d->nb_streams_used++;
    }

    ist->st->discard      = ist->user_set_discard;
    ds->decoding_needed   |= decoding_needed;
    ds->streamcopy_needed |= !decoding_needed;

    if (decoding_needed && ds->sch_idx_dec < 0) {
        int is_audio = ist->st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO;
        int is_unreliable = !!(d->f.ctx->iformat->flags & AVFMT_NOTIMESTAMPS);
        int64_t use_wallclock_as_timestamps;

        ret = av_opt_get_int(d->f.ctx, "use_wallclock_as_timestamps", 0, &use_wallclock_as_timestamps);
        if (ret < 0)
            return ret;

        if (use_wallclock_as_timestamps)
            is_unreliable = 0;

        ds->dec_opts.flags |= (!!ist->fix_sub_duration * DECODER_FLAG_FIX_SUB_DURATION) |
                              (!!is_unreliable * DECODER_FLAG_TS_UNRELIABLE) |
                              (!!(d->loop && is_audio) * DECODER_FLAG_SEND_END_TS)
#if FFMPEG_OPT_TOP
                              | ((ist->top_field_first >= 0) * DECODER_FLAG_TOP_FIELD_FIRST)
#endif
                             ;

        if (ist->framerate.num) {
            ds->dec_opts.flags     |= DECODER_FLAG_FRAMERATE_FORCED;
            ds->dec_opts.framerate  = ist->framerate;
        } else
            ds->dec_opts.framerate  = ist->st->avg_frame_rate;

        if (ist->dec->id == AV_CODEC_ID_DVB_SUBTITLE &&
           (ds->decoding_needed & DECODING_FOR_OST)) {
            av_dict_set(&ds->decoder_opts, "compute_edt", "1", AV_DICT_DONT_OVERWRITE);
            if (ds->decoding_needed & DECODING_FOR_FILTER)
                av_log(ist, AV_LOG_WARNING,
                       "Warning using DVB subtitles for filtering and output at the "
                       "same time is not fully supported, also see -compute_edt [0|1]\n");
        }

        snprintf(ds->dec_name, sizeof(ds->dec_name), "%d:%d", ist->file->index, ist->index);
        ds->dec_opts.name = ds->dec_name;

        ds->dec_opts.codec = ist->dec;
        ds->dec_opts.par   = ist->par;

        ds->dec_opts.log_parent = ist;

        ds->decoded_params = av_frame_alloc();
        if (!ds->decoded_params)
            return AVERROR(ENOMEM);

        ret = dec_init(&ist->decoder, d->sch,
                       &ds->decoder_opts, &ds->dec_opts, ds->decoded_params);
        if (ret < 0)
            return ret;
        ds->sch_idx_dec = ret;

        ret = sch_connect(d->sch, SCH_DSTREAM(d->f.index, ds->sch_idx_stream),
                                  SCH_DEC_IN(ds->sch_idx_dec));
        if (ret < 0)
            return ret;

        d->have_audio_dec |= is_audio;
    }

    if (decoding_needed && ist->par->codec_type == AVMEDIA_TYPE_VIDEO) {
        ret = dec_request_view(ist->decoder, vs, src);
        if (ret < 0)
            return ret;
    } else {
        *src = decoding_needed                             ?
               SCH_DEC_OUT(ds->sch_idx_dec, 0)             :
               SCH_DSTREAM(d->f.index, ds->sch_idx_stream);
    }

    return 0;
}

int ist_filter_add(InputStream *ist, InputFilter *ifilter, int is_simple,
                   const ViewSpecifier *vs, InputFilterOptions *opts,
                   SchedulerNode *src)
{
    Demuxer      *d = demuxer_from_ifile(ist->file);
    DemuxStream *ds = ds_from_ist(ist);
    int64_t tsoffset = 0;
    int ret;

    ret = ist_use(ist, is_simple ? DECODING_FOR_OST : DECODING_FOR_FILTER,
                  vs, src);
    if (ret < 0)
        return ret;

    ret = GROW_ARRAY(ist->filters, ist->nb_filters);
    if (ret < 0)
        return ret;

    ist->filters[ist->nb_filters - 1] = ifilter;

    if (ist->par->codec_type == AVMEDIA_TYPE_VIDEO) {
        const AVPacketSideData *sd = av_packet_side_data_get(ist->par->coded_side_data,
                                                             ist->par->nb_coded_side_data,
                                                             AV_PKT_DATA_FRAME_CROPPING);
        if (ist->framerate.num > 0 && ist->framerate.den > 0) {
            opts->framerate = ist->framerate;
            opts->flags |= IFILTER_FLAG_CFR;
        } else
            opts->framerate = av_guess_frame_rate(d->f.ctx, ist->st, NULL);
        if (sd && sd->size >= sizeof(uint32_t) * 4) {
            opts->crop_top    = AV_RL32(sd->data +  0);
            opts->crop_bottom = AV_RL32(sd->data +  4);
            opts->crop_left   = AV_RL32(sd->data +  8);
            opts->crop_right  = AV_RL32(sd->data + 12);
            if (ds->apply_cropping && ds->apply_cropping != CROP_CODEC &&
                (opts->crop_top | opts->crop_bottom | opts->crop_left | opts->crop_right))
                opts->flags |= IFILTER_FLAG_CROP;
        }
    } else if (ist->par->codec_type == AVMEDIA_TYPE_SUBTITLE) {
        /* Compute the size of the canvas for the subtitles stream.
           If the subtitles codecpar has set a size, use it. Otherwise use the
           maximum dimensions of the video streams in the same file. */
        opts->sub2video_width  = ist->par->width;
        opts->sub2video_height = ist->par->height;
        if (!(opts->sub2video_width && opts->sub2video_height)) {
            for (int j = 0; j < d->f.nb_streams; j++) {
                AVCodecParameters *par1 = d->f.streams[j]->par;
                if (par1->codec_type == AVMEDIA_TYPE_VIDEO) {
                    opts->sub2video_width  = FFMAX(opts->sub2video_width,  par1->width);
                    opts->sub2video_height = FFMAX(opts->sub2video_height, par1->height);
                }
            }
        }

        if (!(opts->sub2video_width && opts->sub2video_height)) {
            opts->sub2video_width  = FFMAX(opts->sub2video_width,  720);
            opts->sub2video_height = FFMAX(opts->sub2video_height, 576);
        }

        if (!d->pkt_heartbeat) {
            d->pkt_heartbeat = av_packet_alloc();
            if (!d->pkt_heartbeat)
                return AVERROR(ENOMEM);
        }
        ds->have_sub2video = 1;
    }

    ret = av_frame_copy_props(opts->fallback, ds->decoded_params);
    if (ret < 0)
        return ret;
    opts->fallback->format = ds->decoded_params->format;
    opts->fallback->width  = ds->decoded_params->width;
    opts->fallback->height = ds->decoded_params->height;

    ret = av_channel_layout_copy(&opts->fallback->ch_layout, &ds->decoded_params->ch_layout);
    if (ret < 0)
        return ret;

    if (copy_ts) {
        tsoffset = d->f.start_time == AV_NOPTS_VALUE ? 0 : d->f.start_time;
        if (!start_at_zero && d->f.ctx->start_time != AV_NOPTS_VALUE)
            tsoffset += d->f.ctx->start_time;
    }
    opts->trim_start_us = ((d->f.start_time == AV_NOPTS_VALUE) || !d->accurate_seek) ?
                          AV_NOPTS_VALUE : tsoffset;
    opts->trim_end_us   = d->recording_time;

    opts->name = av_strdup(ds->dec_name);
    if (!opts->name)
        return AVERROR(ENOMEM);

    opts->flags |= IFILTER_FLAG_AUTOROTATE * !!(ds->autorotate) |
                   IFILTER_FLAG_REINIT     * !!(ds->reinit_filters) |
                   IFILTER_FLAG_DROPCHANGED* !!(ds->drop_changed);

    return 0;
}

static int choose_decoder(const OptionsContext *o, void *logctx,
                          AVFormatContext *s, AVStream *st,
                          enum HWAccelID hwaccel_id, enum AVHWDeviceType hwaccel_device_type,
                          const AVCodec **pcodec)

{
    const char *codec_name = NULL;

    opt_match_per_stream_str(logctx, &o->codec_names, s, st, &codec_name);
    if (codec_name) {
        int ret = find_codec(NULL, codec_name, st->codecpar->codec_type, 0, pcodec);
        if (ret < 0)
            return ret;
        st->codecpar->codec_id = (*pcodec)->id;
        if (recast_media && st->codecpar->codec_type != (*pcodec)->type)
            st->codecpar->codec_type = (*pcodec)->type;
        return 0;
    } else {
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO &&
            hwaccel_id == HWACCEL_GENERIC &&
            hwaccel_device_type != AV_HWDEVICE_TYPE_NONE) {
            const AVCodec *c;
            void *i = NULL;

            while ((c = av_codec_iterate(&i))) {
                const AVCodecHWConfig *config;

                if (c->id != st->codecpar->codec_id ||
                    !av_codec_is_decoder(c))
                    continue;

                for (int j = 0; config = avcodec_get_hw_config(c, j); j++) {
                    if (config->device_type == hwaccel_device_type) {
                        av_log(logctx, AV_LOG_VERBOSE, "Selecting decoder '%s' because of requested hwaccel method %s\n",
                               c->name, av_hwdevice_get_type_name(hwaccel_device_type));
                        *pcodec = c;
                        return 0;
                    }
                }
            }
        }

        *pcodec = avcodec_find_decoder(st->codecpar->codec_id);
        return 0;
    }
}

static int guess_input_channel_layout(InputStream *ist, AVCodecParameters *par,
                                      int guess_layout_max)
{
    if (par->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
        char layout_name[256];

        if (par->ch_layout.nb_channels > guess_layout_max)
            return 0;
        av_channel_layout_default(&par->ch_layout, par->ch_layout.nb_channels);
        if (par->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC)
            return 0;
        av_channel_layout_describe(&par->ch_layout, layout_name, sizeof(layout_name));
        av_log(ist, AV_LOG_WARNING, "Guessed Channel Layout: %s\n", layout_name);
    }
    return 1;
}

static int add_display_matrix_to_stream(const OptionsContext *o,
                                        AVFormatContext *ctx, InputStream *ist)
{
    AVStream *st = ist->st;
    DemuxStream *ds = ds_from_ist(ist);
    AVPacketSideData *sd;
    double rotation = DBL_MAX;
    int hflip = -1, vflip = -1;
    int hflip_set = 0, vflip_set = 0, rotation_set = 0;
    int32_t *buf;

    opt_match_per_stream_dbl(ist, &o->display_rotations, ctx, st, &rotation);
    opt_match_per_stream_int(ist, &o->display_hflips, ctx, st, &hflip);
    opt_match_per_stream_int(ist, &o->display_vflips, ctx, st, &vflip);

    rotation_set = rotation != DBL_MAX;
    hflip_set    = hflip != -1;
    vflip_set    = vflip != -1;

    if (!rotation_set && !hflip_set && !vflip_set)
        return 0;

    sd = av_packet_side_data_new(&st->codecpar->coded_side_data,
                                 &st->codecpar->nb_coded_side_data,
                                 AV_PKT_DATA_DISPLAYMATRIX,
                                 sizeof(int32_t) * 9, 0);
    if (!sd) {
        av_log(ist, AV_LOG_FATAL, "Failed to generate a display matrix!\n");
        return AVERROR(ENOMEM);
    }

    buf = (int32_t *)sd->data;
    av_display_rotation_set(buf,
                            rotation_set ? -(rotation) : -0.0f);

    av_display_matrix_flip(buf,
                           hflip_set ? hflip : 0,
                           vflip_set ? vflip : 0);

    ds->force_display_matrix = 1;

    return 0;
}

static const char *input_stream_item_name(void *obj)
{
    const DemuxStream *ds = obj;

    return ds->log_name;
}

static const AVClass input_stream_class = {
    .class_name = "InputStream",
    .version    = LIBAVUTIL_VERSION_INT,
    .item_name  = input_stream_item_name,
    .category   = AV_CLASS_CATEGORY_DEMUXER,
};

static DemuxStream *demux_stream_alloc(Demuxer *d, AVStream *st)
{
    const char *type_str = av_get_media_type_string(st->codecpar->codec_type);
    InputFile    *f = &d->f;
    DemuxStream *ds;

    ds = allocate_array_elem(&f->streams, sizeof(*ds), &f->nb_streams);
    if (!ds)
        return NULL;

    ds->sch_idx_stream = -1;
    ds->sch_idx_dec    = -1;

    ds->ist.st         = st;
    ds->ist.file       = f;
    ds->ist.index      = st->index;
    ds->ist.class      = &input_stream_class;

    snprintf(ds->log_name, sizeof(ds->log_name), "%cist#%d:%d/%s",
             type_str ? *type_str : '?', d->f.index, st->index,
             avcodec_get_name(st->codecpar->codec_id));

    return ds;
}

static int ist_add(const OptionsContext *o, Demuxer *d, AVStream *st, AVDictionary **opts_used)
{
    AVFormatContext *ic = d->f.ctx;
    AVCodecParameters *par = st->codecpar;
    DemuxStream *ds;
    InputStream *ist;
    const char *framerate = NULL, *hwaccel_device = NULL;
    const char *hwaccel = NULL;
    const char *apply_cropping = NULL;
    const char *hwaccel_output_format = NULL;
    const char *codec_tag = NULL;
    const char *bsfs = NULL;
    char *next;
    const char *discard_str = NULL;
    int ret;

    ds  = demux_stream_alloc(d, st);
    if (!ds)
        return AVERROR(ENOMEM);

    ist = &ds->ist;

    ds->discard     = 1;
    st->discard  = AVDISCARD_ALL;
    ds->first_dts   = AV_NOPTS_VALUE;
    ds->next_dts    = AV_NOPTS_VALUE;
    ds->last_raw_dts = AV_NOPTS_VALUE;

    ds->dec_opts.time_base = st->time_base;

    ds->ts_scale = 1.0;
    opt_match_per_stream_dbl(ist, &o->ts_scale, ic, st, &ds->ts_scale);

    ds->autorotate = 1;
    opt_match_per_stream_int(ist, &o->autorotate, ic, st, &ds->autorotate);

    ds->apply_cropping = CROP_ALL;
    opt_match_per_stream_str(ist, &o->apply_cropping, ic, st, &apply_cropping);
    if (apply_cropping) {
        const AVOption opts[] = {
            { "apply_cropping", NULL, 0, AV_OPT_TYPE_INT,
                    { .i64 = CROP_ALL }, CROP_DISABLED, CROP_CONTAINER, AV_OPT_FLAG_DECODING_PARAM, .unit = "apply_cropping" },
                { "none",      NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CROP_DISABLED  }, .unit = "apply_cropping" },
                { "all",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CROP_ALL       }, .unit = "apply_cropping" },
                { "codec",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CROP_CODEC     }, .unit = "apply_cropping" },
                { "container", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = CROP_CONTAINER }, .unit = "apply_cropping" },
            { NULL },
        };
        const AVClass class = {
            .class_name = "apply_cropping",
            .item_name  = av_default_item_name,
            .option     = opts,
            .version    = LIBAVUTIL_VERSION_INT,
        };
        const AVClass *pclass = &class;

        ret = av_opt_eval_int(&pclass, opts, apply_cropping, &ds->apply_cropping);
        if (ret < 0) {
            av_log(ist, AV_LOG_ERROR, "Invalid apply_cropping value '%s'.\n", apply_cropping);
            return ret;
        }
    }

    opt_match_per_stream_str(ist, &o->codec_tags, ic, st, &codec_tag);
    if (codec_tag) {
        uint32_t tag = strtol(codec_tag, &next, 0);
        if (*next) {
            uint8_t buf[4] = { 0 };
            memcpy(buf, codec_tag, FFMIN(sizeof(buf), strlen(codec_tag)));
            tag = AV_RL32(buf);
        }

        st->codecpar->codec_tag = tag;
    }

    if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        ret = add_display_matrix_to_stream(o, ic, ist);
        if (ret < 0)
            return ret;

        opt_match_per_stream_str(ist, &o->hwaccels, ic, st, &hwaccel);
        opt_match_per_stream_str(ist, &o->hwaccel_output_formats, ic, st,
                                       &hwaccel_output_format);
        if (!hwaccel_output_format && hwaccel && !strcmp(hwaccel, "cuvid")) {
            av_log(ist, AV_LOG_WARNING,
                "WARNING: defaulting hwaccel_output_format to cuda for compatibility "
                "with old commandlines. This behaviour is DEPRECATED and will be removed "
                "in the future. Please explicitly set \"-hwaccel_output_format cuda\".\n");
            ds->dec_opts.hwaccel_output_format = AV_PIX_FMT_CUDA;
        } else if (!hwaccel_output_format && hwaccel && !strcmp(hwaccel, "qsv")) {
            av_log(ist, AV_LOG_WARNING,
                "WARNING: defaulting hwaccel_output_format to qsv for compatibility "
                "with old commandlines. This behaviour is DEPRECATED and will be removed "
                "in the future. Please explicitly set \"-hwaccel_output_format qsv\".\n");
            ds->dec_opts.hwaccel_output_format = AV_PIX_FMT_QSV;
        } else if (!hwaccel_output_format && hwaccel && !strcmp(hwaccel, "mediacodec")) {
            // There is no real AVHWFrameContext implementation. Set
            // hwaccel_output_format to avoid av_hwframe_transfer_data error.
            ds->dec_opts.hwaccel_output_format = AV_PIX_FMT_MEDIACODEC;
        } else if (hwaccel_output_format) {
            ds->dec_opts.hwaccel_output_format = av_get_pix_fmt(hwaccel_output_format);
            if (ds->dec_opts.hwaccel_output_format == AV_PIX_FMT_NONE) {
                av_log(ist, AV_LOG_FATAL, "Unrecognised hwaccel output "
                       "format: %s", hwaccel_output_format);
            }
        } else {
            ds->dec_opts.hwaccel_output_format = AV_PIX_FMT_NONE;
        }

        if (hwaccel) {
            // The NVDEC hwaccels use a CUDA device, so remap the name here.
            if (!strcmp(hwaccel, "nvdec") || !strcmp(hwaccel, "cuvid"))
                hwaccel = "cuda";

            if (!strcmp(hwaccel, "none"))
                ds->dec_opts.hwaccel_id = HWACCEL_NONE;
            else if (!strcmp(hwaccel, "auto"))
                ds->dec_opts.hwaccel_id = HWACCEL_AUTO;
            else {
                enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwaccel);
                if (type != AV_HWDEVICE_TYPE_NONE) {
                    ds->dec_opts.hwaccel_id = HWACCEL_GENERIC;
                    ds->dec_opts.hwaccel_device_type = type;
                }

                if (!ds->dec_opts.hwaccel_id) {
                    av_log(ist, AV_LOG_FATAL, "Unrecognized hwaccel: %s.\n",
                           hwaccel);
                    av_log(ist, AV_LOG_FATAL, "Supported hwaccels: ");
                    type = AV_HWDEVICE_TYPE_NONE;
                    while ((type = av_hwdevice_iterate_types(type)) !=
                           AV_HWDEVICE_TYPE_NONE)
                        av_log(ist, AV_LOG_FATAL, "%s ",
                               av_hwdevice_get_type_name(type));
                    av_log(ist, AV_LOG_FATAL, "\n");
                    return AVERROR(EINVAL);
                }
            }
        }

        opt_match_per_stream_str(ist, &o->hwaccel_devices, ic, st, &hwaccel_device);
        if (hwaccel_device) {
            ds->dec_opts.hwaccel_device = av_strdup(hwaccel_device);
            if (!ds->dec_opts.hwaccel_device)
                return AVERROR(ENOMEM);
        }
    }

    ret = choose_decoder(o, ist, ic, st, ds->dec_opts.hwaccel_id,
                         ds->dec_opts.hwaccel_device_type, &ist->dec);
    if (ret < 0)
        return ret;

    if (ist->dec) {
        ret = filter_codec_opts(o->g->codec_opts, ist->st->codecpar->codec_id,
                                ic, st, ist->dec, &ds->decoder_opts, opts_used);
        if (ret < 0)
            return ret;
    }

    ds->reinit_filters = -1;
    opt_match_per_stream_int(ist, &o->reinit_filters, ic, st, &ds->reinit_filters);

    ds->drop_changed = 0;
    opt_match_per_stream_int(ist, &o->drop_changed, ic, st, &ds->drop_changed);

    if (ds->drop_changed && ds->reinit_filters) {
        if (ds->reinit_filters > 0) {
            av_log(ist, AV_LOG_ERROR, "drop_changed and reinit_filters both enabled. These are mutually exclusive.\n");
            return AVERROR(EINVAL);
        }
        ds->reinit_filters = 0;
    }

    ist->user_set_discard = AVDISCARD_NONE;

    if ((o->video_disable && ist->st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) ||
        (o->audio_disable && ist->st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) ||
        (o->subtitle_disable && ist->st->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) ||
        (o->data_disable && ist->st->codecpar->codec_type == AVMEDIA_TYPE_DATA))
            ist->user_set_discard = AVDISCARD_ALL;

    opt_match_per_stream_str(ist, &o->discard, ic, st, &discard_str);
    if (discard_str) {
        ret = av_opt_set(ist->st, "discard", discard_str, 0);
        if (ret  < 0) {
            av_log(ist, AV_LOG_ERROR, "Error parsing discard %s.\n", discard_str);
            return ret;
        }
        ist->user_set_discard = ist->st->discard;
    }

    ds->dec_opts.flags |= DECODER_FLAG_BITEXACT * !!o->bitexact;

    av_dict_set_int(&ds->decoder_opts, "apply_cropping",
                    ds->apply_cropping && ds->apply_cropping != CROP_CONTAINER, 0);

    if (ds->force_display_matrix) {
        char buf[32];
        if (av_dict_get(ds->decoder_opts, "side_data_prefer_packet", NULL, 0))
            buf[0] = ',';
        else
            buf[0] = '\0';
        av_strlcat(buf, "displaymatrix", sizeof(buf));
        av_dict_set(&ds->decoder_opts, "side_data_prefer_packet", buf, AV_DICT_APPEND);
    }
    /* Attached pics are sparse, therefore we would not want to delay their decoding
     * till EOF. */
    if (ist->st->disposition & AV_DISPOSITION_ATTACHED_PIC)
        av_dict_set(&ds->decoder_opts, "thread_type", "-frame", 0);

    switch (par->codec_type) {
    case AVMEDIA_TYPE_VIDEO:
        opt_match_per_stream_str(ist, &o->frame_rates, ic, st, &framerate);
        if (framerate) {
            ret = av_parse_video_rate(&ist->framerate, framerate);
            if (ret < 0) {
                av_log(ist, AV_LOG_ERROR, "Error parsing framerate %s.\n",
                       framerate);
                return ret;
            }
        }

#if FFMPEG_OPT_TOP
        ist->top_field_first = -1;
        opt_match_per_stream_int(ist, &o->top_field_first, ic, st, &ist->top_field_first);
#endif

        break;
    case AVMEDIA_TYPE_AUDIO: {
        const char *ch_layout_str = NULL;

        opt_match_per_stream_str(ist, &o->audio_ch_layouts, ic, st, &ch_layout_str);
        if (ch_layout_str) {
            AVChannelLayout ch_layout;
            ret = av_channel_layout_from_string(&ch_layout, ch_layout_str);
            if (ret < 0) {
                av_log(ist, AV_LOG_ERROR, "Error parsing channel layout %s.\n", ch_layout_str);
                return ret;
            }
            if (par->ch_layout.nb_channels <= 0 || par->ch_layout.nb_channels == ch_layout.nb_channels) {
                av_channel_layout_uninit(&par->ch_layout);
                par->ch_layout = ch_layout;
            } else {
                av_log(ist, AV_LOG_ERROR,
                    "Specified channel layout '%s' has %d channels, but input has %d channels.\n",
                    ch_layout_str, ch_layout.nb_channels, par->ch_layout.nb_channels);
                av_channel_layout_uninit(&ch_layout);
                return AVERROR(EINVAL);
            }
        } else {
            int guess_layout_max = INT_MAX;
            opt_match_per_stream_int(ist, &o->guess_layout_max, ic, st, &guess_layout_max);
            guess_input_channel_layout(ist, par, guess_layout_max);
        }
        break;
    }
    case AVMEDIA_TYPE_DATA:
    case AVMEDIA_TYPE_SUBTITLE: {
        const char *canvas_size = NULL;

        opt_match_per_stream_int(ist, &o->fix_sub_duration, ic, st, &ist->fix_sub_duration);
        opt_match_per_stream_str(ist, &o->canvas_sizes, ic, st, &canvas_size);
        if (canvas_size) {
            ret = av_parse_video_size(&par->width, &par->height,
                                      canvas_size);
            if (ret < 0) {
                av_log(ist, AV_LOG_FATAL, "Invalid canvas size: %s.\n", canvas_size);
                return ret;
            }
        }
        break;
    }
    case AVMEDIA_TYPE_ATTACHMENT:
    case AVMEDIA_TYPE_UNKNOWN:
        break;
    default: av_assert0(0);
    }

    ist->par = avcodec_parameters_alloc();
    if (!ist->par)
        return AVERROR(ENOMEM);

    ret = avcodec_parameters_copy(ist->par, par);
    if (ret < 0) {
        av_log(ist, AV_LOG_ERROR, "Error exporting stream parameters.\n");
        return ret;
    }

    if (ist->st->sample_aspect_ratio.num)
        ist->par->sample_aspect_ratio = ist->st->sample_aspect_ratio;

    opt_match_per_stream_str(ist, &o->bitstream_filters, ic, st, &bsfs);
    if (bsfs) {
        ret = av_bsf_list_parse_str(bsfs, &ds->bsf);
        if (ret < 0) {
            av_log(ist, AV_LOG_ERROR,
                   "Error parsing bitstream filter sequence '%s': %s\n",
                   bsfs, av_err2str(ret));
            return ret;
        }

        ret = avcodec_parameters_copy(ds->bsf->par_in, ist->par);
        if (ret < 0)
            return ret;
        ds->bsf->time_base_in = ist->st->time_base;

        ret = av_bsf_init(ds->bsf);
        if (ret < 0) {
            av_log(ist, AV_LOG_ERROR, "Error initializing bitstream filters: %s\n",
                   av_err2str(ret));
            return ret;
        }

        ret = avcodec_parameters_copy(ist->par, ds->bsf->par_out);
        if (ret < 0)
            return ret;
    }

    ds->codec_desc = avcodec_descriptor_get(ist->par->codec_id);

    return 0;
}

static const char *input_stream_group_item_name(void *obj)
{
    const DemuxStreamGroup *dsg = obj;

    return dsg->log_name;
}

static const AVClass input_stream_group_class = {
    .class_name = "InputStreamGroup",
    .version    = LIBAVUTIL_VERSION_INT,
    .item_name  = input_stream_group_item_name,
    .category   = AV_CLASS_CATEGORY_DEMUXER,
};

static DemuxStreamGroup *demux_stream_group_alloc(Demuxer *d, AVStreamGroup *stg)
{
    InputFile    *f = &d->f;
    DemuxStreamGroup *dsg;

    dsg = allocate_array_elem(&f->stream_groups, sizeof(*dsg), &f->nb_stream_groups);
    if (!dsg)
        return NULL;

    dsg->istg.stg        = stg;
    dsg->istg.file       = f;
    dsg->istg.index      = stg->index;
    dsg->istg.class      = &input_stream_group_class;

    snprintf(dsg->log_name, sizeof(dsg->log_name), "istg#%d:%d/%s",
             d->f.index, stg->index, avformat_stream_group_name(stg->type));

    return dsg;
}

static int istg_parse_tile_grid(const OptionsContext *o, Demuxer *d, InputStreamGroup *istg)
{
    InputFile *f = &d->f;
    AVFormatContext *ic = d->f.ctx;
    AVStreamGroup *stg = istg->stg;
    const AVStreamGroupTileGrid *tg = stg->params.tile_grid;
    OutputFilterOptions opts;
    AVBPrint bp;
    char *graph_str;
    int autorotate = 1;
    const char *apply_cropping = NULL;
    int  ret;

    if (tg->nb_tiles == 1)
        return 0;

    memset(&opts, 0, sizeof(opts));

    opt_match_per_stream_group_int(istg, &o->autorotate, ic, stg, &autorotate);
    if (autorotate)
        opts.flags |= OFILTER_FLAG_AUTOROTATE;

    opts.flags |= OFILTER_FLAG_CROP;
    opt_match_per_stream_group_str(istg, &o->apply_cropping, ic, stg, &apply_cropping);
    if (apply_cropping) {
        char *p;
        int crop = strtol(apply_cropping, &p, 0);
        if (*p)
            return AVERROR(EINVAL);
        if (!crop)
            opts.flags &= ~OFILTER_FLAG_CROP;
    }

    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);
    for (int i = 0; i < tg->nb_tiles; i++)
        av_bprintf(&bp, "[%d:g:%d:%d]", f->index, stg->index, tg->offsets[i].idx);
    av_bprintf(&bp, "xstack=inputs=%d:layout=", tg->nb_tiles);
    for (int i = 0; i < tg->nb_tiles - 1; i++)
        av_bprintf(&bp, "%d_%d|", tg->offsets[i].horizontal,
                                  tg->offsets[i].vertical);
    av_bprintf(&bp, "%d_%d:fill=0x%02X%02X%02X@0x%02X", tg->offsets[tg->nb_tiles - 1].horizontal,
                                                        tg->offsets[tg->nb_tiles - 1].vertical,
                                                        tg->background[0], tg->background[1],
                                                        tg->background[2], tg->background[3]);
    av_bprintf(&bp, "[%d:g:%d]", f->index, stg->index);
    ret = av_bprint_finalize(&bp, &graph_str);
    if (ret < 0)
        return ret;

    if (tg->coded_width != tg->width || tg->coded_height != tg->height) {
        opts.crop_top    = tg->vertical_offset;
        opts.crop_bottom = tg->coded_height - tg->height - tg->vertical_offset;
        opts.crop_left   = tg->horizontal_offset;
        opts.crop_right  = tg->coded_width - tg->width - tg->horizontal_offset;
    }

    for (int i = 0; i < tg->nb_coded_side_data; i++) {
        const AVPacketSideData *sd = &tg->coded_side_data[i];

        ret = av_packet_side_data_to_frame(&opts.side_data, &opts.nb_side_data, sd, 0);
        if (ret < 0 && ret != AVERROR(EINVAL))
            goto fail;
    }

    ret = fg_create(NULL, &graph_str, d->sch, &opts);
    if (ret < 0)
        goto fail;

    istg->fg = filtergraphs[nb_filtergraphs-1];
    istg->fg->is_internal = 1;

    ret = 0;
fail:
    if (ret < 0)
        av_freep(&graph_str);

    return ret;
}

static int istg_add(const OptionsContext *o, Demuxer *d, AVStreamGroup *stg)
{
    DemuxStreamGroup *dsg;
    InputStreamGroup *istg;
    int ret;

    dsg = demux_stream_group_alloc(d, stg);
    if (!dsg)
        return AVERROR(ENOMEM);

    istg = &dsg->istg;

    switch (stg->type) {
    case AV_STREAM_GROUP_PARAMS_TILE_GRID:
        ret = istg_parse_tile_grid(o, d, istg);
        if (ret < 0)
            return ret;
        break;
    default:
        break;
    }

    return 0;
}

static int dump_attachment(InputStream *ist, const char *filename)
{
    AVStream *st = ist->st;
    int ret;
    AVIOContext *out = NULL;
    const AVDictionaryEntry *e;

    if (!st->codecpar->extradata_size) {
        av_log(ist, AV_LOG_WARNING, "No extradata to dump.\n");
        return 0;
    }
    if (!*filename && (e = av_dict_get(st->metadata, "filename", NULL, 0)))
        filename = e->value;
    if (!*filename) {
        av_log(ist, AV_LOG_FATAL, "No filename specified and no 'filename' tag");
        return AVERROR(EINVAL);
    }

    ret = assert_file_overwrite(filename);
    if (ret < 0)
        return ret;

    if ((ret = avio_open2(&out, filename, AVIO_FLAG_WRITE, &int_cb, NULL)) < 0) {
        av_log(ist, AV_LOG_FATAL, "Could not open file %s for writing.\n",
               filename);
        return ret;
    }

    avio_write(out, st->codecpar->extradata, st->codecpar->extradata_size);
    ret = avio_close(out);

    if (ret >= 0)
        av_log(ist, AV_LOG_INFO, "Wrote attachment (%d bytes) to '%s'\n",
               st->codecpar->extradata_size, filename);

    return ret;
}

static const char *input_file_item_name(void *obj)
{
    const Demuxer *d = obj;

    return d->log_name;
}

static const AVClass input_file_class = {
    .class_name = "InputFile",
    .version    = LIBAVUTIL_VERSION_INT,
    .item_name  = input_file_item_name,
    .category   = AV_CLASS_CATEGORY_DEMUXER,
};

static Demuxer *demux_alloc(void)
{
    Demuxer *d = allocate_array_elem(&input_files, sizeof(*d), &nb_input_files);

    if (!d)
        return NULL;

    d->f.class = &input_file_class;
    d->f.index = nb_input_files - 1;

    snprintf(d->log_name, sizeof(d->log_name), "in#%d", d->f.index);

    return d;
}

int ifile_open(const OptionsContext *o, const char *filename, Scheduler *sch)
{
    Demuxer   *d;
    InputFile *f;
    AVFormatContext *ic;
    const AVInputFormat *file_iformat = NULL;
    int err, ret = 0;
    int64_t timestamp;
    AVDictionary *opts_used = NULL;
    const char*    video_codec_name = NULL;
    const char*    audio_codec_name = NULL;
    const char* subtitle_codec_name = NULL;
    const char*     data_codec_name = NULL;
    int scan_all_pmts_set = 0;

    int64_t start_time     = o->start_time;
    int64_t start_time_eof = o->start_time_eof;
    int64_t stop_time      = o->stop_time;
    int64_t recording_time = o->recording_time;

    d = demux_alloc();
    if (!d)
        return AVERROR(ENOMEM);

    f = &d->f;

    ret = sch_add_demux(sch, input_thread, d);
    if (ret < 0)
        return ret;
    d->sch = sch;

    if (stop_time != INT64_MAX && recording_time != INT64_MAX) {
        stop_time = INT64_MAX;
        av_log(d, AV_LOG_WARNING, "-t and -to cannot be used together; using -t.\n");
    }

    if (stop_time != INT64_MAX && recording_time == INT64_MAX) {
        int64_t start = start_time == AV_NOPTS_VALUE ? 0 : start_time;
        if (stop_time <= start) {
            av_log(d, AV_LOG_ERROR, "-to value smaller than -ss; aborting.\n");
            return AVERROR(EINVAL);
        } else {
            recording_time = stop_time - start;
        }
    }

    if (o->format) {
        if (!(file_iformat = av_find_input_format(o->format))) {
            av_log(d, AV_LOG_FATAL, "Unknown input format: '%s'\n", o->format);
            return AVERROR(EINVAL);
        }
    }

    if (!strcmp(filename, "-"))
        filename = "fd:";

    stdin_interaction &= strncmp(filename, "pipe:", 5) &&
                         strcmp(filename, "fd:") &&
                         strcmp(filename, "/dev/stdin");

    /* get default parameters from command line */
    ic = avformat_alloc_context();
    if (!ic)
        return AVERROR(ENOMEM);
    ic->name = av_strdup(d->log_name);
    if (o->audio_sample_rate.nb_opt) {
        av_dict_set_int(&o->g->format_opts, "sample_rate", o->audio_sample_rate.opt[o->audio_sample_rate.nb_opt - 1].u.i, 0);
    }
    if (o->audio_channels.nb_opt) {
        const AVClass *priv_class;
        if (file_iformat && (priv_class = file_iformat->priv_class) &&
            av_opt_find(&priv_class, "ch_layout", NULL, 0,
                        AV_OPT_SEARCH_FAKE_OBJ)) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%dC", o->audio_channels.opt[o->audio_channels.nb_opt - 1].u.i);
            av_dict_set(&o->g->format_opts, "ch_layout", buf, 0);
        }
    }
    if (o->audio_ch_layouts.nb_opt) {
        const AVClass *priv_class;
        if (file_iformat && (priv_class = file_iformat->priv_class) &&
            av_opt_find(&priv_class, "ch_layout", NULL, 0,
                        AV_OPT_SEARCH_FAKE_OBJ)) {
            av_dict_set(&o->g->format_opts, "ch_layout", o->audio_ch_layouts.opt[o->audio_ch_layouts.nb_opt - 1].u.str, 0);
        }
    }
    if (o->frame_rates.nb_opt) {
        const AVClass *priv_class;
        /* set the format-level framerate option;
         * this is important for video grabbers, e.g. x11 */
        if (file_iformat && (priv_class = file_iformat->priv_class) &&
            av_opt_find(&priv_class, "framerate", NULL, 0,
                        AV_OPT_SEARCH_FAKE_OBJ)) {
            av_dict_set(&o->g->format_opts, "framerate",
                        o->frame_rates.opt[o->frame_rates.nb_opt - 1].u.str, 0);
        }
    }
    if (o->frame_sizes.nb_opt) {
        av_dict_set(&o->g->format_opts, "video_size", o->frame_sizes.opt[o->frame_sizes.nb_opt - 1].u.str, 0);
    }
    if (o->frame_pix_fmts.nb_opt)
        av_dict_set(&o->g->format_opts, "pixel_format", o->frame_pix_fmts.opt[o->frame_pix_fmts.nb_opt - 1].u.str, 0);

    video_codec_name    = opt_match_per_type_str(&o->codec_names, 'v');
    audio_codec_name    = opt_match_per_type_str(&o->codec_names, 'a');
    subtitle_codec_name = opt_match_per_type_str(&o->codec_names, 's');
    data_codec_name     = opt_match_per_type_str(&o->codec_names, 'd');

    if (video_codec_name)
        ret = err_merge(ret, find_codec(NULL, video_codec_name   , AVMEDIA_TYPE_VIDEO   , 0,
                                        &ic->video_codec));
    if (audio_codec_name)
        ret = err_merge(ret, find_codec(NULL, audio_codec_name   , AVMEDIA_TYPE_AUDIO   , 0,
                                        &ic->audio_codec));
    if (subtitle_codec_name)
        ret = err_merge(ret, find_codec(NULL, subtitle_codec_name, AVMEDIA_TYPE_SUBTITLE, 0,
                                        &ic->subtitle_codec));
    if (data_codec_name)
        ret = err_merge(ret, find_codec(NULL, data_codec_name    , AVMEDIA_TYPE_DATA,     0,
                                        &ic->data_codec));
    if (ret < 0) {
        avformat_free_context(ic);
        return ret;
    }

    ic->video_codec_id     = video_codec_name    ? ic->video_codec->id    : AV_CODEC_ID_NONE;
    ic->audio_codec_id     = audio_codec_name    ? ic->audio_codec->id    : AV_CODEC_ID_NONE;
    ic->subtitle_codec_id  = subtitle_codec_name ? ic->subtitle_codec->id : AV_CODEC_ID_NONE;
    ic->data_codec_id      = data_codec_name     ? ic->data_codec->id     : AV_CODEC_ID_NONE;

    ic->flags |= AVFMT_FLAG_NONBLOCK;
    if (o->bitexact)
        ic->flags |= AVFMT_FLAG_BITEXACT;
    ic->interrupt_callback = int_cb;

    if (!av_dict_get(o->g->format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE)) {
        av_dict_set(&o->g->format_opts, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);
        scan_all_pmts_set = 1;
    }
    /* open the input file with generic avformat function */
    err = avformat_open_input(&ic, filename, file_iformat, &o->g->format_opts);
    if (err < 0) {
        if (err != AVERROR_EXIT)
            av_log(d, AV_LOG_ERROR,
                   "Error opening input: %s\n", av_err2str(err));
        if (err == AVERROR_PROTOCOL_NOT_FOUND)
            av_log(d, AV_LOG_ERROR, "Did you mean file:%s?\n", filename);
        return err;
    }
    f->ctx = ic;

    av_strlcat(d->log_name, "/",               sizeof(d->log_name));
    av_strlcat(d->log_name, ic->iformat->name, sizeof(d->log_name));
    av_freep(&ic->name);
    ic->name = av_strdup(d->log_name);

    if (scan_all_pmts_set)
        av_dict_set(&o->g->format_opts, "scan_all_pmts", NULL, AV_DICT_MATCH_CASE);
    remove_avoptions(&o->g->format_opts, o->g->codec_opts);

    ret = check_avoptions(o->g->format_opts);
    if (ret < 0)
        return ret;

    /* apply forced codec ids */
    for (int i = 0; i < ic->nb_streams; i++) {
        const AVCodec *dummy;
        ret = choose_decoder(o, f, ic, ic->streams[i], HWACCEL_NONE, AV_HWDEVICE_TYPE_NONE,
                             &dummy);
        if (ret < 0)
            return ret;
    }

    if (o->find_stream_info) {
        AVDictionary **opts;
        int orig_nb_streams = ic->nb_streams;

        ret = setup_find_stream_info_opts(ic, o->g->codec_opts, &opts);
        if (ret < 0)
            return ret;

        /* If not enough info to get the stream parameters, we decode the
           first frames to get it. (used in mpeg case for example) */
        ret = avformat_find_stream_info(ic, opts);

        for (int i = 0; i < orig_nb_streams; i++)
            av_dict_free(&opts[i]);
        av_freep(&opts);

        if (ret < 0) {
            av_log(d, AV_LOG_FATAL, "could not find codec parameters\n");
            if (ic->nb_streams == 0)
                return ret;
        }
    }

    if (start_time != AV_NOPTS_VALUE && start_time_eof != AV_NOPTS_VALUE) {
        av_log(d, AV_LOG_WARNING, "Cannot use -ss and -sseof both, using -ss\n");
        start_time_eof = AV_NOPTS_VALUE;
    }

    if (start_time_eof != AV_NOPTS_VALUE) {
        if (start_time_eof >= 0) {
            av_log(d, AV_LOG_ERROR, "-sseof value must be negative; aborting\n");
            return AVERROR(EINVAL);
        }
        if (ic->duration > 0) {
            start_time = start_time_eof + ic->duration;
            if (start_time < 0) {
                av_log(d, AV_LOG_WARNING, "-sseof value seeks to before start of file; ignored\n");
                start_time = AV_NOPTS_VALUE;
            }
        } else
            av_log(d, AV_LOG_WARNING, "Cannot use -sseof, file duration not known\n");
    }
    timestamp = (start_time == AV_NOPTS_VALUE) ? 0 : start_time;
    /* add the stream start time */
    if (!o->seek_timestamp && ic->start_time != AV_NOPTS_VALUE)
        timestamp += ic->start_time;

    /* if seeking requested, we execute it */
    if (start_time != AV_NOPTS_VALUE) {
        int64_t seek_timestamp = timestamp;

        if (!(ic->iformat->flags & AVFMT_SEEK_TO_PTS)) {
            int dts_heuristic = 0;
            for (int i = 0; i < ic->nb_streams; i++) {
                const AVCodecParameters *par = ic->streams[i]->codecpar;
                if (par->video_delay) {
                    dts_heuristic = 1;
                    break;
                }
            }
            if (dts_heuristic) {
                seek_timestamp -= 3*AV_TIME_BASE / 23;
            }
        }
        ret = avformat_seek_file(ic, -1, INT64_MIN, seek_timestamp, seek_timestamp, 0);
        if (ret < 0) {
            av_log(d, AV_LOG_WARNING, "could not seek to position %0.3f\n",
                   (double)timestamp / AV_TIME_BASE);
        }
    }

    f->start_time = start_time;
    d->recording_time = recording_time;
    f->input_sync_ref = o->input_sync_ref;
    f->input_ts_offset = o->input_ts_offset;
    f->ts_offset  = o->input_ts_offset - (copy_ts ? (start_at_zero && ic->start_time != AV_NOPTS_VALUE ? ic->start_time : 0) : timestamp);
    d->accurate_seek   = o->accurate_seek;
    d->loop = o->loop;
    d->nb_streams_warn = ic->nb_streams;

    d->duration        = (Timestamp){ .ts = 0,              .tb = (AVRational){ 1, 1 } };
    d->min_pts         = (Timestamp){ .ts = AV_NOPTS_VALUE, .tb = (AVRational){ 1, 1 } };
    d->max_pts         = (Timestamp){ .ts = AV_NOPTS_VALUE, .tb = (AVRational){ 1, 1 } };

    d->readrate = o->readrate ? o->readrate : 0.0;
    if (d->readrate < 0.0f) {
        av_log(d, AV_LOG_ERROR, "Option -readrate is %0.3f; it must be non-negative.\n", d->readrate);
        return AVERROR(EINVAL);
    }
    if (o->rate_emu) {
        if (d->readrate) {
            av_log(d, AV_LOG_WARNING, "Both -readrate and -re set. Using -readrate %0.3f.\n", d->readrate);
        } else
            d->readrate = 1.0f;
    }

    if (d->readrate) {
        d->readrate_initial_burst = o->readrate_initial_burst ? o->readrate_initial_burst : 0.5;
        if (d->readrate_initial_burst < 0.0) {
            av_log(d, AV_LOG_ERROR,
                   "Option -readrate_initial_burst is %0.3f; it must be non-negative.\n",
                   d->readrate_initial_burst);
            return AVERROR(EINVAL);
        }
        d->readrate_catchup = o->readrate_catchup ? o->readrate_catchup : d->readrate * 1.05;
        if (d->readrate_catchup < d->readrate) {
            av_log(d, AV_LOG_ERROR,
                   "Option -readrate_catchup is %0.3f; it must be at least equal to %0.3f.\n",
                   d->readrate_catchup, d->readrate);
            return AVERROR(EINVAL);
        }
    } else {
        if (o->readrate_initial_burst) {
            av_log(d, AV_LOG_WARNING, "Option -readrate_initial_burst ignored "
                   "since neither -readrate nor -re were given\n");
        }
        if (o->readrate_catchup) {
            av_log(d, AV_LOG_WARNING, "Option -readrate_catchup ignored "
                   "since neither -readrate nor -re were given\n");
        }
    }

    /* Add all the streams from the given input file to the demuxer */
    for (int i = 0; i < ic->nb_streams; i++) {
        ret = ist_add(o, d, ic->streams[i], &opts_used);
        if (ret < 0) {
            av_dict_free(&opts_used);
            return ret;
        }
    }

    /* Add all the stream groups from the given input file to the demuxer */
    for (int i = 0; i < ic->nb_stream_groups; i++) {
        ret = istg_add(o, d, ic->stream_groups[i]);
        if (ret < 0)
            return ret;
    }

    /* dump the file content */
    av_dump_format(ic, f->index, filename, 0);

    /* check if all codec options have been used */
    ret = check_avoptions_used(o->g->codec_opts, opts_used, d, 1);
    av_dict_free(&opts_used);
    if (ret < 0)
        return ret;

    for (int i = 0; i < o->dump_attachment.nb_opt; i++) {
        for (int j = 0; j < f->nb_streams; j++) {
            InputStream *ist = f->streams[j];

            if (check_stream_specifier(ic, ist->st, o->dump_attachment.opt[i].specifier) == 1) {
                ret = dump_attachment(ist, o->dump_attachment.opt[i].u.str);
                if (ret < 0)
                    return ret;
            }
        }
    }

    /* Initialize discontinuity buffer for handling interleaved packets */
    d->discont_threshold = DISCONT_THRESHOLD_US;
    d->discont_buffer_size = DISCONT_BUFFER_DEFAULT_SIZE;
    d->discont_timeout_us = DISCONT_TIMEOUT_US;

    ret = discont_buffer_init(&d->discont_buf, d->discont_buffer_size, f->nb_streams);
    if (ret < 0) {
        av_log(d, AV_LOG_ERROR, "Failed to initialize discontinuity buffer\n");
        return ret;
    }

    return 0;
}
