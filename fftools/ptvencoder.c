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

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
#include "libavutil/samplefmt.h"
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

#define PTV_QDEPTH      48     /* demux->decode packet queue (~1s jitter) */
#define PTV_FRAME_QDEPTH 48    /* decode->output jitter buffer (frames); holds the pre-roll cushion */
#define PTV_WD_DEADLINE_US (2 * (int64_t)AV_TIME_BASE)   /* watchdog stall threshold */

/* Diagnostics (env PTV_DIAG=1): per-second stage counters + slow-call
 * breadcrumbs to localize a stall. Temporary, gated, low-overhead (Rule 0). */
static int     g_diag;
static int64_t g_muxed;
/* ffmpeg-style progress line (frame=/fps=/bitrate=/speed=); on unless -nostats. */
static int     g_stats = 1;
static int64_t g_muxed_bytes;
/* PTV_SLOW_US: inject N us of extra per-emitted-frame consumer cost, to model a
 * slow/blocking encoder on a box that has none. Stress knob, gated. */
static int     g_slow;

void show_help_default(const char *opt, const char *arg)
{
    av_log(NULL, AV_LOG_INFO,
        "usage: ptvencoder [options] -i <input> <output>\n"
        "\n"
        "  options:\n"
        "    -i <url>      input (file or udp://...)\n"
        "    -c:v <name>   video encoder (default: h264_videotoolbox, fallback mpeg2video)\n"
        "    -r <rate>     output frame rate / house-clock rate (e.g. 25, 30000/1001); default = source\n"
        "    -f <mux>      output format (default: guessed from output; mpegts for udp://...)\n"
        "    -an           disable audio\n"
        "    -vf <chain>   raw libavfilter chain (any filters in the build, e.g. \"bwdif,scale=1280:720\")\n"
        "    -s <WxH>      scale to WxH (e.g. 1280x720)         [convenience; ignored if -vf given]\n"
        "    --deint       deinterlace (bwdif)                  [convenience; ignored if -vf given]\n"
        "    --hw cuda|cpu filter backend: cuda = bwdif_cuda/scale_cuda (-> NVENC); cpu (default)\n"
        "    --mode live|offline   live = wall-clock paced; offline = media-clock. default: auto from input\n"
        "    -version, -h\n"
        "\n"
        "  Phase 1 increment 6: video filters (deinterlace + scale, CPU or CUDA).\n");
}

/* Free function for AVThreadMessageQueue elements (AVPacket* / AVFrame*; a NULL
 * element is an end-of-stream marker on mux_q). */
static void free_pkt_msg(void *msg)   { av_packet_free(msg); }
static void free_frame_msg(void *msg) { av_frame_free(msg); }

/* Drain an encoder, pushing packets to the mux queue. frame=NULL flushes. */
static int encode_push(AVThreadMessageQueue *mux_q, AVCodecContext *enc,
                       AVStream *ost, AVFrame *frame)
{
    int ret = avcodec_send_frame(enc, frame);
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

typedef struct VideoCtx {
    /* queues */
    AVThreadMessageQueue *video_q;   /* demux  -> decode  (AVPacket*) */
    AVThreadMessageQueue *frame_q;   /* decode -> output  (AVFrame*)  */
    AVThreadMessageQueue *mux_q;     /* output -> mux     (AVPacket*) */
    /* decode side */
    AVCodecContext  *vdec;
    AVRational       ist_tb;         /* decoder pkt time_base (unfiltered frames) */
    AVRational       out_tb;         /* time_base of frames reaching the output (filter sink, or ist_tb) */
    int64_t         *h0;             /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    /* optional video filter (deinterlace / scale), decode-thread local */
    int              filtering;
    AVFilterGraph   *fg;
    AVFilterContext *fsrc, *fsink;
    /* output side */
    AVCodecContext  *venc;
    AVStream        *ost;
    int64_t          tick_dur_us;
    int              live;
    /* counters */
    int64_t          dec_frames, vcorrupt, framedrop, emitted, dup;
    /* watchdog */
    int64_t          last_emit_us;
    volatile int     output_done;
    int              stalled;
} VideoCtx;

/* decode thread: pull packets, decode, hand the latest frame to the output via
 * frame_q (drop-oldest in live so a stalled encoder never blocks decode). */
static void push_frame(VideoCtx *v, AVFrame *out)
{
    if (!v->live) {                                  /* offline: lossless back-pressure */
        if (av_thread_message_queue_send(v->frame_q, &out, 0) < 0)
            av_frame_free(&out);
        return;
    }
    int ret = av_thread_message_queue_send(v->frame_q, &out, AV_THREAD_MESSAGE_NONBLOCK);
    if (ret == AVERROR(EAGAIN)) {                    /* full -> drop oldest, keep newest */
        AVFrame *old;
        if (av_thread_message_queue_recv(v->frame_q, &old, AV_THREAD_MESSAGE_NONBLOCK) >= 0)
            av_frame_free(&old);
        if (av_thread_message_queue_send(v->frame_q, &out, AV_THREAD_MESSAGE_NONBLOCK) < 0) {
            av_frame_free(&out);
            v->framedrop++;
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
static int build_video_filter(VideoCtx *v, AVCodecContext *vdec, AVRational tb,
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
    v->fg = avfilter_graph_alloc();
    if (!v->fg) { ret = AVERROR(ENOMEM); goto end; }

    snprintf(args, sizeof(args),
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
             vdec->width, vdec->height, vdec->pix_fmt, tb.num, tb.den, sar.num, sar.den);
    if ((ret = avfilter_graph_create_filter(&v->fsrc, bsrc, "in", args, NULL, v->fg)) < 0) goto end;
    if ((ret = avfilter_graph_create_filter(&v->fsink, bsink, "out", NULL, NULL, v->fg)) < 0) goto end;

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
    outs->name = av_strdup("in");  outs->filter_ctx = v->fsrc;  outs->pad_idx = 0; outs->next = NULL;
    ins->name  = av_strdup("out"); ins->filter_ctx  = v->fsink; ins->pad_idx  = 0; ins->next  = NULL;
    if ((ret = avfilter_graph_parse_ptr(v->fg, chain, &ins, &outs, NULL)) < 0) goto end;

    if (hw_cuda && hw_device)
        for (unsigned i = 0; i < v->fg->nb_filters; i++)
            v->fg->filters[i]->hw_device_ctx = av_buffer_ref(hw_device);

    if ((ret = avfilter_graph_config(v->fg, NULL)) < 0) goto end;

    *out_w      = av_buffersink_get_w(v->fsink);
    *out_h      = av_buffersink_get_h(v->fsink);
    *out_pixfmt = av_buffersink_get_format(v->fsink);
    if (out_hwfr) {
        AVBufferRef *hf = av_buffersink_get_hw_frames_ctx(v->fsink);
        *out_hwfr = hf ? av_buffer_ref(hf) : NULL;
    }
    av_log(NULL, AV_LOG_INFO, "ptvencoder: filter [%s] -> %dx%d\n", chain, *out_w, *out_h);
    v->filtering = 1;
    ret = 0;
end:
    avfilter_inout_free(&ins);
    avfilter_inout_free(&outs);
    return ret;
}

/* Hand a decoded frame downstream: straight to the jitter buffer, or through the
 * filter graph first. Source PTS is preserved (frame->pts) so the output thread's
 * content-PTS A/V anchoring still holds across the filter. */
static void emit_video(VideoCtx *v, AVFrame *frame, AVFrame *filt)
{
    if (!v->filtering) {
        AVFrame *out = av_frame_alloc();
        if (!out) { av_frame_unref(frame); return; }
        av_frame_move_ref(out, frame);
        if (out->best_effort_timestamp != AV_NOPTS_VALUE)   /* source time in ist_tb (== out_tb) */
            out->pts = out->best_effort_timestamp;
        push_frame(v, out);
        return;
    }
    frame->pts = frame->best_effort_timestamp;   /* carry source time through the graph */
    if (av_buffersrc_add_frame(v->fsrc, frame) < 0)   /* consumes frame */
        return;
    while (av_buffersink_get_frame(v->fsink, filt) >= 0) {
        AVFrame *out = av_frame_alloc();
        if (out) { av_frame_move_ref(out, filt); push_frame(v, out); }
        else     { av_frame_unref(filt); }
    }
}

static void *decode_thread(void *arg)
{
    VideoCtx *v = arg;
    AVPacket *pkt;
    AVFrame  *frame = av_frame_alloc();
    AVFrame  *filt  = av_frame_alloc();
    int ret = 0;

    if (!frame || !filt)
        goto done;
    for (;;) {
        ret = av_thread_message_queue_recv(v->video_q, &pkt, 0);
        if (ret < 0) break;
        ret = avcodec_send_packet(v->vdec, pkt);
        av_packet_free(&pkt);
        while (ret >= 0) {
            ret = avcodec_receive_frame(v->vdec, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
            if (ret < 0) goto done;
            if (frame->flags & AV_FRAME_FLAG_CORRUPT) { v->vcorrupt++; av_frame_unref(frame); continue; }
            int64_t ts = frame->best_effort_timestamp;
            if (ts != AV_NOPTS_VALUE) {
                pthread_mutex_lock(v->h0_lock);
                if (*v->h0 == AV_NOPTS_VALUE) *v->h0 = av_rescale_q(ts, v->ist_tb, AV_TIME_BASE_Q);
                pthread_mutex_unlock(v->h0_lock);
            }
            v->dec_frames++;
            emit_video(v, frame, filt);
        }
    }
    /* flush decoder */
    avcodec_send_packet(v->vdec, NULL);
    while (avcodec_receive_frame(v->vdec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        v->dec_frames++;
        emit_video(v, frame, filt);
    }
    /* flush filter graph */
    if (v->filtering) {
        int fr = av_buffersrc_add_frame(v->fsrc, NULL); (void)fr;
        while (av_buffersink_get_frame(v->fsink, filt) >= 0) {
            AVFrame *out = av_frame_alloc();
            if (out) { av_frame_move_ref(out, filt); push_frame(v, out); }
            else     { av_frame_unref(filt); }
        }
    }
done:
    av_frame_free(&filt);
    av_frame_free(&frame);
    av_thread_message_queue_set_err_recv(v->video_q, AVERROR_EOF);  /* unblock demux */
    av_thread_message_queue_set_err_send(v->frame_q, AVERROR_EOF);  /* tell output done */
    return NULL;
}

static void *output_thread(void *arg)
{
    VideoCtx *v = arg;
    AVFrame *held = av_frame_alloc();
    AVFrame *f;
    int have = 0, ret = 0;
    int64_t tick = 0, wall0 = 0, last_vpts = -1;
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
            have = 1; fresh = 1;
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
            int64_t vpts;
            int64_t src_ts = held->pts;   /* in out_tb: decoder pkt-tb (unfiltered) or sink tb (filtered) */
            if (src_ts != AV_NOPTS_VALUE && *v->h0 != AV_NOPTS_VALUE) {
                int64_t house_us = av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q) - *v->h0;
                if (house_us < 0) house_us = 0;
                vpts = (house_us + v->tick_dur_us / 2) / v->tick_dur_us;
            } else {
                vpts = last_vpts + 1;
            }
            if (vpts <= last_vpts) vpts = last_vpts + 1;   /* monotonic CFR; dup -> next slot */
            held->pts = vpts; held->pkt_dts = AV_NOPTS_VALUE; held->duration = 0;
            last_vpts = vpts;
        }
        ret = encode_push(v->mux_q, v->venc, v->ost, held);
        v->last_emit_us = av_gettime_relative();
        tick++; v->emitted++;
        if (!fresh) v->dup++;
        if (g_slow) av_usleep(g_slow);
        if (ret < 0) break;

        if (g_diag) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                av_log(NULL, AV_LOG_INFO,
                    "[PTV-DIAG] t=%.1fs dec=%"PRId64" vcorrupt=%"PRId64" emitted=%"PRId64
                    " muxed=%"PRId64" dup=%"PRId64" framedrop=%"PRId64" vq=%d frameq=%d muxq=%d\n",
                    (nowd - diag_t0) / 1000000.0, v->dec_frames, v->vcorrupt, v->emitted,
                    g_muxed, v->dup, v->framedrop,
                    av_thread_message_queue_nb_elems(v->video_q),
                    av_thread_message_queue_nb_elems(v->frame_q),
                    av_thread_message_queue_nb_elems(v->mux_q));
                diag_last = nowd;
            }
        }

        if (g_stats) {                          /* ffmpeg-style progress line */
            int64_t nows = av_gettime_relative();
            if (nows - stat_last >= 1000000) {
                double dt    = (nows - stat_last) / 1000000.0;
                double fps   = (v->emitted - stat_prev) / (dt > 0 ? dt : 1);
                double secs  = v->emitted * v->tick_dur_us / 1000000.0;   /* CFR output time */
                double wall  = (nows - wall0) / 1000000.0;
                double speed = wall > 0 ? secs / wall : 0;
                double kbps  = secs > 0 ? g_muxed_bytes * 8.0 / secs / 1000.0 : 0;
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                av_log(NULL, AV_LOG_INFO,
                    "frame=%6"PRId64" fps=%3.0f size=%8"PRId64"KiB time=%02d:%02d:%05.2f "
                    "bitrate=%7.1fkbits/s dup=%"PRId64" drop=%"PRId64" speed=%4.2fx\n",
                    v->emitted, fps, g_muxed_bytes / 1024, hh, mm, ss, kbps,
                    v->dup, v->framedrop, speed);
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
    AVThreadMessageQueue *mux_q;
    AVCodecContext  *dec;
    AVCodecContext  *enc;
    AVStream        *ost;
    AVRational       ist_tb;
    SwrContext      *swr;
    AVAudioFifo     *fifo;
    int              frame_size;
    int              out_rate;
    enum AVSampleFormat out_sfmt;
    AVChannelLayout  out_chl;
    int64_t         *h0;
    pthread_mutex_t *h0_lock;
    int              pts_set;
    int64_t          next_pts;
    int64_t          in_frames, out_frames;
} AudioState;

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
        ret = encode_push(a->mux_q, a->enc, a->ost, f);
        a->out_frames++;
        av_frame_free(&f);
        if (ret < 0) return ret;
    }
    return 0;
}

static int audio_push(AudioState *a, AVFrame *frame)
{
    uint8_t **out = NULL;
    int out_max, got, ret = 0;
    int64_t ts = frame->best_effort_timestamp;

    a->in_frames++;

    /* Audio anchors to the FIRST VIDEO frame (h0, set only by the video decode
     * thread) so A/V share one origin. Audio that arrives before video has
     * anchored — or that precedes the anchor — is DROPPED, otherwise audio leads
     * video by the video start-up (IDR-acquire) delay (the ~1-2s A/V offset). */
    if (!a->pts_set) {
        int64_t h0, house_us;
        if (ts == AV_NOPTS_VALUE)
            return 0;
        pthread_mutex_lock(a->h0_lock);
        h0 = *a->h0;
        pthread_mutex_unlock(a->h0_lock);
        if (h0 == AV_NOPTS_VALUE)
            return 0;                          /* video not anchored yet: drop */
        house_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q) - h0;
        if (house_us < 0)
            return 0;                          /* audio precedes video anchor: drop */
        a->next_pts = av_rescale(house_us, a->out_rate, 1000000);
        a->pts_set  = 1;
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
    /* flush decoder -> resampler -> encoder */
    avcodec_send_packet(a->dec, NULL);
    while (avcodec_receive_frame(a->dec, frame) >= 0) { audio_push(a, frame); av_frame_unref(frame); }
    {
        uint8_t **out = NULL; int got, out_max = 4096;
        if (av_samples_alloc_array_and_samples(&out, NULL, a->out_chl.nb_channels,
                                               out_max, a->out_sfmt, 0) >= 0) {
            while ((got = swr_convert(a->swr, out, out_max, NULL, 0)) > 0)
                av_audio_fifo_write(a->fifo, (void **)out, got);
            av_freep(&out[0]); av_freep(&out);
        }
        audio_drain_fifo(a);
        encode_push(a->mux_q, a->enc, a->ost, NULL);
    }
done:
    av_frame_free(&frame);
    av_thread_message_queue_set_err_recv(a->audio_q, AVERROR_EOF);
    { AVPacket *eof = NULL; av_thread_message_queue_send(a->mux_q, &eof, 0); }
    return NULL;
}

/* ---- demux + mux ---- */

#define PTV_MAX_PASS 16
typedef struct PassStream {
    int        in_index;    /* input stream index being copied 1:1 */
    AVStream  *ost;         /* output stream */
    AVRational in_tb;       /* input/output time_base (copy: identical) */
} PassStream;

typedef struct DemuxArgs {
    AVFormatContext      *ifmt;
    AVThreadMessageQueue *video_q, *audio_q, *mux_q;
    int                   vstream, astream;
    int                   drop;          /* non-blocking + drop on full (network input) */
    PassStream           *pass;          /* copy-passthrough: extra audio, subs, data */
    int                   n_pass;
    int64_t              *h0;             /* house origin (us); copy ts rebased onto it */
    pthread_mutex_t      *h0_lock;
    int64_t               vpkt, apkt, ppkt, vdrop, adrop, pdrop;
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
    int pi;
    for (pi = 0; pi < d->n_pass; pi++) {
        int64_t h0, h0_tb, ref;
        if (out->stream_index != d->pass[pi].in_index)
            continue;
        pthread_mutex_lock(d->h0_lock); h0 = *d->h0; pthread_mutex_unlock(d->h0_lock);
        if (h0 == AV_NOPTS_VALUE) { av_packet_free(&out); return 0; }  /* video not anchored yet */
        h0_tb = av_rescale_q(h0, AV_TIME_BASE_Q, d->pass[pi].in_tb);
        if (out->pts != AV_NOPTS_VALUE) out->pts -= h0_tb;
        if (out->dts != AV_NOPTS_VALUE) out->dts -= h0_tb;
        ref = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
        if (ref != AV_NOPTS_VALUE && ref < 0) { av_packet_free(&out); return 0; }  /* precedes anchor */
        out->stream_index = d->pass[pi].ost->index;
        out->pos = -1;
        d->ppkt++;
        return demux_send(d->mux_q, out, d->drop, &d->pdrop);
    }
    av_packet_free(&out);
    return 0;
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
                av_log(NULL, AV_LOG_INFO, "[PTV-DIAG] demux vpkt=%"PRId64" vdrop=%"PRId64
                       " apkt=%"PRId64" adrop=%"PRId64" ppkt=%"PRId64" pdrop=%"PRId64"\n",
                       d->vpkt, d->vdrop, d->apkt, d->adrop, d->ppkt, d->pdrop);
                diag_last = now;
            }
        }
        AVPacket *out = av_packet_alloc();
        if (!out) { av_packet_unref(pkt); break; }
        av_packet_move_ref(out, pkt);
        if (out->stream_index == d->vstream) {
            d->vpkt++;
            ret = demux_send(d->video_q, out, d->drop, &d->vdrop);
        } else if (d->astream >= 0 && out->stream_index == d->astream) {
            d->apkt++;
            ret = demux_send(d->audio_q, out, d->drop, &d->adrop);
        } else {
            ret = demux_pass(d, out);       /* copy-passthrough, or free if unmapped */
        }
        if (ret < 0)
            break;
    }
end:
    av_thread_message_queue_set_err_send(d->video_q, AVERROR_EOF);
    if (d->astream >= 0)
        av_thread_message_queue_set_err_send(d->audio_q, AVERROR_EOF);
    if (d->n_pass > 0) {                     /* copy-passthrough producer EOF */
        AVPacket *eof = NULL;
        av_thread_message_queue_send(d->mux_q, &eof, 0);
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
    av_thread_message_queue_set_err_recv(m->mux_q, AVERROR_EOF);
    return NULL;
}

static int transcode(const char *in_url, const char *out_url, const char *out_fmt,
                     const char *venc_name, const char *rate_str, int mode, int want_audio,
                     const char *vf, int do_deint, int scale_w, int scale_h, int hw_cuda)
{
    AVFormatContext *ifmt = NULL, *ofmt = NULL;
    AVCodecContext  *vdec = NULL, *venc = NULL, *adec = NULL, *aenc = NULL;
    const AVCodec   *vdecoder = NULL, *vencoder = NULL, *adecoder = NULL, *aencoder = NULL;
    AVStream        *vist = NULL, *aist = NULL;
    AVThreadMessageQueue *video_q = NULL, *audio_q = NULL, *frame_q = NULL, *mux_q = NULL;
    AVBufferRef     *hw_device = NULL, *fhwfr = NULL;
    int              fw = 0, fh = 0, fpix = AV_PIX_FMT_NONE;
    VideoCtx         vc;
    AudioState       as;
    DemuxArgs        da; MuxArgs ma;
    pthread_t        th_demux, th_decode, th_output, th_audio, th_mux, th_wd;
    pthread_mutex_t  h0_lock = PTHREAD_MUTEX_INITIALIZER;
    int64_t          input_h0_us = AV_NOPTS_VALUE;
    int vstream = -1, astream = -1, ret = 0, live, net_input, have_audio = 0;
    int started_audio = 0, hdr_written = 0;
    AVRational out_fps;
    PassStream pass[PTV_MAX_PASS]; int n_pass = 0, si;

    memset(&vc, 0, sizeof(vc)); memset(&as, 0, sizeof(as));
    memset(&da, 0, sizeof(da)); memset(&ma, 0, sizeof(ma));

    if ((ret = avformat_open_input(&ifmt, in_url, NULL, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open input '%s': %s\n", in_url, av_err2str(ret)); return ret;
    }
    if ((ret = avformat_find_stream_info(ifmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "stream info: %s\n", av_err2str(ret)); goto end;
    }
    av_dump_format(ifmt, 0, in_url, 0);   /* ffmpeg-style "Input #0 ... Stream ..." */

    vstream = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &vdecoder, 0);
    if (vstream < 0) { av_log(NULL, AV_LOG_ERROR, "no video stream\n"); ret = vstream; goto end; }
    vist = ifmt->streams[vstream];

    vdec = avcodec_alloc_context3(vdecoder);
    if (!vdec) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_to_context(vdec, vist->codecpar);
    vdec->pkt_timebase = vist->time_base;
    if ((ret = avcodec_open2(vdec, vdecoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "open video decoder: %s\n", av_err2str(ret)); goto end;
    }

    ret = avformat_alloc_output_context2(&ofmt, NULL, out_fmt, out_url);
    if (ret < 0 && !out_fmt)   /* protocol URLs (udp://, srt://) have no extension to guess from */
        ret = avformat_alloc_output_context2(&ofmt, NULL, "mpegts", out_url);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot create output context for '%s': %s (try -f mpegts)\n",
               out_url, av_err2str(ret));
        goto end;
    }

    vencoder = avcodec_find_encoder_by_name(venc_name);
    if (!vencoder) {
        av_log(NULL, AV_LOG_WARNING, "encoder '%s' not found, using mpeg2video\n", venc_name);
        vencoder = avcodec_find_encoder_by_name("mpeg2video");
    }
    if (!vencoder) { ret = AVERROR_ENCODER_NOT_FOUND; goto end; }

    if (rate_str) {
        if (av_parse_video_rate(&out_fps, rate_str) < 0 || out_fps.num <= 0) {
            av_log(NULL, AV_LOG_ERROR, "bad -r '%s'\n", rate_str); ret = AVERROR(EINVAL); goto end;
        }
    } else {
        out_fps = vist->r_frame_rate.num ? vist->r_frame_rate
                : vist->avg_frame_rate.num ? vist->avg_frame_rate : (AVRational){25, 1};
    }

    /* optional video filter: raw -vf chain, or convenience deinterlace/scale (CPU or CUDA) */
    if (vf || do_deint || scale_w > 0) {
        if (hw_cuda &&
            (ret = av_hwdevice_ctx_create(&hw_device, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "cannot create CUDA device (--hw cuda): %s\n", av_err2str(ret)); goto end;
        }
        if ((ret = build_video_filter(&vc, vdec, vist->time_base, vf, do_deint, scale_w, scale_h,
                                      hw_cuda, hw_device, &fw, &fh, &fpix, &fhwfr)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "build video filter: %s\n", av_err2str(ret)); goto end;
        }
    }

    venc = avcodec_alloc_context3(vencoder);
    if (!venc) { ret = AVERROR(ENOMEM); goto end; }
    if (vc.filtering) {
        venc->width = fw; venc->height = fh; venc->pix_fmt = fpix;
        if (fhwfr) venc->hw_frames_ctx = av_buffer_ref(fhwfr);   /* CUDA frames -> NVENC */
    } else {
        venc->width = vdec->width; venc->height = vdec->height;
        venc->pix_fmt = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
    }
    venc->time_base = av_inv_q(out_fps); venc->framerate = out_fps;
    venc->bit_rate = 3000000; venc->gop_size = 2 * (out_fps.num / FFMAX(out_fps.den, 1));
    if (ofmt->oformat->flags & AVFMT_GLOBALHEADER) venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if ((ret = avcodec_open2(venc, vencoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "open video encoder '%s': %s\n", vencoder->name, av_err2str(ret)); goto end;
    }
    vc.ost = avformat_new_stream(ofmt, NULL);
    if (!vc.ost) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_from_context(vc.ost->codecpar, venc);
    vc.ost->time_base = venc->time_base;

    /* audio (optional). Prefer a <=2ch (stereo) source for the AAC transcode; any
     * multichannel stream (e.g. AC-3 5.1) is preserved untouched via passthrough,
     * matching the broadcast ladder convention. */
    if (want_audio) {
        astream = av_find_best_stream(ifmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            AVCodecParameters *cp = ifmt->streams[si]->codecpar;
            if (cp->codec_type == AVMEDIA_TYPE_AUDIO &&
                cp->ch_layout.nb_channels > 0 && cp->ch_layout.nb_channels <= 2) {
                astream = si; break;
            }
        }
        if (astream >= 0)
            adecoder = avcodec_find_decoder(ifmt->streams[astream]->codecpar->codec_id);
    }
    if (astream >= 0) {
        aist = ifmt->streams[astream];
        adec = avcodec_alloc_context3(adecoder);
        aencoder = avcodec_find_encoder_by_name("aac");
        if (adec && aencoder) {
            avcodec_parameters_to_context(adec, aist->codecpar);
            adec->pkt_timebase = aist->time_base;
            if (avcodec_open2(adec, adecoder, NULL) < 0) { av_log(NULL, AV_LOG_WARNING, "audio decoder failed; video only\n"); }
            else {
                aenc = avcodec_alloc_context3(aencoder);
                aenc->sample_rate = 48000;
                aenc->ch_layout   = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO;
                aenc->sample_fmt  = aencoder->sample_fmts ? aencoder->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
                aenc->bit_rate    = 160000;
                aenc->time_base   = (AVRational){1, 48000};
                if (ofmt->oformat->flags & AVFMT_GLOBALHEADER) aenc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                if (avcodec_open2(aenc, aencoder, NULL) < 0) { av_log(NULL, AV_LOG_WARNING, "audio encoder failed; video only\n"); }
                else {
                    as.ost = avformat_new_stream(ofmt, NULL);
                    avcodec_parameters_from_context(as.ost->codecpar, aenc);
                    as.ost->time_base = aenc->time_base;
                    as.out_chl = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO;
                    swr_alloc_set_opts2(&as.swr, &as.out_chl, aenc->sample_fmt, 48000,
                                        &adec->ch_layout, adec->sample_fmt, adec->sample_rate, 0, NULL);
                    if (!as.swr || swr_init(as.swr) < 0) { av_log(NULL, AV_LOG_WARNING, "swr init failed; video only\n"); }
                    else {
                        as.fifo = av_audio_fifo_alloc(aenc->sample_fmt, 2, aenc->frame_size > 0 ? aenc->frame_size : 1024);
                        as.frame_size = aenc->frame_size > 0 ? aenc->frame_size : 1024;
                        as.out_rate = 48000; as.out_sfmt = aenc->sample_fmt;
                        have_audio = 1;
                    }
                }
            }
        }
    }

    /* passthrough (copy): every input stream we don't transcode — extra audio
     * (e.g. AC-3 5.1), DVB subtitles, data/SCTE-35 — mapped 1:1 to the output.
     * Must be created before the header is written. */
    for (si = 0; si < (int)ifmt->nb_streams && n_pass < PTV_MAX_PASS; si++) {
        AVStream *ist = ifmt->streams[si];
        enum AVMediaType mt = ist->codecpar->codec_type;
        AVStream *os;
        AVDictionaryEntry *lang;
        if (si == vstream || (have_audio && si == astream))
            continue;
        if (mt != AVMEDIA_TYPE_AUDIO && mt != AVMEDIA_TYPE_SUBTITLE && mt != AVMEDIA_TYPE_DATA)
            continue;
        os = avformat_new_stream(ofmt, NULL);
        if (!os) { ret = AVERROR(ENOMEM); goto end; }
        if ((ret = avcodec_parameters_copy(os->codecpar, ist->codecpar)) < 0) goto end;
        os->codecpar->codec_tag = 0;
        os->time_base   = ist->time_base;
        os->disposition = ist->disposition;
        lang = av_dict_get(ist->metadata, "language", NULL, 0);
        if (lang) av_dict_set(&os->metadata, "language", lang->value, 0);
        pass[n_pass].in_index = si;
        pass[n_pass].ost      = os;
        pass[n_pass].in_tb    = ist->time_base;
        n_pass++;
    }
    if (n_pass)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: passthrough %d stream(s) (copy)\n", n_pass);

    if (!(ofmt->oformat->flags & AVFMT_NOFILE))
        if ((ret = avio_open(&ofmt->pb, out_url, AVIO_FLAG_WRITE)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "open output '%s': %s\n", out_url, av_err2str(ret)); goto end;
        }
    if ((ret = avformat_write_header(ofmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "write header: %s\n", av_err2str(ret)); goto end;
    }
    hdr_written = 1;
    av_dump_format(ofmt, 0, out_url, 1);  /* ffmpeg-style "Output #0 ... Stream ..." */

    net_input = !strncmp(in_url, "udp://", 6) || !strncmp(in_url, "rtp://", 6) ||
                !strncmp(in_url, "srt://", 6);
    live = mode < 0 ? net_input : mode;

    if ((ret = av_thread_message_queue_alloc(&video_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
    av_thread_message_queue_set_free_func(video_q, free_pkt_msg);
    if ((ret = av_thread_message_queue_alloc(&frame_q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
    av_thread_message_queue_set_free_func(frame_q, free_frame_msg);
    if ((ret = av_thread_message_queue_alloc(&mux_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
    av_thread_message_queue_set_free_func(mux_q, free_pkt_msg);
    if (have_audio) {
        if ((ret = av_thread_message_queue_alloc(&audio_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(audio_q, free_pkt_msg);
    }

    vc.video_q = video_q; vc.frame_q = frame_q; vc.mux_q = mux_q;
    vc.vdec = vdec; vc.venc = venc; vc.ist_tb = vist->time_base;
    vc.out_tb = vc.filtering ? av_buffersink_get_time_base(vc.fsink) : vist->time_base;
    vc.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
    vc.live = live; vc.h0 = &input_h0_us; vc.h0_lock = &h0_lock;
    if (have_audio) {
        as.audio_q = audio_q; as.mux_q = mux_q; as.dec = adec; as.enc = aenc;
        as.ist_tb = aist->time_base; as.h0 = &input_h0_us; as.h0_lock = &h0_lock;
    }

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %dx%d  house %d/%d fps (%s)  v:%s->%s  a:%s  [%s]  in:%s  pull-pipeline\n",
        venc->width, venc->height, out_fps.num, out_fps.den, live ? "live" : "offline",
        vdecoder->name, vencoder->name, have_audio ? "aac" : "none", ofmt->oformat->name,
        net_input ? "net(drop)" : "file(block)");

    da.ifmt = ifmt; da.video_q = video_q; da.audio_q = audio_q; da.mux_q = mux_q;
    da.vstream = vstream; da.astream = have_audio ? astream : -1; da.drop = net_input;
    da.pass = pass; da.n_pass = n_pass; da.h0 = &input_h0_us; da.h0_lock = &h0_lock;
    ma.ofmt = ofmt; ma.mux_q = mux_q; ma.n_producers = (have_audio ? 2 : 1) + (n_pass > 0 ? 1 : 0);

    if ((ret = pthread_create(&th_mux, NULL, mux_thread, &ma))) { ret = AVERROR(ret); goto end; }
    if ((ret = pthread_create(&th_output, NULL, output_thread, &vc))) {
        av_thread_message_queue_set_err_recv(mux_q, AVERROR_EOF); pthread_join(th_mux, NULL);
        ret = AVERROR(ret); goto end;
    }
    if ((ret = pthread_create(&th_decode, NULL, decode_thread, &vc))) {
        av_thread_message_queue_set_err_send(frame_q, AVERROR_EOF);
        pthread_join(th_output, NULL); pthread_join(th_mux, NULL);
        ret = AVERROR(ret); goto end;
    }
    pthread_create(&th_wd, NULL, watchdog_thread, &vc);
    if (have_audio && !pthread_create(&th_audio, NULL, audio_thread, &as))
        started_audio = 1;
    else if (have_audio)
        av_log(NULL, AV_LOG_WARNING, "audio thread create failed; video only\n");

    if ((ret = pthread_create(&th_demux, NULL, demux_thread, &da))) {
        av_thread_message_queue_set_err_send(video_q, AVERROR_EOF);
        if (have_audio) av_thread_message_queue_set_err_send(audio_q, AVERROR_EOF);
        ret = AVERROR(ret);
    } else {
        pthread_join(th_demux, NULL);
    }

    pthread_join(th_decode, NULL);
    pthread_join(th_output, NULL);
    if (started_audio) pthread_join(th_audio, NULL);
    pthread_join(th_wd, NULL);
    pthread_join(th_mux, NULL);

    if (!ret && ma.err < 0) ret = ma.err;

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: done — video dec %"PRId64" out %"PRId64" (dup %"PRId64" framedrop %"PRId64") "
        "demux v:%"PRId64"/drop%"PRId64"%s\n",
        vc.dec_frames, vc.emitted, vc.dup, vc.framedrop, da.vpkt, da.vdrop,
        have_audio ? "" : "  [no audio]");
    if (have_audio)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: audio in %"PRId64" frames, out %"PRId64" aac frames (demux a:%"PRId64"/drop%"PRId64")\n",
               as.in_frames, as.out_frames, da.apkt, da.adrop);

    if (hdr_written)
        av_write_trailer(ofmt);
    if (ret > 0) ret = 0;

end:
    if (ofmt && !(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb)
        avio_closep(&ofmt->pb);
    av_thread_message_queue_free(&video_q);
    av_thread_message_queue_free(&frame_q);
    av_thread_message_queue_free(&audio_q);
    av_thread_message_queue_free(&mux_q);
    if (as.swr)  swr_free(&as.swr);
    if (as.fifo) av_audio_fifo_free(as.fifo);
    avcodec_free_context(&aenc);
    avcodec_free_context(&adec);
    avcodec_free_context(&venc);
    avcodec_free_context(&vdec);
    avfilter_graph_free(&vc.fg);
    av_buffer_unref(&fhwfr);
    av_buffer_unref(&hw_device);
    if (ofmt) avformat_free_context(ofmt);
    avformat_close_input(&ifmt);
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
    { "stats_period",     OPT_TYPE_STRING, 0,                        { .off = 0 }, "stats period", "t" },
    { "y",                OPT_TYPE_BOOL,   0,                        { .off = 0 }, "overwrite output" },
    { "n",                OPT_TYPE_BOOL,   0,                        { .off = 0 }, "never overwrite" },
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

int main(int argc, char **argv)
{
    const char *in_url = NULL, *out_url = NULL, *rate = NULL, *out_fmt = NULL;
    const char *venc = "h264_videotoolbox";
    const char *vfilter = NULL;
    int mode = -1, want_audio = 1, i;
    int do_deint = 0, scale_w = 0, scale_h = 0, hw_cuda = 0;

    init_dynload();
    av_log_set_level(AV_LOG_INFO);
    g_diag = !!getenv("PTV_DIAG");
    { const char *s = getenv("PTV_SLOW_US"); g_slow = s ? atoi(s) : 0; }

    if (argc >= 2 && (!strcmp(argv[1], "-version") || !strcmp(argv[1], "--version"))) {
        printf("ptvencoder (PoC) — FFmpeg %s\n", av_version_info());
        printf("  libavformat   %u.%u.%u\n", AV_VERSION_MAJOR(avformat_version()),
               AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
        printf("  libavcodec    %u.%u.%u\n", AV_VERSION_MAJOR(avcodec_version()),
               AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
        printf("  libavutil     %u.%u.%u\n", AV_VERSION_MAJOR(avutil_version()),
               AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
        return 0;
    }
    if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        show_help_default(NULL, NULL); return 0;
    }
    if (getenv("PTV_PARSE_DEBUG"))   /* validate the ffmpeg-style parser, then exit */
        return ptv_parse_and_print(argc, argv) < 0 ? 1 : 0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc)        in_url = argv[++i];
        else if (!strcmp(argv[i], "-c:v") && i + 1 < argc) venc   = argv[++i];
        else if (!strcmp(argv[i], "-r") && i + 1 < argc)   rate   = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc)   out_fmt = argv[++i];
        else if (!strcmp(argv[i], "-an"))                  want_audio = 0;
        else if (!strcmp(argv[i], "-nostats"))             g_stats = 0;
        else if (!strcmp(argv[i], "-stats"))               g_stats = 1;
        else if ((!strcmp(argv[i], "-vf") || !strcmp(argv[i], "-filter:v")) && i + 1 < argc) vfilter = argv[++i];
        else if (!strcmp(argv[i], "--deint"))              do_deint = 1;
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &scale_w, &scale_h) != 2 || scale_w <= 0 || scale_h <= 0) {
                av_log(NULL, AV_LOG_ERROR, "-s expects WxH (e.g. 1280x720)\n"); return 1;
            }
        }
        else if (!strcmp(argv[i], "--hw") && i + 1 < argc) {
            const char *h = argv[++i];
            if (!strcmp(h, "cuda")) hw_cuda = 1;
            else if (!strcmp(h, "none") || !strcmp(h, "cpu")) hw_cuda = 0;
            else { av_log(NULL, AV_LOG_ERROR, "--hw must be cuda|cpu\n"); return 1; }
        }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "live")) mode = 1;
            else if (!strcmp(m, "offline")) mode = 0;
            else { av_log(NULL, AV_LOG_ERROR, "--mode must be live|offline\n"); return 1; }
        }
        else if (argv[i][0] != '-') out_url = argv[i];
        else { av_log(NULL, AV_LOG_ERROR, "unknown option '%s'\n", argv[i]); return 1; }
    }

    if (!in_url || !out_url) { show_help_default(NULL, NULL); return 1; }
    return transcode(in_url, out_url, out_fmt, venc, rate, mode, want_audio,
                     vfilter, do_deint, scale_w, scale_h, hw_cuda) < 0 ? 1 : 0;
}
