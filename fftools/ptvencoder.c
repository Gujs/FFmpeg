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

#define PTV_MAX_RUNG 8

/* Shared decode side of the ABR ladder (the ffmpeg model: decode the source
 * ONCE, run it through one filter graph — a -filter_complex `split`, a single
 * -filter:v chain, or none — and hand each rung its own frames via that rung's
 * frame_q). One decoder + one graph feeding N independent outputs. */
typedef struct DecodeCtx {
    AVThreadMessageQueue *video_q;            /* demux -> decode (AVPacket*) */
    AVCodecContext  *vdec;
    AVRational       ist_tb;                  /* decoder pkt time_base */
    int64_t         *h0;                      /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    int              live;
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
    AVCodecContext  *venc;
    AVStream        *ost;
    int64_t          tick_dur_us;
    int              live;
    int              is_master;      /* only the master rung prints stats/diag */
    /* shared decode counters + queue, for the master's diag line */
    AVThreadMessageQueue *dbg_video_q;
    int64_t         *dbg_dec_frames, *dbg_vcorrupt;
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

/* Build a SHARED filter_complex graph for the ABR ladder: one buffersrc fed by
 * the decoder, N buffersinks bound to the output labels the rungs map. Mirrors
 * ffmpeg's -filter_complex (decode once, split once, scale per rung) instead of
 * deinterlacing/scaling per output. labels[i] is the bare label (no brackets)
 * that rung i's `-map [label]` selects; sinks[i] receives that branch. Each sink
 * may carry a different resolution / hw_frames_ctx (read per rung afterwards). */
static int build_filter_complex(const char *graph_str, AVCodecContext *vdec,
                                AVRational tb, AVBufferRef *hw_device,
                                const char *const *labels, int n_labels,
                                AVFilterGraph **out_fg, AVFilterContext **out_src,
                                AVFilterContext **sinks)
{
    char args[256];
    AVFilterGraph        *fg    = avfilter_graph_alloc();
    AVFilterGraphSegment *seg   = NULL;
    const AVFilter       *bsrc  = avfilter_get_by_name("buffer");
    const AVFilter       *bsink = avfilter_get_by_name("buffersink");
    AVFilterInOut        *gin = NULL, *gout = NULL, *io;
    AVFilterContext      *src = NULL;
    AVRational sar = vdec->sample_aspect_ratio.num ? vdec->sample_aspect_ratio : (AVRational){1, 1};
    int ret, i;

    if (!fg || !bsrc || !bsink) { ret = AVERROR(ENOMEM); goto fail; }

    /* Build via the segment API so the hw device is assigned to every filter
     * BEFORE it is initialised. Plain `hwupload` (unlike `hwupload_cuda`)
     * hard-requires avctx->hw_device_ctx in its init(); a one-shot
     * avfilter_graph_parse2() inits filters during the parse, too early to set
     * it. Sequence: parse -> create_filters -> SET DEVICE -> apply_opts -> init
     * -> link. After link, gin = the unconnected input ([0:v]) and gout = the
     * unconnected outputs (each rung label), wired to our buffersrc / sinks. */
    if ((ret = avfilter_graph_segment_parse(fg, graph_str, 0, &seg)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex parse: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_create_filters(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex create: %s\n", av_err2str(ret)); goto fail;
    }
    if (hw_device)                                   /* must precede init() (hwupload) */
        for (unsigned k = 0; k < fg->nb_filters; k++)
            fg->filters[k]->hw_device_ctx = av_buffer_ref(hw_device);
    if ((ret = avfilter_graph_segment_apply_opts(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex opts: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_init(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex init: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_link(seg, 0, &gin, &gout)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex link: %s\n", av_err2str(ret)); goto fail;
    }

    snprintf(args, sizeof args,
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
             vdec->width, vdec->height, vdec->pix_fmt, tb.num, tb.den, sar.num, sar.den);
    if ((ret = avfilter_graph_create_filter(&src, bsrc, "in", args, NULL, fg)) < 0) goto fail;
    if (!gin || gin->next) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex needs exactly one video input ([0:v])\n");
        ret = AVERROR(EINVAL); goto fail;
    }
    if ((ret = avfilter_link(src, 0, gin->filter_ctx, gin->pad_idx)) < 0) goto fail;

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
    *out_fg = fg; *out_src = src;
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
                pthread_mutex_lock(d->h0_lock);
                if (*d->h0 == AV_NOPTS_VALUE) *d->h0 = av_rescale_q(ts, d->ist_tb, AV_TIME_BASE_Q);
                pthread_mutex_unlock(d->h0_lock);
            }
            d->dec_frames++;
            emit_video(d, frame, filt);
        }
    }
    /* flush decoder */
    avcodec_send_packet(d->vdec, NULL);
    while (avcodec_receive_frame(d->vdec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        d->dec_frames++;
        emit_video(d, frame, filt);
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
    av_thread_message_queue_set_err_recv(d->video_q, AVERROR_EOF);   /* unblock demux */
    for (i = 0; i < d->n_rung; i++)
        av_thread_message_queue_set_err_send(d->frame_q[i], AVERROR_EOF);  /* tell each output done */
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
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    AVStream             *ost[PTV_MAX_RUNG];     /* audio out stream in each muxer */
    int              n_out;
    AVCodecContext  *dec;
    AVCodecContext  *enc;
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

/* encode one AAC frame once, then fan the packet out (ref-clone) to every output
 * muxer — all rungs carry the same audio. frame=NULL flushes the encoder. */
static int audio_encode_push(AudioState *a, AVFrame *frame)
{
    int ret = avcodec_send_frame(a->enc, frame), i;
    if (ret < 0)
        return ret;
    for (;;) {
        AVPacket *pkt = av_packet_alloc();
        if (!pkt)
            return AVERROR(ENOMEM);
        ret = avcodec_receive_packet(a->enc, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); return 0; }
        if (ret < 0) { av_packet_free(&pkt); return ret; }
        for (i = 0; i < a->n_out; i++) {
            AVPacket *c = av_packet_clone(pkt);
            if (!c) continue;
            av_packet_rescale_ts(c, a->enc->time_base, a->ost[i]->time_base);
            c->stream_index = a->ost[i]->index;
            if (av_thread_message_queue_send(a->mux_q[i], &c, 0) < 0)   /* blocking */
                av_packet_free(&c);
        }
        av_packet_free(&pkt);
    }
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
        audio_encode_push(a, NULL);
    }
done:
    av_frame_free(&frame);
    av_thread_message_queue_set_err_recv(a->audio_q, AVERROR_EOF);
    { int i; for (i = 0; i < a->n_out; i++) {        /* EOF marker to each muxer */
        AVPacket *eof = NULL; av_thread_message_queue_send(a->mux_q[i], &eof, 0); } }
    return NULL;
}

/* ---- demux + mux ---- */

#define PTV_MAX_PASS 16
typedef struct PassStream {
    int        in_index;              /* input stream index being copied 1:1 */
    AVStream  *ost[PTV_MAX_RUNG];     /* output stream in each muxer (fan-out) */
    AVRational in_tb;                 /* input/output time_base (copy: identical) */
} PassStream;

typedef struct DemuxArgs {
    AVFormatContext      *ifmt;
    AVThreadMessageQueue *video_q, *audio_q;
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    int                   n_out;
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
    int pi, i;
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
    av_thread_message_queue_set_err_recv(m->mux_q, AVERROR_EOF);
    return NULL;
}

/* resolved per-output selection from an ffmpeg-style command (see resolve_plan) */
typedef struct Sel {
    int            have;                  /* 1 if -map present (explicit plan) */
    int            vstream;               /* input video stream to transcode (-1 none) */
    const AVCodec *vdec;
    const char    *venc;                  /* -c:v encoder name (NULL = default) */
    const char    *vf;                    /* -filter:v / -vf */
    const char    *vbr;                   /* -b:v */
    int            astream;               /* input audio stream to transcode (-1 none) */
    const AVCodec *adec;
    const char    *aenc;                  /* -c:a:N encoder name (NULL = aac) */
    const char    *abr;                   /* -b:a:N */
    int            copy[PTV_MAX_PASS];    /* input stream indices to passthrough (-c copy) */
    int            n_copy;
} Sel;
static const char *og_get(OptionGroup *g, const char *key);
static int resolve_plan(AVFormatContext *ifmt, OptionGroup *outg, Sel *s);

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

/* transcode: ing = parsed input group; outs = the list of output groups (one
 * per ABR rung); fcomplex = the shared -filter_complex (split). The ffmpeg model
 * — decode once, one filter graph splits to N branches, each output group is an
 * independent muxer/encoder; audio + subs/data decoded/copied once and fanned
 * out. Selection (transcode vs copy) per group comes from its -map/-c. */
static int transcode(OptionGroup *ing, OptionGroupList *outs, const char *fcomplex,
                     const char *hwdev, int mode)
{
    const char *in_url = ing->arg;
    int n_rung = outs->nb_groups;
    AVFormatContext *ifmt = NULL;
    AVCodecContext  *vdec = NULL, *adec = NULL, *aenc = NULL;
    const AVCodec   *vdecoder = NULL, *adecoder = NULL, *aencoder = NULL;
    AVStream        *vist = NULL, *aist = NULL;
    AVThreadMessageQueue *video_q = NULL, *audio_q = NULL;
    AVBufferRef     *hw_device = NULL;
    DecodeCtx        dc;
    AudioState       as;
    DemuxArgs        da;
    Rung             rung[PTV_MAX_RUNG];
    Sel              sel[PTV_MAX_RUNG];
    pthread_t        th_demux, th_decode, th_audio;
    pthread_mutex_t  h0_lock = PTHREAD_MUTEX_INITIALIZER;
    int64_t          input_h0_us = AV_NOPTS_VALUE;
    int vstream = -1, ret = 0, live, net_input, have_audio = 0, hw_cuda = 0;
    int started_audio = 0, started_decode = 0, aborted = 0, r, si;
    AVRational out_fps;
    PassStream pass[PTV_MAX_PASS]; int n_pass = 0;

    if (n_rung > PTV_MAX_RUNG) {
        av_log(NULL, AV_LOG_WARNING, "%d outputs > max %d; using the first %d\n", n_rung, PTV_MAX_RUNG, PTV_MAX_RUNG);
        n_rung = PTV_MAX_RUNG;
    }
    memset(&dc, 0, sizeof dc); memset(&as, 0, sizeof as);
    memset(&da, 0, sizeof da); memset(rung, 0, sizeof rung);

    if ((ret = avformat_open_input(&ifmt, in_url, NULL, &ing->format_opts)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open input '%s': %s\n", in_url, av_err2str(ret)); return ret;
    }
    if ((ret = avformat_find_stream_info(ifmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "stream info: %s\n", av_err2str(ret)); goto end;
    }
    av_dump_format(ifmt, 0, in_url, 0);   /* ffmpeg-style "Input #0 ... Stream ..." */

    /* resolve each output group's -map/-c into its transcode/copy selection */
    for (r = 0; r < n_rung; r++)
        if ((ret = resolve_plan(ifmt, &outs->groups[r], &sel[r])) < 0) goto end;

    /* shared video decoder. Ladder rungs map filter labels [vN]; the source
     * video comes from filter_complex's [0:v] — use the best video stream (or
     * the input video the single non-ladder output maps directly). */
    vstream = sel[0].vstream;
    if (vstream >= 0) vdecoder = sel[0].vdec;
    else              vstream  = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &vdecoder, 0);
    if (vstream < 0 || !vdecoder) { av_log(NULL, AV_LOG_ERROR, "no video stream\n"); ret = AVERROR(EINVAL); goto end; }
    vist = ifmt->streams[vstream];

    vdec = avcodec_alloc_context3(vdecoder);
    if (!vdec) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_to_context(vdec, vist->codecpar);
    vdec->pkt_timebase = vist->time_base;
    if ((ret = avcodec_open2(vdec, vdecoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "open video decoder: %s\n", av_err2str(ret)); goto end;
    }

    /* house rate: -r on the first output, else the source rate */
    {
        const char *rate_str = og_get(&outs->groups[0], "r");
        if (rate_str) {
            if (av_parse_video_rate(&out_fps, rate_str) < 0 || out_fps.num <= 0) {
                av_log(NULL, AV_LOG_ERROR, "bad -r '%s'\n", rate_str); ret = AVERROR(EINVAL); goto end;
            }
        } else {
            out_fps = vist->r_frame_rate.num ? vist->r_frame_rate
                    : vist->avg_frame_rate.num ? vist->avg_frame_rate : (AVRational){25, 1};
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

    /* shared filter graph: -filter_complex (split -> N sinks), a single -filter:v
     * chain (N==1), or none (clone the decoded frame to each rung). */
    dc.n_rung = n_rung;
    if (fcomplex) {
        const char *labels[PTV_MAX_RUNG];
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
        if ((ret = build_filter_complex(fcomplex, vdec, vist->time_base, hw_device,
                                        labels, n_rung, &dc.fg, &dc.fsrc, dc.fsink)) < 0) goto end;
        dc.filtering = 1;
    } else if (n_rung == 1 && sel[0].vf) {
        int fw = 0, fh = 0, fpix = AV_PIX_FMT_NONE; AVBufferRef *hf = NULL;
        if ((ret = build_video_filter(&dc, vdec, vist->time_base, sel[0].vf, 0, 0, 0,
                                      hw_cuda, hw_device, &fw, &fh, &fpix, &hf)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "build video filter: %s\n", av_err2str(ret)); goto end;
        }
        rung[0].fw = fw; rung[0].fh = fh; rung[0].fpix = fpix; rung[0].fhwfr = hf;
    }   /* else: dc.filtering stays 0 -> clone the decoded frame to each rung */

    /* per-rung video encoder, sized from this rung's sink (or the decoder) */
    for (r = 0; r < n_rung; r++) {
        OptionGroup *g = &outs->groups[r];
        const char *out_url = g->arg, *out_fmt = og_get(g, "f");
        const char *venc_name = sel[r].venc ? sel[r].venc : "h264_videotoolbox";
        const AVCodec *vencoder;

        if (fcomplex) {
            rung[r].fw   = av_buffersink_get_w(dc.fsink[r]);
            rung[r].fh   = av_buffersink_get_h(dc.fsink[r]);
            rung[r].fpix = av_buffersink_get_format(dc.fsink[r]);
            { AVBufferRef *hf = av_buffersink_get_hw_frames_ctx(dc.fsink[r]);
              rung[r].fhwfr = hf ? av_buffer_ref(hf) : NULL; }
        } else if (!dc.filtering) {
            rung[r].fw = vdec->width; rung[r].fh = vdec->height;
            rung[r].fpix = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
        }   /* else single -vf: rung[0].fw/fh/fpix/fhwfr already set above */

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

    /* shared audio: decode + AAC-encode ONCE (from the first rung's -map/-c:a),
     * with an output audio stream in EACH muxer; the AAC packets fan out.
     * Multichannel (AC-3 5.1) etc. ride the passthrough list below. */
    if (sel[0].astream >= 0) {
        int astream = sel[0].astream;
        aist = ifmt->streams[astream];
        adecoder = sel[0].adec;
        aencoder = avcodec_find_encoder_by_name(sel[0].aenc ? sel[0].aenc : "aac");
        if (!aencoder) aencoder = avcodec_find_encoder_by_name("aac");
        adec = avcodec_alloc_context3(adecoder);
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
                if (rung[0].ofmt->oformat->flags & AVFMT_GLOBALHEADER) aenc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                { AVDictionary *aopts = NULL; int ar2;
                  if (sel[0].abr) av_dict_set(&aopts, "b", sel[0].abr, 0);
                  ar2 = avcodec_open2(aenc, aencoder, &aopts); av_dict_free(&aopts);
                  if (ar2 < 0) av_log(NULL, AV_LOG_WARNING, "audio encoder failed; video only\n");
                  else {
                    for (r = 0; r < n_rung; r++) {           /* an audio stream in each muxer */
                        AVStream *aos = avformat_new_stream(rung[r].ofmt, NULL);
                        if (!aos) { ret = AVERROR(ENOMEM); goto end; }
                        avcodec_parameters_from_context(aos->codecpar, aenc);
                        aos->time_base = aenc->time_base;
                        as.ost[r] = aos;
                    }
                    as.out_chl = (AVChannelLayout)AV_CHANNEL_LAYOUT_STEREO;
                    swr_alloc_set_opts2(&as.swr, &as.out_chl, aenc->sample_fmt, 48000,
                                        &adec->ch_layout, adec->sample_fmt, adec->sample_rate, 0, NULL);
                    if (!as.swr || swr_init(as.swr) < 0) { av_log(NULL, AV_LOG_WARNING, "swr init failed; video only\n"); }
                    else {
                        as.fifo = av_audio_fifo_alloc(aenc->sample_fmt, 2, aenc->frame_size > 0 ? aenc->frame_size : 1024);
                        as.frame_size = aenc->frame_size > 0 ? aenc->frame_size : 1024;
                        as.out_rate = 48000; as.out_sfmt = aenc->sample_fmt; as.n_out = n_rung;
                        have_audio = 1;
                    }
                  }
                }
            }
        }
    }

    /* shared passthrough (copy): each non-transcoded input stream — extra audio
     * (AC-3 5.1), DVB subtitles, data/SCTE-35 — gets an output stream in EVERY
     * muxer; the copied packets fan out. From the first rung's plan; created
     * before the headers are written. */
    for (si = 0; si < sel[0].n_copy && n_pass < PTV_MAX_PASS; si++) {
        int sidx = sel[0].copy[si];
        AVStream *ist = ifmt->streams[sidx];
        AVDictionaryEntry *lang = av_dict_get(ist->metadata, "language", NULL, 0);
        pass[n_pass].in_index = sidx;
        pass[n_pass].in_tb    = ist->time_base;
        for (r = 0; r < n_rung; r++) {
            AVStream *os = avformat_new_stream(rung[r].ofmt, NULL);
            if (!os) { ret = AVERROR(ENOMEM); goto end; }
            if ((ret = avcodec_parameters_copy(os->codecpar, ist->codecpar)) < 0) goto end;
            os->codecpar->codec_tag = 0;
            os->time_base   = ist->time_base;
            os->disposition = ist->disposition;
            if (lang) av_dict_set(&os->metadata, "language", lang->value, 0);
            pass[n_pass].ost[r] = os;
        }
        n_pass++;
    }
    if (n_pass)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: passthrough %d stream(s) per output (copy)\n", n_pass);

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

    net_input = !strncmp(in_url, "udp://", 6) || !strncmp(in_url, "rtp://", 6) ||
                !strncmp(in_url, "srt://", 6);
    live = mode < 0 ? net_input : mode;

    /* queues: one shared video_q (+audio_q), per-rung frame_q + mux_q */
    if ((ret = av_thread_message_queue_alloc(&video_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
    av_thread_message_queue_set_free_func(video_q, free_pkt_msg);
    if (have_audio) {
        if ((ret = av_thread_message_queue_alloc(&audio_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(audio_q, free_pkt_msg);
    }
    for (r = 0; r < n_rung; r++) {
        if ((ret = av_thread_message_queue_alloc(&rung[r].frame_q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].frame_q, free_frame_msg);
        if ((ret = av_thread_message_queue_alloc(&rung[r].mux_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].mux_q, free_pkt_msg);
    }

    /* shared decode side */
    dc.video_q = video_q; dc.vdec = vdec; dc.ist_tb = vist->time_base;
    dc.h0 = &input_h0_us; dc.h0_lock = &h0_lock; dc.live = live;
    for (r = 0; r < n_rung; r++) dc.frame_q[r] = rung[r].frame_q;

    /* per-rung output side */
    for (r = 0; r < n_rung; r++) {
        VideoCtx *vc = &rung[r].vc;
        vc->frame_q = rung[r].frame_q; vc->mux_q = rung[r].mux_q; vc->venc = rung[r].venc;
        vc->out_tb = dc.filtering ? av_buffersink_get_time_base(dc.fsink[r]) : vist->time_base;
        vc->tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        vc->live = live; vc->h0 = &input_h0_us; vc->h0_lock = &h0_lock;
        vc->is_master = (r == 0);
        vc->dbg_video_q = video_q; vc->dbg_dec_frames = &dc.dec_frames; vc->dbg_vcorrupt = &dc.vcorrupt;
        rung[r].ma.ofmt = rung[r].ofmt; rung[r].ma.mux_q = rung[r].mux_q;
        rung[r].ma.n_producers = 1 + (have_audio ? 1 : 0) + (n_pass > 0 ? 1 : 0);
    }
    if (have_audio) {
        as.audio_q = audio_q; as.dec = adec; as.enc = aenc;
        as.ist_tb = aist->time_base; as.h0 = &input_h0_us; as.h0_lock = &h0_lock;
        for (r = 0; r < n_rung; r++) as.mux_q[r] = rung[r].mux_q;
    }
    da.ifmt = ifmt; da.video_q = video_q; da.audio_q = audio_q;
    da.vstream = vstream; da.astream = have_audio ? sel[0].astream : -1; da.drop = net_input;
    da.pass = pass; da.n_pass = n_pass; da.h0 = &input_h0_us; da.h0_lock = &h0_lock; da.n_out = n_rung;
    for (r = 0; r < n_rung; r++) da.mux_q[r] = rung[r].mux_q;

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: ladder %d rung(s)  house %d/%d fps (%s)  v:%s->enc  a:%s  in:%s  pull-pipeline\n",
        n_rung, out_fps.num, out_fps.den, live ? "live" : "offline", vdecoder->name,
        have_audio ? "aac" : "none", net_input ? "net(drop)" : "file(block)");
    for (r = 0; r < n_rung; r++)
        av_log(NULL, AV_LOG_INFO, "  rung%d: %dx%d -> %s [%s]\n",
               r, rung[r].fw, rung[r].fh, outs->groups[r].arg, rung[r].ofmt->oformat->name);

    /* spawn: N mux + N output + N watchdog + 1 decode + 1 audio + 1 demux */
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
    {
        int pe = pthread_create(&th_decode, NULL, decode_thread, &dc);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        started_decode = 1;
    }
    if (have_audio) {
        if (!pthread_create(&th_audio, NULL, audio_thread, &as)) started_audio = 1;
        else {
            av_log(NULL, AV_LOG_WARNING, "audio thread create failed; video only\n");
            for (r = 0; r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
        }
    }
    {
        int pe = pthread_create(&th_demux, NULL, demux_thread, &da);
        if (pe) {                                   /* couldn't start demux: drain the pipeline */
            av_thread_message_queue_set_err_send(video_q, AVERROR_EOF);
            if (have_audio) av_thread_message_queue_set_err_send(audio_q, AVERROR_EOF);
            for (r = 0; n_pass > 0 && r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
            ret = AVERROR(pe);
        } else {
            pthread_join(th_demux, NULL);
        }
    }

shutdown:
    if (aborted) {                                  /* force the pipeline to unwind */
        av_thread_message_queue_set_err_send(video_q, AVERROR_EOF);
        if (audio_q) av_thread_message_queue_set_err_send(audio_q, AVERROR_EOF);
        for (r = 0; r < n_rung; r++) {
            rung[r].vc.output_done = 1;             /* stop the watchdog */
            if (rung[r].frame_q) av_thread_message_queue_set_err_send(rung[r].frame_q, AVERROR_EOF);
            if (rung[r].mux_q)   av_thread_message_queue_set_err_recv(rung[r].mux_q, AVERROR_EOF);
        }
    }
    if (started_decode) pthread_join(th_decode, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_output) pthread_join(rung[r].th_output, NULL);
    if (started_audio) pthread_join(th_audio, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_wd) pthread_join(rung[r].th_wd, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_mux) {
        if (!ret && rung[r].ma.err < 0) ret = rung[r].ma.err;
        pthread_join(rung[r].th_mux, NULL);
    }

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: done — %d rung(s); video dec %"PRId64" master out %"PRId64
        " (dup %"PRId64" framedrop %"PRId64")  demux v:%"PRId64"/drop%"PRId64" p:%"PRId64"/drop%"PRId64"%s\n",
        n_rung, dc.dec_frames, rung[0].vc.emitted, rung[0].vc.dup, rung[0].vc.framedrop,
        da.vpkt, da.vdrop, da.ppkt, da.pdrop, have_audio ? "" : "  [no audio]");
    if (have_audio)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: audio in %"PRId64" frames, out %"PRId64" aac (demux a:%"PRId64"/drop%"PRId64")\n",
               as.in_frames, as.out_frames, da.apkt, da.adrop);
    if (ret > 0) ret = 0;

end:
    for (r = 0; r < n_rung; r++) {
        if (rung[r].hdr_written) av_write_trailer(rung[r].ofmt);
        if (rung[r].ofmt && !(rung[r].ofmt->oformat->flags & AVFMT_NOFILE) && rung[r].ofmt->pb)
            avio_closep(&rung[r].ofmt->pb);
    }
    av_thread_message_queue_free(&video_q);
    av_thread_message_queue_free(&audio_q);
    for (r = 0; r < n_rung; r++) {
        av_thread_message_queue_free(&rung[r].frame_q);
        av_thread_message_queue_free(&rung[r].mux_q);
        avcodec_free_context(&rung[r].venc);
        av_buffer_unref(&rung[r].fhwfr);
        if (rung[r].ofmt) avformat_free_context(rung[r].ofmt);
    }
    if (as.swr)  swr_free(&as.swr);
    if (as.fifo) av_audio_fifo_free(as.fifo);
    avcodec_free_context(&aenc);
    avcodec_free_context(&adec);
    avcodec_free_context(&vdec);
    avfilter_graph_free(&dc.fg);
    av_buffer_unref(&hw_device);
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
    { "nostats",          OPT_TYPE_BOOL,   0,                        { .off = 0 }, "disable stats" },
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
    char k0[8], k1[12], k2[20]; const char *best = NULL; int i, rank = -1;
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

/* strip a leading "<digits>:" file index and a trailing '?' from a -map value */
static const char *map_spec(const char *v, char *buf, size_t bufsz, int *optional)
{
    const char *colon = strchr(v, ':'), *s = v; size_t L;
    if (colon && colon > v) {
        const char *p; int alldig = 1;
        for (p = v; p < colon; p++) if (*p < '0' || *p > '9') { alldig = 0; break; }
        if (alldig) s = colon + 1;
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
 * No -map -> auto (best video + first <=2ch audio + copy the rest), back-compat. */
static int resolve_plan(AVFormatContext *ifmt, OptionGroup *outg, Sel *s)
{
    int o, si, tcnt[5] = {0}, nmap = 0;
    memset(s, 0, sizeof *s);
    s->vstream = s->astream = -1;
    s->vf = og_get(outg, "filter:v"); if (!s->vf) s->vf = og_get(outg, "vf");
    for (o = 0; o < outg->nb_opts; o++) if (!strcmp(outg->opts[o].key, "map")) nmap++;
    s->have = nmap > 0;

    if (!nmap) {                       /* no -map: auto-select (back-compat) */
        s->vstream = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &s->vdec, 0);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            AVCodecParameters *cp = ifmt->streams[si]->codecpar;
            if (cp->codec_type == AVMEDIA_TYPE_AUDIO &&
                cp->ch_layout.nb_channels > 0 && cp->ch_layout.nb_channels <= 2) { s->astream = si; break; }
        }
        if (s->astream < 0) s->astream = av_find_best_stream(ifmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        if (s->astream >= 0) s->adec = avcodec_find_decoder(ifmt->streams[s->astream]->codecpar->codec_id);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            enum AVMediaType mt = ifmt->streams[si]->codecpar->codec_type;
            if (si == s->vstream || si == s->astream) continue;
            if ((mt == AVMEDIA_TYPE_AUDIO || mt == AVMEDIA_TYPE_SUBTITLE || mt == AVMEDIA_TYPE_DATA)
                && s->n_copy < PTV_MAX_PASS) s->copy[s->n_copy++] = si;
        }
        s->venc = og_get(outg, "c:v"); if (!s->venc) s->venc = og_get(outg, "c");
        s->vbr  = og_get(outg, "b:v"); if (!s->vbr)  s->vbr  = og_get(outg, "b");
        return 0;
    }

    for (o = 0; o < outg->nb_opts; o++) {              /* explicit -map plan */
        char buf[64]; int optional; const char *spec, *mv = outg->opts[o].val;
        if (strcmp(outg->opts[o].key, "map")) continue;
        if (mv[0] == '[') {                            /* filter-output label = this rung's video */
            int idx = tcnt[0]++;                       /* video output index */
            s->venc = og_spec(outg, "c", 'v', idx);    /* -c:v / -c:v:0 (NULL -> default encoder) */
            s->vbr  = og_spec(outg, "b", 'v', idx);    /* -b:v / -b:v:0 */
            continue;                                  /* video comes from the graph, not an input stream */
        }
        spec = map_spec(mv, buf, sizeof buf, &optional);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            enum AVMediaType mt; char t; int ti, idx; const char *codec;
            if (avformat_match_stream_specifier(ifmt, ifmt->streams[si], spec) <= 0) continue;
            mt  = ifmt->streams[si]->codecpar->codec_type;
            t   = mt==AVMEDIA_TYPE_VIDEO?'v':mt==AVMEDIA_TYPE_AUDIO?'a':
                  mt==AVMEDIA_TYPE_SUBTITLE?'s':mt==AVMEDIA_TYPE_DATA?'d':'?';
            ti  = mt==AVMEDIA_TYPE_VIDEO?0:mt==AVMEDIA_TYPE_AUDIO?1:
                  mt==AVMEDIA_TYPE_SUBTITLE?2:mt==AVMEDIA_TYPE_DATA?3:4;
            idx = tcnt[ti]++;
            codec = og_spec(outg, "c", t, idx);
            if (codec && !strcmp(codec, "copy")) {
                if (s->n_copy < PTV_MAX_PASS) s->copy[s->n_copy++] = si;
            } else if (mt == AVMEDIA_TYPE_VIDEO && s->vstream < 0) {
                s->vstream = si; s->venc = codec; s->vbr = og_spec(outg, "b", t, idx);
                s->vdec = avcodec_find_decoder(ifmt->streams[si]->codecpar->codec_id);
            } else if (mt == AVMEDIA_TYPE_AUDIO && s->astream < 0) {
                s->astream = si; s->aenc = codec; s->abr = og_spec(outg, "b", t, idx);
                s->adec = avcodec_find_decoder(ifmt->streams[si]->codecpar->codec_id);
            } else if (s->n_copy < PTV_MAX_PASS) {
                s->copy[s->n_copy++] = si;             /* extra encode streams unsupported -> copy */
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
        spec = map_spec(outg->opts[o].val, buf, sizeof buf, &optional);
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

int main(int argc, char **argv)
{
    OptionParseContext octx;
    OptionGroup *ing;
    OptionGroupList *outs;
    const char *fcomplex = NULL, *hwdev = NULL;
    int mode = -1, ret, gi;

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
    }
    if (octx.groups[1].nb_groups < 1 || octx.groups[0].nb_groups < 1) {
        av_log(NULL, AV_LOG_ERROR,
               "usage: ptvencoder [opts] -i <input> [-filter_complex ..] "
               "[-map .. -c:TYPE .. -b:TYPE ..] <output> [<output> ...]\n");
        uninit_parse_context(&octx); return 1;
    }
    ing  = &octx.groups[1].groups[0];   /* first input */
    outs = &octx.groups[0];             /* all output groups (one per ABR rung) */
    ret  = transcode(ing, outs, fcomplex, hwdev, mode);
    uninit_parse_context(&octx);
    return ret < 0 ? 1 : 0;
}
