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
#include <time.h>

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
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

const char program_name[] = "ptvencoder";
const int  program_birth_year = 2026;

/* ptvencoder's OWN version, independent of the FFmpeg git-describe string (which,
 * on the BtbN box build, reflects fresh-upstream + `git apply` and so does NOT
 * encode which patch revision is applied). Bump by hand on each meaningful fix/
 * feature so a deployed binary self-identifies via the banner / -version. */
#define PTVENCODER_VERSION "0.4.1"   /* 0.4.1: multiview per-slot audio skew = dup-event counter (was arithmetic+non-decreasing, which locked startup jitter -> later-priming slots' audio over-delayed); per-slot skew in PTV_DIAG. 0.4.0: MULTIVIEW (1/2/4-input mosaic — house-clock compositor, per-input jitter buffer + clock, per-slot audio/sub, parallel open); 0.3.0: multiple transcoded audio tracks + per-track -ac/-filter:a/-metadata + source fan-out; 0.2.3: monotonic-DTS clamp on copy path; 0.2.2: no -r preserves source FRAME rate (avg); 0.2.1: 33-bit PTS-wrap on copy-passthrough */

#define PTV_QDEPTH      48     /* demux->decode packet queue (~1s jitter) */
#define PTV_FRAME_QDEPTH 48    /* decode->output jitter buffer (frames); holds the pre-roll cushion */
#define PTV_WD_DEADLINE_US (2 * (int64_t)AV_TIME_BASE)   /* watchdog stall threshold */

/* Diagnostics (env PTV_DIAG=1): per-second stage counters + slow-call
 * breadcrumbs to localize a stall. Temporary, gated, low-overhead (Rule 0). */
static int     g_diag;
/* A/V common-mode lock: the video frame-synchronizer's dup/drop makes the house
 * clock run ahead of source content; that skew is published by the master output
 * thread and added to the audio resampler's target so audio rides the SAME house
 * clock as video (else audio stays source-locked and drifts ~40ms per video dup).
 * On by default; PTV_NO_AVLOCK=1 reverts to the old source-locked audio. */
static int     g_avlock = 1;
static int64_t g_muxed;
/* ffmpeg-style progress line (frame=/fps=/bitrate=/speed=); on unless -nostats. */
static int     g_stats = 1;
static int64_t g_muxed_bytes;
/* -stats_period: interval (us) between progress lines. Default 1s; raise for
 * production (e.g. -stats_period 10 -> every 10s) to keep logs quiet. */
static int64_t g_stats_period_us = 1000000;
/* PTV_SLOW_US: inject N us of extra per-emitted-frame consumer cost, to model a
 * slow/blocking encoder on a box that has none. Stress knob, gated. */
static int     g_slow;

/* PTV_LOG_TS=1: prefix every log line with a local wall-clock timestamp
 * [YYYY-MM-DD HH:MM:SS.mmm], so production logs are self-dated natively
 * (replaces piping through `ts`). Wraps libav's line formatter; serialized so
 * lines from the demux/decode/encode/mux threads don't interleave. */
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;
static void ptv_log_ts_callback(void *avcl, int level, const char *fmt, va_list vl)
{
    static int print_prefix = 1, at_line_start = 1;
    char buf[2048];
    int n, start;

    if (level > av_log_get_level())
        return;
    pthread_mutex_lock(&g_log_mtx);
    av_log_format_line2(avcl, level, fmt, vl, buf, sizeof buf, &print_prefix);
    n = (int)strlen(buf);
    for (start = 0; start < n; ) {
        const char *nl = memchr(buf + start, '\n', n - start);
        int end = nl ? (int)(nl - buf) + 1 : n;
        if (at_line_start) {
            int64_t now = av_gettime();
            time_t s = (time_t)(now / 1000000);
            struct tm tm; char d[24];
            localtime_r(&s, &tm);
            strftime(d, sizeof d, "%Y-%m-%d %H:%M:%S", &tm);
            fprintf(stderr, "[%s.%03d] ", d, (int)((now % 1000000) / 1000));
            at_line_start = 0;
        }
        fwrite(buf + start, 1, (size_t)(end - start), stderr);
        if (nl) at_line_start = 1;
        start = end;
    }
    pthread_mutex_unlock(&g_log_mtx);
}

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
    int ret;
    /* Let the ENCODER choose the GOP: clear the decoder's leftover I/P/B
     * classification. Otherwise mpeg2video (and any pict_type-honoring encoder)
     * tries to replicate the source's frame types — h264's long B-runs trip
     * "too many B-frames in a row" and stall; NVENC's forced-IDR GOP can misalign. */
    if (frame)
        frame->pict_type = AV_PICTURE_TYPE_NONE;
    ret = avcodec_send_frame(enc, frame);
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
#define PTV_MAX_AUDIO 8    /* max transcoded audio output tracks (multi-language, multiview slots) */
#define PTV_MAX_INPUT 4    /* max composited inputs (multiview): 1 / 2 / 4 */
#define PTV_MV_SKEW_CAP_US 250000   /* multiview per-slot audio skew cap (async budget) */

/* Multiview per-input jitter buffer: decode pushes each frame onto `q`; the
 * compositor pops ONE per house tick (absorbing bursty decode delivery, exactly
 * like the single-input frame_q) and dup-holds its last frame when `q` underruns,
 * so a late/dead slot never stalls the mosaic. A depth-1 "latest only" hold
 * instead would discard intra-burst frames -> massive dup/judder. */
typedef struct VideoHold {
    AVThreadMessageQueue *q;   /* decode -> compositor (AVFrame*) */
    int64_t         framedrop; /* drop-oldest count when q overflows (live) */
    pthread_mutex_t lock;      /* guards wall_us + eof */
    int64_t         wall_us;   /* when a frame was last pushed (staleness -> slate) */
    int             eof;       /* decode terminated for this input (terminal) */
} VideoHold;

/* Shared decode side of the ABR ladder (the ffmpeg model: decode the source
 * ONCE, run it through one filter graph — a -filter_complex `split`, a single
 * -filter:v chain, or none — and hand each rung its own frames via that rung's
 * frame_q). One decoder + one graph feeding N independent outputs.
 *
 * Multiview: `hold` is set and `filtering` is 0 — the decode thread stages each
 * frame into the per-input hold instead of running the graph; the compositor
 * owns the (N-input) graph and the frame_q fan. */
typedef struct DecodeCtx {
    AVThreadMessageQueue *video_q;            /* demux -> decode (AVPacket*) */
    AVCodecContext  *vdec;
    AVRational       ist_tb;                  /* decoder pkt time_base */
    int64_t         *h0;                      /* shared A/V input anchor (us) */
    pthread_mutex_t *h0_lock;
    int              live;
    VideoHold       *hold;                    /* multiview: stage frames here (NULL = filter inline) */
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
    int64_t         *house_skew;     /* master publishes house-vs-content skew (us) here */
    AVCodecContext  *venc;
    AVStream        *ost;
    int64_t          tick_dur_us;
    int              live;
    int              passthrough;    /* multiview: compositor already paced+stamped; encode 1:1 */
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

/* Build a SHARED filter_complex graph: N video buffersrcs (one per input, bound
 * to the graph's `[k:v]` labels) -> the user graph (single-input ABR split, or a
 * multiview xstack/hstack/overlay mosaic + split) -> N buffersinks (one per rung
 * label). Mirrors ffmpeg's -filter_complex. n_inputs==1 is the single-input
 * ladder (unchanged); n_inputs>1 is multiview, fed by the compositor.
 *
 * vdecs[k] supplies input k's width/height/pix_fmt/sar; src_tb is the time_base
 * stamped on every buffersrc (single-input: the source stream tb, frames carry
 * source pts; multiview: 1/out_fps, the compositor stamps pts = house tick).
 * labels[i] is rung i's bare output label; sinks[i] receives that branch. */
static int build_filter_complex(const char *graph_str, AVCodecContext **vdecs,
                                int n_inputs, AVRational src_tb, AVBufferRef *hw_device,
                                const char *const *labels, int n_labels,
                                AVFilterGraph **out_fg, AVFilterContext **srcs,
                                AVFilterContext **sinks)
{
    char args[256], name[16];
    AVFilterGraph        *fg    = avfilter_graph_alloc();
    AVFilterGraphSegment *seg   = NULL;
    const AVFilter       *bsrc  = avfilter_get_by_name("buffer");
    const AVFilter       *bsink = avfilter_get_by_name("buffersink");
    AVFilterInOut        *gin = NULL, *gout = NULL, *io;
    int ret, i, k, linked = 0;

    if (!fg || !bsrc || !bsink) { ret = AVERROR(ENOMEM); goto fail; }

    /* Build via the segment API so the hw device is assigned to every filter
     * BEFORE it is initialised. Plain `hwupload` (unlike `hwupload_cuda`)
     * hard-requires avctx->hw_device_ctx in its init(); a one-shot
     * avfilter_graph_parse2() inits filters during the parse, too early to set
     * it. Sequence: parse -> create_filters -> SET DEVICE -> apply_opts -> init
     * -> link. After link, gin = the unconnected inputs ([k:v]) and gout = the
     * unconnected outputs (each rung label), wired to our buffersrcs / sinks. */
    if ((ret = avfilter_graph_segment_parse(fg, graph_str, 0, &seg)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex parse: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_create_filters(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex create: %s\n", av_err2str(ret)); goto fail;
    }
    if (hw_device)                                   /* must precede init() (hwupload) */
        for (unsigned f = 0; f < fg->nb_filters; f++)
            fg->filters[f]->hw_device_ctx = av_buffer_ref(hw_device);
    if ((ret = avfilter_graph_segment_apply_opts(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex opts: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_init(seg, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex init: %s\n", av_err2str(ret)); goto fail;
    }
    if ((ret = avfilter_graph_segment_link(seg, 0, &gin, &gout)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex link: %s\n", av_err2str(ret)); goto fail;
    }

    /* one buffersrc per input, params from that input's decoder */
    for (k = 0; k < n_inputs; k++) {
        AVRational sar = vdecs[k]->sample_aspect_ratio.num ? vdecs[k]->sample_aspect_ratio : (AVRational){1, 1};
        snprintf(name, sizeof name, "in%d", k);
        snprintf(args, sizeof args,
                 "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
                 vdecs[k]->width, vdecs[k]->height, vdecs[k]->pix_fmt, src_tb.num, src_tb.den, sar.num, sar.den);
        if ((ret = avfilter_graph_create_filter(&srcs[k], bsrc, name, args, NULL, fg)) < 0) goto fail;
    }
    /* link each unconnected graph input [K:v] to buffersrc K (single input: [0:v]->src0) */
    for (io = gin; io; io = io->next) {
        int idx = 0;
        if (io->name) { const char *c = io->name; if (*c >= '0' && *c <= '9') idx = atoi(c); }
        if (idx < 0 || idx >= n_inputs) {
            av_log(NULL, AV_LOG_ERROR, "filter_complex input [%s] out of range (n_inputs=%d)\n",
                   io->name ? io->name : "?", n_inputs);
            ret = AVERROR(EINVAL); goto fail;
        }
        if ((ret = avfilter_link(srcs[idx], 0, io->filter_ctx, io->pad_idx)) < 0) goto fail;
        linked++;
    }
    if (linked != n_inputs) {
        av_log(NULL, AV_LOG_ERROR, "filter_complex: %d input(s) linked, expected %d ([0:v]..[%d:v])\n",
               linked, n_inputs, n_inputs - 1);
        ret = AVERROR(EINVAL); goto fail;
    }

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
    *out_fg = fg;
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

/* Multiview: push a decoded frame onto this input's jitter buffer for the
 * compositor (FIFO, one per tick). Carries source pts on frame->pts so the
 * compositor can compute this slot's house skew. Takes ownership of the frame. */
static void stage_hold(VideoHold *h, int live, AVFrame *frame)
{
    AVFrame *nf = av_frame_alloc();
    if (!nf) { av_frame_unref(frame); return; }
    av_frame_move_ref(nf, frame);
    if (nf->best_effort_timestamp != AV_NOPTS_VALUE) nf->pts = nf->best_effort_timestamp;
    push_frame_q(h->q, live, &h->framedrop, nf);    /* drop-oldest in live; consumes nf */
    pthread_mutex_lock(&h->lock);
    h->wall_us = av_gettime_relative();
    pthread_mutex_unlock(&h->lock);
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
            if (d->hold) stage_hold(d->hold, d->live, frame);   /* multiview: compositor samples this */
            else         emit_video(d, frame, filt);
        }
    }
    /* flush decoder */
    avcodec_send_packet(d->vdec, NULL);
    while (avcodec_receive_frame(d->vdec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        d->dec_frames++;
        if (d->hold) stage_hold(d->hold, d->live, frame);
        else         emit_video(d, frame, filt);
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
    if (d->hold) {                          /* multiview: signal terminal EOF to the compositor */
        pthread_mutex_lock(&d->hold->lock);
        d->hold->eof = 1;
        pthread_mutex_unlock(&d->hold->lock);
        av_thread_message_queue_set_err_recv(d->hold->q, AVERROR_EOF);   /* drain then EOF the jitter buffer */
    }
    av_thread_message_queue_set_err_send(d->video_q, AVERROR_EOF);   /* unblock demux (a SENDER) */
    for (i = 0; i < d->n_rung; i++)
        av_thread_message_queue_set_err_recv(d->frame_q[i], AVERROR_EOF);  /* EOF to each output (RECEIVER) */
    return NULL;
}

static void *output_thread(void *arg)
{
    VideoCtx *v = arg;
    AVFrame *held = av_frame_alloc();
    AVFrame *f;
    int have = 0, ret = 0;
    int64_t tick = 0, wall0 = 0, last_vpts = -1;
    int64_t held_src_pts = AV_NOPTS_VALUE;   /* ORIGINAL source pts of held frame (held->pts gets
                                                overwritten to vpts on emit; dups must not re-read it) */
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

    if (v->passthrough) {
        /* multiview: the compositor IS the house clock — it paced this frame and
         * already stamped pts (in venc tb). Encode 1:1, no re-pace / dup / skew
         * (the compositor owns all of that, and the stats/diag line). */
        for (;;) {
            ret = av_thread_message_queue_recv(v->frame_q, &f, 0);
            if (ret < 0) break;
            f->pkt_dts = AV_NOPTS_VALUE; f->duration = 0;
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
            have = 1; fresh = 1; held_src_pts = held->pts;   /* capture before emit overwrites it */
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
            int64_t vpts, content_vpts = -1;
            int64_t src_ts = held_src_pts;   /* ORIGINAL source pts (out_tb); survives dups */
            if (src_ts != AV_NOPTS_VALUE && *v->h0 != AV_NOPTS_VALUE) {
                int64_t house_us = av_rescale_q(src_ts, v->out_tb, AV_TIME_BASE_Q) - *v->h0;
                if (house_us < 0) house_us = 0;
                content_vpts = (house_us + v->tick_dur_us / 2) / v->tick_dur_us;
                vpts = content_vpts;
            } else {
                vpts = last_vpts + 1;
            }
            if (vpts <= last_vpts) vpts = last_vpts + 1;   /* monotonic CFR; dup -> next slot */
            held->pts = vpts; held->pkt_dts = AV_NOPTS_VALUE; held->duration = 0;
            last_vpts = vpts;
            /* Publish how far the house clock now runs AHEAD of source content
             * (vpts - content_vpts, in ticks). Each dup bumps vpts past content via
             * the monotonic guard, so this grows by one tick per dup and persists.
             * The audio path adds it so audio rides the same house clock instead of
             * staying source-locked (which is what drifts ~40ms per dup). */
            if (v->is_master && v->house_skew && content_vpts >= 0)
                *v->house_skew = (vpts - content_vpts) * v->tick_dur_us;
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
            if (nows - stat_last >= g_stats_period_us) {
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
    AVCodecContext  *enc[PTV_MAX_RUNG];          /* one AAC encoder per rung (per-rung -b:a) */
    AVRational       ist_tb;
    SwrContext      *swr;                         /* no -af: plain resample to 48k stereo */
    AVFilterGraph   *afg;                         /* -af present: abuffer -> chain -> abuffersink */
    AVFilterContext *afsrc, *afsink;
    int              use_fg;
    AVAudioFifo     *fifo;
    int              frame_size;
    int              out_rate;
    enum AVSampleFormat out_sfmt;
    AVChannelLayout  out_chl;
    int64_t         *h0;
    pthread_mutex_t *h0_lock;
    int64_t         *house_skew;    /* video's house-vs-content skew (us); -af audio rides it */
    int              pts_set;
    int64_t          next_pts;
    int64_t          in_frames, out_frames;
} AudioState;

/* encode the SAME loudness-processed frame into each rung's own AAC encoder (so
 * per-rung -b:a is honored), routing each rung's packets to its muxer. The frame
 * is only ref'd by avcodec_send_frame, so the one frame feeds all N encoders.
 * frame=NULL flushes every encoder. (ffmpeg's filter -> asplit -> N encoders.) */
static int audio_encode_push(AudioState *a, AVFrame *frame)
{
    int i, ret;
    for (i = 0; i < a->n_out; i++) {
        ret = avcodec_send_frame(a->enc[i], frame);
        if (ret < 0)
            return ret;
        for (;;) {
            AVPacket *pkt = av_packet_alloc();
            if (!pkt)
                return AVERROR(ENOMEM);
            ret = avcodec_receive_packet(a->enc[i], pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
            if (ret < 0) { av_packet_free(&pkt); return ret; }
            av_packet_rescale_ts(pkt, a->enc[i]->time_base, a->ost[i]->time_base);
            pkt->stream_index = a->ost[i]->index;
            if (av_thread_message_queue_send(a->mux_q[i], &pkt, 0) < 0)   /* blocking */
                av_packet_free(&pkt);
        }
    }
    return 0;
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

/* -af path: drain the filtergraph's (fixed-size) output frames straight to the
 * per-rung encoders, stamping each with ITS OWN filter PTS rebased onto the house
 * anchor h0 (in out_rate sample units). This HONORS aresample=async's correction
 * — discarding it for a free sample counter is what let audio drift behind video.
 * Frames whose rebased pts precedes the video anchor (<0) are dropped. */
static int audio_drain_fg(AudioState *a)
{
    AVFrame *filt = av_frame_alloc();
    AVRational sink_tb;
    int64_t h0, h0_samp;
    int ret = 0;
    if (!filt) return AVERROR(ENOMEM);
    pthread_mutex_lock(a->h0_lock); h0 = *a->h0; pthread_mutex_unlock(a->h0_lock);
    h0_samp = (h0 == AV_NOPTS_VALUE) ? 0 : av_rescale(h0, a->out_rate, 1000000);
    sink_tb = av_buffersink_get_time_base(a->afsink);
    while ((ret = av_buffersink_get_frame(a->afsink, filt)) >= 0) {
        if (filt->pts != AV_NOPTS_VALUE) {
            int64_t opts = av_rescale_q(filt->pts, sink_tb, (AVRational){1, a->out_rate}) - h0_samp;
            if (opts < 0) { av_frame_unref(filt); continue; }   /* precedes video anchor */
            filt->pts = opts;
        }
        ret = audio_encode_push(a, filt);
        a->out_frames++;
        av_frame_unref(filt);
        if (ret < 0) break;
    }
    av_frame_free(&filt);
    return (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) ? 0 : ret;
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

    if (a->use_fg) {
        /* -af: feed the graph; aresample async + loudness emit fixed-size frames
         * whose PTS already carries async's A/V correction — drain them straight
         * to the encoders (no FIFO, no free counter).
         *
         * Common-mode A/V lock: add the video's house-vs-content skew to this
         * frame's PTS before it enters the graph, so aresample=async targets the
         * HOUSE clock (where video lives) instead of the source clock. When video
         * dups (house outpaces source), the skew grows and async smoothly fills
         * the matching audio, so audio rides the dup with video — no drift. */
        if (g_avlock && a->house_skew && frame->pts != AV_NOPTS_VALUE) {
            int64_t sk = *a->house_skew;
            if (sk) frame->pts += av_rescale_q(sk, AV_TIME_BASE_Q, a->ist_tb);
        }
        if ((ret = av_buffersrc_add_frame(a->afsrc, frame)) < 0)
            return ret;
        return audio_drain_fg(a);
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
    /* flush decoder -> resampler/filtergraph -> encoder */
    avcodec_send_packet(a->dec, NULL);
    while (avcodec_receive_frame(a->dec, frame) >= 0) { audio_push(a, frame); av_frame_unref(frame); }
    if (a->use_fg) {
        if (av_buffersrc_add_frame(a->afsrc, NULL) >= 0)   /* signal EOF to the graph */
            audio_drain_fg(a);
        audio_encode_push(a, NULL);                        /* flush encoders */
    } else {
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
    av_thread_message_queue_set_err_send(a->audio_q, AVERROR_EOF);   /* unblock demux (a SENDER) */
    { int i; for (i = 0; i < a->n_out; i++) {        /* EOF marker to each muxer */
        AVPacket *eof = NULL; av_thread_message_queue_send(a->mux_q[i], &eof, 0); } }
    return NULL;
}

/* Build the audio filtergraph: abuffer -> [user -af chain] -> aformat -> abuffersink.
 * Mirrors build_video_filter for audio. The trailing aformat pins the sink to the
 * encoder's format (48k stereo + enc sample_fmt) so the graph auto-inserts any
 * needed aresample even when -af omits one; the -af chain (aresample=async,
 * acompressor, alimiter, ...) runs first. Sets a->use_fg on success. */
static int build_audio_filter(AudioState *a, AVCodecContext *adec, AVRational tb,
                              const char *af, enum AVSampleFormat out_fmt)
{
    char args[256], chain[512], chl[64], outchl[64];
    const AVFilter *bsrc  = avfilter_get_by_name("abuffer");
    const AVFilter *bsink = avfilter_get_by_name("abuffersink");
    AVFilterInOut *ins = avfilter_inout_alloc(), *outs = avfilter_inout_alloc();
    int ret;

    if (!bsrc || !bsink || !ins || !outs) { ret = AVERROR(ENOMEM); goto end; }
    a->afg = avfilter_graph_alloc();
    if (!a->afg) { ret = AVERROR(ENOMEM); goto end; }

    av_channel_layout_describe(&adec->ch_layout, chl, sizeof chl);
    snprintf(args, sizeof args,
             "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
             tb.num, tb.den, adec->sample_rate,
             av_get_sample_fmt_name(adec->sample_fmt), chl);
    if ((ret = avfilter_graph_create_filter(&a->afsrc, bsrc, "in", args, NULL, a->afg)) < 0) goto end;
    if ((ret = avfilter_graph_create_filter(&a->afsink, bsink, "out", NULL, NULL, a->afg)) < 0) goto end;

    av_channel_layout_describe(&a->out_chl, outchl, sizeof outchl);   /* -ac:a:N target layout */
    snprintf(chain, sizeof chain,
             "%s%saformat=sample_fmts=%s:sample_rates=48000:channel_layouts=%s",
             af ? af : "", af ? "," : "", av_get_sample_fmt_name(out_fmt), outchl);

    outs->name = av_strdup("in");  outs->filter_ctx = a->afsrc;  outs->pad_idx = 0; outs->next = NULL;
    ins->name  = av_strdup("out"); ins->filter_ctx  = a->afsink; ins->pad_idx  = 0; ins->next  = NULL;
    if ((ret = avfilter_graph_parse_ptr(a->afg, chain, &ins, &outs, NULL)) < 0) goto end;
    if ((ret = avfilter_graph_config(a->afg, NULL)) < 0) goto end;

    /* deliver encoder-sized frames so we can feed them straight to the AAC
     * encoders carrying their own (async-corrected) PTS — no FIFO repackaging. */
    if (a->frame_size > 0)
        av_buffersink_set_frame_size(a->afsink, a->frame_size);

    av_log(NULL, AV_LOG_INFO, "ptvencoder: audio filter [%s]\n", chain);
    a->use_fg = 1;
    ret = 0;
end:
    avfilter_inout_free(&ins);
    avfilter_inout_free(&outs);
    return ret;
}

/* ---- demux + mux ---- */

#define PTV_MAX_PASS 16
typedef struct PassStream {
    int        input;                 /* source input index (multiview); 0 single-input */
    int        in_index;              /* input stream index being copied 1:1 (within that input) */
    AVStream  *ost[PTV_MAX_RUNG];     /* output stream in each muxer (fan-out) */
    AVRational in_tb;                 /* input/output time_base (copy: identical) */
    int64_t    last_dts;              /* last emitted dts (monotonic guard; NOPTS until first) */
} PassStream;

typedef struct DemuxArgs {
    AVFormatContext      *ifmt;
    AVThreadMessageQueue *video_q;
    AVThreadMessageQueue *audio_q[PTV_MAX_AUDIO]; /* one per transcoded audio track */
    AVThreadMessageQueue *mux_q[PTV_MAX_RUNG];   /* one per output muxer (fan-out) */
    int                   n_out;
    int                   vstream;
    int                   astream[PTV_MAX_AUDIO]; /* input stream feeding each audio_q */
    int                   n_audio;
    int                   drop;          /* non-blocking + drop on full (network input) */
    PassStream           *pass;          /* copy-passthrough: extra audio, subs, data */
    int                   n_pass;
    int64_t              *h0;             /* house origin (us); copy ts rebased onto it */
    pthread_mutex_t      *h0_lock;
    int64_t              *house_skew;     /* video's house-vs-content skew (us); copy rides it */
    int64_t              *wrap_off;       /* per input stream: cumulative 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;      /* per input stream: last RAW ts seen (wrap detection) */
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
        /* Ride the house clock: subtract h0, then ADD the video's house-vs-content
         * skew so copied streams (AC-3, subs, data) stay aligned with the dup-shifted
         * video instead of source-locked. Copied audio can't be resampled, so this is
         * a step (the skew grows in ~40ms tick increments) -> a small periodic A/V hop
         * on dense audio; sparse subs/data ride it invisibly. (Smooth would need rate-
         * discipline so dups -> 0; see option 2.) */
        if (g_avlock && d->house_skew)
            h0_tb -= av_rescale_q(*d->house_skew, AV_TIME_BASE_Q, d->pass[pi].in_tb);
        if (out->pts != AV_NOPTS_VALUE) out->pts -= h0_tb;
        if (out->dts != AV_NOPTS_VALUE) out->dts -= h0_tb;
        ref = out->dts != AV_NOPTS_VALUE ? out->dts : out->pts;
        if (ref != AV_NOPTS_VALUE && ref < 0) { av_packet_free(&out); return 0; }  /* precedes anchor */
        /* Monotonic-DTS guard for the copy path. The house-skew rebase above (and
         * source quirks / wrap edges) can nudge a copied stream's dts backward
         * between packets; the mpegts muxer rejects a backward dts with EINVAL and
         * the rung dies — this froze channels under the 50fps field-rate dup-storm,
         * and a rare dup-induced skew dip could trip it even at the right rate.
         * Clamp strictly increasing per copied stream, shifting pts by the same
         * amount so pts>=dts still holds. (On-wire SCTE-35 splice timing rides
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
            demux_send(d->mux_q[i], c, d->drop, &d->pdrop);
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
    raw  = pkt->dts != AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
    if (raw != AV_NOPTS_VALUE) {
        int64_t last = d->wrap_last[pkt->stream_index];
        if (last != AV_NOPTS_VALUE) {
            if (raw - last < -half)      d->wrap_off[pkt->stream_index] += mask;  /* rolled forward */
            else if (raw - last >  half) d->wrap_off[pkt->stream_index] -= mask;  /* late pre-roll pkt */
        }
        d->wrap_last[pkt->stream_index] = raw;
    }
    off = d->wrap_off[pkt->stream_index];
    if (off) {
        if (pkt->pts != AV_NOPTS_VALUE) pkt->pts += off;
        if (pkt->dts != AV_NOPTS_VALUE) pkt->dts += off;
    }
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
        demux_unwrap(d, out);               /* 33-bit source wrap -> monotonic extended ts (ONCE) */
        if (out->stream_index == d->vstream) {
            d->vpkt++;
            ret = demux_send(d->video_q, out, d->drop, &d->vdrop);
        } else {
            /* Fan one source PID to every transcoded audio track on it (a clone
             * each), then hand the original to demux_pass (copy-passthrough; it
             * frees it, whether or not it's a copy stream). demux_unwrap ran ONCE
             * above, so every clone carries the same unwrapped ts (load-bearing:
             * never unwrap per-clone — the per-stream wrap state is stateful). */
            int k;
            for (k = 0; k < d->n_audio; k++) {
                AVPacket *c;
                if (d->astream[k] != out->stream_index) continue;
                if (!(c = av_packet_clone(out))) continue;
                d->apkt++;
                demux_send(d->audio_q[k], c, d->drop, &d->adrop);
            }
            ret = demux_pass(d, out);       /* copy fan + monotonic-DTS clamp; frees out */
        }
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
    av_thread_message_queue_set_err_send(m->mux_q, AVERROR_EOF);   /* unblock producers (SENDERS) */
    return NULL;
}

/* resolved per-output selection from an ffmpeg-style command (see resolve_plan) */
/* One TRANSCODED audio output track (copy audio rides the copy[] passthrough list,
 * NOT this, so it keeps demux_pass's wrap unwrap + monotonic-DTS clamp + SCTE rebase). */
typedef struct AOutSpec {
    int            input;                 /* source input index (multiview -map K:a:N); 0 single-input */
    int            stream;                /* input audio stream index within that input */
    const AVCodec *adec;
    const char    *aenc;                  /* -c:a:N encoder name (NULL = aac) */
    const char    *abr;                   /* -b:a:N */
    const char    *filter;                /* -filter:a:N (NULL = global -af) */
    int            ac;                    /* -ac:a:N output channels (0 = default stereo) */
    const char    *lang;                  /* source language (override -metadata later) */
} AOutSpec;

typedef struct Sel {
    int            have;                  /* 1 if -map present (explicit plan) */
    int            vstream;               /* input video stream to transcode (-1 none) */
    const AVCodec *vdec;
    const char    *venc;                  /* -c:v encoder name (NULL = default) */
    const char    *vf;                    /* -filter:v / -vf */
    const char    *vbr;                   /* -b:v */
    AOutSpec       aout[PTV_MAX_AUDIO];   /* transcoded audio output tracks */
    int            n_aout;
    int            copy[PTV_MAX_PASS];    /* ALL copy: audio (5.1/2ch) + sub + data + scte */
    int            copy_input[PTV_MAX_PASS]; /* source input index per copy stream (multiview) */
    int            n_copy;
} Sel;
static const char *og_get(OptionGroup *g, const char *key);
static void apply_stream_meta(OptionGroup *g, char t, int idx, AVStream *ost);
struct Input;
static int resolve_plan(struct Input *inputs, int n_input, OptionGroup *outg, Sel *s);

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

/* One source input. Single-input uses inputs[0]; multiview uses 1/2/4. Each has
 * its own demuxer + video decoder + clock anchor (h0) + wrap state; multiview
 * decode stages into `hold` for the compositor instead of filtering inline. */
typedef struct Input {
    const char           *url;
    AVFormatContext      *ifmt;
    int                   vstream;
    const AVCodec        *vdecoder;
    AVCodecContext       *vdec;
    AVStream             *vist;
    AVRational            ist_tb;
    AVThreadMessageQueue *video_q;           /* demux -> decode */
    int64_t               h0;                /* this input's A/V anchor (us); decode sets it */
    pthread_mutex_t       h0_lock;
    int64_t               house_skew;        /* compositor publishes; this input's audio/copy ride it */
    VideoHold             hold;              /* multiview: latest decoded frame for the compositor */
    int64_t              *wrap_off;          /* per stream: 33-bit wrap offset (stream tb) */
    int64_t              *wrap_last;         /* per stream: last RAW ts (wrap detection) */
    DecodeCtx             dc;
    DemuxArgs             da;
    pthread_t             th_demux, th_decode;
    int                   started_demux, started_decode;
    int                   open_ret;          /* parallel-open result */
} Input;

/* Multiview compositor = the video house clock. Samples each input's hold at
 * each tick, feeds the N buffersrcs, pulls each rung's composited frame, and
 * publishes per-input house_skew. (Single-input never uses this — decode feeds
 * the graph inline and the per-rung output_thread is the house clock.) */
typedef struct CompositorCtx {
    Input                *inputs;
    int                   n_input;
    AVFilterGraph        *fg;
    AVFilterContext      *fsrc[PTV_MAX_INPUT];
    int                   n_rung;
    AVFilterContext      *fsink[PTV_MAX_RUNG];
    AVThreadMessageQueue *frame_q[PTV_MAX_RUNG];
    int64_t               framedrop[PTV_MAX_RUNG];
    int64_t               tick_dur_us;       /* 1/out_fps in us */
    int                   live;
    int64_t               slate_after_us;    /* stale hold -> black cell (0 = never) */
    /* stats (compositor is the cadence owner in multiview) */
    int64_t               emitted, dup;
    int64_t              *dbg_dec_sum;        /* optional: sum of per-input dec_frames */
} CompositorCtx;

/* fill a planar-YUV / RGB frame with black (held cell for a not-yet-arrived or
 * stale multiview slot): luma 16, chroma 128, RGB 0. */
static void ptv_fill_black(AVFrame *f)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(f->format);
    int p;
    if (!d) return;
    for (p = 0; p < AV_NUM_DATA_POINTERS && f->data[p]; p++) {
        int rgb = !!(d->flags & AV_PIX_FMT_FLAG_RGB);
        int chroma = !rgb && (p == 1 || p == 2);
        int val = rgb ? 0 : (chroma ? 128 : 16);
        int h = (chroma) ? AV_CEIL_RSHIFT(f->height, d->log2_chroma_h) : f->height;
        memset(f->data[p], val, (size_t)f->linesize[p] * h);
    }
}

static AVFrame *make_black_frame(AVCodecContext *vdec)
{
    AVFrame *f = av_frame_alloc();
    if (!f) return NULL;
    f->format = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
    f->width  = vdec->width;
    f->height = vdec->height;
    if (av_frame_get_buffer(f, 0) < 0) { av_frame_free(&f); return NULL; }
    ptv_fill_black(f);
    return f;
}

/* Multiview compositor thread — the video house clock. Each wall-paced tick:
 * sample-and-hold each input's latest decoded frame (dup-hold a stale one;
 * black-slate a missing/long-stale one), feed all N buffersrcs with pts = tick
 * (so the mosaic's xstack/overlay framesync pairs them and never waits), pull
 * each rung's composited+scaled frame, stamp it consecutive-CFR, and hand it to
 * that rung's frame_q. Publishes per-input house_skew for the slot's audio lock.
 * A late/dead/frozen slot NEVER stalls the mosaic (R-MV-6). */
static void *compositor_thread(void *arg)
{
    CompositorCtx *c = arg;
    int n = c->n_input, R = c->n_rung, k, r;
    AVFrame *blackf[PTV_MAX_INPUT] = {0};
    AVFrame *last[PTV_MAX_INPUT] = {0};       /* last frame popped per input (dup source) */
    int64_t last_fresh_us[PTV_MAX_INPUT] = {0};
    int64_t skew_us[PTV_MAX_INPUT] = {0};     /* per-slot audio skew = accumulated dup-hold ticks */
    int      done_in[PTV_MAX_INPUT] = {0};
    int64_t rung_pts[PTV_MAX_RUNG] = {0};
    AVFrame *filt = av_frame_alloc();
    int64_t tick = 0, wall0 = 0;
    const char *pe = getenv("PTV_PREROLL_MS");
    int preroll_ms = pe ? atoi(pe) : 350;
    int n_prime = (preroll_ms > 0 && c->tick_dur_us > 0) ? (int)((int64_t)preroll_ms * 1000 / c->tick_dur_us) : 0;
    int64_t diag_t0 = av_gettime_relative(), diag_last = diag_t0;
    int64_t stat_last = diag_t0, stat_prev = 0;

    if (!filt) goto done;
    if (n_prime > PTV_FRAME_QDEPTH - 8) n_prime = PTV_FRAME_QDEPTH - 8;
    if (n_prime < 0) n_prime = 0;
    for (k = 0; k < n; k++) blackf[k] = make_black_frame(c->inputs[k].vdec);

    /* preroll: prime every input's jitter buffer to ~PTV_PREROLL_MS so bursty
     * decode delivery has a cushion (no startup dup storm) and the mosaic starts
     * with every cell live. A never-arriving input is left to its black cell. */
    {
        int64_t t0 = av_gettime_relative();
        for (;;) {
            int ready = 0, eofall = 1;
            for (k = 0; k < n; k++) {
                pthread_mutex_lock(&c->inputs[k].hold.lock);
                if (!c->inputs[k].hold.eof) eofall = 0;
                pthread_mutex_unlock(&c->inputs[k].hold.lock);
                if (av_thread_message_queue_nb_elems(c->inputs[k].hold.q) >= (n_prime > 0 ? n_prime : 1)) ready++;
            }
            if (ready == n || eofall) break;
            if (av_gettime_relative() - t0 > 3000000) break;   /* 3s: start with what's there */
            av_usleep(5000);
        }
        wall0 = av_gettime_relative();
    }

    for (;;) {
        int all_eof = 1, any_fresh = 0;
        int64_t now_us;
        {                                            /* wall-pace the house tick (also offline:
                                                      * inputs have independent clocks, so the
                                                      * mosaic cadence is the house rate, not media) */
            int64_t target = wall0 + tick * c->tick_dur_us;
            int64_t now = av_gettime_relative();
            if (now < target) av_usleep((unsigned)(target - now));
        }
        now_us = av_gettime_relative();

        for (k = 0; k < n; k++) {                    /* pop ONE frame from this input's jitter
                                                      * buffer (FIFO) -> feed buffersrc k; dup-hold
                                                      * its last frame on underrun, black when stale */
            VideoHold *h = &c->inputs[k].hold;
            AVFrame *f = NULL, *st; int stale, fresh;
            int rr = av_thread_message_queue_recv(h->q, &f, AV_THREAD_MESSAGE_NONBLOCK);
            fresh = (rr >= 0);
            if (fresh) { if (last[k]) av_frame_free(&last[k]); last[k] = f; any_fresh = 1; last_fresh_us[k] = now_us; }
            else if (rr == AVERROR_EOF) done_in[k] = 1;
            if (!done_in[k]) all_eof = 0;
            stale = (c->slate_after_us > 0 && last_fresh_us[k] > 0 && now_us - last_fresh_us[k] > c->slate_after_us);
            /* Per-slot audio skew = accumulated GENUINE dup-holds (queue underran, a real
             * frame is re-shown): the house ran one tick ahead of this cell, so its audio
             * must lag by the same tick to stay in lip-sync with the (held) cell — exactly
             * the single-input dup->skew rule, but event-counted per slot. Event-counting
             * (not the arithmetic rung_pts-content_tks) avoids locking startup/rounding
             * jitter into a permanent over-delay, which de-synced the later-priming slots.
             * Only count while a real frame is held (not black-slated, not EOF). Monotonic
             * by construction (keeps the audio input PTS forward); capped defensively. */
            if (!fresh && !done_in[k] && last[k] && !stale && c->tick_dur_us > 0) {
                if (skew_us[k] < PTV_MV_SKEW_CAP_US) skew_us[k] += c->tick_dur_us;
                c->inputs[k].house_skew = skew_us[k];
            }
            if (last[k] && !stale) st = av_frame_clone(last[k]);
            else                   st = blackf[k] ? av_frame_clone(blackf[k]) : NULL;
            if (!st) continue;
            st->pts = tick; st->pkt_dts = AV_NOPTS_VALUE;
            if (av_buffersrc_add_frame(c->fsrc[k], st) < 0) av_frame_free(&st);
        }

        for (r = 0; r < R; r++) {                    /* pull each rung's composited frame */
            while (av_buffersink_get_frame(c->fsink[r], filt) >= 0) {
                AVFrame *out = av_frame_alloc();
                if (out) {
                    av_frame_move_ref(out, filt);
                    out->pts = rung_pts[r]++;        /* per-rung consecutive CFR */
                    out->pkt_dts = AV_NOPTS_VALUE; out->duration = 0;
                    push_frame_q(c->frame_q[r], c->live, &c->framedrop[r], out);
                    if (r == 0) { c->emitted++; if (!any_fresh) c->dup++; }
                } else av_frame_unref(filt);
            }
        }
        tick++;

        if (g_diag) {
            int64_t nowd = av_gettime_relative();
            if (nowd - diag_last >= 1000000) {
                char db[256]; int dp = 0;
                for (k = 0; k < n && dp < (int)sizeof db - 32; k++)
                    dp += snprintf(db + dp, sizeof db - dp, " in%d:dec=%"PRId64"/skew=%dms",
                                   k, c->inputs[k].dc.dec_frames, (int)(skew_us[k] / 1000));
                av_log(NULL, AV_LOG_INFO,
                    "[PTV-DIAG] mv t=%.1fs emitted=%"PRId64" dup=%"PRId64" muxed=%"PRId64" frameq0=%d%s\n",
                    (nowd - diag_t0) / 1000000.0, c->emitted, c->dup, g_muxed,
                    av_thread_message_queue_nb_elems(c->frame_q[0]), db);
                diag_last = nowd;
            }
        }
        if (g_stats) {
            int64_t nows = av_gettime_relative();
            if (nows - stat_last >= g_stats_period_us) {
                double dt    = (nows - stat_last) / 1000000.0;
                double fps   = (c->emitted - stat_prev) / (dt > 0 ? dt : 1);
                double secs  = c->emitted * c->tick_dur_us / 1000000.0;
                double wall  = (nows - wall0) / 1000000.0;
                double speed = wall > 0 ? secs / wall : 0;
                double kbps  = secs > 0 ? g_muxed_bytes * 8.0 / secs / 1000.0 : 0;
                int hh = (int)(secs / 3600), mm = ((int)secs % 3600) / 60;
                double ss = secs - hh * 3600 - mm * 60;
                av_log(NULL, AV_LOG_INFO,
                    "frame=%6"PRId64" fps=%3.0f size=%8"PRId64"KiB time=%02d:%02d:%05.2f "
                    "bitrate=%7.1fkbits/s dup=%"PRId64" drop=%"PRId64" speed=%4.2fx\n",
                    c->emitted, fps, g_muxed_bytes / 1024, hh, mm, ss, kbps,
                    c->dup, c->framedrop[0], speed);
                stat_last = nows; stat_prev = c->emitted;
            }
        }
        if (g_slow) av_usleep(g_slow);
        if (all_eof) break;                          /* every input terminated -> tear down */
    }

    for (k = 0; k < n; k++) { int fr = av_buffersrc_add_frame(c->fsrc[k], NULL); (void)fr; }   /* flush graph */
    for (r = 0; r < R; r++)
        while (av_buffersink_get_frame(c->fsink[r], filt) >= 0) {
            AVFrame *out = av_frame_alloc();
            if (out) { av_frame_move_ref(out, filt); out->pts = rung_pts[r]++; out->pkt_dts = AV_NOPTS_VALUE; out->duration = 0;
                       push_frame_q(c->frame_q[r], c->live, &c->framedrop[r], out); }
            else     { av_frame_unref(filt); }
        }
done:
    av_frame_free(&filt);
    for (k = 0; k < n; k++) { if (blackf[k]) av_frame_free(&blackf[k]); if (last[k]) av_frame_free(&last[k]); }
    for (r = 0; r < R; r++) av_thread_message_queue_set_err_recv(c->frame_q[r], AVERROR_EOF);
    return NULL;
}

/* open one input on its own thread (parallel open: a dead/slow slot must not
 * delay the others, and serial open would block on its long rw_timeout). */
typedef struct OpenArg { Input *in; AVDictionary **opts; } OpenArg;
static void *open_input_thread(void *arg)
{
    OpenArg *o = arg; Input *in = o->in;
    in->open_ret = avformat_open_input(&in->ifmt, in->url, NULL, o->opts);
    if (in->open_ret >= 0) in->open_ret = avformat_find_stream_info(in->ifmt, NULL);
    return NULL;
}

static int is_net_url(const char *u)
{
    return u && (!strncmp(u, "udp://", 6) || !strncmp(u, "rtp://", 6) || !strncmp(u, "srt://", 6));
}

/* transcode: ins = parsed input group list (1/2/4 inputs; >1 = multiview);
 * outs = the list of output groups (one per ABR rung); fcomplex = the shared
 * -filter_complex. The ffmpeg model — decode each input once, one filter graph
 * (single-input split, or N-input mosaic+split) feeds each rung's independent
 * muxer/encoder; audio + subs/data decoded/copied once and fanned out.
 * Selection (transcode vs copy) per group comes from its -map/-c. */
static int transcode(OptionGroupList *ins, OptionGroupList *outs, const char *fcomplex,
                     const char *hwdev, int mode)
{
    int n_input = ins->nb_groups;
    int n_rung = outs->nb_groups;
    int multiview;
    Input            inputs[PTV_MAX_INPUT];
    AVCodecContext  *vdecs[PTV_MAX_INPUT];
    AVFilterContext *fsrc[PTV_MAX_INPUT] = {0};
    AVThreadMessageQueue *audio_q[PTV_MAX_AUDIO] = {0};
    AVFilterGraph   *fg = NULL;                 /* the shared graph (single or multiview) */
    AVFilterContext **vsink = NULL;            /* per-rung buffersinks (single dc / multi comp) */
    int              filtering = 0;
    AVBufferRef     *hw_device = NULL;
    AudioState       as[PTV_MAX_AUDIO];        /* one per transcoded audio track */
    int              asrc[PTV_MAX_AUDIO];      /* input-local stream feeding each as[] */
    int              asrc_in[PTV_MAX_AUDIO];   /* source input index feeding each as[] */
    int              n_audio = 0;
    Rung             rung[PTV_MAX_RUNG];
    Sel              sel[PTV_MAX_RUNG];
    CompositorCtx    comp;
    pthread_t        th_compositor, th_audio[PTV_MAX_AUDIO];
    int              started_compositor = 0;
    int              started_audio[PTV_MAX_AUDIO] = {0};
    int ret = 0, live, net_input, have_audio = 0, hw_cuda = 0;
    int aborted = 0, r, si, k, kk, n_copy_inputs = 0;
    AVRational out_fps;
    PassStream pass[PTV_MAX_PASS]; int n_pass = 0;
    /* aliases to input 0 (the shared single-input setup code works on it) */
    AVCodecContext *vdec; AVStream *vist; int vstream; const AVCodec *vdecoder;

    if (n_input < 1) { av_log(NULL, AV_LOG_ERROR, "no input\n"); return AVERROR(EINVAL); }
    if (n_input > PTV_MAX_INPUT || n_input == 3) {
        av_log(NULL, AV_LOG_ERROR, "multiview supports 1, 2 or 4 inputs (got %d)\n", n_input);
        return AVERROR(EINVAL);
    }
    multiview = n_input > 1;
    if (multiview && !fcomplex) {
        av_log(NULL, AV_LOG_ERROR, "multiview (%d inputs) requires -filter_complex (mosaic graph)\n", n_input);
        return AVERROR(EINVAL);
    }
    if (n_rung > PTV_MAX_RUNG) {
        av_log(NULL, AV_LOG_WARNING, "%d outputs > max %d; using the first %d\n", n_rung, PTV_MAX_RUNG, PTV_MAX_RUNG);
        n_rung = PTV_MAX_RUNG;
    }
    memset(inputs, 0, sizeof inputs); memset(as, 0, sizeof as); memset(rung, 0, sizeof rung);
    memset(&comp, 0, sizeof comp);
    for (k = 0; k < n_input; k++) {
        inputs[k].url = ins->groups[k].arg;
        inputs[k].h0  = AV_NOPTS_VALUE;
        pthread_mutex_init(&inputs[k].h0_lock, NULL);
        pthread_mutex_init(&inputs[k].hold.lock, NULL);
        if (multiview) {                         /* per-input jitter buffer for the compositor */
            if ((ret = av_thread_message_queue_alloc(&inputs[k].hold.q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
            av_thread_message_queue_set_free_func(inputs[k].hold.q, free_frame_msg);
        }
        /* Take raw 33-bit timestamps; demux_unwrap extends them (libav's
         * correct_ts_overflow extends inconsistently across the B-frame reorder). */
        av_dict_set(&ins->groups[k].format_opts, "correct_ts_overflow", "0", AV_DICT_DONT_OVERWRITE);
    }

    /* open ALL inputs in parallel: a dead/slow slot must not delay the others,
     * and a serial open would block on its (long, multiview) rw_timeout. */
    {
        OpenArg oa[PTV_MAX_INPUT];
        pthread_t th[PTV_MAX_INPUT];
        int started[PTV_MAX_INPUT] = {0};
        for (k = 0; k < n_input; k++) {
            oa[k].in = &inputs[k]; oa[k].opts = &ins->groups[k].format_opts;
            if (pthread_create(&th[k], NULL, open_input_thread, &oa[k]) == 0) started[k] = 1;
            else { inputs[k].open_ret = AVERROR(errno); }
        }
        for (k = 0; k < n_input; k++) if (started[k]) pthread_join(th[k], NULL);
        for (k = 0; k < n_input; k++) {
            if (inputs[k].open_ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "cannot open input %d '%s': %s\n",
                       k, inputs[k].url, av_err2str(inputs[k].open_ret));
                ret = inputs[k].open_ret; goto end;
            }
            av_dump_format(inputs[k].ifmt, k, inputs[k].url, 0);
        }
    }

    /* resolve each output group's -map/-c into its transcode/copy selection */
    for (r = 0; r < n_rung; r++)
        if ((ret = resolve_plan(inputs, n_input, &outs->groups[r], &sel[r])) < 0) goto end;

    /* per-input video decoder. Ladder/mosaic rungs map filter labels [vN]; each
     * input's source video is the best video stream (single-input may also map an
     * input video directly via -map). */
    for (k = 0; k < n_input; k++) {
        int vs; const AVCodec *vd = NULL;
        if (k == 0 && sel[0].vstream >= 0) { vs = sel[0].vstream; vd = sel[0].vdec; }
        else vs = av_find_best_stream(inputs[k].ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &vd, 0);
        if (vs < 0 || !vd) { av_log(NULL, AV_LOG_ERROR, "no video stream in input %d\n", k); ret = AVERROR(EINVAL); goto end; }
        inputs[k].vstream  = vs;
        inputs[k].vdecoder = vd;
        inputs[k].vist     = inputs[k].ifmt->streams[vs];
        inputs[k].ist_tb   = inputs[k].vist->time_base;
        inputs[k].vdec = avcodec_alloc_context3(vd);
        if (!inputs[k].vdec) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_to_context(inputs[k].vdec, inputs[k].vist->codecpar);
        inputs[k].vdec->pkt_timebase = inputs[k].ist_tb;
        /* NOTE: keep single-threaded decode — frame-threaded hangs offline at EOF. */
        if ((ret = avcodec_open2(inputs[k].vdec, vd, NULL)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "open video decoder (input %d): %s\n", k, av_err2str(ret)); goto end;
        }
        vdecs[k] = inputs[k].vdec;
        inputs[k].wrap_off  = av_calloc(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_off));
        inputs[k].wrap_last = av_malloc_array(inputs[k].ifmt->nb_streams, sizeof(*inputs[k].wrap_last));
        if (!inputs[k].wrap_off || !inputs[k].wrap_last) { ret = AVERROR(ENOMEM); goto end; }
        for (si = 0; si < (int)inputs[k].ifmt->nb_streams; si++) inputs[k].wrap_last[si] = AV_NOPTS_VALUE;
    }
    vdec = inputs[0].vdec; vist = inputs[0].vist;
    vstream = inputs[0].vstream; vdecoder = inputs[0].vdecoder; (void)vstream;

    /* house rate: -r on the first output, else preserve the source's actual FRAME
     * rate. Prefer avg_frame_rate over r_frame_rate: for an interlaced source
     * r_frame_rate is the FIELD rate (e.g. 1080i25 -> 50), but the decoder/
     * deinterlacer emits one frame per coded frame (25), so a 50-fps house clock
     * would tick twice per delivered frame -> ~50% duplicates (judder). Force a
     * specific rate (incl. field-doubling deint) with -r. */
    {
        const char *rate_str = og_get(&outs->groups[0], "r");
        if (rate_str) {
            if (av_parse_video_rate(&out_fps, rate_str) < 0 || out_fps.num <= 0) {
                av_log(NULL, AV_LOG_ERROR, "bad -r '%s'\n", rate_str); ret = AVERROR(EINVAL); goto end;
            }
        } else {
            out_fps = vist->avg_frame_rate.num ? vist->avg_frame_rate
                    : vist->r_frame_rate.num ? vist->r_frame_rate : (AVRational){25, 1};
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

    /* shared filter graph: -filter_complex (N video inputs -> split/mosaic ->
     * N sinks), a single -filter:v chain (single-input N==1), or none (clone the
     * decoded frame to each rung). For multiview the compositor owns the graph
     * and feeds N buffersrcs at pts = house tick (src_tb = 1/out_fps); for
     * single-input the decode thread feeds the one buffersrc with source pts. */
    inputs[0].dc.n_rung = n_rung;
    if (fcomplex) {
        const char *labels[PTV_MAX_RUNG];
        AVRational src_tb = multiview ? av_inv_q(out_fps) : inputs[0].ist_tb;
        AVFilterGraph **pfg = multiview ? &fg : &inputs[0].dc.fg;
        AVFilterContext **psinks = multiview ? comp.fsink : inputs[0].dc.fsink;
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
        if ((ret = build_filter_complex(fcomplex, vdecs, n_input, src_tb, hw_device,
                                        labels, n_rung, pfg, fsrc, psinks)) < 0) goto end;
        filtering = 1;
        if (multiview) { vsink = comp.fsink; }
        else { inputs[0].dc.fsrc = fsrc[0]; inputs[0].dc.filtering = 1; vsink = inputs[0].dc.fsink; }
    } else if (n_rung == 1 && sel[0].vf) {
        int fw = 0, fh = 0, fpix = AV_PIX_FMT_NONE; AVBufferRef *hf = NULL;
        if ((ret = build_video_filter(&inputs[0].dc, vdec, inputs[0].ist_tb, sel[0].vf, 0, 0, 0,
                                      hw_cuda, hw_device, &fw, &fh, &fpix, &hf)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "build video filter: %s\n", av_err2str(ret)); goto end;
        }
        av_buffer_unref(&hf);
        filtering = inputs[0].dc.filtering; vsink = inputs[0].dc.fsink;
    }   /* else: filtering stays 0 -> clone the decoded frame to each rung */

    /* per-rung video encoder, sized from this rung's sink (or the decoder) */
    for (r = 0; r < n_rung; r++) {
        OptionGroup *g = &outs->groups[r];
        const char *out_url = g->arg, *out_fmt = og_get(g, "f");
        const char *venc_name = sel[r].venc ? sel[r].venc : "h264_videotoolbox";
        const AVCodec *vencoder;

        if (filtering) {
            rung[r].fw   = av_buffersink_get_w(vsink[r]);
            rung[r].fh   = av_buffersink_get_h(vsink[r]);
            rung[r].fpix = av_buffersink_get_format(vsink[r]);
            { AVBufferRef *hf = av_buffersink_get_hw_frames_ctx(vsink[r]);
              rung[r].fhwfr = hf ? av_buffer_ref(hf) : NULL; }
        } else {
            rung[r].fw = vdec->width; rung[r].fh = vdec->height;
            rung[r].fpix = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
        }

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

    /* shared audio: for EACH transcoded audio track (sel[0].aout[]), decode +
     * loudness-filter ONCE, then encode PER RUNG (per-rung -b:a) into an output
     * audio stream in EACH muxer. Source stream/decoder/codec from the first
     * rung's plan; per-rung -b:a from each rung. Audio COPY (AC-3 5.1, 2ch) rides
     * the passthrough list below (keeping demux_pass's wrap unwrap + DTS clamp). */
    for (k = 0; k < sel[0].n_aout && n_audio < PTV_MAX_AUDIO; k++) {
        AOutSpec      *spec = &sel[0].aout[k];
        AVStream      *kist = inputs[spec->input].ifmt->streams[spec->stream];
        const AVCodec *kdecoder = spec->adec ? spec->adec
                                : avcodec_find_decoder(kist->codecpar->codec_id);
        AVCodecContext *kdec, *encs[PTV_MAX_RUNG] = {0};
        AudioState    *a;
        enum AVSampleFormat sfmt;
        const char    *af;
        AVChannelLayout ochl;
        int            nch = spec->ac > 0 ? spec->ac : 2;   /* -ac:a:N output channels (default stereo) */
        int            eok = 1;

        if (!kdecoder) { av_log(NULL, AV_LOG_WARNING, "audio track %d (stream %d): no decoder; skipped\n", k, spec->stream); continue; }
        av_channel_layout_default(&ochl, nch);              /* 2->stereo, 6->5.1, 1->mono */
        kdec = avcodec_alloc_context3(kdecoder);
        if (!kdec) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_to_context(kdec, kist->codecpar);
        kdec->pkt_timebase = kist->time_base;
        if (avcodec_open2(kdec, kdecoder, NULL) < 0) {
            av_log(NULL, AV_LOG_WARNING, "audio track %d decoder failed; skipped\n", k);
            avcodec_free_context(&kdec); continue;
        }
        /* open all rung encoders into a temp array FIRST — so a mid-rung failure
         * leaves no orphan output streams in the earlier muxers. */
        for (r = 0; r < n_rung; r++) {
            const AVCodec *aenc = avcodec_find_encoder_by_name(spec->aenc ? spec->aenc : "aac");
            const char *abr = (k < sel[r].n_aout) ? sel[r].aout[k].abr : NULL;
            AVDictionary *aopts = NULL; AVCodecContext *e;
            if (!aenc) aenc = avcodec_find_encoder_by_name("aac");
            e = avcodec_alloc_context3(aenc);
            if (!e) { ret = AVERROR(ENOMEM); avcodec_free_context(&kdec); for (si = 0; si < r; si++) avcodec_free_context(&encs[si]); goto end; }
            e->sample_rate = 48000;
            e->ch_layout   = ochl;                          /* from -ac:a:N (stereo / 5.1 / …) */
            e->sample_fmt  = aenc->sample_fmts ? aenc->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
            e->bit_rate    = 160000;
            e->time_base   = (AVRational){1, 48000};
            if (rung[r].ofmt->oformat->flags & AVFMT_GLOBALHEADER) e->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            if (abr) av_dict_set(&aopts, "b", abr, 0);
            if (avcodec_open2(e, aenc, &aopts) < 0) {
                av_log(NULL, AV_LOG_WARNING, "audio track %d encoder (rung %d) failed; track skipped\n", k, r);
                avcodec_free_context(&e); av_dict_free(&aopts); eok = 0; break;
            }
            av_dict_free(&aopts);
            encs[r] = e;
        }
        if (!eok) { for (r = 0; r < n_rung; r++) avcodec_free_context(&encs[r]); avcodec_free_context(&kdec); continue; }

        /* commit: this is audio track n_audio (dense). Add an output stream per rung. */
        a = &as[n_audio];
        a->dec = kdec;
        for (r = 0; r < n_rung; r++) {
            AVStream *aos; AVDictionaryEntry *klang;
            a->enc[r] = encs[r];
            aos = avformat_new_stream(rung[r].ofmt, NULL);
            if (!aos) { ret = AVERROR(ENOMEM); goto end; }
            avcodec_parameters_from_context(aos->codecpar, encs[r]);
            aos->time_base = encs[r]->time_base;
            if ((klang = av_dict_get(kist->metadata, "language", NULL, 0)))
                av_dict_set(&aos->metadata, "language", klang->value, 0);
            apply_stream_meta(&outs->groups[r], 'a', k, aos);   /* CLI -metadata:s:a:N / -disposition:a:N (G5) */
            a->ost[r] = aos;
        }
        sfmt = a->enc[0]->sample_fmt;
        af   = spec->filter;                              /* -filter:a:N (per-track); else global -af */
        if (!af) { af = og_get(&outs->groups[0], "af"); if (!af) af = og_get(&outs->groups[0], "filter:a"); }
        /* No -af: still route through aresample=async so the audio rides the HOUSE
         * clock (raw swr is identity 48k->48k with no resampler to stretch -> drifts).
         * The common-mode house-skew in audio_push then keeps A/V locked. */
        if (!af) af = "aresample=async=1000";
        a->out_chl    = ochl;                             /* from -ac:a:N (stereo / 5.1 / …) */
        a->out_rate   = 48000;
        a->out_sfmt   = sfmt;
        a->n_out      = n_rung;
        a->frame_size = a->enc[0]->frame_size > 0 ? a->enc[0]->frame_size : 1024;
        a->fifo       = av_audio_fifo_alloc(sfmt, 2, a->frame_size);
        a->ist_tb     = kist->time_base;
        if (af && build_audio_filter(a, a->dec, kist->time_base, af, sfmt) < 0) {
            av_log(NULL, AV_LOG_WARNING, "audio track %d filtergraph failed; plain resample\n", k);
            avfilter_graph_free(&a->afg); a->use_fg = 0;
        }
        if (!a->use_fg) {              /* no -af (or graph failed): plain resample */
            swr_alloc_set_opts2(&a->swr, &a->out_chl, sfmt, 48000,
                                &a->dec->ch_layout, a->dec->sample_fmt, a->dec->sample_rate, 0, NULL);
            if (!a->swr || swr_init(a->swr) < 0) { av_log(NULL, AV_LOG_WARNING, "audio track %d swr init failed; skipped\n", k); }
        }
        asrc[n_audio]    = spec->stream;
        asrc_in[n_audio] = spec->input;
        n_audio++;
    }
    have_audio = n_audio > 0;

    /* shared passthrough (copy): each non-transcoded input stream — extra audio
     * (AC-3 5.1), DVB subtitles, data/SCTE-35 — gets an output stream in EVERY
     * muxer; the copied packets fan out. Built GROUPED BY INPUT so each input's
     * demux gets a contiguous pass[] slice (inputs[kk].da.pass). Created before
     * the headers are written. */
    for (kk = 0; kk < n_input; kk++) {
        inputs[kk].da.pass = &pass[n_pass];          /* this input's contiguous slice */
        inputs[kk].da.n_pass = 0;
        for (si = 0; si < sel[0].n_copy && n_pass < PTV_MAX_PASS; si++) {
            int sidx;
            AVStream *ist; AVDictionaryEntry *lang;
            if (sel[0].copy_input[si] != kk) continue;
            sidx = sel[0].copy[si];
            ist  = inputs[kk].ifmt->streams[sidx];
            lang = av_dict_get(ist->metadata, "language", NULL, 0);
            pass[n_pass].input    = kk;
            pass[n_pass].in_index = sidx;
            pass[n_pass].in_tb    = ist->time_base;
            pass[n_pass].last_dts = AV_NOPTS_VALUE;
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
            inputs[kk].da.n_pass++;
        }
        if (inputs[kk].da.n_pass > 0) n_copy_inputs++;
    }
    if (n_pass)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: passthrough %d stream(s) per output (copy), %d input(s)\n",
               n_pass, n_copy_inputs);

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

    net_input = is_net_url(inputs[0].url);
    live = mode < 0 ? net_input : mode;

    /* queues: per-input video_q, one audio_q per transcoded track, per-rung frame_q + mux_q */
    for (k = 0; k < n_input; k++) {
        if ((ret = av_thread_message_queue_alloc(&inputs[k].video_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(inputs[k].video_q, free_pkt_msg);
    }
    for (k = 0; k < n_audio; k++) {
        if ((ret = av_thread_message_queue_alloc(&audio_q[k], PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(audio_q[k], free_pkt_msg);
    }
    for (r = 0; r < n_rung; r++) {
        if ((ret = av_thread_message_queue_alloc(&rung[r].frame_q, PTV_FRAME_QDEPTH, sizeof(AVFrame *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].frame_q, free_frame_msg);
        if ((ret = av_thread_message_queue_alloc(&rung[r].mux_q, PTV_QDEPTH, sizeof(AVPacket *))) < 0) goto end;
        av_thread_message_queue_set_free_func(rung[r].mux_q, free_pkt_msg);
    }

    /* per-input decode side. single-input: inputs[0].dc already holds the graph
     * (fg/fsrc/fsink/filtering) + feeds the rung frame_q inline. multiview: each
     * decode stages into hold; the compositor owns the graph + frame_q fan. */
    for (k = 0; k < n_input; k++) {
        DecodeCtx *d = &inputs[k].dc;
        d->video_q = inputs[k].video_q; d->vdec = inputs[k].vdec; d->ist_tb = inputs[k].ist_tb;
        d->h0 = &inputs[k].h0; d->h0_lock = &inputs[k].h0_lock; d->live = live;
        if (multiview) { d->hold = &inputs[k].hold; d->filtering = 0; d->n_rung = 0; }
        else { d->n_rung = n_rung; for (r = 0; r < n_rung; r++) d->frame_q[r] = rung[r].frame_q; }
    }

    if (multiview) {                                 /* compositor = the video house clock */
        comp.inputs = inputs; comp.n_input = n_input; comp.fg = fg; comp.n_rung = n_rung;
        comp.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        comp.live = live;
        comp.slate_after_us = live ? 5 * (int64_t)AV_TIME_BASE : 0;   /* stale cell -> black after 5s */
        for (k = 0; k < n_input; k++) comp.fsrc[k] = fsrc[k];
        for (r = 0; r < n_rung; r++) { comp.fsink[r] = vsink[r]; comp.frame_q[r] = rung[r].frame_q; }
    }

    /* per-rung output side */
    for (r = 0; r < n_rung; r++) {
        VideoCtx *vc = &rung[r].vc;
        vc->frame_q = rung[r].frame_q; vc->mux_q = rung[r].mux_q; vc->venc = rung[r].venc;
        vc->out_tb = filtering ? av_buffersink_get_time_base(vsink[r]) : inputs[0].ist_tb;
        vc->tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
        vc->live = live; vc->passthrough = multiview;
        vc->h0 = &inputs[0].h0; vc->h0_lock = &inputs[0].h0_lock;
        vc->house_skew = &inputs[0].house_skew;
        vc->is_master = (r == 0);
        vc->dbg_video_q = inputs[0].video_q; vc->dbg_dec_frames = &inputs[0].dc.dec_frames; vc->dbg_vcorrupt = &inputs[0].dc.vcorrupt;
        rung[r].ma.ofmt = rung[r].ofmt; rung[r].ma.mux_q = rung[r].mux_q;
        rung[r].ma.n_producers = 1 + n_audio + n_copy_inputs;   /* video out + N audio + per-input copy fan */
    }
    for (k = 0; k < n_audio; k++) {              /* per-track audio: source from its input's clock */
        as[k].audio_q = audio_q[k];
        as[k].h0 = &inputs[asrc_in[k]].h0; as[k].h0_lock = &inputs[asrc_in[k]].h0_lock;
        as[k].house_skew = &inputs[asrc_in[k]].house_skew;
        for (r = 0; r < n_rung; r++) as[k].mux_q[r] = rung[r].mux_q;
    }
    for (kk = 0; kk < n_input; kk++) {           /* per-input demux args (pass/n_pass set in copy loop) */
        DemuxArgs *d = &inputs[kk].da;
        d->ifmt = inputs[kk].ifmt; d->video_q = inputs[kk].video_q;
        d->vstream = inputs[kk].vstream; d->drop = is_net_url(inputs[kk].url); d->n_out = n_rung;
        d->h0 = &inputs[kk].h0; d->h0_lock = &inputs[kk].h0_lock; d->house_skew = &inputs[kk].house_skew;
        d->wrap_off = inputs[kk].wrap_off; d->wrap_last = inputs[kk].wrap_last;
        for (r = 0; r < n_rung; r++) d->mux_q[r] = rung[r].mux_q;
        d->n_audio = 0;
        for (k = 0; k < n_audio; k++)
            if (asrc_in[k] == kk) { d->audio_q[d->n_audio] = audio_q[k]; d->astream[d->n_audio] = asrc[k]; d->n_audio++; }
    }

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %s %d input(s) %d rung(s)  house %d/%d fps (%s)  v:%s->enc  a:%s  in:%s  pull-pipeline\n",
        multiview ? "MULTIVIEW" : "single", n_input, n_rung, out_fps.num, out_fps.den,
        live ? "live" : "offline", vdecoder->name,
        have_audio ? "aac" : "none", net_input ? "net(drop)" : "file(block)");
    for (r = 0; r < n_rung; r++)
        av_log(NULL, AV_LOG_INFO, "  rung%d: %dx%d -> %s [%s]\n",
               r, rung[r].fw, rung[r].fh, outs->groups[r].arg, rung[r].ofmt->oformat->name);

    /* spawn: N mux + N output + N watchdog + (per input) decode + demux + 1 audio
     * per track + (multiview) 1 compositor. */
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
    if (multiview) {
        int pe = pthread_create(&th_compositor, NULL, compositor_thread, &comp);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        started_compositor = 1;
    }
    for (k = 0; k < n_input; k++) {
        int pe = pthread_create(&inputs[k].th_decode, NULL, decode_thread, &inputs[k].dc);
        if (pe) { ret = AVERROR(pe); aborted = 1; goto shutdown; }
        inputs[k].started_decode = 1;
    }
    for (k = 0; k < n_audio; k++) {
        if (!pthread_create(&th_audio[k], NULL, audio_thread, &as[k])) started_audio[k] = 1;
        else {                                      /* this track produces nothing: send its mux EOF */
            av_log(NULL, AV_LOG_WARNING, "audio thread %d create failed\n", k);
            for (r = 0; r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
        }
    }
    for (k = 0; k < n_input; k++) {
        int pe = pthread_create(&inputs[k].th_demux, NULL, demux_thread, &inputs[k].da);
        if (pe) {                                   /* couldn't start this demux: EOF its consumers */
            av_thread_message_queue_set_err_recv(inputs[k].video_q, AVERROR_EOF);
            { int t; for (t = 0; t < inputs[k].da.n_audio; t++) av_thread_message_queue_set_err_recv(inputs[k].da.audio_q[t], AVERROR_EOF); }
            for (r = 0; inputs[k].da.n_pass > 0 && r < n_rung; r++) { AVPacket *eof = NULL; av_thread_message_queue_send(rung[r].mux_q, &eof, 0); }
            ret = AVERROR(pe);
        } else inputs[k].started_demux = 1;
    }
    for (k = 0; k < n_input; k++) if (inputs[k].started_demux) pthread_join(inputs[k].th_demux, NULL);

shutdown:
    if (aborted) {                                  /* force the pipeline to unwind: release ANY
                                                     * thread blocked in send OR recv on any queue */
        for (k = 0; k < n_input; k++) if (inputs[k].video_q) {
            av_thread_message_queue_set_err_send(inputs[k].video_q, AVERROR_EOF);
            av_thread_message_queue_set_err_recv(inputs[k].video_q, AVERROR_EOF);
        }
        for (k = 0; k < n_audio; k++) if (audio_q[k]) {
            av_thread_message_queue_set_err_send(audio_q[k], AVERROR_EOF);
            av_thread_message_queue_set_err_recv(audio_q[k], AVERROR_EOF);
        }
        for (r = 0; r < n_rung; r++) {
            rung[r].vc.output_done = 1;             /* stop the watchdog */
            if (rung[r].frame_q) {
                av_thread_message_queue_set_err_send(rung[r].frame_q, AVERROR_EOF);
                av_thread_message_queue_set_err_recv(rung[r].frame_q, AVERROR_EOF);
            }
            if (rung[r].mux_q) {
                av_thread_message_queue_set_err_send(rung[r].mux_q, AVERROR_EOF);
                av_thread_message_queue_set_err_recv(rung[r].mux_q, AVERROR_EOF);
            }
        }
    }
    for (k = 0; k < n_input; k++) if (inputs[k].started_decode) pthread_join(inputs[k].th_decode, NULL);
    if (started_compositor) pthread_join(th_compositor, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_output) pthread_join(rung[r].th_output, NULL);
    for (k = 0; k < n_audio; k++) if (started_audio[k]) pthread_join(th_audio[k], NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_wd) pthread_join(rung[r].th_wd, NULL);
    for (r = 0; r < n_rung; r++) if (rung[r].started_mux) {
        if (!ret && rung[r].ma.err < 0) ret = rung[r].ma.err;
        pthread_join(rung[r].th_mux, NULL);
    }

    {
        int64_t dec_sum = 0, vpkt = 0, ppkt = 0;
        int64_t m_emit = multiview ? comp.emitted : rung[0].vc.emitted;
        int64_t m_dup  = multiview ? comp.dup     : rung[0].vc.dup;
        for (k = 0; k < n_input; k++) { dec_sum += inputs[k].dc.dec_frames; vpkt += inputs[k].da.vpkt; ppkt += inputs[k].da.ppkt; }
        av_log(NULL, AV_LOG_INFO,
            "ptvencoder: done — %d input(s) %d rung(s); video dec %"PRId64" out %"PRId64
            " (dup %"PRId64")  demux v:%"PRId64" p:%"PRId64"%s\n",
            n_input, n_rung, dec_sum, m_emit, m_dup, vpkt, ppkt, have_audio ? "" : "  [no audio]");
    }
    if (have_audio) {
        int64_t ain = 0, aout = 0, apkt = 0;
        for (k = 0; k < n_audio; k++) { ain += as[k].in_frames; aout += as[k].out_frames; }
        for (k = 0; k < n_input; k++) apkt += inputs[k].da.apkt;
        av_log(NULL, AV_LOG_INFO, "ptvencoder: audio %d track(s), in %"PRId64" frames, out %"PRId64" aac (demux a:%"PRId64")\n",
               n_audio, ain, aout, apkt);
    }
    if (ret > 0) ret = 0;

end:
    for (r = 0; r < n_rung; r++) {
        if (rung[r].hdr_written) av_write_trailer(rung[r].ofmt);
        if (rung[r].ofmt && !(rung[r].ofmt->oformat->flags & AVFMT_NOFILE) && rung[r].ofmt->pb)
            avio_closep(&rung[r].ofmt->pb);
    }
    for (k = 0; k < n_audio; k++) av_thread_message_queue_free(&audio_q[k]);
    for (r = 0; r < n_rung; r++) {
        av_thread_message_queue_free(&rung[r].frame_q);
        av_thread_message_queue_free(&rung[r].mux_q);
        avcodec_free_context(&rung[r].venc);
        av_buffer_unref(&rung[r].fhwfr);
        if (rung[r].ofmt) avformat_free_context(rung[r].ofmt);
    }
    for (k = 0; k < n_audio; k++) {
        if (as[k].swr)  swr_free(&as[k].swr);
        if (as[k].fifo) av_audio_fifo_free(as[k].fifo);
        avfilter_graph_free(&as[k].afg);
        for (r = 0; r < n_rung; r++) avcodec_free_context(&as[k].enc[r]);
        avcodec_free_context(&as[k].dec);
    }
    avfilter_graph_free(&fg);
    for (k = 0; k < n_input; k++) {
        avfilter_graph_free(&inputs[k].dc.fg);       /* single-input graph (multiview: NULL) */
        avcodec_free_context(&inputs[k].vdec);
        av_thread_message_queue_free(&inputs[k].video_q);
        av_thread_message_queue_free(&inputs[k].hold.q);
        av_freep(&inputs[k].wrap_off);
        av_freep(&inputs[k].wrap_last);
        if (inputs[k].ifmt) avformat_close_input(&inputs[k].ifmt);
        pthread_mutex_destroy(&inputs[k].h0_lock);
        pthread_mutex_destroy(&inputs[k].hold.lock);
    }
    av_buffer_unref(&hw_device);
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
    { "hide_banner",      OPT_TYPE_BOOL,   0,                        { .off = 0 }, "suppress startup banner" },
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

/* Apply CLI -metadata:s:<t>:idx (possibly several) and -disposition:<t>:idx onto
 * an output stream (G5). metadata overrides the source value set earlier;
 * disposition takes ffmpeg's "flag", "+flag+flag" or "0"/"none" forms. */
static void apply_stream_meta(OptionGroup *g, char t, int idx, AVStream *ost)
{
    char key[24]; int i;
    const char *disp = og_spec(g, "disposition", t, idx);
    if (disp) {
        if (!strcmp(disp, "0") || !strcmp(disp, "none")) {
            ost->disposition = 0;
        } else {
            char buf[128], *tok, *sp = NULL; int d = 0;
            snprintf(buf, sizeof buf, "%s", disp);
            for (tok = strtok_r(buf, "+", &sp); tok; tok = strtok_r(NULL, "+", &sp)) {
                int f = av_disposition_from_string(tok);
                if (f > 0) d |= f;
            }
            ost->disposition = d;
        }
    }
    snprintf(key, sizeof key, "metadata:s:%c:%d", t, idx);     /* -metadata:s:a:N key=val (repeatable) */
    for (i = 0; i < g->nb_opts; i++) {
        char kv[256], *eq;
        if (strcmp(g->opts[i].key, key)) continue;
        snprintf(kv, sizeof kv, "%s", g->opts[i].val);
        if ((eq = strchr(kv, '='))) { *eq = 0; av_dict_set(&ost->metadata, kv, eq + 1, 0); }
    }
}

/* strip a leading "<digits>:" file index and a trailing '?' from a -map value */
static const char *map_spec(const char *v, char *buf, size_t bufsz, int *optional, int *file_idx)
{
    const char *colon = strchr(v, ':'), *s = v; size_t L;
    if (file_idx) *file_idx = 0;
    if (colon && colon > v) {
        const char *p; int alldig = 1;
        for (p = v; p < colon; p++) if (*p < '0' || *p > '9') { alldig = 0; break; }
        if (alldig) { if (file_idx) *file_idx = atoi(v); s = colon + 1; }
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
 * No -map -> auto (best video + first <=2ch audio + copy the rest), back-compat
 * (single-input only). Explicit -map K:... selects from input K (multiview). */
static int resolve_plan(Input *inputs, int n_input, OptionGroup *outg, Sel *s)
{
    AVFormatContext *ifmt = inputs[0].ifmt;
    int o, si, tcnt[5] = {0}, nmap = 0, astream = -1;
    memset(s, 0, sizeof *s);
    s->vstream = -1;
    s->vf = og_get(outg, "filter:v"); if (!s->vf) s->vf = og_get(outg, "vf");
    for (o = 0; o < outg->nb_opts; o++) if (!strcmp(outg->opts[o].key, "map")) nmap++;
    s->have = nmap > 0;

    if (!nmap) {                       /* no -map: auto-select on input 0 (back-compat) */
        s->vstream = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &s->vdec, 0);
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            AVCodecParameters *cp = ifmt->streams[si]->codecpar;
            if (cp->codec_type == AVMEDIA_TYPE_AUDIO &&
                cp->ch_layout.nb_channels > 0 && cp->ch_layout.nb_channels <= 2) { astream = si; break; }
        }
        if (astream < 0) astream = av_find_best_stream(ifmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        if (astream >= 0) {            /* one transcoded audio track (back-compat) */
            s->aout[0].input  = 0;
            s->aout[0].stream = astream;
            s->aout[0].adec   = avcodec_find_decoder(ifmt->streams[astream]->codecpar->codec_id);
            s->n_aout = 1;
        }
        for (si = 0; si < (int)ifmt->nb_streams; si++) {
            enum AVMediaType mt = ifmt->streams[si]->codecpar->codec_type;
            if (si == s->vstream || si == astream) continue;
            if ((mt == AVMEDIA_TYPE_AUDIO || mt == AVMEDIA_TYPE_SUBTITLE || mt == AVMEDIA_TYPE_DATA)
                && s->n_copy < PTV_MAX_PASS) { s->copy_input[s->n_copy] = 0; s->copy[s->n_copy++] = si; }
        }
        s->venc = og_get(outg, "c:v"); if (!s->venc) s->venc = og_get(outg, "c");
        s->vbr  = og_get(outg, "b:v"); if (!s->vbr)  s->vbr  = og_get(outg, "b");
        return 0;
    }

    for (o = 0; o < outg->nb_opts; o++) {              /* explicit -map plan */
        char buf[64]; int optional, fidx; const char *spec, *mv = outg->opts[o].val;
        AVFormatContext *kfmt;
        if (strcmp(outg->opts[o].key, "map")) continue;
        if (mv[0] == '[') {                            /* filter-output label = this rung's video */
            int idx = tcnt[0]++;                       /* video output index */
            s->venc = og_spec(outg, "c", 'v', idx);    /* -c:v / -c:v:0 (NULL -> default encoder) */
            s->vbr  = og_spec(outg, "b", 'v', idx);    /* -b:v / -b:v:0 */
            continue;                                  /* video comes from the graph, not an input stream */
        }
        spec = map_spec(mv, buf, sizeof buf, &optional, &fidx);
        if (fidx < 0 || fidx >= n_input) {
            av_log(NULL, AV_LOG_ERROR, "-map %s: input %d out of range (%d input(s))\n", mv, fidx, n_input);
            return AVERROR(EINVAL);
        }
        kfmt = inputs[fidx].ifmt;
        for (si = 0; si < (int)kfmt->nb_streams; si++) {
            enum AVMediaType mt; char t; int ti, idx; const char *codec;
            if (avformat_match_stream_specifier(kfmt, kfmt->streams[si], spec) <= 0) continue;
            mt  = kfmt->streams[si]->codecpar->codec_type;
            t   = mt==AVMEDIA_TYPE_VIDEO?'v':mt==AVMEDIA_TYPE_AUDIO?'a':
                  mt==AVMEDIA_TYPE_SUBTITLE?'s':mt==AVMEDIA_TYPE_DATA?'d':'?';
            ti  = mt==AVMEDIA_TYPE_VIDEO?0:mt==AVMEDIA_TYPE_AUDIO?1:
                  mt==AVMEDIA_TYPE_SUBTITLE?2:mt==AVMEDIA_TYPE_DATA?3:4;
            idx = tcnt[ti]++;
            codec = og_spec(outg, "c", t, idx);
            if (codec && !strcmp(codec, "copy")) {
                if (s->n_copy < PTV_MAX_PASS) { s->copy_input[s->n_copy] = fidx; s->copy[s->n_copy++] = si; }
            } else if (mt == AVMEDIA_TYPE_VIDEO && s->vstream < 0 && fidx == 0) {
                s->vstream = si; s->venc = codec; s->vbr = og_spec(outg, "b", t, idx);
                s->vdec = avcodec_find_decoder(kfmt->streams[si]->codecpar->codec_id);
            } else if (mt == AVMEDIA_TYPE_AUDIO && s->n_aout < PTV_MAX_AUDIO) {
                AOutSpec *a = &s->aout[s->n_aout++];    /* one transcoded audio track per -map */
                const char *acs = og_spec(outg, "ac", t, idx);   /* -ac:a:N output channels */
                a->input = fidx; a->stream = si; a->aenc = codec; a->abr = og_spec(outg, "b", t, idx);
                a->filter = og_spec(outg, "filter", t, idx);     /* -filter:a:N (else global -af) */
                a->ac = acs ? atoi(acs) : 0;                      /* 0 = default stereo */
                a->adec = avcodec_find_decoder(kfmt->streams[si]->codecpar->codec_id);
            } else if (s->n_copy < PTV_MAX_PASS) {
                s->copy_input[s->n_copy] = fidx; s->copy[s->n_copy++] = si;   /* over the cap -> copy */
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
        spec = map_spec(outg->opts[o].val, buf, sizeof buf, &optional, NULL);
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

/* ffmpeg-style startup banner: product name + FFmpeg build id + lib versions.
 * Emitted via av_log (so the [timestamp] prefix and -loglevel gating apply, like
 * the rest of the output); suppressed by -hide_banner. */
static void ptv_show_banner(void)
{
    av_log(NULL, AV_LOG_INFO, "Perception TV Encoder (ptvencoder) %s  FFmpeg %s\n", PTVENCODER_VERSION, av_version_info());
    av_log(NULL, AV_LOG_INFO, "  libavutil      %u.%u.%u\n",
           AV_VERSION_MAJOR(avutil_version()), AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
    av_log(NULL, AV_LOG_INFO, "  libavcodec     %u.%u.%u\n",
           AV_VERSION_MAJOR(avcodec_version()), AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
    av_log(NULL, AV_LOG_INFO, "  libavformat    %u.%u.%u\n",
           AV_VERSION_MAJOR(avformat_version()), AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
    av_log(NULL, AV_LOG_INFO, "  libavfilter    %u.%u.%u\n",
           AV_VERSION_MAJOR(avfilter_version()), AV_VERSION_MINOR(avfilter_version()), AV_VERSION_MICRO(avfilter_version()));
    av_log(NULL, AV_LOG_INFO, "  libswresample  %u.%u.%u\n",
           AV_VERSION_MAJOR(swresample_version()), AV_VERSION_MINOR(swresample_version()), AV_VERSION_MICRO(swresample_version()));
}

int main(int argc, char **argv)
{
    OptionParseContext octx;
    OptionGroupList *ins;
    OptionGroupList *outs;
    const char *fcomplex = NULL, *hwdev = NULL;
    int mode = -1, ret, gi, hide_banner = 0;

    init_dynload();
    av_log_set_level(AV_LOG_INFO);
    g_diag = !!getenv("PTV_DIAG");
    if (getenv("PTV_NO_AVLOCK")) g_avlock = 0;   /* revert to source-locked audio (drifts on dup) */
    { const char *s = getenv("PTV_SLOW_US"); g_slow = s ? atoi(s) : 0; }
    if (getenv("PTV_LOG_TS") && atoi(getenv("PTV_LOG_TS")))   /* native [timestamp] log prefix */
        av_log_set_callback(ptv_log_ts_callback);

    if (argc >= 2 && (!strcmp(argv[1], "-version") || !strcmp(argv[1], "--version"))) {
        printf("Perception TV Encoder (ptvencoder) %s  FFmpeg %s\n", PTVENCODER_VERSION, av_version_info());
        printf("  libavutil      %u.%u.%u\n", AV_VERSION_MAJOR(avutil_version()),
               AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
        printf("  libavcodec     %u.%u.%u\n", AV_VERSION_MAJOR(avcodec_version()),
               AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
        printf("  libavformat    %u.%u.%u\n", AV_VERSION_MAJOR(avformat_version()),
               AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
        printf("  libavfilter    %u.%u.%u\n", AV_VERSION_MAJOR(avfilter_version()),
               AV_VERSION_MINOR(avfilter_version()), AV_VERSION_MICRO(avfilter_version()));
        printf("  libswresample  %u.%u.%u\n", AV_VERSION_MAJOR(swresample_version()),
               AV_VERSION_MINOR(swresample_version()), AV_VERSION_MICRO(swresample_version()));
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
        if (!strcmp(octx.global_opts.opts[gi].key, "stats_period")) {              /* progress-line interval */
            int64_t p; if (av_parse_time(&p, octx.global_opts.opts[gi].val, 1) >= 0 && p > 0) g_stats_period_us = p;
        }
        if (!strcmp(octx.global_opts.opts[gi].key, "hide_banner")) hide_banner = 1; /* suppress startup banner */
    }
    if (!hide_banner)
        ptv_show_banner();
    if (octx.groups[1].nb_groups < 1 || octx.groups[0].nb_groups < 1) {
        av_log(NULL, AV_LOG_ERROR,
               "usage: ptvencoder [opts] -i <input> [-filter_complex ..] "
               "[-map .. -c:TYPE .. -b:TYPE ..] <output> [<output> ...]\n");
        uninit_parse_context(&octx); return 1;
    }
    ins  = &octx.groups[1];             /* all input groups (one per -i; 1/2/4 = multiview) */
    outs = &octx.groups[0];             /* all output groups (one per ABR rung) */
    ret  = transcode(ins, outs, fcomplex, hwdev, mode);
    uninit_parse_context(&octx);
    return ret < 0 ? 1 : 0;
}
