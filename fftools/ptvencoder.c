/*
 * ptvencoder — purpose-built live MPEG-TS re-encoder.
 *
 * A sibling fftools program (alongside ffmpeg/ffprobe/ffplay) that links the
 * same patched libav* libraries but runs on its own house-clock timing engine.
 * See analysis/ptvencoder-functional-spec.md.
 *
 * Phase 1, increment 2 (step 2): the house clock.
 *   demux -> decode -> [house-clock frame-sync] -> encode -> mux  (video only).
 *
 * House clock: output runs on a steady grid H at the chosen output rate. Each
 * decoded frame is mapped onto H via Mi (source PTS -> house time) and the grid
 * is filled by sample-and-hold (dup when the source is behind H, drop when it
 * is ahead). In live mode the grid is paced to a local wall clock; in offline
 * mode it runs at media time (as fast as I/O allows). Re-anchor of Mi on
 * wrap/jump/gap is the next increment; here Mi is a single affine map with a
 * crude backwards guard.
 *
 * This file is licensed under the same terms as FFmpeg (GPL, --enable-gpl).
 */

#include <stdio.h>
#include <string.h>

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/time.h"
#include "libavutil/parseutils.h"
#include "libavutil/mathematics.h"
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"

#include "cmdutils.h"

const char program_name[] = "ptvencoder";
const int  program_birth_year = 2026;

void show_help_default(const char *opt, const char *arg)
{
    av_log(NULL, AV_LOG_INFO,
        "usage: ptvencoder [options] -i <input> <output>\n"
        "\n"
        "  options:\n"
        "    -i <url>      input (file or udp://...)\n"
        "    -c:v <name>   video encoder (default: h264_videotoolbox, fallback mpeg2video)\n"
        "    -r <rate>     output frame rate / house-clock rate (e.g. 25, 30000/1001); default = source\n"
        "    --mode live|offline   live = wall-clock paced; offline = media-clock. default: auto from input\n"
        "    -version, -h\n"
        "\n"
        "  Phase 1 increment 2: house-clock frame-sync (sample-and-hold), video only.\n");
}

/* Drain encoder, writing packets to the muxer. flush=NULL frame. */
static int encode_write(AVFormatContext *ofmt, AVCodecContext *enc,
                        AVStream *ost, AVFrame *frame, AVPacket *pkt)
{
    int ret = avcodec_send_frame(enc, frame);
    if (ret < 0)
        return ret;
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0)
            return ret;
        av_packet_rescale_ts(pkt, enc->time_base, ost->time_base);
        pkt->stream_index = ost->index;
        ret = av_interleaved_write_frame(ofmt, pkt);
        av_packet_unref(pkt);
        if (ret < 0)
            return ret;
    }
    return 0;
}

/* House-clock state for a single video output. */
typedef struct ClockState {
    AVFormatContext *ofmt;
    AVCodecContext  *enc;
    AVStream        *ost;
    AVRational       ist_tb;       /* input stream time_base */
    int64_t          tick_dur_us;  /* microseconds per output frame (grid step) */
    int              live;         /* 1 = pace to wall clock */
    AVPacket        *pkt;          /* scratch for encode_write */

    int64_t          h0_src_us;    /* AV_NOPTS_VALUE until first frame */
    int64_t          wall0_us;     /* wall clock captured at tick 0 */
    int64_t          next_tick;    /* next house-grid index to emit */
    AVFrame         *held;         /* most recent frame (sample-and-hold) */
    int              have_held;
    int              held_emits;   /* times the current held frame was emitted */

    int64_t          dup, drop, emitted, in_frames;
} ClockState;

/* Emit the held frame at the next grid tick (pacing in live mode). */
static int emit_tick(ClockState *c)
{
    int ret;
    if (c->live) {
        int64_t now, target;
        if (c->emitted == 0)
            c->wall0_us = av_gettime_relative();
        target = c->wall0_us + c->next_tick * c->tick_dur_us;
        now = av_gettime_relative();
        if (now < target)
            av_usleep((unsigned)(target - now));
    }
    c->held->pts      = c->next_tick;
    c->held->pkt_dts  = AV_NOPTS_VALUE;
    c->held->duration = 0;
    ret = encode_write(c->ofmt, c->enc, c->ost, c->held, c->pkt);
    c->next_tick++;
    c->emitted++;
    if (++c->held_emits > 1)
        c->dup++;
    return ret;
}

/* Map a decoded frame onto H and fill the grid up to it by sample-and-hold. */
static int clock_push(ClockState *c, AVFrame *frame)
{
    int ret = 0;
    int64_t ts = frame->best_effort_timestamp;
    int64_t house_us;

    c->in_frames++;

    if (ts != AV_NOPTS_VALUE) {
        int64_t src_us = av_rescale_q(ts, c->ist_tb, AV_TIME_BASE_Q);
        if (c->h0_src_us == AV_NOPTS_VALUE)
            c->h0_src_us = src_us;
        house_us = src_us - c->h0_src_us;
        if (house_us < c->next_tick * c->tick_dur_us)   /* backwards guard (real re-anchor = next increment) */
            house_us = c->next_tick * c->tick_dur_us;
    } else {
        house_us = c->next_tick * c->tick_dur_us;
    }

    /* emit the previous frame for every grid tick before this frame's time */
    while (c->have_held && c->next_tick * c->tick_dur_us < house_us) {
        if ((ret = emit_tick(c)) < 0)
            return ret;
    }

    if (c->have_held && c->held_emits == 0)   /* previous frame never reached an output tick */
        c->drop++;

    av_frame_unref(c->held);
    av_frame_move_ref(c->held, frame);
    c->have_held   = 1;
    c->held_emits  = 0;
    return 0;
}

/* Show the final held frame once so the last frame isn't lost. */
static int clock_flush(ClockState *c)
{
    if (c->have_held && c->held_emits == 0)
        return emit_tick(c);
    return 0;
}

static int transcode_video(const char *in_url, const char *out_url,
                           const char *venc_name, const char *rate_str, int mode)
{
    AVFormatContext *ifmt = NULL, *ofmt = NULL;
    AVCodecContext  *dec  = NULL, *enc  = NULL;
    const AVCodec   *decoder = NULL, *encoder = NULL;
    AVStream        *ist = NULL, *ost = NULL;
    AVPacket        *pkt = NULL;
    AVFrame         *frame = NULL;
    ClockState       cs;
    int vstream = -1, ret = 0, skipped = 0, live;
    AVRational out_fps;

    memset(&cs, 0, sizeof(cs));

    if ((ret = avformat_open_input(&ifmt, in_url, NULL, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open input '%s': %s\n", in_url, av_err2str(ret));
        return ret;
    }
    if ((ret = avformat_find_stream_info(ifmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot read stream info: %s\n", av_err2str(ret));
        goto end;
    }

    vstream = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (vstream < 0) {
        av_log(NULL, AV_LOG_ERROR, "no video stream found\n");
        ret = vstream; goto end;
    }
    ist = ifmt->streams[vstream];
    skipped = ifmt->nb_streams - 1;

    dec = avcodec_alloc_context3(decoder);
    if (!dec) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_to_context(dec, ist->codecpar);
    dec->pkt_timebase = ist->time_base;
    if ((ret = avcodec_open2(dec, decoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open decoder: %s\n", av_err2str(ret));
        goto end;
    }

    if ((ret = avformat_alloc_output_context2(&ofmt, NULL, NULL, out_url)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot create output context: %s\n", av_err2str(ret));
        goto end;
    }

    encoder = avcodec_find_encoder_by_name(venc_name);
    if (!encoder) {
        av_log(NULL, AV_LOG_WARNING, "encoder '%s' not found, falling back to mpeg2video\n", venc_name);
        encoder = avcodec_find_encoder_by_name("mpeg2video");
    }
    if (!encoder) { av_log(NULL, AV_LOG_ERROR, "no usable video encoder\n"); ret = AVERROR_ENCODER_NOT_FOUND; goto end; }

    /* output rate = -r, else source frame rate, else 25 */
    if (rate_str) {
        if (av_parse_video_rate(&out_fps, rate_str) < 0 || out_fps.num <= 0) {
            av_log(NULL, AV_LOG_ERROR, "bad -r value '%s'\n", rate_str); ret = AVERROR(EINVAL); goto end;
        }
    } else {
        out_fps = ist->r_frame_rate.num ? ist->r_frame_rate
                : ist->avg_frame_rate.num ? ist->avg_frame_rate
                : (AVRational){25, 1};
    }

    enc = avcodec_alloc_context3(encoder);
    if (!enc) { ret = AVERROR(ENOMEM); goto end; }
    enc->width     = dec->width;
    enc->height    = dec->height;
    enc->pix_fmt   = dec->pix_fmt != AV_PIX_FMT_NONE ? dec->pix_fmt : AV_PIX_FMT_YUV420P;
    enc->time_base = av_inv_q(out_fps);
    enc->framerate = out_fps;
    enc->bit_rate  = 3000000;
    enc->gop_size  = 2 * (out_fps.num / FFMAX(out_fps.den, 1));
    if (ofmt->oformat->flags & AVFMT_GLOBALHEADER)
        enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if ((ret = avcodec_open2(enc, encoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open encoder '%s': %s\n", encoder->name, av_err2str(ret));
        goto end;
    }

    ost = avformat_new_stream(ofmt, NULL);
    if (!ost) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_from_context(ost->codecpar, enc);
    ost->time_base = enc->time_base;

    if (!(ofmt->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&ofmt->pb, out_url, AVIO_FLAG_WRITE)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "cannot open output '%s': %s\n", out_url, av_err2str(ret)); goto end;
        }
    }
    if ((ret = avformat_write_header(ofmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot write header: %s\n", av_err2str(ret)); goto end;
    }

    live = mode < 0 ? (!strncmp(in_url, "udp://", 6) || !strncmp(in_url, "rtp://", 6) ||
                       !strncmp(in_url, "srt://", 6))
                    : mode;

    cs.ofmt = ofmt; cs.enc = enc; cs.ost = ost; cs.ist_tb = ist->time_base;
    cs.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
    cs.live = live;
    cs.h0_src_us = AV_NOPTS_VALUE;
    cs.held = av_frame_alloc();
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!cs.held || !pkt || !frame) { ret = AVERROR(ENOMEM); goto end; }
    cs.pkt = pkt;

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %dx%d  house clock %d/%d fps (%s)  %s -> %s [%s]  (skip %d non-video)\n",
        enc->width, enc->height, out_fps.num, out_fps.den, live ? "live/wall-clock" : "offline/media",
        decoder->name, encoder->name, ofmt->oformat->name, skipped);

    while (av_read_frame(ifmt, pkt) >= 0) {
        if (pkt->stream_index == vstream) {
            ret = avcodec_send_packet(dec, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(dec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
                if (ret < 0) goto end;
                if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
                if ((ret = clock_push(&cs, frame)) < 0) goto end;   /* consumes frame */
            }
        }
        av_packet_unref(pkt);
    }

    avcodec_send_packet(dec, NULL);   /* flush decoder */
    while (avcodec_receive_frame(dec, frame) >= 0) {
        if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
        clock_push(&cs, frame);
    }
    clock_flush(&cs);
    encode_write(ofmt, enc, ost, NULL, pkt);   /* flush encoder */

    av_write_trailer(ofmt);
    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: done — in %"PRId64" frames, out %"PRId64" (dup %"PRId64", drop %"PRId64")\n",
        cs.in_frames, cs.emitted, cs.dup, cs.drop);
    ret = 0;

end:
    if (ofmt && !(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb)
        avio_closep(&ofmt->pb);
    av_frame_free(&cs.held);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&enc);
    avcodec_free_context(&dec);
    if (ofmt) avformat_free_context(ofmt);
    avformat_close_input(&ifmt);
    return ret;
}

int main(int argc, char **argv)
{
    const char *in_url = NULL, *out_url = NULL, *rate = NULL;
    const char *venc = "h264_videotoolbox";
    int mode = -1;   /* -1 auto, 0 offline, 1 live */
    int i;

    init_dynload();
    av_log_set_level(AV_LOG_INFO);

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

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc)        in_url = argv[++i];
        else if (!strcmp(argv[i], "-c:v") && i + 1 < argc) venc   = argv[++i];
        else if (!strcmp(argv[i], "-r") && i + 1 < argc)   rate   = argv[++i];
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

    return transcode_video(in_url, out_url, venc, rate, mode) < 0 ? 1 : 0;
}
