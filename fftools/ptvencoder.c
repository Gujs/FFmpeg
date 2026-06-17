/*
 * ptvencoder — purpose-built live MPEG-TS re-encoder.
 *
 * A sibling fftools program (alongside ffmpeg/ffprobe/ffplay) that links the
 * same patched libav* libraries but runs on its own house-clock timing engine.
 * See analysis/ptvencoder-functional-spec.md.
 *
 * Phase 1, increment 2 (step 1): minimal video transcode
 *   demux -> decode -> encode -> mux  (video stream only).
 * Output PTS are assigned from a frame counter (even CFR grid) — the crude
 * precursor to the house clock, which replaces this in the next step.
 *
 * This file is licensed under the same terms as FFmpeg (GPL, --enable-gpl).
 */

#include <stdio.h>
#include <string.h>

#include "libavutil/avutil.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
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
        "    -version, -h\n"
        "\n"
        "  Phase 1 increment 2 (step 1): minimal video transcode (no house clock yet).\n");
}

/* Drain encoder, writing packets to the muxer. flush=1 sends a NULL frame. */
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

static int transcode_video(const char *in_url, const char *out_url,
                           const char *venc_name)
{
    AVFormatContext *ifmt = NULL, *ofmt = NULL;
    AVCodecContext  *dec  = NULL, *enc  = NULL;
    const AVCodec   *decoder = NULL, *encoder = NULL;
    AVStream        *ist = NULL, *ost = NULL;
    AVPacket        *pkt = NULL;
    AVFrame         *frame = NULL;
    int vstream = -1, ret = 0, skipped = 0;
    int64_t out_pts = 0;
    AVRational fr;

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
        ret = vstream;
        goto end;
    }
    ist = ifmt->streams[vstream];
    skipped = ifmt->nb_streams - 1;

    /* decoder */
    dec = avcodec_alloc_context3(decoder);
    if (!dec) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_to_context(dec, ist->codecpar);
    dec->pkt_timebase = ist->time_base;
    if ((ret = avcodec_open2(dec, decoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open decoder: %s\n", av_err2str(ret));
        goto end;
    }

    /* output muxer (format guessed from out_url extension) */
    if ((ret = avformat_alloc_output_context2(&ofmt, NULL, NULL, out_url)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot create output context: %s\n", av_err2str(ret));
        goto end;
    }

    /* encoder: try requested name, fall back to mpeg2video */
    encoder = avcodec_find_encoder_by_name(venc_name);
    if (!encoder) {
        av_log(NULL, AV_LOG_WARNING, "encoder '%s' not found, falling back to mpeg2video\n", venc_name);
        encoder = avcodec_find_encoder_by_name("mpeg2video");
    }
    if (!encoder) { av_log(NULL, AV_LOG_ERROR, "no usable video encoder\n"); ret = AVERROR_ENCODER_NOT_FOUND; goto end; }

    fr = ist->r_frame_rate.num ? ist->r_frame_rate
       : ist->avg_frame_rate.num ? ist->avg_frame_rate
       : (AVRational){25, 1};

    enc = avcodec_alloc_context3(encoder);
    if (!enc) { ret = AVERROR(ENOMEM); goto end; }
    enc->width     = dec->width;
    enc->height    = dec->height;
    enc->pix_fmt   = dec->pix_fmt != AV_PIX_FMT_NONE ? dec->pix_fmt : AV_PIX_FMT_YUV420P;
    enc->time_base = av_inv_q(fr);
    enc->framerate = fr;
    enc->bit_rate  = 3000000;
    enc->gop_size  = 2 * (fr.num / FFMAX(fr.den, 1));
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
            av_log(NULL, AV_LOG_ERROR, "cannot open output '%s': %s\n", out_url, av_err2str(ret));
            goto end;
        }
    }
    if ((ret = avformat_write_header(ofmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot write header: %s\n", av_err2str(ret));
        goto end;
    }

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %dx%d %d/%d fps  %s -> %s [%s]  (skipping %d non-video stream(s))\n",
        enc->width, enc->height, fr.num, fr.den, decoder->name, encoder->name,
        ofmt->oformat->name, skipped);

    pkt   = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) { ret = AVERROR(ENOMEM); goto end; }

    while (av_read_frame(ifmt, pkt) >= 0) {
        if (pkt->stream_index == vstream) {
            ret = avcodec_send_packet(dec, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(dec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
                if (ret < 0) goto end;

                /* don't feed garbage to the encoder (mid-GOP join, etc.) */
                if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }

                /* crude even-CFR grid (precursor to the house clock) */
                frame->pts = out_pts++;
                frame->pkt_dts = AV_NOPTS_VALUE;
                frame->duration = 0;
                if ((ret = encode_write(ofmt, enc, ost, frame, pkt)) < 0) {
                    av_frame_unref(frame);
                    goto end;
                }
                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    /* flush decoder then encoder */
    avcodec_send_packet(dec, NULL);
    while (avcodec_receive_frame(dec, frame) >= 0) {
        frame->pts = out_pts++;
        frame->pkt_dts = AV_NOPTS_VALUE;
        frame->duration = 0;
        encode_write(ofmt, enc, ost, frame, pkt);
        av_frame_unref(frame);
    }
    encode_write(ofmt, enc, ost, NULL, pkt);  /* flush encoder */

    av_write_trailer(ofmt);
    av_log(NULL, AV_LOG_INFO, "ptvencoder: done, %"PRId64" frames written\n", out_pts);
    ret = 0;

end:
    if (ofmt && !(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb)
        avio_closep(&ofmt->pb);
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
    const char *in_url = NULL, *out_url = NULL;
    const char *venc = "h264_videotoolbox";
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
        show_help_default(NULL, NULL);
        return 0;
    }

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc)        in_url  = argv[++i];
        else if (!strcmp(argv[i], "-c:v") && i + 1 < argc) venc    = argv[++i];
        else if (argv[i][0] != '-')                        out_url = argv[i];
        else { av_log(NULL, AV_LOG_ERROR, "unknown option '%s'\n", argv[i]); return 1; }
    }

    if (!in_url || !out_url) {
        show_help_default(NULL, NULL);
        return 1;
    }

    return transcode_video(in_url, out_url, venc) < 0 ? 1 : 0;
}
