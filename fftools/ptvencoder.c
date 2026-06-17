/*
 * ptvencoder — purpose-built live MPEG-TS re-encoder.
 *
 * A sibling fftools program (alongside ffmpeg/ffprobe/ffplay) that links the
 * same patched libav* libraries but runs on its own house-clock timing engine.
 * See analysis/ptvencoder-functional-spec.md.
 *
 * Phase 1, increment 3: video house-clock + audio (A/V common-mode anchor).
 *   demux -> {video: decode -> house-clock frame-sync -> encode}
 *         -> {audio: decode -> resample-to-48k -> AAC} -> mux.
 *
 * Both video and audio map their source PTS onto one shared input anchor (h0),
 * so the source A/V offset (lip sync) is preserved while the absolute timeline
 * becomes the house clock. Video drives wall-clock pacing in live mode; audio
 * rides the same single-threaded loop.
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
#include "libavutil/samplefmt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/audio_fifo.h"
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswresample/swresample.h"

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
        "    -an           disable audio\n"
        "    --mode live|offline   live = wall-clock paced; offline = media-clock. default: auto from input\n"
        "    -version, -h\n"
        "\n"
        "  Phase 1 increment 3: video house clock + AAC audio (A/V common-mode anchor).\n");
}

/* Drain an encoder, writing packets to the muxer. frame=NULL flushes. */
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

/* ---- video house clock ---- */

typedef struct ClockState {
    AVFormatContext *ofmt;
    AVCodecContext  *enc;
    AVStream        *ost;
    AVRational       ist_tb;
    int64_t          tick_dur_us;
    int              live;
    AVPacket        *pkt;
    int64_t         *h0;           /* shared input anchor (us) */

    int64_t          wall0_us;
    int64_t          next_tick;
    AVFrame         *held;
    int              have_held;
    int              held_emits;
    int64_t          dup, drop, emitted, in_frames;
} ClockState;

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

static int clock_push(ClockState *c, AVFrame *frame)
{
    int ret = 0;
    int64_t ts = frame->best_effort_timestamp;
    int64_t house_us;

    c->in_frames++;
    if (ts != AV_NOPTS_VALUE) {
        int64_t src_us = av_rescale_q(ts, c->ist_tb, AV_TIME_BASE_Q);
        if (*c->h0 == AV_NOPTS_VALUE)
            *c->h0 = src_us;
        house_us = src_us - *c->h0;
        if (house_us < c->next_tick * c->tick_dur_us)   /* backwards guard (re-anchor = next increment) */
            house_us = c->next_tick * c->tick_dur_us;
    } else {
        house_us = c->next_tick * c->tick_dur_us;
    }

    while (c->have_held && c->next_tick * c->tick_dur_us < house_us)
        if ((ret = emit_tick(c)) < 0)
            return ret;

    if (c->have_held && c->held_emits == 0)
        c->drop++;
    av_frame_unref(c->held);
    av_frame_move_ref(c->held, frame);
    c->have_held  = 1;
    c->held_emits = 0;
    return 0;
}

static int clock_flush(ClockState *c)
{
    if (c->have_held && c->held_emits == 0)
        return emit_tick(c);
    return 0;
}

/* ---- audio path (decode -> resample 48k stereo -> AAC -> mux) ---- */

typedef struct AudioState {
    AVFormatContext *ofmt;
    AVCodecContext  *enc;
    AVStream        *ost;
    AVRational       ist_tb;
    SwrContext      *swr;
    AVAudioFifo     *fifo;
    int              frame_size;
    int              out_rate;
    enum AVSampleFormat out_sfmt;
    AVChannelLayout  out_chl;
    AVPacket        *pkt;
    int64_t         *h0;           /* shared input anchor (us) */
    int              pts_set;
    int64_t          next_pts;     /* output sample index */
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
        ret = encode_write(a->ofmt, a->enc, a->ost, f, a->pkt);
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

    if (ts != AV_NOPTS_VALUE && *a->h0 == AV_NOPTS_VALUE)
        *a->h0 = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q);

    if (!a->pts_set && ts != AV_NOPTS_VALUE && *a->h0 != AV_NOPTS_VALUE) {
        int64_t house_us = av_rescale_q(ts, a->ist_tb, AV_TIME_BASE_Q) - *a->h0;
        if (house_us < 0) house_us = 0;
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

static int audio_flush(AudioState *a)
{
    /* flush the resampler, then any buffered full frames */
    uint8_t **out = NULL;
    int got, ret = 0;
    int out_max = 4096;
    if (av_samples_alloc_array_and_samples(&out, NULL, a->out_chl.nb_channels,
                                           out_max, a->out_sfmt, 0) >= 0) {
        while ((got = swr_convert(a->swr, out, out_max, NULL, 0)) > 0)
            av_audio_fifo_write(a->fifo, (void **)out, got);
        av_freep(&out[0]); av_freep(&out);
    }
    if ((ret = audio_drain_fifo(a)) < 0) return ret;
    return encode_write(a->ofmt, a->enc, a->ost, NULL, a->pkt);  /* flush encoder */
}

static int transcode(const char *in_url, const char *out_url,
                     const char *venc_name, const char *rate_str, int mode, int want_audio)
{
    AVFormatContext *ifmt = NULL, *ofmt = NULL;
    AVCodecContext  *vdec = NULL, *venc = NULL, *adec = NULL, *aenc = NULL;
    const AVCodec   *vdecoder = NULL, *vencoder = NULL, *adecoder = NULL, *aencoder = NULL;
    AVStream        *vist = NULL, *aist = NULL;
    AVPacket        *pkt = NULL;
    AVFrame         *frame = NULL;
    ClockState       cs;
    AudioState       as;
    int64_t          input_h0_us = AV_NOPTS_VALUE;
    int vstream = -1, astream = -1, ret = 0, live, have_audio = 0;
    AVRational out_fps;

    memset(&cs, 0, sizeof(cs));
    memset(&as, 0, sizeof(as));

    if ((ret = avformat_open_input(&ifmt, in_url, NULL, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "cannot open input '%s': %s\n", in_url, av_err2str(ret)); return ret;
    }
    if ((ret = avformat_find_stream_info(ifmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "stream info: %s\n", av_err2str(ret)); goto end;
    }

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

    if ((ret = avformat_alloc_output_context2(&ofmt, NULL, NULL, out_url)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "output ctx: %s\n", av_err2str(ret)); goto end;
    }

    /* video encoder */
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

    venc = avcodec_alloc_context3(vencoder);
    if (!venc) { ret = AVERROR(ENOMEM); goto end; }
    venc->width = vdec->width; venc->height = vdec->height;
    venc->pix_fmt = vdec->pix_fmt != AV_PIX_FMT_NONE ? vdec->pix_fmt : AV_PIX_FMT_YUV420P;
    venc->time_base = av_inv_q(out_fps); venc->framerate = out_fps;
    venc->bit_rate = 3000000; venc->gop_size = 2 * (out_fps.num / FFMAX(out_fps.den, 1));
    if (ofmt->oformat->flags & AVFMT_GLOBALHEADER) venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if ((ret = avcodec_open2(venc, vencoder, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "open video encoder '%s': %s\n", vencoder->name, av_err2str(ret)); goto end;
    }
    cs.ost = avformat_new_stream(ofmt, NULL);
    if (!cs.ost) { ret = AVERROR(ENOMEM); goto end; }
    avcodec_parameters_from_context(cs.ost->codecpar, venc);
    cs.ost->time_base = venc->time_base;

    /* audio (optional) */
    if (want_audio)
        astream = av_find_best_stream(ifmt, AVMEDIA_TYPE_AUDIO, -1, -1, &adecoder, 0);
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

    if (!(ofmt->oformat->flags & AVFMT_NOFILE))
        if ((ret = avio_open(&ofmt->pb, out_url, AVIO_FLAG_WRITE)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "open output '%s': %s\n", out_url, av_err2str(ret)); goto end;
        }
    if ((ret = avformat_write_header(ofmt, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "write header: %s\n", av_err2str(ret)); goto end;
    }

    live = mode < 0 ? (!strncmp(in_url, "udp://", 6) || !strncmp(in_url, "rtp://", 6) ||
                       !strncmp(in_url, "srt://", 6)) : mode;

    pkt = av_packet_alloc(); frame = av_frame_alloc(); cs.held = av_frame_alloc();
    if (!pkt || !frame || !cs.held) { ret = AVERROR(ENOMEM); goto end; }

    cs.ofmt = ofmt; cs.enc = venc; cs.ist_tb = vist->time_base;
    cs.tick_dur_us = av_rescale(1000000, out_fps.den, out_fps.num);
    cs.live = live; cs.pkt = pkt; cs.h0 = &input_h0_us;
    if (have_audio) {
        as.ofmt = ofmt; as.enc = aenc; as.ist_tb = aist->time_base; as.pkt = pkt; as.h0 = &input_h0_us;
    }

    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: %dx%d  house %d/%d fps (%s)  v:%s->%s  a:%s  [%s]\n",
        venc->width, venc->height, out_fps.num, out_fps.den, live ? "live" : "offline",
        vdecoder->name, vencoder->name, have_audio ? "aac" : "none", ofmt->oformat->name);

    while (av_read_frame(ifmt, pkt) >= 0) {
        if (pkt->stream_index == vstream) {
            ret = avcodec_send_packet(vdec, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(vdec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
                if (ret < 0) goto end;
                if (frame->flags & AV_FRAME_FLAG_CORRUPT) { av_frame_unref(frame); continue; }
                if ((ret = clock_push(&cs, frame)) < 0) goto end;
            }
        } else if (have_audio && pkt->stream_index == astream) {
            ret = avcodec_send_packet(adec, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(adec, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { ret = 0; break; }
                if (ret < 0) goto end;
                if ((ret = audio_push(&as, frame)) < 0) goto end;
                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    /* flush video */
    avcodec_send_packet(vdec, NULL);
    while (avcodec_receive_frame(vdec, frame) >= 0) {
        if (!(frame->flags & AV_FRAME_FLAG_CORRUPT)) clock_push(&cs, frame);
        else av_frame_unref(frame);
    }
    clock_flush(&cs);
    encode_write(ofmt, venc, cs.ost, NULL, pkt);

    /* flush audio */
    if (have_audio) {
        avcodec_send_packet(adec, NULL);
        while (avcodec_receive_frame(adec, frame) >= 0) { audio_push(&as, frame); av_frame_unref(frame); }
        audio_flush(&as);
    }

    av_write_trailer(ofmt);
    av_log(NULL, AV_LOG_INFO,
        "ptvencoder: done — video in %"PRId64" out %"PRId64" (dup %"PRId64" drop %"PRId64")%s\n",
        cs.in_frames, cs.emitted, cs.dup, cs.drop,
        have_audio ? "" : "  [no audio]");
    if (have_audio)
        av_log(NULL, AV_LOG_INFO, "ptvencoder: audio in %"PRId64" frames, out %"PRId64" aac frames\n",
               as.in_frames, as.out_frames);
    ret = 0;

end:
    if (ofmt && !(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb)
        avio_closep(&ofmt->pb);
    if (as.swr)  swr_free(&as.swr);
    if (as.fifo) av_audio_fifo_free(as.fifo);
    av_frame_free(&cs.held);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&aenc);
    avcodec_free_context(&adec);
    avcodec_free_context(&venc);
    avcodec_free_context(&vdec);
    if (ofmt) avformat_free_context(ofmt);
    avformat_close_input(&ifmt);
    return ret;
}

int main(int argc, char **argv)
{
    const char *in_url = NULL, *out_url = NULL, *rate = NULL;
    const char *venc = "h264_videotoolbox";
    int mode = -1, want_audio = 1, i;

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
        else if (!strcmp(argv[i], "-an"))                  want_audio = 0;
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
    return transcode(in_url, out_url, venc, rate, mode, want_audio) < 0 ? 1 : 0;
}
