/*
 * ptvencoder — hold pictures (2.0.0-pre9, T-078 + T-079).
 *
 * While input 0 holds, the decode thread (idle on an empty video_q) feeds one picture per house tick through the
 * filter graph: the held decoded frame for -freeze_max, then black, EBU 75/0/75/0 colour bars or a slate. Per-frame
 * filters therefore keep running — the SV_TIMESTAMP drawtext clock (test channels) keeps ticking over the frozen
 * frame and over the bars — and every rung gets the picture scaled like content, on the CUDA path too. The pictures
 * leave the graph into one slot per rung (g_hold_pic); the output thread shows the newest in place of its repeat,
 * with the held frame's timing, so house_skew, the sensors and the audio fill still see an ordinary repeat.
 *
 * Bars and slate move the whole picture 4 px every -bars_shift (60 s) around 8 positions, so no edge sits on the
 * same pixel columns for the whole outage (OLED / plasma retention); -bars_dim_after drops the bars to 50 %.
 * No-signal pictures are built in yuv444p and converted to the decoded frame's own format, size and colour
 * properties, so the graph never sees a parameter change.
 */

#include "ptvencoder.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"

int     g_hold_mode         = PTV_HOLD_FREEZE;
int     g_hold_render       = 1;
int64_t g_bars_shift_us     = 60000000;
int64_t g_bars_dim_after_us = 0;
_Atomic(AVFrame *) g_hold_pic[PTV_MAX_RUNG];
const char ptv_hold_tag = 0;

static AVFrame *slate_src;                /* the decoded -hold slate image, any pixel format */

enum { PIC_BLACK, PIC_BARS, PIC_SLATE };

typedef struct HoldPic {
    AVFrame    *pic;                      /* the picture in the decoded frame's shape */
    AVFrame    *yuv;                      /* yuv444p work frame (shifted) */
    AVFrame    *slate;                    /* the slate scaled to the shape, unshifted, yuv444p */
    SwsContext *sws_slate, *sws_out;
    int w, h, fmt, range, csp, kind, step, dim;   /* what pic holds (fmt -1 = nothing) */
    int64_t hold_id;                      /* the hold being rendered (g_src_hold_start) */
    int     logged;                       /* the no-signal picture of this hold was logged */
    int64_t n;                            /* pictures rendered this hold */
} HoldPic;

/* 8 positions on a 4 px circle; the step advances once per -bars_shift */
static const int8_t shift_dx[8] = { 0, 3, 4, 3, 0, -3, -4, -3 };
static const int8_t shift_dy[8] = { 4, 3, 0, -3, -4, -3, 0, 3 };

int ptv_hold_slate_load(const char *path)
{
    AVFormatContext *ic = NULL;
    AVCodecContext *dc = NULL;
    const AVCodec *c = NULL;
    AVPacket *pkt = av_packet_alloc();
    AVFrame *f = av_frame_alloc();
    int ret, si;

    if (!pkt || !f) { ret = AVERROR(ENOMEM); goto end; }
    if ((ret = avformat_open_input(&ic, path, NULL, NULL)) < 0 ||
        (ret = avformat_find_stream_info(ic, NULL)) < 0 ||
        (ret = si = av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO, -1, -1, &c, 0)) < 0)
        goto end;
    if (!(dc = avcodec_alloc_context3(c))) { ret = AVERROR(ENOMEM); goto end; }
    if ((ret = avcodec_parameters_to_context(dc, ic->streams[si]->codecpar)) < 0 ||
        (ret = avcodec_open2(dc, c, NULL)) < 0)
        goto end;
    ret = AVERROR(EAGAIN);
    while (ret == AVERROR(EAGAIN) && av_read_frame(ic, pkt) >= 0) {
        if (pkt->stream_index == si && avcodec_send_packet(dc, pkt) >= 0)
            ret = avcodec_receive_frame(dc, f);
        av_packet_unref(pkt);
    }
    if (ret == AVERROR(EAGAIN) && avcodec_send_packet(dc, NULL) >= 0)
        ret = avcodec_receive_frame(dc, f);
end:
    if (ret >= 0) { slate_src = f; f = NULL; }
    av_frame_free(&f);
    av_packet_free(&pkt);
    avcodec_free_context(&dc);
    avformat_close_input(&ic);
    return ret < 0 ? ret : 0;
}

static int yuv_like(AVFrame **pf, const AVFrame *ref, enum AVPixelFormat fmt)
{
    AVFrame *f = *pf;
    if (f && f->width == ref->width && f->height == ref->height && f->format == fmt)
        return 0;
    av_frame_free(pf);
    if (!(f = av_frame_alloc()))
        return AVERROR(ENOMEM);
    f->format = fmt; f->width = ref->width; f->height = ref->height;
    f->color_range = ref->color_range; f->colorspace = ref->colorspace;
    f->color_primaries = ref->color_primaries; f->color_trc = ref->color_trc;
    f->chroma_location = ref->chroma_location;
    if (av_frame_get_buffer(f, 0) < 0) { av_frame_free(&f); return AVERROR(ENOMEM); }
    *pf = f;
    return 0;
}

/* EBU bars at level L (0.75 = 75/0/75/0) as 8-bit Y'CbCr in the ref's matrix and range */
static void bars_yuv(const AVFrame *ref, double L, uint8_t yuv[8][3])
{
    static const uint8_t rgb[8][3] = { {1,1,1}, {1,1,0}, {0,1,1}, {0,1,0}, {1,0,1}, {1,0,0}, {0,0,1}, {0,0,0} };
    int bt709 = ref->colorspace == AVCOL_SPC_BT709 ||
                (ref->colorspace == AVCOL_SPC_UNSPECIFIED && ref->height > 576);
    int full = ref->color_range == AVCOL_RANGE_JPEG;
    double kr = bt709 ? 0.2126 : 0.299, kb = bt709 ? 0.0722 : 0.114;
    int i;
    for (i = 0; i < 8; i++) {
        double r = L * rgb[i][0], g = L * rgb[i][1], b = L * rgb[i][2];
        double y = kr * r + (1 - kr - kb) * g + kb * b;
        double pb = (b - y) / (2 * (1 - kb)), pr = (r - y) / (2 * (1 - kr));
        yuv[i][0] = (uint8_t)lrint(full ? 255 * y : 16 + 219 * y);
        yuv[i][1] = (uint8_t)av_clip_uint8(lrint(128 + (full ? 255 : 224) * pb));
        yuv[i][2] = (uint8_t)av_clip_uint8(lrint(128 + (full ? 255 : 224) * pr));
    }
}

/* Build the no-signal picture into hp->pic: kind/step/dim in the ref's shape, rebuilt only when one changes */
static int build_pic(HoldPic *hp, const AVFrame *ref, int kind, int step, int dim)
{
    uint8_t col[8][3], blk[3];
    int dx = step >= 0 ? shift_dx[step] : 0, dy = step >= 0 ? shift_dy[step] : 0;
    int w = ref->width, h = ref->height, x, y, p, ret;

    if (hp->fmt == ref->format && hp->w == w && hp->h == h && hp->range == ref->color_range &&
        hp->csp == ref->colorspace && hp->kind == kind && hp->step == step && hp->dim == dim)
        return 0;
    hp->fmt = -1;
    if ((ret = yuv_like(&hp->yuv, ref, AV_PIX_FMT_YUV444P)) < 0)
        return ret;
    bars_yuv(ref, dim ? 0.5 : 0.75, col);
    memcpy(blk, col[7], 3);
    if (kind == PIC_SLATE) {
        if (!hp->slate || hp->slate->width != w || hp->slate->height != h) {
            if ((ret = yuv_like(&hp->slate, ref, AV_PIX_FMT_YUV444P)) < 0)
                return ret;
            if (!hp->sws_slate && !(hp->sws_slate = sws_alloc_context()))
                return AVERROR(ENOMEM);
            if ((ret = sws_scale_frame(hp->sws_slate, hp->slate, slate_src)) < 0)
                return ret;
        }
    }
    for (p = 0; p < 3; p++) {
        uint8_t *row = hp->yuv->data[p];
        for (y = 0; y < h; y++, row += hp->yuv->linesize[p]) {
            int sy = y - dy;
            if (sy < 0 || sy >= h) { memset(row, blk[p], w); continue; }
            for (x = 0; x < w; x++) {
                int sx = x - dx;
                if (sx < 0 || sx >= w)          row[x] = blk[p];
                else if (kind == PIC_BARS)      row[x] = col[sx * 8 / w][p];
                else if (kind == PIC_SLATE)     row[x] = hp->slate->data[p][sy * hp->slate->linesize[p] + sx];
                else                            row[x] = blk[p];
            }
        }
    }
    if (ref->format == AV_PIX_FMT_YUV444P) {
        av_frame_free(&hp->pic);
        if (!(hp->pic = av_frame_clone(hp->yuv)))
            return AVERROR(ENOMEM);
    } else {
        if ((ret = yuv_like(&hp->pic, ref, ref->format)) < 0)
            return ret;
        if (!hp->sws_out && !(hp->sws_out = sws_alloc_context()))
            return AVERROR(ENOMEM);
        if ((ret = sws_scale_frame(hp->sws_out, hp->pic, hp->yuv)) < 0)
            return ret;
    }
    hp->fmt = ref->format; hp->w = w; hp->h = h; hp->range = ref->color_range; hp->csp = ref->colorspace;
    hp->kind = kind; hp->step = step; hp->dim = dim;
    return 0;
}

void ptv_hold_route(AVFrame *out, int rung)
{
    AVFrame *old = atomic_exchange_explicit(&g_hold_pic[rung], out, memory_order_acq_rel);
    av_frame_free(&old);
}

void ptv_hold_render(DecodeCtx *d)
{
    HoldPic *hp = d->hr_pic;
    AVFrame *f;
    int64_t now, tick, hs0, el;
    int frozen, i;

    if (!g_hold_render || d->hold || !d->hr_last || !ptv_src_holding())
        return;
    tick = atomic_load_explicit(&g_house_tick_us, memory_order_relaxed);
    now  = av_gettime_relative();
    if (tick <= 0 || now < d->hr_next)
        return;
    d->hr_next = now - d->hr_next < 2 * tick ? d->hr_next + tick : now + tick;
    if (!hp) {
        if (!(hp = d->hr_pic = av_mallocz(sizeof(*hp))))
            return;
        hp->fmt = -1;
    }
    hs0 = atomic_load_explicit(&g_src_hold_start, memory_order_relaxed);
    if (hs0 != hp->hold_id) { hp->hold_id = hs0; hp->logged = 0; hp->n = 0; }
    el = hs0 ? now - hs0 : 0;
    frozen = g_hold_mode != PTV_HOLD_BLACK && (!g_freeze_max_us || el < g_freeze_max_us);
    if (frozen) {
        f = av_frame_clone(d->hr_last);
    } else {
        int kind = g_hold_mode == PTV_HOLD_BARS ? PIC_BARS : g_hold_mode == PTV_HOLD_SLATE ? PIC_SLATE : PIC_BLACK;
        int step = kind != PIC_BLACK && g_bars_shift_us > 0 ? (int)((el / g_bars_shift_us) % 8) : -1;
        int dim  = kind == PIC_BARS && g_bars_dim_after_us > 0 && el >= g_bars_dim_after_us;
        if (build_pic(hp, d->hr_last, kind, step, dim) < 0)
            return;                          /* the output thread's own repeat / black covers it */
        if (!hp->logged) {
            static const char *const name[] = { "black", "bars", "slate" };
            hp->logged = 1;
            av_log(NULL, AV_LOG_WARNING, "[PTV-SRC] in0 held picture \xe2\x86\x92 %s (%s)\n", name[kind],
                   g_hold_mode == PTV_HOLD_BLACK ? "-hold black" : "-freeze_max reached");
        }
        if ((f = av_frame_clone(hp->pic))) {
            av_frame_copy_props(f, d->hr_last);
            av_frame_side_data_free(&f->side_data, &f->nb_side_data);
            f->flags &= ~(AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST | AV_FRAME_FLAG_KEY);
        }
    }
    if (!f)
        return;
    if (!d->hr_dur)
        d->hr_dur = FFMAX(1, av_rescale_q(tick, AV_TIME_BASE_Q, d->ist_tb));
    f->pts = f->best_effort_timestamp = d->hr_pts;
    if (d->hr_pts != AV_NOPTS_VALUE)
        d->hr_pts += d->hr_dur;
    f->duration = d->hr_dur;
    f->pkt_dts  = AV_NOPTS_VALUE;
    f->opaque   = (void *)&ptv_hold_tag;   /* survives the graph (av_frame_copy_props): the frames it leaves
                                            * behind at the rejoin are recognised and dropped */
    hp->n++;
    if (!d->filtering) {
        for (i = 0; i < d->n_rung; i++) {
            AVFrame *out = i == d->n_rung - 1 ? f : av_frame_clone(f);
            if (out) ptv_hold_route(out, i);
        }
        return;
    }
    if (av_buffersrc_add_frame(d->fsrc, f) < 0) { av_frame_free(&f); return; }
    av_frame_free(&f);
    for (i = 0; i < d->n_rung; i++) {
        AVFrame *out;
        /* everything leaving the graph now is a hold picture — including a real frame a temporal filter
         * (bwdif) still held from before the outage: it is the frozen picture */
        while ((out = av_frame_alloc()) && av_buffersink_get_frame(d->fsink[i], out) >= 0)
            ptv_hold_route(out, i);
        av_frame_free(&out);
    }
}

/* emit_video: a real decoded frame (before the graph). Keeps the freeze picture and the render timeline, and closes
 * the account of a finished hold. */
void ptv_hold_note_frame(DecodeCtx *d, const AVFrame *frame)
{
    if (!g_hold_render || d->hold || frame->hw_frames_ctx)
        return;
    if (!d->hr_last && !(d->hr_last = av_frame_alloc()))
        return;
    av_frame_unref(d->hr_last);
    if (av_frame_ref(d->hr_last, frame) < 0)
        return;
    if (frame->duration > 0)
        d->hr_dur = frame->duration;
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE)
        d->hr_pts = frame->best_effort_timestamp + d->hr_dur;
    /* No picture while real frames flow: at a rejoin the hold lasts until the master shows a fresh frame, and a
     * render pushed between real frames would carry the real frame a temporal filter (bwdif) delays by one out
     * into the hold slot — no real frame ever reached frame_q and the hold never ended (local, bwdif + two
     * mid-GOP outages: STALLED 97.5 s). In a real hold no frame arrives, so rendering starts 0.5 s in. */
    d->hr_next = av_gettime_relative() + 500000;
    if (d->hr_pic && d->hr_pic->n && !ptv_src_holding()) {
        av_log(NULL, AV_LOG_INFO, "[PTV-SRC] in0 hold: %"PRId64" pictures rendered through the filter graph "
               "(PTV_NO_HOLD_RENDER=1 disables)\n", d->hr_pic->n);
        d->hr_pic->n = 0;
    }
}

void ptv_hold_uninit(DecodeCtx *d)
{
    int i;
    av_frame_free(&d->hr_last);
    if (d->hr_pic) {
        av_frame_free(&d->hr_pic->pic);
        av_frame_free(&d->hr_pic->yuv);
        av_frame_free(&d->hr_pic->slate);
        sws_freeContext(d->hr_pic->sws_slate);
        sws_freeContext(d->hr_pic->sws_out);
        av_freep(&d->hr_pic);
    }
    for (i = 0; i < PTV_MAX_RUNG; i++)
        ptv_hold_route(NULL, i);
}
