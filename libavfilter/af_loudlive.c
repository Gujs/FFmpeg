/*
 * Copyright (c) 2026 Gregor Fuis
 *
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

/**
 * @file
 * Live loudness normalization without lookahead.
 *
 * The gain law of loudnorm's dynamic mode, applied causally: every 100 ms the
 * desired gain is target - integrated loudness, except where the short-term
 * loudness strays more than LRA/2 from the integrated one, which is then pulled
 * back to target +- LRA/2. loudnorm centres a 2 s Gaussian on that gain and so
 * holds 3 s of audio; here the gain is smoothed by a one-pole filter and applied
 * to the samples as they arrive: no added latency, nothing held when the input
 * pauses. The gain is held while the input is quiet (below the relative gate).
 * Peaks are not limited here; follow with alimiter.
 */

#include <math.h>

#include "libavutil/opt.h"
#include "avfilter.h"
#include "audio.h"
#include "ebur128.h"
#include "filters.h"
#include "formats.h"

typedef struct LoudLiveContext {
    const AVClass *class;
    double target_i;
    double target_lra;
    double smooth;
    int dual_mono;

    FFEBUR128State *r128;
    int block;          ///< samples per gain update (100 ms)
    int pos;            ///< samples into the current block
    int64_t nb_in;      ///< samples measured so far
    double alpha;       ///< smoothing coefficient per block
    double gain_db;     ///< smoothed gain
    int have_gain;
    double from, to;    ///< linear gain ramp across the current block
} LoudLiveContext;

#define OFFSET(x) offsetof(LoudLiveContext, x)
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption loudlive_options[] = {
    { "I",         "set integrated loudness target",      OFFSET(target_i),   AV_OPT_TYPE_DOUBLE, {.dbl = -24.}, -70., -5., FLAGS },
    { "i",         "set integrated loudness target",      OFFSET(target_i),   AV_OPT_TYPE_DOUBLE, {.dbl = -24.}, -70., -5., FLAGS },
    { "LRA",       "set loudness range target",           OFFSET(target_lra), AV_OPT_TYPE_DOUBLE, {.dbl =   7.},   1., 50., FLAGS },
    { "lra",       "set loudness range target",           OFFSET(target_lra), AV_OPT_TYPE_DOUBLE, {.dbl =   7.},   1., 50., FLAGS },
    { "smooth",    "set gain smoothing time constant (s)", OFFSET(smooth),    AV_OPT_TYPE_DOUBLE, {.dbl = 0.4},    0., 10., FLAGS },
    { "dual_mono", "treat mono input as dual-mono",       OFFSET(dual_mono),  AV_OPT_TYPE_BOOL,   {.i64 = 0},      0,   1,  FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(loudlive);

static void update_gain(AVFilterContext *ctx)
{
    LoudLiveContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    double global, shortterm, relative_threshold, lra2 = s->target_lra / 2., want;

    ff_ebur128_loudness_global(s->r128, &global);
    ff_ebur128_relative_threshold(s->r128, &relative_threshold);
    /* the short-term window is not full yet: steer on the integrated loudness alone */
    if (s->nb_in < 3LL * inlink->sample_rate)
        shortterm = global;
    else
        ff_ebur128_loudness_shortterm(s->r128, &shortterm);

    if (global <= -70. || shortterm <= -70. || shortterm < relative_threshold)
        return;

    want = s->target_i - shortterm + av_clipd(shortterm - global, -lra2, lra2);
    if (!s->have_gain) {
        s->gain_db   = want;
        s->have_gain = 1;
    } else {
        s->gain_db  += s->alpha * (want - s->gain_db);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    LoudLiveContext *s = ctx->priv;
    const int nch = inlink->ch_layout.nb_channels;
    const double *src = (const double *)in->data[0];
    AVFrame *out = in;
    double *dst;
    int n = 0;

    if (!av_frame_is_writable(in)) {
        out = ff_get_audio_buffer(outlink, in->nb_samples);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
    }
    dst = (double *)out->data[0];

    while (n < in->nb_samples) {
        int len = FFMIN(in->nb_samples - n, s->block - s->pos);

        ff_ebur128_add_frames_double(s->r128, src + n * nch, len);
        for (int i = 0; i < len; i++) {
            double g = s->from + (s->to - s->from) * (s->pos + i) / s->block;
            for (int c = 0; c < nch; c++)
                dst[(n + i) * nch + c] = src[(n + i) * nch + c] * g;
        }
        n        += len;
        s->pos   += len;
        s->nb_in += len;
        if (s->pos == s->block) {
            update_gain(ctx);
            s->from = s->to;
            s->to   = pow(10., s->gain_db / 20.);
            s->pos  = 0;
        }
    }

    if (out != in)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    LoudLiveContext *s = ctx->priv;

    s->r128 = ff_ebur128_init(inlink->ch_layout.nb_channels, inlink->sample_rate, 0,
                              FF_EBUR128_MODE_I | FF_EBUR128_MODE_S);
    if (!s->r128)
        return AVERROR(ENOMEM);
    if (inlink->ch_layout.nb_channels == 1 && s->dual_mono)
        ff_ebur128_set_channel(s->r128, 0, FF_EBUR128_DUAL_MONO);

    s->block = inlink->sample_rate / 10;
    s->alpha = s->smooth > 0. ? 1. - exp(-0.1 / s->smooth) : 1.;
    s->from  = s->to = 1.;
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    LoudLiveContext *s = ctx->priv;

    if (s->r128)
        ff_ebur128_destroy(&s->r128);
}

static const AVFilterPad loudlive_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_AUDIO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
};

const FFFilter ff_af_loudlive = {
    .p.name        = "loudlive",
    .p.description = NULL_IF_CONFIG_SMALL("EBU R128 loudness normalization without lookahead"),
    .p.priv_class  = &loudlive_class,
    .priv_size     = sizeof(LoudLiveContext),
    .uninit        = uninit,
    FILTER_INPUTS(loudlive_inputs),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_SINGLE_SAMPLEFMT(AV_SAMPLE_FMT_DBL),
};
