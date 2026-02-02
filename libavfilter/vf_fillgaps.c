/*
 * Copyright (c) 2026 Anthropic / Perception Group Inc.
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
 * Fill gaps in video stream by duplicating frames when wall clock gaps detected.
 *
 * This filter monitors wall clock time between frames. When a gap larger than
 * the threshold is detected (e.g., decoder waiting for keyframe after flush),
 * it duplicates the last frame to fill the gap, maintaining continuous output.
 */

#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/time.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"

typedef struct FillGapsContext {
    const AVClass *class;

    int64_t threshold;          ///< gap threshold in microseconds
    int64_t max_fill;           ///< maximum fill duration in microseconds

    /* Runtime state */
    int64_t last_wallclock;     ///< wall clock time of last frame
    int64_t last_pts;           ///< PTS of last frame
    int64_t frame_duration;     ///< expected frame duration in stream timebase
    AVFrame *last_frame;        ///< last frame for duplication
    int initialized;

    /* Statistics */
    int64_t frames_in;
    int64_t frames_out;
    int64_t frames_dup;
} FillGapsContext;

#define OFFSET(x) offsetof(FillGapsContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption fillgaps_options[] = {
    { "threshold", "wall clock gap threshold to trigger fill (in seconds)",
      OFFSET(threshold), AV_OPT_TYPE_DURATION, { .i64 = 500000 }, 0, INT64_MAX, FLAGS },
    { "max_fill", "maximum duration to fill (in seconds)",
      OFFSET(max_fill), AV_OPT_TYPE_DURATION, { .i64 = 30000000 }, 0, INT64_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(fillgaps);

static av_cold void uninit(AVFilterContext *ctx)
{
    FillGapsContext *s = ctx->priv;

    av_frame_free(&s->last_frame);

    av_log(ctx, AV_LOG_INFO,
           "[FILLGAPS] Stats: %"PRId64" frames in, %"PRId64" frames out, "
           "%"PRId64" frames duplicated to fill gaps\n",
           s->frames_in, s->frames_out, s->frames_dup);
}

static int config_input(AVFilterLink *inlink)
{
    FillGapsContext *s = inlink->dst->priv;
    FilterLink *il = ff_filter_link(inlink);
    AVRational frame_rate = il->frame_rate;

    /* Calculate frame duration in stream timebase */
    if (frame_rate.num > 0 && frame_rate.den > 0) {
        s->frame_duration = av_rescale_q(1, av_inv_q(frame_rate), inlink->time_base);
    } else {
        /* Fallback: assume 30fps */
        s->frame_duration = av_rescale_q(1, (AVRational){1, 30}, inlink->time_base);
    }

    av_log(inlink->dst, AV_LOG_INFO,
           "[FILLGAPS] Initialized: threshold=%"PRId64"ms, max_fill=%"PRId64"ms, "
           "frame_dur=%"PRId64" (timebase=%d/%d, fps=%d/%d)\n",
           s->threshold / 1000, s->max_fill / 1000,
           s->frame_duration, inlink->time_base.num, inlink->time_base.den,
           frame_rate.num, frame_rate.den);

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    FillGapsContext *s = ctx->priv;
    int64_t now_us = av_gettime_relative();
    int ret;

    s->frames_in++;

    if (!s->initialized) {
        /* First frame - initialize state */
        s->last_wallclock = now_us;
        s->last_pts = frame->pts;
        s->last_frame = av_frame_clone(frame);
        s->initialized = 1;

        s->frames_out++;
        return ff_filter_frame(outlink, frame);
    }

    /*
     * Only trigger on wall clock gap - this detects actual decoder stalls
     * (e.g., waiting for keyframe after flush).
     *
     * We intentionally do NOT trigger on PTS gap alone, because:
     * - PTS gap from timestamp discontinuity = content NOT missing
     * - PTS gap from corrupt frame drop = usually 1-2 frames, tolerable
     *
     * Triggering on PTS gap causes buffering growth because we duplicate
     * frames that aren't actually missing, just have discontinuous timestamps.
     */
    int64_t wallclock_gap = now_us - s->last_wallclock;

    if (wallclock_gap > s->threshold && s->last_frame) {
        /*
         * Wall clock gap detected - decoder stalled.
         * Calculate frames to fill based on wall clock time to maintain
         * real-time output rate. Don't use PTS gap as it may reflect
         * timestamp discontinuity rather than missing content.
         */
        int64_t frame_dur_us = av_rescale_q(s->frame_duration, inlink->time_base,
                                            (AVRational){1, 1000000});
        if (frame_dur_us <= 0) frame_dur_us = 33333; /* fallback 30fps */

        int64_t frames_to_fill = (wallclock_gap - s->threshold / 2) / frame_dur_us;

        /* Clamp to reasonable limits */
        if (frames_to_fill < 0) frames_to_fill = 0;
        if (frames_to_fill > 900) frames_to_fill = 900; /* safety limit */

        /* Also check against max_fill */
        int64_t max_fill_frames = s->max_fill / frame_dur_us;
        if (frames_to_fill > max_fill_frames) frames_to_fill = max_fill_frames;

        if (frames_to_fill > 0) {
            av_log(ctx, AV_LOG_WARNING,
                   "[FILLGAPS] Wall clock gap detected: %"PRId64"ms > threshold %"PRId64"ms, "
                   "filling with %"PRId64" duplicate frames\n",
                   wallclock_gap / 1000, s->threshold / 1000, frames_to_fill);

            int64_t fill_pts = s->last_pts + s->frame_duration;

            for (int64_t i = 0; i < frames_to_fill; i++) {
                AVFrame *dup = av_frame_clone(s->last_frame);
                if (!dup) {
                    av_log(ctx, AV_LOG_ERROR, "[FILLGAPS] Failed to clone frame\n");
                    break;
                }

                dup->pts = fill_pts;
                dup->duration = s->frame_duration;

                ret = ff_filter_frame(outlink, dup);
                if (ret < 0) {
                    av_log(ctx, AV_LOG_WARNING,
                           "[FILLGAPS] Failed to output fill frame: %s\n",
                           av_err2str(ret));
                    break;
                }

                s->frames_out++;
                s->frames_dup++;
                fill_pts += s->frame_duration;
            }

            av_log(ctx, AV_LOG_WARNING,
                   "[FILLGAPS] Filled %"PRId64" frames (PTS %"PRId64" to %"PRId64"), "
                   "incoming frame PTS %"PRId64"\n",
                   frames_to_fill, s->last_pts + s->frame_duration,
                   fill_pts - s->frame_duration, frame->pts);
        }
    }

    /* Update state */
    s->last_wallclock = now_us;
    s->last_pts = frame->pts;

    /* Update last_frame cache */
    av_frame_free(&s->last_frame);
    s->last_frame = av_frame_clone(frame);

    s->frames_out++;
    return ff_filter_frame(outlink, frame);
}

static const AVFilterPad fillgaps_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
};

const FFFilter ff_vf_fillgaps = {
    .p.name        = "fillgaps",
    .p.description = NULL_IF_CONFIG_SMALL("Fill gaps in video by duplicating frames based on wall clock time."),
    .p.priv_class  = &fillgaps_class,
    .priv_size     = sizeof(FillGapsContext),
    .uninit        = uninit,
    FILTER_INPUTS(fillgaps_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
};
