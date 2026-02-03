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
 *
 * When timestamps are rebased after a discontinuity (large wall clock gap but
 * small PTS gap), the filter enters "proactive mode" where it outputs frames
 * at the expected wall clock rate using regenerated timestamps.
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
    int proactive;              ///< enable proactive output mode

    /* Runtime state */
    int64_t last_wallclock;     ///< wall clock time of last frame
    int64_t last_pts;           ///< PTS of last frame
    int64_t frame_duration;     ///< expected frame duration in stream timebase
    int64_t frame_dur_us;       ///< frame duration in microseconds
    AVFrame *last_frame;        ///< last frame for duplication
    int initialized;

    /* Proactive output mode state */
    int in_proactive_mode;      ///< currently in proactive output mode
    int64_t output_pts;         ///< next output PTS (in proactive mode)
    int64_t last_output_time;   ///< wall clock time of last output
    int64_t proactive_start;    ///< wall clock when proactive mode started
    int64_t proactive_frames;   ///< frames output in current proactive session
    int64_t recovery_wall_time; ///< wall clock time for recovery detection

    /* Statistics */
    int64_t frames_in;
    int64_t frames_out;
    int64_t frames_dup;
    int64_t proactive_sessions; ///< number of proactive mode activations
} FillGapsContext;

#define OFFSET(x) offsetof(FillGapsContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption fillgaps_options[] = {
    { "threshold", "wall clock gap threshold to trigger fill (in seconds)",
      OFFSET(threshold), AV_OPT_TYPE_DURATION, { .i64 = 500000 }, 0, INT64_MAX, FLAGS },
    { "max_fill", "maximum duration to fill (in seconds)",
      OFFSET(max_fill), AV_OPT_TYPE_DURATION, { .i64 = 30000000 }, 0, INT64_MAX, FLAGS },
    { "proactive", "enable proactive output mode for timestamp rebasing recovery",
      OFFSET(proactive), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(fillgaps);

static av_cold void uninit(AVFilterContext *ctx)
{
    FillGapsContext *s = ctx->priv;

    av_frame_free(&s->last_frame);

    av_log(ctx, AV_LOG_INFO,
           "[FILLGAPS] Stats: %"PRId64" frames in, %"PRId64" frames out, "
           "%"PRId64" frames duplicated, %"PRId64" proactive sessions\n",
           s->frames_in, s->frames_out, s->frames_dup, s->proactive_sessions);
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

    /* Pre-calculate frame duration in microseconds */
    s->frame_dur_us = av_rescale_q(s->frame_duration, inlink->time_base,
                                    (AVRational){1, 1000000});
    if (s->frame_dur_us <= 0) s->frame_dur_us = 33333; /* fallback 30fps */

    av_log(inlink->dst, AV_LOG_INFO,
           "[FILLGAPS] Initialized: threshold=%"PRId64"ms, max_fill=%"PRId64"ms, "
           "proactive=%d, frame_dur=%"PRId64" (%"PRId64"us) (timebase=%d/%d, fps=%d/%d)\n",
           s->threshold / 1000, s->max_fill / 1000, s->proactive,
           s->frame_duration, s->frame_dur_us,
           inlink->time_base.num, inlink->time_base.den,
           frame_rate.num, frame_rate.den);

    return 0;
}

/**
 * Output a duplicate frame with given PTS
 */
static int output_dup_frame(AVFilterContext *ctx, AVFilterLink *outlink,
                            FillGapsContext *s, int64_t pts)
{
    AVFrame *dup = av_frame_clone(s->last_frame);
    if (!dup) {
        av_log(ctx, AV_LOG_ERROR, "[FILLGAPS] Failed to clone frame\n");
        return AVERROR(ENOMEM);
    }

    dup->pts = pts;
    dup->duration = s->frame_duration;

    int ret = ff_filter_frame(outlink, dup);
    if (ret < 0) {
        av_log(ctx, AV_LOG_WARNING,
               "[FILLGAPS] Failed to output frame: %s\n", av_err2str(ret));
        return ret;
    }

    s->frames_out++;
    s->frames_dup++;
    return 0;
}

/**
 * Check if we should exit proactive mode.
 * Exit when input frames are arriving at expected rate for a sustained period,
 * OR when we've caught up with real-time (proactive frames exceed wall clock expectation).
 */
static int should_exit_proactive_mode(AVFilterContext *ctx, FillGapsContext *s, int64_t wallclock_gap)
{
    /* Safety check: Don't let proactive mode run faster than real-time.
     * Compare proactive frames output vs wall clock elapsed since start. */
    int64_t now = av_gettime_relative();
    int64_t elapsed_us = now - s->proactive_start;
    int64_t expected_frames = elapsed_us / s->frame_dur_us;

    if (s->proactive_frames > expected_frames + 30) {
        /* We're ahead of real-time by more than 30 frames - exit to prevent buffer growth */
        av_log(ctx, AV_LOG_WARNING,
               "[FILLGAPS] Exiting proactive mode: output %"PRId64" frames in %"PRId64"ms "
               "(expected %"PRId64") - ahead of real-time\n",
               s->proactive_frames, elapsed_us / 1000, expected_frames);
        return 1;
    }

    /* If input is arriving at expected rate (within 2x frame duration),
     * count towards recovery */
    if (wallclock_gap < s->frame_dur_us * 2) {
        /* Check if we've had sustained good input for 1 second */
        if (s->recovery_wall_time == 0) {
            s->recovery_wall_time = now;
        } else if (now - s->recovery_wall_time > 1000000) {
            /* 1 second of good input - exit proactive mode */
            return 1;
        }
    } else {
        /* Input still slow - reset recovery timer */
        s->recovery_wall_time = 0;
    }
    return 0;
}

/**
 * Process an input frame
 */
static int process_frame(AVFilterContext *ctx, AVFrame *frame)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    FillGapsContext *s = ctx->priv;
    int64_t now_us = av_gettime_relative();
    int ret;

    s->frames_in++;

    if (!s->initialized) {
        /* First frame - initialize state */
        s->last_wallclock = now_us;
        s->last_pts = frame->pts;
        s->output_pts = frame->pts + s->frame_duration;
        s->last_output_time = now_us;
        s->last_frame = av_frame_clone(frame);
        s->initialized = 1;

        s->frames_out++;
        return ff_filter_frame(outlink, frame);
    }

    int64_t wallclock_gap = now_us - s->last_wallclock;

    /* Check if we're in proactive mode */
    if (s->in_proactive_mode) {
        /* In proactive mode - use our generated timestamps */

        /* Check if input timing has recovered */
        if (should_exit_proactive_mode(ctx, s, wallclock_gap)) {
            av_log(ctx, AV_LOG_INFO,
                   "[FILLGAPS] Exiting proactive mode after %"PRId64" frames "
                   "(input timing recovered)\n", s->proactive_frames);
            s->in_proactive_mode = 0;

            /* CRITICAL: Reset wall clock tracking to prevent false gap detection.
             * Use the output_pts we've been maintaining, not the incoming frame's PTS. */
            s->last_wallclock = now_us;
            s->last_pts = s->output_pts - s->frame_duration;
            s->last_output_time = now_us;

            /* Fall through to normal processing with fresh state */
        } else {
            /* Still in proactive mode - use generated PTS */
            frame->pts = s->output_pts;
            s->output_pts += s->frame_duration;
            s->last_output_time = now_us;
            s->proactive_frames++;

            /* Update state */
            s->last_wallclock = now_us;
            av_frame_free(&s->last_frame);
            s->last_frame = av_frame_clone(frame);

            s->frames_out++;
            return ff_filter_frame(outlink, frame);
        }
    }

    /* Normal mode - check for gaps */
    if (wallclock_gap > s->threshold && s->last_frame) {
        /*
         * Wall clock gap detected - decoder stalled.
         * Calculate frames to fill based on wall clock time.
         */
        int64_t frames_to_fill = (wallclock_gap - s->threshold / 2) / s->frame_dur_us;

        /* Clamp to reasonable limits */
        if (frames_to_fill < 0) frames_to_fill = 0;
        if (frames_to_fill > 900) frames_to_fill = 900; /* safety limit */

        /* Also check against max_fill */
        int64_t max_fill_frames = s->max_fill / s->frame_dur_us;
        if (frames_to_fill > max_fill_frames) frames_to_fill = max_fill_frames;

        /*
         * Check PTS gap - if much smaller than wall clock gap, timestamps
         * were likely rebased after discontinuity.
         */
        int64_t pts_gap = 0;
        int64_t max_pts_frames = frames_to_fill;
        int timestamps_rebased = 0;

        if (frame->pts != AV_NOPTS_VALUE && s->last_pts != AV_NOPTS_VALUE &&
            s->frame_duration > 0) {
            pts_gap = frame->pts - s->last_pts;
            if (pts_gap > 0) {
                /* How many frames fit in the PTS gap (leave room for incoming frame) */
                max_pts_frames = (pts_gap / s->frame_duration) - 1;
                if (max_pts_frames < 0) max_pts_frames = 0;

                /* Detect timestamp rebasing: wall clock says 500+ frames, PTS says < 100 */
                if (frames_to_fill > 100 && max_pts_frames < frames_to_fill / 5) {
                    timestamps_rebased = 1;
                }

                if (frames_to_fill > max_pts_frames && !timestamps_rebased) {
                    av_log(ctx, AV_LOG_INFO,
                           "[FILLGAPS] Capping fill from %"PRId64" to %"PRId64" frames "
                           "(wall=%"PRId64"ms but PTS gap=%"PRId64" allows only %"PRId64")\n",
                           frames_to_fill, max_pts_frames,
                           wallclock_gap / 1000, pts_gap, max_pts_frames);
                    frames_to_fill = max_pts_frames;
                }
            } else {
                /* PTS went backwards or stayed same - don't fill at all */
                av_log(ctx, AV_LOG_INFO,
                       "[FILLGAPS] Skipping fill: PTS gap=%"PRId64" (last=%"PRId64" incoming=%"PRId64")\n",
                       pts_gap, s->last_pts, frame->pts);
                frames_to_fill = 0;
            }
        }

        /*
         * If timestamps were rebased and proactive mode is enabled,
         * enter proactive mode for recovery period.
         *
         * In proactive mode, we DON'T bulk-fill frames here. Instead, we:
         * 1. Enter proactive mode and set up the output_pts
         * 2. Output just a few frames (up to 10) to start
         * 3. Let activate() output the rest one at a time at wall-clock rate
         * This prevents buffer buildup from dumping hundreds of frames at once.
         */
        if (timestamps_rebased && s->proactive) {
            av_log(ctx, AV_LOG_WARNING,
                   "[FILLGAPS] Detected timestamp rebasing: wall=%"PRId64"ms (%"PRId64" frames), "
                   "PTS gap=%"PRId64" (%"PRId64" frames) - entering proactive mode\n",
                   wallclock_gap / 1000, frames_to_fill, pts_gap, max_pts_frames);

            s->in_proactive_mode = 1;
            s->proactive_start = now_us;
            s->proactive_frames = 0;
            s->proactive_sessions++;
            s->recovery_wall_time = 0;

            /* Start output PTS from where we left off */
            s->output_pts = s->last_pts + s->frame_duration;

            /* Fill only a small burst (max 10 frames) to get started.
             * activate() will output the rest at wall-clock rate. */
            int64_t initial_fill = frames_to_fill > 10 ? 10 : frames_to_fill;

            av_log(ctx, AV_LOG_INFO,
                   "[FILLGAPS] Proactive mode: initial fill %"PRId64" frames, "
                   "activate() will handle the rest at wall-clock rate\n", initial_fill);

            for (int64_t i = 0; i < initial_fill; i++) {
                ret = output_dup_frame(ctx, outlink, s, s->output_pts);
                if (ret < 0) break;
                s->output_pts += s->frame_duration;
                s->proactive_frames++;
            }

            /* Output incoming frame with generated PTS */
            frame->pts = s->output_pts;
            s->output_pts += s->frame_duration;
            s->last_output_time = now_us;
            s->proactive_frames++;

            /* Update state */
            s->last_wallclock = now_us;
            av_frame_free(&s->last_frame);
            s->last_frame = av_frame_clone(frame);

            /* Schedule activate() to output more frames */
            ff_filter_set_ready(ctx, 100);

            s->frames_out++;
            return ff_filter_frame(outlink, frame);
        }

        /* Normal fill (not proactive mode) */
        if (frames_to_fill > 0) {
            av_log(ctx, AV_LOG_WARNING,
                   "[FILLGAPS] Wall clock gap detected: %"PRId64"ms > threshold %"PRId64"ms, "
                   "filling with %"PRId64" duplicate frames\n",
                   wallclock_gap / 1000, s->threshold / 1000, frames_to_fill);

            int64_t fill_pts = s->last_pts + s->frame_duration;

            for (int64_t i = 0; i < frames_to_fill; i++) {
                ret = output_dup_frame(ctx, outlink, s, fill_pts);
                if (ret < 0) break;
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
    s->last_output_time = now_us;
    s->output_pts = frame->pts + s->frame_duration;

    /* Update last_frame cache */
    av_frame_free(&s->last_frame);
    s->last_frame = av_frame_clone(frame);

    s->frames_out++;
    return ff_filter_frame(outlink, frame);
}

/**
 * Activate function for proactive output mode.
 * Called by filter graph even when no input is available.
 */
static int activate(AVFilterContext *ctx)
{
    FillGapsContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *frame = NULL;
    int ret, status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    /* Try to consume available input */
    ret = ff_inlink_consume_frame(inlink, &frame);
    if (ret < 0)
        return ret;

    if (frame)
        return process_frame(ctx, frame);

    /* No input frame available */

    /* Check if we should output a proactive frame */
    if (s->in_proactive_mode && s->proactive && s->last_frame) {
        int64_t now = av_gettime_relative();
        int64_t since_last_output = now - s->last_output_time;

        /* Should we have output a frame by now? */
        if (since_last_output >= s->frame_dur_us) {
            /* Check max_fill limit */
            int64_t proactive_duration = now - s->proactive_start;
            if (proactive_duration < s->max_fill) {
                /* Output proactive duplicate */
                ret = output_dup_frame(ctx, outlink, s, s->output_pts);
                if (ret < 0)
                    return ret;

                s->output_pts += s->frame_duration;
                s->last_output_time = now;
                s->proactive_frames++;

                av_log(ctx, AV_LOG_DEBUG,
                       "[FILLGAPS] Proactive output: PTS %"PRId64" (session frame %"PRId64")\n",
                       s->output_pts - s->frame_duration, s->proactive_frames);

                /* Schedule next activation */
                ff_filter_set_ready(ctx, 100);
                return 0;
            } else {
                /* Exceeded max_fill - exit proactive mode */
                av_log(ctx, AV_LOG_WARNING,
                       "[FILLGAPS] Exiting proactive mode: exceeded max_fill (%"PRId64"ms), "
                       "output %"PRId64" frames\n",
                       s->max_fill / 1000, s->proactive_frames);
                s->in_proactive_mode = 0;

                /* CRITICAL: Reset wall clock tracking to prevent immediate re-entry.
                 * Without this, the next input frame sees a huge wallclock_gap
                 * (since last_wallclock wasn't updated during proactive output)
                 * and immediately re-enters proactive mode. */
                s->last_wallclock = now;
                s->last_pts = s->output_pts - s->frame_duration;
                s->last_output_time = now;
            }
        } else {
            /* Not time yet - but schedule next check */
            ff_filter_set_ready(ctx, 100);
        }
    }

    /* Handle EOF */
    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        /* Flush any remaining state if needed */
        if (s->in_proactive_mode) {
            av_log(ctx, AV_LOG_INFO,
                   "[FILLGAPS] EOF during proactive mode, output %"PRId64" frames\n",
                   s->proactive_frames);
            s->in_proactive_mode = 0;
        }
        ff_outlink_set_status(outlink, status, s->output_pts);
        return 0;
    }

    /* Request more input */
    if (ff_outlink_frame_wanted(outlink))
        ff_inlink_request_frame(inlink);

    return FFERROR_NOT_READY;
}

static const AVFilterPad fillgaps_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
};

const FFFilter ff_vf_fillgaps = {
    .p.name        = "fillgaps",
    .p.description = NULL_IF_CONFIG_SMALL("Fill gaps in video by duplicating frames based on wall clock time."),
    .p.priv_class  = &fillgaps_class,
    .priv_size     = sizeof(FillGapsContext),
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(fillgaps_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
};
