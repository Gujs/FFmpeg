/*
 * ptvencoder — the resolution ladder as data (2.2, spec §16, step 4a).
 *
 *   -ladder 720p:autox720:2350k@227.111.4.8:6000,576p:autox576:1650k@227.111.4.8:5000,...
 *   [-max_res auto|off|<H>p] [-deint auto|on|off|film] [-timestamp_overlay off|on|<fontfile>] [-lo_at 480] [-b:a:lo 64k]
 *   <output options given once> ladder://?<udp query of every rung>
 *
 * Before the command line is parsed, the one output block is written out once per rung, exactly as the wrapper builds
 * it today: -map [v<label>] first, the block's options, -b:v and minrate / maxrate at -/+10 % with bufsize = maxrate,
 * the audio tier (-b:a:lo at or below -lo_at, else the block's -b:a; a rung's own :ABITRATE wins), and
 * udp://IP:PORT?<query>&bitrate=<2 x video + 2048k>&fifo_size=28672. After the input opens, -max_res drops the rungs
 * above the source height (never all of them), and the video graph is built: format, the timestamp overlay, hwupload +
 * bwdif_cuda / scale_cuda when the encoder runs on CUDA (CPU twins otherwise), split, one scaler per rung (width from
 * the source aspect, as today's -2:H). A -filter_complex given with -ladder is used as is (no -max_res trim then).
 * Single input only in this step; rung sizes fixed at output start with fit-and-pad come next (4b).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"

#include "cmdutils.h"
#include "ptvencoder.h"

#define LADDER_MAX 16
#define TS_FONT "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"

static struct {
    int n;
    struct { char label[16]; int h, vb; char abr[16]; } r[LADDER_MAX];
    const char *max_res, *deint, *ts;
} L = { .max_res = "auto", .deint = "auto", .ts = "off" };

int ptv_ladder_on(void) { return L.n > 0; }

/* LABEL:WxH:VBITRATE[:ABITRATE]@IP:PORT */
static int parse_rung(const char *s, int i, const char **dst)
{
    char buf[160], *at, *f[4], *sp = NULL, *w;
    int n = 0;
    av_strlcpy(buf, s, sizeof buf);
    if (!(at = strchr(buf, '@')) || !at[1]) return AVERROR(EINVAL);
    *at = 0;
    *dst = av_strdup(s + (at + 1 - buf));
    for (w = av_strtok(buf, ":", &sp); w && n < 4; w = av_strtok(NULL, ":", &sp)) f[n++] = w;
    if (n < 3 || !*dst) return AVERROR(EINVAL);
    av_strlcpy(L.r[i].label, f[0], sizeof L.r[i].label);
    if (strncmp(f[1], "autox", 5)) {
        av_log(NULL, AV_LOG_ERROR, "-ladder %s: an explicit width comes with fit-and-pad (4b); use autox<H>\n", s);
        return AVERROR(ENOSYS);
    }
    L.r[i].h  = atoi(f[1] + 5);
    L.r[i].vb = atoi(f[2]);
    if (n == 4) av_strlcpy(L.r[i].abr, f[3], sizeof L.r[i].abr);
    return L.r[i].h > 0 && L.r[i].vb > 0 ? 0 : AVERROR(EINVAL);
}

int ptv_ladder_expand(int *pargc, char ***pargv)
{
    int argc = *pargc, i, j, k, n = 0, last_in = -1, url = -1, lo_at = 480;
    char **argv = *pargv, **tpl, *dst[LADDER_MAX] = { 0 };
    const char **out, *spec = NULL, *lo_abr = NULL;

    for (i = 1; i < argc; i++)
        if (!strcmp(argv[i], "-ladder")) break;
    if (i >= argc) return 0;
    if (!(tpl = av_calloc(argc, sizeof(*tpl))) || !(out = av_calloc(argc * LADDER_MAX + 1, sizeof(*out))))
        return AVERROR(ENOMEM);
    /* take the ladder's own options out; find the last input and the pseudo output */
    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if      (!strcmp(a, "-ladder") && v)            { spec = v; i++; continue; }
        else if (!strcmp(a, "-max_res") && v)           { L.max_res = v; i++; continue; }
        else if (!strcmp(a, "-deint") && v)             { L.deint = v; i++; continue; }
        else if (!strcmp(a, "-timestamp_overlay") && v) { L.ts = v; i++; continue; }
        else if (!strcmp(a, "-lo_at") && v)             { lo_at = atoi(v); i++; continue; }
        else if (!strcmp(a, "-b:a:lo") && v)            { lo_abr = v; i++; continue; }
        if (!strcmp(a, "-i") && v) last_in = n + 1;
        if (!strncmp(a, "ladder://", 9)) url = n;
        tpl[n++] = argv[i];
    }
    if (strcmp(L.deint, "auto") && strcmp(L.deint, "on") && strcmp(L.deint, "off") && strcmp(L.deint, "film")) {
        av_log(NULL, AV_LOG_ERROR, "-deint %s: expected auto, on, off or film\n", L.deint);
        return AVERROR(EINVAL);
    }
    if (last_in < 0 || url != n - 1) {
        av_log(NULL, AV_LOG_ERROR, "-ladder: the output block must end in ladder://?<udp query> after the inputs\n");
        return AVERROR(EINVAL);
    }
    {   /* the rungs */
        char sbuf[2048], *w, *sp = NULL;
        av_strlcpy(sbuf, spec, sizeof sbuf);
        for (w = av_strtok(sbuf, ",", &sp); w; w = av_strtok(NULL, ",", &sp)) {
            int ret;
            if (L.n >= LADDER_MAX) return AVERROR(EINVAL);
            if ((ret = parse_rung(w, L.n, (const char **)&dst[L.n])) < 0) {
                if (ret != AVERROR(ENOSYS))
                    av_log(NULL, AV_LOG_ERROR, "-ladder rung '%s': expected LABEL:autoxH:VBITRATE[:ABITRATE]@IP:PORT\n", w);
                return ret;
            }
            for (j = 0; j < L.n; j++)
                if (!strcmp(L.r[j].label, L.r[L.n].label) || !strcmp(dst[j], dst[L.n])) {
                    av_log(NULL, AV_LOG_ERROR, "-ladder: rung %s repeats a label or a destination\n", w);
                    return AVERROR(EINVAL);
                }
            L.n++;
        }
    }
    /* argv: everything up to the last input, then one block per rung */
    k = 0;
    out[k++] = argv[0];
    for (i = 0; i <= last_in; i++) out[k++] = tpl[i];
    for (j = 0; j < L.n; j++) {
        int vb = L.r[j].vb, hi = vb + vb / 10, has_ba = 0;
        const char *abr = L.r[j].abr[0] ? L.r[j].abr : L.r[j].h <= lo_at && lo_abr ? lo_abr : NULL;
        const char *q = strchr(tpl[url], '?');
        out[k++] = "-map";
        out[k++] = av_asprintf("[v%s]", L.r[j].label);
        for (i = last_in + 1; i < url; i++) {
            out[k++] = tpl[i];
            if (!strcmp(tpl[i], "-b:a") && i + 1 < url) {   /* the rung's audio tier replaces the block's -b:a */
                i++;
                out[k++] = abr ? abr : tpl[i];
                has_ba = 1;
            }
        }
        if (abr && !has_ba) { out[k++] = "-b:a"; out[k++] = abr; }
        out[k++] = "-b:v";     out[k++] = av_asprintf("%dk", vb);
        out[k++] = "-minrate"; out[k++] = av_asprintf("%dk", vb - vb / 10);
        out[k++] = "-maxrate"; out[k++] = av_asprintf("%dk", hi);
        out[k++] = "-bufsize"; out[k++] = av_asprintf("%dk", hi);
        out[k++] = av_asprintf("udp://%s?%s%sbitrate=%d&fifo_size=28672", dst[j], q && q[1] ? q + 1 : "",
                               q && q[1] ? "&" : "", (2 * vb + 2048) * 1000);
        av_log(NULL, AV_LOG_INFO, "[PTV-LADDER] %s: %dp, video %dk (min %dk, max / buffer %dk), audio %s → %s\n",
               L.r[j].label, L.r[j].h, vb, vb - vb / 10, hi, abr ? abr : "as the block", dst[j]);
    }
    out[k] = NULL;
    if (getenv("PTV_LADDER_ARGS")) {   /* TEST: the expanded command, one argument per line (wrapper parity) */
        FILE *fp = fopen(getenv("PTV_LADDER_ARGS"), "w");
        for (i = 0; fp && i < k; i++) fprintf(fp, "%s\n", out[i]);
        if (fp) fclose(fp);
    }
    av_free(tpl);
    *pargc = k;
    *pargv = (char **)out;   /* lives until exit, like argv */
    return 0;
}

int ptv_ladder_finish(OptionGroupList *outs, int src_h, int cuda, const char **graph)
{
    AVBPrint bp;
    int cap = 0, keep[LADDER_MAX], nk = 0, j;
    if (!L.n) return 0;
    if (outs->nb_groups != L.n) {
        av_log(NULL, AV_LOG_ERROR, "-ladder: %d rungs but %d outputs — the ladder must be the only output\n",
               L.n, outs->nb_groups);
        return AVERROR(EINVAL);
    }
    if (!strcmp(L.max_res, "auto"))
        cap = src_h;
    else if (strcmp(L.max_res, "off")) {
        cap = atoi(L.max_res);
        if (cap <= 0) {
            av_log(NULL, AV_LOG_ERROR, "-max_res %s: expected auto, off or <H>p\n", L.max_res);
            return AVERROR(EINVAL);
        }
        if (src_h > 0 && src_h < cap) cap = src_h;   /* an explicit cap only lowers: never upscale */
    }
    for (j = 0; j < L.n; j++)
        if (cap <= 0 || L.r[j].h <= cap) keep[nk++] = j;
    if (!nk) keep[nk++] = L.n - 1;                   /* never trim to nothing: the lowest rung stays */
    if (nk < L.n) {
        if (*graph) {
            av_log(NULL, AV_LOG_ERROR, "-max_res %s trims the ladder, but -filter_complex was given — "
                   "drop it (ptvencoder builds the graph) or use -max_res off\n", L.max_res);
            return AVERROR(EINVAL);
        }
        for (j = 0; j < nk; j++) outs->groups[j] = outs->groups[keep[j]];
        av_log(NULL, AV_LOG_INFO, "[PTV-LADDER] capped at %dp (-max_res %s, source %dp): %d of %d rungs, from %s\n",
               cap, L.max_res, src_h, nk, L.n, L.r[keep[0]].label);
        outs->nb_groups = nk;   /* the dropped groups' option memory stays until exit */
    }
    if (*graph) return 0;
    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);
    av_bprintf(&bp, "[0:v]format=yuv420p");
    /* 2.2 T-421: film = inverse telecine in front of the deinterlacer (CPU, ahead of the upload and the clock overlay):
     * fieldmatch weaves each field with its own picture's other field and leaves flagged interlaced only what is still
     * combed (camera video, a video ticker); bwdif deint=interlaced takes exactly those. No decimate: the rate stays.
     * Fixtures (test-results/t421): 3:2 / 2:2 film 34.6 / 33.5 -> 60.7 / 60.9 dB vs the master; camera identical;
     * fieldmatch 7.1 ms / 1080i frame on one core */
    if (!strcmp(L.deint, "film"))
        av_bprintf(&bp, ",fieldmatch=order=auto:combmatch=full");
    if (strcmp(L.ts, "off"))
        av_bprintf(&bp, ",drawtext=fontfile=%s:text='%%{localtime}':fontcolor=white:fontsize=h/30:box=1:"
                   "boxcolor=black@0.6:boxborderw=5:x=w*0.02:y=h*0.14", strcmp(L.ts, "on") ? L.ts : TS_FONT);
    if (cuda) av_bprintf(&bp, ",hwupload");
    if (strcmp(L.deint, "off"))
        av_bprintf(&bp, ",bwdif%s=mode=send_frame:parity=auto:deint=%s", cuda ? "_cuda" : "",
                   strcmp(L.deint, "on") ? "interlaced" : "all");
    av_bprintf(&bp, ",split=%d", nk);
    for (j = 0; j < nk; j++) av_bprintf(&bp, "[split%d]", j + 1);
    for (j = 0; j < nk; j++)
        av_bprintf(&bp, cuda ? ";[split%d]scale_cuda=-2:%d:interp_algo=2[v%s]" : ";[split%d]scale=-2:%d:flags=bilinear[v%s]",
                   j + 1, L.r[keep[j]].h, L.r[keep[j]].label);
    if (!av_bprint_is_complete(&bp)) return AVERROR(ENOMEM);
    av_bprint_finalize(&bp, (char **)graph);
    av_log(NULL, AV_LOG_INFO, "[PTV-LADDER] graph: %s\n", *graph);
    return 0;
}
