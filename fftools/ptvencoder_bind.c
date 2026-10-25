/*
 * ptvencoder — audio track rules (2.2, spec §7): -map K:a:main | a:langs | a:lang=xxx | a:multich | a:pid=0x.. |
 * a:index=N, each optionally followed by ",key=value" track options. A rule is resolved to concrete source streams
 * when the output starts; each resulting stream is then handled exactly like a classic -map K:a:N.
 *
 * The steps of §7.1, in order: relabel (-audio_lang_override) → roles (audio description never main, never in a
 * mosaic; receiver-mix AD dropped) → classify (HQ stereo / multichannel / LQ stereo) → pick (HQ > multichannel
 * downmix > LQ) → dedup (same language + role = one survivor: a lone broadcaster default flag, else
 * -audio_codec_pref, else more channels, else source order).
 */

#include <stdlib.h>
#include <string.h>

#include "libavformat/avformat.h"
#include "libavutil/avstring.h"
#include "libavutil/dict.h"

#include "ptvencoder.h"

const char *g_audio_lang_override;               /* -audio_lang_override KEY:LANG,… (KEY = 0xPID, audio index, language) */
const char *g_audio_codec_pref = "eac3,ac3,aac,mp2,mp3";   /* -audio_codec_pref: dedup survivor order */
int         g_audio_dedup = 1;                   /* -audio_dedup on|off */
int         g_audio_dedup_chan_strict;           /* -audio_dedup_chan_strict: channel count also splits groups */

enum { B_HQ, B_MC, B_LQ };
#define ROLE_FLAGS (AV_DISPOSITION_VISUAL_IMPAIRED | AV_DISPOSITION_HEARING_IMPAIRED | AV_DISPOSITION_COMMENT | \
                    AV_DISPOSITION_DESCRIPTIONS | AV_DISPOSITION_DUB | AV_DISPOSITION_KARAOKE)

typedef struct Cand {
    int  si, rel;                                 /* stream index, audio-relative index */
    char lang[8];
    int  nch, bucket, role, ad, plain, dflt, pref;
} Cand;

static int is_ad(int disp)
{
    return !!(disp & (AV_DISPOSITION_VISUAL_IMPAIRED | AV_DISPOSITION_DESCRIPTIONS));
}

static int codec_pref(enum AVCodecID id)
{
    const char *name = avcodec_get_name(id), *q = g_audio_codec_pref;
    int r = 0;
    size_t nl = strlen(name);
    while (q && *q) {
        const char *e = strchr(q, ',');
        size_t l = e ? (size_t)(e - q) : strlen(q);
        if (l == nl && !strncmp(q, name, l)) return r;
        r++;
        if (!e) break;
        q = e + 1;
    }
    return r + 1;                                 /* unlisted codecs rank last */
}

static int bucket_of(const AVCodecParameters *p)
{
    int nch = p->ch_layout.nb_channels;
    if (nch > 2) return B_MC;
    switch (p->codec_id) {
    case AV_CODEC_ID_AAC: case AV_CODEC_ID_AAC_LATM: case AV_CODEC_ID_AC3: case AV_CODEC_ID_EAC3:
    case AV_CODEC_ID_DTS: case AV_CODEC_ID_TRUEHD: case AV_CODEC_ID_OPUS: case AV_CODEC_ID_FLAC:
        return B_HQ;
    default:
        return !strncmp(avcodec_get_name(p->codec_id), "pcm_", 4) ? B_HQ : B_LQ;
    }
}

/* §7.1 step 1: KEY:LANG entries, left to right (a later entry sees an earlier relabel) */
static void relabel(Cand *c, int n, const AVFormatContext *f, int fidx, int verbose)
{
    const char *q = g_audio_lang_override;
    while (q && *q) {
        char ent[32], *colon;
        const char *e = strchr(q, ',');
        size_t l = e ? (size_t)(e - q) : strlen(q);
        int i, hit = 0;
        av_strlcpy(ent, q, FFMIN(l + 1, sizeof ent));
        q = e ? e + 1 : q + l;
        if (!(colon = strchr(ent, ':')) || colon == ent || !colon[1]) {
            if (verbose) av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] -audio_lang_override entry '%s' is not KEY:LANG — ignored\n", ent);
            continue;
        }
        *colon = 0;
        for (i = 0; i < n; i++) {
            int match = !strncmp(ent, "0x", 2) ? f->streams[c[i].si]->id == (int)strtol(ent, NULL, 16)
                      : (ent[0] >= '0' && ent[0] <= '9') ? c[i].rel == atoi(ent)
                      : !strcmp(c[i].lang, ent);
            if (!match) continue;
            if (verbose)
                av_log(NULL, AV_LOG_INFO, "[PTV-BIND] in%d 0x%x relabelled %s → %s\n",
                       fidx, f->streams[c[i].si]->id, c[i].lang, colon + 1);
            av_strlcpy(c[i].lang, colon + 1, sizeof c[i].lang);
            hit = 1;
        }
        if (!hit && verbose)
            av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] -audio_lang_override %s: no source track matches — ignored\n", ent);
    }
}

/* dedup survivor among m[0..n): a lone default flag, else codec preference, else more channels, else source order */
static const Cand *survivor(const Cand **m, int n)
{
    const Cand *w = NULL;
    int i, nd = 0;
    for (i = 0; i < n; i++) if (m[i]->dflt) { nd++; w = m[i]; }
    if (nd == 1) return w;
    w = m[0];
    for (i = 1; i < n; i++)
        if (m[i]->pref < w->pref || (m[i]->pref == w->pref && m[i]->nch > w->nch)) w = m[i];
    return w;
}

static int same_group(const Cand *a, const Cand *b)
{
    return !strcmp(a->lang, b->lang) && a->role == b->role && (!g_audio_dedup_chan_strict || a->nch == b->nch);
}

/* the best of the group of g (members of c[] in g's group, from bucket order HQ > MC > LQ; mc_only = multichannel
 * only); dedup off = every member of that bucket. Returns the number written to out. */
static int pick_group(const Cand *c, int n, const Cand *g, int mc_only, const Cand **out, int max)
{
    const Cand *m[64];
    int b, i, k = 0;
    for (b = mc_only ? B_MC : B_HQ; b <= (mc_only ? B_MC : B_LQ) && !k; b++)
        for (i = 0; i < n && k < 64; i++)
            if (c[i].bucket == b && same_group(&c[i], g)) m[k++] = &c[i];
    if (!k) return 0;
    if (g_audio_dedup) { out[0] = survivor(m, k); return 1; }
    for (i = 0; i < k && i < max; i++) out[i] = m[i];
    return i;
}

/* main audio: the best bucket present; inside it the first group (source order) of a plain (non-commentary / dub)
 * track, else the first group; its dedup survivor */
static const Cand *pick_main(const Cand *c, int n)
{
    int b, i, pass;
    for (b = B_HQ; b <= B_LQ; b++)
        for (pass = 0; pass < 2; pass++)
            for (i = 0; i < n; i++)
                if (c[i].bucket == b && !c[i].ad && (pass || c[i].plain)) {
                    const Cand *m[64];
                    int j, k = 0;
                    for (j = 0; j < n && k < 64; j++)
                        if (c[j].bucket == b && same_group(&c[j], &c[i])) m[k++] = &c[j];
                    return survivor(m, k);
                }
    return NULL;
}

int ptv_bind_parse(const char *spec, PtvBindRule *r)
{
    static const char *kinds[] = { "main", "langs", "lang=", "multich", "pid=", "index=" };
    char *tok, *sp = NULL;
    int k;
    memset(r, 0, sizeof *r);
    if (strncmp(spec, "a:", 2)) return 0;
    for (k = 0; k < 6; k++) {
        size_t l = strlen(kinds[k]);
        if (!strncmp(spec + 2, kinds[k], l) && (kinds[k][l - 1] == '=' || !spec[2 + l] || spec[2 + l] == ','))
            break;
    }
    if (k == 6) return 0;                         /* a classic specifier (a:1, a:m:language:eng, …) */
    r->kind = k;
    if (av_strlcpy(r->buf, spec + 2, sizeof r->buf) >= sizeof r->buf) return AVERROR(EINVAL);
    tok = av_strtok(r->buf, ",", &sp);
    if (strchr(tok, '=')) r->arg = strchr(tok, '=') + 1;
    if (r->arg && !r->arg[0]) return AVERROR(EINVAL);
    while ((tok = av_strtok(NULL, ",", &sp))) {
        char *v = strchr(tok, '=');
        if (!v) return AVERROR(EINVAL);
        *v++ = 0;
        if      (!strcmp(tok, "c"))       r->c = v;
        else if (!strcmp(tok, "b"))       r->b = v;
        else if (!strcmp(tok, "ac"))      r->ac = v;
        else if (!strcmp(tok, "af"))      r->af = v;
        else if (!strcmp(tok, "lang"))    r->lang = v;
        else if (!strcmp(tok, "label"))   r->label = v;
        else if (!strcmp(tok, "title"))   r->title = v;
        else if (!strcmp(tok, "default")) r->dflt = atoi(v);
        else return AVERROR(EINVAL);
    }
    return 1;
}

static const char *kind_name(int k)
{
    static const char *n[] = { "a:main", "a:langs", "a:lang", "a:multich", "a:pid", "a:index" };
    return n[k];
}

/* every audio stream of f as a candidate, relabelled (§7.1 step 1) */
static int candidates(const AVFormatContext *f, int fidx, Cand *c, int verbose)
{
    int i, n = 0, rel = 0;
    for (i = 0; i < (int)f->nb_streams && n < 64; i++) {
        const AVStream *st = f->streams[i];
        const AVDictionaryEntry *lg;
        Cand *x;
        if (st->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
        x = &c[n++];
        memset(x, 0, sizeof *x);
        x->si = i; x->rel = rel++;
        lg = av_dict_get(st->metadata, "language", NULL, 0);
        av_strlcpy(x->lang, lg && lg->value[0] ? lg->value : "und", sizeof x->lang);
        x->nch    = st->codecpar->ch_layout.nb_channels;
        x->bucket = bucket_of(st->codecpar);
        x->ad     = is_ad(st->disposition);
        x->role   = st->disposition & ROLE_FLAGS;
        x->plain  = !(st->disposition & (AV_DISPOSITION_COMMENT | AV_DISPOSITION_DUB));
        x->dflt   = !!(st->disposition & AV_DISPOSITION_DEFAULT);
        x->pref   = codec_pref(st->codecpar->codec_id);
    }
    relabel(c, n, f, fidx, verbose);
    return n;
}

int ptv_bind_ident(const AVFormatContext *f, int fidx, int si, char *lang, int *role)
{
    Cand c[64];
    int i, n = candidates(f, fidx, c, 0);
    for (i = 0; i < n; i++)
        if (c[i].si == si) {
            av_strlcpy(lang, c[i].lang, 8);
            *role = c[i].role;
            return 0;
        }
    return AVERROR(EINVAL);
}

int ptv_bind_audio(const AVFormatContext *f, int fidx, const PtvBindRule *r, int mosaic,
                   PtvBindPick *out, int max, int verbose)
{
    return ptv_bind_audio_in(f, fidx, r, mosaic, 0, NULL, 0, out, max, verbose);
}

int ptv_bind_audio_in(const AVFormatContext *f, int fidx, const PtvBindRule *r, int mosaic,
                      uint64_t present, const char *id_lang, int id_role, PtvBindPick *out, int max, int verbose)
{
    Cand c[64];
    const Cand *sel[64];
    int i, n = candidates(f, fidx, c, verbose), ns = 0;

    if (r->kind == PTV_BIND_PID || r->kind == PTV_BIND_INDEX) {   /* static pins: no roles, no dedup */
        int v = r->kind == PTV_BIND_PID ? (int)strtol(r->arg, NULL, 0) : atoi(r->arg);
        for (i = 0; i < n; i++)
            if (r->kind == PTV_BIND_PID ? f->streams[c[i].si]->id == v : c[i].rel == v) { sel[ns++] = &c[i]; break; }
    } else {
        /* roles: receiver-mix AD is narration without the programme; a mosaic never carries AD; the lang= filter */
        int m = 0;
        for (i = 0; i < n; i++) {
            const AVStream *st = f->streams[c[i].si];
            const char *want = r->kind == PTV_BIND_LANG ? r->arg : r->lang;
            if (c[i].ad && ((st->disposition & AV_DISPOSITION_DEPENDENT) || mosaic)) continue;
            if (want && strcmp(c[i].lang, want)) continue;
            if (present && (c[i].si >= 64 || !(present >> c[i].si & 1))) continue;   /* a re-bind: what is there now */
            if (id_lang && (strcmp(c[i].lang, id_lang) || c[i].role != id_role)) continue;   /* ... in its group */
            c[m++] = c[i];
        }
        n = m;
        if (r->kind == PTV_BIND_MAIN || r->kind == PTV_BIND_LANG) {
            const Cand *w = pick_main(c, n);
            if (w) sel[ns++] = w;
        } else {                                  /* a:langs / a:multich: one pick per language + role group */
            const Cand *g[64];
            int ng = 0, j;
            if (r->kind == PTV_BIND_LANGS && (g[0] = pick_main(c, n)))
                ng = 1;                           /* the main track's group first */
            for (i = 0; i < n; i++) {
                for (j = 0; j < ng && !same_group(g[j], &c[i]); j++) ;
                if (j == ng) g[ng++] = &c[i];
            }
            for (j = 0; j < ng && ns < 64; j++)
                ns += pick_group(c, n, g[j], r->kind == PTV_BIND_MULTICH, &sel[ns], 64 - ns);
        }
    }

    for (i = 0; i < ns && i < max; i++) {
        const AVStream *st = f->streams[sel[i]->si];
        PtvBindPick *p = &out[i];
        int stereo_from_mc = r->kind != PTV_BIND_MULTICH && sel[i]->bucket == B_MC;
        p->stream = sel[i]->si;
        av_strlcpy(p->lang, r->label ? r->label : sel[i]->lang, sizeof p->lang);
        av_strlcpy(p->key_lang, sel[i]->lang, sizeof p->key_lang);
        p->disp = (st->disposition & ~AV_DISPOSITION_DEFAULT) |   /* the source's flags carry through (§7.2); */
                  (sel[i]->ad ? AV_DISPOSITION_VISUAL_IMPAIRED : 0) |   /* the rule decides default; AD says VI */
                  ((r->dflt || (i == 0 && !sel[i]->ad && (r->kind == PTV_BIND_MAIN || r->kind == PTV_BIND_LANGS)))
                   ? AV_DISPOSITION_DEFAULT : 0);
        if (verbose)
            av_log(NULL, AV_LOG_INFO, "[PTV-BIND] %-9s → in%d 0x%x %s %dch %s%s%s%s\n",
                   kind_name(r->kind), fidx, st->id, avcodec_get_name(st->codecpar->codec_id), sel[i]->nch, sel[i]->lang,
                   sel[i]->ad ? " AD" : "", stereo_from_mc ? " (downmix)" : "",
                   p->disp & AV_DISPOSITION_DEFAULT ? " default" : "");
    }
    if (!ns && verbose)
        av_log(NULL, AV_LOG_WARNING, "[PTV-BIND] %s%s%s → in%d: no source track qualifies\n",
               kind_name(r->kind), r->arg ? "=" : "", r->arg ? r->arg : "", fidx);
    return FFMIN(ns, max);
}
