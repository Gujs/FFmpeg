/*
 * ptvencoder — 2.1 T-409: the program timeline, source stage (M_in).
 *
 * ONE classifier per input decides every source timing event and applies ONE remedy to every stream of the input
 * (design: analysis/t409-timeline-map-design.md). Since 2.1.0 it is the only path: it replaces the demux absorber, the
 * wall-evidence split, the LAYERA buffer and the door's downstream re-discovery of its own cut (their code stays, not
 * run, until the 2.1 canary soak — design §7 P5 — then it is deleted).
 *
 * The remedy is the program offset k: every packet of the input leaves the demux labelled L = S + k, S = its source
 * time after the 33-bit unwrap. k is realized through the existing offset state — wrap_off[] of every dense stream and
 * prog_off of the sparse ones (so SCTE-35 pts_adjustment and the copies follow unchanged) — and only this file writes it
 * (one writer). An event changes k by whole pictures, for all streams at once: after the demux, the labels of every
 * stream are continuous except at holes deliberately kept.
 *
 * An event: the first dense packet more than PTV_TL_OPEN_MS off its stream's path opens a window; every packet of the
 * input is held until each flowing dense stream has resolved (crossed to a new path, come back to the old one, or
 * passed the crossing point without crossing) or the window times out. Then, once:
 *   - packets off every path that do not continue (one corrupt PES, the first piece of a split step) are dropped;
 *   - the video and at least one audio crossed by the same amount: a PROGRAM event. J = the video's step on its picture
 *     grid; W = the input's wall silence before it (no evidence under an armed bank / deep prime), or under damage the
 *     damage phase's duration if longer. keep = min(J, W) in whole pictures (a step back after a wall gap = a new domain: keep W),
 *     erase = J − keep, k −= erase. The sound after the last
 *     old picture and before the first new one is dropped; the audio's sub-frame rest is the audio's own continuity;
 *   - only one stream crossed: a single-stream step, labels as sent (the content path pads / trims, RELABEL may repair).
 */

#include "ptvencoder.h"
#include "libavutil/mathematics.h"
#include "libavutil/time.h"

#define TL_HELD_MAX   4096
#define TL_WIN_US     1000000       /* an event window closes after 1 s of wall at the latest */
#define TL_PASS_US    300000        /* a partner whose own timeline passed the crossing point by this did not cross */
#define TL_STRAG_US   2000000       /* after an event, packets on a stream's old path are stragglers for this long */

typedef struct TlPkt {
    AVPacket *pkt;
    int64_t   s_us;    /* source time (DTS, or PTS when no DTS), us; AV_NOPTS_VALUE = none */
    int64_t   k_at;    /* the k its labels were given by demux_unwrap */
    int64_t   wc;      /* arrival wall */
    int       si;
    int       path;    /* 0 old, 1 new, 2 drop */
} TlPkt;

typedef struct TlStream {
    int     dense, ok;
    int64_t last_us, step_us, last_wc;
    int64_t strag_last_us, strag_until_wc;   /* the old path after an event (stragglers are dropped) */
    /* classification scratch */
    int     res;        /* 0 unresolved/old, 1 crossed, 2 came back (outlier) */
    int64_t old_end_us, new_first_us, new_last_us;
    int     new_n;
    int     win_passed;                       /* the video went on on its old path during this window (dispatched) */
    int64_t rep_us, rep_new;                  /* label repair held on this stream (us) / this window's change of it */
    int     rl_mark;                          /* flag the first repaired packet for the T-087 content check */
} TlStream;

struct PtvTimeline {
    int64_t   k_us;
    int64_t   ev_n, strag_n, drop_n;
    int       active, trig, nst, cap;   /* nst = streams followed; cap = st[] room (2.2: adoption) */
    int64_t   open_wc, open_s_us;
    int64_t   last_any_wc, gap_us, gap_end_wc;   /* the input's last silence >= g_gap_min_us */
    int64_t   dmg_seen_wc, dmg_start_wc;         /* the input's damage phase: corrupt packets <= 3 s apart */
    int64_t   er_lo, er_hi;                      /* source time the last program event erased (audio arriving late) */
    int64_t   ev_va, ev_dk, ev_wc;               /* the last program event: old content ends at ev_va; its offset move */
    _Atomic int64_t prog_wc;                     /* wall time of the last program event (read by the copy path) */
    TlStream *st;
    TlPkt     held[TL_HELD_MAX];
    int       nheld;
};

int64_t g_tl_open_us = 500000;

PtvTimeline *ptv_tl_alloc(int nb_streams, int cap)
{
    PtvTimeline *t = av_mallocz(sizeof(*t));
    if (!t)
        return NULL;
    t->cap = FFMAX(cap, nb_streams);
    t->st = av_calloc(FFMAX(t->cap, 1), sizeof(*t->st));
    if (!t->st) {
        av_free(t);
        return NULL;
    }
    t->nst = nb_streams;
    return t;
}

void ptv_tl_grow(PtvTimeline *t, int nb_streams)
{
    if (t && nb_streams > t->nst && nb_streams <= t->cap)
        t->nst = nb_streams;                   /* the new slot is zeroed: a stream that has not started */
}

void ptv_tl_free(PtvTimeline **pt)
{
    PtvTimeline *t = *pt;
    int i;
    if (!t)
        return;
    for (i = 0; i < t->nheld; i++)
        av_packet_free(&t->held[i].pkt);
    av_free(t->st);
    av_freep(pt);
}

static int is_dense(const DemuxArgs *d, int si)
{
    int ct = d->ifmt->streams[si]->codecpar->codec_type;
    return ct == AVMEDIA_TYPE_VIDEO || ct == AVMEDIA_TYPE_AUDIO;
}

static int64_t pkt_s_us(const DemuxArgs *d, const AVPacket *p, int64_t k_us)
{
    int64_t ts = p->dts != AV_NOPTS_VALUE ? p->dts : p->pts;
    if (ts == AV_NOPTS_VALUE)
        return AV_NOPTS_VALUE;
    return av_rescale_q(ts, d->ifmt->streams[p->stream_index]->time_base, AV_TIME_BASE_Q) - k_us;
}

static int64_t pkt_dur_us(const DemuxArgs *d, const AVPacket *p)
{
    const AVStream *st = d->ifmt->streams[p->stream_index];
    if (p->duration > 0)
        return av_rescale_q(p->duration, st->time_base, AV_TIME_BASE_Q);
    if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && st->avg_frame_rate.num > 0)
        return av_rescale(AV_TIME_BASE, st->avg_frame_rate.den, st->avg_frame_rate.num);
    if (st->codecpar->sample_rate > 0 && st->codecpar->frame_size > 0)
        return av_rescale(AV_TIME_BASE, st->codecpar->frame_size, st->codecpar->sample_rate);
    return 0;
}

static int64_t tol_us(const TlStream *s)
{
    /* two of the stream's own steps: a backward step of 3 pictures (gate sync_bstep -120 ms at 25p) leaves the path */
    return av_clip64(2 * s->step_us, 40000, g_tl_open_us / 2);
}

/* a window's 1 s runs on live input: an input silence inside it restarts it, so every stream can still cross after the
 * input returns. Decided while the input was silent, an audio-first crossing went out as a single-stream step with the old
 * offset and the video's erase a moment later left it 16.8 s off its pictures (sync_storm 430 s) */
static int64_t win_from(const PtvTimeline *t)
{
    return FFMAX(t->open_wc, t->gap_end_wc);
}

/* a packet off its stream's path by this opens a window: forward the open threshold (smaller forward steps are the
 * stream's loss, padded), backward from g_aglue_ms — the demux absorber's backward band (80 ms) that the timeline replaces;
 * a both-stream step of -120 ms passed unclassified (copied subtitles / SCTE-35 / teletext 2 frames off its picture) */
static int off_path(const TlStream *s, int64_t s_us)
{
    int64_t dev = s_us - (s->last_us + s->step_us);
    return dev > g_tl_open_us || dev < -FFMAX((int64_t)g_aglue_ms * 1000, tol_us(s));
}

/* the normal-path bookkeeping of a packet on its stream's path */
static void path_take(TlStream *s, int64_t s_us, int64_t wc)
{
    if (s->ok) {
        int64_t dl = s_us - s->last_us;
        /* the stream's own cadence: only deltas near it (a gap must not teach it — one 192 ms burst seam on 21 ms AAC
         * made every following frame read 171 ms behind, and the relabel repair ran away: +0.17 s per frame, cap-nmx) */
        if (dl > 0 && dl <= 250000 && (!s->step_us || (dl > s->step_us / 2 && dl < 2 * s->step_us)))
            s->step_us = s->step_us ? s->step_us + (dl - s->step_us) / 8 : dl;
    }
    s->last_us = s_us;
    s->last_wc = wc;
    s->ok      = 1;
}

/* shift a packet's labels from the k it was given to k_to */
static void relabel(const DemuxArgs *d, AVPacket *p, int64_t k_from, int64_t k_to)
{
    const AVStream *st = d->ifmt->streams[p->stream_index];
    int64_t dl;
    if (k_to == k_from)
        return;
    dl = av_rescale_q(k_to - k_from, AV_TIME_BASE_Q, st->time_base);
    if (p->pts != AV_NOPTS_VALUE) p->pts += dl;
    if (p->dts != AV_NOPTS_VALUE) p->dts += dl;
}

/* move the program offset: every dense stream's wrap_off and the sparse streams' prog_off, both edit ledgers alike */
static void k_move(DemuxArgs *d, PtvTimeline *t, int64_t dk_us)
{
    unsigned s;
    if (!dk_us)
        return;
    for (s = 0; s < d->ifmt->nb_streams && s < (unsigned)d->nb_streams_open; s++) {
        if (!is_dense(d, s))
            continue;
        d->wrap_off[s] += av_rescale_q(dk_us, AV_TIME_BASE_Q, d->ifmt->streams[s]->time_base);
        ptv_rsync_post_edit(d, s, dk_us);
    }
    d->prog_off += av_rescale_q(dk_us, AV_TIME_BASE_Q, (AVRational){ 1, 90000 });
    t->k_us     += dk_us;
}

/* Walk one stream's held packets (arrival order) against its path: mark old / new (the run it ends on) / drop (runs
 * that do not continue). Fills the scratch fields; returns the stream's resolution. `final` = the window is closing. */
static int classify_stream(PtvTimeline *t, int si, int final, int64_t cross_s_us)
{
    TlStream *s = &t->st[si];
    int64_t old_last = s->last_us, step = s->step_us ? s->step_us : 20000, tol = tol_us(s);
    int64_t run_first = AV_NOPTS_VALUE, run_last = AV_NOPTS_VALUE, run_base = AV_NOPTS_VALUE;
    int run_n = 0, i, back_n = 0, seen = 0, ended_old = 0;

    s->res = 0; s->new_n = 0; s->old_end_us = old_last;
    for (i = 0; i < t->nheld; i++) {
        TlPkt *h = &t->held[i];
        if (h->si != si || h->s_us == AV_NOPTS_VALUE)
            continue;
        seen++;
        /* a run that goes on at its own cadence stays the run where it crosses the old path's expected position: a
         * program-wide step back of a few pictures overlaps the old labels (gate sync_bstep -120 ms read as an outlier, its
         * first pictures dropped). A corrupt run that returns jumps back — it does not continue itself */
        if (run_n && llabs(h->s_us - (run_last + step)) <= tol) {
            h->path = 1;
            run_last = h->s_us;
            run_n++;
            ended_old = 0;
            continue;
        }
        if (!s->ok || llabs(h->s_us - (old_last + step)) <= tol) {
            h->path = 0;
            old_last = h->s_us;
            ended_old = 1;
            if (run_n)
                back_n++;
            continue;
        }
        ended_old = 0;
        /* a new run: an earlier one did not continue */
        if (run_n) {
            int j;
            for (j = 0; j < i; j++)
                if (t->held[j].si == si && t->held[j].path == 1)
                    t->held[j].path = 2;
        }
        h->path   = 1;
        run_first = run_last = h->s_us;
        run_base  = old_last;
        run_n     = 1;
        back_n    = 0;
    }
    /* a lone packet after an input silence whose jump the silence explains (J ~ W: live content resuming between two
     * outages of a storm) is no corrupt outlier — the wall clock corroborates it (sync_storm 430 s: such scraps dropped,
     * and the next event merged two holes into one J against one W) */
    if (final && run_n == 1 && !ended_old && t->gap_us >= g_gap_min_us &&
        llabs((run_first - (run_base + step)) - t->gap_us) <= FFMAX(1000000, t->gap_us / 10)) {
        s->res          = 1;
        s->old_end_us   = run_base;
        s->new_first_us = run_first;
        s->new_last_us  = run_last;
        s->new_n        = 1;
        return 1;
    }
    if (run_n && (ended_old || (final && run_n < 2))) {
        /* came back to the old path (or a lone packet at the end): the run was corrupt */
        for (i = 0; i < t->nheld; i++)
            if (t->held[i].si == si && t->held[i].path == 1)
                t->held[i].path = 2;
        s->res = back_n || final ? 2 : 0;
        s->old_end_us = old_last;
        return s->res;
    }
    if (run_n >= 2 || (final && run_n)) {
        s->res          = 1;
        s->old_end_us   = run_base;
        s->new_first_us = run_first;
        s->new_last_us  = run_last;
        s->new_n        = run_n;
        return 1;
    }
    if (run_n)
        return 0;                              /* one packet off: wait for the next */
    s->old_end_us = old_last;
    /* never left its path: resolved "did not cross" once its own time passed the crossing point */
    if (cross_s_us != AV_NOPTS_VALUE && (seen || s->win_passed) && old_last > cross_s_us + TL_PASS_US)
        return 3;
    return 0;
}

/* a stream was flowing if it delivered within 2 s of the input's last activity before the event (before the silence
 * the window opened behind, if any) */
static int flowing(const PtvTimeline *t, const TlStream *s)
{
    int64_t ref = t->gap_end_wc && t->open_wc - t->gap_end_wc <= 1000000 ? t->gap_end_wc - t->gap_us : t->open_wc;
    return s->dense && s->ok && ref - s->last_wc <= 2000000;
}

static int has_held(const PtvTimeline *t, int si)
{
    int i;
    for (i = 0; i < t->nheld; i++)
        if (t->held[i].si == si && t->held[i].s_us != AV_NOPTS_VALUE)
            return 1;
    return 0;
}

static int window_resolved(DemuxArgs *d, PtvTimeline *t, int final)
{
    int si, all = 1, r;
    int64_t cross = AV_NOPTS_VALUE;
    r = classify_stream(t, t->trig, final, AV_NOPTS_VALUE);
    if (r == 1)
        cross = t->st[t->trig].old_end_us;
    else if (r != 2)
        return final;
    if (r == 2)
        return 1;                              /* the trigger was one corrupt run: nothing else to wait for */
    for (si = 0; si < t->nst; si++) {
        TlStream *s = &t->st[si];
        if (si == t->trig || !s->dense)
            continue;
        if (!flowing(t, s) && !(final && has_held(t, si)))
            continue;                          /* not flowing before the event: no partner to wait for */
        r = classify_stream(t, si, final, cross);
        if (r == 0 && flowing(t, s))
            all = 0;
    }
    return all || final;
}

static int64_t input_wall_gap(PtvTimeline *t, int64_t first_new_wc)
{
    int64_t w = 0, prev = AV_NOPTS_VALUE;
    int i;
    if (t->gap_end_wc && t->open_wc - t->gap_end_wc <= 1000000)
        w = t->gap_us;                         /* the silence the window opened behind (the PES tail came first) */
    for (i = 0; i < t->nheld && t->held[i].wc <= first_new_wc; i++) {
        if (prev != AV_NOPTS_VALUE && t->held[i].wc - prev > w)
            w = t->held[i].wc - prev;
        prev = t->held[i].wc;
    }
    return w >= g_gap_min_us ? w : 0;
}

static int release(DemuxArgs *d, PtvTimeline *t, int64_t k_old, int64_t k_new, int prog_event,
                   int64_t va_us, int64_t vb_us)
{
    int i, ret = 0;
    for (i = 0; i < t->nheld; i++) {
        TlPkt *h = &t->held[i];
        AVPacket *p = h->pkt;
        int64_t k = k_old;
        h->pkt = NULL;
        if (h->path == 2) {
            t->drop_n++;
            av_packet_free(&p);
            continue;
        }
        if (h->s_us != AV_NOPTS_VALUE && h->si < t->nst && t->st[h->si].dense) {
            if (h->path == 1 && t->st[h->si].res == 1) {
                k = k_new + t->st[h->si].rep_new;   /* labels + repair (relabel adds k_to - k_from) */
                if (t->st[h->si].rl_mark) {
                    p->flags |= PTV_PKT_FLAG_RELABEL;
                    t->st[h->si].rl_mark = 0;
                }
            }
            if (prog_event && d->ifmt->streams[h->si]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                int64_t end = h->s_us + pkt_dur_us(d, p);
                /* the sound after the last old picture / before the first new one belongs to the erased span */
                if ((k == k_old && h->s_us >= va_us) || (k == k_new && end <= vb_us)) {
                    t->drop_n++;
                    av_packet_free(&p);
                    continue;
                }
            }
        } else if (prog_event && h->s_us != AV_NOPTS_VALUE) {
            /* sparse: the nearer side; held in a step-back window they were read after the step began: the new one */
            k = vb_us < va_us || llabs(h->s_us - vb_us) < llabs(h->s_us - va_us) ? k_new : k_old;
        }
        relabel(d, p, h->k_at, k);
        if (ret >= 0)
            ret = ptv_demux_dispatch(d, p);
        else
            av_packet_free(&p);
    }
    t->nheld = 0;
    return ret;
}

static int close_window(DemuxArgs *d, PtvTimeline *t)
{
    int si, vs = d->vstream, n_aud_cross = 0, n_aud_flow = 0, prog_event = 0, dmg = 0;
    int64_t k_old = t->k_us, k_new = t->k_us, J = 0, W = 0, keep = 0, erase = 0, fd = 0, dspan = 0;
    int64_t va = AV_NOPTS_VALUE, vb = AV_NOPTS_VALUE, first_new_wc = INT64_MAX, nw = av_gettime_relative();
    const char *cls = "outlier";
    int ret;

    window_resolved(d, t, 1);
    for (si = 0; si < t->nst; si++) {
        TlStream *s = &t->st[si];
        if (!s->dense || si == vs)
            continue;
        if (d->ifmt->streams[si]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && (flowing(t, s) || has_held(t, si)))
            n_aud_flow++;
        if (s->res == 1)
            n_aud_cross++;
    }
    for (si = 0; si < t->nheld; si++)
        if (t->held[si].path == 1 && t->held[si].wc < first_new_wc)
            first_new_wc = t->held[si].wc;
    if (vs >= 0 && vs < t->nst && t->st[vs].res == 1 && (n_aud_cross || !n_aud_flow)) {
        const AVStream *vst = d->ifmt->streams[vs];
        TlStream *v = &t->st[vs];
        fd = vst->avg_frame_rate.num > 0 ? av_rescale(AV_TIME_BASE, vst->avg_frame_rate.den, vst->avg_frame_rate.num)
                                         : FFMAX(v->step_us, 1);
        va = v->old_end_us + fd;               /* the end of the last old picture */
        vb = v->new_first_us;                  /* the first new picture */
        J  = vb - va;          /* exact: an erase makes the labels continuous (the first new picture one picture after the
                                * last old one = on the grid); only the KEPT part is whole pictures. Rounding J to pictures
                                * over-erased a sub-picture relabel (29.97: -120 ms read -133 ms) — a skipped tick, a 0.07 s
                                * video PTS hole per event (sync_bstep rich2997b) */
        /* W over the same span as J: J runs from the video's last old-path picture, so W counts from when that picture
         * arrived too — not only the latest silence. A content scrap between two outages, dropped as an outlier, cut W to
         * the second outage and erased 14 s of real loss as "flowing" (gate sync_storm 430 s: 15 s of pictures without
         * sound once the audio joined) */
        W  = (atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0 || (int64_t)g_preroll_ms >= 4000) ? 0 :
             FFMAX(input_wall_gap(t, first_new_wc),
                   v->last_wc && first_new_wc != INT64_MAX && first_new_wc - v->last_wc >= g_gap_min_us ?
                   first_new_wc - v->last_wc : 0);
        for (si = 0; si < t->nst; si++)
            if (t->st[si].res == 1 && d->ts_outl && si < d->nb_streams_open &&
                d->ts_outl[si].dmg_wc && nw - d->ts_outl[si].dmg_wc < W + 3000000)
                dmg = 1;
        if (J > 0) {
            /* damage hides lost content from the wall evidence (packets keep coming, pictures do not), but no more than
             * the damage phase lasted: one corrupt PES at a splice proved a 38 s "hole" on a relabel (gate sync_burstjump:
             * a 13 s video PTS hole) and 9094 s across a wrap (mvwrap jump_return_50) */
            dspan = dmg && t->dmg_start_wc ? nw - t->dmg_start_wc : 0;
            keep  = FFMIN(J, (FFMAX(W, dspan) + fd / 2) / fd * fd);
            erase = J - keep;
        } else {
            /* a step back while the input flowed: the new content follows on directly. After a wall gap it is a new
             * domain (an origin restart): the output already filled the gap (bars / silence), so the gap is kept and the
             * new content follows it — kept 0 put it 20 s behind the hold fill, which dropped the returning audio and
             * the copied AC-3 until the labels caught up (gate sync_ac3_kill_return_20, mosaic kill_return_20) */
            keep  = W ? (W + fd / 2) / fd * fd : 0;
            erase = J - keep;
        }
        k_new = k_old - erase;
        prog_event = 1;
        cls = J > 0 && keep == J ? "hole kept" : keep && J > 0 ? "hole, flowing part erased" : J > 0 ? "relabel erased" :
              keep ? "new domain, gap kept" : "step back erased";
        /* the new content resumes at a keyframe only where content was lost (a hole, damage): a relabel while the input
         * flowed keeps the bitstream's references — dropping to the next keyframe left ~1 s video PTS holes (sync_bstep) */
        if (erase && (W > 0 || dmg) && !d->drop_until_kf && g_drop_until_kf) {
            int i;
            for (i = 0; i < t->nheld; i++)
                if (t->held[i].si == vs && t->held[i].path == 1) {
                    if (!(t->held[i].pkt->flags & AV_PKT_FLAG_KEY)) {
                        d->drop_until_kf = 1;  /* the new content resumes at a keyframe */
                        d->kf_arm_us     = nw;
                        d->kf_arm_vdrop  = d->vdrop;
                    }
                    break;
                }
        }
    } else {
        for (si = 0; si < t->nst; si++) {
            TlStream *s = &t->st[si];
            int64_t js, step = s->step_us ? s->step_us : 20000;
            if (s->res != 1)
                continue;
            cls = "single-stream step, labels as sent";
            if (d->ifmt->streams[si]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO || (d->ts_outl && si < d->nb_streams_open &&
                d->ts_outl[si].dmg_wc && nw - d->ts_outl[si].dmg_wc < 3000000))
                continue;
            /* design invariant 6, the 0.9.16.4 rule: one audio stream stepping BACK on clean audio is a relabel (its sound
             * runs on) — repaired on that stream, never a map event; the same stream stepping forward again by about the
             * repair is its return leg and undoes it (T-080 B2: -X/+X nets to zero). Any other forward step is that
             * stream's own loss: labels as sent, the audio door pads it. Gate std/sync_arelabel: -300 ms without it */
            js = s->new_first_us - (s->old_end_us + step);
            /* from 150 ms: a few frames back is the PES interpolation's own jitter (the next PES's label lands under the
             * interpolated cursor), no relabel — repaired, GYF's audio moved +95 ms (43-75 ms steps, labels as sent: the
             * overlap is absorbed). The relabel keys step 300 / 600 ms */
            if (js < -150000)
                s->rep_new = -js;
            else if (js > step && s->rep_us > 0)
                s->rep_new = -FFMIN(js, s->rep_us);
            if (llabs(s->rep_us + s->rep_new) > 2000000)
                s->rep_new = 0;               /* bounded: a repair that keeps growing is no relabel (labels as sent) */
            if (s->rep_new > 0 && g_rlcheck) {
                /* T-087: the audio thread checks the content across the seam (relabel or repeat) and the sensor reads it
                 * accordingly, as after the demux absorber's backward erase this replaces */
                int j;
                for (j = 0; j < d->n_audio; j++)
                    if (d->astream[j] == si && d->aglobal[j] >= 0 && d->aglobal[j] < PTV_MAX_AUDIO) {
                        int g = d->aglobal[j];
                        atomic_store_explicit(&g_rl_step_us[g], s->rep_new, memory_order_relaxed);
                        atomic_store_explicit(&g_rl_wc[g], nw, memory_order_relaxed);
                        atomic_fetch_add_explicit(&g_rl_seq[g], 1, memory_order_relaxed);
                        s->rl_mark = 1;
                    }
            }
            if (s->rep_new) {
                s->rep_us += s->rep_new;
                cls = s->rep_new > 0 ? "single-stream relabel, repaired on the stream" :
                                       "single-stream return leg, repair undone";
            }
        }
    }
    t->ev_n++;
    av_log(NULL, prog_event ? AV_LOG_WARNING : AV_LOG_INFO,
           "[PTV-TL] event #%"PRId64" (stream %d opened): %s", t->ev_n, t->trig, cls);
    if (prog_event)
        av_log(NULL, prog_event ? AV_LOG_WARNING : AV_LOG_INFO,
               " — J %+.3f s, wall %.3f s%s, kept %.3f s, erased %.3f s (%"PRId64" pictures), k %+.3f s",
               J / 1e6, W / 1e6, dmg ? " (damaged)" : "", keep / 1e6, erase / 1e6, fd ? erase / fd : 0,
               k_new / 1e6);
    for (si = 0; si < t->nst; si++)
        if (t->st[si].dense && t->st[si].res == 1)
            av_log(NULL, prog_event ? AV_LOG_WARNING : AV_LOG_INFO, " [s%d %+.3f s]", si,
                   (t->st[si].new_first_us - t->st[si].old_end_us) / 1e6);
    av_log(NULL, prog_event ? AV_LOG_WARNING : AV_LOG_INFO, " window %.2f s, %d held\n",
           (av_gettime_relative() - t->open_wc) / 1e6, t->nheld);

    if (prog_event) {
        atomic_store_explicit(&t->prog_wc, nw, memory_order_relaxed);
        if (d->single)
            atomic_store_explicit(&g_tl_prog_wc, nw, memory_order_relaxed);
        k_move(d, t, k_new - k_old);
        if (k_new != k_old) {
            ptv_vwait_rebase(d, vb, va, t->open_wc, av_rescale_q(k_new - k_old, AV_TIME_BASE_Q, (AVRational){ 1, 90000 }));
            t->ev_va = va;
            t->ev_dk = k_new - k_old;
            t->ev_wc = nw;
        }
    }
    ret = release(d, t, k_old, k_new, prog_event, va, vb);
    /* the paths go on where each stream ended */
    for (si = 0; si < t->nst; si++) {
        TlStream *s = &t->st[si];
        if (!s->dense)
            continue;
        if (s->res == 1 && s->rep_new) {
            d->wrap_off[si] += av_rescale_q(s->rep_new, AV_TIME_BASE_Q, d->ifmt->streams[si]->time_base);
            ptv_rsync_post_edit(d, si, s->rep_new);
            s->new_last_us += s->rep_new;
            s->rep_new = 0;
        }
        if (s->res == 1) {
            if (prog_event) {
                s->strag_last_us  = s->old_end_us;
                s->strag_until_wc = nw + TL_STRAG_US;
            }
            s->last_us = s->new_last_us;
            s->last_wc = nw;
        } else if (prog_event && !has_held(t, si) && !flowing(t, s)) {
            s->ok = 0;                        /* not part of it: its next packet re-seeds the path, no window */
        } else if (s->ok)
            s->last_us = s->old_end_us;
        s->res = 0;
    }
    if (prog_event) {                         /* a stream joining late must not bring back what the event erased */
        t->er_lo = erase > 0 ? va : 0;
        t->er_hi = erase > 0 ? vb : 0;
    }
    t->active = 0;
    return ret;
}

int ptv_tl_ingest(DemuxArgs *d, AVPacket *pkt)
{
    PtvTimeline *t = d->tl;
    int si = pkt->stream_index;
    int64_t nw = av_gettime_relative(), s_us;
    int ret;
    TlStream *s;

    if (t->last_any_wc && nw - t->last_any_wc >= g_gap_min_us) {
        /* the input went silent inside a window the video opened: decide it on the evidence before this silence (a lone
         * picture = an outlier, dropped), then this packet starts afresh with J and W from the same old-path picture.
         * Waiting through the silence gave each storm event the silence AFTER its jump as W and merged a content scrap
         * with the next outage — 14 s of real loss erased (sync_storm 430 s). An audio-first crossing (the video not yet
         * crossed) keeps waiting (win_from) */
        if (t->active && t->trig == d->vstream) {
            int r = close_window(d, t);
            if (r < 0) {
                av_packet_free(&pkt);
                return r;
            }
        }
        t->gap_us     = nw - t->last_any_wc;
        t->gap_end_wc = nw;
    }
    t->last_any_wc = nw;
    if (d->ts_outl)
        for (int i = 0; i < t->nst && i < d->nb_streams_open; i++)
            if (d->ts_outl[i].dmg_wc > t->dmg_seen_wc) {
                if (!t->dmg_seen_wc || d->ts_outl[i].dmg_wc - t->dmg_seen_wc > 3000000)
                    t->dmg_start_wc = d->ts_outl[i].dmg_wc;
                t->dmg_seen_wc = d->ts_outl[i].dmg_wc;
            }
    s_us = pkt_s_us(d, pkt, t->k_us);
    if (si >= t->nst)
        return ptv_demux_dispatch(d, pkt);
    s = &t->st[si];
    s->dense = is_dense(d, si);

    if (t->active) {
        /* a window another stream opened: the video going on on its own old path is not part of any decision until it
         * crosses itself — dispatch it instead of holding it. Held, it stopped the picture and with it the audio's
         * silence fill ("while video flows"): heavy loss on an audio track darkened its PID for the gap plus the window
         * (gate loss_100 on the Cinestar media: 1.6-2.4 s) */
        if (si == d->vstream && t->trig != si && s->ok && s_us != AV_NOPTS_VALUE && !has_held(t, si) &&
            llabs(s_us - (s->last_us + s->step_us)) <= tol_us(s)) {
            path_take(s, s_us, nw);
            s->win_passed = 1;
            if (nw - win_from(t) >= TL_WIN_US || window_resolved(d, t, 0)) {
                ret = ptv_demux_dispatch(d, pkt);
                return ret < 0 ? ret : close_window(d, t);
            }
            return ptv_demux_dispatch(d, pkt);
        }
        if (t->nheld < TL_HELD_MAX) {
            TlPkt *h = &t->held[t->nheld++];
            h->pkt = pkt; h->s_us = s_us; h->k_at = t->k_us; h->wc = nw; h->si = si; h->path = 0;
        } else {
            av_packet_free(&pkt);              /* cannot happen in a 1 s window; never block the demux */
            t->drop_n++;
        }
        if (nw - win_from(t) >= TL_WIN_US || t->nheld >= TL_HELD_MAX || window_resolved(d, t, 0))
            return close_window(d, t);
        return 0;
    }
    /* a sparse copy of the OLD content arriving after a program event (source time before its last old picture) takes the
     * old offset back: it was unwrapped with the new one */
    if (!s->dense && s_us != AV_NOPTS_VALUE && t->ev_dk < 0 && nw - t->ev_wc < 10000000 && s_us < t->ev_va)
        relabel(d, pkt, 0, -t->ev_dk);        /* forward events only: after a step back / restart the NEW content's source
                                               * time lies below the old end too (rich sync_kill_return_20: 6 subs + 2
                                               * clears pushed -40 s and lost) */
    if (!s->dense || s_us == AV_NOPTS_VALUE)
        return ptv_demux_dispatch(d, pkt);
    /* sound of a span the last program event erased, arriving after it (the audio was not flowing at the event: a storm,
     * sync_storm 430 s — 14 s of erased content played under the next pictures, 16 flashes without their beep) */
    if (t->er_hi > t->er_lo && si != d->vstream) {
        if (s_us >= t->er_lo && s_us < t->er_hi) {
            t->strag_n++;
            av_packet_free(&pkt);
            return 0;
        }
        if (s_us >= t->er_hi)
            t->er_lo = t->er_hi = 0;          /* past it (a later restart may reuse these source times) */
    }
    if (!s->ok) {
        path_take(s, s_us, nw);
        return ptv_demux_dispatch(d, pkt);
    }
    /* a straggler of the old path after an event: content of the erased span, never shown */
    if (s->strag_until_wc && nw < s->strag_until_wc &&
        llabs(s_us - (s->strag_last_us + s->step_us)) <= tol_us(s) &&
        llabs(s_us - (s->last_us + s->step_us)) > tol_us(s)) {
        s->strag_last_us = s_us;
        t->strag_n++;
        av_packet_free(&pkt);
        return 0;
    }
    /* below the window threshold, an audio stream stepping forward on its own by about an open repair: the repair's return
     * leg (T-080 B2). Backward steps open a window (off_path), whose close repairs a single audio stream's relabel and
     * erases a both-stream step; any other forward step is that stream's loss, labels as sent (the audio door pads) */
    if (g_aglue_ms > 0 && d->ifmt->streams[si]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
        !(d->ts_outl && si < d->nb_streams_open && d->ts_outl[si].dmg_wc && nw - d->ts_outl[si].dmg_wc < 3000000)) {
        int64_t js = s_us - (s->last_us + s->step_us), rp = 0;
        if (js > (int64_t)g_aglue_ms * 1000 && js <= g_tl_open_us && s->rep_us > 0)
            rp = -FFMIN(js, s->rep_us);
        if (rp) {
            s->rep_us += rp;
            d->wrap_off[si] += av_rescale_q(rp, AV_TIME_BASE_Q, d->ifmt->streams[si]->time_base);
            ptv_rsync_post_edit(d, si, rp);
            relabel(d, pkt, 0, rp);
            s_us += rp;
            t->ev_n++;
            av_log(NULL, AV_LOG_INFO, "[PTV-TL] event #%"PRId64" (stream %d): single-stream %s %+.3f s on the stream "
                   "(repair now %+.3f s)\n", t->ev_n, si, rp > 0 ? "relabel, repaired" :
                   "return leg, repair undone", rp / 1e6, s->rep_us / 1e6);
        }
    }
    /* a forward jump the stream's own wall silence explains (J <= W) is a kept hole whatever its partners do: no offset
     * moves, nothing is relabelled or erased — a window would only hold the input (heavy loss: every lost audio run held
     * all streams up to 1 s, vt-cap-awe loss_100 audio dark 2.1 s, pre8 0.84 s). Windows stay for what can move the
     * offset: a jump beyond its silence (relabel / flowing part), a step back; and under an armed bank (W no evidence) */
    if (off_path(s, s_us) && s_us > s->last_us && s->last_wc &&
        !(atomic_load_explicit(&g_bank_us, memory_order_relaxed) > 0 || (int64_t)g_preroll_ms >= 4000) &&
        s_us - (s->last_us + s->step_us) <= nw - s->last_wc + tol_us(s)) {
        t->ev_n++;
        av_log(NULL, AV_LOG_INFO, "[PTV-TL] event #%"PRId64" (stream %d): hole kept — J %+.3f s within its own silence of "
               "%.3f s, labels as sent, no window\n", t->ev_n, si,
               (s_us - (s->last_us + s->step_us)) / 1e6, (nw - s->last_wc) / 1e6);
        path_take(s, s_us, nw);
        return ptv_demux_dispatch(d, pkt);
    }
    if (off_path(s, s_us)) {
        TlPkt *h = &t->held[0];
        t->active    = 1;
        t->trig      = si;
        t->open_wc   = nw;
        t->open_s_us = s_us;
        t->nheld     = 1;
        h->pkt = pkt; h->s_us = s_us; h->k_at = t->k_us; h->wc = nw; h->si = si; h->path = 0;
        for (int i = 0; i < t->nst; i++) {     /* a stream the window never classifies keeps its path */
            t->st[i].old_end_us = t->st[i].last_us;
            t->st[i].win_passed = 0;
        }
        if (d->disturb_epoch)
            atomic_fetch_add_explicit(d->disturb_epoch, 1, memory_order_relaxed);
        return 0;
    }
    path_take(s, s_us, nw);
    return ptv_demux_dispatch(d, pkt);
}

/* our own cut (the depth door): the span [Va, Vb) was dropped on every stream on the pictures' boundaries — erase it
 * from the program offset; every path re-seeds on its next packet (the door made them continuous) */
void ptv_tl_cut(DemuxArgs *d, int64_t cut_us)
{
    PtvTimeline *t = d->tl;
    int si;
    if (!t || cut_us <= 0)
        return;
    k_move(d, t, -cut_us);
    for (si = 0; si < t->nst; si++)
        t->st[si].ok = 0;
}

/* the copy path: dense copies take the video's placement while the map settles — within 30 s of a program event, or
 * while the source holds — and the old stamping otherwise (house_skew is where the video puts them in steady flow; their
 * wait for the shown picture read as a gap to the copy watch: AC-3 dark 1.8 s under heavy loss, gate vt-cin loss_100) */
int ptv_tl_settling(DemuxArgs *d)
{
    int64_t pw;
    if (!d->tl)
        return 0;
    pw = atomic_load_explicit(&d->tl->prog_wc, memory_order_relaxed);
    return ptv_src_holding() || (pw && av_gettime_relative() - pw < 30000000);
}
