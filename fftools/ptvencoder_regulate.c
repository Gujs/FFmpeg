/*
 * ptvencoder — the input regulator (2.2 T-423)
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

/* 2.2 T-423 (owner design rule 2026-10-24, GOALS 6: transport problems are fixed in FRONT of the encoder). A udp:// input
 * given -regulate D is read here instead of by the demuxer: its TS packets go into one FIFO as they arrive and leave, to a
 * loopback port the input then reads, at their PCR time + D — burst-and-gap delivery (HLS repackaged over SRT/UDP) reaches
 * the encoder as a real-time stream. The open / reopen / reshape logic is unchanged: it reads the loopback URL.
 * Measured with the prototype (test-results/v210-pre7/t423/pcrpace.py): Fintech's 6 s bursts every 6 s, house_skew +6.8 s
 * and 411 repeats direct, 0 ms / 0 through D = 8 s.
 * - the first PCR anchors arrival + D; every packet leaves at anchor + (its PCR - the anchor's), packets without a PCR with
 *   the last PCR'd one;
 * - a PCR jump (backwards or > 1 s: a source restart, a wrap, a loop) continues the schedule from the last scheduled time
 *   (the prototype took a fresh D at a file loop and stalled the encoder 8.5 s; the later of the two moved a bursty source
 *   later at every jump — Fintech's +2 s PCR jumps arrive early in their burst: 12 s held, 305 repeats); only when the
 *   FIFO ran dry (a real outage: the last scheduled time has passed) does it take arrival + D again, as a gateway
 *   re-buffers to its latency; a forward jump that comes with an arrival stop of the same length is a hole (content
 *   lost in real time, T-437) and keeps the PCR schedule;
 * - more than D + 10 s held (a source faster than real time for long): the oldest go, logged;
 * - 2.2 T-439 (regulated by default, D 0.5 s): the schedule follows the source's delivery. Per 10 s, the packet that came
 *   earliest against its PCR should wait exactly D; the schedule runs -(its wait - D) / 30 s faster or slower (at most
 *   2 %) — a de-jitter buffer, not a de-wander one. Measured on 30 min of 8 production sources (test-results/t439): their
 *   arrival against the PCR wanders up to 1 s over minutes (glo-3 origins; TOLLY after a PCR glitch delivered 3.7 s at
 *   1.0067x real time); anchored once, the hold took all of it as delay (0.5 -> 4.9 s), and a source drifting the other
 *   way would empty it for good. Packets later than D pass as they come (as unregulated), reported per minute;
 * - the FIFO grows as it fills (a 0.5 s hold of a 5 Mb/s source is ~250 datagrams), up to the D + 10 s cap. */

#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "libavformat/avio.h"
#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/common.h"
#include "libavutil/mathematics.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/time.h"

#include "ptvencoder.h"

#define REG_DGRAM 1316                              /* 7 TS packets, the source's own datagram size */
#define REG_SV_WIN   10000000                       /* T-439: the servo's window (us) */
#define REG_SV_TAU   30                             /* T-439: its time constant (s) */
#define REG_SV_MAX   20000                          /* T-439: its largest rate change (ppm) */
#define REG_CAP0     2048                           /* T-439: the FIFO's first size (datagrams) */

typedef struct RegItem {
    int64_t due;
    int     len;
    uint8_t data[REG_DGRAM];
} RegItem;

typedef struct Regulator {
    int             k;
    char           *src_url;
    int             port;
    int64_t         delay;
    pthread_t       rd, wr;
    int             rd_ok, wr_ok;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    RegItem        *ring;
    int             cap, cap_max, head, n;
    int64_t         dropped, drop_log_wc, jump_log_wc;
    /* the schedule (reader thread only) */
    int             pcr_pid;
    int64_t         anchor_t, acc, last_pcr, last_due;
    int64_t         last_pcr_wc;                    /* T-437: the previous PCR's arrival (wall) */
    int             pid_moved;                      /* T-439: the PCR just moved PID — its first step is no hole */
    int64_t         first_wc;                       /* the first packet before any PCR (wall), 0 = said / done */
    /* 2.2 T-436: [PTV-BURSTY] at the socket (reader thread only) — the demuxer reads the paced loopback and sees none */
    int64_t         by_last, by_win, by_max, held_min, held_max;
    int             by_cnt;
    /* 2.2 T-439: the drift servo + the late-packet report (reader thread only) */
    int64_t         sv_win, sv_max, sv_rho, sv_frac;
    int64_t         lt_win, lt_worst;
    int             lt_cnt, lt_n;
} Regulator;

static Regulator  *regs[PTV_MAX_INPUT];
static _Atomic int reg_stop;

static int reg_interrupt(void *opaque)
{
    (void)opaque;
    return atomic_load_explicit(&reg_stop, memory_order_relaxed);
}

/* 2.2 T-436: the arrival-stall report of an unregulated input ([PTV-BURSTY], ptvencoder_demux.c), measured where the
 * source arrives — a regulated input's demuxer reads the paced loopback, so its own report goes quiet exactly on the
 * channels that burst. Per 60 s window with a >= 1.5 s stall: the count, the worst, and how much the FIFO held (its
 * lowest near 0 = D barely covers the gaps). A reopen (outage) restarts the measure, as the demuxer's excludes outages. */
static void reg_bursty(Regulator *r, int64_t now)
{
    int64_t held;
    if (r->last_due == AV_NOPTS_VALUE)              /* no schedule yet (before the first PCR) */
        return;
    held = r->last_due - now;
    if (r->by_last && now - r->by_last >= 1500000) {
        r->by_cnt++;
        r->by_max = FFMAX(r->by_max, now - r->by_last);
    }
    r->by_last = now;
    if (!r->by_win) {
        r->by_win   = now;
        r->held_min = r->held_max = held;
    }
    r->held_min = FFMIN(r->held_min, held);
    r->held_max = FFMAX(r->held_max, held);
    if (now - r->by_win >= 60000000) {
        if (r->by_cnt > 0)
            av_log(NULL, AV_LOG_INFO, "[PTV-BURSTY] in%d %d stall%s >=1.5s in the last 60s (worst %.1fs) at the input — "
                   "regulated: %.1f-%.1f s held (D %.1f s)\n", r->k, r->by_cnt, r->by_cnt == 1 ? "" : "s",
                   r->by_max / 1e6, r->held_min / 1e6, r->held_max / 1e6, r->delay / 1e6);
        r->by_win = now;
        r->by_cnt = 0;
        r->by_max = 0;
        r->held_min = r->held_max = held;
    }
}

/* the packet's 33-bit PCR base when it carries one on the regulated PID, else -1. 2.2 T-439: a PID that carries a PCR
 * after the regulated one has carried none for 1 s takes over (the provider moved the PCR with its video:
 * sync_vpidmove_30, 0x100 -> 0x1e1 — the schedule stopped and the input ran unregulated from the move on) */
static int64_t ts_pcr(Regulator *r, const uint8_t *p, int64_t now)
{
    int pid = ((p[1] & 0x1f) << 8) | p[2];
    if (!(p[3] & 0x20) || p[4] < 7 || !(p[5] & 0x10))
        return -1;
    if (r->pcr_pid < 0)
        r->pcr_pid = pid;
    if (pid != r->pcr_pid && r->last_pcr_wc && now - r->last_pcr_wc >= 1000000) {
        av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d PCR moved: pid 0x%x -> 0x%x (none on 0x%x for %.1f s)\n",
               r->k, r->pcr_pid, pid, r->pcr_pid, (now - r->last_pcr_wc) / 1e6);
        r->pcr_pid   = pid;
        r->pid_moved = 1;
    }
    if (pid != r->pcr_pid)
        return -1;
    return ((int64_t)p[6] << 25) | (p[7] << 17) | (p[8] << 9) | (p[9] << 1) | (p[10] >> 7);
}

/* 2.2 T-439: per PCR packet, its wait (due - arrival). The earliest packet of each 10 s should wait D: the servo steers
 * the schedule's rate toward that. A packet that waits less than -100 ms arrived later than D and passes late (its
 * jitter is more than D covers): counted, said per 60 s */
static void reg_servo(Regulator *r, int64_t wait, int64_t now)
{
    if (!r->sv_win) {
        r->sv_win = now;
        r->sv_max = wait;
    }
    r->sv_max = FFMAX(r->sv_max, wait);
    if (now - r->sv_win >= REG_SV_WIN) {
        r->sv_rho = av_clip64(-(r->sv_max - r->delay) / REG_SV_TAU, -REG_SV_MAX, REG_SV_MAX);
        if (getenv("PTV_TEST_REG_NOSERVO"))         /* TEST-ONLY (T-439): the pre-T-439 fixed schedule, for A/B */
            r->sv_rho = 0;
        if (g_diag)
            av_log(NULL, AV_LOG_INFO, "[PTV-REGDIAG] in%d earliest waited %.3f s (D %.3f) rho %+"PRId64"ppm held %d\n",
                   r->k, r->sv_max / 1e6, r->delay / 1e6, r->sv_rho, r->n);
        r->sv_win = now;
        r->sv_max = wait;
    }
    if (!r->lt_win)
        r->lt_win = now;
    r->lt_n++;
    if (wait < -100000) {
        r->lt_cnt++;
        r->lt_worst = FFMAX(r->lt_worst, -wait);
    }
    if (now - r->lt_win >= 60000000) {
        if (r->lt_cnt)
            av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d %d of %d PCR packets later than D in the last 60s (worst %.1f s "
                   "past due) — passed as they came (D %.1f s)\n", r->k, r->lt_cnt, r->lt_n, r->lt_worst / 1e6,
                   r->delay / 1e6);
        r->lt_win   = now;
        r->lt_cnt   = r->lt_n = 0;
        r->lt_worst = 0;
    }
}

static void reg_schedule(Regulator *r, int64_t pcr, int64_t now)
{
    if (r->anchor_t == AV_NOPTS_VALUE) {
        r->anchor_t = now + r->delay;
        r->acc      = 0;
    } else {
        int64_t d = pcr - r->last_pcr, gap = now - r->last_pcr_wc, dus;
        if (d < -(1LL << 32))
            d += 1LL << 33;                         /* the 33-bit wrap */
        dus = av_rescale(d, 1000000, 90000);
        if (d > 90000 && FFABS(gap - dus) <= FFMAX(500000, dus / 4)) {
            /* 2.2 T-437: the arrivals stopped as long as the PCR jumped — content lost in real time (an origin that skips
             * a segment: glo-3 SOFAST, a 5.1-5.6 s stop + the same PCR jump every ~30 s, 7 of 7 within 0.02 s). Kept on
             * its schedule: the hole plays at its moment and the hold stays D. Continuing the schedule here ate the hold at
             * every hole (a Stingray replay: held 8 -> 163 s, FIFO drops, 13 freezes vs 3 unregulated) */
            r->acc += d;
            if (!r->pid_moved && now - r->jump_log_wc >= 10000000) {
                r->jump_log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d hole %.1f s (the source stopped as long as its PCR jumped) — "
                       "kept on its schedule (%.1f s held)\n", r->k, dus / 1e6, (r->anchor_t + av_rescale(r->acc, 1000000,
                       90000) - now) / 1e6);
            }
        } else if (d < 0 || d > 90000) {            /* a jump: continue the schedule (see the top) */
            r->anchor_t = r->last_due >= now ? r->last_due : now + r->delay;
            r->acc      = 0;
            if (now - r->jump_log_wc >= 10000000) {
                r->jump_log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d PCR jump %+.1f s — schedule continued (%.1f s held)\n",
                       r->k, d / 90000.0, (r->anchor_t - now) / 1e6);
            }
        } else {
            r->acc     += d;
            r->sv_frac += dus * r->sv_rho;          /* T-439: the servo's rate, applied as the PCR advances */
            r->anchor_t += r->sv_frac / 1000000;
            r->sv_frac  %= 1000000;
        }
    }
    r->last_pcr    = pcr;
    r->last_pcr_wc = now;
    r->pid_moved   = 0;
    r->last_due    = r->anchor_t + av_rescale(r->acc, 1000000, 90000);
    reg_servo(r, r->last_due - now, now);
}

static void reg_push(Regulator *r, const uint8_t *data, int len, int64_t due)
{
    RegItem *it;
    pthread_mutex_lock(&r->lock);
    if (r->n == r->cap && r->cap < r->cap_max) {    /* T-439: grow, in order from 0 */
        int nc = FFMIN(r->cap * 2, r->cap_max), i;
        RegItem *nr = av_malloc_array(nc, sizeof *nr);
        if (nr) {
            for (i = 0; i < r->n; i++)
                nr[i] = r->ring[(r->head + i) % r->cap];
            av_free(r->ring);
            r->ring = nr;
            r->cap  = nc;
            r->head = 0;
        }
    }
    if (r->n == r->cap) {                           /* more than D + 10 s held: the oldest go */
        r->head = (r->head + 1) % r->cap;
        r->n--;
        r->dropped++;
        if (av_gettime_relative() - r->drop_log_wc >= 10000000) {
            r->drop_log_wc = av_gettime_relative();
            av_log(NULL, AV_LOG_WARNING, "[PTV-REG] in%d more than %.0f s held — the oldest dropped (%"PRId64" so far)\n",
                   r->k, r->delay / 1e6 + 10, r->dropped);
        }
    }
    it = &r->ring[(r->head + r->n) % r->cap];
    it->due = due;
    it->len = len;
    memcpy(it->data, data, len);
    r->n++;
    pthread_cond_signal(&r->cond);
    pthread_mutex_unlock(&r->lock);
}

static void *reg_reader(void *arg)
{
    Regulator *r = arg;
    AVIOInterruptCB icb = { reg_interrupt, NULL };
    uint8_t buf[65536], dg[REG_DGRAM];
    int have = 0, dl = 0;
    int64_t due = AV_NOPTS_VALUE;
    while (!atomic_load_explicit(&reg_stop, memory_order_relaxed)) {
        AVIOContext *pb = NULL;
        int ret = avio_open2(&pb, r->src_url, AVIO_FLAG_READ, &icb, NULL);
        if (ret < 0) {
            av_usleep(1000000);
            continue;
        }
        while (!atomic_load_explicit(&reg_stop, memory_order_relaxed)) {
            int n = avio_read_partial(pb, buf + have, sizeof(buf) - have), i = 0;
            int64_t now = av_gettime_relative();
            if (n <= 0)
                break;                              /* a read error / the source's timeout: reopen */
            reg_bursty(r, now);
            have += n;
            while (have - i >= 188) {
                const uint8_t *p = buf + i;
                int64_t pcr;
                if (p[0] != 0x47) {                 /* resync on the next sync byte */
                    i++;
                    continue;
                }
                if ((pcr = ts_pcr(r, p, now)) >= 0) {
                    if (dl) {                       /* a PCR'd packet starts a new datagram at its own time */
                        reg_push(r, dg, dl, due);
                        dl = 0;
                    }
                    reg_schedule(r, pcr, now);
                    due = r->last_due;
                }
                if (due == AV_NOPTS_VALUE) {
                    due = now + r->delay;           /* before the first PCR: held by D */
                    r->first_wc = now;
                } else if (r->pcr_pid < 0 && r->first_wc && now - r->first_wc > 5000000) {
                    r->first_wc = 0;                /* no PCR to schedule by: passed through after D, said once */
                    av_log(NULL, AV_LOG_WARNING, "[PTV-REG] in%d no PCR in 5 s — passing the input through (delayed "
                           "%.1f s, not regulated)\n", r->k, r->delay / 1e6);
                }
                memcpy(dg + dl, p, 188);
                dl += 188;
                if (dl == REG_DGRAM) {
                    reg_push(r, dg, dl, due);
                    dl = 0;
                }
                i += 188;
            }
            memmove(buf, buf + i, have - i);
            have -= i;
        }
        avio_closep(&pb);
        have = 0;
        r->by_last = 0;                             /* T-436: an outage is not a stall */
    }
    return NULL;
}

static void *reg_writer(void *arg)
{
    Regulator *r = arg;
    struct sockaddr_in to = { 0 };
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
        return NULL;
    to.sin_family      = AF_INET;
    to.sin_port        = htons(r->port);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    while (!atomic_load_explicit(&reg_stop, memory_order_relaxed)) {
        RegItem it;
        int64_t wait;
        pthread_mutex_lock(&r->lock);
        if (!r->n) {
            struct timespec ts;
            int64_t t = av_gettime() + 50000;
            ts.tv_sec  = t / 1000000;
            ts.tv_nsec = (t % 1000000) * 1000;
            pthread_cond_timedwait(&r->cond, &r->lock, &ts);
            pthread_mutex_unlock(&r->lock);
            continue;
        }
        wait = r->ring[r->head].due - av_gettime_relative();
        if (wait > 0) {
            pthread_mutex_unlock(&r->lock);
            av_usleep(FFMIN(wait, 5000));
            continue;
        }
        it = r->ring[r->head];
        r->head = (r->head + 1) % r->cap;
        r->n--;
        pthread_mutex_unlock(&r->lock);
        sendto(fd, it.data, it.len, 0, (struct sockaddr *)&to, sizeof to);
    }
    close(fd);
    return NULL;
}

/* a free loopback UDP port (bound and released: the input binds it next) */
static int reg_port(void)
{
    struct sockaddr_in a = { 0 };
    socklen_t al = sizeof a;
    int fd = socket(AF_INET, SOCK_DGRAM, 0), port = -1;
    if (fd < 0)
        return -1;
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (!bind(fd, (struct sockaddr *)&a, sizeof a) && !getsockname(fd, (struct sockaddr *)&a, &al))
        port = ntohs(a.sin_port);
    close(fd);
    return port;
}

char *ptv_regulate_start(const char *url, int64_t delay_us, int k)
{
    Regulator *r;
    const char *q;
    char *out, *opts, *tok, *save = NULL;
    AVBPrint bp;
    if (!url || strncmp(url, "udp://", 6) || k < 0 || k >= PTV_MAX_INPUT || delay_us <= 0)
        return NULL;
    if (!(r = av_mallocz(sizeof *r)))
        return NULL;
    r->k        = k;
    r->delay    = delay_us;
    r->pcr_pid  = -1;
    r->anchor_t = AV_NOPTS_VALUE;
    r->last_due = AV_NOPTS_VALUE;                   /* T-436: no schedule yet */
    r->cap_max  = (int)((delay_us / 1000000 + 10) * 2000);   /* D + 10 s at up to ~20 Mb/s */
    r->cap      = FFMIN(REG_CAP0, r->cap_max);      /* T-439: grows as it fills */
    r->port     = reg_port();
    r->src_url  = av_strdup(url);
    r->ring     = av_malloc_array(r->cap, sizeof *r->ring);
    if (r->port < 0 || !r->src_url || !r->ring)
        goto fail;
    /* the input reads the loopback port with the source's own options, less the ones that name the source's network */
    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);
    av_bprintf(&bp, "udp://127.0.0.1:%d", r->port);
    if ((q = strchr(url, '?')) && (opts = av_strdup(q + 1))) {
        char sep = '?';
        for (tok = av_strtok(opts, "&", &save); tok; tok = av_strtok(NULL, "&", &save)) {
            if (av_strstart(tok, "localaddr=", NULL) || av_strstart(tok, "reuse=", NULL) ||
                av_strstart(tok, "sources=", NULL) || av_strstart(tok, "block=", NULL))
                continue;
            av_bprintf(&bp, "%c%s", sep, tok);
            sep = '&';
        }
        av_free(opts);
    }
    if (av_bprint_finalize(&bp, &out) < 0)
        goto fail;
    pthread_mutex_init(&r->lock, NULL);
    pthread_cond_init(&r->cond, NULL);
    r->rd_ok = !pthread_create(&r->rd, NULL, reg_reader, r);
    r->wr_ok = !pthread_create(&r->wr, NULL, reg_writer, r);
    regs[k] = r;
    av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d regulated: %s → %s, released at PCR time + %.1f s\n",
           k, url, out, delay_us / 1e6);
    return out;
fail:
    av_free(r->src_url);
    av_free(r->ring);
    av_free(r);
    return NULL;
}

void ptv_regulate_stop(void)
{
    int k;
    atomic_store_explicit(&reg_stop, 1, memory_order_relaxed);
    for (k = 0; k < PTV_MAX_INPUT; k++) {
        Regulator *r = regs[k];
        if (!r)
            continue;
        pthread_mutex_lock(&r->lock);
        pthread_cond_broadcast(&r->cond);
        pthread_mutex_unlock(&r->lock);
        if (r->rd_ok) pthread_join(r->rd, NULL);
        if (r->wr_ok) pthread_join(r->wr, NULL);
        pthread_mutex_destroy(&r->lock);
        pthread_cond_destroy(&r->cond);
        av_free(r->src_url);
        av_free(r->ring);
        av_freep(&regs[k]);
    }
}
