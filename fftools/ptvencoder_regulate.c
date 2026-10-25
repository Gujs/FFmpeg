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
 *   re-buffers to its latency;
 * - more than D + 10 s held (a source faster than real time for long): the oldest go, logged. */

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
    int             cap, head, n;
    int64_t         dropped, drop_log_wc, jump_log_wc;
    /* the schedule (reader thread only) */
    int             pcr_pid;
    int64_t         anchor_t, acc, last_pcr, last_due;
} Regulator;

static Regulator  *regs[PTV_MAX_INPUT];
static _Atomic int reg_stop;

static int reg_interrupt(void *opaque)
{
    (void)opaque;
    return atomic_load_explicit(&reg_stop, memory_order_relaxed);
}

/* the packet's 33-bit PCR base when it carries one on the regulated PID, else -1 */
static int64_t ts_pcr(Regulator *r, const uint8_t *p)
{
    int pid = ((p[1] & 0x1f) << 8) | p[2];
    if (!(p[3] & 0x20) || p[4] < 7 || !(p[5] & 0x10))
        return -1;
    if (r->pcr_pid < 0)
        r->pcr_pid = pid;
    if (pid != r->pcr_pid)
        return -1;
    return ((int64_t)p[6] << 25) | (p[7] << 17) | (p[8] << 9) | (p[9] << 1) | (p[10] >> 7);
}

static void reg_schedule(Regulator *r, int64_t pcr, int64_t now)
{
    if (r->anchor_t == AV_NOPTS_VALUE) {
        r->anchor_t = now + r->delay;
        r->acc      = 0;
    } else {
        int64_t d = pcr - r->last_pcr;
        if (d < -(1LL << 32))
            d += 1LL << 33;                         /* the 33-bit wrap */
        if (d < 0 || d > 90000) {                   /* a jump: continue the schedule (see the top) */
            r->anchor_t = r->last_due >= now ? r->last_due : now + r->delay;
            r->acc      = 0;
            if (now - r->jump_log_wc >= 10000000) {
                r->jump_log_wc = now;
                av_log(NULL, AV_LOG_INFO, "[PTV-REG] in%d PCR jump %+.1f s — schedule continued (%.1f s held)\n",
                       r->k, d / 90000.0, (r->anchor_t - now) / 1e6);
            }
        } else
            r->acc += d;
    }
    r->last_pcr = pcr;
    r->last_due = r->anchor_t + av_rescale(r->acc, 1000000, 90000);
}

static void reg_push(Regulator *r, const uint8_t *data, int len, int64_t due)
{
    RegItem *it;
    pthread_mutex_lock(&r->lock);
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
            have += n;
            while (have - i >= 188) {
                const uint8_t *p = buf + i;
                int64_t pcr;
                if (p[0] != 0x47) {                 /* resync on the next sync byte */
                    i++;
                    continue;
                }
                if ((pcr = ts_pcr(r, p)) >= 0) {
                    if (dl) {                       /* a PCR'd packet starts a new datagram at its own time */
                        reg_push(r, dg, dl, due);
                        dl = 0;
                    }
                    reg_schedule(r, pcr, now);
                    due = r->last_due;
                }
                if (due == AV_NOPTS_VALUE)
                    due = now + r->delay;           /* before the first PCR: held by D */
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
    r->cap      = (int)((delay_us / 1000000 + 10) * 2000);   /* D + 10 s at up to ~20 Mb/s */
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
