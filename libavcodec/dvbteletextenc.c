/*
 * DVB Teletext subtitle encoder
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
 * DVB Teletext subtitle encoder.
 *
 * Encodes text subtitles as DVB Teletext PES payload per ETSI EN 300 472
 * and ETS 300 706. The MPEG-TS muxer handles PES wrapping and the teletext
 * descriptor (0x56) in the PMT using codecpar->extradata.
 *
 * The encoder accepts ASS-formatted subtitle input (as produced by cc_dec
 * or other subtitle decoders), strips formatting, and outputs raw teletext
 * data units suitable for direct insertion into a DVB teletext PES packet.
 */

#include <string.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "ass_split.h"
#include "ass.h"
#include "libavutil/avstring.h"
#include "libavutil/bprint.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

/* Teletext constants per ETS 300 706 / ETSI EN 300 472 */
#define TELETEXT_DATA_IDENTIFIER    0x10  /* EBU data, EN 300 472 Table 1 */
#define TELETEXT_DATA_UNIT_SUBTITLE 0x03  /* EBU teletext subtitle data */
#define TELETEXT_DATA_UNIT_NONSUBT  0x02  /* EBU teletext non-subtitle data */
#define TELETEXT_DATA_UNIT_LENGTH   0x2C  /* 44 bytes per data unit */
#define TELETEXT_FRAMING_CODE       0xE4  /* clock run-in + framing code */
#define TELETEXT_CHARS_PER_ROW      40    /* characters per teletext row */
#define TELETEXT_SUBTITLE_ROW1      23    /* first subtitle row */
#define TELETEXT_SUBTITLE_ROW2      24    /* second subtitle row (bottom) */

/*
 * Hamming 8/4 encoding table.
 * Maps a 4-bit nibble (0-15) to an 8-bit Hamming-encoded byte.
 * Per ETS 300 706 section 8.2.
 */
static const uint8_t hamming84_encode[16] = {
    0x15, 0x02, 0x49, 0x5E, 0x64, 0x73, 0x38, 0x2F,
    0xD0, 0xC7, 0x8C, 0x9B, 0xA1, 0xB6, 0xFD, 0xEA
};

/*
 * Odd parity table for 7-bit ASCII values (0-127).
 * Sets bit 7 so the total number of 1-bits is odd.
 * Per ETS 300 706 section 9.4.2.
 */
static const uint8_t odd_parity[128] = {
    0x80, 0x01, 0x02, 0x83, 0x04, 0x85, 0x86, 0x07,
    0x08, 0x89, 0x8A, 0x0B, 0x8C, 0x0D, 0x0E, 0x8F,
    0x10, 0x91, 0x92, 0x13, 0x94, 0x15, 0x16, 0x97,
    0x98, 0x19, 0x1A, 0x9B, 0x1C, 0x9D, 0x9E, 0x1F,
    0x20, 0xA1, 0xA2, 0x23, 0xA4, 0x25, 0x26, 0xA7,
    0xA8, 0x29, 0x2A, 0xAB, 0x2C, 0xAD, 0xAE, 0x2F,
    0xB0, 0x31, 0x32, 0xB3, 0x34, 0xB5, 0xB6, 0x37,
    0x38, 0xB9, 0xBA, 0x3B, 0xBC, 0x3D, 0x3E, 0xBF,
    0x40, 0xC1, 0xC2, 0x43, 0xC4, 0x45, 0x46, 0xC7,
    0xC8, 0x49, 0x4A, 0xCB, 0x4C, 0xCD, 0xCE, 0x4F,
    0xD0, 0x51, 0x52, 0xD3, 0x54, 0xD5, 0xD6, 0x57,
    0x58, 0xD9, 0xDA, 0x5B, 0xDC, 0x5D, 0x5E, 0xDF,
    0xE0, 0x61, 0x62, 0xE3, 0x64, 0xE5, 0xE6, 0x67,
    0x68, 0xE9, 0xEA, 0x6B, 0xEC, 0x6D, 0x6E, 0xEF,
    0x70, 0xF1, 0xF2, 0x73, 0xF4, 0x75, 0x76, 0xF7,
    0xF8, 0x79, 0x7A, 0xFB, 0x7C, 0xFD, 0xFE, 0x7F
};

typedef struct DVBTeletextEncContext {
    AVClass *class;
    ASSSplitContext *ass_ctx;
    int magazine;           /* magazine number 1-8, default 8 */
    int page;               /* page number in hex (0x00-0xFF), default 0x88 */
    int page_counter;       /* erase page sequence counter (C4 flag) */
} DVBTeletextEncContext;

/**
 * Encode MRAG (Magazine and Row Address Group) as two Hamming 8/4 bytes.
 *
 * @param magazine Magazine number 1-8 (8 is encoded as 0)
 * @param row      Row number 0-31
 * @param dst      Output buffer (at least 2 bytes)
 */
static void encode_mrag(int magazine, int row, uint8_t *dst)
{
    int m = magazine & 0x07; /* magazine 8 → 0 */
    /* byte 1: magazine (3 bits, LSB first) in lower nibble,
     *         row bits 0-0 in upper nibble
     * Per ETS 300 706 section 9.3.1:
     *   Address bits: M1, M2, M3 (magazine), R0, R1, R2, R3 (row)
     *   Transmitted as two Hamming 8/4 coded bytes:
     *   Byte 1 = Hamming(M1 M2 M3 R0) where M1 is bit 0
     *   Byte 2 = Hamming(R1 R2 R3 R4) */
    dst[0] = hamming84_encode[(m & 0x07) | ((row & 0x01) << 3)];
    dst[1] = hamming84_encode[(row >> 1) & 0x0F];
}

/**
 * Write a data unit containing one teletext row.
 *
 * @param buf       Output buffer
 * @param magazine  Magazine number (1-8)
 * @param row       Row number (0-31)
 * @param text      40-byte text content (space-padded, 7-bit ASCII)
 * @param subtitle  1 for subtitle data unit (0x03), 0 for non-subtitle (0x02)
 * @return number of bytes written (always 46: 1+1+1+1+2+40)
 */
static int write_data_unit(uint8_t *buf, int magazine, int row,
                           const uint8_t *text, int subtitle)
{
    int i;

    buf[0] = subtitle ? TELETEXT_DATA_UNIT_SUBTITLE
                      : TELETEXT_DATA_UNIT_NONSUBT;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH; /* 44 bytes */
    /* field_parity (1 bit) | line_offset (4 bits) | reserved (3 bits)
     * field_parity=0 (first field), line_offset varies with row */
    buf[2] = 0x02 | ((row < 16 ? 7 : 20) << 3);
    buf[3] = TELETEXT_FRAMING_CODE;

    /* MRAG */
    encode_mrag(magazine, row, &buf[4]);

    /* text content with odd parity */
    for (i = 0; i < TELETEXT_CHARS_PER_ROW; i++) {
        uint8_t c = text[i];
        if (c >= 0x80)
            c = ' ';  /* replace non-ASCII with space */
        buf[6 + i] = odd_parity[c & 0x7F];
    }

    return 46; /* 1 + 1 + 1 + 1 + 2 + 40 = 46 */
}

/**
 * Write a page header data unit (row 0).
 *
 * The page header contains the page address and control bits encoded
 * with Hamming 8/4, followed by 32 bytes of display text (typically spaces).
 *
 * @param buf       Output buffer
 * @param ctx       Encoder context
 * @param erase     1 to set the erase page flag (C4)
 * @return number of bytes written
 */
static int write_page_header(uint8_t *buf, DVBTeletextEncContext *ctx, int erase)
{
    uint8_t header_text[TELETEXT_CHARS_PER_ROW];
    int page_units, page_tens;
    int i;

    buf[0] = TELETEXT_DATA_UNIT_NONSUBT;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH;
    buf[2] = 0x02 | (7 << 3); /* field_parity=0, line_offset=7 */
    buf[3] = TELETEXT_FRAMING_CODE;

    /* MRAG for row 0 */
    encode_mrag(ctx->magazine, 0, &buf[4]);

    /* Page address: page number as two Hamming 8/4 bytes (BCD) */
    page_units = ctx->page & 0x0F;
    page_tens  = (ctx->page >> 4) & 0x0F;
    buf[6] = hamming84_encode[page_units];
    buf[7] = hamming84_encode[page_tens];

    /* Sub-code bytes (S1, S2, S3, S4) + control bits
     * Per ETS 300 706 section 9.3.1.2
     * S1=0, S2=0 (sub-code), C4=erase flag */
    buf[8]  = hamming84_encode[0];               /* S1 */
    buf[9]  = hamming84_encode[(erase ? 0x08 : 0x00)]; /* S2 + C4 (erase page) */
    buf[10] = hamming84_encode[0];               /* S3 */
    buf[11] = hamming84_encode[0];               /* S4 + C5-C7 */
    buf[12] = hamming84_encode[0];               /* C8-C11 */
    buf[13] = hamming84_encode[0];               /* C12-C14 */

    /* Header display area: 26 bytes (positions 14-39 in data unit payload,
     * 8..33 in the character area after the 8 control bytes).
     * We fill the full 32 character positions with spaces. */
    memset(header_text, ' ', TELETEXT_CHARS_PER_ROW);

    /* Odd parity encode the remaining 32 bytes of header text
     * (positions 6+8=14 through 6+39=45) */
    for (i = 0; i < 32; i++)
        buf[14 + i] = odd_parity[header_text[i] & 0x7F];

    return 46;
}

/**
 * Extract plain text from ASS dialog, stripping override codes.
 */
typedef struct {
    AVBPrint buf;
} TextExtractCtx;

static void text_extract_cb(void *priv, const char *text, int len)
{
    TextExtractCtx *ctx = priv;
    av_bprint_append_data(&ctx->buf, text, len);
}

static void text_newline_cb(void *priv, int forced)
{
    TextExtractCtx *ctx = priv;
    av_bprint_chars(&ctx->buf, '\n', 1);
}

static const ASSCodesCallbacks text_extract_callbacks = {
    .text     = text_extract_cb,
    .new_line = text_newline_cb,
};

static av_cold int dvb_teletext_encode_init(AVCodecContext *avctx)
{
    DVBTeletextEncContext *ctx = avctx->priv_data;
    uint8_t *extradata;
    int magazine_code;
    int teletext_type;

    ctx->ass_ctx = ff_ass_split(avctx->subtitle_header);
    if (!ctx->ass_ctx)
        return AVERROR_INVALIDDATA;

    ctx->page_counter = 0;

    /* Set up extradata for the MPEG-TS muxer's teletext descriptor (0x56).
     * Format: pairs of (teletext_type_magazine, page_number_bcd)
     * teletext_type (5 bits) | magazine_number (3 bits), page_number (8 bits BCD)
     *
     * teletext_type 0x02 = subtitle page
     * magazine 8 is encoded as 0 in the 3-bit field */
    magazine_code = ctx->magazine & 0x07; /* 8 → 0 */
    teletext_type = 0x02; /* subtitle page */

    extradata = av_mallocz(3);
    if (!extradata)
        return AVERROR(ENOMEM);

    extradata[0] = (teletext_type << 3) | magazine_code;
    extradata[1] = ctx->page;
    extradata[2] = 0; /* NUL terminator for safety */

    av_freep(&avctx->extradata);
    avctx->extradata      = extradata;
    avctx->extradata_size = 2;

    av_log(avctx, AV_LOG_INFO,
           "DVB Teletext encoder: magazine %d, page %02X (subtitle)\n",
           ctx->magazine, ctx->page);

    return 0;
}

static int dvb_teletext_encode(AVCodecContext *avctx, unsigned char *buf,
                               int bufsize, const AVSubtitle *sub)
{
    DVBTeletextEncContext *ctx = avctx->priv_data;
    TextExtractCtx extract = { 0 };
    ASSDialog *dialog;
    char *lines[2] = { NULL, NULL };
    int nb_lines = 0;
    uint8_t row_text[TELETEXT_CHARS_PER_ROW];
    int offset = 0;
    int i, ret;

    if (!sub || sub->num_rects == 0)
        return 0;

    /* We need at minimum:
     * 1 byte (data_identifier) + 46 bytes (page header) + 46*2 (content rows) = 139
     * Keep generous margin */
    if (bufsize < 1 + 46 * 3) {
        av_log(avctx, AV_LOG_ERROR, "Buffer too small for teletext packet\n");
        return AVERROR_BUFFER_TOO_SMALL;
    }

    /* Extract plain text from ASS subtitle */
    av_bprint_init(&extract.buf, 0, 256);

    for (i = 0; i < sub->num_rects; i++) {
        const char *ass = sub->rects[i]->ass;

        if (sub->rects[i]->type != SUBTITLE_ASS) {
            av_log(avctx, AV_LOG_WARNING, "Non-ASS subtitle rect, skipping\n");
            continue;
        }

        dialog = ff_ass_split_dialog(ctx->ass_ctx, ass);
        if (!dialog) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }
        ff_ass_split_override_codes(&text_extract_callbacks, &extract, dialog->text);
        ff_ass_free_dialog(&dialog);
    }

    if (!av_bprint_is_complete(&extract.buf)) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (extract.buf.len == 0) {
        av_bprint_finalize(&extract.buf, NULL);
        return 0;
    }

    /* Split text into lines (max 2 for subtitle rows) */
    {
        char *text_str = NULL;
        char *p, *saveptr = NULL;

        av_bprint_finalize(&extract.buf, &text_str);
        if (!text_str)
            return AVERROR(ENOMEM);

        p = av_strtok(text_str, "\n", &saveptr);
        while (p && nb_lines < 2) {
            /* skip empty lines */
            if (*p != '\0')
                lines[nb_lines++] = av_strdup(p);
            p = av_strtok(NULL, "\n", &saveptr);
        }
        av_free(text_str);
    }

    if (nb_lines == 0)
        goto cleanup;

    /* Build teletext PES payload */

    /* Data identifier byte */
    buf[offset++] = TELETEXT_DATA_IDENTIFIER;

    /* Page header (row 0) — erase page before new content */
    offset += write_page_header(buf + offset, ctx, 1);

    /* Content rows: place subtitle text at rows 23 and 24 (bottom of screen) */
    for (i = 0; i < nb_lines; i++) {
        const char *line = lines[i];
        int len, pad_left, j;
        int row = (nb_lines == 1) ? TELETEXT_SUBTITLE_ROW2
                                  : (i == 0 ? TELETEXT_SUBTITLE_ROW1
                                            : TELETEXT_SUBTITLE_ROW2);

        /* Build row text: center the text in 40-char row */
        memset(row_text, ' ', TELETEXT_CHARS_PER_ROW);

        len = strlen(line);
        if (len > TELETEXT_CHARS_PER_ROW)
            len = TELETEXT_CHARS_PER_ROW;

        pad_left = (TELETEXT_CHARS_PER_ROW - len) / 2;
        for (j = 0; j < len; j++) {
            unsigned char c = line[j];
            /* Map to teletext character set (7-bit, Latin G0) */
            if (c < 0x20 || c >= 0x7F)
                c = ' ';
            row_text[pad_left + j] = c;
        }

        offset += write_data_unit(buf + offset, ctx->magazine, row,
                                  row_text, 1);
    }

    ctx->page_counter++;

cleanup:
    for (i = 0; i < nb_lines; i++)
        av_free(lines[i]);
    return offset;

fail:
    av_bprint_finalize(&extract.buf, NULL);
    for (i = 0; i < nb_lines; i++)
        av_free(lines[i]);
    return ret;
}

static av_cold int dvb_teletext_encode_close(AVCodecContext *avctx)
{
    DVBTeletextEncContext *ctx = avctx->priv_data;
    ff_ass_split_free(ctx->ass_ctx);
    return 0;
}

#define OFFSET(x) offsetof(DVBTeletextEncContext, x)
#define SE AV_OPT_FLAG_SUBTITLE_PARAM | AV_OPT_FLAG_ENCODING_PARAM

static const AVOption dvbteletextenc_options[] = {
    { "magazine", "teletext magazine number (1-8)", OFFSET(magazine),
      AV_OPT_TYPE_INT, { .i64 = 8 }, 1, 8, SE },
    { "page", "teletext page number (hex, e.g. 0x88)", OFFSET(page),
      AV_OPT_TYPE_INT, { .i64 = 0x88 }, 0x00, 0xFF, SE },
    { NULL },
};

static const AVClass dvbteletextenc_class = {
    .class_name = "DVB Teletext subtitle encoder",
    .item_name  = av_default_item_name,
    .option     = dvbteletextenc_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_dvb_teletext_encoder = {
    .p.name         = "dvb_teletext",
    CODEC_LONG_NAME("DVB Teletext subtitle encoder"),
    .p.type         = AVMEDIA_TYPE_SUBTITLE,
    .p.id           = AV_CODEC_ID_DVB_TELETEXT,
    .priv_data_size = sizeof(DVBTeletextEncContext),
    .init           = dvb_teletext_encode_init,
    FF_CODEC_ENCODE_SUB_CB(dvb_teletext_encode),
    .close          = dvb_teletext_encode_close,
    .p.priv_class   = &dvbteletextenc_class,
};
