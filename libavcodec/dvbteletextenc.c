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
#define TELETEXT_SUBTITLE_ROW1      22    /* first subtitle row (two-line) */
#define TELETEXT_SUBTITLE_ROW2      23    /* second subtitle row (bottom) */
#define TELETEXT_DATA_UNIT_STUFFING 0xFF  /* stuffing data unit, EN 300 472 */

/*
 * The libzvbi teletext decoder requires PES packets to be exact multiples
 * of 184 bytes (one TS packet payload). With 45-byte PES header:
 *   (pkt_size + 45) % 184 == 0
 * Each data unit is 46 bytes, payload = 1 (data_id) + 46*N.
 * Since 184 = 4*46, we need N ≡ 3 (mod 4), i.e., 3, 7, 11, or 15 units.
 * We always pad to exactly 3 data units (the minimum that passes).
 */
#define TELETEXT_MIN_DATA_UNITS     3

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

/*
 * Bit-reversal table per ETS 300 706 section 7.1.
 *
 * Teletext data is defined with b1 (first transmitted bit) at the LSB.
 * However, DVB PES payload uses MSB-first byte convention. Decoders
 * (libzvbi, CCExtractor/telxcc) bit-reverse each byte of the 44-byte
 * data unit payload to recover the standard representation.
 *
 * The encoder must therefore output bytes in the reversed (MSB-first)
 * form: REVERSE_8[hamming84_encode[nibble]] for Hamming data, and
 * REVERSE_8[odd_parity[char]] for text characters.
 */
static const uint8_t vbi_reverse_8[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0,
    0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8,
    0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4,
    0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC,
    0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2,
    0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA,
    0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6,
    0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE,
    0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1,
    0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9,
    0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5,
    0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED,
    0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3,
    0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB,
    0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7,
    0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF,
    0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

/* Clear displayed subtitle page after this many microseconds with no
 * new CC content.  Prevents stale text from persisting through ad breaks
 * or other content transitions where CC stops. */
#define TELETEXT_ERASE_TIMEOUT_US  10000000  /* 10 seconds */

typedef struct DVBTeletextEncContext {
    AVClass *class;
    ASSSplitContext *ass_ctx;
    int magazine;           /* magazine number 1-8, default 8 */
    int page;               /* page number in hex (0x00-0xFF), default 0x88 */
    int page_counter;       /* erase page sequence counter (C4 flag) */
    int content_active;     /* 1 if display has content (needs erase eventually) */
    int64_t last_content_pts; /* PTS (AV_TIME_BASE) of last content subtitle */
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
 * @param buf         Output buffer
 * @param magazine    Magazine number (1-8)
 * @param row         Row number (0-31)
 * @param text        40-byte text content (space-padded, 7-bit ASCII)
 * @param subtitle    1 for subtitle data unit (0x03), 0 for non-subtitle (0x02)
 * @param line_offset VBI line offset (7-22), must be unique per data unit in a PES
 * @return number of bytes written (always 46: 1+1+1+1+2+40)
 */
static int write_data_unit(uint8_t *buf, int magazine, int row,
                           const uint8_t *text, int subtitle, int line_offset)
{
    int i;

    buf[0] = subtitle ? TELETEXT_DATA_UNIT_SUBTITLE
                      : TELETEXT_DATA_UNIT_NONSUBT;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH; /* 44 bytes */
    /* Per EN 300 472 section 4.4:
     * bits 7-6: reserved "11", bit 5: field_parity, bits 4-0: line_offset
     * Each data unit in a PES must have a unique line_offset since they
     * represent different VBI lines within one video field. */
    buf[2] = 0xE0 | (line_offset & 0x1F); /* reserved="11", field_parity=1 (first field) */
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

    /* Bit-reverse the 42-byte MRAG + content area (bytes 4-45).
     * DVB PES uses MSB-first convention; decoders reverse each byte
     * to recover the standard LSB-first teletext representation.
     * Bytes 2-3 (field_parity, framing_code) are EN 300 472 fields
     * read directly by decoders without reversal. */
    for (i = 0; i < 42; i++)
        buf[4 + i] = vbi_reverse_8[buf[4 + i]];

    return 46; /* 1 + 1 + 1 + 1 + 2 + 40 = 46 */
}

/**
 * Write a stuffing data unit to pad PES payload to required alignment.
 * Per EN 300 472, data_unit_id 0xFF with length 0x2C.
 *
 * @param buf  Output buffer (at least 46 bytes)
 * @return number of bytes written (always 46)
 */
static int write_stuffing_unit(uint8_t *buf)
{
    buf[0] = TELETEXT_DATA_UNIT_STUFFING;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH; /* 44 bytes */
    memset(&buf[2], 0xFF, 44);
    return 46;
}

/**
 * Write a page header data unit (row 0).
 *
 * The page header contains the page address and control bits encoded
 * with Hamming 8/4, followed by 32 bytes of display text (typically spaces).
 *
 * @param buf         Output buffer
 * @param ctx         Encoder context
 * @param erase       1 to set the erase page flag (C4)
 * @param line_offset VBI line offset (7-22)
 * @return number of bytes written
 */
static int write_page_header(uint8_t *buf, DVBTeletextEncContext *ctx,
                             int erase, int line_offset)
{
    uint8_t header_text[TELETEXT_CHARS_PER_ROW];
    int page_units, page_tens;
    int i;

    buf[0] = TELETEXT_DATA_UNIT_SUBTITLE;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH;
    /* Per EN 300 472 section 4.4:
     * bits 7-6: reserved "11", bit 5: field_parity, bits 4-0: line_offset */
    buf[2] = 0xE0 | (line_offset & 0x1F); /* reserved="11", field_parity=1 (first field) */
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
     * S1=0, S2=0 (sub-code), C4=erase flag
     *
     * For subtitle pages: C6=0, C7=1 (Suppress Header), C8=1 (Update Indicator)
     * libzvbi (used by VLC/ffplay) identifies subtitle pages via:
     *   !(C6) && C7 && C8  (C6 must be 0!)
     * CCExtractor/telxcc checks C7 bit for subtitle detection, not C6.
     * Nibble for byte[11]: D1=S4, D2=C5(Newsflash), D3=C6(Subtitle), D4=C7(SuppressHdr)
     * Nibble for byte[12]: D1=C8(Update), D2=C9, D3=C10, D4=C11 */
    buf[8]  = hamming84_encode[0];               /* S1 */
    buf[9]  = hamming84_encode[(erase ? 0x08 : 0x00)]; /* S2 + C4 (erase page) */
    buf[10] = hamming84_encode[0];               /* S3 */
    buf[11] = hamming84_encode[0x08];            /* S4=0, C5=0, C6=0, C7=1(SuppressHdr) */
    buf[12] = hamming84_encode[0x01];            /* C8=1 (Update Indicator), C9=0, C10=0, C11=0 */
    buf[13] = hamming84_encode[0];               /* C12-C14 (charset=0, Latin G0) */

    /* Header display area: 26 bytes (positions 14-39 in data unit payload,
     * 8..33 in the character area after the 8 control bytes).
     * We fill the full 32 character positions with spaces. */
    memset(header_text, ' ', TELETEXT_CHARS_PER_ROW);

    /* Odd parity encode the remaining 32 bytes of header text
     * (positions 6+8=14 through 6+39=45) */
    for (i = 0; i < 32; i++)
        buf[14 + i] = odd_parity[header_text[i] & 0x7F];

    /* Bit-reverse the 42-byte MRAG + content area (bytes 4-45).
     * Bytes 2-3 (field_parity, framing_code) are NOT reversed. */
    for (i = 0; i < 42; i++)
        buf[4 + i] = vbi_reverse_8[buf[4 + i]];

    return 46;
}

/**
 * Convert a UTF-8 string to teletext Latin G0 (7-bit ASCII subset).
 *
 * EIA-608 cc_dec outputs Unicode for special characters (e.g. U+2019
 * RIGHT SINGLE QUOTATION MARK for apostrophes). Teletext only supports
 * 7-bit characters, so we must map multi-byte UTF-8 sequences to their
 * closest ASCII equivalents. Without this, each byte of a multi-byte
 * sequence would be replaced by a space (e.g. "he's" → "he   s").
 *
 * @param src  UTF-8 input string
 * @return     Allocated ASCII string (caller frees with av_free), or NULL on OOM
 */
static char *utf8_to_teletext_g0(const char *src)
{
    AVBPrint bp;
    const uint8_t *s = (const uint8_t *)src;
    int i = 0;
    char *result;

    av_bprint_init(&bp, 0, 256);

    while (s[i]) {
        uint32_t cp;
        int n;

        if (s[i] < 0x80) {
            cp = s[i]; n = 1;
        } else if ((s[i] & 0xE0) == 0xC0 && (s[i + 1] & 0xC0) == 0x80) {
            cp = ((uint32_t)(s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F);
            n = 2;
        } else if ((s[i] & 0xF0) == 0xE0 && (s[i + 1] & 0xC0) == 0x80 &&
                   (s[i + 2] & 0xC0) == 0x80) {
            cp = ((uint32_t)(s[i] & 0x0F) << 12) | ((uint32_t)(s[i + 1] & 0x3F) << 6) |
                 (s[i + 2] & 0x3F);
            n = 3;
        } else if ((s[i] & 0xF8) == 0xF0 && (s[i + 1] & 0xC0) == 0x80 &&
                   (s[i + 2] & 0xC0) == 0x80 && (s[i + 3] & 0xC0) == 0x80) {
            cp = ((uint32_t)(s[i] & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12) |
                 ((uint32_t)(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
            n = 4;
        } else {
            i++; /* invalid byte, skip */
            continue;
        }
        i += n;

        /* Map codepoint to teletext Latin G0 character.
         * Complete coverage of all 79 non-ASCII characters that EIA-608
         * cc_dec can output (ccaption_dec.c charset_overrides[4][128]). */
        if (cp >= 0x20 && cp < 0x7F) {
            av_bprint_chars(&bp, (char)cp, 1);
        } else {
            switch (cp) {
            /* Quotation marks */
            case 0x2018: case 0x2019:             /* ' ' smart single quotes */
                av_bprint_chars(&bp, '\'', 1); break;
            case 0x201C: case 0x201D:             /* " " smart double quotes */
            case 0x00AB: case 0x00BB:             /* « » guillemets */
                av_bprint_chars(&bp, '"', 1);  break;
            /* Dashes and dots */
            case 0x2013: case 0x2014:             /* – — en/em dash */
                av_bprint_chars(&bp, '-', 1);  break;
            case 0x2026:                          /* … ellipsis */
                av_bprintf(&bp, "...");         break;
            case 0x00B7:                          /* · middle dot */
                av_bprint_chars(&bp, '.', 1);  break;
            case 0x00B4:                          /* ´ acute accent */
                av_bprint_chars(&bp, '\'', 1); break;
            /* Lowercase accented vowels */
            case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4: case 0x00E5:
                av_bprint_chars(&bp, 'a', 1); break; /* àáâãäå */
            case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB:
                av_bprint_chars(&bp, 'e', 1); break; /* èéêë */
            case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF:
                av_bprint_chars(&bp, 'i', 1); break; /* ìíîï */
            case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6: case 0x00F8:
                av_bprint_chars(&bp, 'o', 1); break; /* òóôõöø */
            case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC:
                av_bprint_chars(&bp, 'u', 1); break; /* ùúûü */
            case 0x00F1: av_bprint_chars(&bp, 'n', 1); break; /* ñ */
            case 0x00E7: av_bprint_chars(&bp, 'c', 1); break; /* ç */
            /* Capital accented vowels */
            case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4: case 0x00C5:
                av_bprint_chars(&bp, 'A', 1); break; /* ÀÁÂÃÄÅ */
            case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB:
                av_bprint_chars(&bp, 'E', 1); break; /* ÈÉÊË */
            case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF:
                av_bprint_chars(&bp, 'I', 1); break; /* ÌÍÎÏ */
            case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6: case 0x00D8:
                av_bprint_chars(&bp, 'O', 1); break; /* ÒÓÔÕÖØ */
            case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC:
                av_bprint_chars(&bp, 'U', 1); break; /* ÙÚÛÜ */
            case 0x00D1: av_bprint_chars(&bp, 'N', 1); break; /* Ñ */
            case 0x00C7: av_bprint_chars(&bp, 'C', 1); break; /* Ç */
            /* German */
            case 0x00DF: av_bprintf(&bp, "ss");        break; /* ß */
            /* Punctuation */
            case 0x00BF: av_bprint_chars(&bp, '?', 1); break; /* ¿ */
            case 0x00A1: av_bprint_chars(&bp, '!', 1); break; /* ¡ */
            /* Symbols */
            case 0x00AE: av_bprintf(&bp, "(R)");       break; /* ® */
            case 0x00A9: av_bprintf(&bp, "(C)");       break; /* © */
            case 0x2122: av_bprintf(&bp, "TM");        break; /* ™ */
            case 0x2120: av_bprintf(&bp, "SM");        break; /* ℠ */
            case 0x00B0: av_bprint_chars(&bp, 'o', 1); break; /* ° degree */
            case 0x00BD: av_bprintf(&bp, "1/2");       break; /* ½ */
            case 0x00F7: av_bprint_chars(&bp, '/', 1); break; /* ÷ */
            case 0x266A: av_bprint_chars(&bp, '#', 1); break; /* ♪ music note */
            /* Currency */
            case 0x00A2: av_bprint_chars(&bp, 'c', 1); break; /* ¢ */
            case 0x00A3: av_bprint_chars(&bp, '#', 1); break; /* £ (teletext G0 0x23) */
            case 0x00A5: av_bprint_chars(&bp, 'Y', 1); break; /* ¥ */
            case 0x00A4: av_bprint_chars(&bp, '$', 1); break; /* ¤ generic currency */
            case 0x00A6: av_bprint_chars(&bp, '|', 1); break; /* ¦ broken bar */
            /* Whitespace */
            case 0x00A0: av_bprint_chars(&bp, ' ', 1); break; /* NBSP */
            /* Box drawing, full block → space (not displayable) */
            case 0x2588:                              /* █ */
            case 0x250C: case 0x2510:                 /* ┌ ┐ */
            case 0x2514: case 0x2518:                 /* └ ┘ */
            default:
                av_bprint_chars(&bp, ' ', 1); break;
            }
        }
    }

    if (!av_bprint_is_complete(&bp)) {
        av_bprint_finalize(&bp, NULL);
        return NULL;
    }
    av_bprint_finalize(&bp, &result);
    return result;
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

    if (!sub)
        return 0;

    /* Empty subtitle (no rects): either erase stale content or send
     * stuffing-only keepalive for the MPEG-TS interleaver.
     *
     * If content was displayed and no new CC has arrived for 10 seconds,
     * send a page header with erase flag to clear the display. This
     * prevents stale subtitles from persisting through ad breaks.
     *
     * Otherwise send stuffing-only data units which decoders ignore,
     * preserving the currently displayed text while satisfying the
     * MPEG-TS muxer's interleaving requirements. */
    if (sub->num_rects == 0) {
        if (bufsize < 1 + 46 * TELETEXT_MIN_DATA_UNITS)
            return 0;

        if (ctx->content_active && sub->pts != AV_NOPTS_VALUE &&
            ctx->last_content_pts != AV_NOPTS_VALUE &&
            sub->pts - ctx->last_content_pts >= TELETEXT_ERASE_TIMEOUT_US) {
            /* Erase stale content: page header with C4 (erase) flag */
            buf[0] = TELETEXT_DATA_IDENTIFIER;
            offset = 1 + write_page_header(buf + 1, ctx, 1, 7);
            offset += write_stuffing_unit(buf + offset);
            offset += write_stuffing_unit(buf + offset);
            ctx->content_active = 0;
            return offset;
        }

        buf[0] = TELETEXT_DATA_IDENTIFIER;
        offset = 1;
        for (i = 0; i < TELETEXT_MIN_DATA_UNITS; i++)
            offset += write_stuffing_unit(buf + offset);
        return offset;
    }

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

    /* Page header (row 0) — erase page before new content, VBI line 7 */
    offset += write_page_header(buf + offset, ctx, 1, 7);

    /* Content rows: place subtitle text at rows 23 and 24 (bottom of screen)
     * Each data unit gets a unique VBI line offset (8, 9, ...) */
    for (i = 0; i < nb_lines; i++) {
        char *safe_line = utf8_to_teletext_g0(lines[i]);
        const char *line = safe_line ? safe_line : lines[i];
        int len, pad_left, j;
        int row = (nb_lines == 1) ? TELETEXT_SUBTITLE_ROW2
                                  : (i == 0 ? TELETEXT_SUBTITLE_ROW1
                                            : TELETEXT_SUBTITLE_ROW2);

        /* Build row text: center the text in 40-char row, wrapped in
         * Start Box (0x0B) / End Box (0x0A) markers. Teletext subtitle
         * pages use boxed mode where only text between these markers is
         * displayed semi-transparently over video. CCExtractor/telxcc
         * also requires these markers to detect non-empty subtitle pages. */
        memset(row_text, ' ', TELETEXT_CHARS_PER_ROW);

        len = strlen(line);
        if (len > TELETEXT_CHARS_PER_ROW - 4) /* room for 2x start_box + 2x end_box */
            len = TELETEXT_CHARS_PER_ROW - 4;

        /* Double Start Box before text, double End Box after text.
         * Per ETS 300 706 section 12.2, boxed mode requires start/end
         * box codes. Double codes for background transparency. */
        pad_left = (TELETEXT_CHARS_PER_ROW - len - 4) / 2;
        if (pad_left < 0)
            pad_left = 0;
        row_text[pad_left]     = 0x0B; /* Start Box */
        row_text[pad_left + 1] = 0x0B; /* Start Box (double for background) */
        for (j = 0; j < len; j++) {
            unsigned char c = line[j];
            if (c < 0x20 || c >= 0x7F)
                c = ' ';
            row_text[pad_left + 2 + j] = c;
        }
        row_text[pad_left + 2 + len]     = 0x0A; /* End Box */
        row_text[pad_left + 2 + len + 1] = 0x0A; /* End Box (double) */

        offset += write_data_unit(buf + offset, ctx->magazine, row,
                                  row_text, 1, 8 + i);
        av_free(safe_line);
    }

    ctx->page_counter++;
    ctx->content_active    = 1;
    ctx->last_content_pts  = sub->pts;

    /* Pad with stuffing data units to reach TELETEXT_MIN_DATA_UNITS total.
     * This ensures (pkt_size + 45) % 184 == 0 for the libzvbi decoder. */
    {
        /* Count data units written: 1 (page header) + nb_lines (content rows) */
        int data_units = 1 + nb_lines;
        while (data_units < TELETEXT_MIN_DATA_UNITS) {
            offset += write_stuffing_unit(buf + offset);
            data_units++;
        }
    }

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
