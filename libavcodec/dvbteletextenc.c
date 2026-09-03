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

#include <assert.h>     /* static_assert (the subset-table coupling below) */
#include <stdio.h>      /* sscanf in text_extract_naive() */
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
/* The block of caption rows grows UPWARDS from its last row, so a 1-row caption occupies one
 * row, a 2-row one two, and a 4-row roll-up four. Which row it ends on follows the SOURCE's
 * own vertical position (ttx_row_from_cea_row()); a bottom-third caption — very nearly
 * everything — ends on row 23, the conventional DVB-subtitle band. EIA-608 carries up to 4
 * rows (roll-up 4, multi-line pop-on); anything beyond that is dropped. */
#define TELETEXT_SUBTITLE_LAST_ROW  23    /* bottom subtitle row */
#define TELETEXT_MAX_SUBTITLE_ROWS  4     /* at most 4 rows per caption */
#define TELETEXT_DATA_UNIT_STUFFING 0xFF  /* stuffing data unit, EN 300 472 */
/* Page 0xFF in our own magazine is the "filler"/time-filling page (ETS 300 706 9.3.1): a
 * header no receiver displays, used to switch AWAY from the subtitle page. It MUST differ
 * from the subtitle page — otherwise the clear PES emits two headers for the same page, the
 * second one without C4, and there is no away-switch at all. */
#define TELETEXT_FILLER_PAGE        0xFF
#define TELETEXT_FILLER_ALT_PAGE    0xFE  /* when the operator asked for page 0xFF itself */

/*
 * The libzvbi teletext decoder requires PES packets to be exact multiples
 * of 184 bytes (one TS packet payload). With 45-byte PES header:
 *   (pkt_size + 45) % 184 == 0
 * Each data unit is 46 bytes, payload = 1 (data_id) + 46*N.
 * Since 184 = 4*46, we need N ≡ 3 (mod 4), i.e., 3, 7, 11, or 15 units.
 * We always pad to exactly 3 data units (the minimum that passes).
 */
#define TELETEXT_MIN_DATA_UNITS     3
/* A content PES carries the page header + up to TELETEXT_MAX_SUBTITLE_ROWS content rows +
 * the away-switch filler header = up to 6 units, so it pads to the next legal count, 7. */
#define TELETEXT_CONTENT_DATA_UNITS 7

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

/*
 * G0 National Option Subsets per ETS 300 706 Tables 33-39.
 *
 * 13 character positions in the G0 Latin set can be replaced with national
 * characters. The positions (in order) correspond to ASCII codes:
 *   0x23 0x24 0x40 0x5B 0x5C 0x5D 0x5E 0x5F 0x60 0x7B 0x7C 0x7D 0x7E
 * Each subset maps these 13 positions to different Unicode codepoints.
 * Selected via C12-C14 in the page header — see write_page_header().
 *
 * MEASURED, not derived: every row below is what a real libzvbi (Homebrew 0.2.44 — the
 * library VLC links) RENDERS for those 13 bytes under the matching ETS national option
 * code, i.e. the glyphs a viewer actually sees. Two rows were wrong until 2026-08-12
 * (English claimed plain ASCII for 0x5B..0x7E; Italian was shifted in 6 of 13), which put
 * a different LETTER on screen than the caption asked for. Re-measure with
 * test-scripts/teletext-oracle before editing a row.
 *
 * WHICH REGION you measure under matters, and measuring under the wrong one is how the Czech
 * row came to be deleted: ETS 300 706 indexes the G0 subsets by region as well as by code, and
 * libzvbi's default region is 16. Codes 0,1,2,4,5,6 render IDENTICALLY under regions 0 and 16
 * (measured), so only subset 6 (code 3) is region-sensitive — set the region explicitly when
 * re-measuring it.
 */
#define G0_NATIONAL_POSITIONS 13
#define G0_NATIONAL_SUBSETS   7

static const uint8_t g0_position_bytes[G0_NATIONAL_POSITIONS] = {
    0x23, 0x24, 0x40, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
    0x60, 0x7B, 0x7C, 0x7D, 0x7E
};

/* our g0_subset index -> the ETS 300 706 national option code transmitted in C12-C14.
 * The two numberings are BIT-REVERSALS of each other (ETS transmits C12 first, our index
 * reads that bit last), so they can never be used interchangeably:
 *   0 English->0, 1 German->4, 2 Swedish->2, 3 Italian->6, 4 French->1, 5 Spa/Por->5,
 *   6 Czech/Slovak->3. */
static const uint8_t g0_subset_ets_code[G0_NATIONAL_SUBSETS] = { 0, 4, 2, 6, 1, 5, 3 };

/* Unicode codepoints for each national subset's 13 positions. */
static const uint32_t g0_national_subsets[G0_NATIONAL_SUBSETS][G0_NATIONAL_POSITIONS] = {
    /* Subset 0: English (ETS code 0) */
    { 0x00A3, 0x0024, 0x0040, 0x2190, 0x00BD, 0x2192,  /* £ $ @ ← ½ →  */
      0x2191, 0x0023, 0x2014, 0x00BC, 0x2016, 0x00BE,  /* ↑ # — ¼ ‖ ¾  */
      0x00F7 },                                          /* ÷            */
    /* Subset 1: German (ETS code 4) */
    { 0x0023, 0x0024, 0x00A7, 0x00C4, 0x00D6, 0x00DC,  /* # $ § Ä Ö Ü  */
      0x005E, 0x005F, 0x00B0, 0x00E4, 0x00F6, 0x00FC,  /* ^ _ ° ä ö ü  */
      0x00DF },                                          /* ß            */
    /* Subset 2: Swedish/Finnish/Hungarian (ETS code 2) */
    { 0x0023, 0x00A4, 0x00C9, 0x00C4, 0x00D6, 0x00C5,  /* # ¤ É Ä Ö Å  */
      0x00DC, 0x005F, 0x00E9, 0x00E4, 0x00F6, 0x00E5,  /* Ü _ é ä ö å  */
      0x00FC },                                          /* ü            */
    /* Subset 3: Italian (ETS code 6) */
    { 0x00A3, 0x0024, 0x00E9, 0x00B0, 0x00E7, 0x2192,  /* £ $ é ° ç →  */
      0x2191, 0x0023, 0x00F9, 0x00E0, 0x00F2, 0x00E8,  /* ↑ # ù à ò è  */
      0x00EC },                                          /* ì            */
    /* Subset 4: French (ETS code 1) */
    { 0x00E9, 0x00EF, 0x00E0, 0x00EB, 0x00EA, 0x00F9,  /* é ï à ë ê ù  */
      0x00EE, 0x0023, 0x00E8, 0x00E2, 0x00F4, 0x00FB,  /* î # è â ô û  */
      0x00E7 },                                          /* ç            */
    /* Subset 5: Spanish/Portuguese (ETS code 5) */
    { 0x00E7, 0x0024, 0x00A1, 0x00E1, 0x00E9, 0x00ED,  /* ç $ ¡ á é í  */
      0x00F3, 0x00FA, 0x00BF, 0x00FC, 0x00F1, 0x00E8,  /* ó ú ¿ ü ñ è  */
      0x00E0 },                                          /* à            */
    /* Subset 6: Czech/Slovak (ETS code 3) — the one REGION-DEPENDENT row.
     *
     * ETS 300 706 numbers the G0 subsets per REGION (Table 33 is indexed by region AND code),
     * and code 3 is the one our set uses where the regions disagree:
     *   region 0  -> Czech/Slovak. MEASURED with a raw libzvbi (vbi_teletext_set_default_region
     *                (dec, 0)): the 13 bytes below render "#ůčťžýířéáěúš", byte-for-byte this
     *                row.
     *   region 16 -> Turkish, rendering "ğİŞÖÇÜĞışöçü". 16 is libzvbi's own DEFAULT region, so
     *                a receiver that never sets one shows Turkish letters.
     * The 2026-08-12 removal of this row was made off a region-16 reading of code 3 and was
     * wrong: the pre-existing `-cc_lang cze` was accidentally correct for region-0 receivers.
     * Every other subset we transmit renders identically in both regions (measured across
     * codes 0,1,2,4,5,6), so this row is the only one whose glyphs depend on the receiver. */
    { 0x0023, 0x016F, 0x010D, 0x0165, 0x017E, 0x00FD,  /* # ů č ť ž ý  */
      0x00ED, 0x0159, 0x00E9, 0x00E1, 0x011B, 0x00FA,  /* í ř é á ě ú  */
      0x0161 },                                          /* š            */
};

/* MASTER COPY of the subset numbering. Four things must agree on it, and three of them are in
 * other files, so they are tied here rather than left to be noticed:
 *   - g0_national_subsets[] / g0_subset_ets_code[] / subset_names[] and the g0_subset option's
 *     maximum, all below (the static asserts);
 *   - g0_subset_info[] and lang_to_g0_subset() in fftools/ptvencoder.c;
 *   - NAT_NAME[] / NAT_LANGS[] in test-scripts/teletext-oracle/ttxwire.py, which cross-checks
 *     the wire against the descriptor language.
 * A subset added here without the others silently mislabels a channel's charset. */
static_assert(FF_ARRAY_ELEMS(g0_subset_ets_code) == G0_NATIONAL_SUBSETS,
              "g0_subset_ets_code[] must cover every G0 subset");
static_assert(FF_ARRAY_ELEMS(g0_national_subsets) == G0_NATIONAL_SUBSETS,
              "g0_national_subsets[] must cover every G0 subset");

/**
 * Look up a Unicode codepoint in a G0 national option subset.
 *
 * @param cp      Unicode codepoint to look up
 * @param subset  Subset index (0 .. G0_NATIONAL_SUBSETS-1)
 * @return The teletext byte position (0x23-0x7E), or 0 if not found
 */
static uint8_t g0_national_lookup(uint32_t cp, int subset)
{
    int i;
    if (subset < 0 || subset >= G0_NATIONAL_SUBSETS)
        return 0;
    for (i = 0; i < G0_NATIONAL_POSITIONS; i++) {
        if (g0_national_subsets[subset][i] == cp)
            return g0_position_bytes[i];
    }
    return 0;
}

/*
 * Substitutions for the 13 national-option positions when the current subset cannot render
 * that character at all. Those 13 bytes are NOT plain ASCII on the wire: whatever we put
 * there renders as the subset's national glyph (measured: 0x23 => £ on English, ç on
 * Spanish, é on French; 0x7C => ‖ / ò / û). Emitting one raw therefore puts a different
 * LETTER on screen than the caption asked for, which is what g0_put_ascii() below prevents.
 *
 *   src  subst   why
 *   ---  -----   -------------------------------------------------------------------------
 *    #   "No."   number sign; only unreachable on Spanish/Portuguese
 *    $   "USD"   unreachable on French (0x24 is ï there) and on Swedish/Finnish/Hungarian
 *                (0x24 is ¤ there) — measured, not only French as this used to claim
 *    @   "(a)"   every subset except English puts a letter at 0x40
 *    [   "("     keep the bracketing, lose the shape
 *    ]   ")"
 *    {   "("     cc_dec emits { } \ ^ _ | ~ from its extended charset (0x29..0x2F)
 *    }   ")"
 *    \   "/"
 *    |   "/"     also where the ¦ fallback lands
 *    ~   "-"
 *    `   "'"
 *    _   "-"     '_' exists on German/Swedish only
 *    ^   " "     decoration; no letter stands in for a caret
 */
static const struct { char c; const char *sub; } g0_ascii_fallback[] = {
    { '#', "No." }, { '$', "USD" }, { '@', "(a)" }, { '[', "("  }, { ']', ")" },
    { '{', "("   }, { '}', ")"   }, { '\\', "/"  }, { '|', "/"  }, { '~', "-" },
    { '`', "'"   }, { '_', "-"   }, { '^', " "   },
};

/**
 * Append one ASCII character so that it RENDERS as itself under this G0 subset.
 *
 * Characters outside the 13 national positions go out unchanged. The 13 go through the
 * subset's own table first (e.g. '#' is at 0x5F on English/French/Italian but at 0x23 on
 * German/Swedish), and only fall back to the substitution table above when the subset has
 * no position for them.
 */
static void g0_put_ascii(AVBPrint *bp, uint32_t cp, int subset)
{
    uint8_t nb;
    int i;

    for (i = 0; i < G0_NATIONAL_POSITIONS; i++)
        if (g0_position_bytes[i] == cp)
            break;
    if (i == G0_NATIONAL_POSITIONS) {          /* not a national position: safe as-is */
        av_bprint_chars(bp, (char)cp, 1);
        return;
    }
    nb = g0_national_lookup(cp, subset);       /* does this subset show that glyph? */
    if (nb) {
        av_bprint_chars(bp, (char)nb, 1);
        return;
    }
    for (i = 0; i < FF_ARRAY_ELEMS(g0_ascii_fallback); i++)
        if ((uint32_t)(unsigned char)g0_ascii_fallback[i].c == cp) {
            av_bprintf(bp, "%s", g0_ascii_fallback[i].sub);
            return;
        }
    av_bprint_chars(bp, ' ', 1);               /* the two tables cover the same 13 bytes */
}

typedef struct DVBTeletextEncContext {
    AVClass *class;
    ASSSplitContext *ass_ctx;
    int magazine;           /* magazine number 1-8, default 8 */
    int page;               /* page number in hex (0x00-0xFF), default 0x88 */
    int g0_subset;          /* G0 national option subset (0-6), default 0 (English) */
    int content_active;     /* 1 if display has content (needs erase eventually) */
    int last_nb_lines;      /* number of content rows in previous subtitle */
    int last_base_row;      /* teletext row the previous subtitle's LAST line landed on */
    uint32_t last_rows_mask; /* bitmap of the rows the previous subtitle drew */
    int last_contig;        /* previous subtitle's source rows were one unbroken band */
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
 * @param page        Page number in hex (0x00-0xFF) within ctx->magazine
 * @param erase       1 to set the erase page flag (C4)
 * @param line_offset VBI line offset (7-22)
 * @return number of bytes written
 */
static int write_page_header(uint8_t *buf, DVBTeletextEncContext *ctx,
                             int page, int erase, int line_offset)
{
    uint8_t header_text[TELETEXT_CHARS_PER_ROW];
    int page_units, page_tens;
    /* Only the page we announce in the teletext descriptor is flagged as a subtitle
     * page; the blank filler page must not be, or decoders that follow the subtitle
     * flags would treat it as a second subtitle service. */
    int subtitle_page = (page == ctx->page);
    int i;

    buf[0] = subtitle_page ? TELETEXT_DATA_UNIT_SUBTITLE
                           : TELETEXT_DATA_UNIT_NONSUBT;
    buf[1] = TELETEXT_DATA_UNIT_LENGTH;
    /* Per EN 300 472 section 4.4:
     * bits 7-6: reserved "11", bit 5: field_parity, bits 4-0: line_offset */
    buf[2] = 0xE0 | (line_offset & 0x1F); /* reserved="11", field_parity=1 (first field) */
    buf[3] = TELETEXT_FRAMING_CODE;

    /* MRAG for row 0 */
    encode_mrag(ctx->magazine, 0, &buf[4]);

    /* Page address: page number as two Hamming 8/4 bytes (BCD) */
    page_units = page & 0x0F;
    page_tens  = (page >> 4) & 0x0F;
    buf[6] = hamming84_encode[page_units];
    buf[7] = hamming84_encode[page_tens];

    /* Sub-code bytes (S1-S4) + control bits C4-C14, per ETS 300 706 section 9.3.1.1.
     * Each byte below carries ONE Hamming 8/4 nibble whose bits are D1..D4, with D1 the LSB
     * of the value we index hamming84_encode[] with. The layout — get this wrong and the
     * bits land on the neighbouring control (which is exactly what happened here before
     * 2026-08-12, see buf[13]):
     *
     *   buf[8]   D1..D4 = S1
     *   buf[9]   D1..D3 = S2,          D4 = C4  erase page
     *   buf[10]  D1..D4 = S3
     *   buf[11]  D1,D2  = S4,          D3 = C5  newsflash,  D4 = C6  SUBTITLE
     *   buf[12]  D1 = C7 suppress header, D2 = C8 update, D3 = C9 interrupted sequence,
     *            D4 = C10 inhibit display
     *   buf[13]  D1 = C11 magazine serial, D2..D4 = C12..C14 national option subset
     *
     * So the two nibbles below transmit C6=1 (subtitle page) and C7=1 (suppress header) and
     * nothing else — C5=0, C8=C9=C10=0. That combination is precisely what libavcodec's
     * libzvbi wrapper tests for (libzvbi-teletextdec.c: "!(flags1 & 0x40) && flags1 & 0x80
     * && flags2 & 0x01" reading these same two bytes = !newsflash && subtitle &&
     * suppress_header), it is what TSDuck and a raw libzvbi accept as a subtitle page, and
     * it is what production has been shipping. DO NOT CHANGE THESE BITS — in particular
     * there is no "Update Indicator must stay 1" rule here: C8 is 0 and always has been.
     * The earlier version of this comment named these bits one position off. */
    buf[8]  = hamming84_encode[0];                             /* S1 */
    buf[9]  = hamming84_encode[(erase ? 0x08 : 0x00)];         /* S2 + C4 erase page */
    buf[10] = hamming84_encode[0];                             /* S3 */
    buf[11] = hamming84_encode[subtitle_page ? 0x08 : 0x00];   /* S4=0, C5=0, C6=subtitle */
    buf[12] = hamming84_encode[subtitle_page ? 0x01 : 0x00];   /* C7=suppress hdr, C8..C10=0 */
    /* C11=0 (magazine serial), C12-C14 = the ETS national option code.
     *
     * MEASURED with a raw libzvbi across all 16 nibble values: the table it renders is
     * nibble >> 1 — D1 is C11 and charset-irrelevant. That is
     *   0x0 English  0x2 French  0x4 Swedish/Finnish  0x6 Turkish
     *   0x8 German   0xA Portuguese/Spanish  0xC Italian  0xE reserved
     * and our subset numbering is the bit-reversal of the ETS code (g0_subset_ets_code[]),
     * so the nibble is code << 1 (subset 5 Spanish -> code 5 -> 0x0A).
     *
     * Until 2026-08-12 this line wrote the raw subset index, which both mis-shifted it (the
     * code landed on C11-C13) and skipped the reversal, so the two errors did not cancel:
     * -cc_lang spa announced Swedish/Finnish and put "Se#or" / "qu@" on air. */
    {
        int s = (ctx->g0_subset >= 0 && ctx->g0_subset < G0_NATIONAL_SUBSETS)
              ? ctx->g0_subset : 0;
        buf[13] = hamming84_encode[(g0_subset_ets_code[s] << 1) & 0x0F];
    }

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
static char *utf8_to_teletext_g0(const char *src, int g0_subset)
{
    AVBPrint bp;
    const uint8_t *s = (const uint8_t *)src;
    int i = 0;
    char *result;

    /* UNLIMITED, not a 256-byte cap: a full CC screen is up to 15 rows and cc_dec
     * emits multi-byte UTF-8 for accents and symbols, so a real caption can exceed
     * 256 bytes.  A capped AVBPrint would go !is_complete and fail the whole
     * encode, silencing the PID for as long as such captions keep arriving. */
    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);

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
         * cc_dec can output (ccaption_dec.c charset_overrides[4][128]).
         *
         * First try the G0 national option subset — if the codepoint has
         * a native representation, emit it directly. Otherwise fall
         * through to accent-stripping for best-effort ASCII output.
         *
         * ASCII is NOT a free pass: 13 of those codes are national-option positions whose
         * rendered glyph depends on the subset, so they go through g0_put_ascii(). Before
         * 2026-08-12 they were emitted raw and "[MUSIC]" rendered "←MUSIC→" on English. */
        if (cp >= 0x20 && cp < 0x7F) {
            g0_put_ascii(&bp, cp, g0_subset);
        } else {
            uint8_t national_byte = g0_national_lookup(cp, g0_subset);
            if (national_byte) {
                av_bprint_chars(&bp, (char)national_byte, 1);
                continue;
            }
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
            /* ♪ -> '*', which is the same glyph in every subset. '#' (what this used to
             * emit) is a national position and rendered £ on English, ç on Spanish,
             * é on French — a music note turning into a currency symbol. */
            case 0x266A: av_bprint_chars(&bp, '*', 1); break; /* ♪ music note */
            /* Currency */
            case 0x00A2: av_bprint_chars(&bp, 'c', 1); break; /* ¢ */
            /* £ is at 0x23 on English/Italian and the national lookup above already took
             * it there; the subsets that have no £ get the currency code spelled out
             * rather than a byte that renders as some other letter. */
            case 0x00A3: av_bprintf(&bp, "GBP");       break; /* £ */
            case 0x00A5: av_bprint_chars(&bp, 'Y', 1); break; /* ¥ */
            case 0x00A4: g0_put_ascii(&bp, '$', g0_subset);  break; /* ¤ generic currency */
            case 0x00A6: g0_put_ascii(&bp, '|', g0_subset);  break; /* ¦ broken bar */
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
 *
 * cc_dec (real_time=1) prefixes EVERY caption row with "{\an7}{\pos(x,y)}" — one \pos per
 * '\N'-separated row, carrying where that row sat on the 608 screen. \pos reaches us as the
 * .move callback, so we record each row's y while extracting and can put the caption back
 * on the band the source chose. Without it everything was bottom-anchored: measured on
 * Law_Crime, 7 of 22 caption rows are TOP-of-screen in the source and were slammed onto the
 * lower third, on top of the channel's own lower-third graphics.
 */
#define TT_MAX_SEGMENTS 24     /* 15 rows on a 608 screen + slack for multi-rect captions */

typedef struct {
    AVBPrint buf;
    int nb_seg;                     /* segment being built == newlines emitted so far */
    int seg_y[TT_MAX_SEGMENTS];     /* ASS y per segment; -1 = this row carried no \pos */
} TextExtractCtx;

static void text_extract_init(TextExtractCtx *ctx)
{
    int i;
    ctx->nb_seg = 0;
    for (i = 0; i < TT_MAX_SEGMENTS; i++)
        ctx->seg_y[i] = -1;
}

/* one row ends: the next \pos belongs to the next segment */
static void text_extract_nl(TextExtractCtx *ctx)
{
    av_bprint_chars(&ctx->buf, '\n', 1);
    if (ctx->nb_seg < TT_MAX_SEGMENTS - 1)
        ctx->nb_seg++;
}

static void text_extract_cb(void *priv, const char *text, int len)
{
    TextExtractCtx *ctx = priv;
    int i;
    /* ASS \h (hard space) and \n (soft newline) are not handled by
     * ff_ass_split_override_codes and arrive as literal text.
     * Replace them inline: \h → space, \n → newline. */
    for (i = 0; i < len; i++) {
        if (text[i] == '\\' && i + 1 < len) {
            if (text[i + 1] == 'h') {
                av_bprint_chars(&ctx->buf, ' ', 1);
                i++;
                continue;
            } else if (text[i + 1] == 'n') {
                text_extract_nl(ctx);
                i++;
                continue;
            }
        }
        av_bprint_chars(&ctx->buf, text[i], 1);
    }
}

static void text_newline_cb(void *priv, int forced)
{
    text_extract_nl(priv);
}

/* \pos(x,y) arrives here as move(x,y,x,y,-1,-1). cc_dec emits exactly one per row and never
 * a real (animated) \move, so the start point is the row position; first one per segment
 * wins. */
static void text_move_cb(void *priv, int x1, int y1, int x2, int y2, int t1, int t2)
{
    TextExtractCtx *ctx = priv;
    if (ctx->nb_seg < TT_MAX_SEGMENTS && ctx->seg_y[ctx->nb_seg] < 0)
        ctx->seg_y[ctx->nb_seg] = y1;
}

static const ASSCodesCallbacks text_extract_callbacks = {
    .text     = text_extract_cb,
    .new_line = text_newline_cb,
    .move     = text_move_cb,
};

/**
 * Fallback text extraction for a dialog ff_ass_split_override_codes refuses.
 *
 * That function returns AVERROR_INVALIDDATA when it meets a "{\" it cannot parse as an
 * override block, and it stops there — everything after is lost. This is reachable from
 * CAPTION TEXT, not from malformed markup: cc_dec's extended charset maps 0x29 to '{' and
 * 0x2B to '\' and does no ASS escaping, so a caption containing "{\" truncates. Measured
 * 2026-08-12 with the real cc_dec: "AB{\CD" reached the wire as "AB", and a caption that
 * BEGINS that way extracted to nothing at all — which this encoder reads as a source erase
 * and clears the page.
 *
 * So: skip balanced {...} override blocks (keeping \pos for the row), honour \N \n \h, and
 * treat an UNTERMINATED block as the literal text it is. A '{' from caption text followed by
 * a later '}' elsewhere in the line still loses the run between them — bounded, and far from
 * losing the rest of the caption. Fixed in our own layer deliberately: ass_split.c is stock.
 */
static void text_extract_naive(TextExtractCtx *ctx, const char *s)
{
    while (*s) {
        if (s[0] == '{' && s[1] == '\\') {
            const char *e = strchr(s, '}');
            int x, y;
            if (!e) {                          /* unterminated => literal caption text */
                av_bprint_chars(&ctx->buf, *s++, 1);
                continue;
            }
            /* %9d, not %d: the field width bounds what an ARBITRARY caption-text "{\pos(" can
             * feed sscanf. 9 digits cannot overflow an int, so no input reaches undefined
             * behaviour; a longer run of digits simply stops matching. */
            if (sscanf(s, "{\\pos(%9d,%9d)", &x, &y) == 2)
                text_move_cb(ctx, x, y, x, y, -1, -1);
            s = e + 1;
            continue;
        }
        if (s[0] == '\\' && (s[1] == 'N' || s[1] == 'n')) {
            text_extract_nl(ctx);
            s += 2;
            continue;
        }
        if (s[0] == '\\' && s[1] == 'h') {
            av_bprint_chars(&ctx->buf, ' ', 1);
            s += 2;
            continue;
        }
        av_bprint_chars(&ctx->buf, *s++, 1);
    }
}

/**
 * Remove the six HTML-ish tags that arrive inside CAPTION TEXT.
 *
 * MEASURED 2026-08-12 on TruBLU: 20 caption rows carry a literal "</b>" and 20 a "<b>", and
 * they come from the SOURCE's own EIA-608 character stream, not from our pipeline. Two
 * independent proofs: ccextractor (an entirely separate 608 decoder) reads the same tags out of
 * the same capture, and the raw 608 byte dump spells them out in the character positions
 * ("...the rest of his family.</b>"). cc_dec is NOT the origin — it expresses styling as ASS
 * override blocks ({\i1} {\u0} {\c&H...&}, see ccaption_dec.c:512-600) and its
 * charset_overrides[] entries are single codepoints, so it cannot emit a tag at all. Some
 * upstream captioner is leaking its own markup into the caption.
 *
 * Whatever the origin, "</b>" is not text a viewer should read, and teletext has no styling to
 * convert it into. EXACTLY these six tags, whole and case-insensitive, and nothing broader:
 * '<' is an ordinary caption character, so a caption reading "x < y" or "<<< BREAKING" must
 * survive untouched. No "skip to the next '>'" rule for that reason.
 *
 * Applied once to the fully extracted text, so it covers both the ff_ass_split_override_codes
 * path and text_extract_naive() — this is the single place it happens.
 */
static void strip_markup_tags(char *s)
{
    static const char *const tags[] = { "<b>", "</b>", "<i>", "</i>", "<u>", "</u>" };
    char *w = s;

    while (*s) {
        int i;
        size_t n = 0;

        for (i = 0; i < FF_ARRAY_ELEMS(tags); i++) {
            size_t l = strlen(tags[i]);
            if (!av_strncasecmp(s, tags[i], l)) {
                n = l;
                break;
            }
        }
        if (n) {
            s += n;
            continue;
        }
        *w++ = *s++;
    }
    *w = '\0';
}

/**
 * Invert cc_dec's row -> ASS y mapping (ccaption_dec.c:
 * y = ASS_DEFAULT_PLAYRESY * (0.1 + 0.0533 * row), so rows 0..14 give y 28..243).
 *
 * @return the 608 screen row 0..14, or -1 if the row carried no position
 */
static int cea_row_from_ass_y(int y)
{
    int row;

    if (y < 0)
        return -1;
    row = (int)((y / (double)ASS_DEFAULT_PLAYRESY - 0.1) / 0.0533 + 0.5);
    if (row < 0)
        row = 0;
    if (row > 14)
        row = 14;
    return row;
}

/* 608 rows TTX_BOTTOM_BAND_CEA..14 are the lower third — the subtitle band — and they ALL
 * bottom-anchor to row 23 rather than mapping proportionally.
 *
 * Not a taste call: a roll-up's lowest occupied row alternates INSIDE that band from one
 * snapshot to the next as rows scroll, so a proportional map moves the whole block between
 * transmissions. Measured 2026-08-12 on Newsmax2 (RU2, all 60 PACs at 608 row 12) with the
 * band starting at 13: the 1-row snapshot landed on teletext 18 and the 2-row one on 19/20,
 * the base row jumped 38 times in 90 s, and each move forces a C4 full-page erase (C4 rate
 * 12.0% -> 23.4%) — a visible flash and a caption leaving the band viewers read.
 *
 * The band's lower edge is where the corpus puts every roll-up base: Newsmax2 12, TruBLU 13,
 * NTD and Weather_nation 14 (0-based, measured from the sources' own PACs). 11 is the first
 * row below all of them, so the whole band is stable and only genuinely-higher captions
 * (608 row <= 10, i.e. Law_Crime's top pair and its mid-screen pair) depart from row 23. */
#define TTX_BOTTOM_BAND_CEA 11

/* How far apart two of a caption's 608 rows must be before the caption is laid out as separate
 * BANDS instead of one block — more than half the 15-row screen, i.e. a real upper-vs-lower
 * third change. See the split site in dvb_teletext_encode() for why "any gap" is wrong. */
#define TTX_SPLIT_GAP_CEA 7

/**
 * 608 screen row 0..14 -> teletext row 1..23, keeping the caption in the same vertical band.
 *
 * The bottom BAND (see TTX_BOTTOM_BAND_CEA) maps to 23 exactly, so the overwhelmingly common
 * lower-third caption produces byte-identical rows to the plain bottom anchoring this replaces
 * (measured: NTD 3-row roll-up on 21/22/23, TruBLU on 22/23, Newsmax2 on 22/23). Rows 0..10
 * map proportionally onto teletext 1..17. Row 0 is the page header and can never be used.
 */
static int ttx_row_from_cea_row(int cea)
{
    int row;

    if (cea >= TTX_BOTTOM_BAND_CEA)
        return TELETEXT_SUBTITLE_LAST_ROW;
    row = 1 + (cea * 22 + 7) / 14;             /* 1 + round(cea * 22/14) */
    if (row < 1)
        row = 1;
    if (row > TELETEXT_SUBTITLE_LAST_ROW)
        row = TELETEXT_SUBTITLE_LAST_ROW;
    return row;
}

static av_cold int dvb_teletext_encode_init(AVCodecContext *avctx)
{
    DVBTeletextEncContext *ctx = avctx->priv_data;
    uint8_t *extradata;
    int magazine_code;
    int teletext_type;

    ctx->ass_ctx = ff_ass_split(avctx->subtitle_header);
    if (!ctx->ass_ctx)
        return AVERROR_INVALIDDATA;


    /* Set up extradata for the MPEG-TS muxer's teletext descriptor (0x56).
     * Format: pairs of (teletext_type_magazine, page_number_bcd)
     * teletext_type (5 bits) | magazine_number (3 bits), page_number (8 bits BCD)
     *
     * teletext_type 0x02 = subtitle page
     * magazine 8 is encoded as 0 in the 3-bit field */
    magazine_code = ctx->magazine & 0x07; /* 8 → 0 */
    teletext_type = 0x02; /* subtitle page */

    extradata = av_mallocz(2 + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!extradata)
        return AVERROR(ENOMEM);

    extradata[0] = (teletext_type << 3) | magazine_code;
    extradata[1] = ctx->page;

    av_freep(&avctx->extradata);
    avctx->extradata      = extradata;
    avctx->extradata_size = 2;

    {
        /* names AND the ETS national option code actually transmitted in C12-C14, because
         * the two numberings differ (see g0_subset_ets_code[]) and a log line that shows
         * only our index cannot be checked against the wire */
        static const char *subset_names[G0_NATIONAL_SUBSETS] = {
            "English", "German", "Swedish/Finnish/Hungarian", "Italian",
            "French", "Spanish/Portuguese",
            "Czech/Slovak (region-0) / Turkish (region-16)"
        };
        int s = (ctx->g0_subset >= 0 && ctx->g0_subset < G0_NATIONAL_SUBSETS)
              ? ctx->g0_subset : 0;
        static_assert(FF_ARRAY_ELEMS(subset_names) == G0_NATIONAL_SUBSETS,
                      "subset_names[] must cover every G0 subset");
        av_log(avctx, AV_LOG_INFO,
               "DVB Teletext encoder: magazine %d, page %02X, "
               "G0 subset %d (%s) = ETS national option code %d, header nibble 0x%X\n",
               ctx->magazine, ctx->page, s, subset_names[s],
               g0_subset_ets_code[s], g0_subset_ets_code[s] << 1);
    }

    ctx->last_content_pts = AV_NOPTS_VALUE;   /* no content yet: the stale-erase guard below
                                               * must be REAL, not decorative (a 0 here made
                                               * every first timeout fire off pts - 0) */

    return 0;
}

/* The blank filler page to switch away to. Never ctx->page: two headers for the same page in
 * one PES is not an away-switch, it is a second (C4-less) transmission of the subtitle page. */
static int filler_page(const DVBTeletextEncContext *ctx)
{
    return ctx->page == TELETEXT_FILLER_PAGE ? TELETEXT_FILLER_ALT_PAGE
                                             : TELETEXT_FILLER_PAGE;
}

/* Clear whatever is currently displayed. Used both for an explicit source erase and for
 * the stale-content timeout. Two mechanisms in ONE PES, in this order:
 *
 *   1. a blank page header for OUR subtitle page, C4 (erase page) set;
 *   2. the blank filler page in the same magazine (see filler_page(); 8FF by default).
 *
 * (2) is how a real broadcast encoder clears: the GB News page-888 reference keeps a blank
 * 8FF running constantly and switches away from the subtitle page instead of wiping it,
 * which is why C4 appears on only ~5% of its page transmissions. It also terminates the
 * page-888 transmission immediately, so a decoder that only completes a page when the next
 * header arrives raises its clear event now rather than seconds later.
 *
 * (1) is kept because (2) ALONE does not clear libzvbi: ffmpeg's libzvbi-teletextdec.c
 * drops any VBI_EVENT_TTX_PAGE whose page number is outside the txt_page filter (and VLC
 * filters the same way), so a decoder tuned to 888 never sees the 8FF at all and holds the
 * last caption on screen. Measured — do not drop the blank 888 header. */
static int write_erase_page(DVBTeletextEncContext *ctx, unsigned char *buf,
                            int bufsize)
{
    int offset;

    if (bufsize < 1 + 46 * TELETEXT_MIN_DATA_UNITS)
        return 0;

    buf[0] = TELETEXT_DATA_IDENTIFIER;
    offset  = 1 + write_page_header(buf + 1, ctx, ctx->page, 1, 7);
    offset += write_page_header(buf + offset, ctx, filler_page(ctx), 0, 8);
    offset += write_stuffing_unit(buf + offset);
    ctx->content_active = 0;

    return offset;
}

/* Stuffing-only page: decoders ignore it, so the displayed text is preserved
 * while the MPEG-TS interleaver keeps seeing packets on this sparse PID. */
static int write_keepalive_page(unsigned char *buf, int bufsize)
{
    int i, offset;

    if (bufsize < 1 + 46 * TELETEXT_MIN_DATA_UNITS)
        return 0;

    buf[0] = TELETEXT_DATA_IDENTIFIER;
    offset = 1;
    for (i = 0; i < TELETEXT_MIN_DATA_UNITS; i++)
        offset += write_stuffing_unit(buf + offset);

    return offset;
}

static int dvb_teletext_encode(AVCodecContext *avctx, unsigned char *buf,
                               int bufsize, const AVSubtitle *sub)
{
    DVBTeletextEncContext *ctx = avctx->priv_data;
    TextExtractCtx extract = { 0 };
    ASSDialog *dialog;
    char *lines[TELETEXT_MAX_SUBTITLE_ROWS] = { NULL };
    int line_y[TELETEXT_MAX_SUBTITLE_ROWS];      /* each line's ASS y, -1 = unpositioned */
    int line_row[TELETEXT_MAX_SUBTITLE_ROWS];    /* each line's teletext row, ascending */
    int nb_lines = 0;
    int base_row = TELETEXT_SUBTITLE_LAST_ROW;   /* teletext row of the LAST content line */
    uint32_t rows_mask = 0;                      /* bitmap of the rows used, for the C4 test */
    int contig = 1;                              /* source rows form ONE unbroken band */
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
        /* A BACKWARD pts (the emitter's house-stamp baseline moved: timeline rebase, mv
         * REANCHOR2) must RE-BASE this timer, not disarm it. Left alone, the difference
         * below stays negative for the whole size of the step and the 10 s stale erase
         * never fires — the last caption sits on screen through the ad break. */
        if (sub->pts != AV_NOPTS_VALUE && ctx->last_content_pts != AV_NOPTS_VALUE &&
            sub->pts < ctx->last_content_pts)
            ctx->last_content_pts = sub->pts;
        if (ctx->content_active && sub->pts != AV_NOPTS_VALUE &&
            ctx->last_content_pts != AV_NOPTS_VALUE &&
            sub->pts - ctx->last_content_pts >= TELETEXT_ERASE_TIMEOUT_US)
            return write_erase_page(ctx, buf, bufsize);

        return write_keepalive_page(buf, bufsize);
    }

    /* A content PES is padded to TELETEXT_CONTENT_DATA_UNITS: 1 byte (data_identifier) +
     * 46 * 7 = 323 — the worst case is the page header + 4 content rows + the away-switch
     * filler header = 6 units, and 7 is the next unit count with (size + 45) % 184 == 0. */
    if (bufsize < 1 + 46 * TELETEXT_CONTENT_DATA_UNITS) {
        av_log(avctx, AV_LOG_ERROR, "Buffer too small for teletext packet\n");
        return AVERROR_BUFFER_TOO_SMALL;
    }

    /* Extract plain text from ASS subtitle */
    /* UNLIMITED: see utf8_to_teletext_g0().  A capped buffer turns a long caption
     * into an encode failure rather than a truncated page. */
    av_bprint_init(&extract.buf, 0, AV_BPRINT_SIZE_UNLIMITED);
    text_extract_init(&extract);

    for (i = 0; i < sub->num_rects; i++) {
        const char *ass = sub->rects[i]->ass;
        unsigned mark_len;
        int mark_seg, k;

        if (sub->rects[i]->type != SUBTITLE_ASS) {
            av_log(avctx, AV_LOG_WARNING, "Non-ASS subtitle rect, skipping\n");
            continue;
        }

        dialog = ff_ass_split_dialog(ctx->ass_ctx, ass);
        if (!dialog) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }
        mark_len = extract.buf.len;
        mark_seg = extract.nb_seg;
        if (ff_ass_split_override_codes(&text_extract_callbacks, &extract, dialog->text) < 0) {
            /* Roll this rect's partial extraction back and redo it ourselves — see
             * text_extract_naive(). The truncation is the same operation av_bprint_clear()
             * performs, at an offset; guarded on is_complete so it cannot touch a buffer that
             * failed to allocate. */
            if (av_bprint_is_complete(&extract.buf) && extract.buf.len > mark_len) {
                extract.buf.str[mark_len] = '\0';
                extract.buf.len = mark_len;
            }
            for (k = mark_seg; k < TT_MAX_SEGMENTS; k++)
                extract.seg_y[k] = -1;
            extract.nb_seg = mark_seg;
            text_extract_naive(&extract, dialog->text);
        }
        ff_ass_free_dialog(&dialog);
    }

    if (!av_bprint_is_complete(&extract.buf)) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if (extract.buf.len == 0) {
        av_bprint_finalize(&extract.buf, NULL);
        /* An explicit source erase (EIA-608 EDM/EOC) arrives as a caption whose
         * text is empty once the ASS markup is stripped. Clear the page now
         * instead of leaving stale text up until the 10 s stale-content
         * timeout — otherwise a deliberate caption-off is invisible on the wire
         * and the previous caption persists over the following shot. */
        if (ctx->content_active)
            return write_erase_page(ctx, buf, bufsize);
        return 0;
    }

    /* Split text into lines, keeping each line's source position.
     * Walked by hand rather than with av_strtok because av_strtok SKIPS empty tokens: a
     * blanked caption row would then shift every following row onto the wrong \pos. */
    {
        char *text_str = NULL;
        char *p;
        int seg = 0;

        av_bprint_finalize(&extract.buf, &text_str);
        if (!text_str)
            return AVERROR(ENOMEM);
        strip_markup_tags(text_str);       /* source-leaked <b>/<i>/<u>, see the function */

        p = text_str;
        while (p && nb_lines < TELETEXT_MAX_SUBTITLE_ROWS) {
            char *nl = strchr(p, '\n');
            if (nl)
                *nl = '\0';
            /* Map into the teletext G0 repertoire FIRST, then decide whether the
             * line still shows anything.  The mapping turns every codepoint it
             * cannot represent into a space — including NBSP and the U+2588 full
             * block that real EIA-608 encoders use for blanked rows — so a line
             * that looks non-empty as UTF-8 can map to nothing but spaces.
             * Drawing that would put a boxed grey bar on screen where the source
             * asked for a clear one; treating it as empty routes it to the erase
             * below instead. */
            char *mapped = utf8_to_teletext_g0(p, ctx->g0_subset);
            const unsigned char *q = (const unsigned char *)(mapped ? mapped : p);

            while (*q && *q <= ' ')
                q++;
            if (*q != '\0') {
                line_y[nb_lines]   = seg < TT_MAX_SEGMENTS ? extract.seg_y[seg] : -1;
                lines[nb_lines++]  = mapped ? mapped : av_strdup(p);
            } else
                av_free(mapped);
            seg++;
            p = nl ? nl + 1 : NULL;
        }
        av_free(text_str);
    }

    if (nb_lines == 0) {
        /* Markup or whitespace only — same meaning as the empty-text case above:
         * the source is showing nothing. Clear the page rather than returning
         * silently and leaving the previous caption up until the 10 s timeout. */
        if (ctx->content_active)
            return write_erase_page(ctx, buf, bufsize);
        return 0;
    }

    /* WHERE the block goes — computed BEFORE the page header, because a block that lands on
     * different rows than the last one needs C4 (see below).
     *
     * The LAST line lands on base_row and the block grows upwards, one teletext row per
     * caption row, in order (1 row => base, 2 => base-1 + base, ...).
     *
     * base_row is the row the SOURCE's lowest caption row maps to, so a top-of-screen caption
     * stays at the top instead of being slammed onto the lower third (measured on Law_Crime:
     * 7 of 22 caption rows are at 608 rows 0-1). The bottom BAND maps to 23, so
     * bottom-third captions — very nearly everything — are byte-identical to the old
     * unconditional bottom anchoring. A caption whose rows carry no \pos at all (not cc_dec,
     * or a rect we could not position) keeps the plain bottom anchoring. */
    {
        int cea[TELETEXT_MAX_SUBTITLE_ROWS];
        int max_cea = -1, positioned = 1;

        for (i = 0; i < nb_lines; i++) {
            cea[i] = cea_row_from_ass_y(line_y[i]);
            if (cea[i] < 0)
                positioned = 0;
            if (cea[i] > max_cea)
                max_cea = cea[i];
        }
        /* SPLIT a caption whose lines sit in genuinely different SCREEN REGIONS. One block
         * cannot express that: base_row comes from the LOWEST row and the lines are laid
         * consecutively upwards, so a caption occupying 608 rows {0, 14} lands entirely at the
         * bottom, its top line one row above its bottom one, straight over the channel's
         * lower-third graphics — the exact bug honouring \pos exists to prevent, reintroduced
         * inside a single caption.
         *
         * The threshold is a gap of TTX_SPLIT_GAP_CEA, not "any gap", and that is measured, not
         * cautious. cc_dec initialises AND resets its cursor to screen row 10
         * (ccaption_dec.c:293,322) and uses it for anything written before the first PAC
         * arrives, so at the start of a capture it reports row 10 — or 9 after one roll-up
         * shift — for text the SOURCE placed at row 13/14. Measured 2026-08-12: RAV_Espanol's
         * PACs are ALL at 608 row 14 (RU3) and Daystar_esp's at 13/14 only, yet cc_dec's first
         * transmissions reported {9,14} and {10,13}. Splitting on those apparent gaps of 3-5
         * tore two lines of one roll-up sentence 8 teletext rows apart. A gap this large is a
         * decoder start-up artifact; more than half the 15-row screen is a real region change,
         * and it is also more than a 4-row block can absorb. */
        if (positioned)
            for (i = 1; i < nb_lines; i++)
                if (cea[i] - cea[i - 1] >= TTX_SPLIT_GAP_CEA)
                    contig = 0;

        if (contig) {
            /* ONE block, growing upwards from base_row, in source order. */
            if (max_cea >= 0)
                base_row = ttx_row_from_cea_row(max_cea);
            if (base_row < nb_lines)           /* row 0 is the page header: never a content row */
                base_row = nb_lines;
            if (base_row > TELETEXT_SUBTITLE_LAST_ROW)
                base_row = TELETEXT_SUBTITLE_LAST_ROW;
            for (i = 0; i < nb_lines; i++)
                line_row[i] = base_row - (nb_lines - 1 - i);
        } else {
            /* One mapped row per line, then each contiguous BAND collapsed so its lines sit
             * directly on top of each other, then pushed apart upwards until the rows are
             * strictly ascending — a band whose mapped row collides with the band below it
             * moves UP, never down, so the lower (and lower-third) band keeps its place. */
            for (i = 0; i < nb_lines; i++)
                line_row[i] = ttx_row_from_cea_row(cea[i]);
            for (i = nb_lines - 1; i > 0; i--)
                if (cea[i] - cea[i - 1] < TTX_SPLIT_GAP_CEA)   /* same band: directly above */
                    line_row[i - 1] = line_row[i] - 1;
            if (line_row[nb_lines - 1] > TELETEXT_SUBTITLE_LAST_ROW)
                line_row[nb_lines - 1] = TELETEXT_SUBTITLE_LAST_ROW;
            for (i = nb_lines - 1; i > 0; i--)
                if (line_row[i - 1] >= line_row[i])
                    line_row[i - 1] = line_row[i] - 1;
            /* The top line can now be above row 1 (row 0 is the page header). Slide the whole
             * caption down; it cannot run off the bottom, because reaching row 0 at all means
             * the lowest row is at most TELETEXT_MAX_SUBTITLE_ROWS. */
            if (line_row[0] < 1) {
                int d = 1 - line_row[0];
                for (i = 0; i < nb_lines; i++)
                    line_row[i] += d;
            }
            base_row = line_row[nb_lines - 1];
        }
        for (i = 0; i < nb_lines; i++)
            rows_mask |= 1u << line_row[i];
    }

    /* Build teletext PES payload */

    /* Data identifier byte */
    buf[offset++] = TELETEXT_DATA_IDENTIFIER;

    /* Page header (row 0) — smart erase flag:
     * Set C4 (erase) only on first subtitle after silence or when the
     * row count decreases (orphaned rows need clearing). Subsequent
     * updates use erase=0 so the decoder replaces content in-place
     * without a visible blank flash between updates. */
    {
        /* Measured against a real broadcast (GB News page 888: C4 on 9 of 171 page
         * transmissions, ~5%). C4 on every page — what this replaces — is what made the
         * display flash clear->redraw on each update.
         *
         * This is only safe BECAUSE of the away-switch appended below. Measured with a RAW
         * libzvbi (Homebrew 0.2.44, none of libavcodec/libzvbi-teletextdec.c's wrapper
         * logic — that wrapper's last_p5 block re-asserts C4 itself and stores the MUTATED
         * byte, so it self-propagates and CANNOT tell C4-always from C4-once; VLC has no
         * such workaround): raw libzvbi raises VBI_EVENT_TTX_PAGE only for transmissions
         * whose header carries C4. Without the away-switch, 17 of 61 caption updates on
         * TruBLU raised NO event at all — 28% of updates invisible on a real receiver.
         * Do not remove the filler header while C4 is conditional. */
        /* base_row is in the condition because a page is only ever PARTIALLY overwritten: the
         * rows this transmission does not carry keep whatever was on them. Fewer rows than
         * last time, or the same rows shifted to a different band, leaves the leftovers on
         * screen — measured on the fixed Daystar encode before this clause: caption 1 on row
         * 17, caption 2 on rows 22/23, and libzvbi showed all three at once. */
        /* The rows_mask term covers the non-contiguous layout above, where the bottom row and
         * the line count no longer determine WHICH rows are drawn: two captions can share both
         * and still leave orphans. It is gated on either caption being non-contiguous so that
         * the contiguous case — the whole measured corpus — keeps the exact C4 pattern it had
         * (a 2-row caption growing to 3 rows on the same base row must NOT start erasing). */
        int erase = !ctx->content_active || nb_lines < ctx->last_nb_lines ||
                    base_row != ctx->last_base_row ||
                    ((!contig || !ctx->last_contig) && rows_mask != ctx->last_rows_mask);
        offset += write_page_header(buf + offset, ctx, ctx->page, erase, 7);
    }

    /* Content rows. A source caption with more rows than TELETEXT_MAX_SUBTITLE_ROWS used to
     * be silently truncated to two: measured 4 of 56 captions on TruBLU lost their third row
     * ("Send me the" without "address.").
     * Each data unit gets a unique VBI line offset (8, 9, ...) per EN 300 472. */
    for (i = 0; i < nb_lines; i++) {
        const char *line = lines[i];   /* already G0-mapped at split time */
        int len, pad_left, j;
        int row = line_row[i];

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
    }

    /* AWAY-SWITCH: terminate this transmission of the subtitle page with a blank filler-page
     * header, the way a real broadcast encoder does (the GB News page-888 reference runs a
     * blank 8FF continuously between subtitle transmissions, which is why C4 appears on only
     * ~5% of its page transmissions). Every subtitle header is then preceded by a
     * different-page header, i.e. a FRESH page instance rather than a repeat.
     *
     * This is what makes the conditional C4 above safe: measured with a raw libzvbi, a
     * C4-less repeat of the same page raises no VBI_EVENT_TTX_PAGE at all. */
    offset += write_page_header(buf + offset, ctx, filler_page(ctx), 0, 8 + nb_lines);

    ctx->content_active    = 1;
    ctx->last_nb_lines     = nb_lines;
    ctx->last_base_row     = base_row;
    ctx->last_rows_mask    = rows_mask;
    ctx->last_contig       = contig;
    ctx->last_content_pts  = sub->pts;

    /* Pad with stuffing data units to reach TELETEXT_CONTENT_DATA_UNITS total.
     * This ensures (pkt_size + 45) % 184 == 0 for the libzvbi decoder. */
    {
        /* 1 (page header) + nb_lines (content rows) + 1 (away-switch filler) */
        int data_units = 2 + nb_lines;
        while (data_units < TELETEXT_CONTENT_DATA_UNITS) {
            offset += write_stuffing_unit(buf + offset);
            data_units++;
        }
    }

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
    /* Our index, NOT the ETS code (they are bit-reversals of each other, see
     * g0_subset_ets_code[]); the ETS code each one transmits is in parentheses. Subset 6 is
     * the only region-dependent one — see the Czech/Slovak row in g0_national_subsets[]. */
    { "g0_subset", "G0 national option subset (0=English[ETS 0], 1=German[4], "
      "2=Swedish/Finnish/Hungarian[2], 3=Italian[6], 4=French[1], "
      "5=Spanish/Portuguese[5], 6=Czech/Slovak[3], Turkish on a region-16 receiver)",
      OFFSET(g0_subset),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, G0_NATIONAL_SUBSETS - 1, SE },
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
