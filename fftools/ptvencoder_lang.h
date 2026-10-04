/*
 * ptvencoder_lang.h — FROZEN ISO 639-2 rank table for the content-keyed output PID map
 * (-pid_plan v2, 1.2.1-pre1, T-009).
 *
 * GENERATED ONCE by scripts/gen-ptvencoder-lang.py from libavformat/avlanguage.c @ 01388936a8
 * (484 bibliographic codes, 211 aliases). DO NOT REGENERATE: output PID = block + rank, so
 * any change here moves PIDs on every channel of the fleet. A new ISO 639-2 code, should
 * one ever appear, is APPENDED by hand as the next entry of ptv_lang_bibl (rank 484, 485,
 * ... — the 16 reserved offsets) with PTV_LANG_N bumped and PTV_LANG_NSORTED left alone;
 * never inserted, never re-sorted. lavf's own avlanguage.c is NOT consulted at run time.
 */
#ifndef FFTOOLS_PTVENCODER_LANG_H
#define FFTOOLS_PTVENCODER_LANG_H

#include <stdint.h>

#define PTV_LANG_N       484   /* entries in ptv_lang_bibl (bump when appending) */
#define PTV_LANG_NSORTED 484   /* the first N are alphabetical (binary search); appended ones follow */

/* rank -> ISO 639-2/B code (alphabetical, lavf order) */
static const char ptv_lang_bibl[PTV_LANG_N][4] = {
    "aar", "abk", "ace", "ach", "ada", "ady", "afa", "afh",
    "afr", "ain", "aka", "akk", "alb", "ale", "alg", "alt",
    "amh", "ang", "anp", "apa", "ara", "arc", "arg", "arm",
    "arn", "arp", "art", "arw", "asm", "ast", "ath", "aus",
    "ava", "ave", "awa", "aym", "aze", "bad", "bai", "bak",
    "bal", "bam", "ban", "baq", "bas", "bat", "bej", "bel",
    "bem", "ben", "ber", "bho", "bih", "bik", "bin", "bis",
    "bla", "bnt", "bos", "bra", "bre", "btk", "bua", "bug",
    "bul", "bur", "byn", "cad", "cai", "car", "cat", "cau",
    "ceb", "cel", "cha", "chb", "che", "chg", "chi", "chk",
    "chm", "chn", "cho", "chp", "chr", "chu", "chv", "chy",
    "cmc", "cop", "cor", "cos", "cpe", "cpf", "cpp", "cre",
    "crh", "crp", "csb", "cus", "cze", "dak", "dan", "dar",
    "day", "del", "den", "dgr", "din", "div", "doi", "dra",
    "dsb", "dua", "dum", "dut", "dyu", "dzo", "efi", "egy",
    "eka", "elx", "eng", "enm", "epo", "est", "ewe", "ewo",
    "fan", "fao", "fat", "fij", "fil", "fin", "fiu", "fon",
    "fre", "frm", "fro", "frr", "frs", "fry", "ful", "fur",
    "gaa", "gay", "gba", "gem", "geo", "ger", "gez", "gil",
    "gla", "gle", "glg", "glv", "gmh", "goh", "gon", "gor",
    "got", "grb", "grc", "gre", "grn", "gsw", "guj", "gwi",
    "hai", "hat", "hau", "haw", "heb", "her", "hil", "him",
    "hin", "hit", "hmn", "hmo", "hrv", "hsb", "hun", "hup",
    "iba", "ibo", "ice", "ido", "iii", "ijo", "iku", "ile",
    "ilo", "ina", "inc", "ind", "ine", "inh", "ipk", "ira",
    "iro", "ita", "jav", "jbo", "jpn", "jpr", "jrb", "kaa",
    "kab", "kac", "kal", "kam", "kan", "kar", "kas", "kau",
    "kaw", "kaz", "kbd", "kha", "khi", "khm", "kho", "kik",
    "kin", "kir", "kmb", "kok", "kom", "kon", "kor", "kos",
    "kpe", "krc", "krl", "kro", "kru", "kua", "kum", "kur",
    "kut", "lad", "lah", "lam", "lao", "lat", "lav", "lez",
    "lim", "lin", "lit", "lol", "loz", "ltz", "lua", "lub",
    "lug", "lui", "lun", "luo", "lus", "mac", "mad", "mag",
    "mah", "mai", "mak", "mal", "man", "mao", "map", "mar",
    "mas", "may", "mdf", "mdr", "men", "mga", "mic", "min",
    "mis", "mkh", "mlg", "mlt", "mnc", "mni", "mno", "moh",
    "mon", "mos", "mul", "mun", "mus", "mwl", "mwr", "myn",
    "myv", "nah", "nai", "nap", "nau", "nav", "nbl", "nde",
    "ndo", "nds", "nep", "new", "nia", "nic", "niu", "nno",
    "nob", "nog", "non", "nor", "nqo", "nso", "nub", "nwc",
    "nya", "nym", "nyn", "nyo", "nzi", "oci", "oji", "ori",
    "orm", "osa", "oss", "ota", "oto", "paa", "pag", "pal",
    "pam", "pan", "pap", "pau", "peo", "per", "phi", "phn",
    "pli", "pol", "pon", "por", "pra", "pro", "pus", "que",
    "raj", "rap", "rar", "roa", "roh", "rom", "rum", "run",
    "rup", "rus", "sad", "sag", "sah", "sai", "sal", "sam",
    "san", "sas", "sat", "scn", "sco", "sel", "sem", "sga",
    "sgn", "shn", "sid", "sin", "sio", "sit", "sla", "slo",
    "slv", "sma", "sme", "smi", "smj", "smn", "smo", "sms",
    "sna", "snd", "snk", "sog", "som", "son", "sot", "spa",
    "srd", "srn", "srp", "srr", "ssa", "ssw", "suk", "sun",
    "sus", "sux", "swa", "swe", "syc", "syr", "tah", "tai",
    "tam", "tat", "tel", "tem", "ter", "tet", "tgk", "tgl",
    "tha", "tib", "tig", "tir", "tiv", "tkl", "tlh", "tli",
    "tmh", "tog", "ton", "tpi", "tsi", "tsn", "tso", "tuk",
    "tum", "tup", "tur", "tut", "tvl", "twi", "tyv", "udm",
    "uga", "uig", "ukr", "umb", "und", "urd", "uzb", "vai",
    "ven", "vie", "vol", "vot", "wak", "wal", "war", "was",
    "wel", "wen", "wln", "wol", "xal", "xho", "yao", "yap",
    "yid", "yor", "ypk", "zap", "zbl", "zen", "zha", "znd",
    "zul", "zun", "zxx", "zza",
};

/* ISO 639-2/T and ISO 639-1 spellings (+ deprecated scc/scr) -> rank of the /B code */
typedef struct PtvLangAlias { char from[4]; uint16_t rank; } PtvLangAlias;
static const PtvLangAlias ptv_lang_alias[] = {
    { "aa",   0 }, { "ab",   1 }, { "ae",  33 }, { "af",   8 }, { "ak",  10 }, { "am",  16 },
    { "an",  22 }, { "ar",  20 }, { "as",  28 }, { "av",  32 }, { "ay",  35 }, { "az",  36 },
    { "ba",  39 }, { "be",  47 }, { "bg",  64 }, { "bh",  52 }, { "bi",  55 }, { "bm",  41 },
    { "bn",  49 }, { "bo", 425 }, { "bod", 425 }, { "br",  60 }, { "bs",  58 }, { "ca",  70 },
    { "ce",  76 }, { "ces", 100 }, { "ch",  74 }, { "co",  91 }, { "cr",  95 }, { "cs", 100 },
    { "cu",  85 }, { "cv",  86 }, { "cy", 464 }, { "cym", 464 }, { "da", 102 }, { "de", 149 },
    { "deu", 149 }, { "dv", 109 }, { "dz", 117 }, { "ee", 126 }, { "el", 163 }, { "ell", 163 },
    { "en", 122 }, { "eo", 124 }, { "es", 399 }, { "et", 125 }, { "eu",  43 }, { "eus",  43 },
    { "fa", 341 }, { "fas", 341 }, { "ff", 142 }, { "fi", 133 }, { "fj", 131 }, { "fo", 129 },
    { "fr", 136 }, { "fra", 136 }, { "fy", 141 }, { "ga", 153 }, { "gd", 152 }, { "gl", 154 },
    { "gn", 164 }, { "gu", 166 }, { "gv", 155 }, { "ha", 170 }, { "he", 172 }, { "hi", 176 },
    { "ho", 179 }, { "hr", 180 }, { "ht", 169 }, { "hu", 182 }, { "hy",  23 }, { "hye",  23 },
    { "hz", 173 }, { "ia", 193 }, { "id", 195 }, { "ie", 191 }, { "ig", 185 }, { "ii", 188 },
    { "ik", 198 }, { "in", 195 }, { "io", 187 }, { "is", 186 }, { "isl", 186 }, { "it", 201 },
    { "iu", 190 }, { "iw", 172 }, { "ja", 204 }, { "ji", 472 }, { "jv", 202 }, { "jw", 202 },
    { "ka", 148 }, { "kat", 148 }, { "kg", 229 }, { "ki", 223 }, { "kj", 237 }, { "kk", 217 },
    { "kl", 210 }, { "km", 221 }, { "kn", 212 }, { "ko", 230 }, { "kr", 215 }, { "ks", 214 },
    { "ku", 239 }, { "kv", 228 }, { "kw",  90 }, { "ky", 225 }, { "la", 245 }, { "lb", 253 },
    { "lg", 256 }, { "li", 248 }, { "ln", 249 }, { "lo", 244 }, { "lt", 250 }, { "lu", 255 },
    { "lv", 246 }, { "mg", 282 }, { "mh", 264 }, { "mi", 269 }, { "mk", 261 }, { "mkd", 261 },
    { "ml", 267 }, { "mn", 288 }, { "mo", 358 }, { "mr", 271 }, { "mri", 269 }, { "ms", 273 },
    { "msa", 273 }, { "mt", 283 }, { "my",  65 }, { "mya",  65 }, { "na", 300 }, { "nb", 312 },
    { "nd", 303 }, { "ne", 306 }, { "ng", 304 }, { "nl", 115 }, { "nld", 115 }, { "nn", 311 },
    { "no", 315 }, { "nr", 302 }, { "nv", 301 }, { "ny", 320 }, { "oc", 325 }, { "oj", 326 },
    { "om", 328 }, { "or", 327 }, { "os", 330 }, { "pa", 337 }, { "pi", 344 }, { "pl", 345 },
    { "ps", 350 }, { "pt", 347 }, { "qu", 351 }, { "rm", 356 }, { "rn", 359 }, { "ro", 358 },
    { "ron", 358 }, { "ru", 361 }, { "rw", 224 }, { "sa", 368 }, { "sc", 400 }, { "scc", 402 },
    { "scr", 180 }, { "sd", 393 }, { "se", 386 }, { "sg", 363 }, { "si", 379 }, { "sk", 383 },
    { "sl", 384 }, { "slk", 383 }, { "sm", 390 }, { "sn", 392 }, { "so", 396 }, { "sq",  12 },
    { "sqi",  12 }, { "sr", 402 }, { "ss", 405 }, { "st", 398 }, { "su", 407 }, { "sv", 411 },
    { "sw", 410 }, { "ta", 416 }, { "te", 418 }, { "tg", 422 }, { "th", 424 }, { "ti", 427 },
    { "tk", 439 }, { "tl", 423 }, { "tn", 437 }, { "to", 434 }, { "tr", 442 }, { "ts", 438 },
    { "tt", 417 }, { "tw", 445 }, { "ty", 414 }, { "ug", 449 }, { "uk", 450 }, { "ur", 453 },
    { "uz", 454 }, { "ve", 456 }, { "vi", 457 }, { "vo", 458 }, { "wa", 466 }, { "wo", 467 },
    { "xh", 469 }, { "yi", 472 }, { "yo", 473 }, { "za", 478 }, { "zh",  78 }, { "zho",  78 },
    { "zu", 480 },
};
#define PTV_LANG_NALIAS 211

#endif /* FFTOOLS_PTVENCODER_LANG_H */
