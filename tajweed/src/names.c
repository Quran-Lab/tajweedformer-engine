/* Names, versions and small helpers of the tajweed library. */
#include "internal.h"

#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const CLASS_NAMES[ZF_TJ_CLASS_COUNT] = {
    "vowel_sub", "vowel_del", "vowel_ins", "short_to_long", "long_to_short", "cons_sub", "cons_del", "insertion"};
static const int CLASS_CHECK[ZF_TJ_CLASS_COUNT] = {
    ZF_TJ_CHECK_VOWELS, ZF_TJ_CHECK_VOWELS, ZF_TJ_CHECK_VOWELS, ZF_TJ_CHECK_LENGTH,
    ZF_TJ_CHECK_LENGTH, ZF_TJ_CHECK_LETTERS, ZF_TJ_CHECK_LETTERS, ZF_TJ_CHECK_LETTERS};
static const char *const CHECK_NAMES[ZF_TJ_CHECK_COUNT] = {
    "letters", "vowels", "length", "madd", "gemination", "qalqalah", "lazim"};
static const char *const PRESET_NAMES[ZF_TJ_PRESET_COUNT] = {"strict", "balanced", "sensitive", "very_sensitive"};
static const char *const MADD_NAMES[TJ_MADD_CLASSES] = {"none", "fixed2", "muttasil", "free", "lazim"};
static const char *const STATUS_NAMES[3] = {"off", "experimental", "validated"};
static const char *const SOURCE_NAMES[3] = {"calibration", "remote", "app"};

static int find(const char *const *names, int n, const char *s) {
  if (!s) return -1;
  for (int i = 0; i < n; ++i)
    if (!strcmp(names[i], s)) return i;
  return -1;
}

const char *zf_tajweed_class_name(int cls) {
  return cls >= 0 && cls < ZF_TJ_CLASS_COUNT ? CLASS_NAMES[cls] : NULL;
}
int zf_tajweed_class_from_name(const char *name) { return find(CLASS_NAMES, ZF_TJ_CLASS_COUNT, name); }
const char *zf_tajweed_check_name(int check) {
  if (check == ZF_TJ_CHECK_WORD) return "word"; /* API 3 */
  return check >= 0 && check < ZF_TJ_CHECK_COUNT ? CHECK_NAMES[check] : NULL;
}
int zf_tajweed_check_from_name(const char *name) {
  if (name && !strcmp(name, "word")) return ZF_TJ_CHECK_WORD; /* API 3 */
  return find(CHECK_NAMES, ZF_TJ_CHECK_COUNT, name);
}

static const char *const WORD_PRESET_NAMES[ZF_TJ_WORD_PRESET_COUNT] = {"strict", "general", "beginner"};
const char *zf_tajweed_word_preset_name(int preset) {
  if (preset == ZF_TJ_WORD_CUSTOM) return "custom";
  return preset >= 0 && preset < ZF_TJ_WORD_PRESET_COUNT ? WORD_PRESET_NAMES[preset] : NULL;
}
int zf_tajweed_word_preset_from_name(const char *name) { return find(WORD_PRESET_NAMES, ZF_TJ_WORD_PRESET_COUNT, name); }
const char *zf_tajweed_preset_name(int preset) {
  return preset >= 0 && preset < ZF_TJ_PRESET_COUNT ? PRESET_NAMES[preset] : NULL;
}
int zf_tajweed_preset_from_name(const char *name) { return find(PRESET_NAMES, ZF_TJ_PRESET_COUNT, name); }
const char *zf_tajweed_madd_class_name(int madd_class) {
  return madd_class >= 0 && madd_class < TJ_MADD_CLASSES ? MADD_NAMES[madd_class] : NULL;
}
int zf_tajweed_class_check(int cls) { return cls >= 0 && cls < ZF_TJ_CLASS_COUNT ? CLASS_CHECK[cls] : -1; }
const char *zf_tajweed_version(void) { return ZF_TAJWEED_VERSION; }
int zf_tajweed_api_version(void) { return ZF_TAJWEED_API_VERSION; }

/* the heads in ZfTajweedHeads order, which is the ZF_TAJWEED_HEAD_ bit order */
static const struct { const char *name; int frame_ms, cols; } HEADS[12] = {
    {"log_probs", 40, 38},    {"inter1_log_probs", 40, 38}, {"inter4_log_probs", 40, 38},
    {"id_log_probs", 20, 38}, {"timing_log_probs", 20, 38}, {"gem", 20, 2},
    {"attr_tafkheem", 20, 3}, {"attr_ghunna", 20, 5},       {"attr_qalqalah", 20, 3},
    {"attr_face", 20, 4},     {"boundary", 20, 1},          {"conf", 20, 1},
};
const char *zf_tajweed_head_info(unsigned head, int *frame_ms, int *cols) {
  for (int k = 0; k < 12; ++k)
    if (head == (1u << k)) {
      if (frame_ms) *frame_ms = HEADS[k].frame_ms;
      if (cols) *cols = HEADS[k].cols;
      return HEADS[k].name;
    }
  return NULL;
}

/* Top-K (API 3): per row, k passes of a selection over the not yet ranked entries: the next entry is
 * the largest value, the lower id on a tie (a strictly greater value replaces the candidate, scanning
 * ids upwards), which is a stable argsort of the negated row. The margin of the last entry asked for
 * needs the next rank, so one more pass runs when k < cols. NaN is never expected in the heads (they are
 * softmax / log-softmax rows); a NaN entry ranks after every number. */
#define TJ_TOPK_MAX 64
int zf_tajweed_topk(const float *values, int frames, int cols, int k, int *ids, float *values_out,
                    float *margins) {
  if (!values || !ids || frames < 0 || cols < 1 || cols > TJ_TOPK_MAX || k < 1 || k > cols) return -1;
  const int passes = k < cols ? k + 1 : k;
  for (int f = 0; f < frames; ++f) {
    const float *row = values + (size_t)f * cols;
    unsigned char taken[TJ_TOPK_MAX] = {0};
    int rank[TJ_TOPK_MAX + 1];
    for (int j = 0; j < passes; ++j) {
      int best = -1;
      for (int v = 0; v < cols; ++v) {
        if (taken[v]) continue;
        if (best < 0 || row[v] > row[best] || (row[best] != row[best] && row[v] == row[v])) best = v;
      }
      taken[best] = 1;
      rank[j] = best;
    }
    int *io = ids + (size_t)f * k;
    for (int j = 0; j < k; ++j) {
      io[j] = rank[j];
      if (values_out) values_out[(size_t)f * k + j] = row[rank[j]];
      if (margins) margins[(size_t)f * k + j] = j + 1 < cols ? row[rank[j]] - row[rank[j + 1]] : 0.0f;
    }
  }
  return frames;
}

/* The decoded layer's top-1 and runner-up: zf_tajweed_topk at k = 2 (the same ranking as the engine's
 * decoder, so `token` is the greedy argmax). */
int zf_tajweed_top2(const float *lp, int frames, int vocab, ZfTajweedTop2 *out) {
  if (!lp || !out || frames < 0 || vocab < 2) return -1;
  for (int f = 0; f < frames; ++f) {
    int id[2];
    float val[2], mg[2];
    if (zf_tajweed_topk(lp + (size_t)f * vocab, 1, vocab, 2, id, val, mg) != 1) return -1;
    out[f].token = id[0];
    out[f].runner_up = id[1];
    out[f].log_prob = val[0];
    out[f].runner_up_log_prob = val[1];
    out[f].margin = val[0] - val[1];
  }
  return frames;
}

int zf_tj_status_from_name(const char *s) { return find(STATUS_NAMES, 3, s); }
const char *zf_tj_status_name(int status) { return status >= 0 && status < 3 ? STATUS_NAMES[status] : "off"; }
const char *zf_tj_source_name(int source) { return source >= 0 && source < 3 ? SOURCE_NAMES[source] : "?"; }

double zf_tj_lse2(double a, double b) {
  /* numpy.logaddexp's formula (as the phoneme engine's pe_lse2): when the larger argument m has
   * |m| >= 1 and the gap exceeds 40, log1p(exp(-gap)) is below half an ulp of m */
  if (a == b) return a == -INFINITY ? -INFINITY : a + 0.6931471805599453;
  double d = a - b;
  if (d > 0) return (d > 40.0 && fabs(a) >= 1.0) ? a : a + zf_tj_log1p(zf_tj_exp(-d));
  if (d <= 0) return (d < -40.0 && fabs(b) >= 1.0) ? b : b + zf_tj_log1p(zf_tj_exp(d));
  return d; /* NaN */
}

unsigned char *zf_tj_read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  unsigned char *buf = NULL;
  size_t n = 0, cap = 0;
  for (;;) {
    if (cap - n < 65536) {
      size_t nc = cap ? cap * 2 : 65536;
      unsigned char *nb = (unsigned char *)realloc(buf, nc + 1);
      if (!nb) {
        free(buf);
        fclose(f);
        return NULL;
      }
      buf = nb;
      cap = nc;
    }
    size_t r = fread(buf + n, 1, cap - n, f);
    n += r;
    if (r == 0) break;
  }
  int err = ferror(f);
  fclose(f);
  if (err) {
    free(buf);
    return NULL;
  }
  buf[n] = 0;
  *size = n;
  return buf;
}

/* ---- text buffer ------------------------------------------------------------------------------ */

void tj_buf_init(TjBuf *b) { memset(b, 0, sizeof *b); }
void tj_buf_free(TjBuf *b) {
  free(b->p);
  memset(b, 0, sizeof *b);
}

void tj_buf_put(TjBuf *b, const char *s, size_t n) {
  if (b->oom) return;
  if (b->n + n + 1 > b->cap) {
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->n + n + 1) nc *= 2;
    char *np = (char *)realloc(b->p, nc);
    if (!np) {
      b->oom = 1;
      return;
    }
    b->p = np;
    b->cap = nc;
  }
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = 0;
}

void tj_buf_puts(TjBuf *b, const char *s) { tj_buf_put(b, s, strlen(s)); }

void tj_buf_int(TjBuf *b, long long v) {
  char t[32];
  int n = snprintf(t, sizeof t, "%lld", v);
  tj_buf_put(b, t, (size_t)n);
}

void tj_buf_num(TjBuf *b, double v) {
  if (!isfinite(v)) {
    tj_buf_puts(b, "null");
    return;
  }
  char t[64];
  int n = snprintf(t, sizeof t, "%.10g", v);
  /* printf writes the locale's decimal point */
  const struct lconv *lc = localeconv();
  char dp = (lc && lc->decimal_point && lc->decimal_point[0]) ? lc->decimal_point[0] : '.';
  if (dp != '.')
    for (int i = 0; i < n; ++i)
      if (t[i] == dp) t[i] = '.';
  tj_buf_put(b, t, (size_t)n);
}

void tj_buf_str(TjBuf *b, const char *s) {
  tj_buf_put(b, "\"", 1);
  for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
    unsigned c = *p;
    if (c == '"') tj_buf_put(b, "\\\"", 2);
    else if (c == '\\') tj_buf_put(b, "\\\\", 2);
    else if (c == '\n') tj_buf_put(b, "\\n", 2);
    else if (c == '\r') tj_buf_put(b, "\\r", 2);
    else if (c == '\t') tj_buf_put(b, "\\t", 2);
    else if (c < 0x20) {
      char t[8];
      snprintf(t, sizeof t, "\\u%04x", c);
      tj_buf_put(b, t, 6);
    } else {
      tj_buf_put(b, (const char *)p, 1);
    }
  }
  tj_buf_put(b, "\"", 1);
}

void tj_buf_bool(TjBuf *b, int v) { tj_buf_puts(b, v ? "true" : "false"); }
