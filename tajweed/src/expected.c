/* The expected table, expected_tj3.zfx (FORMAT.md): every ayah's tj3 sequence, word ids, per-token
 * site bits, madd entries and the legal forms of every word. Every read is bounds-checked; the file
 * stays in memory and per-token arrays are expanded to int for the public accessors. */
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define ZFX_MAGIC "ZFTJEXP1"
#define ZFX_HEADER 160
#define ZFX_AYAH_REC 20
#define ZFX_MADD_REC 8

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static unsigned rd16(const unsigned char *p) { return (unsigned)p[0] | (unsigned)p[1] << 8; }

void zf_tj_expected_free(TjExpected *e) {
  free(e->data);
  free(e->surah_first);
  free(e->tok);
  free(e->a_tok_off);
  free(e->madd);
  if (e->names)
    for (int i = 0; i <= e->n_names; ++i) free(e->names[i]);
  free(e->names);
  memset(e, 0, sizeof *e);
}

static int fail(TjExpected *e, const char **error, const char *msg) {
  e->data = NULL; /* the caller owns data until success */
  zf_tj_expected_free(e);
  *error = msg;
  return -1;
}

/* checks the forms of one ayah; returns the bytes they take or 0 */
static size_t forms_span(const unsigned char *f, size_t avail, int n_words, int n_names) {
  size_t k = 0;
  for (int w = 0; w < n_words; ++w) {
    if (k + 2 > avail) return 0;
    const int ns = f[k + 1];
    k += 2;
    for (int i = 0; i < ns; ++i) {
      if (k + 1 > avail) return 0;
      const int n = f[k];
      k += 1;
      if (k + (size_t)n > avail) return 0;
      for (int t = 0; t < n; ++t)
        if (f[k + t] <= 1 || f[k + t] >= ZF_TAJWEED_VOCAB - 1) return 0;
      k += (size_t)n;
    }
    if (k + 1 > avail) return 0;
    const int nf = f[k];
    k += 1;
    if (nf == 0 || k + 4 * (size_t)nf > avail) return 0;
    for (int i = 0; i < nf; ++i) {
      const unsigned char *x = f + k + 4 * i;
      if (x[1] >= ns || TJ_FORM_MODE(x[0]) > TJ_FORM_MODE_CANONICAL || (int)rd16(x + 2) > n_names) return 0;
    }
    k += 4 * (size_t)nf;
  }
  return k ? k : 1;
}

int zf_tj_expected_load(const unsigned char *data, size_t size, TjExpected *e, const char **error) {
  memset(e, 0, sizeof *e);
  if (size < ZFX_HEADER || memcmp(data, ZFX_MAGIC, 8)) return fail(e, error, "expected_tj3.zfx: not a ZFTJEXP1 file");
  if (rd32(data + 8) != 1) return fail(e, error, "expected_tj3.zfx: unsupported version");
  if (rd32(data + 12) != ZFX_HEADER || rd32(data + 16) != size)
    return fail(e, error, "expected_tj3.zfx: header or size mismatch (truncated?)");
  const uint32_t n_surah = rd32(data + 20), n_ayah = rd32(data + 24), n_tok = rd32(data + 28),
                 n_madd = rd32(data + 32), forms_bytes = rd32(data + 36);
  if (n_surah == 0 || n_surah > 1000 || n_ayah == 0 || n_ayah > 100000 || n_tok > 50000000 || n_madd > n_tok)
    return fail(e, error, "expected_tj3.zfx: implausible counts");
  memcpy(e->source_sha, data + 40, 32);
  memcpy(e->tokens_sha, data + 72, 32);
  uint32_t off[8];
  for (int i = 0; i < 8; ++i) off[i] = rd32(data + 104 + 4 * i);
  const uint64_t need[8] = {(uint64_t)n_surah * 2, (uint64_t)n_ayah * ZFX_AYAH_REC, n_tok, n_tok,
                            (uint64_t)n_tok * 2, (uint64_t)n_madd * ZFX_MADD_REC, forms_bytes, 4};
  for (int i = 0; i < 8; ++i)
    if (off[i] < ZFX_HEADER || (uint64_t)off[i] + need[i] > size)
      return fail(e, error, "expected_tj3.zfx: a section lies outside the file");
  e->n_surah = (int)n_surah;
  e->n_ayah = (int)n_ayah;
  e->n_tokens = (int)n_tok;
  e->n_madd = (int)n_madd;
  /* wajh names */
  {
    const uint32_t nn = rd32(data + off[7]);
    if (nn > 65535) return fail(e, error, "expected_tj3.zfx: bad name table");
    e->names = (char **)calloc((size_t)nn + 1, sizeof(char *));
    if (!e->names) return fail(e, error, "out of memory");
    e->n_names = (int)nn;
    size_t k = (size_t)off[7] + 4;
    for (uint32_t i = 1; i <= nn; ++i) {
      if (k + 1 > size || k + 1 + data[k] > size) return fail(e, error, "expected_tj3.zfx: bad name table");
      const size_t n = data[k];
      e->names[i] = (char *)malloc(n + 1);
      if (!e->names[i]) return fail(e, error, "out of memory");
      memcpy(e->names[i], data + k + 1, n);
      e->names[i][n] = 0;
      k += 1 + n;
    }
  }
  e->surah_first = (int *)malloc(sizeof(int) * (n_surah + 1));
  e->tok = (int *)malloc(sizeof(int) * 3 * ((size_t)n_tok + 1));
  e->a_tok_off = (int *)malloc(sizeof(int) * 6 * (size_t)n_ayah);
  e->madd = (TjMaddEntry *)malloc(sizeof(TjMaddEntry) * ((size_t)n_madd + 1));
  if (!e->surah_first || !e->tok || !e->a_tok_off || !e->madd) return fail(e, error, "out of memory");
  e->word = e->tok + n_tok;
  e->flags = e->word + n_tok;
  e->a_n_tok = e->a_tok_off + n_ayah;
  e->a_n_words = e->a_n_tok + n_ayah;
  e->a_madd_off = e->a_n_words + n_ayah;
  e->a_n_madd = e->a_madd_off + n_ayah;
  e->a_forms_off = e->a_n_madd + n_ayah;
  /* surahs */
  int acc = 0;
  for (uint32_t s = 0; s < n_surah; ++s) {
    e->surah_first[s] = acc;
    acc += (int)rd16(data + off[0] + 2 * s);
  }
  e->surah_first[n_surah] = acc;
  if ((uint32_t)acc != n_ayah) return fail(e, error, "expected_tj3.zfx: ayah counts do not add up");
  /* tokens */
  for (uint32_t i = 0; i < n_tok; ++i) {
    e->tok[i] = data[off[2] + i];
    e->word[i] = data[off[3] + i];
    e->flags[i] = (int)rd16(data + off[4] + 2 * i);
    if (e->tok[i] >= ZF_TAJWEED_VOCAB || e->tok[i] == ZF_TAJWEED_BLANK)
      return fail(e, error, "expected_tj3.zfx: a token outside tokens.txt");
  }
  /* madd */
  for (uint32_t i = 0; i < n_madd; ++i) {
    const unsigned char *p = data + off[5] + (size_t)i * ZFX_MADD_REC;
    TjMaddEntry *m = &e->madd[i];
    m->pos = (int)rd16(p);
    m->cls = p[2];
    m->kind = p[3];
    m->allowed_mask = p[4];
    m->scoring_mask = p[5];
    m->canonical = p[6];
    m->madd_base = p[7] & 1;
    if (m->cls < 1 || m->cls > TJ_MADD_CONSONANT_LENGTH) return fail(e, error, "expected_tj3.zfx: bad madd class");
  }
  /* ayahs */
  for (uint32_t a = 0; a < n_ayah; ++a) {
    const unsigned char *p = data + off[1] + (size_t)a * ZFX_AYAH_REC;
    const uint32_t to = rd32(p), mo = rd32(p + 4), fo = rd32(p + 8);
    const unsigned nt = rd16(p + 12), nm = rd16(p + 14), nw = p[16];
    if ((uint64_t)to + nt > n_tok || (uint64_t)mo + nm > n_madd || fo >= forms_bytes || nt == 0 || nw == 0)
      return fail(e, error, "expected_tj3.zfx: an ayah record points outside its section");
    for (unsigned i = 0; i < nt; ++i)
      if (e->word[to + i] >= (int)nw) return fail(e, error, "expected_tj3.zfx: a word id past the ayah's words");
    for (unsigned i = 0; i < nm; ++i)
      if (e->madd[mo + i].pos >= (int)nt) return fail(e, error, "expected_tj3.zfx: a madd entry past the ayah");
    if (!forms_span(data + off[6] + fo, forms_bytes - fo, (int)nw, e->n_names))
      return fail(e, error, "expected_tj3.zfx: bad word forms");
    e->a_tok_off[a] = (int)to;
    e->a_n_tok[a] = (int)nt;
    e->a_n_words[a] = (int)nw;
    e->a_madd_off[a] = (int)mo;
    e->a_n_madd[a] = (int)nm;
    e->a_forms_off[a] = (int)fo;
  }
  e->forms = data + off[6];
  e->forms_size = forms_bytes;
  e->data = (unsigned char *)data;
  e->size = size;
  return 0;
}

int zf_tj_expected_index(const TjExpected *e, int surah, int ayah) {
  if (!e->surah_first || surah < 1 || surah > e->n_surah) return -1;
  const int first = e->surah_first[surah - 1], n = e->surah_first[surah] - first;
  if (ayah < 1 || ayah > n) return -1;
  return first + ayah - 1;
}
