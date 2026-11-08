/* The model folder: calibration.json, tokens.txt, expected_tj3.zfx and, with load_engine, the
 * context's engine files (engine glue). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

/* the same rule as zf/stream.h's ZF_NO_PIN_DEFAULT (the scorer builds without the engine's headers) */
/* The pinning default by platform: phones pin (the pool was tuned for big.LITTLE cores, one stream per
 * app), every other build does not (desktop, server, WebAssembly: a pinned caller thread shares one core
 * with everything else the machine runs; on a busy Windows host a pinned stream ran 18 ms a chunk
 * unpinned and 102 to 631 ms pinned, and the server tier's 8 streams 20.4x against 3.1x real time).
 * Pinning never changes a bit, only timing. */
#if defined(__ANDROID__)
#define ZF_TJ_NO_PIN_DEFAULT 0
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define ZF_TJ_NO_PIN_DEFAULT 0 /* iOS, iPadOS (and the other Apple device OSes) */
#else
#define ZF_TJ_NO_PIN_DEFAULT 1 /* macOS */
#endif
#else
#define ZF_TJ_NO_PIN_DEFAULT 1 /* Windows, Linux, WebAssembly */
#endif

ZfTajweedOpenOptions zf_tajweed_open_default_options(void) {
  ZfTajweedOpenOptions o;
  memset(&o, 0, sizeof o);
  o.struct_size = sizeof o;
  o.context = NULL;
  o.load_engine = 1;
  o.threads = 0;
  o.sample_rate = 0;
  o.batch = 1;
  o.no_pin = ZF_TJ_NO_PIN_DEFAULT;
  return o;
}

static char *join(const char *dir, const char *name) {
  size_t a = strlen(dir), b = strlen(name);
  char *p = (char *)malloc(a + b + 2);
  if (!p) return NULL;
  memcpy(p, dir, a);
  size_t k = a;
  if (a && dir[a - 1] != '/' && dir[a - 1] != '\\') p[k++] = '/';
  memcpy(p + k, name, b + 1);
  return p;
}

static unsigned char *read_in(const char *dir, const char *name, size_t *size) {
  char *p = join(dir, name);
  if (!p) return NULL;
  unsigned char *d = zf_tj_read_file(p, size);
  free(p);
  return d;
}

/* tokens.txt: "<name> <id>" per line, ids 0 .. 37 in order */
static int parse_tokens(ZfTajweedModel *m, const unsigned char *d, size_t n, const char **error) {
  size_t i = 0;
  int count = 0;
  while (i < n) {
    size_t e = i;
    while (e < n && d[e] != '\n') e++;
    size_t le = e;
    if (le > i && d[le - 1] == '\r') le--;
    if (le > i) {
      size_t sp = le;
      while (sp > i && d[sp - 1] != ' ' && d[sp - 1] != '\t') sp--;
      if (sp == i || count >= ZF_TAJWEED_VOCAB) goto bad;
      long id = 0;
      for (size_t k = sp; k < le; ++k) {
        if (d[k] < '0' || d[k] > '9') goto bad;
        id = id * 10 + (d[k] - '0');
        if (id > 1000) goto bad;
      }
      if (id != count || sp == le) goto bad;
      size_t ne = sp;
      while (ne > i && (d[ne - 1] == ' ' || d[ne - 1] == '\t')) ne--;
      if (ne == i) goto bad;
      char *s = (char *)malloc(ne - i + 1);
      if (!s) {
        *error = "out of memory";
        return -1;
      }
      memcpy(s, d + i, ne - i);
      s[ne - i] = 0;
      m->token_names[count++] = s;
    }
    i = e + 1;
  }
  if (count != ZF_TAJWEED_VOCAB) goto bad;
  return 0;
bad:
  *error = "tokens.txt: not 38 lines of \"<token> <id>\" with ids 0 .. 37 in order";
  return -1;
}

void zf_tajweed_model_close(ZfTajweedModel *m) {
  if (!m) return;
  if (m->engine) zf_tajweed__engine_close(m);
  for (int i = 0; i < ZF_TAJWEED_VOCAB; ++i) free(m->token_names[i]);
  zf_tj_expected_free(&m->exp);
  free(m);
}

ZfTajweedModel *zf_tajweed_model_open(const char *model_dir, const ZfTajweedOpenOptions *options,
                                      const char **error) {
  const char *dummy;
  if (!error) error = &dummy;
  ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
  if (options) {
    if (options->struct_size < offsetof(ZfTajweedOpenOptions, batch) + sizeof(int)) {
      *error = "options: struct_size too small";
      return NULL;
    }
    /* only as much as the caller's header had (API 3 appended no_pin); the rest keep their defaults */
    size_t n = options->struct_size < sizeof o ? options->struct_size : sizeof o;
    memcpy(&o, options, n);
    o.struct_size = sizeof o;
  }
  if (!model_dir) {
    *error = "model_dir is NULL";
    return NULL;
  }
  ZfTajweedModel *m = (ZfTajweedModel *)calloc(1, sizeof *m);
  if (!m) {
    *error = "out of memory";
    return NULL;
  }
  m->options = o;
  m->options.context = NULL; /* the caller's string is not kept */
  size_t n;
  unsigned char *d = read_in(model_dir, "calibration.json", &n);
  if (!d) {
    *error = "cannot read calibration.json";
    goto fail;
  }
  int rc = zf_tj_calibration_parse((const char *)d, n, &m->cal, error);
  free(d);
  if (rc) goto fail;
  const char *ctx = o.context ? o.context : m->cal.default_context;
  m->ctx = zf_tj_calibration_context(&m->cal, ctx);
  if (!m->ctx) {
    *error = "the context asked for is not in calibration.json";
    goto fail;
  }
  m->ctx_index = (int)(m->ctx - m->cal.ctx);
  /* tokens */
  d = read_in(model_dir, "tokens.txt", &n);
  if (!d) {
    *error = "cannot read tokens.txt";
    goto fail;
  }
  unsigned char sha[32];
  zf_tj_sha256(d, n, sha);
  rc = parse_tokens(m, d, n, error);
  free(d);
  if (rc) goto fail;
  if (m->cal.has_tokens_sha && memcmp(sha, m->cal.tokens_sha, 32)) {
    *error = "tokens.txt does not match calibration.json's tokens_sha256";
    goto fail;
  }
  /* the expected table */
  d = read_in(model_dir, "expected_tj3.zfx", &n);
  if (!d) {
    *error = "cannot read expected_tj3.zfx";
    goto fail;
  }
  if (zf_tj_expected_load(d, n, &m->exp, error)) {
    free(d);
    goto fail;
  }
  if (memcmp(m->exp.tokens_sha, sha, 32)) {
    *error = "expected_tj3.zfx was built against another tokens.txt";
    goto fail;
  }
  if (o.load_engine) {
    char *cdir = join(model_dir, m->ctx->name);
    if (!cdir) {
      *error = "out of memory";
      goto fail;
    }
    rc = zf_tajweed__engine_open(m, cdir, &o, error);
    free(cdir);
    if (rc) goto fail;
  }
  /* the backend for calibration's backend_overrides */
  strcpy(m->backend, m->engine ? "int8" : "fp32");
  m->backend_index = -1;
  for (int i = 0; i < m->ctx->n_backends; ++i)
    if (!strcmp(m->ctx->backend[i], m->backend)) m->backend_index = i;
  return m;
fail:
  zf_tajweed_model_close(m);
  return NULL;
}

const char *zf_tajweed_model_id(const ZfTajweedModel *m) { return m ? m->cal.model_id : NULL; }
const char *zf_tajweed_model_context(const ZfTajweedModel *m) { return m ? m->ctx->name : NULL; }
const char *zf_tajweed_model_context_at(const ZfTajweedModel *m, int index) {
  return m && index >= 0 && index < m->cal.n_contexts ? m->cal.ctx[index].name : NULL;
}
const char *zf_tajweed_model_default_context(const ZfTajweedModel *m) { return m ? m->cal.default_context : NULL; }
int zf_tajweed_model_chunk_ms(const ZfTajweedModel *m) { return m && m->engine ? m->chunk_ms : 0; }
const char *zf_tajweed_token_name(const ZfTajweedModel *m, int id) {
  return m && id >= 0 && id < ZF_TAJWEED_VOCAB ? m->token_names[id] : NULL;
}

int zf_tajweed_ayah_count(const ZfTajweedModel *m, int surah) {
  if (!m || surah < 1 || surah > m->exp.n_surah) return 0;
  return m->exp.surah_first[surah] - m->exp.surah_first[surah - 1];
}

int zf_tajweed_expected_tokens(const ZfTajweedModel *m, int surah, int ayah, const int **tokens) {
  if (!m) return -1;
  int a = zf_tj_expected_index(&m->exp, surah, ayah);
  if (a < 0) return -1;
  if (tokens) *tokens = m->exp.tok + m->exp.a_tok_off[a];
  return m->exp.a_n_tok[a];
}

int zf_tajweed_expected_words(const ZfTajweedModel *m, int surah, int ayah, const int **word_of_token,
                              int *n_words) {
  if (!m) return -1;
  int a = zf_tj_expected_index(&m->exp, surah, ayah);
  if (a < 0) return -1;
  if (word_of_token) *word_of_token = m->exp.word + m->exp.a_tok_off[a];
  if (n_words) *n_words = m->exp.a_n_words[a];
  return m->exp.a_n_tok[a];
}
