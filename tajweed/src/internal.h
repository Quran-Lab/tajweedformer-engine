/* Internal declarations of the tajweed library (zf_tajweed.h): not part of the public interface. */
#ifndef ZF_TAJWEED_INTERNAL_H
#define ZF_TAJWEED_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "zf_tajweed.h"

#ifndef ZF_TAJWEED_VERSION
#define ZF_TAJWEED_VERSION "2.0.0"
#endif

#define TJ_MAX_CONTEXTS 8
#define TJ_NAME_MAX 16
#define TJ_MAX_BACKENDS 4
#define TJ_MADD_CLASSES 5 /* ZF_TJ_MADD_NONE .. ZF_TJ_MADD_LAZIM */
#define TJ_MADD_CONSONANT_LENGTH 5 /* expected-table madd entries that are not on the madd ladder */

/* ---- deterministic double libm (tj_libm.c, from the phoneme engine's pe_libm.c) ---------------- */
double zf_tj_exp(double x);
double zf_tj_log(double x);
double zf_tj_log1p(double x);
/* numpy's add.reduce over a contiguous float32 array, bit for bit (pairwise blocks of 8 up to 128) */
float zf_tj_np_sum_f32(const float *a, int n);
double zf_tj_np_sum_f64(const double *a, int n);
/* numpy.logaddexp's formula */
double zf_tj_lse2(double a, double b);

/* ---- sha256 (tj_sha256.c, the engine's sha256.c under its own name) --------------------------- */
void zf_tj_sha256(const void *data, size_t n, unsigned char out[32]);

/* ---- calibration.json, one context -------------------------------------------------------------- */
typedef struct {
  int present;                  /* the class is in "sites" */
  int status;                   /* ZF_TJ_STATUS_* (absent: OFF) */
  int has_thr[ZF_TJ_PRESET_COUNT];
  double thr[ZF_TJ_PRESET_COUNT];
  double a;                     /* fusion weight of log_probs (1 - a on inter1_log_probs) */
} TjClassCal;

typedef struct {
  int has[2];                   /* "2|4", "4|6" present */
  double cut[2];
} TjCuts;

/* A cuts table ("cuts" or one preset of "preset_cuts"): per madd class, present = a non-empty object. */
typedef struct {
  int present;                  /* the table itself is a non-empty object */
  int cls_present[TJ_MADD_CLASSES];
  TjCuts cls[TJ_MADD_CLASSES];
} TjCutTable;

/* qalqalah / gemination: tau, and per preset "preset_tau" */
typedef struct {
  int status;
  int has_tau;
  double tau;
  int has_ptau[ZF_TJ_PRESET_COUNT];
  double ptau[ZF_TJ_PRESET_COUNT];
} TjLetterCal;

typedef struct {
  char name[TJ_NAME_MAX];
  TjClassCal cls[ZF_TJ_CLASS_COUNT];
  /* madd */
  int madd_status;
  int lazim_status;             /* cuts.lazim.status, else off */
  TjCutTable cuts;
  TjCutTable preset_cuts[ZF_TJ_PRESET_COUNT];
  int has_display;
  double display_a2, display_a6;
  /* letter features */
  TjLetterCal qlq, gem;
  int ghunna_status, tafkheem_status;
  /* "backend_overrides": per backend ("fp32", "int8"), per class, a thresholds table that replaces the
   * class's own when present */
  int n_backends;
  char backend[TJ_MAX_BACKENDS][TJ_NAME_MAX];
  int bo_present[TJ_MAX_BACKENDS][ZF_TJ_CLASS_COUNT];
  int bo_has_thr[TJ_MAX_BACKENDS][ZF_TJ_CLASS_COUNT][ZF_TJ_PRESET_COUNT];
  double bo_thr[TJ_MAX_BACKENDS][ZF_TJ_CLASS_COUNT][ZF_TJ_PRESET_COUNT];
  /* "word" (API 3): the per-word score's level */
  int word_present;
  int word_status;
  int word_default;             /* ZF_TJ_WORD_* */
  int word_has_tau[ZF_TJ_WORD_PRESET_COUNT];
  double word_tau[ZF_TJ_WORD_PRESET_COUNT];
  double word_floor, word_lo, word_hi;
} TjContextCal;

typedef struct {
  char model_id[32];
  char default_context[TJ_NAME_MAX];
  int default_preset;
  int has_tokens_sha;
  unsigned char tokens_sha[32];
  int n_contexts;
  TjContextCal ctx[TJ_MAX_CONTEXTS];
} TjCalibration;

/* Parses calibration.json text. Returns 0, or -1 with *error set. */
int zf_tj_calibration_parse(const char *text, size_t len, TjCalibration *cal, const char **error);
const TjContextCal *zf_tj_calibration_context(const TjCalibration *cal, const char *name);

/* ---- the expected table (expected.c, FORMAT.md) -------------------------------------------------- */
enum {
  TJ_F_GEMINATE = 1u << 0,
  TJ_F_QLQ_SITE = 1u << 1,
  TJ_F_QLQ_SHIFT = 2,     /* 2 bits: 0 none, 1 sughra, 2 kubra */
  TJ_F_GHUNNA_SHIFT = 4,  /* 3 bits: attr_ghunna class */
  TJ_F_TAFKHEEM_SHIFT = 7,/* 2 bits: attr_tafkheem class */
  TJ_F_MADD_SHIFT = 9     /* 3 bits: ZF_TJ_MADD_* or TJ_MADD_CONSONANT_LENGTH */
};

typedef struct {
  int pos;
  int cls;          /* ZF_TJ_MADD_* or TJ_MADD_CONSONANT_LENGTH */
  int allowed_mask; /* bit c: count c allowed by the rule */
  int scoring_mask; /* bit c: count c accepted by the scorer */
  int canonical;
  int kind;         /* 0 fixed, 1 free */
  int madd_base;    /* the entry's base is a madd letter (tajweed_ref.MADD_BASES) */
} TjMaddEntry;

/* A word form (FORMAT.md): flags bit 0 L ibtida, bit 1 R waqf, bits 2-4 mode (plain, sukun, rawm,
 * ishmam), bit 5 cross-ayah. */
#define TJ_FORM_IBTIDA 1
#define TJ_FORM_WAQF 2
#define TJ_FORM_MODE(f) (((f) >> 2) & 7)
#define TJ_FORM_CROSS 32
#define TJ_FORM_MODE_CANONICAL 4 /* a row without a lattice: the canonical word, no path search */
typedef struct {
  int flags;
  int wajh;             /* 0 none, else a name of the string table */
  int n;
  const uint8_t *tok;
} TjForm;

typedef struct {
  unsigned char *data;  /* the whole file */
  size_t size;
  int n_surah, n_ayah;
  int *surah_first;    /* [n_surah + 1]: index of the surah's first ayah */
  /* expanded per token */
  int *tok;             /* [n_tokens] */
  int *word;            /* [n_tokens] */
  int *flags;           /* [n_tokens] */
  int n_tokens;
  /* per ayah */
  int *a_tok_off, *a_n_tok, *a_n_words, *a_madd_off, *a_n_madd, *a_forms_off;
  TjMaddEntry *madd;
  int n_madd;
  const unsigned char *forms; /* the forms section */
  size_t forms_size;
  int n_names;                /* wajh names */
  char **names;               /* [n_names + 1], names[0] NULL */
  unsigned char source_sha[32], tokens_sha[32];
} TjExpected;

int zf_tj_expected_load(const unsigned char *data, size_t size, TjExpected *e, const char **error);
void zf_tj_expected_free(TjExpected *e);
/* ayah index (0 .. n_ayah - 1) or -1 */
int zf_tj_expected_index(const TjExpected *e, int surah, int ayah);

/* ---- the model ----------------------------------------------------------------------------------- */
struct ZfTajweedModel {
  TjCalibration cal;
  const TjContextCal *ctx; /* the context in use */
  int ctx_index;
  char *token_names[ZF_TAJWEED_VOCAB];
  TjExpected exp;
  /* the exact engine (engine glue, ENG): NULL without */
  void *engine;
  int chunk_ms;
  /* the backend the heads come from, for calibration's backend_overrides: "int8" with the exact
   * engine (the engine glue may change it), "fp32" otherwise */
  char backend[TJ_NAME_MAX];
  int backend_index; /* into ctx->backend, -1 none */
  ZfTajweedOpenOptions options;
};

/* Engine glue: engine_stub.c fails both; the real glue replaces that file. */
int zf_tajweed__engine_open(ZfTajweedModel *m, const char *context_dir, const ZfTajweedOpenOptions *o,
                            const char **error);
void zf_tajweed__engine_close(ZfTajweedModel *m);

/* ---- configuration (config.c): the effective settings for the model's own context; c may be NULL
 * (calibration defaults) */
/* Items of the settings (TAJWEED_SETTINGS.md): the 8 classes, then these */
#define TJ_ITEM_MADD ZF_TJ_CLASS_COUNT
#define TJ_ITEM_LAZIM (ZF_TJ_CLASS_COUNT + 1)
#define TJ_ITEM_GEM (ZF_TJ_CLASS_COUNT + 2)
#define TJ_ITEM_QLQ (ZF_TJ_CLASS_COUNT + 3)
#define TJ_ITEM_WORD (ZF_TJ_CLASS_COUNT + 4)
#define TJ_ITEM_COUNT (ZF_TJ_CLASS_COUNT + 5)
/* every check key, the public ones and API 3's "word" */
#define TJ_CHECK_ALL (ZF_TJ_CHECK_COUNT + 1)

typedef struct {
  ZfTajweedEffective pub;
  const TjContextCal *ctx;                 /* the calibration context resolved against */
  char backend[TJ_NAME_MAX];               /* "int8" / "fp32" */
  int backend_index;                       /* into ctx->backend, -1 none */
  int class_preset[ZF_TJ_CLASS_COUNT];     /* the preset a class's threshold comes from, -1 a threshold override */
  int madd_preset, qlq_preset, gem_preset; /* presets of the madd cuts and the letter taus */
  int madd_source, qlq_source, gem_source;
  int lazim_enabled;                       /* lazim and madd both on */
  int check_value[TJ_CHECK_ALL];           /* the explicit value in force (app, else remote), -1 none */
  int refused[TJ_ITEM_COUNT];              /* 0, or the source (remote / app) whose "true" was refused */
  ZfTajweedWordSettings word;              /* API 3 */
} TjEffective;
/* The effective settings. ctx NULL: the model's own context (the scorer's); c may be NULL (defaults). */
int zf_tj_config_resolve_ctx(const ZfTajweedModel *m, const ZfTajweedConfig *c, const TjContextCal *ctx,
                             TjEffective *out);
#define zf_tj_config_resolve(m, c, out) zf_tj_config_resolve_ctx((m), (c), NULL, (out))

/* ---- small helpers ------------------------------------------------------------------------------- */
int zf_tj_status_from_name(const char *s);
const char *zf_tj_status_name(int status);
const char *zf_tj_source_name(int source);
/* Reads a whole file. Returns a malloc'd buffer (NUL-terminated past size) or NULL. */
unsigned char *zf_tj_read_file(const char *path, size_t *size);

/* A growable text buffer. */
typedef struct {
  char *p;
  size_t n, cap;
  int oom;
} TjBuf;
void tj_buf_init(TjBuf *b);
void tj_buf_free(TjBuf *b);
void tj_buf_put(TjBuf *b, const char *s, size_t n);
void tj_buf_puts(TjBuf *b, const char *s);
void tj_buf_int(TjBuf *b, long long v);
/* a JSON number (shortest of %.17g that reads back the same is not needed: %.10g), null when not finite */
void tj_buf_num(TjBuf *b, double v);
void tj_buf_str(TjBuf *b, const char *s); /* JSON string, escaped */
void tj_buf_bool(TjBuf *b, int v);

#endif /* ZF_TAJWEED_INTERNAL_H */
