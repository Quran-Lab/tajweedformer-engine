/* The layered tajweed configuration: calibration.json < remote (the portal's block) <
 * app code, exactly as the shared settings rules (the reference resolver,\1aligned with tajweed_ref.resolve):
 *   - remote block: `v` must be the number 1, else the whole block is ignored ("remote.v"); preset one of
 *     the four; overrides per class (exactly one of preset / threshold, a finite number in -20..20) and
 *     per feature madd / qalqalah / gemination (preset only), other keys inside an override ignored;
 *     checks: known keys with booleans; context: one calibration.json ships; any other key dropped. An
 *     invalid field is dropped (its name listed in the JSON's "dropped"), never the block;
 *   - context and global preset: app, else remote, else calibration's defaults;
 *   - per class and feature: an override (app, then remote) beats a global preset (app, then remote),
 *     else calibration's default_preset; a class's threshold is the override's, else the backend's
 *     (backend_overrides) or the class's own threshold for that preset; none disables the class;
 *   - per item enabled: the check's flag from app, else remote; none: on when validated; false: off;
 *     true: on unless the status is off, which only the app of a debug build can turn on (anything else
 *     is listed in "refused"); lazim only with madd on;
 *   - qalqalah / gemination tau: preset_tau of their preset, else tau; madd cuts: preset_cuts of the madd
 *     preset when that table is not empty, else cuts (the scorer).
 * zf_tajweed_config_effective / _json resolve against the context asked for (app > remote > default);
 * the scorer resolves against its model's own context. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "tj_json.h"

#define OV_MADD ZF_TJ_CLASS_COUNT
#define OV_QLQ (ZF_TJ_CLASS_COUNT + 1)
#define OV_GEM (ZF_TJ_CLASS_COUNT + 2)
#define OV_COUNT (ZF_TJ_CLASS_COUNT + 3)

typedef struct {
  int preset;
  int ov_preset[OV_COUNT]; /* -1 none */
  int ov_has_thr[OV_COUNT];
  double ov_thr[OV_COUNT];
  int check[TJ_CHECK_ALL];
  int has_context;
  /* API 3: the word level, a preset (-1 none) or a custom threshold */
  int word_preset;
  int word_has_thr;
  double word_thr;
  char context[TJ_NAME_MAX];
} TjLayer;

struct ZfTajweedConfig {
  const ZfTajweedModel *m;
  TjLayer remote, app;
  int debug_build;
  char backend[TJ_NAME_MAX]; /* empty: the model's default */
  char **dropped;            /* the remote block's dropped fields, sorted, distinct */
  int n_dropped, cap_dropped;
  TjBuf json;
};

static void layer_clear(TjLayer *l) {
  memset(l, 0, sizeof *l);
  l->preset = -1;
  for (int k = 0; k < OV_COUNT; ++k) l->ov_preset[k] = -1;
  for (int k = 0; k < TJ_CHECK_ALL; ++k) l->check[k] = -1;
  l->word_preset = -1;
}

static int ov_from_name(const char *s) {
  int k = zf_tajweed_class_from_name(s);
  if (k >= 0) return k;
  if (!s) return -1;
  if (!strcmp(s, "madd")) return OV_MADD;
  if (!strcmp(s, "qalqalah")) return OV_QLQ;
  if (!strcmp(s, "gemination")) return OV_GEM;
  return -1;
}

static void dropped_clear(ZfTajweedConfig *c) {
  for (int i = 0; i < c->n_dropped; ++i) free(c->dropped[i]);
  c->n_dropped = 0;
}

/* adds "<prefix><key>" to the dropped list, sorted, distinct */
static void drop(ZfTajweedConfig *c, const char *prefix, const char *key) {
  size_t a = strlen(prefix), b = key ? strlen(key) : 0;
  char *s = (char *)malloc(a + b + 1);
  if (!s) return;
  memcpy(s, prefix, a);
  if (b) memcpy(s + a, key, b);
  s[a + b] = 0;
  int i = 0;
  while (i < c->n_dropped && strcmp(c->dropped[i], s) < 0) i++;
  if (i < c->n_dropped && !strcmp(c->dropped[i], s)) {
    free(s);
    return;
  }
  if (c->n_dropped == c->cap_dropped) {
    int nc = c->cap_dropped ? 2 * c->cap_dropped : 8;
    char **nd = (char **)realloc(c->dropped, sizeof(char *) * (size_t)nc);
    if (!nd) {
      free(s);
      return;
    }
    c->dropped = nd;
    c->cap_dropped = nc;
  }
  memmove(c->dropped + i + 1, c->dropped + i, sizeof(char *) * (size_t)(c->n_dropped - i));
  c->dropped[i] = s;
  c->n_dropped++;
}

ZfTajweedConfig *zf_tajweed_config_create(const ZfTajweedModel *m) {
  if (!m) return NULL;
  ZfTajweedConfig *c = (ZfTajweedConfig *)calloc(1, sizeof *c);
  if (!c) return NULL;
  c->m = m;
  layer_clear(&c->remote);
  layer_clear(&c->app);
  tj_buf_init(&c->json);
  return c;
}

void zf_tajweed_config_destroy(ZfTajweedConfig *c) {
  if (!c) return;
  dropped_clear(c);
  free(c->dropped);
  tj_buf_free(&c->json);
  free(c);
}

/* ---- the remote layer ---------------------------------------------------------------------------- */

static int valid_threshold(double t) { return isfinite(t) && t >= -20.0 && t <= 20.0; }

static int is_block_key(const char *k) {
  return !strcmp(k, "v") || !strcmp(k, "preset") || !strcmp(k, "overrides") || !strcmp(k, "checks") ||
         !strcmp(k, "context") || !strcmp(k, "word");
}

int zf_tajweed_config_set_remote(ZfTajweedConfig *c, const char *json, size_t len) {
  if (!c) return -1;
  layer_clear(&c->remote);
  dropped_clear(c);
  if (!json) return 0;
  const char *err;
  TjJDoc *d = tj_json_parse(json, len, &err);
  if (!d) return -1;
  const TjJ *r = tj_json_root(d), *blk = NULL;
  TjLayer L;
  layer_clear(&L);
  if (r->type != TJ_JOBJ) {
    drop(c, "remote", NULL); /* JSON but not a block */
    goto out;
  }
  /* a whole /v1/config document holds the block under "tajweed"; the block itself has its own keys */
  const TjJ *t = tj_json_get(r, "tajweed");
  if (t) {
    if (t->type == TJ_JOBJ) blk = t;
    else if (t->type != TJ_JNULL) drop(c, "remote", NULL);
  } else {
    for (const TjJ *mm = r->child; mm; mm = mm->next)
      if (is_block_key(mm->key)) blk = r;
  }
  if (!blk) goto out;
  {
    const TjJ *v = tj_json_get(blk, "v");
    if (!v || v->type != TJ_JNUM || v->num != 1.0) {
      drop(c, "remote.v", NULL); /* a newer portal never half-applies */
      goto out;
    }
  }
  for (const TjJ *f = blk->child; f; f = f->next) {
    if (!strcmp(f->key, "v")) continue;
    if (!strcmp(f->key, "preset")) {
      int p = f->type == TJ_JSTR ? zf_tajweed_preset_from_name(f->str) : -1;
      if (p < 0) drop(c, "remote.preset", NULL);
      else L.preset = p;
    } else if (!strcmp(f->key, "overrides")) {
      if (f->type != TJ_JOBJ) {
        drop(c, "remote.overrides", NULL);
        continue;
      }
      for (const TjJ *o = f->child; o; o = o->next) {
        const int k = ov_from_name(o->key);
        const TjJ *pp = o->type == TJ_JOBJ ? tj_json_get(o, "preset") : NULL;
        const TjJ *tt = o->type == TJ_JOBJ ? tj_json_get(o, "threshold") : NULL;
        int ok = k >= 0 && o->type == TJ_JOBJ && (!pp) != (!tt) && !(k >= ZF_TJ_CLASS_COUNT && tt);
        int p = -1;
        if (ok && pp) {
          p = pp->type == TJ_JSTR ? zf_tajweed_preset_from_name(pp->str) : -1;
          ok = p >= 0;
        }
        if (ok && tt) ok = tt->type == TJ_JNUM && valid_threshold(tt->num);
        if (!ok) {
          drop(c, "remote.overrides.", o->key);
          continue;
        }
        L.ov_preset[k] = pp ? p : -1;
        L.ov_has_thr[k] = tt != NULL;
        L.ov_thr[k] = tt ? tt->num : 0;
      }
    } else if (!strcmp(f->key, "checks")) {
      if (f->type != TJ_JOBJ) {
        drop(c, "remote.checks", NULL);
        continue;
      }
      for (const TjJ *x = f->child; x; x = x->next) {
        int k = zf_tajweed_check_from_name(x->key);
        if (k < 0 || (x->type != TJ_JTRUE && x->type != TJ_JFALSE)) {
          drop(c, "remote.checks.", x->key);
          continue;
        }
        L.check[k] = x->type == TJ_JTRUE;
      }
    } else if (!strcmp(f->key, "word")) {
      /* API 3: {"preset": strict | general | beginner} or {"threshold": a finite number}, exactly one (a
       * threshold is clamped to calibration's custom_range when resolved, as tajweed_ref.resolve) */
      const TjJ *pp = f->type == TJ_JOBJ ? tj_json_get(f, "preset") : NULL;
      const TjJ *tt = f->type == TJ_JOBJ ? tj_json_get(f, "threshold") : NULL;
      int ok = f->type == TJ_JOBJ && (!pp) != (!tt), p = -1;
      if (ok && pp) {
        p = pp->type == TJ_JSTR ? zf_tajweed_word_preset_from_name(pp->str) : -1;
        ok = p >= 0;
      }
      if (ok && tt) ok = tt->type == TJ_JNUM && isfinite(tt->num);
      if (!ok) {
        drop(c, "remote.word", NULL);
        continue;
      }
      L.word_preset = pp ? p : -1;
      L.word_has_thr = tt != NULL;
      L.word_thr = tt ? tt->num : 0;
    } else if (!strcmp(f->key, "context")) {
      if (f->type != TJ_JSTR || !f->len || !zf_tj_calibration_context(&c->m->cal, f->str)) {
        drop(c, "remote.context", NULL);
        continue;
      }
      L.has_context = 1;
      strcpy(L.context, f->str);
    } else {
      drop(c, "remote.", f->key);
    }
  }
  c->remote = L;
out:
  tj_json_free(d);
  return c->n_dropped;
}

/* ---- the app layer ------------------------------------------------------------------------------- */

int zf_tajweed_config_set_preset(ZfTajweedConfig *c, const char *preset) {
  if (!c) return -1;
  if (!preset) {
    c->app.preset = -1;
    return 0;
  }
  int p = zf_tajweed_preset_from_name(preset);
  if (p < 0) return -1;
  c->app.preset = p;
  return 0;
}

/* `cls` may also be "madd", "qalqalah" or "gemination" (their preset), API 2 */
int zf_tajweed_config_set_class_preset(ZfTajweedConfig *c, const char *cls, const char *preset) {
  if (!c) return -1;
  int k = ov_from_name(cls);
  if (k < 0) return -1;
  if (!preset) {
    c->app.ov_preset[k] = -1;
    c->app.ov_has_thr[k] = 0;
    return 0;
  }
  int p = zf_tajweed_preset_from_name(preset);
  if (p < 0) return -1;
  c->app.ov_preset[k] = p;
  c->app.ov_has_thr[k] = 0;
  return 0;
}

int zf_tajweed_config_set_class_threshold(ZfTajweedConfig *c, const char *cls, double threshold) {
  if (!c) return -1;
  int k = zf_tajweed_class_from_name(cls);
  if (k < 0 || !valid_threshold(threshold)) return -1;
  c->app.ov_has_thr[k] = 1;
  c->app.ov_thr[k] = threshold;
  c->app.ov_preset[k] = -1;
  return 0;
}

int zf_tajweed_config_clear_class(ZfTajweedConfig *c, const char *cls) {
  if (!c) return -1;
  int k = ov_from_name(cls);
  if (k < 0) return -1;
  c->app.ov_has_thr[k] = 0;
  c->app.ov_preset[k] = -1;
  return 0;
}

int zf_tajweed_config_set_check(ZfTajweedConfig *c, const char *check, int on) {
  if (!c) return -1;
  int k = zf_tajweed_check_from_name(check);
  if (k < 0 || on < -1 || on > 1) return -1;
  c->app.check[k] = on;
  return 0;
}

int zf_tajweed_config_set_context(ZfTajweedConfig *c, const char *context) {
  if (!c) return -1;
  if (!context) {
    c->app.has_context = 0;
    return 0;
  }
  if (!zf_tj_calibration_context(&c->m->cal, context)) return -1;
  c->app.has_context = 1;
  strcpy(c->app.context, context);
  return 0;
}

/* API 3: the word level */
int zf_tajweed_config_set_word_preset(ZfTajweedConfig *c, const char *preset) {
  if (!c) return -1;
  if (!preset) {
    c->app.word_preset = -1;
    c->app.word_has_thr = 0;
    return 0;
  }
  const int p = zf_tajweed_word_preset_from_name(preset);
  if (p < 0) return -1;
  c->app.word_preset = p;
  c->app.word_has_thr = 0;
  return 0;
}

int zf_tajweed_config_set_word_threshold(ZfTajweedConfig *c, double threshold) {
  if (!c || !isfinite(threshold)) return -1;
  c->app.word_preset = -1;
  c->app.word_has_thr = 1;
  c->app.word_thr = threshold;
  return 0;
}

int zf_tajweed_config_clear_word(ZfTajweedConfig *c) {
  if (!c) return -1;
  c->app.word_preset = -1;
  c->app.word_has_thr = 0;
  return 0;
}

int zf_tajweed_config_set_backend(ZfTajweedConfig *c, const char *backend) {
  if (!c) return -1;
  if (!backend) {
    c->backend[0] = 0;
    return 0;
  }
  if (strcmp(backend, "int8") && strcmp(backend, "fp32")) return -1;
  strcpy(c->backend, backend);
  return 0;
}

void zf_tajweed_config_clear_app(ZfTajweedConfig *c) {
  if (c) layer_clear(&c->app);
}

void zf_tajweed_config_set_debug_build(ZfTajweedConfig *c, int debug_build) {
  if (c) c->debug_build = debug_build ? 1 : 0;
}

/* ---- resolution ---------------------------------------------------------------------------------- */

static int check_status(const TjContextCal *x, int k) {
  switch (k) {
    case ZF_TJ_CHECK_MADD: return x->madd_status;
    case ZF_TJ_CHECK_GEMINATION: return x->gem.status;
    case ZF_TJ_CHECK_QALQALAH: return x->qlq.status;
    case ZF_TJ_CHECK_LAZIM: return x->lazim_status;
    default: {
      int st = ZF_TJ_STATUS_OFF;
      for (int c = 0; c < ZF_TJ_CLASS_COUNT; ++c)
        if (zf_tajweed_class_check(c) == k && x->cls[c].status > st) st = x->cls[c].status;
      return st;
    }
  }
}

/* settings_ref.resolve's enabled(item, status); *refused gets the refused layer's source */
static int item_enabled(int status, const TjEffective *E, const TjLayer *app, int check, int debug, int *refused) {
  const int v = E->check_value[check];
  if (v < 0) return status == ZF_TJ_STATUS_VALIDATED;
  if (!v) return 0;
  if (status != ZF_TJ_STATUS_OFF) return 1;
  const int from_app = app->check[check] >= 0;
  if (from_app && debug) return 1;
  *refused = from_app ? ZF_TJ_SOURCE_APP : ZF_TJ_SOURCE_REMOTE;
  return 0;
}

int zf_tj_config_resolve_ctx(const ZfTajweedModel *m, const ZfTajweedConfig *c, const TjContextCal *x,
                             TjEffective *E) {
  TjLayer none;
  layer_clear(&none);
  const TjLayer *app = c ? &c->app : &none, *rem = c ? &c->remote : &none;
  const int debug = c ? c->debug_build : 0;
  ZfTajweedEffective *out = &E->pub;
  memset(E, 0, sizeof *E);
  /* context */
  if (app->has_context) strcpy(out->context, app->context), out->context_source = ZF_TJ_SOURCE_APP;
  else if (rem->has_context) strcpy(out->context, rem->context), out->context_source = ZF_TJ_SOURCE_REMOTE;
  else strcpy(out->context, m->cal.default_context), out->context_source = ZF_TJ_SOURCE_CALIBRATION;
  if (!x) x = m->ctx;
  E->ctx = x;
  /* backend */
  strcpy(E->backend, c && c->backend[0] ? c->backend : m->backend);
  E->backend_index = -1;
  for (int i = 0; i < x->n_backends; ++i)
    if (!strcmp(x->backend[i], E->backend)) E->backend_index = i;
  /* the global preset */
  if (app->preset >= 0) out->preset = app->preset, out->preset_source = ZF_TJ_SOURCE_APP;
  else if (rem->preset >= 0) out->preset = rem->preset, out->preset_source = ZF_TJ_SOURCE_REMOTE;
  else out->preset = m->cal.default_preset, out->preset_source = ZF_TJ_SOURCE_CALIBRATION;
  /* explicit check values: app, else remote */
  for (int k = 0; k < TJ_CHECK_ALL; ++k)
    E->check_value[k] = app->check[k] >= 0 ? app->check[k] : rem->check[k];
  /* per class / feature: override (app, remote), else the global preset */
  int ovp[OV_COUNT], ovh[OV_COUNT], ovs[OV_COUNT];
  double ovt[OV_COUNT];
  for (int k = 0; k < OV_COUNT; ++k) {
    const TjLayer *L = NULL;
    ovs[k] = out->preset_source;
    if (app->ov_preset[k] >= 0 || app->ov_has_thr[k]) L = app, ovs[k] = ZF_TJ_SOURCE_APP;
    else if (rem->ov_preset[k] >= 0 || rem->ov_has_thr[k]) L = rem, ovs[k] = ZF_TJ_SOURCE_REMOTE;
    ovh[k] = L ? L->ov_has_thr[k] : 0;
    ovt[k] = L ? L->ov_thr[k] : 0;
    ovp[k] = L ? L->ov_preset[k] : out->preset;
  }
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
    const TjClassCal *cc = &x->cls[k];
    const int bi = E->backend_index;
    double thr = INFINITY;
    if (ovh[k]) {
      thr = ovt[k];
      E->class_preset[k] = -1;
    } else {
      const int p = ovp[k];
      E->class_preset[k] = p;
      if (bi >= 0 && x->bo_present[bi][k]) thr = x->bo_has_thr[bi][k][p] ? x->bo_thr[bi][k][p] : INFINITY;
      else if (cc->has_thr[p]) thr = cc->thr[p];
    }
    out->class_threshold[k] = thr;
    out->class_source[k] = ovs[k];
    out->class_status[k] = cc->status;
    const int on = item_enabled(cc->status, E, app, zf_tajweed_class_check(k), debug, &E->refused[k]);
    out->class_enabled[k] = on && isfinite(thr);
  }
  E->madd_preset = ovp[OV_MADD], E->madd_source = ovs[OV_MADD];
  E->qlq_preset = ovp[OV_QLQ], E->qlq_source = ovs[OV_QLQ];
  E->gem_preset = ovp[OV_GEM], E->gem_source = ovs[OV_GEM];
  /* checks */
  static const int ITEM_OF_CHECK[ZF_TJ_CHECK_COUNT] = {-1, -1, -1, TJ_ITEM_MADD, TJ_ITEM_GEM, TJ_ITEM_QLQ,
                                                       TJ_ITEM_LAZIM};
  for (int k = 0; k < ZF_TJ_CHECK_COUNT; ++k) {
    const int st = check_status(x, k);
    out->check_status[k] = st;
    int dummy = 0;
    const int item = ITEM_OF_CHECK[k];
    out->check_enabled[k] = item_enabled(st, E, app, k, debug, item >= 0 ? &E->refused[item] : &dummy);
    const int v = E->check_value[k];
    out->check_source[k] = v < 0 ? ZF_TJ_SOURCE_CALIBRATION
                                 : (app->check[k] >= 0 ? ZF_TJ_SOURCE_APP : ZF_TJ_SOURCE_REMOTE);
    if (v == 1 && st == ZF_TJ_STATUS_OFF && !out->check_enabled[k]) out->check_source[k] = ZF_TJ_SOURCE_CALIBRATION;
  }
  /* lazim only with madd */
  out->check_enabled[ZF_TJ_CHECK_LAZIM] = out->check_enabled[ZF_TJ_CHECK_LAZIM] && out->check_enabled[ZF_TJ_CHECK_MADD];
  E->lazim_enabled = out->check_enabled[ZF_TJ_CHECK_LAZIM];
  /* the word level (API 3), tajweed_ref.resolve: the layer's word setting, app over remote, else the
   * block's default_preset; a custom threshold clamped to custom_range; never below floor */
  {
    ZfTajweedWordSettings *w = &E->word;
    w->struct_size = sizeof *w;
    w->available = x->word_present;
    w->status = x->word_present ? x->word_status : ZF_TJ_STATUS_OFF;
    w->floor = x->word_floor;
    w->custom_lo = x->word_lo;
    w->custom_hi = x->word_hi;
    for (int p = 0; p < ZF_TJ_WORD_PRESET_COUNT; ++p)
      w->preset_tau[p] = x->word_has_tau[p] ? (x->word_tau[p] < x->word_floor ? x->word_floor : x->word_tau[p]) : NAN;
    const TjLayer *L = (app->word_preset >= 0 || app->word_has_thr) ? app
                       : (rem->word_preset >= 0 || rem->word_has_thr) ? rem : NULL;
    w->preset_source = L == app ? ZF_TJ_SOURCE_APP : L == rem ? ZF_TJ_SOURCE_REMOTE : ZF_TJ_SOURCE_CALIBRATION;
    double tau = NAN;
    if (L && L->word_has_thr) {
      tau = L->word_thr < x->word_lo ? x->word_lo : L->word_thr;
      if (tau > x->word_hi) tau = x->word_hi;
      w->preset = ZF_TJ_WORD_CUSTOM;
    } else {
      w->preset = L ? L->word_preset : x->word_default;
      if (x->word_present && x->word_has_tau[w->preset]) tau = x->word_tau[w->preset];
    }
    if (tau == tau && tau < x->word_floor) tau = x->word_floor;
    w->threshold = tau;
    const int v = E->check_value[ZF_TJ_CHECK_WORD];
    w->enabled_source = v < 0 ? ZF_TJ_SOURCE_CALIBRATION : (app->check[ZF_TJ_CHECK_WORD] >= 0 ? ZF_TJ_SOURCE_APP
                                                                                            : ZF_TJ_SOURCE_REMOTE);
    int refused = 0;
    const int on = item_enabled(w->status, E, app, ZF_TJ_CHECK_WORD, debug, &refused);
    if (x->word_present) E->refused[TJ_ITEM_WORD] = refused;
    w->enabled = x->word_present && on && tau == tau;
  }
  return 0;
}

/* the context the config asks for */
static const TjContextCal *asked_context(const ZfTajweedConfig *c) {
  const char *name = c->app.has_context ? c->app.context
                     : c->remote.has_context ? c->remote.context
                                             : c->m->cal.default_context;
  const TjContextCal *x = zf_tj_calibration_context(&c->m->cal, name);
  return x ? x : c->m->ctx;
}

int zf_tajweed_config_word(const ZfTajweedConfig *c, ZfTajweedWordSettings *out) {
  if (!c || !out || out->struct_size < offsetof(ZfTajweedWordSettings, available) + sizeof(int)) return -1;
  TjEffective e;
  zf_tj_config_resolve_ctx(c->m, c, asked_context(c), &e);
  const size_t n = out->struct_size < sizeof e.word ? out->struct_size : sizeof e.word;
  memcpy((char *)out + sizeof out->struct_size, (const char *)&e.word + sizeof out->struct_size,
         n - sizeof out->struct_size);
  return 0;
}

int zf_tajweed_config_effective(const ZfTajweedConfig *c, ZfTajweedEffective *out) {
  if (!c || !out) return -1;
  TjEffective e;
  zf_tj_config_resolve_ctx(c->m, c, asked_context(c), &e);
  *out = e.pub;
  return 0;
}

static void preset_or_null(TjBuf *b, int p) {
  if (p >= 0) tj_buf_str(b, zf_tajweed_preset_name(p));
  else tj_buf_puts(b, "null");
}

static void feature(TjBuf *b, const char *name, int status, int enabled, int preset, int source, int with_tau,
                    int has_tau, double tau) {
  tj_buf_puts(b, ",");
  tj_buf_str(b, name);
  tj_buf_puts(b, ":{\"status\":");
  tj_buf_str(b, zf_tj_status_name(status));
  tj_buf_puts(b, ",\"enabled\":");
  tj_buf_bool(b, enabled);
  if (preset >= -1) {
    tj_buf_puts(b, ",\"preset\":");
    preset_or_null(b, preset);
    tj_buf_puts(b, ",\"source\":");
    tj_buf_str(b, zf_tj_source_name(source));
  }
  if (with_tau) {
    tj_buf_puts(b, ",\"tau\":");
    if (has_tau) tj_buf_num(b, tau);
    else tj_buf_puts(b, "null");
  }
  tj_buf_puts(b, "}");
}

static const char *const ITEM_NAMES[TJ_ITEM_COUNT - ZF_TJ_CLASS_COUNT] = {"madd", "lazim", "gemination", "qalqalah",
                                                                         "word"};

static int cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

/* TAJWEED_SETTINGS.md's output: {context, backend, preset, sites{class:{status,enabled,threshold,preset,source}},
 * madd{status,enabled,preset,source}, lazim{status,enabled}, gemination{...,tau}, qalqalah{...,tau}, dropped[],
 * refused[]}, plus checks{} (each check's enabled, status, source), debug_build and model_context */
const char *zf_tajweed_config_json(ZfTajweedConfig *c) {
  if (!c) return NULL;
  TjEffective E;
  zf_tj_config_resolve_ctx(c->m, c, asked_context(c), &E);
  const ZfTajweedEffective *e = &E.pub;
  const TjContextCal *x = E.ctx;
  TjBuf *b = &c->json;
  b->n = 0;
  b->oom = 0;
  tj_buf_puts(b, "{\"context\":");
  tj_buf_str(b, e->context);
  tj_buf_puts(b, ",\"backend\":");
  tj_buf_str(b, E.backend);
  tj_buf_puts(b, ",\"preset\":");
  tj_buf_str(b, zf_tajweed_preset_name(e->preset));
  tj_buf_puts(b, ",\"sites\":{");
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
    if (k) tj_buf_puts(b, ",");
    tj_buf_str(b, zf_tajweed_class_name(k));
    tj_buf_puts(b, ":{\"status\":");
    tj_buf_str(b, zf_tj_status_name(e->class_status[k]));
    tj_buf_puts(b, ",\"enabled\":");
    tj_buf_bool(b, e->class_enabled[k]);
    tj_buf_puts(b, ",\"threshold\":");
    tj_buf_num(b, e->class_threshold[k]);
    tj_buf_puts(b, ",\"preset\":");
    preset_or_null(b, E.class_preset[k]);
    tj_buf_puts(b, ",\"source\":");
    tj_buf_str(b, zf_tj_source_name(e->class_source[k]));
    tj_buf_puts(b, "}");
  }
  tj_buf_puts(b, "}");
  feature(b, "madd", x->madd_status, e->check_enabled[ZF_TJ_CHECK_MADD], E.madd_preset, E.madd_source, 0, 0, 0);
  feature(b, "lazim", x->lazim_status, E.lazim_enabled, -2, 0, 0, 0, 0);
  {
    const int p = E.gem_preset;
    feature(b, "gemination", x->gem.status, e->check_enabled[ZF_TJ_CHECK_GEMINATION], p, E.gem_source, 1,
            x->gem.has_ptau[p] || x->gem.has_tau, x->gem.has_ptau[p] ? x->gem.ptau[p] : x->gem.tau);
  }
  {
    const int p = E.qlq_preset;
    feature(b, "qalqalah", x->qlq.status, e->check_enabled[ZF_TJ_CHECK_QALQALAH], p, E.qlq_source, 1,
            x->qlq.has_ptau[p] || x->qlq.has_tau, x->qlq.has_ptau[p] ? x->qlq.ptau[p] : x->qlq.tau);
  }
  if (x->word_present) { /* API 3 */
    const ZfTajweedWordSettings *w = &E.word;
    tj_buf_puts(b, ",\"word\":{\"status\":");
    tj_buf_str(b, zf_tj_status_name(w->status));
    tj_buf_puts(b, ",\"enabled\":");
    tj_buf_bool(b, w->enabled);
    tj_buf_puts(b, ",\"preset\":");
    tj_buf_str(b, zf_tajweed_word_preset_name(w->preset));
    tj_buf_puts(b, ",\"source\":");
    tj_buf_str(b, zf_tj_source_name(w->preset_source));
    tj_buf_puts(b, ",\"tau\":");
    tj_buf_num(b, w->threshold);
    tj_buf_puts(b, "}");
  }
  tj_buf_puts(b, ",\"dropped\":[");
  for (int i = 0; i < c->n_dropped; ++i) {
    if (i) tj_buf_puts(b, ",");
    tj_buf_str(b, c->dropped[i]);
  }
  tj_buf_puts(b, "],\"refused\":[");
  {
    char names[TJ_ITEM_COUNT][48];
    const char *ptrs[TJ_ITEM_COUNT];
    int n = 0;
    for (int k = 0; k < TJ_ITEM_COUNT; ++k) {
      if (!E.refused[k]) continue;
      const char *item = k < ZF_TJ_CLASS_COUNT ? zf_tajweed_class_name(k) : ITEM_NAMES[k - ZF_TJ_CLASS_COUNT];
      strcpy(names[n], E.refused[k] == ZF_TJ_SOURCE_APP ? "app:" : "remote:");
      strcat(names[n], item);
      ptrs[n] = names[n];
      n++;
    }
    qsort(ptrs, (size_t)n, sizeof(char *), cmp_str);
    for (int i = 0; i < n; ++i) {
      if (i) tj_buf_puts(b, ",");
      tj_buf_str(b, ptrs[i]);
    }
  }
  tj_buf_puts(b, "],\"checks\":{");
  for (int k = 0; k < ZF_TJ_CHECK_COUNT; ++k) {
    if (k) tj_buf_puts(b, ",");
    tj_buf_str(b, zf_tajweed_check_name(k));
    tj_buf_puts(b, ":{\"enabled\":");
    tj_buf_bool(b, e->check_enabled[k]);
    tj_buf_puts(b, ",\"status\":");
    tj_buf_str(b, zf_tj_status_name(e->check_status[k]));
    tj_buf_puts(b, ",\"source\":");
    tj_buf_str(b, zf_tj_source_name(e->check_source[k]));
    tj_buf_puts(b, "}");
  }
  tj_buf_puts(b, "},\"debug_build\":");
  tj_buf_bool(b, c->debug_build);
  tj_buf_puts(b, ",\"model_context\":");
  tj_buf_str(b, c->m->ctx->name);
  tj_buf_puts(b, "}");
  return b->oom ? NULL : b->p;
}
