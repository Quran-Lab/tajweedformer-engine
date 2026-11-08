/* calibration.json ("quranlab.tajweed-calibration/1") into TjCalibration.
 *
 * Unknown keys are ignored (evidence, sources, labels). What the scorer needs is checked: the format,
 * the model id, the default context, every context's statuses, thresholds and cuts. A block that is
 * absent reads as status "off". */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "internal.h"
#include "tj_json.h"

#define CAL_FORMAT "quranlab.tajweed-calibration/1"

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int parse_sha(const char *s, unsigned char out[32]) {
  if (strlen(s) != 64) return -1;
  for (int i = 0; i < 32; ++i) {
    int h = hexval(s[2 * i]), l = hexval(s[2 * i + 1]);
    if (h < 0 || l < 0) return -1;
    out[i] = (unsigned char)(h * 16 + l);
  }
  return 0;
}

/* "status" of a block: absent block = off; a block without "status" = `dflt` */
static int block_status(const TjJ *blk, int dflt, const char **error) {
  if (!blk) return ZF_TJ_STATUS_OFF;
  if (blk->type != TJ_JOBJ) {
    *error = "calibration.json: a check block is not an object";
    return -1;
  }
  const TjJ *s = tj_json_get(blk, "status");
  if (!s) return dflt;
  int st = s->type == TJ_JSTR ? zf_tj_status_from_name(s->str) : -1;
  if (st < 0) *error = "calibration.json: unknown status";
  return st;
}

static int num(const TjJ *obj, const char *key, double *out) {
  const TjJ *v;
  if (!tj_json_get_type(obj, key, TJ_JNUM, &v) || !isfinite(v->num)) return 0;
  *out = v->num;
  return 1;
}

/* a cuts table: {class: {"2|4": x, "4|6": y, "status": ...}}; `status` gets each class's own status */
static int parse_cut_table(const TjJ *o, TjCutTable *t, int *status, const char **error);

static int madd_class_from_name(const char *s) {
  for (int k = 1; k < TJ_MADD_CLASSES; ++k)
    if (!strcmp(zf_tajweed_madd_class_name(k), s)) return k;
  return -1;
}

static int parse_cut_table(const TjJ *o, TjCutTable *t, int *status, const char **error) {
  memset(t, 0, sizeof *t);
  if (!o) return 0;
  if (o->type != TJ_JOBJ) {
    *error = "calibration.json: madd cuts are not objects";
    return -1;
  }
  t->present = o->len > 0;
  for (const TjJ *m = o->child; m; m = m->next) {
    const int k = madd_class_from_name(m->key);
    if (k < 0) continue;
    if (m->type != TJ_JOBJ) {
      *error = "calibration.json: madd cuts are not objects";
      return -1;
    }
    t->cls_present[k] = m->len > 0;
    t->cls[k].has[0] = num(m, "2|4", &t->cls[k].cut[0]);
    t->cls[k].has[1] = num(m, "4|6", &t->cls[k].cut[1]);
    if (status) {
      const TjJ *st = tj_json_get(m, "status");
      if (st) {
        status[k] = st->type == TJ_JSTR ? zf_tj_status_from_name(st->str) : -1;
        if (status[k] < 0) {
          *error = "calibration.json: unknown madd cut status";
          return -1;
        }
      }
    }
  }
  return 0;
}

static int parse_letter(const TjJ *blk, TjLetterCal *l, const char **error) {
  memset(l, 0, sizeof *l);
  l->status = block_status(blk, ZF_TJ_STATUS_OFF, error);
  if (l->status < 0) return -1;
  if (!blk) return 0;
  l->has_tau = num(blk, "tau", &l->tau);
  const TjJ *pt = tj_json_get(blk, "preset_tau");
  if (pt && pt->type == TJ_JOBJ)
    for (int p = 0; p < ZF_TJ_PRESET_COUNT; ++p) l->has_ptau[p] = num(pt, zf_tajweed_preset_name(p), &l->ptau[p]);
  return 0;
}

static int parse_context(const char *name, const TjJ *c, TjContextCal *x, const char **error) {
  memset(x, 0, sizeof *x);
  if (strlen(name) >= TJ_NAME_MAX || !name[0]) {
    *error = "calibration.json: a context name is empty or too long";
    return -1;
  }
  strcpy(x->name, name);
  if (c->type != TJ_JOBJ) {
    *error = "calibration.json: a context is not an object";
    return -1;
  }
  /* fusion: a per class, "default" for the rest; without it the release rule (consonants 0.5, else 0.8) */
  const TjJ *fa = tj_json_get(tj_json_get(c, "fusion"), "a");
  /* tajweed_ref.fusion_a: the class's a, else (a non-empty table) "default" or 0.8, else 0.5 / 0.8 */
  double dflt = 0.8;
  const int has_dflt = fa && fa->type == TJ_JOBJ && fa->len > 0;
  if (has_dflt) num(fa, "default", &dflt);
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
    double a;
    if (fa && num(fa, zf_tajweed_class_name(k), &a)) x->cls[k].a = a;
    else if (has_dflt) x->cls[k].a = dflt;
    else x->cls[k].a = (k == ZF_TJ_CONS_SUB || k == ZF_TJ_CONS_DEL) ? 0.5 : 0.8;
    if (!(x->cls[k].a >= 0.0 && x->cls[k].a <= 1.0)) {
      *error = "calibration.json: fusion a outside [0, 1]";
      return -1;
    }
  }
  /* sites */
  const TjJ *sites = tj_json_get(c, "sites");
  if (sites && sites->type != TJ_JOBJ) {
    *error = "calibration.json: sites is not an object";
    return -1;
  }
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
    const TjJ *s = tj_json_get(sites, zf_tajweed_class_name(k));
    TjClassCal *cc = &x->cls[k];
    cc->status = ZF_TJ_STATUS_OFF;
    if (!s) continue;
    cc->present = 1;
    cc->status = block_status(s, ZF_TJ_STATUS_OFF, error);
    if (cc->status < 0) return -1;
    const TjJ *th = tj_json_get(s, "thresholds");
    if (th && th->type == TJ_JOBJ)
      for (int p = 0; p < ZF_TJ_PRESET_COUNT; ++p) cc->has_thr[p] = num(th, zf_tajweed_preset_name(p), &cc->thr[p]);
  }
  /* backend_overrides */
  const TjJ *bo = tj_json_get(c, "backend_overrides");
  if (bo && bo->type == TJ_JOBJ)
    for (const TjJ *b = bo->child; b; b = b->next) {
      if (x->n_backends >= TJ_MAX_BACKENDS || strlen(b->key) >= TJ_NAME_MAX || b->type != TJ_JOBJ) {
        *error = "calibration.json: bad backend_overrides";
        return -1;
      }
      const int bi = x->n_backends++;
      strcpy(x->backend[bi], b->key);
      const TjJ *bs = tj_json_get(b, "sites");
      for (int k = 0; bs && bs->type == TJ_JOBJ && k < ZF_TJ_CLASS_COUNT; ++k) {
        const TjJ *th = tj_json_get(tj_json_get(bs, zf_tajweed_class_name(k)), "thresholds");
        if (!th || th->type != TJ_JOBJ || !th->len) continue;
        x->bo_present[bi][k] = 1;
        for (int p = 0; p < ZF_TJ_PRESET_COUNT; ++p)
          x->bo_has_thr[bi][k][p] = num(th, zf_tajweed_preset_name(p), &x->bo_thr[bi][k][p]);
      }
    }
  /* madd */
  const TjJ *md = tj_json_get(c, "madd");
  x->madd_status = block_status(md, ZF_TJ_STATUS_OFF, error);
  if (x->madd_status < 0) return -1;
  x->lazim_status = ZF_TJ_STATUS_OFF;
  if (md) {
    int st[TJ_MADD_CLASSES];
    for (int k = 0; k < TJ_MADD_CLASSES; ++k) st[k] = -1;
    if (parse_cut_table(tj_json_get(md, "cuts"), &x->cuts, st, error)) return -1;
    if (st[ZF_TJ_MADD_LAZIM] >= 0) x->lazim_status = st[ZF_TJ_MADD_LAZIM];
    const TjJ *pc = tj_json_get(md, "preset_cuts");
    if (pc && pc->type == TJ_JOBJ)
      for (int p = 0; p < ZF_TJ_PRESET_COUNT; ++p)
        if (parse_cut_table(tj_json_get(pc, zf_tajweed_preset_name(p)), &x->preset_cuts[p], NULL, error)) return -1;
    const TjJ *dl = tj_json_get(md, "display_ladder");
    if (!dl) dl = tj_json_get(md, "display");
    if (dl && num(dl, "A2", &x->display_a2) && num(dl, "A6", &x->display_a6) && x->display_a6 > x->display_a2 &&
        x->display_a2 > 0)
      x->has_display = 1;
  }
  /* qalqalah, gemination */
  if (parse_letter(tj_json_get(c, "qalqalah"), &x->qlq, error)) return -1;
  if (parse_letter(tj_json_get(c, "gemination"), &x->gem, error)) return -1;
  x->ghunna_status = block_status(tj_json_get(c, "ghunna"), ZF_TJ_STATUS_OFF, error);
  if (x->ghunna_status < 0) return -1;
  x->tafkheem_status = block_status(tj_json_get(c, "tafkheem"), ZF_TJ_STATUS_OFF, error);
  if (x->tafkheem_status < 0) return -1;
  /* word (API 3): status, default_preset, presets.<p>.tau, floor, custom_range; tajweed_ref.resolve's
   * defaults: floor 2.0, custom_range [floor, 12.0], default_preset general */
  const TjJ *wd = tj_json_get(c, "word");
  if (wd && wd->type == TJ_JOBJ) {
    x->word_present = 1;
    x->word_status = block_status(wd, ZF_TJ_STATUS_OFF, error);
    if (x->word_status < 0) return -1;
    x->word_floor = 2.0;
    num(wd, "floor", &x->word_floor);
    x->word_lo = x->word_floor;
    x->word_hi = 12.0;
    const TjJ *cr = tj_json_get(wd, "custom_range");
    if (cr && cr->type == TJ_JARR && cr->child && cr->child->next && cr->child->type == TJ_JNUM &&
        cr->child->next->type == TJ_JNUM) {
      x->word_lo = cr->child->num;
      x->word_hi = cr->child->next->num;
    }
    x->word_default = ZF_TJ_WORD_GENERAL;
    const TjJ *dp = tj_json_get(wd, "default_preset");
    if (dp) {
      x->word_default = dp->type == TJ_JSTR ? zf_tajweed_word_preset_from_name(dp->str) : -1;
      if (x->word_default < 0) {
        *error = "calibration.json: unknown word default_preset";
        return -1;
      }
    }
    const TjJ *ps = tj_json_get(wd, "presets");
    for (int p = 0; p < ZF_TJ_WORD_PRESET_COUNT; ++p)
      x->word_has_tau[p] = num(tj_json_get(ps, zf_tajweed_word_preset_name(p)), "tau", &x->word_tau[p]);
    if (!x->word_has_tau[x->word_default]) {
      *error = "calibration.json: the word default_preset has no tau";
      return -1;
    }
  }
  return 0;
}

int zf_tj_calibration_parse(const char *text, size_t len, TjCalibration *cal, const char **error) {
  const char *e = NULL;
  memset(cal, 0, sizeof *cal);
  TjJDoc *d = tj_json_parse(text, len, &e);
  if (!d) {
    *error = "calibration.json is not valid JSON";
    return -1;
  }
  const TjJ *r = tj_json_root(d), *v;
  int rc = -1;
  if (r->type != TJ_JOBJ) {
    *error = "calibration.json: not an object";
    goto out;
  }
  if (!tj_json_get_type(r, "format", TJ_JSTR, &v) || strcmp(v->str, CAL_FORMAT)) {
    *error = "calibration.json: format is not " CAL_FORMAT;
    goto out;
  }
  if (!tj_json_get_type(r, "model_id", TJ_JSTR, &v) || v->len >= sizeof cal->model_id || !v->len) {
    *error = "calibration.json: model_id missing";
    goto out;
  }
  strcpy(cal->model_id, v->str);
  if (tj_json_get_type(r, "tokens_sha256", TJ_JSTR, &v)) {
    if (parse_sha(v->str, cal->tokens_sha)) {
      *error = "calibration.json: tokens_sha256 is not 64 hex digits";
      goto out;
    }
    cal->has_tokens_sha = 1;
  }
  cal->default_preset = ZF_TJ_PRESET_BALANCED;
  if ((v = tj_json_get(r, "default_preset")) != NULL) {
    cal->default_preset = v->type == TJ_JSTR ? zf_tajweed_preset_from_name(v->str) : -1;
    if (cal->default_preset < 0) {
      *error = "calibration.json: unknown default_preset";
      goto out;
    }
  }
  if (tj_json_get_type(r, "classes", TJ_JARR, &v)) {
    int k = 0;
    for (const TjJ *m = v->child; m; m = m->next, ++k)
      if (k >= ZF_TJ_CLASS_COUNT || m->type != TJ_JSTR || strcmp(m->str, zf_tajweed_class_name(k))) {
        *error = "calibration.json: classes differ from the API's (zf_tajweed.h)";
        goto out;
      }
    if (k != ZF_TJ_CLASS_COUNT) {
      *error = "calibration.json: classes differ from the API's (zf_tajweed.h)";
      goto out;
    }
  }
  if (!tj_json_get_type(r, "contexts", TJ_JOBJ, &v) || !v->len) {
    *error = "calibration.json: no contexts";
    goto out;
  }
  for (const TjJ *m = v->child; m; m = m->next) {
    if (zf_tj_calibration_context(cal, m->key)) continue; /* a duplicate key: the first is kept */
    if (cal->n_contexts >= TJ_MAX_CONTEXTS) {
      *error = "calibration.json: too many contexts";
      goto out;
    }
    if (parse_context(m->key, m, &cal->ctx[cal->n_contexts], error)) goto out;
    cal->n_contexts++;
  }
  if (!tj_json_get_type(r, "default_context", TJ_JSTR, &v) || !zf_tj_calibration_context(cal, v->str)) {
    *error = "calibration.json: default_context missing or not a context";
    goto out;
  }
  strcpy(cal->default_context, v->str);
  rc = 0;
out:
  tj_json_free(d);
  return rc;
}

const TjContextCal *zf_tj_calibration_context(const TjCalibration *cal, const char *name) {
  if (!name) return NULL;
  for (int i = 0; i < cal->n_contexts; ++i)
    if (!strcmp(cal->ctx[i].name, name)) return &cal->ctx[i];
  return NULL;
}
