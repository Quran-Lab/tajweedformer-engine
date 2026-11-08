/* The tajweed scorer: one take of a known ayah. A port of the reference scorer\1(tajweed_ref.py, the definition of correct); each step names the function it
 * reproduces.
 *
 *   path      (Path, lattice_path) the expected sequence actually scored: per word one legal form (the
 *             row's forms, then the cross-ayah wasl forms), chosen by minimum edit distance to the greedy
 *             decode of log_probs ('|' and blanks removed), the earlier form on a tie, ending in a stop
 *             before wasl on a tie; per-token metadata (geminate, madd) through an equal-token alignment of
 *             the chosen form to the word's canonical tokens. DEPLOY rule: a long vowel followed by exactly
 *             one word-final consonant in a word read in its stop form is a free (aared) madd, 2 / 4 / 6.
 *   heads     log_probs as they are; inter1_log_probs with '|' merged into blank ('|' itself -inf). Both
 *             score the path (no '|').
 *   windows   every consonant (not the nasal) with the vowel after it, or alone when no vowel follows.
 *   alts      single edits in the window, in generation order: each token substituted by every other
 *             consonant (the 28 and the nasal) or vowel (a i u aa ii uu), then deleted, then (a consonant
 *             with no vowel after it in the window) each of the 6 vowels inserted after it.
 *   score     LLR = log P(alternative) - log P(path) per head, exact full-sequence CTC (single-position edit
 *             rescoring from one alpha / beta pass, tj_ctc.c); fused = a * final + (1 - a) * inter1, NaN
 *             -> -30, clipped to [-30, 30] (the per-head LLRs are reported clipped too).
 *   verdict   per window and class the best alternative (largest fused, the first on a tie); a class flags
 *             when enabled and fused > threshold; the window reports the flagged class with the largest
 *             fused LLR (ties: class order).
 *   H-time    chain_states / viterbi_spans on timing_log_probs over the path (pauses at word ends);
 *             madd: ratio = span / median short-vowel span (>= 3), count by the cuts of the madd preset,
 *             flagged when the count is not in the entry's scoring set (lazim only with the lazim check);
 *             qalqalah: letters b j d T q with no vowel after on the path, mean P(classes 1 + 2) over
 *             [e + 2, e + 14), flagged below tau; gemination: mean P(held) over the span, flagged below tau.
 *             Means are numpy's float32 means (pairwise sum, float32 divide). */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* sched_setaffinity: see score_worker */
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "tj_ctc.h"

enum { OP_SUB = 0, OP_DEL = 1, OP_INS = 2 };

#define T_BAR 1
#define T_CONS_LO 2
#define T_CONS_HI 29
#define T_NASAL 30
#define T_A 31
#define T_AA 34
#define T_V_HI 36
#define CLIP 30.0
#define MIN_SHORT 3 /* tajweed_ref.read_madd: len(sv) >= 3 */

static int is_short(int t) { return t >= T_A && t < T_A + 3; }
static int is_long(int t) { return t >= T_AA && t <= T_V_HI; }
static int is_vowel(int t) { return t >= T_A && t <= T_V_HI; }
static int is_cons28(int t) { return t >= T_CONS_LO && t <= T_CONS_HI; }
static int is_cons(int t) { return is_cons28(t) || t == T_NASAL; }
static int is_qlq(int t) { return t == 3 || t == 6 || t == 9 || t == 17 || t == 22; } /* b j d T q */

static const char *const MODE_NAMES[5] = {"plain", "sukun", "rawm", "ishmam", "canonical"};

typedef struct {
  int present;
  int op, p, x;           /* the edit inside the window */
  double l1, l2, fused, a; /* l1, l2 clipped as reported */
  int enabled, flagged;
  double threshold;
} TjClassVerdict;

typedef struct {
  int i, n;               /* path index, 1 or 2 tokens */
  int word;
  int cls;                /* the reported class or -1 */
  int order[ZF_TJ_CLASS_COUNT], n_order; /* classes in generation order */
  TjClassVerdict v[ZF_TJ_CLASS_COUNT];
} TjWindow;

typedef struct {
  int k;                  /* path index */
  int cls;                /* ZF_TJ_MADD_* */
  int scoring;            /* mask */
  int aared;
  int has_ratio;
  double ratio;
  int count;              /* 0 = none */
  int flagged, reported;
} TjMaddItem;

typedef struct {
  int i;
  int has_p;
  double p;
  int flagged;
} TjLetterItem;

struct ZfTajweedResult {
  /* public arrays */
  ZfTajweedSite *sites;
  int n_sites;
  ZfTajweedMadd *madd;
  int n_madd;
  ZfTajweedLetter *qlq, *gem;
  int n_qlq, n_gem;
  ZfTajweedSummary sum;
  /* what the JSON needs */
  int surah, ayah;
  char context[TJ_NAME_MAX];
  char model_id[32];
  char backend[TJ_NAME_MAX];
  TjEffective eff;
  char token_names[ZF_TAJWEED_VOCAB][24];
  int P, *path;           /* path tokens */
  int n_words;
  int *pick, *pick_flags, *pick_cross;
  char **pick_wajh;       /* borrowed from the model? no: copied */
  long edit_cost;
  int lattice;            /* 1 when the path came from the lattice */
  TjWindow *win;
  int n_win;
  TjMaddItem *mitems;
  int n_mitems, madd_on;
  double unit;
  int has_unit;
  TjLetterItem *qitems, *gitems;
  int n_qitems, n_gitems, qlq_on, gem_on;
  int has_qtau, has_gtau;
  double qtau, gtau;
  int madd_preset;
  /* API 3: the words */
  ZfTajweedWord *words;
  int *word_p;            /* each word's best edit, its position inside the word */
  int n_wordv, words_on, *pword;
  ZfTajweedWordSettings wset;
  TjBuf json;
};

/* ---- the path ----------------------------------------------------------------------------------- */

typedef struct {
  int n_words;
  int *nf;          /* forms per word */
  TjForm **f;       /* [w][i] */
  TjForm *all;
} TjForms;

static void forms_free(TjForms *F) {
  free(F->nf);
  free(F->f);
  free(F->all);
}

static int forms_load(const TjExpected *e, int a, TjForms *F) {
  const int nw = e->a_n_words[a];
  memset(F, 0, sizeof *F);
  F->n_words = nw;
  /* count forms first */
  const unsigned char *f = e->forms + e->a_forms_off[a];
  int total = 0;
  for (int w = 0; w < nw; ++w) {
    const int ns = f[1];
    f += 2;
    for (int s = 0; s < ns; ++s) f += 1 + f[0];
    total += f[0];
    f += 1 + 4 * f[0];
  }
  F->nf = (int *)malloc(sizeof(int) * (size_t)nw);
  F->f = (TjForm **)malloc(sizeof(TjForm *) * (size_t)nw);
  F->all = (TjForm *)malloc(sizeof(TjForm) * (size_t)total);
  if (!F->nf || !F->f || !F->all) return -1;
  f = e->forms + e->a_forms_off[a];
  TjForm *out = F->all;
  for (int w = 0; w < nw; ++w) {
    const int ns = f[1];
    const unsigned char *strs[256];
    f += 2;
    for (int s = 0; s < ns; ++s) {
      strs[s] = f;
      f += 1 + f[0];
    }
    const int nf = f[0];
    f += 1;
    F->nf[w] = nf;
    F->f[w] = out;
    for (int i = 0; i < nf; ++i, f += 4) {
      out->flags = f[0];
      out->n = strs[f[1]][0];
      out->tok = strs[f[1]] + 1;
      out->wajh = (int)f[2] | (int)f[3] << 8;
      out++;
    }
  }
  return 0;
}

/* score_v4_bench._lev_from: rows of tokens a (n) against hyp b (m) from row0; mat [(n + 1) * (m + 1)] */
static void lev_from(const long *row0, const uint8_t *a, int n, const int *b, int m, long *mat) {
  memcpy(mat, row0, sizeof(long) * (size_t)(m + 1));
  for (int i = 1; i <= n; ++i) {
    const long *prev = mat + (size_t)(i - 1) * (m + 1);
    long *cur = mat + (size_t)i * (m + 1);
    /* t[0] = prev[0] + 1; t[j] = min(prev[j] + 1, prev[j - 1] + sub); mat = minimum.accumulate(t - j) + j */
    long run = prev[0] + 1;
    cur[0] = run;
    for (int j = 1; j <= m; ++j) {
      long t = prev[j] + 1, d = prev[j - 1] + (b[j - 1] != a[i - 1]);
      if (d < t) t = d;
      if (t - j < run) run = t - j;
      cur[j] = run + j;
    }
  }
}

/* lattice_path: picks[w] (form index) and the cost; -1 when no path exists or out of memory */
static long lattice_path(const TjForms *F, const int *hyp, int m, int *picks) {
  const int nw = F->n_words, M = m + 1;
  int maxn = 0;
  for (int w = 0; w < nw; ++w)
    for (int i = 0; i < F->nf[w]; ++i)
      if (F->f[w][i].n > maxn) maxn = F->f[w][i].n;
  long *hrows = (long *)malloc(sizeof(long) * (size_t)(nw + 1) * 2 * M);
  int *hhave = (int *)calloc((size_t)(nw + 1) * 2, sizeof(int));
  int *choice = (int *)malloc(sizeof(int) * (size_t)nw * 2 * M);
  long *mat = (long *)malloc(sizeof(long) * (size_t)(maxn + 1) * M);
  long cost = -1;
  if (!hrows || !hhave || !choice || !mat) goto out;
  /* state 0 = "stop", 1 = "wasl"; hrows[w] = the rows entering word w */
  for (int j = 0; j < M; ++j) hrows[j] = j;
  hhave[0] = 1;
  for (int i = 0; i < F->nf[0]; ++i)
    if ((F->f[0][i].flags & TJ_FORM_CROSS) && !(F->f[0][i].flags & TJ_FORM_IBTIDA)) {
      memcpy(hrows + M, hrows, sizeof(long) * (size_t)M);
      hhave[1] = 1;
      break;
    }
  for (int w = 0; w < nw; ++w) {
    const long *rows = hrows + (size_t)w * 2 * M;
    const int *have = hhave + 2 * w;
    long *nrows = hrows + (size_t)(w + 1) * 2 * M;
    int *nhave = hhave + 2 * (w + 1);
    int *ch = choice + (size_t)w * 2 * M;
    for (int i = 0; i < F->nf[w]; ++i) {
      const TjForm *f = &F->f[w][i];
      const int src = (f->flags & TJ_FORM_IBTIDA) ? 0 : 1;
      if (!have[src]) continue;
      lev_from(rows + (size_t)src * M, f->tok, f->n, hyp, m, mat);
      const long *last = mat + (size_t)f->n * M;
      const int dst = (f->flags & TJ_FORM_WAQF) ? 0 : 1;
      long *nr = nrows + (size_t)dst * M;
      int *c = ch + (size_t)dst * M;
      if (!nhave[dst]) {
        memcpy(nr, last, sizeof(long) * (size_t)M);
        for (int j = 0; j < M; ++j) c[j] = i;
        nhave[dst] = 1;
      } else {
        for (int j = 0; j < M; ++j)
          if (last[j] < nr[j]) nr[j] = last[j], c[j] = i;
      }
    }
  }
  const long *endr = hrows + (size_t)nw * 2 * M;
  const int *endh = hhave + 2 * nw;
  int s;
  if (endh[0] && (!endh[1] || endr[m] <= endr[M + m])) s = 0;
  else if (endh[1]) s = 1;
  else goto out;
  cost = endr[(size_t)s * M + m];
  int j = m;
  for (int w = nw - 1; w >= 0; --w) {
    const int fi = choice[(size_t)w * 2 * M + (size_t)s * M + j];
    const TjForm *f = &F->f[w][fi];
    const int src = (f->flags & TJ_FORM_IBTIDA) ? 0 : 1;
    lev_from(hrows + (size_t)w * 2 * M + (size_t)src * M, f->tok, f->n, hyp, m, mat);
    int i = f->n;
    while (i > 0) {
      const long here = mat[(size_t)i * M + j];
      if (j > 0 && here == mat[(size_t)(i - 1) * M + j - 1] + (f->tok[i - 1] != hyp[j - 1])) {
        i--;
        j--;
      } else if (here == mat[(size_t)(i - 1) * M + j] + 1) {
        i--;
      } else {
        j--;
      }
    }
    picks[w] = fi;
    s = src;
  }
out:
  free(hrows);
  free(hhave);
  free(choice);
  free(mat);
  return cost;
}

/* _align_equal: a -> b over equal tokens of a Levenshtein alignment (-1 where not equal) */
static int align_equal(const uint8_t *a, int n, const int *b, int m, int *out) {
  long *D = (long *)malloc(sizeof(long) * (size_t)(n + 1) * (m + 1));
  if (!D) return -1;
  const int M = m + 1;
  for (int i = 0; i <= n; ++i) D[(size_t)i * M] = i;
  for (int j = 0; j <= m; ++j) D[j] = j;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j) {
      long v = D[(size_t)(i - 1) * M + j] + 1, h = D[(size_t)i * M + j - 1] + 1,
           d = D[(size_t)(i - 1) * M + j - 1] + (a[i - 1] != b[j - 1]);
      long x = v < h ? v : h;
      D[(size_t)i * M + j] = d < x ? d : x;
    }
  for (int i = 0; i < n; ++i) out[i] = -1;
  int i = n, j = m;
  while (i > 0 && j > 0) {
    const long here = D[(size_t)i * M + j];
    if (here == D[(size_t)(i - 1) * M + j - 1] + (a[i - 1] != b[j - 1])) {
      if (a[i - 1] == b[j - 1]) out[i - 1] = j - 1;
      i--;
      j--;
    } else if (here == D[(size_t)(i - 1) * M + j] + 1) {
      i--;
    } else {
      j--;
    }
  }
  free(D);
  return 0;
}

/* ---- the H-time chain (v4eval.chain_states + viterbi_spans) ------------------------------------ */

static int chain_states(const int *ids, const unsigned char *after, int n, int *st, int *pos) {
  int S = 0;
  st[S++] = 0;
  for (int k = 0; k < n; ++k) {
    if (st[S - 1] != 0 && k >= 1 && after[k - 1]) st[S++] = 0;
    pos[k] = S;
    st[S++] = ids[k];
  }
  if (st[S - 1] != 0) st[S++] = 0;
  return S;
}

/* spans [S][2] (start, end exclusive), -1 when the state is never on the path; 0, or -1 (no path) */
static int viterbi_spans(const float *lp, int T, int V, const int *st, int S, int *spans) {
  for (int s = 0; s < 2 * S; ++s) spans[s] = -1;
  if (T <= 0 || S <= 0) return -1;
  double *score = (double *)malloc(sizeof(double) * 2 * (size_t)S);
  signed char *back = (signed char *)calloc((size_t)T * S, 1);
  int *path = (int *)malloc(sizeof(int) * (size_t)T);
  if (!score || !back || !path) {
    free(score);
    free(back);
    free(path);
    return -2;
  }
  double *nscore = score + S;
  for (int s = 0; s < S; ++s) score[s] = -INFINITY;
  score[0] = (double)lp[st[0]];
  if (S > 1) score[1] = (double)lp[st[1]];
  for (int t = 1; t < T; ++t) {
    const float *row = lp + (size_t)t * V;
    signed char *b = back + (size_t)t * S;
    for (int s = 0; s < S; ++s) {
      /* numpy argmax over [stay, step, jump]: the first maximum */
      double best = score[s];
      int arg = 0;
      if (s >= 1 && score[s - 1] > best) best = score[s - 1], arg = 1;
      if (S > 2 && s >= 2 && st[s - 1] == 0 && score[s - 2] > best) best = score[s - 2], arg = 2;
      b[s] = (signed char)arg;
      nscore[s] = best + (double)row[st[s]];
    }
    memcpy(score, nscore, sizeof(double) * (size_t)S);
  }
  int s = S - 1;
  if (S > 1 && score[S - 2] > score[S - 1]) s = S - 2;
  int rc = -1;
  if (isfinite(score[s])) {
    for (int t = T - 1; t >= 0; --t) {
      path[t] = s;
      if (t) s -= back[(size_t)t * S + s];
    }
    int t0 = 0;
    for (int t = 1; t <= T; ++t)
      if (t == T || path[t] != path[t0]) {
        spans[2 * path[t0]] = t0;
        spans[2 * path[t0] + 1] = t;
        t0 = t;
      }
    rc = 0;
  }
  free(score);
  free(back);
  free(path);
  return rc;
}

static int cmp_int(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

/* numpy: arr[lo:hi].mean() of a float32 array */
static double f32_mean(const float *v, int n) {
  float s = zf_tj_np_sum_f32(v, n);
  return (double)(s / (float)n);
}

static double clip(double x) {
  if (x != x) return x;
  return x < -CLIP ? -CLIP : x > CLIP ? CLIP : x;
}

static int class_of(int op, int t, int x) {
  if (op == OP_SUB) {
    if (is_vowel(x) && is_vowel(t)) {
      if ((x - T_A) % 3 == (t - T_A) % 3) return is_short(t) ? ZF_TJ_SHORT_TO_LONG : ZF_TJ_LONG_TO_SHORT;
      return ZF_TJ_VOWEL_SUB;
    }
    return ZF_TJ_CONS_SUB;
  }
  if (op == OP_DEL) return is_vowel(t) ? ZF_TJ_VOWEL_DEL : ZF_TJ_CONS_DEL;
  return ZF_TJ_VOWEL_INS;
}

/* ---- the result ----------------------------------------------------------------------------------- */

void zf_tajweed_result_free(ZfTajweedResult *r) {
  if (!r) return;
  free(r->sites);
  free(r->madd);
  free(r->qlq);
  free(r->gem);
  free(r->path);
  free(r->pick);
  if (r->pick_wajh)
    for (int w = 0; w < r->n_words; ++w) free(r->pick_wajh[w]);
  free(r->pick_wajh);
  free(r->win);
  free(r->mitems);
  free(r->qitems);
  free(r->gitems);
  free(r->words);
  free(r->word_p);
  free(r->pword);
  tj_buf_free(&r->json);
  free(r);
}

/* ---- single-unit edits, both heads, computed once (sites and words read the same values) ---------- */
#define EDIT_SLOTS (ZF_TAJWEED_VOCAB + 1 + 6) /* substitution by token, deletion, insertion of a .. uu */
typedef struct {
  TjCtc *cf, *ci;
  double *v;      /* [P][EDIT_SLOTS][2] LLRs */
  uint8_t *done;  /* [P][EDIT_SLOTS] */
} TjEdits;

static int edit_slot(int op, int x) {
  return op == OP_SUB ? x : op == OP_DEL ? ZF_TAJWEED_VOCAB : ZF_TAJWEED_VOCAB + 1 + (x - T_A);
}

/* LLRs (not clipped) of one edit of path position g, on log_probs and on inter1_log_probs */
static void edit_llr(TjEdits *E, int g, int op, int x, double *l1, double *l2) {
  const size_t q = (size_t)g * EDIT_SLOTS + (size_t)edit_slot(op, x);
  double *c = E->v + q * 2;
  if (!E->done[q]) {
    double lf, li;
    if (op == OP_SUB) lf = tj_ctc_sub(E->cf, g, x), li = tj_ctc_sub(E->ci, g, x);
    else if (op == OP_DEL) lf = tj_ctc_del(E->cf, g), li = tj_ctc_del(E->ci, g);
    else lf = tj_ctc_ins(E->cf, g, x), li = tj_ctc_ins(E->ci, g, x);
    c[0] = lf - E->cf->logz;
    c[1] = li - E->ci->logz;
    E->done[q] = 1;
  }
  *l1 = c[0];
  *l2 = c[1];
}

static int lowest_bit(int m) {
  for (int c = 0; c < 8; ++c)
    if (m >> c & 1) return c;
  return 0;
}
static int highest_bit(int m) {
  for (int c = 7; c >= 0; --c)
    if (m >> c & 1) return c;
  return 0;
}

/* ---- the sites' alternatives ------------------------------------------------------------------ */

/* Every alternative of one window (consonant + vowel): subs, the deletion, the insertions, in
 * generation order, each rescored on both heads; per class the best by fused LLR. Reads the two CTCs'
 * alpha and beta and writes only this window, so windows can be scored in any order and on any thread:
 * the bridge's cached columns (TjCtc.e, inp, cont) are a thread's own copy, and a cache only saves
 * recomputing the same values, so every LLR is the same bits as the sequential loop. */
static void score_window(TjWindow *W, const int *path, int P, TjEdits *ed, const TjContextCal *x) {
  const int i = W->i;
  for (int p = 0; p < W->n; ++p) {
    const int g = i + p, t = path[g];
    const int lo_x = is_cons(t) ? T_CONS_LO : T_A, hi_x = is_cons(t) ? T_NASAL : T_V_HI;
    const int ins = is_cons(t) && (p + 1 >= W->n || !is_vowel(path[i + p + 1]));
    (void)P;
    for (int stage = 0; stage < 3; ++stage) {
      const int xa = stage == 0 ? lo_x : stage == 1 ? -1 : T_A;
      const int xb = stage == 0 ? hi_x : stage == 1 ? -1 : (ins ? T_V_HI : T_A - 1);
      for (int xx = xa; xx <= xb; ++xx) {
        if (stage == 0 && xx == t) continue;
        const int op = stage == 0 ? OP_SUB : stage == 1 ? OP_DEL : OP_INS;
        double l1, l2;
        edit_llr(ed, g, op, xx, &l1, &l2); /* this window's positions only: no other thread writes them */
        const int k = class_of(op, t, xx);
        const double av = x->cls[k].a;
        double fu = av * l1 + (1.0 - av) * l2;
        fu = fu != fu ? -CLIP : clip(fu);
        TjClassVerdict *cv = &W->v[k];
        if (!cv->present) W->order[W->n_order++] = k;
        if (!cv->present || fu > cv->fused) {
          cv->present = 1;
          cv->op = op;
          cv->p = p;
          cv->x = op == OP_DEL ? -1 : xx;
          cv->l1 = clip(l1);
          cv->l2 = clip(l2);
          cv->fused = fu;
          cv->a = av;
        }
      }
    }
  }
}

/* The windows over a few threads. Scoring an ayah is almost all bridges (two T-frame log-sum-exp scans
 * per alternative per head, about 40 alternatives a window), 10 to 330 ms on the Alcatel 1 on one
 * core while the engine's cores sit idle after the take; the windows are independent, so they are
 * handed out one at a time from a shared counter. `threads` is the model's (0: up to 4 cores, the
 * engine's default team on a phone); 1, a build without threads, or a thread that cannot be started
 * scores on the calling thread alone. Same bits whichever thread scores which window. */
#if !defined(ZF_TJ_NO_THREADS) && !defined(_MSC_VER) && (!defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__))
#define ZF_TJ_THREADS 1
#include <pthread.h>
#include <stdatomic.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sched.h>
#define ZF_TJ_AFFINITY 1
#endif
#endif

typedef struct {
  TjWindow *win;
  int n_win, P;
  const int *path;
  const TjCtc *cf, *ci;
  const TjContextCal *x;
  double *ev;      /* the shared edit cache (TjEdits.v, .done): windows write disjoint positions */
  uint8_t *edone;
  int widen; /* the caller is pinned to one core: workers may use every core (score_worker) */
#if ZF_TJ_THREADS
  atomic_int next;
#else
  int next;
#endif
} TjScoreJob;

/* a thread's view of a CTC: alpha, beta and logz shared (read only), its own bridge cache */
static int ctc_view(TjCtc *v, const TjCtc *c) {
  *v = *c;
  v->e = (double *)malloc(sizeof(double) * 3 * (size_t)(c->T > 0 ? c->T : 1));
  if (!v->e) return -1;
  v->inp = v->e + c->T;
  v->cont = v->inp + c->T;
  for (int i = 0; i < 3; ++i) v->in_key[i] = v->cont_key[i] = -2;
  return 0;
}

static void *score_worker(void *arg) {
  TjScoreJob *j = (TjScoreJob *)arg;
#if ZF_TJ_AFFINITY
  /* A thread inherits its creator's CPU mask, and the engine pins the thread that runs a stream to
   * one core (engine/src/pool.c, pin_to_core). Scoring on that thread after the take, every worker
   * landed on the same core: on the Alcatel 1 four threads scored Fatiha 1:7 no faster than one. A
   * worker made by a pinned caller therefore asks for every core; the kernel keeps it inside the
   * app's cpuset. Workers only; the caller keeps its pin. */
  if (j->widen) {
    cpu_set_t all;
    CPU_ZERO(&all);
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    for (long c = 0; c < n && c < CPU_SETSIZE; ++c) CPU_SET((int)c, &all);
    sched_setaffinity(0, sizeof all, &all);
  }
#endif
  TjCtc cf, ci;
  const int okf = ctc_view(&cf, j->cf) == 0;
  const int oki = okf && ctc_view(&ci, j->ci) == 0;
  if (!oki) { /* this thread scores nothing; the others take its share */
    if (okf) free(cf.e);
    return NULL;
  }
  for (;;) {
#if ZF_TJ_THREADS
    const int w = atomic_fetch_add(&j->next, 1);
#else
    const int w = j->next++;
#endif
    if (w >= j->n_win) break;
    TjEdits ed = {&cf, &ci, j->ev, j->edone};
    score_window(&j->win[w], j->path, j->P, &ed, j->x);
  }
  free(cf.e);
  free(ci.e);
  return NULL;
}

static int score_windows(ZfTajweedResult *r, const TjCtc *cf, const TjCtc *ci, const TjContextCal *x, int P,
                         int threads, double *ev, uint8_t *edone) {
  TjScoreJob j;
  memset(&j, 0, sizeof j);
  j.win = r->win;
  j.n_win = r->n_win;
  j.P = P;
  j.path = r->path;
  j.cf = cf;
  j.ci = ci;
  j.x = x;
  j.ev = ev;
  j.edone = edone;
#if ZF_TJ_THREADS
  atomic_init(&j.next, 0);
  if (threads <= 0) {
#if defined(_WIN32)
    threads = 4;
#else
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    threads = cpus > 4 ? 4 : cpus > 0 ? (int)cpus : 1;
#endif
  }
  if (threads > 8) threads = 8;
  if (threads > r->n_win / 4) threads = r->n_win / 4 > 0 ? r->n_win / 4 : 1; /* a short ayah: not worth a thread */
  pthread_t tid[8];
  int started = 0;
#if ZF_TJ_AFFINITY
  {
    cpu_set_t mine;
    j.widen = threads > 1 && sched_getaffinity(0, sizeof mine, &mine) == 0 && CPU_COUNT(&mine) == 1;
  }
#endif
  for (int t = 1; t < threads; ++t) {
    if (pthread_create(&tid[started], NULL, score_worker, &j)) break;
    ++started;
  }
  score_worker(&j);
  for (int t = 0; t < started; ++t) pthread_join(tid[t], NULL);
#else
  (void)threads;
  score_worker(&j);
#endif
  /* a thread without memory left its windows to the others; if every one failed, some never ran */
#if ZF_TJ_THREADS
  return atomic_load(&j.next) < j.n_win ? -1 : 0;
#else
  return j.next < j.n_win ? -1 : 0;
#endif
}

/* The two heads' forward-backward passes (about a sixth of an ayah's scoring) are independent: the
 * final head's on a second thread while the calling thread does inter1's. Each pass is the same code
 * on the same inputs either way. */
typedef struct {
  TjCtc *c;
  const double *lp;
  int T, V, J, rc;
  const int *seq;
  int widen;
} TjInitJob;

static void *ctc_init_main(void *arg) {
  TjInitJob *j = (TjInitJob *)arg;
#if ZF_TJ_AFFINITY
  if (j->widen) { /* as score_worker: not on the caller's pinned core */
    cpu_set_t all;
    CPU_ZERO(&all);
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    for (long c = 0; c < n && c < CPU_SETSIZE; ++c) CPU_SET((int)c, &all);
    sched_setaffinity(0, sizeof all, &all);
  }
#endif
  j->rc = tj_ctc_init(j->c, j->lp, j->T, j->V, ZF_TAJWEED_BLANK, j->seq, j->J);
  return NULL;
}

static int ctc_init_pair(TjCtc *cf, const double *lpf, TjCtc *ci, const double *lpi, int T, int V, const int *seq,
                         int J, int threads) {
  TjInitJob jf = {cf, lpf, T, V, J, -1, seq, 0};
#if ZF_TJ_THREADS
  if (threads != 1) {
#if ZF_TJ_AFFINITY
    cpu_set_t mine;
    jf.widen = sched_getaffinity(0, sizeof mine, &mine) == 0 && CPU_COUNT(&mine) == 1;
#endif
    pthread_t t;
    if (pthread_create(&t, NULL, ctc_init_main, &jf) == 0) {
      const int rc = tj_ctc_init(ci, lpi, T, V, ZF_TAJWEED_BLANK, seq, J);
      pthread_join(t, NULL);
      return rc || jf.rc ? -1 : 0;
    }
  }
#else
  (void)threads;
#endif
  ctc_init_main(&jf);
  if (jf.rc) return -1;
  return tj_ctc_init(ci, lpi, T, V, ZF_TAJWEED_BLANK, seq, J) ? -1 : 0;
}

ZfTajweedResult *zf_tajweed_score(const ZfTajweedModel *m, const ZfTajweedConfig *c, const ZfTajweedHeads *h,
                                  int surah, int ayah, const char **error) {
  const char *dummy;
  if (!error) error = &dummy;
  if (!m || !h) {
    *error = "model or heads is NULL";
    return NULL;
  }
  if (!h->log_probs || !h->inter1_log_probs || h->n40 <= 0 || h->vocab != ZF_TAJWEED_VOCAB) {
    *error = "heads: log_probs and inter1_log_probs [n40 > 0, 38] are needed";
    return NULL;
  }
  const int a = zf_tj_expected_index(&m->exp, surah, ayah);
  if (a < 0) {
    *error = "no such ayah in the expected table";
    return NULL;
  }
  const TjExpected *E = &m->exp;
  const TjContextCal *x = m->ctx;
  const int T = h->n40, V = ZF_TAJWEED_VOCAB;
  const int n = E->a_n_tok[a], nw = E->a_n_words[a];
  const int *row = E->tok + E->a_tok_off[a];
  const int *wid = E->word + E->a_tok_off[a];
  const int *flg = E->flags + E->a_tok_off[a];

  ZfTajweedResult *r = (ZfTajweedResult *)calloc(1, sizeof *r);
  TjForms F;
  memset(&F, 0, sizeof F);
  TjCtc cf, ci;
  memset(&cf, 0, sizeof cf);
  memset(&ci, 0, sizeof ci);
  TjEdits ed;
  memset(&ed, 0, sizeof ed);
  double *lpf = NULL, *lpi = NULL;
  int *hyp = NULL, *canon = NULL, *canon_ix = NULL, *canon_off = NULL, *src = NULL, *pword = NULL, *mp = NULL,
      *lo = NULL, *st = NULL, *pos = NULL, *spans = NULL, *sv = NULL, *madd_at = NULL;
  unsigned char *after = NULL, *gemi = NULL;
  float *buf = NULL;
  if (!r) goto oom;
  tj_buf_init(&r->json);
  r->surah = surah;
  r->ayah = ayah;
  strcpy(r->context, x->name);
  strcpy(r->model_id, m->cal.model_id);

  for (int i = 0; i < V; ++i) {
    const char *nm = m->token_names[i];
    size_t k = strlen(nm);
    if (k >= sizeof r->token_names[i]) k = sizeof r->token_names[i] - 1;
    memcpy(r->token_names[i], nm, k);
    r->token_names[i][k] = 0;
  }
  zf_tj_config_resolve(m, c, &r->eff);
  strcpy(r->backend, r->eff.backend);
  const ZfTajweedEffective *eff = &r->eff.pub;

  /* ---- the path ---- */
  if (forms_load(E, a, &F)) goto oom;
  canon = (int *)malloc(sizeof(int) * (size_t)(n + 1));
  canon_ix = (int *)malloc(sizeof(int) * (size_t)(n + 1));
  canon_off = (int *)calloc((size_t)nw + 1, sizeof(int));
  hyp = (int *)malloc(sizeof(int) * (size_t)(T + 1));
  r->pick = (int *)malloc(sizeof(int) * 3 * (size_t)nw);
  r->pick_wajh = (char **)calloc((size_t)nw, sizeof(char *));
  if (!canon || !canon_ix || !canon_off || !hyp || !r->pick || !r->pick_wajh) goto oom;
  r->n_words = nw;
  r->pick_flags = r->pick + nw;
  r->pick_cross = r->pick_flags + nw;
  {
    int k = 0;
    for (int w = 0; w < nw; ++w) {
      canon_off[w] = k;
      for (int i = 0; i < n; ++i)
        if (wid[i] == w && row[i] != T_BAR) canon[k] = row[i], canon_ix[k] = i, k++;
    }
    canon_off[nw] = k;
  }
  /* greedy decode of log_probs: argmax (the first maximum), repeats and blanks removed, then '|' */
  int mh = 0;
  {
    int prev = -1;
    for (int t = 0; t < T; ++t) {
      const float *rw = h->log_probs + (size_t)t * V;
      int best = 0;
      for (int v = 1; v < V; ++v)
        if (rw[v] > rw[best]) best = v;
      if (best != prev && best != ZF_TAJWEED_BLANK && best != T_BAR) hyp[mh++] = best;
      prev = best;
    }
  }
  int no_lattice = 0;
  for (int w = 0; w < nw; ++w)
    for (int i = 0; i < F.nf[w]; ++i) no_lattice |= TJ_FORM_MODE(F.f[w][i].flags) == TJ_FORM_MODE_CANONICAL;
  r->edit_cost = no_lattice ? -1 : lattice_path(&F, hyp, mh, r->pick);
  r->lattice = r->edit_cost >= 0;
  int P = 0;
  for (int w = 0; w < nw; ++w) P += r->lattice ? F.f[w][r->pick[w]].n : canon_off[w + 1] - canon_off[w];
  r->path = (int *)malloc(sizeof(int) * (size_t)(P + 1));
  src = (int *)malloc(sizeof(int) * (size_t)(P + 1));
  pword = (int *)malloc(sizeof(int) * (size_t)(P + 1));
  mp = (int *)malloc(sizeof(int) * 256);
  after = (unsigned char *)calloc((size_t)P + 1, 1);
  gemi = (unsigned char *)calloc((size_t)P + 1, 1);
  if (!r->path || !src || !pword || !mp || !after || !gemi) goto oom;
  r->P = P;
  {
    int k = 0;
    for (int w = 0; w < nw; ++w) {
      const int cn = canon_off[w + 1] - canon_off[w];
      const int *cw = canon + canon_off[w];
      if (r->lattice) {
        const TjForm *f = &F.f[w][r->pick[w]];
        r->pick_flags[w] = f->flags;
        r->pick_cross[w] = (f->flags & TJ_FORM_CROSS) != 0;
        if (f->wajh > 0 && f->wajh <= E->n_names) {
          size_t L = strlen(E->names[f->wajh]);
          r->pick_wajh[w] = (char *)malloc(L + 1);
          if (!r->pick_wajh[w]) goto oom;
          memcpy(r->pick_wajh[w], E->names[f->wajh], L + 1);
        }
        int same = f->n == cn;
        for (int i = 0; same && i < cn; ++i) same = f->tok[i] == cw[i];
        if (same) {
          for (int i = 0; i < cn; ++i) mp[i] = i;
        } else if (align_equal(f->tok, f->n, cw, cn, mp)) {
          goto oom;
        }
        for (int i = 0; i < f->n; ++i) {
          r->path[k] = f->tok[i];
          pword[k] = w;
          src[k] = mp[i] >= 0 ? canon_ix[canon_off[w] + mp[i]] : -1;
          k++;
        }
      } else {
        /* no lattice path (a word without usable forms): the canonical as is */
        r->pick_flags[w] = (w == 0 ? TJ_FORM_IBTIDA : 0) | (w == nw - 1 ? TJ_FORM_WAQF : 0);
        r->pick[w] = -1;
        for (int i = 0; i < cn; ++i, ++k) r->path[k] = cw[i], pword[k] = w, src[k] = canon_ix[canon_off[w] + i];
      }
      if (k > 0 && w < nw - 1) after[k - 1] = 1;
    }
  }
  for (int k = 0; k < P; ++k) gemi[k] = src[k] >= 0 && (flg[src[k]] & TJ_F_GEMINATE);

  /* ---- sites ---- */
  if (P > 0) {
    lpf = (double *)malloc(sizeof(double) * (size_t)T * V);
    lpi = (double *)malloc(sizeof(double) * (size_t)T * V);
    lo = (int *)malloc(sizeof(int) * 2 * (size_t)P);
    r->win = (TjWindow *)calloc((size_t)P, sizeof(TjWindow));
    if (!lpf || !lpi || !lo || !r->win) goto oom;
    for (size_t i = 0; i < (size_t)T * V; ++i) {
      lpf[i] = (double)h->log_probs[i];
      lpi[i] = (double)h->inter1_log_probs[i];
    }
    for (int t = 0; t < T; ++t) {
      double *rw = lpi + (size_t)t * V;
      rw[ZF_TAJWEED_BLANK] = zf_tj_lse2(rw[ZF_TAJWEED_BLANK], rw[T_BAR]);
      rw[T_BAR] = -INFINITY;
    }
    if (ctc_init_pair(&cf, lpf, &ci, lpi, T, V, r->path, P, m->options.threads)) goto oom;
    /* every edit's LLRs, computed once: the windows (on the scoring threads, each writing only its own
     * path positions) and then the words (on this thread, after the join) read the same values */
    ed.cf = &cf;
    ed.ci = &ci;
    ed.v = (double *)malloc(sizeof(double) * (size_t)P * EDIT_SLOTS * 2);
    ed.done = (uint8_t *)calloc((size_t)P * EDIT_SLOTS, 1);
    if (!ed.v || !ed.done) goto oom;
    int *hi = lo + P;
    if (tj_ctc_viterbi_frames(&cf, lo, hi))
      for (int i = 0; i < P; ++i) lo[i] = hi[i] = -1;
    int nsites = 0;
    for (int i = 0; i < P; ++i) {
      if (!is_cons28(r->path[i])) continue;
      TjWindow *W = &r->win[r->n_win++];
      W->i = i;
      W->n = (i + 1 < P && is_vowel(r->path[i + 1])) ? 2 : 1;
      W->word = pword[i];
      W->cls = -1;
    }
    if (score_windows(r, &cf, &ci, x, P, m->options.threads, ed.v, ed.done)) goto oom;
    for (int wi = 0; wi < r->n_win; ++wi) {
      TjWindow *W = &r->win[wi];
      double best = 0;
      for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
        TjClassVerdict *cv = &W->v[k];
        if (!cv->present) continue;
        cv->enabled = eff->class_enabled[k];
        cv->threshold = eff->class_threshold[k];
        cv->flagged = cv->enabled && cv->fused > cv->threshold;
        if (cv->flagged && (W->cls < 0 || cv->fused > best)) W->cls = k, best = cv->fused;
      }
      nsites += W->n_order;
    }
    r->sites = (ZfTajweedSite *)calloc((size_t)nsites + 1, sizeof(ZfTajweedSite));
    if (!r->sites) goto oom;
    for (int wi = 0; wi < r->n_win; ++wi) {
      const TjWindow *W = &r->win[wi];
      for (int q = 0; q < W->n_order; ++q) {
        const int k = W->order[q];
        const TjClassVerdict *cv = &W->v[k];
        const int g = W->i + cv->p;
        ZfTajweedSite *s = &r->sites[r->n_sites++];
        s->cls = k;
        s->status = x->cls[k].status;
        s->pos = src[g];
        s->word = W->word;
        s->expected = cv->op == OP_INS ? -1 : r->path[g];
        s->alt = cv->x;
        s->llr_final = cv->l1;
        s->llr_inter1 = cv->l2;
        s->llr = cv->fused;
        s->threshold = cv->threshold;
        s->flagged = W->cls == k;
        s->frame_lo = cv->op == OP_INS ? -1 : lo[g];
        s->frame_hi = cv->op == OP_INS ? -1 : hi[g];
      }
      r->sum.n_sites++;
      r->sum.n_flagged += W->cls >= 0;
    }
  }

  /* ---- words (API 3; tajweed_ref.word_scores): per word of the path, every single edit inside it
   * (alternatives(): per position, the substitutions, the deletion unless it empties the word, then
   * after a consonant with no vowel after it in the word the six vowel insertions), the fused LLR
   * clipped (NaN -> -30), the first maximum ---- */
  r->wset = r->eff.word;
  r->words_on = r->wset.enabled;
  if (r->words_on && P > 0) {
    r->words = (ZfTajweedWord *)calloc((size_t)nw + 1, sizeof(ZfTajweedWord));
    r->word_p = (int *)calloc((size_t)nw + 1, sizeof(int));
    r->pword = (int *)malloc(sizeof(int) * (size_t)P);
    if (!r->words || !r->word_p || !r->pword) goto oom;
    memcpy(r->pword, pword, sizeof(int) * (size_t)P);
    for (int w = 0; w < nw; ++w) {
      int g0 = -1, g1 = -1;
      for (int k = 0; k < P; ++k)
        if (pword[k] == w) {
          if (g0 < 0) g0 = k;
          g1 = k + 1;
        }
      if (g0 < 0) continue; /* no tokens on the path: no item, as the reference */
      ZfTajweedWord *wv = &r->words[r->n_wordv];
      int *bp = &r->word_p[r->n_wordv];
      r->n_wordv++;
      wv->word = w;
      wv->tok_lo = g0;
      wv->tok_hi = g1;
      wv->threshold = r->wset.threshold;
      wv->cls = -1;
      wv->edit_pos = -1;
      wv->expected = -1;
      wv->alt = -1;
      const int nt = g1 - g0;
      for (int p = 0; p < nt; ++p) {
        const int g = g0 + p, t = r->path[g];
        const int cons = is_cons(t), vow = is_vowel(t);
        const int ins = cons && (p + 1 >= nt || !is_vowel(r->path[g + 1]));
        for (int stage = 0; stage < 3; ++stage) {
          int xa, xb;
          if (stage == 0) {
            if (cons) xa = T_CONS_LO, xb = T_NASAL;
            else if (vow) xa = T_A, xb = T_V_HI;
            else continue;
          } else if (stage == 1) {
            if (nt == 1) continue; /* the deletion would empty the word */
            xa = xb = -1;
          } else {
            if (!ins) continue;
            xa = T_A, xb = T_V_HI;
          }
          for (int xx = xa; xx <= xb; ++xx) {
            if (stage == 0 && xx == t) continue;
            const int op = stage == 0 ? OP_SUB : stage == 1 ? OP_DEL : OP_INS;
            double l1, l2;
            edit_llr(&ed, g, op, xx, &l1, &l2);
            const int k = class_of(op, t, xx);
            const double av = x->cls[k].a;
            double fu = av * l1 + (1.0 - av) * l2;
            fu = fu != fu ? -CLIP : clip(fu);
            if (!wv->has_score || fu > wv->score) {
              wv->has_score = 1;
              wv->score = fu;
              wv->cls = k;
              wv->edit = op;
              wv->edit_pos = g;
              wv->expected = t;
              wv->alt = op == OP_DEL ? -1 : xx;
              wv->llr_final = clip(l1);
              wv->llr_inter1 = clip(l2);
              *bp = p;
            }
          }
        }
      }
      wv->flagged = wv->has_score && wv->score >= wv->threshold;
    }
  }

  /* ---- the H-time chain ---- */
  if (h->timing_log_probs && h->n20 > 0 && P > 0) {
    const int T2 = h->n20;
    st = (int *)malloc(sizeof(int) * (2 * (size_t)P + 2));
    pos = (int *)malloc(sizeof(int) * (size_t)P);
    spans = (int *)malloc(sizeof(int) * 2 * (2 * (size_t)P + 2));
    sv = (int *)malloc(sizeof(int) * (size_t)P);
    buf = (float *)malloc(sizeof(float) * ((size_t)T2 + 1));
    if (!st || !pos || !spans || !sv || !buf) goto oom;
    const int S = chain_states(r->path, after, P, st, pos);
    const int vr = viterbi_spans(h->timing_log_probs, T2, V, st, S, spans);
    if (vr == -2) goto oom;
    if (vr == 0) {
#define SEG0(k) spans[2 * pos[k]]
#define SEG1(k) spans[2 * pos[k] + 1]
      int nsv = 0;
      for (int k = 0; k < P; ++k)
        if (is_short(r->path[k]) && SEG0(k) >= 0) sv[nsv++] = SEG1(k) - SEG0(k);
      if (nsv >= MIN_SHORT) {
        qsort(sv, (size_t)nsv, sizeof(int), cmp_int);
        r->unit = nsv % 2 ? (double)sv[nsv / 2] : ((double)sv[nsv / 2 - 1] + (double)sv[nsv / 2]) / 2.0;
        r->has_unit = r->unit != 0.0;
        r->sum.madd_unit_ok = r->has_unit;
        r->sum.short_vowel_frames = r->unit;
      }
      /* madd */
      if (eff->check_enabled[ZF_TJ_CHECK_MADD]) {
        r->madd_on = 1;
        r->madd_preset = r->eff.madd_preset;
        madd_at = (int *)malloc(sizeof(int) * (size_t)n);
        r->mitems = (TjMaddItem *)calloc((size_t)P, sizeof(TjMaddItem));
        r->madd = (ZfTajweedMadd *)calloc((size_t)P, sizeof(ZfTajweedMadd));
        if (!madd_at || !r->mitems || !r->madd) goto oom;
        for (int i = 0; i < n; ++i) madd_at[i] = -1;
        for (int i = 0; i < E->a_n_madd[a]; ++i) madd_at[E->madd[E->a_madd_off[a] + i].pos] = E->a_madd_off[a] + i;
        const TjCutTable *ct = x->preset_cuts[r->madd_preset].present ? &x->preset_cuts[r->madd_preset] : &x->cuts;
        for (int k = 0; k < P; ++k) {
          const int t = r->path[k];
          if (!is_long(t)) continue;
          TjMaddItem it;
          memset(&it, 0, sizeof it);
          it.k = k;
          int have = 0;
          if (src[k] >= 0 && madd_at[src[k]] >= 0) {
            const TjMaddEntry *me = &E->madd[madd_at[src[k]]];
            if (me->madd_base && me->cls >= ZF_TJ_MADD_FIXED2 && me->cls <= ZF_TJ_MADD_LAZIM) {
              it.cls = me->cls;
              it.scoring = me->scoring_mask ? me->scoring_mask : me->allowed_mask;
              have = 1;
            }
          }
          /* DEPLOY: aared at a stop */
          const int w = pword[k];
          if (r->pick_flags[w] & TJ_FORM_WAQF) {
            int end = k + 1;
            while (end < P && pword[end] == w) end++;
            if (end - k == 2 && is_cons28(r->path[k + 1])) {
              if (have && it.cls == ZF_TJ_MADD_FIXED2) {
                it.cls = ZF_TJ_MADD_FREE, it.scoring = (1 << 2) | (1 << 4) | (1 << 6), it.aared = 1;
              } else if (!have) {
                it.cls = ZF_TJ_MADD_FREE, it.scoring = (1 << 2) | (1 << 4) | (1 << 6), it.aared = 1, have = 1;
              }
            }
          }
          if (!have) continue;
          it.reported = 1;
          if (SEG0(k) >= 0 && r->has_unit) {
            it.has_ratio = 1;
            it.ratio = (double)(SEG1(k) - SEG0(k)) / r->unit;
            /* madd_count: cuts.get(cls) or cuts.get("free") */
            const TjCuts *cu = ct->cls_present[it.cls]            ? &ct->cls[it.cls]
                               : ct->cls_present[ZF_TJ_MADD_FREE] ? &ct->cls[ZF_TJ_MADD_FREE]
                                                                  : NULL;
            if (cu && cu->has[0] && cu->has[1]) {
              const double c24 = cu->cut[0], c46 = cu->cut[1] <= c24 ? c24 : cu->cut[1];
              it.count = it.ratio < c24 ? 2 : it.ratio < c46 ? 4 : 6;
            }
          }
          it.flagged = it.count && !(it.scoring >> it.count & 1);
          if (it.cls == ZF_TJ_MADD_LAZIM && !r->eff.lazim_enabled) it.flagged = 0, it.reported = 0;
          r->mitems[r->n_mitems++] = it;
          ZfTajweedMadd *md = &r->madd[r->n_madd++];
          md->pos = src[k];
          md->word = w;
          md->token = t;
          md->madd_class = it.cls;
          md->status = it.cls == ZF_TJ_MADD_LAZIM ? x->lazim_status : x->madd_status;
          md->ratio = it.has_ratio ? it.ratio : NAN;
          md->count = it.count;
          if (!it.has_ratio) md->display_count = NAN;
          else if (x->has_display)
            md->display_count = it.ratio >= x->display_a2
                                    ? 2.0 + 4.0 * (it.ratio - x->display_a2) / (x->display_a6 - x->display_a2)
                                    : 2.0 * it.ratio / x->display_a2;
          else
            md->display_count = it.count;
          md->allowed_lo = lowest_bit(it.scoring);
          md->allowed_hi = highest_bit(it.scoring);
          md->flagged = it.flagged;
          md->frame_lo = SEG0(k);
          md->frame_hi = SEG0(k) >= 0 ? SEG1(k) - 1 : -1;
          r->sum.n_madd++;
          r->sum.n_madd_flagged += it.flagged;
        }
      }
      /* qalqalah and gemination */
      const int q_on = eff->check_enabled[ZF_TJ_CHECK_QALQALAH] && h->attr_qalqalah;
      const int g_on = eff->check_enabled[ZF_TJ_CHECK_GEMINATION] && h->gem;
      r->qlq_on = q_on;
      r->gem_on = g_on;
      const int qp = r->eff.qlq_preset, gp = r->eff.gem_preset;
      r->has_qtau = x->qlq.has_ptau[qp] || x->qlq.has_tau;
      r->qtau = x->qlq.has_ptau[qp] ? x->qlq.ptau[qp] : x->qlq.tau;
      r->has_gtau = x->gem.has_ptau[gp] || x->gem.has_tau;
      r->gtau = x->gem.has_ptau[gp] ? x->gem.ptau[gp] : x->gem.tau;
      if (q_on || g_on) {
        r->qitems = (TjLetterItem *)calloc((size_t)P, sizeof(TjLetterItem));
        r->gitems = (TjLetterItem *)calloc((size_t)P, sizeof(TjLetterItem));
        r->qlq = (ZfTajweedLetter *)calloc((size_t)P, sizeof(ZfTajweedLetter));
        r->gem = (ZfTajweedLetter *)calloc((size_t)P, sizeof(ZfTajweedLetter));
        if (!r->qitems || !r->gitems || !r->qlq || !r->gem) goto oom;
      }
      for (int k = 0; q_on && k < P; ++k) {
        const int t = r->path[k];
        if (!is_qlq(t) || (k + 1 < P && is_vowel(r->path[k + 1]))) continue;
        TjLetterItem *it = &r->qitems[r->n_qitems++];
        ZfTajweedLetter *q = &r->qlq[r->n_qlq++];
        it->i = k;
        q->frame_lo = q->frame_hi = -1;
        if (SEG0(k) >= 0 && SEG1(k) + 2 < T2) {
          const int b0 = SEG1(k) + 2, b1 = SEG1(k) + 14 < T2 ? SEG1(k) + 14 : T2;
          for (int tt = b0; tt < b1; ++tt) buf[tt - b0] = h->attr_qalqalah[tt * 3 + 1] + h->attr_qalqalah[tt * 3 + 2];
          it->has_p = 1;
          it->p = f32_mean(buf, b1 - b0);
          q->frame_lo = b0;
          q->frame_hi = b1 - 1;
        }
        it->flagged = it->has_p && r->has_qtau && it->p < r->qtau;
        q->pos = src[k];
        q->word = pword[k];
        q->token = t;
        q->status = x->qlq.status;
        q->score = it->has_p ? it->p : NAN;
        q->threshold = r->has_qtau ? r->qtau : NAN;
        q->flagged = it->flagged;
        r->sum.n_qalqalah++;
        r->sum.n_qalqalah_flagged += it->flagged;
      }
      for (int k = 0; g_on && k < P; ++k) {
        if (!gemi[k]) continue;
        TjLetterItem *it = &r->gitems[r->n_gitems++];
        ZfTajweedLetter *g = &r->gem[r->n_gem++];
        it->i = k;
        g->frame_lo = g->frame_hi = -1;
        if (SEG0(k) >= 0 && SEG0(k) < T2) {
          const int b0 = SEG0(k), b1 = SEG1(k) > b0 + 1 ? SEG1(k) : b0 + 1;
          for (int tt = b0; tt < b1; ++tt) buf[tt - b0] = h->gem[tt * 2 + 1];
          it->has_p = 1;
          it->p = f32_mean(buf, b1 - b0);
          g->frame_lo = b0;
          g->frame_hi = b1 - 1;
        }
        it->flagged = it->has_p && r->has_gtau && it->p < r->gtau;
        g->pos = src[k];
        g->word = pword[k];
        g->token = r->path[k];
        g->status = x->gem.status;
        g->score = it->has_p ? it->p : NAN;
        g->threshold = r->has_gtau ? r->gtau : NAN;
        g->flagged = it->flagged;
        r->sum.n_gemination++;
        r->sum.n_gemination_flagged += it->flagged;
      }
#undef SEG0
#undef SEG1
    }
  }
  goto done;
oom:
  *error = "out of memory";
  zf_tajweed_result_free(r);
  r = NULL;
done:
  forms_free(&F);
  tj_ctc_free(&cf);
  tj_ctc_free(&ci);
  free(ed.v);
  free(ed.done);
  free(lpf);
  free(lpi);
  free(hyp);
  free(canon);
  free(canon_ix);
  free(canon_off);
  free(src);
  free(pword);
  free(mp);
  free(lo);
  free(after);
  free(gemi);
  free(st);
  free(pos);
  free(spans);
  free(sv);
  free(madd_at);
  free(buf);
  return r;
}

/* ---- accessors and JSON ------------------------------------------------------------------------- */

int zf_tajweed_result_sites(const ZfTajweedResult *r, const ZfTajweedSite **sites) {
  if (!r) return -1;
  if (sites) *sites = r->sites;
  return r->n_sites;
}
int zf_tajweed_result_madd(const ZfTajweedResult *r, const ZfTajweedMadd **madd) {
  if (!r) return -1;
  if (madd) *madd = r->madd;
  return r->n_madd;
}
int zf_tajweed_result_qalqalah(const ZfTajweedResult *r, const ZfTajweedLetter **q) {
  if (!r) return -1;
  if (q) *q = r->qlq;
  return r->n_qlq;
}
int zf_tajweed_result_gemination(const ZfTajweedResult *r, const ZfTajweedLetter **g) {
  if (!r) return -1;
  if (g) *g = r->gem;
  return r->n_gem;
}
int zf_tajweed_result_words(const ZfTajweedResult *r, const ZfTajweedWord **words) {
  if (!r) return -1;
  if (words) *words = r->words;
  return r->n_wordv;
}

int zf_tajweed_result_path(const ZfTajweedResult *r, const int **tokens, const int **word_of_token) {
  if (!r) return -1;
  if (tokens) *tokens = r->path;
  if (word_of_token) *word_of_token = r->pword;
  return r->P;
}

int zf_tajweed_result_summary(const ZfTajweedResult *r, ZfTajweedSummary *out) {
  if (!r || !out) return -1;
  *out = r->sum;
  return 0;
}

static void tok_str(TjBuf *b, const ZfTajweedResult *r, int id) {
  if (id < 0 || id >= ZF_TAJWEED_VOCAB) tj_buf_puts(b, "null");
  else tj_buf_str(b, r->token_names[id]);
}

static void key_num(TjBuf *b, const char *key, double v) {
  tj_buf_puts(b, key);
  tj_buf_num(b, v);
}

/* the window after an edit, as token strings */
static void alt_window(TjBuf *b, const ZfTajweedResult *r, const TjWindow *W, const TjClassVerdict *cv) {
  tj_buf_puts(b, "[");
  int first = 1;
  for (int p = 0; p < W->n; ++p) {
    const int t = r->path[W->i + p];
    int emit[2], ne = 0;
    if (p == cv->p && cv->op == OP_SUB) emit[ne++] = cv->x;
    else if (!(p == cv->p && cv->op == OP_DEL)) emit[ne++] = t;
    if (p == cv->p && cv->op == OP_INS) emit[ne++] = cv->x;
    for (int e = 0; e < ne; ++e) {
      if (!first) tj_buf_puts(b, ",");
      tok_str(b, r, emit[e]);
      first = 0;
    }
  }
  tj_buf_puts(b, "]");
}

static void alt_label(TjBuf *b, const ZfTajweedResult *r, const TjClassVerdict *cv) {
  const char *op = cv->op == OP_SUB ? "sub" : cv->op == OP_DEL ? "del" : "ins";
  TjBuf lb;
  tj_buf_init(&lb);
  tj_buf_puts(&lb, op);
  tj_buf_int(&lb, cv->p);
  if (cv->op != OP_DEL) {
    tj_buf_puts(&lb, ":");
    tj_buf_puts(&lb, r->token_names[cv->x]);
  }
  if (lb.p) tj_buf_str(b, lb.p);
  else tj_buf_puts(b, "null");
  tj_buf_free(&lb);
}

static void letter_items(TjBuf *b, const ZfTajweedResult *r, const TjLetterItem *v, int nv, const char *pname,
                         int has_tau, double tau) {
  tj_buf_puts(b, "{\"tau\":");
  if (has_tau) tj_buf_num(b, tau);
  else tj_buf_puts(b, "null");
  tj_buf_puts(b, ",\"items\":[");
  for (int i = 0; i < nv; ++i) {
    if (i) tj_buf_puts(b, ",");
    tj_buf_puts(b, "{\"i\":");
    tj_buf_int(b, v[i].i);
    tj_buf_puts(b, ",\"pos\":");
    tj_buf_int(b, r->qlq && v == r->qitems ? r->qlq[i].pos : r->gem[i].pos);
    tj_buf_puts(b, ",\"");
    tj_buf_puts(b, pname);
    tj_buf_puts(b, "\":");
    if (v[i].has_p) tj_buf_num(b, v[i].p);
    else tj_buf_puts(b, "null");
    tj_buf_puts(b, ",\"flagged\":");
    tj_buf_bool(b, v[i].flagged);
    tj_buf_puts(b, "}");
  }
  tj_buf_puts(b, "]}");
}

const char *zf_tajweed_result_json(ZfTajweedResult *r) {
  if (!r) return NULL;
  if (r->json.n) return r->json.p;
  TjBuf *b = &r->json;
  const ZfTajweedEffective *eff = &r->eff.pub;
  tj_buf_puts(b, "{\"format\":\"quranlab.tajweed-verdicts/1\",\"model_id\":");
  tj_buf_str(b, r->model_id);
  tj_buf_puts(b, ",\"context\":");
  tj_buf_str(b, r->context);
  tj_buf_puts(b, ",\"backend\":");
  tj_buf_str(b, r->backend);
  tj_buf_puts(b, ",\"preset\":");
  tj_buf_str(b, zf_tajweed_preset_name(eff->preset));
  tj_buf_puts(b, ",\"ayah\":\"");
  tj_buf_int(b, r->surah);
  tj_buf_puts(b, ":");
  tj_buf_int(b, r->ayah);
  tj_buf_puts(b, "\",\"path\":[");
  for (int k = 0; k < r->P; ++k) {
    if (k) tj_buf_puts(b, ",");
    tok_str(b, r, r->path[k]);
  }
  tj_buf_puts(b, "],\"path_forms\":[");
  for (int w = 0; w < r->n_words; ++w) {
    const int f = r->pick_flags[w];
    if (w) tj_buf_puts(b, ",");
    tj_buf_puts(b, "{\"word\":");
    tj_buf_int(b, w);
    tj_buf_puts(b, ",\"pick\":");
    if (r->lattice) tj_buf_int(b, r->pick[w]);
    else tj_buf_puts(b, "null");
    tj_buf_puts(b, ",\"L\":");
    tj_buf_str(b, f & TJ_FORM_IBTIDA ? "ibtida" : "wasl");
    tj_buf_puts(b, ",\"R\":");
    tj_buf_str(b, f & TJ_FORM_WAQF ? "waqf" : "wasl");
    tj_buf_puts(b, ",\"mode\":");
    tj_buf_str(b, r->lattice ? MODE_NAMES[TJ_FORM_MODE(f) > 4 ? 4 : TJ_FORM_MODE(f)] : "canonical");
    tj_buf_puts(b, ",\"wajh\":");
    if (r->pick_wajh[w]) tj_buf_str(b, r->pick_wajh[w]);
    else tj_buf_puts(b, "null");
    tj_buf_puts(b, ",\"cross\":");
    tj_buf_bool(b, r->lattice && r->pick_cross[w]);
    tj_buf_puts(b, "}");
  }
  tj_buf_puts(b, "],\"path_edit_cost\":");
  if (r->lattice) tj_buf_int(b, r->edit_cost);
  else tj_buf_puts(b, "null");
  tj_buf_puts(b, ",\"sites\":[");
  int si = 0;
  for (int wi = 0; wi < r->n_win; ++wi) {
    const TjWindow *W = &r->win[wi];
    if (wi) tj_buf_puts(b, ",");
    tj_buf_puts(b, "{\"i\":");
    tj_buf_int(b, W->i);
    tj_buf_puts(b, ",\"word\":");
    tj_buf_int(b, W->word);
    tj_buf_puts(b, ",\"expected\":[");
    for (int p = 0; p < W->n; ++p) {
      if (p) tj_buf_puts(b, ",");
      tok_str(b, r, r->path[W->i + p]);
    }
    tj_buf_puts(b, "],\"flagged\":");
    tj_buf_bool(b, W->cls >= 0);
    tj_buf_puts(b, ",\"class\":");
    if (W->cls >= 0) tj_buf_str(b, zf_tajweed_class_name(W->cls));
    else tj_buf_puts(b, "null");
    if (W->cls >= 0) {
      const TjClassVerdict *cv = &W->v[W->cls];
      tj_buf_puts(b, ",\"alternative\":");
      alt_window(b, r, W, cv);
      tj_buf_puts(b, ",\"alt_label\":");
      alt_label(b, r, cv);
      key_num(b, ",\"llr_log_probs\":", cv->l1);
      key_num(b, ",\"llr_inter1\":", cv->l2);
      key_num(b, ",\"fused\":", cv->fused);
      key_num(b, ",\"threshold\":", cv->threshold);
    }
    tj_buf_puts(b, ",\"per_class\":{");
    for (int q = 0; q < W->n_order; ++q) {
      const int k = W->order[q];
      const TjClassVerdict *cv = &W->v[k];
      if (q) tj_buf_puts(b, ",");
      tj_buf_str(b, zf_tajweed_class_name(k));
      tj_buf_puts(b, ":{\"alt\":");
      alt_label(b, r, cv);
      tj_buf_puts(b, ",\"alt_window\":");
      alt_window(b, r, W, cv);
      key_num(b, ",\"llr_log_probs\":", cv->l1);
      key_num(b, ",\"llr_inter1\":", cv->l2);
      key_num(b, ",\"fused\":", cv->fused);
      key_num(b, ",\"a\":", cv->a);
      key_num(b, ",\"threshold\":", cv->threshold);
      tj_buf_puts(b, ",\"enabled\":");
      tj_buf_bool(b, cv->enabled);
      tj_buf_puts(b, ",\"flagged\":");
      tj_buf_bool(b, cv->flagged);
      tj_buf_puts(b, ",\"pos\":");
      tj_buf_int(b, r->sites[si].pos);
      tj_buf_puts(b, "}");
      si++;
    }
    tj_buf_puts(b, "}}");
  }
  tj_buf_puts(b, "]");
  if (r->madd_on) {
    tj_buf_puts(b, ",\"madd\":{\"unit_frames\":");
    if (r->has_unit) tj_buf_num(b, r->unit);
    else tj_buf_puts(b, "null");
    tj_buf_puts(b, ",\"preset\":");
    tj_buf_str(b, zf_tajweed_preset_name(r->madd_preset));
    tj_buf_puts(b, ",\"items\":[");
    for (int i = 0; i < r->n_mitems; ++i) {
      const TjMaddItem *it = &r->mitems[i];
      if (i) tj_buf_puts(b, ",");
      tj_buf_puts(b, "{\"i\":");
      tj_buf_int(b, it->k);
      tj_buf_puts(b, ",\"pos\":");
      tj_buf_int(b, r->madd[i].pos);
      tj_buf_puts(b, ",\"cls\":");
      tj_buf_str(b, zf_tajweed_madd_class_name(it->cls));
      tj_buf_puts(b, ",\"scoring\":[");
      int first = 1;
      for (int c = 0; c < 8; ++c)
        if (it->scoring >> c & 1) {
          if (!first) tj_buf_puts(b, ",");
          tj_buf_int(b, c);
          first = 0;
        }
      tj_buf_puts(b, "],\"aared_at_stop\":");
      tj_buf_bool(b, it->aared);
      tj_buf_puts(b, ",\"ratio\":");
      if (it->has_ratio) tj_buf_num(b, it->ratio);
      else tj_buf_puts(b, "null");
      tj_buf_puts(b, ",\"count\":");
      if (it->count) tj_buf_int(b, it->count);
      else tj_buf_puts(b, "null");
      tj_buf_puts(b, ",\"flagged\":");
      tj_buf_bool(b, it->flagged);
      if (!it->reported) tj_buf_puts(b, ",\"reported\":false");
      tj_buf_puts(b, "}");
    }
    tj_buf_puts(b, "]}");
  }
  if (r->qlq_on) {
    tj_buf_puts(b, ",\"qalqalah\":");
    letter_items(b, r, r->qitems, r->n_qitems, "p_burst", r->has_qtau, r->qtau);
  }
  if (r->gem_on) {
    tj_buf_puts(b, ",\"gemination\":");
    letter_items(b, r, r->gitems, r->n_gitems, "p_held", r->has_gtau, r->gtau);
  }
  if (r->words_on) { /* API 3: tajweed_ref's "words" */
    tj_buf_puts(b, ",\"words\":{\"tau\":");
    tj_buf_num(b, r->wset.threshold);
    tj_buf_puts(b, ",\"preset\":");
    tj_buf_str(b, zf_tajweed_word_preset_name(r->wset.preset));
    tj_buf_puts(b, ",\"flag_rule\":\"score >= tau\",\"items\":[");
    for (int i = 0; i < r->n_wordv; ++i) {
      const ZfTajweedWord *wv = &r->words[i];
      if (i) tj_buf_puts(b, ",");
      tj_buf_puts(b, "{\"word\":");
      tj_buf_int(b, wv->word);
      tj_buf_puts(b, ",\"tokens\":[");
      for (int g = wv->tok_lo; g < wv->tok_hi; ++g) {
        if (g > wv->tok_lo) tj_buf_puts(b, ",");
        tok_str(b, r, r->path[g]);
      }
      tj_buf_puts(b, "],\"score\":");
      if (!wv->has_score) {
        tj_buf_puts(b, "null,\"flagged\":false}");
        continue;
      }
      tj_buf_num(b, wv->score);
      TjClassVerdict cv;
      memset(&cv, 0, sizeof cv);
      cv.op = wv->edit;
      cv.p = r->word_p[i];
      cv.x = wv->alt;
      TjWindow W;
      memset(&W, 0, sizeof W);
      W.i = wv->tok_lo;
      W.n = wv->tok_hi - wv->tok_lo;
      tj_buf_puts(b, ",\"alt\":");
      alt_label(b, r, &cv);
      tj_buf_puts(b, ",\"alt_window\":");
      alt_window(b, r, &W, &cv);
      tj_buf_puts(b, ",\"class\":");
      tj_buf_str(b, zf_tajweed_class_name(wv->cls));
      key_num(b, ",\"llr_log_probs\":", wv->llr_final);
      key_num(b, ",\"llr_inter1\":", wv->llr_inter1);
      tj_buf_puts(b, ",\"flagged\":");
      tj_buf_bool(b, wv->flagged);
      tj_buf_puts(b, "}");
    }
    tj_buf_puts(b, "]}");
  }
  const ZfTajweedSummary *s = &r->sum;
  tj_buf_puts(b, ",\"summary\":{\"n_sites\":");
  tj_buf_int(b, s->n_sites);
  tj_buf_puts(b, ",\"n_flagged\":");
  tj_buf_int(b, s->n_flagged);
  tj_buf_puts(b, ",\"n_madd\":");
  tj_buf_int(b, s->n_madd);
  tj_buf_puts(b, ",\"n_madd_flagged\":");
  tj_buf_int(b, s->n_madd_flagged);
  tj_buf_puts(b, ",\"n_qalqalah\":");
  tj_buf_int(b, s->n_qalqalah);
  tj_buf_puts(b, ",\"n_qalqalah_flagged\":");
  tj_buf_int(b, s->n_qalqalah_flagged);
  tj_buf_puts(b, ",\"n_gemination\":");
  tj_buf_int(b, s->n_gemination);
  tj_buf_puts(b, ",\"n_gemination_flagged\":");
  tj_buf_int(b, s->n_gemination_flagged);
  tj_buf_puts(b, ",\"short_vowel_frames\":");
  tj_buf_num(b, s->short_vowel_frames);
  tj_buf_puts(b, ",\"madd_unit_ok\":");
  tj_buf_bool(b, s->madd_unit_ok);
  tj_buf_puts(b, "}}");
  if (b->oom) {
    tj_buf_free(b);
    return NULL;
  }
  return b->p;
}
