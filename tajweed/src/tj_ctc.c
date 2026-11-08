/* Linear CTC for the tajweed scorer (see tj_ctc.h). The recursions and the edit rescoring are the
 * phoneme engine's (research/phoneme-engine/c/src/ctc.c: pe_ctc_forward, pe_ctc_backward,
 * pe_ctc_rescore, delete_one, pe_ctc_viterbi) with every position a single token. */
#include "tj_ctc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define NEG (-INFINITY)
#define LSE zf_tj_lse2

static double state_em(const TjCtc *c, int t, int s) {
  const double *row = c->lp + (size_t)t * c->V;
  return (s & 1) ? row[c->seq[(s - 1) / 2]] : row[c->blank];
}

/* label state s (odd, s >= 3) may be entered from s - 2 */
static int skip_ok(const TjCtc *c, int s) {
  if (s < 3 || !(s & 1)) return 0;
  const int j = (s - 1) / 2;
  return c->seq[j - 1] != c->seq[j];
}

int tj_ctc_init(TjCtc *c, const double *lp, int T, int V, int blank, const int *seq, int J) {
  memset(c, 0, sizeof *c);
  c->lp = lp;
  c->T = T;
  c->V = V;
  c->blank = blank;
  c->seq = seq;
  c->J = J;
  c->S = 2 * J + 1;
  const int S = c->S;
  c->alpha = (double *)malloc(sizeof(double) * (size_t)T * S);
  c->beta = (double *)malloc(sizeof(double) * (size_t)T * S);
  c->e = (double *)malloc(sizeof(double) * 3 * (size_t)T);
  if (!c->alpha || !c->beta || !c->e) {
    tj_ctc_free(c);
    return -1;
  }
  c->inp = c->e + T;
  c->cont = c->inp + T;
  for (int i = 0; i < 3; ++i) c->in_key[i] = c->cont_key[i] = -2;
  /* forward */
  double *alpha = c->alpha;
  for (int s = 0; s < S; ++s) alpha[s] = NEG;
  alpha[0] = lp[blank];
  if (J) alpha[1] = lp[seq[0]];
  for (int t = 1; t < T; ++t) {
    const double *p = alpha + (size_t)(t - 1) * S;
    double *a = alpha + (size_t)t * S;
    for (int s = 0; s < S; ++s) {
      double v = p[s];
      if (s >= 1) v = LSE(v, p[s - 1]);
      if (J > 1 && skip_ok(c, s)) v = LSE(v, p[s - 2]);
      a[s] = v + state_em(c, t, s);
    }
  }
  const double *last = alpha + (size_t)(T - 1) * S;
  c->logz = J ? LSE(last[S - 1], last[S - 2]) : last[0];
  /* backward */
  double *beta = c->beta;
  double *b = beta + (size_t)(T - 1) * S;
  for (int s = 0; s < S; ++s) b[s] = NEG;
  b[S - 1] = state_em(c, T - 1, S - 1);
  if (J) b[S - 2] = state_em(c, T - 1, S - 2);
  for (int t = T - 2; t >= 0; --t) {
    const double *n = beta + (size_t)(t + 1) * S;
    double *v = beta + (size_t)t * S;
    for (int s = 0; s < S; ++s) {
      double x = n[s];
      if (s + 1 < S) x = LSE(x, n[s + 1]);
      if (J > 1 && s + 2 < S && skip_ok(c, s + 2)) x = LSE(x, n[s + 2]);
      v[s] = x + state_em(c, t, s);
    }
  }
  return 0;
}

void tj_ctc_free(TjCtc *c) {
  free(c->alpha);
  free(c->beta);
  free(c->e);
  c->alpha = c->beta = c->e = NULL;
}

double tj_ctc_del(const TjCtc *c, int k) {
  const int J = c->J, S = c->S, T = c->T;
  const double *alpha = c->alpha, *beta = c->beta;
  if (k == 0) {
    double v = beta[2];
    if (J >= 2) v = LSE(v, beta[3]);
    return v;
  }
  const int skip = k + 1 <= J - 1 && c->seq[k - 1] != c->seq[k + 1];
  double v = NEG;
  int first = 1;
  for (int t = 0; t + 1 < T; ++t) {
    double cc = beta[(size_t)(t + 1) * S + 2 * k + 2];
    if (skip) cc = LSE(cc, beta[(size_t)(t + 1) * S + 2 * k + 3]);
    double x = alpha[(size_t)t * S + 2 * k - 1] + cc;
    v = first ? x : LSE(v, x);
    first = 0;
  }
  if (k == J - 1) v = LSE(v, alpha[(size_t)(T - 1) * S + 2 * k - 1]);
  return v;
}

/* pe_ctc_rescore for one substitution (ins 0) or insertion (ins 1) */
static double bridge(TjCtc *c, int ins, int j, int tok) {
  const int J = c->J, S = c->S, T = c->T;
  const double *alpha = c->alpha, *beta = c->beta;
  int pb, pl, sb, sl, prev = -1, next = -1;
  if (!ins) {
    pb = 2 * j;
    pl = j >= 1 ? 2 * j - 1 : -1;
    sb = 2 * j + 2;
    sl = j + 1 <= J - 1 ? 2 * j + 3 : -1;
    if (j >= 1) prev = j - 1;
    if (j + 1 <= J - 1) next = j + 1;
  } else {
    pb = 2 * j + 2;
    pl = j >= 0 ? 2 * j + 1 : -1;
    sb = 2 * j + 2;
    sl = j + 1 <= J - 1 ? 2 * j + 3 : -1;
    if (j >= 0) prev = j;
    if (j + 1 <= J - 1) next = j + 1;
  }
  const int skip_in = pl >= 0 && c->seq[prev] != tok;
  const int skip_out = sl >= 0 && tok != c->seq[next];
  const int start = pl < 0, end = sl < 0;
  double *inp = c->inp, *cont = c->cont, *e = c->e;
  if (c->in_key[0] != pb || c->in_key[1] != pl || c->in_key[2] != skip_in) {
    for (int t = 1; t < T; ++t) {
      double in = alpha[(size_t)(t - 1) * S + pb];
      if (skip_in) in = LSE(in, alpha[(size_t)(t - 1) * S + pl]);
      inp[t] = in;
    }
    c->in_key[0] = pb, c->in_key[1] = pl, c->in_key[2] = skip_in;
  }
  if (c->cont_key[0] != sb || c->cont_key[1] != sl || c->cont_key[2] != skip_out) {
    for (int t = 0; t < T; ++t) {
      double cc = NEG;
      if (t + 1 < T) {
        cc = beta[(size_t)(t + 1) * S + sb];
        if (skip_out) cc = LSE(cc, beta[(size_t)(t + 1) * S + sl]);
      }
      cont[t] = cc;
    }
    c->cont_key[0] = sb, c->cont_key[1] = sl, c->cont_key[2] = skip_out;
  }
  for (int t = 0; t < T; ++t) e[t] = c->lp[(size_t)t * c->V + tok];
  double x = start ? e[0] : NEG, acc = NEG;
  acc = LSE(acc, x + cont[0]);
  for (int t = 1; t < T; ++t) {
    x = LSE(x, inp[t]) + e[t];
    acc = LSE(acc, x + cont[t]);
  }
  if (end) acc = LSE(acc, x);
  return acc;
}

double tj_ctc_sub(TjCtc *c, int j, int tok) { return bridge(c, 0, j, tok); }
double tj_ctc_ins(TjCtc *c, int j, int tok) { return bridge(c, 1, j, tok); }

int tj_ctc_viterbi_frames(const TjCtc *c, int *lo, int *hi) {
  const int J = c->J, S = c->S, T = c->T;
  for (int j = 0; j < J; ++j) lo[j] = hi[j] = -1;
  signed char *bp = (signed char *)malloc((size_t)T * S);
  double *d = (double *)malloc(sizeof(double) * 2 * (size_t)S);
  int *path = (int *)malloc(sizeof(int) * (size_t)T);
  if (!bp || !d || !path) {
    free(bp);
    free(d);
    free(path);
    return -1;
  }
  double *const block = d, *nd = d + S;
  for (int s = 0; s < S; ++s) d[s] = NEG;
  d[0] = c->lp[c->blank];
  if (J) d[1] = c->lp[c->seq[0]];
  for (int s = 0; s < S; ++s) bp[s] = 0;
  for (int t = 1; t < T; ++t) {
    signed char *b = bp + (size_t)t * S;
    for (int s = 0; s < S; ++s) {
      double stay = d[s], step = s >= 1 ? d[s - 1] : NEG;
      double jump = (J > 1 && skip_ok(c, s)) ? d[s - 2] : NEG;
      double best = stay > step ? stay : step;
      if (jump > best) best = jump;
      b[s] = (signed char)(best == stay ? 0 : best == step ? 1 : 2);
      nd[s] = best + state_em(c, t, s);
    }
    double *tmp = d;
    d = nd;
    nd = tmp;
  }
  int s = (!J || d[S - 1] >= d[S - 2]) ? S - 1 : S - 2;
  int ok = isfinite(d[s]);
  if (ok)
    for (int t = T - 1; t >= 0; --t) {
      path[t] = s;
      s -= bp[(size_t)t * S + s];
    }
  if (ok)
    for (int t = 0; t < T; ++t)
      if (path[t] & 1) {
        int j = (path[t] - 1) / 2;
        if (lo[j] < 0) lo[j] = t;
        hi[j] = t;
      }
  free(block);
  free(bp);
  free(path);
  return ok ? 0 : -1;
}
