/* Linear CTC in log space, double precision, for the tajweed scorer: forward, backward, single-unit
 * edit rescoring and the best path. Ported from the phoneme engine's set-valued CTC
 * (research/phoneme-engine/c, include/pe/ctc.h + src/ctc.c) to single-token positions.
 *
 * A sequence is J token ids; extended states 2j = the blank before position j, 2j + 1 = position j,
 * 2J = the final blank; S = 2J + 1. A path may skip the blank between j and j + 1 only when their
 * tokens differ. Every log-sum-exp folds left in a fixed order with zf_tj_lse2 (numpy.logaddexp's
 * formula on the deterministic libm), so every build gives the same bits. */
#ifndef ZF_TJ_CTC_H
#define ZF_TJ_CTC_H

typedef struct {
  const double *lp; /* [T, V] log-probs; column `blank` is the blank */
  int T, V, blank;
  const int *seq;
  int J, S;
  double *alpha, *beta; /* [T, S] */
  double logz;
  /* the rescoring bridge's cached input / continuation columns */
  double *e, *inp, *cont; /* [T] each */
  int in_key[3], cont_key[3];
} TjCtc;

/* Runs forward and backward. Returns 0, or -1 when out of memory. c->logz is -inf when the sequence
 * cannot be emitted in T frames. */
int tj_ctc_init(TjCtc *c, const double *lp, int T, int V, int blank, const int *seq, int J);
void tj_ctc_free(TjCtc *c);

/* log-likelihood of the sequence with position j replaced by `tok` */
double tj_ctc_sub(TjCtc *c, int j, int tok);
/* ... with `tok` inserted after position j (j = -1: at the front) */
double tj_ctc_ins(TjCtc *c, int j, int tok);
/* ... with position j deleted */
double tj_ctc_del(const TjCtc *c, int j);

/* Best path: per position its first and last frame (-1 when the path never emits it). `lo`, `hi`
 * [J]. Returns 0, or -1 when out of memory or no path exists. */
int tj_ctc_viterbi_frames(const TjCtc *c, int *lo, int *hi);

#endif
