/* zf_tajweed_check: the V4 model folder through zf_tajweed.h, as an SDK runs it.
 *
 * Each listed clip (16 kHz float32 .pcm) is pushed as audio in 20 ms pieces, finished (the flush), and
 * every head of every chunk is fingerprinted from the stream's chunk callback, in the layout of
 * zf_set and zf_stream_check (tokens, chunks, logp_fnv, out_fnv per output per chunk), so
 * bench/v4_conformance_check.py checks it against the same expected file as the stream harness's
 * audio path. It then checks that zf_tajweed_stream_heads hands back the same bits as the chunks did.
 *
 * Timing: "chunk_ms" per clip is the wall time of the pushes and the finish over the chunks run.
 *
 * --heads all (the default here, so every output is checked) | default | <mask>: which heads the stream
 * computes (zf_tajweed_stream_set_heads); the heads not computed print as FNV 0, which
 * bench/v4_conformance_check.py --subset checks. Every clip also checks the decoded layer: the
 * top-1 tokens of zf_tajweed_stream_top2, collapsed as CTC (repeats, then blanks), must be the
 * greedy transcript, and runner-up and margin must agree with the log-probs.
 *
 *   zf_tajweed_check <model_dir> <context> <list.txt> <out.jsonl> [threads] [--heads H]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zf_tajweed.h"

#define N_OUT 12
static const char *const kName[N_OUT] = {"attr_face", "attr_ghunna", "attr_qalqalah", "attr_tafkheem",
                                         "boundary",  "conf",        "gem",           "id_log_probs",
                                         "inter1_log_probs", "inter4_log_probs", "log_probs",
                                         "timing_log_probs"};

static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
#define FNV0 1469598103934665603ull

/* a head's pointer and size, by the name above */
static const float *head(const ZfTajweedHeads *h, int k, size_t *count) {
  const size_t f40 = (size_t)h->n40, f20 = (size_t)h->n20;
  switch (k) {
    case 0: *count = f20 * 4; return h->attr_face;
    case 1: *count = f20 * 5; return h->attr_ghunna;
    case 2: *count = f20 * 3; return h->attr_qalqalah;
    case 3: *count = f20 * 3; return h->attr_tafkheem;
    case 4: *count = f20; return h->boundary;
    case 5: *count = f20; return h->conf;
    case 6: *count = f20 * 2; return h->gem;
    case 7: *count = f20 * 38; return h->id_log_probs;
    case 8: *count = f40 * 38; return h->inter1_log_probs;
    case 9: *count = f40 * 38; return h->inter4_log_probs;
    case 10: *count = f40 * 38; return h->log_probs;
    default: *count = f20 * 38; return h->timing_log_probs;
  }
}

static long g_topk_heads; /* heads ranked to full depth and checked */
static uint64_t *g_fnv;
static int g_chunks, g_cap;
static uint64_t g_logp;
static void on_chunk(void *user, const ZfTajweedHeads *h) {
  (void)user;
  if ((g_chunks + 1) * N_OUT > g_cap) {
    g_cap = 2 * (g_chunks + 1) * N_OUT;
    g_fnv = (uint64_t *)realloc(g_fnv, (size_t)g_cap * sizeof(uint64_t));
    if (!g_fnv) exit(3);
  }
  for (int k = 0; k < N_OUT; ++k) {
    size_t n = 0;
    const float *p = head(h, k, &n);
    g_fnv[(size_t)g_chunks * N_OUT + k] = p ? fnv1a(FNV0, p, n * sizeof(float)) : 0;
    if (k == 10 && p) g_logp = fnv1a(g_logp, p, n * sizeof(float));
  }
  ++g_chunks;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model_dir> <context> <list.txt> <out.jsonl> [threads] [--heads H]\n", argv[0]);
    return 1;
  }
  unsigned heads = ZF_TAJWEED_HEADS_ALL;
  ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
  o.context = argv[2];
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--heads") && i + 1 < argc) {
      ++i;
      heads = !strcmp(argv[i], "default") ? ZF_TAJWEED_HEADS_DEFAULT
              : !strcmp(argv[i], "all")   ? ZF_TAJWEED_HEADS_ALL
                                          : (unsigned)strtoul(argv[i], NULL, 0);
    }
    else o.threads = atoi(argv[i]);
  }
  const char *err = NULL;
  ZfTajweedModel *m = zf_tajweed_model_open(argv[1], &o, &err);
  if (!m) { fprintf(stderr, "model: %s\n", err ? err : "?"); return 2; }
  ZfTajweedStream *s = zf_tajweed_stream_create(m, &err);
  if (!s) { fprintf(stderr, "stream: %s\n", err ? err : "?"); return 2; }
  zf_tajweed_stream_set_callback(s, on_chunk, NULL);
  FILE *list = fopen(argv[3], "r"), *out = fopen(argv[4], "w");
  if (!list || !out) { perror("files"); return 1; }
  char line[4096];
  int bad_heads = 0, bad_top2 = 0, bad_topk = 0;
  double total_ms = 0, total_audio_ms = 0;
  while (fgets(line, sizeof line, list)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) line[--len] = 0;
    if (!len) continue;
    FILE *f = fopen(line, "rb");
    if (!f) { perror(line); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *pcm = (float *)malloc((size_t)bytes);
    if (!pcm || fread(pcm, 1, (size_t)bytes, f) != (size_t)bytes) { perror("read"); return 1; }
    fclose(f);
    const int n = (int)(bytes / 4);
    zf_tajweed_stream_reset(s);
    if (zf_tajweed_stream_set_heads(s, heads)) { fprintf(stderr, "set_heads refused\n"); return 2; }
    g_chunks = 0;
    g_logp = FNV0;
    const double t0 = now_ms();
    for (int at = 0; at < n; at += 320) {
      const int k = n - at < 320 ? n - at : 320;
      if (zf_tajweed_stream_push_audio(s, pcm + at, k) < 0) {
        fprintf(stderr, "push: %s\n", zf_tajweed_stream_last_error(s));
        return 3;
      }
    }
    if (zf_tajweed_stream_finish(s) < 0) { fprintf(stderr, "finish: %s\n", zf_tajweed_stream_last_error(s)); return 3; }
    total_ms += now_ms() - t0;
    total_audio_ms += n / 16.0;
    /* the whole take must be the chunks' bits, concatenated */
    ZfTajweedHeads h;
    zf_tajweed_stream_heads(s, &h);
    for (int k = 0; k < N_OUT; ++k) {
      size_t cnt = 0;
      const float *p = head(&h, k, &cnt);
      const size_t per = cnt / (size_t)(g_chunks ? g_chunks : 1);
      for (int c = 0; p && c < g_chunks; ++c)
        bad_heads += fnv1a(FNV0, p + (size_t)c * per, per * sizeof(float)) != g_fnv[(size_t)c * N_OUT + k];
    }
    int ids[4096];
    const int nt = zf_tajweed_stream_greedy(s, ids, 4096);
    { /* the decoded layer against the greedy transcript and the log-probs it came from */
      const ZfTajweedTop2 *t2 = NULL;
      const int nf = zf_tajweed_stream_top2(s, &t2);
      int col = 0, prev = -1;
      for (int f = 0; f < nf; ++f) {
        const float *row = h.log_probs + (size_t)f * 38;
        if (t2[f].log_prob != row[t2[f].token] || t2[f].runner_up_log_prob != row[t2[f].runner_up] ||
            t2[f].runner_up == t2[f].token || t2[f].margin < 0)
          ++bad_top2;
        for (int v = 0; v < 38; ++v)
          if (v != t2[f].token && row[v] > t2[f].runner_up_log_prob) { ++bad_top2; break; }
        if (t2[f].token != prev && t2[f].token != 0) {
          if (col >= nt || col >= 4096 || ids[col] != t2[f].token) ++bad_top2;
          ++col;
        }
        prev = t2[f].token;
      }
      if (nf != h.n40 || col != nt) ++bad_top2;
    }
    { /* top-K: k = 2 is the top-2 above, field for field; k = cols is a full stable argsort of
       * the row (insertion sort by value, highest first, equal values by the lower id), for every head
       * the stream keeps that has a ranking */
      static const unsigned kRanked[] = {ZF_TAJWEED_HEAD_LOG_PROBS, ZF_TAJWEED_HEAD_INTER1, ZF_TAJWEED_HEAD_INTER4,
                                         ZF_TAJWEED_HEAD_ID,        ZF_TAJWEED_HEAD_TIMING, ZF_TAJWEED_HEAD_GEM,
                                         ZF_TAJWEED_HEAD_TAFKHEEM,  ZF_TAJWEED_HEAD_GHUNNA, ZF_TAJWEED_HEAD_QALQALAH,
                                         ZF_TAJWEED_HEAD_FACE};
      const ZfTajweedTop2 *t2 = NULL;
      const int nf = zf_tajweed_stream_top2(s, &t2);
      if (nf > 0) {
        int *id2 = (int *)malloc(sizeof(int) * 2 * (size_t)nf);
        float *v2 = (float *)malloc(sizeof(float) * 2 * (size_t)nf), *m2 = (float *)malloc(sizeof(float) * 2 * (size_t)nf);
        if (zf_tajweed_stream_topk(s, ZF_TAJWEED_HEAD_LOG_PROBS, 2, id2, v2, m2, nf) != nf) ++bad_topk;
        else
          for (int f = 0; f < nf; ++f)
            bad_topk += id2[2 * f] != t2[f].token || id2[2 * f + 1] != t2[f].runner_up ||
                        v2[2 * f] != t2[f].log_prob || v2[2 * f + 1] != t2[f].runner_up_log_prob ||
                        m2[2 * f] != t2[f].margin;
        free(id2), free(v2), free(m2);
      }
      for (size_t q = 0; q < sizeof kRanked / sizeof *kRanked; ++q) {
        if (!(zf_tajweed_stream_get_heads(s) & kRanked[q])) {
          int dummy;
          if (zf_tajweed_stream_topk(s, kRanked[q], 1, &dummy, NULL, NULL, 0) != -1) ++bad_topk; /* not kept */
          continue;
        }
        int ms = 0, cols = 0;
        zf_tajweed_head_info(kRanked[q], &ms, &cols);
        const int nfr = ms == 40 ? h.n40 : h.n20;
        const float *vals = NULL;
        size_t cnt = 0;
        for (int k = 0; k < N_OUT; ++k)
          if (!strcmp(kName[k], zf_tajweed_head_info(kRanked[q], NULL, NULL))) vals = head(&h, k, &cnt);
        int *ik = (int *)malloc(sizeof(int) * (size_t)cols * (size_t)nfr);
        float *vk = (float *)malloc(sizeof(float) * (size_t)cols * (size_t)nfr);
        float *mk = (float *)malloc(sizeof(float) * (size_t)cols * (size_t)nfr);
        if (!vals || zf_tajweed_stream_topk(s, kRanked[q], cols, ik, vk, mk, nfr) != nfr) {
          ++bad_topk;
        } else {
          for (int f = 0; f < nfr; ++f) {
            const float *row = vals + (size_t)f * cols;
            int ord[64];
            for (int v = 0; v < cols; ++v) { /* stable insertion: a later id goes after an equal value */
              int j = v;
              while (j > 0 && row[ord[j - 1]] < row[v]) { ord[j] = ord[j - 1]; --j; }
              ord[j] = v;
            }
            for (int j = 0; j < cols; ++j)
              bad_topk += ik[(size_t)f * cols + j] != ord[j] || vk[(size_t)f * cols + j] != row[ord[j]] ||
                          mk[(size_t)f * cols + j] != (j + 1 < cols ? row[ord[j]] - row[ord[j + 1]] : 0.0f);
          }
          ++g_topk_heads;
        }
        free(ik), free(vk), free(mk);
      }
    }
    fprintf(out, "{\"features\": \"%s\", \"chunks\": %d, \"tokens\": [", line, g_chunks);
    for (int t = 0; t < nt && t < 4096; ++t) fprintf(out, t ? ", %d" : "%d", ids[t]);
    fprintf(out, "], \"logp_fnv\": \"%016llx\", \"out_fnv\": {", (unsigned long long)g_logp);
    for (int k = 0; k < N_OUT; ++k) {
      fprintf(out, "%s\"%s\": [", k ? ", " : "", kName[k]);
      for (int c = 0; c < g_chunks; ++c)
        fprintf(out, c ? ", \"%016llx\"" : "\"%016llx\"", (unsigned long long)g_fnv[(size_t)c * N_OUT + k]);
      fprintf(out, "]");
    }
    fprintf(out, "}}\n");
    free(pcm);
  }
  fclose(list);
  fclose(out);
  zf_tajweed_stream_destroy(s);
  zf_tajweed_model_close(m);
  fprintf(stderr, "{\"rtf\": %.4f, \"heads_mismatch\": %d, \"top2_mismatch\": %d, \"heads\": \"0x%03x\", "
                  "\"topk_mismatch\": %d, \"topk_heads_checked\": %ld}\n",
          total_audio_ms > 0 ? total_ms / total_audio_ms : 0, bad_heads, bad_top2, heads, bad_topk, g_topk_heads);
  return bad_heads || bad_top2 || bad_topk ? 4 : 0;
}
