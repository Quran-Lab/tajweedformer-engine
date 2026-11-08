/* Several streams at once, each on its own thread: every stream must produce exactly what one stream
 * alone produces. Each thread creates its own stream, decodes every clip of the list with the same
 * push pattern as zf_stream_check, and writes <out>.<i>.jsonl in its format, so each output can be
 * checked against the conformance set on its own.
 *   zf_stream_concurrent <model.onnx> <list.txt> <threads per stream> <streams> <out prefix> [batch] [no_pin]
 * no_pin 1 sets ZfStreamOptions.no_pin (a server's many streams); stderr gets the wall time and the
 * combined real-time factor (feature frames x 10 ms over the wall time, every stream together). */
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zf/stream.h"

typedef struct {
  const char *model, *list, *out;
  int threads, index, failed;
  uint64_t hash;
  int batch, no_pin;
  long long frames;
} Job;

static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static void on_chunk(void *user, const float *lp, int frames, int vocab) {
  Job *j = (Job *)user;
  j->hash = fnv1a(j->hash, lp, (size_t)frames * vocab * sizeof(float));
}

static void *run(void *arg) {
  Job *j = (Job *)arg;
  ZfStreamOptions opt = zf_stream_default_options();
  opt.on_chunk = on_chunk;
  opt.user = j;
  opt.threads = j->threads;
  opt.batch = j->batch;
  opt.no_pin = j->no_pin;
  const char *error = NULL;
  ZfStream *s = zf_stream_create(j->model, &opt, &error);
  if (!s) {
    fprintf(stderr, "stream %d: %s\n", j->index, error ? error : "?");
    j->failed = 1;
    return NULL;
  }
  char path[1024];
  snprintf(path, sizeof path, "%s.%d.jsonl", j->out, j->index);
  FILE *list = fopen(j->list, "r"), *out = fopen(path, "w");
  if (!list || !out) {
    j->failed = 1;
    return NULL;
  }
  static const int pattern[] = {1, 7, 13, 3, 29};
  ZfToken toks[4096];
  char line[4096];
  while (fgets(line, sizeof line, list)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) line[--len] = 0;
    if (!len) continue;
    FILE *f = fopen(line, "rb");
    if (!f) { j->failed = 1; break; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *feats = (float *)malloc((size_t)bytes);
    if (!feats || fread(feats, 1, (size_t)bytes, f) != (size_t)bytes) { j->failed = 1; fclose(f); free(feats); break; }
    fclose(f);
    zf_stream_reset(s);
    j->hash = 1469598103934665603ull;
    const int frames = (int)(bytes / (80 * 4));
    j->frames += frames;
    int n_tok = 0, k = 0;
    for (int at = 0; at < frames;) {
      int n = pattern[(k + j->index) % 5]; /* each stream pushes in a different rhythm */
      ++k;
      if (n > frames - at) n = frames - at;
      const int got = zf_stream_push(s, feats + (size_t)at * 80, n, toks + n_tok, 4096 - n_tok);
      if (got < 0) { j->failed = 1; break; }
      n_tok += got;
      at += n;
    }
    n_tok += zf_stream_finish(s, toks + n_tok, 4096 - n_tok);
    fprintf(out, "{\"features\": \"%s\", \"chunks\": %lld, \"tokens\": [", line, zf_stream_chunks(s));
    for (int t = 0; t < n_tok; ++t) fprintf(out, t ? ", %d" : "%d", toks[t].token);
    fprintf(out, "], \"logp_fnv\": \"%016llx\"}\n", (unsigned long long)j->hash);
    free(feats);
  }
  fclose(list);
  fclose(out);
  zf_stream_destroy(s);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <model.onnx> <list.txt> <threads per stream> <streams> <out prefix> [batch]\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[4]);
  Job *jobs = (Job *)calloc((size_t)n, sizeof(Job));
  pthread_t *tid = (pthread_t *)calloc((size_t)n, sizeof(pthread_t));
  for (int i = 0; i < n; ++i) {
    jobs[i] = (Job){argv[1], argv[2], argv[5], atoi(argv[3]), i, 0, 0, argc > 6 ? atoi(argv[6]) : 1,
                    argc > 7 ? atoi(argv[7]) : 0, 0};
  }
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < n; ++i) pthread_create(&tid[i], NULL, run, &jobs[i]);
  int failed = 0;
  for (int i = 0; i < n; ++i) {
    pthread_join(tid[i], NULL);
    failed |= jobs[i].failed;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  const double wall = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
  long long frames = 0;
  for (int i = 0; i < n; ++i) frames += jobs[i].frames;
  fprintf(stderr, "{\"streams\": %d, \"wall_s\": %.2f, \"audio_s\": %.1f, \"x_realtime\": %.1f}\n", n, wall,
          (double)frames / 100.0, wall > 0 ? (double)frames / 100.0 / wall : 0.0);
  free(jobs);
  free(tid);
  return failed ? 2 : 0;
}
