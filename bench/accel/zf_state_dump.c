/* Every carried state of a stream after every chunk, in the ONNX model's own names, written in zf_run's
 * ZF_DUMP layout (int32 name length, the name, int32 byte count, the bytes; one file a chunk), so the
 * engine's caches can be diffed against ONNX Runtime's state outputs tensor by tensor. The chunk's
 * log-probs go in as "log_probs".
 *   zf_state_dump <model.onnx> <features.f32> <chunk_ms> <chunks> <out dir> [threads] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zf/stream.h"

static void rec(FILE *f, const char *name, const void *p, size_t nb) {
  const int32_t n = (int32_t)strlen(name), b = (int32_t)nb;
  fwrite(&n, 4, 1, f);
  fwrite(name, 1, (size_t)n, f);
  fwrite(&b, 4, 1, f);
  fwrite(p, 1, nb, f);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <model.onnx> <features.f32> <chunk_ms> <chunks> <out dir> [threads]\n", argv[0]);
    return 1;
  }
  ZfStreamOptions o = zf_stream_default_options();
  o.chunk_ms = atoi(argv[3]);
  o.threads = argc > 6 ? atoi(argv[6]) : 1;
  const int want = atoi(argv[4]);
  const char *err = NULL;
  ZfStream *s = zf_stream_create(argv[1], &o, &err);
  if (!s) { fprintf(stderr, "stream: %s\n", err ? err : "?"); return 2; }
  FILE *ff = fopen(argv[2], "rb");
  if (!ff) return 3;
  fseek(ff, 0, SEEK_END);
  const long bytes = ftell(ff);
  fseek(ff, 0, SEEK_SET);
  float *f = (float *)malloc((size_t)bytes);
  if (fread(f, 1, (size_t)bytes, ff) != (size_t)bytes) return 3;
  fclose(ff);
  const int frames = (int)(bytes / 320), ns = zf_stream_state_count(s);
  static ZfToken out[4096];
  long long done = 0;
  for (int at = 0; at < frames && done < want; ++at) {
    if (zf_stream_push(s, f + (size_t)at * 80, 1, out, 4096) < 0) return 4;
    if (zf_stream_chunks(s) == done) continue;
    done = zf_stream_chunks(s);
    char p[1024];
    snprintf(p, sizeof p, "%s/chunk_%03lld.bin", argv[5], done - 1);
    FILE *df = fopen(p, "wb");
    if (!df) return 5;
    int lf = 0, lv = 0;
    const float *lp = zf_stream_last_log_probs(s, &lf, &lv);
    if (lp) rec(df, "log_probs", lp, (size_t)lf * lv * sizeof(float));
    for (int i = 0; i < ns; ++i) {
      const char *name; int64_t dims[8]; int rank, is64;
      zf_stream_state_info(s, i, &name, dims, &rank, &is64);
      size_t n = 1;
      for (int d = 0; d < rank; ++d) n *= (size_t)dims[d];
      const size_t nb = n * (is64 ? 8 : 4);
      void *buf = malloc(nb);
      if (zf_stream_export_state(s, name, buf, nb)) { fprintf(stderr, "state %s failed\n", name); return 6; }
      rec(df, name, buf, nb);
      free(buf);
    }
    fclose(df);
  }
  printf("%lld chunks dumped, %d states\n", done, ns);
  zf_stream_destroy(s);
  free(f);
  return 0;
}
