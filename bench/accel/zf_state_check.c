/* A stream handed over mid-utterance must continue exactly as if nothing happened. For every clip:
 * one stream runs it whole (the reference); a second runs its first k chunks and exports every
 * carried state; a third is reset, imports them all, and takes the frames from the next window on.
 * The two halves' phonemes (ids and frame numbers) and log-prob fingerprints must equal the
 * reference's. With --dump DIR the first clip's states after chunk k are written as raw files named
 * by state, for comparing their layout against the ONNX model's own caches.
 *   zf_state_check <model.onnx> <list of .f32 feature files> <threads> <k> [--dump DIR] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zf/stream.h"

typedef struct { uint64_t hash; } Hash;

static void on_chunk(void *user, const float *lp, int frames, int vocab) {
  Hash *h = (Hash *)user;
  const unsigned char *p = (const unsigned char *)lp;
  for (size_t i = 0; i < (size_t)frames * vocab * sizeof(float); ++i) h->hash = (h->hash ^ p[i]) * 1099511628211ull;
}

static int push_all(ZfStream *s, const float *f, int from, int to, ZfToken *out, int cap) {
  int n = 0;
  for (int at = from; at < to; at += 7) {
    const int m = to - at < 7 ? to - at : 7;
    const int got = zf_stream_push(s, f + (size_t)at * 80, m, out + n, cap - n);
    if (got < 0) return -1;
    n += got;
  }
  return n;
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s <model.onnx> <list> <threads> <k> [--dump DIR]\n", argv[0]); return 1; }
  const int k = atoi(argv[4]);
  const char *dump = argc > 6 && !strcmp(argv[5], "--dump") ? argv[6] : NULL;
  Hash ha = {0}, hb = {0};
  ZfStreamOptions oa = zf_stream_default_options(), ob = oa;
  oa.threads = ob.threads = atoi(argv[3]);
  oa.on_chunk = on_chunk; oa.user = &ha;
  ob.on_chunk = on_chunk; ob.user = &hb;
  const char *err = NULL;
  ZfStream *a = zf_stream_create(argv[1], &oa, &err), *b = zf_stream_create(argv[1], &ob, &err),
           *c = zf_stream_create(argv[1], &ob, &err);
  if (!a || !b || !c) { fprintf(stderr, "stream: %s\n", err ? err : "?"); return 2; }
  const int ns = zf_stream_state_count(b), window = zf_stream_window(b), adv = zf_stream_advance(b);
  FILE *list = fopen(argv[2], "r");
  char line[4096];
  int clips = 0, same = 0, first = 1;
  static ZfToken ta[4096], tb[4096];
  while (fgets(line, sizeof line, list)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (!len) continue;
    FILE *ff = fopen(line, "rb");
    fseek(ff, 0, SEEK_END);
    const long bytes = ftell(ff);
    fseek(ff, 0, SEEK_SET);
    float *f = (float *)malloc((size_t)bytes);
    if (fread(f, 1, (size_t)bytes, ff) != (size_t)bytes) return 3;
    fclose(ff);
    const int frames = (int)(bytes / 320);
    const int split = (k - 1) * adv + window; /* k chunks run exactly */
    if (split + adv > frames) { free(f); continue; }
    ++clips;
    zf_stream_reset(a); ha.hash = 1469598103934665603ull;
    int na = push_all(a, f, 0, frames, ta, 4096);
    na += zf_stream_finish(a, ta + na, 4096 - na);
    zf_stream_reset(b); hb.hash = 1469598103934665603ull;
    int nb = push_all(b, f, 0, split, tb, 4096);
    if (zf_stream_chunks(b) != k) { fprintf(stderr, "%s: %lld chunks, wanted %d\n", line, zf_stream_chunks(b), k); return 4; }
    zf_stream_reset(c);
    for (int i = 0; i < ns; ++i) {
      const char *name; int64_t dims[8]; int rank, is64;
      zf_stream_state_info(b, i, &name, dims, &rank, &is64);
      size_t n = 1;
      for (int d = 0; d < rank; ++d) n *= (size_t)dims[d];
      const size_t nbytes = n * (is64 ? 8 : 4);
      void *buf = malloc(nbytes);
      if (zf_stream_export_state(b, name, buf, nbytes) || zf_stream_import_state(c, name, buf, nbytes)) {
        fprintf(stderr, "state %s failed\n", name); return 5;
      }
      if (dump && first) {
        char p[1024]; snprintf(p, sizeof p, "%s/%s.bin", dump, name);
        FILE *o = fopen(p, "wb"); fwrite(buf, 1, nbytes, o); fclose(o);
      }
      free(buf);
    }
    first = 0;
    /* the third stream reports through b's hash, which carries the chain on */
    ZfStreamOptions dummy = ob; (void)dummy;
    ZfToken *rest = tb + nb;
    /* c was created with ob too, so its chunks extend hb.hash from where b left it */
    int nc = push_all(c, f, k * adv, frames, rest, 4096 - nb);
    nc += zf_stream_finish(c, rest + nc, 4096 - nb - nc);
    const int nt = nb + nc;
    int ok = nt == na && ha.hash == hb.hash && zf_stream_chunks(c) == zf_stream_chunks(a);
    for (int t = 0; ok && t < na; ++t) ok = ta[t].token == tb[t].token && ta[t].frame == tb[t].frame;
    if (!ok) printf("differs: %s (tokens %d vs %d, chunks %lld vs %lld)\n", line, na, nt, zf_stream_chunks(a), zf_stream_chunks(c));
    same += ok;
    free(f);
  }
  printf("%d/%d clips continue bit-identically after a hand-over at chunk %d (%d states)\n", same, clips, k, ns);
  zf_stream_destroy(a); zf_stream_destroy(b); zf_stream_destroy(c);
  return same == clips ? 0 : 6;
}
