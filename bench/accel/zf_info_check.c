/* Model info, error codes and the options' size field, on a stream from files and one from a
 * CoreML model: prints the info, and exits nonzero if any expectation fails.
 *   zf_info_check <model.onnx> [<mlmodelc> <engine.zfs>]
 * Without the CoreML model, the stream from files alone. Also checked: the fields appended to
 * ZfModelInfo reach a caller that asks by size, a caller built before them gets nothing written
 * past its own struct, and a chunk length other than 160, 320 or 480 ms is refused. */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "zf/stream.h"

static int bad = 0;
#define EXPECT(c, what) do { if (!(c)) { printf("FAIL: %s\n", what); ++bad; } } while (0)

static void show(const char *label, ZfStream *s) {
  ZfModelInfo i;
  EXPECT(zf_stream_model_info_sized(s, &i, sizeof i) == 0, "model_info");
  printf("%s: vocab %d blank %d window %d advance %d layers %d left %d rate %d type %s version %s"
         " backend %d chunk %d ms\n  weights %s\n",
         label, i.vocab, i.blank, i.window, i.advance, i.layers, i.left_context, i.sample_rate,
         i.model_type, i.version, i.backend, i.chunk_ms, i.weights_sha256);
  EXPECT(i.vocab == 251 && i.blank == 250 && i.window == 61 && i.advance == 48 && i.layers == 16, "shape");
  EXPECT(strlen(i.weights_sha256) == 64, "weights digest");
  EXPECT(i.backend == ZF_BACKEND_EXACT_CPU && i.chunk_ms == 480, "backend and chunk length");
  /* a caller whose ZfModelInfo ends at weights_sha256: nothing may be written past it */
  union {
    ZfModelInfo info;
    unsigned char raw[sizeof(ZfModelInfo) + 16];
  } old;
  memset(&old, 0xA5, sizeof old);
  EXPECT(zf_stream_model_info(s, &old.info) == 0, "model_info, the old call");
  int past = 0;
  for (size_t b = offsetof(ZfModelInfo, backend); b < sizeof old.raw; ++b) past += old.raw[b] != 0xA5;
  EXPECT(past == 0 && old.info.vocab == i.vocab && !strcmp(old.info.weights_sha256, i.weights_sha256),
         "the old call fills the old fields and nothing past them");
}

int main(int argc, char **argv) {
  if (argc != 2 && argc < 4) return 1;
  const char *err = NULL;
  ZfStream *a = zf_stream_create(argv[1], NULL, &err);
  ZfStream *b = argc >= 4 ? zf_stream_create_from_coreml(argv[2], argv[3], NULL, &err) : NULL;
  if (!a || (argc >= 4 && !b)) { printf("create: %s\n", err ? err : "?"); return 2; }
  show("files ", a);
  if (b) {
    show("coreml", b);
    ZfModelInfo ia, ib;
    zf_stream_model_info(a, &ia);
    zf_stream_model_info(b, &ib);
    EXPECT(!strcmp(ia.weights_sha256, ib.weights_sha256), "both paths name the same weights");
  }

  const char *msg = NULL;
  EXPECT(zf_stream_last_error(a, &msg) == ZF_OK, "no error yet");
  EXPECT(zf_stream_push(a, NULL, 5, NULL, 0) == -1, "push NULL frames fails");
  EXPECT(zf_stream_last_error(a, &msg) == ZF_ERR_ARGUMENT, "... as an argument error");
  printf("argument error message: %s\n", msg);
  char buf[8];
  EXPECT(zf_stream_export_state(a, "no_such_state", buf, sizeof buf) == -1 &&
         zf_stream_last_error(a, NULL) == ZF_ERR_ARGUMENT, "unknown state");
  zf_stream_reset(a);
  EXPECT(zf_stream_last_error(a, NULL) == ZF_OK, "reset clears the error");

  /* a caller built against a header that ended before sample_rate */
  ZfStreamOptions old = zf_stream_default_options();
  old.struct_size = offsetof(ZfStreamOptions, sample_rate);
  old.sample_rate = 48000; /* beyond its struct: must be ignored */
  ZfStream *c = zf_stream_create(argv[1], &old, &err);
  ZfModelInfo ic;
  EXPECT(c && zf_stream_model_info(c, &ic) == 0 && ic.sample_rate == 16000, "old callers keep defaults");
  ZfStreamOptions now = zf_stream_default_options();
  now.sample_rate = 48000;
  ZfStream *d = zf_stream_create(argv[1], &now, &err);
  ZfModelInfo id;
  EXPECT(d && zf_stream_model_info(d, &id) == 0 && id.sample_rate == 48000, "current callers set it");

  /* a caller built against a header that ended at batch: a stray chunk_ms beyond it is ignored */
  ZfStreamOptions before = zf_stream_default_options();
  before.struct_size = offsetof(ZfStreamOptions, on_evidence);
  before.chunk_ms = 200;
  ZfStream *e = zf_stream_create(argv[1], &before, &err);
  EXPECT(e != NULL, "old callers never see chunk_ms");
  /* a chunk length that is not one of the three fails creation, as an argument error */
  ZfStreamOptions odd = zf_stream_default_options();
  odd.chunk_ms = 200;
  err = NULL;
  EXPECT(zf_stream_create(argv[1], &odd, &err) == NULL && err && strstr(err, "chunk_ms"), "chunk_ms 200 refused");
  printf("chunk length error message: %s\n", err ? err : "");
  zf_stream_destroy(a); zf_stream_destroy(b); zf_stream_destroy(c); zf_stream_destroy(d);
  zf_stream_destroy(e);
  printf(bad ? "%d checks failed\n" : "all checks pass\n", bad);
  return bad ? 3 : 0;
}
