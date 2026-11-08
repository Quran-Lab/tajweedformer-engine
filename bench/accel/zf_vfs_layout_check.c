/* Folder layout 3 (tools/zfshare.py) from memory: the shared layers.zfw / frontend.zfw and one context's
 * graph and frames.zfw are put in the engine's in-memory files (zf/vfs.h) under "zfmem:7/", exactly where the
 * CoreML sidecar and the wasm build put theirs, and then zf_stream_check runs on
 * "zfmem:7/<context>/model_engine2.onnx". The graph names ../layers.zfw, so this checks that the weight
 * path is folded lexically (accel/src/ort_ops.c weights_path) and that nothing is read from the disk.
 *
 *   zf_vfs_layout_check <layout-3 folder> <context> <list.txt> <threads> <out.jsonl> [zf_stream_check options]
 *
 * The output is zf_stream_check's, so bench/v4_conformance_check.py compares it with the expected file. */
#define main zf_stream_check_main
#include "zf_stream_check.c"
#undef main

#include "zf/vfs.h"

static int put(const char *dir, const char *rel, const char *as) {
  char path[2048];
  snprintf(path, sizeof path, "%s/%s", dir, rel);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot read %s\n", path);
    return -1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *b = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
  if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
    fclose(f);
    free(b);
    fprintf(stderr, "cannot read %s\n", path);
    return -1;
  }
  fclose(f);
  return zf_vfs_put(as, b, (size_t)n) == 1 ? 0 : -1; /* the vfs owns b from here; 1 = stored */
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <layout-3 folder> <context> <list.txt> <threads> <out.jsonl> [options]\n", argv[0]);
    return 1;
  }
  const char *dir = argv[1], *ctx = argv[2];
  char rel[256], as[300];
  const char *shared[] = {"layers.zfw", "frontend.zfw"};
  const char *per[] = {"model_engine2.onnx.zfg", "model_engine2.onnx.zfg.bin", "frames.zfw"};
  for (int i = 0; i < 2; ++i) {
    snprintf(as, sizeof as, "zfmem:7/%s", shared[i]);
    if (put(dir, shared[i], as) != 0) return 2;
  }
  for (int i = 0; i < 3; ++i) {
    snprintf(rel, sizeof rel, "%s/%s", ctx, per[i]);
    snprintf(as, sizeof as, "zfmem:7/%s", rel);
    if (put(dir, rel, as) != 0) return 2;
  }
  char model[300];
  snprintf(model, sizeof model, "zfmem:7/%s/model_engine2.onnx", ctx);
  char **args = (char **)calloc((size_t)argc, sizeof(char *));
  if (!args) return 2;
  args[0] = argv[0];
  args[1] = model;
  for (int i = 3; i < argc; ++i) args[i - 1] = argv[i];
  const int rc = zf_stream_check_main(argc - 1, args);
  free(args);
  zf_vfs_remove_prefix("zfmem:7/");
  return rc;
}
