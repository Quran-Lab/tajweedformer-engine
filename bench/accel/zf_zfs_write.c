/* zf_zfs_write: the exact engine's files from a CoreML build's weights and its sidecar, written to a folder
 * (zf_zfs_write_files; each file checked against its SHA-256 before it is written).
 *
 *   zf_zfs_write <mlmodelc or weight.bin> <engine.zfs> <out dir>
 *
 * Prints every file with its SHA-256; exit 0 only if all were rebuilt and match. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zf/zfs.h"

static unsigned char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  const long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *b = len >= 0 ? (unsigned char *)malloc((size_t)len + 1) : NULL;
  if (b && fread(b, 1, (size_t)len, f) != (size_t)len) {
    free(b);
    b = NULL;
  }
  fclose(f);
  if (b) *n = (size_t)len;
  return b;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <mlmodelc or weight.bin> <engine.zfs> <out dir>\n", argv[0]);
    return 2;
  }
  char wpath[4096];
  const size_t al = strlen(argv[1]);
  if (al > 11 && !strcmp(argv[1] + al - 11, ".mlmodelc/") ) snprintf(wpath, sizeof wpath, "%sweights/weight.bin", argv[1]);
  else if (al > 9 && !strcmp(argv[1] + al - 9, ".mlmodelc")) snprintf(wpath, sizeof wpath, "%s/weights/weight.bin", argv[1]);
  else snprintf(wpath, sizeof wpath, "%s", argv[1]);
  size_t wn = 0, zn = 0;
  unsigned char *w = slurp(wpath, &wn), *z = slurp(argv[2], &zn);
  if (!w || !z) {
    fprintf(stderr, "cannot read %s\n", !w ? wpath : argv[2]);
    return 1;
  }
  const char *err = NULL;
  const int n = zf_zfs_write_files(w, wn, z, zn, argv[3], &err);
  if (n < 0) {
    fprintf(stderr, "zf_zfs_write_files: %s\n", err ? err : "failed");
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    char name[512];
    unsigned char d[32];
    if (zf_zfs_file_name(z, zn, i, name, sizeof name, d) != 0) return 1;
    for (int j = 0; j < 32; ++j) printf("%02x", d[j]);
    printf("  %s\n", name);
  }
  fprintf(stderr, "%d files written to %s, each checked\n", n, argv[3]);
  free(w);
  free(z);
  return 0;
}
