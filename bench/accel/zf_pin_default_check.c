/* zf_pin_default_check: the stream's pinning default is the platform's (zf/stream.h ZF_NO_PIN_DEFAULT): pinned
 * on Android and iOS, not pinned on Windows, Linux, macOS and WebAssembly; a caller whose header predates
 * `no_pin` (a shorter struct_size) gets the same default, and an explicit value is kept. Exits 0 when all
 * hold; prints the platform's default. Optionally creates a stream on <model> with a short struct to show
 * creation still accepts it.
 *
 *   zf_pin_default_check [model_engine2.onnx] */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "zf/stream.h"

int main(int argc, char **argv) {
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
  const int expected = 0;
#else
  const int expected = 1;
#endif
  int bad = 0;
  ZfStreamOptions o = zf_stream_default_options();
  bad += o.no_pin != expected;
  bad += ZF_NO_PIN_DEFAULT != expected;
  if (argc > 1) {
    /* a caller built before no_pin: its struct ends at chunk_ms; the stream takes the default */
    ZfStreamOptions old = zf_stream_default_options();
    old.struct_size = offsetof(ZfStreamOptions, no_pin);
    old.no_pin = 7; /* beyond the caller's struct: must not be read */
    const char *err = NULL;
    ZfStream *s = zf_stream_create(argv[1], &old, &err);
    bad += s == NULL;
    zf_stream_destroy(s);
    ZfStreamOptions pin = zf_stream_default_options();
    pin.no_pin = !expected; /* explicit, the other way */
    s = zf_stream_create(argv[1], &pin, &err);
    bad += s == NULL;
    zf_stream_destroy(s);
  }
  printf("no_pin default %d (expected %d on this platform): %s\n", o.no_pin, expected, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
