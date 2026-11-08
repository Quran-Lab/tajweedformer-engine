/* zf_stream_pin_check: streams made and destroyed one after another on one thread, pinned, keep their
 * engine team and leave the thread's mask as they found it (engine/src/pool.c: pin_caller,
 * unpin_callers, process_mask).
 *
 * The bug this guards (DELIVERY, Alcatel 1, foreground, cpuset /top-app): a stream pinned its creating
 * thread to one core and destroying it never undid that, so the next stream's pool sized its team from
 * the one-core mask and every take after the first ran on one engine thread (SDK RTF 0.777 against the
 * CLI's 0.24; the CLI keeps one stream, so it never showed).
 *
 *   zf_stream_pin_check <model_engine2.onnx> <clip.f32> [threads] [rounds]
 *
 * Each round: a stream with pinning on (no_pin 0, whatever the platform's default), one clip pushed,
 * the team read (zf_stream_threads_active) and the take's time, then destroyed and the thread's mask
 * compared with the one it had at the start. Then two streams alive at once on the thread. Expected
 * team: min(threads, the process's cores). Exits nonzero on any failure. No bit depends on any of it. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zf/stream.h"

#if defined(_WIN32)
#include <windows.h>
typedef DWORD_PTR Mask;
static Mask thread_mask(void) {
  DWORD_PTR proc = 0, sys = 0;
  GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
  const DWORD_PTR old = SetThreadAffinityMask(GetCurrentThread(), proc);
  if (old) SetThreadAffinityMask(GetCurrentThread(), old);
  return old;
}
static int process_cores(void) {
  DWORD_PTR proc = 0, sys = 0;
  GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
  int n = 0;
  for (; proc; proc >>= 1) n += (int)(proc & 1);
  return n;
}
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
typedef cpu_set_t Mask;
static Mask thread_mask(void) {
  Mask m;
  CPU_ZERO(&m);
  sched_getaffinity(0, sizeof m, &m);
  return m;
}
static int process_cores(void) {
  Mask m;
  CPU_ZERO(&m);
  sched_getaffinity(getpid(), sizeof m, &m);
  return CPU_COUNT(&m);
}
#else
typedef int Mask;
static Mask thread_mask(void) { return 0; }
static int process_cores(void) { return 1; }
#endif

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static float *g_feats;
static int g_frames;

/* one take: the clip's frames pushed, finished; returns ms per chunk, *team the team after it */
static double take(ZfStream *s, int *team) {
  ZfToken toks[256];
  zf_stream_reset(s);
  const double t0 = now_ms();
  for (int at = 0; at < g_frames; at += 16) {
    const int n = g_frames - at < 16 ? g_frames - at : 16;
    if (zf_stream_push(s, g_feats + (size_t)at * 80, n, toks, 256) < 0) return -1;
  }
  zf_stream_finish(s, toks, 256);
  const double dt = now_ms() - t0;
  *team = zf_stream_threads_active(s);
  const long long c = zf_stream_chunks(s);
  return c > 0 ? dt / (double)c : -1;
}

static ZfStream *make(const char *model, int threads) {
  ZfStreamOptions o = zf_stream_default_options();
  o.threads = threads;
  o.no_pin = 0; /* pinned, the phone's default, on any platform */
  const char *err = NULL;
  ZfStream *s = zf_stream_create(model, &o, &err);
  if (!s) fprintf(stderr, "stream: %s\n", err ? err : "?");
  return s;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model_engine2.onnx> <clip.f32> [threads] [rounds]\n", argv[0]);
    return 2;
  }
  const int threads = argc > 3 ? atoi(argv[3]) : 4;
  const int rounds = argc > 4 ? atoi(argv[4]) : 4;
  FILE *f = fopen(argv[2], "rb");
  if (!f) { perror(argv[2]); return 2; }
  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  g_feats = (float *)malloc((size_t)bytes);
  if (!g_feats || fread(g_feats, 1, (size_t)bytes, f) != (size_t)bytes) { perror("read"); return 2; }
  fclose(f);
  g_frames = (int)(bytes / (80 * 4));
  const int cores = process_cores();
  const int expect = threads < cores ? threads : cores;
  const Mask original = thread_mask();
  int bad = 0;
  double first_ms = 0;
  for (int r = 0; r < rounds; ++r) {
    ZfStream *s = make(argv[1], threads);
    if (!s) return 3;
    int team = 0;
    const double ms = take(s, &team);
    zf_stream_destroy(s);
    const Mask after = thread_mask();
    const int restored = memcmp(&after, &original, sizeof after) == 0;
    if (r == 0) first_ms = ms;
    const int ok = team == expect && restored && ms > 0;
    bad += !ok;
    printf("stream %d: team %d of %d expected, %.1f ms a chunk (first %.1f), caller's mask %s: %s\n", r, team,
           expect, ms, first_ms, restored ? "restored" : "NOT restored", ok ? "ok" : "FAIL");
  }
  ZfStream *a = make(argv[1], threads), *b = make(argv[1], threads);
  if (!a || !b) return 3;
  int ta = 0, tb = 0;
  const double ma = take(a, &ta), mb = take(b, &tb);
  zf_stream_destroy(b);
  zf_stream_destroy(a);
  const Mask after = thread_mask();
  const int restored = memcmp(&after, &original, sizeof after) == 0;
  const int ok = ta == expect && tb == expect && restored;
  bad += !ok;
  printf("two streams at once: teams %d and %d of %d expected (%.1f and %.1f ms a chunk), caller's mask %s: %s\n",
         ta, tb, expect, ma, mb, restored ? "restored" : "NOT restored", ok ? "ok" : "FAIL");
  printf("%s\n", bad ? "FAIL" : "all ok");
  free(g_feats);
  return bad ? 1 : 0;
}
