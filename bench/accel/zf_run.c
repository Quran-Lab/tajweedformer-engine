/*
 * zf_run: decode a whole clip through the streaming graph and report what it said and how long it
 * took, with the engine's kernels either in or out.
 *
 * The point is an end-to-end check on the real target: run the rewritten graph (tools/zfgraph.py,
 * whose ZfLinearS8 and ZfSwoosh nodes call the engine's kernels) and the original graph over the
 * same features, and compare the greedy CTC token sequences. The kernels are bit-exact on the
 * matmuls and differ only in floating-point rounding on Swoosh, so the two runs should say exactly
 * the same thing; if they do not, this prints where they part company.
 *
 * States carry from chunk to chunk exactly as the streaming decoders do: every input named <name>
 * takes the value of the output named new_<name>.
 *
 * usage: zf_run <model.onnx> <features.f32> <chunks> <threads> [--no-accel] [--tokens FILE]
 *               [--warmup N] [--profile PREFIX]
 *               [--gap-ms MS]   idle this long before each measured chunk, as a live stream does
 *               [--prefetch STEP]  run the window's front end as its frames arrive, STEP at a
 *                               time, with the idle split between the steps; only the rest is timed
 *               [--outputs a,b|none]  ask for these of the graph's other outputs only (V4's heads;
 *                               default all), as zf_stream_select_outputs does
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "onnxruntime_c_api.h"
#include "zf/layer.h"
#include "zf/ort_ops.h"
#include "zf/decode.h"
#include "zf/pool.h"

#define BINS 80

/* The chunk geometry is the model's to declare, not this harness's to assume.
 *
 * A sherpa-onnx zipformer2 carries it in its metadata: T is the window of feature frames one step
 * consumes and decode_chunk_len is how far the window advances, so T - decode_chunk_len is the
 * right context a step needs beyond the audio it reports on. For the shipped v3.1 those are 61 and
 * 48, which is 480 ms of new audio a step and 130 ms of lookahead, and the output is 12 frames, so
 * the model subsamples by four.
 *
 * Reading them rather than hard-coding them is what lets a graph exported at a different chunk
 * length run through the same engine unchanged: the layers already take their frame counts from
 * their own node attributes, and this was the last place a number was assumed. The values are
 * cross-checked against the graph's declared input and output shapes, and a mismatch is fatal
 * rather than quietly misaligning the audio. */
typedef struct {
  int window;     /* T: feature frames in */
  int shift;      /* decode_chunk_len: feature frames advanced */
  int out_frames; /* encoder output frames per step */
  int subsample;  /* shift / out_frames */
  int from_meta;  /* whether the metadata supplied it */
} ZfGeometry;

static int meta_int(const OrtApi *api, OrtModelMetadata *md, OrtAllocator *alloc, const char *key) {
  char *val = NULL;
  OrtStatus *st = api->ModelMetadataLookupCustomMetadataMap(md, alloc, key, &val);
  if (st) { api->ReleaseStatus(st); return 0; }
  if (!val) return 0;
  const int v = atoi(val);
  alloc->Free(alloc, val);
  return v > 0 ? v : 0;
}

static const OrtApi *g_ort = NULL;

static void die(const char *what, OrtStatus *status) {
  if (status) {
    fprintf(stderr, "%s: %s\n", what, g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
  } else {
    fprintf(stderr, "%s\n", what);
  }
  exit(2);
}

#define CHECK(expr)                 \
  do {                              \
    OrtStatus *_st = (expr);        \
    if (_st) die(#expr, _st);       \
  } while (0)

/* One length-prefixed named blob, for ZF_DUMP. */
static void dump_tensor(FILE *f, const char *name, const void *data, size_t bytes) {
  const int32_t n = (int32_t)strlen(name);
  const int32_t b = (int32_t)bytes;
  fwrite(&n, 4, 1, f);
  fwrite(name, 1, (size_t)n, f);
  fwrite(&b, 4, 1, f);
  fwrite(data, 1, bytes, f);
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Minor and major faults so far.
 *
 * A chunk's join wait is 12 to 23 ms in every chunk, in about eight stalls of one to two
 * milliseconds. A stall that long in a worker is not arithmetic and not a barrier; the candidates
 * are the scheduler and the memory manager. This phone has a gigabyte of RAM and the engine sweeps
 * 63 MB of weights a chunk, so if those pages are file backed and being evicted, every chunk pays
 * to read some of them back. Major faults say so outright. */
static void read_faults(unsigned long *minor, unsigned long *major) {
  *minor = 0;
  *major = 0;
  FILE *f = fopen("/proc/self/stat", "r");
  if (!f) return;
  char buf[1024];
  const size_t n = fread(buf, 1, sizeof buf - 1, f);
  fclose(f);
  buf[n] = 0;
  /* fields after the parenthesised name: state is 3, minflt is 10, majflt is 12 */
  char *p = strrchr(buf, ')');
  if (!p) return;
  int field = 2;
  char *save = NULL;
  for (char *tok = strtok_r(p + 1, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
    ++field;
    if (field == 10) *minor = strtoul(tok, NULL, 10);
    if (field == 12) { *major = strtoul(tok, NULL, 10); return; }
  }
}

static int cmp_double(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model.onnx> <features.f32> <chunks> <threads> [--no-accel]"
                    " [--tokens FILE] [--warmup N] [--profile PREFIX]\n", argv[0]);
    return 1;
  }
  const char *model_path = argv[1];
  const int want_chunks = atoi(argv[3]);
  const int threads = atoi(argv[4]);
  int accel = 1, warmup = 1, gap_ms = 0, prefetch = 0;
  const char *tokens_path = NULL, *profile = NULL;
  const char *outputs = NULL; /* --outputs a,b|none: the other outputs to ask for (default all) */
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--no-accel")) accel = 0;
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_path = argv[++i];
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--profile") && i + 1 < argc) profile = argv[++i];
    else if (!strcmp(argv[i], "--gap-ms") && i + 1 < argc) gap_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--prefetch") && i + 1 < argc) prefetch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--outputs") && i + 1 < argc) outputs = argv[++i];
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf(stderr, "this libonnxruntime does not support API version %d\n", ORT_API_VERSION);
    return 2;
  }

  FILE *ff = fopen(argv[2], "rb");
  if (!ff) { perror("features"); return 1; }
  fseek(ff, 0, SEEK_END);
  const long bytes = ftell(ff);
  fseek(ff, 0, SEEK_SET);
  const int frames = (int)(bytes / (BINS * 4));
  float *feats = (float *)malloc((size_t)frames * BINS * sizeof(float));
  if (!feats || fread(feats, 1, (size_t)bytes, ff) != (size_t)bytes) { perror("read"); return 1; }
  fclose(ff);

  OrtEnv *env = NULL;
  CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "zf", &env));
  OrtSessionOptions *opts = NULL;
  CHECK(g_ort->CreateSessionOptions(&opts));
  /* The engine runs its own pool, so ONNX Runtime needs one thread: its own would only contend.
   * Its dispatch is cheaper than KernelContext_ParallelFor for this graph's many small regions,
   * measured at 53 ms a chunk in the layer matmuls against 61. ZF_ORT_POOL asks for the other. */
  const int ort_threads = getenv("ZF_ORT_POOL") ? threads : 1;
  CHECK(g_ort->SetIntraOpNumThreads(opts, ort_threads));
  CHECK(g_ort->SetInterOpNumThreads(opts, 1));
  CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
  if (profile) CHECK(g_ort->EnableProfiling(opts, profile));
  if (accel) CHECK(zf_ort_register(g_ort, opts, threads));

  const double t_load = now_ms();
  if (getenv("ZF_TRACE")) { fprintf(stderr, "[zf_run] creating session\n"); fflush(stderr); }
  OrtSession *session = NULL;
  CHECK(g_ort->CreateSession(env, model_path, opts, &session));
  const double load_ms = now_ms() - t_load;
  if (getenv("ZF_TRACE")) { fprintf(stderr, "[zf_run] session ready in %.0f ms\n", load_ms); fflush(stderr); }

  size_t n_in = 0, n_out = 0;
  CHECK(g_ort->SessionGetInputCount(session, &n_in));
  CHECK(g_ort->SessionGetOutputCount(session, &n_out));
  OrtAllocator *alloc = NULL;
  CHECK(g_ort->GetAllocatorWithDefaultOptions(&alloc));

  char **in_names = (char **)calloc(n_in, sizeof(char *));
  char **out_names = (char **)calloc(n_out, sizeof(char *));
  OrtValue **in_vals = (OrtValue **)calloc(n_in, sizeof(OrtValue *));
  OrtValue **out_vals = (OrtValue **)calloc(n_out, sizeof(OrtValue *));
  int64_t **in_shapes = (int64_t **)calloc(n_in, sizeof(int64_t *));
  size_t *in_dims = (size_t *)calloc(n_in, sizeof(size_t));
  ONNXTensorElementDataType *in_types =
      (ONNXTensorElementDataType *)calloc(n_in, sizeof(ONNXTensorElementDataType));
  void **state = (void **)calloc(n_in, sizeof(void *));
  size_t *state_bytes = (size_t *)calloc(n_in, sizeof(size_t));
  if (!in_names || !out_names || !in_vals || !out_vals || !in_shapes || !in_dims || !in_types ||
      !state || !state_bytes) {
    fprintf(stderr, "out of memory\n");
    return 3;
  }

  for (size_t i = 0; i < n_in; ++i) {
    CHECK(g_ort->SessionGetInputName(session, i, alloc, &in_names[i]));
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *si = NULL;
    CHECK(g_ort->SessionGetInputTypeInfo(session, i, &ti));
    CHECK(g_ort->CastTypeInfoToTensorInfo(ti, &si));
    CHECK(g_ort->GetTensorElementType(si, &in_types[i]));
    CHECK(g_ort->GetDimensionsCount(si, &in_dims[i]));
    in_shapes[i] = (int64_t *)calloc(in_dims[i], sizeof(int64_t));
    CHECK(g_ort->GetDimensions(si, in_shapes[i], in_dims[i]));
    g_ort->ReleaseTypeInfo(ti);
    size_t elems = 1;
    for (size_t d = 0; d < in_dims[i]; ++d)
      elems *= (size_t)(in_shapes[i][d] > 0 ? in_shapes[i][d] : 1);
    state_bytes[i] = elems * ((in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? 8 : 4);
    state[i] = calloc(1, state_bytes[i]);
    if (!state[i]) { fprintf(stderr, "out of memory for %s\n", in_names[i]); return 3; }
  }
  for (size_t i = 0; i < n_out; ++i)
    CHECK(g_ort->SessionGetOutputName(session, i, alloc, &out_names[i]));

  /* --outputs: ask each Run for log_probs, the new_ states and only the listed other outputs (V4's
   * heads), as zf_stream_select_outputs does; zf_rt then skips the nodes only the others need. */
  if (outputs) {
    size_t k = 0;
    for (size_t o = 0; o < n_out; ++o) {
      const char *nm = out_names[o];
      int keep = !strcmp(nm, "log_probs") || !strncmp(nm, "new_", 4);
      if (!keep && strcmp(outputs, "none")) {
        const size_t L = strlen(nm);
        for (const char *q = outputs; *q && !keep; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q))
          keep = !strncmp(q, nm, L) && (q[L] == ',' || q[L] == 0);
      }
      if (keep) out_names[k++] = out_names[o];
    }
    n_out = k;
  }

  size_t logits_index = 0;
  for (size_t o = 0; o < n_out; ++o)
    if (!strcmp(out_names[o], "log_probs")) logits_index = o;

  ZfGeometry geom = {0, 0, 0, 0, 0};
  {
    OrtModelMetadata *md = NULL;
    CHECK(g_ort->SessionGetModelMetadata(session, &md));
    geom.window = meta_int(g_ort, md, alloc, "T");
    geom.shift = meta_int(g_ort, md, alloc, "decode_chunk_len");
    g_ort->ReleaseModelMetadata(md);
    geom.from_meta = geom.window > 0 && geom.shift > 0;

    /* the graph's own shapes, which must agree */
    int declared_window = 0;
    for (size_t i = 0; i < n_in; ++i)
      if (!strcmp(in_names[i], "x") && in_dims[i] >= 2) declared_window = (int)in_shapes[i][1];
    {
      OrtTypeInfo *ti = NULL;
      const OrtTensorTypeAndShapeInfo *si = NULL;
      size_t rank = 0;
      int64_t od[3] = {0, 0, 0};
      CHECK(g_ort->SessionGetOutputTypeInfo(session, logits_index, &ti));
      CHECK(g_ort->CastTypeInfoToTensorInfo(ti, &si));
      CHECK(g_ort->GetDimensionsCount(si, &rank));
      if (rank == 3) CHECK(g_ort->GetDimensions(si, od, 3));
      g_ort->ReleaseTypeInfo(ti);
      geom.out_frames = (int)od[1];
    }
    if (!geom.window) geom.window = declared_window;
    if (declared_window > 0 && geom.window != declared_window) {
      fprintf(stderr, "chunk geometry: metadata T is %d but the graph declares x as %d frames\n",
              geom.window, declared_window);
      return 2;
    }
    if (geom.out_frames <= 0) {
      fprintf(stderr, "chunk geometry: the graph does not declare how many frames it outputs\n");
      return 2;
    }
    /* Without decode_chunk_len the advance has to be inferred, and the only consistent choice is
     * the one that covers the stream exactly: output frames times the subsampling. Four is what
     * every zipformer2 front end does, and it is checked against the window below. */
    if (!geom.shift) geom.shift = geom.out_frames * 4;
    if (geom.shift % geom.out_frames != 0) {
      fprintf(stderr, "chunk geometry: an advance of %d does not divide into %d output frames\n",
              geom.shift, geom.out_frames);
      return 2;
    }
    geom.subsample = geom.shift / geom.out_frames;
    if (geom.shift > geom.window) {
      fprintf(stderr, "chunk geometry: an advance of %d exceeds the %d frame window\n", geom.shift,
              geom.window);
      return 2;
    }
  }

  OrtMemoryInfo *mem = NULL;
  CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));

  const int max_chunks = (frames - geom.window) / geom.shift + 1;
  const int chunks = want_chunks > 0 && want_chunks < max_chunks ? want_chunks : max_chunks;
  double *chunk_ms = (double *)calloc((size_t)(chunks > 0 ? chunks : 1), sizeof(double));
  double *chunk_wait_ms = (double *)calloc((size_t)(chunks > 0 ? chunks : 1), sizeof(double));
  int *tokens = (int *)calloc((size_t)(chunks > 0 ? chunks : 1) * 32, sizeof(int));
  int *token_frames = (int *)calloc((size_t)(chunks > 0 ? chunks : 1) * 32, sizeof(int));
  ZfToken *token_info = (ZfToken *)calloc((size_t)(chunks > 0 ? chunks : 1) * 32, sizeof(ZfToken));
  int n_tokens = 0;
  ZfDecoder *decoder = NULL;

  /* One pass before timing: the governor idles these cores at 546 MHz and a single chunk is not
   * long enough to pull the clock up, so the first chunks would measure the governor. */
  for (int w = 0; w < warmup && chunks > 0; ++w) {
    for (size_t i = 0; i < n_in; ++i) {
      if (!strcmp(in_names[i], "x"))
        memcpy(state[i], feats, (size_t)geom.window * BINS * sizeof(float));
      CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem, state[i], state_bytes[i], in_shapes[i],
                                                  in_dims[i], in_types[i], &in_vals[i]));
    }
    CHECK(g_ort->Run(session, NULL, (const char *const *)in_names,
                     (const OrtValue *const *)in_vals, n_in, (const char *const *)out_names, n_out,
                     out_vals));
    for (size_t o = 0; o < n_out; ++o) { g_ort->ReleaseValue(out_vals[o]); out_vals[o] = NULL; }
    for (size_t i = 0; i < n_in; ++i) { g_ort->ReleaseValue(in_vals[i]); in_vals[i] = NULL; }
  }
  for (size_t i = 0; i < n_in; ++i) memset(state[i], 0, state_bytes[i]);
  zf_ort_stats_reset();

  /* the warmup pass advanced the engine's own caches, so the measured stream starts fresh */
  if (accel) zf_ort_reset_state();

  const char *dump_dir = getenv("ZF_DUMP");
  unsigned long minflt0 = 0, majflt0 = 0, minflt1 = 0, majflt1 = 0;
  read_faults(&minflt0, &majflt0);
  for (int c = 0; c < chunks; ++c) {
    for (size_t i = 0; i < n_in; ++i) {
      if (!strcmp(in_names[i], "x"))
        memcpy(state[i], feats + (size_t)c * geom.shift * BINS,
               (size_t)geom.window * BINS * sizeof(float));
      CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem, state[i], state_bytes[i], in_shapes[i],
                                                  in_dims[i], in_types[i], &in_vals[i]));
    }
    if (accel && prefetch > 0) {
      /* The window's frames arrive over the gap: the front end runs on each STEP of them as they
       * come, and the timed Run below is what is left once the last one is in. Not timed. */
      const int steps = (geom.shift + prefetch - 1) / prefetch;
      long slice_ns = gap_ms > 0 ? (long)gap_ms * 1000000L / (steps + 1) : 0;
      /* ZF_PF_LATE_MS=L: hold the prefetch back and run it as one burst L ms before the window
       * completes, so its work warms the cores for the rest instead of being spread thin. */
      static int late = -2;
      if (late == -2) late = getenv("ZF_PF_LATE_MS") ? atoi(getenv("ZF_PF_LATE_MS")) : -1;
      if (late >= 0 && gap_ms > late) {
        struct timespec d = {(gap_ms - late) / 1000, (long)((gap_ms - late) % 1000) * 1000000L};
        nanosleep(&d, NULL);
        for (int v = geom.window - geom.shift; v < geom.window; v += prefetch)
          zf_ort_frontend_prefetch(feats + (size_t)c * geom.shift * BINS, geom.window, v);
        struct timespec e = {late / 1000, (long)(late % 1000) * 1000000L};
        nanosleep(&e, NULL);
        slice_ns = -1;
      }
      for (int v = geom.window - geom.shift; slice_ns >= 0 && v < geom.window; v += prefetch) {
        if (slice_ns > 0) {
          struct timespec d = {slice_ns / 1000000000L, slice_ns % 1000000000L};
          nanosleep(&d, NULL);
        }
        zf_ort_frontend_prefetch(feats + (size_t)c * geom.shift * BINS, geom.window, v);
      }
      if (slice_ns > 0) {
        struct timespec d = {slice_ns / 1000000000L, slice_ns % 1000000000L};
        nanosleep(&d, NULL);
      }
    } else if (gap_ms > 0) {
      /* A live stream is idle between chunks (a 480 ms chunk takes a few ms to decode), so the
       * threads park and the cores clock down; back to back measures neither. Not timed. */
      struct timespec gap = {gap_ms / 1000, (long)(gap_ms % 1000) * 1000000L};
      nanosleep(&gap, NULL);
      /* ZF_PREWARM_MS: spin the calling thread this long just before the chunk (not timed), to
       * measure how much of a cold chunk is the cores clocking back up. */
      static int prewarm = -1;
      if (prewarm < 0) prewarm = getenv("ZF_PREWARM_MS") ? atoi(getenv("ZF_PREWARM_MS")) : 0;
      if (prewarm > 0) {
        const double until = now_ms() + prewarm;
        volatile double sink = 0;
        while (now_ms() < until) sink += 1.0;
      }
    }
    const double w0 = zf_pool_wait_ms();
    const double t0 = now_ms();
    CHECK(g_ort->Run(session, NULL, (const char *const *)in_names,
                     (const OrtValue *const *)in_vals, n_in, (const char *const *)out_names, n_out,
                     out_vals));
    chunk_ms[c] = now_ms() - t0;
    chunk_wait_ms[c] = zf_pool_wait_ms() - w0;

    /* greedy CTC over this chunk's 12 frames, with a confidence for each phoneme */
    float *logp = NULL;
    CHECK(g_ort->GetTensorMutableData(out_vals[logits_index], (void **)&logp));
    OrtTensorTypeAndShapeInfo *li = NULL;
    CHECK(g_ort->GetTensorTypeAndShape(out_vals[logits_index], &li));
    int64_t ldims[3] = {1, 0, 0};
    CHECK(g_ort->GetDimensions(li, ldims, 3));
    g_ort->ReleaseTensorTypeAndShapeInfo(li);
    const int nframes = (int)ldims[1], vocab = (int)ldims[2];
    if (!decoder) {
      decoder = zf_decoder_create(vocab, vocab - 1); /* blank is the vocabulary's last id */
      if (!decoder) { fprintf(stderr, "cannot start the decoder\n"); return 3; }
    }
    const int got = zf_decoder_push(decoder, logp, nframes, token_info + n_tokens,
                                    32 * (chunks > 0 ? chunks : 1) - n_tokens);
    for (int t = 0; t < got; ++t) {
      tokens[n_tokens + t] = token_info[n_tokens + t].token;
      token_frames[n_tokens + t] = token_info[n_tokens + t].frame;
    }
    n_tokens += got;

    /* ZF_DUMP: one file a chunk holding this chunk's log_probs and every state output, so the
     * engine can be diffed against the ORT goldens tensor by tensor. Token-level comparison says
     * that the two disagree; this says where. Layout per record: int32 name length, the name,
     * int32 byte count, the bytes. */
    if (dump_dir) {
      char path[512];
      snprintf(path, sizeof path, "%s/chunk_%03d.bin", dump_dir, c);
      FILE *df = fopen(path, "wb");
      if (df) {
        /* ZF_DUMP_EXTRA: only the tensors the graph was asked to expose, leaving out log_probs and
         * the 98 state outputs. A full dump is six megabytes a chunk and this link does not
         * reliably survive pulling that; the exposed tensors alone are a tenth of it. */
        const int extra_only = getenv("ZF_DUMP_EXTRA") != NULL;
        if (!extra_only)
          dump_tensor(df, "log_probs", logp, (size_t)nframes * vocab * sizeof(float));
        for (size_t o = 0; o < n_out; ++o) {
          void *sp = NULL;
          if (g_ort->GetTensorMutableData(out_vals[o], &sp) != NULL) continue;
          if (extra_only && strncmp(out_names[o], "new_", 4) == 0) continue;
          if (strncmp(out_names[o], "new_", 4) == 0) {
            size_t nb = 0;
            for (size_t i = 0; i < n_in; ++i)
              if (!strcmp(out_names[o] + 4, in_names[i])) { nb = state_bytes[i]; break; }
            if (nb) dump_tensor(df, out_names[o] + 4, sp, nb);
            continue;
          }
          if (!strcmp(out_names[o], "log_probs")) continue;
          /* Anything else the graph was asked to expose, which is what makes a node level trace
           * from ONNX Runtime ON THE DEVICE possible. Every reference tensor in this repo until
           * now came from x86, and for any kernel whose result depends on accumulation order the
           * two machines are different programs: that is why the four transcendentals reached a
           * hundred per cent and why nothing whose order matters has moved. */
          OrtTensorTypeAndShapeInfo *ti = NULL;
          if (g_ort->GetTensorTypeAndShape(out_vals[o], &ti) != NULL) continue;
          size_t elems = 0;
          ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
          OrtStatus *s1 = g_ort->GetTensorShapeElementCount(ti, &elems);
          OrtStatus *s2 = g_ort->GetTensorElementType(ti, &et);
          g_ort->ReleaseTensorTypeAndShapeInfo(ti);
          if (s1 || s2) { if (s1) g_ort->ReleaseStatus(s1); if (s2) g_ort->ReleaseStatus(s2); continue; }
          const size_t esz = (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                              et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) ? 4
                           : (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? 8
                           : (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                              et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) ? 1 : 0;
          if (esz && elems) dump_tensor(df, out_names[o], sp, elems * esz);
        }
        fclose(df);
      }
    }
    for (size_t o = 0; o < n_out; ++o) {
      if (strncmp(out_names[o], "new_", 4) != 0) continue;
      for (size_t i = 0; i < n_in; ++i) {
        if (strcmp(out_names[o] + 4, in_names[i]) != 0) continue;
        void *src = NULL;
        CHECK(g_ort->GetTensorMutableData(out_vals[o], &src));
        memcpy(state[i], src, state_bytes[i]);
        break;
      }
    }
    for (size_t o = 0; o < n_out; ++o) { g_ort->ReleaseValue(out_vals[o]); out_vals[o] = NULL; }
    for (size_t i = 0; i < n_in; ++i) { g_ort->ReleaseValue(in_vals[i]); in_vals[i] = NULL; }
  }

  double *sorted = (double *)malloc((size_t)(chunks > 0 ? chunks : 1) * sizeof(double));
  memcpy(sorted, chunk_ms, (size_t)chunks * sizeof(double));
  qsort(sorted, (size_t)chunks, sizeof(double), cmp_double);
  double total = 0.0;
  for (int c = 0; c < chunks; ++c) total += chunk_ms[c];

  ZfOrtStats stats;
  zf_ort_stats(&stats);
  if (profile) {
    char *pf = NULL;
    CHECK(g_ort->SessionEndProfiling(session, alloc, &pf));
  }

  {
    /* What a caller needs to know about latency, from the geometry rather than from a comment. A
     * phoneme is reported once the window that covers it has been consumed, so the newest output
     * frame waits for (window - shift) frames of lookahead and the oldest waits for the whole
     * window; add the compute time to either. */
    const double frame_ms = 10.0;
    const double look = (geom.window - geom.shift) * frame_ms;
    printf(" \"geometry\": {\"window_frames\": %d, \"advance_frames\": %d, \"out_frames\": %d,"
           " \"subsample\": %d, \"from_metadata\": %d, \"audio_per_step_ms\": %.0f,"
           " \"lookahead_ms\": %.0f, \"left_context_frames\": %d},\n",
           geom.window, geom.shift, geom.out_frames, geom.subsample, geom.from_meta,
           geom.shift * frame_ms, look, zf_ort_left_context());
  }
  printf("{\"model\": \"%s\", \"accel\": %d, \"threads\": %d, \"chunks\": %d, \"load_ms\": %.1f,\n",
         model_path, accel, threads, chunks, load_ms);
  printf(" \"chunk_ms_mean\": %.2f, \"chunk_ms_median\": %.2f, \"chunk_ms_min\": %.2f,"
         " \"chunk_ms_p95\": %.2f,\n",
         chunks ? total / chunks : 0.0, chunks ? sorted[chunks / 2] : 0.0, chunks ? sorted[0] : 0.0,
         chunks ? sorted[(int)((chunks - 1) * 0.95)] : 0.0);
  printf(" \"rtf\": %.4f, \"linear_ms_total\": %.2f, \"linear_calls\": %lu, \"linear_gmadds\": %.3f,"
         " \"quant_ms\": %.2f, \"gemm_ms\": %.2f, \"gemm_gmadds\": %.3f, \"deq_ms\": %.2f,"
         " \"swoosh_ms_total\": %.2f, \"swoosh_calls\": %lu,\n",
         chunks ? (total / chunks) / 480.0 : 0.0, stats.linear_ms, stats.linear_calls,
         stats.linear_ms > 0.0 ? stats.linear_madds / stats.linear_ms / 1e6 : 0.0,
         stats.quant_ms, stats.gemm_ms,
         stats.gemm_ms > 0.0 ? stats.linear_madds / stats.gemm_ms / 1e6 : 0.0, stats.deq_ms,
         stats.swoosh_ms, stats.swoosh_calls);
  printf(" \"conv_ms\": %.2f, \"conv_calls\": %lu, \"conv_gmadds\": %.3f,"
         " \"conv_prep_ms\": %.2f, \"conv_gemm_ms\": %.2f,\n",
         stats.conv_ms, stats.conv_calls,
         stats.conv_gemm_ms > 0.0 ? stats.conv_madds / stats.conv_gemm_ms / 1e6 : 0.0,
         stats.conv_prep_ms, stats.conv_gemm_ms);
  printf(" \"attn_matmul_ms\": %.2f, \"attn_matmul_calls\": %lu, \"attn_gmadds\": %.3f,\n",
         stats.matmul_ms, stats.matmul_calls,
         stats.matmul_ms > 0.0 ? stats.matmul_madds / stats.matmul_ms / 1e6 : 0.0);
  printf(" \"layer_ms\": %.2f, \"layer_calls\": %lu,\n", stats.layer_ms,
         stats.layer_calls);
  printf(" \"layer_prologue_ms\": %.2f,\n", stats.layer_prologue_ms);
  printf(" \"engine_one_thread_ms\": %.2f,\n", zf_serial_ms());
  printf(" \"pool_wait_ms\": %.2f, \"pool_regions\": %.0f,\n", zf_pool_wait_ms(),
         zf_pool_wait_count());
  read_faults(&minflt1, &majflt1);
  printf(" \"minor_faults\": %lu, \"major_faults\": %lu,\n", minflt1 - minflt0, majflt1 - majflt0);
  printf(" \"pool_region_ms\": %.2f, \"layer_region_ms\": %.2f, \"frontend_region_ms\": %.2f,\n",
         zf_pool_region_ms(), stats.layer_region_ms, stats.frontend_region_ms);
  {
    double h[5];
    zf_pool_wait_hist(h);
    printf(" \"wait_us_by_len\": [%.0f, %.0f, %.0f, %.0f, %.0f], \"wait_max_us\": %.0f,\n", h[0],
           h[1], h[2], h[3], h[4], zf_pool_wait_max_us());
  }
  printf(" \"layer_gmadds\": %.3f, \"layer_gemm_gmadds\": %.3f,\n",
         stats.layer_stage_ms[9] > 0.0 ? stats.layer_madds / stats.layer_stage_ms[9] / 1e6 : 0.0,
         stats.layer_ms > 0.0 ? stats.layer_madds / stats.layer_ms / 1e6 : 0.0);
  printf(" \"layer_stages\": {");
  for (int i = 0; i < ZF_LAYER_STAGES; ++i)
    printf("%s\"%s\": %.2f", i ? ", " : "", zf_layer_stage_name(i),
           stats.layer_stage_ms[i]);
  printf("},\n");
  printf(" \"frontend_ms\": %.2f, \"frontend_calls\": %lu,\n", stats.frontend_ms,
         stats.frontend_calls);
  printf(" \"block_ms\": %.2f, \"block_calls\": %lu,\n", stats.block_ms,
         stats.block_calls);
  printf(" \"layout_ms\": %.2f, \"layout_calls\": %lu,\n", stats.layout_ms, stats.layout_calls);
  printf(" \"dw_ms\": %.2f, \"dw_kernel_ms\": %.2f, \"dw_calls\": %lu,\n", stats.dw_ms,
         stats.dw_kernel_ms, stats.dw_calls);
  printf(" \"tokens\": [");
  for (int t = 0; t < n_tokens; ++t) printf(t ? ", %d" : "%d", tokens[t]);
  printf("],\n \"token_frames\": [");
  for (int t = 0; t < n_tokens; ++t) printf(t ? ", %d" : "%d", token_frames[t]);
  printf("],\n \"token_prob\": [");
  for (int t = 0; t < n_tokens; ++t) printf(t ? ", %.3f" : "%.3f", token_info[t].prob);
  printf("],\n \"token_margin\": [");
  for (int t = 0; t < n_tokens; ++t) printf(t ? ", %.3f" : "%.3f", token_info[t].margin);
  printf("],\n \"chunk_ms\": [");
  for (int c = 0; c < chunks; ++c) printf(c ? ", %.2f" : "%.2f", chunk_ms[c]);
  printf("],\n \"chunk_wait_ms\": [");
  for (int c = 0; c < chunks; ++c) printf(c ? ", %.2f" : "%.2f", chunk_wait_ms[c]);
  printf("]}\n");

  if (tokens_path) {
    FILE *tf = fopen(tokens_path, "r");
    if (!tf) {
      fprintf(stderr, "cannot open %s\n", tokens_path);
      return 5;
    }
    int expected = 0, index = 0, mismatches = 0;
    while (fscanf(tf, "%d", &expected) == 1) {
      if (index >= n_tokens || tokens[index] != expected) ++mismatches;
      ++index;
    }
    fclose(tf);
    if (index != n_tokens) ++mismatches;
    fprintf(stderr, "token check: expected %d, produced %d, mismatches %d\n", index, n_tokens,
            mismatches);
    if (mismatches) return 6;
  }

  g_ort->ReleaseMemoryInfo(mem);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(opts);
  g_ort->ReleaseEnv(env);
  zf_ort_shutdown();
  return 0;
}
