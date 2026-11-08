/*
 * zf_set: decode a whole benchmark set through the streaming graph, one session for all of it.
 *
 * zf_run answers "what does this clip say and how long did it take". This answers the same for a
 * set, and exists because the set is 600 clips and the session costs 888 ms to create: launching
 * zf_run per clip would spend 35 minutes of a 4 configuration sweep doing nothing but loading the
 * same model. Here the session, the input and output bindings, the state buffers and the feature
 * buffer are all made once and reused, so the cost per clip is the audio and nothing else.
 *
 * What has to be per clip is the STATE, and all of it: every streaming cache the graph carries,
 * the engine's own caches behind the custom ops, and the decoder. A clip that inherits the
 * previous clip's caches is not a stream any more, it is a continuation of the wrong audio, so the
 * reset is the one thing in this file worth checking twice.
 *
 * Output is one JSON line per clip, tokens as vocabulary ids, which is what the scorer reads.
 *
 * usage: zf_set <model.onnx> <list.txt> <threads> <out.jsonl> [--no-accel] [--warmup N]
 *        [--prefetch STEP]  run each window's front end as its frames arrive, STEP frames at a time
 *        where list.txt has one path to a float32 [frames, 80] feature file per line
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "onnxruntime_c_api.h"
#include "zf/decode.h"
#include "zf/layer.h"
#include "zf/ort_ops.h"
#include "zf/pool.h"

#define BINS 80
#define MAX_TOKENS_PER_CHUNK 32

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

#define CHECK(expr)           \
  do {                        \
    OrtStatus *_st = (expr);  \
    if (_st) die(#expr, _st); \
  } while (0)

/* FNV-1a over raw bytes: a fingerprint of every log-prob and every carried state, so two devices
 * can be compared bit for bit without moving the tensors themselves. */
static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* The graph's blank_id: a number (0 for V4), or "last" / absent for the vocabulary's last id (250
 * for v3.1, 238 for b8): -1 then. */
static int meta_blank(OrtModelMetadata *md, OrtAllocator *alloc) {
  char *val = NULL;
  OrtStatus *st = g_ort->ModelMetadataLookupCustomMetadataMap(md, alloc, "blank_id", &val);
  if (st) { g_ort->ReleaseStatus(st); return -1; }
  if (!val) return -1;
  const int v = (val[0] >= '0' && val[0] <= '9') ? atoi(val) : -1;
  alloc->Free(alloc, val);
  return v;
}

static int meta_int(OrtModelMetadata *md, OrtAllocator *alloc, const char *key) {
  char *val = NULL;
  OrtStatus *st = g_ort->ModelMetadataLookupCustomMetadataMap(md, alloc, key, &val);
  if (st) { g_ort->ReleaseStatus(st); return 0; }
  if (!val) return 0;
  const int v = atoi(val);
  alloc->Free(alloc, val);
  return v > 0 ? v : 0;
}

/* The list, read once. Paths are trimmed of trailing whitespace and blank lines are skipped. */
static char **read_list(const char *path, int *count_out) {
  FILE *f = fopen(path, "r");
  if (!f) { perror("list"); exit(1); }
  int cap = 64, n = 0;
  char **out = (char **)malloc((size_t)cap * sizeof(char *));
  char line[4096];
  while (fgets(line, sizeof line, f)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' '))
      line[--len] = 0;
    if (!len) continue;
    if (n == cap) { cap *= 2; out = (char **)realloc(out, (size_t)cap * sizeof(char *)); }
    out[n] = (char *)malloc(len + 1);
    memcpy(out[n], line, len + 1);
    ++n;
  }
  fclose(f);
  *count_out = n;
  return out;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model.onnx> <list.txt> <threads> <out.jsonl> [--no-accel]"
                    " [--warmup N]\n", argv[0]);
    return 1;
  }
  const char *model_path = argv[1];
  const char *list_path = argv[2];
  const int threads = atoi(argv[3]);
  const char *out_path = argv[4];
  int accel = 1, warmup = 2, prefetch = 0;
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--no-accel")) accel = 0;
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--prefetch") && i + 1 < argc) prefetch = atoi(argv[++i]);
  }

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) { fprintf(stderr, "onnxruntime API %d unsupported\n", ORT_API_VERSION); return 2; }

  int n_clips = 0;
  char **clips = read_list(list_path, &n_clips);
  if (!n_clips) { fprintf(stderr, "the list is empty\n"); return 1; }

  OrtEnv *env = NULL;
  CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "zf", &env));
  OrtSessionOptions *opts = NULL;
  CHECK(g_ort->CreateSessionOptions(&opts));
  CHECK(g_ort->SetIntraOpNumThreads(opts, getenv("ZF_ORT_POOL") ? threads : 1));
  CHECK(g_ort->SetInterOpNumThreads(opts, 1));
  CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
  if (accel) CHECK(zf_ort_register(g_ort, opts, threads));

  const double t_load = now_ms();
  OrtSession *session = NULL;
  CHECK(g_ort->CreateSession(env, model_path, opts, &session));
  const double load_ms = now_ms() - t_load;

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
      !state || !state_bytes) { fprintf(stderr, "out of memory\n"); return 3; }

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
    /* The published graph leaves its batch dimension symbolic, which comes back as -1 and which
     * CreateTensorWithDataAsOrtValue refuses. A stream is one utterance, so pin any dimension the
     * graph did not fix to 1; the rewritten graph has all of them fixed already and is unchanged
     * by this. */
    size_t elems = 1;
    for (size_t d = 0; d < in_dims[i]; ++d) {
      if (in_shapes[i][d] <= 0) in_shapes[i][d] = 1;
      elems *= (size_t)in_shapes[i][d];
    }
    state_bytes[i] = elems * ((in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? 8 : 4);
    state[i] = calloc(1, state_bytes[i]);
    if (!state[i]) { fprintf(stderr, "out of memory for %s\n", in_names[i]); return 3; }
  }
  for (size_t o = 0; o < n_out; ++o)
    CHECK(g_ort->SessionGetOutputName(session, o, alloc, &out_names[o]));

  size_t logits_index = 0;
  for (size_t o = 0; o < n_out; ++o)
    if (!strcmp(out_names[o], "log_probs")) logits_index = o;
  /* Every output that is not a carried state (log_probs and, for V4, the tajweed heads), sorted by
   * name so graphs that order their outputs differently print the same keys: each gets an FNV-1a per
   * chunk ("out_fnv"), so a difference shows which head and which chunk. */
  size_t n_head = 0;
  size_t *head_idx = (size_t *)calloc(n_out, sizeof(size_t));
  for (size_t o = 0; o < n_out; ++o)
    if (strncmp(out_names[o], "new_", 4) != 0 && strncmp(out_names[o], "tap_", 4) != 0)
      head_idx[n_head++] = o;
  for (size_t a = 1; a < n_head; ++a)
    for (size_t b = a; b > 0 && strcmp(out_names[head_idx[b - 1]], out_names[head_idx[b]]) > 0; --b) {
      const size_t t = head_idx[b]; head_idx[b] = head_idx[b - 1]; head_idx[b - 1] = t;
    }
  uint64_t *head_fnv = NULL;
  int head_fnv_cap = 0;

  /* Geometry from the model, the same contract zf_run reads: T is the window and
   * decode_chunk_len the advance, so a graph rebuilt at another chunk length needs no change. */
  int window = 0, shift = 0, blank_meta = -1;
  {
    OrtModelMetadata *md = NULL;
    CHECK(g_ort->SessionGetModelMetadata(session, &md));
    window = meta_int(md, alloc, "T");
    shift = meta_int(md, alloc, "decode_chunk_len");
    blank_meta = meta_blank(md, alloc);
    g_ort->ReleaseModelMetadata(md);
    int declared = 0;
    for (size_t i = 0; i < n_in; ++i)
      if (!strcmp(in_names[i], "x") && in_dims[i] >= 2) declared = (int)in_shapes[i][1];
    if (!window) window = declared;
    if (declared > 0 && window != declared) {
      fprintf(stderr, "geometry: metadata T is %d but x declares %d\n", window, declared);
      return 2;
    }
    if (window <= 0 || shift <= 0) { fprintf(stderr, "geometry: missing T/decode_chunk_len\n"); return 2; }
  }

  OrtMemoryInfo *mem = NULL;
  CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));

  /* One buffer big enough for the longest clip, grown on demand and never shrunk. */
  size_t feat_cap = 0;
  float *feats = NULL;
  int tok_cap = 0;
  ZfToken *token_info = NULL;

  FILE *out = fopen(out_path, "w");
  if (!out) { perror("out"); return 1; }

  /* The governor idles these cores at 546 MHz, so the first clip would otherwise measure the
   * clock coming up rather than the model. Warm on the first clip's first chunk, then reset. */
  const double t_all = now_ms();
  long total_chunks = 0;

  for (int c = 0; c < n_clips; ++c) {
    FILE *ff = fopen(clips[c], "rb");
    if (!ff) { fprintf(stderr, "cannot open %s\n", clips[c]); continue; }
    fseek(ff, 0, SEEK_END);
    const long bytes = ftell(ff);
    fseek(ff, 0, SEEK_SET);
    const int frames = (int)(bytes / (BINS * 4));
    if ((size_t)bytes > feat_cap) {
      feats = (float *)realloc(feats, (size_t)bytes);
      if (!feats) { fprintf(stderr, "out of memory for features\n"); return 3; }
      feat_cap = (size_t)bytes;
    }
    if (fread(feats, 1, (size_t)bytes, ff) != (size_t)bytes) { fclose(ff); continue; }
    fclose(ff);

    const int chunks = frames >= window ? (frames - window) / shift + 1 : 0;
    if (chunks * (int)n_head > head_fnv_cap) {
      head_fnv_cap = chunks * (int)n_head;
      head_fnv = (uint64_t *)realloc(head_fnv, (size_t)head_fnv_cap * sizeof(uint64_t));
      if (!head_fnv) { fprintf(stderr, "out of memory for hashes\n"); return 3; }
    }
    if (chunks <= 0) {
      fprintf(out, "{\"features\": \"%s\", \"tokens\": [], \"chunks\": 0}\n", clips[c]);
      continue;
    }
    if (chunks * MAX_TOKENS_PER_CHUNK > tok_cap) {
      tok_cap = chunks * MAX_TOKENS_PER_CHUNK;
      token_info = (ZfToken *)realloc(token_info, (size_t)tok_cap * sizeof(ZfToken));
      if (!token_info) { fprintf(stderr, "out of memory for tokens\n"); return 3; }
    }

    /* Every cache to zero before the clip: the graph's, the engine's, and the decoder's. */
    for (size_t i = 0; i < n_in; ++i) memset(state[i], 0, state_bytes[i]);
    if (accel) zf_ort_reset_state();
    ZfDecoder *decoder = NULL;
    int n_tokens = 0;
    uint64_t logp_hash = 1469598103934665603ull, state_hash = 1469598103934665603ull;

    if (c == 0 && warmup > 0) {
      for (int w = 0; w < warmup; ++w) {
        for (size_t i = 0; i < n_in; ++i) {
          if (!strcmp(in_names[i], "x"))
            memcpy(state[i], feats, (size_t)window * BINS * sizeof(float));
          CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem, state[i], state_bytes[i], in_shapes[i],
                                                      in_dims[i], in_types[i], &in_vals[i]));
        }
        CHECK(g_ort->Run(session, NULL, (const char *const *)in_names,
                         (const OrtValue *const *)in_vals, n_in,
                         (const char *const *)out_names, n_out, out_vals));
        for (size_t o = 0; o < n_out; ++o) { g_ort->ReleaseValue(out_vals[o]); out_vals[o] = NULL; }
        for (size_t i = 0; i < n_in; ++i) { g_ort->ReleaseValue(in_vals[i]); in_vals[i] = NULL; }
      }
      for (size_t i = 0; i < n_in; ++i) memset(state[i], 0, state_bytes[i]);
      if (accel) zf_ort_reset_state();
    }

    for (int k = 0; k < chunks; ++k) {
      for (size_t i = 0; i < n_in; ++i) {
        if (!strcmp(in_names[i], "x"))
          memcpy(state[i], feats + (size_t)k * shift * BINS,
                 (size_t)window * BINS * sizeof(float));
        CHECK(g_ort->CreateTensorWithDataAsOrtValue(mem, state[i], state_bytes[i], in_shapes[i],
                                                    in_dims[i], in_types[i], &in_vals[i]));
      }
      /* --prefetch STEP: feed the window's front end as its frames would arrive, STEP at a time,
       * from the overlap the previous window already had; the Run then only finishes it. */
      if (accel && prefetch > 0)
        for (int v = window - shift; v < window; v += prefetch)
          zf_ort_frontend_prefetch(feats + (size_t)k * shift * BINS, window, v);
      CHECK(g_ort->Run(session, NULL, (const char *const *)in_names,
                       (const OrtValue *const *)in_vals, n_in, (const char *const *)out_names,
                       n_out, out_vals));

      float *logp = NULL;
      CHECK(g_ort->GetTensorMutableData(out_vals[logits_index], (void **)&logp));
      OrtTensorTypeAndShapeInfo *li = NULL;
      CHECK(g_ort->GetTensorTypeAndShape(out_vals[logits_index], &li));
      int64_t ldims[3] = {1, 0, 0};
      CHECK(g_ort->GetDimensions(li, ldims, 3));
      g_ort->ReleaseTensorTypeAndShapeInfo(li);
      if (!decoder) {
        /* blank is the graph's blank_id, else the vocabulary's last id: 250 for v3.1, 238 for b8 */
        decoder = zf_decoder_create((int)ldims[2], blank_meta >= 0 ? blank_meta : (int)ldims[2] - 1);
        if (!decoder) { fprintf(stderr, "cannot start the decoder\n"); return 3; }
      }
      n_tokens += zf_decoder_push(decoder, logp, (int)ldims[1], token_info + n_tokens,
                                  tok_cap - n_tokens);
      logp_hash = fnv1a(logp_hash, logp, (size_t)(ldims[1] * ldims[2]) * sizeof(float));
      for (size_t h = 0; h < n_head; ++h) {
        OrtValue *v = out_vals[head_idx[h]];
        void *data = NULL;
        size_t count = 0;
        OrtTensorTypeAndShapeInfo *hi = NULL;
        CHECK(g_ort->GetTensorMutableData(v, &data));
        CHECK(g_ort->GetTensorTypeAndShape(v, &hi));
        CHECK(g_ort->GetTensorShapeElementCount(hi, &count));
        g_ort->ReleaseTensorTypeAndShapeInfo(hi);
        head_fnv[(size_t)k * n_head + h] = fnv1a(1469598103934665603ull, data, count * sizeof(float));
      }

      for (size_t o = 0; o < n_out; ++o) {
        if (strncmp(out_names[o], "new_", 4) != 0) continue;
        for (size_t i = 0; i < n_in; ++i) {
          if (strcmp(out_names[o] + 4, in_names[i]) != 0) continue;
          void *src = NULL;
          CHECK(g_ort->GetTensorMutableData(out_vals[o], &src));
          memcpy(state[i], src, state_bytes[i]);
          state_hash = fnv1a(state_hash, src, state_bytes[i]);
          break;
        }
      }
      for (size_t o = 0; o < n_out; ++o) { g_ort->ReleaseValue(out_vals[o]); out_vals[o] = NULL; }
      for (size_t i = 0; i < n_in; ++i) { g_ort->ReleaseValue(in_vals[i]); in_vals[i] = NULL; }
    }
    total_chunks += chunks;

    fprintf(out, "{\"features\": \"%s\", \"chunks\": %d, \"tokens\": [", clips[c], chunks);
    for (int t = 0; t < n_tokens; ++t) fprintf(out, t ? ", %d" : "%d", token_info[t].token);
    fprintf(out, "], \"prob\": [");
    for (int t = 0; t < n_tokens; ++t) fprintf(out, t ? ", %.4f" : "%.4f", token_info[t].prob);
    /* The margin over the runner-up at the same frame, at full precision because the question it
     * answers is whether a disagreement with another implementation sits inside float noise. A
     * margin of 1e-6 means the two logits were tied to the last bits and the argmax was decided by
     * rounding; a margin of 0.2 means the implementations genuinely disagree. */
    fprintf(out, "], \"margin\": [");
    for (int t = 0; t < n_tokens; ++t) fprintf(out, t ? ", %.9g" : "%.9g", token_info[t].margin);
    fprintf(out, "], \"logp_fnv\": \"%016llx\", \"state_fnv\": \"%016llx\"",
            (unsigned long long)logp_hash, (unsigned long long)state_hash);
    if (n_head > 1) {
      fprintf(out, ", \"out_fnv\": {");
      for (size_t h = 0; h < n_head; ++h) {
        fprintf(out, "%s\"%s\": [", h ? ", " : "", out_names[head_idx[h]]);
        for (int k = 0; k < chunks; ++k)
          fprintf(out, k ? ", \"%016llx\"" : "\"%016llx\"",
                  (unsigned long long)head_fnv[(size_t)k * n_head + h]);
        fprintf(out, "]");
      }
      fprintf(out, "}");
    }
    fprintf(out, "}\n");
    fflush(out);
    if (decoder) zf_decoder_destroy(decoder);

    if ((c + 1) % 25 == 0)
      fprintf(stderr, "  %d/%d  %.0fs\n", c + 1, n_clips, (now_ms() - t_all) / 1e3);
  }

  fclose(out);
  const double total_ms = now_ms() - t_all;
  const double wait_ms = zf_pool_wait_ms();
  /* Peak resident memory of the whole run (model, arena, buffers), from the kernel: Android's shell
   * has no `time -v`, and a sample of VmRSS from outside misses the peak. */
  long peak_kb = -1;
  {
    FILE *st = fopen("/proc/self/status", "r");
    char line[256];
    while (st && fgets(line, sizeof line, st))
      if (!strncmp(line, "VmHWM:", 6)) peak_kb = atol(line + 6);
    if (st) fclose(st);
  }
  fprintf(stderr, "{\"clips\": %d, \"chunks\": %ld, \"load_ms\": %.1f, \"total_s\": %.1f,"
                  " \"pool_wait_s\": %.1f, \"wait_fraction\": %.3f, \"peak_rss_kb\": %ld}\n",
          n_clips, total_chunks, load_ms, total_ms / 1e3, wait_ms / 1e3,
          total_ms > 0.0 ? wait_ms / total_ms : 0.0, peak_kb);

  /* Loud, because this is the one measurement error that has cost this project the most time.
   *
   * The engine pins one thread per core and has no spare, so anything else wanting CPU shows up as
   * the calling thread waiting at joins rather than as slower arithmetic. A settled phone runs this
   * at a wait fraction near 0.04; the same binary on the same stream, started while the phone was
   * still busy, read 0.47 and reported a mean chunk of 162.9 ms against a true 84.2, with a p95 of
   * 421.8 against 90.4. Nothing about the engine changed between those two runs.
   *
   * Any timing taken above this threshold is measuring the rest of the phone. */
  if (total_ms > 0.0 && wait_ms / total_ms > 0.15)
    fprintf(stderr, "WARNING: %.0f%% of the run was spent waiting at joins. Something else on this "
                    "device was using the cores, so these timings are not the engine's. Check top "
                    "and re-run on a settled phone; token output is unaffected.\n",
            100.0 * wait_ms / total_ms);
  return 0;
}
