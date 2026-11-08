/* zf_stream_check: the zf_set list through the library's stream API (zf/stream.h), pushed in
 * irregular pieces, one JSON line a clip with the tokens and a fingerprint of every chunk's
 * log-probs, so it can be compared with zf_set's output bit for bit.
 *
 * usage: zf_stream_check <model.onnx> <list.txt> <threads> <out.jsonl> [--lead N] [--piece N] [--recreate] [--max-out N] [--complete-runs] [--coreml MLMODELC SIDECAR] [--rate HZ] [--batch N] [--evidence] [--evidence-dir DIR] [--chunk-ms MS]
 *        --piece 0 (the default) pushes 1, 7, 13, 3, 29 frames in turn; N pushes N at a time
 *        A list entry ending in .pcm is raw 16 kHz float audio, without the flush: it is pushed
 *        160 samples (10 ms) at a time through zf_stream_push_audio, then zf_stream_finish_audio.
 *        --batch N runs N chunks at a time (ZfStreamOptions.batch): the same bits.
 *        --evidence sets an evidence callback and adds to each line `ev_fnv`, a fingerprint of
 *        every chunk's evidence (log-probs, logit energy, the taps, the samples, the capture
 *        statistics), and `ev_bad`, the chunks whose evidence broke a rule (log-probs other than
 *        on_chunk's, stack 0 not at twice the frames, a chunk out of order, samples missing from an
 *        audio clip). --evidence-dir DIR also writes each clip's evidence there as float32 files
 *        (<clip>.lp .lse .enc .s0 .s1 .pcm, rows in order) and <clip>.json with their widths and the
 *        capture statistics per chunk. --chunk-ms MS asks for that step (ZfStreamOptions.chunk_ms). */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zf/stream.h"

static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static uint64_t g_hash;
static uint64_t g_chunk_hash; /* the last chunk's log-probs, for the evidence to match */
/* The graph's other outputs (V4's tajweed heads) with log_probs, sorted by name: an FNV-1a per
 * output per chunk, printed as "out_fnv" in zf_set's layout, so a difference names the head and
 * the chunk. */
static ZfStream *g_stream;
static int g_n_out;
static const char *g_out_name[32];
static uint64_t *g_out_fnv;
static int g_out_chunks, g_out_cap;
static void outputs_init(ZfStream *s) {
  g_stream = s;
  g_n_out = 0;
  if (zf_stream_output_count(s) == 0) return;
  g_out_name[g_n_out++] = "log_probs";
  for (int i = 0; i < zf_stream_output_count(s) && g_n_out < 32; ++i)
    g_out_name[g_n_out++] = zf_stream_output_name(s, i);
  for (int a = 1; a < g_n_out; ++a)
    for (int b = a; b > 0 && strcmp(g_out_name[b - 1], g_out_name[b]) > 0; --b) {
      const char *t = g_out_name[b]; g_out_name[b] = g_out_name[b - 1]; g_out_name[b - 1] = t;
    }
}
static void on_chunk(void *user, const float *lp, int frames, int vocab) {
  (void)user;
  g_hash = fnv1a(g_hash, lp, (size_t)frames * vocab * sizeof(float));
  g_chunk_hash = fnv1a(1469598103934665603ull, lp, (size_t)frames * vocab * sizeof(float));
  if (!g_n_out) return;
  if ((g_out_chunks + 1) * g_n_out > g_out_cap) {
    g_out_cap = 2 * (g_out_chunks + 1) * g_n_out;
    g_out_fnv = (uint64_t *)realloc(g_out_fnv, (size_t)g_out_cap * sizeof(uint64_t));
    if (!g_out_fnv) { fprintf(stderr, "out of memory\n"); exit(3); }
  }
  for (int o = 0; o < g_n_out; ++o) {
    int rows = 0, cols = 0;
    const float *p = strcmp(g_out_name[o], "log_probs") ? zf_stream_last_output(g_stream, g_out_name[o], &rows, &cols) : lp;
    if (p == lp) rows = frames, cols = vocab;
    g_out_fnv[(size_t)g_out_chunks * g_n_out + o] =
        p ? fnv1a(1469598103934665603ull, p, (size_t)rows * cols * sizeof(float)) : 0;
  }
  ++g_out_chunks;
}
/* --outputs a,b,c: compute only these of the graph's other outputs (zf_stream_select_outputs; "none"
 * for log_probs and the states alone). The others print as FNV 0, which
 * bench/v4_conformance_check.py --subset expects of an output that was not computed. */
static const char *select_spec;
static int select_outputs(ZfStream *s, const char *spec) {
  static char buf[1024];
  const char *names[32];
  int n = 0;
  snprintf(buf, sizeof buf, "%s", spec);
  if (strcmp(buf, "none"))
    for (char *sv = NULL, *t = strtok_r(buf, ",", &sv); t && n < 32; t = strtok_r(NULL, ",", &sv)) names[n++] = t;
  if (zf_stream_select_outputs(s, names, n)) {
    fprintf(stderr, "--outputs %s: refused\n", spec);
    return -1;
  }
  return 0;
}
static void outputs_print(FILE *out) {
  if (!g_n_out) return;
  fprintf(out, ", \"out_fnv\": {");
  for (int o = 0; o < g_n_out; ++o) {
    fprintf(out, "%s\"%s\": [", o ? ", " : "", g_out_name[o]);
    for (int k = 0; k < g_out_chunks; ++k)
      fprintf(out, k ? ", \"%016llx\"" : "\"%016llx\"", (unsigned long long)g_out_fnv[(size_t)k * g_n_out + o]);
    fprintf(out, "]");
  }
  fprintf(out, "}");
}

/* --evidence: the fingerprint of every chunk's evidence, its rule breaks, and the optional dump */
static uint64_t g_ev_hash;
static int g_ev_bad, g_ev_audio, g_ev_advance;
static long long g_ev_next;
static FILE *g_dump[6], *g_dump_json;
static const char *const kDumpExt[6] = {"lp", "lse", "enc", "s0", "s1", "pcm"};
static int g_dims[6];

static void dump(int which, const float *p, size_t floats, int dim) {
  if (!g_dump[which] || !p) return;
  fwrite(p, sizeof(float), floats, g_dump[which]);
  g_dims[which] = dim;
}

static void on_evidence(void *user, const ZfEvidence *e) {
  (void)user;
  const size_t lp = (size_t)e->frames * e->vocab;
  int bad = fnv1a(1469598103934665603ull, e->log_probs, lp * sizeof(float)) != g_chunk_hash;
  bad |= e->chunk != g_ev_next++;
  bad |= e->stack0 && e->stack0_frames != 2 * e->frames;
  bad |= g_ev_audio && (!e->samples || e->n_samples != g_ev_advance * 160 || !e->capture.has_samples);
  bad |= e->struct_size != sizeof(ZfEvidence);
  g_ev_bad += bad;
  uint64_t h = g_ev_hash;
  h = fnv1a(h, e->log_probs, lp * sizeof(float));
  if (e->logit_lse) h = fnv1a(h, e->logit_lse, (size_t)e->frames * sizeof(float));
  if (e->enc) h = fnv1a(h, e->enc, (size_t)e->frames * e->enc_dim * sizeof(float));
  if (e->stack0) h = fnv1a(h, e->stack0, (size_t)e->stack0_frames * e->stack0_dim * sizeof(float));
  if (e->stack1) h = fnv1a(h, e->stack1, (size_t)e->frames * e->stack1_dim * sizeof(float));
  if (e->samples) h = fnv1a(h, e->samples, (size_t)e->n_samples * sizeof(float));
  h = fnv1a(h, &e->capture, sizeof e->capture);
  g_ev_hash = h;
  dump(0, e->log_probs, lp, e->vocab);
  dump(1, e->logit_lse, (size_t)e->frames, 1);
  dump(2, e->enc, (size_t)e->frames * e->enc_dim, e->enc_dim);
  dump(3, e->stack0, (size_t)e->stack0_frames * e->stack0_dim, e->stack0_dim);
  dump(4, e->stack1, (size_t)e->frames * e->stack1_dim, e->stack1_dim);
  dump(5, e->samples, (size_t)e->n_samples, 1);
  if (g_dump_json) {
    const ZfCaptureStats *c = &e->capture;
    fprintf(g_dump_json, "%s[%lld, %d, %d, %.9g, %.9g, %.9g, %d, %d, %d, %.9g, %.9g]", e->chunk ? ", " : "",
            e->chunk, e->frames, c->has_samples, c->rms, c->peak, c->dc, c->clipped, c->zero_runs,
            c->zero_samples, c->noise_floor, c->bandwidth_hz);
  }
}

static void dump_open(const char *dir, const char *clip_path) {
  if (!dir) return;
  const char *base = strrchr(clip_path, '/');
  base = base ? base + 1 : clip_path;
  char path[4096];
  const int stem = (int)(strrchr(base, '.') ? strrchr(base, '.') - base : (long)strlen(base));
  for (int k = 0; k < 6; ++k) {
    snprintf(path, sizeof path, "%s/%.*s.%s", dir, stem, base, kDumpExt[k]);
    g_dump[k] = fopen(path, "wb");
    g_dims[k] = 0;
  }
  snprintf(path, sizeof path, "%s/%.*s.json", dir, stem, base);
  g_dump_json = fopen(path, "w");
  if (g_dump_json) fprintf(g_dump_json, "{\"capture\": [");
}

static void dump_close(void) {
  if (g_dump_json) {
    fprintf(g_dump_json, "], \"dims\": {");
    for (int k = 0; k < 6; ++k) fprintf(g_dump_json, "%s\"%s\": %d", k ? ", " : "", kDumpExt[k], g_dims[k]);
    fprintf(g_dump_json, "}, \"capture_fields\": [\"chunk\", \"frames\", \"has_samples\", \"rms\", \"peak\", "
                         "\"dc\", \"clipped\", \"zero_runs\", \"zero_samples\", \"noise_floor\", \"bandwidth_hz\"]}\n");
    fclose(g_dump_json);
    g_dump_json = NULL;
  }
  for (int k = 0; k < 6; ++k)
    if (g_dump[k]) fclose(g_dump[k]), g_dump[k] = NULL;
}

/* Every phoneme's fields must be self-consistent: a run of at least one frame that ends at or after
 * it starts, a peak no lower than the first frame, a runner-up that is another id, and `complete`
 * exactly when runs are being completed. Returns how many phonemes break a rule. */
static long g_tokens, g_multi, g_peak_higher;
static int check_fields(const ZfToken *t, int n, int complete_runs) {
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const ZfToken *k = &t[i];
    int ok = k->run >= 1 && k->end_frame >= k->frame && k->peak_prob >= k->prob && k->runner_up != k->token &&
             k->runner_up >= 0 && k->complete == complete_runs && k->peak_margin <= k->peak_prob;
    if (!complete_runs) ok = ok && k->peak_prob == k->prob && k->peak_margin == k->margin && k->end_frame == k->frame;
    if (complete_runs) ok = ok && k->end_frame - k->frame + 1 == k->run && (i + 1 == n || t[i + 1].frame > k->end_frame);
    bad += !ok;
    g_tokens++;
    g_multi += k->run > 1;
    g_peak_higher += k->peak_prob > k->prob;
  }
  return bad;
}

static int room(int n_tok, int max_out) {
  const int left = 4096 - n_tok;
  return max_out >= 0 && max_out < left ? max_out : left;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <model.onnx> <list.txt> <threads> <out.jsonl> [--lead N] [--piece N] [--recreate] [--max-out N] [--complete-runs] [--coreml MLMODELC SIDECAR] [--rate HZ] [--batch N] [--outputs a,b|none] [--no-pin]\n", argv[0]);
    return 1;
  }
  ZfStreamOptions opt = zf_stream_default_options();
  opt.on_chunk = on_chunk;
  opt.threads = atoi(argv[3]);
  int piece = 0, recreate = 0, max_out = -1, evidence = 0;
  const char *evidence_dir = NULL;
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--lead") && i + 1 < argc) opt.prefetch_lead = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--piece") && i + 1 < argc) piece = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--recreate")) recreate = 1;
    /* --max-out N: room for only N phonemes per push; the rest must come out of zf_stream_read */
    else if (!strcmp(argv[i], "--max-out") && i + 1 < argc) max_out = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--complete-runs")) opt.complete_runs = 1;
    else if (!strcmp(argv[i], "--rate") && i + 1 < argc) opt.sample_rate = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) opt.batch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--evidence")) evidence = 1;
    else if (!strcmp(argv[i], "--evidence-dir") && i + 1 < argc) evidence = 1, evidence_dir = argv[++i];
    else if (!strcmp(argv[i], "--chunk-ms") && i + 1 < argc) opt.chunk_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--coreml") && i + 2 < argc) i += 2;
    else if (!strcmp(argv[i], "--outputs") && i + 1 < argc) select_spec = argv[++i];
    else if (!strcmp(argv[i], "--no-pin")) opt.no_pin = 1;
    else if (!strcmp(argv[i], "--pin")) opt.no_pin = 0; /* the phone default, forced on any platform */
  }
  if (evidence) opt.on_evidence = on_evidence;
  const char *error = NULL;
  /* --coreml DIR SIDECAR: the weights come from a CoreML model; argv[1] is then ignored */
  const char *coreml = NULL, *sidecar = NULL;
  for (int i = 5; i + 2 < argc + 1; ++i)
    if (!strcmp(argv[i], "--coreml") && i + 2 < argc) { coreml = argv[i + 1]; sidecar = argv[i + 2]; }
  ZfStream *s = coreml ? zf_stream_create_from_coreml(coreml, sidecar, &opt, &error) : zf_stream_create(argv[1], &opt, &error);
  if (!s) { fprintf(stderr, "stream: %s\n", error ? error : "?"); return 2; }
  outputs_init(s);
  FILE *list = fopen(argv[2], "r"), *out = fopen(argv[4], "w");
  if (!list || !out) { perror("files"); return 1; }
  char line[4096];
  static const int pattern[] = {1, 7, 13, 3, 29};
  int bad_fields = 0;
  ZfToken toks[4096];
  if (select_spec && select_outputs(s, select_spec)) return 2;
  while (fgets(line, sizeof line, list)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ')) line[--len] = 0;
    if (!len) continue;
    FILE *f = fopen(line, "rb");
    if (!f) { perror(line); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    const int frames = (int)(bytes / (80 * 4));
    float *feats = (float *)malloc((size_t)bytes);
    if (!feats || fread(feats, 1, (size_t)bytes, f) != (size_t)bytes) { perror("read"); return 1; }
    fclose(f);
    /* --recreate: a new stream per clip instead of a reset, which must give the same bits */
    if (recreate) {
      zf_stream_destroy(s);
      s = coreml ? zf_stream_create_from_coreml(coreml, sidecar, &opt, &error) : zf_stream_create(argv[1], &opt, &error);
      if (!s) { fprintf(stderr, "stream: %s\n", error ? error : "?"); return 2; }
      outputs_init(s);
      if (select_spec && select_outputs(s, select_spec)) return 2;
    } else {
      zf_stream_reset(s);
    }
    int n_tok = 0, chunks = 0, k = 0;
    uint64_t logp_hash = 1469598103934665603ull;
    const size_t ll = strlen(line);
    g_hash = 1469598103934665603ull;
    g_out_chunks = 0;
    const int is_pcm = ll > 4 && !strcmp(line + ll - 4, ".pcm");
    g_ev_hash = 1469598103934665603ull;
    g_ev_bad = 0;
    g_ev_next = 0;
    g_ev_audio = is_pcm;
    g_ev_advance = zf_stream_advance(s);
    dump_open(evidence_dir, line);
    char ev[96] = "";
    if (is_pcm) {
      const int samples = (int)(bytes / 4);
      for (int at = 0; at < samples; at += 160) {
        const int n = samples - at < 160 ? samples - at : 160;
        const int got = zf_stream_push_audio(s, feats + at, n, toks + n_tok, room(n_tok, max_out));
        if (got < 0) { fprintf(stderr, "audio push failed\n"); return 3; }
        n_tok += got;
      }
      const int got = zf_stream_finish_audio(s, toks + n_tok, room(n_tok, max_out));
      if (got < 0) { fprintf(stderr, "finish failed\n"); return 3; }
      n_tok += got;
      n_tok += zf_stream_read(s, toks + n_tok, 4096 - n_tok);
      bad_fields += check_fields(toks, n_tok, opt.complete_runs);
      dump_close();
      if (evidence)
        snprintf(ev, sizeof ev, ", \"ev_fnv\": \"%016llx\", \"ev_bad\": %d", (unsigned long long)g_ev_hash, g_ev_bad);
      fprintf(out, "{\"features\": \"%s\", \"chunks\": %lld, \"tokens\": [", line, zf_stream_chunks(s));
      for (int t = 0; t < n_tok; ++t) fprintf(out, t ? ", %d" : "%d", toks[t].token);
      fprintf(out, "], \"logp_fnv\": \"%016llx\"%s", (unsigned long long)g_hash, ev);
      outputs_print(out);
      fprintf(out, "}\n");
      free(feats);
      continue;
    }
    for (int at = 0; at < frames;) {
      int n = piece > 0 ? piece : pattern[k++ % 5];
      if (n > frames - at) n = frames - at;
      /* push, and fingerprint every chunk this push ran; one push can complete several */
      const int got = zf_stream_push(s, feats + (size_t)at * 80, n, toks + n_tok, room(n_tok, max_out));
      if (got < 0) { fprintf(stderr, "push failed\n"); return 3; }
      n_tok += got;
      at += n;
    }
    n_tok += zf_stream_finish(s, toks + n_tok, 4096 - n_tok);
    n_tok += zf_stream_read(s, toks + n_tok, 4096 - n_tok);
    bad_fields += check_fields(toks, n_tok, opt.complete_runs);
    chunks = (int)zf_stream_chunks(s);
    logp_hash = g_hash;
    dump_close();
    if (evidence)
      snprintf(ev, sizeof ev, ", \"ev_fnv\": \"%016llx\", \"ev_bad\": %d", (unsigned long long)g_ev_hash, g_ev_bad);
    fprintf(out, "{\"features\": \"%s\", \"chunks\": %d, \"tokens\": [", line, chunks);
    for (int t = 0; t < n_tok; ++t) fprintf(out, t ? ", %d" : "%d", toks[t].token);
    fprintf(out, "], \"logp_fnv\": \"%016llx\"%s", (unsigned long long)logp_hash, ev);
    outputs_print(out);
    fprintf(out, "}\n");
    free(feats);
  }
  fclose(list);
  fclose(out);
  zf_stream_destroy(s);
  fprintf(stderr, "phonemes %ld, runs longer than a frame %ld, peak above the first frame %ld\n", g_tokens, g_multi, g_peak_higher);
  if (bad_fields) fprintf(stderr, "%d phonemes with inconsistent fields\n", bad_fields);
  return bad_fields ? 4 : 0;
}
