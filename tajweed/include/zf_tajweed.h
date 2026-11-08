/* zf_tajweed.h: TajweedFormer V4 in the exact engine. API version 1.
 *
 * One header for the SDKs (Swift, Kotlin through JNI, WebAssembly): open the model folder, stream
 * audio through the exact engine, keep every head the network has, and score a recitation of a
 * known ayah against its expected tj3 sequence with the calibrated tajweed checks.
 *
 *   ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
 *   ZfTajweedModel *m = zf_tajweed_model_open("sdk/v4.0", &o, &err);   context NULL: the default
 *   ZfTajweedStream *s = zf_tajweed_stream_create(m, &err);
 *   zf_tajweed_stream_push_audio(s, samples, n);                         as audio arrives
 *   zf_tajweed_stream_finish(s);                                         end of the take
 *   ZfTajweedHeads h; zf_tajweed_stream_heads(s, &h);
 *   ZfTajweedConfig *c = zf_tajweed_config_create(m);
 *   zf_tajweed_config_set_remote(c, remote_json, len);                   portal block, optional
 *   ZfTajweedResult *r = zf_tajweed_score(m, c, &h, surah, ayah, &err);
 *   const char *json = zf_tajweed_result_json(r);                        or the typed accessors
 *
 * The model folder (sdk/v4.0):
 *   calibration.json          thresholds, presets, madd cuts (quranlab.tajweed-calibration/1)
 *   tokens.txt                tj3, 38 entries, blank 0
 *   expected_tj3.zfx          the expected sequence of every ayah (tools/zfexpected.py)
 *   <context>/                one folder per shipped context (c16_l64, c24_l72): model_engine2.onnx,
 *                             model_engine2.onnx.zfg, model_engine2.onnx.zfg.bin, frontend.zfw,
 *                             layers.zfw
 *
 * Exactness: the heads are the same bits, chunk by chunk, as ONNX Runtime's int8 reference on the
 * same graph (the engine's definition of exact). The scorer is double precision with fixed
 * summation order, so the same heads give the same verdicts on every platform.
 *
 * Threads: a model is immutable after open and may be shared by any number of threads. A stream,
 * a config and a result belong to one thread at a time. Only one stream may be live per process
 * (the engine's rule, zf/stream.h); scoring needs no stream and may run anywhere.
 *
 * Every function that returns a pointer returns NULL on failure and, when `error` is not NULL,
 * sets it to a static message. Every count-returning function returns -1 on failure.
 *
 * This header is frozen for API version 1: structs and enums are only ever appended to under a new
 * ZF_TAJWEED_API_VERSION, and the struct_size fields say which version a caller was built against.
 *
 * API version 2 (additive; every API 1 declaration unchanged): zf_tajweed_config_set_backend (which
 * calibration backend_overrides apply), the features "madd", "qalqalah" and "gemination" in
 * zf_tajweed_config_set_class_preset / _clear_class and under the remote block's "overrides" (a preset
 * only), and resolution exactly as the shared settings rules (zf_tajweed_config_json gives
 * its output layout, with "dropped" and "refused").
 *
 * API version 3 (additive; every API 1 and 2 declaration unchanged): the per-word score: for every word of the
 * scored path, the largest clipped fused LLR over every single-unit edit inside the word, flagged when it
 * reaches the word threshold. Word settings: a word preset ("strict", "general", "beginner") or a custom
 * threshold (clamped to calibration's custom_range, never below its floor), app > remote ("word" in the
 * block) > calibration's default_preset; the check "word" (ZF_TJ_CHECK_WORD) switches it as any other
 * check. Results: zf_tajweed_result_words, zf_tajweed_result_path, and "words" in the result's JSON.
 * API 3 also has, for every stream and model: which heads a stream computes (zf_tajweed_stream_set_heads;
 * a NEW stream computes the scorer's heads, ZF_TAJWEED_HEADS_SCORER, and ZF_TAJWEED_HEADS_ALL is API 2's
 * behaviour), the decoded layer's per-frame top-1 and runner-up (zf_tajweed_top2,
 * zf_tajweed_stream_top2), and ZfTajweedOpenOptions.no_pin (the engine's threads not pinned to cores).
 * The decoded layer goes to any depth with top-K (zf_tajweed_topk, zf_tajweed_stream_topk): every token head
 * and every class head ranked per frame, k = 1 to all of it; top-2 is top-K at k = 2. */
#ifndef ZF_TAJWEED_H
#define ZF_TAJWEED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(ZF_TAJWEED_BUILD_DLL)
#define ZF_TAJWEED_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define ZF_TAJWEED_API __attribute__((visibility("default")))
#else
#define ZF_TAJWEED_API
#endif

#define ZF_TAJWEED_API_VERSION 3
/* The vocabulary: tj3, 38 tokens, blank 0 (tokens.txt). */
#define ZF_TAJWEED_VOCAB 38
#define ZF_TAJWEED_BLANK 0

/* ---------------------------------------------------------------------------------------------
 * Names. Every name below is also accepted and returned as the string in calibration.json and in
 * the portal's tajweed block (calibration.json and the portal's tajweed block).
 * ------------------------------------------------------------------------------------------- */

/* Site classes: single-unit alternatives scored by CTC forward LLR. Indexes into
 * calibration.json's "classes", in this order. */
typedef enum {
  ZF_TJ_VOWEL_SUB = 0,      /* "vowel_sub": a / i / u swapped */
  ZF_TJ_VOWEL_DEL = 1,      /* "vowel_del": a short vowel dropped (sukun said) */
  ZF_TJ_VOWEL_INS = 2,      /* "vowel_ins": a short vowel added after a sakin letter */
  ZF_TJ_SHORT_TO_LONG = 3,  /* "short_to_long": a short vowel said long */
  ZF_TJ_LONG_TO_SHORT = 4,  /* "long_to_short": a long vowel said short */
  ZF_TJ_CONS_SUB = 5,       /* "cons_sub": a consonant swapped for a confusable one */
  ZF_TJ_CONS_DEL = 6,       /* "cons_del": a consonant dropped */
  ZF_TJ_INSERTION = 7,      /* "insertion": a unit added */
  ZF_TJ_CLASS_COUNT = 8
} ZfTajweedClass;

/* Checks the app or the portal turns on and off (the "checks" object). Each class belongs to one
 * check: letters = cons_sub, cons_del, insertion; vowels = vowel_sub, vowel_del, vowel_ins;
 * length = short_to_long, long_to_short. madd, gemination, qalqalah and lazim are their own. */
typedef enum {
  ZF_TJ_CHECK_LETTERS = 0,    /* "letters" */
  ZF_TJ_CHECK_VOWELS = 1,     /* "vowels" */
  ZF_TJ_CHECK_LENGTH = 2,     /* "length" */
  ZF_TJ_CHECK_MADD = 3,       /* "madd": madd counts from the timing head (fixed2, muttasil, free) */
  ZF_TJ_CHECK_GEMINATION = 4, /* "gemination": shadda held */
  ZF_TJ_CHECK_QALQALAH = 5,   /* "qalqalah": the release burst present */
  ZF_TJ_CHECK_LAZIM = 6,      /* "lazim": madd lazim counts */
  ZF_TJ_CHECK_COUNT = 7
} ZfTajweedCheck;

/* Presets: "strict", "balanced", "sensitive", "very_sensitive". */
typedef enum {
  ZF_TJ_PRESET_STRICT = 0,
  ZF_TJ_PRESET_BALANCED = 1,
  ZF_TJ_PRESET_SENSITIVE = 2,
  ZF_TJ_PRESET_VERY_SENSITIVE = 3,
  ZF_TJ_PRESET_COUNT = 4
} ZfTajweedPreset;

/* calibration.json's status of a class or check. OFF is reported only in a debug build whose app
 * turned it on explicitly; EXPERIMENTAL is off unless the app or the portal turns it on. */
typedef enum {
  ZF_TJ_STATUS_OFF = 0,
  ZF_TJ_STATUS_EXPERIMENTAL = 1,
  ZF_TJ_STATUS_VALIDATED = 2
} ZfTajweedStatus;

/* Where an effective setting came from. Precedence: app code > remote config > calibration. */
typedef enum {
  ZF_TJ_SOURCE_CALIBRATION = 0,
  ZF_TJ_SOURCE_REMOTE = 1,
  ZF_TJ_SOURCE_APP = 2
} ZfTajweedSource;

/* Madd classes of the expected table and calibration.json's "madd.cuts". */
typedef enum {
  ZF_TJ_MADD_NONE = 0,
  ZF_TJ_MADD_FIXED2 = 1,   /* "fixed2": tabee'i, badal, ewad, ... */
  ZF_TJ_MADD_MUTTASIL = 2, /* "muttasil" */
  ZF_TJ_MADD_FREE = 3,     /* "free": munfasil, aared, silah kubra */
  ZF_TJ_MADD_LAZIM = 4     /* "lazim" */
} ZfTajweedMaddClass;

/* Name <-> value. Unknown names give -1; out-of-range values give NULL. */
ZF_TAJWEED_API const char *zf_tajweed_class_name(int cls);
ZF_TAJWEED_API int zf_tajweed_class_from_name(const char *name);
ZF_TAJWEED_API const char *zf_tajweed_check_name(int check);
ZF_TAJWEED_API int zf_tajweed_check_from_name(const char *name);
ZF_TAJWEED_API const char *zf_tajweed_preset_name(int preset);
ZF_TAJWEED_API int zf_tajweed_preset_from_name(const char *name);
ZF_TAJWEED_API const char *zf_tajweed_madd_class_name(int madd_class);
/* The check a class belongs to. */
ZF_TAJWEED_API int zf_tajweed_class_check(int cls);
/* "1.0.0"-style version of this library (the engine's), and ZF_TAJWEED_API_VERSION as built. */
ZF_TAJWEED_API const char *zf_tajweed_version(void);
ZF_TAJWEED_API int zf_tajweed_api_version(void);

/* ---------------------------------------------------------------------------------------------
 * The model
 * ------------------------------------------------------------------------------------------- */

typedef struct ZfTajweedModel ZfTajweedModel;

typedef struct {
  size_t struct_size;   /* sizeof(ZfTajweedOpenOptions) as the caller was built */
  const char *context;  /* "c16_l64", "c24_l72", or NULL for calibration.json's default_context */
  int load_engine;      /* 1 (default): the exact engine for this context, so streams can be made.
                           0: calibration, tokens and the expected table only, for scoring heads
                           that came from elsewhere (the Neural Engine, an NPU, a server) */
  int threads;          /* engine threads; 0 = the fast cores */
  int sample_rate;      /* the rate audio is pushed at: 0 or 16000, or 8000 .. 48000 through the
                           engine's resampler */
  int batch;            /* engine chunks run together (stored audio); 0 or 1 for live audio */
  /* API 3, appended */
  int no_pin;           /* 0: the engine pins its threads one to a core (one stream on a phone).
                           1: no pinning (desktops, servers, several streams at once). The default
                           (zf_tajweed_open_default_options, and a caller whose header predates the
                           field) depends on the platform: 0 on Android and iOS, 1 on Windows,
                           Linux, macOS and WebAssembly. Setting it overrides either way. Timing
                           only, never a bit. */
} ZfTajweedOpenOptions;

/* context NULL, load_engine 1, threads 0, sample_rate 0, batch 1, no_pin by platform (0 on Android and
 * iOS, 1 on Windows, Linux, macOS and WebAssembly). */
ZF_TAJWEED_API ZfTajweedOpenOptions zf_tajweed_open_default_options(void);

/* `model_dir` is the model folder (sdk/v4.0). Reads and checks calibration.json (format, model_id,
 * the context present), tokens.txt (38 entries, sha256 against calibration's tokens_sha256 when
 * given), the expected table, and with load_engine the context's engine files. */
ZF_TAJWEED_API ZfTajweedModel *zf_tajweed_model_open(const char *model_dir,
                                                     const ZfTajweedOpenOptions *options,
                                                     const char **error);
ZF_TAJWEED_API void zf_tajweed_model_close(ZfTajweedModel *m);

ZF_TAJWEED_API const char *zf_tajweed_model_id(const ZfTajweedModel *m);       /* "v4.0" */
ZF_TAJWEED_API const char *zf_tajweed_model_context(const ZfTajweedModel *m);  /* "c16_l64" */
/* The contexts calibration.json describes, 0 .. n-1; NULL past the end. */
ZF_TAJWEED_API const char *zf_tajweed_model_context_at(const ZfTajweedModel *m, int index);
ZF_TAJWEED_API const char *zf_tajweed_model_default_context(const ZfTajweedModel *m);
/* Audio per engine chunk in ms (320 for c16, 480 for c24); 0 without the engine. */
ZF_TAJWEED_API int zf_tajweed_model_chunk_ms(const ZfTajweedModel *m);
/* tokens.txt's entry for `id` ("blank", "|", "b", "aa", ...), NULL out of range. */
ZF_TAJWEED_API const char *zf_tajweed_token_name(const ZfTajweedModel *m, int id);

/* The expected table: ayahs in a surah (0 for an unknown surah), and an ayah's expected tj3 sequence
 * (the continuing form; pausal alternatives are the scorer's business). `tokens` points into the
 * model and stays valid until it is closed. Returns the token count, or -1. */
ZF_TAJWEED_API int zf_tajweed_ayah_count(const ZfTajweedModel *m, int surah);
ZF_TAJWEED_API int zf_tajweed_expected_tokens(const ZfTajweedModel *m, int surah, int ayah,
                                              const int **tokens);
/* Word index of every expected token ([n], 0-based, word boundaries "|" belong to the word before),
 * and the number of words. Returns the token count, or -1. */
ZF_TAJWEED_API int zf_tajweed_expected_words(const ZfTajweedModel *m, int surah, int ayah,
                                             const int **word_of_token, int *n_words);

/* ---------------------------------------------------------------------------------------------
 * Heads: what the network says, kept whole for the take
 * ------------------------------------------------------------------------------------------- */

/* Every output of the network after its states (README_heads.txt), row major, frames first.
 * Two frame rates: n40 frames at 40 ms and n20 = 2 x n40 frames at 20 ms, aligned (20 ms frame
 * 2k and 2k + 1 lie in 40 ms frame k). A pointer is NULL when a source does not provide that head;
 * the scorer needs log_probs, inter1_log_probs, timing_log_probs, gem and attr_qalqalah, and skips
 * the checks whose head is missing. */
typedef struct {
  int n40, n20;
  int vocab;                      /* 38 */
  const float *log_probs;         /* [n40, 38] final CTC, log-softmax */
  const float *inter1_log_probs;  /* [n40, 38] stack-1 InterCTC, log-softmax */
  const float *inter4_log_probs;  /* [n40, 38] stack-4 InterCTC, log-softmax (diagnostic) */
  const float *id_log_probs;      /* [n20, 38] identity CTC, log-softmax */
  const float *timing_log_probs;  /* [n20, 38] H-time chain, log-softmax; 0 = pause state */
  const float *gem;               /* [n20, 2] softmax: single, held */
  const float *attr_tafkheem;     /* [n20, 3] softmax: thin, heavy, low heavy */
  const float *attr_ghunna;       /* [n20, 5] softmax: none, plain, geminate, ikhfa, idgham */
  const float *attr_qalqalah;     /* [n20, 3] softmax: none, sughra, kubra */
  const float *attr_face;         /* [n20, 4] softmax: none, imala, tasheel, sakt */
  const float *boundary;          /* [n20] logit: a unit starts here */
  const float *conf;              /* [n20] logit: the frame's label is truthful */
} ZfTajweedHeads;

/* ---------------------------------------------------------------------------------------------
 * The stream: audio in, heads kept
 * ------------------------------------------------------------------------------------------- */

typedef struct ZfTajweedStream ZfTajweedStream;

/* Called after every engine chunk with that chunk's heads (n40 = 8 and n20 = 16 at c16; 12 and 24
 * at c24), valid only during the call. Optional. */
typedef void (*ZfTajweedChunkCallback)(void *user, const ZfTajweedHeads *chunk);

/* A stream over the model's engine (the model must have been opened with load_engine). */
ZF_TAJWEED_API ZfTajweedStream *zf_tajweed_stream_create(ZfTajweedModel *m, const char **error);
ZF_TAJWEED_API void zf_tajweed_stream_destroy(ZfTajweedStream *s);
ZF_TAJWEED_API void zf_tajweed_stream_set_callback(ZfTajweedStream *s, ZfTajweedChunkCallback cb,
                                                   void *user);
/* A new take: every cache and every kept head back to the start. */
ZF_TAJWEED_API void zf_tajweed_stream_reset(ZfTajweedStream *s);
/* Mono float samples in [-1, 1] at the open options' sample_rate. Returns the chunks run, or -1. */
ZF_TAJWEED_API int zf_tajweed_stream_push_audio(ZfTajweedStream *s, const float *samples, int n);
/* The end of the take: the 1.05 s flush the model is decoded with, and every chunk it completes.
 * Returns the chunks run, or -1. Reset before the next take. */
ZF_TAJWEED_API int zf_tajweed_stream_finish(ZfTajweedStream *s);
/* Every head of the take so far. The pointers stay valid until the next push, finish or reset. */
ZF_TAJWEED_API int zf_tajweed_stream_heads(const ZfTajweedStream *s, ZfTajweedHeads *out);
/* What the model heard: greedy CTC on log_probs, blanks and repeats removed (display only; make
 * decisions from the scorer). Writes at most `max` ids and returns how many there are in all. */
ZF_TAJWEED_API int zf_tajweed_stream_greedy(const ZfTajweedStream *s, int *ids, int max);
ZF_TAJWEED_API const char *zf_tajweed_stream_last_error(const ZfTajweedStream *s);

/* API 3. The heads a stream computes, as ZfTajweedHeads' fields. A head that is not asked for is
 * not computed at all (the engine skips every node that feeds only heads nobody asked for), and its
 * ZfTajweedHeads pointer is NULL; asking for one costs exactly its own nodes, and every head that is
 * computed is the same bits whatever else is asked for. A new stream computes
 * ZF_TAJWEED_HEADS_SCORER, what zf_tajweed_score reads; ZF_TAJWEED_HEADS_ALL computes all twelve
 * (API 2's behaviour). log_probs is always computed. */
enum {
  ZF_TAJWEED_HEAD_LOG_PROBS = 1 << 0,
  ZF_TAJWEED_HEAD_INTER1 = 1 << 1,
  ZF_TAJWEED_HEAD_INTER4 = 1 << 2,
  ZF_TAJWEED_HEAD_ID = 1 << 3,
  ZF_TAJWEED_HEAD_TIMING = 1 << 4,
  ZF_TAJWEED_HEAD_GEM = 1 << 5,
  ZF_TAJWEED_HEAD_TAFKHEEM = 1 << 6,
  ZF_TAJWEED_HEAD_GHUNNA = 1 << 7,
  ZF_TAJWEED_HEAD_QALQALAH = 1 << 8,
  ZF_TAJWEED_HEAD_FACE = 1 << 9,
  ZF_TAJWEED_HEAD_BOUNDARY = 1 << 10,
  ZF_TAJWEED_HEAD_CONF = 1 << 11,
  ZF_TAJWEED_HEADS_SCORER = ZF_TAJWEED_HEAD_LOG_PROBS | ZF_TAJWEED_HEAD_INTER1 | ZF_TAJWEED_HEAD_TIMING |
                            ZF_TAJWEED_HEAD_GEM | ZF_TAJWEED_HEAD_QALQALAH,
  ZF_TAJWEED_HEADS_ALL = (1 << 12) - 1
};
/* Set at the start of a take (after create or reset, before the first push): returns 0, or -1 once
 * the take has chunks (the kept arrays would have holes). Unknown bits are ignored. */
ZF_TAJWEED_API int zf_tajweed_stream_set_heads(ZfTajweedStream *s, unsigned heads);
ZF_TAJWEED_API unsigned zf_tajweed_stream_get_heads(const ZfTajweedStream *s);
/* A head's name in the graph and README_heads.txt ("inter1_log_probs", ...) for one ZF_TAJWEED_HEAD_
 * bit, its frame period (40 or 20 ms) and values per frame; NULL for anything else. */
ZF_TAJWEED_API const char *zf_tajweed_head_info(unsigned head, int *frame_ms, int *cols);

/* API 3. The decoded layer: per 40 ms CTC frame of log_probs, the best token and the runner-up with
 * their log-probs (one scan of the row; ties go to the lower id, the engine decoder's rule, so
 * `token` is the greedy decode's argmax). margin = log_prob - runner_up_log_prob >= 0. */
typedef struct {
  int token, runner_up;
  float log_prob, runner_up_log_prob, margin;
} ZfTajweedTop2;
/* For log-probs from any source ([frames, vocab] row major, vocab >= 2). Returns frames, or -1. */
ZF_TAJWEED_API int zf_tajweed_top2(const float *log_probs, int frames, int vocab, ZfTajweedTop2 *out);
/* The take so far, kept by the stream as chunks arrive (n40 entries, valid until the next push,
 * finish or reset). Returns the count, or -1. */
ZF_TAJWEED_API int zf_tajweed_stream_top2(const ZfTajweedStream *s, const ZfTajweedTop2 **out);

/* API 3 (top-K). The decoded layer for any head with a distribution, to any depth: per frame the k best
 * entries, ranked. The token heads (log_probs, inter1, inter4, id, timing: 38 tokens, k 1 to 38, 38 being
 * the whole ranked list) and the class heads (gem 2, attr_tafkheem 3, attr_ghunna 5, attr_qalqalah 3,
 * attr_face 4 classes, k up to their count). boundary and conf are single logits and have no ranking.
 * The order is by value, highest first, and equal values by the lower id: exactly a stable argsort of
 * the negated row, so k = cols is the full argsort and k = 2 gives zf_tajweed_top2's token and
 * runner-up. ids, values and margins are [frames, k] row major (values and margins may be NULL);
 * margins[j] = values[j] - the value ranked j + 1 (even when that one is beyond k), and 0 for the last
 * entry of the row (rank cols - 1). Nothing is ranked unless it is asked for: a partial selection,
 * k passes over the row, per frame. */
/* For any [frames, cols] values (cols >= 1, 1 <= k <= cols). Returns frames, or -1. */
ZF_TAJWEED_API int zf_tajweed_topk(const float *values, int frames, int cols, int k, int *ids, float *values_out,
                                   float *margins);
/* For a head the stream keeps (a ZF_TAJWEED_HEAD_* bit: a token or class head it computes, see
 * zf_tajweed_stream_set_heads), over the whole take so far: writes the first min(frames, max_frames)
 * frames and returns the take's frame count at that head's rate (n40 or n20), or -1 (no stream, a head
 * that is not kept or has no ranking, k out of range). */
ZF_TAJWEED_API int zf_tajweed_stream_topk(const ZfTajweedStream *s, unsigned head, int k, int *ids,
                                          float *values_out, float *margins, int max_frames);

/* ---------------------------------------------------------------------------------------------
 * Configuration: calibration.json < portal (remote) < app code
 * ------------------------------------------------------------------------------------------- */

typedef struct ZfTajweedConfig ZfTajweedConfig;

/* A config holding the model's calibration defaults and no remote or app settings. */
ZF_TAJWEED_API ZfTajweedConfig *zf_tajweed_config_create(const ZfTajweedModel *m);
ZF_TAJWEED_API void zf_tajweed_config_destroy(ZfTajweedConfig *c);

/* The portal's block: either the "tajweed" object itself or a whole /v1/config
 * document holding one. Replaces the previous remote layer; NULL or an absent block clears it.
 * Validation: preset one of the four; an override has exactly one of "preset" or
 * "threshold" (finite, -20..20); checks only known keys with boolean values; context only one the
 * calibration names; unknown keys dropped; an invalid field is dropped, never the whole block.
 * Returns the number of fields dropped (0 when everything was taken), or -1 when the text is not
 * JSON (the remote layer is then cleared). */
ZF_TAJWEED_API int zf_tajweed_config_set_remote(ZfTajweedConfig *c, const char *json, size_t len);

/* App code: each wins over remote and calibration. Passing NULL (or -1 for set_check) removes the
 * app's setting so the lower layers show through. Return 0, or -1 for an unknown name or a value
 * out of range (nothing changes). */
ZF_TAJWEED_API int zf_tajweed_config_set_preset(ZfTajweedConfig *c, const char *preset);
ZF_TAJWEED_API int zf_tajweed_config_set_class_preset(ZfTajweedConfig *c, const char *cls,
                                                      const char *preset);
ZF_TAJWEED_API int zf_tajweed_config_set_class_threshold(ZfTajweedConfig *c, const char *cls,
                                                         double threshold);
ZF_TAJWEED_API int zf_tajweed_config_clear_class(ZfTajweedConfig *c, const char *cls);
ZF_TAJWEED_API int zf_tajweed_config_set_check(ZfTajweedConfig *c, const char *check, int on);
ZF_TAJWEED_API int zf_tajweed_config_set_context(ZfTajweedConfig *c, const char *context);
/* Forget every app setting. */
ZF_TAJWEED_API void zf_tajweed_config_clear_app(ZfTajweedConfig *c);
/* 1 when the app is a debug build: only then can the app (never the portal) turn on a check or a
 * class whose calibration status is "off". Default 0. */
ZF_TAJWEED_API void zf_tajweed_config_set_debug_build(ZfTajweedConfig *c, int debug_build);

/* API 2. The heads' backend, for calibration.json's contexts[ctx].backend_overrides: "int8" (the exact
 * engine) or "fp32" (heads from the Neural Engine, an NPU or a server). NULL restores the default: "int8"
 * when the model was opened with the engine, "fp32" otherwise. Returns 0, or -1 for another name.
 * API 2 also lets zf_tajweed_config_set_class_preset and zf_tajweed_config_clear_class take "madd",
 * "qalqalah" and "gemination" (the preset picks madd's preset_cuts and their preset_tau);
 * zf_tajweed_config_set_class_threshold refuses those three. In the remote block, "v" must be 1 or the
 * whole block is ignored. */
ZF_TAJWEED_API int zf_tajweed_config_set_backend(ZfTajweedConfig *c, const char *backend);

/* API 3: the word level. The check "word": zf_tajweed_check_name(ZF_TJ_CHECK_WORD) is "word" and
 * zf_tajweed_check_from_name("word") gives it; zf_tajweed_config_set_check takes it, and the remote
 * block's "checks" may carry it. It is outside ZfTajweedEffective's check arrays (their size is API 1's),
 * so its state is read with zf_tajweed_config_word. */
#define ZF_TJ_CHECK_WORD 7

/* Word presets (calibration.json word.presets), and ZF_TJ_WORD_CUSTOM for a custom threshold. */
typedef enum {
  ZF_TJ_WORD_STRICT = 0,   /* "strict": fewest false flags */
  ZF_TJ_WORD_GENERAL = 1,  /* "general": the default */
  ZF_TJ_WORD_BEGINNER = 2, /* "beginner": catches the most */
  ZF_TJ_WORD_PRESET_COUNT = 3,
  ZF_TJ_WORD_CUSTOM = -1
} ZfTajweedWordPreset;
ZF_TAJWEED_API const char *zf_tajweed_word_preset_name(int preset);   /* "custom" for ZF_TJ_WORD_CUSTOM */
ZF_TAJWEED_API int zf_tajweed_word_preset_from_name(const char *name); /* -1 (also for "custom") */

/* App code, as the other setters: a word preset (NULL removes the app's word setting), or a custom
 * threshold (any finite number; it is clamped to calibration's custom_range and floor when resolved, as
 * the reference does). Return 0, or -1 for an unknown preset or a threshold that is not finite. */
ZF_TAJWEED_API int zf_tajweed_config_set_word_preset(ZfTajweedConfig *c, const char *preset);
ZF_TAJWEED_API int zf_tajweed_config_set_word_threshold(ZfTajweedConfig *c, double threshold);
ZF_TAJWEED_API int zf_tajweed_config_clear_word(ZfTajweedConfig *c);

/* The word level resolved (for the context the config asks for, as zf_tajweed_config_effective). The
 * remote block takes "word": {"preset": name} or {"threshold": number} (exactly one; else dropped as
 * "remote.word"). */
typedef struct {
  size_t struct_size;   /* set by the caller: sizeof(ZfTajweedWordSettings) */
  int available;        /* the context's calibration has a word block */
  int status;           /* ZF_TJ_STATUS_* of the block */
  int enabled;          /* scored and flagged: the check "word" as any other check (validated: on unless
                           turned off; experimental: only when turned on; off: app + debug build only) */
  int enabled_source;   /* ZF_TJ_SOURCE_* of the check value in force */
  int preset;           /* ZF_TJ_WORD_* in force, or ZF_TJ_WORD_CUSTOM */
  int preset_source;    /* ZF_TJ_SOURCE_* */
  double threshold;     /* the tau in force: the preset's, or the custom threshold clamped */
  double floor;         /* calibration word.floor */
  double custom_lo, custom_hi;              /* calibration word.custom_range */
  double preset_tau[ZF_TJ_WORD_PRESET_COUNT]; /* each preset's tau (floor applied) */
} ZfTajweedWordSettings;
ZF_TAJWEED_API int zf_tajweed_config_word(const ZfTajweedConfig *c, ZfTajweedWordSettings *out);

/* The effective settings, resolved. */
typedef struct {
  int preset;                               /* ZF_TJ_PRESET_*: the global preset in force */
  int preset_source;                        /* ZF_TJ_SOURCE_* */
  int class_enabled[ZF_TJ_CLASS_COUNT];     /* 1 when the class is scored and can flag */
  int class_status[ZF_TJ_CLASS_COUNT];      /* ZF_TJ_STATUS_* from calibration */
  double class_threshold[ZF_TJ_CLASS_COUNT];/* the fused-LLR cut in force */
  int class_source[ZF_TJ_CLASS_COUNT];      /* where the threshold came from */
  int check_enabled[ZF_TJ_CHECK_COUNT];
  int check_status[ZF_TJ_CHECK_COUNT];
  int check_source[ZF_TJ_CHECK_COUNT];
  char context[16];                         /* the context asked for (app > remote > default); the
                                               scorer always uses its model's own context */
  int context_source;
} ZfTajweedEffective;
ZF_TAJWEED_API int zf_tajweed_config_effective(const ZfTajweedConfig *c, ZfTajweedEffective *out);
/* The same as JSON, with each value's source; valid until the config changes or is destroyed. */
ZF_TAJWEED_API const char *zf_tajweed_config_json(ZfTajweedConfig *c);

/* ---------------------------------------------------------------------------------------------
 * Scoring
 * ------------------------------------------------------------------------------------------- */

typedef struct ZfTajweedResult ZfTajweedResult;

/* One scored site: the expected unit at `pos`, the best single-unit alternative of its class, and
 * the per-head and fused LLRs (log P(alternative) - log P(expected), full-sequence CTC forward).
 * fused = a * llr_final + (1 - a) * llr_inter1, `a` from calibration's fusion. flagged = enabled
 * and fused > threshold. */
typedef struct {
  int cls;              /* ZF_TJ_* class */
  int status;           /* ZF_TJ_STATUS_* */
  int pos;              /* index into the expected tokens (an insertion: the unit it follows,
                           -1 at the start) */
  int word;             /* 0-based word of the ayah */
  int expected;         /* tj3 id expected at pos; -1 for an insertion */
  int alt;              /* tj3 id of the best alternative; -1 for a deletion */
  double llr_final;
  double llr_inter1;
  double llr;           /* fused */
  double threshold;     /* in force for this class */
  int flagged;
  int frame_lo, frame_hi; /* 40 ms frames the expected unit holds on the best path; -1 unknown */
} ZfTajweedSite;

/* One madd long vowel, counted from the timing head. */
typedef struct {
  int pos, word;
  int token;            /* aa, ii or uu */
  int madd_class;       /* ZF_TJ_MADD_* */
  int status;
  double ratio;         /* held frames / the take's median short-vowel frames */
  int count;            /* verdict count: 2, 4 or 6 from calibration's cuts */
  double display_count; /* the smooth display count (never for verdicts) */
  int allowed_lo, allowed_hi; /* counts the rule allows */
  int flagged;          /* count outside [allowed_lo, allowed_hi] and the check enabled */
  int frame_lo, frame_hi; /* 20 ms frames */
} ZfTajweedMadd;

/* A letter feature: qalqalah (mean P(burst) after the letter) or gemination (mean P(held) over the
 * letter). flagged = the feature is missing (score below threshold) and the check enabled. */
typedef struct {
  int pos, word;
  int token;
  int status;
  double score;
  double threshold;
  int flagged;
  int frame_lo, frame_hi; /* 20 ms frames read */
} ZfTajweedLetter;

typedef struct {
  int n_sites, n_flagged;
  int n_madd, n_madd_flagged;
  int n_qalqalah, n_qalqalah_flagged;
  int n_gemination, n_gemination_flagged;
  double short_vowel_frames; /* the madd unit: median short-vowel frames, 0 when too few */
  int madd_unit_ok;          /* enough short vowels (calibration's min_short_vowels) for counts */
} ZfTajweedSummary;

/* Score a take of surah:ayah (the whole ayah) with the config in force. The heads may come from
 * zf_tajweed_stream_heads or any other source with the same layout. */
ZF_TAJWEED_API ZfTajweedResult *zf_tajweed_score(const ZfTajweedModel *m, const ZfTajweedConfig *c,
                                                 const ZfTajweedHeads *heads, int surah, int ayah,
                                                 const char **error);
ZF_TAJWEED_API void zf_tajweed_result_free(ZfTajweedResult *r);

/* Arrays owned by the result, valid until it is freed. Each returns its count. */
ZF_TAJWEED_API int zf_tajweed_result_sites(const ZfTajweedResult *r, const ZfTajweedSite **sites);
ZF_TAJWEED_API int zf_tajweed_result_madd(const ZfTajweedResult *r, const ZfTajweedMadd **madd);
ZF_TAJWEED_API int zf_tajweed_result_qalqalah(const ZfTajweedResult *r, const ZfTajweedLetter **q);
ZF_TAJWEED_API int zf_tajweed_result_gemination(const ZfTajweedResult *r, const ZfTajweedLetter **g);
ZF_TAJWEED_API int zf_tajweed_result_summary(const ZfTajweedResult *r, ZfTajweedSummary *out);
/* The whole result as JSON (UTF-8, the reference scorer's verdict layout: "sites", "madd",
 * "qalqalah", "gemination", "summary", with class, check and token names spelled out), valid until
 * the result is freed. API 3 adds "words" when the word check is enabled. */
ZF_TAJWEED_API const char *zf_tajweed_result_json(ZfTajweedResult *r);

/* API 3. The scored path: the tj3 tokens the sites and the words are scored on (the legal reading closest
 * to the greedy decode, '|' left out), and each token's word. Return the token count. */
ZF_TAJWEED_API int zf_tajweed_result_path(const ZfTajweedResult *r, const int **tokens, const int **word_of_token);

/* API 3. One word of the scored path: score = the largest clipped fused LLR over every single-unit edit
 * inside the word (a consonant, or the ikhfa nasal, swapped for each of the 28 consonants and the nasal; a
 * vowel for each other vowel; any unit deleted, unless that empties the word; a vowel inserted after a
 * consonant with no vowel after it), fused = a * LLR(log_probs) + (1 - a) * LLR(inter1_log_probs) with a
 * by the edit's class, clipped to [-30, 30]. The first edit in that order wins ties. flagged = score >=
 * threshold (note: at or above, where sites flag above). */
typedef enum { ZF_TJ_EDIT_SUB = 0, ZF_TJ_EDIT_DEL = 1, ZF_TJ_EDIT_INS = 2 } ZfTajweedEdit;
typedef struct {
  int word;             /* 0-based word of the ayah */
  int tok_lo, tok_hi;   /* the word's tokens on the scored path: [tok_lo, tok_hi) of zf_tajweed_result_path */
  int has_score;        /* 0: the word has no alternative (one unit with nothing to swap), never flagged */
  double score;         /* the best edit's fused LLR, clipped */
  double threshold;     /* the word tau in force */
  int flagged;
  int cls;              /* ZF_TJ_* class of the best edit */
  int edit;             /* ZF_TJ_EDIT_* */
  int edit_pos;         /* path index of the unit edited (an insertion goes after it) */
  int expected;         /* path token at edit_pos */
  int alt;              /* the token put in (substitution, insertion); -1 for a deletion */
  double llr_final;     /* the best edit's LLR on log_probs, clipped */
  double llr_inter1;    /* on inter1_log_probs, clipped */
} ZfTajweedWord;
/* The words scored, in order (none when the word check is not enabled). Returns the count. */
ZF_TAJWEED_API int zf_tajweed_result_words(const ZfTajweedResult *r, const ZfTajweedWord **words);

#ifdef __cplusplus
}
#endif
#endif /* ZF_TAJWEED_H */
