/* Unit tests of the tajweed library's config layers, remote validation, names, the JSON parser and the
 * model folder (on the synthetic model in tests/data/model, made by make_synth.py: random tokens, not
 * Quran text). Usage: zf_tajweed_test_config <tests/data> */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "tj_json.h"
#include "zf_tajweed.h"

static int fails, checks;

#define CHECK(cond)                                                    \
  do {                                                                 \
    checks++;                                                          \
    if (!(cond)) {                                                     \
      fails++;                                                         \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                  \
  } while (0)

static int remote(ZfTajweedConfig *c, const char *s) { return zf_tajweed_config_set_remote(c, s, strlen(s)); }

static ZfTajweedEffective eff(ZfTajweedConfig *c) {
  ZfTajweedEffective e;
  CHECK(zf_tajweed_config_effective(c, &e) == 0);
  return e;
}

static void test_names(void) {
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) CHECK(zf_tajweed_class_from_name(zf_tajweed_class_name(k)) == k);
  for (int k = 0; k < ZF_TJ_CHECK_COUNT; ++k) CHECK(zf_tajweed_check_from_name(zf_tajweed_check_name(k)) == k);
  for (int k = 0; k < ZF_TJ_PRESET_COUNT; ++k) CHECK(zf_tajweed_preset_from_name(zf_tajweed_preset_name(k)) == k);
  CHECK(zf_tajweed_class_from_name("nope") == -1);
  CHECK(zf_tajweed_class_from_name(NULL) == -1);
  CHECK(zf_tajweed_class_name(8) == NULL);
  CHECK(zf_tajweed_class_name(-1) == NULL);
  CHECK(!strcmp(zf_tajweed_madd_class_name(ZF_TJ_MADD_MUTTASIL), "muttasil"));
  CHECK(zf_tajweed_madd_class_name(5) == NULL);
  CHECK(zf_tajweed_class_check(ZF_TJ_CONS_SUB) == ZF_TJ_CHECK_LETTERS);
  CHECK(zf_tajweed_class_check(ZF_TJ_INSERTION) == ZF_TJ_CHECK_LETTERS);
  CHECK(zf_tajweed_class_check(ZF_TJ_VOWEL_INS) == ZF_TJ_CHECK_VOWELS);
  CHECK(zf_tajweed_class_check(ZF_TJ_LONG_TO_SHORT) == ZF_TJ_CHECK_LENGTH);
  CHECK(zf_tajweed_api_version() == ZF_TAJWEED_API_VERSION);
  CHECK(zf_tajweed_version() && zf_tajweed_version()[0]);
}

static void test_json(void) {
  const char *bad[] = {"", "{", "{\"a\":1,}", "[1,]", "01", "1.", "-", "\"\\x\"", "{\"a\" 1}", "tru", "[1] 2",
                       "\"a\x01\"", "\"\\ud800\"", "nan", "{'a':1}", "\"\xc3\x28\""};
  for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
    const char *e = NULL;
    TjJDoc *d = tj_json_parse(bad[i], strlen(bad[i]), &e);
    CHECK(d == NULL && e != NULL);
    tj_json_free(d);
  }
  const char *ok = " {\"a\": [1, -2.5e2, true, false, null], \"b\": \"x\\u00e9\\ud83d\\ude00\", \"a\": 3} ";
  const char *e = NULL;
  TjJDoc *d = tj_json_parse(ok, strlen(ok), &e);
  CHECK(d != NULL);
  if (d) {
    const TjJ *r = tj_json_root(d);
    const TjJ *a = tj_json_get(r, "a");
    CHECK(a && a->type == TJ_JNUM && a->num == 3.0); /* the last duplicate */
    const TjJ *b = tj_json_get(r, "b");
    CHECK(b && b->type == TJ_JSTR && !strcmp(b->str, "x\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(r->child && r->child->type == TJ_JARR && r->child->len == 5 && r->child->child->next->num == -250.0);
  }
  tj_json_free(d);
}

static void test_model(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/model", dir);
  const char *err = NULL;
  /* with the engine: the stub refuses */
  ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
  CHECK(o.load_engine == 1 && o.batch == 1 && o.struct_size == sizeof o);
  CHECK(zf_tajweed_model_open(path, &o, &err) == NULL && err != NULL);
  o.load_engine = 0;
  o.context = "c99_l1";
  CHECK(zf_tajweed_model_open(path, &o, &err) == NULL);
  o.context = "c24_l72";
  ZfTajweedModel *m2 = zf_tajweed_model_open(path, &o, &err);
  CHECK(m2 != NULL);
  if (m2) {
    CHECK(!strcmp(zf_tajweed_model_context(m2), "c24_l72"));
    zf_tajweed_model_close(m2);
  }
  o.context = NULL;
  CHECK(zf_tajweed_model_open("no/such/dir", &o, &err) == NULL);
  ZfTajweedModel *m = zf_tajweed_model_open(path, &o, &err);
  if (!m) fprintf(stderr, "model_open: %s\n", err);
  CHECK(m != NULL);
  if (!m) return;
  CHECK(!strcmp(zf_tajweed_model_id(m), "v4.0"));
  CHECK(!strcmp(zf_tajweed_model_context(m), "c16_l64"));
  CHECK(!strcmp(zf_tajweed_model_default_context(m), "c16_l64"));
  CHECK(zf_tajweed_model_context_at(m, 0) && zf_tajweed_model_context_at(m, 2) == NULL);
  CHECK(zf_tajweed_model_chunk_ms(m) == 0);
  CHECK(!strcmp(zf_tajweed_token_name(m, 0), "<blk>"));
  CHECK(!strcmp(zf_tajweed_token_name(m, 1), "|"));
  CHECK(!strcmp(zf_tajweed_token_name(m, 34), "aa"));
  CHECK(zf_tajweed_token_name(m, 38) == NULL);
  CHECK(zf_tajweed_ayah_count(m, 1) == 7 && zf_tajweed_ayah_count(m, 2) == 5 && zf_tajweed_ayah_count(m, 4) == 8 && zf_tajweed_ayah_count(m, 5) == 0);
  CHECK(zf_tajweed_ayah_count(m, 0) == 0 && zf_tajweed_ayah_count(m, 115) == 0);
  const int *tok = NULL, *wd = NULL;
  int nw = -1;
  int n = zf_tajweed_expected_tokens(m, 1, 1, &tok);
  CHECK(n > 0 && tok != NULL);
  CHECK(zf_tajweed_expected_words(m, 1, 1, &wd, &nw) == n && nw >= 1);
  for (int i = 0; tok && wd && i < n; ++i) {
    CHECK(tok[i] > 0 && tok[i] < 37);
    CHECK(wd[i] >= 0 && wd[i] < nw);
    if (tok[i] == 1) CHECK(i > 0 && wd[i] == wd[i - 1]);
  }
  CHECK(zf_tajweed_expected_tokens(m, 1, 8, &tok) == -1);
  CHECK(zf_tajweed_expected_tokens(m, 5, 1, &tok) == -1);

  /* ---- config: calibration defaults ---- */
  ZfTajweedConfig *c = zf_tajweed_config_create(m);
  ZfTajweedEffective e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_BALANCED && e.preset_source == ZF_TJ_SOURCE_CALIBRATION);
  /* synthetic thresholds: base[preset] + 0.25 * class index; balanced base 2.0 */
  for (int k = 0; k < ZF_TJ_CLASS_COUNT; ++k) {
    CHECK(fabs(e.class_threshold[k] - (2.0 + 0.25 * k)) < 1e-12);
    CHECK(e.class_source[k] == ZF_TJ_SOURCE_CALIBRATION);
  }
  CHECK(e.class_status[ZF_TJ_VOWEL_SUB] == ZF_TJ_STATUS_VALIDATED && e.class_enabled[ZF_TJ_VOWEL_SUB]);
  CHECK(e.class_status[ZF_TJ_VOWEL_INS] == ZF_TJ_STATUS_EXPERIMENTAL && !e.class_enabled[ZF_TJ_VOWEL_INS]);
  CHECK(e.class_status[ZF_TJ_CONS_DEL] == ZF_TJ_STATUS_EXPERIMENTAL && !e.class_enabled[ZF_TJ_CONS_DEL]);
  CHECK(e.class_status[ZF_TJ_INSERTION] == ZF_TJ_STATUS_OFF && !e.class_enabled[ZF_TJ_INSERTION]);
  CHECK(e.check_enabled[ZF_TJ_CHECK_LETTERS] && e.check_status[ZF_TJ_CHECK_LETTERS] == ZF_TJ_STATUS_VALIDATED);
  CHECK(e.check_enabled[ZF_TJ_CHECK_MADD]);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_QALQALAH] && e.check_status[ZF_TJ_CHECK_QALQALAH] == ZF_TJ_STATUS_EXPERIMENTAL);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_GEMINATION]);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_LAZIM] && e.check_status[ZF_TJ_CHECK_LAZIM] == ZF_TJ_STATUS_OFF);
  CHECK(!strcmp(e.context, "c16_l64") && e.context_source == ZF_TJ_SOURCE_CALIBRATION);

  /* ---- remote: a valid block ---- */
  CHECK(remote(c, "{\"v\":1,\"preset\":\"sensitive\",\"overrides\":{\"cons_sub\":{\"preset\":\"strict\"},"
                  "\"vowel_sub\":{\"threshold\":1.25}},\"checks\":{\"qalqalah\":true,\"letters\":true,"
                  "\"lazim\":true},\"context\":\"c24_l72\"}") == 0);
  e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_SENSITIVE && e.preset_source == ZF_TJ_SOURCE_REMOTE);
  CHECK(e.class_threshold[ZF_TJ_VOWEL_SUB] == 1.25 && e.class_source[ZF_TJ_VOWEL_SUB] == ZF_TJ_SOURCE_REMOTE);
  CHECK(fabs(e.class_threshold[ZF_TJ_CONS_SUB] - (3.0 + 0.25 * 5)) < 1e-12); /* strict */
  CHECK(fabs(e.class_threshold[ZF_TJ_VOWEL_DEL] - (1.0 + 0.25 * 1)) < 1e-12); /* sensitive */
  CHECK(e.check_enabled[ZF_TJ_CHECK_QALQALAH] && e.check_source[ZF_TJ_CHECK_QALQALAH] == ZF_TJ_SOURCE_REMOTE);
  /* remote turning on an off check does nothing */
  CHECK(!e.check_enabled[ZF_TJ_CHECK_LAZIM] && e.check_source[ZF_TJ_CHECK_LAZIM] == ZF_TJ_SOURCE_CALIBRATION);
  /* letters turned on explicitly: the experimental cons_del joins, the off insertion does not */
  CHECK(e.class_enabled[ZF_TJ_CONS_DEL]);
  CHECK(!e.class_enabled[ZF_TJ_INSERTION]);
  CHECK(!e.class_enabled[ZF_TJ_VOWEL_INS]); /* vowels on by calibration only */
  CHECK(!strcmp(e.context, "c24_l72") && e.context_source == ZF_TJ_SOURCE_REMOTE);
  CHECK(zf_tajweed_config_json(c) != NULL && strstr(zf_tajweed_config_json(c), "\"source\":\"remote\""));

  /* ---- remote: invalid fields are dropped one by one, never the block ---- */
  CHECK(remote(c, "{\"v\":1,\"preset\":\"loud\",\"overrides\":{\"cons_sub\":{\"preset\":\"strict\",\"threshold\":1},"
                  "\"vowel_sub\":{\"threshold\":25},\"vowel_del\":{\"threshold\":-3.5},\"bogus\":{\"preset\":\"strict\"},"
                  "\"cons_del\":{\"threshold\":\"1\"},\"madd\":{\"threshold\":1},\"qalqalah\":{\"preset\":\"strict\"}},\"checks\":{\"qalqalah\":true,\"tafkheem\":true,\"madd\":1},"
                  "\"context\":\"c99_l1\",\"extra\":1}") == 10);
  e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_BALANCED && e.preset_source == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(e.class_source[ZF_TJ_CONS_SUB] == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(e.class_source[ZF_TJ_VOWEL_SUB] == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(e.class_threshold[ZF_TJ_VOWEL_DEL] == -3.5 && e.class_source[ZF_TJ_VOWEL_DEL] == ZF_TJ_SOURCE_REMOTE);
  CHECK(e.check_enabled[ZF_TJ_CHECK_QALQALAH]);
  CHECK(e.check_enabled[ZF_TJ_CHECK_MADD] && e.check_source[ZF_TJ_CHECK_MADD] == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(e.context_source == ZF_TJ_SOURCE_CALIBRATION);
  /* non-finite / wrong-typed thresholds, override with neither, v other than 1 */
  CHECK(remote(c, "{\"v\":1,\"overrides\":{\"cons_sub\":{},\"vowel_sub\":{\"threshold\":1e999}},\"checks\":[]}") == 3);
  /* API 2: v other than the number 1 ignores the whole block */
  CHECK(remote(c, "{\"v\":2,\"preset\":\"strict\"}") == 1);
  CHECK(eff(c).preset_source == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(remote(c, "{\"preset\":\"strict\"}") == 1);
  CHECK(eff(c).preset_source == ZF_TJ_SOURCE_CALIBRATION);
  e = eff(c);
  CHECK(e.class_source[ZF_TJ_VOWEL_SUB] == ZF_TJ_SOURCE_CALIBRATION);
  /* boundaries of the threshold range are valid */
  CHECK(remote(c, "{\"v\":1,\"overrides\":{\"cons_sub\":{\"threshold\":-20},\"vowel_sub\":{\"threshold\":20}}}") == 0);
  e = eff(c);
  CHECK(e.class_threshold[ZF_TJ_CONS_SUB] == -20.0 && e.class_threshold[ZF_TJ_VOWEL_SUB] == 20.0);
  /* a whole /v1/config document */
  CHECK(remote(c, "{\"asr\":{\"x\":1},\"tajweed\":{\"v\":1,\"preset\":\"strict\"}}") == 0);
  e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_STRICT && e.preset_source == ZF_TJ_SOURCE_REMOTE);
  /* a document without a block clears the layer */
  CHECK(remote(c, "{\"asr\":{\"x\":1}}") == 0);
  e = eff(c);
  CHECK(e.preset_source == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(remote(c, "{\"tajweed\":null}") == 0);
  /* not JSON: -1 and the layer cleared */
  CHECK(remote(c, "{\"v\":1,\"preset\":\"strict\"}") == 0);
  CHECK(remote(c, "{\"v\":1,\"preset\":") == -1);
  e = eff(c);
  CHECK(e.preset_source == ZF_TJ_SOURCE_CALIBRATION);
  CHECK(zf_tajweed_config_set_remote(c, NULL, 0) == 0);
  CHECK(remote(c, "[1,2]") == 1);

  /* ---- app over remote over calibration ---- */
  CHECK(remote(c, "{\"v\":1,\"preset\":\"sensitive\",\"overrides\":{\"cons_sub\":{\"preset\":\"very_sensitive\"}},"
                  "\"checks\":{\"letters\":false,\"qalqalah\":true}}") == 0);
  e = eff(c);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_LETTERS] && !e.class_enabled[ZF_TJ_CONS_SUB]);
  CHECK(zf_tajweed_config_set_preset(c, "strict") == 0);
  CHECK(zf_tajweed_config_set_preset(c, "loud") == -1);
  e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_STRICT && e.preset_source == ZF_TJ_SOURCE_APP);
  /* the portal's per-class override outlives the app's global preset (tajweed_ref.resolve) */
  CHECK(fabs(e.class_threshold[ZF_TJ_CONS_SUB] - (-0.5 + 0.25 * 5)) < 1e-12 &&
        e.class_source[ZF_TJ_CONS_SUB] == ZF_TJ_SOURCE_REMOTE);
  /* the other classes follow the app's preset */
  CHECK(fabs(e.class_threshold[ZF_TJ_VOWEL_DEL] - (3.0 + 0.25 * 1)) < 1e-12 &&
        e.class_source[ZF_TJ_VOWEL_DEL] == ZF_TJ_SOURCE_APP);
  CHECK(zf_tajweed_config_set_class_threshold(c, "cons_sub", 0.5) == 0);
  CHECK(zf_tajweed_config_set_class_threshold(c, "cons_sub", 21.0) == -1);
  CHECK(zf_tajweed_config_set_class_threshold(c, "cons_sub", NAN) == -1);
  CHECK(zf_tajweed_config_set_class_threshold(c, "nope", 1.0) == -1);
  CHECK(zf_tajweed_config_set_class_preset(c, "vowel_sub", "sensitive") == 0);
  CHECK(zf_tajweed_config_set_class_preset(c, "vowel_sub", "loud") == -1);
  CHECK(zf_tajweed_config_set_check(c, "letters", 1) == 0);
  CHECK(zf_tajweed_config_set_check(c, "loud", 1) == -1);
  CHECK(zf_tajweed_config_set_check(c, "letters", 2) == -1);
  e = eff(c);
  CHECK(e.class_threshold[ZF_TJ_CONS_SUB] == 0.5 && e.class_enabled[ZF_TJ_CONS_SUB]);
  CHECK(fabs(e.class_threshold[ZF_TJ_VOWEL_SUB] - 1.0) < 1e-12);
  CHECK(e.check_enabled[ZF_TJ_CHECK_LETTERS] && e.check_source[ZF_TJ_CHECK_LETTERS] == ZF_TJ_SOURCE_APP);
  CHECK(e.class_enabled[ZF_TJ_CONS_DEL] && !e.class_enabled[ZF_TJ_INSERTION]);
  /* remove app settings one by one */
  CHECK(zf_tajweed_config_clear_class(c, "cons_sub") == 0);
  e = eff(c);
  CHECK(e.class_source[ZF_TJ_CONS_SUB] == ZF_TJ_SOURCE_REMOTE); /* back to the portal's override */
  /* madd / qalqalah / gemination take a preset override */
  CHECK(zf_tajweed_config_set_class_preset(c, "madd", "strict") == 0);
  CHECK(zf_tajweed_config_set_class_threshold(c, "madd", 1.0) == -1);
  CHECK(zf_tajweed_config_set_preset(c, NULL) == 0);
  e = eff(c);
  CHECK(fabs(e.class_threshold[ZF_TJ_CONS_SUB] - (-0.5 + 0.25 * 5)) < 1e-12 &&
        e.class_source[ZF_TJ_CONS_SUB] == ZF_TJ_SOURCE_REMOTE);
  CHECK(zf_tajweed_config_set_check(c, "letters", -1) == 0);
  e = eff(c);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_LETTERS] && e.check_source[ZF_TJ_CHECK_LETTERS] == ZF_TJ_SOURCE_REMOTE);
  /* off checks: only an app in a debug build turns them on */
  CHECK(zf_tajweed_config_set_check(c, "lazim", 1) == 0);
  e = eff(c);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_LAZIM]);
  zf_tajweed_config_set_debug_build(c, 1);
  e = eff(c);
  CHECK(e.check_enabled[ZF_TJ_CHECK_LAZIM] && e.check_source[ZF_TJ_CHECK_LAZIM] == ZF_TJ_SOURCE_APP);
  CHECK(zf_tajweed_config_set_check(c, "letters", 1) == 0);
  e = eff(c);
  CHECK(e.class_enabled[ZF_TJ_INSERTION]); /* off class, app on, debug build */
  zf_tajweed_config_set_debug_build(c, 0);
  e = eff(c);
  CHECK(!e.class_enabled[ZF_TJ_INSERTION] && !e.check_enabled[ZF_TJ_CHECK_LAZIM]);
  /* turning a check off works from any layer */
  CHECK(zf_tajweed_config_set_check(c, "madd", 0) == 0);
  e = eff(c);
  CHECK(!e.check_enabled[ZF_TJ_CHECK_MADD]);
  /* app context */
  CHECK(zf_tajweed_config_set_context(c, "c99_l1") == -1);
  CHECK(zf_tajweed_config_set_context(c, "c24_l72") == 0);
  e = eff(c);
  CHECK(!strcmp(e.context, "c24_l72") && e.context_source == ZF_TJ_SOURCE_APP);
  /* API 2: the backend */
  CHECK(zf_tajweed_config_set_backend(c, "int8") == 0);
  CHECK(zf_tajweed_config_set_backend(c, "fp16") == -1);
  CHECK(strstr(zf_tajweed_config_json(c), "\"backend\":\"int8\"") != NULL);
  CHECK(zf_tajweed_config_set_backend(c, NULL) == 0);
  CHECK(strstr(zf_tajweed_config_json(c), "\"backend\":\"fp32\"") != NULL); /* no engine: fp32 */
  CHECK(zf_tajweed_api_version() == ZF_TAJWEED_API_VERSION && ZF_TAJWEED_API_VERSION >= 3);
  { /* API 3: the decoded layer and the head table */
    const float lp[2 * 4] = {-0.1f, -3.0f, -2.0f, -2.0f,  /* best 0, runner-up 2 (tie with 3: lower id) */
                             -5.0f, -0.5f, -0.5f, -9.0f}; /* tie for best: 1 wins, 2 is the runner-up */
    ZfTajweedTop2 t[2];
    CHECK(zf_tajweed_top2(lp, 2, 4, t) == 2);
    CHECK(t[0].token == 0 && t[0].runner_up == 2 && t[0].log_prob == -0.1f && t[0].runner_up_log_prob == -2.0f);
    CHECK(t[0].margin == -0.1f - -2.0f);
    CHECK(t[1].token == 1 && t[1].runner_up == 2 && t[1].margin == 0.0f);
    CHECK(zf_tajweed_top2(lp, 2, 1, t) == -1);
    int ms = 0, cols = 0;
    CHECK(zf_tajweed_head_info(ZF_TAJWEED_HEAD_INTER1, &ms, &cols) && ms == 40 && cols == 38);
    CHECK(!strcmp(zf_tajweed_head_info(ZF_TAJWEED_HEAD_QALQALAH, &ms, &cols), "attr_qalqalah") && ms == 20 &&
          cols == 3);
    CHECK(zf_tajweed_head_info(ZF_TAJWEED_HEADS_SCORER, NULL, NULL) == NULL);
    CHECK((ZF_TAJWEED_HEADS_SCORER & ZF_TAJWEED_HEAD_LOG_PROBS) && ZF_TAJWEED_HEADS_ALL == 0xfff);
  }
  zf_tajweed_config_clear_app(c);
  e = eff(c);
  CHECK(e.preset == ZF_TJ_PRESET_SENSITIVE && e.preset_source == ZF_TJ_SOURCE_REMOTE);
  CHECK(e.check_enabled[ZF_TJ_CHECK_MADD]);
  CHECK(zf_tajweed_config_json(c) != NULL);
  zf_tajweed_config_destroy(c);
  zf_tajweed_model_close(m);
}

/* API 3: the word level (tests/data/model: word validated at c16 with taus strict 7.0, general 5.5,
 * beginner 1.5 (below the floor 2.0); experimental at c24 with beginner 4.0; custom_range [2, 12]) */
static ZfTajweedWordSettings word(ZfTajweedConfig *c) {
  ZfTajweedWordSettings w;
  memset(&w, 0, sizeof w);
  w.struct_size = sizeof w;
  CHECK(zf_tajweed_config_word(c, &w) == 0);
  return w;
}

static void test_word(const char *dir) {
  char path[1024];
  snprintf(path, sizeof path, "%s/model", dir);
  const char *err = NULL;
  ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
  o.load_engine = 0;
  ZfTajweedModel *m = zf_tajweed_model_open(path, &o, &err);
  CHECK(m != NULL);
  if (!m) return;
  ZfTajweedConfig *c = zf_tajweed_config_create(m);
  /* names */
  CHECK(zf_tajweed_check_from_name("word") == ZF_TJ_CHECK_WORD && !strcmp(zf_tajweed_check_name(ZF_TJ_CHECK_WORD), "word"));
  for (int p = 0; p < ZF_TJ_WORD_PRESET_COUNT; ++p)
    CHECK(zf_tajweed_word_preset_from_name(zf_tajweed_word_preset_name(p)) == p);
  CHECK(!strcmp(zf_tajweed_word_preset_name(ZF_TJ_WORD_CUSTOM), "custom"));
  CHECK(zf_tajweed_word_preset_from_name("custom") == -1 && zf_tajweed_word_preset_from_name("balanced") == -1);
  /* calibration defaults */
  ZfTajweedWordSettings w = word(c);
  CHECK(w.available && w.status == ZF_TJ_STATUS_VALIDATED && w.enabled);
  CHECK(w.preset == ZF_TJ_WORD_GENERAL && w.preset_source == ZF_TJ_SOURCE_CALIBRATION && w.threshold == 5.5);
  CHECK(w.floor == 2.0 && w.custom_lo == 2.0 && w.custom_hi == 12.0);
  CHECK(w.preset_tau[ZF_TJ_WORD_STRICT] == 7.0 && w.preset_tau[ZF_TJ_WORD_BEGINNER] == 2.0); /* floor */
  /* remote: a preset, a threshold clamped, invalid blocks dropped */
  CHECK(remote(c, "{\"v\":1,\"word\":{\"preset\":\"strict\"}}") == 0);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_STRICT && w.preset_source == ZF_TJ_SOURCE_REMOTE && w.threshold == 7.0);
  CHECK(remote(c, "{\"v\":1,\"word\":{\"threshold\":40}}") == 0);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_CUSTOM && w.threshold == 12.0);
  CHECK(remote(c, "{\"v\":1,\"word\":{\"threshold\":-3}}") == 0);
  CHECK(word(c).threshold == 2.0);
  CHECK(remote(c, "{\"v\":1,\"word\":{\"preset\":\"general\",\"threshold\":3}}") == 1);
  CHECK(remote(c, "{\"v\":1,\"word\":{\"preset\":\"balanced\"}}") == 1);
  CHECK(remote(c, "{\"v\":1,\"word\":{\"threshold\":\"3\"}}") == 1);
  CHECK(remote(c, "{\"v\":1,\"word\":5}") == 1);
  CHECK(strstr(zf_tajweed_config_json(c), "\"remote.word\"") != NULL);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_GENERAL && w.preset_source == ZF_TJ_SOURCE_CALIBRATION);
  /* app over remote; the beginner tau is raised to the floor */
  CHECK(remote(c, "{\"v\":1,\"word\":{\"preset\":\"strict\"}}") == 0);
  CHECK(zf_tajweed_config_set_word_preset(c, "beginner") == 0);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_BEGINNER && w.preset_source == ZF_TJ_SOURCE_APP && w.threshold == 2.0);
  CHECK(zf_tajweed_config_set_word_preset(c, "nope") == -1);
  CHECK(zf_tajweed_config_set_word_threshold(c, NAN) == -1);
  CHECK(zf_tajweed_config_set_word_threshold(c, INFINITY) == -1);
  CHECK(zf_tajweed_config_set_word_threshold(c, 6.25) == 0);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_CUSTOM && w.threshold == 6.25 && w.preset_source == ZF_TJ_SOURCE_APP);
  CHECK(zf_tajweed_config_set_word_threshold(c, 1.0) == 0);
  CHECK(word(c).threshold == 2.0);
  CHECK(strstr(zf_tajweed_config_json(c), "\"word\":{\"status\":\"validated\",\"enabled\":true,\"preset\":\"custom\"") != NULL);
  CHECK(zf_tajweed_config_clear_word(c) == 0);
  w = word(c);
  CHECK(w.preset == ZF_TJ_WORD_STRICT && w.preset_source == ZF_TJ_SOURCE_REMOTE); /* remote shows through */
  CHECK(zf_tajweed_config_set_word_preset(c, NULL) == 0);
  /* the check: remote off, app on wins */
  CHECK(remote(c, "{\"v\":1,\"checks\":{\"word\":false}}") == 0);
  w = word(c);
  CHECK(!w.enabled && w.enabled_source == ZF_TJ_SOURCE_REMOTE);
  CHECK(zf_tajweed_config_set_check(c, "word", 1) == 0);
  w = word(c);
  CHECK(w.enabled && w.enabled_source == ZF_TJ_SOURCE_APP);
  /* experimental at c24: off until turned on */
  CHECK(zf_tajweed_config_set_check(c, "word", -1) == 0);
  CHECK(remote(c, "{\"v\":1}") == 0);
  CHECK(zf_tajweed_config_set_context(c, "c24_l72") == 0);
  w = word(c);
  CHECK(w.status == ZF_TJ_STATUS_EXPERIMENTAL && !w.enabled && w.threshold == 5.5);
  CHECK(remote(c, "{\"v\":1,\"checks\":{\"word\":true}}") == 0);
  CHECK(word(c).enabled);
  /* a caller built against a shorter struct gets only its part */
  ZfTajweedWordSettings small;
  memset(&small, 0xab, sizeof small);
  small.struct_size = offsetof(ZfTajweedWordSettings, preset);
  CHECK(zf_tajweed_config_word(c, &small) == 0 && small.enabled == 1 && small.preset == (int)0xabababab);
  small.struct_size = 4;
  CHECK(zf_tajweed_config_word(c, &small) == -1);
  zf_tajweed_config_destroy(c);
  zf_tajweed_model_close(m);
}

/* API 3 top-K: against a reference stable argsort and the decoded layer's original top-2 scan, on rows
 * with many ties (values drawn from a few levels) and on distinct values */
static void ref_argsort(const float *row, int cols, int *ord) {
  for (int v = 0; v < cols; ++v) {
    int j = v;
    while (j > 0 && row[ord[j - 1]] < row[v]) {
      ord[j] = ord[j - 1];
      --j;
    }
    ord[j] = v;
  }
}

/* the no_pin default by platform (API 3): pinned on Android and iOS, not elsewhere; a caller whose header
 * predates no_pin gets the default, an explicit value is kept */
static void test_pin_default(const char *dir) {
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
  const int expected = 0;
#else
  const int expected = 1;
#endif
  ZfTajweedOpenOptions o = zf_tajweed_open_default_options();
  CHECK(o.no_pin == expected);
  char path[1024];
  snprintf(path, sizeof path, "%s/model", dir);
  const char *err = NULL;
  o.load_engine = 0;
  o.struct_size = offsetof(ZfTajweedOpenOptions, no_pin); /* an API 2 caller */
  o.no_pin = 7;                                           /* beyond its struct: never read */
  ZfTajweedModel *m = zf_tajweed_model_open(path, &o, &err);
  CHECK(m != NULL && m->options.no_pin == expected);
  zf_tajweed_model_close(m);
  o.struct_size = sizeof o;
  o.no_pin = !expected; /* explicit, the other way */
  m = zf_tajweed_model_open(path, &o, &err);
  CHECK(m != NULL && m->options.no_pin == !expected);
  zf_tajweed_model_close(m);
}

static void test_topk(void) {
  unsigned seed = 12345u;
  int bad = 0, rows_checked = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const int cols = 1 + (int)(trial % 38);
    float row[38];
    for (int v = 0; v < cols; ++v) {
      seed = seed * 1664525u + 1013904223u;
      row[v] = trial % 2 ? -(float)(seed >> 28) * 0.5f : -(float)(seed >> 8) * 1e-6f; /* ties : distinct */
    }
    int ord[38];
    ref_argsort(row, cols, ord);
    for (int k = 1; k <= cols; ++k) {
      int ids[38];
      float vals[38], mg[38];
      if (zf_tajweed_topk(row, 1, cols, k, ids, vals, mg) != 1) { ++bad; continue; }
      for (int j = 0; j < k; ++j)
        bad += ids[j] != ord[j] || vals[j] != row[ord[j]] ||
               mg[j] != (j + 1 < cols ? row[ord[j]] - row[ord[j + 1]] : 0.0f);
    }
    if (cols >= 2) { /* k = 2 against the original single-scan top-2 (p2-cpuopt's) */
      int best = 0, second = -1;
      for (int v = 1; v < cols; ++v) {
        if (row[v] > row[best]) second = best, best = v;
        else if (second < 0 || row[v] > row[second]) second = v;
      }
      ZfTajweedTop2 t;
      CHECK(zf_tajweed_top2(row, 1, cols, &t) == 1);
      bad += t.token != best || t.runner_up != second || t.log_prob != row[best] ||
             t.runner_up_log_prob != row[second] || t.margin != row[best] - row[second];
    }
    rows_checked++;
  }
  CHECK(bad == 0 && rows_checked == 2000);
  /* several frames at once, NULL value / margin outputs, argument checks */
  const float two[2 * 3] = {0.1f, 0.7f, 0.2f, 0.5f, 0.5f, 0.0f};
  int ids[6];
  CHECK(zf_tajweed_topk(two, 2, 3, 3, ids, NULL, NULL) == 2);
  CHECK(ids[0] == 1 && ids[1] == 2 && ids[2] == 0 && ids[3] == 0 && ids[4] == 1 && ids[5] == 2);
  CHECK(zf_tajweed_topk(two, 2, 3, 0, ids, NULL, NULL) == -1);
  CHECK(zf_tajweed_topk(two, 2, 3, 4, ids, NULL, NULL) == -1);
  CHECK(zf_tajweed_topk(NULL, 2, 3, 1, ids, NULL, NULL) == -1);
  CHECK(zf_tajweed_topk(two, 2, 3, 1, NULL, NULL, NULL) == -1);
  CHECK(zf_tajweed_stream_topk(NULL, ZF_TAJWEED_HEAD_LOG_PROBS, 2, ids, NULL, NULL, 1) == -1);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: zf_tajweed_test_config <tests/data>\n");
    return 2;
  }
  test_names();
  test_json();
  test_model(argv[1]);
  test_word(argv[1]);
  test_topk();
  test_pin_default(argv[1]);
  printf("%d/%d checks passed\n", checks - fails, checks);
  return fails ? 1 : 0;
}
