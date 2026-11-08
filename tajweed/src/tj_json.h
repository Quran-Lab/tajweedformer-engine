/* A small strict JSON parser (RFC 8259): the whole text into one tree, no external dependencies.
 *
 * Strict: no comments, no trailing commas, no leading zeros, no control characters in strings, valid
 * UTF-8 escapes only (\uXXXX surrogate pairs joined), nothing but whitespace after the value, depth
 * at most 64. A duplicate key keeps both members; tj_json_get returns the last (as JSON.parse). */
#ifndef ZF_TJ_JSON_H
#define ZF_TJ_JSON_H

#include <stddef.h>

enum { TJ_JNULL = 0, TJ_JFALSE, TJ_JTRUE, TJ_JNUM, TJ_JSTR, TJ_JARR, TJ_JOBJ };

typedef struct TjJ {
  int type;
  double num;          /* TJ_JNUM */
  const char *str;     /* TJ_JSTR: decoded UTF-8, NUL-terminated */
  size_t len;          /* TJ_JSTR: bytes; TJ_JARR / TJ_JOBJ: members */
  const char *key;     /* a member of an object: its key */
  struct TjJ *child;   /* first member */
  struct TjJ *next;    /* next member of the parent */
} TjJ;

typedef struct TjJDoc TjJDoc;

/* NULL on a syntax error (with *error set). */
TjJDoc *tj_json_parse(const char *text, size_t len, const char **error);
void tj_json_free(TjJDoc *d);
const TjJ *tj_json_root(const TjJDoc *d);
/* The last member named `key` of an object, or NULL. */
const TjJ *tj_json_get(const TjJ *obj, const char *key);
/* 1 when the object holds a member `key` whose type is `type`; stores it in *out */
int tj_json_get_type(const TjJ *obj, const char *key, int type, const TjJ **out);

#endif
