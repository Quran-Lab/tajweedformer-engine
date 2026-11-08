/* A small strict JSON parser (see tj_json.h). */
#include "tj_json.h"

#include <locale.h>
#include <stdlib.h>
#include <string.h>

#define TJ_JSON_DEPTH 64

struct TjJDoc {
  TjJ *nodes;
  size_t n_nodes, cap_nodes;
  char *strs;
  size_t n_strs, cap_strs;
  TjJ *root;
};

typedef struct {
  const char *p, *end;
  TjJDoc *d;
  const char *err;
} P;

static TjJ *node(P *ps) {
  TjJDoc *d = ps->d;
  if (d->n_nodes >= d->cap_nodes) {
    ps->err = "json: too many values";
    return NULL;
  }
  TjJ *n = &d->nodes[d->n_nodes++];
  memset(n, 0, sizeof *n);
  return n;
}

static void ws(P *ps) {
  while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hex4(const char *s, unsigned *out) {
  unsigned v = 0;
  for (int i = 0; i < 4; ++i) {
    char c = s[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
    else return -1;
  }
  *out = v;
  return 0;
}

static void put_utf8(char *o, size_t *k, unsigned cp) {
  if (cp < 0x80) {
    o[(*k)++] = (char)cp;
  } else if (cp < 0x800) {
    o[(*k)++] = (char)(0xC0 | (cp >> 6));
    o[(*k)++] = (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    o[(*k)++] = (char)(0xE0 | (cp >> 12));
    o[(*k)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[(*k)++] = (char)(0x80 | (cp & 0x3F));
  } else {
    o[(*k)++] = (char)(0xF0 | (cp >> 18));
    o[(*k)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[(*k)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[(*k)++] = (char)(0x80 | (cp & 0x3F));
  }
}

/* the length of a valid UTF-8 sequence starting at s (max bytes n), or 0 */
static int utf8_len(const unsigned char *s, size_t n) {
  unsigned c = s[0];
  if (c < 0x80) return 1;
  int len;
  unsigned min;
  if ((c & 0xE0) == 0xC0) len = 2, min = 0x80, c &= 0x1F;
  else if ((c & 0xF0) == 0xE0) len = 3, min = 0x800, c &= 0x0F;
  else if ((c & 0xF8) == 0xF0) len = 4, min = 0x10000, c &= 0x07;
  else return 0;
  if ((size_t)len > n) return 0;
  for (int i = 1; i < len; ++i) {
    if ((s[i] & 0xC0) != 0x80) return 0;
    c = (c << 6) | (s[i] & 0x3F);
  }
  if (c < min || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return 0;
  return len;
}

/* a string at ps->p (on the opening quote): decoded into the string pool */
static const char *string(P *ps, size_t *out_len) {
  TjJDoc *d = ps->d;
  ps->p++;
  char *o = d->strs + d->n_strs;
  size_t k = 0;
  for (;;) {
    if (ps->p >= ps->end) {
      ps->err = "json: unterminated string";
      return NULL;
    }
    unsigned char c = (unsigned char)*ps->p;
    if (c == '"') {
      ps->p++;
      break;
    }
    if (c < 0x20) {
      ps->err = "json: control character in a string";
      return NULL;
    }
    if (c == '\\') {
      if (ps->p + 1 >= ps->end) {
        ps->err = "json: bad escape";
        return NULL;
      }
      char e = ps->p[1];
      ps->p += 2;
      switch (e) {
        case '"': o[k++] = '"'; break;
        case '\\': o[k++] = '\\'; break;
        case '/': o[k++] = '/'; break;
        case 'b': o[k++] = '\b'; break;
        case 'f': o[k++] = '\f'; break;
        case 'n': o[k++] = '\n'; break;
        case 'r': o[k++] = '\r'; break;
        case 't': o[k++] = '\t'; break;
        case 'u': {
          unsigned cp;
          if (ps->end - ps->p < 4 || hex4(ps->p, &cp)) {
            ps->err = "json: bad \\u escape";
            return NULL;
          }
          ps->p += 4;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            unsigned lo;
            if (ps->end - ps->p < 6 || ps->p[0] != '\\' || ps->p[1] != 'u' || hex4(ps->p + 2, &lo) ||
                lo < 0xDC00 || lo > 0xDFFF) {
              ps->err = "json: lone surrogate";
              return NULL;
            }
            ps->p += 6;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            ps->err = "json: lone surrogate";
            return NULL;
          }
          if (cp == 0) {
            ps->err = "json: NUL in a string";
            return NULL;
          }
          put_utf8(o, &k, cp);
          break;
        }
        default:
          ps->err = "json: bad escape";
          return NULL;
      }
      continue;
    }
    int n = utf8_len((const unsigned char *)ps->p, (size_t)(ps->end - ps->p));
    if (!n) {
      ps->err = "json: invalid UTF-8";
      return NULL;
    }
    memcpy(o + k, ps->p, (size_t)n);
    k += (size_t)n;
    ps->p += n;
  }
  o[k] = 0;
  d->n_strs += k + 1;
  *out_len = k;
  return o;
}

static int number(P *ps, double *out) {
  const char *s = ps->p, *q = s;
  if (q < ps->end && *q == '-') q++;
  if (q >= ps->end) goto bad;
  if (*q == '0') {
    q++;
  } else if (*q >= '1' && *q <= '9') {
    while (q < ps->end && *q >= '0' && *q <= '9') q++;
  } else {
    goto bad;
  }
  if (q < ps->end && *q == '.') {
    q++;
    if (q >= ps->end || !(*q >= '0' && *q <= '9')) goto bad;
    while (q < ps->end && *q >= '0' && *q <= '9') q++;
  }
  if (q < ps->end && (*q == 'e' || *q == 'E')) {
    q++;
    if (q < ps->end && (*q == '+' || *q == '-')) q++;
    if (q >= ps->end || !(*q >= '0' && *q <= '9')) goto bad;
    while (q < ps->end && *q >= '0' && *q <= '9') q++;
  }
  {
    /* strtod reads the locale's decimal point: copy the token with that point */
    char buf[512];
    size_t n = (size_t)(q - s);
    if (n >= sizeof buf) goto bad;
    memcpy(buf, s, n);
    buf[n] = 0;
    const struct lconv *lc = localeconv();
    char dp = (lc && lc->decimal_point && lc->decimal_point[0]) ? lc->decimal_point[0] : '.';
    if (dp != '.')
      for (size_t i = 0; i < n; ++i)
        if (buf[i] == '.') buf[i] = dp;
    char *e = NULL;
    *out = strtod(buf, &e);
    if (e != buf + n) goto bad;
  }
  ps->p = q;
  return 0;
bad:
  ps->err = "json: bad number";
  return -1;
}

static int lit(P *ps, const char *w) {
  size_t n = strlen(w);
  if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, w, n)) {
    ps->err = "json: unexpected character";
    return -1;
  }
  ps->p += n;
  return 0;
}

static TjJ *value(P *ps, int depth) {
  if (depth > TJ_JSON_DEPTH) {
    ps->err = "json: nested too deep";
    return NULL;
  }
  ws(ps);
  if (ps->p >= ps->end) {
    ps->err = "json: unexpected end";
    return NULL;
  }
  TjJ *n = node(ps);
  if (!n) return NULL;
  char c = *ps->p;
  if (c == '{' || c == '[') {
    const int obj = c == '{';
    n->type = obj ? TJ_JOBJ : TJ_JARR;
    ps->p++;
    ws(ps);
    if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) {
      ps->p++;
      return n;
    }
    TjJ *last = NULL;
    for (;;) {
      const char *key = NULL;
      if (obj) {
        ws(ps);
        size_t kl;
        if (ps->p >= ps->end || *ps->p != '"') {
          ps->err = "json: expected a key";
          return NULL;
        }
        key = string(ps, &kl);
        if (!key) return NULL;
        ws(ps);
        if (ps->p >= ps->end || *ps->p != ':') {
          ps->err = "json: expected ':'";
          return NULL;
        }
        ps->p++;
      }
      TjJ *m = value(ps, depth + 1);
      if (!m) return NULL;
      m->key = key;
      if (last) last->next = m;
      else n->child = m;
      last = m;
      n->len++;
      ws(ps);
      if (ps->p >= ps->end) {
        ps->err = "json: unexpected end";
        return NULL;
      }
      if (*ps->p == ',') {
        ps->p++;
        continue;
      }
      if (*ps->p == (obj ? '}' : ']')) {
        ps->p++;
        return n;
      }
      ps->err = "json: expected ',' or a closing bracket";
      return NULL;
    }
  }
  if (c == '"') {
    n->type = TJ_JSTR;
    n->str = string(ps, &n->len);
    return n->str ? n : NULL;
  }
  if (c == 't') {
    n->type = TJ_JTRUE;
    return lit(ps, "true") ? NULL : n;
  }
  if (c == 'f') {
    n->type = TJ_JFALSE;
    return lit(ps, "false") ? NULL : n;
  }
  if (c == 'n') {
    n->type = TJ_JNULL;
    return lit(ps, "null") ? NULL : n;
  }
  n->type = TJ_JNUM;
  return number(ps, &n->num) ? NULL : n;
}

TjJDoc *tj_json_parse(const char *text, size_t len, const char **error) {
  TjJDoc *d = (TjJDoc *)calloc(1, sizeof *d);
  if (!d) {
    if (error) *error = "out of memory";
    return NULL;
  }
  /* every value takes at least one byte of text; every decoded string at most its own bytes */
  d->cap_nodes = len + 1;
  d->cap_strs = len + 1;
  d->nodes = (TjJ *)malloc(d->cap_nodes * sizeof(TjJ));
  d->strs = (char *)malloc(d->cap_strs);
  if (!d->nodes || !d->strs) {
    tj_json_free(d);
    if (error) *error = "out of memory";
    return NULL;
  }
  P ps = {text, text + len, d, NULL};
  /* a UTF-8 byte order mark is not JSON */
  d->root = value(&ps, 0);
  if (d->root) {
    ws(&ps);
    if (ps.p != ps.end) {
      ps.err = "json: text after the value";
      d->root = NULL;
    }
  }
  if (!d->root) {
    if (error) *error = ps.err ? ps.err : "json: syntax error";
    tj_json_free(d);
    return NULL;
  }
  return d;
}

void tj_json_free(TjJDoc *d) {
  if (!d) return;
  free(d->nodes);
  free(d->strs);
  free(d);
}

const TjJ *tj_json_root(const TjJDoc *d) { return d ? d->root : NULL; }

const TjJ *tj_json_get(const TjJ *obj, const char *key) {
  if (!obj || obj->type != TJ_JOBJ) return NULL;
  const TjJ *found = NULL;
  for (const TjJ *m = obj->child; m; m = m->next)
    if (!strcmp(m->key, key)) found = m;
  return found;
}

int tj_json_get_type(const TjJ *obj, const char *key, int type, const TjJ **out) {
  const TjJ *v = tj_json_get(obj, key);
  if (!v || v->type != type) return 0;
  if (out) *out = v;
  return 1;
}
