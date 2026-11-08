/* Deterministic exp, log and log1p in double precision, and numpy's sum (see internal.h). Copied from
 * the phoneme engine (research/phoneme-engine/c/src/pe_libm.c, pe_ names changed to zf_tj_), with its notices:
 *
 * exp and log are ARM optimized-routines (2018) and log1p is fdlibm, as musl ships them, built the
 * way zipformer-sdk's zf_libm builds its float functions: no FMA, no rounding intrinsics, double_t
 * = double, no errno and no exception signalling. Every step is one IEEE double operation with one
 * rounding, so x86-64, ARMv7, AArch64 and WebAssembly give the same bits (with /fp:precise or
 * -ffp-contract=off). Against the platform libm they differ by at most one ulp.
 *
 * exp, log: Copyright (c) 2018, Arm Limited. SPDX-License-Identifier: MIT.
 * log1p: from FreeBSD /usr/src/lib/msun/src/s_log1p.c:
 *   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *   Developed at SunPro, a Sun Microsystems, Inc. business.
 *   Permission to use, copy, modify, and distribute this software is freely granted, provided that
 *   this notice is preserved.
 * As adapted in musl: Copyright (c) 2005-2020 Rich Felker, et al. SPDX-License-Identifier: MIT. */
#include "internal.h"

#include <stdint.h>

static inline uint64_t asuint64(double f) {
  union {
    double f;
    uint64_t i;
  } u;
  u.f = f;
  return u.i;
}

static inline double asdouble(uint64_t i) {
  union {
    uint64_t i;
    double f;
  } u;
  u.i = i;
  return u.f;
}

/* keeps the compiler from folding an overflow or underflow that must happen at run time */
static double xflow(double y) {
  volatile double v = y;
  return v * y;
}

/* ---- exp ------------------------------------------------------------------------------------- */

#define EXP_TABLE_BITS 7
#define EXP_N (1 << EXP_TABLE_BITS)

static const double exp_invln2N = 0x1.71547652b82fep0 * EXP_N;
static const double exp_negln2hiN = -0x1.62e42fefa0000p-8;
static const double exp_negln2loN = -0x1.cf79abc9e3b3ap-47;
static const double exp_shift = 0x1.8p52;
static const double exp_C2 = 0x1.ffffffffffdbdp-2;
static const double exp_C3 = 0x1.555555555543cp-3;
static const double exp_C4 = 0x1.55555cf172b91p-5;
static const double exp_C5 = 0x1.1111167a4d017p-7;

/* 2^(k/N) ~= H[k]*(1 + T[k]) for int k in [0,N): tab[2k] = asuint64(T[k]), tab[2k+1] =
 * asuint64(H[k]) - (k << 52)/N */
static const uint64_t exp_tab[2 * EXP_N] = {
    0x0ull, 0x3ff0000000000000ull,
    0x3c9b3b4f1a88bf6eull, 0x3feff63da9fb3335ull,
    0xbc7160139cd8dc5dull, 0x3fefec9a3e778061ull,
    0xbc905e7a108766d1ull, 0x3fefe315e86e7f85ull,
    0x3c8cd2523567f613ull, 0x3fefd9b0d3158574ull,
    0xbc8bce8023f98efaull, 0x3fefd06b29ddf6deull,
    0x3c60f74e61e6c861ull, 0x3fefc74518759bc8ull,
    0x3c90a3e45b33d399ull, 0x3fefbe3ecac6f383ull,
    0x3c979aa65d837b6dull, 0x3fefb5586cf9890full,
    0x3c8eb51a92fdeffcull, 0x3fefac922b7247f7ull,
    0x3c3ebe3d702f9cd1ull, 0x3fefa3ec32d3d1a2ull,
    0xbc6a033489906e0bull, 0x3fef9b66affed31bull,
    0xbc9556522a2fbd0eull, 0x3fef9301d0125b51ull,
    0xbc5080ef8c4eea55ull, 0x3fef8abdc06c31ccull,
    0xbc91c923b9d5f416ull, 0x3fef829aaea92de0ull,
    0x3c80d3e3e95c55afull, 0x3fef7a98c8a58e51ull,
    0xbc801b15eaa59348ull, 0x3fef72b83c7d517bull,
    0xbc8f1ff055de323dull, 0x3fef6af9388c8deaull,
    0x3c8b898c3f1353bfull, 0x3fef635beb6fcb75ull,
    0xbc96d99c7611eb26ull, 0x3fef5be084045cd4ull,
    0x3c9aecf73e3a2f60ull, 0x3fef54873168b9aaull,
    0xbc8fe782cb86389dull, 0x3fef4d5022fcd91dull,
    0x3c8a6f4144a6c38dull, 0x3fef463b88628cd6ull,
    0x3c807a05b0e4047dull, 0x3fef3f49917ddc96ull,
    0x3c968efde3a8a894ull, 0x3fef387a6e756238ull,
    0x3c875e18f274487dull, 0x3fef31ce4fb2a63full,
    0x3c80472b981fe7f2ull, 0x3fef2b4565e27cddull,
    0xbc96b87b3f71085eull, 0x3fef24dfe1f56381ull,
    0x3c82f7e16d09ab31ull, 0x3fef1e9df51fdee1ull,
    0xbc3d219b1a6fbffaull, 0x3fef187fd0dad990ull,
    0x3c8b3782720c0ab4ull, 0x3fef1285a6e4030bull,
    0x3c6e149289cecb8full, 0x3fef0cafa93e2f56ull,
    0x3c834d754db0abb6ull, 0x3fef06fe0a31b715ull,
    0x3c864201e2ac744cull, 0x3fef0170fc4cd831ull,
    0x3c8fdd395dd3f84aull, 0x3feefc08b26416ffull,
    0xbc86a3803b8e5b04ull, 0x3feef6c55f929ff1ull,
    0xbc924aedcc4b5068ull, 0x3feef1a7373aa9cbull,
    0xbc9907f81b512d8eull, 0x3feeecae6d05d866ull,
    0xbc71d1e83e9436d2ull, 0x3feee7db34e59ff7ull,
    0xbc991919b3ce1b15ull, 0x3feee32dc313a8e5ull,
    0x3c859f48a72a4c6dull, 0x3feedea64c123422ull,
    0xbc9312607a28698aull, 0x3feeda4504ac801cull,
    0xbc58a78f4817895bull, 0x3feed60a21f72e2aull,
    0xbc7c2c9b67499a1bull, 0x3feed1f5d950a897ull,
    0x3c4363ed60c2ac11ull, 0x3feece086061892dull,
    0x3c9666093b0664efull, 0x3feeca41ed1d0057ull,
    0x3c6ecce1daa10379ull, 0x3feec6a2b5c13cd0ull,
    0x3c93ff8e3f0f1230ull, 0x3feec32af0d7d3deull,
    0x3c7690cebb7aafb0ull, 0x3feebfdad5362a27ull,
    0x3c931dbdeb54e077ull, 0x3feebcb299fddd0dull,
    0xbc8f94340071a38eull, 0x3feeb9b2769d2ca7ull,
    0xbc87deccdc93a349ull, 0x3feeb6daa2cf6642ull,
    0xbc78dec6bd0f385full, 0x3feeb42b569d4f82ull,
    0xbc861246ec7b5cf6ull, 0x3feeb1a4ca5d920full,
    0x3c93350518fdd78eull, 0x3feeaf4736b527daull,
    0x3c7b98b72f8a9b05ull, 0x3feead12d497c7fdull,
    0x3c9063e1e21c5409ull, 0x3feeab07dd485429ull,
    0x3c34c7855019c6eaull, 0x3feea9268a5946b7ull,
    0x3c9432e62b64c035ull, 0x3feea76f15ad2148ull,
    0xbc8ce44a6199769full, 0x3feea5e1b976dc09ull,
    0xbc8c33c53bef4da8ull, 0x3feea47eb03a5585ull,
    0xbc845378892be9aeull, 0x3feea34634ccc320ull,
    0xbc93cedd78565858ull, 0x3feea23882552225ull,
    0x3c5710aa807e1964ull, 0x3feea155d44ca973ull,
    0xbc93b3efbf5e2228ull, 0x3feea09e667f3bcdull,
    0xbc6a12ad8734b982ull, 0x3feea012750bdabfull,
    0xbc6367efb86da9eeull, 0x3fee9fb23c651a2full,
    0xbc80dc3d54e08851ull, 0x3fee9f7df9519484ull,
    0xbc781f647e5a3ecfull, 0x3fee9f75e8ec5f74ull,
    0xbc86ee4ac08b7db0ull, 0x3fee9f9a48a58174ull,
    0xbc8619321e55e68aull, 0x3fee9feb564267c9ull,
    0x3c909ccb5e09d4d3ull, 0x3feea0694fde5d3full,
    0xbc7b32dcb94da51dull, 0x3feea11473eb0187ull,
    0x3c94ecfd5467c06bull, 0x3feea1ed0130c132ull,
    0x3c65ebe1abd66c55ull, 0x3feea2f336cf4e62ull,
    0xbc88a1c52fb3cf42ull, 0x3feea427543e1a12ull,
    0xbc9369b6f13b3734ull, 0x3feea589994cce13ull,
    0xbc805e843a19ff1eull, 0x3feea71a4623c7adull,
    0xbc94d450d872576eull, 0x3feea8d99b4492edull,
    0x3c90ad675b0e8a00ull, 0x3feeaac7d98a6699ull,
    0x3c8db72fc1f0eab4ull, 0x3feeace5422aa0dbull,
    0xbc65b6609cc5e7ffull, 0x3feeaf3216b5448cull,
    0x3c7bf68359f35f44ull, 0x3feeb1ae99157736ull,
    0xbc93091fa71e3d83ull, 0x3feeb45b0b91ffc6ull,
    0xbc5da9b88b6c1e29ull, 0x3feeb737b0cdc5e5ull,
    0xbc6c23f97c90b959ull, 0x3feeba44cbc8520full,
    0xbc92434322f4f9aaull, 0x3feebd829fde4e50ull,
    0xbc85ca6cd7668e4bull, 0x3feec0f170ca07baull,
    0x3c71affc2b91ce27ull, 0x3feec49182a3f090ull,
    0x3c6dd235e10a73bbull, 0x3feec86319e32323ull,
    0xbc87c50422622263ull, 0x3feecc667b5de565ull,
    0x3c8b1c86e3e231d5ull, 0x3feed09bec4a2d33ull,
    0xbc91bbd1d3bcbb15ull, 0x3feed503b23e255dull,
    0x3c90cc319cee31d2ull, 0x3feed99e1330b358ull,
    0x3c8469846e735ab3ull, 0x3feede6b5579fdbfull,
    0xbc82dfcd978e9db4ull, 0x3feee36bbfd3f37aull,
    0x3c8c1a7792cb3387ull, 0x3feee89f995ad3adull,
    0xbc907b8f4ad1d9faull, 0x3feeee07298db666ull,
    0xbc55c3d956dcaebaull, 0x3feef3a2b84f15fbull,
    0xbc90a40e3da6f640ull, 0x3feef9728de5593aull,
    0xbc68d6f438ad9334ull, 0x3feeff76f2fb5e47ull,
    0xbc91eee26b588a35ull, 0x3fef05b030a1064aull,
    0x3c74ffd70a5fddcdull, 0x3fef0c1e904bc1d2ull,
    0xbc91bdfbfa9298acull, 0x3fef12c25bd71e09ull,
    0x3c736eae30af0cb3ull, 0x3fef199bdd85529cull,
    0x3c8ee3325c9ffd94ull, 0x3fef20ab5fffd07aull,
    0x3c84e08fd10959acull, 0x3fef27f12e57d14bull,
    0x3c63cdaf384e1a67ull, 0x3fef2f6d9406e7b5ull,
    0x3c676b2c6c921968ull, 0x3fef3720dcef9069ull,
    0xbc808a1883ccb5d2ull, 0x3fef3f0b555dc3faull,
    0xbc8fad5d3ffffa6full, 0x3fef472d4a07897cull,
    0xbc900dae3875a949ull, 0x3fef4f87080d89f2ull,
    0x3c74a385a63d07a7ull, 0x3fef5818dcfba487ull,
    0xbc82919e2040220full, 0x3fef60e316c98398ull,
    0x3c8e5a50d5c192acull, 0x3fef69e603db3285ull,
    0x3c843a59ac016b4bull, 0x3fef7321f301b460ull,
    0xbc82d52107b43e1full, 0x3fef7c97337b9b5full,
    0xbc892ab93b470dc9ull, 0x3fef864614f5a129ull,
    0x3c74b604603a88d3ull, 0x3fef902ee78b3ff6ull,
    0x3c83c5ec519d7271ull, 0x3fef9a51fbc74c83ull,
    0xbc8ff7128fd391f0ull, 0x3fefa4afa2a490daull,
    0xbc8dae98e223747dull, 0x3fefaf482d8e67f1ull,
    0x3c8ec3bc41aa2008ull, 0x3fefba1bee615a27ull,
    0x3c842b94c3a9eb32ull, 0x3fefc52b376bba97ull,
    0x3c8a64a931d185eeull, 0x3fefd0765b6e4540ull,
    0xbc8e37bae43be3edull, 0x3fefdbfdad9cbe14ull,
    0x3c77893b4d91cd9dull, 0x3fefe7c1819e90d8ull,
    0x3c5305c14160cc89ull, 0x3feff3c22b8f71f1ull,
};

/* scale * (1 + tmp) when the exponent of scale may have overflowed (k is the reduction's integer) */
static double exp_special(double tmp, uint64_t sbits, uint64_t ki) {
  double scale, y;
  if ((ki & 0x80000000) == 0) {
    /* k > 0: the exponent of scale might have overflowed by <= 460 */
    sbits -= 1009ull << 52;
    scale = asdouble(sbits);
    return 0x1p1009 * (scale + scale * tmp);
  }
  /* k < 0: care in the subnormal range */
  sbits += 1022ull << 52;
  scale = asdouble(sbits);
  y = scale + scale * tmp;
  if (y < 1.0) {
    /* round y to the right precision before scaling it into the subnormal range (no double rounding) */
    double hi, lo;
    lo = scale - y + scale * tmp;
    hi = 1.0 + y;
    lo = 1.0 - hi + y + lo;
    y = (hi + lo) - 1.0;
    if (y == 0.0) y = 0.0; /* never -0.0 */
  }
  return 0x1p-1022 * y;
}

static inline uint32_t top12(double x) { return (uint32_t)(asuint64(x) >> 52); }

double zf_tj_exp(double x) {
  uint32_t abstop;
  uint64_t ki, idx, top, sbits;
  double kd, z, r, r2, scale, tail, tmp;

  abstop = top12(x) & 0x7ff;
  if (abstop - top12(0x1p-54) >= top12(512.0) - top12(0x1p-54)) {
    if (abstop - top12(0x1p-54) >= 0x80000000) return 1.0 + x; /* tiny x, and 0 */
    if (abstop >= top12(1024.0)) {
      if (asuint64(x) == 0xfff0000000000000ull) return 0.0; /* exp(-inf) */
      if (abstop >= 0x7ff) return 1.0 + x;                  /* inf or nan */
      if (asuint64(x) >> 63) return xflow(0x1p-767);        /* underflow to 0 */
      return xflow(0x1p769);                                /* overflow to inf */
    }
    abstop = 0; /* large |x|: special cased below */
  }

  /* exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N),2^(1/2N)]; x = ln2/N*k + r, with int k and
   * r in [-ln2/2N, ln2/2N] */
  z = exp_invln2N * x;
  kd = z + exp_shift;
  ki = asuint64(kd);
  kd -= exp_shift;
  r = x + kd * exp_negln2hiN + kd * exp_negln2loN;
  /* 2^(k/N) ~= scale * (1 + tail) */
  idx = 2 * (ki % EXP_N);
  top = ki << (52 - EXP_TABLE_BITS);
  tail = asdouble(exp_tab[idx]);
  /* a valid scale only when -1023*N < k < 1024*N */
  sbits = exp_tab[idx + 1] + top;
  /* exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1) */
  r2 = r * r;
  tmp = tail + r + r2 * (exp_C2 + r * exp_C3) + r2 * r2 * (exp_C4 + r * exp_C5);
  if (abstop == 0) return exp_special(tmp, sbits, ki);
  scale = asdouble(sbits);
  return scale + scale * tmp;
}

/* ---- log ------------------------------------------------------------------------------------- */

#define LOG_TABLE_BITS 7
#define LOG_N (1 << LOG_TABLE_BITS)
#define LOG_OFF 0x3fe6000000000000ull

static const double log_ln2hi = 0x1.62e42fefa3800p-1;
static const double log_ln2lo = 0x1.ef35793c76730p-45;
static const double log_A[5] = {-0x1.0000000000001p-1, 0x1.555555551305bp-2, -0x1.fffffffeb459p-3,
                                0x1.999b324f10111p-3, -0x1.55575e506c89fp-3};
static const double log_B[11] = {-0x1p-1,
                                 0x1.5555555555577p-2,
                                 -0x1.ffffffffffdcbp-3,
                                 0x1.999999995dd0cp-3,
                                 -0x1.55555556745a7p-3,
                                 0x1.24924a344de3p-3,
                                 -0x1.fffffa4423d65p-4,
                                 0x1.c7184282ad6cap-4,
                                 -0x1.999eb43b068ffp-4,
                                 0x1.78182f7afd085p-4,
                                 -0x1.5521375d145cdp-4};

/* tab[i] = {1/c, log(c)} and tab2[i] = {c, c - (double)c} for c near the centre of the ith of N
 * subintervals of [1.6p-1, 1.6p0], chosen so that k*ln2hi + logc and z - chi - clo round exactly */
static const struct {
  double invc, logc;
} log_tab[LOG_N] = {
    {0x1.734f0c3e0de9fp+0, -0x1.7cc7f79e69000p-2},
    {0x1.713786a2ce91fp+0, -0x1.76feec20d0000p-2},
    {0x1.6f26008fab5a0p+0, -0x1.713e31351e000p-2},
    {0x1.6d1a61f138c7dp+0, -0x1.6b85b38287800p-2},
    {0x1.6b1490bc5b4d1p+0, -0x1.65d5590807800p-2},
    {0x1.69147332f0cbap+0, -0x1.602d076180000p-2},
    {0x1.6719f18224223p+0, -0x1.5a8ca86909000p-2},
    {0x1.6524f99a51ed9p+0, -0x1.54f4356035000p-2},
    {0x1.63356aa8f24c4p+0, -0x1.4f637c36b4000p-2},
    {0x1.614b36b9ddc14p+0, -0x1.49da7fda85000p-2},
    {0x1.5f66452c65c4cp+0, -0x1.445923989a800p-2},
    {0x1.5d867b5912c4fp+0, -0x1.3edf439b0b800p-2},
    {0x1.5babccb5b90dep+0, -0x1.396ce448f7000p-2},
    {0x1.59d61f2d91a78p+0, -0x1.3401e17bda000p-2},
    {0x1.5805612465687p+0, -0x1.2e9e2ef468000p-2},
    {0x1.56397cee76bd3p+0, -0x1.2941b3830e000p-2},
    {0x1.54725e2a77f93p+0, -0x1.23ec58cda8800p-2},
    {0x1.52aff42064583p+0, -0x1.1e9e129279000p-2},
    {0x1.50f22dbb2bddfp+0, -0x1.1956d2b48f800p-2},
    {0x1.4f38f4734ded7p+0, -0x1.141679ab9f800p-2},
    {0x1.4d843cfde2840p+0, -0x1.0edd094ef9800p-2},
    {0x1.4bd3ec078a3c8p+0, -0x1.09aa518db1000p-2},
    {0x1.4a27fc3e0258ap+0, -0x1.047e65263b800p-2},
    {0x1.4880524d48434p+0, -0x1.feb224586f000p-3},
    {0x1.46dce1b192d0bp+0, -0x1.f474a7517b000p-3},
    {0x1.453d9d3391854p+0, -0x1.ea4443d103000p-3},
    {0x1.43a2744b4845ap+0, -0x1.e020d44e9b000p-3},
    {0x1.420b54115f8fbp+0, -0x1.d60a22977f000p-3},
    {0x1.40782da3ef4b1p+0, -0x1.cc00104959000p-3},
    {0x1.3ee8f5d57fe8fp+0, -0x1.c202956891000p-3},
    {0x1.3d5d9a00b4ce9p+0, -0x1.b81178d811000p-3},
    {0x1.3bd60c010c12bp+0, -0x1.ae2c9ccd3d000p-3},
    {0x1.3a5242b75dab8p+0, -0x1.a45402e129000p-3},
    {0x1.38d22cd9fd002p+0, -0x1.9a877681df000p-3},
    {0x1.3755bc5847a1cp+0, -0x1.90c6d69483000p-3},
    {0x1.35dce49ad36e2p+0, -0x1.87120a645c000p-3},
    {0x1.34679984dd440p+0, -0x1.7d68fb4143000p-3},
    {0x1.32f5cceffcb24p+0, -0x1.73cb83c627000p-3},
    {0x1.3187775a10d49p+0, -0x1.6a39a9b376000p-3},
    {0x1.301c8373e3990p+0, -0x1.60b3154b7a000p-3},
    {0x1.2eb4ebb95f841p+0, -0x1.5737d76243000p-3},
    {0x1.2d50a0219a9d1p+0, -0x1.4dc7b8fc23000p-3},
    {0x1.2bef9a8b7fd2ap+0, -0x1.4462c51d20000p-3},
    {0x1.2a91c7a0c1babp+0, -0x1.3b08abc830000p-3},
    {0x1.293726014b530p+0, -0x1.31b996b490000p-3},
    {0x1.27dfa5757a1f5p+0, -0x1.2875490a44000p-3},
    {0x1.268b39b1d3bbfp+0, -0x1.1f3b9f879a000p-3},
    {0x1.2539d838ff5bdp+0, -0x1.160c8252ca000p-3},
    {0x1.23eb7aac9083bp+0, -0x1.0ce7f57f72000p-3},
    {0x1.22a012ba940b6p+0, -0x1.03cdc49fea000p-3},
    {0x1.2157996cc4132p+0, -0x1.f57bdbc4b8000p-4},
    {0x1.201201dd2fc9bp+0, -0x1.e370896404000p-4},
    {0x1.1ecf4494d480bp+0, -0x1.d17983ef94000p-4},
    {0x1.1d8f5528f6569p+0, -0x1.bf9674ed8a000p-4},
    {0x1.1c52311577e7cp+0, -0x1.adc79202f6000p-4},
    {0x1.1b17c74cb26e9p+0, -0x1.9c0c3e7288000p-4},
    {0x1.19e010c2c1ab6p+0, -0x1.8a646b372c000p-4},
    {0x1.18ab07bb670bdp+0, -0x1.78d01b3ac0000p-4},
    {0x1.1778a25efbcb6p+0, -0x1.674f145380000p-4},
    {0x1.1648d354c31dap+0, -0x1.55e0e6d878000p-4},
    {0x1.151b990275fddp+0, -0x1.4485cdea1e000p-4},
    {0x1.13f0ea432d24cp+0, -0x1.333d94d6aa000p-4},
    {0x1.12c8b7210f9dap+0, -0x1.22079f8c56000p-4},
    {0x1.11a3028ecb531p+0, -0x1.10e4698622000p-4},
    {0x1.107fbda8434afp+0, -0x1.ffa6c6ad20000p-5},
    {0x1.0f5ee0f4e6bb3p+0, -0x1.dda8d4a774000p-5},
    {0x1.0e4065d2a9fcep+0, -0x1.bbcece4850000p-5},
    {0x1.0d244632ca521p+0, -0x1.9a1894012c000p-5},
    {0x1.0c0a77ce2981ap+0, -0x1.788583302c000p-5},
    {0x1.0af2f83c636d1p+0, -0x1.5715e67d68000p-5},
    {0x1.09ddb98a01339p+0, -0x1.35c8a49658000p-5},
    {0x1.08cabaf52e7dfp+0, -0x1.149e364154000p-5},
    {0x1.07b9f2f4e28fbp+0, -0x1.e72c082eb8000p-6},
    {0x1.06ab58c358f19p+0, -0x1.a55f152528000p-6},
    {0x1.059eea5ecf92cp+0, -0x1.63d62cf818000p-6},
    {0x1.04949cdd12c90p+0, -0x1.228fb8caa0000p-6},
    {0x1.038c6c6f0ada9p+0, -0x1.c317b20f90000p-7},
    {0x1.02865137932a9p+0, -0x1.419355daa0000p-7},
    {0x1.0182427ea7348p+0, -0x1.81203c2ec0000p-8},
    {0x1.008040614b195p+0, -0x1.0040979240000p-9},
    {0x1.fe01ff726fa1ap-1, 0x1.feff384900000p-9},
    {0x1.fa11cc261ea74p-1, 0x1.7dc41353d0000p-7},
    {0x1.f6310b081992ep-1, 0x1.3cea3c4c28000p-6},
    {0x1.f25f63ceeadcdp-1, 0x1.b9fc114890000p-6},
    {0x1.ee9c8039113e7p-1, 0x1.1b0d8ce110000p-5},
    {0x1.eae8078cbb1abp-1, 0x1.58a5bd001c000p-5},
    {0x1.e741aa29d0c9bp-1, 0x1.95c8340d88000p-5},
    {0x1.e3a91830a99b5p-1, 0x1.d276aef578000p-5},
    {0x1.e01e009609a56p-1, 0x1.07598e598c000p-4},
    {0x1.dca01e577bb98p-1, 0x1.253f5e30d2000p-4},
    {0x1.d92f20b7c9103p-1, 0x1.42edd8b380000p-4},
    {0x1.d5cac66fb5ccep-1, 0x1.606598757c000p-4},
    {0x1.d272caa5ede9dp-1, 0x1.7da76356a0000p-4},
    {0x1.cf26e3e6b2ccdp-1, 0x1.9ab434e1c6000p-4},
    {0x1.cbe6da2a77902p-1, 0x1.b78c7bb0d6000p-4},
    {0x1.c8b266d37086dp-1, 0x1.d431332e72000p-4},
    {0x1.c5894bd5d5804p-1, 0x1.f0a3171de6000p-4},
    {0x1.c26b533bb9f8cp-1, 0x1.067152b914000p-3},
    {0x1.bf583eeece73fp-1, 0x1.147858292b000p-3},
    {0x1.bc4fd75db96c1p-1, 0x1.2266ecdca3000p-3},
    {0x1.b951e0c864a28p-1, 0x1.303d7a6c55000p-3},
    {0x1.b65e2c5ef3e2cp-1, 0x1.3dfc33c331000p-3},
    {0x1.b374867c9888bp-1, 0x1.4ba366b7a8000p-3},
    {0x1.b094b211d304ap-1, 0x1.5933928d1f000p-3},
    {0x1.adbe885f2ef7ep-1, 0x1.66acd2418f000p-3},
    {0x1.aaf1d31603da2p-1, 0x1.740f8ec669000p-3},
    {0x1.a82e63fd358a7p-1, 0x1.815c0f51af000p-3},
    {0x1.a5740ef09738bp-1, 0x1.8e92954f68000p-3},
    {0x1.a2c2a90ab4b27p-1, 0x1.9bb3602f84000p-3},
    {0x1.a01a01393f2d1p-1, 0x1.a8bed1c2c0000p-3},
    {0x1.9d79f24db3c1bp-1, 0x1.b5b515c01d000p-3},
    {0x1.9ae2505c7b190p-1, 0x1.c2967ccbcc000p-3},
    {0x1.9852ef297ce2fp-1, 0x1.cf635d5486000p-3},
    {0x1.95cbaeea44b75p-1, 0x1.dc1bd3446c000p-3},
    {0x1.934c69de74838p-1, 0x1.e8c01b8cfe000p-3},
    {0x1.90d4f2f6752e6p-1, 0x1.f5509c0179000p-3},
    {0x1.8e6528effd79dp-1, 0x1.00e6c121fb800p-2},
    {0x1.8bfce9fcc007cp-1, 0x1.071b80e93d000p-2},
    {0x1.899c0dabec30ep-1, 0x1.0d46b9e867000p-2},
    {0x1.87427aa2317fbp-1, 0x1.13687334bd000p-2},
    {0x1.84f00acb39a08p-1, 0x1.1980d67234800p-2},
    {0x1.82a49e8653e55p-1, 0x1.1f8ffe0cc8000p-2},
    {0x1.8060195f40260p-1, 0x1.2595fd7636800p-2},
    {0x1.7e22563e0a329p-1, 0x1.2b9300914a800p-2},
    {0x1.7beb377dcb5adp-1, 0x1.3187210436000p-2},
    {0x1.79baa679725c2p-1, 0x1.377266dec1800p-2},
    {0x1.77907f2170657p-1, 0x1.3d54ffbaf3000p-2},
    {0x1.756cadbd6130cp-1, 0x1.432eee32fe000p-2},
};

static const struct {
  double chi, clo;
} log_tab2[LOG_N] = {
    {0x1.61000014fb66bp-1, 0x1.e026c91425b3cp-56},
    {0x1.63000034db495p-1, 0x1.dbfea48005d41p-55},
    {0x1.650000d94d478p-1, 0x1.e7fa786d6a5b7p-55},
    {0x1.67000074e6fadp-1, 0x1.1fcea6b54254cp-57},
    {0x1.68ffffedf0faep-1, -0x1.c7e274c590efdp-56},
    {0x1.6b0000763c5bcp-1, -0x1.ac16848dcda01p-55},
    {0x1.6d0001e5cc1f6p-1, 0x1.33f1c9d499311p-55},
    {0x1.6efffeb05f63ep-1, -0x1.e80041ae22d53p-56},
    {0x1.710000e86978p-1, 0x1.bff6671097952p-56},
    {0x1.72ffffc67e912p-1, 0x1.c00e226bd8724p-55},
    {0x1.74fffdf81116ap-1, -0x1.e02916ef101d2p-57},
    {0x1.770000f679c9p-1, -0x1.7fc71cd549c74p-57},
    {0x1.78ffffa7ec835p-1, 0x1.1bec19ef50483p-55},
    {0x1.7affffe20c2e6p-1, -0x1.07e1729cc6465p-56},
    {0x1.7cfffed3fc9p-1, -0x1.08072087b8b1cp-55},
    {0x1.7efffe9261a76p-1, 0x1.dc0286d9df9aep-55},
    {0x1.81000049ca3e8p-1, 0x1.97fd251e54c33p-55},
    {0x1.8300017932c8fp-1, -0x1.afee9b630f381p-55},
    {0x1.850000633739cp-1, 0x1.9bfbf6b6535bcp-55},
    {0x1.87000204289c6p-1, -0x1.bbf65f3117b75p-55},
    {0x1.88fffebf57904p-1, -0x1.9006ea23dcb57p-55},
    {0x1.8b00022bc04dfp-1, -0x1.d00df38e04b0ap-56},
    {0x1.8cfffe50c1b8ap-1, -0x1.8007146ff9f05p-55},
    {0x1.8effffc918e43p-1, 0x1.3817bd07a7038p-55},
    {0x1.910001efa5fc7p-1, 0x1.93e9176dfb403p-55},
    {0x1.9300013467bb9p-1, 0x1.f804e4b980276p-56},
    {0x1.94fffe6ee076fp-1, -0x1.f7ef0d9ff622ep-55},
    {0x1.96fffde3c12d1p-1, -0x1.082aa962638bap-56},
    {0x1.98ffff4458a0dp-1, -0x1.7801b9164a8efp-55},
    {0x1.9afffdd982e3ep-1, -0x1.740e08a5a9337p-55},
    {0x1.9cfffed49fb66p-1, 0x1.fce08c19bep-60},
    {0x1.9f00020f19c51p-1, -0x1.a3faa27885b0ap-55},
    {0x1.a10001145b006p-1, 0x1.4ff489958da56p-56},
    {0x1.a300007bbf6fap-1, 0x1.cbeab8a2b6d18p-55},
    {0x1.a500010971d79p-1, 0x1.8fecadd78793p-55},
    {0x1.a70001df52e48p-1, -0x1.f41763dd8abdbp-55},
    {0x1.a90001c593352p-1, -0x1.ebf0284c27612p-55},
    {0x1.ab0002a4f3e4bp-1, -0x1.9fd043cff3f5fp-57},
    {0x1.acfffd7ae1ed1p-1, -0x1.23ee7129070b4p-55},
    {0x1.aefffee510478p-1, 0x1.a063ee00edea3p-57},
    {0x1.b0fffdb650d5bp-1, 0x1.a06c8381f0ab9p-58},
    {0x1.b2ffffeaaca57p-1, -0x1.9011e74233c1dp-56},
    {0x1.b4fffd995badcp-1, -0x1.9ff1068862a9fp-56},
    {0x1.b7000249e659cp-1, 0x1.aff45d0864f3ep-55},
    {0x1.b8ffff987164p-1, 0x1.cfe7796c2c3f9p-56},
    {0x1.bafffd204cb4fp-1, -0x1.3ff27eef22bc4p-57},
    {0x1.bcfffd2415c45p-1, -0x1.cffb7ee3bea21p-57},
    {0x1.beffff86309dfp-1, -0x1.14103972e0b5cp-55},
    {0x1.c0fffe1b57653p-1, 0x1.bc16494b76a19p-55},
    {0x1.c2ffff1fa57e3p-1, -0x1.4feef8d30c6edp-57},
    {0x1.c4fffdcbfe424p-1, -0x1.43f68bcec4775p-55},
    {0x1.c6fffed54b9f7p-1, 0x1.47ea3f053e0ecp-55},
    {0x1.c8fffeb998fd5p-1, 0x1.383068df992f1p-56},
    {0x1.cb0002125219ap-1, -0x1.8fd8e64180e04p-57},
    {0x1.ccfffdd94469cp-1, 0x1.e7ebe1cc7ea72p-55},
    {0x1.cefffeafdc476p-1, 0x1.ebe39ad9f88fep-55},
    {0x1.d1000169af82bp-1, 0x1.57d91a8b95a71p-56},
    {0x1.d30000d0ff71dp-1, 0x1.9c1906970c7dap-55},
    {0x1.d4fffea790fc4p-1, -0x1.80e37c558fe0cp-58},
    {0x1.d70002edc87e5p-1, -0x1.f80d64dc10f44p-56},
    {0x1.d900021dc82aap-1, -0x1.47c8f94fd5c5cp-56},
    {0x1.dafffd86b0283p-1, 0x1.c7f1dc521617ep-55},
    {0x1.dd000296c4739p-1, 0x1.8019eb2ffb153p-55},
    {0x1.defffe54490f5p-1, 0x1.e00d2c652cc89p-57},
    {0x1.e0fffcdabf694p-1, -0x1.f8340202d69d2p-56},
    {0x1.e2fffdb52c8ddp-1, 0x1.b00c1ca1b0864p-56},
    {0x1.e4ffff24216efp-1, 0x1.2ffa8b094ab51p-56},
    {0x1.e6fffe88a5e11p-1, -0x1.7f673b1efbe59p-58},
    {0x1.e9000119eff0dp-1, -0x1.4808d5e0bc801p-55},
    {0x1.eafffdfa51744p-1, 0x1.80006d54320b5p-56},
    {0x1.ed0001a127fa1p-1, -0x1.002f860565c92p-58},
    {0x1.ef00007babcc4p-1, -0x1.540445d35e611p-55},
    {0x1.f0ffff57a8d02p-1, -0x1.ffb3139ef9105p-59},
    {0x1.f30001ee58ac7p-1, 0x1.a81acf2731155p-55},
    {0x1.f4ffff5823494p-1, 0x1.a3f41d4d7c743p-55},
    {0x1.f6ffffca94c6bp-1, -0x1.202f41c987875p-57},
    {0x1.f8fffe1f9c441p-1, 0x1.77dd1f477e74bp-56},
    {0x1.fafffd2e0e37ep-1, -0x1.f01199a7ca331p-57},
    {0x1.fd0001c77e49ep-1, 0x1.181ee4bceacb1p-56},
    {0x1.feffff7e0c331p-1, -0x1.e05370170875ap-57},
    {0x1.00ffff465606ep+0, -0x1.a7ead491c0adap-55},
    {0x1.02ffff3867a58p+0, -0x1.77f69c3fcb2ep-54},
    {0x1.04ffffdfc0d17p+0, 0x1.7bffe34cb945bp-54},
    {0x1.0700003cd4d82p+0, 0x1.20083c0e456cbp-55},
    {0x1.08ffff9f2cbe8p+0, -0x1.dffdfbe37751ap-57},
    {0x1.0b000010cda65p+0, -0x1.13f7faee626ebp-54},
    {0x1.0d00001a4d338p+0, 0x1.07dfa79489ff7p-55},
    {0x1.0effffadafdfdp+0, -0x1.7040570d66bcp-56},
    {0x1.110000bbafd96p+0, 0x1.e80d4846d0b62p-55},
    {0x1.12ffffae5f45dp+0, 0x1.dbffa64fd36efp-54},
    {0x1.150000dd59ad9p+0, 0x1.a0077701250aep-54},
    {0x1.170000f21559ap+0, 0x1.dfdf9e2e3deeep-55},
    {0x1.18ffffc275426p+0, 0x1.10030dc3b7273p-54},
    {0x1.1b000123d3c59p+0, 0x1.97f7980030188p-54},
    {0x1.1cffff8299eb7p+0, -0x1.5f932ab9f8c67p-57},
    {0x1.1effff48ad4p+0, 0x1.37fbf9da75bebp-54},
    {0x1.210000c8b86a4p+0, 0x1.f806b91fd5b22p-54},
    {0x1.2300003854303p+0, 0x1.3ffc2eb9fbf33p-54},
    {0x1.24fffffbcf684p+0, 0x1.601e77e2e2e72p-56},
    {0x1.26ffff52921d9p+0, 0x1.ffcbb767f0c61p-56},
    {0x1.2900014933a3cp+0, -0x1.202ca3c02412bp-56},
    {0x1.2b00014556313p+0, -0x1.2808233f21f02p-54},
    {0x1.2cfffebfe523bp+0, -0x1.8ff7e384fdcf2p-55},
    {0x1.2f0000bb8ad96p+0, -0x1.5ff51503041c5p-55},
    {0x1.30ffffb7ae2afp+0, -0x1.10071885e289dp-55},
    {0x1.32ffffeac5f7fp+0, -0x1.1ff5d3fb7b715p-54},
    {0x1.350000ca66756p+0, 0x1.57f82228b82bdp-54},
    {0x1.3700011fbf721p+0, 0x1.000bac40dd5ccp-55},
    {0x1.38ffff9592fb9p+0, -0x1.43f9d2db2a751p-54},
    {0x1.3b00004ddd242p+0, 0x1.57f6b707638e1p-55},
    {0x1.3cffff5b2c957p+0, 0x1.a023a10bf1231p-56},
    {0x1.3efffeab0b418p+0, 0x1.87f6d66b152bp-54},
    {0x1.410001532aff4p+0, 0x1.7f8375f198524p-57},
    {0x1.4300017478b29p+0, 0x1.301e672dc5143p-55},
    {0x1.44fffe795b463p+0, 0x1.9ff69b8b2895ap-55},
    {0x1.46fffe80475ep+0, -0x1.5c0b19bc2f254p-54},
    {0x1.48fffef6fc1e7p+0, 0x1.b4009f23a2a72p-54},
    {0x1.4afffe5bea704p+0, -0x1.4ffb7bf0d7d45p-54},
    {0x1.4d000171027dep+0, -0x1.9c06471dc6a3dp-54},
    {0x1.4f0000ff03ee2p+0, 0x1.77f890b85531cp-54},
    {0x1.5100012dc4bd1p+0, 0x1.004657166a436p-57},
    {0x1.530001605277ap+0, -0x1.6bfcece233209p-54},
    {0x1.54fffecdb704cp+0, -0x1.902720505a1d7p-55},
    {0x1.56fffef5f54a9p+0, 0x1.bbfe60ec96412p-54},
    {0x1.5900017e61012p+0, 0x1.87ec581afef9p-55},
    {0x1.5b00003c93e92p+0, -0x1.f41080abf0ccp-54},
    {0x1.5d0001d4919bcp+0, -0x1.8812afb254729p-54},
    {0x1.5efffe7b87a89p+0, -0x1.47eb780ed6904p-54},
};

double zf_tj_log(double x) {
  double w, z, r, r2, r3, y, invc, logc, kd, hi, lo, rhi, rlo;
  uint64_t ix, iz, tmp;
  uint32_t top;
  int k, i;

  ix = asuint64(x);
  top = (uint32_t)(ix >> 48);
  if (ix - asuint64(1.0 - 0x1p-4) < asuint64(1.0 + 0x1.09p-4) - asuint64(1.0 - 0x1p-4)) {
    /* close to 1.0 */
    if (ix == asuint64(1.0)) return 0;
    r = x - 1.0;
    r2 = r * r;
    r3 = r * r2;
    y = r3 * (log_B[1] + r * log_B[2] + r2 * log_B[3] +
              r3 * (log_B[4] + r * log_B[5] + r2 * log_B[6] +
                    r3 * (log_B[7] + r * log_B[8] + r2 * log_B[9] + r3 * log_B[10])));
    w = r * 0x1p27;
    rhi = r + w - w;
    rlo = r - rhi;
    w = rhi * rhi * log_B[0]; /* B[0] == -0.5 */
    hi = r + w;
    lo = r - hi + w;
    lo += log_B[0] * rlo * (rhi + r);
    y += lo;
    y += hi;
    return y;
  }
  if (top - 0x0010 >= 0x7ff0 - 0x0010) {
    /* x < 0x1p-1022, inf or nan */
    if (ix * 2 == 0) return asdouble(0xfff0000000000000ull);                 /* log(0) = -inf */
    if (ix == 0x7ff0000000000000ull) return x;                               /* log(inf) = inf */
    if ((top & 0x8000) || (top & 0x7ff0) == 0x7ff0) return (x - x) / (x - x); /* negative or nan */
    /* subnormal: normalise */
    ix = asuint64(x * 0x1p52);
    ix -= 52ull << 52;
  }

  /* x = 2^k z, z in [OFF, 2*OFF) exactly; c is near the centre of z's subinterval */
  tmp = ix - LOG_OFF;
  i = (int)((tmp >> (52 - LOG_TABLE_BITS)) % LOG_N);
  k = (int)((int64_t)tmp >> 52); /* arithmetic shift */
  iz = ix - (tmp & 0xfffull << 52);
  invc = log_tab[i].invc;
  logc = log_tab[i].logc;
  z = asdouble(iz);

  /* log(x) = log1p(z/c-1) + log(c) + k*Ln2; r ~= z/c - 1, |r| < 1/(2*N), without fma */
  r = (z - log_tab2[i].chi - log_tab2[i].clo) * invc;
  kd = (double)k;

  /* hi + lo = r + log(c) + k*Ln2 */
  w = kd * log_ln2hi + logc;
  hi = w + r;
  lo = w - hi + r + kd * log_ln2lo;

  /* log(x) = lo + (log1p(r) - r) + hi */
  r2 = r * r;
  y = lo + r2 * log_A[0] + r * r2 * (log_A[1] + r * log_A[2] + r2 * (log_A[3] + r * log_A[4])) + hi;
  return y;
}

/* ---- log1p ----------------------------------------------------------------------------------- */

static const double ln2_hi = 6.93147180369123816490e-01, /* 3fe62e42 fee00000 */
    ln2_lo = 1.90821492927058770002e-10,                 /* 3dea39ef 35793c76 */
    Lg1 = 6.666666666666735130e-01,                      /* 3FE55555 55555593 */
    Lg2 = 3.999999999940941908e-01,                      /* 3FD99999 9997FA04 */
    Lg3 = 2.857142874366239149e-01,                      /* 3FD24924 94229359 */
    Lg4 = 2.222219843214978396e-01,                      /* 3FCC71C5 1D8E78AF */
    Lg5 = 1.818357216161805012e-01,                      /* 3FC74664 96CB03DE */
    Lg6 = 1.531383769920937332e-01,                      /* 3FC39A09 D078C69F */
    Lg7 = 1.479819860511658591e-01;                      /* 3FC2F112 DF3E5244 */

/* 1+x = 2^k * (1+f) with sqrt(2)/2 < 1+f < sqrt(2): log1p(x) = k*ln2 + log(1+f) + c/u, where c is
 * the rounding error of u = 1+x */
double zf_tj_log1p(double x) {
  union {
    double f;
    uint64_t i;
  } u;
  double hfsq, f = 0, c = 0, s, z, R, w, t1, t2, dk;
  uint32_t hx, hu;
  int k;

  u.f = x;
  hx = (uint32_t)(u.i >> 32);
  k = 1;
  if (hx < 0x3fda827a || hx >> 31) { /* 1+x < sqrt(2)+ */
    if (hx >= 0xbff00000) {          /* x <= -1.0 */
      if (x == -1) return asdouble(0xfff0000000000000ull); /* log1p(-1) = -inf */
      return (x - x) / (x - x);                            /* log1p(x<-1) = nan */
    }
    if (hx << 1 < 0x3ca00000u << 1) return x; /* |x| < 2**-53 */
    if (hx <= 0xbfd2bec4) {                  /* sqrt(2)/2- <= 1+x < sqrt(2)+ */
      k = 0;
      c = 0;
      f = x;
    }
  } else if (hx >= 0x7ff00000) {
    return x;
  }
  if (k) {
    u.f = 1 + x;
    hu = (uint32_t)(u.i >> 32);
    hu += 0x3ff00000 - 0x3fe6a09e;
    k = (int)(hu >> 20) - 0x3ff;
    /* correction term ~ log(1+x)-log(u), avoid underflow in c/u */
    if (k < 54) {
      c = k >= 2 ? 1 - (u.f - x) : x - (u.f - 1);
      c /= u.f;
    } else {
      c = 0;
    }
    /* reduce u into [sqrt(2)/2, sqrt(2)] */
    hu = (hu & 0x000fffff) + 0x3fe6a09e;
    u.i = (uint64_t)hu << 32 | (u.i & 0xffffffff);
    f = u.f - 1;
  }
  hfsq = 0.5 * f * f;
  s = f / (2.0 + f);
  z = s * s;
  w = z * z;
  t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  R = t2 + t1;
  dk = k;
  return s * (hfsq + R) + (dk * ln2_lo + c) - hfsq + f + dk * ln2_hi;
}

/* ---- numpy's sum ----------------------------------------------------------------------------- */

/* numpy's pairwise_sum: eight accumulators up to 128 elements, halves (at a multiple of 8) beyond.
 * add.reduce starts from 0.0 and feeds it the array in buffers of 8192 elements. */
#define PW_BLOCK 128
#define NP_BUFSIZE 8192

static double pw_f64(const double *a, int n) {
  if (n < 8) {
    double res = 0.;
    for (int i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= PW_BLOCK) {
    double r[8], res;
    int i;
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int n2 = n / 2;
  n2 -= n2 % 8;
  return pw_f64(a, n2) + pw_f64(a + n2, n - n2);
}

static float pw_f32(const float *a, int n) {
  if (n < 8) {
    float res = 0.f;
    for (int i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= PW_BLOCK) {
    float r[8], res;
    int i;
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  int n2 = n / 2;
  n2 -= n2 % 8;
  return pw_f32(a, n2) + pw_f32(a + n2, n - n2);
}

double zf_tj_np_sum_f64(const double *a, int n) {
  double acc = 0.0;
  for (int k = 0; k < n; k += NP_BUFSIZE) acc += pw_f64(a + k, n - k < NP_BUFSIZE ? n - k : NP_BUFSIZE);
  return acc;
}

float zf_tj_np_sum_f32(const float *a, int n) {
  float acc = 0.f;
  for (int k = 0; k < n; k += NP_BUFSIZE) acc += pw_f32(a + k, n - k < NP_BUFSIZE ? n - k : NP_BUFSIZE);
  return acc;
}
