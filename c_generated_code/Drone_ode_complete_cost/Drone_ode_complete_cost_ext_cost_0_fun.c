/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_complete_cost_ext_cost_0_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};

/* Drone_ode_complete_cost_ext_cost_0_fun:(i0[13],i1[4],i2[],i3[17])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, *w19=w+22, *w20=w+39, *w21=w+42, *w22=w+55, *w23=w+58, *w24=w+67, *w25=w+71, *w26=w+75, *w27=w+79;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #2: @2 = input[3][0] */
  w2 = arg[3] ? arg[3][0] : 0;
  /* #3: @3 = input[3][1] */
  w3 = arg[3] ? arg[3][1] : 0;
  /* #4: @4 = input[3][2] */
  w4 = arg[3] ? arg[3][2] : 0;
  /* #5: @5 = input[3][3] */
  w5 = arg[3] ? arg[3][3] : 0;
  /* #6: @6 = input[3][4] */
  w6 = arg[3] ? arg[3][4] : 0;
  /* #7: @7 = input[3][5] */
  w7 = arg[3] ? arg[3][5] : 0;
  /* #8: @8 = input[3][6] */
  w8 = arg[3] ? arg[3][6] : 0;
  /* #9: @9 = input[3][7] */
  w9 = arg[3] ? arg[3][7] : 0;
  /* #10: @10 = input[3][8] */
  w10 = arg[3] ? arg[3][8] : 0;
  /* #11: @11 = input[3][9] */
  w11 = arg[3] ? arg[3][9] : 0;
  /* #12: @12 = input[3][10] */
  w12 = arg[3] ? arg[3][10] : 0;
  /* #13: @13 = input[3][11] */
  w13 = arg[3] ? arg[3][11] : 0;
  /* #14: @14 = input[3][12] */
  w14 = arg[3] ? arg[3][12] : 0;
  /* #15: @15 = input[3][13] */
  w15 = arg[3] ? arg[3][13] : 0;
  /* #16: @16 = input[3][14] */
  w16 = arg[3] ? arg[3][14] : 0;
  /* #17: @17 = input[3][15] */
  w17 = arg[3] ? arg[3][15] : 0;
  /* #18: @18 = input[3][16] */
  w18 = arg[3] ? arg[3][16] : 0;
  /* #19: @19 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13, @14, @15, @16, @17, @18) */
  rr=w19;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  *rr++ = w17;
  *rr++ = w18;
  /* #20: @20 = @19[:3] */
  for (rr=w20, ss=w19+0; ss!=w19+3; ss+=1) *rr++ = *ss;
  /* #21: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #22: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #23: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #24: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #25: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #26: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #27: @8 = input[0][6] */
  w8 = arg[0] ? arg[0][6] : 0;
  /* #28: @9 = input[0][7] */
  w9 = arg[0] ? arg[0][7] : 0;
  /* #29: @10 = input[0][8] */
  w10 = arg[0] ? arg[0][8] : 0;
  /* #30: @11 = input[0][9] */
  w11 = arg[0] ? arg[0][9] : 0;
  /* #31: @12 = input[0][10] */
  w12 = arg[0] ? arg[0][10] : 0;
  /* #32: @13 = input[0][11] */
  w13 = arg[0] ? arg[0][11] : 0;
  /* #33: @14 = input[0][12] */
  w14 = arg[0] ? arg[0][12] : 0;
  /* #34: @21 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13, @14) */
  rr=w21;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  /* #35: @22 = @21[:3] */
  for (rr=w22, ss=w21+0; ss!=w21+3; ss+=1) *rr++ = *ss;
  /* #36: @20 = (@20-@22) */
  for (i=0, rr=w20, cs=w22; i<3; ++i) (*rr++) -= (*cs++);
  /* #37: @22 = @20' */
  casadi_copy(w20, 3, w22);
  /* #38: @23 = zeros(3x3) */
  casadi_clear(w23, 9);
  /* #39: @2 = 1 */
  w2 = 1.;
  /* #40: (@23[0] = @2) */
  for (rr=w23+0, ss=(&w2); rr!=w23+1; rr+=1) *rr = *ss++;
  /* #41: @2 = 1 */
  w2 = 1.;
  /* #42: (@23[4] = @2) */
  for (rr=w23+4, ss=(&w2); rr!=w23+5; rr+=1) *rr = *ss++;
  /* #43: @2 = 1 */
  w2 = 1.;
  /* #44: (@23[8] = @2) */
  for (rr=w23+8, ss=(&w2); rr!=w23+9; rr+=1) *rr = *ss++;
  /* #45: @1 = mac(@22,@23,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w22+j, tt=w23+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #46: @0 = mac(@1,@20,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w20+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #47: @2 = 0 */
  w2 = 0.;
  /* #48: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #49: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #50: @4 = input[1][1] */
  w4 = arg[1] ? arg[1][1] : 0;
  /* #51: @5 = input[1][2] */
  w5 = arg[1] ? arg[1][2] : 0;
  /* #52: @6 = input[1][3] */
  w6 = arg[1] ? arg[1][3] : 0;
  /* #53: @25 = vertcat(@3, @4, @5, @6) */
  rr=w25;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #54: @26 = @25' */
  casadi_copy(w25, 4, w26);
  /* #55: @27 = zeros(4x4) */
  casadi_clear(w27, 16);
  /* #56: @3 = 0.001 */
  w3 = 1.0000000000000000e-03;
  /* #57: (@27[0] = @3) */
  for (rr=w27+0, ss=(&w3); rr!=w27+1; rr+=1) *rr = *ss++;
  /* #58: @3 = 200 */
  w3 = 200.;
  /* #59: (@27[5] = @3) */
  for (rr=w27+5, ss=(&w3); rr!=w27+6; rr+=1) *rr = *ss++;
  /* #60: @3 = 200 */
  w3 = 200.;
  /* #61: (@27[10] = @3) */
  for (rr=w27+10, ss=(&w3); rr!=w27+11; rr+=1) *rr = *ss++;
  /* #62: @3 = 200 */
  w3 = 200.;
  /* #63: (@27[15] = @3) */
  for (rr=w27+15, ss=(&w3); rr!=w27+16; rr+=1) *rr = *ss++;
  /* #64: @24 = mac(@26,@27,@24) */
  for (i=0, rr=w24; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w26+j, tt=w27+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #65: @2 = mac(@24,@25,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w24+j, tt=w25+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #66: @0 = (@0+@2) */
  w0 += w2;
  /* #67: @2 = 0 */
  w2 = 0.;
  /* #68: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #69: @9 = (-@9) */
  w9 = (- w9 );
  /* #70: @10 = (-@10) */
  w10 = (- w10 );
  /* #71: @11 = (-@11) */
  w11 = (- w11 );
  /* #72: @24 = vertcat(@8, @9, @10, @11) */
  rr=w24;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  /* #73: @25 = @21[6:10] */
  for (rr=w25, ss=w21+6; ss!=w21+10; ss+=1) *rr++ = *ss;
  /* #74: @8 = ||@25||_F */
  w8 = sqrt(casadi_dot(4, w25, w25));
  /* #75: @24 = (@24/@8) */
  for (i=0, rr=w24; i<4; ++i) (*rr++) /= w8;
  /* #76: {@8, @9, @10, @11} = vertsplit(@24) */
  w8 = w24[0];
  w9 = w24[1];
  w10 = w24[2];
  w11 = w24[3];
  /* #77: @24 = @19[6:10] */
  for (rr=w24, ss=w19+6; ss!=w19+10; ss+=1) *rr++ = *ss;
  /* #78: {@3, @4, @5, @6} = vertsplit(@24) */
  w3 = w24[0];
  w4 = w24[1];
  w5 = w24[2];
  w6 = w24[3];
  /* #79: @7 = (@8*@3) */
  w7  = (w8*w3);
  /* #80: @12 = (@9*@4) */
  w12  = (w9*w4);
  /* #81: @7 = (@7-@12) */
  w7 -= w12;
  /* #82: @12 = (@10*@5) */
  w12  = (w10*w5);
  /* #83: @7 = (@7-@12) */
  w7 -= w12;
  /* #84: @12 = (@11*@6) */
  w12  = (w11*w6);
  /* #85: @7 = (@7-@12) */
  w7 -= w12;
  /* #86: @12 = 0 */
  w12 = 0.;
  /* #87: @12 = (@7<@12) */
  w12  = (w7<w12);
  /* #88: @13 = (@8*@4) */
  w13  = (w8*w4);
  /* #89: @14 = (@9*@3) */
  w14  = (w9*w3);
  /* #90: @13 = (@13+@14) */
  w13 += w14;
  /* #91: @14 = (@10*@6) */
  w14  = (w10*w6);
  /* #92: @13 = (@13+@14) */
  w13 += w14;
  /* #93: @14 = (@11*@5) */
  w14  = (w11*w5);
  /* #94: @13 = (@13-@14) */
  w13 -= w14;
  /* #95: @14 = (@8*@5) */
  w14  = (w8*w5);
  /* #96: @15 = (@9*@6) */
  w15  = (w9*w6);
  /* #97: @14 = (@14-@15) */
  w14 -= w15;
  /* #98: @15 = (@10*@3) */
  w15  = (w10*w3);
  /* #99: @14 = (@14+@15) */
  w14 += w15;
  /* #100: @15 = (@11*@4) */
  w15  = (w11*w4);
  /* #101: @14 = (@14+@15) */
  w14 += w15;
  /* #102: @8 = (@8*@6) */
  w8 *= w6;
  /* #103: @9 = (@9*@5) */
  w9 *= w5;
  /* #104: @8 = (@8+@9) */
  w8 += w9;
  /* #105: @10 = (@10*@4) */
  w10 *= w4;
  /* #106: @8 = (@8-@10) */
  w8 -= w10;
  /* #107: @11 = (@11*@3) */
  w11 *= w3;
  /* #108: @8 = (@8+@11) */
  w8 += w11;
  /* #109: @24 = vertcat(@7, @13, @14, @8) */
  rr=w24;
  *rr++ = w7;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w8;
  /* #110: @25 = (-@24) */
  for (i=0, rr=w25, cs=w24; i<4; ++i) *rr++ = (- *cs++ );
  /* #111: @25 = (@12?@25:0) */
  for (i=0, rr=w25, cs=w25; i<4; ++i) (*rr++)  = (w12?(*cs++):0);
  /* #112: @12 = (!@12) */
  w12 = (! w12 );
  /* #113: @24 = (@12?@24:0) */
  for (i=0, rr=w24, cs=w24; i<4; ++i) (*rr++)  = (w12?(*cs++):0);
  /* #114: @25 = (@25+@24) */
  for (i=0, rr=w25, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #115: @20 = @25[1:4] */
  for (rr=w20, ss=w25+1; ss!=w25+4; ss+=1) *rr++ = *ss;
  /* #116: @22 = (2.*@20) */
  for (i=0, rr=w22, cs=w20; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #117: @12 = ||@20||_F */
  w12 = sqrt(casadi_dot(3, w20, w20));
  /* #118: @7 = @25[0] */
  for (rr=(&w7), ss=w25+0; ss!=w25+1; ss+=1) *rr++ = *ss;
  /* #119: @7 = atan2(@12,@7) */
  w7  = atan2(w12,w7);
  /* #120: @22 = (@22*@7) */
  for (i=0, rr=w22; i<3; ++i) (*rr++) *= w7;
  /* #121: @22 = (@22/@12) */
  for (i=0, rr=w22; i<3; ++i) (*rr++) /= w12;
  /* #122: @20 = @22' */
  casadi_copy(w22, 3, w20);
  /* #123: @23 = zeros(3x3) */
  casadi_clear(w23, 9);
  /* #124: @12 = 1 */
  w12 = 1.;
  /* #125: (@23[0] = @12) */
  for (rr=w23+0, ss=(&w12); rr!=w23+1; rr+=1) *rr = *ss++;
  /* #126: @12 = 1 */
  w12 = 1.;
  /* #127: (@23[4] = @12) */
  for (rr=w23+4, ss=(&w12); rr!=w23+5; rr+=1) *rr = *ss++;
  /* #128: @12 = 1 */
  w12 = 1.;
  /* #129: (@23[8] = @12) */
  for (rr=w23+8, ss=(&w12); rr!=w23+9; rr+=1) *rr = *ss++;
  /* #130: @1 = mac(@20,@23,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w20+j, tt=w23+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #131: @2 = mac(@1,@22,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w22+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #132: @0 = (@0+@2) */
  w0 += w2;
  /* #133: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_cost_ext_cost_0_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_cost_ext_cost_0_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_complete_cost_ext_cost_0_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_cost_ext_cost_0_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_cost_ext_cost_0_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_cost_ext_cost_0_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_cost_ext_cost_0_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 21;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 95;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif