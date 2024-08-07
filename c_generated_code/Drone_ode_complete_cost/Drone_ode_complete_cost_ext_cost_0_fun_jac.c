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
  #define CASADI_PREFIX(ID) Drone_ode_complete_cost_ext_cost_0_fun_jac_ ## ID
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
#define casadi_sq CASADI_PREFIX(sq)

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};

/* Drone_ode_complete_cost_ext_cost_0_fun_jac:(i0[13],i1[4],i2[],i3[17])->(o0,o1[17]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cr, *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, *w19=w+22, *w20=w+39, *w21=w+42, *w22=w+55, *w23=w+58, *w24=w+67, *w25=w+71, *w26=w+75, *w27=w+79, *w28=w+95, *w29=w+99, *w30=w+103, *w31=w+107, *w32=w+110, *w33=w+113, *w34=w+116, *w35=w+119, *w36=w+128, *w37=w+144, *w38=w+147;
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
  /* #68: @22 = zeros(1x3) */
  casadi_clear(w22, 3);
  /* #69: @9 = (-@9) */
  w9 = (- w9 );
  /* #70: @10 = (-@10) */
  w10 = (- w10 );
  /* #71: @11 = (-@11) */
  w11 = (- w11 );
  /* #72: @26 = vertcat(@8, @9, @10, @11) */
  rr=w26;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  /* #73: @28 = @21[6:10] */
  for (rr=w28, ss=w21+6; ss!=w21+10; ss+=1) *rr++ = *ss;
  /* #74: @8 = ||@28||_F */
  w8 = sqrt(casadi_dot(4, w28, w28));
  /* #75: @26 = (@26/@8) */
  for (i=0, rr=w26; i<4; ++i) (*rr++) /= w8;
  /* #76: {@9, @10, @11, @3} = vertsplit(@26) */
  w9 = w26[0];
  w10 = w26[1];
  w11 = w26[2];
  w3 = w26[3];
  /* #77: @29 = @19[6:10] */
  for (rr=w29, ss=w19+6; ss!=w19+10; ss+=1) *rr++ = *ss;
  /* #78: {@4, @5, @6, @7} = vertsplit(@29) */
  w4 = w29[0];
  w5 = w29[1];
  w6 = w29[2];
  w7 = w29[3];
  /* #79: @12 = (@9*@4) */
  w12  = (w9*w4);
  /* #80: @13 = (@10*@5) */
  w13  = (w10*w5);
  /* #81: @12 = (@12-@13) */
  w12 -= w13;
  /* #82: @13 = (@11*@6) */
  w13  = (w11*w6);
  /* #83: @12 = (@12-@13) */
  w12 -= w13;
  /* #84: @13 = (@3*@7) */
  w13  = (w3*w7);
  /* #85: @12 = (@12-@13) */
  w12 -= w13;
  /* #86: @13 = 0 */
  w13 = 0.;
  /* #87: @13 = (@12<@13) */
  w13  = (w12<w13);
  /* #88: @14 = (@9*@5) */
  w14  = (w9*w5);
  /* #89: @15 = (@10*@4) */
  w15  = (w10*w4);
  /* #90: @14 = (@14+@15) */
  w14 += w15;
  /* #91: @15 = (@11*@7) */
  w15  = (w11*w7);
  /* #92: @14 = (@14+@15) */
  w14 += w15;
  /* #93: @15 = (@3*@6) */
  w15  = (w3*w6);
  /* #94: @14 = (@14-@15) */
  w14 -= w15;
  /* #95: @15 = (@9*@6) */
  w15  = (w9*w6);
  /* #96: @16 = (@10*@7) */
  w16  = (w10*w7);
  /* #97: @15 = (@15-@16) */
  w15 -= w16;
  /* #98: @16 = (@11*@4) */
  w16  = (w11*w4);
  /* #99: @15 = (@15+@16) */
  w15 += w16;
  /* #100: @16 = (@3*@5) */
  w16  = (w3*w5);
  /* #101: @15 = (@15+@16) */
  w15 += w16;
  /* #102: @9 = (@9*@7) */
  w9 *= w7;
  /* #103: @10 = (@10*@6) */
  w10 *= w6;
  /* #104: @9 = (@9+@10) */
  w9 += w10;
  /* #105: @11 = (@11*@5) */
  w11 *= w5;
  /* #106: @9 = (@9-@11) */
  w9 -= w11;
  /* #107: @3 = (@3*@4) */
  w3 *= w4;
  /* #108: @9 = (@9+@3) */
  w9 += w3;
  /* #109: @29 = vertcat(@12, @14, @15, @9) */
  rr=w29;
  *rr++ = w12;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w9;
  /* #110: @30 = (-@29) */
  for (i=0, rr=w30, cs=w29; i<4; ++i) *rr++ = (- *cs++ );
  /* #111: @30 = (@13?@30:0) */
  for (i=0, rr=w30, cs=w30; i<4; ++i) (*rr++)  = (w13?(*cs++):0);
  /* #112: @12 = (!@13) */
  w12 = (! w13 );
  /* #113: @29 = (@12?@29:0) */
  for (i=0, rr=w29, cs=w29; i<4; ++i) (*rr++)  = (w12?(*cs++):0);
  /* #114: @30 = (@30+@29) */
  for (i=0, rr=w30, cs=w29; i<4; ++i) (*rr++) += (*cs++);
  /* #115: @31 = @30[1:4] */
  for (rr=w31, ss=w30+1; ss!=w30+4; ss+=1) *rr++ = *ss;
  /* #116: @32 = (2.*@31) */
  for (i=0, rr=w32, cs=w31; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #117: @14 = ||@31||_F */
  w14 = sqrt(casadi_dot(3, w31, w31));
  /* #118: @15 = @30[0] */
  for (rr=(&w15), ss=w30+0; ss!=w30+1; ss+=1) *rr++ = *ss;
  /* #119: @9 = atan2(@14,@15) */
  w9  = atan2(w14,w15);
  /* #120: @33 = (@32*@9) */
  for (i=0, rr=w33, cr=w32; i<3; ++i) (*rr++)  = ((*cr++)*w9);
  /* #121: @33 = (@33/@14) */
  for (i=0, rr=w33; i<3; ++i) (*rr++) /= w14;
  /* #122: @34 = @33' */
  casadi_copy(w33, 3, w34);
  /* #123: @35 = zeros(3x3) */
  casadi_clear(w35, 9);
  /* #124: @3 = 1 */
  w3 = 1.;
  /* #125: (@35[0] = @3) */
  for (rr=w35+0, ss=(&w3); rr!=w35+1; rr+=1) *rr = *ss++;
  /* #126: @3 = 1 */
  w3 = 1.;
  /* #127: (@35[4] = @3) */
  for (rr=w35+4, ss=(&w3); rr!=w35+5; rr+=1) *rr = *ss++;
  /* #128: @3 = 1 */
  w3 = 1.;
  /* #129: (@35[8] = @3) */
  for (rr=w35+8, ss=(&w3); rr!=w35+9; rr+=1) *rr = *ss++;
  /* #130: @22 = mac(@34,@35,@22) */
  for (i=0, rr=w22; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w35+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #131: @2 = mac(@22,@33,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w22+j, tt=w33+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #132: @0 = (@0+@2) */
  w0 += w2;
  /* #133: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #134: @24 = @24' */
  /* #135: @30 = zeros(1x4) */
  casadi_clear(w30, 4);
  /* #136: @25 = @25' */
  /* #137: @36 = @27' */
  for (i=0, rr=w36, cs=w27; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #138: @30 = mac(@25,@36,@30) */
  for (i=0, rr=w30; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w25+j, tt=w36+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #139: @30 = @30' */
  /* #140: @24 = (@24+@30) */
  for (i=0, rr=w24, cs=w30; i<4; ++i) (*rr++) += (*cs++);
  /* #141: {@0, @2, @3, @11} = vertsplit(@24) */
  w0 = w24[0];
  w2 = w24[1];
  w3 = w24[2];
  w11 = w24[3];
  /* #142: output[1][0] = @0 */
  if (res[1]) res[1][0] = w0;
  /* #143: output[1][1] = @2 */
  if (res[1]) res[1][1] = w2;
  /* #144: output[1][2] = @3 */
  if (res[1]) res[1][2] = w3;
  /* #145: output[1][3] = @11 */
  if (res[1]) res[1][3] = w11;
  /* #146: @21 = zeros(13x1) */
  casadi_clear(w21, 13);
  /* #147: @26 = (@26/@8) */
  for (i=0, rr=w26; i<4; ++i) (*rr++) /= w8;
  /* #148: @26 = (-@26) */
  for (i=0, rr=w26, cs=w26; i<4; ++i) *rr++ = (- *cs++ );
  /* #149: @11 = 1 */
  w11 = 1.;
  /* #150: @12 = (@12?@11:0) */
  w12  = (w12?w11:0);
  /* #151: @24 = zeros(4x1) */
  casadi_clear(w24, 4);
  /* #152: @11 = sq(@14) */
  w11 = casadi_sq( w14 );
  /* #153: @3 = sq(@15) */
  w3 = casadi_sq( w15 );
  /* #154: @11 = (@11+@3) */
  w11 += w3;
  /* #155: @3 = (@14/@11) */
  w3  = (w14/w11);
  /* #156: @22 = @22' */
  /* #157: @34 = zeros(1x3) */
  casadi_clear(w34, 3);
  /* #158: @37 = @33' */
  casadi_copy(w33, 3, w37);
  /* #159: @38 = @35' */
  for (i=0, rr=w38, cs=w35; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #160: @34 = mac(@37,@38,@34) */
  for (i=0, rr=w34; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w37+j, tt=w38+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #161: @34 = @34' */
  /* #162: @22 = (@22+@34) */
  for (i=0, rr=w22, cs=w34; i<3; ++i) (*rr++) += (*cs++);
  /* #163: @34 = (@22/@14) */
  for (i=0, rr=w34, cr=w22; i<3; ++i) (*rr++)  = ((*cr++)/w14);
  /* #164: @2 = dot(@32, @34) */
  w2 = casadi_dot(3, w32, w34);
  /* #165: @3 = (@3*@2) */
  w3 *= w2;
  /* #166: @3 = (-@3) */
  w3 = (- w3 );
  /* #167: (@24[0] += @3) */
  for (rr=w24+0, ss=(&w3); rr!=w24+1; rr+=1) *rr += *ss++;
  /* #168: @33 = (@33/@14) */
  for (i=0, rr=w33; i<3; ++i) (*rr++) /= w14;
  /* #169: @33 = (-@33) */
  for (i=0, rr=w33, cs=w33; i<3; ++i) *rr++ = (- *cs++ );
  /* #170: @3 = dot(@33, @22) */
  w3 = casadi_dot(3, w33, w22);
  /* #171: @15 = (@15/@11) */
  w15 /= w11;
  /* #172: @15 = (@15*@2) */
  w15 *= w2;
  /* #173: @3 = (@3+@15) */
  w3 += w15;
  /* #174: @3 = (@3/@14) */
  w3 /= w14;
  /* #175: @31 = (@3*@31) */
  for (i=0, rr=w31, cs=w31; i<3; ++i) (*rr++)  = (w3*(*cs++));
  /* #176: @34 = (@9*@34) */
  for (i=0, rr=w34, cs=w34; i<3; ++i) (*rr++)  = (w9*(*cs++));
  /* #177: @34 = (2.*@34) */
  for (i=0, rr=w34, cs=w34; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #178: @31 = (@31+@34) */
  for (i=0, rr=w31, cs=w34; i<3; ++i) (*rr++) += (*cs++);
  /* #179: (@24[1:4] += @31) */
  for (rr=w24+1, ss=w31; rr!=w24+4; rr+=1) *rr += *ss++;
  /* #180: @30 = (@12*@24) */
  for (i=0, rr=w30, cs=w24; i<4; ++i) (*rr++)  = (w12*(*cs++));
  /* #181: @12 = 1 */
  w12 = 1.;
  /* #182: @13 = (@13?@12:0) */
  w13  = (w13?w12:0);
  /* #183: @24 = (@13*@24) */
  for (i=0, rr=w24, cs=w24; i<4; ++i) (*rr++)  = (w13*(*cs++));
  /* #184: @30 = (@30-@24) */
  for (i=0, rr=w30, cs=w24; i<4; ++i) (*rr++) -= (*cs++);
  /* #185: {@13, @12, @9, @3} = vertsplit(@30) */
  w13 = w30[0];
  w12 = w30[1];
  w9 = w30[2];
  w3 = w30[3];
  /* #186: @14 = (@7*@3) */
  w14  = (w7*w3);
  /* #187: @15 = (@6*@9) */
  w15  = (w6*w9);
  /* #188: @14 = (@14+@15) */
  w14 += w15;
  /* #189: @15 = (@5*@12) */
  w15  = (w5*w12);
  /* #190: @14 = (@14+@15) */
  w14 += w15;
  /* #191: @15 = (@4*@13) */
  w15  = (w4*w13);
  /* #192: @14 = (@14+@15) */
  w14 += w15;
  /* #193: @15 = (@6*@3) */
  w15  = (w6*w3);
  /* #194: @2 = (@7*@9) */
  w2  = (w7*w9);
  /* #195: @15 = (@15-@2) */
  w15 -= w2;
  /* #196: @2 = (@4*@12) */
  w2  = (w4*w12);
  /* #197: @15 = (@15+@2) */
  w15 += w2;
  /* #198: @2 = (@5*@13) */
  w2  = (w5*w13);
  /* #199: @15 = (@15-@2) */
  w15 -= w2;
  /* #200: @2 = (@4*@9) */
  w2  = (w4*w9);
  /* #201: @11 = (@5*@3) */
  w11  = (w5*w3);
  /* #202: @2 = (@2-@11) */
  w2 -= w11;
  /* #203: @11 = (@7*@12) */
  w11  = (w7*w12);
  /* #204: @2 = (@2+@11) */
  w2 += w11;
  /* #205: @11 = (@6*@13) */
  w11  = (w6*w13);
  /* #206: @2 = (@2-@11) */
  w2 -= w11;
  /* #207: @4 = (@4*@3) */
  w4 *= w3;
  /* #208: @5 = (@5*@9) */
  w5 *= w9;
  /* #209: @4 = (@4+@5) */
  w4 += w5;
  /* #210: @6 = (@6*@12) */
  w6 *= w12;
  /* #211: @4 = (@4-@6) */
  w4 -= w6;
  /* #212: @7 = (@7*@13) */
  w7 *= w13;
  /* #213: @4 = (@4-@7) */
  w4 -= w7;
  /* #214: @30 = vertcat(@14, @15, @2, @4) */
  rr=w30;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w2;
  *rr++ = w4;
  /* #215: @14 = dot(@26, @30) */
  w14 = casadi_dot(4, w26, w30);
  /* #216: @14 = (@14/@8) */
  w14 /= w8;
  /* #217: @28 = (@14*@28) */
  for (i=0, rr=w28, cs=w28; i<4; ++i) (*rr++)  = (w14*(*cs++));
  /* #218: (@21[6:10] += @28) */
  for (rr=w21+6, ss=w28; rr!=w21+10; rr+=1) *rr += *ss++;
  /* #219: @1 = @1' */
  /* #220: @31 = zeros(1x3) */
  casadi_clear(w31, 3);
  /* #221: @20 = @20' */
  /* #222: @38 = @23' */
  for (i=0, rr=w38, cs=w23; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #223: @31 = mac(@20,@38,@31) */
  for (i=0, rr=w31; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w20+j, tt=w38+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #224: @31 = @31' */
  /* #225: @1 = (@1+@31) */
  for (i=0, rr=w1, cs=w31; i<3; ++i) (*rr++) += (*cs++);
  /* #226: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #227: (@21[:3] += @1) */
  for (rr=w21+0, ss=w1; rr!=w21+3; rr+=1) *rr += *ss++;
  /* #228: {@14, @15, @2, @4, @7, @13, @6, @12, @5, @9, @3, @11, @0} = vertsplit(@21) */
  w14 = w21[0];
  w15 = w21[1];
  w2 = w21[2];
  w4 = w21[3];
  w7 = w21[4];
  w13 = w21[5];
  w6 = w21[6];
  w12 = w21[7];
  w5 = w21[8];
  w9 = w21[9];
  w3 = w21[10];
  w11 = w21[11];
  w0 = w21[12];
  /* #229: output[1][4] = @14 */
  if (res[1]) res[1][4] = w14;
  /* #230: output[1][5] = @15 */
  if (res[1]) res[1][5] = w15;
  /* #231: output[1][6] = @2 */
  if (res[1]) res[1][6] = w2;
  /* #232: output[1][7] = @4 */
  if (res[1]) res[1][7] = w4;
  /* #233: output[1][8] = @7 */
  if (res[1]) res[1][8] = w7;
  /* #234: output[1][9] = @13 */
  if (res[1]) res[1][9] = w13;
  /* #235: @30 = (@30/@8) */
  for (i=0, rr=w30; i<4; ++i) (*rr++) /= w8;
  /* #236: {@8, @13, @7, @4} = vertsplit(@30) */
  w8 = w30[0];
  w13 = w30[1];
  w7 = w30[2];
  w4 = w30[3];
  /* #237: @8 = (@8+@6) */
  w8 += w6;
  /* #238: output[1][10] = @8 */
  if (res[1]) res[1][10] = w8;
  /* #239: @12 = (@12-@13) */
  w12 -= w13;
  /* #240: output[1][11] = @12 */
  if (res[1]) res[1][11] = w12;
  /* #241: @5 = (@5-@7) */
  w5 -= w7;
  /* #242: output[1][12] = @5 */
  if (res[1]) res[1][12] = w5;
  /* #243: @9 = (@9-@4) */
  w9 -= w4;
  /* #244: output[1][13] = @9 */
  if (res[1]) res[1][13] = w9;
  /* #245: output[1][14] = @3 */
  if (res[1]) res[1][14] = w3;
  /* #246: output[1][15] = @11 */
  if (res[1]) res[1][15] = w11;
  /* #247: output[1][16] = @0 */
  if (res[1]) res[1][16] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_cost_ext_cost_0_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_cost_ext_cost_0_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_cost_ext_cost_0_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_complete_cost_ext_cost_0_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_cost_ext_cost_0_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_cost_ext_cost_0_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_cost_ext_cost_0_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_cost_ext_cost_0_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_cost_ext_cost_0_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 21;
  if (sz_res) *sz_res = 15;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 156;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
