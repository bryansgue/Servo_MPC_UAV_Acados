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
  #define CASADI_PREFIX(ID) Drone_ode_complete_constr_h_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
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

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

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

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[9] = {3, 3, 0, 1, 2, 3, 0, 1, 2};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s4[18] = {1, 13, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0};
static const casadi_int casadi_s5[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[3] = {0, 0, 0};
static const casadi_int casadi_s7[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[3] = {0., 0., 1.};
static const casadi_real casadi_c2[3] = {3.2723905139943781e+02, 6.2619368170575160e+02, 6.2622505275946071e+02};
static const casadi_real casadi_c3[9] = {3.0558700000000000e-03, 0., 0., 0., 1.5969500000000000e-03, 0., 0., 0., 1.5968700000000000e-03};

/* Drone_ode_complete_constr_h_fun:(i0[13],i1[4],i2[],i3[17])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+4, w2, w3, *w4=w+8, w5, w6, w7, *w8=w+23, *w9=w+26, *w10=w+35, *w11=w+44, w12, w13, w14, *w15=w+56, w16, *w17=w+61, w18, w19, *w20=w+72, w21, w22, w23, w24, w25, *w26=w+80, *w27=w+83, *w28=w+86, *w29=w+89, *w30=w+92;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(13x1,2nz) */
  casadi_clear(w1, 2);
  /* #2: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #3: @3 = 2.9 */
  w3 = 2.8999999999999999e+00;
  /* #4: @2 = (@2-@3) */
  w2 -= w3;
  /* #5: @3 = (2.*@2) */
  w3 = (2.* w2 );
  /* #6: @4 = ones(13x1,12nz) */
  casadi_fill(w4, 12, 1.);
  /* #7: {@5, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@4) */
  w5 = w4[0];
  /* #8: @3 = (@3*@5) */
  w3 *= w5;
  /* #9: @2 = sq(@2) */
  w2 = casadi_sq( w2 );
  /* #10: @5 = input[0][1] */
  w5 = arg[0] ? arg[0][1] : 0;
  /* #11: @6 = 0.17 */
  w6 = 1.7000000000000001e-01;
  /* #12: @5 = (@5-@6) */
  w5 -= w6;
  /* #13: @6 = sq(@5) */
  w6 = casadi_sq( w5 );
  /* #14: @2 = (@2+@6) */
  w2 += w6;
  /* #15: @2 = sqrt(@2) */
  w2 = sqrt( w2 );
  /* #16: @6 = (2.*@2) */
  w6 = (2.* w2 );
  /* #17: @3 = (@3/@6) */
  w3 /= w6;
  /* #18: @3 = (-@3) */
  w3 = (- w3 );
  /* #19: (@1[0] = @3) */
  for (rr=w1+0, ss=(&w3); rr!=w1+1; rr+=1) *rr = *ss++;
  /* #20: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #21: @3 = ones(13x1,1nz) */
  w3 = 1.;
  /* #22: {NULL, @7, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@3) */
  w7 = w3;
  /* #23: @5 = (@5*@7) */
  w5 *= w7;
  /* #24: @5 = (@5/@6) */
  w5 /= w6;
  /* #25: @5 = (-@5) */
  w5 = (- w5 );
  /* #26: (@1[1] = @5) */
  for (rr=w1+1, ss=(&w5); rr!=w1+2; rr+=1) *rr = *ss++;
  /* #27: @1 = @1' */
  /* #28: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #29: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #30: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #31: @8 = zeros(3x1) */
  casadi_clear(w8, 3);
  /* #32: @9 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w9);
  /* #33: @10 = zeros(3x3) */
  casadi_clear(w10, 9);
  /* #34: @11 = zeros(3x3) */
  casadi_clear(w11, 9);
  /* #35: @3 = input[0][6] */
  w3 = arg[0] ? arg[0][6] : 0;
  /* #36: @12 = input[0][7] */
  w12 = arg[0] ? arg[0][7] : 0;
  /* #37: @13 = input[0][8] */
  w13 = arg[0] ? arg[0][8] : 0;
  /* #38: @14 = input[0][9] */
  w14 = arg[0] ? arg[0][9] : 0;
  /* #39: @15 = vertcat(@3, @12, @13, @14) */
  rr=w15;
  *rr++ = w3;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  /* #40: @16 = ||@15||_F */
  w16 = sqrt(casadi_dot(4, w15, w15));
  /* #41: @15 = (@15/@16) */
  for (i=0, rr=w15; i<4; ++i) (*rr++) /= w16;
  /* #42: @16 = @15[3] */
  for (rr=(&w16), ss=w15+3; ss!=w15+4; ss+=1) *rr++ = *ss;
  /* #43: @16 = (-@16) */
  w16 = (- w16 );
  /* #44: (@11[3] = @16) */
  for (rr=w11+3, ss=(&w16); rr!=w11+4; rr+=1) *rr = *ss++;
  /* #45: @16 = @15[2] */
  for (rr=(&w16), ss=w15+2; ss!=w15+3; ss+=1) *rr++ = *ss;
  /* #46: (@11[6] = @16) */
  for (rr=w11+6, ss=(&w16); rr!=w11+7; rr+=1) *rr = *ss++;
  /* #47: @16 = @15[1] */
  for (rr=(&w16), ss=w15+1; ss!=w15+2; ss+=1) *rr++ = *ss;
  /* #48: @16 = (-@16) */
  w16 = (- w16 );
  /* #49: (@11[7] = @16) */
  for (rr=w11+7, ss=(&w16); rr!=w11+8; rr+=1) *rr = *ss++;
  /* #50: @16 = @15[3] */
  for (rr=(&w16), ss=w15+3; ss!=w15+4; ss+=1) *rr++ = *ss;
  /* #51: (@11[1] = @16) */
  for (rr=w11+1, ss=(&w16); rr!=w11+2; rr+=1) *rr = *ss++;
  /* #52: @16 = @15[2] */
  for (rr=(&w16), ss=w15+2; ss!=w15+3; ss+=1) *rr++ = *ss;
  /* #53: @16 = (-@16) */
  w16 = (- w16 );
  /* #54: (@11[2] = @16) */
  for (rr=w11+2, ss=(&w16); rr!=w11+3; rr+=1) *rr = *ss++;
  /* #55: @16 = @15[1] */
  for (rr=(&w16), ss=w15+1; ss!=w15+2; ss+=1) *rr++ = *ss;
  /* #56: (@11[5] = @16) */
  for (rr=w11+5, ss=(&w16); rr!=w11+6; rr+=1) *rr = *ss++;
  /* #57: @17 = (2.*@11) */
  for (i=0, rr=w17, cs=w11; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #58: @10 = mac(@17,@11,@10) */
  for (i=0, rr=w10; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w17+j, tt=w11+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #59: @9 = (@9+@10) */
  for (i=0, rr=w9, cs=w10; i<9; ++i) (*rr++) += (*cs++);
  /* #60: @16 = @15[0] */
  for (rr=(&w16), ss=w15+0; ss!=w15+1; ss+=1) *rr++ = *ss;
  /* #61: @16 = (2.*@16) */
  w16 = (2.* w16 );
  /* #62: @11 = (@16*@11) */
  for (i=0, rr=w11, cs=w11; i<9; ++i) (*rr++)  = (w16*(*cs++));
  /* #63: @9 = (@9+@11) */
  for (i=0, rr=w9, cs=w11; i<9; ++i) (*rr++) += (*cs++);
  /* #64: @16 = 0 */
  w16 = 0.;
  /* #65: @18 = 0 */
  w18 = 0.;
  /* #66: @19 = input[1][0] */
  w19 = arg[1] ? arg[1][0] : 0;
  /* #67: @20 = vertcat(@16, @18, @19) */
  rr=w20;
  *rr++ = w16;
  *rr++ = w18;
  *rr++ = w19;
  /* #68: @8 = mac(@9,@20,@8) */
  for (i=0, rr=w8; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w9+j, tt=w20+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #69: @20 = [0, 0, 1] */
  casadi_copy(casadi_c1, 3, w20);
  /* #70: @16 = 9.81 */
  w16 = 9.8100000000000005e+00;
  /* #71: @20 = (@20*@16) */
  for (i=0, rr=w20; i<3; ++i) (*rr++) *= w16;
  /* #72: @8 = (@8-@20) */
  for (i=0, rr=w8, cs=w20; i<3; ++i) (*rr++) -= (*cs++);
  /* #73: @16 = 0.5 */
  w16 = 5.0000000000000000e-01;
  /* #74: @18 = input[0][10] */
  w18 = arg[0] ? arg[0][10] : 0;
  /* #75: @19 = (@12*@18) */
  w19  = (w12*w18);
  /* #76: @19 = (-@19) */
  w19 = (- w19 );
  /* #77: @21 = input[0][11] */
  w21 = arg[0] ? arg[0][11] : 0;
  /* #78: @22 = (@13*@21) */
  w22  = (w13*w21);
  /* #79: @19 = (@19-@22) */
  w19 -= w22;
  /* #80: @22 = input[0][12] */
  w22 = arg[0] ? arg[0][12] : 0;
  /* #81: @23 = (@14*@22) */
  w23  = (w14*w22);
  /* #82: @19 = (@19-@23) */
  w19 -= w23;
  /* #83: @23 = (@3*@18) */
  w23  = (w3*w18);
  /* #84: @24 = (@13*@22) */
  w24  = (w13*w22);
  /* #85: @23 = (@23+@24) */
  w23 += w24;
  /* #86: @24 = (@14*@21) */
  w24  = (w14*w21);
  /* #87: @23 = (@23-@24) */
  w23 -= w24;
  /* #88: @24 = (@3*@21) */
  w24  = (w3*w21);
  /* #89: @25 = (@12*@22) */
  w25  = (w12*w22);
  /* #90: @24 = (@24-@25) */
  w24 -= w25;
  /* #91: @14 = (@14*@18) */
  w14 *= w18;
  /* #92: @24 = (@24+@14) */
  w24 += w14;
  /* #93: @3 = (@3*@22) */
  w3 *= w22;
  /* #94: @12 = (@12*@21) */
  w12 *= w21;
  /* #95: @3 = (@3+@12) */
  w3 += w12;
  /* #96: @13 = (@13*@18) */
  w13 *= w18;
  /* #97: @3 = (@3-@13) */
  w3 -= w13;
  /* #98: @15 = vertcat(@19, @23, @24, @3) */
  rr=w15;
  *rr++ = w19;
  *rr++ = w23;
  *rr++ = w24;
  *rr++ = w3;
  /* #99: @15 = (@16*@15) */
  for (i=0, rr=w15, cs=w15; i<4; ++i) (*rr++)  = (w16*(*cs++));
  /* #100: @20 = zeros(3x1) */
  casadi_clear(w20, 3);
  /* #101: @26 = 
  [[327.239, 00, 00], 
   [00, 626.194, 00], 
   [00, 00, 626.225]] */
  casadi_copy(casadi_c2, 3, w26);
  /* #102: @16 = input[1][1] */
  w16 = arg[1] ? arg[1][1] : 0;
  /* #103: @19 = input[1][2] */
  w19 = arg[1] ? arg[1][2] : 0;
  /* #104: @23 = input[1][3] */
  w23 = arg[1] ? arg[1][3] : 0;
  /* #105: @27 = vertcat(@16, @19, @23) */
  rr=w27;
  *rr++ = w16;
  *rr++ = w19;
  *rr++ = w23;
  /* #106: @28 = zeros(3x1) */
  casadi_clear(w28, 3);
  /* #107: @9 = 
  [[0.00305587, 0, 0], 
   [0, 0.00159695, 0], 
   [0, 0, 0.00159687]] */
  casadi_copy(casadi_c3, 9, w9);
  /* #108: @29 = vertcat(@18, @21, @22) */
  rr=w29;
  *rr++ = w18;
  *rr++ = w21;
  *rr++ = w22;
  /* #109: @28 = mac(@9,@29,@28) */
  for (i=0, rr=w28; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w9+j, tt=w29+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #110: @16 = @28[2] */
  for (rr=(&w16), ss=w28+2; ss!=w28+3; ss+=1) *rr++ = *ss;
  /* #111: @19 = (@21*@16) */
  w19  = (w21*w16);
  /* #112: @23 = @28[1] */
  for (rr=(&w23), ss=w28+1; ss!=w28+2; ss+=1) *rr++ = *ss;
  /* #113: @24 = (@22*@23) */
  w24  = (w22*w23);
  /* #114: @19 = (@19-@24) */
  w19 -= w24;
  /* #115: @24 = @28[0] */
  for (rr=(&w24), ss=w28+0; ss!=w28+1; ss+=1) *rr++ = *ss;
  /* #116: @22 = (@22*@24) */
  w22 *= w24;
  /* #117: @16 = (@18*@16) */
  w16  = (w18*w16);
  /* #118: @22 = (@22-@16) */
  w22 -= w16;
  /* #119: @18 = (@18*@23) */
  w18 *= w23;
  /* #120: @21 = (@21*@24) */
  w21 *= w24;
  /* #121: @18 = (@18-@21) */
  w18 -= w21;
  /* #122: @28 = vertcat(@19, @22, @18) */
  rr=w28;
  *rr++ = w19;
  *rr++ = w22;
  *rr++ = w18;
  /* #123: @27 = (@27-@28) */
  for (i=0, rr=w27, cs=w28; i<3; ++i) (*rr++) -= (*cs++);
  /* #124: @20 = mac(@26,@27,@20) */
  casadi_mtimes(w26, casadi_s1, w27, casadi_s0, w20, casadi_s0, w, 0);
  /* #125: @30 = vertcat(@5, @6, @7, @8, @15, @20) */
  rr=w30;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  for (i=0, cs=w8; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w15; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  /* #126: @0 = mac(@1,@30,@0) */
  casadi_mtimes(w1, casadi_s4, w30, casadi_s3, (&w0), casadi_s2, w, 0);
  /* #127: @5 = 0.8 */
  w5 = 8.0000000000000004e-01;
  /* #128: @6 = 0.8 */
  w6 = 8.0000000000000004e-01;
  /* #129: @6 = (@6-@2) */
  w6 -= w2;
  /* #130: @5 = (@5*@6) */
  w5 *= w6;
  /* #131: @0 = (@0+@5) */
  w0 += w5;
  /* #132: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_constr_h_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_constr_h_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_complete_constr_h_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_constr_h_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_constr_h_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_constr_h_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_constr_h_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10;
  if (sz_res) *sz_res = 14;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 105;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif