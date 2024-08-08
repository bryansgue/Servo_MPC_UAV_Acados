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
  #define CASADI_PREFIX(ID) Drone_ode_complete_constr_h_fun_jac_uxt_zt_ ## ID
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
#define casadi_densify CASADI_PREFIX(densify)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
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

#define CASADI_CAST(x,y) ((x) y)

void casadi_densify(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, casadi_int tr) {
  casadi_int nrow_x, ncol_x, i, el;
  const casadi_int *colind_x, *row_x;
  if (!y) return;
  nrow_x = sp_x[0]; ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x+ncol_x+3;
  casadi_clear(y, nrow_x*ncol_x);
  if (!x) return;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[i + row_x[el]*ncol_x] = CASADI_CAST(casadi_real, *x++);
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[row_x[el]] = CASADI_CAST(casadi_real, *x++);
      }
      y += nrow_x;
    }
  }
}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[9] = {3, 3, 0, 1, 2, 3, 0, 1, 2};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s4[18] = {1, 13, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0};
static const casadi_int casadi_s5[6] = {13, 1, 0, 2, 0, 1};
static const casadi_int casadi_s6[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s7[3] = {0, 0, 0};
static const casadi_int casadi_s8[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s9[8] = {17, 1, 0, 4, 4, 5, 7, 8};
static const casadi_int casadi_s10[3] = {1, 0, 0};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[3] = {0., 0., 1.};
static const casadi_real casadi_c2[3] = {3.2723905139943781e+02, 6.2619368170575160e+02, 6.2622505275946071e+02};
static const casadi_real casadi_c3[9] = {3.0558700000000000e-03, 0., 0., 0., 1.5969500000000000e-03, 0., 0., 0., 1.5968700000000000e-03};

/* Drone_ode_complete_constr_h_fun_jac_uxt_zt:(i0[13],i1[4],i2[],i3[17])->(o0,o1[17x1,4nz],o2[1x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+4, w2, w3, *w4=w+8, w5, w6, w7, w8, w9, w10, w11, *w12=w+27, w13, w14, *w15=w+31, *w16=w+34, *w17=w+43, *w18=w+52, w19, w20, w21, w22, *w23=w+65, w24, *w25=w+70, w26, w27, *w28=w+81, w29, w30, w31, w32, w33, *w34=w+89, *w35=w+92, *w36=w+95, *w37=w+98, *w38=w+101;
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
  /* #9: @6 = sq(@2) */
  w6 = casadi_sq( w2 );
  /* #10: @7 = input[0][1] */
  w7 = arg[0] ? arg[0][1] : 0;
  /* #11: @8 = 0.17 */
  w8 = 1.7000000000000001e-01;
  /* #12: @7 = (@7-@8) */
  w7 -= w8;
  /* #13: @8 = sq(@7) */
  w8 = casadi_sq( w7 );
  /* #14: @6 = (@6+@8) */
  w6 += w8;
  /* #15: @6 = sqrt(@6) */
  w6 = sqrt( w6 );
  /* #16: @8 = (2.*@6) */
  w8 = (2.* w6 );
  /* #17: @3 = (@3/@8) */
  w3 /= w8;
  /* #18: @9 = (-@3) */
  w9 = (- w3 );
  /* #19: (@1[0] = @9) */
  for (rr=w1+0, ss=(&w9); rr!=w1+1; rr+=1) *rr = *ss++;
  /* #20: @9 = (2.*@7) */
  w9 = (2.* w7 );
  /* #21: @10 = ones(13x1,1nz) */
  w10 = 1.;
  /* #22: {NULL, @11, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@10) */
  w11 = w10;
  /* #23: @9 = (@9*@11) */
  w9 *= w11;
  /* #24: @9 = (@9/@8) */
  w9 /= w8;
  /* #25: @10 = (-@9) */
  w10 = (- w9 );
  /* #26: (@1[1] = @10) */
  for (rr=w1+1, ss=(&w10); rr!=w1+2; rr+=1) *rr = *ss++;
  /* #27: @12 = @1' */
  casadi_copy(w1, 2, w12);
  /* #28: @10 = input[0][3] */
  w10 = arg[0] ? arg[0][3] : 0;
  /* #29: @13 = input[0][4] */
  w13 = arg[0] ? arg[0][4] : 0;
  /* #30: @14 = input[0][5] */
  w14 = arg[0] ? arg[0][5] : 0;
  /* #31: @15 = zeros(3x1) */
  casadi_clear(w15, 3);
  /* #32: @16 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w16);
  /* #33: @17 = zeros(3x3) */
  casadi_clear(w17, 9);
  /* #34: @18 = zeros(3x3) */
  casadi_clear(w18, 9);
  /* #35: @19 = input[0][6] */
  w19 = arg[0] ? arg[0][6] : 0;
  /* #36: @20 = input[0][7] */
  w20 = arg[0] ? arg[0][7] : 0;
  /* #37: @21 = input[0][8] */
  w21 = arg[0] ? arg[0][8] : 0;
  /* #38: @22 = input[0][9] */
  w22 = arg[0] ? arg[0][9] : 0;
  /* #39: @23 = vertcat(@19, @20, @21, @22) */
  rr=w23;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w21;
  *rr++ = w22;
  /* #40: @24 = ||@23||_F */
  w24 = sqrt(casadi_dot(4, w23, w23));
  /* #41: @23 = (@23/@24) */
  for (i=0, rr=w23; i<4; ++i) (*rr++) /= w24;
  /* #42: @24 = @23[3] */
  for (rr=(&w24), ss=w23+3; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #43: @24 = (-@24) */
  w24 = (- w24 );
  /* #44: (@18[3] = @24) */
  for (rr=w18+3, ss=(&w24); rr!=w18+4; rr+=1) *rr = *ss++;
  /* #45: @24 = @23[2] */
  for (rr=(&w24), ss=w23+2; ss!=w23+3; ss+=1) *rr++ = *ss;
  /* #46: (@18[6] = @24) */
  for (rr=w18+6, ss=(&w24); rr!=w18+7; rr+=1) *rr = *ss++;
  /* #47: @24 = @23[1] */
  for (rr=(&w24), ss=w23+1; ss!=w23+2; ss+=1) *rr++ = *ss;
  /* #48: @24 = (-@24) */
  w24 = (- w24 );
  /* #49: (@18[7] = @24) */
  for (rr=w18+7, ss=(&w24); rr!=w18+8; rr+=1) *rr = *ss++;
  /* #50: @24 = @23[3] */
  for (rr=(&w24), ss=w23+3; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #51: (@18[1] = @24) */
  for (rr=w18+1, ss=(&w24); rr!=w18+2; rr+=1) *rr = *ss++;
  /* #52: @24 = @23[2] */
  for (rr=(&w24), ss=w23+2; ss!=w23+3; ss+=1) *rr++ = *ss;
  /* #53: @24 = (-@24) */
  w24 = (- w24 );
  /* #54: (@18[2] = @24) */
  for (rr=w18+2, ss=(&w24); rr!=w18+3; rr+=1) *rr = *ss++;
  /* #55: @24 = @23[1] */
  for (rr=(&w24), ss=w23+1; ss!=w23+2; ss+=1) *rr++ = *ss;
  /* #56: (@18[5] = @24) */
  for (rr=w18+5, ss=(&w24); rr!=w18+6; rr+=1) *rr = *ss++;
  /* #57: @25 = (2.*@18) */
  for (i=0, rr=w25, cs=w18; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #58: @17 = mac(@25,@18,@17) */
  for (i=0, rr=w17; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w25+j, tt=w18+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #59: @16 = (@16+@17) */
  for (i=0, rr=w16, cs=w17; i<9; ++i) (*rr++) += (*cs++);
  /* #60: @24 = @23[0] */
  for (rr=(&w24), ss=w23+0; ss!=w23+1; ss+=1) *rr++ = *ss;
  /* #61: @24 = (2.*@24) */
  w24 = (2.* w24 );
  /* #62: @18 = (@24*@18) */
  for (i=0, rr=w18, cs=w18; i<9; ++i) (*rr++)  = (w24*(*cs++));
  /* #63: @16 = (@16+@18) */
  for (i=0, rr=w16, cs=w18; i<9; ++i) (*rr++) += (*cs++);
  /* #64: @24 = 0 */
  w24 = 0.;
  /* #65: @26 = 0 */
  w26 = 0.;
  /* #66: @27 = input[1][0] */
  w27 = arg[1] ? arg[1][0] : 0;
  /* #67: @28 = vertcat(@24, @26, @27) */
  rr=w28;
  *rr++ = w24;
  *rr++ = w26;
  *rr++ = w27;
  /* #68: @15 = mac(@16,@28,@15) */
  for (i=0, rr=w15; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w16+j, tt=w28+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #69: @28 = [0, 0, 1] */
  casadi_copy(casadi_c1, 3, w28);
  /* #70: @24 = 9.81 */
  w24 = 9.8100000000000005e+00;
  /* #71: @28 = (@28*@24) */
  for (i=0, rr=w28; i<3; ++i) (*rr++) *= w24;
  /* #72: @15 = (@15-@28) */
  for (i=0, rr=w15, cs=w28; i<3; ++i) (*rr++) -= (*cs++);
  /* #73: @24 = 0.5 */
  w24 = 5.0000000000000000e-01;
  /* #74: @26 = input[0][10] */
  w26 = arg[0] ? arg[0][10] : 0;
  /* #75: @27 = (@20*@26) */
  w27  = (w20*w26);
  /* #76: @27 = (-@27) */
  w27 = (- w27 );
  /* #77: @29 = input[0][11] */
  w29 = arg[0] ? arg[0][11] : 0;
  /* #78: @30 = (@21*@29) */
  w30  = (w21*w29);
  /* #79: @27 = (@27-@30) */
  w27 -= w30;
  /* #80: @30 = input[0][12] */
  w30 = arg[0] ? arg[0][12] : 0;
  /* #81: @31 = (@22*@30) */
  w31  = (w22*w30);
  /* #82: @27 = (@27-@31) */
  w27 -= w31;
  /* #83: @31 = (@19*@26) */
  w31  = (w19*w26);
  /* #84: @32 = (@21*@30) */
  w32  = (w21*w30);
  /* #85: @31 = (@31+@32) */
  w31 += w32;
  /* #86: @32 = (@22*@29) */
  w32  = (w22*w29);
  /* #87: @31 = (@31-@32) */
  w31 -= w32;
  /* #88: @32 = (@19*@29) */
  w32  = (w19*w29);
  /* #89: @33 = (@20*@30) */
  w33  = (w20*w30);
  /* #90: @32 = (@32-@33) */
  w32 -= w33;
  /* #91: @22 = (@22*@26) */
  w22 *= w26;
  /* #92: @32 = (@32+@22) */
  w32 += w22;
  /* #93: @19 = (@19*@30) */
  w19 *= w30;
  /* #94: @20 = (@20*@29) */
  w20 *= w29;
  /* #95: @19 = (@19+@20) */
  w19 += w20;
  /* #96: @21 = (@21*@26) */
  w21 *= w26;
  /* #97: @19 = (@19-@21) */
  w19 -= w21;
  /* #98: @23 = vertcat(@27, @31, @32, @19) */
  rr=w23;
  *rr++ = w27;
  *rr++ = w31;
  *rr++ = w32;
  *rr++ = w19;
  /* #99: @23 = (@24*@23) */
  for (i=0, rr=w23, cs=w23; i<4; ++i) (*rr++)  = (w24*(*cs++));
  /* #100: @28 = zeros(3x1) */
  casadi_clear(w28, 3);
  /* #101: @34 = 
  [[327.239, 00, 00], 
   [00, 626.194, 00], 
   [00, 00, 626.225]] */
  casadi_copy(casadi_c2, 3, w34);
  /* #102: @24 = input[1][1] */
  w24 = arg[1] ? arg[1][1] : 0;
  /* #103: @27 = input[1][2] */
  w27 = arg[1] ? arg[1][2] : 0;
  /* #104: @31 = input[1][3] */
  w31 = arg[1] ? arg[1][3] : 0;
  /* #105: @35 = vertcat(@24, @27, @31) */
  rr=w35;
  *rr++ = w24;
  *rr++ = w27;
  *rr++ = w31;
  /* #106: @36 = zeros(3x1) */
  casadi_clear(w36, 3);
  /* #107: @16 = 
  [[0.00305587, 0, 0], 
   [0, 0.00159695, 0], 
   [0, 0, 0.00159687]] */
  casadi_copy(casadi_c3, 9, w16);
  /* #108: @37 = vertcat(@26, @29, @30) */
  rr=w37;
  *rr++ = w26;
  *rr++ = w29;
  *rr++ = w30;
  /* #109: @36 = mac(@16,@37,@36) */
  for (i=0, rr=w36; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w16+j, tt=w37+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #110: @24 = @36[2] */
  for (rr=(&w24), ss=w36+2; ss!=w36+3; ss+=1) *rr++ = *ss;
  /* #111: @27 = (@29*@24) */
  w27  = (w29*w24);
  /* #112: @31 = @36[1] */
  for (rr=(&w31), ss=w36+1; ss!=w36+2; ss+=1) *rr++ = *ss;
  /* #113: @32 = (@30*@31) */
  w32  = (w30*w31);
  /* #114: @27 = (@27-@32) */
  w27 -= w32;
  /* #115: @32 = @36[0] */
  for (rr=(&w32), ss=w36+0; ss!=w36+1; ss+=1) *rr++ = *ss;
  /* #116: @30 = (@30*@32) */
  w30 *= w32;
  /* #117: @24 = (@26*@24) */
  w24  = (w26*w24);
  /* #118: @30 = (@30-@24) */
  w30 -= w24;
  /* #119: @26 = (@26*@31) */
  w26 *= w31;
  /* #120: @29 = (@29*@32) */
  w29 *= w32;
  /* #121: @26 = (@26-@29) */
  w26 -= w29;
  /* #122: @36 = vertcat(@27, @30, @26) */
  rr=w36;
  *rr++ = w27;
  *rr++ = w30;
  *rr++ = w26;
  /* #123: @35 = (@35-@36) */
  for (i=0, rr=w35, cs=w36; i<3; ++i) (*rr++) -= (*cs++);
  /* #124: @28 = mac(@34,@35,@28) */
  casadi_mtimes(w34, casadi_s1, w35, casadi_s0, w28, casadi_s0, w, 0);
  /* #125: @38 = vertcat(@10, @13, @14, @15, @23, @28) */
  rr=w38;
  *rr++ = w10;
  *rr++ = w13;
  *rr++ = w14;
  for (i=0, cs=w15; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<3; ++i) *rr++ = *cs++;
  /* #126: @0 = mac(@12,@38,@0) */
  casadi_mtimes(w12, casadi_s4, w38, casadi_s3, (&w0), casadi_s2, w, 0);
  /* #127: @10 = 0.8 */
  w10 = 8.0000000000000004e-01;
  /* #128: @14 = 0.8 */
  w14 = 8.0000000000000004e-01;
  /* #129: @14 = (@14-@6) */
  w14 -= w6;
  /* #130: @10 = (@10*@14) */
  w10 *= w14;
  /* #131: @0 = (@0+@10) */
  w0 += w10;
  /* #132: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #133: @23 = zeros(17x1,4nz) */
  casadi_clear(w23, 4);
  /* #134: @2 = (2.*@2) */
  w2 = (2.* w2 );
  /* #135: @0 = -0.8 */
  w0 = -8.0000000000000004e-01;
  /* #136: @9 = (@9/@8) */
  w9 /= w8;
  /* #137: @9 = (@9*@13) */
  w9 *= w13;
  /* #138: @3 = (@3/@8) */
  w3 /= w8;
  /* #139: @10 = 0 */
  w10 = 0.;
  /* #140: (@38[1] = @10) */
  for (rr=w38+1, ss=(&w10); rr!=w38+2; rr+=1) *rr = *ss++;
  /* #141: @10 = @38[0] */
  for (rr=(&w10), ss=w38+0; ss!=w38+1; ss+=1) *rr++ = *ss;
  /* #142: @3 = (@3*@10) */
  w3 *= w10;
  /* #143: @9 = (@9+@3) */
  w9 += w3;
  /* #144: @9 = (2.*@9) */
  w9 = (2.* w9 );
  /* #145: @0 = (@0+@9) */
  w0 += w9;
  /* #146: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #147: @0 = (@0/@6) */
  w0 /= w6;
  /* #148: @2 = (@2*@0) */
  w2 *= w0;
  /* #149: @10 = (@10/@8) */
  w10 /= w8;
  /* #150: @5 = (@5*@10) */
  w5 *= w10;
  /* #151: @5 = (-@5) */
  w5 = (- w5 );
  /* #152: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #153: @2 = (@2+@5) */
  w2 += w5;
  /* #154: (@23[0] = @2) */
  for (rr=w23+0, ss=(&w2); rr!=w23+1; rr+=1) *rr = *ss++;
  /* #155: @13 = (@13/@8) */
  w13 /= w8;
  /* #156: @11 = (@11*@13) */
  w11 *= w13;
  /* #157: @11 = (-@11) */
  w11 = (- w11 );
  /* #158: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #159: @7 = (2.*@7) */
  w7 = (2.* w7 );
  /* #160: @7 = (@7*@0) */
  w7 *= w0;
  /* #161: @11 = (@11+@7) */
  w11 += w7;
  /* #162: (@23[1] = @11) */
  for (rr=w23+1, ss=(&w11); rr!=w23+2; rr+=1) *rr = *ss++;
  /* #163: @38 = dense(@1) */
  casadi_densify(w1, casadi_s5, w38, 0);
  /* #164: {@11, @7, NULL, NULL, NULL, NULL} = vertsplit(@38) */
  w11 = w38[0];
  w7 = w38[1];
  /* #165: (@23[2] = @11) */
  for (rr=w23+2, ss=(&w11); rr!=w23+3; rr+=1) *rr = *ss++;
  /* #166: (@23[3] = @7) */
  for (rr=w23+3, ss=(&w7); rr!=w23+4; rr+=1) *rr = *ss++;
  /* #167: output[1][0] = @23 */
  casadi_copy(w23, 4, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_constr_h_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_constr_h_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_constr_h_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_complete_constr_h_fun_jac_uxt_zt_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_constr_h_fun_jac_uxt_zt_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_constr_h_fun_jac_uxt_zt_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_constr_h_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s6;
    case 2: return casadi_s7;
    case 3: return casadi_s8;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_constr_h_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s9;
    case 2: return casadi_s10;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_constr_h_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10;
  if (sz_res) *sz_res = 16;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 114;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
