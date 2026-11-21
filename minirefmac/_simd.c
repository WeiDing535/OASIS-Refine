/*
 * minirefmac/_simd.c -- AVX-512 fused Fc + atomic-gradient kernel.
 *
 * This is STEP 2 of the acceleration roadmap (vForce/SIMD C kernel).  It does
 * NOT change the mathematics of the reciprocal-space direct sum implemented in
 * fastkernel.fused_fc_and_gradients -- it is a drop-in, lower-level
 * re-implementation of the per-(symop, atom-chunk) inner loop using explicit
 * AVX-512 intrinsics:
 *
 *     Fc(h) = sum_op sum_j w_j(h) exp(2*pi*i*h.(R_op x_j + t_op))
 *     w_j(h) = occ_j f_j(h) exp(-B_j stol2)                      (real)
 *
 * and the Wirtinger gradients (A_j(h) = conj(G_h) fc_atom_factor(h) w_j(h)):
 *
 *     d f / d x_j,k = -2*pi * sum_h (R_op^T h)_k * Im[ A_j(h) E_op,j(h) ]
 *     d f / d B_j   = -sum_h stol2[h] * Re[ A_j(h) E_op,j(h) ]
 *
 * The only numerical difference vs the reference numpy path is that the
 * complex exponential E = cos(ang) + i*sin(ang) is evaluated with a custom
 * AVX-512 sincos (range reduction to [-pi/4, pi/4] + degree-8 minimax
 * polynomial).  Verified to ~1e-14 absolute error vs libm (see bench_simd).
 *
 * Interface (C linkage, ctypes):
 *   void simd_fc_grad_panel(
 *       const double* hklT,    // [3*n_ref]  COLUMN-MAJOR: hklT[k*n_ref+n] = hkl[n,k]
 *       const double* AG_re,   // [n_ref]
 *       const double* AG_im,   // [n_ref]   AG = conj(G)*fc_atom_factor
 *       const double* R,       // [9]  rotation 3x3 row-major
 *       const double* t,       // [3]  translation
 *       const double* xyzc,    // [nc*3] fractional coords for the chunk
 *       const double* Wt,      // [nc*n_ref] W transposed: Wt[c*n_ref+n]=W[n*nc+c]
 *       const double* stol2,   // [n_ref]
 *       int n_ref, int nc,
 *       double* fc_re,         // [n_ref]  accumulate in/out
 *       double* fc_im,         // [n_ref]  accumulate in/out
 *       double* gx,            // [nc*3]   accumulate in/out (final*=-2pi)
 *       double* gb);           // [nc]     accumulate in/out (final as-is)
 *
 * NOTE: hkl MUST be supplied transposed (3 x n_ref, column-major) so that each
 * Miller-index column hklT[k*n_ref+n] is contiguous and can be loaded with a
 * single unaligned SIMD load.  Passing the (n_ref x 3) row-major array directly
 * would scramble the indices and read out of bounds.
 */
#include <immintrin.h>

/* --- sin/cos: degree-8 minimax polynomials in u = x*x on [-pi/4, pi/4] --- */
static const double SIN_C[9] = {
    1.0,
    -1.6666666666666460e-01,  8.3333333332378939e-03, -1.9841269740735260e-04,
     2.7557264068236997e-06, -2.5035087763275454e-08,  1.3081358262237441e-10,
     2.6813769520711572e-11, -1.0504736295234574e-11
};
static const double COS_C[9] = {
    1.0000000000000089e+00, -5.0000000000005529e-01,  4.1666666667935587e-02,
   -1.3888889024023104e-03,  2.4801663364902582e-05, -2.7581529286196949e-07,
    2.5253702670702260e-09, -4.3038107452875174e-10,  1.6480524583917244e-10
};

static inline void simd_sincos_pd(__m512d ang, __m512d* s_out, __m512d* c_out) {
    const __m512d inv2pi = _mm512_set1_pd(0.15915494309189533576888376337251);
    const __m512d two_pi = _mm512_set1_pd(6.283185307179586476925286766559);
    const __m512d inv_pi2 = _mm512_set1_pd(0.6366197723675813430763296750024);
    const __m512d pi2     = _mm512_set1_pd(1.5707963267948966192313216916398);
    const __m512d sign    = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000ULL));

    /* reduce ang to a in [-pi, pi] */
    __m512d z = _mm512_mul_pd(ang, inv2pi);          /* ang/(2*pi) */
    __m512d k = _mm512_roundscale_pd(z, 0);          /* nearest integer periods */
    /* a = ang - k*2pi = (z - k) * 2pi   (z-k is in [-0.5,0.5], accurate) */
    __m512d a = _mm512_mul_pd(_mm512_sub_pd(z, k), two_pi);
    /* reduce a to a2 in [-pi/4, pi/4] via quadrant index j in {-2..2} */
    __m512d q = _mm512_mul_pd(a, inv_pi2);
    __m512d j = _mm512_roundscale_pd(q, 0);
    __m512d a2 = _mm512_fnmadd_pd(j, pi2, a);        /* a2 = a - j*(pi/2) */

    __m512d u  = _mm512_mul_pd(a2, a2);
    __m512d ss = _mm512_set1_pd(SIN_C[8]);
    __m512d cc = _mm512_set1_pd(COS_C[8]);
    for (int i = 7; i >= 0; i--) {
        ss = _mm512_fmadd_pd(ss, u, _mm512_set1_pd(SIN_C[i]));
        cc = _mm512_fmadd_pd(cc, u, _mm512_set1_pd(COS_C[i]));
    }
    __m512d sx = _mm512_mul_pd(ss, a2);              /* sin(a2) */
    /* cos(a2) = cc */

    __mmask8 m0  = _mm512_cmpeq_pd_mask(j, _mm512_set1_pd(0.0));
    __mmask8 mp1 = _mm512_cmpeq_pd_mask(j, _mm512_set1_pd(1.0));
    __mmask8 mn1 = _mm512_cmpeq_pd_mask(j, _mm512_set1_pd(-1.0));
    __mmask8 mp2 = _mm512_cmpeq_pd_mask(j, _mm512_set1_pd(2.0));
    __mmask8 mn2 = _mm512_cmpeq_pd_mask(j, _mm512_set1_pd(-2.0));
    __mmask8 m2  = _mm512_kor(mp2, mn2);

    /* sin(a) by quadrant */
    __m512d sout = sx;                                   /* j==0 */
    sout = _mm512_mask_blend_pd(mp1, sout, cc);          /* j== 1 -> +cos */
    sout = _mm512_mask_blend_pd(mn1, sout, _mm512_xor_pd(cc, sign)); /* j==-1 -> -cos */
    sout = _mm512_mask_blend_pd(m2,  sout, _mm512_xor_pd(sx, sign)); /* j==+-2 -> -sin */

    /* cos(a) by quadrant */
    __m512d cout = cc;                                   /* j==0 */
    cout = _mm512_mask_blend_pd(mp1, cout, _mm512_xor_pd(sx, sign)); /* j== 1 -> -sin */
    cout = _mm512_mask_blend_pd(mn1, cout, sx);                     /* j==-1 -> +sin */
    cout = _mm512_mask_blend_pd(m2,  cout, _mm512_xor_pd(cc, sign));/* j==+-2 -> -cos */

    *s_out = sout;
    *c_out = cout;
}

void simd_fc_grad_panel(const double* hklT,
                        const double* AG_re, const double* AG_im,
                        const double* R, const double* t,
                        const double* xyzc, const double* Wt,
                        const double* stol2,
                        int n_ref, int nc,
                        double* fc_re, double* fc_im,
                        double* gx, double* gb) {
    const double R0=R[0],R1=R[1],R2=R[2],R3=R[3],R4=R[4],
                 R5=R[5],R6=R[6],R7=R[7],R8=R[8];
    const double t0=t[0],t1=t[1],t2=t[2];
    const __m512d v2pi = _mm512_set1_pd(6.283185307179586476925286766559);
    const __m512d vzero = _mm512_setzero_pd();

    for (int c = 0; c < nc; c++) {
        double x0 = xyzc[c*3+0], x1 = xyzc[c*3+1], x2 = xyzc[c*3+2];
        /* xt = xc @ R.T + t  ==  R @ xc   (a single column vector) */
        double xt0 = x0*R0 + x1*R1 + x2*R2 + t0;
        double xt1 = x0*R3 + x1*R4 + x2*R5 + t1;
        double xt2 = x0*R6 + x1*R7 + x2*R8 + t2;
        __m512d vxt0 = _mm512_set1_pd(xt0);
        __m512d vxt1 = _mm512_set1_pd(xt1);
        __m512d vxt2 = _mm512_set1_pd(xt2);
        const double* Wc = Wt + (size_t)c * n_ref;

        /* per-(atom, axis) gradient accumulators over reflections */
        __m512d gx0 = _mm512_setzero_pd();
        __m512d gx1 = _mm512_setzero_pd();
        __m512d gx2 = _mm512_setzero_pd();

        for (int n = 0; n < n_ref; n += 8) {
            int rem = n_ref - n;
            __mmask8 mask = (__mmask8)((rem >= 8) ? 0xFF : (0xFF >> (8 - rem)));

            __m512d h0 = _mm512_mask_loadu_pd(vzero, mask, hklT + 0 * n_ref + n);
            __m512d h1 = _mm512_mask_loadu_pd(vzero, mask, hklT + 1 * n_ref + n);
            __m512d h2 = _mm512_mask_loadu_pd(vzero, mask, hklT + 2 * n_ref + n);

            /* ang = 2*pi*(h0*xt0 + h1*xt1 + h2*xt2) */
            __m512d dot = _mm512_fmadd_pd(h0, vxt0,
                                _mm512_fmadd_pd(h1, vxt1,
                                    _mm512_mul_pd(h2, vxt2)));
            __m512d ang = _mm512_mul_pd(v2pi, dot);

            __m512d cs, sn;
            simd_sincos_pd(ang, &sn, &cs);

            __m512d wr  = _mm512_mask_loadu_pd(vzero, mask, Wc + n);

            /* Fc += W*(cs + i*sn)  -- fc is indexed by reflection, lane=reflection */
            __m512d fcr = _mm512_mask_loadu_pd(vzero, mask, fc_re + n);
            __m512d fci = _mm512_mask_loadu_pd(vzero, mask, fc_im + n);
            fcr = _mm512_fmadd_pd(wr, cs, fcr);
            fci = _mm512_fmadd_pd(wr, sn, fci);
            _mm512_mask_storeu_pd(fc_re + n, mask, fcr);
            _mm512_mask_storeu_pd(fc_im + n, mask, fci);

            /* AE = (AG_re + i*AG_im) * W * (cs + i*sn) */
            __m512d agr = _mm512_mask_loadu_pd(vzero, mask, AG_re + n);
            __m512d agi = _mm512_mask_loadu_pd(vzero, mask, AG_im + n);
            __m512d wr_agr = _mm512_mul_pd(wr, agr);
            __m512d wr_agi = _mm512_mul_pd(wr, agi);
            __m512d aer = _mm512_fmsub_pd(wr_agr, cs, _mm512_mul_pd(wr_agi, sn));
            __m512d aei = _mm512_fmadd_pd(wr_agr, sn, _mm512_mul_pd(wr_agi, cs));

            /* gb[c] += -stol2 * Re[AE]  (sum over reflections) */
            __m512d st2 = _mm512_mask_loadu_pd(vzero, mask, stol2 + n);
            __m512d neg_aer_st2 = _mm512_mul_pd(_mm512_sub_pd(vzero, st2), aer);
            gb[c] += _mm512_mask_reduce_add_pd(mask, neg_aer_st2);

            /* gx[c*3+k] += (R^T h)[k] * Im[AE]  (accumulate over reflections) */
            __m512d hR0 = _mm512_fmadd_pd(h0, _mm512_set1_pd(R0),
                                _mm512_fmadd_pd(h1, _mm512_set1_pd(R3),
                                    _mm512_mul_pd(h2, _mm512_set1_pd(R6))));
            __m512d hR1 = _mm512_fmadd_pd(h0, _mm512_set1_pd(R1),
                                _mm512_fmadd_pd(h1, _mm512_set1_pd(R4),
                                    _mm512_mul_pd(h2, _mm512_set1_pd(R7))));
            __m512d hR2 = _mm512_fmadd_pd(h0, _mm512_set1_pd(R2),
                                _mm512_fmadd_pd(h1, _mm512_set1_pd(R5),
                                    _mm512_mul_pd(h2, _mm512_set1_pd(R8))));

            gx0 = _mm512_fmadd_pd(hR0, aei, gx0);
            gx1 = _mm512_fmadd_pd(hR1, aei, gx1);
            gx2 = _mm512_fmadd_pd(hR2, aei, gx2);
        }
        /* fold the reflection-lane accumulators into the (nc,3) gradient array */
        gx[c*3+0] += _mm512_mask_reduce_add_pd(0xFF, gx0);
        gx[c*3+1] += _mm512_mask_reduce_add_pd(0xFF, gx1);
        gx[c*3+2] += _mm512_mask_reduce_add_pd(0xFF, gx2);
    }
}
