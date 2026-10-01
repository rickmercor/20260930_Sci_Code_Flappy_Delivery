"""
Return normalized Taylor jets for the local Delta and Gamma.



Every jet is a $3\times3$ matrix with normalization

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),

\qquad 0\le i,j\le2.$$

Rows index the $s$ degree; columns index the $t$ degree. Arbitrary

incoming higher-order coefficients are part of the input. Multiplication

means convolution truncated separately in each variable:

$$(AB)_{ij}=\sum_{a=0}^{i}\sum_{b=0}^{j}A_{ab}B_{i-a,j-b}.$$

Division means multiplication by the reciprocal in this same algebra.

Do not interpret entries as point samples or unnormalized derivatives.



The nodes are $X_0-h$, $X_0$, and $X_0+oh$. Evaluate these weight

expressions on the full geometry jets:

$$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad

a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad

a_+=\frac{h^2/c^2+2/o}{2h(o+1)},$$

$$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad

b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad

b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$

Then compute

$$\Delta=a_-W_-+a_0W_0+a_+W_+,\qquad

\Gamma=b_-W_-+b_0W_0+b_+W_+.$$

Treat geometry coordinates independently; do not impose $o=1$ or $c=3h$.

Use binary64 series arithmetic without rounding or finite differences.

The six RBF-FD weights depend nonlinearly on spacing $h$, asymmetry ratio $o$, and shape parameter $c$. Their full Taylor coefficients interact with the nodal-value coefficients through bivariate convolution. Specializing $o=1$ before differentiation removes geometry sensitivities. The normalized convention retains arbitrary coefficients through bidegree $(2,2)$.

Returns
-------
return delta_coeff, gamma_coeff
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spatial_taylor_jets(
    values_coeff: list[list[list[float]]],
    h_coeff: list[list[float]],
    o_coeff: list[list[float]],
    c_coeff: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    r"""Return normalized Taylor jets for the local Delta and Gamma.

    Every jet is a $3\times3$ matrix with normalization
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),
    \qquad 0\le i,j\le2.$$
    Rows index the $s$ degree; columns index the $t$ degree. Arbitrary
    incoming higher-order coefficients are part of the input. Multiplication
    means convolution truncated separately in each variable:
    $$(AB)_{ij}=\sum_{a=0}^{i}\sum_{b=0}^{j}A_{ab}B_{i-a,j-b}.$$
    Division means multiplication by the reciprocal in this same algebra.
    Do not interpret entries as point samples or unnormalized derivatives.

    The nodes are $X_0-h$, $X_0$, and $X_0+oh$. Evaluate these weight
    expressions on the full geometry jets:
    $$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad
    a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad
    a_+=\frac{h^2/c^2+2/o}{2h(o+1)},$$
    $$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad
    b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad
    b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$
    Then compute
    $$\Delta=a_-W_-+a_0W_0+a_+W_+,\qquad
    \Gamma=b_-W_-+b_0W_0+b_+W_+.$$
    Treat geometry coordinates independently; do not impose $o=1$ or $c=3h$.
    Use binary64 series arithmetic without rounding or finite differences.

    Parameters
    ----------
    values_coeff : list[list[list[float]]]
        Exactly three finite $3\times3$ jets in order $(W_-,W_0,W_+)$.
    h_coeff : list[list[float]]
        Finite $3\times3$ jet of the left spacing, with $h_{00}>0$.
    o_coeff : list[list[float]]
        Finite $3\times3$ jet of the right-to-left gap ratio, with $o_{00}>0$.
    c_coeff : list[list[float]]
        Finite $3\times3$ jet of the RBF shape parameter, with $c_{00}>0$.
        The valid numerical domain requires representable finite binary64
        intermediates and output coefficients.

    Returns
    -------
    tuple[list[list[float]], list[list[float]]]
        The normalized $3\times3$ jet of $\Delta$, followed by that of
        $\Gamma$, both as nested lists in the input row/column convention.

    Raises
    ------
    ValueError
        If there are not three value jets, any matrix is not $3\times3$,
        an input coefficient is nonfinite, or a required baseline is nonpositive.
    ArithmeticError
        Native arithmetic exceptions, including ZeroDivisionError and
        OverflowError, may propagate outside the representable numerical domain.
        Malformed objects outside the typed interface may raise TypeError.
    """
    delta_coeff = [[0.0] * 3 for _ in range(3)]
    gamma_coeff = [[0.0] * 3 for _ in range(3)]
    return delta_coeff, gamma_coeff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spatial_taylor_jets(values_coeff,h_coeff,o_coeff,c_coeff):
    import math
    def _read(a):
        if len(a)!=3 or any(len(row)!=3 for row in a):
            raise ValueError("each Taylor array must have shape (3,3)")
        out=[float(a[i][j]) for i in range(3) for j in range(3)]
        if not all(math.isfinite(x) for x in out):
            raise ValueError("finite coefficients required")
        return out
    def _rows(a): return [a[0:3],a[3:6],a[6:9]]
    def _const(x): return [float(x)]+[0.0]*8
    def _add(a,b): return [x+y for x,y in zip(a,b)]
    def _neg(a): return [-x for x in a]
    def _sub(a,b): return _add(a,_neg(b))
    def _scale(a,x): return [x*y for y in a]
    def _mul(a,b):
        out=[0.0]*9
        for i in range(3):
            for j in range(3):
                out[3*i+j]=math.fsum(a[3*k+l]*b[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1))
        return out
    def _reciprocal(a):
        if a[0]==0: raise ZeroDivisionError("zero Taylor constant")
        out=[0.0]*9;out[0]=1/a[0]
        for total in range(1,5):
            for i in range(3):
                j=total-i
                if 0<=j<3:
                    rem=math.fsum(a[3*k+l]*out[3*(i-k)+j-l] for k in range(i+1) for l in range(j+1) if (k,l)!=(0,0))
                    out[3*i+j]=-rem/a[0]
        return out
    def _div(a,b): return _mul(a,_reciprocal(b))
    def _compose(a,c):
        delta=a.copy();delta[0]=0.0
        out=_const(c[4])
        for k in (3,2,1,0): out=_add(_mul(out,delta),_const(c[k]))
        return out
    if len(values_coeff)!=3: raise ValueError("three value Taylor arrays required")
    wm,w0,wp=[_read(x) for x in values_coeff]
    h,o,c=[_read(x) for x in (h_coeff,o_coeff,c_coeff)]
    if min(h[0],o[0],c[0])<=0: raise ValueError("positive h,o,c required")
    one=_const(1);two=_const(2);h2=_mul(h,h);c2=_mul(c,c)
    am=_neg(_div(_mul(o,_add(_scale(c2,2),_mul(h2,o))),_scale(_mul(_mul(c2,h),_add(o,one)),2)))
    a0=_add(_div(_mul(h,_sub(o,one)),_scale(c2,2)),_div(_sub(o,one),_mul(h,o)))
    ap=_div(_add(_div(h2,c2),_div(two,o)),_scale(_mul(h,_add(o,one)),2))
    bm=_div(_sub(_div(two,h2),_div(_mul(_sub(o,_const(3)),o),c2)),_add(o,one))
    b0=_div(_sub(_div(_add(_sub(_mul(o,o),_scale(o,4)),one),c2),_div(two,h2)),o)
    bp=_div(_add(_scale(c2,2),_mul(h2,_sub(_scale(o,3),one))),_mul(_mul(_mul(c2,h2),o),_add(o,one)))
    delta=_add(_add(_mul(am,wm),_mul(a0,w0)),_mul(ap,wp))
    gamma=_add(_add(_mul(bm,wm),_mul(b0,w0)),_mul(bp,wp))
    return _rows(delta),_rows(gamma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    helper="P=[8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100]\nU=[0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4]\nV=[-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3]\np=[[[x,t,0.0],[s,0.0,0.0],[0.0,0.0,0.0]] for x,s,t in zip(P,U,V)]\n"
    return [
        {"setup":helper+"args=(p[:3],p[3],p[4],p[5])","call":"spatial_taylor_jets(*args)","gold_call":"_oracle_spatial_taylor_jets(*args)","tol":2e-11},
        {"setup":helper+"p=[[[x,0.,0.],[0.,0.,0.],[0.,0.,0.]] for x in P];args=(p[:3],p[3],p[4],p[5])","call":"spatial_taylor_jets(*args)","gold_call":"_oracle_spatial_taylor_jets(*args)","tol":2e-11},
        {"setup":helper+"p[4][0][0]=.8;p[1][2][1]=.03;p[2][2][2]=-.02;args=(p[:3],p[3],p[4],p[5])","call":"spatial_taylor_jets(*args)","gold_call":"_oracle_spatial_taylor_jets(*args)","tol":2e-11},
    ]
