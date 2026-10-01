"""
Lift the positive liquidity inverse to a normalized bivariate jet.



Both input and output use $3\times3$ coefficient matrices with

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^jA(0,0),

\qquad0\le i,j\le2.$$

Rows index the $s$ degree and columns the $t$ degree. All nine incoming

entries are authoritative, including arbitrary higher-order coefficients.



The positive branch is specified by

$$x=\Phi(Z)=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad Z>0.$$

Return the unique formal Taylor jet satisfying

$\Phi(Z(s,t))=x(s,t)$ through bidegree $(2,2)$ and the positive scalar

baseline $\Phi(Z_{00})=x_{00}$. The retained entry $(2,2)$ has total

degree four, so composition terms through total order four are required.

The scalar baseline may use positive_z_from_x; its first returned

component is $Z$, followed by dimensionless elasticity and curvature.

All remaining coefficients must satisfy the formal implicit relation.

Use a cancellation-safe analytic evaluation at small roots consistently

with the scalar solve. Point sampling, polynomial fitting, finite or

complex differences, and differentiating iteration traces are not accepted.

The liquidity inverse is implicit and nonlinear. A retained $(2,2)$ coefficient has total degree four, so every analytic composition must account for contributions through total degree four. A stable scalar baseline and the formal relation $\Phi(Z)=x$ determine the jet on the positive analytic branch. The higher-order calculation is limited to representable binary64 arithmetic.

Returns
-------
return z_coeff
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def implicit_z_taylor_jet(
    x_coeff: list[list[float]],
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> list[list[float]]:
    r"""Lift the positive liquidity inverse to a normalized bivariate jet.

    Both input and output use $3\times3$ coefficient matrices with
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^jA(0,0),
    \qquad0\le i,j\le2.$$
    Rows index the $s$ degree and columns the $t$ degree. All nine incoming
    entries are authoritative, including arbitrary higher-order coefficients.

    The positive branch is specified by
    $$x=\Phi(Z)=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad Z>0.$$
    Return the unique formal Taylor jet satisfying
    $\Phi(Z(s,t))=x(s,t)$ through bidegree $(2,2)$ and the positive scalar
    baseline $\Phi(Z_{00})=x_{00}$. The retained entry $(2,2)$ has total
    degree four, so composition terms through total order four are required.
    The scalar baseline may use positive_z_from_x; its first returned
    component is $Z$, followed by dimensionless elasticity and curvature.
    All remaining coefficients must satisfy the formal implicit relation.
    Use a cancellation-safe analytic evaluation at small roots consistently
    with the scalar solve. Point sampling, polynomial fitting, finite or
    complex differences, and differentiating iteration traces are not accepted.

    Parameters
    ----------
    x_coeff : list[list[float]]
        Finite normalized $3\times3$ jet with $x_{00}>0$. The valid numerical
        domain requires finite representable fourth-order intermediates and
        coefficients. Full binary64 coverage is required of the scalar inverse,
        not of this higher-order lift.
    residual_tol : float, default 1e-13
        Finite scalar-root tolerance $0<\mathrm{residual\_tol}<1$.
        With $y=\sqrt{Z_{00}}$ and
        $g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, require
        $|g(y)-\sqrt{x_{00}}|\le\mathrm{residual\_tol}\sqrt{x_{00}}$.
    max_iter : int, default 24
        Positive integer scalar-root iteration budget, with the same
        acceptance-within-budget semantics as positive_z_from_x.

    Returns
    -------
    list[list[float]]
        The finite normalized $3\times3$ jet of the positive $Z(s,t)$,
        preserving the input row and column convention.

    Raises
    ------
    ValueError
        If the input shape differs from $3\times3$, an input is nonfinite,
        $x_{00}\le0$, or the scalar tolerance or iteration budget is invalid.
    ArithmeticError
        If the scalar solve fails, its implicit slope is invalid, or a
        nonfinite inverse coefficient is produced. Native ZeroDivisionError
        or OverflowError may propagate when fourth-order arithmetic cannot
        be represented. Malformed objects outside the typed API may raise TypeError.
    """
    z_coeff = [[0.0] * 3 for _ in range(3)]
    return z_coeff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_implicit_z_taylor_jet(x_coeff,residual_tol=1e-13,max_iter=24):
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
    def _sqrt(a):
        x=a[0]
        if x<=0: raise ValueError("positive square-root constant required")
        z=math.sqrt(x)
        return _compose(a,[z,1/(2*z),-1/(8*x*z),1/(16*x*x*z),-5/(128*x*x*x*z)])
    def _asinh(a):
        x=a[0];d=1+x*x
        return _compose(a,[math.asinh(x),d**-.5,-.5*x*d**-1.5,(2*x*x-1)*d**-2.5/6,x*(3-2*x*x)*d**-3.5/8])
    x=_read(x_coeff)
    if x[0]<=0: raise ValueError("positive baseline liquidity required")
    z0,elasticity,_=_oracle_positive_z_from_x(x[0],residual_tol,max_iter)
    slope=x[0]/(z0*elasticity)
    if not math.isfinite(slope) or slope<=0: raise ArithmeticError("invalid implicit slope")
    def _phi(z):
        root=_sqrt(z)
        if root[0]<.05:
            coefficients=(2/3,-8/15,16/35,-128/315,256/693,-1024/3003,2048/6435)
            term=_mul(root,z);g=_const(0)
            for coefficient in coefficients:
                g=_add(g,_scale(term,coefficient));term=_mul(term,z)
        else:
            g=_sub(root,_div(_asinh(root),_sqrt(_add(_const(1),z))))
        return _mul(g,g)
    z=_const(z0)
    for total in range(1,5):
        for i in range(3):
            j=total-i
            if 0<=j<3:
                remainder=_sub(_phi(z),x)
                z[3*i+j]=-remainder[3*i+j]/slope
    if not all(math.isfinite(v) for v in z): raise ArithmeticError("nonfinite inverse Taylor coefficients")
    return _rows(z)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"x=[[.0172873290940766,-.003,0.],[.005,0.,0.],[0.,0.,0.]]","call":"implicit_z_taylor_jet(x)","gold_call":"_oracle_implicit_z_taylor_jet(x)","tol":2e-10},
        {"setup":"x=[[.08,0.,0.],[0.,0.,0.],[0.,0.,0.]]","call":"implicit_z_taylor_jet(x)","gold_call":"_oracle_implicit_z_taylor_jet(x)","tol":2e-10},
        {"setup":"x=[[.004,-.0002,.00001],[.0003,.00004,-.000002],[.000005,-.000001,.0000003]]","call":"implicit_z_taylor_jet(x)","gold_call":"_oracle_implicit_z_taylor_jet(x)","tol":2e-10},
        {"setup":"x=[[1e-18,-1e-19,1e-20],[2e-19,3e-20,-2e-21],[1e-20,-1e-21,2e-22]]","call":"[[v/1e-6 for v in row] for row in implicit_z_taylor_jet(x)]","gold_call":"[[v/1e-6 for v in row] for row in _oracle_implicit_z_taylor_jet(x)]","tol":2e-10},
    ]
