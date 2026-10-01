"""
Evaluate a frozen-neighbor local RK4 update in bivariate Taylor algebra.



Each $3\times3$ matrix uses normalized coefficients

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),

\qquad0\le i,j\le2.$$

Rows index the $s$ degree and columns the $t$ degree. Preserve arbitrary

higher-order incoming entries and truncate each degree separately.

Products and analytic compositions use this Taylor algebra, not pointwise

matrix arithmetic, finite differences, or scalar sampling.



For a stage central jet $Y$ and time jet $\theta$, obtain $(\Delta,\Gamma)$

by applying spatial_taylor_jets to $(W_-,Y,W_+)$ and geometry $(h,o,c)$.

This preceding function defines the six nonuniform RBF-FD weight jets.

Require $\Gamma_{00}>0$ at every stage and recompute

$$x=\exp(r\theta)\alpha^2X^2\Gamma,$$

$$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad Z_{00}>0,$$

$$F(Y,\theta)=\frac12\sigma_0^2(1+Z)X^2\Gamma+(r-q)X\Delta-rY.$$

Use implicit_z_taylor_jet for the positive inverse, passing the root controls.

Only the central value and time change between stages. Keep the neighbor

jets and all other parameter jets fixed during stage evolution while

retaining all their perturbation coefficients. The stages and result are

$$k_1=F(W_0,\tau),$$

$$k_2=F(W_0+\psi k_1/2,\tau+\psi/2),$$

$$k_3=F(W_0+\psi k_2/2,\tau+\psi/2),$$

$$k_4=F(W_0+\psi k_3,\tau+\psi),$$

$$R=\frac{k_1+2k_2+2k_3+k_4}{6}.$$

Each stage starts from the original central value plus its own increment.

The output is the annualized increment, not the next option-value jet.

The implicit volatility depends on the stage Gamma $\Gamma$, the discount term on the stage central value, and the liquidity factor on the stage time. The full nonlinear row is therefore recomputed at each of the four RK4 stages. The weighted rate $(k_1+2k_2+2k_3+k_4)/6$ equals the annualized increment directly, avoiding subtraction of nearly equal option values.

Returns
-------
return annualized_coeff, stage_rows
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rk4_taylor_step(
    p_coeff: list[list[list[float]]],
    psi: float = 0.002,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[list[list[float]], tuple[tuple[float, ...], ...]]:
    r"""Evaluate a frozen-neighbor local RK4 update in bivariate Taylor algebra.

    Each $3\times3$ matrix uses normalized coefficients
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),
    \qquad0\le i,j\le2.$$
    Rows index the $s$ degree and columns the $t$ degree. Preserve arbitrary
    higher-order incoming entries and truncate each degree separately.
    Products and analytic compositions use this Taylor algebra, not pointwise
    matrix arithmetic, finite differences, or scalar sampling.

    For a stage central jet $Y$ and time jet $\theta$, obtain $(\Delta,\Gamma)$
    by applying spatial_taylor_jets to $(W_-,Y,W_+)$ and geometry $(h,o,c)$.
    This preceding function defines the six nonuniform RBF-FD weight jets.
    Require $\Gamma_{00}>0$ at every stage and recompute
    $$x=\exp(r\theta)\alpha^2X^2\Gamma,$$
    $$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad Z_{00}>0,$$
    $$F(Y,\theta)=\frac12\sigma_0^2(1+Z)X^2\Gamma+(r-q)X\Delta-rY.$$
    Use implicit_z_taylor_jet for the positive inverse, passing the root controls.
    Only the central value and time change between stages. Keep the neighbor
    jets and all other parameter jets fixed during stage evolution while
    retaining all their perturbation coefficients. The stages and result are
    $$k_1=F(W_0,\tau),$$
    $$k_2=F(W_0+\psi k_1/2,\tau+\psi/2),$$
    $$k_3=F(W_0+\psi k_2/2,\tau+\psi/2),$$
    $$k_4=F(W_0+\psi k_3,\tau+\psi),$$
    $$R=\frac{k_1+2k_2+2k_3+k_4}{6}.$$
    Each stage starts from the original central value plus its own increment.
    The output is the annualized increment, not the next option-value jet.

    Parameters
    ----------
    p_coeff : list[list[list[float]]]
        Exactly twelve finite $3\times3$ jets in the independent order
        $$(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X).$$
        The baseline values of $h,o,c,\alpha,\sigma_0,X$ must be positive.
        Do not re-impose the baseline shape rule under perturbation.
    psi : float, default 0.002
        Finite positive scalar step $\psi$, held constant under perturbation.
        Require $0\le\tau_{00}$ and $\tau_{00}+\psi\le1$.
    residual_tol : float, default 1e-13
        Finite relative scalar-root tolerance in $(0,1)$, forwarded unchanged.
        It uses the unsquared transformed residual defined by
        implicit_z_taylor_jet and positive_z_from_x.
    max_iter : int, default 24
        Positive integer scalar-root iteration budget, forwarded unchanged.
        All fourth-order arithmetic must remain representable in binary64.

    Returns
    -------
    tuple[list[list[float]], tuple[tuple[float, ...], ...]]
        First: the finite normalized $3\times3$ jet of $R$ as nested lists.
        Second: exactly four baseline diagnostic tuples, in stage order.
        Each tuple is $(\tau_{\mathrm{stage}},W_{\mathrm{center}},
        \Delta,\Gamma,x,Z,F)$, all evaluated at $s=t=0$.

    Raises
    ------
    ValueError
        If there are not twelve correctly shaped finite jets, root controls
        or step are invalid, a required baseline is nonpositive, baseline
        stage times leave $[0,1]$, or any baseline stage has $\Gamma\le0$.
    ArithmeticError
        If an implicit solve fails or the annualized coefficients are
        nonfinite. Native arithmetic exceptions may propagate outside the
        representable domain. Malformed objects outside the typed API may
        raise TypeError.
    """
    annualized_coeff = [[0.0] * 3 for _ in range(3)]
    stage_rows = tuple()
    return annualized_coeff, stage_rows

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rk4_taylor_step(p_coeff,psi=.002,residual_tol=1e-13,max_iter=24):
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
    if len(p_coeff)!=12: raise ValueError("12 parameter Taylor arrays required")
    p=[_read(x) for x in p_coeff]
    if not math.isfinite(psi) or psi<=0: raise ValueError("positive finite time step required")
    if not math.isfinite(residual_tol) or not 0<residual_tol<1 or not isinstance(max_iter,int) or max_iter<1:
        raise ValueError("invalid root controls")
    wm,w0,wp,h,o,c,r,q,tau,alpha,sigma,X=p
    if min(h[0],o[0],c[0],alpha[0],sigma[0],X[0])<=0:
        raise ValueError("positive baseline geometry and volatility parameters required")
    if tau[0]<0 or tau[0]+psi>1: raise ValueError("all baseline RK4 stage times must lie in [0,1]")
    def _exp(a):
        base=math.exp(a[0])
        return _compose(a,[base,base,base/2,base/6,base/24])
    stages=[]
    def _rhs(y,shift):
        time=_add(tau,_const(shift))
        delta,gamma=_oracle_spatial_taylor_jets([_rows(wm),_rows(y),_rows(wp)],_rows(h),_rows(o),_rows(c))
        delta=_read(delta);gamma=_read(gamma)
        first,second=_oracle_rbf_fd_weights(h[0],o[0],c[0])
        delta[0],gamma[0]=_oracle_spatial_derivatives((wm[0],y[0],wp[0]),first,second)
        if gamma[0]<=0: raise ValueError("each baseline stage requires positive Gamma")
        x=_mul(_mul(_mul(_exp(_mul(r,time)),_mul(alpha,alpha)),_mul(X,X)),gamma)
        z=_read(_oracle_implicit_z_taylor_jet(_rows(x),residual_tol,max_iter))
        diffusion=_scale(_mul(_mul(_mul(_mul(sigma,sigma),_add(_const(1),z)),_mul(X,X)),gamma),.5)
        drift=_mul(_mul(_sub(r,q),X),delta)
        rate=_sub(_add(diffusion,drift),_mul(r,y))
        rate[0]=_oracle_nonlinear_bs_rhs(delta[0],gamma[0],y[0],X[0],r[0],q[0],sigma[0],z[0])
        stages.append((time[0],y[0],delta[0],gamma[0],x[0],z[0],rate[0]))
        return rate
    k1=_rhs(w0,0)
    k2=_rhs(_add(w0,_scale(k1,psi/2)),psi/2)
    k3=_rhs(_add(w0,_scale(k2,psi/2)),psi/2)
    k4=_rhs(_add(w0,_scale(k3,psi)),psi)
    annualized=_scale(_add(_add(k1,_scale(k2,2)),_add(_scale(k3,2),k4)),1/6)
    if not all(math.isfinite(x) for x in annualized): raise ArithmeticError("nonfinite annualized Taylor coefficients")
    return _rows(annualized),tuple(stages)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    helper="P=[8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100]\nU=[0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4]\nV=[-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3]\np=[[[x,t,0.0],[s,0.0,0.0],[0.0,0.0,0.0]] for x,s,t in zip(P,U,V)]\n"
    return [
        {"setup":helper,"call":"rk4_taylor_step(p)","gold_call":"_oracle_rk4_taylor_step(p)","tol":2e-10},
        {"setup":helper+"p=[[[x,0.,0.],[0.,0.,0.],[0.,0.,0.]] for x in P]","call":"rk4_taylor_step(p)","gold_call":"_oracle_rk4_taylor_step(p)","tol":2e-10},
        {"setup":helper+"p[4][0][0]=1.05;p[9][0][0]=.012;p[1][2][1]=.03","call":"rk4_taylor_step(p,.001)","gold_call":"_oracle_rk4_taylor_step(p,.001)","tol":2e-10},
    ]
