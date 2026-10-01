"""
Return the fourth mixed sensitivity of the local annualized RK4 step.



Coordinate order is $(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X)$.

Form independent affine paths $p(s,t)=p_0+s u+t v$, where $s$ and $t$

correspond to the public problem's $\varepsilon$ and $\eta$.

Seed each coordinate with normalized Taylor coefficients

$$J_{00}(p_k)=p_{0,k},\qquad J_{10}(p_k)=u_k,\qquad

J_{01}(p_k)=v_k,$$

with every other coefficient zero. Rows index the $s$ degree; columns

index the $t$ degree. Call rk4_taylor_step on the twelve jets and the

supplied scalar step. Its first output is $J(R)$ for the annualized

frozen-neighbor RK4 increment. Return

$$\left.\frac{\partial^4 R}{\partial s^2\partial t^2}\right|_{(0,0)}

=2!2!J_{22}(R)=4J_{22}(R).$$

Defaults apply independently to each omitted vector. Geometry coordinates

remain independent after perturbation. Do not fit sampled function values

or round the returned derivative to the public answer's display precision.

The normalized coefficient convention divides derivatives by factorials. The final mixed derivative is therefore $2!2!J_{22}(R)=4J_{22}(R)$. Swapping the two direction vectors transposes the coefficient matrix and leaves the target unchanged. The default authored baseline has $\tau_0=0.75$ and uses the source time step $\psi=0.002$.

Returns
-------
return fourth_interaction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(
    base: tuple[float, ...] | None = None,
    direction_s: tuple[float, ...] | None = None,
    direction_t: tuple[float, ...] | None = None,
    psi: float = 0.002,
) -> float:
    r"""Return the fourth mixed sensitivity of the local annualized RK4 step.

    Coordinate order is $(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X)$.
    Form independent affine paths $p(s,t)=p_0+s u+t v$, where $s$ and $t$
    correspond to the public problem's $\varepsilon$ and $\eta$.
    Seed each coordinate with normalized Taylor coefficients
    $$J_{00}(p_k)=p_{0,k},\qquad J_{10}(p_k)=u_k,\qquad
    J_{01}(p_k)=v_k,$$
    with every other coefficient zero. Rows index the $s$ degree; columns
    index the $t$ degree. Call rk4_taylor_step on the twelve jets and the
    supplied scalar step. Its first output is $J(R)$ for the annualized
    frozen-neighbor RK4 increment. Return
    $$\left.\frac{\partial^4 R}{\partial s^2\partial t^2}\right|_{(0,0)}
    =2!2!J_{22}(R)=4J_{22}(R).$$
    Defaults apply independently to each omitted vector. Geometry coordinates
    remain independent after perturbation. Do not fit sampled function values
    or round the returned derivative to the public answer's display precision.

    Parameters
    ----------
    base : tuple[float, ...] or None
        Twelve finite baseline coordinates. None selects
        $$(8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100).$$
        The baseline values of $h,o,c,\alpha,\sigma_0,X$ must be positive,
        and every baseline stage must have positive Gamma.
    direction_s : tuple[float, ...] or None
        Twelve finite components of $u$. None selects
        $$(0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,
        0.0001,0.005,0.4).$$
    direction_t : tuple[float, ...] or None
        Twelve finite components of $v$. None selects
        $$(-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,
        -0.0002,0.008,-0.3).$$
    psi : float, default 0.002
        Finite positive constant RK step $\psi$. Require
        $0\le\tau_0$ and $\tau_0+\psi\le1$.
        Root controls use rk4_taylor_step defaults. Inputs must keep
        fourth-order calculations and the final derivative finite in binary64.

    Returns
    -------
    float
        The unrounded mixed derivative $4J_{22}(R)$, not the raw normalized
        coefficient, next option value, or a formatted string.

    Raises
    ------
    ValueError
        If any supplied vector has length other than twelve or contains a
        nonfinite entry. Validation errors from rk4_taylor_step, including
        invalid step, baseline domain, stage times, or stage Gamma, propagate.
    ArithmeticError
        If a downstream implicit solve or Taylor calculation fails; native
        arithmetic exceptions may propagate outside the representable domain.
        Malformed objects outside the typed interface may raise TypeError.
    """
    fourth_interaction = 0.0
    return fourth_interaction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve(base=None,direction_s=None,direction_t=None,psi=.002):
    import math
    if base is None: base=[8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100]
    if direction_s is None: direction_s=[0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4]
    if direction_t is None: direction_t=[-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3]
    if any(len(a)!=12 for a in (base,direction_s,direction_t)):
        raise ValueError("base and two directions must have length 12")
    if not all(math.isfinite(float(x)) for a in (base,direction_s,direction_t) for x in a):
        raise ValueError("finite inputs required")
    p=[]
    for x,ds,dt in zip(base,direction_s,direction_t):
        p.append([[float(x),float(dt),0.0],[float(ds),0.0,0.0],[0.0,0.0,0.0]])
    annualized,_=_oracle_rk4_taylor_step(p,psi)
    return float(4*annualized[2][2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    helper="P=[8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100]\nU=[0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4]\nV=[-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3]\np=[[[x,t,0.0],[s,0.0,0.0],[0.0,0.0,0.0]] for x,s,t in zip(P,U,V)]\n"
    return [
        {"setup":"","call":"solve()","gold_call":"_oracle_solve()","tol":2e-9},
        {"setup":helper,"call":"solve(direction_s=V,direction_t=U)","gold_call":"_oracle_solve()","tol":2e-9},
        {"setup":helper,"call":"solve(direction_s=[0.]*12)","gold_call":"_oracle_solve(direction_s=[0.]*12)","tol":2e-12},
        {"setup":helper,"call":"solve(direction_s=[2*x for x in U],direction_t=[-.5*x for x in V])","gold_call":"_oracle_solve()","tol":2e-9},
        {"setup":helper+"P[4]=1.05;P[9]=.012","call":"solve(base=P,psi=.001)","gold_call":"_oracle_solve(base=P,psi=.001)","tol":2e-9},
    ]
