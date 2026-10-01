#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def clustered_grid(left: float, right: float, count: int, alpha: float = 0.4) -> np.ndarray:
    r"""Return the prescribed sinh-clustered log-price grid.

    With $z_i=-1+2i/(M-1)$, midpoint $c=(a+b)/2$ and half-width
    $h=(b-a)/2$, return $x_i=c+h\sinh(\alpha z_i)/\sinh(\alpha)$.
    At $\alpha=0$, use the continuous limit $x_i=c+hz_i$. Set the
    first and last entries to $a$ and $b$ exactly. The count $M$ denotes
    nodes, not intervals; this task fixes the indexing convention in Section 6.

    Parameters
    ----------
    left : float
        Finite left endpoint $a$.
    right : float
        Finite right endpoint $b>a$.
    count : int
        Number of nodes $M\geq2$.
    alpha : float, default 0.4
        Finite clustering strength $\alpha\geq0$; zero gives uniform spacing.

    Returns
    -------
    numpy.ndarray, shape $(\mathrm{count},)$
        Increasing binary64 log-price coordinates including both endpoints.

    Raises
    ------
    ValueError
        If the numeric arguments violate the stated ordering, finiteness,
        integer-count or nonnegative-clustering conditions.

    Inputs must keep the displayed expressions finite and the resulting
    nodes distinct in binary64 arithmetic; behavior beyond this domain is
    otherwise unspecified.
    """
    left, right, alpha = float(left), float(right), float(alpha)
    if (not np.isfinite([left, right, alpha]).all() or left >= right or alpha < 0
            or isinstance(count, (bool, np.bool_)) or not np.isscalar(count)
            or not np.isfinite(count) or int(count) != count or count < 2):
        raise ValueError("finite ordered endpoints, nonnegative alpha, and integer count>=2 required")
    count = int(count)
    z = np.linspace(-1.0, 1.0, count)
    mapped = z if alpha == 0.0 else np.sinh(alpha*z)/np.sinh(alpha)
    x = (left+right)/2.0 + (right-left)/2.0*mapped
    x[0], x[-1] = left, right
    return x

import numpy as np

def floater_hormann_weights(nodes: np.ndarray, degree: int) -> np.ndarray:
    r"""Return normalized Floater--Hormann rational interpolation weights.

    For $M$ increasing nodes and degree $d$, define
    $$\widehat w_i=(-1)^i\sum_{k=\max(0,i-d)}^{\min(i,M-d-1)}
    \prod_{\substack{j=k\\j\ne i}}^{k+d}|x_i-x_j|^{-1},\qquad
    w_i=\widehat w_i/\max_j|\widehat w_j|.$$
    An empty product is one. The first weight is positive and subsequent
    signs alternate. This fixes the immaterial common sign in Section 3.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    degree : int
        Local interpolation degree $0\leq d<M$.

    Returns
    -------
    numpy.ndarray, shape $(M,)$
        Weights in node order, with maximum absolute value one.

    Raises
    ------
    ValueError
        If the node vector or numeric degree violates its stated domain.

    Products and sums must remain finite and nonzero in binary64 arithmetic;
    behavior outside this representable domain is otherwise unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    if (isinstance(degree, (bool, np.bool_)) or not np.isscalar(degree)
            or not np.isfinite(degree) or int(degree) != degree
            or degree < 0 or degree >= x.size):
        raise ValueError("degree must be less than the node count")
    d = int(degree)
    w = np.empty(x.size)
    for i in range(x.size):
        total = 0.0
        for k in range(max(0, i-d), min(i, x.size-d-1)+1):
            others = [j for j in range(k, k+d+1) if j != i]
            total += 1.0/np.prod(np.abs(x[i]-x[others]))
        w[i] = (-1.0 if i % 2 else 1.0)*total
    return w/np.max(np.abs(w))

import numpy as np

def rational_evaluation(nodes: np.ndarray, weights: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate rational cardinal bases and their first two derivatives.

    The rational cardinal functions are
    $$\ell_j(p)=\frac{w_j/(p-x_j)}{\sum_k w_k/(p-x_k)}.$$
    Return $E_{ij}=\ell_j(p_i)$, $(D_1)_{ij}=\ell'_j(p_i)$ and
    $(D_2)_{ij}=\ell''_j(p_i)$, with derivatives taken with respect to $p$.
    At an exact node, use the analytic removable limits; the corresponding
    row of $E$ is a unit vector. Evaluate derivatives accurately near nodes.
    The second derivative belongs to this rational interpolant: do not use
    finite differences or square a nodal first-derivative matrix.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero barycentric weights. Common nonzero rescaling has no
        effect on the result.
    points : float or array_like, shape $(Q,)$
        Finite evaluation coordinates; a scalar produces one output row.
        An empty vector produces three arrays with shape $(0,M)$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Exactly $(E,D_1,D_2)$, each binary64 with shape $(Q,M)$; rows are
        evaluation points and columns are interpolation nodes.

    Raises
    ------
    ValueError
        If the shapes, finiteness, node ordering or nonzero weights are invalid.
    ArithmeticError
        If the evaluated rational denominator is zero or nonfinite.

    Inputs must otherwise allow finite binary64 interpolation and derivative
    values. The analytic denominator is assumed nonzero away from nodes.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    w = np.asarray(weights, dtype=float)
    p = np.atleast_1d(np.asarray(points, dtype=float))
    if w.shape != x.shape or not np.isfinite(w).all() or np.any(w == 0):
        raise ValueError("weights must be a finite nonzero vector matching nodes")
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("points must be a finite scalar or one-dimensional array")
    w = w/np.max(np.abs(w))
    E = np.empty((p.size, x.size))
    D1, D2 = np.empty_like(E), np.empty_like(E)
    for row, point in enumerate(p):
        k = int(np.argmin(np.abs(point-x)))
        other = np.arange(x.size) != k
        diff = point-x[other]
        # Factoring point-x[k] removes the nearest pole before differentiation.
        # This formula is defined at an exact node and stable very close to it.
        b, db, ddb = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
        b[k] = w[k]
        b[other] = w[other]*(point-x[k])/diff
        db[other] = w[other]*(x[k]-x[other])/diff**2
        ddb[other] = -2.0*w[other]*(x[k]-x[other])/diff**3
        total, first, second = np.sum(b), np.sum(db), np.sum(ddb)
        if total == 0.0 or not np.isfinite(total):
            raise ArithmeticError("the rational denominator vanishes or is nonfinite")
        E[row] = b/total
        D1[row] = (db-E[row]*first)/total
        D2[row] = (ddb-E[row]*second-2.0*D1[row]*first)/total
    return E, D1, D2

import numpy as np
from scipy.special import roots_legendre

def merton_operators(nodes: np.ndarray, degree: int, quadrature_order: int,
                     r: float, sigma: float, intensity: float, jump_mean: float,
                     jump_std: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the Merton differential and interior jump-integral matrices.

    Let $\lambda$ be the intensity, $\mu_J,s_J$ the log-jump mean and
    standard deviation, and $\zeta=\exp(\mu_J+s_J^2/2)-1$. With nodal
    rational derivative matrices from the previous steps, form
    $$D=\tfrac12\sigma^2D_2+
    (r-\tfrac12\sigma^2-\lambda\zeta)D_1-(r+\lambda)I.$$
    If $(y_q,\omega_q)$ is the $Q$-point Gauss--Legendre rule mapped to
    $[x_0,x_{M-1}]$, including the mapping factor in $\omega_q$, set
    $$J_{ij}=\sum_{q=1}^Q\omega_q\ell_j(y_q)
    \frac{\exp[-(y_q-x_i-\mu_J)^2/(2s_J^2)]}{\sqrt{2\pi}s_J}.$$
    Thus $J$ excludes intensity and exterior tails. Preserve all matrix rows;
    the time integrator imposes the boundary conditions. Call
    floater_hormann_weights and rational_evaluation to construct these objects.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Number $Q\geq1$ of Gauss--Legendre nodes over the entire interval.
    r : float
        Finite risk-free rate.
    sigma : float
        Finite strictly positive diffusion volatility.
    intensity : float
        Finite nonnegative Poisson jump intensity $\lambda$.
    jump_mean : float
        Finite mean $\mu_J$ of the normally distributed log jump.
    jump_std : float
        Finite strictly positive log-jump standard deviation $s_J$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Exactly $(D,J)$, two dense binary64 arrays of shape $(M,M)$ with
        evaluation-node rows and value-node columns.

    Raises
    ------
    ValueError
        If numeric counts, model parameters or the node vector are invalid.
    ArithmeticError
        If rational_evaluation encounters an invalid rational denominator.

    Parameters and nodes must keep all intermediate formulas finite in
    binary64; behavior outside this representable domain is unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    if (isinstance(quadrature_order, (bool, np.bool_)) or not np.isscalar(quadrature_order)
            or not np.isfinite(quadrature_order) or int(quadrature_order) != quadrature_order
            or quadrature_order < 1):
        raise ValueError("positive integer quadrature_order required")
    q = int(quadrature_order)
    r, sigma, intensity, jump_mean, jump_std = map(float, (r, sigma, intensity, jump_mean, jump_std))
    if (not np.isfinite([r, sigma, intensity, jump_mean, jump_std]).all()
            or sigma <= 0 or intensity < 0 or jump_std <= 0):
        raise ValueError("invalid Merton parameters")
    weights = floater_hormann_weights(x, degree)
    _, D1, D2 = rational_evaluation(x, weights, x)
    compensator = np.expm1(jump_mean+0.5*jump_std**2)
    D = (0.5*sigma**2*D2
         + (r-0.5*sigma**2-intensity*compensator)*D1
         - (r+intensity)*np.eye(x.size))
    abscissae, quadrature_weights = roots_legendre(q)
    half = (x[-1]-x[0])/2.0
    y = (x[-1]+x[0])/2.0 + half*abscissae
    E, _, _ = rational_evaluation(x, weights, y)
    z = (y[None, :]-x[:, None]-jump_mean)/jump_std
    density = np.exp(-0.5*z*z)/(np.sqrt(2.0*np.pi)*jump_std)
    J = (density*(half*quadrature_weights)[None, :])@E
    return D, J

import numpy as np
from scipy.special import ndtr

def american_boundary_tail(nodes: "np.ndarray", strike: float,
                           jump_mean: float, jump_std: float) -> tuple:
    r"""Return the time-independent American Merton boundary and exterior tail.

    Write $a=x_0$ and $z_i=(a-x_i-\mu_J)/s_J$. The task adopts the
    finite-boundary convention in Section 2, not the limiting strike value
    printed in Equation (5.2): $b=(K-Ke^a,0)$. Section 4 gives
    $$R_i=K\Phi(z_i)-K\exp(x_i+\mu_J+s_J^2/2)\Phi(z_i-s_J).$$
    Here $\Phi$ is the standard normal CDF. Neither discounting nor jump
    intensity is included. The exterior payoff is used for all elapsed times.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$, with
        $x_0<0<x_{M-1}$.
    strike : float
        Finite strike $K>0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite normal log-jump standard deviation $s_J>0$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Boundary values of shape $(2,)$, left then right, and exterior
        integral of shape $(M,)$, in node order, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    All displayed expressions must remain finite in binary64; behavior
    outside that representable domain is unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if (x.ndim != 1 or x.size < 2 or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0) or not x[0] < 0 < x[-1]):
        raise ValueError("ordered finite nodes straddling zero required")
    strike, jump_mean, jump_std = map(float, (strike, jump_mean, jump_std))
    if not np.isfinite([strike, jump_mean, jump_std]).all() or strike <= 0 or jump_std <= 0:
        raise ValueError("finite positive strike and jump_std required")
    z = (x[0]-x-jump_mean)/jump_std
    boundary = np.array([strike*(1-np.exp(x[0])), 0.0])
    tail = strike*ndtr(z)-strike*np.exp(x+jump_mean+0.5*jump_std**2)*ndtr(z-jump_std)
    return boundary, tail

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def imex_american(nodes: "np.ndarray", D: "np.ndarray", J: "np.ndarray",
                  strike: float, intensity: float, jump_mean: float,
                  jump_std: float, maturity: float, steps: int) -> tuple:
    r"""Advance price and multiplier with exactly one IT correction per layer.

    Set $P_i=\max(K-Ke^{x_i},0)$, $U^0=P$, $\phi^0=0$ and
    $\Delta t=T/N$. Obtain $b,R$ from american_boundary_tail.
    The startup intermediate interior equations are
    $$(I-\Delta t D)\widetilde U^1=U^0+
      \Delta t\{\lambda(JU^0+R)+\phi^0\}.$$
    At later layers, for $n=1,\ldots,N-1$, use
    $$(3I-2\Delta t D)\widetilde U^{n+1}=4U^n-U^{n-1}
      +2\Delta t\{\lambda[J(2U^n-U^{n-1})+R]+\phi^n\}.$$
    In each linear system, impose the two endpoint values $b$ and retain
    their column contributions in the interior equations. Use the corrected
    histories in the extrapolation, not intermediate histories.
    For each interior component perform the Section 5 IT correction
    $$U^{n+1}=\max(P,\widetilde U^{n+1}-h\phi^n),\qquad
      \phi^{n+1}=\max(0,\phi^n+(P-\widetilde U^{n+1})/h),$$
    where $h=\Delta t$ at startup and $h=2\Delta t/3$ thereafter.
    Maxima are componentwise. Keep corrected endpoint prices equal to $b$
    and endpoint multipliers exactly zero. There are no inner iterations,
    multiplier extrapolation, damping steps or post-hoc clipping elsewhere.
    The multiplier formula is algebraically equivalent to Equation (5.4)
    and avoids cancellation on continuation nodes.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, $x_0<0<x_{M-1}$.
    D : array_like, shape $(M,M)$
        Finite local differential matrix, including compensated drift and loss.
    J : array_like, shape $(M,M)$
        Finite interior jump matrix without intensity or exterior tail.
    strike : float
        Finite positive strike $K$.
    intensity : float
        Finite jump intensity $\lambda\geq0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite positive normal log-jump standard deviation $s_J$.
    maturity : float
        Finite positive final elapsed time $T$.
    steps : int
        Number of time intervals $N\geq1$. A single interval uses only startup.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price history then multiplier history, each of shape $(N+1,M)$,
        with time rows and node columns. Row zero is exactly payoff and zero,
        respectively. Outputs are unrounded binary64 arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, ordering, signs or integer count are invalid.

    The boundary-modified systems must be nonsingular and all intermediate
    expressions finite in binary64. Behavior outside that domain is unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    boundary, tail = american_boundary_tail(x, strike, jump_mean, jump_std)
    D, J = np.asarray(D, dtype=float), np.asarray(J, dtype=float)
    if D.shape != (x.size, x.size) or J.shape != D.shape or not np.isfinite(D).all() or not np.isfinite(J).all():
        raise ValueError("finite matching square matrices required")
    maturity, intensity = float(maturity), float(intensity)
    if (not np.isfinite([maturity, intensity]).all() or maturity <= 0 or intensity < 0
            or isinstance(steps, (bool, np.bool_)) or not np.isscalar(steps)
            or not np.isfinite(steps) or int(steps) != steps or steps < 1):
        raise ValueError("positive maturity, nonnegative intensity and positive integer steps required")
    N = int(steps)
    dt = maturity/N
    payoff = np.maximum(float(strike)-float(strike)*np.exp(x), 0.0)
    U, phi = np.empty((N+1, x.size)), np.zeros((N+1, x.size))
    U[0] = payoff
    identity = np.eye(x.size)
    A1, A2 = identity-dt*D, 3*identity-2*dt*D
    for A in (A1, A2):
        A[0], A[-1] = identity[0], identity[-1]
    factors1 = lu_factor(A1)
    factors2 = lu_factor(A2) if N > 1 else None
    for n in range(N):
        if n == 0:
            h = dt
            rhs = U[0]+dt*(intensity*(J@U[0]+tail)+phi[0])
            factors = factors1
        else:
            h = 2*dt/3
            rhs = 4*U[n]-U[n-1]+2*dt*(intensity*(J@(2*U[n]-U[n-1])+tail)+phi[n])
            factors = factors2
        rhs[0], rhs[-1] = boundary
        intermediate = lu_solve(factors, rhs)
        U[n+1] = np.maximum(payoff, intermediate-h*phi[n])
        phi[n+1] = np.maximum(0.0, phi[n]+(payoff-intermediate)/h)
        U[n+1, 0], U[n+1, -1] = boundary
        phi[n+1, 0] = phi[n+1, -1] = 0.0
    return U, phi

import numpy as np

def put_greeks(nodes: np.ndarray, weights: np.ndarray, values: np.ndarray,
               spot: float, strike: float) -> tuple[float, float, float]:
    r"""Evaluate put price, delta and gamma from the same rational interpolant.

    At $x_*=\log(S/K)$, obtain $(E,D_1,D_2)$ by calling
    rational_evaluation with the supplied nodes and weights. With nodal
    values $U$, compute $u=EU$, $u_x=D_1U$, and $u_{xx}=D_2U$.
    Return
    $$(V,\Delta,\Gamma)=\left(u,\frac{u_x}{S},
       \frac{u_{xx}-u_x}{S^2}\right).$$
    The derivatives are with respect to the underlying price $S$, holding
    the strike fixed. Use analytic rational derivatives, including exact-node
    limits, without finite differencing, interpolation substitution or rounding.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero rational interpolation weights matching the nodes.
    values : array_like, shape $(M,)$
        Finite nodal option values on this same grid.
    spot : float
        Finite strictly positive underlying price $S$ such that
        $\log(S/K)\in[x_0,x_{M-1}]$.
    strike : float
        Finite strictly positive strike $K$ defining the log-price coordinate.

    Returns
    -------
    tuple[float, float, float]
        Exactly the price, delta and gamma in that order, as unrounded scalars.

    Raises
    ------
    ValueError
        If the arrays or prices violate their stated shape, finiteness,
        ordering, sign or evaluation-interval conditions.
    ArithmeticError
        If rational_evaluation encounters an invalid denominator.

    Inputs must allow finite binary64 outputs; behavior outside the stated
    representable domain is otherwise unspecified.
    """
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("finite strictly increasing nodes required")
    spot, strike = float(spot), float(strike)
    if not np.isfinite([spot, strike]).all() or spot <= 0 or strike <= 0:
        raise ValueError("positive finite spot and strike required")
    values = np.asarray(values, dtype=float)
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector matching nodes")
    point = np.log(spot/strike)
    if point < x[0] or point > x[-1]:
        raise ValueError("log(spot/strike) must lie in the grid interval")
    E, D1, D2 = rational_evaluation(x, weights, [point])
    price, ux, uxx = float(E[0]@values), float(D1[0]@values), float(D2[0]@values)
    return price, ux/spot, (uxx-ux)/spot**2

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def merton_mean_jets(nodes: "np.ndarray", degree: int, quadrature_order: int,
                     r: float, sigma: float, intensity: float,
                     jump_mean: float, jump_std: float, strike: float) -> tuple:
    r"""Return the first three jump-mean jets of the finite Merton operators.

    For $q=0,1,2$, the entries in slot $q$ are the ordinary derivatives
    $\partial_{\mu_J}^q D$, $\partial_{\mu_J}^q J$, and
    $\partial_{\mu_J}^q R$. These are derivatives, not Taylor coefficients.
    $D,J$ are exactly the objects defined by merton_operators and $R$ is
    the intensity-free American tail defined by american_boundary_tail.
    Hold nodes, degree, quadrature nodes/weights, strike and every parameter
    except jump_mean fixed. Differentiate the compensated drift as well as
    the normal density and the exterior integral. Compute analytic jets;
    finite differences or complex perturbations of a pricing routine are
    not the requested operator. Slot zero must call the preceding routines.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Global Gauss--Legendre order $Q\geq1$.
    r, sigma, intensity, jump_mean, jump_std : float
        Same domains and meaning as merton_operators: finite rate and mean,
        positive volatility and jump standard deviation, nonnegative intensity.
    strike : float
        Finite positive strike, held fixed in the differentiation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        $(D_{\rm jet},J_{\rm jet},R_{\rm jet})$, shapes $(3,M,M)$,
        $(3,M,M)$ and $(3,M)$. The leading axis is derivative order;
        matrix rows are evaluation nodes and columns are value nodes.

    Raises
    ------
    ValueError
        For violations of the stated numeric domains and shapes.
    ArithmeticError
        For a nonfinite or zero rational denominator.

    The constituent formulas must have finite binary64 values.
    """
    x = np.asarray(nodes, dtype=float)
    D, J = merton_operators(x, degree, quadrature_order, r, sigma,
                            intensity, jump_mean, jump_std)
    _, R = american_boundary_tail(x, strike, jump_mean, jump_std)
    w = floater_hormann_weights(x, degree)
    _, Dx, _ = rational_evaluation(x, w, x)
    eta, omega = roots_legendre(int(quadrature_order))
    y = (x[-1]+x[0])/2 + (x[-1]-x[0])*eta/2
    W = (x[-1]-x[0])*omega/2
    E, _, _ = rational_evaluation(x, w, y)
    mu, s, lam, K = map(float, (jump_mean, jump_std, intensity, strike))
    g = y[None, :] - x[:, None] - mu
    density = np.exp(-0.5*(g/s)**2)/(np.sqrt(2*np.pi)*s)
    J1 = (density*(g/s**2)*W) @ E
    J2 = (density*(g*g/s**4-1/s**2)*W) @ E
    D1 = -lam*np.exp(mu+s*s/2)*Dx
    z = (x[0]-x-mu)/s
    a = z-s
    pdfz = np.exp(-z*z/2)/np.sqrt(2*np.pi)
    pdfa = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    expterm = np.exp(x+mu+s*s/2)
    R1 = K*(-pdfz/s-expterm*(ndtr(a)-pdfa/s))
    R2 = K*(-z*pdfz/s**2-expterm*(ndtr(a)-2*pdfa/s-a*pdfa/s**2))
    return np.stack((D, D1, D1)), np.stack((J, J1, J2)), np.stack((R, R1, R2))

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def differentiate_american(nodes: "np.ndarray", D_jets: "np.ndarray",
                           J_jets: "np.ndarray", R_jets: "np.ndarray",
                           prices: "np.ndarray", multipliers: "np.ndarray",
                           intensity: float, maturity: float) -> tuple:
    r"""Differentiate the complete IMEX/IT trajectory twice in jump mean.

    The supplied prices and multipliers are the base histories from
    imex_american, at the same nodes and model parameters used to form
    merton_mean_jets. Differentiate that finite algorithm, including its
    implicit systems, explicit jump extrapolation, carried multiplier,
    and the IT complementarity correction. The payoff, finite endpoint
    values, grid, quadrature and time intervals are independent of jump mean.
    Use analytic differentiation; do not estimate these derivatives from
    perturbed base solves, automatic differentiation packages, or complex steps.
    Ordinary derivatives and their product rules apply, not factorial-scaled
    Taylor coefficients. The base histories and operator jets are assumed
    mutually consistent; no reconstruction or reoptimization of them is needed.

    At each interior correction, a price strictly above its initial payoff
    identifies continuation. A price equal to payoff identifies exercise;
    for a degenerate exact tie with zero multiplier, use the exercise branch
    for derivative propagation. This makes the interface deterministic at
    ties. On a locally stable base branch pattern it equals differentiation
    of the full finite algorithm. The default task must be checked for this
    stability before interpreting the result as an ordinary derivative.
    Initial derivatives and all endpoint derivatives are exactly zero.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    D_jets, J_jets : array_like, shape $(3,M,M)$
        Finite matrices in derivative-order, evaluation-node, value-node order.
    R_jets : array_like, shape $(3,M)$
        Finite intensity-free exterior integral derivatives.
    prices, multipliers : array_like, shape $(N+1,M)$
        Consistent base histories, $N\geq1$. Row zero gives payoff and zero.
    intensity : float
        Finite nonnegative jump intensity, held fixed.
    maturity : float
        Finite positive horizon; $\Delta t=T/N$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price jets and multiplier jets, each shape $(3,N+1,M)$.
        Slot zero is a copy of the supplied history. Slots one and two
        are its first and second jump-mean derivatives, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    The boundary-modified linear systems must be nonsingular and all
    analytic derivatives must remain finite in binary64 arithmetic.
    """
    x = np.asarray(nodes, dtype=float)
    D, J, R, U, phi = map(lambda a: np.asarray(a, dtype=float),
                          (D_jets, J_jets, R_jets, prices, multipliers))
    M = x.size
    if (x.ndim != 1 or M < 2 or not np.isfinite(x).all()
            or np.any(np.diff(x) <= 0) or not x[0] < 0 < x[-1]):
        raise ValueError('increasing finite nodes straddling zero required')
    if (D.shape != (3,M,M) or J.shape != D.shape or R.shape != (3,M)
            or U.ndim != 2 or U.shape[1] != M or U.shape[0] < 2
            or phi.shape != U.shape or not all(np.isfinite(a).all() for a in (D,J,R,U,phi))):
        raise ValueError('finite jet tensors and matching histories required')
    lam, T = float(intensity), float(maturity)
    if not np.isfinite([lam,T]).all() or lam < 0 or T <= 0:
        raise ValueError('nonnegative intensity and positive maturity required')
    N = U.shape[0]-1
    dt = T/N
    V = np.zeros((3,N+1,M)); psi = np.zeros_like(V)
    V[0], psi[0] = U, phi
    I = np.eye(M)
    A1, A2 = I-dt*D[0], 3*I-2*dt*D[0]
    for A in (A1,A2):
        A[[0,-1]] = I[[0,-1]]
    factor1 = lu_factor(A1)
    factor2 = lu_factor(A2) if N > 1 else None
    for n in range(N):
        first = n == 0
        c, h = (dt,dt) if first else (2*dt,2*dt/3)
        ex = V[:,n] if first else 2*V[:,n]-V[:,n-1]
        pred = V[:,n] if first else 4*V[:,n]-V[:,n-1]
        base_tilde = U[n+1]-h*(phi[n+1]-phi[n])
        rhs1 = pred[1]+c*(lam*(J[1]@ex[0]+J[0]@ex[1]+R[1])+psi[1,n]+D[1]@base_tilde)
        rhs1[[0,-1]] = 0.0
        fac = factor1 if first else factor2
        tilde1 = lu_solve(fac,rhs1)
        rhs2 = pred[2]+c*(lam*(J[2]@ex[0]+2*J[1]@ex[1]+J[0]@ex[2]+R[2])+psi[2,n]
                           +D[2]@base_tilde+2*D[1]@tilde1)
        rhs2[[0,-1]] = 0.0
        tilde2 = lu_solve(fac,rhs2)
        free = U[n+1] > U[0]
        free[[0,-1]] = False
        for q, tilde in ((1,tilde1),(2,tilde2)):
            V[q,n+1] = np.where(free,tilde-h*psi[q,n],0.0)
            psi[q,n+1] = np.where(free,0.0,psi[q,n]-tilde/h)
            V[q,n+1,[0,-1]] = 0.0
            psi[q,n+1,[0,-1]] = 0.0
    return V, psi

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def solve(spot: float = 93.7, strike: float = 100.0, maturity: float = 0.25,
          r: float = 0.05, sigma: float = 0.15, intensity: float = 0.1,
          jump_mean: float = -0.9, jump_std: float = 0.45,
          node_count: int = 193, steps: int = 121, quadrature_order: int = 256,
          degree: int = 3, half_width: float = 1.5, alpha: float = 0.4) -> float:
    r"""Return the second jump-mean derivative of finite-grid American Gamma.

    Compose clustered_grid, floater_hormann_weights, merton_mean_jets,
    imex_american, differentiate_american and put_greeks. The operator-jet
    routine in turn composes merton_operators, american_boundary_tail and
    rational_evaluation. Differentiate the finite algorithm with all inputs
    except jump_mean fixed. The output is $\partial_{\mu_J}^2\Gamma_h(S)$;
    it is neither Gamma itself nor the factorial-scaled quadratic coefficient.
    At degenerate ties use the derivative convention of differentiate_american.

    Parameters
    ----------
    spot, strike, maturity : float
        Finite positive underlying, strike, and elapsed horizon.
    r, sigma, intensity, jump_mean, jump_std : float
        Merton model parameters: finite rate and mean, positive sigma and
        jump_std, nonnegative intensity.
    node_count, steps, quadrature_order, degree : int
        Respectively $M\geq2$, $N\geq1$, $Q\geq1$, and $0\leq d<M$.
    half_width : float
        Finite positive $X$ defining the symmetric domain $[-X,X]$.
        The query must satisfy $|\log(S/K)|\leq X$.
    alpha : float
        Finite nonnegative sinh-clustering parameter; zero means uniform.

    Returns
    -------
    float
        Unrounded second jump-mean derivative of rational Gamma. The
        default parameters are those printed in the function signature.

    Raises
    ------
    ValueError
        For violations of any constituent input domain.
    ArithmeticError
        For an invalid rational denominator.

    All constituent expressions must be finite in binary64 and linear
    systems nonsingular. Ordinary-derivative interpretation requires a
    locally stable branch pattern, as stated in the task.
    """
    X = float(half_width)
    if not np.isfinite(X) or X <= 0:
        raise ValueError('positive half_width required')
    nodes = clustered_grid(-X,X,node_count,alpha)
    weights = floater_hormann_weights(nodes,degree)
    D,J,R = merton_mean_jets(nodes,degree,quadrature_order,r,sigma,intensity,jump_mean,jump_std,strike)
    U,phi = imex_american(nodes,D[0],J[0],strike,intensity,jump_mean,jump_std,maturity,steps)
    V,psi = differentiate_american(nodes,D,J,R,U,phi,intensity,maturity)
    return float(put_greeks(nodes,weights,V[2,-1],spot,strike)[2])
SCICODE_GOLD_EOF
