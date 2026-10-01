"""
Build the compensated Merton differential matrix and the global Gauss--Legendre interior jump matrix from the preceding interpolation routines. Keep the jump intensity and exterior tail separate as specified.

In log-price coordinates, Merton prices satisfy a partial integro-differential equation. The diffusion, compensated drift, discount and jump-loss terms form the differential matrix. Section 4.2 combines BRI interpolation with one Gauss--Legendre rule over the bounded interval to form the interior integral matrix. The exterior contribution is computed separately, and intensity is applied by the time integrator. The exact equations are preserved in the raw function docstring.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def merton_operators(nodes: "np.ndarray", degree: int, quadrature_order: int,
                     r: float, sigma: float, intensity: float, jump_mean: float,
                     jump_std: float) -> tuple["np.ndarray", "np.ndarray"]:
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
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import roots_legendre

def _oracle_merton_operators(nodes: np.ndarray, degree: int, quadrature_order: int,
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
    _oracle_floater_hormann_weights and _oracle_rational_evaluation to construct these objects.

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
        If _oracle_rational_evaluation encounters an invalid rational denominator.

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
    weights = _oracle_floater_hormann_weights(x, degree)
    _, D1, D2 = _oracle_rational_evaluation(x, weights, x)
    compensator = np.expm1(jump_mean+0.5*jump_std**2)
    D = (0.5*sigma**2*D2
         + (r-0.5*sigma**2-intensity*compensator)*D1
         - (r+intensity)*np.eye(x.size))
    abscissae, quadrature_weights = roots_legendre(q)
    half = (x[-1]-x[0])/2.0
    y = (x[-1]+x[0])/2.0 + half*abscissae
    E, _, _ = _oracle_rational_evaluation(x, weights, y)
    z = (y[None, :]-x[:, None]-jump_mean)/jump_std
    density = np.exp(-0.5*z*z)/(np.sqrt(2.0*np.pi)*jump_std)
    J = (density*(half*quadrature_weights)[None, :])@E
    return D, J

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'nodes=[-1.5,-0.7,-0.1,0.4,1.2]', 'call': '[part.tolist() for part in merton_operators(nodes,3,23,0.04,0.21,0.35,-0.4,0.32)]', 'gold_call': '[part.tolist() for part in _oracle_merton_operators(nodes,3,23,0.04,0.21,0.35,-0.4,0.32)]', 'tol': 2e-10}, {'setup': 'nodes=[-1.1,-0.5,0.2,0.9]', 'call': '[part.tolist() for part in merton_operators(nodes,1,19,-0.015,0.27,0.0,0.2,0.35)]', 'gold_call': '[part.tolist() for part in _oracle_merton_operators(nodes,1,19,-0.015,0.27,0.0,0.2,0.35)]', 'tol': 2e-10}, {'setup': 'nodes=[-1.8,-0.8,-0.2,0.1,0.55,1.4]', 'call': '[part.tolist() for part in merton_operators(nodes,2,31,0.065,0.16,0.7,0.15,0.52)]', 'gold_call': '[part.tolist() for part in _oracle_merton_operators(nodes,2,31,0.065,0.16,0.7,0.15,0.52)]', 'tol': 2e-10}]
