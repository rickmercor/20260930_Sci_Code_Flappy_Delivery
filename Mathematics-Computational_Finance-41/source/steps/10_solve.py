"""
Compose the source-method primal solver and its analytic sensitivities to compute the second jump-mean derivative of the final rational Gamma.

The observable is a local curvature of the paper-based finite numerical method with respect to the normal log-jump mean. It requires both the original pricing computation and the differentiated implicit/explicit obstacle evolution. It is not a continuous-model sensitivity.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(spot: 'float'=93.7, strike: 'float'=100.0, maturity: 'float'=0.25, r: 'float'=0.05, sigma: 'float'=0.15, intensity: 'float'=0.1, jump_mean: 'float'=-0.9, jump_std: 'float'=0.45, node_count: 'int'=193, steps: 'int'=121, quadrature_order: 'int'=256, degree: 'int'=3, half_width: 'float'=1.5, alpha: 'float'=0.4) -> 'float':
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
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ndtr, roots_legendre
from scipy.linalg import lu_factor, lu_solve

def _oracle_solve(spot: float = 93.7, strike: float = 100.0, maturity: float = 0.25,
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
    nodes = _oracle_clustered_grid(-X,X,node_count,alpha)
    weights = _oracle_floater_hormann_weights(nodes,degree)
    D,J,R = _oracle_merton_mean_jets(nodes,degree,quadrature_order,r,sigma,intensity,jump_mean,jump_std,strike)
    U,phi = _oracle_imex_american(nodes,D[0],J[0],strike,intensity,jump_mean,jump_std,maturity,steps)
    V,psi = _oracle_differentiate_american(nodes,D,J,R,U,phi,intensity,maturity)
    return float(_oracle_put_greeks(nodes,weights,V[2,-1],spot,strike)[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'solve()', 'gold_call': '_oracle_solve()', 'tol': 2e-07}, {'setup': '', 'call': 'solve(spot=105.0,strike=110.0,maturity=0.4,r=0.025,sigma=0.22,intensity=0.3,jump_mean=-0.35,jump_std=0.3,node_count=65,steps=37,quadrature_order=120,degree=3,half_width=1.2,alpha=0.65)', 'gold_call': '_oracle_solve(spot=105.0,strike=110.0,maturity=0.4,r=0.025,sigma=0.22,intensity=0.3,jump_mean=-0.35,jump_std=0.3,node_count=65,steps=37,quadrature_order=120,degree=3,half_width=1.2,alpha=0.65)', 'tol': 2e-07}, {'setup': '', 'call': 'solve(spot=84.0,strike=80.0,maturity=0.18,r=0.06,sigma=0.19,intensity=0.0,jump_mean=0.12,jump_std=0.5,node_count=49,steps=25,quadrature_order=80,degree=1,half_width=1.7,alpha=0.0)', 'gold_call': '_oracle_solve(spot=84.0,strike=80.0,maturity=0.18,r=0.06,sigma=0.19,intensity=0.0,jump_mean=0.12,jump_std=0.5,node_count=49,steps=25,quadrature_order=80,degree=1,half_width=1.7,alpha=0.0)', 'tol': 2e-07}]
