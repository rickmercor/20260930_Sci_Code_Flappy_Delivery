"""
Build the quadrature nodes and density-relative weights (previous step), evaluate the representing density at those nodes and form the exponential-sum weights omega = q * w(x). Evaluate the regularized kernel on a uniform grid of n_grid points spanning zero to dt inclusive and integrate its square over the step by the composite Simpson rule. Compute the one-step variance, over a step of length dt, of the omega-weighted aggregate of the Ornstein-Uhlenbeck factor increments that the lift uses in place of the Volterra noise, per unit instantaneous variance of the driving noise. Return the ratio of that one-step variance to the integrated square, minus one, all multiplied by ten thousand. Call the earlier step functions rather than reimplementing them. Raise ValueError if dt is not positive, or if n_grid is not an odd integer of at least three.

A faithful finite-dimensional realization reproduces the one-step variance of the Volterra noise it replaces, and how the factor increments over one step covary is fixed by how the source drives the factors. The diagnostic sits within a few basis points of zero when the construction is the prescribed one and hundreds of basis points away when any convention is wrong, so it serves as the admissibility check for the lift.

Returns
-------
float, the lift-fidelity error in basis points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lift_fidelity_error_bp(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int) -> float:
    """Compute the one-step lift-fidelity error of the exponential-sum lift, in basis points.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.
    dt : float
        Length of the time step, positive.
    n_grid : int
        Odd number, at least three, of uniform grid points spanning zero to dt inclusive for
        the composite Simpson rule.

    Returns
    -------
    result : float
        float, the lift-fidelity error in basis points.

    Raises
    ------
    ValueError
        If dt is not positive.
        If n_grid is not an odd integer of at least three.
        If alpha, delta_star, m or n_quad is invalid, as in the earlier steps.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_lift_fidelity_error_bp(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int) -> float:
    """Lift-fidelity error in basis points: 1e4 * (omega^T G omega / int_0^dt K^2 - 1).

    Calls step 3 (nodes and density-relative weights), step 2 (density at the nodes,
    omega = q * w(x)) and step 1 (kernel on the time grid). The one-step Volterra
    variance is the composite Simpson integral of the squared kernel on the uniform
    grid; the factor increments over the step share one Brownian path, so their
    covariance is G_lm = (1 - exp(-(x_l + x_m) dt)) / (x_l + x_m), and the lift's
    one-step variance is the quadratic form omega^T G omega.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if int(n_grid) != n_grid or n_grid < 3 or int(n_grid) % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 3")
    n_grid = int(n_grid)

    x, q = _oracle_block_quadrature_nodes(alpha, m, n_quad)
    omega = q * _oracle_bernstein_density(alpha, delta_star, x)

    t = np.linspace(0.0, float(dt), n_grid)
    kernel = _oracle_regularized_kernel(alpha, delta_star, t)
    f = kernel * kernel
    h = float(dt) / (n_grid - 1)
    exact = h / 3.0 * (f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-1:2]))

    s = x[:, None] + x[None, :]
    cov = (1.0 - np.exp(-s * dt)) / s
    lift = float(omega @ cov @ omega)

    return float(1.0e4 * (lift / exact - 1.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "alpha, delta_star, m, n_quad, dt, n_grid = 0.35, 0.05, 3, 24, 0.25, 200001",
            "call": "lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
            "gold_call": "_oracle_lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
        },
        {
            "setup": "alpha, delta_star, m, n_quad, dt, n_grid = 0.35, 0.05, 3, 24, 0.5, 200001",
            "call": "lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
            "gold_call": "_oracle_lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
        },
        {
            "setup": "alpha, delta_star, m, n_quad, dt, n_grid = 0.10, 0.20, 5, 12, 1.0, 200001",
            "call": "lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
            "gold_call": "_oracle_lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
        },
        {
            "setup": "alpha, delta_star, m, n_quad, dt, n_grid = 0.35, 0.05, 3, 24, 0.25, 3",
            "call": "lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
            "gold_call": "_oracle_lift_fidelity_error_bp(alpha, delta_star, m, n_quad, dt, n_grid)",
        },
        {
            "setup": "def run_model():\n    try:\n        lift_fidelity_error_bp(0.35, 0.05, 3, 24, 0.25, 200000)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_lift_fidelity_error_bp(0.35, 0.05, 3, 24, 0.25, 200000)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
