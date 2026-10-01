"""
Computes the Griffith-consistency normalization constant of the bond-based phase-field damage model for a named normalized spherical kernel ('cubic', 'linear', or 'constant'), following the source's derivation, with any one-dimensional kernel integrals evaluated to better than 1e-12. Validates the normalization that every later damage quantity inherits (the critical threshold of step 6 and the dissipation functional of step 9). Deliberately excluded: any mesh, deformation, or dynamics - the constant is a pure kernel functional.

Phase-field fracture represents crack surface energy by a damage-dependent density whose overall scale must be normalized so that a fully formed crack dissipates Gc per unit area. In local models the constant follows from the optimal one-dimensional damage profile; in this bond-based nonlocal setting there is no such profile - the dissipation of a sharp crack is carried by the kernel-weighted population of bonds crossing the crack surface, so the constant is a functional of the kernel and must be derived within the nonlocal theory as the source does. Local phase-field constants do not apply here. The step must return the constant for the benchmark's cubic B-spline and for the normalized linear and constant kernels.

Returns
-------
float: the damage-model normalization constant c0 (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pfpd_normalization_constant(kernel: str) -> float:
    """Griffith-consistency normalization constant of the damage model.

    Evaluates, for a named normalized spherical kernel, the normalization
    constant of the bond-based phase-field damage model of the source: the
    constant that makes the total crack dissipation of a fully formed
    crack equal to the Griffith energy release rate per unit crack area
    in this nonlocal setting. The constant is a pure functional of the
    kernel shape; its construction must follow the source, and any
    one-dimensional kernel integrals over the unit interval must be
    evaluated exactly or with error below 1e-12. The returned value is
    consumed by step 6 (critical threshold), passed through steps 7 and
    8, and used by the orchestrator of step 9.

    Parameters
    ----------
    kernel : str
        One of 'cubic' (normalized cubic B-spline, shape
        f(q) = 1 - 6 q^2 + 6 q^3 for q <= 1/2 and 2 (1 - q)^3 for
        1/2 < q <= 1), 'linear' (normalized linear kernel, shape 1 - q on
        q <= 1), or 'constant' (normalized constant kernel, shape 1 on
        q <= 1), with q = |DX| / delta and zero beyond q = 1.

    Returns
    -------
    float
        The normalization constant c0 (dimensionless).

    Raises
    ------
    ValueError
        If the kernel name is not one of the three supported strings.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s02_kernel_shape(q, kernel):
    q = np.asarray(q, dtype=float)
    if kernel == 'cubic':
        f = np.where(q <= 0.5, 1.0 - 6.0*q**2 + 6.0*q**3, 2.0*(1.0 - q)**3)
        return np.where(q <= 1.0, np.maximum(f, 0.0), 0.0)
    if kernel == 'linear':
        return np.where(q <= 1.0, 1.0 - q, 0.0)
    if kernel == 'constant':
        return np.where(q <= 1.0, 1.0, 0.0)
    raise ValueError("kernel must be 'cubic', 'linear' or 'constant'")


def _oracle_pfpd_normalization_constant(kernel):
    if kernel not in ('cubic', 'linear', 'constant'):
        raise ValueError("kernel must be 'cubic', 'linear' or 'constant'")
    x, w = np.polynomial.legendre.leggauss(200)
    num = den = 0.0
    for a, b in ((0.0, 0.5), (0.5, 1.0)):
        q = 0.5*(b - a)*(x + 1.0) + a
        wq = 0.5*(b - a)*w
        f = _s02_kernel_shape(q, kernel)
        num += float(np.sum(wq * f * q**3))
        den += float(np.sum(wq * f * q**2))
    return float(num / (2.0 * den))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"name": "normal_cubic_bspline_kernel",
         "setup": "",
         "call": "pfpd_normalization_constant('cubic')",
         "gold_call": "_oracle_pfpd_normalization_constant('cubic')"},
        {"name": "boundary_linear_kernel_exact_rational",
         "setup": "",
         "call": "pfpd_normalization_constant('linear')",
         "gold_call": "_oracle_pfpd_normalization_constant('linear')"},
        {"name": "edge_constant_kernel_exact_rational",
         "setup": "",
         "call": "pfpd_normalization_constant('constant')",
         "gold_call": "_oracle_pfpd_normalization_constant('constant')"},
    ]
