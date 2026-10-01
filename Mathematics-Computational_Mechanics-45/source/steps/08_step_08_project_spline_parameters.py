"""
Transfer one curvature-based spline onto a moved interpolation domain and return

the free variables of the transferred representation.

Moving the upper endpoint of an interpolation domain rescales the underlying

B-spline basis, so the same coefficients describe a different function and the

identified constitutive response would change for no physical reason. The

representation therefore has to be transferred onto the new basis, and the

transfer is posed as a non-negative least-squares problem over the sampled

invariants. Which property of the constitutive function the fit matches between

the old and the new basis is a design decision of the method rather than a free

choice, and it is chosen as the one that leaves the stress and the flow rule

unchanged. The fit is carried out in the physical parameters (d, c) because that

property is linear in them there, which it would not be in the free variables.

The non-negativity keeps the transferred function convex and monotone, so the

structural properties survive the basis change. The basis functions of the new

domain inherit the same continuation rule as the representation itself, so a

sample lying beyond the new upper endpoint is treated consistently with the way

the constitutive function is continued there. The physical parameters are finally

mapped back to free variables by inverting the map that produced them, applied to

entries floored at a small positive value so that a coefficient driven to zero

stays finite.

Returns
-------
np.ndarray of shape (n_c + 1,), the transferred free variables as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_spline_parameters(x_1: float, x_end_old: float, theta_old,
                              x_end_new: float, samples):
    """Transfer a curvature-based spline onto a moved interpolation domain.

    Parameters
    ----------
    x_1 : float
        Left endpoint of the interpolation domain.
    x_end_old : float
        Upper endpoint the current representation is defined on.
    theta_old : array-like of shape (n_c + 1,)
        Free variables of the current representation.
    x_end_new : float
        Updated upper endpoint.
    samples : array-like of shape (n,)
        Invariant values the transfer is fitted on.

    Returns
    -------
    theta_new : np.ndarray of shape (n_c + 1,)
        Free variables of the transferred representation.

    Raises
    ------
    ValueError
        If either upper endpoint fails to exceed x_1, if theta_old holds fewer
        than three entries, or if samples is empty.
    """
    return theta_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import nnls

_COEFFICIENT_FLOOR = 1e-12


def _integrated_basis(x_1, x_end, n_c, x):
    """Integrals of the degree-one basis functions from x_1 up to each sample."""
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    columns = np.empty((x.size, n_c))
    for i in range(n_c):
        c = np.zeros(n_c)
        c[i] = 1.0
        accumulated = np.zeros(n_c)
        for j in range(1, n_c):
            accumulated[j] = accumulated[j - 1] + 0.5 * h * (c[j - 1] + c[j])
        for k, xk in enumerate(x):
            if xk < x_1:
                columns[k, i] = c[0] * (xk - x_1)
            elif xk > x_end:
                columns[k, i] = accumulated[-1] + c[-1] * (xk - x_end)
            else:
                j = min(int((xk - x_1) / h), n_c - 2)
                u = xk - (x_1 + j * h)
                columns[k, i] = (accumulated[j] + c[j] * u
                                 + 0.5 * (c[j + 1] - c[j]) / h * u * u)
    return columns


def _spline_slopes(x_1, x_end, theta, x):
    """First derivative of a curvature-based spline at every sample."""
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    accumulated = np.zeros(n_c)
    for j in range(1, n_c):
        accumulated[j] = accumulated[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    out = np.empty_like(x)
    for k, xk in enumerate(x):
        if xk < x_1:
            out[k] = d + c[0] * (xk - x_1)
        elif xk > x_end:
            out[k] = d + accumulated[-1] + c[-1] * (xk - x_end)
        else:
            j = min(int((xk - x_1) / h), n_c - 2)
            u = xk - (x_1 + j * h)
            out[k] = d + accumulated[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u
    return out


def _oracle_project_spline_parameters(x_1: float, x_end_old: float, theta_old,
                                      x_end_new: float, samples):
    """Reference implementation."""
    theta_old = np.asarray(theta_old, dtype=float)
    values = np.atleast_1d(np.asarray(samples, dtype=float))
    if theta_old.ndim != 1 or theta_old.size < 3:
        raise ValueError("theta_old must be one dimensional with at least three entries")
    if values.size == 0:
        raise ValueError("samples must be non-empty")
    if not (float(x_end_old) > float(x_1) and float(x_end_new) > float(x_1)):
        raise ValueError("both upper endpoints must be strictly greater than x_1")

    n_c = theta_old.size - 1
    target = _spline_slopes(float(x_1), float(x_end_old), theta_old, values)
    design = np.empty((values.size, n_c + 1))
    design[:, 0] = 1.0
    design[:, 1:] = _integrated_basis(float(x_1), float(x_end_new), n_c, values)
    physical, _ = nnls(design, target)
    return np.log(np.expm1(np.maximum(physical, _COEFFICIENT_FLOOR)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# First isochoric elastic invariant sampled at every increment of the first
# forward solve of the prompt configuration.
_I1_SAMPLES = np.array([
    3.005557215916, 3.018729148122, 3.035434808619, 3.052854779267,
    3.069178077664, 3.083377283120, 3.094998401975, 3.103974751920,
    3.110475390693, 3.114792210002, 3.117263037842, 3.118224045628,
    3.117983618675, 3.116810598898, 3.114931359634, 3.112531801517,
    3.109761722056, 3.106740017979, 3.103559860934, 3.100293418464,
    3.054625773597, 3.026406338804, 3.010089355620, 3.002096411400,
    3.000007924915, 3.002140956090, 3.007264767235, 3.014464872697,
    3.023078322023, 3.032622189571, 3.042748521428, 3.053216415426,
    3.063869368300, 3.074617695129, 3.085424672518, 3.096295594311,
    3.107269229436, 3.118411336600, 3.129809982073, 3.141572462578
])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: an expanding domain for the first free energy
        # contribution of the prompt ---
        {
            "setup": """theta_old = [-1.0, -2.0, -2.5, -3.0]
samples = _I1_SAMPLES.copy()
""",
            "call": "[round(v, 10) for v in project_spline_parameters(3.0, 3.02, theta_old, 3.05, samples).tolist()]",
            "gold_call": "[round(v, 10) for v in _oracle_project_spline_parameters(3.0, 3.02, theta_old, 3.05, samples).tolist()]",
        },
        # --- Boundary case: an unmoved domain, where the transfer must return
        # the representation it was given ---
        {
            "setup": """theta_old = [-1.0, -2.0, -2.5, -3.0]
samples = _I1_SAMPLES.copy()
""",
            "call": "[round(v, 8) for v in project_spline_parameters(3.0, 3.02, theta_old, 3.02, samples).tolist()]",
            "gold_call": "[round(v, 8) for v in _oracle_project_spline_parameters(3.0, 3.02, theta_old, 3.02, samples).tolist()]",
        },
        # --- Edge case: an upper endpoint below the left endpoint must raise
        # ValueError ---
        {
            "setup": """theta_old = [-1.0, -2.0, -2.5, -3.0]
samples = _I1_SAMPLES.copy()

def run_model():
    try:
        project_spline_parameters(3.0, 3.02, theta_old, 2.5, samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_project_spline_parameters(3.0, 3.02, theta_old, 2.5, samples)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
