"""
Return the principal components of the viscous deformation-type rate of one

Maxwell branch from its deviatoric non-equilibrium Kirchhoff stress and the dual

dissipation potential.

In the generalized standard material setting the dissipative response is fixed by

a dissipation potential, and the dual form obtained by the Legendre-Fenchel

transform is the convenient one because it gives the evolution law explicitly:

the viscous deformation-type rate is the gradient of the dual potential with

respect to its conjugate stress, d_e = dphi_dual/dtau_neq. Admissibility requires

phi to be convex, non-negative and normalised so that phi(0) = 0. For an

isotropic incompressible branch, volumetric dissipation is excluded and phi

depends on the stress only through the deviatoric stress invariant J_tau, so the

chain rule makes the rate the product of phi'(J_tau) with the gradient of that

invariant with respect to the stress. For a stress that is already traceless

that gradient is parallel to the stress itself, with a constant of

proportionality fixed by the combination of principal invariants J_tau is built

from, so the rate is parallel to tau_neq. Two consequences follow. The rate

inherits the sign of phi'(J_tau), which is non-negative because phi is monotone,

so the reduced dissipation cannot be negative. The rate is also traceless, which

is what keeps the exponential update of the internal variable unimodular and

makes the viscous flow purely isochoric.

Returns
-------
np.ndarray of shape (3,), the principal viscous rate as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def viscous_flow_rate(tau_neq, J_tau: float, phi_spline):
    """Return the principal viscous deformation-type rate of one branch.

    Parameters
    ----------
    tau_neq : array-like of shape (3,)
        Principal components of the deviatoric non-equilibrium Kirchhoff stress.
    J_tau : float
        Deviatoric stress invariant at which the dual potential is evaluated.
    phi_spline : tuple
        (x_1, x_end, theta) for the dual dissipation potential.

    Returns
    -------
    d_e : np.ndarray of shape (3,)
        Principal components of the viscous deformation-type rate.

    Raises
    ------
    ValueError
        If tau_neq does not hold exactly three entries, if J_tau is negative, or
        if phi_spline is not a triple.
    """
    return d_e

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _spline_slope(spline, argument):
    """First derivative of a curvature-based spline at one argument."""
    x_1, x_end, theta = spline
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    slope_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    if argument < x_1:
        return d + c[0] * (argument - x_1)
    if argument > x_end:
        return d + slope_at[-1] + c[-1] * (argument - x_end)
    j = min(int((argument - x_1) / h), n_c - 2)
    u = argument - (x_1 + j * h)
    return d + slope_at[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u


def _oracle_viscous_flow_rate(tau_neq, J_tau: float, phi_spline):
    """Reference implementation."""
    tau = np.asarray(tau_neq, dtype=float)
    if tau.shape != (3,):
        raise ValueError("tau_neq must hold exactly three entries")
    if not float(J_tau) >= 0.0:
        raise ValueError("J_tau must be non-negative")
    if len(phi_spline) != 3:
        raise ValueError("phi_spline must be (x_1, x_end, theta)")

    return 3.0 * _spline_slope(phi_spline, float(J_tau)) * tau

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# Deviatoric branch stress and invariant of the trial state of the first
# increment of the prompt configuration.
_TRIAL_STRESS = np.array([0.126175604025, -0.063087802012, -0.063087802012])
_TRIAL_J_TAU = 0.035820636865


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the dissipation potential of the prompt, on an
        # argument that lies in the constant-curvature continuation ---
        {
            "setup": """tau_neq = _TRIAL_STRESS.copy()
J_tau = _TRIAL_J_TAU
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))
""",
            "call": "[round(v, 12) for v in viscous_flow_rate(tau_neq, J_tau, phi_spline).tolist()]",
            "gold_call": "[round(v, 12) for v in _oracle_viscous_flow_rate(tau_neq, J_tau, phi_spline).tolist()]",
        },
        # --- Boundary case: a stress free branch, where the rate vanishes ---
        {
            "setup": """tau_neq = np.zeros(3)
J_tau = 0.0
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))
""",
            "call": "[round(v, 12) for v in viscous_flow_rate(tau_neq, J_tau, phi_spline).tolist()]",
            "gold_call": "[round(v, 12) for v in _oracle_viscous_flow_rate(tau_neq, J_tau, phi_spline).tolist()]",
        },
        # --- Edge case: a negative stress invariant must raise ValueError ---
        {
            "setup": """tau_neq = _TRIAL_STRESS.copy()
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))

def run_model():
    try:
        viscous_flow_rate(tau_neq, -1.0, phi_spline)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_viscous_flow_rate(tau_neq, -1.0, phi_spline)
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
