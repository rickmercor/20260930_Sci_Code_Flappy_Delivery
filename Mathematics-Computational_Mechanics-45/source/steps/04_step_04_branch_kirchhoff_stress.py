"""
Return the deviatoric non-equilibrium Kirchhoff stress of one Maxwell branch in

its principal basis, together with the scalar stress invariant that the dual

dissipation potential depends on.

The branch free energy is additively split over the isochoric invariants of the

elastic left Cauchy-Green tensor b_e, psi_branch = psi1(I1_e) + psi2(I2_e), and

for an isotropic energy the Kirchhoff stress is coaxial with b_e and follows from

the derivative of the energy with respect to it, tau_A = 2 b_A dpsi_branch/db_A

in the shared principal basis. Evaluating that derivative needs the sensitivity

of each invariant to the principal values: the first invariant contributes

immediately, while the transformed second invariant carries an extra factor

produced by the power and the shift in its definition. Incompressibility means

only the deviatoric part carries constitutive information, so the branch stress

is tau_neq = dev(tau), obtained by subtracting the mean of the three principal

components. The dual dissipation potential of an isotropic incompressible branch

depends on tau_neq through a single deviatoric stress invariant J_tau, a fixed

combination of the first two principal invariants of the non-equilibrium

Kirchhoff stress. That combination is non-negative, vanishes for a purely

volumetric stress state and is convex in tau_neq, so any convex scalar function

of it is a thermodynamically admissible dual potential.

Returns
-------
tuple of an np.ndarray of shape (3,) and a native Python float, the deviatoric branch stress and its invariant
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline):
    """Return the branch stress and its deviatoric invariant.

    Parameters
    ----------
    beta_e : array-like of shape (3,)
        Principal values of the elastic left Cauchy-Green tensor.
    psi1_spline : tuple
        (x_1, x_end, theta) for the first free energy contribution.
    psi2_spline : tuple
        (x_1, x_end, theta) for the second free energy contribution.

    Returns
    -------
    branch_stress : tuple
        The principal components of the deviatoric non-equilibrium Kirchhoff
        stress and the deviatoric stress invariant driving the dual dissipation
        potential, in that order.

    Raises
    ------
    ValueError
        If beta_e does not hold exactly three strictly positive entries or if a
        spline descriptor is not a triple.
    """
    return branch_stress

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_SQRT3 = np.sqrt(3.0)


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


def _oracle_branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline):
    """Reference implementation."""
    beta = np.asarray(beta_e, dtype=float)
    if beta.shape != (3,):
        raise ValueError("beta_e must hold exactly three entries")
    if not np.all(beta > 0.0):
        raise ValueError("beta_e entries must be strictly positive")
    for spline in (psi1_spline, psi2_spline):
        if len(spline) != 3:
            raise ValueError("a spline descriptor must be (x_1, x_end, theta)")

    determinant = float(beta[0] * beta[1] * beta[2])
    unimodular = beta * determinant ** (-1.0 / 3.0)
    I1 = float(unimodular.sum())
    w = float(unimodular[0] * unimodular[1] + unimodular[0] * unimodular[2]
              + unimodular[1] * unimodular[2])
    I2 = float(w ** 1.5 - 3.0 * _SQRT3)

    slope_1 = _spline_slope(psi1_spline, I1)
    slope_2 = _spline_slope(psi2_spline, I2)
    tau_full = 2.0 * beta * (slope_1 + slope_2 * 1.5 * np.sqrt(w) * (I1 - beta))
    tau_neq = tau_full - tau_full.mean()
    J_tau = 1.5 * float(np.dot(tau_neq, tau_neq))
    return tau_neq, J_tau

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# Trial elastic state of the first increment of the prompt configuration.
_TRIAL_PRINCIPALS = np.array([1.1025, 0.952380952380952, 0.952380952380952])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the branch splines of the prompt ---
        {
            "setup": """beta_e = _TRIAL_PRINCIPALS.copy()
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))

def run(fn):
    tau_neq, J_tau = fn(beta_e, psi1_spline, psi2_spline)
    return [[round(v, 12) for v in tau_neq.tolist()], round(float(J_tau), 12)]
""",
            "call": "run(branch_kirchhoff_stress)",
            "gold_call": "run(_oracle_branch_kirchhoff_stress)",
        },
        # --- Boundary case: the undeformed elastic state, where the branch is
        # stress free and the stress invariant vanishes ---
        {
            "setup": """beta_e = np.array([1.0, 1.0, 1.0])
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))

def run(fn):
    tau_neq, J_tau = fn(beta_e, psi1_spline, psi2_spline)
    return [[round(v, 12) for v in tau_neq.tolist()], round(float(J_tau), 12)]
""",
            "call": "run(branch_kirchhoff_stress)",
            "gold_call": "run(_oracle_branch_kirchhoff_stress)",
        },
        # --- Edge case: a negative principal value must raise ValueError ---
        {
            "setup": """beta_e = np.array([1.1, -0.5, 0.9])
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))

def run_model():
    try:
        branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline)
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
