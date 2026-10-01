"""
Advance one Maxwell branch over a single time increment with the implicit

exponential mapping algorithm and return the converged principal values of the

elastic left Cauchy-Green tensor.

Freezing the viscous evolution over an increment gives the trial elastic left

Cauchy-Green tensor b_e_trial = F * inverse(C_v) * transpose(F) formed with the

end-of-increment deformation gradient and the internal variable of the previous

increment. The trial state is then corrected through the viscous flow by an

exponential mapping algorithm, a choice that preserves symmetry and positive

definiteness of b_e throughout the update. Because b_e, b_e_trial and tau_neq are

coaxial for an isotropic branch, the correction is carried out in their shared

principal basis, where the tensor exponential collapses to a scalar relation

between the logarithmic principal elastic strains of the updated state and those

of the trial state. That relation is implicit, because the rate entering it is

the flow rule evaluated at the updated state, so each increment closes on a

three-dimensional nonlinear system in those strains and is solved by a local

Newton iteration. Since the dual potential depends only on a deviatoric stress

invariant the rate is traceless, so the correction has unit determinant and the

update preserves det(b_e), which is what keeps the viscous deformation unimodular

and consistent with incompressibility.

Returns
-------
np.ndarray of shape (3,), the converged principal values of the elastic left Cauchy-Green tensor as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def local_branch_update(beta_trial, dt: float, psi1_spline, psi2_spline,
                        phi_spline, tol: float = 1e-12, max_iter: int = 50):
    """Advance one Maxwell branch over a single increment.

    Parameters
    ----------
    beta_trial : array-like of shape (3,)
        Principal values of the trial elastic left Cauchy-Green tensor.
    dt : float
        Time increment.
    psi1_spline, psi2_spline : tuple
        (x_1, x_end, theta) for the two free energy contributions.
    phi_spline : tuple
        (x_1, x_end, theta) for the dual dissipation potential.
    tol : float
        Tolerance on the residual infinity norm.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    beta_e : np.ndarray of shape (3,)
        Converged principal values of the elastic left Cauchy-Green tensor.

    Raises
    ------
    ValueError
        If beta_trial does not hold exactly three strictly positive entries, if
        dt is not strictly positive, or if max_iter is not a positive integer.
    """
    return beta_e

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_SQRT3 = np.sqrt(3.0)
_FD_STEP = 1e-7


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


def _viscous_rate(beta, psi1_spline, psi2_spline, phi_spline):
    """Viscous deformation-type rate at an elastic state."""
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
    return 3.0 * _spline_slope(phi_spline, J_tau) * tau_neq


def _oracle_local_branch_update(beta_trial, dt: float, psi1_spline, psi2_spline,
                                phi_spline, tol: float = 1e-12, max_iter: int = 50):
    """Reference implementation."""
    beta_trial = np.asarray(beta_trial, dtype=float)
    if beta_trial.shape != (3,):
        raise ValueError("beta_trial must hold exactly three entries")
    if not np.all(beta_trial > 0.0):
        raise ValueError("beta_trial entries must be strictly positive")
    if not float(dt) > 0.0:
        raise ValueError("dt must be strictly positive")
    if not (isinstance(max_iter, int) and max_iter > 0):
        raise ValueError("max_iter must be a positive integer")

    eps_trial = 0.5 * np.log(beta_trial)

    def residual(eps):
        rate = _viscous_rate(np.exp(2.0 * eps), psi1_spline, psi2_spline,
                             phi_spline)
        return eps - eps_trial + float(dt) * rate

    eps = eps_trial.copy()
    for _ in range(max_iter):
        r = residual(eps)
        if np.max(np.abs(r)) < float(tol):
            break
        jacobian = np.empty((3, 3))
        for j in range(3):
            forward = eps.copy()
            backward = eps.copy()
            forward[j] += _FD_STEP
            backward[j] -= _FD_STEP
            jacobian[:, j] = (residual(forward) - residual(backward)) / (2.0 * _FD_STEP)
        eps = eps - np.linalg.solve(jacobian, r)

    return np.exp(2.0 * eps)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the first increment of the prompt configuration,
        # started from an undeformed internal state ---
        {
            "setup": """beta_trial = np.array([1.1025, 0.952380952380952, 0.952380952380952])
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))

def run(fn):
    beta_e = fn(beta_trial, 1.0, psi1_spline, psi2_spline, phi_spline)
    return [
        [round(v, 10) for v in beta_e.tolist()],
        round(float(beta_e[0] * beta_e[1] * beta_e[2]), 10),
    ]
""",
            "call": "run(local_branch_update)",
            "gold_call": "run(_oracle_local_branch_update)",
        },
        # --- Boundary case: an undeformed trial state, where the branch is
        # stress free and the update leaves the state unchanged ---
        {
            "setup": """beta_trial = np.ones(3)
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))

def run(fn):
    beta_e = fn(beta_trial, 1.0, psi1_spline, psi2_spline, phi_spline)
    return [round(v, 12) for v in beta_e.tolist()]
""",
            "call": "run(local_branch_update)",
            "gold_call": "run(_oracle_local_branch_update)",
        },
        # --- Edge case: a non-positive time increment must raise ValueError ---
        {
            "setup": """beta_trial = np.array([1.1025, 0.952380952380952, 0.952380952380952])
psi1_spline = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
psi2_spline = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))
phi_spline = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))

def run_model():
    try:
        local_branch_update(beta_trial, 0.0, psi1_spline, psi2_spline, phi_spline)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_local_branch_update(beta_trial, 0.0, psi1_spline, psi2_spline, phi_spline)
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
