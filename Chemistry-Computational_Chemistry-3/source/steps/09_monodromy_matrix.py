"""
Integrate the variational equations of the six-dimensional Cartesian flow alongside the trajectory for exactly one period, starting from the identity, and return the resulting six by six matrix.

The variational equations must be written in the Cartesian phase-space variables rather than in spherical ones. The orbit crosses the symmetry axis, where the azimuthal angle is undefined and the spherical variational equations are singular; integrating them there returns multipliers that are numerically meaningless. The Cartesian field is regular there because both angular factors of the potential are smooth functions of the Cartesian components away from the origin. Integrate to a relative and absolute tolerance of 1e-10 or tighter; the orbit is strongly unstable and looser settings corrupt the quantities read off it.

Returns
-------
ndarray of shape (6, 6): the matrix propagating an initial displacement over one period.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def monodromy_matrix(state: "np.ndarray", params: dict, period: float) -> "np.ndarray":
    """Integrate the variational equations of the six-dimensional Cartesian flow alongside the trajectory for exactly one period, starting from the identity, and return the resulting six by six matrix.

    Parameters
    ----------
    state : np.ndarray
        Six-component phase-space point on the orbit.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.
    period : float
        Positive period of the orbit, in the model time unit.

    Returns
    -------
    monodromy : np.ndarray
        ndarray of shape (6, 6): the matrix propagating an initial displacement over one period.

    Raises
    ------
    ValueError
        if the state does not have six components or the period is not positive and finite.
    """
    return monodromy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_monodromy_matrix(state: "np.ndarray", params: dict, period: float) -> "np.ndarray":
    y0 = np.asarray(state, float)
    if y0.shape != (6,):
        raise ValueError("state must have exactly six components")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be a positive finite number")
    def _jac(y):
        J = np.empty((6, 6)); eps = 1e-6
        for i in range(6):
            yp = y.copy(); ym = y.copy(); yp[i] += eps; ym[i] -= eps
            J[:, i] = (_oracle_vector_field(yp, params) - _oracle_vector_field(ym, params)) / (2.0 * eps)
        return J
    def _aug(t, z, q):
        y = z[:6]; P = z[6:].reshape(6, 6)
        return np.concatenate([_oracle_vector_field(y, q), (_jac(y) @ P).ravel()])
    z0 = np.concatenate([y0, np.eye(6).ravel()])
    s = solve_ivp(_aug, [0.0, period], z0, args=(params,), method='DOP853', rtol=1e-11, atol=1e-11)
    return s.y[6:, -1].reshape(6, 6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"tol": 1e-05, "setup": "import numpy as np",
         "call": "monodromy_matrix(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 5.92861852)",
         "gold_call": "_oracle_monodromy_matrix(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 5.92861852)"},   # normal
        {"tol": 1e-05, "setup": "import numpy as np",
         "call": "monodromy_matrix(np.array([0.0,0.0,3.63193607,0.61566,0.0,0.10566843]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30), 5.90399)",
         "gold_call": "_oracle_monodromy_matrix(np.array([0.0,0.0,3.63193607,0.61566,0.0,0.10566843]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30), 5.90399)"},   # normal
        {"tol": 1e-05, "setup": "import numpy as np",
         "call": "monodromy_matrix(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 2.9643)",
         "gold_call": "_oracle_monodromy_matrix(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 2.9643)"},   # boundary
        {"tol": 1e-05, "setup": "import numpy as np",
         "call": "monodromy_matrix(np.array([0.0,0.0,3.6,0.6,0.0,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 1.0)",
         "gold_call": "_oracle_monodromy_matrix(np.array([0.0,0.0,3.6,0.6,0.0,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), 1.0)"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(np.array([0.0,0.0,3.65,0.59,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0), -1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(monodromy_matrix)", "gold_call": "_c(_oracle_monodromy_matrix)"},   # exception contract
    ]
