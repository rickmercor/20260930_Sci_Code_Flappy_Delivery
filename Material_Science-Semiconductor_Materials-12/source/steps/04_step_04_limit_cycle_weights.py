"""
Step 04: Periodic-state spectral weights of one residual lead.

Spectral weights of one residual lead in the periodic steady state of the coupling/decoupling pump.

With the pump started at time 0 from any state, the operators of (d, r_1, r_2) at time t are the homogeneous part plus a
sum over residual-lead modes. For residual lead nu (nu = 1 feeds r_1, nu = 2 feeds r_2) the mode of energy omega enters
through the response vector

  u(omega, t) = integral from 0 to t of O(t, s) e_nu exp(-i omega s) ds,

where O(t, s) is the homogeneous propagator of the piecewise-constant drift (step 1 first, starting at s = 0) and e_nu is
the unit vector of component nu. Because the homogeneous propagator contracts from cycle to cycle, the products
conj(u_i) u_j settle into a periodic pattern. This step returns their limits at the ends of the steps,

  Q_ij(omega) = lim over n of conj(u_i(omega, t_n)) u_j(omega, t_n),

with t_n = n (t1 + t2) (end of step 2, index 0 of the second axis) and t_n = n (t1 + t2) + t1 (end of step 1, index 1).
Weighted by 2 lambda_nu / (2 pi) and integrated over the occupied modes of the residual lead, they give that lead's
contribution to the correlation matrix <a_i^+ a_j> of the periodic steady state.

The weights must be exact up to rounding.

Inputs: params as in the drift step, t1 > 0, t2 > 0, omega a finite one-dimensional array, lead equal to 1 or 2.
Output: complex ndarray of shape (len(omega), 2, 3, 3). Invalid params or durations, an invalid omega array, or a lead
other than 1 or 2 raises ValueError.

Returns
-------
numpy.ndarray of shape (len(omega), 2, 3, 3), complex periodic-state spectral weights at the ends of step 2 and step 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def limit_cycle_weights(params: np.ndarray, t1: float, t2: float, omega: np.ndarray, lead: int) -> np.ndarray:
    '''Periodic-steady-state spectral weights Q_ij(omega) of one residual lead at the ends of both steps.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2 : float
        Duration of step 2, positive.
    omega : np.ndarray
        One-dimensional array of residual-lead mode energies.
    lead : int
        1 or 2, the residual lead whose modes are resolved.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (len(omega), 2, 3, 3); [:, 0] at the end of step 2, [:, 1] at the end of step 1.

    Raises
    ------
    ValueError
        If params, the durations or omega are invalid, or lead is not 1 or 2.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_limit_cycle_weights(params: np.ndarray, t1: float, t2: float, omega: np.ndarray,
                                lead: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    if isinstance(lead, (bool, np.bool_)) or not isinstance(lead, (int, np.integer)) or int(lead) not in (1, 2):
        raise ValueError("lead must be 1 or 2")
    nu = int(lead)
    a = _oracle_cycle_propagator(params, t1, t2)
    t1 = float(t1)
    t2 = float(t2)
    m1 = _oracle_reaction_coordinate_drift(params, 1.0, 0.0)
    m2 = _oracle_reaction_coordinate_drift(params, 0.0, 1.0)
    om = np.asarray(omega, dtype=float)
    s1 = _oracle_segment_response(m1, t1, om)[:, :, nu]
    s2 = _oracle_segment_response(m2, t2, om)[:, :, nu]
    e1 = expm(m1 * t1)
    e2 = expm(m2 * t2)
    period = t1 + t2
    u_cycle = np.einsum('nj,ij->ni', s1, e2) + np.exp(-1j * om * t1)[:, None] * s2
    eye = np.eye(3)
    lhs = eye[None, :, :] - np.exp(1j * om * period)[:, None, None] * a[None, :, :]
    w_end2 = np.linalg.solve(lhs, u_cycle[:, :, None])[:, :, 0]
    w_end1 = np.einsum('nj,ij->ni', w_end2, e1) + np.exp(-1j * om * period)[:, None] * s1
    out = np.empty((om.size, 2, 3, 3), dtype=complex)
    out[:, 0] = np.conj(w_end2)[:, :, None] * w_end2[:, None, :]
    out[:, 1] = np.conj(w_end1)[:, :, None] * w_end1[:, None, :]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device, lead 1 modes below its Fermi level ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "w = np.array([-20.0, -2.0, -0.6, -0.1, 0.0])\n"
                     "p_g = p.copy()\n"
                     "w_g = w.copy()\n",
            "call": "limit_cycle_weights(p, 1.128, 4.1, w, 1)",
            "gold_call": "_oracle_limit_cycle_weights(p_g, 1.128, 4.1, w_g, 1)",
            "tol": 1e-09,
        },
        # --- Normal: benchmark device, lead 2 modes up to its Fermi level at the bias ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "w = np.array([-5.0, 0.3, 0.8, 1.0])\n"
                     "p_g = p.copy()\n"
                     "w_g = w.copy()\n",
            "call": "limit_cycle_weights(p, 1.128, 9.0, w, 2)",
            "gold_call": "_oracle_limit_cycle_weights(p_g, 1.128, 9.0, w_g, 2)",
            "tol": 1e-09,
        },
        # --- Boundary: cycle length commensurate with a mode energy, exp(i omega T) = 1 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.25, 0.0, 0.5, 2.5, 1.25, 0.125, 0.25])\n"
                     "w = np.array([-2 * np.pi / 3.0, 0.0])\n"
                     "p_g = p.copy()\n"
                     "w_g = w.copy()\n",
            "call": "limit_cycle_weights(p, 1.0, 2.0, w, 1)",
            "gold_call": "_oracle_limit_cycle_weights(p_g, 1.0, 2.0, w_g, 1)",
            "tol": 1e-09,
        },
        # --- Edge: long steps approach the stationary weights of each step ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([-0.4, -1.5, 0.8, 0.6, 2.0, 0.05, 3.0])\n"
                     "w = np.array([-3.0, -1.5, -0.2])\n"
                     "p_g = p.copy()\n"
                     "w_g = w.copy()\n",
            "call": "limit_cycle_weights(p, 30.0, 45.0, w, 2)",
            "gold_call": "_oracle_limit_cycle_weights(p_g, 30.0, 45.0, w_g, 2)",
            "tol": 1e-09,
        },
        # --- Error: lead 3 must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.1]), 1.0, 1.0, np.array([0.0]), 3)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(limit_cycle_weights)",
            "gold_call": "_probe(_oracle_limit_cycle_weights)",
        },
    ]
