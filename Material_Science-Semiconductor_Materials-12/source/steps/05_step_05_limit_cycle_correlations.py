"""
Step 05: Dot occupation and coherences in the periodic state.

Dot occupation and dot-reaction-coordinate coherences in the periodic steady state of the pump.

The residual leads are at zero temperature. Residual lead 1 is filled up to chemical potential mu_1 = 0 and residual
lead 2 up to mu_2 = V, the bias; both have flat spectral densities 2 lambda_nu. Integrating the spectral weights of the
limit-cycle-weights step over the occupied modes,

  C_ij = sum over nu of (2 lambda_nu / (2 pi)) integral from -infinity to mu_nu of Q^(nu)_ij(omega) d omega,

gives the correlation matrix C_ij = <a_i^+ a_j> of a = (d, r_1, r_2) at the end of step 2 and at the end of step 1 of
every cycle once the pump has forgotten its initial state. This step returns its dot column: C_00 = <d^+ d>, the dot
occupation, C_10 = <r_1^+ d> and C_20 = <r_2^+ d>, the coherences between the reaction coordinates and the dot.

The integrands decay only algebraically in omega and oscillate with the cycle period, so the integrals need a careful
treatment of the long tail. Near the Fermi levels they also have peaks of width about -ln(z_max) / (t1 + t2), where
z_max is the largest eigenvalue modulus of the one-cycle propagator, and these peaks become narrow when the pump forgets
its state slowly. Every returned entry must be accurate to 1e-7 for |eps|, |omega_nu| <= 5, 0.05 <= lambda_nu <= 2,
0 < Gamma_nu <= 5, 0 < t1 + t2 <= 20, 0 < V <= 3 and -ln(z_max) / (t1 + t2) >= 0.002.

Inputs: params as in the drift step, t1 > 0, t2 > 0, bias V > 0. Output: complex ndarray of shape (2, 3), rows [C_00, C_10,
C_20]; row 0 is the end of step 2 and row 1 the end of step 1. Invalid params or durations, or a bias that is not finite and positive,
raises ValueError.

Returns
-------
numpy.ndarray of shape (2, 3), complex dot occupation and coherences <r_nu^+ d> at the ends of step 2 and step 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def limit_cycle_correlations(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    '''Periodic-steady-state dot occupation and coherences <r_nu^+ d> at the ends of step 2 and step 1.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2 : float
        Duration of step 2, positive.
    bias : float
        V = mu_2 - mu_1 with mu_1 = 0, positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, 3) with rows [<d^+ d>, <r_1^+ d>, <r_2^+ d>]; [0] end of step 2, [1] end of step 1.

    Raises
    ------
    ValueError
        If params or the durations are invalid, or the bias is not finite and positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _fermi_sea_nodes(mu, cutoff, near=30.0, near_panel=0.04, far_panel=0.4, order=8):
    """Composite Gauss-Legendre nodes and weights on [mu - cutoff, mu], finer within `near` of mu."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(order)
    edges_near = np.linspace(mu - near, mu, int(round(near / near_panel)) + 1)
    edges_far = np.linspace(mu - cutoff, mu - near, int(round((cutoff - near) / far_panel)) + 1)
    edges = np.unique(np.concatenate([edges_far, edges_near]))
    lo = edges[:-1]
    hi = edges[1:]
    nodes = ((hi - lo) / 2)[:, None] * x[None, :] + ((hi + lo) / 2)[:, None]
    weights = ((hi - lo) / 2)[:, None] * w[None, :]
    return nodes.ravel(), weights.ravel()


def _oracle_limit_cycle_correlations(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    v = _real_float(bias, "bias")
    if v <= 0.0:
        raise ValueError("bias must be positive")
    z_max = np.max(np.abs(np.linalg.eigvals(_oracle_cycle_propagator(p, t1, t2))))
    scale = -np.log(z_max) / (float(t1) + float(t2))
    near_panel = min(0.04, 0.5 * scale)
    far_panel = min(0.4, 5.0 * scale)
    widths = {1: p[2], 2: p[5]}
    potentials = {1: 0.0, 2: v}
    estimates = []
    for cutoff in (400.0, 800.0):
        corr = np.zeros((2, 3, 3), dtype=complex)
        for nu in (1, 2):
            om, wq = _fermi_sea_nodes(potentials[nu], cutoff, near_panel=near_panel, far_panel=far_panel)
            q = _oracle_limit_cycle_weights(p, t1, t2, om, nu)
            corr += (2.0 * widths[nu] / (2.0 * np.pi)) * np.tensordot(wq, q, axes=(0, 0))
        estimates.append(corr)
    # the omitted tail beyond the cutoff decays as cutoff**-2
    best = estimates[1] + (estimates[1] - estimates[0]) / 3.0
    return np.ascontiguousarray(best[:, :, 0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "limit_cycle_correlations(p, 1.128, 4.3, 1.0)",
            "gold_call": "_oracle_limit_cycle_correlations(p_g, 1.128, 4.3, 1.0)",
            "tol": 3e-07,
        },
        # --- Normal: broader lead 1, larger bias ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.25, 0.0, 0.5, 2.5, 1.25, 0.125, 0.25])\n"
                     "p_g = p.copy()\n",
            "call": "limit_cycle_correlations(p, 0.78, 3.6, 1.5)",
            "gold_call": "_oracle_limit_cycle_correlations(p_g, 0.78, 3.6, 1.5)",
            "tol": 3e-07,
        },
        # --- Boundary: long steps, close to the stationary occupations of each step ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([0.8, -0.5, 1.0, 1.5, 0.6, 1.2, 0.8])\n"
                     "p_g = p.copy()\n",
            "call": "limit_cycle_correlations(p, 9.0, 10.0, 0.5)",
            "gold_call": "_oracle_limit_cycle_correlations(p_g, 9.0, 10.0, 0.5)",
            "tol": 3e-07,
        },
        # --- Edge: level below both Fermi levels, strongly coupled narrow lead 2 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([-0.3, 0.4, 0.3, 1.0, -0.2, 0.08, 4.0])\n"
                     "p_g = p.copy()\n",
            "call": "limit_cycle_correlations(p, 2.0, 1.5, 2.0)",
            "gold_call": "_oracle_limit_cycle_correlations(p_g, 2.0, 1.5, 2.0)",
            "tol": 3e-07,
        },
        # --- Error: zero bias must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.1]), 1.0, 1.0, 0.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(limit_cycle_correlations)",
            "gold_call": "_probe(_oracle_limit_cycle_correlations)",
        },
    ]
