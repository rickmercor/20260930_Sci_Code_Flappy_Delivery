"""
Step 01: Drift matrix after the reaction-coordinate mapping.

Drift matrix of a quantum dot coupled to two structured leads after the reaction-coordinate mapping.

A single spinless quantum-dot level of energy eps is tunnel coupled to two noninteracting leads nu = 1, 2 at zero
temperature. Each lead has a Lorentzian spectral coupling density

  J_nu(omega) = Gamma_nu lambda_nu^2 / ((omega - omega_nu)^2 + lambda_nu^2),

with centre omega_nu, width lambda_nu > 0 and strength Gamma_nu > 0, and its tunnel amplitudes are multiplied by a
switching factor g_nu in [0, 1]. The fermionic reaction-coordinate mapping replaces each Lorentzian lead by a single
reaction-coordinate mode r_nu of energy omega_nu, coupled to the dot through g_nu sqrt(Gamma_nu lambda_nu / 2)
(r_nu^+ d + d^+ r_nu), and coupled in turn to a flat residual lead of spectral
density 2 lambda_nu. The residual leads keep the chemical potentials of the original leads.

For the annihilation operators ordered as a = (d, r_1, r_2), the Heisenberg equations take the form
da/dt = M a + (residual-lead noise), with

  M = -diag(i eps, lambda_1 + i omega_1, lambda_2 + i omega_2)
      - i g_1 sqrt(Gamma_1 lambda_1 / 2) (E_01 + E_10) - i g_2 sqrt(Gamma_2 lambda_2 / 2) (E_02 + E_20),

where E_ab is the 3 x 3 matrix with a single one at row a, column b (indices 0, 1, 2 for d, r_1, r_2). hbar = 1 and all
energies and rates share one unit.

Inputs: params = [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2] (seven finite reals with lambda_nu > 0 and
Gamma_nu > 0), switching factors g1 and g2 in [0, 1]. Output: M as a complex ndarray of shape (3, 3). A params array of
the wrong shape or with non-finite entries, a non-positive width or strength, or a switching factor outside [0, 1]
raises ValueError.

Returns
-------
numpy.ndarray of shape (3, 3), complex drift matrix M of (d, r_1, r_2) for the given switching factors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reaction_coordinate_drift(params: np.ndarray, g1: float, g2: float) -> np.ndarray:
    '''Drift matrix of (dot, reaction coordinate 1, reaction coordinate 2) for given switching factors.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2]; widths and strengths positive.
    g1 : float
        Switching factor of lead 1, in [0, 1].
    g2 : float
        Switching factor of lead 2, in [0, 1].

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, 3), the matrix M of the Heisenberg equations for (d, r_1, r_2).

    Raises
    ------
    ValueError
        If params does not have shape (7,), has non-finite entries or a non-positive width or strength, or a
        switching factor lies outside [0, 1].
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _checked_params(params):
    """Return params as a validated float array of shape (7,)."""
    import numpy as np
    try:
        p = np.asarray(params, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("params must be a real array")
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise ValueError("params must be a finite array of shape (7,)")
    if p[2] <= 0.0 or p[3] <= 0.0 or p[5] <= 0.0 or p[6] <= 0.0:
        raise ValueError("lead widths and strengths must be positive")
    return p


def _real_float(x, name):
    """Return x as a finite float or raise ValueError."""
    import numpy as np
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _oracle_reaction_coordinate_drift(params: np.ndarray, g1: float, g2: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    g1 = _real_float(g1, "g1")
    g2 = _real_float(g2, "g2")
    if not (0.0 <= g1 <= 1.0 and 0.0 <= g2 <= 1.0):
        raise ValueError("switching factors must lie in [0, 1]")
    eps, w1, l1, G1, w2, l2, G2 = p
    m = -np.diag([1j * eps, l1 + 1j * w1, l2 + 1j * w2]).astype(complex)
    k1 = g1 * np.sqrt(G1 * l1 / 2.0)
    k2 = g2 * np.sqrt(G2 * l2 / 2.0)
    m[0, 1] += -1j * k1
    m[1, 0] += -1j * k1
    m[0, 2] += -1j * k2
    m[2, 0] += -1j * k2
    return m

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: lead 1 coupled, lead 2 switched off ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "reaction_coordinate_drift(p, 1.0, 0.0)",
            "gold_call": "_oracle_reaction_coordinate_drift(p_g, 1.0, 0.0)",
            "tol": 1e-12,
        },
        # --- Normal: lead 2 coupled, lead 1 switched off ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "reaction_coordinate_drift(p, 0.0, 1.0)",
            "gold_call": "_oracle_reaction_coordinate_drift(p_g, 0.0, 1.0)",
            "tol": 1e-12,
        },
        # --- Boundary: both leads partly coupled, negative level and centre energies ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([-0.4, -1.5, 0.8, 0.6, 2.0, 0.05, 3.0])\n"
                     "p_g = p.copy()\n",
            "call": "reaction_coordinate_drift(p, 0.3, 0.7)",
            "gold_call": "_oracle_reaction_coordinate_drift(p_g, 0.3, 0.7)",
            "tol": 1e-12,
        },
        # --- Edge: both leads decoupled, only the diagonal survives ---
        {
            "setup": "import numpy as np\n"
                     "p = np.array([0.9, 0.2, 1.0, 1.0, 0.5, 2.0, 0.4])\n"
                     "p_g = p.copy()\n",
            "call": "reaction_coordinate_drift(p, 0.0, 0.0)",
            "gold_call": "_oracle_reaction_coordinate_drift(p_g, 0.0, 0.0)",
            "tol": 1e-12,
        },
        # --- Error: negative width must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, -0.1, 1.0, 1.0, 0.2, 0.1]), 1.0, 0.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(reaction_coordinate_drift)",
            "gold_call": "_probe(_oracle_reaction_coordinate_drift)",
        },
        # --- Error: switching factor above one must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.1]), 1.5, 0.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(reaction_coordinate_drift)",
            "gold_call": "_probe(_oracle_reaction_coordinate_drift)",
        },
    ]
