"""
Step 06: Pumped charge, switching work and energy efficiency.

Pumped charge, switching work and energy efficiency per cycle of the coupling/decoupling pump.

In the periodic steady state the dot exchanges charge with lead 1 only during step 1 and with lead 2 only during step 2,
so the mean number of electrons pumped into lead 2 per cycle is the dot occupation at the end of step 1 minus the dot
occupation at the end of step 2. Switching the tunnel couplings instantaneously leaves the state unchanged, so the work
done on the device at a switching instant is the jump of the expectation value of the tunnelling Hamiltonian between the
dot and the leads. After the reaction-coordinate mapping, the tunnelling energy with lead nu while it is coupled is
2 sqrt(Gamma_nu lambda_nu / 2) Re <r_nu^+ d>.

This step returns, in the periodic steady state,
  N_pump, the electrons pumped into lead 2 per cycle;
  W_a, the work at the switch from step 1 to step 2 (lead 1 off, lead 2 on);
  W_b, the work at the switch from step 2 to the next step 1 (lead 2 off, lead 1 on);
  eta = N_pump V / (W_a + W_b), the fraction of the switching work stored as chemical energy in lead 2.

Energies are in the unit of params and the bias, and each quantity must be accurate to 5e-6 within
the ranges of the correlation step.

Inputs: params as in the drift step, t1 > 0, t2 > 0, bias V > 0. Output: float ndarray [N_pump, W_a, W_b, eta]. Invalid
inputs raise ValueError.

Returns
-------
numpy.ndarray of shape (4,), floats [N_pump, W_a, W_b, eta] in the periodic state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def pumping_performance(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    '''Pumped charge per cycle, the two switching works and the energy efficiency in the periodic steady state.

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
        Float array [N_pump, W_a, W_b, eta].

    Raises
    ------
    ValueError
        If params, the durations or the bias are invalid.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_pumping_performance(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    corr = _oracle_limit_cycle_correlations(p, t1, t2, bias)
    v = float(bias)
    m_both = _oracle_reaction_coordinate_drift(p, 1.0, 1.0)
    kappa1 = abs(m_both[0, 1])
    kappa2 = abs(m_both[0, 2])
    end2, end1 = corr[0], corr[1]
    n_pump = end1[0].real - end2[0].real
    w_a = 2.0 * kappa2 * end1[2].real - 2.0 * kappa1 * end1[1].real
    w_b = 2.0 * kappa1 * end2[1].real - 2.0 * kappa2 * end2[2].real
    return np.array([n_pump, w_a, w_b, n_pump * v / (w_a + w_b)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device, moderate step 2 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "pumping_performance(p, 1.128, 4.0, 1.0)",
            "gold_call": "_oracle_pumping_performance(p_g, 1.128, 4.0, 1.0)",
            "tol": 5e-06,
        },
        # --- Normal: benchmark device, long step 2 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "pumping_performance(p, 1.128, 9.0, 1.0)",
            "gold_call": "_oracle_pumping_performance(p_g, 1.128, 9.0, 1.0)",
            "tol": 5e-06,
        },
        # --- Boundary: broad lead 1 and a short step 1 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.25, 0.0, 0.5, 2.5, 1.25, 0.125, 0.25])\n"
                     "p_g = p.copy()\n",
            "call": "pumping_performance(p, 0.7793, 3.5, 1.0)",
            "gold_call": "_oracle_pumping_performance(p_g, 0.7793, 3.5, 1.0)",
            "tol": 5e-06,
        },
        # --- Edge: equal couplings, electrons flow with the bias and N_pump is negative ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.5, 1.0, 0.6, 1.0, 1.0, 0.6, 1.0])\n"
                     "p_g = p.copy()\n",
            "call": "pumping_performance(p, 6.0, 6.0, 1.0)",
            "gold_call": "_oracle_pumping_performance(p_g, 6.0, 6.0, 1.0)",
            "tol": 5e-06,
        },
        # --- Error: negative duration must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.1]), -1.0, 1.0, 1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(pumping_performance)",
            "gold_call": "_probe(_oracle_pumping_performance)",
        },
    ]
