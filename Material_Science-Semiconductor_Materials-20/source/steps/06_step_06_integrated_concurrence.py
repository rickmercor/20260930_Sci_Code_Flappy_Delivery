"""
Step 06 - Single-time-integrated concurrence of the photon pairs.

Single-time-integrated concurrence of the emitted photon pairs.

The degree of polarisation entanglement of photon pairs that are detected together is
measured by the concurrence of their two-photon polarisation state. Including every
simultaneous detection event over the whole emission, without temporal filtering, gives
the single-time-integrated concurrence

    C_bar = 2 | int_0^inf G_HV dt | / ( int_0^inf G_HH dt + int_0^inf G_VV dt ),

built from the time-integrated equal-time correlations of step 05. C_bar lies between 0
and 1; phonon-induced decoherence and the fine-structure splitting lower it, and at the
temperatures of interest it decreases as the lattice warms.

Parameters are the lattice temperature, dot = [E_B (meV), delta (meV), g (meV),
gamma (1/ps)] as in step 05 and the lattice array of step 01. The value must be exact to
at least eight significant figures.

Returns
-------
float, the single-time-integrated concurrence C_bar (dimensionless, between 0 and 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrated_concurrence(T: float, dot: np.ndarray, lattice: np.ndarray) -> float:
    '''Single-time-integrated concurrence of the photon pairs at lattice temperature T.

    Parameters
    ----------
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    c_bar : float
        The single-time-integrated concurrence, dimensionless, in [0, 1].

    Raises
    ------
    ValueError
        If T is not finite and positive, if dot is not a numpy array of four finite real
        numbers with g > 0 and gamma > 0, or if lattice is invalid (as in step 01).
    '''
    return c_bar

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_integrated_concurrence(T: float, dot: np.ndarray, lattice: np.ndarray) -> float:
    ghh, gvv, ghv_re, ghv_im = _oracle_integrated_pair_correlations(T, dot, lattice)
    return float(2.0 * np.hypot(ghv_re, ghv_im) / (ghh + gvv))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark dot and cavity at 30 K ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "integrated_concurrence(30.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_concurrence(30.0, dot.copy(), lat.copy())",
            "tol": 1e-8,
        },
        # --- Boundary: the published parameter set at 4 K, close to the phonon-free limit ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.5, 0.1, 0.1, 0.25])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "integrated_concurrence(4.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_concurrence(4.0, dot.copy(), lat.copy())",
            "tol": 1e-8,
        },
        # --- Edge: large fine-structure splitting and a small biexciton binding energy ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([0.8, 0.4, 0.25, 0.5])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "integrated_concurrence(15.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_concurrence(15.0, dot.copy(), lat.copy())",
            "tol": 1e-8,
        },
        # --- Invalid: negative temperature ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.0])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        integrated_concurrence(-5.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_integrated_concurrence(-5.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
