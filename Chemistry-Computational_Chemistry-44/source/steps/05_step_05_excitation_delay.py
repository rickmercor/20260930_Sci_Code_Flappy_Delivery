"""
Step 05: Excitation delay at a given confidence level.

Time needed to reach a given statistical confidence that the target level has participated in the flow of probability.

The non-participation probability P_not(t_k) of the previous step is sampled at increasing times t_k and never
increases. For a confidence level of b percent, the excitation delay tau_b is the earliest time at which
P_not(t) <= 1 - b/100, with P_not(t) taken as the straight line between consecutive samples. If the first sample
already satisfies the condition, tau_b = t_0. If no sample reaches the level, the confidence is not attained within the
history and tau_b is returned as +infinity.

tau_90, for example, is the time at which one is 90 % confident that the target has participated.

Returns
-------
float, excitation delay tau_b in the units of times, or inf if the confidence level is never reached
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def excitation_delay(times: "np.ndarray", nonparticipation: "np.ndarray", confidence: float) -> float:
    '''Earliest time at which the linearly interpolated non-participation probability falls to 1 - confidence / 100.

    Parameters
    ----------
    times : np.ndarray
        Shape (K,), strictly increasing sample times, K >= 2.
    nonparticipation : np.ndarray
        Shape (K,), non-increasing non-participation probabilities P_not(t_k).
    confidence : float
        Confidence level b in percent, 0 < b < 100.

    Returns
    -------
    result : float
        The excitation delay tau_b, or float('inf') if P_not never falls to 1 - b/100.

    Raises
    ------
    ValueError
        If the arrays have different lengths or fewer than two entries, the times are not strictly increasing, or the
        confidence is not strictly between 0 and 100.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_excitation_delay(times: "np.ndarray", nonparticipation: "np.ndarray", confidence: float) -> float:
    """Reference implementation."""
    import numpy as np
    t = np.asarray(times, dtype=float).ravel()
    p = np.asarray(nonparticipation, dtype=float).ravel()
    if t.size != p.size or t.size < 2:
        raise ValueError("times and nonparticipation must have the same length of at least 2")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not 0.0 < confidence < 100.0:
        raise ValueError("confidence must lie strictly between 0 and 100")
    level = 1.0 - confidence / 100.0
    reached = np.nonzero(p <= level)[0]
    if reached.size == 0:
        return float("inf")
    k = int(reached[0])
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (t[k] - t[k - 1]) * (p[k - 1] - level) / (p[k - 1] - p[k]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 90 % confidence reached between two samples ---
        {
            "setup": "import numpy as np\n"
                     "t = np.array([0.0, 10.0, 20.0, 30.0, 40.0])\n"
                     "p = np.array([1.0, 0.6, 0.3, 0.05, 0.01])\n",
            "call": "excitation_delay(t, p, 90.0)",
            "gold_call": "_oracle_excitation_delay(t, p, 90.0)",
            "tol": 1e-12,
        },
        # --- Normal: a plateau followed by a drop, 50 % confidence on a non-uniform grid ---
        {
            "setup": "import numpy as np\n"
                     "t = np.array([0.0, 1.5, 4.0, 4.5, 9.0, 12.0])\n"
                     "p = np.array([1.0, 0.8, 0.62, 0.62, 0.41, 0.2])\n",
            "call": "excitation_delay(t, p, 50.0)",
            "gold_call": "_oracle_excitation_delay(t, p, 50.0)",
            "tol": 1e-12,
        },
        # --- Boundary: the level is met exactly at a sample ---
        {
            "setup": "import numpy as np\n"
                     "t = np.array([100.0, 200.0, 300.0])\n"
                     "p = np.array([0.5, 0.25, 0.0])\n",
            "call": "excitation_delay(t, p, 75.0)",
            "gold_call": "_oracle_excitation_delay(t, p, 75.0)",
            "tol": 1e-12,
        },
        # --- Edge: the first sample already satisfies the confidence level ---
        {
            "setup": "import numpy as np\n"
                     "t = np.array([5.0, 6.0, 7.0])\n"
                     "p = np.array([0.05, 0.04, 0.03])\n",
            "call": "excitation_delay(t, p, 90.0)",
            "gold_call": "_oracle_excitation_delay(t, p, 90.0)",
            "tol": 1e-12,
        },
        # --- Boundary: the confidence level is never reached, so the delay is infinite ---
        {
            "setup": "import numpy as np\n"
                     "t = np.array([0.0, 1.0, 2.0, 3.0])\n"
                     "p = np.array([1.0, 0.9, 0.85, 0.8])\n",
            "call": "excitation_delay(t, p, 50.0) == float('inf')",
            "gold_call": "_oracle_excitation_delay(t, p, 50.0) == float('inf')",
        },
        # --- Error: a confidence of 100 % must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.0, 1.0]), np.array([1.0, 0.5]), 100.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(excitation_delay)",
            "gold_call": "_probe(_oracle_excitation_delay)",
        },
    ]
