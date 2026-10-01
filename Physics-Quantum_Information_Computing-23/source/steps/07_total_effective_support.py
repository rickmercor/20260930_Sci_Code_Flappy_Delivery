"""
Evaluates the grouped-protocol overhead on fSim(θ, φ) after truncation.

The prompt asks for one scalar on a locked fractional gate. This step only assembles the gate, the transfer matrix, and the truncated support, then returns the overhead of that table.

Returns
-------
A Python float, the scalar overhead.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def total_effective_support(theta: float, phi: float, tau: float) -> float:
    """Return the grouped-protocol overhead of fSim(theta, phi).
    Form the gate, take its Pauli-transfer matrix, keep the truncated
    non-identity nonzero support at cutoff tau, and return the finite-shot
    overhead of that table. Zero coefficients are excluded. An empty
    retained support returns 0. The identity gate at tau = 0 is valid and
    must not fail.
    Parameters
    ----------
    theta : float
        Exchange angle, finite.
    phi : float
        Conditional phase, finite.
    tau : float
        Nonnegative finite magnitude cutoff.
    Returns
    -------
    total : float
        The scalar overhead.
    Raises
    ------
    ValueError
        If theta or phi is not finite, if tau is not a nonnegative finite
        number, and any ValueError raised by the earlier steps for their
        invalid inputs.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_total_effective_support(theta: float, phi: float, tau: float) -> float:
    import numpy as np
    tau = float(tau)
    if tau != tau or tau == float("inf") or tau < 0.0:
        raise ValueError("tau must be a nonnegative finite number")
    unitary = _oracle_fsim_unitary(theta, phi)
    chi = _oracle_pauli_transfer_matrix(unitary)
    table = _oracle_retained_pair_table(chi, tau)
    return float(_oracle_partition_shot_overhead(table))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
theta, phi, tau = 0.0, np.pi, 1.0e-4
""",
            "call": "total_effective_support(theta, phi, tau)",
            "gold_call": "_oracle_total_effective_support(theta, phi, tau)",
        },
        {
            "setup": """import numpy as np
theta, phi, tau = np.pi / 2.0, 0.0, 1.0e-4
""",
            "call": "total_effective_support(theta, phi, tau)",
            "gold_call": "_oracle_total_effective_support(theta, phi, tau)",
        },
        {
            "setup": """import numpy as np
theta, phi, tau = 0.0, np.pi / 2.0, 1.0e-4
""",
            "call": "total_effective_support(theta, phi, tau)",
            "gold_call": "_oracle_total_effective_support(theta, phi, tau)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        total_effective_support(0.0, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_total_effective_support(0.0, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
theta, phi, tau = 0.0, 0.0, 0.0
""",
            "call": "total_effective_support(theta, phi, tau)",
            "gold_call": "_oracle_total_effective_support(theta, phi, tau)",
        },
        {
            "setup": """import numpy as np
theta, phi, tau = 0.0, np.pi, 0.0
""",
            "call": "total_effective_support(theta, phi, tau)",
            "gold_call": "_oracle_total_effective_support(theta, phi, tau)",
        },
    ]
