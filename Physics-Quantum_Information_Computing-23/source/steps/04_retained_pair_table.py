"""
Extracts the truncated non-identity Pauli-transfer support as a numeric table of (α, β, χ) rows.

Only Pauli pairs with a nonzero target coefficient enter the sampling plan. Small coefficients may be dropped by a magnitude cutoff, and the identity pair is omitted because its estimate is trivial.

Returns
-------
A float64 ndarray of shape (K, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def retained_pair_table(chi: np.ndarray, tau: float) -> np.ndarray:
    """Return retained (α, β, χ_U(α, β)) rows after dropping the identity pair.
    Keep every nonzero entry with |χ_U(α, β)| ≥ tau except the (I, I) slot
    (α = β = 0). Zero coefficients are excluded even when tau = 0. Rows are
    ordered by increasing α, then increasing β.
    Parameters
    ----------
    chi : np.ndarray
        Real array of shape (16, 16).
    tau : float
        Nonnegative finite magnitude cutoff. Zero is allowed.
    Returns
    -------
    table : np.ndarray
        Array of shape (K, 3). Each row is (α, β, χ_U(α, β)). K = 0 is allowed
        and returns shape (0, 3).
    Raises
    ------
    ValueError
        If chi is not a finite real 16×16 array, if any entry has a nonzero
        imaginary part, or tau is not a nonnegative finite number.
    """
    return np.zeros((0, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_retained_pair_table(chi: np.ndarray, tau: float) -> np.ndarray:
    import numpy as np
    raw = np.asarray(chi)
    if raw.shape != (16, 16):
        raise ValueError("chi must have shape (16, 16)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("chi must be finite")
    as_c = np.asarray(raw, dtype=complex)
    if np.any(np.abs(as_c.imag) > 0.0):
        raise ValueError("chi must be real")
    C = np.asarray(as_c.real, dtype=float)
    tau = float(tau)
    if tau != tau or tau == float("inf") or tau < 0.0:
        raise ValueError("tau must be a nonnegative finite number")
    rows = []
    for alpha in range(16):
        for beta in range(16):
            if alpha == 0 and beta == 0:
                continue
            value = float(C[alpha, beta])
            if value == 0.0:
                continue
            if abs(value) >= tau:
                rows.append((float(alpha), float(beta), value))
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
chi = np.eye(16, dtype=float)
tau = 0.5
""",
            "call": "retained_pair_table(chi.copy(), tau)",
            "gold_call": "_oracle_retained_pair_table(chi.copy(), tau)",
        },
        {
            "setup": """import numpy as np
chi = np.eye(16, dtype=float)
tau = 1.5
""",
            "call": "retained_pair_table(chi.copy(), tau)",
            "gold_call": "_oracle_retained_pair_table(chi.copy(), tau)",
        },
        {
            "setup": """import numpy as np
chi = _oracle_pauli_transfer_matrix(_oracle_fsim_unitary(0.0, np.pi))
tau = 1.0e-4
""",
            "call": "retained_pair_table(chi.copy(), tau)",
            "gold_call": "_oracle_retained_pair_table(chi.copy(), tau)",
        },
        {
            "setup": """import numpy as np
chi = np.eye(16, dtype=float)
def run_model():
    try:
        retained_pair_table(chi.copy(), -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_retained_pair_table(chi.copy(), -0.1)
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
chi = np.eye(16, dtype=complex)
chi[1, 1] = 1.0 + 0.3j
def run_model():
    try:
        retained_pair_table(chi.copy(), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_retained_pair_table(chi.copy(), 0.0)
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
chi = np.zeros((16, 16), dtype=float)
chi[0, 0] = 1.0
tau = 0.0
""",
            "call": "retained_pair_table(chi.copy(), tau)",
            "gold_call": "_oracle_retained_pair_table(chi.copy(), tau)",
        },
    ]
