"""
Implement srg_regulator, which evaluates elementwise the similarity-renormalization-group
denominator factor f(D_p, D_q; s) defined in the signature.

Similarity-renormalization-group variants of quasiparticle-self-consistent second-order
Green's-function theory depend on a flow parameter s.

Returns
-------
np.ndarray of the broadcast shape of delta_p and delta_q: renormalized denominator factor f(D_p, D_q; s) in hartree^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def srg_regulator(delta_p: "np.ndarray", delta_q: "np.ndarray", s: float) -> "np.ndarray":
    '''Renormalized denominator factor f(D_p, D_q; s), evaluated elementwise.

    Parameters
    ----------
    delta_p : np.ndarray
        Energy differences D_p in hartree, any shape.
    delta_q : np.ndarray
        Energy differences D_q in hartree, broadcast-compatible with delta_p.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.

    Returns
    -------
    factor : np.ndarray
        Broadcast shape of delta_p and delta_q, in hartree^-1:
        f = (D_p + D_q) / (D_p^2 + D_q^2) * [1 - exp(-(D_p^2 + D_q^2) s)],
        with f extended continuously to the points where D_p = D_q = 0.

    Raises
    ------
    ValueError
        If s is negative or not finite, or delta_p and delta_q cannot be broadcast together.
    '''
    return factor

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_srg_regulator(delta_p: "np.ndarray", delta_q: "np.ndarray", s: float) -> "np.ndarray":
    s = float(s)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("the flow parameter s must be finite and non-negative")
    try:
        dp, dq = np.broadcast_arrays(np.asarray(delta_p, dtype=float), np.asarray(delta_q, dtype=float))
    except ValueError as exc:
        raise ValueError("delta_p and delta_q cannot be broadcast together") from exc
    denom = dp * dp + dq * dq
    safe = np.where(denom > 0.0, denom, 1.0)
    factor = (dp + dq) / safe * -np.expm1(-safe * s)
    return np.where(denom > 0.0, factor, 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        srg_regulator(dp.copy(), dq.copy(), s)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_srg_regulator(dp.copy(), dq.copy(), s)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: mixed-sign 2h1p/2p1h-like denominators with a moderate flow parameter ---
        {
            "setup": """import numpy as np
dp = np.array([-1.8, -0.6, 0.35, 1.2, 2.9, -0.05])
dq = np.array([-1.5, -0.9, 0.40, 0.7, 3.4, 0.02])
s = 1.4
""",
            "call": "srg_regulator(dp.copy(), dq.copy(), s)",
            "gold_call": "_oracle_srg_regulator(dp.copy(), dq.copy(), s)",
            "tol": 1e-12,
        },
        # --- Typical: broadcasting a column of D_p against a row of D_q ---
        {
            "setup": """import numpy as np
dp = np.linspace(-2.0, 2.0, 7).reshape(7, 1)
dq = np.array([[-1.1, -0.3, 0.0, 0.25, 1.7]])
s = 0.525
""",
            "call": "srg_regulator(dp.copy(), dq.copy(), s)",
            "gold_call": "_oracle_srg_regulator(dp.copy(), dq.copy(), s)",
            "tol": 1e-12,
        },
        # --- Boundary: s = 0 switches the self-energy off (all factors vanish) ---
        {
            "setup": """import numpy as np
dp = np.array([-1.0, 0.5, 2.0])
dq = np.array([-0.4, 0.5, 1.0])
s = 0.0
""",
            "call": "srg_regulator(dp.copy(), dq.copy(), s)",
            "gold_call": "_oracle_srg_regulator(dp.copy(), dq.copy(), s)",
            "tol": 1e-12,
        },
        # --- Edge: exact zeros, cancelling pairs (D_q = -D_p) and a very large flow parameter
        #     that recovers the unregularized factor away from zero ---
        {
            "setup": """import numpy as np
dp = np.array([0.0, 0.8, -0.3, 0.6, 1.5])
dq = np.array([0.0, -0.8, 0.3, 0.6, 0.5])
s = 1.0e6
""",
            "call": "srg_regulator(dp.copy(), dq.copy(), s)",
            "gold_call": "_oracle_srg_regulator(dp.copy(), dq.copy(), s)",
            "tol": 1e-12,
        },
        # --- Invalid: negative flow parameter ---
        {
            "setup": """import numpy as np
dp = np.array([1.0, 2.0])
dq = np.array([0.5, 1.0])
s = -0.1
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
