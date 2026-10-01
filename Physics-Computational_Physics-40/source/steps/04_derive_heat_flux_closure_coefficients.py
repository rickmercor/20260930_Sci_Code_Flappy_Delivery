"""
Convert the matched Padé parameters into real closure coefficients.

The parity-symmetric three-pole form yields Q2=0 and wave-number-dependent Q1 and Q3.

Returns
-------
np.ndarray: float rows [kappa, Re(zeta), Im(zeta), Q1, Q3].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def derive_closure_coefficients(pade_table: np.ndarray) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), Q1, Q3]``.

    Parameters
    ----------
    pade_table : np.ndarray
        Rows containing ``kappa, Re(zeta), Im(zeta), Im(a1), Im(b1)``.

    Returns
    -------
    np.ndarray
        Real wave-number-dependent closure table.

    Raises
    ------
    ValueError
        If the table is invalid, ``a1`` vanishes, or the real coefficients are nonpositive.
    """
    return np.empty((np.asarray(pade_table).shape[0], 5), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_derive_closure_coefficients(pade_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    table = np.asarray(pade_table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 5 or table.shape[0] == 0:
        raise ValueError("pade_table must have shape (m, 5)")
    if not np.all(np.isfinite(table)):
        raise ValueError("pade_table must be finite")
    alpha = table[:, 3]
    beta = table[:, 4]
    if np.any(np.abs(alpha) <= np.finfo(float).eps):
        raise ValueError("a1 cannot vanish")
    q1 = beta / alpha - 3.0
    q3 = -1.0 / alpha
    result = np.column_stack(
        (table[:, 0], table[:, 1], table[:, 2], q1, q3)
    )
    if not np.all(np.isfinite(result)) or np.any(q1 <= 0.0) or np.any(q3 <= 0.0):
        raise ValueError("the requested branch must yield positive real closure coefficients")
    return result.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {
            "setup": "pade = np.array([[0.27,2.949345526,-0.013194298,-5.837968340,-21.529567529],[0.40,2.271681244,-0.116898820,-1.280632005,-5.576137954],[0.58,1.852900921,-0.293093806,-0.743048670,-3.682696918]])",
            "call": "float(np.sum(derive_closure_coefficients(pade) * np.arange(1, 16).reshape(3, 5)))",
            "gold_call": "float(np.sum(_oracle_derive_closure_coefficients(pade) * np.arange(1, 16).reshape(3, 5)))",
        },
        {
            "setup": "pade = np.array([[0.4, 2.0, -0.1, -2.0, -8.0]])",
            "call": "float(np.sum(derive_closure_coefficients(pade)[0, 3:]))",
            "gold_call": "float(np.sum(_oracle_derive_closure_coefficients(pade)[0, 3:]))",
        },
        {
            "setup": "pade = np.array([[0.4, 2.0, -0.1, 0.0, -8.0]]);\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "status(lambda: derive_closure_coefficients(pade))",
            "gold_call": "status(lambda: _oracle_derive_closure_coefficients(pade))",
        },
    ]
