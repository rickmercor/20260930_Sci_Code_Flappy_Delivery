"""
Recover the two free rational-response parameters from each kinetic root.

Maxwellian parity reduces the root-pair constraint to a real two-parameter solve.

Returns
-------
np.ndarray: float rows [kappa, Re(zeta), Im(zeta), Im(a1), Im(b1)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def match_pade_coefficients(root_table: np.ndarray) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), Im(a1), Im(b1)]``.

    Parameters
    ----------
    root_table : np.ndarray
        Root rows with columns ``kappa, Re(zeta), Im(zeta), residual``.

    Returns
    -------
    np.ndarray
        Real table containing the root and matched Padé coefficients.

    Raises
    ------
    ValueError
        If the table is invalid, a root is on the wrong branch, or matching is singular.
    """
    return np.empty((np.asarray(root_table).shape[0], 5), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_match_pade_coefficients(root_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    roots = np.asarray(root_table, dtype=float)
    if roots.ndim != 2 or roots.shape[1] != 4 or roots.shape[0] == 0:
        raise ValueError("root_table must have shape (m, 4)")
    if not np.all(np.isfinite(roots)):
        raise ValueError("root_table must be finite")
    matched = np.empty((roots.shape[0], 5), dtype=float)
    for row, (kappa, real_part, imag_part, _) in enumerate(roots):
        if kappa <= 0.0 or real_part <= 0.0 or imag_part >= 0.0:
            raise ValueError("roots must lie on the requested kinetic branch")
        zeta = complex(real_part, imag_part)
        a_column = zeta - 2.0 * kappa**2 * zeta**3
        b_column = kappa**2 * zeta
        right_side = -(1.0 + kappa**2 - 2.0 * kappa**2 * zeta**2)
        system = np.array(
            [[(1j * a_column).real, (1j * b_column).real],
             [(1j * a_column).imag, (1j * b_column).imag]],
            dtype=float,
        )
        rhs = np.array([right_side.real, right_side.imag], dtype=float)
        if abs(np.linalg.det(system)) <= 1e-18:
            raise ValueError("root-matching system is singular")
        alpha, beta = np.linalg.solve(system, rhs)
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("matched coefficients must be finite")
        matched[row] = (kappa, real_part, imag_part, alpha, beta)
    return matched

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {
            "setup": "roots = np.array([[0.27,2.949345526068,-0.013194298484,5e-16],[0.40,2.271681243633,-0.116898820041,4e-15],[0.58,1.852900921096,-0.293093806499,8e-16]])",
            "call": "float(np.sum(match_pade_coefficients(roots) * np.arange(1, 16).reshape(3, 5)))",
            "gold_call": "float(np.sum(_oracle_match_pade_coefficients(roots) * np.arange(1, 16).reshape(3, 5)))",
        },
        {
            "setup": "roots = np.array([[0.4, 2.271681243633, -0.116898820041, 1e-15]])",
            "call": "float(match_pade_coefficients(roots)[0, 3])",
            "gold_call": "float(_oracle_match_pade_coefficients(roots)[0, 3])",
        },
        {
            "setup": "roots = np.array([[0.4, -2.0, -0.1, 0.0]]);\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "status(lambda: match_pade_coefficients(roots))",
            "gold_call": "status(lambda: _oracle_match_pade_coefficients(roots))",
        },
    ]
