"""
Construct the symmetric retained-magnitude kernel and its row-normalized Hamiltonian proposal from a finite real symmetric Hamiltonian, using an inclusive off-diagonal magnitude cutoff and requiring every state to retain at least one reversible neighbor.

Hamiltonian-guided Fock-space moves use a symmetric retained graph. The retained magnitude matrix preserves the connection strengths, and row normalization produces a state-dependent categorical proposal. Hermiticity and symmetric thresholding guarantee reverse support, while the diagonal represents no configuration move.

Returns
-------
tuple[np.ndarray, np.ndarray], containing the retained magnitude matrix W and row-stochastic proposal Q, each with shape (n_states, n_states) in binary64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_retained_kernel(H: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the retained Hamiltonian magnitudes and proposal matrix.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian with shape ``(n_states, n_states)``.
    cutoff : float
        Strictly positive finite threshold. A nonzero off-diagonal entry is retained when
        ``abs(H[x, y]) >= cutoff``.

    Returns
    -------
    retained_magnitudes : np.ndarray
        Symmetric nonnegative matrix ``W`` of retained off-diagonal magnitudes.
    proposal : np.ndarray
        Row-stochastic Hamiltonian proposal ``Q``.

    Raises
    ------
    ValueError
        If the inputs are invalid, the Hamiltonian is not symmetric, or a state has no
        retained reversible neighbor.
    """
    return np.empty_like(np.asarray(H), dtype=float), np.empty_like(np.asarray(H), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_retained_kernel(H: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    matrix = np.asarray(H, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("H must be square with at least two states")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("H must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("H must be symmetric within atol=1e-12")
    try:
        threshold = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be finite and strictly positive") from exc
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("cutoff must be finite and strictly positive")

    magnitudes = np.abs(matrix)
    retained = (magnitudes >= threshold) & (~np.eye(matrix.shape[0], dtype=bool))
    W = np.where(retained, magnitudes, 0.0)
    row_mass = np.sum(W, axis=1, dtype=float)
    if np.any(row_mass <= 0.0):
        raise ValueError("every state must have at least one retained neighbor")
    Q = W / row_mass[:, None]
    return W.astype(float), Q.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    return [
        {
            "setup": """import numpy as np
H = np.array([[-1.0,-0.8,0.35,0.0],[-0.8,0.2,-0.2,0.4],[0.35,-0.2,0.7,-0.5],[0.0,0.4,-0.5,-0.1]], dtype=float)
cutoff = 0.2
""",
            "call": "np.concatenate(build_retained_kernel(H, cutoff)).ravel()",
            "gold_call": "np.concatenate(_oracle_build_retained_kernel(H, cutoff)).ravel()",
        },
        {
            "setup": """import numpy as np
H = np.array([[4.0,0.4,0.0],[0.4,-2.0,-0.4],[0.0,-0.4,7.0]], dtype=float)
cutoff = 0.4
""",
            "call": "np.concatenate(build_retained_kernel(H, cutoff)).ravel()",
            "gold_call": "np.concatenate(_oracle_build_retained_kernel(H, cutoff)).ravel()",
        },
        {
            "setup": """import numpy as np
H = np.array([[9.0,-2.0],[-2.0,-11.0]], dtype=float)
cutoff = 0.1
""",
            "call": "np.concatenate(build_retained_kernel(H, cutoff)).ravel()",
            "gold_call": "np.concatenate(_oracle_build_retained_kernel(H, cutoff)).ravel()",
        },
        {
            "setup": """import numpy as np
H = np.array([[0.0,0.5],[0.4,0.0]], dtype=float)
def run_model():
    try:
        build_retained_kernel(H, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_retained_kernel(H, 0.2)
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
H = np.array([[0.0,0.8,0.0],[0.8,0.0,0.0],[0.0,0.0,0.0]], dtype=float)
def run_model():
    try:
        build_retained_kernel(H, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_retained_kernel(H, 0.2)
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
H = np.array([[0.0,1.0],[1.0,np.nan]], dtype=float)
def run_model():
    try:
        build_retained_kernel(H, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_retained_kernel(H, 0.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
