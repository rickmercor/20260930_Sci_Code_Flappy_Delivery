"""
Construct the occupation-change spectrum.

Inputs are the AO difference density dP and the ordered overlap factors
G, H. All three are shape (n, n) and finite. Misaligned arrays or invalid
numerical values raise ValueError.

Form the G-image of dP. Its eigenvalues are the occupation-change numbers
delta. Map those eigenvectors with H to obtain the AO coefficients V.
Reconstructions use dP = V δ V^T.

Returns
-------
return delta, V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coefficient_pair(
    dP: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return occupation-change numbers and AO coefficients.

    Form the G-image of dP. Its eigenvalues are delta. Map those
    eigenvectors with H to obtain V.

    Parameters
    ----------
    dP : np.ndarray
        AO difference density, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).

    Returns
    -------
    delta : np.ndarray
        Occupation-change numbers, shape (n,).
    V : np.ndarray
        AO coefficients, shape (n, n).

    Raises
    ------
    ValueError
        If dP and the overlap factors are misaligned or if numerical
        values are invalid.
    """
    return delta, V

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coefficient_pair(dP: np.ndarray, G: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if dP.ndim != 2 or dP.shape[0] != dP.shape[1] or dP.shape[0] == 0:
        raise ValueError("dP must be a nonempty square matrix")
    n = dP.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("dP and the overlap factors must have the same shape")
    if not np.all(np.isfinite(dP)):
        raise ValueError("dP must be finite")
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("overlap factors must be finite")
    if np.max(np.abs(dP - dP.T)) > 1e-10:
        raise ValueError("dP must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    dP = 0.5 * (dP + dP.T)
    G = 0.5 * (G + G.T)
    H = 0.5 * (H + H.T)
    dP_tilde = G @ dP @ G
    dP_tilde = 0.5 * (dP_tilde + dP_tilde.T)
    delta, U = np.linalg.eigh(dP_tilde)
    V = H @ U
    return delta.astype(float), V.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
dP = np.diag([1.0, -1.0, 0.2, -0.2])
G = np.eye(4)
H = np.eye(4)

def run_model():
    delta, V = coefficient_pair(dP.copy(), G.copy(), H.copy())
    return np.sort(np.asarray(delta, dtype=float))

def run_gold():
    delta_g, V_g = _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
    return np.sort(np.asarray(delta_g, dtype=float))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
dP = np.diag([1.0, -1.0, 0.2, -0.2])
G = np.eye(4)
H = np.eye(4)

def run_model():
    delta, V = coefficient_pair(dP.copy(), G.copy(), H.copy())
    recon = np.asarray(V) @ np.diag(np.asarray(delta, dtype=float)) @ np.asarray(V).T
    return np.max(np.abs(recon - dP))

def run_gold():
    delta_g, V_g = _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
    recon = V_g @ np.diag(delta_g) @ V_g.T
    return np.max(np.abs(recon - dP))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
dP = np.array([[0.0, 0.3], [0.3, 0.0]])
S = np.array([[1.0, 0.2], [0.2, 1.0]])
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T

def run_model():
    delta, V = coefficient_pair(dP.copy(), G.copy(), H.copy())
    return np.sort(np.asarray(delta, dtype=float))

def run_gold():
    delta_g, _ = _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
    return np.sort(np.asarray(delta_g, dtype=float))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
dP = np.eye(2)
G = np.eye(3)
H = np.eye(2)

def run_model():
    try:
        coefficient_pair(dP.copy(), G.copy(), H.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
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
dP = np.array([[0.0, 0.35, 0.1], [0.35, 0.0, 0.2], [0.1, 0.2, 0.0]], dtype=float)
dP = 0.5 * (dP + dP.T)
S = np.array([[1.0, 0.25, 0.05], [0.25, 1.0, 0.15], [0.05, 0.15, 1.0]], dtype=float)
S = 0.5 * (S + S.T)
w, u = np.linalg.eigh(S)
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T
delta_ao = np.sort(np.linalg.eigvalsh(dP))

def run_model():
    delta, V = coefficient_pair(dP.copy(), G.copy(), H.copy())
    delta = np.sort(np.asarray(delta, dtype=float))
    return np.concatenate([delta, np.array([np.max(np.abs(delta - delta_ao))])])

def run_gold():
    delta_g, _ = _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
    delta_g = np.sort(np.asarray(delta_g, dtype=float))
    return np.concatenate([delta_g, np.array([np.max(np.abs(delta_g - delta_ao))])])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
dP = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=float)
S = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)
w, u = np.linalg.eigh(0.5 * (S + S.T))
G = (u * np.sqrt(w)) @ u.T
H = (u * (1.0 / np.sqrt(w))) @ u.T
delta_wrong, U_wrong = np.linalg.eigh(G @ dP @ G)
V_wrong = G @ U_wrong
wrong_err = np.max(np.abs(V_wrong @ np.diag(delta_wrong) @ V_wrong.T - dP))

def run_model():
    delta, V = coefficient_pair(dP.copy(), G.copy(), H.copy())
    recon = np.asarray(V) @ np.diag(np.asarray(delta, dtype=float)) @ np.asarray(V).T
    return np.array([
        np.max(np.abs(recon - dP)),
        wrong_err,
    ])

def run_gold():
    delta_g, V_g = _oracle_coefficient_pair(dP.copy(), G.copy(), H.copy())
    recon = V_g @ np.diag(delta_g) @ V_g.T
    return np.array([
        np.max(np.abs(recon - dP)),
        wrong_err,
    ])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
