"""
Compute OVOCV electron-promotion numbers for the spectator relaxation.

The source paper decomposes the change from the frozen intermediate
to the final ROKS state into occupied-virtual pairs ranked by
significance. Recover the promotion numbers of those pairs from the
paper. Do not replace them by NOCV eigenvalues of the raw difference
density, and do not mix the primary open-shell promotion into this
stage.

Returns
-------
return dQ
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ovocv_relaxation_promotions(
    P_INT: np.ndarray,
    P_ES: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Return OVOCV promotion numbers of the spectator relaxation.

    Parameters
    ----------
    P_INT : np.ndarray
        Intermediate frozen-state AO spatial density, shape (n, n).
    P_ES : np.ndarray
        Final-state AO spatial density, shape (n, n).
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    np.ndarray
        Promotion numbers of the spectator OVOCV pairs, sorted in
        descending order.

    Raises
    ------
    ValueError
        If the densities are misaligned, are not valid ROKS spatial
        densities, or are non-finite.
    """
    return dQ

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ovocv_relaxation_promotions(P_INT: np.ndarray, P_ES: np.ndarray, S: np.ndarray) -> np.ndarray:
    P_INT = np.asarray(P_INT, dtype=float)
    P_ES = np.asarray(P_ES, dtype=float)
    S = np.asarray(S, dtype=float)
    if P_INT.ndim != 2 or P_INT.shape[0] != P_INT.shape[1] or P_INT.shape[0] == 0:
        raise ValueError("P_INT must be a nonempty square matrix")
    n = P_INT.shape[0]
    if P_ES.shape != (n, n) or S.shape != (n, n):
        raise ValueError("P_INT, P_ES, and S must have the same shape")
    if not np.all(np.isfinite(P_INT)) or not np.all(np.isfinite(P_ES)):
        raise ValueError("densities must be finite")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_INT = 0.5 * (P_INT + P_INT.T)
    P_ES = 0.5 * (P_ES + P_ES.T)
    S = 0.5 * (S + S.T)
    w_s, u_s = np.linalg.eigh(S)
    if np.any(w_s <= 1e-12):
        raise ValueError("S must be positive definite")
    sh = (u_s * np.sqrt(w_s)) @ u_s.T
    sih = (u_s * (1.0 / np.sqrt(w_s))) @ u_s.T

    def _doubles(P):
        a = 0.5 * ((sh @ P @ sh) + (sh @ P @ sh).T)
        w, y = np.linalg.eigh(a)
        order = np.argsort(w)[::-1]
        w = w[order]
        C = sih @ y[:, order]
        mask = np.abs(w - 2.0) <= 1e-6
        if np.count_nonzero(mask) == 0:
            raise ValueError("no occupation-2 spectator orbitals were found")
        return C[:, mask]

    C_i = _doubles(P_INT)
    C_f = _doubles(P_ES)
    if C_i.shape[1] != C_f.shape[1]:
        raise ValueError("intermediate and final spectator ranks must match")
    s = np.linalg.svd(C_i.T @ S @ C_f, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    dQ = 1.0 - s**2
    return np.sort(dQ)[::-1].astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np
th = 0.4
C_d_i = np.eye(5)[:, :2]
C_d_f = np.array([
    [1.0, 0.0],
    [0.0, np.cos(th)],
    [0.0, np.sin(th)],
    [0.0, 0.0],
    [0.0, 0.0],
])
h = np.eye(5)[:, 3]
l = np.eye(5)[:, 4]
P_INT = 2.0 * (C_d_i @ C_d_i.T) + np.outer(h, h) + np.outer(l, l)
P_ES = 2.0 * (C_d_f @ C_d_f.T) + np.outer(h, h) + np.outer(l, l)
S = np.eye(5)

def run_model():
    return ovocv_relaxation_promotions(P_INT, P_ES, S)

def run_gold():
    return _oracle_ovocv_relaxation_promotions(P_INT, P_ES, S)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
th = 0.4
C_d_i = np.eye(5)[:, :2]
C_d_f = np.array([
    [1.0, 0.0],
    [0.0, np.cos(th)],
    [0.0, np.sin(th)],
    [0.0, 0.0],
    [0.0, 0.0],
])
h = np.eye(5)[:, 3]
l = np.eye(5)[:, 4]
P_INT = 2.0 * (C_d_i @ C_d_i.T) + np.outer(h, h) + np.outer(l, l)
P_ES = 2.0 * (C_d_f @ C_d_f.T) + np.outer(h, h) + np.outer(l, l)
S = np.eye(5)
dQ_g = _oracle_ovocv_relaxation_promotions(P_INT, P_ES, S)

def run_model():
    dQ = np.asarray(ovocv_relaxation_promotions(P_INT, P_ES, S))
    return np.array([np.sum(dQ), dQ[0]])

def run_gold():
    return np.array([np.sum(dQ_g), dQ_g[0]])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
P_INT = np.eye(3)
P_ES = np.eye(4)
S = np.eye(3)

def run_model():
    try:
        ovocv_relaxation_promotions(P_INT, P_ES, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_ovocv_relaxation_promotions(P_INT, P_ES, S)
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
P_INT = np.eye(3).astype(float)
P_INT[0, 0] = np.nan
P_ES = np.eye(3)
S = np.eye(3)

def run_model():
    try:
        ovocv_relaxation_promotions(P_INT, P_ES, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_ovocv_relaxation_promotions(P_INT, P_ES, S)
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
