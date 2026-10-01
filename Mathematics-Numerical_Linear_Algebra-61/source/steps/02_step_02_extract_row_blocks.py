"""
Split a packed (A, b) into constraint rows A_p, b_p and complementary rows A_r, b_r for a 0-based index set I. Require 1 <= mp < m and distinct indices.

The constraint rows are treated as given. Selecting them once, with 0-based indices, makes the affine set A_p x = b_p and the complementary residual deterministic.

Returns
-------
1d ndarray: packed (mp, n, mr, A_p, b_p, A_r, b_r)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def extract_row_blocks(pack: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Pack A_p, b_p, A_r, b_r for 0-based constraint row indices.

    Parameters
    ----------
    pack : np.ndarray
        Packed $(A, b)$ from construct_consistent_system.
    indices : np.ndarray
        Distinct 0-based constraint rows, length $m_p$ with $1 \le m_p < m$.

    Returns
    -------
    block_pack : np.ndarray
        Concatenation of $(m_p, n, m_r)$, $A_p.\mathrm{ravel}()$, $b_p$,
        $A_r.\mathrm{ravel}()$, $b_r$.

    Raises
    ------
    ValueError
        If pack is too short or its length does not match the header, if
        packed shapes are not positive, if indices are not distinct, if
        not $1 \le m_p < m$, or if any index is out of range.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_system(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 2:
        raise ValueError("pack is too short")
    m, n = int(round(float(pack[0]))), int(round(float(pack[1])))
    if m < 1 or n < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + m * n + m
    if pack.size != need:
        raise ValueError("pack length does not match header")
    A = pack[2 : 2 + m * n].reshape(m, n)
    b = pack[2 + m * n :]
    return A, b

def _oracle_extract_row_blocks(pack, indices):
    A, b = _unpack_system(pack)
    m, n = A.shape
    idx = np.asarray(indices, dtype=int).reshape(-1)
    mp = idx.size
    if mp < 1 or mp >= m:
        raise ValueError("require 1 <= mp < m")
    if np.unique(idx).size != mp:
        raise ValueError("indices must be distinct")
    if np.any(idx < 0) or np.any(idx >= m):
        raise ValueError("indices out of range")
    mask = np.ones(m, dtype=bool)
    mask[idx] = False
    Ir = np.where(mask)[0]
    A_p, b_p = A[idx], b[idx]
    A_r, b_r = A[Ir], b[Ir]
    mr = A_r.shape[0]
    return np.concatenate(
        ([float(mp), float(n), float(mr)], A_p.ravel(), b_p, A_r.ravel(), b_r)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, seed = 12, 8, 7
s = np.array([8.0, 6.0, 4.5, 1.1, 0.8, 0.55, 0.4, 0.3])
rng = np.random.default_rng(seed)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
A = U @ np.diag(s) @ V.T
b = A @ rng.standard_normal(n)
pack = np.concatenate(([float(m), float(n)], A.ravel(), b))
indices = np.array([2, 8, 3])
""",
            "call": "extract_row_blocks(pack, indices)",
            "gold_call": "_oracle_extract_row_blocks(pack, indices)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 5, 3, 1
s = np.array([3.0, 1.2, 0.4])
rng = np.random.default_rng(seed)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
A = U @ np.diag(s) @ V.T
b = A @ rng.standard_normal(n)
pack = np.concatenate(([float(m), float(n)], A.ravel(), b))
indices = np.array([0])
""",
            "call": "extract_row_blocks(pack, indices)",
            "gold_call": "_oracle_extract_row_blocks(pack, indices)",
        },
        {
            "setup": """import numpy as np
m, n, seed = 3, 2, 0
s = np.array([2.0, 0.5])
rng = np.random.default_rng(seed)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
A = U @ np.diag(s) @ V.T
b = A @ rng.standard_normal(n)
pack = np.concatenate(([float(m), float(n)], A.ravel(), b))
indices = np.array([1])
""",
            "call": "extract_row_blocks(pack, indices)",
            "gold_call": "_oracle_extract_row_blocks(pack, indices)",
        },
        {
            "setup": """import numpy as np
pack = np.concatenate(([5.0, 3.0], np.arange(15.0), np.arange(5.0)))
indices = np.array([0, 0])
def run_model():
    try:
        extract_row_blocks(pack, indices)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_row_blocks(pack, indices)
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
pack = np.concatenate(([5.0, 3.0], np.arange(15.0), np.arange(5.0)))
indices = np.array([0, 1, 2, 3, 4])
def run_model():
    try:
        extract_row_blocks(pack, indices)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_row_blocks(pack, indices)
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
