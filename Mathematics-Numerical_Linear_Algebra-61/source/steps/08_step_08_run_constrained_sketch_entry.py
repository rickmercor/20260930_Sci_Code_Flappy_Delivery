"""
Run the constrained sketched iteration by calling the earlier sub-problem functions. Return coordinate coord of the iterate after n_steps. Require q, n_steps, ell >= 1 and 0 <= coord < n.

The orchestrator is the prompt instance. The earlier steps are not decorative: dropping the constraint, the projection, or the truncation window is a different algorithm.

Returns
-------
native Python float: one coordinate of the final iterate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_constrained_sketch_entry(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    indices: np.ndarray,
    q: int,
    n_steps: int,
    ell: int,
    coord: int,
) -> float:
    """Run the constrained sketched iteration; return one coordinate.

    Parameters
    ----------
    m : int
        Row dimension, $m > n \ge 2$.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape $(n,)$.
    data_seed : int
        RNG seed for the consistent system.
    sketch_seed : int
        RNG seed for the Gaussian sketches.
    indices : np.ndarray
        Distinct 0-based constraint rows, length $m_p$ with $1 \le m_p < m$.
    q : int
        Sketch width, $q \ge 1$.
    n_steps : int
        Number of sketched steps, $n_{\mathrm{steps}} \ge 1$.
    ell : int
        Orthogonalization window length, $\ell \ge 1$.
    coord : int
        0-based output coordinate, $0 \le \mathrm{coord} < n$.

    Returns
    -------
    entry : float
        Coordinate $\mathrm{coord}$ of the iterate after $n_{\mathrm{steps}}$.

    Raises
    ------
    ValueError
        If $q$, $n_{\mathrm{steps}}$, or $\ell$ is not an integer $\ge 1$,
        or if $\mathrm{coord}$ is out of range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_blocks(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(v))) for v in p[:3]]
    if min(mp, n, mr) < 1:
        raise ValueError("packed shapes must be positive")
    need = 3 + mp * n + mp + mr * n + mr
    if p.size != need:
        raise ValueError("block pack length does not match header")
    i = 3
    A_p = p[i : i + mp * n].reshape(mp, n)
    i += mp * n
    b_p = p[i : i + mp]
    i += mp
    A_r = p[i : i + mr * n].reshape(mr, n)
    i += mr * n
    b_r = p[i:]
    return A_p, b_p, A_r, b_r


def _unpack_sketches(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("sketch pack is too short")
    n_steps, m_r, q = [int(round(float(v))) for v in p[:3]]
    if min(n_steps, m_r, q) < 1:
        raise ValueError("packed sketch shapes must be positive")
    need = 3 + n_steps * m_r * q
    if p.size != need:
        raise ValueError("sketch pack length does not match header")
    out = []
    i = 3
    for _ in range(n_steps):
        out.append(p[i : i + m_r * q].reshape(m_r, q))
        i += m_r * q
    return out


def _oracle_run_constrained_sketch_entry(
    m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord
):
    for name, val in (("q", q), ("n_steps", n_steps), ("ell", ell), ("coord", coord)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    q, n_steps, ell, coord = int(q), int(n_steps), int(ell), int(coord)
    if min(q, n_steps, ell) < 1:
        raise ValueError("require q, n_steps, ell >= 1")
    n = int(n)
    if coord < 0 or coord >= n:
        raise ValueError("coord out of range")

    sys_pack = _oracle_construct_consistent_system(m, n, s, data_seed)
    block_pack = _oracle_extract_row_blocks(sys_pack, indices)
    A_p, _b_p, A_r, b_r = _unpack_blocks(block_pack)
    m_r = A_r.shape[0]
    x = _oracle_constraint_initial_iterate(block_pack)
    sk_pack = _oracle_draw_gaussian_sketches(m_r, q, n_steps, sketch_seed)
    sketches = _unpack_sketches(sk_pack)
    hist = []
    for k in range(n_steps):
        S = sketches[k]
        d = _oracle_projected_sketched_direction(block_pack, x, S)
        j0 = max(k - ell + 1, 0)
        if k == 0 or j0 >= k:
            window = np.zeros((n, 0))
        else:
            window = np.column_stack(hist[j0:k])
        p = _oracle_window_orthogonalize(d, window)
        x = _oracle_line_search_step(A_r, b_r, x, S, p)
        hist.append(p)
    return float(x[coord])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, q, n_steps, ell, coord = 12, 8, 2, 3, 2, 0
s = np.array([8.0, 6.0, 4.5, 1.1, 0.8, 0.55, 0.4, 0.3])
indices = np.array([2, 8, 3])
data_seed, sketch_seed = 7, 11
""",
            "call": "run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
            "gold_call": "_oracle_run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
        },
        {
            "setup": """import numpy as np
m, n, q, n_steps, ell, coord = 5, 3, 2, 2, 2, 0
s = np.array([3.0, 1.2, 0.4])
indices = np.array([0])
data_seed, sketch_seed = 1, 4
""",
            "call": "run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
            "gold_call": "_oracle_run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
        },
        {
            "setup": """import numpy as np
m, n, q, n_steps, ell, coord = 3, 2, 1, 1, 1, 1
s = np.array([2.0, 0.5])
indices = np.array([1])
data_seed, sketch_seed = 0, 2
""",
            "call": "run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
            "gold_call": "_oracle_run_constrained_sketch_entry(m, n, s, data_seed, sketch_seed, indices, q, n_steps, ell, coord)",
        },
        {
            "setup": """import numpy as np
s = np.array([3.0, 1.2, 0.4])
indices = np.array([0])
def run_model():
    try:
        run_constrained_sketch_entry(5, 3, s, 1, 4, indices, 2, 2, 2, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_constrained_sketch_entry(5, 3, s, 1, 4, indices, 2, 2, 2, 3)
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
s = np.array([3.0, 1.2, 0.4])
indices = np.array([0])
def run_model():
    try:
        run_constrained_sketch_entry(5, 3, s, 1, 4, indices, 2, 0, 2, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_constrained_sketch_entry(5, 3, s, 1, 4, indices, 2, 0, 2, 0)
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
