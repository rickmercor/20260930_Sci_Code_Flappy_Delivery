"""
Assemble the global seven-band first- and second-derivative matrices of the integrated-kernel scheme on a uniform asset grid.

The asset grid is S_i = i h, i = 0, ..., N - 1, with h = s_max / (N - 1), and the shape parameter is c = c_over_h * h. Rows 0 and N - 1 carry the Dirichlet data and are left empty here. Every interior row i with 3 <= i <= N - 4 uses the closed-form centred weights on the nodes i - 3, ..., i + 3. The rows next to each boundary, i = 1, 2 and i = N - 3, N - 2, cannot be centred; they use the first or last seven nodes (0, ..., 6 or N - 7, ..., N - 1) with weights from the integrated-kernel exactness system evaluated at S_i. Every non-empty row therefore has exactly seven non-zeros.

Returns
-------
np.ndarray, float, shape (2, N, N): index 0 the first-derivative matrix and index 1 the second-derivative matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_differentiation_matrices(n_nodes: int, s_max: float, c_over_h: float) -> np.ndarray:
    '''Global differentiation matrices of the seven-node scheme.

    Parameters
    ----------
    n_nodes : int
        Number of grid nodes N, including both end points (N >= 7).
    s_max : float
        Right end of the asset grid [0, s_max], s_max > 0.
    c_over_h : float
        Ratio of the kernel shape parameter to the grid spacing, > 0.

    Returns
    -------
    matrices : np.ndarray
        Shape (2, N, N) float array: [0] first-derivative matrix, [1]
        second-derivative matrix. Rows 0 and N - 1 are zero.

    Raises
    ------
    ValueError
        If n_nodes is not an integer of at least 7, or if s_max or c_over_h is
        not a finite positive number.

    Notes
    -----
    Build the rows by calling ``analytic_interior_weights`` and
    ``integrated_kernel_weights`` rather than re-deriving them. Include every
    import your implementation needs inside the function body.
    '''
    return np.zeros((2, int(n_nodes), int(n_nodes)), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_assemble_differentiation_matrices(n_nodes: int, s_max: float, c_over_h: float) -> np.ndarray:
    import numpy as np

    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 7:
        raise ValueError("n_nodes must be an integer >= 7")
    for name, value in (("s_max", s_max), ("c_over_h", c_over_h)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    n = int(n_nodes)
    grid = np.linspace(0.0, float(s_max), n)
    h = grid[1] - grid[0]
    c = float(c_over_h) * h

    mats = np.zeros((2, n, n))
    centred = _oracle_analytic_interior_weights(h, c)
    for i in range(1, n - 1):
        if 3 <= i <= n - 4:
            idx = np.arange(i - 3, i + 4)
            w = centred
        else:
            idx = np.arange(0, 7) if i < 3 else np.arange(n - 7, n)
            w = _oracle_integrated_kernel_weights(grid[idx], grid[i], c)
        mats[0, i, idx] = w[0]
        mats[1, i, idx] = w[1]
    return mats

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the pricing grid of the task, N = 61 on [0, 300], c = 4h.
        {
            "setup": """import numpy as np
""",
            "call": "assemble_differentiation_matrices(61, 300.0, 4.0)",
            "gold_call": "_oracle_assemble_differentiation_matrices(61, 300.0, 4.0)",
        },
        # --- Pinned: on centred rows the closed forms differentiate 1, S and S^2
        # exactly, so D1 @ 1 = 0, D1 @ S = 1 and D2 @ S^2 = 2 there, and the
        # empty Dirichlet row 0 maps everything to 0, independently of any
        # reference implementation.
        {
            "setup": """import numpy as np
S = np.linspace(0.0, 300.0, 61)
one = np.ones(61)
def exactness_digest(fn):
    # Each part is rounded at a resolution its round-off cannot reach: the
    # sums D1 @ S and D2 @ S^2 cancel terms of size up to 1e5, so they are
    # compared to 5 decimals; D1 @ 1 cancels only O(1) terms and the Dirichlet
    # row is exactly zero, so they are scaled by 1e6 and kept to 3 decimals.
    m = fn(61, 300.0, 4.0)
    rows = np.arange(3, 58)
    keep = np.concatenate([[0], rows])
    return np.concatenate([np.round(1e6 * (m[0] @ one)[keep], 3),
                           np.round((m[0] @ S)[rows], 5),
                           np.round((m[1] @ S ** 2)[rows], 5)]) + 0.0
EXPECTED = np.concatenate([np.zeros(56), np.ones(55), 2.0 * np.ones(55)])
""",
            "call": "exactness_digest(assemble_differentiation_matrices)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned structure: empty Dirichlet rows, seven non-zeros in every
        # second-derivative row and in the one-sided first-derivative rows, six
        # in the centred first-derivative rows (whose centre weight is exactly
        # zero), and one-sided supports on the first and last seven nodes.
        {
            "setup": """import numpy as np
def pattern_digest(fn):
    m = fn(15, 1.4, 4.0)
    nz1 = (np.abs(m[0]) > 0).sum(axis=1).astype(float)
    nz2 = (np.abs(m[1]) > 0).sum(axis=1).astype(float)
    first = np.flatnonzero(m[1][2]).astype(float)
    last = np.flatnonzero(m[1][12]).astype(float)
    return np.concatenate([nz1, nz2, first, last])
EXPECTED = np.concatenate([[0.0, 7.0, 7.0] + [6.0] * 9 + [7.0, 7.0, 0.0],
                           [0.0] + [7.0] * 13 + [0.0], np.arange(0.0, 7.0), np.arange(8.0, 15.0)])
""",
            "call": "pattern_digest(assemble_differentiation_matrices)",
            "gold_call": "EXPECTED",
        },
        # --- Boundary: the smallest admissible grid, where only one row is
        # centred and four rows are one-sided.
        {
            "setup": """import numpy as np
""",
            "call": "assemble_differentiation_matrices(7, 3.0, 4.0)",
            "gold_call": "_oracle_assemble_differentiation_matrices(7, 3.0, 4.0)",
        },
        # --- Edge: a flat kernel relative to the spacing.
        {
            "setup": """import numpy as np
""",
            "call": "assemble_differentiation_matrices(21, 50.0, 12.0)",
            "gold_call": "_oracle_assemble_differentiation_matrices(21, 50.0, 12.0)",
        },
        # --- Invalid: too few nodes for a seven-node stencil ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_differentiation_matrices(6, 1.0, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_differentiation_matrices(6, 1.0, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive domain length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_differentiation_matrices(11, -1.0, 4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_differentiation_matrices(11, -1.0, 4.0)
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
