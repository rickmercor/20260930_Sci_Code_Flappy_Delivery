"""
Compute first- and second-derivative weights on an arbitrary stencil from the exactness conditions of the integrated kernel.

On a stencil of distinct nodes y_1, ..., y_m the weights are fixed by exactness on the integrated kernel: applied to the nodal values of any translate phi2(. - y_k), k = 1, ..., m, of the kernel's second antiderivative, the discrete first- and second-derivative operators must return that translate's exact first and second derivatives at the evaluation point. No polynomial terms are appended. This is the construction the scheme uses on the one-sided seven-node stencils next to each boundary.

Returns
-------
np.ndarray, float, shape (2, m): row 0 the first-derivative weights and row 1 the second-derivative weights, ordered like the input nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def integrated_kernel_weights(nodes: np.ndarray, centre: float, c: float) -> np.ndarray:
    '''Derivative weights from the integrated-kernel exactness system.

    Parameters
    ----------
    nodes : np.ndarray
        One-dimensional array of m >= 2 distinct, finite stencil nodes.
    centre : float
        Point at which the derivatives are approximated (finite).
    c : float
        Shape parameter of the kernel, c > 0.

    Returns
    -------
    weights : np.ndarray
        Shape (2, m) float array: row 0 first-derivative weights, row 1
        second-derivative weights, in the order of ``nodes``.

    Raises
    ------
    ValueError
        If nodes is not a finite one-dimensional array of at least two
        distinct values, if centre is not finite, or if c is not a finite
        positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.asarray(nodes).size), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_integrated_kernel_weights(nodes: np.ndarray, centre: float, c: float) -> np.ndarray:
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("nodes must be a one-dimensional array with at least two entries")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("nodes must be distinct")
    if isinstance(centre, bool) or not np.isfinite(float(centre)):
        raise ValueError("centre must be finite")
    if isinstance(c, bool) or not np.isfinite(float(c)) or float(c) <= 0.0:
        raise ValueError("c must be a finite positive number")
    centre = float(centre)
    c = float(c)

    m = nodes.size
    # Interpolation matrix built from the second antiderivative (step 01).
    offsets = (nodes[None, :] - nodes[:, None]).ravel()
    matrix = _oracle_integrated_kernels(offsets, c)[:, 2].reshape(m, m)
    # Right-hand sides: the first and second derivatives of each translate
    # phi2(. - y_k) at the evaluation point are phi1 and phi.
    rhs = _oracle_integrated_kernels(centre - nodes, c)
    w1 = np.linalg.solve(matrix, rhs[:, 1])
    w2 = np.linalg.solve(matrix, rhs[:, 0])
    return np.vstack([w1, w2]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: one-sided seven-node stencil at the left boundary, the
        # configuration of the second grid row (unit spacing, c = 4h).
        {
            "setup": """import numpy as np
nodes = np.arange(7, dtype=float)
""",
            "call": "integrated_kernel_weights(nodes, 1.0, 4.0)",
            "gold_call": "_oracle_integrated_kernel_weights(nodes, 1.0, 4.0)",
        },
        # --- Normal: the third grid row, still one-sided, on the pricing grid
        # spacing h = 5 with c = 20.
        {
            "setup": """import numpy as np
nodes = 5.0 * np.arange(7, dtype=float)
""",
            "call": "integrated_kernel_weights(nodes, 10.0, 20.0)",
            "gold_call": "_oracle_integrated_kernel_weights(nodes, 10.0, 20.0)",
        },
        # --- Pinned structure: on a symmetric stencil the first-derivative
        # weights are antisymmetric and the second-derivative weights symmetric,
        # so this digest vanishes for any correct implementation. The rounding
        # resolves 1e-9 in the weights: far above linear-solver round-off
        # (about 1e-12 at this conditioning, and it varies between LAPACK
        # builds) and far below any genuine defect.
        {
            "setup": """import numpy as np
nodes = np.arange(-3.0, 4.0)
def symmetry_digest(fn):
    w = fn(nodes, 0.0, 4.0)
    return np.round(np.concatenate([w[0] + w[0][::-1], w[1] - w[1][::-1]]) * 1e6, 3) + 0.0
EXPECTED = np.zeros(14)
""",
            "call": "symmetry_digest(integrated_kernel_weights)",
            "gold_call": "EXPECTED",
        },
        # --- Boundary: right-hand one-sided stencil evaluated at its last node.
        {
            "setup": """import numpy as np
nodes = 270.0 + 5.0 * np.arange(7, dtype=float)
""",
            "call": "integrated_kernel_weights(nodes, 300.0, 20.0)",
            "gold_call": "_oracle_integrated_kernel_weights(nodes, 300.0, 20.0)",
        },
        # --- Edge: smallest stencil on non-uniform nodes with a wide kernel.
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 0.4, 1.3])
""",
            "call": "integrated_kernel_weights(nodes, 0.4, 2.5)",
            "gold_call": "_oracle_integrated_kernel_weights(nodes, 0.4, 2.5)",
        },
        # --- Invalid: repeated node ---
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 1.0, 1.0, 2.0])
def run_model():
    try:
        integrated_kernel_weights(nodes, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrated_kernel_weights(nodes, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative shape parameter ---
        {
            "setup": """import numpy as np
nodes = np.arange(7, dtype=float)
def run_model():
    try:
        integrated_kernel_weights(nodes, 1.0, -4.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrated_kernel_weights(nodes, 1.0, -4.0)
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
