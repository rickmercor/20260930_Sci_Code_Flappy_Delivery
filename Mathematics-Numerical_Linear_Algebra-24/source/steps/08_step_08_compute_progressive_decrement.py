"""
Run the multi-iteration pipeline and report its final scalar.



Each iteration builds the selectors, scores the elements from that iteration's

residual, determines the accepted state starting from the threshold currently

in force, and solves the accepted system. The threshold is state that persists

across iterations rather than being restarted at each one; exactly what a

later iteration inherits from its predecessor's accepted threshold is fixed by

the established convention and is not restated here.



The returned scalar is the squared Newton decrement of the final iteration,

``lambda_squared = -g.T @ delta_x``.

Returns
-------
one finite float equal to -residuals[-1].T @ delta_x of the final iteration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_progressive_decrement(
    regularizer: np.ndarray,
    residuals: np.ndarray,
    index_pairs: np.ndarray,
    element_hessians: np.ndarray,
    eigenvalue_floor: float = 1e-8,
    fallback_threshold: float = 1e-12,
) -> float:
    """Compute the squared decrement of the final warm-started iteration.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is a nonempty square two-dimensional array that equals its
    transpose to within ``1e-12`` absolute; ``residuals`` has shape
    ``(n_iterations, n_dof)`` with at least one iteration; ``index_pairs`` is
    an integer array of shape ``(n_elements, 2)`` with at least one element,
    every index in ``[0, n_dof)``, and the two indices of each element
    distinct; ``element_hessians`` has shape
    ``(n_iterations, n_elements, 2, 2)`` with iteration and element counts
    matching the two preceding arguments, and is symmetric along its last two
    axes to the same tolerance; every numeric entry is finite; and
    ``eigenvalue_floor`` and ``fallback_threshold`` are both finite and
    strictly positive. ``ValueError`` is also raised if any iteration exhausts
    every element without reaching a positive-definite assembly, or if the
    resulting decrement is not finite. The three symmetry and index conditions
    are verified rather than assumed.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    residuals : np.ndarray
        Finite assembled residuals of shape ``(n_iterations, n_dof)``, in
        Newton-iteration order.
    index_pairs : np.ndarray
        Ordered zero-based element pairs of shape ``(n_elements, 2)``.
    element_hessians : np.ndarray
        Re-evaluated symmetric local Hessians of shape
        ``(n_iterations, n_elements, 2, 2)``.
    eigenvalue_floor : float, optional
        Positive local spectral floor.
    fallback_threshold : float, optional
        Positive convention constant governing when the round-based selection
        is abandoned in favour of admitting everything that remains.

    Returns
    -------
    float
        Finite squared Newton decrement of the last iteration.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
# Convention constants supplied by the source, withheld from the public contract.
_CONTRACTION = 0.5
_RELEASE = 2.0


def _validated_selectors(index_pairs: np.ndarray, n_dof: int) -> np.ndarray:
    pairs = np.asarray(index_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
        raise ValueError("index_pairs must have shape (n_elements, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("index_pairs must contain integers")
    if (
        np.any(pairs < 0)
        or np.any(pairs >= n_dof)
        or np.any(pairs[:, 0] == pairs[:, 1])
    ):
        raise ValueError("index_pairs contain invalid indices")
    selectors = np.zeros((pairs.shape[0], 2, n_dof), dtype=float)
    for element, pair in enumerate(pairs.astype(int)):
        selectors[element, np.arange(2), pair] = 1.0
    return selectors


def _project_local(hessian: np.ndarray, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(hessian)
    projected = (vectors * np.maximum(values, floor)) @ vectors.T
    return 0.5 * (projected + projected.T)


def _assemble(
    regularizer: np.ndarray, hessians: np.ndarray, selectors: np.ndarray
) -> np.ndarray:
    result = regularizer.copy()
    for hessian, selector in zip(hessians, selectors):
        result += selector.T @ hessian @ selector
    return 0.5 * (result + result.T)


def _accepted_state(
    regularizer: np.ndarray,
    residual: np.ndarray,
    hessians: np.ndarray,
    selectors: np.ndarray,
    scores: np.ndarray,
    delta_init: float,
    alpha: float,
    floor: float,
    fallback: float,
) -> np.ndarray:
    accepted = _assemble(regularizer, hessians, selectors)
    processed = np.zeros(hessians.shape[0], dtype=bool)
    try:
        np.linalg.cholesky(accepted)
        return np.concatenate(([float(delta_init)], processed.astype(float)))
    except np.linalg.LinAlgError:
        pass
    delta = float(delta_init)
    if not np.isfinite(delta):
        delta = alpha * float(np.linalg.norm(residual, ord=np.inf))
    projected = np.stack([_project_local(h, floor) for h in hessians])
    while True:
        chosen = (~processed) if delta <= fallback else (~processed) & (scores > delta)
        if not np.any(chosen):
            delta *= alpha
            continue
        for element in np.flatnonzero(chosen):
            selector = selectors[element]
            accepted += selector.T @ (projected[element] - hessians[element]) @ selector
        accepted = 0.5 * (accepted + accepted.T)
        processed |= chosen
        try:
            np.linalg.cholesky(accepted)
            return np.concatenate(([delta], processed.astype(float)))
        except np.linalg.LinAlgError:
            if np.all(processed):
                raise ValueError(
                    "fully projected global Hessian is not positive definite"
                )
            delta *= alpha


def _oracle_compute_progressive_decrement(regularizer, residuals, index_pairs, element_hessians, eigenvalue_floor=1e-8, fallback_threshold=1e-12):
    """Reference end-to-end multi-iteration computation."""
    
    alpha = 0.5  # source contraction factor, withheld from the public contract
    beta = 2.0  # source release factor, withheld from the public contract
    regularizer = np.asarray(regularizer, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    hessians = np.asarray(element_hessians, dtype=float)
    if (
        regularizer.ndim != 2
        or regularizer.shape[0] != regularizer.shape[1]
        or regularizer.shape[0] == 0
    ):
        raise ValueError("regularizer must be a nonempty square matrix")
    n_dof = regularizer.shape[0]
    if residuals.ndim != 2 or residuals.shape[1] != n_dof or residuals.shape[0] < 1:
        raise ValueError("residuals must have shape (n_iterations, n_dof)")
    if hessians.ndim != 4 or hessians.shape[2:] != (2, 2):
        raise ValueError(
            "element_hessians must have shape (n_iterations, n_elements, 2, 2)"
        )
    if hessians.shape[0] != residuals.shape[0]:
        raise ValueError("iteration counts must agree")
    if hessians.shape[1] != np.asarray(index_pairs).shape[0]:
        raise ValueError("element counts must agree")
    if (
        not np.all(np.isfinite(regularizer))
        or not np.all(np.isfinite(residuals))
        or not np.all(np.isfinite(hessians))
    ):
        raise ValueError("numeric inputs must be finite")
    if not np.allclose(regularizer, regularizer.T, atol=1e-12, rtol=0.0):
        raise ValueError("regularizer must be symmetric")
    if not np.allclose(hessians, np.swapaxes(hessians, 2, 3), atol=1e-12, rtol=0.0):
        raise ValueError("element Hessians must be symmetric")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    if not np.isfinite(fallback_threshold) or fallback_threshold <= 0.0:
        raise ValueError("fallback_threshold must be positive and finite")

    selectors = _oracle_build_selection_matrices(index_pairs, n_dof)

    delta = float("inf")
    decrement = 0.0
    for iteration in range(residuals.shape[0]):
        gradient = residuals[iteration]
        local = hessians[iteration]
        scores = _oracle_compute_local_residual_scores(selectors, gradient)
        state = _oracle_find_accepted_state(
            regularizer,
            gradient,
            local,
            selectors,
            scores,
            delta,
            float(eigenvalue_floor),
            float(fallback_threshold),
        )
        delta = float(state[0])
        mask = np.asarray(state[1:]) > 0.5
        effective = np.stack(
            [
                _oracle_project_element_hessian(local[e], float(eigenvalue_floor))
                if mask[e]
                else local[e]
                for e in range(local.shape[0])
            ]
        )
        accepted = _oracle_assemble_global_hessian(regularizer, effective, selectors)
        step = _oracle_solve_newton_step(accepted, gradient)
        decrement = float(-gradient @ step)
        delta *= float(beta)

    if not np.isfinite(decrement):
        raise ValueError("the squared decrement is not finite")
    return decrement

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return four whole-pipeline integrations and one invalid case."""
    return [
        {
            "setup": """import numpy as np
M = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
P = np.array([-3.5, -1.2, -4.5, -0.8, 1.9, -0.45, 1.6, 1.85, -0.3, 1.6, 1.9, -0.7])
Q = np.array([0.45, 0.7, 0.3, 0.55, 0.3, 0.45, 0.3, 0.45, 0.3, 0.55, 0.3, 0.45])
R = np.array([1.4, 1.15, 1.9, 1.2, 1.9, 0.95, 1.65, 1.9, 0.95, 1.65, 1.9, 1.4])
H1 = np.stack([np.array([[P[e], Q[e]], [Q[e], R[e]]]) for e in range(12)])
s = np.array([0.0, 2.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H2 = H1 + s[:, None, None] * np.eye(2)
Hs = np.stack([H1, H2])
g = np.array([
    [24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875],
    [36.0, -18.0, 9.0, -5.0, 2.25, -1.125, 0.5625, -0.5625, 0.28125, -0.28125],
])
""",
            "call": "compute_progressive_decrement(M, g, pairs, Hs)",
            "gold_call": "_oracle_compute_progressive_decrement(M, g, pairs, Hs)",
        },
        {
            "setup": """import numpy as np
M = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
P = np.array([-3.5, -1.2, -4.5, -0.8, 1.9, -0.45, 1.6, 1.85, -0.3, 1.6, 1.9, -0.7])
Q = np.array([0.45, 0.7, 0.3, 0.55, 0.3, 0.45, 0.3, 0.45, 0.3, 0.55, 0.3, 0.45])
R = np.array([1.4, 1.15, 1.9, 1.2, 1.9, 0.95, 1.65, 1.9, 0.95, 1.65, 1.9, 1.4])
H1 = np.stack([np.array([[P[e], Q[e]], [Q[e], R[e]]]) for e in range(12)])
s = np.array([0.0, 2.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H2 = H1 + s[:, None, None] * np.eye(2)
Hs = np.stack([H1, H2])
g = np.array([
    [24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875],
    [36.0, -18.0, 9.0, -5.0, 2.25, -1.125, 0.5625, -0.5625, 0.28125, -0.28125],
])
g = g[:1]
Hs = Hs[:1]
""",
            "call": "compute_progressive_decrement(M, g, pairs, Hs)",
            "gold_call": "_oracle_compute_progressive_decrement(M, g, pairs, Hs)",
        },
        {
            "setup": """import numpy as np
M = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
P = np.array([-3.5, -1.2, -4.5, -0.8, 1.9, -0.45, 1.6, 1.85, -0.3, 1.6, 1.9, -0.7])
Q = np.array([0.45, 0.7, 0.3, 0.55, 0.3, 0.45, 0.3, 0.45, 0.3, 0.55, 0.3, 0.45])
R = np.array([1.4, 1.15, 1.9, 1.2, 1.9, 0.95, 1.65, 1.9, 0.95, 1.65, 1.9, 1.4])
H1 = np.stack([np.array([[P[e], Q[e]], [Q[e], R[e]]]) for e in range(12)])
s = np.array([0.0, 2.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H2 = H1 + s[:, None, None] * np.eye(2)
Hs = np.stack([H1, H2])
g = np.array([
    [24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875],
    [36.0, -18.0, 9.0, -5.0, 2.25, -1.125, 0.5625, -0.5625, 0.28125, -0.28125],
])
Hs = np.stack([H1, H2, H1])
g = np.vstack([g,
    np.array([20.0, -40.0, 10.0, -5.0, 2.5, -1.25, 0.625, -0.625, 0.3125, -0.3125])])
""",
            "call": "compute_progressive_decrement(M, g, pairs, Hs)",
            "gold_call": "_oracle_compute_progressive_decrement(M, g, pairs, Hs)",
        },
        {
            "setup": """import numpy as np
M = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
P = np.array([-3.5, -1.2, -4.5, -0.8, 1.9, -0.45, 1.6, 1.85, -0.3, 1.6, 1.9, -0.7])
Q = np.array([0.45, 0.7, 0.3, 0.55, 0.3, 0.45, 0.3, 0.45, 0.3, 0.55, 0.3, 0.45])
R = np.array([1.4, 1.15, 1.9, 1.2, 1.9, 0.95, 1.65, 1.9, 0.95, 1.65, 1.9, 1.4])
H1 = np.stack([np.array([[P[e], Q[e]], [Q[e], R[e]]]) for e in range(12)])
s = np.array([0.0, 2.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H2 = H1 + s[:, None, None] * np.eye(2)
Hs = np.stack([H1, H2])
g = np.array([
    [24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875],
    [36.0, -18.0, 9.0, -5.0, 2.25, -1.125, 0.5625, -0.5625, 0.28125, -0.28125],
])
M = 6.0 * np.eye(10)
""",
            "call": "compute_progressive_decrement(M, g, pairs, Hs)",
            "gold_call": "_oracle_compute_progressive_decrement(M, g, pairs, Hs)",
        },
        {
            "setup": """import numpy as np
M = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
P = np.array([-3.5, -1.2, -4.5, -0.8, 1.9, -0.45, 1.6, 1.85, -0.3, 1.6, 1.9, -0.7])
Q = np.array([0.45, 0.7, 0.3, 0.55, 0.3, 0.45, 0.3, 0.45, 0.3, 0.55, 0.3, 0.45])
R = np.array([1.4, 1.15, 1.9, 1.2, 1.9, 0.95, 1.65, 1.9, 0.95, 1.65, 1.9, 1.4])
H1 = np.stack([np.array([[P[e], Q[e]], [Q[e], R[e]]]) for e in range(12)])
s = np.array([0.0, 2.6, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H2 = H1 + s[:, None, None] * np.eye(2)
Hs = np.stack([H1, H2])
g = np.array([
    [24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875],
    [36.0, -18.0, 9.0, -5.0, 2.25, -1.125, 0.5625, -0.5625, 0.28125, -0.28125],
])
g = np.vstack([g, g[-1]])
def run_model():
    try:
        compute_progressive_decrement(M, g, pairs, Hs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_progressive_decrement(M, g, pairs, Hs)
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
