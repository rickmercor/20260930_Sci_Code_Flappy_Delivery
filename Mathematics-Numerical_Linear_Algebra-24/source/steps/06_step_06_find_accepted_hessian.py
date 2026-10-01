"""
Report the state at which the assembled system becomes factorizable.



The starting point is the unprojected assembly ``H = M + sum(S_e.T @ H_e @

S_e)``. If it already admits a Cholesky factorization, the state is reported

with nothing projected. Otherwise elements are admitted to the projected set

in successive rounds governed by the threshold, each admitted element

contributing ``S_e.T @ (Hhat_e - H_e) @ S_e`` exactly once and never twice,

until a round leaves the assembly factorizable.



How the threshold is initialized when none is yet in force, how it moves from

one round to the next, and the level at which the elements still remaining are

admitted unconditionally are all fixed by the established convention and are

not restated here.



The returned state carries both the threshold in force at acceptance and the

cumulative projected set, because a later iteration resumes from that state

rather than starting fresh.

Returns
-------
float np.ndarray of shape (n_elements + 1,): accepted threshold, then the 0/1 mask
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_accepted_state(
    regularizer: np.ndarray,
    residual: np.ndarray,
    element_hessians: np.ndarray,
    selection_matrices: np.ndarray,
    scores: np.ndarray,
    delta_init: float,
    eigenvalue_floor: float = 1e-8,
    fallback_threshold: float = 1e-12,
) -> np.ndarray:
    """Return the accepted threshold and cumulative projected set.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is square, finite, and equal to its transpose to within
    ``1e-12`` absolute; ``residual`` has shape ``(n_dof,)`` and is finite;
    ``element_hessians`` has shape ``(n_elements, 2, 2)``, is finite, and is
    symmetric along its last two axes to the same tolerance;
    ``selection_matrices`` has shape ``(n_elements, 2, n_dof)``, is finite,
    has every entry equal to ``0`` or ``1``, and has exactly one unit entry
    per row; ``scores`` has shape ``(n_elements,)`` and is finite and
    nonnegative; ``delta_init`` is a scalar that is strictly positive and not
    NaN (``inf`` is permitted); and ``eigenvalue_floor`` and
    ``fallback_threshold`` are both finite and strictly positive.
    ``ValueError`` is also raised if the loop exhausts every element and the
    fully projected assembly is still not positive definite.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    residual : np.ndarray
        Finite assembled residual of shape ``(n_dof,)``.
    element_hessians : np.ndarray
        Ordered symmetric matrices of shape ``(n_elements, 2, 2)``.
    selection_matrices : np.ndarray
        Ordered 0/1 selectors of shape ``(n_elements, 2, n_dof)``, float-typed
        with exactly one unit entry per row.
    scores : np.ndarray
        Ordered nonnegative element scores of shape ``(n_elements,)``.
    delta_init : float
        Positive threshold in force when the loop starts. Pass ``inf`` when no
        projection has been required yet, in which case the first finite
        threshold is derived from ``residual`` on entry to the loop.
    eigenvalue_floor : float, optional
        Positive local spectral floor.
    fallback_threshold : float, optional
        Positive convention constant governing when the round-based selection
        is abandoned in favour of admitting everything that remains.

    Returns
    -------
    np.ndarray
        Length ``n_elements + 1``. Entry 0 is the threshold in force when the
        factorization succeeded, and is returned unchanged as ``delta_init``
        when the unprojected assembly is already positive definite. Entries 1
        onward are the cumulative 0/1 projected indicator.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
# Convention constant supplied by the source, withheld from the public contract.
_CONTRACTION = 0.5


def _project_local(hessian: np.ndarray, floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(hessian)
    projected = (vectors * np.maximum(values, floor)) @ vectors.T
    return 0.5 * (projected + projected.T)


def _assemble(
    regularizer: np.ndarray,
    hessians: np.ndarray,
    selectors: np.ndarray,
) -> np.ndarray:
    result = regularizer.copy()
    for hessian, selector in zip(hessians, selectors):
        result += selector.T @ hessian @ selector
    return 0.5 * (result + result.T)


def _oracle_find_accepted_state(regularizer, residual, element_hessians, selection_matrices, scores, delta_init, eigenvalue_floor=1e-8, fallback_threshold=1e-12):
    """Reference tightening loop with incremental element updates."""
    
    alpha = 0.5  # source contraction factor, withheld from the public contract
    regularizer = np.asarray(regularizer, dtype=float)
    residual = np.asarray(residual, dtype=float)
    hessians = np.asarray(element_hessians, dtype=float)
    selectors = np.asarray(selection_matrices, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if regularizer.ndim != 2 or regularizer.shape[0] != regularizer.shape[1]:
        raise ValueError("regularizer must be square")
    if not np.all(np.isfinite(regularizer)) or not np.allclose(
        regularizer, regularizer.T, atol=1e-12, rtol=0.0
    ):
        raise ValueError("regularizer must be finite and symmetric")
    n_dof = regularizer.shape[0]
    if residual.shape != (n_dof,) or not np.all(np.isfinite(residual)):
        raise ValueError("residual has incompatible values or shape")
    if hessians.ndim != 3 or hessians.shape[1:] != (2, 2):
        raise ValueError("element_hessians must have shape (n_elements, 2, 2)")
    if selectors.shape != (hessians.shape[0], 2, n_dof):
        raise ValueError("selection_matrices have incompatible shape")
    if scores.shape != (hessians.shape[0],) or np.any(scores < 0.0):
        raise ValueError("scores have incompatible values or shape")
    if (
        not np.all(np.isfinite(hessians))
        or not np.all(np.isfinite(selectors))
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(hessians, np.swapaxes(hessians, 1, 2), atol=1e-12, rtol=0.0):
        raise ValueError("element Hessians must be symmetric")
    if not np.all((selectors == 0.0) | (selectors == 1.0)) or not np.all(
        np.sum(selectors, axis=2) == 1.0
    ):
        raise ValueError("selection matrices must contain one unit per row")
    if not np.isscalar(delta_init) or np.isnan(delta_init) or delta_init <= 0.0:
        raise ValueError("delta_init must be a positive scalar or inf")
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive and finite")
    if not np.isfinite(fallback_threshold) or fallback_threshold <= 0.0:
        raise ValueError("fallback_threshold must be positive and finite")

    n_elements = hessians.shape[0]
    accepted = _oracle_assemble_global_hessian(regularizer, hessians, selectors)
    processed = np.zeros(n_elements, dtype=bool)
    try:
        np.linalg.cholesky(accepted)
        return np.concatenate(([float(delta_init)], processed.astype(float)))
    except np.linalg.LinAlgError:
        pass

    delta = float(delta_init)
    if not np.isfinite(delta):
        delta = float(alpha) * float(np.linalg.norm(residual, ord=np.inf))
    projected = np.stack(
        [
            _oracle_project_element_hessian(hessian, float(eigenvalue_floor))
            for hessian in hessians
        ]
    )
    while True:
        chosen = _oracle_select_projection_mask(
            scores, delta, processed, delta <= fallback_threshold
        ).astype(bool)
        if not np.any(chosen):
            delta *= float(alpha)
            continue
        for element in np.flatnonzero(chosen):
            difference = projected[element] - hessians[element]
            selector = selectors[element]
            accepted += selector.T @ difference @ selector
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
            delta *= float(alpha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return uninitialized-threshold, warm-start, already-SPD, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
regularizer = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
rows = np.array([
    [-4.0, 0.5, 1.5], [-0.5, 0.75, 1.25], [-4.0, 0.25, 2.0], [-0.75, 0.5, 1.25],
    [2.0, 0.25, 2.0], [-0.5, 0.5, 1.0], [1.75, 0.25, 1.75], [2.0, 0.5, 2.0],
    [-0.25, 0.25, 1.0], [1.75, 0.5, 1.75], [2.0, 0.25, 2.0], [2.25, 0.5, 1.5],
])
element_hessians = np.stack([np.array([[p, q], [q, r]]) for p, q, r in rows])
selection_matrices = np.zeros((12, 2, 10))
for e, (i, j) in enumerate(pairs):
    selection_matrices[e, 0, i] = 1.0
    selection_matrices[e, 1, j] = 1.0
g = np.array([24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875])
scores = np.max(np.abs(np.einsum("eij,j->ei", selection_matrices, g)), axis=1)
delta_init = float("inf")
""",
            "call": "find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
            "gold_call": "_oracle_find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
        },
        {
            "setup": """import numpy as np
regularizer = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
rows = np.array([
    [-4.0, 0.5, 1.5], [-0.5, 0.75, 1.25], [-4.0, 0.25, 2.0], [-0.75, 0.5, 1.25],
    [2.0, 0.25, 2.0], [-0.5, 0.5, 1.0], [1.75, 0.25, 1.75], [2.0, 0.5, 2.0],
    [-0.25, 0.25, 1.0], [1.75, 0.5, 1.75], [2.0, 0.25, 2.0], [2.25, 0.5, 1.5],
])
element_hessians = np.stack([np.array([[p, q], [q, r]]) for p, q, r in rows])
selection_matrices = np.zeros((12, 2, 10))
for e, (i, j) in enumerate(pairs):
    selection_matrices[e, 0, i] = 1.0
    selection_matrices[e, 1, j] = 1.0
g = np.array([24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875])
scores = np.max(np.abs(np.einsum("eij,j->ei", selection_matrices, g)), axis=1)
delta_init = 1.0
""",
            "call": "find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
            "gold_call": "_oracle_find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
        },
        {
            "setup": """import numpy as np
regularizer = 40.0 * np.eye(4)
element_hessians = np.array([
    [[-6.0, 1.0], [1.0, 1.0]],
    [[1.0, 0.5], [0.5, -4.0]],
])
selection_matrices = np.zeros((2, 2, 4))
selection_matrices[0, 0, 0] = 1.0
selection_matrices[0, 1, 1] = 1.0
selection_matrices[1, 0, 2] = 1.0
selection_matrices[1, 1, 3] = 1.0
g = np.array([3.0, -1.0, 1.0, -0.5])
scores = np.array([3.0, 1.0])
delta_init = 2.75
""",
            "call": "find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
            "gold_call": "_oracle_find_accepted_state(regularizer, g, element_hessians, selection_matrices, scores, delta_init)",
        },
        {
            "setup": """import numpy as np
regularizer = 0.4 * np.eye(10)
pairs = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,6],[3,8],[2,9]])
rows = np.array([
    [-4.0, 0.5, 1.5], [-0.5, 0.75, 1.25], [-4.0, 0.25, 2.0], [-0.75, 0.5, 1.25],
    [2.0, 0.25, 2.0], [-0.5, 0.5, 1.0], [1.75, 0.25, 1.75], [2.0, 0.5, 2.0],
    [-0.25, 0.25, 1.0], [1.75, 0.5, 1.75], [2.0, 0.25, 2.0], [2.25, 0.5, 1.5],
])
element_hessians = np.stack([np.array([[p, q], [q, r]]) for p, q, r in rows])
selection_matrices = np.zeros((12, 2, 10))
for e, (i, j) in enumerate(pairs):
    selection_matrices[e, 0, i] = 1.0
    selection_matrices[e, 1, j] = 1.0
g = np.array([24.0, -12.0, 6.0, -3.0, 1.5, -0.75, 0.375, -0.375, 0.1875, -0.1875])
scores = np.max(np.abs(np.einsum("eij,j->ei", selection_matrices, g)), axis=1)
delta_init = float("inf")
element_hessians[3, 0, 1] += 0.5
def run_model():
    try:
        find_accepted_state(regularizer, g, element_hessians, selection_matrices,
                            scores, delta_init)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_find_accepted_state(regularizer, g, element_hessians, selection_matrices,
                                    scores, delta_init)
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
