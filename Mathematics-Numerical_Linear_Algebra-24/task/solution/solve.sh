#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def build_selection_matrices(index_pairs, n_dof):
    """Reference construction with strict index validation."""
    
    pairs = np.asarray(index_pairs)
    if not isinstance(n_dof, (int, np.integer)) or int(n_dof) < 2:
        raise ValueError("n_dof must be an integer at least 2")
    if pairs.ndim != 2 or pairs.shape[0] < 1 or pairs.shape[1] != 2:
        raise ValueError("index_pairs must have shape (n_elements, 2)")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("index_pairs must contain integers")
    pairs = pairs.astype(int, copy=False)
    if np.any(pairs < 0) or np.any(pairs >= int(n_dof)):
        raise ValueError("index_pairs contain an out-of-range index")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("the two indices of an element must be distinct")
    result = np.zeros((pairs.shape[0], 2, int(n_dof)), dtype=float)
    rows = np.arange(2)
    for e, pair in enumerate(pairs):
        result[e, rows, pair] = 1.0
    return result

import numpy as np
def compute_local_residual_scores(selection_matrices, residual):
    """Reference implementation of the local residual proxy."""
    
    selections = np.asarray(selection_matrices, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if selections.ndim != 3 or selections.shape[0] < 1 or selections.shape[1] < 1:
        raise ValueError("selection_matrices must have shape (E, m, n)")
    if residual.ndim != 1 or residual.size != selections.shape[2]:
        raise ValueError("residual shape is incompatible with selections")
    if not np.all(np.isfinite(selections)) or not np.all(np.isfinite(residual)):
        raise ValueError("inputs must be finite")
    if not np.all((selections == 0.0) | (selections == 1.0)):
        raise ValueError("selection matrices must be Boolean")
    if not np.all(np.sum(selections, axis=2) == 1.0):
        raise ValueError("each selection row must contain one unit entry")
    restricted = np.einsum("emn,n->em", selections, residual)
    return np.max(np.abs(restricted), axis=1)

import numpy as np

def project_element_hessian(hessian, eigenvalue_floor=1e-8):
    """Reference symmetric eigenvalue-clamping projection."""
  
    hessian = np.asarray(hessian, dtype=float)
    if (
        hessian.ndim != 2
        or hessian.shape[0] < 1
        or hessian.shape[0] != hessian.shape[1]
    ):
        raise ValueError("hessian must be a nonempty square matrix")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("hessian must be finite")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("hessian must be symmetric")
    if not np.isscalar(eigenvalue_floor) or not np.isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be finite")
    if float(eigenvalue_floor) <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    values, vectors = np.linalg.eigh(hessian)
    projected = (vectors * np.maximum(values, float(eigenvalue_floor))) @ vectors.T
    return 0.5 * (projected + projected.T)

import numpy as np

def assemble_global_hessian(regularizer, element_hessians, selection_matrices):
    """Reference finite-element assembly."""
    
    regularizer = np.asarray(regularizer, dtype=float)
    local = np.asarray(element_hessians, dtype=float)
    selections = np.asarray(selection_matrices, dtype=float)
    if (
        regularizer.ndim != 2
        or regularizer.shape[0] < 1
        or regularizer.shape[0] != regularizer.shape[1]
    ):
        raise ValueError("regularizer must be a nonempty square matrix")
    if local.ndim != 3 or selections.ndim != 3:
        raise ValueError("element_hessians and selections must be rank three")
    if local.shape[0] < 1 or local.shape[0] != selections.shape[0]:
        raise ValueError("element counts must agree")
    if local.shape[1] != local.shape[2] or local.shape[1] != selections.shape[1]:
        raise ValueError("local dimensions must agree")
    if selections.shape[2] != regularizer.shape[0]:
        raise ValueError("global dimensions must agree")
    if (
        not np.all(np.isfinite(regularizer))
        or not np.all(np.isfinite(local))
        or not np.all(np.isfinite(selections))
    ):
        raise ValueError("all inputs must be finite")
    if not np.allclose(regularizer, regularizer.T, rtol=0.0, atol=1e-12):
        raise ValueError("regularizer must be symmetric")
    if not np.allclose(local, np.swapaxes(local, 1, 2), rtol=0.0, atol=1e-12):
        raise ValueError("each element Hessian must be symmetric")
    result = regularizer.copy()
    for hessian, selection in zip(local, selections):
        result += selection.T @ hessian @ selection
    return 0.5 * (result + result.T)

import numpy as np
def select_projection_mask(scores, delta, projected_mask, force_all=False):
    """Reference implementation of the strict gate and fallback."""
    
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(projected_mask)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("scores must be finite and nonnegative")
    if mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("projected_mask must be Boolean and match scores")
    if not np.isscalar(delta) or np.isnan(float(delta)) or float(delta) < 0.0:
        raise ValueError("delta must be a nonnegative scalar")
    if not isinstance(force_all, (bool, np.bool_)):
        raise ValueError("force_all must be Boolean")  # noqa: TRY004
    remaining = ~mask
    chosen = remaining if force_all else remaining & (scores > float(delta))
    return chosen.astype(int)

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


def find_accepted_state(regularizer, residual, element_hessians, selection_matrices, scores, delta_init, eigenvalue_floor=1e-8, fallback_threshold=1e-12):
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
    accepted = assemble_global_hessian(regularizer, hessians, selectors)
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
            project_element_hessian(hessian, float(eigenvalue_floor))
            for hessian in hessians
        ]
    )
    while True:
        chosen = select_projection_mask(
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

import numpy as np

def solve_newton_step(accepted_hessian, residual):
    """Reference solve for the accepted positive-definite system."""
    
    hessian = np.asarray(accepted_hessian, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("accepted_hessian must be square")
    if residual.shape != (hessian.shape[0],):
        raise ValueError("residual has incompatible shape")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(residual)):
        raise ValueError("inputs must be finite")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("accepted_hessian must be symmetric")
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError("accepted_hessian must be positive definite") from exc
    return np.linalg.solve(hessian, -residual)

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


def compute_progressive_decrement(regularizer, residuals, index_pairs, element_hessians, eigenvalue_floor=1e-8, fallback_threshold=1e-12):
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

    selectors = build_selection_matrices(index_pairs, n_dof)

    delta = float("inf")
    decrement = 0.0
    for iteration in range(residuals.shape[0]):
        gradient = residuals[iteration]
        local = hessians[iteration]
        scores = compute_local_residual_scores(selectors, gradient)
        state = find_accepted_state(
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
                project_element_hessian(local[e], float(eigenvalue_floor))
                if mask[e]
                else local[e]
                for e in range(local.shape[0])
            ]
        )
        accepted = assemble_global_hessian(regularizer, effective, selectors)
        step = solve_newton_step(accepted, gradient)
        decrement = float(-gradient @ step)
        delta *= float(beta)

    if not np.isfinite(decrement):
        raise ValueError("the squared decrement is not finite")
    return decrement
SCICODE_GOLD_EOF
