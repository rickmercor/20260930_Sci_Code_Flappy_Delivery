"""
Perform exactly num_iters updates of the paper’s subspace relaxation iteration, with the mass scaling and the treatment of the pseudoinverse across iterations following the source, and no tolerance-based exit. Validate the iteration count.

The update is a projected Newton step in the subspace the paper selects. How the pseudoinverse is treated between iterations is a convention the source paper states explicitly.

Returns
-------
float64 array (I, 3, 3): the relaxed configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def shr_relax(coords, num_iters):
    """coords: float array (I, 3, 3) starting configuration; num_iters:
    exact number of relaxation updates, validated.

    Returns float64 array (I, 3, 3): the relaxed configuration after
    exactly num_iters updates of the source iteration."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: subspace harmonic relaxation by Newton-Raphson iterations."""

import numpy as np

_MASSES = (1.0, 16.0, 1.0)
_LI = 3




def _stiff_pseudoinverse(blocks):
    K = np.asarray(blocks, dtype=np.float64)
    if K.ndim != 3 or K.shape[1:] != (9, 9) or not np.all(np.isfinite(K)):
        raise ValueError("invalid blocks")
    I = K.shape[0]
    Kt = np.zeros_like(K)
    for i in range(I):
        lam, U = np.linalg.eigh(K[i])
        order = np.argsort(lam)[::-1][:_LI]
        if np.any(lam[order] <= 0.0):
            raise ValueError("nonpositive selected eigenvalue")
        for l in order:
            Kt[i] += (1.0 / lam[l]) * np.outer(U[:, l], U[:, l])
    return Kt.astype(np.float64, copy=False)


def _oracle_shr_relax(coords, num_iters):
    r = np.asarray(coords, dtype=np.float64).copy()
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    if isinstance(num_iters, (bool, np.bool_)) or not isinstance(num_iters, (int, np.integer)) or num_iters < 0:
        raise ValueError("invalid iteration count")
    I = r.shape[0]
    sm = np.repeat(np.asarray(_MASSES, dtype=np.float64), 3) ** -0.5
    Kt = _stiff_pseudoinverse(_oracle_block_mass_scaled_hessian(r))
    for _ in range(int(num_iters)):
        g = _oracle_cluster_gradient(r)
        for i in range(I):
            step = sm * (Kt[i] @ (sm * g[i].reshape(9)))
            r[i] = r[i] - step.reshape(3, 3)
    return r.astype(np.float64, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'coords = _oracle_reference_configuration(np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]]))\nnum_iters = 2', "call": "shr_relax(coords, num_iters)", "gold_call": "_oracle_shr_relax(coords, num_iters)", "tol": 1e-08},
        {"setup": 'coords = _oracle_reference_configuration(np.array([[[1.15, 0.05, 0.1], [0.1, 0.05, 0.0], [-0.35, 1.05, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]]))\nnum_iters = 1', "call": "shr_relax(coords, num_iters)", "gold_call": "_oracle_shr_relax(coords, num_iters)", "tol": 1e-08},
        {"setup": 'coords = _oracle_reference_configuration(np.array([[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]]]))\nnum_iters = 0', "call": "shr_relax(coords, num_iters)", "gold_call": "_oracle_shr_relax(coords, num_iters)", "tol": 1e-09},
        {"setup": 'coords = _oracle_reference_configuration(np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]]))\nnum_iters = 3', "call": "shr_relax(coords, num_iters)", "gold_call": "_oracle_shr_relax(coords, num_iters)", "tol": 1e-08},
        {"setup": 'coords = _oracle_reference_configuration(np.array([[[-0.01695, 0.578721, 0.791478], [-0.031673, 0.012649, -0.009394], [0.023738, 0.584845, -0.859335]], [[1.92156, 0.555159, 0.809247], [1.931787, 0.015884, -0.030533], [1.90437, 0.566946, -0.841134]]]))\nnum_iters = 3', "call": "shr_relax(coords, num_iters)", "gold_call": "_oracle_shr_relax(coords, num_iters)", "tol": 1e-08},
    ]
