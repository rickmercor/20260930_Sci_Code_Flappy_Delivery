"""
Complete a sparsely observed, multiplicatively separable N-way array from exact, noiseless observations, returning the full array.

The completion is exact for a separable array whose sampling pattern determines it uniquely (up to compensating scalars that cancel in every cell): the recovered per-axis factors are combined into the full outer product, and the result reproduces every observed cell. Conventions the tests depend on: the result has shape dims and is compared cell by cell; axes are resolved one at a time using select_recursion_axis, extract_complementary_pattern and fit_resolved_factor.

Returns
-------
np.ndarray: the fully completed separable array of shape dims
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def complete_separable_landscape(obs: dict, dims: tuple) -> np.ndarray:
    """Complete a sparse, multiplicatively separable N-way array.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled array that is an outer product of one factor vector
        per axis (no interaction terms), with exact, noiseless values.
    dims : tuple
        Length-N tuple giving the size of each axis.

    Returns
    -------
    completed : np.ndarray
        Array of shape dims, the full outer product of the recovered per-axis
        factor vectors.

    Raises
    ------
    ValueError
        If obs is empty, dims has fewer than two axes or a non-positive size,
        or an observed index is out of bounds for dims.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_complete_separable_landscape(obs: dict, dims: tuple) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    if len(dims) < 2:
        raise ValueError("dims must have at least two axes")
    if any(d <= 0 for d in dims):
        raise ValueError("all entries of dims must be positive")
    for idx in obs:
        if len(idx) != len(dims):
            raise ValueError("index tuple length must match len(dims)")
        if any(not (0 <= i < d) for i, d in zip(idx, dims)):
            raise ValueError("observed index out of bounds for dims")

    labels = list(range(len(dims)))
    dim_map = {lbl: dims[lbl] for lbl in labels}

    def recurse(obs_: dict, labels_: list):
        kp = _oracle_select_recursion_axis(obs_)
        x = _oracle_extract_complementary_pattern(obs_, kp)
        complementary = sorted({idx[:kp] + idx[kp + 1:] for idx in obs_})
        lab = labels_[kp]
        rest = [l for l in labels_ if l != lab]
        obs2 = {complementary[i]: float(x[i]) for i in range(len(complementary))}
        if len(rest) >= 2:
            us = recurse(obs2, rest)
        else:
            v = np.zeros(dim_map[rest[0]])
            for idx, val in obs2.items():
                v[idx[0]] = val
            us = {rest[0]: v}
        others = [us[l] for l in labels_ if l != lab]
        us[lab] = _oracle_fit_resolved_factor(obs_, kp, others, dim_map[lab])
        return us

    factors = recurse(obs, labels)
    result = factors[labels[0]]
    for lbl in labels[1:]:
        result = np.multiply.outer(result, factors[lbl])
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
obs = {(0,0,0): 10.0, (0,0,1): 20.0, (1,0,0): 15.0, (0,1,0): 14.0, (1,1,1): 42.0}
dims = (2, 2, 2)
""",
            "call": "complete_separable_landscape(obs.copy(), dims)",
            "gold_call": "_oracle_complete_separable_landscape(obs.copy(), dims)",
        },
        {
            "setup": """
obs = {(0,0): 6.0, (0,1): 15.0, (0,2): 21.0, (1,0): 8.0, (1,2): 28.0}
dims = (2, 3)
""",
            "call": "complete_separable_landscape(obs.copy(), dims)",
            "gold_call": "_oracle_complete_separable_landscape(obs.copy(), dims)",
        },
        {
            "setup": """
obs = {(0,0,1):0.4864,(0,0,2):0.5472,(0,1,1):0.7296,(0,2,2):0.9576,
       (1,0,1):0.3584,(1,1,1):0.5376,(2,0,2):0.7416,(2,1,0):1.3596,
       (2,1,1):0.9888,(2,1,2):1.1124,(2,2,1):1.1536,(2,3,1):0.9064,
       (3,0,0):0.8976,(3,1,1):0.9792,(3,2,2):1.2852,(4,1,1):0.6816,
       (5,0,1):0.5120}
dims = (6, 4, 3)
""",
            "call": "complete_separable_landscape(obs.copy(), dims)",
            "gold_call": "_oracle_complete_separable_landscape(obs.copy(), dims)",
        },
    ]
