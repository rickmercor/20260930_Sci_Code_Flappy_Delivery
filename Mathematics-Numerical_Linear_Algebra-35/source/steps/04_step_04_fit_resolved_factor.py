"""
Recover the factor vector of one resolved axis of a sparsely observed, multiplicatively separable N-way array, given the factor vectors of all the other axes, as the vector that best reproduces the observed cells.

Conventions the tests depend on: other_factors lists the other axes' factor vectors in increasing axis position, skipping the resolved axis; the returned vector has one entry per index of the resolved axis (length size); every index of the resolved axis must appear in at least one observation.

Returns
-------
np.ndarray: 1-D factor vector of the resolved axis, of length size
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_resolved_factor(obs: dict, axis: int, other_factors: list, size: int) -> np.ndarray:
    """Fit the resolved axis's factor vector from the observed cells.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values.
    axis : int
        Zero-based position of the axis whose factor is being recovered.
    other_factors : list
        The N-1 factor vectors (1-D arrays) of the other axes, in increasing
        axis position, skipping `axis`.
    size : int
        Number of indices along the resolved axis (length of the result).

    Returns
    -------
    u : np.ndarray
        One-dimensional array of length size, the factor of the resolved axis.

    Raises
    ------
    ValueError
        If obs is empty, axis is out of range, other_factors has the wrong
        length, or some index of the resolved axis is never observed.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fit_resolved_factor(obs: dict, axis: int, other_factors: list, size: int) -> np.ndarray:
    if not obs:
        raise ValueError("obs must be a non-empty dict of observations")
    n_dims = len(next(iter(obs)))
    if not (0 <= axis < n_dims):
        raise ValueError(f"axis must be in [0, {n_dims}), got {axis}")
    if len(other_factors) != n_dims - 1:
        raise ValueError("other_factors must hold one vector per non-resolved axis")
    others = [np.asarray(f, dtype=float) for f in other_factors]
    num = np.zeros(size)
    den = np.zeros(size)
    for idx, val in obs.items():
        c = 1.0
        o = 0
        for i in range(n_dims):
            if i == axis:
                continue
            c *= others[o][idx[i]]
            o += 1
        num[idx[axis]] += c * val
        den[idx[axis]] += c * c
    if np.any(den == 0.0):
        raise ValueError("every index of the resolved axis must be observed with a nonzero complementary product")
    return num / den

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
obs = {(0,0,0): 10.0, (0,0,1): 20.0, (1,0,0): 15.0, (0,1,0): 14.0, (1,1,1): 42.0}
axis = 0
other_factors = [np.array([1.0, 1.4]), np.array([1.0, 2.0])]
size = 2
""",
            "call": "fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
            "gold_call": "_oracle_fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
        },
        {
            "setup": """
import numpy as np
obs = {(0,0): 6.0, (0,1): 15.0, (0,2): 21.0}
axis = 0
other_factors = [np.array([2.0, 5.0, 7.0])]
size = 1
""",
            "call": "fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
            "gold_call": "_oracle_fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
        },
        {
            "setup": """
import numpy as np
obs = {(0,0,1):0.4864,(0,0,2):0.5472,(0,1,1):0.7296,(0,2,2):0.9576,
       (1,0,1):0.3584,(1,1,1):0.5376,(2,0,2):0.7416,(2,1,0):1.3596,
       (2,1,1):0.9888,(2,1,2):1.1124,(2,2,1):1.1536,(2,3,1):0.9064,
       (3,0,0):0.8976,(3,1,1):0.9792,(3,2,2):1.2852,(4,1,1):0.6816,
       (5,0,1):0.5120}
axis = 0
other_factors = [np.array([1.0, 1.5, 1.75, 1.375]), np.array([1.0, 8.0/11.0, 9.0/11.0])]
size = 6
""",
            "call": "fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
            "gold_call": "_oracle_fit_resolved_factor(obs.copy(), axis, [f.copy() for f in other_factors], size)",
        },
    ]
