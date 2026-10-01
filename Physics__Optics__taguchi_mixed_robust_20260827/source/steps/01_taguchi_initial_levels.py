"""
Construct interior factor levels and their common difference.

The same level geometry is used by each archived campaign.

Returns
-------
np.ndarray of shape (s+1,), float: ordered levels followed by level difference.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def taguchi_initial_levels(v_min: float, v_max: float, s: int=3) -> np.ndarray:
    """Return interior levels followed by their level difference.

Parameters
----------
v_min, v_max : float
    Strict lower and upper factor bounds.
s : int
    Positive number of interior levels.

Returns
-------
out : numpy.ndarray
    Float array of shape `(s+1,)` ordered as
    `[level_1, ..., level_s, level_difference]`.

Conventions
-----------
Require finite v_min < v_max and a positive integer s. The returned positive spacing must be representable as float64. These intervals initialize levels; this function imposes no subsequent campaign clipping.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(s + 1, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_taguchi_initial_levels(v_min, v_max, s=3):
    lower, upper = float(v_min), float(v_max)
    count = int(s)
    if count != s or count < 1 or not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Require finite v_min<v_max and positive integer s")
    wide = np.longdouble
    spacing = (wide(upper) - wide(lower)) / (count + 1)
    levels = wide(lower) + np.arange(1, count + 1, dtype=wide) * spacing
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(np.append(levels, spacing), dtype=float)
    if not np.isfinite(output).all() or output[-1] <= 0:
        raise ValueError("Levels or positive spacing are not representable")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight distinct regimes cover canonical, symmetric, narrow, large, and higher-cardinality bounds."""
    return [{'setup': '', 'call': 'taguchi_initial_levels(1.0, 10.0, 3)', 'gold_call': '_oracle_taguchi_initial_levels(1.0, 10.0, 3)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-10.0, 10.0, 3)', 'gold_call': '_oracle_taguchi_initial_levels(-10.0, 10.0, 3)'},
        {'setup': '', 'call': 'taguchi_initial_levels(2.0, 6.0, 1)', 'gold_call': '_oracle_taguchi_initial_levels(2.0, 6.0, 1)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-2.0, 7.0, 4)', 'gold_call': '_oracle_taguchi_initial_levels(-2.0, 7.0, 4)'},
        {'setup': '', 'call': 'taguchi_initial_levels(1e-9, 5e-9, 3)', 'gold_call': '_oracle_taguchi_initial_levels(1e-9, 5e-9, 3)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-1e6, 2e6, 5)', 'gold_call': '_oracle_taguchi_initial_levels(-1e6, 2e6, 5)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-1.25, -0.75, 2)', 'gold_call': '_oracle_taguchi_initial_levels(-1.25, -0.75, 2)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-0.125, 0.875, 7)', 'gold_call': '_oracle_taguchi_initial_levels(-0.125, 0.875, 7)'},
        {'setup': '', 'call': 'taguchi_initial_levels(-1e308,1e308,3)/1e308', 'gold_call': '_oracle_taguchi_initial_levels(-1e308,1e308,3)/1e308'},
        {'setup': '', 'call': 'taguchi_initial_levels(-8.,12.,4)', 'gold_call': '_oracle_taguchi_initial_levels(-8.,12.,4)'}]
