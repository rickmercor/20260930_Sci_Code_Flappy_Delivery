"""
Total the seven sector finite coefficients and apply the factor four for the other energy order and hard-leg hemisphere.

Take the five same-hemisphere finite coefficients I1,I2,I3,I4,I5 and the two opposite-hemisphere rows II1 and II2, use the epsilon-zero entry of each II row, sum all seven finite coefficients and multiply by four for the other energy order and hard-leg hemisphere. Return a Python float. The factor four restores the images this decomposition leaves implicit: the other energy order and the other hard-leg hemisphere. Accumulate in a dtype at least as wide as the inputs; summation order is not prescribed. Raise ValueError for complex dtype, bool entries, invalid shapes or nonfinite values.

Returns
-------
finite_part : float Four times the sum of the seven finite-coefficient estimates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def combine_sector_finite_parts(same_hemisphere: np.ndarray, opposite: np.ndarray) -> float:
    """Total the seven sector finite coefficients and apply the factor four for the other energy order and hard-leg hemisphere.

    Parameters
    ----------
    same_hemisphere : sequence of five real numbers
        The I1,I2,I3,I4,I5 finite coefficients before symmetry multiplication.
    opposite : ndarray, shape (2, 5)
        The II1 and II2 coefficient rows, epsilon powers -4 through zero.

    Returns
    -------
    finite_part : float
        Four times the sum of the seven finite-coefficient estimates.

    Raises
    ------
    ValueError
        Complex dtype, invalid shapes or nonfinite values.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_combine_sector_finite_parts(same_hemisphere: np.ndarray, opposite: np.ndarray) -> float:
    import numpy as np
    same = np.asarray(same_hemisphere)
    opp = np.asarray(opposite)
    if np.iscomplexobj(same) or np.iscomplexobj(opp):
        raise ValueError('complex dtype is not accepted')
    if same.dtype == bool or opp.dtype == bool:
        raise ValueError('bool entries are not accepted')
    same = same.astype(np.longdouble)
    opp = opp.astype(np.longdouble)
    if same.ndim != 1 or same.size != 5:
        raise ValueError('same_hemisphere must hold the five I-sector finite coefficients')
    if opp.ndim != 2 or opp.shape != (2,5):
        raise ValueError('opposite must be the two II rows of five coefficients')
    if not (np.all(np.isfinite(same)) and np.all(np.isfinite(opp))):
        raise ValueError('inputs must be finite')
    return float(4*np.sum(np.concatenate([same, opp[:,4]]),dtype=np.longdouble))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nsame=[-19.809694,-42.912132,1.486685,17.945853,9.703003]\nopposite=np.zeros((2,5));opposite[0,4]=39.674766;opposite[1,4]=-3.552380\n', 'call': '#case:normal\ncombine_sector_finite_parts(same,opposite)', 'gold_call': '_oracle_combine_sector_finite_parts(same,opposite)', 'tol': 1e-09},
     {'setup': 'import numpy as np\nsame=[0.0,0.0,0.0,0.0,0.0]\nopposite=np.zeros((2,5))\n', 'call': '#case:boundary\ncombine_sector_finite_parts(same,opposite)', 'gold_call': '_oracle_combine_sector_finite_parts(same,opposite)', 'tol': 1e-09},
     {'setup': 'import numpy as np\nsame=[1.5,-2.25,3.0,-4.75,5.125]\nopposite=np.arange(10,dtype=float).reshape(2,5)\n', 'call': '#case:edge\ncombine_sector_finite_parts(same,opposite)', 'gold_call': '_oracle_combine_sector_finite_parts(same,opposite)', 'tol': 1e-09},
     {'setup': 'import numpy as np\nsame=[1.0,2.0,3.0,4.0,5.0]\nopposite=np.full((2,5),0.5)\n', 'call': '#case:normal\ncombine_sector_finite_parts(same,opposite)', 'gold_call': '_oracle_combine_sector_finite_parts(same,opposite)', 'tol': 1e-09},
     {'setup': 'import numpy as np\ndef check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\nsame=[1.0,2.0,3.0]\nopposite=np.zeros((2,5))\n', 'call': '#case:edge\ncheck(lambda:combine_sector_finite_parts(same,opposite))', 'gold_call': 'check(lambda:_oracle_combine_sector_finite_parts(same,opposite))'}]
