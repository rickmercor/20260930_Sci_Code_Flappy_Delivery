"""
Recover conditional fixation-time derivatives and mixed logarithmic sensitivity.

A conditional first-passage mean is a ratio of a weighted moment to its event probability. Mixed differentiation includes both normalization responses and their cross term.

Returns
-------
Shape (5,), ordered conditional mean time, its epsilon derivative, its zeta derivative, its mixed derivative, and the mixed derivative of its natural logarithm. Conditioning is on hitting frequency one before zero. Derivatives are not factorial-scaled.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conditional_time_statistics(moments: 'np.ndarray') -> 'np.ndarray':
    """Recover conditional fixation-time derivatives and mixed logarithmic sensitivity.

    Parameters
    ----------
    moments : np.ndarray
        Finite (4,2) jet, columns fixation probability p and fixation-weighted
        time moment u. Rows are value, epsilon, zeta, epsilon-zeta.
        Baseline 0 < p <= 1 and u > 0; derivatives can have either sign.

    Returns
    -------
    result : np.ndarray
        Shape (5,), ordered conditional mean time, its epsilon derivative,
        its zeta derivative, its mixed derivative, and the mixed derivative
        of its natural logarithm. Conditioning is on hitting frequency one
        before zero. Derivatives are not factorial-scaled.

    Raises
    ------
    ValueError
        If moments is not finite real shape (4,2), baseline probability is
        outside (0,1], baseline weighted moment is nonpositive, or a computed
        result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_conditional_time_statistics(moments: 'np.ndarray') -> 'np.ndarray':
    a=_arr(moments,2,(4,2))
    if not 0<a[0,0]<=1 or a[0,1]<=0:raise ValueError('positive valid hitting probability and moment required')
    logt=_log(a[:,1])-_log(a[:,0]);t=_exp(logt)
    return _arr(np.r_[t,logt[3]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge and declared-invalid test cases."""
    return [{'setup': 'import numpy as np\nimport copy\n# normal\na0=np.array([[0.14083730852556037, 1021407.5860845979], [99.96638518409058, 725056029.3506497], [155.09143446032866, 1124874765.2853467], [49748.945265414586, 148913277546.98444]], dtype=float)\n', 'call': 'conditional_time_statistics(a0.copy())', 'gold_call': '_oracle_conditional_time_statistics(a0.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# boundary\na0=np.array([[0.2, 0.4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)\n', 'call': 'conditional_time_statistics(a0.copy())', 'gold_call': '_oracle_conditional_time_statistics(a0.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# edge\na0=np.array([[1e-08, 0.4], [2e-09, 0.1], [-3e-09, 0.2], [1e-09, -0.03]], dtype=float)\n', 'call': 'conditional_time_statistics(a0.copy())', 'gold_call': '_oracle_conditional_time_statistics(a0.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# invalid_declared_condition\na0=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    return 0\n', 'call': 'expect_value_error(conditional_time_statistics, copy.deepcopy(a0))', 'gold_call': 'expect_value_error(_oracle_conditional_time_statistics, copy.deepcopy(a0))', 'tol': 0}]
