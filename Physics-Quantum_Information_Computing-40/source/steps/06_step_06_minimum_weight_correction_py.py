"""
Deduplicate valid corrections and select the member with minimum Bernoulli log-likelihood weight.

In multi-result mode the source returns the minimum-weight correction using $\sum_j\hat e_j\log((1-p_j)/p_j)$ rather than merely the first valid vector encountered.

Returns
-------
real `np.ndarray` of length `N+1` equal to `[minimum_weight, correction_bits...]`
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimum_weight_correction(corrections: "np.ndarray", p: "np.ndarray") -> "np.ndarray":
    '''Return the minimum-weight unique correction.

    Parameters
    ----------
    corrections : np.ndarray
        Binary array of shape (R,N), R>=1; duplicate rows are allowed.
    p : np.ndarray
        Error probabilities of shape (N,), all strictly between 0 and 0.5.

    Returns
    -------
    result : np.ndarray
        Float vector [minimum_weight, correction_bits...]. Weight is the sum over
        selected bits of log((1-p_j)/p_j). Duplicates are removed preserving first
        appearance; exact weight ties choose the earliest remaining row.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_minimum_weight_correction(corrections: "np.ndarray", p: "np.ndarray") -> "np.ndarray":
    c = np.asarray(corrections, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    seen = set()
    unique = []
    for row in c:
        key = tuple(int(v) for v in row)
        if key not in seen:
            seen.add(key)
            unique.append(row.copy())
    u = np.vstack(unique)
    llr = np.log((1.0 - p) / p)
    weights = u @ llr
    k = int(np.argmin(weights))
    return np.concatenate([[float(weights[k])], u[k].astype(float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    return [
        {
            "setup": '''import numpy as np\nc=np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0]],dtype=int)\np=np.array([.08,.12,.05,.16])\n''',
            "call": "minimum_weight_correction(c.copy(),p.copy())",
            "gold_call": "_oracle_minimum_weight_correction(c.copy(),p.copy())",
            "tol": 1e-13,
        },
        {
            "setup": '''import numpy as np\nc=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=int)\np=np.array([.11,.11,.11])\n''',
            "call": "minimum_weight_correction(c.copy(),p.copy())",
            "gold_call": "_oracle_minimum_weight_correction(c.copy(),p.copy())",
            "tol": 1e-13,
        },
        {
            "setup": '''import numpy as np\nc=np.array([[1,1,0,0,1],[0,0,1,1,0],[1,0,0,1,0],[0,0,1,1,0]],dtype=int)\np=np.array([.19,.04,.08,.16,.07])\n''',
            "call": "minimum_weight_correction(c.copy(),p.copy())",
            "gold_call": "_oracle_minimum_weight_correction(c.copy(),p.copy())",
            "tol": 1e-13,
        },
    ]
