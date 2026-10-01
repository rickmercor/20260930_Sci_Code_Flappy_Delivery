"""
Estimate synchronized lambda-window means and their covariance of the mean.

ATI samples the target-minus-auxiliary energy gap at each lambda. Synchronized block rows preserve cross-window correlations that enter GLS.

Returns
-------
return np.vstack((means, covariance)).astype(float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def gap_block_statistics(block_gaps):
    """Estimate the lambda-window mean gap and its correlated uncertainty.

    Parameters
    ----------
    block_gaps : array-like, shape (B,L)
        B synchronized block estimates of U1-U0 in eV/atom at L ordered
        lambda windows; require L>=3 and B>=L+1.

    Returns
    -------
    numpy.ndarray, shape (L+1,L)
        Row 0 contains the L mean gaps in input-window order. Rows 1..L
        contain the LxL sample covariance matrix of those means, in the same
        row/column order, in (eV/atom)^2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gap_block_statistics(block_gaps):
    """Estimate correlated lambda-window means and covariance of their mean."""
    import numpy as np
    values = np.asarray(block_gaps, dtype=float)
    if values.ndim != 2 or values.shape[0] < values.shape[1] + 1 or values.shape[1] < 3:
        raise ValueError("block_gaps must have shape (B,L), with L>=3 and B>=L+1")
    if not np.all(np.isfinite(values)):
        raise ValueError("block_gaps must be finite")
    means = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1) / values.shape[0]
    covariance = np.atleast_2d(covariance).astype(float)
    if np.min(np.linalg.eigvalsh(covariance)) <= 1.0e-16:
        raise ValueError("covariance of the mean must be positive definite")
    return np.vstack((means, covariance)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\na=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[2,0,0],[-2,0,0]],float)', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[2,0,0],[-2,0,0]],float)+np.array([3.,-2.,.5])', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(11).normal(size=(8,4));a[:,2]=0.6*a[:,0]+0.8*a[:,2]', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(12).normal(size=(9,5));a=a@np.array([[1,.2,0,0,0],[0,1,.3,0,0],[0,0,1,.4,0],[0,0,0,1,.5],[.1,0,0,0,1.]])', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(13).normal(size=(7,3))[:,::-1]', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(14).normal(size=(10,4))*np.array([.01,1.,10.,.2])', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(15).normal(size=(6,5))+1000.0', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}, {'setup': 'import numpy as np\na=np.random.default_rng(16).normal(size=(11,5));a+=np.linspace(-2,3,11)[:,None]*np.array([.1,-.2,.05,.3,-.1])', 'call': 'gap_block_statistics(a)', 'gold_call': '_oracle_gap_block_statistics(a)'}]
