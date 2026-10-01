"""
Retain paired finite-size covariance in batch localization estimates.

The paper fits separate disorder batches and reports means and standard errors of their localization lengths. Pairing size estimates through shared quenched disorder adds a covariance of their sample means.

Returns
-------
return statistics
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def paired_batch_statistics(short_fits: 'np.ndarray | list | tuple', long_fits: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Parameters
    ----------
    short_fits, long_fits : array_like, shape (B,5)
        Matching finite real fits for paired batches, B>=2. Columns are xi,xi_O,A,B_amplitude,residual; xi>0 and residual>=0.
    
    Returns
    -------
    statistics : real ndarray, shape (6,)
        [mu_short,mu_long,se_short,se_long,covariance_of_means,max_residual]. Means are arithmetic means of batch xi. Standard errors use unbiased sample variances divided by B. Covariance is unbiased paired sample covariance divided by B. Maximum residual spans both sizes. Length units are lattice spacings and covariance units are squared spacings. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_paired_batch_statistics(short_fits: 'np.ndarray | list | tuple', long_fits: 'np.ndarray | list | tuple') -> 'np.ndarray':
    a = np.asarray(short_fits, dtype=float)
    b = np.asarray(long_fits, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 5 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a[:, 0] <= 0) or np.any(b[:, 0] <= 0) or np.any(a[:, 4] < 0) or np.any(b[:, 4] < 0):
        raise ValueError('matching (B,5) fit arrays need B>=2, finite entries, positive lengths, nonnegative residuals')
    paired = np.column_stack((a[:, 0], b[:, 0]))
    means = np.mean(paired, axis=0)
    covariance = np.cov(paired, rowvar=False, ddof=1) / len(paired)
    return np.array([means[0], means[1], np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1]), covariance[0, 1], np.max(np.r_[a[:, 4], b[:, 4]])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# two-batch covariance boundary\nargs=([[2.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.02]], [[3.0, 1.0, 1.0, 0.0, 0.03], [5.0, 1.0, 1.0, 0.0, 0.04]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# perfect positive paired covariance\nargs=([[1.0, 1.0, 1.0, 0.0, 0.01], [3.0, 1.0, 1.0, 0.0, 0.01], [5.0, 1.0, 1.0, 0.0, 0.01], [7.0, 1.0, 1.0, 0.0, 0.01]], [[2.0, 1.0, 1.0, 0.0, 0.02], [6.0, 1.0, 1.0, 0.0, 0.02], [10.0, 1.0, 1.0, 0.0, 0.02], [14.0, 1.0, 1.0, 0.0, 0.02]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# perfect negative paired covariance\nargs=([[1.0, 1.0, 1.0, 0.0, 0.01], [2.0, 1.0, 1.0, 0.0, 0.01], [3.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.01]], [[8.0, 1.0, 1.0, 0.0, 0.03], [6.0, 1.0, 1.0, 0.0, 0.03], [4.0, 1.0, 1.0, 0.0, 0.03], [2.0, 1.0, 1.0, 0.0, 0.03]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# orthogonal centered batch fluctuations\nargs=([[2.0, 1.0, 1.0, 0.0, 0.02], [4.0, 1.0, 1.0, 0.0, 0.02], [2.0, 1.0, 1.0, 0.0, 0.02], [4.0, 1.0, 1.0, 0.0, 0.02]], [[2.0, 1.0, 1.0, 0.0, 0.01], [2.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.01]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# constant short-chain estimates\nargs=([[3.0, 1.0, 1.0, 0.0, 0.02], [3.0, 1.0, 1.0, 0.0, 0.02], [3.0, 1.0, 1.0, 0.0, 0.02]], [[1.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.01], [7.0, 1.0, 1.0, 0.0, 0.01]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# zero sampling uncertainty\nargs=([[3.0, 1.0, 1.0, 0.0, 0.01], [3.0, 1.0, 1.0, 0.0, 0.01], [3.0, 1.0, 1.0, 0.0, 0.01]], [[4.0, 1.0, 1.0, 0.0, 0.04], [4.0, 1.0, 1.0, 0.0, 0.04], [4.0, 1.0, 1.0, 0.0, 0.04]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# finite-sample outlier variance\nargs=([[1.0, 1.0, 1.0, 0.0, 0.02], [1.0, 1.0, 1.0, 0.0, 0.02], [1.0, 1.0, 1.0, 0.0, 0.02], [1.0, 1.0, 1.0, 0.0, 0.02], [16.0, 1.0, 1.0, 0.0, 0.02]], [[2.0, 1.0, 1.0, 0.0, 0.03], [2.0, 1.0, 1.0, 0.0, 0.03], [2.0, 1.0, 1.0, 0.0, 0.03], [2.0, 1.0, 1.0, 0.0, 0.03], [12.0, 1.0, 1.0, 0.0, 0.03]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# pairing changes covariance but preserves marginals\nargs=([[1.0, 1.0, 1.0, 0.0, 0.03], [2.0, 1.0, 1.0, 0.0, 0.03], [4.0, 1.0, 1.0, 0.0, 0.03], [8.0, 1.0, 1.0, 0.0, 0.03]], [[4.0, 1.0, 1.0, 0.0, 0.01], [1.0, 1.0, 1.0, 0.0, 0.01], [8.0, 1.0, 1.0, 0.0, 0.01], [2.0, 1.0, 1.0, 0.0, 0.01]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# common localization-length offset\nargs=([[1001.0, 1.0, 1.0, 0.0, 0.01], [1002.0, 1.0, 1.0, 0.0, 0.01], [1003.0, 1.0, 1.0, 0.0, 0.01]], [[1002.0, 1.0, 1.0, 0.0, 0.02], [1004.0, 1.0, 1.0, 0.0, 0.02], [1006.0, 1.0, 1.0, 0.0, 0.02]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}, {'setup': 'import numpy as np\n# short-size fit residual controls eligibility\nargs=([[2.0, 1.0, 1.0, 0.0, 0.01], [3.0, 1.0, 1.0, 0.0, 0.2], [5.0, 1.0, 1.0, 0.0, 0.02]], [[3.0, 1.0, 1.0, 0.0, 0.01], [4.0, 1.0, 1.0, 0.0, 0.03], [6.0, 1.0, 1.0, 0.0, 0.04]])\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'paired_batch_statistics(*deepcopy(args))', 'gold_call': '_oracle_paired_batch_statistics(*deepcopy(args))'}]
