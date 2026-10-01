"""
Select the largest admissible localization certificate.

The benchmark supplements finite-size localization comparisons by accounting for correlated sampling uncertainty and the largest fit residual. For each row s_delta=sqrt(max(0,se_s^2+se_l^2-2cov)), D=|mu_l-mu_s|+penalty*s_delta, and J=mu_l-penalty*se_l.

Returns
-------
return certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_localization_certificate(couplings: 'np.ndarray | list | tuple', statistics: 'np.ndarray | list | tuple', drift_limit: float, fit_limit: float, penalty: float) -> 'np.ndarray':
    """Parameters
    ----------
    couplings : array_like, shape (C,)
        Nonempty finite real interaction vector.
    statistics : array_like, shape (C,6)
        Finite rows [mu_short,mu_long,se_short,se_long,covariance_of_means,max_residual]. Means are positive, errors and residuals nonnegative, and |covariance|<=se_short*se_long+1e-12.
    drift_limit, fit_limit, penalty : float
        Finite nonnegative acceptance bounds and uncertainty penalty.
    
    Returns
    -------
    certificate : real ndarray, shape (7,)
        [J,V,mu_long,se_long,se_delta,D,max_residual]. Eligible rows have D<=drift_limit, residual<=fit_limit and J>0. Maximize J; scores within 1e-10 of the maximum tie by smaller V then earlier row. Empty eligibility returns [-1,0,0,0,0,0,0]. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_localization_certificate(couplings: 'np.ndarray | list | tuple', statistics: 'np.ndarray | list | tuple', drift_limit: float, fit_limit: float, penalty: float) -> 'np.ndarray':
    v = np.asarray(couplings, dtype=float)
    a = np.asarray(statistics, dtype=float)
    if v.ndim != 1 or v.size < 1 or a.shape != (len(v), 6) or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('couplings and finite (C,6) statistics must match')
    if np.any(a[:, :2] <= 0) or np.any(a[:, 2:4] < 0) or np.any(a[:, 5] < 0) or np.any(np.abs(a[:, 4]) > a[:, 2] * a[:, 3] + 1e-12):
        raise ValueError('statistics must represent positive lengths and a positive semidefinite covariance')
    if any(not np.isfinite(q) or q < 0 for q in (drift_limit, fit_limit, penalty)):
        raise ValueError('limits and penalty must be finite and nonnegative')
    diff_se = np.sqrt(np.maximum(0., a[:, 2] ** 2 + a[:, 3] ** 2 - 2 * a[:, 4]))
    drift = np.abs(a[:, 1] - a[:, 0]) + penalty * diff_se
    score = a[:, 1] - penalty * a[:, 3]
    ids = np.flatnonzero((drift <= drift_limit) & (a[:, 5] <= fit_limit) & (score > 0))
    if len(ids) == 0:
        return np.array([-1., 0., 0., 0., 0., 0., 0.])
    maximum = np.max(score[ids])
    tied = ids[score[ids] >= maximum - 1e-10]
    i = min(tied, key=lambda j: (v[j], j))
    return np.array([score[i], v[i], a[i, 1], a[i, 3], diff_se[i], drift[i], a[i, 5]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# eligible maximum lower-confidence length\nargs=([0.2, 0.8], [[3, 4, 0.2, 0.3, 0.04, 0.02], [4, 5, 0.3, 0.4, 0.06, 0.03]], 3.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# finite-size drift excludes longer candidate\nargs=([0.2, 0.8], [[3, 4, 0.2, 0.3, 0.04, 0.02], [1, 10, 0.1, 0.1, 0, 0.01]], 3.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# oscillatory-fit misfit excludes longer candidate\nargs=([0.2, 0.8], [[3, 4, 0.2, 0.3, 0.04, 0.02], [9, 10, 0.1, 0.1, 0, 0.2]], 3.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# sampling penalty reverses mean-length ranking\nargs=([0.2, 0.8], [[4, 5, 0.1, 0.1, 0, 0.02], [5, 6, 2, 2, 4, 0.03]], 10.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# positive pairing rescues finite-size consistency\nargs=([0.2], [[4, 4, 1, 1, 1, 0.02]], 0.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# anticorrelated pairing defeats consistency\nargs=([0.2], [[4, 4, 1, 1, -1, 0.02]], 1.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# tied utility selects smaller repulsion\nargs=([0.8, 0.2], [[4, 5, 0.2, 0.5, 0.05, 0.02], [4, 5, 0.2, 0.5, 0.05, 0.03]], 3.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# duplicate coupling resolves by input order\nargs=([0.2, 0.2], [[4, 5, 0.2, 0.5, 0.05, 0.02], [4, 5, 0.2, 0.5, 0.05, 0.03]], 3.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# no admissible design sentinel\nargs=([0.2, 0.8], [[1, 10, 0.1, 0.1, 0, 0.02], [1, 9, 0.1, 0.1, 0, 0.03]], 0.1, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# inclusive physical acceptance boundaries\nargs=([0.2], [[3, 4, 0, 0, 0, 0.07]], 1.0, 0.07, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# zero uncertainty penalty uses mean length\nargs=([0.2, 0.8], [[4, 5, 0.1, 0.1, 0, 0.02], [5, 6, 2, 2, 4, 0.03]], 10.0, 0.1, 0.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}, {'setup': 'import numpy as np\n# nonpositive confidence length excluded\nargs=([0.2], [[1, 1, 2, 2, 4, 0.01]], 2.0, 0.1, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 1] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'select_localization_certificate(*deepcopy(args))', 'gold_call': '_oracle_select_localization_certificate(*deepcopy(args))'}]
