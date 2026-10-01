"""
Accumulate the Euclidean arc length of the ordered surface-response path.

This final orchestrator reduces the source-grounded response table to the single benchmark scalar.

Returns
-------
float cumulative Euclidean path length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cumulative_triangle_surface_path(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> float:
    '''Return the cumulative Euclidean length of the ordered response path.

    Parameters
    ----------
    s_values : np.ndarray
        Nonempty one-dimensional array of positive finite kinematic invariants.
    gamma : float
        Finite nonzero Baikov exponent parameter.
    probes : np.ndarray
        Finite shape-(m,3) array of dimensionless off-cut probe coordinates.

    Returns
    -------
    float
        Sum_j ||r(s_j) - r(s_{j-1})||_2 as a native Python float, with 0.0
        returned for a one-point path.

    Raises
    ------
    ValueError
        If any upstream s_values, gamma, or probes precondition fails.
    '''
    return path_length

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cumulative_triangle_surface_path(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> float:
    table=_oracle_triangle_surface_path_table(s_values,gamma,probes)
    if table.shape[0]<2: return 0.0
    return float(np.linalg.norm(np.diff(table,axis=0),axis=1).sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'import numpy as np\ns_values=np.array([2.,3.,5.]); gamma=-2.; probes=np.array([[.05,-.03,.02],[-.04,.06,.01],[.02,.01,-.05]])','call':'cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','tol':3e-9},
        {'setup':'import numpy as np\ns_values=np.array([1.5,4.]); gamma=-3.; probes=np.array([[.03,.02,-.01],[-.02,.04,.05],[.06,-.01,.02]])','call':'cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','tol':3e-9},
        {'setup':'import numpy as np\ns_values=np.array([2.5,3.5,6.,8.]); gamma=-1.75; probes=np.array([[-.05,.01,.03],[.01,-.04,.02],[.04,.03,-.02]])','call':'cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','tol':3e-9},
        {'setup':'import numpy as np\ns_values=np.array([4.]); gamma=-2.5; probes=np.array([[.05,-.03,.02],[-.04,.06,.01],[.02,.01,-.05]])','call':'cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_cumulative_triangle_surface_path(s_values.copy(),gamma,probes.copy())','tol':1e-15},
    ]
