"""
Evaluate the source surface polynomial on an ordered kinematic probe path.

This aggregation step repeatedly evaluates the completed source surface construction at the prescribed off-cut probes.

Returns
-------
np.ndarray shape-(n,m) response table.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triangle_surface_path_table(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> "np.ndarray":
    '''Evaluate normalized source-surface responses along an ordered s path.

    Parameters
    ----------
    s_values : np.ndarray
        Nonempty one-dimensional array of positive finite kinematic invariants.
    gamma : float
        Finite nonzero Baikov exponent parameter passed to the surface-term map.
    probes : np.ndarray
        Finite shape-(m,3) array of dimensionless off-cut probe coordinates.

    Returns
    -------
    np.ndarray
        Shape-(n,m) response table whose j,l entry is
        S_{s_j}(s_j * probes[l]) / s_j^2, preserving s_values order.

    Raises
    ------
    ValueError
        If s_values are invalid, probes do not have shape (m,3) or contain
        non-finite values, or an upstream gamma precondition fails.
    '''
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _eval_surface(c,z):
    basis=np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],dtype=int)
    z=np.asarray(z,dtype=float)
    return float(sum(float(v)*np.prod(z**m) for v,m in zip(c,basis)))

def _oracle_triangle_surface_path_table(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> "np.ndarray":
    sv=np.asarray(s_values,dtype=float); probes=np.asarray(probes,dtype=float)
    if sv.ndim!=1 or sv.size<1 or np.any(~np.isfinite(sv)) or np.any(sv<=0): raise ValueError('invalid s_values')
    if probes.ndim!=2 or probes.shape[1]!=3 or np.any(~np.isfinite(probes)): raise ValueError('invalid probes')
    rows=[]
    for s in sv:
        c=_oracle_triangle_surface_polynomial(float(s),gamma)
        rows.append([_eval_surface(c,float(s)*u)/(float(s)*float(s)) for u in probes])
    return np.asarray(rows,dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':'import numpy as np\ns_values=np.array([2.,3.,5.]); gamma=-2.; probes=np.array([[.05,-.03,.02],[-.04,.06,.01],[.02,.01,-.05]])','call':'triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','tol':1e-9},
        {'setup':'import numpy as np\ns_values=np.array([4.]); gamma=-3.; probes=np.array([[.03,.02,-.01],[-.02,.04,.05]])','call':'triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','tol':1e-9},
        {'setup':'import numpy as np\ns_values=np.array([1.5,2.5,6.]); gamma=-1.75; probes=np.array([[-.05,.01,.03],[.01,-.04,.02],[.04,.03,-.02],[0.,.02,.01]])','call':'triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','gold_call':'_oracle_triangle_surface_path_table(s_values.copy(),gamma,probes.copy())','tol':1e-9},
    ]
