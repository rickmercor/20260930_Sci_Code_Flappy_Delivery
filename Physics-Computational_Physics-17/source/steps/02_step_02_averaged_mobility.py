"""
Return the isotropic conformation average of reciprocal Rotne-Prager-Yamakawa mobility in three dimensions, normalized to the specified self_mobility. Inputs are a symmetric nonnegative squared-separation matrix with zero diagonal and positive finite radius and self_mobility. Include overlapping-bead separations and the zero-separation limit. Raise ValueError for invalid inputs.

Return the isotropic conformation average of reciprocal Rotne-Prager-Yamakawa mobility in three dimensions, normalized to the specified self_mobility. Inputs are a symmetric nonnegative squared-separation matrix with zero diagonal and positive finite radius and self_mobility. Include overlapping-bead separations and the zero-separation limit. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def averaged_mobility(squared_separations, radius, self_mobility):
    """Return the isotropic conformation average of reciprocal Rotne-Prager-Yamakawa mobility in three dimensions, normalized to the specified self_mobility. Inputs are a symmetric nonnegative squared-separation matrix with zero diagonal and positive finite radius and self_mobility. Include overlapping-bead separations and the zero-separation limit. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_averaged_mobility(squared_separations, radius, self_mobility):
    import numpy as np
    from scipy.special import erf
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(self_mobility) or self_mobility<=0:
        raise ValueError('invalid mobility inputs')
    z=d/(6*radius**2)
    m=np.ones_like(z)
    mask=z>0
    m[mask]=erf(1/np.sqrt(z[mask]))-np.sqrt(z[mask]/np.pi)*(-np.expm1(-1/z[mask]))
    return self_mobility*m

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np', 'call': 'averaged_mobility(np.array([[0.,1.],[1.,0.]]),0.5,1.)', 'gold_call': '_oracle_averaged_mobility(np.array([[0.,1.],[1.,0.]]),0.5,1.)'}, {'setup': 'import numpy as np', 'call': 'averaged_mobility(np.zeros((2,2)),0.5,1.)', 'gold_call': '_oracle_averaged_mobility(np.zeros((2,2)),0.5,1.)'}, {'setup': 'import numpy as np', 'call': 'averaged_mobility(np.array([[0.,100.],[100.,0.]]),0.5,2.)', 'gold_call': '_oracle_averaged_mobility(np.array([[0.,100.],[100.,0.]]),0.5,2.)'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: averaged_mobility(np.zeros((2,2)),0.,1.))', 'gold_call': '_invalid_result(lambda: _oracle_averaged_mobility(np.zeros((2,2)),0.,1.))'}]
