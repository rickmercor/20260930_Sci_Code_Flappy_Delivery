"""
Return the conformation-averaged dot-product covariance of an independent spatial Gaussian velocity field. Its Cartesian covariance at positions separated by r is amplitude/3 times the identity times exp(-r*r/(2*length**2)), with unit temporal delta correlation. The isotropic Gaussian displacement has mean squared length given by squared_separations. This is a symmetric finite nonnegative square matrix with zero diagonal; amplitude is finite nonnegative and length finite positive. Include self correlations. Raise ValueError for invalid inputs.

Return the conformation-averaged dot-product covariance of an independent spatial Gaussian velocity field. Its Cartesian covariance at positions separated by r is amplitude/3 times the identity times exp(-r*r/(2*length**2)), with unit temporal delta correlation. The isotropic Gaussian displacement has mean squared length given by squared_separations. This is a symmetric finite nonnegative square matrix with zero diagonal; amplitude is finite nonnegative and length finite positive. Include self correlations. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def averaged_spatial_covariance(squared_separations, amplitude, length):
    """Return the spatial velocity covariance averaged over Gaussian conformations; the matrix contains dot-product, not single-component, covariances. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_averaged_spatial_covariance(squared_separations, amplitude, length):
    import numpy as np
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(amplitude) or amplitude<0 or not np.isfinite(length) or length<=0:
        raise ValueError('invalid spatial covariance inputs')
    return amplitude*(1+d/(3*length**2))**(-1.5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {'setup':'import numpy as np','call':'averaged_spatial_covariance(np.array([[0.,1.],[1.,0.]]),4.,1.5)','gold_call':'_oracle_averaged_spatial_covariance(np.array([[0.,1.],[1.,0.]]),4.,1.5)'},
      {'setup':'import numpy as np','call':'averaged_spatial_covariance(np.zeros((2,2)),2.,0.7)','gold_call':'_oracle_averaged_spatial_covariance(np.zeros((2,2)),2.,0.7)'},
      {'setup':'import numpy as np','call':'averaged_spatial_covariance(np.array([[0.,6.,9.],[6.,0.,3.],[9.,3.,0.]]),1.5,2.)','gold_call':'_oracle_averaged_spatial_covariance(np.array([[0.,6.,9.],[6.,0.,3.],[9.,3.,0.]]),1.5,2.)'},
      {'setup':'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1','call':'_invalid_result(lambda: averaged_spatial_covariance(np.eye(2),4.,0.))','gold_call':'_invalid_result(lambda: _oracle_averaged_spatial_covariance(np.eye(2),4.,0.))'}]
