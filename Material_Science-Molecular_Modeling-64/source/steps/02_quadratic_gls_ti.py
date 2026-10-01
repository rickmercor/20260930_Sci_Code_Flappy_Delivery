"""
Fit and integrate the quadratic alchemical energy-gap curve with correlated uncertainty.

For U_lambda=(1-lambda)U0+lambda U1, dF/dlambda=<U1-U0>_lambda. The paper fits a quadratic gap curve; integrating its GLS coefficients gives F_TI.

Returns
-------
return np.array([integral, standard_error, curvature], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def quadratic_gls_ti(lambdas, statistics):
    """Integrate a quadratic alchemical gap fitted with its full covariance.

    Returns a float64 ndarray of length 3 ordered as
    [F_TI_eV_per_atom, standard_error_eV_per_atom,
    quadratic_second_derivative_eV_per_atom].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_quadratic_gls_ti(lambdas, statistics):
    """Fit a quadratic energy-gap curve by GLS and integrate it from 0 to 1."""
    import numpy as np
    lam = np.asarray(lambdas, dtype=float)
    stats = np.asarray(statistics, dtype=float)
    if lam.ndim != 1 or lam.size < 3 or stats.shape != (lam.size + 1, lam.size):
        raise ValueError("statistics must have shape (L+1,L) for a length-L lambda vector")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(stats)):
        raise ValueError("inputs must be finite")
    if np.any(lam < 0.0) or np.any(lam > 1.0) or len(np.unique(lam)) != lam.size:
        raise ValueError("lambda values must be distinct and lie in [0,1]")
    means = stats[0]
    covariance = stats[1:]
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ValueError("covariance must be positive definite")
    design = np.column_stack((np.ones(lam.size), lam, lam * lam))
    inverse = np.linalg.inv(covariance)
    normal = design.T @ inverse @ design
    coefficient_covariance = np.linalg.inv(normal)
    coefficients = coefficient_covariance @ design.T @ inverse @ means
    integral_weights = np.array([1.0, 0.5, 1.0 / 3.0], dtype=float)
    integral = float(integral_weights @ coefficients)
    standard_error = float(np.sqrt(integral_weights @ coefficient_covariance @ integral_weights))
    curvature = float(2.0 * coefficients[2])
    return np.array([integral, standard_error, curvature], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nl=np.array([0.,.5,1.]);m=1-2*l+.6*l*l;c=np.diag([.02,.03,.01])**2;s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([0.,.2,.6,1.]);m=-.4+.3*l-.2*l*l;c=.0001*(.3**np.abs(np.subtract.outer(np.arange(4),np.arange(4))));s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([1.,.75,.25,0.]);m=.2-.1*l+.04*l*l;c=np.diag([.03,.02,.01,.025])**2;s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([0.,.1,.35,.7,1.]);m=-.7+.8*l-.5*l*l;c=.00004*(.6**np.abs(np.subtract.outer(np.arange(5),np.arange(5))));s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.linspace(0,1,7);m=.05-.9*l+.9*l*l;c=np.diag(np.linspace(.01,.025,7))**2;s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([0.,.3,.8,1.]);m=.2+.4*l;c=np.array([[4,1,0,0],[1,5,1,0],[0,1,6,1],[0,0,1,7]],float)*1e-5;s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([0.,.15,.4,.65,.85,1.]);m=1.2-2.1*l+1.7*l*l;c=np.diag([.01,.012,.017,.02,.014,.011])**2;s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}, {'setup': 'import numpy as np\nl=np.array([0.,.07,.19,.43,.72,.91,1.]);m=-.2+.9*l-.65*l*l;c=.00003*(.45**np.abs(np.subtract.outer(np.arange(7),np.arange(7))));s=np.vstack((m,c))', 'call': 'quadratic_gls_ti(l,s)', 'gold_call': '_oracle_quadratic_gls_ti(l,s)'}]
