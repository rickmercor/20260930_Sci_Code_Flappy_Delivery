"""
Construct the paper's nodal induction-residual indicator.

First apply the lumped projection to the weak residual load:



R_i = residual_load_i / lumped_mass_i.



Compute the mass-weighted magnetic mean



H_bar = sum_i(lumped_mass_i * magnetic_field_i) / domain_measure,



and the magnetic normalization scale



s_H = max_i ||magnetic_field_i - H_bar||_2.



Return the nodal indicator



indicator_i = ||R_i||_2 / s_H



as a length-n real array in node order. Raise ValueError if s_H is no larger than machine epsilon.

Returns
-------
Return one length-n real NumPy array in node order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rescaled_induction_residual(lumped_mass, domain_measure,
                                magnetic_field, residual_load):
    """Project and normalize an induction residual load.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal masses.
    domain_measure : float
        Positive measure used by the magnetic mean.
    magnetic_field : array_like, shape (n,3)
        Cartesian nodal magnetic samples.
    residual_load : array_like, shape (n,3)
        Cartesian weak residual loads.
    Returns
    -------
    ndarray, shape (n,)
        Nonnegative rescaled indicator in node order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rescaled_induction_residual(lumped_mass, domain_measure,
                                        magnetic_field, residual_load):
    """Project and rescale the induction residual using paper Eqs. (50)-(51)."""
    import math
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    measure = float(domain_measure)
    magnetic = np.asarray(magnetic_field, dtype=float)
    load = np.asarray(residual_load, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or magnetic.shape != (mass.size, 3) or load.shape != magnetic.shape:
        raise ValueError("inconsistent nodal shapes")
    if (not math.isfinite(measure) or measure <= 0 or not np.all(np.isfinite(mass))
            or np.any(mass <= 0) or not np.all(np.isfinite(magnetic)) or not np.all(np.isfinite(load))):
        raise ValueError("inputs must be finite with positive masses and measure")
    projected = load / mass[:, None]
    magnetic_mean = np.sum(mass[:, None] * magnetic, axis=0) / measure
    scale = float(np.max(np.linalg.norm(magnetic - magnetic_mean, axis=1)))
    if scale <= np.finfo(float).eps:
        raise ValueError("magnetic normalization scale is zero")
    return (np.linalg.norm(projected, axis=1) / scale).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nm=np.array([.5,.5]); h=np.array([[1.,0,0],[0.,1.,0]]); b=np.array([[.1,0,0],[0,.2,0]])', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([.2,.3,.5]); h=np.eye(3); b=m[:,None]*np.array([[1.,2.,0.],[0,3.,4.],[2.,0,1.]])', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([1.,2.]); h=np.array([[-1.,0,2.],[2.,1.,0]]); b=np.ones((2,3))', 'call': 'rescaled_induction_residual(m,3.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,3.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([.1,.2,.3,.4]); h=np.arange(12.).reshape(4,3)/5; b=np.flip(h,axis=0)*m[:,None]', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([.25,.75]); h=np.array([[3.,2.,1.],[1.,2.,4.]]); b=np.array([[-.2,.4,0],[.3,-.6,.9]])', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([.4,.35,.25]); h=np.array([[.2,.4,.8],[.3,.1,.6],[.9,.2,.1]]); b=np.eye(3)*m[:,None]', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}, {'setup': 'import numpy as np\nm=np.array([.15,.2,.25,.4]); h=np.linspace(-1,1,12).reshape(4,3); b=np.linspace(.2,1.3,12).reshape(4,3)*m[:,None]', 'call': 'rescaled_induction_residual(m,1.,h,b)', 'gold_call': '_oracle_rescaled_induction_residual(m,1.,h,b)'}]
