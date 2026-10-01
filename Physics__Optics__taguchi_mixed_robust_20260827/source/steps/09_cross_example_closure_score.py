"""
Assemble the normalized shared-loss closure score.

Five independently scaled channels connect the two archived fiber designs.

Returns
-------
np.ndarray of shape (12,), float: six raw diagnostics, five normalized channels, and their maximum.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cross_example_closure_score(ddf_error: float, unit_order_response: float, theory_order_response: float, alpha_ddf: float, alpha_guiding: float, p0_relative_errors: np.ndarray, tolerances: np.ndarray) -> np.ndarray:
    """Return component errors and their normalized minimax score.

Parameters
----------
ddf_error, unit_order_response, theory_order_response : float
    Finite nonnegative archived response values.
alpha_ddf, alpha_guiding : float
    Finite inferred power-loss coefficients in `km^-1`.
p0_relative_errors : numpy.ndarray
    Nonempty one-dimensional relative launch-normalization errors.
tolerances : numpy.ndarray
    Five positive scales in DDF, unit-order, theoretical-order,
    shared-loss, and launch-normalization order.

Returns
-------
out : numpy.ndarray
    Shape `(12,)`, ordered as the six raw diagnostics
    `[DDF,unit,theory,alpha_DDF,alpha_guiding,max_abs_P0_error]`,
    the five normalized channels, and the joint maximum.

Conventions
-----------
Inferred alpha_ddf and alpha_guiding may have either sign; do not clip or reject finite negative slopes. Only the first three response arguments must be nonnegative. The loss channel is abs(alpha_ddf-alpha_guiding)/tolerances[3]. All inputs and returned diagnostics must be finite, and tolerances must be positive.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(12, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cross_example_closure_score(ddf_error, unit_order_response,
                                      theory_order_response, alpha_ddf,
                                      alpha_guiding, p0_relative_errors, tolerances):
    responses = np.asarray([ddf_error, unit_order_response, theory_order_response], dtype=float)
    slopes = np.asarray([alpha_ddf, alpha_guiding], dtype=float)
    errors = np.asarray(p0_relative_errors, dtype=float)
    scales = np.asarray(tolerances, dtype=float)
    if errors.ndim != 1 or errors.size == 0 or scales.shape != (5,):
        raise ValueError("Require nonempty error vector and five tolerances")
    if not all(np.isfinite(array).all() for array in (responses, slopes, errors, scales)):
        raise ValueError("Inputs must be finite")
    if np.any(responses < 0) or np.any(scales <= 0):
        raise ValueError("Responses must be nonnegative and tolerances positive")
    peak_error = np.max(np.abs(errors))
    wide = np.longdouble
    mismatch = abs(wide(slopes[0]) - wide(slopes[1]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        channels = np.asarray(np.array([*responses, mismatch, peak_error], dtype=wide)
                              / scales.astype(wide), dtype=float)
    output = np.concatenate((responses, slopes, [peak_error], channels, [np.max(channels)]))
    if not np.isfinite(output).all():
        raise ValueError("Closure diagnostics are not representable as float64")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight noncanonical cases vary the active closure channel and vector length."""
    return [{'setup': 'e=np.array([.02,-.03]);t=np.array([.1,.2,.05,.01,.08])', 'call': 'cross_example_closure_score(.04,.12,.03,.021,.024,e,t)', 'gold_call': '_oracle_cross_example_closure_score(.04,.12,.03,.021,.024,e,t)'},
        {'setup': 'e=np.zeros(3);t=np.ones(5)', 'call': 'cross_example_closure_score(0.,0.,0.,.02,.02,e,t)', 'gold_call': '_oracle_cross_example_closure_score(0.,0.,0.,.02,.02,e,t)'},
        {'setup': 'e=np.array([-.4,.1,.2]);t=np.array([2.,2.,2.,.5,.25])', 'call': 'cross_example_closure_score(1.5,.3,.2,.7,.2,e,t)', 'gold_call': '_oracle_cross_example_closure_score(1.5,.3,.2,.7,.2,e,t)'},
        {'setup': 'e=np.array([1e-12]);t=np.array([1e-9,1e-8,1e-7,1e-6,1e-5])', 'call': 'cross_example_closure_score(1e-10,2e-10,3e-10,4e-5,4.01e-5,e,t)', 'gold_call': '_oracle_cross_example_closure_score(1e-10,2e-10,3e-10,4e-5,4.01e-5,e,t)'},
        {'setup': 'e=np.linspace(-.06,.09,7);t=np.array([.04,.25,.07,.003,.12])', 'call': 'cross_example_closure_score(.018,.19,.055,.043,.041,e,t)', 'gold_call': '_oracle_cross_example_closure_score(.018,.19,.055,.043,.041,e,t)'},
        {'setup': 'e=np.array([-.01,.3]);t=np.array([.5,.4,.3,.2,.1])', 'call': 'cross_example_closure_score(.11,.12,.13,.04,.15,e,t)', 'gold_call': '_oracle_cross_example_closure_score(.11,.12,.13,.04,.15,e,t)'},
        {'setup': 'e=np.array([.013,-.017,.019,-.023]);t=np.array([.031,.17,.044,.0022,.075])', 'call': 'cross_example_closure_score(.029,.141,.038,.027,.0261,e,t)', 'gold_call': '_oracle_cross_example_closure_score(.029,.141,.038,.027,.0261,e,t)'},
        {'setup': 'e=np.array([-.7]);t=np.array([.2,.3,.4,.5,.6])', 'call': 'cross_example_closure_score(.19,.28,.37,1.2,.08,e,t)', 'gold_call': '_oracle_cross_example_closure_score(.19,.28,.37,1.2,.08,e,t)'},
        {'setup': '', 'call': 'cross_example_closure_score(.1,.2,.3,-.01,.02,np.zeros(1),np.ones(5))', 'gold_call': '_oracle_cross_example_closure_score(.1,.2,.3,-.01,.02,np.zeros(1),np.ones(5))'},
        {'setup': '', 'call': 'cross_example_closure_score(0.,0.,0.,-1e308,1e308,np.zeros(1),np.array([1.,1.,1.,1e308,1.]))', 'gold_call': '_oracle_cross_example_closure_score(0.,0.,0.,-1e308,1e308,np.zeros(1),np.array([1.,1.,1.,1e308,1.]))'}]
