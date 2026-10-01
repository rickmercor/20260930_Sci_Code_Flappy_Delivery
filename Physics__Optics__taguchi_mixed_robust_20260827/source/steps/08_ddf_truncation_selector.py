"""
Compare nested DDF profiles across folds and reference losses.

The returned diagnostics preserve admissibility and response information for every nested order.

Returns
-------
np.ndarray of shape (8,), float: three responses, three minima, retained order, and retained response.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ddf_truncation_selector(log_a: np.ndarray, c0: float, validation_z_km: np.ndarray, references: np.ndarray, dense_z_km: np.ndarray, relative_tolerance: float=0.03) -> np.ndarray:
    """Return robust nested-profile and identifiability diagnostics.

Parameters
----------
log_a : numpy.ndarray
    Length-3 base-10 logarithmic factors `[A1,A2,A3]`.
c0 : float
    Finite nonzero signed coefficient.
validation_z_km : numpy.ndarray
    Float array of shape `(F,Z)` containing validation locations by fold.
references : numpy.ndarray
    Strictly positive reference profiles of shape `(F,R,Z)`.
dense_z_km : numpy.ndarray
    Nonempty one-dimensional nonnegative feasibility grid.
relative_tolerance : float
    Nonnegative fractional score tolerance for choosing the lowest order.

Returns
-------
out : numpy.ndarray
    Float array of shape `(8,)` ordered as `[three_worst_order_RMS_scores,
    three_dense_profile_minima,selected_order,selected_score]`.

Conventions
-----------
For each nested order, a validation profile with any value <=0 receives score exactly 1000000.0; this is a sentinel, not an RMS. Otherwise calculate sqrt(mean((1-sqrt(reference/profile))**2)) over locations separately for every fold/reference pair, then take the maximum. Dense minima are calculated separately for each order. Dense feasibility means minimum>0; the validation sentinel alone does not remove a dense-feasible order. Let best be the minimum score among dense-feasible orders. Select the lowest order whose score is <= best*(1+relative_tolerance)+1e-15 and whose dense minimum is positive. The selected order is one-based. No dense-feasible order raises ValueError. Inputs and returned diagnostics must be finite; grids are nonnegative, c0 is nonzero, references positive and tolerance nonnegative.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(8, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_ddf_truncation_selector(log_a, c0, validation_z_km, references, dense_z_km, relative_tolerance=0.03):
    logs = np.asarray(log_a, dtype=float)
    validation_z = np.asarray(validation_z_km, dtype=float)
    refs = np.asarray(references, dtype=float)
    dense_z = np.asarray(dense_z_km, dtype=float)
    coefficient, tolerance = float(c0), float(relative_tolerance)
    if logs.shape != (3,) or validation_z.ndim != 2 or min(validation_z.shape) < 1: raise ValueError("Require three logarithms and nonempty (F,Z) validation grid")
    if refs.ndim != 3 or refs.shape[0] != validation_z.shape[0] or refs.shape[2] != validation_z.shape[1] or refs.shape[1] < 1: raise ValueError("References must have matching shape (F,R,Z)")
    if dense_z.ndim != 1 or dense_z.size == 0: raise ValueError("Require a nonempty one-dimensional dense grid")
    if not all(np.isfinite(values).all() for values in (logs, validation_z, refs, dense_z, [coefficient, tolerance])): raise ValueError("All inputs must be finite")
    if coefficient == 0 or tolerance < 0 or np.any(refs <= 0) or np.any(validation_z < 0) or np.any(dense_z < 0): raise ValueError("Invalid coefficient, tolerance, references or distances")
    wide = np.longdouble
    def profiles(points):
        result = np.ones((3, points.size), dtype=wide)
        active = points.ravel() > 0
        if np.any(active):
            distances = points.ravel()[active].astype(wide)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                exponent = logs.astype(wide)[:, None] * np.log(wide(10))
                exponent = exponent + np.log(abs(wide(coefficient))) + np.log(distances)
                factors = np.sign(coefficient) * np.exp(exponent)
                linear = factors[0]
                quadratic = factors[1] ** 2 / 2
                cubic = factors[2] ** 3 / 6
                result[:, active] = np.stack((1 + linear, 1 + linear + quadratic, 1 + linear + quadratic + cubic))
        if not np.isfinite(result).all(): raise ValueError("Nested profile evaluation is not representable")
        return result.reshape((3,) + points.shape)
    validation = profiles(validation_z)
    minima = np.min(profiles(dense_z), axis=1)
    scores = np.full(3, wide(1e6), dtype=wide)
    for order, profile in enumerate(validation):
        if np.all(profile > 0):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                errors = abs(1 - np.sqrt(refs.astype(wide)) / np.sqrt(profile[:, None, :]))
                scale = np.max(errors, axis=2, keepdims=True)
                scaled = np.divide(errors, scale, out=np.zeros_like(errors), where=scale != 0)
                scores[order] = np.max(scale[..., 0] * np.sqrt(np.mean(scaled ** 2, axis=2)))
    if not np.isfinite(scores).all(): raise ValueError("Nested scores must be finite")
    feasible = minima > 0
    if not np.any(feasible): raise ValueError("No positive nested profile on the dense grid")
    best = np.min(scores[feasible])
    threshold = best * (1 + wide(tolerance)) + wide(1e-15)
    selected = int(np.flatnonzero(feasible & (scores <= threshold))[0])
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.concatenate((scores, minima, [selected + 1, scores[selected]])), dtype=float)
    if not np.isfinite(result).all(): raise ValueError("Returned diagnostics must be representable as float64")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
def test_cases():
    return [
        {'setup': 'z=np.array([[1.,2.,3.]]);a=np.array([-1.,-2.,-3.]);c=-1.;r=np.exp(-.043*z)[:,None,:];d=np.linspace(0.,3.,31)', 'call': 'ddf_truncation_selector(a,c,z,r,d,0.)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,0.)'},
        {'setup': 'z=np.array([[1.,3.,5.],[2.,4.,6.]]);a=np.array([-1.3,-1.5,-3.]);c=-.9;r=np.exp(-.052*z)[:,None,:];d=np.linspace(0.,8.,81)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.001)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.001)'},
        {'setup': 'z=np.array([[2.,5.,8.],[1.,4.,9.]]);a=np.array([-1.4,-1.7,-1.2]);c=-.8;r=np.stack((np.exp(-.038*z),np.exp(-.061*z)),axis=1);d=np.linspace(0.,10.,101)', 'call': 'ddf_truncation_selector(a,c,z,r,d,0.)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,0.)'},
        {'setup': 'z=np.array([[4.,8.,10.]]);a=np.array([-.7,-.8,-2.]);c=-1.1;r=np.exp(-.071*z)[:,None,:];d=np.linspace(0.,10.,201)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.05)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.05)'},
        {'setup': 'z=np.array([[1.,2.,3.],[1.5,2.5,3.5]]);a=np.array([-1.2,-2.,-2.5]);c=-.7;r=np.stack((np.exp(-.027*z),np.exp(-.049*z)),axis=1);d=np.linspace(0.,4.,41)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.2)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.2)'},
        {'setup': 'z=np.array([[.6,2.4,6.8],[1.2,4.2,8.6]]);a=np.array([-1.44,-1.79,-1.61]);c=-.96;al=np.array([.219,.267])*np.log(10)/10;r=np.exp(-al[None,:,None]*z[:,None,:]);d=np.linspace(.05,9.55,191)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.027)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.027)'},
        {'setup': 'z=np.array([[8.7,5.8,2.9],[7.2,4.3,1.4]]);a=np.array([-1.51,-1.73,-1.46]);c=-1.02;al=np.array([.273,.231])*np.log(10)/10;r=np.exp(-al[None,:,None]*z[:,None,:]);d=np.linspace(9.6,.1,191)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.041)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.041)'},
        {'setup': 'z=np.array([[.1,1.,10.],[.2,2.,8.],[.5,5.,9.]]);a=np.array([-2.5,-1.6,-1.4]);c=-1.2;r=np.stack((np.exp(-.034*z),np.exp(-.057*z)),axis=1);d=np.linspace(0.,10.,401)', 'call': 'ddf_truncation_selector(a,c,z,r,d,.002)', 'gold_call': '_oracle_ddf_truncation_selector(a,c,z,r,d,.002)'},
        {'setup': 'def _raises_value_error(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1.0\n    return 0.0', 'call': '_raises_value_error(ddf_truncation_selector,np.array([0.,-10.,-10.]),-2.,np.array([[1.,2.]]),np.ones((1,1,2)),np.array([1.,2.]))', 'gold_call': '_raises_value_error(_oracle_ddf_truncation_selector,np.array([0.,-10.,-10.]),-2.,np.array([[1.,2.]]),np.ones((1,1,2)),np.array([1.,2.]))'},
        {'setup': '', 'call': 'ddf_truncation_selector(np.array([400.,400.,400.]),-1.,np.array([[0.]]),np.array([[[1.]]]),np.array([0.]))', 'gold_call': '_oracle_ddf_truncation_selector(np.array([400.,400.,400.]),-1.,np.array([[0.]]),np.array([[[1.]]]),np.array([0.]))'},
        {'setup': 'scale=np.array([1e154,1e154,1e154,1.,1.,1.,1.,1e154])', 'call': 'ddf_truncation_selector(np.array([-1.,-200.,-200.]),-1.,np.array([[9.]]),np.array([[[1e308]]]),np.array([0.]))/scale', 'gold_call': '_oracle_ddf_truncation_selector(np.array([-1.,-200.,-200.]),-1.,np.array([[9.]]),np.array([[[1e308]]]),np.array([0.]))/scale'},
    ]
