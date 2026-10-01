"""
Evaluate four archived cubic DDF-profile conventions.

Each row uses the same logarithmic factors, coefficient, and distance grid.

Returns
-------
np.ndarray of shape (4,len(z_km)), float: candidate profiles in declared order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dispersion_profile_candidates(log_a: np.ndarray, c0: float, z_km: np.ndarray) -> np.ndarray:
    """Return four normalized candidate profiles.

Parameters
----------
log_a : numpy.ndarray
    Length-3 base-10 logarithmic factors `[A1,A2,A3]`.
c0 : float
    Finite signed coefficient.
z_km : numpy.ndarray
    Nonempty one-dimensional nonnegative distances in km.

Returns
-------
profiles : numpy.ndarray
    Float array of shape `(4,len(z_km))`; rows follow candidate order
    `0,1,2,3` from the task table.

Conventions
-----------
All inputs must be finite. If c0 is zero, every profile equals one for any finite log_a and distance. At zero distance, every profile equals one. Negative finite profiles are returned without clipping; admissibility is decided by callers. Every returned component must be finite in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, np.asarray(z_km).size), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dispersion_profile_candidates(log_a, c0, z_km):
    logs = np.asarray(log_a, dtype=float)
    distances = np.asarray(z_km, dtype=float)
    coefficient = float(c0)
    if logs.shape != (3,) or distances.ndim != 1 or distances.size == 0:
        raise ValueError("Require three logarithms and nonempty one-dimensional distances")
    if not np.isfinite(logs).all() or not np.isfinite(distances).all() or not np.isfinite(coefficient) or np.any(distances < 0):
        raise ValueError("Require finite inputs and nonnegative distances")
    output = np.ones((4, distances.size))
    active = distances > 0
    if coefficient == 0 or not np.any(active):
        return output
    wide = np.longdouble
    distance = distances[active].astype(wide)
    factor = wide(coefficient)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled = np.exp(logs.astype(wide)[:, None] * np.log(wide(10)) + np.log(distance)[None, :])
        first, second, third = scaled
        common_linear = factor * first
        common_quadratic = factor * second**2 / 2
        composite_quadratic = (factor * second)**2 / 2
        common_cubic = factor * third**3 / 6
        composite_cubic = (factor * third)**3 / 6
        output[:, active] = np.asarray(np.vstack((
            1 + common_linear + composite_quadratic + common_cubic,
            1 + common_linear + common_quadratic + common_cubic,
            1 + common_linear + composite_quadratic + composite_cubic,
            1 + common_linear + factor * second * distance / 2 + factor * third * distance**2 / 6,
        )), dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("Profiles are not representable as finite float64 values")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Eight regimes cover launch, sign, scaling, grids, cancellation, and high-order separation."""
    return [{'setup': 'a=np.array([-1.,-1.,-1.]);z=np.array([0.])', 'call': 'dispersion_profile_candidates(a,-1.0,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-1.0,z)'},
        {'setup': 'a=np.array([-1.2,-1.3,-1.4]);z=np.array([1.,3.,5.])', 'call': 'dispersion_profile_candidates(a,-1.0,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-1.0,z)'},
        {'setup': 'a=np.array([-2.,-2.,-2.]);z=np.array([0.,2.,4.,6.])', 'call': 'dispersion_profile_candidates(a,-0.8,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-0.8,z)'},
        {'setup': 'a=np.array([-1.5,-1.1,-1.8]);z=np.array([2.,4.,8.,10.])', 'call': 'dispersion_profile_candidates(a,-1.2,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-1.2,z)'},
        {'setup': 'a=np.array([-3.,-2.5,-2.]);z=np.array([0.5,1.5])', 'call': 'dispersion_profile_candidates(a,0.5,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,0.5,z)'},
        {'setup': 'a=np.array([0.,-3.,-6.]);z=np.array([0.01,0.1,1.])', 'call': 'dispersion_profile_candidates(a,-0.05,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-0.05,z)'},
        {'setup': 'a=np.array([-1.47,-1.82,-1.63]);z=np.linspace(.15,8.65,19)', 'call': 'dispersion_profile_candidates(a,-.97,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-.97,z)'},
        {'setup': 'a=np.array([-.83,-1.31,-1.69]);z=np.array([.35,2.7,7.1,9.4])', 'call': 'dispersion_profile_candidates(a,-.89,z)', 'gold_call': '_oracle_dispersion_profile_candidates(a,-.89,z)'},
        {'setup': '', 'call': 'dispersion_profile_candidates(np.array([400.,400.,400.]),0.,np.array([0.,1.]))', 'gold_call': '_oracle_dispersion_profile_candidates(np.array([400.,400.,400.]),0.,np.array([0.,1.]))'},
        {'setup': '', 'call': 'dispersion_profile_candidates(np.array([400.,400.,400.]),-1.,np.array([0.]))', 'gold_call': '_oracle_dispersion_profile_candidates(np.array([400.,400.,400.]),-1.,np.array([0.]))'}]
