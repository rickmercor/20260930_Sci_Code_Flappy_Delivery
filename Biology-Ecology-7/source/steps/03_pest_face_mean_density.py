"""
Compute the period-averaged density of a specified pest strain on a positive periodic orbit supported entirely on a pest-only boundary face. The calculation must use the pulse-adjusted periodic balance of the selected face rather than substituting an equilibrium density from the unsprayed model. Return the average density over one complete inter-spray interval for the requested resident strain, while validating that the selected face actually supports a positive periodic state.

On a pest-only boundary face, the parasitoid population is absent and the remaining strains interact through density-dependent Lotka–Volterra competition. Periodicity imposes a balance between continuous per-capita growth and the logarithmic effect of the pulse. Integrating the resident equations over one period therefore produces relationships among the period-averaged densities, intrinsic growth rates, competition coefficients, and pulse effects. These identities are specific to the pulsed periodic orbit and should not be replaced by the equilibrium conditions of the corresponding untreated system.

Returns
-------
the finite period-averaged density (1/tau) * integral_0^tau N_which(t) dt for the requested resident pest strain, as a single float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pest_face_mean_density(params: dict, present: object, tau: float, which: int) -> float:
    """Period-averaged density of a pest strain on the tau-periodic orbit of a pest-only face.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h' as in the other steps).
    present : sequence of int
        Indices (subset of {0, 1, 2}) of the strains present on the face; non-empty, no repeats.
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
        Index of the strain whose period-averaged density (1/tau) int_0^tau N_which dt is returned.

    Returns
    -------
    float
        The period-averaged density of strain `which` on the orbit of that face.

    Raises
    ------
    ValueError
        On invalid inputs, if `which` is not in `present`, if the parasitoid index 3 is in `present`,
        or if the face carries no positive tau-periodic orbit (the averaged densities are not all positive).
    """

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

import math
import numpy as np


def _s03_params(params):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in ("a", "B", "h"):
        if k not in params:
            raise ValueError("missing parameter %s" % k)
    a = np.array(params["a"], dtype=float).ravel()
    B = np.array(params["B"], dtype=float)
    h = np.array(params["h"], dtype=float).ravel()
    if a.shape != (3,) or B.shape != (3, 3) or h.shape != (4,):
        raise ValueError("expected a of length 3, B of shape 3x3 and h of length 4")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(B)) and np.all(np.isfinite(h))):
        raise ValueError("non-finite parameter")
    if np.any(a <= 0.0) or np.any(np.diag(B) <= 0.0) or np.any(B < 0.0):
        raise ValueError("a_i and B_ii must be positive, B_ij non-negative")
    if np.any(h <= -1.0):
        raise ValueError("pulse factors must satisfy h > -1")
    return a, B, h


def _oracle_pest_face_mean_density(params: dict, present: object, tau: float, which: int) -> float:
    a, B, h = _s03_params(params)
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    T = [int(i) for i in present]
    if len(T) == 0 or len(set(T)) != len(T) or any(i < 0 or i > 2 for i in T):
        raise ValueError("present must be a non-empty set of distinct strain indices in {0, 1, 2}")
    which = int(which)
    if which not in T:
        raise ValueError("which must be one of the present strains")
    at = a + np.log1p(h[:3]) / tau
    mean = np.linalg.solve(B[np.ix_(T, T)], at[T])
    if np.any(mean <= 0.0):
        raise ValueError("this face carries no positive tau-periodic orbit (a period average would be non-positive)")
    return float(mean[T.index(which)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [0, 1, 2], 2.0, 0)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [0, 1, 2], 2.0, 0)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [0, 1, 2], 2.0, 1)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [0, 1, 2], 2.0, 1)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [2], 2.0, 2)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [2], 2.0, 2)',
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZ0 = dict(PZ, h=[0.0, 0.0, 0.0, 0.0])",
            "call": 'pest_face_mean_density(PZ0, [0, 1, 2], 2.0, 2)',
            "gold_call": '_oracle_pest_face_mean_density(PZ0, [0, 1, 2], 2.0, 2)',
        },
    ]
