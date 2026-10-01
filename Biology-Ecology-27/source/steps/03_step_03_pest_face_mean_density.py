"""
Period-averaged densities on a pest-only face orbit from the averages identity.

On any face that contains pest strains only, a tau-periodic orbit must satisfy, for every present strain i, a(i) tau + ln(1 + h(i)) = sum over present j of b(ij) times the integral of N(j) over one interval. This step solves that linear system for the period averages (1/tau) int N(j) dt of the present strains and returns the average of strain which. A non-positive solution component means that the face carries no positive tau-periodic orbit, and a ValueError is raised; it is also raised when which is not present or the parasitoid is listed. Deliberately excluded: the orbit itself (its post-spray state needs shooting, step 4) and any face containing the parasitoid (its saturating response has no such identity).

Integrating dN(i)/dt / N(i) = f(i) over one interval of a tau-periodic orbit and adding the spray jump gives the identity int f(i) dt = -ln(1 + h(i)) for every present species; on a pest-only face f(i) is linear in the densities, so the period averages solve B(T) <N> = a(T) + ln(1+h(T))/tau, exactly as in the single-species case where <N> = [a tau + ln(1+h)]/(b tau). Benchmark values at tau = 2 on the three-strain face: <NA> = 0.2823692486639, <NB> = 0.4652515063581, <NC> = 0.2771645763793 (they agree to 1e-13 with the quadrature along the shot orbit). On every two-strain face one component of the solution is negative for all tau in [1.2, 3.0] (cyclic dominance: B excludes A, C excludes B, A excludes C), so no two-strain orbit exists there; on a single-strain face the average is the spray-adjusted capacity a(i)/b(ii) + ln(1+h(i))/(b(ii) tau), e.g. 0.8884282243 for strain C.

Returns
-------
float — period-averaged density of one strain on a pest-only face orbit (0.2823692486639 for strain A on the A-B-C face at tau = 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pest_face_mean_density(params: dict, present: tuple[int, ...], tau: float, which: int) -> float:
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


def _oracle_pest_face_mean_density(params: dict, present, tau: float, which: int) -> float:
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
            "name": 'normal_three_strain_face_strain_A',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [0, 1, 2], 2.0, 0)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [0, 1, 2], 2.0, 0)',
        },
        {
            "name": 'normal_three_strain_face_strain_B',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [0, 1, 2], 2.0, 1)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [0, 1, 2], 2.0, 1)',
        },
        {
            "name": 'variant_single_strain_face_equals_adjusted_capacity',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'pest_face_mean_density(PZ, [2], 2.0, 2)',
            "gold_call": '_oracle_pest_face_mean_density(PZ, [2], 2.0, 2)',
        },
        {
            "name": 'edge_unsprayed_face_average_is_the_equilibrium',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZ0 = dict(PZ, h=[0.0, 0.0, 0.0, 0.0])",
            "call": 'pest_face_mean_density(PZ0, [0, 1, 2], 2.0, 2)',
            "gold_call": '_oracle_pest_face_mean_density(PZ0, [0, 1, 2], 2.0, 2)',
        },
    ]
