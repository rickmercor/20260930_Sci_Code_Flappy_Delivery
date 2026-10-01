"""
Return the threshold surface charge density, defined as the infimum above which the counterions' mean adsorption distances stay strictly ordered by charge per unit volume up to the top of a given charge window.

At a positively charged wall the counterions are the species of negative valency. Close to a strongly charged wall they are expected to stack outward in order of decreasing charge per unit volume, alpha_i = |z_i| / v_i, so that the counterion with the largest alpha has the smallest mean adsorption distance of step 05 and the one with the smallest alpha the largest. That order is an argument about saturated layers. At weaker charge the counterions may sit in a different order, and as the charge grows they reach the stratified order through exchanges of the kind located in step 06.



The electrolyte is specified by the signed valencies and relative volumes of all species and by the reservoir volume fractions of all but the last species, whose fraction is fixed by electroneutrality as in step 02. Over the window of surface charge densities from sigma_lo to sigma_hi, given in elementary charges per square nanometre, return the threshold sigma*: the infimum such that the counterions' mean adsorption distances increase strictly as alpha decreases at every charge satisfying sigma* < sigma <= sigma_hi. If that strict order already holds across the whole closed window, return sigma_lo. The answer must be accurate to a relative 1e-8. Within the windows used here, no two exchanges of the same pair of counterions lie within a factor 1.2 of each other in charge.

Returns
-------
float: the onset surface charge density of stratification ordered by charge per volume, in e per square nanometre
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stratification_onset(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray", sigma_lo_e_per_nm2: float, sigma_hi_e_per_nm2: float) -> float:
    '''Threshold surface charge density for counterion stratification ordered by charge per volume.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species; at least two species
        have negative valency, and the last species closes electroneutrality.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_free : np.ndarray
        Real array of length N - 1 holding the reservoir volume fractions of the first N - 1
        species; every entry strictly positive.
    sigma_lo_e_per_nm2 : float
        Lower end of the charge window in elementary charges per square nanometre; strictly
        positive.
    sigma_hi_e_per_nm2 : float
        Upper end of the charge window in elementary charges per square nanometre; larger than
        sigma_lo_e_per_nm2.

    Returns
    -------
    sigma_star : float
        The threshold (infimum) in elementary charges per square nanometre such that the strict
        order holds for every sigma satisfying sigma_star < sigma <= sigma_hi_e_per_nm2. If the
        strict order holds across the whole closed window, returns sigma_lo_e_per_nm2.

    Raises
    ------
    ValueError
        If fewer than two species have negative valency, if two counterions share the same alpha,
        if the window is not 0 < sigma_lo < sigma_hi, if the counterions are not ordered by
        decreasing alpha at sigma_hi, or if the reservoir cannot be closed as in step 02.
    '''
    return sigma_star  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stratification_onset(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray", sigma_lo_e_per_nm2: float, sigma_hi_e_per_nm2: float) -> float:
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    lo = float(sigma_lo_e_per_nm2)
    hi = float(sigma_hi_e_per_nm2)
    if not (0.0 < lo < hi):
        raise ValueError("the window must satisfy 0 < sigma_lo < sigma_hi")
    n = zz.size
    pb = _oracle_reservoir_composition(zz, vv, phi_free)[:n]
    counter = np.where(zz < 0.0)[0]
    if counter.size < 2:
        raise ValueError("at least two counterion species are needed")
    alpha = np.abs(zz[counter]) / vv[counter]
    if np.unique(alpha).size != alpha.size:
        raise ValueError("counterions with equal charge per volume have no stratified order")
    inner_to_outer = counter[np.argsort(-alpha)]

    # the order is strict iff every neighbouring pair in alpha order is strict, so the onset is
    # the last exchange of any neighbouring pair inside the window
    s_lo, s_hi = lo / 100.0, hi / 100.0
    n_grid = int(np.ceil(np.log(s_hi / s_lo) / np.log(1.2))) + 1
    grid = np.geomspace(s_lo, s_hi, n_grid)
    X = np.array([_oracle_adsorption_moments(eps_r, T_K, a_ang, s, zz, vv, pb)[n:2 * n] for s in grid])
    onset = s_lo
    for k in range(inner_to_outer.size - 1):
        inner, outer = int(inner_to_outer[k]), int(inner_to_outer[k + 1])
        gap = X[:, outer] - X[:, inner]
        if gap[-1] <= 0.0:
            raise ValueError("the counterions are not ordered by charge per volume at sigma_hi")
        bad = np.where(gap <= 0.0)[0]
        if bad.size == 0:
            continue
        m = int(bad[-1])
        if gap[m] == 0.0:
            cross = grid[m]
        else:
            cross = _oracle_exchange_charge(eps_r, T_K, a_ang, zz, vv, pb, outer, inner, grid[m], grid[m + 1])
        onset = max(onset, cross)
    return float(onset * 100.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nF = np.array([2.0e-3, 1.0e-3, 5.0e-4])",
            "call": "np.log(stratification_onset(78.5, 298.15, 5.00, Z, V, F, 2.0, 5.0))",
            "gold_call": "np.log(_oracle_stratification_onset(78.5, 298.15, 5.00, Z, V, F, 2.0, 5.0))",
            "tol": 4e-9,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nF = np.array([2.0e-3, 1.0e-3, 5.0e-4])",
            "call": "np.log(stratification_onset(78.5, 298.15, 5.00, Z, V, F, 4.0, 10.0))",
            "gold_call": "np.log(_oracle_stratification_onset(78.5, 298.15, 5.00, Z, V, F, 4.0, 10.0))",
            "tol": 4e-9,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nF = np.array([2.0e-3, 1.0e-3, 5.0e-4])",
            "call": "np.log(stratification_onset(157.0, 149.075, 5.00, Z, V, F, 2.0, 5.0))",
            "gold_call": "np.log(_oracle_stratification_onset(157.0, 149.075, 5.00, Z, V, F, 2.0, 5.0))",
            "tol": 4e-9,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-2.0, 1.0, -1.0])\nV = np.array([6.0, 0.3, 0.8])\nF = np.array([4.0e-3, 1.6e-3])",
            "call": "np.log(stratification_onset(40.0, 330.0, 4.50, Z, V, F, 0.2, 3.0))",
            "gold_call": "np.log(_oracle_stratification_onset(40.0, 330.0, 4.50, Z, V, F, 0.2, 3.0))",
            "tol": 4e-9,
        },
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nF = np.array([2.0e-3, 1.0e-3, 5.0e-4])",
            "call": "_raises(lambda: stratification_onset(78.5, 298.15, 5.00, Z, V, F, 0.5, 2.0))",
            "gold_call": "_raises(lambda: _oracle_stratification_onset(78.5, 298.15, 5.00, Z, V, F, 0.5, 2.0))",
        },
    ]
