"""
Return the adsorbed amount per unit area and the mean adsorption distance of every ionic species next to a positively charged planar wall, together with the charge-balance ratio of the layer and the reduced wall potential.

A planar wall with uniform positive surface charge density sigma faces the size-asymmetric lattice electrolyte of steps 03 and 04, in a solvent of relative permittivity eps_r at temperature T with lattice cell edge a, and in contact with an electroneutral reservoir far away. The wall is described by the reduced surface parameter of step 01, the composition at any potential by step 03 and the wall potential by step 04; together with the uniform pressure of step 04 they fix the potential and the composition at every distance x from the wall.

For species i, with number density c_i(x) = phi_i(x) / (a**3 v_i) and reservoir value c_i^b, the adsorbed amount per unit area is

Gamma_i = integral over x from 0 to infinity of [c_i(x) - c_i^b] dx,

positive for species drawn to the wall and negative for species pushed away, and the mean adsorption distance is

<x>_i = (1 / Gamma_i) integral over x from 0 to infinity of x [c_i(x) - c_i^b] dx.

The diffuse layer carries the charge that balances the wall, so the valency-weighted sum of the adsorbed amounts equals -sigma / e; the ratio of that sum to -sigma / e is returned as a check. The profiles approach the reservoir only asymptotically and do not end at any finite distance. Every returned value must be converged to a relative accuracy of 1e-9.

Returns
-------
np.ndarray of length 2 N + 2: the N adsorbed amounts (inverse square angstrom), the N mean adsorption distances (angstrom), the charge-balance ratio, then the reduced wall potential
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adsorption_moments(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray") -> "np.ndarray":
    '''Adsorbed amounts and mean adsorption distances of a size-asymmetric electrolyte at a charged wall.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    sigma_e_per_A2 : float
        Surface charge density of the wall in elementary charges per square angstrom; strictly
        positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one, and at least
        one species of negative valency.

    Returns
    -------
    result : np.ndarray
        Real array of length 2 N + 2 holding the N adsorbed amounts in inverse square angstrom,
        then the N mean adsorption distances in angstrom, both in the input species order, then the
        ratio of sum_i z_i Gamma_i to -sigma / e, then the reduced wall potential e psi(0) / (k_B T).

    Raises
    ------
    ValueError
        If sigma_e_per_A2 is not strictly positive, or if eps_r, T_K or a_ang is not strictly
        positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _moment_rhs(s, y, zz, vv, pb, a3, kfac):
    """d/ds of the distance from the wall and of the running moments, with s = ln(Psi)."""
    n = zz.size
    P = np.exp(s)
    st = _oracle_local_composition(zz, vv, pb, P)
    dxds = -P / np.sqrt(kfac * _squared_field(zz, vv, pb, P))
    exc = (st[:n] - pb) / (a3 * vv)
    return np.concatenate([[dxds], exc * dxds, y[0] * exc * dxds])


def _oracle_adsorption_moments(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray") -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    sigma = float(sigma_e_per_A2)
    if sigma <= 0.0:
        raise ValueError("the surface charge density must be strictly positive")
    lB, a3, zeta = _oracle_electrostatic_scales(eps_r, T_K, a_ang, sigma)
    n = zz.size
    Ps = float(_oracle_wall_state(zz, vv, pb, zeta)[0])
    kfac = 8.0 * np.pi * lB / a3
    psi_far = 1e-7
    sol = solve_ivp(_moment_rhs, (np.log(Ps), np.log(psi_far)), np.zeros(1 + 2 * n),
                    method="DOP853", rtol=1e-12, atol=1e-20, args=(zz, vv, pb, a3, kfac))
    y = sol.y[:, -1]
    x_far = y[0]
    G = y[1:1 + n].copy()
    M = y[1 + n:].copy()
    # beyond psi_far the field and every excess are linear in Psi, which decays as exp(-kappa x)
    kappa = np.sqrt(kfac * _squared_field(zz, vv, pb, psi_far)) / psi_far
    slope = (_oracle_local_composition(zz, vv, pb, psi_far)[:n] - pb) / (a3 * vv) / psi_far
    G += slope * psi_far / kappa
    M += slope * psi_far * (x_far / kappa + 1.0 / kappa ** 2)
    ratio = float((zz * G).sum() / (-sigma))
    return np.concatenate([G, M / G, [ratio, Ps]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
def _relative_vector(value):
    value = np.asarray(value, dtype=float)
    flags = (value != 0.0).astype(float)
    signs = np.sign(value)
    safe = np.maximum(np.abs(value), np.finfo(float).tiny)
    return np.concatenate((flags, signs, np.log(safe)))
"""
    return [
        {
            "setup": helper + "Z = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "_relative_vector(adsorption_moments(78.5, 298.15, 5.00, 0.0325, Z, V, PB))",
            "gold_call": "_relative_vector(_oracle_adsorption_moments(78.5, 298.15, 5.00, 0.0325, Z, V, PB))",
            "tol": 5e-11,
        },
        {
            "setup": helper + "Z = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "_relative_vector(adsorption_moments(78.5, 298.15, 5.00, 0.10, Z, V, PB))",
            "gold_call": "_relative_vector(_oracle_adsorption_moments(78.5, 298.15, 5.00, 0.10, Z, V, PB))",
            "tol": 5e-11,
        },
        {
            "setup": helper + "Z = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-5, 1.0e-5, 5.0e-6, 5.1875e-5])",
            "call": "_relative_vector(adsorption_moments(78.5, 298.15, 5.00, 0.010, Z, V, PB))",
            "gold_call": "_relative_vector(_oracle_adsorption_moments(78.5, 298.15, 5.00, 0.010, Z, V, PB))",
            "tol": 5e-11,
        },
        {
            "setup": helper + "Z = np.array([-1.0, 1.0])\nV = np.array([1.0, 1.0])\nPB = np.array([0.005, 0.005])",
            "call": "_relative_vector(adsorption_moments(78.5, 298.15, 4.00, 0.025, Z, V, PB))",
            "gold_call": "_relative_vector(_oracle_adsorption_moments(78.5, 298.15, 4.00, 0.025, Z, V, PB))",
            "tol": 5e-11,
        },
        {
            "setup": helper + "Z = np.array([-2.0, 1.0, -1.0])\nV = np.array([6.0, 0.3, 0.8])\nPB = np.array([4.0e-3, 1.6e-3, 3.2e-3])",
            "call": "_relative_vector(adsorption_moments(40.0, 330.0, 4.50, 0.030, Z, V, PB))",
            "gold_call": "_relative_vector(_oracle_adsorption_moments(40.0, 330.0, 4.50, 0.030, Z, V, PB))",
            "tol": 5e-11,
        },
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nZ = np.array([-1.0, 1.0])\nV = np.array([1.0, 1.0])\nPB = np.array([0.005, 0.005])",
            "call": "_raises(lambda: adsorption_moments(78.5, 298.15, 4.00, 0.0, Z, V, PB))",
            "gold_call": "_raises(lambda: _oracle_adsorption_moments(78.5, 298.15, 4.00, 0.0, Z, V, PB))",
        },
    ]
