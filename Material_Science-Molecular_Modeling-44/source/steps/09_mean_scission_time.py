"""
Orchestrate the whole pipeline in SI units and return the mean time to the first scission, 1 / k_chain in hours, of an n_bonds chain of atoms of mass `mass`, anchored at atom 0 and held at the end-to-end distance stretch_ratio * R_0 (R_0 = N l_e times the sine of half the equilibrium valence angle, the length of the undeformed zigzag). Every bond stores the Morse stretching energy of the earlier steps, and every interior atom stores a harmonic bending energy k_phi (theta_v - theta_v_e)^2 / 2 in its valence angle theta_v, the angle between the two bonds meeting at that atom, with equilibrium valence angle valence_angle_deg. Convert to reduced units (beta_de = d_e / (k_b T), a_le = a l_e, beta_kphi = k_phi / (k_b T)) and to the angle convention of the earlier steps, obtain the constant chain force from the dFRC step (its optimum converged to 1e-12 in both variables, as that step requires), build a Gauss-Legendre grid of n_theta nodes on [0, pi], take the collinear reference barrier top as the initial threshold, build the bending kernel with n_omega azimuths, converge the self-consistent thresholds, form the intact weights at the converged thresholds, locate every bond's stationary points, tabulate every bond's profile on n_well equally spaced lengths from 0 to its barrier top, evaluate every bond-resolved rate with its kinetic prefactor (source, Sec. S1.3): the one-sided thermal average of the velocity of the bond-length coordinate of that bond, (2 pi m_l / (k_b T))^(-1/2) in m/s with m_l the effective mass of that coordinate, divided by l_e to match the reduced length unit; sum the bond-resolved rates and return the reciprocal of the chain-scission rate in hours. Call the earlier step functions rather than reimplementing them.

The end-to-end quantity is the mean time a loaded chain survives before losing its first backbone bond, the molecular input to network-scale damage models. It is set by the sum of the bond-resolved rupture rates, which differ along the chain because the truncation of orientational correlations at the ends changes the barriers of the bonds near the ends.

Returns
-------
float, the mean time to first scission 1 / k_chain in hours.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_scission_time(n_bonds: int, stretch_ratio: float, d_e: float, a: float, l_e: float, k_phi: float,
                               valence_angle_deg: float, temperature: float, mass: float,
                               k_b: float = 1.380649e-23, n_theta: int = 400, n_omega: int = 512,
                               n_well: int = 20001) -> float:
    """Orchestrate the whole pipeline in SI units and return the mean time to the first scission, 1 / k_chain in hours, of an n_bonds chain of atoms of mass `mass`, anchored at atom 0 and held at the end-to-end distance stretch_ratio * R_0 (R_0 = N l_e times the sine of half the equilibrium valence angle, the length of the undeformed zigzag). Every bond stores the Morse stretching energy of the earlier steps, and every interior atom stores a harmonic bending energy k_phi (theta_v - theta_v_e)^2 / 2 in its valence angle theta_v, the angle between the two bonds meeting at that atom, with equilibrium valence angle valence_angle_deg. Convert to reduced units (beta_de = d_e / (k_b T), a_le = a l_e, beta_kphi = k_phi / (k_b T)) and to the angle convention of the earlier steps, obtain the constant chain force from the dFRC step (its optimum converged to 1e-12 in both variables, as that step requires), build a Gauss-Legendre grid of n_theta nodes on [0, pi], take the collinear reference barrier top as the initial threshold, build the bending kernel with n_omega azimuths, converge the self-consistent thresholds, form the intact weights at the converged thresholds, locate every bond's stationary points, tabulate every bond's profile on n_well equally spaced lengths from 0 to its barrier top, evaluate every bond-resolved rate with its kinetic prefactor (source, Sec. S1.3): the one-sided thermal average of the velocity of the bond-length coordinate of that bond, (2 pi m_l / (k_b T))^(-1/2) in m/s with m_l the effective mass of that coordinate, divided by l_e to match the reduced length unit; sum the bond-resolved rates and return the reciprocal of the chain-scission rate in hours. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n_bonds : int
        Number of bonds N (>= 2); atoms 0..N with atom 0 anchored.
    stretch_ratio : float
        Prescribed end-to-end distance divided by the undeformed zigzag length R_0 (positive).
    d_e : float
        Morse dissociation energy in J.
    a : float
        Morse range parameter in 1/m.
    l_e : float
        Equilibrium bond length in m.
    k_phi : float
        Bending stiffness in J/rad^2 (non-negative).
    valence_angle_deg : float
        Equilibrium valence angle at each interior atom in degrees, in (0, 180).
    temperature : float
        Temperature in K.
    mass : float
        Atomic mass in kg (all atoms).
    k_b : float
        Boltzmann constant in J/K (default 1.380649e-23).
    n_theta : int
        Gauss-Legendre nodes on [0, pi] (default 400).
    n_omega : int
        Midpoint azimuthal nodes of the kernel (default 512).
    n_well : int
        Equally spaced lengths from 0 to the barrier top for the well integral (default 20001).

    Returns
    -------
    tau_chain : float
        Mean time to first scission in hours.

    Raises
    ------
    ValueError
        If n_bonds, n_theta or n_well is not a valid integer, stretch_ratio, d_e, a, l_e, temperature, mass or k_b is not finite positive, k_phi is negative, valence_angle_deg is outside (0, 180), or any earlier step raises (for example a stretch whose chain force lies at or above the critical force).
    """
    return tau_chain

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _oracle_mean_scission_time(n_bonds: int, stretch_ratio: float, d_e: float, a: float, l_e: float, k_phi: float,
                               valence_angle_deg: float, temperature: float, mass: float,
                               k_b: float = 1.380649e-23, n_theta: int = 400, n_omega: int = 512,
                               n_well: int = 20001) -> float:
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 2:
        raise ValueError("n_bonds must be an integer >= 2")
    for name, value in (("stretch_ratio", stretch_ratio), ("d_e", d_e), ("a", a), ("l_e", l_e),
                        ("temperature", temperature), ("mass", mass), ("k_b", k_b)):
        _check_positive(name, value)
    if not np.isfinite(k_phi) or k_phi < 0.0:
        raise ValueError("k_phi must be finite and non-negative")
    if not np.isfinite(valence_angle_deg) or valence_angle_deg <= 0.0 or valence_angle_deg >= 180.0:
        raise ValueError("valence_angle_deg must lie strictly inside (0, 180)")
    if not isinstance(n_theta, (int, np.integer)) or n_theta < 2:
        raise ValueError("n_theta must be an integer >= 2")
    if not isinstance(n_well, (int, np.integer)) or n_well < 3:
        raise ValueError("n_well must be an integer >= 3")
    beta = 1.0 / (k_b * temperature)
    beta_de, a_le = beta * d_e, a * l_e
    beta_kphi = beta * k_phi
    phi_e = np.pi - np.deg2rad(valence_angle_deg)      # angle between successive bond vectors = pi - valence angle
    f_red = _oracle_dfrc_chain_force(stretch_ratio, int(n_bonds), beta_de, a_le, beta_kphi, phi_e)[2]
    pref_anchor = 1.0 / np.sqrt(2.0 * np.pi * mass * beta) / l_e           # bond 1: one mobile atom, mass m (S40)
    pref_inner = 1.0 / np.sqrt(2.0 * np.pi * (mass / 2.0) * beta) / l_e    # bonds >= 2: relative coordinate, m/2
    x, w = np.polynomial.legendre.leggauss(int(n_theta))
    theta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w
    l_bar_1d = _oracle_collinear_reference_rate(f_red, beta_de, a_le, pref_inner)[1]
    kernel = _oracle_bending_kernel(theta, beta_kphi, phi_e, n_omega)
    thresholds = _oracle_self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le,
                                                    int(n_bonds), l_bar_1d)
    log_i = np.array([_oracle_log_intact_weight(theta, thresholds[i], f_red, beta_de, a_le)
                      for i in range(int(n_bonds))])
    total = 0.0
    for i in range(int(n_bonds)):
        l_min, l_bar, _ = _oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, i, f_red, beta_de, a_le)
        well = np.linspace(0.0, l_bar, int(n_well))
        profile = _oracle_constrained_pmf(theta, w_theta, kernel, log_i, i, well, f_red, beta_de, a_le)[0]
        prefactor = pref_anchor if i == 0 else pref_inner
        total += _oracle_bond_tst_rate(well, profile, beta_de, prefactor)
    return float(1.0 / total / 3600.0)                                       # mean time to first scission in hours

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nn_bonds, stretch_ratio = 12, 1.22\nd_e, a, l_e = 1.13e-18, 1.409e10, 1.525e-10\nk_phi, valence_angle_deg = 5.0e-19, 109.5\ntemperature, mass = 293.15, 1.99e-26\n",
            "call": "np.log(mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "gold_call": "np.log(_oracle_mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nn_bonds, stretch_ratio = 12, 1.22\nd_e, a, l_e = 1.13e-18, 1.409e10, 1.525e-10\nk_phi, valence_angle_deg = 5.0e-19, 109.5\ntemperature, mass = 293.15, 1.99e-26\nn_bonds, stretch_ratio = 2, 1.22\n",
            "call": "np.log(mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "gold_call": "np.log(_oracle_mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nn_bonds, stretch_ratio = 12, 1.22\nd_e, a, l_e = 1.13e-18, 1.409e10, 1.525e-10\nk_phi, valence_angle_deg = 5.0e-19, 109.5\ntemperature, mass = 293.15, 1.99e-26\nn_bonds, stretch_ratio, k_phi = 4, 1.15, 2.0e-19\n",
            "call": "np.log(mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "gold_call": "np.log(_oracle_mean_scission_time(n_bonds, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass))",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nn_bonds, stretch_ratio = 12, 1.22\nd_e, a, l_e = 1.13e-18, 1.409e10, 1.525e-10\nk_phi, valence_angle_deg = 5.0e-19, 109.5\ntemperature, mass = 293.15, 1.99e-26\ndef run_model():\n    try:\n        mean_scission_time(1, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mean_scission_time(1, stretch_ratio, d_e, a, l_e, k_phi, valence_angle_deg, temperature, mass)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
