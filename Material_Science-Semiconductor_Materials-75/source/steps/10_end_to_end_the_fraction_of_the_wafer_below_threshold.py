"""
Run the whole pipeline and return the percentage of the wafer whose valley splitting falls below the threshold when both the germanium matrix element and the polar vector vary from dot to dot.

This final step chains the whole pipeline and reports the number the task asks for.

The two low-lying valley states of a dot are described by a two-by-two Hamiltonian in valley space,
and its splitting is twice the length of the vector of its three valley components. The
valley-magnetic field of step 08 builds that vector from the strain, the dot geometry of step 07,
the polar vector, the vertical electric field and the applied magnetic field, each channel admitted
or excluded by the classification of step 06. The alloy adds a random term on the operators a real
scalar potential can reach, and the polar vector of each dot scatters about its built-in value.

The inputs describe the device: the two confinement energies and the orientation of the dot, the
in-plane effective mass and the lattice constant, the magnitude and in-plane angle of the magnetic
field and its out-of-plane component, the magnitude and angle of the built-in polar vector, the
vertical electric field, the strain tensor and the twenty-nine coupling constants, and the wafer:
the spread of the germanium matrix element, the spread of each in-plane component of the polar
vector and the splitting threshold. Which operator is closed to the alloy is not an input; it is
read from the time-reversal parities of step 05.

Report the percentage of the wafer whose valley splitting falls below the threshold, converged to a
relative precision of 1e-9.

Returns
-------
float: the percentage of the wafer whose valley splitting falls below the threshold
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def valley_low_tail_percent(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float,
                            m_t_rel: float, a_ang: float, B_inplane_T: float,
                            theta_B_deg: float, Bz_T: float, P0: float, theta_P_deg: float,
                            Fz_MV_per_m: float, strain: np.ndarray, coeffs: np.ndarray,
                            sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> float:
    '''Percentage of the wafer whose valley splitting falls below a threshold.

    Parameters
    ----------
    hw_maj_meV : float
        Confinement energy along the dot major axis, in millielectron volts.
    hw_min_meV : float
        Confinement energy along the dot minor axis, in millielectron volts.
    alpha_deg : float
        Angle of the dot major axis to the crystal x axis, in degrees.
    m_t_rel : float
        In-plane effective mass in units of the free electron mass.
    a_ang : float
        Cubic lattice constant in angstrom.
    B_inplane_T : float
        Magnitude of the in-plane magnetic field, in tesla.
    theta_B_deg : float
        Angle of the in-plane magnetic field to the crystal x axis, in degrees.
    Bz_T : float
        Out-of-plane component of the magnetic field, in tesla.
    P0 : float
        Magnitude of the built-in polar vector.
    theta_P_deg : float
        Angle of the built-in polar vector to the crystal x axis, in degrees.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three real symmetric strain tensor, dimensionless.
    coeffs : np.ndarray
        Length-29 real array of coupling constants in microelectron volts per unit of the
        corresponding candidate term, indexed as in step 06.
    sigma_ge_ueV : float
        Standard deviation of each real component of the germanium matrix element, in
        microelectron volts.
    sigma_P : float
        Standard deviation of each in-plane component of the random polar vector.
    e_threshold_ueV : float
        Splitting threshold in microelectron volts.

    Returns
    -------
    result : float
        The percentage of the wafer below the threshold, converged to a relative precision of
        1e-9.

    Raises
    ------
    ValueError
        Propagated from the earlier steps for invalid dot, coupling or wafer data.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_valley_low_tail_percent(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float, m_t_rel: float, a_ang: float, B_inplane_T: float, theta_B_deg: float, Bz_T: float, P0: float, theta_P_deg: float, Fz_MV_per_m: float, strain: np.ndarray, coeffs: np.ndarray, sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> float:
    import numpy as np
    shell = _oracle_plane_wave_shell_at_X(5, 4)
    if shell.size != 24:
        raise ValueError("the conduction-band shell at X must carry eight plane waves")
    group = _oracle_wave_vector_group_at_X(2)
    basis = _oracle_valleyor_basis(shell, group)
    rep = _oracle_valley_representation(basis, shell, group)
    tau = _oracle_tau_symmetry_data(rep, basis, shell)
    allowed = _oracle_allowed_valley_channels(group, tau)
    geom = _oracle_dot_geometry(hw_maj_meV, hw_min_meV, alpha_deg, m_t_rel, a_ang, Bz_T)
    tb = np.deg2rad(theta_B_deg)
    tp = np.deg2rad(theta_P_deg)
    B_vec = np.array([B_inplane_T * np.cos(tb), B_inplane_T * np.sin(tb), Bz_T])
    P_vec = np.array([P0 * np.cos(tp), P0 * np.sin(tp), 0.0])
    parity = np.rint(np.asarray(tau, dtype=float).ravel()[96:99]).astype(int)
    stats = _oracle_wafer_valley_statistics(coeffs, allowed, parity, P_vec, B_vec, Fz_MV_per_m,
                                            strain, geom, sigma_ge_ueV, sigma_P, e_threshold_ueV)
    return float(stats[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP = ("import numpy as np\n"
              "_c = np.array([4.20e4, 3.10e4, 2.50e3, 1.90e3, 1.80e5, 5.60e1, 4.40e1, 3.90e1, 2.80e1, 7.30e3,\n"
              "               6.10e3, 8.80e3, 9.40e3, 2.15e1, 1.75e1, 6.00e4, 1.40e3, 1.10e3, 9.00e2, 1.30e2,\n"
              "               4.70e1, 3.50e1, 3.00e1, 4.00e1, 1.20e1, 1.20e4, 1.50e5, 1.20e5, 6.00e1])\n"
              "_E = np.array([[3.0e-3, 6.0e-4, 2.0e-4], [6.0e-4, 1.8e-3, 1.0e-4], [2.0e-4, 1.0e-4, -1.2e-3]])")
    return [
        {
            "setup": _SETUP,
            "call": 'np.array([valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.30, 240.0)])',
            "gold_call": 'np.array([_oracle_valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.30, 240.0)])',
            "tol": 1e-08,
        },  # normal, the device and wafer of the task
        {
            "setup": _SETUP,
            "call": 'np.array([valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.0, 240.0)])',
            "gold_call": 'np.array([_oracle_valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.0, 240.0)])',
            "tol": 1e-08,
        },  # edge, every dot carries exactly the built-in polar vector
        {
            "setup": _SETUP,
            "call": 'np.array([valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 1.00, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.30, 240.0)])',
            "gold_call": 'np.array([_oracle_valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 1.00, 1.00, 45.0, 6.0, _E, _c, 90.0, 0.30, 240.0)])',
            "tol": 1e-08,
        },  # boundary, a strongly tilted field that enlarges the tilt channel and the orbital correction
        {
            "setup": _SETUP,
            "call": 'np.array([valley_low_tail_percent(1.30, 1.30, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 0.0, _E, _c, 45.0, 0.30, 200.0)])',
            "gold_call": 'np.array([_oracle_valley_low_tail_percent(1.30, 1.30, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 0.0, _E, _c, 45.0, 0.30, 200.0)])',
            "tol": 1e-08,
        },  # normal, a circular dot in a cleaner alloy with no gate field, where the momentum and Fz channels switch off
        {
            "setup": _SETUP,
            "call": 'np.array([valley_low_tail_percent(0.60, 1.20, 60.0, 0.19, 5.431, 1.20, 30.0, 0.40, 0.80, -20.0, 2.0, _E, _c, 70.0, 0.50, 260.0)])',
            "gold_call": 'np.array([_oracle_valley_low_tail_percent(0.60, 1.20, 60.0, 0.19, 5.431, 1.20, 30.0, 0.40, 0.80, -20.0, 2.0, _E, _c, 70.0, 0.50, 260.0)])',
            "tol": 1e-08,
        },  # normal, a soft dot, a rotated field and polar vector, and a wider asymmetry spread
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, -1.0, 0.30, 240.0))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_valley_low_tail_percent(1.10, 1.45, 30.0, 0.19, 5.431, 1.50, 75.0, 0.20, 1.00, 45.0, 6.0, _E, _c, -1.0, 0.30, 240.0))])',
        },  # contract, a non-positive alloy spread must raise ValueError
    ]
