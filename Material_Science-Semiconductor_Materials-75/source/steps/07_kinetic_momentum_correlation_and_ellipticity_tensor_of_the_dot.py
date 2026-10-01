"""
Return the ground-state expectation of the symmetrised product of the in-plane kinetic momenta of the elliptical dot in a perpendicular magnetic field, and the ellipticity tensor of the dot.

The device is a gate-defined quantum dot in a strained silicon quantum well, and the electron is
treated as strictly two-dimensional. In the plane the confinement is harmonic and elliptical, with
confinement energies hbar omega_maj along the major axis and hbar omega_min along the minor axis,
the major axis being the weakly confined one. The major axis lies at an angle alpha to the crystal
x axis. A magnetic field component Bz threads the dot perpendicular to the plane; the in-plane
field components have no orbital effect on a strictly two-dimensional electron.

Two objects describing the dot are needed downstream.

The first is the ground-state expectation of kx ky. The effective Hamiltonian at X contains a term
proportional to the product of the two in-plane momenta, often left out of two-valley models of
silicon but of the same order as the kinetic terms. In a magnetic field the momentum in that term
is the kinetic momentum hbar k = p + e A of an electron of charge -e, and because the two kinetic
components no longer commute the product must be taken in its symmetrised, Hermitian form,
(kx ky + ky kx)/2. The result does not depend on the gauge. The in-plane Hamiltonian is still
quadratic, so the ground state is Gaussian, but the field couples the two directions of motion: the
expectation is no longer the zero-field covariance of the two oscillator axes rotated into the
crystal frame, and it has to be obtained from the ground state of the full quadratic Hamiltonian in
phase space. At zero field it reduces to that rotated covariance, it vanishes for a circular dot
and for a dot aligned with the crystal axes, and it is even in Bz. Report it in units of one over
the squared cubic lattice constant, that is as the dimensionless number <kx ky> a^2, converged to a
relative precision of 1e-10.

The second is the ellipticity tensor. The shape asymmetry of an elliptical dot is a director, not a
polar vector, because the two ends of the major axis are equivalent. The order parameter is the
traceless symmetric two-by-two tensor n n^T - I/2 built from the unit vector n along the major
axis. It encodes orientation only, so it does not vanish in the circular limit; it is a supplied
geometric descriptor of the dot rather than a measure of elongation.

Return the momentum correlation followed by Q_xy, Q_xx and Q_yy.

Useful constants: hbar squared over twice the free electron mass is 3.8099821161548593 electron
volt angstrom squared, and hbar e over the free electron mass is 0.11576763 millielectron volt per
tesla.

Returns
-------
np.ndarray of length 4: <kx ky> a^2 (dimensionless), then Q_xy, Q_xx, Q_yy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dot_geometry(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float,
                 m_t_rel: float, a_ang: float, Bz_T: float) -> np.ndarray:
    '''Ground-state kinetic momentum correlation and ellipticity tensor of the elliptical dot.

    Parameters
    ----------
    hw_maj_meV : float
        Confinement energy along the major axis in millielectron volts; positive and not larger
        than hw_min_meV.
    hw_min_meV : float
        Confinement energy along the minor axis in millielectron volts; positive.
    alpha_deg : float
        Angle of the major axis to the crystal x axis, in degrees.
    m_t_rel : float
        In-plane effective mass in units of the free electron mass; positive.
    a_ang : float
        Cubic lattice constant in angstrom; positive.
    Bz_T : float
        Magnetic field component perpendicular to the plane of the dot, in tesla.

    Returns
    -------
    result : np.ndarray
        Real array of length 4 holding the ground-state expectation of (kx ky + ky kx)/2 for the
        kinetic momenta, times the squared lattice constant, converged to a relative precision of
        1e-10, then the xy, xx and yy entries of the ellipticity tensor.

    Raises
    ------
    ValueError
        If either confinement energy is not positive, if hw_maj_meV exceeds hw_min_meV, or if the
        effective mass or lattice constant is not positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dot_geometry(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float, m_t_rel: float, a_ang: float, Bz_T: float) -> np.ndarray:
    import numpy as np
    if hw_maj_meV <= 0.0 or hw_min_meV <= 0.0:
        raise ValueError("confinement energies must be positive")
    if hw_maj_meV > hw_min_meV:
        raise ValueError("the major axis is the weakly confined one, so hw_maj cannot exceed hw_min")
    if m_t_rel <= 0.0 or a_ang <= 0.0:
        raise ValueError("effective mass and lattice constant must be positive")
    hbar2_2m0 = 3.8099821161548593      # eV A^2
    hbar_e_over_m0 = 0.11576763         # meV per tesla
    hm = 2.0e3 * hbar2_2m0 / m_t_rel    # hbar^2/m in meV A^2
    w0 = np.sqrt(hw_maj_meV * hw_min_meV)
    ell2 = hm / w0                       # squared oscillator length in A^2
    al = np.deg2rad(alpha_deg)
    R = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
    # dimensionless quadratic form: lengths in ell, momenta in 1/ell, energies in w0
    K = R @ np.diag([hw_maj_meV ** 2, hw_min_meV ** 2]) @ R.T / w0 ** 2
    b = hbar_e_over_m0 * Bz_T / m_t_rel / w0           # eB ell^2 / hbar
    # (x, y, pi_x, pi_y) = T (x, y, p_x, p_y) in the symmetric gauge
    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                  [0.0, -0.5 * b, 1.0, 0.0], [0.5 * b, 0.0, 0.0, 1.0]])
    Mpi = np.zeros((4, 4))
    Mpi[:2, :2] = K
    Mpi[2:, 2:] = np.eye(2)
    M = T.T @ Mpi @ T
    J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    w, V = np.linalg.eigh(M)
    Mh = V @ np.diag(np.sqrt(w)) @ V.T
    Mih = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    N = Mh @ J @ Mh
    wn, Vn = np.linalg.eigh(N.T @ N)
    absN = Vn @ np.diag(np.sqrt(np.clip(wn, 0.0, None))) @ Vn.T
    G = 0.5 * Mih @ absN @ Mih          # ground-state covariance, symmetrised
    Gpi = T @ G @ T.T
    kxky = Gpi[2, 3] / ell2 * a_ang ** 2
    nvec = np.array([np.cos(al), np.sin(al)])
    Q = np.outer(nvec, nvec) - 0.5 * np.eye(2)
    return np.array([kxky, Q[0, 1], Q[0, 0], Q[1, 1]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(1.10, 1.45, 30.0, 0.19, 5.431, 0.20)',
            "gold_call": '_oracle_dot_geometry(1.10, 1.45, 30.0, 0.19, 5.431, 0.20)',
        },  # normal, the device of the task
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(1.10, 1.45, 30.0, 0.19, 5.431, 0.0)',
            "gold_call": '_oracle_dot_geometry(1.10, 1.45, 30.0, 0.19, 5.431, 0.0)',
        },  # edge, zero field, where the rotated oscillator covariance is exact
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(0.80, 2.10, 112.5, 0.32, 5.658, 4.0)',
            "gold_call": '_oracle_dot_geometry(0.80, 2.10, 112.5, 0.32, 5.658, 4.0)',
        },  # normal, a strong field that visibly mixes the two oscillator axes, at an obtuse orientation
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(0.80, 2.10, 112.5, 0.32, 5.658, -4.0)',
            "gold_call": '_oracle_dot_geometry(0.80, 2.10, 112.5, 0.32, 5.658, -4.0)',
        },  # boundary, reversing the field leaves the symmetrised correlation unchanged
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(1.10, 1.45, 0.0, 0.19, 5.431, 1.0)',
            "gold_call": '_oracle_dot_geometry(1.10, 1.45, 0.0, 0.19, 5.431, 1.0)',
        },  # edge, axes aligned with the crystal, so the correlation vanishes even in a field
        {
            "setup": "import numpy as np",
            "call": 'dot_geometry(1.30, 1.30, 30.0, 0.19, 5.431, 0.5)',
            "gold_call": '_oracle_dot_geometry(1.30, 1.30, 30.0, 0.19, 5.431, 0.5)',
        },  # edge, a circular dot, where the ellipticity tensor does not vanish
        {
            "setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: dot_geometry(1.60, 1.20, 30.0, 0.19, 5.431, 0.2)), _raises(lambda: dot_geometry(1.10, 1.45, 30.0, 0.0, 5.431, 0.2)), _raises(lambda: dot_geometry(-1.10, 1.45, 30.0, 0.19, 5.431, 0.2)), _raises(lambda: dot_geometry(1.10, 1.45, 30.0, 0.19, -5.431, 0.2))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_dot_geometry(1.60, 1.20, 30.0, 0.19, 5.431, 0.2)), _raises(lambda: _oracle_dot_geometry(1.10, 1.45, 30.0, 0.0, 5.431, 0.2)), _raises(lambda: _oracle_dot_geometry(-1.10, 1.45, 30.0, 0.19, 5.431, 0.2)), _raises(lambda: _oracle_dot_geometry(1.10, 1.45, 30.0, 0.19, -5.431, 0.2))])',
        },  # contract, inverted axes, a non-positive mass, a negative energy and a negative lattice constant must raise ValueError
    ]
