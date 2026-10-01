"""
Sum the allowed channels with their coupling constants into the three components of the valley-magnetic field of a dot with a given polar vector.

The classification of step 06 says which background combinations are permitted to multiply which
valley operator. Turning that into numbers requires the device data: the polar vector of the dot,
the applied magnetic field, the vertical electric field, the background strain, and the dot
geometry of step 07. Each candidate term is evaluated on those data, multiplied by its supplied
coupling constant, and added into the component of the valley-magnetic field belonging to the
valley operator it is allowed to couple to. Terms allowed only with the identity shift both valleys
together and are therefore not part of any component; terms allowed with none of the four are
simply absent, however large their coupling constants.

The result is a three-component object that enters the valley Hamiltonian the way a magnetic field
enters the Zeeman Hamiltonian of a spin, even though the valley operators do not transform as a
spin. Its components are built from physically different objects: some are pure strain or pure
geometry and survive at zero applied field, some are linear in the magnetic field, some are
quadratic in it, some need the field tilted out of the plane, and some depend on the polar vector
of the dot, which later steps will allow to vary from dot to dot.

The candidate list and its index order are exactly those of step 06. The coupling constants are
supplied as one number per candidate term, in microelectron volts per unit of the corresponding
combination, with magnetic fields in tesla, Fz in megavolts per metre, and kx ky as the
dimensionless number of step 07, so no further unit conversion is needed.

Returns
-------
np.ndarray of length 3: the valley-magnetic components B1, B2, B3 in microelectron volts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def valley_magnetic_components(coeffs: np.ndarray, allowed: np.ndarray, P_vec: np.ndarray,
                               B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray,
                               dot_geom: np.ndarray) -> np.ndarray:
    '''The three components of the valley-magnetic field, in microelectron volts.

    Parameters
    ----------
    coeffs : np.ndarray
        Length-29 real array of coupling constants, one per candidate term, in microelectron
        volts per unit of that term.
    allowed : np.ndarray
        Output of step 06: length-116 real array holding the twenty-nine by four table.
    P_vec : np.ndarray
        Length-3 real array holding the in-plane polar vector of the dot; its z entry is unused.
    B_vec : np.ndarray
        Length-3 real array holding the magnetic field in tesla.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three real symmetric strain tensor, dimensionless.
    dot_geom : np.ndarray
        Output of step 07: length-4 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 3 holding the components of the valley-magnetic field belonging to
        valley operators 1, 2 and 3, in microelectron volts.

    Raises
    ------
    ValueError
        If coeffs does not hold exactly twenty-nine entries.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _term_vector(P, B, E, Q, kxky, Fz):
    import numpy as np
    Px, Py, Pz = P
    Bx, By, Bz = B
    exx, eyy, ezz = E[0, 0], E[1, 1], E[2, 2]
    exy, exz, eyz = E[0, 1], E[0, 2], E[1, 2]
    Qxy, Qxx, Qyy = Q[0, 1], Q[0, 0], Q[1, 1]
    return np.array([
        exy, exx - eyy, ezz, exx + eyy, kxky,
        Px * By - Py * Bx, Px * Bx - Py * By, Px * By + Py * Bx, Px * Bx + Py * By,
        (Px * By - Py * Bx) * ezz, (Px * By - Py * Bx) * (exx + eyy),
        (Px * Bx - Py * By) * exy, (Px * By + Py * Bx) * (exx - eyy),
        Bz * (Px * By + Py * Bx), Bz * (Px * Bx + Py * By), Bz * (exx - eyy),
        Bx * exz + By * eyz, Bx * eyz + By * exz, Bz * exy,
        Qxy, (Px * Bx - Py * By) * Qxy, (Px * By + Py * Bx) * (Qxx - Qyy),
        Px * Py, Bx * By, Fz, Fz * exy, Px * eyz + Py * exz, Px * exz + Py * eyz,
        Bz * (Bx * Bx - By * By)], dtype=float)


def _oracle_valley_magnetic_components(coeffs: np.ndarray, allowed: np.ndarray, P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray, dot_geom: np.ndarray) -> np.ndarray:
    import numpy as np
    c = np.asarray(coeffs, dtype=float).ravel()
    if c.size != 29:
        raise ValueError("coeffs must hold exactly one constant per candidate term")
    A = np.rint(np.asarray(allowed, dtype=float)).astype(int).reshape(29, 4)
    E = np.asarray(strain, dtype=float).reshape(3, 3)
    g = np.asarray(dot_geom, dtype=float).ravel()
    Q = np.array([[g[2], g[1], 0.0], [g[1], g[3], 0.0], [0.0, 0.0, 0.0]])
    X = _term_vector(np.asarray(P_vec, dtype=float), np.asarray(B_vec, dtype=float), E, Q, g[0],
                     float(Fz_MV_per_m))
    return np.array([float(c @ (A[:, j] * X)) for j in (1, 2, 3)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP = ("import numpy as np\n"
              "_s = _oracle_plane_wave_shell_at_X(5, 4)\n"
              "_g = _oracle_wave_vector_group_at_X(2)\n"
              "_b = _oracle_valleyor_basis(_s, _g)\n"
              "_r = _oracle_valley_representation(_b, _s, _g)\n"
              "_t = _oracle_tau_symmetry_data(_r, _b, _s)\n"
              "_A = _oracle_allowed_valley_channels(_g, _t)\n"
              "_geom = _oracle_dot_geometry(1.10, 1.45, 30.0, 0.19, 5.431, 0.20)\n"
              "_c = np.array([4.20e4, 3.10e4, 2.50e3, 1.90e3, 1.80e5, 5.60e1, 4.40e1, 3.90e1, 2.80e1, 7.30e3,\n"
              "               6.10e3, 8.80e3, 9.40e3, 2.15e1, 1.75e1, 6.00e4, 1.40e3, 1.10e3, 9.00e2, 1.30e2,\n"
              "               4.70e1, 3.50e1, 3.00e1, 4.00e1, 1.20e1, 1.20e4, 1.50e5, 1.20e5, 6.00e1])\n"
              "_tb = np.deg2rad(75.0)\n_tp = np.deg2rad(45.0)\n"
              "_B = np.array([1.50*np.cos(_tb), 1.50*np.sin(_tb), 0.20])\n"
              "_P = np.array([1.00*np.cos(_tp), 1.00*np.sin(_tp), 0.0])\n"
              "_E = np.array([[3.0e-3, 6.0e-4, 2.0e-4], [6.0e-4, 1.8e-3, 1.0e-4], [2.0e-4, 1.0e-4, -1.2e-3]])")
    return [
        {
            "setup": _SETUP,
            "call": 'valley_magnetic_components(_c, _A, _P, _B, 6.0, _E, _geom)',
            "gold_call": '_oracle_valley_magnetic_components(_c, _A, _P, _B, 6.0, _E, _geom)',
        },  # normal, the device of the task at its built-in polar vector
        {
            "setup": _SETUP + "\n_B0 = np.array([0.0, 0.0, 0.0])",
            "call": 'valley_magnetic_components(_c, _A, _P, _B0, 0.0, _E, _geom)',
            "gold_call": '_oracle_valley_magnetic_components(_c, _A, _P, _B0, 0.0, _E, _geom)',
        },  # edge, no magnetic or vertical electric field, so only strain, geometry and polar-vector channels survive
        {
            "setup": _SETUP + "\n_Bf = np.array([1.50*np.cos(_tb), 1.50*np.sin(_tb), 0.0])",
            "call": 'valley_magnetic_components(_c, _A, _P, _Bf, 6.0, _E, _geom)',
            "gold_call": '_oracle_valley_magnetic_components(_c, _A, _P, _Bf, 6.0, _E, _geom)',
        },  # boundary, a strictly in-plane field switches off the tilt-activated channel
        {
            "setup": _SETUP + "\n_P2 = np.array([0.35, -1.20, 0.0])",
            "call": 'valley_magnetic_components(_c, _A, _P2, _B, -3.0, _E, _geom)',
            "gold_call": '_oracle_valley_magnetic_components(_c, _A, _P2, _B, -3.0, _E, _geom)',
        },  # normal, a dot whose polar vector has wandered from the built-in one, with the gate field reversed
        {
            "setup": _SETUP + "\n_Ex = np.zeros((3, 3))\n_geom0 = _oracle_dot_geometry(1.10, 1.45, 0.0, 0.19, 5.431, 0.20)",
            "call": 'valley_magnetic_components(_c, _A, _P, _B, 6.0, _Ex, _geom0)',
            "gold_call": '_oracle_valley_magnetic_components(_c, _A, _P, _B, 6.0, _Ex, _geom0)',
        },  # boundary, unstrained and crystal-aligned, leaving only field and polar-vector channels
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: valley_magnetic_components(_c[:22], _A, _P, _B, 6.0, _E, _geom))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_valley_magnetic_components(_c[:22], _A, _P, _B, 6.0, _E, _geom))])',
        },  # contract, a wrong number of coupling constants must raise ValueError
    ]
