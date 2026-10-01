"""
Average the alloy Rice law over the dot-to-dot spread of the polar vector and return the spread of the time-reversal-odd component together with the percentage of the wafer below a splitting threshold, without and with that spread.

The valley-magnetic field of step 08 is fixed once the fields of a dot are fixed. Across a wafer of
nominally identical dots two things are not fixed, and both come from where the germanium atoms
happen to sit.

The first is the alloy matrix element between the two valley states. It is a sum over many randomly
placed atoms and hence a complex Gaussian of zero mean with no preferred phase, whose two real
components are independent with standard deviation sigma_ge. The alloy potential is a real scalar,
even under time reversal, so it can only be carried by the time-reversal-even valley operators. It
adds to the valley-magnetic components on those two operators and leaves the component on the odd
operator untouched.

The second is the in-plane asymmetry of each dot. The local electrostatics that set the polar vector
are disturbed by the same random alloy, so the polar vector of a dot is the built-in vector plus a
random in-plane vector whose x and y components are independent Gaussians of zero mean and standard
deviation sigma_P, independent of the alloy matrix element. Every valley-magnetic component that
depends on the polar vector moves with it, on all three operators at once, so the three components
are correlated from dot to dot. Every channel that can reach the time-reversal-odd operator is
linear in the polar vector, so that component is an affine function of the random vector; if the
supplied couplings and table make it anything else, the calculation is not defined and must be
refused.

The splitting of a dot is twice the length of its three-component vector. For a given polar vector
the dot can fall below a threshold only if the odd component alone stays below half the threshold,
and then the squared length in the plane of the two even operators, in units of sigma_ge squared,
follows a non-central chi-square law with two degrees of freedom whose non-centrality is set by the
deterministic even components of that dot. The wafer fraction is that conditional probability
averaged over the Gaussian polar vector. The average has to be computed deterministically to reach
the stated precision, which a sampling estimate cannot, and because the integrand switches off
where the odd component reaches half the threshold, a quadrature that ignores where that happens
converges poorly.

Return three numbers: the standard deviation across the wafer of the odd component, the
percentage below threshold if every dot had exactly the built-in polar vector, and the percentage
below threshold on the wafer, each percentage converged to a relative precision of 1e-9.

Returns
-------
np.ndarray of length 3: the standard deviation of the protected component in microelectron volts, then the percentage below threshold with the polar vector fixed, then the percentage with it random
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wafer_valley_statistics(coeffs: np.ndarray, allowed: np.ndarray, tau_parities: np.ndarray,
                            P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float,
                            strain: np.ndarray, dot_geom: np.ndarray, sigma_ge_ueV: float,
                            sigma_P: float, e_threshold_ueV: float) -> np.ndarray:
    '''Spread of the protected component and the low tail of the wafer with a random polar vector.

    Parameters
    ----------
    coeffs : np.ndarray
        Length-29 coupling constants, as in step 08.
    allowed : np.ndarray
        Output of step 06: length-116 allowed-channel table.
    tau_parities : np.ndarray
        Shape (3,), the time-reversal parity (+1 or -1) of valley operators 1, 2 and 3.
    P_vec : np.ndarray
        Length-3 built-in polar vector; only its in-plane entries are used and randomised.
    B_vec : np.ndarray
        Length-3 magnetic field in tesla.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three strain tensor.
    dot_geom : np.ndarray
        Output of step 07: length-4 real array.
    sigma_ge_ueV : float
        Standard deviation of each real component of the germanium matrix element, in
        microelectron volts; positive.
    sigma_P : float
        Standard deviation of each in-plane component of the random polar vector; not negative.
    e_threshold_ueV : float
        Splitting threshold in microelectron volts; positive.

    Returns
    -------
    result : np.ndarray
        Array of length 3: the standard deviation across the wafer of the valley-magnetic
        component on the time-reversal-odd operator, in microelectron volts; the percentage of
        dots below the threshold with the polar vector fixed at P_vec; and the percentage below
        the threshold with the random polar vector. Percentages are converged to a relative
        precision of 1e-9.

    Raises
    ------
    ValueError
        If sigma_ge_ueV is not positive, if sigma_P is negative, if e_threshold_ueV is not
        positive, if the parities do not single out exactly one time-reversal-odd operator, or if
        the component on that operator is not an affine function of the in-plane polar vector.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ncx2_cdf_two_dof(x, lam):
    import numpy as np
    if x <= 0.0:
        return 0.0
    half_x, half_lam = 0.5 * float(x), 0.5 * float(lam)
    total, poisson, inner, term = 0.0, np.exp(-half_lam), 0.0, 1.0
    for j in range(4000):
        inner += term
        total += poisson * (1.0 - np.exp(-half_x) * inner)
        if j > 8 and poisson < 1e-18 and j > 4.0 * half_lam:
            break
        term *= half_x / (j + 1.0)
        poisson *= half_lam / (j + 1.0)
    return float(total)


def _oracle_wafer_valley_statistics(coeffs: np.ndarray, allowed: np.ndarray, tau_parities: np.ndarray, P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray, dot_geom: np.ndarray, sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> np.ndarray:
    import numpy as np
    from scipy import integrate
    sigma = float(sigma_ge_ueV)
    sP = float(sigma_P)
    e_th = float(e_threshold_ueV)
    if sigma <= 0.0:
        raise ValueError("the germanium standard deviation must be positive")
    if sP < 0.0:
        raise ValueError("the polar-vector standard deviation cannot be negative")
    if e_th <= 0.0:
        raise ValueError("the splitting threshold must be positive")
    parity = np.rint(np.asarray(tau_parities, dtype=float).ravel()).astype(int)
    odd = [j for j in range(3) if parity[j] == -1]
    if len(odd) != 1:
        raise ValueError("expected exactly one time-reversal-odd valley operator")
    k = odd[0]
    pl = [j for j in range(3) if j != k]
    P0 = np.asarray(P_vec, dtype=float).ravel().copy()

    def _comps(dx, dy):
        P = P0.copy()
        P[0] += dx
        P[1] += dy
        return _oracle_valley_magnetic_components(coeffs, allowed, P, B_vec, Fz_MV_per_m,
                                                  strain, dot_geom)

    v0 = _comps(0.0, 0.0)
    vxp, vxm = _comps(1.0, 0.0), _comps(-1.0, 0.0)
    vyp, vym = _comps(0.0, 1.0), _comps(0.0, -1.0)
    vpp, vpm, vmp, vmm = _comps(1.0, 1.0), _comps(1.0, -1.0), _comps(-1.0, 1.0), _comps(-1.0, -1.0)
    Jx, Jy = 0.5 * (vxp - vxm), 0.5 * (vyp - vym)
    Hxx, Hyy = vxp + vxm - 2.0 * v0, vyp + vym - 2.0 * v0
    Hxy = 0.25 * (vpp - vpm - vmp + vmm)
    scale = max(1.0, float(np.max(np.abs(np.concatenate([v0, Jx, Jy])))))
    if max(abs(Hxx[k]), abs(Hyy[k]), abs(Hxy[k])) > 1e-9 * scale:
        raise ValueError("the protected component must be affine in the polar vector")
    var_k = sP ** 2 * (Jx[k] ** 2 + Jy[k] ** 2)
    std_k = float(np.sqrt(var_k))

    R2 = 0.25 * e_th ** 2

    def _cond(v):
        r2 = R2 - v[k] ** 2
        if r2 <= 0.0:
            return 0.0
        return _ncx2_cdf_two_dof(r2 / sigma ** 2, (v[pl[0]] ** 2 + v[pl[1]] ** 2) / sigma ** 2)

    fixed = _cond(v0)
    if sP == 0.0:
        return np.array([std_k, 100.0 * fixed, 100.0 * fixed], dtype=float)

    gn = float(np.hypot(Jx[k], Jy[k]))
    L = 12.0
    if gn > 0.0:
        ex = np.array([Jx[k], Jy[k]]) / gn
    else:
        ex = np.array([1.0, 0.0])
    ey = np.array([-ex[1], ex[0]])
    if gn > 0.0:
        u_lo = max(-L, (-np.sqrt(R2) - v0[k]) / (gn * sP))
        u_hi = min(L, (np.sqrt(R2) - v0[k]) / (gn * sP))
    else:
        u_lo, u_hi = (-L, L) if v0[k] ** 2 < R2 else (0.0, 0.0)
    if u_hi <= u_lo:
        return np.array([std_k, 100.0 * fixed, 0.0], dtype=float)
    norm = 1.0 / (2.0 * np.pi)

    def _inner(w, u):
        d = sP * (u * ex + w * ey)
        return norm * np.exp(-0.5 * (u * u + w * w)) * _cond(_comps(d[0], d[1]))

    val, _ = integrate.dblquad(_inner, u_lo, u_hi, -L, L, epsabs=1e-13, epsrel=1e-11)
    return np.array([std_k, 100.0 * fixed, 100.0 * val], dtype=float)

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
              "_E = np.array([[3.0e-3, 6.0e-4, 2.0e-4], [6.0e-4, 1.8e-3, 1.0e-4], [2.0e-4, 1.0e-4, -1.2e-3]])\n"
              "_par = np.array([1.0, 1.0, -1.0])")
    return [
        {
            "setup": _SETUP,
            "call": 'wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)',
            "tol": 1e-08,
        },  # normal, the wafer of the task
        {
            "setup": _SETUP,
            "call": 'wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.0, 240.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.0, 240.0)',
            "tol": 1e-08,
        },  # edge, no spread of the polar vector, so both percentages coincide and the odd component is a constant
        {
            "setup": _SETUP,
            "call": 'wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 160.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 160.0)',
            "tol": 1e-08,
        },  # boundary, half the threshold only just above the built-in odd component, so a narrow band of polar vectors contributes
        {
            "setup": _SETUP + '\n_c3 = _c.copy()\n_c3[[5, 9, 10, 11, 12, 20, 21]] = 0.0',
            "call": 'wafer_valley_statistics(_c3, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c3, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)',
            "tol": 1e-08,
        },  # edge, every odd channel switched off, so nothing lifts the tail while the even components still fluctuate
        {
            "setup": _SETUP,
            "call": 'wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 60.0, 0.80, 300.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 60.0, 0.80, 300.0)',
            "tol": 1e-08,
        },  # normal, a cleaner alloy with a much wider spread of dot asymmetry
        {
            "setup": _SETUP,
            "call": 'wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.01, 60.0)',
            "gold_call": '_oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.01, 60.0)',
            "tol": 1e-8,
        },  # edge, half the threshold far below the odd component for every polar vector that matters, so no dot falls below it
        {
            "setup": _SETUP + '\n_Abad = _A.reshape(29, 4).copy()\n_Abad[22, 3] = 1.0\n_Abad = _Abad.ravel()\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 0.0, 0.30, 240.0)), _raises(lambda: wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, -0.10, 240.0)), _raises(lambda: wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 0.0)), _raises(lambda: wafer_valley_statistics(_c, _A, np.array([1.0, 1.0, 1.0]), _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)), _raises(lambda: wafer_valley_statistics(_c, _Abad, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 0.0, 0.30, 240.0)), _raises(lambda: _oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, -0.10, 240.0)), _raises(lambda: _oracle_wafer_valley_statistics(_c, _A, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 0.0)), _raises(lambda: _oracle_wafer_valley_statistics(_c, _A, np.array([1.0, 1.0, 1.0]), _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0)), _raises(lambda: _oracle_wafer_valley_statistics(_c, _Abad, _par, _P, _B, 6.0, _E, _geom, 90.0, 0.30, 240.0))])',
        },  # contract, a non-positive alloy spread, a negative polar spread, a non-positive threshold, no odd operator, and a quadratic term placed on the odd operator must all raise ValueError
    ]
