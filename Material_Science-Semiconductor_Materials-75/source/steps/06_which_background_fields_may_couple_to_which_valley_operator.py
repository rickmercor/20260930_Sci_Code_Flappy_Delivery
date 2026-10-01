"""
Decide, for each of twenty-nine candidate field combinations, which valley operators it may couple to under both the crystal symmetry and time reversal.

A perturbation can split the two valleys only if it enters the effective Hamiltonian multiplied
by a valley Pauli matrix, and it can do so only if the product is invariant under every operation
of the group of the wave vector and under time reversal. Step 05 supplied the transformation law
of the valley operators; what remains is to apply the same test to candidate combinations of the
background fields available in a real device and see which pairings survive.

Five background quantities are considered. A polar vector P describes a built-in preferred
in-plane direction, such as the asymmetry of a gate-defined dot; it transforms with the point
matrix and is even under time reversal. The vertical electric field Fz set by the gates is the z
component of another polar vector and behaves the same way. An axial vector B is the magnetic
field; it transforms with the point matrix times its determinant and is odd under time reversal.
A symmetric second-rank tensor eps is the strain; it transforms by conjugation with the point
matrix and is even under time reversal. A traceless symmetric in-plane tensor Q encodes the
ellipticity of the dot and transforms and behaves under time reversal exactly as strain does. In
addition the in-plane momentum enters through the single product kx ky; each momentum component
transforms with the point matrix and is odd under time reversal. Because these are macroscopic
fields and momenta, the fractional translations of the non-symmorphic operations play no role in
how they transform, so only the point part of each operation acts on them. The valley operators,
by contrast, carry the signs of step 05, which already include the translations.

The crystal test compares a candidate before and after an operation: the transformed value must
equal the original times the sign that the intended valley operator carries under that operation,
for every operation and for field configurations general enough that no accidental coincidence
survives. The time-reversal test is separate and equally binding: the parity of the field
combination times the parity of the valley operator must be even. A term that fails either test
is forbidden.

The candidate list, indexed 0 to 28 in this exact order, is

   0  eps_xy                              15  Bz (eps_xx - eps_yy)
   1  eps_xx - eps_yy                     16  Bx eps_xz + By eps_yz
   2  eps_zz                              17  Bx eps_yz + By eps_xz
   3  eps_xx + eps_yy                     18  Bz eps_xy
   4  kx ky                               19  Q_xy
   5  Px By - Py Bx                       20  (Px Bx - Py By) Q_xy
   6  Px Bx - Py By                       21  (Px By + Py Bx)(Q_xx - Q_yy)
   7  Px By + Py Bx                       22  Px Py
   8  Px Bx + Py By                       23  Bx By
   9  (Px By - Py Bx) eps_zz              24  Fz
  10  (Px By - Py Bx)(eps_xx + eps_yy)    25  Fz eps_xy
  11  (Px Bx - Py By) eps_xy              26  Px eps_yz + Py eps_xz
  12  (Px By + Py Bx)(eps_xx - eps_yy)    27  Px eps_xz + Py eps_yz
  13  Bz (Px By + Py Bx)                  28  Bz (Bx^2 - By^2)
  14  Bz (Px Bx + Py By)

and the valley operators are indexed 0 to 3, with 0 the identity, which carries sign +1 under
every operation and is even under time reversal. A term paired with the identity shifts both
valleys together and does not split them, but reporting it keeps the classification complete.

Returns
-------
np.ndarray of length 116: the 29x4 allowed-channel table of ones and zeros, flattened row-major
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def allowed_valley_channels(group: np.ndarray, tau_data: np.ndarray) -> np.ndarray:
    '''Table of which candidate field combinations may couple to which valley operator.

    Parameters
    ----------
    group : np.ndarray
        Output of step 02: length-384 real array holding the thirty-two operations.
    tau_data : np.ndarray
        Output of step 05: length-99 real array of crystal signs and time-reversal parities.

    Returns
    -------
    result : np.ndarray
        Real array of length 116 holding a twenty-nine by four table of ones and zeros, flattened
        row-major. Entry (k, j) is 1 when candidate term k may couple to valley operator j under
        both the crystal symmetry and time reversal, and 0 otherwise.

    Raises
    ------
    ValueError
        If the group array or the tau data does not have the expected length.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _group_from_flat(v):
    v = np.asarray(v, dtype=float).ravel()
    return np.rint(v[:288]).astype(int).reshape(32, 3, 3), v[288:384].reshape(32, 3) * np.pi


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


def _oracle_allowed_valley_channels(group: np.ndarray, tau_data: np.ndarray) -> np.ndarray:
    import numpy as np
    # time-reversal parity of each candidate combination, in index order
    _TERM_TR = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                         1, 1, -1, -1, -1, -1, 1, -1, -1,
                         1, 1, 1, 1, 1, 1, -1])
    if np.asarray(group, dtype=float).ravel().size != 384:
        raise ValueError("the group array must carry thirty-two operations")
    if np.asarray(tau_data, dtype=float).ravel().size != 99:
        raise ValueError("the tau data must carry 96 crystal signs and 3 parities")
    rot, _ = _group_from_flat(group)
    td = np.asarray(tau_data, dtype=float).ravel()
    signs = np.rint(td[:96]).astype(int).reshape(3, 32)
    trp = np.rint(td[96:99]).astype(int)
    sgn = np.vstack([np.ones(32, dtype=int), signs])
    trv = np.concatenate([[1], trp])
    nt = _TERM_TR.size
    ok = np.ones((nt, 4), dtype=int)
    rng = np.random.default_rng(20260831)
    for _ in range(8):
        Pv = rng.normal(size=3)
        Bv = rng.normal(size=3)
        kvec = rng.normal(size=3)
        Fv = np.array([0.0, 0.0, rng.normal()])
        E = rng.normal(size=(3, 3))
        E = E + E.T
        q = rng.normal(size=(2, 2))
        q = q + q.T
        q = q - np.trace(q) / 2.0 * np.eye(2)
        Q = np.zeros((3, 3))
        Q[:2, :2] = q
        base = _term_vector(Pv, Bv, E, Q, kvec[0] * kvec[1], Fv[2])
        for g in range(32):
            R = rot[g]
            d = int(round(float(np.linalg.det(R))))
            kg = R @ kvec
            Fg = R @ Fv
            got = _term_vector(R @ Pv, d * (R @ Bv), R @ E @ R.T, R @ Q @ R.T, kg[0] * kg[1], Fg[2])
            for j in range(4):
                ok[np.abs(got - sgn[j, g] * base) > 1e-9, j] = 0
    for j in range(4):
        ok[_TERM_TR * trv[j] != 1, j] = 0
    return ok.astype(float).ravel()

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
              "_t = _oracle_tau_symmetry_data(_r, _b, _s)")
    return [
        {
            "setup": _SETUP,
            "call": 'allowed_valley_channels(_g, _t)',
            "gold_call": '_oracle_allowed_valley_channels(_g, _t)',
        },  # normal, the full classification table
        {
            "setup": _SETUP,
            "call": 'allowed_valley_channels(_g, _t).reshape(29, 4).sum(axis=0)',
            "gold_call": '_oracle_allowed_valley_channels(_g, _t).reshape(29, 4).sum(axis=0)',
        },  # boundary, how many candidates reach each valley operator
        {
            "setup": _SETUP + "\n_flip = _t.copy()\n_flip[96:] = 1.0",
            "call": 'allowed_valley_channels(_g, _flip)',
            "gold_call": '_oracle_allowed_valley_channels(_g, _flip)',
        },  # edge, with every valley operator even the time-reversal-odd channels must disappear
        {
            "setup": _SETUP,
            "call": 'np.array([float(allowed_valley_channels(_g, _t).reshape(29, 4)[k, j]) for k in (16, 17, 22, 23, 24, 25, 26, 27, 28) for j in range(4)])',
            "gold_call": 'np.array([float(_oracle_allowed_valley_channels(_g, _t).reshape(29, 4)[k, j]) for k in (16, 17, 22, 23, 24, 25, 26, 27, 28) for j in range(4)])',
        },  # boundary, the rows whose fate is not decided by resemblance to a spin coupling
        {
            "setup": _SETUP + '\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\n',
            "call": 'np.array([_raises(lambda: allowed_valley_channels(_g[:200], _t)), _raises(lambda: allowed_valley_channels(_g, _t[:96]))])',
            "gold_call": 'np.array([_raises(lambda: _oracle_allowed_valley_channels(_g[:200], _t)), _raises(lambda: _oracle_allowed_valley_channels(_g, _t[:96]))])',
        },  # contract, a truncated group or tau array must raise ValueError
    ]
