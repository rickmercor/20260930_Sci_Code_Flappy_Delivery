"""
Runs the complete chain and assembles the confinement audit. Rows 1..len(radii)+1 correspond to the radii in the given order followed by A_star, with the 19 columns [A, E_0 (exact ground state), E_e - E_0, |d_e|^2, E_h - E_0, |d_h|^2, w_h/w_e, E_e^bare, E_h^bare, |d_bare|^2, E_e^min - E_0^min, E_h^min - E_0^min, |d_e^min|^2, |d_h^min|^2, E_(2p)w - E_(1s)w, |d_w|^2, f_e, f_h, (E_e - E_0) - E_e^bare], where the exact quantities use the basis (N_n, L_max) and the grid N_r, the minimal model is the source's 1s + 1p model, the weak-confinement model uses N_g, and the electron-like resonance is the largest-weight L = 1 transition with the hole-like one the largest weight below it. The head row (row 0) holds [E_e - E_0 at A_star, its Coulomb shift (E_e - E_0) - E_e^bare at A_star, w_h/w_e at A_star, f_e at A_star, the independence radius A_10 on [A_lo, A_hi] at the given level, c_EC at A_star, the exciton Bohr radius a_X = 4 pi eps eps_0 hbar^2/(mu e^2), the source's estimate 3 pi hbar^2 eps eps_0 (1/m_e - 1/m_h)/e^2 of the independence radius, A_star, N_n, L_max, N_r, N_g, len(radii), the kinetic splitting z_11^2 - z_10^2 (the coefficient of 1/A^2), the 1s-1s direct Coulomb integral in units of e^2/(4 pi eps eps_0 A), the strongly confined limit of the ratio of the weak-confinement transition energy to the non-interacting electron transition (the weak-confinement model with the Coulomb term omitted, at A_star), the ratio of the weak-confinement transition energy to the exact electron-like resonance at A_star] followed by a zero.

The head scalar is the exact electron-like THz resonance of the confined pair at the crossover radius, where the source's two approximate descriptions meet and neither is exact; the table records how the exact energies, dipole weights and renormalisation factors compare with the non-interacting pair, with the source's minimal two-state model and with its shrunken-exciton model as the crystal grows from the strongly confined limit to the exciton Bohr radius.

Returns
-------
A float64 array of shape (len(radii) + 2, 19): the head row followed by one row per radius and the A_star row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def confinement_audit(me: float, mh: float, eps: float, radii: list, N_n: int, L_max: int, N_r: int, N_g: int, A_star: float, A_lo: float, A_hi: float, level: float) -> "np.ndarray":
    """Runs the complete chain and assembles the confinement audit. Rows 1..len(radii)+1 correspond to the radii in the
    given order followed by A_star, with the 19 columns [A, E_0 (exact ground state), E_e - E_0, |d_e|^2, E_h - E_0,
    |d_h|^2, w_h/w_e, E_e^bare, E_h^bare, |d_bare|^2, E_e^min - E_0^min, E_h^min - E_0^min, |d_e^min|^2, |d_h^min|^2,
    E_(2p)w - E_(1s)w, |d_w|^2, f_e, f_h, (E_e - E_0) - E_e^bare], where the exact quantities use the basis (N_n,
    L_max) and the grid N_r, the minimal model is the source's 1s + 1p model, the weak-confinement model uses N_g, and
    the electron-like resonance is the largest-weight L = 1 transition with the hole-like one the largest weight below
    it. The head row (row 0) holds [E_e - E_0 at A_star, its Coulomb shift (E_e - E_0) - E_e^bare at A_star, w_h/w_e
    at A_star, f_e at A_star, the independence radius A_10 on [A_lo, A_hi] at the given level, c_EC at A_star, the
    exciton Bohr radius a_X = 4 pi eps eps_0 hbar^2/(mu e^2), the source's estimate 3 pi hbar^2 eps eps_0 (1/m_e -
    1/m_h)/e^2 of the independence radius, A_star, N_n, L_max, N_r, N_g, len(radii), the kinetic splitting z_11^2 -
    z_10^2 (the coefficient of 1/A^2), the 1s-1s direct Coulomb integral in units of e^2/(4 pi eps eps_0 A), the
    strongly confined limit of the ratio of the weak-confinement transition energy to the non-interacting electron
    transition (the weak-confinement model with the Coulomb term omitted, at A_star), the ratio of the
    weak-confinement transition energy to the exact electron-like resonance at A_star] followed by a zero.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Derived quantities: the renormalisation factors f_e = |d_e|^2/|d_bare|^2 and f_h = |d_h|^2/|d_bare|^2 of the
    electron-like and hole-like resonances relative to the non-interacting squared dipole, the weight ratio w_h/w_e
    (hole-like over electron-like), the Coulomb shift of the electron-like resonance relative to the non-interacting
    electron transition, the source's coupling constant c_EC = |E_C| 4 pi eps eps_0 A/e^2 with E_C the off-diagonal
    element between the two states of the minimal L = 1 block, and the independence radius A_10, the radius at which
    f_e reaches the level 1 + 0.1 (root-found on a stated bracket).

    Args:
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        radii: list of floats, the audit radii in nm (at least one, all positive), in the order of the table rows.
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        N_g: int, the number of uniform intervals of the finite-difference grid on [0, r_max] at the coarsest
            Richardson level; at least 500.
        A_star: float, the reference radius in nm (positive) of the head row and of the last table row.
        A_lo: float, the lower end of the bracket in nm on which the root is sought (0 < A_lo < A_hi).
        A_hi: float, the upper end of the bracket in nm.
        level: float, the target renormalisation factor f_e (above 1; 1.1 for a ten percent enhancement).

    Returns:
        A float64 array of shape (len(radii) + 2, 19): the head row followed by one row per radius and the A_star row.

    Raises:
        ValueError: for an empty or non-positive radii list, a non-positive A_star, or invalid arguments of the
        underlying steps.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def _oracle_confinement_audit(me: float, mh: float, eps: float, radii: list, N_n: int, L_max: int, N_r: int, N_g: int, A_star: float, A_lo: float, A_hi: float, level: float) -> "np.ndarray":
    radii = [float(a) for a in radii]
    if len(radii) < 1 or min(radii) <= 0: raise ValueError("radii must be positive")
    if not (A_star > 0): raise ValueError("A_star must be positive")
    rows = []
    n1 = len(_pair_basis(N_n, L_max, 1))
    for A in list(radii) + [A_star]:
        tr = _oracle_thz_transitions(A, me, mh, eps, N_n, L_max, N_r, n1)
        ie = int(np.argmax(tr[:, 2]))
        below = [k for k in range(ie) if tr[k, 0] < tr[ie, 0]]
        if not below: raise ValueError("no resonance below the dominant one")
        ih = int(below[int(np.argmax(tr[below, 2]))])
        E0 = float(_oracle_pair_levels(A, me, mh, eps, N_n, L_max, 0, N_r, 1)[0])
        if abs(E0 - tr[0, 3]) > 1e-12: raise ValueError("ground-state energies of the two block solvers disagree")
        bare = _oracle_bare_response(A, me, mh, N_r)
        mn = _oracle_minimal_scr(A, me, mh, eps, N_r)
        wc = _oracle_wcr_model(A, me, mh, eps, N_g)
        rows.append([A, E0, tr[ie, 0], tr[ie, 1], tr[ih, 0], tr[ih, 1], tr[ih, 2] / tr[ie, 2], bare[0], bare[1], bare[2], mn[2], mn[1], mn[4], mn[3], wc[2], wc[3],
                     tr[ie, 1] / bare[2], tr[ih, 1] / bare[3], tr[ie, 0] - bare[0]])
    A10 = _oracle_independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level)
    mu = me * mh / (me + mh)
    aX = 2.0 * _HB() / mu * eps / _E2()
    A_est = 1.5 * _HB() * eps / _E2() * (1.0 / me - 1.0 / mh)   # source's Eq. (26): 3 pi hbar^2 eps/(e^2) (1/me - 1/mh)
    Z = _oracle_box_spectrum(A_star, max(N_n, 1), max(L_max, 1))
    split = float(Z[1, 0] ** 2 - Z[0, 0] ** 2)                              # k_1p^2 - k_1s^2 in units of 1/A^2
    direct = float(_oracle_coulomb_integral(A_star, 0, 1, 0, 1, 0, 1, 0, 1, 0, N_r)) * A_star   # 1s-1s direct integral in e^2/(4 pi eps eps_0 A)
    names = ["A", "E0", "dE_e", "d2_e", "dE_h", "d2_h", "ratio", "dE_e_bare", "dE_h_bare", "d2_bare", "dE_e_min", "dE_h_min", "d2_e_min", "d2_h_min", "dE_w", "d2_w", "renorm_e", "renorm_h", "shift_e"]
    table = np.zeros((1 + len(rows), len(names)))
    table[1:] = np.array(rows)
    star = rows[-1]
    wc0 = _oracle_wcr_model(A_star, me, mh, np.inf, N_g)                     # Coulomb term omitted: the strongly confined limit of the model
    limit = float(wc0[2] / _oracle_bare_response(A_star, me, mh, N_r)[0])
    table[0, :18] = [star[2], star[18], star[6], star[16], A10, _oracle_minimal_scr(A_star, me, mh, eps, N_r)[6], aX, A_est, A_star, N_n, L_max, N_r, N_g, len(radii), split, direct,
                     limit, star[14] / star[2]]
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nme = 0.07\nmh = 0.13\neps = 12.85\nradii = [5.0, 10.0]\nN_n = 6\nL_max = 5\nN_r = 2001\nN_g = 1000\nA_star = 15.0\nA_lo = 3.0\nA_hi = 9.0\nlevel = 1.1\n', 'call': 'confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level)', 'gold_call': '_oracle_confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level)', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# boundary: a single audit radius and near-equal masses\nme = 0.09\nmh = 0.11\neps = 12.85\nradii = [4.0]\nN_n = 4\nL_max = 3\nN_r = 1001\nN_g = 800\nA_star = 8.0\nA_lo = 0.5\nA_hi = 2.5\nlevel = 1.1\n', 'call': 'confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level)', 'gold_call': '_oracle_confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level)', 'tol': 1e-07},
        {'setup': 'import numpy as np\nme = 0.07\nmh = 0.13\neps = 12.85\nradii = [10.0]\nN_n = 6\nL_max = 5\nN_r = 2001\nN_g = 1000\nA_star = 15.0\nA_lo = 4.0\nA_hi = 8.0\nlevel = 1.1\n', 'call': 'float(np.asarray(confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level))[0, 0])', 'gold_call': 'float(np.asarray(_oracle_confinement_audit(me, mh, eps, radii, N_n, L_max, N_r, N_g, A_star, A_lo, A_hi, level))[0, 0])', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: an empty list of audit radii leaves no rows to fill and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: confinement_audit(0.07, 0.13, 12.85, [], 2, 1, 401, 500, 15.0, 3.0, 9.0, 1.1))', 'gold_call': '_catches_value_error(lambda: _oracle_confinement_audit(0.07, 0.13, 12.85, [], 2, 1, 401, 500, 15.0, 3.0, 9.0, 1.1))'},
    ]
