"""
Evaluates the source's minimal strong-confinement model, the exact model restricted to the single-particle states 1s and 1p (N_n = 1, L_max = 1), and returns [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC], where the L = 0 block spans |1s,1s> and |(1p 1p) L = 0>, the L = 1 block spans |(1s 1p)> and |(1p 1s)>, E_h and E_e are its lower and higher eigenvalues with their squared transition dipoles from the ground state, E_C (eV, negative) is the off-diagonal Coulomb element between the two L = 1 basis states and c_EC = |E_C| 4 pi eps eps_0 A/e^2 its dimensionless magnitude.

The minimal two-state description of the source captures the mechanism: the two bare p-like pair states of equal parity are coupled by the Coulomb interaction, their symmetric combination loses dipole strength and their antisymmetric one gains it, so the lower resonance fades and the upper one is amplified as the coupling grows relative to the kinetic splitting.

Returns
-------
A float64 array of shape (7,), [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimal_scr(A: float, me: float, mh: float, eps: float, N_r: int) -> "np.ndarray":
    """Evaluates the source's minimal strong-confinement model, the exact model restricted to the single-particle states
    1s and 1p (N_n = 1, L_max = 1), and returns [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC], where the L
    = 0 block spans |1s,1s> and |(1p 1p) L = 0>, the L = 1 block spans |(1s 1p)> and |(1p 1s)>, E_h and E_e are its
    lower and higher eigenvalues with their squared transition dipoles from the ground state, E_C (eV, negative) is
    the off-diagonal Coulomb element between the two L = 1 basis states and c_EC = |E_C| 4 pi eps eps_0 A/e^2 its
    dimensionless magnitude.

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

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float64 array of shape (7,), [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC].

    Raises:
        ValueError: for a non-positive A, mass or eps, or for an invalid N_r.
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

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def _cg(j1, m1, j2, m2, J, M, _memo={}):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (Racah formula, integer arguments)"""
    if m1 + m2 != M or J < abs(j1 - j2) or J > j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J: return 0.0
    key = (j1, m1, j2, m2, J, M)
    if key in _memo: return _memo[key]
    pre = (2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2) * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1)
    pre *= factorial(J + M) * factorial(J - M) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)
    s = 0.0
    for k in range(0, j1 + j2 + J + 2):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0: continue
        s += (-1) ** k / np.prod([float(factorial(x)) for x in d])
    _memo[key] = sqrt(pre) * s
    return _memo[key]

def _ylm_element(l2, m2, k, q, l1, m1):
    """<l2 m2| Y_kq |l1 m1> = sqrt((2 l1 + 1)(2 k + 1)/(4 pi (2 l2 + 1))) <l1 0 k 0|l2 0> <l1 m1 k q|l2 m2>"""
    if m2 != m1 + q: return 0.0
    return sqrt((2 * l1 + 1) * (2 * k + 1) / (4 * pi * (2 * l2 + 1))) * _cg(l1, 0, k, 0, l2, 0) * _cg(l1, m1, k, q, l2, m2)

def _angular_factor(le2, lh2, le1, lh1, k, L):
    """angular part of <(le2 lh2) L 0| P_k(cos theta_eh) |(le1 lh1) L 0> times (4 pi/(2k+1)) sum_q Y_kq(e) Y_kq^*(h)"""
    val = 0.0
    for me1 in range(-le1, le1 + 1):
        mh1 = -me1
        if abs(mh1) > lh1: continue
        c1 = _cg(le1, me1, lh1, mh1, L, 0)
        if c1 == 0.0: continue
        for me2 in range(-le2, le2 + 1):
            mh2 = -me2
            if abs(mh2) > lh2: continue
            c2 = _cg(le2, me2, lh2, mh2, L, 0)
            if c2 == 0.0: continue
            q = me2 - me1
            ge = _ylm_element(le2, me2, k, q, le1, me1)
            gh = _ylm_element(lh2, mh2, k, -q, lh1, mh1)
            val += c1 * c2 * ge * gh * (-1) ** q
    return 4.0 * pi / (2 * k + 1) * val

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

def _poisson_potentials(F, r, w, k):
    """Phi_k(r1) = int F(r2) r_<^k / r_>^(k+1) dr2 for every row of F (rows already carry the r2^2 weight)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = cumulative_simpson(F * r ** k, x=r, initial=0)
        rinv = np.where(r > 0, r ** (-(k + 1.0)), 0.0)
        outer_tot = cumulative_simpson(F * rinv, x=r, initial=0)
        outer = outer_tot[:, -1:] - outer_tot
        Phi = rinv * inner + r ** k * outer
    Phi[:, 0] = outer_tot[:, -1] if k == 0 else 0.0
    return Phi

def _coulomb_matrix(A, eps, N_n, L_max, L, N_r, states, r, w, R):
    """matrix of -e^2/(4 pi eps eps_0 |r_e - r_h|) in the coupled basis (eV)"""
    cc = _E2() / eps
    N = len(states)
    idx = {s: i for i, s in enumerate(states)}
    lpairs = sorted(set((s[1], s[3]) for s in states))
    ang = {}
    Fcache = {}
    def _F(l2, l1):
        key = (l2, l1)
        if key not in Fcache:
            Fcache[key] = np.array([R[(n2, l2)] * R[(n1, l1)] * r * r for n2 in range(1, N_n + 1) for n1 in range(1, N_n + 1)])
        return Fcache[key]
    Phicache = {}
    def _Phi(l2, l1, k):
        key = (l2, l1, k)
        if key not in Phicache: Phicache[key] = _poisson_potentials(_F(l2, l1), r, w, k)
        return Phicache[key]
    V = np.zeros((N, N))
    for (le2, lh2) in lpairs:
        for (le1, lh1) in lpairs:
            if (le2, lh2) < (le1, lh1): continue
            block = np.zeros((N_n * N_n, N_n * N_n))
            for k in range(max(abs(le1 - le2), abs(lh1 - lh2)), min(le1 + le2, lh1 + lh2) + 1):
                if (le1 + le2 + k) % 2 or (lh1 + lh2 + k) % 2: continue
                a = _angular_factor(le2, lh2, le1, lh1, k, L)
                if a == 0.0: continue
                Rk = np.dot(_F(le2, le1) * w, _Phi(lh2, lh1, k).T)      # (Nn^2 e-pairs, Nn^2 h-pairs)
                # block index: row (ne2, nh2), col (ne1, nh1): Rk[(ne2,ne1),(nh2,nh1)]
                Rk4 = Rk.reshape(N_n, N_n, N_n, N_n)            # [ne2, ne1, nh2, nh1]
                block += a * Rk4.transpose(0, 2, 1, 3).reshape(N_n * N_n, N_n * N_n)
            rows = [idx[(ne, le2, nh, lh2)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            cols = [idx[(ne, le1, nh, lh1)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            V[np.ix_(rows, cols)] = -cc * block
            if (le2, lh2) != (le1, lh1): V[np.ix_(cols, rows)] = -cc * block.T
    return V

def _kinetic_diagonal(A, me, mh, states, Z):
    return np.array([_HB() / me * (Z[(ne, le)] / A) ** 2 + _HB() / mh * (Z[(nh, lh)] / A) ** 2 for (ne, le, nh, lh) in states])

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def _dipole_block(states1, states0, r, w, R):
    """<(states1) L=1, M=0| z_e - z_h |(states0) L=0> in nm"""
    def _ang_z(l2, l1, m):
        if abs(l1 - l2) != 1 or abs(m) > min(l1, l2): return 0.0
        l = min(l1, l2)
        return sqrt(((l + 1) ** 2 - m * m) / ((2 * l + 1) * (2 * l + 3)))
    radcache = {}
    def _rad(a2, a1):
        key = (a2, a1)
        if key not in radcache: radcache[key] = float(np.sum(w * R[a2] * R[a1] * r ** 3))
        return radcache[key]
    D = np.zeros((len(states1), len(states0)))
    angcache = {}
    def _angsum(le2, lh2, le1, lh1, which):
        key = (le2, lh2, le1, lh1, which)
        if key not in angcache:
            val = 0.0
            for m in range(-min(le1, lh1), min(le1, lh1) + 1):
                c1 = _cg(le1, m, lh1, -m, 0, 0); c2 = _cg(le2, m, lh2, -m, 1, 0)
                if c1 == 0.0 or c2 == 0.0: continue
                val += c1 * c2 * (_ang_z(le2, le1, m) if which == "e" else _ang_z(lh2, lh1, -m))
            angcache[key] = val
        return angcache[key]
    for j, (ne1, le1, nh1, lh1) in enumerate(states0):
        for i, (ne2, le2, nh2, lh2) in enumerate(states1):
            val = 0.0
            if (nh2, lh2) == (nh1, lh1) and abs(le2 - le1) == 1:
                val += _angsum(le2, lh2, le1, lh1, "e") * _rad((ne2, le2), (ne1, le1))
            if (ne2, le2) == (ne1, le1) and abs(lh2 - lh1) == 1:
                val -= _angsum(le2, lh2, le1, lh1, "h") * _rad((nh2, lh2), (nh1, lh1))
            D[i, j] = val
    return D

def _oracle_minimal_scr(A: float, me: float, mh: float, eps: float, N_r: int) -> "np.ndarray":
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    _check_pair(A, me, mh, eps, 1, 1)
    r, w, R, Z = _radial_functions(A, 1, 1, N_r)
    st0 = _pair_basis(1, 1, 0); st1 = _pair_basis(1, 1, 1)
    H0 = np.diag(_kinetic_diagonal(A, me, mh, st0, Z)) + _coulomb_matrix(A, eps, 1, 1, 0, N_r, st0, r, w, R)
    H1 = np.diag(_kinetic_diagonal(A, me, mh, st1, Z)) + _coulomb_matrix(A, eps, 1, 1, 1, N_r, st1, r, w, R)
    w0, v0 = eigh(H0); w1, v1 = eigh(H1)
    D = _dipole_block(st1, st0, r, w, R)
    d = np.dot(v1.T, np.dot(D, v0[:, 0]))
    i_e = int(np.argmax(w1)); i_h = 1 - i_e
    EC = H1[0, 1]                       # <(1s)(1p)| V |(1p)(1s)> coupling, eV (negative)
    c = -EC * A * eps / _E2()     # |E_C| in units of e^2/(4 pi eps A)
    return np.array([w0[0], w1[i_h] - w0[0], w1[i_e] - w0[0], d[i_h] ** 2, d[i_e] ** 2, EC, c])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nA = 10.0\nme = 0.07\nmh = 0.13\neps = 12.85\nN_r = 2001\n', 'call': 'minimal_scr(A, me, mh, eps, N_r)', 'gold_call': '_oracle_minimal_scr(A, me, mh, eps, N_r)'},
        {'setup': 'import numpy as np\nA = 15.0\nme = 0.07\nmh = 0.13\neps = 12.85\nN_r = 2001\n', 'call': 'minimal_scr(A, me, mh, eps, N_r)', 'gold_call': '_oracle_minimal_scr(A, me, mh, eps, N_r)'},
        {'setup': 'import numpy as np\n# boundary: a light electron, m_e/M = 0.25, where the kinetic splitting dominates the coupling\nA = 5.0\nme = 0.05\nmh = 0.15\neps = 12.85\nN_r = 1001\n', 'call': 'minimal_scr(A, me, mh, eps, N_r)', 'gold_call': '_oracle_minimal_scr(A, me, mh, eps, N_r)'},
        {'setup': 'import numpy as np\nA = 8.0\nme = 0.067\nmh = 0.45\neps = 12.85\nN_r = 2001\n', 'call': 'minimal_scr(A, me, mh, eps, N_r)', 'gold_call': '_oracle_minimal_scr(A, me, mh, eps, N_r)'},
        {'setup': 'import numpy as np\n# invalid input: a negative permittivity is unphysical and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: minimal_scr(10.0, 0.07, 0.13, -12.85, 1001))', 'gold_call': '_catches_value_error(lambda: _oracle_minimal_scr(10.0, 0.07, 0.13, -12.85, 1001))'},
    ]
