"""
Evaluates the radial two-particle Coulomb integral R^k[(ne2 le2)(ne1 le1); (nh2 lh2)(nh1 lh1)] = int_0^A int_0^A R_{ne2 le2}(r_e) R_{ne1 le1}(r_e) r_e^2 R_{nh2 lh2}(r_h) R_{nh1 lh1}(r_h) r_h^2 r_<^k / r_>^{k+1} dr_e dr_h in nm^-1 for the normalised box radial functions of the sphere of radius A, exact to 1e-10 (r_< and r_> are the smaller and the larger of r_e and r_h).

Every Coulomb matrix element between confined pair states reduces to radial integrals of this form, one per multipole order k; the kernel is the electrostatic potential of a shell of charge, whose derivative jumps where the two radii coincide, and getting these integrals exact is what makes the exact model exact.

Returns
-------
A float, the radial integral R^k in nm^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coulomb_integral(A: float, k: int, ne2: int, le2: int, ne1: int, le1: int, nh2: int, lh2: int, nh1: int, lh1: int, N_r: int) -> float:
    """Evaluates the radial two-particle Coulomb integral R^k[(ne2 le2)(ne1 le1); (nh2 lh2)(nh1 lh1)] = int_0^A int_0^A
    R_{ne2 le2}(r_e) R_{ne1 le1}(r_e) r_e^2 R_{nh2 lh2}(r_h) R_{nh1 lh1}(r_h) r_h^2 r_<^k / r_>^{k+1} dr_e dr_h in
    nm^-1 for the normalised box radial functions of the sphere of radius A, exact to 1e-10 (r_< and r_> are the
    smaller and the larger of r_e and r_h).

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
        k: int, the multipole order of the Coulomb kernel (k >= 0).
        ne2: int, the radial quantum number (n >= 1) of the electron state R_{ne2 le2}.
        le2: int, the angular momentum (l >= 0) of the electron state R_{ne2 le2}.
        ne1: int, the radial quantum number (n >= 1) of the electron state R_{ne1 le1}.
        le1: int, the angular momentum (l >= 0) of the electron state R_{ne1 le1}.
        nh2: int, the radial quantum number (n >= 1) of the hole state R_{nh2 lh2}.
        lh2: int, the angular momentum (l >= 0) of the hole state R_{nh2 lh2}.
        nh1: int, the radial quantum number (n >= 1) of the hole state R_{nh1 lh1}.
        lh1: int, the angular momentum (l >= 0) of the hole state R_{nh1 lh1}.
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float, the radial integral R^k in nm^-1.

    Raises:
        ValueError: if A <= 0, k < 0, a quantum number is invalid (n >= 1, l >= 0 required) or N_r is not an odd
        integer of at least 201.
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

def _oracle_coulomb_integral(A: float, k: int, ne2: int, le2: int, ne1: int, le1: int, nh2: int, lh2: int, nh1: int, lh1: int, N_r: int) -> float:
    if not (A > 0) or k < 0: raise ValueError("A > 0 and k >= 0 required")
    if min(ne1, ne2, nh1, nh2) < 1 or min(le1, le2, lh1, lh2) < 0: raise ValueError("radial quantum numbers n >= 1 and l >= 0 required")
    N_n = max(ne1, ne2, nh1, nh2); L_max = max(le1, le2, lh1, lh2)
    r, w, R, Z = _radial_functions(A, N_n, L_max, N_r)
    Fe = (R[(ne2, le2)] * R[(ne1, le1)] * r * r)[None, :]
    Fh = (R[(nh2, lh2)] * R[(nh1, lh1)] * r * r)[None, :]
    Phi = _poisson_potentials(Fh, r, w, k)
    return float(np.sum(w * Fe[0] * Phi[0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nA = 10.0\nk = 0\nne2, le2, ne1, le1 = 1, 0, 1, 0\nnh2, lh2, nh1, lh1 = 1, 0, 1, 0\nN_r = 2001\n', 'call': 'float(coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))', 'gold_call': 'float(_oracle_coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))'},
        {'setup': 'import numpy as np\nA = 10.0\nk = 1\nne2, le2, ne1, le1 = 1, 1, 1, 0\nnh2, lh2, nh1, lh1 = 1, 0, 1, 1\nN_r = 2001\n', 'call': 'float(coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))', 'gold_call': 'float(_oracle_coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))'},
        {'setup': 'import numpy as np\nA = 15.0\nk = 2\nne2, le2, ne1, le1 = 2, 2, 1, 0\nnh2, lh2, nh1, lh1 = 3, 1, 1, 1\nN_r = 2001\n', 'call': 'float(coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))', 'gold_call': 'float(_oracle_coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))'},
        {'setup': 'import numpy as np\nA = 5.0\nk = 0\nne2, le2, ne1, le1 = 2, 1, 2, 1\nnh2, lh2, nh1, lh1 = 1, 2, 3, 2\nN_r = 1001\n', 'call': 'float(coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))', 'gold_call': 'float(_oracle_coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))'},
        {'setup': 'import numpy as np\n# boundary: k = 3 is the only multipole order the triangle rule allows between these s and f states\nA = 7.5\nk = 3\nne2, le2, ne1, le1 = 1, 3, 2, 0\nnh2, lh2, nh1, lh1 = 2, 0, 1, 3\nN_r = 2001\n', 'call': 'float(coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))', 'gold_call': 'float(_oracle_coulomb_integral(A, k, ne2, le2, ne1, le1, nh2, lh2, nh1, lh1, N_r))'},
        {'setup': 'import numpy as np\n# invalid input: an even number of Simpson points is not a Simpson grid and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: coulomb_integral(10.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2000))', 'gold_call': '_catches_value_error(lambda: _oracle_coulomb_integral(10.0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2000))'},
    ]
