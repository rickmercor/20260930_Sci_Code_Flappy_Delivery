"""
Returns [E_e, E_h, |d_e|^2, |d_h|^2] for the non-interacting electron-hole pair in the sphere: the 1s -> 1p transition energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) of the electron and of the hole (eV) and their squared dipoles |<1p_z| z |1s>|^2 = (int R_{11} R_{10} r^3 dr)^2/3 (nm^2), exact to 1e-10.

Without the Coulomb interaction each carrier responds separately, with a resonance inversely proportional to its mass and a dipole set by the box alone; these bare values are the reference against which the Coulomb renormalisation of the confined pair is measured.

Returns
-------
A float64 array of shape (4,), [E_e, E_h, |d_e|^2, |d_h|^2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bare_response(A: float, me: float, mh: float, N_r: int) -> "np.ndarray":
    """Returns [E_e, E_h, |d_e|^2, |d_h|^2] for the non-interacting electron-hole pair in the sphere: the 1s -> 1p
    transition energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) of the electron and of the hole (eV) and their squared
    dipoles |<1p_z| z |1s>|^2 = (int R_{11} R_{10} r^3 dr)^2/3 (nm^2), exact to 1e-10.

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
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float64 array of shape (4,), [E_e, E_h, |d_e|^2, |d_h|^2].

    Raises:
        ValueError: for a non-positive A or mass, or for an invalid N_r.
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

def _oracle_bare_response(A: float, me: float, mh: float, N_r: int) -> "np.ndarray":
    if not (A > 0 and me > 0 and mh > 0): raise ValueError("A, me, mh must be positive")
    r, w, R, Z = _radial_functions(A, 1, 1, N_r)
    dE = (Z[(1, 1)] ** 2 - Z[(1, 0)] ** 2) / A ** 2
    d2 = float(np.sum(w * R[(1, 1)] * R[(1, 0)] * r ** 3)) ** 2 / 3.0
    return np.array([_HB() / me * dE, _HB() / mh * dE, d2, d2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nA = 10.0\nme = 0.07\nmh = 0.13\nN_r = 2001\n', 'call': 'bare_response(A, me, mh, N_r)', 'gold_call': '_oracle_bare_response(A, me, mh, N_r)'},
        {'setup': 'import numpy as np\nA = 15.0\nme = 0.07\nmh = 0.13\nN_r = 2001\n', 'call': 'bare_response(A, me, mh, N_r)', 'gold_call': '_oracle_bare_response(A, me, mh, N_r)'},
        {'setup': 'import numpy as np\n# boundary: equal masses give equal electron and hole transitions\nA = 5.0\nme = 0.5\nmh = 0.5\nN_r = 1001\n', 'call': 'bare_response(A, me, mh, N_r)', 'gold_call': '_oracle_bare_response(A, me, mh, N_r)'},
        {'setup': 'import numpy as np\nA = 8.0\nme = 0.067\nmh = 0.45\nN_r = 2001\n', 'call': 'bare_response(A, me, mh, N_r)', 'gold_call': '_oracle_bare_response(A, me, mh, N_r)'},
        {'setup': 'import numpy as np\n# invalid input: a vanishing radius is not a sphere and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: bare_response(0.0, 0.07, 0.13, 1001))', 'gold_call': '_catches_value_error(lambda: _oracle_bare_response(0.0, 0.07, 0.13, 1001))'},
    ]
