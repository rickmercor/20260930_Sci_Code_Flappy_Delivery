"""
Solves the source's weak-confinement model and returns [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2]: the lowest l = 0 and l = 1 eigenvalues (eV) of the relative-motion radial equation with the shrunken-exciton centre-of-mass term chi(r) on 0 < r < r_max, and the relative-coordinate squared dipole (int R_2p R_1s r^3 dr)^2/3 in nm^2, each converged to 1e-9 eV or 1e-10 nm^2 (the reference discretises u = r R with three-point finite differences on N_g, 2 N_g and 4 N_g intervals and extrapolates the O(h^2) and O(h^4) errors to zero; the potential diverges as (r_max - r)^-2 at the outer wall, which any method must respect). With eps infinite the Coulomb term is omitted, which gives the model's strongly confined limit, in which the transition energy scales as 1/A^2.

For a crystal large compared with the exciton, the source keeps the centre of mass in a box shrunk by the size of the exciton and lets that size be set by the relative coordinate itself; the resulting one-dimensional problem is the source's bridge between the bulk exciton and the strongly confined pair, and its predictions can be tested against the exact model wherever the latter converges.

Returns
-------
A float64 array of shape (4,), [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wcr_model(A: float, me: float, mh: float, eps: float, N_g: int) -> "np.ndarray":
    """Solves the source's weak-confinement model and returns [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2]: the lowest
    l = 0 and l = 1 eigenvalues (eV) of the relative-motion radial equation with the shrunken-exciton centre-of-mass
    term chi(r) on 0 < r < r_max, and the relative-coordinate squared dipole (int R_2p R_1s r^3 dr)^2/3 in nm^2, each
    converged to 1e-9 eV or 1e-10 nm^2 (the reference discretises u = r R with three-point finite differences on N_g,
    2 N_g and 4 N_g intervals and extrapolates the O(h^2) and O(h^4) errors to zero; the potential diverges as (r_max
    - r)^-2 at the outer wall, which any method must respect). With eps infinite the Coulomb term is omitted, which
    gives the model's strongly confined limit, in which the transition energy scales as 1/A^2.

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
        eps: float, the relative permittivity that screens the Coulomb attraction (positive); an infinite value omits
            the Coulomb term.
        N_g: int, the number of uniform intervals of the finite-difference grid on [0, r_max] at the coarsest
            Richardson level; at least 500.

    Returns:
        A float64 array of shape (4,), [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2].

    Raises:
        ValueError: for a non-positive A, mass or eps, or for N_g < 500.
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

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def _wcr_fd(A, me, mh, eps, N_g):
    """three-point finite-difference eigenvalues of the weak-confinement radial equation for l = 0 and 1 on the
    uniform grid of N_g intervals of [0, r_max], r_max = A min(me, mh)/mu, with u = r R and u(0) = u(r_max) = 0"""
    M = me + mh; mu = me * mh / M
    r_max = A * min(me, mh) / mu
    h = r_max / N_g
    x = h * np.arange(1, N_g)
    rho = mu * x / min(me, mh)
    chi = _HB() / M * (pi / (A - rho)) ** 2
    res = []
    for l in (0, 1):
        Vl = -_E2() / eps / x + _HB() / mu * l * (l + 1) / x ** 2 + chi
        diag = _HB() / mu * 2.0 / h ** 2 + Vl
        off = -_HB() / mu / h ** 2 * np.ones(N_g - 2)
        wl, vl = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        u = vl[:, 0] / sqrt(np.sum(vl[:, 0] ** 2) * h)
        res.append((float(wl[0]), u, x, h))
    E1s, u1s, x, h = res[0]; E2p, u2p, _, _ = res[1]
    d = np.sum(u2p * u1s * x) * h / sqrt(3.0)
    return np.array([E1s, E2p, E2p - E1s, d * d])

def _oracle_wcr_model(A: float, me: float, mh: float, eps: float, N_g: int) -> "np.ndarray":
    _check_pair(A, me, mh, eps, 1, 1)
    if not (isinstance(N_g, (int, np.integer)) and N_g >= 500): raise ValueError("N_g must be an integer >= 500")
    # the finite-difference error is O(h^2) with an O(h^4) remainder: two Richardson levels on N_g, 2 N_g, 4 N_g
    e1 = _wcr_fd(A, me, mh, eps, N_g); e2 = _wcr_fd(A, me, mh, eps, 2 * N_g); e4 = _wcr_fd(A, me, mh, eps, 4 * N_g)
    r12 = (4.0 * e2 - e1) / 3.0; r24 = (4.0 * e4 - e2) / 3.0
    return (16.0 * r24 - r12) / 15.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nA = 15.0\nme = 0.07\nmh = 0.13\neps = 12.85\nN_g = 1000\n', 'call': 'wcr_model(A, me, mh, eps, N_g)', 'gold_call': '_oracle_wcr_model(A, me, mh, eps, N_g)', 'tol': 1e-07},
        {'setup': 'import numpy as np\nA = 10.0\nme = 0.07\nmh = 0.13\neps = 12.85\nN_g = 1000\n', 'call': 'wcr_model(A, me, mh, eps, N_g)', 'gold_call': '_oracle_wcr_model(A, me, mh, eps, N_g)', 'tol': 1e-07},
        {'setup': 'import numpy as np\nA = 30.0\nme = 0.067\nmh = 0.45\neps = 12.85\nN_g = 1000\n', 'call': 'wcr_model(A, me, mh, eps, N_g)', 'gold_call': '_oracle_wcr_model(A, me, mh, eps, N_g)', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# boundary: the Coulomb term omitted (eps infinite), the strongly confined limit of the model\nA = 15.0\nme = 0.07\nmh = 0.13\neps = np.inf\nN_g = 1000\n', 'call': 'wcr_model(A, me, mh, eps, N_g)', 'gold_call': '_oracle_wcr_model(A, me, mh, eps, N_g)', 'tol': 1e-07},
        {'setup': 'import numpy as np\nA = 6.0\nme = 0.5\nmh = 0.5\neps = 17.0\nN_g = 800\n', 'call': 'wcr_model(A, me, mh, eps, N_g)', 'gold_call': '_oracle_wcr_model(A, me, mh, eps, N_g)', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: a grid below the stated minimum cannot reach the required convergence and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: wcr_model(15.0, 0.07, 0.13, 12.85, 100))', 'gold_call': '_catches_value_error(lambda: _oracle_wcr_model(15.0, 0.07, 0.13, 12.85, 100))'},
    ]
