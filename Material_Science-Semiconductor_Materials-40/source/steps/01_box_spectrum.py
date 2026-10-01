"""
Tabulates the first N_n positive zeros z_{ln} of the spherical Bessel functions j_l for l = 0..L_max, each to 1e-13 (row l, column n - 1), which fix the single-particle box states of a particle confined in the sphere of radius A: wavevectors z_{ln}/A and kinetic energies hbar^2 z_{ln}^2/(2 m A^2).

The particle-in-a-sphere states are the building blocks of every strong-confinement description of a nanocrystal; their exact zeros, not the large-argument asymptotics, are needed because the lowest states carry most of the weight.

Returns
-------
A float64 array of shape (L_max + 1, N_n) holding the Bessel zeros z_{ln}.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def box_spectrum(A: float, N_n: int, L_max: int) -> "np.ndarray":
    """Tabulates the first N_n positive zeros z_{ln} of the spherical Bessel functions j_l for l = 0..L_max, each to
    1e-13 (row l, column n - 1), which fix the single-particle box states of a particle confined in the sphere of
    radius A: wavevectors z_{ln}/A and kinetic energies hbar^2 z_{ln}^2/(2 m A^2).

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
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).

    Returns:
        A float64 array of shape (L_max + 1, N_n) holding the Bessel zeros z_{ln}.

    Raises:
        ValueError: if A <= 0, N_n < 1 or L_max < 0.
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

def _oracle_box_spectrum(A: float, N_n: int, L_max: int) -> "np.ndarray":
    if not (A > 0) or not (isinstance(N_n, (int, np.integer)) and N_n >= 1) or not (isinstance(L_max, (int, np.integer)) and L_max >= 0):
        raise ValueError("A > 0, N_n >= 1 and L_max >= 0 required")
    Zt = np.zeros((L_max + 1, N_n))
    for l in range(L_max + 1): Zt[l] = _bessel_zeros(l, N_n)
    return Zt

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nA = 10.0\nN_n = 6\nL_max = 5\n', 'call': 'box_spectrum(A, N_n, L_max)', 'gold_call': '_oracle_box_spectrum(A, N_n, L_max)'},
        {'setup': 'import numpy as np\nA = 5.0\nN_n = 3\nL_max = 2\n', 'call': 'box_spectrum(A, N_n, L_max)', 'gold_call': '_oracle_box_spectrum(A, N_n, L_max)'},
        {'setup': 'import numpy as np\nA = 15.0\nN_n = 8\nL_max = 7\n', 'call': 'box_spectrum(A, N_n, L_max)', 'gold_call': '_oracle_box_spectrum(A, N_n, L_max)'},
        {'setup': 'import numpy as np\n# boundary: the smallest basis, a single s state whose zero is pi\nA = 2.5\nN_n = 1\nL_max = 0\n', 'call': 'box_spectrum(A, N_n, L_max)', 'gold_call': '_oracle_box_spectrum(A, N_n, L_max)'},
        {'setup': 'import numpy as np\n# invalid input: a non-positive radius is not a sphere and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: box_spectrum(-1.0, 3, 2))', 'gold_call': '_catches_value_error(lambda: _oracle_box_spectrum(-1.0, 3, 2))'},
    ]
