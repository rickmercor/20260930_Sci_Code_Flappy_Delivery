"""
Finds by root finding (to 1e-10 nm) the radius A_10 in [A_lo, A_hi] at which the electron-like renormalisation factor f_e(A) = |d_e|^2/|d_bare|^2 of the exact model with the basis (N_n, L_max) reaches the given level (1.1 for a ten percent enhancement), where d_e belongs to the largest-weight L = 1 resonance and d_bare is the non-interacting 1s -> 1p dipole of the same sphere.

Below some crystal radius the electron and the hole may be treated as independent particles; the source estimates that radius analytically from its two-state model and finds it much smaller than the exciton Bohr radius, and the exact model can locate it without any of the approximations of that estimate.

Returns
-------
A float, the independence radius A_10 in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def independence_radius(me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, A_lo: float, A_hi: float, level: float) -> float:
    """Finds by root finding (to 1e-10 nm) the radius A_10 in [A_lo, A_hi] at which the electron-like renormalisation
    factor f_e(A) = |d_e|^2/|d_bare|^2 of the exact model with the basis (N_n, L_max) reaches the given level (1.1 for
    a ten percent enhancement), where d_e belongs to the largest-weight L = 1 resonance and d_bare is the
    non-interacting 1s -> 1p dipole of the same sphere.

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
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        A_lo: float, the lower end of the bracket in nm on which the root is sought (0 < A_lo < A_hi).
        A_hi: float, the upper end of the bracket in nm.
        level: float, the target renormalisation factor f_e (above 1; 1.1 for a ten percent enhancement).

    Returns:
        A float, the independence radius A_10 in nm.

    Raises:
        ValueError: unless 0 < A_lo < A_hi and level > 1, for invalid model arguments, or if the level is not
        bracketed by [A_lo, A_hi].
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

def _renormalisation(A, me, mh, eps, N_n, L_max, N_r):
    """[dE_e, d2_e, dE_h, d2_h, E0, renorm_e, renorm_h, ratio] of the exact model: the electron-like resonance carries
    the largest weight dE |d|^2 and the hole-like one is the largest weight among the resonances below it"""
    tr = _oracle_thz_transitions(A, me, mh, eps, N_n, L_max, N_r, len(_pair_basis(N_n, L_max, 1)))
    ie = int(np.argmax(tr[:, 2]))                      # electron-like: the largest resonance weight
    below = [k for k in range(ie) if tr[k, 0] < tr[ie, 0]]
    if not below: raise ValueError("no resonance below the dominant one")
    ih = int(below[int(np.argmax(tr[below, 2]))])        # hole-like: the largest weight below it
    bare = _oracle_bare_response(A, me, mh, N_r)
    return np.array([tr[ie, 0], tr[ie, 1], tr[ih, 0], tr[ih, 1], tr[0, 3], tr[ie, 1] / bare[2], tr[ih, 1] / bare[3], tr[ih, 2] / tr[ie, 2]])

def _oracle_independence_radius(me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, A_lo: float, A_hi: float, level: float) -> float:
    if not (0 < A_lo < A_hi) or not (level > 1.0): raise ValueError("0 < A_lo < A_hi and level > 1 required")
    _f = lambda A: _renormalisation(A, me, mh, eps, N_n, L_max, N_r)[5] - level
    flo, fhi = _f(A_lo), _f(A_hi)
    if flo * fhi > 0: raise ValueError("the renormalisation level is not bracketed by [A_lo, A_hi]")
    return brentq(_f, A_lo, A_hi, xtol=1e-10, rtol=1e-12, maxiter=100)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nme = 0.07\nmh = 0.13\neps = 12.85\nN_n = 4\nL_max = 3\nN_r = 1001\nA_lo = 3.0\nA_hi = 9.0\nlevel = 1.1\n', 'call': 'float(independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'gold_call': 'float(_oracle_independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'tol': 1e-07},
        {'setup': 'import numpy as np\nme = 0.09\nmh = 0.11\neps = 12.85\nN_n = 4\nL_max = 3\nN_r = 1001\nA_lo = 0.5\nA_hi = 2.5\nlevel = 1.1\n', 'call': 'float(independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'gold_call': 'float(_oracle_independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# boundary: a five percent level, reached at a smaller radius than the ten percent one\nme = 0.07\nmh = 0.13\neps = 12.85\nN_n = 3\nL_max = 2\nN_r = 1001\nA_lo = 2.0\nA_hi = 8.0\nlevel = 1.05\n', 'call': 'float(independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'gold_call': 'float(_oracle_independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level))', 'tol': 1e-07},
        {'setup': 'import numpy as np\n# invalid input: an inverted bracket must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n', 'call': '_catches_value_error(lambda: independence_radius(0.07, 0.13, 12.85, 2, 1, 401, 9.0, 3.0, 1.1))', 'gold_call': '_catches_value_error(lambda: _oracle_independence_radius(0.07, 0.13, 12.85, 2, 1, 401, 9.0, 3.0, 1.1))'},
    ]
