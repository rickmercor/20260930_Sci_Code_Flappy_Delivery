"""
Returns [E_0, g, <L_z>, <S_z>, E_2 - E_0] for the ground Kramers doublet of the full model at zero classical field.

The g-factor measured by electron paramagnetic resonance on a Kramers doublet is set by how the doublet's orbital and spin moments respond to the field; treating the Zeeman term as a first-order perturbation of the exactly computed doublet gives that response without any finite-field extrapolation, and the same construction survives when the cavity has dressed the spin with photons.

Returns
-------
A numpy float64 array of shape (5,): [E_0 in cm^-1, g, <L_z>, <S_z>, E_2 - E_0 in cm^-1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kramers_g_factor(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Returns [E_0, g, <L_z>, <S_z>, E_2 - E_0] for the ground Kramers doublet of the full model at zero classical field.

    E_0 is the doublet energy, g its exact effective g-factor (the splitting of the two eigenvalues of the
    doublet-projected Zeeman operator s L_z + g_e S_z), <L_z> and <S_z> the orbital and spin moments in the doublet
    state of larger Zeeman eigenvalue, and E_2 - E_0 the distance to the next level. With g_c > 0 and n_ph >= 1 the
    doublet is the cavity-dressed one.

    Model and conventions. A trigonal transition-metal complex with a doubly degenerate 2E ground term carries a
    single electron (single-particle scenario) or a single hole (single-hole scenario) in the orbital doublet |m_l =
    +1>, |m_l = -1> with spin |up>, |down>, coupled linearly to a doubly degenerate vibrational e mode and to one
    quantized cavity mode. Units cm^-1 for every energy and tesla for the classical field, with the Bohr magneton mu_B
    = 0.4668644814 cm^-1/T and the free-electron g-factor g_e = 2.00231930436.

    Electronic operators: L_z = |1><1| - |-1><-1|, S_z = (|up><up| - |down><down|)/2, spin-orbit coupling H_soc = xi
    L_z S_z, classical Zeeman interaction H_Zee = mu_B B_z (s L_z + g_e S_z) with s = +1 for the single-particle and s
    = -1 for the single-hole scenario (the orbital moment of the hole is reversed), and sigma_y = i(|down><up| -
    |up><down|).

    Vibrational e mode: two oscillator coordinates x, y of energy hw, basis states |n_x, n_y> with n_x + n_y <= n_max
    ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending (N = (n_max + 1)(n_max + 2)/2
    states); annihilation operators a_x, a_y with <n_x - 1, n_y| a_x |n_x, n_y> = sqrt(n_x) and <n_x, n_y - 1| a_y
    |n_x, n_y> = sqrt(n_y); dimensionless coordinates x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, Q_+- = x +- i
    y, number operator n_vib = a_x^+ a_x + a_y^+ a_y, vibrational angular momentum l_vib = i (a_x a_y^+ - a_x^+ a_y),
    H_vib = hw (n_vib + 1). Every operator matrix is the exact matrix of the untruncated operator between retained
    states (products evaluated in a basis extended by two shells and cut back), never a product of truncated matrices.

    Linear Jahn-Teller coupling H_JT = F (Q_+ |1><-1| + Q_- |-1><1|) with F = sqrt(2 E_JT hw), so that E_JT = F^2/(2
    hw) is the classical Jahn-Teller stabilization energy; the frozen-distortion (static) product F rho is identified
    with its value at the classical minimum, F rho = 2 E_JT. The static model keeps F rho as a fixed parameter and has
    no vibrational dynamics: H_static = xi L_z S_z + F rho (|1><-1| + |-1><1|) + H_Zee on the four electronic-spin
    states, ordered (m_l = +1, -1) x (m_s = up, down).

    Cavity: one mode of energy hwc with photon states |0>, ..., |n_ph>, creation operator b^+, H_c = hwc b^+ b, and
    the cavity Zeeman interaction H_cZ = i g_c sigma_y (b^+ - b) with the coupling energy g_c; H_cZ acts on spin and
    photon only. Full model H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ on the product space (m_l = +1, -1) x (m_s =
    up, down) x oscillator basis x photon number 0..n_ph, index (((i_ml 2 + i_ms) N + i_vib)(n_ph + 1) + i_ph),
    dimension D = 4 N (n_ph + 1); J = l_vib - L_z/2 commutes with H and labels every level by a half-integer |j|.

    Kramers doublet and g-factor: at zero classical field the ground level is a Kramers doublet; its effective
    g-factor is the eigenvalue splitting of the 2 x 2 matrix of the Zeeman operator per mu_B B_z, s L_z + g_e S_z,
    projected on the doublet (first-order degenerate perturbation theory, the B_z -> 0 limit of the Zeeman splitting
    divided by mu_B B_z); <L_z> and <S_z> are the expectation values in the doublet state of larger Zeeman eigenvalue.
    Cavity correction: g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) defines the dimensionless leading coefficient
    kappa, obtained by exact second-order perturbation theory in H_cZ (one-photon intermediate states; the doublet
    energy shifts by w2 g_c^2 with a Kramers-degenerate w2; the doublet's Zeeman matrix changes through the first- and
    second-order corrections of its two eigenvectors including their normalization); kappa_fd = (g(g_c) - g(0))
    (hwc/g_c)^2 is its finite-coupling estimate. Spin-free reference: with xi = 0 and no spin the ground vibronic
    level of the linear E x e problem is a doublet with |j| = 1/2 whose Ham reduction factors p and q are the
    magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on it.

    Reference parameters: xi = 800 cm^-1, E_JT = 800 cm^-1 (F rho = 1600 cm^-1), hw = 300 cm^-1, hwc = 3200 cm^-1
    (audit set 1600, 3200, 6400 cm^-1), g_c = 100 cm^-1, n_max = 20, n_ph = 2, single-particle scenario.

    Args:
        xi: positive float, the spin-orbit coupling strength in cm^-1.
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        g_c: non-negative float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: non-negative integer, the highest photon number retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (5,): [E_0 in cm^-1, g, <L_z>, <S_z>, E_2 - E_0 in cm^-1].

    Raises:
        ValueError: if the two lowest levels are not a degenerate Kramers doublet, if a third level is degenerate with
        them, or if a Hamiltonian argument is invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh
from functools import lru_cache

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _kramers_g(H0: np.ndarray, Z: np.ndarray, extra: list):
    """Exact g-factor of the lowest Kramers doublet of H0 from the Zeeman operator Z (per mu_B B_z):
    eigenvalue splitting of the 2 x 2 projection of Z; also expectation values of the operators in `extra`
    in the doublet state of larger Zeeman eigenvalue. Z and the entries of `extra` may be dense matrices or
    the 1-D diagonals of diagonal operators."""
    E, C = eigh(H0, driver="evr", subset_by_index=[0, min(3, H0.shape[0] - 1)])
    if E[1] - E[0] > 1e-8: raise ValueError("the two lowest levels are not a degenerate Kramers doublet")
    if H0.shape[0] > 2 and E[2] - E[1] < 1e-8: raise ValueError("more than two degenerate lowest levels")
    P = C[:, :2]
    M = P.conj().T @ _apply(Z, P)
    w, U = np.linalg.eigh(M)
    v = P @ U[:, 1]
    vals = [float(np.real(np.vdot(v, _apply(A, v)))) for A in extra]
    return float(E[0]), float(w[1] - w[0]), vals, (E[2] - E[0] if H0.shape[0] > 2 else 0.0)

def _model_operators(n_max: int, n_ph: int, particle: bool, which: str = "ZJLSV"):
    """Zeeman (per mu_B B_z), conserved J = l_vib - L_z/2, L_z, S_z and the cavity coupling operator
    i sigma_y (b^+ - b) on the full product space, ordering (m_l, m_s, vibration, photon). `which` selects
    which of Z, J, L_z, S_z, V to build; the others are returned as None. L_z, S_z and Z are diagonal in this
    ordering and are returned as their 1-D diagonals, to be applied with `_apply`; only J and V are dense."""
    ops = _oracle_vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    nb = n_ph + 1
    blk = np.repeat(np.arange(4), N * nb)
    lz_d = np.where(blk < 2, 1.0, -1.0); sz_d = np.where(blk % 2 == 0, 0.5, -0.5)
    Lz = lz_d.astype(complex) if "L" in which else None
    Sz = sz_d.astype(complex) if "S" in which else None
    Z = (s * lz_d + _GE() * sz_d).astype(complex) if "Z" in which else None
    J = None
    if "J" in which:
        J = np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), ops[3]), np.eye(nb))
        J[np.diag_indices_from(J)] -= 0.5 * lz_d
    V = None
    if "V" in which:
        sy = np.array([[0.0, -1j], [1j, 0.0]]); bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
        V = 1j * np.kron(np.kron(np.kron(np.eye(2), sy), np.eye(N)), bph.T - bph)
    return Z, J, Lz, Sz, V

@lru_cache(maxsize=256)
def _kramers_doublet(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool):
    """Cached core of the Kramers-doublet solve, returned as a plain tuple so no caller can mutate it.
    The calculation is pure in its arguments, and the audit of the last step asks for the same
    finite-coupling doublet twice per cavity energy: once directly, once through the cavity shift."""
    H0 = _oracle_model_hamiltonian(xi, E_JT, hw, 0.0, g_c, hwc, n_max, n_ph, particle)
    Z, J, Lz, Sz, V = _model_operators(n_max, n_ph, particle, "ZLS")
    E0, g, vals, gap = _kramers_g(H0, Z, [Lz, Sz])
    return (E0, g, vals[0], vals[1], float(gap))

def _oracle_kramers_g_factor(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Exact effective g-factor of the ground Kramers doublet of the full model at zero classical field:
    [E_0, g, <L_z>, <S_z>, E_2 - E_0], with <L_z>, <S_z> in the doublet state of larger Zeeman eigenvalue."""
    if int(n_ph) != n_ph or n_ph < 0: raise ValueError("n_ph must be a non-negative integer")
    return np.array(_kramers_doublet(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 0.0\nhwc = 3200.0\nn_max = 10\nn_ph = 0\nparticle = True\n',
         'call': 'kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 100.0\nhwc = 3200.0\nn_max = 8\nn_ph = 2\nparticle = True\n',
         'call': 'kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 0.0\nhwc = 3200.0\nn_max = 10\nn_ph = 0\nparticle = False\n',
         'call': 'kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: weak Jahn-Teller coupling with the cavity present\nxi = 400.0\nE_JT = 150.0\nhw = 300.0\ng_c = 50.0\nhwc = 1600.0\nn_max = 8\nn_ph = 2\nparticle = False\n',
         'call': 'kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: a negative photon count must raise ValueError\ng_c = 0.0\nhwc = 3200.0\nn_max = 4\nn_ph = -1\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle))'},
    ]
