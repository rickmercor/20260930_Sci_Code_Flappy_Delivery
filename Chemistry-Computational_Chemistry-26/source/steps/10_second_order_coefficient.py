"""
Returns [kappa, w2 hwc, g(0)]: the exact leading cavity coefficient kappa of the ground Kramers doublet's g-factor, the second-order doublet energy shift expressed through the dimensionless w2 hwc, and the cavity-free g-factor, from second-order perturbation theory in the cavity Zeeman interaction.

Because the cavity Zeeman interaction changes the photon number, its first-order effect on the doublet vanishes and the g-factor responds only at second order, through the admixture of one-photon states with reversed spin; the leading coefficient follows from how those admixtures modify the doublet's Zeeman matrix, a calculation in which the two spin sectors of the intermediate manifold enter together.

Returns
-------
A numpy float64 array of shape (3,): [kappa, w2 hwc, g(0)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def second_order_coefficient(xi: float, E_JT: float, hw: float, hwc: float, n_max: int, particle: bool) -> "np.ndarray":
    """Returns [kappa, w2 hwc, g(0)]: the exact leading cavity coefficient kappa of the ground Kramers doublet's g-factor, the second-order doublet energy shift expressed through the dimensionless w2 hwc, and the cavity-free g-factor, from second-order perturbation theory in the cavity Zeeman interaction.

    The expansion is g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) and E_0(g_c) = E_0(0) + w2 g_c^2 + O(g_c^4),
    evaluated about the zero-field, zero-coupling eigenbasis with one-photon intermediate states: the first-order
    vectors of the two doublet states, their second-order vectors including the normalization correction, the
    Kramers-degenerate second-order energy matrix, and the second-order change of the doublet's Zeeman matrix, from
    which kappa is the change of the eigenvalue splitting per (g_c/hwc)^2.

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
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (3,): [kappa, w2 hwc, g(0)].

    Raises:
        ValueError: if the lowest level is not an isolated Kramers doublet, if the second-order doublet shift is not
        Kramers-degenerate, or if a Hamiltonian argument is invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

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

def _second_order_kappa(H0: np.ndarray, Z: np.ndarray, V: np.ndarray, hwc: float):
    """Exact second-order coefficient of the Kramers-doublet g-factor in the coupling g_c multiplying V:
    kappa = hwc^2 d^2 g / d g_c^2 / 2 ... expressed as g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4);
    also returns the second-order doublet energy shift coefficient w2 (E_0(g_c) = E_0 + w2 g_c^2 + ...).
    Z may be a dense matrix or the 1-D diagonal of a diagonal operator."""
    E, C = eigh(H0, driver="evr")
    if E[1] - E[0] > 1e-8 or E[2] - E[1] < 1e-8: raise ValueError("the lowest level is not an isolated Kramers doublet")
    E0 = E[0]; P0 = C[:, :2]; Q = C[:, 2:]; EQ = E[2:]
    if np.abs(P0.conj().T @ (V @ P0)).max() > 1e-10: raise ValueError("the coupling has matrix elements inside the doublet")
    Rd = 1.0 / (E0 - EQ)
    Vq = Q.conj().T @ (V @ P0)
    A1 = Q @ (Rd[:, None] * Vq)
    A2 = Q @ (Rd[:, None] * (Q.conj().T @ (V @ A1)))
    norm2 = -0.5 * np.sum(np.abs(Vq) ** 2 * (Rd ** 2)[:, None], axis=0)
    A2 = A2 + P0 * norm2[None, :]
    W = Vq.conj().T @ (Rd[:, None] * Vq)
    w2 = float(np.real(W[0, 0]))
    if abs(W[0, 0] - W[1, 1]) > 1e-9 * max(1.0, abs(w2)) or abs(W[0, 1]) > 1e-9 * max(1.0, abs(w2)):
        raise ValueError("the second-order doublet shift is not Kramers-degenerate")
    M0 = P0.conj().T @ _apply(Z, P0)
    M1 = P0.conj().T @ _apply(Z, A1) + A1.conj().T @ _apply(Z, P0)
    M2 = P0.conj().T @ _apply(Z, A2) + A2.conj().T @ _apply(Z, P0) + A1.conj().T @ _apply(Z, A1)
    w0, U = np.linalg.eigh(M0)
    m1 = U.conj().T @ M1 @ U; m2 = U.conj().T @ M2 @ U
    g0 = float(w0[1] - w0[0])
    c2 = float(np.real(m2[1, 1] - m2[0, 0]))
    if g0 > 1e-12: c2 += 2.0 * float(abs(m1[0, 1]) ** 2) / g0
    return g0, c2 * hwc * hwc, w2

def _oracle_second_order_coefficient(xi: float, E_JT: float, hw: float, hwc: float, n_max: int, particle: bool) -> "np.ndarray":
    """Exact leading cavity correction of the g-factor of the dynamic model: [kappa, w2 hwc, g(0)] with
    g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) and E_0(g_c) = E_0(0) + w2 g_c^2 + O(g_c^4), from second-order
    perturbation theory in the cavity coupling (one-photon intermediate states suffice)."""
    if not (np.isfinite(hwc) and hwc > 0): raise ValueError("hwc must be positive")
    H0 = _oracle_model_hamiltonian(xi, E_JT, hw, 0.0, 0.0, hwc, n_max, 1, particle)
    Z, J, Lz, Sz, V = _model_operators(n_max, 1, particle, "ZV")
    g0, kappa, w2 = _second_order_kappa(H0, Z, V, hwc)
    return np.array([kappa, w2 * hwc, g0], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nhwc = 3200.0\nn_max = 12\nparticle = True\n',
         'call': 'second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'gold_call': '_oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nhwc = 1600.0\nn_max = 10\nparticle = False\n',
         'call': 'second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'gold_call': '_oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: weak Jahn-Teller coupling and a high cavity energy\nxi = 400.0\nE_JT = 150.0\nhw = 300.0\nhwc = 6400.0\nn_max = 8\nparticle = True\n',
         'call': 'second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'gold_call': '_oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 1500.0\nE_JT = 400.0\nhw = 250.0\nhwc = 2400.0\nn_max = 8\nparticle = False\n',
         'call': 'second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'gold_call': '_oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: a vanishing cavity energy must raise ValueError\nhwc = 0.0\nn_max = 6\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, particle))'},
    ]
