"""
Returns [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]: the exact second-order cavity coefficient of the frozen-distortion model (its four electronic-spin states times the photon numbers 0 and 1, H = H_static + hwc b^+ b + H_cZ) together with the two perturbative estimates of the same coefficient and the cavity-free static g-factor.

Applying the exact second-order theory to the frozen-distortion model itself separates two questions, whether its perturbative estimate of the cavity correction is right for its own model and how much the quantized vibration changes the answer; the static coefficient is what the molecular g-factors are compared against.

Returns
-------
A numpy float64 array of shape (4,): [kappa_static, kappa_source_weak, kappa_source_strong, g(0)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_cavity(xi: float, F_rho: float, hwc: float, particle: bool) -> "np.ndarray":
    """Returns [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]: the exact second-order cavity coefficient of the frozen-distortion model (its four electronic-spin states times the photon numbers 0 and 1, H = H_static + hwc b^+ b + H_cZ) together with the two perturbative estimates of the same coefficient and the cavity-free static g-factor.

    The perturbative estimates of the same coefficient that this chain compares against, expressed in this notation,
    are kappa_source_weak = +-hwc/(F rho) (weak spin-orbit regime) and kappa_source_strong = +-8 hwc (F rho)^2/xi^3
    (strong spin-orbit regime), upper signs for the single-particle and lower signs for the single-hole scenario.

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
        F_rho: positive float, the frozen distortion product F rho in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (4,): [kappa_static, kappa_source_weak, kappa_source_strong, g(0)].

    Raises:
        ValueError: if xi, F_rho or hwc is not positive, or if the lowest level is not an isolated Kramers doublet.
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

def _oracle_static_cavity(xi: float, F_rho: float, hwc: float, particle: bool) -> "np.ndarray":
    """The same second-order cavity coefficient for the source's static model (four states times the photon
    number 0, 1) against the source's own estimates: [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]
    with kappa_source_weak = +-hwc/F_rho and kappa_source_strong = +-8 hwc F_rho^2/xi^3 (upper sign single-particle)."""
    if not (np.isfinite(xi) and xi > 0 and np.isfinite(F_rho) and F_rho > 0 and np.isfinite(hwc) and hwc > 0): raise ValueError("xi, F_rho and hwc must be positive")
    s = 1.0 if particle else -1.0
    Lz2 = np.diag([1.0, -1.0]); Sz2 = np.diag([0.5, -0.5]); I2 = np.eye(2); sy = np.array([[0.0, -1j], [1j, 0.0]])
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    Hes = xi * np.kron(Lz2, Sz2) + F_rho * np.kron(P1m + P1m.T, I2)
    bph = np.diag([1.0], 1); IP = np.eye(2)
    H0 = np.kron(Hes, IP) + np.kron(np.eye(4), hwc * (bph.T @ bph))
    Z = np.kron(s * np.kron(Lz2, I2) + _GE() * np.kron(I2, Sz2), IP)
    V = 1j * np.kron(np.kron(I2, sy), bph.T - bph)
    g0, kappa, w2 = _second_order_kappa(H0.astype(complex), Z.astype(complex), V, hwc)
    return np.array([kappa, s * hwc / F_rho, s * 8.0 * hwc * F_rho * F_rho / xi ** 3, g0], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nF_rho = 1600.0\nhwc = 3200.0\nparticle = True\n',
         'call': 'static_cavity(xi, F_rho, hwc, particle)',
         'gold_call': '_oracle_static_cavity(xi, F_rho, hwc, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nF_rho = 1600.0\nhwc = 3200.0\nparticle = False\n',
         'call': 'static_cavity(xi, F_rho, hwc, particle)',
         'gold_call': '_oracle_static_cavity(xi, F_rho, hwc, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: weak spin-orbit coupling, strong distortion, high cavity energy\nxi = 200.0\nF_rho = 3000.0\nhwc = 12800.0\nparticle = True\n',
         'call': 'static_cavity(xi, F_rho, hwc, particle)',
         'gold_call': '_oracle_static_cavity(xi, F_rho, hwc, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 2000.0\nF_rho = 300.0\nhwc = 6400.0\nparticle = False\n',
         'call': 'static_cavity(xi, F_rho, hwc, particle)',
         'gold_call': '_oracle_static_cavity(xi, F_rho, hwc, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# invalid input: a vanishing spin-orbit coupling is outside the model and must raise ValueError\nxi = 0.0\nF_rho = 1600.0\nhwc = 3200.0\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: static_cavity(xi, F_rho, hwc, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_static_cavity(xi, F_rho, hwc, particle))'},
    ]
