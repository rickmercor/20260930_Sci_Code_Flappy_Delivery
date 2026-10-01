"""
Returns [E_0, g_exact, g_formula, <L_z>] for the frozen-distortion (static) relativistic Jahn-Teller model on the four electronic-spin states at zero classical field.

With the distortion frozen at a fixed value, spin-orbit coupling and the Jahn-Teller mixing compete inside each spin sector to set how much orbital moment survives in the Kramers doublet; the closed-form g-factor of that four-state competition is the reference that the exact vibronic treatment is measured against.

Returns
-------
A numpy float64 array of shape (4,): [E_0 in cm^-1, g_exact, g_formula, <L_z>].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_model(xi: float, F_rho: float, particle: bool) -> "np.ndarray":
    """Returns [E_0, g_exact, g_formula, <L_z>] for the frozen-distortion (static) relativistic Jahn-Teller model on the four electronic-spin states at zero classical field.

    E_0 is the energy of the ground Kramers doublet of H_static at B_z = 0, g_exact its effective g-factor from the
    doublet-projected Zeeman operator s L_z + g_e S_z, g_formula the closed form g_e - s 2 xi/sqrt(xi^2 + 4 F_rho^2)
    of the same doublet, and <L_z> the orbital moment in the doublet state of larger Zeeman eigenvalue.

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
        particle: True for the single-particle scenario (s = +1), False for the single-hole scenario (s = -1).

    Returns:
        A numpy float64 array of shape (4,): [E_0 in cm^-1, g_exact, g_formula, <L_z>].

    Raises:
        ValueError: if xi or F_rho is not positive.
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

def _oracle_static_model(xi: float, F_rho: float, particle: bool) -> "np.ndarray":
    """The source's static relativistic Jahn-Teller model (four states, frozen distortion):
    [E_0, g_exact, g_formula, <L_z>] with g_exact from the exact Kramers-doublet Zeeman splitting and
    g_formula = g_e -+ 2 xi / sqrt(xi^2 + 4 F_rho^2)."""
    if not (np.isfinite(xi) and xi > 0 and np.isfinite(F_rho) and F_rho > 0): raise ValueError("xi and F_rho must be positive")
    s = 1.0 if particle else -1.0
    Lz = np.diag([1.0, -1.0]); Sz = np.diag([0.5, -0.5]); I2 = np.eye(2)
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    H0 = xi * np.kron(Lz, Sz) + F_rho * np.kron(P1m + P1m.T, I2)
    Z = s * np.kron(Lz, I2) + _GE() * np.kron(I2, Sz)
    E0, g, vals, _ = _kramers_g(H0.astype(complex), Z.astype(complex), [np.kron(Lz, I2).astype(complex)])
    gf = _GE() - s * 2.0 * xi / sqrt(xi * xi + 4.0 * F_rho * F_rho)
    return np.array([E0, g, gf, vals[0]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nF_rho = 1600.0\nparticle = True\n',
         'call': 'static_model(xi, F_rho, particle)',
         'gold_call': '_oracle_static_model(xi, F_rho, particle)'},
        {'setup': 'import numpy as np\nxi = 800.0\nF_rho = 1600.0\nparticle = False\n',
         'call': 'static_model(xi, F_rho, particle)',
         'gold_call': '_oracle_static_model(xi, F_rho, particle)'},
        {'setup': 'import numpy as np\n# boundary: weak spin-orbit coupling against a strong distortion\nxi = 200.0\nF_rho = 3000.0\nparticle = True\n',
         'call': 'static_model(xi, F_rho, particle)',
         'gold_call': '_oracle_static_model(xi, F_rho, particle)'},
        {'setup': 'import numpy as np\n# boundary: strong spin-orbit coupling against a weak distortion\nxi = 2000.0\nF_rho = 300.0\nparticle = False\n',
         'call': 'static_model(xi, F_rho, particle)',
         'gold_call': '_oracle_static_model(xi, F_rho, particle)'},
        {'setup': 'import numpy as np\n# invalid input: a vanishing distortion product is outside the model and must raise ValueError\nxi = 800.0\nF_rho = 0.0\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: static_model(xi, F_rho, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_static_model(xi, F_rho, particle))'},
    ]
