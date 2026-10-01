"""
Assembles the full Hamiltonian H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ of the cavity-modified dynamic relativistic E x e Jahn-Teller model on the product space in the stated index order as a (D, D) complex Hermitian matrix in cm^-1 with D = 4 N (n_ph + 1).

Quantizing the distortion turns the four-state static problem into a vibronic one in which the orbital moment is shared between the electronic doublet and the vibration, and the cavity Zeeman term couples the spin to the photon number; only the full matrix, with every term in the same basis, lets the exact Kramers doublet and its response to fields be computed.

Returns
-------
A numpy complex128 Hermitian array of shape (D, D) in cm^-1, D = 4 N (n_ph + 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def model_hamiltonian(xi: float, E_JT: float, hw: float, Bz: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Assembles the full Hamiltonian H = H_soc + H_Zee + H_vib + H_JT + H_c + H_cZ of the cavity-modified dynamic relativistic E x e Jahn-Teller model on the product space in the stated index order as a (D, D) complex Hermitian matrix in cm^-1 with D = 4 N (n_ph + 1).

    The vibrational operators are the exact ones of the retained basis (products formed in the extended basis and cut
    back).

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
        Bz: finite float, the classical field in tesla.
        g_c: non-negative float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: non-negative integer, the highest photon number retained.
        particle: True for the single-particle scenario (s = +1), False for the single-hole scenario (s = -1).

    Returns:
        A numpy complex128 Hermitian array of shape (D, D) in cm^-1, D = 4 N (n_ph + 1).

    Raises:
        ValueError: if xi, hw or hwc is not positive, if E_JT or g_c is negative, if Bz is not finite, if n_max is not
        a positive integer or n_ph not a non-negative integer, or if the assembled matrix is not Hermitian.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _MUB():
    """Bohr magneton in cm^-1 per tesla."""
    return 0.4668644814

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def _oracle_model_hamiltonian(xi: float, E_JT: float, hw: float, Bz: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Full Hamiltonian of the cavity-modified dynamic relativistic E x e Jahn-Teller model, shape (D, D),
    D = 4 N (n_ph + 1), ordering (m_l = +1, -1) x (m_s = up, down) x oscillator basis x photon number 0..n_ph."""
    _check_pos("xi", xi); _check_pos("hw", hw); _check_pos("hwc", hwc)
    if not (np.isfinite(E_JT) and E_JT >= 0): raise ValueError("E_JT must be non-negative")
    if not (np.isfinite(g_c) and g_c >= 0): raise ValueError("g_c must be non-negative")
    if not np.isfinite(Bz): raise ValueError("Bz must be finite")
    _check_int("n_max", n_max, 1); _check_int("n_ph", n_ph, 0)
    ops = _oracle_vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    I2 = np.eye(2); IN = np.eye(N); IP = np.eye(n_ph + 1)
    Lz2 = np.diag([1.0, -1.0]); Sz2 = np.diag([0.5, -0.5])
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    F = sqrt(2.0 * E_JT * hw)
    Hes = xi * np.kron(Lz2, Sz2) + _MUB() * Bz * (s * np.kron(Lz2, I2) + _GE() * np.kron(I2, Sz2))
    Hvib = hw * (ops[2] + IN)
    HJT = F * (np.kron(np.kron(P1m, I2), ops[0]) + np.kron(np.kron(P1m.T, I2), ops[1]))
    bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
    Hc = hwc * (bph.T @ bph)
    H = (np.kron(np.kron(Hes, IN), IP) + np.kron(np.kron(np.kron(I2, I2), Hvib), IP) + np.kron(HJT, IP)
         + np.kron(np.kron(np.kron(I2, I2), IN), Hc))
    if g_c > 0:
        sy = np.array([[0.0, -1j], [1j, 0.0]])
        H = H + 1j * g_c * np.kron(np.kron(np.kron(I2, sy), IN), bph.T - bph)
    H = np.asarray(H, dtype=complex)
    if np.abs(H - H.conj().T).max() > 1e-10: raise ValueError("Hamiltonian not Hermitian")
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nBz = 0.0\ng_c = 0.0\nhwc = 3200.0\nn_max = 2\nn_ph = 0\nparticle = True\n',
         'call': 'model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)'},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nBz = 0.5\ng_c = 100.0\nhwc = 3200.0\nn_max = 2\nn_ph = 1\nparticle = True\n',
         'call': 'model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)'},
        {'setup': 'import numpy as np\nxi = 500.0\nE_JT = 150.0\nhw = 250.0\nBz = -1.0\ng_c = 30.0\nhwc = 1000.0\nn_max = 3\nn_ph = 2\nparticle = False\n',
         'call': 'model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)'},
        {'setup': 'import numpy as np\n# boundary: no Jahn-Teller coupling and the smallest basis\nxi = 800.0\nE_JT = 0.0\nhw = 300.0\nBz = 0.0\ng_c = 0.0\nhwc = 3200.0\nn_max = 1\nn_ph = 0\nparticle = True\n',
         'call': 'model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle)'},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: a vanishing vibrational energy must raise ValueError\nhw = 0.0\nBz = 0.0\ng_c = 0.0\nhwc = 3200.0\nn_max = 2\nn_ph = 0\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_model_hamiltonian(xi, E_JT, hw, Bz, g_c, hwc, n_max, n_ph, particle))'},
    ]
