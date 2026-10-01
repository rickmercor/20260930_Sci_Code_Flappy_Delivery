"""
Returns [E_0, E_1 - E_0, p, q] for the spin-free linear E x e Jahn-Teller problem H = hw (n_vib + 1) + F (Q_+ |1><-1| + Q_- |-1><1|) on the orbital doublet times the oscillator basis.

Vibronic mixing quenches the electronic orbital moment and the off-diagonal electronic operators of the doublet by the Ham factors, the quantum counterpart of the static picture in which a frozen distortion simply mixes the two orbital states; for linear coupling the two factors are tied by q = (1 + p)/2.

Returns
-------
A numpy float64 array of shape (4,): [E_0 in cm^-1, E_1 - E_0 in cm^-1, p, q].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ham_reduction(E_JT: float, hw: float, n_max: int) -> "np.ndarray":
    """Returns [E_0, E_1 - E_0, p, q] for the spin-free linear E x e Jahn-Teller problem H = hw (n_vib + 1) + F (Q_+ |1><-1| + Q_- |-1><1|) on the orbital doublet times the oscillator basis.

    E_0 is the ground vibronic doublet energy, E_1 - E_0 the distance to the first excited vibronic level, and p and q
    are the Ham reduction factors, the magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x =
    |1><-1| + |-1><1| on the ground doublet; the ground level must be an isolated doublet.

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
        E_JT: non-negative float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.

    Returns:
        A numpy float64 array of shape (4,): [E_0 in cm^-1, E_1 - E_0 in cm^-1, p, q].

    Raises:
        ValueError: if hw is not positive, E_JT is negative, n_max is not a positive integer, or the ground level is
        not an isolated doublet.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def _oracle_ham_reduction(E_JT: float, hw: float, n_max: int) -> "np.ndarray":
    """Spin-free linear E x e Jahn-Teller problem: [E_0, E_1 - E_0, p, q] with E_0 the ground vibronic doublet
    (|j| = 1/2), E_1 the first excited vibronic level (|j| = 3/2), and the Ham reduction factors p and q,
    the magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on the
    ground doublet (q = (1 + p)/2 for the linear coupling)."""
    _check_pos("hw", hw)
    if not (np.isfinite(E_JT) and E_JT >= 0): raise ValueError("E_JT must be non-negative")
    _check_int("n_max", n_max, 1)
    ops = _oracle_vibronic_operators(n_max); N = ops.shape[1]
    I2 = np.eye(2); IN = np.eye(N)
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    F = sqrt(2.0 * E_JT * hw)
    H = np.kron(I2, hw * (ops[2] + IN)) + F * (np.kron(P1m, ops[0]) + np.kron(P1m.T, ops[1]))
    E, C = eigh(np.asarray(H, dtype=complex), driver="evr", subset_by_index=[0, 3])
    if E[1] - E[0] > 1e-8 or E[2] - E[1] < 1e-8: raise ValueError("the spin-free ground level is not an isolated vibronic doublet")
    P = C[:, :2]
    Lz = np.kron(np.diag([1.0, -1.0]), IN).astype(complex); sx = np.kron(P1m + P1m.T, IN).astype(complex)
    p = float(abs(np.linalg.eigvalsh(P.conj().T @ (Lz @ P))[1]))
    q = float(abs(np.linalg.eigvalsh(P.conj().T @ (sx @ P))[1]))
    return np.array([E[0], E[2] - E[0], p, q], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nE_JT = 800.0\nhw = 300.0\nn_max = 12\n',
         'call': 'ham_reduction(E_JT, hw, n_max)',
         'gold_call': '_oracle_ham_reduction(E_JT, hw, n_max)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nE_JT = 150.0\nhw = 300.0\nn_max = 10\n',
         'call': 'ham_reduction(E_JT, hw, n_max)',
         'gold_call': '_oracle_ham_reduction(E_JT, hw, n_max)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: weak coupling, the Ham factors close to their free values\nE_JT = 50.0\nhw = 400.0\nn_max = 8\n',
         'call': 'ham_reduction(E_JT, hw, n_max)',
         'gold_call': '_oracle_ham_reduction(E_JT, hw, n_max)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nE_JT = 400.0\nhw = 200.0\nn_max = 14\n',
         'call': 'ham_reduction(E_JT, hw, n_max)',
         'gold_call': '_oracle_ham_reduction(E_JT, hw, n_max)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# invalid input: a negative stabilization energy must raise ValueError\nE_JT = -100.0\nhw = 300.0\nn_max = 8\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: ham_reduction(E_JT, hw, n_max))',
         'gold_call': '_catches_value_error(lambda: _oracle_ham_reduction(E_JT, hw, n_max))'},
    ]
