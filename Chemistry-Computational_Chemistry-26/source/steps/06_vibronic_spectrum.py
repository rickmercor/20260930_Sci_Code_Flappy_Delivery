"""
Returns the lowest n_levels levels of the molecular model without classical field and without cavity (n_ph = 0) as an (n_levels, 3) array of rows [E_k, |j_k|, <L_z S_z>_k] in ascending order of energy.

The linear E x e problem organizes its vibronic levels by a half-integer angular momentum; spin-orbit coupling then splits each spin-degenerate vibronic level into Kramers doublets with opposite alignment of orbital and spin moments, and the resulting ladder of doublets, not the bare electronic states, is what the fields and the cavity act on.

Returns
-------
A numpy float64 array of shape (n_levels, 3): rows [E_k in cm^-1, |j_k|, <L_z S_z>_k].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vibronic_spectrum(xi: float, E_JT: float, hw: float, n_max: int, particle: bool, n_levels: int) -> "np.ndarray":
    """Returns the lowest n_levels levels of the molecular model without classical field and without cavity (n_ph = 0) as an (n_levels, 3) array of rows [E_k, |j_k|, <L_z S_z>_k] in ascending order of energy.

    |j_k| = sqrt(<J^2>) is the magnitude of the conserved vibronic angular momentum J = l_vib - L_z/2 in the k-th
    eigenstate and <L_z S_z>_k the spin-orbit expectation value in it; degenerate levels appear as repeated rows.

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
        n_max: positive integer, the highest oscillator shell retained.
        particle: True for the single-particle scenario, False for the single-hole scenario.
        n_levels: positive integer, the number of lowest levels returned, at most the dimension 4 N.

    Returns:
        A numpy float64 array of shape (n_levels, 3): rows [E_k in cm^-1, |j_k|, <L_z S_z>_k].

    Raises:
        ValueError: if n_levels is not a positive integer or exceeds the dimension, or if a Hamiltonian argument is
        invalid.
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

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

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

def _oracle_vibronic_spectrum(xi: float, E_JT: float, hw: float, n_max: int, particle: bool, n_levels: int) -> "np.ndarray":
    """Lowest n_levels vibronic levels of the molecular model (no field, no cavity), shape (n_levels, 3):
    [E_k, |j_k|, <L_z S_z>_k] with |j_k| = sqrt(<J^2>) the magnitude of the conserved vibronic angular momentum
    J = l_vib - L_z/2 and <L_z S_z> the spin-orbit expectation value."""
    _check_int("n_levels", n_levels, 1)
    H = _oracle_model_hamiltonian(xi, E_JT, hw, 0.0, 0.0, 1.0, n_max, 0, particle)
    if n_levels > H.shape[0]: raise ValueError("n_levels exceeds the basis size")
    Z, J, Lz, Sz, V = _model_operators(n_max, 0, particle, "JLS")
    E, C = eigh(H, driver="evr", subset_by_index=[0, n_levels - 1])
    J2 = J @ J; LS = Lz * Sz
    out = []
    for k in range(n_levels):
        v = C[:, k]
        out.append([E[k], sqrt(max(float(np.real(np.vdot(v, J2 @ v))), 0.0)), float(np.real(np.vdot(v, _apply(LS, v))))])
    return np.array(out, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nn_max = 8\nparticle = True\nn_levels = 6\n',
         'call': 'vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'gold_call': '_oracle_vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 300.0\nE_JT = 150.0\nhw = 300.0\nn_max = 8\nparticle = False\nn_levels = 4\n',
         'call': 'vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'gold_call': '_oracle_vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 400.0\nhw = 400.0\nn_max = 6\nparticle = True\nn_levels = 8\n',
         'call': 'vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'gold_call': '_oracle_vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: the ground level alone\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\nn_max = 4\nparticle = True\nn_levels = 1\n',
         'call': 'vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'gold_call': '_oracle_vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: zero levels requested must raise ValueError\nn_max = 4\nparticle = True\nn_levels = 0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels))',
         'gold_call': '_catches_value_error(lambda: _oracle_vibronic_spectrum(xi, E_JT, hw, n_max, particle, n_levels))'},
    ]
