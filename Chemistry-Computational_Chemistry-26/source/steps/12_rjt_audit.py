"""
Orchestrates the chain and returns the audit table: a head row [kappa_p at the first cavity energy, followed by zeros] and one row per cavity energy with the 25 columns listed below.

The head scalar is the leading cavity correction of the Kramers-doublet g-factor of the vibronically exact single-particle complex, the number that the frozen-distortion treatment estimates perturbatively; the table records, for each cavity energy, the exact coefficients of both scenarios and of the static model against the perturbative estimates, the finite-coupling shift, and the vibronic and static g-factors and moments that explain them.

Returns
-------
A numpy float64 array of shape (len(hwc_list) + 1, 25): the head row followed by one row per cavity energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rjt_audit(xi: float, E_JT: float, hw: float, g_c: float, hwc_list: list, n_max: int, n_ph: int) -> "np.ndarray":
    """Orchestrates the chain and returns the audit table: a head row [kappa_p at the first cavity energy, followed by zeros] and one row per cavity energy with the 25 columns listed below.

    The chain: build the oscillator basis, the ladder operators and the vibronic operators (checking that their sizes
    agree); assemble the full Hamiltonian at the first cavity energy with the coupling g_c for the single-particle
    scenario and record its dimension D and the residual max |[H, J]| of the conserved J = l_vib - L_z/2 (raising if
    it exceeds 1e-6); evaluate the static model for both scenarios at F rho = 2 E_JT; the zero-field vibronic Kramers
    doublets of both scenarios; the three lowest vibronic levels of the single-particle model; the spin-free Ham
    factors; and, for every cavity energy, the exact leading coefficients of both scenarios, the static coefficients
    and estimates of both scenarios, the finite-coupling shift with kappa_fd and the dressed <S_z> of the
    single-particle scenario with n_ph photon states, and w2 hwc of the single-particle scenario.

    Columns of every row after the head row: [hwc, kappa_p, kappa_h, kappa_static_p, kappa_static_h,
    kappa_source_weak_p, kappa_source_weak_h, kappa_source_strong_p, g_p(g_c) - g_p(0), kappa_fd_p, <S_z>_p(g_c), w2
    hwc (single-particle), g_p, g_h, g_static_p, g_static_h, g_formula_p, <L_z>_p, <L_z>_static_p, p_Ham, q_Ham, E_2 -
    E_0, |j_0|, D, max |[H, J]|], where p and h denote the single-particle and the single-hole scenario, g_formula is
    the closed form g_e - 2 xi/sqrt(xi^2 + 4 F_rho^2) of the static doublet, and E_2 - E_0 and |j_0| are the
    spin-orbit splitting of the lowest vibronic level and the |j| label of the ground doublet. The perturbative
    estimates of the same coefficient that this chain compares against, expressed in this notation, are
    kappa_source_weak = +-hwc/(F rho) (weak spin-orbit regime) and kappa_source_strong = +-8 hwc (F rho)^2/xi^3
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
        E_JT: positive float, the classical Jahn-Teller stabilization energy in cm^-1.
        hw: positive float, the vibrational energy in cm^-1.
        g_c: positive float, the cavity Zeeman coupling energy in cm^-1 for the finite-coupling shift.
        hwc_list: non-empty list of positive floats, the cavity energies in cm^-1 (the first one defines the head
        scalar).
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: positive integer, the highest photon number retained for the finite-coupling calculation.

    Returns:
        A numpy float64 array of shape (len(hwc_list) + 1, 25): the head row followed by one row per cavity energy.

    Raises:
        ValueError: if hwc_list is empty or contains a non-positive value, if xi, hw, E_JT or g_c is not positive, if
        n_max or n_ph is not a positive integer, or if any underlying step raises.
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

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

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

def _oracle_rjt_audit(xi: float, E_JT: float, hw: float, g_c: float, hwc_list: list, n_max: int, n_ph: int) -> "np.ndarray":
    """Orchestrate the chain. Row i >= 1 (per cavity frequency, 25 columns):
    [hwc, kappa_p, kappa_h, kappa_static_p, kappa_static_h, kappa_source_weak_p, kappa_source_weak_h,
     kappa_source_strong_p, dg_p(g_c), kappa_fd_p(g_c), <S_z>_p(g_c), w2 hwc (particle), g_p, g_h, g_static_p,
     g_static_h, g_formula_p, <L_z>_p, <L_z>_static_p, p_Ham, q_Ham, E_2 - E_0, |j_0|, D, max |[H, J]|];
    head row 0: [kappa_p for the first cavity frequency, followed by zeros]."""
    if not isinstance(hwc_list, (list, tuple, np.ndarray)) or len(hwc_list) == 0: raise ValueError("hwc_list must be non-empty")
    _check_pos("xi", xi); _check_pos("hw", hw); _check_pos("g_c", g_c)
    if not (np.isfinite(E_JT) and E_JT > 0): raise ValueError("E_JT must be positive")
    _check_int("n_max", n_max, 1); _check_int("n_ph", n_ph, 1)
    F_rho = 2.0 * E_JT
    basis = _oracle_oscillator_basis(n_max)
    lad = _oracle_ladder_operators(n_max)
    ops = _oracle_vibronic_operators(n_max)
    if basis.shape[0] != lad.shape[1] or ops.shape[1] != lad.shape[1]: raise ValueError("basis and operator sizes disagree")
    Hfull = _oracle_model_hamiltonian(xi, E_JT, hw, 0.0, g_c, float(hwc_list[0]), n_max, n_ph, True)
    Zf, Jf, Lzf, Szf, Vf = _model_operators(n_max, n_ph, True, "J")
    dim = float(Hfull.shape[0])
    jres = float(np.abs(Hfull @ Jf - Jf @ Hfull).max())
    if jres > 1e-6: raise ValueError("J = l_vib - L_z/2 is not conserved by the assembled Hamiltonian")
    stat_p = _oracle_static_model(xi, F_rho, True); stat_h = _oracle_static_model(xi, F_rho, False)
    gp = _oracle_kramers_g_factor(xi, E_JT, hw, 0.0, float(hwc_list[0]), n_max, 0, True)
    gh = _oracle_kramers_g_factor(xi, E_JT, hw, 0.0, float(hwc_list[0]), n_max, 0, False)
    spec = _oracle_vibronic_spectrum(xi, E_JT, hw, n_max, True, 3)
    ham = _oracle_ham_reduction(E_JT, hw, n_max)
    p_ham = ham[2]
    rows = []
    for hwc in hwc_list:
        hwc = float(hwc)
        if not (np.isfinite(hwc) and hwc > 0): raise ValueError("every cavity frequency must be positive")
        kp = _oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, True)
        kh = _oracle_second_order_coefficient(xi, E_JT, hw, hwc, n_max, False)
        sp = _oracle_static_cavity(xi, F_rho, hwc, True); sh = _oracle_static_cavity(xi, F_rho, hwc, False)
        sh_p = _oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, True)
        gc_p = _oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, True)
        rows.append([hwc, kp[0], kh[0], sp[0], sh[0], sp[1], sh[1], sp[2], sh_p[0], sh_p[1], gc_p[3], kp[1],
                     gp[1], gh[1], stat_p[1], stat_h[1], stat_p[2], gp[2], stat_p[3], p_ham, ham[3], spec[2, 0] - spec[0, 0], spec[0, 1], dim, jres])
    head = np.zeros(len(rows[0])); head[0] = rows[0][1]
    return np.array([head] + rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 100.0\nhwc_list = [3200.0, 1600.0, 6400.0]\nn_max = 12\nn_ph = 2\n',
         'call': 'rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph)',
         'gold_call': '_oracle_rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# boundary: a single cavity energy in a small basis\ng_c = 100.0\nhwc_list = [3200.0]\nn_max = 8\nn_ph = 2\n',
         'call': 'rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph)',
         'gold_call': '_oracle_rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 400.0\nE_JT = 150.0\nhw = 300.0\ng_c = 50.0\nhwc_list = [1600.0]\nn_max = 8\nn_ph = 2\n',
         'call': 'float(np.asarray(rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph))[0, 0])',
         'gold_call': 'float(np.asarray(_oracle_rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph))[0, 0])',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: an empty list of cavity energies must raise ValueError\ng_c = 100.0\nhwc_list = []\nn_max = 6\nn_ph = 2\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph))',
         'gold_call': '_catches_value_error(lambda: _oracle_rjt_audit(xi, E_JT, hw, g_c, hwc_list, n_max, n_ph))'},
    ]
