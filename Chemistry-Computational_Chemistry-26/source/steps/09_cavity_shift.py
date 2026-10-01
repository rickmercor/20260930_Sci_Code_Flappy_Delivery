"""
Returns [g(g_c) - g(0), kappa_fd, g(0)]: the change of the exact Kramers-doublet g-factor produced by the cavity at the finite coupling g_c with n_ph photon states, the finite-coupling estimate kappa_fd = (g(g_c) - g(0)) (hwc/g_c)^2 of the leading coefficient, and the cavity-free g-factor.

At a finite coupling the cavity shift of the g-factor contains every order in the coupling; dividing it by (g_c/hwc)^2 gives an estimate of the leading coefficient that is contaminated by the fourth-order term, which is why the exact leading coefficient has to come from perturbation theory rather than from a single finite-coupling calculation.

Returns
-------
A numpy float64 array of shape (3,): [g(g_c) - g(0), kappa_fd, g(0)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_shift(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Returns [g(g_c) - g(0), kappa_fd, g(0)]: the change of the exact Kramers-doublet g-factor produced by the cavity at the finite coupling g_c with n_ph photon states, the finite-coupling estimate kappa_fd = (g(g_c) - g(0)) (hwc/g_c)^2 of the leading coefficient, and the cavity-free g-factor.

    g(0) is evaluated without photon states (n_ph = 0) and g(g_c) with the n_ph photon states requested.

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
        g_c: positive float, the cavity Zeeman coupling energy in cm^-1.
        hwc: positive float, the cavity energy in cm^-1.
        n_max: positive integer, the highest oscillator shell retained.
        n_ph: positive integer, the highest photon number retained for the coupled calculation.
        particle: True for the single-particle scenario, False for the single-hole scenario.

    Returns:
        A numpy float64 array of shape (3,): [g(g_c) - g(0), kappa_fd, g(0)].

    Raises:
        ValueError: if g_c is not positive, if n_ph is not a positive integer, or if a g-factor argument is invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _oracle_cavity_shift(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Cavity-induced change of the exact g-factor at a finite coupling: [g(g_c) - g(0), kappa_fd, g(0)] with
    kappa_fd = (g(g_c) - g(0)) (hwc / g_c)^2 the finite-coupling estimate of the second-order coefficient."""
    if not (np.isfinite(g_c) and g_c > 0): raise ValueError("g_c must be positive")
    if int(n_ph) != n_ph or n_ph < 1: raise ValueError("n_ph must be a positive integer")
    g1 = _oracle_kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)[1]
    g0 = _oracle_kramers_g_factor(xi, E_JT, hw, 0.0, hwc, n_max, 0, particle)[1]
    return np.array([g1 - g0, (g1 - g0) * (hwc / g_c) ** 2, g0], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 100.0\nhwc = 3200.0\nn_max = 8\nn_ph = 2\nparticle = True\n',
         'call': 'cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\ng_c = 50.0\nhwc = 1600.0\nn_max = 8\nn_ph = 3\nparticle = False\n',
         'call': 'cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 400.0\nE_JT = 150.0\nhw = 300.0\ng_c = 25.0\nhwc = 6400.0\nn_max = 6\nn_ph = 2\nparticle = True\n',
         'call': 'cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# boundary: a single photon state, the smallest coupled calculation\ng_c = 100.0\nhwc = 3200.0\nn_max = 6\nn_ph = 1\nparticle = True\n',
         'call': 'cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'gold_call': '_oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nxi = 800.0\nE_JT = 800.0\nhw = 300.0\n# invalid input: a vanishing coupling has no finite-coupling estimate and must raise ValueError\ng_c = 0.0\nhwc = 3200.0\nn_max = 6\nn_ph = 2\nparticle = True\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle))',
         'gold_call': '_catches_value_error(lambda: _oracle_cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle))'},
    ]
