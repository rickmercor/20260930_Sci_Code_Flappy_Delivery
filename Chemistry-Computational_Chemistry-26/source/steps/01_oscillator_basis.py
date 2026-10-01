"""
Returns the ordered basis of the two-dimensional oscillator as an (N, 2) integer array of pairs (n_x, n_y) with n_x + n_y <= n_max, ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending, N = (n_max + 1)(n_max + 2)/2.

The doubly degenerate vibration of a Jahn-Teller centre is a two-dimensional isotropic oscillator; a fixed ordering of its product states is what makes every matrix of the chain reproducible and lets the conserved vibronic angular momentum be read off the eigenvectors.

Returns
-------
A numpy int64 array of shape (N, 2) whose row k holds (n_x, n_y) of the k-th basis state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def oscillator_basis(n_max: int) -> "np.ndarray":
    """Returns the ordered basis of the two-dimensional oscillator as an (N, 2) integer array of pairs (n_x, n_y) with n_x + n_y <= n_max, ordered by the shell n = n_x + n_y ascending and, within a shell, by n_x ascending, N = (n_max + 1)(n_max + 2)/2.

    This step fixes the ordering that every vibrational matrix of the chain uses.

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
        n_max: non-negative integer, the highest oscillator shell retained.

    Returns:
        A numpy int64 array of shape (N, 2) whose row k holds (n_x, n_y) of the k-th basis state.

    Raises:
        ValueError: if n_max is not a non-negative integer.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _osc_states(n_max: int):
    """(n_x, n_y) pairs with n_x + n_y <= n_max: shells n = 0..n_max, n_x ascending within a shell."""
    return [(nx, n - nx) for n in range(n_max + 1) for nx in range(n + 1)]

def _oracle_oscillator_basis(n_max: int) -> "np.ndarray":
    """Ordered two-dimensional oscillator basis: (N, 2) integer array of (n_x, n_y), N = (n_max+1)(n_max+2)/2."""
    if int(n_max) != n_max or n_max < 0: raise ValueError("n_max must be a non-negative integer")
    return np.array(_osc_states(n_max), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn_max = 3\n',
         'call': 'oscillator_basis(n_max)',
         'gold_call': '_oracle_oscillator_basis(n_max)'},
        {'setup': 'import numpy as np\n# boundary: the single shell n = 0\nn_max = 0\n',
         'call': 'oscillator_basis(n_max)',
         'gold_call': '_oracle_oscillator_basis(n_max)'},
        {'setup': 'import numpy as np\nn_max = 7\n',
         'call': 'oscillator_basis(n_max)',
         'gold_call': '_oracle_oscillator_basis(n_max)'},
        {'setup': 'import numpy as np\n# invalid input: a negative shell count has no basis and must raise ValueError\nn_max = -1\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: oscillator_basis(n_max))',
         'gold_call': '_catches_value_error(lambda: _oracle_oscillator_basis(n_max))'},
    ]
