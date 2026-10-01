"""
Implement assemble_rotation_matrix, which builds the rotational coupling
matrix of a set of quadrupolar mixed modes for one azimuthal order m in a
two-zone rotation model (core rotating at the cyclic rate nu_core, envelope
at nu_env).

The perturbed eigenfrequencies nu of the rotating star, projected on the
basis of N unperturbed mixed modes of frequencies nu_0,i, are the solutions
of the quadratic eigenvalue problem

    nu^2 a = nu R a + diag(nu_0,i^2) a,

where R is the N x N matrix returned here, expressed in cyclic-frequency
units (microhertz) so that every term of the equation is in microhertz
squared; the rotational operator is taken at leading order in the rotation
rate, so R is proportional to m. Its diagonal element R_ii equals 2 m times
the standard first-order rotational splitting per unit m of mixed mode i in
the two-zone model, in which the core rotation rate is weighted by the
trapping fraction zeta_i and multiplied by (1 - 1 / L^2), the asymptotic
Ledoux factor of high-order g modes (L^2 = l (l + 1) = 6), and the envelope
rotation rate is weighted by 1 - zeta_i. Its off-diagonal element R_ij
equals 2 m times the near-degeneracy coupling of modes i and j, which is the
core rotation rate times the core-cavity coefficient gamma_c(i, j)
(compute_core_coupling) plus the envelope rotation rate times the
envelope-cavity coefficient gamma_e(i, j); gamma_e is the envelope-cavity
integral of the same rotational kernel (in the p-mode cavity the kernel
reduces to rho r^2 xi_r,i xi_r,j) and is fixed by gamma_c through the
orthogonality of the two mixed modes. The trapping fractions are the
asymptotic zeta values of the modes (compute_trapping_fraction).

Returns
-------
np.ndarray of shape (N, N), symmetric, in microhertz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_rotation_matrix(nu_modes: "np.ndarray", q: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, m: int) -> "np.ndarray":
    '''Rotational coupling matrix R (in microhertz) of N unperturbed l = 2 mixed modes.

    Parameters
    ----------
    nu_modes : np.ndarray
        One-dimensional array of N >= 1 distinct unperturbed mixed-mode
        frequencies, in microhertz, strictly positive.
    q : float
        Coupling factor in (0, 1).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.
    nu_core : float
        Core rotation rate Omega_core / (2 pi), in nanohertz.
    nu_env : float
        Envelope rotation rate Omega_env / (2 pi), in nanohertz.
    m : int
        Azimuthal order, an integer with |m| <= 2.

    Returns
    -------
    rot_matrix : np.ndarray
        Symmetric float array of shape (N, N), in microhertz, with the
        first-order splitting terms on the diagonal and the near-degeneracy
        coupling terms off the diagonal; it is identically zero for m = 0
        and changes sign with m.

    Raises
    ------
    ValueError
        If nu_modes is not a one-dimensional array of distinct strictly
        positive values, if m is not an integer with |m| <= 2, or if q,
        delta_nu or delta_pi is outside its admissible range.
    '''
    return rot_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_rotation_matrix(nu_modes: "np.ndarray", q: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, m: int) -> "np.ndarray":
    """Reference implementation of the two-zone rotational coupling matrix."""
    nu_modes = np.atleast_1d(np.asarray(nu_modes, dtype=float))
    if nu_modes.ndim != 1 or nu_modes.size < 1 or np.any(nu_modes <= 0.0):
        raise ValueError("nu_modes must be a one-dimensional array of strictly positive frequencies")
    if np.unique(nu_modes).size != nu_modes.size:
        raise ValueError("nu_modes must be distinct")
    if isinstance(m, bool) or int(m) != m or abs(int(m)) > 2:
        raise ValueError("m must be an integer with |m| <= 2")
    m = int(m)
    nu_core_uhz, nu_env_uhz = float(nu_core) * 1e-3, float(nu_env) * 1e-3
    big_l2 = _degree_norm() ** 2
    n_modes = nu_modes.size
    zeta = np.array([_oracle_compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi) for nu in nu_modes])
    rot_matrix = np.zeros((n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        rot_matrix[i, i] = 2.0 * m * (zeta[i] * (1.0 - 1.0 / big_l2) * nu_core_uhz + (1.0 - zeta[i]) * nu_env_uhz)
        for j in range(i + 1, n_modes):
            gamma_c = _oracle_compute_core_coupling(nu_modes[i], nu_modes[j], zeta[i], zeta[j], delta_pi)
            gamma_e = big_l2 / (1.0 - big_l2) * gamma_c
            rot_matrix[i, j] = rot_matrix[j, i] = 2.0 * m * (gamma_c * nu_core_uhz + gamma_e * nu_env_uhz)
    return rot_matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = """import numpy as np
q = 0.0360054005
delta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850
nu_core, nu_env = 745.0, 61.0
nu_modes = np.array([337.1091, 341.4943, 344.5364, 351.7109, 359.3149, 366.805,
                     368.675, 375.878, 384.557, 393.1859, 395.2084, 403.547])
"""
    invalid_setup = """
def run_model():
    try:
        assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's twelve modes, m = +2 ---
        {
            "setup": task + "m = 2\n",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Normal: the two observed modes only, m = -1, envelope rotating faster than the core ---
        {
            "setup": """import numpy as np
q = 0.0360054005
delta_nu, nu_p, delta_pi = 26.50, 368.100, 60.850
nu_core, nu_env = 120.0, 300.0
nu_modes = np.array([366.805, 368.675])
m = -1
""",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Boundary: m = 0 gives the zero matrix ---
        {
            "setup": task + "m = 0\n",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Edge: a single mode (1 x 1 matrix holding only the first-order term) ---
        {
            "setup": task + "nu_modes = np.array([368.675])\nm = 1\n",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Normal: fast rotation (core 4000 nHz, envelope 333.3 nHz) on the task's modes, m = +2 ---
        {
            "setup": task + "nu_core, nu_env = 4000.0, 333.3\nm = 2\n",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Normal: three neighbouring modes of a strongly coupled low-frequency star, m = +1 ---
        {
            "setup": """import numpy as np
q = 0.35
delta_nu, nu_p, delta_pi = 29.00, 381.101, 64.422
nu_core, nu_env = 900.0, 120.0
nu_modes = np.array([150.11226753, 151.42329751, 152.79691138])
m = 1
""",
            "call": "assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
            "gold_call": "_oracle_assemble_rotation_matrix(nu_modes.copy(), q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m)",
        },
        # --- Invalid: azimuthal order beyond the degree ---
        {
            "setup": task + "m = 3\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
