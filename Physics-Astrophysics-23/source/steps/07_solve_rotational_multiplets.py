"""
Implement solve_rotational_multiplets, which computes the perturbed
frequencies of a set of coupled quadrupolar mixed modes for one azimuthal
order by solving the rotating eigenvalue problem exactly in the basis of the
unperturbed modes.

With R the rotational coupling matrix (assemble_rotation_matrix) and nu_0,i
the unperturbed frequencies, the perturbed frequencies are the positive
eigenvalues nu of the quadratic eigenvalue problem nu^2 a = nu R a +
diag(nu_0,i^2) a, solved without any perturbative expansion in the rotation
rate. Each positive eigenfrequency is attributed to the unperturbed mode
that carries the largest weight in its eigenvector a, so that the multiplet
component of mode i is the eigenfrequency whose eigenvector is dominated by
|a_i|.

Returns
-------
np.ndarray of shape (N,), perturbed frequencies in microhertz, entry i belonging to unperturbed mode i
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_rotational_multiplets(nu_modes: "np.ndarray", rot_matrix: "np.ndarray") -> "np.ndarray":
    '''Perturbed frequencies of N coupled mixed modes for one azimuthal order.

    Parameters
    ----------
    nu_modes : np.ndarray
        One-dimensional array of N >= 1 distinct unperturbed mixed-mode
        frequencies, in microhertz, strictly positive.
    rot_matrix : np.ndarray
        Symmetric real array of shape (N, N), in microhertz, the rotational
        coupling matrix for the azimuthal order considered.

    Returns
    -------
    nu_perturbed : np.ndarray
        Float array of shape (N,), in microhertz: entry i is the positive
        eigenfrequency whose eigenvector is dominated by unperturbed mode i.

    Raises
    ------
    ValueError
        If nu_modes is not a one-dimensional array of distinct strictly
        positive values, if rot_matrix is not a real symmetric (N, N) array,
        or if the dominant-component rule does not attribute exactly one
        positive eigenfrequency to every unperturbed mode.
    '''
    return nu_perturbed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_rotational_multiplets(nu_modes: "np.ndarray", rot_matrix: "np.ndarray") -> "np.ndarray":
    """Reference implementation: companion linearization of the quadratic eigenvalue problem."""
    nu_modes = np.atleast_1d(np.asarray(nu_modes, dtype=float))
    rot_matrix = np.asarray(rot_matrix, dtype=float)
    n_modes = nu_modes.size
    if nu_modes.ndim != 1 or n_modes < 1 or np.any(nu_modes <= 0.0):
        raise ValueError("nu_modes must be a one-dimensional array of strictly positive frequencies")
    if np.unique(nu_modes).size != n_modes:
        raise ValueError("nu_modes must be distinct")
    if rot_matrix.shape != (n_modes, n_modes) or not np.allclose(rot_matrix, rot_matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("rot_matrix must be a real symmetric (N, N) array")
    # Linearization: with b = nu a, [R  D; I  0] [b; a] = nu [b; a], D = diag(nu_0^2).
    companion = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
    companion[:n_modes, :n_modes] = rot_matrix
    companion[:n_modes, n_modes:] = np.diag(nu_modes ** 2)
    companion[n_modes:, :n_modes] = np.eye(n_modes)
    eigenvalues, eigenvectors = np.linalg.eig(companion)
    eigenvalues = eigenvalues.real
    positive = np.where(eigenvalues > 0.0)[0]
    if positive.size != n_modes:
        raise ValueError("the quadratic eigenvalue problem must have exactly N positive eigenvalues")
    weights = np.abs(eigenvectors[n_modes:, positive])
    weights /= np.linalg.norm(weights, axis=0, keepdims=True)
    dominant = np.argmax(weights, axis=0)  # unperturbed mode dominating each positive eigenvector
    if np.unique(dominant).size != n_modes:
        raise ValueError("the dominant-component rule does not label the eigenfrequencies one to one")
    nu_perturbed = np.zeros(n_modes, dtype=float)
    nu_perturbed[dominant] = eigenvalues[positive]
    return nu_perturbed

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
        solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Normal: the task's twelve modes with the m = +2 coupling matrix ---
        {
            "setup": task + "rot_matrix = _oracle_assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, 2)\n",
            "call": "solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
            "gold_call": "_oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
        },
        # --- Normal: the same modes with m = -2 (must equal the negated negative branch of m = +2) ---
        {
            "setup": task + "rot_matrix = _oracle_assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, -2)\n",
            "call": "solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
            "gold_call": "_oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
        },
        # --- Boundary: zero coupling matrix returns the unperturbed frequencies ---
        {
            "setup": task + "rot_matrix = np.zeros((12, 12))\n",
            "call": "solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
            "gold_call": "_oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
        },
        # --- Edge: fast rotation (core 4000 nHz, envelope 333.3 nHz) for which the perturbed
        #     components of neighbouring modes cross in frequency, so the labelling must follow the
        #     eigenvectors and not the frequency order ---
        {
            "setup": task + "rot_matrix = _oracle_assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, 4000.0, 333.3, 2)\n",
            "call": "solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
            "gold_call": "_oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
        },
        # --- Edge: two strongly coupled modes with a hand-built symmetric matrix (exact 2 x 2 problem) ---
        {
            "setup": "import numpy as np\nnu_modes = np.array([366.805, 368.675])\nrot_matrix = np.array([[0.0012, 0.0009], [0.0009, 0.0007]])\n",
            "call": "solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
            "gold_call": "_oracle_solve_rotational_multiplets(nu_modes.copy(), rot_matrix.copy())",
        },
        # --- Invalid: three of the task's modes with a hand-built symmetric coupling matrix whose
        #     entries are comparable to the mode spacing, as for very fast rotation, so that the
        #     dominant-component rule gives two positive eigenfrequencies to the first mode and none
        #     to the second ---
        {
            "setup": "import numpy as np\nnu_modes = np.array([366.805, 368.675, 375.878])\nrot_matrix = np.array([[20.0, -5.0, -4.0], [-5.0, 17.0, -1.0], [-4.0, -1.0, 3.0]])\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-symmetric coupling matrix ---
        {
            "setup": "import numpy as np\nnu_modes = np.array([366.805, 368.675])\nrot_matrix = np.array([[0.0012, 0.0009], [-0.0009, 0.0007]])\n" + invalid_setup,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
