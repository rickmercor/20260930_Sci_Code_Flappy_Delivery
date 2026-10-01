"""
The cell problem is posed on periodic potentials with zero mean, a space of dimension N - 1 for N sample points, and the spectral analysis needs matrix representations of two operators on that space: the cell operator A v = -div(C grad v) and the reference operator A_0 v = -C_0 Laplacian v of a uniform comparison medium. Both matrices are formed in the same real orthonormal basis of mean-free fields, orthonormal for the Euclidean inner product over the sample points, and their entries are the pixel averages of q_i times the operator image of q_j, so that a coefficient vector c represents the field sum_i c_i q_i and c_1 . [A] c_2 equals the average of grad v_1 . C grad v_2. The basis used here is trigonometric: for every frequency pair in a half-space of the symmetric frequency set, the normalised cosine and sine modes. In that basis the reference matrix is diagonal with entries C_0 (2 pi)^2 |k / L|^2 / N, positive because the zero frequency has been excluded, so it defines an inner product on coefficient vectors; the cell matrix is real symmetric and reduces to the reference matrix when the conductivity is uniform and equal to C_0. Any other orthonormal mean-free basis represents the same operators up to an orthogonal change of coordinates, which leaves every generalised eigenvalue, every trace and every reconstructed field unchanged.

Returns
-------
dict with float array basis of shape (N - 1, n, n) whose slices are the orthonormal mean-free basis fields, float array operator of shape (N - 1, N - 1) representing the cell operator, and float array reference_operator of the same shape representing the reference operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_mean_free_operator_matrices(
    conductivity: np.ndarray,
    cell_lengths: tuple,
    reference_conductivity: float,
) -> dict:
    """Represent the cell operator and the reference operator on mean-free periodic fields.

    Raises
    ------
    ValueError
        If conductivity is not a square array of odd size at least three with finite non-negative entries, if a cell length is not strictly positive, or if reference_conductivity is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mean_free_trigonometric_basis(n_pixels, cell_lengths):
    """Orthonormal cosine and sine modes over a half-space of the odd frequency set."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels).astype(int)
    axes = [np.arange(n_pixels) * float(cell_lengths[d]) / n_pixels for d in range(2)]
    x_1, x_2 = np.meshgrid(axes[0], axes[1], indexing="ij")
    scale = np.sqrt(2.0 / (n_pixels * n_pixels))
    modes = []
    for k_1 in integer_frequencies:
        for k_2 in integer_frequencies:
            if k_1 > 0 or (k_1 == 0 and k_2 > 0):
                angle = 2.0 * np.pi * (k_1 * x_1 / float(cell_lengths[0]) + k_2 * x_2 / float(cell_lengths[1]))
                modes.append(scale * np.cos(angle))
                modes.append(scale * np.sin(angle))
    return np.stack(modes, axis=0)


def _oracle_assemble_mean_free_operator_matrices(
    conductivity: np.ndarray,
    cell_lengths: tuple,
    reference_conductivity: float,
) -> dict:
    """Reference implementation."""
    conductivity = np.asarray(conductivity, dtype=float)
    if conductivity.ndim != 2 or conductivity.shape[0] != conductivity.shape[1] or conductivity.shape[0] < 3 or conductivity.shape[0] % 2 == 0:
        raise ValueError("conductivity must be a square array of odd size at least three")
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity < 0.0):
        raise ValueError("conductivity must be finite and non-negative")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    reference_conductivity = float(reference_conductivity)
    if reference_conductivity <= 0.0:
        raise ValueError("reference_conductivity must be strictly positive")
    n_pixels = conductivity.shape[0]
    n_points = n_pixels * n_pixels
    basis = _mean_free_trigonometric_basis(n_pixels, lengths)
    flat = basis.reshape(basis.shape[0], n_points)
    image = _oracle_apply_periodic_conductivity_operator(basis, conductivity, lengths)  # noqa: F821
    operator = flat @ image.reshape(basis.shape[0], n_points).T / n_points
    uniform = np.full((n_pixels, n_pixels), reference_conductivity)
    reference_image = _oracle_apply_periodic_conductivity_operator(basis, uniform, lengths)  # noqa: F821
    reference = flat @ reference_image.reshape(basis.shape[0], n_points).T / n_points
    return {
        "basis": basis,
        "operator": 0.5 * (operator + operator.T),
        "reference_operator": 0.5 * (reference + reference.T),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
import scipy.linalg as sla
N, L = 15, (1.0, 1.0)
rng = np.random.default_rng(11)
C = np.where(rng.random((N, N)) < 0.35, 8.0, 1.0)
def summarize(out):
    A, A0, Q = out["operator"], out["reference_operator"], out["basis"]
    flat = Q.reshape(Q.shape[0], -1)
    w = sla.eigh(A, A0, eigvals_only=True)
    return (
        A.shape, Q.shape,
        round(float(np.abs(A - A.T).max()), 12),
        round(float(np.abs(flat @ flat.T - np.eye(Q.shape[0])).max()), 10),
        round(float(np.abs(flat.sum(axis=1)).max()), 10),
        round(float(np.trace(A)), 6), round(float(np.trace(A0)), 6),
        round(float(w.min()), 9), round(float(w.max()), 9),
        round(float(np.linalg.eigvalsh(A0).min()), 9),
    )
""",
            "call": "summarize(assemble_mean_free_operator_matrices(C, L, 4.5))",
            "gold_call": "summarize(_oracle_assemble_mean_free_operator_matrices(C, L, 4.5))",
        },
        {
            "setup": """import numpy as np
N, L = 9, (2.0, 0.5)
def summarize(out):
    A, A0 = out["operator"], out["reference_operator"]
    w = np.sort(np.linalg.eigvalsh(A0))
    return (
        round(float(np.abs(A - A0).max()), 10),
        round(float(w[0]), 9), round(float(w[-1]), 9), round(float(w.sum()), 6),
        round(float(np.abs(A0 - np.diag(np.diag(A0))).max()), 10),
    )
""",
            "call": "summarize(assemble_mean_free_operator_matrices(np.full((N, N), 3.0), L, 3.0))",
            "gold_call": "summarize(_oracle_assemble_mean_free_operator_matrices(np.full((N, N), 3.0), L, 3.0))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.ones((9, 9)), (1.0, 1.0), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(assemble_mean_free_operator_matrices)",
            "gold_call": "run(_oracle_assemble_mean_free_operator_matrices)",
        },
    ]
