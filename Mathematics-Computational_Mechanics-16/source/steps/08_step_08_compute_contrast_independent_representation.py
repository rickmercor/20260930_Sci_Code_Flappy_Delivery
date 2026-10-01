"""
For a two-phase composite the spectral analysis separates the geometry from the material contrast. Writing the conductivity as C_m chi + C_i (1 - chi) with chi the indicator of the matrix, choosing the reference medium equal to the inclusion conductivity C_i, and introducing the contrast z = C_i / C_m together with s = z / (z - 1), the generalised eigenvalue problem of the preconditioned operator becomes div(chi grad psi) = mu Laplacian psi with mu = s lambda. This problem involves the indicator alone, its eigenvalues lie in [0, 1] because chi takes the values zero and one, and it accumulates on the two ends of that interval: eigenstates fluctuating only inside the inclusions have mu near zero and those fluctuating only in the matrix have mu near one. Normalising the eigenstates to unit mean squared gradient, the coupling of each to a macroscopic gradient E is the normalised projection E . g_j with g_j the average of chi grad psi_j, which does not depend on the contrast either. The effective flux for any contrast then follows from the geometric data alone: average of C times E plus (C_i - C_m) times the sum over j of g_j (g_j . E) / (mu_j - s). The step solves the geometric problem, counts the eigenvalues within a tolerance of each end of the interval, evaluates the representation at the requested contrast and compares it with a direct solution of the cell problem at that contrast.

Returns
-------
dict with float array geometric_eigenvalues of shape (N - 1,) in ascending order, tuple of two native ints limit_counts for the windows at zero and at one, float array normalised_projections of shape (N - 1,), native float max_projection, native float contrast_parameter s, float array effective_column of shape (2,) from the representation, float array direct_column of shape (2,) from the direct solution, and native float consistency_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_contrast_independent_representation(
    phase: np.ndarray,
    cell_lengths: tuple,
    matrix_phase: int,
    matrix_conductivity: float,
    contrast: float,
    macroscopic_gradient: np.ndarray,
    limit_tolerance: float,
) -> dict:
    """Solve the geometric eigenvalue problem of a two-phase cell and evaluate its contrast-independent spectral representation.

    Raises
    ------
    ValueError
        If phase is not a square integer array of odd size at least three, if matrix_phase does not occur in phase or every sample point carries it, if matrix_conductivity is not strictly positive, if contrast is not strictly positive or equals one, if macroscopic_gradient does not have two components, if a cell length is not strictly positive, or if limit_tolerance is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as sla


def _frequency_multipliers(n_pixels, cell_lengths):
    """Spectral derivative multipliers 2 pi i k_d / L_d on the odd grid, one array per axis."""
    integer_frequencies = np.fft.fftfreq(n_pixels, d=1.0 / n_pixels)
    k_1, k_2 = np.meshgrid(integer_frequencies, integer_frequencies, indexing="ij")
    return (2j * np.pi * k_1 / float(cell_lengths[0]), 2j * np.pi * k_2 / float(cell_lengths[1]))


def _spectral_gradient(field, multipliers):
    """Exact derivative of the trigonometric interpolant along both axes."""
    transform = np.fft.fft2(field, axes=(-2, -1))
    return tuple(np.real(np.fft.ifft2(m * transform, axes=(-2, -1))) for m in multipliers)


def _spectral_divergence(component_1, component_2, multipliers):
    """Spectral divergence of a vector field; its mean vanishes by construction."""
    transform = (multipliers[0] * np.fft.fft2(component_1, axes=(-2, -1))
                 + multipliers[1] * np.fft.fft2(component_2, axes=(-2, -1)))
    return np.real(np.fft.ifft2(transform, axes=(-2, -1)))


def _oracle_compute_contrast_independent_representation(
    phase: np.ndarray,
    cell_lengths: tuple,
    matrix_phase: int,
    matrix_conductivity: float,
    contrast: float,
    macroscopic_gradient: np.ndarray,
    limit_tolerance: float,
) -> dict:
    """Reference implementation."""
    phase = np.asarray(phase)
    if (phase.ndim != 2 or phase.shape[0] != phase.shape[1] or phase.shape[0] < 3
            or phase.shape[0] % 2 == 0 or not np.issubdtype(phase.dtype, np.integer)):
        raise ValueError("phase must be a square integer array of odd size at least three")
    indicator = (phase == int(matrix_phase)).astype(float)
    if indicator.sum() == 0.0 or indicator.sum() == indicator.size:
        raise ValueError("matrix_phase must occur without filling the whole cell")
    matrix_conductivity = float(matrix_conductivity)
    contrast = float(contrast)
    if matrix_conductivity <= 0.0:
        raise ValueError("matrix_conductivity must be strictly positive")
    if contrast <= 0.0 or contrast == 1.0:
        raise ValueError("contrast must be strictly positive and different from one")
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    limit_tolerance = float(limit_tolerance)
    if limit_tolerance <= 0.0:
        raise ValueError("limit_tolerance must be strictly positive")
    n_pixels = phase.shape[0]
    n_points = n_pixels * n_pixels
    multipliers = _frequency_multipliers(n_pixels, lengths)
    geometric = _oracle_assemble_mean_free_operator_matrices(indicator, lengths, 1.0)  # noqa: F821
    basis = geometric["basis"]
    mu, vectors = sla.eigh(geometric["operator"], geometric["reference_operator"])
    order = np.argsort(mu, kind="stable")
    mu = mu[order]
    vectors = vectors[:, order]
    eigenstates = np.tensordot(vectors.T, basis, axes=(1, 0))
    gradient_1, gradient_2 = _spectral_gradient(eigenstates, multipliers)
    couplings = np.stack([
        (indicator * gradient_1).mean(axis=(1, 2)),
        (indicator * gradient_2).mean(axis=(1, 2)),
    ], axis=1)
    projections = couplings @ gradient
    inclusion_conductivity = contrast * matrix_conductivity
    s_parameter = contrast / (contrast - 1.0)
    conductivity = np.where(indicator > 0.5, matrix_conductivity, inclusion_conductivity)
    effective_column = (conductivity.mean() * gradient
                        + (inclusion_conductivity - matrix_conductivity) * (couplings.T @ (projections / (mu - s_parameter))))
    direct = _oracle_assemble_mean_free_operator_matrices(conductivity, lengths, inclusion_conductivity)  # noqa: F821
    load_field = _spectral_divergence(conductivity * gradient[0], conductivity * gradient[1], multipliers)
    load = basis.reshape(basis.shape[0], n_points) @ load_field.ravel() / n_points
    fluctuation = np.tensordot(np.linalg.solve(direct["operator"], load), basis, axes=(0, 0))
    fluctuation_1, fluctuation_2 = _spectral_gradient(fluctuation, multipliers)
    direct_column = np.array([
        (conductivity * (gradient[0] + fluctuation_1)).mean(),
        (conductivity * (gradient[1] + fluctuation_2)).mean(),
    ])
    return {
        "geometric_eigenvalues": mu,
        "limit_counts": (int((mu < limit_tolerance).sum()), int((mu > 1.0 - limit_tolerance).sum())),
        "normalised_projections": projections,
        "max_projection": float(np.abs(projections).max()),
        "contrast_parameter": float(s_parameter),
        "effective_column": effective_column,
        "direct_column": direct_column,
        "consistency_residual": float(np.abs(effective_column - direct_column).max()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
N = 21
DISCS = [(0.27, 0.31, 0.24, 3), (0.76, 0.72, 0.21, 3), (0.78, 0.22, 0.155, 2), (0.24, 0.79, 0.135, 2)]
ax = np.arange(N) / N
X, Y = np.meshgrid(ax, ax, indexing="ij")
PHASE = np.ones((N, N), dtype=int)
for cx, cy, r, ph in DISCS:
    dx = np.abs(X - cx); dx = np.minimum(dx, 1 - dx)
    dy = np.abs(Y - cy); dy = np.minimum(dy, 1 - dy)
    PHASE[np.sqrt(dx * dx + dy * dy) < r] = ph
def summarize(out):
    mu = out["geometric_eigenvalues"]
    return (
        mu.shape, round(float(mu.min()), 9), round(float(mu.max()), 9), round(float(mu[100]), 9),
        tuple(int(v) for v in out["limit_counts"]), round(float(out["max_projection"]), 9),
        round(float(out["contrast_parameter"]), 12),
        tuple(round(float(v), 9) for v in out["effective_column"]),
        int(out["consistency_residual"] < 1e-9), int(np.all(np.diff(mu) >= 0.0)),
        round(float(np.sum(out["normalised_projections"] ** 2)), 9),
    )
""",
            "call": "summarize(compute_contrast_independent_representation(PHASE, (1.0, 1.0), 1, 1.0, 10.0, np.array([1.0, 0.0]), 1e-4))",
            "gold_call": "summarize(_oracle_compute_contrast_independent_representation(PHASE, (1.0, 1.0), 1, 1.0, 10.0, np.array([1.0, 0.0]), 1e-4))",
        },
        {
            "setup": """import numpy as np
N = 15
ax = np.arange(N) / N
X, Y = np.meshgrid(ax * 2.0, ax, indexing="ij")
PHASE = np.where((X - 1.0) ** 2 + (Y - 0.5) ** 2 < 0.3 ** 2, 2, 1)
def summarize(fn):
    weak = fn(PHASE, (2.0, 1.0), 1, 2.0, 0.25, np.array([0.0, 1.0]), 1e-3)
    strong = fn(PHASE, (2.0, 1.0), 1, 2.0, 40.0, np.array([0.0, 1.0]), 1e-3)
    return (
        tuple(int(v) for v in weak["limit_counts"]),
        round(float(np.abs(weak["geometric_eigenvalues"] - strong["geometric_eigenvalues"]).max()), 12),
        round(float(abs(weak["max_projection"] - strong["max_projection"])), 12),
        round(float(weak["contrast_parameter"]), 12), round(float(strong["contrast_parameter"]), 12),
        tuple(round(float(v), 9) for v in weak["effective_column"]),
        tuple(round(float(v), 9) for v in strong["effective_column"]),
        int(weak["consistency_residual"] < 1e-9), int(strong["consistency_residual"] < 1e-8),
        int(weak["effective_column"][1] < 2.0), int(strong["effective_column"][1] > 2.0),
    )
""",
            "call": "summarize(compute_contrast_independent_representation)",
            "gold_call": "summarize(_oracle_compute_contrast_independent_representation)",
        },
        {
            "setup": """import numpy as np
PHASE = np.ones((9, 9), dtype=int)
PHASE[3:6, 3:6] = 2
def run(fn):
    try:
        fn(PHASE, (1.0, 1.0), 1, 1.0, 1.0, np.array([1.0, 0.0]), 1e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(compute_contrast_independent_representation)",
            "gold_call": "run(_oracle_compute_contrast_independent_representation)",
        },
    ]
