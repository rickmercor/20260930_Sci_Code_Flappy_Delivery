"""
Only a small part of the spectrum carries the cell solution, so a reduced model keeps the eigenstates whose components are largest in magnitude and discards the rest. Because the eigenstates are orthonormal for the energetic inner product, the relative error of the truncated fluctuation in the energetic norm is available without forming any field: it is the root of the tail sum of squared components over the total sum. The number of retained eigenstates is therefore chosen as the smallest count for which that error falls strictly below a prescribed threshold when eigenstates are admitted in order of decreasing component magnitude. The reduced effective response is the average flux of the truncated field, average of C times E plus the sum of p_j beta_j over the retained set, which is also what the spectral form gives when the sum is restricted to that set and, since the eigenstates diagonalise the cell operator, what the energy of the truncated field gives as well. The relative error of each component of the reduced flux against the full flux is returned, together with the retained count, the achieved energetic error, the error that one fewer eigenstate would leave, and the retained fraction of the spectrum.

Returns
-------
dict with native int n_retained, integer array selected_indices of shape (n_retained,) in decreasing order of component magnitude, float array selected_eigenvalues of the same shape, native float energy_error achieved with n_retained eigenstates, native float energy_error_before with one fewer, float array truncated_column of shape (2,), float array relative_errors of shape (2,), and native float retained_fraction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_truncated_reduced_model(
    eigenvalues: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    mean_conductivity: float,
    effective_column: np.ndarray,
    macroscopic_gradient: np.ndarray,
    energy_error_threshold: float,
) -> dict:
    """Select the dominant eigenstates by an energetic error threshold and form the reduced effective flux.

    Raises
    ------
    ValueError
        If the array shapes are mutually inconsistent, if every component is zero, if mean_conductivity is not strictly positive, if macroscopic_gradient or effective_column does not have two components, or if energy_error_threshold is not strictly between zero and one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_truncated_reduced_model(
    eigenvalues: np.ndarray,
    components: np.ndarray,
    spectral_vectors: np.ndarray,
    mean_conductivity: float,
    effective_column: np.ndarray,
    macroscopic_gradient: np.ndarray,
    energy_error_threshold: float,
) -> dict:
    """Reference implementation."""
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    components = np.asarray(components, dtype=float).ravel()
    spectral_vectors = np.asarray(spectral_vectors, dtype=float)
    effective_column = np.asarray(effective_column, dtype=float).ravel()
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    n_modes = components.size
    if eigenvalues.shape != (n_modes,) or spectral_vectors.shape != (n_modes, 2):
        raise ValueError("eigenvalues and spectral_vectors must match the number of components")
    if gradient.shape != (2,) or effective_column.shape != (2,):
        raise ValueError("macroscopic_gradient and effective_column must have two components")
    mean_conductivity = float(mean_conductivity)
    if mean_conductivity <= 0.0:
        raise ValueError("mean_conductivity must be strictly positive")
    threshold = float(energy_error_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("energy_error_threshold must lie strictly between zero and one")
    total = float(np.sum(components ** 2))
    if total <= 0.0:
        raise ValueError("at least one component must be non-zero")
    order = np.argsort(-np.abs(components), kind="stable")
    squared = components[order] ** 2
    tail = total - np.concatenate([[0.0], np.cumsum(squared)])
    errors = np.sqrt(np.maximum(tail, 0.0) / total)
    n_retained = int(np.argmax(errors < threshold))
    selected = order[:n_retained]
    truncated_column = mean_conductivity * gradient + spectral_vectors[selected].T @ components[selected]
    return {
        "n_retained": n_retained,
        "selected_indices": selected,
        "selected_eigenvalues": eigenvalues[selected],
        "energy_error": float(errors[n_retained]),
        "energy_error_before": float(errors[n_retained - 1]) if n_retained > 0 else 1.0,
        "truncated_column": truncated_column,
        "relative_errors": np.abs(truncated_column - effective_column) / np.abs(effective_column),
        "retained_fraction": n_retained / float(n_modes),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
M = 300
LAM = np.sort(rng.uniform(-0.8, 0.8, M))
P = rng.standard_normal(M) * np.exp(-np.arange(M) / 40.0)
BETA = rng.standard_normal((M, 2)) * 0.05
E = np.array([1.0, 0.0])
FULL = 4.0 * E + BETA.T @ P
def summarize(out):
    return (
        int(out["n_retained"]), out["selected_indices"].shape,
        round(float(out["energy_error"]), 10), round(float(out["energy_error_before"]), 10),
        int(out["energy_error"] < 0.1 <= out["energy_error_before"]),
        tuple(round(float(v), 10) for v in out["truncated_column"]),
        tuple(round(float(v), 10) for v in out["relative_errors"]),
        round(float(out["retained_fraction"]), 10),
        int(np.all(np.diff(np.abs(P[out["selected_indices"]])) <= 0.0)),
        round(float(out["selected_eigenvalues"][0]), 10),
    )
""",
            "call": "summarize(build_truncated_reduced_model(LAM, P, BETA, 4.0, FULL, E, 0.1))",
            "gold_call": "summarize(_oracle_build_truncated_reduced_model(LAM, P, BETA, 4.0, FULL, E, 0.1))",
        },
        {
            "setup": """import numpy as np
LAM = np.array([-0.5, 0.1, 0.3, 0.6])
P = np.array([0.0, 3.0, -4.0, 0.0])
BETA = np.array([[1.0, 0.0], [0.5, 0.2], [-0.1, 0.4], [0.0, 1.0]])
E = np.array([0.0, 1.0])
FULL = 2.0 * E + BETA.T @ P
def summarize(fn):
    strict = fn(LAM, P, BETA, 2.0, FULL, E, 0.5)
    loose = fn(LAM, P, BETA, 2.0, FULL, E, 0.999)
    return (
        int(strict["n_retained"]), tuple(int(v) for v in strict["selected_indices"]),
        round(float(strict["energy_error"]), 12), round(float(strict["energy_error_before"]), 12),
        tuple(round(float(v), 12) for v in strict["truncated_column"]),
        tuple(round(float(v), 12) for v in strict["relative_errors"]),
        int(loose["n_retained"]), round(float(loose["energy_error_before"]), 12),
        round(float(loose["retained_fraction"]), 12),
    )
""",
            "call": "summarize(build_truncated_reduced_model)",
            "gold_call": "summarize(_oracle_build_truncated_reduced_model)",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.zeros(4), np.ones(4), np.zeros((4, 2)), 1.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(build_truncated_reduced_model)",
            "gold_call": "run(_oracle_build_truncated_reduced_model)",
        },
    ]
