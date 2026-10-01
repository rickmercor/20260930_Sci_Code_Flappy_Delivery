"""
Regularize the nonorthogonal projected eigenproblem by diagonalizing the overlap matrix, retaining precisely the modes whose eigenvalues satisfy the strict relative rule $s_i > \texttt{relative\_cutoff}\,s_{\max}$, and solving in that positive resolved subspace. Return the lowest shifted energy and the corresponding full-basis coefficient vector in the original nonorthogonal Krylov coordinates, normalized so that $v^\dagger S v=1$, with a deterministic phase.

Time-evolved Krylov vectors can become nearly linearly dependent, so small overlap eigenvalues amplify perturbations in a generalized eigenvalue problem. Thresholding removes those unresolved directions before canonical orthogonalization. Canonical orthogonalization is used only to solve the reduced problem; after mapping the Ritz vector back to the original Krylov coordinates, its physical norm is the overlap metric $v^\dagger S v$. The phase is physically irrelevant but is fixed for deterministic numerical testing by rotating the component of greatest magnitude to be real and nonnegative.

Returns
-------
tuple (shifted_energy, state_coefficients, overlap_eigenvalues, retained_indices), with a native float, a complex128 vector of length n, a float64 vector of length n in ascending order, and an int64 index vector
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import numpy as np

def solve_resolved_krylov_ground_state(
    overlap: np.ndarray,
    shifted_hamiltonian: np.ndarray,
    relative_cutoff: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the thresholded projected generalized eigenproblem.

    Parameters
    ----------
    overlap : np.ndarray
        Hermitian overlap matrix of shape ``(n, n)``.
    shifted_hamiltonian : np.ndarray
        Hermitian projected shifted-Hamiltonian matrix of shape ``(n, n)``.
    relative_cutoff : float
        Finite scalar strictly between 0 and 1. A mode is retained only when
        its eigenvalue is strictly greater than ``relative_cutoff * s_max``.

    Returns
    -------
    shifted_energy : float
        Lowest generalized eigenvalue in the retained subspace.
    state_coefficients : np.ndarray
        Complex128 full-basis coefficient vector of length ``n`` in the
        original nonorthogonal Krylov coordinates, normalized to
        ``state_coefficients.conj() @ overlap @ state_coefficients = 1``
        and phase fixed deterministically.
    overlap_eigenvalues : np.ndarray
        Ascending float64 overlap eigenvalues.
    retained_indices : np.ndarray
        Int64 indices of the strictly retained overlap eigenmodes.

    Raises
    ------
    ValueError
        If matrix shapes, Hermiticity, finiteness, cutoff, or the resolved
        positive subspace is invalid.
    """
    return (
        float("nan"),
        np.empty(len(overlap), dtype=np.complex128),
        np.empty(len(overlap), dtype=np.float64),
        np.empty(0, dtype=np.int64),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_resolved_krylov_ground_state(
    overlap: np.ndarray,
    shifted_hamiltonian: np.ndarray,
    relative_cutoff: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    import math
    from numbers import Real

    import numpy as np

    if isinstance(relative_cutoff, bool) or not isinstance(relative_cutoff, Real):
        raise ValueError("relative_cutoff must be a finite real scalar in (0, 1)")
    cutoff = float(relative_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("relative_cutoff must lie strictly between 0 and 1")

    try:
        s = np.asarray(overlap, dtype=np.complex128)
        h = np.asarray(shifted_hamiltonian, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap and shifted_hamiltonian must be numeric matrices") from exc
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] < 1 or h.shape != s.shape:
        raise ValueError("both matrices must have the same nonempty square shape")
    if not np.all(np.isfinite(s.real)) or not np.all(np.isfinite(s.imag)):
        raise ValueError("overlap must be finite")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("shifted_hamiltonian must be finite")
    if not np.allclose(s, s.conjugate().T, rtol=0.0, atol=1.0e-10):
        raise ValueError("overlap must be Hermitian within atol=1e-10")
    if not np.allclose(h, h.conjugate().T, rtol=0.0, atol=1.0e-10):
        raise ValueError("shifted_hamiltonian must be Hermitian within atol=1e-10")

    s = 0.5 * (s + s.conjugate().T)
    h = 0.5 * (h + h.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    s_max = float(eigenvalues[-1])
    if not math.isfinite(s_max) or s_max <= 0.0:
        raise ValueError("overlap must possess a positive resolved eigenvalue")
    retained = np.flatnonzero(eigenvalues > cutoff * s_max)
    if retained.size == 0 or np.any(eigenvalues[retained] <= 0.0):
        raise ValueError("the strict cutoff leaves no positive resolved subspace")

    transform = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
    reduced_hamiltonian = transform.conjugate().T @ h @ transform
    reduced_hamiltonian = 0.5 * (reduced_hamiltonian + reduced_hamiltonian.conjugate().T)
    energies, vectors = np.linalg.eigh(reduced_hamiltonian)
    state = transform @ vectors[:, 0]
    metric_norm_squared = np.vdot(state, s @ state)
    metric_norm_real = float(np.real(metric_norm_squared))
    metric_norm_tolerance = 1.0e-10 * max(1.0, abs(metric_norm_real))
    if (
        not np.isfinite(metric_norm_squared.real)
        or not np.isfinite(metric_norm_squared.imag)
        or abs(float(np.imag(metric_norm_squared))) > metric_norm_tolerance
        or metric_norm_real <= 0.0
    ):
        raise ValueError("the ground-state coefficient vector has invalid overlap norm")
    state = state / math.sqrt(metric_norm_real)
    pivot = int(np.argmax(np.abs(state)))
    if abs(state[pivot]) > 0.0:
        state = state * np.exp(-1j * np.angle(state[pivot]))
    state = np.asarray(state, dtype=np.complex128)

    return float(energies[0]), state, eigenvalues.astype(np.float64), retained.astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
S = np.diag([1.0, 0.4, 0.02]).astype(complex)
H = np.array([[-1.0, 0.1, 0.0], [0.1, 0.2, 0.03j], [0.0, -0.03j, 0.6]], dtype=complex)

def pack_result(result):
    energy, state, eigenvalues, retained = result
    return np.concatenate([
        np.asarray([
            energy, state.ndim, state.size,
            eigenvalues.ndim, eigenvalues.size, retained.ndim, retained.size,
        ], dtype=float),
        state.real.ravel(), state.imag.ravel(), eigenvalues.ravel(),
        retained.astype(float).ravel(),
    ])
""",
            "call": "pack_result(solve_resolved_krylov_ground_state(S, H, 1.0e-3))",
            "gold_call": "pack_result(_oracle_solve_resolved_krylov_ground_state(S, H, 1.0e-3))",
        },
        {
            "setup": """import numpy as np
S = np.diag([1.0e-4, 1.0]).astype(complex)
H = np.diag([-1.0e-3, 2.0]).astype(complex)

def pack_result(result):
    energy, state, eigenvalues, retained = result
    return np.concatenate([
        np.asarray([
            energy, state.ndim, state.size,
            eigenvalues.ndim, eigenvalues.size, retained.ndim, retained.size,
        ], dtype=float),
        state.real.ravel(), state.imag.ravel(), eigenvalues.ravel(),
        retained.astype(float).ravel(),
    ])
""",
            "call": "pack_result(solve_resolved_krylov_ground_state(S, H, 1.0e-4))",
            "gold_call": "pack_result(_oracle_solve_resolved_krylov_ground_state(S, H, 1.0e-4))",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.999999], [0.999999, 1.0]], dtype=complex)
H = np.array([[-0.4, -0.2j], [0.2j, 0.7]], dtype=complex)

def pack_result(result):
    energy, state, eigenvalues, retained = result
    return np.concatenate([
        np.asarray([
            energy, state.ndim, state.size,
            eigenvalues.ndim, eigenvalues.size, retained.ndim, retained.size,
        ], dtype=float),
        state.real.ravel(), state.imag.ravel(), eigenvalues.ravel(),
        retained.astype(float).ravel(),
    ])
""",
            "call": "pack_result(solve_resolved_krylov_ground_state(S, H, 1.0e-5))",
            "gold_call": "pack_result(_oracle_solve_resolved_krylov_ground_state(S, H, 1.0e-5))",
        },
        {
            "setup": """import numpy as np
S = np.eye(2, dtype=complex)
H = np.eye(3, dtype=complex)
def run_model():
    try:
        solve_resolved_krylov_ground_state(S, H, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_resolved_krylov_ground_state(S, H, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
S = np.array([[1.0, 0.2], [0.1, 1.0]], dtype=complex)
H = np.eye(2, dtype=complex)
def run_model():
    try:
        solve_resolved_krylov_ground_state(S, H, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_resolved_krylov_ground_state(S, H, 1.0e-4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
