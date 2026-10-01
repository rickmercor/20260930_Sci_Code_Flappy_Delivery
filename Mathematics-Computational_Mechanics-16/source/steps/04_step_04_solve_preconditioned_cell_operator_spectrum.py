"""
Preconditioning the cell operator by the reference operator produces S = I - A_0^{-1} A on mean-free periodic potentials. This operator is bounded and self-adjoint for the energetic inner product (v_1, v_2) = average of grad v_1 . C_0 grad v_2, and its eigenpairs are those of the generalised problem A phi = (1 - lambda) A_0 phi. With the matrices of the previous step the problem is a real symmetric definite pencil, solved so that the eigenvectors are orthonormal for the reference matrix, which is the discrete form of the energetic normalisation. The eigenvalues are returned in ascending order. The spectrum lies in the interval [1 - M / C_0, 1 - m / C_0], with m and M the smallest and largest phase conductivities, because outside that interval the shifted conductivity C + (lambda - 1) C_0 is definite of one sign and the eigenvalue equation has no non-trivial solution. Inside the interval the spectrum accumulates on the values 1 - C_p / C_0, one per phase: those eigenstates fluctuate within phase p and are constant elsewhere, and they play no role in the cell solution, whereas the remaining eigenvalues, which lie in the transitions between the accumulation values, belong to eigenstates with non-trivial behaviour at the interfaces. The step counts the eigenvalues within a stated tolerance of each accumulation value and the eigenvalues that are farther than that tolerance from all of them.

Returns
-------
dict with float array eigenvalues of shape (N - 1,) in ascending order, float array eigenvectors of shape (N - 1, N - 1) whose columns are orthonormal for reference_operator, tuple of two native floats bounds, tuple of native floats plateau_values with one entry per phase, tuple of native ints plateau_counts, native int transition_count, and native float orthonormality_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_preconditioned_cell_operator_spectrum(
    operator: np.ndarray,
    reference_operator: np.ndarray,
    phase_conductivities: np.ndarray,
    reference_conductivity: float,
    plateau_tolerance: float,
) -> dict:
    """Solve the generalised eigenvalue problem of the preconditioned cell operator and classify its spectrum.

    Raises
    ------
    ValueError
        If operator and reference_operator are not square matrices of the same shape, if either differs from its transpose by more than 1e-8 in absolute value, if reference_operator is not positive definite, if a phase conductivity or reference_conductivity is not strictly positive, or if plateau_tolerance is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as sla


def _oracle_solve_preconditioned_cell_operator_spectrum(
    operator: np.ndarray,
    reference_operator: np.ndarray,
    phase_conductivities: np.ndarray,
    reference_conductivity: float,
    plateau_tolerance: float,
) -> dict:
    """Reference implementation."""
    operator = np.asarray(operator, dtype=float)
    reference_operator = np.asarray(reference_operator, dtype=float)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1] or operator.shape != reference_operator.shape:
        raise ValueError("operator and reference_operator must be square matrices of the same shape")
    if np.abs(operator - operator.T).max() > 1e-8 or np.abs(reference_operator - reference_operator.T).max() > 1e-8:
        raise ValueError("both matrices must be symmetric")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    reference_conductivity = float(reference_conductivity)
    if conductivities.size < 1 or np.any(conductivities <= 0.0) or reference_conductivity <= 0.0:
        raise ValueError("conductivities must be strictly positive")
    plateau_tolerance = float(plateau_tolerance)
    if plateau_tolerance <= 0.0:
        raise ValueError("plateau_tolerance must be strictly positive")
    try:
        shifted, vectors = sla.eigh(operator, reference_operator)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError("reference_operator must be positive definite") from error
    eigenvalues = 1.0 - shifted
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    gram = vectors.T @ reference_operator @ vectors
    plateau_values = tuple(float(1.0 - c / reference_conductivity) for c in conductivities)
    on_plateau = np.zeros(eigenvalues.size, dtype=bool)
    counts = []
    for value in plateau_values:
        near = np.abs(eigenvalues - value) < plateau_tolerance
        counts.append(int(near.sum()))
        on_plateau |= near
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": vectors,
        "bounds": (float(1.0 - conductivities.max() / reference_conductivity),
                   float(1.0 - conductivities.min() / reference_conductivity)),
        "plateau_values": plateau_values,
        "plateau_counts": tuple(counts),
        "transition_count": int((~on_plateau).sum()),
        "orthonormality_residual": float(np.abs(gram - np.eye(eigenvalues.size)).max()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
N, L = 21, (1.0, 1.0)
DISCS = [(0.27, 0.31, 0.24, 3), (0.76, 0.72, 0.21, 3), (0.78, 0.22, 0.155, 2), (0.24, 0.79, 0.135, 2)]
COND = np.array([1.0, 3.0, 10.0])
def prepare():
    ax = np.arange(N) / N
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    phase = np.ones((N, N), dtype=int)
    for cx, cy, r, ph in DISCS:
        dx = np.abs(X - cx); dx = np.minimum(dx, 1 - dx)
        dy = np.abs(Y - cy); dy = np.minimum(dy, 1 - dy)
        phase[np.sqrt(dx * dx + dy * dy) < r] = ph
    C = COND[phase - 1]
    k = np.fft.fftfreq(N, d=1.0 / N)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    G = (2j * np.pi * K1, 2j * np.pi * K2)
    modes = []
    for k1 in k.astype(int):
        for k2 in k.astype(int):
            if k1 > 0 or (k1 == 0 and k2 > 0):
                a = 2 * np.pi * (k1 * X + k2 * Y)
                modes += [np.sqrt(2 / N ** 2) * np.cos(a), np.sqrt(2 / N ** 2) * np.sin(a)]
    Q = np.stack(modes)
    def apply(V, c):
        vh = np.fft.fft2(V)
        g1 = np.real(np.fft.ifft2(G[0] * vh)); g2 = np.real(np.fft.ifft2(G[1] * vh))
        return -np.real(np.fft.ifft2(G[0] * np.fft.fft2(c * g1) + G[1] * np.fft.fft2(c * g2)))
    flat = Q.reshape(len(Q), -1)
    A = flat @ apply(Q, C).reshape(len(Q), -1).T / N ** 2
    A0 = flat @ apply(Q, np.full((N, N), 5.5)).reshape(len(Q), -1).T / N ** 2
    return 0.5 * (A + A.T), 0.5 * (A0 + A0.T)
A, A0 = prepare()
def summarize(out):
    lam = out["eigenvalues"]
    return (
        lam.shape, out["eigenvectors"].shape,
        round(float(lam[0]), 9), round(float(lam[-1]), 9), round(float(lam[200]), 9),
        tuple(round(float(v), 9) for v in out["bounds"]),
        tuple(round(float(v), 9) for v in out["plateau_values"]),
        tuple(int(v) for v in out["plateau_counts"]), int(out["transition_count"]),
        int(out["orthonormality_residual"] < 1e-9), int(np.all(np.diff(lam) >= 0.0)),
    )
""",
            "call": "summarize(solve_preconditioned_cell_operator_spectrum(A, A0, COND, 5.5, 1e-4))",
            "gold_call": "summarize(_oracle_solve_preconditioned_cell_operator_spectrum(A, A0, COND, 5.5, 1e-4))",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
M = rng.standard_normal((30, 30))
A0 = M @ M.T + 30 * np.eye(30)
D = np.diag(np.where(np.arange(30) < 10, 0.4, 1.6))
R = np.linalg.cholesky(A0)
A = R @ D @ R.T
def summarize(out):
    lam = out["eigenvalues"]
    V = out["eigenvectors"]
    return (
        round(float(lam.min()), 9), round(float(lam.max()), 9),
        tuple(int(v) for v in out["plateau_counts"]), int(out["transition_count"]),
        round(float(np.abs(V.T @ A0 @ V - np.eye(30)).max()), 8),
        round(float(np.abs(A @ V - A0 @ V * (1 - lam)).max()), 7),
    )
""",
            "call": "summarize(solve_preconditioned_cell_operator_spectrum(A, A0, np.array([0.8, 3.2]), 2.0, 1e-6))",
            "gold_call": "summarize(_oracle_solve_preconditioned_cell_operator_spectrum(A, A0, np.array([0.8, 3.2]), 2.0, 1e-6))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(np.eye(4), np.eye(4), np.array([1.0, 3.0]), 2.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(solve_preconditioned_cell_operator_spectrum)",
            "gold_call": "run(_oracle_solve_preconditioned_cell_operator_spectrum)",
        },
    ]
