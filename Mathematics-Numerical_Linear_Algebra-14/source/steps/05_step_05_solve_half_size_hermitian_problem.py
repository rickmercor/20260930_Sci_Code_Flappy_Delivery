"""
Solve the half-size Hermitian eigenproblem and return eigenpairs with deterministic phase, exact-multiplicity residual-pivot gauges, and balanced Loewdin gauges for tight clusters.

The complex matrix formed from the two blocks is Hermitian. A simple eigenvalue leaves one phase to fix, whereas a repeated or roundoff-scale clustered eigenvalue leaves an effectively unresolved unitary eigenspace gauge. This step fixes both: isolated vectors use an orientation-dependent phase pivot, exact multiplicities use a canonical residual-pivot basis extracted from their spectral projector, and tight clusters balance the selected projector columns by symmetric Loewdin orthogonalization. The paper explicitly notes that recovering a real spectral decomposition from a complex Hermitian eigenbasis needs additional postprocessing in the presence of multiple or tightly clustered eigenvalues.

Returns
-------
one complex (n + 1, n) array with sigma_tilde in row 0 ordered by decreasing magnitude and U in rows 1:, using the phase pivot for simple eigenvalues, the residual-pivot gauge for exact multiplicities, and the balanced Loewdin gauge for tight clusters; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_half_size_hermitian_problem(
    blocks: np.ndarray,
) -> np.ndarray:
    """Return the ordered, phase-fixed eigensystem of ``blocks[0] + 1j * blocks[1]``.

    Raises ``ValueError`` unless every one of the following holds: ``blocks`` is a
    three-dimensional array of shape ``(2, n, n)`` with ``n`` at least one; every
    entry is finite; ``blocks[0]`` is symmetric to an absolute tolerance of
    ``1e-10``; ``blocks[1]`` is skew-symmetric to the same tolerance, so a
    symmetric second block is rejected rather than diagonalized; and every
    eigenvector or clustered eigenspace carries a residual pivot of modulus
    above ``100`` machine epsilons, so its gauge is well defined.

    Parameters
    ----------
    blocks : np.ndarray
        Finite real array of shape ``(2, n, n)`` with symmetric ``blocks[0]``
        and skew-symmetric ``blocks[1]``.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(n + 1, n)``.  Row zero stores the real signed
        eigenvalues sorted by decreasing magnitude with stable ties; rows
        ``1:`` store the matching eigenvectors as the columns of ``U``.

        Consecutive eigenvalues from ``numpy.linalg.eigh`` belong to one
        numerical cluster when the cluster span is at most
        ``256 * eps * max(1, ||K||_2)``, for ``K = blocks[0] + 1j * blocks[1]``.
        Replace all values in a multiple cluster by their arithmetic mean.  To
        choose its coordinate flag, repeatedly project every as-yet-unused
        coordinate vector into the cluster and orthogonally away from accepted
        columns, use two modified-Gram--Schmidt passes, and accept the largest
        residual norm.  Residual norms within relative ``1e-12`` of the largest
        are tied; use the earliest coordinate for a nonnegative cluster and the
        latest for a negative cluster.  If the cluster span is at most
        ``32 * eps * max(1, ||K||_2)``, normalize each accepted residual and
        rotate its selected coordinate to real nonnegative, as for an exact
        multiplicity.  Otherwise form the matrix of spectral-projector columns
        at the selected coordinates and apply symmetric Loewdin
        orthogonalization: if ``C`` contains those columns, form
        ``G = C.conj().T @ C``, replace it by its Hermitian part, and use
        ``C @ G**(-1/2)``, with the inverse square root obtained from
        ``numpy.linalg.eigh``.  Finally rotate column ``j`` so its entry at
        selected coordinate ``j`` is real nonnegative.  Reject an inverse-root
        eigenvalue no larger than ``(100*eps)**2`` or an unstable phase pivot.
        A one-element cluster keeps the solver vector and applies the same
        earliest/nonnegative versus latest/negative rule to entries within
        relative ``1e-12`` of its largest modulus.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _nla14_projector_pivot_gauge(
    cluster_vectors: np.ndarray,
    negative_orientation: bool,
    balanced: bool = False,
) -> np.ndarray:
    """Choose a deterministic basis from a Hermitian spectral projector."""
    projector = cluster_vectors @ cluster_vectors.conj().T
    projector = 0.5 * (projector + projector.conj().T)
    order, multiplicity = cluster_vectors.shape
    accepted = np.empty((order, 0), dtype=complex)
    unused = list(range(order))
    pivots = []
    floor = 100.0 * np.finfo(float).eps

    for _ in range(multiplicity):
        residuals = []
        scores = np.empty(len(unused), dtype=float)
        for location, coordinate in enumerate(unused):
            residual = projector[:, coordinate].copy()
            for _pass in range(2):
                if accepted.shape[1]:
                    residual -= accepted @ (accepted.conj().T @ residual)
            residuals.append(residual)
            scores[location] = np.linalg.norm(residual)

        best = float(scores.max())
        tied = np.flatnonzero(scores >= best * (1.0 - 1e-12))
        location = int(tied[-1] if negative_orientation else tied[0])
        coordinate = unused.pop(location)
        pivots.append(coordinate)
        residual = residuals[location]
        norm = float(np.linalg.norm(residual))
        if norm <= floor:
            raise ValueError("a repeated eigenspace has no stable residual pivot")

        vector = residual / norm
        pivot_value = vector[coordinate]
        if abs(pivot_value) <= floor:
            raise ValueError("a repeated eigenspace has no stable phase pivot")
        vector /= pivot_value / abs(pivot_value)
        accepted = np.column_stack((accepted, vector))

    if balanced:
        columns = projector[:, pivots]
        gram = columns.conj().T @ columns
        gram = 0.5 * (gram + gram.conj().T)
        gram_values, gram_vectors = np.linalg.eigh(gram)
        if float(gram_values[0]) <= floor**2:
            raise ValueError("a tight eigenspace has no stable inverse square root")
        inverse_root = (gram_vectors / np.sqrt(gram_values)) @ gram_vectors.conj().T
        accepted = columns @ inverse_root
        for column, coordinate in enumerate(pivots):
            pivot_value = accepted[coordinate, column]
            if abs(pivot_value) <= floor:
                raise ValueError("a tight eigenspace has no stable phase pivot")
            accepted[:, column] /= pivot_value / abs(pivot_value)

    return accepted


def _oracle_solve_half_size_hermitian_problem(
    blocks: np.ndarray,
) -> np.ndarray:
    """Reference Hermitian eigensolve with deterministic phases."""
    pair = np.asarray(blocks, dtype=float)
    if (
        pair.ndim != 3
        or pair.shape[0] != 2
        or pair.shape[1] != pair.shape[2]
        or pair.shape[1] < 1
    ):
        raise ValueError("blocks must have shape (2, n, n)")
    if not np.all(np.isfinite(pair)):
        raise ValueError("blocks must contain only finite entries")
    hessian, omega = pair
    if not np.allclose(hessian, hessian.T, atol=1e-10, rtol=0.0):
        raise ValueError("blocks[0] must be symmetric")
    if not np.allclose(omega + omega.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("blocks[1] must be skew-symmetric")

    hermitian = hessian + 1j * omega
    values, vectors = np.linalg.eigh(hermitian)
    cluster_tolerance = (
        256.0
        * np.finfo(float).eps
        * max(1.0, float(np.linalg.norm(hermitian, ord=2)))
    )
    exact_tolerance = cluster_tolerance / 8.0

    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[stop] - values[start] <= cluster_tolerance:
            stop += 1

        if stop - start == 1:
            magnitudes = np.abs(vectors[:, start])
            tied = np.flatnonzero(
                magnitudes >= magnitudes.max() * (1.0 - 1e-12)
            )
            pivot = int(tied[-1] if values[start] < 0.0 else tied[0])
            pivot_value = vectors[pivot, start]
            if abs(pivot_value) <= 100.0 * np.finfo(float).eps:
                raise ValueError("an eigenvector has no stable phase pivot")
            vectors[:, start] /= pivot_value / abs(pivot_value)
        else:
            representative = float(np.mean(values[start:stop]))
            span = float(values[stop - 1] - values[start])
            vectors[:, start:stop] = _nla14_projector_pivot_gauge(
                vectors[:, start:stop],
                negative_orientation=representative < 0.0,
                balanced=span > exact_tolerance,
            )
            values[start:stop] = representative
        start = stop

    order = np.argsort(-np.abs(values), kind="stable")
    ordered_values = values[order]
    ordered_vectors = vectors[:, order]
    return np.vstack((ordered_values.astype(complex), ordered_vectors))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return simple, signed, exact/tight clusters, and invalid block pairs."""
    return [
        {
            "setup": """import numpy as np
H = np.diag([3.0, 0.5])
Omega = np.zeros((2, 2))
blocks = np.stack((H, Omega))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
H = np.array([[2.0, 0.3], [0.3, -1.0]])
Omega = np.array([[0.0, 0.4], [-0.4, 0.0]])
blocks = np.stack((H, Omega))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
blocks = np.array([[[-0.2]], [[0.0]]])
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
blocks = np.stack((np.array([[0.0, 3.0], [3.0, 0.0]]), np.zeros((2, 2))))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
blocks = np.stack((np.diag([2.0, -4.0]), np.array([[0.0, 0.7], [-0.7, 0.0]])))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
w = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=complex)
w /= np.linalg.norm(w)
K = 4.0 * np.eye(4) - 5.0 * np.outer(w, w.conj())
blocks = np.stack((K.real, K.imag))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
w = np.array([1.0, -1.0j, -1.0, 1.0j], dtype=complex)
w /= np.linalg.norm(w)
K = -3.0 * np.eye(4) + 5.0 * np.outer(w, w.conj())
blocks = np.stack((K.real, K.imag))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(20260831)
U0 = np.linalg.qr(rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4)))[0]
values = np.array([-1.0, 4.0, 4.0 + 6e-14, 4.0 + 1.2e-13])
K = U0 @ np.diag(values) @ U0.conj().T
blocks = np.stack((K.real, K.imag))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(20260901)
U0 = np.linalg.qr(rng.standard_normal((5,5)) + 1j*rng.standard_normal((5,5)))[0]
values = np.array([-5.0 - 1.5e-13, -5.0 - 7.5e-14, -5.0, 0.4, 2.0])
K = U0 @ np.diag(values) @ U0.conj().T
blocks = np.stack((K.real, K.imag))
""",
            "call": "solve_half_size_hermitian_problem(blocks)",
            "gold_call": "_oracle_solve_half_size_hermitian_problem(blocks)",
        },
        {
            "setup": """import numpy as np
blocks = np.stack((np.array([[1.0, 0.3], [0.9, 2.0]]), np.zeros((2, 2))))
def run_model():
    try:
        solve_half_size_hermitian_problem(blocks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_half_size_hermitian_problem(blocks)
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
blocks = np.zeros((2, 2, 3))
def run_model():
    try:
        solve_half_size_hermitian_problem(blocks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_half_size_hermitian_problem(blocks)
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
blocks = np.stack((np.array([[np.nan, 0.0], [0.0, 1.0]]), np.zeros((2, 2))))
def run_model():
    try:
        solve_half_size_hermitian_problem(blocks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_half_size_hermitian_problem(blocks)
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
blocks = np.array([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 0.2], [0.2, 0.0]]])
def run_model():
    try:
        solve_half_size_hermitian_problem(blocks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_half_size_hermitian_problem(blocks)
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
