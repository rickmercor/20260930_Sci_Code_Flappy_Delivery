"""
Solve the effective generalized eigenproblem after relative-cutoff whitening of its positive metric subspace.

Solve the effective generalized eigenproblem after relative-cutoff whitening of its positive metric subspace.

Returns
-------
One-dimensional floating array containing energy, rank, generalized residual norm, and the metric-normalized lowest vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_eidos_update(
    matrix_pencil: np.ndarray,
    relative_cutoff: float,
) -> np.ndarray:
    """Solve a singular-metric variational update.

    Parameters
    ----------
    matrix_pencil : np.ndarray
        Effective [Hamiltonian, metric] array with
        shape (2, n, n).
    relative_cutoff : float
        Positive relative metric-eigenvalue cutoff
        below one.

    Returns
    -------
    np.ndarray
        [energy, retained_rank, residual_norm,
        coefficient_vector...].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_eidos_update(
    matrix_pencil,
    relative_cutoff,
):
    """Reference positive-subspace whitening solution."""
    import numpy as np

    pencil = np.asarray(matrix_pencil, dtype=float)

    if (
        pencil.ndim != 3
        or pencil.shape[0] != 2
        or pencil.shape[1] < 1
        or pencil.shape[1] != pencil.shape[2]
    ):
        raise ValueError(
            "matrix_pencil must have shape (2,n,n)"
        )
    if not np.all(np.isfinite(pencil)):
        raise ValueError("matrix_pencil must be finite")
    if (
        not np.isfinite(relative_cutoff)
        or relative_cutoff <= 0.0
        or relative_cutoff >= 1.0
    ):
        raise ValueError(
            "relative_cutoff must lie between zero and one"
        )

    hamiltonian, metric = pencil

    if not np.allclose(
        hamiltonian,
        hamiltonian.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "effective Hamiltonian must be symmetric"
        )
    if not np.allclose(
        metric,
        metric.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "effective metric must be symmetric"
        )

    values, vectors = np.linalg.eigh(metric)
    scale = float(values[-1])

    if scale <= 0.0 or values[0] < -1e-9 * scale:
        raise ValueError(
            "metric must be positive semidefinite and nonzero"
        )

    keep = values > relative_cutoff * scale

    if not np.any(keep):
        raise ValueError("cutoff removes the full space")

    whitening = (
        vectors[:, keep]
        / np.sqrt(values[keep])[None, :]
    )

    reduced = (
        whitening.T @ hamiltonian @ whitening
    )
    reduced = 0.5 * (reduced + reduced.T)

    energies, reduced_vectors = np.linalg.eigh(reduced)
    energy = float(energies[0])
    vector = whitening @ reduced_vectors[:, 0]

    vector /= np.sqrt(
        float(vector @ metric @ vector)
    )

    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector

    residual = np.linalg.norm(
        hamiltonian @ vector
        - energy * metric @ vector
    )

    return np.concatenate(
        (
            [
                energy,
                float(np.count_nonzero(keep)),
                residual,
            ],
            vector,
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    call = (
        "solve_eidos_update("
        "matrix_pencil, relative_cutoff)"
    )
    gold = (
        "_oracle_solve_eidos_update("
        "matrix_pencil, relative_cutoff)"
    )

    return [
        {
            "setup": (
                "import numpy as np\n"
                "matrix_pencil=np.array(["
                "np.diag([2.,0.]),"
                "np.diag([1.,0.])]); "
                "relative_cutoff=1e-12"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "matrix_pencil=np.array(["
                "np.diag([8.,3.,0.]),"
                "np.diag([4.,1.,0.])]); "
                "relative_cutoff=1e-12"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "metric=np.array([[1.,1.],[1.,1.]]); "
                "matrix_pencil=np.stack(["
                "1.5*metric,metric]); "
                "relative_cutoff=1e-12"
            ),
            "call": call,
            "gold_call": gold,
        },
    ]
