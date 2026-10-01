"""
Apply one determinant-one electron-label transformation to every determinant.

An SL(n) matrix acting on the left mixes electron labels while multiplying the determinant by det(V)=1. The starting state is unchanged, but a different row is exposed to the one-orbital update.

Returns
-------
Floating transformed-orbital array with the same shape as determinant_orbitals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transform_orbital_frames(
    determinant_orbitals: np.ndarray,
    mixing_matrices: np.ndarray,
) -> np.ndarray:
    """Transform determinant orbital rows by determinant-one matrices.

    Parameters
    ----------
    determinant_orbitals : np.ndarray
        Orbital rows with shape
        (n_det, n_electrons, n_orbitals).
    mixing_matrices : np.ndarray
        One SL(n) matrix per determinant.

    Returns
    -------
    np.ndarray
        Left-transformed orbital rows with the input shape.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transform_orbital_frames(
    determinant_orbitals,
    mixing_matrices,
):
    """Reference state-preserving row transformation."""
    import numpy as np

    orbitals = np.asarray(determinant_orbitals, dtype=float)
    mixing = np.asarray(mixing_matrices, dtype=float)

    if (
        orbitals.ndim != 3
        or orbitals.shape[0] < 1
        or orbitals.shape[1] < 1
    ):
        raise ValueError(
            "determinant_orbitals must be three-dimensional"
        )

    n_det, n, m = orbitals.shape

    if n > m or mixing.shape != (n_det, n, n):
        raise ValueError("mixing matrices do not align")
    if (
        not np.all(np.isfinite(orbitals))
        or not np.all(np.isfinite(mixing))
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(
        np.linalg.det(mixing),
        1.0,
        rtol=0.0,
        atol=2e-10,
    ):
        raise ValueError(
            "every mixing matrix must have determinant one"
        )

    return np.einsum(
        "dij,djm->dim",
        mixing,
        orbitals,
        optimize=True,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = """import numpy as np
determinant_orbitals=np.array([[[1.,0.,0.],[0.,1.,0.]]])"""

    call = (
        "transform_orbital_frames("
        "determinant_orbitals, mixing_matrices)"
    )
    gold = (
        "_oracle_transform_orbital_frames("
        "determinant_orbitals, mixing_matrices)"
    )

    return [
        {
            "setup": (
                base
                + "\nmixing_matrices="
                "np.array([[[1.,0.3],[0.,1.]]])"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                base
                + "\nmixing_matrices="
                "np.array([[[2.,0.],[0.,0.5]]])"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                base
                + "\ndeterminant_orbitals="
                "np.concatenate([determinant_orbitals,"
                "determinant_orbitals[:,:,::-1]])"
                + "\nmixing_matrices="
                "np.repeat(np.eye(2)[None],2,axis=0)"
            ),
            "call": call,
            "gold_call": gold,
        },
    ]
