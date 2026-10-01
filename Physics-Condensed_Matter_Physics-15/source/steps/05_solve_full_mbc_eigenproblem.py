"""
Solve the companion linearization of the full MBC eigenproblem.

## The exact modes satisfy the full frequency-dependent phonon equation, in which the

curvature contributes a frequency-linear term. That makes the problem quadratic in

$\omega$ rather than linear in $\omega^2$, and it is solved by any spectrally equivalent

linearization: a companion matrix on the doubled space, a generalized matrix pencil, or

a direct quadratic-eigenvalue solver. The doubled spectrum pairs each physical branch

with a nonpositive-frequency partner, so only the positive-frequency roots are physical

and retained. Eigenvectors are defined only up to a global phase, so a deterministic

phase convention is applied before they are returned.

Returns
-------
modes : np.ndarray     Complex array with shape $(N + 1, N)$. The first row contains ascending     positive frequencies and each remaining column contains the associated     normalized displacement vector.  Raises ------ ValueError     If k_tilde or g_tilde cannot be represented by real numeric arrays; if     they are not equal-size, nonempty, finite square matrices; if k_tilde     is not symmetric within atol=$10^{-12}$ or is not positive definite; if     g_tilde is not antisymmetric within atol=$10^{-12}$; if     imaginary_tolerance is not strictly positive and finite; if the     companion spectrum does not contain exactly $N$ eligible positive roots;     or if an eligible root has a zero or nonfinite displacement norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_full_mbc_eigenproblem(
    k_tilde: "np.ndarray",
    g_tilde: "np.ndarray",
    imaginary_tolerance: float = 1.0e-9,
) -> "np.ndarray":
    """Solve the companion linearization of the full MBC eigenproblem.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-definite matrix with shape $(N, N)$.
    g_tilde : np.ndarray
        Real antisymmetric matrix with shape $(N, N)$.
    imaginary_tolerance : float
        Strictly positive finite upper bound on the magnitude of an eligible
        companion root's imaginary part.

    Returns
    -------
    modes : np.ndarray
        Complex array with shape $(N + 1, N)$. The first row contains ascending
        positive frequencies and each remaining column contains the associated
        normalized displacement vector.

    Raises
    ------
    ValueError
        If k_tilde or g_tilde cannot be represented by real numeric arrays; if
        they are not equal-size, nonempty, finite square matrices; if k_tilde
        is not symmetric within atol=$10^{-12}$ or is not positive definite; if
        g_tilde is not antisymmetric within atol=$10^{-12}$; if
        imaginary_tolerance is not strictly positive and finite; if the
        companion spectrum does not contain exactly $N$ eligible positive roots;
        or if an eligible root has a zero or nonfinite displacement norm.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_full_mbc_eigenproblem(
    k_tilde: "np.ndarray",
    g_tilde: "np.ndarray",
    imaginary_tolerance: float = 1.0e-9,
) -> "np.ndarray":
    """Solve the companion linearization of the full MBC eigenproblem.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-definite matrix with shape $(N, N)$.
    g_tilde : np.ndarray
        Real antisymmetric matrix with shape $(N, N)$.
    imaginary_tolerance : float
        Strictly positive finite upper bound on the magnitude of an eligible
        companion root's imaginary part.

    Returns
    -------
    modes : np.ndarray
        Complex array with shape $(N + 1, N)$. The first row contains ascending
        positive frequencies and each remaining column contains the associated
        normalized displacement vector.

    Raises
    ------
    ValueError
        If k_tilde or g_tilde cannot be represented by real numeric arrays; if
        they are not equal-size, nonempty, finite square matrices; if k_tilde
        is not symmetric within atol=$10^{-12}$ or is not positive definite; if
        g_tilde is not antisymmetric within atol=$10^{-12}$; if
        imaginary_tolerance is not strictly positive and finite; if the
        companion spectrum does not contain exactly $N$ eligible positive roots;
        or if an eligible root has a zero or nonfinite displacement norm.
    """
    try:
        matrix_raw = np.asarray(k_tilde)
        curvature_raw = np.asarray(g_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_tilde and g_tilde must be numeric arrays") from exc
    if np.iscomplexobj(matrix_raw) and np.any(np.imag(matrix_raw) != 0.0):
        raise ValueError("k_tilde must be real-valued")
    if np.iscomplexobj(curvature_raw) and np.any(np.imag(curvature_raw) != 0.0):
        raise ValueError("g_tilde must be real-valued")
    try:
        matrix = np.asarray(matrix_raw, dtype=float)
        curvature = np.asarray(curvature_raw, dtype=float)
        tolerance = float(imaginary_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must contain real numeric values") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("k_tilde must be a nonempty square matrix")
    if curvature.shape != matrix.shape:
        raise ValueError("g_tilde must have the same square shape as k_tilde")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(curvature)):
        raise ValueError("k_tilde and g_tilde must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_tilde must be symmetric within atol=1e-12")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("k_tilde must be positive definite")
    if not np.allclose(curvature, -curvature.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("g_tilde must be antisymmetric within atol=1e-12")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("imaginary_tolerance must be strictly positive and finite")

    n_coordinates = matrix.shape[0]
    identity = np.eye(n_coordinates, dtype=complex)
    zero = np.zeros((n_coordinates, n_coordinates), dtype=complex)
    quadratic_block = matrix + 2.0 * curvature.conj().T @ curvature
    companion = np.block(
        [
            [zero, identity],
            [quadratic_block.astype(complex), 2j * curvature],
        ]
    )
    roots, companion_vectors = np.linalg.eig(companion)
    eligible = np.flatnonzero(
        (roots.real > 0.0) & (np.abs(roots.imag) <= tolerance)
    )
    if eligible.size != n_coordinates:
        raise ValueError("the companion spectrum must contain exactly N eligible roots")

    eligible = eligible[np.argsort(roots[eligible].real, kind="stable")]
    frequencies = roots[eligible].real.astype(float)
    displacements = np.empty((n_coordinates, n_coordinates), dtype=complex)
    for output_column, eigenvector_index in enumerate(eligible):
        displacement = companion_vectors[:n_coordinates, eigenvector_index]
        norm = np.linalg.norm(displacement)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("an eligible root has an invalid displacement vector")
        displacement = displacement / norm
        pivot = int(np.argmax(np.abs(displacement)))
        displacement = displacement * np.exp(-1j * np.angle(displacement[pivot]))
        if displacement[pivot].real < 0.0:
            displacement = -displacement
        displacements[:, output_column] = displacement

    return np.vstack((frequencies.astype(complex), displacements))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    summary = '''\ndef summarize_full_modes(table, groups):\n    table = np.asarray(table, dtype=complex)\n    frequencies = table[0]\n    displacements = table[1:]\n    norms = np.linalg.norm(displacements, axis=0)\n    normalized = displacements / norms\n    pieces = [frequencies.real.astype(float), frequencies.imag.astype(float), norms]\n    for group in groups:\n        block = normalized[:, group]\n        projector = block @ block.conj().T\n        pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n'''
    return [
        {
            "setup": summary + "K=16.0*np.eye(2); G=np.array([[0.0,-0.2],[0.2,0.0]]); groups=[[0],[1]]",
            "call": "summarize_full_modes(solve_full_mbc_eigenproblem(K, G), groups)",
            "gold_call": "summarize_full_modes(_oracle_solve_full_mbc_eigenproblem(K, G), groups)",
        },
        {
            "setup": summary + "K=np.array([[9.0]]); G=np.zeros((1,1)); groups=[[0]]",
            "call": "summarize_full_modes(solve_full_mbc_eigenproblem(K, G), groups)",
            "gold_call": "summarize_full_modes(_oracle_solve_full_mbc_eigenproblem(K, G), groups)",
        },
        {
            "setup": summary + "K=np.diag([9.0,9.0,25.0,25.0]); G=np.array([[0.0,-0.1,0.0,0.0],[0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.35],[0.0,0.0,-0.35,0.0]]); groups=[[0],[1],[2],[3]]",
            "call": "summarize_full_modes(solve_full_mbc_eigenproblem(K, G), groups)",
            "gold_call": "summarize_full_modes(_oracle_solve_full_mbc_eigenproblem(K, G), groups)",
        },
        {
            "setup": summary + "K=9.0*np.eye(2); G=np.zeros((2,2)); groups=[[0,1]]",
            "call": "summarize_full_modes(solve_full_mbc_eigenproblem(K, G), groups)",
            "gold_call": "summarize_full_modes(_oracle_solve_full_mbc_eigenproblem(K, G), groups)",
        },
    ]
