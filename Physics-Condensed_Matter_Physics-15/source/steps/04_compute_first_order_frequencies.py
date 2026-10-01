"""
Compute the two first-order MBC-split modes.

## Treating the curvature as a first-order perturbation of the degenerate problem, the

shift is obtained by restricting the frequency-linear curvature term to the target

eigenspace and diagonalizing it there. That restricted operator is Hermitian whenever

the curvature is real and antisymmetric, so its eigenvalues $\lambda_s$ are real shifts

in $\omega^2$. Consistently with discarding the second-order curvature term, the shifts

are carried to the frequency linearly, $\omega_s=\omega_0+\lambda_s/(2\omega_0)$, rather

than through $\sqrt{\omega_0^2+\lambda_s}$, which would retain terms of the order

already dropped. The eigenvectors of the restricted operator, expressed back in the

original coordinates, form the displacement basis that identifies these two modes inside

the full spectrum.

Returns
-------
first_order_modes : np.ndarray     Complex array with shape $(N + 1, 2)$. The first row contains the two     first-order frequencies omega0 + lam/(2*omega0) in ascending order, where     lam are the eigenvalues of the projected first-order operator, and the     remaining $N$ rows contain their associated orthonormal displacement vectors.  Raises ------ ValueError     If unperturbed_analysis is not a finite real array with shape     $(N + 1, N)$, a nonnegative sorted spectrum, and a symmetric idempotent     rank-two projector within atol=$10^{-8}$; if g_tilde is not a finite real     antisymmetric $(N, N)$ array within atol=$10^{-12}$; or if omega0 is not     strictly positive and finite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_first_order_frequencies(
    unperturbed_analysis: "np.ndarray",
    g_tilde: "np.ndarray",
    omega0: float,
) -> "np.ndarray":
    """Compute the two first-order MBC-split modes.

    Parameters
    ----------
    unperturbed_analysis : np.ndarray
        Real packed array from analyze_unperturbed_subspace with shape
        $(N + 1, N)$.
    g_tilde : np.ndarray
        Real antisymmetric curvature matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite degenerate frequency.

    Returns
    -------
    first_order_modes : np.ndarray
        Complex array with shape $(N + 1, 2)$. The first row contains the two
        first-order frequencies omega0 + lam/(2*omega0) in ascending order, where
        lam are the eigenvalues of the projected first-order operator, and the
        remaining $N$ rows contain their associated orthonormal displacement
        vectors.

    Raises
    ------
    ValueError
        If unperturbed_analysis is not a finite real array with shape
        $(N + 1, N)$, a nonnegative sorted spectrum, and a symmetric idempotent
        rank-two projector within atol=$10^{-8}$; if g_tilde is not a finite real
        antisymmetric $(N, N)$ array within atol=$10^{-12}$; or if omega0 is not
        strictly positive and finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_first_order_frequencies(
    unperturbed_analysis: "np.ndarray",
    g_tilde: "np.ndarray",
    omega0: float,
) -> "np.ndarray":
    """Compute the two first-order MBC-split modes.

    Parameters
    ----------
    unperturbed_analysis : np.ndarray
        Real packed array from analyze_unperturbed_subspace with shape
        $(N + 1, N)$.
    g_tilde : np.ndarray
        Real antisymmetric curvature matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite degenerate frequency.

    Returns
    -------
    first_order_modes : np.ndarray
        Complex array with shape $(N + 1, 2)$. The first row contains the two
        first-order frequencies omega0 + lam/(2*omega0) in ascending order, where
        lam are the eigenvalues of the projected first-order operator, and the
        remaining $N$ rows contain their associated orthonormal displacement
        vectors.

    Raises
    ------
    ValueError
        If unperturbed_analysis is not a finite real array with shape
        $(N + 1, N)$, a nonnegative sorted spectrum, and a symmetric idempotent
        rank-two projector within atol=$10^{-8}$; if g_tilde is not a finite real
        antisymmetric $(N, N)$ array within atol=$10^{-12}$; or if omega0 is not
        strictly positive and finite.
    """
    try:
        analysis_raw = np.asarray(unperturbed_analysis)
        curvature_raw = np.asarray(g_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis and curvature must be numeric arrays") from exc
    if np.iscomplexobj(analysis_raw) and np.any(np.imag(analysis_raw) != 0.0):
        raise ValueError("unperturbed_analysis must be real-valued")
    if np.iscomplexobj(curvature_raw) and np.any(np.imag(curvature_raw) != 0.0):
        raise ValueError("g_tilde must be real-valued")
    try:
        analysis = np.asarray(analysis_raw, dtype=float)
        curvature = np.asarray(curvature_raw, dtype=float)
        target = float(omega0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("inputs must contain real numeric values") from exc

    if analysis.ndim != 2 or analysis.shape[0] != analysis.shape[1] + 1:
        raise ValueError("unperturbed_analysis must have shape (N+1, N)")
    n_coordinates = analysis.shape[1]
    if n_coordinates < 2 or not np.all(np.isfinite(analysis)):
        raise ValueError("unperturbed_analysis must be finite with N >= 2")
    spectrum = analysis[0]
    projector = analysis[1:]
    if np.any(spectrum < 0.0) or np.any(np.diff(spectrum) < -1.0e-12):
        raise ValueError("the packed spectrum must be nonnegative and sorted")
    if not np.allclose(projector, projector.T, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must be symmetric")
    if not np.allclose(projector @ projector, projector, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must be idempotent")
    if not np.isclose(np.trace(projector), 2.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("the packed projector must have rank two")
    if curvature.shape != (n_coordinates, n_coordinates):
        raise ValueError("g_tilde must have shape (N, N)")
    if not np.all(np.isfinite(curvature)):
        raise ValueError("g_tilde must be finite")
    if not np.allclose(curvature, -curvature.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("g_tilde must be antisymmetric within atol=1e-12")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("omega0 must be strictly positive and finite")

    projector_values, projector_vectors = np.linalg.eigh(projector)
    basis = projector_vectors[:, np.argsort(projector_values)[-2:]]
    restricted = basis.conj().T @ (2j * target * curvature) @ basis
    restricted = 0.5 * (restricted + restricted.conj().T)
    shifts_in_squared_frequency, restricted_vectors = np.linalg.eigh(restricted)
    frequencies = target + shifts_in_squared_frequency / (2.0 * target)
    order = np.argsort(frequencies.real, kind="stable")
    frequencies = np.asarray(frequencies.real[order], dtype=float)
    displacements = basis.astype(complex) @ restricted_vectors[:, order]
    return np.vstack((frequencies.astype(complex), displacements))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    summary = '''\ndef summarize_first_order(table, degenerate):\n    table = np.asarray(table, dtype=complex)\n    frequencies = table[0]\n    basis = table[1:]\n    norms = np.linalg.norm(basis, axis=0)\n    normalized = basis / norms\n    if degenerate:\n        projectors = [normalized @ normalized.conj().T]\n    else:\n        projectors = [np.outer(normalized[:, j], normalized[:, j].conj()) for j in range(2)]\n    pieces = [frequencies.real.astype(float), frequencies.imag.astype(float), norms]\n    for projector in projectors:\n        pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n'''
    return [
        {
            "setup": summary + "analysis=np.vstack((np.array([2.0,4.0,4.0]),np.diag([0.0,1.0,1.0]))); G=np.array([[0.0,0.0,0.0],[0.0,0.0,0.25],[0.0,-0.25,0.0]]); omega0=4.0",
            "call": "summarize_first_order(compute_first_order_frequencies(analysis, G, omega0), False)",
            "gold_call": "summarize_first_order(_oracle_compute_first_order_frequencies(analysis, G, omega0), False)",
        },
        {
            "setup": summary + "analysis=np.vstack((np.array([3.0,3.0]),np.eye(2))); G=np.array([[0.0,-0.1],[0.1,0.0]]); omega0=3.0",
            "call": "summarize_first_order(compute_first_order_frequencies(analysis, G, omega0), False)",
            "gold_call": "summarize_first_order(_oracle_compute_first_order_frequencies(analysis, G, omega0), False)",
        },
        {
            "setup": summary + "analysis=np.vstack((np.array([1.0,5.0,5.0,8.0]),np.diag([0.0,1.0,1.0,0.0]))); G=np.zeros((4,4)); G[1,2]=1e-10; G[2,1]=-1e-10; omega0=5.0",
            "call": "summarize_first_order(compute_first_order_frequencies(analysis, G, omega0), False)",
            "gold_call": "summarize_first_order(_oracle_compute_first_order_frequencies(analysis, G, omega0), False)",
        },
        {
            "setup": summary + "analysis=np.vstack((np.array([2.0,2.0]),np.eye(2))); G=np.zeros((2,2)); omega0=2.0",
            "call": "summarize_first_order(compute_first_order_frequencies(analysis, G, omega0), True)",
            "gold_call": "summarize_first_order(_oracle_compute_first_order_frequencies(analysis, G, omega0), True)",
        },
    ]
