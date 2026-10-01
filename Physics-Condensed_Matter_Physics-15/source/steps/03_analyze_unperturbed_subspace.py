"""
Return the unperturbed spectrum and target-subspace projector.

## Setting the curvature to zero leaves the conventional harmonic problem $\widetilde

K\epsilon^{(0)}=\omega^{(0)2}\epsilon^{(0)}$, which is the starting point for

perturbation theory. Frequencies are the nonnegative square roots of its eigenvalues.

Where the conventional dynamical matrix retains an artificial symmetry it predicts

degenerate branches, and the modes that the curvature will split are found by collecting

the eigenvectors whose frequencies coincide with the target $\omega_0$. A degenerate

eigenspace has no preferred basis, so the subspace is reported as its orthogonal

projector, which is basis-independent.

Returns
-------
analysis : np.ndarray     Real array with shape $(N + 1, N)$. The first row contains the sorted     unperturbed frequencies and the remaining rows contain the rank-two     orthogonal projector onto the target degenerate subspace.  Raises ------ ValueError     If k_tilde cannot be represented by a real numeric array; if it is not     a nonempty finite square matrix symmetric with rtol=0 and atol=$10^{-12}$;     if it has an eigenvalue below -$10^{-10}$; if omega0 or     degeneracy_tolerance is not strictly positive and finite; or if     exactly two frequencies are not within degeneracy_tolerance of omega0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def analyze_unperturbed_subspace(
    k_tilde: "np.ndarray",
    omega0: float,
    degeneracy_tolerance: float = 1.0e-8,
) -> "np.ndarray":
    """Return the unperturbed spectrum and target-subspace projector.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-semidefinite matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    degeneracy_tolerance : float
        Strictly positive finite absolute tolerance for selecting frequencies
        equal to omega0.

    Returns
    -------
    analysis : np.ndarray
        Real array with shape $(N + 1, N)$. The first row contains the sorted
        unperturbed frequencies and the remaining rows contain the rank-two
        orthogonal projector onto the target degenerate subspace.

    Raises
    ------
    ValueError
        If k_tilde cannot be represented by a real numeric array; if it is not
        a nonempty finite square matrix symmetric with rtol=0 and atol=$10^{-12}$;
        if it has an eigenvalue below -$10^{-10}$; if omega0 or
        degeneracy_tolerance is not strictly positive and finite; or if
        exactly two frequencies are not within degeneracy_tolerance of omega0.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_analyze_unperturbed_subspace(
    k_tilde: "np.ndarray",
    omega0: float,
    degeneracy_tolerance: float = 1.0e-8,
) -> "np.ndarray":
    """Return the unperturbed spectrum and target-subspace projector.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-semidefinite matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    degeneracy_tolerance : float
        Strictly positive finite absolute tolerance for selecting frequencies
        equal to omega0.

    Returns
    -------
    analysis : np.ndarray
        Real array with shape $(N + 1, N)$. The first row contains the sorted
        unperturbed frequencies and the remaining rows contain the rank-two
        orthogonal projector onto the target degenerate subspace.

    Raises
    ------
    ValueError
        If k_tilde cannot be represented by a real numeric array; if it is not
        a nonempty finite square matrix symmetric with rtol=0 and atol=$10^{-12}$;
        if it has an eigenvalue below -$10^{-10}$; if omega0 or
        degeneracy_tolerance is not strictly positive and finite; or if
        exactly two frequencies are not within degeneracy_tolerance of omega0.
    """
    try:
        matrix_raw = np.asarray(k_tilde)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_tilde must be a numeric array") from exc
    if np.iscomplexobj(matrix_raw) and np.any(np.imag(matrix_raw) != 0.0):
        raise ValueError("k_tilde must be real-valued")
    try:
        matrix = np.asarray(matrix_raw, dtype=float)
        target = float(omega0)
        tolerance = float(degeneracy_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix and scalar inputs must be real numeric values") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("k_tilde must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("k_tilde must be finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("k_tilde must be symmetric within atol=1e-12")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("omega0 must be strictly positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("degeneracy_tolerance must be strictly positive and finite")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) < -1.0e-10:
        raise ValueError("k_tilde must be positive semidefinite")
    frequencies = np.sqrt(np.clip(eigenvalues, 0.0, None))
    target_indices = np.flatnonzero(np.abs(frequencies - target) <= tolerance)
    if target_indices.size != 2:
        raise ValueError("exactly two frequencies must belong to the target subspace")

    basis = eigenvectors[:, target_indices]
    projector = basis @ basis.T
    projector = 0.5 * (projector + projector.T)
    return np.vstack((frequencies, projector))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "K=np.diag([4.0,16.0,16.0,36.0]); omega0=4.0",
            "call": "analyze_unperturbed_subspace(K, omega0)",
            "gold_call": "_oracle_analyze_unperturbed_subspace(K, omega0)",
        },
        {
            "setup": "K=9.0*np.eye(2); omega0=3.0",
            "call": "analyze_unperturbed_subspace(K, omega0)",
            "gold_call": "_oracle_analyze_unperturbed_subspace(K, omega0)",
        },
        {
            "setup": "Q=np.array([[1.0,1.0,0.0],[1.0,-1.0,0.0],[0.0,0.0,np.sqrt(2.0)]])/np.sqrt(2.0); K=Q@np.diag([25.0,25.0,49.0])@Q.T; omega0=5.0",
            "call": "analyze_unperturbed_subspace(K, omega0)",
            "gold_call": "_oracle_analyze_unperturbed_subspace(K, omega0)",
        },
        {
            "setup": "K=np.diag([0.0,9.0,9.0]); omega0=3.0",
            "call": "analyze_unperturbed_subspace(K, omega0)",
            "gold_call": "_oracle_analyze_unperturbed_subspace(K, omega0)",
        },
    ]
