"""
Select the two full modes with greatest target-subspace weight.

## The projected first-order pair and the full spectrum are computed independently, so the

two full modes that evolve from the target eigenspace have to be identified rather than

assumed. Frequency adjacency is not a reliable selector, because a branch from outside

the target eigenspace can sit between them. The reliable test is overlap: with

$P_B=BB^\dagger$ the projector onto the span of the first-order displacement basis, each

full mode carries the normalized weight $p_j=(\epsilon_j^\dagger

P_B\epsilon_j)/(\epsilon_j^\dagger\epsilon_j)$, a Rayleigh quotient in $[0,1]$ that is

invariant under the eigenvector's phase. The two largest weights select the target pair.

Returns
-------
matched_modes : np.ndarray     Complex array with shape $(N + 2, 2)$. Row zero contains ascending target     frequencies, row one contains their projection weights, and the final     $N$ rows contain their normalized displacement vectors.  Raises ------ ValueError     If first_order_modes is not a finite complex array with shape     $(N + 1, 2)$; if its frequencies are not ascending, positive, and real     within atol=$10^{-9}$; if its displacement columns are not orthonormal     within atol=$10^{-8}$; if full_modes does not have shape $(N + 1, M)$ with $M$     at least two; if its entries are nonfinite; if its frequencies are not     positive and real within atol=$10^{-9}$; or if its displacement columns are     not normalized within atol=$10^{-8}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def match_target_modes(
    first_order_modes: "np.ndarray",
    full_modes: "np.ndarray",
) -> "np.ndarray":
    """Select the two full modes with greatest target-subspace weight.

    Parameters
    ----------
    first_order_modes : np.ndarray
        Complex packed table from compute_first_order_frequencies with shape
        $(N + 1, 2)$. Its first row contains two ascending positive
        first-order frequencies, and its remaining rows contain the associated
        orthonormal displacement basis.
    full_modes : np.ndarray
        Complex modal table with shape $(N + 1, M)$, $M$ at least two. Its first
        row contains positive real frequencies and the remaining rows contain
        normalized displacement vectors.

    Returns
    -------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$. Row zero contains ascending target
        frequencies, row one contains their projection weights, and the final
        $N$ rows contain their normalized displacement vectors.

    Raises
    ------
    ValueError
        If first_order_modes is not a finite complex array with shape
        $(N + 1, 2)$; if its frequencies are not ascending, positive, and real
        within atol=$10^{-9}$; if its displacement columns are not orthonormal
        within atol=$10^{-8}$; if full_modes does not have shape $(N + 1, M)$ with $M$
        at least two; if its entries are nonfinite; if its frequencies are not
        positive and real within atol=$10^{-9}$; or if its displacement columns are
        not normalized within atol=$10^{-8}$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_match_target_modes(
    first_order_modes: "np.ndarray",
    full_modes: "np.ndarray",
) -> "np.ndarray":
    """Select the two full modes with greatest target-subspace weight.

    Parameters
    ----------
    first_order_modes : np.ndarray
        Complex packed table from compute_first_order_frequencies with shape
        $(N + 1, 2)$. Its first row contains two ascending positive
        first-order frequencies, and its remaining rows contain the associated
        orthonormal displacement basis.
    full_modes : np.ndarray
        Complex modal table with shape $(N + 1, M)$, $M$ at least two. Its first
        row contains positive real frequencies and the remaining rows contain
        normalized displacement vectors.

    Returns
    -------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$. Row zero contains ascending target
        frequencies, row one contains their projection weights, and the final
        $N$ rows contain their normalized displacement vectors.

    Raises
    ------
    ValueError
        If first_order_modes is not a finite complex array with shape
        $(N + 1, 2)$; if its frequencies are not ascending, positive, and real
        within atol=$10^{-9}$; if its displacement columns are not orthonormal
        within atol=$10^{-8}$; if full_modes does not have shape $(N + 1, M)$ with $M$
        at least two; if its entries are nonfinite; if its frequencies are not
        positive and real within atol=$10^{-9}$; or if its displacement columns are
        not normalized within atol=$10^{-8}$.
    """
    try:
        first_order = np.asarray(first_order_modes, dtype=complex)
        modes = np.asarray(full_modes, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("first-order and full modes must be numeric arrays") from exc

    if first_order.ndim != 2 or first_order.shape[1] != 2 or first_order.shape[0] < 3:
        raise ValueError("first_order_modes must have shape (N+1, 2) with N >= 2")
    if not np.all(np.isfinite(first_order.real)) or not np.all(np.isfinite(first_order.imag)):
        raise ValueError("first_order_modes must be finite")
    n_coordinates = first_order.shape[0] - 1
    first_order_frequencies = first_order[0]
    if (
        np.any(np.abs(first_order_frequencies.imag) > 1.0e-9)
        or np.any(first_order_frequencies.real <= 0.0)
        or np.any(np.diff(first_order_frequencies.real) < -1.0e-12)
    ):
        raise ValueError("first-order frequencies must be ascending, positive, and real")
    first_order_basis = first_order[1:]
    gram = first_order_basis.conj().T @ first_order_basis
    if not np.allclose(gram, np.eye(2), rtol=0.0, atol=1.0e-8):
        raise ValueError("first-order displacement columns must be orthonormal")
    projector = first_order_basis @ first_order_basis.conj().T
    if modes.ndim != 2 or modes.shape[0] != n_coordinates + 1 or modes.shape[1] < 2:
        raise ValueError("full_modes must have shape (N+1, M) with M >= 2")
    if not np.all(np.isfinite(modes.real)) or not np.all(np.isfinite(modes.imag)):
        raise ValueError("full_modes must be finite")

    frequencies = modes[0]
    if np.any(np.abs(frequencies.imag) > 1.0e-9) or np.any(frequencies.real <= 0.0):
        raise ValueError("full-mode frequencies must be positive and real")
    displacements = modes[1:]
    norms = np.linalg.norm(displacements, axis=0)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("full-mode displacement columns must be normalized")

    weights = np.real(
        np.sum(displacements.conj() * (projector @ displacements), axis=0)
    )
    weights = np.clip(weights, 0.0, 1.0)
    selected = np.argsort(-weights, kind="stable")[:2]
    selected = selected[np.argsort(frequencies.real[selected], kind="stable")]
    return np.vstack(
        (
            frequencies.real[selected].astype(complex),
            weights[selected].astype(complex),
            displacements[:, selected],
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    summary = '''\ndef summarize_matched_modes(table):\n    table = np.asarray(table, dtype=complex)\n    frequencies = table[0]\n    weights = table[1]\n    displacements = table[2:]\n    norms = np.linalg.norm(displacements, axis=0)\n    normalized = displacements / norms\n    pieces = [frequencies.real.astype(float), frequencies.imag.astype(float), weights.real.astype(float), weights.imag.astype(float), norms]\n    for column in range(normalized.shape[1]):\n        projector = np.outer(normalized[:, column], normalized[:, column].conj())\n        pieces.extend((projector.real.ravel(), projector.imag.ravel()))\n    return np.concatenate(pieces)\n'''
    return [
        {
            "setup": summary + "B=np.eye(4,dtype=complex)[:,1:3]; first=np.vstack((np.array([3.9,4.1],dtype=complex),B)); D=np.eye(4,dtype=complex); modes=np.vstack((np.array([2.0,3.9,4.1,7.0],dtype=complex),D))",
            "call": "summarize_matched_modes(match_target_modes(first, modes))",
            "gold_call": "summarize_matched_modes(_oracle_match_target_modes(first, modes))",
        },
        {
            "setup": summary + "first=np.vstack((np.array([2.9,3.1],dtype=complex),np.eye(2,dtype=complex))); modes=np.vstack((np.array([2.9,3.1],dtype=complex),np.eye(2,dtype=complex)))",
            "call": "summarize_matched_modes(match_target_modes(first, modes))",
            "gold_call": "summarize_matched_modes(_oracle_match_target_modes(first, modes))",
        },
        {
            "setup": summary + "B=np.array([[0.0,0.0],[1.0,1.0],[1.0,-1.0]],dtype=complex)/np.sqrt(2.0); first=np.vstack((np.array([4.9,5.1],dtype=complex),B)); D=np.array([[np.sqrt(0.8),np.sqrt(0.1),np.sqrt(0.1)],[np.sqrt(0.1),np.sqrt(0.8),np.sqrt(0.1)],[np.sqrt(0.1),np.sqrt(0.1),np.sqrt(0.8)]],dtype=complex); modes=np.vstack((np.array([4.8,5.0,5.2],dtype=complex),D))",
            "call": "summarize_matched_modes(match_target_modes(first, modes))",
            "gold_call": "summarize_matched_modes(_oracle_match_target_modes(first, modes))",
        },
        {
            "setup": summary + "B=np.diag([1.0j,-1.0j]); first=np.vstack((np.array([2.0,2.0],dtype=complex),B)); modes=np.vstack((np.array([1.9,2.1],dtype=complex),np.diag([1.0j,-1.0j])))",
            "call": "summarize_matched_modes(match_target_modes(first, modes))",
            "gold_call": "summarize_matched_modes(_oracle_match_target_modes(first, modes))",
        },
    ]
