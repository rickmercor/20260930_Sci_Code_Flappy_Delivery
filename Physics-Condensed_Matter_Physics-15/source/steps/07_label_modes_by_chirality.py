"""
Compute chiralities and order the two modes as plus and minus branches.

## A circularly polarized mode carries nonzero phonon angular momentum along the rotation

axis. For a basis ordered in Cartesian $(x,y)$ pairs, the $z$ component is measured by

the direct sum of the $2$-by-$2$ block $j_z$ over those pairs, and the normalized

expectation $\chi_j=(\epsilon_j^\dagger J_z\epsilon_j)/(\epsilon_j^\dagger\epsilon_j)$

is again a phase-invariant Rayleigh quotient bounded by $\pm1$. A degenerate pair split

by the curvature carries opposite chirality, so one mode is labelled the ${+}$ branch

and the other the ${-}$ branch. That labelling, and not the frequency order, is what

indexes the branch-dependent spectroscopic parameters downstream.

Returns
-------
branch_data : np.ndarray     Real array with shape $(2, 2)$. Row zero contains frequencies and row one     contains chiralities; columns are ordered [plus, minus].  Raises ------ ValueError     If matched_modes is not a finite numeric array with shape $(N + 2, 2)$,     $N$ at least two; if its frequencies are not positive and real within     atol=$10^{-9}$; if its weights are not real and in $[0,1]$ within atol=$10^{-9}$;     if its displacement vectors are not normalized within atol=$10^{-8}$; if     xy_pairs is not an integer array that covers every coordinate exactly     once; if a chirality expectation has an imaginary component larger     than $10^{-9}$; or if the two chiralities are not respectively separable     into one strictly positive and one strictly negative value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def label_modes_by_chirality(
    matched_modes: "np.ndarray",
    xy_pairs: "np.ndarray",
) -> "np.ndarray":
    """Compute chiralities and order the two modes as plus and minus branches.

    Parameters
    ----------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$ from match_target_modes. Row zero
        contains frequencies, row one contains projection weights, and the
        remaining $N$ rows contain normalized displacement vectors.
    xy_pairs : np.ndarray
        Integer array with shape $(N/2, 2)$ that partitions all coordinate
        indices into ordered Cartesian (x, y) pairs.

    Returns
    -------
    branch_data : np.ndarray
        Real array with shape $(2, 2)$. Row zero contains frequencies and row one
        contains chiralities; columns are ordered [plus, minus].

    Raises
    ------
    ValueError
        If matched_modes is not a finite numeric array with shape $(N + 2, 2)$,
        $N$ at least two; if its frequencies are not positive and real within
        atol=$10^{-9}$; if its weights are not real and in $[0,1]$ within atol=$10^{-9}$;
        if its displacement vectors are not normalized within atol=$10^{-8}$; if
        xy_pairs is not an integer array that covers every coordinate exactly
        once; if a chirality expectation has an imaginary component larger
        than $10^{-9}$; or if the two chiralities are not respectively separable
        into one strictly positive and one strictly negative value.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_label_modes_by_chirality(
    matched_modes: "np.ndarray",
    xy_pairs: "np.ndarray",
) -> "np.ndarray":
    """Compute chiralities and order the two modes as plus and minus branches.

    Parameters
    ----------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$ from match_target_modes. Row zero
        contains frequencies, row one contains projection weights, and the
        remaining $N$ rows contain normalized displacement vectors.
    xy_pairs : np.ndarray
        Integer array with shape $(N/2, 2)$ that partitions all coordinate
        indices into ordered Cartesian (x, y) pairs.

    Returns
    -------
    branch_data : np.ndarray
        Real array with shape $(2, 2)$. Row zero contains frequencies and row one
        contains chiralities; columns are ordered [plus, minus].

    Raises
    ------
    ValueError
        If matched_modes is not a finite numeric array with shape $(N + 2, 2)$,
        $N$ at least two; if its frequencies are not positive and real within
        atol=$10^{-9}$; if its weights are not real and in $[0,1]$ within atol=$10^{-9}$;
        if its displacement vectors are not normalized within atol=$10^{-8}$; if
        xy_pairs is not an integer array that covers every coordinate exactly
        once; if a chirality expectation has an imaginary component larger
        than $10^{-9}$; or if the two chiralities are not respectively separable
        into one strictly positive and one strictly negative value.
    """
    try:
        modes = np.asarray(matched_modes, dtype=complex)
        pair_raw = np.asarray(xy_pairs)
    except (TypeError, ValueError) as exc:
        raise ValueError("matched_modes and xy_pairs must be numeric arrays") from exc

    if modes.ndim != 2 or modes.shape[1] != 2 or modes.shape[0] < 4:
        raise ValueError("matched_modes must have shape (N+2, 2) with N >= 2")
    if not np.all(np.isfinite(modes.real)) or not np.all(np.isfinite(modes.imag)):
        raise ValueError("matched_modes must be finite")
    n_coordinates = modes.shape[0] - 2
    frequencies = modes[0]
    weights = modes[1]
    displacements = modes[2:]
    if np.any(np.abs(frequencies.imag) > 1.0e-9) or np.any(frequencies.real <= 0.0):
        raise ValueError("frequencies must be positive and real")
    if np.any(np.abs(weights.imag) > 1.0e-9):
        raise ValueError("projection weights must be real")
    if np.any(weights.real < -1.0e-9) or np.any(weights.real > 1.0 + 1.0e-9):
        raise ValueError("projection weights must lie in [0,1]")
    if not np.allclose(
        np.linalg.norm(displacements, axis=0),
        1.0,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("displacement vectors must be normalized")

    if pair_raw.ndim != 2 or pair_raw.shape[1] != 2:
        raise ValueError("xy_pairs must have shape (N/2, 2)")
    try:
        pairs = np.asarray(pair_raw, dtype=int)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("xy_pairs must contain integer indices") from exc
    if not np.array_equal(pair_raw, pairs):
        raise ValueError("xy_pairs must contain integer indices")
    if pairs.size != n_coordinates:
        raise ValueError("xy_pairs must cover every coordinate exactly once")
    if not np.array_equal(np.sort(pairs.ravel()), np.arange(n_coordinates)):
        raise ValueError("xy_pairs must cover every coordinate exactly once")

    angular_momentum = np.zeros((n_coordinates, n_coordinates), dtype=complex)
    for x_index, y_index in pairs:
        angular_momentum[x_index, y_index] = -1j
        angular_momentum[y_index, x_index] = 1j

    chiralities = np.empty(2, dtype=float)
    for mode_index in range(2):
        displacement = displacements[:, mode_index]
        expectation = displacement.conj() @ angular_momentum @ displacement
        if abs(expectation.imag) > 1.0e-9:
            raise ValueError("chirality expectation must be real")
        chiralities[mode_index] = float(expectation.real)

    plus_index = int(np.argmax(chiralities))
    minus_index = int(np.argmin(chiralities))
    if chiralities[plus_index] <= 0.0 or chiralities[minus_index] >= 0.0:
        raise ValueError("one mode must have positive and one negative chirality")
    order = np.array([plus_index, minus_index], dtype=int)
    return np.vstack((frequencies.real[order], chiralities[order])).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "plus=np.array([1.0,1.0j])/np.sqrt(2.0); minus=np.array([1.0,-1.0j])/np.sqrt(2.0); matched=np.vstack((np.array([4.2,3.8],dtype=complex),np.array([0.99,0.98],dtype=complex),np.column_stack((plus,minus)))); pairs=np.array([[0,1]])",
            "call": "label_modes_by_chirality(matched, pairs)",
            "gold_call": "_oracle_label_modes_by_chirality(matched, pairs)",
        },
        {
            "setup": "minus=np.array([1.0,-1.0j])/np.sqrt(2.0); plus=np.array([1.0,1.0j])/np.sqrt(2.0); matched=np.vstack((np.array([2.9,3.1],dtype=complex),np.ones(2,dtype=complex),np.column_stack((minus,plus)))); pairs=np.array([[0,1]])",
            "call": "label_modes_by_chirality(matched, pairs)",
            "gold_call": "_oracle_label_modes_by_chirality(matched, pairs)",
        },
        {
            "setup": "plus=np.array([1.0,1.0j,1.0,1.0j])/2.0; minus=np.array([1.0,-1.0j,1.0,-1.0j])/2.0; matched=np.vstack((np.array([5.0,5.2],dtype=complex),np.array([0.9,0.8],dtype=complex),np.column_stack((plus,minus)))); pairs=np.array([[0,1],[2,3]])",
            "call": "label_modes_by_chirality(matched, pairs)",
            "gold_call": "_oracle_label_modes_by_chirality(matched, pairs)",
        },
        {
            "setup": "plus=np.array([1.0,1.0j,1.0,1.0j])/2.0; minus=np.array([1.0,-1.0j,1.0,-1.0j])/2.0; matched=np.vstack((np.array([4.0,4.1],dtype=complex),np.ones(2,dtype=complex),np.column_stack((plus,minus)))); pairs=np.array([[0,3],[2,1]])",
            "call": "label_modes_by_chirality(matched, pairs)",
            "gold_call": "_oracle_label_modes_by_chirality(matched, pairs)",
        },
    ]
