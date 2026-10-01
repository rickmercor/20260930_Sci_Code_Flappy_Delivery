"""
Track the topmost Kramers subspace continuously along an in-plane momentum path.

At zero momentum the target is the lowest Kramers pair. At every later point,
diagonalise the full confined-hole Hamiltonian, group the ordered eigenvectors into
adjacent Kramers pairs, and continue the previous two-dimensional subspace by choosing
the candidate projector that maximises Tr(P_previous P_candidate). This prevents a
band-order swap from silently changing the physical branch. For every path point return
the pair-mean energy, its numerical splitting, the two eigenvalues of the green
projector restricted to the tracked subspace, and the zero-based index of the selected
pair in the energy ordering.

Returns
-------
numpy.ndarray of shape (len(k_values), 5)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tracked_topmost_subband_path(width: float, k_values: "np.ndarray", direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    '''Gauge-invariant observables for the continuously tracked topmost Kramers pair.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    k_values : numpy.ndarray
        One-dimensional finite array of wave-vector magnitudes in 1/nm. It must
        start at exactly zero and then increase strictly.
    direction : tuple
        Two finite Cartesian components defining a nonzero in-plane direction.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.

    Returns
    -------
    numpy.ndarray
        Float array with one row per path point and columns
        [mean_energy, pair_splitting, green_low, green_high, pair_index].
        Energies are in meV; green contents and pair_index are dimensionless.

    Raises
    ------
    ValueError
        If the path or another argument is invalid.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_tracked_topmost_subband_path(width: float, k_values: "np.ndarray", direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    try:
        path = np.asarray(k_values, dtype=float)
    except Exception as exc:
        raise ValueError("k_values must be a finite one-dimensional path") from exc
    if path.ndim != 1 or path.size == 0 or not np.all(np.isfinite(path)):
        raise ValueError("k_values must be a finite one-dimensional path")
    if path[0] != 0.0 or np.any(path < 0.0) or np.any(np.diff(path) <= 0.0):
        raise ValueError("k_values must start at zero and then increase strictly")

    first_ham = _oracle_hole_hamiltonian(width, float(path[0]), direction, num_modes, params)
    splitting = float(params["delta"])
    levels, states = np.linalg.eigh(_oracle_spin_orbit_hamiltonian(splitting))
    upper = states[:, levels > 0.5 * splitting]
    internal_green = upper @ upper.conj().T
    full_green = np.kron(np.eye(int(num_modes)), internal_green)

    rows = []
    previous_projector = None
    for point, kk in enumerate(path):
        ham = first_ham if point == 0 else _oracle_hole_hamiltonian(
            width, float(kk), direction, num_modes, params,
        )
        energies, vectors = np.linalg.eigh(ham)
        pair_count = energies.size // 2

        if previous_projector is None:
            selected = 0
        else:
            scores = np.empty(pair_count, dtype=float)
            for pair_index in range(pair_count):
                candidate = vectors[:, 2 * pair_index:2 * pair_index + 2]
                candidate_projector = candidate @ candidate.conj().T
                scores[pair_index] = float(np.real(np.trace(
                    previous_projector @ candidate_projector
                )))
            selected = int(np.argmax(scores))

        pair = vectors[:, 2 * selected:2 * selected + 2]
        previous_projector = pair @ pair.conj().T
        restricted = pair.conj().T @ full_green @ pair
        restricted = 0.5 * (restricted + restricted.conj().T)
        green = np.linalg.eigvalsh(restricted)
        lo = 2 * selected
        rows.append([
            0.5 * float(energies[lo] + energies[lo + 1]),
            float(abs(energies[lo + 1] - energies[lo])),
            float(np.real(green[0])),
            float(np.real(green[1])),
            float(selected),
        ])
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    cu2o = "import numpy as np\nCU2O = {'gamma1': 1.76, 'gamma2': 0.7532, 'gamma3': -0.3668, 'eta1': -0.020, 'eta2': -0.0037, 'eta3': -0.0337, 'delta': 131.0}"
    tracking = "import numpy as np\nTRACK = {'gamma1': 1.9, 'gamma2': -0.5, 'gamma3': -1.2, 'eta1': 0.01, 'eta2': -0.125, 'eta3': -0.04, 'delta': 78.0}"
    return [
        {
            "setup": cu2o,
            "call": "tracked_topmost_subband_path(6.4, np.array([0.0, 0.20, 0.35]), (1.0, 1.0), 6, CU2O)",
            "gold_call": "_oracle_tracked_topmost_subband_path(6.4, np.array([0.0, 0.20, 0.35]), (1.0, 1.0), 6, CU2O)",
        },
        {
            "setup": cu2o,
            "call": "tracked_topmost_subband_path(12.0, np.array([0.0, 0.05, 0.20]), (2.0, 1.0), 3, CU2O)",
            "gold_call": "_oracle_tracked_topmost_subband_path(12.0, np.array([0.0, 0.05, 0.20]), (2.0, 1.0), 3, CU2O)",
        },
        {
            "setup": tracking,
            "call": "tracked_topmost_subband_path(9.4, np.linspace(0.0, 2.1, 61), (0.2, -1.0), 4, TRACK)",
            "gold_call": "_oracle_tracked_topmost_subband_path(9.4, np.linspace(0.0, 2.1, 61), (0.2, -1.0), 4, TRACK)",
        },
    ]
