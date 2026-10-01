"""
Encode thermodynamic eligibility for every binary record and metabolite state.

candidate_pattern_mask contains one binary row per candidate and one column per reaction. driving_forces_kj contains one row per metabolite state and the same reaction columns.

Apply the matching source's thermodynamic meaning of a binary reaction state. A candidate/state pair is eligible only when every reaction that remains thermodynamically binding under that source interpretation satisfies the inclusive minimum_driving_force threshold at that state.

Return numerical 1.0 for an eligible candidate/state pair and 0.0 otherwise. Preserve candidate and state order.

candidate_pattern_mask must be a finite, non-empty, two-dimensional array containing only 0.0 and 1.0. driving_forces_kj must be finite, non-empty, two-dimensional, and reaction-aligned. minimum_driving_force must be a finite real scalar. Raise ValueError if the public contract is violated.

Returns
-------
2D NumPy float array of shape (n_candidates, n_states), containing only 0.0 and 1.0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def classify_pattern_state_eligibility(candidate_pattern_mask: np.ndarray, driving_forces_kj: np.ndarray, minimum_driving_force: float) -> np.ndarray:
    """Encode pattern-state thermodynamic eligibility."""
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_classify_pattern_state_eligibility(candidate_pattern_mask, driving_forces_kj, minimum_driving_force):
    import numpy as np
    pattern = np.asarray(candidate_pattern_mask, dtype=float)
    forces = np.asarray(driving_forces_kj, dtype=float)
    if pattern.ndim != 2 or min(pattern.shape) < 1:
        raise ValueError('candidate_pattern_mask must be a non-empty two-dimensional array')
    if not np.all(np.isfinite(pattern)) or not np.all((pattern == 0.0) | (pattern == 1.0)):
        raise ValueError('candidate_pattern_mask must contain only numerical 0.0 and 1.0 values')
    if forces.ndim != 2 or forces.shape[0] < 1 or forces.shape[1] != pattern.shape[1]:
        raise ValueError('driving_forces_kj must align with state rows and pattern reaction columns')
    if not np.all(np.isfinite(forces)):
        raise ValueError('driving_forces_kj must contain only finite values')
    try:
        floor = float(minimum_driving_force)
    except (TypeError, ValueError) as exc:
        raise ValueError('minimum_driving_force must be a real scalar') from exc
    if not np.isfinite(floor):
        raise ValueError('minimum_driving_force must be finite')
    meets = forces >= floor
    eligible = np.all((pattern[:, None, :] == 0.0) | meets[None, :, :], axis=2)
    return eligible.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '''import numpy as np
candidate_pattern_mask = np.array(
    [
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
    ],
    dtype=float,
)
driving_forces_kj = np.array(
    [
        [1.0, 1.5, 0.1, 1.2],
        [0.5, 2.0, 4.0, 1.3],
        [2.0, 0.2, 3.0, 0.8],
    ],
    dtype=float,
)
minimum_driving_force = 0.8
''',
            'call': 'classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
            'gold_call': '_oracle_classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
        },
        {
            'setup': '''import numpy as np
candidate_pattern_mask = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=float,
)
driving_forces_kj = np.array(
    [
        [0.2, 1.25, 0.1],
        [2.0, 0.4, 1.25],
        [3.0, 2.0, 3.0],
    ],
    dtype=float,
)
minimum_driving_force = 1.25
''',
            'call': 'classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
            'gold_call': '_oracle_classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
        },
        {
            'setup': '''import numpy as np
candidate_pattern_mask = np.array(
    [
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ],
    dtype=float,
)
driving_forces_kj = np.array(
    [
        [1.1, 1.5, 0.7, 2.0],
        [0.7, 2.0, 1.4, 0.6],
        [2.0, 0.6, 1.2, 1.7],
    ],
    dtype=float,
)

co = np.array([2, 0, 1], dtype=int)
so = np.array([1, 2, 0], dtype=int)
rp = np.array([3, 1, 0, 2], dtype=int)

candidate_pattern_mask = candidate_pattern_mask[co][:, rp]
driving_forces_kj = driving_forces_kj[so][:, rp]

minimum_driving_force = 1.0
''',
            'call': 'classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
            'gold_call': '_oracle_classify_pattern_state_eligibility(candidate_pattern_mask.copy(), driving_forces_kj.copy(), minimum_driving_force)',
        },
        {
            'setup': '''import numpy as np
candidate_pattern_mask = np.array(
    [[1.0, 2.0]],
    dtype=float,
)
driving_forces_kj = np.array(
    [[2.0, 3.0]],
    dtype=float,
)
minimum_driving_force = 1.0

def candidate_wrapper():
    try:
        classify_pattern_state_eligibility(
            candidate_pattern_mask,
            driving_forces_kj,
            minimum_driving_force,
        )
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_classify_pattern_state_eligibility(
            candidate_pattern_mask,
            driving_forces_kj,
            minimum_driving_force,
        )
    except ValueError:
        return 1.0
    return 0.0
''',
            'call': 'candidate_wrapper()',
            'gold_call': 'gold_wrapper()',
        },
    ]
