"""
Apply the complete panel analysis and return the selected scalar.

profile_values is a finite array of shape (n, 6), containing one aligned six-column profile per row.



auxiliary_values is a finite positive one-dimensional array aligned with the profile rows.



pair_rows identifies the profile row associated with each candidate pair.



member_pairs contains the two constituent profile-row indices for each candidate pair.



candidate_ids contains one unique integer identifier per candidate pair.



anchor_index identifies the shared anchor profile row.



Apply the complete source-defined analysis matching the supplied inputs and return one finite positive dimensionless scalar.

Returns
-------
selected_scalar : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def resolve_panel_scalar(
    profile_values: np.ndarray,
    auxiliary_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    candidate_ids: np.ndarray,
    anchor_index: int,
    temperature: float,
    gas_constant: float,
) -> float:
    """Apply the complete panel analysis and return one scalar.

    Parameters
    ----------
    profile_values
        Finite array of shape (n, 6), containing one aligned numerical
        profile per row.
    auxiliary_values
        Finite positive one-dimensional array aligned with profile rows.
    pair_rows
        Integer array of shape (p,), containing the profile-row index
        associated with each candidate pair.
    member_pairs
        Integer array of shape (p, 2), containing two constituent
        profile-row indices per candidate pair.
    candidate_ids
        Integer array of shape (p,), containing unique candidate IDs.
    anchor_index
        Integer index of the shared anchor profile row.
    temperature
        Finite positive absolute temperature.
    gas_constant
        Finite positive gas constant in compatible energy units.

    Returns
    -------
    float
        Finite positive dimensionless selected scalar.
    """
    selected_scalar = 0.0
    return selected_scalar

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_panel_scalar(
    profile_values: np.ndarray,
    auxiliary_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    candidate_ids: np.ndarray,
    anchor_index: int,
    temperature: float,
    gas_constant: float,
) -> float:
    """Reference implementation for resolve_panel_scalar."""
    import numpy as np

    profiles = np.asarray(
        profile_values,
        dtype=float,
    )

    auxiliary = np.asarray(
        auxiliary_values,
        dtype=float,
    )

    pair_rows_raw = np.asarray(
        pair_rows,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    candidate_ids_raw = np.asarray(
        candidate_ids,
    )

    if (
        profiles.ndim != 2
        or profiles.shape[0] < 1
        or profiles.shape[1] != 6
    ):
        raise ValueError(
            "profile_values must have shape (n, 6) with n >= 1"
        )

    if not np.all(
        np.isfinite(profiles)
    ):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if auxiliary.shape != (
        profiles.shape[0],
    ):
        raise ValueError(
            "auxiliary_values must align with profile rows"
        )

    if (
        not np.all(
            np.isfinite(auxiliary)
        )
        or np.any(
            auxiliary <= 0.0
        )
    ):
        raise ValueError(
            "auxiliary_values must be finite and strictly positive"
        )

    if (
        pair_rows_raw.ndim != 1
        or pair_rows_raw.size < 1
        or not np.issubdtype(
            pair_rows_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "pair_rows must be a non-empty integer array"
        )

    if (
        member_pairs_raw.ndim != 2
        or member_pairs_raw.shape
        != (
            pair_rows_raw.size,
            2,
        )
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be an integer array of shape (p, 2)"
        )

    if (
        candidate_ids_raw.ndim != 1
        or candidate_ids_raw.shape
        != pair_rows_raw.shape
        or not np.issubdtype(
            candidate_ids_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "candidate_ids must be an integer array aligned "
            "with pair_rows"
        )

    if isinstance(
        anchor_index,
        (bool, np.bool_),
    ) or not isinstance(
        anchor_index,
        (int, np.integer),
    ):
        raise ValueError(
            "anchor_index must be an integer"
        )

    if isinstance(
        temperature,
        (bool, np.bool_),
    ) or isinstance(
        gas_constant,
        (bool, np.bool_),
    ):
        raise ValueError(
            "temperature and gas_constant must be real scalars"
        )

    try:
        temperature_value = float(
            temperature
        )

        gas_constant_value = float(
            gas_constant
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "temperature and gas_constant must be real scalars"
        ) from exc

    if (
        not np.isfinite(
            temperature_value
        )
        or temperature_value <= 0.0
        or not np.isfinite(
            gas_constant_value
        )
        or gas_constant_value <= 0.0
    ):
        raise ValueError(
            "temperature and gas_constant must be finite "
            "and strictly positive"
        )

    pair_indices = pair_rows_raw.astype(
        int,
        copy=False,
    )

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    ids = candidate_ids_raw.astype(
        int,
        copy=False,
    )

    anchor = int(
        anchor_index
    )

    n_profiles = profiles.shape[0]

    if (
        anchor < 0
        or anchor >= n_profiles
        or np.any(
            pair_indices < 0
        )
        or np.any(
            pair_indices >= n_profiles
        )
        or np.any(
            members < 0
        )
        or np.any(
            members >= n_profiles
        )
    ):
        raise ValueError(
            "one or more supplied profile indices are out of range"
        )

    if np.unique(
        ids
    ).size != ids.size:
        raise ValueError(
            "candidate_ids must be unique"
        )

    difference_pairs = np.array(
        [
            [1, 0],
            [1, 2],
            [3, 2],
            [3, 4],
            [5, 4],
        ],
        dtype=int,
    )

    directed_differences = _oracle_compute_indexed_differences(
        profiles,
        difference_pairs,
    )

    thermal_scale = (
        temperature_value
        * gas_constant_value
    )

    factor_states = _oracle_transform_directed_differences(
        directed_differences,
        thermal_scale,
    )

    profile_observables = _oracle_evaluate_factor_observables(
        factor_states,
    )

    profile_efficiencies = (
        profile_observables[:, 2]
    )

    eligibility_values = _oracle_compute_pair_departure_factors(
        profile_efficiencies,
        pair_indices,
        members,
        anchor,
    )

    derived_pair_states = _oracle_derive_pair_factor_states(
        factor_states,
        members,
        anchor,
    )

    derived_pair_observables = _oracle_evaluate_factor_observables(
        derived_pair_states,
    )

    priority_values = _oracle_compute_aligned_ratio_scores(
        profile_efficiencies[
            pair_indices
        ],
        derived_pair_observables[:, 2],
    )

    selection = _oracle_select_bounded_maximum(
        ids,
        eligibility_values,
        priority_values,
        1.0 / 1.5,
        1.5,
    )

    selected_scalar = float(
        selection[1]
    )

    if (
        not np.isfinite(
            selected_scalar
        )
        or selected_scalar <= 0.0
    ):
        raise ValueError(
            "selected scalar must be finite and strictly positive"
        )

    return selected_scalar

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for resolve_panel_scalar."""
    return [
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.0000000000, 10.0000000000, -5.0000000000, 11.0000000000, -9.0000000000, 9.0000000000],
        [0.8300000000, 11.9117730727, -5.2661270964, 8.7068103221, -10.9851495158, 6.3916967451],
        [-1.3700000000, 9.3461192809, -5.7956579575, 10.5272075527, -7.8261877581, 12.2030104154],
        [2.1100000000, 11.3049159672, -3.2038735159, 11.5476724027, -7.5477898213, 9.8944362498],
        [-0.4600000000, 8.4387809600, -6.3473123473, 11.1848999947, -10.3959642184, 8.1206949136],
        [1.7200000000, 11.9624636336, -3.5080366739, 12.5114616080, -5.5151478567, 10.6540336374],
        [-2.0300000000, 7.8718690602, -5.1678034104, 9.1288952388, -11.4496572021, 8.6000289427],
        [0.5500000000, 8.4099096992, -7.0607395030, 8.7207120852, -9.1610270102, 10.1286843970],
        [-1.1100000000, 9.8315309035, -7.1664243653, 9.5877017195, -11.4036186831, 5.9528652218],
        [2.3700000000, 12.7338531870, -2.8203232121, 11.4156022457, -7.2007919226, 9.5852039446],
        [-0.7300000000, 11.3368170682, -4.3117623652, 11.0561027890, -9.5643420912, 13.2730253349],
        [1.4100000000, 11.8756931546, -3.8719795847, 9.2485756671, -6.4868849562, 12.1556268828],
        [-1.9200000000, 7.5016150991, -8.5752707863, 3.5026096063, -15.7571171562, 1.5831586988],
        [0.6400000000, 7.4792162521, -8.0107240745, 6.6221136419, -9.8910333832, 8.6125202046],
        [2.0800000000, 13.4933950559, -4.8511788676, 9.3067584217, -10.1854580908, 6.1471296217],
        [-0.3700000000, 10.3752611435, -7.0228451052, 7.0231611924, -12.5405883186, 4.2906736507],
        [1.1600000000, 9.9307303603, -4.3057569220, 7.7723979012, -9.9884592516, 6.2425605907],
        [-1.2500000000, 10.0747948114, -7.2395822250, 6.0091250605, -11.0820801038, 4.8589755723],
        [0.9200000000, 10.6248740716, -6.6107233996, 9.3501334511, -11.3158569905, 5.8352966435],
        [-2.1400000000, 7.9752499858, -8.5016458323, 4.8827173001, -11.8273703041, 3.2511205871],
        [2.6500000000, 12.2516842997, -4.7959300662, 12.1622588881, -5.5275032253, 9.1238157906],
        [-0.5800000000, 8.6766731174, -6.8830164796, 8.2431093452, -9.3024595368, 7.7309160515],
        [2.3100000000, 13.5568415240, -1.9629366677, 13.3547182886, -5.8891962795, 14.3075890228],
        [-1.0400000000, 7.0125613298, -8.9913863785, 6.2983635635, -8.7289276239, 8.5729511841],
        [0.2600000000, 6.5613864798, -5.4172108756, 11.6045074918, -7.9508077125, 6.3741664299],
        [1.5300000000, 14.1023630933, -3.1771939547, 13.8420553728, -4.1917752170, 15.0284832752],
        [-1.6800000000, 7.3698934261, -5.2918242561, 7.5889353449, -11.6656747007, 8.0210416472],
        [0.4700000000, 11.0903073716, -3.2625499578, 9.9863224604, -9.1377450211, 10.2119310618],
    ],
    dtype=float,
)

auxiliary_values = np.array(
    [
        1.00000000,
        1.24712453,
        0.01215118,
        12.25664524,
        0.30705613,
        1.60197070,
        0.43277457,
        13.11383445,
        0.71153984,
        4.07256218,
        0.00900619,
        1.06815148,
        19.24790659,
        148.95903215,
        0.64229130,
        1.47534142,
        60.35383301,
        0.64017392,
        11.86555005,
        4.53542080,
        2.05447213,
        13.06247953,
        0.11600204,
        13.34126623,
        0.60611801,
        0.01694306,
        4.87733424,
        1.74173583,
    ],
    dtype=float,
)

pair_rows = np.arange(
    10,
    28,
    dtype=int,
)

member_pairs = np.array(
    [
        [2, 6],
        [1, 9],
        [1, 3],
        [3, 7],
        [1, 8],
        [8, 9],
        [3, 9],
        [1, 5],
        [3, 8],
        [5, 9],
        [5, 8],
        [3, 5],
        [2, 3],
        [5, 7],
        [4, 5],
        [2, 5],
        [3, 6],
        [6, 9],
    ],
    dtype=int,
)

candidate_ids = np.arange(
    1,
    19,
    dtype=int,
)

anchor_index = 0
temperature = 298.0
gas_constant = 1.9872036e-3
""",
            "call": "resolve_panel_scalar(profile_values, auxiliary_values, pair_rows, member_pairs, candidate_ids, anchor_index, temperature, gas_constant)",
            "gold_call": "_oracle_resolve_panel_scalar(profile_values, auxiliary_values, pair_rows, member_pairs, candidate_ids, anchor_index, temperature, gas_constant)",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.0000000000, 10.0000000000, -5.0000000000, 11.0000000000, -9.0000000000, 9.0000000000],
        [2.1100000000, 11.3049159672, -3.2038735159, 11.5476724027, -7.5477898213, 9.8944362498],
        [0.5500000000, 8.4099096992, -7.0607395030, 8.7207120852, -9.1610270102, 10.1286843970],
        [-0.4600000000, 8.4387809600, -6.3473123473, 11.1848999947, -10.3959642184, 8.1206949136],
        [1.7200000000, 11.9624636336, -3.5080366739, 12.5114616080, -5.5151478567, 10.6540336374],
        [0.6400000000, 7.4792162521, -8.0107240745, 6.6221136419, -9.8910333832, 8.6125202046],
        [0.2600000000, 6.5613864798, -5.4172108756, 11.6045074918, -7.9508077125, 6.3741664299],
    ],
    dtype=float,
)

auxiliary_values = np.array(
    [
        1.00000000,
        12.25664524,
        13.11383445,
        0.30705613,
        1.60197070,
        148.95903215,
        0.60611801,
    ],
    dtype=float,
)

pair_rows = np.array(
    [5, 6],
    dtype=int,
)

member_pairs = np.array(
    [
        [1, 2],
        [3, 4],
    ],
    dtype=int,
)

candidate_ids = np.array(
    [4, 15],
    dtype=int,
)

anchor_index = 0
temperature = 298.0
gas_constant = 1.9872036e-3

permutation = np.array(
    [1, 0],
    dtype=int,
)

def run_model():
    baseline = resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    permuted = resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows[permutation],
        member_pairs[permutation],
        candidate_ids[permutation],
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - permuted
    )

def run_gold():
    baseline = _oracle_resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    permuted = _oracle_resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows[permutation],
        member_pairs[permutation],
        candidate_ids[permutation],
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - permuted
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.0000000000, 10.0000000000, -5.0000000000, 11.0000000000, -9.0000000000, 9.0000000000],
        [2.1100000000, 11.3049159672, -3.2038735159, 11.5476724027, -7.5477898213, 9.8944362498],
        [0.5500000000, 8.4099096992, -7.0607395030, 8.7207120852, -9.1610270102, 10.1286843970],
        [-0.4600000000, 8.4387809600, -6.3473123473, 11.1848999947, -10.3959642184, 8.1206949136],
        [1.7200000000, 11.9624636336, -3.5080366739, 12.5114616080, -5.5151478567, 10.6540336374],
        [0.6400000000, 7.4792162521, -8.0107240745, 6.6221136419, -9.8910333832, 8.6125202046],
        [0.2600000000, 6.5613864798, -5.4172108756, 11.6045074918, -7.9508077125, 6.3741664299],
    ],
    dtype=float,
)

auxiliary_values = np.array(
    [
        1.00000000,
        12.25664524,
        13.11383445,
        0.30705613,
        1.60197070,
        148.95903215,
        0.60611801,
    ],
    dtype=float,
)

pair_rows = np.array(
    [5, 6],
    dtype=int,
)

member_pairs = np.array(
    [
        [1, 2],
        [3, 4],
    ],
    dtype=int,
)

candidate_ids = np.array(
    [4, 15],
    dtype=int,
)

anchor_index = 0
temperature = 298.0
gas_constant = 1.9872036e-3

def run_model():
    baseline = resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    reversed_members = resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs[:, ::-1],
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - reversed_members
    )

def run_gold():
    baseline = _oracle_resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    reversed_members = _oracle_resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs[:, ::-1],
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - reversed_members
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.0000000000, 10.0000000000, -5.0000000000, 11.0000000000, -9.0000000000, 9.0000000000],
        [2.1100000000, 11.3049159672, -3.2038735159, 11.5476724027, -7.5477898213, 9.8944362498],
        [0.5500000000, 8.4099096992, -7.0607395030, 8.7207120852, -9.1610270102, 10.1286843970],
        [-0.4600000000, 8.4387809600, -6.3473123473, 11.1848999947, -10.3959642184, 8.1206949136],
        [1.7200000000, 11.9624636336, -3.5080366739, 12.5114616080, -5.5151478567, 10.6540336374],
        [0.6400000000, 7.4792162521, -8.0107240745, 6.6221136419, -9.8910333832, 8.6125202046],
        [0.2600000000, 6.5613864798, -5.4172108756, 11.6045074918, -7.9508077125, 6.3741664299],
    ],
    dtype=float,
)

auxiliary_values = np.array(
    [
        1.00000000,
        12.25664524,
        13.11383445,
        0.30705613,
        1.60197070,
        148.95903215,
        0.60611801,
    ],
    dtype=float,
)

pair_rows = np.array(
    [5, 6],
    dtype=int,
)

member_pairs = np.array(
    [
        [1, 2],
        [3, 4],
    ],
    dtype=int,
)

candidate_ids = np.array(
    [4, 15],
    dtype=int,
)

anchor_index = 0
temperature = 298.0
gas_constant = 1.9872036e-3

row_shifts = np.array(
    [
        2.4,
        -1.3,
        4.2,
        0.7,
        -3.1,
        5.8,
        -2.6,
    ],
    dtype=float,
).reshape(
    7,
    1,
)

def run_model():
    baseline = resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    shifted = resolve_panel_scalar(
        profile_values + row_shifts,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - shifted
    )

def run_gold():
    baseline = _oracle_resolve_panel_scalar(
        profile_values,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    shifted = _oracle_resolve_panel_scalar(
        profile_values + row_shifts,
        auxiliary_values,
        pair_rows,
        member_pairs,
        candidate_ids,
        anchor_index,
        temperature,
        gas_constant,
    )

    return abs(
        baseline - shifted
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

profile_values = np.array(
    [
        [0.0, 10.0, -5.0, 11.0, -9.0, 9.0],
        [2.1, 11.3, -3.2, 11.5, -7.5, 9.9],
        [0.5, 8.4, -7.0, 8.7, -9.1, 10.1],
        [0.6, 7.4, -8.0, 6.6, -9.8, 8.6],
    ],
    dtype=float,
)

auxiliary_values = np.array(
    [1.0, 2.0, 3.0],
    dtype=float,
)

pair_rows = np.array(
    [3],
    dtype=int,
)

member_pairs = np.array(
    [
        [1, 2],
    ],
    dtype=int,
)

candidate_ids = np.array(
    [4],
    dtype=int,
)

anchor_index = 0
temperature = 298.0
gas_constant = 1.9872036e-3

def run_model():
    try:
        resolve_panel_scalar(
            profile_values,
            auxiliary_values,
            pair_rows,
            member_pairs,
            candidate_ids,
            anchor_index,
            temperature,
            gas_constant,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_resolve_panel_scalar(
            profile_values,
            auxiliary_values,
            pair_rows,
            member_pairs,
            candidate_ids,
            anchor_index,
            temperature,
            gas_constant,
        )
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
