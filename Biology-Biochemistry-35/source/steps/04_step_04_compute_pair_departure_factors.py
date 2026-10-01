"""
Compute one departure factor for each supplied pair from aligned positive profile values.

profile_values is a finite positive one-dimensional array containing one value per profile row.



pair_rows identifies the profile row associated with each requested pair.



member_pairs contains the two constituent profile-row indices for each requested pair.



anchor_index identifies one shared anchor profile row.



For pair p, calculate:



profile_values[pair_rows[p]] × profile_values[anchor_index]

------------------------------------------------------------

profile_values[member_pairs[p, 0]] × profile_values[member_pairs[p, 1]]



Return the factors in the supplied pair order.

Returns
-------
departure_factors : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_pair_departure_factors(
    profile_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Compute aligned departure factors for supplied profile pairs.

    Parameters
    ----------
    profile_values
        Finite positive one-dimensional array containing one value per
        profile row.
    pair_rows
        Integer array of shape (p,) containing the profile-row index
        associated with each requested pair.
    member_pairs
        Integer array of shape (p, 2) containing two constituent
        profile-row indices per requested pair.
    anchor_index
        Integer index of the shared anchor profile row.

    Returns
    -------
    np.ndarray
        Finite positive array of shape (p,), preserving pair order.
    """
    departure_factors = np.empty(
        pair_rows.shape[0],
        dtype=float,
    )
    return departure_factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_pair_departure_factors(
    profile_values: np.ndarray,
    pair_rows: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Reference implementation for compute_pair_departure_factors."""
    import numpy as np

    values = np.asarray(
        profile_values,
        dtype=float,
    )

    pair_rows_raw = np.asarray(
        pair_rows,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    if (
        values.ndim != 1
        or values.size < 1
    ):
        raise ValueError(
            "profile_values must be a non-empty one-dimensional array"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "profile_values must contain only finite values"
        )

    if np.any(
        values <= 0.0
    ):
        raise ValueError(
            "profile_values must be strictly positive"
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
        or member_pairs_raw.shape[1] != 2
        or member_pairs_raw.shape[0] != pair_rows_raw.size
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be an integer array of shape (p, 2)"
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

    anchor = int(
        anchor_index
    )

    pair_indices = pair_rows_raw.astype(
        int,
        copy=False,
    )

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    n_profiles = values.size

    if (
        anchor < 0
        or anchor >= n_profiles
    ):
        raise ValueError(
            "anchor_index is out of range"
        )

    if (
        np.any(pair_indices < 0)
        or np.any(pair_indices >= n_profiles)
        or np.any(members < 0)
        or np.any(members >= n_profiles)
    ):
        raise ValueError(
            "pair_rows or member_pairs contains an out-of-range index"
        )

    log_values = np.log(
        values
    )

    log_factors = (
        log_values[pair_indices]
        + log_values[anchor]
        - log_values[members[:, 0]]
        - log_values[members[:, 1]]
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        departure_factors = np.exp(
            log_factors
        )

    if (
        not np.all(
            np.isfinite(departure_factors)
        )
        or np.any(
            departure_factors <= 0.0
        )
    ):
        raise ValueError(
            "departure factors must be finite and strictly positive"
        )

    return departure_factors.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_pair_departure_factors."""
    return [
        {
            "setup": """import numpy as np
profile_values = np.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    dtype=float,
)
pair_rows = np.array(
    [4, 5],
    dtype=int,
)
member_pairs = np.array(
    [
        [1, 2],
        [2, 3],
    ],
    dtype=int,
)
anchor_index = 0
""",
            "call": "compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
            "gold_call": "_oracle_compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
profile_values = np.array(
    [0.8, 1.3, 2.1, 0.7, 3.4, 1.8, 5.2],
    dtype=float,
)
pair_rows = np.array(
    [5, 6, 4],
    dtype=int,
)
member_pairs = np.array(
    [
        [1, 2],
        [2, 3],
        [1, 3],
    ],
    dtype=int,
)
anchor_index = 0
""",
            "call": "compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
            "gold_call": "_oracle_compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
profile_values = np.array(
    [
        1.0e-9,
        2.0e-8,
        3.0e-7,
        4.0e-6,
        5.0e-5,
    ],
    dtype=float,
)
pair_rows = np.array(
    [3, 4],
    dtype=int,
)
member_pairs = np.array(
    [
        [1, 2],
        [2, 3],
    ],
    dtype=int,
)
anchor_index = 0
""",
            "call": "compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
            "gold_call": "_oracle_compute_pair_departure_factors(profile_values, pair_rows, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
profile_values = np.array(
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
anchor_index = 0

def run_model():
    try:
        compute_pair_departure_factors(
            profile_values,
            pair_rows,
            member_pairs,
            anchor_index,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_pair_departure_factors(
            profile_values,
            pair_rows,
            member_pairs,
            anchor_index,
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
