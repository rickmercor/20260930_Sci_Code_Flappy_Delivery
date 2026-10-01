"""
Derive one five-factor state for each supplied member pair.

factor_states is a finite positive array of shape (n, 5), containing one aligned five-factor state per profile row.



member_pairs is an integer array of shape (p, 2), containing two profile-row indices for each requested pair.



anchor_index identifies one shared anchor profile row.



Apply the pair-state relation defined by the matching treatment.



Return one five-factor state per requested pair, preserving the supplied pair order and factor-column order.

Returns
-------
pair_factor_states : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def derive_pair_factor_states(
    factor_states: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Derive one five-factor state for each supplied member pair.

    Parameters
    ----------
    factor_states
        Finite positive array of shape (n, 5), containing one aligned
        five-factor state per profile row.
    member_pairs
        Integer array of shape (p, 2), containing two profile-row
        indices for each requested pair.
    anchor_index
        Integer index of the shared anchor profile row.

    Returns
    -------
    np.ndarray
        Finite positive array of shape (p, 5), preserving pair and
        factor-column order.
    """
    pair_factor_states = np.empty(
        (
            member_pairs.shape[0],
            5,
        ),
        dtype=float,
    )
    return pair_factor_states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_derive_pair_factor_states(
    factor_states: np.ndarray,
    member_pairs: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Reference implementation for derive_pair_factor_states."""
    import numpy as np

    states = np.asarray(
        factor_states,
        dtype=float,
    )

    member_pairs_raw = np.asarray(
        member_pairs,
    )

    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] != 5
    ):
        raise ValueError(
            "factor_states must have shape (n, 5) with n >= 1"
        )

    if not np.all(
        np.isfinite(states)
    ):
        raise ValueError(
            "factor_states must contain only finite values"
        )

    if np.any(
        states <= 0.0
    ):
        raise ValueError(
            "factor_states must be strictly positive"
        )

    if (
        member_pairs_raw.ndim != 2
        or member_pairs_raw.shape[0] < 1
        or member_pairs_raw.shape[1] != 2
        or not np.issubdtype(
            member_pairs_raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "member_pairs must be a non-empty integer array "
            "of shape (p, 2)"
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

    members = member_pairs_raw.astype(
        int,
        copy=False,
    )

    n_profiles = states.shape[0]

    if (
        anchor < 0
        or anchor >= n_profiles
    ):
        raise ValueError(
            "anchor_index is out of range"
        )

    if (
        np.any(members < 0)
        or np.any(members >= n_profiles)
    ):
        raise ValueError(
            "member_pairs contains an out-of-range index"
        )

    log_states = np.log(
        states
    )

    log_pair_states = (
        log_states[members[:, 0]]
        + log_states[members[:, 1]]
        - log_states[anchor]
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        pair_factor_states = np.exp(
            log_pair_states
        )

    if (
        not np.all(
            np.isfinite(pair_factor_states)
        )
        or np.any(
            pair_factor_states <= 0.0
        )
    ):
        raise ValueError(
            "derived pair factor states must be finite "
            "and strictly positive"
        )

    return pair_factor_states.astype(
        float,
        copy=False,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for derive_pair_factor_states."""
    return [
        {
            "setup": """import numpy as np
factor_states = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 5.0, 7.0, 11.0],
        [0.5, 2.0, 4.0, 8.0, 16.0],
        [4.0, 0.25, 2.0, 0.5, 8.0],
    ],
    dtype=float,
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
            "call": "derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
            "gold_call": "_oracle_derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
factor_states = np.array(
    [
        [1.2, 0.8, 2.5, 1.1, 3.4],
        [0.9, 1.7, 0.6, 2.2, 1.4],
        [2.1, 0.5, 1.8, 0.7, 4.3],
        [0.4, 3.2, 1.3, 2.8, 0.9],
        [1.6, 1.1, 0.8, 3.5, 2.4],
    ],
    dtype=float,
)
member_pairs = np.array(
    [
        [0, 2],
        [3, 4],
        [2, 4],
    ],
    dtype=int,
)
anchor_index = 1
""",
            "call": "derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
            "gold_call": "_oracle_derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
factor_states = np.array(
    [
        [1.0e-8, 2.0e-7, 3.0e-6, 4.0e-5, 5.0e-4],
        [2.0e-8, 5.0e-7, 7.0e-6, 9.0e-5, 1.1e-3],
        [3.0e-8, 4.0e-7, 8.0e-6, 6.0e-5, 9.0e-4],
    ],
    dtype=float,
)
member_pairs = np.array(
    [
        [1, 2],
    ],
    dtype=int,
)
anchor_index = 0
""",
            "call": "derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
            "gold_call": "_oracle_derive_pair_factor_states(factor_states, member_pairs, anchor_index)",
        },
        {
            "setup": """import numpy as np
factor_states = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
    ],
    dtype=float,
)
member_pairs = np.array(
    [
        [0, 2],
    ],
    dtype=int,
)
anchor_index = 0

def run_model():
    try:
        derive_pair_factor_states(
            factor_states,
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
        _oracle_derive_pair_factor_states(
            factor_states,
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
