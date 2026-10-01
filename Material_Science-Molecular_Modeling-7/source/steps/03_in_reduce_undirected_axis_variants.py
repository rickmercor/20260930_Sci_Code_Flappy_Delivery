"""
Reduce the supplied undirected axis pairs separately for every decoration. Read the stabilizer flags from decoration_orbit_data, apply only the active group operations to both sites of every axis pair, and treat the mapped pair as unordered. Match every mapped pair back to axis_pairs, build the complete axis orbits, and keep the smallest label from each orbit. Return the canonical decoration code, occupation count, retained axis label and stabilizer-orbit size.

A chemical ordering usually keeps only part of the parent symmetry. Therefore, two transformation axes can be treated as equivalent only when an operation in the decoration stabilizer maps one unordered axis pair to the other. Each row of decoration_orbit_data contains the canonical decoration code, the number of occupied sites, the parent-orbit size and one stabilizer flag for every group operation. For each decoration, the active stabilizer operations divide the supplied axis pairs into orbits. The smallest axis label is kept from each orbit, and the orbit size records how many supplied axes belong to that equivalence set.

Returns
-------
np.ndarray of integers with shape (m, 4), containing [canonical_code, occupation_count, representative_axis_index, stabilizer_orbit_size] for every retained axis representative
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reduce_undirected_axis_variants(
    decoration_orbit_data: np.ndarray,
    group_action: np.ndarray,
    axis_pairs: np.ndarray,
) -> np.ndarray:
    """
    Reduce the supplied undirected axis variants using the stabilizer
    of each chemical decoration.

    Parameters
    ----------
    decoration_orbit_data : np.ndarray
        Integer array of shape (n_decorations, 3 + group_order).
        Column 0 contains the canonical decoration code.
        Column 1 contains the occupation count.
        Column 2 contains the parent-symmetry orbit size.
        The remaining columns contain one 0/1 stabilizer flag for
        every row of group_action.
    group_action : np.ndarray
        Integer array of shape (group_order, n_sites).
        Each row is one permutation of the parent-site indices.
    axis_pairs : np.ndarray
        Integer array of shape (n_axes, 2).
        Each row contains one unordered pair of parent-site indices
        representing a candidate transformation axis.

    Returns
    -------
    retained_axis_data : np.ndarray
        Integer array of shape (n_retained, 4).
        Each row contains the canonical decoration code, occupation
        count, retained axis label and stabilizer-orbit size.
    """
    return np.empty(
        (0, 4),
        dtype=int,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduce_undirected_axis_variants(
    decoration_orbit_data: np.ndarray,
    group_action: np.ndarray,
    axis_pairs: np.ndarray,
) -> np.ndarray:
    """Reference implementation for ordering-dependent axis reduction."""

    if not isinstance(
        decoration_orbit_data,
        np.ndarray,
    ):
        raise ValueError(
            "decoration_orbit_data must be a NumPy array."
        )

    if decoration_orbit_data.ndim != 2:
        raise ValueError(
            "decoration_orbit_data must be two-dimensional."
        )

    if not np.issubdtype(
        decoration_orbit_data.dtype,
        np.integer,
    ):
        raise ValueError(
            "decoration_orbit_data must contain integers."
        )

    if not isinstance(
        group_action,
        np.ndarray,
    ):
        raise ValueError(
            "group_action must be a NumPy array."
        )

    if group_action.ndim != 2:
        raise ValueError(
            "group_action must be two-dimensional."
        )

    if not np.issubdtype(
        group_action.dtype,
        np.integer,
    ):
        raise ValueError(
            "group_action must contain integers."
        )

    if not isinstance(
        axis_pairs,
        np.ndarray,
    ):
        raise ValueError(
            "axis_pairs must be a NumPy array."
        )

    if (
        axis_pairs.ndim != 2
        or axis_pairs.shape[1] != 2
    ):
        raise ValueError(
            "axis_pairs must have shape (n_axes, 2)."
        )

    if not np.issubdtype(
        axis_pairs.dtype,
        np.integer,
    ):
        raise ValueError(
            "axis_pairs must contain integers."
        )

    group_order, n_sites = group_action.shape

    if n_sites < 1:
        raise ValueError(
            "group_action must contain at least one site."
        )

    expected_sites = np.arange(
        n_sites,
        dtype=int,
    )

    for operation in group_action:
        if not np.array_equal(
            np.sort(operation),
            expected_sites,
        ):
            raise ValueError(
                "Every row of group_action must be a permutation "
                "of 0 through n_sites - 1."
            )

    expected_width = 3 + group_order

    if decoration_orbit_data.shape[1] != expected_width:
        raise ValueError(
            "decoration_orbit_data must contain three metadata columns "
            "followed by one stabilizer flag for every group operation."
        )

    stabilizer_flags = decoration_orbit_data[
        :,
        3:,
    ]

    if np.any(
        (stabilizer_flags != 0)
        & (stabilizer_flags != 1)
    ):
        raise ValueError(
            "Stabilizer flags must be 0 or 1."
        )

    if (
        decoration_orbit_data.shape[0] > 0
        and np.any(
            np.sum(
                stabilizer_flags,
                axis=1,
            ) == 0
        )
    ):
        raise ValueError(
            "Every decoration must contain at least one "
            "stabilizer operation."
        )

    if (
        np.any(axis_pairs < 0)
        or np.any(axis_pairs >= n_sites)
    ):
        raise ValueError(
            "Every axis-pair site must be a valid site index."
        )

    normalized_pairs = [
        tuple(
            sorted(
                (
                    int(first),
                    int(second),
                )
            )
        )
        for first, second in axis_pairs.tolist()
    ]

    if any(
        first == second
        for first, second in normalized_pairs
    ):
        raise ValueError(
            "Each axis pair must contain two different sites."
        )

    if len(
        set(normalized_pairs)
    ) != len(normalized_pairs):
        raise ValueError(
            "The undirected axis pairs must be unique."
        )

    pair_to_label = {
        pair: label
        for label, pair in enumerate(
            normalized_pairs
        )
    }

    n_axes = len(
        normalized_pairs
    )

    output_rows = []

    for decoration_row in decoration_orbit_data:
        canonical_code = int(
            decoration_row[0]
        )

        occupation_count = int(
            decoration_row[1]
        )

        active_operations = np.flatnonzero(
            decoration_row[3:]
        )

        adjacency = [
            {label}
            for label in range(
                n_axes
            )
        ]

        for operation_index in active_operations.tolist():
            operation = group_action[
                int(operation_index)
            ]

            for axis_label, (
                first,
                second,
            ) in enumerate(
                normalized_pairs
            ):
                mapped_pair = tuple(
                    sorted(
                        (
                            int(
                                operation[first]
                            ),
                            int(
                                operation[second]
                            ),
                        )
                    )
                )

                if mapped_pair not in pair_to_label:
                    raise ValueError(
                        "axis_pairs must be closed under every "
                        "ordering-preserving operation."
                    )

                mapped_label = pair_to_label[
                    mapped_pair
                ]

                adjacency[
                    axis_label
                ].add(
                    mapped_label
                )

                adjacency[
                    mapped_label
                ].add(
                    axis_label
                )

        unseen = set(
            range(n_axes)
        )

        while unseen:
            start = min(
                unseen
            )

            orbit = {
                start
            }

            stack = [
                start
            ]

            while stack:
                current = stack.pop()

                for neighbour in adjacency[
                    current
                ]:
                    if neighbour not in orbit:
                        orbit.add(
                            neighbour
                        )

                        stack.append(
                            neighbour
                        )

            unseen.difference_update(
                orbit
            )

            representative = min(
                orbit
            )

            output_rows.append(
                [
                    canonical_code,
                    occupation_count,
                    representative,
                    len(orbit),
                ]
            )

    if not output_rows:
        return np.empty(
            (0, 4),
            dtype=int,
        )

    return np.asarray(
        output_rows,
        dtype=int,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return unit-test cases for ordering-dependent axis reduction."""

    return [
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4, 5],
    [0, 5, 4, 3, 2, 1],
    [1, 0, 5, 4, 3, 2],
    [1, 2, 3, 4, 5, 0],
    [2, 1, 0, 5, 4, 3],
    [2, 3, 4, 5, 0, 1],
    [3, 2, 1, 0, 5, 4],
    [3, 4, 5, 0, 1, 2],
    [4, 3, 2, 1, 0, 5],
    [4, 5, 0, 1, 2, 3],
    [5, 0, 1, 2, 3, 4],
    [5, 4, 3, 2, 1, 0],
], dtype=int)

decoration_orbit_data = np.array([
    [48, 2, 6, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [40, 2, 6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [36, 2, 3, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [56, 3, 6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [52, 3, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [42, 3, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
], dtype=int)

axis_pairs = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
], dtype=int)""",
            "call": """reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
            "gold_call": """_oracle_reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3],
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [1, 2, 3, 0],
    [2, 1, 0, 3],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [3, 2, 1, 0],
], dtype=int)

decoration_orbit_data = np.array([
    [15, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

axis_pairs = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 2],
    [1, 3],
], dtype=int)""",
            "call": """reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
            "gold_call": """_oracle_reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4],
], dtype=int)

decoration_orbit_data = np.array([
    [24, 2, 1, 1],
], dtype=int)

axis_pairs = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 4],
], dtype=int)""",
            "call": """reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
            "gold_call": """_oracle_reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 0],
    [2, 3, 4, 5, 0, 1],
    [3, 4, 5, 0, 1, 2],
    [4, 5, 0, 1, 2, 3],
    [5, 0, 1, 2, 3, 4],
], dtype=int)

decoration_orbit_data = np.array([
    [42, 3, 2, 1, 0, 1, 0, 1, 0],
], dtype=int)

axis_pairs = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
], dtype=int)""",
            "call": """reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
            "gold_call": """_oracle_reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 0],
    [2, 3, 4, 5, 0, 1],
    [3, 4, 5, 0, 1, 2],
    [4, 5, 0, 1, 2, 3],
    [5, 0, 1, 2, 3, 4],
], dtype=int)

decoration_orbit_data = np.empty(
    (0, 9),
    dtype=int,
)

axis_pairs = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
], dtype=int)""",
            "call": """reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
            "gold_call": """_oracle_reduce_undirected_axis_variants(
    decoration_orbit_data,
    group_action,
    axis_pairs,
)""",
        },
    ]
