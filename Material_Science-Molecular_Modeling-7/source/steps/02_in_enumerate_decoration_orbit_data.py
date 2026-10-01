"""
A chemical ordering can be represented by a binary occupation vector. For the Mg-Nd panel, a value of 1 means Nd and a value of 0 means Mg. Two decorations should not be counted separately when a symmetry operation of the parent structure converts one into the other. All decorations connected in this way form one symmetry orbit. One representative is selected from every orbit. Here the representative is the decoration having the largest binary code. The orbit size tells how many decorations are related to the representative by parent symmetry. The symmetry operations that leave the representative unchanged form its stabilizer. For a finite group, the orbit size and stabilizer size are related by the orbit-stabilizer theorem. These values are needed in the next step because chemical ordering can reduce the symmetry available to a transformation pathway.

A chemical ordering can be represented by a binary occupation vector. For the Mg-Nd panel, a value of 1 means Nd and a value of 0 means Mg. Two decorations should not be counted separately when a symmetry operation of the parent structure converts one into the other. All decorations connected in this way form one symmetry orbit. One representative is selected from every orbit. Here the representative is the decoration having the largest binary code. The orbit size tells how many decorations are related to the representative by parent symmetry. The symmetry operations that leave the representative unchanged form its stabilizer. For a finite group, the orbit size and stabilizer size are related by the orbit-stabilizer theorem. These values are needed in the next step because chemical ordering can reduce the symmetry available to a transformation pathway.

Returns
-------
np.ndarray of integers with shape (n_decorations, 3 + group_order), containing the canonical binary code, occupation count, parent-orbit size, and stabilizer-membership flags for every symmetry-distinct decoration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def enumerate_decoration_orbit_data(
    group_action: np.ndarray,
    occupation_counts: np.ndarray,
) -> np.ndarray:
    """
    Enumerate symmetry-distinct binary decorations and their stabilizer data.

    Parameters
    ----------
    group_action : np.ndarray
        Integer array of shape (group_order, n_sites). Each row contains one
        parent site permutation in image notation.
    occupation_counts : np.ndarray
        One-dimensional integer array containing the allowed numbers of
        occupied sites.

    Returns
    -------
    decoration_orbit_data : np.ndarray
        Integer array of shape (n_decorations, 3 + group_order).
        Column 0 contains the canonical binary decoration code.
        Column 1 contains the occupation count.
        Column 2 contains the parent-symmetry orbit size.
        The remaining columns contain 0/1 stabilizer-membership flags in
        the same order as the rows of group_action.
    """
    return np.empty(
        (
            0,
            3 + group_action.shape[0],
        ),
        dtype=int,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from itertools import combinations


def _oracle_enumerate_decoration_orbit_data(
    group_action: np.ndarray,
    occupation_counts: np.ndarray,
) -> np.ndarray:
    if not isinstance(group_action, np.ndarray):
        raise ValueError("group_action must be a NumPy array.")

    if group_action.ndim != 2:
        raise ValueError("group_action must be two-dimensional.")

    if not np.issubdtype(group_action.dtype, np.integer):
        raise ValueError("group_action must contain integers.")

    if not isinstance(occupation_counts, np.ndarray):
        raise ValueError("occupation_counts must be a NumPy array.")

    if occupation_counts.ndim != 1:
        raise ValueError("occupation_counts must be one-dimensional.")

    if not np.issubdtype(occupation_counts.dtype, np.integer):
        raise ValueError("occupation_counts must contain integers.")

    n_group, n_sites = group_action.shape

    if n_sites < 1:
        raise ValueError("group_action must contain at least one site.")

    expected = np.arange(n_sites, dtype=int)

    for operation in group_action:
        if not np.array_equal(np.sort(operation), expected):
            raise ValueError(
                "Every row of group_action must be a permutation."
            )

    if np.any(
        (occupation_counts < 0)
        | (occupation_counts > n_sites)
    ):
        raise ValueError(
            "occupation_counts must lie between 0 and n_sites."
        )

    def binary_code(bits):
        code = 0

        for bit in bits:
            code = (code << 1) | int(bit)

        return int(code)

    def transform(bits, operation):
        transformed = np.zeros(
            n_sites,
            dtype=int,
        )

        for site in range(n_sites):
            transformed[
                int(operation[site])
            ] = bits[site]

        return transformed

    rows = []

    for occupation_count in occupation_counts.tolist():
        seen = set()

        for occupied_sites in combinations(
            range(n_sites),
            int(occupation_count),
        ):
            decoration = np.zeros(
                n_sites,
                dtype=int,
            )

            if occupied_sites:
                decoration[
                    list(occupied_sites)
                ] = 1

            decoration_tuple = tuple(
                int(value)
                for value in decoration.tolist()
            )

            if decoration_tuple in seen:
                continue

            orbit = {}

            for operation in group_action:
                image = transform(
                    decoration,
                    operation,
                )

                orbit[
                    tuple(
                        int(value)
                        for value in image.tolist()
                    )
                ] = image

            seen.update(
                orbit.keys()
            )

            canonical = max(
                orbit.values(),
                key=binary_code,
            )

            canonical_code = binary_code(
                canonical
            )

            stabilizer_flags = []

            for operation in group_action:
                image = transform(
                    canonical,
                    operation,
                )

                stabilizer_flags.append(
                    int(
                        np.array_equal(
                            image,
                            canonical,
                        )
                    )
                )

            rows.append(
                [
                    canonical_code,
                    int(occupation_count),
                    len(orbit),
                    *stabilizer_flags,
                ]
            )

    rows.sort(
        key=lambda row: (
            list(
                occupation_counts.tolist()
            ).index(row[1]),
            -row[0],
        )
    )

    if not rows:
        return np.empty(
            (
                0,
                3 + n_group,
            ),
            dtype=int,
        )

    return np.asarray(
        rows,
        dtype=int,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
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

occupation_counts = np.array([2, 3], dtype=int)""",
            "call": """enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
            "gold_call": """_oracle_enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4],
], dtype=int)

occupation_counts = np.array([2], dtype=int)""",
            "call": """enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
            "gold_call": """_oracle_enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [2, 1, 0, 3],
    [3, 2, 1, 0],
], dtype=int)

occupation_counts = np.array([1, 2], dtype=int)""",
            "call": """enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
            "gold_call": """_oracle_enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
        },
        {
            "setup": """import numpy as np

group_action = np.array([
    [0, 1, 2, 3, 4],
], dtype=int)

occupation_counts = np.array([0, 5], dtype=int)""",
            "call": """enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
            "gold_call": """_oracle_enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
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

occupation_counts = np.array([], dtype=int)""",
            "call": """enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
            "gold_call": """_oracle_enumerate_decoration_orbit_data(
    group_action,
    occupation_counts,
)""",
        },
    ]
