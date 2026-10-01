"""
Classify the DFT and MLIP profiles contained in every aligned row. The first four columns are metadata. Split the remaining values into equal DFT and MLIP profiles. Use the six profile-shape rules and the supplied class_labels. Return the four metadata columns followed by the DFT class and MLIP class. Reject a profile when it does not match exactly one allowed class.

The Mg-Nd paper separates Burgers transformation-energy profiles into six classes. Classes 1 and 2 are barrierless profiles, classes 3 and 4 contain an interior energy maximum, and classes 5 and 6 contain an interior energy minimum. The two classes inside each pair differ by which endpoint has the lower energy. DFT and MLIP profiles must be classified separately. An additive energy shift does not change a profile class because only profile shape and relative endpoint energies are used.

Returns
-------
np.ndarray of integers with shape (n_retained, 6), containing [canonical_code, occupation_count, representative_axis_index, stabilizer_orbit_size, dft_class, mlip_class] for every retained pathway
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def classify_transformation_landscapes(
    aligned_profile_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """
    Assign one transformation-landscape class to every retained DFT
    profile and every retained MLIP profile.

    Parameters
    ----------
    aligned_profile_rows : np.ndarray
        Numerical array of shape (n_retained, 4 + 2 * n_samples).
        The first four columns contain retained-axis metadata.
        The next n_samples values contain the DFT profile and the last
        n_samples values contain the paired MLIP profile.
    class_labels : np.ndarray
        One-dimensional integer array of length 6.
        The entries give the labels used for the six transformation
        landscape classes in class-definition order.

    Returns
    -------
    classified_rows : np.ndarray
        Integer array of shape (n_retained, 6).
        Columns 0 through 3 contain the retained-axis metadata.
        Column 4 contains the DFT class and column 5 contains the
        MLIP class.
    """
    return np.empty(
        (0, 6),
        dtype=int,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_classify_transformation_landscapes(
    aligned_profile_rows: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Reference implementation for six-class profile assignment."""

    if not isinstance(
        aligned_profile_rows,
        np.ndarray,
    ):
        raise ValueError(
            "aligned_profile_rows must be a NumPy array."
        )

    if aligned_profile_rows.ndim != 2:
        raise ValueError(
            "aligned_profile_rows must be two-dimensional."
        )

    if not np.issubdtype(
        aligned_profile_rows.dtype,
        np.number,
    ):
        raise ValueError(
            "aligned_profile_rows must contain numerical values."
        )

    if not isinstance(
        class_labels,
        np.ndarray,
    ):
        raise ValueError(
            "class_labels must be a NumPy array."
        )

    if (
        class_labels.ndim != 1
        or class_labels.shape != (6,)
    ):
        raise ValueError(
            "class_labels must have shape (6,)."
        )

    if not np.issubdtype(
        class_labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "class_labels must contain integers."
        )

    if len(
        set(
            int(value)
            for value in class_labels.tolist()
        )
    ) != 6:
        raise ValueError(
            "class_labels must contain six different values."
        )

    if aligned_profile_rows.shape[1] < 10:
        raise ValueError(
            "aligned_profile_rows must contain four metadata "
            "columns and two profiles having at least three "
            "samples each."
        )

    profile_width = (
        aligned_profile_rows.shape[1]
        - 4
    )

    if profile_width % 2 != 0:
        raise ValueError(
            "The DFT and MLIP profile blocks must have equal width."
        )

    n_samples = profile_width // 2

    if n_samples < 3:
        raise ValueError(
            "Every profile must contain at least three samples."
        )

    if not np.all(
        np.isfinite(
            aligned_profile_rows
        )
    ):
        raise ValueError(
            "aligned_profile_rows must contain finite values."
        )

    metadata = aligned_profile_rows[
        :,
        :4,
    ]

    if not np.all(
        metadata
        == np.rint(metadata)
    ):
        raise ValueError(
            "The first four metadata columns must be integer-valued."
        )

    def _classify_profile(
        profile: np.ndarray,
    ) -> int:
        differences = np.diff(
            profile
        )

        if np.all(
            differences < 0
        ):
            class_position = 0

        elif np.all(
            differences > 0
        ):
            class_position = 1

        else:
            first_energy = float(
                profile[0]
            )

            last_energy = float(
                profile[-1]
            )

            if first_energy == last_energy:
                raise ValueError(
                    "A nonmonotonic profile must have different "
                    "endpoint energies."
                )

            interior = profile[
                1:-1
            ]

            has_barrier = bool(
                np.max(interior)
                > max(
                    first_energy,
                    last_energy,
                )
            )

            has_intermediate = bool(
                np.min(interior)
                < min(
                    first_energy,
                    last_energy,
                )
            )

            if has_barrier == has_intermediate:
                raise ValueError(
                    "Every nonmonotonic profile must contain exactly "
                    "one allowed interior feature."
                )

            first_is_higher = (
                first_energy
                > last_energy
            )

            if has_barrier:
                class_position = (
                    2
                    if first_is_higher
                    else 3
                )

            else:
                class_position = (
                    4
                    if first_is_higher
                    else 5
                )

        return int(
            class_labels[
                class_position
            ]
        )

    output_rows = []

    for row in aligned_profile_rows:
        dft_profile = row[
            4:
            4 + n_samples
        ]

        mlip_profile = row[
            4 + n_samples:
        ]

        output_rows.append(
            [
                *(
                    int(value)
                    for value in row[:4]
                ),
                _classify_profile(
                    dft_profile
                ),
                _classify_profile(
                    mlip_profile
                ),
            ]
        )

    if not output_rows:
        return np.empty(
            (0, 6),
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
    """Return unit-test cases for transformation-profile classification."""

    return [
        {
            "setup": """import numpy as np

retained_axis_data = np.array([
    [48, 2, 0, 1],
    [48, 2, 1, 2],
    [40, 2, 0, 2],
    [40, 2, 2, 1],
    [36, 2, 0, 2],
    [36, 2, 1, 1],
    [56, 3, 0, 2],
    [56, 3, 2, 1],
    [52, 3, 0, 1],
    [52, 3, 1, 1],
    [52, 3, 2, 1],
    [42, 3, 0, 3],
], dtype=int)

profile_keys = np.array([
    [52, 1],
    [48, 0],
    [42, 2],
    [40, 2],
    [36, 2],
    [48, 2],
    [56, 1],
    [40, 0],
    [42, 0],
    [36, 1],
    [52, 0],
    [48, 1],
    [42, 1],
    [56, 2],
    [36, 0],
    [52, 2],
    [40, 1],
    [56, 0],
], dtype=int)

dft_profiles = np.array([
    [18, 16, 13, 9, 5, 2, 0],
    [0, 2, 5, 9, 13, 16, 18],
    [0, 2, 5, 9, 13, 16, 18],
    [2, 12, 21, 25, 22, 15, 8],
    [8, 0, -5, -4, 2, 10, 18],
    [18, 16, 13, 9, 5, 2, 0],
    [8, 15, 22, 25, 21, 12, 2],
    [8, 15, 22, 25, 21, 12, 2],
    [0, 2, 5, 9, 13, 16, 18],
    [18, 10, 2, -4, -5, 0, 8],
    [8, 0, -5, -4, 2, 10, 18],
    [18, 16, 13, 9, 5, 2, 0],
    [0, 2, 5, 9, 13, 16, 18],
    [2, 12, 21, 25, 22, 15, 8],
    [8, 0, -5, -4, 2, 10, 18],
    [12, 7, 1, -4, -5, -1, 8],
    [8, 15, 22, 25, 21, 12, 2],
    [8, 15, 22, 25, 21, 12, 2],
], dtype=float)

mlip_profiles = np.array([
    [17, 15, 12, 8, 5, 2, 0],
    [0, 2, 6, 10, 13, 16, 19],
    [0, 2, 6, 10, 13, 16, 19],
    [1, 3, 6, 10, 14, 17, 20],
    [7, 1, -4, -3, 2, 11, 17],
    [17, 12, 6, -1, -3, 2, 5],
    [9, 14, 21, 24, 20, 11, 3],
    [3, 11, 20, 24, 21, 14, 9],
    [0, 2, 6, 10, 13, 16, 19],
    [17, 9, 1, -3, -4, 1, 7],
    [7, 1, -4, -3, 2, 11, 17],
    [17, 12, 6, -1, -3, 2, 5],
    [0, 2, 6, 10, 13, 16, 19],
    [10, 18, 24, 26, 20, 12, 4],
    [7, 1, -4, -3, 2, 11, 17],
    [7, 1, -4, -3, 2, 8, 11],
    [3, 11, 20, 24, 21, 14, 9],
    [9, 14, 21, 24, 20, 11, 3],
], dtype=float)

profile_lookup = {
    tuple(key): index
    for index, key in enumerate(profile_keys)
}

selected_rows = np.array([
    profile_lookup[(int(row[0]), int(row[2]))]
    for row in retained_axis_data
], dtype=int)

aligned_profile_rows = np.hstack([
    retained_axis_data.astype(float),
    dft_profiles[selected_rows],
    mlip_profiles[selected_rows],
])

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
            "gold_call": """_oracle_classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

profile_palette = np.array([
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5],
    [5, 7, 9, 6, 1],
    [1, 6, 9, 7, 5],
    [5, 1, -2, 0, 2],
    [1, 0, -2, 2, 5],
], dtype=float)

metadata = np.array([
    [1, 1, 0, 1],
    [2, 1, 0, 1],
    [4, 1, 0, 1],
    [8, 1, 0, 1],
    [16, 1, 0, 1],
    [32, 1, 0, 1],
], dtype=float)

aligned_profile_rows = np.hstack([
    metadata,
    profile_palette,
    profile_palette,
])

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
            "gold_call": """_oracle_classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

metadata = np.array([
    [3, 2, 0, 1],
    [5, 2, 1, 1],
], dtype=float)

dft_profiles = np.array([
    [5, 4, 3, 2, 1],
    [5, 7, 9, 6, 1],
], dtype=float)

mlip_profiles = np.array([
    [1, 2, 3, 4, 5],
    [1, 6, 9, 7, 5],
], dtype=float)

aligned_profile_rows = np.hstack([
    metadata,
    dft_profiles,
    mlip_profiles,
])

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
            "gold_call": """_oracle_classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

metadata = np.array([
    [3, 2, 0, 1],
    [5, 2, 1, 1],
], dtype=float)

dft_profiles = np.array([
    [5, 4, 3, 2, 1],
    [5, 7, 9, 6, 1],
], dtype=float)

mlip_profiles = np.array([
    [1, 2, 3, 4, 5],
    [1, 6, 9, 7, 5],
], dtype=float)

aligned_profile_rows = np.hstack([
    metadata,
    dft_profiles,
    mlip_profiles,
])

class_labels = np.array(
    [1, 2, 3, 6, 5, 4],
    dtype=int,
)""",
            "call": """classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
            "gold_call": """_oracle_classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
        },
        {
            "setup": """import numpy as np

aligned_profile_rows = np.empty(
    (0, 14),
    dtype=float,
)

class_labels = np.array(
    [1, 2, 3, 4, 5, 6],
    dtype=int,
)""",
            "call": """classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
            "gold_call": """_oracle_classify_transformation_landscapes(
    aligned_profile_rows,
    class_labels,
)""",
        },
    ]
