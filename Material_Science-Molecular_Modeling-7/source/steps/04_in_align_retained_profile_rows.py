"""
Match every retained axis row with its energy-profile row. Use the canonical decoration code and retained axis label as the key. Keep the first four columns of retained_axis_data and append the matching DFT profile followed by the matching MLIP profile. Preserve the order of retained_axis_data. Every retained key must occur exactly once in profile_keys.

After symmetry reduction, only the retained decoration-pathway pairs should be used for the DFT and MLIP comparison. The input energy table is given in a mixed order, so row position cannot be used to identify a pathway. Each row must be identified by the pair formed from the canonical decoration code and the supplied pathway label. The DFT and MLIP profiles from the same row must remain together. Rows belonging to pathways that were removed by symmetry should not be included.

Returns
-------
np.ndarray of floating-point values with shape (n_retained, 4 + 2*n_samples). Each row contains [canonical_code, occupation_count, representative_axis_index, stabilizer_orbit_size], followed by the DFT profile and the MLIP profile.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def align_retained_profile_rows(
    retained_axis_data: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
) -> np.ndarray:
    """
    Match every retained decoration-axis pair with its DFT and MLIP
    energy-profile row.

    Parameters
    ----------
    retained_axis_data : np.ndarray
        Integer array of shape (n_retained, 4).
        Each row contains the canonical decoration code, occupation
        count, retained axis label and stabilizer-orbit size.
    profile_keys : np.ndarray
        Integer array of shape (n_profiles, 2).
        Each row contains a canonical decoration code and pathway
        label identifying one energy-profile row.
    dft_profiles : np.ndarray
        Numerical array of shape (n_profiles, n_samples).
        Each row contains one DFT transformation-energy profile.
    mlip_profiles : np.ndarray
        Numerical array of shape (n_profiles, n_samples).
        Each row contains the MLIP profile paired with the DFT profile
        having the same profile key.

    Returns
    -------
    aligned_profile_rows : np.ndarray
        Floating-point array of shape
        (n_retained, 4 + 2 * n_samples).
        The first four columns contain retained-axis metadata,
        followed by the matched DFT profile and MLIP profile.
    """
    return np.empty(
        (
            0,
            4 + 2 * dft_profiles.shape[1],
        ),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_align_retained_profile_rows(
    retained_axis_data: np.ndarray,
    profile_keys: np.ndarray,
    dft_profiles: np.ndarray,
    mlip_profiles: np.ndarray,
) -> np.ndarray:
    """Reference implementation for key-based profile alignment."""

    arrays = {
        "retained_axis_data": retained_axis_data,
        "profile_keys": profile_keys,
        "dft_profiles": dft_profiles,
        "mlip_profiles": mlip_profiles,
    }

    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

    if (
        retained_axis_data.ndim != 2
        or retained_axis_data.shape[1] != 4
    ):
        raise ValueError(
            "retained_axis_data must have shape (n_retained, 4)."
        )

    if not np.issubdtype(
        retained_axis_data.dtype,
        np.integer,
    ):
        raise ValueError(
            "retained_axis_data must contain integers."
        )

    if (
        profile_keys.ndim != 2
        or profile_keys.shape[1] != 2
    ):
        raise ValueError(
            "profile_keys must have shape (n_profiles, 2)."
        )

    if not np.issubdtype(
        profile_keys.dtype,
        np.integer,
    ):
        raise ValueError(
            "profile_keys must contain integers."
        )

    if dft_profiles.ndim != 2:
        raise ValueError(
            "dft_profiles must be two-dimensional."
        )

    if mlip_profiles.ndim != 2:
        raise ValueError(
            "mlip_profiles must be two-dimensional."
        )

    if not np.issubdtype(
        dft_profiles.dtype,
        np.number,
    ):
        raise ValueError(
            "dft_profiles must contain numerical values."
        )

    if not np.issubdtype(
        mlip_profiles.dtype,
        np.number,
    ):
        raise ValueError(
            "mlip_profiles must contain numerical values."
        )

    if profile_keys.shape[0] != dft_profiles.shape[0]:
        raise ValueError(
            "profile_keys and dft_profiles must have the same "
            "number of rows."
        )

    if profile_keys.shape[0] != mlip_profiles.shape[0]:
        raise ValueError(
            "profile_keys and mlip_profiles must have the same "
            "number of rows."
        )

    if dft_profiles.shape[1] != mlip_profiles.shape[1]:
        raise ValueError(
            "DFT and MLIP profiles must have the same number "
            "of samples."
        )

    if dft_profiles.shape[1] < 1:
        raise ValueError(
            "Every profile must contain at least one sample."
        )

    if not np.all(
        np.isfinite(dft_profiles)
    ):
        raise ValueError(
            "dft_profiles must contain finite values."
        )

    if not np.all(
        np.isfinite(mlip_profiles)
    ):
        raise ValueError(
            "mlip_profiles must contain finite values."
        )

    profile_lookup = {}

    for row_index, key_row in enumerate(
        profile_keys
    ):
        key = (
            int(key_row[0]),
            int(key_row[1]),
        )

        if key in profile_lookup:
            raise ValueError(
                "Every profile key must occur exactly once."
            )

        profile_lookup[key] = int(
            row_index
        )

    selected_rows = []
    seen_retained_keys = set()

    for retained_row in retained_axis_data:
        key = (
            int(retained_row[0]),
            int(retained_row[2]),
        )

        if key in seen_retained_keys:
            raise ValueError(
                "retained_axis_data must not contain duplicate keys."
            )

        seen_retained_keys.add(
            key
        )

        if key not in profile_lookup:
            raise ValueError(
                "A retained decoration-axis key is missing from "
                "profile_keys."
            )

        selected_rows.append(
            profile_lookup[key]
        )

    output_width = (
        4
        + dft_profiles.shape[1]
        + mlip_profiles.shape[1]
    )

    if retained_axis_data.shape[0] == 0:
        return np.empty(
            (0, output_width),
            dtype=float,
        )

    selected_rows = np.asarray(
        selected_rows,
        dtype=int,
    )

    return np.hstack(
        (
            retained_axis_data.astype(
                float
            ),
            np.asarray(
                dft_profiles[selected_rows],
                dtype=float,
            ),
            np.asarray(
                mlip_profiles[selected_rows],
                dtype=float,
            ),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return unit-test cases for retained-profile alignment."""

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
], dtype=float)""",
            "call": """align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
            "gold_call": """_oracle_align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
        },
        {
            "setup": """import numpy as np

retained_axis_data = np.array([
    [12, 2, 1, 1],
    [10, 2, 0, 2],
], dtype=int)

profile_keys = np.array([
    [10, 1],
    [12, 1],
    [10, 0],
    [12, 0],
], dtype=int)

dft_profiles = np.array([
    [9, 8, 7],
    [2, 7, 0],
    [0, 2, 4],
    [3, 2, 1],
], dtype=float)

mlip_profiles = np.array([
    [8, 7, 6],
    [3, 8, 1],
    [1, 3, 5],
    [4, 3, 2],
], dtype=float)""",
            "call": """align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
            "gold_call": """_oracle_align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
        },
        {
            "setup": """import numpy as np

retained_axis_data = np.array([
    [7, 3, 2, 1],
    [7, 3, 0, 1],
    [7, 3, 1, 1],
], dtype=int)

profile_keys = np.array([
    [7, 1],
    [7, 2],
    [7, 0],
], dtype=int)

dft_profiles = np.array([
    [3, 2, 1, 0, -1],
    [0, 1, 4, 1, 0],
    [-2, -1, 0, 1, 2],
], dtype=float)

mlip_profiles = np.array([
    [4, 3, 2, 1, 0],
    [1, 2, 5, 2, 1],
    [-1, 0, 1, 2, 3],
], dtype=float)""",
            "call": """align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
            "gold_call": """_oracle_align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
        },
        {
            "setup": """import numpy as np

retained_axis_data = np.empty(
    (0, 4),
    dtype=int,
)

profile_keys = np.array([
    [9, 0],
    [9, 1],
], dtype=int)

dft_profiles = np.array([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
], dtype=float)

mlip_profiles = np.array([
    [0, 2, 4, 6],
    [6, 4, 2, 0],
], dtype=float)""",
            "call": """align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
            "gold_call": """_oracle_align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
        },
        {
            "setup": """import numpy as np

retained_axis_data = np.array([
    [36, 2, 0, 1],
], dtype=int)

profile_keys = np.array([
    [36, 2],
    [36, 0],
    [36, 1],
], dtype=int)

dft_profiles = np.array([
    [1, 2, 3, 4, 5],
    [18, 10, 2, -4, 8],
    [8, 0, -5, 0, 18],
], dtype=float)

mlip_profiles = np.array([
    [2, 3, 4, 5, 6],
    [17, 9, 1, -3, 7],
    [7, 1, -4, 1, 17],
], dtype=float)""",
            "call": """align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
            "gold_call": """_oracle_align_retained_profile_rows(
    retained_axis_data,
    profile_keys,
    dft_profiles,
    mlip_profiles,
)""",
        },
    ]
