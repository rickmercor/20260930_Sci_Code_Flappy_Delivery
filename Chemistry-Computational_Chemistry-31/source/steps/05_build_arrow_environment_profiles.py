"""
Collapse ordered mechanisms into unique arrow-environment profiles.

The source compares mechanisms through the arrow environments represented by their elementary rules. Each candidate is converted into a binary set-union profile, so an environment is recorded once even when it occurs in multiple rules or repeated copies of the same rule.

Returns
-------
Binary integer array containing one union profile per mechanism.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_arrow_environment_profiles(
    mechanisms: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
) -> 'np.ndarray':
    """Build one binary union profile for every mechanism.

    ``mechanisms`` uses the row layout returned by
    ``generate_parsimonious_mechanisms``. ``rule_arrow_matrix`` has one row per
    one-based rule label and one column per possible arrow environment. Rule
    repetition does not duplicate an environment because each mechanism is
    represented by a set union.

    Returns
    -------
    np.ndarray
        Binary integer array with shape ``(n_mechanisms, n_environments)``.

    Raises
    ------
    ValueError
        If an input is malformed or a used rule label is out of range.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_arrow_environment_profiles(
    mechanisms: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    mechs = np.asarray(mechanisms)
    arrows = np.asarray(rule_arrow_matrix)

    if mechs.ndim != 2 or mechs.shape[1] < 3:
        raise ValueError(
            "mechanisms must be a two-dimensional padded table"
        )
    if (
        arrows.ndim != 2
        or arrows.shape[0] == 0
        or arrows.shape[1] == 0
    ):
        raise ValueError(
            "rule_arrow_matrix must be nonempty and two-dimensional"
        )

    for value, name in (
        (mechs, "mechanisms"),
        (arrows, "rule_arrow_matrix"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")

    if np.any(mechs != np.rint(mechs)):
        raise ValueError("mechanisms must contain integers")
    if np.any((arrows != 0) & (arrows != 1)):
        raise ValueError("rule_arrow_matrix must be binary")

    mechs = mechs.astype(int, copy=False)
    profiles = np.zeros(
        (mechs.shape[0], arrows.shape[1]),
        dtype=int,
    )
    capacity = mechs.shape[1] - 2

    for index, row in enumerate(mechs):
        n_steps = int(row[1])

        if n_steps <= 0 or n_steps > capacity:
            raise ValueError("mechanism step count is invalid")

        labels = row[2:2 + n_steps]

        if (
            np.any(labels < 1)
            or np.any(labels > arrows.shape[0])
        ):
            raise ValueError(
                "mechanism contains an out-of-range rule label"
            )
        if np.any(row[2 + n_steps:] != 0):
            raise ValueError(
                "mechanism padding must be zero"
            )

        profiles[index] = np.any(
            arrows[labels - 1] != 0,
            axis=0,
        ).astype(int)

    return profiles

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nm=np.array([[2,3,1,2,3,0],[7,2,2,4,0,0]])\na=np.array([[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1]])",
            "call": "build_arrow_environment_profiles(m.copy(),a.copy())",
            "gold_call": "_oracle_build_arrow_environment_profiles(m.copy(),a.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.array([[1,4,1,1,2,1]])\na=np.array([[1,0,0],[0,1,1]])",
            "call": "build_arrow_environment_profiles(m.copy(),a.copy())",
            "gold_call": "_oracle_build_arrow_environment_profiles(m.copy(),a.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.empty((0,5),dtype=int)\na=np.eye(3,dtype=int)",
            "call": "build_arrow_environment_profiles(m.copy(),a.copy())",
            "gold_call": "_oracle_build_arrow_environment_profiles(m.copy(),a.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.array([[1,2,1,5]])\na=np.eye(3,dtype=int)\ndef check(fn):\n try: fn(m.copy(),a.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(build_arrow_environment_profiles)",
            "gold_call": "check(_oracle_build_arrow_environment_profiles)",
        },
    ]
