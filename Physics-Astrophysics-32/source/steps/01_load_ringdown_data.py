"""
Recorded spherical 22 and 64 ringdown multipoles and differences.

Return the seven rows in the Problem Statement in the stated order. $amplitude_scale$ and $time_scale$ are finite and strictly positive; the first multiplies every column except time and the second every time entry. The task instance uses one for both.

Returns
-------
`numpy.ndarray` of floating dtype and shape `(7, 9)`. Every returned component is compared at numerical tolerance `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
_BASE_DATA = np.array([
    [0.00, 0.607954649691, 0.295051751110, 0.000628378919, -0.000337532774, -0.037867242596, -0.001909207733, 0.000044617819, -0.000067295330],
    [3.00, 0.147374778600, -0.432772573681, 0.000144374885, 0.000297055983, -0.013031289479, 0.008698702522, -0.000040102718, -0.000121982286],
    [6.50, -0.303886427800, 0.075363262462, 0.000046548106, 0.000234305339, 0.029983620352, 0.019511427616, -0.000092589566, -0.000105810338],
    [10.50, 0.206496858961, 0.222099527098, 0.000017845774, 0.000196568821, 0.048606458833, 0.001108395423, 0.000046898531, -0.000025239149],
    [15.00, -0.060781083081, -0.123253180953, 0.000032917024, 0.000059030411, 0.059110971375, -0.059988566549, 0.000061689148, -0.000106702268],
    [20.00, 0.087855912970, 0.131424789334, -0.000048979985, -0.000053579051, 0.031337457207, 0.013158966817, -0.000027079682, 0.000062232067],
    [26.00, -0.044012797140, -0.023897529367, 0.000015531811, 0.000054570561, -0.034883268283, 0.023852694832, -0.000009366161, -0.000023349673],
], dtype=float)
def load_ringdown_data(amplitude_scale: float = 1.0, time_scale: float = 1.0) -> np.ndarray:
    """Return the supplied spherical-multipole record at a common scale.

    Parameters
    ----------
    amplitude_scale : float, default=1.0
        Positive multiplier applied to waveform and resolution-difference
        columns.
    time_scale : float, default=1.0
        Positive multiplier applied to every supplied time sample.

    Returns
    -------
    numpy.ndarray, shape (7, 9)
        Time; real and imaginary 22 and 64 multipoles; then real and
        imaginary two-resolution differences for those multipoles. Components
        are compared at numerical tolerance 1e-9.

    Raises
    ------
    ValueError
        If either scale is nonfinite or not strictly positive.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""
Recorded spherical 22 and 64 ringdown multipoles and differences.

Return the seven rows in the Problem Statement in the stated order. `amplitude_scale` and `time_scale` are finite and strictly positive; the first multiplies every non-time column and the second every time entry. The task instance uses one for both.

Returns
-------
`numpy.ndarray` of floating dtype and shape `(7, 9)`.

Every returned component is compared at numerical tolerance `1e-9`.
"""
"""Recorded spherical 22 and 64 ringdown multipoles and differences."""
import numpy as np
"""Recorded spherical 22 and 64 ringdown multipoles and differences."""
import numpy as np
_BASE_DATA = np.array([
    [0.00, 0.607954649691, 0.295051751110, 0.000628378919, -0.000337532774, -0.037867242596, -0.001909207733, 0.000044617819, -0.000067295330],
    [3.00, 0.147374778600, -0.432772573681, 0.000144374885, 0.000297055983, -0.013031289479, 0.008698702522, -0.000040102718, -0.000121982286],
    [6.50, -0.303886427800, 0.075363262462, 0.000046548106, 0.000234305339, 0.029983620352, 0.019511427616, -0.000092589566, -0.000105810338],
    [10.50, 0.206496858961, 0.222099527098, 0.000017845774, 0.000196568821, 0.048606458833, 0.001108395423, 0.000046898531, -0.000025239149],
    [15.00, -0.060781083081, -0.123253180953, 0.000032917024, 0.000059030411, 0.059110971375, -0.059988566549, 0.000061689148, -0.000106702268],
    [20.00, 0.087855912970, 0.131424789334, -0.000048979985, -0.000053579051, 0.031337457207, 0.013158966817, -0.000027079682, 0.000062232067],
    [26.00, -0.044012797140, -0.023897529367, 0.000015531811, 0.000054570561, -0.034883268283, 0.023852694832, -0.000009366161, -0.000023349673],
], dtype=float)
def _oracle_load_ringdown_data(amplitude_scale: float = 1.0, time_scale: float = 1.0) -> np.ndarray:
    scale = float(amplitude_scale)
    time_multiplier = float(time_scale)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(time_multiplier) or time_multiplier <= 0.0:
        raise ValueError("amplitude_scale and time_scale must be finite and positive")
    data = _BASE_DATA.copy()
    data[:, 0] *= time_multiplier
    data[:, 1:] *= scale
    return data

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Nominal and non-task-scale records."""
    return [
        {
            "setup": "import numpy as np\nscale, time_scale = 0.85, 0.90",
            "call": "load_ringdown_data(scale, time_scale)",
            "gold_call": "_oracle_load_ringdown_data(scale, time_scale)",
        },
        {
            "setup": "import numpy as np\nscale, time_scale = 1.15, 1.10",
            "call": "load_ringdown_data(scale, time_scale)",
            "gold_call": "_oracle_load_ringdown_data(scale, time_scale)",
        },
        {
            "setup": "def catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return True\n    return False",
            "call": "catches_value_error(lambda: load_ringdown_data(0.0, 1.0)) and catches_value_error(lambda: load_ringdown_data(float('nan'), 1.0)) and catches_value_error(lambda: load_ringdown_data(1.0, 0.0)) and catches_value_error(lambda: load_ringdown_data(1.0, float('inf')))",
            "gold_call": "catches_value_error(lambda: _oracle_load_ringdown_data(0.0, 1.0)) and catches_value_error(lambda: _oracle_load_ringdown_data(float('nan'), 1.0)) and catches_value_error(lambda: _oracle_load_ringdown_data(1.0, 0.0)) and catches_value_error(lambda: _oracle_load_ringdown_data(1.0, float('inf')))",
        },
    ]
