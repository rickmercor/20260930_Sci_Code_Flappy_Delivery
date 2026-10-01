"""
Run the complete propagation and angular-density pipeline with plotting disabled, then return one dimensionless float: the fraction of retained annular power in bins whose polar center angle is at least theta_min. Recover common-scale annular powers by multiplying the angular density by each angular-bin width. Divide the selected-bin sum by the total retained-bin sum. Return 0.0 when retained power is zero. The cutoff defaults to pi/6 and must be a finite real scalar in [0,pi/2]. This is the final orchestrator of the scalar task.

The angular density from Step 7 is peak-normalized radial power divided by unequal polar angular-bin widths. Multiplication by those widths recovers annular powers up to a common factor, which cancels in the selected-to-total ratio. Summing density values directly would incorrectly weight narrow angular bins. Reconstruct radial edges from the uniformly spaced radial-bin centers and use arcsin(radial_edges/k0) for angular edges. Select whole bins by their center angles, including equality; do not interpolate partial-bin power. At the specified 1024-bin benchmark cutoff pi/6, the selected bins begin at the radial edge k0/2.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wide_angle_power_fraction(
    U0, P, k0, h, zlength, dx, dy, mx, B, C,
    nbins=None, theta_min=np.pi / 6
):
    """Return the retained annular-power fraction above a polar cutoff.

    Propagation inputs follow propagate_and_angular_power. Run that
    pipeline with plot=False; nbins defaults to the transverse size.
    Multiply angular density by angular-bin widths to recover relative
    annular powers. Derive radial edges from the uniform radial centers
    and angular edges from arcsin(radial_edges/k0).
    Select whole bins whose center angle is at least theta_min.
    theta_min is a finite real scalar in [0, pi/2], measured in radians.
    Return one float in [0, 1], or 0.0 when retained power is zero.

    Raises
    ------
    ValueError
        If theta_min is non-scalar, non-real, Boolean, non-finite,
        or outside [0, pi/2].
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wide_angle_power_fraction(
    U0, P, k0, h, zlength, dx, dy, mx, B, C,
    nbins=None, theta_min=np.pi / 6
):
    """Return the power fraction selected by polar bin-center angle.

    Uniform radial centers determine the angular-bin widths. Multiplying
    density by these widths recovers common-scale annular powers.
    Select bins whose center angle is at least theta_min. The cutoff
    must be a finite real scalar in [0, pi/2]. Zero retained power gives 0.
    """
    raw_angle = np.asarray(theta_min)
    if raw_angle.ndim != 0 or raw_angle.dtype.kind not in "iuf":
        raise ValueError("theta_min must be a finite real scalar.")

    minimum_angle = float(raw_angle)
    if not np.isfinite(minimum_angle) or not 0 <= minimum_angle <= np.pi / 2:
        raise ValueError("theta_min must lie in [0, pi/2].")

    outputs = _oracle_propagate_and_angular_power(
        U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=nbins, plot=False
    )
    centers, density, angles = outputs[4:]
    if len(centers) == 0:
        return 0.0

    radial_edges = 2 * centers[0] * np.arange(len(centers) + 1, dtype=float)
    angular_widths = np.diff(
        np.arcsin(np.clip(radial_edges / k0, 0.0, 1.0))
    )
    weights = density * angular_widths
    total = np.sum(weights)
    if total == 0:
        return 0.0

    return float(np.sum(weights[angles >= minimum_angle]) / total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

size = 8
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=8, theta_min=np.pi / 6,
)
expected = 0.009155490467795136
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

size = 6
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.13,
    dx=1.0, dy=1.0, mx=3, B=0.1, C=1.4,
    nbins=6, theta_min=np.pi / 4,
)
expected = 0.0007894726979697183
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

size = 6
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=6, theta_min=0.0,
)
expected = 1.0
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

size = 8
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=8, theta_min=np.pi / 2,
)
expected = 0.0
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

initial = np.zeros((4, 4), dtype=complex)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=4, theta_min=np.pi / 6,
)
expected = 0.0
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

size = 8
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.0,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=8, theta_min=np.pi / 6,
)
expected = 0.00910310361937183
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np

size = 8
positions = np.arange(size) - size // 2
first_grid, second_grid = np.meshgrid(positions, positions, indexing="ij")
initial = (
    np.exp(-(first_grid**2 + second_grid**2) / 16.0)
    * np.exp(0.03j * first_grid)
)
parameters = dict(
    U0=initial, P=4, k0=2.0, h=0.1, zlength=0.25,
    dx=1.0, dy=1.0, mx=1, B=0.07, C=1.1,
    nbins=8, theta_min=0.5974064166453502,
)
expected = 0.009155490467795136
""",
            "call": "wide_angle_power_fraction(**parameters)",
            "gold_call": "expected",
        },
        {
            "setup": """
import numpy as np


def invalid_cutoff():
    try:
        wide_angle_power_fraction(
            np.eye(4), 4, 2.0, 0.1, 0.25, 1.0, 1.0, 1, 0.07, 1.1,
            theta_min=-0.01,
        )
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "invalid_cutoff()",
            "gold_call": "1",
        },
    ]
