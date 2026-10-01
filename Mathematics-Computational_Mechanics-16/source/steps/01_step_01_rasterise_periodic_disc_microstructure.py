"""
A rectangular periodic cell is sampled on a regular grid with an odd number of pixels along each axis, the sample points being x_d = i L_d / n for i = 0, ..., n - 1, so that the origin is a sample point and no sample point lies on the far edge of the cell. The odd count matters for everything that follows: the trigonometric interpolant through the samples then has an unambiguous frequency set -m, ..., m with n = 2 m + 1 and no unpaired highest frequency, which is what makes the spectral derivative used later a real symmetric operator. The microstructure is a matrix carrying circular inclusions; each disc is given by its centre, its radius and its phase label, and a sample point belongs to a disc when its periodic minimum-image distance to the centre is strictly below the radius, so discs that straddle a cell edge wrap around. Phase 1 is the matrix and labels 2 and above are inclusions; the conductivity map is the phase conductivity looked up pixel by pixel. The step also returns the pixel counts, the volume fractions and the arithmetic-mean conductivity, which is the Voigt bound the effective conductivity must not exceed.

Returns
-------
dict with integer array phase of shape (n, n) holding the one-based phase label of every sample point, float array conductivity of the same shape, tuple of native ints pixel_counts with one entry per phase, tuple of native floats volume_fractions, and native float mean_conductivity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rasterise_periodic_disc_microstructure(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
) -> dict:
    """Sample a periodic matrix-inclusion microstructure on an odd pixel grid.

    Raises
    ------
    ValueError
        If n_pixels is not an odd integer of at least three, if a cell length is not strictly positive, if a disc radius is not strictly positive, if a disc phase label is below two or above the number of phases, if a phase conductivity is not strictly positive, or if a sample point lies within 1e-9 of a disc boundary.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sample_points(n_pixels, cell_lengths):
    """Sample coordinates x_d = i L_d / n on the two axes, meshed in ij order."""
    axes = [np.arange(n_pixels) * float(cell_lengths[d]) / n_pixels for d in range(2)]
    return np.meshgrid(axes[0], axes[1], indexing="ij")


def _oracle_rasterise_periodic_disc_microstructure(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_pixels, (int, np.integer)) or int(n_pixels) < 3 or int(n_pixels) % 2 == 0:
        raise ValueError("n_pixels must be an odd integer of at least three")
    n_pixels = int(n_pixels)
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    if conductivities.size < 1 or np.any(conductivities <= 0.0):
        raise ValueError("phase conductivities must be strictly positive")
    grid = _sample_points(n_pixels, lengths)
    phase = np.ones((n_pixels, n_pixels), dtype=int)
    for disc in discs:
        centre_1, centre_2, radius, label = float(disc[0]), float(disc[1]), float(disc[2]), int(disc[3])
        if radius <= 0.0:
            raise ValueError("disc radius must be strictly positive")
        if label < 2 or label > conductivities.size:
            raise ValueError("disc phase label must lie between two and the number of phases")
        offsets = []
        for d, centre in enumerate((centre_1, centre_2)):
            delta = np.abs(grid[d] - centre) % lengths[d]
            offsets.append(np.minimum(delta, lengths[d] - delta))
        distance = np.sqrt(offsets[0] ** 2 + offsets[1] ** 2)
        if np.any(np.abs(distance - radius) < 1e-9):
            raise ValueError("a sample point lies on a disc boundary")
        phase[distance < radius] = label
    conductivity = conductivities[phase - 1]
    counts = tuple(int((phase == q + 1).sum()) for q in range(conductivities.size))
    total = float(n_pixels * n_pixels)
    return {
        "phase": phase,
        "conductivity": conductivity,
        "pixel_counts": counts,
        "volume_fractions": tuple(c / total for c in counts),
        "mean_conductivity": float(conductivity.mean()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
DISCS = [(0.27, 0.31, 0.24, 3), (0.76, 0.72, 0.21, 3), (0.78, 0.22, 0.155, 2), (0.24, 0.79, 0.135, 2)]
def summarize(out):
    return (
        out["phase"].shape, tuple(int(v) for v in out["pixel_counts"]),
        tuple(round(float(v), 12) for v in out["volume_fractions"]),
        round(float(out["mean_conductivity"]), 12),
        round(float(out["conductivity"][0, 0]), 12), int(out["phase"][12, 14]),
    )
""",
            "call": "summarize(rasterise_periodic_disc_microstructure(45, (1.0, 1.0), DISCS, np.array([1.0, 3.0, 10.0])))",
            "gold_call": "summarize(_oracle_rasterise_periodic_disc_microstructure(45, (1.0, 1.0), DISCS, np.array([1.0, 3.0, 10.0])))",
        },
        {
            "setup": """import numpy as np
DISCS = [(0.07, 1.43, 0.4, 2), (1.13, 0.52, 0.3, 4)]
def summarize(out):
    phase = out["phase"]
    return (
        tuple(int(v) for v in out["pixel_counts"]),
        int(phase[0, 0]), int(phase[0, 14]), int(phase[14, 0]), int(phase[10, 5]),
        round(float(out["mean_conductivity"]), 12),
    )
""",
            "call": "summarize(rasterise_periodic_disc_microstructure(15, (1.5, 1.5), DISCS, np.array([2.0, 0.5, 7.0, 4.0])))",
            "gold_call": "summarize(_oracle_rasterise_periodic_disc_microstructure(15, (1.5, 1.5), DISCS, np.array([2.0, 0.5, 7.0, 4.0])))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(16, (1.0, 1.0), [(0.5, 0.5, 0.2, 2)], np.array([1.0, 3.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(rasterise_periodic_disc_microstructure)",
            "gold_call": "run(_oracle_rasterise_periodic_disc_microstructure)",
        },
    ]
