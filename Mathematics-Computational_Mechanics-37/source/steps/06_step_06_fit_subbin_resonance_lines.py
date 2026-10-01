"""
A receptor_record of finite length cannot say where a resonance is to better than the spacing of its frequency bins, and that spacing is coarse next to the accuracy the solver itself achieves. This stage closes that gap. It is the measuring instrument of the task: everything before it produces a receptor_record, and everything after it works with the numbers this stage reads off.

The three displacement components are transformed in time and collapsed into one curve by taking the root of the summed squared moduli, which puts every family on the same axis. Only the non-negative half of the frequency axis is kept, its bins spaced at the reciprocal of the receptor_record duration, with neither windowing nor zero padding applied.

Each resonance is then found twice over. First coarsely: given a target frequency, the line belonging to it is the bin of largest amplitude within a stated fractional window of that target, which is wide enough to hold a line that the discretisation has moved and narrow enough never to reach its neighbours. Second, and this is the point of the stage, finely: the bin index alone is a quantised estimate, so the position is sharpened from the shape of the peak. Take the natural logarithms of the amplitude at the winning bin and at its two immediate neighbours and pass a parabola through those three values. Writing $a$, $b$ and $c$ for those logarithms in order, the vertex of that parabola sits at

$$delta = (a - c) / [2 (a - 2 b + c)]$$

bins from the winning bin, and the sharpened frequency is $(k + delta)$ times the bin spacing with $k$ the winning bin index. The denominator is the discrete curvature of the log amplitude, which must be strictly negative for the parabola to have a maximum at all; a positive or vanishing curvature means the three amplitudes do not describe a peak and the sharpening is refused. A parabola through logarithms rather than through the amplitudes themselves is chosen because a resonance line in an unwindowed transform is close to Gaussian in shape near its crown, and a Gaussian is exactly a parabola once the logarithm is taken.

The refinement is defined only when the winning bin has a neighbour on each side and all three amplitudes are strictly positive.

Returns
-------
dict, the collapsed spectrum with the sharpened line of every target resonance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_subbin_resonance_lines(
    receptor_record: np.ndarray,
    time_step: float,
    target_frequencies: np.ndarray,
    search_fraction: float,
) -> dict:
    """Collapse the receptor_record into one spectrum and sharpen the line belonging to each target frequency.

    Parameters
    ----------
    receptor_record : np.ndarray
        Receptor displacement samples, shape (n_samples, 3).
    time_step : float
        Spacing between samples in second.
    target_frequencies : np.ndarray
        Frequencies in hertz near which lines are sought, one dimension, every entry above zero.
    search_fraction : float
        Half-width of the search window as a fraction of the target, above zero and below one.

    Returns
    -------
    dict
        Under the keys frequencies, intensity, bin_spacing, line_index, line_offset, line_curvature, line_intensity and refined_frequency.

    Raises
    ------
    ValueError
        When receptor_record is not shaped (n_samples, 3) with two samples or more, when a recorded value fails to be finite, when time_step fails to sit above zero, when target_frequencies is not a one-dimensional array of at least one finite entry above zero, when search_fraction leaves the open interval (0, 1), when a search window admits no frequency bin, or when a winning bin cannot be sharpened, whether because it sits at an end of the spectrum, because one of the three amplitudes fails to sit above zero, or because the curvature of the log amplitude fails to be negative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _log_vertex(low, middle, high):
    """Vertex position in bins, and the discrete curvature, of the parabola through three log amplitudes."""
    left, centre, right = np.log(low), np.log(middle), np.log(high)
    curvature = left - 2.0 * centre + right
    return float(0.5 * (left - right) / curvature), float(curvature)


def _oracle_fit_subbin_resonance_lines(
    receptor_record: np.ndarray,
    time_step: float,
    target_frequencies: np.ndarray,
    search_fraction: float,
) -> dict:
    """Reference implementation."""
    samples = np.asarray(receptor_record, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 2:
        raise ValueError("record wants shape (n_samples, 3) and two samples at the very least")
    if not np.isfinite(samples).all():
        raise ValueError("no recorded displacement may be infinite or undefined")
    interval = float(time_step)
    if interval <= 0.0:
        raise ValueError("dt wants a value above zero")
    targets = np.asarray(target_frequencies, dtype=float)
    if targets.ndim != 1 or targets.size < 1:
        raise ValueError("target_frequencies wants a one-dimensional array of at least one entry")
    if not np.isfinite(targets).all() or targets.min() <= 0.0:
        raise ValueError("every target frequency wants a finite value above zero")
    fraction = float(search_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("search_fraction wants a value inside the open interval (0, 1)")

    n_samples = samples.shape[0]
    frequencies = np.fft.rfftfreq(n_samples, d=interval)
    intensity = np.sqrt(np.sum(np.abs(np.fft.rfft(samples, axis=0)) ** 2, axis=1))
    spacing = float(frequencies[1] - frequencies[0])

    index = np.zeros(targets.size, dtype=np.int64)
    offset = np.zeros(targets.size)
    curvature = np.zeros(targets.size)
    crown = np.zeros(targets.size)
    sharpened = np.zeros(targets.size)
    for slot, target in enumerate(targets):
        inside = (frequencies >= target * (1.0 - fraction)) & (frequencies <= target * (1.0 + fraction))
        if not inside.any():
            raise ValueError("the window around %g hertz admits no frequency bin" % target)
        winner = int(np.argmax(np.where(inside, intensity, -np.inf)))
        if not 1 <= winner <= intensity.size - 2:
            raise ValueError("a winning bin sits at an end of the spectrum and cannot be sharpened")
        neighbourhood = intensity[winner - 1:winner + 2]
        if neighbourhood.min() <= 0.0:
            raise ValueError("a winning bin and both neighbours want strictly positive amplitude")
        low, middle, high = (float(entry) for entry in neighbourhood)
        vertex, bend = _log_vertex(low, middle, high)
        if bend >= 0.0:
            raise ValueError("the three log amplitudes give a parabola that does not open downwards")
        index[slot] = winner
        offset[slot] = vertex
        curvature[slot] = bend
        crown[slot] = middle
        sharpened[slot] = (winner + vertex) * spacing
    return {
        "frequencies": frequencies,
        "intensity": intensity,
        "bin_spacing": spacing,
        "line_index": index,
        "line_offset": offset,
        "line_curvature": curvature,
        "line_intensity": crown,
        "refined_frequency": sharpened,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# cosines seated exactly on bins 2, 3 and 4 of a 33 sample receptor_record: the unnormalised transform of
# A cos(2 pi k n / N) reaches A N / 2 at bin k, so the amplitudes are 16.5, 33 and 8.25, the
# curvature of their logarithms is log(16.5) - 2 log(33) + log(8.25) = -3 log(2), and the vertex
# of the parabola sits exactly one sixth of a bin below bin 3
LENGTH, STEP = 33, 1.0e-9
positions = np.arange(LENGTH)
TRACE = np.zeros((LENGTH, 3))
TRACE[:, 0] = (2.0 * np.cos(2 * np.pi * 3 * positions / LENGTH)
               + np.cos(2 * np.pi * 2 * positions / LENGTH)
               + 0.5 * np.cos(2 * np.pi * 4 * positions / LENGTH))
TARGET = np.array([3.0 / (LENGTH * STEP)])
def digest(out):
    spacing = 1.0 / (LENGTH * STEP)
    return (int(out["line_index"][0]), round(float(out["intensity"][2]), 10),
            round(float(out["line_intensity"][0]), 10), round(float(out["intensity"][4]), 10),
            round(float(out["line_offset"][0]), 12),
            round(float(out["line_curvature"][0] / np.log(2.0)), 12),
            round(float(out["refined_frequency"][0] / spacing), 12))
""",
            "call": "digest(fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.2))",
            "gold_call": "digest(_oracle_fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.2))",
        },
        {
            "setup": """import numpy as np
# a clean sine whose frequency deliberately sits between two bins, so the sharpening has work to do
STEP, LENGTH = 2.0e-9, 4001
clock = np.arange(LENGTH) * STEP
SPACING = 1.0 / (LENGTH * STEP)
TRUE = (37.0 + 0.3) * SPACING
TRACE = np.stack([np.cos(2 * np.pi * TRUE * clock), np.zeros(LENGTH), np.zeros(LENGTH)], axis=1)
TARGET = np.array([TRUE])
def digest(out):
    coarse = abs(out["line_index"][0] * SPACING - TRUE)
    fine = abs(out["refined_frequency"][0] - TRUE)
    # the sharpened position must beat the bin centre; on an unwindowed receptor_record the parabola
    # through the logarithms is biased and recovers roughly half of the offset, not all of it
    return (int(out["line_index"][0]), int(fine < coarse), round(float(fine / coarse), 6),
            round(float(out["line_offset"][0]), 8), round(float(out["line_curvature"][0]), 8),
            round(float(out["refined_frequency"][0]), 4), round(float(out["bin_spacing"]), 6),
            out["frequencies"].shape, out["intensity"].shape)
""",
            "call": "digest(fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.05))",
            "gold_call": "digest(_oracle_fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.05))",
        },
        {
            "setup": """import numpy as np
# several targets at once, with a drift near zero frequency that no window may capture
STEP, LENGTH = 2.0e-9, 8001
clock = np.arange(LENGTH) * STEP
TRACE = np.zeros((LENGTH, 3))
TRACE[:, 0] = np.exp(-clock / 4e-6)
for f, a in ((0.62e6, 1.0), (1.27e6, 0.6), (1.88e6, 0.4), (2.51e6, 0.25)):
    TRACE[:, 0] += a * np.cos(2 * np.pi * f * clock)
TARGET = np.array([0.63e6, 1.29e6, 1.90e6, 2.54e6])
def digest(out):
    return (tuple(int(k) for k in out["line_index"]),
            tuple(np.round(out["line_offset"], 8)),
            tuple(np.round(out["refined_frequency"], 3)),
            tuple(np.round(out["line_curvature"], 6)),
            int(np.all(out["line_curvature"] < 0.0)),
            round(float(out["intensity"][0] / out["line_intensity"][0]), 6))
""",
            "call": "digest(fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.05))",
            "gold_call": "digest(_oracle_fit_subbin_resonance_lines(TRACE, STEP, TARGET, 0.05))",
        },
        {
            "setup": """import numpy as np
CLEAN = np.zeros((1000, 3))
CLEAN[:, 0] = np.cos(np.arange(1000) * 0.1)
SPOILT = CLEAN.copy()
SPOILT[17, 2] = float("nan")
CONSTANT = np.zeros((64, 3))
CONSTANT[:, 0] = 1.0
def verdict(fn, trace=CLEAN, targets=np.array([1.6e7]), fraction=0.05, step=1.0e-9):
    try:
        fn(trace, step, targets, fraction)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": ("(verdict(fit_subbin_resonance_lines, trace=SPOILT), "
                     "verdict(fit_subbin_resonance_lines, targets=np.array([1.0e4]), fraction=0.001), "
                     "verdict(fit_subbin_resonance_lines, targets=np.array([-1.0e6])), "
                     "verdict(fit_subbin_resonance_lines, fraction=1.5), "
                     "verdict(fit_subbin_resonance_lines, trace=CONSTANT, targets=np.array([1.0e5]), fraction=0.9), "
                     "verdict(fit_subbin_resonance_lines))"),
            "gold_call": ("(verdict(_oracle_fit_subbin_resonance_lines, trace=SPOILT), "
                          "verdict(_oracle_fit_subbin_resonance_lines, targets=np.array([1.0e4]), fraction=0.001), "
                          "verdict(_oracle_fit_subbin_resonance_lines, targets=np.array([-1.0e6])), "
                          "verdict(_oracle_fit_subbin_resonance_lines, fraction=1.5), "
                          "verdict(_oracle_fit_subbin_resonance_lines, trace=CONSTANT, targets=np.array([1.0e5]), fraction=0.9), "
                          "verdict(_oracle_fit_subbin_resonance_lines))"),
        },
    ]
