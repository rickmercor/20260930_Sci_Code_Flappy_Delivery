"""
Accumulate the weighted phase points of both families into a histogram along the order parameter. Each family arrives as a flat list of interior positions together with the weight of the trajectory each position came from. Reactant-side points enter with their weight unchanged, while barrier-side points enter with theirs scaled by the fraction of reactant-side trajectories terminating at the inner interface. Bins are of equal width and span the stated bounds, with a point lying on a shared edge assigned to the upper bin.

Once every trajectory carries a single weight, a phase-space average is more convenient to write as a weighted sum over phase points than as a sum over paths, because histogramming assigns the same weight to every slice of a given trajectory. The resulting per-point weights are not normalised, which is harmless for a free energy that will be reported on a shifted scale. The asymmetry between the two families is the signature of an unbounded reactant coordinate: without it the barrier-side trajectories are over-represented relative to the reactant-side ones and the whole profile tilts. Where the phase points sit along the coordinate is decided upstream, so this step is a pure accumulator and the placement rule lives in exactly one place.

Returns
-------
np.ndarray of shape (n_bins,) and dtype float, the accumulated weight in each bin
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def conditional_density_histogram(plus_values: np.ndarray, plus_weights: np.ndarray,
                                  minus_values: np.ndarray, minus_weights: np.ndarray,
                                  end_fraction: float, lower_bound: float,
                                  upper_bound: float, n_bins: int) -> np.ndarray:
    '''Unnormalised conditional density along the order parameter.

    Parameters
    ----------
    plus_values : np.ndarray
        (A,) order-parameter positions of every barrier-side interior phase point.
    plus_weights : np.ndarray
        (A,) weight of the trajectory each barrier-side position came from.
    minus_values : np.ndarray
        (B,) order-parameter positions of every reactant-side interior phase point.
    minus_weights : np.ndarray
        (B,) weight of the trajectory each reactant-side position came from.
    end_fraction : float
        Fraction of reactant-side trajectories terminating at the inner interface.
    lower_bound : float
        Lower edge of the histogram.
    upper_bound : float
        Upper edge of the histogram.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    density : np.ndarray
        (n_bins,) accumulated weight per bin.
    
    Raises
    ------
    ValueError
        the barrier-side positions and weights must be 1D and equal in length.
        the reactant-side positions and weights must be 1D and equal in length.
        n_bins must be a positive integer.
        the histogram bounds must be finite.
        lower_bound must lie below upper_bound.
        end_fraction must be finite and non-negative.
    '''
    return density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_conditional_density_histogram(plus_values: np.ndarray, plus_weights: np.ndarray,
                                          minus_values: np.ndarray, minus_weights: np.ndarray,
                                          end_fraction: float, lower_bound: float,
                                          upper_bound: float, n_bins: int) -> np.ndarray:
    """Reference implementation."""
    plus_values = np.asarray(plus_values, dtype=float)
    plus_weights = np.asarray(plus_weights, dtype=float)
    minus_values = np.asarray(minus_values, dtype=float)
    minus_weights = np.asarray(minus_weights, dtype=float)
    if plus_values.ndim != 1 or plus_weights.ndim != 1 or plus_values.size != plus_weights.size:
        raise ValueError("the barrier-side positions and weights must be 1D and equal in length")
    if minus_values.ndim != 1 or minus_weights.ndim != 1 or minus_values.size != minus_weights.size:
        raise ValueError("the reactant-side positions and weights must be 1D and equal in length")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 1:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(float(lower_bound)) or not np.isfinite(float(upper_bound)):
        raise ValueError("the histogram bounds must be finite")
    if float(lower_bound) >= float(upper_bound):
        raise ValueError("lower_bound must lie below upper_bound")
    if not np.isfinite(float(end_fraction)) or float(end_fraction) < 0.0:
        raise ValueError("end_fraction must be finite and non-negative")
    nb = int(n_bins)
    edges = np.linspace(float(lower_bound), float(upper_bound), nb + 1)
    density = np.zeros(nb, dtype=float)
    for values, weights, scale in ((plus_values, plus_weights, float(end_fraction)),
                                   (minus_values, minus_weights, 1.0)):
        for value, weight in zip(values, weights):
            slot = int(np.searchsorted(edges, float(value), side="right")) - 1
            density[min(max(slot, 0), nb - 1)] += scale * float(weight)
    return density

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nplus_values = np.array([-0.6, -0.1, 0.35, 0.8, 0.95])\nplus_weights = np.array([0.20, 0.20, 0.35, 0.35, 0.10])\nminus_values = np.array([-1.5, -1.2, -1.05])\nminus_weights = np.array([0.5, 0.25, 0.25])\nend_fraction = 0.45\n",
            "call": "conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.6, 1.0, 8)",
            "gold_call": "_oracle_conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.6, 1.0, 8)"
        },
        {
            "setup": "import numpy as np\n# boundary: a bounded coordinate, where the two families enter on equal footing\nplus_values = np.array([-0.5, 0.0, 0.5])\nplus_weights = np.array([1.0, 1.0, 2.0])\nminus_values = np.array([-1.2, -1.1])\nminus_weights = np.array([1.5, 0.5])\nend_fraction = 1.0\n",
            "call": "conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.4, 1.0, 4)",
            "gold_call": "_oracle_conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.4, 1.0, 4)"
        },
        {
            "setup": "import numpy as np\n# edge: points sitting exactly on shared edges go to the upper bin\nplus_values = np.array([-0.35, 0.30])\nplus_weights = np.array([1.0, 1.0])\nminus_values = np.array([-1.0])\nminus_weights = np.array([2.0])\nend_fraction = 0.5\n",
            "call": "conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.0, 1.0, 4)",
            "gold_call": "_oracle_conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.0, 1.0, 4)"
        },
        {
            "setup": "import numpy as np\n# edge: one wide bin collects every phase point of both families\nplus_values = np.array([0.1, 0.4]); plus_weights = np.array([1.0, 1.0])\nminus_values = np.array([-1.2]); minus_weights = np.array([3.0])\nend_fraction = 0.75\n",
            "call": "conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.5, 1.0, 1)",
            "gold_call": "_oracle_conditional_density_histogram(plus_values, plus_weights, minus_values, minus_weights, end_fraction, -1.5, 1.0, 1)"
        },
        {
            "setup": "import numpy as np\npv = np.array([0.1, 0.4]); pw = np.array([1.0])\nmv = np.array([-1.2]); mw = np.array([3.0])\ndef run(f):\n    try:\n        f(pv, pw, mv, mw, 0.5, -1.5, 1.0, 4); return 0   # mismatched barrier-side lengths\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(conditional_density_histogram)",
            "gold_call": "run(_oracle_conditional_density_histogram)"
        },
        {
            "setup": "import numpy as np\npv = np.array([0.1]); pw = np.array([1.0])\nmv = np.array([-1.2]); mw = np.array([3.0])\ndef run(f):\n    try:\n        f(pv, pw, mv, mw, 0.5, 1.0, -1.5, 4); return 0   # bounds the wrong way round\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run(conditional_density_histogram)",
            "gold_call": "run(_oracle_conditional_density_histogram)"
        }
    ]
