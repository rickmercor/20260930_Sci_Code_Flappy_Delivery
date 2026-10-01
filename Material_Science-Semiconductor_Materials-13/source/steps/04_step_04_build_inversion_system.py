"""
Return the weighted linear system whose solution is the emission-rate spectral density of one binned capacitance transient or a batch of temperature-indexed transients, as the weighted exponential kernel augmented by the weighted right-hand side.

Writing the measured relaxation as a superposition of exponentials over a continuum of emission rates turns the decomposition into a Fredholm integral equation of the first kind whose kernel is the decaying exponential of the product of time and rate. Discretising the rate axis on a fixed grid replaces the integral by a matrix acting on a vector of spectral weights, one weight per grid rate, so that recovering the spectrum becomes an ordinary linear inverse problem. Because the levels of interest span several decades of emission rate, the grid is laid out geometrically rather than uniformly.




Two preparations of the data are required before the kernel can act on it. First, the quiescent capacitance must be removed, and the only estimate of it that does not presuppose the answer is the last value of the record, at which every exponential contributing to the transient has decayed as far as the measurement allows. Second, the residual must be made positive and monotonically decaying, because a spectral density is a non-negative object; for majority carrier capture the deflections are negative, so the baseline-subtracted record is negated. The shape of the transient is untouched by either operation.


The rows of the system carry very unequal relative statistical weight, since each binned point is the average of a different number of raw samples and its standard error falls as the square root of that count. This benchmark defines the row multiplier to be exactly the square root of the count. The common noise standard deviation of a given transient is deliberately not divided out: although that common factor would not change an unregularized fit, including it would rescale the residual relative to the fixed Tikhonov penalty and would therefore define a different inverse problem. Omitting the count scaling altogether would let the single-sample short-time bins, the noisiest points in the record, dictate the recovered spectrum. A temperature sweep applies the same construction independently to every record; the batched interface preserves the leading temperature axis while sharing one emission-rate grid.

Returns
-------
np.ndarray of shape (n_kept, n_rates + 1) or (n_records, n_kept, n_rates + 1), float: weighted kernel columns followed by the weighted right-hand side, in pF for the last column.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_inversion_system(binned: np.ndarray, emission_grid: np.ndarray) -> np.ndarray:
    """Return weighted exponential kernels augmented by their right-hand sides.

    Parameters
    ----------
    binned : np.ndarray
        One record of shape (n_kept, 3), or a temperature-indexed batch of
        shape (n_records, n_kept, 3), with n_kept >= 1. The final axis holds
        bin mean time in s, bin mean capacitance in pF and raw sample count.
        Every sample count must be strictly positive.
    emission_grid : np.ndarray
        Emission rates in 1/s at which the spectral density is discretised, as
        a non-empty one-dimensional array whose entries are all finite and
        strictly positive.

    Returns
    -------
    system : np.ndarray
        For one record, shape (n_kept, n_rates + 1); for a batch, shape
        (n_records, n_kept, n_rates + 1). The first n_rates entries of the
        final axis are the weighted exponential kernel and the last entry is
        the weighted, baseline-removed and sign-corrected transient. Batch
        records are processed independently and retain their input order.

    Notes
    -----
    Each row is multiplied by exactly the square root of its bin sample count.
    The common per-transient noise scale is not divided out, because doing so
    would rescale the residual relative to the fixed penalty in the next step.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return system  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_inversion_system(binned: np.ndarray, emission_grid: np.ndarray) -> np.ndarray:
    import numpy as np

    binned = np.asarray(binned, dtype=float)
    emission_grid = np.asarray(emission_grid, dtype=float)

    if (binned.ndim not in (2, 3) or binned.shape[-1] != 3
            or binned.shape[-2] < 1 or (binned.ndim == 3 and binned.shape[0] < 1)):
        raise ValueError(
            "binned must have shape (n_kept, 3) or (n_records, n_kept, 3)")
    if emission_grid.ndim != 1 or emission_grid.size < 1:
        raise ValueError("emission_grid must be a non-empty one-dimensional array")
    if np.any(emission_grid <= 0.0) or not np.all(np.isfinite(emission_grid)):
        raise ValueError("emission_grid entries must be finite and > 0")
    if np.any(binned[..., 2] <= 0.0):
        raise ValueError("bin sample counts must be strictly positive")

    times = binned[..., 0]
    values = binned[..., 1]
    counts = binned[..., 2]

    # The last binned value is the only baseline estimate that does not
    # presuppose the decomposition; negating makes the residual positive for
    # majority carrier capture.
    residual = -(values - values[..., -1, None])

    # Exact benchmark row multiplier.  This retains the relative precision of
    # the bin means but deliberately excludes the common per-transient noise
    # scale, whose inclusion would rescale the residual against the fixed
    # Tikhonov penalty used by the next step.
    weights = np.sqrt(counts)

    kernel = np.exp(-times[..., None] * emission_grid)

    system = np.empty(times.shape + (emission_grid.size + 1,), dtype=float)
    system[..., :-1] = kernel * weights[..., None]
    system[..., -1] = residual * weights

    return system

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark rate grid on a two-level binned transient ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 60)
counts = np.maximum(np.round(np.geomspace(1.0, 5000.0, 60)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
binned = np.column_stack([times, values, counts])
emission_grid = np.geomspace(0.1, 1.0e5, 150)
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Valid: two temperature records retain their leading batch axis ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 0.8, 45)
counts = np.maximum(np.round(np.geomspace(1.0, 3000.0, 45)), 1.0)
v1 = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
v2 = 204.6 - 0.68 * np.exp(-88.0 * times) - 0.08 * np.exp(-980.0 * times)
binned = np.stack([np.column_stack([times, v1, counts]),
                    np.column_stack([times, v2, counts[::-1]])])
emission_grid = np.geomspace(0.5, 2.0e4, 75)
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Valid: the right-hand side vanishes at the last bin by construction ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 40)
counts = np.full(40, 3.0)
values = 204.5 - 0.6 * np.exp(-90.0 * times)
binned = np.column_stack([times, values, counts])
emission_grid = np.geomspace(1.0, 1.0e4, 50)
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Valid: weighting is visible in the Frobenius norm of the kernel block ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 0.5, 25)
counts = np.geomspace(1.0, 900.0, 25)
values = 120.0 - 0.3 * np.exp(-15.0 * times) - 0.04 * np.exp(-800.0 * times)
binned = np.column_stack([times, values, counts])
emission_grid = np.geomspace(0.5, 5.0e3, 80)
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Boundary: one bin and one grid rate ---
        {
            "setup": """import numpy as np
binned = np.array([[0.01, 200.0, 4.0]])
emission_grid = np.array([100.0])
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Edge: a minority-carrier transient makes the right-hand side negative ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 1.0, 30)
counts = np.full(30, 10.0)
values = 150.0 + 0.2 * np.exp(-70.0 * times)
binned = np.column_stack([times, values, counts])
emission_grid = np.geomspace(1.0, 1.0e4, 40)
""",
            "call": "build_inversion_system(binned, emission_grid) + 1000.0",
            "gold_call": "_oracle_build_inversion_system(binned, emission_grid) + 1000.0",
        },
        # --- Invalid: non-positive emission rate on the grid ---
        {
            "setup": """import numpy as np
binned = np.array([[0.01, 200.0, 4.0], [0.1, 199.0, 40.0]])
def run_model():
    try:
        build_inversion_system(binned, np.array([0.0, 10.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_inversion_system(binned, np.array([0.0, 10.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of columns in the binned record ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_inversion_system(np.zeros((5, 2)), np.geomspace(1.0, 100.0, 10))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_inversion_system(np.zeros((5, 2)), np.geomspace(1.0, 100.0, 10))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
