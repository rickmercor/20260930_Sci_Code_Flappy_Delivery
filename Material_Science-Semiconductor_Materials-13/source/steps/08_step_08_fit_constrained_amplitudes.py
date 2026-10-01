"""
Return the individual capacitance deflections and quiescent capacitance of one binned transient or a temperature-indexed batch, obtained by independent weighted linear least-squares fits with the emission rates held fixed.

A multi-exponential transient is linear in its amplitudes and non-linear only in its decay rates. Holding the rates fixed therefore collapses the decomposition from a non-convex optimisation over twice as many parameters, with multiple local minima and a well-documented sensitivity to the starting guess, to a single linear least-squares system with a direct solution. The design matrix has one column of decaying exponentials per level plus a constant column for the quiescent capacitance, which must be fitted rather than assumed because subtracting a baseline read off the last sample would bias every amplitude by whatever has not yet decayed.




Fixing the rates is what makes weak levels recoverable at all. When rates and amplitudes are free together, a weak component can trade a shift in its rate against a change in its amplitude along a nearly flat valley of the residual surface, so the fitted pair is poorly determined and, if the assumed number of components is too large, an entirely spurious component appears between the physical ones and splits the signal three ways. Removing the rates from the parameter set halves the dimension, eliminates that valley, and turns the amplitude of even a strongly subordinate level into a well-conditioned linear estimate.




The price is that the amplitudes inherit the accuracy of the rates supplied to them. Constraining the fit to a rate that is slightly wrong forces a compensating error onto the corresponding amplitude, because the residual is minimised by rescaling an exponential of the wrong decay constant to best cover the data. That coupling is why the rates handed to this stage are recomputed from a regression across all temperatures rather than taken one temperature at a time, and it sets the ultimate accuracy of the whole procedure. As in the inversion, the rows carry the unequal statistical weight of the binned averages and must be scaled by the square root of their sample counts. In batch form, each temperature has its own row of fixed rates and its own independently fitted baseline; no coefficient is shared across records.

Returns
-------
np.ndarray of shape (n_rates + 1,) or (n_records, n_rates + 1), float: absolute capacitance deflections followed by fitted quiescent capacitance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_constrained_amplitudes(binned: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Return the level deflections and the quiescent capacitance at fixed rates.

    Parameters
    ----------
    binned : np.ndarray
        One record of shape (n_kept, 3), or a temperature-indexed batch of
        shape (n_records, n_kept, 3). The final axis holds bin mean time in s,
        bin mean capacitance in pF and raw sample count.
    rates : np.ndarray
        For one record, shape (n_rates,); for a batch, shape
        (n_records, n_rates). These are the emission rates in 1/s at which the
        exponential components are held, all strictly positive. Each record
        must have at least n_rates + 1 binned points.

    Returns
    -------
    components : np.ndarray
        For one record, shape (n_rates + 1,); for a batch, shape
        (n_records, n_rates + 1). The final axis holds the magnitude of each
        level deflection in pF, in rate order, followed by that record's fitted
        quiescent capacitance in pF.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain, including when there are
        fewer binned points than fitted coefficients (rates.size + 1).
    """
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_constrained_amplitudes(binned: np.ndarray, rates: np.ndarray) -> np.ndarray:
    import numpy as np

    binned = np.asarray(binned, dtype=float)
    rates = np.asarray(rates, dtype=float)

    single = binned.ndim == 2 and rates.ndim == 1
    batch = binned.ndim == 3 and rates.ndim == 2
    if not single and not batch:
        raise ValueError(
            "use binned/rates shapes (n_kept, 3)/(n_rates,) or "
            "(n_records, n_kept, 3)/(n_records, n_rates)")
    if binned.shape[-1] != 3 or binned.shape[-2] < 1:
        raise ValueError("the final binned axis must have length 3")
    if rates.shape[-1] < 1:
        raise ValueError("rates must contain at least one rate per record")
    if batch and (binned.shape[0] < 1 or rates.shape[0] != binned.shape[0]):
        raise ValueError("binned and rates must have the same non-zero batch size")
    if np.any(rates <= 0.0) or not np.all(np.isfinite(rates)):
        raise ValueError("rates must be finite and strictly positive")
    n_rates = rates.shape[-1]
    if binned.shape[-2] < n_rates + 1:
        raise ValueError("at least as many binned points as fitted coefficients are required")
    if np.any(binned[..., 2] <= 0.0):
        raise ValueError("bin sample counts must be strictly positive")

    def fit_one(one_binned, one_rates):
        times = one_binned[:, 0]
        values = one_binned[:, 1]
        weights = np.sqrt(one_binned[:, 2])

        # One exponential column per level plus a constant column, so the
        # quiescent capacitance is fitted rather than read off the last sample.
        columns = [np.exp(-rate * times) for rate in one_rates]
        columns.append(np.ones_like(times))
        design = np.column_stack(columns)
        solution, *_ = np.linalg.lstsq(
            design * weights[:, None], values * weights, rcond=None)

        components = np.empty(one_rates.size + 1, dtype=float)
        components[:-1] = np.abs(solution[:-1])
        components[-1] = solution[-1]
        return components

    if single:
        return fit_one(binned, rates)
    return np.stack([fit_one(binned[i], rates[i])
                     for i in range(binned.shape[0])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: exact rates recover the input deflections of two levels ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
binned = np.column_stack([times, values, counts])
rates = np.array([61.468, 1451.65])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Valid: each batched temperature uses its own rates and baseline ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 0.8, 70)
counts = np.maximum(np.round(np.geomspace(1.0, 5000.0, 70)), 1.0)
r1 = np.array([61.468, 1451.65])
r2 = np.array([92.0, 870.0])
v1 = 204.5 - 0.75 * np.exp(-r1[0] * times) - 0.05 * np.exp(-r1[1] * times)
v2 = 207.2 - 0.42 * np.exp(-r2[0] * times) - 0.13 * np.exp(-r2[1] * times)
binned = np.stack([np.column_stack([times, v1, counts]),
                    np.column_stack([times, v2, counts[::-1]])])
rates = np.stack([r1, r2])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Valid: the weak level's deflection when its rate is 5% too low ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
binned = np.column_stack([times, values, counts])
rates = np.array([61.468, 1379.07])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Valid: the fitted quiescent capacitance of a noisy record ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = (204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
          + rng.standard_normal(89) * 0.0032 / np.sqrt(counts))
binned = np.column_stack([times, values, counts])
rates = np.array([61.468, 1451.65])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Boundary: a single exponential component ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-4, 1.0, 50)
counts = np.geomspace(1.0, 2000.0, 50)
values = 150.0 - 0.4 * np.exp(-200.0 * times)
binned = np.column_stack([times, values, counts])
rates = np.array([200.0])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Edge: three components, one of them outside the acquisition window ---
        {
            "setup": """import numpy as np
times = np.geomspace(1.0e-5, 1.0, 89)
counts = np.maximum(np.round(np.geomspace(1.0, 8000.0, 89)), 1.0)
values = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
binned = np.column_stack([times, values, counts])
rates = np.array([61.468, 1451.65, 4.0e5])
""",
            "call": "fit_constrained_amplitudes(binned, rates) + 1000.0",
            "gold_call": "_oracle_fit_constrained_amplitudes(binned, rates) + 1000.0",
        },
        # --- Invalid: a non-positive fixed rate ---
        {
            "setup": """import numpy as np
binned = np.column_stack([np.geomspace(1.0e-4, 1.0, 20), np.full(20, 100.0), np.full(20, 5.0)])
def run_model():
    try:
        fit_constrained_amplitudes(binned, np.array([0.0, 100.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fit_constrained_amplitudes(binned, np.array([0.0, 100.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer binned points than fitted coefficients ---
        {
            "setup": """import numpy as np
binned = np.column_stack([np.array([0.01, 0.1]), np.array([100.0, 99.5]), np.array([2.0, 20.0])])
def run_model():
    try:
        fit_constrained_amplitudes(binned, np.array([10.0, 100.0, 1000.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fit_constrained_amplitudes(binned, np.array([10.0, 100.0, 1000.0]))
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
