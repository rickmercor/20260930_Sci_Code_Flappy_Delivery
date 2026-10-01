"""
Retrieve the comprehensive model and nominal parameters from Supplementary Note S2, including the estimated IC rate and the lifetime beside Fig. S26. Retrieve the ROKS T₂ and T₁ energies from Table S9. Return the values in consistent SI units and the triplet gap in eV.

Use Eqs. S5–S9 and the Supplementary Note S2 parameter set, not main-text Table 2. Treat approximate printed values as exact nominal inputs for the hypothetical benchmark. The observed tail lifetime is not the intrinsic triplet lifetime.

Returns
-------
np.ndarray, shape (8,), float64: [tau_s_seconds, i, u, c, theta_eV, lambda_per_second, gap_eV, kappa_per_second]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def prepare_inputs(source: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """Convert the retrieved source quantities to the kinetic benchmark.

    Parameters
    ----------
    source : array-like, shape (8,)
        [tau_S1_ns, k_ISC, k_rISC2, k_IC, kBT_meV,
         tail_lifetime_ms, E_T2_eV, E_T1_eV]. Rates are in s^-1.
        Retrieve the Note S2 / Fig. S26 / Table S9 quantities, not
        main-text Table 2. Printed approximate values are nominal inputs.
    kappa : float
        Independently known T1 radiative rate in s^-1, strictly positive.

    Returns
    -------
    np.ndarray
        Shape (8,): [tau_s, i, u, c, theta, lambda, gap, kappa],
        in seconds, s^-1, and eV. lambda is the inverse tail lifetime
        and gap is E_T2-E_T1. Do not identify lambda with intrinsic d.

    Raises
    ------
    ValueError
        If input is not real numeric and finite, source has the wrong
        shape, any of its first six values or kappa is nonpositive,
        E_T2<E_T1, lambda<=kappa, or converted results are nonfinite.
    """
    return np.empty(8, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_prepare_inputs(source, kappa=1.0):
    import numpy as np
    try:
        if np.iscomplexobj(source) or np.iscomplexobj(kappa):
            raise ValueError("real inputs required")
        s = np.asarray(source, dtype=float)
        k = float(kappa)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real numeric inputs required") from exc
    if s.shape != (8,) or not np.all(np.isfinite(s)) or not np.isfinite(k):
        raise ValueError("finite source shape (8,) and finite kappa required")
    if np.any(s[:6] <= 0) or k <= 0 or s[6] < s[7]:
        raise ValueError("invalid lifetimes, rates, thermal energy or gap")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = np.array([s[0]*1e-9, s[1], s[2], s[3], s[4]*1e-3,
                        1000.0/s[5], s[6]-s[7], k], dtype=float)
    if not np.all(np.isfinite(out)) or out[5] <= k:
        raise ValueError("finite converted inputs with lambda>kappa required")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "import numpy as np\ns = np.array([6., 6e7, 7.5e5, 30., 25.6, 120., 2.953, 2.912])",
            "call": "prepare_inputs(s, 1.0)",
            "gold_call": "_oracle_prepare_inputs(s, 1.0)"
        },
        {
            "setup": "import numpy as np\ns = np.array([10., 2e7, 1e5, 10., 30., 50., 2.8, 2.8])",
            "call": "prepare_inputs(s, 1.0)",
            "gold_call": "_oracle_prepare_inputs(s, 1.0)"
        },
        {
            "setup": "import numpy as np\ns = np.array([1., 1e6, 2e4, 0.1, 10., 1., 1.9, 1.7])",
            "call": "prepare_inputs(s, 999.0)",
            "gold_call": "_oracle_prepare_inputs(s, 999.0)"
        },
        {
            "setup": "import numpy as np\ns = np.array([6., 6e7, 7.5e5, 30., 25.6, 120., 2.953, 2.912])\ndef _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(prepare_inputs, s, 10.0)",
            "gold_call": "_expect_value_error(_oracle_prepare_inputs, s, 10.0)"
        }
    ]
