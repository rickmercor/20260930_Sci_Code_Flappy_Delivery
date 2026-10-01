"""
Return the updated upper endpoint of one spline interpolation domain from the

invariants the forward solve sampled, using a smooth upper-tail statistic and a

relaxed update.

The interpolation domain of a non-equilibrium constitutive function cannot be

fixed in advance, because the invariant range the branch visits is decided by its

internal evolution rather than by the applied deformation. The activated range is

therefore read off the samples produced by the current forward solve. Taking the

raw sample maximum makes the update non-differentiable and hypersensitive to a

single outlying increment, so the activated value is instead a smooth upper-tail

statistic of the samples governed by alpha_smooth > 0: it tends to the sample

maximum as alpha_smooth grows and stays differentiable and tolerant of isolated

extremes at moderate values. Whichever form that statistic takes, it has to be

evaluated in a numerically stable arrangement so that the sample range cannot

overflow. The endpoint is then not set onto the activated value outright but

relaxed towards it under eta in (0, 1], which lets the domain expand or contract

gradually over the outer iterations instead of chasing the samples of a single

solve. The left endpoint is untouched, since it is fixed by the constitutive

normalisation of the invariant.

Returns
-------
float, the updated upper endpoint as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def adapt_domain_endpoint(x_end: float, samples, alpha_smooth: float,
                          eta: float) -> float:
    """Return the relaxed update of one interpolation domain endpoint.

    Parameters
    ----------
    x_end : float
        Current upper endpoint of the interpolation domain.
    samples : array-like of shape (n,)
        Invariant values sampled by the current forward solve.
    alpha_smooth : float
        Smoothing parameter of the upper-tail statistic, strictly positive.
    eta : float
        Relaxation factor of the endpoint update, in (0, 1].

    Returns
    -------
    x_end_new : float
        Updated upper endpoint of the interpolation domain.

    Raises
    ------
    ValueError
        If samples is empty, if alpha_smooth is not strictly positive, or if eta
        lies outside (0, 1].
    """
    return x_end_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adapt_domain_endpoint(x_end: float, samples, alpha_smooth: float,
                                  eta: float) -> float:
    """Reference implementation."""
    values = np.atleast_1d(np.asarray(samples, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be one dimensional and non-empty")
    if not float(alpha_smooth) > 0.0:
        raise ValueError("alpha_smooth must be strictly positive")
    if not 0.0 < float(eta) <= 1.0:
        raise ValueError("eta must lie in (0, 1]")

    shift = float(values.max())
    x_act = shift + float(np.log(np.mean(np.exp(float(alpha_smooth)
                                                * (values - shift))))) / float(alpha_smooth)
    return float((1.0 - float(eta)) * float(x_end) + float(eta) * x_act)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

# Deviatoric stress invariant sampled at every increment of the first forward
# solve of the prompt configuration.
_J_TAU_SAMPLES = np.array([
    0.027404751567, 0.092621156883, 0.176010389886, 0.264288891691,
    0.348396808165, 0.422717768448, 0.484373582100, 0.532519125080,
    0.567673569032, 0.591152905499, 0.604640443955, 0.609895897646,
    0.608580572874, 0.602168056285, 0.591911541177, 0.578845012014,
    0.563802275980, 0.547443569482, 0.530283693342, 0.512718462668,
    0.273347354791, 0.130806279818, 0.049796804042, 0.010342445273,
    0.000039345825, 0.010767951861, 0.037150426152, 0.075243421030,
    0.122002872048, 0.175298253692, 0.233561109169, 0.295646827770,
    0.360788531202, 0.428546941560, 0.498773882622, 0.571585197065,
    0.647341668113, 0.726638180733, 0.810302338214, 0.899404430426
])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario ---
        {
            "setup": """x_end = 0.10
samples = _J_TAU_SAMPLES.copy()
""",
            "call": "round(float(adapt_domain_endpoint(x_end, samples, 5.0, 0.5)), 12)",
            "gold_call": "round(float(_oracle_adapt_domain_endpoint(x_end, samples, 5.0, 0.5)), 12)",
        },
        # --- Boundary case: a single sample and full relaxation, where the
        # endpoint moves onto the sample itself ---
        {
            "setup": """samples = np.array([0.7])
""",
            "call": "round(float(adapt_domain_endpoint(0.10, samples, 5.0, 1.0)), 12)",
            "gold_call": "round(float(_oracle_adapt_domain_endpoint(0.10, samples, 5.0, 1.0)), 12)",
        },
        # --- Edge case: a non-positive smoothing parameter must raise ValueError ---
        {
            "setup": """samples = _J_TAU_SAMPLES.copy()

def run_model():
    try:
        adapt_domain_endpoint(0.10, samples, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_adapt_domain_endpoint(0.10, samples, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
