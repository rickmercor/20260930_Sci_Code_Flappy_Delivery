"""
Run the complete deterministic pipeline and return the 50-percent-completeness magnitude.

Compose the preceding stages on the fixed mixed-band exposure and rotating-source injection tables, the stated spherical longitude, latitude, distance, radial speed and tangential speed, the $0.040$ arcsec/pixel scale, and the stated WCS rate.  Two search-configuration arguments carry the graded values as their defaults: the orbit inclination $i$ and the branch sign $\kappa$, so calling the function with no arguments evaluates the graded search.  The retention significance $\nu_{\rm ret}$ passed to the forced-photometry stage is fixed by the registered source and is not an argument of this function.  Reject a nonfinite inclination or a branch sign outside $\{-1,+1\}$.  The final observable is the $m_{50}$ returned by the logistic fit, so no source counts, candidate identifiers, or intermediate maps are reported as the answer.

Returns
-------
One finite native Python `float`: the deterministic 50-percent-completeness magnitude for the full coupled calculation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Final end-to-end completeness calculation."""

import numpy as np


def completeness_magnitude(inclination_rad: float = 0.0614, branch_sign: int = -1) -> float:
    """Return the fitted 50-percent recovery magnitude for the stated search.

    The forced-photometry stage is evaluated at the retention significance
    fixed by the registered source. That significance is a fixed calibration
    of the search rather than a configurable argument, so it is supplied
    internally and the same value is used for every call.

    Parameters
    ----------
    inclination_rad : float, optional
        Orbit inclination in radians used to build the signed spherical state.
    branch_sign : int, optional
        Ascending or descending tangential branch sign, either ``-1`` or ``+1``.

    Returns
    -------
    float
        The fitted 50-percent-completeness magnitude.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_completeness_magnitude(inclination_rad: float = 0.0614, branch_sign: int = -1) -> float:
    """Compose the eight preceding numerical stages using only their oracles."""
    import numpy as np

    inclination = float(inclination_rad)
    if not np.isfinite(inclination) or branch_sign not in (-1, 1):
        raise ValueError("invalid search configuration")
    exposures, injections = _oracle_observation_plan()
    state = _oracle_spherical_orbit_state(0.640, 0.061, 42.60, -0.00014, 0.00418, inclination, branch_sign)
    path = _oracle_apparent_registration_path(state, exposures, 0.040, np.array([502.65520879, -2.27302087]))
    xi, zeta = _oracle_matched_filter_statistics(path, exposures, injections)
    significance = _oracle_registered_coadd(xi, zeta, path, exposures, injections)
    candidates = _oracle_deduplicated_candidates(significance)
    recoveries = _oracle_injection_recoveries(candidates, xi, zeta, path, exposures, injections, 8.80)
    return _oracle_logistic_completeness_fit(recoveries)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge whole-instance differential cases."""
    return [
        {"setup": "import numpy as np\ninc, branch = 0.0634, -1", "call": "completeness_magnitude(inc, branch)", "gold_call": "_oracle_completeness_magnitude(inc, branch)"},
        {"setup": "import numpy as np\ninc, branch = 0.0611, -1", "call": "completeness_magnitude(inc, branch)", "gold_call": "_oracle_completeness_magnitude(inc, branch)"},
        {"setup": "import numpy as np\ninc, branch = 0.0672, 1", "call": "completeness_magnitude(inc, branch)", "gold_call": "_oracle_completeness_magnitude(inc, branch)"},
    ]
