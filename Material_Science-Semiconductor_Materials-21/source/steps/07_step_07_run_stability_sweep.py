"""
Sweep the equal-Knudsen line through every transport regime and return the worst convergence rate the accelerated scheme attains on it.

A multiscale scheme is only as good as its slowest regime, so the figure of merit is not the rate at any one operating point but the largest rate anywhere in the range claimed. Sweeping the equal-Knudsen line, with all three scattering scales tied together and logarithmically spaced, walks the solver from the diffusive limit, where the unaccelerated loop stalls with a rate indistinguishable from one, to the ballistic limit, where both schemes converge in a handful of iterations and the acceleration has nothing left to buy. The accelerated rate is therefore not monotonic in the Knudsen number: it falls toward zero at both ends and peaks somewhere in the transition regime, where the closure term is neither the exact Chapman-Enskog expression it cancels against in the diffusive limit nor negligible against a strongly damped sweep. Locating that interior peak is the whole point of the sweep - sampling only the two limits, or only the diffusive side where the acceleration looks most impressive, would report a worst-case rate orders of magnitude too optimistic. Everything below the sweep itself is already in place: one call per sweep point to the routine that returns the pair of rates reaches the coefficients, the angular integrals, the plain matrix and the two accelerated operators in turn, so this step reduces to laying out the grid, driving that routine across it and taking the maximum of the accelerated column.

Returns
-------
float: the largest accelerated-scheme convergence rate over the sweep.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_stability_sweep(kn_min: float = 1e-2, kn_max: float = 1e2,
                        n_points: int = 41, C_e: float = 1.0, C_p: float = 1.0,
                        v_e: float = 1.0, v_p: float = 1.0) -> float:
    """Return the worst accelerated convergence rate on the equal-Knudsen line.

    Parameters
    ----------
    kn_min, kn_max : float
        Inclusive ends of the logarithmic sweep, finite and strictly positive
        with kn_min <= kn_max.
    n_points : int
        Number of logarithmically spaced sweep points, at least 2.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    worst_rate : float
        The largest accelerated-scheme convergence rate over the sweep, as a
        native Python float.

    Notes
    -----
    This is the final, orchestrating step: call the public function of sub-problem
    06 (``compute_convergence_rates``) at each sweep point. Import inside the body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_stability_sweep(kn_min: float = 1e-2, kn_max: float = 1e2,
                                n_points: int = 41, C_e: float = 1.0, C_p: float = 1.0,
                                v_e: float = 1.0, v_p: float = 1.0) -> float:
    import numpy as np

    scalars = [float(v) for v in (kn_min, kn_max, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in scalars):
        raise ValueError("sweep ends, capacities and speeds must be finite and > 0")
    if float(kn_min) > float(kn_max):
        raise ValueError("kn_min must not exceed kn_max")
    if not (isinstance(n_points, (int, np.integer)) and not isinstance(n_points, bool)
            and int(n_points) >= 2):
        raise ValueError("n_points must be an integer >= 2")

    # -- Logarithmic spacing with both ends inclusive.  Step 06 supplies the
    #    chained result; the direct oracle composition keeps every earlier
    #    reference step reachable and verifies that the two paths agree.
    grid = np.logspace(np.log10(float(kn_min)), np.log10(float(kn_max)), int(n_points))
    worst = -np.inf
    for kn in grid:
        kn = float(kn)
        coefficients = _oracle_compute_system_coefficients(
            kn, kn, kn, C_e, C_p, v_e, v_p)
        kn_p = float(np.asarray(coefficients, dtype=float).ravel()[0])
        moments_e = _oracle_compute_angular_moments(kn * v_e)
        moments_p = _oracle_compute_angular_moments(kn_p * v_p)
        unaccelerated = _oracle_build_unaccelerated_matrix(
            coefficients, moments_e, moments_p)
        operators = _oracle_build_accelerated_operators(
            coefficients, moments_e, moments_p, unaccelerated, kn,
            v_e=v_e, v_p=v_p)
        direct_rates = _oracle_compute_spectral_radii(unaccelerated, operators)
        rates = _oracle_compute_convergence_rates(
            kn, kn, kn, C_e=C_e, C_p=C_p, v_e=v_e, v_p=v_p)
        if not np.allclose(direct_rates, rates, rtol=1.0e-12, atol=1.0e-14):
            raise RuntimeError("direct and chained oracle convergence rates disagree")
        worst = max(worst, float(np.asarray(rates, dtype=float).ravel()[1]))
    return float(worst)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: a coarse sweep over the same span as the benchmark.
        {"setup": "import numpy as np\n",
         "call": "run_stability_sweep(1e-2, 1e2, 8)",
         "gold_call": "_oracle_run_stability_sweep(1e-2, 1e2, 8)"},
        # Edge: a sweep confined to the diffusive side of the peak.
        {"setup": "import numpy as np\n",
         "call": "run_stability_sweep(1e-3, 1e-1, 11)",
         "gold_call": "_oracle_run_stability_sweep(1e-3, 1e-1, 11)"},
        # Edge: unequal capacities and speeds over a ballistic span.
        {"setup": "import numpy as np\n",
         "call": "run_stability_sweep(1.0, 1e3, 13, C_e=0.6, C_p=1.4, v_e=1.2, v_p=0.8)",
         "gold_call": "_oracle_run_stability_sweep(1.0, 1e3, 13, C_e=0.6, C_p=1.4, v_e=1.2, v_p=0.8)"},
        # Boundary: a degenerate two-point sweep with coincident ends.
        {"setup": "import numpy as np\n",
         "call": "run_stability_sweep(2.0, 2.0, 2)",
         "gold_call": "_oracle_run_stability_sweep(2.0, 2.0, 2)"},
    ]
