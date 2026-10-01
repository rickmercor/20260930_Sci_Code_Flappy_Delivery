"""
Chain the sub-problem functions 01-09 end to end over a band of wave numbers and return the aggregate relative departure of a conventional asymptotic closure from the wave-number-dependent one.

At every wave number the anchored closure and the conventional one give different three-moment operators, and the same initial perturbation therefore evolves differently under each. Aggregating the per-wave-number departures with equal weight makes the comparison a statement about the band rather than about one favourable mode.

Returns
-------
float: the dimensionless band-aggregated relative departure of the benchmark closure, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_closure_fidelity_pipeline(wavenumbers: np.ndarray = (0.2, 0.3, 0.4, 0.5, 0.6),
                                  amplitude: float = 0.02,
                                  time_step: float = 0.005,
                                  step_count: int = 8000,
                                  benchmark: str = "HP") -> float:
    """Run the whole closure-fidelity comparison over a band of wave numbers.

    At each wave number the reference closure is the one anchored on the exact
    kinetic least-damped root pair, and the closure under assessment is the
    named conventional asymptotic member. Both are started from a Maxwellian
    equilibrium with a cosine density ripple of the given amplitude. This
    function uses cosine amplitudes internally: the initial normalised density
    and pressure perturbations both equal amplitude, and velocity is zero.
    These are twice the corresponding positive-k coefficients in a two-sided
    Fourier expansion. The common factor cancels in the relative departure.

    Parameters
    ----------
    wavenumbers : np.ndarray
        One-dimensional sequence of wave numbers divided by the Debye wave
        number, each at least 0.1. This is the root solver's admissible input
        bound; successful anchoring additionally requires a numerically
        nonsingular two-root fit. Exceptionally weak damping near the lower
        bound can make that fit numerically singular even for admissible inputs.
    amplitude : float
        Amplitude of the initial cosine density ripple, strictly positive.
    time_step : float
        Time increment in inverse plasma frequencies, strictly positive.
    step_count : int
        Number of increments to take, at least one.
    benchmark : str
        Which conventional asymptotic closure is assessed: ``"HP"``, ``"R31"``
        or ``"R30"``.

    Returns
    -------
    aggregate_deviation : float
        The root mean square, over the supplied wave numbers, of the relative
        departure of the benchmark closure's field history from the anchored
        closure's field history, dimensionless, as a native Python float.

    Raises
    ------
    ValueError
        If ``wavenumbers`` is not a non-empty one-dimensional array of finite
        entries of at least 0.1, if ``amplitude`` or ``time_step`` is not a
        finite number greater than zero, if ``step_count`` is not an integer
        greater than zero, if ``benchmark`` is not one of the three supported
        labels, if a located kinetic root fails to satisfy the dispersion
        relation to within 1e-8, or if the two-root anchoring system fails the
        numerical nonsingularity check of sub-problem 03.
    """
    return aggregate_deviation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_closure_fidelity_pipeline(wavenumbers: np.ndarray = (0.2, 0.3, 0.4, 0.5, 0.6),
                                          amplitude: float = 0.02,
                                          time_step: float = 0.005,
                                          step_count: int = 8000,
                                          benchmark: str = "HP") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    band = np.asarray(wavenumbers, dtype=float)
    if band.ndim != 1 or band.size < 1:
        raise ValueError("wavenumbers must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(band)) or np.any(band < 0.1):
        raise ValueError("wavenumbers entries must be finite and at least 0.1")
    for name, value in (("amplitude", amplitude), ("time_step", time_step)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
    if (isinstance(step_count, bool)
            or not isinstance(step_count, (int, np.integer))
            or int(step_count) < 1):
        raise ValueError("step_count must be an integer greater than zero")
    if benchmark not in ("HP", "R31", "R30"):
        raise ValueError("benchmark must be one of 'HP', 'R31' or 'R30'")

    amplitude = float(amplitude)
    time_step = float(time_step)
    step_count = int(step_count)

    # -- The reference pipeline calls the _oracle_ twin of every earlier step, so
    #    that it stays independent of the submitted public functions. Chaining
    #    through the public names instead would put the same defect on both sides
    #    of the comparison and let an incorrect submission pass. The public chain
    #    is graded separately by integration test 3, which composes the public
    #    steps 01-09 by hand and checks them against this reference.

    # -- Sub-problem 04: the conventional closure is wave-number independent, so
    #    its parameters are fixed once for the whole band.
    asymptotic = _oracle_build_asymptotic_pade_coefficients(benchmark)
    # -- Sub-problem 05.
    benchmark_parameters = _oracle_evaluate_closure_parameters(asymptotic)

    # -- The cosine density ripple of a Maxwellian carries an equal pressure
    #    ripple and no flow.
    initial_state = np.array([amplitude, 0.0, amplitude], dtype=complex)

    deviations = []
    for wavenumber in band:
        wavenumber = float(wavenumber)

        # -- Sub-problem 02: the least-damped kinetic root, and its mirror image
        #    across the imaginary axis, which the Maxwellian parity supplies.
        root = _oracle_solve_kinetic_root(wavenumber)
        phase_speeds = np.array([root, -np.conj(root)], dtype=complex)

        # -- Sub-problem 01: the exact response at those two phase speeds, which
        #    is what the approximant is anchored on. On a genuine root it must
        #    equal minus the squared wave number.
        response_values = _oracle_evaluate_kinetic_response(phase_speeds)
        residual = float(np.max(np.abs(np.asarray(response_values, dtype=complex)
                                       + wavenumber ** 2)))
        if not np.isfinite(residual) or residual > 1.0e-8:
            raise ValueError("the located phase speeds do not satisfy the "
                             "dispersion relation to within 1e-8")

        # -- Sub-problems 03 and 05: the anchored closure at this wave number.
        matched = _oracle_solve_matched_pade_coefficients(phase_speeds, response_values)
        matched_parameters = _oracle_evaluate_closure_parameters(matched)

        # -- Sub-problems 06, 07 and 08 for each closure in turn.
        reference_field = _oracle_compute_field_amplitude_history(
            _oracle_integrate_moment_history(
                _oracle_build_moment_evolution_matrix(wavenumber, matched_parameters),
                initial_state, time_step, step_count),
            wavenumber)
        test_field = _oracle_compute_field_amplitude_history(
            _oracle_integrate_moment_history(
                _oracle_build_moment_evolution_matrix(wavenumber, benchmark_parameters),
                initial_state, time_step, step_count),
            wavenumber)

        # -- Sub-problem 09.
        deviations.append(float(_oracle_compute_relative_field_deviation(reference_field,
                                                                         test_field)))

    return float(np.sqrt(np.mean(np.asarray(deviations, dtype=float) ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the reported configuration (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_closure_fidelity_pipeline(), 9)",
            "gold_call": "round(_oracle_run_closure_fidelity_pipeline(), 9)",
        },
        # --- Integration: the same band assessed against the member that trades
        #     one adiabatic order for one fluid order ---
        {
            "setup": """import numpy as np
benchmark = 'R31'
""",
            "call": "round(run_closure_fidelity_pipeline((0.2, 0.3, 0.4, 0.5, 0.6), 0.02, 0.005, 8000, benchmark), 9)",
            "gold_call": "round(_oracle_run_closure_fidelity_pipeline((0.2, 0.3, 0.4, 0.5, 0.6), 0.02, 0.005, 8000, benchmark), 9)",
        },
        # --- Integration: a single short-wavelength mode with a much larger
        #     amplitude and a coarser step, which exercises every step away from
        #     the reported configuration ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_closure_fidelity_pipeline((0.9,), 1.0, 0.02, 1500, 'R30'), 9)",
            "gold_call": "round(_oracle_run_closure_fidelity_pipeline((0.9,), 1.0, 0.02, 1500, 'R30'), 9)",
        },
        # --- Integration (boundary): a very short run, where the two closures have
        #     barely separated and the departure is small ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_closure_fidelity_pipeline((0.4,), 0.02, 0.005, 5), 9)",
            "gold_call": "round(_oracle_run_closure_fidelity_pipeline((0.4,), 0.02, 0.005, 5), 9)",
        },
        # --- Integration (edge): a band extended beyond the reported one on both
        #     sides, which forces the root tracking over a wider range. The lower
        #     edge stops where the anchoring system is still well conditioned;
        #     below about a fifth of the Debye wave number the two anchors merge
        #     onto the real axis and no longer determine the coefficients. ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_closure_fidelity_pipeline((0.22, 0.45, 1.2), 0.02, 0.005, 2000), 9)",
            "gold_call": "round(_oracle_run_closure_fidelity_pipeline((0.22, 0.45, 1.2), 0.02, 0.005, 2000), 9)",
        },
        # --- Invalid: a non-positive wave number in the band ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_closure_fidelity_pipeline((0.2, 0.0))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_closure_fidelity_pipeline((0.2, 0.0))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an unrecognised benchmark closure ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_closure_fidelity_pipeline((0.4,), 0.02, 0.005, 100, 'R32')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_closure_fidelity_pipeline((0.4,), 0.02, 0.005, 100, 'R32')
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
