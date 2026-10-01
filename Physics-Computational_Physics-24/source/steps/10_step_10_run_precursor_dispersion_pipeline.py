"""
Chain the sub-problem functions 01-09 end to end on the circulating-fuel testbed and return the ratio of the stationary standard deviations that the exact jump process and the diffusive description predict for the selected inventory.

The whole measurement is a chain of exact linear-algebra steps: locate the circulating critical point, place the reactor a prescribed distance below it, solve the source-driven stationary state, assemble the two competing diffusion matrices there, and reduce each stationary covariance to the dispersion of one inventory. The returned ratio is the factor by which the diffusive description misstates the dispersion the exact process would show.

Returns
-------
float: the dimensionless ratio of the two predicted stationary standard deviations, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_precursor_dispersion_pipeline(subcriticality: float = 0.025,
                                      tau_core: float = 7.5,
                                      tau_excore: float = 12.5,
                                      source_rate: float = 8800.0,
                                      generation_time: float = 1.0e-3,
                                      delayed_fraction: float = 0.0065,
                                      region: str = "excore") -> float:
    """Run the full dispersion comparison on the circulating-fuel testbed.

    The delayed-group shares are ``(0.033, 0.219, 0.196, 0.395, 0.115, 0.042)``
    with decay constants ``(0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)`` inverse
    seconds, and the prompt fission multiplicity takes the values zero to five
    with abundances ``(0.027, 0.158, 0.339, 0.305, 0.133, 0.038)``.

    Parameters
    ----------
    subcriticality : float
        Dimensionless distance below the reactivity at which the circulating
        configuration is critical (subcriticality > 0).
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    region : str
        Which aggregate the dispersion is reported for: ``"excore"`` for the
        total ex-core precursor inventory, ``"core"`` for the total in-core
        precursor inventory, or ``"neutron"`` for the neutron population.

    Returns
    -------
    dispersion_ratio : float
        The stationary standard deviation the exact jump process predicts for
        the selected aggregate, divided by the standard deviation the diffusive
        description predicts for it, dimensionless, as a native Python float.

    Raises
    ------
    ValueError
        If ``subcriticality``, ``tau_core``, ``tau_excore``, ``source_rate`` or
        ``generation_time`` is not a finite number greater than zero, if
        ``delayed_fraction`` is not a finite number strictly between zero and
        one, or if ``region`` is not one of ``"excore"``, ``"core"`` or
        ``"neutron"``.
    """
    return dispersion_ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_precursor_dispersion_pipeline(subcriticality: float = 0.025,
                                              tau_core: float = 7.5,
                                              tau_excore: float = 12.5,
                                              source_rate: float = 8800.0,
                                              generation_time: float = 1.0e-3,
                                              delayed_fraction: float = 0.0065,
                                              region: str = "excore") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    for name, value in (("subcriticality", subcriticality), ("tau_core", tau_core),
                        ("tau_excore", tau_excore), ("source_rate", source_rate),
                        ("generation_time", generation_time)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
    if not (isinstance(delayed_fraction, (int, float, np.floating, np.integer))
            and not isinstance(delayed_fraction, bool) and np.isfinite(delayed_fraction)
            and 0.0 < float(delayed_fraction) < 1.0):
        raise ValueError("delayed_fraction must be a finite number strictly between zero and one")
    if region not in ("excore", "core", "neutron"):
        raise ValueError("region must be one of 'excore', 'core' or 'neutron'")

    subcriticality = float(subcriticality)
    tau_core = float(tau_core)
    tau_excore = float(tau_excore)
    source_rate = float(source_rate)
    generation_time = float(generation_time)
    delayed_fraction = float(delayed_fraction)

    # -- Nuclear data of the testbed.
    group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
    decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
    multiplicity = np.arange(6, dtype=float)
    abundance = np.array([0.027, 0.158, 0.339, 0.305, 0.133, 0.038])
    n_groups = decay_constants.size

    # -- Sub-problem 01: the two moments of the prompt fission multiplicity.
    multiplicity_moments = _oracle_evaluate_multiplicity_moments(multiplicity, abundance)

    # -- Sub-problems 03 and 04: locate the circulating critical point. The
    #    reference assembly is made at zero reactivity because the drift loss is
    #    a property of the precursor rows alone.
    reference_matrix = _oracle_build_kinetics_matrix(0.0, delayed_fraction, group_fractions,
                                                     decay_constants, generation_time,
                                                     tau_core, tau_excore)
    reactivity_loss = _oracle_compute_drift_reactivity_loss(reference_matrix, delayed_fraction,
                                                            decay_constants, generation_time)
    reactivity = float(reactivity_loss) - subcriticality

    # -- Sub-problems 02 and 03: the event rates and the drift at the operating
    #    reactivity.
    event_rates = _oracle_compute_neutron_event_rates(reactivity, delayed_fraction,
                                                      generation_time, multiplicity_moments)
    kinetics_matrix = _oracle_build_kinetics_matrix(reactivity, delayed_fraction,
                                                    group_fractions, decay_constants,
                                                    generation_time, tau_core, tau_excore)

    # -- Sub-problem 05: the source-driven stationary state the noise sits on.
    state = _oracle_solve_steady_state_populations(kinetics_matrix, source_rate)

    # -- Sub-problems 06 and 07: the two competing diffusion matrices.
    jump_diffusion = _oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions,
                                                         decay_constants, multiplicity_moments,
                                                         tau_core, tau_excore, source_rate)
    diffusive_noise = _oracle_build_diffusive_noise_matrix(state, float(event_rates[3]))

    # -- Sub-problem 08: the stationary covariance of each description.
    jump_covariance = _oracle_solve_stationary_covariance(kinetics_matrix, jump_diffusion)
    diffusive_covariance = _oracle_solve_stationary_covariance(kinetics_matrix, diffusive_noise)

    # -- Sub-problem 09: reduce both covariances to the selected aggregate.
    weights = np.zeros(1 + 2 * n_groups, dtype=float)
    if region == "neutron":
        weights[0] = 1.0
    elif region == "core":
        weights[1:1 + n_groups] = 1.0
    else:
        weights[1 + n_groups:] = 1.0

    return float(_oracle_compute_inventory_dispersion_ratio(jump_covariance,
                                                            diffusive_covariance, weights))

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
            "call": "round(run_precursor_dispersion_pipeline(), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(), 10)",
        },
        # --- Integration: the in-core inventory of the same configuration ---
        {
            "setup": """import numpy as np
region = 'core'
""",
            "call": "round(run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, region), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, region), 10)",
        },
        # --- Integration: the neutron population, where the two descriptions very
        #     nearly agree, so the ratio sits close to one ---
        {
            "setup": """import numpy as np
region = 'neutron'
""",
            "call": "round(run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, region), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, region), 10)",
        },
        # --- Integration (boundary): a shallow subcriticality, where the
        #     stationary population is large and the two descriptions converge ---
        {
            "setup": """import numpy as np
subcriticality = 0.005
""",
            "call": "round(run_precursor_dispersion_pipeline(subcriticality), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(subcriticality), 10)",
        },
        # --- Integration: a deep subcriticality, one of the sweep points the task
        #     reports, where the discrepancy is largest ---
        {
            "setup": """import numpy as np
subcriticality = 0.1
""",
            "call": "round(run_precursor_dispersion_pipeline(subcriticality), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(subcriticality), 10)",
        },
        # --- Integration (edge): a ten times slower loop with a different fuel and
        #     a weaker source, which exercises every step away from the testbed ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_precursor_dispersion_pipeline(0.04, 75.0, 125.0, 1200.0, 5.0e-4, 0.0075, 'excore'), 10)",
            "gold_call": "round(_oracle_run_precursor_dispersion_pipeline(0.04, 75.0, 125.0, 1200.0, 5.0e-4, 0.0075, 'excore'), 10)",
        },
        # --- Invalid: a non-positive subcriticality, which places the reactor at
        #     or above the circulating critical point ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_precursor_dispersion_pipeline(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_precursor_dispersion_pipeline(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an unrecognised aggregate ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, 'loop')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_precursor_dispersion_pipeline(0.025, 7.5, 12.5, 8800.0, 1.0e-3, 0.0065, 'loop')
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
