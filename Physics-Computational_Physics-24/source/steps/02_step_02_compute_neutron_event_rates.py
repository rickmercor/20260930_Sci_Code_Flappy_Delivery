"""
Convert the point-kinetic constants and the fission multiplicity moments into the per-neutron event rates of the equivalent stochastic process and into the amplitude of the neutron noise.

The deterministic point-kinetic constants fix, but do not display, the elementary per-neutron rates at which a neutron induces a fission or is lost to capture and leakage; those rates are recovered by demanding that the mean production and removal of the jump process reproduce the point-kinetic balance. The same rates, weighted by the moments of the fission multiplicity, set the amplitude of the single Brownian term that the diffusive description places on the neutron equation.

Returns
-------
np.ndarray of shape (4,), float: fission rate, loss rate, mean delayed yield and neutron noise coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_neutron_event_rates(reactivity: float, delayed_fraction: float,
                                generation_time: float, multiplicity_moments: np.ndarray
                                ) -> np.ndarray:
    """Return the per-neutron event rates and the neutron noise amplitude.

    Parameters
    ----------
    reactivity : float
        Dimensionless reactivity of the configuration; may be negative.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    multiplicity_moments : np.ndarray
        Array of shape (2,) holding the mean prompt fission multiplicity and the
        mean square net prompt gain of a fission, as returned by sub-problem 01.
        The mean multiplicity must be strictly positive.

    Returns
    -------
    rates : np.ndarray
        Array of shape (4,) holding, in order, the per-neutron fission rate in
        inverse seconds, the per-neutron capture-and-leakage rate in inverse
        seconds, the dimensionless mean delayed-neutron yield of a fission, and
        the scalar coefficient in inverse square-root seconds that multiplies
        the square root of the neutron population in the diffusion term of the
        neutron equation.

    Raises
    ------
    ValueError
        If ``reactivity`` is not a finite number; if ``delayed_fraction`` is not
        a finite number strictly between zero and one; if ``generation_time`` is
        not a finite number greater than zero; if ``multiplicity_moments`` is not
        a finite array of shape (2,) whose first entry is strictly positive or
        whose second entry is negative; or if the resulting capture-and-leakage
        rate is negative, which places the configuration above prompt critical
        and leaves the stochastic process without a stationary state.
    """
    return rates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_neutron_event_rates(reactivity: float, delayed_fraction: float,
                                        generation_time: float,
                                        multiplicity_moments: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    reactivity = _finite("reactivity", reactivity)
    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    if generation_time <= 0.0:
        raise ValueError("generation_time must be greater than zero")

    moments = np.asarray(multiplicity_moments, dtype=float)
    if moments.shape != (2,):
        raise ValueError("multiplicity_moments must be an array of shape (2,)")
    if not np.all(np.isfinite(moments)):
        raise ValueError("multiplicity_moments must contain only finite entries")
    mean_multiplicity, mean_square_net_gain = float(moments[0]), float(moments[1])
    if mean_multiplicity <= 0.0:
        raise ValueError("the mean prompt multiplicity must be greater than zero")
    if mean_square_net_gain < 0.0:
        raise ValueError("the mean square net prompt gain must be non-negative")

    # Prompt production per neutron per second is the prompt share of the
    # inverse generation time, so the fission rate follows from dividing it by
    # the mean number of prompt neutrons a fission releases.
    fission_rate = (1.0 - delayed_fraction) / (mean_multiplicity * generation_time)
    # Total removal per neutron per second is fixed by the reactivity; what is
    # not fission is capture and leakage.
    loss_rate = (1.0 - reactivity) / generation_time - fission_rate
    if loss_rate < 0.0:
        raise ValueError("the capture-and-leakage rate must be non-negative")

    # The delayed yield is fixed by requiring the fission rate to reproduce the
    # delayed production of the point-kinetic equations.
    delayed_yield = delayed_fraction / (generation_time * fission_rate)

    # Fission and removal both contribute to the variance of the neutron
    # population; fission does so through the mean square of its net gain.
    noise_coefficient = np.sqrt(mean_square_net_gain * fission_rate + loss_rate)

    return np.array([fission_rate, loss_rate, delayed_yield, noise_coefficient],
                    dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the operating point of the testbed (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = -0.022820635778950265
delayed_fraction = 0.0065
generation_time = 1.0e-3
multiplicity_moments = np.array([2.473, 3.391])
""",
            "call": "sig(compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
            "gold_call": "sig(_oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
        },
        # --- Boundary: exactly zero reactivity ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = 0.0
delayed_fraction = 0.0065
generation_time = 1.0e-3
multiplicity_moments = np.array([2.473, 3.391])
""",
            "call": "sig(compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
            "gold_call": "sig(_oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
        },
        # --- Valid: a slow, strongly subcritical configuration with a different
        #     fuel, where the loss rate dominates the fission rate ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = -0.35
delayed_fraction = 0.0021
generation_time = 5.0e-2
multiplicity_moments = np.array([3.1, 6.4])
""",
            "call": "sig(compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e2)",
            "gold_call": "sig(_oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e2)",
        },
        # --- Edge: a multiplicity distribution concentrated on one neutron, so
        #     that fission contributes nothing to the neutron noise ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = -0.01
delayed_fraction = 0.007
generation_time = 2.0e-3
multiplicity_moments = np.array([1.0, 0.0])
""",
            "call": "sig(compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
            "gold_call": "sig(_oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments), 1.0e3)",
        },
        # --- Invalid: reactivity far above prompt critical, which drives the
        #     capture-and-leakage rate negative ---
        {
            "setup": """import numpy as np
reactivity = 0.9
delayed_fraction = 0.0065
generation_time = 1.0e-3
multiplicity_moments = np.array([2.473, 3.391])
def run_model():
    try:
        compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a delayed fraction of one, which leaves no prompt neutrons ---
        {
            "setup": """import numpy as np
reactivity = -0.01
delayed_fraction = 1.0
generation_time = 1.0e-3
multiplicity_moments = np.array([2.473, 3.391])
def run_model():
    try:
        compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_neutron_event_rates(reactivity, delayed_fraction, generation_time, multiplicity_moments)
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
