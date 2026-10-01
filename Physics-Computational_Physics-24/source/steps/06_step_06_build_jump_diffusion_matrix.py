"""
Assemble the diffusion matrix of the exact continuous-time Markov jump process at a given state, accumulating over every elementary event the product of its rate with the outer product of its state increment.

Every elementary event of the jump process moves the state by a fixed integer increment, and events that move more than one component at once, a fission that emits prompt and delayed products together, a decay in the core that converts a precursor into a neutron, a transfer that moves a precursor between regions, correlate those components. Because all event rates are affine in the state, the second-moment equations of this process close exactly on the state at which the rates are evaluated.

Returns
-------
np.ndarray of shape (1 + 2 * n_groups, 1 + 2 * n_groups), float: the symmetric diffusion matrix in inverse seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_jump_diffusion_matrix(state: np.ndarray, event_rates: np.ndarray,
                                group_fractions: np.ndarray, decay_constants: np.ndarray,
                                multiplicity_moments: np.ndarray, tau_core: float,
                                tau_excore: float, source_rate: float) -> np.ndarray:
    """Assemble the diffusion matrix of the exact jump process at a given state.

    Parameters
    ----------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the neutron population, the
        in-core precursor populations and the ex-core precursor populations in
        the ordering of sub-problem 03; every entry is non-negative.
    event_rates : np.ndarray
        Array of shape (4,) as returned by sub-problem 02: the per-neutron
        fission rate, the per-neutron capture-and-leakage rate, the mean delayed
        yield of a fission and the neutron noise coefficient. The last entry is
        not used here.
    group_fractions : np.ndarray
        One-dimensional array of the delayed-group shares of the total delayed
        fraction, summing to one.
    decay_constants : np.ndarray
        One-dimensional array of the decay constants of the delayed groups in
        inverse seconds, in the same group order.
    multiplicity_moments : np.ndarray
        Array of shape (2,) holding the mean prompt fission multiplicity and the
        mean square net prompt gain of a fission, as returned by sub-problem 01.
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).

    Returns
    -------
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of shape
        (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds, the
        instantaneous covariance rate of the jump process at the supplied state.

    Raises
    ------
    ValueError
        If ``state`` is not a one-dimensional finite array of non-negative
        entries whose length is one more than twice the number of delay groups,
        if ``event_rates`` is not a finite array of shape (4,) whose first three
        entries are non-negative, if ``multiplicity_moments`` is not a finite
        array of shape (2,), if ``group_fractions`` and ``decay_constants`` are
        not non-empty one-dimensional finite arrays of equal length with
        strictly positive entries, if the ``group_fractions`` entries do not sum
        to one within 1e-10, or if ``tau_core``, ``tau_excore`` or
        ``source_rate`` is not a finite number greater than zero.
    """
    return diffusion_matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_jump_diffusion_matrix(state: np.ndarray, event_rates: np.ndarray,
                                        group_fractions: np.ndarray,
                                        decay_constants: np.ndarray,
                                        multiplicity_moments: np.ndarray,
                                        tau_core: float, tau_excore: float,
                                        source_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _positive(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number greater than zero")
        return float(value)

    tau_core = _positive("tau_core", tau_core)
    tau_excore = _positive("tau_excore", tau_excore)
    source_rate = _positive("source_rate", source_rate)

    shares = np.asarray(group_fractions, dtype=float)
    lambdas = np.asarray(decay_constants, dtype=float)
    for name, array in (("group_fractions", shares), ("decay_constants", lambdas)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} entries must be strictly positive")
    if shares.size != lambdas.size:
        raise ValueError("group_fractions and decay_constants must have equal length")
    if abs(float(shares.sum()) - 1.0) > 1e-10:
        raise ValueError("group_fractions entries must sum to one")

    rates = np.asarray(event_rates, dtype=float)
    moments = np.asarray(multiplicity_moments, dtype=float)
    if rates.shape != (4,) or not np.all(np.isfinite(rates)):
        raise ValueError("event_rates must be a finite array of shape (4,)")
    if np.any(rates[:3] < 0.0):
        raise ValueError("the first three event_rates entries must be non-negative")
    if moments.shape != (2,) or not np.all(np.isfinite(moments)):
        raise ValueError("multiplicity_moments must be a finite array of shape (2,)")

    populations = np.asarray(state, dtype=float)
    n_groups = shares.size
    size = 1 + 2 * n_groups
    if populations.ndim != 1 or populations.size != size:
        raise ValueError("state must be one-dimensional of length 1 + 2 * n_groups")
    if not np.all(np.isfinite(populations)) or np.any(populations < 0.0):
        raise ValueError("state entries must be finite and non-negative")

    fission_rate, loss_rate, delayed_yield = float(rates[0]), float(rates[1]), float(rates[2])
    mean_multiplicity, mean_square_net_gain = float(moments[0]), float(moments[1])
    neutrons = float(populations[0])
    core = populations[1:1 + n_groups]
    excore = populations[1 + n_groups:]
    core_slice = slice(1, 1 + n_groups)
    excore_slice = slice(1 + n_groups, size)

    matrix = np.zeros((size, size), dtype=float)

    # Capture-and-leakage and the external source each move the neutron count by
    # one and nothing else.
    matrix[0, 0] += neutrons * loss_rate + source_rate

    # A fission moves the neutron count by its net prompt gain and simultaneously
    # creates delayed precursors, so it correlates the neutron row with every
    # precursor row. The per-group delayed yield is Poisson, so its own second
    # moment carries an extra diagonal piece.
    group_yield = delayed_yield * shares
    matrix[0, 0] += neutrons * fission_rate * mean_square_net_gain
    cross = neutrons * fission_rate * (mean_multiplicity - 1.0) * group_yield
    matrix[0, core_slice] += cross
    matrix[core_slice, 0] += cross
    matrix[core_slice, core_slice] += neutrons * fission_rate * (
        np.outer(group_yield, group_yield) + np.diag(group_yield))

    # A decay inside the core destroys a precursor and creates a neutron, so it
    # is anti-correlated between the two; a decay in the loop only destroys.
    core_decay = core * lambdas
    matrix[0, 0] += float(core_decay.sum())
    for j in range(n_groups):
        matrix[0, 1 + j] -= core_decay[j]
        matrix[1 + j, 0] -= core_decay[j]
        matrix[1 + j, 1 + j] += core_decay[j]
        matrix[1 + n_groups + j, 1 + n_groups + j] += excore[j] * lambdas[j]

    # A transfer moves one precursor from one region to the other; both
    # directions carry the same increment up to a sign, so their rates add.
    transfer = core / tau_core + excore / tau_excore
    for j in range(n_groups):
        matrix[1 + j, 1 + j] += transfer[j]
        matrix[1 + n_groups + j, 1 + n_groups + j] += transfer[j]
        matrix[1 + j, 1 + n_groups + j] -= transfer[j]
        matrix[1 + n_groups + j, 1 + j] -= transfer[j]

    return 0.5 * (matrix + matrix.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the testbed stationary state (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([352.0, 2492.439075512, 7445.082213733, 2379.199688901,
                  2224.164601468, 208.067409903, 30.604978237,
                  3596.593179121, 8983.507949306, 1660.872378464,
                  778.360315460, 22.739607572, 1.320603284])
event_rates = np.array([401.738778811, 621.081856968, 0.016179667841, 44.535132827])
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
multiplicity_moments = np.array([2.473, 3.391])
tau_core, tau_excore, source_rate = 7.5, 12.5, 8800.0
""",
            "call": "sig(build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e6)",
            "gold_call": "sig(_oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e6)",
        },
        # --- Valid: a two-group loop with a very different rate balance ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([120.0, 900.0, 340.0, 210.0, 55.0])
event_rates = np.array([80.0, 260.0, 0.031, 20.5])
group_fractions = np.array([0.62, 0.38])
decay_constants = np.array([0.017, 0.72])
multiplicity_moments = np.array([3.1, 6.4])
tau_core, tau_excore, source_rate = 2.0, 9.0, 640.0
""",
            "call": "sig(build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e4)",
            "gold_call": "sig(_oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e4)",
        },
        # --- Boundary: an empty precursor inventory, leaving only the neutron
        #     events to contribute ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([500.0, 0.0, 0.0, 0.0, 0.0])
event_rates = np.array([401.738778811, 621.081856968, 0.016179667841, 44.535132827])
group_fractions = np.array([0.4, 0.6])
decay_constants = np.array([0.02, 0.5])
multiplicity_moments = np.array([2.473, 3.391])
tau_core, tau_excore, source_rate = 7.5, 12.5, 8800.0
""",
            "call": "sig(build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e6)",
            "gold_call": "sig(_oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e6)",
        },
        # --- Edge: a single group with no neutrons present, so only the precursor
        #     decay and transfer events survive ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([0.0, 4200.0, 3100.0])
event_rates = np.array([401.738778811, 621.081856968, 0.016179667841, 44.535132827])
group_fractions = np.array([1.0])
decay_constants = np.array([0.0124])
multiplicity_moments = np.array([2.473, 3.391])
tau_core, tau_excore, source_rate = 7.5, 12.5, 8800.0
""",
            "call": "sig(build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e3)",
            "gold_call": "sig(_oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate), 1.0e3)",
        },
        # --- Invalid: a negative population, which no jump process can occupy ---
        {
            "setup": """import numpy as np
state = np.array([352.0, -12.0, 40.0, 30.0, 5.0])
event_rates = np.array([401.738778811, 621.081856968, 0.016179667841, 44.535132827])
group_fractions = np.array([0.4, 0.6])
decay_constants = np.array([0.02, 0.5])
multiplicity_moments = np.array([2.473, 3.391])
tau_core, tau_excore, source_rate = 7.5, 12.5, 8800.0
def run_model():
    try:
        build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a state whose length does not match the group count ---
        {
            "setup": """import numpy as np
state = np.array([352.0, 40.0, 30.0, 5.0])
event_rates = np.array([401.738778811, 621.081856968, 0.016179667841, 44.535132827])
group_fractions = np.array([0.4, 0.6])
decay_constants = np.array([0.02, 0.5])
multiplicity_moments = np.array([2.473, 3.391])
tau_core, tau_excore, source_rate = 7.5, 12.5, 8800.0
def run_model():
    try:
        build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_jump_diffusion_matrix(state, event_rates, group_fractions, decay_constants, multiplicity_moments, tau_core, tau_excore, source_rate)
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
