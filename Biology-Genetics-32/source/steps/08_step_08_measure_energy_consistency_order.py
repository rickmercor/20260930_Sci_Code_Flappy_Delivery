"""
Refine the partition over a sequence of step counts and fit the observed decay order of the energy-balance defect against the step size.

Two errors of a variational time integrator behave quite differently and must not be conflated. The error in the state carries no general guarantee: it depends on the loading resolving whatever structure the exact response has, and it can sit at a plateau, independent of the step size, whenever the discretisation is too coarse to see a regime change. The defect of the energy balance, by contrast, is bounded unconditionally, because the local truncation error is entirely the gap between the true work and the frozen-state work over one step, which is controlled by how far the state moves within that step. Fitting the defect against the step size on a logarithmic scale therefore certifies the integrator in the only sense a theorem underwrites, and a fitted order that departs from the expected one is a symptom in the implementation rather than in the model.

Returns
-------
float, the least-squares slope of the natural logarithm of the energy-balance defect against the natural logarithm of the step size, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def measure_energy_consistency_order(step_counts, k: float = 1.0, a: float = 0.15,
                                     rho: float = 0.10, q_initial: float = -0.15,
                                     T: float = 1.0, S_max: float = 5.0,
                                     ell_inf: float = 0.5, lam: float = 1.0,
                                     b: float = None) -> float:
    """Fit the decay order of the energy-balance defect under refinement.

    For every step count the loading path is rebuilt on the corresponding
    uniform partition, the globally minimising trajectory is integrated over
    it, and the accumulated energy-balance defect is recorded. The reported
    order is the slope of an ordinary least-squares fit of the natural
    logarithm of the defect against the natural logarithm of the step size.

    Parameters
    ----------
    step_counts : sequence of int
        Two or more distinct partition sizes, each at least one.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node of every partition.
    T, S_max, ell_inf, lam : float
        Settings of the stimulus protocol and its interaction potential.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    order : float
        The fitted decay order of the defect in the step size, as a native
        Python float.

    Raises
    ------
    ValueError
        If step_counts holds fewer than two partition sizes, holds a repeated
        size, or holds an entry below one; if T, S_max, ell_inf, lam, k or
        rho is not a finite number strictly greater than zero; if a is not a
        finite non-negative number; if b is neither None nor a finite
        non-negative number; if q_initial is not a finite number; or if the
        energy-balance defect fails to be positive at some partition of the
        sweep, leaving no order to fit.
    """
    return order  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import numpy as np
def _oracle_measure_energy_consistency_order(step_counts, k: float = 1.0, a: float = 0.15,
                                             rho: float = 0.10, q_initial: float = -0.15,
                                             T: float = 1.0, S_max: float = 5.0,
                                             ell_inf: float = 0.5, lam: float = 1.0,
                                             b: float = None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    counts = [int(value) for value in np.asarray(step_counts, dtype=int).ravel().tolist()]
    if len(counts) < 2:
        raise ValueError("step_counts must hold at least two partition sizes")
    if len(set(counts)) != len(counts):
        raise ValueError("step_counts must hold distinct partition sizes")
    if min(counts) < 1:
        raise ValueError("every entry of step_counts must be at least one")
    for name, value in (("T", T), ("S_max", S_max), ("ell_inf", ell_inf),
                        ("lam", lam), ("k", k), ("rho", rho)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number strictly greater than zero")
    if not (isinstance(a, (int, float, np.floating, np.integer))
            and np.isfinite(float(a)) and float(a) >= 0.0):
        raise ValueError("a must be a finite non-negative number")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and np.isfinite(float(b)) and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")
    if not (isinstance(q_initial, (int, float, np.floating, np.integer))
            and not isinstance(q_initial, bool) and np.isfinite(float(q_initial))):
        raise ValueError("q_initial must be a finite number")

    window, amplitude, saturation, rate = float(T), float(S_max), float(ell_inf), float(lam)
    curvature, half_gap, threshold = float(k), float(a), float(rho)
    far_gap = half_gap if b is None else float(b)
    level = 0.5 * curvature * (half_gap ** 2 - far_gap ** 2)

    # Sub-problems 01, 05 and 07 are written out here rather than imported, so
    # that the refinement sweep this step reports is reproducible from this
    # file alone and cannot be changed by whichever copy of an earlier step
    # happens to be in scope.
    def _path(count):
        time = window * np.arange(count + 1, dtype=float) / float(count)
        stimulus = 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * time / window))
        return saturation * (1.0 - np.exp(-rate * stimulus))

    def _advance(previous, load):
        def _objective(state):
            centre = -half_gap if state <= 0.0 else far_gap
            offset = 0.0 if state <= 0.0 else level
            return (0.5 * curvature * (state - centre) ** 2 + offset - state * load
                    + threshold * abs(state - previous))

        knots = sorted(set((0.0, previous)))
        edges = [-np.inf] + knots + [np.inf]
        candidates = list(knots)
        for lower, upper in zip(edges[:-1], edges[1:]):
            if not upper > lower:
                continue
            if np.isneginf(lower):
                probe = upper - 1.0
            elif np.isposinf(upper):
                probe = lower + 1.0
            else:
                probe = 0.5 * (lower + upper)
            centre = -half_gap if probe <= 0.0 else far_gap
            sign = 1.0 if probe > previous else -1.0
            stationary = centre + (load - threshold * sign) / curvature
            if np.isfinite(lower):
                stationary = max(stationary, lower)
            if np.isfinite(upper):
                stationary = min(stationary, upper)
            candidates.append(float(stationary))

        values = [_objective(state) for state in candidates]
        best = min(values)
        tolerance = 1.0e-13 * (1.0 + abs(best))
        tied = [state for state, value in zip(candidates, values) if value <= best + tolerance]
        tied.sort(key=lambda state: (abs(state - previous), state))
        return float(tied[0])

    def _defect(potential):
        states = np.empty(potential.size, dtype=float)
        states[0] = float(q_initial)
        for index in range(1, potential.size):
            states[index] = _advance(float(states[index - 1]), float(potential[index]))
        centre = np.where(states <= 0.0, -half_gap, far_gap)
        offset = np.where(states <= 0.0, 0.0, level)
        stored = 0.5 * curvature * (states - centre) ** 2 + offset - states * potential
        work = float(np.sum(-states[:-1] * np.diff(potential)))
        dissipated = threshold * float(np.sum(np.abs(np.diff(states))))
        return float((stored[0] + work) - (stored[-1] + dissipated))

    sizes, defects = [], []
    for count in counts:
        defect = _defect(_path(count))
        if not defect > 0.0:
            raise ValueError("the energy-balance defect must be positive to admit a fitted order")
        sizes.append(window / float(count))
        defects.append(defect)

    # An ordinary least-squares slope on the logarithmic scale; the intercept
    # is the fitted constant of the estimate and is not reported.
    log_size = np.log(np.asarray(sizes, dtype=float))
    log_defect = np.log(np.asarray(defects, dtype=float))
    design = np.column_stack((log_size, np.ones_like(log_size)))
    slope, _ = np.linalg.lstsq(design, log_defect, rcond=None)[0]

    return float(slope)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark landscape refined over four partitions ---
        {
            "setup": """import numpy as np
counts = [50, 100, 200, 400]
""",
            "call": "float(1.0e6 * measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "float(1.0e6 * _oracle_measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Valid: a stiffer, wider landscape under a weaker stimulus ---
        {
            "setup": """import numpy as np
counts = [60, 120, 240]
""",
            "call": "float(1.0e6 * measure_energy_consistency_order(counts, 3.0, 0.30, 0.05, -0.30, 1.0, 2.0, 0.9, 1.0))",
            "gold_call": "float(1.0e6 * _oracle_measure_energy_consistency_order(counts, 3.0, 0.30, 0.05, -0.30, 1.0, 2.0, 0.9, 1.0))",
        },
        # --- Valid: the asymmetric benchmark landscape refined dyadically ---
        {
            "setup": """import numpy as np
counts = [100, 200, 400, 800]
""",
            "call": "float(1.0e6 * measure_energy_consistency_order(counts, 1.07, 0.130, 0.0970, -0.130, 1.0, 5.0, 0.5, 1.0, 0.075))",
            "gold_call": "float(1.0e6 * _oracle_measure_energy_consistency_order(counts, 1.07, 0.130, 0.0970, -0.130, 1.0, 5.0, 0.5, 1.0, 0.075))",
        },
        # --- Boundary: the smallest admissible sweep, two partitions ---
        {
            "setup": """import numpy as np
counts = [80, 160]
""",
            "call": "float(1.0e6 * measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "float(1.0e6 * _oracle_measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Edge: partitions listed out of order, which must not change the fit ---
        {
            "setup": """import numpy as np
counts = [400, 50, 200, 100]
""",
            "call": "float(1.0e6 * measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "float(1.0e6 * _oracle_measure_energy_consistency_order(counts, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Invalid: a sweep of a single partition, which fixes no slope ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        measure_energy_consistency_order([100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_measure_energy_consistency_order([100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a repeated partition size ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        measure_energy_consistency_order([100, 100, 200])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_measure_energy_consistency_order([100, 100, 200])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a partition of no intervals, which tabulates nothing to
        # measure a defect on ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        measure_energy_consistency_order([0, 100])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_measure_energy_consistency_order([0, 100])
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
