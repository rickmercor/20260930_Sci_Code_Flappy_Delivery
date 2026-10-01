#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np 

def build_loading_path(n_steps: int, T: float = 1.0, S_max: float = 5.0,
                               ell_inf: float = 0.5, lam: float = 1.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError("n_steps must be an integer")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be at least one")
    for name, value in (("T", T), ("S_max", S_max), ("ell_inf", ell_inf), ("lam", lam)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number strictly greater than zero")

    n_steps = int(n_steps)
    window = float(T)
    amplitude = float(S_max)
    ceiling = float(ell_inf)
    rate = float(lam)

    # The nodes are laid out from the index rather than by accumulation, so the
    # last node lands on T exactly and the spacing carries no drift.
    time = window * np.arange(n_steps + 1, dtype=float) / float(n_steps)

    # One raised-cosine excursion: S(0) = S(T) = 0 with a single peak S_max at
    # mid-window, and a vanishing rate at both ends of the window.
    stimulus = 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * time / window))

    # The interaction potential saturates towards its ceiling, so an ever
    # larger stimulus buys ever less drive on the chromatin state.
    potential = ceiling * (1.0 - np.exp(-rate * stimulus))

    return np.column_stack((time, stimulus, potential)).astype(float)

import numpy as np 

def evaluate_double_well(q, k: float = 1.0, a: float = 0.15,
                                 b: float = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    states = np.asarray(q, dtype=float).ravel()
    if states.size < 1:
        raise ValueError("q must hold at least one state")
    if not np.all(np.isfinite(states)):
        raise ValueError("q must hold finite states only")
    if not (isinstance(k, (int, float, np.floating, np.integer))
            and np.isfinite(float(k)) and float(k) > 0.0):
        raise ValueError("k must be a finite number strictly greater than zero")
    if not (isinstance(a, (int, float, np.floating, np.integer))
            and np.isfinite(float(a)) and float(a) >= 0.0):
        raise ValueError("a must be a finite non-negative number")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and np.isfinite(float(b)) and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    curvature = float(k)
    repressed = float(a)
    active = repressed if b is None else float(b)

    # The neutral state is assigned to the repressed well, so the centre of
    # the occupied well is -a on {q <= 0} and +b on {q > 0}. This convention
    # only matters at the corner itself, where both branches agree in value.
    centre = np.where(states <= 0.0, -repressed, active)

    # Continuity of the energy at the corner raises the far well by exactly
    # the amount by which the two bottoms are unequally placed; the repressed
    # well is the zero of energy, so only the far branch carries the level.
    level = 0.5 * curvature * (repressed ** 2 - active ** 2)
    offset = np.where(states <= 0.0, 0.0, level)

    energy = 0.5 * curvature * (states - centre) ** 2 + offset
    force = curvature * (states - centre)

    return np.vstack((energy, force)).astype(float)

import numpy as np

def solve_incremental_step(q_prev: float, ell: float, k: float = 1.0,
                                   a: float = 0.15, rho: float = 0.10,
                                   b: float = None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_prev", q_prev), ("ell", ell), ("k", k), ("a", a), ("rho", rho)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(k) <= 0.0:
        raise ValueError("k must be strictly greater than zero")
    if float(a) < 0.0:
        raise ValueError("a must be non-negative")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and not isinstance(b, bool) and np.isfinite(float(b))
                              and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    previous = float(q_prev)
    load = float(ell)
    curvature, half_gap, threshold = float(k), float(a), float(rho)
    far_gap = half_gap if b is None else float(b)

    # Continuity of the free energy at the barrier fixes the level of the far
    # well; the near well is the zero of energy.
    level = 0.5 * curvature * (half_gap ** 2 - far_gap ** 2)

    def _objective(state):
        centre = -half_gap if state <= 0.0 else far_gap
        offset = 0.0 if state <= 0.0 else level
        return (0.5 * curvature * (state - centre) ** 2 + offset - state * load
                + threshold * abs(state - previous))

    # The objective is smooth except where the landscape has its corner and
    # where the dissipation term changes sign; those two abscissae cut the
    # line into at most three pieces, each carrying a convex quadratic.
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
    # A strict comparison would let rounding decide a genuine tie between the
    # two basins, so near-optimal candidates are gathered and the tie is
    # settled by the stated convention rather than by floating-point noise.
    tolerance = 1.0e-13 * (1.0 + abs(best))
    tied = [state for state, value in zip(candidates, values) if value <= best + tolerance]
    tied.sort(key=lambda state: (abs(state - previous), state))

    return float(tied[0])

import numpy as np

def return_map_step(q_prev: float, ell: float, rho: float = 0.10,
                            stiffness: float = 1.0, offset: float = 0.15,
                            curvature: float = 0.0, tol: float = 1.0e-13,
                            max_iter: int = 100) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_prev", q_prev), ("ell", ell), ("rho", rho),
                        ("stiffness", stiffness), ("offset", offset),
                        ("curvature", curvature), ("tol", tol)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if float(stiffness) <= 0.0:
        raise ValueError("stiffness must be strictly greater than zero")
    if float(curvature) < 0.0:
        raise ValueError("curvature must be non-negative")
    if float(tol) <= 0.0:
        raise ValueError("tol must be strictly greater than zero")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be an integer of at least one")

    previous, load, threshold = float(q_prev), float(ell), float(rho)
    linear, shift, anharmonic = float(stiffness), float(offset), float(curvature)
    tolerance, budget = float(tol), int(max_iter)

    def _force(state):
        return linear * (state + shift) + anharmonic * np.sinh(state)

    def _slope(state):
        return linear + anharmonic * np.cosh(state)

    # Elastic predictor: the drive the frozen state feels under the new load.
    trial = load - _force(previous)
    if abs(trial) <= threshold:
        return float(previous)

    # Corrector: place the drive exactly on the boundary of the elastic range,
    # on the side the violation points to.
    direction = 1.0 if trial > 0.0 else -1.0
    target = load - direction * threshold

    # Bracket the root by marching away from the previous state in the
    # direction of the increment; the branch force is strictly increasing, so
    # one-sided expansion always succeeds.
    lower, upper = previous, previous
    span = max(1.0, abs(previous))
    for _ in range(200):
        if direction > 0.0:
            upper = previous + span
            if _force(upper) >= target:
                break
        else:
            lower = previous - span
            if _force(lower) <= target:
                break
        span *= 2.0
    else:
        raise ValueError("the branch force could not be bracketed around the target")

    # Safeguarded Newton: take the Newton step when it stays inside the
    # bracket and makes progress, and bisect otherwise.
    state = 0.5 * (lower + upper)
    for _ in range(budget):
        residual = _force(state) - target
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            upper = state
        else:
            lower = state
        step = state - residual / _slope(state)
        if not (lower < step < upper):
            step = 0.5 * (lower + upper)
        if abs(step - state) <= tolerance * (1.0 + abs(state)):
            state = step
            break
        state = step

    return float(state)

import numpy as np 

def integrate_energetic_evolution(loading: np.ndarray, k: float = 1.0,
                                          a: float = 0.15, rho: float = 0.10,
                                          q_initial: float = -0.15,
                                          b: float = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("loading must be finite throughout")
    for name, value in (("q_initial", q_initial), ("k", k), ("a", a), ("rho", rho)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(k) <= 0.0:
        raise ValueError("k must be strictly greater than zero")
    if float(a) < 0.0:
        raise ValueError("a must be non-negative")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and not isinstance(b, bool) and np.isfinite(float(b))
                              and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    curvature, half_gap, threshold = float(k), float(a), float(rho)
    far_gap = half_gap if b is None else float(b)

    # Continuity of the free energy at the barrier fixes the level of the far
    # well; the near well is the zero of energy.
    level = 0.5 * curvature * (half_gap ** 2 - far_gap ** 2)

    # One increment of the globally minimising scheme. It is written out here
    # rather than imported from sub-problem 03, so that the trajectory this
    # step defines is reproducible from this file alone and cannot be changed
    # by whichever copy of an earlier step happens to be in scope.
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

    potential = table[:, 2]
    trajectory = np.empty(potential.size, dtype=float)
    trajectory[0] = float(q_initial)

    # The state is carried forward explicitly: every increment is anchored at
    # the state the previous one returned, which is where the memory lives.
    for index in range(1, potential.size):
        trajectory[index] = _advance(float(trajectory[index - 1]), float(potential[index]))

    return trajectory

# ORACLE SOLUTION

import numpy as np
def track_local_branch(loading: np.ndarray, k: float = 1.0, a: float = 0.15,
                               rho: float = 0.10, q_initial: float = -0.15,
                               b: float = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("loading must be finite throughout")
    for name, value in (("q_initial", q_initial), ("k", k), ("a", a), ("rho", rho)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(q_initial) > 0.0:
        raise ValueError("q_initial must lie in the well centred at -a, so it must not be positive")
    if float(k) <= 0.0:
        raise ValueError("k must be strictly greater than zero")
    if float(a) < 0.0:
        raise ValueError("a must be non-negative")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and not isinstance(b, bool) and np.isfinite(float(b))
                              and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    linear, shift, threshold = float(k), float(a), float(rho)
    far = shift if b is None else float(b)
    tolerance, budget = 1.0e-13, 100

    # One return-map increment on the branch of the occupied well, at the
    # vanishing anharmonicity this step calls for. It is written out here
    # rather than imported from sub-problem 04, so that the branch this step
    # defines is reproducible from this file alone and cannot be changed by
    # whichever copy of an earlier step happens to be in scope.
    def _return_map(previous, load, centre):
        def _force(state):
            return linear * (state - centre)

        # Elastic predictor: the drive the frozen state feels under the new load.
        trial = load - _force(previous)
        if abs(trial) <= threshold:
            return float(previous)

        # Corrector: place the drive exactly on the boundary of the elastic
        # range, on the side the violation points to.
        direction = 1.0 if trial > 0.0 else -1.0
        target = load - direction * threshold

        lower, upper = previous, previous
        span = max(1.0, abs(previous))
        for _ in range(200):
            if direction > 0.0:
                upper = previous + span
                if _force(upper) >= target:
                    break
            else:
                lower = previous - span
                if _force(lower) <= target:
                    break
            span *= 2.0
        else:
            raise ValueError("the branch force could not be bracketed around the target")

        # Safeguarded Newton: take the Newton step when it stays inside the
        # bracket and makes progress, and bisect otherwise.
        state = 0.5 * (lower + upper)
        for _ in range(budget):
            residual = _force(state) - target
            if abs(residual) <= tolerance:
                break
            if residual > 0.0:
                upper = state
            else:
                lower = state
            step = state - residual / linear
            if not (lower < step < upper):
                step = 0.5 * (lower + upper)
            if abs(step - state) <= tolerance * (1.0 + abs(state)):
                state = step
                break
            state = step

        return float(state)

    potential = table[:, 2]
    trajectory = np.empty(potential.size, dtype=float)
    trajectory[0] = float(q_initial)
    crossed = float(q_initial) > 0.0

    for index in range(1, potential.size):
        if crossed:
            # The far branch holds the state for the rest of the cycle.
            trajectory[index] = _return_map(float(trajectory[index - 1]),
                                            float(potential[index]), far)
            continue
        state = _return_map(float(trajectory[index - 1]), float(potential[index]),
                            -shift)
        if state > 0.0:
            # The occupied branch is exhausted. The viscous transient that
            # carries the state across the barrier runs at frozen loading and
            # leaves the displacement from the well bottom unchanged, so the
            # state arrives on the far branch a + b further along.
            state += shift + far
            crossed = True
        trajectory[index] = state

    return trajectory

import numpy as np 

def energy_balance_defect(loading: np.ndarray, trajectory: np.ndarray,
                                  k: float = 1.0, a: float = 0.15,
                                  rho: float = 0.10, b: float = None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    states = np.asarray(trajectory, dtype=float).ravel()
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 2:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 2")
    if states.size != table.shape[0]:
        raise ValueError("trajectory must hold one state per node of the loading table")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(states))):
        raise ValueError("loading and trajectory must be finite throughout")
    if not (isinstance(rho, (int, float, np.floating, np.integer))
            and np.isfinite(float(rho)) and float(rho) > 0.0):
        raise ValueError("rho must be a finite number strictly greater than zero")
    if not (isinstance(k, (int, float, np.floating, np.integer))
            and np.isfinite(float(k)) and float(k) > 0.0):
        raise ValueError("k must be a finite number strictly greater than zero")
    if not (isinstance(a, (int, float, np.floating, np.integer))
            and np.isfinite(float(a)) and float(a) >= 0.0):
        raise ValueError("a must be a finite non-negative number")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and np.isfinite(float(b)) and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    potential = table[:, 2]

    # The landscape of sub-problem 02, written out here rather than imported,
    # so that the defect this step reports is reproducible from this file
    # alone. Continuity at the barrier raises the far well by the amount by
    # which the two bottoms are unequally placed.
    curvature = float(k)
    repressed = float(a)
    active = repressed if b is None else float(b)
    centre = np.where(states <= 0.0, -repressed, active)
    level = 0.5 * curvature * (repressed ** 2 - active ** 2)
    offset = np.where(states <= 0.0, 0.0, level)
    free_energy = 0.5 * curvature * (states - centre) ** 2 + offset

    stored = free_energy - states * potential

    # The work of one increment is credited at frozen state, so the state is
    # held at the earlier node while the potential moves to the later one.
    work = float(np.sum(-states[:-1] * np.diff(potential)))

    # A one-homogeneous dissipation potential accumulates as the total
    # variation of the trajectory, weighted by the threshold.
    dissipated = float(rho) * float(np.sum(np.abs(np.diff(states))))

    return float((stored[0] + work) - (stored[-1] + dissipated))

# ORACLE SOLUTION

import numpy as np
def measure_energy_consistency_order(step_counts, k: float = 1.0, a: float = 0.15,
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

import numpy as np 

def summarize_cycle(loading: np.ndarray, trajectory: np.ndarray,
                            rho: float = 0.10) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    states = np.asarray(trajectory, dtype=float).ravel()
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 2:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 2")
    if states.size != table.shape[0]:
        raise ValueError("trajectory must hold one state per node of the loading table")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(states))):
        raise ValueError("loading and trajectory must be finite throughout")
    if not (isinstance(rho, (int, float, np.floating, np.integer))
            and np.isfinite(float(rho)) and float(rho) > 0.0):
        raise ValueError("rho must be a finite number strictly greater than zero")

    time = table[:, 0]
    potential = table[:, 2]
    threshold = float(rho)

    peak = float(np.max(states))
    residual = float(states[-1])

    # A one-homogeneous dissipation potential accumulates as the total
    # variation of the trajectory, weighted by the threshold.
    dissipated = threshold * float(np.sum(np.abs(np.diff(states))))

    # The work the environment supplies is minus the integral of the state
    # against the interaction potential, which for a closed cycle in the
    # potential is the area the trajectory encloses in that plane.
    area = float(-np.trapezoid(states, potential)) if hasattr(np, "trapezoid") \
        else float(-np.trapz(states, potential))

    crossings = np.flatnonzero(states > 0.0)
    if crossings.size == 0:
        crossing_potential, crossing_time = -1.0, -1.0
    else:
        index = int(crossings[0])
        crossing_potential, crossing_time = float(potential[index]), float(time[index])

    return np.array([peak, residual, dissipated, area,
                     crossing_potential, crossing_time], dtype=float)

import numpy as np 

def identify_material_parameters(q_peak: float, q_residual: float,
                                         dissipation: float, delta: float,
                                         ell_max: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_peak", q_peak), ("q_residual", q_residual),
                        ("dissipation", dissipation), ("delta", delta),
                        ("ell_max", ell_max)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    peak, residual = float(q_peak), float(q_residual)
    cost, offset, ceiling = float(dissipation), float(delta), float(ell_max)
    if not peak > residual > 0.0:
        raise ValueError("the record must satisfy 0 < q_residual < q_peak")
    if cost <= 0.0 or ceiling <= 0.0:
        raise ValueError("dissipation and ell_max must be strictly positive")

    span = peak - residual

    # Every constant is a function of the threshold alone: the two levels fix
    # the curvature and the far well, the irreversible cost fixes the near
    # well, and the free-energy difference is what is left to satisfy.
    def _unfold(threshold):
        curvature = (ceiling - 2.0 * threshold) / span
        if not curvature > 0.0:
            return None
        near = cost / threshold - 2.0 * peak + residual
        far = residual - threshold / curvature
        return curvature, near, far

    def _residual(threshold):
        unfolded = _unfold(threshold)
        if unfolded is None:
            return -np.inf
        curvature, near, far = unfolded
        return 0.5 * curvature * (near * near - far * far) - offset

    # Only thresholds below all three of these leave a landscape that is a
    # landscape at all: beyond them the curvature, the near well or the far
    # well would have to be non-positive.
    upper = min(0.5 * ceiling,
                cost / (2.0 * peak - residual),
                residual * ceiling / (span + 2.0 * residual)) * (1.0 - 1.0e-12)
    lower = 1.0e-12 * upper

    # A vanishing threshold makes the near well arbitrarily wide, so the
    # residual opens positive; the identification is the first crossing, the
    # later ones belonging to landscapes the admissible interval excludes.
    grid = lower + (upper - lower) * np.linspace(0.0, 1.0, 4097)
    values = np.array([_residual(float(point)) for point in grid], dtype=float)
    bracket = None
    for index in range(grid.size - 1):
        if values[index] > 0.0 >= values[index + 1]:
            bracket = (float(grid[index]), float(grid[index + 1]))
            break
    if bracket is None:
        raise ValueError("the record admits no bistable landscape with this offset")

    low, high = bracket
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _residual(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low <= 1.0e-16 * (1.0 + high):
            break

    threshold = 0.5 * (low + high)
    curvature, near, far = _unfold(threshold)
    if not (near > 0.0 and far > 0.0):
        raise ValueError("the recovered landscape is not bistable")

    return np.array([curvature, near, far, threshold], dtype=float)

# ORACLE SOLUTION

import numpy as np
def run_epigenetic_switch_comparison(q_peak: float = 0.448487,
                                             q_residual: float = 0.165654,
                                             dissipation: float = 0.0835480,
                                             delta: float = 0.00603213,
                                             ell_inf: float = 0.5,
                                             lam: float = 1.0,
                                             S_max: float = 5.0,
                                             T: float = 1.0,
                                             n_steps: int = 4000,
                                             refinement=None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- The pipeline is an end-to-end chain of the ten earlier golden
    #    implementations, called by their  names, so that the
    #    result never depends on any other implementation of a step.

    # -- Validate the orchestrator inputs.
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 2:
        raise ValueError("n_steps must be an integer of at least two")
    for name, value in (("ell_inf", ell_inf), ("lam", lam), ("S_max", S_max), ("T", T)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number strictly greater than zero")
    sweep = (100, 200, 400, 800, 1600, 3200, 6400) if refinement is None else tuple(
        int(value) for value in refinement)

    # -- Sub-problem 01: tabulate the stimulus protocol and its potential.
    loading = build_loading_path(int(n_steps), float(T), float(S_max),
                                 float(ell_inf), float(lam))
    ceiling = float(np.max(loading[:, 2]))

    # -- Sub-problem 10: the landscape is read out of the record, not assumed.
    recovered = np.asarray(identify_material_parameters(
        float(q_peak), float(q_residual), float(dissipation), float(delta), ceiling),
        dtype=float)
    k, a, b, rho = (float(recovered[0]), float(recovered[1]),
                    float(recovered[2]), float(recovered[3]))
    start = -a

    # -- Sub-problem 02: the landscape must really be bistable, since a
    #    vanishing barrier makes the two selection principles coincide and the
    #    comparison the run exists to make would be empty.
    barrier = np.asarray(evaluate_double_well([start, 0.0], k, a, b), dtype=float)
    if not float(barrier[0, 1]) > 0.0:
        raise ValueError("the landscape carries no barrier, so the two principles coincide")

    # -- The mark must survive its own unloading arm: were the active well
    #    raised by more than the threshold can hold, the globally minimising
    #    evolution would carry it back across the barrier at baseline and the
    #    record could not have been produced by either principle.
    if not 0.5 * k * (a - b) < rho:
        raise ValueError("the recovered landscape cannot retain the mark at baseline")

    # -- Sub-problems 03 and 04: probe both increments at the peak of the
    #    stimulus. If neither principle can move the mark at the strongest
    #    drive the protocol reaches, no crossing can occur anywhere and the
    #    ratio is undefined; this is settled before the partition is swept.
    probe_global = float(solve_incremental_step(start, ceiling, k, a, rho, b))
    probe_local = float(return_map_step(start, ceiling, rho, k, a, 0.0))
    if probe_global <= 0.0 and probe_local <= 0.0:
        raise ValueError("the stimulus is too weak for either principle to cross the barrier")

    # -- Sub-problems 05 and 06: the two arms, on one and the same partition.
    global_arm = integrate_energetic_evolution(loading, k, a, rho, start, b)
    local_arm = track_local_branch(loading, k, a, rho, start, b)

    # -- Sub-problem 07: the scheme over-dissipates rather than manufacturing
    #    energy, so a negative defect would mean the trajectory is not the one
    #    the incremental scheme produces.
    defect = float(energy_balance_defect(loading, global_arm, k, a, rho, b))
    if defect < -1.0e-12:
        raise ValueError("the discrete energy balance is violated, so the trajectory is invalid")

    # -- Sub-problem 08: the crossings are only as trustworthy as the
    #    integrator, so the defect must decay under refinement before the
    #    located nodes are reported.
    order = float(measure_energy_consistency_order(sweep, k, a, rho, start, float(T),
                                                   float(S_max), float(ell_inf),
                                                   float(lam), b))
    if not order > 0.5:
        raise ValueError("the energy balance does not close under refinement")

    # -- Sub-problem 09: reduce each arm to its reportable quantities; entry
    #    four of the summary is the interaction potential at the crossing.
    global_summary = np.asarray(summarize_cycle(loading, global_arm, rho), dtype=float)
    local_summary = np.asarray(summarize_cycle(loading, local_arm, rho), dtype=float)

    # -- The identification is only credible if the identified landscape
    #    reproduces the record it was read from, which the globally selected
    #    arm must do to within the resolution of the partition.
    tolerance = 1.0e-3 * (1.0 + abs(float(q_peak)))
    if abs(float(global_summary[0]) - float(q_peak)) > tolerance:
        raise ValueError("the identified landscape does not reproduce the recorded peak")
    if abs(float(global_summary[1]) - float(q_residual)) > tolerance:
        raise ValueError("the identified landscape does not reproduce the recorded residual")

    if float(global_summary[4]) < 0.0 or float(local_summary[4]) < 0.0:
        raise ValueError("one of the two arms never carries the mark past the barrier")
    if not float(local_summary[4]) > 0.0:
        raise ValueError("the branch-tracking crossing sits at zero potential, so the ratio is undefined")

    return float(global_summary[4] / local_summary[4])
SCICODE_GOLD_EOF
