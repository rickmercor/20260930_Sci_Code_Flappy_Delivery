#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_local_crossing_probabilities(path_counts) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    counts = np.asarray(path_counts)
    if counts.dtype == bool:
        raise ValueError("path_counts must be numeric, not boolean")
    if not np.issubdtype(counts.dtype, np.number):
        raise ValueError("path_counts must be numeric")
    counts = counts.astype(float)
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("path_counts must have shape (n_ensembles, 4)")
    if counts.shape[0] < 2:
        raise ValueError("at least two ensembles are required")
    if not np.all(np.isfinite(counts)):
        raise ValueError("path_counts must be finite")
    if np.any(counts < 0.0):
        raise ValueError("path_counts must be non-negative")
    if np.any(np.abs(counts - np.rint(counts)) > 1e-9):
        raise ValueError("path_counts must be whole numbers of sampled paths")

    # The four populations of a row split into two arrival groups, each holding
    # the segments that leave downwards and upwards respectively.
    grouped = counts.reshape(counts.shape[0], 2, 2)
    totals = grouped.sum(axis=2)
    if np.any(totals <= 0.0):
        raise ValueError("every arrival side must carry at least one sampled path")

    # Departure is exhaustive, so normalising within an arrival group is the
    # whole content of the estimator. An empty departure population is a
    # genuine zero, not a missing measurement.
    probabilities = grouped / totals[:, :, None]

    return probabilities

def build_state_space(n_interfaces: int) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the module-level import is not in scope here.
    import numpy as np

    if isinstance(n_interfaces, bool) or not isinstance(n_interfaces, (int, np.integer)):
        raise ValueError("n_interfaces must be an integer")
    if int(n_interfaces) < 4:
        raise ValueError("n_interfaces must be at least 4")

    last = int(n_interfaces) - 1
    rows = []

    # The excursions that leave the first interface into the reactant state and come back. They start and end on the same side, hence the (+1, +1) label.
    rows.append((-1, 1, 1))

    # The ensemble straddling the first interface. A segment arriving from the interface above and departing towards it would have to touch the first interface without entering the reactant state, so that type is absent.
    rows.append((0, -1, -1))
    rows.append((0, -1, 1))
    rows.append((0, 1, -1))

    for i in range(1, last):
        for k in (-1, 1):
            # Segments of the outermost ordinary ensemble that arrive from above start beyond the last interface, i.e. inside the product state, and are collected there instead.
            if i == last - 1 and k == 1:
                continue
            for departure in (-1, 1):
                rows.append((i, k, departure))

    # Every segment that begins in the product state is one state.
    rows.append((last, 0, 0))

    states = np.array(rows, dtype=int)
    if states.shape[0] != 4 * last - 1:
        raise ValueError("state enumeration is inconsistent with the interface count")

    return states

def build_transition_matrix(states, probabilities) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 3 or probs.shape[1:] != (2, 2):
        raise ValueError("probabilities must have shape (n_ensembles, 2, 2)")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")

    n_states = labels.shape[0]
    last = int(labels[:, 0].max())
    if probs.shape[0] != last:
        raise ValueError("probabilities must supply one entry per ensemble lambda_0..lambda_(N-1)")

    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    matrix = np.zeros((n_states, n_states), dtype=float)
    for state, row in index.items():
        i, _, departure = state
        if i == last:
            matrix[row, row] = 1.0            # the measurement stops here
            continue
        # A reactant excursion always resumes in the straddling ensemble,
        # arriving from below; every other state steps to the neighbour it
        # departed towards, arriving from the side it just left.
        target_i = 0 if i == -1 else i + departure
        arrival = -1 if i == -1 else -departure
        if target_i == -1:
            matrix[row, reactant] = 1.0
            continue
        if target_i == last:
            matrix[row, product] = 1.0
            continue
        for onward, column in ((-1, 0), (1, 1)):
            weight = float(probs[target_i, 0 if arrival == -1 else 1, column])
            target = index.get((target_i, arrival, onward))
            if target is None:
                if weight > 0.0:
                    raise ValueError("non-zero probability into a segment type that does not exist")
                continue
            matrix[row, target] += weight

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError("the assembled matrix is not row-stochastic")

    return matrix

def compute_crossing_probability(transition_matrix, states) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    labels = np.asarray(states)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("states and transition_matrix disagree on the state count")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition_matrix must be finite")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition_matrix must be row-stochastic")

    last = int(labels[:, 0].max())
    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    # Both boundaries are pinned, so only the remaining states are unknowns.
    free = [s for s in range(matrix.shape[0]) if s not in (reactant, product)]
    system = np.eye(len(free)) - matrix[np.ix_(free, free)]
    forcing = matrix[np.ix_(free, [product])].ravel()
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the hitting-probability system is singular")
    solution = np.linalg.solve(system, forcing)

    hitting = np.zeros(matrix.shape[0], dtype=float)
    hitting[free] = solution
    hitting[product] = 1.0                    # the reactant entry stays pinned at zero

    # One obligatory step out of the reactant excursion turns the pinned
    # hitting probabilities into the complement of the return probability.
    crossing_probability = float(matrix[reactant] @ hitting)
    if not 0.0 < crossing_probability <= 1.0:
        raise ValueError("the crossing probability must lie in (0, 1]")

    return crossing_probability

def compute_overlap_free_times(states, reactant_parts, ensemble_parts) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    reactant = np.asarray(reactant_parts, dtype=float)
    parts = np.asarray(ensemble_parts, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if reactant.shape != (3,):
        raise ValueError("reactant_parts must have shape (3,)")
    if parts.ndim != 3 or parts.shape[1:] != (4, 3):
        raise ValueError("ensemble_parts must have shape (n_ensembles, 4, 3)")
    for name, block in (("reactant_parts", reactant), ("ensemble_parts", parts)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
        if np.any(block < 0.0):
            raise ValueError(f"{name} must be non-negative")

    last = int(labels[:, 0].max())
    if parts.shape[0] != last:
        raise ValueError("ensemble_parts must supply one entry per ensemble lambda_0..lambda_(N-1)")

    overlap_free_times = np.zeros(labels.shape[0], dtype=float)
    for state, (i, k, departure) in enumerate(labels):
        i, k, departure = int(i), int(k), int(departure)
        if i == last:
            continue                       # the product state stops the clock
        if i == -1:
            pieces = reactant
        else:
            row = (0 if k == -1 else 2) + (0 if departure == -1 else 1)
            pieces = parts[i, row]
        # Only the middle and trailing pieces are new trajectory; the leading
        # piece repeats what the previous segment already accounted for.
        overlap_free_times[state] = float(pieces[1] + pieces[2])

    return overlap_free_times

def compute_visit_counts(transition_matrix, states) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    labels = np.asarray(states)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("states and transition_matrix disagree on the state count")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition_matrix must be finite")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition_matrix must be row-stochastic")

    last = int(labels[:, 0].max())
    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    # Everything except the absorbing product state can be revisited.
    transient = [s for s in range(matrix.shape[0]) if s != product]
    sub = matrix[np.ix_(transient, transient)]
    system = np.eye(len(transient)) - sub.T
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the product state is not reachable from every transient state")

    injection = np.zeros(len(transient), dtype=float)
    injection[transient.index(reactant)] = 1.0
    occupation = np.linalg.solve(system, injection)
    if np.any(occupation < -1e-9):
        raise ValueError("expected visit counts must be non-negative")

    visit_counts = np.zeros(matrix.shape[0], dtype=float)
    visit_counts[transient] = np.maximum(occupation, 0.0)

    return visit_counts

def compute_interface_dwell_times(states, visit_counts, overlap_free_times) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    visits = np.asarray(visit_counts, dtype=float)
    times = np.asarray(overlap_free_times, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if visits.shape != (labels.shape[0],):
        raise ValueError("visit_counts must have one entry per state")
    if times.shape != (labels.shape[0],):
        raise ValueError("overlap_free_times must have one entry per state")
    for name, block in (("visit_counts", visits), ("overlap_free_times", times)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
        if np.any(block < 0.0):
            raise ValueError(f"{name} must be non-negative")

    last = int(labels[:, 0].max())
    dwell_times = np.zeros(last, dtype=float)
    for state, (i, _, _) in enumerate(labels):
        i = int(i)
        if i == last:
            continue                # the product state closes the passage
        # The reactant excursion and the straddling ensemble are both centred
        # on the reactant interface and form a single group.
        dwell_times[max(i, 0)] += visits[state] * times[state]

    if not np.all(np.isfinite(dwell_times)):
        raise ValueError("the dwell profile is not finite")
    if dwell_times.sum() <= 0.0:
        raise ValueError("a passage must accumulate a positive time")

    return dwell_times

def compute_conditional_passage_time(transition_matrix, states, overlap_free_times,
                                             reactant_middle_time: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    labels = np.asarray(states)
    times = np.asarray(overlap_free_times, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if labels.shape[0] != matrix.shape[0] or times.shape != (matrix.shape[0],):
        raise ValueError("states, transition_matrix and overlap_free_times must agree in size")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(times)):
        raise ValueError("transition_matrix and overlap_free_times must be finite")
    if np.any(times < 0.0):
        raise ValueError("overlap_free_times must be non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition_matrix must be row-stochastic")
    if isinstance(reactant_middle_time, bool) or not isinstance(
            reactant_middle_time, (int, float, np.integer, np.floating)):
        raise ValueError("reactant_middle_time must be a real number")
    if not np.isfinite(reactant_middle_time) or float(reactant_middle_time) < 0.0:
        raise ValueError("reactant_middle_time must be finite and non-negative")

    last = int(labels[:, 0].max())
    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    # Both the return to the reactant state and the arrival in the product
    # state end the excursion, so both are destinations of the system.
    interior = [s for s in range(matrix.shape[0]) if s not in (reactant, product)]
    system = np.eye(len(interior)) - matrix[np.ix_(interior, interior)]
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the mean first passage system is singular")
    solution = np.linalg.solve(system, times[interior])

    passage = np.zeros(matrix.shape[0], dtype=float)
    passage[interior] = solution

    # One obligatory step out of the reactant excursion, then move the start of
    # the clock to the last crossing of the reactant boundary.
    stopped = float(times[reactant] + matrix[reactant] @ passage)
    conditional_passage_time = stopped - float(reactant_middle_time)
    if conditional_passage_time <= 0.0:
        raise ValueError("the reactant excursion cannot be longer than the stopped time")

    return conditional_passage_time

def compute_rate_constant(reactant_time: float, conditional_passage_time: float,
                                  crossing_probability: float, time_step: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    values = {"reactant_time": reactant_time,
              "conditional_passage_time": conditional_passage_time,
              "crossing_probability": crossing_probability,
              "time_step": time_step}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if float(reactant_time) < 0.0:
        raise ValueError("reactant_time must be non-negative")
    if float(conditional_passage_time) <= 0.0:
        raise ValueError("conditional_passage_time must be positive")
    if not 0.0 < float(crossing_probability) <= 1.0:
        raise ValueError("crossing_probability must lie in (0, 1]")
    if float(time_step) <= 0.0:
        raise ValueError("time_step must be positive")

    # The interval between two successive upward crossings of the reactant
    # boundary spans one excursion below it and one full excursion above it.
    interval = (float(reactant_time) + float(conditional_passage_time)) * float(time_step)
    flux = 1.0 / interval

    return float(flux * float(crossing_probability))

def compute_bottleneck_dwell(dwell_times, rate_constant: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    profile = np.asarray(dwell_times, dtype=float)
    if profile.ndim != 1:
        raise ValueError("dwell_times must be one dimensional")
    if profile.size < 2:
        raise ValueError("dwell_times must cover at least one interface above the boundary")
    if not np.all(np.isfinite(profile)):
        raise ValueError("dwell_times must be finite")
    if np.any(profile < 0.0):
        raise ValueError("dwell_times must be non-negative")
    if isinstance(rate_constant, bool) or not isinstance(
            rate_constant, (int, float, np.integer, np.floating)):
        raise ValueError("rate_constant must be a real number")
    if not np.isfinite(rate_constant) or float(rate_constant) <= 0.0:
        raise ValueError("rate_constant must be a positive finite number")

    total = float(profile.sum())
    if total <= 0.0:
        raise ValueError("a passage must accumulate a positive time")

    # The reactant boundary carries the waiting time of the rare event, which
    # is not part of the transit; the bottleneck is sought above it.
    above = profile[1:]
    largest = float(above.max())
    if largest <= 0.0:
        raise ValueError("no time is accumulated above the reactant boundary")

    share = largest / total

    # The profile sums to the mean duration of one passage, so dividing the
    # share by the rate restores the absolute timescale.
    return float(share / float(rate_constant))

def run_kcl_dissociation_pipeline(time_step: float = 0.02,
                                          well_scale: float = 1.0) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    for name, value in (("time_step", time_step), ("well_scale", well_scale)):
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")

    n_interfaces = 16

    # -- Sampled path-type populations, ordered (arrive below / depart below,
    #    arrive below / depart above, arrive above / depart below,
    #    arrive above / depart above) for the ensembles centred on
    #    lambda_0 ... lambda_14. The straddling ensemble has no fourth type.
    path_counts = np.array([[3820, 844, 1970, 0],
                            [2538, 1845, 1477, 2421],
                            [2446, 1915, 1412, 2365],
                            [2455, 2061, 1370, 2271],
                            [2432, 2180, 1453, 2333],
                            [2226, 2125, 1536, 2354],
                            [2278, 2305, 1449, 2089],
                            [2216, 2370, 1456, 1969],
                            [2021, 2279, 1522, 2171],
                            [2022, 2395, 1597, 2396],
                            [1916, 2375, 1501, 2363],
                            [2037, 2634, 1498, 2466],
                            [1950, 2624, 1294, 2220],
                            [1900, 2650, 1289, 2297],
                            [1832, 2643, 1401, 2585]], dtype=float)

    # -- Mean leading, middle and trailing piece of every path type, in phase
    #    points, in the same ensemble and type order.
    ensemble_parts = np.array([
        [[0.0, 47.3, 0.0], [0.0, 0.0, 26.8], [24.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[18.4, 11.4, 18.0], [18.7, 8.5, 24.5], [25.2, 9.3, 19.2], [25.7, 12.3, 25.4]],
        [[23.1, 108.5, 22.7], [23.4, 80.8, 29.2], [29.9, 88.6, 23.9], [30.4, 117.5, 30.1]],
        [[27.6, 52.0, 27.2], [27.9, 38.7, 33.7], [34.4, 42.5, 28.4], [34.9, 56.3, 34.6]],
        [[32.0, 45.7, 31.6], [32.3, 34.0, 38.1], [38.8, 37.3, 32.8], [39.3, 49.4, 39.0]],
        [[36.4, 40.1, 36.0], [36.7, 29.9, 42.5], [43.2, 32.8, 37.2], [43.7, 43.4, 43.4]],
        [[41.1, 35.2, 40.7], [41.4, 26.2, 47.2], [47.9, 28.8, 41.9], [48.4, 38.1, 48.1]],
        [[45.9, 30.9, 45.5], [46.2, 23.0, 52.0], [52.7, 25.2, 46.7], [53.2, 33.5, 52.9]],
        [[49.9, 27.1, 49.5], [50.2, 20.2, 56.0], [56.7, 22.2, 50.7], [57.2, 29.4, 56.9]],
        [[54.7, 23.8, 54.3], [55.0, 17.7, 60.8], [61.5, 19.5, 55.5], [62.0, 25.8, 61.7]],
        [[59.5, 20.9, 59.1], [59.8, 15.6, 65.6], [66.3, 17.1, 60.3], [66.8, 22.7, 66.5]],
        [[64.1, 18.4, 63.7], [64.4, 13.7, 70.2], [70.9, 15.0, 64.9], [71.4, 19.9, 71.1]],
        [[68.6, 16.1, 68.2], [68.9, 12.0, 74.7], [75.4, 13.2, 69.4], [75.9, 17.5, 75.6]],
        [[72.9, 14.2, 72.5], [73.2, 10.5, 79.0], [79.7, 11.6, 73.7], [80.2, 15.3, 79.9]],
        [[77.5, 12.4, 77.1], [77.8, 9.3, 83.6], [84.3, 10.2, 78.3], [84.8, 13.5, 84.5]],
    ], dtype=float)
    ensemble_parts[2, :, 1] *= float(well_scale)

    # -- The excursions confined to the bound state are all middle piece.
    reactant_parts = np.array([0.0, 43.7, 0.0])

    # -- Sub-problems 01-03: the chain itself.
    probabilities = compute_local_crossing_probabilities(path_counts)
    states = build_state_space(n_interfaces)
    transition_matrix = build_transition_matrix(states, probabilities)

    # -- Sub-problem 04: probability that a departure commits.
    crossing_probability = compute_crossing_probability(transition_matrix, states)

    # -- Sub-problems 05-07: the accumulated-time profile along the coordinate.
    overlap_free_times = compute_overlap_free_times(
        states, reactant_parts, ensemble_parts)
    visit_counts = compute_visit_counts(transition_matrix, states)
    dwell_times = compute_interface_dwell_times(
        states, visit_counts, overlap_free_times)

    # -- Sub-problems 08-09: the independent flux route to the timescale.
    conditional_passage_time = compute_conditional_passage_time(
        transition_matrix, states, overlap_free_times, float(reactant_parts[1]))
    rate_constant = compute_rate_constant(
        float(reactant_parts[1]), conditional_passage_time,
        crossing_probability, float(time_step))

    # -- Sub-problem 10: the reported dwell time.
    return float(compute_bottleneck_dwell(dwell_times, rate_constant))
SCICODE_GOLD_EOF
