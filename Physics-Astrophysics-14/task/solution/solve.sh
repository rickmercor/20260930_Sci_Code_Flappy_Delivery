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


def boosted_turnaround_potential(position, acceleration, raw_potential,
                                        seed_ids, scale_factor, hubble_rate,
                                        omega_m, turnaround_overdensity):
    def numeric_vector(value, name):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must be one-dimensional with at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        return array

    def integral_ids(value, name, size, nonempty=True):
        try:
            array = np.asarray(value, dtype=object)
        except Exception as exc:
            raise ValueError(name + " must be a one-dimensional integer sequence") from exc
        if array.ndim != 1 or (nonempty and array.size == 0):
            raise ValueError(name + " must be a nonempty one-dimensional integer sequence")
        ids = []
        for item in array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError(name + " must contain integral node IDs")
            node = int(item)
            if node < 0 or node >= size:
                raise ValueError(name + " contains an invalid node ID")
            ids.append(node)
        if len(set(ids)) != len(ids):
            raise ValueError(name + " must contain distinct node IDs")
        return ids

    position_array = numeric_vector(position, "position")
    acceleration_array = numeric_vector(acceleration, "acceleration")
    potential_array = numeric_vector(raw_potential, "raw_potential")
    if position_array.shape != acceleration_array.shape or position_array.shape != potential_array.shape:
        raise ValueError("particle arrays must have equal length")
    seeds = integral_ids(seed_ids, "seed_ids", position_array.size)
    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
        ("turnaround_overdensity", turnaround_overdensity, False),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
            raise ValueError(name + " lies outside its valid range")
        if name == "scale_factor":
            scale_factor = scalar
        elif name == "hubble_rate":
            hubble_rate = scalar
        elif name == "omega_m":
            omega_m = scalar
        else:
            turnaround_overdensity = scalar

    def wide_result():
        # Exact products avoid overflow and underflow before compensating factors.
        from fractions import Fraction as F
        x, acceleration, potential = [[F(float(v)) for v in array]
                                      for array in (position_array, acceleration_array, potential_array)]
        scale, hubble, matter, density = map(F, (scale_factor, hubble_rate, omega_m, turnaround_overdensity))
        gradient = -sum(acceleration[j] for j in seeds) / len(seeds)
        coefficient = hubble * hubble * matter * scale * scale * density / 4
        def field(center_x, center_potential):
            return [potential[i] - center_potential - gradient * (x[i] - center_x)
                    - coefficient * (x[i] - center_x)**2 for i in range(len(x))]
        preliminary = field(sum(x[j] for j in seeds) / len(seeds),
                            sum(potential[j] for j in seeds) / len(seeds))
        center = min(seeds, key=lambda node: (preliminary[node], node))
        try:
            result = [float(v) for v in field(x[center], potential[center])]
        except OverflowError as exc:
            raise ValueError("boosted potential cannot be represented as finite floats") from exc
        if not all(np.isfinite(result)):
            raise ValueError("boosted potential cannot be represented as finite floats")
        return result

    return wide_result()

import numpy as np


def descend_to_minimum(potential, adjacency, seed_ids):
    try:
        values = np.asarray(potential)
    except Exception as exc:
        raise ValueError("potential must be a one-dimensional numeric array") from exc
    if values.ndim != 1 or values.size < 3:
        raise ValueError("potential must have at least three entries")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError("potential must contain real numeric values")
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("potential must contain finite real values") from exc
    if not np.all(np.isfinite(values)) or np.unique(values).size != values.size:
        raise ValueError("potential values must be finite and pairwise distinct")
    node_count = int(values.size)

    if not isinstance(adjacency, (list, tuple, np.ndarray)) or len(adjacency) != node_count:
        raise ValueError("adjacency must contain one row per node")
    rows = []
    for node, raw_row in enumerate(adjacency):
        try:
            row_array = np.asarray(raw_row, dtype=object)
        except Exception as exc:
            raise ValueError("adjacency rows must be one-dimensional") from exc
        if row_array.ndim != 1:
            raise ValueError("adjacency rows must be one-dimensional")
        row = []
        for item in row_array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError("adjacency must contain integral node IDs")
            neighbor = int(item)
            if neighbor < 0 or neighbor >= node_count:
                raise ValueError("adjacency contains an invalid node ID")
            row.append(neighbor)
        if any(left >= right for left, right in zip(row, row[1:])):
            raise ValueError("adjacency rows must be sorted and unique")
        if node in row:
            raise ValueError("the graph must not contain self-loops")
        rows.append(row)
    for node, row in enumerate(rows):
        for neighbor in row:
            if node not in rows[neighbor]:
                raise ValueError("the graph must be undirected")
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in rows[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if len(reached) != node_count:
        raise ValueError("the graph must be connected")

    try:
        seed_array = np.asarray(seed_ids, dtype=object)
    except Exception as exc:
        raise ValueError("seed_ids must be a one-dimensional integer sequence") from exc
    if seed_array.ndim != 1 or seed_array.size == 0:
        raise ValueError("seed_ids must be nonempty and one-dimensional")
    seeds = []
    for item in seed_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("seed_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("seed_ids contains an invalid node ID")
        seeds.append(node)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed_ids must be distinct")

    current = min(seeds, key=lambda node: (values[node], node))
    while True:
        lower = [neighbor for neighbor in rows[current] if values[neighbor] < values[current]]
        if not lower:
            return int(current)
        current = min(lower, key=lambda node: (values[node], node))

import heapq
import numpy as np


def locate_host_well(potential, adjacency, minimum_id, subgroup_min_size):
    try:
        values = np.asarray(potential)
    except Exception as exc:
        raise ValueError("potential must be a one-dimensional numeric array") from exc
    if values.ndim != 1 or values.size < 3:
        raise ValueError("potential must have at least three entries")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError("potential must contain real numeric values")
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("potential must contain finite real values") from exc
    if not np.all(np.isfinite(values)) or np.unique(values).size != values.size:
        raise ValueError("potential values must be finite and pairwise distinct")
    node_count = int(values.size)

    if not isinstance(adjacency, (list, tuple, np.ndarray)) or len(adjacency) != node_count:
        raise ValueError("adjacency must contain one row per node")
    rows = []
    for node, raw_row in enumerate(adjacency):
        try:
            row_array = np.asarray(raw_row, dtype=object)
        except Exception as exc:
            raise ValueError("adjacency rows must be one-dimensional") from exc
        if row_array.ndim != 1:
            raise ValueError("adjacency rows must be one-dimensional")
        row = []
        for item in row_array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError("adjacency must contain integral node IDs")
            neighbor = int(item)
            if neighbor < 0 or neighbor >= node_count:
                raise ValueError("adjacency contains an invalid node ID")
            row.append(neighbor)
        if any(left >= right for left, right in zip(row, row[1:])):
            raise ValueError("adjacency rows must be sorted and unique")
        if node in row:
            raise ValueError("the graph must not contain self-loops")
        rows.append(row)
    for node, row in enumerate(rows):
        for neighbor in row:
            if node not in rows[neighbor]:
                raise ValueError("the graph must be undirected")
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in rows[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if len(reached) != node_count:
        raise ValueError("the graph must be connected")

    if isinstance(minimum_id, (bool, np.bool_)) or not isinstance(minimum_id, (int, np.integer)):
        raise ValueError("minimum_id must be integral")
    minimum_id = int(minimum_id)
    if minimum_id < 0 or minimum_id >= node_count:
        raise ValueError("minimum_id is out of range")
    if isinstance(subgroup_min_size, (bool, np.bool_)) or not isinstance(subgroup_min_size, (int, np.integer)):
        raise ValueError("subgroup_min_size must be integral")
    subgroup_min_size = int(subgroup_min_size)
    if subgroup_min_size < 1:
        raise ValueError("subgroup_min_size must be positive")
    if any(values[neighbor] < values[minimum_id] for neighbor in rows[minimum_id]):
        raise ValueError("minimum_id must be a local minimum")

    internal = {minimum_id}
    surface = set(rows[minimum_id])
    groups = []
    while surface:
        contour = min(surface, key=lambda node: (values[node], node))
        lower_neighbors = [neighbor for neighbor in rows[contour] if values[neighbor] < values[contour]]
        if all(neighbor in internal for neighbor in lower_neighbors):
            internal.add(contour)
            surface.discard(contour)
            surface.update(neighbor for neighbor in rows[contour] if neighbor not in internal)
            continue

        branch = {contour}
        queued = set()
        exploratory = []
        for neighbor in rows[contour]:
            if neighbor not in internal and neighbor not in branch and neighbor not in queued:
                heapq.heappush(exploratory, (values[neighbor], neighbor))
                queued.add(neighbor)
        found_deeper = False
        while exploratory and exploratory[0][0] < values[contour]:
            _, node = heapq.heappop(exploratory)
            queued.discard(node)
            if node in internal or node in branch:
                continue
            branch.add(node)
            if values[node] < values[minimum_id]:
                found_deeper = True
                break
            for neighbor in rows[node]:
                if neighbor not in internal and neighbor not in branch and neighbor not in queued:
                    heapq.heappush(exploratory, (values[neighbor], neighbor))
                    queued.add(neighbor)
        if found_deeper:
            internal.add(contour)
            return ([int(node) for node in sorted(internal)], int(contour),
                    [[int(node) for node in group] for group in groups])

        internal.update(branch)
        surface.difference_update(branch)
        for node in branch:
            surface.update(neighbor for neighbor in rows[node] if neighbor not in internal)
        if len(branch) >= subgroup_min_size:
            groups.append([int(node) for node in sorted(branch)])
    raise ValueError("no deeper exit is reachable")

import numpy as np


def host_energy_mask(position, canonical_velocity, boosted_potential,
                           well_ids, saddle_id, minimum_id, scale_factor,
                           hubble_rate, omega_m, omega_lambda,
                           turnaround_overdensity):
    def numeric_vector(value, name):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must have at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        return array

    position_array = numeric_vector(position, "position")
    velocity_array = numeric_vector(canonical_velocity, "canonical_velocity")
    boosted_array = numeric_vector(boosted_potential, "boosted_potential")
    if position_array.shape != velocity_array.shape or position_array.shape != boosted_array.shape:
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    try:
        well_array = np.asarray(well_ids, dtype=object)
    except Exception as exc:
        raise ValueError("well_ids must be a one-dimensional integer sequence") from exc
    if well_array.ndim != 1 or well_array.size == 0:
        raise ValueError("well_ids must be nonempty and one-dimensional")
    well = []
    for item in well_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("well_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("well_ids contains an invalid node ID")
        well.append(node)
    if len(set(well)) != len(well):
        raise ValueError("well_ids must be distinct")
    well_set = set(well)

    converted_ids = []
    for name, item in (("saddle_id", saddle_id), ("minimum_id", minimum_id)):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(name + " must be integral")
        node = int(item)
        if node < 0 or node >= node_count or node not in well_set:
            raise ValueError(name + " must be a valid member of the well")
        converted_ids.append(node)
    saddle_id, minimum_id = converted_ids

    scalars = []
    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
        ("omega_lambda", omega_lambda, False),
        ("turnaround_overdensity", turnaround_overdensity, False),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
            raise ValueError(name + " lies outside its valid range")
        scalars.append(scalar)
    scale_factor, hubble_rate, omega_m, omega_lambda, turnaround_overdensity = scalars

    def wide_result():
        from fractions import Fraction as F
        x, velocity, boosted = [[F(float(v)) for v in array]
                                for array in (position_array, velocity_array, boosted_array)]
        scale, hubble, matter, vacuum, density = map(F, scalars)
        mean = sum(velocity[j] for j in well) / len(well)
        coefficient = hubble * hubble * (matter * (1 + density) - 2 * vacuum) * scale * scale / 4
        displacement = [v - x[minimum_id] for v in x]
        potential = [boosted[i] + coefficient * q * q for i, q in enumerate(displacement)]
        total = [potential[i] + (scale * hubble * q + (velocity[i] - mean) / scale)**2 / 2
                 for i, q in enumerate(displacement)]
        try:
            converted = ([float(v) for v in potential], [float(v) for v in total])
        except OverflowError as exc:
            raise ValueError("physical potentials and energies cannot be represented as finite floats") from exc
        if not all(np.all(np.isfinite(v)) for v in converted):
            raise ValueError("physical potentials and energies cannot be represented as finite floats")
        return converted

    physical_potential, energy = wide_result()
    escape = float(physical_potential[saddle_id])
    mask = [0] * node_count
    for node in well:
        mask[node] = int(energy[node] < escape)
    return (mask, escape, [float(value) for value in physical_potential],
            [float(value) for value in energy])

import numpy as np


def rebound_subgroup(position, canonical_velocity, acceleration,
                           raw_potential, adjacency, subgroup_ids, scale_factor,
                           hubble_rate, omega_m, omega_lambda,
                           turnaround_overdensity, candidate_threshold):
    arrays = []
    for value, name in ((position, "position"),
                        (canonical_velocity, "canonical_velocity"),
                        (acceleration, "acceleration"),
                        (raw_potential, "raw_potential")):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must have at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        arrays.append(array)
    position_array, velocity_array, acceleration_array, raw_array = arrays
    if any(array.shape != position_array.shape for array in arrays[1:]):
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    if not isinstance(adjacency, (list, tuple, np.ndarray)) or len(adjacency) != node_count:
        raise ValueError("adjacency must contain one row per node")
    rows = []
    for node, raw_row in enumerate(adjacency):
        row_array = np.asarray(raw_row, dtype=object)
        if row_array.ndim != 1:
            raise ValueError("adjacency rows must be one-dimensional")
        row = []
        for item in row_array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError("adjacency must contain integral node IDs")
            neighbor = int(item)
            if neighbor < 0 or neighbor >= node_count:
                raise ValueError("adjacency contains an invalid node ID")
            row.append(neighbor)
        if any(left >= right for left, right in zip(row, row[1:])) or node in row:
            raise ValueError("adjacency rows must be sorted, unique and loop-free")
        rows.append(row)
    for node, row in enumerate(rows):
        for neighbor in row:
            if node not in rows[neighbor]:
                raise ValueError("the graph must be undirected")
    reached = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in rows[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if len(reached) != node_count:
        raise ValueError("the graph must be connected")

    if isinstance(candidate_threshold, (bool, np.bool_)) or not isinstance(candidate_threshold, (int, np.integer)):
        raise ValueError("candidate_threshold must be integral")
    candidate_threshold = int(candidate_threshold)
    if candidate_threshold < 1:
        raise ValueError("candidate_threshold must be positive")
    subgroup_array = np.asarray(subgroup_ids, dtype=object)
    if subgroup_array.ndim != 1:
        raise ValueError("subgroup_ids must be one-dimensional")
    subgroup = []
    for item in subgroup_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("subgroup_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("subgroup_ids contains an invalid node ID")
        subgroup.append(node)
    if len(subgroup) < candidate_threshold or any(left >= right for left, right in zip(subgroup, subgroup[1:])):
        raise ValueError("subgroup_ids must be sorted, unique and meet the threshold")

    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
        ("omega_lambda", omega_lambda, False),
        ("turnaround_overdensity", turnaround_overdensity, False),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
            raise ValueError(name + " lies outside its valid range")

    local_potential = boosted_turnaround_potential(
        position_array, acceleration_array, raw_array, subgroup, scale_factor,
        hubble_rate, omega_m, turnaround_overdensity)
    local_minimum = min(subgroup, key=lambda node: (local_potential[node], node))
    local_well, local_saddle, _ = locate_host_well(
        local_potential, rows, local_minimum, candidate_threshold)
    local_mask, _, _, _ = host_energy_mask(
        position_array, velocity_array, local_potential, local_well,
        local_saddle, local_minimum, scale_factor, hubble_rate, omega_m,
        omega_lambda, turnaround_overdensity)
    local_bound = [int(node) for node in subgroup if local_mask[node] == 1]
    bound_set = set(local_bound)
    surface = sorted({neighbor for node in local_bound for neighbor in rows[node]
                      if neighbor not in bound_set})
    return local_bound, [int(node) for node in surface]

import numpy as np


def bulk_binding_mask(position, canonical_velocity,
                            host_physical_potential, host_individual_mask,
                            subgroup_bound_ids, subgroup_surface_ids,
                            host_escape, minimum_id, scale_factor, hubble_rate):
    arrays = []
    for value, name in ((position, "position"),
                        (canonical_velocity, "canonical_velocity"),
                        (host_physical_potential, "host_physical_potential")):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size == 0:
            raise ValueError(name + " must be nonempty and one-dimensional")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        arrays.append(array)
    position_array, velocity_array, physical_array = arrays
    if position_array.shape != velocity_array.shape or position_array.shape != physical_array.shape:
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    mask_array = np.asarray(host_individual_mask, dtype=object)
    if mask_array.ndim != 1 or mask_array.size != node_count:
        raise ValueError("host_individual_mask must have one entry per node")
    mask = []
    for value in mask_array.tolist():
        if isinstance(value, (bool, np.bool_)):
            mask.append(int(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            mask.append(int(value))
        else:
            raise ValueError("host_individual_mask must contain only zero and one")

    def ids(value, name):
        array = np.asarray(value, dtype=object)
        if array.ndim != 1:
            raise ValueError(name + " must be one-dimensional")
        converted = []
        for item in array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError(name + " must contain integral node IDs")
            node = int(item)
            if node < 0 or node >= node_count:
                raise ValueError(name + " contains an invalid node ID")
            converted.append(node)
        if len(set(converted)) != len(converted):
            raise ValueError(name + " must contain distinct node IDs")
        return converted

    subgroup = ids(subgroup_bound_ids, "subgroup_bound_ids")
    surface = ids(subgroup_surface_ids, "subgroup_surface_ids")
    if isinstance(minimum_id, (bool, np.bool_)) or not isinstance(minimum_id, (int, np.integer)):
        raise ValueError("minimum_id must be integral")
    minimum_id = int(minimum_id)
    if minimum_id < 0 or minimum_id >= node_count:
        raise ValueError("minimum_id is out of range")
    scalars = []
    for name, value, positive in (("host_escape", host_escape, None),
                                  ("scale_factor", scale_factor, True),
                                  ("hubble_rate", hubble_rate, False)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar):
            raise ValueError(name + " must be finite")
        if positive is True and scalar <= 0.0:
            raise ValueError(name + " must be positive")
        if positive is False and scalar < 0.0:
            raise ValueError(name + " must be nonnegative")
        scalars.append(scalar)
    host_escape, scale_factor, hubble_rate = scalars

    output = list(mask)
    if not subgroup:
        return [int(value) for value in output]
    if not surface:
        raise ValueError("a nonempty subgroup must have a nonempty surface")
    if set(subgroup) & set(surface):
        raise ValueError("subgroup and surface IDs must be disjoint")
    host_ids = [node for node, value in enumerate(mask) if value == 1]
    if not host_ids:
        raise ValueError("a nonempty subgroup requires a host velocity sample")

    def wide_proxy():
        from fractions import Fraction as F
        x = [F(float(v)) for v in position_array]
        velocity = [F(float(v)) for v in velocity_array]
        scale, hubble = F(scale_factor), F(hubble_rate)
        host_velocity = sum(velocity[j] for j in host_ids) / len(host_ids)
        q = sum(x[j] - x[minimum_id] for j in subgroup) / len(subgroup)
        residual = sum(velocity[j] for j in subgroup) / len(subgroup) - host_velocity
        bulk = scale * hubble * q + residual / scale
        exact = F(float(np.max(physical_array[surface]))) + bulk * bulk / 2
        try:
            value = float(exact)
        except OverflowError as exc:
            raise ValueError("bulk proxy energy cannot be represented as a finite float") from exc
        if not np.isfinite(value):
            raise ValueError("bulk proxy energy cannot be represented as a finite float")
        return value

    proxy = wide_proxy()
    if proxy > host_escape:
        for node in subgroup:
            output[node] = 0
    return [int(value) for value in output]

def bound_mass_fraction(position, canonical_velocity, acceleration,
                              raw_potential, adjacency, seed_ids, scale_factor,
                              hubble_rate, omega_m, omega_lambda,
                              turnaround_overdensity, subgroup_min_size):
    boosted = boosted_turnaround_potential(
        position, acceleration, raw_potential, seed_ids, scale_factor,
        hubble_rate, omega_m, turnaround_overdensity)
    minimum = descend_to_minimum(boosted, adjacency, seed_ids)
    well, saddle, branches = locate_host_well(
        boosted, adjacency, minimum, subgroup_min_size)
    mask, escape, physical_potential, _ = host_energy_mask(
        position, canonical_velocity, boosted, well, saddle, minimum,
        scale_factor, hubble_rate, omega_m, omega_lambda,
        turnaround_overdensity)
    for branch in branches:
        rebound_ids, surface_ids = rebound_subgroup(
            position, canonical_velocity, acceleration, raw_potential, adjacency,
            branch, scale_factor, hubble_rate, omega_m, omega_lambda,
            turnaround_overdensity, subgroup_min_size)
        if rebound_ids:
            mask = bulk_binding_mask(
                position, canonical_velocity, physical_potential, mask,
                rebound_ids, surface_ids, escape, minimum, scale_factor,
                hubble_rate)
    return float(sum(mask) / len(well))
SCICODE_GOLD_EOF
