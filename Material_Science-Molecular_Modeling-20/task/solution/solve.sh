#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def validate_topology(points: list, kinds: list, edge_points: list, candidate_edges: list, linker_type: str) -> list[int]:
    import math
    if linker_type not in ('ditopic', 'multitopic'):
        raise ValueError('unsupported linker type')
    if len(kinds) != len(points) or any((kind not in ('V', 'EC') for kind in kinds)):
        raise ValueError('inconsistent labels')
    if any((len(p) != 3 or not all((math.isfinite(x) for x in p)) for p in points + edge_points)):
        raise ValueError('invalid point')
    seen = set()
    valid = []
    for edge_index, pair in enumerate(candidate_edges):
        if len(pair) != 2 or any((type(i) is not int or i < 0 or i >= len(points) for i in pair)):
            raise ValueError('invalid edge')
        i, j = pair
        if (i, j) in seen:
            raise ValueError('duplicate edge')
        seen.add((i, j))
        a, b = (points[i], points[j])
        vector = [b[k] - a[k] for k in range(3)]
        length_sq = sum((x * x for x in vector))
        if length_sq <= 1e-16:
            raise ValueError('coincident endpoints')
        end_kind = 'V' if linker_type == 'ditopic' else 'EC'
        if kinds[i] != 'V' or kinds[j] != end_kind:
            continue
        for point in edge_points:
            projection = sum(((point[k] - a[k]) * vector[k] for k in range(3))) / length_sq
            distance_sq = sum(((point[k] - a[k] - projection * vector[k]) ** 2 for k in range(3)))
            if 0.0 < projection < 1.0 and distance_sq <= 1e-16:
                valid.append(edge_index)
                break
    return valid

def periodic_neighbor(a_frac: list, b_frac: list, cell: list) -> tuple:
    import itertools
    import math
    if any((len(point) != 3 or any((not math.isfinite(x) or not 0 <= x < 1 for x in point)) for point in (a_frac, b_frac))):
        raise ValueError('invalid fractional point')
    if len(cell) != 3 or any((len(row) != 3 or not all((math.isfinite(x) for x in row)) for row in cell)):
        raise ValueError('invalid cell')
    a, b, c = cell
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    if abs(determinant) <= 1e-12:
        raise ValueError('singular cell')
    images = []
    for shift in itertools.product((-1, 0, 1), repeat=3):
        fractional_delta = [b_frac[k] + shift[k] - a_frac[k] for k in range(3)]
        cartesian_delta = [sum((fractional_delta[k] * cell[k][j] for k in range(3))) for j in range(3)]
        distance = math.sqrt(sum((x * x for x in cartesian_delta)))
        if distance > 1e-08:
            images.append((shift, distance))
    if not images:
        raise ValueError('no positive image')
    minimum = min((distance for shift, distance in images))
    return min(((shift, distance) for shift, distance in images if distance <= minimum + 1e-12))

def reassign_fragments(distance_matrix: list) -> tuple:
    import math
    if not distance_matrix:
        return ([], 0.0)
    ncols = len(distance_matrix[0])
    if any((len(row) != ncols for row in distance_matrix)):
        raise ValueError('ragged distance matrix')
    if any((not math.isfinite(distance) or distance < 0 for row in distance_matrix for distance in row)):
        raise ValueError('invalid distance')
    cutoff = 4.0
    pairs = sorted(((distance, i, j) for i, row in enumerate(distance_matrix) for j, distance in enumerate(row) if distance <= cutoff + 1e-08))
    assignment = [-1] * len(distance_matrix)
    used_columns = set()
    total_distance = 0.0
    for distance, i, j in pairs:
        if assignment[i] < 0 and j not in used_columns:
            assignment[i] = j
            used_columns.add(j)
            total_distance += distance
    return (assignment, float(total_distance))

def fine_alignment_loss(node_groups: list, linker_groups: list) -> float:
    import math
    if len(node_groups) != len(linker_groups):
        raise ValueError('edge count mismatch')
    for group in node_groups + linker_groups:
        if not group or any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in group)):
            raise ValueError('invalid atom group')
    total = 0.0
    for node_atoms, linker_atoms in zip(node_groups, linker_groups):
        minimum = min((sum(((a[k] - b[k]) ** 2 for k in range(3))) for a in node_atoms for b in linker_atoms))
        total += minimum
    return float(total)

def propagate_coordinate(start: list, direction: list, linker_length: float, const_length: float, offset_length: float) -> tuple:
    import math
    if any((len(vector) != 3 or not all((math.isfinite(x) for x in vector)) for vector in (start, direction))):
        raise ValueError('invalid vector')
    if any((not math.isfinite(length) or length < 0 for length in (linker_length, const_length, offset_length))):
        raise ValueError('invalid length')
    norm = math.sqrt(sum((x * x for x in direction)))
    if norm <= 1e-08:
        raise ValueError('zero direction')
    target = linker_length + 2 * const_length + 2 * offset_length
    if target <= 0:
        raise ValueError('nonpositive target')
    endpoint = tuple((start[k] + target * direction[k] / norm for k in range(3)))
    return (float(target), endpoint)

def fractional_cell_loss(old_cart: list, new_cart: list, old_cell: list, new_cell: list) -> float:
    import math
    for cell in (old_cell, new_cell):
        if len(cell) != 3 or any((len(row) != 3 or not all((math.isfinite(x) for x in row)) for row in cell)):
            raise ValueError('invalid cell')
    if any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in old_cart + new_cart)):
        raise ValueError('invalid coordinate')
    if len(old_cart) != len(new_cart):
        raise ValueError('shape mismatch')

    def inverse(m):
        determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        if abs(determinant) <= 1e-12:
            raise ValueError('singular cell')
        inv = [[0.0] * 3 for _ in range(3)]
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / determinant
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / determinant
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / determinant
        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / determinant
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / determinant
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / determinant
        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / determinant
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / determinant
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / determinant
        return inv
    inv_old = inverse(old_cell)
    inv_new = inverse(new_cell)
    total = 0.0
    for old_point, new_point in zip(old_cart, new_cart):
        old_fractional = [sum((old_point[k] * inv_old[k][j] for k in range(3))) for j in range(3)]
        new_fractional = [sum((new_point[k] * inv_new[k][j] for k in range(3))) for j in range(3)]
        total += sum(((new_fractional[j] - old_fractional[j]) ** 2 for j in range(3)))
    return float(total)

def cap_defects(fragment: list, marked_index: int, axis_index: int, sites: list, vacant_vectors: list) -> list:
    import math
    if not fragment or any((type(i) is not int or i < 0 or i >= len(fragment) for i in (marked_index, axis_index))) or marked_index == axis_index:
        raise ValueError('invalid atom indices')
    if len(sites) != len(vacant_vectors):
        raise ValueError('site count mismatch')
    if any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in fragment + sites + vacant_vectors)):
        raise ValueError('invalid coordinate')

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def unit(vector):
        norm = math.sqrt(sum((x * x for x in vector)))
        if norm <= 1e-08:
            raise ValueError('zero axis')
        return [x / norm for x in vector]
    origin = fragment[marked_index]
    local = [[point[k] - origin[k] for k in range(3)] for point in fragment]
    source = unit(local[axis_index])
    result = []
    for site, vacant in zip(sites, vacant_vectors):
        target = unit(vacant)
        normal = cross(source, target)
        sine = math.sqrt(sum((x * x for x in normal)))
        cosine = max(-1.0, min(1.0, sum((source[k] * target[k] for k in range(3)))))
        if sine <= 1e-12:
            if cosine >= 0:
                rotated = local
            else:
                index = min(range(3), key=lambda k: (abs(source[k]), k))
                basis = [float(k == index) for k in range(3)]
                axis = unit(cross(source, basis))
                rotated = [[2 * sum((axis[k] * point[k] for k in range(3))) * axis[j] - point[j] for j in range(3)] for point in local]
        else:
            axis = [x / sine for x in normal]
            rotated = []
            for point in local:
                cross_term = cross(axis, point)
                dot = sum((axis[k] * point[k] for k in range(3)))
                rotated.append([point[k] * cosine + cross_term[k] * sine + axis[k] * dot * (1 - cosine) for k in range(3)])
        result.append([tuple((site[k] + point[k] for k in range(3))) for point in rotated])
    return result

def rank_candidates(candidates: list) -> int:
    import math

    ids = [candidate["id"] for candidate in candidates]
    if (
        any(type(i) is not int for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid candidate ids")

    scored = []

    for candidate in candidates:
        frac = candidate["frac_points"]
        cell = candidate["old_cell"]
        edges = candidate["edges"]

        if len(frac) < 2 or len(edges) != len(frac) - 1:
            raise ValueError("tree size mismatch")

        if len(cell) != 3 or any(
            len(row) != 3
            or not all(math.isfinite(x) for x in row)
            for row in cell
        ):
            raise ValueError("invalid old cell")

        if any(
            len(point) != 3
            or any(
                not math.isfinite(x) or not 0 <= x < 1
                for x in point
            )
            for point in frac
        ):
            raise ValueError("invalid fractional coordinate")

        if (
            len(candidate["node_groups"]) != len(edges)
            or len(candidate["connection_sites"]) != len(edges)
            or len(candidate["fragment_sites"])
            != len(candidate["fragment_groups"])
            or len(candidate["linker_lengths"])
            != len(candidate["fragment_groups"])
        ):
            raise ValueError("assignment shape mismatch")

        if any(
            len(point) != 3
            or not all(math.isfinite(x) for x in point)
            for point in (
                candidate["connection_sites"]
                + candidate["fragment_sites"]
            )
        ):
            raise ValueError("invalid assignment geometry")

        distances = [
            [
                math.dist(site, anchor)
                for anchor in candidate["fragment_sites"]
            ]
            for site in candidate["connection_sites"]
        ]

        reached = {0}
        old = [None] * len(frac)
        old[0] = tuple(
            sum(frac[0][k] * cell[k][j] for k in range(3))
            for j in range(3)
        )
        directions = []

        for edge in edges:
            if len(edge) != 2 or any(
                type(i) is not int or i < 0 or i >= len(frac)
                for i in edge
            ):
                raise ValueError("invalid tree edge")

            i, j = edge
            if i not in reached or j in reached:
                raise ValueError("tree is not in propagation order")

            shift, _ = periodic_neighbor(
                frac[i], frac[j], cell
            )

            fractional_delta = [
                frac[j][k] + shift[k] - frac[i][k]
                for k in range(3)
            ]
            direction = [
                sum(
                    fractional_delta[k] * cell[k][axis]
                    for k in range(3)
                )
                for axis in range(3)
            ]

            directions.append(direction)
            old[j] = tuple(
                old[i][k] + direction[k]
                for k in range(3)
            )
            reached.add(j)

        valid = validate_topology(
            old,
            candidate["kinds"],
            candidate["edge_points"],
            edges,
            candidate["linker_type"],
        )
        if len(valid) != len(edges):
            continue

        assignment, _ = reassign_fragments(distances)
        if any(index < 0 for index in assignment):
            continue

        assigned_groups = [
            candidate["fragment_groups"][index]
            for index in assignment
        ]
        fine_loss = fine_alignment_loss(
            candidate["node_groups"], assigned_groups
        )
        if fine_loss > 0.20 + 1e-8:
            continue

        new = [None] * len(frac)
        new[0] = old[0]

        for edge_index, (i, j) in enumerate(edges):
            _, new[j] = propagate_coordinate(
                new[i],
                directions[edge_index],
                candidate["linker_lengths"][assignment[edge_index]],
                candidate["const_length"],
                candidate["offset_length"],
            )

        if any(
            type(i) is not int or i < 0 or i >= len(new)
            for i in candidate["cap_nodes"]
        ):
            raise ValueError("invalid cap node")

        caps = cap_defects(
            candidate["cap_fragment"],
            candidate["marked_index"],
            candidate["axis_index"],
            [new[i] for i in candidate["cap_nodes"]],
            candidate["vacant_vectors"],
        )

        if any(
            math.dist(point, node) < 0.25 - 1e-8
            for cap in caps
            for atom_index, point in enumerate(cap)
            if atom_index != candidate["marked_index"]
            for node in new
        ):
            continue

        cell_loss = fractional_cell_loss(
            old, new, cell, candidate["new_cell"]
        )
        scored.append((cell_loss, candidate["id"]))

    if not scored:
        raise ValueError("no admissible candidate")

    minimum = min(loss for loss, candidate_id in scored)

    return min(
        candidate_id
        for loss, candidate_id in scored
        if loss <= minimum + 1e-12
    )
SCICODE_GOLD_EOF
