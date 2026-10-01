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
def decompose_independent_subnetworks(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    vectors = product - source
    if np.any(np.all(vectors == 0.0, axis=1)):
        raise ValueError("every reaction must change at least one species")

    count = vectors.shape[0]
    basis = []
    for index in range(count):
        trial = basis + [index]
        if np.linalg.matrix_rank(vectors[trial], tol=1e-9) == len(trial):
            basis.append(index)

    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(first, second):
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    # Each non-basis reaction closes a fundamental circuit with the basis
    # reactions carrying a nonzero coefficient; circuits define the blocks.
    basis_matrix = vectors[basis].T
    for index in range(count):
        if index in basis:
            continue
        coefficients = np.linalg.lstsq(basis_matrix, vectors[index], rcond=None)[0]
        for member, value in zip(basis, coefficients):
            if abs(value) > 1e-9:
                join(index, member)

    labels = np.empty(count, dtype=int)
    names = {}
    for index in range(count):
        root = find(index)
        if root not in names:
            names[root] = len(names)
        labels[index] = names[root]
    return labels

import numpy as np
def compute_elementary_flux_modes(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Reference implementation (canonical-basis double description)."""
    import numpy as np
    from math import gcd

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    if np.any(np.all(product == source, axis=1)):
        raise ValueError("every reaction must change at least one species")
    stoich = (product - source).astype(np.int64).T
    reactions = stoich.shape[1]

    def primitive(vector):
        divisor = 0
        for value in vector:
            divisor = gcd(divisor, int(abs(value)))
        return vector // divisor

    rays = [np.eye(reactions, dtype=np.int64)[index] for index in range(reactions)]
    for row in stoich:
        values = [int(row @ ray) for ray in rays]
        supports = [frozenset(np.flatnonzero(ray).tolist()) for ray in rays]
        kept = [ray for ray, value in zip(rays, values) if value == 0]
        positive = [index for index, value in enumerate(values) if value > 0]
        negative = [index for index, value in enumerate(values) if value < 0]
        for plus in positive:
            for minus in negative:
                union = supports[plus] | supports[minus]
                # Combinatorial adjacency test: no third ray may have its
                # support inside the union of the two parent supports.
                adjacent = all(
                    not supports[other] <= union
                    for other in range(len(rays))
                    if other not in (plus, minus)
                )
                if adjacent:
                    combined = values[plus] * rays[minus] - values[minus] * rays[plus]
                    kept.append(primitive(combined))
        rays = kept

    unique = {}
    for ray in rays:
        unique[tuple(int(value) for value in ray)] = ray
    ordered = sorted(unique.values(), key=lambda ray: tuple(np.flatnonzero(ray).tolist()))
    if not ordered:
        return np.zeros((0, reactions), dtype=int)
    return np.array(ordered, dtype=int)

import numpy as np
def build_reaction_graph(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    flux_modes: np.ndarray,
) -> np.ndarray:
    """Reference implementation (exhaustive cyclic-order search)."""
    import itertools
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    modes = np.asarray(flux_modes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    if np.any(np.all(product == source, axis=1)):
        raise ValueError("every reaction must change at least one species")
    count = source.shape[0]
    if modes.ndim != 2 or modes.shape[0] == 0 or modes.shape[1] != count:
        raise ValueError("flux_modes must have shape (p, r) with p >= 1")
    if not np.all((modes == 0.0) | (modes == 1.0)):
        raise ValueError("the elementary flux modes must be unitary")
    if np.any(modes.sum(axis=1) == 0.0) or np.any(modes.sum(axis=0) == 0.0):
        raise ValueError("the flux modes must be nonzero and cover every reaction")

    supports = [tuple(np.flatnonzero(row).tolist()) for row in modes]
    targets = {frozenset(support) for support in supports}
    sources = [tuple(row) for row in source.tolist()]
    siblings = [[j for j in range(count) if sources[j] == sources[i]] for i in range(count)]

    def close(edges):
        closed = set(edges)
        for tail, head in edges:
            for other in siblings[head]:
                closed.add((tail, other))
        return closed

    def cycle_sets(edges):
        successors = [sorted(h for t, h in edges if t == node) for node in range(count)]
        found = set()
        for start in range(count):
            stack = [(start, (start,))]
            while stack:
                node, path = stack.pop()
                for nxt in successors[node]:
                    if nxt == start:
                        found.add(frozenset(path))
                    elif nxt > start and nxt not in path:
                        stack.append((nxt, path + (nxt,)))
        return found

    def cyclic_orders(support):
        for perm in itertools.permutations(support[1:]):
            ring = (support[0],) + perm
            yield {(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))}

    def search(position, edges):
        closed = close(edges)
        if any(tail == head for tail, head in closed):
            return None
        cycles = cycle_sets(closed)
        if not cycles <= targets:
            return None
        if position == len(supports):
            return closed if cycles == targets else None
        for ring in cyclic_orders(supports[position]):
            result = search(position + 1, edges | ring)
            if result is not None:
                return result
        return None

    graph = search(0, set())
    if graph is None:
        raise ValueError("no common-source and flux-mode compatible graph exists")
    return np.array(sorted(graph), dtype=int).reshape(-1, 2)

import numpy as np
def compute_translation_complexes(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    reaction_edges: np.ndarray,
) -> np.ndarray:
    """Reference implementation (graph traversal of alpha_i - alpha_j = y_s(j) - y_p(i))."""
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    if source.ndim != 2 or source.shape != product.shape or source.shape[0] == 0:
        raise ValueError("complex arrays must share a two-dimensional (r, m) shape")
    for values in (source, product):
        if not np.all(np.isfinite(values)):
            raise ValueError("complex entries must be finite")
        if np.any(values < 0.0) or np.any(values != np.round(values)):
            raise ValueError("complex entries must be nonnegative integers")
    count, species = source.shape
    edges = np.asarray(reaction_edges)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("reaction_edges must have shape (q, 2)")
    if not np.all(np.isfinite(edges.astype(float))) or np.any(edges != np.round(edges)):
        raise ValueError("reaction indices must be integers")
    edges = edges.astype(int)
    if np.any(edges < 0) or np.any(edges >= count) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("edges must join two different existing reactions")
    source = source.astype(np.int64)
    product = product.astype(np.int64)

    neighbours = [[] for _ in range(count)]
    for tail, head in edges:
        # alpha[head] = alpha[tail] + product[tail] - source[head]
        neighbours[tail].append((head, product[tail] - source[head]))
        neighbours[head].append((tail, source[head] - product[tail]))

    alpha = np.zeros((count, species), dtype=np.int64)
    component = np.full(count, -1)
    for seed in range(count):
        if component[seed] >= 0:
            continue
        component[seed] = seed
        queue = [seed]
        while queue:
            current = queue.pop(0)
            for other, shift in neighbours[current]:
                if component[other] < 0:
                    component[other] = seed
                    alpha[other] = alpha[current] + shift
                    queue.append(other)
    for tail, head in edges:
        if not np.array_equal(product[tail] + alpha[tail], source[head] + alpha[head]):
            raise ValueError("the product-to-source equations are inconsistent")
    for seed in np.unique(component):
        members = np.flatnonzero(component == seed)
        translated = np.vstack([source[members] + alpha[members], product[members] + alpha[members]])
        alpha[members] += np.maximum(0, -translated.min(axis=0))
    return alpha.astype(int)

import numpy as np
def compute_network_deficiency(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    shift = np.asarray(translation, dtype=float)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("complex arrays must be two-dimensional with r >= 1")
    if product.shape != source.shape or shift.shape != source.shape:
        raise ValueError("the three arrays must share the (r, m) shape")
    for values in (source, product, shift):
        if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
            raise ValueError("entries must be finite integers")
    tails_c = source + shift
    heads_c = product + shift
    if min(source.min(), product.min(), tails_c.min(), heads_c.min()) < 0.0:
        raise ValueError("original and translated complexes must be nonnegative")
    if np.any(np.all(source == product, axis=1)):
        raise ValueError("every reaction must change at least one species")

    index = {}
    for row in np.vstack([tails_c, heads_c]):
        index.setdefault(tuple(row), len(index))
    size = len(index)
    tails = [index[tuple(row)] for row in tails_c]
    heads = [index[tuple(row)] for row in heads_c]
    linked = np.eye(size, dtype=bool)
    reach = np.eye(size, dtype=bool)
    for tail, head in zip(tails, heads):
        linked[tail, head] = linked[head, tail] = True
        reach[tail, head] = True
    for pivot in range(size):
        linked |= linked[:, [pivot]] & linked[[pivot], :]
        reach |= reach[:, [pivot]] & reach[[pivot], :]
    linkage = len({tuple(row) for row in linked})
    rank = int(np.linalg.matrix_rank(product - source, tol=1e-9))
    reversible = all(reach[head, tail] for tail, head in zip(tails, heads))
    return np.array([size, linkage, rank, size - linkage - rank, int(reversible)], dtype=int)

import numpy as np
def build_generalized_network(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import itertools
    import numpy as np

    source = np.asarray(source_complexes, dtype=float)
    product = np.asarray(product_complexes, dtype=float)
    shift = np.asarray(translation, dtype=float)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("complex arrays must be two-dimensional with r >= 1")
    if product.shape != source.shape or shift.shape != source.shape:
        raise ValueError("the three arrays must share the (r, m) shape")
    for values in (source, product, shift):
        if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
            raise ValueError("entries must be finite integers")
    source = source.astype(int)
    product = product.astype(int)
    shift = shift.astype(int)
    tails_c = source + shift
    heads_c = product + shift
    if min(source.min(), product.min(), tails_c.min(), heads_c.min()) < 0:
        raise ValueError("original and translated complexes must be nonnegative")

    order = []
    for k in range(source.shape[0]):
        for complex_ in (tuple(tails_c[k]), tuple(heads_c[k])):
            if complex_ not in order:
                order.append(complex_)
    labels = {complex_: [] for complex_ in order}
    for k in range(source.shape[0]):
        kinetic = tuple(source[k])
        if kinetic not in labels[tuple(tails_c[k])]:
            labels[tuple(tails_c[k])].append(kinetic)

    vertex = {}
    stoichiometric, kinetic_rows = [], []
    for complex_ in order:
        if not labels[complex_]:
            raise ValueError("a translated complex is the source of no reaction")
        for kinetic in labels[complex_]:
            vertex[(complex_, kinetic)] = len(stoichiometric)
            stoichiometric.append(complex_)
            kinetic_rows.append(kinetic)

    edges = []
    for k in range(source.shape[0]):
        tail = vertex[(tuple(tails_c[k]), tuple(source[k]))]
        head_complex = tuple(heads_c[k])
        head = vertex[(head_complex, labels[head_complex][0])]
        edges.append((tail, head, k))
    for complex_ in order:
        copies = labels[complex_]
        for first, second in itertools.pairwise(copies):
            edges.append((vertex[(complex_, first)], vertex[(complex_, second)], -1))
    return (
        np.array(stoichiometric, dtype=int),
        np.array(kinetic_rows, dtype=int),
        np.array(edges, dtype=int).reshape(-1, 3),
    )

import numpy as np
def compute_tree_constants(
    num_vertices: int, gcrn_edges: np.ndarray, edge_rates: np.ndarray
) -> np.ndarray:
    """Reference implementation (directed matrix-tree theorem)."""
    import numpy as np

    def is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not is_integer(num_vertices) or num_vertices < 1:
        raise ValueError("num_vertices must be a positive integer")
    size = int(num_vertices)
    edges = np.asarray(gcrn_edges)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError("gcrn_edges must have shape (E, 2) or (E, 3)")
    if not np.all(np.isfinite(edges.astype(float))) or np.any(edges != np.round(edges)):
        raise ValueError("edge entries must be integers")
    ends = edges[:, :2].astype(int)
    if np.any(ends < 0) or np.any(ends >= size) or np.any(ends[:, 0] == ends[:, 1]):
        raise ValueError("edges must join two different existing vertices")
    rates = np.asarray(edge_rates, dtype=float)
    if rates.shape != (ends.shape[0],) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("edge_rates must hold one finite positive rate per edge")

    weights = np.zeros((size, size))
    linked = np.eye(size, dtype=bool)
    for (tail, head), rate in zip(ends, rates):
        weights[tail, head] += rate
        linked[tail, head] = linked[head, tail] = True
    for pivot in range(size):
        linked |= linked[:, [pivot]] & linked[[pivot], :]
    laplacian = np.diag(weights.sum(axis=1)) - weights
    constants = np.ones(size)
    for vertex in range(size):
        others = [u for u in np.flatnonzero(linked[vertex]) if u != vertex]
        if others:
            constants[vertex] = np.linalg.det(laplacian[np.ix_(others, others)])
    return constants

import numpy as np
def solve_phantom_rate(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    rate_constants: np.ndarray,
    tree_constants_fn,
    rate_bracket: tuple = (1e-8, 1e8),
) -> np.ndarray:
    """Reference implementation (log-bisection on the consistency residual)."""
    import numpy as np

    kinetic = np.asarray(kinetic_complexes, dtype=float)
    edges = np.asarray(gcrn_edges)
    rates = np.asarray(rate_constants, dtype=float).ravel()
    if kinetic.ndim != 2 or kinetic.shape[0] == 0:
        raise ValueError("kinetic_complexes must have shape (V, m)")
    if edges.ndim != 2 or edges.shape[1] != 3 or edges.shape[0] == 0:
        raise ValueError("gcrn_edges must have shape (E, 3)")
    edges = edges.astype(int)
    size = kinetic.shape[0]
    if np.any(edges[:, :2] < 0) or np.any(edges[:, :2] >= size):
        raise ValueError("edge endpoints must be existing vertices")
    labels = edges[:, 2]
    if np.any(labels < -1) or (np.any(labels >= 0) and labels.max() >= rates.size):
        raise ValueError("every effective edge needs a rate constant")
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("rate constants must be finite and positive")
    low, high = (float(value) for value in rate_bracket)
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
        raise ValueError("rate_bracket must satisfy 0 < low < high")

    parent = list(range(size))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node

    forest = []
    for position, (tail, head, _) in enumerate(edges):
        root_t, root_h = find(tail), find(head)
        if root_t != root_h:
            parent[root_h] = root_t
            forest.append(position)
    classes = len({find(v) for v in range(size)})
    differences = kinetic[edges[forest, 1]] - kinetic[edges[forest, 0]]
    rank = np.linalg.matrix_rank(differences, tol=1e-9) if forest else 0
    deficiency = size - classes - rank
    phantom = int(np.sum(labels == -1))
    if phantom == 0 and deficiency == 0:
        return np.zeros(0, dtype=float)
    if not (phantom == 1 and deficiency == 1):
        raise ValueError("unsupported numbers of phantom edges and kinetic deficiency")
    cokernel = np.linalg.svd(differences.T)[2][-1]

    def residual(log_sigma):
        edge_rates = np.where(labels >= 0, rates[np.maximum(labels, 0)], np.exp(log_sigma))
        tree = np.asarray(tree_constants_fn(size, edges, edge_rates), dtype=float)
        if tree.shape != (size,) or np.any(tree <= 0.0) or not np.all(np.isfinite(tree)):
            raise ValueError("tree constants must be finite and positive")
        gaps = np.log(tree[edges[forest, 1]]) - np.log(tree[edges[forest, 0]])
        return float(cokernel @ gaps)

    left, right = np.log(low), np.log(high)
    value_left, value_right = residual(left), residual(right)
    if value_left == 0.0:
        return np.array([low])
    if value_right == 0.0:
        return np.array([high])
    if value_left * value_right > 0.0:
        raise ValueError("no phantom rate in the bracket satisfies the condition")
    for _ in range(200):
        middle = 0.5 * (left + right)
        value = residual(middle)
        if value == 0.0 or right - left < 1e-15:
            left = right = middle
            break
        if (value > 0.0) == (value_left > 0.0):
            left, value_left = middle, value
        else:
            right = middle
    return np.array([float(np.exp(0.5 * (left + right)))])

import numpy as np
def parametrize_steady_states(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    tree_constants: np.ndarray,
    free_species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    kinetic = np.asarray(kinetic_complexes, dtype=float)
    edges = np.asarray(gcrn_edges)
    tree = np.asarray(tree_constants, dtype=float)
    if kinetic.ndim != 2 or kinetic.shape[0] == 0 or kinetic.shape[1] == 0:
        raise ValueError("kinetic_complexes must have shape (V, m)")
    if not np.all(np.isfinite(kinetic)):
        raise ValueError("kinetic complexes must be finite")
    size, species = kinetic.shape
    if edges.ndim != 2 or edges.shape[1] < 2 or edges.shape[0] == 0:
        raise ValueError("gcrn_edges must have shape (E, 2) or (E, 3)")
    ends = edges[:, :2].astype(int)
    if np.any(ends < 0) or np.any(ends >= size):
        raise ValueError("edge endpoints must be existing vertices")
    if tree.shape != (size,) or not np.all(np.isfinite(tree)) or np.any(tree <= 0.0):
        raise ValueError("tree constants must be finite and positive")
    free = [int(value) for value in np.asarray(free_species, dtype=float).ravel()]
    if len(set(free)) != len(free) or any(value < 0 or value >= species for value in free):
        raise ValueError("free_species must hold distinct in-range species indices")

    matrix = kinetic[ends[:, 1]] - kinetic[ends[:, 0]]
    rhs = np.log(tree[ends[:, 1]]) - np.log(tree[ends[:, 0]])
    others = [index for index in range(species) if index not in free]
    determined = len(others)
    if np.linalg.matrix_rank(matrix, tol=1e-9) != determined:
        raise ValueError("the free species do not match the dimension of the solution set")
    if determined and np.linalg.matrix_rank(matrix[:, others], tol=1e-9) != determined:
        raise ValueError("the free species do not determine the other log-concentrations")

    offset = np.zeros(species)
    exponents = np.zeros((species, len(free)))
    for column, index in enumerate(free):
        exponents[index, column] = 1.0
    if determined:
        pseudo = np.linalg.pinv(matrix[:, others])
        particular = pseudo @ rhs
        scale = max(1.0, float(np.linalg.norm(rhs)))
        if np.linalg.norm(matrix[:, others] @ particular - rhs) > 1e-9 * scale:
            raise ValueError("the tree constants admit no complex-balanced equilibrium")
        offset[others] = particular
        if free:
            exponents[others] = -pseudo @ matrix[:, free]
    elif np.linalg.norm(rhs) > 1e-9:
        raise ValueError("the tree constants admit no complex-balanced equilibrium")
    return offset, exponents

import numpy as np
def solve_class_steady_state(
    log_offset: np.ndarray,
    exponents: np.ndarray,
    conservation_matrix: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    """Reference implementation (damped Newton on log-totals)."""
    import numpy as np

    offset = np.asarray(log_offset, dtype=float)
    powers = np.asarray(exponents, dtype=float)
    weights = np.asarray(conservation_matrix, dtype=float)
    target = np.asarray(totals, dtype=float)
    if offset.ndim != 1 or offset.size == 0:
        raise ValueError("log_offset must be a nonempty vector")
    species = offset.size
    if powers.ndim != 2 or powers.shape[0] != species:
        raise ValueError("exponents must have shape (m, d)")
    laws = powers.shape[1]
    if weights.size == 0:
        weights = np.zeros((0, species))
    if weights.shape != (laws, species) or target.shape != (laws,):
        raise ValueError("conservation data must have shapes (d, m) and (d,)")
    for values in (offset, powers, weights, target):
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must be finite")
    if np.any(weights < 0.0) or np.any(weights.sum(axis=1) <= 0.0) or np.any(target <= 0.0):
        raise ValueError("weights must be nonnegative with nonzero rows and totals positive")
    if laws == 0:
        return np.exp(offset)

    def evaluate(point):
        state = np.exp(offset + powers @ point)
        conserved = weights @ state
        return state, conserved, np.log(conserved) - np.log(target)

    point = np.zeros(laws)
    state, conserved, gap = evaluate(point)
    for _ in range(400):
        if np.max(np.abs(gap)) < 1e-14:
            break
        jacobian = (weights * state) @ powers / conserved[:, None]
        try:
            step = np.linalg.solve(jacobian, -gap)
        except np.linalg.LinAlgError:
            raise ValueError("singular Jacobian while imposing the conservation laws")
        damping, norm = 1.0, np.linalg.norm(gap)
        while True:
            trial = evaluate(point + damping * step)
            if np.all(np.isfinite(trial[2])) and np.linalg.norm(trial[2]) < (1.0 - 1e-4 * damping) * norm:
                break
            damping *= 0.5
            if damping < 1e-12:
                raise ValueError("the conservation laws could not be imposed")
        point = point + damping * step
        state, conserved, gap = trial
    if np.max(np.abs(conserved / target - 1.0)) > 1e-12:
        raise ValueError("the conservation laws could not be imposed")
    return state

import numpy as np
def run_equilibrium_pipeline(
    source_complexes: "np.ndarray | None" = None,
    product_complexes: "np.ndarray | None" = None,
    rate_constants: "np.ndarray | None" = None,
    conservation_matrix: "np.ndarray | None" = None,
    totals: "np.ndarray | None" = None,
    free_species: "np.ndarray | None" = None,
    target_species: "int | None" = None,
    rate_bracket: tuple = (1e-8, 1e8),
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    reactions = [([1], [0]), ([0], [1]), ([0], [2]), ([2], [0]), ([2], [3]),
                 ([3, 4], [6]), ([6], [3, 4]), ([6], [0, 5]), ([1, 5], [7]),
                 ([7], [1, 5]), ([7], [1, 4]), ([2, 5], [8]), ([8], [2, 5]),
                 ([8], [2, 4])]
    bench_source = np.zeros((14, 9), dtype=int)
    bench_product = np.zeros((14, 9), dtype=int)
    for row, (left, right) in enumerate(reactions):
        for species in left:
            bench_source[row, species] += 1
        for species in right:
            bench_product[row, species] += 1
    if source_complexes is None:
        source_complexes = bench_source
    if product_complexes is None:
        product_complexes = bench_product
    if rate_constants is None:
        rate_constants = [1.93, 1.72, 1.90, 0.19, 0.51, 3.39, 2.90,
                          1.45, 0.28, 4.31, 0.35, 0.60, 2.84, 1.66]
    if conservation_matrix is None:
        conservation_matrix = [[1, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1]]
    if totals is None:
        totals = [3.31, 4.21]
    if free_species is None:
        free_species = [6, 4]
    if target_species is None:
        target_species = 6
    source = np.asarray(source_complexes)
    product = np.asarray(product_complexes)
    rates = np.asarray(rate_constants, dtype=float)
    labels = decompose_independent_subnetworks(source, product)
    if rates.shape != (source.shape[0],) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("rate_constants must hold one finite positive value per reaction")
    def is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if not (is_integer(target_species) and 0 <= int(target_species) < source.shape[1]):
        raise ValueError("target_species must be an integer index of an existing species")

    translation = np.zeros(source.shape, dtype=int)
    for block in np.unique(labels):
        members = np.flatnonzero(labels == block)
        block_source, block_product = source[members], product[members]
        summary = compute_network_deficiency(
            block_source, block_product, np.zeros(block_source.shape, dtype=int)
        )
        if summary[3] == 0 and summary[4] == 1:
            continue
        modes = compute_elementary_flux_modes(block_source, block_product)
        graph = build_reaction_graph(block_source, block_product, modes)
        translation[members] = compute_translation_complexes(
            block_source, block_product, graph
        )

    merged = compute_network_deficiency(source, product, translation)
    if merged[3] != 0 or merged[4] != 1:
        raise ValueError("the merged translated network is not weakly reversible with deficiency zero")
    stoichiometric, kinetic, edges = build_generalized_network(source, product, translation)
    phantom = solve_phantom_rate(
        kinetic, edges, rates, compute_tree_constants, rate_bracket
    )
    edge_rates = np.array(
        [rates[k] if k >= 0 else phantom[0] for k in edges[:, 2]], dtype=float
    )
    tree = compute_tree_constants(stoichiometric.shape[0], edges, edge_rates)
    offset, exponents = parametrize_steady_states(
        kinetic, edges, tree, np.asarray(free_species)
    )
    state = solve_class_steady_state(
        offset, exponents, np.asarray(conservation_matrix, dtype=float), np.asarray(totals, dtype=float)
    )
    result = float(state[int(target_species)])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("the steady-state concentration must be finite and positive")
    return result
SCICODE_GOLD_EOF
