#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def enumerate_valid_phasings(ploidy: int, genotypes: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    if not (isinstance(ploidy, (int, np.integer)) and not isinstance(ploidy, bool)
            and int(ploidy) >= 1):
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)

    counts = np.asarray(genotypes, dtype=float)
    if counts.ndim != 1 or counts.size < 1 or counts.size > 4:
        raise ValueError("genotypes must be a one-dimensional array of one to four entries")
    if not np.all(np.isfinite(counts)):
        raise ValueError("genotypes must contain only finite entries")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    counts = np.round(counts).astype(int)
    if np.any(counts < 0) or np.any(counts > ploidy):
        raise ValueError("genotypes entries must lie between zero and ploidy inclusive")

    n_positions = int(counts.size)
    # The allele patterns a single haplotype can carry over the P positions,
    # listed in ascending lexicographic order.
    patterns = [np.array(p, dtype=int)
                for p in itertools.product((0, 1), repeat=n_positions)]

    # Choosing a multiset of K patterns with repetition already produces rows in
    # ascending lexicographic order, which is the canonical form, and the
    # combinations are generated in ascending lexicographic order themselves.
    phasings = []
    for choice in itertools.combinations_with_replacement(range(len(patterns)), ploidy):
        matrix = np.array([patterns[i] for i in choice], dtype=int)
        if np.array_equal(matrix.sum(axis=0), counts):
            phasings.append(matrix)

    return np.array(phasings, dtype=int).reshape(-1, ploidy, n_positions)

def compute_phasing_potentials(reads: np.ndarray, phasings: np.ndarray,
                                       positions: np.ndarray,
                                       error_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    states = np.asarray(phasings, dtype=float)
    if states.ndim != 3 or states.size < 1:
        raise ValueError("phasings must be a non-empty three-dimensional array")
    if not np.all(np.isin(states, (0.0, 1.0))):
        raise ValueError("phasings entries must be 0 or 1")
    states = states.astype(int)

    columns = np.asarray(positions, dtype=float)
    if columns.ndim != 1 or columns.size != states.shape[2]:
        raise ValueError("positions must be one-dimensional and match the phasing width")
    if not np.allclose(columns, np.round(columns), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    columns = np.round(columns).astype(int)
    if np.unique(columns).size != columns.size:
        raise ValueError("positions entries must be distinct")
    if np.any(columns < 0) or np.any(columns >= fragments.shape[1]):
        raise ValueError("positions entries must be valid column indices of reads")

    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    epsilon = float(error_rate)

    width = int(columns.size)
    observed = fragments[:, columns]
    # Only fragments with a called allele at every position of the run inform
    # the joint phase of that run.
    informative = observed[np.all(observed >= 0, axis=1)]
    if informative.shape[0] == 0:
        return np.ones(states.shape[0], dtype=float)

    potentials = np.empty(states.shape[0], dtype=float)
    for m in range(states.shape[0]):
        # distance[r, k] is the Hamming distance between fragment r and
        # haplotype k of this phasing over the run.
        distance = (states[m][None, :, :] != informative[:, None, :]).sum(axis=2)
        likelihood = (epsilon ** distance) * ((1.0 - epsilon) ** (width - distance))
        potentials[m] = float(likelihood.sum())

    return potentials

def build_snp_line_graph(reads: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    # A fragment covering fewer than two SNPs carries no phase information and
    # therefore contributes no edge to the SNP graph.
    pairs = set()
    for fragment in fragments:
        covered = np.flatnonzero(fragment >= 0).tolist()
        for first, second in itertools.combinations(covered, 2):
            pairs.add((int(first), int(second)))

    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)

def build_transition_matrix(parent_positions: np.ndarray,
                                    parent_phasings: np.ndarray,
                                    child_positions: np.ndarray,
                                    child_phasings: np.ndarray,
                                    joint_positions: np.ndarray,
                                    joint_phasings: np.ndarray,
                                    joint_potentials: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _positions(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.allclose(array, np.round(array), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} entries must be integer valued")
        array = np.round(array).astype(int)
        if np.unique(array).size != array.size:
            raise ValueError(f"{name} entries must be distinct")
        return array

    def _phasings(values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim != 3 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty three-dimensional array")
        if not np.all(np.isin(array, (0.0, 1.0))):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(int)

    def _canonical(matrix):
        return matrix[np.lexsort(matrix[:, ::-1].T)]

    parent_cols = _positions(parent_positions, "parent_positions")
    child_cols = _positions(child_positions, "child_positions")
    joint_cols = _positions(joint_positions, "joint_positions")
    parents = _phasings(parent_phasings, "parent_phasings")
    children = _phasings(child_phasings, "child_phasings")
    joints = _phasings(joint_phasings, "joint_phasings")

    if not (parents.shape[1] == children.shape[1] == joints.shape[1]):
        raise ValueError("the three phasing arrays must share the same ploidy")
    if parents.shape[2] != parent_cols.size or children.shape[2] != child_cols.size:
        raise ValueError("each phasing width must match its position array")
    if joints.shape[2] != joint_cols.size:
        raise ValueError("the joint phasing width must match joint_positions")
    if not (set(parent_cols.tolist()) <= set(joint_cols.tolist())
            and set(child_cols.tolist()) <= set(joint_cols.tolist())):
        raise ValueError("parent and child positions must be contained in joint_positions")

    weights = np.asarray(joint_potentials, dtype=float)
    if weights.ndim != 1 or weights.size != joints.shape[0]:
        raise ValueError("joint_potentials must be one-dimensional of length Mj")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("joint_potentials entries must be finite and non-negative")

    order = joint_cols.tolist()
    parent_index = [order.index(int(c)) for c in parent_cols]
    child_index = [order.index(int(c)) for c in child_cols]

    parent_key = {_canonical(p).tobytes(): i for i, p in enumerate(parents)}
    child_key = {_canonical(c).tobytes(): j for j, c in enumerate(children)}

    support = np.zeros((parents.shape[0], children.shape[0]), dtype=float)
    for m, joint in enumerate(joints):
        i = parent_key.get(_canonical(joint[:, parent_index]).tobytes())
        j = child_key.get(_canonical(joint[:, child_index]).tobytes())
        if i is not None and j is not None:
            support[i, j] += weights[m]

    row_sums = support.sum(axis=1)
    transition = np.empty_like(support)
    uniform = np.full(children.shape[0], 1.0 / float(children.shape[0]))
    for i in range(parents.shape[0]):
        transition[i] = support[i] / row_sums[i] if row_sums[i] > 0.0 else uniform

    return transition

def decode_map_phasings(nodes: np.ndarray, node_potentials: list,
                                edges: np.ndarray, transitions: list) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    vertices = np.asarray(nodes, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("nodes must be a two-dimensional array with two columns")
    if vertices.size and not (np.all(np.isfinite(vertices))
                              and np.allclose(vertices, np.round(vertices),
                                              rtol=0.0, atol=1e-12)):
        raise ValueError("nodes entries must be finite and integer valued")
    n_vertices = int(vertices.shape[0])

    potentials = [np.asarray(p, dtype=float) for p in node_potentials]
    if len(potentials) != n_vertices:
        raise ValueError("node_potentials must hold one array per vertex")
    for p in potentials:
        if p.ndim != 1 or p.size < 1:
            raise ValueError("each node potential must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
            raise ValueError("node potential entries must be finite and strictly positive")

    links = np.asarray(edges, dtype=float)
    if links.size == 0:
        links = links.reshape(0, 2)
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError("edges must be a two-dimensional array with two columns")
    if links.size and not (np.all(np.isfinite(links))
                           and np.allclose(links, np.round(links), rtol=0.0, atol=1e-12)):
        raise ValueError("edges entries must be finite and integer valued")
    links = np.round(links).astype(int)
    if len(transitions) != links.shape[0]:
        raise ValueError("transitions must hold one matrix per edge")

    matrices = []
    for e in range(links.shape[0]):
        parent, child = int(links[e, 0]), int(links[e, 1])
        if not (0 <= parent < n_vertices and 0 <= child < n_vertices):
            raise ValueError("edges entries must be valid vertex indices")
        if parent >= child:
            raise ValueError("each edge must run from a smaller to a larger vertex index")
        matrix = np.asarray(transitions[e], dtype=float)
        if matrix.shape != (potentials[parent].size, potentials[child].size):
            raise ValueError("each transition matrix must match the states of its edge")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("transition entries must be finite and non-negative")
        matrices.append(matrix)

    # Connected components of the undirected graph the edges induce.
    neighbours = [[] for _ in range(n_vertices)]
    for e in range(links.shape[0]):
        neighbours[int(links[e, 0])].append(int(links[e, 1]))
        neighbours[int(links[e, 1])].append(int(links[e, 0]))
    component_of = -np.ones(n_vertices, dtype=int)
    components = []
    for seed in range(n_vertices):
        if component_of[seed] >= 0:
            continue
        stack, members = [seed], []
        component_of[seed] = len(components)
        while stack:
            v = stack.pop()
            members.append(v)
            for w in neighbours[v]:
                if component_of[w] < 0:
                    component_of[w] = len(components)
                    stack.append(w)
        components.append(sorted(members))

    states = np.zeros(n_vertices, dtype=int)
    for members in components:
        sizes = [int(potentials[t].size) for t in members]
        volume = 1
        for s in sizes:
            volume *= s
        if volume > 2000000:
            raise ValueError("a component admits more than 2000000 assignments")
        inside = [e for e in range(links.shape[0])
                  if component_of[int(links[e, 0])] == component_of[members[0]]]
        slot = {t: i for i, t in enumerate(members)}
        best_score, best_assignment = -1.0, None
        # itertools.product enumerates in ascending lexicographic order, so a
        # strict improvement test keeps the smallest maximiser.
        for assignment in itertools.product(*[range(s) for s in sizes]):
            score = 1.0
            for t, index in zip(members, assignment):
                score *= float(potentials[t][index])
            for e in inside:
                score *= float(matrices[e][assignment[slot[int(links[e, 0])]],
                                           assignment[slot[int(links[e, 1])]]])
            if score > best_score:
                best_score, best_assignment = score, assignment
        for t, index in zip(members, best_assignment):
            states[t] = int(index)

    return states

def select_next_position(nodes: np.ndarray, node_phased: np.ndarray,
                                 position_phased: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    vertices = np.asarray(nodes, dtype=float)
    if vertices.size == 0:
        vertices = vertices.reshape(0, 2)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("nodes must be a two-dimensional array with two columns")
    if vertices.size and not (np.all(np.isfinite(vertices))
                              and np.allclose(vertices, np.round(vertices),
                                              rtol=0.0, atol=1e-12)):
        raise ValueError("nodes entries must be finite and integer valued")
    vertices = np.round(vertices).astype(int)
    n_vertices = int(vertices.shape[0])

    processed = np.asarray(node_phased, dtype=float)
    assigned = np.asarray(position_phased, dtype=float)
    if processed.ndim != 1 or processed.size != n_vertices:
        raise ValueError("node_phased must be one-dimensional of length U")
    if assigned.ndim != 1 or assigned.size < 1:
        raise ValueError("position_phased must be a non-empty one-dimensional array")
    if not np.all(np.isin(processed, (0.0, 1.0))) or not np.all(np.isin(assigned, (0.0, 1.0))):
        raise ValueError("node_phased and position_phased entries must be 0 or 1")
    processed = processed.astype(bool)
    assigned = assigned.astype(bool)
    if n_vertices and (np.any(vertices < 0) or np.any(vertices >= assigned.size)):
        raise ValueError("nodes entries must be valid SNP indices")

    move = np.zeros(3 + n_vertices, dtype=int)
    move[1] = -1
    move[2] = -1

    # A vertex can extend the assembly only while one of its two SNPs is unphased.
    extendable = np.array([not (assigned[vertices[t, 0]] and assigned[vertices[t, 1]])
                           for t in range(n_vertices)], dtype=bool)

    frontier = np.zeros(n_vertices, dtype=bool)
    for t in range(n_vertices):
        if processed[t] or not extendable[t]:
            continue
        pair = set(vertices[t].tolist())
        for other in range(n_vertices):
            if other != t and processed[other] and len(pair & set(vertices[other].tolist())) == 1:
                frontier[t] = True
                break

    if np.any(frontier):
        connectivity = {}
        for t in np.flatnonzero(frontier):
            for snp in vertices[t].tolist():
                if not assigned[snp]:
                    connectivity[int(snp)] = connectivity.get(int(snp), 0) + 1
        chosen = max(sorted(connectivity), key=lambda snp: connectivity[snp])
        move[0] = 0
        move[1] = int(chosen)
        move[3:] = frontier.astype(int)
        return move

    remaining = [t for t in range(n_vertices) if (not processed[t]) and extendable[t]]
    if remaining:
        move[0] = 1
        move[2] = int(remaining[0])
        return move

    move[0] = 2
    return move

def generate_position_candidates(positions: np.ndarray, haplotypes: np.ndarray,
                                         genotypes: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    matrix = np.asarray(haplotypes, dtype=float)
    if matrix.ndim != 2 or matrix.size < 1:
        raise ValueError("haplotypes must be a non-empty two-dimensional array")
    if not np.all(np.isin(matrix, (-1.0, 0.0, 1.0))):
        raise ValueError("haplotypes entries must be -1, 0 or 1")
    matrix = matrix.astype(int)
    ploidy, n_snps = matrix.shape

    counts = np.asarray(genotypes, dtype=float)
    if counts.ndim != 1 or counts.size != n_snps:
        raise ValueError("genotypes must be one-dimensional of length n_snps")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    counts = np.round(counts).astype(int)
    if np.any(counts < 0) or np.any(counts > ploidy):
        raise ValueError("genotypes entries must lie between zero and the ploidy inclusive")

    window = np.asarray(positions, dtype=float)
    if window.ndim != 1 or window.size < 1:
        raise ValueError("positions must be a non-empty one-dimensional array")
    if not np.allclose(window, np.round(window), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    window = np.round(window).astype(int)
    if np.any(np.diff(window) <= 0):
        raise ValueError("positions entries must be strictly increasing")
    if np.any(window < 0) or np.any(window >= n_snps):
        raise ValueError("positions entries must be valid SNP indices")

    # Each window column is either frozen by the assembly so far or free, in
    # which case it is any placement of its alternate-allele count on the rows.
    column_options = []
    for snp in window.tolist():
        column = matrix[:, snp]
        if np.all(column >= 0):
            if int(column.sum()) != int(counts[snp]):
                raise ValueError("an assigned column contradicts its called genotype")
            column_options.append([column.copy()])
        else:
            column_options.append([np.array(pattern, dtype=int)
                                   for pattern in itertools.product((0, 1), repeat=ploidy)
                                   if sum(pattern) == int(counts[snp])])

    candidates = [np.stack(choice, axis=1)
                  for choice in itertools.product(*column_options)]

    return np.array(candidates, dtype=int).reshape(-1, ploidy, int(window.size))

def score_phasing_candidates(candidates: np.ndarray, positions: np.ndarray,
                                     reads: np.ndarray, neighbor_pairs: np.ndarray,
                                     neighbor_phasings: np.ndarray, error_rate: float,
                                     weights: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    matrices = np.asarray(candidates, dtype=float)
    if matrices.ndim != 3 or matrices.size < 1:
        raise ValueError("candidates must be a non-empty three-dimensional array")
    if not np.all(np.isin(matrices, (0.0, 1.0))):
        raise ValueError("candidates entries must be 0 or 1")
    matrices = matrices.astype(int)
    ploidy = int(matrices.shape[1])

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    window = np.asarray(positions, dtype=float)
    if window.ndim != 1 or window.size != matrices.shape[2]:
        raise ValueError("positions must be one-dimensional and match the candidate width")
    if not np.allclose(window, np.round(window), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    window = np.round(window).astype(int)
    if np.unique(window).size != window.size:
        raise ValueError("positions entries must be distinct")
    if np.any(window < 0) or np.any(window >= fragments.shape[1]):
        raise ValueError("positions entries must be valid SNP indices")

    pairs = np.asarray(neighbor_pairs, dtype=float)
    if pairs.size == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("neighbor_pairs must be a two-dimensional array with two columns")
    decoded = np.asarray(neighbor_phasings, dtype=float)
    if decoded.size == 0:
        decoded = decoded.reshape(0, ploidy, 2)
    if decoded.ndim != 3 or decoded.shape[1] != ploidy or decoded.shape[2] != 2:
        raise ValueError("neighbor_phasings must have shape (m, K, 2) for the candidate ploidy")
    if pairs.shape[0] != decoded.shape[0]:
        raise ValueError("neighbor_pairs and neighbor_phasings must have the same length")
    if pairs.size and not (np.all(np.isfinite(pairs))
                           and np.allclose(pairs, np.round(pairs), rtol=0.0, atol=1e-12)):
        raise ValueError("neighbor_pairs entries must be finite and integer valued")
    if decoded.size and not np.all(np.isin(decoded, (0.0, 1.0))):
        raise ValueError("neighbor_phasings entries must be 0 or 1")
    pairs = np.round(pairs).astype(int)
    decoded = decoded.astype(int)
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= fragments.shape[1])):
        raise ValueError("neighbor_pairs entries must be valid SNP indices")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("each neighbour vertex must name two distinct SNPs")

    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    epsilon = float(error_rate)

    coefficients = np.asarray(weights, dtype=float).ravel()
    if coefficients.size != 3 or not np.all(np.isfinite(coefficients)) \
            or np.any(coefficients < 0.0):
        raise ValueError("weights must hold three finite non-negative entries")

    def _canonical(matrix):
        return matrix[np.lexsort(matrix[:, ::-1].T)]

    observed = fragments[:, window]
    observed = observed[np.any(observed >= 0, axis=1)]
    order = window.tolist()

    n_candidates = int(matrices.shape[0])
    likelihood = np.zeros(n_candidates, dtype=float)
    correction = np.zeros(n_candidates, dtype=float)
    agreement = np.zeros(n_candidates, dtype=float)
    for i, candidate in enumerate(matrices):
        for fragment in observed:
            called = fragment >= 0
            distance = (candidate[:, called] != fragment[None, called]).sum(axis=1)
            covered = int(called.sum())
            likelihood[i] += float(np.log(np.sum(
                epsilon ** distance * (1.0 - epsilon) ** (covered - distance))))
            correction[i] += float(distance.min())
        matched = 0
        for j in range(pairs.shape[0]):
            first, second = int(pairs[j, 0]), int(pairs[j, 1])
            if first in order and second in order:
                projection = candidate[:, [order.index(first), order.index(second)]]
                if np.array_equal(_canonical(projection), _canonical(decoded[j])):
                    matched += 1
        agreement[i] = float(matched)

    def _rescale(values):
        low, high = float(values.min()), float(values.max())
        if high - low <= 0.0:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    return (coefficients[0] * _rescale(likelihood)
            - coefficients[1] * _rescale(correction)
            + coefficients[2] * _rescale(agreement))

def compute_block_adjusted_mec(reads: np.ndarray, haplotypes: np.ndarray,
                                       blocks: np.ndarray, ploidy: int) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    assembly = np.asarray(haplotypes, dtype=float)
    for name, array in (("reads", fragments), ("haplotypes", assembly)):
        if array.ndim != 2 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional array")
        if not np.all(np.isin(array, (-1.0, 0.0, 1.0))):
            raise ValueError(f"{name} entries must be -1, 0 or 1")
    fragments = fragments.astype(int)
    assembly = assembly.astype(int)
    if fragments.shape[1] != assembly.shape[1]:
        raise ValueError("reads and haplotypes must agree on the number of SNPs")

    labels = np.asarray(blocks, dtype=float)
    if labels.ndim != 1 or labels.size != fragments.shape[1]:
        raise ValueError("blocks must be one-dimensional of length n_snps")
    if not np.allclose(labels, np.round(labels), rtol=0.0, atol=1e-12):
        raise ValueError("blocks entries must be integer valued")
    labels = np.round(labels).astype(int)
    if np.any(labels < 0):
        raise ValueError("blocks entries must be non-negative")
    if np.any(np.all(assembly >= 0, axis=0) != (labels > 0)):
        raise ValueError("a SNP must carry a positive block label exactly when it is resolved")

    if not (isinstance(ploidy, (int, np.integer)) and not isinstance(ploidy, bool)
            and int(ploidy) >= 1):
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)
    if ploidy != assembly.shape[0]:
        raise ValueError("ploidy must match the first axis of haplotypes")

    penalty = 1.0 - 1.0 / float(ploidy)
    resolved_snp = np.all(assembly >= 0, axis=0)
    total = 0.0
    called = 0
    for fragment in fragments:
        covered = np.flatnonzero(fragment >= 0)
        if covered.size == 0:
            continue
        called += int(covered.size)

        # An allele at an unresolved SNP costs one against every haplotype, so
        # it is charged outright and takes no part in the matching.
        unresolved = covered[~resolved_snp[covered]]
        total += float(unresolved.size)

        # Blocks carry independent haplotype labels, so the best-fitting
        # haplotype is chosen inside each block on its own.
        resolved = covered[resolved_snp[covered]]
        spanned = sorted({int(b) for b in labels[resolved]})
        for block in spanned:
            group = resolved[labels[resolved] == block]
            window = assembly[:, group]
            mismatch = (window != fragment[group][None, :]).astype(int)
            total += float(mismatch.sum(axis=1).min())
        total += (max(len(spanned), 1) - 1) * penalty

    if called == 0:
        raise ValueError("the fragment set must call at least one allele")

    return float(total / called)

def run_phapcompass_short_pipeline(fragments: tuple = (
        "0-10-----------", "00111----------", "011------------",
        "0110-----------", "10100----------", "1100-----------",
        "1110-----------", "11110----------", "-010-----------",
        "-111-----------", "--110----------", "-----00000-----",
        "-----0011------", "-----0101------", "-----100-------",
        "-----100-------", "-----1010------", "-----11101-----",
        "-----11111-----", "------0101-----", "------100------",
        "------1001-----", "----------00-0-", "----------000--",
        "----------0000-", "----------0000-", "----------1100-",
        "-----------000-", "-----------100-", "-----------111-",
        "--------------1", "--------------0", "--------------1"),
        genotypes: tuple = (3, 3, 2, 1, 1, 3, 3, 2, 2, 3, 2, 3, 1, 1, 2),
        ploidy: int = 4,
        error_rate: float = 0.02,
        likelihood_weight: float = 1.0 / 12.0,
        mec_weight: float = 10.0 / 12.0,
        agreement_weight: float = 1.0 / 12.0) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(ploidy, bool) or not isinstance(ploidy, (int, np.integer)) or int(ploidy) < 1:
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)
    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    weights = np.array([likelihood_weight, mec_weight, agreement_weight], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("the three weights must be finite and non-negative")

    rows = list(fragments)
    if len(rows) < 1 or not all(isinstance(row, str) for row in rows):
        raise ValueError("fragments must be a non-empty sequence of strings")
    n_snps = len(rows[0])
    if n_snps < 1 or any(len(row) != n_snps for row in rows):
        raise ValueError("every fragment must be a non-empty string of the same length")
    if any(character not in "01-" for row in rows for character in row):
        raise ValueError("fragments may only contain the characters 0, 1 and -")
    reads = np.array([[-1 if c == "-" else int(c) for c in row] for row in rows], dtype=int)

    calls = np.asarray(genotypes, dtype=float)
    if calls.ndim != 1 or calls.size != n_snps:
        raise ValueError("genotypes must hold one entry per SNP")
    if not np.allclose(calls, np.round(calls), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    calls = np.round(calls).astype(int)
    if np.any(calls < 0) or np.any(calls > ploidy):
        raise ValueError("genotypes entries must lie between zero and ploidy inclusive")

    # -- Sub-problem 03: the vertices of the SNP line graph, in genomic order.
    nodes = build_snp_line_graph(reads)
    n_vertices = int(nodes.shape[0])

    # -- Sub-problems 01 and 02: the phasing state space and read-evidence
    #    potential of every vertex.
    node_phasings = [enumerate_valid_phasings(ploidy, calls[nodes[t]])
                     for t in range(n_vertices)]
    node_potentials = [compute_phasing_potentials(reads, node_phasings[t],
                                                          nodes[t], error_rate)
                       for t in range(n_vertices)]

    # -- Sub-problems 01, 02 and 04: every pair of vertices sharing exactly one
    #    SNP spans a three-SNP run whose potential becomes a transition.
    edge_list = []
    transitions = []
    for parent in range(n_vertices):
        for child in range(parent + 1, n_vertices):
            shared = set(nodes[parent].tolist()) & set(nodes[child].tolist())
            if len(shared) != 1:
                continue
            joint = np.array(sorted(set(nodes[parent].tolist()) | set(nodes[child].tolist())),
                             dtype=int)
            joint_phasings = enumerate_valid_phasings(ploidy, calls[joint])
            joint_potentials = compute_phasing_potentials(reads, joint_phasings,
                                                                  joint, error_rate)
            transitions.append(build_transition_matrix(
                nodes[parent], node_phasings[parent], nodes[child], node_phasings[child],
                joint, joint_phasings, joint_potentials))
            edge_list.append((parent, child))
    edges = np.array(edge_list, dtype=int).reshape(-1, 2)

    # -- Sub-problem 05: one decoded phasing per vertex.
    states = decode_map_phasings(nodes, node_potentials, edges, transitions)

    # -- Sub-problems 06, 07 and 08: stitch the local phasings into one global
    #    assembly, one variant at a time.
    haplotypes = -np.ones((ploidy, n_snps), dtype=int)
    blocks = np.zeros(n_snps, dtype=int)
    node_phased = np.zeros(n_vertices, dtype=int)
    position_phased = np.zeros(n_snps, dtype=int)
    block_id = 0
    while True:
        move = select_next_position(nodes, node_phased, position_phased)
        action = int(move[0])
        if action == 2:
            break
        if action == 1:
            vertex = int(move[2])
            block_id += 1
            phasing = node_phasings[vertex][int(states[vertex])]
            for column, snp in enumerate(nodes[vertex].tolist()):
                if not position_phased[snp]:
                    haplotypes[:, snp] = phasing[:, column]
                    position_phased[snp] = 1
                    blocks[snp] = block_id
            node_phased[vertex] = 1
            continue
        target = int(move[1])
        frontier = np.flatnonzero(move[3:] > 0)
        touching = [t for t in frontier.tolist() if target in nodes[t].tolist()]
        window = np.array(sorted({int(snp) for t in touching
                                  for snp in nodes[t].tolist()}), dtype=int)
        candidates = generate_position_candidates(window, haplotypes, calls)
        neighbour_phasings = np.array([node_phasings[t][int(states[t])] for t in touching],
                                      dtype=int)
        scores = score_phasing_candidates(candidates, window, reads,
                                                  nodes[touching], neighbour_phasings,
                                                  error_rate, weights)
        best = candidates[int(np.argmax(scores))]
        haplotypes[:, target] = best[:, int(np.flatnonzero(window == target)[0])]
        position_phased[target] = 1
        blocks[target] = block_id
        for t in touching:
            node_phased[t] = 1

    # -- Sub-problem 09: score the finished assembly against the fragments.
    return float(compute_block_adjusted_mec(reads, haplotypes, blocks, ploidy))
SCICODE_GOLD_EOF
