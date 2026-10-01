#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def enumerate_noncrossing_pairings(order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(order) or float(order) != float(int(order)):
        raise ValueError("order must be an integer value")
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least one")

    def _pairings_on(points):
        # A non-crossing pairing of an ordered list of points is built by pairing
        # the first point with one of the points at odd offset from it; the two
        # arcs it cuts out are then paired independently and never cross it.
        if not points:
            return [[]]
        collected = []
        head = points[0]
        for offset in range(1, len(points), 2):
            partner = points[offset]
            inside = points[1:offset]
            outside = points[offset + 1:]
            for left in _pairings_on(inside):
                for right in _pairings_on(outside):
                    collected.append([(head, partner)] + left + right)
        return collected

    n_points = 2 * order
    rows = []
    for pairs in _pairings_on(list(range(n_points))):
        involution = np.zeros(n_points, dtype=np.int64)
        for first, second in pairs:
            involution[first] = second
            involution[second] = first
        rows.append(involution)

    pairings = np.array(rows, dtype=np.int64)
    # Lexicographic order by the involution makes the enumeration canonical.
    keys = tuple(pairings[:, column] for column in range(n_points - 1, -1, -1))
    return pairings[np.lexsort(keys)]

def compute_kreweras_block_table(pairings: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    table = np.asarray(pairings)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("pairings must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(np.asarray(table, dtype=float))):
        raise ValueError("pairings must contain only finite entries")
    if not np.allclose(np.asarray(table, dtype=float),
                       np.round(np.asarray(table, dtype=float)), rtol=0.0, atol=1e-12):
        raise ValueError("pairings entries must be integer valued")
    table = np.asarray(np.round(np.asarray(table, dtype=float)), dtype=np.int64)

    n_points = table.shape[1]
    if n_points % 2 != 0:
        raise ValueError("pairings must have an even number of columns")
    if np.any(table < 0) or np.any(table >= n_points):
        raise ValueError("pairings entries must be valid point labels")

    block_table = np.zeros_like(table)
    for row in range(table.shape[0]):
        involution = table[row]
        if np.any(involution[involution] != np.arange(n_points)):
            raise ValueError("each row of pairings must be an involution")
        if np.any(involution == np.arange(n_points)):
            raise ValueError("each row of pairings must be fixed-point free")

        # The Kreweras complement is the permutation obtained by following the
        # cyclic successor and then the pairing; its cycles are the blocks.
        successor = (np.arange(n_points) + 1) % n_points
        complement = involution[successor]

        labels = np.full(n_points, -1, dtype=np.int64)
        next_label = 0
        for start in range(n_points):
            if labels[start] >= 0:
                continue
            point = start
            while labels[point] < 0:
                labels[point] = next_label
                point = int(complement[point])
            next_label += 1
        block_table[row] = labels

    return block_table

def build_nested_design_projectors(family_sizes: np.ndarray,
                                           include_intercept: bool = True) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    sizes = np.asarray(family_sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size < 1:
        raise ValueError("family_sizes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sizes)):
        raise ValueError("family_sizes must contain only finite entries")
    if not np.allclose(sizes, np.round(sizes), rtol=0.0, atol=1e-12):
        raise ValueError("family_sizes entries must be integer valued")
    sizes = np.asarray(np.round(sizes), dtype=np.int64)
    if np.any(sizes < 1):
        raise ValueError("every family must contain at least one individual")

    n_families = int(sizes.size)
    n_individuals = int(sizes.sum())

    # Membership matrix of the family level; the individual level is the identity.
    membership = np.zeros((n_individuals, n_families), dtype=float)
    start = 0
    for family, size in enumerate(sizes):
        membership[start:start + int(size), family] = 1.0
        start += int(size)

    # Orthogonal projection onto the family column space. Its Gram matrix is
    # diagonal, so the projection is a block of reciprocal family sizes.
    family_projection = membership @ (membership.T / sizes[:, None].astype(float))

    if bool(include_intercept):
        # The population mean spans the constant vector, which lies inside the
        # family column space, so its projection is subtracted from the coarser
        # increment only.
        mean_projection = np.full((n_individuals, n_individuals),
                                  1.0 / float(n_individuals), dtype=float)
    else:
        mean_projection = np.zeros((n_individuals, n_individuals), dtype=float)

    between = family_projection - mean_projection
    within = np.eye(n_individuals, dtype=float) - family_projection

    projectors = np.stack([between, within])
    # Symmetrise to remove the asymmetry that floating point accumulation leaves.
    return 0.5 * (projectors + np.transpose(projectors, (0, 2, 1)))

def evaluate_design_functionals(projector: np.ndarray,
                                        family_sizes: np.ndarray,
                                        n_traits: int,
                                        order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    matrix = np.asarray(projector, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("projector must be a square two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("projector must contain only finite entries")

    sizes = np.asarray(family_sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size < 1 or not np.all(np.isfinite(sizes)):
        raise ValueError("family_sizes must be a non-empty one-dimensional finite array")
    if not np.allclose(sizes, np.round(sizes), rtol=0.0, atol=1e-12):
        raise ValueError("family_sizes entries must be integer valued")
    sizes = np.asarray(np.round(sizes), dtype=np.int64)
    if np.any(sizes < 1):
        raise ValueError("every family must contain at least one individual")
    if int(sizes.sum()) != matrix.shape[0]:
        raise ValueError("family_sizes must sum to the dimension of projector")

    for name, value in (("n_traits", n_traits), ("order", order)):
        if not _is_number(value) or float(value) != float(int(value)) or int(value) < 1:
            raise ValueError(f"{name} must be an integer value of at least one")
    n_traits = int(n_traits)
    order = int(order)

    n_individuals = matrix.shape[0]
    membership = np.zeros((n_individuals, int(sizes.size)), dtype=float)
    start = 0
    for family, size in enumerate(sizes):
        membership[start:start + int(size), family] = 1.0
        start += int(size)

    # Design operators: the family level carries the family membership outer
    # product, the individual level the identity, so it reduces to the projector.
    operators = [matrix @ (membership @ membership.T), matrix.copy()]

    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            # Words of the given length are enumerated by reading the offset in
            # binary, which is the index order the functional table uses.
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            if length == 1:
                value = float(np.trace(operators[word[0]]))
            else:
                accumulated = operators[word[0]]
                for letter in word[1:-1]:
                    accumulated = accumulated @ operators[letter]
                value = float(np.einsum("ij,ji->", accumulated, operators[word[-1]]))
            functionals[base + offset] = value / float(n_traits)

    return functionals

def build_variance_components(n_traits: int,
                                      tau_genetic: float,
                                      rho_genetic: float,
                                      tau_residual: float,
                                      rho_residual: float,
                                      rotation_angle: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(n_traits) or float(n_traits) != float(int(n_traits)) or int(n_traits) < 2:
        raise ValueError("n_traits must be an integer value of at least two")
    for name, value in (("tau_genetic", tau_genetic), ("tau_residual", tau_residual)):
        if not _is_number(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number greater than zero")
    for name, value in (("rho_genetic", rho_genetic), ("rho_residual", rho_residual)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not _is_number(rotation_angle):
        raise ValueError("rotation_angle must be a finite number")

    n_traits = int(n_traits)
    angle = float(rotation_angle)

    # Step eigenvalue profiles: one positive level on the leading directions.
    genetic_spectrum = np.zeros(n_traits, dtype=float)
    genetic_spectrum[:int(np.ceil(n_traits * float(rho_genetic)))] = float(tau_genetic)
    residual_spectrum = np.zeros(n_traits, dtype=float)
    residual_spectrum[:int(np.ceil(n_traits * float(rho_residual)))] = float(tau_residual)

    # Eigenbasis of the family-level component: a plane rotation inside every
    # pair of mirrored coordinates, leaving a central coordinate fixed when the
    # number of traits is odd.
    basis = np.eye(n_traits, dtype=float)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    for low in range(n_traits // 2):
        high = n_traits - 1 - low
        basis[low, low] = cosine
        basis[low, high] = -sine
        basis[high, low] = sine
        basis[high, high] = cosine

    genetic = basis @ (genetic_spectrum[:, None] * basis.T)
    residual = np.diag(residual_spectrum)

    components = np.stack([genetic, residual])
    # Symmetrise to remove the asymmetry that floating point accumulation leaves.
    return 0.5 * (components + np.transpose(components, (0, 2, 1)))

def evaluate_spectral_functionals(components: np.ndarray, order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    block = np.asarray(components, dtype=float)
    if block.ndim != 3 or block.shape[0] != 2:
        raise ValueError("components must have shape (2, p, p)")
    if block.shape[1] != block.shape[2] or block.shape[1] < 1:
        raise ValueError("components must hold square matrices of positive size")
    if not np.all(np.isfinite(block)):
        raise ValueError("components must contain only finite entries")

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(order) or float(order) != float(int(order)) or int(order) < 1:
        raise ValueError("order must be an integer value of at least one")
    order = int(order)

    n_traits = block.shape[1]
    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            # Words of the given length are enumerated by reading the offset in
            # binary, which is the index order the functional table uses.
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            if length == 1:
                value = float(np.trace(block[word[0]]))
            else:
                accumulated = block[word[0]]
                for letter in word[1:-1]:
                    accumulated = accumulated @ block[letter]
                value = float(np.einsum("ij,ji->", accumulated, block[word[-1]]))
            functionals[base + offset] = value / float(n_traits)

    return functionals

def evaluate_moment_expansion(design_functionals: np.ndarray,
                                      spectral_functionals: np.ndarray,
                                      block_table: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    table = np.asarray(block_table)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("block_table must be a non-empty two-dimensional array")
    as_float = np.asarray(table, dtype=float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError("block_table must contain only finite entries")
    if not np.allclose(as_float, np.round(as_float), rtol=0.0, atol=1e-12):
        raise ValueError("block_table entries must be integer valued")
    table = np.asarray(np.round(as_float), dtype=np.int64)
    if np.any(table < 0):
        raise ValueError("block_table entries must be non-negative")
    if table.shape[1] % 2 != 0:
        raise ValueError("block_table must have an even number of columns")

    order = table.shape[1] // 2
    required = (1 << (order + 1)) - 2
    design = np.asarray(design_functionals, dtype=float).ravel()
    spectral = np.asarray(spectral_functionals, dtype=float).ravel()
    for name, values in (("design_functionals", design), ("spectral_functionals", spectral)):
        if values.size < required:
            raise ValueError(f"{name} must be tabulated to at least the moment order")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")

    def _index(word):
        offset = 0
        for letter in word:
            offset = 2 * offset + letter
        return (1 << len(word)) - 2 + offset

    moment = 0.0
    for row in range(table.shape[0]):
        labels = table[row]
        blocks = {}
        for position, label in enumerate(labels):
            blocks.setdefault(int(label), []).append(position)
        for code in range(1 << order):
            # Each assignment of a level to the order factors of the moment is
            # one binary word of length equal to the moment order.
            levels = [(code >> (order - 1 - place)) & 1 for place in range(order)]
            term = 1.0
            for positions in blocks.values():
                positions = sorted(positions)
                if positions[0] % 2 == 0:
                    # Even-labelled points carry the design operators; the point
                    # at position v belongs to the v/2-th factor of the moment.
                    word = tuple(levels[v // 2] for v in positions)
                    term *= design[_index(word)]
                else:
                    # Odd-labelled points carry the covariance components, with
                    # the last point wrapping back onto the first factor.
                    word = tuple(levels[((v + 1) // 2) % order] for v in positions)
                    term *= spectral[_index(word)]
            moment += term

    return float(moment)

def build_surrogate_spectral_functionals(genetic_level: float,
                                                 genetic_fraction: float,
                                                 residual_level: float,
                                                 residual_fraction: float,
                                                 order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    for name, value in (("genetic_level", genetic_level),
                        ("residual_level", residual_level)):
        if not _is_number(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be a finite number of at least zero")
    for name, value in (("genetic_fraction", genetic_fraction),
                        ("residual_fraction", residual_fraction)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not _is_number(order) or float(order) != float(int(order)) or int(order) < 1:
        raise ValueError("order must be an integer value of at least one")

    genetic_level = float(genetic_level)
    genetic_fraction = float(genetic_fraction)
    residual_level = float(residual_level)
    residual_fraction = float(residual_fraction)
    order = int(order)

    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            n_genetic = sum(1 for letter in word if letter == 0)
            n_residual = length - n_genetic
            if n_genetic >= 1:
                # The genetic profile vanishes outside its own interval, which
                # lies inside the residual one, so the integral is taken there.
                value = (genetic_fraction * genetic_level ** n_genetic
                         * residual_level ** n_residual)
            else:
                value = residual_fraction * residual_level ** n_residual
            functionals[base + offset] = value

    return functionals

def solve_joint_moment_equations(design_functionals: np.ndarray,
                                         moments: np.ndarray,
                                         residual_fraction: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    design = np.asarray(design_functionals, dtype=float)
    if design.ndim != 1 or design.size < 14:
        raise ValueError("design_functionals must be a one-dimensional array of at least fourteen entries")
    if not np.all(np.isfinite(design)):
        raise ValueError("design_functionals must contain only finite entries")
    observed = np.asarray(moments, dtype=float)
    if observed.ndim != 1 or observed.size != 3 or not np.all(np.isfinite(observed)):
        raise ValueError("moments must be a one-dimensional array of three finite entries")
    if not _is_number(residual_fraction) or not (0.0 < float(residual_fraction) <= 1.0):
        raise ValueError("residual_fraction must be a finite number in the interval (0, 1]")
    rho_residual = float(residual_fraction)

    family, individual = float(design[0]), float(design[1])
    family_family, family_individual = float(design[2]), float(design[3])
    individual_family, individual_individual = float(design[4]), float(design[5])
    if family == 0.0:
        raise ValueError("the family-level design functional must not vanish")

    block_table = compute_kreweras_block_table(
        enumerate_noncrossing_pairings(3))

    def _profile_moments(residual_level):
        # The first moment is linear in the genetic profile mean, and the second
        # is linear in the genetic profile's second moment once that mean and the
        # residual level are fixed, so both follow without iteration.
        mean = (observed[0] - individual * rho_residual * residual_level) / family
        residual_mean = rho_residual * residual_level
        known = (family_family * mean ** 2
                 + (family_individual + individual_family) * mean * residual_mean
                 + 2.0 * family * individual * mean * residual_level
                 + individual ** 2 * residual_mean * residual_level
                 + individual_individual * residual_mean ** 2)
        second = (observed[1] - known) / family ** 2
        return mean, second

    def _third_moment_residual(residual_level):
        mean, second = _profile_moments(residual_level)
        if not (mean > 0.0 and second > 0.0):
            return float("nan")
        surrogate = build_surrogate_spectral_functionals(
            second / mean, mean ** 2 / second, residual_level, rho_residual, 3)
        return evaluate_moment_expansion(design, surrogate, block_table) - observed[2]

    # The genetic profile mean is positive only below this residual level, which
    # bounds the admissible interval from above.
    upper = observed[0] / (individual * rho_residual) if individual * rho_residual > 0.0 else np.inf
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("the moment equations admit no admissible solution")
    low, high = upper * 1e-9, upper * (1.0 - 1e-9)
    f_low, f_high = _third_moment_residual(low), _third_moment_residual(high)

    if not (np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0.0):
        # Fall back to a deterministic scan for the sign change; the root is
        # unique, so any bracket containing it gives the same answer.
        grid = np.linspace(low, high, 512)
        values = [_third_moment_residual(t) for t in grid]
        low = high = None
        for i in range(len(grid) - 1):
            a, b = values[i], values[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a * b < 0.0:
                low, high, f_low = grid[i], grid[i + 1], a
                break
        if low is None:
            raise ValueError("the moment equations admit no admissible solution")

    for _ in range(200):
        middle = 0.5 * (low + high)
        f_middle = _third_moment_residual(middle)
        if not np.isfinite(f_middle):
            raise ValueError("the moment equations admit no admissible solution")
        if f_middle == 0.0 or (high - low) <= 1e-15 * max(1.0, abs(middle)):
            break
        if f_low * f_middle < 0.0:
            high = middle
        else:
            low, f_low = middle, f_middle

    residual_level = 0.5 * (low + high)
    mean, second = _profile_moments(residual_level)
    if not (mean > 0.0 and second > 0.0):
        raise ValueError("the moment equations admit no admissible solution")
    genetic_level = second / mean
    genetic_fraction = mean ** 2 / second
    if not (0.0 < genetic_fraction < rho_residual and genetic_level > 0.0
            and residual_level > 0.0):
        raise ValueError("the moment equations admit no admissible solution")

    return np.array([genetic_level, genetic_fraction, residual_level], dtype=float)

def run_genetic_spectrum_bias_pipeline(rotation_angle: float = 0.6283185307179586,
                                               tau_genetic: float = 1.4,
                                               rho_genetic: float = 0.35,
                                               tau_residual: float = 0.6,
                                               rho_residual: float = 0.8,
                                               n_traits: int = 240,
                                               n_families: int = 300,
                                               size_cycle: tuple = (1, 2, 3),
                                               quantity: str = "rho_genetic") -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    # -- Validate the orchestrator inputs.
    if not _is_number(rotation_angle):
        raise ValueError("rotation_angle must be a finite number")
    for name, value in (("tau_genetic", tau_genetic), ("tau_residual", tau_residual)):
        if not _is_number(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number greater than zero")
    for name, value in (("rho_genetic", rho_genetic), ("rho_residual", rho_residual)):
        if not _is_number(value) or not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not float(rho_genetic) < float(rho_residual):
        raise ValueError("rho_genetic must be strictly below rho_residual")
    if not _is_number(n_traits) or float(n_traits) != float(int(n_traits)) or int(n_traits) < 2:
        raise ValueError("n_traits must be an integer value of at least two")
    if not _is_number(n_families) or float(n_families) != float(int(n_families)) or int(n_families) < 1:
        raise ValueError("n_families must be an integer value of at least one")
    cycle = list(size_cycle) if not isinstance(size_cycle, (str, bytes)) else []
    if len(cycle) < 1 or not all(_is_number(v) and float(v) == float(int(v)) and int(v) >= 1
                                 for v in cycle):
        raise ValueError("size_cycle must be a non-empty sequence of integers of at least one")
    if quantity not in ("rho_genetic", "tau_genetic", "tau_residual",
                        "moment_1", "moment_2", "moment_3"):
        raise ValueError("quantity must be one of 'rho_genetic', 'tau_genetic', "
                         "'tau_residual', 'moment_1', 'moment_2' or 'moment_3'")

    n_traits = int(n_traits)
    n_families = int(n_families)
    cycle = [int(v) for v in cycle]
    family_sizes = np.array([cycle[i % len(cycle)] for i in range(n_families)], dtype=np.int64)

    # -- Sub-problem 03: the two level increments of the nested design, with the
    #    unknown population mean removed from the coarser one. The joint fit uses
    #    the between-family increment alone.
    projectors = build_nested_design_projectors(family_sizes, True)

    # -- Sub-problem 04: the design functionals, tabulated to word length three
    #    because three moments are matched.
    design_between = evaluate_design_functionals(projectors[0], family_sizes,
                                                         n_traits, 3)

    # -- Sub-problems 05 and 06: the true covariance components and the traces of
    #    their ordered products, which is where the eigenbasis mismatch enters.
    components = build_variance_components(n_traits, tau_genetic, rho_genetic,
                                                   tau_residual, rho_residual, rotation_angle)
    spectral = evaluate_spectral_functionals(components, 3)

    # -- Sub-problems 01 and 02: the combinatorial skeleton of each moment.
    block_tables = [compute_kreweras_block_table(
        enumerate_noncrossing_pairings(order)) for order in (1, 2, 3)]

    # -- Sub-problem 07: the limiting trace moments the design actually produces.
    moments = np.array([evaluate_moment_expansion(design_between, spectral, table)
                        for table in block_tables], dtype=float)

    # -- Sub-problems 08 and 09: the joint fit, whose inner loop rebuilds the
    #    working model's functionals for each candidate parameter triple.
    estimates = solve_joint_moment_equations(design_between, moments, rho_residual)

    reported = {"tau_genetic": float(estimates[0]),
                "rho_genetic": float(estimates[1]),
                "tau_residual": float(estimates[2]),
                "moment_1": float(moments[0]),
                "moment_2": float(moments[1]),
                "moment_3": float(moments[2])}
    return float(reported[quantity])
SCICODE_GOLD_EOF
