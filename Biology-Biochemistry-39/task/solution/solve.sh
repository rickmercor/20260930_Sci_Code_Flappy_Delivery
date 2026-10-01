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
def build_loop_tree(dot_bracket: str) -> "np.ndarray":
    """Reference implementation (one bracket-matching pass, one loop-ownership pass)."""
    import numpy as np

    if not isinstance(dot_bracket, str) or not dot_bracket:
        raise ValueError("dot_bracket must be a non-empty string")
    if set(dot_bracket) - set("()."):
        raise ValueError("dot_bracket may contain only '(', ')' and '.'")
    partner = [-1] * len(dot_bracket)
    stack = []
    for position, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced closing bracket")
            opening = stack.pop()
            partner[opening] = position
            partner[position] = opening
    if stack:
        raise ValueError("unbalanced opening bracket")

    openings = [position for position, symbol in enumerate(dot_bracket) if symbol == "("]
    row_of = {position: row for row, position in enumerate(openings)}
    exterior = len(openings)
    tree = np.zeros((exterior + 1, 4), dtype=np.int64)
    tree[exterior] = (-1, len(dot_bracket), -1, 0)
    # The innermost loop still open at each position owns that position.
    owners = [exterior]
    for position, symbol in enumerate(dot_bracket):
        owner = owners[-1]
        if symbol == ".":
            tree[owner, 3] = 1
        elif symbol == "(":
            row = row_of[position]
            tree[row] = (position, partner[position], owner, 0)
            owners.append(row)
        else:
            owners.pop()
    return tree

import numpy as np
def screen_design_obstructions(tree: "np.ndarray") -> "np.ndarray":
    """Reference implementation (child counts and helix lengths in 5' order)."""
    import numpy as np

    table = np.asarray(tree)
    if table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 1:
        raise ValueError("tree must be a two-dimensional array with four columns")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("tree must hold integers")
    exterior = table.shape[0] - 1
    if table[exterior, 0] != -1 or table[exterior, 2] != -1:
        raise ValueError("the last row must be the exterior loop")
    if not np.all((table[:, 3] == 0) | (table[:, 3] == 1)):
        raise ValueError("unpaired flags must be 0 or 1")
    parents = table[:exterior, 2]
    rows = np.arange(exterior)
    if not np.all(((parents >= 0) & (parents < rows)) | (parents == exterior)):
        raise ValueError("every base pair must name an earlier pair or the exterior loop")

    children = np.bincount(parents, minlength=exterior + 1) if exterior else np.zeros(1, dtype=np.int64)
    unpaired = table[:, 3]
    five_pair = int(np.sum(children[:exterior] >= 4)) + int(children[exterior] >= 5)
    three_pair_unpaired = (
        int(np.sum((children[:exterior] >= 2) & (unpaired[:exterior] == 1)))
        + int(children[exterior] >= 3 and unpaired[exterior] == 1)
    )

    # A pair continues its parent's helix when the parent is a base pair with
    # a single child and no unpaired position; parents precede children.
    stacked = (children == 1) & (unpaired == 0)
    length = np.zeros(exterior, dtype=np.int64)
    shortest = 0
    for row in range(exterior):
        parent = int(parents[row])
        length[row] = length[parent] + 1 if parent != exterior and stacked[parent] else 1
        if not stacked[row]:
            shortest = int(length[row]) if shortest == 0 else min(shortest, int(length[row]))
    return np.array([five_pair, three_pair_unpaired, shortest], dtype=np.int64)

import numpy as np
def enumerate_proper_loop_assignments(closing_code: int, n_children: int) -> "np.ndarray":
    """Reference implementation (filtered lexicographic product)."""
    import itertools

    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(closing_code) and int(closing_code) in (-1, 0, 1, 2, 3)):
        raise ValueError("closing_code must be -1, 0, 1, 2 or 3")
    if not (_is_integer(n_children) and 0 <= int(n_children) <= 8):
        raise ValueError("n_children must be an integer from 0 to 8")
    closing = int(closing_code)
    count = int(n_children)
    colour = (0, 1, 2, 2)
    complement = (1, 0, 2, 2)

    rows = []
    for combo in itertools.product(range(4), repeat=count):
        colours = [colour[code] for code in combo]
        if closing >= 0:
            colours.append(complement[closing])
        if colours.count(0) > 1 or colours.count(1) > 1 or colours.count(2) > 2:
            continue
        weak = [code for code in combo if code >= 2]
        if len(weak) == 2 and weak[0] == weak[1]:
            continue
        # An opposite weak child would let the stacked A and U swap partners.
        if closing >= 2 and any(code != closing for code in weak):
            continue
        rows.append(combo)
    return np.array(rows, dtype=np.int64).reshape(len(rows), count)

import numpy as np
def count_level_constrained_designs(
    tree: "np.ndarray",
    modulus: int,
    leaf_mask: int,
    gray_mask: int,
) -> "np.ndarray":
    """Reference implementation (memoised loop recursion over pair, identity and residue)."""
    from functools import lru_cache

    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(modulus) and 1 <= int(modulus) <= 8):
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    for name, mask in (("leaf_mask", leaf_mask), ("gray_mask", gray_mask)):
        if not (_is_integer(mask) and 0 <= int(mask) < 2 ** size):
            raise ValueError(f"{name} must be an integer in [0, 2**modulus)")
    obstructions = screen_design_obstructions(tree)
    if int(obstructions[0]) or int(obstructions[1]):
        raise ValueError("the tree contains a loop motif that no sequence can design")
    table = np.asarray(tree, dtype=np.int64)
    exterior = table.shape[0] - 1
    if exterior > 30:
        raise ValueError("at most 30 base pairs are supported")
    children = [[] for _ in range(exterior + 1)]
    for row in range(exterior):
        children[int(table[row, 2])].append(row)
    unpaired = [bool(flag) for flag in table[:, 3]]
    leaves, weak = int(leaf_mask), int(gray_mask)
    shift = (1, -1, 0, 0)
    proper = {}

    def _times(first, second):
        product = [0] * (len(first) + len(second) - 1)
        for i, a in enumerate(first):
            if a:
                for j, b in enumerate(second):
                    product[i + j] += a * b
        return product

    def _loop(row, closing, residue):
        key = (closing, len(children[row]))
        if key not in proper:
            proper[key] = enumerate_proper_loop_assignments(*key)
        total = [0]
        for combo in proper[key]:
            product = [1]
            for child, code in zip(children[row], combo):
                product = _times(product, _pair(child, int(code), residue))
                if not any(product):
                    break
            total += [0] * max(0, len(product) - len(total))
            for g, value in enumerate(product):
                total[g] += value
        return total

    @lru_cache(maxsize=None)
    def _pair(row, code, residue):
        if code >= 2 and not (weak >> residue) & 1:
            return (0,)
        inner = (residue + shift[code]) % size
        if unpaired[row] and not (leaves >> inner) & 1:
            return (0,)
        body = _loop(row, code, inner)
        return tuple([0] + body) if code <= 1 else tuple(body)

    if unpaired[exterior] and not leaves & 1:
        counts = [0]
    else:
        counts = _loop(exterior, -1, 0)
    result = np.zeros(exterior + 1, dtype=np.int64)
    for g, value in enumerate(counts[: exterior + 1]):
        result[g] = value
    return result

import numpy as np
def tabulate_assignment_counts(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Reference implementation (one constrained count per residue assignment)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)):
        raise ValueError("modulus must be an integer from 1 to 8")
    if not 1 <= int(modulus) <= 8:
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    everything = 2 ** size - 1
    rows = [
        count_level_constrained_designs(tree, size, assignment, everything ^ assignment)
        for assignment in range(2 ** size)
    ]
    return np.vstack(rows).astype(np.int64)

import numpy as np
def find_minimal_modulus(tree: "np.ndarray", max_modulus: int) -> int:
    """Reference implementation (increasing scan of assignment tables)."""
    import numpy as np

    if isinstance(max_modulus, bool) or not isinstance(max_modulus, (int, np.integer)):
        raise ValueError("max_modulus must be an integer from 1 to 8")
    if not 1 <= int(max_modulus) <= 8:
        raise ValueError("max_modulus must be an integer from 1 to 8")
    for modulus in range(1, int(max_modulus) + 1):
        table = tabulate_assignment_counts(tree, modulus)
        if np.any(table > 0):
            return modulus
    raise ValueError("no modulus up to max_modulus admits a separated design")

import numpy as np
def count_distinct_separated_designs(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Reference implementation (Moebius inversion over the occupied unpaired residues)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)):
        raise ValueError("modulus must be an integer from 1 to 8")
    if not 1 <= int(modulus) <= 8:
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    everything = 2 ** size - 1
    total = None
    # Classify each separated design by the exact set S of residues its
    # unpaired positions occupy; its weak pairs must then avoid S. Designs
    # with unpaired residues inside S are counted by subset constraints, and
    # inclusion-exclusion over the subsets A of S keeps those occupying S
    # exactly.
    for exact in range(2 ** size):
        subset = exact
        while True:
            sign = -1 if bin(exact ^ subset).count("1") % 2 else 1
            counts = count_level_constrained_designs(tree, size, subset, everything ^ exact)
            contribution = [sign * int(value) for value in counts]
            total = contribution if total is None else [a + b for a, b in zip(total, contribution)]
            if subset == 0:
                break
            subset = (subset - 1) & exact
    return np.array(total, dtype=np.int64)

import numpy as np
def evaluate_gc_ensemble(
    distinct_counts: "np.ndarray",
    assignment_counts: "np.ndarray",
    length: int,
    weight: float,
) -> "np.ndarray":
    """Reference implementation (log-domain Boltzmann sums)."""
    import math

    import numpy as np

    distinct = np.asarray(distinct_counts, dtype=float)
    table = np.asarray(assignment_counts, dtype=float)
    if (distinct.ndim != 1 or distinct.size == 0 or not np.all(np.isfinite(distinct))
            or np.any(distinct < 0) or not np.any(distinct > 0)):
        raise ValueError("distinct_counts must be non-negative, finite and not all zero")
    if (table.ndim != 2 or table.shape[1] != distinct.size or not np.all(np.isfinite(table))
            or np.any(table < 0)):
        raise ValueError("assignment_counts must be a non-negative table with k + 1 columns")
    if np.any(table.sum(axis=0) < distinct):
        raise ValueError("every distinct design must be counted in some assignment row")
    minimum_length = max(1, 2 * (distinct.size - 1))
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or int(length) < minimum_length:
        raise ValueError("length must be an integer of at least max(1, 2 k)")
    if (isinstance(weight, bool) or not isinstance(weight, (int, float, np.integer, np.floating))
            or not math.isfinite(float(weight)) or abs(float(weight)) > 50.0):
        raise ValueError("weight must be a finite real number with |weight| <= 50")
    beta = float(weight)
    strong = np.arange(distinct.size, dtype=float)

    def _log_weighted_total(counts):
        # log of sum over entries of count * exp(2 * weight * g), without overflow
        grid = np.broadcast_to(strong, counts.shape)
        positive = counts > 0
        exponents = np.log(counts[positive]) + 2.0 * beta * grid[positive]
        top = float(np.max(exponents))
        return top + math.log(float(np.sum(np.exp(exponents - top))))

    log_designs = _log_weighted_total(distinct)
    positive_numerator = (distinct > 0) & (strong > 0)
    if np.any(positive_numerator):
        numerator_exponents = (
            np.log(distinct[positive_numerator])
            + np.log(2.0 * strong[positive_numerator])
            + 2.0 * beta * strong[positive_numerator]
        )
        numerator_top = float(np.max(numerator_exponents))
        log_nucleotides = numerator_top + math.log(
            float(np.sum(np.exp(numerator_exponents - numerator_top)))
        )
        fraction = math.exp(log_nucleotides - log_designs) / int(length)
    else:
        fraction = 0.0
    proposals = math.exp(_log_weighted_total(table) - log_designs)
    return np.array([fraction, proposals], dtype=float)

import numpy as np
def calibrate_gc_weight(
    distinct_counts: "np.ndarray",
    length: int,
    target_fraction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    """Reference implementation (bracketed bisection on the expected fraction)."""
    import math

    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value)))

    arguments = (("target_fraction", target_fraction), ("lower", lower), ("upper", upper),
                 ("tolerance", tolerance))
    for name, value in arguments:
        if not _is_real(value):
            raise ValueError(f"{name} must be a finite real number")
    target = float(target_fraction)
    low, high = float(lower), float(upper)
    if not 0.0 < target < 1.0:
        raise ValueError("target_fraction must lie strictly between 0 and 1")
    if not -50.0 <= low < high <= 50.0:
        raise ValueError("the bracket must satisfy -50 <= lower < upper <= 50")
    if not 1e-14 <= float(tolerance) <= 1e-6:
        raise ValueError("tolerance must lie in [1e-14, 1e-6]")
    counts = np.asarray(distinct_counts)
    table = counts.reshape(1, -1) if counts.ndim == 1 else counts

    def _residual(value):
        return float(evaluate_gc_ensemble(counts, table, length, value)[0]) - target

    if not (_residual(low) < 0.0 < _residual(high)):
        raise ValueError("the bracket does not enclose the target fraction")
    for _ in range(400):
        if high - low <= float(tolerance):
            break
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _residual(middle) < 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

import numpy as np
def estimate_gc_weight(
    dot_bracket: str = "((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))",
    target_fraction: float = 0.5,
    max_modulus: int = 8,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    if isinstance(max_modulus, bool) or not isinstance(max_modulus, (int, np.integer)):
        raise ValueError("max_modulus must be an integer from 1 to 8")
    if not 1 <= int(max_modulus) <= 8:
        raise ValueError("max_modulus must be an integer from 1 to 8")
    tree = build_loop_tree(dot_bracket)
    five_pair, three_pair_unpaired, shortest_helix = (
        int(value) for value in screen_design_obstructions(tree)
    )
    if five_pair or three_pair_unpaired:
        raise ValueError("the target contains a loop motif that no sequence can design")
    # Helices of three or more pairs guarantee a modulus-2 design.
    limit = min(int(max_modulus), 2) if shortest_helix >= 3 else int(max_modulus)
    modulus = find_minimal_modulus(tree, limit)
    distinct = count_distinct_separated_designs(tree, modulus)
    length = len(dot_bracket)
    weight = calibrate_gc_weight(distinct, length, target_fraction, -50.0, 50.0, tolerance)
    table = tabulate_assignment_counts(tree, modulus)
    proposals = float(evaluate_gc_ensemble(distinct, table, length, weight)[1])
    if not 1.0 - 1e-12 <= proposals <= 2 ** modulus * (1.0 + 1e-12):
        raise ValueError("the assignment table and the distinct counts are inconsistent")
    return float(weight)
SCICODE_GOLD_EOF
