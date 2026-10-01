#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def parse_target(
    structure: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    if not isinstance(structure, str) or not structure:
        raise ValueError("Supply a nonempty dot-bracket string")
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    unpaired: list[int] = []
    for position, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unmatched ')' at position {position}")
            pairs.append((stack.pop(), position))
        elif symbol == ".":
            unpaired.append(position)
        else:
            raise ValueError(f"Invalid symbol {symbol!r} at position {position}")
    if stack:
        raise ValueError(f"Unmatched '(' at positions {stack}")
    return tuple(sorted(pairs)), tuple(unpaired)

def analyse_target(structure: str) -> tuple[int, int, int, int]:
    pairs, unpaired = parse_target(structure)
    pair_set = set(pairs)
    lengths: list[int] = []
    for left, right in pairs:
        if (left - 1, right + 1) in pair_set:
            continue
        length = 0
        while (left, right) in pair_set:
            length += 1
            left, right = left + 1, right - 1
        lengths.append(length)

    nodes: tuple[tuple[int, int] | None, ...] = (None,) + pairs
    children = {node: [] for node in nodes}
    leaves = {node: [] for node in nodes}
    for pair in pairs:
        enclosing = [p for p in pairs if p[0] < pair[0] and pair[1] < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        children[parent].append(pair)
    for position in unpaired:
        enclosing = [p for p in pairs if p[0] < position < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        leaves[parent].append(position)

    m5_count = 0
    m3_bullet_count = 0
    for node in nodes:
        exposed = len(children[node]) + int(node is not None)
        m5_count += int(exposed >= 5)
        m3_bullet_count += int(exposed >= 3 and bool(leaves[node]))
    return (len(lengths), min(lengths) if lengths else 0, m5_count, m3_bullet_count)

def check_local_coloring(parent_color: int, child_colors: tuple[int, ...]) -> int:
    if parent_color not in {-1, 0, 1, 2}:
        raise ValueError("parent_color must be -1, 0, 1, or 2")
    if any(color not in {0, 1, 2} for color in child_colors):
        raise ValueError("Each child colour must be 0, 1, or 2")
    observed = list(child_colors)
    if parent_color != -1:
        observed.append({0: 1, 1: 0, 2: 2}[parent_color])
    return int(observed.count(0) <= 1 and observed.count(1) <= 1 and observed.count(2) <= 2)

from functools import lru_cache
from itertools import product


def find_separated_coloring(structure: str, m: int = 2) -> tuple[int, ...] | None:
    if not isinstance(m, int) or m < 2:
        raise ValueError("m must be an integer of at least 2")
    pairs, unpaired = parse_target(structure)
    _, _, m5_count, m3_count = analyse_target(structure)
    if m5_count or m3_count:
        return None

    nodes = (None,) + pairs
    children = {node: [] for node in nodes}
    leaves = {node: [] for node in nodes}
    for pair in pairs:
        enclosing = [p for p in pairs if p[0] < pair[0] and pair[1] < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        children[parent].append(pair)
    for position in unpaired:
        enclosing = [p for p in pairs if p[0] < position < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        leaves[parent].append(position)

    colors = (0, 1, 2)
    delta = {0: 1, 1: -1, 2: 0}
    leaf_residue = 0
    choices = {}

    @lru_cache(None)
    def _solve(node, color, level):
        if color == 2 and level == leaf_residue:
            return False
        below = (level + delta[color]) % m
        if leaves[node] and below != leaf_residue:
            return False
        for assigned in product(colors, repeat=len(children[node])):
            if not check_local_coloring(color, assigned):
                continue
            if all(_solve(child, child_color, below)
                   for child, child_color in zip(children[node], assigned)):
                choices[(node, color, level)] = assigned
                return True
        return False

    roots = tuple(children[None])
    for root_colors in product(colors, repeat=len(roots)):
        if not check_local_coloring(-1, root_colors):
            continue
        if not all(_solve(node, color, 0) for node, color in zip(roots, root_colors)):
            continue
        coloring = {}
        work = [(node, color, 0) for node, color in reversed(tuple(zip(roots, root_colors)))]
        while work:
            node, color, level = work.pop()
            coloring[node] = color
            below = (level + delta[color]) % m
            assigned = choices[(node, color, level)]
            work.extend((child, child_color, below)
                        for child, child_color in reversed(tuple(zip(children[node], assigned))))
        return tuple(coloring[pair] for pair in pairs)
    return None

def construct_sequence(structure: str, coloring: tuple[int, ...]) -> tuple[int, ...]:
    pairs, _ = parse_target(structure)
    if len(coloring) != len(pairs) or any(color not in {0, 1, 2} for color in coloring):
        raise ValueError("Supply exactly one valid colour code per ordered pair")
    color_by_pair = dict(zip(pairs, coloring))
    children = {pair: [] for pair in pairs}
    children[None] = []
    for pair in pairs:
        enclosing = [p for p in pairs if p[0] < pair[0] and pair[1] < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        children[parent].append(pair)

    grey = [pair for pair in pairs if color_by_pair[pair] == 2]
    graph = {pair: [] for pair in grey}
    for parent, direct_children in children.items():
        siblings = [child for child in direct_children if child in graph]
        for index, first in enumerate(siblings):
            for second in siblings[index + 1:]:
                graph[first].append((second, 1))
                graph[second].append((first, 1))
        if parent in graph:
            for child in siblings:
                graph[parent].append((child, 0))
                graph[child].append((parent, 0))

    orientation = {}
    for start in grey:
        if start in orientation:
            continue
        orientation[start] = 0
        pending = [start]
        while pending:
            pair = pending.pop()
            for neighbor, flip in graph[pair]:
                expected = orientation[pair] ^ flip
                if neighbor not in orientation:
                    orientation[neighbor] = expected
                    pending.append(neighbor)
                elif orientation[neighbor] != expected:
                    raise ValueError("Conflicting grey-pair orientation constraints")

    sequence = [0] * len(structure)
    for pair, color in color_by_pair.items():
        left, right = pair
        if color == 0:
            bases = (2, 1)
        elif color == 1:
            bases = (1, 2)
        else:
            bases = (0, 3) if orientation[pair] == 0 else (3, 0)
        sequence[left], sequence[right] = bases
    return tuple(sequence)

def evaluate_design(
    sequence: tuple[int, ...],
    structure: str,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> tuple[int, int, int, int, int, int, int, float]:
    pairs, _ = parse_target(structure)
    if not sequence or len(sequence) != len(structure) or any(base not in {0, 1, 2, 3} for base in sequence):
        raise ValueError("Sequence must match target length and use codes 0, 1, 2, 3")
    if not isinstance(minimum_span, int) or minimum_span < 0:
        raise ValueError("minimum_span must be a nonnegative integer")

    allowed = {(2, 1), (1, 2), (0, 3), (3, 0), (2, 3), (3, 2)}
    n = len(sequence)
    score = [[0] * n for _ in range(n)]
    count = [[1] * n for _ in range(n)]

    def _entry(matrix, left, right, empty):
        return matrix[left][right] if left <= right else empty

    for width in range(2, n + 1):
        for left in range(n - width + 1):
            right = left + width - 1
            best = _entry(score, left, right - 1, 0)
            ways = _entry(count, left, right - 1, 1)
            for partner in range(left, right - minimum_span):
                if (sequence[partner], sequence[right]) not in allowed:
                    continue
                candidate = _entry(score, left, partner - 1, 0) + _entry(score, partner + 1, right - 1, 0) + 1
                new_ways = _entry(count, left, partner - 1, 1) * _entry(count, partner + 1, right - 1, 1)
                if candidate > best:
                    best, ways = candidate, new_ways
                elif candidate == best:
                    ways += new_ways
            score[left][right], count[left][right] = best, ways

    compatible = int(all(
        (sequence[left], sequence[right]) in allowed and right - left - 1 >= minimum_span
        for left, right in pairs
    ))
    maximum_pairs = score[0][n - 1]
    optimal_count = count[0][n - 1]
    unique_target = int(compatible == 1 and len(pairs) == maximum_pairs and optimal_count == 1)
    n_g = sequence.count(2)
    n_c = sequence.count(1)
    gc_percentage = round(100.0 * (n_g + n_c) / n, gc_decimals)
    return (compatible, len(pairs), maximum_pairs, optimal_count, unique_target, n_g, n_c, gc_percentage)

def run_pipeline(
    structure: str = "((((((...)))(((...)))(((...))))))",
    m: int = 2,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> float:
    parse_target(structure)
    _, _, m5_count, m3_count = analyse_target(structure)
    if m5_count or m3_count:
        return -1.0
    coloring = find_separated_coloring(structure, m)
    if coloring is None:
        return -1.0
    sequence = construct_sequence(structure, coloring)
    evaluation = evaluate_design(sequence, structure, minimum_span, gc_decimals)
    return evaluation[7]
SCICODE_GOLD_EOF
