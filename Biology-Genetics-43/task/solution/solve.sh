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
def build_loop_tree(structure: str, min_hairpin: int = 0) -> np.ndarray:
    """Reference implementation (one left-to-right stack pass)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not isinstance(structure, str) or len(structure) == 0:
        raise ValueError("structure must be a non-empty string")
    if set(structure) - set("()."):
        raise ValueError("structure may only contain '(', ')' and '.'")
    if not (_is_int(min_hairpin) and min_hairpin >= 0):
        raise ValueError("min_hairpin must be a nonnegative integer")
    n = len(structure)
    rows = [[-1, 0, n + 1, 0]]
    open_rows = [0]  # rows whose loops are open at the current position
    for position, symbol in enumerate(structure, start=1):
        current = open_rows[-1]
        if symbol == "(":
            rows.append([current, position, 0, 0])
            open_rows.append(len(rows) - 1)
        elif symbol == ")":
            if current == 0:
                raise ValueError("unbalanced brackets: ')' without a matching '('")
            if position - rows[current][1] - 1 < min_hairpin:
                raise ValueError("a pair encloses fewer than min_hairpin positions")
            rows[current][2] = position
            open_rows.pop()
        else:
            rows[current][3] += 1
    if len(open_rows) != 1:
        raise ValueError("unbalanced brackets: '(' without a matching ')'")
    return np.array(rows, dtype=np.int64)

import numpy as np
def profile_helix_motifs(tree: np.ndarray) -> np.ndarray:
    """Reference implementation (helix chains followed from their first pair)."""
    import numpy as np

    table = np.asarray(tree)
    if (table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 1
            or not np.issubdtype(table.dtype, np.integer)):
        raise ValueError("tree must be a two-dimensional integer array with four columns")
    parent = [int(x) for x in table[:, 0]]
    unpaired = [int(x) for x in table[:, 3]]
    if parent[0] != -1 or unpaired[0] < 0:
        raise ValueError("row 0 must describe the exterior loop")
    n_pairs = table.shape[0] - 1
    children = [[] for _ in range(n_pairs + 1)]
    for k in range(1, n_pairs + 1):
        if not (0 <= parent[k] < k) or unpaired[k] < 0:
            raise ValueError("every pair needs an earlier parent row and a nonnegative count")
        children[parent[k]].append(k)

    def _continues(v):
        # A pair extends its helix into the single pair it encloses.
        return v >= 1 and len(children[v]) == 1 and unpaired[v] == 0

    lengths = []
    for k in range(1, n_pairs + 1):
        if _continues(parent[k]):
            continue  # k is inside a helix that started higher up
        length, v = 1, k
        while _continues(v):
            v = children[v][0]
            length += 1
        lengths.append(length)
    five_pair, three_pair = 0, 0
    for v in range(n_pairs + 1):
        pair_count = len(children[v]) + (1 if v >= 1 else 0)
        if pair_count >= 5:
            five_pair += 1
        if pair_count >= 3 and unpaired[v] >= 1:
            three_pair += 1
    shortest = min(lengths) if lengths else 0
    return np.array([shortest, len(lengths), five_pair, three_pair], dtype=np.int64)

import numpy as np
def list_proper_child_contents(parent_content: int, n_children: int) -> np.ndarray:
    """Reference implementation (enumeration under the color-count and gray-content rules)."""
    import itertools
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_int(parent_content) and -1 <= parent_content <= 3):
        raise ValueError("parent_content must be an integer in {-1, 0, 1, 2, 3}")
    if not (_is_int(n_children) and n_children >= 0):
        raise ValueError("n_children must be a nonnegative integer")
    parent_content, n_children = int(parent_content), int(n_children)
    color = ("black", "white", "gray", "gray")
    complement = {"black": "white", "white": "black", "gray": "gray"}
    # At most one black, one white and two gray entries fit in the list.
    capacity = 4 - (1 if parent_content >= 0 else 0)
    if n_children > capacity:
        return np.zeros((0, n_children), dtype=np.int64)
    rows = []
    for combo in itertools.product(range(4), repeat=n_children):
        colors = [color[c] for c in combo]
        if parent_content >= 0:
            colors.append(complement[color[parent_content]])
        if colors.count("black") > 1 or colors.count("white") > 1 or colors.count("gray") > 2:
            continue
        grays = [c for c in combo if c >= 2]
        if len(set(grays)) != len(grays):
            continue  # two gray enclosed pairs must differ
        if parent_content >= 2 and any(c != parent_content for c in grays):
            continue  # a gray enclosed pair must match a gray closing pair
        rows.append(combo)
    return np.array(rows, dtype=np.int64).reshape(len(rows), n_children)

import numpy as np
def count_restricted_designs(
    tree: np.ndarray,
    modulus: int,
    leaf_residues: "Sequence[int]",
    gray_residues: "Sequence[int]",
) -> np.ndarray:
    """Reference implementation (memoized tree recursion over pair, content and residue)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    table = np.asarray(tree)
    if (table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 1
            or not np.issubdtype(table.dtype, np.integer)):
        raise ValueError("tree must be a two-dimensional integer array with four columns")
    parent = [int(x) for x in table[:, 0]]
    unpaired = [int(x) for x in table[:, 3]]
    if parent[0] != -1 or unpaired[0] < 0:
        raise ValueError("row 0 must describe the exterior loop")
    n_pairs = table.shape[0] - 1
    children = [[] for _ in range(n_pairs + 1)]
    for k in range(1, n_pairs + 1):
        if not (0 <= parent[k] < k) or unpaired[k] < 0:
            raise ValueError("every pair needs an earlier parent row and a nonnegative count")
        children[parent[k]].append(k)
    if not (_is_int(modulus) and modulus >= 1):
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    allowed = []
    for residues in (leaf_residues, gray_residues):
        values = list(residues)
        if not all(_is_int(r) and 0 <= r < m for r in values):
            raise ValueError("residues must be integers in [0, modulus)")
        allowed.append({int(r) for r in values})
    leaf_ok, gray_ok = allowed
    shift = (1, -1, 0, 0)
    gc_pair = (2, 2, 0, 0)
    tuples, memo = {}, {}

    def _proper(content, count):
        key = (content, count)
        if key not in tuples:
            rows = list_proper_child_contents(content, count)
            tuples[key] = [tuple(int(x) for x in row) for row in rows]
        return tuples[key]

    def _loop(v, content, inner):
        # Sequences of the pairs directly enclosed by loop v (and below), given
        # the closing content and the residue ``inner`` of its interior.
        if unpaired[v] > 0 and inner not in leaf_ok:
            return 0, 0
        total, gc_total = 0, 0
        for combo in _proper(content, len(children[v])):
            count, gc_sum = 1, 0
            for kid, kid_content in zip(children[v], combo):
                c, g = _pair(kid, kid_content, inner)
                count, gc_sum = count * c, gc_sum * c + count * g
                if count == 0:
                    break
            total += count
            gc_total += gc_sum
        return total, gc_total

    def _pair(k, content, residue):
        key = (k, content, residue)
        if key not in memo:
            if content >= 2 and residue not in gray_ok:
                memo[key] = (0, 0)
            else:
                c, g = _loop(k, content, (residue + shift[content]) % m)
                memo[key] = (c, g + gc_pair[content] * c)
        return memo[key]

    count, gc_total = _loop(0, -1, 0)
    return np.array([count, gc_total], dtype=np.int64)

import numpy as np
def compute_sampler_normalizer(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Reference implementation (one restricted count per residue set)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus < 1:
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    total, admissible = 0, 0
    for mask in range(2 ** m):
        leaf = [r for r in range(m) if (mask >> r) & 1]
        gray = [r for r in range(m) if not (mask >> r) & 1]
        count = int(count_restricted_designs(tree, m, leaf, gray)[0])
        total += count
        admissible += 1 if count > 0 else 0
    return np.array([total, admissible], dtype=np.int64)

import numpy as np
def find_minimal_modulus(tree: np.ndarray, helix_profile: np.ndarray, max_modulus: int) -> int:
    """Reference implementation (motif guard, helix-length guarantee, then an increasing scan)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    profile = np.asarray(helix_profile)
    if (profile.shape != (4,) or not np.issubdtype(profile.dtype, np.integer)
            or np.any(profile < 0)):
        raise ValueError("helix_profile must be a length-4 array of nonnegative integers")
    if not (_is_int(max_modulus) and max_modulus >= 2):
        raise ValueError("max_modulus must be an integer of at least 2")
    shortest, n_helices, five_pair, three_pair = (int(x) for x in profile)
    if five_pair + three_pair > 0:
        raise ValueError("a loop carries a locally undesignable motif")
    if n_helices > 0 and shortest >= 3:
        return 2  # every helix has three or more pairs: modulus 2 always works
    for m in range(2, int(max_modulus) + 1):
        if int(compute_sampler_normalizer(tree, m)[0]) > 0:
            return m
    raise ValueError("no modulus up to max_modulus admits a separated sequence")

import numpy as np
def count_distinct_designs(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Reference implementation (inclusion-exclusion over disjoint allowed residue sets)."""
    import itertools
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus < 1:
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    # Each residue is allowed for unpaired positions (0), for A-U-type pairs
    # (1) or for neither (2). A sequence whose occupied residue sets are L and
    # G (disjoint) is counted by every labelling with L inside the first class
    # and G inside the second; the residues it leaves free each contribute
    # +1 + 1 - 1 = 1, so the signed sum counts it exactly once, while
    # sequences with L and G overlapping are counted by no labelling at all.
    count, gc_total = 0, 0
    for labels in itertools.product(range(3), repeat=m):
        leaf = [r for r in range(m) if labels[r] == 0]
        gray = [r for r in range(m) if labels[r] == 1]
        sign = -1 if (m - len(leaf) - len(gray)) % 2 else 1
        c, g = count_restricted_designs(tree, m, leaf, gray)
        count += sign * int(c)
        gc_total += sign * int(g)
    return np.array([count, gc_total], dtype=np.int64)

import numpy as np
def estimate_design_gc_content(
    structure: str = "((.....))((..((...((((.((......))))((((.....)))(((...))))))...)).))(((.(((((......))))))))",
    max_modulus: int = 9,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    table = build_loop_tree(structure, 0)
    profile = profile_helix_motifs(table)
    m_star = find_minimal_modulus(table, profile, max_modulus)
    count, gc_total = (int(x) for x in count_distinct_designs(table, m_star))
    if count <= 0:
        raise ValueError("no separated sequence at the smallest workable modulus")
    length = int(table[0, 2]) - 1  # row 0 stores n + 1 as its 3' position
    return float(np.float64(gc_total) / (np.float64(count) * length))
SCICODE_GOLD_EOF
