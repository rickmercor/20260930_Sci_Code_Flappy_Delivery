#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from numbers import Integral


def surface_descriptor(genus, marks):
    if isinstance(genus, bool) or not isinstance(genus, Integral) or genus not in (0, 1):
        raise ValueError('genus outside domain')
    try:
        values = tuple(marks)
    except (TypeError, ValueError):
        raise ValueError('marks must be an integer sequence') from None
    if not 1 <= len(values) <= 3:
        raise ValueError('boundary count outside domain')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in values):
        raise ValueError('invalid mark count')
    g = int(genus)
    b = tuple(sorted(int(n) for n in values))
    edges = sum(b) + 3 * len(b) + 6 * g - 6
    loops = 2 * g + len(b) - 1
    degree = edges - loops
    if not 0 <= edges <= 11 or (edges == 0 and (g, b) != (0, (3,))) or (edges > 0 and degree <= 0):
        raise ValueError('surface outside domain')
    return g, b, edges, loops, degree

def nonseparating_cuts(genus, marks):
    g, b, _, _, _ = surface_descriptor(genus, marks)
    out = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            rest = tuple(n for k, n in enumerate(b) if k not in (i, j))
            child = (g, tuple(sorted(rest + (b[i] + b[j] + 2,))))
            out.append(((child,), b[i] * b[j]))
    if g:
        for i, n in enumerate(b):
            rest = b[:i] + b[i + 1:]
            for a in range(n + 1):
                child = (g - 1, tuple(sorted(rest + (a + 1, n - a + 1))))
                out.append(((child,), n / 2))
    return tuple(out)

def separating_cuts(genus, marks):
    g, b, _, _, _ = surface_descriptor(genus, marks)
    out = []
    for i, n in enumerate(b):
        rest = b[:i] + b[i + 1:]
        for h in range(g + 1):
            for a in range(n + 1):
                for mask in range(1 << len(rest)):
                    left = tuple(x for k, x in enumerate(rest) if mask & (1 << k))
                    right = tuple(x for k, x in enumerate(rest) if not mask & (1 << k))
                    pair = tuple(sorted(((h, tuple(sorted(left + (a + 1,)))), (g - h, tuple(sorted(right + (n - a + 1,)))))))
                    if any(part in ((0, (1,)), (0, (2,))) for part in pair):
                        continue
                    out.append((pair, n / 2))
    return tuple(out)

def aggregate_cuts(genus, marks):
    totals = {}
    for parts, weight in nonseparating_cuts(genus, marks) + separating_cuts(genus, marks):
        key = tuple(sorted((g, tuple(sorted(b))) for g, b in parts))
        totals[key] = totals.get(key, 0) + weight
    return tuple(sorted(totals.items()))

from fractions import Fraction
from functools import lru_cache


def reduced_surface_weight(genus, marks):
    g, b, _, _, _ = surface_descriptor(genus, marks)

    @lru_cache(None)
    def visit(g, b):
        if (g, b) == (0, (3,)):
            return Fraction(1)
        degree = sum(b) + 2 * len(b) + 4 * g - 5
        total = Fraction(0)
        for parts, mult in aggregate_cuts(g, b):
            term = Fraction(mult)
            for child_g, child_b in parts:
                term *= visit(child_g, child_b)
            total += term
        return total / degree

    return float(visit(g, b))

from math import fsum


def first_cut_distribution(genus, marks):
    g, b, edges, _, _ = surface_descriptor(genus, marks)
    if edges == 0:
        raise ValueError('terminal surface has no first cut')
    weights = []
    for parts, mult in aggregate_cuts(g, b):
        weight = float(mult)
        for child_g, child_b in parts:
            weight *= reduced_surface_weight(child_g, child_b)
        weights.append(weight)
    total = fsum(weights)
    return total, tuple(weight / total for weight in weights)

from math import fsum


def surface_collision(genus, marks):
    _, probabilities = first_cut_distribution(genus, marks)
    return float(fsum(p * p for p in probabilities))
SCICODE_GOLD_EOF
