#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _validate_population(sites, name, n_sites):
    """Check one population's genotype strings and return its lineage count."""
    if len(sites) != n_sites:
        raise ValueError("all three populations must cover the same sites")
    widths = {len(s) for s in sites}
    if len(widths) != 1:
        raise ValueError("%s must use the same number of lineages at every site" % name)
    width = widths.pop()
    if width == 0:
        raise ValueError("%s must sample at least one lineage" % name)
    for s in sites:
        for ch in s:
            if ch != "." and not ch.isdigit():
                raise ValueError("genotype characters must be digits or '.'")
    return width


def site_allele_counts(P1_sites: list[str], P2_sites: list[str], P3_sites: list[str]) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Reference implementation."""
    P1_sites = list(P1_sites)
    P2_sites = list(P2_sites)
    P3_sites = list(P3_sites)
    n_sites = len(P1_sites)
    if n_sites == 0:
        raise ValueError("at least one site is required")
    _validate_population(P1_sites, "P1", n_sites)
    _validate_population(P2_sites, "P2", n_sites)
    _validate_population(P3_sites, "P3", n_sites)

    counts = []
    totals = []
    for site in range(n_sites):
        strings = (P1_sites[site], P2_sites[site], P3_sites[site])
        labels = sorted({ch for s in strings for ch in s if ch != "."})
        table = [[0, 0, 0] for _ in labels]
        total = [0, 0, 0]
        index = {label: k for k, label in enumerate(labels)}
        for j, s in enumerate(strings):
            for ch in s:
                if ch == ".":
                    continue
                table[index[ch]][j] += 1
                total[j] += 1
        counts.append(table)
        totals.append(total)
    return counts, totals

def rarefaction_subsample_sizes(totals: list[list[int]]) -> list[int]:
    """Reference implementation."""
    totals = [list(t) for t in totals]
    if len(totals) == 0:
        raise ValueError("at least one site is required")
    for t in totals:
        if len(t) != 3:
            raise ValueError("each site needs three non-missing sample sizes")
        for value in t:
            if int(value) != value or value < 0:
                raise ValueError("sample sizes must be non-negative integers")
    return [int(min(t)) for t in totals]

from math import comb


def _check_site_table(counts, totals, g):
    """Validate one site's count table against its sample sizes and g."""
    counts = [list(row) for row in counts]
    totals = list(totals)
    if len(counts) == 0:
        raise ValueError("at least one allele is required")
    if len(totals) != 3:
        raise ValueError("totals must hold three non-missing sample sizes")
    if int(g) != g or g < 0:
        raise ValueError("g must be a non-negative integer")
    for row in counts:
        if len(row) != 3:
            raise ValueError("each allele needs three counts")
        for value in row:
            if int(value) != value or value < 0:
                raise ValueError("allele counts must be non-negative integers")
    for j in range(3):
        if sum(row[j] for row in counts) != totals[j]:
            raise ValueError("allele counts must sum to the non-missing sample size")
        if g > totals[j]:
            raise ValueError("g cannot exceed a non-missing sample size")
    return counts, totals, int(g)


def rarefied_pair_private_counts(counts: list[list[int]], totals: list[int], g: int) -> tuple[float, float]:
    """Reference implementation."""
    counts, totals, g = _check_site_table(counts, totals, g)
    pi_P1P3 = 0.0
    pi_P2P3 = 0.0
    for row in counts:
        absent = [comb(totals[j] - row[j], g) / comb(totals[j], g) for j in range(3)]
        present = [1.0 - x for x in absent]
        pi_P1P3 += present[0] * absent[1] * present[2]
        pi_P2P3 += absent[0] * present[1] * present[2]
    return float(pi_P1P3), float(pi_P2P3)

from math import comb


def _check_site_table(counts, totals, g):
    """Validate one site's count table against its sample sizes and g."""
    counts = [list(row) for row in counts]
    totals = list(totals)
    if len(counts) == 0:
        raise ValueError("at least one allele is required")
    if len(totals) != 3:
        raise ValueError("totals must hold three non-missing sample sizes")
    if int(g) != g or g < 0:
        raise ValueError("g must be a non-negative integer")
    for row in counts:
        if len(row) != 3:
            raise ValueError("each allele needs three counts")
        for value in row:
            if int(value) != value or value < 0:
                raise ValueError("allele counts must be non-negative integers")
    for j in range(3):
        if sum(row[j] for row in counts) != totals[j]:
            raise ValueError("allele counts must sum to the non-missing sample size")
        if g > totals[j]:
            raise ValueError("g cannot exceed a non-missing sample size")
    return counts, totals, int(g)


def rarefied_allelic_richness(counts: list[list[int]], totals: list[int], g: int) -> float:
    """Reference implementation."""
    counts, totals, g = _check_site_table(counts, totals, g)
    pooled = sum(totals)
    kappa = 0.0
    for row in counts:
        kappa += 1.0 - comb(pooled - sum(row), g) / comb(pooled, g)
    return float(kappa)

def window_dstar_sums(positions: list[int], pair_counts: list[list[float]], richness: list[float], window_size: int) -> tuple[list[int], list[float], list[float]]:
    """Reference implementation."""
    positions = [p for p in positions]
    pair_counts = [list(pc) for pc in pair_counts]
    richness = [float(k) for k in richness]
    n_sites = len(positions)
    if n_sites == 0:
        raise ValueError("at least one site is required")
    if len(pair_counts) != n_sites or len(richness) != n_sites:
        raise ValueError("per-site inputs must cover the same sites")
    if int(window_size) != window_size or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    for p in positions:
        if int(p) != p or p < 0:
            raise ValueError("positions must be non-negative integers")
    for pc in pair_counts:
        if len(pc) != 2:
            raise ValueError("each site needs two private allele counts")
    for k in richness:
        if k < 0.0:
            raise ValueError("allelic richness must be non-negative")

    window_size = int(window_size)
    numerator_by_window = {}
    denominator_by_window = {}
    for site in range(n_sites):
        pi_P1P3, pi_P2P3 = pair_counts[site]
        if pi_P1P3 <= 0.0 and pi_P2P3 <= 0.0:
            continue
        window = int(positions[site]) // window_size
        numerator_by_window[window] = (
            numerator_by_window.get(window, 0.0) + (pi_P2P3 - pi_P1P3))
        denominator_by_window[window] = (
            denominator_by_window.get(window, 0.0) + richness[site])

    window_indices = sorted(
        w for w in denominator_by_window if denominator_by_window[w] > 0.0)
    numerators = [float(numerator_by_window[w]) for w in window_indices]
    denominators = [float(denominator_by_window[w]) for w in window_indices]
    return window_indices, numerators, denominators

import numpy as np


def block_bootstrap_moments(numerators: list[float], denominators: list[float], n_replicates: int, seed: int) -> tuple[float, float]:
    """Reference implementation."""
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if numerators.ndim != 1 or denominators.ndim != 1:
        raise ValueError("window sums must be one dimensional")
    if numerators.size == 0 or numerators.size != denominators.size:
        raise ValueError("numerators and denominators must cover the same windows")
    if not np.all(denominators > 0.0):
        raise ValueError("every retained window needs a positive denominator")
    if int(n_replicates) != n_replicates or n_replicates < 2:
        raise ValueError("n_replicates must be an integer of at least two")

    n_blocks = numerators.size
    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_replicates), dtype=float)
    for r in range(int(n_replicates)):
        drawn = rng.integers(0, n_blocks, size=n_blocks)
        replicates[r] = numerators[drawn].sum() / denominators[drawn].sum()
    return float(replicates.mean()), float(replicates.std(ddof=1))

def run_full_pipeline(positions: list[int], P1_sites: list[str], P2_sites: list[str], P3_sites: list[str], window_size: int = 500,
                              n_replicates: int = 2000, seed: int = 2026) -> float:
    """Reference implementation."""
    positions = [p for p in positions]
    counts, totals = site_allele_counts(P1_sites, P2_sites, P3_sites)
    if len(positions) != len(counts):
        raise ValueError("positions must cover the same sites as the genotypes")

    subsample_sizes = rarefaction_subsample_sizes(totals)

    pair_counts = []
    richness = []
    for site in range(len(counts)):
        g = subsample_sizes[site]
        pair_counts.append(
            list(rarefied_pair_private_counts(counts[site], totals[site], g)))
        richness.append(rarefied_allelic_richness(counts[site], totals[site], g))

    window_indices, numerators, denominators = window_dstar_sums(
        positions, pair_counts, richness, window_size)
    if len(window_indices) == 0:
        raise ValueError("no window has a positive denominator")

    mean, _sd = block_bootstrap_moments(
        numerators, denominators, n_replicates, seed)
    return float(mean)
SCICODE_GOLD_EOF
