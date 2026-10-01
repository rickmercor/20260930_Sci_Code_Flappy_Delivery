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


def _children_by_parent(edges):
    """Group edges by parent node as (child, left, right) entries."""
    table = {}
    for child, parent, left, right in edges:
        table.setdefault(parent, []).append((int(child), float(left), float(right)))
    return table


def _next_child_edge(node, position, table, sequence_length):
    """Earliest start of a child edge of node that opens after position."""
    starts = [left for _, left, _ in table.get(node, ()) if left > position]
    return min(starts) if starts else sequence_length


def _descendants(node, position, table, cache, sequence_length):
    """Descendant leaves of node at position, memoized with an expiry bound."""
    cached = cache.get(node)
    if cached is not None and cached[1] > position:
        return cached[0]
    spanning = [
        entry for entry in table.get(node, ()) if entry[1] <= position < entry[2]
    ]
    expiry = _next_child_edge(node, position, table, sequence_length)
    if not spanning:
        cache[node] = (frozenset([node]), expiry)
        return cache[node][0]
    reached = set()
    for child, _, right in spanning:
        reached |= _descendants(child, position, table, cache, sequence_length)
        expiry = min(expiry, right, cache[child][1])
    cache[node] = (frozenset(reached), expiry)
    return cache[node][0]


def _clade_end(node, position, table, breakpoints, sequence_length):
    """Smallest position above the given one where the clade of node differs."""
    current = _descendants(node, position, table, {}, sequence_length)
    for point in breakpoints:
        if point <= position:
            continue
        if _descendants(node, point, table, {}, sequence_length) != current:
            return point
    return sequence_length


def arg_carrier_genotypes(
    edges: list,
    mutations: list,
    haplotype_pairs: list,
    sequence_length: float = 100.0,
) -> tuple:
    """Reference implementation."""
    if len(mutations) == 0:
        raise ValueError("at least one mutation is required")
    sequence_length = float(sequence_length)
    for _, position in mutations:
        if not 0.0 <= float(position) < sequence_length:
            raise ValueError("mutation position outside [0, sequence_length)")
    for pair in haplotype_pairs:
        if len(pair) != 2:
            raise ValueError("each individual needs exactly two haplotype leaves")

    table = _children_by_parent(edges)
    breakpoints = sorted(
        {float(edge[2]) for edge in edges} | {float(edge[3]) for edge in edges}
    )
    breakpoints = [point for point in breakpoints if 0.0 < point < sequence_length]
    cache = {}
    columns = []
    extents = []
    for node, position in sorted(mutations, key=lambda item: float(item[1])):
        position = float(position)
        for stale in [key for key, value in cache.items() if value[1] <= position]:
            del cache[stale]
        carriers = _descendants(int(node), position, table, cache, sequence_length)
        columns.append(
            [sum(1 for hap in pair if hap in carriers) for pair in haplotype_pairs]
        )
        extents.append(
            _clade_end(int(node), position, table, breakpoints, sequence_length)
        )
    return np.array(columns, dtype=float).T, np.array(extents, dtype=float)

import numpy as np


def standardize_arg_genotypes(
    genotypes: "np.ndarray",
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
) -> "np.ndarray":
    """Reference implementation."""
    genotypes = np.asarray(genotypes, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be a two-dimensional array")
    if np.any(genotypes < 0.0) or np.any(genotypes > 2.0):
        raise ValueError("diploid allele counts must lie in {0, 1, 2}")
    if int(min_minor_allele_count) < 1:
        raise ValueError("min_minor_allele_count must be at least 1")

    n_haplotypes = 2 * genotypes.shape[0]
    allele_count = genotypes.sum(axis=0)
    minor_count = np.minimum(allele_count, n_haplotypes - allele_count)
    keep = minor_count >= int(min_minor_allele_count)
    if not np.any(keep):
        raise ValueError("no mutation passes the minor allele count filter")

    retained = genotypes[:, keep]
    freq = retained.sum(axis=0) / n_haplotypes
    weight = (freq * (1.0 - freq)) ** (float(alpha) / 2.0)
    return (retained - 2.0 * freq) * weight

import numpy as np


def build_arg_grm(x_std: "np.ndarray") -> tuple:
    """Reference implementation."""
    x_std = np.asarray(x_std, dtype=float)
    if x_std.ndim != 2:
        raise ValueError("x_std must be a two-dimensional array")
    n_individuals = x_std.shape[0]
    cross = x_std @ x_std.T
    total = float(np.trace(cross))
    if total <= 0.0:
        raise ValueError("x_std carries no variance, the scaling factor is undefined")
    scale = total / n_individuals
    return cross / scale, scale

import numpy as np


def hutchinson_trace_squared(
    x_std: "np.ndarray",
    scale: float,
    num_probes: int = 200,
    seed: int = 101,
) -> float:
    """Reference implementation."""
    x_std = np.asarray(x_std, dtype=float)
    if int(num_probes) < 1:
        raise ValueError("num_probes must be at least 1")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")

    n_individuals = x_std.shape[0]
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((int(num_probes), n_individuals))
    applied = (x_std @ (x_std.T @ probes.T)).T / float(scale)
    return float(np.mean(np.sum(applied * applied, axis=1)))

import numpy as np


def rhe_variance_component(
    grm: "np.ndarray",
    phenotype: "np.ndarray",
    trace_r_squared: float,
) -> tuple:
    """Reference implementation."""
    grm = np.asarray(grm, dtype=float)
    trait = np.asarray(phenotype, dtype=float).ravel()
    n_individuals = grm.shape[0]
    if trait.size != n_individuals:
        raise ValueError("phenotype length does not match the relatedness matrix")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    trace_r = float(n_individuals)
    if abs(float(trace_r_squared) - trace_r) <= 0.0:
        raise ValueError("the moment conditions are not independent")
    standardised = centred * np.sqrt(n_individuals / scale)
    quadratic = float(standardised @ (grm @ standardised))
    sigma_g_squared = (quadratic - trace_r) / (float(trace_r_squared) - trace_r)
    sigma_e_squared = (float(standardised @ standardised)
                       - sigma_g_squared * trace_r) / n_individuals
    return float(sigma_g_squared), float(sigma_e_squared)

import numpy as np


def loco_covariance_product(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    vectors: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    if len(blocks) == 0:
        raise ValueError("at least one genotype block is required")
    focal_index = int(focal_index)
    if not 0 <= focal_index < len(blocks):
        raise ValueError("focal_index is out of range")
    matrices = [np.asarray(block, dtype=float) for block in blocks]
    n_individuals = matrices[0].shape[0]
    for block in matrices:
        if block.ndim != 2 or block.shape[0] != n_individuals:
            raise ValueError("blocks disagree on the number of individuals")
    retained = sum(block.shape[1] for index, block in enumerate(matrices)
                   if index != focal_index)
    if retained == 0:
        raise ValueError("no variants remain once the focal chromosome is dropped")
    pooled = sum(float(np.trace(block @ block.T)) for block in matrices)
    if pooled <= 0.0:
        raise ValueError("the genotype blocks carry no variance")
    scale = pooled / n_individuals
    target = np.asarray(vectors, dtype=float)
    accumulated = np.zeros_like(target)
    for index, block in enumerate(matrices):
        if index == focal_index:
            continue
        accumulated = accumulated + block @ (block.T @ target)
    return float(sigma_e_squared) * target + (
        float(sigma_g_squared) / scale) * accumulated

import numpy as np


def conjugate_gradient_solve(
    blocks: list,
    focal_index: int,
    sigma_g_squared: float,
    sigma_e_squared: float,
    phenotype: "np.ndarray",
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> tuple:
    """Reference implementation."""
    matrices = [np.asarray(block, dtype=float) for block in blocks]
    if len(matrices) == 0:
        raise ValueError("at least one genotype block is required")
    focal_index = int(focal_index)
    if not 0 <= focal_index < len(matrices):
        raise ValueError("focal_index is out of range")
    n_individuals = matrices[0].shape[0]
    trait = np.asarray(phenotype, dtype=float).ravel()
    if trait.size != n_individuals:
        raise ValueError("phenotype length does not match the genotype blocks")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be at least 1")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    right_hand_side = centred * np.sqrt(n_individuals / scale)

    retained = sum(block.shape[1] for index, block in enumerate(matrices)
                   if index != focal_index)
    if retained == 0:
        raise ValueError("no variants remain once the focal chromosome is dropped")
    pooled = sum(float(np.trace(block @ block.T)) for block in matrices)
    if pooled <= 0.0:
        raise ValueError("the genotype blocks carry no variance")
    scale = pooled / n_individuals
    weight = float(sigma_g_squared) / scale

    def _apply(vector):
        accumulated = np.zeros_like(vector)
        for index, block in enumerate(matrices):
            if index == focal_index:
                continue
            accumulated = accumulated + block @ (block.T @ vector)
        return float(sigma_e_squared) * vector + weight * accumulated

    solution = np.zeros(n_individuals, dtype=float)
    residual = right_hand_side.copy()
    direction = right_hand_side.copy()
    delta_new = float(residual @ residual)
    threshold = float(tolerance) * float(np.linalg.norm(right_hand_side))
    iterations = 0
    for _ in range(int(max_iterations)):
        product = _apply(direction)
        step = delta_new / float(direction @ product)
        solution = solution + step * direction
        residual = residual - step * product
        iterations += 1
        delta_old, delta_new = delta_new, float(residual @ residual)
        if np.sqrt(delta_new) <= threshold:
            break
        direction = residual + (delta_new / delta_old) * direction
    return solution, int(iterations)

import numpy as np


def blup_residual_phenotype(
    prediction: "np.ndarray",
    phenotype: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    predicted = np.asarray(prediction, dtype=float).ravel()
    trait = np.asarray(phenotype, dtype=float).ravel()
    if predicted.size != trait.size:
        raise ValueError("prediction and phenotype differ in length")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    standardised = centred * np.sqrt(trait.size / scale)
    return standardised - predicted

import numpy as np


def grammar_gamma_statistic(
    genotype_column: "np.ndarray",
    residual: "np.ndarray",
) -> float:
    """Reference implementation."""
    variant = np.asarray(genotype_column, dtype=float).ravel()
    trait = np.asarray(residual, dtype=float).ravel()
    if variant.size != trait.size:
        raise ValueError("genotype column and residual differ in length")
    denominator = float(variant @ variant)
    if denominator <= 0.0:
        raise ValueError("the variant carries no variation")
    projection = float(variant @ trait)
    return projection * projection / denominator

import numpy as np


def run_full_pipeline(
    chromosome_edges: list,
    chromosome_mutations: list,
    chromosome_spans: list,
    haplotype_pairs: list,
    phenotype: list,
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
    num_probes: int = 200,
    probe_seed: int = 101,
    focal_index: int = 0,
    focal_variant: int = 0,
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> float:
    """Reference implementation."""
    if not (len(chromosome_edges) == len(chromosome_mutations) == len(chromosome_spans)):
        raise ValueError("the per-chromosome lists disagree in length")
    if not 0 <= int(focal_index) < len(chromosome_edges):
        raise ValueError("focal_index does not name one of the chromosomes")

    blocks = []
    for edges, mutations, span in zip(
        chromosome_edges, chromosome_mutations, chromosome_spans
    ):
        genotypes, _clade_end = arg_carrier_genotypes(
            edges, mutations, haplotype_pairs, span
        )
        blocks.append(standardize_arg_genotypes(
            genotypes, alpha, min_minor_allele_count
        ))
    if not 0 <= int(focal_variant) < blocks[int(focal_index)].shape[1]:
        raise ValueError("focal_variant does not name a retained variant")

    pooled = np.hstack(blocks)
    grm, scale = build_arg_grm(pooled)
    trace_r_squared = hutchinson_trace_squared(
        pooled, scale, num_probes, probe_seed
    )
    sigma_g_squared, sigma_e_squared = rhe_variance_component(
        grm, phenotype, trace_r_squared
    )
    solution, _steps = conjugate_gradient_solve(
        blocks, focal_index, sigma_g_squared, sigma_e_squared, phenotype,
        tolerance, max_iterations,
    )
    prediction = loco_covariance_product(
        blocks, focal_index, sigma_g_squared, 0.0, solution
    )
    residual = blup_residual_phenotype(prediction, phenotype)
    return grammar_gamma_statistic(
        blocks[int(focal_index)][:, int(focal_variant)], residual
    )
SCICODE_GOLD_EOF
