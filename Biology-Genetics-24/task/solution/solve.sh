#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np  # noqa: E402, F811

def impute_predicted_expression(
    genotypes: "np.ndarray",
    weights: "np.ndarray",
    intercepts: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    genotypes = np.asarray(genotypes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be two dimensional")
    if weights.ndim != 3:
        raise ValueError("weights must be three dimensional")
    if intercepts.ndim != 2:
        raise ValueError("intercepts must be two dimensional")
    if genotypes.shape[1] != weights.shape[1]:
        raise ValueError("genotypes and weights must have the same SNP dimension")
    if intercepts.shape != (weights.shape[0], weights.shape[2]):
        raise ValueError("intercepts must match the resource and gene dimensions")
    if not (
        np.all(np.isfinite(genotypes))
        and np.all(np.isfinite(weights))
        and np.all(np.isfinite(intercepts))
    ):
        raise ValueError("all inputs must be finite")
    if np.any((genotypes < 0.0) | (genotypes > 2.0)):
        raise ValueError("genotype dosages must lie in [0, 2]")

    return np.einsum("ns,rsg->nrg", genotypes, weights) + intercepts[None, :, :]

import numpy as np  # noqa: E402, F811

def residualize_training_covariates(
    predicted_expression: "np.ndarray",
    covariates: "np.ndarray",
    training_mask: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    expression = np.asarray(predicted_expression, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    mask = np.asarray(training_mask)
    if expression.ndim != 3:
        raise ValueError("predicted_expression must be three dimensional")
    if covariates.ndim != 2:
        raise ValueError("covariates must be two dimensional")
    if covariates.shape[0] != expression.shape[0]:
        raise ValueError("expression and covariates must have the same sample count")
    if mask.shape != (expression.shape[0],):
        raise ValueError("training_mask must have one entry per sample")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("training_mask must be binary")
    if not (np.all(np.isfinite(expression)) and np.all(np.isfinite(covariates))):
        raise ValueError("expression and covariates must be finite")
    mask = mask.astype(bool)
    design = np.column_stack([np.ones(expression.shape[0]), covariates])
    if int(mask.sum()) < design.shape[1]:
        raise ValueError("training_mask selects too few rows for the design")
    if np.linalg.matrix_rank(design[mask]) != design.shape[1]:
        raise ValueError("the training covariate design must have full column rank")

    flat_expression = expression.reshape(expression.shape[0], -1)
    coefficients = np.linalg.lstsq(
        design[mask], flat_expression[mask], rcond=None
    )[0]
    fitted = design @ coefficients
    training_mean = flat_expression[mask].mean(axis=0)
    adjusted = flat_expression - fitted + training_mean
    return adjusted.reshape(expression.shape)

import numpy as np  # noqa: E402, F811

def _association_statistic(values, labels, method_index):
    """Return a case-control location difference."""
    cases = values[labels == 1]
    controls = values[labels == 0]
    if method_index == 0:
        return float(cases.mean() - controls.mean())
    return float(np.median(cases) - np.median(controls))


def compute_split_associations(
    adjusted_expression: "np.ndarray",
    labels: "np.ndarray",
    split_ids: "np.ndarray",
    n_permutations: int,
    seed: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    expression = np.asarray(adjusted_expression, dtype=float)
    labels = np.asarray(labels)
    split_ids = np.asarray(split_ids)
    if expression.ndim != 3:
        raise ValueError("adjusted_expression must be three dimensional")
    if labels.shape != (expression.shape[0],):
        raise ValueError("labels must have one entry per sample")
    if split_ids.shape != (expression.shape[0],):
        raise ValueError("split_ids must have one entry per sample")
    if not np.all(np.isfinite(expression)):
        raise ValueError("adjusted_expression must be finite")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    if not np.all((split_ids == 0) | (split_ids == 1)):
        raise ValueError("split_ids must contain only zero and one")
    if set(np.unique(split_ids).tolist()) != {0, 1}:
        raise ValueError("both training and validation splits must be present")
    for split_index in (0, 1):
        split_labels = labels[split_ids == split_index]
        if set(np.unique(split_labels).tolist()) != {0, 1}:
            raise ValueError("each split must contain at least one case and control")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n_resources = expression.shape[1]
    n_genes = expression.shape[2]
    effects = np.empty((2, n_resources, 2, n_genes), dtype=float)
    p_values = np.empty_like(effects)
    rng = np.random.default_rng(int(seed))
    for split_index in (0, 1):
        in_split = split_ids == split_index
        split_labels = labels[in_split].astype(int)
        split_expression = expression[in_split]
        permuted_labels = np.array(
            [rng.permutation(split_labels) for _ in range(n_permutations)],
            dtype=int,
        )
        for resource_index in range(n_resources):
            for method_index in range(2):
                for gene_index in range(n_genes):
                    values = split_expression[:, resource_index, gene_index]
                    observed = _association_statistic(
                        values, split_labels, method_index
                    )
                    null_statistics = np.array(
                        [
                            _association_statistic(values, permuted, method_index)
                            for permuted in permuted_labels
                        ]
                    )
                    exceedances = np.count_nonzero(
                        np.abs(null_statistics) + 1e-12 >= abs(observed)
                    )
                    effects[
                        split_index, resource_index, method_index, gene_index
                    ] = observed
                    p_values[
                        split_index, resource_index, method_index, gene_index
                    ] = (1.0 + exceedances) / (n_permutations + 1.0)
    return effects, p_values

import numpy as np  # noqa: E402, F811


def _benjamini_hochberg(values):
    """Return stable Benjamini-Hochberg adjusted values."""
    count = values.size
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    adjusted = ordered * count / np.arange(1, count + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(count, dtype=float)
    result[order] = adjusted
    return result


def adjust_discovery_records(
    effects: "np.ndarray",
    p_values: "np.ndarray",
    fdr_threshold: float,
    effect_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    effects = np.asarray(effects, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    if effects.ndim != 4 or p_values.ndim != 4 or effects.shape != p_values.shape:
        raise ValueError("effects and p_values must be matching four-dimensional arrays")
    if any(size == 0 for size in effects.shape):
        raise ValueError("effects and p_values must be non-empty in every dimension")
    if not np.all(np.isfinite(effects)):
        raise ValueError("effects must be finite")
    if not np.all(np.isfinite(p_values)) or np.any(
        (p_values < 0.0) | (p_values > 1.0)
    ):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    if not (
        isinstance(fdr_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(fdr_threshold)
        and 0.0 < float(fdr_threshold) <= 1.0
    ):
        raise ValueError("fdr_threshold must lie in (0, 1]")
    if not (
        isinstance(effect_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(effect_threshold)
        and float(effect_threshold) >= 0.0
    ):
        raise ValueError("effect_threshold must be finite and non-negative")

    fdr_values = np.empty_like(p_values)
    for index in np.ndindex(p_values.shape[:-1]):
        fdr_values[index] = _benjamini_hochberg(p_values[index])
    significant = (
        (fdr_values < float(fdr_threshold))
        & (np.abs(effects) >= float(effect_threshold))
    ).astype(np.uint8)
    return fdr_values, significant

import numpy as np  # noqa: E402, F811


def _logistic_map(value, midpoint, steepness):
    """Map a non-negative standardized effect smoothly to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))


def score_reproducible_genes(
    effects: "np.ndarray",
    fdr_values: "np.ndarray",
    significant: "np.ndarray",
    mean_midpoint: float,
    mean_steepness: float,
    max_midpoint: float,
    max_steepness: float,
) -> "np.ndarray":
    """Reference implementation."""
    effects = np.asarray(effects, dtype=float)
    fdr_values = np.asarray(fdr_values, dtype=float)
    significant = np.asarray(significant)
    if (
        effects.ndim != 4
        or fdr_values.shape != effects.shape
        or significant.shape != effects.shape
    ):
        raise ValueError("effects, fdr_values, and significant must be matching four-dimensional arrays")
    if any(size == 0 for size in effects.shape):
        raise ValueError("score arrays must be non-empty in every dimension")
    if not np.all(np.isfinite(effects)):
        raise ValueError("effects must be finite")
    if not np.all(np.isfinite(fdr_values)) or np.any(
        (fdr_values < 0.0) | (fdr_values > 1.0)
    ):
        raise ValueError("fdr_values must be finite and lie in [0, 1]")
    if not np.all((significant == 0) | (significant == 1)):
        raise ValueError("significant must be binary")
    significant = significant.astype(bool)
    if not np.any(significant):
        raise ValueError("at least one discovery record is required")
    midpoints = (mean_midpoint, max_midpoint)
    steepnesses = (mean_steepness, max_steepness)
    if not all(
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
        for value in midpoints
    ):
        raise ValueError("effect-map midpoints must be finite")
    if not all(
        isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
        and float(value) > 0.0
        for value in steepnesses
    ):
        raise ValueError("effect-map steepnesses must be finite and positive")

    n_resources = effects.shape[1]
    n_methods = effects.shape[2]
    n_genes = effects.shape[3]
    candidate = significant.any(axis=(0, 1, 2))
    hit_counts = significant.sum(axis=(0, 1, 2)).astype(float)
    breadth_counts = np.zeros(n_genes, dtype=float)
    for gene_index in range(n_genes):
        breadth_counts[gene_index] = sum(
            bool(significant[:, resource_index, method_index, gene_index].any())
            for resource_index in range(n_resources)
            for method_index in range(n_methods)
        )
    max_hits = float(hit_counts[candidate].max())
    max_breadth = float(breadth_counts[candidate].max())

    standardized = np.zeros_like(effects)
    for method_index in range(n_methods):
        method_hits = significant[:, :, method_index, :]
        method_values = effects[:, :, method_index, :][method_hits]
        if method_values.size == 0:
            continue
        median = float(np.median(method_values))
        first_quartile = float(np.quantile(method_values, 0.25))
        third_quartile = float(np.quantile(method_values, 0.75))
        scale = max(third_quartile - first_quartile, 1e-6)
        standardized[:, :, method_index, :] = np.abs(
            (effects[:, :, method_index, :] - median) / scale
        )

    gene_scores = np.zeros((n_genes, 4), dtype=float)
    for gene_index in range(n_genes):
        if not candidate[gene_index]:
            continue
        reproducibility = (
            0.6 * hit_counts[gene_index] / max_hits
            + 0.4 * breadth_counts[gene_index] / max_breadth
        )
        gene_hits = significant[:, :, :, gene_index]
        z_values = standardized[:, :, :, gene_index][gene_hits]
        mean_component = _logistic_map(
            float(z_values.mean()), float(mean_midpoint), float(mean_steepness)
        )
        max_component = _logistic_map(
            float(z_values.max()), float(max_midpoint), float(max_steepness)
        )
        effect_score = 0.7 * mean_component + 0.3 * max_component
        directions = np.sign(effects[:, :, :, gene_index][gene_hits])
        if np.all(directions > 0.0) or np.all(directions < 0.0):
            effect_score *= 1.1
        gene_fdr = fdr_values[:, :, :, gene_index][gene_hits]
        confidence = 0.6 * (1.0 - float(gene_fdr.min())) + 0.4 * (
            1.0 - float(gene_fdr.mean())
        )
        discovery_score = (
            0.4 * reproducibility + 0.3 * effect_score + 0.3 * confidence
        )
        gene_scores[gene_index] = (
            reproducibility,
            effect_score,
            confidence,
            discovery_score,
        )
    return gene_scores

import numpy as np  # noqa: E402, F811

def propagate_pathway_support(
    term_fdr: "np.ndarray",
    term_gene_membership: "np.ndarray",
    fdr_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    term_fdr = np.asarray(term_fdr, dtype=float)
    membership = np.asarray(term_gene_membership)
    if term_fdr.ndim != 2 or membership.ndim != 2:
        raise ValueError("term_fdr and term_gene_membership must be two dimensional")
    if any(size == 0 for size in term_fdr.shape) or any(
        size == 0 for size in membership.shape
    ):
        raise ValueError("pathway arrays must be non-empty")
    if term_fdr.shape[0] != membership.shape[0]:
        raise ValueError("pathway arrays must have the same term count")
    if not np.all(np.isfinite(term_fdr)) or np.any(
        (term_fdr < 0.0) | (term_fdr > 1.0)
    ):
        raise ValueError("term_fdr must be finite and lie in [0, 1]")
    if not np.all((membership == 0) | (membership == 1)):
        raise ValueError("term_gene_membership must be binary")
    if not (
        isinstance(fdr_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(fdr_threshold)
        and 0.0 < float(fdr_threshold) < 1.0
    ):
        raise ValueError("fdr_threshold must lie in (0, 1)")

    significant = term_fdr < float(fdr_threshold)
    robustness = significant.sum(axis=1).astype(float)
    strength = np.zeros(term_fdr.shape[0], dtype=float)
    for term_index in range(term_fdr.shape[0]):
        if robustness[term_index] > 0.0:
            minimum = max(
                float(term_fdr[term_index][significant[term_index]].min()),
                np.finfo(float).tiny,
            )
            strength[term_index] = -np.log10(minimum)
    weights = robustness * strength
    term_summary = np.column_stack([robustness, strength, weights])
    pathway_scores = membership.astype(float).T @ weights
    return term_summary, pathway_scores

import numpy as np  # noqa: E402, F811


def _average_rank_percentiles(values):
    """Map average ascending ranks to [0, 1], preserving ties."""
    count = values.size
    if count == 1:
        return np.ones(1, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks / (count - 1.0)


def integrate_target_evidence(
    discovery_scores: "np.ndarray",
    pathway_scores: "np.ndarray",
    druggability_scores: "np.ndarray",
    hub_scores: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    arrays = [
        np.asarray(discovery_scores, dtype=float),
        np.asarray(pathway_scores, dtype=float),
        np.asarray(druggability_scores, dtype=float),
        np.asarray(hub_scores, dtype=float),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all evidence inputs must be one dimensional")
    if arrays[0].size == 0 or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all evidence inputs must be non-empty and have matching shapes")
    if any(
        not np.all(np.isfinite(array)) or np.any(array < 0.0)
        for array in arrays
    ):
        raise ValueError("all evidence values must be finite and non-negative")
    candidate = arrays[0] > 0.0
    if not np.any(candidate):
        raise ValueError("at least one positive discovery score is required")

    normalized = np.zeros((arrays[0].size, 4), dtype=float)
    for component_index, array in enumerate(arrays):
        normalized[candidate, component_index] = _average_rank_percentiles(
            array[candidate]
        )
    core_score = (
        0.45 * normalized[:, 0]
        + 0.25 * normalized[:, 1]
        + 0.25 * normalized[:, 2]
        + 0.05 * normalized[:, 3]
    )
    core_score[~candidate] = 0.0
    return np.column_stack([normalized, core_score])

import numpy as np  # noqa: E402, F811

def aggregate_drug_evidence(
    core_scores: "np.ndarray",
    evidence_rows: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    core_scores = np.asarray(core_scores, dtype=float)
    evidence_rows = np.asarray(evidence_rows, dtype=float)
    if core_scores.ndim != 1 or core_scores.size == 0:
        raise ValueError("core_scores must be a non-empty vector")
    if not np.all(np.isfinite(core_scores)) or np.any(core_scores < 0.0):
        raise ValueError("core_scores must be finite and non-negative")
    candidate_indices = np.flatnonzero(core_scores > 0.0)
    if candidate_indices.size == 0:
        raise ValueError("at least one candidate target is required")
    if evidence_rows.ndim != 2 or evidence_rows.shape[1] != 3 or evidence_rows.shape[0] == 0:
        raise ValueError("evidence_rows must be a non-empty matrix with three columns")
    if not np.all(np.isfinite(evidence_rows)) or np.any(evidence_rows[:, 2] < 0.0):
        raise ValueError("evidence rows must be finite with non-negative weights")
    identifiers = evidence_rows[:, :2]
    if not np.all(identifiers == np.floor(identifiers)) or np.any(identifiers < 0.0):
        raise ValueError("drug and gene identifiers must be non-negative integers")
    drug_indices = identifiers[:, 0].astype(int)
    gene_indices = identifiers[:, 1].astype(int)
    if np.any(gene_indices >= core_scores.size):
        raise ValueError("gene identifiers must index core_scores")
    if np.any(core_scores[gene_indices] <= 0.0):
        raise ValueError("every evidence gene must be a candidate target")
    unique_drugs = np.unique(drug_indices)
    if not np.array_equal(unique_drugs, np.arange(unique_drugs[-1] + 1)):
        raise ValueError("drug identifiers must be contiguous from zero")

    candidate_order = np.lexsort(
        (candidate_indices, -core_scores[candidate_indices])
    )
    ordered_genes = candidate_indices[candidate_order]
    gene_ranks = np.zeros(core_scores.size, dtype=int)
    gene_ranks[ordered_genes] = np.arange(1, ordered_genes.size + 1)
    gene_weights = np.zeros(core_scores.size, dtype=float)
    gene_weights[ordered_genes] = (
        1.0 - gene_ranks[ordered_genes] / ordered_genes.size + 1e-6
    )

    n_drugs = unique_drugs.size
    scores = np.zeros(n_drugs, dtype=float)
    target_counts = np.zeros(n_drugs, dtype=int)
    best_ranks = np.zeros(n_drugs, dtype=int)
    for drug_index in range(n_drugs):
        rows = drug_indices == drug_index
        genes = gene_indices[rows]
        scores[drug_index] = np.sum(
            gene_weights[genes] * evidence_rows[rows, 2]
        )
        target_counts[drug_index] = np.unique(genes).size
        best_ranks[drug_index] = int(gene_ranks[genes].min())
    ranking_order = np.lexsort(
        (unique_drugs, best_ranks, -target_counts, -scores)
    )
    drug_ranks = np.empty(n_drugs, dtype=int)
    drug_ranks[ranking_order] = np.arange(1, n_drugs + 1)
    table = np.column_stack(
        [unique_drugs, scores, target_counts, best_ranks, drug_ranks]
    ).astype(float)
    return table[ranking_order]

import numpy as np  # noqa: E402



def run_full_pipeline(
    cohort_seed: int = 20260328,
    association_seed: int = 271828,
    n_permutations: int = 255,
) -> float:
    """Reference implementation chaining every earlier step."""
    _BASE_ALLELE_PROBABILITY = np.array([0.15, 0.25, 0.35, 0.55, 0.40, 0.20])
    _CASE_PROBABILITY_SHIFT = np.array([0.55, 0.35, 0.25, -0.35, 0.00, 0.15])
    _EXPRESSION_WEIGHTS = np.array(
        [
            [
                [1.2, 0.0, 0.5, 0.0],
                [0.8, 0.1, -0.6, 0.0],
                [0.0, 1.0, 0.4, 0.0],
                [0.0, -0.8, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.9],
                [0.1, 0.2, -0.3, -0.8],
            ],
            [
                [1.0, 0.1, -0.4, 0.0],
                [0.6, 0.0, 0.8, 0.1],
                [0.2, 0.9, -0.5, 0.0],
                [0.0, -0.7, 0.6, 0.0],
                [0.0, 0.0, 0.0, 0.8],
                [0.2, 0.3, 0.2, -0.7],
            ],
        ],
        dtype=float,
    )
    _EXPRESSION_INTERCEPTS = np.array(
        [[0.2, -0.1, 0.0, 0.3], [-0.2, 0.15, 0.1, -0.1]], dtype=float
    )
    _TERM_FDR = np.array(
        [
            [0.008, 0.04, 0.12, 0.03, 0.50, 0.06],
            [0.03, 0.20, 0.04, 0.07, 0.08, 0.09],
            [0.20, 0.40, 0.50, 0.30, 0.60, 0.70],
        ],
        dtype=float,
    )
    _TERM_GENE_MEMBERSHIP = np.array(
        [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8
    )
    _DRUGGABILITY = np.array([0.2, 0.5, 0.9, 0.1], dtype=float)
    _HUB_SCORES = np.array([0.4, 0.1, 0.8, 0.6], dtype=float)
    _DRUG_EVIDENCE = np.array(
        [
            [0, 1, 4.0],
            [0, 0, 0.8],
            [1, 2, 4.0],
            [1, 0, 3.0],
            [2, 1, 1.2],
            [2, 2, 1.0],
            [3, 0, 6.0],
            [3, 1, 0.5],
        ],
        dtype=float,
    )
    if not isinstance(cohort_seed, (int, np.integer)):
        raise ValueError("cohort_seed must be an integer")
    if not isinstance(association_seed, (int, np.integer)):
        raise ValueError("association_seed must be an integer")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")

    n_samples = 24
    labels = np.tile(np.array([0, 1], dtype=np.uint8), 12)
    probabilities = (
        _BASE_ALLELE_PROBABILITY[None, :]
        + labels[:, None] * _CASE_PROBABILITY_SHIFT[None, :]
    )
    rng = np.random.default_rng(int(cohort_seed))
    genotypes = rng.binomial(2, probabilities).astype(float)
    sex = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 6)
    pc1 = np.linspace(-1.15, 1.15, n_samples)
    covariates = np.column_stack([sex, pc1])
    split_ids = np.array([0] * 16 + [1] * 8, dtype=np.uint8)
    training_mask = (split_ids == 0).astype(np.uint8)

    predicted = impute_predicted_expression(  # noqa: F821
        genotypes, _EXPRESSION_WEIGHTS, _EXPRESSION_INTERCEPTS
    )
    adjusted = residualize_training_covariates(  # noqa: F821
        predicted, covariates, training_mask
    )
    effects, p_values = compute_split_associations(  # noqa: F821
        adjusted,
        labels,
        split_ids,
        int(n_permutations),
        int(association_seed),
    )
    fdr_values, significant = adjust_discovery_records(  # noqa: F821
        effects, p_values, 0.1, 0.5
    )
    gene_scores = score_reproducible_genes(  # noqa: F821
        effects,
        fdr_values,
        significant,
        2.0,
        0.5,
        3.0,
        0.4,
    )
    _, pathway_scores = propagate_pathway_support(  # noqa: F821
        _TERM_FDR, _TERM_GENE_MEMBERSHIP, 0.1
    )
    integrated = integrate_target_evidence(  # noqa: F821
        gene_scores[:, 3], pathway_scores, _DRUGGABILITY, _HUB_SCORES
    )
    drug_ranking = aggregate_drug_evidence(  # noqa: F821
        integrated[:, 4], _DRUG_EVIDENCE
    )
    return float(drug_ranking[0, 1])
SCICODE_GOLD_EOF
