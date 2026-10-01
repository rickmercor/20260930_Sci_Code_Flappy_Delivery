"""
Discovery genes are prioritized by convergent evidence rather than one test. Reproducibility is R_g = 0.6 hits_g/max(hits) + 0.4 breadth_g/max(breadth), where breadth counts distinct resource-method combinations. Within each method, significant signed effects are centered by their median and scaled by max(IQR, 1e-6); their absolute standardized values enter E_g = 0.7 f_mean(mean z) + 0.3 f_max(max z), with a factor 1.1 for one-signed support. Confidence is C_g = 0.6(1 - min FDR) + 0.4(1 - mean FDR), and the discovery score is S_g = 0.4 R_g + 0.3 E_g + 0.3 C_g. Non-discovery genes receive zeros.

Inputs

------

effects: Float array of shape (n_splits, n_resources, n_methods, n_genes).

fdr_values: Float array with the same shape.

significant: Binary array with the same shape.

Returns

-------

gene_scores: Float array of shape (n_genes, 4), with columns R_g, E_g, C_g, and S_g.

Returns
-------
np.ndarray of shape (n_genes, 4), the reproducibility, effect, confidence, and discovery scores
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def score_reproducible_genes(
    effects: "np.ndarray",
    fdr_values: "np.ndarray",
    significant: "np.ndarray",
    mean_midpoint: float,
    mean_steepness: float,
    max_midpoint: float,
    max_steepness: float,
) -> "np.ndarray":
    """Form reproducibility-aware gene discovery scores.

    Parameters
    ----------
    effects : np.ndarray
        Signed effects shaped (n_splits, n_resources, n_methods, n_genes).
    fdr_values : np.ndarray
        Adjusted values with the same shape as `effects`.
    significant : np.ndarray
        Binary retained-record indicators with the same shape.
    mean_midpoint : float
        Midpoint of the logistic map for a gene's mean standardized effect.
    mean_steepness : float
        Positive steepness of the mean-effect logistic map.
    max_midpoint : float
        Midpoint of the logistic map for a gene's maximum standardized effect.
    max_steepness : float
        Positive steepness of the maximum-effect logistic map.

    Raises
    ------
    ValueError
        If arrays are not matching non-empty four-dimensional arrays, effects
        or FDR values are invalid, `significant` is not binary, no discovery
        record is present, a midpoint is non-finite, or a steepness is not
        finite and positive.

    Returns
    -------
    gene_scores : np.ndarray
        Per-gene columns R_g, E_g, C_g, and S_g.
    """
    return gene_scores  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _logistic_map(value, midpoint, steepness):
    """Map a non-negative standardized effect smoothly to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))


def _oracle_score_reproducible_genes(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
effects = np.array([[[[1.0, -2.0, 0.1], [1.5, -1.0, 0.2]], [[2.0, -0.5, 0.3], [2.5, -1.5, 0.4]]], [[[0.8, -2.2, 0.0], [1.2, -0.8, 0.0]], [[1.8, -0.4, 0.0], [2.2, -1.2, 0.0]]]])
fdr = np.full_like(effects, 0.05)
sig = np.zeros_like(effects, dtype=np.uint8)
sig[:, :, :, 0] = 1
sig[0, 0, :, 1] = 1
""",
            "call": "score_reproducible_genes(effects, fdr, sig, 2.0, 0.5, 3.0, 0.4).tolist()",
            "gold_call": "_oracle_score_reproducible_genes(effects, fdr, sig, 2.0, 0.5, 3.0, 0.4).tolist()",
        },
        {
            "setup": """import numpy as np
effects = np.array([[[[2.0]]]])
fdr = np.array([[[[0.0]]]])
sig = np.array([[[[1]]]], dtype=np.uint8)
""",
            "call": "score_reproducible_genes(effects, fdr, sig, 0.0, 1.0, 0.0, 1.0).tolist()",
            "gold_call": "_oracle_score_reproducible_genes(effects, fdr, sig, 0.0, 1.0, 0.0, 1.0).tolist()",
        },
        {
            "setup": """import numpy as np
effects = np.zeros((1, 1, 1, 2))
fdr = np.ones_like(effects)
sig = np.zeros_like(effects, dtype=np.uint8)
def run_model():
    try:
        score_reproducible_genes(effects, fdr, sig, 2.0, 0.5, 3.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_score_reproducible_genes(effects, fdr, sig, 2.0, 0.5, 3.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
