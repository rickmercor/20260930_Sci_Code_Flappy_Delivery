"""
The genotype-first prioritization chain begins with a seeded synthetic dosage cohort, projects dosages through two frozen expression resources, removes covariates using training-fitted coefficients, and combines training and validation association records without held-out leakage. Reproducibility-aware discovery scores receive recurrent pathway, druggability, and network evidence before candidate gene ranks weight the supplied drug-target rows. The reported scalar is the DrugScore in the first row of the final drug ranking.

Inputs

------

cohort_seed: Integer seed for genotype generation.

association_seed: Integer seed for label permutations.

n_permutations: Positive number of label permutations per split.

Returns

-------

top_drug_score: Float DrugScore of the highest-ranked synthetic compound.

Returns
-------
float, the highest-ranked synthetic compound's DrugScore as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    cohort_seed: int = 20260328,
    association_seed: int = 271828,
    n_permutations: int = 255,
) -> float:
    """Run the complete genotype-first target-to-drug prioritization chain.

    Parameters
    ----------
    cohort_seed : int
        Seed for conditional binomial genotype generation.
    association_seed : int
        Seed for split-specific phenotype-label permutations.
    n_permutations : int
        Positive number of random label permutations per split.

    Raises
    ------
    ValueError
        If either seed is not an integer or `n_permutations` is not a positive
        integer.

    Returns
    -------
    top_drug_score : float
        Rank-weighted evidence score of the highest-ranked drug.
    """
    return top_drug_score  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402



def _oracle_run_full_pipeline(
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

    predicted = _oracle_impute_predicted_expression(  # noqa: F821
        genotypes, _EXPRESSION_WEIGHTS, _EXPRESSION_INTERCEPTS
    )
    adjusted = _oracle_residualize_training_covariates(  # noqa: F821
        predicted, covariates, training_mask
    )
    effects, p_values = _oracle_compute_split_associations(  # noqa: F821
        adjusted,
        labels,
        split_ids,
        int(n_permutations),
        int(association_seed),
    )
    fdr_values, significant = _oracle_adjust_discovery_records(  # noqa: F821
        effects, p_values, 0.1, 0.5
    )
    gene_scores = _oracle_score_reproducible_genes(  # noqa: F821
        effects,
        fdr_values,
        significant,
        2.0,
        0.5,
        3.0,
        0.4,
    )
    _, pathway_scores = _oracle_propagate_pathway_support(  # noqa: F821
        _TERM_FDR, _TERM_GENE_MEMBERSHIP, 0.1
    )
    integrated = _oracle_integrate_target_evidence(  # noqa: F821
        gene_scores[:, 3], pathway_scores, _DRUGGABILITY, _HUB_SCORES
    )
    drug_ranking = _oracle_aggregate_drug_evidence(  # noqa: F821
        integrated[:, 4], _DRUG_EVIDENCE
    )
    return float(drug_ranking[0, 1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "cohort_seed = 20260328\nassociation_seed = 271828\nn_permutations = 255\n",
            "call": "float(run_full_pipeline(cohort_seed, association_seed, n_permutations))",
            "gold_call": "float(_oracle_run_full_pipeline(cohort_seed, association_seed, n_permutations))",
        },
        {
            "setup": "cohort_seed = 20260328\nassociation_seed = 0\nn_permutations = 63\n",
            "call": "float(run_full_pipeline(cohort_seed, association_seed, n_permutations))",
            "gold_call": "float(_oracle_run_full_pipeline(cohort_seed, association_seed, n_permutations))",
        },
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(cohort_seed="bad")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(cohort_seed="bad")
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
