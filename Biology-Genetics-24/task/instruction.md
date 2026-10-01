# Biology-Genetics-24

## Background

Human genetic data can nominate therapeutic targets even when disease-relevant expression measurements are missing. Transcriptome imputation supplies a genetically regulated expression layer, but linkage disequilibrium, co-regulation, tissue choice, and model architecture make any isolated association uncertain.

Genotype-first prioritization addresses this uncertainty by seeking support that recurs across expression resources and statistical analyses. It then combines genetic discovery evidence with pathway coherence, network context, and target tractability before mapping genes to compounds.

The resulting quantities are rankings for experimental follow-up, not causal claims or clinical recommendations. Such frameworks are evaluated through held-out gene replication, enrichment for curated disease biology, robustness to score weights, and recovery of pharmacologically supported compounds.

## Problem

Inherited variation can support therapeutic hypothesis generation when disease-matched transcriptomics are unavailable, but a single transcriptome imputation resource or association test is vulnerable to tissue, model, and sampling effects. A recent genotype-first framework instead ranks genes by convergent training and validation evidence, adds pathway, druggability, and network context, and carries the resulting target ranks into drug evidence aggregation.

Use the recent framework to determine the highest-ranked drug for the deterministic synthetic instance below. Apply its published default discovery and target-integration definitions, together with the specified numerical conventions where the source leaves an implementation choice open.

The configuration is:

- `n_samples = 24`, `cohort_seed = 20260328`, `association_seed = 271828`, and `B = 255` label permutations per split.
- `y = tile([0, 1], 12)`, `base_allele_probability = [0.15, 0.25, 0.35, 0.55, 0.40, 0.20]`, `case_probability_shift = [0.55, 0.35, 0.25, -0.35, 0.00, 0.15]`, and `X = default_rng(cohort_seed).binomial(2, base_allele_probability + y[:, None] * case_probability_shift)`.
- `sex = tile([0, 0, 1, 1], 6)`, `PC1 = linspace(-1.15, 1.15, 24)`, and the covariate matrix contains `sex` then `PC1`, without an intercept column.
- The samples indexed `0` through `15` form training, and those indexed `16` through `23` form validation.
- Frozen expression weights, indexed by resource, SNP, then gene, are `W = [[[1.2,0.0,0.5,0.0],[0.8,0.1,-0.6,0.0],[0.0,1.0,0.4,0.0],[0.0,-0.8,0.2,0.0],[0.0,0.0,0.0,0.9],[0.1,0.2,-0.3,-0.8]],[[1.0,0.1,-0.4,0.0],[0.6,0.0,0.8,0.1],[0.2,0.9,-0.5,0.0],[0.0,-0.7,0.6,0.0],[0.0,0.0,0.0,0.8],[0.2,0.3,0.2,-0.7]]]` and `alpha = [[0.2,-0.1,0.0,0.3],[-0.2,0.15,0.1,-0.1]]`.
- The two association methods are the case-control mean difference followed by the case-control median difference. Create one RNG from `association_seed`; for training then validation, draw `B` random permutations of that split's labels and reuse those labelings for every resource, method, and gene. Use the two-sided empirical value `(1 + count(|T_perm| + 1e-12 >= |T_obs|))/(B + 1)`.
- Apply Benjamini-Hochberg adjustment across the four genes separately within each split-resource-method slice. A record is significant when `FDR < 0.1` and `|effect| >= 0.5`, with training and validation records forming the discovery evidence and the split dimension excluded from breadth counting.
- For effect standardization within each method, use the median and NumPy's default linear 0.25 and 0.75 quantiles over significant signed effects and divide by `max(IQR, 1e-6)`. For the paper's otherwise unspecified smooth maps, use `f_mean(z) = 1/(1 + exp(-0.5*(z - 2.0)))` and `f_max(z) = 1/(1 + exp(-0.4*(z - 3.0)))`.
- The pathway scan FDR values are `term_fdr = [[0.008,0.04,0.12,0.03,0.50,0.06],[0.03,0.20,0.04,0.07,0.08,0.09],[0.20,0.40,0.50,0.30,0.60,0.70]]`, the strict pathway threshold is `0.1`, and term-gene membership is `[[1,0,1,0],[0,1,1,0],[0,0,0,1]]`; when a term recurs, use its minimum significant FDR for its strength.
- The candidate-gene inputs are `druggability = [0.2,0.5,0.9,0.1]` and `hub = [0.4,0.1,0.8,0.6]`. Percentile-normalize each integrated component among positive-discovery genes only as `(average_ascending_rank - 1)/(n_candidates - 1)`, assigning a sole candidate percentile one; non-discovery genes remain zero.
- The drug evidence rows `[drug_index, gene_index, combined_evidence_weight]` are `[[0,1,4.0],[0,0,0.8],[1,2,4.0],[1,0,3.0],[2,1,1.2],[2,2,1.0],[3,0,6.0],[3,1,0.5]]`. Order positive-discovery genes by decreasing integrated score with smaller gene index breaking ties, set `GeneWeight = 1 - gene_rank/n_candidates + 1e-6`, and sum `GeneWeight * combined_evidence_weight` over all rows for each drug; break drug-score ties by more distinct targets, better best-gene rank, then smaller drug index.

Your final answer must be a single number: the `DrugScore` of the highest-ranked synthetic drug at full precision.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_impute_predicted_expression

Goal
----
Genetically predicted expression is the fixed-weight linear projection of allelic dosages. For individual i, resource r, and gene g, E_hat[i,r,g] = alpha[r,g] + sum_j X[i,j] * W[r,j,g], where X contains dosage values from zero to two, W contains frozen cis-SNP weights, and alpha is the resource-specific intercept.

Inputs

------

genotypes: Float array of shape (n_samples, n_snps).

weights: Float array of shape (n_resources, n_snps, n_genes).

intercepts: Float array of shape (n_resources, n_genes).

Returns

-------

predicted_expression: Float array of shape (n_samples, n_resources, n_genes).

```python
def impute_predicted_expression(
    genotypes: "np.ndarray",
    weights: "np.ndarray",
    intercepts: "np.ndarray",
) -> "np.ndarray":
    """Project genotype dosages through frozen expression-weight models.

    Parameters
    ----------
    genotypes : np.ndarray
        Dosage matrix with shape (n_samples, n_snps) and entries in [0, 2].
    weights : np.ndarray
        Frozen cis-SNP weights with shape (n_resources, n_snps, n_genes).
    intercepts : np.ndarray
        Resource-gene intercepts with shape (n_resources, n_genes).

    Raises
    ------
    ValueError
        If an input has the wrong dimensionality, the SNP or resource-gene
        dimensions do not agree, an input is non-finite, or a dosage lies
        outside [0, 2].

    Returns
    -------
    predicted_expression : np.ndarray
        Predicted expression with shape (n_samples, n_resources, n_genes).
    """
    return predicted_expression  # noqa: F821
```

### Step 2

02_residualize_training_covariates

Goal
----
Covariate adjustment must be learned without validation leakage. For every resource-gene column, ordinary least squares fits an intercept and the supplied covariates on training samples only, applies those coefficients to every sample, subtracts the fitted covariate contribution, and adds back the unadjusted training mean so the expression scale is retained.

Inputs

------

predicted_expression: Float array of shape (n_samples, n_resources, n_genes).

covariates: Float array of shape (n_samples, n_covariates), without an intercept column.

training_mask: Binary array of shape (n_samples,).

Returns

-------

adjusted_expression: Float array with the same shape as predicted_expression.

```python
def residualize_training_covariates(
    predicted_expression: "np.ndarray",
    covariates: "np.ndarray",
    training_mask: "np.ndarray",
) -> "np.ndarray":
    """Residualize predicted expression using training-fitted covariates.

    Parameters
    ----------
    predicted_expression : np.ndarray
        Predicted expression with shape (n_samples, n_resources, n_genes).
    covariates : np.ndarray
        Covariate matrix with shape (n_samples, n_covariates), excluding the
        regression intercept.
    training_mask : np.ndarray
        Binary vector selecting the samples used to fit the regression.

    Raises
    ------
    ValueError
        If dimensions or sample counts disagree, inputs are non-finite,
        `training_mask` is not binary, too few training rows are selected, or
        the training design containing an intercept is rank deficient.

    Returns
    -------
    adjusted_expression : np.ndarray
        Covariate-adjusted expression with the same shape as the input.
    """
    return adjusted_expression  # noqa: F821
```

### Step 3

03_compute_split_associations

Goal
----
Training and validation evidence are computed separately across frozen expression resources. Two differential-expression summaries are used for each resource-gene pair: the case-control mean difference and the case-control median difference. Their two-sided empirical p-values use one seeded stream of label permutations per split, reused for every resource, method, and gene, with the finite-permutation correction p = (1 + c) / (B + 1).

Inputs

------

adjusted_expression: Float array of shape (n_samples, n_resources, n_genes).

labels: Binary phenotype array of shape (n_samples,).

split_ids: Integer array of shape (n_samples,), with training coded 0 and validation coded 1.

n_permutations: Positive integer B.

seed: Integer seed for the permutation generator.

Returns

-------

effects: Float array of shape (2, n_resources, 2, n_genes), with mean then median methods.

p_values: Float array with the same shape as effects.

```python
def compute_split_associations(
    adjusted_expression: "np.ndarray",
    labels: "np.ndarray",
    split_ids: "np.ndarray",
    n_permutations: int,
    seed: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute split-specific mean and median permutation associations.

    Parameters
    ----------
    adjusted_expression : np.ndarray
        Adjusted values with shape (n_samples, n_resources, n_genes).
    labels : np.ndarray
        Binary case-control labels with one entry per sample.
    split_ids : np.ndarray
        Split identifiers, zero for training and one for validation.
    n_permutations : int
        Positive number of random label permutations per split.
    seed : int
        Seed for a single NumPy random generator.

    Raises
    ------
    ValueError
        If array dimensions or sample counts disagree, values are non-finite,
        labels are not binary, split identifiers are not exactly zero or one,
        either split lacks a case or control, `n_permutations` is not a
        positive integer, or `seed` is not an integer.

    Returns
    -------
    effects : np.ndarray
        Mean and median case-control differences with shape
        (2, n_resources, 2, n_genes).
    p_values : np.ndarray
        Corrected empirical two-sided p-values with the same shape.
    """
    return association_results  # noqa: F821
```

### Step 4

04_adjust_discovery_records

Goal
----
Discovery records are selected without using held-out data. Within each split, expression resource, and association method, gene-wise p-values are adjusted by the Benjamini-Hochberg step-up procedure. A record is retained only when its adjusted value is strictly below the FDR threshold and its absolute effect meets the effect threshold.

Inputs

------

effects: Float array of shape (n_splits, n_resources, n_methods, n_genes).

p_values: Float array with the same shape.

fdr_threshold: Threshold in (0, 1].

effect_threshold: Non-negative absolute-effect threshold.

Returns

-------

fdr_values: Float array with the same shape as effects.

significant: Binary uint8 array with the same shape.

```python
def adjust_discovery_records(
    effects: "np.ndarray",
    p_values: "np.ndarray",
    fdr_threshold: float,
    effect_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Adjust gene-wise p-values and identify discovery records.

    Parameters
    ----------
    effects : np.ndarray
        Effect array shaped (n_splits, n_resources, n_methods, n_genes).
    p_values : np.ndarray
        Nominal p-values with the same shape as `effects`.
    fdr_threshold : float
        Positive FDR cutoff no greater than one; comparison is strict.
    effect_threshold : float
        Finite non-negative threshold for the absolute effect.

    Raises
    ------
    ValueError
        If the arrays are not matching four-dimensional non-empty arrays,
        effects are non-finite, p-values lie outside [0, 1],
        `fdr_threshold` is not in (0, 1], or `effect_threshold` is negative or
        non-finite.

    Returns
    -------
    fdr_values : np.ndarray
        Benjamini-Hochberg adjusted values with the input shape.
    significant : np.ndarray
        Binary uint8 indicators of retained discovery records.
    """
    return discovery_results  # noqa: F821
```

### Step 5

05_score_reproducible_genes

Goal
----
Discovery genes are prioritized by convergent evidence rather than one test. Reproducibility is R_g = 0.6 hits_g/max(hits) + 0.4 breadth_g/max(breadth), where breadth counts distinct resource-method combinations. Within each method, significant signed effects are centered by their median and scaled by max(IQR, 1e-6); their absolute standardized values enter E_g = 0.7 f_mean(mean z) + 0.3 f_max(max z), with a factor 1.1 for one-signed support. Confidence is C_g = 0.6(1 - min FDR) + 0.4(1 - mean FDR), and the discovery score is S_g = 0.4 R_g + 0.3 E_g + 0.3 C_g. Non-discovery genes receive zeros.

Inputs

------

effects: Float array of shape (n_splits, n_resources, n_methods, n_genes).

fdr_values: Float array with the same shape.

significant: Binary array with the same shape.

Returns

-------

gene_scores: Float array of shape (n_genes, 4), with columns R_g, E_g, C_g, and S_g.

```python
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
```

### Step 6

06_propagate_pathway_support

Goal
----
Pathway evidence is propagated from recurrent enrichment rather than from one foreground cutoff. For term t, significant scans have FDR below the threshold, robustness(t) is their count, strength(t) = -log10 of their minimum FDR, and weight(t) = strength(t) * robustness(t). A gene's PathwayScore is the sum of the weights for significant terms containing that gene; unsupported terms and genes contribute zero.

Inputs

------

term_fdr: Float array of shape (n_terms, n_scans).

term_gene_membership: Binary array of shape (n_terms, n_genes).

fdr_threshold: Threshold in (0, 1).

Returns

-------

term_summary: Float array of shape (n_terms, 3), containing robustness, strength, and weight.

pathway_scores: Float array of shape (n_genes,).

```python
def propagate_pathway_support(
    term_fdr: "np.ndarray",
    term_gene_membership: "np.ndarray",
    fdr_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Propagate recurrent pathway-enrichment evidence to genes.

    Parameters
    ----------
    term_fdr : np.ndarray
        Adjusted pathway values with shape (n_terms, n_scans).
    term_gene_membership : np.ndarray
        Binary term-gene membership matrix with shape (n_terms, n_genes).
    fdr_threshold : float
        Finite strict significance threshold in (0, 1).

    Raises
    ------
    ValueError
        If arrays are not non-empty two-dimensional arrays with the same term
        count, FDR values are non-finite or outside [0, 1], membership is not
        binary, or `fdr_threshold` is not finite and in (0, 1).

    Returns
    -------
    term_summary : np.ndarray
        Per-term robustness, strength, and weight columns.
    pathway_scores : np.ndarray
        Sum of recurrent significant term weights for each gene.
    """
    return pathway_results  # noqa: F821
```

### Step 7

07_integrate_target_evidence

Goal
----
Integrated target prioritization balances the genotype-derived discovery signal with biological and translational evidence. Each component is percentile-normalized among discovery genes by mapping the smallest average rank to zero and the largest to one, with tied values assigned their average rank. The published operating point is core_score_g = 0.45 DE_g + 0.25 Path_g + 0.25 Drug_g + 0.05 Hub_g; non-discovery genes remain zero.

Inputs

------

discovery_scores: Non-negative float array of shape (n_genes,), with zero marking non-discovery genes.

pathway_scores: Non-negative float array of shape (n_genes,).

druggability_scores: Non-negative float array of shape (n_genes,).

hub_scores: Non-negative float array of shape (n_genes,).

Returns

-------

integrated_scores: Float array of shape (n_genes, 5), with four percentile components and core_score.

```python
def integrate_target_evidence(
    discovery_scores: "np.ndarray",
    pathway_scores: "np.ndarray",
    druggability_scores: "np.ndarray",
    hub_scores: "np.ndarray",
) -> "np.ndarray":
    """Percentile-normalize evidence and compute integrated target scores.

    Parameters
    ----------
    discovery_scores : np.ndarray
        Non-negative discovery scores; positive entries define candidates.
    pathway_scores : np.ndarray
        Non-negative pathway support for the same genes.
    druggability_scores : np.ndarray
        Non-negative druggability evidence for the same genes.
    hub_scores : np.ndarray
        Non-negative network hub evidence for the same genes.

    Raises
    ------
    ValueError
        If inputs are not matching non-empty one-dimensional arrays, contain
        non-finite or negative values, or no positive discovery score exists.

    Returns
    -------
    integrated_scores : np.ndarray
        Columns DE_g, Path_g, Drug_g, Hub_g, and core_score_g, with zero rows
        for non-discovery genes.
    """
    return integrated_scores  # noqa: F821
```

### Step 8

08_aggregate_drug_evidence

Goal
----
Drug hypotheses aggregate evidence over ranked discovery genes. Candidate genes are ordered by decreasing core_score with smaller gene index breaking ties, and rank r among n candidates receives GeneWeight = 1 - r/n + 1e-6. Each evidence row contributes GeneWeight times its supplied clinical-stage and source weight. Drugs are ordered by summed DrugScore, then by more distinct targets, better best-gene rank, and smaller drug index.

Inputs

------

core_scores: Non-negative float array of shape (n_genes,), with positive values marking candidates.

evidence_rows: Float array of shape (n_rows, 3), holding drug index, gene index, and evidence weight.

Returns

-------

drug_ranking: Float array of shape (n_drugs, 5), holding drug index, DrugScore, target count, best gene rank, and drug rank.

```python
def aggregate_drug_evidence(
    core_scores: "np.ndarray",
    evidence_rows: "np.ndarray",
) -> "np.ndarray":
    """Rank drugs by rank-weighted target evidence.

    Parameters
    ----------
    core_scores : np.ndarray
        Non-negative integrated target scores; positive entries are candidates.
    evidence_rows : np.ndarray
        Rows containing integer drug index, integer candidate-gene index, and
        a finite non-negative evidence weight. Drug indices must be contiguous
        from zero.

    Raises
    ------
    ValueError
        If `core_scores` is not a non-empty finite non-negative vector, no
        candidate exists, `evidence_rows` is not a non-empty matrix with three
        columns, identifiers are not integers in range, drug identifiers are
        not contiguous from zero, an evidence gene is not a candidate, or an
        evidence weight is non-finite or negative.

    Returns
    -------
    drug_ranking : np.ndarray
        Rows containing drug index, DrugScore, distinct-target count, best
        gene rank, and one-based drug rank.
    """
    return drug_ranking  # noqa: F821
```

### Step 9

09_run_full_pipeline

Goal
----
The genotype-first prioritization chain begins with a seeded synthetic dosage cohort, projects dosages through two frozen expression resources, removes covariates using training-fitted coefficients, and combines training and validation association records without held-out leakage. Reproducibility-aware discovery scores receive recurrent pathway, druggability, and network evidence before candidate gene ranks weight the supplied drug-target rows. The reported scalar is the DrugScore in the first row of the final drug ranking.

Inputs

------

cohort_seed: Integer seed for genotype generation.

association_seed: Integer seed for label permutations.

n_permutations: Positive number of label permutations per split.

Returns

-------

top_drug_score: Float DrugScore of the highest-ranked synthetic compound.

```python
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
```
