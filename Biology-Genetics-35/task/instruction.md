# Biology-Genetics-35

## Background

Colocalisation analysis distinguishes whether association signals for different traits are best explained by a shared causal variant or by distinct causal variants. Fine-mapped signals can be represented by variant-level log Bayes factors, while variant availability may differ between studies and between individual signals.

At catalogue scale, supported pairwise relationships can be represented as a graph whose connected structure summarizes relationships among signals. Signal metadata and credible-set annotations provide complementary information for evaluating the biological coherence and robustness of that structure.

## Problem

A consortium has supplied signal-level log Bayes factors from two association-result collections, together with signal identities and credible-set status. Use recent human-genetics literature to identify the scalable signal-level colocalisation treatment matching this representation and recover and apply the source-defined stages needed for this graph-sensitivity endpoint. Treat all supplied signals as already assigned to one overlapping genomic region; source genomic sorting and region grouping have therefore already been completed. Evaluate every unordered pair of retained signals, construct the non-isolated colocalisation-component graph, and repeat the component analysis after retaining only relationships whose two endpoint signals both have a credible set.
Return the signed change, in percentage points, in the fraction of non-isolated components classified as colliders:
restricted credible-set graph minus full graph.
Numerical instance
- Scope: signals S0 through S23; variants V0 through V16
- Table fields: collection, every non-baseline tested entry, signal-specific unavailable entries, and credible-set status
- Baseline: every unspecified tested signal-variant entry has log Bayes factor -12
- Availability: an entry listed as unavailable is not an observed baseline entry
- Collection coverage: C0 tests all variants; C1 does not test V2
- Logarithms: natural
| Signal | Collection | Non-baseline log Bayes factors | Additional unavailable variants | Credible set |
|---|---|---|---|---|
| S0 | C1 | V0: 18 | none | yes |
| S1 | C0 | V12: 18+ln(0.20), V8: 18+ln(0.25), V1: 18+ln(0.55) | V15 | yes |
| S2 | C0 | V9: 18, V2: 18+ln(1.4) | none | yes |
| S3 | C1 | V14: 18 | none | yes |
| S4 | C1 | V11: 4.9, V4: 4.9 | none | yes |
| S5 | C0 | V3: 18 | none | yes |
| S6 | C0 | V15: 18 | none | no |
| S7 | C1 | V10: 18+ln(0.5), V5: 18+ln(0.5) | none | yes |
| S8 | C1 | V13: 24, V6: 14 | none | yes |
| S9 | C1 | V4: 18 | V5 | yes |
| S10 | C1 | V7: 18+ln(0.5), V16: 18+ln(0.5) | none | yes |
| S11 | C1 | V1: 18 | none | yes |
| S12 | C0 | V11: 18 | none | yes |
| S13 | C1 | V3: 18+ln(0.55), V12: 18+ln(0.25), V8: 18+ln(0.20) | none | yes |
| S14 | C0 | V13: 14, V6: 24 | none | yes |
| S15 | C0 | V0: 18 | V8 | yes |
| S16 | C0 | V7: 18 | none | no |
| S17 | C1 | V9: 18 | none | yes |
| S18 | C0 | V5: 18+ln(0.5), V14: 18+ln(0.5) | V0 | yes |
| S19 | C1 | V11: 18 | none | no |
| S20 | C0 | V10: 18 | none | yes |
| S21 | C0 | V4: 18 | none | no |
| S22 | C0 | V16: 18 | none | yes |
| S23 | C1 | V15: 18 | none | yes |
Each retained signal is a graph node carrying `(dataset ID, trait ID, signal ID)`; signal Si has signal ID `5000+i` and, except for the repeated identities below, dataset ID `100+i` and trait ID `200+i`.
| Signals | Dataset ID | Trait ID |
|---|---:|---:|
| S5, S11 | 10 | 1000 |
| S3, S20 | 20 | 2000 |
| S16, S22 | 30 | 3000 |
| S9, S19 | 40 | 4000 |
| S0, S15 | 50 | 5000 |
- Identity integrity: all complete triplets are distinct
- Prior coefficients: p1 = 0.0001, p2 = 0.0001, p12 = 0.00001
- Inclusive decision thresholds: reciprocal overlap = 0.5; normalized H4 posterior = 0.8
- Arithmetic controls: IEEE-754 binary64, no intermediate rounding, lexicographic edge order
The peer-reviewed article governs the scientific definitions and decision rules; commit 8d23d6aed05de7596ae992ceb0be8b7682f95665 of the article's linked reference implementation governs executable details not fully determined by prose; and the supplied priors and arithmetic controls take precedence over source defaults.
In <reasoning>, identify the governing article; give a brief source-specific methodological justification that identifies the screen, the pair-eligibility logic, the H3-versus-H4 distinction, and the component diagnostic you applied; and report the retained-signal count, the full and restricted edge/component/collider counts, the two collider fractions, and the signed subtraction.
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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Screen eligible signals

Goal
----
Select article-eligible association signals before any pairwise analysis.

 

For this benchmark's deterministic reconciliation with the pinned active-trim code, a cell is available for screening only when its collection tests that variant, the supplied value is finite, and the value is greater than -100000. A signal is retained when the maximum available log Bayes factor is at least 5. Preserve original signal order. NaN denotes an unavailable cell; infinity is invalid.

```python
def screen_signal_indices(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
) -> "np.ndarray":
    """
    Return indices of signals meeting the source eligibility screen.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Two-dimensional signal-by-variant log Bayes-factor array.
    collection_id : np.ndarray
        One-dimensional zero-based collection identifier per signal.
    tested_variant_mask : np.ndarray
        Boolean collection-by-variant test-coverage array.
 
    Returns
    -------
    np.ndarray
        One-dimensional integer indices in original signal order.
 
    Raises
    ------
    ValueError
        If shapes, identifiers, dtypes, or finite-value requirements fail.
    """
    return result
```

### Step 2

Normalize observed and unavailable cells

Goal
----
Construct the benchmark-normalized retained signal matrix while preserving unavailable cells.

 

For each retained signal, a cell is observed only if the signal's collection tests the variant, the supplied value is finite, and the value is greater than -100000. Return two float planes: plane 0 contains observed log Bayes factors and -1000000 elsewhere; plane 1 contains the corresponding 1/0 observation indicator. Retained indices must be strictly increasing.

```python
def prepare_signal_matrix(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    retained_indices: "np.ndarray",
) -> "np.ndarray":
    """
    Build benchmark values and availability planes for retained signals.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Signal-by-variant log Bayes factors, with NaN for missing cells.
    collection_id : np.ndarray
        Zero-based collection identifier per signal.
    tested_variant_mask : np.ndarray
        Boolean collection-by-variant coverage mask.
    retained_indices : np.ndarray
        Strictly increasing retained signal indices.
 
    Returns
    -------
    np.ndarray
        Float array of shape (2, n_retained, n_variants).
 
    Raises
    ------
    ValueError
        If the aligned input contract is violated.
    """
    return result
```

### Step 3

Build shared collection variant masks

Goal
----
Derive collection-pair variant universes from retained signal availability.

 

A collection's universe contains every variant observed in at least one retained signal from that collection. The mask for an ordered collection pair is the intersection of the two collection universes. Collection identifiers must be contiguous from zero, and every collection must be represented.

```python
def collection_shared_masks(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
) -> "np.ndarray":
    """
    Return collection-pair intersections of collection-level variant unions.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Zero-based collection identifier per retained signal.
 
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_collections, n_collections, n_variants).
 
    Raises
    ------
    ValueError
        If planes, values, or collection identifiers are invalid.
    """
    return result
```

### Step 4

Compute reciprocal signal coverage

Goal
----
Compute directional Bayes-factor-mass coverage for every retained signal pair.

 

For direction i to j, the numerator is signal i's Bayes-factor mass over collection-shared variants observed in both signals. The denominator is signal i's mass over all variants observed in i. Use stable log-sum-exp. Store i-to-j and j-to-i in the last axis; a pair with no mutually observed shared variant has zero coverage in both directions.

```python
def reciprocal_signal_coverage(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
) -> "np.ndarray":
    """
    Compute both directional coverage fractions for all signal pairs.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Collection identifier per retained signal.
    shared_masks : np.ndarray
        Collection-pair shared-variant masks.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_signals, n_signals, 2).
 
    Raises
    ------
    ValueError
        If aligned shapes, masks, or observed values are invalid.
    """
    return result
```

### Step 5

Evaluate H0 through H4 evidence in batches

Goal
----
Evaluate source-matched H0-H4 log evidence for every signal pair.

 

Work on the collection-pair shared variant axis, including padded values. H0 has log evidence 0. H1 and H2 use each signal alone with prior p1 or p2. H3 uses prior p1*p2 and sums every ordered distinct-variant assignment. H4 uses prior p12 and sums equal-variant assignments. Use stable log-sum-exp throughout. chunk_size controls only outer signal-row blocking and cannot change results.

```python
def batched_hypothesis_evidence(
    prepared: "np.ndarray",
    retained_collection_id: "np.ndarray",
    shared_masks: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    chunk_size: int,
) -> "np.ndarray":
    """
    Return stable pairwise log evidence for hypotheses H0 through H4.
 
    Parameters
    ----------
    prepared : np.ndarray
        Two-plane retained signal matrix.
    retained_collection_id : np.ndarray
        Collection identifier per retained signal.
    shared_masks : np.ndarray
        Collection-pair shared-variant masks.
    p1, p2, p12 : float
        Positive finite prior coefficients.
    chunk_size : int
        Positive outer-row block size.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_signals, n_signals, 5).
 
    Raises
    ------
    ValueError
        If inputs, priors, or chunk size violate the contract.
    """
    return result
```

### Step 6

Select posterior-supported graph edges

Goal
----
Convert pairwise evidence into a canonically ordered colocalisation graph.

 

Normalize each pair's five log evidences by stable log-sum-exp. For i<j, keep an edge only when both directional coverage values are at least overlap_min and the normalized H4 posterior is at least h4_threshold. Return rows [i, j, PP.H4] in lexicographic endpoint order.

```python
def select_colocalisation_edges(
    log_evidence: "np.ndarray",
    coverage: "np.ndarray",
    overlap_min: float,
    h4_threshold: float,
) -> "np.ndarray":
    """
    Select eligible H4-supported signal-pair edges.
 
    Parameters
    ----------
    log_evidence : np.ndarray
        Pairwise H0-H4 log evidence with shape (n, n, 5).
    coverage : np.ndarray
        Directional coverage with shape (n, n, 2).
    overlap_min : float
        Inclusive minimum for both coverage directions.
    h4_threshold : float
        Inclusive normalized-H4 posterior threshold.
 
    Returns
    -------
    np.ndarray
        Float array of shape (n_edges, 3) containing [i, j, PP.H4].
 
    Raises
    ------
    ValueError
        If shapes, evidence, coverage, or thresholds are invalid.
    """
    return result
```

### Step 7

Apply the credible-set endpoint restriction

Goal
----
Apply the credible-set endpoint sensitivity restriction to graph edges.

 

Retain an edge exactly when both indexed endpoints have a credible set. Preserve the input row order and all three columns. Endpoint columns must be integer-valued, distinct, ordered as i<j, in range, and free of duplicates.

```python
def restrict_credible_set_edges(
    edges: "np.ndarray",
    has_credible_set: "np.ndarray",
) -> "np.ndarray":
    """
    Keep graph edges whose two endpoints have credible sets.
 
    Parameters
    ----------
    edges : np.ndarray
        Float edge rows [i, j, weight].
    has_credible_set : np.ndarray
        Boolean flag per graph node.
 
    Returns
    -------
    np.ndarray
        Filtered float array with shape (n_retained_edges, 3).
 
    Raises
    ------
    ValueError
        If the edge table or endpoint mask is invalid.
    """
    return result
```

### Step 8

Summarize identity-collider components

Goal
----
Summarize non-isolated connected components and their identity colliders.

 

Nodes are indexed by rows of signal_metadata. Its columns are [dataset ID, trait ID, signal ID], and complete triplets must be unique. A component is a collider exactly when two of its distinct signal nodes share the same ordered (dataset ID, trait ID) pair. Exclude isolated nodes. Return [minimum node index, node count, edge count, collider flag] sorted by minimum node index.

```python
def summarize_collider_components(
    edges: "np.ndarray",
    signal_metadata: "np.ndarray",
) -> "np.ndarray":
    """
    Return canonical summaries of non-isolated graph components.
 
    Parameters
    ----------
    edges : np.ndarray
        Float edge rows [i, j, weight].
    signal_metadata : np.ndarray
        Integer node rows [dataset ID, trait ID, signal ID].
 
    Returns
    -------
    np.ndarray
        Integer array with shape (n_components, 4).
 
    Raises
    ------
    ValueError
        If graph or metadata inputs are invalid.
    """
    return result
```

### Step 9

Compute the signed component-rate contrast

Goal
----
Compute the signed percentage-point change in component collider fraction.

 

Each component table has canonical rows [minimum node, node count, edge count, collider flag] in strictly increasing minimum-node order. Both tables must contain at least one component. Return 100 times the restricted mean collider flag minus 100 times the full mean collider flag. Do not return a relative percent change.

```python
def collider_rate_contrast(
    full_components: "np.ndarray",
    restricted_components: "np.ndarray",
) -> float:
    """
    Return restricted-minus-full collider fraction in percentage points.
 
    Parameters
    ----------
    full_components : np.ndarray
        Full-graph component summary with four integer columns.
    restricted_components : np.ndarray
        Restricted-graph component summary with four integer columns.
 
    Returns
    -------
    float
        Signed percentage-point contrast.
 
    Raises
    ------
    ValueError
        If either component table is empty or structurally invalid.
    """
    return result
```

### Step 10

Orchestrate the complete audit

Goal
----
Run the complete source-calibrated colocalisation and collider sensitivity audit.

 

Chain every preceding stage in order. Subset collection identifiers, metadata, and credible-set flags by the retained signal indices before graph analysis. Build the full components from all selected edges and the restricted components from edges whose two endpoints have credible sets. Return the signed restricted-minus-full collider fraction in percentage points.

```python
def evaluate_colocalisation_audit(
    log_bf: "np.ndarray",
    collection_id: "np.ndarray",
    tested_variant_mask: "np.ndarray",
    signal_metadata: "np.ndarray",
    has_credible_set: "np.ndarray",
    p1: float,
    p2: float,
    p12: float,
    overlap_min: float,
    h4_threshold: float,
    chunk_size: int,
) -> float:
    """
    Return the end-to-end credible-set collider-rate contrast.
 
    Parameters
    ----------
    log_bf : np.ndarray
        Signal-by-variant log Bayes factors.
    collection_id : np.ndarray
        Collection identifier per signal.
    tested_variant_mask : np.ndarray
        Collection-by-variant tested mask.
    signal_metadata : np.ndarray
        Integer rows [dataset ID, trait ID, signal ID].
    has_credible_set : np.ndarray
        Boolean credible-set flag per signal.
    p1, p2, p12 : float
        Positive prior coefficients.
    overlap_min, h4_threshold : float
        Inclusive pair eligibility thresholds.
    chunk_size : int
        Positive evidence row-block size.
 
    Returns
    -------
    float
        Restricted-minus-full collider fraction in percentage points.
 
    Raises
    ------
    ValueError
        If any stage input or resulting component table is invalid.
    """
    return result
```
