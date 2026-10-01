# Chemistry-Computational_Chemistry-20

## Background

An assigned spectroscopic network is straightforward to invert because every measured transition already identifies its upper and lower states. An unassigned line list is harder because both the vertex energies and the incidence pattern are unknown.

The scan offset in this task perturbs the recurrence structure whenever compared line pairs contain different numbers of block-B measurements. It therefore has to be estimated before the topology is reconstructed. Once the topology has been recovered, the entire network supplies a stronger global calibration equation, so the seed offset is replaced by a joint weighted estimate.

An assigned transition network can be treated as an uncertainty-weighted linear inverse problem. In this task, the final joint fit combines all measured routes through the reconstructed network while refining the scan-registration offset together with the energy levels.

## Problem

An unassigned high-resolution absorption spectrum contains 19 distinct transitions from one isotopologue. Treat each transition as an edge of an unknown connected bipartite energy-level graph. Exactly one otherwise allowed transition between the two reconstructed level partitions is absent. Reconstruct the graph from recurring pairwise frequency differences and four-line closures, then perform an uncertainty-weighted spectroscopic-network inversion to determine the physical frequency of that missing transition.

For line $i$, the reported centre satisfies

$$\widetilde f_i=f_i+bI_i+\epsilon_i,$$

where $f_i$ is the physical transition frequency, $I_i=1$ for scan B and $I_i=0$ for scan A, $b$ is an unknown constant registration offset affecting scan B, and the independent error $\epsilon_i$ has supplied standard uncertainty $\sigma_i$. The instrument establishes the closed bracket

$$0.0046\le b\le0.0049\ \mathrm{cm}^{-1}.$$

Every listed line is a distinct transition. There are no contaminants or duplicate transitions.

The measured data are

$$
\begin{array}{c c c c}
\hline
\text{Line} & \text{Scan} & \widetilde f_i\;(\mathrm{cm}^{-1}) & \sigma_i\;(10^{-6}\,\mathrm{cm}^{-1}) \\
\hline
\mathrm{L01} & \mathrm{B} & 6291.6002896 & 2.0 \\
\mathrm{L02} & \mathrm{A} & 6305.3101133 & 1.6 \\
\mathrm{L03} & \mathrm{A} & 6309.9630221 & 1.4 \\
\mathrm{L04} & \mathrm{B} & 6314.7284247 & 1.8 \\
\mathrm{L05} & \mathrm{A} & 6321.1374199 & 1.4 \\
\mathrm{L06} & \mathrm{B} & 6323.6823052 & 1.9 \\
\mathrm{L07} & \mathrm{A} & 6333.0911534 & 1.7 \\
\mathrm{L08} & \mathrm{B} & 6338.3752148 & 2.1 \\
\mathrm{L09} & \mathrm{B} & 6339.5096148 & 1.5 \\
\mathrm{L10} & \mathrm{B} & 6352.0897687 & 1.5 \\
\mathrm{L11} & \mathrm{A} & 6361.4986205 & 1.6 \\
\mathrm{L12} & \mathrm{A} & 6367.9123463 & 1.8 \\
\mathrm{L13} & \mathrm{B} & 6374.7508896 & 1.6 \\
\mathrm{L14} & \mathrm{A} & 6397.8742901 & 1.9 \\
\mathrm{L15} & \mathrm{B} & 6404.2927505 & 1.7 \\
\mathrm{L16} & \mathrm{A} & 6422.2346684 & 2.0 \\
\mathrm{L17} & \mathrm{B} & 6435.9539540 & 1.7 \\
\mathrm{L18} & \mathrm{B} & 6445.3675344 & 1.8 \\
\mathrm{L19} & \mathrm{A} & 6451.7765297 & 1.5 \\
\hline
\end{array}
$$

Infer a topology-mining seed $b_{\mathrm{seed}}$ from recurring-difference closure. For every unordered pair of distinct lines, orient the pair from the larger to the smaller reported frequency. Let $d$ be its positive reported-frequency difference and let $q$ be the scan-B indicator of the higher-frequency line minus that of the lower-frequency line. The offset bracket is too narrow to reverse any such ordering.

For every unordered combination of two line-disjoint oriented pairs with unequal $q$, generate the value of $b$ that makes their two offset-corrected differences equal. Count each unordered pair-of-pairs once. Retain every candidate in the closed offset bracket, preserving repeated numerical values with their multiplicity, and define $b_{\mathrm{seed}}$ as the ordinary median of the retained candidates.

Use $b_{\mathrm{seed}}$ only to form the topology-discovery frequencies

$$f_i^{(0)}=\widetilde f_i-b_{\mathrm{seed}}I_i.$$

Form every unordered pairwise difference

$$\delta_{ij}=\left|f_i^{(0)}-f_j^{(0)}\right|.$$

Sort the occurrences by $(\delta_{ij},\min(i,j),\max(i,j))$. Starting from the smallest unassigned difference, place that occurrence and every following occurrence no more than $\Delta=8.0\times10^{-6}\ \mathrm{cm}^{-1}$ above the cluster’s first value into the same cluster; then begin the next cluster. Retain only clusters containing at least two pair occurrences.

Use these retained recurring-difference clusters and their compatible four-line closures to reconstruct the energy-level topology. Every spectral line must belong to exactly one recovered vertex in each of the two colour classes. Two-colour the connected recovered energy-level graph, call the larger colour class $U$ and the smaller colour class $L$, and identify the unique pair in $U\times L$ that is not joined by a listed transition.

Keeping this reconstructed topology fixed, jointly fit all energy-level values and $b$ to the original reported centres using

$$\widetilde f_i=E^U_{u(i)}-E^L_{\ell(i)}+bI_i+\epsilon_i$$

and ordinary inverse-variance-weighted linear least squares with the fixed weights

$$w_i=\sigma_i^{-2}.$$

Include all 19 lines. Do not reject lines, alter the supplied uncertainties, apply robust reweighting or hold $b$ fixed at $b_{\mathrm{seed}}$. Fix the additive energy gauge by setting to zero the level in $L$ whose sorted tuple of incident line IDs is lexicographically smallest.

Calculate the missing physical transition as the fitted upper-level energy minus the fitted lower-level energy. Do not add or subtract the fitted scan offset after forming this energy difference. Report the result in $\mathrm{cm}^{-1}$ rounded to seven digits after the decimal point.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

step_01_cluster_recurring_differences

Goal
----
Recover the topology-discovery offset and recurring-difference groups.

```python
def cluster_recurring_differences(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple[float, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Apply the deterministic first-value clustering rule to line differences.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi
        Non-empty aligned one-dimensional arrays of unique identifiers,
        scan labels A or B, and finite pairwise-distinct reported centres,
        followed by finite ordered bounds for the seed-offset bracket.
    delta
        Finite strictly positive maximum displacement from a cluster's first
        difference.
    min_occurrences
        Integer threshold at least 2 for retaining a cluster.

    Returns
    -------
    b_seed : float
        Ordinary median of the multiplicity-preserving closure candidates
        retained inside the supplied offset bracket.
    corrected_frequencies : np.ndarray
        Seed-corrected frequencies aligned with ``line_ids``.
    occurrences : np.ndarray
        Integer array with shape ``(n_occurrences, 3)``.  Each row is
        ``(cluster_index, i, j)`` with ``i < j``.
    counts : np.ndarray
        Number of pair occurrences in each retained cluster.

    Raises
    ------
    ValueError
        If the aligned line data, labels, reported centres, bounds, clustering
        tolerance or occurrence threshold violate the stated contracts, or if
        the bracket contains no eligible closure candidate.
    """
    return b_seed, corrected_frequencies, occurrences, counts  # placeholder
```

### Step 2

step_02_instantiate_k22_motifs

Goal
----
Recover the local four-transition closure motifs.

```python
def instantiate_k22_motifs(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Return the local closure motifs supported by recurrence data.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of
        ``cluster_recurring_differences`` in step 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_motifs, 4)``. Each row
        ``(h1, l1, h2, l2)`` holds the storage indices of the four distinct
        lines of one local closure: ``(h1, l1)`` and ``(h2, l2)`` are two
        pair occurrences of the same retained cluster, each ordered from the
        higher to the lower seed-corrected frequency. One row is returned for
        every line-disjoint combination of two pair occurrences in every
        retained cluster, so a closure whose two pairings fall in two
        retained clusters appears once for each. Within a row, the pair with
        the smaller (higher-line identifier, lower-line identifier) tuple
        comes first; rows are sorted lexicographically by the string
        identifiers of ``(h1, l1, h2, l2)``.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs or no line-disjoint local motif
        can be recovered.
    """
    return motifs  # placeholder
```

### Step 3

step_03_derive_local_vertex_constraints

Goal
----
Derive canonical local shared-level constraints from closure motifs.

```python
def derive_local_vertex_constraints(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> "np.ndarray":
    """Return canonical line pairs that witness common local vertices.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``instantiate_k22_motifs`` in
        step 2.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_pairs, 2)`` containing the sorted unique
        line-index pairs implied to share a local energy-level vertex.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs or no local vertex constraint
        can be derived.

    Index and ordering convention
    -----------------------------
    Entries are zero-based positions in the supplied arrays. Order the two
    members of each pair by ascending string-valued line identifier, then sort
    the unique rows lexicographically by their two identifier values. This
    ordering is by identifier, not by integer storage position.
    """
    return sharing_pairs
```

### Step 4

step_04_stitch_energy_level_cliques

Goal
----
Stitch local shared-level constraints into global energy-level cliques.

```python
def stitch_energy_level_cliques(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> "np.ndarray":
    """Consolidate compatible local vertex witnesses into global levels.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of
        ``derive_local_vertex_constraints`` in step 3.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_levels, n_lines)``.  Entry ``(k, i)`` is
        1 exactly when transition line ``i`` is incident on recovered level
        ``k``. Rows are ordered lexicographically by the sorted tuple of
        incident line identifiers.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, a stitched component has fewer
        than three lines, no global level is recovered, or a transition line
        does not belong to exactly two recovered levels.
    """
    return level_incidence  # placeholder
```

### Step 5

step_05_build_bipartite_level_network

Goal
----
Recover the bipartite level network and its unique missing transition.

```python
def build_bipartite_level_network(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", int]:
    """Partition the stitched levels into upper and lower vertex classes.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``stitch_energy_level_cliques`` in
        step 4.

    Returns
    -------
    upper_incidence : np.ndarray
        Integer array of shape ``(n_upper, n_lines)``.
    lower_incidence : np.ndarray
        Integer array of shape ``(n_lower, n_lines)``.
    endpoints : np.ndarray
        Integer array of shape ``(n_lines, 2)`` containing each observed
        transition's upper and lower indices.
    missing_edge : np.ndarray
        Length-two integer array containing the unique unobserved upper and
        lower indices.
    gauge_lower : int
        Index of the lower level fixed to zero in the weighted inversion.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, the recovered level graph is
        disconnected or non-bipartite, the colour classes have equal size, a
        line lacks one endpoint in each class, or the cross partition has
        anything other than one absent pair.
    """
    return upper_incidence, lower_incidence, endpoints, missing_edge, gauge_lower  # placeholder
```

### Step 6

step_06_invert_network_and_predict

Goal
----
Fit the recovered network and evaluate its unique missing transition.

```python
def invert_network_and_predict(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2
        ) -> tuple["np.ndarray", "np.ndarray", int, "np.ndarray"]:
    """Fit the gauged network and evaluate its unique absent edge.

    Parameters
    ----------
    line_ids, scans, reported_frequencies, b_lo, b_hi, delta, min_occurrences
        Inputs satisfying the contract of ``build_bipartite_level_network``
        in step 5.
    standard_uncertainties
        Finite strictly positive standard uncertainties aligned with the
        spectral lines and expressed in the same units as their frequencies.

    Returns
    -------
    parameters : np.ndarray
        Fitted vector containing upper energies, nongauge lower energies, and
        the refined block-B offset in that order.
    residuals : np.ndarray
        ``reported_frequencies - A @ parameters`` evaluated through the
        centred system to avoid cancellation.
    rank : int
        Rank of the weighted design matrix.
    prediction : np.ndarray
        Length-four float array containing, in order, the missing physical
        frequency, refined block-B offset, fitted missing-upper energy, and
        fitted missing-lower energy under the selected gauge.

    Raises
    ------
    ValueError
        If an earlier step rejects its inputs, the uncertainty array is not
        aligned, finite and strictly positive, the weighted design matrix is
        rank deficient, or a returned fitted quantity is not finite.

    Parameter ordering convention
    -----------------------------
    Use the upper and lower row ordering defined for build_bipartite_level_network:
    ascending lexicographic order of sorted string-valued incident line-ID tuples
    within each colour class. List upper energies in that order, followed by
    nongauge lower energies in their class order, followed by the scan-B offset.
    """
    return parameters, residuals, rank, prediction
```

### Step 7

step_07_missing_transition_orchestrator

Goal
----
Run the complete reconstruction and return the missing-line frequency.

```python
def reconstruct_missing_transition_frequency(
        line_ids: "np.ndarray", scans: "np.ndarray",
        reported_frequencies: "np.ndarray",
        standard_uncertainties: "np.ndarray",
        b_lo: float, b_hi: float, delta: float,
        min_occurrences: int = 2) -> float:
    """Execute all reconstruction stages and return the held-out frequency.

    Parameters
    ----------
    line_ids
        Unique transition-line identifiers.
    scans
        Scan labels A or B aligned with the lines.
    reported_frequencies
        Finite pairwise-distinct reported line centres.
    standard_uncertainties
        Finite strictly positive standard uncertainties aligned with the
        lines and expressed in the same units as the frequencies.
    b_lo, b_hi
        Closed instrument bracket for the topology-seed offset.
    delta
        Positive first-value clustering tolerance.
    min_occurrences
        Integer retained-cluster threshold of at least 2.

    Returns
    -------
    float
        Physical frequency of the unique unobserved transition.

    Raises
    ------
    ValueError
        If any earlier step rejects its inputs or independently returned
        intermediate representations are inconsistent.
    """
    return predicted_frequency  # placeholder
```
