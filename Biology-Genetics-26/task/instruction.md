# Biology-Genetics-26

## Background

Introgression transfers genetic material between diverged lineages, leaving local genealogies that disagree with the species tree. Incomplete lineage sorting also produces discordance, so a directional claim requires evidence that separates ordinary coalescent variation from an excess associated with a particular branch pair.

Full-tree approaches retain information discarded by isolated quartets. They can compare where discordant subtrees attach while accounting for whether the relevant branches are present, then use the rearrangements needed to reconcile each gene tree with the species tree as directional evidence.

An internal recipient adds another layer because several descendant lineages may or may not have coalesced before gene flow. Their lineage configurations and independent parental choices form a finite mixture. The observed attachment frequency is therefore a nonlinear function of the per-lineage introgression probability rather than a direct mixture proportion.

## Problem

Discordant gene trees can reveal introgression, but unequal branch observability and incomplete lineage sorting can confound both event detection and direction.

Evaluate candidates A and B using a full-gene-tree framework for attachment support and direction. Attachment codes `1`, `2`, and `0` mean focal-test, focal-uncle, and neither; branch-presence columns are test, uncle, and focal. Rooted trees are unordered nested binary tuples; canonical forms recursively place the child with the lexicographically smaller repr first, and a branch mask is the sum of `2**j` over its descendant leaves. If reconciliation has multiple minimum paths, use the lexicographically first sequence of canonical topology repr strings. For stochastic direction ties, use `np.random.default_rng(17)`, process tied patterns in listed order, draw a vector whose length is the corresponding multiplicity with `0` for test and `1` for focal, and consume one additional draw only for an aggregate tie.

For a selected three-descendant recipient, its two listed intervals are, in order, the coalescent-unit lengths ancestral only to the sister pair and ancestral to all three descendants before introgression. Infer the per-lineage introgression probability `gamma` under the same framework; the supplied 26-entry conditional vectors use the source configuration order, with local entry `m` in each configuration block corresponding to donor-subset mask integer `m`.

Use this configuration:

- `attachment_codes_by_candidate = [np.repeat([1, 2, 0, 2, 0], [50, 22, 8, 10, 10]), np.repeat([2, 1, 0], [30, 50, 20])]`
- `branch_presence_by_candidate = [np.repeat([[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0]], [72, 8, 18, 2], axis=0), np.repeat([[1, 1, 1], [1, 0, 1]], [60, 40], axis=0)]`
- `left_clade = ((0, 1), 2)` and `right_clade = ((5, 6), 7)`
- `species_tree = (((left_clade, 3), 4), right_clade)`
- `gene_tree_patterns_by_candidate = [[((left_clade, (right_clade, 3)), 4), (left_clade, ((right_clade, 3), 4)), ((left_clade, right_clade), (3, 4)), ((left_clade, (3, 4)), right_clade), (((left_clade, 4), right_clade), 3)], [((((right_clade, (0, 1)), 2), 3), 4), ((right_clade, (0, 1)), ((2, 3), 4))]]`
- `gene_tree_multiplicities_by_candidate = [[11, 7, 13, 12, 7], [31, 19]]`
- `candidate_clade_masks = [[224, 7], [224, 3]]`, with rows ordered A then B and columns ordered test then focal
- `z_cutoff = -1.96`
- `recipient_intervals_by_mask = {7: [0.37, 0.42], 224: [0.51, 0.33]}`
- `parent_attachment_probabilities_by_mask = {7: [0.08, 0.36, 0.43, 0.70, 0.31, 0.62, 0.68, 0.91, 0.12, 0.58, 0.47, 0.88, 0.10, 0.51, 0.61, 0.89, 0.11, 0.56, 0.49, 0.87, 0.15, 0.84, 0.13, 0.82, 0.14, 0.80], 224: [0.09, 0.41, 0.35, 0.68, 0.33, 0.64, 0.59, 0.90, 0.13, 0.55, 0.50, 0.86, 0.11, 0.49, 0.63, 0.88, 0.12, 0.57, 0.46, 0.85, 0.16, 0.83, 0.14, 0.81, 0.15, 0.79]}`

Identify the supported directed event and give a compact numerical audit trail that makes the attachment decision, directional reconciliation, coalescent mixture, and admissible-root selection independently checkable. The final answer is a single number: `gamma` for that event; grading uses relative tolerance zero and absolute tolerance `1e-8` for every non-integer intermediate and the final number.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the derived numerical checkpoints needed to substantiate the event and the admissible root.
Do not paste the supplied input arrays, full topology sequences along reconciliation paths, or per-history tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_summarize_candidate_evidence

Goal
----
Summarize discordant attachments and branch observability for each candidate pair.

```python
import numpy as np


def summarize_candidate_evidence(
    attachment_codes_by_candidate: list,
    branch_presence_by_candidate: list,
) -> np.ndarray:
    '''Count attachments and branch availability for every candidate.

    Parameters
    ----------
    attachment_codes_by_candidate : list
        One-dimensional arrays containing only 0, 1, and 2.
    branch_presence_by_candidate : list
        Matching binary arrays with test, uncle, and focal columns.

    Returns
    -------
    summary : np.ndarray
        Integer array with six columns in the declared order.

    Raises
    ------
    ValueError
        If the candidate sequences are empty or unequal in length, an attachment-code array is not a one-dimensional integer array containing only {0, 1, 2}, or a presence array is not a matching binary integer array of shape (n, 3).
    '''
    return summary
```

### Step 2

02_screen_attachment_candidates

Goal
----
Correct unequal branch availability and screen focal-test attachment excesses.

```python
import numpy as np


def screen_attachment_candidates(
    candidate_summary: np.ndarray,
    z_cutoff: float = -1.96,
) -> np.ndarray:
    '''Apply availability correction and the signed avuncular screen.

    Parameters
    ----------
    candidate_summary : np.ndarray
        Nonnegative rows [raw_test, raw_uncle, avail_test, avail_uncle, avail_focal, joint_test_focal].
    z_cutoff : float
        Finite inclusive lower-tail cutoff.

    Returns
    -------
    screen : np.ndarray
        Columns [corrected_test, corrected_uncle, raw_z_score, corrected_z_score, supported].

    Raises
    ------
    ValueError
        If candidate_summary is not a nonempty numeric array of shape (c, 6) with finite nonnegative entries, either tested availability is nonpositive, an attachment count exceeds its availability, or z_cutoff is not finite and numeric.
    '''
    return screen
```

### Step 3

03_reconcile_rooted_nni

Goal
----
Reconcile compressed rooted gene-tree topologies to a rooted species tree by minimum nearest-neighbor interchanges and tabulate candidate-branch participation.

```python
import numpy as np


def reconcile_rooted_nni(
    species_tree: tuple,
    gene_trees: list,
    candidate_clade_masks: np.ndarray,
) -> np.ndarray:
    '''Find shortest rooted-NNI reconciliations and branch appearances.

    Parameters
    ----------
    species_tree : tuple
        Rooted binary nested tuple with integer leaves.
    gene_trees : list
        Rooted binary nested tuples on the species-tree leaves.
    candidate_clade_masks : np.ndarray
        Two species-tree clade bitmasks ordered test then focal.

    Returns
    -------
    reconciliation_summary : np.ndarray
        Columns [minimum_nni_distance, number_of_shortest_paths, test_clade_appearances, focal_clade_appearances].

    Raises
    ------
    ValueError
        If the species tree is not a rooted binary tree with three to eight unique nonnegative integer leaves, a gene tree does not contain the same leaves exactly once, candidate masks do not identify two distinct non-root species-tree clades, or a gene tree cannot be reconciled by rooted NNI.
    '''
    return reconciliation_summary
```

### Step 4

04_aggregate_recipient_votes

Goal
----
Convert reconciliation-triple appearances into one direction vote per gene tree.

```python
import numpy as np


def aggregate_recipient_votes(
    reconciliation_summary: np.ndarray,
    multiplicities: np.ndarray,
    seed: int,
) -> np.ndarray:
    '''Aggregate one recipient vote per represented gene tree.

    Parameters
    ----------
    reconciliation_summary : np.ndarray
        Columns [minimum_nni_distance, number_of_shortest_paths, test_clade_appearances, focal_clade_appearances].
    multiplicities : np.ndarray
        Positive integer multiplicity for each row.
    seed : int
        Seed for np.random.default_rng; untied rows consume no draws, tied rows consume multiplicity draws in input order, and an aggregate tie consumes one final draw.

    Returns
    -------
    vote_summary : np.ndarray
        Test votes, focal votes, and the selected recipient index.

    Raises
    ------
    ValueError
        If reconciliation_summary is not a nonempty (p, 4) array of nonnegative integers, multiplicities are not one positive integer per row, or seed is not an integer.
    '''
    return vote_summary
```

### Step 5

05_compute_three_descendant_configurations

Goal
----
Compute the seven lineage-configuration probabilities for a rooted three-descendant internal recipient.

```python
import numpy as np


def compute_three_descendant_configurations(
    first_interval: float,
    second_interval: float,
) -> np.ndarray:
    '''Evaluate the seven three-descendant lineage configurations.

    Parameters
    ----------
    first_interval : float
        Finite nonnegative interval ancestral only to the sister pair.
    second_interval : float
        Finite nonnegative interval ancestral to all three descendants.

    Returns
    -------
    configuration_probabilities : np.ndarray
        Seven probabilities in source order.

    Raises
    ------
    ValueError
        If either interval is not finite and numeric or is negative.
    '''
    return configuration_probabilities
```

### Step 6

06_compute_parent_history_weights

Goal
----
Expand seven lineage configurations into the 26 donor-subset history weights.

```python
import numpy as np


def compute_parent_history_weights(
    configuration_probabilities: np.ndarray,
    gamma: float,
) -> np.ndarray:
    '''Compute all three-descendant parent-history weights.

    Parameters
    ----------
    configuration_probabilities : np.ndarray
        Seven finite nonnegative probabilities summing to one.
    gamma : float
        Finite probability in [0, 1].

    Returns
    -------
    parent_history_weights : np.ndarray
        Twenty-six normalized weights in the task-declared parent-history array order.

    Raises
    ------
    ValueError
        If the configuration vector is not seven finite nonnegative values summing to one or gamma is not a finite value in [0, 1].
    '''
    return parent_history_weights
```

### Step 7

07_estimate_three_descendant_gamma

Goal
----
Invert the 26-history attachment mixture for the per-lineage donor probability.

```python
import numpy as np


def estimate_three_descendant_gamma(
    attachment_count: int,
    joint_availability: int,
    configuration_probabilities: np.ndarray,
    parent_attachment_probabilities: np.ndarray,
) -> float:
    '''Solve the three-descendant attachment mixture for gamma.

    Parameters
    ----------
    attachment_count : int
        Nonnegative attachment count.
    joint_availability : int
        Positive availability not smaller than the attachment count.
    configuration_probabilities : np.ndarray
        Seven finite probabilities summing to one.
    parent_attachment_probabilities : np.ndarray
        Twenty-six finite conditional probabilities in [0, 1], in the task-declared parent-history array order.

    Returns
    -------
    gamma_hat : float
        The unique model root in [0, 1].

    Raises
    ------
    ValueError
        If the counts are not valid nonnegative integers with positive joint availability, the configuration or conditional vectors violate their declared probability contracts, or the attachment equation does not have exactly one root in [0, 1].
    '''
    return gamma_hat
```

### Step 8

08_run_full_pipeline

Goal
----
Chain attachment screening, rooted-NNI direction inference, and the three-descendant parent-history mixture into the requested scalar.

```python
def run_full_pipeline(seed: int = 17) -> float:
    '''Run the deterministic full-gene-tree introgression analysis.

    Parameters
    ----------
    seed : int
        Seed passed to the Step 04 np.random.default_rng tie process.

    Returns
    -------
    gamma_hat : float
        Estimated per-lineage donor probability.

    Raises
    ------
    ValueError
        If seed is not an integer or the fixed configuration does not yield one supported candidate, a modeled recipient, and a mixture consistent with the observed attachment frequency.
    '''
    return gamma_hat
```
