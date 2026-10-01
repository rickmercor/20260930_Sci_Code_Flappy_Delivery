# Biology-Genetics-3

## Background

Genotype representation graphs compress phased genetic variation by assigning mutations to nodes whose descendant leaves are exactly the carrying haplotypes. This shared topology saves space, but a carrier-set correction becomes a graph-editing problem rather than a local table update.

Bulk allele polarization creates many such corrections together. Shared traversal methods amortize overlapping graph searches while retaining exact carrier semantics across both rare and common updates.

The resulting edits are assessed through reachability, structural reuse, and the work saved relative to independent mutation searches. Small synthetic graphs expose these interactions without requiring genetic data files or specialized infrastructure.

## Problem

Phased genetic variation can be stored in a leaf-labeled directed acyclic graph whose internal nodes share descendant haplotypes across mutations. Revising the ancestral allele at a multiallelic site changes the mutation records and their carrier sets, so the replacements must be remapped without violating the graph's exact reachability semantics.

For one deterministic graph, evaluate five multiallelic sites with the batched mutation-remapping method and compare its candidate-discovery work with independent remapping. Apply the published carrier-set, graph-editing, and candidate-aware work conventions to the configuration below. Every mutation record rewritten at a polarized site, including each retained non-ancestral alternate, counts as a replacement mutation remapped in the batch.

Use the following configuration:

- `children = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [15, 16], [17, 18], [19, 20], [21, 22], [22, 23], [24, 25]]`
- `n_samples = 14`
- `alternate_carriers = [[[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], [[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]`
- `observed_haplotypes = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]`
- `ancestral_alternate_index = [0, 1, 0, 1, 0]`
- `word_size = 4`
- `dense_threshold = 0.25`

Your final answer must be a single number: the full-precision ratio of batched candidate-discovery work to independent candidate-discovery work for the resulting replacement mutations.

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

01_generate_polarization_updates

Goal
----
Construct carrier updates for multiallelic polarization.

Multiallelic polarization replaces an ancestral alternate by the former

reference allele while retaining every non-ancestral alternate. For each site

with observed set C and original alternate carrier sets A_j, the

former-reference carriers are C minus the union of all A_j. Surviving

alternates remain in input order and the former-reference replacement is last.

```python
def generate_polarization_updates(
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
) -> "np.ndarray":
    '''Generate replacement carrier rows for multiallelic polarization.

    Parameters
    ----------
    alternate_carriers : np.ndarray
        Binary array with one carrier row per alternate allele at each site.
    observed_haplotypes : np.ndarray
        Binary observed-haplotype mask for every site.
    ancestral_alternate_index : np.ndarray
        Index of the alternate allele inferred to be ancestral at each site.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes, fewer than two alternates, no
        samples, non-binary entries, overlapping alternate carrier sets,
        alternate carriers outside the observed set, or an invalid ancestral
        alternate index.

    Returns
    -------
    updates : np.ndarray
        Binary replacement carrier rows, ordered by site.
    '''
    return updates  # noqa: F821
```

### Step 2

02_encode_adaptive_reaches

Goal
----
Encode graph-node reaches with adaptive sparse or dense storage.

A genotype representation graph node represents the union of the sample

leaves below its children. The multitree property makes sibling reaches

disjoint. A reach with density d / N at least dense_threshold is stored

densely; otherwise it remains sparse. Integer bitmasks retain the exact reach

while a flag records the adaptive choice.

```python
def encode_adaptive_reaches(
    children: list[list[int]], n_samples: int, dense_threshold: float
) -> "np.ndarray":
    '''Encode exact descendant reaches and their adaptive storage class.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered child identifiers for every graph node.
    n_samples : int
        Number of leading sample leaves.
    dense_threshold : float
        Density threshold in the interval (0, 1].

    Raises
    ------
    ValueError
        If n_samples is invalid or above 62, dense_threshold is outside (0, 1],
        the graph is empty or not topologically ordered, a sample has children,
        an internal node has no children, a child is repeated, or sibling
        descendant reaches overlap.

    Returns
    -------
    reach_encoding : np.ndarray
        Integer columns for reach bitmask, reach size, and dense-storage flag.
    '''
    return reach_encoding  # noqa: F821
```

### Step 3

03_pack_batch_memberships

Goal
----
Pack a batch of mutation memberships into fixed-width words.

Shared mutation discovery stores one compatibility bit per update at each

active node. Consecutive updates occupy consecutive low-to-high bits in

fixed-width words, so update i uses word floor(i / w) and bit i modulo w. Leaf

words are seeded from replacement carrier membership.

Inputs

```python
def pack_batch_memberships(update_carriers: "np.ndarray", word_size: int) -> "np.ndarray":
    '''Pack leaf membership across a mutation-update batch.

    Parameters
    ----------
    update_carriers : np.ndarray
        Binary carrier rows for the batch.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If update_carriers is not a nonempty two-dimensional binary array or
        word_size is not an integer from 1 through 62.

    Returns
    -------
    packed_memberships : np.ndarray
        Packed integer words for each sample leaf.
    '''
    return packed_memberships  # noqa: F821
```

### Step 4

04_discover_shared_candidates

Goal
----
Discover compatible graph nodes for a shared update batch.

Batched candidate discovery propagates all mutation compatibilities in one

reverse-topological pass. Leaf state is packed carrier membership. Each

internal word starts from the valid-bit mask for that word and is intersected

with the corresponding word of every child, including a zero-state child. Thus

a node retains update bit i exactly when D(n) is contained in S_i. A zero state

prunes the branch above that node. Words use low-to-high bit order; unused high

bits in every word, including the partial final word, are invalid.

```python
def discover_shared_candidates(
    children: list[list[int]],
    n_samples: int,
    packed_leaf_memberships: "np.ndarray",
    word_size: int,
    n_updates: int,
) -> "np.ndarray":
    '''Propagate packed compatibility states through one shared traversal.

    Parameters
    ----------
    children : list[list[int]]
        Child identifiers for every graph node in canonical topological order:
        sample rows are empty, internal rows are nonempty, child identifiers
        are distinct integers, and every child precedes its parent.
    n_samples : int
        Number of leading sample leaves.
    packed_leaf_memberships : np.ndarray
        Nonnegative integer membership words for the sample leaves. Word j
        carries updates j * word_size through (j + 1) * word_size - 1 in
        low-to-high bit order; every unused high bit must be zero.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.
    n_updates : int
        Number of represented mutation updates.

    Raises
    ------
    ValueError
        If n_samples, word_size, or n_updates is invalid; the graph is empty,
        malformed, or not topologically ordered; packed_leaf_memberships has
        the wrong shape, a non-integer dtype (float and bool arrays are
        rejected even when their values are whole numbers), negative values,
        or bits outside the update range.

    Returns
    -------
    state_words : np.ndarray
        Packed compatibility words for every node as an independent int64
        array. The input array is not modified.
    '''
    return state_words  # noqa: F821
```

### Step 5

05_plan_snapshot_attachments

Goal
----
Plan mutation attachments against a read-only graph snapshot.

Mutation application consumes candidates discovered on one read-only graph

snapshot. An exact-reach node is reused directly; when several nodes reach

exactly the carrier set, the smallest node identifier is reused. Otherwise

compatible internal nodes are considered by decreasing descendant count, with

equal counts taken in increasing node-identifier order, disjoint reaches

are retained, and uncovered carriers attach as leaves. Each plan stores an

exact node identifier or a bitmask of snapshot child identifiers.

```python
def plan_snapshot_attachments(
    children: list[list[int]],
    n_samples: int,
    reach_encoding: "np.ndarray",
    state_words: "np.ndarray",
    update_carriers: "np.ndarray",
    word_size: int,
) -> "np.ndarray":
    '''Plan reuse and new-node children from one discovery snapshot.

    Among several nodes whose reach equals the carrier set, the smallest
    identifier is reused. Candidates are visited by decreasing reach size,
    breaking ties by increasing node identifier.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    reach_encoding : np.ndarray
        Reach mask, reach size, and adaptive storage flag for each node.
    state_words : np.ndarray
        Packed compatibility state for each node.
    update_carriers : np.ndarray
        Binary carrier rows for the update batch.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If the snapshot has more than 62 nodes; n_samples or word_size is
        invalid; the graph is malformed; any numerical input has an incompatible
        shape, type, or value; or reach and compatibility data contradict the
        update carriers.

    Returns
    -------
    attachment_plans : np.ndarray
        Rows containing exact node id or -1, followed by a child-node bitmask.
    '''
    return attachment_plans  # noqa: F821
```

### Step 6

06_apply_snapshot_batch

Goal
----
Apply a batch of attachment plans to the graph snapshot.

A batch applies plans made against a fixed read-only snapshot. Exact matches

attach to existing nodes. Every nonempty child mask appends a distinct mutation

node in row order, even when an earlier row appended an equivalent node,

because intra-batch structure was absent during discovery.

```python
def apply_snapshot_batch(
    children: list[list[int]], n_samples: int, attachment_plans: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    '''Apply fixed-snapshot attachment plans in deterministic row order.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    attachment_plans : np.ndarray
        Exact node identifier and child-node bitmask for each update. A row
        equal to (-1, 0) appends no node and retains attachment identifier -1.

    Raises
    ------
    ValueError
        If n_samples or the graph is invalid, the snapshot has more than 62
        nodes, attachment_plans is not a nonempty integer array with two columns,
        a plan combines exact reuse with children, or a plan refers to a node
        outside the discovery snapshot.

    Returns
    -------
    edited_adjacency : np.ndarray
        Binary parent-by-child adjacency matrix after the batch.
    attachment_nodes : np.ndarray
        Node identifier assigned to each replacement mutation.
    '''
    return result  # noqa: F821
```

### Step 7

07_audit_independent_discovery

Goal
----
Audit independent per-mutation graph discovery.

Independent discovery starts from one mutation's carrier leaves and proceeds

upward only through compatible nodes. A reached internal node is counted before

its compatibility test; a compatible node is also counted as a candidate and

activates its parents. Packing each visit set as a node bitmask makes both

per-mutation work and the shared union exact.

```python
def audit_independent_discovery(
    snapshot_adjacency: "np.ndarray", n_samples: int, update_carriers: "np.ndarray"
) -> "np.ndarray":
    '''Audit independent traversal visits and candidate counts.

    Parameters
    ----------
    snapshot_adjacency : np.ndarray
        Binary parent-by-child adjacency matrix of the read-only discovery
        snapshot, in topological node order.
    n_samples : int
        Number of leading sample leaves.
    update_carriers : np.ndarray
        Binary carrier rows for the audited mutations.

    Raises
    ------
    ValueError
        If snapshot_adjacency is not a square nonempty binary matrix with at most
        62 nodes, n_samples is invalid, sample rows have children, an internal
        node has no children or a nonpreceding child, or update_carriers has an
        incompatible shape or non-binary entries.

    Returns
    -------
    audit_counts : np.ndarray
        Visited-node bitmask and compatible internal candidate count per update.
    '''
    return audit_counts  # noqa: F821
```

### Step 8

08_compute_batch_work_ratio

Goal
----
Compute the shared-batch traversal work ratio.

The independent discovery work model sums visited nodes and generated

candidates across mutations. The batched model charges q packed-word

operations per distinct visited node and the same total candidates, where q is

the ceiling of mutation count divided by word size. Their quotient measures

batched work relative to independent work.

```python
def compute_batch_work_ratio(
    audit_counts: "np.ndarray", n_nodes: int, word_size: int
) -> float:
    '''Compute the candidate-aware batched-to-independent work ratio.

    Parameters
    ----------
    audit_counts : np.ndarray
        Visited-node bitmask and candidate count for each mutation.
    n_nodes : int
        Number of nodes in the audited graph, from 1 through 62.
    word_size : int
        Number of usable update bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If audit_counts is not a nonempty integer array with two columns, a
        visited mask or candidate count is negative, a visited mask refers
        outside the graph, n_nodes or word_size is invalid, or independent work
        is zero.

    Returns
    -------
    work_ratio : float
        Batched candidate-discovery work divided by independent work.
    '''
    return work_ratio  # noqa: F821
```

### Step 9

09_run_full_pipeline

Goal
----
Orchestrate the complete multiallelic GRG batch-cost benchmark.



The full calculation converts multiallelic polarization records into



replacement carrier sets, encodes the snapshot's reaches and packed mutation



state, discovers and applies fixed-snapshot attachments, verifies that every replacement



reaches exactly its carriers, audits independent



discovery on the unedited discovery snapshot, and evaluates the paper's



batched-to-independent work quotient. This orchestrator is the final Studio



step.

```python
def run_full_pipeline(
    children: list[list[int]],
    n_samples: int,
    alternate_carriers: "np.ndarray",
    observed_haplotypes: "np.ndarray",
    ancestral_alternate_index: "np.ndarray",
    word_size: int,
    dense_threshold: float,
) -> float:
    '''Run multiallelic remapping and return the snapshot-discovery work ratio.

    Parameters
    ----------
    children : list[list[int]]
        Topologically ordered discovery-snapshot graph.
    n_samples : int
        Number of leading sample leaves.
    alternate_carriers : np.ndarray
        Binary original alternate carrier rows by site.
    observed_haplotypes : np.ndarray
        Binary observed-haplotype masks by site.
    ancestral_alternate_index : np.ndarray
        Ancestral alternate index for every site.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.
    dense_threshold : float
        Adaptive dense-storage threshold in the interval (0, 1].

    Raises
    ------
    ValueError
        If the graph or n_samples is malformed, the polarization arrays have
        incompatible shapes or values, word_size is outside 1 through 62,
        dense_threshold is outside (0, 1], or the applied batch leaves a
        replacement mutation on a node whose descendant samples are not exactly
        its carriers.

    Returns
    -------
    work_ratio : float
        Batched candidate-discovery work divided by independent work, with
        both quantities evaluated on the read-only discovery snapshot.
    '''
    return work_ratio  # noqa: F821
```
