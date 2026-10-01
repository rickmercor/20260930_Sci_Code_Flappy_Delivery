# Biology-Genetics-13

## Background

Large phased cohorts contain extensive genealogical sharing. A genotype representation graph uses a leaf-labeled directed acyclic graph to encode that sharing, assigning a mutation to a node whose descendant leaves are exactly its carrier haplotypes.

Changing carriers after graph construction is difficult because the genotype information is embedded in reachability. Recent work treats bulk changes as graph edits and shares discovery work across many mutations, while retaining compact state for both rare and common carrier sets.

The method yields an edited graph with exact carrier-set semantics and exposes how much repeated traversal a batch can eliminate. Its behavior is checked through reachability equivalence, deterministic topology updates, overlap in visited nodes, and the compactness of the resulting graph.

## Problem

Phased genetic variation can be stored in a leaf-labeled directed acyclic graph whose internal nodes share descendant haplotypes across mutations. Because a mutation's carrier set is the reach of its assigned node, changing an allele orientation requires reuse-aware graph editing rather than replacing a genotype-matrix column.

For one deterministic graph, perform a bulk biallelic polarization edit using shared reverse-topological remapping with packed per-update state and adaptive carrier-set storage. Apply the edit batch from one read-only discovery snapshot, then quantify the traversal reuse available to the supplied audit batch on the edited graph.

Your task is to solve one concrete deterministic example of this pipeline. Use the following configuration:

- `children = [[], [], [], [], [], [], [], [], [], [], [], [], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [13, 14], [15, 16], [18, 19, 20]]`
- `n_samples = 12`
- `alternate_carriers = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]`
- `observed_haplotypes = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]`
- `ancestral_is_alternate = [1, 1, 1, 1, 1, 1]`
- `audit_carriers = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]`
- `word_size = 4`
- `dense_threshold = 0.25`
- The edit updates are applied in row order, a reach-size tie is broken by the smaller node identifier, and each appended node receives the smallest unused identifier.
- The node identifiers are topologically ordered with samples `0` through `11` as leaves; visit accounting includes each seeded leaf and every internal node reached before a failed compatibility test prunes its branch.

Determine the polarized carrier sets, the packed shared-discovery state, the adaptive descendant representations, the fixed-snapshot attachment nodes, and the independent visit counts needed to evaluate traversal overlap. Your final answer must be a single number: the post-edit audit overlap factor `rho(5)` at full precision.

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

01_polarize_carrier_sets.py

Goal
----
Biallelic polarization changes the represented mutation when the alternate allele is ancestral. For site i, the alternate-carrier indicator A_i is replaced within the observed haplotype indicator C_i by S_i = C_i - A_i; a site whose alternate allele is not ancestral retains A_i. All indicators are binary, and an alternate carrier must also be observed.

Inputs

------

alternate_carriers: Binary array of shape (k, n_samples).

observed_haplotypes: Binary array of shape (k, n_samples).

ancestral_is_alternate: Binary array of shape (k,).

Returns

-------

target_carriers: Binary uint8 array of shape (k, n_samples).

```python
import numpy as np


def polarize_carrier_sets(
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
) -> np.ndarray:
    """Construct carrier sets after biallelic allele polarization.

    Parameters
    ----------
    alternate_carriers : np.ndarray
        Binary alternate-carrier matrix with shape (k, n_samples).
    observed_haplotypes : np.ndarray
        Binary observed-haplotype matrix with the same shape.
    ancestral_is_alternate : np.ndarray
        Binary vector indicating which sites require complementation.

    Raises
    ------
    ValueError
        If the carrier matrices are not matching two-dimensional binary arrays,
        if `ancestral_is_alternate` is not binary with one entry per site, or if
        an alternate carrier is not observed.

    Returns
    -------
    target_carriers : np.ndarray
        Binary uint8 carrier matrix after polarization.
    """
    return target_carriers  # noqa: F821
```

### Step 2

02_pack_leaf_memberships

Goal
----
Shared candidate discovery carries one compatibility bit per update. If carrier matrix entry X_{i,s} is one, bit i mod w is set in word floor(i/w) for leaf s, where w is the selected word width. The resulting q = ceil(k/w) words let one bitwise operation update several mutation states at once.

Inputs

------

carrier_matrix: Binary array of shape (k, n_samples).

word_size: Number of active bits in each packed word.

Returns

-------

packed_memberships: Integer array of shape (n_samples, ceil(k / word_size)).

```python
import numpy as np


def pack_leaf_memberships(carrier_matrix: np.ndarray, word_size: int = 32) -> np.ndarray:
    """Pack per-update leaf memberships into fixed-width words.

    Parameters
    ----------
    carrier_matrix : np.ndarray
        Binary carrier matrix with shape (k, n_samples).
    word_size : int, optional
        Number of update bits assigned to each word.

    Raises
    ------
    ValueError
        If `carrier_matrix` is empty, is not two dimensional, or is not binary,
        or if `word_size` is outside [1, 62].

    Returns
    -------
    packed_memberships : np.ndarray
        int64 array with one row per sample and one column per word.
    """
    return packed_memberships  # noqa: F821
```

### Step 3

03_encode_adaptive_descendants

Goal
----
A genotype representation graph stores the reach D(n) of node n as the union of the disjoint reaches of its children, with D(s) = {s} for a sample leaf. A reach of density |D(n)| / N at least tau is represented by a width-N bitset; a lower-density reach remains a sparse list. Ordered node identifiers place every child before its parent.

Inputs

------

children: Child node identifiers for every graph node.

n_samples: Number of leaf nodes at the beginning of the node array.

dense_threshold: Carrier-set density at which bitset storage is used.

Returns

-------

encoding: Native numeric descendant lists, representation flags, storage payloads, and cardinalities.

```python
from typing import Any


def encode_adaptive_descendants(children: list, n_samples: int, dense_threshold: float) -> dict[str, Any]:
    """Encode every node reach with adaptive sparse or dense storage.

    Parameters
    ----------
    children : list
        List whose entry n contains the child identifiers of node n.
    n_samples : int
        Number of sample leaves, identified by 0 through n_samples - 1.
    dense_threshold : float
        Inclusive density threshold for dense bitset storage.

    Raises
    ------
    ValueError
        If `children` is empty, `n_samples` does not define a nonempty leaf
        prefix, or `dense_threshold` is outside [0, 1]. Also raised if a leaf
        has children, an internal node has no children, a child list has the
        wrong type or repeats an identifier, a child does not precede its
        parent, or sibling child reaches overlap.

    Returns
    -------
    encoding : dict
        Descendant lists, dense flags, encoded payloads, and cardinalities.
    """
    return encoding  # noqa: F821
```

### Step 4

04_discover_shared_candidates

Goal
----
Batched candidate discovery starts from the union of target carriers and visits the ordered graph upward once. A leaf state is its packed update membership, while an internal state is the wordwise intersection of every child state. A nonzero internal state identifies the updates for which the node reach is compatible and propagates to parents; an all-zero state terminates that branch.

Inputs

------

children: Ordered child identifiers for the fixed graph snapshot.

n_samples: Number of sample leaves.

packed_leaf_memberships: Packed update bits for each sample leaf.

descendant_lists: Reach D(n) for every graph node.

n_updates: Number of updates represented by the packed words.

word_size: Number of active bits per word.

Returns

-------

discovery: Packed node states, per-update candidate lists, candidate sizes, and visited nodes.

```python
import numpy as np


def discover_shared_candidates(
    children: list,
    n_samples: int,
    packed_leaf_memberships: np.ndarray,
    descendant_lists: list,
    n_updates: int,
    word_size: int = 32,
) -> dict:
    """Discover reuse candidates for a batch in one upward traversal.

    Parameters
    ----------
    children : list
        Ordered graph child lists.
    n_samples : int
        Number of sample leaves.
    packed_leaf_memberships : np.ndarray
        Packed leaf states with shape (n_samples, ceil(n_updates / word_size)).
    descendant_lists : list
        Numeric reach list for every node.
    n_updates : int
        Number of target carrier sets in the batch.
    word_size : int, optional
        Number of update bits in each packed word.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` does not define its leaf prefix, an
        internal node has no children, a leaf has children, or a child does not
        precede its parent. Also raised if `n_updates` is not positive,
        `word_size` is outside [1, 62], `descendant_lists` has the wrong length,
        the packed array has the wrong shape, a packed value is negative or
        non-integer, or a bit outside the batch is set.

    Returns
    -------
    discovery : dict
        Native numeric node states, candidates, candidate sizes, and visits.
    """
    return discovery  # noqa: F821
```

### Step 5

05_plan_greedy_attachments

Goal
----
A compatible node can be reused because its reach is contained in the target carrier set. An exact reach receives the mutation without a topology change. Otherwise, compatible nodes are considered by decreasing reach size and retained only when their reaches are disjoint from those already selected; any carriers left uncovered remain direct leaf children of a new mutation node.

Inputs

------

candidate_lists: Compatible internal node identifiers for every update.

descendant_lists: Reach D(n) for every snapshot node.

target_carriers: Binary target carrier matrix.

n_samples: Number of sample leaves.

Returns

-------

attachment_plan: Exact matches, selected candidates, uncovered leaves, target lists, and creation flags.

```python
import numpy as np


def plan_greedy_attachments(
    candidate_lists: list,
    descendant_lists: list,
    target_carriers: np.ndarray,
    n_samples: int,
) -> dict:
    """Plan deterministic reuse-aware attachments on a fixed snapshot.

    Parameters
    ----------
    candidate_lists : list
        Compatible internal node identifiers for each update.
    descendant_lists : list
        Reach list for each node in the discovery snapshot.
    target_carriers : np.ndarray
        Binary carrier matrix with one row per update.
    n_samples : int
        Number of sample leaves.

    Raises
    ------
    ValueError
        If `target_carriers` has the wrong shape, is empty, or is not binary;
        if `n_samples` is invalid; if the candidate count differs from the
        update count; if a target carrier set is empty; if a candidate
        identifier is out of range; or if a candidate reach is not contained
        in its target.

    Returns
    -------
    attachment_plan : dict
        Native numeric lists describing exact reuse or new-node children.
    """
    return attachment_plan  # noqa: F821
```

### Step 6

06_apply_snapshot_batch

Goal
----
Candidate discovery is separated from mutation application. Every update in one batch consumes a plan derived from the same read-only snapshot, so a new node created for an earlier update cannot become a candidate for a later update in that batch. Exact matches reuse a snapshot node, while other plans append one node in fixed update order.

Inputs

------

children: Ordered child lists for the discovery snapshot.

attachment_plan: Fixed-snapshot exact matches and new-node child plans.

n_samples: Number of sample leaves.

Returns

-------

updated_graph: Updated child lists, attachment nodes, new node identifiers, and edge count.

```python
from typing import Any


def apply_snapshot_batch(children: list, attachment_plan: dict, n_samples: int) -> dict[str, Any]:
    """Apply a fixed-snapshot attachment plan in deterministic order.

    Parameters
    ----------
    children : list
        Ordered child lists for the snapshot graph.
    attachment_plan : dict
        Exact nodes, selected candidates, uncovered leaves, and creation flags.
    n_samples : int
        Number of sample leaves.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` is invalid, required plan fields are
        missing, plan-field lengths differ, or `creates_node` is not binary.
        Also raised for an inconsistent exact-match plan, an out-of-range
        selected candidate, an invalid uncovered sample, or empty or repeated
        children in a new-node plan.

    Returns
    -------
    updated_graph : dict
        Updated graph and per-update attachment identifiers.
    """
    return updated_graph  # noqa: F821
```

### Step 7

07_trace_independent_visits

Goal
----
Independent mutation mapping runs the same upward compatibility recurrence once per target. A carrier leaf is compatible, an internal node is compatible only when every child is compatible, and only compatible nodes propagate to their parents. A reached internal node is counted as visited even when its failed compatibility test terminates that branch.

Inputs

------

children: Ordered child lists for the updated graph.

n_samples: Number of sample leaves.

carrier_matrix: Binary audit carrier sets, one row per independent traversal.

Returns

-------

traces: Visit matrix, visited counts, and compatible internal candidates.

```python
import numpy as np


def trace_independent_visits(children: list, n_samples: int, carrier_matrix: np.ndarray) -> dict:
    """Trace independent reuse-aware visits for several target carrier sets.

    Parameters
    ----------
    children : list
        Ordered child lists for the graph.
    n_samples : int
        Number of sample leaves.
    carrier_matrix : np.ndarray
        Binary target matrix with one row per independent traversal.

    Raises
    ------
    ValueError
        If the graph is empty, `n_samples` does not define its leaf prefix, a
        leaf has children, an internal node has no children, or a child does not
        precede its parent. Also raised if `carrier_matrix` has the wrong shape,
        is not binary, or contains an empty audit carrier set.

    Returns
    -------
    traces : dict
        Visit matrix, row counts, and per-target candidates.
    """
    return traces  # noqa: F821
```

### Step 8

08_compute_overlap_factor

Goal
----
Traversal overlap compares repeated independent work with the distinct nodes touched by a shared pass. For visit sets V_i, the overlap factor is rho(k) = sum_i |V_i| / |union_i V_i|. A batch using word width w propagates q = ceil(k/w) words per node, so rho(k) greater than q is the paper's asymptotic criterion for shared discovery to reduce traversal work.

Inputs

------

visit_matrix: Binary matrix whose rows identify independent visit sets.

word_size: Number of update bits in one packed word.

Returns

-------

overlap: Independent and distinct visit counts, word count, overlap factor, and advantage flag.

```python
import numpy as np


def compute_overlap_factor(visit_matrix: np.ndarray, word_size: int = 32) -> dict:
    """Compute traversal overlap and the packed-word advantage criterion.

    Parameters
    ----------
    visit_matrix : np.ndarray
        Binary matrix with one independent traversal per row.
    word_size : int, optional
        Number of update bits represented by one word.

    Raises
    ------
    ValueError
        If `visit_matrix` is empty, is not two dimensional, is not binary, or
        contains a row with no visits, or if `word_size` is outside [1, 62].

    Returns
    -------
    overlap : dict
        Visit counts, packed word count, overlap factor, and advantage flag.
    """
    return overlap  # noqa: F821
```

### Step 9

09_run_full_pipeline

Goal
----
The full calculation polarizes the first carrier batch, packs its leaf memberships, constructs adaptive node reaches, discovers snapshot candidates, plans and applies reuse-aware edits, traces an audit batch independently on the edited graph, and evaluates the audit overlap factor. The final scalar is rho(k) for those audit traversals.

Inputs

------

children: Ordered graph child lists.

n_samples: Number of sample leaves.

alternate_carriers: Binary alternate-carrier matrix for the edit batch.

observed_haplotypes: Binary observed-haplotype matrix for the edit batch.

ancestral_is_alternate: Binary polarization flags for the edit batch.

audit_carriers: Binary carrier matrix for independent post-edit traversals.

word_size: Number of update bits per packed word.

dense_threshold: Density threshold for dense reach storage.

Returns

-------

overlap_factor: float, the post-edit traversal overlap factor.

```python
import numpy as np


def run_full_pipeline(
    children: list,
    n_samples: int,
    alternate_carriers: np.ndarray,
    observed_haplotypes: np.ndarray,
    ancestral_is_alternate: np.ndarray,
    audit_carriers: np.ndarray,
    word_size: int = 32,
    dense_threshold: float = 0.03125,
) -> float:
    """Run batched editing and return the post-edit overlap factor.

    Parameters
    ----------
    children : list
        Ordered child lists for the initial graph.
    n_samples : int
        Number of sample leaves.
    alternate_carriers : np.ndarray
        Alternate-carrier matrix for the polarization batch.
    observed_haplotypes : np.ndarray
        Observed-haplotype matrix for the polarization batch.
    ancestral_is_alternate : np.ndarray
        Binary flags selecting carrier complementation.
    audit_carriers : np.ndarray
        Target carrier sets for post-edit independent traversals.
    word_size : int, optional
        Number of update bits per packed word.
    dense_threshold : float, optional
        Inclusive density threshold for dense reach storage.

    Raises
    ------
    ValueError
        If `dense_threshold` is outside [0, 1].

    Returns
    -------
    overlap_factor : float
        Audit traversal overlap factor rho(k).
    """
    return overlap_factor  # noqa: F821
```
