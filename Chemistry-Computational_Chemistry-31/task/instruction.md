# Chemistry-Computational_Chemistry-31

## Background

Enzyme databases contain many overall reactions but comparatively few complete, stepwise mechanisms. This gap limits systematic comparison of catalytic strategies and makes it harder to propose chemically plausible paths for reactions that have no mechanistic annotation.

Computational mechanism inference seeks to convert overall stoichiometry into testable multistep hypotheses. Common strategies represent local chemical transformations as reusable rules, enforce conservation and intermediate feasibility, and compare proposed paths with curated mechanistic knowledge before more expensive structural or quantum-chemical validation.

## Problem

A transferase-like reaction has been encoded as signed changes in local chemical-moiety counts, together with a library of elementary rules and a curated archive of arrow environments. Execute the deterministic NumPy block below once to construct the benchmark instance.

```python
import numpy as np

n_layers = 5
layer_width = 4
source_index = 0
layer_start = 1
target_index = layer_start + n_layers * layer_width
cycle_start = target_index + 1
catalyst_index = cycle_start + 6
n_moieties = catalyst_index + 1
columns = []

def add_rule(consumed, produced, catalyst_delta=0):
    column = np.zeros(n_moieties, dtype=int)
    for index, amount in consumed.items():
        column[index] -= amount
    for index, amount in produced.items():
        column[index] += amount
    column[-1] += catalyst_delta
    columns.append(column)

# A balanced six-rule support whose intermediates are initially unavailable.
for offset in range(5):
    add_rule({cycle_start + offset: 1}, {cycle_start + offset + 1: 1})
add_rule(
    {cycle_start + 5: 1, source_index: 1},
    {cycle_start: 1, target_index: 1},
)

# Six-step alternatives through five intermediate layers.
for node in range(layer_width):
    add_rule(
        {source_index: 1},
        {layer_start + node: 1},
        catalyst_delta=-1,
    )
for layer in range(n_layers - 1):
    left = layer_start + layer * layer_width
    right = left + layer_width
    for a in range(layer_width):
        for b in range(layer_width):
            add_rule({left + a: 1}, {right + b: 1})
final_layer = layer_start + (n_layers - 1) * layer_width
for node in range(layer_width):
    add_rule(
        {final_layer + node: 1},
        {target_index: 1},
        catalyst_delta=1,
    )

rule_matrix = np.stack(columns, axis=1)
permutation_rng = np.random.default_rng(26091331)
rule_permutation = np.concatenate(
    (np.arange(6), permutation_rng.permutation(np.arange(6, rule_matrix.shape[1])))
)
rule_matrix = rule_matrix[:, rule_permutation]
target = np.zeros(n_moieties, dtype=int)
target[source_index], target[target_index] = -1, 1
radii = np.array([1, 2, 3, 4], dtype=int)
changes_by_radius = np.stack(
    [np.zeros_like(target), np.zeros_like(target), target, 2 * target]
)
initial_counts = np.zeros(n_moieties, dtype=int)
initial_counts[source_index] = 1
exempt_moieties = np.zeros(n_moieties, dtype=bool)
exempt_moieties[catalyst_index] = True

arrow_rng = np.random.default_rng(26091343)
rule_arrow_matrix = np.zeros((rule_matrix.shape[1], 83), dtype=int)
for rule in range(rule_matrix.shape[1]):
    width = 4 + (3 * rule) % 5
    features = arrow_rng.choice(83, size=width, replace=False)
    rule_arrow_matrix[rule, features] = 1
reference_profiles = (arrow_rng.random((13, 83)) < 0.11).astype(int)
for reference in range(reference_profiles.shape[0]):
    reference_profiles[reference, (11 * reference + 5) % 83] = 1

max_steps = 6
top_k = 10
candidate_limit = 18
```

Use the primary source's decomposed minimum-rule, chemical-ordering, and unordered-similarity procedure to retain and re-rank ten feasible mechanisms. For deterministic choices left open by the source, compare expanded tuples of one-based rule labels lexicographically, use the lexicographically smallest feasible ordering of each multiset, and preserve the original parsimony order when re-ranking scores tie. In the reasoning audit, candidate numbering may use either one-based cut-search ranks assigned before infeasible candidates are removed or one-based retained-list positions assigned after filtering. State which convention you use and apply it consistently; both conventions are accepted. Return one hundred times the winning candidate's maximum archive similarity as a single finite decimal. In `<reasoning>`, explain how fixing the OrderRules step count to the sum of the minRules multiplicities makes it a feasibility-only sequencing stage and why its dummy objective cannot affect the solution. Using the official M-CSA records identified in the browsing sources, cross-check entry 355 by reporting its organism, UniProt accession, PDB identifier and resolution and linking that record to the paper’s empirical maximum-rule bound; reconstruct the nucleophile, electrophile, intermediate, and collapse product across entry 355 steps 1–2; assign the roles of Phe215, Asn336, Cys164, and His303 in step 3; and, for arrow environment 393883, report every occurrence in entries 355, 33, and 78 and explain what recurrence across those mechanisms shows about reuse of local chemistry. Then audit the selected adaptive resolution, optimal rule count, rejected balanced support, first feasible ordered mechanism, retained candidate ranks under the declared convention, union-profile cardinalities, stable similarity order and leading-score margin, and the winner’s rank under that convention, ordered rules, closest reference, overlap counts, and similarity.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise, but include every source fact and audit quantity explicitly requested above.
Do not paste the input matrices, full rule-count tables, search tree, or per-reference similarity table.

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

select_informative_radius

Goal
----
Select the first moiety radius that resolves a nonzero transformation.

```python
def select_informative_radius(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
) -> 'np.ndarray':
    """Select the first radius whose overall moiety-change vector is nonzero.

    Parameters
    ----------
    changes_by_radius
        Integer array with shape ``(n_radii, n_moieties)`` in increasing-radius
        order. Each row is the overall product-minus-reactant moiety change at
        that radius.
    radii
        Strictly increasing positive integer radii with shape ``(n_radii,)``.

    Returns
    -------
    np.ndarray
        Integer vector ``[selected_radius, *selected_change]``.

    Raises
    ------
    ValueError
        If the inputs are malformed or every supplied change vector is zero.
    """
    return result
```

### Step 2

rank_parsimonious_rule_sets

Goal
----
Rank exact moiety-balance rule multisets under iterative support cuts.

```python
def rank_parsimonious_rule_sets(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    max_steps: int,
    limit: int,
) -> 'np.ndarray':
    """Return the source-style cut-ranked rule-count vectors.

    A candidate is a nonnegative integer count vector whose rule columns sum
    exactly to ``target_change`` and whose total count does not exceed
    ``max_steps``. Select candidates iteratively by minimum total count. For an
    equal objective, compare their expanded tuples of one-based rule labels
    lexicographically.

    After selecting a candidate with support ``S``, every subsequently ranked
    integer count vector ``y`` must satisfy

    ``sum(y[r] for r in S) <= len(S) - 1``.

    This integer support cutoff excludes the previously selected support and
    every count vector containing that complete support before the next
    candidate is selected.

    Parameters
    ----------
    rule_matrix
        Integer moiety-change matrix with shape ``(n_moieties, n_rules)``.
    target_change
        Nonzero integer overall moiety-change vector.
    max_steps
        Positive upper bound on total rule multiplicity.
    limit
        Positive maximum number of ranked candidates to return.

    Returns
    -------
    np.ndarray
        Integer array with one row per candidate. Column zero is the total rule
        count and the remaining columns are counts for rules 1 through
        ``n_rules``. The shape is ``(n_found, n_rules + 1)``.

    Raises
    ------
    ValueError
        If dimensions, integer domains, or scalar bounds are invalid.
    """
    return result
```

### Step 3

order_rule_multiset

Goal
----
Order a selected rule multiset under prefix moiety availability.

```python
def order_rule_multiset(
    rule_counts: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
) -> 'np.ndarray':
    """Find the lexicographically smallest chemically feasible rule ordering.

    Rules are identified by one-based column labels. After every prefix, the
    initial counts plus cumulative rule changes must be nonnegative for every
    non-exempt moiety. Labeled catalytic/cofactor moieties marked ``True`` in
    ``exempt_moieties`` do not participate in this availability test. Every
    selected rule copy must be used exactly once.

    Returns
    -------
    np.ndarray
        One-dimensional integer array of one-based rule labels. Return an empty
        integer array when the balanced multiset has no feasible ordering.

    Raises
    ------
    ValueError
        If shapes, integer counts, or initial availability are invalid.
    """
    return result
```

### Step 4

generate_parsimonious_mechanisms

Goal
----
Generate the first feasible mechanisms from cut-ranked rule sets.

```python
def generate_parsimonious_mechanisms(
    rule_matrix: 'np.ndarray',
    target_change: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> 'np.ndarray':
    """Filter cut-ranked balanced multisets through chemical ordering.

    Obtain up to ``candidate_limit`` rule-count vectors from
    ``rank_parsimonious_rule_sets``. Test them in that order with
    ``order_rule_multiset``, skip every multiset without a feasible ordering,
    and retain at most ``top_k`` mechanisms.

    Returns
    -------
    np.ndarray
        Integer array with ``max_steps + 2`` columns. Each row stores the
        one-based rank of its balanced candidate, its step count, and its
        one-based ordered rule labels padded on the right with zeros.

    Raises
    ------
    ValueError
        If scalar search bounds are not positive integers, an array violates
        an upstream step contract, or an ordered result cannot fit the
        declared maximum length.
    """
    return result
```

### Step 5

build_arrow_environment_profiles

Goal
----
Collapse ordered mechanisms into unique arrow-environment profiles.

```python
def build_arrow_environment_profiles(
    mechanisms: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
) -> 'np.ndarray':
    """Build one binary union profile for every mechanism.

    ``mechanisms`` uses the row layout returned by
    ``generate_parsimonious_mechanisms``. ``rule_arrow_matrix`` has one row per
    one-based rule label and one column per possible arrow environment. Rule
    repetition does not duplicate an environment because each mechanism is
    represented by a set union.

    Returns
    -------
    np.ndarray
        Binary integer array with shape ``(n_mechanisms, n_environments)``.

    Raises
    ------
    ValueError
        If an input is malformed or a used rule label is out of range.
    """
    return result
```

### Step 6

maximum_archive_similarity

Goal
----
Score candidate arrow profiles against a curated mechanism archive.

```python
def maximum_archive_similarity(
    candidate_profiles: 'np.ndarray',
    reference_profiles: 'np.ndarray',
) -> 'np.ndarray':
    """Find each candidate's largest unordered archive similarity.

    Treat nonzero entries as set membership. Ignore empty reference profiles.
    For every candidate-reference pair, use intersection size divided by union
    size. If several references maximize the score, choose the smallest
    one-based reference label.

    Returns
    -------
    np.ndarray
        Float array with four columns: maximum similarity, best one-based
        reference label, intersection size, and union size.

    Raises
    ------
    ValueError
        If profile shapes are incompatible, entries are non-finite, or the
        archive contains no nonempty reference.
    """
    return result
```

### Step 7

rerank_mechanisms

Goal
----
Re-rank parsimonious mechanisms by maximum archive similarity.

```python
def rerank_mechanisms(
    mechanisms: 'np.ndarray',
    similarity_summary: 'np.ndarray',
) -> 'np.ndarray':
    """Sort mechanisms by descending similarity with a stable parsimony tie.

    Preserve input order when similarity values tie. The input mechanism table
    follows ``generate_parsimonious_mechanisms`` and the summary follows
    ``maximum_archive_similarity``.

    Returns
    -------
    np.ndarray
        Float table whose columns are similarity, original balanced-candidate
        rank, best reference label, intersection size, union size, step count,
        and the padded one-based rule sequence.

    Raises
    ------
    ValueError
        If the two tables are malformed, incompatible, or contain invalid
        similarity values.
    """
    return result
```

### Step 8

adaptive_moiety_mechanism_similarity

Goal
----
Orchestrate adaptive moiety-rule search and archive re-ranking.

```python
def adaptive_moiety_mechanism_similarity(
    changes_by_radius: 'np.ndarray',
    radii: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
    rule_arrow_matrix: 'np.ndarray',
    reference_profiles: 'np.ndarray',
    max_steps: int,
    top_k: int,
    candidate_limit: int,
) -> float:
    """Run the seven preceding stages and return the winning similarity percent.

    The routine must call the preceding public functions. It selects the first
    informative radius, generates the requested parsimonious feasible
    mechanisms, constructs their arrow-environment profiles, scores and
    re-ranks them, and returns one hundred times the first row's similarity.

    Returns
    -------
    float
        Percentage equal to ``100`` times the winning maximum archive
        similarity.

    Raises
    ------
    ValueError
        If the supplied instance yields no feasible mechanism or an earlier
        public contract is violated.
    """
    return result
```
