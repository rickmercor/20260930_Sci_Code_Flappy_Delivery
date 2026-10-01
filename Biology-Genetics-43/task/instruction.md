# Biology-Genetics-43

## Background

Designing an RNA sequence that folds into a prescribed secondary structure is a basic operation of synthetic biology and RNA engineering, and its simplest formal version asks for a sequence whose unique maximum-base-pair structure is the target. Even in that base-pair-maximization model the existence of a design is computationally hard to decide in general, so designers rely on local combinatorial certificates that guarantee a sequence folds as intended, and on samplers that draw sequences uniformly from a certified family. The nucleotide composition of such families, notably their G+C content, matters in practice because it governs helix stability and the folding kinetics of the designed molecules.

## Problem

I design RNA sequences for target secondary structures under base-pair maximization: G-C, A-U and G-U pairs are allowed, hairpin loops may have any size, and a sequence is a design when the target is its unique maximum-pair structure. My 90-nucleotide target, in dot-bracket notation, is `((.....))((..((...((((.((......))))((((.....)))(((...))))))...)).))(((.(((((......))))))))`. I draw candidates only from the level-separation certificate of the tree-coloring framework for this model, taken modulo m: every unpaired position is A, every pair is G-C, C-G, A-U or U-A, the coloring is proper in every loop with the orientation constraints on A-U-type pairs applied to stacked pairs as well, and no A-U-type pair lies on a level that an unpaired position occupies modulo m. Let m* be the smallest m ≥ 2 for which my target has at least one such sequence. I want the expected G+C content, as a fraction of all 90 nucleotides, of a sequence drawn uniformly from the distinct sequences that are level-separated modulo m*, which is the distribution the published rejection sampler for a fixed modulus is built to deliver; report it to at least six significant figures.

In `<reasoning>`, identify the primary source paper and DOI, and state the paper's time complexity for the fixed-residue-set decision procedure and its average-case complexity for the complete uniform modulo-separated generator. Also state m*; the number of distinct sequences level-separated modulo m*; how many of the 2^m* residue sets that can be reserved for unpaired positions carry at least one sequence; the sampler normalizer at m*; the acceptance probability of the published sampler at m*, to at least four significant figures, with the acceptance rule it applies to each proposal; the orientation rule you applied to A-U-type pairs; the length of the shortest helix in my target and whether the published guarantee of a modulus-2 design covers it; the aggregate G+C nucleotide count over the distinct sequences; and the expected G+C fraction. These are the quantities needed to identify the method, verify the uniform-sampling correction, and determine the final number. Do not restate supplied input data or give bulk per-residue-set tables.

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

01_build_loop_tree

Goal
----
Convert a dot-bracket secondary structure into its loop table, one row for the exterior loop and one row per base pair.

```python
def build_loop_tree(structure: str, min_hairpin: int = 0) -> np.ndarray:
    """Return the loop table of a dot-bracket secondary structure.

    Positions are numbered 1 to ``n``. Row 0 describes the exterior loop and
    rows ``1..P`` describe the ``P`` base pairs in increasing order of their
    5' position. Row ``k`` holds ``[parent, i, j, u]``: the 5' and 3'
    positions ``i < j`` of the pair, the row ``parent`` of the innermost pair
    enclosing it (0 when no pair encloses it), and the number ``u`` of
    unpaired positions whose innermost enclosing pair is this one. Row 0 is
    ``[-1, 0, n + 1, u0]`` with ``u0`` the number of unpaired positions
    enclosed by no pair.

    Parameters
    ----------
    structure : str
        Non-empty string over ``'('``, ``')'`` and ``'.'``.
    min_hairpin : int
        Nonnegative minimum number of positions strictly between the two
        positions of every pair.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(P + 1, 4)``.

    Raises
    ------
    ValueError
        If ``structure`` is not a non-empty string over the three symbols,
        if its brackets are unbalanced, if ``min_hairpin`` is not a
        nonnegative integer (booleans are rejected), or if some pair encloses
        fewer than ``min_hairpin`` positions.
    """
    return table
```

### Step 2

02_profile_helix_motifs

Goal
----
Measure the helix lengths of a loop tree and count its loops that carry either of the two locally undesignable motifs of base-pair maximization.

```python
def profile_helix_motifs(tree: np.ndarray) -> np.ndarray:
    """Return the helix and forbidden-motif profile of a loop tree.

    ``tree`` is a loop table with rows ``[parent, i, j, u]`` as returned by
    ``build_loop_tree``: row 0 is the exterior loop, row ``k >= 1`` a base
    pair whose innermost enclosing pair is row ``parent`` (0 for none) and
    which directly encloses ``u`` unpaired positions. The pairs directly
    enclosed by a loop are the rows whose ``parent`` is that loop.

    A helix is a maximal chain of pairs ``v_1, ..., v_L`` in which every
    ``v_a`` with ``a < L`` directly encloses exactly one pair, ``v_{a+1}``,
    and no unpaired position; ``L`` is its length. A loop's pair count is the
    number of pairs it directly encloses, plus one for its closing pair when
    it is not the exterior loop. A loop carries the five-pair motif when its
    pair count is at least 5, and the three-pair motif when its pair count is
    at least 3 and it directly encloses at least one unpaired position.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)``.

    Returns
    -------
    np.ndarray
        Integer array ``[shortest helix length, number of helices, number of
        loops with the five-pair motif, number of loops with the three-pair
        motif]``; the shortest helix length is 0 when ``P = 0``.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        whose row 0 has parent ``-1``, whose every row has a nonnegative
        unpaired count, and whose every row ``k >= 1`` has a parent in
        ``[0, k - 1]``.
    """
    return profile
```

### Step 3

03_list_proper_child_contents

Goal
----
List every assignment of pair contents to the pairs directly enclosed by one loop that keeps the loop proper, given the content of its closing pair.

```python
def list_proper_child_contents(parent_content: int, n_children: int) -> np.ndarray:
    """Return the proper content tuples for the pairs directly enclosed by a loop.

    Pair contents are coded ``0`` = G-C, ``1`` = C-G, ``2`` = A-U and
    ``3`` = U-A, the first base being the 5' base of the pair. A pair is
    black for G-C, white for C-G and gray for A-U or U-A; the complement of
    black is white, of white is black and of gray is gray. ``parent_content``
    is the content of the loop's closing pair, or ``-1`` for the exterior
    loop, which has no closing pair.

    The loop is proper when two conditions hold. First, the list made of the
    complement of the closing pair's color followed by the colors of the
    enclosed pairs (for the exterior loop, only the enclosed pairs' colors)
    holds at most one black, at most one white and at most two gray entries.
    Second, a gray enclosed pair has the same content as a gray closing pair,
    and two gray enclosed pairs have different contents.

    Parameters
    ----------
    parent_content : int
        Content code of the closing pair in ``{0, 1, 2, 3}``, or ``-1``.
    n_children : int
        Nonnegative number of pairs directly enclosed by the loop.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(K, n_children)`` holding every content
        tuple of the enclosed pairs (in 5'-to-3' order) that makes the loop
        proper, one per row, rows in increasing lexicographic order. When
        ``n_children`` is 0 it holds one empty row; when no tuple is proper
        it has no rows.

    Raises
    ------
    ValueError
        If ``parent_content`` is not an integer in ``{-1, 0, 1, 2, 3}`` or
        ``n_children`` is not a nonnegative integer (booleans are rejected).
    """
    return contents
```

### Step 4

04_count_restricted_designs

Goal
----
Count the proper sequences of a loop tree whose unpaired positions and A-U-type pairs fall on prescribed residue classes of their levels, together with their total G+C content.

```python
def count_restricted_designs(
    tree: np.ndarray,
    modulus: int,
    leaf_residues: "Sequence[int]",
    gray_residues: "Sequence[int]",
) -> np.ndarray:
    """Return the number of restricted proper sequences and their total G+C count.

    ``tree`` is a loop table with rows ``[parent, i, j, u]`` as returned by
    ``build_loop_tree`` (row 0 the exterior loop, row ``k >= 1`` a pair whose
    innermost enclosing pair is row ``parent``, 0 for none, and which
    directly encloses ``u`` unpaired positions). A sequence places A at every
    unpaired position and one content on every pair (``0`` = G-C, ``1`` =
    C-G, ``2`` = A-U, ``3`` = U-A), such that every loop is proper in the
    sense of ``list_proper_child_contents``.

    A pair shifts the level of everything it encloses by ``+1`` if it is
    G-C, ``-1`` if it is C-G and ``0`` if it is A-U or U-A. The level of a
    pair is the sum of the shifts of the pairs strictly enclosing it; the
    level of an unpaired position is the sum of the shifts of all pairs
    enclosing it (0 in the exterior loop). A sequence is counted when every
    unpaired position has its level modulo ``modulus`` in ``leaf_residues``
    and every A-U or U-A pair has its level modulo ``modulus`` in
    ``gray_residues``.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)``.
    modulus : int
        Positive modulus applied to levels (Python ``%`` convention, so
        residues lie in ``[0, modulus)``).
    leaf_residues : sequence of int
        Residues allowed for unpaired positions; may be empty.
    gray_residues : sequence of int
        Residues allowed for A-U and U-A pairs; may be empty.

    Returns
    -------
    np.ndarray
        Integer array ``[count, gc_total]``: the number of counted sequences
        and the number of G and C nucleotides summed over all of them.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        whose row 0 has parent ``-1``, whose every row has a nonnegative
        unpaired count, and whose every row ``k >= 1`` has a parent in
        ``[0, k - 1]``, if ``modulus`` is not a positive integer, or if a
        residue is not an integer in ``[0, modulus)`` (booleans are rejected
        throughout).
    """
    return counts
```

### Step 5

05_compute_sampler_normalizer

Goal
----
Sum, over every set of residues reserved for unpaired positions, the number of proper sequences whose A-U-type pairs use only the remaining residues, and count the sets that admit at least one sequence.

```python
def compute_sampler_normalizer(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Return the per-residue-set sequence total and the number of admissible sets.

    For a residue set ``S`` contained in ``{0, ..., modulus - 1}``, let
    ``F(S)`` be the count returned by ``count_restricted_designs(tree,
    modulus, S, T)`` with ``T`` the complement of ``S`` in ``{0, ...,
    modulus - 1}``: the number of proper sequences whose unpaired positions
    all have level residues in ``S`` and whose A-U and U-A pairs all have
    level residues outside ``S``. Return the sum of ``F(S)`` over all
    ``2 ** modulus`` sets ``S`` (the empty set and the full set included)
    and the number of sets with ``F(S) > 0``.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    modulus : int
        Positive modulus applied to levels.

    Returns
    -------
    np.ndarray
        Integer array ``[total, admissible]``.

    Raises
    ------
    ValueError
        If ``modulus`` is not a positive integer (booleans are rejected) or
        ``tree`` is not a valid loop table in the sense of
        ``count_restricted_designs``.
    """
    return normalizer
```

### Step 6

06_find_minimal_modulus

Goal
----
Find the smallest modulus of at least 2 at which a loop tree admits a sequence whose A-U-type pairs and unpaired positions occupy disjoint level residues.

```python
def find_minimal_modulus(tree: np.ndarray, helix_profile: np.ndarray, max_modulus: int) -> int:
    """Return the smallest workable modulus of a loop tree.

    ``helix_profile`` is the array ``[shortest helix length, number of
    helices, loops with the five-pair motif, loops with the three-pair
    motif]`` returned by ``profile_helix_motifs(tree)``. If it reports any
    loop with either motif, the structure has no design and ``ValueError``
    is raised. If there is at least one helix and every helix has at least
    three pairs, a sequence is guaranteed at modulus 2 and 2 is returned
    without further computation. Otherwise return the smallest ``m`` in
    ``[2, max_modulus]`` for which the total returned by
    ``compute_sampler_normalizer(tree, m)`` is positive.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    helix_profile : np.ndarray
        Length-4 array of nonnegative integers from ``profile_helix_motifs``.
    max_modulus : int
        Largest modulus tried; at least 2.

    Returns
    -------
    int
        The smallest workable modulus.

    Raises
    ------
    ValueError
        If ``helix_profile`` is not a length-4 array of nonnegative
        integers, if ``max_modulus`` is not an integer of at least 2
        (booleans are rejected), if the profile reports a loop with either
        motif, if no modulus up to ``max_modulus`` admits a sequence, or,
        when the modulus scan runs, if ``tree`` is not a valid loop table in
        the sense of ``count_restricted_designs``.
    """
    return modulus
```

### Step 7

07_count_distinct_designs

Goal
----
Count the distinct proper sequences whose unpaired positions and A-U-type pairs occupy disjoint level residues modulo m, and total their G+C content.

```python
def count_distinct_designs(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Return the number of distinct separated sequences and their total G+C count.

    Sequences, levels and residues are as in ``count_restricted_designs``:
    A at every unpaired position, one content per pair, every loop proper,
    and levels taken modulo ``modulus``. A sequence is separated modulo
    ``modulus`` when no residue is shared by an unpaired position and an
    A-U or U-A pair, i.e. the set of level residues of its unpaired
    positions and the set of level residues of its A-U-type pairs are
    disjoint. Count every separated sequence exactly once and total the
    number of G and C nucleotides over them.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    modulus : int
        Positive modulus applied to levels.

    Returns
    -------
    np.ndarray
        Integer array ``[count, gc_total]`` over the distinct separated
        sequences.

    Raises
    ------
    ValueError
        If ``modulus`` is not a positive integer (booleans are rejected) or
        ``tree`` is not a valid loop table in the sense of
        ``count_restricted_designs``.
    """
    return counts
```

### Step 8

08_estimate_design_gc_content

Goal
----
Compose every earlier step to obtain the expected G+C fraction of a sequence drawn uniformly from the separated sequences of a target structure at its smallest workable modulus.

```python
def estimate_design_gc_content(
    structure: str = "((.....))((..((...((((.((......))))((((.....)))(((...))))))...)).))(((.(((((......))))))))",
    max_modulus: int = 9,
) -> float:
    """Return the expected G+C fraction of a uniform separated sequence at the smallest workable modulus.

    Build the loop table of ``structure`` (no minimum hairpin size), take
    its helix and motif profile, and find the smallest modulus ``m* >= 2``
    (up to ``max_modulus``) at which a separated sequence exists. Among the
    distinct sequences separated modulo ``m*`` (as in
    ``count_distinct_designs``), each equally likely, return the expected
    number of G and C nucleotides divided by the length ``n`` of
    ``structure``. The defaults reproduce the problem statement.

    Parameters
    ----------
    structure : str
        Dot-bracket target structure.
    max_modulus : int
        Largest modulus tried; at least 2.

    Returns
    -------
    float
        Expected G+C fraction of a uniformly drawn separated sequence.

    Raises
    ------
    ValueError
        If ``structure`` is not a valid dot-bracket string, if a loop
        carries a locally undesignable motif, if ``max_modulus`` is not an
        integer of at least 2, or if no modulus up to ``max_modulus`` admits
        a separated sequence.
    """
    return 0.0
```
