# Biology-Biochemistry-39

## Background

RNA inverse folding seeks nucleotide sequences whose optimal secondary structure is a prescribed target. Even when every base pair is scored equally, the problem is computationally hard in general, which has motivated structural criteria that certify families of sequences as unique optimal folds while leaving freedom to tune their nucleotide composition.

## Problem

I am designing RNA sequences for one target secondary structure in the pure base-pair-maximization model: G–C, A–U and G–U pairs each score one, structures are pseudoknot-free, and any two positions may pair however close they are. I only use sequences certified by the modular level-separation criterion for designs in this model, so every sequence I use has the target as its unique optimal structure: A sits at every unpaired position, each base pair is written G·C, C·G, A·U or U·A, and the modulus is the smallest value at which my target admits at least one certified sequence. On the certified sequences w I place the distribution proportional to exp(π·N_GC(w)), where N_GC(w) is the number of G and C nucleotides in w, and I want the value of π at which the expected G+C fraction of the whole sequence is exactly 0.5. The target is 98 nucleotides long; its dot-bracket string, written 5′ to 3′ and given on its own line with nothing before or after it, is

`((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))`

Report π to ten significant figures.

In `<reasoning>`, the scalars and statements I need you to state are: the certification rules you applied; the modulus and why no smaller modulus serves this target; the number of certified sequences; the expected G+C fraction at π = 0; the value the expected G+C fraction approaches as π grows without bound and how many certified sequences have that composition; why only one π meets the target; and π. Those are the derived quantities that determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating supplied input data, which this problem does not need.

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

01_build_loop_tree

Goal
----
Convert a dot-bracket secondary structure into the loop tree of its base pairs and exterior loop, flagging every loop that holds unpaired positions.

```python
def build_loop_tree(dot_bracket: str) -> "np.ndarray":
    """Return the loop tree of a pseudoknot-free secondary structure.

    Positions are 0-based and ``n`` is the length of ``dot_bracket``. With
    ``k`` base pairs the result has ``k + 1`` rows. Rows ``0 .. k - 1``
    describe the base pairs in increasing order of their 5' position, each
    as ``[i, j, parent, unpaired]``: ``i < j`` are the paired positions,
    ``parent`` is the row of the closest pair enclosing ``(i, j)`` or ``k``
    when no pair encloses it, and ``unpaired`` is 1 when some unpaired
    position lies strictly between ``i`` and ``j`` but inside no pair nested
    in ``(i, j)``, else 0. Row ``k`` is the exterior loop
    ``[-1, n, -1, unpaired]``, whose flag is 1 when some unpaired position
    lies inside no pair, else 0.

    Parameters
    ----------
    dot_bracket : str
        Structure written with ``(``, ``)`` and ``.``.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(k + 1, 4)``.

    Raises
    ------
    ValueError
        If ``dot_bracket`` is not a non-empty string, contains a character
        other than ``(``, ``)`` and ``.``, or has unbalanced brackets.
    """
    return tree
```

### Step 2

02_screen_design_obstructions

Goal
----
Count the two loop motifs that make a target undesignable under base-pair maximization and report the shortest helix of the loop tree.

```python
def screen_design_obstructions(tree: "np.ndarray") -> "np.ndarray":
    """Return the forbidden-motif counts and the shortest helix length.

    ``tree`` is a loop tree as returned by ``build_loop_tree``: rows
    ``[i, j, parent, unpaired]`` for the ``k`` base pairs in 5' order, then
    the exterior-loop row ``[-1, n, -1, unpaired]``. The children of a row
    are the base-pair rows naming it as ``parent``. Return
    ``[five_pair, three_pair_unpaired, shortest_helix]``:

    * ``five_pair`` counts pair-closed loops with four or more children and
      exterior loops with five or more children;
    * ``three_pair_unpaired`` counts pair-closed loops with two or more
      children and exterior loops with three or more children when the
      loop's ``unpaired`` flag is 1;
    * ``shortest_helix`` is the smallest number of base pairs in a helix,
      a maximal chain of base pairs in which every pair except the last has
      exactly one child and an ``unpaired`` flag of 0; it is 0 when ``k`` is
      0.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)``.

    Returns
    -------
    np.ndarray
        Integer array ``[five_pair, three_pair_unpaired, shortest_helix]``.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        and at least one row, if its last row does not have ``-1`` in columns
        0 and 2, if an ``unpaired`` flag is not 0 or 1, or if a base-pair row
        names a parent that is neither an earlier base-pair row nor the
        exterior row.
    """
    return screen
```

### Step 3

03_enumerate_proper_loop_assignments

Goal
----
List every assignment of pair identities to the child pairs of one loop that leaves the loop locally proper for a given closing pair.

```python
def enumerate_proper_loop_assignments(closing_code: int, n_children: int) -> "np.ndarray":
    """Return the proper pair identities for the children of one loop.

    Pair identities are coded by their 5' nucleotide first: 0 is G-C,
    1 is C-G, 2 is A-U and 3 is U-A. Codes 0 and 1 are the two strong
    colours and codes 2 and 3 share the weak colour; the complement of a
    colour exchanges the two strong colours and keeps the weak one.
    ``closing_code`` is the identity of the pair closing the loop, or ``-1``
    for the exterior loop, which has no closing pair.

    An assignment of identities to the ``n_children`` child pairs, listed in
    5' order, is proper when both conditions hold:

    * the colour list made of the complement of the closing pair's colour
      (omitted for the exterior loop) and the colours of the children holds
      at most one G-C, at most one C-G and at most two weak entries;
    * two weak children have different identities, and a weak child of a
      weak closing pair has the same identity as the closing pair.

    Return all proper assignments as rows in increasing lexicographic order,
    the first child being the most significant.

    Parameters
    ----------
    closing_code : int
        Identity 0, 1, 2 or 3 of the closing pair, or -1 for the exterior
        loop.
    n_children : int
        Number of child pairs, between 0 and 8 inclusive.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(r, n_children)``; ``r`` is 1 when
        ``n_children`` is 0 and may be 0 when no assignment is proper.

    Raises
    ------
    ValueError
        If ``closing_code`` is not an integer in ``{-1, 0, 1, 2, 3}`` or
        ``n_children`` is not an integer from 0 to 8 (booleans are rejected
        for both).
    """
    return assignments
```

### Step 4

04_count_level_constrained_designs

Goal
----
Count the locally proper designs of a loop tree whose unpaired positions and weak pairs fall on prescribed level residues, resolved by the number of strong pairs.

```python
def count_level_constrained_designs(
    tree: "np.ndarray",
    modulus: int,
    leaf_mask: int,
    gray_mask: int,
) -> "np.ndarray":
    """Return level-constrained proper design counts by number of strong pairs.

    ``tree`` is a loop tree from ``build_loop_tree`` with ``k`` base pairs.
    A design writes A at every unpaired position and gives every base pair
    an identity code (0 G-C, 1 C-G, 2 A-U, 3 U-A, 5' nucleotide first). It
    is proper when, in every loop including the exterior loop, the
    identities of the child pairs in 5' order form one of the rows that
    ``enumerate_proper_loop_assignments`` returns for the identity of the
    loop's closing pair (``-1`` for the exterior loop).

    The level of a base pair is the number of G-C pairs minus the number of
    C-G pairs among the pairs strictly enclosing it. The level of an
    unpaired position is that difference over all pairs enclosing the
    position. The residue of a level is its value reduced into
    ``[0, modulus)``, so a level of -1 has residue ``modulus - 1``.

    Count the proper designs in which every unpaired position has a residue
    ``r`` whose bit ``2**r`` is set in ``leaf_mask`` and every A-U or U-A
    pair has a residue whose bit is set in ``gray_mask``. Entry ``g`` of the
    result is the number of those designs with exactly ``g`` pairs coded
    G-C or C-G.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` with ``k`` at most 30.
    modulus : int
        Level modulus, from 1 to 8.
    leaf_mask : int
        Allowed residues of unpaired positions, in ``[0, 2**modulus)``.
    gray_mask : int
        Allowed residues of A-U and U-A pairs, in ``[0, 2**modulus)``.

    Returns
    -------
    np.ndarray
        Integer array of length ``k + 1``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8, if a mask is not an
        integer in ``[0, 2**modulus)`` (booleans are rejected), if ``tree``
        has more than 30 base pairs, if the tree contains a five-pair loop
        or a three-pair loop with an unpaired position, or if
        ``screen_design_obstructions`` rejects the tree representation.
    """
    return counts
```

### Step 5

05_tabulate_assignment_counts

Goal
----
Tabulate, for every modular assignment of residues to unpaired positions, the proper designs whose weak pairs avoid those residues, resolved by the number of strong pairs.

```python
def tabulate_assignment_counts(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Return the separated design counts of every modular assignment.

    For each assignment ``s`` in ``0 .. 2**modulus - 1``, read as a set of
    residues through its bits, row ``s`` of the result is
    ``count_level_constrained_designs(tree, modulus, s, gray)`` with
    ``gray`` the complementary set ``2**modulus - 1 - s``: the proper
    designs whose unpaired positions all have residues in ``s`` and whose
    A-U and U-A pairs all have residues outside ``s``, indexed by the number
    of G-C and C-G pairs.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    modulus : int
        Level modulus, from 1 to 8.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(2**modulus, k + 1)``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8 (booleans are rejected)
        or if ``count_level_constrained_designs`` rejects ``tree``.
    """
    return table
```

### Step 6

06_find_minimal_modulus

Goal
----
Find the smallest level modulus at which the target admits at least one separated design.

```python
def find_minimal_modulus(tree: "np.ndarray", max_modulus: int) -> int:
    """Return the smallest modulus with a nonzero separated design count.

    Scan ``modulus = 1, 2, ..., max_modulus`` and return the first modulus
    for which ``tabulate_assignment_counts(tree, modulus)`` has a nonzero
    entry.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    max_modulus : int
        Largest modulus to try, from 1 to 8.

    Returns
    -------
    int
        The smallest admissible modulus.

    Raises
    ------
    ValueError
        If ``max_modulus`` is not an integer from 1 to 8 (booleans are
        rejected), if no modulus up to ``max_modulus`` admits a separated
        design, or if ``tabulate_assignment_counts`` rejects ``tree``.
    """
    return modulus
```

### Step 7

07_count_distinct_separated_designs

Goal
----
Count the distinct designs that are separated for at least one modular assignment, resolved by the number of strong pairs.

```python
def count_distinct_separated_designs(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Return the number of distinct separated designs by number of strong pairs.

    A proper design (as defined for ``count_level_constrained_designs``) is
    separated at ``modulus`` when no residue is shared by an unpaired
    position and an A-U or U-A pair, that is, when it is counted in at least
    one row of ``tabulate_assignment_counts(tree, modulus)``. Entry ``g`` of
    the result is the number of distinct separated designs with exactly
    ``g`` pairs coded G-C or C-G, each design counted once.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    modulus : int
        Level modulus, from 1 to 8.

    Returns
    -------
    np.ndarray
        Integer array of length ``k + 1``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8 (booleans are rejected)
        or if ``count_level_constrained_designs`` rejects ``tree``.
    """
    return distinct
```

### Step 8

08_evaluate_gc_ensemble

Goal
----
Evaluate the expected G+C fraction of GC-weighted distinct designs and the mean number of proposals an exact rejection sampler spends per accepted design.

```python
def evaluate_gc_ensemble(
    distinct_counts: "np.ndarray",
    assignment_counts: "np.ndarray",
    length: int,
    weight: float,
) -> "np.ndarray":
    """Return the expected G+C fraction and the proposals per accepted design.

    ``distinct_counts[g]`` is the number of distinct designs with ``g`` G-C
    or C-G pairs; ``assignment_counts[s, g]`` is the number of those designs
    counted in assignment row ``s``, where a design may be counted in several
    rows and every design is counted in at least one. Unpaired positions are
    A, so a design with ``g`` strong pairs holds ``2 g`` G and C nucleotides
    among ``length`` positions.

    Give each distinct design the probability proportional to
    ``exp(weight * N)``, with ``N`` its number of G and C nucleotides, and
    return ``[fraction, proposals]``:

    * ``fraction`` is the expected value of ``N / length``;
    * ``proposals`` is the expected number of proposals per accepted design
      of the sampler that picks a row ``s`` with probability proportional to
      the ``exp(weight * N)``-weighted count of row ``s``, draws a design
      counted in row ``s`` with probability proportional to
      ``exp(weight * N)``, accepts it with probability one over the number
      of rows counting it, and otherwise starts again.

    Both values must hold to a relative ``1e-12`` for every ``|weight|`` up
    to 50, where single Boltzmann factors exceed double precision.

    Parameters
    ----------
    distinct_counts : np.ndarray
        Non-negative counts of length ``k + 1``, not all zero.
    assignment_counts : np.ndarray
        Non-negative counts of shape ``(rows, k + 1)``.
    length : int
        Sequence length, at least ``max(1, 2 k)``.
    weight : float
        Boltzmann weight per G or C nucleotide, with ``|weight| <= 50``.

    Returns
    -------
    np.ndarray
        Float array ``[fraction, proposals]``.

    Raises
    ------
    ValueError
        If ``distinct_counts`` is not a non-empty one-dimensional array of
        finite non-negative numbers with a positive entry, if
        ``assignment_counts`` is not a two-dimensional array of finite
        non-negative numbers with ``k + 1`` columns, if a column of
        ``assignment_counts`` sums to less than the matching entry of
        ``distinct_counts``, if ``length`` is not an integer of at least
        ``max(1, 2 k)``, or if ``weight`` is not a finite real number with
        ``|weight| <= 50`` (booleans are rejected).
    """
    return summary
```

### Step 9

09_calibrate_gc_weight

Goal
----
Bisect the GC Boltzmann weight until the expected G+C fraction of the weighted distinct designs meets a target.

```python
def calibrate_gc_weight(
    distinct_counts: "np.ndarray",
    length: int,
    target_fraction: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    """Return the GC weight whose expected G+C fraction equals the target.

    Let ``r(w)`` be the ``fraction`` entry of ``evaluate_gc_ensemble``
    called with ``distinct_counts``, a one-row table equal to
    ``distinct_counts``, ``length`` and weight ``w``, minus
    ``target_fraction``. Require ``r(lower) < 0 < r(upper)``. Then repeat:
    take the midpoint of the bracket, replace ``lower`` by it when ``r`` is
    negative there and ``upper`` otherwise, until
    ``upper - lower <= tolerance``; return the midpoint of the final bracket.

    Parameters
    ----------
    distinct_counts : np.ndarray
        Non-negative design counts indexed by the number of G-C and C-G
        pairs, as accepted by ``evaluate_gc_ensemble``.
    length : int
        Sequence length.
    target_fraction : float
        Target expected G+C fraction, strictly between 0 and 1.
    lower : float
        Lower end of the weight bracket, at least -50.
    upper : float
        Upper end of the weight bracket, above ``lower`` and at most 50.
    tolerance : float
        Final bracket width, from ``1e-14`` to ``1e-6`` inclusive.

    Returns
    -------
    float
        The calibrated weight.

    Raises
    ------
    ValueError
        If ``target_fraction``, ``lower``, ``upper`` or ``tolerance`` is not
        a finite real number (booleans are rejected), if ``target_fraction``
        is not strictly between 0 and 1, if the bracket does not satisfy
        ``-50 <= lower < upper <= 50``, if ``tolerance`` is not in
        ``[1e-14, 1e-6]``, if ``r(lower) < 0 < r(upper)`` fails, or if
        ``evaluate_gc_ensemble`` rejects its inputs.
    """
    return weight
```

### Step 10

10_estimate_gc_weight

Goal
----
Compose every earlier step to find the GC weight at which the distinct separated designs of a target, at its smallest admissible modulus, reach a target expected G+C fraction.

```python
def estimate_gc_weight(
    dot_bracket: str = "((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))",
    target_fraction: float = 0.5,
    max_modulus: int = 8,
    tolerance: float = 1e-12,
) -> float:
    """Return the GC weight that gives the target expected G+C fraction.

    Build the loop tree of ``dot_bracket`` with ``build_loop_tree`` and
    screen it with ``screen_design_obstructions``. When the shortest helix
    has at least three pairs, limit the modulus search to
    ``min(max_modulus, 2)``; otherwise search up to ``max_modulus``. Take the
    modulus from ``find_minimal_modulus``, the distinct design counts from
    ``count_distinct_separated_designs`` at that modulus, and return the
    weight from ``calibrate_gc_weight`` with the bracket ``[-50, 50]``, the
    sequence length ``len(dot_bracket)``, ``target_fraction`` and
    ``tolerance``. As a consistency check, ``evaluate_gc_ensemble`` with the
    ``tabulate_assignment_counts`` table at that modulus must report between
    1 and ``2**modulus`` proposals per accepted design at the returned
    weight, within a relative ``1e-12``. The defaults reproduce the problem
    statement.

    Parameters
    ----------
    dot_bracket : str
        Target secondary structure.
    target_fraction : float
        Target expected G+C fraction, strictly between 0 and 1.
    max_modulus : int
        Largest modulus to try, from 1 to 8.
    tolerance : float
        Final bracket width of the weight search, from ``1e-14`` to
        ``1e-6`` inclusive.

    Returns
    -------
    float
        The calibrated GC weight.

    Raises
    ------
    ValueError
        If ``max_modulus`` is not an integer from 1 to 8 (booleans are
        rejected), if the target has a five-pair loop or a three-pair loop
        with an unpaired position, if the proposals check fails, or if any
        earlier step rejects its input.
    """
    return weight
```
