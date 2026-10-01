# Biology-Biochemistry-44

## Background

Probabilistic alignment models score the relationship between a sequence and its homolog by summing over every way the two can be aligned, so the total weight of that ensemble summarizes how compatible the sequences are while accounting for alignment uncertainty. Mutational scanning and sequence design ask how this compatibility changes when bases of the reference are replaced, and such changes are finite discrete edits rather than infinitesimal perturbations. When two edits are made together, their joint effect can differ from what their separate effects predict, and this epistasis between neighbouring residues governs how far single-site scans can be trusted when several changes are introduced at once.

## Problem

I compare a 2,000-base reference RNA x with a homologous RNA y under a three-state pair model, and I want to know exactly how far the combined effect of substituting two neighbouring bases of x departs from the product of their separate effects. A match state M emits a base a of x aligned with a base b of y with factor e_M(a, b), equal to 0.16 when a = b, 0.05 for the transitions A-G and C-U and 0.03 for every transversion; a state X emits a base a of x against a gap with factor e_X(a), and a state Y emits a base b of y against a gap with factor e_Y(b), where in A, C, G, U order e_X = (0.12, 0.38, 0.38, 0.12) and e_Y = (0.16, 0.34, 0.34, 0.16). M is entered from M, X or Y with no further factor, X and Y are each entered only from M with an extra factor g_open or from themselves with an extra factor g_extend, the first column is scored as though it followed M, an alignment may end in any state, and Z(s) is the summed weight of all global alignments of a sequence s with y. I fixed g_open and g_extend so that, when each alignment of x with y is weighted by its share of Z(x), the expected number of gap-opening columns (X or Y columns preceded by M, the first column counting as preceded by M) is 115 and the expected number of gap-extension columns (X or Y columns preceded by the same state) is 32, the counts in my curated alignment of this pair. With bases coded 0, 1, 2, 3 for A, C, G, U and rng = numpy.random.default_rng(20260914), x = rng.integers(0, 4, 2000) is drawn first, then u = rng.random(2000) and then v = rng.integers(0, 4, 2000), and y is built by visiting i = 1, ..., 2000 in order and skipping base i of x when u_i < 0.06, writing v_i when 0.06 <= u_i < 0.22 and writing base i of x otherwise, where u_i and v_i are entry i of u and v. For every i from 1 to 1999 and every choice of new bases at positions i and i + 1, each different from the base it replaces, let epsilon = ln Z(double mutant) + ln Z(x) - ln Z(x with only the substitution at i) - ln Z(x with only the substitution at i + 1), with g_open and g_extend kept at their fitted values. I want the mean of epsilon over all these double substitutions, to at least six significant figures. In your reasoning, state, each to at least eight significant figures, g_open, g_extend and ln Z(x), the double substitution with the largest |epsilon| together with its epsilon, and the fraction of double substitutions with epsilon > 0, and tell me what these results imply for predicting double-substitution effects from quantities computed on the reference sequence alone.

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

01_build_homolog_pair

Goal
----
Generate a coded reference sequence and a homolog derived from it by seeded random deletions and substitutions.

```python
def build_homolog_pair(
    seed: int,
    length: int,
    delete_probability: float,
    substitute_probability: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return a coded reference sequence ``x`` and its synthetic homolog ``y``.

    Bases are coded 0, 1, 2, 3 for A, C, G, U. A generator
    ``numpy.random.default_rng(seed)`` makes exactly three draws in this
    order: ``x = integers(0, 4, length)``, then ``u = random(length)``, then
    ``v = integers(0, 4, length)``. Visiting positions ``i = 0, ...,
    length - 1`` in order, ``y`` omits base ``i`` when ``u[i] <
    delete_probability``, writes ``v[i]`` when ``delete_probability <= u[i] <
    delete_probability + substitute_probability`` (which may equal ``x[i]``)
    and writes ``x[i]`` otherwise.

    Parameters
    ----------
    seed : int
        Nonnegative integer seed.
    length : int
        Positive number of bases in ``x``.
    delete_probability : float
        Probability in ``[0, 1]`` that a base is omitted from ``y``.
    substitute_probability : float
        Probability in ``[0, 1]`` that a base is redrawn; the two
        probabilities sum to at most 1.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` as one-dimensional ``int64`` arrays of codes.

    Raises
    ------
    ValueError
        If ``seed`` is not a nonnegative integer, ``length`` is not a
        positive integer (booleans are rejected for both), a probability is
        not a finite number in ``[0, 1]``, the probabilities sum to more
        than 1, or ``y`` would be empty.
    """
    return x, y
```

### Step 2

02_compute_log_forward_tables

Goal
----
Compute the natural-log forward weights of every partial global alignment of a coded sequence x with a coded sequence y under a three-state pair model.

```python
def compute_log_forward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Return natural-log forward weights of the pair model on every lattice point.

    Bases are coded 0 to 3. An alignment column is in state M (base ``a``
    of ``x`` against base ``b`` of ``y``, factor ``match_factors[a, b]``),
    X (base ``a`` of ``x`` against a gap, factor ``x_gap_factors[a]``) or Y
    (base ``b`` of ``y`` against a gap, factor ``y_gap_factors[b]``). An M
    column carries only its emission factor whatever the previous state. An
    X or Y column carries ``gap_open`` times its emission factor when the
    previous column is M and ``gap_extend`` times it when the previous column
    is the same gap state; X never follows Y and Y never follows X. The first
    column is scored as though it followed M, every base of both sequences is
    used, and an alignment may end in any state.

    Entry ``[s, i, j]`` is the natural log of the summed weight of all
    partial alignments of the first ``i`` bases of ``x`` with the first
    ``j`` bases of ``y`` whose last column is in state ``s`` (0 = M, 1 = X,
    2 = Y), or ``-inf`` when there are none. The empty alignment counts as
    an M column: entry ``[0, 0, 0]`` is 0 and entries ``[1, 0, 0]`` and
    ``[2, 0, 0]`` are ``-inf``. Values must stay exact to double precision
    even when the summed weights are far below the smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3, lengths
        ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.

    Returns
    -------
    np.ndarray
        Float array of shape ``(3, L + 1, K + 1)``.

    Raises
    ------
    ValueError
        If a sequence is empty, not one-dimensional, not of integer type or
        holds a code outside 0 to 3, if a factor array has the wrong shape,
        or if any factor is not finite and positive.
    """
    return log_forward
```

### Step 3

03_compute_log_backward_tables

Goal
----
Compute the natural logs of the derivatives of the total alignment weight with respect to every forward weight of the pair model.

```python
def compute_log_backward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Return natural-log derivatives of the total weight with respect to forward weights.

    The model, the coding of bases and states and the forward weights
    ``F[s, i, j]`` are those of ``compute_log_forward_tables``. Let ``Z``
    be the summed weight of all complete alignments, the sum of the three
    forward weights at ``(L, K)``. Treat ``Z`` as a function of the forward
    weights through the forward recursion applied at every lattice point,
    with predecessors outside the lattice contributing nothing. Entry
    ``[s, i, j]`` is the natural log of the derivative of ``Z`` with respect
    to ``F[s, i, j]``, and ``-inf`` where that derivative is zero. The three
    entries at ``(L, K)`` are therefore 0, and entry ``[0, 0, 0]`` equals the
    natural log of ``Z``. Values must stay exact to double precision even
    when the weights are far below the smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3, lengths
        ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.

    Returns
    -------
    np.ndarray
        Float array of shape ``(3, L + 1, K + 1)``.

    Raises
    ------
    ValueError
        If a sequence is empty, not one-dimensional, not of integer type or
        holds a code outside 0 to 3, if a factor array has the wrong shape,
        or if any factor is not finite and positive.
    """
    return log_backward
```

### Step 4

04_compute_expected_gap_transition_counts

Goal
----
Compute the posterior expected numbers of gap-opening and gap-extension columns in the alignment ensemble of x with y from the reference forward and backward tables.

```python
def compute_expected_gap_transition_counts(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Return the expected numbers of gap-opening and gap-extension columns.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` for the same arguments. Each complete
    alignment of ``x`` with ``y`` is drawn with probability equal to its
    weight divided by the summed weight ``Z`` of all complete alignments. A
    gap-opening column is an X or Y column whose previous column is M, the
    first column counting as following M; a gap-extension column is an X or
    Y column whose previous column is the same gap state. Return the expected
    number of each kind of column, exact to double precision.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3, lengths
        ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.
    log_forward, log_backward : np.ndarray
        Tables of shape ``(3, L + 1, K + 1)`` for these arguments.

    Returns
    -------
    np.ndarray
        Float array ``[expected_openings, expected_extensions]``.

    Raises
    ------
    ValueError
        If a sequence or factor is invalid as in ``compute_log_forward_tables``,
        if a table has the wrong shape or holds NaN or ``+inf``, or if the
        total weight ``Z`` is zero.
    """
    return expected_counts

# EXPECTED RETURN
# np.ndarray: [expected number of gap-opening columns, expected number of gap-extension columns].
```

### Step 5

05_calibrate_gap_factors

Goal
----
Find the gap-opening and gap-extension factors at which the expected numbers of gap-opening and gap-extension columns in the alignment ensemble of x with y equal two target counts.

```python
def calibrate_gap_factors(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    open_count: float,
    extend_count: float,
) -> "np.ndarray":
    """Return the gap factors whose expected transition counts match two targets.

    The pair model is that of ``compute_log_forward_tables``, with the match
    and gap emission factors given and the two transition factors unknown.
    Find positive ``gap_open`` and ``gap_extend`` at which the values
    returned by ``compute_expected_gap_transition_counts`` equal
    ``open_count`` and ``extend_count``. If the equations have more than one
    positive solution, return any positive pair satisfying them. Both
    expected counts at the returned factors must reproduce their targets to
    a relative precision of ``1e-10``.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    open_count, extend_count : float
        Finite positive target expected counts.

    Returns
    -------
    np.ndarray
        Float array ``[gap_open, gap_extend]``.

    Raises
    ------
    ValueError
        If a sequence or factor array is invalid as in
        ``compute_log_forward_tables``, if a target count is not a finite
        positive number, or if no positive pair of factors reproduces the
        targets.
    """
    return gap_factors
```

### Step 6

06_evaluate_single_substitution_log_weights

Goal
----
Obtain the exact natural-log total alignment weight after every single-base substitution of x from the reference forward and backward tables.

```python
def evaluate_single_substitution_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Return the natural-log total weight of x after each single-base substitution.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` evaluated for the reference ``x``
    and ``y``. Entry ``[i, c]`` is the natural log of the summed weight of all
    complete alignments of ``y`` with ``x`` after base ``i`` (0-based) is set
    to code ``c``, every other factor of the model unchanged; for ``c ==
    x[i]`` it is the reference value. Results must agree with a full
    recomputation for the substituted sequence to double precision, including
    when the weights are far below the smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Reference sequences, non-empty one-dimensional integer arrays of
        codes 0 to 3, lengths ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    log_forward, log_backward : np.ndarray
        Reference tables of shape ``(3, L + 1, K + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(L, 4)``.

    Raises
    ------
    ValueError
        If a sequence or factor array is invalid as in
        ``compute_log_forward_tables``, if a table has the wrong shape or
        holds NaN or ``+inf``, or if the reference total weight is zero.
    """
    return single_log_weights
```

### Step 7

07_compute_neighbour_double_log_weights

Goal
----
Obtain the exact natural-log total alignment weight after every double substitution of two neighbouring bases of x from the reference forward and backward tables.

```python
def compute_neighbour_double_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Return the natural-log total weight of x after each neighbouring double substitution.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` evaluated for the reference ``x``
    and ``y``. Entry ``[i, c, d]`` is the natural log of the summed weight of
    all complete alignments of ``y`` with ``x`` after base ``i`` (0-based) is
    set to code ``c`` and base ``i + 1`` to code ``d``, every other factor of
    the model unchanged. A code equal to the base already present leaves that
    base unchanged, so entry ``[i, x[i], x[i + 1]]`` is the reference value.
    Results must agree with a full recomputation for the substituted sequence
    to double precision, including when the weights are far below the
    smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Reference sequences, one-dimensional integer arrays of codes 0 to 3,
        lengths ``L >= 2`` and ``K >= 1``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.
    log_forward, log_backward : np.ndarray
        Reference tables of shape ``(3, L + 1, K + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(L - 1, 4, 4)``.

    Raises
    ------
    ValueError
        If a sequence or factor is invalid as in ``compute_log_forward_tables``,
        if ``x`` has fewer than two bases, if a table has the wrong shape or
        holds NaN or ``+inf``, or if the reference total weight is zero.
    """
    return double_log_weights
```

### Step 8

08_summarize_neighbour_epistasis

Goal
----
Convert single and neighbouring double substitution log weights into log-scale epistasis values and summarize them over all genuine double substitutions.

```python
def summarize_neighbour_epistasis(
    x: "np.ndarray",
    single_log_weights: "np.ndarray",
    double_log_weights: "np.ndarray",
) -> "np.ndarray":
    """Return summary statistics of log-scale epistasis between neighbouring substitutions.

    ``single_log_weights[i, c]`` is ``ln Z`` after base ``i`` of ``x`` is set
    to code ``c`` and ``double_log_weights[i, c, d]`` is ``ln Z`` after bases
    ``i`` and ``i + 1`` are set to ``c`` and ``d``, as returned by
    ``evaluate_single_substitution_log_weights`` and
    ``compute_neighbour_double_log_weights``; the reference value
    ``ln Z(x)`` is ``single_log_weights[0, x[0]]``. For every ``i`` from 0 to
    ``L - 2`` and every ``c != x[i]`` and ``d != x[i + 1]``, the epistasis is
    ``double[i, c, d] + ln Z(x) - single[i, c] - single[i + 1, d]``. Only
    these ``9 (L - 1)`` genuine double substitutions enter the summary. The
    largest absolute epistasis is resolved in favour of the earliest entry in
    ``(i, c, d)`` order when values are exactly equal.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional integer array of codes 0 to 3 with ``L >= 2``.
    single_log_weights : np.ndarray
        Shape ``(L, 4)``, finite.
    double_log_weights : np.ndarray
        Shape ``(L - 1, 4, 4)``, finite.

    Returns
    -------
    np.ndarray
        Float array ``[mean, extreme, position, code_i, code_next,
        positive_fraction, count]``: the mean epistasis; the epistasis with
        the largest absolute value, its 1-based position ``i + 1`` and the
        codes ``c`` and ``d``; the fraction of strictly positive values; and
        the number of genuine double substitutions.

    Raises
    ------
    ValueError
        If ``x`` is not a one-dimensional integer array of codes 0 to 3 with
        at least two bases, if an array has the wrong shape or a non-finite
        entry, or if the unsubstituted entries ``single[i, x[i]]`` and
        ``double[i, x[i], x[i + 1]]`` differ from ``ln Z(x)`` by more than
        ``1e-8 * max(1, |ln Z(x)|)``.
    """
    return summary
```

### Step 9

09_estimate_mean_neighbour_epistasis

Goal
----
Compose every earlier step to obtain the mean log-scale epistasis over all neighbouring double substitutions of a synthetic reference sequence aligned with its homolog under gap factors fitted to observed transition counts.
Orchestrator: yes - builds the sequence pair (build_homolog_pair), fits the gap factors (calibrate_gap_factors), computes the reference forward and backward tables (compute_log_forward_tables, compute_log_backward_tables), confirms the fitted expected counts (compute_expected_gap_transition_counts), scores every single substitution (evaluate_single_substitution_log_weights) and every neighbouring double substitution (compute_neighbour_double_log_weights), and summarizes the epistasis (summarize_neighbour_epistasis), consuming each output rather than reimplementing any step.

```python
def estimate_mean_neighbour_epistasis(
    seed: int = 20260914,
    length: int = 2000,
    delete_probability: float = 0.06,
    substitute_probability: float = 0.16,
    match_factors: "tuple" = ((0.16, 0.03, 0.05, 0.03), (0.03, 0.16, 0.03, 0.05),
                              (0.05, 0.03, 0.16, 0.03), (0.03, 0.05, 0.03, 0.16)),
    x_gap_factors: "tuple" = (0.12, 0.38, 0.38, 0.12),
    y_gap_factors: "tuple" = (0.16, 0.34, 0.34, 0.16),
    open_count: float = 115.0,
    extend_count: float = 32.0,
) -> float:
    """Return the mean log-scale epistasis over all neighbouring double substitutions.

    Build ``(x, y)`` with ``build_homolog_pair(seed, length,
    delete_probability, substitute_probability)``. Use the pair model of
    ``compute_log_forward_tables`` with the given emission factors and with
    the gap factors returned by ``calibrate_gap_factors`` for ``open_count``
    and ``extend_count`` on this pair. Let ``Z(s)`` be the summed weight of
    all complete alignments of ``y`` with a sequence ``s``. For every pair of
    neighbouring positions of ``x`` and every choice of new codes at both,
    each different from the base it replaces, the epistasis is ``ln
    Z(double) + ln Z(x) - ln Z(first single) - ln Z(second single)`` as in
    ``summarize_neighbour_epistasis``. Return the mean over all ``9 (length -
    1)`` such double substitutions. The defaults reproduce the problem
    statement.

    Parameters
    ----------
    seed, length, delete_probability, substitute_probability :
        Arguments of ``build_homolog_pair``; ``length`` must be at least 2.
    match_factors : tuple
        Nested ``(4, 4)`` finite positive factors, row = base of ``x``.
    x_gap_factors, y_gap_factors : tuple
        Four finite positive gap emission factors each.
    open_count, extend_count : float
        Target expected numbers of gap-opening and gap-extension columns.

    Returns
    -------
    float
        Mean epistasis as a native Python float.

    Raises
    ------
    ValueError
        If any argument is invalid for the step that consumes it, including
        ``length < 2`` and target counts that no positive gap factors
        reproduce.
    """
    return 0.0
```
