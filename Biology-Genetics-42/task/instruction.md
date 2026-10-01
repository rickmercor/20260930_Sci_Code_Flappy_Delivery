# Biology-Genetics-42

## Background

RANKED NETWORKS AND WHY LABELS ARE DROPPED
A rooted phylogenetic network records an evolutionary history in which lineages both split and merge. Splitting is speciation or duplication; merging is hybridization, recombination or horizontal transfer, and a merging vertex has two parents rather than one. A network is ranked when its internal vertices are totally ordered in time, so that the history is a sequence of events rather than an unordered graph, and unlabelled when the identities of the tips are discarded. Dropping the labels is what makes histories over entirely different taxa comparable, and it is also what makes the comparison hard: without labels there is no correspondence between the tips of one network and those of the other, so a distance cannot be assembled tip by tip and has to be built from the shape and timing of the whole object.
ENCODING A NETWORK AS A MATRIX
The approach converts each network into a triangular matrix of counts, one row and column per interval between consecutive events, whose entries record how many lineages persist across a span of the history without being involved in any event. A network with n leaves and m hybridizations has n + 2m such intervals, so the matrix is (n+2m) by (n+2m). The encoding is a complete invariant: the matrix determines the network and the network determines the matrix, so a distance between two matrices is a genuine distance between two histories. Because the entries are plain counts they carry no timing on their own, and a second matrix supplies the durations that turn counts of lineages into total branch length. Distances are then ordinary matrix norms of the difference.
COMPARING NETWORKS OF DIFFERENT COMPLEXITY
Since the matrix size is n + 2m, two networks with the same number of leaves but different numbers of hybridizations produce matrices of different sizes. This is the common case in practice, because the number of reticulation events is usually what is being inferred and is rarely the same across two samples or two posterior draws. Comparing them requires putting the two event sequences into correspondence and filling the gaps in the shorter one, so that both encodings have the same dimensions and their entries refer to comparable positions in the two histories. How that correspondence is established is a modelling decision rather than a mathematical necessity: several reconciliations are conceivable, they disagree, and the resulting distance depends on which is used.
WHERE THIS SITS IN PRACTICE
Distances of this kind are used to compare posterior distributions of networks inferred from viral sequence data, where ancestral recombination graphs differ from one another in both topology and reticulation count, and where summary statistics that ignore either the timing or the reticulations lose exactly the signal of interest.

## Problem

Phylogenetic networks extend trees to histories in which lineages merge as well as split, as happens under recombination, hybridization, and horizontal gene transfer. Comparing two such histories quantitatively calls for an encoding that captures the order of events and their times, does not depend on tip labels, and reduces to a matrix so that standard norms can be used. A recent family of metrics does this by mapping each rooted, ranked, unlabelled network to a triangular matrix whose entries record how many lineages persist between events, and weighting those entries by elapsed time.
 
A difficulty arises when two networks contain different numbers of hybridization events. Their encodings then have different dimensions and cannot be compared entry by entry, so their event sequences must first be reconciled and one or both encodings refined before a norm is taken. The two networks below have the same number of leaves but different numbers of hybridizations.
 
Compute the weighted L1 distance between network A and network B.
 
Network A has six internal vertices, v1 to v6, ranked from the root, and leaves L1 to L5. Vertex v3 has two parents and is a hybridization; every other internal vertex is a bifurcation. Its edges, written from ancestor to descendant, are:
 
(*, v1), (v1, v2), (v1, v3), (v2, v3), (v2, v6), (v3, v4), (v4, L1), (v4, v5), (v5, L2), (v5, L3), (v6, L4), (v6, L5)
 
Times run from the root at 10.0 to the tips at 1.0. The six internal events v1 to v6 occur at 8.5, 6.5, 6.0, 5.5, 3.0, and 1.5, respectively.
 
Network B has eight internal vertices, v1 to v8, ranked from the root, and leaves L1 to L5. Vertices v3 and v5 have two parents and are hybridizations; every other internal vertex is a bifurcation. Its edges are:
 
(*, v1), (v1, v2), (v1, v3), (v2, v3), (v2, v5), (v3, v4), (v4, v5), (v4, v7), (v5, v6), (v6, L1), (v6, v8), (v7, L2), (v7, L3), (v8, L4), (v8, L5)
 
Times again run from the root at 10.0 to the tips at 1.0, with the eight internal events occurring at 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, and 3.0.
 
For this benchmark, write event sequences in root-to-tip order. The source paper writes its event vectors in the opposite, recent-to-root orientation and aligns events left to right; therefore, in the root-to-tip representation used here, align bifurcation events first from the tipward (most recent) end. Network A consequently receives its unmatched bifurcation position before its first root-to-tip event. Then align hybridization events separately within each span between consecutive aligned bifurcations, again from the tipward end. When one network has an unmatched hybridization in a span, insert an artificial event in the other network at the unmatched position.
 
For the calculation, an artificial event subdivides the interval in which it is inserted: in the augmented F-matrix, its inserted row and column repeat those of the interval it divides. Assign artificial-event times using the source paper's interval-subdivision rule and briefly state that rule in <reasoning>. Using one-based matrix indices 1 through N and boundary indices 0 through N, for j < i define the time weight W[i,j] = u[j-1] - u[i], with diagonal and upper-triangular weights equal to zero. The required distance is the sum over all matrix entries of |F_A[i,j] W_A[i,j] - F_B[i,j] W_B[i,j]|.
 
In <reasoning>, report the two reconciled event-sign sequences, the two reconciled ten-entry event-time vectors, and the twelve nonzero weighted-L1 contribution terms, together with their sum. These two sign sequences, two time vectors, and twelve contribution terms are explicitly required and permitted. Include the two weighted counts before differencing at entry (8,7), and state the dimensions, lineage-count diagonals, and endpoint/order checks used to validate the result. Do not print full F-matrices or other pipeline dumps. Report the final distance to six decimal places.
 
Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
 
The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Report the required derived evidence above; apart from it, show only the scalars needed to justify the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_lineage_counts.py

Goal
----
Return the number of lineages present in each ranked interval of a network, in order from the root.

```python
def lineage_counts(edges, size):
    """Return the lineage count in each ranked interval.

    An edge is present in interval ``k`` when its upper endpoint has rank
    below ``k`` and its lower endpoint has rank at least ``k``. The stem is
    written with upper endpoint ``"*"`` and has rank zero. Internal vertices
    are named ``"v<number>"`` and leaves are named ``"L<number>"``.

    Parameters
    ----------
    edges : list of tuple of str
        Edges as ``(ancestor, descendant)`` pairs.
    size : int
        Number of ranked intervals. Internal vertex ranks must lie in
        ``1..size-1``.

    Returns
    -------
    list of int
        One lineage count per interval, ordered from root to tip.

    Raises
    ------
    ValueError
        If ``size`` is not an integer of at least two; if ``edges`` is empty
        or malformed; if a vertex label is invalid; if an internal rank lies
        outside ``1..size-1``; or if an edge is not directed from an older
        event to a younger event.
    """
    return []
```

### Step 2

02_count_matrix.py

Goal
----
Return the lower-triangular matrix whose entry (i, j) counts the lineages present throughout intervals j to i without being involved in any event.

```python
def count_matrix(edges, size):
    """Construct the lower-triangular F-matrix of persisting lineages.

    Parameters
    ----------
    edges : list of tuple of str
        Edges as ``(ancestor, descendant)`` pairs. The root is ``"*"``,
        internal vertices are named ``"v<number>"``, and leaves are named
        ``"L<number>"``.
    size : int
        Number of ranked intervals. Internal vertex ranks must lie in
        ``1..size-1``.

    Returns
    -------
    list of list of int
        A ``size`` by ``size`` lower-triangular matrix. For ``j <= i``,
        entry ``(i, j)`` is the number of lineages that persist continuously
        through every ranked interval from ``j`` through ``i``. Entries above
        the diagonal are zero.

    Raises
    ------
    ValueError
        If ``size`` is not an integer of at least two; if ``edges`` is empty
        or malformed; if a vertex label is invalid; if an internal rank lies
        outside ``1..size-1``; or if an edge does not point from an older
        event to a younger event.
    """
    return []
```

### Step 3

03_event_signs.py

Goal
----
Return the sequence of event types read off the lineage counts, using +1 for a bifurcation and -1 for a hybridization, in order from the root.

```python
def event_signs(counts):
    """Classify ranked network events from consecutive lineage counts.

    Parameters
    ----------
    counts : list of int
        Lineage counts ordered from the rootward interval to the most recent
        interval.

    Returns
    -------
    list of int
        Event signs between consecutive intervals: ``+1`` for a bifurcation
        and ``-1`` for a hybridization.

    Raises
    ------
    ValueError
        If ``counts`` is not a sequence of at least two positive integers; if
        the first count is not one; or if any consecutive change is not
        exactly ``+1`` or ``-1``.
    """
    return []
```

### Step 4

04_reconcile_timed_matrices.py

Goal
----
Construct the joint refined F-matrices and boundary times for two ranked networks whose event histories need not have the same length.

```python
def reconcile_timed_matrices(signs_a, matrix_a, times_a,
                             signs_b, matrix_b, times_b):
    """Return the joint event refinement of two timed F-matrix encodings.
 
    Parameters
    ----------
    signs_a, signs_b : list of int
        Real event signs in root-to-tip order: +1 for a bifurcation and -1
        for a hybridization. Each sequence begins with a bifurcation.
    matrix_a, matrix_b : list of list of int
        Original lower-triangular F-matrices. A sequence with K real events
        has a (K+1)-by-(K+1) matrix. Diagonal differences equal the supplied
        event signs and the first diagonal entry is one.
    times_a, times_b : list of float
        Strictly decreasing boundary times [root, real-event times..., tip].
        A sequence with K events has K+2 boundary times. The two networks
        retain their own clocks; equal event coordinates do not imply equal
        times.
 
    Returns
    -------
    tuple
        (refined_matrix_a, refined_times_a, refined_matrix_b, refined_times_b).
        The matrices are M-by-M nested integer lists and each boundary vector
        is a list of M+1 floats, for the common refined interval count M.
 
        Event correspondence is the source's recent-first correspondence:
        bifurcations with the same rank counted from the most recent end
        correspond. Hybridizations correspond by their recency rank within
        each resulting branching span, including the terminal spans. Missing
        event coordinates are artificial events, and all results are returned
        in root-to-tip order. No real event may be moved or discarded.
 
        Artificial events leave the lineage set unchanged. Each refined
        F-entry counts persistence through its refined interval span. The
        original times are retained, and a run of r artificial events divides
        its enclosing original interval into r+1 equal durations. Either or
        both networks may be refined; M need not equal the larger input size.
        Inputs are not modified.
 
    Raises
    ------
    ValueError
        If signs are empty, are not integer +1/-1 values, or do not begin with
        +1; if a matrix is not square with nonnegative integer entries and
        zeros above the diagonal; if matrix size, positive diagonal, or
        diagonal differences disagree with the signs; or if a boundary vector
        has the wrong length, non-finite/non-numeric entries, or times that
        are not strictly decreasing. Booleans are not integers or times here.
        Lists and tuples are accepted for the input sequences and matrix rows.
    """
    return ([], [], [], [])
```

### Step 5

05_weight_matrix.py

Goal
----
Return the matrix of elapsed times attached to each entry of a count matrix, given the interval times.

```python
def weight_matrix(times):

    """Compute the lower-triangular time-weight matrix for aligned intervals.



    If ``times`` contains ``N + 1`` strictly decreasing interval-boundary

    times, the returned matrix has shape ``N x N``. With zero-based Python

    indices, for row ``i`` and column ``j < i``,

    ``W[i][j] = times[j] - times[i + 1]``. Diagonal and upper-triangular

    entries are zero. This is the zero-based form of the paper's

    ``W_{i,j} = u_{j-1} - u_i`` convention.



    Parameters

    ----------

    times : list of float

        ``N + 1`` aligned interval-boundary times ordered from root to tip.



    Returns

    -------

    list of list of float

        An ``N x N`` lower-triangular matrix of nonnegative elapsed-time

        weights.



    Raises

    ------

    ValueError

        If ``times`` contains fewer than two entries, contains non-numeric or

        non-finite values, or is not strictly decreasing from root to tip.

    """

    return []
```

### Step 6

06_weighted_l1.py

Goal
----
Return the sum over all entries of the absolute difference between the two networks' weighted counts.

```python
def weighted_l1(matrix_a, weights_a, matrix_b, weights_b):

    """Compute the weighted L1 distance between two aligned F-matrices.



    Parameters

    ----------

    matrix_a, matrix_b : list of list of int

        Aligned lower-triangular lineage-count matrices.

    weights_a, weights_b : list of list of float

        Corresponding lower-triangular elapsed-time weight matrices.



    Returns

    -------

    float

        The sum of all lower-triangular weighted absolute differences.



    Raises

    ------

    ValueError

        If the four matrices are empty, non-square, or have different shapes;

        if count entries are invalid; if weight entries are non-numeric or

        non-finite; or if any matrix has nonzero entries above its diagonal.

    """

    return 0.0
```

### Step 7

07_network_distance.py

Goal
----
Return the weighted L1 distance after jointly refining the two ranked timed encodings to compatible event coordinates.

```python
def network_distance(edges_a, size_a, times_a, edges_b, size_b, times_b):
    """Compute the weighted L1 distance between two ranked networks.
 
    The calculation constructs each F-matrix, aligns event sequences in
    root-to-tip order, inserts any required artificial events, builds the two
    time-weight matrices, and sums the lower-triangular weighted differences.
 
    Parameters
    ----------
    edges_a, edges_b : list of tuple of str
        Edge lists for the two networks. The root is written ``"*"``;
        internal vertices are ``"v<number>"`` and leaves are ``"L<number>"``.
    size_a, size_b : int
        Numbers of ranked intervals in the two networks.
    times_a, times_b : list of float
        Boundary-time vectors ``[root, one time per real event, tip]``,
        ordered from root to tip.
 
    Returns
    -------
    float
        The weighted L1 distance between the reconciled F-matrices.
 
    Raises
    ------
    ValueError
        If either network encoding, its ranked-interval count, or its event
        times are invalid; or if the two reconciled matrices cannot be given
        a common aligned dimension.
    """
    return 0.0
```
