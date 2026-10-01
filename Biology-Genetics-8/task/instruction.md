# Biology-Genetics-8

## Background

Evolutionary histories are usually drawn as trees, in which every lineage has exactly one
ancestor. That picture fails whenever genetic material moves sideways rather than only
downwards. Recombination, hybridisation and horizontal gene transfer all produce lineages
with two ancestors, and representing them requires a directed acyclic graph, a
phylogenetic network, in which some vertices have two incoming edges. Two such graphs are
central to modern population genetics: the ancestral recombination graph, which records
the history of a sample under recombination, and the reassortment graph, which does the
same for segmented viral genomes. Both are routinely inferred from sequence data, and both
are produced in large numbers, since Bayesian inference returns a posterior distribution
over histories rather than a single answer.
That immediately creates a measurement problem. To summarise a posterior over networks, to
build a credible set, to ask whether two inference procedures agree, or to test whether an
estimated history differs meaningfully from a null expectation, one needs a way to say how
far apart two networks are. For trees this is long settled and many distances exist. For
networks the situation is much thinner, and the available options tend to fail in one of
two ways. Most require the two graphs to carry the same taxon labels, which makes them
useless for comparing samples drawn from different sets of individuals or different time
points. Those that avoid labels generally discard the ordering of internal events in time.
Discarding that ordering is a serious loss rather than a technical simplification. The
coalescent and birth-death processes that underlie almost all inference in this area
generate histories in which internal events are ordered in time, and that ordering carries
real information about population size, selection and the tempo of reticulation. A history
in which a hybridisation occurs early among few lineages is a different biological claim
from one in which the same hybridisation occurs late among many, even when the two graphs
have the same shape. A comparison that treats them as identical is discarding exactly the
signal the inference was run to recover.
The line of work this task is drawn from addresses both problems at once, for rooted
networks whose internal events are ranked in time and whose leaves are unlabelled. Its
strategy is to encode each network as an integer matrix in a way that is bijective, so
that no information is lost and no two distinct networks collide, and then to obtain
distances between networks as ordinary matrix norms applied to the difference of their
encodings. Because the encoding is a bijection onto a set of matrices cut out by linear
constraints, it also makes the space of networks enumerable and opens the way to summary
statistics such as means over a posterior sample. The approach extends an earlier encoding
built for ranked trees, and is developed for histories sampled at a single time point as
well as for the heterochronous case common in rapidly evolving pathogens.
When the networks carry branch lengths in addition to a ranking, comparing encodings
directly measures a difference in counts of lineages, which is blind to how much time
separates the events involved. The same line of work therefore extends the encoding so
that the comparison is expressed in units of evolutionary time, drawing on the event
times of each network.

## Problem

Reticulate processes such as recombination, hybridisation and horizontal gene transfer
produce evolutionary histories that are graphs rather than trees, and comparing two such
histories quantitatively is a long-standing obstacle. Most existing distances require the
two graphs to carry the same taxon labels, which prevents comparison across
non-overlapping samples, or they discard the temporal ordering of internal events, which
is precisely the information that coalescent and birth-death inference produces. A
distance for rooted, ranked, unlabelled networks avoids both problems: the leaves carry no
labels, and the ranking of speciation and hybridisation events in time is retained.
The route to such a distance is a bijective triangular integer matrix encoding of the
ranked network. When the networks also carry branch lengths, the encoding is combined
with the event times of each network before the two are compared, so that the distance is
expressed in units of evolutionary time rather than in counts of lineages.
Build the two timed ranked networks given below, form the triangular encoding and the
weight matrix of each, and report the time-weighted distance between them under the
Euclidean matrix norm. Both networks have four leaves and two hybridisation events, so
their encodings have the same dimension and no event alignment is required. Use the
following configuration:
* internal vertices v1 to v7 are ranked in time by their index, v1 being the oldest and
  nearest the root and v7 the most recent; the four leaves are L1 to L4
* network A has the twelve directed edges, written ancestor to descendant:
  (v1,v2), (v1,v3), (v2,v4), (v2,v6), (v3,v5), (v3,v6), (v4,L1), (v4,v7),
  (v5,L2), (v5,v7), (v6,L3), (v7,L4)
* network B has the twelve directed edges:
  (v1,v2), (v1,v3), (v2,L1), (v2,L2), (v3,L3), (v3,v4), (v4,v5), (v4,v6),
  (v5,v6), (v5,v7), (v6,v7), (v7,L4)
* the event times of network A are u0 = 0, u1 = 1, u2 = 2, u3 = 3, u4 = 4, u5 = 5,
  u6 = 10, u7 = 11, u8 = 12, where u0 is the time at the top of the stem above the root,
  uk is the time of the event at vertex vk, and u8 is the sampling time of every leaf
* the event times of network B are u0 = 0, u1 = 2, u2 = 4, u3 = 6, u4 = 7, u5 = 8,
  u6 = 9, u7 = 11, u8 = 12, under the same convention
* both networks are isochronous: all four leaves of a network are sampled at its final
  time
* leaf labels carry no information and must not enter the computation
Your final answer must be a single number: the time-weighted distance between network A
and network B.
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

01_classify_vertices

Goal
----
Classify the vertices of a ranked network by their in and out degrees.

```python
def classify_vertices(edges: list[tuple[str, str]]) -> dict:
    '''Classify every vertex of a ranked network by in and out degree.
    Parameters
    ----------
    edges : list[tuple[str, str]]
        Directed edges written as (ancestor, descendant) pairs.
    Returns
    -------
    classification : dict
        Mapping with keys 'root' (str), 'speciation' and 'hybridisation' and
        'leaves' (sorted lists of str), 'n' (int, the number of leaves),
        'm' (int, the number of hybridisation vertices) and 'size' (int, the
        number of events, equal to n + 2m).
    '''
    return {}
```

### Step 2

02_event_sequence

Goal
----
Read the ranking of the internal vertices as an ordered sequence of event types.

```python
def event_sequence(classification: dict) -> list[int]:
    '''List the event types of the network in rank order.
    Parameters
    ----------
    classification : dict
        The vertex classification as returned by the first step.
    Returns
    -------
    events : list[int]
        One entry per internal vertex, ordered by rank from the root, coded
        0 for a speciation and 1 for a hybridisation.
    '''
    return []
```

### Step 3

03_lineage_sets

Goal
----
Determine which ancestral lineages are present in each time interval.

```python
def lineage_sets(edges: list[tuple[str, str]], size: int,
                 root: str) -> list[list[int]]:
    '''List the lineages alive in each interval of the network.
    Parameters
    ----------
    edges : list[tuple[str, str]]
        Directed edges written ancestor to descendant.
    size : int
        The number of events, equal to n + 2m for a network with n leaves and
        m hybridisations.
    root : str
        The root vertex.
    Returns
    -------
    lineages : list[list[int]]
        One entry per interval, each a sorted list of integer indices into the
        edge list formed by prepending the stem edge above the root to the
        network edges and sorting.
    '''
    return []
```

### Step 4

04_encoding_matrix

Goal
----
Build the triangular integer encoding from the interval lineage sets.

```python
def encoding_matrix(lineage_sets: list[list[int]]) -> list[list[int]]:
    '''Build the triangular encoding from the lineage sets.

    Parameters
    ----------
    lineage_sets : list[list[int]]
        The lineage sets as returned by the third step: for each interval, the
        sorted indices of the lineages alive in it.

    Returns
    -------
    matrix : list[list[int]]
        An N by N matrix. Entry (i, j) with j at most i counts the lineages
        present in both interval j and interval i; every entry above the
        diagonal is zero.
    '''
    return []
```

### Step 5

05_diagonal_profile

Goal
----
Extract the diagonal and subdiagonal and check them against the event sequence.

```python
def diagonal_profile(matrix: list[list[int]], events: list[int]) -> dict:
    '''Read the diagonal and subdiagonal of an encoding and check them against
    the event sequence.
    Parameters
    ----------
    matrix : list[list[int]]
        The triangular encoding as returned by the fourth step.
    events : list[int]
        The event codes as returned by the second step, 0 for a speciation and
        1 for a hybridisation.
    Returns
    -------
    profile : dict
        Mapping with keys 'diagonal' (list of ints), 'subdiagonal' (list of ints)
        and 'consistent' (bool).
    '''
    return {}
```

### Step 6

06_validate_encoding

Goal
----
Test a candidate matrix against the structural constraints of the encoding space.

```python
def validate_encoding(matrix: list[list[int]], n: int, m: int) -> dict:
    '''Check a matrix against the constraints defining the encoding space.
    Parameters
    ----------
    matrix : list[list[int]]
        Candidate triangular encoding.
    n : int
        Number of leaves.
    m : int
        Number of hybridisations.
    Returns
    -------
    report : dict
        Mapping with keys 'size_ok', 'P1', 'P2', 'P3', 'P4', 'P5' and 'valid',
        all bool.
    '''
    return {}
```

### Step 7

07_weight_matrix

Goal
----
Build the weight matrix that converts the integer encoding into units of time.

```python
def weight_matrix(times: list[float]) -> list[list[float]]:
    '''Build the time-weight matrix of one ranked network.

    Parameters
    ----------
    times : list[float]
        Strictly increasing event times u_0 to u_N, where u_0 is the time at the
        top of the stem above the root, u_k is the time of the event of rank k,
        and u_N is the sampling time of the leaves.

    Returns
    -------
    weights : list[list[float]]
        An N by N matrix. Entry (i, j) with j strictly less than i is the elapsed
        time from the start of interval j to the event of rank i, expressed as a
        positive branch length. All other entries are zero.
    '''
    return []
```

### Step 8

08_weighted_distance

Goal
----
Take the Euclidean norm of the difference of the two weighted encodings.

```python
def weighted_distance(matrix_a: list[list[int]], weights_a: list[list[float]],
                      matrix_b: list[list[int]], weights_b: list[list[float]]) -> float:
    '''Distance between two timed ranked networks under the Euclidean matrix norm.
    Parameters
    ----------
    matrix_a, matrix_b : list[list[int]]
        The triangular encodings of the two networks, of equal size.
    weights_a, weights_b : list[list[float]]
        The corresponding weight matrices, of the same size.
    Returns
    -------
    distance : float
        The norm of the difference of the two weighted encodings, rounded to
        6 decimal places.
    '''
    return 0.0
```

### Step 9

09_timed_network_distance

Goal
----
Run the whole pipeline: classify, encode, validate, weight, and report the distance.

```python
def timed_network_distance(edges_a: list[tuple[str, str]], times_a: list[float],
                           edges_b: list[tuple[str, str]], times_b: list[float]) -> float:
    '''Distance between two timed ranked networks given their edges and event times.
    Parameters
    ----------
    edges_a, edges_b : list[tuple[str, str]]
        Directed edges written ancestor to descendant. Internal vertices are named
        v1 upwards in rank order; leaf names carry no information.
    times_a, times_b : list[float]
        Strictly increasing event times u_0 to u_N of the respective networks.
    Returns
    -------
    distance : float
        The time-weighted distance, rounded to 6 decimal places.
    '''
    return 0.0
```
