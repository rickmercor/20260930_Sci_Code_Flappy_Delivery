# Biology-Ecology-2

## Background

A food web records who eats whom in an ecological community. Because the identity of
the species filling a given position differs from place to place, comparing communities
by their species lists reveals little about whether they are organised in the same way.
Comparative ecology therefore works with ecological roles instead: a top predator in one
savanna and a top predator in another play equivalent parts in their respective systems
even when they share no taxonomy. The long-standing question is which arrangements of
interactions recur across communities, because those recurring arrangements are the ones
likely to persist as species distributions shift under climate change, and they are what
conservation planning would most want to protect.
Early comparative work described whole networks by summary quantities such as connectance,
modularity or the distribution of trophic levels. Those descriptors are informative but
coarse: two webs can agree on all of them and still be assembled differently at the level
of individual species. A finer approach characterises each species by the local interaction
patterns it participates in, since the small subnetworks a species appears in, and the
position it occupies inside them, capture something durable about its functional part in
the community. Species roles defined this way have been shown to be conserved over
evolutionary time, which is what makes them a credible basis for comparison across
ecosystems rather than merely a convenient statistic.
Comparing two webs then becomes a matching problem: find the correspondence between the
species of one web and those of another that best preserves these roles. Matching each
species to exactly one counterpart is the natural first formulation, and it is how the
problem has usually been posed. That restriction is awkward biologically, because real
communities contain species with overlapping or partially redundant roles, and a
one-to-one rule must arbitrarily choose one of several equally good counterparts and
discard the rest. It is also awkward computationally, since the resulting combinatorial
search scales poorly and has typically been attacked with stochastic optimisation, making
results slow and hard to reproduce exactly.
Relaxing the one-to-one requirement to allow a species to be partially matched with
several others turns the comparison into a transport problem, for which there is a mature
mathematical theory and efficient deterministic algorithms. The relaxation is not merely a
computational convenience: the strength assigned to each partial match is itself
ecologically meaningful, since it exposes exactly the role redundancies that a strict
pairing would hide. From a set of such comparisons across many communities one can ask
which species are consistently well matched everywhere, take those species and the
interactions among them as the persistent core of the system, and measure how coherently
that core aligns across the dataset. The line of work this task is drawn from develops
that programme, and applies it at continental scale to mammal food webs across
sub-Saharan Africa.

## Problem

Climate change reorganises food webs by shifting which species occur together and how they interact, so ecologists increasingly ask which interaction structures persist across ecosystems rather than which species do. Answering that requires comparing food webs from different communities by the ecological roles species occupy within them, rather than by species identity, since analogous roles are filled by different taxa in different places. Traditional comparisons align species one to one, which forces every species in one web onto a single counterpart in another and discards the redundancy that arises when several species occupy overlapping roles. Relaxing that restriction turns the comparison into a transport problem in which a species may be partially aligned with several others, and the persistent structure of a web is then read off from the species whose roles are most consistently matched across the whole dataset.

This task takes three directed food webs and computes, for one of them, how strongly the alignments of its most consistently matched species close into transitive triangles across the other two webs. The pipeline runs from local structure to that final score: each species is summarised by how often it occupies each distinct structural position within the directed three-node subgraphs of its own web; those summaries are turned into a dissimilarity between species from different webs; that dissimilarity drives a many-to-many alignment with a tradeoff between direct role matching and neighbour-role consistency; the resulting alignments then yield a per-species role similarity, a backbone of the best-matched species, and finally a transitivity score for that backbone.

Build the three webs from the arc lists below, compute the six directed pairwise alignments, meaning both directions for each of the three web pairs, and report the mean transitivity score over the backbone of the first web. Use the following configuration:

* three food webs on six species each, labelled 0 to 5, with arcs written as (prey, predator):
  - web 1: (0,3), (1,5), (2,5), (3,0), (4,5), (5,0), (5,1)
  - web 2: (0,4), (1,3), (1,4), (2,0), (3,2), (3,4), (4,3), (5,2)
  - web 3: (0,5), (1,4), (2,3), (2,5), (4,1), (4,5), (5,1), (5,2)
* structural positions are counted over induced three-node subgraphs only
* species importance distributions are uniform on each web
* tradeoff parameter alpha = 0.7, self-alignment parameter epsilon = 5.0, step size gamma = 1.0
* the alignment iteration is run for exactly 300 iterations with no early stopping, from the uniform initialisation
* a species' role similarity aggregates only its alignments to the two other webs, never to its own
* the backbone consists of the 3 species of highest role similarity
* wherever an alignment between the second and third webs is required, use the alignment from the second web to the third, with rows indexed by species of the second web and columns by species of the third, and not its transpose

Your final answer must be a single number: the mean transitivity score over the backbone of web 1.

For the compact reasoning audit, report the following numerical checkpoints: the number of distinguishable three-node role components; the web-1 per-species role-incidence totals; the smallest web-1 to web-3 dissimilarity; after the first web-1 to web-2 alignment iteration from the uniform initialisation, the total transported mass and the alignment entries for web-1 species 0 to web-2 species 0 and for web-1 species 2 to web-2 species 5; the final total transported masses for web 1 to web 2, web 1 to web 3, and web 2 to web 3; the largest positive row-budget overshoot among the returned alignments; the web-1 role-similarity values needed to justify the selected top-three backbone; and the three backbone-species transitivity values.

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

01_build_web

Goal
----
Turn an arc list into a directed food web and its underlying undirected graph.

```python
def build_web(arcs: list[tuple[int, int]], n: int) -> dict:
    '''Build the directed and underlying undirected adjacency of a food web.
    Parameters
    ----------
    arcs : list[tuple[int, int]]
        Trophic links written as (prey, predator) pairs of species labels.
    n : int
        Number of species, labelled 0 to n-1.
    Returns
    -------
    web : dict
        Mapping with keys 'n' (int), 'n_arcs' (int), 'directed' and 'undirected'
        (each a list of n rows of n ints), and 'connected' (bool).
    '''
    web = {}
    return web  # placeholder
```

### Step 2

02_motif_profiles

Goal
----
Count how often each species occupies each structural position in the web's three-node subgraphs.

```python
def motif_profiles(directed: list[list[int]]) -> list[list[int]]:
    '''Count each species' occupancy of every distinguishable three-node position.

    Parameters
    ----------
    directed : list[list[int]]
        Directed adjacency of one web, n rows of n entries in {0, 1}.

    Returns
    -------
    profiles : list[list[int]]
        One row per species; each row has one count per distinguishable
        three-node position. The role-column ordering may be any deterministic
        global ordering, but the same ordering must be used consistently
        across all calls.
    '''
    profiles = []
    return profiles  # placeholder
```

### Step 3

03_cost_matrix

Goal
----
Turn two sets of species profiles into a pairwise dissimilarity matrix.

```python
def cost_matrix(profiles_a: list[list[int]],
                profiles_b: list[list[int]]) -> list[list[float]]:
    '''Pairwise dissimilarity between species of two webs.
    Parameters
    ----------
    profiles_a : list[list[int]]
        Profiles of the first web, one row per species.
    profiles_b : list[list[int]]
        Profiles of the second web, one row per species.
    Returns
    -------
    cost : list[list[float]]
        Dissimilarity of every species of the first web against every species of
        the second, each entry rounded to 6 decimal places.
    '''
    cost = []
    return cost  # placeholder
```

### Step 4

04_alignment_iteration

Goal
----
Apply one iteration of the proximal alignment scheme.

```python
def alignment_iteration(A1: list[list[int]], A2: list[list[int]],
                        C: list[list[float]], T: list[list[float]],
                        alpha: float, eps: float, gamma: float,
                        mu: list[float], nu: list[float]) -> list[list[float]]:
    '''One iteration of the two-half-step proximal alignment scheme.
    Parameters
    ----------
    A1, A2 : list[list[int]]
        Undirected adjacency of the first and second web.
    C : list[list[float]]
        Dissimilarity matrix between the two webs.
    T : list[list[float]]
        Current alignment.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    mu, nu : list[float]
        Species importance budgets of the first and second web.
    Returns
    -------
    T_next : list[list[float]]
        Updated alignment, each entry rounded to 9 decimal places.
    '''
    T_next = []
    return T_next  # placeholder
```

### Step 5

05_align

Goal
----
Run the alignment scheme from the uniform start for a fixed number of iterations.

```python
def align(A1: list[list[int]], A2: list[list[int]], C: list[list[float]],
          alpha: float, eps: float, gamma: float,
          mu: list[float], nu: list[float], iterations: int) -> list[list[float]]:
    '''Iterate the alignment scheme from the uniform start.
    Parameters
    ----------
    A1, A2 : list[list[int]]
        Undirected adjacency of the first and second web.
    C : list[list[float]]
        Dissimilarity matrix between the two webs.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    mu, nu : list[float]
        Species importance budgets of the first and second web.
    iterations : int
        Number of iterations to perform; no early stopping.
    Returns
    -------
    T : list[list[float]]
        Final alignment, each entry rounded to 6 decimal places.
    '''
    T = []
    return T  # placeholder
```

### Step 6

06_role_similarity

Goal
----
Score each species of one web by how much cheaply-matched mass it carries to the other webs.

```python
def role_similarity(costs_to_others: list[list[list[float]]],
                    alignments_to_others: list[list[list[float]]]) -> list[float]:
    '''Aggregate cheaply-matched alignment mass for each species of one web.
    Parameters
    ----------
    costs_to_others : list[list[list[float]]]
        One dissimilarity matrix per other web, all sharing the same row set.
    alignments_to_others : list[list[list[float]]]
        The corresponding alignments, in the same order.
    Returns
    -------
    scores : list[float]
        One score per species of the focal web, rounded to 6 decimal places.
    '''
    scores = []
    return scores  # placeholder
```

### Step 7

07_backbone

Goal
----
Select the species that form the web's backbone.

```python
def backbone(scores: list[float], k: int) -> list[int]:
    '''Select the k highest-scoring species.
    Parameters
    ----------
    scores : list[float]
        Role similarity score per species.
    k : int
        Backbone size, at least 2.
    Returns
    -------
    members : list[int]
        Labels of the selected species, in ascending order.
    '''
    members = []
    return members  # placeholder
```

### Step 8

08_transitivity

Goal
----
Measure how much of a species' alignment mass closes a triangle across two other webs.

```python
def transitivity(T_ip: list[list[float]], T_iq: list[list[float]],
                 T_pq: list[list[float]], j: int) -> float:
    '''Share of a species' paired alignment mass that closes a triangle.
    Parameters
    ----------
    T_ip : list[list[float]]
        Alignment from the focal web to the first other web.
    T_iq : list[list[float]]
        Alignment from the focal web to the second other web.
    T_pq : list[list[float]]
        Alignment between the two other webs.
    j : int
        Species of the focal web being scored.
    Returns
    -------
    score : float
        Closing share, rounded to 6 decimal places.
    '''
    score = 0.0
    return score  # placeholder
```

### Step 9

09_backbone_transitivity

Goal
----
Run the whole pipeline and return the mean transitivity over the first web's backbone.

```python
def backbone_transitivity(webs: list[tuple[list[tuple[int, int]], int]],
                         alpha: float, eps: float, gamma: float,
                         iterations: int, k: int) -> float:
    '''Mean transitivity over the backbone of the first web.
    Parameters
    ----------
    webs : list[tuple[list[tuple[int, int]], int]]
        Three webs, each an (arc list, species count) pair.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    iterations : int
        Alignment iterations, with no early stopping.
    k : int
        Backbone size.
    Returns
    -------
    score : float
        Mean transitivity over the backbone, rounded to 6 decimal places.
    '''
    score = 0.0
    return score  # placeholder
```
