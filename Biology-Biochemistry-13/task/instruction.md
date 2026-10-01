# Biology-Biochemistry-13

## Background

Closed-form parametrizations of the positive steady states of mass-action networks let one read off robustness and multistationarity without committing to particular parameter values. Structure-based frameworks obtain them by combining independent network decompositions with network translations, which rewrite a network that is not weakly reversible or not deficiency zero as a dynamically equivalent generalized network whose equilibria can be parametrized.

The task below applies that route to a bacterial two-component osmosensing module and reduces the result to one steady-state concentration.

## Problem

I model EnvZ–OmpR osmosensing as a mass-action network on nine species: the sensor kinase EnvZ in free (X), ADP-bound (XD), ATP-bound (XT) and phosphorylated (Xp) form, the response regulator OmpR (Y) and phospho-OmpR (Yp), and the complexes XpY, XDYp and XTYp. Its fourteen reactions, with my rate constants in consistent units in brackets, are XD → X (1.93), X → XD (1.72), X → XT (1.90), XT → X (0.19), XT → Xp (0.51), Xp + Y → XpY (3.39), XpY → Xp + Y (2.90), XpY → X + Yp (1.45), XD + Yp → XDYp (0.28), XDYp → XD + Yp (4.31), XDYp → XD + Y (0.35), XT + Yp → XTYp (0.60), XTYp → XT + Yp (2.84) and XTYp → XT + Y (1.66). The stoichiometric compatibility class I measured has total EnvZ 3.31 and total OmpR 4.21, each summed over every species that contains that protein. I want its positive steady states obtained analytically from the network structure rather than by simulation, using the elementary-flux-mode-based network translation in which reactions that leave the same complex keep one translated source complex, and then parametrizing the positive equilibria of the resulting generalized network. Report the steady-state concentration of the phosphotransfer complex XpY in this class to ten significant figures; if the class has more than one positive steady state, report the largest such XpY. In `<reasoning>`, the scalars I need you to state are: the reactions forming each block of the finest independent decomposition, the number of elementary flux modes of the full network and the reactions of those that are not reversible pairs, the kinetic deficiency of the translated generalized network of the whole system, its vertex count, linkage-class count and kinetic rank, which translated complex carries more than one kinetic complex, and the part of the network that produces that deficiency, and the number of positive steady states in this class together with the argument that fixes that number. Those are the derived quantities that determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied data and bulk tables, which this problem does not need.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_decompose_independent_subnetworks

Goal
----
Partition the reactions of a mass-action network into its finest independent decomposition.

```python
import numpy as np
def decompose_independent_subnetworks(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Return the block label of every reaction in the finest independent decomposition.

    Reaction ``k`` converts the complex ``source_complexes[k]`` into the
    complex ``product_complexes[k]``; both rows are nonnegative integer
    stoichiometric vectors over the same ordered species, and the reaction
    vector is ``product_complexes[k] - source_complexes[k]``. The finest
    independent decomposition is the partition of the reactions into the
    largest number of blocks such that the rank of the matrix of all reaction
    vectors equals the sum of the ranks of the reaction vectors of each block.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(r,)`` holding the block of each reaction.
        Blocks are numbered ``0, 1, 2, ...`` in increasing order of the
        smallest reaction index they contain.

    Raises
    ------
    ValueError
        If the two arrays are not two-dimensional with the same shape and at
        least one reaction, if any entry is negative, non-finite or not an
        integer, or if some reaction has identical source and product
        complexes.
    """
    return np.zeros(0, dtype=int)
```

### Step 2

02_compute_elementary_flux_modes

Goal
----
Enumerate the elementary flux modes of a reaction network as primitive integer vectors.

```python
import numpy as np
def compute_elementary_flux_modes(
    source_complexes: np.ndarray, product_complexes: np.ndarray
) -> np.ndarray:
    """Return the elementary flux modes of an irreversible reaction network.

    Let ``N`` be the ``(m, r)`` stoichiometric matrix whose column ``k`` is
    ``product_complexes[k] - source_complexes[k]``. A flux mode is a nonzero
    vector ``v >= 0`` with ``N @ v = 0``; it is elementary when no other flux
    mode has a support (set of nonzero entries) strictly contained in its
    support. Every reaction is irreversible, so a reversible pair appears as
    two separate reactions.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(p, r)``, one elementary flux mode per
        row, each scaled to nonnegative integers whose greatest common divisor
        is 1. Rows are ordered by comparing the increasing tuples of the
        indices of their nonzero entries lexicographically. A network without
        flux modes gives shape ``(0, r)``.

    Raises
    ------
    ValueError
        If the two arrays are not two-dimensional with the same shape and at
        least one reaction, if any entry is negative, non-finite or not an
        integer, or if some reaction has identical source and product
        complexes.
    """
    return np.zeros((0, 0), dtype=int)
```

### Step 3

03_build_reaction_graph

Goal
----
Construct a directed graph on the reactions whose simple cycles are exactly the elementary flux modes and which respects common source complexes.

```python
import numpy as np
def build_reaction_graph(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    flux_modes: np.ndarray,
) -> np.ndarray:
    """Return the edges of a compatible reaction-to-reaction graph.

    The vertices are the reactions ``0, ..., r - 1``. The returned graph must
    satisfy both conditions below, and any graph that does is acceptable.

    * Common-source compatibility: if reactions ``i`` and ``j`` have the same
      source complex and ``(k, i)`` is an edge, then ``(k, j)`` is an edge.
    * Flux-mode compatibility: the vertex sets of the simple directed cycles
      of the graph are exactly the supports of the rows of ``flux_modes``.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    flux_modes : np.ndarray
        Elementary flux modes of the network, shape ``(p, r)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(q, 2)``; row ``(i, j)`` is the directed
        edge from reaction ``i`` to reaction ``j``. Edges are distinct, have
        ``i != j``, and rows are sorted in increasing lexicographic order.

    Raises
    ------
    ValueError
        If the complex arrays are invalid (not equal ``(r, m)`` shapes with
        ``r >= 1``, a negative, non-finite or non-integer entry, or a reaction
        with identical source and product), if ``flux_modes`` is not a
        nonempty ``(p, r)`` array, if any flux-mode entry is not 0 or 1, if
        some mode is zero or some reaction belongs to no mode, or if no graph
        satisfies both compatibility conditions; returning a graph that
        violates either condition does not satisfy this contract.
    """
    return np.zeros((0, 2), dtype=int)
```

### Step 4

04_compute_translation_complexes

Goal
----
Propagate translation complexes along the reaction-to-reaction graph so that every edge becomes product-to-source compatible.

```python
import numpy as np
def compute_translation_complexes(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    reaction_edges: np.ndarray,
) -> np.ndarray:
    """Return the translation complex of every reaction.

    Reaction ``k`` is translated to ``source_complexes[k] + alpha[k] ->
    product_complexes[k] + alpha[k]``. Every edge ``(i, j)`` of the
    reaction-to-reaction graph must become product-to-source compatible, that
    is, the translated product of reaction ``i`` equals the translated source
    of reaction ``j``. On each weakly connected component of the graph (a
    reaction touched by no edge is a component by itself), ``alpha`` is
    anchored at zero on the component's lowest-index reaction and propagated
    along the edges; afterwards, for every species separately, all
    translation complexes of the component are raised by the smallest
    nonnegative integer that makes every translated source and product
    complex of that component nonnegative.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    reaction_edges : np.ndarray
        Integer array with shape ``(q, 2)`` of directed edges between
        reactions.

    Returns
    -------
    np.ndarray
        Integer array ``alpha`` with shape ``(r, m)``.

    Raises
    ------
    ValueError
        If the complex arrays are invalid (not equal ``(r, m)`` shapes with
        ``r >= 1``, or a negative, non-finite or non-integer entry), if
        ``reaction_edges`` is not a ``(q, 2)`` integer array of in-range,
        non-self-loop edges (an out-of-range index must raise ValueError,
        not IndexError), or if no choice of translation complexes makes
        every edge product-to-source compatible (the equations along the
        edges are inconsistent).
    """
    return np.zeros((0, 0), dtype=int)
```

### Step 5

05_compute_network_deficiency

Goal
----
Count complexes, linkage classes and stoichiometric rank of a (translated) network and report its deficiency and weak reversibility.

```python
import numpy as np
def compute_network_deficiency(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return ``[n, l, s, deficiency, weakly_reversible]`` of a translated network.

    Reaction ``k`` of the translated network is
    ``source_complexes[k] + translation[k] -> product_complexes[k] +
    translation[k]``; pass a zero ``translation`` for the original network.
    ``n`` is the number of distinct complexes of the translated network,
    ``l`` its number of linkage classes, ``s`` the rank of its stoichiometric
    matrix, ``deficiency = n - l - s``, and ``weakly_reversible`` is 1 when
    every reaction lies on a directed cycle of the complex graph and 0
    otherwise.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    translation : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    np.ndarray
        Integer array with shape ``(5,)``.

    Raises
    ------
    ValueError
        If the three arrays do not share a two-dimensional ``(r, m)`` shape
        with ``r >= 1``, if any entry is non-finite or not an integer, if the
        original or translated complexes have a negative entry, or if some
        reaction has identical source and product complexes.
    """
    return np.zeros(5, dtype=int)
```

### Step 6

06_build_generalized_network

Goal
----
Turn a translated network into a generalized network with stoichiometric and kinetic complexes, splitting shared translated complexes by phantom edges.

```python
import numpy as np
def build_generalized_network(
    source_complexes: np.ndarray,
    product_complexes: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(stoichiometric_complexes, kinetic_complexes, edges)``.

    Reaction ``k`` has translated source ``c_s(k) = source_complexes[k] +
    translation[k]``, translated product ``c_p(k) = product_complexes[k] +
    translation[k]`` and kinetic complex ``source_complexes[k]``. Distinct
    translated complexes are ordered by first appearance when the reactions
    are scanned in index order, reading ``c_s(k)`` before ``c_p(k)``. A
    translated complex that is the translated source of reactions with ``d``
    different kinetic complexes becomes ``d`` vertices, one per kinetic
    complex, ordered by the smallest index of a reaction carrying that
    kinetic complex; vertices are numbered by walking the translated
    complexes in their order and, within each, its vertices in their order.
    Edges are listed as follows: first one effective edge per reaction ``k``
    in increasing ``k``, from the vertex of ``c_s(k)`` with kinetic complex
    ``source_complexes[k]`` to the first vertex of ``c_p(k)``, with third
    entry ``k``; then, walking the translated complexes in order, one phantom
    edge from each of its vertices to the next vertex of the same translated
    complex, with third entry ``-1``.

    Parameters
    ----------
    source_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    product_complexes : np.ndarray
        Integer array with shape ``(r, m)``.
    translation : np.ndarray
        Integer array with shape ``(r, m)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Integer arrays with shapes ``(V, m)`` (stoichiometric complex of each
        vertex), ``(V, m)`` (kinetic complex of each vertex) and ``(E, 3)``
        (rows ``[tail vertex, head vertex, reaction index or -1]``).

    Raises
    ------
    ValueError
        If the three arrays do not share a two-dimensional ``(r, m)`` shape
        with ``r >= 1``, if any entry is non-finite or not an integer, if an
        original or translated complex has a negative entry, or if some
        translated complex is the translated source of no reaction, so that
        it has no kinetic complex (raise ValueError there, not a KeyError or
        IndexError).
    """
    return (np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int), np.zeros((0, 3), dtype=int))
```

### Step 7

07_compute_tree_constants

Goal
----
Evaluate the spanning-tree constant of every vertex of a weighted generalized network.

```python
import numpy as np
def compute_tree_constants(
    num_vertices: int, gcrn_edges: np.ndarray, edge_rates: np.ndarray
) -> np.ndarray:
    """Return the tree constant of every vertex.

    The tree constant ``K_v`` is the sum, over all spanning trees of the
    linkage class of ``v`` that are directed towards ``v`` (every other vertex
    of the class has exactly one outgoing tree edge and a directed tree path
    to ``v``), of the product of the rates of the tree edges. A vertex that
    is alone in its linkage class has ``K_v = 1``. Parallel edges are
    distinct edges.

    Parameters
    ----------
    num_vertices : int
        Number of vertices ``V``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 2)`` or ``(E, 3)``; the first two
        columns are the tail and head vertex of each directed edge and any
        further column is ignored.
    edge_rates : np.ndarray
        Positive rate of each edge, shape ``(E,)``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(V,)``.

    Raises
    ------
    ValueError
        If ``num_vertices`` is not a positive integer, if an edge index is out
        of range or an edge is a self-loop, or if ``edge_rates`` does not
        have one finite, strictly positive entry per edge (a zero rate
        included).
    """
    return np.zeros(0, dtype=float)
```

### Step 8

08_solve_phantom_rate

Goal
----
Fix the rate of the phantom edge so that complex-balanced equilibria of a generalized network with kinetic deficiency one exist.

```python
import numpy as np
def solve_phantom_rate(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    rate_constants: np.ndarray,
    tree_constants_fn,
    rate_bracket: tuple = (1e-8, 1e8),
) -> np.ndarray:
    """Return the phantom-edge rate required for complex-balanced equilibria.

    Row ``e`` of ``gcrn_edges`` is ``[tail, head, k]``: an effective edge
    (``k >= 0``) has rate ``rate_constants[k]`` and a phantom edge
    (``k = -1``) has the unknown rate ``sigma``. Tree constants are obtained
    as ``tree_constants_fn(V, gcrn_edges, edge_rates)`` with ``V`` the number
    of vertices and ``edge_rates`` the rate of every edge. Complex-balanced
    equilibria exist when some vector ``z`` satisfies ``(y_h - y_t) @ z =
    log(K_h) - log(K_t)`` for every edge, where ``y_v`` is the kinetic
    complex of vertex ``v`` and ``K_v`` its tree constant. The kinetic
    deficiency is ``V`` minus the number of linkage classes minus the rank of
    the vectors ``y_h - y_t`` over all edges.

    If there is no phantom edge and the kinetic deficiency is 0, return an
    empty array. If there is exactly one phantom edge and the kinetic
    deficiency is 1, return the ``sigma`` inside ``rate_bracket`` that
    satisfies the existence condition, with relative accuracy ``1e-12``.

    Parameters
    ----------
    kinetic_complexes : np.ndarray
        Integer array with shape ``(V, m)``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 3)``.
    rate_constants : np.ndarray
        Positive rate constant of every reaction, indexed by ``k``.
    tree_constants_fn : callable
        Function ``(V, gcrn_edges, edge_rates) -> (V,)`` tree constants.
    rate_bracket : tuple
        ``(low, high)`` with ``0 < low < high``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(0,)`` or ``(1,)``.

    Raises
    ------
    ValueError
        If the arrays are malformed (edge indices out of range, a reaction
        index without a finite positive rate constant), if the bracket is not
        ``0 < low < high`` with finite ends, if the numbers of phantom edges
        and the kinetic deficiency are not one of the two supported pairs
        (for example one phantom edge with kinetic deficiency 0), or if no
        ``sigma`` in the closed bracket satisfies the condition.
    """
    return np.zeros(0, dtype=float)
```

### Step 9

09_parametrize_steady_states

Goal
----
Express the complex-balanced equilibria of a generalized network as a log-linear function of designated free species.

```python
import numpy as np
def parametrize_steady_states(
    kinetic_complexes: np.ndarray,
    gcrn_edges: np.ndarray,
    tree_constants: np.ndarray,
    free_species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(log_offset, exponents)`` of the complex-balanced equilibria.

    With ``z = log(x)`` and ``y_v`` the kinetic complex of vertex ``v``, the
    complex-balanced equilibria are the solutions of ``(y_h - y_t) @ z =
    log(K_h) - log(K_t)`` over every edge ``(t, h)``. Taking ``z`` of the
    ``d`` species listed in ``free_species`` as free coordinates, every
    solution is ``z = log_offset + exponents @ z[free_species]``. The rows of
    the free species have zero offset and are the corresponding unit rows.

    Parameters
    ----------
    kinetic_complexes : np.ndarray
        Integer array with shape ``(V, m)``.
    gcrn_edges : np.ndarray
        Integer array with shape ``(E, 2)`` or ``(E, 3)`` whose first two
        columns are the tail and head vertex of each edge.
    tree_constants : np.ndarray
        Positive tree constant of each vertex, shape ``(V,)``.
    free_species : np.ndarray
        Distinct species indices, shape ``(d,)`` (possibly empty).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float arrays with shapes ``(m,)`` and ``(m, d)``.

    Raises
    ------
    ValueError
        If the arrays are malformed or an edge index is out of range, if a
        tree constant is not finite and positive, if ``free_species`` holds a
        repeated or out-of-range index, if the free species do not
        parametrize the solution set one-to-one (the solutions do not form a
        ``d``-dimensional family in which the free coordinates determine all
        others), or if the equations have no solution (relative residual
        above ``1e-9``); returning a least-squares solution in either of the
        last two situations does not satisfy this contract.
    """
    return (np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float))
```

### Step 10

10_solve_class_steady_state

Goal
----
Impose the conservation laws on a log-linear steady-state parametrization and return the positive steady state of the class.

```python
import numpy as np
def solve_class_steady_state(
    log_offset: np.ndarray,
    exponents: np.ndarray,
    conservation_matrix: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    """Return the positive steady state whose conserved totals equal ``totals``.

    Candidate states are ``x(z) = exp(log_offset + exponents @ z)`` with
    ``z`` in ``R**d``. Return the ``x(z)`` satisfying
    ``conservation_matrix @ x(z) = totals`` with a relative error below
    ``1e-12`` in every total; the inputs are such that exactly one such state
    exists. When ``d = 0`` the only candidate is ``exp(log_offset)``, which is
    returned without any condition being imposed.

    Parameters
    ----------
    log_offset : np.ndarray
        Float array with shape ``(m,)``.
    exponents : np.ndarray
        Float array with shape ``(m, d)``.
    conservation_matrix : np.ndarray
        Nonnegative weights with shape ``(d, m)``; every row has a positive
        entry.
    totals : np.ndarray
        Positive conserved totals with shape ``(d,)``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(m,)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent (the number of conservation laws must
        equal ``d``; raise ValueError rather than letting a linear-algebra
        error propagate), if any input is non-finite, if a weight is
        negative or a row of weights is zero, if a total is not positive
        (zero included), or if no state meeting the tolerance is found.
    """
    return np.zeros(0, dtype=float)
```

### Step 11

11_run_equilibrium_pipeline

Goal
----
Compose every earlier step to obtain one species concentration at the positive steady state of a stoichiometric compatibility class.

```python
import numpy as np
def run_equilibrium_pipeline(
    source_complexes: "np.ndarray | None" = None,
    product_complexes: "np.ndarray | None" = None,
    rate_constants: "np.ndarray | None" = None,
    conservation_matrix: "np.ndarray | None" = None,
    totals: "np.ndarray | None" = None,
    free_species: "np.ndarray | None" = None,
    target_species: "int | None" = None,
    rate_bracket: tuple = (1e-8, 1e8),
) -> float:
    """Return one species concentration at the positive steady state of a class.

    The mass-action network is decomposed into its finest independent
    subnetworks. Every subnetwork that is not both weakly reversible and of
    deficiency zero is translated through its elementary flux modes, a
    compatible reaction-to-reaction graph and the resulting translation
    complexes, while the other subnetworks keep a zero translation. The
    translated subnetworks are merged back into one network, which must be
    weakly reversible with deficiency zero. Its generalized network is built,
    the phantom rate is fixed when the kinetic deficiency requires it, tree
    constants are evaluated with the reaction rate constants (and the phantom
    rate), the complex-balanced equilibria are parametrized by
    ``free_species``, and the conserved totals are imposed.

    Any argument left as ``None`` takes its value from the EnvZ-OmpR
    benchmark of the problem statement: species ordered X, XD, XT, Xp, Y,
    Yp, XpY, XDYp, XTYp; the fourteen reactions and rate constants in the
    order listed there; conservation rows for total EnvZ and total OmpR
    with totals 3.31 and 4.21; free species XpY and Y (indices 6 and 4);
    and target species XpY (index 6).

    Parameters
    ----------
    source_complexes, product_complexes : np.ndarray
        Integer arrays with shape ``(r, m)``.
    rate_constants : np.ndarray
        Positive mass-action rate constant of each reaction, shape ``(r,)``.
    conservation_matrix : np.ndarray
        Nonnegative conservation weights, shape ``(d, m)``.
    totals : np.ndarray
        Positive conserved totals, shape ``(d,)``.
    free_species : np.ndarray
        Species indices used as free coordinates, shape ``(d,)``.
    target_species : int
        Index of the species whose steady-state concentration is returned.
    rate_bracket : tuple
        Search bracket for the phantom rate.

    Returns
    -------
    float
        Steady-state concentration of ``target_species``.

    Raises
    ------
    ValueError
        If any stage rejects its input, if the merged translated network is
        not weakly reversible with deficiency zero, if ``rate_constants`` does
        not hold ``r`` finite positive values, or if ``target_species`` is not
        a valid species index.
    """
    return 0.0
```
