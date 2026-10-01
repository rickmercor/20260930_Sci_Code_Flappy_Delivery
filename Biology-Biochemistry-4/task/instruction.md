# Biology-Biochemistry-4

## Background

Proteins transmit regulatory information between distant sites through their three-dimensional fold, a phenomenon called allostery. The classical pictures, concerted transitions between discrete conformational states on one hand and entropic modulation of internal fluctuations on the other, do not by themselves describe the topological redundancy of a folded chain: a residue interacts with many spatial neighbours, so a perturbation at one site can reach another through a large number of alternative routes, and the capacity of the fold to communicate is a property of that whole set of routes rather than of any single one.

Graph representations of protein structure make this explicit. Each residue, usually represented by its C-alpha atom, is a vertex, and pairs of residues closer than a contact cutoff are joined by edges. The covalent backbone guarantees that the graph is connected, while the tertiary contacts add cycles; the number of independent cycles, the cycle rank, counts the interactions that exceed the minimum needed for connectivity and is the source of alternative signalling paths. Elastic network models such as the Gaussian network model already exploit this graph through its Laplacian (Kirchhoff) matrix, whose pseudoinverse predicts residue fluctuations and correlates well with crystallographic B-factors.

Combinatorial graph theory adds an exact statistical mechanics to this picture. The spanning trees of a graph are its minimal connected acyclic subgraphs; each contains exactly one path between any two vertices, so the set of spanning trees enumerates the ways a signal can propagate without redundancy. The matrix-tree theorem expresses the weighted count of all spanning trees as a determinant of the reduced Laplacian, and the Burton-Pemantle theorem expresses the probability that a random weighted spanning tree contains a given set of edges as a determinant built from the pseudoinverse. These identities turn exponentially large sums into linear algebra, so ensemble averages, entropies and heat capacities of the tree ensemble, and of the sub-ensemble of trees routing a signal between two chosen residues, can be evaluated without stochastic sampling.

Because all such quantities derive from the weighted Laplacian, the effect of a point mutation that slightly rearranges the backbone, opening or closing a few marginal contacts, can be computed exactly and compared between wild-type and mutant structures. The comparison separates two questions that are easily conflated: whether the overall communication capacity of a channel changes, and whether the internal distribution of that capacity across residues and routes is reorganised, which is what information-theoretic divergences between the two channel distributions measure.

## Problem

Allosteric regulation is the transmission of a perturbation between two distant sites of a protein, and it is increasingly analysed on the residue contact network in which C-alpha atoms are vertices and spatial contacts are edges. Node-centric measures such as contact density or shortest-path betweenness rank individual residues but say nothing about the collective thermodynamics of the many redundant routes a signal can take. A recent statistical-mechanical framework fills this gap by treating every spanning tree of the weighted contact graph as a topological microstate: the canonical partition function of that ensemble is evaluated exactly through Kirchhoff's matrix-tree theorem, effective resistances and edge-to-edge correlations follow from the Moore-Penrose pseudoinverse of the graph Laplacian, and the probability that a random spanning tree contains a given simple path is obtained exactly from the Burton-Pemantle theorem. A communication channel between two residues is then a truncated ensemble of simple paths with a channel entropy, a participation ratio, a residue-resolved allosteric importance, and a channel heat capacity that the framework decomposes exactly into an energetic component, a topological component and a cross-coupling component; comparing wild-type and mutant structures through these quantities and through the Jensen-Shannon divergence of their channel distributions reveals how a point mutation reroutes a signal without changing the overall capacity of the fold. The inputs of the method are the C-alpha coordinates of the two structures, a contact cutoff, an effective temperature and the two channel endpoints; the outputs are the channel thermodynamic functions and their shifts.

Consider the following 26-residue synthetic helical hairpin, whose C-alpha coordinates (in angstrom, residues numbered 1 to 26) define the wild-type structure:

| residue | x | y | z |
|---|---|---|---|
| 1 | 2.30 | 0.00 | 0.00 |
| 2 | -0.40 | 2.27 | 1.50 |
| 3 | -2.16 | -0.79 | 3.00 |
| 4 | 1.15 | -1.99 | 4.50 |
| 5 | 1.76 | 1.48 | 6.00 |
| 6 | -1.76 | 1.48 | 7.50 |
| 7 | -1.15 | -1.99 | 9.00 |
| 8 | 2.16 | -0.79 | 10.50 |
| 9 | 0.40 | 2.27 | 12.00 |
| 10 | -2.30 | 0.00 | 13.50 |
| 11 | 0.40 | -2.27 | 15.00 |
| 12 | 2.38 | 1.63 | 17.41 |
| 13 | 4.75 | 2.30 | 18.00 |
| 14 | 7.12 | 1.63 | 17.41 |
| 15 | 11.26 | 1.48 | 15.00 |
| 16 | 7.74 | 1.48 | 13.50 |
| 17 | 8.35 | -1.99 | 12.00 |
| 18 | 11.66 | -0.78 | 10.50 |
| 19 | 9.90 | 2.27 | 9.00 |
| 20 | 7.20 | -0.00 | 7.50 |
| 21 | 9.90 | -2.26 | 6.00 |
| 22 | 11.66 | 0.79 | 4.50 |
| 23 | 8.35 | 1.99 | 3.00 |
| 24 | 7.74 | -1.48 | 1.50 |
| 25 | 11.26 | -1.48 | 0.00 |
| 26 | 10.65 | 1.99 | -1.50 |

The mutant structure is identical except that the C-alpha atom of residue 20 is displaced by (+0.80, -0.30, +0.40) angstrom. Use a contact cutoff r_c = 7.8 angstrom, with the distance-dependent Boltzmann contact weights of the framework applied to every residue pair within the cutoff, covalent neighbours included. Take the effective temperature to be kT*, the temperature at which the global spanning-tree heat capacity of the wild-type contact graph, C(kT) = Var(E)/(kT)^2 with the exact tree-ensemble variance of the total contact length, is maximal: the maximum is unique on the interval from 0.2 to 5 angstrom, and kT* must be located to an absolute accuracy of 1e-10 angstrom. Use the same kT* for the mutant. The channel of interest connects residue 5 (effector site) to residue 23 (active site).

Locate kT* and construct the weighted contact Laplacian of each structure at that temperature, then evaluate the global spanning-tree thermodynamics. For the channel, enumerate every simple path from residue 5 to residue 23 with at most nine residues, obtain each path's spanning-tree probability from the pseudoinverse of the Laplacian, keep the framework's active sub-ensemble carrying 99 percent of the enumerated corridor's cumulative path probability, and evaluate the active channel's entropy, participation ratio, heat capacity and its energetic, topological and cross-coupling components for the wild type and for the mutant, together with the Jensen-Shannon divergence between the two active channel distributions and the residue whose allosteric importance falls most in relative terms, as the framework reports importance shifts. In your reasoning, report kT* and the peak value of the wild-type global heat capacity, the wild-type and mutant channel heat capacities with their energetic, topological and cross-coupling components, the Jensen-Shannon divergence, the framework's convergence ratio of the wild-type channel (truncated to full effective resistance) at an envelope of six residues, and state in one sentence what the source reports about how the three components of a channel's heat capacity shift relative to one another under mutation.

Report Delta C_X, the change of the cross-coupling component of the channel heat capacity from the wild type to the mutant (mutant minus wild type), in units of the Boltzmann constant, as a single number.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 13 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

contact_laplacian

Goal
----
Build the weighted Kirchhoff (Laplacian) matrix of the residue contact graph from the C-alpha coordinates. Every pair of residues whose C-alpha distance does not exceed the contact cutoff r_c is an edge, and each edge carries the source's distance-dependent Boltzmann weight at the effective temperature kT; covalently bonded neighbours are treated exactly like every other contact. Row sums vanish.

```python
def contact_laplacian(coords: "np.ndarray", r_c: float, kT: float) -> "np.ndarray":
    """Build the weighted Kirchhoff (Laplacian) matrix of the residue contact graph from the C-alpha coordinates. Every pair of residues whose C-alpha distance does not exceed the contact cutoff r_c is an edge, and each edge carries the source's distance-dependent Boltzmann weight at the effective temperature kT; covalently bonded neighbours are treated exactly like every other contact. Row sums vanish.

    Parameters
    ----------
    coords : np.ndarray
        C-alpha coordinates in angstrom, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom; pairs with distance <= r_c are edges.
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    L : np.ndarray
        Weighted Laplacian of shape (N, N).

    Raises
    ------
    ValueError
        If coords is not a finite (N, 3) array with N >= 3, or r_c or kT is not positive.
    """
    return L
```

### Step 2

global_tree_thermodynamics

Goal
----
Evaluate the global thermodynamics of the spanning-tree ensemble of the weighted contact graph: the log partition function ln Z from Kirchhoff's matrix-tree theorem, the free energy F = -kT ln Z, the mean energy <E>, the entropy S = (<E> - F)/kT and the heat capacity C = Var(E)/(kT)^2, where the energy of a tree is the total contact length of its edges and averages run over all spanning trees with their Boltzmann weights. Use exact identities of the framework for the ensemble averages; do not sample or finite-difference.

```python
def global_tree_thermodynamics(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Evaluate the global thermodynamics of the spanning-tree ensemble of the weighted contact graph: the log partition function ln Z from Kirchhoff's matrix-tree theorem, the free energy F = -kT ln Z, the mean energy <E>, the entropy S = (<E> - F)/kT and the heat capacity C = Var(E)/(kT)^2, where the energy of a tree is the total contact length of its edges and averages run over all spanning trees with their Boltzmann weights. Use exact identities of the framework for the ensemble averages; do not sample or finite-difference.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of a connected contact graph, shape (N, N).
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    thermo : np.ndarray
        Array [ln Z, F, <E>, S, C]. Graded at tolerance 1e-9 against the exact identities: <E> = sum_e d_e P(e in T) with P(e in T) = w_e R_e, and Var(E) = sum_e d_e^2 P(e)(1 - P(e)) + sum_(e != f) d_e d_f Cov(e, f) with Cov(e, f) = -K[e, f]^2 from the edge response matrix, where the generalised inverse behind R_e and K is the grounded inverse inv(L[1:, 1:]) padded with zeros (an SVD pseudoinverse of the singular L can miss the tolerance).

    Raises
    ------
    ValueError
        If L is not a finite symmetric Laplacian of size >= 3 with zero row sums and at least one edge, kT is not positive, or the graph is disconnected.
    """
    return thermo
```

### Step 3

edge_transfer_current

Goal
----
Compute the edge-to-edge response matrix of the framework from the Moore-Penrose pseudoinverse of the weighted Laplacian: for every ordered pair of edges the dynamic edge-to-edge distance, scaled by the square root of the product of the two edge weights so that the matrix is dimensionless. Edges are the off-diagonal negative entries of L, listed in lexicographic order of (i, j) with i < j; the diagonal equals the marginal probability that an edge belongs to a random spanning tree.

```python
def edge_transfer_current(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Compute the edge-to-edge response matrix of the framework from the Moore-Penrose pseudoinverse of the weighted Laplacian: for every ordered pair of edges the dynamic edge-to-edge distance, scaled by the square root of the product of the two edge weights so that the matrix is dimensionless. Edges are the off-diagonal negative entries of L, listed in lexicographic order of (i, j) with i < j; the diagonal equals the marginal probability that an edge belongs to a random spanning tree.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of a connected contact graph, shape (N, N).
    kT : float
        Effective temperature in angstrom (accepted for interface uniformity; the response matrix depends on L alone).

    Returns
    -------
    K : np.ndarray
        Edge response matrix of shape (E, E): K[a, b] = sqrt(w_a w_b) * (e_i - e_j)^T G (e_k - e_l) for edges a = (i, j), b = (k, l), where G is the generalised inverse of L. Because edge-difference vectors are orthogonal to the constant null vector, any generalised inverse gives the same K; the graded values (tolerance 1e-9) are produced with the grounded inverse G = inv(L[1:, 1:]) padded with a zero first row and column, which is exact and well conditioned, whereas an SVD-based pseudoinverse of the full singular matrix can miss the tolerance.

    Raises
    ------
    ValueError
        If L is not a finite symmetric Laplacian of size >= 3 with zero row sums and at least one edge, or kT is not positive.
    """
    return K
```

### Step 4

heat_capacity_peak_temperature

Goal
----
Find the effective temperature kT* inside the open interval (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2 of the contact graph is maximal, where the contact weights, the tree energy and the exact tree-ensemble variance are those of the earlier steps, rebuilt at every temperature considered. The maximum is unique in the interval. Return kT* to an absolute accuracy of 1e-10 angstrom; a method whose result is limited by the precision of function values alone does not reach that accuracy.

```python
def heat_capacity_peak_temperature(coords: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float) -> float:
    """Find the effective temperature kT* inside the open interval (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2 of the contact graph is maximal, where the contact weights, the tree energy and the exact tree-ensemble variance are those of the earlier steps, rebuilt at every temperature considered. The maximum is unique in the interval. Return kT* to an absolute accuracy of 1e-10 angstrom; a method whose result is limited by the precision of function values alone does not reach that accuracy.

    Parameters
    ----------
    coords : np.ndarray
        C-alpha coordinates in angstrom, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom.
    kT_lo : float
        Lower end of the search interval (angstrom), > 0.
    kT_hi : float
        Upper end of the search interval (angstrom), > kT_lo.

    Returns
    -------
    kT_peak : float
        Temperature of the unique interior maximum of C(kT), to 1e-10.

    Raises
    ------
    ValueError
        If coords is not a finite (N, 3) array with N >= 3, r_c is not positive, the interval is not 0 < kT_lo < kT_hi, or C has no interior maximum bracketed by the interval.
    """
    return kT_peak
```

### Step 5

channel_paths

Goal
----
Enumerate every simple path of the contact graph from residue s to residue t whose number of nodes lies between 2 and max_nodes inclusive, following the framework's node-length envelope. Return them as an integer array with one row per path, node indices in order of traversal, unused trailing entries set to -1, rows sorted lexicographically by node sequence.

```python
def channel_paths(L: "np.ndarray", s: int, t: int, max_nodes: int) -> "np.ndarray":
    """Enumerate every simple path of the contact graph from residue s to residue t whose number of nodes lies between 2 and max_nodes inclusive, following the framework's node-length envelope. Return them as an integer array with one row per path, node indices in order of traversal, unused trailing entries set to -1, rows sorted lexicographically by node sequence.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of the contact graph, shape (N, N).
    s : int
        Source residue index.
    t : int
        Target residue index.
    max_nodes : int
        Maximum number of nodes in a path (>= 2).

    Returns
    -------
    paths : np.ndarray
        Sorted path table of shape (m, max_nodes).

    Raises
    ------
    ValueError
        If L is not a valid Laplacian, s or t is not a valid residue index, s == t, or max_nodes < 2.
    """
    return paths
```

### Step 6

path_energy_table

Goal
----
For every path of the table compute, from the weighted Laplacian and the edge response matrix, the framework's three per-path quantities: the exact probability that a random weighted spanning tree contains all edges of the path, the physical path length (sum of the C-alpha distances of its contacts), and the topological contribution to the path's effective energy as the source defines it, in angstrom. Rows are aligned with the input paths.

```python
def path_energy_table(L: "np.ndarray", kT: float, K: "np.ndarray", paths: "np.ndarray") -> "np.ndarray":
    """For every path of the table compute, from the weighted Laplacian and the edge response matrix, the framework's three per-path quantities: the exact probability that a random weighted spanning tree contains all edges of the path, the physical path length (sum of the C-alpha distances of its contacts), and the topological contribution to the path's effective energy as the source defines it, in angstrom. Rows are aligned with the input paths.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of the contact graph, shape (N, N).
    kT : float
        Effective temperature in angstrom.
    K : np.ndarray
        Edge response matrix of shape (E, E) in lexicographic edge order.
    paths : np.ndarray
        Path table of shape (m, max_nodes) padded with -1.

    Returns
    -------
    table : np.ndarray
        Array of shape (m, 3) with columns [P, E_E, E_T]: P = det of the path's block of K; E_E = sum of the C-alpha distances of the path's edges (recovered from the weights as d = -kT ln w); E_T = -kT ln det of the same block with the sqrt(w_a w_b) factors removed, i.e. the unweighted edge-to-edge distance block, so that P = exp(-(E_E + E_T)/kT).

    Raises
    ------
    ValueError
        If L is not a valid Laplacian, kT is not positive, K is not E x E, paths is empty or malformed, or a path uses a non-edge.
    """
    return table
```

### Step 7

active_channel_weights

Goal
----
Apply the framework's thermodynamics-driven truncation to a vector of raw path probabilities: rank the paths by ascending surprisal and retain the leading paths that together carry the cumulative-probability share eta of the corridor, then return the occupancy weights of the retained paths, normalised to one, with zero for every discarded path.

```python
def active_channel_weights(P: "np.ndarray", eta: float) -> "np.ndarray":
    """Apply the framework's thermodynamics-driven truncation to a vector of raw path probabilities: rank the paths by ascending surprisal and retain the leading paths that together carry the cumulative-probability share eta of the corridor, then return the occupancy weights of the retained paths, normalised to one, with zero for every discarded path.

    Parameters
    ----------
    P : np.ndarray
        Raw path probabilities, shape (m,), all positive.
    eta : float
        Cumulative-probability threshold in (0, 1].

    Returns
    -------
    p : np.ndarray
        Occupancy weights of shape (m,) summing to one.

    Raises
    ------
    ValueError
        If P is empty or has a non-positive or non-finite entry, or eta is not in (0, 1].
    """
    return p
```

### Step 8

channel_thermodynamics

Goal
----
From the occupancy weights of the active channel and the per-path table, compute the channel entropy S = -sum p ln p, the participation ratio, the channel heat capacity C = Var(E)/(kT)^2 of the effective path energy E = E_E + E_T, and its decomposition into the energetic component from E_E, the topological component from E_T and the cross-coupling component from their covariance, all averaged over the active distribution.

```python
def channel_thermodynamics(p: "np.ndarray", table: "np.ndarray", kT: float) -> "np.ndarray":
    """From the occupancy weights of the active channel and the per-path table, compute the channel entropy S = -sum p ln p, the participation ratio, the channel heat capacity C = Var(E)/(kT)^2 of the effective path energy E = E_E + E_T, and its decomposition into the energetic component from E_E, the topological component from E_T and the cross-coupling component from their covariance, all averaged over the active distribution.

    Parameters
    ----------
    p : np.ndarray
        Occupancy weights of shape (m,) summing to one (zeros for inactive paths).
    table : np.ndarray
        Per-path table of shape (m, 3) with columns [P, E_E, E_T].
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    thermo : np.ndarray
        Array [S, PR, C, C_E, C_T, C_X].

    Raises
    ------
    ValueError
        If p is not a probability vector aligned with the rows of table, or kT is not positive.
    """
    return thermo
```

### Step 9

allosteric_importance

Goal
----
Compute the allosteric importance of every residue for a channel: the fraction of the active channel ensemble whose path passes through that residue as an interior residue. The two endpoints of the channel lie on every path and are assigned zero. Return one value per residue of the protein.

```python
def allosteric_importance(paths: "np.ndarray", p: "np.ndarray", n_res: int) -> "np.ndarray":
    """Compute the allosteric importance of every residue for a channel: the fraction of the active channel ensemble whose path passes through that residue as an interior residue. The two endpoints of the channel lie on every path and are assigned zero. Return one value per residue of the protein.

    Parameters
    ----------
    paths : np.ndarray
        Path table of shape (m, max_nodes) padded with -1.
    p : np.ndarray
        Occupancy weights of shape (m,) summing to one.
    n_res : int
        Number of residues N.

    Returns
    -------
    importance : np.ndarray
        Array of shape (n_res,): importance[k] = sum of the occupancy weights of the active paths that contain residue k strictly between their two endpoints; the endpoints and residues on no active path receive 0.

    Raises
    ------
    ValueError
        If paths and p are not aligned, p is not a probability vector, n_res < 3, or a path indexes a residue >= n_res.
    """
    return importance
```

### Step 10

channel_divergence

Goal
----
Compute the Jensen-Shannon divergence (natural logarithm) between the active channel distributions of two structures of the same protein, treating each path as a symbol of the channel alphabet identified by its node sequence, with a path absent from one active ensemble carrying zero probability there.

```python
def channel_divergence(paths_a: "np.ndarray", p_a: "np.ndarray", paths_b: "np.ndarray", p_b: "np.ndarray") -> float:
    """Compute the Jensen-Shannon divergence (natural logarithm) between the active channel distributions of two structures of the same protein, treating each path as a symbol of the channel alphabet identified by its node sequence, with a path absent from one active ensemble carrying zero probability there.

    Parameters
    ----------
    paths_a : np.ndarray
        Path table of structure A, shape (m_a, max_nodes).
    p_a : np.ndarray
        Occupancy weights of structure A, shape (m_a,).
    paths_b : np.ndarray
        Path table of structure B, shape (m_b, max_nodes).
    p_b : np.ndarray
        Occupancy weights of structure B, shape (m_b,).

    Returns
    -------
    d_js : float
        Jensen-Shannon divergence in nats.

    Raises
    ------
    ValueError
        If either path table is not aligned with its probability vector, or a probability vector does not sum to one.
    """
    return d_js
```

### Step 11

channel_convergence_ratio

Goal
----
Compute the framework's convergence ratio of a channel at a given node-length envelope: the effective resistance between the two endpoints evaluated on the source's truncated channel subgraph built from the residues visited by the enumerated simple paths, divided by the exact effective resistance of the full contact graph. Construct the truncated subgraph exactly as the source prescribes for this convergence test.

```python
def channel_convergence_ratio(L: "np.ndarray", s: int, t: int, max_nodes: int) -> float:
    """Compute the framework's convergence ratio of a channel at a given node-length envelope: the effective resistance between the two endpoints evaluated on the source's truncated channel subgraph built from the residues visited by the enumerated simple paths, divided by the exact effective resistance of the full contact graph. Construct the truncated subgraph exactly as the source prescribes for this convergence test.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of the full contact graph, shape (N, N).
    s : int
        Source residue index.
    t : int
        Target residue index.
    max_nodes : int
        Node-length envelope L of the enumerated simple paths (>= 2).

    Returns
    -------
    ratio : float
        Truncated-to-full effective-resistance ratio, dimensionless.

    Raises
    ------
    ValueError
        If L is not a valid Laplacian, s or t is not a valid residue index, s == t, or max_nodes < 2.
    """
    return ratio
```

### Step 12

importance_shift_table

Goal
----
Tabulate the relative shift of allosteric importance from the wild type to the mutant, in percent of the wild-type value, for every residue whose wild-type importance is at least 0.03, which is the source's reporting rule for these shifts; residues below that threshold are assigned zero. Return one value per residue.

```python
def importance_shift_table(importance_wt: "np.ndarray", importance_mut: "np.ndarray") -> "np.ndarray":
    """Tabulate the relative shift of allosteric importance from the wild type to the mutant, in percent of the wild-type value, for every residue whose wild-type importance is at least 0.03, which is the source's reporting rule for these shifts; residues below that threshold are assigned zero. Return one value per residue.

    Parameters
    ----------
    importance_wt : np.ndarray
        Wild-type allosteric importance of every residue, shape (n_res,).
    importance_mut : np.ndarray
        Mutant allosteric importance of every residue, shape (n_res,).

    Returns
    -------
    shifts : np.ndarray
        Relative shifts in percent, shape (n_res,).

    Raises
    ------
    ValueError
        If the two vectors differ in length, have fewer than three entries, or contain negative or non-finite values.
    """
    return shifts
```

### Step 13

mutation_cross_coupling_shift

Goal
----
Orchestrate the full pipeline for the wild-type and the mutant coordinates: locate the wild type's heat-capacity peak temperature kT* in (kT_lo, kT_hi) and evaluate both structures at that temperature; build each contact Laplacian, its global thermodynamics and edge response matrix, the channel path table between residues s and t inside the node-length envelope, the per-path energies, the active channel weights at threshold eta, the channel thermodynamics, the residue importances, the convergence ratio of each channel at the envelope, the divergence between the two channels and the table of importance shifts, and return the shift of the cross-coupling component of the channel heat capacity, mutant minus wild type. Call the earlier step functions rather than reimplementing them.

```python
def mutation_cross_coupling_shift(coords_wt: "np.ndarray", coords_mut: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float,
                                          s: int, t: int, max_nodes: int, eta: float) -> float:
    """Orchestrate the full pipeline for the wild-type and the mutant coordinates: locate the wild type's heat-capacity peak temperature kT* in (kT_lo, kT_hi) and evaluate both structures at that temperature; build each contact Laplacian, its global thermodynamics and edge response matrix, the channel path table between residues s and t inside the node-length envelope, the per-path energies, the active channel weights at threshold eta, the channel thermodynamics, the residue importances, the convergence ratio of each channel at the envelope, the divergence between the two channels and the table of importance shifts, and return the shift of the cross-coupling component of the channel heat capacity, mutant minus wild type. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    coords_wt : np.ndarray
        Wild-type C-alpha coordinates, shape (N, 3).
    coords_mut : np.ndarray
        Mutant C-alpha coordinates, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom.
    kT_lo : float
        Lower end of the peak-temperature search interval (angstrom).
    kT_hi : float
        Upper end of the peak-temperature search interval (angstrom).
    s : int
        Source residue index.
    t : int
        Target residue index.
    max_nodes : int
        Maximum number of nodes in a channel path.
    eta : float
        Cumulative-probability threshold of the active channel.

    Returns
    -------
    delta_cx : float
        Shift of the cross-coupling heat-capacity component at kT*.

    Raises
    ------
    ValueError
        If the coordinate arrays are not finite (N, 3) arrays of the same shape, or any parameter is invalid for the steps above.
    """
    return delta_cx
```
