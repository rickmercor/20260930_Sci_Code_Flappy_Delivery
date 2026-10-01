# Physics-Quantum_Information_Computing-13

## Background

Quantum LDPC codes try to get a useful combination of sparse checks, scalable blocklength, and efficient decoding. Hypergraph-product constructions are particularly attractive because a classical sparse bipartite graph can be lifted into a CSS quantum code. For a seed adjacency matrix $H\in\mathbb F_2^{B\times A}$, the resulting square hypergraph-product construction has qubits associated with $A\times A$ and $B\times B$. In that representation an $X$ error $(X_A,X_B)$ generates syndrome $H X_A+X_BH$, while the corresponding $Z$ error generates $Z_AH^\top+H^\top Z_B$.

The interesting part of this decoder is that it does not try to guess the entire physical error from the syndrome in one shot. It first looks at which syndrome rows and columns are active, then grows carefully chosen regions of the seed graph. A $\beta$-neighborhood is deliberately strict: a vertex joins only when more than a $\beta$ fraction of its incident edges point into the current set. That tiny word “more” matters in this benchmark because $\beta=0.5$, so exact half-degree cases must stay out. The underlying definition and the alternating graph-growth construction are central pieces of the method.

The second ingredient is peeling. A peelable set can be matched to unique neighbors in an order that makes the relevant submatrix triangular and therefore invertible over $\mathbb F_2$. The source explicitly warns that several valid peelings can exist, and that the selected partner set becomes unique only after an algorithm for choosing the peeling has been fixed. That is why this benchmark gives an explicit lexicographic rule instead of leaving tie-breaking to the solver.

Once the row-side and column-side regions have been identified and peeled, the decoder uses the corresponding embedded binary inverses to remove one part of the syndrome and then the other. The same structure works in transposed form for $Z$ errors. Computationally, this is not an arbitrary matrix-inversion trick: the peeling makes the selected submatrix triangular, so the inverse action can also be viewed as forward substitution.

The concrete task deliberately asks for both Pauli sectors and then compresses the four correction matrices into one correction-density number. That keeps the benchmark answer machine-scorable while forcing a solver to get every important piece right: strict neighborhoods, alternating growth, peeling orientation, $\mathbb F_2$ inversion, multiplication order, and the $X/Z$ transpose symmetry.

## Problem

A recent linear-time decoder for self-hypergraph-product quantum expander codes replaces stabilizer-subset searches with a row-and-column localization procedure; evaluate one deterministic instance using that published construction rather than Small Set Flip, Small Set Find, ReShape, minimum-weight decoding, or unrestricted solution of the complete check system. Let $A={0,\ldots,8}$ and $B={0,\ldots,7}$, and reconstruct the binary $B\times A$ seed matrix from row integers [192,3,289,52,384,14,28,72], where bit $j$ of each integer is column $j$ and bit $0$ is index $0$. Reconstruct the $B\times A$ $X$-syndrome from [192,0,0,6,0,2,6,384] and the $A\times B$ $Z$-syndrome from [0,0,36,1,36,36,1,129,0] using the same convention, and use the two non-graph decoder parameters (1/2,2) in their published order.

Recover from the source the decoder definition, its strict generalized-neighborhood convention, the associated growth and peeling construction, and the four published correction formulas, then apply them over $\mathbb F_2$ with the prompt's deterministic lexicographic peeling tie-break whenever more than one legal peeling exists. In <reasoning>, identify the numbered decoder definition, the numbered source definitions governing generalized neighborhoods/growth/peeling, and the four correction-equation numbers; state the B-seeded localization recurrence and the convention for embedding the peeling-selected inverse; also report $\beta$ and $\ell$, the dimensions of $H$, $W$, and $U$, the cardinalities of the two final localized regions in each Pauli sector, the four correction Hamming weights, and the combined number of correction-matrix entries. Do not report full support sets, peeling partner lists, correction matrices, or residual matrices. Compute the total fraction of entries equal to one across the four correction matrices and return that single correction-density scalar.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines.
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

01_matrix_supports.py

Goal
----
Identify the active row and column indices of a binary syndrome-like matrix.

```python
def matrix_supports(M: "np.ndarray") -> tuple[list[int], list[int]]:
    """Return ascending row-support and column-support indices of a binary matrix.

    Parameters
    ----------
    M : np.ndarray
        Nonempty two-dimensional binary matrix.

    Returns
    -------
    result : tuple[list[int], list[int]]
        Row-support indices followed by column-support indices.
    """
    return [], []
```

### Step 2

02_beta_neighborhood.py

Goal
----
Compute one strict fractional neighborhood on either side of the bipartite graph.

```python
def beta_neighborhood(
    H: "np.ndarray",
    vertices: list[int],
    source_side: str,
    beta: float,
) -> list[int]:
    """Return the strict beta-neighborhood on the opposite graph side.

    Parameters
    ----------
    H : np.ndarray
        Binary B-by-A adjacency matrix.
    vertices : list[int]
        Active vertices on source_side.
    source_side : str
        Either "A" or "B".
    beta : float
        Threshold in [0, 1).

    Returns
    -------
    result : list[int]
        Ascending qualifying opposite-side vertex indices.
    """
    return []
```

### Step 3

03_grow_beta_region.py

Goal
----
Run the alternating neighborhood expansion for a fixed number of rounds.

```python
def grow_beta_region(
    H: "np.ndarray",
    seed: list[int],
    seed_side: str,
    beta: float,
    rounds: int,
) -> tuple[list[int], list[int]]:
    """Run the source-defined alternating beta-region growth.

    Returns
    -------
    result : tuple[list[int], list[int]]
        Final ascending A-side region and B-side region.
    """
    return [], []
```

### Step 4

04_deterministic_peeling.py

Goal
----
Construct the benchmark's uniquely specified peeling and partner set.

```python
def deterministic_peeling(
    H: "np.ndarray",
    active: list[int],
    active_side: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute the deterministic lexicographic peeling and partner set."""
    return [], []
```

### Step 5

05_gf2_inverse.py

Goal
----
Invert a square binary matrix exactly over $\mathbb F_2$.

```python
def gf2_inverse(M: "np.ndarray") -> "np.ndarray":
    """Return the inverse of a square binary matrix over GF(2)."""
    return M
```

### Step 6

06_embedded_inverse.py

Goal
----
Invert a selected square submatrix of $H$ and embed the inverse back into the full $A\times B$ index space.

```python
def embedded_inverse(
    H: "np.ndarray",
    a_indices: list[int],
    b_indices: list[int],
) -> "np.ndarray":
    """Embed the GF(2) inverse of H[b_indices, a_indices] into A-by-B coordinates."""
    return H
```

### Step 7

07_decode_x_sector.py

Goal
----
Assemble the previous graph-growth, peeling, and embedded-inverse components into the Pauli-$X$ half of the decoder.

```python
def decode_x_sector(
    H: "np.ndarray",
    W: "np.ndarray",
    beta: float,
    rounds: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Decode the X sector and return the A-by-A and B-by-B corrections."""
    return W, W
```

### Step 8

08_decode_z_sector.py

Goal
----
Implement the transposed Pauli-$Z$ half of the decoder.

```python
def decode_z_sector(
    H: "np.ndarray",
    U: "np.ndarray",
    beta: float,
    rounds: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Decode the Z sector and return the A-by-A and B-by-B corrections."""
    return U, U
```

### Step 9

09_run_quantum_expander_decoder.py

Goal
----
Run the two previously implemented sector decoders and reduce their four corrections to the single benchmark scalar.

```python
def run_quantum_expander_decoder(
    H: "np.ndarray",
    W: "np.ndarray",
    U: "np.ndarray",
    beta: float,
    rounds: int,
) -> float:
    """Run both Pauli-sector decoders and return the total correction density."""
    return 0.0
```
