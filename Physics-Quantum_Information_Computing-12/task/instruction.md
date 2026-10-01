# Physics-Quantum_Information_Computing-12

## Background

A binary one-dimensional MERA is a layered tensor network. Read from the physical chain upwards it is a real-space renormalisation: each layer first removes short-range entanglement with two-site unitaries (disentanglers) and then coarse-grains pairs of sites into one with isometries W, which satisfy W^dagger W = 1. Read from the top down it is a quantum circuit that prepares the state, injecting a fresh qubit in |0> at every isometry. With bond dimension 2 every renormalised site is one qubit. Because every tensor is unitary or isometric, a local reduced density matrix depends only on the tensors inside its past causal cone; in a binary MERA that cone stays at most three sites wide at every level, which is why local expectation values of an infinite MERA can be evaluated with a cost proportional to the number of layers (Vidal 2007; Evenbly and Vidal 2009).

For a bipartition of a pure state into A and B, rho_B = Tr_A |Psi><Psi|. Its eigenvalues lambda_i are the squared Schmidt coefficients; the Renyi entropies are S_alpha = log2(Tr rho_B^alpha)/(1 - alpha), the entanglement spectrum is zeta_i = -log2 lambda_i, and the Schmidt gap is lambda_0 - lambda_1. In a gapped one-dimensional ground state the half-chain entropy saturates with the size of the halves (area law); at a critical point it grows logarithmically with a prefactor set by the central charge (Calabrese and Cardy 2009). The transverse-field Ising chain H = -sum X_i X_{i+1} - g sum Z_i is critical at g = 1, ordered for g < 1 and paramagnetic for g > 1, and its ground state is exactly solvable by free fermions (Pfeuty 1970). Pauli matrices: X = [[0,1],[1,0]], Y = [[0,-i],[i,0]], Z = [[1,0],[0,-1]].

## Problem

Circuits built on the multiscale entanglement renormalization ansatz can hold an infinite spin chain, critical or gapped, in a handful of qubits, and small instances of them now run on trapped-ion processors. What makes them attractive is the hierarchy: short-range entanglement is added layer by layer, and the entanglement between two semi-infinite halves of the chain then costs an effort that grows with the number of layers rather than with the size of either half. The task is to exploit that for one concrete circuit, a four-layer, bond-dimension-2 circuit variationally tuned to the ground state of the transverse-field Ising chain H = -sum_i X_i X_{i+1} - g sum_i Z_i at g = 1.25, on the paramagnetic side of the transition. The input is the table of gate angles below; the output is the Schmidt gap of the half-chain bipartition of the infinite chain this circuit prepares.

The state is prepared from the top down. Above the top layer (layer 4) every renormalised site is in |0>. Going down, each layer first applies one isometry per site and then a row of disentanglers. Isometry j takes site j of the level above onto the left input qubit of a two-qubit block, with a freshly prepared |0> on the right input qubit, and places its outputs on sites (2j-1, 2j) of the level below, left output on 2j-1. The disentanglers of the same layer then act on every pair (2j, 2j+1), with the left input and output on site 2j. All isometries within a layer are the same block, all disentanglers within a layer are the same block, the pattern repeats over every integer j, and layer 1 outputs the physical spins, one qubit per site. Every block has the form G(x1, x2) = YX(x1) XY(x2), where XY(x) = exp(-i pi x X(x)Y / 2) and YX(x) = exp(-i pi x Y(x)X / 2), the first tensor factor acts on the left qubit, and angles are in units of pi.

Split the physical chain into A = (-inf, 1] and B = [2, inf) and let rho_B be the reduced density matrix of B in the state the circuit prepares. With its eigenvalues ordered lambda_0 >= lambda_1 >= ..., the Schmidt gap is lambda_0 - lambda_1. The quantity is a property of this bond-dimension-2 circuit state, not of the exact Ising ground state, and the two differ in the digits you are asked to report. The chain is infinite in both directions and the circuit has exactly the four layers listed.

Gate angles [x1, x2] for each layer, in units of pi:

- Layer 1 (physical): disentangler [0.16148386263647466, -0.056145691508003415]; isometry [-0.02633623257988793, 0.13609039706878254]
- Layer 2: disentangler [0.13118304250021381, -0.0383352598604525]; isometry [-0.019513199676447934, 0.1145574729714855]
- Layer 3: disentangler [0.08035122878854976, -0.021415347769675463]; isometry [-0.0096600845365713, 0.06892662021737829]
- Layer 4 (top): disentangler [0.010069626069892758, 0.01006953360769033]; isometry [-0.061861621487427465, 0.08198065488845328]

## What to report

- The second Renyi entropy of B, S2 = -log2 Tr(rho_B^2), in bits.
- The three largest eigenvalues lambda_0, lambda_1 and lambda_2 of rho_B.
- The entanglement energies zeta_0 = -log2 lambda_0 and zeta_1 = -log2 lambda_1.
- The Schmidt gap lambda_0 - lambda_1 to four decimal places; this is the number that goes in the final-answer tag.
- Name the sources you relied on and say what each one supplied.

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

block_gate

Goal
----
Build the real 4x4 matrix of one symmetry-respecting two-qubit MERA block from its rotation
angles. Returns the matrix as a float array.

```python
def block_gate(angles: list) -> "np.ndarray":
    """Build the real 4x4 matrix of one symmetry-respecting two-qubit MERA block from its
    rotation angles. Returns the matrix as a float array.

    Args:
        angles: list of 2 floats [x1, x2] or 4 floats [y1, y2, x1, x2], units of pi.

    Returns:
        np.ndarray: real 4x4 block matrix, rows = output basis index, columns = input.
    """
    return None
```

### Step 2

layer_tensors

Goal
----
Turn a disentangler block matrix and an isometry block matrix into the MERA tensors U (four
legs) and W (three legs). Returns the tuple (U, W).

```python
def layer_tensors(gate_u: "np.ndarray", gate_w: "np.ndarray") -> tuple:
    """Turn a disentangler block matrix and an isometry block matrix into the MERA tensors U
    (four legs) and W (three legs). Returns the tuple (U, W).

    Args:
        gate_u: 4x4 real matrix of the disentangler block.
        gate_w: 4x4 real matrix of the isometry block.

    Returns:
        tuple: (U, W) as float arrays of shapes (2, 2, 2, 2) and (2, 2, 2).
    """
    return None
```

### Step 3

doubled_layer_map

Goal
----
Apply one layer of the doubled, SWAP-contracted transition map that carries the purity of
the half chain B = [2, inf) down the causal cone of the boundary. Returns the two-copy
boundary operator one level lower.

```python
def doubled_layer_map(Q: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    """Apply one layer of the doubled, SWAP-contracted transition map that carries the purity
    of the half chain B = [2, inf) down the causal cone of the boundary. Returns the two-
    copy boundary operator one level lower.

    Args:
        Q: float array of shape (2,)*8, the two-copy boundary operator of the upper level.
        U: disentangler tensor U[a, b, c, d].
        W: isometry tensor W[a, b, c].

    Returns:
        np.ndarray: float array of shape (2,)*8, legs [k1_0, k1_1, b1_0, b1_1, k2_0, k2_1, b2_0, b2_1].
    """
    return None
```

### Step 4

renyi2_from_doubled

Goal
----
Close the doubled boundary operator at the physical level and convert the resulting purity
into the second Renyi entanglement entropy in bits. Returns S2.

```python
def renyi2_from_doubled(Q: "np.ndarray") -> float:
    """Close the doubled boundary operator at the physical level and convert the resulting
    purity into the second Renyi entanglement entropy in bits. Returns S2.

    Args:
        Q: float array of shape (2,)*8 at the physical level.

    Returns:
        float: second Renyi entropy S2 = -log2(purity), in bits.
    """
    return None
```

### Step 5

edge_layer_map

Goal
----
Apply one layer of the causal-cone circuit of the subsystem A = (-inf, 1] while keeping the
B-side qubit that leaves the cone in an edge register. Returns the joint state of the
boundary pair and the enlarged register one level lower.

```python
def edge_layer_map(R: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    """Apply one layer of the causal-cone circuit of the subsystem A = (-inf, 1] while keeping
    the B-side qubit that leaves the cone in an edge register. Returns the joint state of
    the boundary pair and the enlarged register one level lower.

    Args:
        R: float array of shape (2, 2, d, 2, 2, d).
        U: disentangler tensor U[a, b, c, d].
        W: isometry tensor W[a, b, c].

    Returns:
        np.ndarray: float array of shape (2, 2, 2d, 2, 2, 2d), legs [k0, k1, e, b0, b1, f].
    """
    return None
```

### Step 6

edge_reduced_state

Goal
----
At the physical level, discard the boundary pair to obtain the density matrix of the edge
register, whose spectrum is the entanglement spectrum of the half chain. Returns the
register density matrix.

```python
def edge_reduced_state(R: "np.ndarray") -> "np.ndarray":
    """At the physical level, discard the boundary pair to obtain the density matrix of the
    edge register, whose spectrum is the entanglement spectrum of the half chain. Returns
    the register density matrix.

    Args:
        R: float array of shape (2, 2, d, 2, 2, d) at the physical level.

    Returns:
        np.ndarray: d x d float array, the edge-register density matrix.
    """
    return None
```

### Step 7

entanglement_spectrum

Goal
----
Diagonalise a real symmetric density matrix and return its eigenvalues in descending order,
the entanglement spectrum lambda_0 >= lambda_1 >= ... .

```python
def entanglement_spectrum(rho: "np.ndarray") -> "np.ndarray":
    """Diagonalise a real symmetric density matrix and return its eigenvalues in descending
    order, the entanglement spectrum lambda_0 >= lambda_1 >= ... .

    Args:
        rho: real symmetric d x d matrix.

    Returns:
        np.ndarray: float array of the eigenvalues in descending order.
    """
    return None
```

### Step 8

schmidt_gap

Goal
----
Schmidt gap lambda_0 - lambda_1 of the half chain B = [2, inf) for the infinite binary MERA
built from the given per-layer angles. Returns the gap.

```python
def schmidt_gap(u_angles: list, w_angles: list) -> float:
    """Schmidt gap lambda_0 - lambda_1 of the half chain B = [2, inf) for the infinite binary
    MERA built from the given per-layer angles. Returns the gap.

    Args:
        u_angles: list of per-layer disentangler angle lists, bottom layer first.
        w_angles: list of per-layer isometry angle lists, bottom layer first.
        Uses: block_gate, layer_tensors, doubled_layer_map, renyi2_from_doubled,
        edge_layer_map, edge_reduced_state, entanglement_spectrum

    Returns:
        float: Schmidt gap lambda_0 - lambda_1.
    """
    return None
```
