# Part 1 — Scientific Problem

## Background

The source paper develops an approach for predicting energetics associated with adsorbates on metallic surfaces. Adsorption and reaction energies are key descriptors of catalytic activity, but computing each of them with density functional theory is costly, which limits screening of bimetallic catalysts.

## Problem

The source paper presents a computational approach for predicting reaction energies for adsorbates on metal surfaces.

The fixture represents a local adsorption environment containing a bidentate adsorbate bound to a stepped bimetallic surface, together with an additional surface atom. The task is to determine the predicted reaction energy for this local environment.

Recover from the source paper the representation and construction rules used to convert the supplied structural and electronic information into the model input. In particular, determine which atomic, geometric, electronic, and neighborhood information is required, how these quantities are constructed, and how the resulting representation is processed to obtain the predicted reaction energy. Apply the paper-defined procedure to the numerical fixture in Part 2. Do not replace the paper's construction rules with generic atomistic-machine-learning conventions.

The numerical values required for the deterministic evaluation are supplied entirely in Part 2. These values are fixture data and are not expected to be recovered from the source paper.

This task is a paper-inspired synthetic numerical instance. The numerical tensors used in the forward evaluation are synthetic fixture quantities rather than published model parameters.

Do not perform a new electronic-structure calculation, molecular-dynamics calculation, model training, or parameter fitting. Do not substitute a separately generated model or independently chosen numerical parameters for the specified fixture.

# Part 2 — Problem Statement

Apply the paper-defined construction from Part 1 to the following numerical fixture and then evaluate the deterministic synthetic forward pass.

## Local structural and electronic data

A bidentate fragment is adsorbed on a stepped Ni–Ga local environment. The supplied structure contains nine atoms.

```text
atomic_numbers = [28, 28, 31, 28, 8, 6, 8, 1, 28]

bonded_adsorbate = [4, 6]

is_surface = [1, 1, 1, 1, 0, 0, 0, 0, 1]

work_function = 5.14

homo_lumo = 8.37

step = 1.0
```

Coordinates are in Å, with atom indices `t = 0,...,8`:

```text
coordinates =

[[ 0.00,  0.00,  0.00],
 [ 2.50,  0.00,  0.00],
 [ 1.25,  2.16,  0.00],
 [ 1.25,  0.72, -2.05],
 [ 0.15,  0.10,  1.82],
 [ 1.25,  0.10,  2.48],
 [ 2.35,  0.10,  1.82],
 [ 1.25,  0.10,  3.56],
 [ 8.50,  8.50,  8.50]]
```

Clean-surface / gas-phase orbital occupancies are supplied in `(s,p,d,f)` order:

```text
occupancies =

[[0.62, 0.18,  8.41, 0.00],
 [0.58, 0.22,  8.38, 0.00],
 [1.82, 0.95, 10.00, 0.00],
 [0.71, 0.08,  8.55, 0.00],
 [1.84, 4.21,  0.00, 0.00],
 [1.58, 2.36,  0.00, 0.00],
 [1.79, 4.28, 0.00, 0.00],
 [0.98,  0.00,  0.00, 0.00],
 [0.65, 0.15,  8.40, 0.00]]
```

Covalent radii in Å:

```text
H  = 0.31
C  = 0.76
O  = 0.66
Ni = 1.24
Ga = 1.22
```

Pauling electronegativities:

```text
H  = 2.20
C  = 2.55
O  = 3.44
Ni = 1.91
Ga = 1.81
```

## Synthetic numerical contract

The following quantities are fixture inputs for this synthetic instance. They are not fitted values to be recovered from the source paper.

### Node-channel order

The eight node channels used by this fixture, in this order, are

```text
(Z, covalent radius, Pauling electronegativity, n_s, n_p, n_d, site_electronic, Step)
```

`f` occupancy is omitted. `site_electronic` is the work function on surface atoms and the HOMO–LUMO gap on adsorbate atoms: one shared channel, not two channels and not a zero on the other atom type. `Step` is broadcast to every selected node.

### Feature scaling

After assembling those eight channels in that order, scale them using:

```text
feature_scales =
[31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00]
```

Do not refit these values on the supplied structure.

These are fixed fixture scales.

### Frozen random tensors

The synthetic forward pass uses hidden width 8.

The two integer seeds below are fixture inputs. They are not published model parameters and are not recoverable from the source paper. Use them exactly as printed.

```text
graph_map_seed = 20260309
linear_head_seed = 20260309 + 17
```

Do not invent or replace either seed.

Initialize each stream with `numpy.random.default_rng` (NumPy's PCG64 `Generator`). Do not use `numpy.random.RandomState`, `numpy.random.seed`, or a PyTorch generator. The paper's PyTorch implementation does not define these fixture tensors.

Initialize `rng = numpy.random.default_rng(graph_map_seed)`. Using that same stream continuously, for each learned message-passing layer required by the source-paper construction, draw the corresponding fixture tensors with one `rng.normal` call per array. Each layer uses this triple, filled in C-contiguous (row-major) order:

```text
W_self   = rng.normal(0.0, 0.35, size=(8, 8))
W_neigh  = rng.normal(0.0, 0.35, size=(8, 8))
b        = rng.normal(0.0, 0.08, size=(8,))
```

Repeat that triple, with no reseed, once for every message-passing layer required by the source paper, in the paper's layer order. Do not loop over entries inside a tensor.

Initialize `rng_lin = numpy.random.default_rng(linear_head_seed)`. Draw:

```text
W = rng_lin.normal(0.0, 0.35, size=(1, 8))
b = rng_lin.normal(0.0, 0.08, size=(1,))
```

again with one call per tensor and C-contiguous fill.

### Synthetic inference conventions

Once the paper-defined node features, graph, and embedding layers have been constructed, evaluate them with these fixture conventions only:

- Batch normalization and dropout are identity operations. Do not reconstruct training-time batch-normalization statistics or dropout.
- If a node has an empty neighbor set under the paper's graph rule, its neighbor contribution is the zero vector.

Recover from the source paper how the constructed graph representation is propagated and reduced to the final prediction, including the neighborhood definition, aggregation, layer ordering, pointwise transformations, and graph-level reduction, and also whether a node is included in its own neighbor set, which layers receive a pointwise activation, and how node states are reduced to one graph-level vector `z` of length 8.

Apply the drawn linear head to that vector to obtain the predicted reaction energy in eV.

The tagged value is that energy: not a sentinel, not a published error metric, and not a quantity computed from any other fixture.

Do not dump the input coordinate or occupancy matrices, reconstructed weight tensors, or per-layer hidden states.


Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_select_local_adsorption_site

Goal
----
Select the local adsorption site used by the source paper's GNN.

```python
import numpy as np

def select_local_adsorption_site(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
) -> np.ndarray:
    """
    Return the atom indices that form the local adsorption site.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coordinates : np.ndarray
        Cartesian coordinates in Angstrom, shape (n_atoms, 3).
    bonded_adsorbate : np.ndarray
        Indices of the adsorbate atoms that bind the surface.

    Returns
    -------
    np.ndarray
        Sorted unique local-site indices.

    Raises
    ------
    ValueError
        If coordinates are not (n_atoms, 3), bonded indices are
        invalid, distinct atoms coincide, or numerical values are
        invalid.
    """
    return node_index
```

### Step 2

02_construct_node_attributes

Goal
----
Build the per-atom node attributes of the source paper.

```python
import numpy as np

def construct_node_attributes(
    atomic_numbers: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    """
    Construct the scaled node-feature matrix.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    bonded_adsorbate : np.ndarray
        Indices of the surface-bonded adsorbate atoms.
    is_surface : np.ndarray
        0/1 mask, 1 for clean-surface metal atoms, shape (n_atoms,).
    occupancies : np.ndarray
        Per-atom (s, p, d, f) occupancies, shape (n_atoms, 4).
    work_function : float
        Average work function of the clean surface.
    homo_lumo : float
        Gas-phase adsorbate HOMO-LUMO gap.
    step : float
        Facet encoding supplied with the fixture.
    feature_scales : np.ndarray
        Fitted channel scales, shape (8,).
    node_index : np.ndarray
        Local-site atom indices.

    Returns
    -------
    np.ndarray
        Scaled node features with shape (n_selected, 8).

    Raises
    ------
    ValueError
        If occupancies, masks, scales, or node indices are misaligned
        or invalid. A bonded adsorbate atom that is also marked
        surface is an invalid mask.
    """
    return node_features
```

### Step 3

03_build_ase_connectivity

Goal
----
Build the undirected first-neighbor graph of the local adsorption site.

```python
import numpy as np

def build_ase_connectivity(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    """
    Return the bidirectional first-neighbor edge index.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coordinates : np.ndarray
        Cartesian coordinates in Angstrom, shape (n_atoms, 3).
    node_index : np.ndarray
        Local-site atom indices. Edge endpoints are numbered in this
        selected order, starting at 0.

    Returns
    -------
    np.ndarray
        Directed edges with shape (2, 2 n_undirected), stored as
        (src, dst).

    Raises
    ------
    ValueError
        If coordinates are not (n_atoms, 3), node indices are out of
        range, numerical values are invalid, or no pair satisfies
        the cutoff.
    """
    return edge_index
```

### Step 4

04_graphsage_embed

Goal
----
Evaluate the Graph Embedding Layers of the source paper.

```python
import numpy as np

def graphsage_embed(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    sage_weights: dict,
) -> np.ndarray:
    """
    Embed the local-site nodes.

    Parameters
    ----------
    node_features : np.ndarray
        Scaled node attributes with shape (n_nodes, 8).
    edge_index : np.ndarray
        Bidirectional first-neighbor edges with shape (2, n_edges).
    sage_weights : dict
        Frozen tensors keyed W_self_k, W_neigh_k, and b_k for
        layers k = 1, 2, 3. Each W has shape (8, 8); each b has
        shape (8,).

    Returns
    -------
    np.ndarray
        Embedded node states with shape (n_nodes, 8).

    Raises
    ------
    ValueError
        If node features or edges are invalid, or weights are
        incompatible.
    """
    return node_embeddings
```

### Step 5

05_pool_graph_embedding

Goal
----
Reduce the variable-sized node embedding to a graph-level vector.

```python
import numpy as np


def pool_graph_embedding(node_embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce the node states to one graph-level vector.

    Parameters
    ----------
    node_embeddings : np.ndarray
        GraphSAGE node states with shape (n_nodes, 8).

    Returns
    -------
    np.ndarray
        Pooled embedding with shape (8,).

    Raises
    ------
    ValueError
        If node embeddings are empty, are not (n_nodes, 8), or contain
        non-finite values.
    """
    return graph_embedding
```

### Step 6

06_linear_energy_head

Goal
----
Evaluate the paper's linear reaction-energy head.

```python
import numpy as np

def linear_energy_head(
    graph_embedding: np.ndarray,
    linear_weights: dict,
) -> float:
    """
    Evaluate the linear energy head.

    Parameters
    ----------
    graph_embedding : np.ndarray
        Pooled embedding with shape (8,).
    linear_weights : dict
        Frozen tensors W with shape (1, 8) and b with shape (1,).

    Returns
    -------
    float
        Predicted reaction energy in eV.

    Raises
    ------
    ValueError
        If the embedding length or linear weights are incompatible, or
        if values are non-finite.
    """
    return energy
```

### Step 7

07_run_gnn_pipeline

Goal
----
Evaluate the predicted reaction energy on the
supplied local adsorption environment.

```python
import numpy as np

def run_gnn_pipeline(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    sage_weights: dict,
    linear_weights: dict,
) -> float:
    """
    Execute the complete localized-site GNN pipeline.

    Returns
    -------
    float
        Predicted reaction energy in eV.

    Raises
    ------
    ValueError
        If any input array is invalid or a pipeline stage returns a
        non-finite energy.
    """
    return energy
```
