# Chemistry-Computational_Chemistry-49

## Background

Molecular crystal structure prediction searches the packing landscape of a given molecule for the arrangements it can adopt in the solid state, then ranks the candidates by lattice energy. The discrimination required is severe: competing polymorphs typically lie within a few kJ mol⁻¹ of one another, so a ranking method must combine exhaustive, unbiased sampling of trial packings with an energy model accurate at that scale. A production workflow therefore chains quasi-random trial-structure generation under a fixed space group and Z′, geometric screening that discards chemically broken or physically unrealistic packings while retaining genuinely porous ones, batched structural relaxation under a staged pressure schedule, removal of duplicate packings by a crystal packing-similarity test, and finally a relative-energy ranking of the experimental structure against the global minimum of the surviving unique set. Recent work replaces the classical force field in that chain with machine-learned interatomic potentials to make the relaxation stage affordable at the scale the search demands.

### Required analysis

Using the source paper's ranking protocol and the supplied fixture, determine the relative lattice energy of the experimental packing.

### Final result

The requested answer is the relative lattice energy in kJ/mol for the supplied numerical instance.

## Problem

Report the source paper's relative lattice energy of the experimental packing, in kJ/mol, for the numerical instance printed below.

The paper's production ranking protocol is the method. The tagged scalar is that protocol applied to this fixture, not a published MACE or SevenNet average, not a sentinel, and not E/atom. Recover the paper identities from the source; they are not restated here.

The evaluation instance is this prompt. There is no second workspace file, and nothing else is injected at runtime. The optional `snippet.py` file is empty and contains no task data. Copy the printed arrays into your own code if you need them. Do not run MACE–OFF, SevenNet, DFT, or VASP. Do not train or refit any potential. Do not replace the protocol by a single energy evaluation of the experimental cell or by a density-only generator. Use the supplied deterministic Lennard-Jones energy model in place of a foundation MLIP.

### Printed instance

The search is already constrained to P1 with Z′ = 1. One UFF-lowest conformer has already been selected.

- nuclear charges Z = [7, 7]
- selected conformer (Å), molecular frame:

```
N  (0.00,  0.00, -0.55)
N  (0.00,  0.00,  0.55)
```

- eight Sobol trial vectors s ∈ [0, 1)^6, rows 0…7:

```
0  (0.2424242424242423, 0.2424242424242423, 0.2424242424242423, 0.00, 0.00, 0.30)
1  (0.2484848484848484, 0.2484848484848484, 0.2484848484848484, 0.02, 0.00, 0.30)
2  (0.2606060606060606, 0.2606060606060606, 0.2606060606060606, 0.00, 0.00, 0.30)
3  (0.3181818181818181, 0.3181818181818181, 0.3181818181818181, 0.00, 0.05, 0.30)
4  (0.2575757575757575, 0.2575757575757575, 0.9242424242424241, 0.00, 0.00, 0.30)
5  (0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.00, 0.00, 0.30)
6  (0.2575757575757575, 0.2575757575757575, 0.2575757575757575, 0.00, 0.00, 0.9363636363636363)
7  (0.2303030303030303, 0.2303030303030303, 0.2303030303030303, 0.00, 0.00, 0.30)
```

- experimental reference (Å), present in the ranking pool:

```
cell = (4.10, 4.10, 4.10)
N  (2.05, 2.05, 1.50)
N  (2.05, 2.05, 2.60)
```

Covalent radii (Å): H 0.31, C 0.76, N 0.71, O 0.66. van der Waals radii (Å): H 1.20, C 1.70, N 1.55, O 1.52.

### Named instance reductions

These replace production machinery so the instance is deterministic. They are not the paper's withheld factors, and they are not an ordered recipe.

The paper's iterative shrink of a large coarse cell is the closed-form map

a, b, c = 12 (0.20 + 0.55 s_{0:3})
θ = 2π s_3 ,  φ = π s_4
d = 1.10 (0.70 + 1.00 s_5)

The molecule is a rigid diatomic of length d along (sin φ cos θ, sin φ sin θ, cos φ), centroid at the cell centre. P1 adds no further images of the asymmetric unit.

Neighbor lists include every intermolecular pair between the asymmetric unit and a neighboring image (each axis translation in {−1, 0, 1}, excluding the zero translation) with d < 5.0 Å. Each unordered image pair is counted exactly once: for atoms i < j take all 26 translations, and for i = j take only the 13 translations whose (tx, ty, tz) is lexicographically positive, so that (i, j, t) and its equivalent (j, i, −t) are never both included. Equivalently, the intermolecular energy is one half of the full sum over all 26 non-zero translations and all ordered atom pairs. The intramolecular bond is excluded from that list and contributes ½ k (d − 1.10)² with k = 80 kJ mol⁻¹ Å⁻². Intermolecular pairs use 4ε [(σ/d)¹² − (σ/d)⁶] with ε = 1.8 kJ mol⁻¹ and σ = 3.00 Å. Any pair closer than 0.50 Å is an invalid energy-model configuration.

The paper's batched relaxation is replaced, on this instance, by the two-step damped cell-length map, molecular Cartesian offset held fixed. The incoming neighbor list is used for the first enthalpy gradient; the list is then rebuilt after every cell-length update:

L ← clip( L − 0.12 clip(∇_L H, −1, 1) , 2.2, 12.0 )

∇_L H is a three-component central finite difference of the fixture enthalpy at probe step h = 1e-4 Å: for each cell axis α, (∇_L H)_α = [H(L + h ê_α) − H(L − h ê_α)] / (2h). That probe is an instance setting. It is not an analytic derivative.

Each paper pressure stage applies that damped map twice (four cell-length updates: two under the first-round pressure, then two with the pressure released).

The fixture enthalpy coupling is λ = 1 kJ mol⁻¹ Å⁻³ GPa⁻¹, so H = E + P λ V on this small cell. The stage pressures are a paper identity.

On this small-Z cell, framework contact is sphere overlap of the paper's largest-atom van der Waals framework spheres on the 27 neighboring images. Those contacts form a lattice-offset cycle graph; the paper's framework test uses that graph. The sphere size is a paper identity.

On this small-Z cell the paper's image cluster has tied origin-distances. The instance keys that make that cluster unique are: domain `{−2, −1, 0, 1, 2}³`; distance key = origin-distance rounded to 9 decimals; remaining-tie key = `(na, nb, nc)`; pairing key = shared `(na, nb, nc)`, not an independent radius sort of the centroids. A generated match family is scored by its lowest fixture energy. Every relaxed generated packing, not only the kept unique representatives, is compared against the experimental packing; each generated packing that matches increments the hit count, and the injected experimental packing is itself excluded from that count. A hit does not replace the experimental unique's energy with that family's energy: the experimental unique keeps the relaxed seed's own fixture energy unless the seed itself matches an already-kept unique under the printed pairing keys.

Packing similarity itself is a paper identity, not an instance reduction. Recover that test from the source. Two packings match when the maximum paired-centroid residual is at most 0.2 Å and the paper's remaining similarity criteria hold.

Return a single decimal number in kJ/mol, to at least four decimal places: the paper's relative lattice energy of the experimental packing for this instance. In the reasoning, report the paper hit count for this instance.

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

generate_sobol_trial_crystals

Goal
----
Generate Sobol trial crystals and append the experimental reference.

```python
def generate_sobol_trial_crystals(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Generate the trial-crystal batch.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    conformer_coords : np.ndarray
        Conformer coordinates, shape (n_atoms, 3).
    sobol_vectors : np.ndarray
        Sobol samples in [0, 1), shape (n_trials, 6).
    experimental_cell : np.ndarray
        Experimental cell lengths, shape (3,).
    experimental_coords : np.ndarray
        Experimental coordinates, shape (n_atoms, 3).

    Returns
    -------
    cells : np.ndarray
        Cell lengths, shape (n_trials + 1, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_trials + 1, n_atoms, 3).
    is_experimental : np.ndarray
        0/1 flags, shape (n_trials + 1,).

    Raises
    ------
    ValueError
        If Sobol vectors, conformer coordinates, or the experimental
        packing are invalid.
    """
    return cells, coords, is_experimental
```

### Step 2

apply_triple_radius_validation

Goal
----
Apply triple-radius validation to trial crystals.

```python
def apply_triple_radius_validation(
    atomic_numbers: "np.ndarray",
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "np.ndarray":
    """
    Return a 0/1 keep mask.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).

    Returns
    -------
    np.ndarray
        Integer keep mask of shape (n_crystals,).

    Raises
    ------
    ValueError
        If cells or coordinates are misaligned, or numerical values
        are invalid.
    """
    return keep_mask
```

### Step 3

build_batched_pbc_neighbor_lists

Goal
----
Build batched periodic neighbor lists for the validated crystals.

```python
def build_batched_pbc_neighbor_lists(
    cells: "np.ndarray",
    coords: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Return padded periodic neighbor lists.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).

    Returns
    -------
    neighbor_pairs : np.ndarray
        Integer tuples (i, j, tx, ty, tz), shape (n_crystals, max_pairs, 5).
        Unused slots are -1.
    neighbor_counts : np.ndarray
        Number of valid pairs per crystal, shape (n_crystals,).

    Raises
    ------
    ValueError
        If cells and coordinates are misaligned or non-finite.
    """
    return neighbor_pairs, neighbor_counts
```

### Step 4

two_stage_batched_relax

Goal
----
Relax a validated crystal batch with the two-step damped cell-length map.

```python
def two_stage_batched_relax(
    cells: "np.ndarray",
    coords: "np.ndarray",
    neighbor_pairs: "np.ndarray",
    neighbor_counts: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Relax a validated crystal batch.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_crystals, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_crystals, n_atoms, 3).
    neighbor_pairs : np.ndarray
        Padded neighbor tuples, shape (n_crystals, max_pairs, 5). Used for
        the first enthalpy gradient. Unused slots are -1.
    neighbor_counts : np.ndarray
        Valid pair counts, shape (n_crystals,). The first gradient of each
        crystal uses neighbor_pairs[m, :neighbor_counts[m]].

    Returns
    -------
    cells : np.ndarray
        Relaxed cell lengths.
    coords : np.ndarray
        Relaxed coordinates.
    energies : np.ndarray
        Fixture lattice energies, kJ/mol.
    keep_index : np.ndarray
        Indices retained after relaxation.

    Notes
    -----
    ∇_L H is a central finite difference of the fixture enthalpy at
    h = 1e-4 Å: (H(L + h ê_α) − H(L − h ê_α)) / (2h) on each cell axis.
    After every cell-length update the neighbor list is rebuilt with the
    same 5.0 Å intermolecular rule.

    Raises
    ------
    ValueError
        If the batch, neighbor lists, or fixture energies are invalid.
    """
    return cells, coords, energies, keep_index
```

### Step 5

deduplicate_ccdc_packing

Goal
----
Remove duplicate packings with the paper's similarity rule.

```python
def deduplicate_ccdc_packing(
    cells: "np.ndarray",
    coords: "np.ndarray",
    energies: "np.ndarray",
    is_experimental: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Deduplicate the relaxed packing list.

    Tied-image selection, index pairing, and lower-energy
    experimental inheritance are the instance keys in the module
    background. Packing similarity is from the source;
    the 0.2 Å residual is the stated match cut.

    Parameters
    ----------
    cells : np.ndarray
        Cell lengths, shape (n_relaxed, 3).
    coords : np.ndarray
        Cartesian coordinates, shape (n_relaxed, n_atoms, 3).
    energies : np.ndarray
        Fixture energies, shape (n_relaxed,).
    is_experimental : np.ndarray
        0/1 flags, shape (n_relaxed,).

    Returns
    -------
    unique_index : np.ndarray
        Indices of the kept representatives.
    unique_cells, unique_coords, unique_energies, unique_flags
        Deduplicated blocks.

    Raises
    ------
    ValueError
        If geometries, energies, or experimental flags are misaligned.
    """
    return unique_index, unique_cells, unique_coords, unique_energies, unique_flags
```

### Step 6

relative_lattice_energy

Goal
----
Score the unique list with the paper's relative lattice energy.

```python
def relative_lattice_energy(
    unique_energies: "np.ndarray",
    unique_is_experimental: "np.ndarray",
    unique_cells: "np.ndarray",
    unique_coords: "np.ndarray",
    relaxed_cells: "np.ndarray",
    relaxed_coords: "np.ndarray",
    relaxed_is_experimental: "np.ndarray",
) -> "tuple[float, int]":
    """
    Relative lattice energy of the experimental unique packing.

    Parameters
    ----------
    unique_energies : np.ndarray
        Unique fixture energies, kJ/mol.
    unique_is_experimental : np.ndarray
        Unique 0/1 flags.
    unique_cells, unique_coords
        Unique geometries.
    relaxed_cells, relaxed_coords, relaxed_is_experimental
        Relaxed batch before duplicate removal.

    Returns
    -------
    delta_E : float
        Relative lattice energy, kJ/mol.
    n_hits : int
        Hit count.

    Raises
    ------
    ValueError
        If unique and relaxed blocks are misaligned or empty.
    """
    return delta_E, n_hits
```

### Step 7

run_bomlip_csp

Goal
----
Orchestrate the complete BOMLIP-CSP ranking pipeline.

```python
def run_bomlip_csp(
    atomic_numbers: "np.ndarray",
    conformer_coords: "np.ndarray",
    sobol_vectors: "np.ndarray",
    experimental_cell: "np.ndarray",
    experimental_coords: "np.ndarray",
) -> float:
    """
    Execute the complete ranking pipeline.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    conformer_coords : np.ndarray
        Conformer coordinates, shape (n_atoms, 3).
    sobol_vectors : np.ndarray
        Sobol samples in [0, 1), shape (n_trials, 6).
    experimental_cell : np.ndarray
        Experimental cell lengths, shape (3,).
    experimental_coords : np.ndarray
        Experimental coordinates, shape (n_atoms, 3).

    Returns
    -------
    float
        Relative lattice energy, kJ/mol.

    Raises
    ------
    ValueError
        If the experimental packing is missing from the trial pool, or
        if any pipeline stage is invalid.
    """
    return delta_E
```
