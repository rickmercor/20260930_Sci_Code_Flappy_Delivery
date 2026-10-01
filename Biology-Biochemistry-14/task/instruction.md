# Biology-Biochemistry-14

## Background

Cells coordinate development and tissue function by exchanging ligands that bind receptors on other cells, either through membrane-bound partners that need direct contact or through secreted factors that spread over short distances. Spatial transcriptomic assays record gene expression together with the position of every cell, and splicing-aware measurements add an estimate of how fast each transcript is changing, so ligand–receptor communication can be mapped and followed in time at the resolution of single cells.

## Problem

I profiled a small patch of mouse fetal liver at cellular resolution and fitted a transcription–splicing–degradation velocity model, and I want to know whether albumin–FcRn signalling around one cell is currently strengthening or weakening. Albumin (Alb) is the secreted ligand and FcRn is its receptor, a heterodimer of the B2m and Fcgrt subunits. The table gives each cell's centroid in µm and the model's predicted unspliced (u) and spliced (s) abundances; the fitted gene-level rates per unit latent time (transcription α, splicing β, degradation γ) are α = 1.0, β = 0.9, γ = 0.30 for Alb, α = 1.0, β = 0.8, γ = 0.25 for B2m, and α = 0.9, β = 0.6, γ = 0.30 for Fcgrt.

| cell | x | y | Alb u | Alb s | B2m u | B2m s | Fcgrt u | Fcgrt s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1 | 1.5 | 0.20 | 5.57 | 1.02 | 4.33 | 0.00 | 0.27 |
| 2 | 13.1 | 3.9 | 0.35 | 7.05 | 1.27 | 3.94 | 0.03 | 0.23 |
| 3 | 19.9 | -2.8 | 0.51 | 7.10 | 1.29 | 3.85 | 0.11 | 0.37 |
| 4 | 34.9 | 0.1 | 0.37 | 6.03 | 0.92 | 3.68 | 0.06 | 0.17 |
| 5 | 6.3 | 11.1 | 0.23 | 5.67 | 1.35 | 3.66 | 0.01 | 0.32 |
| 6 | 17.7 | 8.2 | 0.53 | 7.21 | 1.36 | 4.30 | 0.02 | 0.16 |
| 7 | 27.4 | 8.0 | 0.38 | 6.71 | 1.24 | 3.98 | 0.10 | 0.41 |
| 8 | 38.8 | 11.5 | 0.25 | 5.82 | 1.04 | 3.69 | 0.19 | 0.46 |
| 9 | 0.3 | 19.8 | 0.43 | 5.98 | 1.25 | 3.65 | 0.07 | 0.27 |
| 10 | 9.9 | 21.3 | 0.44 | 6.31 | 1.14 | 3.84 | 0.05 | 0.13 |
| 11 | 19.9 | 24.1 | 0.55 | 6.48 | 1.02 | 4.06 | 0.20 | 0.25 |
| 12 | 30.7 | 21.4 | 0.46 | 7.49 | 1.16 | 4.20 | 0.00 | 0.17 |
| 13 | 7.2 | 30.0 | 0.49 | 6.63 | 1.00 | 4.05 | 0.19 | 0.44 |
| 14 | 14.0 | 33.1 | 0.56 | 5.98 | 1.35 | 4.04 | 0.08 | 0.29 |
| 15 | 27.5 | 29.1 | 0.22 | 7.25 | 1.35 | 3.96 | 0.10 | 0.47 |
| 16 | 39.8 | 31.6 | 0.40 | 6.45 | 1.21 | 4.29 | 0.06 | 0.37 |
| 17 | 53.5 | 5.1 | 0.73 | 4.37 | 1.33 | 3.85 | 0.63 | 1.18 |
| 18 | 55.4 | 26.1 | 0.45 | 4.20 | 1.26 | 4.13 | 0.41 | 0.97 |
| 19 | 60.5 | 13.0 | 0.67 | 4.28 | 1.19 | 3.66 | 0.69 | 0.56 |
| 20 | 85.8 | -8.6 | 0.17 | 0.43 | 1.30 | 3.74 | 1.20 | 1.04 |
| 21 | 111.6 | -7.0 | 0.23 | 0.27 | 1.11 | 4.30 | 0.81 | 1.51 |
| 22 | 136.6 | -4.9 | 0.09 | 0.48 | 1.15 | 4.34 | 0.90 | 2.59 |
| 23 | 102.8 | 11.1 | 0.13 | 0.20 | 1.21 | 4.15 | 0.65 | 2.48 |
| 24 | 128.5 | 10.5 | 0.24 | 0.54 | 1.12 | 4.29 | 1.19 | 1.86 |
| 25 | 151.2 | 12.2 | 0.29 | 0.11 | 0.94 | 4.19 | 1.28 | 1.90 |
| 26 | 84.0 | 36.7 | 0.19 | 0.43 | 1.36 | 4.18 | 1.10 | 1.43 |
| 27 | 113.0 | 28.7 | 0.06 | 0.26 | 1.03 | 4.13 | 1.07 | 1.70 |
| 28 | 138.3 | 32.4 | 0.10 | 0.28 | 1.15 | 3.86 | 0.96 | 2.32 |

Score this pair with the published cellular-resolution ligand–receptor framework for spatial transcriptomes, as specified in its journal article, treating albumin as a diffusible ligand with a 200 µm neighbourhood radius and a Gaussian kernel whose width is the largest that leaves at most a fraction 1e-9 of a point release beyond that radius, reading FcRn at the receiving cell itself, and using that framework's extension that turns RNA velocity into the rate of change of the signalling score. What is the signalling velocity of the neighbourhood of cell 17, to ten significant figures?

In <reasoning>, state the scoring conventions of that framework that fix the number; the kernel width; the signalling velocity of cell 17 alone; the ligand-driven and receptor-driven parts of the neighbourhood value; whether signalling around cell 17 is strengthening or weakening; and which partner's change drives that trend. Those are the derived scalars and conclusions that determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied table.

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

01_build_delaunay_neighbourhoods

Goal
----
Build the neighbourhood of every cell from the Delaunay triangulation of the cell centroids.

```python
def build_delaunay_neighbourhoods(positions: "np.ndarray") -> "np.ndarray":
    """Return the Delaunay neighbourhood indicator matrix of a set of cell centroids.

    Entry ``[i, k]`` is 1.0 when cell ``k`` belongs to the neighbourhood of
    cell ``i`` and 0.0 otherwise. The neighbourhood of a cell is the cell
    itself together with every cell joined to it by an edge of the Delaunay
    triangulation of all centroids. Centroids are assumed to be in general
    position (no four on the boundary of a common empty circle).

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` with ``n >= 3`` centroid coordinates.

    Returns
    -------
    np.ndarray
        Symmetric float array of shape ``(n, n)`` with ones on the diagonal.

    Raises
    ------
    ValueError
        If ``positions`` is not a finite numeric array of shape ``(n, 2)``
        with ``n >= 3``, if two centroids coincide, or if all centroids lie
        on one straight line.
    """
    return result
```

### Step 2

02_derive_diffusion_bandwidth

Goal
----
Fix the width of the Gaussian diffusion kernel from the neighbourhood radius and the fraction of released ligand allowed to escape it.

```python
def derive_diffusion_bandwidth(radius: float, tail_mass: float) -> float:
    """Return the largest Gaussian bandwidth that keeps a release inside a radius.

    A unit amount of ligand released at a point of the tissue plane is spread
    as an isotropic two-dimensional Gaussian with standard deviation
    ``sigma`` along each axis. Return the largest ``sigma`` for which the
    fraction of that amount lying at distance ``radius`` or more from the
    release point does not exceed ``tail_mass``.

    Parameters
    ----------
    radius : float
        Neighbourhood radius, in the same length unit as ``sigma``.
    tail_mass : float
        Largest admissible fraction of the release beyond ``radius``.

    Returns
    -------
    float
        The bandwidth ``sigma``.

    Raises
    ------
    ValueError
        If ``radius`` is not a finite positive number, or if ``tail_mass``
        is not a finite number strictly between 0 and 1 (booleans are
        rejected for both).
    """
    return result
```

### Step 3

03_compute_ligand_sharing_weights

Goal
----
Compute, for every sending cell, the fraction of its secreted ligand that each cell of the tissue receives.

```python
def compute_ligand_sharing_weights(
    positions: "np.ndarray",
    radius: float,
    tail_mass: float,
) -> "np.ndarray":
    """Return the fractions of each sender's diffusible ligand received by every cell.

    Build the framework's finite-radius Gaussian transport operator from the
    supplied cell centroids and diffusion parameters. Entry ``[k, i]`` is the
    fraction of material originating at cell ``k`` that is assigned to cell
    ``i``; the finite support includes receivers exactly ``radius`` away.
    Apply the source method's conventions for kernel bandwidth,
    neighbourhood membership, self contribution and conservation of each
    sender's released amount.

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` with ``n >= 1`` centroid coordinates.
    radius : float
        Neighbourhood radius, in the unit of ``positions``.
    tail_mass : float
        Fraction of a release allowed beyond ``radius``; fixes the bandwidth.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, n)``; rows identify the material's cell of
        origin and columns identify the receiving cell.

    Raises
    ------
    ValueError
        If ``positions`` is not a finite numeric array of shape ``(n, 2)``
        with ``n >= 1``, if ``radius`` is not a finite number of at least
        the self-distance ``1e-9``, or if ``tail_mass`` is not a finite
        number strictly between 0 and 1 (booleans are rejected for both).
    """
    return result
```

### Step 4

04_compute_total_expression_rates

Goal
----
Convert a gene's unspliced and spliced abundances and its fitted kinetic rates into the abundance used for signalling and its rate of change in every cell.

```python
def compute_total_expression_rates(
    unspliced: "np.ndarray",
    spliced: "np.ndarray",
    transcription_rate: float,
    splicing_rate: float,
    degradation_rate: float,
) -> "np.ndarray":
    """Return each cell's total mRNA abundance of a gene and its time derivative.

    Apply the transcription-splicing-degradation balance to the supplied RNA
    species. For every cell, return the expression state used by the
    signalling framework and the instantaneous derivative of that same state.
    Derive both columns from the coupled kinetic model and the input
    abundances; do not substitute a single RNA species for the framework's
    expression state.

    Parameters
    ----------
    unspliced : np.ndarray
        Nonnegative float array of shape ``(n,)``, ``n >= 1``.
    spliced : np.ndarray
        Nonnegative float array of shape ``(n,)``.
    transcription_rate : float
        Nonnegative transcription rate.
    splicing_rate : float
        Positive splicing rate constant.
    degradation_rate : float
        Positive degradation rate constant.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, 2)``: signalling abundance, then its rate
        of change under the fitted kinetics.

    Raises
    ------
    ValueError
        If ``unspliced`` or ``spliced`` is not a finite nonnegative numeric
        array of shape ``(n,)`` with ``n >= 1``, if their lengths differ, if
        ``transcription_rate`` is not a finite nonnegative number, or if
        ``splicing_rate`` or ``degradation_rate`` is not a finite positive
        number (booleans are rejected for the rates).
    """
    return result
```

### Step 5

05_compute_cell_signalling_velocity_terms

Goal
----
Split the rate of change of each cell's ligand-receptor signalling score into the part driven by the changing ligand supply and the part driven by the changing receptor.

```python
def compute_cell_signalling_velocity_terms(
    sharing_weights: "np.ndarray",
    ligand_rates: "np.ndarray",
    receptor_subunit_rates: "np.ndarray",
) -> "np.ndarray":
    """Return the ligand-driven and receptor-driven parts of each cell's signalling velocity.

    ``ligand_rates[k]`` holds sender ``k``'s ligand state and its rate, while
    ``sharing_weights[k, i]`` maps material originating at sender ``k`` to
    receiver ``i``. ``receptor_subunit_rates[j, i]`` holds the corresponding
    state and rate for receptor subunit ``j`` at receiver ``i``. Construct the
    paper-defined cell signalling score and differentiate it into the
    contribution attributable to ligand dynamics and the contribution
    attributable to receptor dynamics. Apply the framework's multi-subunit
    receptor convention at the receiving cell.

    Parameters
    ----------
    sharing_weights : np.ndarray
        Finite nonnegative float array of shape ``(n, n)``, senders by receivers.
    ligand_rates : np.ndarray
        Finite float array of shape ``(n, 2)`` with nonnegative column 0.
    receptor_subunit_rates : np.ndarray
        Finite float array of shape ``(m, n, 2)``, ``m >= 1``, with
        nonnegative abundances in the last axis' entry 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, 2)``: ligand-driven contribution, then
        receptor-driven contribution.

    Raises
    ------
    ValueError
        If any input is not a finite numeric array of the stated shape, if
        ``sharing_weights`` has a negative entry, or if a ligand or receptor
        subunit abundance is negative.
    """
    return result
```

### Step 6

06_average_over_neighbourhoods

Goal
----
Summarise per-cell values at the level of spatial neighbourhoods by averaging each cell's value with those of its neighbours.

```python
def average_over_neighbourhoods(values: "np.ndarray", neighbourhoods: "np.ndarray") -> "np.ndarray":
    """Return the mean of per-cell values over every cell's neighbourhood.

    ``neighbourhoods[i, k]`` is 1.0 when cell ``k`` belongs to the
    neighbourhood of cell ``i`` and 0.0 otherwise; every cell belongs to its
    own neighbourhood and membership is symmetric. Entry ``i`` of the result
    (row ``i`` when ``values`` is two-dimensional) is the arithmetic mean of
    ``values`` over the members of cell ``i``'s neighbourhood.

    Parameters
    ----------
    values : np.ndarray
        Finite float array of shape ``(n,)`` or ``(n, d)``.
    neighbourhoods : np.ndarray
        Float array of shape ``(n, n)`` holding only 0.0 and 1.0, symmetric,
        with ones on the diagonal.

    Returns
    -------
    np.ndarray
        Float array with the shape of ``values``.

    Raises
    ------
    ValueError
        If ``neighbourhoods`` is not a numeric ``(n, n)`` array of zeros and
        ones that is symmetric with ones on the diagonal, or if ``values`` is
        not a finite numeric array of shape ``(n,)`` or ``(n, d)``.
    """
    return result
```

### Step 7

07_estimate_neighbourhood_signalling_velocity

Goal
----
Compose every earlier step to obtain the rate of change of diffusible ligand-receptor signalling for the neighbourhood of one cell.

```python
def estimate_neighbourhood_signalling_velocity(
    positions: "np.ndarray",
    ligand_counts: "np.ndarray",
    ligand_kinetics: tuple,
    receptor_subunit_counts: "np.ndarray",
    receptor_subunit_kinetics: "np.ndarray",
    cell_index: int,
    radius: float = 200.0,
    tail_mass: float = 1e-9,
) -> float:
    """Return the signalling velocity of one cell's neighbourhood for a diffusible ligand.

    ``ligand_counts[k]`` and ``receptor_subunit_counts[j, k]`` contain the two
    RNA species for the ligand and receptor subunits, with their fitted
    kinetics supplied in the corresponding rate arguments. Compose the six
    preceding public functions as the building blocks of the article's
    end-to-end calculation. Preserve their array orientations and scientific
    conventions, and return the resulting neighbourhood signalling velocity
    for ``cell_index`` without duplicating their internal algorithms here.

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` of cell centroids.
    ligand_counts : np.ndarray
        Float array of shape ``(n, 2)``: unspliced, spliced.
    ligand_kinetics : tuple
        ``(transcription_rate, splicing_rate, degradation_rate)`` of the ligand.
    receptor_subunit_counts : np.ndarray
        Float array of shape ``(m, n, 2)``, ``m >= 1``: unspliced, spliced.
    receptor_subunit_kinetics : np.ndarray
        Float array of shape ``(m, 3)`` of subunit rates, in the ligand's order.
    cell_index : int
        Zero-based index of the cell whose neighbourhood is reported.
    radius : float
        Neighbourhood radius of the diffusion kernel.
    tail_mass : float
        Fraction of a release allowed beyond ``radius``.

    Returns
    -------
    float
        The neighbourhood signalling velocity of cell ``cell_index``.

    Raises
    ------
    ValueError
        If an array does not have the stated shape (the kinetics included),
        if a kinetic rate is a boolean, if ``cell_index`` is not an integer
        in ``[0, n)`` (booleans are rejected), or if any composed step
        rejects its input.
    """
    return result
```
