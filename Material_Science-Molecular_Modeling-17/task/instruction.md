# Material_Science-Molecular_Modeling-17

## Background

Liquid-state theory relates the pair structure of a fluid to its thermodynamics through the radial distribution function, the static structure factor and the direct correlation function. When the pair potential is unknown, as for many complex and colloidal fluids, a route is needed that runs from measured structure to free energy without it, by mapping the repulsive core onto a hard-sphere reference and treating the remainder perturbatively.

## Problem

Liquid-state theory connects the microscopic structure of a fluid to its thermodynamics, but the usual routes require the pair potential, which for many complex fluids is unknown or only approximately known. A recent source develops a route that runs the other way: starting from the radial distribution function measured at a single state point, and without using the potential that produced it, it constructs the bulk excess free energy by mapping the repulsive core onto an effective hard-sphere reference and treating the remainder of the interaction perturbatively. Recover that construction from the source. The load-bearing choices are the source's and are not derivable from the statement below: how the effective hard-sphere diameter is assigned to the structure, which correlation function weights the tail inside the integrated interaction strength, which closure is used at each stage of the reconstruction, and how the reference free energy is evaluated. This problem is calibrated so that each of those choices changes the reported number by far more than the grading tolerance.

Implement eight functions on a fixed radial grid $r_i = 0.02\,i$ for $i = 1, \dots, 1024$, in units where the length scale $\sigma$ and the thermal energy $k_B T$ are one. Every three-dimensional Fourier transform on this grid is the type-1 discrete sine transform of $r f(r)$ with wavenumbers $k_j = \pi j / (1025 \times 0.02)$, $j = 1, \dots, 1024$; every Picard solve mixes the indirect correlation function $h - c$ from a zero start; every integral is a trapezoid sum on the grid.

`stf_target_rdf(rho, beta_eps, rc, n_iter)` generates the structural input: the radial distribution function of a cut-and-shifted 12-6 Mie fluid at number density `rho`, well depth `beta_eps` and cut radius `rc`, by a Picard solution of the Ornstein-Zernike relation with the closure the source prescribes for soft interactions, run for exactly `n_iter` iterations with mixing 0.15 and no tolerance exit, negative values clipped to zero. From this point on the generating potential is treated as unknown and must not be used by any later function. `stf_ibi_potential(g_target, rho, gamma, n_ibi)` reconstructs the reduced effective pair potential from that structure by iterative Boltzmann inversion, started from the low-density inversion and updated exactly `n_ibi` times with mixing `gamma`, using the closure the source prescribes for interactions that carry a hard core; the inner integral-equation solve is warm-started between inversion steps and takes exactly 80 iterations with mixing 0.15 per step, and the potential is held at 60 inside the core where the structure is below 1e-6. `stf_wca_split(bphi, g_target)` splits the reconstructed potential into a reference part and a tail part at the position of its minimum outside the core, in the decomposition the source adopts, returning the two rows so that they sum to the input. `stf_reference_rdf(bphi_ref, rho, n_iter)` solves for the reference system's radial distribution function with the hard-core closure, by exactly `n_iter` Picard iterations with mixing 0.15. `stf_hs_diameter(g_ref, g_target, bphi_ref, rho)` assigns the effective hard-sphere diameter by the source's procedure, scanning diameters from 0.86 to 1.04 on 19 equally spaced points, refining the best bracket by exactly 12 golden-section steps and returning the bracket midpoint, with the hard-sphere structures for comparison taken from the closed-form solution of the hard-core closure for hard spheres in $k$ space, carried through the Ornstein-Zernike relation and transformed onto the grid, and with peaks of the measured function taken as strict local maxima that exceed 1.02; the measured function is noise-free and is used as it stands at every distance, nothing in it is reset to one, and the cutoff serves only as the upper limit of the structure comparison; it also returns the cutoff distance the procedure uses and, for comparison, the diameter the conventional prescription would give from the reference potential, as an integral from zero with the integrand taken as one at the origin. `stf_integrated_strength(g_target, bphi_ref, bphi_tail)` returns the reduced integrated strength of the tail by the thermodynamic-integration route, with the approximation the source adopts for the correlation weight, and by the second-virial route the source gives as its dilute-limit alternative. `stf_free_energy(rho, sigma, a_strength)` returns the reduced excess free energy per particle from the hard-sphere reference at the assigned diameter, using the equation of state the source recommends, plus the tail term. `stf_audit(rho, beta_eps, rc, gamma, n_ibi)`, the orchestrator, must call the earlier functions rather than reimplementing them, generating the structure with 400 iterations and solving every reference system with 300, and returns nine values: the split radius, the comparison cutoff, the source's diameter, the conventional diameter, the two integrated strengths, the hard-sphere reference free energy per particle at the source's diameter, and the excess free energy per particle by each of the two strength routes.

All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on arrays that are not of length 1024 or not finite, on non-finite or non-positive densities, well depths, cut radii, mixing parameters or diameters, on non-integral or non-positive iteration counts, and on a packing fraction at or above one.

Evaluate the audit at `rho = 0.4`, `beta_eps = 0.5`, `rc = 3.5`, `gamma = 0.2` and `n_ibi = 800`.

As the final answer, report the reduced excess free energy per particle obtained with the thermodynamic-integration strength, to six significant figures.

In your reasoning report the conventions you used, and justify each from the source: how the source assigns the effective hard-sphere diameter to a measured structure, and how that differs from the conventional prescription; which correlation function the source uses to weight the tail inside the integrated strength, and how that differs from the usual first-order perturbation treatment; how the source's cutoff for comparing structures is set from the peaks of the measured function; what the two rows of the split take inside the split radius; which closure the source uses for hard-core interactions and which for soft ones; the equation of state the source recommends for the reference; the relation the source gives between the second-virial route and the difference of virial coefficients; and the two-parameter form the source fits to the density dependence of the integrated strength.

Report numerically, as evidence that the chain was executed: the split radius; the comparison cutoff; the source's diameter and the conventional diameter; the integrated strength by each of the two routes; the hard-sphere reference free energy per particle at the source's diameter; and the excess free energy per particle by each of the two routes. These are the scalars that determine the final number.

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

stf_target_rdf

Goal
----
Generates the structural input, the radial distribution function of a cut-and-shifted 12-6 fluid, by a fixed-count integral-equation solve on the fixed radial grid.

```python
def stf_target_rdf(rho: float, beta_eps: float, rc: float, n_iter: int) -> "np.ndarray":
    r"""rho: positive float, number density in units of sigma^-3.
    beta_eps: positive float, well depth of the 12-6 Mie potential in units of k_B T.
    rc: positive float, cut-and-shift radius in units of sigma.
    n_iter: positive integer, number of fixed-point iterations of the integral-equation solve.

    Returns a numpy float64 array of shape $(1024,)$: the radial distribution function of the
    cut-and-shifted 12-6 fluid on the fixed radial grid $r_i = 0.02\,i$, $i = 1..1024$, obtained by
    solving the Ornstein-Zernike relation with the closure the source prescribes for soft
    interactions, by a Picard iteration of exactly n_iter steps with mixing 0.15 and no tolerance
    exit. The iteration mixes the indirect correlation function $h - c$ from a zero start, and
    every three-dimensional Fourier transform on the grid is the type-1 discrete sine transform of
    $r f(r)$ with wavenumbers $k_j = \pi j / (1025 \times 0.02)$, $j = 1..1024$. Negative values
    are clipped to zero. This array is the structural input to every later step; the potential
    that generated it must not be used downstream.

    Raises:
        ValueError: on non-finite or non-positive rho, beta_eps or rc, or non-integral or
            non-positive n_iter.
    """
    return None
```

### Step 2

stf_ibi_potential

Goal
----
Reconstructs the reduced effective pair potential from the radial distribution function by a fixed number of iterative Boltzmann inversion steps.

```python
def stf_ibi_potential(g_target: "np.ndarray", rho: float, gamma: float, n_ibi: int) -> "np.ndarray":
    r"""g_target: $(1024,)$ array, the radial distribution function on the fixed grid.
    rho: positive float, number density.
    gamma: positive float, the mixing parameter of the inversion update.
    n_ibi: positive integer, number of inversion iterations.

    Returns a numpy float64 array of shape $(1024,)$: the reduced effective pair potential
    $\beta\phi(r)$ reconstructed from g_target by iterative Boltzmann inversion, started from the
    low-density inversion and updated exactly n_ibi times with the closure the source prescribes
    for interactions that carry a hard core. Inside the core, where g_target is below 1e-6, the
    potential is held at 60. The inner integral-equation solve is warm-started between inversion
    steps and takes exactly 80 fixed-point iterations with mixing 0.15 per step.

    Raises:
        ValueError: on a g_target that is not of length 1024 or not finite, on non-finite or
            non-positive rho or gamma, or non-integral or non-positive n_ibi.
    """
    return None
```

### Step 3

stf_wca_split

Goal
----
Splits the reconstructed potential into a short-range reference part and a tail part at the position of its minimum.

```python
def stf_wca_split(bphi: "np.ndarray", g_target: "np.ndarray") -> "np.ndarray":
    r"""bphi: $(1024,)$ array, reduced effective pair potential on the fixed grid.
    g_target: $(1024,)$ array, the radial distribution function used to locate the core.

    Returns a numpy float64 array of shape $(2, 1024)$: row 0 the reduced reference (core)
    potential and row 1 the reduced tail potential, split at the position of the minimum of
    bphi outside the core, following the decomposition the source adopts. The two rows sum to
    bphi at every grid point. What each row takes inside the split radius follows the source.

    Raises:
        ValueError: on arrays not of length 1024 or not finite.
    """
    return None
```

### Step 4

stf_reference_rdf

Goal
----
Solves for the radial distribution function of the reference (core-only) system by a fixed-count integral-equation solve.

```python
def stf_reference_rdf(bphi_ref: "np.ndarray", rho: float, n_iter: int) -> "np.ndarray":
    r"""bphi_ref: $(1024,)$ array, the reduced reference potential on the fixed grid.
    rho: positive float, number density.
    n_iter: positive integer, number of fixed-point iterations.

    Returns a numpy float64 array of shape $(1024,)$: the radial distribution function of the
    reference system, from the Ornstein-Zernike relation with the closure the source prescribes
    for interactions that carry a hard core, by a Picard iteration of exactly n_iter steps with
    mixing 0.15 and no tolerance exit.

    Raises:
        ValueError: on a bphi_ref not of length 1024 or not finite, non-finite or non-positive
            rho, or non-integral or non-positive n_iter.
    """
    return None
```

### Step 5

stf_hs_diameter

Goal
----
Assigns the effective hard-sphere diameter by the source's structure-matching procedure, reports the comparison cutoff it uses, and the diameter the conventional prescription would give.

```python
def stf_hs_diameter(g_ref: "np.ndarray", g_target: "np.ndarray", bphi_ref: "np.ndarray", rho: float) -> "np.ndarray":
    r"""g_ref: $(1024,)$ array, the reference-system radial distribution function.
    g_target: $(1024,)$ array, the measured radial distribution function.
    bphi_ref: $(1024,)$ array, the reduced reference potential.
    rho: positive float, number density.

    Returns a numpy float64 array of shape $(3,)$: the effective hard-sphere diameter the
    source's procedure assigns to this structure, the cutoff distance below which that procedure
    compares structures, and for comparison the diameter the conventional prescription would give
    from bphi_ref. The source's procedure scans diameters from 0.86 to 1.04 on 19 equally spaced
    points, then refines the best bracket by exactly 12 golden-section steps and returns the
    bracket midpoint; hard-sphere structures for the comparison come from the closed-form
    solution of the hard-core closure for hard spheres in $k$ space (Wertheim), carried through
    the Ornstein-Zernike relation and transformed onto the grid. Peaks of the measured function
    are strict local maxima that exceed 1.02; the measured function is used as it stands at every
    distance and the cutoff serves only as the upper limit of the comparison. The conventional
    diameter is the integral from zero of one minus the Boltzmann factor of bphi_ref, with the
    integrand taken as one at the origin. Integrals are trapezoid sums on the grid.

    Raises:
        ValueError: on arrays not of length 1024 or not finite, or non-finite or non-positive rho.
    """
    return None
```

### Step 6

stf_integrated_strength

Goal
----
Computes the integrated strength of the tail interaction by the thermodynamic-integration route and by the second-virial route.

```python
def stf_integrated_strength(g_target: "np.ndarray", bphi_ref: "np.ndarray", bphi_tail: "np.ndarray") -> "np.ndarray":
    r"""g_target: $(1024,)$ array, the measured radial distribution function.
    bphi_ref, bphi_tail: $(1024,)$ arrays, the reduced reference and tail potentials.

    Returns a numpy float64 array of shape $(2,)$: the reduced integrated interaction strength
    of the tail by the thermodynamic-integration route with the approximation the source adopts
    for the correlation weight, and the reduced integrated strength by the second-virial route the
    source gives as its dilute-limit alternative. Both are in units of $k_B T\,\sigma^3$.

    Raises:
        ValueError: on arrays not of length 1024 or not finite.
    """
    return None
```

### Step 7

stf_free_energy

Goal
----
Assembles the reduced excess free energy per particle from the hard-sphere reference at the assigned diameter and the integrated tail strength.

```python
def stf_free_energy(rho: float, sigma: float, a_strength: float) -> "np.ndarray":
    r"""rho: positive float, number density.
    sigma: positive float, the effective hard-sphere diameter.
    a_strength: finite float, the reduced integrated tail strength.

    Returns a numpy float64 array of shape $(3,)$: the reduced excess free energy per particle
    $\beta F^{\rm exc}/N$ of the fluid, the reduced excess free energy per particle of the
    hard-sphere reference alone from the equation of state the source recommends, and that
    reference's reduced compressibility factor $\beta P/\rho$.

    Raises:
        ValueError: on non-finite or non-positive rho or sigma, non-finite a_strength, or a
            packing fraction at or above one.
    """
    return None
```

### Step 8

stf_audit

Goal
----
Runs the whole study from the generated structure to the excess free energy, reporting every intermediate the source's construction depends on.

```python
def stf_audit(rho: float, beta_eps: float, rc: float, gamma: float, n_ibi: int) -> "np.ndarray":
    r"""rho, beta_eps, rc: as in the first step. gamma, n_ibi: as in the second step.

    The orchestrator. It must call the earlier functions rather than reimplementing them, using
    400 iterations to generate the structural input and 300 for every reference-system solve.
    Returns a numpy float64 array of shape $(9,)$: the split radius, the comparison cutoff, the
    effective hard-sphere diameter by the source's procedure, the diameter by the conventional
    prescription, the integrated strength by the thermodynamic-integration route, the integrated
    strength by the second-virial route, the hard-sphere reference free energy per particle at the
    source's diameter, the reduced excess free energy per particle using the thermodynamic-
    integration strength, and the same using the second-virial strength.

    Raises:
        ValueError: whenever any of the functions it calls would raise.
    """
    return None
```
