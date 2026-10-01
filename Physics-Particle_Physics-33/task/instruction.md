# Physics-Particle_Physics-33

## Background

This task uses a recent Krylov-plus-large-N_c truncation of the Kogut-Susskind Hamiltonian for SU(N_c) lattice QCD with staggered fermions. Truncation T3 is the 1+1D qutrit theory on the three retained link irreps of Section IV.A.3. Truncation T1 is the corresponding qubit restriction of Section IV.A.1. Both act on a periodic chain. Section III.A, Eq. (4), generates allowed configurations from the free vacuum by a bounded number of hops. For the T1 and T3 Hamiltonians of Section IV.A that Krylov-truncated set is the invariant coordinate block of product configurations reachable through the retained hopping transitions, which is the vacuum-connected block used here. The lowest spectral gap is taken inside that block, without further restriction to a translation, reflection or other symmetry sector. The index is (Delta_T3 - Delta_T1) / E_meson at the same couplings.

## Problem

Compute a truncation gap index for the source paper's 1+1D T3 versus T1 Kogut-Susskind truncations of SU(N_c) lattice QCD with staggered fermions. The requested scalar is not reported in the source paper. Compare each truncation on its vacuum-connected block: all product-basis states connected to the all-singlet configuration by a chain of nonzero off-diagonal Hamiltonian entries. Use n_links=6, n_colors=3, coupling=1.0, mass=1.0 on a periodic chain. An off-diagonal entry connects its two states when its magnitude is strictly greater than 1e-12. The spectral gap is the second smallest eigenvalue minus the smallest, counted with multiplicity, on the stated block. Report the index after one final Python round(index, 10). Functions return unrounded values; do not round intermediate calculations.

The functions for the coding component are assemble_t3_hamiltonian, assemble_t1_hamiltonian, vacuum_connected_indices, lowest_spectral_gap, subspace_gap, isolated_meson_energy, and compute_truncation_gap_index.

Define the truncation gap index as (Delta_T3 - Delta_T1) / E_meson, where each Delta is the gap on that truncation's vacuum-connected block and E_meson is the isolated-meson energy.

In <reasoning>, identify the local-operator convention, the electric and mass terms, the four T3 hopping channels and their N_c=3 strengths, and the T1 hopping restriction. Algebraically equivalent expressions, including identity shifts of a Hamiltonian, are acceptable. Report both vacuum-block sizes, the T3 block's two lowest vacuum-referenced eigenvalues E_k - <0|H|0> with |0> the all-singlet configuration, both block gaps and their signed difference, the isolated-meson energy, and the T3 gap on the full product space. Explain why that last diagnostic is not the vacuum-block gap at these parameters. Report diagnostic energies, gaps and numerical coefficients to at least nine decimal places, or give exact expressions. Diagnostic numerical tolerance is 5e-9 absolute.

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

01_assemble_t3_hamiltonian

Goal
----
Assemble the 1+1D T3 Kogut-Susskind qutrit Hamiltonian.

```python
def assemble_t3_hamiltonian(n_links: int, n_colors: int, coupling: float,
                            mass: float) -> "np.ndarray":
    """Return the dense T3 qutrit Hamiltonian on a periodic chain.

    Parameters
    ----------
    n_links : int
        Number of links on the periodic chain. Either 4 or 6 (an even periodic chain).
    n_colors : int
        Number of colors N_c. Integer from 3 through 64.
    coupling : float
        Finite Yang-Mills coupling g.
    mass : float
        Finite staggered fermion mass m.

    Returns
    -------
    hamiltonian : np.ndarray, shape (3**n_links, 3**n_links), float
        Dense real symmetric T3 Hamiltonian in the computational basis above.

    Raises
    ------
    ValueError
        If either count is not an integer in its stated range, coupling or mass
        is not a finite real scalar, or the parameters produce non-finite entries.
    """
    return hamiltonian
```

### Step 2

02_assemble_t1_hamiltonian

Goal
----
Assemble the 1+1D T1 Kogut-Susskind qubit Hamiltonian.

```python
def assemble_t1_hamiltonian(n_links: int, n_colors: int, coupling: float,
                            mass: float) -> "np.ndarray":
    """Return the dense T1 qubit Hamiltonian on a periodic chain.

    Parameters
    ----------
    n_links : int
        Number of links on the periodic chain. Either 4 or 6 (an even periodic chain).
    n_colors : int
        Number of colors N_c. Integer from 3 through 64.
    coupling : float
        Finite Yang-Mills coupling g.
    mass : float
        Finite staggered fermion mass m.

    Returns
    -------
    hamiltonian : np.ndarray, shape (2**n_links, 2**n_links), float
        Dense real symmetric T1 Hamiltonian in the computational basis above.

    Raises
    ------
    ValueError
        If either count is not an integer in its stated range, coupling or mass
        is not a finite real scalar, or the parameters produce non-finite entries.
    """
    return hamiltonian
```

### Step 3

03_vacuum_connected_indices

Goal
----
Basis states reachable from the free vacuum.

```python
def vacuum_connected_indices(hamiltonian: object, tolerance: float) -> "np.ndarray":
    """Sorted basis indices in the vacuum block of a Hamiltonian.

    Parameters
    ----------
    hamiltonian : array-like, shape (dim, dim)
        Square Hamiltonian in a product computational basis whose index 0 is the vacuum.
        Finite entries. dim is at least 1.
    tolerance : float
        Magnitude threshold on an off-diagonal entry. Finite and at least 0. An entry
        links two indices when its absolute value is strictly greater than tolerance.

    Returns
    -------
    indices : np.ndarray, shape (n_connected,), int
        Strictly increasing basis indices reachable from index 0. Always contains 0.

    Raises
    ------
    ValueError
        If the Hamiltonian is not a non-empty finite real square matrix, or if
        tolerance is not a finite non-negative real scalar.
    """
    return indices
```

### Step 4

04_lowest_spectral_gap

Goal
----
Lowest spectral gap of a finite real square matrix.

```python
def lowest_spectral_gap(hamiltonian: object) -> float:
    """Lowest spectral gap of a real square matrix after symmetrization.

    Parameters
    ----------
    hamiltonian : array-like of shape (n, n)
        Finite real square matrix with n at least 2. It is symmetrized before
        diagonalization.

    Returns
    -------
    gap : float
        Second-lowest eigenvalue minus the lowest eigenvalue.

    Raises
    ------
    ValueError
        If hamiltonian is not a finite real square matrix with at least two rows,
        or a finite spectral gap cannot be computed.
    """
    return 0.0
```

### Step 5

05_subspace_gap

Goal
----
Lowest spectral gap of a Hamiltonian restricted to a block.

```python
def subspace_gap(hamiltonian: object, indices: object) -> float:
    """Lowest gap of a Hamiltonian restricted to selected basis indices.

    Parameters
    ----------
    hamiltonian : array-like, shape (dim, dim)
        Square Hamiltonian with finite entries.
    indices : array-like, shape (n_kept,), int
        Strictly increasing basis indices to keep. At least two entries, each in
        range(dim).

    Returns
    -------
    gap : float
        Second smallest minus smallest eigenvalue of the restricted, symmetrized block,
        with multiplicity. Native Python float.

    Raises
    ------
    ValueError
        If hamiltonian is not a finite real square matrix, or if indices is not a
        strictly increasing one-dimensional integer array of at least two valid rows,
        or a finite spectral gap cannot be computed.
    """
    return 0.0
```

### Step 6

06_isolated_meson_energy

Goal
----
Isolated meson energy of the truncated Kogut-Susskind theory.

```python
def isolated_meson_energy(n_colors: int, coupling: float, mass: float) -> float:
    """Energy of an isolated fundamental meson.

    Parameters
    ----------
    n_colors : int
        Number of colors N_c. Integer at least 2, representable as finite float64.
    coupling : float
        Finite Yang-Mills coupling g.
    mass : float
        Finite staggered fermion mass m.

    Returns
    -------
    energy : float
        Isolated-pair energy 2 m + (g^2 / 2) C_2(N).

    Raises
    ------
    ValueError
        If n_colors is not an integer at least 2, if coupling or mass is not a
        finite real scalar, or if those inputs produce a non-finite energy.
    """
    return 0.0
```

### Step 7

07_compute_truncation_gap_index

Goal
----
Truncation gap index of the SU(Nc) link truncations.

```python
def compute_truncation_gap_index(n_links: int, n_colors: int, coupling: float,
                                 mass: float) -> float:
    """Vacuum-block gap of T3 minus that of T1, per isolated meson.

    Parameters
    ----------
    n_links : int
        Number of links on the periodic chain. Either 4 or 6 (an even periodic chain).
    n_colors : int
        Number of colors. Integer from 3 through 64.
    coupling : float
        Gauge coupling. Finite.
    mass : float
        Fermion mass. Finite. Together with coupling it must give a non-zero
        isolated-meson energy.

    Returns
    -------
    index : float
        (T3 vacuum-block gap minus T1 vacuum-block gap) divided by the isolated-meson
        energy. Native Python float.

    Raises
    ------
    ValueError
        If either count is not an integer in its stated range, coupling or mass is
        not a finite real scalar, the isolated-meson energy is zero, or a finite
        Hamiltonian, spectral gap or final index cannot be computed.
    """
    return 0.0
```
