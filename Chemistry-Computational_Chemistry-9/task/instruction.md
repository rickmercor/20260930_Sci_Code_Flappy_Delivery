# Chemistry-Computational_Chemistry-9

## Background

Second-order Møller–Plesset perturbation theory adds electron correlation to a Hartree–Fock reference at modest cost and remains a workhorse of molecular quantum chemistry, both on its own and as the correlation ingredient of double-hybrid density functionals. Empirically reweighted variants of it improve many reaction energies and barrier heights.

The simultaneous stretch of both O–H bonds of water at a fixed bond angle is a widely used test of correlated electronic-structure methods.

## Problem

Spin-component-scaled second-order Møller–Plesset (MP2) theory applies one fixed pair of weights to the opposite-spin and same-spin parts of the MP2 correlation energy, although the best pair differs from one chemical system to another. A 2025 study replaced the fixed pair with weights that vary from system to system, with global parameters trained on a representative subset of a large main-group thermochemistry benchmark. Given a closed-shell molecule at a fixed geometry and a basis set, the scheme returns a total energy (electronic energy plus nuclear repulsion).

Apply the system-dependent variant of that scheme with two fitted parameters, with its published parameter values, to the symmetric double stretch of water. The source does not state this variant's fitted values consistently in every place; use the reading that is consistent with the weights it reports for its training molecules. Both structures have C2v symmetry with an H–O–H angle of 110.565°; both O–H bonds are R = 1.84345 bohr in the reference structure and R = 3.68690 bohr in the stretched structure. Describe each structure as a closed-shell singlet by a single restricted Hartree–Fock determinant, correlate all ten electrons with no frozen core, and use the 6-31G basis set below. Its functions are contracted Cartesian Gaussians whose contraction coefficients multiply normalized primitive Gaussians; the s and p members of an SP shell share their exponents.

The basis consists of the following shells.

Oxygen, shell O-1 (s, six primitives). Exponents: 5484.671660, 825.2349460, 188.0469580, 52.96450000, 16.89757040, 5.799635340. Coefficients: 0.001831074430, 0.01395017220, 0.06844507810, 0.2327143360, 0.4701928980, 0.3585208530.

Oxygen, shell O-2 (sp, three primitives). Exponents: 15.53961625, 3.599933586, 1.013761750. s coefficients: -0.1107775495, -0.1480262627, 1.130767015. p coefficients: 0.07087426823, 0.3397528391, 0.7271585773.

Oxygen, shell O-3 (sp, one primitive). Exponent: 0.2700058226. s coefficient 1.0, p coefficient 1.0.

Each hydrogen, shell H-1 (s, three primitives). Exponents: 18.73113696, 2.825394365, 0.6401216923. Coefficients: 0.03349460434, 0.2347269535, 0.8137573261.

Each hydrogen, shell H-2 (s, one primitive). Exponent: 0.1612777588. Coefficient 1.0.

Compute the energy required for the stretch, ΔE = E(3.68690 bohr) − E(1.84345 bohr), in hartree, as predicted by that two-parameter scheme.

Give the final answer as a single finite decimal number in hartree with at least 6 significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

compute_ao_integrals

Goal
----
Implement compute_ao_integrals, which evaluates the one- and two-electron integrals of a
molecule over a basis of contracted Cartesian Gaussian functions.

```python
def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    '''Atomic-orbital integrals of a molecule in a contracted Cartesian Gaussian basis.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,), all positive.
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3); no two nuclei coincide.
    shells : list
        Basis shells in the order that defines the atomic-orbital ordering. Each entry is a
        tuple (atom, l, exponents, coefficients): atom is the row of coords the shell is
        centred on, l is the angular momentum (0 for s, 1 for p, 2 for d), exponents is a
        sequence of positive primitive exponents and coefficients an equal-length sequence
        of contraction coefficients of either sign. The coefficients multiply normalized
        primitive Gaussians, and every contracted Cartesian function is normalized to unit
        self-overlap. An s shell contributes one function; a p shell contributes three,
        ordered x, y, z; a d shell contributes six Cartesian functions, ordered xx, xy, xz,
        yy, yz, zz.

    Returns
    -------
    integrals : tuple
        (overlap, core_hamiltonian, eri, e_nuc):
        overlap : np.ndarray, shape (n_bf, n_bf), <mu|nu>.
        core_hamiltonian : np.ndarray, shape (n_bf, n_bf), kinetic energy plus attraction
            to all nuclei, in hartree.
        eri : np.ndarray, shape (n_bf, n_bf, n_bf, n_bf), electron-repulsion integrals in
            chemists' notation, eri[p, q, r, s] = (pq|rs), in hartree.
        e_nuc : float, nuclear-repulsion energy in hartree (0.0 for a single atom).

    Raises
    ------
    ValueError
        If a shell has an angular momentum other than 0, 1 or 2, refers to a missing atom,
        has exponents and coefficients of different lengths, or has a non-positive
        exponent.
    '''
    return (overlap, core_hamiltonian, eri, e_nuc)
```

### Step 2

run_rhf

Goal
----
Implement run_rhf, which solves the closed-shell restricted Hartree-Fock equations for a
molecule whose atomic-orbital integrals are given.

```python
def run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    '''Closed-shell restricted Hartree-Fock ground state.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix, shape (n_bf, n_bf), symmetric positive definite.
    core_hamiltonian : np.ndarray
        Core Hamiltonian (kinetic plus nuclear attraction), shape (n_bf, n_bf), hartree.
    eri : np.ndarray
        Electron-repulsion integrals (pq|rs) in chemists' notation, shape (n_bf,)*4, hartree.
    e_nuc : float
        Nuclear-repulsion energy, hartree.
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= n_bf.

    Returns
    -------
    result : tuple
        (e_hf, mo_energies, mo_coeff):
        e_hf : float, total restricted Hartree-Fock energy including e_nuc, hartree.
        mo_energies : np.ndarray, shape (n_bf,), canonical orbital energies in ascending
            order, hartree.
        mo_coeff : np.ndarray, shape (n_bf, n_bf), canonical molecular-orbital
            coefficients (column k is orbital k), orthonormal in the overlap metric;
            the first n_occ columns are the doubly occupied orbitals.
        The solution is the aufbau closed-shell self-consistent solution, converged so
        that e_hf is accurate to 1e-10 hartree and every element of the occupied-orbital
        density matrix to 1e-8. Orbital phases (and rotations within exactly degenerate
        sets) are arbitrary.

    Raises
    ------
    ValueError
        If the matrix shapes are inconsistent, the overlap matrix is not positive
        definite, n_occ is outside 1..n_bf, or the iterations fail to converge.
    '''
    return (e_hf, mo_energies, mo_coeff)
```

### Step 3

mp2_spin_components

Goal
----
Implement mp2_spin_components, which evaluates the second-order Møller-Plesset correlation
energy of a closed-shell molecule resolved into its opposite-spin and same-spin parts.

```python
def mp2_spin_components(eri: "np.ndarray", mo_coeff: "np.ndarray", mo_energies: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Opposite-spin and same-spin parts of the closed-shell MP2 correlation energy.

    Parameters
    ----------
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals (pq|rs), chemists' notation,
        shape (n_bf,)*4, hartree.
    mo_coeff : np.ndarray
        Canonical restricted Hartree-Fock orbital coefficients, shape (n_bf, n_mo),
        columns ordered by orbital energy; the first n_occ columns are doubly occupied.
    mo_energies : np.ndarray
        Canonical orbital energies, shape (n_mo,), hartree.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n_mo; every orbital is correlated.

    Returns
    -------
    components : np.ndarray
        Array [e_os, e_ss] of shape (2,), hartree: the opposite-spin and same-spin
        contributions to the MP2 correlation energy, whose sum is the full MP2
        correlation energy of the reference.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ leaves no virtual orbital, or an
        occupied orbital energy is not below every virtual orbital energy.
    '''
    return components
```

### Step 4

unrelaxed_mp2_density

Goal
----
Implement unrelaxed_mp2_density, which builds the unrelaxed second-order Møller-Plesset
one-particle reduced density matrix of a closed-shell molecule in its canonical
molecular-orbital basis.

```python
def unrelaxed_mp2_density(eri: "np.ndarray", mo_coeff: "np.ndarray", mo_energies: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Unrelaxed MP2 one-particle density matrix (spin-summed) in the canonical MO basis.

    Parameters
    ----------
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals (pq|rs), chemists' notation,
        shape (n_bf,)*4, hartree.
    mo_coeff : np.ndarray
        Canonical restricted Hartree-Fock orbital coefficients, shape (n_bf, n_mo),
        columns ordered by orbital energy; the first n_occ columns are doubly occupied.
    mo_energies : np.ndarray
        Canonical orbital energies, shape (n_mo,), hartree.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n_mo; every orbital is correlated.

    Returns
    -------
    density : np.ndarray
        Symmetric unrelaxed MP2 density matrix of shape (n_mo, n_mo) in the basis of
        mo_coeff, summed over both spins.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ leaves no virtual orbital, or an
        occupied orbital energy is not below every virtual orbital energy.
    '''
    return density
```

### Step 5

correlation_indices

Goal
----
Implement correlation_indices, which measures the dynamic and nondynamic electron
correlation of a closed-shell molecule from the natural-orbital occupations of its
one-particle density matrix.

```python
def correlation_indices(density: "np.ndarray", n_electrons: int) -> "np.ndarray":
    '''Per-electron dynamic, nondynamic and total correlation indices.

    Parameters
    ----------
    density : np.ndarray
        Spin-summed one-particle density matrix of a closed-shell state in an orthonormal
        orbital basis, symmetric, shape (n_mo, n_mo). Its eigenvalues (spatial natural
        occupations, nominally between 0 and 2; small non-positive eigenvalues may occur and
        are skipped) are shared equally by the two spins, so each spatial occupation eta
        contributes two natural spin orbitals of occupation eta / 2.
    n_electrons : int
        Number of electrons N, positive and even.

    Returns
    -------
    indices : np.ndarray
        Array [I_D, I_ND, I_T] of shape (3,), each dimensionless. Natural spin orbitals
        whose occupation is not strictly positive are left out of the sums. The
        per-electron normalization is fixed by one reference state: a two-electron
        density whose two spatial natural orbitals each hold one electron (all four
        natural spin orbitals at occupation 1/2) has I_D = 0 and I_ND = I_T = 1/2.

    Raises
    ------
    ValueError
        If density is not a square symmetric matrix, n_electrons is not a positive even
        integer, or a spatial occupation exceeds 2 by more than 1e-10.
    '''
    return indices
```

### Step 6

correlation_driven_weights

Goal
----
Implement correlation_driven_weights, which converts the correlation indices of a molecule
into the opposite-spin and same-spin weights of its spin-component-scaled MP2 energy.

```python
def correlation_driven_weights(indices: "np.ndarray") -> "np.ndarray":
    '''Opposite-spin and same-spin weights of the two-parameter correlation-driven scheme.

    Parameters
    ----------
    indices : np.ndarray
        Array [I_D, I_ND, I_T] of per-electron dynamic, nondynamic and total correlation
        indices, shape (3,), with I_D >= 0, I_ND >= 0, I_T > 0 and I_T = I_D + I_ND
        (to within 1e-8 relative).

    Returns
    -------
    weights : np.ndarray
        Array [c_OS, c_SS] of shape (2,), dimensionless.

    Raises
    ------
    ValueError
        If indices does not hold three finite values, I_D or I_ND is negative, I_T is not
        positive, or I_T differs from I_D + I_ND.
    '''
    return weights
```

### Step 7

cd_scs_mp2_energy

Goal
----
Implement cd_scs_mp2_energy, which returns the total correlation-driven spin-component-scaled
MP2 energy of a closed-shell molecule at one fixed geometry.

```python
def cd_scs_mp2_energy(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int) -> float:
    '''Correlation-driven spin-component-scaled MP2 total energy at one geometry.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells (atom, l, exponents, coefficients) in the format of
        compute_ao_integrals (normalized contracted Cartesian functions).
    n_electrons : int
        Number of electrons of the closed-shell singlet, positive and even.

    Returns
    -------
    energy : float
        E_HF + c_OS * E_OS + c_SS * E_SS in hartree, where E_HF includes nuclear
        repulsion and (c_OS, c_SS) are the two-parameter correlation-driven weights of
        this molecule at this geometry.

    Raises
    ------
    ValueError
        If n_electrons is not a positive even integer, the closed-shell reference has no
        virtual orbital, or any input is rejected by the steps it relies on.
    '''
    return energy
```

### Step 8

stretch_energy

Goal
----
Implement stretch_energy, which returns the correlation-driven spin-component-scaled MP2
energy change between two geometries of the same closed-shell molecule.

```python
def stretch_energy(charges: "np.ndarray", coords_initial: "np.ndarray", coords_final: "np.ndarray", shells: list, n_electrons: int) -> float:
    '''Correlation-driven SCS-MP2 energy change E(final) - E(initial) of one molecule.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords_initial : np.ndarray
        Nuclear positions of the initial geometry in bohr, shape (n_atoms, 3).
    coords_final : np.ndarray
        Nuclear positions of the final geometry in bohr, shape (n_atoms, 3), same atom order.
    shells : list
        Basis shells (atom, l, exponents, coefficients) in the format of
        compute_ao_integrals; the same shells are used at both geometries.
    n_electrons : int
        Number of electrons of the closed-shell singlet, positive and even.

    Returns
    -------
    delta_e : float
        E(final) - E(initial) in hartree, each energy evaluated with the two-parameter
        correlation-driven weights of the molecule at that geometry.

    Raises
    ------
    ValueError
        If the two geometries do not have the shape (n_atoms, 3) implied by charges, or
        any input is rejected by the steps it relies on.
    '''
    return delta_e
```
