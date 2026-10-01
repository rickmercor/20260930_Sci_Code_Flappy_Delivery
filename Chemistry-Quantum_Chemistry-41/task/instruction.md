# Chemistry-Quantum_Chemistry-41

## Background

Photoelectron and electron-attachment spectroscopies measure the energies needed to remove an electron from a molecule or to add one to it. In many-body perturbation theory these energies are the poles of the one-particle Green's function, and the corresponding quasiparticle energies replace the orbital energies of mean-field theory. Hartree–Fock orbital energies, read through Koopmans' theorem, give only a first estimate of these energies.

Green's-function methods correct mean-field levels with a self-energy and exist in several self-consistent variants.

Quasiparticle energies depend on the one-particle basis, and compact augmented Gaussian basis sets have been designed for quasiparticle and excited-state calculations on large molecules.

The nitrogen molecule is a standard small-molecule benchmark for quasiparticle methods.

## Problem

Quasiparticle energies, the energies needed to remove an electron from a molecule or to add one to it, are routinely computed with many-body Green's-function methods, and the difference between the lowest unoccupied and the highest occupied quasiparticle level estimates the fundamental gap. A 2026 study introduced a similarity-renormalization-group variant of quasiparticle-self-consistent second-order Green's-function theory and published several parametrizations of it, with empirical parameters fitted to benchmark ionization potentials and electron affinities. Given a closed-shell molecule and a Gaussian basis set, the method yields its quasiparticle energies.

Apply that method to the nitrogen molecule N2 at a bond length of 1.0977 Å, using the published values of the parametrization that its developers obtained by fitting all of its empirical parameters, not only its flow parameter, to ionization potentials alone rather than to a balance of ionization potentials and electron affinities. Use the standard single-zeta member of the augmented all-electron MOLOPT Gaussian basis-set family introduced in 2025 for GW and Bethe–Salpeter calculations on large molecules (the standard set, not its reduced or short-range variants), with the exponents and contraction coefficients exactly as its developers published them. The contraction coefficients multiply normalized primitive Gaussians, and every shell uses real spherical-harmonic (pure) angular functions rather than Cartesian ones.

Start the quasiparticle self-consistency from the converged closed-shell restricted Hartree–Fock solution, include all electrons and all orbitals (no frozen core and no truncation of the virtual space), evaluate all integrals exactly (no density fitting), and iterate until no quasiparticle energy changes by more than 1e-10 hartree between cycles. Take 1 bohr = 0.52917721092 Å and 1 hartree = 27.211386245988 eV.

Compute the quasiparticle HOMO–LUMO gap of N2, the lowest unoccupied minus the highest occupied self-consistent quasiparticle energy, in eV.

In your reasoning, report the parameter values you used; the basis set you used, with its contracted-function composition per nitrogen atom and its most diffuse primitive exponent; the number of basis functions; the nuclear-repulsion energy; the restricted Hartree–Fock total energy and HOMO–LUMO gap; the converged quasiparticle HOMO and LUMO energies; and the symmetry labels of the molecular orbitals that form the HOMO and the LUMO.

Give the final answer as the gap in eV. Between the <final_answer> tags place only that value, written as one finite decimal number with at least six significant figures, with no units, symbols, words or any other text.

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

normalize_contraction

Goal
----
Implement normalize_contraction, which converts the tabulated contraction coefficients of one
contracted Gaussian shell into weights of unnormalized radial primitives that give a
unit-norm contracted radial function.

```python
def normalize_contraction(l: int, exponents: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    '''Weights of unnormalized radial primitives for a unit-norm contracted Gaussian shell.

    Parameters
    ----------
    l : int
        Angular momentum of the shell, a non-negative integer.
    exponents : np.ndarray
        Primitive exponents beta_k (bohr^-2), shape (n,), n >= 1, all positive.
    coefficients : np.ndarray
        Tabulated contraction coefficients c_k, shape (n,), of either sign; zeros are allowed
        but not all of them may vanish. c_k multiplies the normalized radial primitive
        N_k r^l exp(-beta_k r^2), where N_k makes int_0^inf (N_k r^l exp(-beta_k r^2))^2 r^2 dr = 1.

    Returns
    -------
    weights : np.ndarray
        Shape (n,). Weights w_k such that the radial function
        R(r) = sum_k w_k r^l exp(-beta_k r^2) is proportional to sum_k c_k N_k r^l exp(-beta_k r^2)
        with a positive factor and satisfies int_0^inf R(r)^2 r^2 dr = 1.

    Raises
    ------
    ValueError
        If l is negative, exponents and coefficients are empty or of different lengths, an
        exponent is not positive, or every coefficient is zero.
    '''
    return weights
```

### Step 2

compute_ao_integrals

Goal
----
Implement compute_ao_integrals, which evaluates the overlap, core-Hamiltonian and
electron-repulsion integrals and the nuclear-repulsion energy of a molecule over a basis of
contracted real solid-harmonic Gaussian functions with s, p and d angular momentum.

```python
def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    '''Atomic-orbital integrals of a molecule in a contracted solid-harmonic Gaussian basis.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,), all positive.
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells in the order that defines the atomic-orbital ordering. Each entry is a
        tuple (atom, l, exponents, coefficients): atom is the row of coords the shell is
        centred on, l its angular momentum (0, 1 or 2), exponents and coefficients the
        primitive exponents and tabulated contraction coefficients in the convention of
        normalize_contraction. A shell contributes 2l + 1 functions. Each function is the
        product of an angular polynomial in the displacement (x, y, z) from its atom, the
        radial sum sum_k w_k exp(-beta_k r^2) with the weights w_k of normalize_contraction,
        and a positive factor giving unit self-overlap. The polynomials, in this order, are
        1 for s; x, y, z for p; and xy, yz, 2z^2 - x^2 - y^2, xz, x^2 - y^2 for d.

    Returns
    -------
    integrals : tuple
        (overlap, core_hamiltonian, eri, e_nuc):
        overlap : np.ndarray, shape (n_bf, n_bf), <mu|nu>.
        core_hamiltonian : np.ndarray, shape (n_bf, n_bf), electron kinetic energy plus the
            attraction to all nuclei, in hartree.
        eri : np.ndarray, shape (n_bf, n_bf, n_bf, n_bf), electron-repulsion integrals in
            chemists' notation, eri[p, q, r, s] = (pq|rs), in hartree.
        e_nuc : float, nuclear-repulsion energy in hartree.

    Raises
    ------
    ValueError
        If charges and coords describe different numbers of atoms, two nuclei coincide, a
        shell has an angular momentum other than 0, 1 or 2 or refers to a missing atom, or a
        shell's exponents and coefficients are invalid as in normalize_contraction.
    '''
    return (overlap, core_hamiltonian, eri, e_nuc)
```

### Step 3

run_rhf

Goal
----
Implement run_rhf, which solves the closed-shell restricted Hartree-Fock equations for given
atomic-orbital integrals and returns the total energy, the canonical orbital energies and the
molecular-orbital coefficients.

```python
def run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    '''Closed-shell restricted Hartree-Fock solution.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf), symmetric positive definite.
    core_hamiltonian : np.ndarray
        Core Hamiltonian h (kinetic energy plus nuclear attraction), shape (n_bf, n_bf), in
        hartree.
    eri : np.ndarray
        Electron-repulsion integrals (pq|rs) in chemists' notation, shape
        (n_bf, n_bf, n_bf, n_bf), in hartree.
    e_nuc : float
        Nuclear-repulsion energy in hartree.
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= n_bf.

    Returns
    -------
    result : tuple
        (e_total, orbital_energies, coefficients):
        e_total : float, electronic energy plus e_nuc, in hartree.
        orbital_energies : np.ndarray, shape (n_bf,), canonical orbital energies in ascending
            order, in hartree.
        coefficients : np.ndarray, shape (n_bf, n_bf), orbital coefficients as columns in the
            same order, with C^T S C = 1; the first n_occ columns are occupied.
        The solution is the aufbau solution, converged until the orbital-gradient matrix
        F P S - S P F has no element larger than 1e-10 (P the density matrix, F the Fock
        matrix).

    Raises
    ------
    ValueError
        If the matrices have inconsistent shapes, n_occ is outside 1..n_bf, or the iterations
        do not converge within 200 cycles.
    '''
    return (e_total, orbital_energies, coefficients)
```

### Step 4

srg_regulator

Goal
----
Implement srg_regulator, which evaluates elementwise the similarity-renormalization-group
denominator factor f(D_p, D_q; s) defined in the signature.

```python
def srg_regulator(delta_p: "np.ndarray", delta_q: "np.ndarray", s: float) -> "np.ndarray":
    '''Renormalized denominator factor f(D_p, D_q; s), evaluated elementwise.

    Parameters
    ----------
    delta_p : np.ndarray
        Energy differences D_p in hartree, any shape.
    delta_q : np.ndarray
        Energy differences D_q in hartree, broadcast-compatible with delta_p.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.

    Returns
    -------
    factor : np.ndarray
        Broadcast shape of delta_p and delta_q, in hartree^-1:
        f = (D_p + D_q) / (D_p^2 + D_q^2) * [1 - exp(-(D_p^2 + D_q^2) s)],
        with f extended continuously to the points where D_p = D_q = 0.

    Raises
    ------
    ValueError
        If s is negative or not finite, or delta_p and delta_q cannot be broadcast together.
    '''
    return factor
```

### Step 5

srg_self_energy

Goal
----
Implement srg_self_energy, which builds the renormalized static second-order self-energy
matrix, with separate scaling of its same-spin and opposite-spin parts, in the basis of the
current molecular orbitals.

```python
def srg_self_energy(eri_mo: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Spin-component-scaled renormalized static second-order self-energy (MO basis).

    Parameters
    ----------
    eri_mo : np.ndarray
        Electron-repulsion integrals over the current real spatial orbitals in chemists'
        notation, eri_mo[p, q, r, s] = (pq|rs), shape (n, n, n, n), in hartree.
    orbital_energies : np.ndarray
        Current orbital energies e_p, shape (n,), in hartree; orbitals 0..n_occ-1 are the
        doubly occupied ones (indices i, j), the others are virtual (indices a, b).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n - 1.
    s : float
        Flow parameter in hartree^-2 (finite, non-negative).
    c_ss : float
        Same-spin scaling factor.
    c_os : float
        Opposite-spin scaling factor.

    Returns
    -------
    sigma : np.ndarray
        Symmetric matrix of shape (n, n), in hartree:
        sigma[p, q] = sum_{i,j,a} f(D^{pa}_{ij}, D^{qa}_{ij}; s) (pi|aj) [(c_ss + c_os)(qi|aj) - c_ss (qj|ai)]
                    + sum_{a,b,i} f(D^{pi}_{ab}, D^{qi}_{ab}; s) (pa|ib) [(c_ss + c_os)(qa|ib) - c_ss (qb|ia)],
        with D^{pa}_{ij} = e_p + e_a - e_i - e_j, D^{pi}_{ab} = e_p + e_i - e_a - e_b and f the
        factor of srg_regulator.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is outside 1..n-1, c_ss or c_os is not finite,
        or s is invalid as in srg_regulator.
    '''
    return sigma
```

### Step 6

qs_fock_matrix

Goal
----
Implement qs_fock_matrix, which assembles, in the atomic-orbital basis, the effective
one-particle matrix of one quasiparticle-self-consistency cycle of the renormalized
second-order Green's-function method for given orbitals and orbital energies.

```python
def qs_fock_matrix(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", coefficients: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Atomic-orbital matrix of the Fock operator plus the renormalized static self-energy.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf).
    core_hamiltonian : np.ndarray
        Core Hamiltonian h, shape (n_bf, n_bf), in hartree.
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals (mu nu|lambda sigma) in chemists'
        notation, shape (n_bf, n_bf, n_bf, n_bf), in hartree.
    coefficients : np.ndarray
        Current orbital coefficients C as columns, shape (n_bf, n_bf), with C^T S C = 1; the
        first n_occ columns are the doubly occupied orbitals.
    orbital_energies : np.ndarray
        Current orbital energies, shape (n_bf,), in hartree, in the column order of C.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf - 1.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    fock : np.ndarray
        Symmetric matrix of shape (n_bf, n_bf), in hartree:
        the atomic-orbital representation of the closed-shell Fock operator of the determinant
        in which the first n_occ columns of C are doubly occupied, plus that of the operator
        whose matrix in the orbitals C is sigma, the result of srg_self_energy for the
        integrals transformed to the orbitals C and the orbital energies given. The
        atomic-orbital representation of an operator is the matrix of its elements between
        atomic orbitals, so that C^T fock C is the matrix of the same operator in the orbitals
        C.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, or n_occ, s, c_ss or c_os is invalid as in
        srg_self_energy.
    '''
    return fock
```

### Step 7

run_srg_qsgf2

Goal
----
Implement run_srg_qsgf2, which iterates the quasiparticle-self-consistent cycle of the
renormalized second-order Green's-function method from a restricted Hartree-Fock start to
self-consistency and returns the converged quasiparticle energies.

```python
def run_srg_qsgf2(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Converged quasiparticle energies of the renormalized qs second-order scheme.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf).
    core_hamiltonian : np.ndarray
        Core Hamiltonian h, shape (n_bf, n_bf), in hartree.
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals in chemists' notation, shape
        (n_bf, n_bf, n_bf, n_bf), in hartree.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf - 1.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    qp_energies : np.ndarray
        Shape (n_bf,), in hartree, ascending: the quasiparticle energies at self-consistency,
        i.e. the generalized eigenvalues, with respect to the overlap S, of the matrix F of
        qs_fock_matrix evaluated with its own S-orthonormal eigenvectors as orbitals (the n_occ
        lowest occupied) and its own eigenvalues as orbital energies. The cycle starts from the
        converged restricted Hartree-Fock orbitals and orbital energies of run_rhf and is
        converged until no quasiparticle energy changes by more than 1e-10 hartree between
        cycles and the orbital-gradient matrix F P S - S P F (P the density matrix) has no
        element larger than 1e-9.

    Raises
    ------
    ValueError
        If the inputs are invalid as in qs_fock_matrix or run_rhf, or the cycle does not
        converge within 300 iterations.
    '''
    return qp_energies
```

### Step 8

quasiparticle_gap

Goal
----
Implement quasiparticle_gap, the end-to-end pipeline, which returns the quasiparticle
HOMO-LUMO gap of a closed-shell molecule, in electronvolts, from the renormalized
quasiparticle-self-consistent second-order Green's-function method in a given contracted
Gaussian basis.

```python
def quasiparticle_gap(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int, s: float, c_ss: float, c_os: float) -> float:
    '''Quasiparticle HOMO-LUMO gap (eV) of a closed-shell molecule.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells as in compute_ao_integrals.
    n_electrons : int
        Number of electrons, a positive even integer smaller than 2 n_bf.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    gap : float
        (e_LUMO - e_HOMO) * 27.211386245988, in eV, where e_HOMO and e_LUMO are the
        quasiparticle energies with indices n_electrons/2 - 1 and n_electrons/2 of
        run_srg_qsgf2 for the integrals of compute_ao_integrals.

    Raises
    ------
    ValueError
        If n_electrons is not a positive even integer with n_electrons/2 < n_bf, or the
        inputs are invalid as in compute_ao_integrals or run_srg_qsgf2.
    '''
    return gap
```
