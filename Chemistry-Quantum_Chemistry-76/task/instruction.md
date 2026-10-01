# Chemistry-Quantum_Chemistry-76

## Background

Electron correlation is conventionally split into a weak part, which a single Slater determinant plus low order perturbation theory handles well, and a strong part, which appears whenever several configurations become nearly degenerate and which defeats a single determinant entirely. Bond dissociation is the standard laboratory for the strong part: as a bond stretches, the bonding and antibonding orbitals approach degeneracy and the closed shell determinant becomes a poor starting point long before anything interesting has happened chemically. Stretched hydrogen chains are the usual test case, because every bond can be pulled at once and the exact answer stays computable.

One productive way to organise the problem is by seniority, the number of orbitals a determinant leaves singly occupied. Numerical experiments over the past decade have shown that when orbitals are chosen well, almost all of the strong correlation lives in the seniority conserving part of the Coulomb Hamiltonian, so a wavefunction built entirely from paired electrons already captures the difficult physics. The difficulty is that a full configuration interaction restricted to seniority zero still scales exponentially, which is why a large literature has grown around pair based methods such as pair coupled cluster, antisymmetrised geminal powers and natural orbital functionals.

What such a wavefunction misses is the weak correlation, and recovering it perturbatively requires more than the reference itself: it requires excited states. For a single determinant those are the familiar singly and doubly substituted determinants and their energies are immediate. For a correlated pair wavefunction neither the enumeration nor the matrix elements are immediate, and supplying them is the technical content of the line of work this problem comes from. Perfect pairing, in which each electron pair is confined to its own two orbital space, has been used as a multireference ansatz for decades; what is recent is a complete and orthonormal set of low lying excited states built on it, together with closed form expressions for their excitation energies and their couplings to the reference.

The correction itself uses an Epstein-Nesbet partition, in which the zeroth order Hamiltonian is the diagonal of the true Hamiltonian in the reference and excited state basis. This is the crudest sensible choice, and it is the one used throughout this series so that different references can be compared on equal footing. Numerical tests on stretched hydrogen chains and on multiply bonded diatomics show that a correction built this way recovers a large share of the weak correlation the pair reference misses, and that it is competitive with a complete active space self consistent field treatment of the same valence space at a small fraction of the cost.

## Problem

A pair based reference for strongly correlated electrons can be corrected perturbatively once a complete set of its excited states is available, and your task is to carry that correction through on one concrete model and report a single total energy in hartree.

The model has eight collinear nuclei of unit charge at x = 0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2 and 23.6 bohr and eight electrons, each nucleus carrying one normalised s type Gaussian primitive of exponent 0.30, and every one and two electron integral is the analytic value over those primitives. Orthogonalise the eight functions symmetrically, then build four valence bond spaces from the symmetric and antisymmetric combinations of the orthogonalised functions on centres one and two, on centres three and four, on centres five and six, and on centres seven and eight, taking the symmetric combination as the bonding orbital of its space. Work in that orbital basis and keep it frozen, so that the only variational freedom is one amplitude per bond space controlling how its electron pair is shared between its bonding and antibonding orbital, and make those four amplitudes stationary for the Coulomb Hamiltonian.

The reference is the state in which every bond space carries the lower of the two pair states its two orbitals can form. Its excited states are all the remaining members of the orthonormal product basis in which each bond space is empty, is doubly filled, carries the reference pair state or the orthogonal partner of that pair state, or has one or both of its orbitals singly occupied, with all singly occupied orbitals coupled into an overall singlet. Take the diagonal of the Coulomb Hamiltonian in that basis as the zeroth order Hamiltonian and add the second order Epstein-Nesbet correction, which is minus the sum over the excited states of the squared coupling of that state to the reference through the Coulomb Hamiltonian, divided by the excitation energy of that state, where the excitation energy is its own Coulomb expectation value minus the Coulomb expectation value of the reference and is therefore positive. A state with four singly occupied orbitals spans a two dimensional singlet space, and the answer depends on how that space is resolved, so fix it as follows. Every such excitation is built from two moves, each move opens one pair of orbitals whether the two orbitals of that pair lie in the same bond space or in different ones, and the two basis vectors are the state in which each of those two pairs is separately coupled to a singlet together with the vector orthogonal to it within the same two dimensional space. Report the corrected total energy including nuclear repulsion, to eight decimal places, and in your reasoning give the scalars that determine it, namely the four stationary amplitudes, the intrabond pair-transfer integral of each bond space, the swap and the split excitation energy of the first bond space, the electronic energy of the reference, the nuclear repulsion, and the second order contribution of the single excitations, of the double excitations and of the pair transfers together with their total, naming in one line each the relations you used to obtain them.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.

You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
  Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
  lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Overlap and core Hamiltonian of the primitive basis

Goal
----
Return the overlap matrix and the core Hamiltonian of a minimal basis of one normalised s Gaussian primitive per collinear centre, stacked into a single array.

```python
def ao_overlap_and_core(centres: np.ndarray, exponent: float,
                        charges: np.ndarray) -> np.ndarray:
    '''Overlap and core Hamiltonian of the primitive basis.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr. One normalised s type
        primitive sits on each centre. The number of centres must be even.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.
    charges : np.ndarray
        Nuclear charge of each centre, one entry per centre.

    Returns
    -------
    stacked : np.ndarray
        Array of shape (2, N, N) whose first slice is the overlap matrix and
        whose second slice is the core Hamiltonian.
    
    Raises
    ------
    ValueError
        If `centres` is not a one dimensional array holding an even number of
        distinct positions, if `exponent` is not a positive scalar, or if
        `charges` does not hold one entry per centre.
'''
    return stacked
```

### Step 2

Two electron repulsion integrals of the primitive basis

Goal
----
Return the electron repulsion integrals of the same primitive basis in chemists' notation.

```python
def ao_two_electron_integrals(centres: np.ndarray, exponent: float) -> np.ndarray:
    '''Electron repulsion integrals of the primitive basis.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr, an even number of them.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.

    Returns
    -------
    eri : np.ndarray
        Array of shape (N, N, N, N) holding (mn|ab) in chemists' notation.
    
    Raises
    ------
    ValueError
        If `centres` is not a one dimensional array holding an even number of
        distinct positions, or if `exponent` is not a positive scalar.
'''
    return eri
```

### Step 3

Orbitals of the valence bond spaces

Goal
----
Orthogonalise the primitives symmetrically and return the coefficients of the bonding and antibonding orbital of each valence bond space, bonding orbital of space a in column 2a.

```python
def vbs_orbital_coefficients(overlap: np.ndarray, n_units: int) -> np.ndarray:
    '''Coefficients of the bond space orbitals in the primitive basis.

    Parameters
    ----------
    overlap : np.ndarray
        Overlap matrix of the primitives, shape (N, N) with N even.
    n_units : int
        Number of valence bond spaces, equal to half the number of primitives.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (N, N). Column 2a is the bonding orbital of bond space
        a and column 2a+1 is its antibonding partner.
    
    Raises
    ------
    ValueError
        If `overlap` is not a square matrix, if its dimension is not twice
        `n_units`, or if it is not positive definite.
'''
    return coefficients
```

### Step 4

Integral families of the pairing theory

Goal
----
Transform the one and two electron integrals into the bond space orbitals and return the one electron matrix together with the direct, exchange, pair-transfer and mean field families.

```python
def pair_integral_families(coefficients: np.ndarray, core: np.ndarray,
                           eri: np.ndarray) -> np.ndarray:
    '''One electron matrix and the direct, exchange, pair-transfer families.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).

    Returns
    -------
    families : np.ndarray
        Array of shape (5, N, N) holding, in order, the transformed one
        electron matrix, the direct family, the exchange family, the
        pair-transfer family and the mean field combination 2J - K.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, or if `core` and `eri` do not
        match its dimension.
'''
    return families
```

### Step 5

Stationary pair amplitudes

Goal
----
Solve the self consistent stationarity condition for the gap of every valence bond space with the orbitals held fixed.

```python
def optimal_pair_gaps(families: np.ndarray, n_units: int) -> np.ndarray:
    '''Gaps that make the reference energy stationary.

    Parameters
    ----------
    families : np.ndarray
        Integral families of shape (5, N, N) in the bond space basis.
    n_units : int
        Number of valence bond spaces, equal to N // 2.

    Returns
    -------
    gaps : np.ndarray
        Array of shape (n_units,) holding one stationary gap per bond space.
    
    Raises
    ------
    ValueError
        If `families` does not have shape (5, N, N), if N is not twice
        `n_units`, if the intrabond pair-transfer integral of any bond space is
        not positive, or if the self consistent iteration fails to converge.
'''
    return gaps
```

### Step 6

Energy of the pair reference

Goal
----
Return the electronic energy of the pair reference for a given set of gaps, nuclear repulsion excluded.

```python
def pp_reference_energy(families: np.ndarray, gaps: np.ndarray) -> float:
    '''Electronic energy of the pair reference.

    Parameters
    ----------
    families : np.ndarray
        Integral families of shape (5, N, N) in the bond space basis.
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    energy : float
        Electronic energy of the reference in hartree, nuclear repulsion
        excluded.
    
    Raises
    ------
    ValueError
        If `families` does not have shape (5, N, N), or if `gaps` is not a one
        dimensional array holding one entry per bond space.
'''
    return energy
```

### Step 7

Generalised Fock matrix of the reference

Goal
----
Return the generalised Fock matrix of the pair reference, which is not symmetric because the orbitals are frozen rather than optimised.

```python
def generalized_fock_matrix(coefficients: np.ndarray, core: np.ndarray,
                            eri: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    '''Generalised Fock matrix of the pair reference.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    fock : np.ndarray
        Array of shape (N, N). It is not symmetric in general.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core` and `eri` do not
        match its dimension, or if `gaps` does not hold one amplitude per bond
        space.
'''
    return fock
```

### Step 8

Second order contribution of the single excitations

Goal
----
Return the second order Epstein-Nesbet contribution of the swaps, the splits and the single electron transfers.

```python
def single_excitation_en2(coefficients: np.ndarray, core: np.ndarray,
                          eri: np.ndarray, gaps: np.ndarray,
                          fock: np.ndarray) -> float:
    '''Second order energy from swaps, splits and single electron transfers.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).
    fock : np.ndarray
        Generalised Fock matrix of the reference, shape (N, N), as returned by
        the preceding step. It is not symmetric in general.

    Returns
    -------
    energy : float
        Contribution of the single excitation family in hartree, negative.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core`, `eri` or `fock` do
        not match its dimension, if `gaps` does not hold one amplitude per bond
        space, or if any excitation energy is not strictly positive, meaning at
        or below 1e-6 hartree.
'''
    return energy
```

### Step 9

Second order contribution of the double excitations

Goal
----
Return the second order Epstein-Nesbet contribution of every double excitation, including both seniority four singlets of a double split.

```python
def double_excitation_en2(coefficients: np.ndarray, core: np.ndarray,
                          eri: np.ndarray, gaps: np.ndarray) -> float:
    '''Second order energy from the double excitation family.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    energy : float
        Contribution of the double excitation family in hartree, negative.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core` and `eri` do not
        match its dimension, if `gaps` does not hold one amplitude per bond
        space, or if any excitation energy is not strictly positive, meaning at
        or below 1e-6 hartree.
'''
    return energy
```

### Step 10

Second order contribution of the pair transfers

Goal
----
Return the second order Epstein-Nesbet contribution of the seniority zero, seniority two and seniority four pair transfers between bond spaces.

```python
def pair_transfer_en2(coefficients: np.ndarray, core: np.ndarray,
                      eri: np.ndarray, gaps: np.ndarray) -> float:
    '''Second order energy from transferring correlated pairs between spaces.

    Parameters
    ----------
    coefficients : np.ndarray
        Orbital coefficients in the primitive basis, shape (N, N).
    core : np.ndarray
        Core Hamiltonian in the primitive basis, shape (N, N).
    eri : np.ndarray
        Electron repulsion integrals in the primitive basis, shape (N,N,N,N).
    gaps : np.ndarray
        One gap per bond space, shape (N // 2,).

    Returns
    -------
    energy : float
        Contribution of the pair transfer family in hartree, negative.
    
    Raises
    ------
    ValueError
        If `coefficients` is not a square matrix, if `core` and `eri` do not
        match its dimension, if `gaps` does not hold one amplitude per bond
        space, or if any excitation energy is not strictly positive, meaning at
        or below 1e-6 hartree.
'''
    return energy
```

### Step 11

Corrected total energy of the model

Goal
----
Run the whole pipeline and return the reference energy plus the complete second order correction plus nuclear repulsion.

```python
def total_pp_en2_energy(centres: np.ndarray = (0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6),
                        exponent: float = 0.30, charges: np.ndarray = (1.0,) * 8,
                        n_units: int = 4) -> float:
    '''Corrected total energy of the pair reference.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr, an even number of them.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.
    charges : np.ndarray
        Nuclear charge of each centre.
    n_units : int
        Number of valence bond spaces, half the number of centres.

    Returns
    -------
    energy : float
        Reference energy plus the complete second order correction plus
        nuclear repulsion, in hartree.
    
    Raises
    ------
    ValueError
        If `centres` does not hold an even number of distinct positions, if
        `exponent` is not positive, if `charges` does not hold one entry per
        centre, or if `n_units` is not half the number of centres.
'''
    return energy
```
