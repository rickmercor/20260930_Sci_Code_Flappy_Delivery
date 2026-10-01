# Chemistry-Quantum_Chemistry-44

## Background

## Excited states from the polarisation propagator

Most accurate wavefunction methods for electronic excitation energies either diagonalise the
Hamiltonian in a space of excited configurations or extract poles from a response function. The
algebraic-diagrammatic construction (ADC) belongs to the second family. It expands the polarisation
propagator in perturbation theory and recasts the result as a Hermitian eigenvalue problem, so that
excitation energies appear as eigenvalues of a secular matrix and transition properties follow from
its eigenvectors. In the intermediate-state formulation the rows and columns of that matrix are
correlated excited states obtained by applying excitation operators to the correlated ground state
and orthonormalising them class by class, which gives the scheme a transparent structure: single
excitations, double excitations and the coupling between them, each treated through a chosen order.
The resulting methods are size-consistent, and the second- and third-order members are widely used
because their cost grows only as the fifth and sixth power of system size.

## Why the partitioning matters

A perturbation expansion begins by splitting the Hamiltonian into a solvable zeroth-order part and a
perturbation. The standard choice takes the Fock operator as the zeroth-order part, which leads to
Møller-Plesset theory and to the familiar energy denominators built from differences of orbital
energies. The same denominators appear in every ADC matrix element beyond first order. When the gap
between occupied and virtual levels becomes small, as in stretched bonds, diradicals or some
transition-metal complexes, those denominators blow up and the method fails in a way that no amount
of higher-order correction can repair. Brillouin-Wigner perturbation theory avoids the divergence by
placing the exact energy in the denominators, but in its original form it gives up size-consistency.

Recent work has shown that a regular theory can keep size-consistency if the correction is moved into
the zeroth-order Hamiltonian instead. A correlation-dependent, one-particle level-shift operator is
added to the Fock operator, and the size of the shift on each level follows how strongly that level
takes part in the correlation. The orbitals that diagonalise the shifted operator are called dressed
orbitals. The strength of the shift in the occupied and virtual spaces can be tuned separately, and
its sign decides whether the gap is widened, which regularises the expansion, or narrowed. Carrying such a
repartitioning into an excited-state method raises questions of its own: which parts of the
excited-state matrix it changes, how the ground-state amplitudes that feed that matrix are modified,
and whether the resulting excitation energies improve on the conventional scheme.

## Small hydrogen clusters as a proving ground

Clusters of a few hydrogen atoms are the standard laboratory for such questions. Their correlation
strength can be dialled continuously by changing the geometry, from nearly independent H2 units to
strongly coupled rings and chains, and with small Gaussian basis sets they remain small enough for
exact diagonalisation. Every step of a correlated excited-state calculation can therefore be checked
against full configuration interaction, and systematic differences between partitionings show up
clearly rather than being hidden in basis-set or geometry effects.

## Problem

Third-order algebraic-diagrammatic construction (ADC) schemes for the polarisation propagator are
usually built on the Møller-Plesset partitioning of the Hamiltonian. That partitioning is
size-consistent but not regular: its correlation energy, and with it every secular-matrix element
beyond first order, diverges as orbital-energy gaps close. A size-consistent Brillouin-Wigner-type
repartitioning removes the divergence. A one-particle operator W built from first-order doubles
amplitudes is added to the Fock operator, the orbitals are rotated and shifted until f + W is
diagonal and consistent with the amplitudes it is built from, and the ordinary ADC hierarchy is then
run on that new zeroth-order Hamiltonian. Your task is to carry this repartitioned third-order
scheme through for a small, moderately correlated hydrogen cluster and to report its lowest singlet
excitation energy.

Work in spin orbitals built from the canonical closed-shell restricted Hartree-Fock (RHF) orbitals,
with antisymmetrised integrals <pq||rs> = <pq|rs> - <pq|sr>, occupied indices i, j, k, virtual
indices a, b, c, and first-order doubles t_ij^ab = <ij||ab> / (e_a + e_b - e_i - e_j), where the e
are the current zeroth-order orbital energies. W has no occupied-virtual block; its occupied block
is W_ij = -(A0/8) sum over k, a, b of (<ik||ab> t_jk^ab + <jk||ab> t_ik^ab), and its virtual block
is W_ab = -(B0/8) sum over i, j, c of (<ac||ij> t_ij^bc + <bc||ij> t_ij^ac). Starting from the
canonical orbitals and energies, rebuild t and W in the current orbitals, diagonalise f + W
separately within the occupied and within the virtual block (f is the Hartree-Fock Fock operator and
is never itself modified), adopt the eigenvalues as the new zeroth-order energies and the
eigenvectors as the new orbitals, and repeat until the second-order energy E(2) = -(1/4) sum of
<ij||ab> t_ij^ab is stationary to 1e-12 hartree and f + W is diagonal. In the converged dressed
orbitals H0 = f + W is diagonal while f is not, and everything else in the Hamiltonian, -W included,
is the perturbation. Build the strict third-order ADC secular matrix for this partitioning in its
Hermitian intermediate-state form (precursor states orthogonalised to the ground state and, for the
doubles, to the singles, then orthonormalised symmetrically in the Löwdin sense within each
excitation class), with the singles-singles block through third order, the singles-doubles coupling
through second order and the doubles-doubles block through first order, taking every ground-state
amplitude and density that enters from Rayleigh-Schrödinger perturbation theory for this H0 with no
further approximation. The excitation energies are its eigenvalues.

The cluster has four hydrogen nuclei at (0.00, 0.00, 0.00), (1.60, 0.00, 0.00), (1.90, 2.40, 0.20)
and (0.20, 2.50, 0.00) bohr and four electrons. Use the 6-31G basis on every atom: an inner s
function contracted from exponents 18.7311370, 2.8253937 and 0.6401217 with coefficients 0.03349460,
0.23472695 and 0.81375733 (multiplying normalised primitives, the contracted function then
normalised), plus an outer s function with exponent 0.1612778. Take the lowest closed-shell RHF
solution, correlate all electrons, and set A0 = -0.75 and B0 = 0.5. Among states with Ms = 0, those
symmetric under exchange of the alpha and beta spin labels are singlets (quintets also fall there,
but they have no single-excitation part) and the antisymmetric ones are triplets. A state is
predominantly a single excitation when the singles part of its normalised eigenvector carries more
than half of the squared norm. Use 1 hartree = 27.211386245988 eV.

Report as the final answer the excitation energy, in eV, of the lowest predominantly
single-excitation singlet state, to at least four decimal places. In the reasoning give the scalars
that determine it, each as a single value: the RHF total energy, the two dressed occupied orbital
energies, E(2) and the third-order ground-state energy E(3) of this partitioning, all in hartree to
at least six decimal places; the two lowest predominantly single-excitation singlet and the two
lowest predominantly single-excitation triplet excitation energies, in eV to at least four decimal
places; and the lowest singlet root that is not predominantly a single excitation, with its
single-excitation weight. State also how W enters the zeroth- plus first-order singles-singles
block, the values of A0 and B0 recommended for the second-order member of the same family, and the
conditions on A0 and B0 under which the second-order and the third-order ground-state energies of a
two-electron system dissociated in a minimal basis become exact.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_one_electron_integrals

Goal
----
Step 01 - Overlap and core-Hamiltonian matrices of a hydrogen cluster in a contracted s-type Gaussian basis.

```python
import numpy as np
import numpy.typing as npt


def one_electron_integrals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list) -> np.ndarray:
    '''Overlap and core-Hamiltonian matrices over contracted s-type Gaussians.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.

    Returns
    -------
    one_electron : numpy.ndarray
        Array of shape (2, n, n) with n = n_atoms * len(exponents). Element
        [0] is the overlap matrix and element [1] the core Hamiltonian, kinetic
        energy plus nuclear attraction, in hartree. Basis functions are ordered
        atom by atom and, within an atom, shell by shell.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        exponents and coefficients are empty or differ in length, the
        exponent and coefficient arrays of a shell differ in length, an
        exponent is not positive and finite, or the coefficients of a shell
        are not finite or are all zero.
    '''
    return one_electron
```

### Step 2

02_electron_repulsion_integrals

Goal
----
Step 02 - Two-electron repulsion integrals over the same contracted s-type Gaussians.

```python
import numpy as np
import numpy.typing as npt


def electron_repulsion_integrals(coords: npt.ArrayLike, exponents: list, coefficients: list) -> np.ndarray:
    '''Two-electron repulsion integrals over contracted s-type Gaussians.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.

    Returns
    -------
    eri : numpy.ndarray
        Array of shape (n, n, n, n) holding (pq|rs) in hartree, chemists'
        notation, with the basis ordered atom by atom and shell by shell.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, exponents and coefficients are empty or differ in length,
        the exponent and coefficient arrays of a shell differ in length, an
        exponent is not positive and finite, or the coefficients of a shell
        are not finite or are all zero.
    '''
    return eri
```

### Step 3

03_rhf_canonical_orbitals

Goal
----
Step 03 - Closed-shell restricted Hartree-Fock reference and its canonical orbitals.

```python
import numpy as np
import numpy.typing as npt


def rhf_canonical_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int) -> np.ndarray:
    '''Canonical orbitals of the closed-shell restricted Hartree-Fock determinant.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.

    Returns
    -------
    orbitals : numpy.ndarray
        Array of shape (n + 1, n). Row 0 holds the canonical orbital energies
        in hartree in ascending order. Rows 1 to n hold the coefficient matrix
        C, whose column p expands orbital p in the basis functions, each
        column sign-fixed so that its largest-magnitude entry is positive (the
        lowest index decides between entries equal to a relative 1e-8).

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, or the SCF iterations do not converge within 1000
        cycles.
    '''
    return orbitals
```

### Step 4

04_dressed_orbitals

Goal
----
Step 04 - Self-consistent orbital dressing by the two-parameter regularisation operator.

```python
import numpy as np
import numpy.typing as npt


def dressed_orbitals(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float) -> np.ndarray:
    '''Self-consistently dressed orbitals of the two-parameter regularised partitioning.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.
    A0 : float
        Regularisation parameter of the occupied block of W.
    B0 : float
        Regularisation parameter of the virtual block of W.

    Returns
    -------
    dressed : numpy.ndarray
        Array of shape (n + 1, n). Row 0 holds the converged dressed spatial
        orbital energies in hartree, occupied ones first and each block in
        ascending order (every occupied level lies below every virtual one).
        Rows 1 to n hold the dressed orbitals as columns expanded in the basis
        functions, each sign-fixed so that its largest-magnitude entry is
        positive (the lowest index decides between entries equal to a
        relative 1e-8).

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, the SCF iterations do not converge within 1000 cycles,
        A0 or B0 is not finite, the dressed occupied and virtual levels
        overlap during the iterations, or the dressing does not converge
        within 2000 cycles.
    '''
    return dressed
```

### Step 5

05_second_order_ground_state

Goal
----
Step 05 - Second-order ground-state amplitudes and energies for a non-canonical zeroth-order Hamiltonian.

```python
import numpy as np
import numpy.typing as npt


def second_order_ground_state(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    '''Second-order Rayleigh-Schrödinger ground state for a diagonal H0 and a non-diagonal Fock matrix.

    Parameters
    ----------
    orbital_energies : array_like
        Zeroth-order one-particle energies e_p of the m spin orbitals in
        hartree, occupied spin orbitals first. H0 = sum_p e_p a+_p a_p.
    fock : array_like
        Fock matrix f_pq of the reference determinant in the same spin-orbital
        basis, shape (m, m), symmetric, with a vanishing occupied-virtual block.
        It need not be diagonal.
    eri : array_like
        Antisymmetrised two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> in
        physicists' notation over real orbitals, shape (m, m, m, m).
    n_occ : int
        Number of occupied spin orbitals; they are the first n_occ.

    Returns
    -------
    ground_state : numpy.ndarray
        One-dimensional array of length 2 + n_occ**2 * n_vir**2 + n_occ * n_vir,
        with n_vir = m - n_occ. Element 0 is E(2) and element 1 is E(3), in
        hartree. Then follow the second-order doubles t_ij^ab(2) for all
        occupied i, j and virtual a, b (virtual indices counted from 0 within
        the virtual block) in C order of (i, j, a, b), and finally the
        second-order singles t_i^a(2) in C order of (i, a). Amplitudes are
        coefficients of a+_a a+_b a_i a_j Phi_0 (with the factor 1/4 of the
        expansion) and of a+_a a_i Phi_0 in the second-order correction
        Psi(2) to the intermediate-normalised wavefunction.

    Raises
    ------
    ValueError
        If orbital_energies is not one-dimensional with at least two entries,
        fock or eri does not have the matching shape, any input is not finite,
        n_occ is not an integer strictly between 0 and m, fock is not
        symmetric or has a non-zero occupied-virtual block (tolerance 1e-10),
        eri is not antisymmetric within each index pair or not symmetric
        under exchange of the two pairs (tolerance 1e-10), or some virtual
        orbital energy does not lie above every occupied one.
    '''
    return ground_state
```

### Step 6

06_adc3_singles_block

Goal
----
Step 06 - Singles-singles block of the third-order intermediate-state secular matrix.

```python
import numpy as np
import numpy.typing as npt


def adc3_singles_block(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    '''Singles-singles block of the strict third-order secular matrix.

    Parameters
    ----------
    orbital_energies : array_like
        Zeroth-order one-particle energies e_p of the m spin orbitals in
        hartree, occupied spin orbitals first. H0 = sum_p e_p a+_p a_p.
    fock : array_like
        Fock matrix f_pq of the reference determinant in the same spin-orbital
        basis, shape (m, m), symmetric, with a vanishing occupied-virtual block.
        It need not be diagonal.
    eri : array_like
        Antisymmetrised two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> in
        physicists' notation over real orbitals, shape (m, m, m, m).
    n_occ : int
        Number of occupied spin orbitals; they are the first n_occ.

    Returns
    -------
    singles_block : numpy.ndarray
        Symmetric array of shape (n_occ * n_vir, n_occ * n_vir) in hartree,
        with n_vir = m - n_occ. Row and column i * n_vir + a belong to the
        single excitation from occupied spin orbital i to virtual spin orbital
        a (virtual indices counted from 0 within the virtual block); every
        pair is included whatever the spins.

    Raises
    ------
    ValueError
        If orbital_energies is not one-dimensional with at least two entries,
        fock or eri does not have the matching shape, any input is not finite,
        n_occ is not an integer strictly between 0 and m, fock is not
        symmetric or has a non-zero occupied-virtual block (tolerance 1e-10),
        eri is not antisymmetric within each index pair or not symmetric
        under exchange of the two pairs (tolerance 1e-10), or some virtual
        orbital energy does not lie above every occupied one.
    '''
    return singles_block
```

### Step 7

07_adc3_spin_sector_spectrum

Goal
----
Step 07 - Complete third-order secular matrix for restricted orbitals, split by spin symmetry and diagonalised.

```python
import numpy as np
import numpy.typing as npt


def adc3_spin_sector_spectrum(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int, flip_parity: int) -> np.ndarray:
    '''Eigenvalues and single-excitation weights of one spin sector of the strict third-order matrix.

    Parameters
    ----------
    orbital_energies : array_like
        Zeroth-order one-particle energies e_p of the m spin orbitals in
        hartree, occupied spin orbitals first. H0 = sum_p e_p a+_p a_p.
        Spin orbitals are ordered p = 2k + s (s = 0 alpha, s = 1 beta).
    fock : array_like
        Fock matrix f_pq of the reference determinant in the same spin-orbital
        basis, shape (m, m), symmetric, with a vanishing occupied-virtual block.
        It need not be diagonal.
    eri : array_like
        Antisymmetrised two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> in
        physicists' notation over real orbitals, shape (m, m, m, m).
    n_occ : int
        Number of occupied spin orbitals; they are the first n_occ.
    flip_parity : int
        +1 for the sector symmetric under exchange of alpha and beta labels
        (singlets), -1 for the antisymmetric sector (triplets).

    Returns
    -------
    spectrum : numpy.ndarray
        Array of shape (2, m_sector), m_sector being the number of Ms = 0
        single and double excitations in the requested sector. Row 0 holds
        the excitation energies in hartree in ascending order, row 1 the
        single-excitation weight of each eigenvector.

    Raises
    ------
    ValueError
        If orbital_energies is not one-dimensional with at least two entries,
        fock or eri does not have the matching shape, any input is not finite,
        n_occ is not an integer strictly between 0 and m, fock is not
        symmetric or has a non-zero occupied-virtual block (tolerance 1e-10),
        eri is not antisymmetric within each index pair or not symmetric
        under exchange of the two pairs (tolerance 1e-10), or some virtual
        orbital energy does not lie above every occupied one. Also if
        flip_parity is not +1 or -1, the number of spin orbitals or n_occ
        is odd, alpha and beta orbital energies differ, fock is not
        spin-diagonal with identical alpha and beta blocks, or eri does
        not conserve spin or changes under exchange of alpha and beta
        labels (tolerance 1e-10).
    '''
    return spectrum
```

### Step 8

08_spin_resolved_excitations

Goal
----
Step 08 - Predominantly single-excitation singlet and triplet states of a molecule.

```python
import numpy as np
import numpy.typing as npt


def spin_resolved_excitations(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float, n_states: int) -> np.ndarray:
    '''Lowest predominantly single-excitation singlet and triplet states at the regularised third order.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.
    A0 : float
        Regularisation parameter of the occupied block of W.
    B0 : float
        Regularisation parameter of the virtual block of W.
    n_states : int
        Number of states wanted in each spin sector, a positive integer.

    Returns
    -------
    states : numpy.ndarray
        Array of shape (2, n_states) in hartree. Row 0 holds the excitation
        energies of the n_states lowest singlets whose single-excitation
        weight exceeds 0.5, row 1 the same for triplets, each in ascending
        order.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, the SCF iterations do not converge within 1000 cycles,
        A0 or B0 is not finite, the dressed occupied and virtual levels
        overlap during the iterations, or the dressing does not converge
        within 2000 cycles. Also if n_states is not a positive integer or
        either sector holds fewer than n_states predominantly
        single-excitation states.
    '''
    return states
```

### Step 9

09_lowest_singlet_excitation

Goal
----
Step 09 - Orchestrator: lowest predominantly single-excitation singlet excitation energy in eV.

```python
import numpy as np
import numpy.typing as npt


def lowest_singlet_excitation(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float) -> float:
    '''Lowest predominantly single-excitation singlet excitation energy in electronvolts.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.
    A0 : float
        Regularisation parameter of the occupied block of W.
    B0 : float
        Regularisation parameter of the virtual block of W.

    Returns
    -------
    excitation_ev : float
        Excitation energy in eV of the lowest singlet whose single-excitation
        weight exceeds 0.5, with 1 hartree = 27.211386245988 eV.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, the SCF iterations do not converge within 1000 cycles,
        A0 or B0 is not finite, the dressed occupied and virtual levels
        overlap during the iterations, or the dressing does not converge
        within 2000 cycles. Also if either spin sector has no state whose
        single-excitation weight exceeds 0.5.
    '''
    return excitation_ev
```
