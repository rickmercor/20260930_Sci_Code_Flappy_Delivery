# Chemistry-Quantum_Chemistry-36

## Background

### Excitonic coupling

When two chromophores sit close enough to interact but far enough apart that their electron clouds do not overlap, an excitation on one of them can move to the other. The matrix element that drives this, the excitonic coupling, is the Coulomb interaction between the transition density of the molecule that gives up its excitation and the transition density of the molecule that receives it. At large distance it reduces to the familiar interaction of two transition dipoles, which underlies Forster energy transfer; at shorter distance the full shape of the transition densities matters. In a symmetric dimer the coupling also shows up directly in the spectrum: the two degenerate locally excited states mix and split, and to first order the splitting is twice the coupling.

This gives two routes to the same number. The fragment route computes the transition density of each isolated molecule and couples them through the Coulomb operator. The supersystem route computes the excited states of the dimer as a whole and reads the coupling from the splitting. An exact theory must give the same answer both ways, because the exact states of two non-interacting molecules are products of the states of each.

### Coupled-cluster excited states

Coupled-cluster theory writes the ground state as an exponential of excitation operators acting on a reference determinant, and truncating that operator at single and double excitations gives CCSD. The ground-state energy of CCSD is size extensive: for two non-interacting molecules it is exactly the sum of the two molecular energies, because the cluster operator of the dimer is the sum of the two molecular cluster operators. Excited states follow from the equation-of-motion (EOM) formulation, in which the similarity-transformed Hamiltonian is represented in the space of the reference and its singly and doubly excited determinants and then diagonalised. That matrix is not symmetric, so each state has a right and a left eigenvector, and transition properties are asymmetric products of the two, built with the Lambda amplitudes that make the CC energy functional stationary. Excitation energies of a molecule that is locally excited inside a larger non-interacting system are size intensive in EOM-CCSD.

### Why the two routes can disagree

The subtlety lies in the truncation. A doubly excited determinant of the dimer is a double excitation relative to the dimer reference, and it can be a double excitation on A, a double excitation on B, or a single excitation on each molecule. The product of a single excitation on A with a double excitation on B is a triple excitation of the dimer and is not in the space at all. Quantities that are purely local, like excitation energies, do not notice this. Quantities that couple an excitation on one molecule to something happening on the other can, because the dimer's description of the correlated partner molecule is truncated in a way that the separate molecular calculations are not. Whether the excitonic coupling suffers from this, and by how much, is a quantitative question that depends on the level of truncation and on how strongly correlated the chromophore is.

### The model

A Pariser-Parr-Pople pi-electron Hamiltonian makes the question exactly computable. It keeps one orbital per conjugated atom, neglects differential overlap between different atoms, and describes the electron repulsion with a distance-dependent interpolation between the one-centre repulsion and the bare Coulomb law. For a small chromophore the whole space of determinants of a dimer is small enough that CCSD, EOM-CCSD and full configuration interaction can all be carried out without further approximation, so any disagreement between the fragment and supersystem couplings is a property of the truncated coupled-cluster model itself and not of a basis set, an integral approximation or a numerical threshold. A dipolar chromophore with a nitrogen donor is used, stacked antiparallel so that the two molecules are related by symmetry and the dimer carries no static dipole.

## Problem

An enamine-type pi chromophore C1=C2-N3 is described by the Pariser-Parr-Pople model, and two copies of it are stacked face to face with their chain axes pointing in opposite directions. Coupled-cluster theory truncated at double excitations can give the excitonic coupling of such a pair either from a calculation on the whole dimer or from transition densities of the separate molecules, and the task is to find how far the two routes disagree.

Place the atoms of one molecule on the x axis with bond lengths 1.34 angstrom (C1-C2) and 1.40 angstrom (C2-N3), resonance integrals -2.4 eV and -2.0 eV, site energies 0, 0 and -3.0 eV, one-centre repulsions U of 11.13, 11.13 and 16.76 eV and core charges Z of 1, 1 and 2, so that each molecule holds four pi electrons in two doubly occupied orbitals. Electrons on sites mu and nu, in the same molecule or in different ones, repel through gamma = 14.397 / sqrt(r^2 + a^2) eV with r in angstrom and a = 28.794 / (U_mu + U_nu), and the Hamiltonian is H = sum h_mu,nu a+_mu,sigma a_nu,sigma + sum_mu U_mu n_mu,up n_mu,down + sum_{mu<nu} gamma_mu,nu (n_mu - Z_mu)(n_nu - Z_nu), constant core repulsion included and with no resonance integral between the molecules. Molecule B is molecule A turned by 180 degrees about the axis parallel to y through the midpoint of A and lifted along z, so atom k of B sits at x = 2 x_c - x_k and z equal to the stacking distance, where x_c is the mean x coordinate of the atoms of A; use a stacking distance of 7.0 angstrom.

Treat each molecule with closed-shell RHF, CCSD and EOM-CCSD in the full space of singly and doubly excited determinants, and take its lowest singlet excited state. The fragment coupling V^c is the Coulomb contraction sum_{mu,nu} gamma_AB[mu,nu] rho^10_mu rho^01_nu of the downward EOM-CCSD transition density of A, <L| exp(-T) n_mu exp(T) |Phi_0>, with the upward one of B, <Phi_0| (1 + Lambda) exp(-T) n_nu exp(T) R |Phi_0>, where L and R are the left and right eigenvectors of that state normalised to L.R = 1 and B carries the same site transition densities as A. The supersystem coupling V^s comes from CCSD and EOM-CCSD on the eight-electron dimer, with its reference the product of the two molecular RHF determinants and with the intermolecular Coulomb terms multiplied by a factor lambda: it is half of the splitting of the two dimer singlet states that correlate with the chosen monomer state, taken to first order, that is the limit of that splitting divided by 2 lambda as lambda goes to zero. Report as your final answer the separability error Delta = 100 (|V^c| - |V^s|) / |V^s| in percent, with both couplings evaluated in that exact first-order sense and not at a finite lambda.

The scalars that determine the final number, and that your reasoning should set out alongside it, are the repulsion gamma in eV between C1 and N3 of the same molecule and between C1 of A and C1 of B at 7.0 angstrom; the gap between the lowest virtual and the highest occupied RHF orbital energy of the monomer; its CCSD total pi energy and the EOM-CCSD excitation energy of the chosen state; the CCSD total pi energy of the interacting dimer at 7.0 angstrom on its own RHF reference; |V^c| and |V^s| in eV at 7.0 angstrom; |V^s| in eV at a stacking distance of 30.0 angstrom; and Delta at 7.0 angstrom and again at 30.0 angstrom. In the same reasoning, give the literature origin, by authors and year, of three ingredients: the neglect of differential overlap between different atoms in the model Hamiltonian, the evaluation of an excitonic coupling as the Coulomb interaction of complete molecular transition densities, and the analytic-derivative treatment in which the response of non-variational wavefunction parameters to a perturbation is carried by one perturbation-independent linear equation.

Output Format Requirements:
Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep `<reasoning>` short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

ppp_monomer_matrices

Goal
----
One-electron and electron-repulsion matrices of a linear Pariser-Parr-Pople pi chain.

```python
def ppp_monomer_matrices(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                         hubbard_u: "np.ndarray") -> "np.ndarray":
    '''PPP one-electron matrix and Ohno repulsion matrix of a linear chain.

    Parameters
    ----------
    bonds : numpy.ndarray
        One-dimensional array of the n_sites - 1 bond lengths along the chain, in
        angstrom; all must be positive.
    betas : numpy.ndarray
        Resonance integral of each bond, in eV, same length as bonds.
    alphas : numpy.ndarray
        Site energy of each atom, in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsion of each atom, in eV, length n_sites; all must be positive.

    Returns
    -------
    matrices : numpy.ndarray
        Real array of shape (2, n_sites, n_sites): index 0 is the one-electron
        matrix h and index 1 is the Ohno repulsion matrix gamma.

    Raises
    ------
    ValueError
        If bonds is empty or has a non-positive entry, if betas does not have the
        length of bonds, if alphas or hubbard_u does not have length n_sites, or if
        any one-centre repulsion is not positive.
    '''
    return matrices
```

### Step 2

antiparallel_stack_interactions

Goal
----
Intermolecular Ohno repulsion matrix between two identical chains stacked face to face in antiparallel orientation.

```python
def antiparallel_stack_interactions(bonds: "np.ndarray", hubbard_u: "np.ndarray",
                                    separation: float) -> "np.ndarray":
    '''Ohno repulsion between the atoms of A and of its antiparallel stacked copy B.

    Parameters
    ----------
    bonds : numpy.ndarray
        One-dimensional array of the n_sites - 1 bond lengths of the chain, in
        angstrom; all must be positive.
    hubbard_u : numpy.ndarray
        One-centre repulsion of each atom, in eV, length n_sites; all must be positive.
    separation : float
        Distance between the planes of the two molecules along z, in angstrom; must
        be positive.

    Returns
    -------
    gamma_ab : numpy.ndarray
        Real array of shape (n_sites, n_sites) whose element [mu, nu] is the Ohno
        repulsion between atom mu of molecule A and atom nu of molecule B.

    Raises
    ------
    ValueError
        If bonds is empty or has a non-positive entry, if hubbard_u does not have
        length n_sites or has a non-positive entry, or if separation is not a
        positive finite number.
    '''
    return gamma_ab
```

### Step 3

ppp_rhf_orbitals

Goal
----
Closed-shell restricted Hartree-Fock orbitals of a PPP pi system with atomic core charges.

```python
def ppp_rhf_orbitals(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                     n_occ: int) -> "np.ndarray":
    '''Converged closed-shell RHF orbital energies and phase-fixed orbital coefficients.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV, whose
        diagonal holds the one-centre repulsions.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.

    Returns
    -------
    orbitals : numpy.ndarray
        Real array of shape (n_sites + 1, n_sites). Row 0 is the list of orbital
        energies in increasing order; rows 1 to n_sites form the coefficient matrix
        C with orbital k in column k, its largest-magnitude component positive.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, or if n_occ is not an integer
        with 1 <= n_occ < n_sites.
    '''
    return orbitals
```

### Step 4

ccsd_energy

Goal
----
Coupled-cluster singles and doubles ground-state energy of a closed-shell PPP pi system.

```python
def ccsd_energy(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray", n_occ: int) -> float:
    '''CCSD ground-state total pi energy on the RHF reference.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.

    Returns
    -------
    energy : float
        The CCSD total energy <Phi_0| exp(-T) H exp(T) |Phi_0> in eV, with H the PPP
        operator including the core-core repulsion.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, or if n_occ is not an integer
        with 1 <= n_occ < n_sites.
    '''
    return energy
```

### Step 5

eom_ccsd_singlet_energies

Goal
----
Equation-of-motion CCSD excitation energies of the lowest singlet states of a closed-shell PPP pi system.

```python
def eom_ccsd_singlet_energies(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                              n_occ: int, n_roots: int) -> "np.ndarray":
    '''Lowest singlet EOM-CCSD excitation energies.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.
    n_roots : int
        Number of singlet excited states wanted; must be at least 1 and no larger
        than the number of singlet excited states in the singles and doubles space.

    Returns
    -------
    energies : numpy.ndarray
        Real array of shape (n_roots,) with the EOM-CCSD singlet excitation
        energies in eV in increasing order.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, if n_occ is not an integer with
        1 <= n_occ < n_sites, or if n_roots is not a positive integer or exceeds the
        number of singlet excited states available.
    '''
    return energies
```

### Step 6

eom_transition_density_product

Goal
----
Product of the two EOM-CCSD site transition densities between the ground state and one singlet excited state.

```python
def eom_transition_density_product(h: "np.ndarray", gamma: "np.ndarray", core_charges: "np.ndarray",
                                   n_occ: int, root: int) -> "np.ndarray":
    '''Outer product of the EOM-CCSD down and up site transition densities.

    Parameters
    ----------
    h : numpy.ndarray
        Symmetric one-electron matrix of shape (n_sites, n_sites), in eV.
    gamma : numpy.ndarray
        Symmetric electron-repulsion matrix of shape (n_sites, n_sites), in eV.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals; must satisfy 1 <= n_occ < n_sites.
    root : int
        Index, counting from zero, of the singlet excited state in increasing order
        of EOM-CCSD excitation energy.

    Returns
    -------
    product : numpy.ndarray
        Real array of shape (n_sites, n_sites) whose element [mu, nu] is
        rho^{10}_mu * rho^{01}_nu for the selected state.

    Raises
    ------
    ValueError
        If h or gamma is not a square symmetric matrix of the same size, if
        core_charges does not have length n_sites, if n_occ is not an integer with
        1 <= n_occ < n_sites, or if root is negative or not smaller than the number
        of singlet excited states available.
    '''
    return product
```

### Step 7

fragment_excitonic_coupling

Goal
----
Excitonic coupling of the antiparallel stacked homodimer assembled from the EOM-CCSD transition densities of the isolated molecules.

```python
def fragment_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                root: int, separation: float) -> float:
    '''Magnitude of the excitonic coupling built from monomer EOM-CCSD transition densities.

    Parameters
    ----------
    bonds : numpy.ndarray
        Bond lengths of the chain in angstrom, length n_sites - 1.
    betas : numpy.ndarray
        Resonance integral of each bond in eV.
    alphas : numpy.ndarray
        Site energies in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsions in eV, length n_sites.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals of one molecule.
    root : int
        Index, counting from zero, of the monomer singlet excited state.
    separation : float
        Stacking distance between the molecular planes in angstrom.

    Returns
    -------
    coupling : float
        |V^c| in eV.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the monomer matrices, the
        antiparallel stack geometry, the RHF, CCSD or EOM-CCSD steps it feeds, or
        if root does not select an available singlet excited state.
    '''
    return coupling
```

### Step 8

supersystem_excitonic_coupling

Goal
----
Excitonic coupling of the antiparallel stacked homodimer obtained from a CCSD and EOM-CCSD treatment of the dimer as a whole.

```python
def supersystem_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                   hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                   root: int, separation: float) -> float:
    '''Half of the first-order EOM-CCSD splitting of the dimer's locally excited singlet pair.

    Parameters
    ----------
    bonds : numpy.ndarray
        Bond lengths of the chain in angstrom, length n_sites - 1.
    betas : numpy.ndarray
        Resonance integral of each bond in eV.
    alphas : numpy.ndarray
        Site energies in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsions in eV, length n_sites.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals of one molecule.
    root : int
        Index, counting from zero, of the monomer singlet excited state whose dimer
        pair is split.
    separation : float
        Stacking distance between the molecular planes in angstrom.

    Returns
    -------
    coupling : float
        |V^s| in eV, the limit of half the splitting divided by lambda.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the monomer matrices, the
        antiparallel stack geometry, the RHF, CCSD or EOM-CCSD steps it feeds, or
        if root does not select an available singlet excited state.
    '''
    return coupling
```

### Step 9

separability_error_percent

Goal
----
Relative separability error of the CCSD excitonic coupling: the fragment coupling measured against the supersystem coupling.

```python
def separability_error_percent(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                               hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                               root: int, separation: float) -> float:
    '''Relative separability error of the CCSD excitonic coupling in percent.

    Parameters
    ----------
    bonds : numpy.ndarray
        Bond lengths of the chain in angstrom, length n_sites - 1.
    betas : numpy.ndarray
        Resonance integral of each bond in eV.
    alphas : numpy.ndarray
        Site energies in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsions in eV, length n_sites.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals of one molecule.
    root : int
        Index, counting from zero, of the monomer singlet excited state.
    separation : float
        Stacking distance between the molecular planes in angstrom.

    Returns
    -------
    delta : float
        100 * (|V^c| - |V^s|) / |V^s|.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the steps it feeds, or if the
        supersystem coupling vanishes so that the relative error is undefined.
    '''
    return delta
```
