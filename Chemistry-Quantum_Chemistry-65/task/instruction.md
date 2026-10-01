# Chemistry-Quantum_Chemistry-65

## Background

Photoelectron spectroscopy measures the energy needed to remove an electron from a molecule, and the lowest of these energies, the first ionization energy, controls charge injection, redox behaviour and the level alignment of organic semiconductors. Many-body Green's function theory describes ionization through quasiparticles: the one-electron states of a reference calculation are dressed by the dynamical screening of the other electrons, which moves their energies and shifts part of their spectral weight into satellite features. The most widely used member of this family evaluates the self-energy once, from the Green's function and the random-phase screened interaction of a mean-field reference, and is known to depend on that reference; starting from Hartree-Fock orbitals it usually overestimates the ionization energies of small molecules.

Coupled-cluster theory approaches the same excitations from the wave-function side, through exponential excitation operators and equation-of-motion treatments of charged states. The two families have long been developed separately, and relating their working equations clarifies which correlation effects each contains and suggests ways to add missing effects, such as vertex corrections or self-consistency, without leaving a sum-over-states description of the self-energy.

Conjugated hydrocarbons are a classic proving ground for such methods. In the pi-electron approximation each carbon atom contributes one orbital and one electron, and semi-empirical Hamiltonians with interpolated Coulomb interactions capture the main features of polyene spectra at a cost that still allows exact diagonalization of the model for short chains. Exact model results then serve as a clean reference for approximate many-body methods, and the strength of the electron-electron interaction can be varied smoothly to explore how the accuracy of each approximation changes between weak and strong correlation.

## Problem

One-shot GW on Hartree-Fock orbitals (G0W0@HF) is a standard route to molecular ionization energies, but it tends to overestimate them. A recent reformulation shows that G0W0 is exactly the equation-of-motion treatment of an electron-boson Hamiltonian transformed twice, first with direct-ring coupled-cluster doubles amplitudes and then with the de-excitation amplitudes of extended coupled-cluster theory. Within the same framework the linearized GW one-body density matrix follows from perturbation theory and can be fed back as a static correction to the Fock matrix, which lowers the ionization energies. Because that correction always pushes the ionization energy down, it should help where G0W0@HF is too high and overshoot once correlation makes G0W0@HF too low. The task is to locate this crossover for a pi-electron model of all-trans octatetraene against its exact solution: the input is a Pariser-Parr-Pople Hamiltonian whose electron interaction is scaled by a factor lambda, and the output is the break-even scale.

Octatetraene is modelled by eight carbon sites on a planar zigzag chain with all C-C-C angles equal to 120 degrees, one pi electron per site and zero differential overlap. Along the chain the first, third, fifth and seventh bonds are 1.35 Angstrom long with hopping -2.60 eV, the other bonds are 1.46 Angstrom long with hopping -2.20 eV, and non-bonded sites have no hopping. The repulsion between sites p and q at distance r_pq (in Angstrom) is gamma_pq = lambda U / sqrt(1 + (U r_pq / 14.397)^2) with U = 11.13 eV, so gamma_pp = lambda U. The same function carries the attraction of every neutral carbon core, giving the site energy -W - sum over q != p of gamma_pq with W = 11.16 eV, so lambda scales repulsion and core attraction together. The eight electrons fill four doubly occupied restricted Hartree-Fock orbitals.

On the canonical Hartree-Fock orbitals, the screening is the full (not Tamm-Dancoff) direct random-phase approximation, without exchange, in the spin-restricted singlet particle-hole space. The plain ionization energy IP_G0W0 is minus the eigenvalue of the doubly transformed electron-boson effective Hamiltonian, whose eigenvalues equal those of the full-frequency, non-diagonal G0W0 supermatrix, for the state with the largest biorthogonal spectral weight on the highest occupied orbital. The corrected value IP_G0W0+gamma comes from the same effective Hamiltonian after its one-hole/one-particle Fock block is augmented by the static Coulomb-minus-exchange potential of the change in the per-spin density matrix. That change is the symmetrized linearized GW density matrix (the Hartree-Fock density plus the frequency integral of G0 Sigma_c G0 built from the Hartree-Fock Green's function and the G0W0 correlation self-energy), with its occupied-occupied, virtual-virtual and occupied-virtual blocks, minus the Hartree-Fock density matrix; the hole-boson and particle-boson blocks keep the Hartree-Fock orbital energies. The reference IP_FCI is the full configuration interaction energy of the cation ground state minus that of the neutral ground state of the same Hamiltonian. The break-even scale lambda_be is the value of lambda between 0.5 and 2.0 at which IP_G0W0 + IP_G0W0+gamma = 2 IP_FCI.

Give lambda_be to three decimal places as the final answer. In the reasoning, report at lambda = 1 the Hartree-Fock highest-occupied orbital energy, the lowest direct random-phase excitation energy, IP_G0W0 with its quasiparticle weight, IP_G0W0+gamma, IP_FCI, and the smallest occupied and largest virtual natural occupation numbers of the linearized GW density matrix per spin. Give the scales at which IP_G0W0 and IP_G0W0+gamma each equal IP_FCI, the three ionization energies at lambda_be, and lambda_be for butadiene and hexatriene built by the same rules. Finally, adopt the Pariser-Parr-Pople parameters of the density-matrix renormalization group study of the low-lying excited states of linear polyenes that transferred its on-site repulsion and mean hopping from a fit to benzene: use that U, and let the hopping magnitude vary linearly with bond length about that study's undistorted C-C bond length with its electron-phonon slope (shorter bonds hop more strongly). Keep W and the octatetraene geometry, and recompute lambda_be for octatetraene.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> focused: report every quantity requested above with its value and the relation used to obtain it, without a long derivation.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_ppp_polyene_hamiltonian

Goal
----
Build the one-electron core Hamiltonian and the scaled Ohno electron repulsion of a Pariser-Parr-Pople all-trans polyene.

```python
def ppp_polyene_hamiltonian(n_sites: int, scale: float, params: "np.ndarray") -> "np.ndarray":
    '''One-electron core Hamiltonian and scaled Ohno repulsion matrix of a PPP all-trans polyene, in eV.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms n, an even integer of at least 2. Sites are numbered 0 to n-1 along the chain and the
        bond between sites k and k+1 is short (double) for even k and long (single) for odd k.
    scale : float
        Positive factor lambda that multiplies the Ohno repulsion and, through it, the core attraction.
    params : np.ndarray
        Shape (6,), [W, U, t_short, t_long, r_short, r_long]: site valence-state ionization energy W (eV), on-site
        repulsion U (eV), hopping integrals of the short and long bonds (eV, negative for bonding) and the short and long
        bond lengths (Angstrom). All C-C-C angles are 120 degrees in a planar zigzag chain.

    Returns
    -------
    hamiltonian : np.ndarray
        Shape (2, n, n). Element [1] is the repulsion matrix gamma_pq = lambda U / sqrt(1 + (U r_pq / 14.397)^2), with
        r_pq the distance in Angstrom between sites p and q (gamma_pp = lambda U). Element [0] is the core Hamiltonian:
        h_pq equals t_short or t_long for bonded neighbours and zero for other pairs, and
        h_pp = -W - sum over q != p of gamma_pq.

    Raises
    ------
    ValueError
        If n_sites is odd or smaller than 2, or scale is not positive.
    '''
    return hamiltonian
```

### Step 2

02_restricted_hartree_fock

Goal
----
Solve the closed-shell restricted Hartree-Fock equations of a zero-differential-overlap pi Hamiltonian.

```python
def restricted_hartree_fock(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Canonical closed-shell Hartree-Fock orbital energies and orbitals of a zero-differential-overlap Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Shape (n, n), symmetric one-electron core Hamiltonian in the orthonormal site basis (eV).
    gamma : np.ndarray
        Shape (n, n), symmetric site repulsion matrix (eV); the only nonzero two-electron integrals are
        (pp|qq) = gamma_pq.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    result : np.ndarray
        Shape (n + 1, n). Row 0 holds the orbital energies in ascending order. Rows 1 to n hold the orbital coefficient
        matrix C, whose column k is orbital k in the site basis, normalized, and signed so that its first component
        (lowest site index) with magnitude above 1e-6 is positive. The Fock matrix is
        F = h + diag(gamma @ diag(D)) - D * gamma / 2 (elementwise product) with D = 2 C_occ C_occ^T, converged until
        no element of D changes by more than 1e-12 between iterations.

    Raises
    ------
    ValueError
        If h and gamma do not have the same square shape or n_occ is outside 1 <= n_occ < n.
    '''
    return result
```

### Step 3

03_ring_ecc_amplitudes

Goal
----
Compute the direct-ring extended coupled-cluster doubles amplitudes that block-diagonalize the quasi-boson Hamiltonian.

```python
def ring_ecc_amplitudes(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Right (t) and left (z) direct-ring extended coupled-cluster doubles amplitudes of a closed-shell reference.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied and every virtual energy exceeds every occupied energy.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    amplitudes : np.ndarray
        Shape (2, m, m) with m = n_occ * (n - n_occ). Element [0] is t and element [1] is z. The composite index of the
        spatial occupied-virtual pair (i, a) is i * (n - n_occ) + (a - n_occ). A and B are the spin-summed singlet
        direct-ring blocks, t is the symmetric root of B + A t + t A + t B t = 0 given by t = Y X^-1 from the
        positive-frequency direct random-phase eigenvectors, and z = t (1 - t t)^-1.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n or eri does not have shape (n, n, n, n).
    '''
    return amplitudes
```

### Step 4

04_ecc_quasiparticle_ionization

Goal
----
Obtain the highest-occupied quasiparticle ionization energy and its spectral weight from the equation-of-motion treatment of the doubly similarity-transformed electron-boson Hamiltonian.

```python
def ecc_quasiparticle_ionization(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int,
                                 fock: "np.ndarray") -> "np.ndarray":
    '''Ionization energy and spectral weight of the highest-occupied-orbital quasiparticle from the ECC effective Hamiltonian.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.
    fock : np.ndarray
        Shape (n, n), symmetric matrix placed in the one-hole/one-particle block (eV): diag(orbital_energies) for plain
        G0W0, or that matrix plus a static self-energy correction. The 2h1p and 2p1h blocks always use
        orbital_energies.

    Returns
    -------
    quasiparticle : np.ndarray
        Shape (2,), [ionization energy in eV, spectral weight]. The effective Hamiltonian is built from the closed-shell
        (spin-adapted) direct-ring amplitudes t and z of the same orbitals and screening, so its eigenvalues equal those
        of the corresponding G0W0 supermatrix. The weight of an eigenstate on orbital n_occ-1 is the product of the
        left and right eigenvector components on that orbital with the eigenvectors biorthonormal; the returned state
        is the one with the largest weight, and the ionization energy is minus its eigenvalue.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n, or eri or fock has the wrong shape.
    '''
    return quasiparticle
```

### Step 5

05_linearized_gw_density_matrix

Goal
----
Compute the linearized GW one-body density matrix of a closed-shell reference with direct-ring screening.

```python
def linearized_gw_density_matrix(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Per-spin linearized GW one-body density matrix in the canonical Hartree-Fock orbital basis.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    density : np.ndarray
        Shape (n, n), symmetric, normalized per spin so that the Hartree-Fock matrix would be 1 on each occupied
        diagonal element and its trace is n_occ. With indices i, j occupied, a, b virtual and nu the direct-RPA modes,
        gamma_ij = delta_ij - sum_{a,nu} M_{ia,nu} M_{ja,nu} / ((e_i - e_a - Omega_nu)(e_j - e_a - Omega_nu)),
        gamma_ab = sum_{i,nu} M_{ai,nu} M_{bi,nu} / ((e_i - e_a - Omega_nu)(e_i - e_b - Omega_nu)),
        gamma_ia = gamma_ai = [sum_{b,nu} M_{ib,nu} M_{ab,nu} / (e_i - e_b - Omega_nu)
                    - sum_{j,nu} M_{ij,nu} M_{aj,nu} / (e_j - e_a - Omega_nu)] / (e_i - e_a).
        These blocks are the frequency integral of G0 Sigma_c G0 with the Hartree-Fock Green's function G0.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n or eri does not have shape (n, n, n, n).
    '''
    return density
```

### Step 6

06_static_self_energy_correction

Goal
----
Compute the static self-energy correction generated by the change of the one-body density matrix away from Hartree-Fock.

```python
def static_self_energy_correction(eri: "np.ndarray", density: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Static Coulomb-minus-exchange potential of the per-spin density change relative to Hartree-Fock.

    Parameters
    ----------
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the canonical Hartree-Fock
        orbital basis (eV).
    density : np.ndarray
        Shape (n, n), symmetric per-spin correlated one-body density matrix in the same basis; the Hartree-Fock
        reference is 1 on the first n_occ diagonal elements and zero elsewhere.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    sigma : np.ndarray
        Shape (n, n), symmetric correction in eV,
        sigma_pq = sum_{rs} [2 (pq|rs) - (pr|qs)] Delta_rs with Delta = density - density_HF.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n or the shapes of eri and density do not match.
    '''
    return sigma
```

### Step 7

07_fci_ionization_energy

Goal
----
Compute the exact (full configuration interaction) first ionization energy of a zero-differential-overlap pi Hamiltonian.

```python
def fci_ionization_energy(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> float:
    '''Full configuration interaction first ionization energy of a closed-shell zero-differential-overlap Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Shape (n, n), symmetric one-electron Hamiltonian in the orthonormal site basis (eV).
    gamma : np.ndarray
        Shape (n, n), symmetric site repulsion matrix (eV). The two-electron energy of an occupation pattern is
        sum_p gamma_pp n_p,up n_p,down + sum_{p<q} gamma_pq n_p n_q, where n_p is the total occupation of site p.
    n_occ : int
        Number of electrons of each spin in the neutral system, 1 <= n_occ <= n.

    Returns
    -------
    ionization_energy : float
        E0(n_occ spin-up, n_occ - 1 spin-down) - E0(n_occ spin-up, n_occ spin-down) in eV, each E0 the lowest
        eigenvalue of the full configuration interaction matrix in that sector, as a Python float.

    Raises
    ------
    ValueError
        If h and gamma do not have the same square shape or n_occ is outside 1 <= n_occ <= n.
    '''
    return ionization_energy
```

### Step 8

08_break_even_interaction_scale

Goal
----
Find the interaction scale at which the density-corrected G0W0 ionization energy of a polyene stops being more accurate than plain G0W0 (orchestrator).

```python
def break_even_interaction_scale(n_sites: int, params: "np.ndarray", scale_low: float, scale_high: float) -> "np.ndarray":
    '''Break-even interaction scale of density-corrected versus plain G0W0@HF ionization energies of a PPP polyene.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms of the all-trans polyene, even and at least 2; n_sites / 2 orbitals are doubly occupied.
    params : np.ndarray
        Shape (6,), [W, U, t_short, t_long, r_short, r_long] of the PPP model in eV and Angstrom, as in
        ppp_polyene_hamiltonian.
    scale_low, scale_high : float
        Positive bracket 0 < scale_low < scale_high of the interaction scale lambda.

    Returns
    -------
    result : np.ndarray
        Shape (4,), [lambda_be, IP_G0W0, IP_G0W0+gamma, IP_FCI], the last three in eV at lambda_be. IP_G0W0 is the
        highest-occupied quasiparticle ionization energy with the Hartree-Fock Fock matrix, IP_G0W0+gamma the same with
        the Fock block augmented by the static correction of the linearized GW density matrix, and IP_FCI the full
        configuration interaction value. lambda_be is the root of IP_G0W0 + IP_G0W0+gamma - 2 IP_FCI inside the
        bracket, located to 1e-7.

    Raises
    ------
    ValueError
        If the bracket is not positive and increasing, or the break-even function has the same sign at both ends.
    '''
    return result
```
