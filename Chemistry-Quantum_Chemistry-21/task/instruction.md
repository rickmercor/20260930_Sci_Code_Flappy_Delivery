# Chemistry-Quantum_Chemistry-21

## Background

Downfolding constructs an effective Hamiltonian for a small active space by integrating out the
environment orbitals. The simplest choice, CASCI, keeps the bare interaction among the active
orbitals and lets the environment enter only through its static mean field. The source instead uses
the random-phase approximation as the low-level method: the particle-hole response of the
environment screens the two-body interaction of the active space (constrained RPA in a static limit,
its particle-hole-only variant, or the moment-conserving mRPA), a one-body correction removes the
mean-field interaction already contained in the screened two-body term, and the constant term of the
effective Hamiltonian is fixed by a Phi-functional argument so that the total energy from an exact
active-space solver includes the RPA correlation of the environment without double counting. The
result is a polarisable embedding whose total ground-state energies the source compares across the
screening flavours and against full CI and NEVPT2 for small molecules. A Pariser-Parr-Pople chain,
the standard semi-empirical model of conjugated pi systems, provides a fully specified test system
in which every step of the construction can be carried out in a few hundred lines of code.

## Problem

Active-space methods treat a handful of strongly correlated orbitals exactly and the rest of the
molecule at a lower level, and a recent source builds the effective active-space Hamiltonian from the
random-phase approximation: the environment orbitals are integrated out, their particle-hole response
screens the two-body interaction among the active orbitals (constrained RPA in the static limit, its
particle-hole-only variant, or the moment-conserving mRPA), a one-body correction removes the
mean-field interaction that the screened two-body term already contains, and the constant term of the
effective Hamiltonian is derived from a Phi-functional so that the total energy obtained by solving
the effective Hamiltonian exactly contains the RPA correlation of the environment without any double
counting. The source characterises the resulting polarisable embedding by the total ground-state
energies it produces.

Apply the source's scheme to a Pariser-Parr-Pople chain of N = 10 sites at half filling (10
electrons), written in its charge-neutral form

  H = sum_k t_k sum_sigma (a+_{k,sigma} a_{k+1,sigma} + h.c.) + U sum_i n_{i,up} n_{i,down}
      + sum_{i<j} V_ij (n_i - 1)(n_j - 1),

with alternating nearest-neighbour hopping t_k = -t (1 + delta (-1)^k) on bond k (k = 0, ..., 8; bond
k joins sites k and k + 1), delta = 0.07, on-site repulsion U = 4 t, the Ohno inter-site repulsion
V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) with kappa = 6 t for sites at unit spacing
(n_i = n_{i,up} + n_{i,down}), and t as the unit of energy. The mean-field reference is the
restricted closed-shell Hartree-Fock solution of this chain with canonical orbitals (its total
energy, including the constant of the charge-neutral form, is -10.2390 t; use it to check your
mean-field step); the active space is the (4,4) space of the two highest occupied and the two lowest
unoccupied canonical orbitals, all other orbitals form the environment, and the effective Hamiltonian
is solved exactly (full configuration interaction in the active space).

Build the source's downfolded Hamiltonian for constrained-RPA screening at screening frequency zero
with vanishing broadening, exactly as the source defines it: its low-level RPA and correlation-energy
functional, its constrained polarisability, its screened two-body term, its one-body correction, and
its constant energy shift. One convention is fixed here because the source prints it inconsistently:
the coupling block B of its particle-hole response matrices enters with the positive sign, B = +v,
as its Appendix B relation v = (1/2)[(A + B) - (A - B)] = B requires, and that relation governs this
task. Report the correlation energy of the downfolded Hamiltonian,
E_corr = E_tot - E_HF, where E_tot is the total energy of the downfolded Hamiltonian (constant term
plus exact active-space energy) and E_HF the restricted Hartree-Fock total energy of the full chain,
in units of t.

State the conventions you adopted and justify each from the source. In the reasoning give the RPA
correlation energy of the full chain, the RPA correlation energy of the active orbitals alone
evaluated with the screened interaction, the mean-field correction of the active orbitals that
enters the constant energy shift, and the active-space ground-state energy of the effective
Hamiltonian without its constant term (the exact active-space energy E_CAS = E_tot - E_0), all in
units of t.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

ppp_rhf

Goal
----
Solve the restricted closed-shell Hartree-Fock problem of the task's Pariser-Parr-Pople chain at half filling (one electron per site): n_sites sites on a line with unit spacing, nearest-neighbour hopping t_k = -t (1 + delta (-1)^k) on bond k (k = 0 joins sites 0 and 1), on-site repulsion U between opposite spins, and the Ohno inter-site repulsion V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) in its charge-neutral form sum_{i<j} V_ij (n_i - 1)(n_j - 1), so that the one-body matrix carries the on-site shift -sum_{j != i} V_ij and the model has the constant sum_{i<j} V_ij. Start from the eigenvectors of the one-body matrix and iterate the Fock matrix to self-consistency until the one-particle density matrix changes by less than 1e-10 in every element. Return an array whose row 0 holds the n_sites canonical orbital energies in ascending order and whose rows 1 to n_sites hold the orbital coefficient matrix C (C[i, p] = coefficient of site i in orbital p, orbitals in the same ascending order), each column with its phase fixed so that its coefficient on site 0 is positive.

```python
def ppp_rhf(n_sites: int, t: float, delta: float, U: float, kappa: float) -> "np.ndarray":
    """Solve the restricted closed-shell Hartree-Fock problem of the task's Pariser-Parr-Pople chain at half filling (one electron per site): n_sites sites on a line with unit spacing, nearest-neighbour hopping t_k = -t (1 + delta (-1)^k) on bond k (k = 0 joins sites 0 and 1), on-site repulsion U between opposite spins, and the Ohno inter-site repulsion V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) in its charge-neutral form sum_{i<j} V_ij (n_i - 1)(n_j - 1), so that the one-body matrix carries the on-site shift -sum_{j != i} V_ij and the model has the constant sum_{i<j} V_ij. Start from the eigenvectors of the one-body matrix and iterate the Fock matrix to self-consistency until the one-particle density matrix changes by less than 1e-10 in every element. Return an array whose row 0 holds the n_sites canonical orbital energies in ascending order and whose rows 1 to n_sites hold the orbital coefficient matrix C (C[i, p] = coefficient of site i in orbital p, orbitals in the same ascending order), each column with its phase fixed so that its coefficient on site 0 is positive.

    Parameters
    ----------
    n_sites : int
        Even number of sites (at least 2); the chain holds n_sites electrons.
    t : float
        Positive hopping scale; energies are returned in the units of t, U and kappa.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.

    Returns
    -------
    orbitals : numpy.ndarray
        Array of shape (n_sites + 1, n_sites), float64: row 0 the canonical orbital energies in ascending order, rows 1 to n_sites the coefficient matrix C with C[i, p] the coefficient of site i in orbital p, each column with a positive coefficient on site 0.

    Raises
    ------
    ValueError
        If n_sites is odd or below 2, |delta| >= 1, or t, U or kappa is not positive.
    """
    return orbitals
```

### Step 2

mo_integrals

Goal
----
Transform the two-electron interaction of the chain of step 01 (on-site U, Ohno V_ij with parameter kappa, unit site spacing, so that in the site basis the only non-zero integrals are (ii|jj) = V_ij with V_ii = U) to the basis of the orbitals C returned by step 01 and return the full four-index tensor of two-electron integrals in chemists' notation, element [p, q, r, s] = (pq|rs) = sum_ij C[i, p] C[i, q] V_ij C[j, r] C[j, s].

```python
def mo_integrals(C: "np.ndarray", U: float, kappa: float) -> "np.ndarray":
    """Transform the two-electron interaction of the chain of step 01 (on-site U, Ohno V_ij with parameter kappa, unit site spacing, so that in the site basis the only non-zero integrals are (ii|jj) = V_ij with V_ii = U) to the basis of the orbitals C returned by step 01 and return the full four-index tensor of two-electron integrals in chemists' notation, element [p, q, r, s] = (pq|rs) = sum_ij C[i, p] C[i, q] V_ij C[j, r] C[j, s].

    Parameters
    ----------
    C : numpy.ndarray
        Square (n, n) orbital coefficient matrix, C[i, p] the coefficient of site i in orbital p.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.

    Returns
    -------
    eri : numpy.ndarray
        Array of shape (n, n, n, n), float64, with eri[p, q, r, s] = (pq|rs).

    Raises
    ------
    ValueError
        If C is not a finite square matrix or U or kappa is not positive.
    """
    return eri
```

### Step 3

rpa_correlation_energy

Goal
----
Return the RPA correlation energy of the source's low-level method for a particle-hole space described by the vector delta_eps of orbital-energy differences (one entry per particle-hole pair) and the symmetric matrix v_ph of interaction integrals between pairs (element [I, J] = (ia|jb) for pairs I = (i, a) and J = (j, b), in the same pair order): build the RPA matrices from these two ingredients as the source's Eq. (2.8) does, with the sign of the coupling block fixed by its Appendix B, solve the source's eigenproblem, and evaluate the correlation-energy functional the source uses (its Eq. (2.22), Appendix B). Return 0.0 for an empty pair space.

```python
def rpa_correlation_energy(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> float:
    """Return the RPA correlation energy of the source's low-level method for a particle-hole space described by the vector delta_eps of orbital-energy differences (one entry per particle-hole pair) and the symmetric matrix v_ph of interaction integrals between pairs (element [I, J] = (ia|jb) for pairs I = (i, a) and J = (j, b), in the same pair order): build the RPA matrices from these two ingredients as the source's Eq. (2.8) does, with the sign of the coupling block fixed by its Appendix B, solve the source's eigenproblem, and evaluate the correlation-energy functional the source uses (its Eq. (2.22), Appendix B). Return 0.0 for an empty pair space.

    Parameters
    ----------
    delta_eps : numpy.ndarray
        One-dimensional array of length n of positive orbital-energy differences, one per particle-hole pair.
    v_ph : numpy.ndarray
        Symmetric (n, n) matrix of interaction integrals (ia|jb) between the pairs, in the order of delta_eps.

    Returns
    -------
    e_corr : float
        The RPA correlation energy in the energy units of the inputs (0.0 for n = 0).

    Raises
    ------
    ValueError
        If delta_eps has a non-positive entry, v_ph is not symmetric, or the shapes do not match.
    """
    return e_corr
```

### Step 4

rpa_static_kernel

Goal
----
For a particle-hole space described exactly as in step 03 (same delta_eps and v_ph, same pair order), return the n x n static kernel matrix K of the source's RPA: the sum over the modes nu of the source's eigenproblem, built from the two inputs as in step 03, of the outer product of the mode's transition-density vector (X+Y)^nu with itself divided by the mode's excitation energy Omega_nu, with the eigenvectors normalised as the source's Eq. (2.11) prescribes. K is the matrix through which the zero-frequency limit of the source's screened interaction, Eq. (2.12), is expressed once step 06 contracts it with the bare integrals and applies the pole factor of that limit. Return a (0, 0) array for an empty pair space.

```python
def rpa_static_kernel(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> "np.ndarray":
    """For a particle-hole space described exactly as in step 03 (same delta_eps and v_ph, same pair order), return the n x n static kernel matrix K of the source's RPA: the sum over the modes nu of the source's eigenproblem, built from the two inputs as in step 03, of the outer product of the mode's transition-density vector (X+Y)^nu with itself divided by the mode's excitation energy Omega_nu, with the eigenvectors normalised as the source's Eq. (2.11) prescribes. K is the matrix through which the zero-frequency limit of the source's screened interaction, Eq. (2.12), is expressed once step 06 contracts it with the bare integrals and applies the pole factor of that limit. Return a (0, 0) array for an empty pair space.

    Parameters
    ----------
    delta_eps : numpy.ndarray
        One-dimensional array of length n of positive orbital-energy differences, one per particle-hole pair.
    v_ph : numpy.ndarray
        Symmetric (n, n) matrix of interaction integrals (ia|jb) between the pairs, in the order of delta_eps.

    Returns
    -------
    kernel : numpy.ndarray
        Symmetric (n, n) array, float64: sum over modes of (X+Y)^nu (X+Y)^nu^T / Omega_nu; shape (0, 0) for n = 0.

    Raises
    ------
    ValueError
        If delta_eps has a non-positive entry, v_ph is not symmetric, or the shapes do not match.
    """
    return kernel
```

### Step 5

constrained_ph_pairs

Goal
----
Return the particle-hole space of the source's constrained RPA (its Eq. (2.14)) for n_orb spatial orbitals of which the first n_occ are doubly occupied and the orbitals listed in active form the active space, as an integer array of spin-orbital index pairs (i, a). Spin-orbitals are indexed 2p for the spin-up and 2p + 1 for the spin-down function of spatial orbital p; a pair is an occupied spin-orbital i and a virtual spin-orbital a of the same spin. List the pairs the source keeps in the constrained polarisability, ordered by i ascending and, within the same i, by a ascending.

```python
def constrained_ph_pairs(n_orb: int, n_occ: int, active: "list[int]") -> "np.ndarray":
    """Return the particle-hole space of the source's constrained RPA (its Eq. (2.14)) for n_orb spatial orbitals of which the first n_occ are doubly occupied and the orbitals listed in active form the active space, as an integer array of spin-orbital index pairs (i, a). Spin-orbitals are indexed 2p for the spin-up and 2p + 1 for the spin-down function of spatial orbital p; a pair is an occupied spin-orbital i and a virtual spin-orbital a of the same spin. List the pairs the source keeps in the constrained polarisability, ordered by i ascending and, within the same i, by a ascending.

    Parameters
    ----------
    n_orb : int
        Number of spatial orbitals (at least 2).
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= n_orb - 1; they are orbitals 0 to n_occ - 1.
    active : list[int]
        Distinct indices in [0, n_orb - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    pairs : numpy.ndarray
        Integer array of shape (m, 2); row k = (i, a) is the k-th kept spin-orbital particle-hole pair.

    Raises
    ------
    ValueError
        If n_orb is below 2, n_occ is not in [1, n_orb - 1], or active is empty, has repeated entries or entries outside [0, n_orb - 1].
    """
    return pairs
```

### Step 6

screened_interaction

Goal
----
Return the static (zero-frequency, vanishing broadening) screened interaction of the source, its Eqs. (2.12)-(2.13) at frequency zero, among the active orbitals: eri are the bare integrals of step 02 over all spatial orbitals, pairs the spin-orbital particle-hole pairs of the response (step 05, spin-orbital indexing as defined there), kernel the static kernel of step 04 for that pair space (same pair order), and active the active spatial orbitals. Return the four-index tensor W[t, u, v, w] of screened integrals in chemists' notation over the active orbitals in the order sorted(active), with every element screened.

```python
def screened_interaction(eri: "np.ndarray", pairs: "np.ndarray", kernel: "np.ndarray", active: "list[int]") -> "np.ndarray":
    """Return the static (zero-frequency, vanishing broadening) screened interaction of the source, its Eqs. (2.12)-(2.13) at frequency zero, among the active orbitals: eri are the bare integrals of step 02 over all spatial orbitals, pairs the spin-orbital particle-hole pairs of the response (step 05, spin-orbital indexing as defined there), kernel the static kernel of step 04 for that pair space (same pair order), and active the active spatial orbitals. Return the four-index tensor W[t, u, v, w] of screened integrals in chemists' notation over the active orbitals in the order sorted(active), with every element screened.

    Parameters
    ----------
    eri : numpy.ndarray
        Bare two-electron integrals (n, n, n, n) in chemists' notation over the spatial orbitals.
    pairs : numpy.ndarray
        Integer array (m, 2) of spin-orbital particle-hole pairs (i, a) as returned by step 05.
    kernel : numpy.ndarray
        Static kernel (m, m) of step 04 for exactly these pairs, in the same order.
    active : list[int]
        Distinct indices in [0, n - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    W : numpy.ndarray
        Array of shape (nA, nA, nA, nA), float64, nA = len(active): the screened integrals (tu|vw) over sorted(active).

    Raises
    ------
    ValueError
        If eri is not a four-index tensor with equal axes, pairs is not an (m, 2) array of valid spin-orbital indices, kernel is not (m, m), or active is invalid as in step 05.
    """
    return W
```

### Step 7

effective_one_body

Goal
----
Return the effective one-body matrix of the source's downfolded Hamiltonian, Eq. (2.20), over the active orbitals (order sorted(active)): the one-body term of the source's CASCI-type effective Hamiltonian (its Eq. (2.5)) built from the one-body integrals h_mo in the orbital basis, the bare integrals eri of step 02 and the doubly occupied orbitals outside the active space (the first n_occ orbitals are occupied), combined with the source's correction term of Eq. (2.19) for the screened two-body interaction veff of step 06 (given on the active block, same order).

```python
def effective_one_body(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]") -> "np.ndarray":
    """Return the effective one-body matrix of the source's downfolded Hamiltonian, Eq. (2.20), over the active orbitals (order sorted(active)): the one-body term of the source's CASCI-type effective Hamiltonian (its Eq. (2.5)) built from the one-body integrals h_mo in the orbital basis, the bare integrals eri of step 02 and the doubly occupied orbitals outside the active space (the first n_occ orbitals are occupied), combined with the source's correction term of Eq. (2.19) for the screened two-body interaction veff of step 06 (given on the active block, same order).

    Parameters
    ----------
    h_mo : numpy.ndarray
        Symmetric (n, n) one-body integrals in the orbital basis.
    eri : numpy.ndarray
        Bare two-electron integrals (n, n, n, n) in chemists' notation, same basis.
    veff : numpy.ndarray
        Screened integrals (nA, nA, nA, nA) over sorted(active), from step 06.
    n_occ : int
        Number of doubly occupied orbitals (orbitals 0 to n_occ - 1).
    active : list[int]
        Distinct indices in [0, n - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    t_eff : numpy.ndarray
        Symmetric (nA, nA) array, float64: the effective one-body matrix over sorted(active).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, h_mo is not square, or active is invalid as in step 05.
    """
    return t_eff
```

### Step 8

energy_shift

Goal
----
Return the electronic contribution of the environment to the constant term of the source's downfolded Hamiltonian, its Eq. (2.21) (derived in its Appendix A.2), for the inputs of step 07 together with the two RPA correlation energies the expression needs, both obtained with step 03 by the caller: e_rpa_full for the full system with the bare interaction and e_rpa_active for the active orbitals alone with the screened interaction. Assemble the mean-field energy of the environment orbitals (Eq. (2.6)), the mean-field correction of the active orbitals that the screening of veff induces, and the two correlation energies with the signs the source's derivation prescribes.

```python
def energy_shift(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]", e_rpa_full: float, e_rpa_active: float) -> float:
    """Return the electronic contribution of the environment to the constant term of the source's downfolded Hamiltonian, its Eq. (2.21) (derived in its Appendix A.2), for the inputs of step 07 together with the two RPA correlation energies the expression needs, both obtained with step 03 by the caller: e_rpa_full for the full system with the bare interaction and e_rpa_active for the active orbitals alone with the screened interaction. Assemble the mean-field energy of the environment orbitals (Eq. (2.6)), the mean-field correction of the active orbitals that the screening of veff induces, and the two correlation energies with the signs the source's derivation prescribes.

    Parameters
    ----------
    h_mo : numpy.ndarray
        Symmetric (n, n) one-body integrals in the orbital basis.
    eri : numpy.ndarray
        Bare two-electron integrals (n, n, n, n) in chemists' notation, same basis.
    veff : numpy.ndarray
        Screened integrals (nA, nA, nA, nA) over sorted(active), from step 06.
    n_occ : int
        Number of doubly occupied orbitals (orbitals 0 to n_occ - 1).
    active : list[int]
        Distinct indices in [0, n - 1] of the active spatial orbitals (non-empty).
    e_rpa_full : float
        RPA correlation energy of the full system with the bare interaction (step 03).
    e_rpa_active : float
        RPA correlation energy of the active orbitals alone with the screened interaction (step 03).

    Returns
    -------
    e_shift : float
        The electronic energy shift E_E of the downfolded Hamiltonian in the energy units of the inputs.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, active is invalid as in step 05, or either energy is not finite.
    """
    return e_shift
```

### Step 9

active_space_fci

Goal
----
Return the ground-state energy of the effective Hamiltonian H = sum_tu t_eff[t, u] a+_t a_u + (1/2) sum_tuvw veff[t, u, v, w] a+_t a+_v a_w a_u, where the creation and annihilation operators run over both spins of every active spatial orbital, t_eff is the (real, symmetric) one-body matrix and veff the two-body tensor in chemists' notation (tu|vw), with n_elec electrons (n_elec even) distributed over the active orbitals: the lowest eigenvalue of the full configuration-interaction Hamiltonian in the sector with equal numbers of spin-up and spin-down electrons. Return 0.0 for n_elec = 0.

```python
def active_space_fci(t_eff: "np.ndarray", veff: "np.ndarray", n_elec: int) -> float:
    """Return the ground-state energy of the effective Hamiltonian H = sum_tu t_eff[t, u] a+_t a_u + (1/2) sum_tuvw veff[t, u, v, w] a+_t a+_v a_w a_u, where the creation and annihilation operators run over both spins of every active spatial orbital, t_eff is the (real, symmetric) one-body matrix and veff the two-body tensor in chemists' notation (tu|vw), with n_elec electrons (n_elec even) distributed over the active orbitals: the lowest eigenvalue of the full configuration-interaction Hamiltonian in the sector with equal numbers of spin-up and spin-down electrons. Return 0.0 for n_elec = 0.

    Parameters
    ----------
    t_eff : numpy.ndarray
        Real symmetric (nA, nA) one-body matrix over the active orbitals.
    veff : numpy.ndarray
        Two-body tensor (nA, nA, nA, nA) in chemists' notation over the same orbitals.
    n_elec : int
        Even number of electrons, 0 <= n_elec <= 2 nA.

    Returns
    -------
    e_cas : float
        The lowest eigenvalue of H in the S_z = 0 sector (0.0 for n_elec = 0), same energy units as the inputs.

    Raises
    ------
    ValueError
        If t_eff is not square, veff does not have the matching four-index shape, or n_elec is odd, negative or larger than twice the number of orbitals.
    """
    return e_cas
```

### Step 10

downfolded_correlation_energy

Goal
----
Orchestrator. For the chain of step 01 with the given parameters and the active spatial orbitals active (indices into the ascending canonical orbitals of step 01), run the whole downfolding: solve the mean-field problem with step 01, transform the integrals with step 02 (and the one-body matrix of the chain to the same basis), evaluate the RPA correlation energy of the full chain with step 03 on its complete spin-orbital particle-hole space, build the constrained particle-hole space with step 05, its static kernel with step 04 and the screened interaction with step 06, evaluate with step 03 the RPA correlation energy of the active orbitals alone with the screened interaction (the source's prescription for that term), build the effective one-body matrix with step 07 and the constant shift with step 08, solve the active space with step 09 using the number of electrons that occupy the active orbitals in the mean-field reference, and return the correlation energy E_corr = E_tot - E_HF, where E_tot is the total energy of the downfolded Hamiltonian (constant term plus active-space energy) and E_HF the restricted Hartree-Fock energy of the full chain from step 01 (the constant of the model cancels). The value is returned in the energy units in which t, U and kappa are given (it scales linearly with a common scaling of the three); for t = 1 this is the value in units of t. Call the earlier step functions rather than reimplementing them.

```python
def downfolded_correlation_energy(n_sites: int, t: float, delta: float, U: float, kappa: float, active: "list[int]") -> float:
    """Orchestrator. For the chain of step 01 with the given parameters and the active spatial orbitals active (indices into the ascending canonical orbitals of step 01), run the whole downfolding: solve the mean-field problem with step 01, transform the integrals with step 02 (and the one-body matrix of the chain to the same basis), evaluate the RPA correlation energy of the full chain with step 03 on its complete spin-orbital particle-hole space, build the constrained particle-hole space with step 05, its static kernel with step 04 and the screened interaction with step 06, evaluate with step 03 the RPA correlation energy of the active orbitals alone with the screened interaction (the source's prescription for that term), build the effective one-body matrix with step 07 and the constant shift with step 08, solve the active space with step 09 using the number of electrons that occupy the active orbitals in the mean-field reference, and return the correlation energy E_corr = E_tot - E_HF, where E_tot is the total energy of the downfolded Hamiltonian (constant term plus active-space energy) and E_HF the restricted Hartree-Fock energy of the full chain from step 01 (the constant of the model cancels). The value is returned in the energy units in which t, U and kappa are given (it scales linearly with a common scaling of the three); for t = 1 this is the value in units of t. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n_sites : int
        Even number of sites (at least 2).
    t : float
        Positive hopping scale.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.
    active : list[int]
        Distinct indices in [0, n_sites - 1] of the active canonical orbitals (non-empty).

    Returns
    -------
    e_corr : float
        E_corr = E_tot - E_HF in the energy units of t, U and kappa (native Python float).

    Raises
    ------
    ValueError
        If n_sites is odd or below 2, active is invalid, |delta| >= 1, or t, U or kappa is not positive.
    """
    return e_corr
```
