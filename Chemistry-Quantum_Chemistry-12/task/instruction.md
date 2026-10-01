# Chemistry-Quantum_Chemistry-12

## Background

Green's-function methods describe electron removal and addition through a frequency-dependent self-energy Sigma(omega): the quasiparticle energies that solve omega = e_p + Sigma_pp(omega) are the ionisation potentials and electron affinities. In the GW approximation the correlation self-energy is built from the random-phase screening of the Coulomb interaction. An alternative is to build it from the excited states of the Bethe-Salpeter equation (their excitation energies and transition amplitudes), which describe screening more accurately than the random-phase approximation for molecules; such a self-energy is a sum over intermediate states of products of two scattering amplitudes, and when the two amplitudes differ the imaginary part of the self-energy is no longer of Fermi-golden-rule form and the spectral function can become negative in some energy ranges. Rewriting the self-energy as a complete square of the amplitudes restores positivity at the price of new diagrams. The Pariser-Parr-Pople model - a tight-binding chain with alternating bonds, an on-site repulsion and a long-range Ohno interaction, with a neutralising background - is the standard pi-electron model on which such methods can be evaluated exactly in plain linear algebra: Hartree-Fock, static screening, the Bethe-Salpeter problem and the self-energy all reduce to finite matrices.

## Problem

Many-body perturbation theory describes the removal of an electron from a molecule through a
frequency-dependent self-energy; in the GW approximation the self-energy is built from the random-phase
screening of the Coulomb interaction. A recent source turns this around: it builds the correlation
self-energy from the excited states of the Bethe-Salpeter equation (their energies and transition
amplitudes) instead of from the random-phase approximation, and then repairs a flaw of that construction
- a spectral function that is not positive semidefinite - by rewriting the self-energy as a complete
square of two scattering amplitudes, which doubles one of its terms and adds a new one.

Apply the source's minimal positive-semidefinite self-energy to an open Pariser-Parr-Pople chain of
eight sites with one electron per site (a closed-shell singlet, four doubly occupied orbitals):
nearest-neighbour hopping -t(1 + delta(-1)^i) between sites i and i+1 (sites numbered from 0) with
t = 1 and delta = 0.15; on-site repulsion U = 4; Ohno interaction V_ij = U / sqrt(1 + (U|i - j|/kappa)^2)
between different sites with kappa = 1.5; site energies (-1.1, -1.8, -1.3, -1.6, -1.2, -2.0, -1.4, -1.7)
in units of t; and a neutralising +1 background charge on every site, so that the one-electron matrix
carries -sum_{j != i} V_ij on its diagonal. Use the restricted closed-shell Hartree-Fock determinant of
this Hamiltonian as the reference, the statically screened interaction of the direct random-phase
approximation as the screened interaction, the source's Bethe-Salpeter kernel and excitation manifold,
the source's two half-diagrams and the complete-square combination it prescribes for its minimal
positive-semidefinite self-energy, and its one-shot treatment of the quasiparticle equation on the
Hartree-Fock reference, with a graphical (not linearised) solution of that equation.

Report as the final answer the first ionisation potential of the chain at this level, i.e. minus the
quasiparticle energy of the highest occupied orbital, to at least four decimal places. In the reasoning,
also report the Hartree-Fock energy of the highest occupied orbital, the lowest singlet excitation energy
of the Bethe-Salpeter problem, the value of the correlation self-energy of the highest occupied orbital
at its quasiparticle energy, and the quasiparticle renormalisation factor Z = [1 - dSigma/domega]^-1 of
that orbital evaluated at its quasiparticle energy. State the conventions you adopted and justify each
from the source.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, state the conventions you adopted and the intermediate values requested above, concisely.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

ppp_hamiltonian

Goal
----
Return the one-electron matrix h and the site-site interaction matrix V of an open Pariser-Parr-Pople chain of K sites, stacked as an array of shape (2, K, K). With sites numbered i = 0..K-1: the nearest-neighbour hopping is h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i); the interaction is the Ohno form V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) for i != j and V_ii = U; the diagonal of h is h_ii = eps_i - sum_{j != i} V_ij, i.e. the site energy shifted by the interaction with a +1 neutralising background on every site. All other elements of h are zero. Raise ValueError if eps_site is not one-dimensional with at least two entries, or if t <= 0, U <= 0, kappa <= 0 or |delta| >= 1.

```python
def ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    """Return the one-electron matrix h and the site-site interaction matrix V of an open Pariser-Parr-Pople chain of K sites, stacked as an array of shape (2, K, K). With sites numbered i = 0..K-1: the nearest-neighbour hopping is h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i); the interaction is the Ohno form V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) for i != j and V_ii = U; the diagonal of h is h_ii = eps_i - sum_{j != i} V_ij, i.e. the site energy shifted by the interaction with a +1 neutralising background on every site. All other elements of h are zero. Raise ValueError if eps_site is not one-dimensional with at least two entries, or if t <= 0, U <= 0, kappa <= 0 or |delta| >= 1.

    Parameters
    ----------
    t : float
        Nearest-neighbour hopping (positive).
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion (positive).
    kappa : float
        Ohno screening length (positive).
    eps_site : numpy.ndarray
        Site energies, length K >= 2.

    Returns
    -------
    hV : numpy.ndarray
        Array (2, K, K): hV[0] = h, hV[1] = V.

    Raises
    ------
    ValueError
        If eps_site is not 1-D with at least two sites, or t, U, kappa are not positive, or |delta| >= 1.
    """
    return hV
```

### Step 2

rhf_reference

Goal
----
Return the restricted closed-shell Hartree-Fock reference of the density-density Hamiltonian whose two-electron integrals in the site basis are (pq|rs) = delta_pq delta_rs V_pr, with n_pairs doubly occupied orbitals: start from the eigenvectors of h, iterate the closed-shell density D = 2 C_occ C_occ^T with the damping D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then diagonalise the Fock matrix built from the converged D once more. Return an array of shape (K + 1, K): row 0 holds the orbital energies in ascending order and rows 1..K the coefficient matrix C (columns = orbitals in the same order), each column phase-fixed so that its first component of magnitude above 1e-8 is positive. Raise ValueError if h and V are not symmetric K x K matrices with K >= 2, if n_pairs is not in 1..K-1, or if the iteration does not converge in 5000 steps.

```python
def rhf_reference(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Return the restricted closed-shell Hartree-Fock reference of the density-density Hamiltonian whose two-electron integrals in the site basis are (pq|rs) = delta_pq delta_rs V_pr, with n_pairs doubly occupied orbitals: start from the eigenvectors of h, iterate the closed-shell density D = 2 C_occ C_occ^T with the damping D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then diagonalise the Fock matrix built from the converged D once more. Return an array of shape (K + 1, K): row 0 holds the orbital energies in ascending order and rows 1..K the coefficient matrix C (columns = orbitals in the same order), each column phase-fixed so that its first component of magnitude above 1e-8 is positive. Raise ValueError if h and V are not symmetric K x K matrices with K >= 2, if n_pairs is not in 1..K-1, or if the iteration does not converge in 5000 steps.

    Parameters
    ----------
    h : numpy.ndarray
        One-electron matrix (K, K).
    V : numpy.ndarray
        Site-site interaction (K, K).
    n_pairs : int
        Number of doubly occupied orbitals.

    Returns
    -------
    ref : numpy.ndarray
        Array (K + 1, K): row 0 = orbital energies ascending, rows 1..K = C.

    Raises
    ------
    ValueError
        If the matrices are not symmetric (K, K) with K >= 2, n_pairs is out of range, or the SCF does not converge.
    """
    return ref
```

### Step 3

mo_two_electron_integrals

Goal
----
Return the chemists' two-electron integrals of the density-density Hamiltonian in the orbital basis C: (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js, as an array of shape (K, K, K, K) indexed [p, q, r, s]. Raise ValueError if V or C is not K x K.

```python
def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """Return the chemists' two-electron integrals of the density-density Hamiltonian in the orbital basis C: (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js, as an array of shape (K, K, K, K) indexed [p, q, r, s]. Raise ValueError if V or C is not K x K.

    Parameters
    ----------
    V : numpy.ndarray
        Site-site interaction (K, K).
    C : numpy.ndarray
        Orbital coefficients (K, K), columns = orbitals.

    Returns
    -------
    eri : numpy.ndarray
        Array (K, K, K, K) with eri[p, q, r, s] = (pq|rs).

    Raises
    ------
    ValueError
        If V or C is not (K, K).
    """
    return eri
```

### Step 4

static_screened_interaction

Goal
----
Return the statically screened interaction W0 - the zero-frequency screened interaction of the direct random-phase approximation (spin-summed, real orbitals), the screening used by GW and by the source's Bethe-Salpeter kernel - in the orbital basis, as an array with the same index layout as the integrals, W0[p, q, r, s] = (pq|W0|rs) in chemists' notation; the occupied-virtual pairs (i, a) run over i = 0..n_occ-1 (occupied) and a = n_occ..K-1 (virtual). Raise ValueError if eri is not (K, K, K, K) for K = len(eps), if n_occ is not in 1..K-1, or if any occupied-virtual gap e_a - e_i is not positive.

```python
def static_screened_interaction(eps: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Return the statically screened interaction W0 - the zero-frequency screened interaction of the direct random-phase approximation (spin-summed, real orbitals), the screening used by GW and by the source's Bethe-Salpeter kernel - in the orbital basis, as an array with the same index layout as the integrals, W0[p, q, r, s] = (pq|W0|rs) in chemists' notation; the occupied-virtual pairs (i, a) run over i = 0..n_occ-1 (occupied) and a = n_occ..K-1 (virtual). Raise ValueError if eri is not (K, K, K, K) for K = len(eps), if n_occ is not in 1..K-1, or if any occupied-virtual gap e_a - e_i is not positive.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K, ascending.
    eri : numpy.ndarray
        Chemists' integrals (K, K, K, K).
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    W0 : numpy.ndarray
        Array (K, K, K, K) with W0[p, q, r, s] = (pq|W0|rs).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is out of range, or an occupied-virtual gap is not positive.
    """
    return W0
```

### Step 5

bse_excitations

Goal
----
Solve the source's spin-adapted Bethe-Salpeter excitation problem for the singlet manifold (triplet = 0) or the triplet manifold (triplet = 1). The occupied-virtual pairs (i, a), i = 0..n_occ-1, a = n_occ..K-1, are ordered i-major and a-minor; eri and W0 are chemists'-notation arrays with the layout of steps 03 and 04. Keep the n = n_occ (K - n_occ) positive roots Omega_nu in ascending order; normalise each eigenvector to sum_{ia} (X^2 - Y^2) = 1 and fix its sign so that the component of X + Y of largest magnitude is positive. Return an array of shape (2n + 1, n): row 0 = Omega, rows 1..n = X (row nu, column = pair), rows n+1..2n = Y. Raise ValueError if the shapes are inconsistent, n_occ is out of range, triplet is not 0 or 1, the reference is unstable, or a squared excitation energy is not positive.

```python
def bse_excitations(eps: "np.ndarray", eri: "np.ndarray", W0: "np.ndarray", n_occ: int, triplet: int) -> "np.ndarray":
    """Solve the source's spin-adapted Bethe-Salpeter excitation problem for the singlet manifold (triplet = 0) or the triplet manifold (triplet = 1). The occupied-virtual pairs (i, a), i = 0..n_occ-1, a = n_occ..K-1, are ordered i-major and a-minor; eri and W0 are chemists'-notation arrays with the layout of steps 03 and 04. Keep the n = n_occ (K - n_occ) positive roots Omega_nu in ascending order; normalise each eigenvector to sum_{ia} (X^2 - Y^2) = 1 and fix its sign so that the component of X + Y of largest magnitude is positive. Return an array of shape (2n + 1, n): row 0 = Omega, rows 1..n = X (row nu, column = pair), rows n+1..2n = Y. Raise ValueError if the shapes are inconsistent, n_occ is out of range, triplet is not 0 or 1, the reference is unstable, or a squared excitation energy is not positive.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K.
    eri : numpy.ndarray
        Bare chemists' integrals (K, K, K, K).
    W0 : numpy.ndarray
        Statically screened interaction (K, K, K, K).
    n_occ : int
        Number of occupied orbitals.
    triplet : int
        0 for the singlet manifold, 1 for the triplet manifold.

    Returns
    -------
    bse : numpy.ndarray
        Array (2n + 1, n): Omega, then X, then Y.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ or triplet is out of range, or the reference is unstable.
    """
    return bse
```

### Step 6

self_energy_residues

Goal
----
Return the residue tensor of the source's minimal positive-semidefinite self-energy for every excitation nu of a singlet solution of step 05 and every intermediate orbital k, as an array R of shape (n, K, K, K) indexed [nu, k, p, q]: R[nu, k] is the rank-one residue matrix of that pole, normalised so that the self-energy of step 07 is exactly the sum of R over its poles with no further factor. The source writes the hole-intermediate case (occupied k); for a virtual intermediate k = c use the particle reading in which the roles of X and Y in b are exchanged - this resolves the source's 'treated identically' for the particle branch and is fixed by the exact second-order limit. The sums run over the occupied-virtual pairs of step 05 in the same order; (pq|rs) are the bare integrals and (pq|W0|rs) the screened ones, both in chemists' notation. Raise ValueError if the shapes are inconsistent or n_occ is out of range.

```python
def self_energy_residues(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Return the residue tensor of the source's minimal positive-semidefinite self-energy for every excitation nu of a singlet solution of step 05 and every intermediate orbital k, as an array R of shape (n, K, K, K) indexed [nu, k, p, q]: R[nu, k] is the rank-one residue matrix of that pole, normalised so that the self-energy of step 07 is exactly the sum of R over its poles with no further factor. The source writes the hole-intermediate case (occupied k); for a virtual intermediate k = c use the particle reading in which the roles of X and Y in b are exchanged - this resolves the source's 'treated identically' for the particle branch and is fixed by the exact second-order limit. The sums run over the occupied-virtual pairs of step 05 in the same order; (pq|rs) are the bare integrals and (pq|W0|rs) the screened ones, both in chemists' notation. Raise ValueError if the shapes are inconsistent or n_occ is out of range.

    Parameters
    ----------
    eri : numpy.ndarray
        Bare chemists' integrals (K, K, K, K).
    W0 : numpy.ndarray
        Statically screened interaction (K, K, K, K).
    bse : numpy.ndarray
        Singlet solution of step 05, shape (2n + 1, n).
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    res : numpy.ndarray
        Array (n, K, K, K): R[nu, k, p, q] = c_p c_q, the residue matrix of pole (nu, k).

    Raises
    ------
    ValueError
        If the shapes are inconsistent or n_occ is out of range.
    """
    return res
```

### Step 7

psd_self_energy

Goal
----
Return the source's minimal positive-semidefinite self-energy at a real frequency omega as the real symmetric (K, K) matrix: the residue matrices R of step 06 summed over the singlet excitations Omega_nu of step 05 and over the intermediate orbitals, each divided by omega minus its pole (singlets only; no imaginary broadening). Raise ValueError if the shapes are inconsistent, n_occ is out of range, or omega lies within 1e-12 of a pole.

```python
def psd_self_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, omega: float) -> "np.ndarray":
    """Return the source's minimal positive-semidefinite self-energy at a real frequency omega as the real symmetric (K, K) matrix: the residue matrices R of step 06 summed over the singlet excitations Omega_nu of step 05 and over the intermediate orbitals, each divided by omega minus its pole (singlets only; no imaginary broadening). Raise ValueError if the shapes are inconsistent, n_occ is out of range, or omega lies within 1e-12 of a pole.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K.
    Om : numpy.ndarray
        Singlet excitation energies, length n.
    res : numpy.ndarray
        Residue tensor (n, K, K, K) from step 06.
    n_occ : int
        Number of occupied orbitals.
    omega : float
        Real frequency.

    Returns
    -------
    sigma : numpy.ndarray
        The (K, K) self-energy matrix at omega.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is out of range, or omega coincides with a pole.
    """
    return sigma
```

### Step 8

quasiparticle_energy

Goal
----
Return the one-shot quasiparticle energy of orbital p (0-based) on the Hartree-Fock reference: the solution of omega = e_p + Sigma_pp(omega) with the self-energy of step 07, found graphically as the root of f(omega) = omega - e_p - Sigma_pp(omega) nearest e_p by bisection: start from the bracket [e_p - 0.5, e_p + 0.5], widen both ends by 0.25 (at most 40 times) until f changes sign across it, then bisect keeping the sub-bracket in which f changes sign until its width is below 1e-12 and return its midpoint. No linearisation. Raise ValueError if p is out of range or no root is bracketed.

```python
def quasiparticle_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, p: int) -> float:
    """Return the one-shot quasiparticle energy of orbital p (0-based) on the Hartree-Fock reference: the solution of omega = e_p + Sigma_pp(omega) with the self-energy of step 07, found graphically as the root of f(omega) = omega - e_p - Sigma_pp(omega) nearest e_p by bisection: start from the bracket [e_p - 0.5, e_p + 0.5], widen both ends by 0.25 (at most 40 times) until f changes sign across it, then bisect keeping the sub-bracket in which f changes sign until its width is below 1e-12 and return its midpoint. No linearisation. Raise ValueError if p is out of range or no root is bracketed.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K.
    Om : numpy.ndarray
        Singlet excitation energies, length n.
    res : numpy.ndarray
        Residue tensor (n, K, K, K) from step 06.
    n_occ : int
        Number of occupied orbitals.
    p : int
        Orbital index, 0-based.

    Returns
    -------
    e_qp : float
        The quasiparticle energy of orbital p.

    Raises
    ------
    ValueError
        If p is out of range or no root is bracketed.
    """
    return e_qp
```

### Step 9

psd_ionization_potential

Goal
----
Orchestrator. Return the first ionisation potential of the PPP chain at the source's PSD-I level, IP = -e_HOMO^QP, by calling the earlier steps rather than re-implementing them: the Hamiltonian (step 01), the Hartree-Fock reference (02), the orbital-basis integrals (03), the static RPA screening (04), the singlet Bethe-Salpeter solution (05; also solve the triplet manifold and raise ValueError if either manifold has a non-positive excitation energy), the self-energy residues (06), the self-energy (07) and the graphical quasiparticle energy (08) of the highest occupied orbital p = n_pairs - 1; verify that the returned energy satisfies e = e_p + Sigma_pp(e) to 1e-9 with step 07 and raise ValueError otherwise. Raise ValueError for invalid model parameters as in the earlier steps.

```python
def psd_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", n_pairs: int) -> float:
    """Orchestrator. Return the first ionisation potential of the PPP chain at the source's PSD-I level, IP = -e_HOMO^QP, by calling the earlier steps rather than re-implementing them: the Hamiltonian (step 01), the Hartree-Fock reference (02), the orbital-basis integrals (03), the static RPA screening (04), the singlet Bethe-Salpeter solution (05; also solve the triplet manifold and raise ValueError if either manifold has a non-positive excitation energy), the self-energy residues (06), the self-energy (07) and the graphical quasiparticle energy (08) of the highest occupied orbital p = n_pairs - 1; verify that the returned energy satisfies e = e_p + Sigma_pp(e) to 1e-9 with step 07 and raise ValueError otherwise. Raise ValueError for invalid model parameters as in the earlier steps.

    Parameters
    ----------
    t : float
        Nearest-neighbour hopping (positive).
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion (positive).
    kappa : float
        Ohno screening length (positive).
    eps_site : numpy.ndarray
        Site energies, length K >= 2.
    n_pairs : int
        Number of doubly occupied orbitals.

    Returns
    -------
    IP : float
        The PSD-I first ionisation potential.

    Raises
    ------
    ValueError
        If the model parameters are invalid, the reference is unstable, or the quasiparticle equation is not satisfied.
    """
    return IP
```
