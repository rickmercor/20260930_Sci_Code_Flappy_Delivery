# Chemistry-Quantum_Chemistry-51

## Background

Koopmans' theorem identifies ionisation potentials with the negatives of Hartree-Fock orbital energies and is the workhorse estimate behind most orbital-energy-based reasoning in chemistry, but it neglects electron correlation and orbital relaxation entirely. The extended Koopmans' theorem removes the first limitation without a separate calculation on the ionised system: for any wavefunction with known one- and two-particle reduced density matrices, the ionisation energies are the eigenvalues of a generalised eigenvalue problem in which a generalised Fock matrix (the contraction of the integrals with the densities, also known as the orbital Lagrangian) plays the role of the Fock matrix and the one-particle density matrix plays the role of the overlap. For expectation-value methods such as Hartree-Fock, multiconfiguration self-consistent field and configuration interaction this is a textbook construction; for projective methods such as coupled cluster it requires the response (Lagrangian) densities, in which the de-excitation multipliers of the energy functional replace the missing bra.

Pair coupled cluster doubles restricts the cluster operator to electron-pair excitations. The resulting wavefunction is equivalent to an antisymmetric product of one-reference-orbital geminals, is size-extensive, captures strong pair (static) correlation at mean-field-like cost, and is strongly orbital dependent: its orbitals are therefore optimised variationally, which makes every occupied-occupied, occupied-virtual and virtual-virtual rotation a parameter and yields natural orbitals that typically localise on bonds. Because the method is seniority-zero, its response one-particle density matrix is diagonal in its own orbital basis and its two-particle density matrix has only a few non-zero index patterns, all of them low-order polynomials in the amplitudes and the multipliers.

Pariser-Parr-Pople chains are the standard pi-electron models of conjugated polyenes: nearest-neighbour hopping with bond alternation, an on-site repulsion, a screened long-range Coulomb tail, and a neutral background charge on every site that turns the half-filled chain into the neutral molecule. Their two-electron integrals are of density-density form, which keeps every integral transformation and Fock build elementary while retaining the full structure of a correlated ionisation problem.

## Problem

The ionisation potentials of a correlated wavefunction can be read off a generalised eigenvalue problem built from its one- and two-particle reduced density matrices (the extended Koopmans' theorem), and a recent scheme does this for the pair-coupled-cluster-doubles (pCCD) wavefunction whose orbitals are optimised variationally: the response density matrices that the orbital optimisation produces anyway feed a generalised Fock matrix, so correlated ionisation potentials follow at mean-field cost once the pCCD calculation is done.

Apply that scheme to a half-filled open Pariser-Parr-Pople chain of eight sites (eight electrons, four pairs, real restricted closed-shell orbitals): nearest-neighbour hopping -t (1 + delta (-1)^i) between sites i and i+1 for i = 0, ..., 6 with t = 1 (the unit of energy) and delta = 0.15, on-site repulsion U = 6, Ohno interaction V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) with kappa = 4 (so V_ii = U), site energies eps_i = -4 + (0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25) for i = 0, ..., 7, and the neutral-background convention h_ii = eps_i - sum_{j != i} V_ij; the two-electron integrals are of density-density form, (pq|rs) = delta_pq delta_rs V_pr in the site basis (chemists' notation).

Start from the restricted Hartree-Fock orbitals, obtain the pCCD wavefunction with its orbitals optimised as the source prescribes (every orbital rotation is a variational parameter; converge the amplitude equations to 1e-12 and the orbital gradient to 1e-10), and work in the resulting natural-orbital basis. Form the source's response one- and two-particle density matrices and its generalised Fock matrix, discard the natural orbitals whose per-spin occupation lies below 5e-5, and solve the source's extended-Koopmans' eigenvalue problem in the form the source uses (symmetric orthogonalisation with the inverse square root of the one-particle density matrix, the generalised Fock matrix symmetrised). Take as ionisation potentials the negatives of the eigenvalues whose unit-norm eigenvectors of the orthogonalised problem carry more than half of their squared norm on the four strongly occupied natural orbitals, and report the smallest of them, the first ionisation potential, in units of t.

In the reasoning, report the restricted Hartree-Fock energy, the orbital-optimised pCCD energy, the per-spin natural occupation numbers, the other ionisation potentials of occupied character and the Koopmans estimate from the Hartree-Fock orbital energy (these are the scalars that determine the final number), and state the conventions you adopted, justifying each from the source.

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
Return the array [h, V] of shape (2, K, K) for the open PPP chain with K = len(eps_site) sites: h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i) for i = 0..K-2 (all other off-diagonal elements zero), h_ii = eps_i - sum_{j != i} V_ij (every site carries a +1 background charge), and the Ohno interaction V_ij = U / sqrt(1 + (U |i - j| / kappa)^2), so V_ii = U. The two-electron integrals of the model are of density-density form, (pq|rs) = delta_pq delta_rs V_pr in the site basis (chemists' notation). Raise ValueError if eps_site is not a 1-D array with at least two entries, if t, U or kappa is not positive, or if |delta| >= 1.

```python
def ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    '''One- and two-body matrices of the open PPP chain in the site basis.

    Parameters
    ----------
    t : float
        Hopping scale, t > 0.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion, U > 0.
    kappa : float
        Ohno screening parameter, kappa > 0.
    eps_site : np.ndarray
        Site energies eps_i, 1-D array of length K >= 2.

    Returns
    -------
    hV : np.ndarray
        Array of shape (2, K, K): hV[0] the one-body matrix h, hV[1] the site-basis interaction matrix V.

    Raises
    ------
    ValueError
        If eps_site is not a 1-D array with at least two entries, or t, U or kappa is not positive, or |delta| >= 1.
    '''
    return hV
```

### Step 2

rhf_orbitals

Goal
----
Return the restricted closed-shell Hartree-Fock orbitals C (K, K) of the Hamiltonian with one-body matrix h and density-density integrals (pq|rs) = delta_pq delta_rs V_pr, with P = n_pairs doubly occupied orbitals: solve the Roothaan equations F C = C eps, F = h + J - K/2 with J_pq = delta_pq sum_r V_pr D_rr and K_pq = V_pq D_pq for the density D = 2 C_occ C_occ^T, by damped fixed-point iteration (new density averaged 1:1 with the previous one) started from the eigenvectors of h, until the density changes by less than 1e-12 element-wise; then diagonalise the final Fock matrix once more. Columns ordered by ascending orbital energy. Column phase convention: in every column the first component with |c| > 1e-8 is positive. Raise ValueError if h or V is not a symmetric (K, K) matrix with K >= 2, if n_pairs is outside [1, K - 1], or if the iteration does not converge within 2000 cycles.

```python
def rhf_orbitals(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    '''Restricted Hartree-Fock orbitals of the density-density Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Symmetric one-body matrix (K, K).
    V : np.ndarray
        Symmetric site-basis interaction matrix (K, K).
    n_pairs : int
        Number of doubly occupied orbitals P, 1 <= P < K.

    Returns
    -------
    C : np.ndarray
        Orbital coefficient matrix (K, K), column k the k-th orbital in ascending orbital-energy order, phase-fixed.

    Raises
    ------
    ValueError
        If h or V is not a symmetric (K, K) matrix with K >= 2, if n_pairs is outside [1, K - 1], or if the iteration does not converge.
    '''
    return C
```

### Step 3

mo_two_electron_integrals

Goal
----
Return the four-index array of chemists' two-electron integrals (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js in the orbital basis C for the density-density site-basis integrals (pq|rs)_site = delta_pq delta_rs V_pr. Raise ValueError if V and C are not both (K, K) matrices.

```python
def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    '''Two-electron integrals (pq|rs) in an orbital basis for the density-density model.

    Parameters
    ----------
    V : np.ndarray
        Site-basis interaction matrix (K, K).
    C : np.ndarray
        Orbital coefficients (K, K), one orbital per column.

    Returns
    -------
    Vm : np.ndarray
        Array (K, K, K, K) of chemists' integrals (pq|rs) in the orbital basis.

    Raises
    ------
    ValueError
        If V and C are not both (K, K) matrices.
    '''
    return Vm
```

### Step 4

pccd_amplitudes

Goal
----
Return the pCCD amplitudes c (P, K - P) in the given orbital basis (orbitals 0..P-1 doubly occupied in the reference). pCCD (pair coupled cluster doubles) wavefunction |Psi> = exp(T)|Phi0> with T = sum_{i occ, a vir} c_ia P+_a P_i, P+_p = a+_{p alpha} a+_{p beta} the pair creation operator, |Phi0> the closed-shell determinant of the first P orbitals of the given basis; amplitudes c (P, K - P) solve the projected equations <Phi_i^a| exp(-T) H exp(T) |Phi0> = 0 for every pair-excited determinant |Phi_i^a> = P+_a P_i |Phi0>. The Hamiltonian is H = sum_pq h_pq E_pq + (1/2) sum_pqrs (pq|rs) (E_pq E_rs - delta_qr E_ps) with E_pq the spin-summed excitation operator. Solve the equations to a residual below 1e-12 in every component, starting from c = 0 (the solution continuously connected to the reference). Raise ValueError if the shapes are inconsistent, n_pairs is outside [1, K - 1], or the equations do not converge.

```python
def pccd_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", n_pairs: int) -> "np.ndarray":
    '''Pair coupled cluster doubles amplitudes from the projected equations.

    Parameters
    ----------
    hm : np.ndarray
        One-body matrix (K, K) in the orbital basis.
    Vm : np.ndarray
        Chemists' integrals (pq|rs), array (K, K, K, K), orbital basis.
    n_pairs : int
        Number of occupied (doubly filled) orbitals P, the first P orbitals of the basis.

    Returns
    -------
    c : np.ndarray
        Amplitudes c_ia, array (P, K - P), row i occupied, column a virtual.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_pairs is outside [1, K - 1], or the equations do not converge.
    '''
    return c
```

### Step 5

pccd_lambda_amplitudes

Goal
----
Return the Lagrange multipliers lambda (P, K - P) of the pCCD energy functional L = <Phi0| exp(-T) H exp(T) |Phi0> + sum_{ia} lambda_ia <Phi_i^a| exp(-T) H exp(T) |Phi0>, i.e. the solution of dL/dc_ia = 0 at the converged amplitudes c (the pCCD Lambda equations, a linear system for lambda). pCCD (pair coupled cluster doubles) wavefunction |Psi> = exp(T)|Phi0> with T = sum_{i occ, a vir} c_ia P+_a P_i, P+_p = a+_{p alpha} a+_{p beta} the pair creation operator, |Phi0> the closed-shell determinant of the first P orbitals of the given basis; amplitudes c (P, K - P) solve the projected equations <Phi_i^a| exp(-T) H exp(T) |Phi0> = 0 for every pair-excited determinant |Phi_i^a> = P+_a P_i |Phi0>. Raise ValueError if the shapes are inconsistent or c does not satisfy the pCCD equations to 1e-8.

```python
def pccd_lambda_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    '''Lagrange multipliers (de-excitation amplitudes) of the pCCD energy functional.

    Parameters
    ----------
    hm : np.ndarray
        One-body matrix (K, K) in the orbital basis.
    Vm : np.ndarray
        Chemists' integrals (pq|rs), array (K, K, K, K).
    c : np.ndarray
        Converged pCCD amplitudes (P, K - P).

    Returns
    -------
    lam : np.ndarray
        Multipliers lambda_ia, array (P, K - P).

    Raises
    ------
    ValueError
        If the shapes are inconsistent or c does not satisfy the pCCD equations (residual above 1e-8).
    '''
    return lam
```

### Step 6

pccd_response_rdms

Goal
----
Return the response reduced density matrices of the pCCD wavefunction with amplitudes c and multipliers lam, per spin (alpha electrons), as one array (3, K, K) in the orbital basis (orbitals 0..P-1 occupied in the reference): rdms[0] the diagonal 1-RDM diag(gamma_p); rdms[1][p, q] = Gamma^{p qbar}_{p qbar} = <a+_{p alpha} a+_{q beta} a_{q beta} a_{p alpha}> (equal to the same-spin element Gamma^{pq}_{pq} for p != q, with the diagonal set to gamma_p); rdms[2][p, q] = Gamma^{p pbar}_{q qbar} = <a+_{p alpha} a+_{p beta} a_{q beta} a_{q alpha}> (pair transfer q -> p). Response (Lagrangian) density matrices: gamma_pq = <Phi0| (1 + Lambda) exp(-T) a+_{p alpha} a_{q alpha} exp(T) |Phi0> and Gamma^{pq}_{rs} = <Phi0| (1 + Lambda) exp(-T) a+_p a+_q a_s a_r exp(T) |Phi0> (spin labels as indicated), with the de-excitation operator Lambda = sum_{ia} lambda_ia P+_i P_a; the 1-RDM is diagonal and the only non-zero 2-RDM blocks are Gamma^{pq}_{pq} = Gamma^{p qbar}_{p qbar} (same-orbital-pair, p != q) and the pair-transfer block Gamma^{p pbar}_{q qbar} (Gamma^{p pbar}_{p pbar} = gamma_p). Raise ValueError if c and lam are not 2-D arrays of the same shape.

```python
def pccd_response_rdms(c: "np.ndarray", lam: "np.ndarray") -> "np.ndarray":
    '''Response one- and two-particle reduced density matrices of pCCD (per spin).

    Parameters
    ----------
    c : np.ndarray
        pCCD amplitudes (P, K - P).
    lam : np.ndarray
        Lambda amplitudes (P, K - P).

    Returns
    -------
    rdms : np.ndarray
        Array (3, K, K): rdms[0] = diag(gamma_p), rdms[1][p, q] = Gamma^{p qbar}_{p qbar} (= Gamma^{pq}_{pq}; diagonal gamma_p), rdms[2][p, q] = Gamma^{p pbar}_{q qbar}.

    Raises
    ------
    ValueError
        If c and lam are not 2-D arrays of the same shape.
    '''
    return rdms
```

### Step 7

generalized_fock

Goal
----
Return the generalised Fock matrix (orbital Lagrangian) of pCCD, F[p, q] = -<Phi0| (1 + Lambda) exp(-T) a+_{q alpha} [H, a_{p alpha}] exp(T) |Phi0> (alpha electron removed; H = sum h_pq E_pq + (1/2) sum (pq|rs) (E_pq E_rs - delta_qr E_ps)), evaluated from the per-spin response density matrices rdms of the preceding step, with the pair-transfer block entering only through its symmetrised form (Gamma^{p pbar}_{q qbar} + Gamma^{q qbar}_{p pbar}) / 2. Raise ValueError if the shapes are inconsistent.

```python
def generalized_fock(hm: "np.ndarray", Vm: "np.ndarray", rdms: "np.ndarray") -> "np.ndarray":
    '''Generalised Fock matrix of pCCD from the response density matrices.

    Parameters
    ----------
    hm : np.ndarray
        One-body matrix (K, K) in the orbital basis.
    Vm : np.ndarray
        Chemists' integrals (pq|rs), array (K, K, K, K).
    rdms : np.ndarray
        Response density matrices (3, K, K) in the layout of the preceding step.

    Returns
    -------
    F : np.ndarray
        Generalised Fock matrix F (K, K), element F[p, q].

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    '''
    return F
```

### Step 8

oo_pccd_orbitals

Goal
----
Return the orbitals C (K, K) that minimise the pCCD energy E(C) = <Phi0| exp(-T) H exp(T) |Phi0> (amplitudes re-solved from the projected equations at every orbital set; the first P columns doubly occupied in the reference) over all real orthogonal rotations of the starting orbitals C0, for the Hamiltonian with site-basis one-body matrix h and density-density integrals (pq|rs) = delta_pq delta_rs V_pr. Occupied-occupied, occupied-virtual and virtual-virtual rotations are all non-redundant. Converge until every component of the orbital gradient of the energy functional (dE/dkappa_pq for the rotation generator kappa, obtained from the generalised Fock matrix of the preceding step) is below 1e-10 in magnitude. Return the pCCD natural orbitals of the minimum: the columns ordered by descending per-spin occupation number of the response 1-RDM (the 1-RDM is diagonal in this basis). Column phase convention: in every column the first component with |c| > 1e-8 is positive. Raise ValueError if the shapes are inconsistent, n_pairs is outside [1, K - 1], or the optimisation does not converge.

```python
def oo_pccd_orbitals(h: "np.ndarray", V: "np.ndarray", C0: "np.ndarray", n_pairs: int) -> "np.ndarray":
    '''Variationally orbital-optimised pCCD: the natural orbitals that minimise the pCCD energy.

    Parameters
    ----------
    h : np.ndarray
        Site-basis one-body matrix (K, K).
    V : np.ndarray
        Site-basis interaction matrix (K, K).
    C0 : np.ndarray
        Starting orbitals (K, K), e.g. the RHF orbitals.
    n_pairs : int
        Number of electron pairs P.

    Returns
    -------
    C : np.ndarray
        Optimised natural orbitals (K, K), columns sorted by descending occupation number, phase-fixed.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_pairs is outside [1, K - 1], or the optimisation does not converge.
    '''
    return C
```

### Step 9

first_ionization_potential

Goal
----
Orchestrator. Return the first ionisation potential of the half-filled open PPP chain (K = len(eps_site) sites, K electrons, P = K/2 pairs) from the extended Koopmans' theorem built on orbital-optimised pCCD: build the model of step 1, take the RHF orbitals, optimise the orbitals (step 8, started from the RHF orbitals), and in the resulting natural-orbital basis obtain the amplitudes, the multipliers, the response density matrices and the generalised Fock matrix F of the earlier steps; discard every natural orbital whose per-spin occupation gamma_p is below cutoff; symmetrise F on the kept orbitals, F_s = (F + F^T)/2, transform F' = gamma^{-1/2} F_s gamma^{-1/2} with gamma the diagonal 1-RDM on the kept orbitals, and diagonalise F'. The ionisation potentials are the negatives of those eigenvalues whose unit-norm eigenvectors of F' carry more than half of their squared norm on the P most strongly occupied natural orbitals; return the smallest of them. Call the earlier step functions rather than reimplementing them. Raise ValueError if the number of sites is odd or below 4, if cutoff is outside [0, 1), or if no eigenvector of occupied character exists.

```python
def first_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", cutoff: float) -> float:
    '''First ionisation potential of the half-filled PPP chain from the extended Koopmans' theorem on orbital-optimised pCCD.

    Parameters
    ----------
    t : float
        Hopping scale, t > 0.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        On-site repulsion, U > 0.
    kappa : float
        Ohno screening parameter, kappa > 0.
    eps_site : np.ndarray
        Site energies, 1-D array of even length K >= 4.
    cutoff : float
        Natural orbitals with per-spin occupation below this value are discarded, 0 <= cutoff < 1.

    Returns
    -------
    ip : float
        The first (smallest) ionisation potential in units of t, as a Python float.

    Raises
    ------
    ValueError
        If the number of sites is odd or below 4, cutoff is outside [0, 1), or no eigenvector of occupied character exists.
    '''
    return ip
```
