# Chemistry-Quantum_Chemistry-19

## Background

Photoemission and electron attachment are described, in many-body perturbation theory, by the one-particle Green's function and its self-energy. Hedin's equations give the self-energy exactly in terms of the screened Coulomb interaction W and a vertex function; the GW approximation keeps the lowest-order vertex and has become, over the last decade, a genuinely competitive quantum chemistry method for valence and core ionization energies, electron affinities and band gaps. Its known weaknesses (multireference situations, strongly correlated materials, satellite structure) have driven a large effort to include vertex corrections, which come in two kinds: inner-vertex corrections improve the polarizability and therefore W itself beyond the direct random-phase approximation, while outer-vertex corrections add terms of higher order in W to the self-energy. The leading outer-vertex correction is the second-order term in W, the G3W2 self-energy, and its affordable relatives keep only its bare and singly screened pieces (the second-order screened exchange, 2SOSEX). A growing body of evidence shows that the two kinds of corrections should be balanced against each other, because the success of plain GW rests partly on a cancellation between the two.

Truncated perturbative self-energies have a structural problem that is independent of their accuracy. The exact self-energy has a sum-over-states (spectral) representation whose residues are positive semidefinite matrices; this guarantees a positive spectral function. The G3W2 and 2SOSEX self-energies do not possess this form, and negative spectral weight is observed in practice. Two strategies repair this. Stefanucci and co-workers showed that a finite number of additional diagrams can restore positive semidefiniteness, a procedure applied recently to 2SOSEX by Bruneval, Förster and Pavlyukh. The algebraic diagrammatic construction (ADC) of Schirmer and Cederbaum goes further: it rewrites a perturbative self-energy, through infinite resummations, in the exact analytic form U^dagger (omega - K - C)^(-1) U, in which a coupling block connects the one-particle space to excited configurations, K carries their zeroth-order energies and C their couplings. Solving the Dyson equation with such a self-energy is equivalent to diagonalizing a Hermitian effective Hamiltonian whose eigenvalues are the quasiparticle and satellite energies, and the ADC(n) hierarchy for the bare Coulomb interaction has long been a workhorse for ionization spectra.

The work behind this task transfers the ADC idea from the bare interaction to the screened one. The GW self-energy is already of sum-over-states form; its configurations are two-hole-one-particle and two-particle-one-hole states whose energies are dressed by the direct-RPA excitation energies and whose couplings are the GW effective integrals. Applying the ADC identification to the successive terms of the G3W2 self-energy, written with dressed poles only, produces a hierarchy of schemes: ADC-2SOSEX corrects the couplings and is identical to the positive-semidefinite GW+2SOSEX self-energy; ADC(3)-G3W2 adds a coupling block within the configuration space, which turns the construction into a genuine infinite resummation; and the complete ADC-G3W2 scheme adds cubic couplings together with three-hole-two-particle and three-particle-two-hole configurations labelled by two RPA modes. All members of the hierarchy are Hermitian, positive semidefinite by construction, and interpretable as nonperturbative resummations of outer-vertex corrections, and they establish a formal bridge between the GW family and the conventional ADC schemes.

On a set of 58 valence ionization potentials of small molecules with full configuration interaction reference values, the authors found that the plain GW scheme remains the most accurate member of the family (mean absolute error 0.38 eV), that the screened-exchange and third-order schemes increase the errors (0.62 and 0.65 eV), and that the complete scheme partially recovers (0.54 eV), in line with the expectation that outer-vertex corrections alone overshoot until matching inner-vertex corrections are included. The method is formulated for any closed-shell reference and any set of two-electron integrals. Pi-electron model Hamiltonians of the Pariser-Parr-Pople type, with a single orbital per centre, on-site repulsion and Ohno-screened intersite interactions, offer a compact setting in which the entire construction can be carried out explicitly and compared with exact diagonalization, which is the setting used here.

## Problem

The GW approximation is the standard many-body route to charged excitations of molecules, and the first correction beyond it, the second-order self-energy in the screened interaction (G3W2), is known to break a basic analytic property of the exact self-energy: it is not positive semidefinite, so spectral functions can become negative. A recently proposed remedy recasts the G3W2 self-energy in the algebraic diagrammatic construction (ADC), which completes it through infinite resummations into a sum-over-states form. The outcome is a real symmetric effective Hamiltonian in which the one-particle space couples to two-hole-one-particle and two-particle-one-hole configurations labelled by an orbital and a direct-RPA mode and, in the complete scheme, to three-hole-two-particle and three-particle-two-hole configurations labelled by an orbital and an ordered pair of modes; the eigenvalues of that Hamiltonian are the quasiparticle and satellite energies, and the simplest member of the same family (no corrections at all) is the upfolded GW problem. Your task is to carry the complete construction through for a small conjugated chain and to report the vertex correction it makes to the first ionization potential relative to GW.

The chain is a Pariser-Parr-Pople model of six pi centres, numbered 0 to 5 and placed on a straight line, with the Hamiltonian H = sum over centres of eps_mu n_mu + sum over bonds of t (c_mu^dagger c_nu + h.c., both spins) + U sum over centres of n_mu_up n_mu_down + one half sum over ordered pairs of distinct centres of V_mu_nu (n_mu - 1)(n_nu - 1), where V_mu_nu is the Ohno interaction 14.397 / sqrt((14.397 / U)^2 + r_mu_nu^2) eV for a distance r in Angstrom. Use exactly this configuration:

- bond lengths between consecutive centres (Angstrom): 1.34, 1.47, 1.36, 1.46, 1.38
- hopping integrals on those bonds (eV): -2.23, -1.62, -2.02, -2.05, -2.10
- site energies eps_mu (eV): -14.2, -11.2, -10.5, -11.9, -10.4, -10.4
- on-site repulsion U = 14.0 eV
- six electrons in a closed-shell (spin-restricted) treatment, three doubly occupied orbitals

Start from the converged restricted Hartree-Fock determinant of this Hamiltonian, express everything in its canonical spatial orbitals, and take the screening from the singlet direct random-phase approximation (no exchange in the kernel, eigenvectors normalized as X^T X - Y^T Y = 1) through the spin-adapted GW effective integrals M(p, q, nu) = sqrt(2) sum over (i, a) of [bracket p a bar q i X(ia, nu) + bracket p i bar q a Y(ia, nu)] with two-electron integrals in Dirac notation. Build the upfolded GW Hamiltonian (the effective integrals as couplings, configuration energies eps_i - Omega_nu and eps_a + Omega_nu) and the complete ADC-G3W2 Hamiltonian, which in addition carries the second-order screened-exchange and the cubic corrections to the couplings, the configuration-coupling block inside each two-body sector, and the three-body sectors with their couplings to the orbitals and to the two-body configurations. Diagonalize each Hamiltonian once, without the diagonal approximation and without any regularization of energy denominators, and take the principal ionization potential of a scheme as minus the eigenvalue whose eigenvector carries the largest squared weight on the highest occupied orbital.

Report the vertex correction Delta = IP(ADC-G3W2) - IP(ADC-GW) for this chain, in eV, to four decimal places. Your final answer must be a single number: Delta.

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

ppp_fock_matrix

Goal
----
This step builds that Hamiltonian and solves the closed-shell restricted Hartree-Fock problem for it. The Fock matrix in the site basis is the one-body matrix plus the Coulomb term (the Ohno matrix applied to the vector of site populations, on the diagonal) minus one half of the elementwise product of the spin-summed density matrix with the Ohno matrix. Iterate to self-consistency with a tight threshold (change of the density matrix below 1e-10) so that the Fock matrix is reproducible to better than 1e-8 eV; plain iteration with averaging of successive density matrices converges for the chains of this task. The converged Fock matrix is the reference for everything that follows: its eigenvalues are the orbital energies and its eigenvectors the canonical orbitals.

```python
def ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    '''Converged restricted Hartree-Fock Fock matrix of a linear PPP chain.

    Parameters
    ----------
    bond_lengths : array_like of float, shape (n - 1,)
        Distances in Angstrom between consecutive centres along the line.
    hoppings : array_like of float, shape (n - 1,)
        Hopping integrals in eV between consecutive centres (bond k joins
        centres k and k + 1).
    site_energies : array_like of float, shape (n,)
        Site energies in eV of the n centres (before the background shift).
    hubbard_u : float
        On-site repulsion U in eV (also the on-site value of the Ohno matrix).
    n_occ : int
        Number of doubly occupied orbitals (the chain holds 2 n_occ electrons).

    Returns
    -------
    fock : np.ndarray, shape (n, n)
        Converged Fock matrix in the site basis, in eV.
        Raises ValueError if the array lengths are inconsistent, if any
        bond length is not positive, if hubbard_u is not positive, or if
        n_occ is not an integer between 1 and n.
    '''
    fock = np.zeros((len(site_energies), len(site_energies)))
    return fock
```

### Step 2

mo_eri_tensor

Goal
----
This step diagonalizes a converged Fock matrix, orders the canonical orbitals by increasing orbital energy, fixes the arbitrary sign of every orbital so that its coefficient of largest magnitude is positive, and returns the two-electron integrals in physicist (Dirac) notation, bracket p q bar r s, equal to the double sum over centres mu and nu of C(mu, p) C(nu, q) C(mu, r) C(nu, s) V(mu, nu). In this notation electron one carries the first and third index and electron two the second and fourth index, so that the chemist integral (p r | q s) and the Dirac integral bracket p q bar r s are the same number. Getting this index convention right is essential: the expressions of the later steps distinguish bracket i c bar k q from bracket i k bar c q, and for the PPP integrals these are different numbers.

```python
def mo_eri_tensor(fock, ohno):
    '''Two-electron integrals in the canonical orbital basis (Dirac notation).

    Parameters
    ----------
    fock : array_like of float, shape (n, n)
        Symmetric Fock matrix in the site basis, in eV.
    ohno : array_like of float, shape (n, n)
        Symmetric Ohno interaction matrix in the site basis (U on the
        diagonal), in eV.

    Returns
    -------
    eri : np.ndarray, shape (n, n, n, n)
        eri[p, q, r, s] = bracket p q bar r s in the canonical orbitals of
        fock, ordered by increasing orbital energy, each orbital phased so
        that its largest-magnitude site coefficient is positive.
        Raises ValueError if the matrices are not square, of the same size,
        symmetric within 1e-10, and finite.
    '''
    n = np.asarray(fock).shape[0]
    eri = np.zeros((n, n, n, n))
    return eri
```

### Step 3

drpa_excitation_energies

Goal
----
This step returns the dRPA excitation energies in increasing order. They are the "dressed" energies that enter every pole of the self-energy of this task: the two-hole-one-particle configurations sit at eps_i - Omega_nu, the two-particle-one-hole ones at eps_a + Omega_nu, and the three-body ones at eps_i - Omega_nu - Omega_mu and eps_a + Omega_nu + Omega_mu. Replacing the dRPA by its Tamm-Dancoff approximation (dropping B) changes all of them.

```python
def drpa_excitation_energies(orbital_energies, eri, n_occ):
    '''Singlet direct-RPA excitation energies of a closed-shell reference.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    eri : array_like of float, shape (n, n, n, n)
        Two-electron integrals bracket p q bar r s in Dirac notation, in eV.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).

    Returns
    -------
    omega : np.ndarray, shape (n_occ * (n - n_occ),)
        dRPA excitation energies in increasing order, in eV.
        Raises ValueError if the shapes are inconsistent, if n_occ is not an
        integer between 1 and n - 1, or if the pair problem has no real
        positive spectrum.
    '''
    n = len(orbital_energies)
    omega = np.zeros(int(n_occ) * (n - int(n_occ)))
    return omega
```

### Step 4

gw_effective_integrals

Goal
----
Each dRPA eigenvector is defined up to an overall sign, which propagates to M(p, q, nu) for that mode. Every physical quantity built later is bilinear in the integrals of a given mode, so this sign drops out. The tests of this step therefore compare sign-insensitive combinations; the oracle fixes the sign by making the largest-magnitude component of X + Y of each mode positive, and the modes are ordered by increasing excitation energy.

```python
def gw_effective_integrals(orbital_energies, eri, n_occ):
    '''GW effective integrals M(p, q, nu) built from the singlet dRPA modes.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    eri : array_like of float, shape (n, n, n, n)
        Two-electron integrals bracket p q bar r s in Dirac notation, in eV.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).

    Returns
    -------
    M : np.ndarray, shape (n, n, n_occ * (n - n_occ))
        M[p, q, nu] with the modes nu ordered by increasing dRPA excitation
        energy and each mode phased so that the largest-magnitude component
        of X + Y is positive, in eV.
        Raises ValueError on inconsistent shapes, non-finite input, or
        n_occ outside 1 .. n - 1.
    '''
    n = len(orbital_energies)
    M = np.zeros((n, n, int(n_occ) * (n - int(n_occ))))
    return M
```

### Step 5

sosex_coupling_block

Goal
----
The first step beyond GW adds the second-order screened exchange diagrams. In the ADC form they appear as a correction to the coupling block only, and the scheme that keeps this correction together with the GW blocks is the ADC-2SOSEX scheme; it is equivalent to the positive-semidefinite GW+2SOSEX self-energy. The correction to the 2h1p coupling of configuration (i, nu) with orbital q is a sum over occupied-virtual pairs (k, c) of two terms, each a GW effective integral of the mode nu times a bare two-electron integral, divided by a dressed denominator that combines the orbital gap eps_c - eps_k with plus or minus Omega_nu; the two terms differ in the index order of the effective integral and of the bare integral and in the sign of Omega_nu in the denominator. The 2p1h block is its particle-hole mirror image. The explicit expressions are Eqs. (16a) and (16b) of the source paper; this step implements them for one branch.

```python
def sosex_coupling_block(orbital_energies, eri, M, omega, n_occ, branch):
    '''Second-order screened-exchange correction to the ADC coupling block.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    eri : array_like of float, shape (n, n, n, n)
        Two-electron integrals bracket p q bar r s in Dirac notation, in eV.
    M : array_like of float, shape (n, n, n_modes)
        GW effective integrals M[p, q, nu], in eV.
    omega : array_like of float, shape (n_modes,)
        dRPA excitation energies, in eV, in the same order as the modes of M.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).
    branch : str
        "hole" for the 2h1p block (rows (i, nu), i occupied) or "particle"
        for the 2p1h block (rows (a, nu), a virtual).

    Returns
    -------
    U2 : np.ndarray, shape (n_rows * n_modes, n)
        The correction block, rows ordered with the orbital index outermost
        (increasing) and the mode index innermost, columns over all n
        orbitals, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    U2 = np.zeros((n_rows * len(omega), n))
    return U2
```

### Step 6

adc3_diagonal_block

Goal
----
The block is Hermitian and, for a real orbital basis, symmetric. Each of its elements is a sum over the orbitals of the opposite type (virtual c for the hole block, occupied k for the particle block) of a product of two GW effective integrals, one connecting the row orbital to c in the column mode and one connecting the column orbital to c in the row mode, each product divided by a dressed denominator that combines an orbital gap with one excitation energy, symmetrized over the two configurations with a factor one half. The explicit expressions are Eqs. (17a) and (17b) of the source paper; this step implements them for one branch.

```python
def adc3_diagonal_block(orbital_energies, M, omega, n_occ, branch):
    '''Configuration-coupling block C among the 2h1p (or 2p1h) configurations.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    M : array_like of float, shape (n, n, n_modes)
        GW effective integrals M[p, q, nu], in eV.
    omega : array_like of float, shape (n_modes,)
        dRPA excitation energies, in eV, in the same order as the modes of M.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).
    branch : str
        "hole" for the block among 2h1p configurations (i, nu) or
        "particle" for the block among 2p1h configurations (a, nu).

    Returns
    -------
    C : np.ndarray, shape (n_rows * n_modes, n_rows * n_modes)
        The coupling block, configurations ordered with the orbital index
        outermost (increasing) and the mode index innermost, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    C = np.zeros((n_rows * len(omega), n_rows * len(omega)))
    return C
```

### Step 7

third_order_coupling_block

Goal
----
Completing the full G3W2 self-energy within the ADC form requires the remaining pieces of its hole and particle branches, those that are cubic in the GW effective integrals. Part of them enters as a further correction to the coupling between the one-particle space and the 2h1p (or 2p1h) configurations. For the hole branch this correction to the coupling of configuration (i, nu) with orbital q is a sum over an extra dRPA mode mu and over pairs of orbitals of four distinct terms, each a product of three effective integrals (two of them carrying the extra mode mu, one carrying the configuration mode nu) divided by a product of two dressed denominators; three of the terms run over an occupied orbital k and a virtual orbital c, the fourth over two virtual orbitals c and d, and the terms carry different signs, a factor one half on the first, and denominators in which Omega_mu enters with either sign and in one case together with Omega_nu. The particle branch is the particle-hole mirror image with two occupied orbitals k and l in the last term. The explicit expressions are Eqs. (18a) and (18b) of the source paper; this step implements them for one branch.

```python
def third_order_coupling_block(orbital_energies, M, omega, n_occ, branch):
    '''Cubic (third) contribution to the ADC coupling block of one branch.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    M : array_like of float, shape (n, n, n_modes)
        GW effective integrals M[p, q, nu], in eV.
    omega : array_like of float, shape (n_modes,)
        dRPA excitation energies, in eV, in the same order as the modes of M.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).
    branch : str
        "hole" for the 2h1p block (rows (i, nu), i occupied) or "particle"
        for the 2p1h block (rows (a, nu), a virtual).

    Returns
    -------
    U3 : np.ndarray, shape (n_rows * n_modes, n)
        The cubic correction to the coupling block, rows ordered with the
        orbital index outermost (increasing) and the mode index innermost,
        columns over all n orbitals, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    U3 = np.zeros((n_rows * len(omega), n))
    return U3
```

### Step 8

three_body_row_block

Goal
----
Three new blocks come with these configurations: the direct coupling between the one-particle space and a three-body configuration, which is a single sum over the orbitals of the opposite type of a product of two effective integrals over a dressed denominator carrying the second mode (with a definite overall sign that differs between the hole and the particle branches); the coupling between a three-body configuration and a two-body configuration, which is a single effective integral of the second mode and is diagonal in the first mode; and the diagonal three-body energies. The explicit expressions are Eqs. (19), (20) and (21) of the source paper. This step returns, for one branch, the whole row of the effective Hamiltonian that belongs to the three-body configurations, as the horizontal concatenation [coupling to orbitals | coupling to two-body configurations | diagonal energies], with both configuration indices ordered orbital outermost, first mode, then second mode innermost.

```python
def three_body_row_block(orbital_energies, M, omega, n_occ, branch):
    '''Row block of the 3h2p (or 3p2h) configurations of the ADC Hamiltonian.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    M : array_like of float, shape (n, n, n_modes)
        GW effective integrals M[p, q, nu], in eV.
    omega : array_like of float, shape (n_modes,)
        dRPA excitation energies, in eV, in the same order as the modes of M.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).
    branch : str
        "hole" for the 3h2p configurations (i, nu, mu), "particle" for the
        3p2h configurations (a, nu, mu).

    Returns
    -------
    row : np.ndarray, shape (n_rows * n_modes**2, n + n_rows * n_modes + n_rows * n_modes**2)
        Horizontal concatenation of the coupling block to the n orbitals,
        the coupling block to the two-body configurations (p, lambda) of the
        same branch, and the diagonal block of three-body energies, in eV.
        Three-body configurations are ordered orbital outermost, then the
        first mode, then the second mode; two-body configurations orbital
        outermost, mode innermost.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    nm = len(omega)
    row = np.zeros((n_rows * nm * nm, n + n_rows * nm + n_rows * nm * nm))
    return row
```

### Step 9

adc_g3w2_vertex_correction

Goal
----
The eigenvalues below the Fermi level are ionization energies (with a minus sign): each eigenvector has a component on every orbital, and the principal ionization potential of the highest occupied orbital is minus the eigenvalue whose eigenvector carries the largest squared weight on that orbital. The quantity requested by the task is the vertex correction to this principal ionization potential, the ADC-G3W2 value minus the ADC-GW value. It isolates the effect of the second-order screened exchange and of its ADC completion (the cubic couplings and the three-body configurations); the intermediate schemes that stop at the screened-exchange couplings or at the configuration-coupling block move the ionization potential in the opposite direction to the complete scheme, so the sign of this correction is a sharp test of the full construction.

```python
def adc_g3w2_vertex_correction(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    '''Vertex correction to the principal ionization potential of a PPP chain.

    Parameters
    ----------
    bond_lengths : array_like of float, shape (n - 1,)
        Distances in Angstrom between consecutive centres along the line.
    hoppings : array_like of float, shape (n - 1,)
        Hopping integrals in eV between consecutive centres.
    site_energies : array_like of float, shape (n,)
        Site energies in eV of the n centres (before the background shift).
    hubbard_u : float
        On-site repulsion U in eV.
    n_occ : int
        Number of doubly occupied orbitals, between 1 and n - 1.

    Returns
    -------
    delta : float
        IP(ADC-G3W2) minus IP(ADC-GW) for the principal ionization of the
        highest occupied orbital (the eigenvalue of each effective
        Hamiltonian whose eigenvector has the largest squared weight on
        that orbital), in eV.
        Raises ValueError for inconsistent or non-physical input.
    '''
    delta = 0.0
    return delta
```
