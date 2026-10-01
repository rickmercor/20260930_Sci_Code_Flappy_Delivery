# Exact Coulomb correlations of an electron-hole pair in a spherical nanocrystal: the THz resonance at the confinement crossover

## Background

A semiconductor nanocrystal confines the electron and the hole it hosts within a few nanometres,
and the pair's internal excitations, transitions between confined levels of either carrier or
between exciton states, fall in the terahertz range. Optical pump and terahertz probe experiments
therefore look directly at the structure of the confined pair, and their interpretation needs the
positions and dipole moments of the terahertz resonances. Unlike interband optics, which sees the
pair energy as a whole, the terahertz field couples to the electron and to the hole with opposite
signs, so the two carriers respond separately and their responses interfere.

Two limits are classical. When the crystal is small compared with the exciton, confinement
dominates: each carrier occupies a particle-in-a-sphere state, the Coulomb attraction is a
correction, and the response shows an electron-like and a hole-like resonance. When the crystal is
large, the pair forms a Wannier exciton whose relative motion is hydrogen-like and whose centre of
mass is quantised by the crystal walls; only the internal transitions of the exciton respond. The
source of this task develops approximate descriptions of both limits from the same effective-mass
Hamiltonian, one expanding the pair in a few bare box states, the other separating the relative
and centre-of-mass motions with an exciton size that adapts to the confinement, and shows that the
Coulomb interaction shifts the resonances and moves dipole weight from one to the other even in
small crystals.

Between the limits neither description is exact, and the exact answer is a two-particle
eigenproblem with the full Coulomb interaction inside the sphere. Rotational symmetry makes it
tractable: the pair Hamiltonian is block diagonal in the total angular momentum, the Coulomb
interaction reduces to radial integrals of a multipole kernel that must be handled exactly across
its kink, and the dipole selection rules connect the ground block to a single excited block. With
the basis stated, the exact model yields the ground state, the resonance energies, the dipole
moments and the size at which the carriers cease to be independent, and each of the source's
approximations can be tested against it.

This task builds that exact model, evaluates the confined pair and its terahertz transitions from
the strongly confined regime to the exciton Bohr radius, compares them with the non-interacting
pair and with the source's two models, locates the exact independence radius, and asks for the
exact electron-like resonance energy in a crystal of one exciton Bohr radius.

## Problem

A semiconductor nanocrystal probed by optical pump and THz probe reveals the internal structure of
the electron-hole pair it confines: the transitions between confined pair states fall in the THz
range and the THz field couples to the electron and to the hole with opposite signs, so the response
carries separate electron-like and hole-like resonances whose energies and dipole moments are
renormalised by the Coulomb interaction. The source of this task treats a single pair in a hard
spherical nanocrystal with effective masses and a screened Coulomb attraction, and develops two
approximate descriptions: a strong-confinement model that expands the pair in bare
particle-in-a-sphere states and, in its numerical work, keeps only the 1s and 1p states of each
particle (a minimal two-state description of each angular-momentum block, and a 729-state
extended basis as a check), and a weak-confinement model that separates centre-of-mass and
relative motion with a centre-of-mass box shrunk by the size of the exciton, the size being set by
the relative coordinate itself. It shows that the Coulomb interaction blueshifts the resonances,
suppresses the hole-like peak and amplifies the electron-like one even in small crystals, derives
an analytic estimate of the radius below which the carriers may be treated as independent, and
argues that the two models connect smoothly. The exact diagonalisation of the pair Hamiltonian in
a large basis it calls formidable and leaves undone: its values in the crossover region between
the two models, where neither is exact, are the question of this task.

Your task is to build the exact model in a stated basis, evaluate the confined pair's ground state,
its THz transitions and their dipole moments across the crossover from strong confinement to the
exciton Bohr radius, compare with the non-interacting pair, with the source's minimal
strong-confinement model and with its weak-confinement model, locate the exact independence radius
and set it against the source's estimate, and report the exact electron-like THz resonance energy
of the pair in a crystal whose radius equals the exciton Bohr radius.

## Conventions that fix every number

A single spinless electron-hole pair in a spherical nanocrystal of radius A; each particle is
confined by an infinite hard wall (zero potential for r < A, infinite for r >= A); isotropic
effective masses m_e and m_h in units of the free electron mass; the Coulomb attraction
-e^2/(4 pi eps eps_0 |r_e - r_h|) screened by the relative permittivity eps and nothing else (no
dielectric mismatch, no exchange). Units: hbar^2/(2 m_0) = 0.03809985 eV nm^2 and
e^2/(4 pi eps_0) = 1.439964 eV nm; energies in eV, lengths in nm. Single-particle box states are
psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A) Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the
spherical Bessel function j_l (n = 1, 2, ...; l = 0, 1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2),
unit normalisation over the sphere and spherical harmonics in the Condon-Shortley phase convention.

The exact model is the eigenproblem of the two-particle Hamiltonian in the product basis of
single-particle states with n <= N_n and l <= L_max for each particle. It is block diagonal in the
total angular momentum L and in the parity (-1)^{l_e + l_h}; the coupled basis
|(n_e l_e)(n_h l_h) L M> = sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is
used, the ground state lying in the L = 0 even block and the dipole-active excited states in the
L = 1 odd block. Coulomb matrix elements follow from the multipole expansion
1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k / r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h) with the radial
integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e) r_e^2
R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k / r_>^{k+1} dr_e dr_h in nm^-1, exact to 1e-10 (the
kernel has a kink on r_e = r_h, so the inner integral is taken piecewise, by cumulative quadrature
or a Poisson-equation solve, never by a plain product rule), and the angular factor of the coupled
states, the sum over the projections m with the Clebsch-Gordan coefficients <l_e m l_h (-m)|L 0> of
both states of the products <l_e' m'| Y_kq |l_e m> <l_h' (-m')| Y_k,-q |l_h (-m)> (-1)^q with
q = m' - m.

THz response: the dipole operator is d = -e (r_e - r_h); for a field along z the transition dipoles
from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k> are d_k = <k| z_e - z_h |0> in
nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k - E_0 with peak
weights w_k = (E_k - E_0)|d_k|^2. The electron-like resonance is the eigenstate with the largest
weight and the hole-like resonance the largest-weight eigenstate below it in energy (for
m_e < m_h the electron transition lies higher). Derived quantities: the renormalisation factors
f_e = |d_e|^2/|d_bare|^2 and f_h = |d_h|^2/|d_bare|^2 relative to the non-interacting squared dipole,
the weight ratio w_h/w_e, the Coulomb shift (E_e - E_0) - E_e^bare of the electron-like resonance,
the source's coupling constant c_EC = |E_C| 4 pi eps eps_0 A/e^2 with E_C the off-diagonal element
between the two states of the minimal L = 1 block, and the independence radius A_10, the radius at
which f_e reaches 1.1, root-found on a stated bracket.

Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions have
energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole
|<1p_z| z |1s>|^2 = (int R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement
model, the exact model restricted to N_n = 1 and L_max = 1 (the L = 0 block spans |1s,1s> and the
L = 0 combination of |1p,1p>, the L = 1 block spans |(1s 1p)> and |(1p 1s)>); and the source's
weak-confinement model, the relative-motion radial equation
-(hbar^2/2 mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the
shrunken-exciton centre-of-mass term chi(r) = hbar^2 pi^2 / (2 M (A - rho(r))^2),
rho(r) = mu r / min(m_e, m_h), M = m_e + m_h, mu = m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu
(where rho reaches A and chi diverges), whose lowest l = 0 state (1s)_w and lowest l = 1 state (2p)_w
give the transition energy and the relative-coordinate squared dipole
|<2p_z| z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged
to 1e-9 eV (energies) or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the
reference's uniform radial grids (Simpson quadrature for the box integrals; a three-point
finite-difference eigenproblem for u = r R with u(0) = u(r_max) = 0 on N_g intervals, extrapolated to
zero step), and any quadrature or discretisation converged to the same accuracy is acceptable.

Design settings: m_e = 0.07, m_h = 0.13 (M = 0.2, m_e/M = 0.35), eps = 12.85, so that the exciton
Bohr radius a_X = 4 pi eps eps_0 hbar^2/(mu e^2) is 14.94 nm; basis N_n = 6, L_max = 5; grids
N_r = 2001, N_g = 1000; audit radii 5 and 10 nm followed by A_star = 15 nm; independence-radius
bracket [3, 9] nm at the level 1.1.

## Box states and Coulomb integrals

Tabulate the Bessel zeros z_{ln} for l = 0..L_max and n = 1..N_n to 1e-13. Evaluate the radial
Coulomb integrals R^k exactly, and report the 1s-1s direct integral at k = 0 in units of
e^2/(4 pi eps eps_0 A).

## The exact pair and its THz transitions

Assemble the L = 0 and L = 1 blocks and diagonalise them; return the lowest eigenvalues of each
block and, for the L = 1 eigenstates, the transition energies, squared dipoles and weights from the
ground state. Explain why the field's opposite coupling to the two carriers lets Coulomb mixing
suppress one resonance and amplify the other, and why for equal masses the lower resonance loses
its dipole entirely.

## Reference models

Compute the non-interacting transitions and dipoles, the source's minimal model with its two-state
blocks, its coupling constant E_C and c_EC, and its weak-confinement model. State how the
weak-confinement transition energy at A_star compares with the exact electron-like resonance and
explain the origin of the difference, including the strongly confined limit of the ratio of the
weak-confinement transition energy to the non-interacting electron transition (the weak-confinement
model with its Coulomb term omitted).

## Independence radius and audit

Root-find the exact independence radius A_10 on the bracket, compare it with the source's estimate
3 pi hbar^2 eps eps_0 (1/m_e - 1/m_h)/e^2, and explain why that estimate, which keeps only the
excited-state mixing of the two-state model, falls short of the exact value. Run the complete chain
and assemble the audit table with one row per audit radius followed by the A_star row and the
columns [A, E_0, E_e - E_0, |d_e|^2, E_h - E_0, |d_h|^2, w_h/w_e, E_e^bare, E_h^bare, |d_bare|^2,
minimal-model E_e - E_0, minimal-model E_h - E_0, minimal |d_e|^2, minimal |d_h|^2,
weak-confinement transition energy, weak-confinement |d|^2, f_e, f_h, Coulomb shift of the
electron-like resonance], and a head row [E_e - E_0 at A_star, its Coulomb shift, w_h/w_e at
A_star, f_e at A_star, A_10, c_EC at A_star, a_X, the source's estimate of the independence radius,
A_star, N_n, L_max, N_r, N_g, number of audit radii, the kinetic splitting z_11^2 - z_10^2 (the
coefficient of 1/A^2), the 1s-1s direct Coulomb integral in units of e^2/(4 pi eps eps_0 A), the
strongly confined limit of the ratio of the weak-confinement transition energy to the non-interacting
electron transition (the weak-confinement model with the Coulomb term omitted, evaluated at A_star)
and the ratio of the weak-confinement transition energy to the exact electron-like resonance at
A_star].

## What to report

Report, as the final answer, the exact electron-like THz resonance energy E_e - E_0 of the pair in
eV at A_star = 15 nm with the design settings, to six significant figures.

Your reasoning should also report, as evidence that the chain was executed: at A_star the exact
ground-state energy, the electron-like squared dipole and renormalisation factor f_e, the hole-like
transition energy, squared dipole and f_h, the weight ratio w_h/w_e, and the Coulomb shift of the
electron-like resonance in eV and in THz; the non-interacting electron and hole transition energies
and squared dipole at A_star; the minimal-model transition energies and squared dipoles at A_star;
c_EC at A_star and the 1s-1s direct integral in units of e^2/(4 pi eps eps_0 A); the
weak-confinement transition energy and squared dipole at A_star, the ratio of that energy to the
exact electron-like resonance at A_star, at 10 nm and at 5 nm, and the strongly confined limit of
the ratio of the weak-confinement transition energy to the non-interacting electron transition; the
exact independence radius A_10 and the source's estimate; the electron-like transition energy, f_e
and w_h/w_e at 10 nm and at 5 nm; the value E_e - E_0 at A_star that the source's extended basis of
three radial and three angular states per particle (N_n = 3, L_max = 2) gives, and how far the
design basis is from convergence (state the (N_n, L_max) = (8, 6) value of E_e - E_0 at A_star).
Say, in a sentence or two, why Coulomb mixing suppresses the hole-like and amplifies the
electron-like dipole and why the lower resonance vanishes for equal masses; why the source's
estimate of the independence radius falls short of the exact value; how the weak-confinement
transition energy at A_star compares with the exact electron-like resonance and where the difference
comes from; and why the exact quantities are defined at a stated basis and to what accuracy they are
converged. Report, with a citation, the source's numerical value of the two-state coupling constant
E_C in units of e^2/(eps A) and of the kinetic splitting k_1p^2 - k_1s^2 in units of 1/A^2; its
independence-radius estimates and exciton Bohr radii for m_e/M = 0.25, 0.35 and 0.45; its statement
on how the ratio of the two peaks responds to an extension of the basis beyond the minimal one; and
its statement on how the resonance of the strong-confinement model moves relative to the
weak-confinement result when the basis is enlarged; and set each against your exact results,
including the exact A_10 for m_e/M = 0.45, whether the ten percent level is reached below 25 nm for
m_e/M = 0.25, the weight ratio w_h/w_e at A_star in the minimal, the 729-state and the design basis,
and the sequence of E_e - E_0 at A_star over the minimal, 729-state, design and (8, 6) bases.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

box_spectrum

Goal
----
Tabulates the first N_n positive zeros z_{ln} of the spherical Bessel functions j_l for l = 0..L_max, each to 1e-13 (row l, column n - 1), which fix the single-particle box states of a particle confined in the sphere of radius A: wavevectors z_{ln}/A and kinetic energies hbar^2 z_{ln}^2/(2 m A^2).

```python
def box_spectrum(A: float, N_n: int, L_max: int) -> "np.ndarray":
    """Tabulates the first N_n positive zeros z_{ln} of the spherical Bessel functions j_l for l = 0..L_max, each to
    1e-13 (row l, column n - 1), which fix the single-particle box states of a particle confined in the sphere of
    radius A: wavevectors z_{ln}/A and kinetic energies hbar^2 z_{ln}^2/(2 m A^2).

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).

    Returns:
        A float64 array of shape (L_max + 1, N_n) holding the Bessel zeros z_{ln}.

    Raises:
        ValueError: if A <= 0, N_n < 1 or L_max < 0.
    """
    return None
```

### Step 2

coulomb_integral

Goal
----
Evaluates the radial two-particle Coulomb integral R^k[(ne2 le2)(ne1 le1); (nh2 lh2)(nh1 lh1)] = int_0^A int_0^A R_{ne2 le2}(r_e) R_{ne1 le1}(r_e) r_e^2 R_{nh2 lh2}(r_h) R_{nh1 lh1}(r_h) r_h^2 r_<^k / r_>^{k+1} dr_e dr_h in nm^-1 for the normalised box radial functions of the sphere of radius A, exact to 1e-10 (r_< and r_> are the smaller and the larger of r_e and r_h).

```python
def coulomb_integral(A: float, k: int, ne2: int, le2: int, ne1: int, le1: int, nh2: int, lh2: int, nh1: int, lh1: int, N_r: int) -> float:
    """Evaluates the radial two-particle Coulomb integral R^k[(ne2 le2)(ne1 le1); (nh2 lh2)(nh1 lh1)] = int_0^A int_0^A
    R_{ne2 le2}(r_e) R_{ne1 le1}(r_e) r_e^2 R_{nh2 lh2}(r_h) R_{nh1 lh1}(r_h) r_h^2 r_<^k / r_>^{k+1} dr_e dr_h in
    nm^-1 for the normalised box radial functions of the sphere of radius A, exact to 1e-10 (r_< and r_> are the
    smaller and the larger of r_e and r_h).

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        k: int, the multipole order of the Coulomb kernel (k >= 0).
        ne2: int, the radial quantum number (n >= 1) of the electron state R_{ne2 le2}.
        le2: int, the angular momentum (l >= 0) of the electron state R_{ne2 le2}.
        ne1: int, the radial quantum number (n >= 1) of the electron state R_{ne1 le1}.
        le1: int, the angular momentum (l >= 0) of the electron state R_{ne1 le1}.
        nh2: int, the radial quantum number (n >= 1) of the hole state R_{nh2 lh2}.
        lh2: int, the angular momentum (l >= 0) of the hole state R_{nh2 lh2}.
        nh1: int, the radial quantum number (n >= 1) of the hole state R_{nh1 lh1}.
        lh1: int, the angular momentum (l >= 0) of the hole state R_{nh1 lh1}.
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float, the radial integral R^k in nm^-1.

    Raises:
        ValueError: if A <= 0, k < 0, a quantum number is invalid (n >= 1, l >= 0 required) or N_r is not an odd
        integer of at least 201.
    """
    return None
```

### Step 3

pair_levels

Goal
----
Returns the m lowest eigenvalues (eV, ascending) of the exact two-particle Hamiltonian in the coupled basis of total angular momentum L (M = 0) and parity (-1)^L, built from the single-particle box states with n <= N_n and l <= L_max of each particle: the kinetic energies on the diagonal and the Coulomb matrix -e^2/(4 pi eps eps_0) sum_k (4 pi/(2k + 1)) A_k R^k, where A_k is the angular factor of the coupled states, the sum over the projections m with the Clebsch-Gordan coefficients <l_e m l_h (-m)|L 0> of both states of the products <l_e' m'| Y_kq |l_e m> <l_h' (-m')| Y_k,-q |l_h (-m)> (-1)^q with q = m' - m, and R^k the radial integral of the previous step.

```python
def pair_levels(A: float, me: float, mh: float, eps: float, N_n: int, L_max: int, L: int, N_r: int, m: int) -> "np.ndarray":
    """Returns the m lowest eigenvalues (eV, ascending) of the exact two-particle Hamiltonian in the coupled basis of
    total angular momentum L (M = 0) and parity (-1)^L, built from the single-particle box states with n <= N_n and l
    <= L_max of each particle: the kinetic energies on the diagonal and the Coulomb matrix -e^2/(4 pi eps eps_0) sum_k
    (4 pi/(2k + 1)) A_k R^k, where A_k is the angular factor of the coupled states, the sum over the projections m
    with the Clebsch-Gordan coefficients <l_e m l_h (-m)|L 0> of both states of the products <l_e' m'| Y_kq |l_e m>
    <l_h' (-m')| Y_k,-q |l_h (-m)> (-1)^q with q = m' - m, and R^k the radial integral of the previous step.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        L: int, the total angular momentum of the block: 0 (the even ground-state block) or 1 (the odd dipole-active
            block).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        m: int, the number of lowest eigenstates to return, from 1 to the size of the block.

    Returns:
        A float64 array of shape (m,), the m lowest eigenvalues of the L block in eV.

    Raises:
        ValueError: for a non-positive A, mass or eps, for L not in {0, 1}, for N_n < 1 or L_max < 1, for an invalid
        N_r, or for m outside 1..(size of the block).
    """
    return None
```

### Step 4

thz_transitions

Goal
----
Diagonalises the L = 0 and L = 1 blocks of the exact model and returns, for the m lowest L = 1 eigenstates |k> in ascending energy, the rows [E_k - E_0, |d_k|^2, (E_k - E_0)|d_k|^2, E_0] with E_0 the ground-state energy of the L = 0 block, d_k = <k| z_e - z_h |0> the transition dipole in nm (the electron and hole dipole contributions enter with opposite signs, and the coupled-basis matrix elements of z_e and z_h involve the radial integrals int R_{n'l'} R_{nl} r^3 dr and the angular factors <l' m| cos theta |l m> = sqrt(((l_< + 1)^2 - m^2)/((2 l_< + 1)(2 l_< + 3))) for l' = l +- 1), and (E_k - E_0)|d_k|^2 the resonance weight.

```python
def thz_transitions(A: float, me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, m: int) -> "np.ndarray":
    """Diagonalises the L = 0 and L = 1 blocks of the exact model and returns, for the m lowest L = 1 eigenstates |k> in
    ascending energy, the rows [E_k - E_0, |d_k|^2, (E_k - E_0)|d_k|^2, E_0] with E_0 the ground-state energy of the L
    = 0 block, d_k = <k| z_e - z_h |0> the transition dipole in nm (the electron and hole dipole contributions enter
    with opposite signs, and the coupled-basis matrix elements of z_e and z_h involve the radial integrals int
    R_{n'l'} R_{nl} r^3 dr and the angular factors <l' m| cos theta |l m> = sqrt(((l_< + 1)^2 - m^2)/((2 l_< + 1)(2
    l_< + 3))) for l' = l +- 1), and (E_k - E_0)|d_k|^2 the resonance weight.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        m: int, the number of lowest eigenstates to return, from 1 to the size of the block.

    Returns:
        A float64 array of shape (m, 4) with rows [E_k - E_0 (eV), |d_k|^2 (nm^2), weight (eV nm^2), E_0 (eV)].

    Raises:
        ValueError: for invalid model arguments (as in the previous step) or for m outside 1..(size of the L = 1
        block).
    """
    return None
```

### Step 5

bare_response

Goal
----
Returns [E_e, E_h, |d_e|^2, |d_h|^2] for the non-interacting electron-hole pair in the sphere: the 1s -> 1p transition energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) of the electron and of the hole (eV) and their squared dipoles |<1p_z| z |1s>|^2 = (int R_{11} R_{10} r^3 dr)^2/3 (nm^2), exact to 1e-10.

```python
def bare_response(A: float, me: float, mh: float, N_r: int) -> "np.ndarray":
    """Returns [E_e, E_h, |d_e|^2, |d_h|^2] for the non-interacting electron-hole pair in the sphere: the 1s -> 1p
    transition energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) of the electron and of the hole (eV) and their squared
    dipoles |<1p_z| z |1s>|^2 = (int R_{11} R_{10} r^3 dr)^2/3 (nm^2), exact to 1e-10.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float64 array of shape (4,), [E_e, E_h, |d_e|^2, |d_h|^2].

    Raises:
        ValueError: for a non-positive A or mass, or for an invalid N_r.
    """
    return None
```

### Step 6

minimal_scr

Goal
----
Evaluates the source's minimal strong-confinement model, the exact model restricted to the single-particle states 1s and 1p (N_n = 1, L_max = 1), and returns [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC], where the L = 0 block spans |1s,1s> and |(1p 1p) L = 0>, the L = 1 block spans |(1s 1p)> and |(1p 1s)>, E_h and E_e are its lower and higher eigenvalues with their squared transition dipoles from the ground state, E_C (eV, negative) is the off-diagonal Coulomb element between the two L = 1 basis states and c_EC = |E_C| 4 pi eps eps_0 A/e^2 its dimensionless magnitude.

```python
def minimal_scr(A: float, me: float, mh: float, eps: float, N_r: int) -> "np.ndarray":
    """Evaluates the source's minimal strong-confinement model, the exact model restricted to the single-particle states
    1s and 1p (N_n = 1, L_max = 1), and returns [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC], where the L
    = 0 block spans |1s,1s> and |(1p 1p) L = 0>, the L = 1 block spans |(1s 1p)> and |(1p 1s)>, E_h and E_e are its
    lower and higher eigenvalues with their squared transition dipoles from the ground state, E_C (eV, negative) is
    the off-diagonal Coulomb element between the two L = 1 basis states and c_EC = |E_C| 4 pi eps eps_0 A/e^2 its
    dimensionless magnitude.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.

    Returns:
        A float64 array of shape (7,), [E_0, E_h - E_0, E_e - E_0, |d_h|^2, |d_e|^2, E_C, c_EC].

    Raises:
        ValueError: for a non-positive A, mass or eps, or for an invalid N_r.
    """
    return None
```

### Step 7

wcr_model

Goal
----
Solves the source's weak-confinement model and returns [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2]: the lowest l = 0 and l = 1 eigenvalues (eV) of the relative-motion radial equation with the shrunken-exciton centre-of-mass term chi(r) on 0 < r < r_max, and the relative-coordinate squared dipole (int R_2p R_1s r^3 dr)^2/3 in nm^2, each converged to 1e-9 eV or 1e-10 nm^2 (the reference discretises u = r R with three-point finite differences on N_g, 2 N_g and 4 N_g intervals and extrapolates the O(h^2) and O(h^4) errors to zero; the potential diverges as (r_max - r)^-2 at the outer wall, which any method must respect). With eps infinite the Coulomb term is omitted, which gives the model's strongly confined limit, in which the transition energy scales as 1/A^2.

```python
def wcr_model(A: float, me: float, mh: float, eps: float, N_g: int) -> "np.ndarray":
    """Solves the source's weak-confinement model and returns [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2]: the lowest
    l = 0 and l = 1 eigenvalues (eV) of the relative-motion radial equation with the shrunken-exciton centre-of-mass
    term chi(r) on 0 < r < r_max, and the relative-coordinate squared dipole (int R_2p R_1s r^3 dr)^2/3 in nm^2, each
    converged to 1e-9 eV or 1e-10 nm^2 (the reference discretises u = r R with three-point finite differences on N_g,
    2 N_g and 4 N_g intervals and extrapolates the O(h^2) and O(h^4) errors to zero; the potential diverges as (r_max
    - r)^-2 at the outer wall, which any method must respect). With eps infinite the Coulomb term is omitted, which
    gives the model's strongly confined limit, in which the transition energy scales as 1/A^2.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Args:
        A: float, the nanocrystal radius in nm (positive).
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive); an infinite value omits
            the Coulomb term.
        N_g: int, the number of uniform intervals of the finite-difference grid on [0, r_max] at the coarsest
            Richardson level; at least 500.

    Returns:
        A float64 array of shape (4,), [E_(1s)w, E_(2p)w, E_(2p)w - E_(1s)w, |d_w|^2].

    Raises:
        ValueError: for a non-positive A, mass or eps, or for N_g < 500.
    """
    return None
```

### Step 8

independence_radius

Goal
----
Finds by root finding (to 1e-10 nm) the radius A_10 in [A_lo, A_hi] at which the electron-like renormalisation factor f_e(A) = |d_e|^2/|d_bare|^2 of the exact model with the basis (N_n, L_max) reaches the given level (1.1 for a ten percent enhancement), where d_e belongs to the largest-weight L = 1 resonance and d_bare is the non-interacting 1s -> 1p dipole of the same sphere.

```python
def independence_radius(me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, A_lo: float, A_hi: float, level: float) -> float:
    """Finds by root finding (to 1e-10 nm) the radius A_10 in [A_lo, A_hi] at which the electron-like renormalisation
    factor f_e(A) = |d_e|^2/|d_bare|^2 of the exact model with the basis (N_n, L_max) reaches the given level (1.1 for
    a ten percent enhancement), where d_e belongs to the largest-weight L = 1 resonance and d_bare is the
    non-interacting 1s -> 1p dipole of the same sphere.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Derived quantities: the renormalisation factors f_e = |d_e|^2/|d_bare|^2 and f_h = |d_h|^2/|d_bare|^2 of the
    electron-like and hole-like resonances relative to the non-interacting squared dipole, the weight ratio w_h/w_e
    (hole-like over electron-like), the Coulomb shift of the electron-like resonance relative to the non-interacting
    electron transition, the source's coupling constant c_EC = |E_C| 4 pi eps eps_0 A/e^2 with E_C the off-diagonal
    element between the two states of the minimal L = 1 block, and the independence radius A_10, the radius at which
    f_e reaches the level 1 + 0.1 (root-found on a stated bracket).

    Args:
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        A_lo: float, the lower end of the bracket in nm on which the root is sought (0 < A_lo < A_hi).
        A_hi: float, the upper end of the bracket in nm.
        level: float, the target renormalisation factor f_e (above 1; 1.1 for a ten percent enhancement).

    Returns:
        A float, the independence radius A_10 in nm.

    Raises:
        ValueError: unless 0 < A_lo < A_hi and level > 1, for invalid model arguments, or if the level is not
        bracketed by [A_lo, A_hi].
    """
    return None
```

### Step 9

confinement_audit

Goal
----
Runs the complete chain and assembles the confinement audit. Rows 1..len(radii)+1 correspond to the radii in the given order followed by A_star, with the 19 columns [A, E_0 (exact ground state), E_e - E_0, |d_e|^2, E_h - E_0, |d_h|^2, w_h/w_e, E_e^bare, E_h^bare, |d_bare|^2, E_e^min - E_0^min, E_h^min - E_0^min, |d_e^min|^2, |d_h^min|^2, E_(2p)w - E_(1s)w, |d_w|^2, f_e, f_h, (E_e - E_0) - E_e^bare], where the exact quantities use the basis (N_n, L_max) and the grid N_r, the minimal model is the source's 1s + 1p model, the weak-confinement model uses N_g, and the electron-like resonance is the largest-weight L = 1 transition with the hole-like one the largest weight below it. The head row (row 0) holds [E_e - E_0 at A_star, its Coulomb shift (E_e - E_0) - E_e^bare at A_star, w_h/w_e at A_star, f_e at A_star, the independence radius A_10 on [A_lo, A_hi] at the given level, c_EC at A_star, the exciton Bohr radius a_X = 4 pi eps eps_0 hbar^2/(mu e^2), the source's estimate 3 pi hbar^2 eps eps_0 (1/m_e - 1/m_h)/e^2 of the independence radius, A_star, N_n, L_max, N_r, N_g, len(radii), the kinetic splitting z_11^2 - z_10^2 (the coefficient of 1/A^2), the 1s-1s direct Coulomb integral in units of e^2/(4 pi eps eps_0 A), the strongly confined limit of the ratio of the weak-confinement transition energy to the non-interacting electron transition (the weak-confinement model with the Coulomb term omitted, at A_star), the ratio of the weak-confinement transition energy to the exact electron-like resonance at A_star] followed by a zero.

```python
def confinement_audit(me: float, mh: float, eps: float, radii: list, N_n: int, L_max: int, N_r: int, N_g: int, A_star: float, A_lo: float, A_hi: float, level: float) -> "np.ndarray":
    """Runs the complete chain and assembles the confinement audit. Rows 1..len(radii)+1 correspond to the radii in the
    given order followed by A_star, with the 19 columns [A, E_0 (exact ground state), E_e - E_0, |d_e|^2, E_h - E_0,
    |d_h|^2, w_h/w_e, E_e^bare, E_h^bare, |d_bare|^2, E_e^min - E_0^min, E_h^min - E_0^min, |d_e^min|^2, |d_h^min|^2,
    E_(2p)w - E_(1s)w, |d_w|^2, f_e, f_h, (E_e - E_0) - E_e^bare], where the exact quantities use the basis (N_n,
    L_max) and the grid N_r, the minimal model is the source's 1s + 1p model, the weak-confinement model uses N_g, and
    the electron-like resonance is the largest-weight L = 1 transition with the hole-like one the largest weight below
    it. The head row (row 0) holds [E_e - E_0 at A_star, its Coulomb shift (E_e - E_0) - E_e^bare at A_star, w_h/w_e
    at A_star, f_e at A_star, the independence radius A_10 on [A_lo, A_hi] at the given level, c_EC at A_star, the
    exciton Bohr radius a_X = 4 pi eps eps_0 hbar^2/(mu e^2), the source's estimate 3 pi hbar^2 eps eps_0 (1/m_e -
    1/m_h)/e^2 of the independence radius, A_star, N_n, L_max, N_r, N_g, len(radii), the kinetic splitting z_11^2 -
    z_10^2 (the coefficient of 1/A^2), the 1s-1s direct Coulomb integral in units of e^2/(4 pi eps eps_0 A), the
    strongly confined limit of the ratio of the weak-confinement transition energy to the non-interacting electron
    transition (the weak-confinement model with the Coulomb term omitted, at A_star), the ratio of the
    weak-confinement transition energy to the exact electron-like resonance at A_star] followed by a zero.

    Model and conventions: a single spinless electron-hole pair in a spherical nanocrystal of radius A, each particle
    confined by an infinite hard wall (zero potential for r < A, infinite for r >= A), with isotropic effective masses
    m_e and m_h in units of the free electron mass and the Coulomb attraction -e^2/(4 pi eps eps_0 |r_e - r_h|)
    screened by the relative permittivity eps; hbar^2/(2 m_0) = 0.03809985 eV nm^2 and e^2/(4 pi eps_0) = 1.439964 eV
    nm, energies in eV, lengths in nm. Single-particle box states are psi_{nlm}(r) = N_{nl} j_l(z_{ln} r/A)
    Y_{lm}(theta, phi) with z_{ln} the n-th positive zero of the spherical Bessel function j_l (n = 1, 2, ...; l = 0,
    1, ...), kinetic energy hbar^2 z_{ln}^2/(2 m A^2), unit normalisation over the sphere and spherical harmonics in
    the Condon-Shortley phase convention. The exact model is the eigenproblem of the two-particle Hamiltonian in the
    product basis of single-particle states with n <= N_n and l <= L_max for each particle; it is block diagonal in
    the total angular momentum L and in the parity (-1)^{l_e + l_h}, and the coupled basis |(n_e l_e)(n_h l_h) L M> =
    sum_m <l_e m l_h (M - m)|L M> |n_e l_e m>|n_h l_h (M - m)> with M = 0 is used, the ground state lying in the L = 0
    even block and the dipole-active excited states in the L = 1 odd block. Coulomb matrix elements follow from the
    multipole expansion 1/|r_e - r_h| = sum_k (4 pi/(2k + 1)) (r_<^k/r_>^{k+1}) sum_q Y_kq(r_e) Y_kq^*(r_h), with the
    radial integrals R^k[(n_e' l_e')(n_e l_e); (n_h' l_h')(n_h l_h)] = int int R_{n_e' l_e'}(r_e) R_{n_e l_e}(r_e)
    r_e^2 R_{n_h' l_h'}(r_h) R_{n_h l_h}(r_h) r_h^2 r_<^k/r_>^{k+1} dr_e dr_h (in nm^-1), which must be exact to
    1e-10: the kernel has a kink on r_e = r_h, so the inner integral is taken piecewise (cumulative quadrature or a
    Poisson-equation solve), never by a plain product rule. THz response: the dipole operator is d = -e (r_e - r_h);
    for a field along z the transition dipoles from the ground state |0> (L = 0) to the L = 1, M = 0 eigenstates |k>
    are d_k = <k| z_e - z_h |0> in nm, and the linear conductivity of the source has resonances at hbar omega_k = E_k
    - E_0 with peak weights w_k = (E_k - E_0) |d_k|^2; the electron-like resonance is the eigenstate with the largest
    weight and the hole-like resonance the largest-weight eigenstate below it in energy (for m_e < m_h the electron
    transition lies higher). Reference models: the non-interacting pair, whose electron and hole 1s -> 1p transitions
    have energies hbar^2 (z_{11}^2 - z_{10}^2)/(2 m A^2) (z_{10} = pi) and squared dipole |<1p_z| z |1s>|^2 = (int
    R_{11} R_{10} r^3 dr)^2/3 each; the source's minimal strong-confinement model, which is the exact model restricted
    to N_n = 1 and L_max = 1; and the source's weak-confinement model, the relative-motion radial equation -(hbar^2/2
    mu) [R'' + (2/r) R' - l(l+1) R/r^2] - e^2 R/(4 pi eps eps_0 r) + chi(r) R = E R with the shrunken-exciton
    centre-of-mass term chi(r) = hbar^2 pi^2/(2 M (A - rho(r))^2), rho(r) = mu r/min(m_e, m_h), M = m_e + m_h, mu =
    m_e m_h/M, on 0 < r < r_max = A min(m_e, m_h)/mu (where rho reaches A and chi diverges), whose lowest l = 0 state
    (1s)_w and lowest l = 1 state (2p)_w give the transition energy and the relative-coordinate squared dipole |<2p_z|
    z |1s>|^2 = (int R_{2p} R_{1s} r^3 dr)^2/3. Every eigenvalue and integral is to be converged to 1e-9 eV (energies)
    or 1e-10 (lengths, dimensionless numbers); N_r and N_g are the sizes of the reference's uniform radial grids
    (Simpson quadrature for the box integrals; a three-point finite-difference eigenproblem for u = r R with u(0) =
    u(r_max) = 0 on N_g intervals, extrapolated to zero step), and any quadrature or discretisation converged to the
    same accuracy is acceptable.

    Derived quantities: the renormalisation factors f_e = |d_e|^2/|d_bare|^2 and f_h = |d_h|^2/|d_bare|^2 of the
    electron-like and hole-like resonances relative to the non-interacting squared dipole, the weight ratio w_h/w_e
    (hole-like over electron-like), the Coulomb shift of the electron-like resonance relative to the non-interacting
    electron transition, the source's coupling constant c_EC = |E_C| 4 pi eps eps_0 A/e^2 with E_C the off-diagonal
    element between the two states of the minimal L = 1 block, and the independence radius A_10, the radius at which
    f_e reaches the level 1 + 0.1 (root-found on a stated bracket).

    Args:
        me: float, the electron effective mass in units of the free electron mass (positive).
        mh: float, the hole effective mass in units of the free electron mass (positive).
        eps: float, the relative permittivity that screens the Coulomb attraction (positive).
        radii: list of floats, the audit radii in nm (at least one, all positive), in the order of the table rows.
        N_n: int, the number of radial box states per angular momentum (n = 1..N_n).
        L_max: int, the largest single-particle angular momentum of the basis (l = 0..L_max).
        N_r: int, the number of points of the uniform radial Simpson grid on [0, A]; an odd integer of at least 201.
        N_g: int, the number of uniform intervals of the finite-difference grid on [0, r_max] at the coarsest
            Richardson level; at least 500.
        A_star: float, the reference radius in nm (positive) of the head row and of the last table row.
        A_lo: float, the lower end of the bracket in nm on which the root is sought (0 < A_lo < A_hi).
        A_hi: float, the upper end of the bracket in nm.
        level: float, the target renormalisation factor f_e (above 1; 1.1 for a ten percent enhancement).

    Returns:
        A float64 array of shape (len(radii) + 2, 19): the head row followed by one row per radius and the A_star row.

    Raises:
        ValueError: for an empty or non-positive radii list, a non-positive A_star, or invalid arguments of the
        underlying steps.
    """
    return None
```
