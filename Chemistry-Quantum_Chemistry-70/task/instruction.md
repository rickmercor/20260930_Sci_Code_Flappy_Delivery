# Chemistry-Quantum_Chemistry-70

## Background

Doubly excited states of two-electron atoms lie above the first ionization threshold, and most of them decay within femtoseconds by autoionization, the Coulomb interaction pushing one electron into the continuum while the other drops to the ion ground state. A few of them cannot do this. Autoionization conserves total orbital angular momentum, total spin and parity, and a continuum made of a ground-state ion and one free electron with orbital angular momentum l has total angular momentum l and parity (-1)^l. A state whose parity differs from (-1)^L, an unnatural-parity state, therefore has no continuum to decay into in the nonrelativistic limit. Such states are metastable against electron emission and decay by emitting photons, in electric-dipole transitions to lower levels of opposite parity.

The same kind of protection appears in other few-body Coulomb systems. In the positronium molecule, for example, an excited level lies above the threshold for breakup into two ground-state positronium atoms but, with the wrong symmetry under exchange of the two atoms for the only open channel, cannot break up without emitting a photon. Its radiative decay populates both a lower bound state and the dissociation continuum. For systems of this size the method of choice is a variational expansion in explicitly correlated Gaussians, products of Gaussians in the interparticle distances multiplied by simple angular factors. Their matrix elements are analytic, and sets with exponents in geometric progressions describe compact bound states, extended Rydberg states and a square-integrable discretization of the continuum on the same footing.

Radiative decay rates follow from Fermi's golden rule applied to the coupling between the atom and the quantized electromagnetic field. In the long-wavelength limit only the electric-dipole operator contributes, and the rate grows with the cube of the photon energy. Transition moments computed in the length and velocity forms agree for exact eigenstates, so their agreement is a standard check on the quality of approximate wavefunctions. Oscillator strength spread over a continuum can be handled either by resolving the continuum energy by energy, for example with complex-scaled Hamiltonians, or by summing over the discrete pseudostates of a large square-integrable basis, which reproduces energy-integrated quantities once the basis spans the relevant region.

## Problem

A bound state that lies above a dissociation or ionization threshold but cannot break up, because no open channel shares its symmetry, lives only as long as photon emission allows. Helium has a state of this kind: the doubly excited 2p^2 3P^e level lies far above the He+(1s) + e- threshold, yet every 1s el continuum with total orbital angular momentum one has odd parity, so in the nonrelativistic limit the level cannot autoionize. Its radiative width is shared between discrete lines to the bound 1snp 3P^o levels and a continuous distribution over the final energies of the 1s ep continuum, where the atom is left ionized. The input is the nonrelativistic two-electron Hamiltonian of helium and the output is one branching fraction of the radiative decay of this level.

Treat helium with an infinitely heavy nucleus of charge Z = 2 and the spin-free Hamiltonian H = -(nabla_1^2 + nabla_2^2)/2 - 2/r1 - 2/r2 + 1/r12 in hartree atomic units. Neglect spin-orbit and all other relativistic effects, magnetic and higher electric multipoles, and nuclear motion. Spontaneous emission is treated in the electric-dipole approximation: the rate from the initial state i to a final state f is (4/3) alpha^3 omega^3 times |<f| r1_vec + r2_vec |i>|^2 summed over photon polarizations and final magnetic sublevels, with omega = E_i - E_f, alpha = 7.2973525693e-3 and one atomic unit of time equal to 2.4188843265857e-17 s. The continuum part of the width is not a sum of lines and has to be obtained as a distribution over the final-state energy, from a calculation that represents continuum final states.

Expand both parities in explicitly correlated Gaussians (1 - P12)[(r1_vec x r2_vec)_z g] for the even states and (1 - P12)[z1 g] for the odd states, with g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 the exchange of the two electrons. Write grid(lo, hi, n) for the n exponents lo (hi / lo)^(i / (n - 1)) with i = 0 to n - 1, in bohr^-2. The even set has 819 functions: a and b taken from grid(0.012, 14, 13) with a <= b, each pair combined with c = 0 and with each of grid(0.01, 4, 8). The odd set has 4640 functions in three blocks: a from grid(0.25, 200, 11) with b from grid(2e-4, 5, 22) and c = 0 or one of grid(0.015, 1.2, 3); the same block with a and b exchanged; and a and b both from grid(0.015, 10, 13) with c = 0 or one of grid(0.01, 10, 15). Solve each parity by removing the eigenvectors of the overlap matrix of the unnormalized functions whose eigenvalue is at most 1e-13 times the largest before diagonalizing. Take the initial state to be the lowest even-parity root, and the discrete final states to be the odd-parity roots below the one-electron threshold -Z^2 / 2 hartree. Obtain the continuum distribution at a rotation angle of 0.25 radian and integrate it over the final-state energy from the threshold to the initial energy on a uniform grid of 8001 points with the trapezoidal rule. The result must be stationary with respect to the rotation angle over a plateau, and the answer is converged only if it is.

Let A_tot be the total electric-dipole decay rate of the 2p^2 3P^e level, the discrete lines plus the integrated continuum distribution, and A(1s2p) its partial rate to the 1s2p 3P^o level. Compute the fraction f = 1 - A(1s2p) / A_tot and give f to three significant figures as the final answer. In your reasoning also give the nonrelativistic energy of the 2p^2 3P^e level in hartree, the photon energy of the 2p^2 3P^e to 1s2p 3P^o line in hartree, A(1s2p) and A_tot in s^-1, the share of A_tot carried by the 1s3p 3P^o level, the share of A_tot carried by the continuum, and the size of the variation of f across the plateau in the rotation angle. From the literature, also report the nonrelativistic infinite-nuclear-mass energy of the 1s2p 3P^o level of helium in hartree, and, from the NIST Atomic Spectra Database, at the precision that database tabulates them, the excitation energies above the helium ground state of the 2p^2 3P level and of the 2p3d 3D^o level in eV and the observed vacuum wavelength of the 1s2p 3P^o to 2p^2 3P line in nm.

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

01_ecg_overlap

Goal
----
Step 01: Overlap matrix of antisymmetrized L = 1 correlated Gaussians. Overlap matrix between spatially antisymmetric two-electron explicitly correlated Gaussian functions of total orbital angular momentum one, for either parity.

```python
def ecg_overlap(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    '''Overlap matrix of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1. Row i holds the exponents (a, b, c) of bra function i, in bohr^-2.
        Every exponent must be finite and non-negative, with a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the ket functions.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] (total L = 1, even parity) or
        "odd" for phi = (1 - P12)[z1 g] (total L = 1, odd parity), with
        g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 the exchange of the two electron positions.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries S_ij = integral of phi_i phi_j over r1_vec and r2_vec
        (unnormalized functions, bohr units).

    Raises
    ------
    ValueError
        If an exponent array has the wrong shape, holds a negative or non-finite exponent, gives a
        non-integrable Gaussian (a b + c (a + b) <= 0), or if kind is not "even" or "odd".
    '''
    return result
```

### Step 2

02_ecg_kinetic

Goal
----
Step 02: Kinetic-energy matrix. Kinetic-energy matrix between the antisymmetrized L = 1 correlated Gaussians of either parity, for two electrons around an infinitely heavy nucleus.

```python
def ecg_kinetic(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    '''Kinetic-energy matrix of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2 for the bra functions; finite, non-negative,
        with a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the ket functions.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 the exchange of the two electron positions.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries T_ij = integral of phi_i [-(nabla_1^2 + nabla_2^2)/2] phi_j over
        r1_vec and r2_vec, in hartree times the overlap units of the unnormalized functions.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, or if kind is not "even" or "odd".
    '''
    return result
```

### Step 3

03_ecg_coulomb

Goal
----
Step 03: Coulomb potential-energy matrix. Coulomb potential-energy matrix, nuclear attraction plus electron repulsion, between antisymmetrized L = 1 correlated Gaussians.

```python
def ecg_coulomb(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str, Z: float) -> "np.ndarray":
    '''Coulomb matrix -Z/r1 - Z/r2 + 1/r12 of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2; finite, non-negative, a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge in units of the proton charge, Z > 0.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries V_ij = integral of phi_i (-Z/r1 - Z/r2 + 1/r12) phi_j over both
        electrons, in hartree times the overlap units of the unnormalized functions.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, if kind is not "even" or "odd", or if Z is not positive.
    '''
    return result
```

### Step 4

04_ecg_dipole

Goal
----
Step 04: Electric-dipole coupling matrix between the two parities. Electric-dipole coupling matrix between even-parity and odd-parity antisymmetrized L = 1 correlated Gaussians, in the length or the velocity form.

```python
def ecg_dipole(exps_even: "np.ndarray", exps_odd: "np.ndarray", gauge: str) -> "np.ndarray":
    '''Dipole coupling matrix between even (1^+) and odd (1^-) antisymmetrized correlated Gaussians.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2 for the even functions; finite,
        non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the odd functions.
    gauge : str
        "length" for D_ij = integral of chi_i (y1 + y2) phi_j, or "velocity" for
        D_ij = integral of chi_i (d/dy1 + d/dy2) phi_j, where chi_i = (1 - P12)[(r1_vec x r2_vec)_x g_i],
        phi_j = (1 - P12)[z1 g_j], g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 exchanges the electrons.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) holding D_ij for the unnormalized functions. For normalized states the length form
        has units of bohr and the velocity form units of inverse bohr.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, or if gauge is not "length" or "velocity".
    '''
    return result
```

### Step 5

05_rotated_levels

Goal
----
Step 05: Complex-rotated L = 1 triplet levels and coefficients. Complex-rotated L = 1 triplet levels and expansion coefficients of a two-electron atom in a correlated-Gaussian basis of one parity.

```python
def rotated_levels(exps: "np.ndarray", kind: str, Z: float, theta: float, cut: float) -> "np.ndarray":
    '''Complex-rotated L = 1 triplet eigenvalues and expansion coefficients for a two-electron atom.

    Parameters
    ----------
    exps : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2; finite, non-negative, a b + c (a + b) > 0.
    kind : str
        "even" for the basis phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    theta : float
        Rotation angle in radians, 0 <= theta < pi/4. The electron coordinates are scaled by exp(i theta), so the
        kinetic and potential parts of H = -(nabla_1^2 + nabla_2^2)/2 - Z/r1 - Z/r2 + 1/r12 are multiplied by
        exp(-2 i theta) and exp(-i theta) respectively. theta = 0 gives the ordinary real variational problem.
    cut : float
        Relative threshold, 0 < cut < 1. Eigenvectors of the overlap matrix of the unnormalized basis functions (as
        defined for ecg_overlap) with eigenvalue s <= cut * max(s) are removed before the rotated Hamiltonian is
        diagonalized in the remaining space.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (k, n + 1), one row per retained eigenstate, where k is the number of retained overlap
        eigenvectors. Element [j, 0] is the eigenvalue E_j in hartree and elements [j, 1:] are the coefficients of
        eigenstate j in the original unnormalized basis. Rows are sorted by ascending real part of E_j, ties broken by
        ascending imaginary part. Each coefficient vector is scaled so that c^T S c = 1 with S the overlap matrix of
        the unnormalized functions, and the remaining sign is fixed by requiring the coefficient of largest modulus to
        have a positive real part, or a positive imaginary part if that real part is zero.

    Raises
    ------
    ValueError
        If the exponents are malformed or non-integrable, kind is not "even" or "odd", Z is not positive, theta is
        outside [0, pi/4), or cut is not strictly between 0 and 1.
    '''
    return result
```

### Step 6

06_bound_line_rates

Goal
----
Step 06: E1 rates of the lines to the bound odd-parity levels. Partial electric-dipole rates of the lines from the lowest even-parity L = 1 triplet state to the bound odd-parity L = 1 triplet levels.

```python
def bound_line_rates(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, cut: float) -> "np.ndarray":
    '''Photon energies and E1 rates of the lines to the bound odd-parity levels below the lowest even-parity state.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3) of exponents (a, b, c) in bohr^-2 for the even basis (1 - P12)[(r1_vec x r2_vec) g];
        finite, non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), same rules, for the odd basis (1 - P12)[r1_vec g], g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    cut : float
        Relative threshold, 0 < cut < 1, on the eigenvalues of the overlap matrix of the unnormalized basis functions,
        applied separately to each basis before the Hamiltonian -(nabla_1^2 + nabla_2^2)/2 - Z/r1 - Z/r2 + 1/r12 is
        diagonalized without rotation.

    Returns
    -------
    result : np.ndarray
        Real array of shape (k, 3) with one row per odd-parity eigenstate whose energy lies below both the one-electron
        threshold -Z^2/2 hartree and the lowest even-parity energy E_i, in ascending order of E_f. The columns are E_f
        in hartree, the photon energy omega = E_i - E_f in hartree, and the rate
        A_f = (4/3) alpha^3 omega^3 * 2 * D_f^2 / t_au in s^-1, where D_f is the length-form dipole matrix element
        between the two states normalized with their own overlap matrices, the factor 2 is the polarization and
        sublevel multiplicity, alpha = 7.2973525693e-3 and t_au = 2.4188843265857e-17 s. k may be zero.

    Raises
    ------
    ValueError
        If either exponent array is malformed or non-integrable, Z is not positive, or cut is not strictly between
        0 and 1.
    '''
    return result
```

### Step 7

07_dissociation_spectrum

Goal
----
Step 07: Energy-resolved rate into the continuum final states. Energy-resolved rate of the radiative transitions that leave the atom in the continuum above the one-electron threshold.

```python
def dissociation_spectrum(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                          energies: "np.ndarray") -> "np.ndarray":
    '''Rate per unit photon energy into continuum final states, at given final-state energies.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3) of exponents (a, b, c) in bohr^-2 for the even basis (1 - P12)[(r1_vec x r2_vec) g];
        finite, non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), same rules, for the odd basis (1 - P12)[r1_vec g], g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    theta : float
        Rotation angle in radians, 0 < theta < pi/4, applied to both bases as for rotated_levels.
    cut : float
        Relative threshold, 0 < cut < 1, on the overlap eigenvalues, applied separately to each basis.
    energies : np.ndarray
        One-dimensional array of q >= 1 finite final-state energies E_F in hartree at which the distribution is
        evaluated.

    Returns
    -------
    result : np.ndarray
        Real array of shape (q,): the energy-resolved electric-dipole continuum rate density in s^-1 per hartree
        at each supplied final-state energy. The polarization and sublevel multiplicity is 2.
        alpha = 7.2973525693e-3 and t_au = 2.4188843265857e-17 s.

    Raises
    ------
    ValueError
        If either exponent array is malformed or non-integrable, Z is not positive, theta is outside (0, pi/4), cut is
        not strictly between 0 and 1, or energies is not a one-dimensional array of at least one finite value.
    '''
    return result
```

### Step 8

08_shake_fraction

Goal
----
Step 08: Shake fraction of the radiative width (orchestrator). Orchestrator: fraction of the total electric-dipole radiative width of the lowest even-parity L = 1 triplet state that does not end on the lowest odd-parity L = 1 triplet level.

```python
def shake_fraction(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                   n_grid: int) -> float:
    '''Fraction of the total E1 decay rate of the lowest 1^+ triplet state not carried by its strongest line.

    Parameters
    ----------
    exps_even : np.ndarray
        Array of shape (n, 3) of exponents (a, b, c) in bohr^-2 for the even basis (1 - P12)[(r1_vec x r2_vec) g];
        finite, non-negative, a b + c (a + b) > 0.
    exps_odd : np.ndarray
        Array of shape (m, 3), same rules, for the odd basis (1 - P12)[r1_vec g], g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    theta : float
        Rotation angle in radians, 0 < theta < pi/4, used for the continuum part only.
    cut : float
        Relative threshold, 0 < cut < 1, on the overlap eigenvalues, applied separately to each basis.
    n_grid : int
        Number of points, n_grid >= 2, of the uniform grid of final-state energies spanning the closed interval from
        the one-electron threshold -Z^2/2 hartree to the lowest even-parity energy E_i, on which the continuum
        distribution is evaluated and integrated by the trapezoidal rule.

    Returns
    -------
    result : float
        1 - A_1 / A_tot, where A_1 is the rate of the line to the lowest bound odd-parity level and A_tot is the sum
        of the rates of every line to a bound odd-parity level and the integral over final-state energy of the
        continuum distribution.

    Raises
    ------
    ValueError
        If the inputs are invalid as for the line table and the continuum distribution, if n_grid is smaller than 2,
        or if no bound odd-parity level lies below the initial state.
    '''
    return result
```
