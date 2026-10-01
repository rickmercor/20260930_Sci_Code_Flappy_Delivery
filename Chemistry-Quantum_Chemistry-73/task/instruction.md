# Chemistry-Quantum_Chemistry-73

## Background

Kohn-Sham density functional theory would be exact with the exact exchange-correlation functional, and several exact conditions on that functional are known. One of them concerns systems with a fractional number of electrons, which arise for example when a molecule is pulled apart into fragments that are far from each other: the exact ground-state energy varies linearly between integer electron numbers, and for a fractional spin the energy does not change from the integer-spin value. Local and semilocal approximations violate both conditions. For fractional charge their energy curve bends below the straight line, which is the delocalization error that favours spreading electrons over separated fragments, and for fractional spin their energy lies above the flat line, which is the static-correlation error.

The total error of an approximate calculation can be split into a part caused by the approximate density and a part caused by the approximate functional evaluated on the exact density. Density-corrected methods reduce the first part by evaluating the functional on a better density, commonly one from Hartree-Fock theory, and they are most useful for systems whose energies are sensitive to the density. For one-electron systems the Hartree-Fock density is exact, so on that criterion density correction should change little, although densities that are formally worse may still compensate for errors of the functional.

One-dimensional models with a softened Coulomb interaction are widely used to test such ideas because their exact solutions and local-density functionals can be computed to high precision on a grid, while retaining the qualitative physics of bond breaking, self-interaction and fractional charges.

## Problem

Density-corrected density functional theory evaluates an approximate functional on a density other than its own self-consistent one, most often the Hartree-Fock density, and it is usually expected to help only when the approximate density is itself the source of the error. Bonds that break into fragments of equal charge test that expectation: a semilocal functional spreads the electron over both fragments, the resulting error at long range reflects the functional's behaviour at fractional electron number, and the exact one-electron density removes little of it. A density that dissociates to integer charges can nevertheless cancel most of that error when the same functional is evaluated on it, even though this density is less accurate than the exact one. The task is to find, for a one-dimensional model of H2+, the internuclear separation beyond which such a localized density gives a smaller binding-energy error than the self-consistent calculation.

Work in atomic units with one electron on a line and protons of unit charge. In H2+ the protons sit at x = -R/2 and x = +R/2 and the electron feels v(x) = -1/sqrt((x - R/2)^2 + 1) - 1/sqrt((x + R/2)^2 + 1); the H atom has a single proton at x = 0 with the corresponding single term. Electrons interact through w(u) = 1/sqrt(u^2 + 1). Every softened denominator in this model carries the same softening length b = 1 bohr, as written. The approximate functional is exchange-only local spin-density theory for this interaction: for a normalized real orbital phi with density n = phi^2 carried by one spin-up electron, E[phi] = <phi| -(1/2) d^2/dx^2 + v |phi> + J[n] + E_x[n, 0], where J[n] = (1/2) double integral of n(x) n(x') w(x - x') and E_x[n_up, n_down] = sum over spins of the integral of n_s(x) eps_x(n_s(x)). Here eps_x(n) is the exact exchange energy per electron of the uniform one-dimensional electron gas of density n in which every electron has the same spin and electrons interact through w; there is no correlation term. The repulsion between the protons is the same in every treatment and is left out.

For every treatment the binding energy is its energy for H2+ at separation R minus its energy for H, and its error is that binding energy minus the exact one, built from the lowest eigenvalues of -(1/2) d^2/dx^2 + v for H2+ and for H. In the self-consistent treatment both energies are the lowest values of E over normalized orbitals. In the localized treatment the H2+ energy is E evaluated on phi_loc, the normalized orbital within the span of the two lowest eigenstates of -(1/2) d^2/dx^2 + v for H2+ that has the largest probability on the half-line x < 0, and the H energy is E evaluated on the exact ground-state orbital of H. Find the separation R_x at which the magnitudes of the two errors are equal, the localized treatment having the larger error at shorter separations and the smaller one beyond, and give R_x in bohr to three decimal places as the final answer.

Report every number for the following discretization, so that the intermediate values below are reproducible rather than merely close. Represent orbitals on the uniform grid x_j = -L + j h with spacing h = 0.025 bohr and half-width L = 30 bohr, take the kinetic energy from the three-point second difference on that grid with the orbital vanishing outside it, and evaluate every integral, including the normalization and the double integral in J, as h times a sum over grid points. Take the self-consistent energy to be the lowest value the functional attains at a density that reproduces itself, which for these inversion-symmetric nuclei is the symmetric solution, and iterate until successive densities agree to better than 1e-10 in integrated absolute difference. When comparing the probabilities of the two half-lines that select phi_loc, count the grid point at the origin with half weight. Refine R_x inside the bracket 4.5 to 5.6 bohr. Where a quantity is asked for in kcal/mol, convert with 627.5094740631 kcal/mol per hartree. R_x is a property of this one-dimensional model exactly as printed here; it is not a separation tabulated in any study of three-dimensional molecules.

In your reasoning give, as the scalars that determine and check it: the exact and the self-consistent energies of H; the fractional-charge deviation 2E(1/2) - E(1), where E(N) is the lowest value of the functional for H with N electrons in one spin-up orbital (density N phi^2); the fractional-spin deviation, the lowest energy of H with half an electron of each spin in the same orbital minus E(1); at R_x, the self-consistent energy of H2+, the energy of the functional evaluated on phi_loc, the error of the self-consistent treatment and the splitting between the two lowest H2+ eigenvalues; the error at R_x when the functional is evaluated on the exact ground-state orbital of H2+ instead of phi_loc (with the same H reference); and the separation at which the localized error magnitude equals the magnitude of that exact-orbital error. For comparison with the literature on density-corrected functionals, also give the model's density sensitivity of the H2+ binding energy at R_x in kcal/mol as that literature defines it, with the self-consistent densities of this functional in place of the local-density-approximation densities and the exact densities in place of the Hartree-Fock densities, the threshold above which that literature classifies a calculation as density-sensitive (abnormal), the published PBE density sensitivity of three-dimensional H2+ stretched to 5.0 Angstrom, and the published local spin-density total energy, with correlation, of this one-dimensional hydrogen atom. Every quantity and literature value requested in this paragraph counts as one of the few scalars to show in your reasoning; give each as a number, with computed energies in hartree to five decimal places and computed separations in bohr to three.

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

01_polarized_exchange_per_electron

Goal
----
Step 01: Exchange energy per electron of the spin-polarized soft-Coulomb uniform gas.

```python
def polarized_exchange_per_electron(density: "np.ndarray", softening: float) -> "np.ndarray":
    '''Exchange energy per electron of the spin-polarized uniform 1D gas with pair interaction 1/sqrt(u^2 + b^2).

    Parameters
    ----------
    density : np.ndarray
        Array of any shape of non-negative linear densities n of one spin channel, in electrons per bohr.
    softening : float
        Softening length b > 0 of the pair interaction, in bohr.

    Returns
    -------
    result : np.ndarray
        Array of the same shape as density holding eps_x(n) in hartree, with eps_x(0) = 0, accurate to within 1e-10
        hartree for every density.

    Raises
    ------
    ValueError
        If any density is negative or not finite, or if the softening is not strictly positive.
    '''
    return result  # placeholder
```

### Step 2

02_polarized_exchange_potential

Goal
----
Step 02: Local exchange potential of one spin channel.

```python
def polarized_exchange_potential(density: "np.ndarray", softening: float) -> "np.ndarray":
    '''Exchange potential d[n eps_x(n)]/dn of one spin channel for the soft-Coulomb uniform-gas exchange.

    Parameters
    ----------
    density : np.ndarray
        Array of any shape of non-negative spin densities n, in electrons per bohr.
    softening : float
        Softening length b > 0 of the pair interaction 1/sqrt(u^2 + b^2), in bohr.

    Returns
    -------
    result : np.ndarray
        Array of the same shape as density holding v_x(n) in hartree, with v_x(0) = 0.

    Raises
    ------
    ValueError
        If any density is negative or not finite, or if the softening is not strictly positive.
    '''
    return result  # placeholder
```

### Step 3

03_soft_coulomb_states

Goal
----
Step 03: Lowest eigenstates of soft-Coulomb protons on a finite-difference grid.

```python
def soft_coulomb_states(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float, n_states: int) -> "np.ndarray":
    '''Lowest eigenvalues and normalized eigenvectors of the finite-difference soft-Coulomb one-electron Hamiltonian.

    Parameters
    ----------
    nuclei : np.ndarray
        Shape (P,), P >= 1, positions of unit point charges in bohr.
    softening : float
        Softening length b > 0 of the electron-nucleus attraction, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr; 2L / h_x must be an integer M within a relative tolerance of 1e-9.
    n_states : int
        Number of lowest states to return, 1 <= n_states <= M + 1.

    Returns
    -------
    result : np.ndarray
        Shape (n_states, M + 2). Row k holds state k in order of increasing energy: column 0 is the energy in hartree and
        columns 1 to M + 1 are the orbital values at x_0, ..., x_M, normalized so that h_x sum_j phi_j^2 = 1 and signed
        so that the first grid point from the left with |phi_j| >= 10^-3 max|phi| is positive.

    Raises
    ------
    ValueError
        If nuclei is empty, the softening, spacing or half-width is not positive, 2L / h_x is not an integer, or
        n_states is outside 1 <= n_states <= M + 1.
    '''
    return result  # placeholder
```

### Step 4

04_hartree_exchange_energy

Goal
----
Step 04: Hartree plus local spin-density exchange energy on a grid.

```python
def hartree_exchange_energy(density_up: "np.ndarray", density_down: "np.ndarray", spacing: float, softening: float) -> float:
    '''Rectangle-rule Hartree energy of the total density plus soft-Coulomb local spin-density exchange energy.

    Parameters
    ----------
    density_up : np.ndarray
        Shape (G,), non-negative spin-up density on a uniform grid, in electrons per bohr.
    density_down : np.ndarray
        Shape (G,), non-negative spin-down density on the same grid.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    softening : float
        Softening length b > 0 of the pair interaction, in bohr.

    Returns
    -------
    result : float
        J[n_up + n_down] + E_x[n_up, n_down] in hartree.

    Raises
    ------
    ValueError
        If the two densities are not one-dimensional arrays of equal length, if any density is negative or not finite,
        or if the spacing or softening is not strictly positive.
    '''
    return result  # placeholder
```

### Step 5

05_lsda_ground_state_energy

Goal
----
Step 05: Lowest self-consistent LSDA energy of a one-electron system.

```python
def lsda_ground_state_energy(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    '''Lowest self-consistent exchange-only LSDA energy of one spin-up electron in a symmetric soft-Coulomb field.

    Parameters
    ----------
    nuclei : np.ndarray
        Shape (P,), P >= 1, positions of unit point charges in bohr; the set of positions must be symmetric under
        x -> -x within 1e-9 bohr.
    softening : float
        Softening length b > 0 used for both the electron-nucleus and the electron-electron interaction, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.

    Returns
    -------
    result : float
        The lowest total electronic energy in hartree among the self-consistent solutions, without nuclear repulsion.

    Raises
    ------
    ValueError
        If the nuclear configuration is not symmetric under x -> -x, or for any invalid grid or softening as in
        soft_coulomb_states.
    RuntimeError
        If the self-consistent iteration fails to converge within 5000 cycles.
    '''
    return result  # placeholder
```

### Step 6

06_maximally_localized_orbital

Goal
----
Step 06: Maximally left-localized orbital in a two-state span.

```python
def maximally_localized_orbital(orbital_one: "np.ndarray", orbital_two: "np.ndarray", spacing: float, half_width: float) -> "np.ndarray":
    '''Normalized combination of two orthonormal grid orbitals with the largest probability on x < 0.

    Parameters
    ----------
    orbital_one : np.ndarray
        Shape (M + 1,), real orbital on the grid x_j = -L + j*h_x.
    orbital_two : np.ndarray
        Shape (M + 1,), real orbital on the same grid, orthonormal to orbital_one within 1e-8 under h_x sum_j.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x = M.

    Returns
    -------
    result : np.ndarray
        Shape (M + 1,), the combination c_1 orbital_one + c_2 orbital_two with c_1^2 + c_2^2 = 1 that maximizes
        h_x sum_j s_j phi_j^2, with s_j = 1 for x_j < 0, 1/2 at x_j = 0 and 0 for x_j > 0, signed so that
        h_x sum_j s_j phi_j > 0.

    Raises
    ------
    ValueError
        If the orbitals do not have length 2L / h_x + 1, are not orthonormal within 1e-8, or if the maximizing
        combination is not unique because the two extreme left-half probabilities coincide within 1e-12.
    '''
    return result  # placeholder
```

### Step 7

07_dissociation_errors

Goal
----
Step 07: Binding-energy errors of SCF and density-corrected LSDA for H2+.

```python
def dissociation_errors(separation: float, softening: float, spacing: float, half_width: float) -> "np.ndarray":
    '''Binding-energy errors of SCF LSDA and of LSDA on the exact and on the localized orbital, plus the H2+ splitting.

    Parameters
    ----------
    separation : float
        Internuclear separation R > 0 of H2+ in bohr, nuclei at -R/2 and +R/2.
    softening : float
        Softening length b > 0 of all interactions, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.

    Returns
    -------
    result : np.ndarray
        Shape (4,): the error of the self-consistent LSDA binding energy, the error of LSDA evaluated on the exact H2+
        ground-state orbital, the error of LSDA evaluated on the maximally left-localized orbital, each relative to the
        exact binding energy E_g(R) - E_H, and the splitting E_u(R) - E_g(R), all in hartree.

    Raises
    ------
    ValueError
        If the separation is not strictly positive, or for any invalid grid or softening as in soft_coulomb_states.
    '''
    return result  # placeholder
```

### Step 8

08_crossover_separation

Goal
----
Step 08: Crossover separation of localized density correction and SCF LSDA (orchestrator).

```python
def crossover_separation(softening: float, spacing: float, half_width: float, r_min: float, r_max: float, n_scan: int) -> float:
    '''Separation, refined in the first scan bracket, where the localized density-corrected error equals the SCF LSDA error in magnitude.

    Parameters
    ----------
    softening : float
        Softening length b > 0 of all interactions, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.
    r_min : float
        Smallest separation of the scan in bohr, r_min > 0.
    r_max : float
        Largest separation of the scan in bohr, r_max > r_min.
    n_scan : int
        Number of evenly spaced scan separations, n_scan >= 2.

    Returns
    -------
    result : float
        Crossover separation R_x in bohr.

    Raises
    ------
    ValueError
        If 0 < r_min < r_max or n_scan >= 2 is violated, if g(r_min) <= 0 so that the scan starts past the crossover,
        or if g stays positive over the whole scan.
    '''
    return result  # placeholder
```
