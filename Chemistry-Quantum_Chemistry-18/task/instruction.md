# Chemistry-Quantum_Chemistry-18

## Background

The on-top two-electron density, the probability density of finding two electrons at the same point, is a central quantity in electronic structure theory. It fixes the short-range part of the electron-electron repulsion, it is the input of on-top pair-density functionals that are combined with multiconfigurational wavefunctions, and its ratio to the square of the one-electron density is often used to discuss static and dynamic correlation and bond breaking.

In practice the on-top density is almost always obtained from wavefunctions built from one-electron functions, such as configuration interaction or coupled-cluster expansions in finite basis sets. The electron-electron cusp of the exact wavefunction cannot be represented by any finite product expansion, so the part of the wavefunction near electron coalescence converges very slowly. Energies are only mildly affected by this, but properties that sample the coalescence region directly, such as the on-top density and the electron-electron delta-function expectation value, converge much more slowly with the number of orbitals or with the maximum angular momentum.

Natural orbitals give the most compact expansion of a two-electron wavefunction, so they show the best possible convergence of any orbital expansion. Two-electron harmonium, two electrons with Coulomb repulsion confined by an isotropic harmonic potential, has closed-form ground states for particular confinement strengths. It keeps the electron-electron cusp while removing the nuclear cusp, and it serves as a benchmark system for studying how on-top densities converge with the number of natural orbitals, and for checking asymptotic laws proposed for that convergence.

## Problem

The probability of finding two electrons at the same point, the on-top two-electron density, enters correlation energies, on-top density functionals and pictures of chemical bonding, yet it is usually computed from wavefunctions expanded in one-electron functions. Such expansions reproduce energies quickly but on-top densities very slowly, because the electron-electron cusp is spread over infinitely many natural orbitals. The task is to find how fast natural-orbital expansions of growing size recover the on-top density where the electrons of a strongly correlated two-electron system are actually found.

Consider harmonium, two electrons in the isotropic harmonic trap with Hamiltonian H = sum over i = 1, 2 of [-(1/2) nabla_i^2 + (1/2) omega^2 r_i^2] + 1/r12, in atomic units. For omega = 1/10 the singlet ground state is exactly Psi(r1_vec, r2_vec) = C (1 + r12/2 + r12^2/20) exp[-(r1^2 + r2^2)/20] with energy 1/2 hartree, where C > 0 normalizes the spatial function to one. Write Psi = sum over n of lambda_n phi_n(r1_vec) phi_n(r2_vec) with orthonormal real natural orbitals phi_n and real natural amplitudes lambda_n, so that sum over n of lambda_n^2 = 1; the occupation number of phi_n is lambda_n^2. The one-electron density is rho1 = 2 sum over n of lambda_n^2 phi_n^2, the on-top density is Phi(r_vec) = |Psi(r_vec, r_vec)|^2 = (sum over n of lambda_n phi_n^2)^2, and the reduced on-top density is phi(r_vec) = 4 Phi / rho1^2; all three depend only on r = |r_vec|. Retaining the set of N natural orbitals whose occupation numbers are at least a threshold, with all members of a degenerate set kept together, gives the estimate phi_N(r) = [(sum over retained n of lambda_n phi_n(r)^2) / (sum over retained n of lambda_n^2 phi_n(r)^2)]^2 times (sum over retained n of lambda_n^2). Let r* be the radius at which the radial density 4 pi r^2 rho1(r) of the exact state is largest.

Lowering the threshold enlarges the retained set, and phi_N(r*) / phi(r*) approaches one as 1 + G N^(-1/3) + o(N^(-1/3)). Find the coefficient G for the omega = 1/10 ground state, with every number converged well beyond the digits you quote.

Give G to three significant figures as the final answer. In your reasoning give r* in bohr and phi(r*); for the threshold 10^-7, N and the ratio F = phi_N(r*) / phi(r*); the two integrals of the on-top density that fix the asymptotics of the weakly occupied natural orbitals, namely J over the volume element and the radial one, together with the coefficient a of the large-n law |lambda_n| -> a n^(-4/3) obeyed by the amplitudes ordered by decreasing magnitude and counted with shell degeneracy; and, for the s-type natural orbitals ordered by decreasing |lambda_k|, the limit B = lim over K to infinity of K [phi_K(0) - phi(0)], where phi_K(0) = [(sum over k <= K of lambda_k phi_k(0)^2) / (sum over k <= K of lambda_k^2 phi_k(0)^2)]^2 uses the K leading s orbitals without renormalization. From the literature, also report the lower and upper switching thresholds R0 and R1 of the fully translated on-top functional of multiconfiguration pair-density functional theory, where R = 4 Pi / rho^2 is the same reduced on-top density, and the coefficient A of the fifth-power term of its switching polynomial; for the helium ground state, the numerical prefactor of the leading term of the partial-wave increments of the electron-electron delta-function expectation value; and, for two-electron harmonium in the limit of vanishing confinement, the common ratio of the geometric progression formed by its natural-orbital occupancies.

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

01_pair_partial_wave

Goal
----
Step 01: Legendre coefficients of the correlation factor. Legendre coefficient of order l of the correlation factor p(r12) of a two-electron pair function, as a function of the two radii.

```python
def pair_partial_wave(r1: "np.ndarray", r2: "np.ndarray", l: int, c: float) -> "np.ndarray":
    '''Legendre coefficient p_l(r1, r2) of p(r12) = 1 + r12/2 + c r12^2.

    Parameters
    ----------
    r1 : np.ndarray
        Radii of electron 1, non-negative; broadcast against r2.
    r2 : np.ndarray
        Radii of electron 2, non-negative; broadcast against r1.
    l : int
        Legendre order, l >= 0.
    c : float
        Coefficient of r12^2 in the correlation factor, c >= 0.

    Returns
    -------
    result : np.ndarray
        Float array with the broadcast shape of r1 and r2 holding p_l(r1, r2). Where both radii are zero only the l = 0
        coefficient is nonzero and equals 1.

    Raises
    ------
    ValueError
        If l is negative.
    '''
    return result
```

### Step 2

02_pair_normalization

Goal
----
Step 02: Normalization of the pair function. Normalization constant C of the singlet pair function Psi = C p(r12) exp[-omega (r1^2 + r2^2) / 2] with p(s) = 1 + s/2 + c s^2.

```python
def pair_normalization(omega: float, c: float) -> float:
    '''Positive constant C that normalizes Psi = C p(r12) exp[-omega (r1^2 + r2^2)/2] to one.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.

    Returns
    -------
    result : float
        The normalization constant C as a Python float, accurate to 1e-10 relative.

    Raises
    ------
    ValueError
        If omega is not positive or c is negative.
    '''
    return result
```

### Step 3

03_weak_orbital_asymptotics

Goal
----
Step 03: Constants of the weak natural-orbital asymptotics. Constants of the weak natural-orbital asymptotics of a two-electron singlet, from its on-top density.

```python
def weak_orbital_asymptotics(omega: float, c: float) -> "np.ndarray":
    '''Constants governing the weakly occupied natural orbitals of the pair function Psi.

    Parameters
    ----------
    omega : float
        Trap frequency, omega > 0, of the Gaussian factor exp[-omega (r1^2 + r2^2) / 2].
    c : float
        Coefficient, c >= 0, of the quadratic term of the correlation factor p(s) = 1 + s / 2 + c s^2.

    Returns
    -------
    result : np.ndarray
        Real array [J, I1, a, K] of four constants of the weak natural-orbital asymptotics, for the normalized pair
        function with these parameters.

        J is the integral of the on-top density raised to the power 3/8 over all space, in the volume element
        d^3r = 4 pi r^2 dr. I1 is the integral of the on-top density raised to the power 1/8 over the radius from
        0 to infinity, in the element dr. a is the coefficient of the large-n law |lambda_n| -> a n^(-4/3) obeyed by
        the natural amplitudes ordered by decreasing magnitude and counted with the 2l + 1 degeneracy of each shell.
        K is the coefficient of the large-n law N_s -> K n^(1/3) for the number of l = 0 natural orbitals among the n
        most occupied ones. All four are positive.

    Raises
    ------
    ValueError
        If omega is not positive or c is negative.
    '''
    return result
```

### Step 4

04_radial_densities

Goal
----
Step 04: Exact densities at a given radius. On-top two-electron density, one-electron density and reduced on-top density at distance r from the trap centre for the pair function Psi.

```python
def radial_densities(omega: float, c: float, r: float) -> "np.ndarray":
    '''Phi(r), rho1(r) and phi(r) = 4 Phi(r) / rho1(r)^2 for the normalized pair function Psi.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).

    Returns
    -------
    result : np.ndarray
        Float array [Phi(r) in bohr^-6, rho1(r) in bohr^-3, phi(r)], each accurate to 1e-10 relative.

    Raises
    ------
    ValueError
        If r is negative.
    '''
    return result
```

### Step 5

05_natural_amplitudes

Goal
----
Step 05: Natural amplitudes of one angular momentum channel. Natural amplitudes of one angular momentum channel of the pair function Psi, normalized with respect to the whole function.

```python
def natural_amplitudes(omega: float, c: float, l: int, n_keep: int) -> "np.ndarray":
    '''Leading natural amplitudes of angular momentum l, ordered by decreasing magnitude.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    l : int
        Angular momentum of the channel, 0 <= l <= 12.
    n_keep : int
        Number of amplitudes returned, 1 <= n_keep <= 80.

    Returns
    -------
    result : np.ndarray
        Float array of shape (n_keep,) with the n_keep amplitudes of largest magnitude of the l channel, signed and
        ordered by decreasing magnitude, normalized so that the amplitudes of all channels, each counted 2l + 1 times,
        have squares summing to one. Each amplitude is accurate to 1e-10 absolute.

    Raises
    ------
    ValueError
        If omega is not positive, c is negative, or l or n_keep lies outside its range.
    '''
    return result
```

### Step 6

06_shell_orbital_densities

Goal
----
Step 06: Shell densities of natural orbitals at a given radius. Natural amplitudes of angular momentum l paired with the summed squares of their natural orbitals at distance r from the centre.

```python
def shell_orbital_densities(omega: float, c: float, l: int, n_keep: int, r: float) -> "np.ndarray":
    '''Pairs (lambda_k, (2l + 1) R_k(r)^2 / (4 pi)) for the n_keep l-channel natural orbitals of largest amplitude magnitude.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.
    l : int
        Angular momentum of the channel, 0 <= l <= 12.
    n_keep : int
        Number of radial functions, 1 <= n_keep <= 80.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).

    Returns
    -------
    result : np.ndarray
        Float array of shape (n_keep, 2); column 0 holds the signed amplitudes ordered by decreasing magnitude and column 1
        the shell sums (2l + 1) R_k(r)^2 / (4 pi) in bohr^-3 of the same orbitals. Column 0 is accurate to 1e-10 absolute
        and column 1 to 1e-6 relative plus 1e-12 absolute for k <= 15.

    Raises
    ------
    ValueError
        If r is negative.
    '''
    return result
```

### Step 7

07_truncated_reduced_ontop

Goal
----
Step 07: Thresholded truncated reduced on-top density. Reduced on-top density at distance r from the centre estimated from the natural orbitals whose occupation numbers reach a threshold.

```python
def truncated_reduced_ontop(omega: float, c: float, r: float, occ_min: float) -> "np.ndarray":
    '''Truncated estimate phi_N(r) from all natural orbitals with occupation lambda_n^2 >= occ_min.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, 0.05 <= omega <= 5 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, 0 <= c <= 0.5.
    r : float
        Distance from the trap centre in bohr, 0 <= r <= 5 omega^(-1/2).
    occ_min : float
        Occupation threshold, 1e-9 <= occ_min <= 0.1; no occupation of the pair function lies within 10 percent of it.

    Returns
    -------
    result : np.ndarray
        Float array [phi_N(r), N, K] where N is the number of retained natural orbitals counting degeneracy and K the number
        of retained s-type orbitals. phi_N(r) is accurate to 1e-6 relative.

    Raises
    ------
    ValueError
        If occ_min lies outside its range.
    '''
    return result
```

### Step 8

08_overestimation_at_density_maximum

Goal
----
Step 08: Overestimation at the radial density maximum and its large-N coefficient (orchestrator). Orchestrator: overestimation of the reduced on-top density by a thresholded natural-orbital expansion at the maximum of the radial electron density, at a finite threshold and in the large-N limit.

```python
def overestimation_at_density_maximum(omega: float, c: float, occ_min: float) -> "np.ndarray":
    '''Radius r* of maximal radial density, the ratio F at threshold occ_min, and the large-N coefficient G.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, 0.05 <= omega <= 5 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, 0 <= c <= 0.5.
    occ_min : float
        Occupation threshold, 1e-9 <= occ_min <= 0.1; no occupation of the pair function lies within 10 percent of it.

    Returns
    -------
    result : np.ndarray
        Float array [r*, F, G]: r* in bohr accurate to 1e-6, F = phi_N(r*) / phi(r*) at the given threshold, accurate
        to 1e-6 relative, and G the coefficient of the large-N law phi_N(r*) / phi(r*) = 1 + G N^(-1/3) + o(N^(-1/3)),
        accurate to 1e-6 relative. G is the value implied by the asymptotics of the weakly occupied natural orbitals
        evaluated with the exact on-top density at r*, not a fit to the finite-threshold sequence.

    Raises
    ------
    ValueError
        If occ_min lies outside its range.
    '''
    return result
```
