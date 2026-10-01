# Physics-Condensed_Matter_Physics-32

## Background

Topological phases of non-interacting fermions are classified through the ground state of a quadratic Hamiltonian, for example by a Chern number or a winding number of the occupied bands. At nonzero temperature, or in a driven-dissipative steady state, the system is described instead by a mixed state. For free fermions this mixed state is Gaussian and fully determined by its single-particle correlation matrix. When the spectrum of that correlation matrix keeps a gap around one half, the so-called purity gap, the mixed state can be adiabatically connected to a pure state and inherits its topological classification. Diagnosing topology directly from the thermal state is then a well-posed problem, but different diagnostics behave very differently with temperature and system size.

In one dimension, a central tool is Resta's many-body twist operator, the exponential of the total dipole moment in units of the chain length. Its ground-state expectation value yields the electric polarisation, and in an inversion-symmetric chain that polarisation is quantised and reproduces the Zak phase of the occupied band. Evaluated in a thermal state, the phase of the same expectation value defines a mixed-state geometric phase. A known difficulty with such large-gauge operators is that the modulus of their expectation value can vanish in the thermodynamic limit, even when the phase remains meaningful, which makes the global quantity hard to measure in large systems.

The Su-Schrieffer-Heeger chain is the standard testing ground. It has two sublattices with alternating hopping, a trivial phase and a topological phase with charge centres on the bonds. Adding an inversion-symmetric longer-range hop produces a further sector with the opposite winding. Because the charge centres sit at distinct inversion-symmetric positions in the different phases, one can also ask whether local operators, acting on only a few neighbouring sites, capture the same information. That would suit quantum-gas microscopes that read out site occupations one snapshot at a time. Computing thermal expectation values in such chains is simple in principle, since quadratic Hamiltonians reduce everything to single-particle matrices. In practice, exponentially large and small factors in inverse temperature and system size make naive formulas overflow or cancel.

## Problem

The Su-Schrieffer-Heeger (SSH) chain with inversion-symmetric next-nearest-neighbour hopping hosts a trivial phase and two topological phases with winding numbers +1 and -1. At nonzero temperature its thermal state is a mixed Gaussian state, and topology can be diagnosed from the ensemble geometric phase, the phase of the thermal expectation value of Resta's many-body twist operator. That phase stays quantised to 0 or pi by inversion symmetry, but the modulus of the same expectation value decays exponentially with system size at any nonzero temperature, so for a chain of given length the global diagnostic stops being measurable below some inverse temperature. Local twist operators acting on two cells, on an intercell bond or on a next-nearest-neighbour link remain usable, and the signs of differences of their magnitudes at the centre of the chain identify the phase. Given the hopping amplitudes, the number of unit cells and a threshold modulus, the calculation asked for here returns the inverse temperature at which the modulus of the global twist expectation value falls to that threshold.

The chain has N unit cells j = 0, ..., N - 1 with orbitals A_j and B_j and periodic boundary conditions, and its Hamiltonian is H = -sum_j (t1 a_j^dag b_j + t2 b_j^dag a_{j+1} + t3 a_j^dag b_{j+1} + h.c.), with cell indices modulo N. The thermal state is rho = exp(-beta H) / Tr exp(-beta H) at zero chemical potential, in the full Fock space (particle number not fixed). The global twist operator is T = exp(i (2 pi / N) sum_j j n_j), with n_j the total occupation of cell j, that is, cell positions x_j = j and no intracell offset. The local twist operators place A_l at x_l - delta_x and B_l at x_l + delta_x with delta_x = 1/4, and multiply exp(i (2 pi / N) x) over the orbitals they act on. T^intra2_c acts on A_c, B_c, A_{c+1} and B_{c+1}. T^inter_c acts on B_c and A_{c+1}, and T^nnn_c acts on A_c and B_{c+1}. The centre cell is c = (N - 1)/2 for odd N, and for a cell index taken modulo N the position used is the unreduced j + 1. The local indicators are Delta_T3 = |<T^intra2_c>| - max(|<T^inter_c>|, |<T^nnn_c>|) and Delta_I = |<T^inter_c>| - |<T^nnn_c>|.

Use a chain of N = 20001 unit cells with t1 = 1.0, t2 = 0.6 and t3 = 1.5, which lies in the sector with winding number -1. Find the inverse temperature beta* at which |<T>| = 0.01; on 2 <= beta <= 30, ln|<T>| rises with beta and crosses ln 0.01 exactly once. At beta* the ensemble geometric phase arg <T> must equal pi, and the local indicators, evaluated for a separate periodic chain of 151 unit cells with the same hoppings at the same beta* (so N = 151 in their phases and centre cell c = 75), must be negative, confirming the sector. The chain is too long for linear algebra on 2N-by-2N matrices, and at these inverse temperatures |<T>|, Tr exp(-beta H) and the individual Boltzmann factors are not representable in double precision, so all quantities must be evaluated in logarithmic form to an accuracy of about 1e-9 in ln|<T>|.

What to report: beta* as the final answer. In <reasoning>, also state ln|<T>| and arg <T> at beta = 8 and at beta = 12 for the N = 20001 chain, its low-temperature limit of ln|<T>| (the value at beta = 50, which agrees with the zero-temperature value to better than 1e-12), Delta_T3 and Delta_I for the 151-cell chain at beta*, the value beta* takes for the plain SSH chain with t1 = 1.0, t2 = 2.0, t3 = 0 and N = 20001, and name the sources you relied on and say what each supplied.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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
import math
```

### Step 1

extended_ssh_hamiltonian

Goal
----
Build the single-particle Hamiltonian matrix of the extended Su-Schrieffer-Heeger chain in real space.

```python
def extended_ssh_hamiltonian(n_cells: int, t1: float, t2: float, t3: float, periodic: bool) -> "np.ndarray":
    '''Return the real symmetric single-particle Hamiltonian of the extended SSH chain.

    The chain has n_cells unit cells j = 0, ..., N - 1, each with orbitals A_j and B_j,
    ordered in the basis as (A_0, B_0, A_1, B_1, ..., A_{N-1}, B_{N-1}). The many-body
    Hamiltonian is

        H = - sum_j ( t1 a_j^dag b_j + t2 b_j^dag a_{j+1} + t3 a_j^dag b_{j+1} + h.c. ),

    so the matrix element between orbitals coupled by an amplitude t is -t. With
    periodic = True the cell index j + 1 is taken modulo N; with periodic = False every
    term that would involve cell N is omitted. Couplings between the same pair of
    orbitals add.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2 (numpy integers accepted; bool is not).
    t1, t2, t3 : float
        Intracell, intercell and next-nearest-neighbour hopping amplitudes, finite and >= 0.
    periodic : bool
        Periodic (True) or open (False) boundary conditions.

    Returns
    -------
    h : np.ndarray
        Float array of shape (2N, 2N) in the stated basis order.

    Raises
    ------
    ValueError
        If n_cells is a bool, not an integer, or smaller than 2, or any hopping amplitude
        is negative or not finite.
    '''
    return h
```

### Step 2

fermi_correlation_matrix

Goal
----
Compute the single-particle correlation matrix of a free-fermion thermal state at any inverse temperature, including zero temperature.

```python
def fermi_correlation_matrix(h: "np.ndarray", beta: float) -> "np.ndarray":
    '''Return the correlation matrix f = (1 + exp(beta h))^{-1} of the thermal state.

    h is a real symmetric single-particle Hamiltonian with eigen-decomposition
    h = sum_n e_n v_n v_n^T. First replace by exactly zero every eigenvalue with
    |e_n| <= 1e-12 * max(1, max_n |e_n|). Then f = sum_n p_n v_n v_n^T with occupation
    p_n = 1 / (1 + exp(beta e_n)); for beta = inf, p_n is 1, 1/2 or 0 for e_n < 0, = 0 or
    > 0. The occupations must be evaluated without overflow for every beta in [0, inf],
    including beta = 1e300, and must be exactly 1/2 for zeroed eigenvalues at any beta.
    The result does not depend on the choice of eigenvectors within degenerate subspaces.

    Parameters
    ----------
    h : np.ndarray
        Finite real symmetric matrix of shape (M, M), M >= 1.
    beta : float
        Inverse temperature, 0 <= beta <= inf (not NaN).

    Returns
    -------
    f : np.ndarray
        Real symmetric float array of shape (M, M).

    Raises
    ------
    ValueError
        If h is not a finite real square symmetric matrix (tolerance 1e-12 on
        h - h^T), or beta is negative or NaN.
    '''
    return f
```

### Step 3

log_partition_function

Goal
----
Compute the logarithm of the grand-canonical partition function of free fermions from their single-particle energies.

```python
def log_partition_function(energies: "np.ndarray", beta: float) -> float:
    '''Return ln Z = sum_n ln(1 + exp(-beta e_n)) for single-particle energies e_n.

    First replace by exactly zero every energy with |e_n| <= 1e-12 * max(1, max_n |e_n|).
    The sum must be evaluated without overflow or loss of the O(1) part: the result
    must be accurate to relative 1e-12 whenever it is finite, including beta * |e_n| up
    to 1e300.

    Parameters
    ----------
    energies : np.ndarray
        Nonempty finite real 1-D array of single-particle energies.
    beta : float
        Inverse temperature, finite and >= 0.

    Returns
    -------
    log_z : float
        ln Z.

    Raises
    ------
    ValueError
        If energies is not a nonempty finite real 1-D array, or beta is negative, NaN or
        infinite.
    '''
    return log_z
```

### Step 4

bloch_hamiltonian

Goal
----
Evaluate the two-band Bloch Hamiltonian of the translation-invariant extended SSH chain at a crystal momentum.

```python
def bloch_hamiltonian(k: float, t1: float, t2: float, t3: float) -> "np.ndarray":
    '''Return H(k) = -[[0, h(k)], [conj(h(k)), 0]] with h(k) = t1 + t2 exp(-i k) + t3 exp(+i k).

    This is the Bloch form of the step-01 Hamiltonian with periodic boundary conditions
    in the basis (A, B), with Bloch phases attached to unit-cell positions only (no
    intracell phase).

    Parameters
    ----------
    k : float
        Crystal momentum, finite.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01, finite and >= 0.

    Returns
    -------
    parts : np.ndarray
        Float array of shape (2, 2, 2): parts[0] = Re H(k), parts[1] = Im H(k).

    Raises
    ------
    ValueError
        If k is not finite or a hopping amplitude is negative or not finite.
    '''
    return parts
```

### Step 5

global_twist_expectation

Goal
----
Evaluate the thermal expectation value of the many-body twist (large-gauge translation) operator of the periodic SSH chain in logarithmic polar form.

```python
def global_twist_expectation(n_cells: int, t1: float, t2: float, t3: float, beta: float) -> "np.ndarray":
    '''Return [ln|<T>|, arg <T>] for the thermal state of the periodic step-01 chain.

    The state is rho = exp(-beta H) / Tr exp(-beta H) for the periodic extended SSH chain
    with N = n_cells cells, and the twist operator is T = exp(i (2 pi / N) sum_j j n_j),
    with n_j = a_j^dag a_j + b_j^dag b_j the occupation of cell j (unit-cell positions
    x_j = j, no intracell offset). Return the natural logarithm of |<T>| and the phase
    arg <T> in (-pi, pi]; a computed phase within 1e-9 of -pi is returned as +pi.
    The result must be accurate to 1e-9 in ln|<T>| and in the phase, and finite,
    for chains of up to N = 20001 cells and for beta * max|e| up to 1e4, where
    |<T>|, Tr exp(-beta H) and the individual transfer factors are not representable in
    double precision. The Bloch spectrum is assumed gapped (h(k) of step 04 never
    vanishes). The momentum mesh must resolve the occupied band: for normalized
    negative-energy eigenvectors u_m of h(k_m), with k_m = 2 pi m / N and u_N = u_0,
    require |u_(m+1)^dag u_m| >= 0.1 at every adjacent pair, including closure.
    Raise ValueError if any overlap magnitude is below 0.1.
    The supported domain also excludes severe cancellation at a twist zero:
    let lambda_1, lambda_2 be the eigenvalues of
    P = exp(-beta h(k_(N-1))) ... exp(-beta h(k_0)), with k_m = 2 pi m / N,
    and s = (-1)^(N-1). Both cancellation ratios
    r_i = |1 + s lambda_i| / (1 + |lambda_i|) must be at least 0.1.
    If either ratio is below 0.1, raise ValueError, even if <T> is nonzero.
    Both restrictions are dimensionless and invariant under eigenvector phase
    choices. They retain this task's large-N, low-temperature regime and beta = 0
    for odd N on a resolved mesh, but exclude even-N cancellation as beta tends to zero.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature, finite and >= 0.

    Returns
    -------
    value : np.ndarray
        Float array [ln|<T>|, arg <T>].

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 2, a hopping amplitude is invalid, or beta is
        negative, NaN or infinite, or an adjacent occupied-band overlap magnitude
        or a transfer cancellation ratio is below 0.1.
    '''
    return value
```

### Step 6

diagonal_twist_expectation

Goal
----
Evaluate the thermal expectation value of an operator diagonal in site occupations, exp(i sum_i theta_i n_i), from the single-particle correlation matrix of a Gaussian state.

```python
def diagonal_twist_expectation(f: "np.ndarray", theta: "np.ndarray") -> "np.ndarray":
    '''Return [ln|<O>|, arg <O>] for O = exp(i sum_i theta_i n_i) in the Gaussian state with correlations f.

    f is the real symmetric correlation matrix <c_i^dag c_j> of a number-non-conserving
    free-fermion thermal state (for example step 02), and n_i = c_i^dag c_i. Return the
    natural logarithm of |<O>| and arg <O> in (-pi, pi]; a computed phase within 1e-9
    of -pi is returned as +pi. ln|<O>| must be finite and accurate to 1e-9 whenever
    <O> != 0, including when |<O>| is far below the smallest positive double.

    Parameters
    ----------
    f : np.ndarray
        Finite real symmetric array of shape (M, M) with eigenvalues in [0, 1].
    theta : np.ndarray
        Finite real array of shape (M,) of phases (radians).

    Returns
    -------
    value : np.ndarray
        Float array [ln|<O>|, arg <O>].

    Raises
    ------
    ValueError
        If f is not a finite real square symmetric matrix, theta has the wrong shape or
        is not finite, or <O> = 0 exactly.
    '''
    return value
```

### Step 7

local_twist_phases

Goal
----
Construct the site phases of the local twist operators that act within a unit cell, across a bond, or along the next-nearest-neighbour link of the SSH chain.

```python
def local_twist_phases(n_cells: int, kind: str, j: int, delta_x: float) -> "np.ndarray":
    '''Return theta (length 2N) such that the local twist operator is exp(i sum_i theta_i n_i).

    Orbitals are ordered as in step 01. Cell l has position x_l = l, orbital A_l sits at
    x_l - delta_x and B_l at x_l + delta_x, and delta_k = 2 pi / N. Writing u(orbital, x)
    for "add delta_k * x to the entry of that orbital", the operators are

    - "intra":  u(A_j, j - delta_x), u(B_j, j + delta_x)
    - "inter":  u(B_j, j + delta_x), u(A_{j+1}, (j + 1) - delta_x)
    - "nnn":    u(A_j, j - delta_x), u(B_{j+1}, (j + 1) + delta_x)
    - "intra2": u(A_j, j - delta_x), u(B_j, j + delta_x), u(A_{j+1}, (j + 1) - delta_x),
                u(B_{j+1}, (j + 1) + delta_x)

    where the orbital index j + 1 is taken modulo N but the position (j + 1) is not
    reduced (for j = N - 1 it is N). Entries not touched are zero.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2.
    kind : str
        One of "intra", "inter", "nnn", "intra2".
    j : int
        Cell index, 0 <= j <= N - 1.
    delta_x : float
        Intracell half-separation of the orbitals, finite.

    Returns
    -------
    theta : np.ndarray
        Float array of shape (2N,).

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 2, kind is not recognised, j is out of range or
        not an integer, or delta_x is not finite.
    '''
    return theta
```

### Step 8

local_indicators

Goal
----
Compute the local topological indicators of the extended SSH chain at nonzero temperature from centre-of-chain twist expectation values.

```python
def local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    '''Return [Delta_T3, Delta_I] for the periodic chain at inverse temperature beta.

    Use the periodic step-01 Hamiltonian, the step-02 correlation matrix at beta, the
    step-07 phases with centre cell c = (N - 1) / 2 (N odd), and the step-06 expectation
    values. With |<O>| the modulus of each expectation value,

        Delta_T3 = |<T^intra2_c>| - max(|<T^inter_c>|, |<T^nnn_c>|),
        Delta_I  = |<T^inter_c>| - |<T^nnn_c>|.

    Parameters
    ----------
    n_cells : int
        Odd number of unit cells N >= 3.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature as in step 02.
    delta_x : float
        Intracell half-separation as in step 07.

    Returns
    -------
    indicators : np.ndarray
        Float array [Delta_T3, Delta_I].

    Raises
    ------
    ValueError
        If n_cells is even or smaller than 3, or the inputs are invalid as in steps 01, 02
        and 07.
    '''
    return indicators
```

### Step 9

bloch_local_indicators

Goal
----
Compute the local topological indicators of a very long periodic extended SSH chain without forming any matrix of the size of the chain.

```python
def bloch_local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    '''Return [Delta_T3, Delta_I] exactly as defined in step 08, for chains of up to 20001 cells.

    The chain, thermal state (including the step-02 zero-mode rule and beta = inf), twist
    phases (step 07, centre cell c = (N - 1) / 2) and indicators are exactly those of step 08;
    only the admissible size differs. The result must agree with step 08 to 1e-10 wherever
    step 08 can be evaluated, and must be computable for N up to 20001, where the
    2N-by-2N correlation matrix cannot be formed.

    Parameters
    ----------
    n_cells : int
        Odd number of unit cells N >= 3.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature, 0 <= beta <= inf, as in step 02.
    delta_x : float
        Intracell half-separation, finite, as in step 07.

    Returns
    -------
    indicators : np.ndarray
        Float array [Delta_T3, Delta_I].

    Raises
    ------
    ValueError
        If n_cells is even or smaller than 3, or the inputs are invalid as in steps 01, 02
        and 07.
    '''
    return indicators
```

### Step 10

twist_crossing_beta

Goal
----
Find the inverse temperature at which the modulus of the global twist expectation value falls to a prescribed small value, and confirm the topological sector locally.

```python
def twist_crossing_beta(n_cells: int, t1: float, t2: float, t3: float, target: float,
                        beta_bracket: "np.ndarray", delta_x: float, n_local: int) -> float:
    '''Return beta* in the bracket with |<T>| = target, after a local consistency check.

    With the step-05 value, the bracket (b_lo, b_hi) satisfies ln|<T>| < ln(target) at b_lo
    and > ln(target) at b_hi, and for every input used with this function ln|<T>| crosses
    ln(target) exactly once inside the bracket. All global-twist evaluations must lie
    in step 05's supported overlap/cancellation domain. Reduce the bisection bracket width
    to 1e-12, or stop when its endpoints
    are adjacent representable floating-point values if that width is unattainable.
    Return the midpoint of the final bracket.
    Then, with the step-05 phase phi at beta* and the step-09 indicators at beta* for a
    chain of n_local cells with the same hoppings and the given delta_x, verify consistency.
    For n_local <= 151, also require agreement of steps 08 and 09 to absolute 1e-10
    and use the step-08 indicators in the consistency check.
    Let p = 0 for odd N and p = pi for even N (the parity factor (-1)^(N-1)). If phi is within
    1e-6 of p (modulo 2 pi) the chain must be trivial, Delta_T3 > 0; if phi is within 1e-6
    of p + pi (modulo 2 pi) it must be topological, Delta_T3 < 0; any other phase, or a
    failed check, raises ValueError.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2 for the global twist.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    target : float
        Threshold modulus, 0 < target < 1.
    beta_bracket : np.ndarray
        Inverse temperatures (b_lo, b_hi) with 0 <= b_lo < b_hi < inf.
    delta_x : float
        Intracell half-separation for the local indicators, as in step 07.
    n_local : int
        Odd number of unit cells >= 3 of the chain used for the local indicators (step 09).

    Returns
    -------
    beta_star : float
        Inverse temperature at which |<T>| = target.

    Raises
    ------
    ValueError
        If target is not in (0, 1), the bracket is not increasing or does not enclose the
        crossing, either consistency check fails, or the inputs are invalid as in steps 05
        and 09 (with n_local in place of n_cells for step 09).
    '''
    return beta_star
```
