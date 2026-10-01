# Material_Science-Semiconductor_Materials-28

## Background

Carrier relaxation, polaron formation and coherent lattice motion in photoexcited semiconductors are governed by the coupled nonequilibrium dynamics of electrons and phonons. An open-quantum-system description propagates only equal-time quantities, the electronic density matrix, the phonon correlators and the coherent phonon amplitude, and its closure decides how memory, dissipative broadening and coherent phonons enter; the driven Holstein dimer is the smallest model in which such a hierarchy can be tested against an exact solution.

## Problem

Carrier relaxation, polaron formation and coherent lattice motion in photoexcited semiconductors are governed by the coupled nonequilibrium dynamics of electrons and phonons, and the smallest model that contains all three is a driven Holstein dimer: two sites, one electron of each spin and a single phonon mode coupled to the site-density difference. A recent source derives, directly from the density matrix of the electron-phonon system, a closed set of five coupled equations of motion for equal-time quantities, the electronic one-body reduced density matrix, the phonon density matrix, the anomalous phonon correlator, the electron-phonon correlation and the coherent phonon amplitude, in which memory, dissipative broadening and coherent-phonon dynamics arise on an equal footing without two-time correlators, and it benchmarks the hierarchy against the exact solution of the strongly driven dimer. Recover that construction from the source and evaluate it exactly. The load-bearing choices are the source's and are not derivable from the statement below: how the hierarchy is closed at second order in the coupling and which terms of the correlation equation carry the spontaneous and the stimulated phonon processes; how the coherent phonon enters the one-body Hamiltonian; how the two identical spins enter the phonon and coherent-phonon equations; what the energy functional of the hierarchy is and why it is conserved once the drive is over; and what separates the coherent level of the theory from the full hierarchy.

The model is the source's. In units of the hopping energy $t_{hop}$ and of $\hbar = 1$, the Hamiltonian is $\hat H = -t_{hop}\sum_{\sigma}(\hat c^\dagger_{1\sigma}\hat c_{2\sigma} + \hat c^\dagger_{2\sigma}\hat c_{1\sigma}) + \omega_{ph}\hat b^\dagger\hat b + \frac{g}{\sqrt 2}(\hat b^\dagger + \hat b)(\hat n_1 - \hat n_2) + \hat H^{ext}(t)$ with one electron of each spin, the dimensionless parameters $\gamma = \omega_{ph}/t_{hop}$ and $\lambda = 2g^2/(t_{hop}\,\omega_{ph})$, and the drive $\hat H^{ext}(t) = v\,(\hat n_1 - \hat n_2)\,\sin(\pi t/4)\,e^{-t^2/(2T_p^2)}$ with $T_p = 1/t_{hop}$, switched on at $t = 0$. The phonon space is truncated at the occupation $n_{ph}$ and the exact dynamics is the time-dependent Schrodinger equation in the product basis of the two site indices and the phonon number. The reduced quantities are, for one spin, $\rho_{ij} = \langle \hat c^\dagger_j \hat c_i\rangle$, the coherent amplitude $B = \langle \hat b\rangle$, the excess phonon number $\delta\rho_{qq} = \langle \hat b^\dagger \hat b\rangle - |B|^2$, the anomalous correlator $\delta\bar\rho = \langle \hat b\hat b\rangle - B^2$ and the electron-phonon correlation $\delta\rho^q_{ij} = \langle \hat c^\dagger_j\hat c_i\hat b\rangle - \rho_{ij}B$; the site-1 occupation of one spin is $\rho_{11}$. Two levels of the theory are propagated: the coherent level, in which only the one-body density matrix and the coherent amplitude evolve and the correlations are held at zero, and the full hierarchy of the source's five equations. Both are initialised with the reduced quantities of the exact ground state of the dimer, so that the comparison with the exact dynamics isolates the propagation; the source instead starts its hierarchy at a stationary point of its equations.

Implement eight functions with these conventions: the basis index of the exact problem is $(2 s_\uparrow + s_\downarrow)(n_{ph}+1) + n$ with $s_\sigma \in \{0, 1\}$ the site of the electron of spin $\sigma$ and $n$ the phonon number; the hierarchy is carried as a real vector of 17 numbers, $\rho_{11}, \rho_{22}, \mathrm{Re}\,\rho_{12}, \mathrm{Im}\,\rho_{12}, \delta\rho_{qq}, \mathrm{Re}\,\delta\bar\rho, \mathrm{Im}\,\delta\bar\rho$, the four real parts of $\delta\rho^q$ in row-major order, its four imaginary parts, $\mathrm{Re}\,B, \mathrm{Im}\,B$; every propagation, exact or approximate, is converged to a relative tolerance of $10^{-12}$ and is deterministic; energies are expectation values of the time-independent Hamiltonian, in units of $t_{hop}$.

`hd_hamiltonian(lam, gam, nph)` returns, shape $(2, D, D)$ with $D = 4(n_{ph}+1)$, the time-independent Hamiltonian and the drive operator $\hat n_1 - \hat n_2$. `hd_ground_state(lam, gam, nph)` returns the exact ground-state energy, the gap to the first excited state, and the ground-state values of $\rho_{12}$, $\langle\hat b^\dagger\hat b\rangle$, $\langle\hat b\hat b\rangle$, $\rho^q_{11}$ and $\rho^q_{12}$. `hd_exact_dynamics(lam, gam, nph, v, tf)` returns, after the exact propagation to tf, $\rho_{11}$, the real and imaginary parts of $\rho_{12}$, the phonon number, the real and imaginary parts of $B$, and the energy. `hd_eom_rhs(t, state, lam, gam, v, full)` returns the time derivative of the packed state under the source's equations, the full hierarchy for full = 1 and the coherent level for full = 0. `hd_initial_state(lam, gam, nph, full)` returns the packed initial state at the chosen level. `hd_energy(state, lam, gam)` returns the source's energy functional of the hierarchy split into electronic, phonon and electron-phonon parts, and their sum. `hd_eom_dynamics(lam, gam, nph, v, tf, full)` returns the packed state at tf. `hd_audit(lam, gam, nph, v, tf)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns ten values: the exact ground-state energy and gap; $\rho_{11}$ at tf and the energy after the drive from the exact dynamics; $\rho_{11}$ at tf from the coherent level; $\rho_{11}$ at tf, the energy after the drive, the excess phonon number at tf and the real part of $B$ at tf from the full hierarchy; and the difference between the full-hierarchy and the exact $\rho_{11}$ at tf.

All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, on a negative coupling, a non-positive phonon frequency ratio, a non-integral or non-positive phonon cutoff, a negative drive amplitude or final time, a packed state of the wrong shape, or a level flag other than 0 or 1; the orchestrator also raises when the energy functional of the full hierarchy evaluated in its initial state differs from the exact ground-state energy, when the trace of a propagated density matrix departs from one, when the energy of the full hierarchy is not conserved between $t = 8$ and tf, or when tf is below 8.

Evaluate the audit at $\lambda = 0.5$, $\gamma = 0.5$, $n_{ph} = 30$, $v = t_{hop}$, $t_f = 30/t_{hop}$.

In your reasoning report the conventions you used, and justify each from the source: how the hierarchy is closed and which terms of the electron-phonon correlation equation carry the spontaneous and the stimulated phonon processes; how the coherent phonon amplitude enters the one-body Hamiltonian and why the two spins double its source; what the source's energy functional is and that it is conserved after the drive; how the source's drive is defined; and what the source finds the coherent level to miss relative to the full hierarchy.

Report numerically, as evidence that the chain was executed: the exact ground-state energy and gap; the site-1 occupation at $t_f$ and the energy after the drive from the exact dynamics; the site-1 occupation at $t_f$ from the coherent level; the site-1 occupation at $t_f$, the energy after the drive, the excess phonon number at $t_f$ and the real part of the coherent amplitude at $t_f$ from the full hierarchy; and the deviation of the full-hierarchy occupation from the exact one. These are the scalars that determine the final number.

As the final answer, report the site-1 occupation of one spin at $t_f$ from the full hierarchy, to six significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Report the numerical evidence requested above concisely.
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

hd_hamiltonian

Goal
----
Builds the Hamiltonian and the drive operator of the two-electron Holstein dimer in a truncated phonon Fock space.

```python
def hd_hamiltonian(lam: float, gam: float, nph: int) -> "np.ndarray":
    r"""lam: non-negative float, the dimensionless coupling $\lambda = 2g^2/(t_{hop}\,\omega_{ph})$. gam: positive
    float, the ratio $\gamma = \omega_{ph}/t_{hop}$. nph: positive integer, the highest phonon occupation kept.

    Returns a numpy float64 array of shape $(2, D, D)$ with $D = 4(nph+1)$: the time-independent
    Hamiltonian of the driven Holstein dimer with one electron of each spin, in units of $t_{hop}$,
    and the drive operator $\hat n_1 - \hat n_2$, both in the product basis whose index is
    $(2 s_\uparrow + s_\downarrow)(nph+1) + n$, where $s_\sigma \in \{0, 1\}$ is the site (0 for site 1) of
    the electron of spin $\sigma$ and $n$ the phonon number. The Hamiltonian carries the hopping
    $-t_{hop}$ between the sites for each spin, the phonon energy $\omega_{ph}\,\hat b^\dagger\hat b$ and
    the coupling $(g/\sqrt 2)(\hat b^\dagger + \hat b)(\hat n_1 - \hat n_2)$, with the phonon space
    truncated at occupation nph.

    Raises:
        ValueError: on a negative or non-finite lam, a non-positive or non-finite gam, or a
            non-integral or non-positive nph.
    """
    return None
```

### Step 2

hd_ground_state

Goal
----
Computes the exact ground state of the dimer and the reduced quantities of the source's hierarchy evaluated in it.

```python
def hd_ground_state(lam: float, gam: float, nph: int) -> "np.ndarray":
    r"""Parameters as in the Hamiltonian step.

    Returns a numpy float64 array of shape $(7,)$: the exact ground-state energy, the gap to the first
    excited state, and the reduced quantities of the ground state that the source's hierarchy
    propagates, for one spin: the off-diagonal element $\rho_{12} = \langle \hat c^\dagger_2 \hat c_1\rangle$
    of the one-body reduced density matrix, the phonon number $\langle \hat b^\dagger \hat b\rangle$, the
    anomalous phonon correlator $\langle \hat b \hat b\rangle$, and the electron-phonon correlators
    $\rho^q_{11} = \langle \hat c^\dagger_1 \hat c_1 \hat b\rangle$ and $\rho^q_{12} = \langle \hat c^\dagger_2
    \hat c_1 \hat b\rangle$, all real in the ground state.

    Raises:
        ValueError: on invalid lam, gam or nph.
    """
    return None
```

### Step 3

hd_exact_dynamics

Goal
----
Propagates the exact ground state of the dimer under the source's Gaussian-damped drive and reports the reduced observables.

```python
def hd_exact_dynamics(lam: float, gam: float, nph: int, v: float, tf: float) -> "np.ndarray":
    r"""lam, gam, nph: as in the Hamiltonian step. v: non-negative float, the drive amplitude in units of
    $t_{hop}$. tf: non-negative float, the final time in units of $1/t_{hop}$.

    Returns a numpy float64 array of shape $(7,)$: after propagating the exact ground state with the
    time-dependent Schrodinger equation under the drive $v\,(\hat n_1 - \hat n_2)\sin(\pi t/4)\,e^{-t^2/2}$ from
    $t = 0$ to tf, the site-1 occupation $\rho_{11}$ of one spin, the real and imaginary parts of
    $\rho_{12} = \langle \hat c^\dagger_2 \hat c_1\rangle$, the phonon number, the real and imaginary parts of
    the coherent phonon amplitude $\langle \hat b\rangle$, and the expectation value of the time-independent
    Hamiltonian. The propagation is converged to a relative tolerance of $10^{-12}$ and is
    deterministic.

    Raises:
        ValueError: on invalid lam, gam or nph, or on a negative or non-finite v or tf.
    """
    return None
```

### Step 4

hd_eom_rhs

Goal
----
Evaluates one time derivative of the source's second-order equal-time electron-phonon hierarchy for the driven spin-symmetric Holstein dimer, or of its coherent restriction.

```python
def hd_eom_rhs(t: float, state: "np.ndarray", lam: float, gam: float, v: float, full: int) -> "np.ndarray":
    r"""Evaluate the source's five coupled EOMs for the one-mode spin-symmetric Holstein dimer.

    Parameters
    ----------
    t : float
        Finite time in units of $t_{hop}^{-1}$.
    state : "np.ndarray"
        Shape (17,), finite. The packing is
        $\rho_{11}, \rho_{22}, \mathrm{Re}\,\rho_{12}, \mathrm{Im}\,\rho_{12},
        \delta\rho_{qq}, \mathrm{Re}\,\delta\bar\rho, \mathrm{Im}\,\delta\bar\rho$,
        then the four real parts and four imaginary parts of $\delta\rho^q_{ij}$ in row-major
        $(i,j)$ order, followed by $\mathrm{Re}\,B, \mathrm{Im}\,B$.
    lam : float
        Dimensionless electron-phonon coupling $\lambda \ge 0$.
    gam : float
        Positive phonon-frequency ratio $\gamma$.
    v : float
        Non-negative drive amplitude.
    full : int
        1 for the full hierarchy of source Eqs. (6a)-(6e); 0 for the coherent restriction
        containing only the electronic 1-RDM and coherent phonon amplitude.

    Notes
    -----
    Use $\omega=\gamma$, $g=\sqrt{\lambda\gamma/2}$ and
    $G=(g/\sqrt{2})\,\mathrm{diag}(1,-1)$. Apply the source's Eq. (7) for the effective
    one-body Hamiltonian with the stated oscillatory Gaussian drive. Reduce Eqs. (6a)-(6e)
    exactly to this single-mode, two-identical-spin dimer. Preserve the source's index order,
    complex conjugations, spontaneous/stimulated terms and spin-degeneracy factors. When
    ``full == 0``, the connected quantities $\delta\rho_{qq}$, $\delta\bar\rho$ and
    $\delta\rho^q$ are held fixed at zero rather than evolved.

    Returns
    -------
    "np.ndarray"
        Float64 array of shape (17,), the derivative in exactly the same packing as ``state``.

    Raises
    ------
    ValueError
        If ``t`` is non-finite; ``state`` does not contain exactly 17 finite values; ``lam`` is
        negative; ``gam`` is non-positive; ``v`` is negative; or ``full`` is not 0 or 1.
    """
    return None
```

### Step 5

hd_initial_state

Goal
----
Packs the exact ground state's reduced quantities into the initial state of the hierarchy at the chosen level.

```python
def hd_initial_state(lam: float, gam: float, nph: int, full: int) -> "np.ndarray":
    r"""Parameters as before; full: 0 or 1.

    Returns a numpy float64 array of shape $(17,)$: the packed state of the hierarchy evaluated in the
    exact ground state of the dimer, the initial condition of the propagation; with full = 0 the
    correlation entries (excess phonon number, anomalous correlator, electron-phonon correlation) are
    zero and only the density matrix and the coherent amplitude are kept.

    Raises:
        ValueError: on invalid lam, gam or nph, or a full flag other than 0 or 1.
    """
    return None
```

### Step 6

hd_energy

Goal
----
Evaluates the source's energy functional of the hierarchy, split into electronic, phonon and electron-phonon parts.

```python
def hd_energy(state: "np.ndarray", lam: float, gam: float) -> "np.ndarray":
    r"""state: array-like of shape $(17,)$, the packed state. lam, gam: as before.

    Returns a numpy float64 array of shape $(4,)$: the source's energy functional of the hierarchy,
    the expectation value of the time-independent Hamiltonian written in the propagated quantities:
    the electronic part $2\,\mathrm{Tr}(h_0\rho)$ with $h_0 = -t_{hop}\sigma_x$, the phonon part
    $\omega(|B|^2 + \delta\rho_{qq})$, the electron-phonon part $4\,\mathrm{Re}\sum_i G_{ii}(B\rho_{ii} +
    \delta\rho^q_{ii})$, and their sum, the factors of two again counting the two spins.

    Raises:
        ValueError: on a state of the wrong shape or with non-finite entries, or invalid lam or gam.
    """
    return None
```

### Step 7

hd_eom_dynamics

Goal
----
Propagates the hierarchy from the exact ground state's reduced quantities under the drive at the chosen level of theory.

```python
def hd_eom_dynamics(lam: float, gam: float, nph: int, v: float, tf: float, full: int) -> "np.ndarray":
    r"""Parameters as before.

    Returns a numpy float64 array of shape $(17,)$: the packed state of the hierarchy at time tf,
    obtained by propagating the initial state of the previous step under the source's equations
    with the same drive as the exact dynamics, converged to a relative tolerance of $10^{-12}$ and
    deterministic.

    Raises:
        ValueError: on invalid lam, gam, nph, v or tf, or a full flag other than 0 or 1.
    """
    return None
```

### Step 8

hd_audit

Goal
----
Runs the whole chain from the exact ground state through the exact, coherent and full-hierarchy dynamics and reports the observables side by side.

```python
def hd_audit(lam: float, gam: float, nph: int, v: float, tf: float) -> "np.ndarray":
    r"""Parameters as before.

    The orchestrator. It must call the earlier functions rather than reimplementing them.
    Returns a numpy float64 array of shape $(10,)$: the exact ground-state energy and gap; the
    site-1 occupation of one spin at tf and the energy after the drive from the exact dynamics; the
    site-1 occupation at tf from the coherent level; the site-1 occupation, the energy after the
    drive, the excess phonon number and the real part of the coherent amplitude at tf from the full
    hierarchy; and the site-1 occupation at tf from the full hierarchy minus that of the exact
    dynamics.

    Raises:
        ValueError: whenever any of the functions it calls would raise; when the energy functional of
            the full hierarchy evaluated in its initial state differs from the exact ground-state energy
            by more than 1e-9; when the trace of the propagated density matrix of either level departs
            from one by more than 1e-9; and when the energy of the full hierarchy at tf differs from its
            value at time 8, after the drive has decayed, by more than 1e-8 (tf at least 8).
    """
    return None
```
