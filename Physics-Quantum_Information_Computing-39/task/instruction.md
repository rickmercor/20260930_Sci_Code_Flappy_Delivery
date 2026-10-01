# Physics-Quantum_Information_Computing-39

## Background

Collective coupling to a common environment can generate correlations between qubits even without a direct qubit-qubit interaction. Such entanglement can be strongest during a transient and then weaken as the system relaxes. Periodic driving offers a way to sustain part of that transient behavior, but identifying a useful resonance requires both spectral information and a description that retains environmental memory. Auxiliary-space descriptions connect these questions by representing the same open process through stationary states, decaying modes, and periodic propagators.

## Problem

Two identically driven qubits can retain entanglement associated with a decaying collective mode when their common environment retains memory; determine the strongest stabilized concurrence in the finite calculation specified here. Work with $\hbar=1$, inverse-time unit $\Omega=1$, the symmetric triplet basis $(|00\rangle,(|01\rangle+|10\rangle)/\sqrt{2},|11\rangle)$, and an auxiliary oscillator with exactly $N=4$ number states and $a_{n-1,n}=\sqrt{n}$:

$$
J_x=\frac{1}{\sqrt{2}}\begin{pmatrix}0&1&0\\1&0&1\\0&1&0\end{pmatrix},\quad
J_z=\operatorname{diag}(1,0,-1),\quad
H_B=\omega_b I_3\otimes a^\dagger a+gJ_z\otimes(a+a^\dagger),\quad
(\omega_b,g,\kappa)=(1.8,0.7,0.5)\Omega.
$$

The oscillator-vacuum damping convention and the local drive are

$$
\mathcal{L}_B(R)=-i[H_B,R]+\kappa\left[(I_3\otimes a)R(I_3\otimes a^\dagger)-\frac{1}{2}\{I_3\otimes a^\dagger a,R\}\right],\qquad
H_{\mathrm{sys}}(t)=[\Omega+\epsilon\cos(\omega_d t+\phi_0)]J_x,\quad \phi_0=\frac{\pi}{7},
$$

with the oscillator cutoff fixed at $N=4$. The quench from $|00\rangle\langle00|\otimes|0\rangle\langle0|$ has stationary and quench-weighted transient contributions under the full undriven generator $\mathcal{L}_0=\mathcal{L}_B-i[\Omega J_x\otimes I_N,\cdot]$. For each conjugate mode pair whose positive-frequency rate is $-\Gamma+i\nu$ with $0<\Gamma<0.8\Omega$ and $0<\nu<4\Omega$, its score is the largest two-qubit concurrence of its undamped one-pair reconstruction about the stationary reduced state on phases $\theta_k=2\pi k/128$, over reconstructed density matrices whose smallest eigenvalue is at least $-10^{-10}$. The pair with the largest score, with ties within $10^{-10}$ resolved by the smaller positive frequency, defines $(\Gamma_*,\nu_*)$ and the nine drives

$$
\epsilon/\Omega\in\{0.4,0.8,1.2\},\qquad
\omega_d\in\{\nu_*-\Gamma_*/2,\nu_*,\nu_*+\Gamma_*/2\}.
$$

For every drive, the periodic influence propagator consists of the uniform auxiliary map $e^{\Delta t\mathcal{L}_B}$ and symmetric local half steps, integrating the sinusoidal local Hamiltonian exactly on each half interval, with $\Delta t=2\pi/(M\omega_d)$ for both $M=24$ and $M=48$. For the attracting full auxiliary periodic state, $\overline{C}_M$ is the uniform mean of the two-qubit concurrence after tracing out the oscillator at the $M$ left interval boundaries in a period. Report the largest value of $(4\overline{C}_{48}-\overline{C}_{24})/3$ over the nine drives. In the reasoning, justify the modal weighting, the admissible mode-pair reconstruction, and the retained-memory periodic propagation, and report $(\Gamma_*,\nu_*)$, the winning drive, its two mean concurrences, and the final dimensionless scalar to absolute error at most $10^{-7}$.

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

01_build_collective_generators

Goal
----
Build the uniform bath generator and the undriven collective generator.

```python
def build_collective_generators(
    levels: int, omega: float, bath_frequency: float, coupling: float, decay: float
) -> "np.ndarray":
    r"""Build the uniform bath generator and the undriven collective generator.

    Parameters
    ----------
    levels : int
        Oscillator cutoff $N\ge2$.
    omega : float
        Finite positive static drive scale $\Omega$ in inverse time units.
    bath_frequency : float
        Finite positive oscillator frequency $\omega_b$ in inverse time units.
    coupling : float
        Finite nonnegative collective coupling $g$ in inverse time units.
    decay : float
        Finite positive oscillator damping rate $\kappa$ in inverse time units.

    Returns
    -------
    generators : np.ndarray
        Complex $(2,D,D)$ array in grouped Liouville order; slice zero is
        $\mathcal{L}_B$ and slice one is $\mathcal{L}_0=\mathcal{L}_B-i[\Omega J_x\otimes I_N,\cdot]$.

    Raises
    ------
    ValueError
        If levels is not an integer at least two, parameters are nonreal or
        nonfinite, a frequency or decay is nonpositive, or coupling is negative.
    """
    return
```

### Step 2

02_resolve_quench_modes

Goal
----
Resolve a normalized quench into stationary and weighted transient contributions.

```python
def resolve_quench_modes(
    generator: "np.ndarray", initial_state: "np.ndarray", levels: int
) -> "np.ndarray":
    r"""Resolve a normalized quench into stationary and weighted transient contributions.

    Parameters
    ----------
    generator : np.ndarray
        Finite complex $(D,D)$ undriven generator in grouped Liouville order,
        $D=9N^2$, with a simple attracting spectrum.
    initial_state : np.ndarray
        Finite complex $(D,)$ vector of a normalized positive joint density matrix.
    levels : int
        Oscillator cutoff $N\ge2$.

    Returns
    -------
    modes : np.ndarray
        Complex $(D,D+1)$ array. Each row contains its rate in column zero
        and the weighted joint density contribution in the remaining columns.
        The stationary row comes first with rate exactly zero; transient rows
        are sorted by imaginary rate rounded to ten decimals, then real rate.

    Raises
    ------
    ValueError
        If shapes or finiteness fail, the initial state fails density checks
        at 1e-8, the zero eigenvalue is not unique within 1e-9, another rate
        has real part at least -1e-10, eigenvalue separation is below 1e-8,
        or the right-eigenvector matrix has condition number above 1e10.
    """
    return
```

### Step 3

03_select_entangling_mode

Goal
----
Select a transient conjugate pair by its physically admissible entanglement score.

```python
def select_entangling_mode(
    modes: "np.ndarray",
    levels: int,
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
) -> "np.ndarray":
    r"""Select a transient conjugate pair by its physically admissible entanglement score.

    Parameters
    ----------
    modes : np.ndarray
        Weighted mode table $(D,D+1)$ returned by the preceding spectral step;
        row zero is the stationary contribution.
    levels : int
        Oscillator cutoff $N\ge2$, with $D=9N^2$.
    phase_count : int
        Number $K\ge4$ of phases $\theta_k=2\pi k/K$.
    decay_limit : float
        Finite positive upper bound for $\Gamma=-\operatorname{Re}\gamma$.
    frequency_limit : float
        Finite positive upper bound for $\nu=\operatorname{Im}\gamma$.

    Returns
    -------
    selection : np.ndarray
        Real $(3,)$ vector $(\Gamma_*,\nu_*,C_*)$ for the winning
        pair in the strict window $0<\Gamma<\Gamma_{\max}$ and $0<\nu<\nu_{\max}$.
        Eigenvalues at most $10^{-10}$ below zero are clipped only when evaluating
        concurrence. Each pair is scored by its maximum over admissible phases.
        Score ties within $10^{-10}$ use the smallest positive frequency across pairs.

    Raises
    ------
    ValueError
        If dimensions, finite values, integer bounds, or positive limits fail;
        the stationary state fails density checks; or no mode has an admissible
        phase in the specified strict spectral window.
    """
    return
```

### Step 4

04_integrate_local_half_steps

Goal
----
Integrate the sinusoidal collective drive on both halves of every interval.

```python
def integrate_local_half_steps(
    omega: float, amplitude: float, drive_frequency: float, intervals: int, phase: float
) -> "np.ndarray":
    r"""Integrate the sinusoidal collective drive on both halves of every interval.

    Parameters
    ----------
    omega : float
        Finite positive static scale $\Omega$.
    amplitude : float
        Finite nonnegative drive amplitude $\epsilon$.
    drive_frequency : float
        Finite positive angular frequency $\omega_d$.
    intervals : int
        Number $M\ge4$ of intervals per period.
    phase : float
        Finite initial phase $\phi_0$ in radians.

    Returns
    -------
    half_steps : np.ndarray
        Complex $(M,2,9,9)$ local channel array; index one selects the
        early half at zero and the late half at one, in chronological order.

    Raises
    ------
    ValueError
        If a scalar is nonreal or nonfinite, omega or drive_frequency is
        nonpositive, amplitude is negative, or intervals is not an integer at least four.
    """
    return
```

### Step 5

05_contract_auxiliary_cycle

Goal
----
Dress the uniform auxiliary map and contract the chronological cycle.

```python
def contract_auxiliary_cycle(
    bath_generator: "np.ndarray", half_steps: "np.ndarray", dt: float, levels: int
) -> "np.ndarray":
    r"""Dress the uniform auxiliary map and contract the chronological cycle.

    Parameters
    ----------
    bath_generator : np.ndarray
        Finite complex $(D,D)$ uniform generator $\mathcal{L}_B$, $D=9N^2$.
    half_steps : np.ndarray
        Finite complex $(M,2,9,9)$ early/late local channels, $M\ge4$.
    dt : float
        Finite positive interval length used for the supplied half steps.
    levels : int
        Auxiliary cutoff $N\ge2$.

    Returns
    -------
    cycle_stack : np.ndarray
        Complex $(M+1,D,D)$ array; the first $M$ slices are the chronological
        $Q_j$ and the final slice is their cycle product $F$.

    Raises
    ------
    ValueError
        If integer bounds, shapes or finite-value checks fail, or dt is not positive.
    """
    return
```

### Step 6

06_recover_periodic_qubit_states

Goal
----
Recover the attracting auxiliary cycle and its reduced qubit micromotion.

```python
def recover_periodic_qubit_states(
    cycle_stack: "np.ndarray", levels: int
) -> "np.ndarray":
    r"""Recover the attracting auxiliary cycle and its reduced qubit micromotion.

    Parameters
    ----------
    cycle_stack : np.ndarray
        Finite complex $(M+1,D,D)$ array with $M\ge4$, $D=9N^2$;
        the first $M$ slices are physical channels and the last is their product.
    levels : int
        Oscillator cutoff $N\ge2$.

    Returns
    -------
    qubit_states : np.ndarray
        Complex $(M,4,4)$ normalized positive two-qubit states, in the
        order $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$, at left boundaries.

    Raises
    ------
    ValueError
        If shape, finite values or integer bounds fail; the normalized fixed-point
        system has condition number above 1e12; a reduced state violates density
        checks at 1e-8; or the propagated joint cycle fails closure by more than 1e-8.
    """
    return
```

### Step 7

07_evaluate_stabilization_grid

Goal
----
Evaluate the period-averaged entanglement estimator for each candidate drive.

```python
def evaluate_stabilization_grid(
    bath_generator: "np.ndarray",
    omega: float,
    levels: int,
    amplitudes: "np.ndarray",
    frequencies: "np.ndarray",
    intervals: int,
    phase: float,
) -> "np.ndarray":
    r"""Evaluate the period-averaged entanglement estimator for each candidate drive.

    Parameters
    ----------
    bath_generator : np.ndarray
        Finite complex $(D,D)$ uniform auxiliary generator, $D=9N^2$.
    omega : float
        Finite positive static scale $\Omega$.
    levels : int
        Oscillator cutoff $N\ge2$.
    amplitudes : np.ndarray
        Nonempty finite real one-dimensional array of nonnegative amplitudes.
    frequencies : np.ndarray
        Nonempty finite real one-dimensional array of positive angular frequencies.
    intervals : int
        Coarse resolution $M\ge4$; fine resolution is $2M$.
    phase : float
        Finite initial drive phase in radians.

    Returns
    -------
    estimates : np.ndarray
        Real array of shape $(n_\epsilon,n_\omega)$ containing $E_M$,
        preserving the supplied amplitude-row and frequency-column order.

    Raises
    ------
    ValueError
        If grid arrays, dimensions, finite values, positive frequencies, or
        integer bounds fail, or an upstream fixed-point/density/closure check
        fails under its documented tolerances.
    """
    return
```

### Step 8

08_compute_stabilized_entanglement

Goal
----
Select the transient resonance and compute the strongest driven entanglement estimator.

```python
def compute_stabilized_entanglement(
    levels: int,
    omega: float,
    bath_frequency: float,
    coupling: float,
    decay: float,
    amplitudes: "np.ndarray",
    detunings: "np.ndarray",
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
    intervals: int,
    phase: float,
) -> float:
    r"""Select the transient resonance and compute the strongest driven entanglement estimator.

    Parameters
    ----------
    levels : int
        Oscillator cutoff $N\ge2$.
    omega : float
        Finite positive static scale $\Omega$.
    bath_frequency : float
        Finite positive oscillator frequency $\omega_b$.
    coupling : float
        Finite nonnegative collective coupling $g$.
    decay : float
        Finite positive damping rate $\kappa$.
    amplitudes : np.ndarray
        Nonempty finite real vector of nonnegative drive amplitudes.
    detunings : np.ndarray
        Nonempty finite real vector $d_k$, whose derived frequencies must be positive.
    phase_count : int
        Number of spectral-scoring phases, at least four.
    decay_limit : float
        Finite positive strict decay-window upper bound.
    frequency_limit : float
        Finite positive strict frequency-window upper bound.
    intervals : int
        Coarse time resolution $M\ge4$; fine resolution is $2M$.
    phase : float
        Finite initial drive phase in radians.

    Returns
    -------
    entanglement : float
        Largest finite, dimensionless two-resolution concurrence estimator over
        the mode-selected drive grid, with all preceding conventions retained.

    Raises
    ------
    ValueError
        If an upstream parameter violates its declared shape, finiteness,
        integer or physical bound; the quench spectrum is not simple and
        attracting; no admissible mode is found; a derived drive frequency
        is nonpositive; or an upstream numerical physical-state check fails.
    """
    return
```
