# Physics-Optics-26

## Background

When only a few emitters (atoms, quantum dots or molecules) supply the gain of a single cavity mode, as in plasmonic nanocavity lasers, steady-state superradiant lasers and quantum-dot nanolasers, a semiclassical treatment of the gain medium breaks down: photon statistics, atom-field correlations and correlations between different emitters all matter when only a few quanta are present. Such lasers are usually modelled with the second-order cumulant (cluster) expansion, which neglects third-order cumulants and closes a small nonlinear set of moment equations; for a few strongly coupled emitters its reliability can only be judged against an exact solution.

The exact benchmark expands the cavity field in the damping eigenoperators of the thermal cavity Liouvillian (the Briegel-Englert damping basis) and the emitters in a permutation-symmetric operator basis built from single-emitter damping eigenoperators, whose size grows only polynomially with $N$. In this product basis the stationary equations of the sector with zero total coherence couple each radial field level only to its nearest neighbours, so the stationary state follows from a downward matrix continued fraction. The termination of that continued fraction at large radial index, which would generally require a quadratic matrix equation, reduces here to a linear matrix equation with a unique solution whenever the cavity is lossy.

## Problem

In a few-emitter laser, a handful of incoherently pumped two-level emitters share one lossy cavity mode. The cavity can then induce a mutual coherence between distinct emitters, the pair coherence $C_2=\langle\tau_+^{(1)}\tau_-^{(2)}\rangle$, whose sign tells whether a pair of emitters radiates in phase (superradiant, positive) or out of phase (subradiant, negative). The standard workhorse for such systems is the second-order cumulant closure of the moment hierarchy, and nothing inside the closure itself shows whether it can be trusted for a handful of strongly coupled emitters. A recent exact finite-$N$ construction for the pumped Tavis-Cummings laser provides that benchmark. Your task is to use it to quantify, at one specific operating point, how far the closure's pair coherence is from the exact stationary value.

Here is the exact setup to use.

Model: $N$ identical two-level emitters, with $\tau_+=|e\rangle\langle g|$, $\tau_-=\tau_+^\dagger$, $\sigma_z=|e\rangle\langle e|-|g\rangle\langle g|$, $S_\pm=\sum_j\tau_\pm^{(j)}$ and $S_z=\sum_j\sigma_z^{(j)}$, are coupled symmetrically to one cavity mode $a$, with photon-number operator $\hat n=a^\dagger a$, Hamiltonian

$$H=\hbar\omega\,\hat n+\frac{\hbar\Omega}{2}S_z-\frac{\hbar g}{2}\left(a^\dagger S_-+aS_+\right),$$

and detuning $\Delta=\Omega-\omega$. With $\mathcal{D}[L]\rho=L\rho L^\dagger-\frac{1}{2}\{L^\dagger L,\rho\}$, the state obeys

$$\dot\rho=-\frac{i}{\hbar}[H,\rho]+A(1+\nu)\,\mathcal{D}[a]\rho+A\nu\,\mathcal{D}[a^\dagger]\rho+\sum_{j=1}^{N}\left\{B(1-s)\,\mathcal{D}[\tau_-^{(j)}]\rho+Bs\,\mathcal{D}[\tau_+^{(j)}]\rho+\frac{C-B/2}{2}\,\mathcal{D}[\sigma_z^{(j)}]\rho\right\}.$$

Here $A$ is the cavity energy-decay rate, $\nu$ the thermal occupation of the cavity reservoir, $B$ the longitudinal atomic rate, $s$ the pump parameter and $C\ge B/2$ the transverse atomic rate. The coupling $g$ is exactly the one written in $H$ above (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).

- Parameters: $N=4$, $A=1.0$, $B=0.8$, $C=0.5$, $g=1.25$, $s=0.88$, $\Delta=0.3$, $\nu=0.05$ (all rates in the same units).
- "Exact" means the true stationary state of this master equation with the full cavity Hilbert space: any truncation used must be converged so that it does not affect the reported digits, and no weak-pump, mean-field, adiabatic-elimination or cumulant approximation may enter the exact part. The source paper provides an exact construction for this; consult the paper for it. Do not assume that a generic quadratic-matrix termination of a continued fraction, or a zero terminal condition, is what the paper uses.
- "Second-order closure" means the second-order cumulant closure exactly as benchmarked in the source paper (which third-order moments are factorized, which rates appear, and how its physical stationary solution is selected). Consult the paper for the precise closure; do not substitute a different truncation scheme or select a stationary root by your own criterion.

Report $\Delta_C=C_2^{(4)}[\mathrm{exact}]-C[\mathrm{closure}]$, the exact stationary pair coherence minus the second-order-closure pair coherence at this configuration. The answer is graded to an absolute tolerance of $10^{-8}$. In your reasoning, report individually: the exact stationary mean photon number $\langle\hat n\rangle$, the exact $g^{(2)}(0)=\langle a^{\dagger2}a^2\rangle/\langle\hat n\rangle^2$, the exact single-emitter excited-state population $p_e$, the exact pair coherence $C_2^{(4)}$, the closure mean photon number, closure excited-state population and closure pair coherence, whether the exact and closure pair coherences have the same sign, the method used for the exact stationary state, the first two radial damping-basis coefficients $c_1$ and $c_2$ of the exact stationary state as defined in the paper's appendix on observable functionals, and the smallest singular value of the bulk-slope matrix $M^{(1)}$ that enters the paper's asymptotic closure of its continued fraction, evaluated in the paper's own basis normalization for this configuration.

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

field_multiplication_coefficients

Goal
----
Compute the multiplication coefficients of the lossy thermal cavity's damping eigenoperator of radial index $n$ and coherence order $k$: the coefficients of the eigenoperators produced when it is multiplied from the left or from the right by $a$ or $a^\dagger$.

```python
import numpy as np


def field_multiplication_coefficients(n: int, k: int, nu: float) -> "np.ndarray":
    r"""Briegel-Englert multiplication coefficients of the field damping eigenoperator $\rho_n^{(k)}$.

    Args:
        n (int): radial damping index, $n\ge0$.
        k (int): coherence order $k$ (any integer: positive, zero or negative).
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: if $n$ is negative or not an integer, if $k$ is not an integer, or if $\nu<0$.

    Expected return:
        np.ndarray of shape $(2,2,3)$, $f[\mathrm{side},\mu,d]$: $\mathrm{side}=0$ is left
        multiplication and $\mathrm{side}=1$ right multiplication; $\mu=0$ is the annihilation
        operator $a$ (target coherence order $k-1$) and $\mu=1$ the creation operator $a^\dagger$
        (target coherence order $k+1$); $d=0,1,2$ for target radial index $n-1$, $n$, $n+1$.
        Entries that do not occur are zero, every $d=0$ entry vanishes at $n=0$, and all entries
        are non-negative.
    """
    return None
```

### Step 2

atomic_multiplication_matrices

Goal
----
Compute the matrices of left and right multiplication by the collective ladders $S_+$ and $S_-$ on the paper's permutation-invariant atomic damping basis $R_m$ for $N$ emitters at pump parameter $s$.

```python
import numpy as np


def atomic_multiplication_matrices(N: int, s: float) -> "np.ndarray":
    r"""Collective-ladder multiplication matrices on the permutation-invariant atomic damping basis.

    Args:
        N (int): number of emitters, $N\ge1$.
        s (float): pump parameter, $0<s<1$.

    Raises:
        ValueError: if $N$ is not a positive integer or $s$ is not strictly inside $(0,1)$.

    Expected return:
        np.ndarray of shape $(4,D,D)$, $D$ the basis dimension. Slice $t$ is the matrix of
        $t=0$: $S_+$ from the left, $t=1$: $S_-$ from the left, $t=2$: $S_+$ from the right,
        $t=3$: $S_-$ from the right; element $[t,\alpha,\beta]$ is the coefficient of $R_\alpha$
        in the product formed from $R_\beta$ (columns = source, rows = target). Basis ordering:
        compositions $(m_0,m_z,m_+,m_-)$ of $N$ with $m_0$ descending from $N$ to $0$, then $m_z$
        descending, then $m_+$ descending, so index $0$ is $(N,0,0,0)$. Each column has at most
        four non-zero entries and every non-zero entry changes the coherence charge $m_+-m_-$ by
        exactly one unit.
    """
    return None
```

### Step 3

radial_blocks

Goal
----
Assemble the three $D$-by-$D$ blocks that couple radial level $n$ to levels $n-1$, $n$ and $n+1$ in the stationary sector of vanishing total coherence $K=k+(m_+-m_-)=0$, for the full pumped Tavis-Cummings Liouvillian of the model.

```python
import numpy as np


def radial_blocks(n: int, N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Block-Jacobi radial blocks $[M_n,G_n,F_n]$ of the stationary ($K=0$) sector.

    Args:
        n (int): radial index, $n\ge0$.
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: if $n<0$, $A\le0$, $B\le0$, $C<B/2$, or the Step 2 domain is violated.

    Expected return:
        np.ndarray (complex) of shape $(3,D,D)$ stacking $[M_n,G_n,F_n]$: $M_n$ is the
        within-level block, $G_n$ carries amplitude from level $n+1$ down to $n$, and $F_n$
        carries amplitude from level $n-1$ up to $n$ and vanishes identically at $n=0$. Basis
        ordering as in Step 2; the field coherence order attached to basis element $\beta$ is
        minus its atomic coherence charge.
    """
    return None
```

### Step 4

asymptotic_transfer_matrix

Goal
----
Compute the large-radial-index limit $R_\infty$ of the stationary transfer matrices $X_{n+1}=R_n X_n$, which terminates the downward matrix continued fraction.

```python
import numpy as np


def asymptotic_transfer_matrix(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Large-radial-index transfer matrix $R_\infty$ that terminates the continued fraction.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: under the same conditions as Step 3.

    Expected return:
        np.ndarray (complex) of shape $(D,D)$: the asymptotic transfer matrix, independent of any
        radial truncation, bounded, and well defined for every $A>0$. Basis ordering as in
        Step 2.
    """
    return None
```

### Step 5

stationary_radial_coefficients

Goal
----
Obtain the stationary state of the $K=0$ sector as a downward matrix continued fraction started from the terminal transfer matrix of Step 4 at radial index $n_{\max}$, and return its damping-basis coefficients at every radial level.

```python
import numpy as np


def stationary_radial_coefficients(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    r"""Damping-basis coefficients $X_0,\ldots,X_{n_{\max}}$ of the unique stationary state.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.
        n_max (int): radial index $n_{\max}\ge2$ at which the terminal transfer matrix is imposed.

    Raises:
        ValueError: if $n_{\max}<2$ or not an integer, or under the conditions of Step 3.

    Expected return:
        np.ndarray (complex) of shape $(n_{\max}+1,D)$; row $n$ holds $X_n$ in the Step 2
        ordering. Row $0$ has its index-$0$ component equal to exactly $1$. For a converged
        $n_{\max}$ the rows decay rapidly with $n$ in the few-quanta regime and are insensitive
        to $n_{\max}$.
    """
    return None
```

### Step 6

exact_stationary_observables

Goal
----
Extract the exact stationary mean photon number, zero-delay second-order coherence, single-emitter excited-state population and inter-emitter pair coherence from the damping-basis coefficients of Step 5.

```python
import numpy as np


def exact_stationary_observables(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    r"""Exact stationary photon statistics and one- and two-atom observables.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.
        n_max (int): radial truncation index, $n_{\max}\ge2$.

    Raises:
        ValueError: if $N<2$, or under the conditions of Steps 3 and 5.

    Expected return:
        np.ndarray of shape $(4,)$: $[\langle\hat n\rangle,g^{(2)}(0),p_e,C_2^{(N)}]$ with
        $\langle\hat n\rangle\ge0$ the mean photon number, $g^{(2)}(0)$ the zero-delay
        second-order coherence, $p_e\in[0,1]$ the excited-state population of one emitter and
        $C_2^{(N)}=\langle\tau_+^{(1)}\tau_-^{(2)}\rangle$ the pair coherence of two distinct
        emitters (real by permutation symmetry; either sign is possible).
    """
    return None
```

### Step 7

second_order_closure_stationary

Goal
----
Compute the stationary mean photon number, excited-state population, atom-field coherence $Z_1$ and pair coherence $C$ predicted by the second-order cumulant closure of the moment hierarchy, as benchmarked in the paper.

```python
import numpy as np


def second_order_closure_stationary(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    r"""Stationary solution of the paper's second-order cumulant closure.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.

    Raises:
        ValueError: if $N<2$, $A\le0$, $B\le0$, $C<B/2$, or $s$ outside $(0,1)$.

    Expected return:
        np.ndarray of shape $(5,)$:
        $[\langle\hat n\rangle,p_e,\operatorname{Re}Z_1,\operatorname{Im}Z_1,C]$ with
        $Z_1=\langle a^\dagger\tau_-^{(j)}\rangle$ and $C=\langle\tau_+^{(i)}\tau_-^{(j)}\rangle$
        ($i\ne j$), the physically selected stationary solution (non-negative photon number,
        population in $[0,1]$).
    """
    return None
```

### Step 8

pair_coherence_closure_error

Goal
----
Chain the earlier steps into the full pipeline: build the exact stationary state with the damping-basis continued fraction (Steps 1-5), extract its pair coherence (Step 6), solve the second-order closure (Step 7), and return the closure error of the pair coherence. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

```python
import numpy as np


def pair_coherence_closure_error(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> float:
    r"""Exact minus second-order-closure pair coherence.

    Args:
        N (int): number of identical two-level emitters.
        A (float): cavity energy-decay rate, $A>0$.
        B (float): longitudinal atomic rate, $B>0$ (pump rate $Bs$, decay rate $B(1-s)$).
        C (float): transverse atomic rate, $C\ge B/2$.
        g (float): Tavis-Cummings coupling exactly as in the model Hamiltonian
            (not the Briegel-Englert $g_{\mathrm{BE}}$ convention).
        s (float): pump parameter, $0<s<1$.
        Delta (float): detuning $\Delta=\Omega-\omega$.
        nu (float): thermal occupation $\nu\ge0$ of the cavity reservoir.
        n_max (int): radial truncation index $n_{\max}\ge2$ for the exact solution.

    Raises:
        ValueError: under the conditions of Steps 6 and 7.

    Expected return:
        float: $C_2^{(N)}[\mathrm{exact}]-C[\mathrm{closure}]$; positive when the closure
        underestimates the pair coherence, including when it predicts the wrong sign.
    """
    return None
```
