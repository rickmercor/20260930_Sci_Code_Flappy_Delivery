# Sensitivity of a spacetime recovery

## Background

# Repeated syndrome extraction in quantum memories

A quantum memory uses stabilizer measurements to infer errors without reading out the encoded state. Faulty measurements make a single syndrome insufficient: successive records mix newly accumulated data errors with errors in syndrome extraction. The temporal record is therefore central to assessing whether encoded information can be recovered.

The encoded information is nonlocal: stabilizer checks can agree for different logical states. Error correction must distinguish equivalence classes of physical faults using measurements that reveal only part of the underlying error history. Decoding methods are studied for their ability to preserve logical information across different physical noise and measurement regimes.

## Problem

Determine the curvature of the conditional failure log odds for a quantum-memory recovery inferred from repeated faulty syndrome measurements on a periodic $2\times2$ torus, using exact finite inference without sampling. Coordinates are $x,y\in\{0,1\}$ modulo two, horizontal and vertical edges have indices $2(x+2y)$ and $2(x+2y)+1$, $X$ stabilizers are vertex stars and $Z$ stabilizers are plaquette boundaries, and only the three stabilizers of each type with $x+2y=0,1,2$ are measured; use these ordered independent generators and logical supports $L_X^0=\{h(0,y)\}$, $L_X^1=\{v(x,0)\}$, $L_Z^0=\{h(x,0)\}$ and $L_Z^1=\{v(0,y)\}$, with free coordinates ranging over the torus.

There are five data-error intervals $t=0,\ldots,4$, initially zero syndrome, independent per-qubit probabilities $P(I)=1-p$ and $P(X)=P(Y)=P(Z)=p/3$, independent measurement-bit error probability $p_m$ after intervals zero through three, and an exact final measurement after interval four, with inputs

$$
p_0=0.08,\qquad p_m=0.06,\qquad
C=\begin{pmatrix}930&70\\30&970\end{pmatrix},
$$

$$
d_X=\begin{pmatrix}1&0&1\\0&1&0\\1&1&0\\0&0&1\\1&0&0\end{pmatrix},\qquad
d_Z=\begin{pmatrix}0&1&1\\1&0&0\\0&1&0\\1&1&1\\0&0&1\end{pmatrix}.
$$

Here $d_X$ contains successive differences of the measured $Z$ checks detecting $X$ components, $d_Z$ contains differences of the measured $X$ checks detecting $Z$ components, rows increase in time, and $C_{ij}$ counts samples with estimated component $i$ and true component $j$, using this same fixed table in both update directions. For each component with check matrix $H$, choose the lexicographically first independent pivot columns of $H$ from left-to-right binary elimination, define the pure-error section $D$ by $DH^{\mathsf T}=I_3$ with all nonpivot columns zero, and define cumulative logical coordinates by the unique decomposition of the XOR of its five physical data-error vectors in the ordered row basis $(D,L^0,L^1,S_0,S_1,S_2)$; the final perfect syndrome fixes the pure-error coordinate, while the logical coordinate remains a separately conditioned sector.

At $p_0$, apply serial $X$-then-$Z$ minimization of the complete spacetime component energy with the conditional-reliability soft update, initially using the unconditioned $X$ marginal, with the stated exact final measurement, and stop when both inferred data-error histories repeat in consecutive complete sweeps or after four sweeps. For each channel solve at inverse temperature one, choose the smallest cumulative logical identifier $2\ell_0+\ell_1$ admitting an energy within $10^{-12}$ of the global minimum, then the lexicographically smallest physical data history in time-major, increasing-edge order within that same global energy window; measurement errors are determined by that history and the fixed detector record.

Freeze the resulting joint logical recovery $\widehat\ell$ and let $S(p)$ be its exact conditional success probability, obtained by summing all physical data and measurement histories in that sector and dividing by the total probability of the fixed detector records, with $C$, $p_m$ and $\widehat\ell$ held fixed as $p$ varies. The single requested dimensionless scalar is

$$
\left.\frac{\mathrm d^2}{\mathrm dp^2}\log\frac{1-S(p)}{S(p)}\right|_{p=p_0},
$$

to absolute accuracy $10^{-6}$, using natural logarithms; in the reasoning justify which spacetime couplings are reweighted and how the temporal boundary is treated under the stated noise model, distinguish the fixed logical sector from a minimum-energy representative, and report the sweep count, terminal channel identifiers and $S(p_0),S'(p_0),S''(p_0)$ as numerical checkpoints.

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

01_build_binary_quotient

Goal
----
Construct a canonical binary quotient that separates syndrome and logical coordinates.

```python
def build_binary_quotient(
    checks: "np.ndarray", stabilizers: "np.ndarray", logicals: "np.ndarray"
) -> "np.ndarray":
    r"""Construct a canonical binary quotient that separates syndrome and logical
    coordinates.

    Parameters
    ----------
    checks : np.ndarray
        Binary array $H$ of shape $(m,n)$, $1\le m\le3$ and $m+2\le n\le10$.
    stabilizers : np.ndarray
        Binary array $S$ of shape $(n-m-2,n)$.
    logicals : np.ndarray
        Binary array $L$ of shape $(2,n)$, ordered logical generators.

    Returns
    -------
    result : np.ndarray
        Integer array of shape $(2,n,n)$ containing $B$ and $B^{-1}$.

    Raises
    ------
    ValueError
        If arrays are nonbinary, dimensions disagree, checks are dependent,
        a supplied generator changes syndrome, or the complete basis is singular.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 2

02_build_detector_constraints

Goal
----
Assemble temporal detector constraints and cumulative logical observables.

```python
def build_detector_constraints(
    quotient: "np.ndarray", check_count: int, rounds: int
) -> "np.ndarray":
    r"""Assemble temporal detector constraints and cumulative logical observables.

    Parameters
    ----------
    quotient : np.ndarray
        Binary inverse pair of shape $(2,n,n)$ in syndrome/logical/stabilizer order.
    check_count : int
        Number $m$ of independent checks, from one to three.
    rounds : int
        Number $T$ of data intervals, from one to eight.

    Returns
    -------
    result : np.ndarray
        Binary matrix of shape $(Tm+2,Tn+(T-1)m)$; detector rows then logical rows.

    Raises
    ------
    ValueError
        If the quotient is not an inverse pair, its dimensions are invalid, or
        check_count or rounds is outside its documented integer range.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 3

03_build_conditional_branches

Goal
----
Construct energy-labelled physical transitions under conditional reliability.

```python
def build_conditional_branches(
    quotient: "np.ndarray",
    check_count: int,
    opposite: "np.ndarray",
    p: float,
    counts: "np.ndarray",
    initial: bool,
) -> "np.ndarray":
    r"""Construct energy-labelled physical transitions under conditional reliability.

    Parameters
    ----------
    quotient : np.ndarray
        Binary inverse pair of shape $(2,n,n)$.
    check_count : int
        Independent check count $m\in\{1,2,3\}$.
    opposite : np.ndarray
        Binary opposite-component data history of shape $(T,n)$, $1\le T\le8$.
    p : float
        Total depolarizing rate in $(0,3/4]$.
    counts : np.ndarray
        Nonnegative finite $(2,2)$ table with positive row sums and $a,b\ge1/2$.
    initial : bool
        Whether to use the unconditioned component marginal.

    Returns
    -------
    result : np.ndarray
        Float array of shape $(T,2^n,n+2)$ with rows in physical-word order.
        Columns contain quotient jump identifier, data energy, then $n$ physical bits.

    Raises
    ------
    ValueError
        If the inverse pair, dimensions, binary history, rate, counts or Boolean
        initialization flag violates the stated contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 4

04_solve_spacetime_sectors

Goal
----
Find the globally admissible logical sector and reconstruct its least history.

```python
def solve_spacetime_sectors(
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    branches: "np.ndarray",
    pm: float,
    tie_tol: float,
) -> "np.ndarray":
    r"""Find the globally admissible logical sector and reconstruct its least history.

    Parameters
    ----------
    constraints : np.ndarray
        Binary temporal matrix of shape $(Tm+2,Tn+(T-1)m)$ with repeated spatial blocks.
    detectors : np.ndarray
        Binary detector record of shape $(T,m)$, $1\le T\le8$, $1\le m\le3$.
    branches : np.ndarray
        Finite float array $(T,2^n,n+2)$ of exact ordered words and jump labels;
        its energy column may contain any finite values, with $m+2\le n\le10$.
    pm : float
        Independent measurement error rate in $(0,1/2]$.
    tie_tol : float
        Nonnegative finite global energy window $\delta$.

    Returns
    -------
    result : np.ndarray
        Float vector of length $6+Tn+(T-1)m$: chosen logical identifier, chosen
        history energy, four exact sector minima, then all data bits and measurement
        bits.

    Raises
    ------
    ValueError
        If detector/constraint dimensions or repeated blocks are inconsistent,
        spatial syndrome/logical rows are dependent, branch words or labels are
        invalid, or pm or tie_tol lies outside its domain.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 5

05_iterate_spacetime_recovery

Goal
----
Propagate reliability between two complete spacetime recovery problems.

```python
def iterate_spacetime_recovery(
    quotients: "np.ndarray",
    constraints: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> "np.ndarray":
    r"""Propagate reliability between two complete spacetime recovery problems.

    Parameters
    ----------
    quotients : np.ndarray
        Binary array $(2,2,n,n)$, channels $X,Z$, each basis followed by its inverse.
    constraints : np.ndarray
        Binary array $(2,Tm+2,Tn+(T-1)m)$ matching the quotients.
    detectors : np.ndarray
        Binary array $(2,T,m)$, $X$-detected and $Z$-detected records.
    p : float
        Total data depolarizing rate in $(0,3/4]$.
    pm : float
        Measurement flip probability in $(0,1/2]$.
    counts : np.ndarray
        Fixed $(2,2)$ estimate/true table with valid nonnegative entries and
        reliabilities.
    max_sweeps : int
        Positive completed-sweep budget, at most eight.
    tie_tol : float
        Finite nonnegative whole-history energy window.

    Returns
    -------
    result : np.ndarray
        Float array $(N,2,6+Tn+(T-1)m)$ containing every completed sweep,
        with channel records in the single-solve format and repeated terminal sweep
        included.

    Raises
    ------
    ValueError
        If channel dimensions, quotient/constraint identities, binary records,
        rates, counts, sweep budget or energy tolerance violates its contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 6

06_build_joint_transition_jets

Goal
----
Form the correlated Pauli quotient kernel and its first two rate derivatives.

```python
def build_joint_transition_jets(
    quotients: "np.ndarray", check_count: int, p: float
) -> "np.ndarray":
    r"""Form the correlated Pauli quotient kernel and its first two rate derivatives.

    Parameters
    ----------
    quotients : np.ndarray
        Binary inverse pairs of shape $(2,2,n,n)$ for the two physical components.
    check_count : int
        Independent check count $m$ from one to three.
    p : float
        Total depolarizing probability in $(0,3/4]$.

    Returns
    -------
    result : np.ndarray
        Float array $(3,2^{2(m+2)})$ with kernel value, first derivative and
        second derivative with respect to the same scalar $p$.

    Raises
    ------
    ValueError
        If the two inverse pairs, their common dimensions, check count or rate
        violates the stated contract.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 7

07_propagate_conditioned_jets

Goal
----
Condition joint logical-sector probability derivatives on the full temporal record.

```python
def propagate_conditioned_jets(
    constraints: "np.ndarray", detectors: "np.ndarray", kernel: "np.ndarray", pm: float
) -> "np.ndarray":
    r"""Condition joint logical-sector probability derivatives on the full temporal
    record.

    Parameters
    ----------
    constraints : np.ndarray
        Two matching binary temporal matrices of shape $(2,Tm+2,Tn+(T-1)m)$.
    detectors : np.ndarray
        Fixed binary records of shape $(2,T,m)$, with one to eight intervals.
    kernel : np.ndarray
        Finite joint derivative array $(3,2^{2(m+2)})$; row zero is nonnegative
        and row sums equal $(1,0,0)$ to absolute tolerance $10^{-9}$.
    pm : float
        Fixed measurement flip probability in $(0,1/2]$.

    Returns
    -------
    result : np.ndarray
        Float array $(3,4,4)$ of conditional sector probabilities, first derivatives
        and second derivatives, with logical axes ordered $X,Z$.

    Raises
    ------
    ValueError
        If dimensions, temporal blocks, binary records, kernel normalization or
        measurement rate is invalid, or the detector record has zero evidence.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```

### Step 8

08_compute_log_odds_curvature

Goal
----
Evaluate the rate curvature of the frozen recovery’s conditional failure odds.

```python
def compute_log_odds_curvature(
    checks: "np.ndarray",
    stabilizers: "np.ndarray",
    logicals: "np.ndarray",
    detectors: "np.ndarray",
    p: float,
    pm: float,
    counts: "np.ndarray",
    max_sweeps: int,
    tie_tol: float,
) -> float:
    r"""Evaluate the rate curvature of the frozen recovery’s conditional failure odds.

    Parameters
    ----------
    checks : np.ndarray
        Binary $(2,m,n)$ checks, ordered by the error component they detect.
    stabilizers : np.ndarray
        Binary $(2,n-m-2,n)$ component stabilizer generators.
    logicals : np.ndarray
        Binary $(2,2,n)$ ordered component logical generators.
    detectors : np.ndarray
        Binary $(2,T,m)$ fixed detection-event records.
    p : float
        Nominal total depolarizing probability in $(0,3/4]$.
    pm : float
        Fixed measurement probability in $(0,1/2]$.
    counts : np.ndarray
        Fixed nonnegative $(2,2)$ estimated/true table with positive row sums
        and both fitted reliabilities at least one half.
    max_sweeps : int
        Completed-sweep budget from one to eight.
    tie_tol : float
        Nonnegative finite whole-history degeneracy window.

    Returns
    -------
    result : float
        One finite dimensionless second derivative of conditional failure log odds.

    Raises
    ------
    ValueError
        If an upstream generator, record, dimension, probability, reliability or
        iteration contract fails, or the selected success mass is outside $(0,1)$.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result
```
