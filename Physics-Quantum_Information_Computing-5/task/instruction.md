# Physics-Quantum_Information_Computing-5

## Background

Sparse quantum-state simulation reduces memory by retaining a limited set of amplitudes, but the effect of discarding amplitudes depends on the coordinates used to represent the state. Local changes of basis can concentrate probability while preserving a compact description of the physical coordinate frame. Entangling local transformations can also provide temporary representations in which a different sparse projection is useful.

Response calculations ask how such approximations change under coherent perturbations of the physical evolution. They must account for the movement of the representation as well as the physical state. Smooth local responses can be studied between discrete changes of retained support or optimization decisions, providing information beyond an unperturbed fidelity calculation.

## Problem

Two independent coherent control perturbations can change both the retained amplitudes and the local coordinates of a sparsely represented quantum state. Compute the mixed response $\partial_s^2\partial_u^2 F(0,0)$ of the physical-state fidelity for the experiment below, using the 2026 reduced-density eigenbasis procedure with guarded single-qubit snapshot passes and its transient two-qubit refinement.

The exact and approximate evolutions start from the same normalized five-qubit state $\psi_x=z_x/(\sum_y|z_y|^2)^{1/2}$, with identity local frames, where

$$
z_x=\sin(0.37(x+1))+0.17\cos(0.11(x+1)^2)
+i[\cos(0.23(x+2))-0.13\sin(0.07(x+1)^2)],\qquad 0\le x<32.
$$

For dimensionless $s,u\in\mathbb R$, the five angles of gate $t$ are $A_{t,r}+sB_{t,r}+uC_{t,r}$, where $t=1,\ldots,12$, $r=1,\ldots,5$, $B_{t,r}=0.17\sin[t(r+1)]+0.09\cos[(t+2)r]$, $C_{t,r}=0.13\cos[(t+1)(r+2)]-0.07\sin[(t+3)r]$, and $A$ and the ordered pairs are given below, with all angles in radians and the rightmost factor acting first:

$$
G_t(s,u)=e^{-ic_t(s,u)(Z\otimes Z)/2}
[R_Y(a_t(s,u))\otimes R_X(b_t(s,u))]
[R_Z(d_t(s,u))\otimes R_Y(e_t(s,u))],\qquad
R_P(\theta)=\cos(\theta/2)I-i\sin(\theta/2)P,
$$

where $X|b\rangle=|1-b\rangle$, $Z|b\rangle=(-1)^b|b\rangle$ and $Y=iXZ$.

| t (1-based) | q1 | q2 | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 0.73 | -0.42 | 0.61 | 0.19 | -0.37 |
| 2 | 2 | 3 | -0.58 | 0.91 | -0.47 | 0.36 | 0.52 |
| 3 | 4 | 0 | 1.07 | 0.33 | 0.82 | -0.29 | 0.64 |
| 4 | 1 | 2 | 0.41 | -1.13 | 0.57 | 0.83 | -0.46 |
| 5 | 3 | 4 | -0.87 | 0.62 | 1.19 | -0.51 | 0.28 |
| 6 | 2 | 0 | 0.94 | -0.76 | -0.63 | 0.47 | 1.02 |
| 7 | 4 | 1 | -0.31 | 1.21 | 0.74 | -0.68 | 0.39 |
| 8 | 3 | 2 | 0.66 | 0.48 | -1.09 | 0.92 | -0.57 |
| 9 | 0 | 3 | -1.17 | 0.59 | 0.43 | -0.34 | 0.88 |
| 10 | 1 | 4 | 0.53 | -0.97 | 0.96 | 0.71 | -0.22 |
| 11 | 2 | 4 | 0.82 | 0.37 | -0.78 | -1.03 | 0.56 |
| 12 | 3 | 1 | -0.69 | 1.04 | 0.68 | 0.24 | -0.93 |

After each physical gate, compress globally to budget $k=11$, normalize, run at most three single-qubit passes, and then run one transient pair sweep on $(0,1),(2,3),(1,2),(3,4)$ in that order; every stage is unconditional, with no initial compression, delayed truncation or adaptive trigger.

The mixed response measures how the curvature in one control-error direction changes in the other direction, beyond separate quadratic sensitivities. This derivative extension is part of the present experiment. The following conventions define a unique local response of this finite procedure:

| Convention | Value |
| --- | --- |
| Coordinates | $U_j$ maps working to physical coordinates; global bit $b_j(x)=(x\gg j)\mathbin{\&}1$; pair index $2b_{q_1}(x)+b_{q_2}(x)$ |
| Truncation | Top $k$ squared magnitudes at $(s,u)=(0,0)$, smaller integer index on an exact tie, followed by normalization |
| Spectra | Eigenvalues decreasing; if any adjacent gap is at most $10^{-10}$, use identity for that proposal, constant in both parameters |
| Eigenvector gauge | At $(s,u)=(0,0)$, largest-magnitude component positive real, lowest index on an exact tie; continue with that same component positive real near the origin |
| Single-qubit order | $0,1,\ldots,4$; stop after a pass with no accepted update or after three passes |
| Guard | Accept only a participation-ratio decrease exceeding $10^{-12}$ |
| Local continuation | Fix all retained supports, acceptance decisions, pass counts, spectral-skip decisions and phase pivots to their values at $(s,u)=(0,0)$; differentiate all continuous amplitudes, RDMs, eigenframes and normalizations on that branch |
| Derivatives | Ordinary partial derivatives $v_{ab}=\partial_s^a\partial_u^b v(0,0)$ for $0\le a,b\le2$; the coefficient of $s^a u^b$ is $v_{ab}/(a!b!)$; no stochastic draws |

The reference evolves exactly under the same perturbed gates, and the scalar target is the ordinary mixed derivative $\partial_s^2\partial_u^2$ at the origin of $F(s,u)=|\langle\psi_{\mathrm{exact}}(s,u)|\psi_{\mathrm{approx}}(s,u)\rangle|^2$, after restoring all accumulated physical frames.

Provide a scientific derivation and numerical evaluation of this response. Your reasoning must address:

1. The retrieved single-qubit snapshot convention and complete transient pair-refinement sequence, including where its acceptance guard is evaluated and what is retained when a trial is rejected.
2. The second derivatives along $s$ of a reduced density matrix and of a normalized retained vector. Define the amplitude grouping, retained mass or norm, and derivative notation used in these expressions.
3. The coupled moving-eigenpair response through bidegree $(2,2)$, including the second-order eigenvector normalization term and continuation of the fixed positive-real pivot gauge. Equivalent direct-gauge or coefficient-recurrence derivations are acceptable.
4. The second derivative of an accumulated frame product and the mixed $(2,2)$ derivatives of the physical overlap and its squared modulus, accounting for the movement of both states.
5. The computed checks $F(0,0)$ and $\partial_sF(0,0)$, each to absolute tolerance $10^{-7}$, followed by the requested $\partial_s^2\partial_u^2F(0,0)$. Use absolute and relative tolerance $10^{-7}$ for the final response.

Output Format Requirements:
Complete the derivation and numerical calculation before writing the final response. Present the defining response equations and computed scalar checks in <reasoning>...</reasoning>, and give exactly one finite decimal in <final_answer>...</final_answer>. The final-answer tags contain only the requested mixed derivative, with no units or additional values. Keep the explanation focused on the requested derivation and calculation; a general procedure summary does not supply those equations or numerical checks. Report numerical values supported by your calculation. Do not invent completed calculations or substitute placeholder checks. Do not reproduce the input table, full state vectors or per-iteration traces.

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

01_propagate_frame_jets

Goal
----
Propagate supplied derivative orders through a physical gate in evolving local coordinates.

```python
def propagate_frame_jets(
    state: "np.ndarray", bases: "np.ndarray", gate: "np.ndarray", qubits: "np.ndarray"
) -> "np.ndarray":
    r"""Propagate supplied derivative orders through a physical gate in evolving local coordinates.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite ordinary derivatives of a normalized complex state, $2 \le N \le 6$.
    bases : np.ndarray, shape (J, N, 2, 2)
        Ordinary derivatives of local unitary maps from working to physical coordinates.
    gate : np.ndarray, shape (J, 4, 4)
        Ordinary derivatives of a physical unitary in the ordered pair basis.
    qubits : np.ndarray, shape (2,)
        Distinct integer indices in $[0,N-1]$.

    Returns
    -------
    propagated : np.ndarray, shape (J, 2**N)
        All supplied complex derivative orders of the propagated working state.

    Raises
    ------
    ValueError
        If shapes, finiteness, base normalization, base unitarity or qubit indices violate the stated domain.
    """
    return
```

### Step 2

02_reduced_density_jets

Goal
----
Compute local reduced-density responses including mixed orders.

```python
def reduced_density_jets(state: "np.ndarray", qubits: "np.ndarray") -> "np.ndarray":
    r"""Compute local reduced-density responses including mixed orders.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet with $1 \le N \le 6$.
    qubits : np.ndarray, shape (L,)
        One or two distinct integer qubit indices in their local order.

    Returns
    -------
    rdm : np.ndarray, shape (J, 2**L, 2**L)
        The reduced matrix and its supplied ordinary partial derivatives.

    Raises
    ------
    ValueError
        If the state shape, base normalization, finiteness or subsystem indices are invalid.
    """
    return
```

### Step 3

03_natural_frame_jets

Goal
----
Continue a local natural eigenframe through every supplied derivative order in a fixed phase convention.

```python
def natural_frame_jets(rdm: "np.ndarray", gap_tol: float = 1e-10) -> "np.ndarray":
    r"""Continue a local natural eigenframe through every supplied derivative order in a fixed phase convention.

    Parameters
    ----------
    rdm : np.ndarray, shape (J, D, D)
        Finite Hermitian jet, $D\in\{2,4\}$; base trace one and eigenvalues at least $-10^{-9}$.
    gap_tol : float, optional
        Real finite threshold in $[0,10^{-8}]$, default $10^{-10}$.

    Returns
    -------
    frames : np.ndarray, shape (J, D, D)
        The eigenframe and its supplied ordinary partial derivatives in the fixed pivot gauge.

    Raises
    ------
    ValueError
        If the shape, finiteness, Hermiticity, base density-matrix conditions or threshold are invalid.
    """
    return
```

### Step 4

04_compress_state_jets

Goal
----
Differentiate sparse projection, normalization and participation on a fixed retained support.

```python
def compress_state_jets(state: "np.ndarray", k: int) -> tuple:
    r"""Differentiate sparse projection, normalization and participation on a fixed retained support.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite complex derivative rows with positive base mass, $1 \le N \le 6$.
    k : int
        Retained budget in $[1,2^N]$.

    Returns
    -------
    result : tuple
        Ordered tuple of normalized state jet, shape (J, 2**N), retained-mass jet, shape (J,), and participation-ratio jet, shape (J,).

    Raises
    ------
    ValueError
        If the shape, finiteness, positive base mass or integer budget is invalid.
    """
    return
```

### Step 5

05_guarded_single_jet

Goal
----
Accept or reject a lossy single-qubit coordinate response using the base-state guard.

```python
def guarded_single_jet(
    state: "np.ndarray",
    bases: "np.ndarray",
    frame: "np.ndarray",
    qubit: int,
    k: int,
    accept_tol: float = 1e-12,
) -> tuple:
    r"""Accept or reject a lossy single-qubit coordinate response using the base-state guard.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet, $1 \le N \le 6$.
    bases : np.ndarray, shape (J, N, 2, 2)
        Jets of the accumulated local physical-frame maps.
    frame : np.ndarray, shape (J, 2, 2)
        Jet of the trial unitary in current local coordinates.
    qubit : int
        Local index in $[0,N-1]$.
    k : int
        Retained budget in $[1,2^N]$.
    accept_tol : float, optional
        Finite real tolerance in $[0,10^{-8}]$, default $10^{-12}$.

    Returns
    -------
    result : tuple
        Ordered tuple of state jet (J, 2**N), frame jets (J, N, 2, 2), retained-mass jet (J,), and numerical acceptance flag 0 or 1.

    Raises
    ------
    ValueError
        If any stated shape, base normalization, base unitarity, index, budget, finiteness or tolerance condition is invalid.
    """
    return
```

### Step 6

06_snapshot_jet_sweep

Goal
----
Propagate a state response through finite passes of snapshot-based local coordinate refinement.

```python
def snapshot_jet_sweep(
    state: "np.ndarray",
    bases: "np.ndarray",
    k: int,
    max_passes: int = 3,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    r"""Propagate a state response through finite passes of snapshot-based local coordinate refinement.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet, $1 \le N \le 6$.
    bases : np.ndarray, shape (J, N, 2, 2)
        Local physical-frame jets.
    k : int
        Retained budget in $[1,2^N]$.
    max_passes : int, optional
        One to three passes, default three.
    gap_tol : float, optional
        Finite real spectral threshold in $[0,10^{-8}]$, default $10^{-10}$.
    accept_tol : float, optional
        Finite real guard tolerance in $[0,10^{-8}]$, default $10^{-12}$.

    Returns
    -------
    result : tuple
        Ordered tuple of state jet (J, 2**N), frame jets (J, N, 2, 2), committed-mass jet (J,), and total number of accepted trials as an integer.

    Raises
    ------
    ValueError
        If any stated shape, base state/frame condition, budget, pass count, finiteness or tolerance is invalid.
    """
    return
```

### Step 7

07_transient_pair_jet

Goal
----
Differentiate a transient two-qubit refinement with two distinct sparse projections.

```python
def transient_pair_jet(
    state: "np.ndarray",
    qubits: "np.ndarray",
    k: int,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    r"""Differentiate a transient two-qubit refinement with two distinct sparse projections.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet, $2 \le N \le 6$.
    qubits : np.ndarray, shape (2,)
        Two distinct ordered integer qubit indices.
    k : int
        Retained budget in $[1,2^N]$.
    gap_tol : float, optional
        Finite real spectral threshold in $[0,10^{-8}]$, default $10^{-10}$.
    accept_tol : float, optional
        Finite real guard tolerance in $[0,10^{-8}]$, default $10^{-12}$.

    Returns
    -------
    result : tuple
        Ordered tuple of committed state jet (J, 2**N), retained-mass jet (J,), and numerical acceptance flag 0 or 1.

    Raises
    ------
    ValueError
        If the state, subsystem indices, budget or either tolerance violates the stated domain.
    """
    return
```

### Step 8

08_circuit_response_jets

Goal
----
Compose differentiated gate propagation, local frame updates and transient pair sweeps.

```python
def circuit_response_jets(
    initial: "np.ndarray",
    gates: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
) -> tuple:
    r"""Compose differentiated gate propagation, local frame updates and transient pair sweeps.

    Parameters
    ----------
    initial : np.ndarray, shape (J, 2**N)
        Finite normalized initial-state jet, $2 \le N \le 6$.
    gates : np.ndarray, shape (M, J, 4, 4)
        Finite physical unitary jets; $0 \le M \le 24$.
    pairs : np.ndarray, shape (M, 2)
        Distinct ordered integer qubit indices for every gate.
    k : int
        Retained budget in $[1,2^N]$.
    max_passes : int, optional
        One to three single-qubit passes per gate, default three.

    Returns
    -------
    result : tuple
        Ordered tuple of final working-state jet (J, 2**N), accumulated local-frame jets (J, N, 2, 2), and committed-mass jet (J,).

    Raises
    ------
    ValueError
        If any shape, finiteness, base state/gate condition, index, budget or pass count is invalid.
    """
    return
```

### Step 9

09_fidelity_curvature

Goal
----
Compute a pure or mixed curvature response of physical fidelity to coherent control errors.

```python
def fidelity_curvature(
    initial: "np.ndarray",
    angles: "np.ndarray",
    slopes: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
    cross_slopes: "np.ndarray | None" = None,
) -> float:
    r"""Compute a pure or mixed curvature response of physical fidelity to coherent control errors.

    Parameters
    ----------
    initial : np.ndarray, shape (2**N,)
        Finite normalized complex vector, independent of $s,u$, $2 \le N \le 6$.
    angles : np.ndarray, shape (M, 5)
        Finite real base angles in column order $(a,b,c,d,e)$, with $0 \le M \le 24$.
    slopes : np.ndarray, shape (M, 5)
        Finite real derivatives of angles with respect to dimensionless $s$.
    pairs : np.ndarray, shape (M, 2)
        Ordered distinct integer qubit indices.
    k : int
        Retained budget in $[1,2^N]$.
    max_passes : int, optional
        One to three single-qubit passes per gate, default three.

    cross_slopes : np.ndarray or None, shape (M, 5), optional
        Finite real angle derivatives with respect to u. An array requests the mixed (2,2) derivative; None requests the pure (2,0) derivative. A zero array therefore gives zero mixed response, not the pure response.

    Returns
    -------
    curvature : float
        The requested finite dimensionless ordinary (2,2) or (2,0) partial derivative of physical fidelity at zero.

    Raises
    ------
    ValueError
        If any shape, finiteness, base normalization, real-angle condition, index, budget or pass count is invalid.
    """
    return
```
