# Physics-Condensed_Matter_Physics-50

## Background

An open-boundary MPS represents a many-qubit amplitude tensor as an ordered product of low-rank site tensors.  In right-canonical form, the tensors to the right of any cut define an orthonormal basis, while the Schmidt singular values stored at that cut quantify the bipartite weight carried by each virtual mode.  Applying a neighboring two-site gate temporarily enlarges the virtual bond, so a conventional TEBD update forms a local two-site tensor, performs a singular-value decomposition, and retains a prescribed subspace.

An operator-Schmidt decomposition expresses a two-site gate as a sum of products of one-site operators.  This makes it possible to insert a paired left/right operator on the enlarged virtual bond while keeping the result in MPS form.  A stochastic-projector method replaces deterministic truncation by sampling a fixed-size singular-mode subspace and compensating the sampled projector so its ensemble average has the required identity action.

The benchmark uses fixed histories rather than runtime random draws.  Bra and ket histories are independent, their propagated MPS states are not normalized, and a Hermitian observable can therefore yield complex cross-trajectory samples even though the exact physical expectation is real.  The requested finite-sample statistic is the direct arithmetic mean of those complex samples, followed by taking its real part; it is not a norm-ratio or reweighted estimator.

## Problem

A six-qubit open-boundary matrix-product state (MPS) has physical dimension \(2\) and bond dimensions \((1,2,2,2,2,2,1)\); tensor axes are ordered as `(physical, left bond, right bond)`, site \(1\) is the most-significant computational-basis index, and all arithmetic uses binary64 real or complex precision. With zero-based indices \(n=0,\ldots,5\), \(\sigma=0,1\), \(a=0,\ldots,\chi_n-1\), and \(b=0,\ldots,\chi_{n+1}-1\), define
\[
x=11(n+1)+7(\sigma+1)+5(a+1)+3(b+1)+(n+1)(\sigma+1)(a+b+2),
\]
and
\[
A_n[\sigma,a,b]
=
\sin(0.37x)
+0.31\cos\!\left(0.19\left[x^2+3a+5b\right]\right)
+i\left\{
\cos\!\left(0.29\left[x+2ab\right]\right)
+0.23\sin\!\left(0.41\left[x^2+\sigma+b\right]\right)
\right\}.
\]

The fixed-size subset law, its associated inclusion probabilities, and the compensating left/right stochastic selectors must be obtained from the approved source construction; they must not be replaced by generic deterministic truncation, independent mode sampling, or an alternative stochastic-projector rule.

Normalize the contracted state once before transforming it to right-canonical form, and evolve it using the source-defined TEBD-based stochastic-projector construction with retained dimension \(\tilde{\chi}=2\), exponent \(c=1\), inverse cutoff \(r_{\mathrm{inv}}=10^{-6}\), no random-mode completion, and no normalization of stochastic trajectories.

The six chronological gate events are specified by scientific one-based bond indices \((2,4,3,2,4,3)\), Pauli axes \((X,Y,Z,Y,Z,X)\), and angles \((0.31,0.27,0.23,0.41,0.35,0.29)\) radians; each gate is \(\exp(i\theta P\otimes P)\). Construct every stochastic projector from the deterministic rank-two TEBD reference trajectory, retain the complete four-mode local decomposition at each event, and reuse that fixed time-ordered projector bank for all histories. Selected mode indices below are zero-based, each pair is unordered but written increasingly, and the six fixed ket histories are
\[
\begin{aligned}
&[(0,1),(0,2),(1,2),(0,1),(0,3),(1,3)],\\
&[(0,2),(1,2),(0,1),(0,3),(1,3),(0,1)],\\
&[(1,2),(0,1),(0,3),(1,3),(0,1),(0,2)],\\
&[(0,1),(0,3),(1,3),(0,1),(0,2),(1,2)],\\
&[(0,3),(1,3),(0,1),(0,2),(1,2),(0,1)],\\
&[(1,3),(0,1),(0,2),(1,2),(0,1),(0,3)].
\end{aligned}
\]
The independent bra histories are
\[
\begin{aligned}
&[(0,1),(1,2),(1,3),(0,1),(1,2),(1,3)],\\
&[(0,3),(0,1),(0,2),(0,3),(0,1),(0,2)],\\
&[(1,2),(1,3),(0,1),(1,2),(1,3),(0,1)],\\
&[(0,1),(0,2),(0,3),(0,1),(0,2),(0,3)],\\
&[(1,3),(0,1),(1,2),(1,3),(0,1),(1,2)],\\
&[(0,2),(0,3),(0,1),(0,2),(0,3),(0,1)].
\end{aligned}
\]

Let
\[
|\eta\rangle
=
\bigotimes_{n=1}^{6}
\left[
\cos\!\left(\frac{\alpha_n}{2}\right)|0\rangle
+
e^{i\phi_n}\sin\!\left(\frac{\alpha_n}{2}\right)|1\rangle
\right],
\]
with
\[
\boldsymbol{\alpha}
=
(0.41,0.73,1.02,0.58,0.91,0.36)
\]
and
\[
\boldsymbol{\phi}
=
(0.17,-0.29,0.43,0.61,-0.37,0.52),
\]
and define the dimensionless Hermitian observable
\[
O
=
0.70|\eta\rangle\langle\eta|
+
0.20(X\otimes I\otimes Y\otimes Z\otimes I\otimes X)
-
0.15(I\otimes Y\otimes X\otimes I\otimes Z\otimes Y).
\]

For each SVD, order singular values nonincreasingly and use a complete reduced SVD. During right-canonicalization, phase-fix each retained row of \(V^\dagger\) by locating its first maximum-magnitude entry, dividing that row by the entry’s complex phase, and multiplying the corresponding column of \(U\) by the same phase. For each local TEBD SVD, apply the analogous convention to each column of \(U\): divide the column by the phase of its first maximum-magnitude entry and multiply the corresponding row of \(V^\dagger\) by that phase. Treat any local mode satisfying \(s_k\le 10^{-6}s_0\), or any adjacent active gap satisfying \(s_k-s_{k+1}\le 10^{-10}s_0\), as outside the benchmark domain.

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

C1 — Right-Canonical Initial MPS

Goal
----
Transform a finite six-site open-boundary matrix-product state into the frozen right-canonical representation and retain the five internal Schmidt spectra.

```python
def canonicalize_initial_mps(
    site_tensors: "tuple[np.ndarray, ...]",
) -> "tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]":
    """Return a normalized right-canonical six-site MPS and five spectra.

    Each input tensor has axis order ``(physical, left_bond, right_bond)``.
    The outer bond dimensions must be one and every internal Schmidt rank must
    be exactly two with nondegenerate positive singular values.  The returned
    tensors have shapes ``(2,1,2)``, four copies of ``(2,2,2)``, and
    ``(2,2,1)``; the five spectra are real arrays of shape ``(2,)``.  Raise
    ``ValueError`` when these shape, finiteness, rank, or nondegeneracy
    conditions are not satisfied.
    """
    return canonical_tensors, bond_singular_values
```

### Step 2

C2 — Operator-Schmidt Gate Factors

Goal
----
Construct the ordered rank-two operator-Schmidt factors for every two-site Pauli-coupling gate in the benchmark circuit.

```python
def build_operator_schmidt_factors(
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return left and right rank-two gate factors of shape ``(L,2,2,2)``.

    ``pauli_axes`` uses ``0`` for X, ``1`` for Y, and ``2`` for Z.  Angles are
    real radians strictly between zero and pi/2.  Raise ``ValueError`` for
    mismatched shapes, invalid axis codes, nonfinite data, or invalid angles.
    """
    return left_gate_factors, right_gate_factors
```

### Step 3

C3 — Local TEBD Singular Decomposition

Goal
----
Form the complete local post-gate tensor and its canonicalized matrix, then compute the deterministic full singular-value decomposition used by the projector construction.

```python
def compute_local_tebd_svd(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return ``(C, Theta, X, s, Yh)`` for one internal rank-two bond.

    ``C``, ``Theta``, ``X``, and ``Yh`` are complex ``(4,4)`` arrays and ``s``
    is a real ``(4,)`` array.  All modes must exceed ``1e-6*s[0]`` and adjacent
    gaps must exceed ``1e-10*s[0]``.  Raise ``ValueError`` when the local input
    shapes or these complete-spectrum domain conditions are violated.
    """
    return post_gate_tensor, canonicalized_tensor, left_singular_vectors, singular_values, right_singular_vectors_h
```

### Step 4

C4 — TEBD Projector Bases

Goal
----
Construct the left and right TEBD projector bases that reproduce the complete local singular decomposition before subspace selection.

```python
def construct_tebd_projector_bases(
    left_tensor: "np.ndarray",
    right_tensor: "np.ndarray",
    left_bond_singular_values: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_singular_vectors: "np.ndarray",
    singular_values: "np.ndarray",
    right_singular_vectors_h: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return ``(Q_L, Q_R)`` with shapes ``(2,2,4)`` and ``(2,4,2)``.

    The inputs must describe a valid nondegenerate complete four-mode local SVD
    on an internal rank-two bond.  No mode is truncated in this construction.
    """
    return left_projector_basis, right_projector_basis
```

### Step 5

C5 — Fixed-Size Subspace Statistics

Goal
----
Enumerate the fixed-size singular-mode subspaces and compute their source-defined joint, marginal, and sequential conditional selection probabilities.

```python
def compute_subspace_statistics(
    singular_values: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return subsets, joint probabilities, marginals, and scan conditionals.

    For the frozen ``D=4`` and ``retained_dimension=2`` contract, the shapes are
    ``(6,2)``, ``(6,)``, ``(4,)``, and ``(4,2)``.  Subsets are lexicographically
    ordered.  Conditional entry ``[k,m]`` is the probability of accepting mode
    ``k`` after ``m`` earlier acceptances; unreachable zero-over-zero states are
    represented by zero.  Raise ``ValueError`` unless the spectrum is a finite,
    nonnegative, nonincreasing four-vector and the retained dimension is two.
    """
    return subsets, joint_probabilities, inclusion_probabilities, conditional_probabilities
```

### Step 6

C6 — Reference Projector Bank

Goal
----
Build the complete time-ordered TEBD projector bank along the deterministic rank-two reference trajectory.

```python
def build_reference_projector_bank(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_singular_values: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return time-ordered projector bases, spectra, and inclusion marginals.

    For ``L`` gate events the outputs have shapes ``(L,2,2,4)``,
    ``(L,2,4,2)``, ``(L,4)``, and ``(L,4)``.  Scientific bond indices are
    one-based and must select internal rank-two bonds.
    """
    return left_projector_bank, right_projector_bank, local_spectra, inclusion_probabilities
```

### Step 7

C7 — Weighted History Propagation

Goal
----
Propagate fixed stochastic subspace histories through the stored TEBD projector bank without intermediate normalization.

```python
def propagate_weighted_histories(
    canonical_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    left_gate_factors: "np.ndarray",
    right_gate_factors: "np.ndarray",
    left_projector_bank: "np.ndarray",
    right_projector_bank: "np.ndarray",
    inclusion_probabilities: "np.ndarray",
    histories: "np.ndarray",
) -> "np.ndarray":
    """Return the unnormalized dense states for all fixed histories.

    ``histories`` has shape ``(M,L,2)`` and contains strictly increasing
    zero-based mode pairs.  The return is a complex array of shape ``(M,64)``;
    no trajectory is normalized during or after propagation.  Raise
    ``ValueError`` for invalid shapes, indices, ordering, or probabilities.
    """
    return trajectory_states
```

### Step 8

C8 — Direct Cross-Trajectory Estimator

Goal
----
Evaluate independently paired bra/ket trajectory observables and their direct unreweighted finite-ensemble mean.

```python
def compute_direct_cross_estimator(
    bra_states: "np.ndarray",
    ket_states: "np.ndarray",
    observable: "np.ndarray",
) -> "tuple[np.ndarray, complex]":
    """Return the complex sample vector and its direct complex mean.

    The state arrays have matching shape ``(M,64)`` and the observable is a
    finite Hermitian ``(64,64)`` matrix.  The returned shapes are ``(M,)`` and
    scalar complex, respectively.  Raise ``ValueError`` for mismatched,
    nonfinite, empty, or non-Hermitian inputs.
    """
    return cross_samples, direct_mean
```

### Step 9

C9 — Final Estimate

Goal
----
Compose the complete six-site stochastic-projector pipeline and report the real part of the direct paired-trajectory estimate.

```python
def compute_final_estimate(
    site_tensors: "tuple[np.ndarray, ...]",
    bond_indices: "np.ndarray",
    pauli_axes: "np.ndarray",
    angles: "np.ndarray",
    ket_histories: "np.ndarray",
    bra_histories: "np.ndarray",
    observable: "np.ndarray",
) -> float:
    """Return the frozen dimensionless direct estimate rounded to 10 decimals.

    Histories must have matching shape ``(M,L,2)`` and correspond to the ``L``
    chronological gate events.  The retained subspace dimension is fixed at two.
    """
    return final_estimate
```
