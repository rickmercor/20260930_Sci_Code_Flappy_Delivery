# Mathematics-Numerical_Linear_Algebra-2

## Background

Double saddle-point linear systems arise in PDE-constrained optimization and contain multiple coupled constraint blocks. Efficient iterative solution of these systems depends strongly on preconditioners that preserve this block structure while keeping the cost of applying the preconditioner manageable. One important component is a Schur-complement-like coupling contribution associated with the SPD block $E$ and constraint matrix $C$. For large-scale problems, approximating such coupling contributions without explicitly constructing dense matrices is an important problem in numerical linear algebra.

## Problem

Double saddle-point linear systems arise in PDE-constrained optimization, where a diagonal-plus-randomized correction approximates the coupling contribution $C^TE^{-1}C$. Assess the response of the correction's leading generalized eigenspace to two simultaneous sketch perturbations.

For this benchmark, use:

* $p=10$, $m=80$, $k=9$, $r=3$, and $\varepsilon=10^{-6}$.
* One NumPy `default_rng(2026)` generator, drawing $G\in\mathbb{R}^{p\times p}$, $C\in\mathbb{R}^{p\times m}$, and $\Omega\in\mathbb{R}^{m\times k}$ in that order via `standard_normal`.
* $E=GG^T+pI_p$; the same rule applies to other input dimensions in the code subproblems.
* Direct dense solves for the exact and diagonal-surrogate inverse actions.
* $\Omega(s,t)=\Omega+sP+tQ$, where $P_{ij}=\sin(ij)/\sqrt{m}$ and $Q_{ij}=\cos(ij)/\sqrt{m}$ for one-based indices $1\leq i\leq m$, $1\leq j\leq k$, and angles in radians.

Use the source's full regularized construction in Algorithm 1, Step 6, with $E$, $C$, and $\varepsilon$ held fixed, and write its recovered correction as $D(s,t)=\widehat{\Delta}(s,t)$. For its source-defined sample matrix $W(s,t)$, introduce the benchmark metric $T(s,t)=I_m+W(s,t)W(s,t)^T$ and let $U_r(s,t)$ span the $r$ largest algebraic generalized eigenvalues of $D u=\lambda T u$, normalized by $U_r^TTU_r=I_r$.

The selected cluster has a positive gap to its complement near the origin, although eigenvalues within either cluster may be repeated; define its invariant projector in the original coordinates by $\Pi(s,t)=U_r(s,t)U_r(s,t)^TT(s,t)$. The source determines the correction construction, while the metric and spectral sensitivity query are benchmark-defined extensions.

Report the single scalar

$$
\left\|\left.\frac{\partial^2\Pi(s,t)}{\partial s\,\partial t}\right|_{s=t=0}\right\|_F.
$$

Compute the mixed derivative analytically or by automatic differentiation, including variation of both the correction and the metric; finite parameter differences are not the requested calculation. In the reasoning, give the source identities and the projector-derivative equations, and report the gap between the $r$th and $(r+1)$th largest generalized eigenvalues, $\|\Pi_s\|_F$, and $\|\Pi_t\|_F$ at the origin. Report the final scalar to at least 10 decimal places; the absolute answer tolerance is $10^{-9}$.

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

01_generate_deterministic_inputs

Goal
----
Generate the numerical inputs required by the subsequent randomized preconditioning subproblems and return them as a single state. Use one NumPy default_rng(seed), draw $G\in\mathbb{R}^{p\times p}$, then $C\in\mathbb{R}^{p\times m}$, then $\Omega\in\mathbb{R}^{m\times k}$, all via standard_normal, and form $E=GG^T+pI_p$.

```python
def generate_deterministic_inputs(
    seed: int, p: int, m: int, k: int
) -> tuple:
    r"""
    Deterministically generate the numerical inputs required by the
    subsequent randomized preconditioning subproblems.

    Parameters
    ----------
    seed : int
        Non-negative RNG seed.
    p : int
        Dimension parameter for the SPD block.
    m : int
        Column dimension of the constraint matrix.
    k : int
        Sketch size.

    Returns
    -------
    tuple
        $(E,C,\Omega)$.

    Raises
    ------
    ValueError
        If seed is negative, $p<1$, $m<p$, or $k<1$.
    """
    return E, C, Omega
```

### Step 2

02_solve_diagonal_and_exact_actions

Goal
----
Given the SPD block $E$, constraint matrix $C$, and sketch matrix $\Omega$, compute the inverse actions required by the subsequent randomized preconditioning subproblems and return the resulting state.

```python
def solve_diagonal_and_exact_actions(
    E: np.ndarray,
    C: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    r"""
    Compute the numerical action states required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $E$ is not square, symmetric, and SPD, $C$ has an incompatible
        shape with $E$, or $\Omega$ has an incompatible shape with $C$.
    """
    return Y, Y_D
```

### Step 3

03_form_sample_matrix

Goal
----
Form the sample matrix required by the subsequent randomized preconditioning subproblems from the outputs of the preceding step.

```python
def form_sample_matrix(
    C: np.ndarray,
    Y: np.ndarray,
    Y_D: np.ndarray,
) -> np.ndarray:
    r"""
    Form the matrix state required by the subsequent randomized
    preconditioning subproblems.

    Raises
    ------
    ValueError
        If $C$, $Y$, or $Y_D$ is not a 2D array, if $Y$ and $Y_D$ have
        different shapes, or if $C$ and $Y$ have incompatible row dimensions.
    """
    return W
```

### Step 4

04_thin_qr_orthonormal_basis

Goal
----
Extract the orthonormal basis required by the subsequent randomized preconditioning subproblems from the sample matrix.

```python
def thin_qr_orthonormal_basis(W: np.ndarray) -> np.ndarray:
    r"""
    Construct the orthonormal basis required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $W$ is not a 2D array or its dimensions satisfy $m<k$.
    """
    return V
```

### Step 5

05_form_regularized_core

Goal
----
Construct the core matrix required by the subsequent randomized preconditioning subproblem from the previously computed state and sketch.

```python
def form_regularized_core(
    W: np.ndarray,
    V: np.ndarray,
    Omega: np.ndarray,
    eps: float,
) -> np.ndarray:
    r"""
    Form the core matrix required by the subsequent randomized
    preconditioning subproblems.

    Raises
    ------
    ValueError
        If $W$, $V$, or $\Omega$ is not a 2D array, if $W$, $V$, and
        $\Omega$ do not have identical shapes, or if $\varepsilon\leq0$
        (the parameter eps).
    """
    return H
```

### Step 6

06_assemble_low_rank_correction

Goal
----
Assemble the low-rank correction matrix required by the subsequent randomized preconditioning subproblem from the previously computed basis and core.

```python
def assemble_low_rank_correction(
    V: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    r"""
    Assemble the correction matrix required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $V$ is not a 2D array, if $H$ is not a square 2D array, or if
        the column dimension of $V$ does not match the dimension of $H$.
    """
    return Delta_hat
```

### Step 7

07_differentiate_correction_sketch

Goal
----
Compute the first and mixed second derivatives of the recovered correction under the two-parameter sketch perturbation $\Omega(s,t)=\Omega+sP+tQ$.

```python
def differentiate_correction_sketch(
    W: np.ndarray,
    Omega: np.ndarray,
    W_s: np.ndarray,
    W_t: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    eps: float,
) -> np.ndarray:
    r"""
    Compute the derivative state required by the subsequent
    generalized-projector subproblems.

    Raises
    ------
    ValueError
        If any input array is not 2D, their shapes differ, or
        $\varepsilon\leq0$ or is nonfinite.
    """
    return derivatives
```

### Step 8

08_mixed_generalized_projector_derivative

Goal
----
Compute the mixed derivative of the invariant projector onto the leading generalized eigenspace of a symmetric matrix pencil with a varying positive-definite metric.

```python
def mixed_generalized_projector_derivative(
    D: np.ndarray,
    T: np.ndarray,
    D_s: np.ndarray,
    T_s: np.ndarray,
    D_t: np.ndarray,
    T_t: np.ndarray,
    D_st: np.ndarray,
    T_st: np.ndarray,
    r: int,
) -> np.ndarray:
    r"""
    Compute the mixed derivative state required by the final numerical
    evaluation.

    Raises
    ------
    ValueError
        If the arrays do not share a nonempty square shape, are not finite
        and symmetric to absolute tolerance $10^{-12}$, $T$ is not SPD,
        or $r$ is not an integer satisfying $1\leq r<n$.
    """
    return projector_st
```

### Step 9

09_full_pipeline_mixed_sketch_sensitivity

Goal
----
Run the randomized correction, its sketch derivatives, and the generalized spectral sensitivity calculation, returning $\|\Pi_{st}\|_F$.

```python
def full_pipeline_mixed_sketch_sensitivity(
    seed: int,
    p: int,
    m: int,
    k: int,
    eps: float,
    r: int,
) -> float:
    r"""
    Return the final scalar quantity for the deterministic benchmark
    instance.

    Raises
    ------
    ValueError
        If seed is negative, $p<1$, $m<p$, $k<1$, $k>m$,
        $\varepsilon$ is nonpositive or nonfinite, or $r$ is not an
        integer satisfying $1\leq r<m$.
    """
    return mixed_sensitivity
```
