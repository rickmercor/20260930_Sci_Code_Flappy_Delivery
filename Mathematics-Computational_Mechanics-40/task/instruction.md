# Mathematics-Computational_Mechanics-40

## Background

Projection-based model reduction represents a high-dimensional dynamical state in a low-dimensional trial space. Implicit integration of nonlinear systems remains expensive when residual and Jacobian evaluations reconstruct the full state at every Newton iteration.

Polynomial structure permits the fixed full-order operators to be expanded and contracted with a reduced basis before online simulation. For a quadratic residual, this produces low-degree state-polynomial coefficient tensors for the projected residuals and Newton matrices. The resulting reduced systems retain the underlying projection exactly and avoid the sampling approximation and tolerance choice introduced by conventional hyper-reduction.

Galerkin projection tests against the trial basis, while least-squares Petrov-Galerkin projection uses a state-dependent test basis tied to the residual Jacobian. Such methods are assessed through agreement with unreduced projection schemes, nonlinear convergence, state prediction error, and online cost.

## Problem

Projection-based model reduction can make implicit nonlinear dynamics inexpensive only if evaluating the reduced residual and Jacobian no longer scales with the full-order dimension. For polynomial systems, exact offline tensor contractions can replace approximate hyper-reduction, but Galerkin and least-squares Petrov-Galerkin (LSPG) projections generally produce different reduced states.

The full-order dynamics are \(\dot{x}=f(x,u)=C+Ax+F(x\otimes x)+Bu+N(u\otimes x)\), with a scalar input and \(F(x\otimes x)=q(x)\) defined below. From the supplied snapshots, construct a POD trial basis and advance one Crank-Nicolson time step with both projections. Derive and use the state-polynomial coefficient tensors for the reduced Galerkin Newton system and the LSPG normal system before either nonlinear iteration. A factorization through an auxiliary full-order residual library and its Gram matrix, or any full-order residual/Jacobian evaluation inside the iterations, does not satisfy this requirement.

Use the following deterministic configuration:

- `snapshots = [[1.20, 1.00, 0.75, -0.35, -0.65, -0.90], [0.35, 0.60, 0.90, 1.10, 1.30, 1.45], [-0.55, -0.25, 0.05, 0.45, 0.85, 1.10], [0.85, 0.55, 0.20, -0.10, -0.45, -0.70], [-0.25, 0.05, 0.35, 0.70, 1.00, 1.25]]`
- `energy_tolerance = 0.035`, using the smallest retained rank for which the discarded squared singular-value fraction is strictly below this value
- POD sign convention: orient each retained left singular vector so its largest-magnitude entry is positive
- `C = [0.08, -0.04, 0.03, 0.05, -0.02]`
- `A = [[-0.70, 0.20, 0.00, -0.10, 0.05], [0.15, -0.50, 0.25, 0.00, -0.08], [-0.10, 0.12, -0.60, 0.18, 0.10], [0.05, -0.14, 0.22, -0.45, 0.16], [0.10, 0.02, -0.12, 0.20, -0.55]]`
- `q_0(x) = 0.50*x_0*x_1 - 0.35*x_2^2 + 0.12*x_3*x_4`
- `q_1(x) = -0.42*x_0^2 + 0.30*x_1*x_3 - 0.18*x_2*x_4`
- `q_2(x) = 0.28*x_0*x_2 + 0.40*x_1^2 - 0.25*x_3*x_4`
- `q_3(x) = -0.33*x_1*x_2 + 0.22*x_0*x_4 + 0.15*x_3^2`
- `q_4(x) = 0.31*x_2*x_3 - 0.27*x_1*x_4 + 0.20*x_0^2`, with row-major Kronecker ordering
- `B = [0.15, -0.08, 0.04, 0.10, -0.12]`
- `N = [[0.10, -0.05, 0.00, 0.03, 0.00], [0.02, 0.08, -0.04, 0.00, 0.01], [-0.03, 0.02, 0.06, -0.02, 0.00], [0.00, 0.04, 0.01, 0.07, -0.03], [0.05, 0.00, -0.02, 0.03, 0.09]]`
- `x_previous = [1.75, -1.92, -0.96, -0.42, -2.00]`, projected as `z_previous = Phi^T x_previous`
- `u_previous = 0.41`, `u_current = -0.85`, and `dt = 1.6`
- reduced Crank-Nicolson residual convention: use the projected previous-state anchor \(x_{\mathrm{anchor}}=\Phi z_{\mathrm{previous}}\), so \(r_{\mathrm{CN}}(z)=\Phi z-\Phi z_{\mathrm{previous}}-\frac{dt}{2}[f(\Phi z,u_{\mathrm{current}})+f(\Phi z_{\mathrm{previous}},u_{\mathrm{previous}})]\), without dividing the residual by `dt`; the supplied full `x_previous` is used only to form `z_previous`
- both Newton initial states equal `z_previous`, with unit step length, convergence tolerance `1e-10`, and at most `25` updates
- compact-versus-direct system equivalence threshold `1e-12` in maximum absolute component error
- numerical grading convention: a reported derived real scalar or vector component is accepted when \(|v-v_{\mathrm{ref}}|\le 10^{-10}+10^{-9}|v_{\mathrm{ref}}|\); integer ranks, counts, and shapes are exact, while convergence and equivalence claims are evaluated against their stated upper bounds

Give a compact numerical certificate that makes the POD truncation, the offline coefficient-tensor construction, one initial-state check for each projected system, both terminal stationarity conditions, independent compact/direct agreement, and the signed reconstruction checkable. Choose your own notation and presentation order. Do not reproduce full coefficient tensors, per-iteration paths, or a generic implementation pipeline. The final answer is a single number: the signed component-0 gap `x_LSPG[0] - x_Galerkin[0]`.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the key dimensions and scalars needed to audit the result.
Do not paste the input matrices, full coefficient tensors, per-iteration paths, or per-fold candidate tables.

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
```

### Step 1

01_select_pod_basis

Goal
----
Proper orthogonal decomposition represents a snapshot matrix X by its leading left singular vectors. The retained dimension n is the smallest index satisfying 1 - sum_{i=1}^n sigma_i^2 / sum_i sigma_i^2 < energy_tolerance. Singular vectors have arbitrary signs, so each retained vector is oriented with its largest-magnitude entry positive to make the numerical basis deterministic.

```python
import numpy as np


def select_pod_basis(snapshots: np.ndarray, energy_tolerance: float) -> np.ndarray:
    """Select and orient a POD basis using a truncated-energy criterion.

    Parameters
    ----------
    snapshots : np.ndarray
        Finite, nonempty snapshot matrix of shape (n_state, n_snapshots) with
        strictly positive total squared singular-value energy.
    energy_tolerance : float
        Required upper bound in the open interval (0, 1) on discarded energy.

    Returns
    -------
    basis : np.ndarray
        Orthonormal POD basis of shape (n_state, n_reduced).

    Raises
    ------
    ValueError
        If snapshots is not a finite, nonempty, positive-energy matrix or if
        energy_tolerance is not a scalar strictly between 0 and 1.
    """
    return basis
```

### Step 2

02_assemble_quadratic_operator

Goal
----
A quadratic full-order term has the matrix form F(x kron x), where column i * N + j multiplies x_i x_j under row-major Kronecker ordering. A sparse monomial table stores each nonzero as (output_index, first_state_index, second_state_index, coefficient); repeated entries add to the same tensor coefficient.

```python
import numpy as np


def assemble_quadratic_operator(n_state: int, terms: np.ndarray) -> np.ndarray:
    """Assemble the full quadratic operator from sparse monomial terms.

    Parameters
    ----------
    n_state : int
        Positive dimension N of the full-order state.
    terms : np.ndarray
        Array of shape (n_terms, 4) containing output, first input, second input,
        and coefficient columns. The first three columns must contain
        integer-valued indices in the half-open range [0, N).

    Returns
    -------
    quadratic : np.ndarray
        Quadratic operator F of shape (N, N * N).

    Raises
    ------
    ValueError
        If n_state is not a positive integer, if terms is not a finite array
        with four columns, or if a term index is nonintegral or outside [0, N).
    """
    return quadratic
```

### Step 3

03_assemble_residual_polynomial

Goal
----
For a fixed implicit time step, a quadratic full-order dynamical system restricted to a trial basis has a full residual that is a quadratic polynomial in the unknown reduced state. Separating its constant, linear, and ordered quadratic coefficients exposes the structure needed for hyper-reduction-free tensor contraction while preserving direct full-order verification.

```python
import numpy as np


def assemble_residual_polynomial(
    basis: np.ndarray,
    constant: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    input_vector: np.ndarray,
    bilinear: np.ndarray,
    previous_state: np.ndarray,
    current_input: float,
    previous_input: float,
    time_step: float,
) -> dict:
    """Assemble the Crank-Nicolson residual polynomial in reduced coordinates.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    constant : np.ndarray
        Finite full-order constant operator C of shape (N,).
    linear : np.ndarray
        Finite full-order linear operator A of shape (N, N).
    quadratic : np.ndarray
        Finite quadratic operator F of shape (N, N * N), using row-major
        Kronecker ordering.
    input_vector : np.ndarray
        Finite scalar-input operator B of shape (N,).
    bilinear : np.ndarray
        Finite scalar-input/state operator N of shape (N, N).
    previous_state : np.ndarray
        Finite reduced state at the previous time level, shape (n,).
    current_input, previous_input : float
        Finite scalar inputs at the current and previous time levels.
    time_step : float
        Finite, positive Crank-Nicolson step size.

    Returns
    -------
    polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n), such
        that the ordered last two axes multiply z_j z_k.

    Raises
    ------
    ValueError
        If a documented shape, finiteness, or positivity requirement fails.
    """
    return polynomial
```

### Step 4

04_precompute_hrf_tensors

Goal
----
The hyper-reduction-free formulation expands the projected Newton operators as low-degree polynomials of the reduced state. For a quadratic residual, the Galerkin residual remains quadratic, the residual Jacobian is affine, the least-squares normal matrix is quadratic, and the least-squares gradient is cubic. Their coefficient tensors can be contracted before the online nonlinear iteration.

```python
import numpy as np


def precompute_hrf_tensors(basis: np.ndarray, residual_polynomial: dict) -> dict:
    """Precompute coefficient tensors for exact Galerkin and LSPG evaluation.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
        Quadratic axes are ordered as output, first state index, second state
        index and are not assumed symmetric.

    Returns
    -------
    tensors : dict
        Exactly galerkin_constant (n,), galerkin_linear (n, n),
        galerkin_quadratic (n, n, n), lspg_gradient_constant (n,),
        lspg_gradient_linear (n, n), lspg_gradient_quadratic (n, n, n),
        lspg_gradient_cubic (n, n, n, n), lspg_normal_constant (n, n),
        lspg_normal_linear (n, n, n), and lspg_normal_quadratic
        (n, n, n, n). State-polynomial coefficient indices precede normal-
        matrix row and column indices.

    Raises
    ------
    ValueError
        If the dictionary keys, shapes, or finiteness requirements fail.
    """
    return tensors
```

### Step 5

05_evaluate_hrf_galerkin

Goal
----
The Galerkin online kernel evaluates a contracted quadratic residual and its exact reduced-state Jacobian. The ordered quadratic tensor need not be symmetric, so differentiation must preserve the contributions from both state-index placements rather than silently symmetrizing its storage.

```python
import numpy as np


def evaluate_hrf_galerkin(tensors: dict, state: np.ndarray) -> dict:
    """Evaluate the Galerkin residual and Jacobian from contracted tensors.

    Parameters
    ----------
    tensors : dict
        The exact 10-array output of precompute_hrf_tensors.
    state : np.ndarray
        Finite, nonempty reduced state z of shape (n,).

    Returns
    -------
    system : dict
        Exactly residual (n,) and jacobian (n, n).

    Raises
    ------
    ValueError
        If tensor keys, tensor shapes, state shape, or finiteness is invalid.
    """
    return system
```

### Step 6

06_evaluate_hrf_lspg

Goal
----
The hyper-reduction-free LSPG online kernel evaluates a cubic coefficient expansion for the normal residual and a quadratic expansion for the Gauss-Newton matrix. These degrees arise from multiplying an affine residual Jacobian by a quadratic residual, and they retain the nested reduced-state interactions that a generic one-line Gram shortcut conceals.

```python
import numpy as np


def evaluate_hrf_lspg(tensors: dict, state: np.ndarray) -> dict:
    """Evaluate the LSPG gradient and normal matrix from contracted tensors.

    Parameters
    ----------
    tensors : dict
        The exact 10-array output of precompute_hrf_tensors.
    state : np.ndarray
        Finite, nonempty reduced state z of shape (n,).

    Returns
    -------
    system : dict
        Exactly residual (n,) for the LSPG gradient and jacobian (n, n) for
        the Gauss-Newton normal matrix.

    Raises
    ------
    ValueError
        If tensor keys, tensor shapes, state shape, or finiteness is invalid.
    """
    return system
```

### Step 7

07_certify_hrf_tensors

Goal
----
Exactness is a defining property of the hyper-reduction-free construction. At arbitrary reduced states, contracted Galerkin and LSPG systems can be compared with independently evaluated full residuals and residual Jacobians. This detects tensor-axis mistakes, missing quadratic derivative placements, and omitted cubic interactions without accepting a sampled approximation.

```python
import numpy as np


def certify_hrf_tensors(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    states: np.ndarray,
) -> dict:
    """Measure compact-versus-direct errors at one or more reduced states.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
    tensors : dict
        Ten finite arrays with the key and axis conventions documented for
        precompute_hrf_tensors. They need not already be consistent with the
        supplied polynomial; the returned errors diagnose consistency.
    states : np.ndarray
        Finite array of shape (m, n), with m positive.

    Returns
    -------
    certificate : dict
        Exactly errors of shape (m, 4). Columns are maximum absolute errors
        for Galerkin residual, Galerkin Jacobian, LSPG gradient, and LSPG
        normal matrix, in that order.

    Raises
    ------
    ValueError
        If dictionary keys, shapes, or finiteness requirements fail.
    """
    return certificate
```

### Step 8

08_solve_hrf_pair

Goal
----
The paired reduced solve performs all nonlinear work through precontracted coefficient tensors. Galerkin and LSPG share an initial reduced state but evaluate polynomials of different degrees and stop against different stationarity residuals. No full-order state, residual, Jacobian, or residual-library Gram matrix is available inside the iteration.

```python
import numpy as np


def solve_hrf_pair(
    tensors: dict,
    previous_state: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Solve the tensor-expanded Galerkin and LSPG reduced systems.

    Parameters
    ----------
    tensors : dict
        The exact 10-array output of precompute_hrf_tensors.
    previous_state : np.ndarray
        Finite, nonempty shared initial reduced state, shape (n,).
    tolerance : float
        Finite, positive convergence tolerance for each scheme's own
        stationarity residual norm.
    max_iterations : int
        Positive maximum number of unit Newton updates per scheme.

    Returns
    -------
    solution : dict
        Exactly states (2, n), iterations (2,), and convergence_norms (2,),
        with Galerkin in row 0 and LSPG in row 1.

    Raises
    ------
    ValueError
        If an input contract fails or either iteration does not converge.
    """
    return solution
```

### Step 9

09_compute_projection_gap

Goal
----
The projection gap is certified from the polynomial residual rather than from the online solver alone. Re-evaluating both projected systems at the shared initial state and at the two terminal states detects internally consistent solves built from incorrectly contracted coefficient tensors.

```python
import numpy as np


def compute_projection_gap(
    basis: np.ndarray,
    residual_polynomial: dict,
    tensors: dict,
    previous_state: np.ndarray,
    solution: dict,
    component_index: int = 0,
) -> dict:
    """Certify two reduced solutions and compute their signed component gap.

    Parameters
    ----------
    basis : np.ndarray
        Finite trial basis Phi of shape (N, n).
    residual_polynomial : dict
        Exactly constant (N,), linear (N, n), and quadratic (N, n, n).
    tensors : dict
        Ten finite arrays with the key and axis conventions documented for
        precompute_hrf_tensors. They need not be consistent with the supplied
        polynomial; the returned errors diagnose consistency.
    previous_state : np.ndarray
        Finite reduced state at the previous time level, shape (n,).
    solution : dict
        Dictionary containing finite states of shape (2, n), with Galerkin in
        row 0 and LSPG in row 1.
    component_index : int
        Full-order component index in the half-open range [0, N).

    Returns
    -------
    certificate : dict
        Exactly gap, reconstructed_components (2,), direct_residual_norms
        (2,), and compact_errors (3, 4). Error rows correspond to the shared
        initial, Galerkin terminal, and LSPG terminal states; columns follow
        certify_hrf_tensors.

    Raises
    ------
    ValueError
        If a key, shape, finiteness, or component-index requirement fails.
    """
    return certificate
```

### Step 10

10_run_full_pipeline

Goal
----
The end-to-end calculation must preserve one basis orientation and one tensor-axis convention through residual assembly, offline contraction, two different reduced Newton systems, and independent compact/direct certification. The final scalar is formed only after both terminal states satisfy their own stationarity conditions.

```python
def run_full_pipeline(energy_tolerance: float = 0.035) -> float:
    """Run the deterministic HRF projection-gap calculation.

    Parameters
    ----------
    energy_tolerance : float
        POD discarded-energy threshold in the open interval (0, 1).

    Returns
    -------
    projection_gap : float
        Component-0 reconstructed LSPG state minus reconstructed Galerkin state.

    Raises
    ------
    ValueError
        If energy_tolerance is not a scalar strictly between 0 and 1.
    """
    return projection_gap
```
