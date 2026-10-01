# Mathematics-Computational_Mechanics-47

## Background

Projection-based model-order reduction replaces a large semi-discrete system by its restriction to a low-dimensional trial subspace, usually obtained from the proper orthogonal decomposition of solution snapshots. For linear dynamics the resulting operators can be precomputed once, and the online cost genuinely collapses to the reduced dimension. Nonlinear dynamics normally break that separation: an implicit time integrator must solve a Newton system at every step, and both the residual and its Jacobian are naturally evaluated by reconstructing the high-dimensional state, so the operation count keeps scaling with the full-order dimension.

Hyper-reduction is the standard remedy. It samples a small subset of the discrete equations and reweights them so that the projected residual and Jacobian are approximated cheaply. The saving is real, but it is bought with a second layer of approximation whose accuracy is governed by user-set sampling parameters, and for strongly nonlinear problems the number of sampled points needed to stay accurate can grow until the saving disappears.

A different route exploits structure rather than sparsity. Many semi-discrete systems are polynomial in the state, and those that are not can often be rewritten in polynomial form by a lifting transformation that introduces auxiliary fields defined algebraically from the original ones. Once the right-hand side is a fixed low-degree polynomial, the objects a reduced Newton solver needs are contractions of the full-order operators with the trial basis, and the state dependence enters only through low-dimensional Kronecker factors. Whether the resulting reduced solver is set up as a Galerkin projection or as a residual-minimizing Petrov-Galerkin projection changes which contractions are required and how the converged reduced state responds to the choice of test space.

Accuracy in this setting is judged against the original, unlifted high-fidelity trajectory, and is bounded below by how well the trial subspace can represent that trajectory at all. Reporting a prediction error together with that lower bound separates the error contributed by the subspace from the error contributed by the reduced dynamics.

## Problem

Projection-based reduced-order models of nonlinear dynamical systems normally lose their offline-online cost split, because the Newton solve at each implicit time step has to be rebuilt from the reconstructed full-order state; hyper-reduction restores the scaling only by layering a second, tolerance-dependent approximation on top of the projection. For a suitable class of semi-discrete systems that second approximation is avoidable altogether, and a reduced Newton solver can be built whose online cost does not scale with the full-order dimension. Solve one deterministic instance of such a solver and compare, over five unseen parameter values, the excess trajectory errors of its least-squares Petrov-Galerkin and Galerkin projections above the best approximation in their common trial subspace.

The full-order model is $\partial_t q = \kappa\,\partial_x^2 q - q^3 + u(x,t)$ on $x\in(0,1)$ with $\kappa=0.005$, Dirichlet data $q(0,t)=0$ and $q(1,t)=1$, and initial condition $q_0(x)=x(1-x)\left[6(1-x)^2e^{-x}-10e^{x}\sin(x/6)\right]+x$, while the source is $u(x,t)=\dfrac{a\sin(2\pi t)}{1+100\left(x-\tfrac14\right)^2}+\dfrac{b\sin(4\pi t)}{1+100\left(x-\tfrac34\right)^2}$ with $\boldsymbol\mu=(a,b)$. Discretize in space with second-order central differences on the $31$ interior nodes $x_i=i\Delta x$, $\Delta x=1/32$, imposing both Dirichlet values through ghost nodes, and integrate in time with backward Euler at $\Delta t=0.01$ on the levels $t_m=m\Delta t$, driving every implicit full-order Newton solve below a residual $2$-norm of $10^{-12}$. Training trajectories come from the five parameter sets $(-2,0)$, $(-1,-2)$, $(0,1)$, $(1,-1)$ and $(2,2)$ over $t\in[0,2]$, and the common test set is $\mathcal T=\{(-1.65,0.85),(0.85,0.75),(-0.35,-1.55),(1.15,-0.95),(0.45,1.35)\}$ over the longer horizon $t\in[0,5]$.

Apply the source methodology's lifted, hyper-reduction-free construction to this cubic system, preserving the stated spatial discretization and ghost-node boundary treatment throughout the lifted model. Build its block-structured POD trial space from the training data using a strict per-field neglected-squared-singular-value tolerance of $10^{-4}$ and the smallest admissible number of modes in each block. Carry out the source method's offline assembly and then march both its LSPG and Galerkin models with backward Euler, unit Newton step length, the previous accepted state as the initial iterate, and a $10^{-12}$ stopping tolerance, while requiring the online stage to use only arrays sized by the retained dimensions and the two inputs. For each $\boldsymbol\mu\in\mathcal T$, let $E_{\rm L}(\boldsymbol\mu)$ and $E_{\rm G}(\boldsymbol\mu)$ be the squared Frobenius trajectory errors of the LSPG and Galerkin temperature reconstructions and let $E_{\rm P}(\boldsymbol\mu)=\lVert(\mathbf I-\mathbf V\mathbf V^{\top})\mathbf Q_{\boldsymbol\mu}\rVert_F^2$ be the common projection floor; report $\max_{\boldsymbol\mu\in\mathcal T}[E_{\rm L}(\boldsymbol\mu)-E_{\rm P}(\boldsymbol\mu)]/[E_{\rm G}(\boldsymbol\mu)-E_{\rm P}(\boldsymbol\mu)]$, rounded to six decimal places, and in `<reasoning>` recover the lift, boundary contributions, exact offline residual-and-Jacobian representation, and the two projected Newton systems, give the retained mode counts and all five parameterwise excess-error ratios that determine the maximum, give $E_{\rm L}/E_{\rm P}$ and $E_{\rm G}/E_{\rm P}$ for the maximizing parameter before subtraction, and use the source heat-equation experiment to identify which HRF projection consistently gives the lowest speedup and to distinguish lifting's ROM-evaluation error from its effect on state-prediction error.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_cubic_heat_rhs

Goal
----
Evaluate the semi-discrete right-hand side of a diffusion equation carrying a cubic sink and a two-lobe unsteady source.

```python
from numbers import Real

import math
import numpy as np


def cubic_heat_rhs(
    state: np.ndarray,
    time: float,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> np.ndarray:
    """Evaluate the semi-discrete right-hand side of the cubic-reaction model.

    The field lives on the n interior nodes x_i = i * dx of (0, domain_length)
    with dx = domain_length / (n + 1) and i = 1, ..., n. The governing equation
    is

        dq/dt = diffusivity * d2q/dx2 - q**3 + s(x, time),

    the second derivative is approximated by the second-order central
    difference that uses boundary_values[0] as the ghost value at x = 0 and
    boundary_values[1] as the ghost value at x = domain_length, and the source
    is

        s(x, t) = amplitudes[0] * sin(2 * pi * t) / (1 + 100 * (x / L - 1 / 4)**2)
                + amplitudes[1] * sin(4 * pi * t) / (1 + 100 * (x / L - 3 / 4)**2)

    with L = domain_length.

    Parameters
    ----------
    state : np.ndarray
        Finite real nodal field of shape (n,) with n >= 1.
    time : float
        Finite real evaluation time.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.

    Returns
    -------
    derivative : np.ndarray
        Float array of shape (n,) holding dq/dt at the interior nodes.

    Raises
    ------
    ValueError
        If any array has the wrong rank or shape, if any entry is not finite,
        or if domain_length or diffusivity is not finite and positive.
    """
    return np.empty(0, dtype=float)
```

### Step 2

02_march_cubic_fom

Goal
----
Advance the full-order cubic-reaction model in time with an implicit one-step scheme solved by a Newton iteration.

```python
from numbers import Integral, Real

import math
import numpy as np


def march_cubic_fom(
    initial_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> np.ndarray:
    """March the semi-discrete cubic-reaction model with backward Euler.

    Time levels are t_m = m * time_step for m = 0, ..., n_steps. At each level
    the implicit state q_m solves

        q_m - q_{m-1} - time_step * cubic_heat_rhs(q_m, t_m, ...) = 0,

    which is solved by Newton iteration started from q_{m-1} with unit step
    length. The iteration stops as soon as the 2-norm of that residual is
    below residual_tolerance, and in any case after max_newton_iterations
    updates. Column 0 of the output is initial_state.

    Parameters
    ----------
    initial_state : np.ndarray
        Finite real nodal field of shape (n,) with n >= 1.
    time_step : float
        Finite strictly positive step size.
    n_steps : int
        Non-negative number of time steps to take.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.
    residual_tolerance : float
        Finite strictly positive Newton stopping tolerance.
    max_newton_iterations : int
        Positive cap on Newton updates per time step.

    Returns
    -------
    trajectory : np.ndarray
        Float array of shape (n, n_steps + 1) holding the nodal field at every
        time level.

    Raises
    ------
    ValueError
        If any array has the wrong rank or shape, if any entry is not finite,
        or if a scalar control violates the stated sign or type requirement.
    """
    return np.empty((0, 0), dtype=float)
```

### Step 3

03_lifted_polynomial_operators

Goal
----
Assemble the constant, linear, quadratic, input, and bilinear operators of the lifted at-most-quadratic system.

```python
from numbers import Integral, Real

import math
import numpy as np


def lifted_polynomial_operators(
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the operators of the lifted quadratic system.

    The joint state is x = [q; w], where q is the temperature on the n
    interior nodes x_i = i * dx of (0, domain_length), dx = domain_length /
    (n + 1), and w is the auxiliary field defined pointwise by w = q * q. The
    operators must satisfy

        dx/dt = C + A x + F (x kron x) + B u + N (u kron x)

    for every state that lies on w = q * q, where the temperature block
    reproduces the same central-difference diffusion, cubic sink and two-lobe
    source used by the unlifted model, the auxiliary block is the exact time
    derivative implied by the definition of w, both blocks impose
    boundary_values[0] at x = 0 and boundary_values[1] at x = domain_length
    through ghost nodes, and

        u(t) = [a * sin(2 * pi * t), b * sin(4 * pi * t)]

    multiplies the spatial profiles 1 / (1 + 100 * (x / L - 1 / 4)**2) and
    1 / (1 + 100 * (x / L - 3 / 4)**2) respectively, with L = domain_length.

    Storage conventions. Columns of F follow numpy's Kronecker layout, so the
    monomial x_p * x_q occupies column p * (2 n) + q with zero-based p and q;
    each monomial is placed entirely in the single column whose first index is
    the smaller of the two, leaving every column with p > q equal to zero.
    Columns of N likewise follow numpy's layout, so u_k * x_p occupies column
    k * (2 n) + p.

    Parameters
    ----------
    n_nodes : int
        Positive number of interior spatial nodes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.

    Returns
    -------
    constant_operator : np.ndarray
        Float array of shape (2 n,).
    linear_operator : np.ndarray
        Float array of shape (2 n, 2 n).
    quadratic_operator : np.ndarray
        Float array of shape (2 n, 4 n**2).
    input_operator : np.ndarray
        Float array of shape (2 n, 2).
    bilinear_operator : np.ndarray
        Float array of shape (2 n, 4 n).

    Raises
    ------
    ValueError
        If n_nodes is not a positive integer, if boundary_values has the wrong
        shape or a non-finite entry, or if domain_length or diffusivity is not
        finite and positive.
    """
    return (
        np.empty(0, dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
        np.empty((0, 0), dtype=float),
    )
```

### Step 4

04_block_pod_basis

Goal
----
Build the block-diagonal trial basis of the lifted system from temperature snapshots alone.

```python
from numbers import Real

import math
import numpy as np


def block_pod_basis(
    state_snapshots: np.ndarray,
    energy_tolerance: float,
) -> tuple[np.ndarray, int, int]:
    """Assemble the block-diagonal proper-orthogonal-decomposition trial basis.

    The auxiliary snapshot matrix is obtained from state_snapshots by the same
    pointwise squaring that defines the auxiliary field, so no second full-order
    solve is performed. Each of the two snapshot matrices is factorised by a
    thin singular value decomposition, and the retained mode count is the
    smallest k for which

        1 - (sum of the first k squared singular values)
            / (sum of all squared singular values) < energy_tolerance,

    evaluated independently for the two blocks. Each retained left singular
    vector is rescaled by +1 or -1 so that its entry of largest absolute value
    is positive; if several entries tie in absolute value, the one with the
    smallest index decides. The returned basis has the temperature modes in its
    upper-left block and the auxiliary modes in its lower-right block, with
    zeros elsewhere.

    Parameters
    ----------
    state_snapshots : np.ndarray
        Finite real array of shape (n, m) with n >= 1 and m >= 1 holding the
        temperature at m sampled states.
    energy_tolerance : float
        Finite tolerance in (0, 1) on the neglected fraction of squared
        singular values.

    Returns
    -------
    trial_basis : np.ndarray
        Float array of shape (2 n, n1 + n2).
    n_temperature_modes : int
        Number of retained temperature modes n1.
    n_auxiliary_modes : int
        Number of retained auxiliary modes n2.

    Raises
    ------
    ValueError
        If state_snapshots has the wrong rank, is empty or holds a non-finite
        entry, or if energy_tolerance is not a finite scalar in (0, 1).
    """
    return (np.empty((0, 0), dtype=float), 0, 0)
```

### Step 5

05_hrf_operator_gram

Goal
----
Precompute, once and offline, the Gram matrix of the column blocks that span the approximate residual and its reduced Jacobian.

```python
from numbers import Real

import math
import numpy as np


def hrf_operator_gram(
    trial_basis: np.ndarray,
    constant_operator: np.ndarray,
    linear_operator: np.ndarray,
    quadratic_operator: np.ndarray,
    input_operator: np.ndarray,
    bilinear_operator: np.ndarray,
) -> np.ndarray:
    """Form the offline Gram matrix of the reduced-space column blocks.

    Let Phi be the trial basis with N rows and n columns, let n_u be the number
    of inputs, and let C, A, F, B and N_op be the constant, linear, quadratic,
    input and bilinear operators of the polynomial system

        dx/dt = C + A x + F (x kron x) + B u + N_op (u kron x).

    Define the column-block matrix

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    with the six blocks in that order and with C contributing a single column.
    Return the Gram matrix K^T K. The Kronecker layouts are numpy's, matching
    the column conventions of F and N_op.

    Parameters
    ----------
    trial_basis : np.ndarray
        Finite real array of shape (N, n) with N >= 1 and n >= 1.
    constant_operator : np.ndarray
        Finite real array of shape (N,).
    linear_operator : np.ndarray
        Finite real array of shape (N, N).
    quadratic_operator : np.ndarray
        Finite real array of shape (N, N**2).
    input_operator : np.ndarray
        Finite real array of shape (N, n_u) with n_u >= 1.
    bilinear_operator : np.ndarray
        Finite real array of shape (N, n_u * N).

    Returns
    -------
    gram : np.ndarray
        Float array of shape (d, d) with d = 2 * n + n**2 + n_u * n + 1 + n_u.

    Raises
    ------
    ValueError
        If any argument has the wrong rank, if the shapes are mutually
        inconsistent, or if any entry is not finite.
    """
    return np.empty((0, 0), dtype=float)
```

### Step 6

06_kronecker_square_jacobian

Goal
----
Differentiate the Kronecker square of the reduced state with respect to that state.

```python
import numpy as np


def kronecker_square_jacobian(reduced_state: np.ndarray) -> np.ndarray:
    """Return the exact derivative of v kron v with respect to v.

    For a vector v of length n, the Kronecker square v kron v is the vector of
    length n**2 whose entry p * n + q is v_p * v_q, using numpy's layout with
    zero-based p and q. This function returns the n**2 by n matrix J whose
    entry (p * n + q, r) is the partial derivative of v_p * v_q with respect to
    v_r, so that for any matrix M with n**2 columns the derivative of
    M (v kron v) with respect to v is M J.

    Parameters
    ----------
    reduced_state : np.ndarray
        Finite real array of shape (n,) with n >= 1.

    Returns
    -------
    derivative : np.ndarray
        Float array of shape (n**2, n).

    Raises
    ------
    ValueError
        If reduced_state is not a non-empty 1-D array of finite real entries.
    """
    return np.empty((0, 0), dtype=float)
```

### Step 7

07_residual_coefficient_vector

Goal
----
Express the approximate residual of a linear multistep scheme as a coefficient vector against the precomputed column blocks.

```python
from numbers import Real

import math
import numpy as np


def residual_coefficient_vector(
    reduced_history: np.ndarray,
    input_history: np.ndarray,
    time_step: float,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Return the coefficients of the approximate residual in the column basis.

    Let Phi be the trial basis, let the polynomial right-hand side be

        f(x, u) = C + A x + F (x kron x) + B u + N_op (u kron x),

    and let the residual of the linear multistep scheme at step m be

        r = sum_j alphas[j] * Phi xhat_{m-j}
            - time_step * sum_j betas[j] * f(Phi xhat_{m-j}, u_{m-j}),

    where row j of reduced_history holds xhat_{m-j} and row j of input_history
    holds u_{m-j}. Return the vector c for which r = K c, where

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    is the same six-block matrix, in the same order and with the same numpy
    Kronecker layouts, whose Gram matrix is precomputed offline.

    Parameters
    ----------
    reduced_history : np.ndarray
        Finite real array of shape (tau + 1, n) with tau >= 0 and n >= 1.
    input_history : np.ndarray
        Finite real array of shape (tau + 1, n_u) with n_u >= 1.
    time_step : float
        Finite strictly positive step size.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.

    Returns
    -------
    coefficients : np.ndarray
        Float array of shape (d,) with d = 2 * n + n**2 + n_u * n + 1 + n_u.

    Raises
    ------
    ValueError
        If any array has the wrong rank, if the histories and coefficient
        arrays disagree in length, if any entry is not finite, or if time_step
        is not finite and positive.
    """
    return np.empty(0, dtype=float)
```

### Step 8

08_reduced_newton_direction

Goal
----
Solve one reduced Newton system for the implicit level of a projected polynomial model.

```python
from numbers import Real

import math
import numpy as np


def reduced_newton_direction(
    gram: np.ndarray,
    coefficients: np.ndarray,
    reduced_state: np.ndarray,
    current_input: np.ndarray,
    time_step: float,
    alpha_zero: float,
    beta_zero: float,
    scheme: str,
) -> tuple[np.ndarray, float]:
    """Return one reduced Newton direction and the current convergence measure.

    Let K be the six-block matrix

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    whose Gram matrix K^T K is supplied as gram, and let r = K coefficients be
    the approximate residual at the implicit level. Let J be the derivative of
    r with respect to the reduced state at reduced_state, with all preceding
    levels held fixed, so that J = K Z for a coefficient matrix Z determined by
    alpha_zero, beta_zero, time_step, reduced_state and current_input.

    The test basis is Phi, the first block of K, when scheme is "galerkin", and
    is J when scheme is "lspg". The step direction p solves

        (Psi^T J) p = -(Psi^T r),

    and the returned convergence measure is the 2-norm of Psi^T r. The routine
    receives no full-order quantity other than gram and must not form one.

    Parameters
    ----------
    gram : np.ndarray
        Finite real symmetric array of shape (d, d) with
        d = 2 * n + n**2 + n_u * n + 1 + n_u.
    coefficients : np.ndarray
        Finite real array of shape (d,) representing the residual in the
        column basis.
    reduced_state : np.ndarray
        Finite real array of shape (n,) holding the current implicit level.
    current_input : np.ndarray
        Finite real array of shape (n_u,) holding the input at the implicit
        level.
    time_step : float
        Finite strictly positive step size.
    alpha_zero : float
        Finite state coefficient of the implicit level.
    beta_zero : float
        Finite rate coefficient of the implicit level.
    scheme : str
        Either "galerkin" or "lspg".

    Returns
    -------
    step_direction : np.ndarray
        Float array of shape (n,).
    criterion_norm : float
        Native Python float equal to the 2-norm of the test-basis-projected
        residual before the step.

    Raises
    ------
    ValueError
        If any array has the wrong rank or an inconsistent shape, if any entry
        is not finite, if time_step is not finite and positive, or if scheme is
        not one of the two recognised names.
    """
    return (np.empty(0, dtype=float), 0.0)
```

### Step 9

09_march_reduced_rom

Goal
----
March the reduced state of the projected polynomial model with a reduced Newton solver.

```python
from numbers import Integral, Real

import math
import numpy as np


def march_reduced_rom(
    gram: np.ndarray,
    initial_reduced_state: np.ndarray,
    time_step: float,
    n_steps: int,
    amplitudes: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    scheme: str,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> np.ndarray:
    """Advance the reduced state with the reduced Newton solver.

    Time levels are t_m = m * time_step for m = 0, ..., n_steps, and the input
    at level m is

        u(t_m) = [amplitudes[0] * sin(2 * pi * t_m),
                  amplitudes[1] * sin(4 * pi * t_m)].

    The linear multistep weights alphas and betas have tau + 1 entries, index j
    referring to level m - j; any level with a negative index is replaced by
    level 0. At each step the iterate starts from the previous accepted level.
    One iteration evaluates the residual coefficients of the current iterate
    and the corresponding Newton direction with its convergence measure; if
    that measure is below residual_tolerance the iterate is accepted, otherwise
    the direction is added with unit step length and the iteration repeats, up
    to max_newton_iterations times. Column 0 of the output is
    initial_reduced_state.

    Parameters
    ----------
    gram : np.ndarray
        Finite real symmetric array of shape (d, d) with
        d = 2 * n + n**2 + 2 * n + 1 + 2 for two inputs.
    initial_reduced_state : np.ndarray
        Finite real array of shape (n,) with n >= 1.
    time_step : float
        Finite strictly positive step size.
    n_steps : int
        Non-negative number of time steps to take.
    amplitudes : np.ndarray
        Finite real array of shape (2,) holding the two source amplitudes.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.
    scheme : str
        Either "galerkin" or "lspg".
    residual_tolerance : float
        Finite strictly positive convergence tolerance.
    max_newton_iterations : int
        Positive cap on iterations per time step.

    Returns
    -------
    trajectory : np.ndarray
        Float array of shape (n, n_steps + 1).

    Raises
    ------
    ValueError
        If any array has the wrong rank or an inconsistent shape, if any entry
        is not finite, if a scalar control violates its stated sign or type
        requirement, or if scheme is not one of the two recognised names.
    """
    return np.empty((0, 0), dtype=float)
```

### Step 10

10_run_hrf_rom_pipeline

Goal
----
Run the complete hyper-reduction-free reduced-order pipeline and return one requested scalar.

```python
from numbers import Integral, Real

import math
import numpy as np


def run_hrf_rom_pipeline(
    quantity: str,
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    time_step: float,
    n_training_steps: int,
    n_online_steps: int,
    training_amplitudes: np.ndarray,
    test_amplitudes: np.ndarray,
    energy_tolerance: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> float:
    """Run the reduced-order pipeline end to end and report one scalar.

    The initial temperature on the interior nodes x_i = i * dx of
    (0, domain_length), with dx = domain_length / (n_nodes + 1), is

        q0(x) = s (1 - s) [6 (1 - s)**2 exp(-s) - 10 exp(s) sin(s / 6)] + s,
        s = x / domain_length.

    Call march_cubic_fom once per row of training_amplitudes over
    n_training_steps steps and concatenate the resulting trajectories
    column-wise into the temperature snapshot matrix. Pass that matrix to
    block_pod_basis to obtain the trial basis, build the operators of the
    lifted system with lifted_polynomial_operators, and precompute the offline
    bank with hrf_operator_gram. The initial reduced state is the projection of
    the lifted initial state onto the trial basis, the lifted initial state
    being the initial temperature followed by its pointwise square. Advance the
    reduced state with march_reduced_rom over n_online_steps steps at
    test_amplitudes, and obtain the reference trajectory by calling
    march_cubic_fom over the same horizon at test_amplitudes.

    Write Q for the reference temperature trajectory, Qt for the temperature
    block of the reconstructed reduced trajectory, and V for the temperature
    modes of the trial basis. The recognised values of quantity are

        "lspg_error_ratio"        squared Frobenius norm of Q - Qt under the
                                  least-squares scheme, divided by the squared
                                  Frobenius norm of (I - V V^T) Q
        "galerkin_error_ratio"    the same ratio under the Galerkin scheme
        "lspg_state_error"        squared Frobenius norm of Q - Qt under the
                                  least-squares scheme, divided by the squared
                                  Frobenius norm of Q
        "galerkin_state_error"    the same relative error under the Galerkin
                                  scheme
        "projection_error"        squared Frobenius norm of (I - V V^T) Q,
                                  divided by the squared Frobenius norm of Q
        "temperature_mode_count"  the number of retained temperature modes
        "auxiliary_mode_count"    the number of retained auxiliary modes
        "worst_excess_error_ratio" for a two-dimensional test_amplitudes array,
                                  compute both LSPG and Galerkin at every row
                                  and return the maximum of

                                  (E_lspg - E_projection)
                                  / (E_galerkin - E_projection),

                                  where each E is the corresponding squared
                                  Frobenius trajectory error

    Parameters
    ----------
    quantity : str
        One of the eight recognised names listed above.
    n_nodes : int
        Positive number of interior spatial nodes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.
    time_step : float
        Finite strictly positive step size shared by both horizons.
    n_training_steps : int
        Non-negative number of steps in each training trajectory.
    n_online_steps : int
        Non-negative number of steps in the online horizon.
    training_amplitudes : np.ndarray
        Finite real array of shape (n_train, 2) with n_train >= 1.
    test_amplitudes : np.ndarray
        Finite real array of shape (2,) for the first seven reporting names,
        or shape (n_test, 2) with n_test >= 1 for
        "worst_excess_error_ratio".
    energy_tolerance : float
        Finite tolerance in (0, 1) on the neglected fraction of squared
        singular values.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.
    residual_tolerance : float
        Finite strictly positive tolerance shared by both Newton solvers.
    max_newton_iterations : int
        Positive cap on iterations per time step.

    Returns
    -------
    value : float
        Native Python float equal to the requested scalar.

    Raises
    ------
    ValueError
        If quantity is not recognised, if any array has the wrong rank or
        shape, if any entry is not finite, or if a scalar control violates its
        stated sign or type requirement.
    """
    return 0.0
```
