# Mathematics-Computational_Finance-16

## Background

Rare-event simulation and posterior calibration for financial risk can be expressed through reverse diffusions and Hamilton-Jacobi-Bellman equations. Their state dimension grows when wealth, predictors, volatility, and other risk factors evolve jointly, making conventional grids impractical.

Backward stochastic differential equation methods replace a spatial grid by regression along simulated factor trajectories. Low-rank functional tensor representations compress the multivariate value function and expose local linear problems, while derivative-aware losses target the gradient needed for reverse-diffusion drift corrections.

The resulting calculation produces a time-indexed approximation of the value and its sensitivity to each factor. Numerical reliability is assessed through regression residuals, stable tensor orthogonalization, sensitivity to regularization, and comparison of the recovered drift with reference solutions.

## Problem

Rare-event simulation and posterior calibration for risk-sensitive portfolios can be formulated as reverse-diffusion sampling on a multivariate financial state. The associated high-dimensional Hamilton-Jacobi-Bellman equation has a value gradient that determines the drift correction of the scenario sampler. A backward stochastic differential equation regression with a functional tensor-train value representation can estimate one time slice without a state-space grid.

For a standardized state whose coordinates represent log wealth, a return predictor, and a volatility factor, use the affine drift `f(x) = F x + b`, diagonal diffusion `sigma_diag`, and affine trajectory policy `u(x) = U x + c`. Represent both time slices in the quadratic tensor-product basis `phi(z) = [1, z, z^2]`, with every core stored in `(left rank, basis, right rank)` order and contracted from the first state coordinate to the third.

Your task is to solve one deterministic backward interval with the explicit value-plus-gradient BSDE loss and two consecutive forward ALS micro-steps, first at the middle core and then at the last core after a reduced SVD core shift. Use the derivative-weighted local system, residual-scaled regularization update, and reverse-diffusion feedback convention with the following configuration:

- `states = [[-0.8, -0.5, 0.2], [-0.6, 0.3, -0.4], [-0.3, 0.7, 0.6], [-0.1, -0.8, -0.7], [0.2, -0.2, 0.9], [0.4, 0.5, -0.1], [0.7, -0.6, 0.4], [0.9, 0.1, -0.8], [0.5, 0.9, 0.7], [-0.7, 0.8, -0.2]]`
- `noises = [[0.2, -1.1, 0.5], [-0.7, 0.4, 1.2], [1.1, -0.3, -0.8], [-1.3, 0.9, 0.2], [0.6, 1.0, -1.0], [-0.2, -0.7, 0.9], [0.8, 0.2, -0.5], [-0.9, -1.2, 0.7], [0.3, 0.6, 1.1], [1.2, -0.5, -0.3]]`
- `F = [[0.05, -0.08, 0.02], [0.03, -0.12, 0.04], [-0.01, 0.06, -0.09]]`
- `b = [0.01, -0.02, 0.015]`
- `sigma_diag = [0.35, 0.22, 0.18]`
- `U = [[-0.25, 0.08, -0.04], [0.05, -0.18, 0.06], [-0.03, 0.04, -0.12]]`
- `c = [0.02, -0.01, 0.015]`
- `dt = 0.15`
- `next_cores = [[[[0.75, -0.20], [0.10, 0.35], [-0.08, 0.12]]], [[[0.90, -0.15], [0.20, 0.25], [-0.10, 0.18]], [[0.05, 0.70], [-0.30, 0.12], [0.22, -0.08]]], [[[0.80], [0.15], [-0.05]], [[-0.20], [0.60], [0.10]]]]`
- `initial_cores = [[[[0.7071067811865476, 0.0], [0.0, 1.0], [0.7071067811865476, 0.0]]], [[[0.4, -0.1], [0.15, 0.35], [-0.05, 0.12]], [[0.08, 0.3], [-0.22, 0.1], [0.18, -0.07]]], [[[0.7071067811865476], [0.0], [0.7071067811865476]], [[0.0], [1.0], [0.0]]]]`
- `tau_initial = 0.07`
- `gamma = 0.2`
- `query = [0.35, -0.15, 0.55]`
- Treat the displayed univariate basis as orthonormal in the Hilbert inner product used for regularization, so the value-function norm equals the Frobenius norm of the active core when the other cores are orthonormal.
- The deterministic integral uses the right endpoint, the stochastic integral uses the left endpoint, and the same supplied noise row is used in the state step and BSDE feature row.
- Each ridge micro-step solves `(A^T A + tau I)c_core = A^T y`; tensor cores and local coefficients use C-order vectorization, and all algebra uses binary64 arithmetic.
- At each advanced state `x_prime`, set `g = grad V_next(x_prime)` and `u_prime = U x_prime + c`, and use the nonlinear generator `h = tr(F) + 0.5 ||diag(sigma_diag) g||_2^2 + u_prime dot diag(sigma_diag) g`.
- After the middle-core update, use `tau_next = gamma ||A_2 c_2 - y||_2^2 / ||c_2||_2^2`, with the numerator equal to the unnormalized squared residual and no division by the sample count.
- Recover the feedback at `query` as `-diag(sigma_diag) grad V(query)`.

Determine the advanced states, continuation values and gradients, BSDE targets, derivative-weighted local systems, adapted regularization, shifted tensor-train representation, and feedback drift correction at `query`. In the concise reasoning, state the value-plus-gradient BSDE identity and nonlinear generator, derivative-channel construction, adaptive regularization rule, and feedback rule, then report `y_0`, `||A_2||_F`, the leading C-order middle-core coefficient, `tau_next`, the updated last-core Frobenius norm, the first query-gradient component, and the first feedback component; these seven scalars are sufficient, and complete intermediate arrays are not required. Your final answer must be a single number: the first component of the adapted feedback at `query` at full precision.

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

advance_controlled_diffusion

Goal
----
A controlled factor state follows an Euler-Maruyama step of the reverse diffusion. The same noise increment later enters the derivative term of the BSDE regression, coupling the discrete state trajectory to the local regression problem solved at each time slice. For row k, the affine drift is f(x_k) = F x_k + b, the feedback is u(x_k) = U x_k + c, and diagonal diffusion s gives x'_k = x_k + [f(x_k) + s * u(x_k)] dt + s * xi_k sqrt(dt).

Inputs
------
states: Float array of shape (K, d).
noises: Float array of shape (K, d).
drift_matrix: Float array of shape (d, d).
drift_bias: Float array of shape (d,).
sigma_diag: Positive float array of shape (d,).
policy_matrix: Float array of shape (d, d).
policy_bias: Float array of shape (d,).
dt: Positive time increment.

Returns
-------
next_states: Float array of shape (K, d).

```python
def advance_controlled_diffusion(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance every controlled state by one Euler-Maruyama interval.

    Parameters
    ----------
    states : np.ndarray
        Current states with shape (K, d).
    noises : np.ndarray
        Standard-normal innovations with the same shape as `states`.
    drift_matrix : np.ndarray
        Matrix F of the affine drift.
    drift_bias : np.ndarray
        Bias b of the affine drift.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    policy_matrix : np.ndarray
        Matrix U of the affine feedback policy.
    policy_bias : np.ndarray
        Bias c of the affine feedback policy.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If an input has an incompatible shape, contains a non-finite value, if
        `sigma_diag` is not positive, or if `dt` is not finite and positive.

    Returns
    -------
    next_states : np.ndarray
        Advanced states with shape (K, d).
    """
    return next_states  # noqa: F821
```

### Step 2

evaluate_tt_continuation

Goal
----
An extended tensor train represents a scalar function by contracting one matrix-valued core per coordinate against a univariate basis. Returning the value beside every partial derivative supplies the continuation data required by a BSDE time step. A core with basis mode m_i uses the coordinate-specific monomial vector phi_i(z) = [1, z, ..., z^(m_i-1)]; a partial derivative replaces only that vector by phi_i'(z). The prescribed quadratic instance is recovered when every m_i equals three.

Inputs
------
points: Float array of shape (K, d).
cores: Sequence of d compatible order-three TT cores with basis modes at least two.

Returns
-------
continuation: Float array of shape (K, d + 1), with values in column zero and gradients in the remaining columns.

```python
def evaluate_tt_continuation(points: np.ndarray, cores: list[np.ndarray]) -> np.ndarray:
    """Evaluate a coordinate-wise monomial tensor train and its gradient.

    Parameters
    ----------
    points : np.ndarray
        Evaluation points with shape (K, d).
    cores : list[np.ndarray]
        Compatible TT cores with shapes (r_{i-1}, m_i, r_i), r_0 = r_d = 1,
        and m_i at least two. Core i uses monomials of degrees 0 through m_i-1.

    Raises
    ------
    ValueError
        If `points` is empty or non-finite, if the number or shape of the cores
        is incompatible, if a basis mode is less than two, or if a core is non-finite.

    Returns
    -------
    continuation : np.ndarray
        Values and gradients with shape (K, d + 1).
    """
    return continuation  # noqa: F821
```

### Step 3

form_bsde_targets

Goal
----
A right-endpoint discretization of the deterministic BSDE integral defines the regression target that is later fit against the left-endpoint value-plus-gradient operator at each time step. y_{n+1} = V_{n+1}(x_{n+1}) - h(u, grad V_{n+1}) dt. For affine drift f, h is tr(F) + 0.5 ||s * grad V||^2 + u dot (s * grad V), where s contains the diagonal diffusion entries and u is evaluated at the advanced state.

Inputs
------
continuation: Array of value and gradient data with shape (K, d + 1).
next_states: Advanced states with shape (K, d).
policy_matrix: Affine policy matrix with shape (d, d).
policy_bias: Affine policy bias with shape (d,).
drift_trace: Divergence of the affine drift.
sigma_diag: Positive diffusion diagonal with shape (d,).
dt: Positive time increment.

Returns
-------
targets: Float array of shape (K,).

```python
def form_bsde_targets(
    continuation: np.ndarray,
    next_states: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    drift_trace: float,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Form right-endpoint nonlinear BSDE regression targets.

    Parameters
    ----------
    continuation : np.ndarray
        Value in column zero and gradient in the remaining columns.
    next_states : np.ndarray
        Advanced states with shape (K, d).
    policy_matrix : np.ndarray
        Matrix of the affine policy evaluated at `next_states`.
    policy_bias : np.ndarray
        Bias of the affine policy.
    drift_trace : float
        Divergence of the affine drift.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If shapes are incompatible, an input is non-finite, `sigma_diag` is not
        positive, or `dt` is not finite and positive.

    Returns
    -------
    targets : np.ndarray
        BSDE targets with shape (K,).
    """
    return targets  # noqa: F821
```

### Step 4

build_weighted_features

Goal
----
The explicit BSDE regression operator is the identity plus a noise-weighted gradient, so its feature construction must represent both the pointwise value and the coordinate-specific derivative contribution that the operator mixes in. The feature construction uses one all-value channel and one channel per coordinate. In derivative channel i, every coordinate keeps phi(z) = [1, z, z^2] except coordinate i, which uses Sigma_i phi'(z), where Sigma_i = s_i xi_i sqrt(dt).

Inputs
------
points: Left-endpoint states with shape (K, d).
noises: Fixed innovations with shape (K, d).
sigma_diag: Positive diffusion diagonal with shape (d,).
dt: Positive time increment.

Returns
-------
weighted_features: Float array of shape (d + 1, K, d, 3).

```python
def build_weighted_features(
    points: np.ndarray,
    noises: np.ndarray,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build value and noise-weighted derivative feature channels.

    Parameters
    ----------
    points : np.ndarray
        Left-endpoint states with shape (K, d).
    noises : np.ndarray
        Standard-normal innovations with the same shape as `points`.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or non-finite entry, if
        `sigma_diag` is not positive, or if `dt` is not finite and positive.

    Returns
    -------
    weighted_features : np.ndarray
        Channel tensor with shape (d + 1, K, d, 3).
    """
    return weighted_features  # noqa: F821
```

### Step 5

assemble_local_system

Goal
----
Fixing all tensor-train cores except core j makes the explicit BSDE loss linear in that core, turning the nonlinear tensor-train fit into a sequence of local linear regressions solved one core at a time in an alternating least-squares sweep. The feature tensor contains one value channel followed by q complete coordinate-ordered derivative blocks, for 1 + qd channels. Contracting the fixed cores to the left and right gives channel-specific environment vectors. Their outer product with the local feature vector is summed over every block and vectorized in C order to yield one row of the local regression matrix.

Inputs
------
weighted_features: Array of shape (1 + qd, K, d, m) for a positive integer q.
cores: Sequence of d compatible order-three TT cores.
core_index: Zero-based position of the core being optimized.

Returns
-------
design_matrix: Float array of shape (K, r_{j-1} m r_j).

```python
def assemble_local_system(
    weighted_features: np.ndarray,
    cores: list[np.ndarray],
    core_index: int,
) -> np.ndarray:
    """Assemble the derivative-aware local BSDE regression matrix.

    Parameters
    ----------
    weighted_features : np.ndarray
        One value channel followed by q positive coordinate-ordered derivative
        blocks, with shape (1 + q*d, K, d, m).
    cores : list[np.ndarray]
        Compatible TT cores with basis mode m and exterior ranks one.
    core_index : int
        Zero-based index of the core treated as the local unknown.

    Raises
    ------
    ValueError
        If the feature tensor is malformed, does not contain a whole positive
        number of derivative blocks, or is non-finite, if the cores are not a
        finite compatible tensor train, or if `core_index` is outside [0, d).

    Returns
    -------
    design_matrix : np.ndarray
        Local matrix with shape (K, r_{j-1} m r_j).
    """
    return design_matrix  # noqa: F821
```

### Step 6

solve_ridge_core

Goal
----
An ALS micro-step minimizes a regularized linear least-squares objective while every other tensor-train core is fixed, turning one sweep of the tensor-train fit into an ordinary ridge regression. With design matrix A, target y, and positive regularization tau, the local coefficient vector satisfies (A^T A + tau I)c = A^T y. C-order reshaping restores the three-mode core used by the tensor train.

Inputs
------
design_matrix: Float array of shape (K, p).
targets: Float array of shape (K,).
tau: Positive ridge magnitude.
core_shape: Three positive integers with product p.

Returns
-------
updated_core: Float array with shape core_shape.

```python
def solve_ridge_core(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    tau: float,
    core_shape: tuple[int, int, int],
) -> np.ndarray:
    """Solve one regularized ALS core update.

    Parameters
    ----------
    design_matrix : np.ndarray
        Local regression matrix with shape (K, p).
    targets : np.ndarray
        Regression targets with shape (K,).
    tau : float
        Finite positive ridge magnitude.
    core_shape : tuple[int, int, int]
        Positive core dimensions whose product is p.

    Raises
    ------
    ValueError
        If matrix and target shapes disagree, an input is non-finite, `tau` is
        not positive, or `core_shape` does not contain three positive integers
        with product equal to the number of matrix columns.

    Returns
    -------
    updated_core : np.ndarray
        Updated tensor-train core with shape `core_shape`.
    """
    return updated_core  # noqa: F821
```

### Step 7

advance_adaptive_sweep

Goal
----
After a local ALS update, adapting the regularization to the current residual keeps the ridge penalty scaled to the fit, and shifting the optimized core through the tensor train with a gauge-fixed SVD lets the next local system be solved for the neighboring core while preserving the value-function norm as the active core's Frobenius norm. The regularization magnitude is reset to gamma times the squared residual divided by the squared Frobenius norm of the updated core. Shifting the core one position to the right uses a reduced singular value decomposition: the left singular vectors remain in the middle TT component, while the singular values and right singular vectors are absorbed into the last component. The bond gauge is right anchored: absorbed-right rows are ordered by decreasing squared norm with stable original-index ties, then each row is oriented by its first largest-magnitude coefficient and the same sign is applied to the matching left singular vector. A zero absorbed row falls back to the corresponding left-vector pivot. The next local system sums one value channel and every complete derivative-channel block before solving with the adapted regularization.

Inputs
------
weighted_features: Three-coordinate BSDE feature channels of shape (1 + 3q, K, 3, m).
middle_design: Local matrix used for the middle-core update.
targets: BSDE target vector.
left_core: Left-orthonormal first TT core.
updated_middle_core: Result of the preceding local solve.
right_core: Right TT component before the shift.
gamma: Positive relative regularization weight.

Returns
-------
adaptive_state: One-dimensional array containing tau_next, the shifted middle core, and the updated last core in C order.

```python
def advance_adaptive_sweep(
    weighted_features: np.ndarray,
    middle_design: np.ndarray,
    targets: np.ndarray,
    left_core: np.ndarray,
    updated_middle_core: np.ndarray,
    right_core: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Adapt regularization, shift the TT core, and update the last core.

    Parameters
    ----------
    weighted_features : np.ndarray
        One value channel followed by q positive three-coordinate derivative
        blocks, with shape (1 + 3*q, K, 3, m).
    middle_design : np.ndarray
        Local matrix from the preceding middle-core micro-step.
    targets : np.ndarray
        BSDE target vector with one value per sample.
    left_core : np.ndarray
        First TT core with shape (1, m, r_1).
    updated_middle_core : np.ndarray
        Updated middle core with shape (r_1, m, r_2).
    right_core : np.ndarray
        Last TT core with shape (r_2, m, 1).
    gamma : float
        Finite positive relative regularization weight.

    Raises
    ------
    ValueError
        If features, matrix, targets, or cores have incompatible shapes or
        non-finite values, if the TT ranks cannot be preserved by the SVD shift,
        if `gamma` is not positive, or if the adaptive magnitude is not positive.

    Returns
    -------
    adaptive_state : np.ndarray
        Packed tau, right-anchored shifted middle core, and updated last core
        as float64. The bond gauge is right anchored: absorbed-right rows are
        ordered by decreasing squared norm (stable ties), each row is oriented
        so its first largest-magnitude coefficient is positive, and the same
        sign is applied to the matching left singular vector. A zero right row
        uses the corresponding left-vector pivot.
    """
    return adaptive_state  # noqa: F821
```

### Step 8

recover_feedback_control

Goal
----
The value-function gradient determines the reverse-diffusion feedback for diagonal diffusion, connecting the fitted tensor-train continuation value back to the sampler's drift correction. u*(x) = -s * grad V(x). The adaptive sweep state stores the final two TT components after the middle-to-right core shift. The shared basis mode m is inferred from the first core, and each coordinate uses monomials [1, z, ..., z^(m-1)]. Reconstructing the packed components and replacing one basis vector at a time by its derivative gives all three gradient components at the query state.

Inputs
------
query: Three-factor state with shape (3,).
left_core: First TT core with shape (1, m, r_1), with m at least two.
adaptive_state: Packed tau, shifted middle core, and updated last core.
sigma_diag: Positive diffusion diagonal with shape (3,).

Returns
-------
feedback: Float array of shape (3,).

```python
def recover_feedback_control(
    query: np.ndarray,
    left_core: np.ndarray,
    adaptive_state: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """Recover the three-factor monomial-basis feedback from the adapted TT.

    Parameters
    ----------
    query : np.ndarray
        Evaluation state with shape (3,).
    left_core : np.ndarray
        First TT core with shape (1, m, r_1), with m at least two. All three
        coordinates use monomials of degrees 0 through m-1.
    adaptive_state : np.ndarray
        Packed tau, shifted middle core, and updated last core.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.

    Raises
    ------
    ValueError
        If an input is non-finite, if array shapes or the packed length are
        incompatible with a three-core shared-monomial-basis TT, if the basis
        mode is less than two, if the packed rank is not positive, or if
        `sigma_diag` or the stored tau is not positive.

    Returns
    -------
    feedback : np.ndarray
        Gradient feedback vector with shape (3,).
    """
    return feedback  # noqa: F821
```

### Step 9

run_full_pipeline

Goal
----
The complete one-interval regression chains every earlier step into one deterministic backward BSDE update: advancing the controlled states, evaluating the continuation tensor train, forming right-endpoint targets, building derivative-aware features, and performing two ALS micro-steps connected by the adaptive core shift. Advances controlled factor states, evaluates the continuation tensor train, forms right-endpoint BSDE targets, and builds the derivative-aware left-endpoint features. A middle-core ridge update is followed by residual-scaled regularization, an orthogonal core shift, and a last-core update. The resulting value gradient gives the feedback control, whose first component is the reported scalar.

Inputs
------
states, noises: Arrays of shape (K, 3).
drift_matrix, drift_bias: Affine drift parameters.
sigma_diag: Positive diagonal diffusion entries.
policy_matrix, policy_bias: Affine trajectory-policy parameters.
next_cores: Three compatible continuation TT cores with basis mode three.
initial_cores: Three compatible current-time TT cores with an orthonormal first and last component.
dt, tau_initial, gamma: Positive scalar parameters.
query: Three-factor evaluation state.

Returns
-------
feedback_component: Native float, the first component of the adapted feedback at query.

```python
def run_full_pipeline(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    next_cores: list[np.ndarray],
    initial_cores: list[np.ndarray],
    dt: float,
    tau_initial: float,
    gamma: float,
    query: np.ndarray,
) -> float:
    """Run the adaptive tensor-train BSDE regression pipeline.

    Parameters
    ----------
    states : np.ndarray
        Left-endpoint states with shape (K, 3).
    noises : np.ndarray
        Fixed innovations with shape (K, 3).
    drift_matrix : np.ndarray
        Affine drift matrix with shape (3, 3).
    drift_bias : np.ndarray
        Affine drift bias with shape (3,).
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    policy_matrix : np.ndarray
        Affine trajectory-policy matrix with shape (3, 3).
    policy_bias : np.ndarray
        Affine trajectory-policy bias with shape (3,).
    next_cores : list[np.ndarray]
        Three compatible continuation TT cores with basis mode three.
    initial_cores : list[np.ndarray]
        Three compatible current-time TT cores with orthonormal exterior components.
    dt : float
        Finite positive time increment.
    tau_initial : float
        Finite positive initial ridge magnitude.
    gamma : float
        Finite positive relative regularization weight.
    query : np.ndarray
        Three-factor state at which the feedback is evaluated.

    Raises
    ------
    ValueError
        If the state dimension is not three, any array shape or TT rank is
        incompatible, an input is non-finite, an exterior current-time core is
        not orthonormal, or `dt`, `tau_initial`, or `gamma` is not positive.

    Returns
    -------
    feedback_component : float
        First adapted feedback component at `query` as a native float.
    """
    return feedback_component  # noqa: F821
```
