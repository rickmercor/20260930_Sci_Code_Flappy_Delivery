# Physics-Quantum_Information_Computing-31

## Background

Quantum state tomography is the task of reconstructing an unknown density matrix, the Hermitian, positive semidefinite, trace-one operator describing a quantum system's state, from measurement data collected on many copies of that state. For a $d$-level system, a general density matrix has $d^2-1$ real free parameters, but physically relevant states are frequently low rank (pure states have rank one, and many noisy but weakly mixed states have only a few dominant eigenvalues), so estimators that exploit a known or assumed rank bound can use far fewer measurement resources than full tomography would require.

Measurement resources for tomography come in several forms. One axis is how measurements are physically implemented: individual copies of the state can be measured one at a time, or, when the experimental apparatus allows it, several copies can be measured jointly with a single collective measurement acting on their tensor product. Joint measurements can extract more information per copy than individual measurements, but require coherent storage and manipulation of multiple copies at once, which is often the harder experimental resource to scale. A second axis is which specific measurement settings are used to probe the state and how many repetitions (shots) are taken per setting, which determines the statistical noise in the resulting frequency estimates.

Because any real estimator is built from a finite, noisy measurement record, a central theoretical question is how to certify how close the estimator's output is to the true state, and under what conditions on the measurement design that certificate holds. In the broader theory of low-rank matrix recovery, from which many quantum tomography algorithms borrow ideas, such certificates are typically stated in terms of how well the measurement process preserves distances (an isometry property) when restricted to the relevant class of low-rank matrices. A measurement design that satisfies a strong, uniform isometry property over all low-rank matrices gives a very robust guarantee, but such a property can be difficult or expensive to achieve; a substantial line of recent work studies weaker isometry conditions, restricted to more specific, lower-dimensional sets tied to the particular unknown matrix being recovered, that are easier to satisfy while still yielding a rigorous, quantitative recovery guarantee for a broad class of recovery algorithms based on locating a suitable stationary point of an optimization objective.

Recovery algorithms for low-rank matrices, including density matrices, often work directly with a low-rank factorization of the unknown object rather than with the full matrix, since this automatically enforces the rank constraint and low-rank positivity and reduces the number of optimization variables. Fitting such a factorized model to noisy measurement data is generally a nonconvex optimization problem, and different papers propose different first-order or specialized update rules for finding a good factor, with varying degrees of tuning required and varying convergence behavior.

## Problem

Study a deterministic low-rank quantum-state reconstruction experiment in which a recent landscape theorem sets the regularization strength through a conditional Frobenius-error budget: compute the final empirical error as a fraction of that conditional bound. The relevant theorem treats unconstrained factorized nuclear-norm-regularized least squares, assumes a lower isometry on admissible recovery errors and an upper isometry only on the true rank stratum's tangent space, and gives an explicit bound for every exact second-order critical point under a rank-independent isometry-ratio threshold. This is a synthetic diagnostic with the target and realized noise available for setting the budget; the finite numerical iterate is compared empirically with the theorem's conditional envelope.

Construct the target by `rng=np.random.default_rng(31)`, drawing `Araw=rng.standard_normal((4,4))+1j*rng.standard_normal((4,4))`, taking NumPy's unadjusted `Q,_=np.linalg.qr(Araw)`, and setting $\rho^\star=Q\operatorname{diag}(0.6,0.4,0,0)Q^\dagger$, Hermitian-symmetrized; write $Q_\perp$ for the last two columns of $Q$. For the acquisition warm start, use three size-four joint-measurement rounds with outcomes $(2,2,2)$, hence $K=6$ and $N=12$, draw `Z=(rng.standard_normal((2,6))+1j*rng.standard_normal((2,6)))/np.sqrt(2)` from a separate `np.random.default_rng(4100)`, and take the supplied out-of-support block $E=(ZZ^\dagger-6I_2)/12$, with the other error blocks zero by instance convention. Let $\hat\rho_1$ be the nearest rank-at-most-two, positive-semidefinite, trace-one matrix to $\overline Y=\rho^\star+Q_\perp E Q_\perp^\dagger$ in Frobenius norm, and define $e_1=\|\hat\rho_1-\rho^\star\|_F$.

Let $H_k$ be the orthonormal Hermitian basis consisting of four diagonal matrix units in increasing order, followed by $(E_{ij}+E_{ji})/\sqrt2$ and $\mathrm{i}(E_{ij}-E_{ji})/\sqrt2$ for lexicographically ordered $i<j$; define $A_k=\sqrt{w_k}QH_kQ^\dagger$ with squared weights
$$w=(1.00,1.08,9.00,6.00,1.12,1.20,1.04,1.10,1.16,1.06,1.14,1.18,1.02,1.09,7.00,8.00),$$
$\mathcal A(M)_k=\operatorname{Re}\operatorname{tr}(A_kM)$, and $y=\mathcal A(\rho^\star)+\xi$, where `xi=np.random.default_rng(2026).normal(scale=0.001,size=16)`. Derive the sharp lower constant $\alpha$ on differences $M-\rho^\star$ for rank-at-most-two PSD $M$, the sharp tangent-space upper constant $\beta_T$, and the full-Hermitian-space upper constant $\beta_{\rm full}$, and use the theorem's own hypotheses to decide which upper constant governs applicability. With true and search ranks both two and the realized noise norm $\eta=\|\mathcal A^*(\xi)\|_{\rm op}$, denote the source theorem's explicit bound by $B(\lambda)$; evaluate $B(0)$ and choose the largest $\lambda\ge0$ for which $B(\lambda)\le C$, where the error budget is $C=e_1/2$, accounting for both branches of the bound's positive-part term and justifying the maximizing choice.

For that chosen $\lambda$, minimize $f_\lambda(F)=\tfrac12\|\mathcal A(FF^\dagger)-y\|_2^2+\lambda\|F\|_F^2$ over unconstrained $F\in\mathbb C^{4\times2}$, using the real-coordinate Euclidean gradient and exactly 5000 Armijo steps with sufficient-decrease constant $10^{-4}$, trial step reset to $1$ each iteration and repeated halving; initialize $F_0$ from the top two eigenpairs of $\hat\rho_1$ by taking square roots of their nonnegative eigenvalues. Report the single dimensionless number $\|F_{5000}F_{5000}^\dagger-\rho^\star\|_F/B(\lambda)$, with reasoning giving the measurement geometry, budget-selected regularization and resulting error; no trace normalization is imposed on the recovered matrix, and the comparison does not assert second-order criticality of the finite iterate.

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

01_acquisition_error_block.py

Goal
----
Compute the part of a bounded-joint-measurement tomography estimator's averaging error that lies strictly outside the true state's support, for a fixed sequence of round outcomes.

```python
def pi_perp_error_block(Z: "np.ndarray", K: int, N: int, V_perp: "np.ndarray") -> "np.ndarray":
    """Embed the exact out-of-support averaging error of a bounded-joint-measurement
    tomography estimator into the full Hilbert space.

    Parameters
    ----------
    Z : np.ndarray
        (m, K) complex matrix, standard complex Gaussian in the coordinates of an
        orthonormal basis of the orthogonal complement of the true state's support
        (m is the dimension of that complement).
    K : int
        Sum of the per-round outcome values across all measurement rounds.
    N : int
        Total number of copies measured (rounds times copies per round).
    V_perp : np.ndarray
        (d, m) complex matrix whose columns are an orthonormal basis of the
        orthogonal complement of the true state's support, in the full d-dimensional space.

    Returns
    -------
    result : np.ndarray
        (d, d) complex Hermitian matrix, the error block embedded in the full space.

    Raises
    ------
    ValueError
        If Z's row count does not match V_perp's column count, or N <= 0.
    """
    return result
```

### Step 2

02_density_projection.py

Goal
----
Project a Hermitian matrix onto the set of rank-r-or-less density matrices (Hermitian, positive semidefinite, trace 1, rank at most r), in Frobenius norm.

```python
def project_rank_r_density_matrix(Ybar: "np.ndarray", r: int) -> "np.ndarray":
    """Find the closest rank-<=r density matrix to a given Hermitian matrix, in Frobenius norm.

    Parameters
    ----------
    Ybar : np.ndarray
        (d, d) Hermitian complex matrix to project.
    r : int
        Maximum allowed rank of the output, 1 <= r <= d.

    Returns
    -------
    result : np.ndarray
        (d, d) complex Hermitian matrix, positive semidefinite, trace 1, rank at most r,
        minimizing the Frobenius distance to Ybar among all such matrices.

    Raises
    ------
    ValueError
        If Ybar is not square, not Hermitian (within a small tolerance), or r is not in [1, d].
    """
    return result
```

### Step 3

03_regularized_backtrack_step.py

Goal
----
Advance the PSD factor for nuclear-norm-regularized sensing.

```python
def regularized_backtrack_step(F: "np.ndarray", A_ops: "np.ndarray", y: "np.ndarray", lam: float) -> "np.ndarray":
    """Take one real-coordinate gradient step on regularized factored sensing.

    Parameters
    ----------
    F : np.ndarray
        Complex factor of shape (d,r).
    A_ops : np.ndarray
        Hermitian measurement operators of shape (m,d,d).
    y : np.ndarray
        Real measurement vector of shape (m,).
    lam : float
        Nonnegative coefficient of the squared Frobenius norm of F in
        one-half squared measurement residual plus lam times that norm.

    Returns
    -------
    result : np.ndarray
        Unconstrained factor after one Armijo step. Use the real-coordinate
        Euclidean gradient on real and imaginary parts of F, sufficient-decrease
        constant 1e-4, initial trial step 1, and repeated halving. Return the first
        accepted trial; do not normalize F.

    Raises
    ------
    ValueError
        If lam is negative or A_ops and y have different lengths.
    FloatingPointError
        If the line-search step underflows before acceptance.
    """
    return result
```

### Step 4

04_regularized_recovery.py

Goal
----
Run regularized factored sensing from an acquisition estimate.

```python
def regularized_recovery(rho_init: "np.ndarray", r: int, A_ops: "np.ndarray", y: "np.ndarray", n_iters: int, lam: float) -> "np.ndarray":
    """Recover an unconstrained PSD matrix from a spectral factor warm start.

    Parameters
    ----------
    rho_init : np.ndarray
        Hermitian PSD initial matrix of shape (d,d). If truncation discards
        a positive eigenvalue, assume a strict gap at the retained boundary.
    r : int
        Number of largest eigenpairs in the initial factor, 1 <= r <= d.
        Clip numerical negative eigenvalues to zero before square roots.
    A_ops : np.ndarray
        Hermitian operators of shape (m,d,d).
    y : np.ndarray
        Observations of shape (m,), matching A_ops.
    n_iters : int
        Nonnegative number of regularized_backtrack_step updates.
    lam : float
        Nonnegative regularization coefficient for every update.

    Returns
    -------
    result : np.ndarray
        Recovered matrix F F-dagger, shape (d,d), after exactly n_iters updates.
        Eigenvector phases and unitary rotations within a fully retained
        eigenspace are immaterial.

    Raises
    ------
    ValueError
        If n_iters or lam is negative, r is outside [1,d], or measurement lengths differ.
    FloatingPointError
        If an update's line search underflows.
    """
    return result
```

### Step 5

05_adjoint_noise_norm.py

Goal
----
Compute the Hilbert-Schmidt adjoint of a linear measurement map applied to a given noise vector, and return its operator norm.

```python
def adjoint_operator_norm(xi: "np.ndarray", A_ops: "np.ndarray") -> float:
    """Compute the operator norm of the adjoint measurement map applied to a noise vector.

    Parameters
    ----------
    xi : np.ndarray
        (m,) real array, the measurement noise/residual associated with each operator.
    A_ops : np.ndarray
        (m, d, d) complex array, m Hermitian measurement operators.

    Returns
    -------
    result : float
        The operator norm of sum_i xi_i * A_ops[i], as a native Python float.

    Raises
    ------
    ValueError
        If xi and A_ops have mismatched lengths.
    """
    return 0.0
```

### Step 6

06_sensing_isometry_constants.py

Goal
----
Compute measurement geometry for the low-rank landscape theorem.

```python
def sensing_isometry_constants(A_ops: "np.ndarray", V_support: "np.ndarray") -> "np.ndarray":
    """Compute full-space lower and tangent-space upper squared-norm constants.

    Parameters
    ----------
    A_ops : np.ndarray
        Hermitian operators of shape (m,d,d), spanning the real Hermitian space.
    V_support : np.ndarray
        Orthonormal columns of shape (d,r_star), 1 <= r_star <= d, spanning
        the target support. The tangent space is that of rank-r_star Hermitian
        matrices at a PSD target with this support.

    Returns
    -------
    result : np.ndarray
        Real array [alpha_full,beta_tangent,beta_full], shape (3,).
        These are sharp extremal squared-norm ratios of the measurement map:
        minimum on all Hermitian matrices, maximum on the tangent space,
        and maximum on all Hermitian matrices. alpha_full always bounds
        restricted PSD recovery errors below; equality with the sharp
        restricted constant needs separate justification.
    """
    return result
```

### Step 7

07_conditional_landscape_bound.py

Goal
----
Evaluate the landscape theorem's conditional Frobenius-error bound expression for an exact second-order critical point, given the measurement design's isometry constants and the noise level.

```python
def certification_bound(op_norm: float, r_star: int, r: int, alpha: float, beta: float, lam: float) -> float:
    """Evaluate the conditional theorem bound on the recovery error of an exact
    second-order critical point.

    Parameters
    ----------
    op_norm : float
        The operator norm of the adjoint measurement map applied to the noise, ||A^*(xi)||_op.
    r_star : int
        The true rank of the target state.
    r : int
        The ansatz rank used by the recovery method, r >= r_star.
    alpha : float
        The restricted lower isometry constant of the measurement design.
    beta : float
        The tangent-space upper isometry constant of the measurement design, beta >= alpha > 0.
    lam : float
        The regularization weight used by the recovery method, lam >= 0.

    Returns
    -------
    result : float
        The conditional theorem-bound value on the Frobenius-norm recovery error, as a
        native Python float. This function evaluates the bound expression; it does not
        establish that a separately computed finite iterate is an exact critical point.

    Raises
    ------
    ValueError
        If alpha <= 0, beta < alpha, or the ratio beta/alpha does not satisfy the bound's
        applicability threshold.
    """
    return 0.0
```

### Step 8

08_regularization_from_budget.py

Goal
----
Invert the conditional landscape envelope to choose a regularization budget.

```python
def regularization_from_budget(op_norm: float, r_star: int, r: int, alpha: float, beta: float, budget: float) -> float:
    """Choose the largest regularization satisfying the conditional error budget.

    Parameters
    ----------
    op_norm : float
        Nonnegative realized adjoint-noise operator norm eta.
    r_star : int
        Positive true rank.
    r : int
        Search rank with r_star <= r <= 10*r_star. In this domain the source
        bound is strictly increasing with nonnegative regularization.
    alpha : float
        Positive lower squared-norm isometry constant.
    beta : float
        Tangent upper constant, at least alpha, satisfying the source threshold.
    budget : float
        Finite nonnegative upper limit on the conditional Frobenius-error bound.

    Returns
    -------
    result : float
        Largest lambda >= 0 for which the landscape bound is at most budget,
        as a native float. Both branches of its positive-part term are included.

    Raises
    ------
    ValueError
        If budget is below the zero-regularization bound, or the source
        isometry applicability conditions fail.
    """
    return result
```

### Step 9

09_recovery_bound_utilization.py

Goal
----
Run the complete acquisition-to-recovery bound-utilization pipeline.

```python
def recovery_bound_utilization(seed_rho: int, eigvals: "np.ndarray", seed_Z: int, J: "np.ndarray", N: int, weights: "np.ndarray", seed_noise: int, noise_scale: float, n_iters: int, budget_fraction: float) -> float:
    """Combine acquisition, regularized sensing, and conditional landscape analysis.

    Parameters
    ----------
    seed_rho : int
        Seed for a 4-by-4 complex standard-normal matrix (real array first,
        imaginary array second); use its unadjusted NumPy QR factor Q.
    eigvals : np.ndarray
        Two positive target eigenvalues summing to one; remaining eigenvalues zero.
    seed_Z : int
        Seed for a (2,sum(J)) standard complex Gaussian draw, real array first.
    J : np.ndarray
        Positive integer round outcomes. Acquisition uses their sum K.
    N : int
        Positive total acquisition sample count. Use the supplied out-of-support
        error (Z Z-dagger minus K times identity) divided by N, embedded with
        Q's last two columns; other error blocks vanish by task convention.
    weights : np.ndarray
        Sixteen positive squared measurement weights: four diagonals first,
        then real and positive-upper-imaginary off-diagonal Hermitian elements
        for lexicographic index pairs. Rotate each canonical element by Q and
        scale by the square root of its weight. The first weight must be minimal,
        making the full-space lower constant sharp on rank-2 PSD errors.
        The sharp tangent-to-lower ratio must satisfy the source threshold.
    seed_noise : int
        Seed for independent real normal noise, in measurement-operator order.
    noise_scale : float
        Nonnegative standard deviation of the sixteen noise entries.
    n_iters : int
        Nonnegative count of unconstrained regularized Armijo updates. Start
        with the top-two-eigenpair factor of the nearest rank-2 density matrix
        to the acquisition estimate.
    budget_fraction : float
        Positive fraction of the acquisition Frobenius error defining the
        conditional recovery-error budget. Choose the largest nonnegative
        regularization satisfying that budget. The budget must be feasible.
        At least one of noise_scale and the resulting regularization is positive.

    Returns
    -------
    result : float
        Actual recovered-matrix Frobenius error divided by the landscape theorem's
        conditional bound, as a finite native float. Values above one are allowed:
        a finite iterate need not be an exact second-order critical point.
        All randomness uses separate generators.

    Raises
    ------
    ValueError
        If the error budget is infeasible, the selected bound is zero, or
        the isometry constants violate the bound applicability condition.
    FloatingPointError
        If a recovery line search underflows.
    """
    return result
```
