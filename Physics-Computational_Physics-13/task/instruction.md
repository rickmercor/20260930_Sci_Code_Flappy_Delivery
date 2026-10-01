# Physics-Computational_Physics-13

## Background

Simulating a machine a robot arm, a vehicle suspension, a wind-turbine drivetrain, the human musculoskeletal system means integrating the motion of many bodies tied to one another by joints. Written with Lagrange multipliers enforcing those joints, the equations of motion form a differential-algebraic system of index three, and every implicit time step requires Newton iterations whose linear systems have a saddle-point structure: a mechanical block holding inertia and elasticity, the constraint Jacobian above and below it, and a structural zero in the corner where the multipliers meet themselves. That zero is what makes the matrix indefinite rather than positive definite, and it is why the multiplier unknowns cannot be preconditioned directly. Realistic models reach a million degrees of freedom, and because the inertial term scales as the inverse square of the time step, condition numbers of ten to the eighth and beyond are routine at the step sizes engineers actually use.

Neither established family of solvers copes well with this. Sparse direct factorisation is robust but suffers fill-in, costing memory that grows faster than linearly and time that grows faster still, which puts million-degree-of-freedom transient simulations out of reach on a single machine. Krylov methods have the right memory profile but converge uselessly slowly on a matrix this ill-conditioned unless a preconditioner clusters the spectrum, and designing one for an indefinite saddle-point problem is hard. The natural target, the Schur complement of the mechanical block, is dense, because inverting that block couples any two constraints that share a body and, through chains of such couplings, effectively every constraint to every other.

The line of work this task belongs to attacks the two blocks with different tools and glues them together. The mechanical block is handled by algebraic multigrid, whose V-cycle costs work proportional to the number of unknowns and whose approximation quality, unlike that of an incomplete factorisation, does not degrade as the discretisation is refined. The constraint block is handled by exploiting a fact that is structural rather than numerical: the sparsity of the constraint Jacobian is a picture of the mechanism itself, one row per constraint nonzero only in the columns of the two bodies its joint connects, so the graph of the machine can be read straight out of the matrix. Clustering constraints that share a body into small groups yields a block-diagonal surrogate for the dense Schur complement whose blocks are a few entries across, and the couplings that clustering necessarily misses, the ones running around closed kinematic loops, are put back through a dense correction whose rank is the number of independent loops and which therefore stays small however large the machine grows. Over-constrained models, common in practice and fatal to a Schur-based method because they make the exact complement singular, are absorbed by a small Tikhonov shift that perturbs only the Newton search direction and vanishes at convergence.

What raises this above engineering is that the combination admits an analysis. Under an idealisation in which the constraint block of the preconditioner is the exact contraction of the multigrid operator with the Jacobian, the eigenvalues of the preconditioned system obey a scalar quadratic relation, and their distance from unity is bounded by a closed-form expression in a single quantity, namely how well the V-cycle approximates the inverse of the mechanical block. Nothing about the problem size, the number of constraints or the conditioning of the mechanical block appears in that bound, which is the formal content of the claim that such a solver converges in a number of iterations independent of how large the model is. The practical preconditioner, however, is not the idealised one, since clustering, loop correction and regularisation are all perturbations the derivation never sees. The gap between what the theory bounds and what the spectrum actually does is therefore the quantity that decides which component a practitioner should improve first, and whether size-independence is a property of the method or only of its idealisation.

## Problem

Implicit integration of an index-3 constrained multibody system produces at every Newton step a symmetric indefinite saddle-point system whose mechanical block is severely ill-conditioned and whose constraint block is structurally zero. A hybrid preconditioner for such systems replaces the inverse of the mechanical block by a single algebraic-multigrid V-cycle, and the dense constraint Schur complement by a topology-clustered block-diagonal surrogate that is repaired by a dense correction over a feedback edge set of the kinematic graph and then Tikhonov-shifted against constraint redundancy. The spectral theory of that framework bounds the distance from unity of every nonzero eigenvalue of the preconditioned operator by a closed-form function of one scalar and of nothing else: the sharpest constant $\gamma$ for which the operator the V-cycle inverts, $\tilde K$, is spectrally equivalent to the mechanical block $K$ in the two-sided sense $(1-\gamma)\tilde K \preceq K \preceq (1+\gamma)\tilde K$. The derivation assumes the constraint block of the preconditioner is the exact contraction of the V-cycle with the constraint Jacobian, which is precisely what the clustering, the loop correction and the shift give up, so the predictive value of the bound for the method as implemented is an empirical question rather than a theorem. For the testbed below, compute the observed worst-case deviation of the preconditioned spectrum from unity divided by the bound that theory predicts for it.

The testbed is a mechanism of $N_b = 24$ rigid bodies with six generalised coordinates each, indexed so that body $b$ occupies coordinates $6b$ through $6b+5$:

- Joints are a closed chain $\{b,\,b+1 \bmod N_b\}$ for $b = 0,\ldots,N_b-1$, followed by $N_c = 4$ chords, chord $c$ joining body $i_c = c\lfloor N_b/N_c\rfloor \bmod N_b$ to body $(i_c + \lfloor N_b/3\rfloor + c) \bmod N_b$; each joint is an unordered body pair, self-loops and pairs already present are discarded, and the surviving joints keep this order, giving $n_E$ joints.
- The mechanical block is $K = M/(\beta h^2) + K_{\mathrm{el}}$ with $h = 10^{-3}$ s and $\beta = 0.25$. The mass $M$ is diagonal, body $b$ contributing $(m_b, m_b, m_b, 0.100\,m_b, 0.125\,m_b, 0.150\,m_b)$ with $m_b = 1 + 0.5\,(b \bmod 5)$. Joint $k$ joining bodies $i$ and $j$ contributes to $K_{\mathrm{el}}$, for each of the six coordinate offsets $d$, a scalar graph-Laplacian block of weight $k_k = k_0\,10^{\,3\cos(2\pi k/n_E)}$ with $k_0 = 10^{8}$ N m$^{-1}$, acting on the coordinate pair $(6i+d,\,6j+d)$.
- Each joint is a spherical joint carrying three constraints, so the Jacobian has $3n_E$ rows ordered by joint and then by $d = 0,1,2$. With $\varphi_k = 2\pi(k+1)/n_E$, row $3k+d$ of joint $k$ has entries $1$ at column $6i+d$, $-1$ at column $6j+d$, $\sin(\varphi_k + d)$ at column $6i+3+((d+1)\bmod 3)$ and $-\cos(\varphi_k + d)$ at column $6j+3+((d+2)\bmod 3)$.

The multigrid component is smoothed aggregation with prolongator damping $\omega = 2/3$, Galerkin coarse operators, one pre- and one post-smoothing sweep per level with the framework's symmetric smoother taken block-wise per body on the finest level and scalar on all coarser ones, coarsening while the level order exceeds 24 and at most four times, and an exact solve at the coarsest level. Entry $(i,j)$ of a level operator is a strong connection when $\lvert K_{ij}\rvert \ge \theta\max_{k\neq i}\lvert K_{ik}\rvert$ with $\theta = 0.25$, and this directed relation is symmetrised by logical or, so $i$ and $j$ are tied if either direction is strong. Aggregates are then grown by one greedy pass over the unknowns in ascending index: the first unassigned unknown encountered opens an aggregate and absorbs every unknown still unassigned that is tied to it, unknowns already assigned are never reassigned, and an unknown with no tie forms a singleton. Constraint clusters are grown by the same rule on the constraint adjacency graph, in which two constraints are tied when they share a body, except that a seed stops absorbing once its cluster reaches the framework's maximum cluster size; the cut constraints are held out of that pass and clustered among themselves afterwards, the feedback edge set being the joints closing a cycle during one union-find pass in joint order. The dense loop correction is added to, not substituted for, the cut constraints' own block-diagonal entries, and the framework's default relative Tikhonov coefficient is then applied to the assembled matrix. Your final answer must be a single number: the ratio of the observed worst-case deviation to its theoretical bound for this configuration.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_assemble_effective_stiffness

Goal
----
Assemble the effective stiffness matrix, the (1,1) block of the saddle-point system produced by the Newton-Raphson linearisation of the index-3 multibody equations of motion.

```python
import numpy as np

def assemble_effective_stiffness(n_bodies: int, n_chords: int, h: float,
                                 beta: float, k_scale: float) -> np.ndarray:
    """Assemble the effective stiffness matrix of the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).
    h : float
        Integration time step in seconds (h > 0).
    beta : float
        Newmark parameter of the implicit integrator (0 < beta <= 1).
    k_scale : float
        Reference joint stiffness in N/m (k_scale > 0).

    Returns
    -------
    stiffness : np.ndarray
        Symmetric positive-definite matrix of shape (6 * n_bodies,
        6 * n_bodies) holding the inertial and elastic contributions to the
        tangent operator.
    """
    return stiffness  # placeholder
```

### Step 2

02_assemble_constraint_jacobian

Goal
----
Assemble the constraint Jacobian, the off-diagonal block of the saddle-point system, whose block-sparse pattern mirrors the topology of the mechanism.

```python
import numpy as np

def assemble_constraint_jacobian(n_bodies: int, n_chords: int) -> np.ndarray:
    """Assemble the constraint Jacobian of the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).

    Returns
    -------
    jacobian : np.ndarray
        Full-row-rank matrix of shape (3 * n_edges, 6 * n_bodies), where
        n_edges is the number of joints of the kinematic graph. Rows are
        ordered by joint and then by the three scalar constraints of that
        joint.
    """
    return jacobian  # placeholder
```

### Step 3

03_build_strength_aggregates

Goal
----
Build the strength-of-connection graph of a level operator and partition its unknowns into aggregates by a deterministic greedy sweep.

```python
import numpy as np

def build_strength_aggregates(matrix: np.ndarray, theta: float) -> np.ndarray:
    """Partition the unknowns of a level operator into aggregates.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric level operator of shape (n, n) with n >= 1.
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.

    Returns
    -------
    aggregates : np.ndarray
        Integer array of shape (n,) giving the aggregate index of every
        unknown. Aggregate indices are consecutive and start at zero, in the
        order the aggregates are created.
    """
    return aggregates  # placeholder
```

### Step 4

04_build_smoothed_prolongator

Goal
----
Turn an aggregation into a transfer operator by applying one damped Jacobi smoothing step to the piecewise-constant tentative prolongator.

```python
import numpy as np

def build_smoothed_prolongator(matrix: np.ndarray, aggregates: np.ndarray,
                               omega: float) -> np.ndarray:
    """Build the smoothed-aggregation prolongator of a level.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric level operator of shape (n, n) with a nonzero diagonal.
    aggregates : np.ndarray
        Integer array of shape (n,) holding consecutive aggregate indices
        starting at zero.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.

    Returns
    -------
    prolongator : np.ndarray
        Transfer operator of shape (n, n_coarse), where n_coarse is the number
        of aggregates.
    """
    return prolongator  # placeholder
```

### Step 5

05_assemble_vcycle_operator

Goal
----
Assemble the dense matrix representing one multigrid V-cycle applied to the effective stiffness, which is the approximate inverse used inside the preconditioner.

```python
import numpy as np

def assemble_vcycle_operator(stiffness: np.ndarray, aggregates: np.ndarray,
                             prolongator: np.ndarray, theta: float, omega: float,
                             block_size: int, n_min: int,
                             max_levels: int) -> np.ndarray:
    """Assemble the dense operator of one multigrid V-cycle.

    Parameters
    ----------
    stiffness : np.ndarray
        Symmetric positive-definite operator of shape (n, n).
    aggregates : np.ndarray
        Finest-level aggregation of shape (n,), as produced by the aggregation
        step, holding consecutive aggregate indices starting at zero.
    prolongator : np.ndarray
        Finest-level transfer operator of shape (n, n_coarse), as produced by
        the prolongator step from the same aggregation.
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.
    block_size : int
        Size of the smoother blocks on the finest level (block_size >= 1);
        coarser levels always use scalar blocks.
    n_min : int
        Level order at or below which the operator is inverted exactly
        (n_min >= 1).
    max_levels : int
        Maximum number of coarsening steps before an exact solve
        (max_levels >= 0).

    Returns
    -------
    vcycle_operator : np.ndarray
        Matrix of shape (n, n) whose action on a vector equals one V-cycle
        applied to that vector as right-hand side with a zero initial guess.
    """
    return vcycle_operator  # placeholder
```

### Step 6

06_compute_spectral_equivalence

Goal
----
Measure the sharpest spectral-equivalence constant between the multigrid V-cycle operator and the effective stiffness it approximates.

```python
import numpy as np


def compute_spectral_equivalence(stiffness: np.ndarray,
                                 vcycle_operator: np.ndarray) -> float:
    """Measure the sharpest spectral-equivalence constant of the V-cycle.

    Parameters
    ----------
    stiffness : np.ndarray
        Symmetric positive-definite operator of shape (n, n).
    vcycle_operator : np.ndarray
        Symmetric approximate inverse of the stiffness, shape (n, n).

    Returns
    -------
    gamma : float
        Largest deviation from unity over the spectrum of the V-cycle operator
        composed with the stiffness, as a native Python float.
    """
    return gamma  # placeholder
```

### Step 7

07_partition_constraint_graph

Goal
----
Recover the kinematic graph from the constraint Jacobian, cut its cycles, and group the constraints into small clusters for the block-diagonal Schur approximation.

```python
import numpy as np

def partition_constraint_graph(jacobian: np.ndarray, dofs_per_body: int,
                               max_cluster: int) -> np.ndarray:
    """Cluster the constraints and mark those on a feedback edge set.

    Parameters
    ----------
    jacobian : np.ndarray
        Constraint Jacobian of shape (m, n); every row must be nonzero in the
        columns of exactly two bodies.
    dofs_per_body : int
        Number of generalised coordinates per body (dofs_per_body >= 1); the
        column count must be an exact multiple of it.
    max_cluster : int
        Maximum number of constraints per cluster (max_cluster >= 1).

    Returns
    -------
    partition : np.ndarray
        Integer array of shape (m, 2). Column zero holds the cluster index of
        every constraint, consecutively numbered from zero; column one is one
        for constraints lying on a cut joint and zero otherwise.
    """
    return partition  # placeholder
```

### Step 8

08_assemble_schur_preconditioner

Goal
----
Assemble the block-diagonal Schur complement approximation, add the dense low-rank correction on the cut constraints, and regularise it against constraint redundancy.

```python
import numpy as np

def assemble_schur_preconditioner(jacobian: np.ndarray, vcycle_operator: np.ndarray,
                                  partition: np.ndarray,
                                  eps_rel: float) -> np.ndarray:
    """Assemble the regularised Schur complement approximation.

    Parameters
    ----------
    jacobian : np.ndarray
        Constraint Jacobian of shape (m, n).
    vcycle_operator : np.ndarray
        V-cycle approximate inverse of the mechanical block, shape (n, n).
    partition : np.ndarray
        Integer array of shape (m, 2) holding cluster indices in column zero
        and cut flags in column one.
    eps_rel : float
        Tikhonov shift relative to the matrix one-norm, eps_rel >= 0.

    Returns
    -------
    schur_prec : np.ndarray
        Symmetric matrix of shape (m, m): block diagonal over the clusters,
        plus the dense correction on the cut constraints, plus the shift.
    """
    return schur_prec  # placeholder
```

### Step 9

09_compute_spectral_deviation

Goal
----
Form the block-triangular preconditioned saddle-point operator and measure how far its nonzero spectrum spreads away from unity.

```python
import numpy as np

def compute_spectral_deviation(stiffness: np.ndarray, jacobian: np.ndarray,
                               vcycle_operator: np.ndarray,
                               schur_prec: np.ndarray) -> float:
    """Measure the worst deviation from unity of the preconditioned spectrum.

    Parameters
    ----------
    stiffness : np.ndarray
        Mechanical block of the saddle-point system, shape (n, n).
    jacobian : np.ndarray
        Constraint Jacobian, shape (m, n).
    vcycle_operator : np.ndarray
        V-cycle approximate inverse of the mechanical block, shape (n, n).
    schur_prec : np.ndarray
        Nonsingular Schur complement approximation, shape (m, m).

    Returns
    -------
    deviation : float
        Largest modulus of one minus an eigenvalue, over the eigenvalues of
        the preconditioned operator that are not numerically zero, as a native
        Python float.
    """
    return deviation  # placeholder
```

### Step 10

10_compute_bound_sharpness

Goal
----
Evaluate the theoretical eigenvalue bound implied by the spectral-equivalence constant and divide the observed spectral deviation by it.

```python
import numpy as np

def compute_bound_sharpness(gamma: float, deviation: float) -> float:
    """Divide the observed spectral deviation by its theoretical bound.

    Parameters
    ----------
    gamma : float
        Spectral-equivalence constant of the V-cycle, 0 <= gamma < 1.
    deviation : float
        Observed largest deviation from unity of the preconditioned spectrum
        (deviation >= 0).

    Returns
    -------
    sharpness : float
        Observed deviation divided by the theoretical bound implied by gamma,
        as a native Python float.
    """
    return sharpness  # placeholder
```

### Step 11

11_run_bound_sharpness_pipeline

Goal
----
Chain the sub-problem functions 01-10 end to end on the multibody testbed and return the bound sharpness of the hybrid preconditioner.

```python
import numpy as np

def run_bound_sharpness_pipeline(n_bodies: int = 24, n_chords: int = 4,
                                 h: float = 1.0e-3, beta: float = 0.25,
                                 k_scale: float = 1.0e8, theta: float = 0.25,
                                 omega: float = 0.6666666666666666,
                                 block_size: int = 6, n_min: int = 24,
                                 max_levels: int = 4, max_cluster: int = 6,
                                 eps_rel: float = 1.0e-8) -> float:
    """Run the full bound-sharpness measurement on the multibody testbed.

    Parameters
    ----------
    n_bodies : int
        Number of rigid bodies in the mechanism (n_bodies >= 4).
    n_chords : int
        Number of long-range chords added to the closed chain (n_chords >= 0).
    h : float
        Integration time step in seconds (h > 0).
    beta : float
        Newmark parameter of the implicit integrator (0 < beta <= 1).
    k_scale : float
        Reference joint stiffness in N/m (k_scale > 0).
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.
    omega : float
        Damping factor of the prolongator smoothing sweep, 0 < omega <= 1.
    block_size : int
        Size of the smoother blocks on the finest level (block_size >= 1).
    n_min : int
        Level order at or below which the operator is inverted exactly
        (n_min >= 1).
    max_levels : int
        Maximum number of coarsening steps before an exact solve
        (max_levels >= 0).
    max_cluster : int
        Maximum number of constraints per cluster (max_cluster >= 1).
    eps_rel : float
        Tikhonov shift relative to the matrix one-norm, eps_rel >= 0.

    Returns
    -------
    sharpness : float
        Observed spectral deviation of the preconditioned saddle-point
        operator divided by its theoretical bound, as a native Python float.
    """
    return sharpness  # placeholder
```
