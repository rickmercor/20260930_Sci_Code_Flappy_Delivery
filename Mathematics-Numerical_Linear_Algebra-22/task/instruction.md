# Mathematics-Numerical_Linear_Algebra-22

## Background

Computing $\exp(-tA)b$ for a large symmetric positive semidefinite matrix over an entire time interval is a core primitive in evolution simulation, arising in geophysical electromagnetic modeling, chemical kinetics, network analysis, and nonlocal or fractional diffusion. A shared-pole rational approximant needs only one shifted linear solve per pole regardless of how many time points are evaluated, and when the shared poles are real and negative, every shifted system is symmetric positive definite, so the solves can use Cholesky factorization or preconditioned CG and can be distributed one per processor. Complex shared poles are well studied; the real-pole case has received far less attention, and recent work has produced both closed-form concentrated-pole choices and distinct-pole constructions with markedly better accuracy on wide time intervals. A separate line of recent work addresses the memory bottleneck of Krylov subspace methods for matrix functions $f(A)b$: rather than storing and fully orthogonalizing a long Arnoldi basis, the recurrence is truncated to a short window and the resulting loss of the orthogonality condition needed by the projection formula is repaired algebraically, with the usable basis size governed by monitoring the conditioning of the generated vectors. Generators of the form $A^{1/2}$ connect these threads: semigroups driven by the square root of an elliptic operator model anomalous diffusion, and every shifted resolvent of $A^{1/2}$ is a Stieltjes-type matrix function of $A$ itself, accessible to such Krylov machinery.

## Problem

Families of rational approximants to $\exp(-tz)$ sharing a fixed set of real negative poles, accurate uniformly over a whole time interval, turn the action of an operator exponential into a set of independent symmetric positive definite solves; applied to the square root of a discretized elliptic operator, each shifted resolvent of $A^{1/2}$ must itself be approximated from a Krylov subspace of $A$ built under aggressive storage truncation. Your task is to solve one concrete deterministic example of this pipeline. Use the following configuration:

- $L_1$ = the $24 \times 24$ tridiagonal matrix with $2$ on the diagonal and $-1$ on the sub- and superdiagonals; $M = L_1 \otimes I_{24} + I_{24} \otimes L_1$; $D = \mathrm{diag}(d_0, \dots, d_{575})$ with $d_k = 1 + 3\cos^2(0.7\,k)$; $A = D^{1/2} M D^{1/2}$ (so $N = 576$)
- $b_k = \cos(0.3\,k) + 0.5$ for $k = 0, \dots, 575$, normalized to unit Euclidean norm
- time interval $T = [10^{-2}, 1]$; degree $n = 21$ shared real poles; evaluation time $t = 1$
- sample sets for the error evaluation: $Z = \{0\} \cup$ 3000 logarithmically spaced points in $[10^{-6}, 10^{6}]$, and 40 logarithmically spaced times in $T$
- pole interval $[c, d] = [-551.5183157669,\ -10.1118823417]$, whose discrete time-uniform error on the sample sets above, evaluated with residues obtained from a standard double-precision solve, must be confirmed not to exceed $10^{-5}$
- Krylov process: orthogonalize each new basis vector only against the single most recent previous basis vector; extend until the 2-norm condition number of the assembled basis, including the newest vector, first exceeds $10^{4}$, and take the vectors preceding that newest one as the approximation basis

For the given pole interval, place the $n$ poles and the associated nonnegative interpolation nodes as the extremal configuration for the condenser formed by the pole interval and the nonnegative real axis, minimizing the ratio of the rational nodal function's maximum modulus on the nonnegative axis to its minimum modulus on the pole interval. Determine the time-dependent residues by interpolation of $\exp(-tz)$ at the nodes, and confirm the supplied pole interval by evaluating the worst-case interpolation error of $\exp(-tz)$ over the sample sets above; use the interval exactly as given rather than searching for another. Approximate $u = \exp(-A^{1/2})\,b$ in partial-fraction form, evaluating each shifted resolvent of $A^{1/2}$ applied to $b$ from the truncated Krylov decomposition of $A$, after modifying its projected matrix by the rank-one correction that enforces orthogonality of the trailing basis vector against the image of the basis under $A$: each resolvent is the corresponding function of the corrected projected matrix, formed via its principal square root, applied to the first coordinate vector and scaled by the Euclidean norm of $b$. Your final answer must be a single number: the entry of index 321 (0-based) of $u$.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

In <reasoning>, report the following scalars, with a brief statement of how each was obtained: the ratio $\eta_1/\eta_2$ of the inner to the outer endpoint of the symmetric condenser $[-\eta_2, -\eta_1] \cup [\eta_1, \eta_2]$ and the complete elliptic integral of the first kind at the parameter $\mu = 1 - (\eta_1/\eta_2)^2$; the smallest and largest poles and interpolation nodes; the 2-norm condition number of the residue interpolation system and the largest residue magnitude at $t = 1$; the discrete time-uniform error; the Krylov dimension $m$ and the trailing recurrence coefficient $h_{m+1,m}$; the Euclidean norm of the coefficient vector $c_m$ that expresses the trailing basis vector's component in the span of the approximation basis, before any scaling by $h_{m+1,m}$; and the smallest and largest eigenvalues of the corrected projected matrix, with whether its principal square root is real. Do not paste the input matrices, full pole or residue vectors, per-iteration optimizer paths, or per-sample error tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

condenser_parameters

Goal
----
Compute the parameters of the Mobius transformation that reduces the two-plate condenser formed by a negative pole interval and the nonnegative real axis to a symmetric condenser, together with the resulting elliptic parameter.



Given an interval [c, d] with c < d < 0, the condenser formed by the union of [c, d] and the nonnegative real axis is mapped by the transformation z -> (z - varsigma)/(z - varrho) onto a symmetric condenser consisting of the union of [-eta2, -eta1] and [eta1, eta2], with 0 < eta1 < eta2, where eta1 = -(d - varsigma)/(d - varrho) and eta2 = -(c - varsigma)/(c - varrho) are the negated images of the right and left endpoints of [c, d]. The fifth quantity is the elliptic parameter mu = 1 - (eta1/eta2)^2 of the symmetric condenser, that is, the square of the Jacobi modulus, in the convention of scipy.special.ellipk and scipy.special.ellipj; it is not the modulus itself.



Return the five quantities varsigma, varrho, eta1, eta2, mu in that order as a one-dimensional float array of length 5.



The function raises ValueError if either input is not finite, or if the ordering c < d < 0 does not hold.

```python
def condenser_parameters(c: float, d: float) -> np.ndarray:
    '''Compute Möbius and elliptic parameters reducing [c,d] u [0,inf) to a symmetric condenser.

    Parameters
    ----------
    c : float
        Left endpoint of the pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the pole interval. Must be finite with c < d < 0.

    Returns
    -------
    params : np.ndarray
        One-dimensional float array of length 5 holding, in order, the two Möbius
        parameters varsigma and varrho, the two symmetric-condenser endpoints
        eta1 and eta2, and the elliptic parameter mu = 1 - (eta1/eta2)**2.

    Raises
    ------
    ValueError
        If either input is not finite, or if the ordering c < d < 0 does not hold.
    '''
    return params  # placeholder
```

### Step 2

zolotarev_poles_nodes

Goal
----
Construct the extremal pole and interpolation-node configuration of degree n from the reduced condenser parameters.



Given the five parameters produced by the condenser reduction and a degree n of at least one, solve the Zolotarev problem on the reduced symmetric condenser using Jacobi elliptic functions with the reduced parameter mu (the squared modulus, as taken by scipy.special.ellipj), and transport the resulting configuration back to the original condenser through the inverse of the Mobius map those parameters define. The n poles lie in the original pole interval on the negative real axis and the n interpolation nodes lie on the nonnegative real axis; the two sets are recovered from the same symmetric-condenser quantity by two different inverse maps.



Return a two-row array: row 0 holds the poles and row 1 the interpolation nodes. The columns are ordered so that the pole entries are strictly decreasing, and column i pairs the i-th pole with the i-th node.



The function raises ValueError if the parameter array is not one-dimensional of length 5, if it contains a non-finite entry, if the parameter entry mu is not at least zero and strictly below one, if the second symmetric-condenser endpoint is zero, or if n is not a positive integer.

```python
def zolotarev_poles_nodes(params: np.ndarray, n: int) -> np.ndarray:
    '''Construct extremal poles and interpolation nodes from reduced condenser parameters.

    Parameters
    ----------
    params : np.ndarray
        One-dimensional array of length 5 holding, in order, the two Möbius
        parameters varsigma and varrho, the two symmetric-condenser endpoints
        eta1 and eta2, and the elliptic parameter mu.
    n : int
        Degree of the configuration. Must be a positive integer.

    Returns
    -------
    config : np.ndarray
        Float array of shape (2, n). Row 0 holds the poles in strictly decreasing
        order; row 1 holds the paired interpolation nodes.

    Raises
    ------
    ValueError
        If the parameter array is not one-dimensional of length 5, if it contains a
        non-finite entry, if the parameter entry mu is not at least zero and
        strictly below one, if the second symmetric-condenser endpoint is zero, or
        if n is not a positive integer.
    '''
    return config  # placeholder
```

### Step 3

interpolation_residues

Goal
----
Compute the residues of the partial-fraction rational interpolant of the exponential exp(-t z) determined by a given pole set and interpolation-node set, at a single time t.

The interpolant is a sum over j of alpha_j(t) divided by (z - sigma_j), with the poles sigma_j fixed and independent of t. The residues alpha_j(t) are determined by requiring that the interpolant agree with exp(-t z) at every interpolation node. Return the residues as a one-dimensional array of length n, with entry j the residue attached to pole j, in the same column order as the inputs.

This system is to be solved in precision beyond double, using at least fifty significant decimal digits throughout the elimination, with the residues rounded to double only on return. The matrix entries are to be formed in that same extended precision from the given poles and nodes rather than computed in double and promoted. Carry out that arithmetic with Python's standard-library decimal module; third-party multiprecision packages such as mpmath are not installed.

The inputs are the pole array and the node array from the extremal configuration, both one-dimensional and of equal length, and the time t.

The function raises ValueError if the two input arrays are not one-dimensional of equal positive length, if any entry of either array is not finite, if any pole is not strictly negative, if any node is negative, or if t is not finite and strictly positive.

```python
def interpolation_residues(poles: np.ndarray, nodes: np.ndarray, t: float) -> np.ndarray:
    '''Compute residues of the partial-fraction interpolant of exp(-t z) at given nodes.

    Parameters
    ----------
    poles : np.ndarray
        One-dimensional array of n strictly negative, finite poles.
    nodes : np.ndarray
        One-dimensional array of n nonnegative, finite interpolation nodes.
    t : float
        Time parameter. Must be finite and strictly positive.

    Returns
    -------
    alpha : np.ndarray
        One-dimensional float array of length n; entry j is the residue attached
        to poles[j].

    Raises
    ------
    ValueError
        If the two input arrays are not one-dimensional of equal positive length, if
        any entry of either array is not finite, if any pole is not strictly
        negative, if any node is negative, or if t is not finite and strictly
        positive.
    '''
    return alpha  # placeholder
```

### Step 4

discrete_uniform_error

Goal
----
Evaluate the discrete time-uniform interpolation error of the degree-n extremal configuration associated with a candidate pole interval.

For the given interval [c, d], construct the extremal pole and node configuration of degree n as in the previous steps. For each supplied time t, form the partial-fraction interpolant of exp(-t z) determined by that configuration, evaluate it at every supplied sample point z, and take the largest absolute deviation from exp(-t z). Return the largest such deviation over all supplied times, as a single float.

The interpolation systems that determine the residues inside this objective are solved in standard double-precision floating-point arithmetic.

The inputs are the two interval endpoints, the degree, a one-dimensional array of times, and a one-dimensional array of sample points on the nonnegative real axis.

The function raises ValueError if either endpoint is not finite, if the ordering c < d < 0 does not hold, if n is not a positive integer, if either input array is not one-dimensional and non-empty, if any time is not finite and strictly positive, or if any sample point is not finite and nonnegative.

```python
def discrete_uniform_error(c: float, d: float, n: int, times: np.ndarray,
                           z_samples: np.ndarray) -> float:
    '''Discrete time-uniform interpolation error of the degree-n configuration on [c,d].

    Parameters
    ----------
    c : float
        Left endpoint of the candidate pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the candidate pole interval. Must be finite with c < d < 0.
    n : int
        Degree of the configuration. Must be a positive integer.
    times : np.ndarray
        One-dimensional non-empty array of finite, strictly positive times.
    z_samples : np.ndarray
        One-dimensional non-empty array of finite, nonnegative sample points.

    Returns
    -------
    err : float
        The largest absolute deviation of the interpolant from exp(-t z) over all
        supplied times and sample points, as a native Python float.

    Raises
    ------
    ValueError
        If either endpoint is not finite, if the ordering c < d < 0 does not hold,
        if n is not a positive integer, if either input array is not one-dimensional
        and non-empty, if any time is not finite and strictly positive, or if any
        sample point is not finite and nonnegative.
    '''
    return err  # placeholder
```

### Step 5

truncated_arnoldi_basis

Goal
----
Build a Krylov decomposition of a matrix by a truncated orthogonalization recurrence, stopping adaptively on the conditioning of the generated basis.

Starting from the vector b divided by its Euclidean norm, generate successive vectors by applying A to the most recent basis vector and subtracting its components along the trunc most recently generated basis vectors, one at a time, each subtraction applied before the next coefficient is computed. The coefficient used at each subtraction, together with the norm of the remainder, populates the rectangular upper Hessenberg recurrence matrix H of shape (j+1, j) satisfying that A applied to the first j basis vectors equals the first j+1 basis vectors multiplied by H.

After each new vector is appended, form the 2-norm condition number of the basis assembled so far, including that newest vector. Continue while it remains at or below the threshold; the first time it exceeds the threshold, stop and take the subspace dimension m to be the number of vectors preceding the newest one.

Return a single packed array of shape (N + m + 1, m + 1), where N is the dimension of A. Its first N rows hold the m + 1 basis vectors as columns, the first m of which are the approximation basis and the last of which is the trailing vector. Its remaining m + 1 rows hold the recurrence matrix of shape (m + 1, m) in their first m columns, with the final column of that block set to zero.

The function raises ValueError if A is not a square two-dimensional finite array, if b is not a finite one-dimensional array of matching length with nonzero norm, if trunc is not a positive integer, if tau is not finite and greater than one, if the recurrence breaks down with a zero remainder, or if the threshold is not exceeded before the basis dimension reaches the size of A.

```python
def truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, trunc: int,
                            tau: float) -> np.ndarray:
    '''Truncated Krylov decomposition with conditioning-based adaptive stopping.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    b : np.ndarray
        Finite starting vector of length N with nonzero norm.
    trunc : int
        Number of most recent basis vectors to orthogonalize against.
        Must be a positive integer.
    tau : float
        Basis condition number threshold. Must be finite and greater than one.

    Returns
    -------
    packed : np.ndarray
        Float array of shape (N + m + 1, m + 1). Rows 0..N-1 hold the m+1 basis
        vectors as columns; rows N..N+m hold the (m+1, m) recurrence matrix in
        their first m columns, with the final column of that block zero.

    Raises
    ------
    ValueError
        If A is not a square two-dimensional finite array, if b is not a finite one-
        dimensional array of matching length with nonzero norm, if trunc is not a
        positive integer, if tau is not finite and greater than one, if the
        recurrence breaks down with a zero remainder, or if the threshold is not
        exceeded before the basis dimension reaches the size of A.
    '''
    return packed  # placeholder
```

### Step 6

harmonic_rank_one_update

Goal
----
Repair a truncated Krylov decomposition by a rank-one modification of its projected matrix, enforcing the harmonic orthogonality condition.

The projected evaluation of a matrix function from a Krylov decomposition is justified only when the trailing basis vector satisfies an orthogonality condition, which a truncated recurrence does not provide. Decompose the trailing vector into a component in the span of the approximation basis and a remainder satisfying the required condition, then absorb the first component into the projected matrix.

Here the condition to enforce is that the corrected trailing vector be orthogonal to the image of the approximation basis under A. Determine the coefficient vector of the component of the trailing vector in the basis span that achieves this, and form the corrected projected matrix by adding to the leading m by m block of the recurrence matrix a rank-one term built from that coefficient vector scaled by the trailing recurrence coefficient, acting only on the last column.

The coefficient vector is the solution of the resulting square linear system of order m, computed by direct LU factorization with partial pivoting.

The inputs are the matrix A and the packed decomposition produced by the previous step. Return the corrected projected matrix as a square array of order m.

The function raises ValueError if A is not a square two-dimensional finite array, if the packed array is not two-dimensional with at least two columns and a row count exceeding its column count by at least one, if the leading block width does not match the order of A, if the packed array is not finite, or if the square system determining the coefficient vector is singular.

```python
def harmonic_rank_one_update(A: np.ndarray, packed: np.ndarray) -> np.ndarray:
    '''Apply the harmonic rank-one correction to a truncated Krylov decomposition.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    packed : np.ndarray
        Packed decomposition of shape (N + m + 1, m + 1): its first N rows hold
        the m+1 basis vectors as columns, and its remaining m+1 rows hold the
        (m+1, m) recurrence matrix in their first m columns.

    Returns
    -------
    h_tilde : np.ndarray
        Corrected projected matrix, a float array of shape (m, m).

    Raises
    ------
    ValueError
        If A is not a square two-dimensional finite array, if the packed array is
        not two-dimensional with at least two columns and a row count exceeding its
        column count by at least one, if the leading block width does not match the
        order of A, if the packed array is not finite, or if the square system
        determining the coefficient vector is singular.
    '''
    return h_tilde  # placeholder
```

### Step 7

evaluate_propagator

Goal
----
Assemble the full pipeline and return a single entry of the approximated propagator applied to a vector.

Verify that the discrete time-uniform error at the supplied pole interval does not exceed the supplied bound. Reduce that interval to its condenser parameters, construct the extremal configuration of degree n from them, and obtain the residues of the interpolant of exp(-t z) at the evaluation time.

Independently, build the truncated Krylov decomposition of A from b with the given truncation window and condition threshold, and correct its projected matrix by the harmonic rank-one modification.

Form the principal square root S of the corrected projected matrix. For each pole sigma_i, solve the shifted system whose matrix is S minus sigma_i times the identity and whose right-hand side is the first coordinate vector, weight the solutions by the corresponding residues and sum them, apply the approximation basis once to the summed vector, and scale by the Euclidean norm of b. Accumulate the weighted solutions in increasing pole index before applying the basis.

Return the entry of the resulting vector at the requested index, as a native Python float.

The function raises ValueError if any input fails the validity conditions of the steps it invokes, if the index is not an integer in range for the dimension of A, if the error bound is not finite and positive, or if the discrete time-uniform error at the supplied interval exceeds that bound.

```python
def evaluate_propagator(A: np.ndarray, b: np.ndarray, n: int, c: float, d: float,
                        times: np.ndarray, z_samples: np.ndarray, t_eval: float,
                        trunc: int, tau: float, index: int,
                        err_bound: float) -> float:
    '''Evaluate one entry of the approximated propagator exp(-t_eval * A**(1/2)) b.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    b : np.ndarray
        Finite starting vector of length N with nonzero norm.
    n : int
        Degree of the shared-pole rational approximation. Positive integer.
    c : float
        Left endpoint of the pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the pole interval. Must be finite with c < d < 0.
    times : np.ndarray
        One-dimensional non-empty array of finite, strictly positive times.
    z_samples : np.ndarray
        One-dimensional non-empty array of finite, nonnegative sample points.
    t_eval : float
        Evaluation time. Must be finite and strictly positive.
    trunc : int
        Truncation window for the Krylov recurrence. Positive integer.
    tau : float
        Basis condition number threshold. Finite and greater than one.
    index : int
        Zero-based entry of the result vector to return.
    err_bound : float
        Upper bound the discrete time-uniform error must satisfy.

    Returns
    -------
    value : float
        The requested entry of the approximated propagator applied to b.

    Raises
    ------
    ValueError
        If any input fails the validity conditions of the steps it invokes, if the
        index is not an integer in range for the dimension of A, if the error bound
        is not finite and positive, or if the discrete time-uniform error at the
        supplied interval exceeds that bound.
    '''
    return value  # placeholder
```
