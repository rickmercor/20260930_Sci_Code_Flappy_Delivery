# Biology-Ecology-8

## Background

The task reconstructs latent ecological interaction intensities from replicated counts under imperfect detection. A structured nonnegative factorization couples between-group interactions and within-group similarity matrices, with feature-dependent detectability and nonconvex entrywise half-power regularization. The requested value is defined by a prescribed finite optimization run, not a convergence or global-optimality claim.

## Problem

Ecological interaction counts confound underlying interaction intensity with imperfect detection, while sparsity in both between-group interactions and within-group similarities complicates network reconstruction. Locate the method that addresses this problem through structured sparse nonnegative factorization with entrywise half-power regularization, a feature-dependent detection model, and adaptive augmented-Lagrangian penalties, and use it to recover one latent plant–pollinator interaction intensity from the prescribed finite computation.

Use four plants, three pollinators, and three independent survey replicates, with rows denoting plants and columns denoting pollinators:
\[
Y^{(1)}=
\begin{pmatrix}
0&3&0\\
2&0&4\\
6&1&0\\
0&2&5
\end{pmatrix},\qquad
Y^{(2)}=
\begin{pmatrix}
1&5&0\\
1&0&3\\
4&0&1\\
0&4&6
\end{pmatrix},\qquad
Y^{(3)}=
\begin{pmatrix}
0&4&0\\
3&1&5\\
5&1&0\\
0&3&4
\end{pmatrix},
\]
\[
x=(-0.9,-0.2,0.4,1.0),\qquad
z=(-0.8,0.1,0.9),\qquad
Z_{(i,j),:}=(1+0.2x_i,\;0.5+0.1z_j),
\]
where \(Z\) orders pairs lexicographically, with the pollinator index varying fastest.
Every entry is observed, including zeros, and each replicate has its own latent count under the source’s observation model.
Use the source’s absolute-singular-vector initialization with rank \(F=2\) and average detection level \(p_0=0.55\), together with these initial conditions and parameters:
\[
\alpha^{(0)}=(0.4,0.3)^T,\qquad
A_X^{(0)}=M_X^{(0)},\qquad W_X^{(0)}=0,
\]
\[
\begin{array}{c|ccc}
X&UU&UV&VV\\ \hline
M_X&UU^T&UV^T&VV^T\\
\lambda_X&0.8&1.1&0.9\\
\rho_X^{(0)}&0.08&0.06&0.10
\end{array}
\qquad
\gamma=1.5,\qquad
\epsilon_X(k)=\frac{0.5}{k^{1.2}},\quad k=1,\ldots,12.
\]

Evaluate twelve outer iterations of the retrieved algorithm under the following numerical contract:
| Component | Prescribed convention |
|---|---|
| Detection subproblem | Exactly 80 inner ADMM updates per outer iteration, in the source’s order \(p,\alpha,\omega\), with inner penalty \(1\) |
| Detection initialization | Carry \(\alpha\) from the preceding outer iteration; reset the inner scaled dual \(\omega\) to zero |
| Detection output | Use the final bounded auxiliary \(p\), reshaped to \(4\times3\), in the factor updates |
| Factor subproblems | Exactly eight projected-gradient updates of \(U\), followed by exactly eight of \(V\), starting from the current factors |
| Backtracking | Restart \(t=0.1\) at every projected-gradient update and halve it until the acceptance condition below holds |
| Acceptance condition | For current block \(Q\), gradient \(g\), and candidate \(Q'=\max(Q-tg,0)\), require \(\Phi(Q')\leq\Phi(Q)+10^{-4}\langle g,Q'-Q\rangle_F\), where \(\Phi\) is that block’s augmented objective |
| Likelihood domain | No numerical floor or pseudocounts; \(0\log0=0\), and candidates with zero intensity at a positive-count entry have infinite objective |
| Auxiliary subproblems | Global entrywise minimizers, choosing zero when zero and a nonzero minimizer tie |
| Outer penalty updates | The source’s residual-based rule, applied separately to each \(X\) after all auxiliary updates |
| Arithmetic | Double precision or higher; no early stopping |

Derive the block objectives from the source’s unscaled augmented Lagrangian using residual \(M_X-A_X\) and scaled dual \(W_X=H_X/\rho_X\), interpreting each half-power penalty as the sum of square roots of absolute entries and retaining all matrix entries, including diagonals.
The iteration budgets define the requested finite-run statistic rather than a claim of convergence or global optimality.
Report ((U^{(12)}V^{(12)T})_{2,3}), using one-based indices, to six decimal places, and identify the retrieved source; justify the computation with compact equations or precise explanations for the replicated-count likelihood, the global zero-versus-nonzero auxiliary selection condition, and the complete scaled-dual update when a penalty changes.


Coding subproblem scope: the auxiliary minimization routine must also support
entrywise nonnegative weights and strictly positive penalties, each broadcast
to the shape of its signed center array without expanding that shape. Support
positive weight/penalty ratios from 1e-240 through 1e240 and nonzero center
magnitudes from 1e-200 through 1e200, preserving relative accuracy of 1e-9
on active entries away from the switching threshold. This is a separable generalization of
the same auxiliary objective. The finite reconstruction requested here uses
the scalar block coefficients in the table above. The final orchestrator
must directly compose the primitive steps for its first cycle, check that
the factor sweep does not increase its fixed-block augmented objective,
then continue from that full state through the outer-cycle routine.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

01_spectral_start.py

Goal
----
Construct the scale-aware rank-limited initialization from replicated ecological counts.

```python
def spectral_start(Y: 'np.ndarray', rank: int, p0: float) -> tuple:
    """Aggregate independent replicate counts and initialize nonnegative factors.

    Parameters
    ----------
    Y : ndarray, shape (I,J,M)
        Finite nonnegative integer counts; I,J,M >= 1. Last axis is replicate.
    rank : int
        Retained rank, 1 <= rank <= min(I,J); booleans are excluded.
    p0 : float
        Finite initial mean detection probability, 0 < p0 <= 1.

    Returns
    -------
    S, U, V : tuple of ndarrays
        S=sum(Y,axis=2), shape (I,J). Take the descending SVD L,s,Rt
        of S/(M*p0). U=abs(L[:,:rank])*sqrt(s[:rank]);
        V=abs(Rt[:rank,:].T)*sqrt(s[:rank]). Do not fit an NMF or
        rectify the reconstructed matrix instead of the singular vectors.

    Raises
    ------
    ValueError
        Non-real or nonnumeric input; invalid dimensions, counts, rank or p0;
        any retained singular value <= 1e-12*s[0], or adjacent singular
        values separated by <= 1e-12*s[0] when their upper index is retained
        (including the rank cutoff); failed SVD; or nonfinite result.
    """
    return result
```

### Step 2

02_detection_update.py

Goal
----
Solve the feature-dependent detection subproblem for a prescribed number of ADMM iterations.

```python
def detection_update(S: 'np.ndarray', intensity: 'np.ndarray', Z: 'np.ndarray', alpha: 'np.ndarray', replicates: int, steps: int = 80, penalty: float = 1.0) -> tuple:
    """Perform a finite constrained detection ADMM solve.

    Parameters
    ----------
    S, intensity : ndarrays, common shape (I,J), I,J >= 1
        Nonnegative finite aggregate counts (S may be real) and latent means.
        Positive S requires strictly positive intensity.
    Z : ndarray, shape (I*J,R), R >= 1
        Finite real design; row-major pair order. Rank deficiency is allowed.
    alpha : ndarray, shape (R,)
        Finite starting coefficients, with no sign constraint.
    replicates, steps : int
        Positive integers, booleans excluded. Execute exactly steps updates.
    penalty : float
        Finite strictly positive inner ADMM penalty eta.

    Returns
    -------
    P, alpha_new, omega : tuple of ndarrays
        P has shape (I,J); alpha_new shape (R,); omega shape (I*J,).
        Reset omega=0 on entry. Each iteration uses b=Z@alpha-omega,
        d=eta*b-replicates*intensity.ravel(order='C'), then
        p=clip((d+sqrt(d*d+4*eta*S.ravel()))/(2*eta),0,1),
        alpha=pinv(Z)@(p+omega), omega=omega+p-Z@alpha, in this order.
        Use Moore-Penrose pinv with relative singular cutoff 1e-15.
        Return auxiliary p, NOT Z@alpha; there is no convergence stopping.
        Evaluate the quadratic root without catastrophic cancellation.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; inconsistent or empty shapes; nonfinite
        entries; negative counts/intensities; zero intensity at positive count;
        invalid integer controls or penalty; failed pseudoinverse; or a
        nonfinite intermediate/result.
    """
    return result
```

### Step 3

03_factor_terms.py

Goal
----
Evaluate the Poisson factor objective and the derivatives of all three Gram-product constraints.

```python
def factor_terms(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int) -> tuple:
    """Evaluate the factor objective and both exact Euclidean gradients.

    Parameters
    ----------
    S, P : ndarrays, shape (I,J), I,J >= 1
        Finite aggregate counts S>=0 and auxiliary detection probabilities
        0<=P<=1; S may be real. P is fixed during differentiation.
    U, V : ndarrays, shapes (I,F), (J,F), F >= 1
        Finite nonnegative factors. Positive counts require (U@V.T)>0.
    B : tuple/list of three arrays
        Centers ordered UU,UV,VV with shapes (I,I),(I,J),(J,J).
        All finite; within-group centers must be exactly symmetric.
    rho : ndarray, shape (3,)
        Finite strictly positive penalties ordered UU,UV,VV.
    replicates : int
        Positive integer, booleans excluded.

    Returns
    -------
    value, grad_U, grad_V : tuple
        Native float and arrays shaped like U,V. Evaluate
        sum(replicates*P*L - S*log(L)) +
        sum_X rho_X/2 * ||M_X-B_X||_F**2,
        where L=U@V.T and M=(U@U.T,U@V.T,V@V.T).
        Include all matrix entries, including diagonals and both symmetric
        off-diagonal entries. Interpret 0*log(0) and 0/0 count ratios as zero.
        Derive gradients with respect to the factors, including both
        appearances of a factor in its within-group Gram matrix.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; malformed/nonfinite inputs; incompatible
        or empty dimensions; invalid probabilities, negative counts/factors,
        asymmetric within-group centers, nonpositive penalties, invalid
        replicate count, zero intensity at positive count, or nonfinite
        objective/gradient/product.
    """
    return result
```

### Step 4

04_factor_sweep.py

Goal
----
Apply ordered finite projected-gradient sweeps with the prescribed Armijo rule.

```python
def factor_sweep(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int, steps: int = 8, initial_step: float = 0.1, armijo: float = 0.0001) -> tuple:
    """Update U then V by finite projected-gradient Armijo sweeps.

    Parameters
    ----------
    S, P, U, V, B, rho, replicates
        Exactly the shapes, domains, objective and gradient contract of
        factor_terms. Do not mutate any input.
    steps : int
        Nonnegative updates per block, booleans excluded; zero returns copies.
    initial_step : float
        Finite strictly positive starting t, reset at EVERY update.
    armijo : float
        Finite acceptance coefficient strictly between zero and one.

    Returns
    -------
    U_new, V_new : tuple of ndarrays
        Perform steps updates of U with V frozen, then steps updates of V
        with the newly updated U frozen. Freeze B,rho,P for the whole sweep.
        Q'=maximum(Q-t*g,0), halving t until Phi(Q') <=
        Phi(Q)+armijo*sum(g*(Q'-Q)). Use the projected displacement.
        A trial with zero intensity at positive count or nonfinite terms
        is rejected; do not floor intensities. No convergence stopping.
        An unchanged projected candidate is accepted by the same inequality.
        At most 100 candidate trials per gradient update, starting at t0.

    Raises
    ------
    ValueError
        Any violation of factor_terms' input contract; non-real/nonnumeric
        initial_step/armijo or values outside the ranges above; invalid steps;
        or no acceptable candidate in 100 trials.

    Inherited requirements (explicit)
    ---------------------------------
    S,P must have common nonempty shape (I,J); S>=0 and 0<=P<=1. U,V have shapes (I,F),(J,F), F>=1, and are nonnegative. B has shapes (I,I),(I,J),(J,J), with exactly symmetric first/last matrices; rho is length three and strictly positive. All entries are finite real numbers. Replicates is a positive nonboolean integer. A positive S entry requires positive U@V.T. Nonnumeric, non-real, nonfinite, malformed inputs or nonfinite initial objective/gradients raise ValueError.
    """
    return result
```

### Step 5

05_half_prox.py

Goal
----
Compute global half-power proximal minimizers with entrywise broadcast weights and penalties, preserving branch selection and relative accuracy across numerical scales.

```python
def half_prox(
    b: 'np.ndarray',
    weight: 'np.ndarray | float',
    penalty: 'np.ndarray | float',
) -> 'np.ndarray':
    """Return the entrywise global minimizer of
    weight*sqrt(abs(a)) + penalty*(a-b)**2/2 over real a.

    b is a nonempty finite real array of any shape, including shape ().
    weight and penalty are finite real scalars or arrays. Each must be
    independently broadcastable TO b.shape without expanding that shape.
    Every weight is nonnegative; every penalty is strictly positive.
    A zero weight gives a=b at that entry. Signs are unrestricted.
    When zero and a nonzero minimizer tie, return zero. Selecting a
    stationary point without comparing its objective with zero is invalid.

    Return a new float ndarray with exactly b.shape; mutate no input.
    The effective ratio weight/penalty and its two-thirds power must be
    finite. A positive weight whose effective ratio underflows to zero
    is outside the numerical contract and must raise ValueError.
    For positive effective ratios from 1e-240 through 1e240 and nonzero
    abs(b) from 1e-200 through 1e200, valid finite outputs must not fail
    merely because an avoidable intermediate power or squared objective
    overflows. On active entries away from the switching threshold,
    preserve relative accuracy of 1e-9 even for outputs below 1e-9.
    At the switching threshold, evaluate its represented floating-point
    value consistently and choose zero for equality. Do not use an
    absolute tolerance that erases genuinely active small-scale entries.

    Raise ValueError for nonnumeric/complex/nonfinite inputs, empty b,
    incompatible broadcasting, invalid coefficient signs, a nonfinite
    effective ratio/threshold/output, or the ratio underflow above.
    """
    return a
```

### Step 6

06_adapt_duals.py

Goal
----
Update three independent penalties and their scaled dual variables without altering the intended unscaled update.

```python
def adapt_duals(products: tuple, auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', gamma: float, tolerance: 'np.ndarray') -> tuple:
    """Adapt three penalties while preserving the unscaled dual update.

    Parameters
    ----------
    products, auxiliaries, duals : tuple/list of three finite real matrices
        Corresponding shapes (I,I),(I,J),(J,J), I,J>=1. Matrices may be
        signed and need not be symmetric for this algebraic operation.
        Residual = products - auxiliaries; duals contain W=H/rho.
    rho, tolerance : ndarray, shape (3,)
        Finite rho>0 and tolerance>=0, in UU,UV,VV order.
    gamma : float
        Finite multiplier strictly greater than one.

    Returns
    -------
    rho_new, duals_new, residual_norms : tuple
        Penalty vector, tuple of three arrays, and length-3 Frobenius norms.
        For each block, increase rho by gamma iff norm(residual)>tolerance;
        equality leaves rho unchanged. Apply H_new=H+rho_old*residual,
        then express H_new using the NEW penalty. Never mutate inputs.

    Raises
    ------
    ValueError
        Non-real/nonnumeric inputs; missing blocks, malformed or inconsistent
        shapes, nonfinite entries, nonpositive rho, negative tolerance,
        gamma<=1, or nonfinite computed norms, penalties or duals.
    """
    return result
```

### Step 7

07_outer_cycle.py

Goal
----
Combine the detection, factor, nonconvex auxiliary, and adaptive-dual operations in the prescribed order.

```python
def outer_cycle(S: 'np.ndarray', replicates: int, Z: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', alpha: 'np.ndarray', auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', weights: 'np.ndarray', iteration: int, detection_steps: int = 80, factor_steps: int = 8) -> tuple:
    """Advance one prescribed sparse-network outer iteration.

    Parameters
    ----------
    S, replicates, Z, U, V, alpha
        Data and factors satisfying detection_update and factor_terms contracts.
    auxiliaries, duals : tuple/list of three arrays
        Shapes (I,I),(I,J),(J,J), finite real entries. Each within-group
        array must be exactly symmetric. Auxiliary entries may be negative.
    rho, weights : ndarray, shape (3,)
        Finite rho>0 and weights>=0, ordered UU,UV,VV.
    iteration : int
        Positive one-based outer index; booleans excluded.
    detection_steps : int
        Positive finite-update budget for detection_update, default 80.
    factor_steps : int
        Nonnegative budget per factor in factor_sweep, default 8.

    Returns
    -------
    U_new,V_new,alpha_new,A_new,W_new,rho_new,P : tuple
        Reset detection omega, perform detection with penalty 1, then update
        U and V with B=A-W, initial step .1 and Armijo coefficient 1e-4.
        With updated factors, globally minimize all three auxiliary blocks
        centered at M_X+W_X using weights and OLD rho. Then adapt penalties
        and duals with gamma=1.5 and tolerance_X=.5/iteration**1.2.
        P is the finite detection auxiliary from BEFORE the factor sweep;
        do not recompute it using final factors. Inputs remain unchanged.

    Raises
    ------
    ValueError
        Any violated earlier-step contract (including their numerical
        failures); invalid iteration; invalid weights; malformed auxiliary
        or dual blocks; or asymmetric within-group auxiliary/dual matrices.

    Inherited requirements (explicit)
    ---------------------------------
    S is nonempty (I,J), nonnegative; U,V are nonnegative (I,F),(J,F), F>=1. A,W each have shapes (I,I),(I,J),(J,J), with exactly symmetric first/last matrices. Z is (I*J,R), R>=1; alpha is (R,); rho and weights are (3,), rho>0 and weights>=0. All entries must be finite and real. Positive S requires positive U@V.T. Replicates, iteration and detection_steps are positive nonboolean integers; factor_steps is a nonnegative nonboolean integer. Violations, failed pseudoinverse, nonfinite computed terms, or a line search failing all 100 candidates raise ValueError.
    """
    return result
```

### Step 8

08_sparse_intensity.py

Goal
----
Return the prescribed latent intensity by directly composing Steps 01–06 for initialization and the first cycle, then passing that updated state to Step 07 for all remaining cycles.

```python
def sparse_intensity(Y: 'np.ndarray', Z: 'np.ndarray', rank: int, p0: float, alpha0: 'np.ndarray', weights: 'np.ndarray', rho0: 'np.ndarray', outer_steps: int = 12, target: tuple = (1, 2)) -> float:
    """Orchestrate the finite sparse ecological-network reconstruction.

    Parameters
    ----------
    Y, rank, p0
        spectral_start contract, with Y shape (I,J,M).
    Z, alpha0
        detection_update design and coefficient contracts; row-major pairs.
    weights, rho0 : ndarray, shape (3,)
        Finite weights>=0 and rho0>0, in UU,UV,VV order.
    outer_steps : int
        Nonnegative number of outer iterations; booleans excluded.
    target : tuple/list of two ints
        ZERO-based plant,pollinator indices, within (I,J); booleans excluded.

    Returns
    -------
    result : float
        (U@V.T)[target] after exactly outer_steps cycles. Initialize factors
        with spectral_start, auxiliaries with the three factor products,
        outer duals with zeros and alpha=alpha0. Cycle indices start at one;
        use 80 detection updates and eight updates per factor each cycle.
        For outer_steps=0 return the initialized intensity. No rounding,
        detection multiplier, convergence stopping or global-optimum claim.
        All input contracts are checked even if outer_steps=0.

    Raises
    ------
    ValueError
        Any violation of the earlier-step contracts or their numerical
        failure conditions; malformed/nonfinite weights or rho0; invalid
        outer_steps or target. Inputs must not be mutated.


    Composition requirements
    ------------------------
    Call spectral_start directly. When outer_steps>=1, implement cycle 1
    directly using detection_update, factor_terms, factor_sweep, half_prox,
    and adapt_duals, consuming their returned values in that order.
    Evaluate factor_terms before and after the factor_sweep with the same
    P, B=A-W and OLD rho. Raise ValueError if the final objective exceeds
    the initial objective by more than 1e-10*(1+abs(initial objective)).
    Then form all three products, compute all auxiliaries using OLD rho,
    and update rho and W. For cycle indices 2..outer_steps, pass the full
    evolving state into outer_cycle. Do not restart or duplicate cycle 1.
    The zero-cycle path still validates inherited input contracts.

    Inherited requirements (explicit)
    ---------------------------------
    Y is a nonempty (I,J,M) real finite nonnegative integer array. Rank is a nonboolean integer in [1,min(I,J)]; 0<p0<=1, finite. Retained singular values must exceed 1e-12*s[0]; adjacent gaps whose upper index is retained, including the cutoff, must exceed 1e-12*s[0]. Z is finite real (I*J,R), R>=1, and alpha0 is finite real (R,). Weights and rho0 are finite real (3,), weights>=0, rho0>0. Outer_steps is a nonnegative nonboolean integer; target contains two nonboolean integers in [0,I) and [0,J). Violations, failed SVD/pseudoinverse, nonfinite computed values, zero intensity at positive count in a current iterate, or exhaustion of 100 line-search trials raise ValueError. These validations also apply when outer_steps=0.
    """
    return result
```
