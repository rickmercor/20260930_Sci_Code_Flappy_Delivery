# Four-step residual of an adaptively fitted inverse cube-root iteration

## Background

# Scientific background

Matrix functions such as signs, square roots, inverse roots, and polar factors are recurring primitives in numerical optimization and preconditioned neural-network training. Their direct evaluation by eigendecomposition can be expensive or poorly matched to accelerator hardware, so multiplication-based polynomial iterations are attractive. Fixed-coefficient iterations, however, converge at a rate set once and for all by the classical derivation, regardless of where the residual spectrum actually sits at a given iteration.

Recent work addresses this gap by adapting a single scalar of the step to the spectrum encountered at each iteration. For the inverse $p$-th root the classical iteration advances a coupled pair of matrices under coefficients that the derivation fixes once and for all, while the accelerated version refits the step against the current residual. A low-dimensional Gaussian sketch approximates the residual-fitting objective without explicit spectral estimates, retaining matrix-multiplication structure while reducing the fitting cost from cubic to sketch-dependent quadratic scaling in the matrix dimension. Because the fitting objective has degree $2p$, from $p=3$ upward its stationary condition is a quintic or higher and admits no solution by radicals, so the coefficient must be selected numerically over a bounded interval. The task isolates this adaptive mechanism on a small symmetric positive definite spectrum so that the complete computation is deterministic and independently reproducible.

## Problem

A coupled multiplication-only iteration for the inverse $p$-th root of a symmetric positive definite matrix can be accelerated by adapting one scalar coefficient to the evolving residual spectrum. Using IEEE-754 float64 arithmetic, take the symmetric positive definite dimensionless matrix

$$
A=\begin{bmatrix}
1.386992364457539&-0.23851410329282224&0.9305755769562332&-0.24521904288029747&0.863387655933195\\
-0.23851410329282224&0.6204795094212204&-0.0384098637531652&-0.2427201929548865&-0.192940910922431\\
0.9305755769562332&-0.0384098637531652&2.983520953383781&-0.3076184799486571&1.3954411171107222\\
-0.24521904288029747&-0.2427201929548865&-0.3076184799486571&1.383883409686169&-0.7374298713308978\\
0.863387655933195&-0.192940910922431&1.3954411171107222&-0.7374298713308978&1.9522749162799236
\end{bmatrix},
$$

and the fixed realized sketch

$$
S=\begin{bmatrix}
-0.3851135890587634&0.8108476339474449&-0.8913718361585312&0.767607275753405&-1.1712404048300675\\
0.5452417714953117&-1.0441262392158024&-1.837068108918472&-0.5937675843336032&-1.4639144154991144\\
0.5532782302206338&0.02167111699934259&0.5094464961689524&0.09130282542822574&-0.3538590292305842
\end{bmatrix}.
$$

Recover from the published construction its coupled starting pair and the scaling constant that pair uses, its residual, which single scalar of the step is refitted at every iteration, the sketched post-step fitting loss, and the form of the update, and state briefly in the reasoning the recovered starting pair, that scaling constant, the residual, both the $X$ and the $M$ recurrence of the update, and the fitting loss. Work at root order $p=3$, run four adaptive iterations while reusing the displayed sketch $S$ at every fit, and write each scalar fitting loss as the ascending polynomial $m_k(\alpha)=\sum_{j}c_{k,j}\alpha^j$. Restrict every fitted coefficient to the closed interval $[\ell,u]=[1/3,1]$. Build each iteration's candidate set from the interval endpoints together with the real roots of $m_k'$, treating a root as real when its imaginary magnitude is at most $10^{-10}$, clipping roots within $10^{-12}$ of an endpoint onto that endpoint, and deduplicating within $10^{-12}$ so that the smaller representative survives. Score the candidates by Horner's rule and select by a single ascending scan rather than by a global argument-minimum: the smallest candidate is provisionally best, and a later candidate replaces the running best only when it improves the score by more than $10^{-14}$. Use the displayed arrays directly rather than regenerating them, and do not round any intermediate quantity. In the short reasoning, report to at least 10 significant digits the scaling constant, $\|R_0\|_F$, $c_{0,1}$, $\widetilde\alpha_3$, $m_3(\widetilde\alpha_3)$ and $\|R_3\|_F$, state how many of the four iterations select an interior coefficient rather than an interval endpoint, and give the final residual of one diagnostic rerun in which $S$ is replaced by $I_5$ in every coefficient fit and nothing else changes; that diagnostic residual is not the answer. Also in that short reasoning, derive the post-step residual expansion from the coupled update rather than asserting it, give the degree of the scalar fitting loss together with the highest sketched trace moment its coefficients require, say why the stationary condition cannot be solved in closed form at $p=3$, state whether the Frobenius residual falls at every one of the four steps, and explain why the identity-sketch rerun is a different quantity rather than a better estimate of the required one. Return the four-step residual's Frobenius norm, rounded to 12 digits after the decimal point, as the single dimensionless numerical target.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

01_scaled_starting_matrix

Goal
----
The coupled inverse Newton iteration for the inverse `$p$`-th root carries a pair of matrices rather than a single iterate. The pair stays tied by `$M = X ** p A$` at every iteration, so the residual `$R = I - M$` measures how far `$X$` is from the inverse `$p$$-th root of$$A$` without `$X$` ever being formed, and the whole run can be reported from `$M$` alone.



Both members of the pair start from one positive scaling constant: `$X$` starts from the identity divided by that constant, and `$M$` from `$A$` divided by the `$p$$-th power of the same constant, which is what makes the tie hold at$$k = 0$`. The published constant is the one that normalises the starting `$M$` exactly, that is, the unique positive scalar for which the Frobenius norm of the starting `$M$` equals `$(p + 1) / 2$`. Because the Frobenius norm dominates every eigenvalue of a symmetric matrix, that choice places the whole spectrum of the starting `$M$` inside ``(0, (p + 1) / 2]``, which is the region in which the iteration contracts. That bound is one-sided: it holds the starting residual spectrum below one but not above zero, so a negative eigenvalue in the starting residual is expected rather than a symptom of a wrong constant.



This step returns the starting `$M$` of that scaled pair. Its residual is read off it by the defining relation and is not returned here.



The normalization is homogeneous, so multiplying `$A$` by any positive scalar must leave the returned `$M$` unchanged. That invariant is required over the full finite float64 range: the implementation must not first square entries in a way that makes a representable nonzero matrix look like zero, or makes its Frobenius norm overflow. If an input is accepted by the `$1e-12$` symmetry tolerance, it represents the symmetric matrix ``(A + A.T) / 2``; that projection is made before the norm and the scaling are formed so the returned member still has the symmetry on which the coupled construction relies.

```python
def scaled_starting_matrix(A: np.ndarray, p: int) -> np.ndarray:
    """Return the residual-bearing member of the scaled starting pair.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is real
    rather than complex; ``A`` is a nonempty square two-dimensional array; every
    entry of ``A`` is finite; ``A`` is symmetric to an absolute tolerance of
    ``1e-12``, so a matrix that is asymmetric only at the ``1e-13`` level is
    accepted and projected to its symmetric part rather than rejected; that
    symmetric part is nonzero; and ``p`` is an integer, not a bool, with ``p >=
    1``. The finite-input contract includes magnitudes for which a naive
    sum-of-squares Frobenius norm overflows or underflows.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    p : int
        Root order. The iteration targets the inverse ``p``-th root of ``A``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the starting ``M``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 2

02_sketched_loss_coefficients

Goal
----
One accelerated iteration replaces `$M$$by$$(I + alpha R) ** p M$`, where `$R = I - M$` is the residual carried into the step and ``alpha`` is the single free scalar the accelerated method refits at that iteration. The residual after the step is read off the new `$M$` by the same defining relation. Every iterate of the coupled pair is a polynomial in the original matrix, so `$R$` commutes with `$M$` and no ordering convention is needed anywhere below.



Eliminating `$M$` in favour of `$R$$turns the post-step residual into a matrix polynomial in$$R$` alone whose scalar coefficients are polynomials in ``alpha``: one degree higher in `$R$` than the order of the step, and of degree exactly the order of the step in ``alpha``. The constant term in ``alpha`` is the residual carried in, unchanged, which is why setting ``alpha`` to zero freezes the iteration rather than restarting it. Deriving that expansion is the first half of this step; it depends on `$p$` alone and forms no matrix.



The free scalar is then chosen by minimising how large that post-step residual is, but measuring it in full would cost a cubic number of operations in the matrix dimension at every iteration. The published fit instead compresses the post-step residual from the left with a fixed sketch `$S$` and minimises the squared Frobenius norm of the compressed matrix, so the quantity being minimised is a scalar polynomial in ``alpha``. This step returns its coefficients in ascending order.



Because `$R$$is symmetric, so is the post-step residual at every$`alpha``, and the squared Frobenius norm of `$S$` against it collapses onto a trace of `$S$` against a single power of `$R$`. Each coefficient of the loss is therefore a linear combination of the scalar sketched moments ``trace(S @ R ** i @ S.T)``, and how far `$i$` has to run is fixed by the degree of the expansion alone. Accumulating those scalars, rather than the matrix powers behind them, is what reduces the fit from cubic to sketch-dependent quadratic scaling in the matrix dimension; it is also why the sketch must be applied before the norm is squared rather than after.



The derivation assumes an exactly symmetric residual. An input admitted by the absolute symmetry tolerance therefore represents ``(R + R.T) / 2`` and must be projected onto that invariant subspace before any moment is accumulated. This is observable when a small residual is paired with a large sketch, and is not the same as silently using the accepted but nonsymmetric entries. The construction is also degree-general: no coefficient table for one particular `$p$` may be hard-coded, and the returned length and highest moment must continue to follow from the input order when the derivative degree is well above five.

```python
def sketched_loss_coefficients(R: np.ndarray, S: np.ndarray, p: int) -> np.ndarray:
    """Reduce the sketched post-step residual to a scalar polynomial in ``alpha``.

    Raises ``ValueError`` unless every one of the following holds: ``R`` and
    ``S`` are real rather than complex; ``R`` is a nonempty square
    two-dimensional array; ``R`` is symmetric to an absolute tolerance of
    ``1e-12``, so a residual that is asymmetric only at the ``1e-13`` level is
    accepted and projected to ``(R + R.T) / 2`` rather than rejected; ``S`` is
    two-dimensional with at least one row and with its column count equal to the
    dimension of ``R``, so a sketch of the wrong width is rejected rather than
    broadcast; every entry of both is finite; and ``p`` is an integer, not a
    bool, with ``p >= 1``.

    Parameters
    ----------
    R : np.ndarray
        Nonempty finite real symmetric residual of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order of the accelerated step.

    Returns
    -------
    np.ndarray
        Float64 vector of shape ``(2 * p + 1,)`` holding the coefficients of the
        sketched loss in ascending powers of ``alpha``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 3

03_fit_bounded_coefficient

Goal
----
The fitted coefficient of an iteration is the constrained minimiser of the sketched loss on a closed interval. This step performs that whole minimisation and returns the coefficient itself; no intermediate set is reported.



A polynomial is smooth, so its minimum over a closed interval is attained either at one of the two endpoints or at a stationary point strictly inside, and only that finite set has to be examined. At order `$p$` the sketched loss has degree `$2 * p$`, so its derivative has degree `$2 * p - 1$`. From `$p = 3$` upward that derivative is a quintic or higher, which by Abel's theorem has no solution by radicals, so the stationary points cannot be written in closed form and must be obtained numerically as the eigenvalues of the derivative's companion matrix. That numerical route is why the conventions below are part of the specification rather than an implementation detail: a real stationary point arrives with a small spurious imaginary part, and a stationary point sitting on an endpoint arrives displaced from it.



The admissible set is built by these conventions, applied in this order.



1. Differentiate the ascending coefficient vector term by term. 2. Write the derivative in descending order and drop its leading zeros. If fewer than two coefficients remain, the derivative has no companion matrix and the polynomial contributes no stationary point at all. 3. Take the companion-matrix eigenvalues of what remains. An eigenvalue counts as real when its imaginary magnitude is at most `$1e-10$`, and is discarded otherwise; the surviving point is its real part. 4. A surviving point within `$1e-12$` of an endpoint is clipped onto that endpoint, and a point outside the closed interval after clipping is dropped. 5. Both endpoints are admissible unconditionally, whatever the derivative does. 6. Sort every admissible point into ascending order, then discard any point within `$1e-12$` of one already kept, so that the smaller representative of a cluster survives.



The coefficient is then chosen from that ascending set. Each candidate is scored by evaluating the loss at it with Horner's rule on the ascending coefficient vector. Horner is not decoration here: by the last iteration the sketched loss has fallen many orders of magnitude, its coefficients no longer share a scale, and evaluating the powers separately loses the cancellation that produces the small value.



The comparison is a single ascending scan rather than a global argument-minimum. The first candidate is provisionally the best. Each later candidate replaces the current best only when its score is smaller than the current best score by more than `$1e-14$`; otherwise the earlier, and therefore smaller, candidate stays in place. Scores within `$1e-14$` of each other are in that sense tied, and the fit never trades a numerically indistinguishable improvement for a larger coefficient. Because every comparison is made against the running best and not against the global minimum, the scan can settle on a candidate whose score is not the smallest in the set: once the running best has moved, a later candidate that improves on it by less than the tolerance is refused, even though a global argument-minimum would have selected it.



Uniformly scaling every loss coefficient does not move a stationary point, but it does interact with the *absolute* `$1e-14$` score tolerance. The implementation must therefore remain defined when finite coefficients are so large that forming ``j * c[j]`` directly would overflow, and when they are so small that a solver normalizes them to find the roots. Root finding may use a uniformly normalized derivative, but a normalized Horner scan must scale the comparison tolerance by the same factor; otherwise it silently replaces the specified absolute tie rule with a relative one. No restriction to a convenient coefficient magnitude is part of the public contract.

```python
def fit_bounded_coefficient(c: np.ndarray, lower: float, upper: float) -> float:
    """Minimise the sketched loss over the closed coefficient interval.

    Raises ``ValueError`` unless every one of the following holds: ``c`` is real
    rather than complex; ``c`` is one-dimensional with at least two entries;
    every entry of ``c`` is finite, including uniformly tiny or near-overflow
    vectors for which derivative formation must be scaled; and ``lower`` and
    ``upper`` are finite with ``lower < upper``, so an inverted or degenerate
    interval is rejected rather than reordered.

    Parameters
    ----------
    c : np.ndarray
        Finite real loss coefficients in ascending powers of ``alpha``, with at
        least two entries.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval, strictly above
        ``lower``.

    Returns
    -------
    float
        The fitted coefficient: the admissible point selected by the ascending
        running-best scan.
    """
    return result  # noqa: F821 - required model stub
```

### Step 4

04_advance_inverse_newton

Goal
----
Apply one accelerated step to the residual-bearing member of the coupled pair.



The step multiplies `$M$` by the `$p$$-th power of$$I + alpha R$`, where `$R = I - M$` is the residual carried into the step. The partner `$X$` advances by the same factor on the right over the same step, and the tie `$M = X ** p A$` is what makes that consistent, but every reported quantity of the run depends on `$M$` alone, so `$X$` is not carried here.



The factor is raised to an integer power rather than applied `$p$$times to a product, so the step remains a pure matrix-multiplication kernel and the residual after it is exactly the polynomial in$$R$` that the scalar expansion describes.

```python
def advance_inverse_newton(M: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Advance the residual-bearing matrix by one accelerated step.

    Raises ``ValueError`` unless every one of the following holds: ``M`` is real
    rather than complex; ``M`` is a nonempty square two-dimensional array; every
    entry of ``M`` is finite; ``alpha`` is finite; and ``p`` is an integer, not a
    bool, with ``p >= 1``.

    Parameters
    ----------
    M : np.ndarray
        Finite real square matrix of shape ``(n, n)`` before the step.
    alpha : float
        Fitted coefficient for this step.
    p : int
        Root order.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the matrix after the step.
    """
    return result  # noqa: F821 - required model stub
```

### Step 5

05_coupled_partner_factor

Goal
----
*The iteration carries a pair, and every step so far has reported only the residual-bearing member `$M$`. The partner `$X$` is the object the method is actually computing: it converges to the inverse `$p$$-th root of the input, while$$M$` only certifies how far it has to go. This step returns the partner after a whole run of steps whose coefficients are already known.*



The two members start from the same positive scaling constant, the one that normalises the starting `$M$`. `$M$` starts from the input divided by the `$p$`-th power of that constant, as an earlier step returns; `$X$` starts from the identity divided by the constant itself, to the first power. That asymmetry between the two starting divisions is exactly what makes the tie `$M = X ** p A$` hold at the start.



Over one accelerated step `$M$` is multiplied by the `$p$$-th power of$$I + alpha R$`, where `$R = I - M$` is the residual carried into that step, while `$X$` is multiplied by the same factor only once. Because every iterate is a polynomial in the input, the factor commutes with `$X$`, and the tie is preserved step by step: raising the once-multiplied `$X$` to the `$p$$-th power reproduces the$$p$$-times-multiplied$$M$`. The residual driving each step must therefore be recomputed from the current `$M$`, not from the current `$X$`, and the two members have to advance in lockstep within a single loop.



The coefficients are supplied in the order they were fitted, one per step, and the number of steps is their count.



The two scaling operations have different homogeneity. Rescaling `$A$` leaves the carried `$M$` unchanged but rescales the starting partner by the inverse `$p$`-th root of that factor. Both must remain representable whenever the true float64 result is representable, even if a naive Frobenius norm of `$A$` is zero or infinite. As in the starting-matrix step, an `$A$` admitted by the symmetry tolerance represents its symmetric projection; the same projected matrix must be used for both members or the coupled tie is already broken at the start.

```python
def coupled_partner_factor(A: np.ndarray, alphas: np.ndarray, p: int) -> np.ndarray:
    """Return the partner member of the coupled pair after the supplied steps.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``alphas`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array; every entry of ``A`` is finite; ``A`` is symmetric to
    an absolute tolerance of ``1e-12`` and is then projected to its symmetric
    part; that symmetric part is nonzero, including when a naive Frobenius norm
    underflows or overflows;
    ``alphas`` is a nonempty one-dimensional array of finite values, so a run of
    no steps is rejected rather than returning the starting partner; and ``p`` is
    an integer, not a bool, with ``p >= 1``.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    alphas : np.ndarray
        Finite real fitted coefficients, one per step, in the order applied.
    p : int
        Root order. The iteration targets the inverse ``p``-th root of ``A``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the partner after every
        supplied step has been applied.
    """
    return result  # noqa: F821 - required model stub
```

### Step 6

06_fitted_coefficient_sequence

Goal
----
A whole run of the accelerated method is a single loop in which the state is the residual-bearing member of the coupled pair. Each pass reads the residual off the current state, contracts the post-step residual against the fixed sketch to get a scalar loss, minimises that loss over the closed coefficient interval, and advances the state by the accelerated step that the fitted coefficient defines.



This step reports the coefficient the fit selects at each pass, in order. The sketch is fixed once for the whole run rather than redrawn per iteration, so the run is deterministic and every coefficient after the first depends on all the coefficients before it; the trajectory is not a set of independent fits. Whether a given pass settles on an endpoint or on an interior stationary point is a property of the run, not something imposed in advance, and both outcomes occur on a typical trajectory.



The returned trajectory has one entry per step, so its length is the number of iterations requested and the starting state contributes no entry.



The trajectory depends on the direction and spectrum of `$A$` but not on its positive scalar magnitude: the scale is removed by the starting construction. That homogeneity and the symmetric projection used for tolerance-accepted inputs must survive composition of the earlier steps, including for finite matrices whose naive Frobenius norm is zero or infinite.

```python
def fitted_coefficient_sequence(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Return the fitted coefficient selected at each step of a whole run.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``S`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array that is finite, symmetric to an absolute tolerance of
    ``1e-12`` and projected to its nonzero symmetric part without a scale-unsafe
    norm; ``S`` is a finite two-dimensional
    array with at least one row and with its column count equal to the dimension
    of ``A``; ``lower`` and ``upper`` are finite with ``lower < upper``; ``p`` is
    an integer, not a bool, with ``p >= 1``; and ``iterations`` is an integer, not
    a bool, with ``iterations >= 1``.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval.
    iterations : int
        Number of accelerated steps to run.

    Returns
    -------
    np.ndarray
        Float64 vector of shape ``(iterations,)`` holding the fitted coefficient
        of each step in order.
    """
    return result  # noqa: F821 - required model stub
```

### Step 7

07_residual_trajectory

Goal
----
The quantity the method is judged on is not the coefficient it picks but how fast the residual falls. This step reports that fall: the Frobenius norm of the residual before any step has been taken, and again after each step of a whole run.



The residual is read off the residual-bearing member of the coupled pair by the defining relation, so no partner and no matrix inverse is ever formed, and the norm is the ordinary Frobenius norm of that residual. Because the starting state contributes the first entry and every step contributes one more, the trajectory is one entry longer than the coefficient trajectory of the same run; its first entry depends on the input and the order alone, and never on the sketch, the interval or the number of steps.



A correct run is strictly decreasing on a well-scaled input, and the decrease accelerates: the published scaling places the starting residual inside a region where the accelerated step contracts, and each fitted coefficient is chosen to make the very next entry as small as the sketch can see. The last entry is what a whole-run report reduces to.



Because the starting member is homogeneous of degree zero in `$A$`, this entire residual path is unchanged by a positive scalar rescaling of `$A$`. The path must retain that invariant across the finite float64 range and must use the symmetric projection of any input admitted by the absolute symmetry tolerance.

```python
def residual_trajectory(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Return the Frobenius residual norm before and after every step of a run.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``S`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array that is finite, symmetric to an absolute tolerance of
    ``1e-12`` and projected to its nonzero symmetric part without a scale-unsafe
    norm; ``S`` is a finite two-dimensional
    array with at least one row and with its column count equal to the dimension
    of ``A``; ``lower`` and ``upper`` are finite with ``lower < upper``; ``p`` is
    an integer, not a bool, with ``p >= 1``; and ``iterations`` is an integer, not
    a bool, with ``iterations >= 1``.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval.
    iterations : int
        Number of accelerated steps to run.

    Returns
    -------
    np.ndarray
        Float64 vector of shape ``(iterations + 1,)`` holding the Frobenius
        residual norm before any step and after each step.
    """
    return result  # noqa: F821 - required model stub
```

### Step 8

08_compute_inverse_root_residual

Goal
----
Report a whole run of the accelerated method as one number: the Frobenius norm of the residual left after the requested number of steps, rounded once at the reporting stage to twelve digits after the decimal point.



Everything the number depends on has already been built. The run is driven from the scaled starting state, each step fits its own coefficient against the fixed sketch and advances the state, and the residual path records the result; this step composes those pieces and takes the last entry of the path. Rounding happens once, at the end, and never inside the loop, because rounding an iterate would change the state that the next fit sees.



The run is also certified rather than merely executed. The partner member of the coupled pair is advanced over the same coefficients and the tie `$M = X ** p A$` is checked against the state the loop ends on, so a run whose two members have drifted apart is rejected instead of reported. Both inputs are dimensionless, so the reported residual is dimensionless.



The reported residual is homogeneous of degree zero in `$A$` even though the partner used for certification is not. The orchestrator must therefore preserve the scale-safe starting constructions and the symmetric projection of an input accepted by tolerance; replacing them locally with a naive norm is not an equivalent replay of the earlier steps.

```python
def compute_inverse_root_residual(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    """Return the rounded Frobenius residual left by a whole accelerated run.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``S`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array that is finite, symmetric to an absolute tolerance of
    ``1e-12`` and projected to its nonzero symmetric part without a scale-unsafe
    norm; ``S`` is a finite two-dimensional
    array with at least one row and with its column count equal to the dimension
    of ``A``; ``lower`` and ``upper`` are finite with ``lower < upper``; ``p`` is
    an integer, not a bool, with ``p >= 1``; ``iterations`` is an integer, not a
    bool, with ``iterations >= 1``; and the run stays finite with its two coupled
    members still tied at the end.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval.
    iterations : int
        Number of accelerated steps to run.

    Returns
    -------
    float
        The Frobenius residual norm after the final step, rounded to twelve
        digits after the decimal point.
    """
    return result  # noqa: F821 - required model stub
```
