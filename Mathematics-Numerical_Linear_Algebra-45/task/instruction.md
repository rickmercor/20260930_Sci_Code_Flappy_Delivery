# Mathematics-Numerical_Linear_Algebra-45

## Background

The matrix step function of a symmetric matrix whose spectrum splits into two separated blocks is the spectral projector onto the block above the step. It is the central object of linear-scaling electronic structure theory, where it is the single-particle density matrix built from a Fock or Kohn-Sham matrix, and it is also the object behind the matrix sign function that appears in homomorphic comparison and in orthogonalisation steps of large optimisers. Methods that reach it using nothing but matrix addition, scaling and multiplication are attractive because those are the operations that parallelise well on dense hardware, so the cost of such a method is quoted as a count of non-scalar matrix-matrix multiplications, a figure that is independent of the architecture and the linear-algebra library and that also sets the critical path length of a parallel run.

The dominant family of such methods is the recursive polynomial expansion: the iterate is repeatedly replaced by the image of a low-degree component polynomial, so that after a handful of iterations the composite polynomial has enormous degree at very small cost. Classical members of the family are the McWeeny purification polynomial and the second-degree spectral projection scheme, both of which use component polynomials that are monotone on the unit interval and fix its two endpoints. Faster convergence is obtained by relaxing monotonicity and by feeding the recursion bounds on the two eigenvalues that border the gap, and a further gain comes from raising the degree of the component polynomial, provided that degree can still be evaluated in few multiplications. The relevant constraint here is not the degree itself but the number of non-scalar multiplications an evaluation requires, and how large a class of polynomials of that degree a given multiplication budget can actually represent.

In exact dense arithmetic, any factorisation of the same polynomial gives the same matrix and its trace can be recovered by mapping the eigenvalues independently. Sparse matrix implementations change that equivalence: small entries are commonly discarded to control fill-in, and discarding an intermediate product does not commute with diagonalisation or with the later scaled additions. Once a deterministic entry filter is inserted after every raw product, the evaluation schedule and the eigenvector basis become part of the numerical problem. Symmetrising each raw product before filtering preserves the self-adjoint structure needed by the projector iteration, while a strict threshold and a fixed basis make the resulting perturbation reproducible.

Progress through such a run is not described by a single quantity. During the initial phase the eigenvalues are still far from $0$ and $1$ and the useful measure is the condition number of the step-function problem, which equals the reciprocal of the separation between the two interior eigenvalues, so a run that never lets that separation shrink below its starting value is forward stable. Only when the condition number approaches unity does the deviation from idempotency start to fall, and from that point on the asymptotic order of convergence of the component polynomial governs the run. Because the two phases respond to different properties of the component polynomial, choosing the polynomial applied at each iteration involves a trade-off: the choice that widens the gap most in the current iteration also moves the step to a new location, and that new location fixes how much amplification is available in the iteration after it.

## Problem

A recursive polynomial expansion for a symmetric matrix step function has a sharp crossover between spending its multiplication budget on conditioning the spectral gap and spending it on purification toward an idempotent projector. Use the non-accelerated degree-eight expansion from the relevant literature, with exact interior eigenvalue bounds, trivial outer bounds $[0,1]$, the published component-polynomial family and member-selection rule, and the published memory-efficient matrix evaluation.

For each trial gap width $\xi$, take $n=120$, put the step at $\mu=0.35$, set the two interior eigenvalues to $\mu\pm\xi/2$, place $\operatorname{round}(n(1-\mu))$ eigenvalues equidistantly from the upper interior eigenvalue to $1$, and place the rest equidistantly from $0$ to the lower interior eigenvalue. Both endpoints of each equidistant block are included; if the resulting ordered eigenvalue vector is $\lambda$, start from
$$
X_0=Q\operatorname{diag}(\lambda)Q^{\mathsf T},\qquad
Q_{jk}=\sqrt{\frac{2}{n+1}}\sin\!\left(\frac{\pi jk}{n+1}\right),
\quad 1\le j,k\le n.
$$

Model sparse-product error inside the published evaluator as follows. For every raw non-scalar product $P=AB$ formed by that evaluator, and before any subsequent scaled-matrix addition, replace $P$ by $\mathcal F_\tau(P)$ with $\tau=10^{-6}$, where
$$
[\mathcal F_\tau(P)]_{jk}=
\begin{cases}
0,&\left|\tfrac12(P_{jk}+P_{kj})\right|<\tau,\\
\tfrac12(P_{jk}+P_{kj}),&\text{otherwise}.
\end{cases}
$$
Do not filter the scaled-matrix additions; starting from $X_0$, spend exactly $30$ non-scalar matrix-matrix multiplications and define $E(\xi)=\operatorname{Tr}(X-X^2)$ for the final iterate.

Determine the unique $\xi_*\in[2.5\times10^{-4},3.5\times10^{-4}]$ for which $E(\xi_*)=0.1$ and report it to at least ten significant digits. In the short reasoning give $E(2.5\times10^{-4})$, $E(3.0\times10^{-4})$, $E(3.5\times10^{-4})$, and $D_-$, the cumulative number of entries set to zero by $\mathcal F_\tau$ over every raw product in the run at $\xi=2.5\times10^{-4}$; count $(j,k)$ and $(k,j)$ separately and count an entry each time it satisfies the strict threshold test.

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

01_sp8_family_coefficients

Goal
----
Return the monomial coefficients of one member of the eight-member family of degree-eight component polynomials used by the recursive expansion when no acceleration is applied.

```python
import numpy as np

def sp8_family_coefficients(index: int) -> np.ndarray:
    """Return the monomial coefficients of one component polynomial of the family.

    Parameters
    ----------
    index : int
        Member index L of the family, an integer with 0 <= L <= 7.

    Returns
    -------
    coeffs : np.ndarray
        Shape (9,) float array [b0, b1, ..., b8] such that the component
        polynomial is p(x) = b0 + b1*x + ... + b8*x**8.

    Raises
    ------
    ValueError
        If ``index`` is not an integer (a bool counts as not an integer), or
        if it falls outside the range 0 <= index <= 7. The function must
        raise rather than return a placeholder or clamp the index.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(9, dtype=float)  # placeholder
```

### Step 2

02_select_family_member

Goal
----
Choose the component polynomial applied at one iteration of the non-accelerated recursion and return the images of the two interior eigenvalues under it.

```python
import numpy as np

def select_family_member(lam_lumo: float, lam_homo: float, iteration: int) -> np.ndarray:
    """Select one component polynomial and advance the interior eigenvalues.

    Parameters
    ----------
    lam_lumo : float
        Largest eigenvalue below the step, with 0 <= lam_lumo < lam_homo.
    lam_homo : float
        Smallest eigenvalue above the step, with lam_homo <= 1.
    iteration : int
        One-based index of the recursion iteration (iteration >= 1).

    Returns
    -------
    state : np.ndarray
        Shape (3,) float array [index, new_lam_lumo, new_lam_homo], where
        ``index`` is the selected family member stored as a float and the
        remaining two entries are the images of the interior eigenvalues
        under the selected component polynomial.

    Raises
    ------
    ValueError
        If ``lam_lumo`` or ``lam_homo`` is not a real number or is not
        finite; if they do not satisfy 0 <= lam_lumo < lam_homo <= 1; if
        ``iteration`` is not an integer (a bool counts as not an integer);
        or if ``iteration`` is less than 1. The function must raise rather
        than return a placeholder or reorder the two eigenvalues itself.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(3, dtype=float)  # placeholder
```

### Step 3

03_degree_eight_evaluation_coefficients

Goal
----
Solve for the ten evaluation coefficients that let an arbitrary degree-eight matrix polynomial be assembled from three non-scalar multiplications.

```python
import numpy as np

def degree_eight_evaluation_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """Solve the coefficient-matching system of the three-multiplication scheme.

    Parameters
    ----------
    coeffs : np.ndarray
        Shape (9,) array [b0, b1, ..., b8] of monomial coefficients of a
        polynomial of degree exactly eight, so that b8 is non-zero.

    Returns
    -------
    evaluation : np.ndarray
        Shape (10,) float array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.

    Raises
    ------
    ValueError
        If ``coeffs`` is not a one-dimensional array of length 9, if any of
        its entries is not finite, or if its leading entry b8 is zero, in
        which case the polynomial is not of degree exactly eight and the
        scheme is undefined. The function must raise rather than return a
        placeholder, a NaN or a lower-degree fallback.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(10, dtype=float)  # placeholder
```

### Step 4

04_workspace_rearrangement_scalars

Goal
----
Compute the four auxiliary scalars that let the three-multiplication evaluation be carried out inside a workspace of only three matrices.

```python
import numpy as np

def workspace_rearrangement_scalars(evaluation: np.ndarray) -> np.ndarray:
    """Compute the workspace rearrangement scalars of the evaluation scheme.

    Parameters
    ----------
    evaluation : np.ndarray
        Shape (10,) array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.

    Returns
    -------
    scalars : np.ndarray
        Shape (4,) float array [r1, r2, r3, r4] of rearrangement scalars.

    Raises
    ------
    ValueError
        If ``evaluation`` is not a one-dimensional array of length 10, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or pad a short input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(4, dtype=float)  # placeholder
```

### Step 5

05_apply_degree_eight_polynomial

Goal
----
Apply a degree-eight matrix polynomial to a square matrix using exactly three non-scalar multiplications and three matrix-sized workspace slots.

```python
import numpy as np

def apply_degree_eight_polynomial(matrix: np.ndarray, evaluation: np.ndarray,
                                  scalars: np.ndarray,
                                  drop_tolerance: float = 0.0) -> np.ndarray:
    """Evaluate a degree-eight matrix polynomial with three matrix products.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n, n) square float array, the argument of the polynomial.
    evaluation : np.ndarray
        Shape (10,) array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.
    scalars : np.ndarray
        Shape (4,) array [r1, r2, r3, r4] of workspace rearrangement scalars.
    drop_tolerance : float
        Finite nonnegative threshold. Zero disables filtering. When positive,
        every raw product P is symmetrized as 0.5*(P + P.T), then entries
        whose absolute value is strictly below the threshold are set to zero,
        before any subsequent scaled-matrix addition.

    Returns
    -------
    result : np.ndarray
        Shape (n, n) float array holding the polynomial evaluated at
        ``matrix``.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square two-dimensional array or has
        a non-finite entry; if ``evaluation`` is not a finite
        one-dimensional array of length 10; if ``scalars`` is not a finite
        one-dimensional array of length 4; if ``drop_tolerance`` is not a
        finite nonnegative real scalar; or if filtering is requested for a
        matrix that is not symmetric to absolute tolerance 1e-10. The
        function must raise rather than return a placeholder or broadcast a
        rectangular input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros_like(np.asarray(matrix, dtype=float))  # placeholder
```

### Step 6

06_equidistant_step_spectrum

Goal
----
Build the eigenvalue list of a normalised test problem with a prescribed step location and a prescribed gap around it.

```python
import numpy as np

def equidistant_step_spectrum(size: int, mu: float, gap: float) -> np.ndarray:
    """Build the eigenvalue list of the normalised test problem.

    Parameters
    ----------
    size : int
        Number of eigenvalues n, an integer with n >= 2.
    mu : float
        Step location, with 0 < mu < 1.
    gap : float
        Width of the gap around the step, with 0 < gap and the resulting
        interior eigenvalues strictly inside (0, 1).

    Returns
    -------
    eigenvalues : np.ndarray
        Shape (n,) float array of eigenvalues in ascending order, holding the
        block below the step followed by the block above it.

    Raises
    ------
    ValueError
        If ``size`` is not an integer (a bool counts as not an integer) or
        is less than 2; if ``mu`` or ``gap`` is not a real finite number; if
        ``mu`` does not satisfy 0 < mu < 1; if ``gap`` is not positive; if
        the resulting interior eigenvalues do not lie strictly inside
        (0, 1); or if either block would hold fewer than two eigenvalues.
        The function must raise rather than return a placeholder or clip the
        spectrum back into the unit interval.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(int(size), dtype=float)  # placeholder
```

### Step 7

07_symmetric_matrix_from_spectrum

Goal
----
Turn a list of eigenvalues into a dense symmetric matrix by conjugating it with a fixed orthogonal transform.

```python
import numpy as np

def symmetric_matrix_from_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    """Build a dense symmetric matrix with the prescribed spectrum.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Shape (n,) array of real eigenvalues, with n >= 1.

    Returns
    -------
    matrix : np.ndarray
        Shape (n, n) symmetric float array whose eigenvalues are the entries
        of ``eigenvalues``.

    Raises
    ------
    ValueError
        If ``eigenvalues`` is not a non-empty one-dimensional array, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or flatten a two-dimensional input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((len(eigenvalues), len(eigenvalues)), dtype=float)  # placeholder
```

### Step 8

08_idempotency_trace

Goal
----
Measure how far a symmetric iterate is from a projector by evaluating the trace of the difference between the matrix and its square.

```python
import numpy as np

def idempotency_trace(matrix: np.ndarray) -> float:
    """Evaluate the trace of the difference between a matrix and its square.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n, n) square symmetric float array.

    Returns
    -------
    measure : float
        The trace of ``matrix`` minus the trace of its square, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square two-dimensional array, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or a NaN.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```

### Step 9

09_critical_initial_gap

Goal
----
Chain the sub-problem functions 01-08 end to end and return the idempotency measure of the iterate reached once a given multiplication budget has been spent.

```python
import numpy as np


def critical_initial_gap(size: int = 120, mu: float = 0.35,
                         multiplications: int = 30,
                         target_trace: float = 0.1,
                         gap_lower: float = 2.5e-4,
                         gap_upper: float = 3.5e-4,
                         bisection_steps: int = 40,
                         drop_tolerance: float = 1.0e-6) -> float:
    """Invert the fixed-budget idempotency trace on a supplied gap bracket.

    Parameters
    ----------
    size : int
        Number of eigenvalues in the equidistant test spectrum.
    mu : float
        Step location strictly inside the unit interval.
    multiplications : int
        Non-scalar matrix-matrix multiplication budget, at least 3.
        Any remainder after division by three is left unused.
    target_trace : float
        Finite target value of the final idempotency trace.
    gap_lower, gap_upper : float
        Finite positive endpoints with gap_lower < gap_upper. Both gaps must
        define valid test spectra, and their final trace residuals must have
        opposite signs (an endpoint residual equal to zero is accepted).
    bisection_steps : int
        Positive number of bisection updates to perform.
    drop_tolerance : float
        Finite nonnegative threshold passed to every component-polynomial
        application. Zero disables intermediate-product filtering.

    Returns
    -------
    gap : float
        Midpoint of the final bisection bracket, as a native Python float.

    Raises
    ------
    ValueError
        If an integer parameter is invalid; if a floating parameter is not
        real and finite; if ``drop_tolerance`` is negative; if the gap
        endpoints are not positive and ordered; if either endpoint does not
        define a valid spectrum; or if the two endpoint residuals do not
        bracket zero.

    Notes
    -----
    This is the final orchestrating step. Each residual evaluation must call
    the public functions of sub-problems 01-08 and feed their returned values
    through the matrix recursion. Include every import needed by the function
    body. Do not substitute a separately coded scalar-spectrum recurrence.
    """
    return 0.0  # placeholder
```
