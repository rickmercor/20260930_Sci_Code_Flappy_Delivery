# Mathematics-Numerical_Linear_Algebra-48

## Background

Regularized log-determinants \(\log\det(A+I)\) appear throughout Gaussian-process marginal likelihoods, determinantal point processes, and covariance-kernel statistics. Direct evaluation is cubic in dimension, so modern randomized numerical linear algebra replaces factorization by matrix–vector products: Hutchinson-type trace estimation combined with Lanczos quadrature for quadratic forms of \(\log(A+I)\).

When \(A\) is numerically low-rank, a Nyström sketch \(\widehat{A}\) yields a cheap SPD preconditioner \(\widehat{P}=\widehat{A}+I\). Multiplicativity of the determinant then splits
\[
\operatorname{tr}\log(A+I)=\operatorname{tr}\log\widehat{P}+\operatorname{tr}\log\!\bigl(\widehat{P}^{-1/2}(A+I)\widehat{P}^{-1/2}\bigr).
\]
The first term is exact from the sketch eigenvalues; the second lives on a better-conditioned matrix and can often be estimated with surprisingly few probes. Recent work shows that allocating almost the entire matvec budget to the Nyström stage and using a single Gaussian probe for the residual is highly effective whenever spectral decay is at least moderate, and proposes an adaptive “detective” test—comparing residual diagnostics at two nested sketch ranks, computed without spending additional matrix–vector products to detect the rare slow-decay regime and reallocate probes to stochastic Lanczos quadrature.

Complementary theory for operator-monotone matrix functions supplies nuclear-norm certificates for the residual \(\log(A+I)-\log(\widehat{A}+I)\). Combining the detective budget allocation with such a certificate produces a single scalar that simultaneously reflects algorithmic branching, stochastic quadrature, and a priori residual control precisely the quantitative object computed in this task.

## Problem

Regularized log-determinants of large SPSD matrices are routinely approximated from matrix–vector products by combining a randomized low-rank preconditioner with a stochastic Krylov estimate of the residual. When nearly the entire matvec budget is spent on the preconditioner, a single residual probe is often enough—but only if a nested residual diagnostic confirms that further sketch enrichment still reduces the residual; otherwise the residual budget must be redistributed across several probes. The computational object of interest is one deterministic scalar assembled from that adaptive estimate and a nuclear-norm certificate for the logarithm residual of an operator-monotone matrix function.

Construct the \(n=32\) diagonal SPSD instance with eigenvalues \(\lambda_i=i^{-2}/\mu\), \(\mu=10^{-2}\). Allocate matvecs with \((\ell,m,\beta)=(16,5,3/4)\) to form a Gaussian Nyström preconditioner, decide adaptively how many residual probes to use, and estimate \(\operatorname{tr}\log(A+I)\) via the exact preconditioner log-determinant plus Lanczos quadrature on the symmetrically preconditioned residual. The two sketches compared by the diagnostic are nested: the coarse one is the leading block of columns of the fine one, and each Nyström core is stabilized by a shift of order machine epsilon times \(\|A\Omega\|_2\). Draw every Gaussian from one Generator seeded \(0\), advanced in algorithmic order: the diagnostic sketch first, then the enlargement block, then the residual probe. Separately form the expectation-style nuclear residual bound for \(f(x)=\log(1+x)\) at oversampling \(p=2\) and target rank \(k=r-p\), where \(r\) is the sketch width retained after the adaptive decision. Return the estimate plus \(10^{-3}\) times that bound.

Do not replace the adaptive budget rule, the residual quadrature, or the nuclear certificate by a single dense eigendecomposition of \(A+I\).

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

01_construct_scaled_logdet

Goal
----
Build the scaled algebraic SPSD diagonal spectrum and return its exact regularized log-determinant tr log(A+I).

```python
import numpy as np


def construct_scaled_logdet(n: int, mu: float) -> float:
    """Build A = diag(i^{-2}/mu) and return tr log(A+I).

    Parameters
    ----------
    n : int
        Matrix order.
    mu : float
        Positive regularization scale used in A = H/mu.

    Returns
    -------
    float
        Exact value of sum_i log(1 + lambda_i(A)).
        
    Raises
    ------
    ValueError
        If ``n < 1``, or if ``mu <= 0``.
    """
    return 0.0
```

### Step 2

02_nystrom_frobenius_norm

Goal
----
Stabilized Nyström approximation; return the Frobenius norm of Ahat.

```python
import numpy as np


def nystrom_frobenius_norm(A: np.ndarray, Omega: np.ndarray) -> float:
    """Compute ||Ahat||_F for the stabilized Nyström sketch of A with sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Gaussian sketching matrix.

    Returns
    -------
    float
        Frobenius norm of the Nyström approximant Ahat.

    Raises
    ------
    ValueError
        If ``A`` is not a two-dimensional square array; if ``Omega`` is not
        two-dimensional with ``Omega.shape[0] == A.shape[0]``; if ``Omega``
        has no columns; or if ``A`` or ``Omega`` contains a non-finite entry.
    """
    return 0.0
```

### Step 3

03_leave_one_out_errF2

Goal
----
Leave-one-out squared Frobenius Nyström residual estimator.

```python
import numpy as np


def leave_one_out_errF2(A: np.ndarray, Omega: np.ndarray) -> float:
    """Leave-one-out estimate of ||A - Ahat||_F^2 for Nyström sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Sketching matrix with s >= 2.

    Returns
    -------
    float
        Estimated squared Frobenius residual of the Nyström approximation.

    Raises
    ------
    ValueError
        If ``A`` is not a two-dimensional square array; if ``Omega`` is not
        two-dimensional with ``Omega.shape[0] == A.shape[0]``; if ``A`` or
        ``Omega`` contains a non-finite entry; or if ``Omega`` has fewer than
        two columns, since each replicate leaves one column out.
    """
    return 0.0
```

### Step 4

04_detective_switch

Goal
----
Detective switching test; return 1.0 iff the one-sample branch is selected.

```python
import numpy as np


def detective_one_sample_flag(
    errF2_fine: float,
    errF2_coarse: float,
    ell: int,
    m: int,
    beta: float,
) -> float:
    """Return 1.0 if the one-sample branch is chosen, else 0.0.

    

    Parameters
    ----------
    errF2_fine : float
        Leave-one-out squared residual at rank floor(beta*ell).
    errF2_coarse : float
        Leave-one-out squared residual at rank floor(beta^2*ell).
    ell, m : int
        Budget parameters.
    beta : float
        Detective fraction in (0, 1).

    Returns
    -------
    float
        1.0 for one-sample, 0.0 for alpha-rank.

    Raises
    ------
    ValueError
        If ``beta`` is not strictly inside ``(0, 1)``, or if ``ell < 1`` or
        ``m < 1``.
    """
    return 0.0
```

### Step 5

05_preconditioner_logdet

Goal
----
Exact preconditioner contribution t1 = tr log(Ahat + I).

```python
import numpy as np


def preconditioner_logdet(A: np.ndarray, Omega: np.ndarray) -> float:
    """Return tr log(Ahat + I) for the Nyström approximant of A with sketch Omega.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    Omega : ndarray, shape (n, s)
        Sketching matrix.

    Returns
    -------
    float
        The preconditioner log-determinant tr log(Ahat + I).

    Raises
    ------
    ValueError
        If ``A`` is not a two-dimensional square array; if ``Omega`` is not
        two-dimensional with ``Omega.shape[0] == A.shape[0]``; if ``Omega``
        has no columns; or if ``A`` or ``Omega`` contains a non-finite entry.
    """
    return 0.0
```

### Step 6

06_slq_preconditioned_quadratic

Goal
----
One-probe Lanczos quadrature for w^T log(M) w on the preconditioned matrix.

```python
import numpy as np


def slq_preconditioned_quadratic(
    A: np.ndarray, Omega: np.ndarray, w: np.ndarray, m: int
) -> float:
    """Approximate w^T log(M) w with m Lanczos steps, M = P^{-1/2}(A+I)P^{-1/2}.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        SPSD matrix.
    Omega : ndarray, shape (n, s)
        Sketch defining Ahat and P = Ahat + I.
    w : ndarray, shape (n,)
        Probe vector.
    m : int
        Number of Lanczos steps.

    Returns
    -------
    float
        Quadrature approximation of w^T log(M) w.

    Raises
    ------
    ValueError
        If ``m < 1``; if ``A`` is not a two-dimensional square array; if
        ``Omega`` is not two-dimensional with ``Omega.shape[0] == A.shape[0]``;
        if ``Omega`` has no columns; if ``w`` does not have length
        ``A.shape[0]``; or if ``A``, ``Omega`` or ``w`` contains a non-finite
        entry.
    """
    return 0.0
```

### Step 7

07_nuclear_residual_certificate

Goal
----
Nuclear-norm residual certificate from the operator-monotone log bound.

```python
import numpy as np

def nuclear_residual_certificate(lam: np.ndarray, r: int, p: int) -> float:
    """Return the nuclear-norm residual certificate R for f(x) = log(1+x).

    The target rank is determined by the accepted Nyström rank r and the
    oversampling p. Eigenvalues are sorted internally into nonincreasing order.

    Parameters
    ----------
    lam : ndarray, shape (n,)
        Eigenvalues of A.
    r : int
        Nyström rank used by the accepted detective branch.
    p : int
        Oversampling parameter (>= 2).

    Returns
    -------
    float
        Nuclear-norm residual certificate R.

    Raises
    ------
    ValueError
        If ``p < 2``; if ``r < p``; or if ``lam`` has fewer than ``r``
        entries.
    """
    return 0.0
```

### Step 8

08_detective_logdet_orchestrator

Goal
----
detective Nyström+SLQ estimate plus scaled nuclear certificate.

```python
import numpy as np


def detective_logdet_certificate(
    n: int = 32,
    mu: float = 1e-2,
    ell: int = 16,
    m: int = 5,
    beta: float = 0.75,
    seed: int = 0,
    p: int = 2,
    cert_scale: float = 1e-3,
) -> float:
    """Run the detective Nyström+SLQ estimator and return E_hat + cert_scale * R.

    Parameters
    ----------
    n : int
        Matrix order for the algebraic diagonal instance.
    mu : float
        Regularization scale in A_ii = i^{-2}/mu.
    ell : int
        Nyström matvec budget parameter.
    m : int
        Lanczos depth / residual matvec block size.
    beta : float
        Detective fraction in (0, 1).
    seed : int
        NumPy Generator seed; all Gaussians drawn in algorithmic order.
    p : int
        Oversampling for the nuclear-norm certificate (k = r - p).
    cert_scale : float
        Multiplier for the certificate R in the returned scalar.

    Returns
    -------
    float
        E_hat + cert_scale * R.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``mu <= 0``; if ``beta`` is not strictly inside
        ``(0, 1)``; if ``ell < 1`` or ``m < 1``; if either nested rank
        ``floor(beta * ell)`` or ``floor(beta**2 * ell)`` is below 2, so the
        leave-one-out diagnostic cannot be formed; if the alpha-rank branch
        is selected and ``floor((ell + m - floor(beta * ell)) / m) < 1``
        leaves no residual probe; or if the assembled preconditioner
        violates the Loewner-order consequences of ``0 <= Ahat <= A``, namely
        ``||Ahat||_F > ||A||_F`` or ``tr log(Ahat + I) > tr log(A + I)``
        beyond a relative tolerance of 1e-8, or either quantity is
        non-finite.
    """
    return 0.0
```
