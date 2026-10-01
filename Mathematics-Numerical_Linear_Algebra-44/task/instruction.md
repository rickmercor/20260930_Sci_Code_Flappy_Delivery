# Mathematics-Numerical_Linear_Algebra-44

## Background

Oblivious subspace embeddings reduce ambient dimension while approximately preserving norms on a fixed subspace. In the sketched metric one obtains an SVD-like factorization and a nearest \(S^\top S\)-orthogonal matrix. The reported scalar is fixed by how those sources relate polar gaps of that factor to the realized distortion.

## Problem

Sketching replaces the Euclidean metric by the semi-inner product induced by \(S^\top S\), where \(S\) is an oblivious subspace embedding. For tall matrices this yields an SVD-like factorization whose left factors are orthonormal in that sketched metric, and from it a nearest sketched-orthogonal matrix.

Take the configuration \(m=144\), \(n=8\), \(s=72\), sketch parameters \((p,q)=(149,23)\), and bandwidth \(h=1/(n-1)\). Form the tall Gaussian radial-basis matrix \(A\in\mathbb{R}^{m\times n}\) with entries
\[
A_{ij}=\frac{1}{\sqrt{m}}\exp\!\Big(-\frac{(t_i-c_j)^2}{2h^2}\Big)
\]
on the endpoint-inclusive grids
\[
t_i=\frac{i-1}{m-1}\ (i=1,\dots,m),\qquad c_j=\frac{j-1}{n-1}\ (j=1,\dots,n)
\]
of \([0,1]\). Build the deterministic sketch \(S=\sqrt{m/s}\,DFE\in\mathbb{R}^{s\times m}\) at those \((p,q)\), replacing only the usual random factors in the subsampled-trigonometric construction by these surrogates: with 1-based index \(i=1,\ldots,m\), set \(e_i=+1\) if \((i\bmod p)\) is a nonzero quadratic residue modulo \(p\) and \(e_i=-1\) otherwise (\(E=\mathrm{diag}(e_1,\ldots,e_m)\)); let \(D\) retain the rows of the identity indexed by \(\{(qk)\bmod m\}_{k=0}^{s-1}\) as **0-based** indices in \(\{0,\ldots,m-1\}\), listed in that stride order (not sorted); and let \(F\) be the discrete cosine transform factor from that construction. Ambient sizes and the scale \(\sqrt{m/s}\) are otherwise unchanged.

Let \(P\) be the nearest factor of \(A\) in the \(S^\top S\)-induced Frobenius norm among range-constrained \(S^\top S\)-orthonormal matrices (Frobenius uniqueness; spectral nearness need not coincide). Report the dimensionless ratio of \(P\)’s spectral sketched-to-Euclidean polar cross-gap to \(P\)’s spectral own-polar gap after the orthogonality scale fixed by the realized embedding distortion of \(S\) on \(\mathrm{Range}(A)\). How that scale is taken from the spectral orthogonality certificate is determined by the sources.

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

01_construct_kernel_matrix

Goal
----
The tall Gaussian RBF data matrix A.

```python
"""Step 1: """

import numpy as np

def construct_kernel_matrix(m, n, h=None):
    """m, n: integers with m >= n >= 1, the row and column counts. h: positive
    bandwidth, or None to use the default centre spacing: 1/(n-1) when n > 1,
    and 1.0 when n == 1 (the n-1 denominator is undefined). Returns (m, n)
    float64: the scaled Gaussian radial-basis matrix for this configuration.
    Raises ValueError for m < n, non-positive sizes, or a non-positive
    bandwidth."""
    return np.zeros((m, n))
```

### Step 2

02_build_sketch_operator

Goal
----
the deterministic subsampled trigonometric sketch operator.

```python
import numpy as np


def build_sketch_operator(m, s, p, q):
    """m: ambient row count; s: sketch dimension with 1 <= s <= m; p: a prime
    strictly greater than m; q: a positive integer. Returns (s, m) float64:
    the deterministic subsampled trigonometric sketch at these parameters.
    Raises ValueError for bad sizes, a non-prime or too-small p, or a q that
    does not yield s distinct selected rows."""
    return np.zeros((s, m))
```

### Step 3

03_compute_sts_singular_values

Goal
----
the S^T S-singular values of A.

```python
import numpy as np

def compute_sts_singular_values(A, S):
    """A: (m, n) tall matrix with m >= n; S: (s, m) sketch operator with
    s >= n. Returns (n,) float64: the S^T S-singular values of A in
    nonincreasing order. Raises ValueError on shape mismatch, s < n, or
    non-finite input."""
    return np.zeros(A.shape[1])
```

### Step 4

04_compute_sts_right_factor

Goal
----
the right orthogonal factor V of the S^T S-SVD.

```python
import numpy as np

def compute_sts_right_factor(A, S):
    """A: (m, n) tall matrix; S: (s, m) sketch operator with s >= n. Returns
    (n, n) float64: the orthogonal right factor V from the S^T S-SVD of A,
    with diagonal-preferring column signs and det(V) = +1. Raises
    ValueError on shape mismatch, s < n, or a rank-deficient sketch S A."""
    return np.zeros((A.shape[1], A.shape[1]))
```

### Step 5

05_form_nearest_sts_orthogonal

Goal
----
the nearest S^T S-orthogonal matrix P = W V^T.

```python
import numpy as np


def form_nearest_sts_orthogonal(A, S, V):
    """A: (m, n) tall matrix; S: (s, m) sketch operator with s >= n; V: (n, n)
    right factor from the S^T S-SVD under the instance sign convention.
    Returns (m, n) float64: the unique Frobenius-nearest matrix to A among
    factors with S^T S-orthonormal columns spanning Range(A). Raises
    ValueError on shape mismatch, s < n, a rank-deficient sketch, or a V
    that violates the instance sign convention."""
    return np.zeros_like(np.asarray(A, dtype=np.float64))
```

### Step 6

06_form_nearest_euclidean_orthogonal

Goal
----
the nearest Euclidean orthogonal matrix T (thin polar factor)

```python
import numpy as np

def form_nearest_euclidean_orthogonal(A):
    """A: (m, n) tall matrix of full column rank with m >= n. Returns (m, n)
    float64: the unique Frobenius-nearest matrix with orthonormal columns
    (a spectral-norm minimizer as well). Raises ValueError for m < n,
    non-finite entries, or a rank-deficient A."""
    return np.zeros_like(np.asarray(A, dtype=np.float64))
```

### Step 7

07_realized_embedding_distortion

Goal
----
the realized subspace-embedding distortion of S on Range(A).

```python
import numpy as np


def realized_embedding_distortion(A, S):
    """A: (m, n) tall full-column-rank matrix; S: (s, m) sketch operator with
    s >= n. Returns float: the realized spectral embedding distortion of S
    on Range(A) (the tightest eps for the Problem statement's squared-norm
    inequality on that subspace). Raises ValueError on shape mismatch,
    s < n, or a rank-deficient A."""
    return 0.0
```

### Step 8

08_orchestrate_sketched_polar_audit

Goal
----
the sketched-polar scaled gap ratio Xi.

```python
import numpy as np

def orchestrate_sketched_polar_audit(m, n, s, p, q, h=None):
    """m, n, s, p, q, h: instance parameters for the kernel matrix and
    deterministic sketch. Returns float: the dimensionless scalar from the
    Problem statement for this instance. Must obtain V from the S^T S
    right-factor step and pass it into the nearest-STS step. Raises
    ValueError on invalid parameters, a rank-deficient sketch, a distortion
    outside (0, 1), or a V that violates the instance sign convention."""
    return 0.0
```
