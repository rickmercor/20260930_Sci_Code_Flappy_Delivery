# Physics-Condensed_Matter_Physics-19

## Background

Last-passage percolation with geometric waiting times is an exactly solvable zero-temperature directed polymer in the Kardar–Parisi–Zhang class, and making the waiting-time matrix symmetric moves its fluctuations from the unitary to the orthogonal random-matrix universality class. Beyond the last-passage time itself, the full Robinson–Schensted–Knuth shape of the matrix carries the statistics of several non-intersecting polymers.

The task below asks for an exact finite-size statistic of that shape.

## Problem

I study a symmetric version of geometric last-passage percolation, a zero-temperature directed polymer whose waiting-time matrix $W$ is $20\times20$ and symmetric, $W_{ij}=W_{ji}$, with independent entries on and above the diagonal. Each off-diagonal entry is geometric with success probability 0.8, $\mathbb{P}(W_{ij}=k)=0.8\cdot0.2^k$ for $k=0,1,2,\ldots$, and each diagonal entry has $\mathbb{P}(W_{ii}=k)=(1-\sqrt{0.2})\,0.2^{k/2}$. Let $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_{20}\ge0$ be the rows of the shape that the Robinson–Schensted–Knuth correspondence assigns to $W$, so that $\lambda_1$ is the last-passage time from $(1,1)$ to $(20,20)$. I want the exact value of $\mathbb{P}(\lambda_2=21)$, not a simulation estimate, reported to ten significant figures.

The numerical defaults are $q=0.2$, $N=20$, target row length 21, and 120 sites for the shifted-row representation; the reorthogonalization threshold is $\eta=0.75$. The target is the infinite-support probability, with the finite cutoff used as a numerical approximation.

Report the probability that $\lambda_i-i=19$ for some row index $i$, the conditional probability given that event that $\lambda_1=20$, the conditional probability given that event that $\lambda_2=21$, and the final probability, each to six significant figures in the supporting reasoning. Identify the printed ingredients of the published exact Pfaffian sampler and its polynomial construction that require correction for this calculation, and state their corrected forms. Include the shape-law argument and the relation between the shifted-row event and the requested probability.

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

01_build_skew_inner_product

Goal
----
Build the matrix of the skew-symmetric bilinear form under which the skew-orthogonal polynomials of the shifted RSK rows of a symmetric geometric waiting-time matrix are defined.

```python
import numpy as np
def build_skew_inner_product(q: float, n_sites: int) -> np.ndarray:
    """Return the matrix of the beta = 1 skew form on the sites 0, 1, ..., n_sites - 1.

    The waiting-time matrix is symmetric with independent entries on and above
    the diagonal: an off-diagonal entry takes the value k = 0, 1, 2, ... with
    probability ``(1 - q) * q**k`` and a diagonal entry with probability
    ``(1 - sqrt(q)) * q**(k / 2)``. For the RSK shape ``lambda`` of an N x N
    such matrix, the shifted rows ``h_i = lambda_i + N - i`` are distinct
    non-negative integers with joint law proportional to
    ``prod_{i<j} |h_i - h_j| * prod_i w(h_i)`` for a per-site weight ``w``,
    normalised here to ``w(0) = 1``. Return the matrix ``G`` of the discrete
    beta = 1 skew form associated with this ensemble, represented so that the
    skew product of two sampled functions is ``<f, g> = f @ G @ g``.
    Use the sign-kernel convention ``epsilon(x, y) = 0.5 * sign(y - x)``:
    its value is +0.5 for x < y, zero for x = y, and -0.5 for x > y.

    Parameters
    ----------
    q : float
        Geometric ratio of the off-diagonal entries, ``0 < q < 1``.
    n_sites : int
        Number of sites ``0, 1, ..., n_sites - 1``; at least 2.

    Returns
    -------
    np.ndarray
        Real skew-symmetric array of shape ``(n_sites, n_sites)``.

    Raises
    ------
    ValueError
        If ``q`` is not a finite real number in the open interval (0, 1)
        (booleans are rejected), or if ``n_sites`` is not an integer of at
        least 2 (booleans are rejected).
    """
    return gram
```

### Step 2

02_skew_orthonormalize

Goal
----
Reduce one new vector against a partially built symplectic basis so that it becomes skew-orthogonal to every complete pair, either opening a new pair or completing the unpaired last vector.

```python
import numpy as np
def skew_orthonormalize(
    basis: np.ndarray,
    vector: np.ndarray,
    gram: np.ndarray,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Skew-orthonormalize one vector against a partially built symplectic basis.

    The skew product is ``<f, g> = f @ gram @ g``. The ``n`` columns
    ``s_1, ..., s_n`` of ``basis`` (in this 1-based labelling ``s_i`` is
    ``basis[:, i - 1]``) are grouped into consecutive complete pairs
    ``(s_1, s_2), (s_3, s_4), ...`` with ``<s_{2k-1}, s_{2k}> = 1`` and zero
    skew product between columns of different pairs; when ``n`` is odd the
    last column ``s_n`` is unpaired and has zero skew product with every
    earlier column. Return ``(v, h)`` with ``h`` of length ``n + 1`` such that
    ``vector = sum_{i=1}^{n} h_i s_i + h_{n+1} v`` and ``v`` has zero skew
    product with every column of every complete pair, where

    * if ``n`` is even, ``h_{n+1} = 1`` (``v`` opens a new pair and is not
      rescaled);
    * if ``n`` is odd, ``h_n = 0`` and ``<s_n, v> = 1`` (``v`` completes the
      pair of ``s_n``).

    The reduction against the complete pairs is applied in passes, and a new
    pass is made while the previous pass shrank the Euclidean norm of the
    working vector below ``eta`` times its value before that pass; at least
    one pass is always made. The inputs are not modified.

    Parameters
    ----------
    basis : np.ndarray
        Array of shape ``(L, n)`` with ``n >= 0`` columns as described above.
    vector : np.ndarray
        Array of shape ``(L,)`` to be reduced.
    gram : np.ndarray
        Real skew-symmetric array of shape ``(L, L)`` defining the skew product.
    eta : float
        Reorthogonalization threshold, ``0 < eta <= 1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(v, h)``: the new basis vector of shape ``(L,)`` and the coefficient
        vector of shape ``(n + 1,)``, ``h[i - 1]`` holding ``h_i``.

    Raises
    ------
    ValueError
        If ``basis`` is not two-dimensional, ``vector`` is not one-dimensional
        with the same length ``L`` as the columns of ``basis``, ``gram`` does
        not have shape ``(L, L)``, any entry is non-finite, ``eta`` is not a
        real number with ``0 < eta <= 1`` (booleans are rejected), or ``n`` is
        odd and ``<s_n, vector>`` is zero, so that the pair cannot be
        completed.
    """
    return v, h
```

### Step 3

03_run_symplectic_arnoldi

Goal
----
Generate a skew-orthonormal polynomial basis on a set of nodes, together with the upper Hessenberg matrix of its multiplication-by-x recurrence.

```python
import numpy as np
def run_symplectic_arnoldi(
    gram: np.ndarray,
    nodes: np.ndarray,
    n_vectors: int,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a skew-orthonormal Krylov basis and its Hessenberg recurrence matrix.

    Functions are represented by their values at the ``L`` nodes and the skew
    product is ``<f, g> = f @ gram @ g``. Column 0 of ``S`` is the constant
    function 1. Use the normalization and reorthogonalization conventions of
    ``skew_orthonormalize``. ``H`` is upper Hessenberg and records the
    multiplication-by-node recurrence for the returned basis. Hence
    ``nodes[:, None] * S[:, :-1] = S @ H``, column ``j`` is a polynomial of
    degree ``j`` in the node values, and consecutive columns ``(2k, 2k + 1)``
    form skew-orthonormal pairs.

    Parameters
    ----------
    gram : np.ndarray
        Real skew-symmetric array of shape ``(L, L)``.
    nodes : np.ndarray
        Node values, shape ``(L,)``.
    n_vectors : int
        Number of basis columns, ``1 <= n_vectors <= L``.
    eta : float
        Reorthogonalization threshold passed to each reduction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(S, H)`` with shapes ``(L, n_vectors)`` and
        ``(n_vectors, n_vectors - 1)``.

    Raises
    ------
    ValueError
        If ``gram`` is not a finite ``(L, L)`` array, ``nodes`` is not a
        finite array of shape ``(L,)``, ``n_vectors`` is not an integer with
        ``1 <= n_vectors <= L`` (booleans are rejected), or any reduction
        raises ``ValueError``.
    """
    return basis, hessenberg
```

### Step 4

04_assemble_pfaffian_kernel

Goal
----
Assemble the 2 x 2 matrix-valued Pfaffian kernel of a finite beta = 1 discrete particle ensemble from its skew-orthonormal polynomials.

```python
import numpy as np
def assemble_pfaffian_kernel(polys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a Pfaffian correlation kernel of a finite beta = 1 ensemble on sites.

    The sites are ``x = 0, 1, ..., L - 1`` with positive weights ``w(x)`` given
    by ``weights``. Column ``k`` of ``polys`` holds the values at the sites of a
    polynomial ``R_k`` of degree ``k``, ``k = 0, ..., N - 1`` with ``N`` even,
    and the family is skew-orthonormal under the skew product
    ``<f, g> = sum_{x, y} f(x) g(y) * 0.5 * sign(y - x) * w(x) * w(y)``:
    ``<R_{2k}, R_{2k+1}> = 1`` and every other skew product between members of
    different pairs vanishes. The ensemble places ``N`` particles on distinct
    sites with probability proportional to
    ``prod_{i<j} |h_i - h_j| * prod_i w(h_i)``.

    Return a real skew-symmetric ``(2L, 2L)`` array ``K`` whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x`` and such that, for every
    set ``X`` of distinct sites, the Pfaffian of the principal submatrix of
    ``K`` on the rows and columns of the sites in ``X`` (sites in increasing
    order) equals the probability that every site of ``X`` is occupied. The
    Pfaffian convention is ``Pf([[0, a], [-a, 0]]) = a``.

    Parameters
    ----------
    polys : np.ndarray
        Array of shape ``(L, N)``, ``N`` even and positive.
    weights : np.ndarray
        Positive finite site weights, shape ``(L,)``.

    Returns
    -------
    np.ndarray
        The kernel ``K`` of shape ``(2L, 2L)``.

    Raises
    ------
    ValueError
        If ``polys`` is not a finite two-dimensional array with an even,
        positive number of columns, or ``weights`` is not a finite positive
        array of shape ``(L,)`` matching the rows of ``polys``.
    """
    return kernel
```

### Step 5

05_condition_kernel

Goal
----
Produce the Pfaffian kernel of a discrete Pfaffian point process restricted to the remaining sites, conditioned on a set of sites being all occupied or all empty.

```python
import numpy as np
def condition_kernel(kernel: np.ndarray, sites: list[int], occupied: bool) -> np.ndarray:
    """Return the kernel of a discrete Pfaffian point process after conditioning on sites.

    ``kernel`` is a real skew-symmetric ``(2L, 2L)`` array whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x = 0, ..., L - 1``; the
    Pfaffian of its principal submatrix on any set of distinct sites is the
    probability that all of them are occupied, with
    ``Pf([[0, a], [-a, 0]]) = a``. Condition on the event that every site in
    ``sites`` is occupied (``occupied=True``) or that every site in ``sites``
    is empty (``occupied=False``). Return a skew-symmetric kernel for the
    remaining sites, kept in increasing order and relabelled
    ``0, 1, ..., L - len(sites) - 1``, with the same layout and Pfaffian
    convention, whose principal Pfaffians are the conditional probabilities
    that the corresponding remaining sites are all occupied. The input is not
    modified.

    Parameters
    ----------
    kernel : np.ndarray
        Skew-symmetric array of shape ``(2L, 2L)``.
    sites : list[int]
        Distinct site labels in ``0, ..., L - 1``; at least one and fewer than
        ``L``.
    occupied : bool
        ``True`` to condition on all listed sites occupied, ``False`` to
        condition on all of them empty.

    Returns
    -------
    np.ndarray
        Array of shape ``(2(L - len(sites)), 2(L - len(sites)))``.

    Raises
    ------
    ValueError
        If ``kernel`` is not a finite square array of even order, ``sites`` is
        empty, contains repeated or out-of-range labels, non-integers or
        booleans, or lists every site, ``occupied`` is not a bool, or the
        conditioning event has probability zero (the 2 x 2 block system that
        defines the conditioning is singular).
    """
    return conditioned
```

### Step 6

06_compute_gap_probability

Goal
----
Compute the probability that a discrete Pfaffian point process leaves every site of a given set empty.

```python
import numpy as np
def compute_gap_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Return the probability that no site in ``sites`` is occupied.

    ``kernel`` is a real skew-symmetric ``(2L, 2L)`` array whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x = 0, ..., L - 1``; the
    Pfaffian of its principal submatrix on any set of distinct sites is the
    probability that all of them are occupied, with
    ``Pf([[0, a], [-a, 0]]) = a``. An empty ``sites`` gives 1.0. The input is
    not modified.

    Parameters
    ----------
    kernel : np.ndarray
        Skew-symmetric array of shape ``(2L, 2L)``.
    sites : list[int]
        Distinct site labels in ``0, ..., L - 1``.

    Returns
    -------
    float
        The probability that every listed site is empty.

    Raises
    ------
    ValueError
        If ``kernel`` is not a finite square array of even order, or
        ``sites`` contains repeated or out-of-range labels, non-integers or
        booleans.
    """
    return 0.0
```

### Step 7

07_compute_single_occupancy_p

Goal
----
Compute the probability that exactly one site of a given set is occupied in a discrete Pfaffian point process.

```python
import numpy as np
def compute_single_occupancy_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Return the probability that exactly one site in ``sites`` is occupied.

    ``kernel`` is a real skew-symmetric ``(2L, 2L)`` array whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x = 0, ..., L - 1``; the
    Pfaffian of its principal submatrix on any set of distinct sites is the
    probability that all of them are occupied, with
    ``Pf([[0, a], [-a, 0]]) = a``. The probability that all listed sites are
    empty is assumed positive. An empty ``sites`` gives 0.0. The input is not
    modified.

    Parameters
    ----------
    kernel : np.ndarray
        Skew-symmetric array of shape ``(2L, 2L)``.
    sites : list[int]
        Distinct site labels in ``0, ..., L - 1``.

    Returns
    -------
    float
        The probability that exactly one listed site is occupied.

    Raises
    ------
    ValueError
        If ``kernel`` is not a finite square array of even order, or
        ``sites`` contains repeated or out-of-range labels, non-integers or
        booleans.
    """
    return 0.0
```

### Step 8

08_compute_second_row_pmf

Goal
----
Compose every earlier step to obtain the exact probability that the second row of the RSK shape of a symmetric geometric waiting-time matrix has a given length.

```python
import numpy as np
def compute_second_row_pmf(
    q: float = 0.2,
    n_rows: int = 20,
    row_length: int = 21,
    n_sites: int = 120,
) -> float:
    """Return P(lambda_2 = row_length) for a symmetric geometric waiting-time matrix.

    The ``n_rows x n_rows`` waiting-time matrix is symmetric with independent
    entries on and above the diagonal: an off-diagonal entry takes the value
    ``k = 0, 1, 2, ...`` with probability ``(1 - q) * q**k`` and a diagonal
    entry with probability ``(1 - sqrt(q)) * q**(k / 2)``. ``lambda`` is the
    shape that the Robinson-Schensted-Knuth correspondence assigns to the
    matrix, so ``lambda_1`` is the last-passage time from ``(1, 1)`` to
    ``(n_rows, n_rows)``. The shifted rows ``h_i = lambda_i + n_rows - i`` are
    treated exactly on the sites ``0, 1, ..., n_sites - 1``, which truncate
    the non-negative integers. Return the probability that the second row has
    length ``row_length``. The defaults reproduce the problem statement; at
    these defaults the 120-site truncation changes the result by less than
    1e-15.

    Parameters
    ----------
    q : float
        Geometric ratio of the off-diagonal entries, ``0 < q < 1``.
    n_rows : int
        Matrix size, an even integer of at least 2.
    row_length : int
        Positive integer length of the second row.
    n_sites : int
        Number of sites, with ``row_length + n_rows - 1 < n_sites``.

    Returns
    -------
    float
        The probability ``P(lambda_2 = row_length)``.

    Raises
    ------
    ValueError
        If ``q`` is not a finite real number in (0, 1), ``n_rows`` is not an
        even integer of at least 2, ``row_length`` is not a positive
        integer, ``n_sites`` is not an integer with
        ``row_length + n_rows - 1 < n_sites`` (booleans are rejected for all
        integers), or any stage rejects its input.
    """
    return 0.0
```
