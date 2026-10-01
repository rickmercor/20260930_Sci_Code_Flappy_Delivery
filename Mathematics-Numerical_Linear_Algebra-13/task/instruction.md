# Error reduction from a bounded shared-index representation

## Background

# Scientific background

Quantizing both factors of a matrix product creates an error that cannot be assessed from either factor in isolation. An entrywise perturbation in one factor propagates through the energy of the corresponding contracted slice of the other, while simultaneous perturbations in both factors create an additional coupled contribution. Under independent zero-mean error fields, these effects admit an exact finite-dimensional expected-error identity rather than requiring a Monte Carlo estimate.

An invertible transform on the contracted dimension can be applied to one factor and undone on the other without changing the full-precision product. Positive diagonal transforms redistribute channel ranges before quantization and can therefore reduce product-level error, but independent per-output transforms would require additional transformed copies of the opposite factor. A single bounded transform preserves reuse, and the resulting range-law optimization has a convex log-domain formulation with a globally certifiable solution. The scientific question is whether the reduction in propagated quantization error justifies this shared representation change for the fixed product.

## Problem

For a low-precision matrix product, determine the percentage by which a bounded, product-preserving change of representation reduces the exact expected squared Frobenius error when both factors are quantized with independent, non-overloading subtractive dither. Use the dimensionless matrices

$$
A=\begin{bmatrix}
16.31231506&-7.54254968\\
14.37223598&-2.75656887\\
-18.20787217&-3.14210042\\
2.81871387&-0.48440943\\
-5.16046682&-4.66856835\\
-15.94578266&-6.77234276\\
-12.58770833&-2.09423959\\
-14.18828266&-5.59286129\\
-8.93508206&-1.13268159\\
11.14212066&1.96062568\\
7.46273074&-3.55797030\\
7.97673266&-4.06988699\\
15.01512042&6.70256452\\
-10.29476925&1.10146802
\end{bmatrix},
$$

$$
B=\begin{bmatrix}
-8.30783179&-1.56753650&-5.29208479&-6.96446941&-3.55882920&-11.26925957&-2.79502846&1.85744954&-3.57038034&13.66170249&-1.84632400&4.23266944&-15.87690823&-9.69696113\\
2.75894695&-3.74235899&-5.40883066&9.97208014&9.84474059&-6.74859084&-3.07721095&-0.94076857&-6.66295278&-6.63739874&7.47277449&4.78762101&-4.06900093&-9.10419672
\end{bmatrix}.
$$

Use signed bit widths $b_A=2$ and $b_B=3$, one quantization range per transformed row of $A$, and one range per transformed column of $B$. Restrict the shared-index transform to the source's normalized two-channel positive diagonal family, which carries a single free parameter $x$, the natural logarithm of the scale applied to the first shared channel, constrained to $-0.95\le x\le0.95$. Use the source's signed-bit subtractive-dither variance law and its exact expected product-error identity. Determine the bounded global minimum of that full, generally nonsmooth objective. If $E_0$ is the full expected error at the identity transform $x=0$ and $E_\star$ is the bounded minimum, report the single finite percentage $100(1-E_\star/E_0)$. Use natural logarithms and IEEE-754 double precision, without random sampling or Monte Carlo approximation. State the intermediate scalars your own derivation actually turns on; the restriction below on pasted vectors and tables is aimed at bulk listings, not at those scalars.

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

01_compute_dither_variance_coefficient

Goal
----
Compute the variance coefficient for a signed uniform quantizer.



For signed bit width `$b >= 2$`, the largest positive index is

`$q_b = 2**(b - 1) - 1$`. Non-overloading subtractive dither gives a uniform

error over one step, so a group with range `$R$` has variance

`$c_b * R**2$`, where `$c_b = 1 / (12 * q_b**2)$`.

```python
def compute_dither_variance_coefficient(bits: int) -> float:
    """Return the signed-quantizer dither variance coefficient.

    Raises ``ValueError`` unless ``bits`` is an integer of at least 2 and the
    resulting coefficient is finite and strictly positive. NumPy integer
    scalars such as ``np.int64(11)`` must be accepted as integers. Boolean
    values, including ``np.bool_``, are not accepted as integers.

    Parameters
    ----------
    bits : int
        Signed quantizer bit width.

    Returns
    -------
    float
        The finite coefficient ``1 / (12 * (2**(bits - 1) - 1)**2)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 2

02_construct_two_channel_switch_partition

Goal
----
Construct the exact active-range partition for a bounded two-channel gauge.



For `$H(x) = diag(exp(x), exp(-x))$`, a row range of ``A H`` or a

column range of `$H^{-1} B$` is a maximum of two exponentials.  The full

error is smooth only between their tie points.  This step returns every

distinct in-bound tie, the two endpoints, and one probe in each open cell.

```python
import numpy as np


def construct_two_channel_switch_partition(
    A: np.ndarray,
    B: np.ndarray,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Build the bounded partition induced by all row and column range ties.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    ``bound`` is a non-Boolean scalar that is finite and strictly positive;
    and ``tolerance`` is a non-Boolean scalar that is finite, strictly
    positive, and smaller than ``bound``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``, with ``m >= 1``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``, with ``n >= 1``.
    bound : float
        Finite positive log-scale bound; the domain is ``[-bound, bound]``.
    tolerance : float, optional
        Finite positive distance used to cluster coincident tie points.  It
        must be smaller than ``bound``.

    Returns
    -------
    np.ndarray
        A finite vector ``[s, z_0, ..., z_{s-1}, p_0, ..., p_{s-2}]``.
        The ``z`` values are strictly increasing partition points and each
        ``p_j`` is the midpoint of ``(z_j, z_{j+1})``.
    """
    return result
```

### Step 3

03_encode_partition_active_branches

Goal
----
Encode the active max-range channel throughout every partition cell.



The partition from the previous step removes all nonzero tie points.  At each

cell midpoint, the maximizing channel therefore identifies the active row and

column range law throughout that open cell.  The packed numerical table keeps

the variable-size certificate compatible with strict array comparison.

```python
import numpy as np


def encode_partition_active_branches(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Encode active channels for every open cell of a switch partition.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    ``tolerance`` is a non-Boolean scalar that is finite and strictly
    positive; and ``partition`` is a finite one-dimensional packed vector that
    is internally consistent, meaning its header ``s`` is an integer of at
    least 2 matching a total length of ``2 * s``, its ``s`` switches increase
    with consecutive gaps larger than ``tolerance``, and its ``s - 1`` probes
    each lie strictly inside their own cell and equal that cell's midpoint to
    within ``tolerance``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    partition : np.ndarray
        Packed vector ``[s, switches, probes]`` from the partition step.
    tolerance : float, optional
        Finite positive structural tolerance.

    Returns
    -------
    np.ndarray
        Packed vector beginning with the number of cells.  Each following
        record is ``[lower, upper, probe, a_0, ..., a_{m-1}, b_0, ...,
        b_{n-1}]``, where every active index is exactly 0 or 1.
    """
    return result
```

### Step 4

04_reconstruct_active_error_branch

Goal
----
Reconstruct one smooth branch of the exact product-error objective.



Away from a range tie, every max-defined variance selects one channel.  The

three-term product error on that cell is exactly

`$P*exp(4*x) + Q*exp(-4*x) + C$`.  This step recovers the coefficients and a

numerical certificate at a supplied interior probe without sampling or curve

fitting.

```python
import numpy as np


def reconstruct_active_error_branch(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Recover the exact exponential branch active at ``probe``.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2; ``tolerance`` is a
    non-Boolean scalar that is finite and strictly positive; and ``probe`` is
    a finite non-Boolean scalar whose distance to every nonzero row and column
    range tie is strictly greater than ``tolerance``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    bits_a : int
        Non-Boolean signed bit width for the left factor, at least 2.
    bits_b : int
        Non-Boolean signed bit width for the right factor, at least 2.
    probe : float
        Finite log scale that is farther than ``tolerance`` from every
        nonzero range tie.
    tolerance : float, optional
        Finite positive tie-distance tolerance.

    Returns
    -------
    np.ndarray
        ``[P, Q, C, E, E_prime, E_second, m, n, active_A, active_B]``.
        The active arrays contain channel indices 0 or 1.
    """
    return result
```

### Step 5

05_enumerate_bounded_error_candidates

Goal
----
Enumerate a finite global-minimization certificate on the bounded gauge.



All nonsmooth points come from active-range switches.  Within an open cell,

the exact objective is a convex three-term exponential branch, so that cell

contributes at most one interior stationary candidate.  The returned table

contains every endpoint, every distinct switch, and every admissible

stationary point together with its branch data and derivative.

```python
import numpy as np


def enumerate_bounded_error_candidates(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Construct the complete finite candidate table for the bounded problem.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2; ``bound`` is a
    non-Boolean scalar that is finite and strictly positive; and ``tolerance``
    is a non-Boolean scalar that is finite, strictly positive, and smaller
    than ``bound``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    bits_a : int
        Non-Boolean signed bit width for the left factor, at least 2.
    bits_b : int
        Non-Boolean signed bit width for the right factor, at least 2.
    bound : float
        Finite positive log-scale bound.
    tolerance : float, optional
        Finite positive switch-clustering and candidate-deduplication distance,
        strictly smaller than ``bound``.

    Returns
    -------
    np.ndarray
        Packed vector ``[N, records]`` with eight values per sorted record:
        ``[x, kind, cell, P, Q, C, E, E_prime]``.  Kind is -1 for the lower
        endpoint, 0 for an interior switch, 1 for the upper endpoint, and 2
        for an interior stationary point.
    """
    return result
```

### Step 6

06_compute_expected_product_error

Goal
----
Evaluate the exact expected squared error of a product with both factors noisy.



For mutually independent zero-mean entry errors with variance fields `$v_A$`

and `$v_B$`, the result is the sum of the left propagated contribution, the

right propagated contribution, and the simultaneous-error contribution:



``sum_ik v_A[ik] ||B[k,:]||_2^2``;

``sum_kj v_B[kj] ||A[:,k]||_2^2``; and

`$sum_k (sum_i v_A[ik]) (sum_j v_B[kj])$`.

```python
import numpy as np


def compute_expected_product_error(
    A_tilde: np.ndarray,
    B_tilde: np.ndarray,
    variance_a: np.ndarray,
    variance_b: np.ndarray,
) -> np.ndarray:
    """Return the three exact error contributions and their total.

    Raises ``ValueError`` unless the factor arrays are finite, nonempty, and
    two-dimensional with a common contracted dimension; the two variance
    arrays have exactly the corresponding factor shapes; and all variances are
    finite and nonnegative.

    Parameters
    ----------
    A_tilde : np.ndarray
        Transformed left factor of shape ``(m, K)``.
    B_tilde : np.ndarray
        Transformed right factor of shape ``(K, n)``.
    variance_a : np.ndarray
        Left entrywise variance field of shape ``(m, K)``.
    variance_b : np.ndarray
        Right entrywise variance field of shape ``(K, n)``.

    Returns
    -------
    np.ndarray
        Finite float array ``[left, right, simultaneous, total]`` of shape
        ``(4,)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 7

07_certify_bounded_shared_scaling

Goal
----
Certify the global bounded optimum of the nonsmooth two-channel objective.



The certificate joins the finite candidate construction to the original

three-term error identity.  Besides the minimizer, it records one-sided

derivatives, the separation from the next distinct candidate, reconstruction

residuals on every smooth cell, and preservation of the unquantized product.

```python
import numpy as np


def certify_bounded_shared_scaling(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return a numerical global-optimality certificate on the closed bound.

    The reported switch count is the size of the bounded partition, which
    always includes both closed-bound endpoints ``-bound`` and ``+bound`` in
    addition to every distinct in-bound range tie.

    Raises ``ValueError`` unless the identity-transform expected error is
    finite and strictly positive, as well as for invalid arguments.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    bits_a : int
        Non-Boolean signed bit width for the left factor, at least 2.
    bits_b : int
        Non-Boolean signed bit width for the right factor, at least 2.
    bound : float
        Finite positive log-scale bound.
    tolerance : float, optional
        Finite positive switch and tie tolerance, below ``bound``.

    Returns
    -------
    np.ndarray
        A finite shape-``(14,)`` array containing ``x_star``, reciprocal
        scales, optimized and identity errors, switch and candidate counts,
        candidate kind and index, left and right derivatives, second-candidate
        gap, maximum branch residual, and product-preservation residual.
    """
    return result  # noqa: F821 - required model stub
```

### Step 8

08_compute_quantized_product_error_reduction

Goal
----
Run the complete certified bounded error-reduction calculation.



The pipeline constructs the nonsmooth range partition, encodes every active

cell, reconstructs exact exponential branches, enumerates the finite global

candidate set, checks the original three-term identity, and returns the

certified percentage reduction from the identity transform.

```python
import numpy as np


def compute_quantized_product_error_reduction(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> float:
    """Return the full expected-error reduction from bounded shared scaling.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2 producing finite
    positive coefficients; ``bound`` is finite and strictly positive;
    ``tolerance`` is finite, strictly positive, and smaller than ``bound``;
    and the identity-transform expected error is finite and strictly positive.
    This step is the orchestrator: it calls every earlier step in order, so
    all of them must be defined.

    Parameters
    ----------
    A : np.ndarray
        Left factor of shape ``(m, 2)``.
    B : np.ndarray
        Right factor of shape ``(2, n)``.
    bits_a : int
        Signed bit width for the left factor.
    bits_b : int
        Signed bit width for the right factor.
    bound : float
        Positive closed log-scale bound for ``x``.
    tolerance : float, optional
        Positive switch, tie, and certificate tolerance.

    Returns
    -------
    float
        Finite percentage ``100 * (1 - E_star / E_0)``.
    """
    return result  # noqa: F821 - required model stub
```
