# Physics-Quantum_Information_Computing-25

## Background

Quantum low-density parity-check codes can reduce the physical-qubit overhead of quantum error correction. Many families have a hypergraph structure that prevents direct use of standard graph-matching decoders. The decoder must also keep pace with syndrome extraction, roughly one microsecond per round on superconducting hardware. Degeneracy, meaning that many different errors share one syndrome, makes fast iterative decoders unreliable, while accurate post-processing is too slow.

## Problem

I am evaluating a recent real-time decoder for quantum LDPC codes that is designed to be both accurate and fast enough for dedicated hardware. Offline, it applies a full-rank binary row transformation and a column permutation to the check matrix, splitting it into a block-diagonal part and a sparse off-diagonal remainder; online, it decodes each syndrome with a hierarchical greedy search over that split. I want to quantify its sensitivity to uncertainty in the relative calibration of data-fault and outcome-flip weights on a small hypergraph-product code under single-round noise.

The code is the hypergraph product of two ring codes, the ring code of length $L$ having the $L\times L$ check matrix with ones at $(i,i)$ and $(i,\,i+1 \bmod L)$. Take $H_1$ of length 3 and $H_2$ of length 4, with $H_X=(H_1\otimes I_4\mid I_3\otimes H_2^{T})$ and $H_Z=(I_3\otimes H_2\mid H_1^{T}\otimes I_4)$. Data qubits are numbered 0 to 23 by the columns of $H_X$, and $X$-type checks 0 to 11 by its rows. A $Z$ fault hits data qubit $j$ with probability $p_j=0.010+0.001\,j$, and in the single syndrome round the outcome of check $i$ flips with probability $q_i=0.006+0.0005\,i$. All 36 faults are independent, so the decoder works with the $12\times36$ matrix $(H_X\mid I_{12})$, whose last 12 columns are the outcome flips.

Use the following explicit extension of the decoder’s noise-weight model. For each of the 36 columns, set $W=\mathrm{round}_{\mathrm{even}}\!\left(10^{6}\ln\frac{1-p}{p}\right)$, where $p$ is that column’s actual fault probability and $\mathrm{round}_{\mathrm{even}}$ rounds to the nearest integer, resolving exact half-integer ties to the even integer. A calibration gain $\alpha$ is uniformly distributed on $[3/4,\,5/4]$, independently of the fault configuration. The decoder assigns weight $\alpha W$ to each data column and weight $W$ to each outcome-flip column. The actual fault probabilities $p_j$ and $q_i$ do not change with $\alpha$. This finite-precision gain model is part of this task. The decoder uses the decoupling that its authors prescribe for hypergraph-product codes and caps each level of its search at 3 iterations. A decoding fails when the data part of its correction differs from the actual data error by anything other than a product of $Z$-type stabilizers.

Correction to the published pseudo-code: use the all-zero off-diagonal guess, completed by its block decodings, as the initial incumbent. For zero syndrome under these weights, identify the published outer initialization and determine whether its first accepted update improves on the completed zero guess.

Compute the probability that a fault configuration with at most three faults occurs and the decoder fails on it, averaged over $\alpha$. Sum the unconditional independent-fault probabilities of all such configurations; do not renormalize by the probability of at most three faults. Integrate over the exact intervals of constant decoder correction, rather than sampling a gain grid. Values at isolated decision boundaries have zero measure. At either search level, resolve equal candidate objectives by the lowest index of the newly set guess bit, using the source matrix column order within that guess.

Also determine the number of diagonal blocks, the composition and shape of one block, the shape of the off-diagonal part and the row transformation; the search rule applied at each level; how correction intervals determine the gain average; the gain-averaged numbers of failing configurations with zero, one, two and three faults (these counts may be nonintegers); the gain-averaged two-fault and three-fault probability contributions; and the final probability.

Report nonzero gain-averaged failing counts to at least six decimal places, and probability contributions and the final probability to at least eleven decimal places.

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

01_build_phenomenological_check_matrix

Goal
----
Build the single-round decoding matrix of a hypergraph-product code, with one column per data-qubit fault and one per check-outcome flip, together with the stabilizers of the opposite type.

```python
def build_phenomenological_check_matrix(
    h1: "np.ndarray",
    h2: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    r"""Return the fault-to-syndrome matrix and the opposite-type check matrix.

    With ``h1`` $=h_1$ of shape $(m_1, n_1)$ and ``h2`` $=h_2$ of shape
    $(m_2, n_2)$, the code has $n_1 n_2 + m_1 m_2$ data qubits and $m_1 n_2$
    $X$-type checks, $H_X = (h_1 \otimes I_{n_2} \mid I_{m_1} \otimes h_2^{T})$
    and $H_Z = (I_{n_1} \otimes h_2 \mid h_1^{T} \otimes I_{m_2})$ (Kronecker
    products, entries reduced modulo 2). The decoding matrix is
    $D = (H_X \mid I_{m_1 n_2})$: its first $n_1 n_2 + m_1 m_2$ columns are the
    data-qubit $Z$ faults in $H_X$ column order and its last $m_1 n_2$ columns
    are the outcome flips of the checks in row order.

    Parameters
    ----------
    h1 : np.ndarray
        Nonempty 2-D binary (0/1) parity-check matrix $h_1$ of the first classical code.
    h2 : np.ndarray
        Nonempty 2-D binary (0/1) parity-check matrix $h_2$ of the second classical code.

    Returns
    -------
    tuple of np.ndarray
        ``(D, H_Z)`` as integer 0/1 arrays of shapes
        $(m_1 n_2,\ n_1 n_2 + m_1 m_2 + m_1 n_2)$ and $(n_1 m_2,\ n_1 n_2 + m_1 m_2)$.

    Raises
    ------
    ValueError
        If either input is not a nonempty 2-D array whose entries are all 0 or 1.
    """
    return d_matrix, hz
```

### Step 2

02_decouple_hypergraph_product

Goal
----
Find the column permutation that brings the decoding matrix of a hypergraph-product code into the decoupled form used by the hierarchical decoder, with the block count that decoder prescribes for this code family.

```python
def decouple_hypergraph_product(h1: "np.ndarray", h2: "np.ndarray") -> "np.ndarray":
    r"""Return the column permutation of the decoupled decoding matrix.

    $D$ is the matrix of ``build_phenomenological_check_matrix(h1, h2)``,
    of shape $(m, n)$ with $m = m_1 n_2$. Use the off-diagonal part
    $A = h_1 \otimes I_{n_2}$, the first $n_1 n_2$ columns of $D$, and
    $K = m_1$ diagonal blocks. For each zero-based block index $i$,
    the block occupies rows $i n_2$ through $(i+1) n_2 - 1$. Its
    identity comes from the outcome-flip columns of those rows; its
    $B_i$ consists of the $m_2$ columns of $I_{m_1} \otimes h_2^{T}$ belonging
    to block $i$, restricted to those rows. Thus each
    $D_i = (I_{n_2} \mid B_i)$ has shape $(n_2,\ n_2 + m_2)$.

    Return the integer array ``perm`` of length $n$ such that
    ``D[:, perm]`` $= (\mathrm{diag}(D_1, \dots, D_K) \mid A)$, with no row
    transformation needed. Column $j$ of ``D[:, perm]`` is column ``perm[j]``
    of $D$. Order the blocks by increasing row index. Within each block, place
    the identity columns before $B_i$; inside each identity part,
    each $B_i$ and $A$, keep increasing original column indices.

    Parameters
    ----------
    h1 : np.ndarray
        Nonempty 2-D binary matrix $h_1$ of shape $(m_1, n_1)$.
    h2 : np.ndarray
        Nonempty 2-D binary matrix $h_2$ of shape $(m_2, n_2)$.

    Returns
    -------
    np.ndarray
        Integer permutation of ``range(n)``, $n = n_1 n_2 + m_1 m_2 + m_1 n_2$.

    Raises
    ------
    ValueError
        If either input is not a nonempty 2-D array whose entries are all 0 or 1.
    """
    return perm
```

### Step 3

03_compute_decoupling_transform

Goal
----
Compute the full-rank binary row transformation that, together with a given column permutation, puts a decoding matrix into the block-decoupled form, and confirm that the form is reached.

```python
def compute_decoupling_transform(
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    n_blocks: int,
    block_cols: int,
) -> "np.ndarray":
    r"""Return the row transformation $T$ of the decoupled decoding matrix.

    $D$ has shape $(m, n)$. With $K$ = ``n_blocks``, $b$ = ``block_cols`` and
    $m_D = m / K$ rows per block, return the $(m, m)$ binary matrix $T$ for
    which $D' = T D_{\pi}$ (arithmetic modulo 2), where $D_{\pi}$ = ``D[:, perm]``,
    is decoupled: for every block $i$, the columns $i b, \dots, (i+1) b - 1$
    of $D'$ vanish outside rows $i m_D, \dots, (i+1) m_D - 1$, and inside those
    rows their first $m_D$ columns form the identity matrix. The columns
    after $K b$ are unconstrained.

    Parameters
    ----------
    d_matrix : np.ndarray
        2-D binary (0/1) matrix $D$ of shape $(m, n)$.
    perm : np.ndarray
        Integer permutation of ``range(n)``.
    n_blocks : int
        Positive number of blocks $K$ dividing $m$.
    block_cols : int
        Columns per block $b$, with $m_D \le b$ and $K b \le n$.

    Returns
    -------
    np.ndarray
        Integer 0/1 array of shape $(m, m)$.

    Raises
    ------
    ValueError
        If ``d_matrix`` is not a 2-D binary array, ``perm`` is not a
        permutation of ``range(n)``, ``n_blocks`` or ``block_cols`` violate
        the stated conditions, or no binary matrix $T$ produces the
        decoupled form.
    """
    return t_matrix
```

### Step 4

04_greedy_guess_block

Goal
----
Compute the exact correction profile of a diagonal block over an interval of affine weight calibration gains.

```python
def greedy_guess_block(
    b_matrix: "np.ndarray",
    syndrome: "np.ndarray",
    weight_f: "np.ndarray",
    weight_g: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return the gain-dependent correction profile for one block $(I \mid B)$.

    Each weight row is the integer pair $(a, b)$ representing the
    affine objective coefficient $a + b\alpha$ at calibration gain $\alpha$.
    ``gain_interval`` is ``((lo_num, lo_den), (hi_num, hi_den))``, with
    positive denominators and $0 < \mathrm{lo} < \mathrm{hi}$. At each fixed
    gain use the specified monotone greedy decoder. Return its complete
    piecewise-constant correction profile, including every change on an open
    interval. Values at isolated breakpoints and interval endpoints do not
    contribute to the requested average and are excluded from the profile
    contract.

    The result is ``(breakpoints, errors)``. ``breakpoints`` is a tuple of
    reduced integer numerator/positive-denominator pairs in strictly
    increasing order, starting at $\mathrm{lo}$ and ending at $\mathrm{hi}$.
    Row $i$ of the integer 0/1 array ``errors`` applies between breakpoint
    $i$ and $i+1$. Adjacent rows must differ; merge adjacent intervals with
    the same full correction even when internal greedy decisions changed.
    Integer affine coefficients make the breakpoints rational. A sampled gain
    grid does not define this exact profile.

    For a fixed gain start from $g = 0$ and $f = s$, where $s$ = ``syndrome``.
    Form every candidate that sets one currently-zero bit of $g$ to one; bits
    are never cleared. Recompute $f = (B g + s) \bmod 2$ and the full weighted
    cost of $(f, g)$. Select the lowest-cost candidate, breaking equal
    costs by the lowest newly set $B$-column index. Accept only a strict
    improvement; otherwise stop. Stop also when no zero bits remain or
    after ``max_iter`` rounds. This rule applies throughout the gain interval.

    Parameters
    ----------
    b_matrix : np.ndarray
        Binary array $B$ of shape $(m_D, n_B)$, both dimensions positive.
    syndrome : np.ndarray
        Binary vector $s$ of length $m_D$.
    weight_f : np.ndarray
        Integer affine coefficients of shape $(m_D, 2)$.
    weight_g : np.ndarray
        Integer affine coefficients of shape $(n_B, 2)$.
    max_iter : int
        Non-negative round cap.
    gain_interval : tuple
        Two positive rational endpoints as defined above.

    Returns
    -------
    tuple
        Rational breakpoints and errors of shape $(N_{\mathrm{int}},\ m_D + n_B)$,
        where $N_{\mathrm{int}}$ is the number of intervals.

    Raises
    ------
    ValueError
        If a binary array, coefficient array, cap or interval violates
        the stated shape or domain.

    """
    return result
```

### Step 5

05_compute_left_syndromes

Goal
----
For a guess of the error on the off-diagonal part of the decoupled matrix, compute the syndrome that each diagonal block must then explain.

```python
def compute_left_syndromes(
    syndrome: "np.ndarray",
    t_matrix: "np.ndarray",
    a_matrix: "np.ndarray",
    right_error: "np.ndarray",
    n_blocks: int,
) -> "np.ndarray":
    r"""Return the per-block syndromes left for the diagonal blocks.

    The decoupled matrix is $D' = T D P = (\mathrm{diag}(D_1, \dots, D_K) \mid A)$
    and ``syndrome`` is the measured syndrome of the original matrix $D$. For
    the off-diagonal error ``right_error`` (on the columns of $A$),
    return the $(K, m/K)$ array whose row $i$ is the syndrome that
    block $D_i$ must reproduce, over rows $i m/K, \dots, (i+1) m/K - 1$ of
    $D'$, for the complete permuted error to satisfy the decoupled
    syndrome equation modulo 2.

    Parameters
    ----------
    syndrome : np.ndarray
        Binary vector of length $m$.
    t_matrix : np.ndarray
        Binary $(m, m)$ transformation $T$.
    a_matrix : np.ndarray
        Binary off-diagonal part $A$ of shape $(m, n_A)$.
    right_error : np.ndarray
        Binary vector of length $n_A$.
    n_blocks : int
        Positive number of blocks $K$ dividing $m$.

    Returns
    -------
    np.ndarray
        Integer 0/1 array of shape $(K, m/K)$.

    Raises
    ------
    ValueError
        If any array is not binary with the stated shape, or ``n_blocks``
        is not a positive integer dividing $m$.
    """
    return left_syndromes
```

### Step 6

06_hierarchical_decode

Goal
----
Compute the full correction profile of the hierarchical decoder under affine calibration uncertainty.

```python
def hierarchical_decode(
    syndrome: "np.ndarray",
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    t_matrix: "np.ndarray",
    n_blocks: int,
    block_cols: int,
    weights: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return the exact gain-dependent hierarchical correction profile.

    Each weight row is the integer pair $(a, b)$ representing the
    affine objective coefficient $a + b\alpha$ at calibration gain $\alpha$.
    ``gain_interval`` is ``((lo_num, lo_den), (hi_num, hi_den))``, with
    positive denominators and $0 < \mathrm{lo} < \mathrm{hi}$. At each fixed
    gain use the specified monotone greedy decoder. Return its complete
    piecewise-constant correction profile, including every change on an open
    interval. Values at isolated breakpoints and interval endpoints do not
    contribute to the requested average and are excluded from the profile
    contract.

    The result is ``(breakpoints, errors)``. ``breakpoints`` is a tuple of
    reduced integer numerator/positive-denominator pairs in strictly
    increasing order, starting at $\mathrm{lo}$ and ending at $\mathrm{hi}$.
    Row $i$ of the integer 0/1 array ``errors`` applies between breakpoint
    $i$ and $i+1$. Adjacent rows must differ; merge adjacent intervals with
    the same full correction even when internal greedy decisions changed.
    Integer affine coefficients make the breakpoints rational. A sampled gain
    grid does not define this exact profile.

    $D' = T D_{\pi} \bmod 2$, where $D_{\pi}$ = ``D[:, perm]``, contains
    $K$ = ``n_blocks`` diagonal blocks $(I \mid B_i)$ followed by $A$. Each
    fixed-gain outer search starts from $r = 0$ completed by all block
    decodings. Each round tries every candidate that sets one currently-zero
    $r$ bit to one; bits are never cleared. For each candidate, compute
    $(T s + A r) \bmod 2$, where $s$ = ``syndrome``, and decode every block on
    its own residual rows using ``greedy_guess_block`` at that gain with the
    same ``max_iter``. Compare the complete objective of all block errors and
    $r$. Accept the lowest-cost candidate only on a strict improvement,
    resolving equal costs by the lowest newly set $A$-column index; otherwise
    stop. Stop after ``max_iter`` rounds or when no zero bits remain. The
    zero-completed incumbent is the explicit correction to the source's
    infinite-incumbent initialization.

    Weights refer to original $D$ columns; keep them attached through ``perm``.
    Each returned correction is in original column order. Use the block
    profiles from ``greedy_guess_block`` and the residual syndromes from
    ``compute_left_syndromes``. Both inner and outer decision changes affect
    the returned profile; discard only boundaries where the full returned
    correction stays identical on the two neighboring open intervals.

    Parameters
    ----------
    syndrome : np.ndarray
        Binary length-$m$ syndrome $s$ in the original row basis.
    d_matrix : np.ndarray
        Binary array $D$ of shape $(m, n)$.
    perm : np.ndarray
        Integer permutation of ``range(n)``.
    t_matrix : np.ndarray
        Binary invertible $(m, m)$ row transformation $T$.
    n_blocks : int
        Positive block count $K$ dividing $m$.
    block_cols : int
        Columns per block $b$, strictly larger than $m/K$; $K b \le n$.
    weights : np.ndarray
        Integer affine coefficients of shape $(n, 2)$, in original column order.
    max_iter : int
        Non-negative round cap at both levels.
    gain_interval : tuple
        Two positive rational endpoints as defined above.

    Returns
    -------
    tuple
        Rational breakpoints and binary errors of shape $(N_{\mathrm{int}},\ n)$,
        where $N_{\mathrm{int}}$ is the number of intervals. Every correction
        $e$ satisfies $D e = s \pmod 2$.

    Raises
    ------
    ValueError
        If shapes, binary entries, integer coefficients, cap, interval or
        decoupled layout violate the stated requirements.

    """
    return result
```

### Step 7

07_truncated_failure_probability

Goal
----
Integrate stabilizer-equivalent decoding success over calibration gain and sum the unconditional fault probabilities.

```python
def truncated_failure_probability(
    d_matrix: "np.ndarray",
    hz: "np.ndarray",
    probabilities: "np.ndarray",
    decode_fn: "Callable[[np.ndarray], tuple]",
    max_faults: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return gain-averaged failing counts and probability contributions by fault order.

    Faults independently occupy the columns of $D$ with the supplied
    probabilities. The first ``hz.shape[1]`` columns are data faults. For each
    support of size at most ``max_faults``, including the empty support, use
    its binary syndrome to obtain ``(breakpoints, corrections)`` from
    ``decode_fn``. The profile covers ``gain_interval`` in the rational format
    of ``hierarchical_decode``. The gain is uniform on this interval and
    independent of the faults.

    On a profile interval, failure means that the actual data error $\oplus$
    the correction's data part is outside the $\mathrm{GF}(2)$ row space of
    ``hz``. Let $\lambda(S)$ be the total length of failing gain intervals
    divided by $\mathrm{hi} - \mathrm{lo}$ for support $S$. Return arrays
    ``counts`` and ``contributions``, each of length ``max_faults`` $+\,1$,
    where ``counts[k]`` sums $\lambda(S)$ over $k$-fault supports and
    ``contributions[k]`` sums $\lambda(S)$ times the unconditional
    probability of each support, including absence of all other faults.
    Counts can be nonintegral. Do not condition on the retained fault order.

    Parameters
    ----------
    d_matrix : np.ndarray
        Binary array $D$ of shape $(m, n)$.
    hz : np.ndarray
        Binary stabilizer generators with $1 \le n_{\mathrm{data}} \le n$ columns.
    probabilities : np.ndarray
        Length-$n$ probabilities in $[0, 1]$, including deterministic faults.
    decode_fn : callable
        Pure deterministic syndrome-to-profile mapping; corrections have
        $n$ entries and satisfy $D c = s \pmod 2$. Profiles may be cached.
    max_faults : int
        Integer in $[0, n]$.
    gain_interval : tuple
        Two rational endpoints, ``((lo_num, lo_den), (hi_num, hi_den))``,
        $0 < \mathrm{lo} < \mathrm{hi}$.

    Returns
    -------
    tuple
        Two floating arrays ``(counts, contributions)``, ordered by fault count.

    Raises
    ------
    ValueError
        If inputs violate the stated domains or a returned profile has
        invalid rational endpoints, does not cover the requested interval,
        has unordered boundaries or invalid corrections, or has a wrong syndrome.

    """
    return result
```

### Step 8

08_estimate_decoder_failure_probability

Goal
----
Compose all earlier steps to evaluate calibration-averaged failure for the ring-code hypergraph product. Orchestrator: yes.

```python
def estimate_decoder_failure_probability(
    ring_lengths: tuple = (3, 4),
    data_prob: tuple = (0.010, 0.001),
    meas_prob: tuple = (0.006, 0.0005),
    max_iter: int = 3,
    max_faults: int = 3,
    gain_interval: tuple = ((3, 4), (5, 4)),
    weight_scale: int = 1000000,
) -> float:
    r"""Return the truncated failure probability averaged over calibration gain.

    Build the length-$L$ ring checks, with ones at $(i, i)$ and
    $(i, (i+1) \bmod L)$, and their phenomenological hypergraph product using
    steps 01–03. With $(d_0, d_1)$ = ``data_prob`` and $(e_0, e_1)$ =
    ``meas_prob``, the actual fault probabilities are $p_j = d_0 + j d_1$ on
    data columns and $q_i = e_0 + i e_1$ on outcome-flip columns. For each
    column let $W$ be nearest-integer rounding, ties to even, of
    $\sigma \ln\frac{1-p}{p}$ with $\sigma$ = ``weight_scale`` and $p$ that
    column's probability. Data-column decoder coefficients are $(0, W)$ and
    outcome-column coefficients are $(W, 0)$: gain multiplies only the
    data-column weights. This fixed-point calibration model is an explicit
    task extension; the underlying monotone hierarchical decoder is unchanged.

    Compose the exact hierarchical correction profiles and the gain-averaged
    fault enumeration, and sum its contributions for orders
    $0, \dots,$ ``max_faults``. The gain is uniform on ``gain_interval`` and
    independent of the actual faults. Every default is specified in the
    problem statement.

    Parameters
    ----------
    ring_lengths : tuple
        Two integers at least 2.
    data_prob : tuple
        Two finite numbers $(d_0, d_1)$ defining data probabilities strictly between 0 and 1.
    meas_prob : tuple
        Two finite numbers $(e_0, e_1)$ defining outcome probabilities strictly between 0 and 1.
    max_iter : int
        Non-negative cap on rounds at both levels.
    max_faults : int
        Integer from 0 to the total number of fault columns.
    gain_interval : tuple
        ``((lo_num, lo_den), (hi_num, hi_den))``, with $0 < \mathrm{lo} < \mathrm{hi}$
        and positive denominators.
    weight_scale : int
        Positive integer $\sigma$ multiplying the log-odds before rounding.

    Returns
    -------
    float
        Unconditional gain-averaged truncated failure probability.

    Raises
    ------
    ValueError
        If an argument violates the stated domain or a composed step rejects its input.

    """
    return result
```
