# Mathematics-Numerical_Linear_Algebra-32

## Background

A rank-r truncated SVD is optimal among rank-r matrices, but it needs the full matrix and does not fit a streaming or one-pass budget. The usual one-pass substitute draws a short right sketch for a rangefinder and a left sketch for the corange, orthonormalizes the rangefinder, and recovers coefficients from the sketched corange. That reconstruction degrades when the spectrum is flat, because the short rangefinder mixes dominant and tail directions. Classical power iteration would correct the rangefinder, but it revisits A. A wider right sketch formed in the same pass can act as a cheap stand-in for A on the rangefinder, after which the original sketched-corange solve and a rank truncation still produce the approximation. The prompt instance asks for one entry of that reconstruction, not a residual norm or the optimal truncated SVD.

## Problem

One-pass low-rank approximation reconstructs A from sketches collected in a single visit: a short right sketch for a rangefinder, a left sketch for the corange, and a wider right sketch that can stand in for A on that rangefinder when the singular spectrum decays slowly. The computational method takes a rectangular matrix, those three sketches, and a target rank, and returns one entry of the rank-truncated reconstruction obtained after a single amplification of the rangefinder.

Use numpy Generator default_rng(7) to form thin QR factors of standard_normal draws of shapes (12, 10) and (10, 10), and set A = U diag(s) V^T with s = (4, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7). Using default_rng(11), draw the three sketches as standard_normal matrices of shapes (10, 4), (7, 12), and (10, 8) in that order. Your final answer must be a single number: the entry in row 2, column 0 of the rank-3 reconstruction, indexing from 0.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_construct_flat_spectrum_matrix

Goal
----
Build A = U diag(s) V^T from thin QR factors of Gaussians drawn by default_rng(seed). The singular values s are prescribed and positive. Require m >= n >= 1.

```python
import numpy as np

def construct_flat_spectrum_matrix(m: int, n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Build A of shape (m, n) with singular values s.

    Parameters
    ----------
    m : int
        Row dimension, m >= n >= 1.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape (n,).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    A : np.ndarray
        Array of shape (m, n).

    Raises
    ------
    ValueError
        If `m` or `n` is not an integer, if `m >= n >= 1` does not hold,
        if `s` does not have shape `(n,)`, or if `s` is not positive and
        finite.
    """
    return np.zeros((m, n))
```

### Step 2

02_draw_gaussian_test_matrices

Goal
----
Draw three independent standard_normal test matrices from default_rng(seed), in order Omega of shape (n, s_width), Psi of shape (d, m), and Phi of shape (n, l). Require min(l, d) > s_width >= 1 and s_width <= min(m, n). Pack them as one vector: six shape integers followed by the three matrices in row-major order.

```python
import numpy as np

def draw_gaussian_test_matrices(
    m: int, n: int, s_width: int, d: int, l: int, seed: int
) -> np.ndarray:
    """Draw Omega (n, s_width), Psi (d, m), Phi (n, l) and pack them.

    Parameters
    ----------
    m, n : int
        Dimensions of A.
    s_width : int
        Rangefinder sketch width.
    d : int
        Corange sketch height, must exceed s_width.
    l : int
        Amplifier sketch width, must exceed s_width.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        One-dimensional packing of (Omega, Psi, Phi).

    Raises
    ------
    ValueError
        If `m`, `n`, `s_width`, `d`, or `l` is not an integer; if
        `m >= 1` and `n >= 1` do not both hold; if `s_width >= 1` does
        not hold; if `min(l, d) > s_width` does not hold; or if
        `s_width <= min(m, n)` does not hold.
    """
    return np.zeros(6)
```

### Step 3

03_form_one_pass_sketches

Goal
----
Unpack the test-matrix pack into Omega, Psi, and Phi. Form the rangefinder Y = A Omega, the corange sketch W = Psi A, and the wider amplifier Z = A Phi. Return those three sketches packed with the same 6-entry shape header.

```python
import numpy as np

def form_one_pass_sketches(A: np.ndarray, test_pack: np.ndarray) -> np.ndarray:
    """Return the packed sketches (Y, W, Z) = (A Omega, Psi A, A Phi).

    Raises
    ------
    ValueError
        If `test_pack` is too short or its packed shapes are not
        positive or do not match its own length header; if `A` is not
        2D or is empty; or if `Omega`, `Psi`, or `Phi` (unpacked from
        `test_pack`) have shapes incompatible with `A` (`Omega` must
        have `A`'s number of rows... i.e. `n` rows, `Psi` must have
        `A`'s number of columns i.e. `m` columns, and `Phi` must have
        `n` rows).
    """
    return np.zeros(6)
```

### Step 4

04_amplify_rangefinder

Goal
----
Apply the wider amplifier Z to the rangefinder Y for exactly q >= 1 steps. Each step takes a thin QR of Z^T Yhat and replaces Yhat by Z X. Reject q = 0: an unamplified rangefinder is a different method.

```python
import numpy as np

def amplify_rangefinder(Y: np.ndarray, Z: np.ndarray, q: int) -> np.ndarray:
    """Apply q re-orthonormalized amplifier steps to Y using Z.

    Raises
    ------
    ValueError
        If `Y` or `Z` is not 2D, if `Y` and `Z` do not have the same
        number of rows, if `require l >= s_width >= 1` does not hold,
        if `q` is not an integer, if `q < 1`, or if the amplifier core
        is rank deficient.
    """
    return np.zeros_like(Y)
```

### Step 5

05_rangefinder_thin_factors

Goal
----
Compute thin QR factors of the amplified rangefinder Yhat and return them stacked as [Q; R], with Q of shape (m, s_width) on top of the s_width-by-s_width triangular factor. Require m >= s_width >= 1 and a full-column-rank rangefinder.

```python
import numpy as np

def rangefinder_thin_factors(Yhat: np.ndarray) -> np.ndarray:
    """Return vstack(Q, R) for the reduced factorization of Yhat.

    Raises
    ------
    ValueError
        If `Yhat` is not 2D, if `m >= s_width >= 1` does not hold, or
        if the rangefinder is rank deficient.
    """
    return np.zeros((1, 1))
```

### Step 6

06_sketched_coefficient_matrix

Goal
----
Recover the coefficient matrix B by solving (Psi Q) B = W in the least-squares sense. Require d >= s_width and matching inner dimensions. This is a sketched corange fit, not Q^T A.

```python
import numpy as np

def sketched_coefficient_matrix(Q: np.ndarray, Psi: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Solve (Psi Q) B = W in the least-squares sense.

    Raises
    ------
    ValueError
        If `Q`, `Psi`, or `W` is not 2D; if `Q` is empty; if `Psi` does
        not have as many columns as `Q` has rows; if `W` does not have
        as many rows as `Psi`; if `d >= s_width` does not hold; or if
        `W` has no columns.
    """
    return np.zeros((1, 1))
```

### Step 7

07_reconstruction_entry

Goal
----
Form Q times the best rank-r approximation of B and return the entry in the given 0-based row and column. Require 1 <= rank <= min(B.shape) and indices inside the reconstruction.

```python
import numpy as np

def reconstruction_entry(
    Q: np.ndarray, B: np.ndarray, rank: int, row: int, col: int
) -> float:
    """Return (Q [[B]]_rank)[row, col] with 0-based indices.

    Raises
    ------
    ValueError
        If `Q` or `B` is not 2D, if `Q` and `B` have incompatible inner
        dimensions, if `rank` is not an integer, if `rank` does not lie
        between 1 and `min(B.shape)`, if `row` or `col` is not an
        integer, or if `row` or `col` is out of bounds.
    """
    return 0.0
```

### Step 8

08_run_sketch_power_entry

Goal
----
Orchestrator. Call steps 1-7 in order: construct A, draw the three test matrices, form (Y, W, Z), amplify the rangefinder for q steps, take thin factors, recover B from the sketched corange, and return the requested entry of the rank-r reconstruction. This is the scalar required by the prompt.

```python
import numpy as np

def run_sketch_power_entry(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    s_width: int,
    d: int,
    l: int,
    q: int,
    rank: int,
    row: int,
    col: int,
) -> float:
    """End-to-end one-pass amplified reconstruction; return one entry."""
    return 0.0
```
