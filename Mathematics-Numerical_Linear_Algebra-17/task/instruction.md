# Mathematics-Numerical_Linear_Algebra-17

## Background

Krylov subspace methods build an expanding basis for span{b, Ab, ..., A^{m-1}b} and extract spectral information from a small projected matrix. The classical Arnoldi process produces an orthonormal basis and an upper Hessenberg projection whose eigenvalues are Ritz values of the original operator. When the subspace must be kept small, Krylov-Schur restarting compresses the decomposition onto the Schur vectors of the wanted Ritz values and then expands it again with further Arnoldi steps.

When Euclidean inner products are replaced by a short random embedding, an inexpensive basis can still be generated, but the cheap projected matrix need not stay similar to the classical one. Its eigenvalues then need not be Ritz values, and a further algebraic step is required before they can be read as such; how that step interacts with compression, with the vector carried across a restart, and with the orthogonality of the re-expanded basis determines the numbers produced within a restart cycle.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Problem

Large-scale eigenvalue estimates from Krylov subspaces are usually extracted from a Hessenberg projection built by the Arnoldi process, and Krylov-Schur restarting keeps the basis small when many iterations are needed. Randomized Gram-Schmidt replaces Euclidean inner products by sketched ones so that a short embedding Omega produces an Omega-orthonormal Krylov basis cheaply; the resulting projected matrix is generally not similar to the classical Arnoldi projection, so its eigenvalues are not true Ritz values, and a randomized Krylov-Schur method inherits this defect at every restart. A recent similarity-restoring variant repairs the decomposition at the end of each cycle with a least-squares correction so that the compressed matrix again carries exact Ritz values.

Use numpy Generator default_rng(7) to form a thin QR factor of a standard_normal draw of shape (12, 12) and set A = Q diag(s) Q^T with s = (8, 6, 5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15); draw b as standard_normal(12) from the same generator. Using default_rng(11), draw a Gaussian sketch Omega of shape (8, 12) and use it for every sketched inner product. Run the similarity-restoring randomized Krylov-Schur method of the cited source in its default configuration with expansion order m = 4 and restart order l = 2, retaining the two largest Ritz values, for exactly one restart cycle: the initial four sketched Arnoldi steps with their correction, the compression to order 2, the re-expansion to order 4, and the correction that closes the cycle. Your final answer must be a single number: the Euclidean norm of the least-squares correction vector computed at the end of that cycle.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_assemble_sketched_instance

Goal
----
Build a packed SPD eigenproblem together with one Gaussian sketch. Form A from a thin QR of Gaussians drawn by default_rng(data_seed), draw b from that generator, then draw Omega from default_rng(sketch_seed). Pack (n, d, A.ravel(), b, Omega.ravel()). Require n >= 2, d >= 1, and positive s.

```python
import numpy as np

def assemble_sketched_instance(
    n: int, d: int, s: np.ndarray, data_seed: int, sketch_seed: int
) -> np.ndarray:
    """Pack A, b, and Omega for a clustered-spectrum eigenproblem.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    d : int
        Sketch dimension, d >= 1.
    s : np.ndarray
        Positive eigenvalues, shape (n,).
    data_seed : int
        RNG seed for the SPD factor draw and start vector.
    sketch_seed : int
        Independent RNG seed for the sketch.

    Returns
    -------
    pack : np.ndarray
        Concatenation of (n, d, A.ravel(), b, Omega.ravel()).

    Raises
    ------
    ValueError
        If n or d is not an integer, if n < 2 or d < 1, if s does not
        have shape (n,), or if any entry of s is nonpositive or nonfinite.
    """
    return np.zeros(1)
```

### Step 2

02_start_krylov_pack

Goal
----
From a packed sketched instance, Omega-normalize the start vector and pack an m-column Krylov state with that first column accepted and no Hessenberg columns written yet. Pack (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next) with n_hess = 0. Require 1 <= m < n and d >= m.

```python
import numpy as np

def start_krylov_pack(inst_pack: np.ndarray, m: int) -> np.ndarray:
    """Pack an m-column Krylov state with only the start vector accepted.

    Parameters
    ----------
    inst_pack : np.ndarray
        Packed (n, d, A.ravel(), b, Omega.ravel()).
    m : int
        Subspace length, 1 <= m < n, with d >= m.

    Returns
    -------
    state : np.ndarray
        Packed (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next)
        with n_hess = 0 and U[:, 0] the Omega-normalized start vector.

    Raises
    ------
    ValueError
        If the instance pack is short or the wrong length, if packed
        shapes are not positive, if m is not an integer, if m is not in
        1, ..., n-1, if d < m, or if Omega @ b is zero.
    """
    return np.zeros(1)
```

### Step 3

03_sketched_action_pack

Goal
----
Return the packed sketched action of w against the columns of U: the coefficient vector, the leftover, and the leftover continuation scale measured by Omega. Pack (k, h, leftover, scale). Require d >= k >= 1.

```python
import numpy as np

def sketched_action_pack(
    Omega: np.ndarray, U: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Pack sketched coefficients, leftover, and continuation scale.

    Parameters
    ----------
    Omega : np.ndarray
        Sketch of shape (d, n) with d >= k.
    U : np.ndarray
        Accepted basis, shape (n, k), k >= 1.
    w : np.ndarray
        New action, length n.

    Returns
    -------
    pack : np.ndarray
        Concatenation of (k, h, leftover, scale).

    Raises
    ------
    ValueError
        If Omega or U is not 2-dimensional, if U has no columns, if
        Omega does not have shape (d, n) with d >= k, if w does not
        have length n, if an input is nonfinite, or if the sketched
        leftover has zero scale (Arnoldi breakdown).
    """
    return np.zeros(1)
```

### Step 4

04_append_krylov_column

Goal
----
Incorporate one packed sketched action into the Krylov state. A non-final column is accepted into the basis; the final action is stored as leftover and continuation scale rather than as an extra basis column. The action width must equal n_hess + 1.

```python
import numpy as np

def append_krylov_column(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Accept a mid-subspace column, or store the final leftover.

    Parameters
    ----------
    state : np.ndarray
        Packed (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next).
    action : np.ndarray
        Packed (k, h, leftover, scale) with k = n_hess + 1.

    Returns
    -------
    state : np.ndarray
        Updated Krylov pack with n_hess increased by one.

    Raises
    ------
    ValueError
        If either pack is short, the wrong length, or has an invalid
        header, if the Krylov pack is already complete, if the action
        width is not n_hess + 1, or if the continuation scale is not
        positive.
    """
    return np.zeros(1)
```

### Step 5

05_leftover_euclidean_coeffs

Goal
----
From a completed Krylov pack (all Hessenberg columns written), return the coefficients that make the leftover Euclidean-orthogonal to the accepted basis.

```python
import numpy as np

def leftover_euclidean_coeffs(state: np.ndarray) -> np.ndarray:
    """Return coefficients making the leftover Euclidean-orthogonal to range(U).

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n_hess = m).

    Returns
    -------
    hhat : np.ndarray
        Vector of length m.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, or if U.T @ U is not SPD.
    """
    return np.zeros(1)
```

### Step 6

06_restore_last_column

Goal
----
Using those coefficients, restore last-vector similarity on the uncorrected Hessenberg and return the corrected m-by-m matrix. Only the last column changes.

```python
import numpy as np

def restore_last_column(state: np.ndarray, hhat: np.ndarray) -> np.ndarray:
    """Return the corrected Hessenberg after a last-column restoration.

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n_hess = m).
    hhat : np.ndarray
        Euclidean leftover coefficients, length m.

    Returns
    -------
    Hbar : np.ndarray
        Corrected m-by-m Hessenberg. Only the last column changes.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, or if hhat does not have length m
        or is nonfinite.
    """
    return np.zeros(1)
```

### Step 7

07_dominant_real_eigenvalue

Goal
----
Return the largest real eigenvalue of a square matrix as a native Python float. Reject a matrix whose eigenvalues are not all real.

```python
import numpy as np

def dominant_real_eigenvalue(Hbar: np.ndarray) -> float:
    """Return the largest real eigenvalue as a Python float.

    Parameters
    ----------
    Hbar : np.ndarray
        Square matrix whose spectrum must be real.

    Returns
    -------
    value : float
        Largest real eigenvalue.

    Raises
    ------
    ValueError
        If Hbar is not a nonempty square matrix, or if any eigenvalue
        is non-real.
    """
    return 0.0
```

### Step 8

08_compress_ritz_pack

Goal
----
Compress a completed, corrected order-m sketched Krylov decomposition to order l for a restart. Take the completed Krylov pack, the Euclidean leftover coefficients hhat, and the corrected matrix Hbar; keep the orthonormal Schur vectors V_l of Hbar for its l largest eigenvalues (descending, each column signed so its largest-magnitude entry is positive); return the pack (n, l, (U V_l).ravel(), (V_l^T Hbar V_l).ravel(), leftover - U hhat, V_l^T (h_next e_m)). The carried vector is not rescaled. Require 1 <= l < m, real eigenvalues, and a gap between the l-th and (l+1)-th eigenvalues.

```python
import numpy as np

def compress_ritz_pack(
    state: np.ndarray, hhat: np.ndarray, Hbar: np.ndarray, l: int
) -> np.ndarray:
    """Pack the order-l Krylov decomposition that starts the next restart cycle.

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n, m, n_hess, U.ravel(), H.ravel(),
        leftover, h_next) with n_hess = m.
    hhat : np.ndarray
        Euclidean leftover coefficients, length m.
    Hbar : np.ndarray
        Corrected m-by-m matrix, equal to H with h_next * hhat added to
        its last column.
    l : int
        Restart order, 1 <= l < m.

    Returns
    -------
    decomp : np.ndarray
        Packed (n, l, U_l.ravel(), S_l.ravel(), uhat, c_l) where
        U_l = U @ V_l, S_l = V_l.T @ Hbar @ V_l, uhat = leftover - U @ hhat
        (carried unnormalized), and c_l = V_l.T @ (h_next * e_m). V_l holds
        the orthonormal Schur vectors of Hbar for its l largest eigenvalues,
        ordered so the diagonal of S_l decreases; each column of V_l is
        signed so that its entry of largest magnitude (lowest index on ties)
        is positive.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, if hhat or Hbar has the wrong shape or
        is nonfinite, if Hbar is not the last-column restoration of H, if l
        is not an integer in 1, ..., m-1, if Hbar has a non-real eigenvalue,
        or if the l-th and (l+1)-th largest eigenvalues coincide.
    """
    return np.zeros(1)
```

### Step 9

09_expand_krylov_cycle

Goal
----
Expand an order-l Krylov decomposition A U_l = U_l H_l + uhat c_l^T back to order m by m - l sketched Gram-Schmidt steps. Place U_l, then uhat unchanged, as the first l + 1 columns; put H_l in the leading block and c_l in row l; for each new action w = A U[:, k] solve the sketched least-squares problem min ||Omega U[:, :k+1] h - Omega w||_2, subtract, and Omega-normalize. Return the standard completed Krylov pack (n, m, m, U.ravel(), H.ravel(), leftover, h_next). Require l + 1 <= m < n and d >= m.

```python
import numpy as np

def expand_krylov_cycle(
    inst_pack: np.ndarray, decomp: np.ndarray, m: int
) -> np.ndarray:
    """Return the completed order-m Krylov pack of one restart cycle.

    Parameters
    ----------
    inst_pack : np.ndarray
        Packed (n, d, A.ravel(), b, Omega.ravel()).
    decomp : np.ndarray
        Packed (n, l, U_l.ravel(), H_l.ravel(), uhat, c_l) satisfying
        A U_l = U_l H_l + uhat c_l^T.
    m : int
        Expansion order, l + 1 <= m < n, with d >= m.

    Returns
    -------
    state : np.ndarray
        Completed Krylov pack (n, m, m, U.ravel(), H.ravel(), leftover,
        h_next) with U[:, :l] = U_l, U[:, l] = uhat kept as given (not
        rescaled), H[:l, :l] = H_l, H[l, :l] = c_l, and columns l, ..., m-1
        of H filled by m - l sketched Gram-Schmidt steps: for each new
        action w = A U[:, k], the coefficients solve
        min || Omega U[:, :k+1] h - Omega w ||_2, the leftover is
        w - U[:, :k+1] h, and the next column (or the final leftover) is
        that vector divided by || Omega leftover ||_2, which is stored as
        the subdiagonal entry (or h_next).

    Raises
    ------
    ValueError
        If either pack is short, the wrong length, has an invalid header,
        or is nonfinite, if the two packs disagree on n, if m is not an
        integer with l + 1 <= m < n, if d < m, or if a sketched leftover
        has zero scale (breakdown).
    """
    return np.zeros(1)
```

### Step 10

10_run_restarted_correction_norm

Goal
----
Run one full restart cycle of similarity-restoring randomized Krylov-Schur: assemble the sketched instance, start the Krylov pack, expand it for m sketched actions, solve the Euclidean leftover coefficients and restore the last column, check the corrected Ritz value lies in the spectrum, compress to order l, re-expand to order m, solve the leftover coefficients again, restore and check again, and return the Euclidean norm of the second coefficient vector. Require 2 <= m < n, d >= m, 1 <= l < m. Call every earlier public step and use every output.

```python
import numpy as np

def run_restarted_correction_norm(
    n: int,
    m: int,
    d: int,
    l: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
) -> float:
    """Norm of the least-squares correction closing one restart cycle.

    Parameters
    ----------
    n : int
        Dimension, n >= 3.
    m : int
        Expansion order, 2 <= m < n.
    d : int
        Sketch dimension, d >= m.
    l : int
        Restart order, 1 <= l < m.
    s : np.ndarray
        Positive eigenvalues, shape (n,).
    data_seed : int
        RNG seed for the SPD factor draw and start vector.
    sketch_seed : int
        Independent RNG seed for the sketch.

    Returns
    -------
    value : float
        Euclidean norm of the leftover coefficient vector computed on the
        re-expanded order-m basis at the end of the first restart cycle.

    Raises
    ------
    ValueError
        If n, m, d, or l is not an integer, if n < 3, if m is not in
        2, ..., n-1, if d < m, if l is not in 1, ..., m-1, or if a
        corrected Ritz value exceeds max(s) beyond roundoff.
    """
    return 0.0
```
