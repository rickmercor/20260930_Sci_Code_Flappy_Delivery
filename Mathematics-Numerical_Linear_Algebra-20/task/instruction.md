# Mathematics-Numerical_Linear_Algebra-20

## Background

Banded-plus-semiseparable (BPS) matrices are structured matrices that combine a banded component with low-rank structure in the strictly lower and upper triangular regions. This representation appears in applications including spectral methods for differential equations, signal processing, and eigenvalue problems. It allows storage to grow linearly with matrix dimensions when the ranks and bandwidths remain fixed. The challenge is to preserve this structure during matrix factorization. The paper developed a structure-preserving Householder QR framework by showing that intermediate matrices remain within a structured perturbation space even though individual Householder transformations do not strictly preserve the original BPS form, which ensures that the complete QR factorization can still be represented compactly.

The factor matrix stores the upper-triangular $R$ factor together with the Householder reflector information and retains BPS structure with predictable changes in rank and bandwidth. This structural preservation allows for QR factorization to be performed in $O(n)$ operations for fixed structural parameters, giving a complete linear-complexity solver for BPS linear systems. For symmetric BPS matrices, the same structural ideas extend to the $RQ$ product used in QR-based eigenvalue computations. Since the $RQ$ product also preserves symmetric BPS structure, each iteration of the associated QR eigenvalue algorithm can likewise be performed in $O(n)$, while avoiding the formation of dense orthogonal matrices.

## Problem

Banded-plus-semiseparable (BPS) matrices combine a banded component with lower- and upper-semiseparable components. Their structure can be preserved during Householder QR factorization, allowing the factorization to be represented without forming a dense matrix.

Consider the nonsymmetric matrix

$$
A=B+\operatorname{tril}(UV^\top,-1)+\operatorname{triu}(WS^\top,1)
$$

with $n=100000$, lower bandwidth $\ell=2$, upper bandwidth $m=3$, lower semiseparable rank $r=2$, and upper semiseparable rank $p=2$.

Using one-based indices $i=1,\ldots,n$, define the nonzero entries of the banded component by

$$
B_{i,i}=5+0.2\sin(0.01i),\qquad
B_{i,i-1}=0.6,\qquad
B_{i,i-2}=-0.1,
$$

$$
B_{i,i+1}=-0.8,\qquad
B_{i,i+2}=0.15,\qquad
B_{i,i+3}=-0.05,
$$

whenever the indicated indices lie in $1,\ldots,n$, with every other entry of $B$ equal to zero.

Define the generator rows as

$$
U_{i,:}=\frac{1}{\sqrt n}\left[1+0.2\sin(0.017i),\; 0.7+0.15\cos(0.011i) \right],
$$

$$
V_{i,:}=\frac{1}{\sqrt n}\left[0.9+0.1\cos(0.013i),\; -0.6+0.12\sin(0.019i)\right],
$$

$$
W_{i,:}=\frac{1}{\sqrt n}\left[0.8+0.18\sin(0.023i),\;0.5+0.10\cos(0.029i)\right],
$$

$$
S_{i,:}=\frac{1}{\sqrt n}\left[-0.7+0.11\cos(0.031i),\; 0.6+0.14\sin(0.037i)\right].
$$

Use the structured BPS Householder QR factorization in which the factor matrix $F$ stores $R$ in its upper triangle and the Householder vectors below the diagonal. Each Householder vector $y_k$ has zeros before index $k$ and is normalized so that $y_k[k]=1$. The Householder coefficients are collected in

$$
\tau=[\tau_1,\ldots,\tau_{n-1},0]^\top,
$$

with each reflector represented as

$$
I-\tau_k y_k y_k^\top.
$$

Use IEEE-754 binary64 arithmetic and the standard sign-stable Householder convention, which maps the active column to a multiple whose sign is opposite that of its leading entry, and compute $\tau_{73129}$ to 8 digits past the decimal point. Your final answer is the single scalar, $\tau_{73129}$. Do not report full matrices, full coefficient vectors, or per-iteration sequences. Briefly describe the structured QR method used and include only the few scalar intermediates needed to justify the final value.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Briefly state the structured method used and the few scalar intermediates that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_construct_banded_component

Goal
----
Implements construct_banded_components, which builds the compact banded part of the deterministic banded-plus-semiseparable (BPS) matrix used by the QR factorization task.

```python
import numpy as np

def construct_banded_component(n: int) -> np.ndarray:
    """Constructs the compact six-diagonal representation of B.
    Parameters
    ----------
    n: int
      The postiive matrix dimension

    Returns
    -------
    bands: np.ndarray
      A float64 array of shape (6,n), where the rows correspond to the diagonal
      offsets (-2, -1, 0, +1, +2, +3). All entries outside of the matrix boundaries
      are zero
      
    Raises
    ------
    ValueError
        If the requested matrix dimension is not a valid positive integer.

    """
    return bands
```

### Step 2

02_construct_semiseparable_generators

Goal
----
Implements construct_semiseparable_generators, which builds the four deterministic low-rank generator matrices used in the BPS representation

```python
import numpy as np

def construct_semiseparable_generators(n: int):
    """Constructs the deterministic semiseparable generators, U, V, W, and S.
    Parameters
    ----------
    n: int
      The postiive matrix dimension

    Returns
    -------
    U: np.ndarray
      float64 array of shape (n,2)
    V: np.ndarray
      float64 array of shape (n,2)
    W: np.ndarray
      float64 array of shape (n,2)
    S: np.ndarray
      float64 array of shape (n,2)

    Raises
    ------
    ValueError
        If the requested matrix dimension is not a valid positive integer.
    """
    return U, V, W, S
```

### Step 3

03_precompute_bps_quantities

Goal
----
This step performs the long-range precomputations needed to make the structured QR recurrence efficient. It computes A-transpose times U directly from the banded and semiseparable representation, forms an augmented upper generator by combining S with A-transpose times U, and builds suffix Gram matrices for the active portions of U. It also builds prefix products that accumulate the interaction between rows of U and W. These lookup quantities replace repeated long inner products during the QR iterations with fixed-size matrix operations, and the step initializes the structured perturbation state used by the first Householder update.

```python
import numpy as np

def precompute_bps_quantities(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
) -> tuple[dict, dict]:
    """
    Precompute structured contractions and initialize the perturbation state.

    Parameters
    ----------
    bands : np.ndarray
        Compact banded component with shape (6, n), ordered by offsets
        (-2, -1, 0, +1, +2, +3).
    U, V : np.ndarray
        Lower-semiseparable generators, each with shape (n, 2).
    W, S : np.ndarray
        Upper-semiseparable generators, each with shape (n, 2).

    Returns
    -------
    precomp : dict
        Contains A_T_U with shape (n, 2), S_tilde with shape (n, 4),
        UU_lookup with shape (n, 2, 2), and UW_lookup with shape (n, 2, 2).
    state : dict
        Contains zero initial matrices J, K, E, X, Y, Z and the zero-based
        recurrence index k=0.

    Raises
    ------
    ValueError
        If the band representation or generator arrays have incompatible dimensions, 
        contain invalid numerical data, or do not define a valid BPS input.
    """
    return precomp, state
```

### Step 4

04_form_structured_householder

Goal
----
Form one normalized Householder vector from the current structured BPS trailing state.

```python
import numpy as np

def form_structured_householder(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    precomp: dict,
    state: dict,
) -> dict:
    """
    Form the next normalized Householder reflector from the compact BPS state.

    Parameters
    ----------
    bands : np.ndarray
        Compact banded component with shape (6, n).
    U, V, W, S : np.ndarray
        Semiseparable generators, each with shape (n, 2).
    precomp : dict
        Precomputed A_T_U, S_tilde, UU_lookup, and UW_lookup.
    state : dict
        Current matrices J, K, E, X, Y, Z and zero-based index k.

    Returns
    -------
    reflector : dict
        Contains kbar, bhat, tau, rkk, qvec, c, and the active-column norm.

    Raises
    ------
    ValueError
        If the BPS inputs, precomputed quantities, or structured state are incomplete, 
        dimensionally inconsistent, numerically invalid, or do not define a valid 
        active Householder step.
    """
    return reflector
```

### Step 5

05_compute_structured_row_update

Goal
----
Compute the structured upper-row update produced by one Householder transformation.

```python
import numpy as np

def compute_structured_row_update(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    precomp: dict,
    state: dict,
    reflector: dict,
) -> dict:
    """
    Compute the upper semiseparable and banded row produced at one QR step.

    Parameters
    ----------
    bands : np.ndarray
        Compact banded component with shape (6, n).
    U, V, W, S : np.ndarray
        Semiseparable generators, each with shape (n, 2).
    precomp : dict
        Contains A_T_U, S_tilde, UU_lookup, and UW_lookup.
    state : dict
        Current structured perturbation state.
    reflector : dict
        Contains kbar, bhat, tau, and rkk from Step 4.

    Returns
    -------
    row_update : dict
        Contains w_tilde with shape (4,), d_tilde with length at most 6,
        and compact y.T @ A coefficients used by Step 6.

    Raises
    ------
    ValueError
        If the BPS inputs, precomputed quantities, structured state, or 
        reflector data are incomplete, dimensionally inconsistent, or 
        incompatible with the active QR step.
    """
    return row_update
```

### Step 6

06_update_structured_perturbation

Goal
----
Advance the six compact perturbation matrices after one structured Householder step.

```python
import numpy as np

def update_structured_perturbation(
    U: np.ndarray,
    W: np.ndarray,
    state: dict,
    reflector: dict,
    row_update: dict,
) -> dict:
    """
    Update the compact perturbation matrices for the next QR iteration.

    Parameters
    ----------
    U, W : np.ndarray
        Generator matrices with shape (n, 2).
    state : dict
        Current matrices J, K, E, X, Y, Z and zero-based index k.
    reflector : dict
        Householder data containing kbar, bhat, and tau.
    row_update : dict
        Contains wS_yA, wU_yA, sparse_yA, and band_row_tail.

    Returns
    -------
    next_state : dict
        Updated matrices J, K, E, X, Y, Z and incremented index k.

    Raises
    ------
    ValueError
        If the generator arrays, structured state, reflector data, or 
        row-update data are incomplete, dimensionally inconsistent, or 
        incompatible with the active QR step.
    """
    return next_state
```

### Step 7

07_compute_target_tau

Goal
----
Run the complete structured BPS Householder QR pipeline through a requested iteration.

```python
import numpy as np

def compute_target_tau(
    n: int = 100000,
    target_index: int = 73129,
) -> float:
    """
    Compute a target Householder coefficient with the complete structured pipeline.

    Parameters
    ----------
    n : int
        Positive BPS matrix dimension.
    target_index : int
        One-based Householder coefficient index in the range 1 through n.

    Returns
    -------
    tau_target : float
        The deterministic Householder scaling coefficient at target_index.

    Raises
    ------
    ValueError
        If the requested matrix dimension or target coefficient index is 
        invalid or lies outside the admissible range.
    """
    return tau_target
```
