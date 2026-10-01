# Physics-Quantum_Information_Computing-23

## Background

Quantum process characterization asks how faithfully an implemented channel realizes a known target, typically through a fidelity between the two maps. Full process tomography reconstructs every matrix element and is expensive. Randomized benchmarking reports an average decay after twirling, which is scalable but erases circuit-specific coherent errors. Direct fidelity estimation sits between those extremes: it importance-samples Pauli matrix elements of the target and estimates the overlap from local preparations and measurements that leave the circuit under study intact. How those samples are chosen, how many distinct preparation–measurement settings are required, and how the shot budget scales with the target’s Pauli-transfer structure are the design questions that control whether such a protocol can sit inside a repeated calibration loop.

## Problem

Direct fidelity estimation of a quantum channel importance-samples the target’s Pauli-transfer support and estimates the entanglement fidelity from local Pauli preparations and measurements. The wanted quantity is the target-dependent finite-shot overhead of that protocol after compatible Pauli pairs have been collected, evaluated on a locked two-qubit fractional gate, to absolute accuracy 10^{−6}.

Take the family fSim(θ, φ) = CP(−φ) R_XX(θ) R_YY(θ), where R_XX(θ) = exp(−i θ XX/2) and R_YY(θ) = exp(−i θ YY/2), at θ = φ = π/8. Drop the identity Pauli pair. Exclude every zero coefficient. Retain every remaining coefficient whose magnitude is at least 10^{−4}. When squared coefficients are compared, use values rounded to twelve decimal places, and break remaining ties by output index, then input index, in the lexicographic {I,X,Y,Z}⊗{I,X,Y,Z} order. Form any group cost from the unrounded coefficients. Your final answer must be a single number: that target-dependent finite-shot overhead.

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

two_qubit_pauli

Goal
----
Returns the 4×4 matrix of a two-qubit Pauli given its lexicographic code in {I, X, Y, Z}⊗{I, X, Y, Z}.

```python
import numpy as np


def two_qubit_pauli(code: int) -> np.ndarray:
    """Return the 4×4 matrix of the two-qubit Pauli with the given code.

    Codes run from 0 through 15 in lexicographic tensor order
    {I, X, Y, Z}⊗{I, X, Y, Z}, so code 0 is I⊗I and code 5 is X⊗X.

    Parameters
    ----------
    code : int
        Integer in {0, …, 15}.

    Returns
    -------
    matrix : np.ndarray
        Complex array of shape (4, 4).

    Raises
    ------
    ValueError
        If code is not an integer in {0, …, 15}.
    """
    return np.zeros((4, 4), dtype=complex)
```

### Step 2

fsim_unitary

Goal
----
Builds the two-qubit fractional gate fSim(θ, φ) = CP(−φ) R_XX(θ) R_YY(θ) with the standard half-angle generators.

```python
import numpy as np


def fsim_unitary(theta: float, phi: float) -> np.ndarray:
    """Return the 4×4 unitary fSim(theta, phi).

    Use R_XX(θ) = exp(−i θ XX / 2), R_YY(θ) = exp(−i θ YY / 2), and
    CP(φ) = diag(1, 1, 1, exp(i φ)). Then fSim(0, π) is exactly CZ.

    Parameters
    ----------
    theta : float
        Exchange angle, finite.
    phi : float
        Conditional phase, finite.

    Returns
    -------
    unitary : np.ndarray
        Complex array of shape (4, 4).

    Raises
    ------
    ValueError
        If theta or phi is not a finite number.
    """
    return np.eye(4, dtype=complex)
```

### Step 3

pauli_transfer_matrix

Goal
----
Forms the 16×16 real Pauli-transfer matrix of a two-qubit unitary.

```python
import numpy as np


def pauli_transfer_matrix(unitary: np.ndarray) -> np.ndarray:
    """Return the 16×16 real Pauli-transfer matrix of a two-qubit unitary.

    Index rows and columns in the same lexicographic Pauli order as
    two_qubit_pauli. The (α, β) entry is (1/4) Tr[P_α U P_β U†].

    Parameters
    ----------
    unitary : np.ndarray
        Complex array of shape (4, 4), unitary to numerical tolerance.

    Returns
    -------
    chi : np.ndarray
        Real array of shape (16, 16).

    Raises
    ------
    ValueError
        If unitary is not a finite 4×4 array, or is not unitary within
        absolute tolerance 1e-8.
    """
    return np.zeros((16, 16))
```

### Step 4

retained_pair_table

Goal
----
Extracts the truncated non-identity Pauli-transfer support as a numeric table of (α, β, χ) rows.

```python
import numpy as np
def retained_pair_table(chi: np.ndarray, tau: float) -> np.ndarray:
    """Return retained (α, β, χ_U(α, β)) rows after dropping the identity pair.
    Keep every nonzero entry with |χ_U(α, β)| ≥ tau except the (I, I) slot
    (α = β = 0). Zero coefficients are excluded even when tau = 0. Rows are
    ordered by increasing α, then increasing β.
    Parameters
    ----------
    chi : np.ndarray
        Real array of shape (16, 16).
    tau : float
        Nonnegative finite magnitude cutoff. Zero is allowed.
    Returns
    -------
    table : np.ndarray
        Array of shape (K, 3). Each row is (α, β, χ_U(α, β)). K = 0 is allowed
        and returns shape (0, 3).
    Raises
    ------
    ValueError
        If chi is not a finite real 16×16 array, if any entry has a nonzero
        imaginary part, or tau is not a nonnegative finite number.
    """
    return np.zeros((0, 3))
```

### Step 5

pair_jointly_accessible

Goal
----
Returns 1 if two retained Pauli pairs can be estimated in one preparation–measurement setting.

```python
import numpy as np
def pair_jointly_accessible(pair_a: np.ndarray, pair_b: np.ndarray) -> float:
    """Return 1 if two retained pairs can be estimated together, else 0.
    Each pair is a length-3 row (alpha, beta, coefficient) from the
    retained-support table. The test is the hardware compatibility of the
    two output labels and of the two input labels.
    Parameters
    ----------
    pair_a : np.ndarray
        Length-3 row (alpha, beta, coefficient).
    pair_b : np.ndarray
        Length-3 row (alpha, beta, coefficient).
    Returns
    -------
    flag : float
        1.0 if the two pairs share a setting, otherwise 0.0.
    Raises
    ------
    ValueError
        If either argument is not a finite length-3 array, or if any Pauli
        label is not an integer in {0, ..., 15}.
    """
    return 0.0
```

### Step 6

partition_shot_overhead

Goal
----
Returns the target-dependent finite-shot overhead of the compatible partition of a retained-support table.

```python
import numpy as np
def partition_shot_overhead(pair_table: np.ndarray) -> float:
    """Return the target-dependent finite-shot overhead of pair_table.
    The input is the retained-support table. Two rows may share a group
    only when they are jointly accessible and they agree on whether each
    register's Pauli is the identity. An empty table returns 0.
    Order rows by decreasing squared coefficient rounded to twelve decimal
    places, breaking ties by increasing output label, then input label.
    Repeatedly start a group with the first remaining row and scan the
    other remaining rows once in that order. Add a row when it is jointly
    accessible with every current member and agrees with their identity
    status on each register. Repeat on the unassigned rows.

    For each group with unrounded coefficients u, its contribution is
    (sum(abs(u)))**2 / sum(u**2). Return the sum of these contributions
    over groups. Rounding is used only for ordering, never for group costs.

    Parameters
    ----------
    pair_table : np.ndarray
        Array of shape (K, 3) with rows (alpha, beta, coefficient).
    Returns
    -------
    overhead : float
        Nonnegative scalar. Empty input yields 0.
    Raises
    ------
    ValueError
        If pair_table is not a finite array of shape (K, 3), if any Pauli
        label is not an integer in {0, ..., 15}, if any coefficient is
        zero, or if a formed block has vanishing coefficient mass.
    """
    return 0.0
```

### Step 7

total_effective_support

Goal
----
Evaluates the grouped-protocol overhead on fSim(θ, φ) after truncation.

```python
import numpy as np
def total_effective_support(theta: float, phi: float, tau: float) -> float:
    """Return the grouped-protocol overhead of fSim(theta, phi).
    Form the gate, take its Pauli-transfer matrix, keep the truncated
    non-identity nonzero support at cutoff tau, and return the finite-shot
    overhead of that table. Zero coefficients are excluded. An empty
    retained support returns 0. The identity gate at tau = 0 is valid and
    must not fail.
    Parameters
    ----------
    theta : float
        Exchange angle, finite.
    phi : float
        Conditional phase, finite.
    tau : float
        Nonnegative finite magnitude cutoff.
    Returns
    -------
    total : float
        The scalar overhead.
    Raises
    ------
    ValueError
        If theta or phi is not finite, if tau is not a nonnegative finite
        number, and any ValueError raised by the earlier steps for their
        invalid inputs.
    """
    return 0.0
```
