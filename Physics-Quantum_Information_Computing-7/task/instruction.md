# Physics-Quantum_Information_Computing-7

## Background

Imaginary-time evolution suppresses high-energy components and connects a maximally mixed initial operator to finite-temperature Gibbs physics. Direct matrix propagation scales exponentially with qubit count, while other many-body representations can be limited by entanglement growth or sign structure.

A sparse Pauli expansion provides an operator-space alternative in which local Hamiltonian generators transform only the stored Pauli strings. The relevant nonunitary map has a method-specific commuting/anticommuting structure, and repeated application produces support growth that must be controlled without erasing the sequential dependence of the evolution.

Because nonunitary scale factors may overflow even when only coefficient ratios matter, the propagation is evaluated in a common-factor-free form and trace-scaled after each elementary generator. Truncation can also make the approximate Hermitian operator unsuitable as a density operator, so the final observable is extracted through a squared-state overlap construction that preserves a physical readout without building dense matrices.

## Problem

Sparse imaginary-time simulation in the Pauli basis can approximate finite-temperature and low-energy quantum states without dense matrices, but nonunitary propagation expands the stored operator support and truncation can compromise the physical readout. For this deterministic instance, compute a positive-semidefinite squared-state total-energy estimate for a four-qubit open transverse-field Ising chain.

Use $H=-J\sum_{i=0}^{2}Z_iZ_{i+1}-h\sum_{i=0}^{3}X_i$ with $J=1.0$, $h=0.73$, 0-based qubit indices, open boundaries, and Pauli codes $0=I$, $1=X$, $2=Y$, $3=Z$; order the Hamiltonian strings as $ZZII, IZZI, IIZZ, XIII, IXII, IIXI, IIIX$. Start from the identity expansion $R_0=IIII$ with coefficient $1$, set $\Delta\tau=0.16$, $\tau_f=0.64$, $N=\lceil\tau_f/\Delta\tau\rceil=4$, $K=12$, and $\epsilon_0=10^{-14}$, and process every complete first-order Trotter layer in the stated Hamiltonian order with generator parameter $\theta=a_j\Delta\tau$.

For each elementary generator, apply the method-specific commuting/anticommuting imaginary-time Pauli rule in a globally scale-free representation obtained by removing the common positive $\cosh(\theta)$ factor; evaluate the remaining hyperbolic ratios without directly forming large $\cosh$ or $\sinh$ values, form products in generator-left order, and preserve parent-then-branch row order. Immediately merge duplicate Pauli strings in first-occurrence order using cancellation-stable real summation, discard a merged coefficient only when $|c|\le\epsilon_0$, retain the $K$ largest magnitudes with ties resolved by the earlier row index while returning survivors in original order, and divide all coefficients by the unique identity coefficient before the next generator; updates are strictly sequential and no intermediate value is rounded.

After the fourth layer, evaluate $E=\operatorname{Tr}(HR^2)/\operatorname{Tr}(R^2)$ in units of $J$ from sparse Pauli-coefficient overlaps rather than by constructing dense matrices. In `<reasoning>`, state the unrescaled propagation identity, explain the numerical role of per-generator trace normalization, characterize the source method's fixed-$K$ choice relative to coefficient-threshold truncation, state the overlap reduction that avoids explicitly forming $R^2$, and report the scale-free factors, final support count, numerator, denominator, and ordinary unsquared estimate; place only $E$ rounded to 10 digits after the decimal point in `<final_answer>`.

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

build_open_tfim_pauli_terms

Goal
----
Construct the ordered Pauli decomposition of an open transverse-field Ising chain.

```python
from numbers import Integral, Real

import numpy as np


def build_open_tfim_pauli_terms(
    n_qubits: int,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the ordered open-chain transverse-field Ising Hamiltonian.

    The Pauli encoding is 0=I, 1=X, 2=Y, and 3=Z.  Bond
    strings are returned first in increasing left-site order, followed by
    single-site field strings in increasing site order.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Must be a positive integer and cannot be bool.
    coupling : float
        Finite real nearest-neighbour coupling J in
        -J * sum_i Z_i Z_{i+1}.
    field : float
        Finite real transverse-field strength h in
        -h * sum_i X_i.

    Returns
    -------
    pauli_codes : np.ndarray
        Integer array of shape (2*n_qubits - 1, n_qubits) containing the
        ordered Pauli strings.
    coefficients : np.ndarray
        Float array of shape (2*n_qubits - 1,) containing the Hamiltonian
        coefficients in the same order.

    Raises
    ------
    ValueError
        If n_qubits is not a positive integer, or if either scalar is not
        a finite real number.
    """
    return (
        np.empty((2 * int(n_qubits) - 1, int(n_qubits)), dtype=np.int8),
        np.empty(2 * int(n_qubits) - 1, dtype=float),
    )
```

### Step 2

propagate_itpp_scale_free

Goal
----
Apply one globally scale-free imaginary-time Pauli-propagation update.

```python
from numbers import Real

import math
import numpy as np


def propagate_itpp_scale_free(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    generator: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one globally rescaled two-sided imaginary-time Pauli update.

    For each stored Pauli row P, determine the branch from its commutation
    relation with the involutory generator Q. Derive the scale-free branch
    weights from exp(-theta * Q / 2) P exp(-theta * Q / 2) after removing
    the common positive factor cosh(theta) from the entire output. The
    coefficient of each commuting parent row is therefore unchanged; do
    not normalize rows or branches independently. Form generated rows as
    QP, preserve parent order,
    and keep every representable scale-free coefficient finite for finite
    inputs, including large absolute theta.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Integer array of shape (m, n) with entries in {0, 1, 2, 3}.
    coefficients : np.ndarray
        Finite real array of shape (m,).
    generator : np.ndarray
        Non-identity integer Pauli code of shape (n,).
    theta : float
        Finite real parameter in exp(-theta * Q / 2).

    Returns
    -------
    propagated_codes : np.ndarray
        Integer array with m to 2*m rows and n columns.
    propagated_coefficients : np.ndarray
        Finite float array aligned with propagated_codes.

    Raises
    ------
    ValueError
        If shapes, Pauli codes, coefficient values, generator, or theta are
        invalid.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )
```

### Step 3

merge_pauli_terms

Goal
----
Merge duplicate Pauli strings with deterministic ordering and stable real summation.

```python
from numbers import Real

import math
import numpy as np


def merge_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    zero_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate strings in first-occurrence order.

    Coefficients belonging to identical Pauli rows are summed with a
    cancellation-stable real summation. A merged row is retained only when the
    strict condition abs(sum) > zero_tol holds.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Integer array of shape (m, n) with entries in {0, 1, 2, 3}.
    coefficients : np.ndarray
        Finite real array of shape (m,).
    zero_tol : float, optional
        Finite nonnegative cancellation threshold.

    Returns
    -------
    merged_codes : np.ndarray
        Unique Pauli rows in first-occurrence order, with shape (r, n).
    merged_coefficients : np.ndarray
        Float array of shape (r,) containing the merged coefficients.

    Raises
    ------
    ValueError
        If shapes, Pauli codes, coefficients, or zero_tol are invalid.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )
```

### Step 4

truncate_pauli_terms

Goal
----
Apply deterministic fixed-cardinality truncation to a sparse Pauli expansion.

```python
from numbers import Integral

import numpy as np


def truncate_pauli_terms(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    max_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain at most the largest max_terms coefficient magnitudes.

    The input rows must already be unique. Ranking is stable: equal magnitudes
    are resolved by the original row index. After selecting the survivors,
    return them in their original input order.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Unique integer Pauli rows of shape (m, n).
    coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    max_terms : int
        Positive maximum number of retained terms; bool is not accepted.

    Returns
    -------
    truncated_codes : np.ndarray
        Selected Pauli rows, with shape (min(m, max_terms), n).
    truncated_coefficients : np.ndarray
        Selected float coefficients in original row order.

    Raises
    ------
    ValueError
        If the arrays are invalid, rows are not unique, or max_terms is not
        a positive integer.
    """
    return (
        np.empty((0, pauli_codes.shape[1]), dtype=np.int8),
        np.empty(0, dtype=float),
    )
```

### Step 5

normalize_pauli_trace

Goal
----
Normalize a sparse Pauli expansion by its unique identity coefficient.

```python
from numbers import Real

import numpy as np


def normalize_pauli_trace(
    pauli_codes: np.ndarray,
    coefficients: np.ndarray,
    identity_tol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale a unique Pauli expansion so its identity coefficient is one.

    If R = sum_P c_P P on n qubits, then the corresponding trace-one
    state is R / 2**n once c_I = 1. The row order is unchanged.

    Parameters
    ----------
    pauli_codes : np.ndarray
        Unique integer Pauli rows of shape (m, n).
    coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    identity_tol : float, optional
        Finite nonnegative lower bound for the absolute identity coefficient.

    Returns
    -------
    normalized_codes : np.ndarray
        Copy of the input Pauli rows.
    normalized_coefficients : np.ndarray
        Float coefficients divided by the unique identity coefficient.

    Raises
    ------
    ValueError
        If the arrays are invalid, rows are not unique, there is not exactly
        one identity row, or its coefficient is not finite and larger than
        identity_tol in magnitude.
    """
    return (
        np.empty_like(pauli_codes, dtype=np.int8),
        np.empty_like(coefficients, dtype=float),
    )
```

### Step 6

pauli_coefficient_overlap

Goal
----
Evaluate a cancellation-stable normalized Hilbert--Schmidt overlap of two sparse Pauli expansions.

```python
import math
import numpy as np


def pauli_coefficient_overlap(
    left_codes: np.ndarray,
    left_coefficients: np.ndarray,
    right_codes: np.ndarray,
    right_coefficients: np.ndarray,
) -> complex:
    """Compute the normalized Hilbert--Schmidt overlap from sparse coefficients.

    Both expansions must use unique Pauli rows with the same qubit width. Match
    equal rows by their encoded support rather than by array position. Accumulate
    real and imaginary contributions separately with cancellation-resistant
    summation so that a finite representable residual is retained.

    Parameters
    ----------
    left_codes : np.ndarray
        Unique integer Pauli rows for A, with shape (m, n).
    left_coefficients : np.ndarray
        Finite real or complex coefficients of shape (m,).
    right_codes : np.ndarray
        Unique integer Pauli rows for B, with shape (r, n).
    right_coefficients : np.ndarray
        Finite real or complex coefficients of shape (r,).

    Returns
    -------
    overlap : complex
        Native Python complex value of sum_P conj(a_P) * b_P.

    Raises
    ------
    ValueError
        If array shapes, code values, uniqueness, widths, or coefficient values
        are invalid.
    """
    return 0j
```

### Step 7

observable_state_overlap

Goal
----
Evaluate the phase-aware real overlap entering a squared-state observable numerator.

```python
from numbers import Real

import math
import numpy as np


def observable_state_overlap(
    observable_codes: np.ndarray,
    observable_coefficients: np.ndarray,
    state_codes: np.ndarray,
    state_coefficients: np.ndarray,
    imag_tol: float = 1e-12,
) -> float:
    """Compute the sparse normalized trace overlap entering Tr(O R**2).

    Both O and R use unique Pauli rows and finite real coefficients. Form every
    product in observable-left order with its complete tensor-product Pauli
    phase, match the product against the support of R, and accumulate all real
    and imaginary contributions cancellation-resistently. Apply imag_tol only to
    the final summed imaginary component, since nonzero termwise phases may
    cancel in the Hermitian contraction.

    Parameters
    ----------
    observable_codes : np.ndarray
        Unique integer Pauli rows for O, with shape (m, n).
    observable_coefficients : np.ndarray
        Finite real coefficients of shape (m,).
    state_codes : np.ndarray
        Unique integer Pauli rows for R, with shape (r, n).
    state_coefficients : np.ndarray
        Finite real coefficients of shape (r,).
    imag_tol : float, optional
        Finite nonnegative relative tolerance for the final imaginary residual.

    Returns
    -------
    numerator : float
        Native Python float equal to the normalized trace overlap.

    Raises
    ------
    ValueError
        If arrays, Pauli codes, uniqueness, widths, coefficients, imag_tol, or
        the final reality check are invalid.
    """
    return 0.0
```

### Step 8

run_itpp_squared_energy

Goal
----
Run the complete sequential sparse imaginary-time Pauli-propagation pipeline and return the squared-state energy.

```python
from numbers import Integral, Real

import math
import numpy as np


def run_itpp_squared_energy(
    n_qubits: int,
    coupling: float,
    field: float,
    delta_tau: float,
    final_tau: float,
    max_terms: int,
    zero_tol: float = 1e-14,
) -> float:
    """Run sequential sparse imaginary-time propagation and a squared-state estimator.

    Build the ordered open-chain Hamiltonian, start from the identity expansion,
    and use ceil(final_tau / delta_tau) full first-order Trotter layers. Call
    build_open_tfim_pauli_terms once, then for every elementary generator
    call propagate_itpp_scale_free, merge_pauli_terms,
    truncate_pauli_terms, and normalize_pauli_trace in that order.
    Finally call pauli_coefficient_overlap and
    observable_state_overlap to return Tr(H R**2) / Tr(R**2) in
    normalized Pauli units.

    Parameters
    ----------
    n_qubits : int
        Positive number of qubits.
    coupling : float
        Finite real nearest-neighbour coupling.
    field : float
        Finite real transverse-field strength.
    delta_tau : float
        Finite strictly positive Trotter step size.
    final_tau : float
        Finite nonnegative target imaginary time.
    max_terms : int
        Positive fixed-rank Pauli truncation cap.
    zero_tol : float, optional
        Finite nonnegative threshold used for merged cancellation and identity
        normalization.

    Returns
    -------
    energy : float
        Native Python float equal to Tr(H R**2) / Tr(R**2).

    Raises
    ------
    ValueError
        If scalar controls are invalid, an intermediate expansion cannot be
        trace-normalized, or the final overlap ratio is non-finite.
    """
    return 0.0
```
