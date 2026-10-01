# Physics-Quantum_Information_Computing-1

## Background

Time-evolution quantum Krylov methods approximate low-energy eigenstates by spanning a compact nonorthogonal subspace with repeatedly evolved copies of a reference state. They can avoid the deep coherent circuits required by phase estimation, but direct measurement of every projected Hamiltonian term can be expensive, and the overlap matrix becomes ill-conditioned when time-evolved basis vectors approach linear dependence.

The selected approach replaces term-by-term projected-Hamiltonian measurements with a symmetric finite-difference reconstruction of the generator from shifted time-evolution unitaries. It balances statistical and truncation errors through an optimized time shift, reduces the relevant spectral norm by centering a known symmetry-sector energy interval, and reuses the same propagator data to infer higher projected Hamiltonian powers.

Because the Krylov basis is nonorthogonal, a coefficient vector expressed in the original basis is normalized by the overlap metric rather than by its Euclidean length. Equivalently, one may work in canonically orthonormalized coordinates, but then the vector and every projected operator must be transformed into that same coordinate system before taking moments.

A classical determinant-based Lanczos continuation uses the resulting moment sequence to refine the lowest Ritz value. Higher moments are especially sensitive to finite-sampling perturbations, so the continuation retains only a resolved positive-overlap subspace and terminates at the last accepted order when the moment data cease to define a physically admissible Jacobi extension.

## Problem

Estimate the lowest energy of a quantum many-body Hamiltonian from a supplied time-propagator sequence using the recent symmetric time-shift Krylov reconstruction and its Hamiltonian-moment Lanczos mitigation. Use binary64 arithmetic, $\hbar=1$, energies in Hartree (Ha), 0-based matrix and Krylov indices, central-difference degree $J=3$, Krylov dimension $n=4$, total shot count $M=1.0\times10^{14}$, symmetry-sector bounds $E_{\min}=-1.2$ Ha and $E_{\max}=1.8$ Ha, stride $\tau/\delta_t=4$, and relative overlap cutoff $10^{-4}$; do not round intermediate values. The entries below are the nonnegative-time expectations $g[m]=\langle\phi_0|e^{-i(H-c)m\delta_t}|\phi_0\rangle$ for $m=0,\ldots,15$ after applying the method's symmetry-sector energy shift, and negative-time expectations are fixed by unitarity:

```text
[
1.0+0.0j,
0.99429356005856029-0.025590728300645980j,
0.97730685925486915-0.050590494877476529j,
0.94943860829937488-0.074423809017078665j,
0.91133730900167487-0.096544241297493519j,
0.86388714249963516-0.116449649912794870j,
0.80818080703044082-0.133693203305266430j,
0.74548930367185484-0.147895607047029390j,
0.67722726275148382-0.158754282232542730j,
0.60491163012666282-0.166049446589216800j,
0.53012217788209992-0.169649646725830630j,
0.45445491616160788-0.169514288978398760j,
0.37948284322780795-0.165693225887655530j,
0.30671045822235593-0.158324864969491470j,
0.23753537514773893-0.147629912812195420j,
0.17321244307405181-0.133906119606736620j
]
```

Treat the resolved Ritz coefficients as coordinates in the original nonorthogonal Krylov basis, so physical normalization and every projected moment use the overlap metric; an orthonormal-coordinate reformulation is equivalent only when the state and all projected operators are transformed consistently. Apply the source method through derivative order $2J$, use Hermitian symmetrization only to remove binary64 roundoff asymmetry, resolve the positive overlap subspace at the supplied cutoff, and run the paper-specific determinant moment continuation through at most order $J$ under its noise-aware termination convention. In `<reasoning>`, state the method-specific identities for propagator differentiation, optimized time-shift selection, sector-energy centering, and the determinant recurrence, then report the optimized time shift, retained overlap modes, raw Ritz energy, overlap-metric normalization check, moment sequence, accepted Lanczos order, and decisive stopping scalar. Return the final mitigated energy in the original energy origin, rounded to 12 digits after the decimal point.

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

compute_central_difference_coefficients

Goal
----
Construct the symmetric central finite-difference coefficient rows for derivative orders $q=1,\ldots,q_{\max}$, where $J=\texttt{degree}$ and $q_{\max}=\texttt{max\_order}\le 2J$, on the integer stencil $j=-J,\ldots,J$. Each row must differentiate every polynomial of degree at most $2J$ exactly at the origin and obey centered-stencil parity: odd derivative rows are antisymmetric and even derivative rows are symmetric.

```python
from numbers import Integral

import numpy as np

STEP_NAME = "compute_central_difference_coefficients"
STEP_ID = "01"
STEP_DESCRIPTION = r"""Construct the symmetric central finite-difference coefficient rows for derivative orders 1 through `max_order` on the integer stencil from `-degree` through `degree`. The row for derivative order `q` must differentiate every polynomial of degree at most `2 * degree` exactly at the origin and must obey the parity of a centered derivative stencil."""
STEP_SCIENTIFIC_BACKGROUND = r"""The time-shift Krylov reconstruction obtains powers of a Hamiltonian from derivatives of a unitary propagator at zero time. On a symmetric stencil, the required weights are the derivatives at the origin of the Lagrange cardinal polynomials, equivalently the unique solution of the polynomial-exactness moment equations. Odd derivative rows are antisymmetric and even derivative rows are symmetric; enforcing that parity removes roundoff-scale violations without changing the finite-difference order."""
EXPECTED_RETURN_LINE = "np.ndarray of shape (max_order, 2 * degree + 1), with row q - 1 containing the float64 weights for derivative order q on nodes -degree,...,degree"
IS_FINAL_ORCHESTRATOR = False


def compute_central_difference_coefficients(
    degree: int,
    max_order: int,
) -> np.ndarray:
    """Construct centered finite-difference coefficient rows.

    Parameters
    ----------
    degree : int
        Positive half-width of the symmetric stencil.
    max_order : int
        Largest derivative order, from 1 through ``2 * degree``.

    Returns
    -------
    coefficients : np.ndarray
        Float64 array of shape ``(max_order, 2 * degree + 1)``. Row
        ``q - 1`` contains the weights for derivative order ``q`` on nodes
        ``-degree, ..., degree``.

    Raises
    ------
    ValueError
        If ``degree`` or ``max_order`` is not an integer in the stated range.
    """
    return np.empty((max_order, 2 * degree + 1), dtype=np.float64)
```

### Step 2

compute_msd_preprocessing

Goal
----
Compute the symmetry-sector energy-center shift, the minimized shifted spectral norm, the optimized finite-difference time shift, and the two coefficient-dependent constants that balance finite-sampling and truncation perturbations. Evaluate the optimized time shift through logarithms so that a representable result remains finite even when direct powers of the spectral norm would overflow.

```python
import math

from numbers import Integral, Real

import numpy as np

def compute_msd_preprocessing(
    first_derivative_coefficients: np.ndarray,
    krylov_dimension: int,
    degree: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
) -> np.ndarray:
    """Compute the energy shift and optimized time-shift data.

    Parameters
    ----------
    first_derivative_coefficients : np.ndarray
        Real centered first-derivative weights of length ``2 * degree + 1``.
    krylov_dimension : int
        Positive Krylov dimension ``n``.
    degree : int
        Positive central-difference degree ``J``.
    total_shots : float
        Positive finite total shot count ``M``.
    sector_minimum : float
        Finite lower energy bound for the symmetry sector.
    sector_maximum : float
        Finite upper energy bound, strictly larger than ``sector_minimum``.

    Returns
    -------
    preprocessing : np.ndarray
        Float64 array ``[shift, shifted_norm, delta_t, alpha_nJ, beta_nJ]``.

    Raises
    ------
    ValueError
        If an input has an invalid type, shape, range, or non-finite value, or
        if the optimized result is not representable as finite float64.
    """
    return np.empty(5, dtype=np.float64)
```

### Step 3

build_shifted_propagators

Goal
----
Build the propagator tensor for symmetric finite-difference shifts $j=-J,\ldots,J$, where $J=\texttt{degree}$. For row index $k'$ and column index $k$, use the signed sample index $m=\texttt{stride}\,(k-k')+j$ and set the corresponding entry to $g_m$. Read $g_m$ from the supplied nonnegative-time scalar sequence for $m\ge 0$ and use $g_m=\overline{g_{-m}}$ for $m<0$.

```python
from numbers import Integral

import numpy as np

def build_shifted_propagators(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
) -> np.ndarray:
    """Build Toeplitz propagator matrices for symmetric shifts.

    Parameters
    ----------
    g_nonnegative : np.ndarray
        Complex sequence ``g[m]`` for nonnegative integer indices starting at 0.
    krylov_dimension : int
        Positive matrix dimension ``n``.
    degree : int
        Positive shift half-width ``J``.
    stride : int
        Positive integer ratio ``tau / delta_t``.

    Returns
    -------
    propagators : np.ndarray
        Complex128 array of shape ``(2 * degree + 1, n, n)`` ordered by
        ``j = -degree, ..., degree``.

    Raises
    ------
    ValueError
        If dimensions are invalid, samples are non-finite, or the sequence is
        too short for every required signed index.
    """
    return np.empty(
        (2 * degree + 1, krylov_dimension, krylov_dimension),
        dtype=np.complex128,
    )
```

### Step 4

reconstruct_projected_matrices

Goal
----
Use the centered finite-difference rows and the symmetric-shift propagator tensor to reconstruct the overlap matrix and approximate projected powers of the shifted Hamiltonian. Let $\delta_t=\texttt{delta\_t}$. Obtain the overlap matrix from the zero-shift slice. For derivative order $q$, multiply the corresponding weighted propagator sum by $i^q/\delta_t^q$. Replace the overlap matrix and every reconstructed power matrix $A$ by its Hermitian part $(A+A^\dagger)/2$.

```python
from numbers import Real

import numpy as np

def reconstruct_projected_matrices(
    propagators: np.ndarray,
    coefficients: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct overlap and projected shifted-Hamiltonian powers.

    Parameters
    ----------
    propagators : np.ndarray
        Complex array of shape ``(2 * degree + 1, n, n)`` ordered by symmetric
        shift labels.
    coefficients : np.ndarray
        Real array of shape ``(q_max, 2 * degree + 1)`` with derivative rows
        ordered from 1 through ``q_max``.
    delta_t : float
        Positive finite time shift.

    Returns
    -------
    overlap : np.ndarray
        Hermitian complex128 overlap matrix of shape ``(n, n)``.
    power_matrices : np.ndarray
        Hermitian complex128 array of shape ``(q_max, n, n)`` whose row
        ``q - 1`` approximates the projected ``q``-th power.

    Raises
    ------
    ValueError
        If shapes are inconsistent, entries are non-finite, ``delta_t`` is not
        positive and finite, or a finite complex128 result cannot be formed.
    """
    return (
        np.empty((propagators.shape[1], propagators.shape[1]), dtype=np.complex128),
        np.empty(
            (coefficients.shape[0], propagators.shape[1], propagators.shape[1]),
            dtype=np.complex128,
        ),
    )
```

### Step 5

solve_resolved_krylov_ground_state

Goal
----
Regularize the nonorthogonal projected eigenproblem by diagonalizing the overlap matrix, retaining precisely the modes whose eigenvalues satisfy the strict relative rule $s_i > \texttt{relative\_cutoff}\,s_{\max}$, and solving in that positive resolved subspace. Return the lowest shifted energy and the corresponding full-basis coefficient vector in the original nonorthogonal Krylov coordinates, normalized so that $v^\dagger S v=1$, with a deterministic phase.

```python
from numbers import Real

import numpy as np

def solve_resolved_krylov_ground_state(
    overlap: np.ndarray,
    shifted_hamiltonian: np.ndarray,
    relative_cutoff: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the thresholded projected generalized eigenproblem.

    Parameters
    ----------
    overlap : np.ndarray
        Hermitian overlap matrix of shape ``(n, n)``.
    shifted_hamiltonian : np.ndarray
        Hermitian projected shifted-Hamiltonian matrix of shape ``(n, n)``.
    relative_cutoff : float
        Finite scalar strictly between 0 and 1. A mode is retained only when
        its eigenvalue is strictly greater than ``relative_cutoff * s_max``.

    Returns
    -------
    shifted_energy : float
        Lowest generalized eigenvalue in the retained subspace.
    state_coefficients : np.ndarray
        Complex128 full-basis coefficient vector of length ``n`` in the
        original nonorthogonal Krylov coordinates, normalized to
        ``state_coefficients.conj() @ overlap @ state_coefficients = 1``
        and phase fixed deterministically.
    overlap_eigenvalues : np.ndarray
        Ascending float64 overlap eigenvalues.
    retained_indices : np.ndarray
        Int64 indices of the strictly retained overlap eigenmodes.

    Raises
    ------
    ValueError
        If matrix shapes, Hermiticity, finiteness, cutoff, or the resolved
        positive subspace is invalid.
    """
    return (
        float("nan"),
        np.empty(len(overlap), dtype=np.complex128),
        np.empty(len(overlap), dtype=np.float64),
        np.empty(0, dtype=np.int64),
    )
```

### Step 6

compute_projected_hamiltonian_moments

Goal
----
Evaluate shifted-Hamiltonian moments from the projected power matrices, the overlap matrix, and the approximate Krylov ground-state coefficients in the original nonorthogonal basis. Set the zeroth moment to one and use the physical quotient $\mu_q=v^\dagger M^{(q)}v/(v^\dagger S v)$ for every supplied power matrix, returning only real values after checking roundoff-scale imaginary residues.

```python
import numpy as np

def compute_projected_hamiltonian_moments(
    power_matrices: np.ndarray,
    overlap: np.ndarray,
    state_coefficients: np.ndarray,
) -> np.ndarray:
    """Compute physical moments in a nonorthogonal projected basis.

    Parameters
    ----------
    power_matrices : np.ndarray
        Hermitian complex array of shape ``(q_max, n, n)`` ordered by powers
        1 through ``q_max`` in the original Krylov coordinates.
    overlap : np.ndarray
        Hermitian complex overlap matrix of shape ``(n, n)`` defining the
        physical metric in those coordinates.
    state_coefficients : np.ndarray
        Nonzero complex coefficient vector of length ``n`` in the same
        original Krylov coordinates.

    Returns
    -------
    moments : np.ndarray
        Float64 vector of length ``q_max + 1`` beginning with ``mu_0 = 1``.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, positive overlap norm, or
        real-valued moment consistency is invalid.
    """
    return np.empty(power_matrices.shape[0] + 1, dtype=np.float64)
```

### Step 7

run_moment_lanczos

Goal
----
Continue the energy estimate with the determinant form of the moment-Lanczos recurrence through at most $max_order$. Use the task conventions $L_-1 = 1$, $L_0 = 1$, $M_-1 = 0$, and $M_0 = mu_1$; after each tridiagonal energy is formed, terminate before accepting a strict energy increase. After an accepted order below $max_order$, evaluate the next squared off-diagonal coefficient before attempting another order and terminate if it is negative or exactly zero. Do not evaluate a next off-diagonal candidate after accepting $max_order$. Return stop code 0 for completion, 1 for a negative square, 2 for an energy increase, and 3 for exact zero termination.

```python
import math

from numbers import Integral, Real

import numpy as np

def run_moment_lanczos(
    moments: np.ndarray,
    max_order: int,
    energy_shift: float,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run determinant-based moment-Lanczos energy mitigation.

    Parameters
    ----------
    moments : np.ndarray
        Real finite sequence ``[mu_0, ..., mu_{2 * max_order}]`` with
        ``mu_0 = 1``.
    max_order : int
        Positive maximum tridiagonal order.
    energy_shift : float
        Finite scalar added to every shifted-Hamiltonian Ritz energy.

    Returns
    -------
    final_energy : float
        Last accepted lowest Ritz energy after adding ``energy_shift``.
    accepted_order : int
        Order of the last accepted tridiagonal matrix.
    trial_energies : np.ndarray
        One-dimensional float64 original-energy trials in chronological order, including an
        energy-increase trial if that trial triggers stop code 2.
    beta_squared : np.ndarray
        One-dimensional float64 squared off-diagonal candidates evaluated after
        accepted orders below ``max_order``, before attempting another order.
    alpha_values : np.ndarray
        One-dimensional float64 diagonal recurrence values in chronological order.
    stop_code : int
        ``0`` completed, ``1`` negative beta square, ``2`` strict energy
        increase, or ``3`` exact zero beta square.

    Raises
    ------
    ValueError
        If inputs are invalid, moments are insufficient, or a determinant
        denominator becomes singular before a defined termination branch.
    """
    return (
        float("nan"),
        0,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        0,
    )
```

### Step 8

run_msd_energy_pipeline

Goal
----
Run the complete deterministic symmetric time-shift Krylov and moment-Lanczos pipeline, with central-difference degree $J=\texttt{degree}$. The public orchestrator must call each preceding public step in order, use derivative rows through order $2J$, use derivative order $q=1$ for preprocessing and the projected Hamiltonian, solve with the supplied strict overlap cutoff, compute moments through order $2J$, and use maximum Lanczos order $J$; return only the final mitigated energy.

```python
import numpy as np

def run_msd_energy_pipeline(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
    total_shots: float,
    sector_minimum: float,
    sector_maximum: float,
    relative_cutoff: float,
) -> float:
    """Run the complete shifted-propagator Krylov energy pipeline.

    Parameters
    ----------
    g_nonnegative : np.ndarray
        Finite complex sequence of nonnegative-time propagator expectations.
    krylov_dimension : int
        Positive Krylov dimension ``n``.
    degree : int
        Positive central-difference degree ``J``.
    stride : int
        Positive integer ratio ``tau / delta_t``.
    total_shots : float
        Positive finite shot count used in time-shift optimization.
    sector_minimum : float
        Finite lower energy bound of the symmetry sector.
    sector_maximum : float
        Finite upper energy bound, strictly larger than ``sector_minimum``.
    relative_cutoff : float
        Finite overlap cutoff strictly between 0 and 1.

    Returns
    -------
    final_energy : float
        Final moment-Lanczos mitigated ground-state energy in the original
        energy origin.

    Raises
    ------
    ValueError
        If any preceding scientific step rejects the supplied inputs or its
        deterministic intermediate result.
    """
    return float("nan")
```
