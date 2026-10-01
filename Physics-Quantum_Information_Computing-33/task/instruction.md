# Physics-Quantum_Information_Computing-33

## Background

Nonstabilizerness, or magic, is the resource that makes a quantum state useful beyond stabilizer simulation. Stabilizer methods quantify that resource by comparing a state against the efficiently simulable stabilizer set, including recent constructions that treat Pauli data as the spectrum of a fictitious statistical-mechanics ensemble. Temperature-dependent monotones extracted from that ensemble interpolate between different magic diagnostics.

## Problem

Nonstabilizerness is the resource that separates universal quantum computation from classically efficiently simulable stabilizer dynamics. A 2026 statistical-mechanics construction of magic maps a quantum state's Pauli expectations onto a fictitious ensemble and extracts a temperature-dependent work monotone relative to the unique n-qubit stabilizer reference; the construction is stated for pure states and then extended to mixed states. The primary inputs are the state and an inverse-temperature parameter β; the output is the stabilizer work as a single real number.

Consider a single qubit prepared in a noisy magic state with Bloch vector r = (0.6, 0.5, 0.3), that is ρ = (I + 0.6 X + 0.5 Y + 0.3 Z)/2, at inverse temperature β = 2. Return the stabilizer work W₂(ρ) of this mixed state as a single number, using the construction's mixed-state extension.

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

single_qubit_pauli

Goal
----
Return a phase-free single-qubit Pauli matrix.

```python
import numpy as np

def single_qubit_pauli(kind: int) -> np.ndarray:
    """Return a phase-free single-qubit Pauli matrix.

    The integer ``kind`` selects an element of {I, X, Y, Z} by
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z. These are the generators of the
    n-qubit Pauli set P_n = {I, X, Y, Z}^{otimes n} used to build the
    signed Pauli spectrum.

    Parameters
    ----------
    kind : int
        Pauli label in {0, 1, 2, 3}.

    Returns
    -------
    P : np.ndarray
        Complex array of shape (2, 2).
    """
    return P
```

### Step 2

n_qubit_pauli

Goal
----
Return one n-qubit Pauli string in canonical enumeration order.

```python
import numpy as np

def n_qubit_pauli(n: int, index: int) -> np.ndarray:
    """Return one n-qubit Pauli string in canonical enumeration order.

    Index ``k`` runs through 0, ..., 4**n - 1. The base-4 digits of ``k``,
    least-significant digit first, label qubits 0, 1, ..., n-1 with
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z. The operator is the Kronecker product
    with qubit n-1 as the leftmost factor, so computational-basis bit 0 is
    the least-significant bit. In particular ``k = 0`` is the identity and
    ``n = 1`` recovers the four single-qubit Paulis.

    Parameters
    ----------
    n : int
        Number of qubits, n >= 1.
    index : int
        Enumeration index in {0, 1, ..., 4**n - 1}.

    Returns
    -------
    P : np.ndarray
        Complex array of shape (2**n, 2**n).
    """
    return P
```

### Step 3

pauli_spectrum

Goal
----
Signed Pauli spectrum of an n-qubit density matrix.

```python
import numpy as np

def pauli_spectrum(psi: np.ndarray, n: int) -> np.ndarray:
    """Signed Pauli spectrum of an n-qubit density matrix.

    For P running through the canonical enumeration of P_n, return the
    real numbers Tr(psi P). The signed convention is required: do not
    replace Tr(psi P) by |Tr(psi P)|. The identity contribution is
    Tr(psi I) = 1 for a normalized state and is retained.

    Parameters
    ----------
    psi : np.ndarray
        Density matrix of shape (2**n, 2**n).
    n : int
        Number of qubits.

    Returns
    -------
    spec : np.ndarray
        1-D real array of length 4**n in canonical Pauli order.
    """
    return spec
```

### Step 4

bloch_density_matrix

Goal
----
Single-qubit density matrix from a Bloch vector.

```python
import numpy as np

def bloch_density_matrix(r: np.ndarray) -> np.ndarray:
    """Single-qubit density matrix with Bloch vector ``r``.

    Return rho = (I + r_x X + r_y Y + r_z Z)/2 using the phase-free
    Paulis of ``single_qubit_pauli``. Supported domain: ``r`` is a real
    array of shape (3,) with finite entries and Euclidean norm at most
    1 (a tolerance of 1e-9 above 1 is accepted). Raise ValueError
    otherwise.

    Parameters
    ----------
    r : np.ndarray
        Real Bloch vector of shape (3,).

    Returns
    -------
    rho : np.ndarray
        Complex Hermitian array of shape (2, 2) with unit trace.
    """
    return rho
```

### Step 5

stabilizer_partition_function

Goal
----
Stabilizer partition function of a pure state's Pauli spectrum.

```python
import numpy as np

def stabilizer_partition_function(spectrum: np.ndarray, beta: float) -> float:
    """Stabilizer partition function of a pure state's Pauli spectrum.

    Evaluate Definition 3 of the source paper on the supplied signed
    Pauli spectrum at inverse temperature ``beta``. The identity
    contribution is included. Recover the paper's generating function.
    This is the pure-state quantity; do not apply any mixed-state
    extension here.

    Supported domain: ``spectrum`` is a 1-D real array of length 4**n
    (n >= 1) with every entry in [-1, 1]; ``beta`` is finite and >= 0.
    The result must be finite for every admissible ``beta``, including
    large ``beta`` where a direct ``cosh`` overflows. Raise ValueError
    for inputs outside this domain.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D real array of length 4**n with entries in [-1, 1].
    beta : float
        Inverse-temperature parameter, finite and >= 0.

    Returns
    -------
    z : float
        Stabilizer partition function Z_beta.
    """
    return 0.0
```

### Step 6

decomposition_spf

Goal
----
Decomposition-averaged stabilizer partition function of a single-qubit mixed state.

```python
import numpy as np

def decomposition_spf(
    r: np.ndarray, weights: np.ndarray, bloch_vectors: np.ndarray, beta: float
) -> float:
    """Average pure-state partition function over one convex decomposition.

    For the decomposition rho = sum_i weights[i] |phi_i><phi_i| of the
    single-qubit state with Bloch vector ``r``, where |phi_i> is the
    pure state with unit Bloch vector ``bloch_vectors[i]``, return
    sum_i weights[i] Z_beta(phi_i) with Z_beta the pure-state
    partition function of ``stabilizer_partition_function`` evaluated
    on the signed Pauli spectrum of each |phi_i><phi_i| (build the
    density matrices with ``bloch_density_matrix`` and the spectra with
    ``pauli_spectrum``).

    Supported domain: ``r`` finite real of shape (3,) with norm at most
    1; ``weights`` finite real of shape (m,), m >= 1, nonnegative and
    summing to 1 within 1e-9; ``bloch_vectors`` finite real of shape
    (m, 3) with every row of unit norm within 1e-9; the decomposition
    must reproduce ``r``, i.e. max_j |sum_i weights[i] bloch_vectors[i, j]
    - r[j]| <= 1e-8; ``beta`` finite and >= 0. Raise ValueError for
    inputs outside this domain, including a decomposition that does not
    reproduce ``r``.

    Parameters
    ----------
    r : np.ndarray
        Bloch vector of the decomposed state, shape (3,).
    weights : np.ndarray
        Decomposition probabilities, shape (m,).
    bloch_vectors : np.ndarray
        Unit Bloch vectors of the pure components, shape (m, 3).
    beta : float
        Inverse-temperature parameter, finite and >= 0.

    Returns
    -------
    z_avg : float
        sum_i weights[i] Z_beta(phi_i).
    """
    return 0.0
```

### Step 7

roof_spf

Goal
----
Mixed-state stabilizer partition function of a single qubit (Definition 8).

```python
import numpy as np

def roof_spf(r: np.ndarray, beta: float, grid_points: int = 4000) -> float:
    """Mixed-state stabilizer partition function of a single qubit.

    Return Z_beta(rho) of Definition 8 of the source paper for the
    single-qubit state with Bloch vector ``r``: the supremum over all
    convex decompositions rho = sum_i p_i |phi_i><phi_i| into pure
    states of sum_i p_i Z_beta(phi_i), with Z_beta the pure-state
    quantity of ``stabilizer_partition_function`` (evaluate the
    components through ``decomposition_spf``). The result must equal
    the pure-state value when ``r`` is a unit vector, must equal the
    stabilizer value e^{-beta}(2 cosh beta + 2) when ``r`` lies inside
    the stabilizer octahedron |r_x| + |r_y| + |r_z| <= 1, and must be
    at least the eigendecomposition average for every ``r``.

    The supremum is attained by a decomposition with at most four
    components; locate it globally (for example a linear program over
    ``grid_points`` candidate pure states on the Bloch sphere, then a
    continuous refinement of the support) and return the value to an
    absolute accuracy of 1e-9 or better.

    Supported domain: ``r`` finite real of shape (3,) with Euclidean
    norm at most 1; ``beta`` finite and >= 0; ``grid_points`` integer
    >= 500. Raise ValueError otherwise.

    Parameters
    ----------
    r : np.ndarray
        Bloch vector of the state, shape (3,).
    beta : float
        Inverse-temperature parameter, finite and >= 0.
    grid_points : int, optional
        Size of the initial candidate set of pure states, default 4000.

    Returns
    -------
    z_roof : float
        Mixed-state stabilizer partition function Z_beta(rho).
    """
    return 0.0
```

### Step 8

noisy_magic_roof_work

Goal
----
Stabilizer work of a noisy single-qubit magic state under the mixed-state extension.

```python
import numpy as np

def noisy_magic_roof_work(r: np.ndarray = (0.6, 0.5, 0.3), beta: float = 2.0) -> float:
    """Mixed-state stabilizer work of a single noisy magic qubit.

    Build the mixed-state partition function Z_beta(rho) of the state
    with Bloch vector ``r`` with ``roof_spf`` (Definition 8 of the
    source paper), form its core Z_beta(rho) - 4 e^{-beta} (Definition
    4), and return the stabilizer work of Definition 6,
    W_beta(rho) = log2[Z^c_beta(STAB_1) / Z^c_beta(rho)] with the
    single-qubit stabilizer reference Z^c_beta(STAB_1) =
    2 e^{-beta}(cosh beta - 1) and the paper's base-2 logarithm.

    The result must be exactly 0 (return 0.0) whenever ``r`` lies in
    the stabilizer octahedron |r_x| + |r_y| + |r_z| <= 1, must reduce
    to the pure-state work when ``r`` is a unit vector, and must never
    exceed the work computed from the eigendecomposition average.

    Supported domain: ``r`` finite real of shape (3,) with Euclidean
    norm at most 1; ``beta`` finite with 0.01 <= beta <= 10. Raise
    ValueError otherwise.

    Parameters
    ----------
    r : np.ndarray, optional
        Bloch vector of the state, default (0.6, 0.5, 0.3).
    beta : float, optional
        Inverse-temperature parameter, default 2.0.

    Returns
    -------
    work : float
        Mixed-state stabilizer work W_beta(rho).
    """
    return 0.0
```
