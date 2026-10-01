# Spectral-Band Entanglement in a Discrete QSSF Simulation

## Background

# Quantum fluctuations in nonlinear optical waveguides

Dispersive optical waveguides reshape propagating pulses through the combined
effects of chromatic dispersion and intensity-dependent refractive index. In
quantum descriptions, fluctuations around a coherent mean field can develop
correlations between different temporal or spectral modes. These correlations
are relevant to squeezing, quantum information processing and the generation of
nonclassical light.

Gaussian approximations describe such fluctuations through their first and
second moments. A selected group of modes can be in a mixed state even when the
complete field is pure, because it is entangled with the modes outside that
group. Reliable numerical predictions therefore require both accurate
propagation and a consistent treatment of canonical commutation relations,
subsystem selection and covariance normalization.

## Problem

Calculate the von Neumann entanglement entropy of a designated spectral band after propagation of a coherent sech pulse in a lossless dispersive Kerr waveguide model. Use the finite-step, frozen-coefficient quantum split-step Fourier (QSSF) method to evolve vacuum fluctuations concurrently with the classical mean field. The target is the Gaussian state produced by this prescribed discrete evolution at the specified step size.

Discretize the temporal domain using $N_t = 128$ grid points over a temporal window of duration $T = 40.0$ in normalized units. Propagate $A(0, t) = A_0 \operatorname{sech}(A_0 t)$ with amplitude $A_0 = 1.5$ over $n_{\text{steps}} = 40$ steps of size $\Delta z = 0.01$. Use the spectral symbol $D(\omega)=d_2\omega^2/2+\beta_3\omega^3$, positive linear phase $\exp(iD\Delta z)$ and positive Kerr phase, with $d_2 = 1.0$, $\beta_3 = 0.08$, and $\gamma = 1.0$.

Use the QSSF source's dispersion–nonlinear-offset rule to center a spectral window $\mathcal{W}$ of bandwidth $\Delta\omega = 3.0$; here this rule defines the analyzed band and carries no assumption that the input is a stationary soliton. Compute the reduced Gaussian covariance and its Williamson spectrum. Report the band's von Neumann entropy in nats at $z = 0.40$, to absolute accuracy $10^{-7}$.

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

qssf_dispersion_propagator

Goal
----
Evaluate the dispersion symbol and its linear half-step propagator on the supplied angular-frequency grid, and find the unique real center prescribed by the dispersion–nonlinear-offset window rule. Return all three quantities as the documented numerical array.

```python
def qssf_dispersion_propagator(
    w: "np.ndarray",
    dz: float,
    d2: float = 1.0,
    beta3: float = 0.08,
    A0: float = 1.5,
    gamma: float = 1.0,
) -> "np.ndarray":
    """
    Compute the linear half-step propagator, dispersion symbol, and
    prescribed analysis-band center.

    Parameters
    ----------
    w : "np.ndarray"
        Array of angular frequencies.
    dz : float
        Spatial step size.
    d2 : float
        Group velocity dispersion coefficient.
    beta3 : float
        Third-order dispersion coefficient.
    A0 : float
        Sech-pulse amplitude.
    gamma : float
        Kerr nonlinearity.

    Returns
    -------
    np.ndarray
        Array of shape (3, len(w)) containing [P, D, w_RR].
    """
    return result
```

### Step 2

qssf_coupled_step

Goal
----
Advance the classical envelope and both quantum Bogoliubov maps in one coupled QSSF step, using the prescribed frozen field and the distinct annihilation and creation Fourier bases.

```python
def qssf_coupled_step(
    A: "np.ndarray",
    U: "np.ndarray",
    V: "np.ndarray",
    P: "np.ndarray",
    dz: float,
    gamma: float = 1.0,
) -> "np.ndarray":
    """Return the jointly propagated classical and quantum state.

    Parameters
    ----------
    A, P : "np.ndarray"
        Time envelope and spectral half-step, each of shape (N,).
    U, V : "np.ndarray"
        Frequency-basis Bogoliubov matrices, each of shape (N,N).
    dz : float
        Finite nonnegative propagation distance.
    gamma : float, default 1.0
        Finite real Kerr coefficient.

    Returns
    -------
    result : "np.ndarray"
        Complex (2*N+1,N) array: A_new row, U_new block, V_new block.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data or invalid scalar parameters.
    """
    return result
```

### Step 3

qssf_symplecticity_check

Goal
----
Compute both normalized Frobenius residuals of the bosonic canonical relations. Return diagnostic errors for canonical or noncanonical finite matrix pairs without imposing a physicality threshold in this diagnostic routine.

```python
def qssf_symplecticity_check(U: "np.ndarray", V: "np.ndarray") -> "np.ndarray":
    """
    Calculate the normalized Frobenius errors for symplectic commutator
    relations.

    Parameters
    ----------
    U : "np.ndarray"
        Normal Bogoliubov matrix, shape (Nt, Nt).
    V : "np.ndarray"
        Anomalous Bogoliubov matrix, shape (Nt, Nt).

    Returns
    -------
    np.ndarray
        One-dimensional float array of shape (2,) containing [err1, err2].
    """
    return result
```

### Step 4

qssf_second_moments

Goal
----
Select the requested output modes while retaining every input vacuum channel. Construct their normal and anomalous moments and the real grouped-quadrature covariance, validating the canonical map and the window indices.

```python
def qssf_second_moments(
    U: "np.ndarray", V: "np.ndarray", window: "np.ndarray"
) -> "np.ndarray":
    """
    Calculate the second-order moments and assemble the 2n x 2n covariance
    matrix V_q.

    Parameters
    ----------
    U : "np.ndarray"
        Normal Bogoliubov matrix, shape (Nt, Nt).
    V : "np.ndarray"
        Anomalous Bogoliubov matrix, shape (Nt, Nt).
    window : "np.ndarray"
        Integer array of selected frequency mode indices, shape (n,).

    Returns
    -------
    np.ndarray
        Real symmetric covariance matrix V_q of shape (2n, 2n).
    """
    return result
```

### Step 5

qssf_williamson_decomposition

Goal
----
Construct a real canonical coordinate map that diagonalizes the reduced covariance. Retain every symplectic mode, including degenerate modes; accept equivalent choices of canonical basis.

```python
def qssf_williamson_decomposition(Vq: "np.ndarray") -> "np.ndarray":
    """Return a Williamson spectrum and a canonical diagonalizing map.

    Parameters
    ----------
    Vq : "np.ndarray"
        Physical real covariance of shape (2*n, 2*n), in grouped order.

    Returns
    -------
    result : "np.ndarray"
        Real array (2*n+1, 2*n). First row: duplicated descending nu.
        Remaining rows: W with W Vq W.T = D and W Omega W.T = Omega.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite or nonreal entries, nonsymmetry,
        lack of positive definiteness or violation of physicality.
    """
    return result
```

### Step 6

qssf_williamson_entropy

Goal
----
Validate the actual covariance's physicality and the supplied canonical coordinates at their respective tolerances, then evaluate entropy, purity and effective occupied mode number.

```python
def qssf_williamson_entropy(
    Vq: "np.ndarray", decomposition: "np.ndarray"
) -> "np.ndarray":
    """Validate a canonical decomposition and evaluate its diagnostics.

    Parameters
    ----------
    Vq : "np.ndarray"
        Real physical covariance (2*n, 2*n), grouped quadrature order.
    decomposition : "np.ndarray"
        Real array (2*n+1, 2*n): diag(D), then W from the preceding step.

    Returns
    -------
    result : "np.ndarray"
        Float vector (4+n,), [SvN, S2, purity, K_eff, nu...].

    Raises
    ------
    ValueError
        For invalid covariance, malformed or nonfinite decomposition,
        unsorted or nonphysical spectrum, or failed canonical identities.
    """
    return result
```

### Step 7

qssf_full_simulation

Goal
----
Compose all six preceding functions to propagate the coupled state, validate its commutators and Williamson coordinates, and return the selected spectral-band diagnostic.

```python
def qssf_full_simulation(
    n_steps: int = 40,
    dz: float = 0.01,
    Nt: int = 128,
    T: float = 40.0,
    A0: float = 1.5,
    beta3: float = 0.08,
    d2: float = 1.0,
    gamma: float = 1.0,
    delta_w: float = 3.0,
    quantity: str = "entropy",
) -> float:
    """
    Run the complete QSSF simulation pipeline and return the requested
    observable.

    Parameters
    ----------
    n_steps : int
        Number of propagation steps.
    dz : float
        Spatial step size.
    Nt : int
        Grid size.
    T : float
        Temporal window duration.
    A0 : float
        Sech-pulse amplitude.
    beta3 : float
        Third-order dispersion coefficient.
    d2 : float
        Second-order dispersion coefficient.
    gamma : float
        Kerr nonlinear coefficient.
    delta_w : float
        Prescribed analysis-band width.
    quantity : str
        Target observable name.

    Returns
    -------
    float
        Computed observable value.
    """
    return 0.0
```
