# Physics-Computational_Physics-40

## Background

Low-order plasma moment equations are inexpensive but lose collisionless phase mixing unless the unresolved heat flux is closed with kinetic information. A recent approach makes the three-moment closure depend on wave number by forcing its Padé response to share the least-damped Vlasov–Poisson roots, thereby targeting the modes that control long-time macroscopic evolution.

The task below applies that root-matched closure to user-supplied Fourier amplitudes and reduces the result to one electric-field-energy diagnostic.

## Problem

A collisionless Maxwellian electron plasma can retain Landau damping in a low-order fluid description when the unresolved heat flux is anchored to the dominant kinetic dispersion roots instead of to fixed asymptotic coefficients. In units where $k_p=\omega_{pe}=v_t=1$, compute the base-10 logarithm of the normalized total electric-field energy at $t_f=40$ for positive wave numbers $\tilde{k}=(0.18,0.27,0.40,0.58,0.82)$, density amplitudes $A=(0.011,0.017,0.020,0.014,0.009)$, and phases $\varphi=(0.2,-0.4,0.7,-0.1,0.5)$. Define the kinetic branch by the positive-real, least-damped root of $1+\zeta Z(\zeta)+\tilde{k}^{2}=0$, with $Z(\zeta)=i\sqrt{\pi}\,w(\zeta)$ and $w$ the Faddeeva function, and use the parity-symmetric root-matched three-pole closure with quadratic denominator coefficient $b_2=-2$; do not substitute a published fit for its wave-number-dependent coefficients. Initialize the positive-$k$ Fourier modes from an isothermal Maxwellian perturbation, $\hat n_k(0)=A_k e^{i\varphi_k}/2$, $\hat u_k(0)=0$, and $\hat p_k(0)=\hat n_k(0)$, and evolve the linearized three-moment–Poisson system with the explicit midpoint rule using $\Delta t=0.005$ and exactly $t_f/\Delta t$ steps. Use $\mathcal E(t)\propto\sum_{k>0}|\hat n_k(t)/\tilde{k}|^2$ without adding a second factor for the omitted negative-$k$ partners, and return $\log_{10}[\mathcal E(t_f)/\mathcal E(0)]$ rounded to 10 digits after the decimal point. In `<reasoning>`, justify the kinetic-root branch and symmetry, recover the closure-to-moment equations rather than quoting fitted curves, give enough numerical checkpoints to make the tagged result auditable, explain what the measured attenuation implies physically, and state which part of the supplied spectrum controls the late-time result.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Build Kinetic Root Seeds

Goal
----
Construct deterministic starting estimates for each normalized wave number.

```python
import numpy as np
def build_kinetic_root_seeds(kappa: np.ndarray) -> np.ndarray:
    """Return deterministic Bohm--Gross seed rows.

    For every ``k = kappa[j]``, use
    ``omega_r = sqrt(1 + 3*k**2)``,
    ``gamma = -sqrt(pi/8)*k**(-3)*exp(-1/(2*k**2) - 3/2)``, and
    ``zeta = (omega_r + 1j*gamma)/(sqrt(2)*k)``. Return
    ``[k, zeta.real, zeta.imag]`` in each row.

    Parameters
    ----------
    kappa : np.ndarray
        Strictly increasing positive normalized wave numbers.

    Returns
    -------
    np.ndarray
        Float array with shape ``(len(kappa), 3)``.

    Raises
    ------
    ValueError
        If ``kappa`` is not a finite, positive, strictly increasing nonempty vector.
    """
    return np.empty((np.asarray(kappa).size, 3), dtype=float)
```

### Step 2

Solve Least-Damped Kinetic Roots

Goal
----
Refine the positive-real kinetic root at each supplied wave number.

```python
from scipy.special import wofz
import numpy as np
def solve_least_damped_roots(
    seed_table: np.ndarray, residual_tol: float = 1e-12
) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), residual]``.

    Parameters
    ----------
    seed_table : np.ndarray
        Seed rows with columns ``kappa, Re(zeta), Im(zeta)``.
    residual_tol : float
        Positive convergence threshold for the dispersion residual.

    Returns
    -------
    np.ndarray
        Refined real-valued root table.

    Raises
    ------
    ValueError
        If the inputs are invalid or the requested root does not converge.
    """
    return np.empty((np.asarray(seed_table).shape[0], 4), dtype=float)
```

### Step 3

Match Root-Constrained Padé Coefficients

Goal
----
Recover the two free rational-response parameters from each kinetic root.

```python
import numpy as np
def match_pade_coefficients(root_table: np.ndarray) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), Im(a1), Im(b1)]``.

    Parameters
    ----------
    root_table : np.ndarray
        Root rows with columns ``kappa, Re(zeta), Im(zeta), residual``.

    Returns
    -------
    np.ndarray
        Real table containing the root and matched Padé coefficients.

    Raises
    ------
    ValueError
        If the table is invalid, a root is on the wrong branch, or matching is singular.
    """
    return np.empty((np.asarray(root_table).shape[0], 5), dtype=float)
```

### Step 4

Derive Heat-Flux Closure Coefficients

Goal
----
Convert the matched Padé parameters into real closure coefficients.

```python
import numpy as np
def derive_closure_coefficients(pade_table: np.ndarray) -> np.ndarray:
    """Return rows ``[kappa, Re(zeta), Im(zeta), Q1, Q3]``.

    Parameters
    ----------
    pade_table : np.ndarray
        Rows containing ``kappa, Re(zeta), Im(zeta), Im(a1), Im(b1)``.

    Returns
    -------
    np.ndarray
        Real wave-number-dependent closure table.

    Raises
    ------
    ValueError
        If the table is invalid, ``a1`` vanishes, or the real coefficients are nonpositive.
    """
    return np.empty((np.asarray(pade_table).shape[0], 5), dtype=float)
```

### Step 5

Assemble Three-Moment Generators

Goal
----
Build the Fourier-space linear operator for every closure row.

```python
import numpy as np
def assemble_moment_generators(closure_table: np.ndarray) -> np.ndarray:
    """Return one complex ``3 x 3`` generator per closure row.

    Parameters
    ----------
    closure_table : np.ndarray
        Rows containing ``kappa, Re(zeta), Im(zeta), Q1, Q3``.

    Returns
    -------
    np.ndarray
        Complex array with shape ``(m, 3, 3)`` for state ``[n,u,p]``.

    Raises
    ------
    ValueError
        If the closure table is invalid or contains nonpositive ``kappa``, ``Q1``, or ``Q3``.
    """
    return np.empty((np.asarray(closure_table).shape[0], 3, 3), dtype=complex)
```

### Step 6

Initialize Isothermal Fourier Modes

Goal
----
Form the complex positive-wave-number density, velocity, and pressure states.

```python
import numpy as np
def initialize_isothermal_modes(
    generators: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    """Return complex initial rows ``[n,u,p]`` aligned to the generators.

    Parameters
    ----------
    generators : np.ndarray
        Complex array with shape ``(m, 3, 3)``.
    amplitudes, phases : np.ndarray
        Real vectors of length ``m`` for cosine amplitudes and phases.

    Returns
    -------
    np.ndarray
        Complex initial-mode array with shape ``(m, 3)``.

    Raises
    ------
    ValueError
        If the arrays are invalid, misaligned, nonfinite, or contain a negative amplitude.
    """
    return np.empty((np.asarray(generators).shape[0], 3), dtype=complex)
```

### Step 7

Propagate Modes with Explicit Midpoint

Goal
----
Advance every independent Fourier state to the requested final time.

```python
from numbers import Real
import numpy as np
def propagate_midpoint_modes(
    generators: np.ndarray,
    initial_modes: np.ndarray,
    delta_t: float,
    final_time: float,
) -> np.ndarray:
    """Return the final complex mode rows after exact-count midpoint steps.

    Parameters
    ----------
    generators : np.ndarray
        Complex array with shape ``(m, 3, 3)``.
    initial_modes : np.ndarray
        Complex array with shape ``(m, 3)``.
    delta_t, final_time : float
        Positive step and nonnegative exactly reachable final time.

    Returns
    -------
    np.ndarray
        Complex final-mode array with shape ``(m, 3)``.

    Raises
    ------
    ValueError
        If mode data or time controls are invalid, including a nonintegral step count.
    """
    return np.empty_like(np.asarray(initial_modes, dtype=complex))
```

### Step 8

Compute Logarithmic Field Energy

Goal
----
Reduce the initial and final density spectra to one normalized diagnostic.

```python
import numpy as np
def compute_log_field_energy(
    kappa: np.ndarray, initial_modes: np.ndarray, final_modes: np.ndarray
) -> float:
    """Return ``log10(E_final / E_initial)`` as a native float.

    Parameters
    ----------
    kappa : np.ndarray
        Positive wave-number vector of length ``m``.
    initial_modes, final_modes : np.ndarray
        Complex state arrays with shape ``(m, 3)``.

    Returns
    -------
    float
        Base-10 logarithmic normalized field energy.

    Raises
    ------
    ValueError
        If the arrays are invalid or either field energy is nonpositive or nonfinite.
    """
    return 0.0
```

### Step 9

Run Root-Matched Closure Pipeline

Goal
----
Compose every public step into the end-to-end energy calculation.

```python
import numpy as np
def run_root_matched_energy(
    kappa: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    delta_t: float,
    final_time: float,
    residual_tol: float = 1e-12,
) -> float:
    """Run the complete root-matched closure pipeline.

    Parameters
    ----------
    kappa, amplitudes, phases : np.ndarray
        Aligned wave-number, amplitude, and phase vectors.
    delta_t, final_time, residual_tol : float
        Propagation step, final time, and positive root-residual tolerance.

    Returns
    -------
    float
        Base-10 logarithmic normalized field energy.

    Raises
    ------
    ValueError
        If an input is invalid or the pipeline does not produce a finite result.
    """
    return 0.0
```
