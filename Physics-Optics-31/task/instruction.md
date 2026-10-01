# Physics-Optics-31

## Background

Free-electron wave packets interacting with structured light may recoil and deform in regimes where eikonal or no-recoil descriptions are insufficient. Direct grid propagation resolves this behavior but makes the Schrödinger stage costly in multiscale electron–field simulations.

A recent mesh-free approach replaces the spatial mesh by a weighted, overcomplete ensemble of thawed Gaussian packets. Their centers and widths evolve locally and are recombined coherently, permitting a direct comparison with a split-step Fourier reference in momentum space.

## Problem

A recent mesh-free description of electron-light interactions represents one wave packet by an importance-weighted ensemble of independently propagated Gaussian states and benchmarks it against a grid-based Schrodinger solver; using that source method, determine the relative unit-normalized momentum-amplitude shape discrepancy for the following reduced nondimensional quasi-static nanorod measurement.

Take hbar=m=e=1, initial center q0=(-2.4,0.65), momentum p0=(4,0), target width Gamma0=diag(0.8,4), representing width Gamma=4 Gamma0, 32 Gaussian phase-space samples, dipole parameters R=0.4, E0=0.12, epsilon_m=-24.061+1.5068i and omega=1.3, and propagation from t=0 for 40 steps of 0.02. For this reduced Python benchmark, generate the design with `scipy.stats.qmc.Sobol(4, scramble=True, seed=23).random_base2(5)` before the normal inverse-CDF and covariance map, which fixes the finite scrambled net without introducing a new physical assumption.

Use a 48 by 40 uniform reference grid on x in [-5,3] and y in [-2,2], normalize both endpoint fields with their quadrature-weighted norms, compare their Fourier amplitudes rather than complex phases, and repeat with 16 and 64 Gaussian samples and with E0=0 while leaving every other default fixed; the tagged answer remains the 32-sample field-on shape discrepancy. Include both endpoints: use x_i = -5 + 8i/(nx-1) and y_j = -2 + 4j/(ny-1), for i = 0,...,nx-1 and j = 0,...,ny-1, with nx=48 and ny=40 for the requested run. For every reference-grid time step, use a potential half step evaluated at the temporal midpoint, the exact Fourier kinetic full step, and the same potential half step.

Treat these as my simulation measurements rather than a published table, explain what the discrepancy and controls imply about the reduced mesh-free representation, and report as compact diagnostics the field-minus-zero-field changes in ensemble-mean position and momentum, the maximum over packets of the two field-on endpoint Hagedorn compatibility Frobenius norms, and the raw-array complex momentum-field discrepancy defined next. For that complex diagnostic, after separate momentum-grid normalization compare the analytic mesh-free physical-coordinate transform, including its exp(-i k_x q_x - i k_y q_y) translation phase, directly with dx dy times fftshift(fft2(ifftshift(psi_grid))) on the shifted FFT nodes: the latter treats the samples at indices (24,20) as its coordinate origin, and neither a coordinate-origin phase correction nor a fitted global phase is applied.

In `<reasoning>`, identify only the source-specific phase-space weighting, coefficient, width-evolution, local-field and reference-propagation conventions needed for reproducibility, and round the requested discrepancy to ten digits after the decimal point.

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

01_generate_weighted_sobol_nodes

Goal
----
Generate deterministic importance-weighted Sobol phase-space nodes.

```python
from numbers import Integral, Real
from scipy.stats import norm, qmc
import numpy as np

def generate_weighted_sobol_nodes(
    q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    n_nodes: int, hbar: float = 1.0, seed: int = 23,
) -> np.ndarray:
    """Return normally distributed Sobol nodes ordered as ``[q, p]``.

    Parameters are length-d centers, d-by-d positive-definite width matrices,
    a positive power-of-two node count (including one), positive ``hbar``,
    and integer ``seed``.
    The deterministic Python realization is SciPy's scrambled ``qmc.Sobol`` in
    dimension ``2*d`` followed by ``random_base2(log2(n_nodes))``. Map its
    inverse-normal values ``xi`` as
    ``[q0,p0] + sqrt(hbar) * xi @ chol(Sigma).T``, where
    ``Sigma = block_diag(inv(gamma0)+inv(gamma), gamma0+gamma)`` and the
    Cholesky factor is lower triangular.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_nodes, 2*d)``.

    Raises
    ------
    ValueError
        If centers or widths are misaligned or non-finite, either width is not
        symmetric positive-definite, ``n_nodes`` is not a power-of-two integer,
        ``hbar`` is not positive finite, or ``seed`` is not an integer.
    """
    return nodes
```

### Step 2

02_compute_gaussian_expansion_coefficients

Goal
----
Evaluate the overlap-to-weight ratio at every phase-space node.

```python
from numbers import Real
import numpy as np

def compute_gaussian_expansion_coefficients(
    nodes: np.ndarray, q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return coefficients for normalized Gaussian basis states.

    ``nodes`` has shape ``(N,2*d)`` and widths are real symmetric positive
    definite. For ``z=(q,p)``, use
    ``W(z)=exp(-(z-[q0,p0]) @ inv(Sigma) @ (z-[q0,p0])/(2*hbar))`` with
    ``Sigma=block_diag(inv(gamma0)+inv(gamma),gamma0+gamma)``. The normalized
    Gaussian is proportional to ``det(gamma)^0.25`` times
    ``exp((-(x-q)@gamma@(x-q)/2 + 1j*p@(x-q))/hbar)``. Return its conjugate
    overlap with the normalized target Gaussian divided by
    ``N*(2*pi*hbar)^d*W(z)``.

    Returns
    -------
    np.ndarray
        Complex coefficient vector of length N.

    Raises
    ------
    ValueError
        If the nodes, centers, or widths are misaligned or non-finite, either
        width is not symmetric positive-definite, or ``hbar`` is not positive finite.
    """
    return coefficients
```

### Step 3

03_evaluate_quasistatic_dipole

Goal
----
Return potential, gradient, and Hessian for one or many two-dimensional points.

```python
from numbers import Real
import numpy as np
def evaluate_quasistatic_dipole(
    t: float, positions: np.ndarray,
    field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the oscillating nanorod potential and its first two derivatives.

    Positions end in dimension two. With
    ``c=abs((dielectric-1)/(dielectric+1))``, define the scalar field as
    ``phi=field_amplitude*c*x`` for ``x*x+y*y < radius**2`` and
    ``phi=field_amplitude*c*radius**2*x/(x*x+y*y)`` otherwise, including the
    boundary. Return the potential ``-charge*cos(omega*t)*phi`` and its spatial
    gradient and Hessian. ``dielectric`` enters only through ``c``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Potential ``(...)``, gradient ``(...,2)``, and Hessian ``(...,2,2)``.

    Raises
    ------
    ValueError
        If positions do not end in dimension two, any input is non-finite,
        ``radius`` is not positive, or the dielectric contrast is singular.
    """
    return potential, gradient, hessian
```

### Step 4

04_initialize_hagedorn_state

Goal
----
Convert phase-space nodes and a basis width into the propagated parameter state.

```python
from numbers import Real
import numpy as np
def initialize_hagedorn_state(
    nodes: np.ndarray, gamma: np.ndarray, hbar: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize ``q, p, Q, P, S`` for all phase-space nodes.

    ``nodes`` has shape ``(N,2*d)`` and ``gamma`` is real symmetric positive
    definite. Set ``q`` and ``p`` to copies of the two node blocks,
    ``Q_j = sqrt(hbar) * gamma**(-1/2)``,
    ``P_j = 1j * sqrt(hbar) * gamma**(1/2)``, and ``S_j = 0`` for every node,
    using the symmetric eigendecomposition roots.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Shapes ``(N,d)``, ``(N,d)``, ``(N,d,d)``, ``(N,d,d)``, and ``(N,)``.

    Raises
    ------
    ValueError
        If nodes are not a finite ``(N,2*d)`` array, ``gamma`` is not a finite
        symmetric positive-definite ``(d,d)`` matrix, or ``hbar`` is not
        positive finite.
    """
    return q, p, Q, P, S
```

### Step 5

05_propagate_hagedorn_state

Goal
----
Advance every independent Gaussian with the source drift-kick-drift map.

```python
from numbers import Integral, Real
import numpy as np
def propagate_hagedorn_state(
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray, dt: float, n_steps: int,
    mass: float, field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance a batch of two-dimensional Hagedorn states.

    Each step uses half drifts of ``q`` and ``Q``, one midpoint field/Hessian
    evaluation, centered kicks, and a midpoint Lagrangian action. Inputs are
    copied and the final arrays are returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Final ``q, p, Q, P, S`` arrays with their input shapes.

    Raises
    ------
    ValueError
        If the state arrays are non-finite or misaligned, if ``n_steps`` is not
        a nonnegative integer, or if ``dt`` or ``mass`` is not positive finite.
    """
    return q, p, Q, P, S
```

### Step 6

06_reconstruct_tgwp_momentum

Goal
----
Sum closed Fourier transforms of the propagated Gaussian components.

```python
from numbers import Real
import numpy as np

def reconstruct_tgwp_momentum(
    x: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return the normalized TGWP momentum field on shifted FFT nodes.

    Finite, strictly monotone, uniform real-space axes, each with at least
    two points and nonzero spacing, define
    ``k=fftshift(2*pi*fftfreq(n,d=spacing))`` in each direction. For each packet
    let ``C=P@inv(Q)`` and ``delta=p-hbar*k``; its Fourier factor is
    ``pi**(-1/2)*(2*pi*hbar)/sqrt(det(Q)*det(-1j*C))`` times
    ``exp(-0.5j*delta@inv(C)@delta/hbar - 1j*q@k + 1j*S/hbar)``. Sum these
    factors with ``coefficients`` and normalize with
    ``sqrt(sum(abs(wave)**2)*abs(dkx*dky))``. Use principal complex square roots,
    which remain on one branch for the specified short-time benchmark.

    Returns
    -------
    np.ndarray
        Complex momentum field of shape ``(len(x),len(y))``.

    Raises
    ------
    ValueError
        If axes are invalid, the Gaussian arrays are misaligned or non-finite,
        ``hbar`` is not positive finite, a required width is singular or its
        numerical inverse/determinant is invalid, or the reconstructed field
        has zero or non-finite norm.
    """
    return momentum
```

### Step 7

07_initialize_grid_wavefunction

Goal
----
Evaluate the same two-dimensional Gaussian used by the mesh-free expansion.

```python
from numbers import Real
import numpy as np

def initialize_grid_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    gamma0: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return the discretely normalized target Gaussian on uniform axes.

    ``q0`` and ``p0`` have length two and ``gamma0`` is a real symmetric
    positive-definite 2-by-2 matrix. The finite one-dimensional axes must
    each contain at least two strictly monotone, uniformly spaced points with
    nonzero spacing. The sampled field must have finite nonzero cell norm.

    Returns
    -------
    np.ndarray
        Complex field of shape ``(len(x),len(y))``.

    Raises
    ------
    ValueError
        If axes are invalid, centers or widths have invalid shapes, inputs
        are non-finite, ``gamma0`` is not symmetric positive-definite, ``hbar``
        is not positive finite, or the sampled field has zero or non-finite norm.
    """
    return wavefunction
```

### Step 8

08_propagate_split_step_reference

Goal
----
Alternate potential half-steps and an exact spectral kinetic step.

```python
from numbers import Integral, Real
import numpy as np

def propagate_split_step_reference(
    psi: np.ndarray, x: np.ndarray, y: np.ndarray,
    dt: float, n_steps: int, mass: float,
    field_amplitude: float, radius: float, dielectric: complex,
    omega: float, charge: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Propagate a two-dimensional field with midpoint-potential Strang steps.

    The potential is sampled at each temporal midpoint. Apply, in order, a
    potential half-step, a Fourier kinetic full-step, and the same potential
    half-step. The finite one-dimensional axes must each contain at least
    two strictly monotone, uniformly spaced points with nonzero spacing.
    The input field shape must equal ``(len(x),len(y))`` and its cell norm
    must be finite and nonzero. After propagation, normalize the output to
    unit cell-area-weighted L2 norm, including when ``n_steps=0``. The input
    field is copied and remains unchanged.

    Returns
    -------
    np.ndarray
        Final complex field with the input shape and unit cell-area-weighted norm.

    Raises
    ------
    ValueError
        If the field and axes are misaligned, non-finite or invalid as above,
        ``n_steps`` is not a nonnegative integer, ``dt``, ``mass``, or ``hbar``
        is not positive finite, or the input/output cell norm is zero or
        non-finite. During propagation, invalid potential parameters also raise
        ValueError under the contract of ``evaluate_quasistatic_dipole``.
    """
    return propagated_wavefunction
```

### Step 9

09_compute_momentum_amplitude_error

Goal
----
Reduce two endpoint fields to one unit-normalized Fourier-amplitude L2 error.

```python
from numbers import Real
import numpy as np

def compute_momentum_amplitude_error(
    meshfree_momentum: np.ndarray,
    reference_wave: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """Return the L2 distance between unit-normalized momentum amplitudes.

    The first field is already in shifted momentum order; the second is a
    real-space grid field transformed with the aligned continuous FFT convention.

    Returns
    -------
    float
        Nonnegative unit-normalized momentum-amplitude shape discrepancy.

    Raises
    ------
    ValueError
        If the fields are misaligned or non-finite, either spacing is not
        positive finite, or either momentum field has zero or non-finite norm.
    """
    return 0.0
```

### Step 10

10_run_meshfree_grid_benchmark

Goal
----
Compose every public scientific step and return the endpoint amplitude error.

```python
from numbers import Integral, Real
import numpy as np

def run_meshfree_grid_benchmark(
    n_nodes: int = 32,
    nx: int = 48,
    ny: int = 40,
    dt: float = .02,
    n_steps: int = 40,
    field_amplitude: float = .12,
    gamma_scale: float = 4.0,
    seed: int = 23,
) -> float:
    """Run the prescribed two-dimensional quasi-static benchmark.

    Defaults use hbar=mass=charge=1, q0=(-2.4,.65), p0=(4,0), gamma0=diag(.8,4),
    R=.4, dielectric=-24.061+1.5068j, omega=1.3, x=[-5,3], and y=[-2,2].
    Include both endpoints: x_i=-5+8*i/(nx-1), i=0,...,nx-1, and
    y_j=-2+4*j/(ny-1), j=0,...,ny-1. Grid counts ``nx`` and ``ny`` are
    integers at least 8; ``n_steps`` is a nonnegative integer; ``n_nodes``
    is a positive power of two, including one. ``dt`` and ``gamma_scale``
    are positive finite reals, and ``field_amplitude`` is a finite real.
    ``seed`` is the integer Sobol seed.

    Returns
    -------
    float
        Unit-normalized momentum-amplitude shape discrepancy.

    Raises
    ------
    ValueError
        If grid or step counts are invalid, ``n_nodes`` violates the power-of-two
        contract, or ``dt``, ``gamma_scale``, or ``field_amplitude`` violates
        its stated finite-scalar contract.
    """
    return 0.0
```
