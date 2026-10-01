# Dimensionless response of a damaged nonlocal material-point cell

## Background

# Scientific background

Nonlocal continuum models replace spatial derivatives by weighted interactions between material points inside a finite horizon. They are attractive for fracture because displacement discontinuities can form without remeshing, but correspondence formulations must still reconstruct deformation gradients accurately from incomplete or damaged neighborhoods.

Deleting failed bonds changes the moment matrix used by that reconstruction and can make it poorly conditioned near an evolving crack. A continuous bond phase variable separates two effects: energetic degradation reduces the stress transmitted by damaged bonds, while a thresholded kinematic degradation reduces their influence on the deformation reconstruction only after substantial damage. The separation is important because applying the current phase to both effects in the same update changes the discrete time ordering and can feed a newly detected crack immediately back into the kinematics.

Bond-associated correction enforces the observed deformation of each bond while retaining a first-order nonlocal point gradient. A normalized force state then combines the corrected constitutive stress with a transverse stabilization term, and its antisymmetric assembly supplies internal force density while preserving global linear momentum under nodal quadrature. The normalization linking fracture energy to a bond driving-force threshold depends on the chosen spherical kernel, so it must be evaluated consistently rather than replaced by a kernel-independent constant.

## Problem

Evaluate one deterministic quasistatic update of a three-dimensional bond-associated nonlocal solid in double-precision arithmetic, using zero-based point indices, no randomness, and the inclusive neighborhood convention $0<r_{kn}\leq\delta$. The reference points $X_k$ in mm and nodal volumes $V_k$ in mm$^3$ are ordered as

$$
X=\begin{bmatrix}
0&0&0\\1&0&0\\0&1&0\\0&0&1\\1&1&0\\1&0&1\\0&1&1\\1&1&1
\end{bmatrix},\qquad
V=\begin{bmatrix}1.00&0.95&1.05&1.10&0.90&1.08&0.98&1.02\end{bmatrix}.
$$

Set $x_k=X_k+U_k$, where $U_{k,x}=0.004X_{k,x}+0.002X_{k,y}X_{k,z}$, $U_{k,y}=-0.001X_{k,y}+0.0015X_{k,x}X_{k,z}$, and $U_{k,z}=0.002X_{k,z}-0.001X_{k,x}X_{k,y}$, then add $(0.0012,-0.0007,0.0009)$ mm to $U_7$. The symmetric previous bond history $\mathcal{Y}^n$ has zero diagonal and, for $i<j$, initially equals $0.015+0.01((i+j)\bmod 3)$ MPa, after which the four entries $(0,7)$, $(1,6)$, $(2,5)$, and $(3,4)$ are replaced by $0.50$, $0.30$, $0.12$, and $0.08$ MPa respectively and mirrored. Use horizon $\delta=2.0$ mm, ascending kernel coefficients $(1,-2,1)$ so that $\omega(r)=(1-r/\delta)^2$, $E_Y=32000$ MPa, $\nu=0.25$, $G_c=0.1$ N/mm, and kinematic threshold $s_c=0.8$. Apply the spherical-kernel Griffith normalization, delayed kinematic degradation, first-order phase-weighted moment reconstruction, endpoint-averaged bond correction, Saint-Venant--Kirchhoff stress, maximum-principal-Cauchy-stress driving force, irreversible phase update, energetic stress degradation, normalized bond-force stabilization, and antisymmetric internal-force assembly prescribed for this formulation. Build all kinematic moments and shape gradients from the phase implied by $\mathcal{Y}^n$, then use the newly updated phase only for energetic stress degradation; do not rebuild the kinematics during this update. Report the single dimensionless response

$$
R=\frac{\delta}{E_Y}\sqrt{\frac{\sum_kV_k\lVert B_k^{\mathrm{int}}\rVert_2^2}{\sum_kV_k}}.
$$

In the short reasoning, explain the kernel normalization, phase-weighted reconstruction, corrected bond stress, and force assembly, and report $Q_d$, $Q_s$, and $Q_\times$ below, where $D_k$ and $S_k$ are the contributions to $B_k^{\mathrm{int}}$ from the direct bond-stress term and the transverse stabilization term respectively, each assembled with the same antisymmetric directed-bond rule:

$$
B_k^{\mathrm{int}}=D_k+S_k,\qquad
Q_d=\frac{\sum_kV_k\lVert D_k\rVert_2^2}{\sum_kV_k},\qquad
Q_s=\frac{\sum_kV_k\lVert S_k\rVert_2^2}{\sum_kV_k},\qquad
Q_\times=\frac{2\sum_kV_kD_k\cdot S_k}{\sum_kV_k}.
$$

These three terms determine $R=(\delta/E_Y)\sqrt{Q_d+Q_s+Q_\times}$; report only $R$ in the final-answer tag.

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

01_compute_kernel_normalization

Goal
----
Compute the spherical-kernel normalization used by the fracture threshold.

```python
def compute_kernel_normalization(kernel_coefficients: np.ndarray) -> float:
    r"""Return the spherical-kernel normalization $c_0$.

    Raises ``ValueError`` unless ``kernel_coefficients`` is a nonempty,
    one-dimensional, finite array; the kernel is nonnegative at 1001 equally
    spaced points on $[0,1]$; and both defining integrals and the resulting
    normalization are strictly positive and finite.

    Parameters
    ----------
    kernel_coefficients : np.ndarray
        Ascending coefficients $(a_0,a_1,\ldots,a_p)$ of $\omega_s$.

    Returns
    -------
    float
        Positive dimensionless normalization $c_0$.
    """
    return NotImplemented
```

### Step 2

02_update_bond_phase_history

Goal
----
Apply the irreversible bond-history and phase update.

```python
def update_bond_phase_history(
    trial_driving_force: np.ndarray,
    previous_history: np.ndarray,
    critical_driving_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Update irreversible history and the bond phase field.

    Raises ``ValueError`` unless the two input matrices have the same square
    shape of order at least two, are finite, nonnegative, symmetric, and have
    zero diagonals, and ``critical_driving_force`` is finite and strictly
    positive.

    Parameters
    ----------
    trial_driving_force : np.ndarray
        Symmetric $(N,N)$ trial energy-density matrix in MPa.
    previous_history : np.ndarray
        Symmetric $(N,N)$ previous maximum energy-density matrix in MPa.
    critical_driving_force : float
        Positive threshold $Y_c$ in MPa.

    Returns
    -------
    history_new : np.ndarray
        Symmetric $(N,N)$ irreversible history in MPa.
    phase_new : np.ndarray
        Symmetric $(N,N)$ dimensionless phase field in $[0,1]$.
    """
    return NotImplemented
```

### Step 3

03_build_kinematic_moments

Goal
----
Assemble the phase-weighted first-order moment matrix at every material point.

```python
def build_kinematic_moments(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    r"""Build all phase-weighted moment matrices $M_k$.

    Raises ``ValueError`` unless positions have shape $(N,3)$ with $N\geq4$,
    volumes have shape $(N,)$ and are positive, ``previous_phase`` is a finite
    symmetric $(N,N)$ array in $[0,1]$ with zero diagonal, ``horizon`` is
    positive and finite, the coefficient array is finite and one-dimensional,
    $0\leq s_c<1$, and every assembled moment is positive definite.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_phase : np.ndarray
        Symmetric old-time phase field of shape $(N,N)$.
    horizon : float
        Neighborhood radius $\delta$ in mm, inclusive at its boundary.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    kinematic_threshold : float
        Threshold $s_c$ separating intact and degraded kinematics.

    Returns
    -------
    np.ndarray
        Positive-definite moment matrices of shape $(N,3,3)$ in mm$^5$.
    """
    return NotImplemented
```

### Step 4

04_compute_bond_shape_gradients

Goal
----
Compute the volume-weighted first-order shape gradients of all active bonds.

```python
def compute_bond_shape_gradients(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    moments: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    r"""Compute $\nabla\phi_{kn}$ for every directed bond.

    Raises ``ValueError`` unless the point, volume, phase, horizon, kernel, and
    threshold inputs satisfy the moment-builder contract; ``moments`` has shape
    $(N,3,3)$ with finite symmetric positive-definite matrices; and every
    active kernel value is nonnegative.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_phase : np.ndarray
        Symmetric old-time phase field of shape $(N,N)$.
    moments : np.ndarray
        Moment matrices of shape $(N,3,3)$ in mm$^5$.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    kinematic_threshold : float
        Kinematic degradation threshold $s_c$ in $[0,1)$.

    Returns
    -------
    np.ndarray
        Shape-gradient array of shape $(N,N,3)$ in mm$^{-1}$, zero off bonds.
    """
    return NotImplemented
```

### Step 5

05_compute_point_deformation_gradients

Goal
----
Recover the nonlocal deformation gradient at every material point.

```python
def compute_point_deformation_gradients(
    displacements: np.ndarray,
    shape_gradients: np.ndarray,
) -> np.ndarray:
    r"""Compute all nonlocal point deformation gradients $\bar F_k$.

    Raises ``ValueError`` unless ``displacements`` is a finite array of shape
    $(N,3)$ with $N\geq2$, ``shape_gradients`` is a finite array of shape
    $(N,N,3)$, and every diagonal shape-gradient vector is zero.

    Parameters
    ----------
    displacements : np.ndarray
        Point displacements of shape $(N,3)$ in mm.
    shape_gradients : np.ndarray
        Directed volume-weighted gradients of shape $(N,N,3)$ in mm$^{-1}$.

    Returns
    -------
    np.ndarray
        Dimensionless deformation gradients of shape $(N,3,3)$.
    """
    return NotImplemented
```

### Step 6

06_update_bond_constitutive_state

Goal
----
Apply the bond-associated correction, elastic law, tensile driving force, and damage update.

```python
def update_bond_constitutive_state(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    point_gradients: np.ndarray,
    previous_history: np.ndarray,
    kernel_normalization: float,
    horizon: float,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Update stress, history, and phase for every active bond.

    Raises ``ValueError`` unless both position arrays have the same finite
    shape $(N,3)$ with $N\geq2$; point gradients have finite shape $(N,3,3)$;
    previous history is finite, nonnegative, symmetric, and zero-diagonal with
    shape $(N,N)$; $c_0$, $\delta$, $E_Y$, and $G_c$ are positive and finite;
    $-1<\nu<0.5$; and every active corrected gradient has positive determinant.

    Parameters
    ----------
    reference_positions, current_positions : np.ndarray
        Reference and current coordinates of shape $(N,3)$ in mm.
    point_gradients : np.ndarray
        Dimensionless point gradients of shape $(N,3,3)$.
    previous_history : np.ndarray
        Previous symmetric bond history of shape $(N,N)$ in MPa.
    kernel_normalization : float
        Positive dimensionless constant $c_0$.
    horizon : float
        Inclusive bond horizon $\delta$ in mm.
    youngs_modulus : float
        Young's modulus $E_Y$ in MPa.
    poisson_ratio : float
        Poisson ratio $\nu$.
    fracture_energy : float
        Critical energy release rate $G_c$ in N/mm.

    Returns
    -------
    degraded_piola : np.ndarray
        Bond stresses of shape $(N,N,3,3)$ in MPa, zero off active bonds.
    history_new : np.ndarray
        Updated symmetric history of shape $(N,N)$ in MPa.
    phase_new : np.ndarray
        Updated symmetric phase field of shape $(N,N)$ in $[0,1]$.
    """
    return NotImplemented
```

### Step 7

07_compute_internal_force_density

Goal
----
Assemble the bond-associated force state and internal force density.

```python
def compute_internal_force_density(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    shape_gradients: np.ndarray,
    degraded_piola: np.ndarray,
) -> np.ndarray:
    r"""Assemble $B_k^{\mathrm{int}}$ from directed bond force states.

    Raises ``ValueError`` unless reference positions have finite shape $(N,3)$
    with $N\geq2$; volumes are finite and positive with shape $(N,)$; the
    horizon is positive and finite; kernel coefficients are a nonempty finite
    vector; shape gradients and degraded stresses have shapes $(N,N,3)$ and
    $(N,N,3,3)$ with finite entries and zero diagonals; active kernel values
    are nonnegative; and every point has positive kernel volume $\omega_k^0$.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    shape_gradients : np.ndarray
        Directed volume-weighted gradients of shape $(N,N,3)$ in mm$^{-1}$.
    degraded_piola : np.ndarray
        Directed bond stresses of shape $(N,N,3,3)$ in MPa.

    Returns
    -------
    np.ndarray
        Internal force densities of shape $(N,3)$ in N/mm$^3$.
    """
    return NotImplemented
```

### Step 8

08_compute_nonlocal_response_index

Goal
----
Run the complete nonlocal damage-and-force pipeline and return one response index.

```python
def compute_nonlocal_response_index(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    volumes: np.ndarray,
    previous_history: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
) -> float:
    r"""Return the deterministic dimensionless response index $R$.

    Raises ``ValueError`` when any earlier stage rejects its inputs, when
    ``current_positions`` and ``reference_positions`` do not share shape
    $(N,3)$, or when the final weighted RMS or response index is nonfinite.
    The old phase is computed from ``previous_history`` and $Y_c$ before any
    current-step damage update; no randomness is used.

    Parameters
    ----------
    reference_positions, current_positions : np.ndarray
        Reference and current coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_history : np.ndarray
        Symmetric previous bond history of shape $(N,N)$ in MPa.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    youngs_modulus : float
        Young's modulus $E_Y$ in MPa.
    poisson_ratio : float
        Poisson ratio $\nu$ in $(-1,0.5)$.
    fracture_energy : float
        Critical energy release rate $G_c$ in N/mm.
    kinematic_threshold : float
        Kinematic degradation threshold $s_c$ in $[0,1)$.

    Returns
    -------
    float
        Finite dimensionless response index $R$.
    """
    return NotImplemented
```
