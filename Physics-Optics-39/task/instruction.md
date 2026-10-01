# Physics-Optics-39

## Background

The optical convention is \(n_s=1-\delta_s+i\beta_s\) with forward propagation \(\exp(ik_sn_sz)\), intensity attenuation \(\mu_s=2k_s\beta_s\), and \(U=\sqrt I\exp(i\phi)\). The paraxial envelope obeys \(\partial_\eta U=(i/2)\nabla^2U\) in pixel coordinates. These fix the sign and normalization from which the retained transport orders follow.

The main paper supplies the fast local/non-local transport and inversion framework; its Supplementary Methods supply the leading wave correction, and its released implementation specifies spectral residual allocation and common-depth projection. Calibration and uncertainty propagation are an author-derived extension to this framework. The grid, phantoms, observations, covariance, fixed iteration settings, and derivative stencil define a synthetic benchmark.

The inverse calibration map is local to the declared root. Uncertainty is assigned to the three measured moments; the prescribed quadratic polynomial includes the curvature of this inverse map. Its Gaussian moments define a normal-equivalent robustness index, rather than an exact quantile of the full nonlinear retrieval distribution. Beam parameters are dimensionless; the discrepancy, its bias, its standard deviation, and its bound are in percentage points.

For code tests, `strength` multiplies the four comparison-object modulation amplitudes, `noise_scale` multiplies measurement standard deviations, and `steps`, `relax`, and `quantile` replace their canonical settings. Calibration-grid dimensions and the calibration object remain fixed. An explicit `observed` vector or `covariance` replaces the corresponding canonical input; the moment order is unchanged. Public interfaces specify array packing. Intermediate optical depth is real-valued, and all tested exponential evaluations are finite.

## Problem

A three-line phase-contrast beam has an uncertain spectrum and propagation scale calibrated from a known object. Using the attached paper's fast local and non-local polychromatic inversions with leading diffraction pressure, determine how this calibration uncertainty affects their signed contrast discrepancy on a second object. Let \(F(\theta)\) be the calibration-moment map below, \(\hat\theta\) its root at the measured vector \(z\), and \(B(\theta)=100(C_{\rm map}-C_{\rm local})/C_*\) the finite-iteration discrepancy for noiseless data synthesized at that same \(\theta\). For \(\epsilon\sim N(0,\Sigma)\), let \(q(\epsilon)\) be the total-degree-two Taylor polynomial of \(B(F^{-1}(z+\epsilon))\) about zero, using the stated finite-resolution derivative convention. Report the lower normal-equivalent bound \(R=\mathbb E[q]-1.645\sqrt{\operatorname{Var}(q)}\) in percentage points. In the reasoning report the six-value certificate \((\hat\theta_1,\hat\theta_2,\hat\theta_3,B(\hat\theta),\mathbb E[q]-B(\hat\theta),\sqrt{\operatorname{Var}(q)})\), justify the source-dependent operators and the uncertainty transformation, and interpret the bound physically. Each of these seven reported numbers has relative tolerance \(5\times10^{-4}\), or absolute tolerance \(10^{-8}\) for a zero reference; internal values remain unrounded.

Instance and conventions

- Coordinates: periodic \((n_y,n_x)=(35,41)\), unit pixel pitch, \(X=2\pi x/41\), \(Y=2\pi y/35\), zero-based indices, uniform pixel means. Reference depth is \(\tau=\mu_0t\); the line attenuation ratios and refractive ratios are \(\gamma=(1.6,1,0.6)\) and \(d=\delta/\beta=(16,22,30)\). Incident fractions are \(f=(e^{\theta_1},e^{\theta_2},1)/(1+e^{\theta_1}+e^{\theta_2})\), and \(\eta=L/(kp^2)=e^{\theta_3}(4.2,2.5,1.1)\).
- Calibration object: \(\tau_c=0.93+0.21\cos X+0.14\sin Y+0.11\cos(X+Y)+0.075\sin(2X-Y)\). For its non-local total detector prediction \(D_c(\theta)\), \(F(\theta)=(\langle D_c\rangle,\langle D_c\cos X\rangle,\langle D_c\sin(2X-Y)\rangle)\). The observed vector is \(z=(0.41517240210080686,-0.06127696643781137,-0.05212904606788967)\), in incident-total-flux units. Use the root in \([-0.8,0.3]\times[-0.2,0.9]\times[-0.2,0.1]\), resolved to maximum moment residual \(2\times10^{-12}\).
- Measurement covariance, in squared incident-total-flux units: \(\Sigma=3\times10^{-9}\begin{pmatrix}4&0.8&-0.6\\0.8&1&0.25\\-0.6&0.25&0.64\end{pmatrix}\). The derivative convention for both \(F\) and \(B\) is centered three-point axial and four-corner mixed differences in \(\theta\), at spacings \((5\times10^{-5},5\times10^{-5},5\times10^{-6})\); these define their local quadratic models.
- Comparison object: \(\tau_*=0.72+0.22\cos X+0.13\sin Y+0.09\cos(X+Y)+0.04\sin(2X-Y)\). Set \(w=\cos(X+Y)+0.3\sin(2X-Y)\), \(C[\tau]=\langle\tau w\rangle/\langle w^2\rangle\), and \(C_*=C[\tau_*]\). Both reconstructions use their own predictions of the same synthesized non-local detector image at the evaluated beam parameters, 12 simultaneous spectral updates, relaxation \(0.4\), and common-depth projection after each update. Their initial depth is \(-\log\max(H_{a_0}D,10^{-12})\), with \(a_0=\sum_s f_s\eta_sd_s/2\) and \(H_a\) the homogeneous-object inverse with coefficient \(a\).
- Discrete optics: centered periodic first differences composed for repeated derivatives, nearest-neighbor five-point amplitude Laplacian, bilinear periodic mass redistribution, and the paper's quadratic local closure. Spectral inverses use the continuous Laplacian on \(q_j=2\pi\operatorname{fftfreq}(n_j)\), with NumPy FFT normalization. Local residual back-transport is identity; non-local back-transport uses the current frozen ray geometry. Pressure is an additive signed source-intensity correction, mapped in the non-local model and added directly to the local closure.
- Spectral consistency: measured-minus-predicted total residual, positive-prediction allocation with denominator offset \(10^{-12}\), and incident-fraction-weighted averaging of reference line depths. Projection floors each line transmission at \(10^{-12}\); initialization floors the filtered total intensity. The paraxial phase convention is the one used in the supplied paper.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

spectral_rays

Goal
----
Determine the per-line object intensities and transverse ray displacements for the single-material object.

```python
import numpy as np

def spectral_rays(tau: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, eta: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Construct single-material spectral intensities and dimensionless ray shifts.

    Parameters
    ----------
    tau : float ndarray, shape (ny,nx)
        Finite reference optical depth (dimensionless), ny,nx >= 3.
    fractions : float ndarray, shape (S,)
        Strictly positive incident line fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    eta : float ndarray, shape (S,)
        Nonnegative L/(k_s*p**2), in squared-pixel units normalized by p**2.
    db : float ndarray, shape (S,)
        Nonnegative refractive ratio delta_s/beta_s.

    Returns
    -------
    out : float ndarray, shape (S,3,ny,nx)
        Axis 1 is [intensity, y displacement, x displacement]; intensity is
        relative to incident total flux, displacements are in pixels. Spatial
        derivatives are centered periodic first differences on unit pitch.

    Raises
    ------
    ValueError
        If tau is not two-dimensional or the spectral vector lengths differ.
    """
    return np.zeros((len(fractions), 3, *np.shape(tau)), dtype=float)
```

### Step 2

local_transport

Goal
----
Evaluate the local geometrical detector-intensity model through quadratic propagation order.

```python
import numpy as np

def local_transport(intensity: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the local second-order WKB0 intensity closure.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Positive object-plane intensities, normalized to total incident flux.
    shifts : float ndarray, shape (S,2,ny,nx)
        Per-line dimensionless displacements ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Local WKB0 detector predictions in incident-flux units, through
        quadratic order in shifts. Every derivative is the centered periodic
        first difference; second derivatives compose that same operator.
        This output is the geometrical contribution; the wave correction is a separate output.

    Raises
    ------
    ValueError
        If intensity and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(intensity), dtype=float)
```

### Step 3

diffraction_pressure

Goal
----
Evaluate the leading wave correction to the source intensity under the declared paraxial convention.

```python
import numpy as np

def diffraction_pressure(intensity: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Evaluate the source-plane leading WKB1 diffraction-pressure correction.

    Parameters
    ----------
    intensity : float ndarray, shape (S,ny,nx)
        Strictly positive spectral object intensities in incident-flux units.
    eta : float ndarray, shape (S,)
        Nonnegative dimensionless propagation parameters L/(k_s*p**2).

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Signed leading wave correction in incident-flux units on the source
        grid. Spatial derivatives use the discretization in the main prompt.

    Raises
    ------
    ValueError
        If intensity is not a positive three-dimensional array or eta length differs.
    """
    return np.zeros(np.shape(intensity), dtype=float)
```

### Step 4

ray_push

Goal
----
Evaluate the detector-plane spectral intensity under the declared finite-displacement transport model.

```python
import numpy as np

def ray_push(source: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Conservatively redistribute source mass along the finite eikonal map.

    Parameters
    ----------
    source : float ndarray, shape (S,ny,nx)
        Signed spectral source masses in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Finite mapped displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Detector-plane spectral masses under the periodic bilinear
        finite-displacement model. Units and spectral order match source.

    Raises
    ------
    ValueError
        If source and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(source), dtype=float)
```

### Step 5

ray_adjoint

Goal
----
Evaluate the source-plane residual operator for the declared non-local reconstruction.

```python
import numpy as np

def ray_adjoint(detector: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Evaluate the discrete adjoint of spectral transport.

    Parameters
    ----------
    detector : float ndarray, shape (S,ny,nx)
        Signed detector residuals in incident-flux units.
    shifts : float ndarray, shape (S,2,ny,nx)
        Forward source-to-detector displacements in pixels, ordered [y,x].

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Source-grid spectral residuals in incident-flux units. The adjoint
        uses the Euclidean pixel inner product and the declared forward map.

    Raises
    ------
    ValueError
        If detector and shifts do not have the documented compatible shapes.
    """
    return np.zeros(np.shape(detector), dtype=float)
```

### Step 6

spectral_residual

Goal
----
Determine the spectral residuals associated with the measured detector image.

```python
import numpy as np

def spectral_residual(measured: np.ndarray, predicted: np.ndarray, weight_floor: float) -> np.ndarray:
    """Evaluate the spectral detector residuals.

    Parameters
    ----------
    measured : float ndarray, shape (ny,nx)
        Positive total measured intensity in incident-flux units.
    predicted : float ndarray, shape (S,ny,nx)
        Signed per-line forward predictions in the same units.
    weight_floor : float
        Strictly positive intensity added once to the positive-flux denominator.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Per-line detector residuals in incident-flux units, with the
        spectral-consistency convention defined in the main prompt.

    Raises
    ------
    ValueError
        If the array shapes are incompatible or weight_floor is not positive.
    """
    return np.zeros(np.shape(predicted), dtype=float)
```

### Step 7

energy_inverse

Goal
----
Evaluate the per-line homogeneous-object inverse on a source residual.

```python
import numpy as np

def energy_inverse(residual: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply the energy-resolved FFT-diagonal homogeneous-object preconditioner.

    Parameters
    ----------
    residual : float ndarray, shape (S,ny,nx)
        Signed source residuals in incident-flux units.
    coefficients : float ndarray, shape (S,)
        Nonnegative a_s=eta_s*db_s/2, measured in pixel squared units.

    Returns
    -------
    out : float ndarray, shape (S,ny,nx)
        Real per-line inverse responses, in the input residual units, using
        the declared continuous-Laplacian and NumPy FFT conventions.

    Raises
    ------
    ValueError
        If residual is not three-dimensional, coefficients have the wrong length,
        or a coefficient is negative.
    """
    return np.zeros(np.shape(residual), dtype=float)
```

### Step 8

thickness_projection

Goal
----
Determine the common reference depth and its spectrally consistent object intensities.

```python
import numpy as np

def thickness_projection(components: np.ndarray, fractions: np.ndarray, attenuation: np.ndarray, intensity_floor: float) -> np.ndarray:
    """Project updated spectral components to one fraction-weighted optical depth.

    Parameters
    ----------
    components : float ndarray, shape (S,ny,nx)
        Finite updated spectral intensities; signed inputs are allowed.
    fractions : float ndarray, shape (S,)
        Positive incident spectral fractions summing to one.
    attenuation : float ndarray, shape (S,)
        Positive absorption coefficients relative to the reference coefficient.
    intensity_floor : float
        Positive lower bound, less than one, on each transmission I_s/f_s.

    Returns
    -------
    out : float ndarray, shape (S+1,ny,nx)
        Row 0 is the projected dimensionless reference optical depth. Rows
        1..S are the corresponding single-material spectral intensities,
        in incident-flux units. Use the declared incident-fraction weights
        and transmission floor. The reference depth is real-valued.

    Raises
    ------
    ValueError
        If component/vector shapes disagree, a fraction or attenuation is nonpositive,
        or intensity_floor is outside (0,1).
    """
    return np.zeros((len(fractions) + 1, *np.shape(components)[-2:]), dtype=float)
```

### Step 9

inverse_calibration_jet

Goal
----
Determine the response derivatives in calibration-measurement coordinates.

```python
import numpy as np

def inverse_calibration_jet(calibration_jacobian: np.ndarray, calibration_hessians: np.ndarray, response_gradient: np.ndarray, response_hessian: np.ndarray) -> np.ndarray:
    """Transform the response jet through a locally invertible calibration map.

    Parameters
    ----------
    calibration_jacobian : float ndarray, shape (n,n)
        Nonsingular dF_i/dtheta_j; rows are measured observables.
    calibration_hessians : float ndarray, shape (n,n,n)
        Entry [i,j,k] is d2F_i/(dtheta_j*dtheta_k).
    response_gradient : float ndarray, shape (n,)
        Gradient of the scalar response B with respect to theta.
    response_hessian : float ndarray, shape (n,n)
        Symmetric Hessian of B with respect to theta.

    Returns
    -------
    out : float ndarray, shape (n+1,n)
        Row zero is the gradient of B composed with the local inverse of F;
        the remaining n rows are its Hessian, both in measurement coordinates.
        Units are response per measurement, and response per measurement squared.
    """
    return np.zeros((len(response_gradient) + 1, len(response_gradient)), dtype=float)
```

### Step 10

gaussian_quadratic_bound

Goal
----
Determine the uncertainty certificate for the prescribed quadratic response.

```python
import numpy as np

def gaussian_quadratic_bound(value: float, gradient: np.ndarray, hessian: np.ndarray, covariance: np.ndarray, quantile: float) -> np.ndarray:
    """Evaluate Gaussian moments and a lower normal-equivalent bound of a quadratic.

    Parameters
    ----------
    value : float
        Response at zero perturbation, measured in percentage points.
    gradient : float ndarray, shape (n,)
        Linear coefficients with respect to calibration measurements.
    hessian : float ndarray, shape (n,n)
        Symmetric second derivatives with respect to those measurements.
    covariance : float ndarray, shape (n,n)
        Symmetric positive-semidefinite covariance of zero-mean Gaussian noise.
    quantile : float
        Nonnegative normal-equivalent standard-deviation multiplier.

    Returns
    -------
    out : float ndarray, shape (4,)
        Ordered [mean, standard deviation, lower bound, curvature bias].
        All four entries are in percentage points and refer to the
        quadratic Taylor polynomial defined in the problem.
    """
    return np.zeros((4,), dtype=float)
```

### Step 11

calibrated_contrast_bound

Goal
----
Evaluate the bound R for the parameterized benchmark. Compose and use every preceding public function; nested evaluations are permitted.

```python
import numpy as np

def calibrated_contrast_bound(observed: np.ndarray=None, covariance: np.ndarray=None, noise_scale: float=1.0, strength: float=1.0, steps: int=12, relax: float=0.4, quantile: float=1.645) -> float:
    """Evaluate the benchmark's lower normal-equivalent bound.

    Parameters
    ----------
    observed : float ndarray, shape (3,), or None
        Calibration moments in incident-total-flux units; None selects the
        three fixed data values stated in the problem. The root is interior
        to the specified calibration box and has nonsingular Jacobian.
    covariance : float ndarray, shape (3,3), or None
        Positive-semidefinite calibration-measurement covariance; None selects
        the stated correlated matrix, in squared incident-total-flux units.
    noise_scale : float
        Nonnegative multiplier of standard deviations, at most 2.
    strength : float
        Test-phantom modulation multiplier in [0.2,1.3]; calibration is unchanged.
    steps : int
        Retrieval iteration count, 1..20, with simultaneous line corrections.
    relax : float
        Retrieval relaxation in (0,0.6].
    quantile : float
        Nonnegative normal-equivalent lower-bound multiplier.

    Returns
    -------
    out : float
        The lower normal-equivalent bound of the quadratic approximation to
        B(F^{-1}(z+epsilon)), in percentage points, for the declared covariance.
        The public implementation must call and use every preceding public
        function, including through its nested calibration/retrieval evaluations.
    """
    return 0.0
```
