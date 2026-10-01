# Physics-Optics-21

## Background

Computed laminography is the tomographic technique used when the specimen is a plate: a printed
circuit board, a joint in an aircraft skin, a painting on panel. Rotating such an object in a
conventional scanner produces views in which the rays travel almost parallel to the plate and are
absorbed far more strongly than in any other direction, which starves those projections of
information. Inclining the rotation axis with respect to the plate keeps every ray at a workable
angle, at the cost of an angular range that never closes: the sampled region of frequency space has a
missing cone, and the reconstruction problem stays underdetermined no matter how many views are taken.

What fills that gap is prior knowledge about the object, added to the reconstruction as a penalty on
the image. Penalising the total variation of the volume expresses the expectation that the object is
piecewise smooth, which suppresses noise while keeping edges. Penalising a measure of rank expresses a
different expectation: that the slices of the volume are not independent, so that the stack as a whole
is far simpler than the sum of its parts. Plate-like specimens satisfy both, and they satisfy them
anisotropically, because the structure repeats through the stack while varying within each layer.

Because these penalties are not differentiable, the reconstruction is computed with a first-order
primal-dual method, which alternates cheap ascent steps on dual variables, one per penalty, with a
proximal step on the image. Everything that makes such a scheme work in practice lives in three
places: the proximal mappings, which must be known in closed form; the step sizes, which are tied to
the norm of the stacked forward operator; and the exactness of the operator used to bring residuals
back from the detector to the volume.

## Problem

A flat, plate-like object such as a printed circuit board cannot be scanned by ordinary computed
tomography, because the rays that graze its plane travel through far too much material. Computed
laminography solves this by tilting the scan: the source and detector rotate about an axis inclined to
the plate, so every ray crosses it at a useful angle. The price is that the angular coverage is
incomplete, the reconstruction problem is underdetermined, and a plain algebraic inversion leaves
aliasing artefacts that swamp the layered structure the scan is meant to resolve. For the single
configuration specified below, reconstruct the volume with a regularised iterative scheme that
combines a direction-dependent gradient penalty with a low-rank penalty on the stack of slices, and
report one number: the root-mean-square error of the reconstruction against the true volume.

**Imaging model.** The unknown volume is $f\in\mathbb R^{N_1\times N_2\times N_3}$, the measured
projections are $g$, and the scanner is described by a linear system matrix $A$, so that $Af=g$. The
reconstruction is obtained by minimising a data-fidelity term plus regularisation, and the minimiser
is computed with a first-order primal-dual scheme.

**Configuration.** Geometry, object, data and truth are fixed by this code; run it as written.

```python
import numpy as np

nx, ny, nz = 48, 48, 16                 # volume, f[i, j, k]; k stacks the plate layers
det_size, det_pixel = 64, 1.0           # detector pixels and their size in mm
voxel_size = 0.5                        # mm
n_views, theta_deg = 24, 45.0           # views over a full turn, and the laminographic slope
sod, sdd = 230.0, 700.0                 # source-object and source-detector distances in mm

rng = np.random.default_rng(0)          # a plate with repeated routing layers
base = np.zeros((nx, ny))
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
for _ in range(6):
    y0 = rng.integers(6, ny - 6)
    x0, x1 = sorted(rng.integers(4, nx - 4, 2))
    base[x0:x1, y0 - 1:y0 + 1] = 1.0
for _ in range(4):
    cy, cx = rng.uniform(8, nx - 8, 2)
    base += 0.8 * (((xx - cx) ** 2 + (yy - cy) ** 2) < 3.5 ** 2).T
f_true = np.zeros((nx, ny, nz))
for k in range(nz):
    if k % 4 in (0, 1):                 # two-voxel layers separated by two-voxel gaps
        f_true[:, :, k] = base * (1.0 if k % 8 < 4 else 0.7)
f_true = np.clip(f_true, 0.0, None)
```

The projections are the noiseless forward projection of `f_true` through the geometry above, plus
Gaussian noise of zero mean and standard deviation equal to 1 % of the largest noiseless projection
value, drawn with `np.random.default_rng(1)` in one call of the shape of the projection array.

**Scheme settings.** Regularisation weights $\lambda_1=\lambda_2=0.2$ along the two in-plane
directions, $\lambda_3=0.01$ along the stacking direction and $\lambda_4=40$ on the low-rank term.
Start from $f=0$ with all dual variables zero, set the primal and dual step sizes to $1/L$ with $L$ the
spectral norm of the stacked operator, and run exactly 30 iterations. Each iteration closes by forming
the extrapolated image as $\bar f_{n+1}=f_n+\gamma\,(f_{n+1}-f_n)$ with $\gamma=1$, so the extrapolated
image equals the new one. Estimate $L$ by 50 power iterations starting from the volume whose entries
are all 1.

**Required reasoning.** Justify the computation, covering:
- (a) the minimisation problem being solved: the data-fidelity term, the three direction-dependent
  gradient penalties and the low-rank penalty, and why the weight along the stacking direction is set
  far below the in-plane ones for this class of object;
- (b) the primal-dual scheme: the splitting of the objective, the proximal mappings of the conjugates
  of the data term and of each gradient term, and the step-size rule;
- (c) the proximal mapping of the low-rank penalty: what transform it is built on, what is thresholded
  and how the result is returned to the spatial domain;
- (d) the relation between the projector and the operator used to bring the residual back to the
  volume, and how you verified it numerically;
- (e) the value of $L$ for this configuration, and the resulting step sizes;
- (f) the experimental setting reported in the source publication for the phantom whose scanning
  geometry this configuration reuses: the class of phantom and the noise model the publication adds
  to its projections;
- (g) how the answer moves under three single-change variants of the same run, everything else held
  fixed: dropping the low-rank penalty ($\lambda_4=0$); making the gradient penalty isotropic
  ($\lambda_3=\lambda_1=\lambda_2=0.2$); and thresholding the singular values of the frontal slices of
  the volume itself instead of the transformed ones. Give each resulting error and its relative
  distance from the answer;
- (h) what the answer becomes when the iteration is closed instead with the textbook primal-dual
  extrapolation $\bar f_{n+1}=2f_{n+1}-f_n$: the resulting error at thirty iterations, which of the two
  closures gives the lower error there, and how close the two come to each other when both are run on
  to two hundred iterations;
- (i) the error of the prescribed run along its trajectory, after ten and after fifty iterations, and
  what that says about whether thirty iterations is a converged point;
- (j) the root-mean-square error of the zero volume against the truth, and the fraction of it that the
  reconstruction reaches;
- (k) the resulting root-mean-square error.

**Fixed conventions.** Volumes are `float64` arrays `f[i, j, k]` of shape `(nx, ny, nz)`; projections
are `float64` arrays `g[v, w, u]` of shape `(n_views, det_size, det_size)`. For view $v$ the source
sits at $S_v=\mathrm{sod}\,(\sin\theta\cos\phi_v,\ \sin\theta\sin\phi_v,\ \cos\theta)$ with
$\phi_v=2\pi v/n_{\text{views}}$; the detector plane is perpendicular to $S_v$ at distance
$\mathrm{sdd}$ from the source, with in-plane axes $e_1=(-\sin\phi_v,\cos\phi_v,0)$ and
$e_2=\hat S_v\times e_1$, and the detector centre at index `det_size / 2`. Voxel centres are offset by
half a voxel from the grid corner, so the volume is centred on the origin: voxel `f[i, j, k]` sits at
$\big((i+\tfrac12-n_x/2),\ (j+\tfrac12-n_y/2),\ (k+\tfrac12-n_z/2)\big)\times$`voxel_size` millimetres.
The first detector coordinate is measured along $e_1$ and the second along $e_2$, and projections
are indexed `g[v, w, u]` with `u` the first and `w` the second. A voxel contributes to the detector by bilinear deposition at its
perspective image, clamped to the detector edges. Gradients are
forward differences with periodic wrap. Use NumPy/SciPy and the standard library only; no file or
network I/O.

**The answer.** Report

$$E=\sqrt{\frac{1}{N_1N_2N_3}\sum_{i,j,k}\big(\hat f_{i,j,k}-f^{\text{true}}_{i,j,k}\big)^2},$$

the root-mean-square error of the reconstruction against the true volume, as a single decimal number
with at least four significant figures.

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

cl_detector_coords

Goal
----
Compute, for every view, the detector coordinates at which each voxel centre is imaged by the rotational cone-beam laminography geometry. Return one array holding the two coordinate maps for all views, so that the projector and its transpose consume exactly the same geometry. Voxel centres are offset by half a voxel from the grid corner, and the detector centre sits at index det_size / 2. The submitted function must import inside itself whatever it uses.

```python
def cl_detector_coords(shape: "tuple", n_views: int, theta_deg: float, sod: float,
                       sdd: float, voxel_size: float, det_pixel: float,
                       det_size: int) -> "np.ndarray":
    """Detector coordinates of every voxel centre, for every view.

    Args:
        shape: (nx, ny, nz) of the volume, in voxels.
        n_views: number of views, equally spaced over a full turn.
        theta_deg: laminographic slope in degrees.
        sod: source-to-object distance, mm.
        sdd: source-to-detector distance, mm.
        voxel_size: isotropic voxel size, mm.
        det_pixel: detector pixel size, mm.
        det_size: number of detector pixels along each side.

    Returns:
        np.ndarray: float64, shape (2, n_views, nz, ny, nx); index 0 carries the coordinate
            along e1 and index 1 the coordinate along e2, both in detector pixels.
    """
    return coords
```

### Step 2

cl_forward_project

Goal
----
Apply the system matrix A: deposit every voxel's value bilinearly at its detector image, using the coordinates of step 1, and accumulate over all voxels. Contributions are clamped to the detector edges, so a voxel imaged outside the detector deposits on the border pixel rather than being discarded. The operator is linear in the volume for fixed geometry. The submitted function must import inside itself whatever it uses.

```python
def cl_forward_project(volume: "np.ndarray", coords: "np.ndarray",
                       det_size: int) -> "np.ndarray":
    """Project a volume onto the detector by bilinear deposition.

    Args:
        volume: float64 array of shape (nx, ny, nz).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        det_size: number of detector pixels along each side.

    Returns:
        np.ndarray: float64, shape (n_views, det_size, det_size).
    """
    return projections
```

### Step 3

cl_back_project

Goal
----
Apply the exact discrete transpose of step 2 for the same geometry: read the projection values at the same four detector pixels, weight them with the same bilinear weights and accumulate into the voxel. This is the transpose, not an inverse and not a filtered back-projection; it must satisfy the inner-product identity with step 2 to machine precision for arbitrary inputs. The submitted function must import inside itself whatever it uses.

```python
def cl_back_project(projections: "np.ndarray", coords: "np.ndarray",
                    shape: "tuple") -> "np.ndarray":
    """Apply the exact transpose of the projector of step 2.

    Args:
        projections: float64 array of shape (n_views, det_size, det_size).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        shape: (nx, ny, nz) of the volume.

    Returns:
        np.ndarray: float64, shape (nx, ny, nz).
    """
    return volume
```

### Step 4

anisotropic_gradient

Goal
----
Apply the forward difference along one axis with periodic wrap, or its exact transpose when adjoint is true. Axis 0 and 1 are the in-plane directions and axis 2 is the stacking direction; the three are applied with different weights by the reconstruction, which is what makes the regularisation anisotropic. Raise ValueError for any axis other than 0, 1 or 2. The submitted function must import inside itself whatever it uses.

```python
def anisotropic_gradient(volume: "np.ndarray", axis: int,
                         adjoint: bool = False) -> "np.ndarray":
    """Forward difference along an axis with periodic wrap, or its transpose.

    Args:
        volume: float64 array of shape (nx, ny, nz).
        axis: 0, 1 or 2.
        adjoint: if True, apply the transpose instead of the forward difference.

    Returns:
        np.ndarray: float64, same shape as the input.

    Raises:
        ValueError: if axis is not 0, 1 or 2.
    """
    return result
```

### Step 5

tensor_svt

Goal
----
Apply the proximal mapping of the low-rank penalty used by the source method: the tensor singular value thresholding operator. It transforms the volume along the stacking direction, applies matrix singular value thresholding with the given threshold to the transformed slices, and returns the result to the spatial domain, taking the real part. Raise ValueError for a negative threshold. The submitted function must import inside itself whatever it uses.

```python
def tensor_svt(tensor: "np.ndarray", threshold: float) -> "np.ndarray":
    """Proximal mapping of the tensor nuclear norm.

    Args:
        tensor: float64 array of shape (n1, n2, n3).
        threshold: non-negative soft-thresholding level.

    Returns:
        np.ndarray: float64, same shape as the input.

    Raises:
        ValueError: if threshold is negative.
    """
    return thresholded
```

### Step 6

operator_norm

Goal
----
Estimate the spectral norm L of the stacked operator built from the projector of step 2 and the three gradients of step 4, by the power iteration on the normal operator fixed in the background: start from the unnormalised volume whose entries are all 1 and run exactly n_power iterations of b = Bx, s = ||b||, x = b/s, returning sqrt(s) from the last iteration. The step sizes of the reconstruction are 1/L, so this value fixes them. The submitted function must import inside itself whatever it uses.

```python
def operator_norm(coords: "np.ndarray", shape: "tuple", det_size: int,
                  n_power: int = 50) -> float:
    """Spectral norm of the stacked operator, by power iteration.

    Args:
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        shape: (nx, ny, nz) of the volume.
        det_size: number of detector pixels along each side.
        n_power: number of power iterations.

    Returns:
        float: sqrt(s) from the last iteration of the recurrence given in the scientific
            background, where s is the Euclidean norm of B x before renormalisation.
    """
    return norm
```

### Step 7

agslr_cp_update

Goal
----
Apply one iteration of the reconstruction scheme to the current state. The state is the tuple (x, xbar, y, p, q, r) holding the image, the extrapolated image, the dual variable of the data term and the three dual variables of the gradient terms. Update the four duals with the proximal mappings of their conjugates, take the proximal step on the image with the low-rank operator of step 5, and form the new extrapolated image exactly as the source method prints it. Must call the earlier step functions by name; no inlined reimplementation. Raise ValueError if any of the first three regularisation weights is not positive. The submitted function must import inside itself whatever it uses.

```python
def agslr_cp_update(state: "tuple", projections: "np.ndarray", coords: "np.ndarray",
                    det_size: int, lambdas: "tuple", tau: float, sigma: float,
                    gamma: float = 1.0) -> tuple:
    """One iteration of the reconstruction scheme.

    Args:
        state: (x, xbar, y, p, q, r); x, xbar, p, q, r are float64 arrays of shape
            (nx, ny, nz) and y is a float64 array shaped like the projections.
        projections: float64 array of shape (n_views, det_size, det_size).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        det_size: number of detector pixels along each side.
        lambdas: (lambda1, lambda2, lambda3, lambda4), the three gradient weights and the
            low-rank weight.
        tau: primal step size.
        sigma: dual step size.
        gamma: relaxation parameter.

    Returns:
        tuple: the updated (x, xbar, y, p, q, r).

    Raises:
        ValueError: if lambda1, lambda2 or lambda3 is not positive.
    """
    return updated
```

### Step 8

run_agslr_cp

Goal
----
Run the whole reconstruction: build the geometry with step 1, estimate the spectral norm with step 6, set both step sizes to its reciprocal, start from a zero image with zero duals, and apply n_iter iterations of step 7 with relaxation 1. Return the reconstructed volume. Must call the earlier step functions by name; no inlined reimplementation. The submitted function must import inside itself whatever it uses.

```python
def run_agslr_cp(projections: "np.ndarray", shape: "tuple", n_views: int,
                 theta_deg: float, sod: float, sdd: float, voxel_size: float,
                 det_pixel: float, det_size: int, lambdas: "tuple",
                 n_iter: int = 30, n_power: int = 50) -> "np.ndarray":
    """Reconstruct a volume from laminographic projections.

    Args:
        projections: float64 array of shape (n_views, det_size, det_size).
        shape: (nx, ny, nz) of the volume.
        n_views: number of views.
        theta_deg: laminographic slope in degrees.
        sod: source-to-object distance, mm.
        sdd: source-to-detector distance, mm.
        voxel_size: isotropic voxel size, mm.
        det_pixel: detector pixel size, mm.
        det_size: number of detector pixels along each side.
        lambdas: (lambda1, lambda2, lambda3, lambda4).
        n_iter: number of iterations.
        n_power: number of power iterations for the spectral norm.

    Returns:
        np.ndarray: float64, shape (nx, ny, nz), the reconstruction.
    """
    return reconstruction
```
