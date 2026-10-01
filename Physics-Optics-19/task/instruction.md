# Physics-Optics-19

## Background

For an incoherent scene $b(\mathbf x)$ the sensor image is the superposition
$I(\mathbf u)=\int b(\mathbf x)\,h(\mathbf u\,|\,\mathbf x)\,d\mathbf x$ of point spread functions. It is a
convolution only if $h$ depends on $\mathbf x$ through a shift. Coma, astigmatism and field curvature
change the PSF across the field, and distortion changes where it lands.

**Two limits on blur.**
- **Geometric:** the spot set by the transverse ray aberrations.
- **Diffraction:** the Airy spot, with first dark ring at $1.22\lambda N$ for working f-number $N$.

The larger one dominates. Stopping down shrinks the geometric spot and enlarges the diffraction spot.

**Scalar diffraction.** For a field $U$ on a surface $\Sigma$, the Rayleigh–Sommerfeld integral of the
first kind gives
$U(P_1)=\frac{1}{j\lambda}\iint_\Sigma U\,\frac{e^{jkr_{01}}}{r_{01}}\cos\theta\,dS$ ($r\gg\lambda$,
$\theta$ measured from the surface normal). It underlies "Huygens PSF" computations; Fresnel and
Fraunhofer are its paraxial and far-field limits. Fourier propagators need the pupil phase on a grid
fine enough for its local frequency, which grows fast with defocus and field.

**Wavefronts and pupils.** The phase is $k\delta$, with $\delta=\int n\,ds$ and $k=2\pi/\lambda$.
- **Wave aberration:** a perfect system produces a spherical wave converging on the image point, and
  the departure from that sphere is the wave aberration.
- **Pupils:** the images of the aperture stop in object and image space. The chief ray passes through
  the stop centre.
- **Paraxial optics:** $2\times2$ unimodular matrices on ray height and reduced angle.

**Why the naive approaches fail.**
- **One PSF per scene point:** too expensive.
- **One convolution:** misses off-axis aberration.
- **Geometric PSFs:** miss diffraction.
- **Plane-pupil Fourier propagators:** need very dense grids.

## Problem

A camera lens of a few refracting surfaces images an incoherent scene onto a pixelated sensor. Its PSF
changes across the field because of off-axis aberrations, and at small apertures diffraction is as large
as the geometric blur.

**Task.** Render the sensor image of a scene through such a lens, combining real ray tracing with scalar
diffraction and approximating the shift-variant superposition integral with a few rendered PSFs, and report
one number from that rendering. The pipeline is decomposed into these eight functions, whose signatures fix
the interfaces used throughout:

```python
def trace_sequential_rays(origins: "np.ndarray", directions: "np.ndarray", surfaces: "np.ndarray",
                          n_object: float) -> tuple: ...  # real 3D ray trace: last-surface hits, directions, OPL, validity
def paraxial_exit_pupil(surfaces: "np.ndarray", n_object: float, stop_z: float,
                        stop_radius: float) -> tuple: ...  # paraxial image of the stop, used for every field
def reference_sphere_field(surfaces: "np.ndarray", n_object: float, stop_z: float, stop_radius: float,
                           tan_field: tuple, pupil_samples: "np.ndarray", wavelength: float,
                           sensor_z: float) -> tuple: ...  # field on the sphere centred on the chief-ray landing, through the pupil centre
def rayleigh_sommerfeld_psf(points: "np.ndarray", field: "np.ndarray", normals: "np.ndarray",
                            pixel_x: "np.ndarray", pixel_y: "np.ndarray", sensor_z: float,
                            wavelength: float) -> "np.ndarray": ...  # Monte-Carlo Rayleigh-Sommerfeld (1st kind) intensity
def chief_ray_landing(scene_shape: tuple, tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                      stop_z: float, sensor_z: float) -> tuple: ...  # field grid and real chief-ray landing (distortion map)
def weighted_latent_images(scene: "np.ndarray", landing: "np.ndarray", node_rows: list, node_cols: list,
                           pixel_pitch: float, sensor_shape: tuple,
                           sensor_center: tuple) -> "np.ndarray": ...  # node-weighted latent images, deposited at landing pixels
def sum_of_convolutions(weighted: "np.ndarray", kernels: "np.ndarray") -> "np.ndarray": ...  # sum of latent images * unit-sum kernels
def render_measurement(scene: "np.ndarray", tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                       stop_z: float, stop_radius: float, sensor_z: float, wavelength: float,
                       pixel_pitch: float, sensor_shape: tuple, sensor_center: tuple,
                       node_rows: list, node_cols: list, pupil_samples: "np.ndarray",
                       kernel_size: int) -> "np.ndarray": ...  # orchestrator calling functions 3-7
```

**Setting and conventions.**
- **Physics:** scalar, monochromatic light (vacuum wavelength λ, in mm); lengths in mm; propagation
  along $+z$.
- **Scene:** at infinity and incoherent; each sample is a collimated beam along
  $(\tan\theta_x,\tan\theta_y,1)/\|\cdot\|$.
- **Stop:** a uniform circular stop in the object medium, in front of the first surface.
- **Sensor:** a plane in air.
- **Surface table:** `(M, 7)` rows `[z_vertex, curvature, conic, a4, a6, n_after, semi_aperture]`, with
  sag $cr^2/(1+\sqrt{1-(1+\kappa)c^2r^2})+a_4r^4+a_6r^6$.
- **Arrays and pixels:** arrays are `[row, col]` = `[y, x]`; scene rows follow $\tan\theta_y$ and
  columns $\tan\theta_x$, on inclusive equally spaced grids, with row 0 and column 0 at the smallest
  $\tan\theta_y$ and $\tan\theta_x$ and both increasing with the index. The pixel containing $(x,y)$ is column
  $\lfloor(x-x_0)/p+W/2\rfloor$, row $\lfloor(y-y_0)/p+H/2\rfloor$.
- **Reference sphere:** it is centred on the chief ray's landing point on the sensor and passes through
  the centre of the exit pupil, so its radius is the distance between those two points.
- **Phase:** it increases with optical path, is counted from the incident plane wavefront through the
  stop centre, and is referenced to the chief ray's own value on the sphere (phase 0). Lost rays have
  amplitude 0.
- **Pupil samples:** equal-area stop positions, each representing the same share of the aperture; no
  stop-to-sphere Jacobian.
- **Normalisation (task convention):** for $N$ equal-phase samples of nonnegative amplitude $a_i$ on a sphere of radius $R$,
  the intensity at the sphere centre is $(\tfrac1N\sum_ia_i)^2/(\lambda R)^2$.
- **Interpolation:** weights are tensor products of piecewise-linear hats on the supplied scene row and column nodes, evaluated at each scene sample's own indices; each one-dimensional hat is one at its node and zero at the other nodes.
- **Kernels:** odd-sized, centred on the node's chief-ray landing, sampled at pixel-pitch offsets,
  normalised to unit sum.
- **Environment:** `float64`/`complex128`; only NumPy/SciPy and the standard library; no I/O.

**Reference configurations** (used in the explanation below; λ = 550 nm, object medium air):
- *Pupil samples:* the N-point golden-angle set
  $s_i=\sqrt{(i+\tfrac12)/N}\,(\cos\varphi_i,\sin\varphi_i)$ with $\varphi_i=\pi(3-\sqrt5)(i+\tfrac12)$,
  for $i=0,\dots,N-1$; each $s_i$ is the stop-plane position $(x,y)$ in units of the stop radius, with
  the cosine term along $x$.
- *Reference singlet:* surface table `[[0, 1/3, 0, 0, 0, 1.5, 2], [1, -1/8, 0, 0, 0, 1.0, 2]]` (a biconvex
  lens in air, radii 3 mm and −8 mm, 1 mm thick, n = 1.5), stop plane at z = −0.2 mm, sensor at
  z = 5 mm, stop radius 0.1 mm or 0.3 mm as stated.
- *Reference ellipsoid:* a single Cartesian ellipsoid with vertex at z = 0, vertex radius 2 mm, glass
  n = 1.5 after it, a 0.3 mm stop at z = −0.5 mm, and the sensor at its focus inside the glass.
- *Reference rendering:* the reference singlet with the 0.3 mm stop.
  - Scene: $7\times5$, where the sample in row $r$, column $c$ has intensity $((5r+c) \bmod 4)+1$, over
    $\tan\theta_x\in[-0.02,0.02]$ and $\tan\theta_y\in[0.10,0.16]$.
  - Sensor: $40\times32$ pixels of 8 µm, centred at $(x_0,y_0)=(0,0.585)$ mm.
  - Nodes and kernels: PSF nodes at scene rows (0, 3, 6) and columns (0, 4), N = 400 pupil samples, and
    $11\times11$ kernels.

**Required explanation.** Before the final answer, explain in prose (LaTeX allowed):
- (a) the centre and radius of the reference sphere, and why a sphere is used;
- (b) which ray–sphere intersection is used and how the optical path is corrected for it; the exact
  signed ray parameter of the on-axis chief ray's sphere intersection and the resulting path
  correction, for the reference singlet and for the reference ellipsoid;
- (c) how the optical path of collimated rays is referenced; and, for the reference singlet at 20°
  field with the 0.1 mm stop, the range of the reference term across the stop in wavelengths, and
  where the computed focus would land if the term were omitted;
- (d) the derivation of the discrete intensity estimator from the Rayleigh–Sommerfeld integral, a
  check of its normalisation, and the weight carried by each sample;
- (e) the angle used in the obliquity factor, and the distance used in the kernel;
- (f) the exit-pupil formulas and their exact values (as fractions) for the reference singlet, with its
  on-axis reference-sphere radius and effective focal length; and the conic constant, focal
  distance, exit pupil and on-axis sphere radius of the reference ellipsoid;
- (g) the derivation of the sum-of-convolutions form from a PSF interpolation, where the interpolation
  weights are evaluated, and what the rendering reduces to when all node PSFs are equal;
- (h) when diffraction rather than aberration limits the blur, with the working f-number and the Airy
  radius of the reference singlet for the 0.1 mm and 0.3 mm stops;
- (i) the distance from the true PSF at which Monte-Carlo sampling artefacts (aliased replicas) appear
  for N equal-area pupil samples, how it scales with N, and its value for the reference singlet with
  the 0.1 mm stop and N = 600;
- (j) the published simulator this pipeline follows (with a link), and, from that publication and its
  public code:
  - the intensity prefactor printed in the publication and the normalisation its code actually
    applies;
  - the coordinate in which it parameterises the scene and its PSFs;
- (k) the following results, computed with your implementation and reported to at least four
  significant figures (row and column indices are 0-based):
  - **Reference singlet, 20° field** ($\tan\theta_y=\tan20°$): the chief-ray landing height and the
    reference-sphere radius.
  - **Reference singlet, 30° field:** the chief-ray landing height, and its departure from
    $f\tan\theta$.
  - **Wavefront error:** with the 0.1 mm stop at 20° field and N = 300, the peak-to-valley of
    $(\delta_i-\delta_{\text{chief}})/\lambda$ over the transmitted samples.
  - **Strehl estimate:** $|N^{-1}\sum_i v_i|^2$ (the normalised intensity at the sphere centre) with
    N = 600, for the 0.1 mm and 0.3 mm stops at 20° field and for the 0.3 mm stop on axis.
  - **Reference rendering:** the total of the measurement, the value and pixel of its maximum, and
    the row of greatest total intensity and the column of greatest total intensity, with their sums.

**Result to report.** The scientific result to report is the largest value of the rendered measurement
for the reference rendering, in scene intensity units, to at least six significant figures.

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

trace_sequential_rays

Goal
----
Trace real rays through the sequential surface table. Return, for every ray, its intersection with the last surface, its unit direction after refraction there, the optical path length from its origin, and a validity flag. Solve aspheric intersections numerically to full double precision. A ray that misses a surface (no real intersection ahead of it), lands outside a semi-aperture or is totally internally reflected is invalid from then on, and its position, direction and path are NaN. Input directions need not be normalised; plane surfaces (curvature 0) and a non-air object medium must work.

```python
def trace_sequential_rays(origins: "np.ndarray", directions: "np.ndarray", surfaces: "np.ndarray",
                          n_object: float) -> tuple:
    """Trace rays through sequential rotationally symmetric aspheric surfaces.

    Args:
        origins: float (N, 3), start points [x, y, z] in mm, in the object medium before surface 1.
        directions: float (N, 3), ray directions (any length).
        surfaces: float (M, 7), rows [z_vertex, curvature, conic, a4, a6, n_after, semi_aperture]
            along +z (mm, 1/mm, -, 1/mm^3, 1/mm^5, -, mm).
        n_object: refractive index of the starting medium.

    Returns:
        tuple (positions (N, 3) at the last surface in mm, directions_out (N, 3) unit, opl (N,)
        optical path from the origin in mm, valid (N,) bool); invalid rows are NaN.
    """
    return positions, directions_out, opl, valid
```

### Step 2

paraxial_exit_pupil

Goal
----
Return the axial position and radius of the exit pupil: the paraxial image of the aperture stop formed by all surfaces after it. The position is a signed coordinate on the same axis as the surface vertices and the radius a positive length; the pupil may be virtual, that is, lie before the last surface. Must handle a stop at the first vertex and a non-air object medium.

```python
def paraxial_exit_pupil(surfaces: "np.ndarray", n_object: float, stop_z: float,
                        stop_radius: float) -> tuple:
    """Paraxial exit pupil: image of the aperture stop through every surface after it.

    Args:
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: refractive index of the medium containing the stop.
        stop_z: stop plane position in mm (before surface 1).
        stop_radius: stop radius in mm.

    Returns:
        tuple (z_exit_pupil, exit_pupil_radius) of floats in mm; the radius is positive.
    """
    return z_exit_pupil, exit_pupil_radius
```

### Step 3

reference_sphere_field

Goal
----
For one field direction, return the complex field the lens delivers onto the exit-pupil reference sphere. That sphere passes through the paraxial exit-pupil centre of Step 2, and its radius is the distance from its centre to that point. Rays start at the stop points stop_radius·sample at z = stop_z, all travelling along the field direction, and are traced with Step 1; each is then continued in the last medium until it meets that sphere on the cap containing the exit-pupil centre, which may lie behind the last surface. Return each ray's point on the sphere and its complex amplitude, with phase increasing with optical path and the chief ray's own value on the sphere as the zero of phase: amplitude 1 for a ray that reaches the sphere, 0 for a ray lost anywhere, whose point is set to the exit-pupil centre. Also return unit sphere normals pointing to the centre, the centre and the radius. Must call trace_sequential_rays and paraxial_exit_pupil by name; raise ValueError if the chief ray is lost.

```python
def reference_sphere_field(surfaces: "np.ndarray", n_object: float, stop_z: float, stop_radius: float,
                           tan_field: tuple, pupil_samples: "np.ndarray", wavelength: float,
                           sensor_z: float) -> tuple:
    """Complex field on the exit-pupil reference sphere for one field direction.

    Args:
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: object-medium index (contains the stop).
        stop_z, stop_radius: stop plane position and radius, mm.
        tan_field: (tan_theta_x, tan_theta_y); rays travel along (tx, ty, 1) normalised.
        pupil_samples: float (N, 2), stop coordinates (x, y) in units of the stop radius; column 0 is x.
        wavelength: vacuum wavelength, mm.
        sensor_z: sensor plane position, mm (in the last medium).

    Returns:
        tuple (points (N, 3) on the sphere in mm, lost rays at the exit-pupil centre;
        field (N,) complex, amplitude*exp(1j*phase), chief phase 0, lost rays 0;
        normals (N, 3) unit, toward the centre; center (3,) sphere centre in mm;
        radius float in mm).

    Raises:
        ValueError: if the chief ray does not reach the sensor.
    """
    return points, field, normals, center, radius
```

### Step 4

rayleigh_sommerfeld_psf

Goal
----
Propagate the sampled field to a rectangular grid of sensor points with the first Rayleigh–Sommerfeld integral evaluated as a Monte-Carlo sum over the samples, and return the intensity. The obliquity factor uses the given normal at each sample; the kernel keeps the exact point-to-point distance (no paraxial or far-field simplification); every sample carries the same weight (equal-area stop samples, no Jacobian). Normalisation (task convention; the source prints a different prefactor): for N equal-phase samples of nonnegative amplitude a_i on a sphere of radius R, the intensity at the sphere centre is (mean a_i)²/(λR)², so repeating every sample changes nothing. The sensor is in air; the field follows the Step 3 phase convention; the output is indexed [pixel_y, pixel_x].

```python
def rayleigh_sommerfeld_psf(points: "np.ndarray", field: "np.ndarray", normals: "np.ndarray",
                            pixel_x: "np.ndarray", pixel_y: "np.ndarray", sensor_z: float,
                            wavelength: float) -> "np.ndarray":
    """Intensity on a sensor grid from scattered samples of a complex field.

    Args:
        points: float (N, 3), sample positions, mm.
        field: complex (N,), field at the samples.
        normals: float (N, 3), unit source-surface normals toward the sensor side.
        pixel_x: float (W,), sensor x coordinates, mm.
        pixel_y: float (H,), sensor y coordinates, mm.
        sensor_z: sensor plane position, mm.
        wavelength: vacuum wavelength, mm.

    Returns:
        float (H, W), intensity at (pixel_y[row], pixel_x[col], sensor_z).
    """
    return intensity
```

### Step 5

chief_ray_landing

Goal
----
Build the scene's field-direction grid and map every sample to the sensor with its real chief ray (from the stop centre along the sample's direction, traced with Step 1 and continued to the sensor plane). Column j has tanθx equal to the j-th of Ws equally spaced values from tx_min to tx_max, and row i has tanθy equal to the i-th of Hs values from ty_min to ty_max (inclusive). Must call trace_sequential_rays by name; lost chief rays give NaN landings.

```python
def chief_ray_landing(scene_shape: tuple, tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                      stop_z: float, sensor_z: float) -> tuple:
    """Field-direction grid of the scene and the real chief-ray landing of each sample.

    Args:
        scene_shape: (Hs, Ws).
        tan_bounds: (tx_min, tx_max, ty_min, ty_max), inclusive.
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: object-medium index.
        stop_z: stop plane position, mm (chief rays start at (0, 0, stop_z)).
        sensor_z: sensor plane position, mm.

    Returns:
        tuple (tan_grid (Hs, Ws, 2) with [..., 0] = tan_theta_x and [..., 1] = tan_theta_y,
        landing (Hs, Ws, 2) sensor [x, y] in mm, NaN if lost).
    """
    return tan_grid, landing
```

### Step 6

weighted_latent_images

Goal
----
Split the scene into per-node latent images. The nodes form a separable sub-grid: node_rows are strictly increasing scene row indices from 0 to Hs-1, node_cols likewise for Ws (a single node on an axis, any index, has weight 1 everywhere along it; any other invalid list raises ValueError). Each sample is weighted, for every node, by the node's piecewise-linear weight along rows times that along columns, evaluated at the sample's own indices (weights sum to one per sample), and added into the pixel where its chief ray lands: column floor((x-x0)/p+W/2), row floor((y-y0)/p+H/2). Samples with NaN or off-sensor landings are dropped.

```python
def weighted_latent_images(scene: "np.ndarray", landing: "np.ndarray", node_rows: list, node_cols: list,
                           pixel_pitch: float, sensor_shape: tuple,
                           sensor_center: tuple) -> "np.ndarray":
    """Per-node weighted latent images on the sensor grid.

    Args:
        scene: float (Hs, Ws), scene intensity on the field-direction grid.
        landing: float (Hs, Ws, 2), chief-ray landing [x, y] of each sample, mm.
        node_rows, node_cols: lists of Gy / Gx node indices into the scene rows / columns.
        pixel_pitch: pixel pitch p, mm.
        sensor_shape: (H, W).
        sensor_center: (x0, y0), mm.

    Returns:
        float (Gy, Gx, H, W); [gy, gx] is the latent image of node (node_rows[gy], node_cols[gx]).

    Raises:
        ValueError: if a node list of length > 1 is not strictly increasing from 0 to n-1.
    """
    return weighted
```

### Step 7

sum_of_convolutions

Goal
----
Form the measurement as the sum over nodes of each node's latent image blurred by that node's kernel, after normalising each kernel to unit sum. Kernels are square with odd size K and centre c = (K-1)/2; kernels[gy, gx, c+dy, c+dx] is the share a point in pixel (row, col) sends to pixel (row+dy, col+dx). The output has the sensor's shape; light leaving the sensor is lost. Raise ValueError unless kernels are (Gy, Gx, K, K) with K odd and matching the node grid.

```python
def sum_of_convolutions(weighted: "np.ndarray", kernels: "np.ndarray") -> "np.ndarray":
    """Sum over nodes of each latent image blurred by its unit-sum kernel.

    Args:
        weighted: float (Gy, Gx, H, W), per-node latent images (Step 6).
        kernels: float (Gy, Gx, K, K), K odd, not necessarily normalised; kernels[gy, gx, c+dy, c+dx]
            (c = (K-1)//2) is the share a point in pixel (row, col) sends to (row+dy, col+dx).

    Returns:
        float (H, W), measurement.

    Raises:
        ValueError: on inconsistent kernel shape or even K.
    """
    return measurement
```

### Step 8

render_measurement

Goal
----
Render the measurement of the scene through the lens, from the scene's field-direction bounds to the image on the sensor, using the five earlier steps. One PSF is rendered per node of the node_rows × node_cols sub-grid; every kernel is kernel_size × kernel_size with kernel_size odd, sampled at offsets spaced by the pixel pitch from that node's own sphere centre, with the centre index at zero offset. Must call chief_ray_landing, reference_sphere_field, rayleigh_sommerfeld_psf, weighted_latent_images and sum_of_convolutions by name, without inlined reimplementation.

```python
def render_measurement(scene: "np.ndarray", tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                       stop_z: float, stop_radius: float, sensor_z: float, wavelength: float,
                       pixel_pitch: float, sensor_shape: tuple, sensor_center: tuple,
                       node_rows: list, node_cols: list, pupil_samples: "np.ndarray",
                       kernel_size: int) -> "np.ndarray":
    """Interpolated wave-optics measurement of an incoherent scene.

    Must call chief_ray_landing, reference_sphere_field, rayleigh_sommerfeld_psf,
    weighted_latent_images and sum_of_convolutions by name.

    Args:
        scene: float (Hs, Ws), intensity on the grid defined by tan_bounds (inclusive bounds).
        surfaces, n_object, stop_z, stop_radius, sensor_z, wavelength: lens, stop, sensor plane
            and vacuum wavelength as in Step 3 (mm); the sensor is in air.
        pixel_pitch, sensor_shape, sensor_center: pitch (mm), (H, W) and (x0, y0) (mm).
        node_rows, node_cols: PSF node indices into the scene grid (see Step 6).
        pupil_samples: float (N, 2), normalised stop coordinates (x, y) used for every PSF; column 0 is x.
        kernel_size: odd K, kernel size in pixels.

    Returns:
        float (H, W), measurement.
    """
    return measurement
```
