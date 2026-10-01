# Physics-Computational_Physics-9

## Background

Many astrophysical systems collapsing molecular cloud cores, outbursting young stars, and above all protoplanetary discs are shaped by the gravity of their own dilute gas, referred to as self-gravity. In young, massive discs, self-gravity drives gravitational instability, which transports angular momentum, sustains a turbulent state ("gravito-turbulence"), can amplify magnetic fields through a dynamo process, and, when radiative cooling is efficient, fragments the disc into bound clumps a candidate channel for giant-planet and binary-star formation. Simulating these processes requires solving the Poisson equation for the gravitational potential of the gas at every time step of a (magneto)hydrodynamic simulation, so the Poisson solver's accuracy and cost directly limit what physics can be studied.

Because global disc simulations are expensive, much of this work is done in the shearing box: a small co-rotating Cartesian patch of the disc in which the differential (Keplerian) rotation appears as a linear background shear flow. The patch is periodic in the azimuthal direction and shear-periodic in the radial direction, its radial boundaries slide past each other at the shear speed, so the box is strictly periodic in the radial coordinate only at special instants. Crucially, a realistic disc is vertically stratified: the gas density falls off with height and the region above and below the disc is effectively vacuum, so the correct vertical boundary condition for the potential is decay at infinity, not periodicity.

This mix of boundary conditions is awkward for the standard numerical tool kit. Fast Fourier transforms give spectral accuracy at N log N cost but inherently assume the input is one period of a periodic signal, which would surround the disc with infinite spurious vertical copies of itself; iterative multigrid solvers handle general boundaries but need accurate boundary potentials from multipole expansions or screening methods, and give up spectral accuracy. A family of spectral techniques developed in computational physics addresses free-space ("unbound") problems by modifying the Green's function of the Poisson equation, and recent work in disc astrophysics adapts this idea to the shearing box's hybrid case of two (shear-)periodic in-plane directions and one vacuum vertical direction. The resulting solvers execute as a single three-dimensional FFT convolution compatible with pencil-decomposed parallel transform libraries and consume only a few percent of the runtime of production magnetohydrodynamic simulations, enabling high-resolution local studies of gravito-turbulence and disc fragmentation. Benchmarking such a solver requires a density configuration for which the potential under the same mixed boundary conditions is known independently, so that the numerical field can be compared against it point by point.

## Problem

Self-gravitating, vertically stratified protoplanetary discs are studied in the shearing box, a local patch that is (shear-)periodic in the two horizontal directions but faces vacuum above and below the disc. Fast Fourier transforms give spectral accuracy at $N\log N$ cost but assume full periodicity, so they surround the disc with spurious vertical images, while multigrid solvers handle the open vertical boundary at the price of spectral accuracy; bridging that gap is what makes self-gravitating stratified boxes tractable at production resolutions. Given a density field sampled on a cubic shearing-box grid, the method returns the gravitational potential satisfying $\nabla^2\Phi = 4\pi G\rho$ under in-plane (shear-)periodicity and decay as $|z|\to\infty$, and the quantity of interest here is the accuracy it achieves against an exactly known reference.

The computation turns on three things: the wave-vectors that make the sheared box Fourier-transformable at a given time, the vertical kernel that encodes vacuum rather than periodic boundaries, and the modification and grid enlargement that make that kernel exactly representable by FFTs on a finite domain. The reference field is the potential of the same source under the same boundary conditions, obtainable in closed form because the source is separable.

Benchmark this solver on a cubic box $L_x = L_y = L_z = L = 6$ resolved by $N_x = N_y = N_z = N = 32$ cell-centred points per dimension, $x_i = -L/2 + (i + 1/2)L/N$, hosting the density $\rho(x,y,z) = \cos(2\pi x/L)\cos(2\pi y/L)\,e^{-z^2/(2H^2)}$ with $H = 1$ inside the box and vacuum outside, at $G = 1$, $\Omega = 1$, Keplerian shear rate $q = 3/2$, time $t = 0$, kernel truncation factor $\alpha = 1.1$ in units of $L_z$, and a fourfold vertical enlargement of the computational domain. Shift the numerical and the exact potential each by the constant that makes its own minimum over the grid equal to $1$, form the pointwise relative error $\epsilon = |(\Phi_{\mathrm{num}} - \Phi_{\mathrm{ref}})/\Phi_{\mathrm{ref}}|$ on the shifted fields, and report its root-mean-square over all $N^3$ cells, $\lVert\epsilon\rVert_2 = \sqrt{N^{-3}\sum_{ijk}\epsilon_{ijk}^2}$. Your reasoning should also report, as evidence that the pipeline ran: the two gauge-fixing constants it used; the largest pointwise value of $\epsilon$ and the cell where it occurs; how far the horizontal-slice mean of $\epsilon$ falls from the outermost slices to those nearest the midplane; and $\lVert\epsilon\rVert_2$ recomputed at $N = 16$ and $N = 64$ together with what the three values imply about the order of convergence.


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

01_build_density_grid

Goal
----
Build the cell-centered coordinate cube of the shearing box and evaluate the single-mode, vertically Gaussian source density on it.

```python
import numpy as np

def build_density_grid(N: int, L: float, H: float, m_x: int = 1,
                       m_y: int = 1) -> np.ndarray:
    """Build the cell-centered grid and the single-mode Gaussian density.

    Parameters
    ----------
    N : int
        Number of cells per dimension of the cubic box (N >= 4).
    L : float
        Box size in every dimension (L > 0).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    m_x : int
        Integer mode number of the density in x, kx = 2*pi*m_x/L.
    m_y : int
        Integer mode number of the density in y, ky = 2*pi*m_y/L (m_x and
        m_y must not both be zero).

    Returns
    -------
    grid : np.ndarray
        Array of shape (4, N, N, N). grid[0], grid[1], grid[2] are the
        cell-centered x, y, z coordinate cubes produced with "ij" indexing
        from x_i = -L/2 + (i + 1/2) * L/N, and grid[3] is the density
        cos(kx*x) * cos(ky*y) * exp(-0.5 * (z/H)**2) evaluated on them.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((4, N, N, N), dtype=float)  # placeholder
```

### Step 2

02_compute_shear_wavevectors

Goal
----
Build the shear-corrected in-plane wave-vector grids of the fully periodic frame at time t.

```python
import numpy as np

def compute_shear_wavevectors(Nx: int, Ny: int, Lx: float, Ly: float,
                              Omega: float, t: float) -> np.ndarray:
    """Build the shear-corrected in-plane wave-vector grids at time t.

    Parameters
    ----------
    Nx : int
        Number of grid cells in the x direction (Nx >= 1).
    Ny : int
        Number of grid cells in the y direction (Ny >= 1).
    Lx : float
        Box size in the x direction (Lx > 0).
    Ly : float
        Box size in the y direction (Ly > 0).
    Omega : float
        Orbital frequency of the shearing box (finite real number).
    t : float
        Time at which the wave-vectors are evaluated (finite real number).

    Returns
    -------
    k_grids : np.ndarray
        Array of shape (2, Nx, Ny). k_grids[0][i, j] is the shear-corrected
        radial wavenumber kx(t) for mode (i, j) and k_grids[1][i, j] is the
        azimuthal wavenumber ky, both in FFT frequency ordering.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((2, Nx, Ny), dtype=float)  # placeholder
```

### Step 3

03_greens_function_hat

Goal
----
Evaluate the analytical Fourier-space Green's function of the truncated free-space kernel elementwise on wavenumber grids.

```python
import numpy as np

def greens_function_hat(k_perp: np.ndarray, kz: np.ndarray, L: float) -> np.ndarray:
    """Evaluate the truncated free-space Green's function in Fourier space.

    Parameters
    ----------
    k_perp : np.ndarray
        In-plane wavenumber magnitudes sqrt(kx^2 + ky^2), all >= 0. May be
        any shape broadcastable against kz.
    kz : np.ndarray
        Vertical wavenumbers, broadcastable against k_perp.
    L : float
        Truncation half-width of the kernel, L = alpha * Lz with alpha > 1
        (L > 0).

    Returns
    -------
    g_hat : np.ndarray
        The analytic Fourier-space Green's function evaluated elementwise on
        the broadcast shape of (k_perp, kz), as a float array.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.broadcast(np.asarray(k_perp), np.asarray(kz)).shape, dtype=float)  # placeholder
```

### Step 4

04_zero_pad_density

Goal
----
Zero-pad the density cube along its vertical (last) axis to emulate the aperiodic vertical convolution.

```python
import numpy as np

def zero_pad_density(rho: np.ndarray, pad_factor: int) -> np.ndarray:
    """Zero-pad a 3D density cube along its vertical (last) axis.

    Parameters
    ----------
    rho : np.ndarray
        Density of shape (Nx, Ny, Nz) with only finite values.
    pad_factor : int
        Integer enlargement factor of the vertical dimension (pad_factor
        >= 2). The padded cube has Nz_pad = pad_factor * Nz vertical cells.

    Returns
    -------
    rho_padded : np.ndarray
        Float array of shape (Nx, Ny, pad_factor * Nz) whose first Nz
        vertical slices equal rho and whose remaining slices are zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(rho).shape[:2] + (pad_factor * np.asarray(rho).shape[2],), dtype=float)  # placeholder
```

### Step 5

05_spectral_convolution_potential

Goal
----
Solve for the gravitational potential with a single 3D FFT convolution against the Fourier-space kernel and crop to the physical cells.

```python
import numpy as np

def spectral_convolution_potential(rho_padded: np.ndarray, greens_hat_grid: np.ndarray,
                                   Nz_out: int, G: float = 1.0) -> np.ndarray:
    """Solve for the potential by one 3D spectral convolution.

    Parameters
    ----------
    rho_padded : np.ndarray
        Zero-padded density of shape (Nx, Ny, Nz_pad) with finite values.
    greens_hat_grid : np.ndarray
        Fourier-space Green's function sampled on the FFT wavenumber grid of
        the padded domain, real array of the same shape as rho_padded.
    Nz_out : int
        Number of physical vertical cells to return (1 <= Nz_out <= Nz_pad).
    G : float
        Gravitational constant (G > 0).

    Returns
    -------
    phi : np.ndarray
        Real potential of shape (Nx, Ny, Nz_out) on the physical cells.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(rho_padded).shape[:2] + (int(Nz_out),), dtype=float)  # placeholder
```

### Step 6

06_analytic_layer_potential

Goal
----
Evaluate the closed-form error-function reference potential of a truncated Gaussian single-mode layer with vacuum vertical boundaries.

```python
import math

import numpy as np

def analytic_layer_potential(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             kx: float, ky: float, H: float, Lz: float,
                             G: float = 1.0) -> np.ndarray:
    """Closed-form potential of a truncated Gaussian single-mode layer.

    Parameters
    ----------
    x : np.ndarray
        x coordinates, broadcastable against y and z.
    y : np.ndarray
        y coordinates, broadcastable against x and z.
    z : np.ndarray
        Vertical coordinates, broadcastable against x and y. Every value
        must lie inside the source slab, |z| <= Lz/2; this closed form is
        the interior solution and is not valid above or below the slab,
        where the potential instead decays exponentially.
    kx : float
        In-plane wavenumber of the density mode in x (finite).
    ky : float
        In-plane wavenumber of the density mode in y (finite; kx and ky
        must not both be zero).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    Lz : float
        Vertical extent of the slab hosting the density (Lz > 0).
    G : float
        Gravitational constant (G > 0).

    Returns
    -------
    phi : np.ndarray
        The analytic potential evaluated elementwise on the broadcast shape
        of (x, y, z), as a float array.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape, dtype=float)  # placeholder
```

### Step 7

07_relative_error_rms

Goal
----
Compute the RMS relative error between two potentials after shifting each so its minimum equals 1.

```python
import numpy as np

def relative_error_rms(phi_num: np.ndarray, phi_ref: np.ndarray) -> float:
    """RMS relative error between two potentials after min-to-1 shifting.

    Parameters
    ----------
    phi_num : np.ndarray
        Numerical potential, any shape with at least one element and only
        finite values.
    phi_ref : np.ndarray
        Reference potential with the same shape as phi_num and only finite
        values.

    Returns
    -------
    error_norm : float
        The root-mean-square of |(phi_num_shifted - phi_ref_shifted) /
        phi_ref_shifted|, where each field is shifted so its minimum is 1,
        as a native Python float.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```

### Step 8

08_run_benchmark_pipeline

Goal
----
Chain the sub-problem functions 01-07 end to end on the mixed-boundary benchmark and return the RMS relative-error norm.

```python
import numpy as np

def run_benchmark_pipeline(N: int, L: float, H: float, alpha: float = 1.1,
                           pad_factor: int = 4, G: float = 1.0,
                           Omega: float = 1.0, t: float = 0.0,
                           m_x: int = 1, m_y: int = 1) -> float:
    """Run the full spectral Poisson benchmark and return the error norm.

    Parameters
    ----------
    N : int
        Number of cells per dimension of the cubic box (N >= 4).
    L : float
        Box size in every dimension (L > 0).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    alpha : float
        Green's function truncation factor, half-width alpha * L (alpha > 1).
    pad_factor : int
        Vertical zero-padding factor (integer, pad_factor >= 2 * alpha).
    G : float
        Gravitational constant (G > 0).
    Omega : float
        Orbital frequency of the shearing box (finite real number).
    t : float
        Time at which the wave-vectors are evaluated (finite real number).
    m_x : int
        Integer mode number of the density in x, kx = 2*pi*m_x/L.
    m_y : int
        Integer mode number of the density in y, ky = 2*pi*m_y/L (m_x and
        m_y must not both be zero).

    Returns
    -------
    error_norm : float
        The RMS relative-error norm between the spectral and the analytic
        potential after min-to-1 shifting, as a native Python float.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 (``build_density_grid``, ``compute_shear_wavevectors``,
    ``greens_function_hat``, ``zero_pad_density``,
    ``spectral_convolution_potential``, ``analytic_layer_potential``,
    ``relative_error_rms``) and feed each returned value into the next, rather
    than reimplementing them. Include every import your implementation needs
    (for example ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```
