# Convergence-qualified meshfree free-electron propagation

## Background

The benchmark is a new dimensionless instance of the paper's two-dimensional time-dependent Schrödinger model. The localized traveling interference potential above is supplied analytically so that the meshfree and grid methods see exactly the same Hamiltonian. It is not a result table from the paper.

The meshfree calculation must use the paper's full phase-space Gaussian representation, packet-local harmonic dynamics, action phase, complex normalization, and coherent superposition. The grid calculation is an independent Strang split-step reference. The candidate tests mirror the paper's stated practice of matching time-step self-convergence and increasing the number of Gaussians until the error saturates.

All arrays retain the displayed order. Grid quadrature uses `dx*dy`; Fourier wave numbers are `2*pi*fftfreq`; L2 errors are relative to the second argument; the TGWP is globally phase-aligned only for the separately reported aligned diagnostic and never before the required complex L2 error. Every maximum and exact tie uses the first occurrence, no intermediate result is rounded, and only the final selected score is rounded.

## Problem

For the new nondimensional two-dimensional electron-light instance below, reproduce the primary paper's meshfree thawed-Gaussian wave-packet (TGWP) propagation and its independent split-step grid reference, then return the quality-adjusted speedup of the selected TGWP candidate after applying every stated accuracy and convergence limit. Use the Hamiltonian `H=-1/2 nabla^2+V(t,x,y)`, the normalized initial Gaussian with `q0=[-3.5,0.2]`, `p0=[2.2,0.1]`, and `Gamma0=diag(0.7,1.2)`, and representing-packet width `Gamma=diag(1.4,2.4)`, with `hbar=m=1` and final time `T=2.4`. The prescribed potential is the localized optical-interference term `V=A exp(-0.5[((x-xc)/wx)^2+(y/wy)^2]) cos(kx*x+ky*y-omega*t+phase)` with ordered parameters `[A,xc,wx,wy,kx,ky,omega,phase]`, base values `[0,0,1.3,1.0,1.6,0.65,3.45,0.3]`, and the two interaction amplitudes `A=[0.35,0.65]`.

Use the periodic grid `x=linspace(-8,8,48,endpoint=False)`, `y=linspace(-6,6,36,endpoint=False)`, a 192-step Strang reference, scrambled Sobol seed `20260902`, and ordered TGWP candidates `(log2 packet count, step count)=[(4,32),(5,32),(6,32),(6,48),(6,64)]`. For every candidate and amplitude evaluate the relative complex-wavefunction L2 error, relative modulus L2 error, TGWP norm error, time-refinement error against twice as many TGWP steps, and packet-basis saturation error against twice as many Sobol packets; take the maximum over the two amplitudes and require these five maxima to be at most `[0.55,0.48,0.19,0.001,0.40]` in that order. Define the FFT cost proxy as `192*Ngrid*log2(Ngrid)`, the TGWP cost proxy as `Npackets*Nsteps`, `speedup=FFT_cost/TGWP_cost`, and `quality_adjusted_speedup=speedup/(1+worst_L2+worst_abs_L2+worst_time_refinement+worst_basis_refinement)`.

Advance the packet centers and Hagedorn matrices by kick-drift-kick, with the gradient and Hessian evaluated at the interval endpoints. Filter infeasible candidates before maximizing the unrounded quality-adjusted speedup, preserve the supplied order for exact ties, and round only the selected score to six decimals. In `<reasoning>`, report the selected packet count and step count, its five error/refinement maxima, its speedup proxy, and the identity and value of its largest feasibility ratio; for each of the time and basis gates, identify the otherwise-feasible candidate it rejects and give the failing error value. Also report the selected candidate's complex L2 error at `A=0.35`, its worst normalized-overlap loss against the reference, and the quality-adjusted speedup of the runner-up feasible candidate. Justify the scientific basis of each stage of your pipeline. Relative reporting tolerances are `1e-4` for the final score, complex and modulus L2 errors, basis-refinement errors, norm error, governing ratio, overlap loss and runner-up score, `5e-2` for time-refinement errors, and `1e-6` for the cost proxy; packet/step counts and feasibility limits remain exact.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

phase_space_sobol_packets

Goal
----
Construct the paper's weighted full-phase-space Gaussian packet ensemble.

```python
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

def phase_space_sobol_packets(log2_count, seed, q0, p0,
                              initial_gamma_diag, basis_gamma_diag, hbar=1.0):
    """Return a float array of shape (2**log2_count,6).

    Columns are [qx,qy,px,py,coefficient_real,coefficient_imag].

    Raises
    ------
    ValueError
        If dimensions, widths, hbar, seed, or log2_count are invalid.
    """
    return np.empty((2 ** int(log2_count), 6), dtype=float)
```

### Step 2

interference_potential_derivatives

Goal
----
Evaluate the common time-dependent optical potential and its spatial derivatives.

```python
import numpy as np

def interference_potential_derivatives(points, time, potential_parameters):
    """Return a float array of shape (N,6).

    Columns are [V,dV_dx,dV_dy,d2V_dxx,d2V_dxy,d2V_dyy].

    Raises
    ------
    ValueError
        If inputs are non-finite, points are not (N,2), or widths are non-positive.
    """
    return np.empty((len(points), 6), dtype=float)
```

### Step 3

stormer_verlet_centers_actions

Goal
----
Propagate packet centers and classical actions with the paper's second-order scheme.

```python
import numpy as np

def stormer_verlet_centers_actions(packet_rows, final_time, step_count,
                                    potential_parameters, mass=1.0, hbar=1.0):
    """Return a float array of shape (step_count+1,N,5).

    The last axis is [qx,qy,px,py,action] at every stored time.

    Raises
    ------
    ValueError
        If packet shape, time, step count, mass, hbar, or potential is invalid.
    """
    return np.empty((int(step_count) + 1, len(packet_rows), 5), dtype=float)
```

### Step 4

hagedorn_width_evolution

Goal
----
Propagate matrix-valued Hagedorn widths and continuous complex normalizations.

```python
import numpy as np

def hagedorn_width_evolution(center_trajectory, final_time, potential_parameters,
                              basis_gamma_diag, mass=1.0, hbar=1.0):
    """Return a float array of shape (N,18).

    Columns are row-major Re(Q)[4], Im(Q)[4], Re(P)[4], Im(P)[4],
    Re(gamma), Im(gamma).

    Raises
    ------
    ValueError
        If the trajectory, widths, time, mass, hbar, or potential is invalid.
    """
    return np.empty((center_trajectory.shape[1], 18), dtype=float)
```

### Step 5

coherent_tgwp_reconstruction

Goal
----
Reconstruct the complex meshfree TGWP wavefunction by coherent superposition.

```python
import numpy as np

def coherent_tgwp_reconstruction(grid_x, grid_y, packet_rows,
                                  final_centers_actions, final_width_rows, hbar=1.0):
    """Return a float array of shape (len(grid_y),len(grid_x),2).

    The last axis is [real(psi),imag(psi)].

    Raises
    ------
    ValueError
        If grids or packet, center, width, or hbar contracts are invalid.
    """
    return np.empty((len(grid_y), len(grid_x), 2), dtype=float)
```

### Step 6

strang_split_step_reference

Goal
----
Propagate the same state with the independent periodic-grid Strang reference.

```python
import numpy as np

def strang_split_step_reference(grid_x, grid_y, q0, p0, initial_gamma_diag,
                                 final_time, step_count, potential_parameters,
                                 mass=1.0, hbar=1.0):
    """Return a float array of shape (len(grid_y),len(grid_x),2).

    The last axis is [real(psi),imag(psi)].

    Raises
    ------
    ValueError
        If the uniform grids, state, time, steps, constants, or potential are invalid.
    """
    return np.empty((len(grid_y), len(grid_x), 2), dtype=float)
```

### Step 7

wavefunction_error_diagnostics

Goal
----
Measure complex, modulus, overlap, norm, and phase-aligned wavefunction errors.

```python
import numpy as np

def wavefunction_error_diagnostics(tgwp_field, reference_field, dx, dy):
    """Return a length-6 float array.

    Order is [TGWP_norm,reference_norm,relative_L2,relative_abs_L2,
    overlap_magnitude,global_phase_aligned_relative_L2].

    Raises
    ------
    ValueError
        If packed fields, spacings, shapes, or norms are invalid.
    """
    return np.zeros(6, dtype=float)
```

### Step 8

tgwp_convergence_table

Goal
----
Apply grid accuracy, time refinement, and packet-basis saturation tests to all candidates.

```python
import numpy as np

def tgwp_convergence_table(log2_counts, step_counts, amplitudes,
                           grid_x, grid_y, q0, p0,
                           initial_gamma_diag, basis_gamma_diag,
                           final_time, potential_parameters, grid_steps,
                           sobol_seed, feasibility_limits,
                           mass=1.0, hbar=1.0):
    """Return a float array of shape (number_of_candidates,12).

    Columns follow the exact order stated in the Step scientific background.

    Raises
    ------
    ValueError
        If candidate, grid, physical, or feasibility inputs violate their domains.
    """
    return np.empty((len(log2_counts), 12), dtype=float)
```

### Step 9

select_meshfree_configuration

Goal
----
Select the convergence-qualified TGWP configuration and return its adjusted speedup.

```python
def select_meshfree_configuration(log2_counts=(4,5,6,6,6),
                                  step_counts=(32,32,32,48,64),
                                  amplitudes=(.35,.65),
                                  grid_x=None, grid_y=None,
                                  q0=(-3.5,.2), p0=(2.2,.1),
                                  initial_gamma_diag=(.7,1.2),
                                  basis_gamma_diag=(1.4,2.4),
                                  final_time=2.4,
                                  potential_parameters=(0.,0.,1.3,1.,1.6,.65,3.45,.3),
                                  grid_steps=192, sobol_seed=20260902,
                                  feasibility_limits=(.55,.48,.19,.001,.40),
                                  rounding_digits=6, mass=1.0, hbar=1.0):
    """Return the selected quality-adjusted speedup as one float scalar.

    Passing no arguments evaluates the benchmark configuration in the Problem
    statement; ``None`` for either grid generates its stated periodic grid.

    Raises
    ------
    ValueError
        If no candidate is feasible or any delegated contract is invalid.
    """
    return 0.0
```
