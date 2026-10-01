# Physics-Astrophysics-33

## Background

This reduced two-dimensional transverse transport experiment isolates coupled density and thermal-energy fluxes relative to magnetic surfaces. External controls maintain a nonseparable magnetic-field, outward-flow and turbulence configuration. Its target is the optimal induced-pressure response to a small initially pressure-balanced density disturbance about a normalized zero-flux state, not a global solar-wind solution.

## Problem

Consider the linear pressure susceptibility of a doubly periodic transverse coronal-plasma layer governed by slaved strong-turbulence transport relative to magnetic flux surfaces. This is a flux-only two-scalar relaxation experiment: external controls hold the magnetic field, parallel flow, dominant outward Elsasser amplitude and parallel correlation scale fixed, while all nonflux sources, parallel transport, momentum evolution and magnetic-field evolution are excluded.

Use length unit L0, velocity unit V0, density unit rho0 and pressure unit rho0 V0 squared, with normalized magnetic strength \(b=B/(V0\sqrt{4\pi\,\mathrm{rho0}})\), gamma = 5/3 and critical-balance parameter chi_A = 1. The dimensionless periodic coordinates are x,y in [0, 2 pi), and the fixed profiles are

\[
\begin{aligned}
b(x,y)&=\exp[0.8\cos x+0.65\sin y+0.4\cos(x+2y+0.3)],\\
u(x,y)&=0.9+0.3\sin(x+0.4)+0.2\cos(2y)+0.15\sin(x-y+0.2),\\
z_+(x,y)&=0.04\exp[0.15\sin(2x-y-0.3)],\\
\ell_\parallel(x,y)&=0.8[1+0.12\cos(x+y+0.2)].
\end{aligned}
\]

The reference state is the positive simultaneous zero-mass-flux and zero-thermal-flux branch, with dimensionless density and pressure each having nodal arithmetic mean one. The discretized experiment has 21 equally spaced x nodes and 23 equally spaced y nodes, periodic tensor-product trigonometric collocation of the continuum linearized fluxes, pointwise coefficient multiplication and no dealiasing. At time zero the infinitesimal density disturbance has zero nodal mean and the pressure disturbance is identically zero. What is the supremum, over nonzero admissible initial density disturbances, of the induced pressure RMS at time 1200 L0/V0 divided by the initial density RMS, with both disturbances expressed in the fixed units above and RMS defined by the unweighted nodal arithmetic mean?

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

stationary_density

Goal
----
Determine the positive zero-mass-flux state of a transverse, periodic coronal

transport layer. Field strength b and parallel velocity u are externally fixed.

Units absorb the reference density and magnetic permeability, so v_A=b/sqrt(r).

The slaved turbulent mass flux divided by its positive diffusivity is

-gradient(r) + 2\*r\*(gradient(ln b) + M_A\*gradient(ln u)), with M_A=u/v_A.

The spatial mean of r is prescribed; M_A is a local state variable.

```python
import numpy as np

def stationary_density(b: np.ndarray, u: np.ndarray, mean_density: float) -> np.ndarray:
    """Return the normalized zero-flux density and its characteristic speed.

    b and u are same-length nodal vectors of normalized field strength and
    parallel velocity. mean_density fixes the arithmetic mean of the density.
    Return an (N+1,) float array: density samples followed by the constant
    characteristic speed, in the units and branch specified above.

    Raise ValueError for mismatched, empty, non-vector or nonfinite inputs,
    nonpositive b/mean_density, negative u, or a nonfinite derived state.
    """
    return out
```

### Step 2

stationary_pressure

Goal
----
Normalize the thermal zero-flux state on the density equilibrium of step 1.

Thermal energy is p/(gamma-1). Its turbulent flux divided by diffusivity is

-gradient(E_th) + 2\*gamma\*E_th\*(gradient(ln b) + M_A\*gradient(ln u)).

The corresponding mass flux is the one in step 1 and vanishes at every point.

Density and pressure have prescribed, separately conserved nodal means.

```python
import numpy as np

def stationary_pressure(rho: np.ndarray, mean_pressure: float, gamma: float) -> np.ndarray:
    """Return the pressure of the simultaneous zero-flux equilibrium.

    rho is the positive nodal density vector, mean_pressure is the prescribed
    arithmetic pressure mean, and gamma is the ratio of specific heats.
    Return a same-shaped float pressure vector in the stated normalized units.

    Raise ValueError for invalid shapes, nonfinite values, nonpositive density
    or mean pressure, gamma<=1, or nonfinite derived output.
    """
    return out
```

### Step 3

transport_tangent

Goal
----
Obtain the local tangent fluxes about the simultaneous zero-flux equilibrium.

The geometry, outward flow, dominant Elsasser amplitude z and parallel scale

ell are externally maintained, while density r and pressure p may respond.

Use v_A=b/sqrt(r), M_A=u/v_A, eta=z\*l_perp/4, and critical balance

chi=z\*ell/(v_A\*l_perp). Mass and thermal fluxes are those in steps 1 and 2.

```python
import numpy as np

def transport_tangent(background: np.ndarray, rho: np.ndarray, pressure: np.ndarray,
                      gamma: float, chi: float) -> np.ndarray:
    """Return the local scalar diffusivity and three transverse tangent vectors.

    background is (N,8) with columns [b,u,z,ell,k_x,k_y,u_x,u_y]; rho and
    pressure are (N,) equilibrium vectors. gamma is the heat-capacity ratio
    and chi is the critical-balance parameter. Return an (N,7) float array
    with columns [eta,a_x,a_y,h_x,h_y,c_x,c_y] defined above.

    Raise ValueError for wrong shapes, nonfinite inputs, nonpositive b/ell/rho/
    pressure/chi, negative u/z, gamma<=1, or a nonfinite derived output.
    """
    return out
```

### Step 4

flux_generator

Goal
----
Construct the two-dimensional density-pressure evolution generator from the

tangent fluxes of step 3. Both fields obey their flux-only conservative balance

in the magnetic-surface frame; no source or parallel transport is retained.

```python
import numpy as np

def flux_generator(coefficients: np.ndarray, shape: tuple, periods: tuple) -> np.ndarray:
    """Return the conservative tensor-grid generator.

    coefficients is (nx*ny,7), shape contains the two odd integer node counts,
    and periods contains the two positive lengths. Return a real square
    (2*nx*ny,2*nx*ny) matrix in density-then-pressure order.
    Raise ValueError for wrong shapes, nonfinite data, invalid node counts,
    negative eta or nonpositive periods.
    """
    return out
```

### Step 5

pressure_response

Goal
----
Propagate an initially pressure-balanced density perturbation under the

autonomous flux generator of step 4. Initial pressure perturbation is zero;

the density perturbation must have zero arithmetic mean. The dynamics conserve

both means. Output is the induced pressure at the requested elapsed time.

```python
import numpy as np

def pressure_response(generator: np.ndarray, time: float) -> np.ndarray:
    """Return the finite-time density-to-pressure response, with both means removed.

    generator is the (2*N,2*N) density-then-pressure evolution matrix and
    time is the nonnegative elapsed time in consistent units. Return an
    (N,N) real matrix from initial density to induced pressure, applying
    the stated zero-mean restrictions and zero initial pressure.

    Raise ValueError for non-square or odd-sized matrices, fewer than four
    rows, nonfinite inputs, negative time or a nonfinite propagated output.
    """
    return out
```

### Step 6

optimal_response

Goal
----
Find the largest possible RMS response of a real linear density-to-pressure

map. Its columns and rows use the same uniform spatial grid, and its null

constant mode has already been removed. The norm is unweighted spatial RMS,

not a norm weighted by the nonuniform equilibrium density or pressure.

```python
import numpy as np

def optimal_response(response: np.ndarray) -> float:
    """Return the optimal RMS amplitude ratio.

    response is a nonempty finite real square matrix on a uniform grid.
    Return one nonnegative native Python float, using the RMS amplitude
    convention above. Raise ValueError for an invalid input matrix.
    """
    return out
```

### Step 7

pressure_susceptibility

Goal
----
Final orchestrator. Combine steps 1-6 for the optimal induced-pressure RMS

from an initially pressure-balanced, zero-mean density disturbance in a

two-dimensional periodic transverse layer. Each field is normalized by its

prescribed spatial mean. Only transverse conservative turbulent fluxes evolve.

```python
import numpy as np

def pressure_susceptibility(params: dict | None = None) -> float:
    """Return the complete two-dimensional equilibrium/response scalar.

    params is None or a dictionary overriding the named benchmark inputs
    above. Return one native Python float: induced-pressure RMS divided by
    initial-density RMS, each expressed relative to its prescribed mean.
    Raise ValueError for unknown keys, non-dicts, nonscalar/nonfinite values,
    nonintegral/even node counts or counts<3, negative time/flow_scale/z_scale,
    nonpositive periods/lpar_scale/rho_mean/p_mean/chi, gamma<=1, or a
    nonfinite derived state. Field amplitudes may be finite signed reals.
    """
    return out
```
