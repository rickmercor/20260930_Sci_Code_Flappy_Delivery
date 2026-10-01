# Mathematics-Computational_Mechanics-28

## Background

Sea ice covers a small fraction of the ocean but exerts a disproportionate influence on climate, regulating the exchange of heat, moisture and momentum between ocean and atmosphere and setting the polar albedo. Representing it in a climate model means representing a material that is neither a fluid nor an elastic solid: over hundreds of kilometres it behaves as a two-dimensional continuum that resists convergence far more strongly than it resists shear, fails on a yield surface once the internal stress is large enough, and is otherwise so viscous as to be nearly rigid. The deformation that results is not smooth. It concentrates into narrow bands, the leads and pressure ridges that satellite imagery shows as a fabric of intersecting lines, and because those bands govern where the ocean is exposed to the atmosphere, a discretisation that smears them out gets the thermodynamics wrong as well as the mechanics. What makes them hard to compute is that the constitutive law is strongly nonlinear and nearly degenerate in the rigid limit, so a direct implicit solve is impractical and the scheme has to relax towards the answer instead.

## Problem

Sea ice is modelled in climate and forecasting systems as a two-dimensional continuum whose internal stresses follow a nonlinear viscous-plastic rheology: the ice yields on an elliptical curve in principal stress space, deforms plastically once the stress reaches that curve, and behaves as an extremely viscous, nearly rigid medium otherwise. The narrow bands of intense deformation that result, leads and pressure ridges, collectively linear kinematic features, are what the discretisation has to resolve, and their structure is highly sensitive to how the velocity is discretised. The source discretises the momentum balance and the constitutive law as a first-order mixed system with a Local Discontinuous Galerkin method, and solves the resulting nonlinear system by relaxing an artificial elastic stress in pseudo-time until it converges to the viscous-plastic solution. Your job is to implement that scheme exactly as the source specifies it and audit the deformation field it produces on the fixed configurations below. The construction rests on a handful of conventions that the source states and that a plausible alternative reading would get wrong; they are not reproduced here, and recovering them from the literature is the substance of the problem. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

Work in SI units on the square domain of side 512 km with the source's own parameter values: ice density 900, air density 1.3 and ocean density 1026 kg per cubic metre, air and ocean drag coefficients 1.2e-3 and 5.5e-3, Coriolis parameter 1.46e-4 per second, ice strength parameter 27.5e3 newtons per square metre, ice concentration parameter 20, yield-curve eccentricity 2, and viscous regularisation threshold 2e-9 per second. Each configuration is a uniform nx by nx quadrilateral mesh with an integer field index v and a wind speed, the three being (nx, v, wind) = (3, 1, 8.0), (3, 2, 11.0) and (4, 1, 9.5). Element (i, j) carries ice thickness 0.3 + 0.05 sin((3+v) pi xc) cos((2+v) pi yc) metres and concentration 0.90 + 0.05 sin((2+v) pi (xc+yc)), with xc and yc the element-centre coordinates divided by the domain side.

Fix the following numerical conventions so the results are reproducible. Velocities are bilinear on each element, with the four nodal values ordered anticlockwise from the lower-left corner; stresses and strain rates are constant on each element. Integrate faces with two-point Gauss-Legendre quadrature and elements with the tensor-product two-by-two rule. The forcing at each velocity node is the source's external force: the concentration-weighted sum of an air drag and an ocean drag, each quadratic in the velocity of the driving fluid relative to the ice, plus a Coriolis term proportional to the ice density, the local thickness and the Coriolis parameter that acts on the ocean velocity minus the ice velocity; use each element's thickness and concentration at all four of its nodes, with air velocity wind*[sin(pi x) cos(pi y) + 0.8 (x - 0.5), -cos(pi x) sin(pi y) + 0.8 (y - 0.5)] and ocean velocity 0.01*[2y - 1, -2x + 1] at coordinates normalised by the domain side. Start from rest with zero stress and run exactly 800 sub-iterations with both stabilisation parameters equal to 500, physical time step 600 seconds, and flux parameters a = 0.25 and b = 3.0. The velocity entering the traction fluxes of the stress divergence, including the velocity-jump penalty, is taken at the previous sub-iterate, so every velocity update is an element-local mass-matrix solve. Solve every local system directly; no iterative tolerance enters the reported numbers.

Implement nine functions. ice_strength(H, A, Pstar, C) returns the source's empirical ice strength. effective_deformation(eps, dmin, e) returns the source's effective deformation rate. vp_viscosities(P, Delta, e) returns the bulk and shear viscosities, bulk first. vp_stress(eps, P, dmin, e) returns the source's viscous-plastic stress. ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, on_boundary) returns the velocity flux and the traction flux on one face. ldg_strain(U, S, nx, h, a, b) returns the element-wise strain rate. ldg_divergence(U, S, nx, h, a, b) returns the weak stress divergence tested against every velocity basis function. mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b) advances the pseudo-time system by one sub-iteration. sea_ice_audit(wind_scale), the final step, must be assembled by calling the earlier functions: it runs the three configurations at the declared wind multiplied by wind_scale and returns a float64 array (3, 5) whose rows are [largest shear deformation, mean shear deformation, the largest absolute value taken by any scalar velocity component over all nodes, the ratio of the final stress residual to the immediately preceding one, summed shear deformation], with the three deformation entries expressed in units of 1e-6 per second and the residual the largest absolute difference between the relaxed stress and its target.

Evaluate sea_ice_audit with wind_scale = 1.0. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. In your reasoning report, for each configuration, give the largest and summed shear deformation, the largest velocity component and the stress-residual ratio, then the sum of the three summed shear deformations, which is the final answer. Give every reported number to at least six significant figures. State what the residual ratios say about the pseudo-time relaxation, how they relate to the first stabilisation parameter, and why they fall short of the contraction that parameter alone would predict, and state which viscosity dominates the other and why. Also state, in ONE sentence each, the source conventions your numbers actually depend on, covering the ice strength relation, the effective deformation rate, the two viscosities, the stress closure, the numerical fluxes, which state each term of the pseudo-time update is evaluated at, and what the source proves about where that relaxation converges; for each, name the choice the source makes and say what an obvious alternative reading would have been. Report the required values as a compact list of numbers and each convention as a single sentence, not as a derivation. As the final answer, report the sum over the three configurations of the fifth column to six significant figures.

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

ice_strength

Goal
----
Return the source's empirical ice strength from the ice thickness and concentration. It is proportional to the thickness and depends exponentially on how far the concentration falls short of full cover, with the two material constants supplied as arguments. Recover the exact relation from the paper. Raise ValueError on a negative thickness, on a concentration outside [0, 1], or on a non-positive material constant.

```python
import numpy as np


def ice_strength(H, A, Pstar, C):
    """H: ice thickness; A: ice concentration in [0, 1]; Pstar, C: material
    constants. Returns a float64 array shaped like H."""
    return np.zeros_like(np.asarray(H, dtype=np.float64))
```

### Step 2

effective_deformation

Goal
----
Return the source's effective deformation rate for a stack of two-dimensional strain-rate tensors. It combines a regularisation floor, a term built from the TRACE-FREE part of the strain rate weighted by the yield-curve eccentricity, and a term built from the trace, all under a square root. The numerical factor on the trace-free term, the power of the eccentricity that multiplies it, and the fact that this term uses the trace-free part rather than the full tensor are the source's conventions; recover them from the paper. Note that the medium is two-dimensional, so the trace-free part is formed accordingly. Raise ValueError unless the input is a stack of 2x2 tensors and the floor and eccentricity are positive.

```python
import numpy as np


def effective_deformation(eps, dmin, e):
    """eps: (..., 2, 2) strain-rate tensors; dmin: regularisation floor;
    e: yield-curve eccentricity. Returns a float64 array of shape eps.shape[:-2]."""
    return np.zeros(np.asarray(eps, dtype=np.float64).shape[:-2])
```

### Step 3

vp_viscosities

Goal
----
Return the source's two nonlinear viscosities, stacked with the bulk one first and the shear one second. The bulk viscosity is the ice strength divided by the effective deformation rate times a numerical factor, and the shear viscosity follows from it through the yield-curve eccentricity. Both the numerical factor and the power of the eccentricity, including whether it multiplies or divides, are the source's conventions; recover them from the paper. Raise ValueError on a non-positive effective deformation rate or eccentricity.

```python
import numpy as np


def vp_viscosities(P, Delta, e):
    """P: ice strength; Delta: effective deformation rate; e: eccentricity.
    Returns (2, ...) float64 with the bulk viscosity first."""
    return np.zeros((2,) + np.shape(np.asarray(P, dtype=np.float64)))
```

### Step 4

vp_stress

Goal
----
Return the source's viscous-plastic stress for a stack of strain-rate tensors. It is built from the two viscosities of the previous sub-problem acting on the strain rate and on its trace, together with an isotropic pressure contribution proportional to the ice strength. Which viscosity multiplies the strain rate directly, which combination multiplies the trace, and the numerical factor carried by the pressure term are the source's conventions; recover them from the paper. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def vp_stress(eps, P, dmin, e):
    """eps: (..., 2, 2) strain rates; P: ice strength; dmin, e as before.
    Returns a float64 array of the same shape as eps."""
    return np.zeros_like(np.asarray(eps, dtype=np.float64))
```

### Step 5

ldg_numerical_fluxes

Goal
----
Return the source's two Local Discontinuous Galerkin numerical fluxes on one face, the velocity flux first and the traction flux second. On an interior face each is built from the average across the face plus a term in the jump, and the traction flux carries an additional penalty on the velocity jump. The sign that the flux parameter carries in each of the two fluxes is the source's convention and the two are NOT the same; recover it from the paper. Take the jump of a quantity as its value in the element owning the given outward normal minus its value in the neighbour. On a boundary face the no-slip condition fixes the velocity flux and the traction flux is the element traction corrected by a term in the velocity whose coefficient the source states. Raise ValueError unless the flux parameters satisfy the source's admissible ranges.

```python
import numpy as np


def ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, on_boundary):
    """uL, uR: (2,) velocities either side; sL, sR: (2, 2) stresses either side;
    n: (2,) outward normal of the left element; a, b: flux parameters;
    on_boundary: True on a boundary face, where uR and sR are ignored.
    Returns (2, 2) float64."""
    return np.zeros((2, 2))
```

### Step 6

ldg_strain

Goal
----
Recover the element-wise strain rate on a uniform nx by nx quadrilateral mesh of the square domain from the source's mixed formulation. Because the stress and strain spaces are element-constant, the volume term of that equation vanishes and each element's strain rate is determined entirely by the velocity fluxes on its four faces: integrate the symmetric part of the outer product of the velocity flux with the outward normal over the element boundary and divide by the element area. Velocities are bilinear on each element with the four nodal values ordered anticlockwise from the lower-left corner; integrate each face with two-point Gauss-Legendre quadrature. Raise ValueError unless nx and the element size are positive. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def ldg_strain(U, S, nx, h, a, b):
    """U: (nx, nx, 4, 2) nodal velocities; S: (nx, nx, 2, 2) element stresses;
    nx: elements per side; h: element size; a, b: flux parameters.
    Returns (nx, nx, 2, 2) float64."""
    return np.zeros((nx, nx, 2, 2))
```

### Step 7

ldg_divergence

Goal
----
Return the weak stress-divergence term of the source's momentum equation, tested against every velocity basis function on every element. It combines a volume contribution pairing the element stress with the gradient of the test function and a boundary contribution pairing the traction flux with the test function on each face. The sign each contribution carries follows from integrating the divergence by parts once; recover the arrangement from the paper. Use two-point Gauss-Legendre quadrature on the faces and the tensor-product two-by-two rule in the element. Raise ValueError unless nx and the element size are positive. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def ldg_divergence(U, S, nx, h, a, b):
    """Arguments as for the strain sub-problem. Returns (nx, nx, 4, 2) float64,
    one two-component value per velocity node per element."""
    return np.zeros((nx, nx, 4, 2))
```

### Step 8

mevp_subcycle

Goal
----
Advance the source's modified elastic-viscous-plastic system by ONE sub-iteration and return the new nodal velocity, the new element stress and the viscous-plastic stress target that the stress relaxes towards, flattened and concatenated in that order. The stress is relaxed towards the target evaluated at the CURRENT velocity, with the relaxation weight set by the first stabilisation parameter. The velocity is then updated from the momentum equation, which carries two separate inertia terms, one referred to the previous sub-iteration and weighted by the second stabilisation parameter and one referred to the previous physical time level, together with the stress divergence evaluated at the NEW stress and the given forcing. The exact weights, and which state the stress target and the stress divergence are evaluated at, are the source's conventions; recover them from the paper. The velocity that enters the traction fluxes of that divergence, including the velocity-jump penalty, is taken at the PREVIOUS sub-iterate, so every velocity update is an element-local mass-matrix solve as the problem statement's direct local solves require. The stress closure is evaluated with the source's viscous regularisation threshold 2e-9 per second and yield-curve eccentricity 2. Raise ValueError on a non-positive stabilisation parameter or time step. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b):
    """U, S: current velocity and stress; Un: velocity at the previous physical
    time level; P: (nx, nx) ice strength; H: (nx, nx) thickness;
    Fv: (nx, nx, 4, 2) forcing; alpha, beta: stabilisation parameters;
    dt: physical time step. Returns a flat float64 array."""
    return np.zeros(nx * nx * 16)
```

### Step 9

sea_ice_audit

Goal
----
Run the source's scheme on the three declared configurations and return one row each. A configuration is (nx, field index v, wind speed): (3, 1, 8.0), (3, 2, 11.0) and (4, 1, 9.5). The domain is the square of side 512 km. Element (i, j) carries thickness 0.3 + 0.05 sin((3+v) pi xc) cos((2+v) pi yc) and concentration 0.90 + 0.05 sin((2+v) pi (xc+yc)) with xc, yc the element-centre coordinates divided by the domain side. The forcing at each velocity node is the source's external force: the concentration-weighted sum of an air drag and an ocean drag, each quadratic in the velocity of the driving fluid RELATIVE to the ice, plus a Coriolis term proportional to the ice density, the local thickness and the Coriolis parameter that acts on the ocean velocity minus the ice velocity; use each element's thickness and concentration at all four of its nodes, with air velocity wind*[sin(pi x) cos(pi y) + 0.8 (x - 0.5), -cos(pi x) sin(pi y) + 0.8 (y - 0.5)] and ocean velocity 0.01*[2y - 1, -2x + 1] at normalised coordinates and the densities and drag coefficients of the source's parameter table. Start from rest with zero stress and run exactly 800 sub-iterations at alpha = beta = 500, physical time step 600 s, and flux parameters a = 0.25 and b = 3.0, with the wind multiplied by wind_scale. Each row is [largest shear deformation, mean shear deformation, the largest absolute value taken by any scalar velocity component over all nodes, the ratio of the final stress residual to the immediately preceding one, summed shear deformation], where the shear deformation is the source's own diagnostic, the three deformation entries are expressed in units of 1e-6 per second, and the residual is the largest absolute difference between the relaxed stress and its target. Raise ValueError unless wind_scale is positive and finite. Assemble by calling the earlier sub-problem functions, every one of them.

```python
import numpy as np


def sea_ice_audit(wind_scale):
    """wind_scale: positive multiplier on the declared wind speed.
    Returns (3, 5) float64 with one row per configuration."""
    return np.zeros((3, 5))
```
