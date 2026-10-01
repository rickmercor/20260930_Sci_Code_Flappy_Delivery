# Mathematics-Computational_Mechanics-18

## Background

Dynamic crack propagation couples an evolving near-tip singularity to elastic waves radiating from the moving tip. Two ingredients separate a dynamic-fracture computation from a static one. First, the crack-tip energy release rate must be measured by a J-integral whose integrand includes the kinetic-energy density and an inertia contribution, not the strain-energy density alone; second, the conversion of that energy release rate to a mode-I stress intensity factor carries a velocity-dependent universal function that diverges as the crack speed approaches the Rayleigh wave speed, so the quasi-static conversion is only valid in the zero-speed limit.

The hybrid s-version isogeometric strategy discretises the field by superposing a fine local Lagrange mesh, which resolves the near-tip region and the crack path, on a coarse global B-spline (isogeometric) patch that carries the smooth far field. The two discretisations are tied by a global-local coupling stiffness; because the global B-spline basis is smooth across its knot spans, that coupling integrates with a single standard Gauss rule even where a local element crosses a global knot line, which is the property a Lagrange global basis would lack. The J-integral is evaluated in its equivalent-domain form using a weight function that is one on an inner region around the tip and zero on an outer ring, so the contour integral becomes an area integral over a band of local elements.

## Problem

Simulating a crack that runs at a finite fraction of the elastic wave speed needs two things a static fracture code does not: a near-tip field that resolves the moving singularity, and an energy-release measure that accounts for the kinetic energy the moving material carries. A recent hybrid s-version isogeometric strategy supplies both by superposing a fine local Lagrange field on a smooth global B-spline patch and extracting the dynamic stress intensity factor from a domain-form J-integral. Your job is to implement the source's formulation exactly as specified and audit it on the fixed configuration below. The load-bearing choices - the velocity-dependent factor that converts energy release to the dynamic stress intensity factor, the kinetic and inertia contributions to the dynamic J-integral, the B-spline global basis that enters the global-local coupling, and the J-integral weight-function radius - are the source's; recover them from the paper. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

The material is plane strain with Young's modulus E = 1.0, Poisson ratio nu = 0.3 and density rho = 1.0; the crack tip sits at the origin and runs in the +x direction, so x1 is the x axis. The global field is a quadratic (degree two) B-spline patch on the square [-1.5, 1.5]^2 with the open knot vector [-1.5, -1.5, -1.5, 0, 1.5, 1.5, 1.5] in each direction and identity geometry, so the parametric coordinate equals the physical coordinate and there are four basis functions per direction. The local field is a 3x3 mesh of bilinear Q4 elements on the node lines [-1.5, -0.5, 0.5, 1.5] (local element size h_L = 1.0). The prescribed displacement state is u = (0.30x + 0.12xy + 0.06y^2, 0.18y + 0.15xy + 0.09x^2). The prescribed velocity state is the field (0.20y - 0.10x, 0.16x + 0.05y) scaled by crack_velocity/0.4, and the prescribed acceleration state is minus 0.6 times that same scaled velocity field. The crack-tip speed crack_velocity is the single free parameter and must stay below the shear wave speed.

Evaluate the source's model at the given crack speed. Build the quadratic B-spline basis and its derivatives from the Cox-de Boor recursion; the two elastic wave speeds and the velocity-dependent universal factor A_I; the J-integral weight function, one within the source's integration radius of the tip and zero outside; the strain- and kinetic-energy densities; the global-local coupling stiffness, integrated element by element over the local mesh with the standard three-point Gauss rule per direction and the global basis sampled at the physical Gauss location; the dynamic and quasi-static J-integrals over the local mesh; and the dynamic and quasi-static mode-I stress intensity factors from them. Collect the run into the audit vector [A_I, V1, V2, J_dynamic, J_static, K_dynamic, K_static, coupling_norm, q_count, W_ref, K_ref, partition], where V1 and V2 are the dilatational and shear wave speeds, J_dynamic and J_static are the magnitudes of the dynamic and quasi-static J-integrals, K_dynamic and K_static are the corresponding stress intensity factors, coupling_norm is the Frobenius norm of the coupling stiffness, q_count is the number of local element-nodes inside the J-domain, W_ref and K_ref are the reference energy densities, and partition is the sum of the global basis values at parameter 0.3.

Evaluate the audit at crack_velocity = 0.4. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. As the final answer, report the dynamic mode-I stress intensity factor K_dynamic (the sixth audit entry) at crack_velocity = 0.4 to six significant figures.

In your reasoning report numerically the velocity factor A_I, the dynamic and quasi-static J-integral magnitudes, both mode-I stress intensity factors, the coupling-stiffness Frobenius norm, and how A_I and the dynamic stress intensity factor change between crack speeds 0.3 and 0.5. Do not enumerate the coupling-matrix entries, the per-Gauss-point fields, or the basis-function tables.

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

bspline_basis

Goal
----
Return every quadratic (degree p) B-spline basis function and its first parametric derivative at a single parameter value from an open knot vector, evaluated by the recursion the paper uses to build its global patch (the paper's Eqs. 34-36). Row 0 holds the n basis values N_{i,p}(xi) and row 1 holds dN_{i,p}/dxi, in ascending basis index. Recover the exact recursion and the derivative rule from the paper.

```python
import numpy as np


def bspline_basis(knots, p, xi):
    """knots: (m+1,) open knot vector; p: degree (>=1); xi: parameter in the knot range.
    Returns a float64 array (2, n) where n = len(knots)-p-1: row 0 the basis values
    N_{i,p}(xi) and row 1 the derivatives dN_{i,p}/dxi (paper Eqs. 34-36). Raises
    ValueError if len(knots) < p+2 or p < 1."""
    n = len(np.asarray(knots)) - p - 1
    return np.zeros((2, max(n, 0)))
```

### Step 2

wave_factor

Goal
----
Return the velocity-dependent factor A_I that converts the dynamic energy release rate to the mode-I dynamic stress intensity factor for a crack running at speed V, together with the two elastic wave speeds and the two Rayleigh factors it is built from (the paper's Eqs. 10-14). Plane strain, isotropic. Recover the exact universal function and the two wave-speed expressions from the paper; the quasi-static limit is a different, velocity-independent constant.

```python
import numpy as np


def wave_factor(E, nu, rho, V):
    """E: Young's modulus; nu: Poisson ratio; rho: density; V: crack-tip speed.
    Returns a float64 array [A_I, V1, V2, beta1, beta2] with the mode-I dynamic
    universal factor A_I and the dilatational/shear wave speeds (paper Eqs. 10-14).
    Raises ValueError unless 0 < V < shear wave speed."""
    return np.zeros(5)
```

### Step 3

q_weight

Goal
----
Return the nodal values of the weight (virtual crack extension) function used by the domain-form J-integral, for local nodes given by their position relative to the crack tip at the origin (the paper's Eqs. 21-23). A node takes the value one when it lies within the source's declared J-integration radius of the tip and zero outside. Recover the exact radius rule from the paper (it is a fixed multiple of the local element size).

```python
import numpy as np


def q_weight(node_xy, h_L):
    """node_xy: (N, 2) local node coordinates relative to the crack tip at the origin;
    h_L: local element size. Returns a float64 array (N,): 1.0 for a node inside the
    source's J-integration radius, else 0.0 (paper Eqs. 21-23). Raises ValueError on a
    non-positive or nonfinite h_L."""
    return np.zeros(np.asarray(node_xy).shape[0])
```

### Step 4

energy_densities

Goal
----
Return the strain-energy density and the kinetic-energy density at a point from the stress, strain, density and velocity there (the paper's Eqs. 18-19). Stress and strain are given in engineering Voigt order (xx, yy, xy) with the shear entry the engineering shear strain. These two densities are the ones that enter the dynamic J-integral integrand.

```python
import numpy as np


def energy_densities(sigma, eps, rho, vel):
    """sigma: (3,) Voigt stress (xx, yy, xy); eps: (3,) Voigt strain with engineering
    shear; rho: density; vel: (2,) velocity. Returns a float64 array [W, K] with the
    strain-energy and kinetic-energy densities (paper Eqs. 18-19)."""
    return np.zeros(2)
```

### Step 5

coupling_matrix

Goal
----
Assemble the coupling stiffness between the global B-spline patch and the overlaid local Lagrange (bilinear Q4) mesh in the s-version isogeometric method (the paper's Eq. 46): the integral over the local domain of the global strain-displacement operator transpose times the plane-strain elasticity matrix times the local strain-displacement operator. Integrate element by element over the local mesh with the standard three-point Gauss rule per direction; the global geometry is the identity map, so the global basis is evaluated at the physical Gauss location. Because the global B-spline basis is smooth across its knot spans, no sub-element partitioning is used. Recover the coupling operator and the plane-strain matrix from the paper. Equation 46 is an integral over the whole local domain, so the result is assembled and not a collection of per-element blocks: element contributions are scatter-added into the degrees of freedom of the local nodes they share, and a node used by several elements carries one pair of columns in total. Local nodes are numbered by first appearance while scanning the element list, each contributing its x degree of freedom before its y, so the operator is (2*nG, 2*nL) with nL the number of distinct local nodes.

```python
import numpy as np


def coupling_matrix(gknx, gkny, p, loc_elems, E, nu):
    """gknx, gkny: open knot vectors of the global quadratic B-spline patch (identity
    geometry); p: global degree; loc_elems: list of (4, 2) physical node arrays of the
    local Q4 elements; E, nu: plane-strain elastic constants. Returns a float64 array
    (2*nG, 2*nL) holding the ASSEMBLED global-local coupling stiffness of paper Eq. 46,
    integrated with the standard 3x3 Gauss rule per local element; nG is the number of
    global basis functions and nL the number of distinct local nodes. Element
    contributions are scatter-added into the degrees of freedom of the shared local
    nodes, so a node used by several elements carries one pair of columns and not one
    pair per element. Local nodes are numbered by first appearance while scanning
    loc_elems in order, each contributing its x degree of freedom before its y."""
    nG = (len(gknx) - p - 1) * (len(gkny) - p - 1)
    seen = []
    for xy in loc_elems:
        for x, y in np.asarray(xy, dtype=np.float64):
            k = (round(float(x), 9), round(float(y), 9))
            if k not in seen:
                seen.append(k)
    return np.zeros((2 * nG, 2 * len(seen)))
```

### Step 6

dynamic_j_integral

Goal
----
Evaluate the domain (equivalent-domain-integral) form of the crack-tip J-integral over the local mesh, given nodal displacement, velocity and acceleration fields and the nodal weight function, integrated element by element with the standard 3x3 Gauss rule (the paper's Eq. 17 for static=False, its quasi-static reduction Eq. 20 for static=True). The crack extends in the +x direction, so x1 is the x axis. In the dynamic form the crack-tip energy flux carries the kinetic-energy density inside the Eshelby term AND an explicit inertia term in acceleration and velocity gradient; the quasi-static form keeps only the strain-energy density and drops the inertia term. Stresses follow from the plane-strain elasticity matrix. Recover the exact integrand from the paper; omitting the kinetic and inertia contributions gives the wrong (quasi-static) value.

```python
import numpy as np


def dynamic_j_integral(loc_elems, u, ud, udd, qnod, E, nu, rho, static=False):
    """loc_elems: list of (4, 2) local Q4 node arrays; u, ud, udd: (n_elem, 4, 2) nodal
    displacement, velocity and acceleration; qnod: (n_elem, 4) nodal weight function; E,
    nu, rho: material constants; static: if True use the quasi-static integrand (paper
    Eq. 20), else the dynamic one (paper Eq. 17). Returns float: the J-integral over the
    local domain (x1 is the +x crack direction), 3x3 Gauss per element."""
    return 0.0
```

### Step 7

dsif

Goal
----
Convert an energy release rate (J-integral value) into the mode-I stress intensity factor (the paper's Eq. 9 when dynamic=True, its quasi-static counterpart Eq. 16 when dynamic=False). The dynamic conversion divides by the velocity-dependent factor A_I; the quasi-static conversion uses the plane-strain effective modulus and is independent of the crack speed. Recover both relations from the paper.

```python
import numpy as np


def dsif(J, A_I, E, nu, dynamic=True):
    """J: energy release rate (>= 0); A_I: the velocity factor from wave_factor; E, nu:
    plane-strain constants; dynamic: if True use the dynamic conversion (paper Eq. 9),
    else the quasi-static one (paper Eq. 16). Returns float: the mode-I stress intensity
    factor. Raises ValueError if J < 0."""
    return 0.0
```

### Step 8

hsiga_audit

Goal
----
Run the full hybrid s-version isogeometric dynamic-fracture audit for a crack running at the given speed, on the declared fixed configuration (a quadratic global B-spline patch on [-1.5, 1.5]^2 with an overlaid 3x3 local Q4 mesh, the declared displacement and velocity fields with the velocity scaled by the crack speed, and E = 1.0, nu = 0.3, rho = 1.0). Return the audit vector [A_I, V1, V2, J_dynamic, J_static, K_dynamic, K_static, coupling_norm, q_count, W_ref, K_ref, partition] where: A_I, V1, V2 come from the wave factor; J_dynamic and J_static are the magnitudes of the dynamic and quasi-static J-integrals over the local mesh; K_dynamic and K_static are the corresponding mode-I stress intensity factors; coupling_norm is the Frobenius norm of the global-local coupling stiffness; q_count is the total number of local element-nodes inside the J-domain; W_ref and K_ref are the strain- and kinetic-energy densities at the declared reference state; partition is the sum of the global basis values at parameter 0.3. Assemble by calling the earlier sub-problem functions. The stress intensity factors must use the dynamic conversion for K_dynamic and the quasi-static conversion for K_static.

```python
import numpy as np


def hsiga_audit(crack_velocity):
    """crack_velocity: positive crack-tip speed below the shear wave speed. Builds the
    declared fixed configuration (velocity field scaled by crack_velocity) and returns a
    float64 array (12,): [A_I, V1, V2, J_dynamic, J_static, K_dynamic, K_static,
    coupling_norm, q_count, W_ref, K_ref, partition]. Assembled by calling the earlier
    sub-problem functions. Raises ValueError on a non-positive or nonfinite crack_velocity."""
    return np.zeros(12)
```
