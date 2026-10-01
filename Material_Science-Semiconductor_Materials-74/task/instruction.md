# Material_Science-Semiconductor_Materials-74

## Background

Numerical simulation is central to semiconductor device design, and the stationary drift-diffusion model is widely used for predicting static device characteristics such as current-voltage curves and breakdown voltages, which are performance indicators for industrial design and academic research. It couples Poisson's equation for the electrostatic potential to a continuity equation for each carrier species, and it is hard to discretize well: the transport is convection-dominated, the coupling between carrier densities and the potential is fully implicit in the steady state, and a usable scheme must preserve physical properties such as local current conservation and non-negative carrier densities.

Commercial semiconductor simulators predominantly rely on one classical discretization: an exponentially fitted, edge-based finite-volume flux. It is excellent at preserving the positivity of particle concentrations in convection-dominated scenarios (where the electric field drift overwhelms thermal diffusion), but it assumes each mesh edge is orthogonal to the face of the control volume it crosses: it only works reliably on high-quality Delaunay meshes with dual Voronoi grids.

Real device geometries routinely violate that assumption. On distorted or non-Delaunay meshes, this classical flux suffers from unphysical numerical oscillations or fails to converge entirely, and the discrete current is no longer conserved cell by cell. Generating a high-quality mesh for every geometry is often not practical, so the mesh quality requirement is a genuine bottleneck. While an earlier family of mesh-flexible finite-volume schemes relaxed these strict mesh constraints, it used central-difference formulations that lacked the necessary stabilization for strong electric fields, leading to stability degradation.

The research group behind this work addresses that bottleneck from the discretization side rather than the meshing side. Their aim is a single scheme that keeps the mesh flexibility of that second family together with the stabilization the classical flux provides, so that neither has to be traded away for the other. The resulting scheme preserves fundamental physical properties like local current conservation and maintains high accuracy even on heavily distorted, nearly degenerate grids (e.g., triangles with nearly 180-degree obtuse angles) where classical methods crash. The authors validate their method against the classical scheme on both high- and low-quality meshes and then apply it to an industrial power device.

## Problem

Simulating carrier transport in a semiconductor device with the stationary drift-diffusion model normally relies on an exponentially fitted, edge-based finite-volume flux, but that classical construction is tied to a Voronoi dual and therefore loses accuracy and local conservation once the mesh is distorted or non-Delaunay, which is exactly the regime encountered in real device geometries. A recent line of work removes the orthogonality requirement by carrying the electrostatic potential and both carrier densities on two interlocking families of control volumes at once and discretizing the transport of each carrier along both mesh directions of a single quadrilateral cell, so that the flux across one interface picks up a contribution from the transverse direction whenever the two outward normals of that cell are not perpendicular. Given the geometry of one such interior cell, along with the potentials, carrier densities, and diffusion coefficients at its four corner control volumes, the method returns the outward electron and hole fluxes across both interfaces.

Consider one interior cell with primal-edge length: `s = 1.6`, dual-edge length: `s_star = 2.1`, included angle:  `theta = pi / 3`, primal outward normal: `n = (1, 0)`, and dual outward normal: `n_star = (1/2, sqrt(3)/2)`. The potentials are `psi_K = 0.25`, `psi_L = -0.35`, `psi_Kstar = 0.50`, `psi_Lstar = -0.15`, the electron densities are `(n_K, n_L, n_Kstar, n_Lstar) = (1.7, 0.9, 1.3, 1.1)`, the hole densities are `(p_K, p_L, p_Kstar, p_Lstar) = (0.8, 1.5, 1.2, 0.6)`, and the diffusion coefficients are `D_n = 1.15` and `D_p = 0.85`; all quantities are dimensionless and the exponential fitting uses the Bernoulli function `B(t) = t / (exp(t) - 1)` with its continuous value `B(0) = 1`. An auxiliary cell carries the same edge lengths, potentials, densities, and diffusion coefficients but has `theta_perp = pi / 2`, `n_perp = (1, 0)` and `n_star_perp = (0, 1)`.

Evaluate both cell geometries, build the outward electron and hole fluxes across the primal and the dual interface of the non-orthogonal cell, and then re-derive those same four fluxes as the neighbouring pair of control volumes would see them. In your reasoning, report the pairwise cancellation residual obtained by adding each forward flux to its reverse-oriented counterpart, compare the auxiliary cell's four fluxes with the non-orthogonal ones, and state what the construction does and does not guarantee about carrier positivity on a general non-orthogonal mesh. The construction's primal flux differs from the flux obtained by keeping, in both of its terms, the drift coefficient that the construction itself freezes on the primal edge; for each carrier, evaluate that alternative primal flux and subtract the construction's primal flux `F_KL` from it. Your final answer must be a single number: the defect these two differences produce in the signed electrical current carried across that interface, expressed in the same scaled units as the carrier fluxes and rounded to 10 decimal places.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_diamond_metrics

Goal
----
Compute the area and the directional coupling factor of one interior diamond cell.

A diamond cell is the quadrilateral spanned by the two primal cell centres and the two dual cell centres that share an interface. It is fixed by the primal edge length |sigma|, the dual edge length |sigma*|, the angle theta between the primal and dual directions, and the two outward unit normals n_KL and n_K**L**.

|D| = 0.5 * |sigma| * |sigma*| * sin(theta)

c   = n_KL . n_K*L*

The scalar c is the factor that multiplies every cross-direction term of the flux discretization; it vanishes exactly when the two normals are orthogonal.

```python
def compute_diamond_metrics(
    primal_edge_length: float,
    dual_edge_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
) -> np.ndarray:
    """Return the diamond-cell area and the primal-dual normal projection.

    Parameters
    ----------
    primal_edge_length : float
        Length |sigma| of the primal edge. Must be finite and > 0.
    dual_edge_length : float
        Length |sigma*| of the dual edge. Must be finite and > 0.
    angle_radians : float
        Angle theta between the primal and dual directions, in radians.
        Must be finite and must give a strictly positive diamond area.
    primal_normal : np.ndarray
        Outward unit normal n_KL of the primal edge, shape (2,), finite.
    dual_normal : np.ndarray
        Outward unit normal n_K*L* of the dual edge, shape (2,), finite.

    Returns
    -------
    metrics : np.ndarray
        Array of shape (2,) holding [diamond_area, normal_projection], where
        diamond_area = 0.5 * |sigma| * |sigma*| * sin(theta) and
        normal_projection = n_KL . n_K*L*.

    Raises
    ------
    ValueError
        If primal_edge_length or dual_edge_length is not finite or is <= 0,
        if angle_radians is not finite, if primal_normal or dual_normal is not
        a finite array of shape (2,), or if the resulting diamond area is
        <= 0.
    """
    return metrics  # noqa: F821
```

### Step 2

02_bernoulli_factor

Goal
----
Evaluate the Bernoulli factor that carries the drift term inside the flux coefficient.

    B(t) = t / (exp(t) - 1)   for t != 0,

    B(0) = 1.

B is the classical exponential-fitting weight: B(t) -> 0 in the convection-dominated limit t -> +inf and B(0) = 1 in the diffusion-dominated limit. The naive quotient loses all significant digits for small |t| because exp(t) - 1 cancels catastrophically, so the implementation must use expm1 well away from the origin, and the Taylor series

B(t) = 1 - t/2 + t^2/12 - t^4/720 + O(t^6)

near it.

```python
def evaluate_bernoulli(argument: float) -> float:
    """Return the Bernoulli factor B(argument), continuous at the origin.

    Parameters
    ----------
    argument : float
        The Bernoulli argument t. Must be finite.

    Returns
    -------
    value : float
        B(t) = t / (exp(t) - 1) for t != 0, and B(0) = 1, evaluated without
        catastrophic cancellation for small |t|.

    Raises
    ------
    ValueError
        If argument is not finite.
    """
    return value  # noqa: F821
```

### Step 3

03_primal_electron_flux

Goal
----
Compute the electron flux across the primal interface of one interior diamond cell, for a locally conservative primal-dual finite-volume discretization of the stationary drift-diffusion model that does not assume mesh orthogonality.

The flux is assembled from two contributions. One acts along the primal edge direction and reproduces the classical exponentially fitted form built from the electron densities at K and L and the Bernoulli weights of the primal potential jump. The other is a cross-direction contribution built from the dual-cell densities at K* and L* and the Bernoulli weights of the dual potential jump; it enters weighted by the projection of the two outward normals on each other and by the mixed edge-length product, and it is the term that a one-dimensional edge-based scheme omits.

```python
def compute_primal_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
) -> float:
    """Return the outward primal electron flux across the interface.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    coupling : float
        Normal projection c = n_KL . n_K*L*. Must be finite.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    n_k : float
        Electron density at cell K. Must be finite.
    n_l : float
        Electron density at cell L. Must be finite.
    n_kstar : float
        Electron density at cell K*. Must be finite.
    n_lstar : float
        Electron density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward primal electron flux across the interface, oriented from
        K to L (K* to L*).

    Raises
    ------
    ValueError
        If area, primal_length, dual_length or diffusion is not finite or is
        <= 0, or if coupling, any potential or any density is not finite.
    """
    return flux  # noqa: F821
```

### Step 4

04_dual_electron_flux

Goal
----
Compute the electron flux across the dual interface of the same interior diamond cell treated in the previous step.

The dual flux mirrors the primal one: its diagonal contribution runs along the dual edge direction and is built from the dual-cell electron densities and the Bernoulli weights of the dual potential jump, while its cross-direction contribution is built from the primal-cell densities and the Bernoulli weights of the primal potential jump. The two contributions carry different edge-length prefactors, and the cross-direction one is weighted by the same projection of the two outward normals that appears in the primal flux.

```python
def compute_dual_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
) -> float:
    """Return the outward dual electron flux across the interface.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    coupling : float
        Normal projection c = n_KL . n_K*L*. Must be finite.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    n_k : float
        Electron density at cell K. Must be finite.
    n_l : float
        Electron density at cell L. Must be finite.
    n_kstar : float
        Electron density at cell K*. Must be finite.
    n_lstar : float
        Electron density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward dual electron flux across the interface, oriented from
        K to L (K* to L*).

    Raises
    ------
    ValueError
        If area, primal_length, dual_length or diffusion is not finite or is
        <= 0, or if coupling, any potential or any density is not finite.
    """
    return flux  # noqa: F821
```

### Step 5

05_primal_hole_flux

Goal
----
Compute the hole flux across the primal interface of the same interior diamond cell treated in the previous steps.

Holes have the opposite charge sign, so the self-adjoint change of variable that linearizes their transport carries the opposite exponential factor. The resulting flux has the same two-contribution structure as its electron counterpart and the same geometric prefactors, but every Bernoulli weight is evaluated at the negative of the argument used for electrons, which exchanges the roles of the upwind and downwind cells.

```python
def compute_primal_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    """Return the outward primal hole flux across the interface.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    coupling : float
        Normal projection c = n_KL . n_K*L*. Must be finite.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    p_k : float
        Hole density at cell K. Must be finite.
    p_l : float
        Hole density at cell L. Must be finite.
    p_kstar : float
        Hole density at cell K*. Must be finite.
    p_lstar : float
        Hole density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward primal hole flux across the interface, oriented from
        K to L (K* to L*).

    Raises
    ------
    ValueError
        If area, primal_length, dual_length or diffusion is not finite or is
        <= 0, or if coupling, any potential or any density is not finite.
    """
    return flux  # noqa: F821
```

### Step 6

06_dual_hole_flux

Goal
----
Compute the hole flux across the dual interface of the same interior diamond cell treated in the previous steps.



This is the dual-direction counterpart of the primal hole flux: it carries the dual edge-length prefactor on its diagonal contribution and the mixed prefactor with the normal projection on its cross-direction contribution, and it uses the same reversed Bernoulli arguments that distinguish holes from electrons.

```python
def compute_dual_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    """Return the outward dual hole flux across the interface.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    coupling : float
        Normal projection c = n_KL . n_K*L*. Must be finite.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    p_k : float
        Hole density at cell K. Must be finite.
    p_l : float
        Hole density at cell L. Must be finite.
    p_kstar : float
        Hole density at cell K*. Must be finite.
    p_lstar : float
        Hole density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward dual hole flux across the interface, oriented from
        K to L (K* to L*).

    Raises
    ------
    ValueError
        If area, primal_length, dual_length or diffusion is not finite or is
        <= 0, or if coupling, any potential or any density is not finite.
    """
    return flux  # noqa: F821
```

### Step 7

07_reversed_interface_fluxes

Goal
----
Rebuild the four carrier fluxes of the same interior interface as seen from the neighbouring pair of cells.

Every interior interface is shared by two primal cells and two dual cells, and each side of it computes its own outward flux. Re-deriving the fluxes from the neighbour's point of view means exchanging the primal labels K and L, exchanging the dual labels K* and L*, and reversing both outward unit normals, then evaluating exactly the same discretization again. The diamond geometry itself is unchanged. This step must recompute all four fluxes from the swapped data rather than reuse the forward values, because the relation between the two orientations is the property being established, not an assumption.

```python
def compute_reversed_interface_fluxes(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    """Return the four interface fluxes seen from the neighbouring cells.

    Parameters
    ----------
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    angle_radians : float
        Angle theta between the primal and dual directions, in radians. Must be
        finite and must give a strictly positive diamond area.
    primal_normal : np.ndarray
        Forward outward unit normal n_KL, shape (2,), finite.
    dual_normal : np.ndarray
        Forward outward unit normal n_K*L*, shape (2,), finite.
    electron_diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    hole_diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    n_k, n_l, n_kstar, n_lstar : float
        Electron densities at cells K, L, K* and L*. Must be finite.
    p_k, p_l, p_kstar, p_lstar : float
        Hole densities at cells K, L, K* and L*. Must be finite.

    Returns
    -------
    fluxes : np.ndarray
        Array of shape (4,) holding the reverse-oriented fluxes in the order
        [primal_electron, dual_electron, primal_hole, dual_hole].

    Raises
    ------
    ValueError
        If primal_length, dual_length, electron_diffusion or hole_diffusion is
        not finite or is <= 0, if angle_radians is not finite or gives a
        non-positive diamond area, if either normal is not a finite array of
        shape (2,), or if any potential or density is not finite.
    """
    return fluxes  # noqa: F821
```

### Step 8

08_orthogonal_sg_limit

Goal
----
Evaluate the same four carrier fluxes on the auxiliary diamond whose primal and dual outward normals are orthogonal.


Orthogonal normals make the projection between the two directions vanish, so the cross-direction contribution of every flux disappears and each flux collapses to a single exponentially fitted edge term along its own direction. Recovering this limit is the consistency check that ties the non-orthogonal discretization back to the classical edge-based scheme it generalizes. The auxiliary diamond keeps the edge lengths, potentials, densities and diffusion coefficients of the original one and changes only the angle and the two normals, so its area is not the same as the non-orthogonal one and has to be recomputed.

```python
def compute_orthogonal_sg_fluxes(
    area: float,
    primal_length: float,
    dual_length: float,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    """Return the four decoupled carrier fluxes of the orthogonal diamond.

    Parameters
    ----------
    area : float
        Area |D_perp| of the auxiliary orthogonal diamond. Must be finite
        and > 0.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    electron_diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    hole_diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    n_k, n_l, n_kstar, n_lstar : float
        Electron densities at cells K, L, K* and L*. Must be finite.
    p_k, p_l, p_kstar, p_lstar : float
        Hole densities at cells K, L, K* and L*. Must be finite.

    Returns
    -------
    fluxes : np.ndarray
        Array of shape (4,) holding the orthogonal-limit fluxes in the order
        [primal_electron, dual_electron, primal_hole, dual_hole].

    Raises
    ------
    ValueError
        If area, primal_length, dual_length, electron_diffusion or
        hole_diffusion is not finite or is <= 0, or if any potential or density
        is not finite.
    """
    return fluxes  # noqa: F821
```

### Step 9

09_consistency_defect

Goal
----
Compute, for one carrier, the two drift coefficients that the construction freezes on the edges of an interior diamond cell, and the defect that the construction introduces into the primal flux when it re-associates those coefficients with the two mesh directions.




Write z = +1 for electrons and z = -1 for holes, and define the carrier's scaled variable Phi = rho * exp(-z * psi) at each cell, where rho is the carrier density. On an edge whose two endpoints carry potentials a and b, the frozen coefficient is the reciprocal of the mean of exp(-z * u) along the edge, with u varying linearly between a and b, which evaluates to

    E(a, b) = exp(z * a) * B(z * (a - b)),    B(t) = t / (exp(t) - 1).

The primal edge joins the two dual cell centres and the dual edge joins the two primal cell centres, so the two coefficients of the cell are


    E_primal = E(psi_Kstar, psi_Lstar),    E_dual = E(psi_K, psi_L).

Keeping E_primal in both terms of the primal flux gives one flux; the construction instead places E_dual on the term along the primal direction. The cross-direction terms are identical, so the unpaired flux minus the construction's flux is

    defect = |sigma|^2 * D / (2 |D|) * (E_primal - E_dual) * (Phi_K - Phi_L).

```python
def compute_consistency_defect(
    area: float,
    primal_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    dens_k: float,
    dens_l: float,
    charge_sign: float,
) -> np.ndarray:
    """Return the two frozen edge coefficients and the primal-flux defect of one carrier.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    diffusion : float
        Carrier diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    dens_k, dens_l : float
        Carrier densities at the primal cells K and L. Must be finite.
    charge_sign : float
        +1.0 for electrons, -1.0 for holes.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) holding [coefficient_primal_edge,
        coefficient_dual_edge, defect], where defect is the unpaired primal
        flux minus the construction's primal flux.

    Raises
    ------
    ValueError
        If area, primal_length or diffusion is not finite or is <= 0, if
        charge_sign is not exactly +1.0 or -1.0, or if any potential or density
        is not finite.
    """
    return result  # noqa: F821
```

### Step 10

10_pipeline

Goal
----
Run the complete interior-diamond diagnostic end to end and return the requested scalar.

The orchestrator chains every preceding step: it forms the diamond area and the normal projection, evaluates the four carrier fluxes of the non-orthogonal diamond, rebuilds the same four fluxes seen by the neighbouring cells and checks that each forward/reverse pair cancels to machine precision, evaluates the auxiliary orthogonal diamond and checks that its decoupled fluxes are finite, and evaluates the primal-flux defect of each carrier. The electron flux carries the sign opposite to the electron current, so the signed electrical current combines the two carriers as -F(n) + F(p); the value returned is the same signed combination of the two defects.

```python
def compute_signed_current_defect(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    angle_radians_perp: float,
    primal_normal_perp: np.ndarray,
    dual_normal_perp: np.ndarray,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    """Return the defect of the signed electrical current across the primal interface.

    Parameters
    ----------
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    angle_radians : float
        Angle theta of the non-orthogonal diamond, in radians. Must be finite
        and must give a strictly positive area.
    primal_normal : np.ndarray
        Outward unit normal n_KL of the non-orthogonal diamond, shape (2,).
    dual_normal : np.ndarray
        Outward unit normal n_K*L* of the non-orthogonal diamond, shape (2,).
    angle_radians_perp : float
        Angle theta_perp of the auxiliary diamond, in radians. Must be finite
        and must give a strictly positive area.
    primal_normal_perp : np.ndarray
        Outward unit normal of the auxiliary diamond, shape (2,).
    dual_normal_perp : np.ndarray
        Outward unit normal of the auxiliary diamond, shape (2,).
    electron_diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    hole_diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    n_k, n_l, n_kstar, n_lstar : float
        Electron densities at cells K, L, K* and L*. Must be finite.
    p_k, p_l, p_kstar, p_lstar : float
        Hole densities at cells K, L, K* and L*. Must be finite.

    Returns
    -------
    current_defect : float
        The signed combination -delta_n + delta_p of the two primal-flux
        defects, where each defect is the flux obtained by keeping the
        primal-edge frozen coefficient in both of its terms minus the
        construction's primal flux. The signs follow from the electron carrier
        flux representing -J_n / q and the hole carrier flux representing
        +J_p / q.

    Raises
    ------
    ValueError
        If primal_length, dual_length, electron_diffusion or hole_diffusion is
        not finite or is <= 0, if either angle is not finite or gives a
        non-positive diamond area, if any normal is not a finite array of shape
        (2,), if any potential or density is not finite, if any exponential
        fitting weight is not finite and strictly positive, or if the
        forward/reverse interface fluxes fail to cancel pairwise to within
        1e-12.
    """
    return current_defect  # noqa: F821
```
