# Terminal current of a PN junction from a harmonic-average discrete duality finite volume scheme

## Background

Every simulation that predicts a current-voltage curve, a breakdown voltage or a switching loss for a power device rests on the drift-diffusion model: two continuity equations for the electron and hole densities, closed by a Poisson equation for the electrostatic potential. What makes the system hard is not its size but its character. Across a PN junction the potential swings by the built-in voltage over a depletion width of a fraction of a micrometre, so the drift term overwhelms the diffusion term and the equations become strongly convection dominated. Discretise the flux with a plain centred difference and the computed carrier densities oscillate, go negative, and take the Newton iteration with them. The classical cure, introduced by Scharfetter and Gummel in 1969 and still the backbone of every commercial device simulator, is to solve the one-dimensional flux equation exactly along each mesh edge under the assumption that the potential varies linearly on it. The exact solution weights the two endpoint densities by the Bernoulli function of the potential difference, which interpolates smoothly between a centred difference when the edge is diffusion dominated and pure upwinding when it is drift dominated, and which guarantees that the discrete densities stay positive.

That construction carries a hidden geometric price. The Scharfetter-Gummel flux is one-dimensional along the edge joining two nodes, so the finite volume method built on it needs control volumes whose interfaces are perpendicular to those edges. That is exactly what the Voronoi diagram provides, and a Voronoi diagram is the dual of a Delaunay triangulation. The method therefore inherits a hard requirement that the mesh be Delaunay. For a rectangle this is no burden, but real devices are not rectangles: a thyristor or an insulated gate bipolar transistor has re-entrant corners, thin implanted layers, and junction depths spanning three orders of magnitude, and a mesh generator asked to respect all of that while also staying Delaunay will either fail or produce elements so small that the simulation becomes unaffordable. Where the Delaunay condition is violated the Voronoi construction degenerates, control volume interfaces fall outside the elements they are supposed to separate, and the scheme returns unphysical local maxima in the carrier density or refuses to converge at all.

The discrete duality finite volume framework attacks this by refusing to choose a single mesh. It stores one unknown per primal cell and one per primal vertex, so that the same scalar field is represented twice, and it reconstructs a gradient on the quadrilateral diamond cells spanned by each primal edge and its dual counterpart. Because the diamond gradient uses both the primal and the dual difference, it is consistent on completely general polygonal meshes, including non-Delaunay ones, meshes with hanging nodes, and meshes whose triangles are nearly degenerate. The duality of the name refers to a discrete Green formula that makes the primal divergence and the dual gradient formally adjoint, which is what delivers local conservation on both meshes at once. The cost is a doubled number of unknowns and a coupling that the one-dimensional Scharfetter-Gummel argument does not know how to handle, because the diamond gradient mixes two directions that are in general not orthogonal.

Marrying the two ideas is therefore not a matter of substituting one flux into the other. The exponential coefficient that appears when the continuity equations are written in self-adjoint form has to be frozen on each direction of the diamond separately, and the pairing between a frozen coefficient and the density difference it multiplies is not the one the derivation hands you: the natural pairing leaves an uncancelled exponential of the potential and the scheme overflows on the first Newton step. Getting that pairing right is what makes the exponentials telescope, recovers a genuine Scharfetter-Gummel structure along each direction, and leaves behind a cross term proportional to the cosine of the angle between the primal and dual directions. That cross term is precisely the part of the discretisation that has no counterpart in a Voronoi-based scheme, and it vanishes only on the orthogonal meshes where the classical method already worked. Whether the resulting scheme is worth its extra unknowns is an empirical question about how much of the computed terminal current the cross term actually carries, and about how far the answer drifts when the mesh is deliberately ruined.

## Problem

Stationary drift-diffusion simulation of semiconductor devices is normally discretised with the finite volume Scharfetter-Gummel method, whose control volumes come from the Voronoi dual and therefore require a Delaunay primal mesh; meshes of that quality are difficult to generate for real device geometries, and on skewed elements the method loses accuracy and can stop converging altogether. A discrete duality finite volume discretisation lifts that restriction by carrying unknowns on a primal mesh and a dual mesh simultaneously and reconstructing a gradient on the diamond cells built from the two, but a plain discrete duality treatment of the carrier flux is centred rather than upwinded and degrades on the convection-dominated transport across a PN junction. The scheme considered here closes that gap by freezing the exponential coefficient of the self-adjoint form of each continuity equation, on each of the two directions of a diamond cell, as the harmonic average of the exponential of the linear potential projection along that direction, so that a Scharfetter-Gummel structure is recovered along each direction separately while the diamond gradient keeps the full mesh generality. Compute the terminal current of the device specified below.

The device is the unit square measured in micrometres, with net doping $+10^{15}\,\mathrm{cm^{-3}}$ below $y = 0.5$, $-10^{15}\,\mathrm{cm^{-3}}$ above it, and the mean of the two exactly on it. The edges $y = 0$ and $y = 1$ are Ohmic contacts held at $0\,\mathrm{V}$ and $0.4\,\mathrm{V}$ respectively, the edges $x = 0$ and $x = 1$ carry no contact, and the net recombination rate vanishes everywhere. Use silicon at $300\,\mathrm{K}$: permittivity $1.035941\times10^{-12}\,\mathrm{C\,V^{-1}cm^{-1}}$, elementary charge $1.602192\times10^{-19}\,\mathrm{C}$, thermal voltage $0.025852\,\mathrm{V}$, effective intrinsic density $1.087386\times10^{10}\,\mathrm{cm^{-3}}$, and electron and hole diffusion coefficients $36.63227$ and $12.16336\,\mathrm{cm^{2}s^{-1}}$. Nondimensionalise lengths by $1\,\mathrm{\mu m}$, potentials by the thermal voltage, carrier densities by $10^{15}\,\mathrm{cm^{-3}}$ and diffusivities by the electron value, and report the answer in those units.

The primal mesh divides the square into 6 columns and 12 rows of equal rectangles and cuts each rectangle along the diagonal rising from its lower left to its upper right corner. Every vertex strictly interior to the square is then displaced by $(s\,\Delta x/2.5,\ s\,\Delta y/2.5)$, where $\Delta x$ and $\Delta y$ are the rectangle side lengths and $s$ is $+1$ when the sum of the vertex column and row indices is even and $-1$ otherwise. Solve the resulting coupled nonlinear system to a maximum scaled residual of $10^{-10}$. Your reasoning should also establish how much the computed current changes when the non-orthogonal coupling is removed globally by setting the scalar product of the two diamond normals to zero everywhere it appears, including the Poisson operator and both carrier-flux operators, and how far the result moves when the vertex displacement is removed. Within that short explanation, state the Ohmic-contact construction, the diamond-gradient and Bernoulli harmonic-average construction, the crossed-coefficient swap and hole-argument convention, the treatment of degenerate primal boundary cells and boundary dual cells, and the terminal-current sign and conservation check. Your final answer must be a single number: the total terminal current at the cathode contact, in the scaled units above, taken positive when conventional current leaves the device through that contact.

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

01_build_ddfv_mesh

Goal
----
Build the diamond-cell geometry table of the discrete duality finite volume mesh for the distorted triangulation of the unit square.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_ddfv_mesh(nx: int, ny: int, distortion: float) -> np.ndarray:
    """Build the diamond-cell geometry table of the discrete duality mesh.

    The vertex at column i and row j carries index j * (nx + 1) + i. Triangles
    are numbered rectangle by rectangle in that same row-major order, the
    triangle below the splitting diagonal preceding the one above it. Global
    node indices run over primal cells first (one per triangle, in triangle
    order), then boundary primal cells (one per boundary edge, in ascending
    vertex-pair order), then dual cells (one per vertex, in vertex order).
    Diamonds are listed with the interior primal edges first and the boundary
    primal edges last, each group in ascending vertex-pair order, and for an
    interior primal edge K is the lower-numbered and L the higher-numbered of
    the two triangles sharing it.

    Parameters
    ----------
    nx : int
        Number of rectangle columns across the unit square (nx >= 2).
    ny : int
        Number of rectangle rows up the unit square (ny >= 2).
    distortion : float
        Interior-vertex displacement as a fraction of the rectangle side
        lengths, 0 <= distortion < 0.5.

    Returns
    -------
    diamonds : np.ndarray
        Array of shape (n_diamonds, 9). Column 0 is the global index of the
        primal cell K, column 1 that of the primal cell L, columns 2 and 3
        those of the dual cells K_star and L_star, column 4 the diamond area,
        column 5 the primal edge length, column 6 the dual edge length,
        column 7 the scalar product of the two unit normals, and column 8 the
        flag 1.0 when the primal edge lies on the boundary and 0.0 otherwise.

    Raises
    ------
    ValueError
        If nx or ny is not an integer at least 2, distortion is not finite or
        lies outside [0, 0.5), or the requested mesh contains a degenerate
        diamond.
    """
    return diamonds  # placeholder
```

### Step 2

02_build_ddfv_node_table

Goal
----
Build the node table of the discrete duality mesh, holding the coordinate, the control volume measure, the equation class and the contact tag of every unknown.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_ddfv_node_table(nx: int, ny: int, distortion: float) -> np.ndarray:
    """Build the node table of the discrete duality mesh.

    The global node ordering matches build_ddfv_mesh. The vertex at column i
    and row j carries index j * (nx + 1) + i, triangles are numbered rectangle
    by rectangle in that same row-major order with the triangle below the
    splitting diagonal first, and the node indices then run over primal cells
    in triangle order, then boundary primal cells in ascending vertex-pair
    order, then dual cells in vertex order.

    Parameters
    ----------
    nx : int
        Number of rectangle columns across the unit square (nx >= 2).
    ny : int
        Number of rectangle rows up the unit square (ny >= 2).
    distortion : float
        Interior-vertex displacement as a fraction of the rectangle side
        lengths, 0 <= distortion < 0.5.

    Returns
    -------
    nodes : np.ndarray
        Array of shape (n_nodes, 5). Columns 0 and 1 hold the node
        coordinate, column 2 the control volume measure (zero for a boundary
        edge, whose cell is degenerate), column 3 the equation class code in
        0 to 5, and column 4 the contact tag, 0 away from a contact, 1 on the
        cathode at y equal to zero and 2 on the anode at y equal to one.

    Raises
    ------
    ValueError
        If nx or ny is not an integer at least 2, or distortion is not finite
        or lies outside [0, 0.5).
    """
    return nodes  # placeholder
```

### Step 3

03_evaluate_bernoulli

Goal
----
Evaluate the Bernoulli function that carries the exponential fitting of the harmonic-average flux, in a form that stays accurate at the removable singularity and is finite in both asymptotic limits.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def evaluate_bernoulli(t: np.ndarray) -> np.ndarray:
    """Evaluate the Bernoulli function B(t) = t / (exp(t) - 1), with B(0) = 1.

    Parameters
    ----------
    t : np.ndarray
        Array of scaled potential differences, of any shape. Values may span
        the full double precision range in both directions.

    Returns
    -------
    values : np.ndarray
        Array of the same shape as t holding B(t), finite and strictly
        positive for every finite input.

    Raises
    ------
    ValueError
        If t contains a non-finite value.
    """
    return values  # placeholder
```

### Step 4

04_build_junction_state

Goal
----
Evaluate the scaled net doping at every unknown and the Ohmic contact values of the potential and the two carrier densities.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_junction_state(nodes: np.ndarray, anode_voltage: float) -> np.ndarray:
    """Evaluate the scaled doping and the Ohmic contact values at every node.

    Parameters
    ----------
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table: coordinate, control volume measure, equation
        class code and contact tag.
    anode_voltage : float
        Voltage applied at the anode contact, in volts. The cathode contact
        is grounded.

    Returns
    -------
    state : np.ndarray
        Array of shape (n_nodes, 4). Column 0 is the scaled net doping,
        column 1 the Ohmic electron density, column 2 the Ohmic hole density
        and column 3 the Ohmic potential in units of the thermal voltage.
        Columns 1 to 3 are meaningful only at Dirichlet nodes but are
        evaluated everywhere, because they also serve as the charge-neutral
        initial guess of the Newton iteration.

    Raises
    ------
    ValueError
        If nodes is not a finite array of shape (n_nodes, 5), or
        anode_voltage is not finite.
    """
    return state  # placeholder
```

### Step 5

05_assemble_ddfv_laplacian

Goal
----
Assemble the discrete duality Laplacian, the matrix that maps the nodal potential to the divergence of its diamond-reconstructed gradient on both the primal and the dual mesh.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_ddfv_laplacian(diamonds: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Assemble the discrete duality Laplacian of the potential.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.

    Returns
    -------
    laplacian : np.ndarray
        Array of shape (n_nodes, n_nodes). Applied to a nodal potential
        vector it returns the divergence of the diamond-reconstructed
        gradient on interior primal and dual rows, the outward normal
        gradient on contact-free boundary edge rows, and zeros on Dirichlet
        rows, which are overwritten downstream.

    Raises
    ------
    ValueError
        If either table has the wrong shape, a diamond has non-positive area,
        or a diamond node index lies outside the node table.
    """
    return laplacian  # placeholder
```

### Step 6

06_assemble_harmonic_flux_matrix

Goal
----
Assemble the harmonic-average discrete duality flux matrix for one carrier at a frozen potential, the operator whose row sums are the discrete continuity residuals.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_harmonic_flux_matrix(diamonds: np.ndarray, nodes: np.ndarray,
                                  potential: np.ndarray, diffusivity: float,
                                  is_hole: bool) -> np.ndarray:
    """Assemble the harmonic-average flux matrix for one carrier.

    The exponential fitting factor is supplied by evaluate_bernoulli, which
    this function calls rather than reimplementing.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    potential : np.ndarray
        Nodal electrostatic potential of shape (n_nodes,), in units of the
        thermal voltage.
    diffusivity : float
        Scaled diffusion coefficient of the carrier (diffusivity > 0).
    is_hole : bool
        True for the hole equation, which reverses every Bernoulli argument,
        and False for the electron equation.

    Returns
    -------
    flux_matrix : np.ndarray
        Array of shape (n_nodes, n_nodes) which, applied to a nodal carrier
        density vector, returns the discrete flux balance of every unknown.

    Raises
    ------
    ValueError
        If the geometry, node table or potential has the wrong shape, the
        potential is non-finite, diffusivity is not finite and positive,
        is_hole is not boolean, a diamond area is non-positive, or a diamond
        node index lies outside the node table.
    """
    return flux_matrix  # placeholder
```

### Step 7

07_assemble_coupled_residual

Goal
----
Assemble the residual of the full coupled Poisson and continuity system, with the Dirichlet and Neumann rows substituted in.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_coupled_residual(nodes: np.ndarray, state: np.ndarray,
                              laplacian: np.ndarray, electron_matrix: np.ndarray,
                              hole_matrix: np.ndarray,
                              unknowns: np.ndarray) -> np.ndarray:
    """Assemble the residual of the coupled drift-diffusion system.

    Parameters
    ----------
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    state : np.ndarray
        Doping and Ohmic contact values of shape (n_nodes, 4) as returned by
        build_junction_state.
    laplacian : np.ndarray
        Discrete duality Laplacian of shape (n_nodes, n_nodes).
    electron_matrix : np.ndarray
        Electron flux matrix of shape (n_nodes, n_nodes) assembled at the
        potential held in unknowns.
    hole_matrix : np.ndarray
        Hole flux matrix of shape (n_nodes, n_nodes) assembled at the same
        potential.
    unknowns : np.ndarray
        Current iterate of shape (n_nodes, 3) holding the potential, the
        electron density and the hole density.

    Returns
    -------
    residual : np.ndarray
        Array of shape (3 * n_nodes,) stacking the Poisson, electron and
        hole residuals in that order.

    Raises
    ------
    ValueError
        If nodes, state, unknowns or any operator has the wrong shape, or if
        unknowns contains a non-finite value.
    """
    return residual  # placeholder
```

### Step 8

08_solve_drift_diffusion

Goal
----
Solve the coupled nonlinear drift-diffusion system by voltage continuation and full Newton iteration, calling the assembly functions of sub-problems 04 to 07 and returning the converged potential and carrier densities.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_drift_diffusion(diamonds: np.ndarray, nodes: np.ndarray,
                          anode_voltage: float, n_steps: int = 4,
                          tolerance: float = 1.0e-10,
                          max_iterations: int = 50) -> np.ndarray:
    """Solve the coupled drift-diffusion system to the requested residual.

    The contact state, the Laplacian, the two flux matrices and the residual
    are supplied by build_junction_state, assemble_ddfv_laplacian,
    assemble_harmonic_flux_matrix and assemble_coupled_residual, which this
    function calls rather than reimplementing. Only the Jacobian, which needs
    the derivative of the Bernoulli function and has no sub-problem of its
    own, is assembled here.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    anode_voltage : float
        Target voltage applied at the anode contact, in volts.
    n_steps : int
        Number of equal voltage continuation steps used to reach the target
        bias from zero (n_steps >= 1).
    tolerance : float
        Convergence threshold on the maximum absolute scaled residual
        (tolerance > 0).
    max_iterations : int
        Maximum Newton iterations allowed per continuation step
        (max_iterations >= 1).

    Returns
    -------
    solution : np.ndarray
        Array of shape (n_nodes, 3) holding the converged potential in
        thermal voltages, the electron density and the hole density, both in
        units of the reference doping.

    Raises
    ------
    ValueError
        If an input shape or scalar bound is invalid, a diamond area is
        non-positive, the Newton system is singular or non-finite, or the
        requested tolerance is not reached within max_iterations.
    """
    return solution  # placeholder
```

### Step 9

09_compute_terminal_current

Goal
----
Sum the converged harmonic-average fluxes over the boundary diamonds of one contact to obtain the terminal current there.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_terminal_current(diamonds: np.ndarray, nodes: np.ndarray,
                             solution: np.ndarray, contact_tag: int) -> float:
    """Sum the converged fluxes over one contact to obtain its terminal current.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    solution : np.ndarray
        Converged state of shape (n_nodes, 3) holding the potential, the
        electron density and the hole density.
    contact_tag : int
        Which contact to sum over: 1 for the grounded cathode along y equal
        to zero, 2 for the biased anode along y equal to one.

    Returns
    -------
    current : float
        Total terminal current at that contact in scaled units, as a native
        Python float, positive when conventional current leaves the device
        through the contact.

    Raises
    ------
    ValueError
        If an input has the wrong shape, solution is non-finite, contact_tag
        is not 1 or 2, a diamond area is non-positive, or the requested
        contact has no boundary diamond.
    """
    return current  # placeholder
```

### Step 10

10_run_ddfv_ha_pipeline

Goal
----
Chain the sub-problem functions 01 to 09 end-to-end on the distorted PN junction and return the terminal current at the requested contact. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_ddfv_mesh, build_ddfv_node_table, evaluate_bernoulli, build_junction_state, assemble_ddfv_laplacian, assemble_harmonic_flux_matrix, assemble_coupled_residual, solve_drift_diffusion, compute_terminal_current) rather than reimplementing them.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_ddfv_ha_pipeline(nx: int = 6, ny: int = 12, distortion: float = 0.40,
                         anode_voltage: float = 0.4, n_steps: int = 4,
                         tolerance: float = 1.0e-10, max_iterations: int = 50,
                         contact_tag: int = 1, drop_cross_term: bool = False) -> float:
    """Run the full harmonic-average discrete duality measurement.

    Parameters
    ----------
    nx : int
        Number of rectangle columns across the unit square (nx >= 2).
    ny : int
        Number of rectangle rows up the unit square (ny >= 2).
    distortion : float
        Interior-vertex displacement as a fraction of the rectangle side
        lengths, 0 <= distortion < 0.5.
    anode_voltage : float
        Voltage applied at the anode contact, in volts.
    n_steps : int
        Number of equal voltage continuation steps (n_steps >= 1).
    tolerance : float
        Convergence threshold on the maximum absolute scaled residual
        (tolerance > 0).
    max_iterations : int
        Maximum Newton iterations per continuation step (max_iterations >= 1).
    contact_tag : int
        Contact to report: 1 for the grounded cathode, 2 for the anode.
    drop_cross_term : bool
        When True the scalar product of the two diamond normals is zeroed
        before solving, which removes the non-orthogonal coupling from the
        Poisson operator and both carrier-flux operators.

    Returns
    -------
    current : float
        Terminal current at the requested contact in scaled units, as a
        native Python float, positive when conventional current leaves the
        device through that contact.

    Raises
    ------
    ValueError
        If an integer or scalar bound is invalid, contact_tag is not 1 or 2,
        or drop_cross_term is not boolean.
    """
    return current  # placeholder
```
