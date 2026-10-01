# Biology-Biochemistry-45

## Background

Epithelial monolayers are confluent sheets of polygonal cells whose shapes are set by cell-area elasticity and junctional tension, and whose junctions remodel through T1 neighbour exchanges. Stretched epithelial sheets, like ductile solids, can localise deformation into a neck, and cell-resolved vertex models are used to ask how junction remodelling and the mechanics of the tissue's free edges shape that instability.

## Problem

An ordered epithelial stripe stretched along its length can neck, and I want to know how a line tension along its free edges changes the load at which necking begins. I model the sheet with a vertex model in which every cell has the dimensionless energy e = (a − 1)²/2 + κ(p − χ)²/2 of its area a and perimeter p, with lengths in units of the square root of the preferred cell area, κ = 0.16 and χ = 3.5. The stripe consists of ten rows of cells running along the load, cut from the stress-free regular honeycomb so that one pair of edges of every cell is perpendicular to the load and both sides of the stripe are zigzag free edges, and it is long enough to be treated as periodic along the load. Its free edges carry a constant line tension T = 0.03, which adds an energy T per unit length of free edge. The stretch is measured from the stress-free honeycomb, and at every stretch all vertices relax quasi-statically to mechanical equilibrium. A junction undergoes a T1 neighbour exchange as soon as its length falls to 0.1. From this description I want the tensile load the stripe carries at necking bifurcation divided by the necking-bifurcation load of the identical stripe without line tension, reported to ten significant figures.

In `<reasoning>`, state the stress-free honeycomb edge length; the necking-bifurcation stretch of each stripe; where in the tensioned stripe the first neighbour exchanges occur and the ordered ten-row vector of shortest load-perpendicular junction lengths at that moment; the tensile load of each stripe; the separate contributions to the load increase from the direct boundary-energy derivative and from reshaping the two outer cell rows; how these results compare with published vertex simulations that add boundary line tension to a stretched epithelial stripe; and the final ratio. Those derived quantities determine the final number and distinguish a relaxed finite-width calculation from a straight-edge force estimate; the output requirements below exclude restating the supplied model.

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

01_build_stress_free_stripe

Goal
----
Build the vertex mesh of an ordered epithelial stripe, periodic along the load, made of regular hexagonal cells at the stress-free size of the area-perimeter cell energy.

```python
def build_stress_free_stripe(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
) -> tuple:
    """Return the vertex mesh of the stress-free stripe.

    A cell of dimensionless area ``a`` and perimeter ``p`` carries the energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2``. The stripe is built
    from regular hexagons whose edge length ``d`` minimises ``e`` among
    regular hexagons (relative accuracy ``1e-13``). It has ``n_rows`` rows
    running along the load direction ``x`` and is periodic along ``x`` with
    ``n_columns`` cells per row and period ``n_columns * sqrt(3) * d``.

    The cell in row ``r`` (``r = 0`` lowest) and column ``c`` is centred at
    ``((c + (r % 2) / 2) * sqrt(3) * d, 1.5 * d * r)`` and has two edges
    perpendicular to ``x``. Its six vertices, counter-clockwise from the
    lower end of its right perpendicular edge, lie at the centre plus ``d``
    times ``(sqrt(3)/2, -1/2)``, ``(sqrt(3)/2, 1/2)``, ``(0, 1)``,
    ``(-sqrt(3)/2, 1/2)``, ``(-sqrt(3)/2, -1/2)`` and ``(0, -1)``.

    Cells are listed row by row and, within a row, by column. A vertex at
    ``(x, y)`` is stored at ``(x - s * period, y)`` with the integer
    ``s = floor(x / period + 1e-9)``, and the cell records ``s`` so that its
    unwrapped vertex is ``vertices[v] + (s * period, 0)``. Stored positions
    that agree within ``1e-9 * d`` are one vertex; vertices are numbered in
    order of first appearance in the cell listing and keep the stored
    position of that first appearance.

    Parameters
    ----------
    n_rows : int
        Number of cell rows, at least 1.
    n_columns : int
        Number of cells per row within one period, at least 2.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    tuple
        ``(vertices, cells, shifts, period)``: float array ``(V, 2)``, integer
        arrays ``(n_rows * n_columns, 6)`` of vertex indices and periodic
        shifts, and the float period.

    Raises
    ------
    ValueError
        If ``n_rows`` is not an integer of at least 1, ``n_columns`` is not
        an integer of at least 2 (booleans are rejected), or ``kappa`` or
        ``chi`` is not a finite positive real number.
    """
    return vertices, cells, shifts, period
```

### Step 2

02_compute_cell_geometry

Goal
----
Evaluate the areas, perimeters and edge lengths of every cell of a periodic vertex mesh and flag the edges that lie on a free boundary of the tissue.

```python
def compute_cell_geometry(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
) -> tuple:
    """Return the areas, perimeters, edge lengths and free-edge flags of the cells.

    The mesh follows ``build_stress_free_stripe``: local vertex ``j`` of cell
    ``i`` is at ``vertices[cells[i, j]] + (shifts[i, j] * period, 0)``, and
    local edge ``j`` of cell ``i`` runs from its local vertex ``j`` to local
    vertex ``(j + 1) % 6``. Areas are signed, positive for counter-clockwise
    vertex order. Two cell edges are the same junction when they join the
    same two vertex indices with the same difference of periodic shifts
    between their ends. An edge is free (flag 1) when that junction belongs
    to exactly one cell, and interior (flag 0) when it belongs to two.

    Parameters
    ----------
    vertices : np.ndarray
        Finite float array of shape ``(V, 2)``.
    cells : np.ndarray
        Integer array of shape ``(C, 6)`` with entries in ``[0, V)``.
    shifts : np.ndarray
        Integer array of the same shape as ``cells``.
    period : float
        Positive period along ``x``.

    Returns
    -------
    tuple
        ``(areas, perimeters, edge_lengths, free_edges)``: float arrays of
        shapes ``(C,)``, ``(C,)`` and ``(C, 6)``, and an integer ``(C, 6)``
        array of 0/1 flags.

    Raises
    ------
    ValueError
        If ``vertices`` is not a nonempty finite ``(V, 2)`` array, ``cells``
        is not a nonempty integer ``(C, 6)`` array with entries in
        ``[0, V)``, ``shifts`` is not an integer array of the shape of
        ``cells``, ``period`` is not a finite positive real number, or a
        junction belongs to more than two cells.
    """
    return areas, perimeters, edge_lengths, free_edges
```

### Step 3

03_compute_stripe_energy

Goal
----
Evaluate the total vertex-model energy of a periodic stripe whose free edges carry a constant line tension, together with its gradient with respect to every vertex position.

```python
def compute_stripe_energy(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> tuple:
    """Return the total energy of the stripe and its gradient.

    With the mesh, areas, perimeters, edge lengths and free-edge flags of
    ``compute_cell_geometry``, the total energy is the sum over cells of
    ``(a - 1)**2 / 2 + kappa * (p - chi)**2 / 2`` plus ``line_tension``
    times the summed length of all free edges. The period is held fixed.
    Return the energy and its partial derivatives with respect to the
    stored coordinates of every vertex (a vertex that appears through
    several periodic images collects the derivative of each image).

    Parameters
    ----------
    vertices, cells, shifts, period
        Periodic mesh as in ``compute_cell_geometry``.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    line_tension : float
        Non-negative line tension of the free edges.

    Returns
    -------
    tuple
        ``(energy, gradient)``: a float and a float array shaped like
        ``vertices``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number,
        ``line_tension`` is not a finite non-negative real number (booleans
        are rejected), any edge has zero length, or
        ``compute_cell_geometry`` rejects the mesh.
    """
    return energy, gradient
```

### Step 4

04_compute_virial_stresses

Goal
----
Evaluate the Virial stress tensor of every cell of the stripe from the vertex forces of the cell's own energy, including the line tension on its free edges.

```python
def compute_virial_stresses(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
) -> "np.ndarray":
    """Return the Virial stress components of every cell.

    Cell ``i`` owns the energy ``E_i = (a_i - 1)**2 / 2 + kappa * (p_i - chi)**2
    / 2`` plus ``line_tension`` times the length of its own free edges (the
    free-edge flags of ``compute_cell_geometry``). With ``r_b`` the unwrapped
    positions of its six vertices, its Virial stress is
    ``sigma_i = (1 / a_i) * sum_b r_b (outer) dE_i/dr_b``, where the outer
    product takes the position component first; ``E_i`` does not change
    when the cell is translated, so the origin of ``r_b`` is immaterial.
    Return ``[sigma_xx, sigma_yy, sigma_xy]`` for every cell.

    Parameters
    ----------
    vertices, cells, shifts, period
        Periodic mesh as in ``compute_cell_geometry``.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    line_tension : float
        Non-negative line tension of the free edges.

    Returns
    -------
    np.ndarray
        Float array of shape ``(C, 3)``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number,
        ``line_tension`` is not a finite non-negative real number (booleans
        are rejected), any cell area is not positive, any edge has zero
        length, or ``compute_cell_geometry`` rejects the mesh.
    """
    return stresses
```

### Step 5

05_relax_stripe

Goal
----
Relax every vertex of the periodic stripe at fixed period to the mechanical equilibrium reached from the supplied configuration.

```python
def relax_stripe(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    tolerance: float,
) -> "np.ndarray":
    """Return the equilibrium vertex positions reached from ``vertices``.

    Holding ``period`` fixed, minimise the energy of
    ``compute_stripe_energy`` starting from ``vertices`` and return the
    local minimum in whose basin of energy descent the starting
    configuration lies. Every component of the energy gradient at the
    returned positions must be at most ``tolerance`` in magnitude, and the
    mean of the returned vertex positions must equal that of ``vertices``
    (the energy is unchanged by a rigid translation).

    Parameters
    ----------
    vertices, cells, shifts, period
        Periodic mesh as in ``compute_cell_geometry``.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    line_tension : float
        Non-negative line tension of the free edges.
    tolerance : float
        Positive gradient tolerance, at most ``1e-6``.

    Returns
    -------
    np.ndarray
        Float array shaped like ``vertices``.

    Raises
    ------
    ValueError
        If ``tolerance`` is not a finite positive real number at most
        ``1e-6`` (booleans are rejected), if the gradient tolerance is not
        reached, or if ``compute_stripe_energy`` rejects its input.
    """
    return relaxed
```

### Step 6

06_compute_row_loads

Goal
----
Resolve the tensile load that the periodic stripe transmits along the load into the contributions of its cell rows, using the cells' Virial stresses.

```python
def compute_row_loads(
    vertices: "np.ndarray",
    cells: "np.ndarray",
    shifts: "np.ndarray",
    period: float,
    kappa: float,
    chi: float,
    line_tension: float,
    n_rows: int,
) -> "np.ndarray":
    """Return the axial load carried by each row of cells.

    The cells are listed row by row as in ``build_stress_free_stripe``, with
    the same number of cells in each of the ``n_rows`` rows. The load of row
    ``r`` is the sum over its cells of ``a_i * sigma_xx_i`` divided by
    ``period``, where ``a_i`` is the cell area of ``compute_cell_geometry``
    and ``sigma_xx_i`` the first component returned by
    ``compute_virial_stresses``. The row loads add up to the derivative of
    the total energy of ``compute_stripe_energy`` with respect to the period
    when every vertex ``x`` coordinate is scaled in proportion to it.

    Parameters
    ----------
    vertices, cells, shifts, period
        Periodic mesh as in ``compute_cell_geometry``.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    line_tension : float
        Non-negative line tension of the free edges.
    n_rows : int
        Number of cell rows, at least 1, dividing the number of cells.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_rows,)``, row 0 first.

    Raises
    ------
    ValueError
        If ``n_rows`` is not an integer of at least 1 that divides the number
        of cells (booleans are rejected), or if ``compute_virial_stresses``
        rejects its input.
    """
    return row_loads
```

### Step 7

07_locate_bifurcation_stretch

Goal
----
Find the stretch at which the shortest load-perpendicular junction of the relaxed stripe first shrinks to the neighbour-exchange threshold.

```python
def locate_bifurcation_stretch(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
    line_tension: float,
    threshold: float,
    tolerance: float,
) -> float:
    """Return the stretch at which a load-perpendicular junction reaches ``threshold``.

    Build the stripe of ``build_stress_free_stripe``. At stretch ``lam`` its
    vertex ``x`` coordinates and its period are multiplied by ``lam``, and
    ``relax_stripe`` (gradient tolerance ``1e-12``) is applied from that
    configuration with the given ``line_tension``. The load-perpendicular
    junctions are local edge 0 of every cell in ``compute_cell_geometry``.
    Let ``g(lam)`` be the length of the shortest of them minus
    ``threshold``. Step ``lam`` upward from 1 in increments of 0.05 until
    ``g(lam) <= 0``, then bisect the last increment, keeping ``g > 0`` at
    the lower end and ``g <= 0`` at the upper end, until the bracket is no
    wider than ``tolerance``, and return its midpoint.

    Parameters
    ----------
    n_rows, n_columns, kappa, chi
        Stripe as in ``build_stress_free_stripe``.
    line_tension : float
        Non-negative line tension of the free edges.
    threshold : float
        Positive junction length at which a neighbour exchange occurs.
    tolerance : float
        Positive bracket width, at most ``1e-6``.

    Returns
    -------
    float
        The stretch at the first threshold crossing.

    Raises
    ------
    ValueError
        If ``line_tension`` is not a finite non-negative real number,
        ``threshold`` is not a finite positive real number, ``tolerance`` is
        not a finite positive real number at most ``1e-6`` (booleans are
        rejected), ``g(1) <= 0``, ``g`` stays positive up to stretch 4, or an
        earlier step rejects a state it visits.
    """
    return 0.0
```

### Step 8

08_estimate_tension_load_ratio

Goal
----
Compose every earlier step to obtain the ratio of the stripe's tensile load at necking bifurcation with a free-edge line tension to the bifurcation load of the same stripe without it.

```python
def estimate_tension_load_ratio(
    n_rows: int = 10,
    kappa: float = 0.16,
    chi: float = 3.5,
    line_tension: float = 0.03,
    threshold: float = 0.1,
    n_columns: int = 2,
    tolerance: float = 1e-12,
) -> float:
    """Return the bifurcation-load ratio of the stripe with and without line tension.

    For the free-edge line tension ``t`` equal to ``line_tension`` and to
    zero, take the stretch ``lam_t`` of ``locate_bifurcation_stretch`` (with
    ``threshold`` and ``tolerance``), relax the affinely stretched
    stress-free stripe of ``build_stress_free_stripe`` at ``lam_t`` with
    ``relax_stripe`` (gradient tolerance ``1e-12``), and sum
    ``compute_row_loads`` over the rows to get the load ``F_t``. Return
    ``F_line_tension / F_0``. The defaults reproduce the problem statement.

    Parameters
    ----------
    n_rows, kappa, chi
        Stripe as in ``build_stress_free_stripe``.
    line_tension : float
        Positive line tension of the free edges.
    threshold : float
        Positive junction length at which a neighbour exchange occurs.
    n_columns : int
        Cells per row within one period, at least 2.
    tolerance : float
        Positive bracket width of the bifurcation stretches, at most ``1e-6``.

    Returns
    -------
    float
        The load ratio ``F_line_tension / F_0``.

    Raises
    ------
    ValueError
        If ``line_tension`` is not a finite positive real number (booleans
        are rejected); if, for either tension, the load at
        ``(1 - 1e-6) * lam_t`` is not below the load at ``lam_t`` or some
        junction at ``lam_t`` is shorter than ``threshold - 1e-6``; or if an
        earlier step rejects its input.
    """
    return 0.0
```
