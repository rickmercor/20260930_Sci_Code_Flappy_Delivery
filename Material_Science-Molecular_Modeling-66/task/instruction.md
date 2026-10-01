# Shape dependence of a conditionally convergent Coulomb lattice sum on an orthorhombic cell

## Background

Electrostatic lattice sums underlie the Madelung constants of ionic crystals and the treatment of long-range electrostatics in periodic simulations of condensed matter. For a three-dimensional array of neutral unit cells, the Coulomb sum at an ion converges only conditionally: its value depends on the order in which cells are accumulated, which physically corresponds to the macroscopic shape of the crystal being built. Resolving this ambiguity, and separating what belongs to the periodic lattice from what belongs to the crystal's surface, has been a long-standing theme in the theory of ionic crystals and in molecular simulation.

## Problem

A crystal is built by repeating an orthorhombic unit cell that holds point charges summing to zero.
The electrostatic potential at one ion of the central cell, summed over the whole array, is only
conditionally convergent, so it has no value until the region being summed is fixed. Fix it by
summing over a finite crystal of a stated shape and size, and the sum becomes unambiguous.

Your task is to measure how much of that potential is carried by the shape of the crystal rather
than by the periodic array itself, on a cell whose three edges differ, at a state point the source
does not tabulate.

## Source-dependent decisions

Recover these decisions from the source and justify each one. Do not substitute a familiar textbook convention.

- Write the exact contribution assigned to an ion in a non-central cell, including any reference placement used in that contribution.
- Identify the precise finite set of lattice vectors for a crystal of stated shape and size, including the treatment of the central cell.
- Identify the source's decomposition of a finite result, and decide which term is returned by its periodic construction.
- Identify the electrostatic boundary convention attached to that periodic result.
- Determine how the source converts a site potential into its dimensionless reported constant.

These decisions are intentionally not restated here. They are separately graded and each changes either the implementation, the interpretation of the result, or its normalization. The source uses equal cell edges. For the orthorhombic extension below, carry all three edges through Cartesian positions, reciprocal vectors, cell volume and neighbour distances.

## The chain to build

Implement these eight functions in order. Each returns a numpy float64 array.

1. `clc_shape_offsets(px, py, pz, spherical)`, shape `(N, 3)`: the integer lattice vectors of the
   finite crystal.
2. `clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, px, py, pz, spherical)`, shape
   `(1,)`: the finite lattice sum at the reference ion.
3. `clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)`, shape
   `(1,)`: the shape independent bulk potential.
4. `clc_boundary_term(cell_edges, basis_pos, basis_q, ref_index, px, py, pz, spherical, alpha,
   n_real, n_rec)`, shape `(1,)`: the finite sum minus the bulk potential.
5. `clc_nearest_neighbour(cell_edges, basis_pos, ref_index, span)`, shape `(1,)`: the Cartesian
   distance from the reference ion to the nearest other ion of the crystal.
6. `clc_shape_sequence(cell_edges, basis_pos, basis_q, ref_index, p, ratios_z, alpha, n_real,
   n_rec)`, shape `(n_ratios, 2)`: for each shape, the finite-minus-bulk residual and the source's analytical macroscopic boundary contribution. Use physical side lengths proportional to `(L_x, L_y, ratio_z*L_z)` in that boundary construction.
7. `clc_boundary_decomposition(cell_edges, basis_pos, basis_q, ref_index, p, pz, fit_sizes, alpha, n_real, n_rec, span)`, shape `(6,)`: a fixed-shape size extrapolation returning the asymptotic boundary term, target finite-size correction, reconstructed target residual, fit RMSE, jackknife spread and the signed finite cell-cubic normalized potential $M_{\mathrm{cell}}$.
8. `clc_audit(cell_edges, basis_pos, basis_q, ref_index, p, pz, alpha, n_real, n_rec, span)`, shape `(22,)`: one record assembled from all seven earlier steps. It checks independent residuals at flattened, intermediate and cell-cubic aspect ratios, compares their fitted and analytical macroscopic boundaries, then returns the audit index $J$ defined in the final step contract.

Each signature states its own contract: array shapes, index ordering, reduced against Cartesian
coordinates, the treatment of the central cell, the rounding rule where a thickness comes from a
ratio, and the exact error conditions. Follow them as written.

Validation, in every function: reject a non finite input; reject cell_edges of the wrong shape or
with a non positive entry; reject a basis with fewer than two ions, a mismatched charge array, a
reduced position outside the half open unit interval, a non neutral cell, or two coincident
positions; reject a reference index outside the basis; reject a non integer or non positive size, span or cutoff; reject a splitting parameter $\alpha$ that is not a positive finite real number; in the audit reject p below six or a thickness outside one through p-2. Raise
`ValueError` in each case.

## Benchmark extrapolation conventions

For Step 7, follow the target aspect ratio at each fit half width $s$ using the prescribed rounded thickness

$$
z_s=\max\!\left\{1,\left\lfloor\frac{p_zs}{p}+\frac12\right\rfloor\right\},
$$

the half-up rule with a minimum half width of one. Evaluate $y(s)$ as the finite direct potential minus the periodic bulk potential. Fit

$$
y(s)=B+\frac{c_1}{s}+\frac{c_2}{s^2}+\frac{c_3}{s^3}
$$

by weighted least squares with point weights $s$, equivalently multiply each design-matrix row and response by $\sqrt{s}$. At target size $p$, define the finite-size correction as $y(p)-B$ and the reconstructed target residual as $y(p)$. Define RMSE as the unweighted root-mean-square residual over the six fit sizes. Define the jackknife spread as the maximum absolute change in $B$ among the six leave-one-size-out refits. Here $B$ is the specified regression estimate of the asymptotic boundary contribution.

The sixth Step 7 value is the signed finite cell-cubic diagnostic

$$
M_{\mathrm{cell}}=d_{\mathrm{nn}}\phi_{\mathrm{cube}},
$$

where $\phi_{\mathrm{cube}}$ is the direct finite potential with half widths $(p,p,p)$ at the supplied reference ion. Preserve its sign in this product. At the benchmark's negative reference ion $q_{\mathrm{ref}}=-1$, this diagnostic is positive. This is a finite direct diagnostic, not the bulk periodic Madelung constant.

For Step 8, use

$$
z_{\mathrm{mid}}=\left\lfloor\frac{p+p_z+1}{2}\right\rfloor
$$

for the intermediate thickness. Run Step 7 independently at $p_z$, $z_{\mathrm{mid}}$ and $p$, then apply the stated three-score reduction. The spherical direct-potential diagnostic uses half widths $(p,p,p)$ with `spherical=True`.

## The state point

An orthorhombic cell with edges 1.0, 1.3 and 0.42 along x, y and z. A two ion basis with a charge
of plus one at the cell origin and a charge of minus one at reduced position one half along x, with
the potential evaluated at the negative ion. Crystal half width sixteen cells along x and y, and
four cells along z for the flattened crystal. Splitting parameter four, real space cutoff three,
reciprocal cutoff nine, neighbour search span two. Use convergence-fit half widths 4, 8, 12, 16, 20 and 24 in Step 7.

## What to report

State each source-dependent convention above and explain why it is required. Then report these scalars for the stated state point: the cell-cubic and flattened finite sums, the periodic bulk potential, the flattened-minus-cell-cubic contrast, the nearest-neighbour distance, the positive finite cell-cubic normalized potential $M_{\mathrm{cell}}$, the fitted asymptotic-boundary estimate $B$, the target finite-size correction, the reconstructed target residual, the fit RMSE, the leave-one-size-out jackknife spread, the intermediate and cell-cubic fitted boundary intercepts, the source's analytical macroscopic boundary values for the flattened, intermediate and cell-cubic shapes, the three aspect scores, the cross-shape curvature, and $J$. Distinguish the finite-sample fitted intercepts from the analytical boundary values.

## The final answer

For each aspect $i\in\{\mathrm{flat},\mathrm{mid},\mathrm{cube}\}$, compute

$$
q_i=\frac{|\mathrm{correction}_i|}{|B_i|+|\mathrm{correction}_i|}
+\frac{\mathrm{RMSE}_i}{\mathrm{jackknife}_i+\mathrm{RMSE}_i}.
$$

Compute

$$
\mathrm{curvature}=\frac{|B_{\mathrm{flat}}-2B_{\mathrm{mid}}+B_{\mathrm{cube}}|}
{|B_{\mathrm{flat}}|+|B_{\mathrm{mid}}|+|B_{\mathrm{cube}}|}.
$$

Then compute

$$
\begin{aligned}
J&=\operatorname{median}(q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}})\\
&\quad+\operatorname{std}_{\mathrm{population}}(q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}})
+\mathrm{curvature},
\end{aligned}
$$

where the population standard deviation uses $\mathrm{ddof}=0$.

Report $J$ to eight decimal places. Report the other requested scalars to at least eight decimal places, with their scientific labels. Accepted absolute errors are $10^{-5}$ for each aspect score $q_i$ and for $J$, and $5\times10^{-9}$ for the other scalars. Use unrounded intermediate values. $J$ is the Step 8 three-shape stability audit index, not any direct finite-minus-bulk residual or single-shape score.

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

clc_shape_offsets

Goal
----
Builds the integer lattice vectors of a finite crystal of given shape and size, with the central cell removed.

```python
def clc_shape_offsets(px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    r"""px, py, pz: positive integers, the half widths of the finite crystal in cells along each
    axis. spherical: bool; when true the box is intersected with the sphere of radius
    $\min(p_x, p_y, p_z)$ in index space, with a tolerance of $10^{-12}$ on the radius.

    A finite crystal of a given shape and size is the set of integer lattice vectors $\mathbf{n}$
    with $|n_i| \le p_i$, with the central cell $\mathbf{n} = \mathbf{0}$ removed. These are cell
    indices, not Cartesian vectors, so the sphere is applied to the indices themselves.

    Return them sorted lexicographically by $(n_x, n_y, n_z)$, ascending.

    Returns a numpy float64 array of shape $(N, 3)$.

    Raises:
        ValueError: if any of px, py, pz is not an integer of at least one, or if the resulting
            set is empty.
    """
    return None
```

### Step 2

clc_direct_potential

Goal
----
Evaluates the finite Coulomb lattice sum at one ion of an orthorhombic unit cell over a finite crystal of given shape and size.

```python
def clc_direct_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    r"""cell_edges: array of shape (3,) of positive floats, the edge lengths of the orthorhombic
    unit cell along x, y and z. basis_pos: array of shape (m, 3) with m at least two, the
    positions of the ions in reduced coordinates, each component in $[0, 1)$. basis_q: array of
    shape (m,), their charges, summing to zero. ref_index: integer in $0 \le \mathrm{ref} < m$.
    px, py, pz, spherical: as in the offsets step.

    A reduced coordinate is carried to a Cartesian one by multiplying it componentwise by the cell
    edges, and a cell index $\mathbf{n}$ sits at the Cartesian point $\mathbf{n}\odot\mathbf{L}$.

    The potential at the reference ion has two parts. The other ions of the central cell contribute
    $q_j/|\mathbf{r}_j - \mathbf{r}_{\mathrm{ref}}|$ in Cartesian distance. Every other cell
    contributes, for each basis ion $j$, the difference
    $q_j(1/|\mathbf{n}\odot\mathbf{L} + \mathbf{r}_j - \mathbf{r}_{\mathrm{ref}}|
    - 1/|\mathbf{n}\odot\mathbf{L}|)$, that is the charge at its own site minus the same charge at
    the translated reference site $\mathbf{r}_{\mathrm{ref}} + \mathbf{n}\odot\mathbf{L}$.
    This site is the cell origin in reference-centered coordinates, at displacement
    $\mathbf{n}\odot\mathbf{L}$ from the reference ion. Apply the subtraction for every basis
    ion including the reference one.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: on cell_edges of the wrong shape or with a non-positive entry, a non-finite
            basis, fewer than two ions, a mismatched charge array, a reduced position outside
            $[0, 1)$, a non-neutral cell, two coincident positions, a ref_index outside its range,
            an invalid px, py or pz, or a non-finite result.
    """
    return None
```

### Step 3

clc_ewald_potential

Goal
----
Evaluates the shape independent bulk potential of the infinite orthorhombic lattice by Ewald summation with the tinfoil convention.

```python
def clc_ewald_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    r"""cell_edges, basis_pos, basis_q, ref_index: as in the direct potential step. alpha: finite
    positive real number, the Ewald splitting parameter, in inverse length. n_real: positive integer, the half
    width in cells of the real space image sum. n_rec: positive integer, the half width in integer
    reciprocal indices.

    Compute the site potential using the standard three-dimensional Gaussian-split Ewald
    construction with unit Coulomb prefactor and conducting (tinfoil) boundary conditions.
    The real-space screened interaction is $q_j\,\mathrm{erfc}(\alpha d)/d$, which fixes the
    splitting-parameter convention. Derive the reciprocal-space contribution and the removal
    of the screening cloud's self potential consistently with this interaction.

    Use exactly the supplied finite summation boxes: real-space cell indices satisfy
    $|n_i|\le n_{\mathrm{real}}$, and reciprocal indices satisfy $|b_i|\le n_{\mathrm{rec}}$,
    including both signs of every index. Use the orthorhombic reciprocal vectors with components
    $k_i=2\pi b_i/L_i$ and the cell volume $V=L_xL_yL_z$. Include real-space terms only when
    the Cartesian distance is $d>10^{-12}$, excluding the reference ion's zero-image singularity.
    Omit the reciprocal zero mode $\mathbf{b}=\mathbf{0}$ and fix the additive potential reference
    by this standard zero-mode convention; add no constant offset or macroscopic surface term.
    Return the finite-cutoff estimate, without enlarging either cutoff or extrapolating the sums.

    For converged cutoffs the result does not depend on $\alpha$, which is the check that the three
    parts are balanced.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: any condition the direct potential step rejects on the cell or the reference,
            a non-finite or non-positive alpha, a non-integer or non-positive cutoff, or a non-finite result.
    """
    return None
```

### Step 4

clc_boundary_term

Goal
----
Isolates the shape dependent boundary term together with the finite size correction, as the finite lattice sum minus the Ewald bulk potential.

```python
def clc_boundary_term(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    r"""All arguments as in the direct potential and Ewald steps.

    The finite lattice sum of a given shape and size differs from the shape independent bulk potential
    by a boundary term that depends on the shape but not the size and a finite size correction that
    vanishes as the crystal grows. Return their sum, that is the finite lattice sum of this shape
    and size minus the Ewald bulk potential of the same cell and reference ion.

    For a neutral cell, a vanishing total dipole $\mathbf{M}=\sum_j q_j\mathbf{r}_j$ eliminates
    the asymptotic boundary contribution to the total cell energy, but need not eliminate the
    boundary contribution to an individual site potential. Return the finite-minus-bulk site
    potential as specified; do not replace it by zero solely because $\mathbf{M}=\mathbf{0}$.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: any condition either underlying step rejects, or a non-finite result.
    """
    return None
```

### Step 5

clc_nearest_neighbour

Goal
----
Finds the Cartesian distance from the reference ion to its nearest neighbour, searching every basis ion in every cell within a given span.

```python
def clc_nearest_neighbour(cell_edges: "np.ndarray", basis_pos: "np.ndarray", ref_index: int, span: int) -> "np.ndarray":
    r"""cell_edges, basis_pos, ref_index: as in the direct potential step, except that no charges
    are involved and no neutrality is required. span: positive integer, the half width in cells of
    the search.

    Return the Cartesian distance from the reference ion to the nearest other ion of the crystal,
    searching every basis ion in every cell with $|n_i| \le \mathrm{span}$ and excluding the
    reference ion itself. On an orthorhombic cell the nearest ion is not always the one nearest in
    reduced coordinates, because the edges scale the axes differently.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: on cell_edges of the wrong shape or with a non-positive entry, a basis with
            fewer than two ions or a non-finite or out of range reduced position, a ref_index
            outside its range, a non-integer or non-positive span, or no neighbour found.
    """
    return None
```

### Step 6

clc_shape_sequence

Goal
----
Scans finite shape residuals and the source's macroscopic boundary contribution across crystal shapes.

```python
def clc_shape_sequence(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, ratios_z: "np.ndarray", alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    r"""cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec: as in the boundary term
    step. p: integer at least one, the half width of the crystal in cells along x and y. ratios_z:
    one dimensional array of positive floats, the thickness ratios to scan.

    For each ratio take $p_x=p_y=p$ and $p_z=\max(1,\lfloor p\,\mathrm{ratio}+1/2\rfloor)$
    for the finite rectangular crystal. The second result for that ratio is the exact
    macroscopic boundary contribution of the source's rectangular-prism construction,
    evaluated at the physical asymptotic side-length ratio
    $L_x:L_y:(\mathrm{ratio}\,L_z)$. Apply the source's pair boundary contribution to
    every basis charge relative to the supplied reference ion. Use the unreduced
    within-cell displacement, not a minimum-image displacement. The analytic value is
    independent of $p$; it need not equal the finite residual or the separately fitted
    intercept because finite-size and thickness-rounding effects remain.

    Returns a numpy float64 array of shape $(n_{\mathrm{ratios}},2)$ in ratios_z order.
    Column 0 is the finite direct-minus-periodic residual. Column 1 is the analytic
    macroscopic rectangular-prism boundary contribution.

    Raises:
        ValueError: any condition the boundary term step rejects, a non-integer or non-positive p,
            an empty ratios_z, or a non-positive entry in ratios_z.
    """
    return None
```

### Step 7

clc_boundary_decomposition

Goal
----
Fits the paper-defined boundary and finite-size decomposition across several crystal sizes, then checks its stability by leave-one-size-out refits.

```python
def clc_boundary_decomposition(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, fit_sizes: "np.ndarray", alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    r"""Separate a finite flattened-crystal residual into an asymptotic boundary term and a finite-size correction.

    cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec and span follow the earlier contracts. p is an integer at least four and pz is an integer from one through p. fit_sizes is a one-dimensional array of at least six distinct, strictly increasing integers, each at least four.

    Let $\rho=p_z/p$. For every $s$ in fit_sizes, set $s_z=\max(1,\lfloor s\rho+0.5\rfloor)$ and obtain $y(s)$ from clc_boundary_term for the rectangular crystal $(s,s,s_z)$. Fit
        $y(s)=B+c_1/s+c_2/s^2+c_3/s^3$
    by weighted least squares after multiplying row $s$ by $\sqrt{s}$. $B$ is the intercept estimate from this prescribed finite-sample regression, not an exact analytic evaluation of the macroscopic boundary contribution. Evaluate the target residual $y(p)$ separately and define its finite-size correction as $y(p)-B$. Compute the unweighted fit RMSE. Refit after deleting each size and report the largest absolute change in $B$ as the jackknife spread. Finally compute the cell-cubic Madelung constant from clc_direct_potential and clc_nearest_neighbour.

    The sixth value is the signed finite direct diagnostic $M_{\mathrm{cell}} = d_{\mathrm{nn}}\,\phi_{\mathrm{cube}}$ for the supplied ref_index, where $\phi_{\mathrm{cube}}$ is the direct finite potential for half widths $(p,p,p)$. Retain the sign of $\phi_{\mathrm{cube}}$, with no absolute value or additional reference-charge factor. For the benchmark's negative reference ion ($q_{\mathrm{ref}} = -1$), this quantity is positive. It is distinct from the bulk periodic Madelung constant.

    Return a numpy float64 array $[B, y(p)-B, y(p), \mathrm{rmse}, \mathrm{jackknife\_spread}, \mathrm{madelung}]$.

    Raises:
        ValueError: if an upstream contract fails; p or pz is invalid; fit_sizes has the wrong shape, fewer than six values, a non-finite or non-integer value, a value below four, duplicates, or non-increasing order; or a result is non-finite.
    """
    return None
```

### Step 8

clc_audit

Goal
----
Assembles the audit from all seven earlier steps, cross-checks three aspect-ratio residuals and macroscopic boundary values, and reduces their contamination, instability and curvature to J.

```python
def clc_audit(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, p: int, pz: int, alpha: float, n_real: int, n_rec: int, span: int) -> "np.ndarray":
    r"""Assemble the complete lattice-sum audit from the seven preceding public functions.

    cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec and span follow the earlier contracts. p is an integer at least six and pz is an integer from one through $p-2$. Let $z_{\mathrm{mid}} = \lfloor(p+p_z+1)/2\rfloor$. Evaluate flattened, intermediate and cell-cubic aspect ratios $p_z/p$, $z_{\mathrm{mid}}/p$ and $1$ with fit_sizes $[4,8,12,16,20,24]$.

    For each aspect $i$, call clc_boundary_decomposition and define $q_i = \lvert\mathrm{correction}_i\rvert/(\lvert B_i\rvert+\lvert\mathrm{correction}_i\rvert) + \mathrm{rmse}_i/(\mathrm{jackknife}_i+\mathrm{rmse}_i)$. Define $\mathrm{curvature} = \lvert B_{\mathrm{flat}}-2B_{\mathrm{mid}}+B_{\mathrm{cube}}\rvert/(\lvert B_{\mathrm{flat}}\rvert+\lvert B_{\mathrm{mid}}\rvert+\lvert B_{\mathrm{cube}}\rvert)$. Define $J = \operatorname{median}([q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}}]) + \operatorname{std}([q_{\mathrm{flat}},q_{\mathrm{mid}},q_{\mathrm{cube}}],\mathrm{ddof}=0) + \mathrm{curvature}$. Cross-check every reconstructed residual against both direct-minus-bulk and clc_shape_sequence.

    Use $z_{\mathrm{mid}} = \lfloor(p + p_z + 1)/2\rfloor$ for the intermediate thickness. Run the Step 7 decomposition independently at $p_z$, $z_{\mathrm{mid}}$ and $p$.

    Return a numpy float64 array of shape $(22,)$ in this order: cell-cubic direct potential, flattened direct potential, Ewald bulk potential, flattened minus cell-cubic contrast, nearest-neighbour distance, cell-cubic Madelung constant, spherical direct potential, flattened $B$, flattened correction, flattened reconstructed residual, flattened fit RMSE, flattened jackknife spread, intermediate $B$, cell-cubic $B$, $q_{\mathrm{flat}}$, $q_{\mathrm{mid}}$, $q_{\mathrm{cube}}$, curvature, analytic macroscopic boundary values for flat, intermediate and cell-cubic physical aspect ratios, and $J$.

    Obtain the analytic macroscopic values from column 1 of clc_shape_sequence. They
    are independent source-method checks, not replacements for the prescribed
    finite-sample regression intercepts or for the final $J$ definition.

    The sixth value is the signed finite direct diagnostic $M_{\mathrm{cell}} = d_{\mathrm{nn}}\,\phi_{\mathrm{cube}}$ for the supplied ref_index, following Step 7. Retain the potential's sign, with no absolute value or additional reference-charge factor; the benchmark's value at $q_{\mathrm{ref}} = -1$ is positive.

    The seventh value is clc_direct_potential for the same cell and reference ion with half widths $(p,p,p)$ and spherical=True. Its spherical region is defined in the integer cell-index space by Step 1.

    Raises:
        ValueError: if an earlier contract fails, p is below six, pz is outside one through $p-2$, a cross-step identity fails, a denominator is zero, or a result is non-finite.
    """
    return None
```
