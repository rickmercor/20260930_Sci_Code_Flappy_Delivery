# Mathematics-Numerical_Linear_Algebra-46

## Background

Peridynamic models of solids compute internal forces by integrating bond interactions over a finite horizon, so cracks need no special treatment of discontinuities. In correspondence formulations the deformation gradient at a point is recovered by inverting a kernel-weighted second-moment matrix of its bonds, and any fracture model that removes or weakens bonds therefore also changes how well that matrix can be inverted.

The task below follows one continuous-damage model of this kind through a prescribed separation history and asks how well-conditioned that recovery remains at a point beside the crack.

## Problem

I am testing how well the kinematic approximation holds up next to a crack in a bond-associated correspondence peridynamics model that never deletes bonds: each bond instead carries a continuous damage variable s driven by the history of its maximum-principal-stress crack driving force, and a separate weakening of the bond's weight in the shape tensor switches on only above a threshold damage s_c. My specimen is a lattice of nodes at (i, j, k)Δx with 0 ≤ i ≤ 9, 0 ≤ j ≤ 7, 0 ≤ k ≤ 4 and Δx = 0.4 mm, with horizon δ = 3.015Δx, the model's cubic B-spline kernel, a Saint Venant–Kirchhoff material with E = 32 GPa and ν = 0.25, G_c = 3 J/m² and s_c = 0.95. Starting from an undamaged block, I impose four quasi-static states in which every node with j ≥ 4 is translated rigidly by +(λ/2)(cos 60°, sin 60°, 0) and every node with j ≤ 3 by −(λ/2)(cos 60°, sin 60°, 0), with λ = 0.8, 1.6, 2.4 and 1.0 µm in that order, and I process each state by one pass of the model's force-evaluation algorithm, with every constitutive and damage ingredient as the model defines it. I need the spectral (2-norm) condition number of the degraded shape tensor (the kernel-weighted second-moment matrix of the node's bonds) of node P = (5, 3, 0), assembled from the bond damage values left by the last state, to ten significant figures.

In <reasoning>, state the model's expressions for c_0 (the normalization constant of its bond fracture energy), for Y_c (its critical crack driving force), for the bond-associated deformation gradient, for the bond damage variable and for the bond's weight in the shape tensor (its kinematic weight), and the scalars that fix the answer: c_0 for this kernel and Y_c; the history value, as a multiple of Y_c, above which a bond's kinematic weight drops below one; the largest history value among P's bonds, as a multiple of Y_c; how many of P's bonds end with a kinematic weight below one, and on which side of the separation plane their far ends lie; how the last state changes P's shape tensor; the condition number of P's undamaged shape tensor; the acute angle between the face normal e_z and the final eigenvector of the smallest eigenvalue; the condition number obtained when the whole history is rerun with s_c = 0; and the final condition number. Those expressions and scalars are what determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied lattice data and bulk per-bond tables, which this problem does not need.

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

01_compute_critical_driving_force

Goal
----
Calibrate the bond fracture energy of a spherical-kernel nonlocal model against Griffith's energy release rate and return the resulting normalization constant and critical crack driving force.

```python
def compute_critical_driving_force(
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    fracture_energy: float,
    horizon: float,
) -> tuple:
    """Return the fracture-energy normalization constant and critical driving force.

    The radial kernel profile ``p(rho)``, with ``rho = r / horizon`` in
    ``[0, 1]``, is piecewise polynomial: on the interval
    ``[profile_breaks[j], profile_breaks[j + 1]]`` it equals
    ``sum_m profile_coeffs[j, m] * rho**m``. The kernel is
    ``omega(dX) = C * p(|dX| / horizon)`` for any constant ``C > 0``.

    The bond-wise fracture energy of the body ``Omega`` is

        E_Gamma = int_Omega int_{H(X)} (omega(X' - X) / omega_0)
                  * G_c * s(X, X') / (c_0 * horizon) dV' dV,

    where ``H(X)`` is the ball of radius ``horizon`` around ``X``,
    ``omega_0 = int_{H(X)} omega dV'`` is the kernel integral of a full
    neighbourhood, and the bond phase field ``s`` equals one on every bond
    that crosses a planar crack in an infinite body and zero on every other
    bond. ``c_0`` is the constant for which ``E_Gamma`` equals ``G_c`` per
    unit crack area. The critical crack driving force is
    ``Y_c = G_c / (2 * c_0 * horizon)``.

    Parameters
    ----------
    profile_breaks : np.ndarray
        Strictly increasing 1D array starting at 0 and ending at 1.
    profile_coeffs : np.ndarray
        2D array with one row of ascending-power coefficients per interval.
    fracture_energy : float
        Griffith energy release rate ``G_c`` (positive).
    horizon : float
        Horizon radius ``delta`` (positive).

    Returns
    -------
    tuple
        ``(c_0, Y_c)`` as two floats.

    Raises
    ------
    ValueError
        If the breaks are not a strictly increasing 1D array of finite
        numbers from 0 to 1, if ``profile_coeffs`` is not a finite 2D array
        with one row per interval, if either moment
        ``int_0^1 p(rho) rho**2 drho`` or ``int_0^1 p(rho) rho**3 drho`` is
        not positive, or if ``fracture_energy`` or ``horizon`` is not a
        finite positive number.
    """
    return 0.0, 0.0
```

### Step 2

02_build_bond_family

Goal
----
Build the reference lattice of material points, every ordered bond inside the horizon, and the normalized kernel weight carried by each bond.

```python
def build_bond_family(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
) -> tuple:
    """Return lattice positions, ordered bonds and normalized kernel weights.

    The lattice has ``grid_shape = (N_x, N_y, N_z)`` nodes. Node
    ``(i, j, k)`` has index ``(i * N_y + j) * N_z + k`` and reference
    position ``spacing * (i, j, k)``. Every ordered pair of distinct nodes
    ``(a, b)`` with ``|X_b - X_a| <= horizon`` is a bond; bonds are sorted by
    start index and then by end index. The radial profile ``p(rho)``,
    ``rho = r / horizon``, equals ``sum_m profile_coeffs[j, m] * rho**m`` on
    ``[profile_breaks[j], profile_breaks[j + 1]]``; a radius on an interior
    break uses the interval to its right. The bond weight is
    ``omega = p(|X_b - X_a| / horizon) / (4 * pi * horizon**3 * M2)`` with
    ``M2 = int_0^1 p(rho) rho**2 drho``, so that the kernel integrates to one
    over the full ball of radius ``horizon``.

    Parameters
    ----------
    grid_shape : tuple
        Three positive integers ``(N_x, N_y, N_z)``.
    spacing : float
        Positive lattice spacing.
    horizon : float
        Horizon radius, at least ``spacing``.
    profile_breaks : np.ndarray
        Strictly increasing 1D array starting at 0 and ending at 1.
    profile_coeffs : np.ndarray
        2D array with one row of ascending-power coefficients per interval.

    Returns
    -------
    tuple
        ``(positions, bond_start, bond_end, bond_weight)``: an
        ``(N, 3)`` float array, two ``(N_b,)`` integer arrays and an
        ``(N_b,)`` float array.

    Raises
    ------
    ValueError
        If ``grid_shape`` is not three positive integers, if ``spacing`` is
        not a finite positive number, if ``horizon`` is not finite or is
        smaller than ``spacing``, if the breaks are not a strictly
        increasing 1D array of finite numbers from 0 to 1, if
        ``profile_coeffs`` is not a finite 2D array with one row per
        interval, or if ``M2`` is not positive.
    """
    return positions, bond_start, bond_end, bond_weight
```

### Step 3

03_assemble_nodal_deformation_gradients

Goal
----
Recover nodal deformation gradients by degraded reproducing-kernel fits on truncated neighborhoods, with selectable polynomial enrichment and nonuniform quadrature volumes.

```python
def assemble_nodal_deformation_gradients(
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
    bond_weight: "np.ndarray",
    kinematic_weight: "np.ndarray",
    nodal_volume: "float | np.ndarray",
    basis: str = "C1",
) -> tuple:
    """Return physical shape tensors and derivatives of weighted local fits.

    For bond ``b`` from node ``k = bond_start[b]`` to ``n = bond_end[b]``
    let ``dX_b = X_n - X_k`` and ``dx_b = x_n - x_k`` (reference and current
    positions). Let ``V_n`` be the volume of the END node and
    ``c_b = bond_weight[b] * kinematic_weight[b] * V_n``. For every basis,
    return the physical 3x3 tensor ``K_k = sum_b c_b dX_b dX_b^T``.

    At each node, fit the three components of ``dx_b`` by polynomials of
    ``dX_b`` that minimize the weighted sum of squared residuals. ``C1``
    uses the span of ``(X, Y, Z)``; ``RK1`` uses ``(1, X, Y, Z)``; ``RK2``
    uses ``(1, X, Y, Z, X**2, Y**2, Z**2, X*Y, X*Z, Y*Z)``. Return the
    derivative of the fitted vector polynomial at ``dX = 0`` as ``F_k``;
    the constant and quadratic coefficients participate in the fit but
    are not themselves returned. Rows of ``F_k`` index current components
    and columns index reference derivatives. Rank of each node's weighted
    polynomial fit is judged on the dimensionless monomials obtained by
    dividing ``dX_b`` by that node's largest bond length. Implementations
    must rescale before the rank test. Physical-unit conditioning of the
    unscaled basis must not trigger the rank error. Returned derivatives
    must use physical units. The shape tensor is always 3x3, even for
    enriched fits.

    Parameters
    ----------
    ref_positions : np.ndarray
        ``(N, 3)`` reference positions.
    cur_positions : np.ndarray
        ``(N, 3)`` current positions.
    bond_start, bond_end : np.ndarray
        ``(N_b,)`` integer node indices of each bond.
    bond_weight : np.ndarray
        ``(N_b,)`` nonnegative kernel weights.
    kinematic_weight : np.ndarray
        ``(N_b,)`` nonnegative kinematic weights.
    nodal_volume : float or np.ndarray
        Positive common volume, or a positive finite ``(N,)`` array.
    basis : str
        One of ``"C1"``, ``"RK1"``, ``"RK2"``. Defaults to correspondence.

    Returns
    -------
    tuple
        ``(shape_tensors, nodal_gradients)``, two ``(N, 3, 3)`` arrays.

    Raises
    ------
    ValueError
        If the position arrays are not finite ``(N, 3)`` arrays of equal
        shape, if the bond arrays do not share one 1D length, if an index
        lies outside ``[0, N)``, if a weight is negative or not finite, if
        volumes have an invalid shape or nonpositive/nonfinite entries,
        ``basis`` is unsupported, or a node's weighted polynomial fit
        lacks full column rank on that dimensionless basis. Inputs must
        not be modified.
    """
    return shape_tensors, nodal_gradients
```

### Step 4

04_compute_bond_deformation_gradients

Goal
----
Build the bond-associated deformation gradient of every bond from the two nodal gradients at its ends and the bond's own current image.

```python
def compute_bond_deformation_gradients(
    nodal_gradients: "np.ndarray",
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
) -> "np.ndarray":
    """Return the bond-associated deformation gradient of every bond.

    For bond ``b`` from node ``k = bond_start[b]`` to ``n = bond_end[b]``
    let ``dX_b = X_n - X_k`` and ``dx_b = x_n - x_k`` and let
    ``F_avg = (F_k + F_n) / 2`` be the mean of the two nodal gradients. The
    bond gradient ``F_b`` is the unique 3x3 matrix that agrees with
    ``F_avg`` on every vector orthogonal to ``dX_b`` and maps ``dX_b``
    exactly onto ``dx_b``.

    Parameters
    ----------
    nodal_gradients : np.ndarray
        ``(N, 3, 3)`` nodal deformation gradients.
    ref_positions : np.ndarray
        ``(N, 3)`` reference positions.
    cur_positions : np.ndarray
        ``(N, 3)`` current positions.
    bond_start, bond_end : np.ndarray
        ``(N_b,)`` integer node indices of each bond.

    Returns
    -------
    np.ndarray
        ``(N_b, 3, 3)`` bond deformation gradients.

    Raises
    ------
    ValueError
        If the arrays are not finite with shapes ``(N, 3, 3)``, ``(N, 3)``
        and ``(N, 3)``, if the bond index arrays do not share one 1D length,
        if an index lies outside ``[0, N)``, or if a bond has zero reference
        length.
    """
    return bond_gradients
```

### Step 5

05_compute_crack_driving_force

Goal
----
Evaluate the principal-stress crack driving force of every bond from its bond-associated deformation gradient for a Saint Venant-Kirchhoff material.

```python
def compute_crack_driving_force(
    bond_gradients: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
) -> "np.ndarray":
    """Return the crack driving force of every bond.

    For each bond gradient ``F`` the Green-Lagrange strain is
    ``E = (F^T F - I) / 2`` and the Saint Venant-Kirchhoff second
    Piola-Kirchhoff stress is ``S = lam * tr(E) * I + 2 * mu * E`` with
    ``lam = Y_mod * nu / ((1 + nu) * (1 - 2 * nu))`` and
    ``mu = Y_mod / (2 * (1 + nu))``. The Cauchy stress is
    ``sigma = F S F^T / det(F)`` and ``sigma_1`` is its largest eigenvalue.
    The driving force is ``max(sigma_1, 0)**2 / (2 * Y_mod)``.

    Parameters
    ----------
    bond_gradients : np.ndarray
        ``(N_b, 3, 3)`` bond deformation gradients.
    youngs_modulus : float
        Positive Young's modulus ``Y_mod``.
    poisson_ratio : float
        Poisson's ratio ``nu`` in ``(-1, 0.5)``.

    Returns
    -------
    np.ndarray
        ``(N_b,)`` nonnegative crack driving forces.

    Raises
    ------
    ValueError
        If ``bond_gradients`` is not a finite ``(N_b, 3, 3)`` array, if any
        gradient has a nonpositive determinant, if ``youngs_modulus`` is not
        a finite positive number, or if ``poisson_ratio`` is not a finite
        number in ``(-1, 0.5)``.
    """
    return driving_force
```

### Step 6

06_update_bond_phase_field

Goal
----
Advance each bond's history variable, evaluate its closed-form phase field, and convert the phase field into the delayed kinematic weight used by the shape tensor.

```python
def update_bond_phase_field(
    driving_force: "np.ndarray",
    history: "np.ndarray",
    critical_driving_force: float,
    kinematic_threshold: float,
) -> tuple:
    """Return the updated history, bond phase field and kinematic weight.

    The new history is ``H = max(history, driving_force)`` bond by bond. The
    bond phase field is ``s = min(1, H / (H + Y_c))`` with
    ``Y_c = critical_driving_force``. The kinematic weight is ``1`` where
    ``s <= s_c`` and ``((1 - s) / (1 - s_c))**2`` where ``s > s_c``, with
    ``s_c = kinematic_threshold``.

    Parameters
    ----------
    driving_force : np.ndarray
        ``(N_b,)`` nonnegative crack driving forces of the current state.
    history : np.ndarray
        ``(N_b,)`` nonnegative history values left by the previous state.
    critical_driving_force : float
        Positive critical crack driving force ``Y_c``.
    kinematic_threshold : float
        Threshold phase field ``s_c`` in ``[0, 1)``.

    Returns
    -------
    tuple
        ``(new_history, phase_field, kinematic_weight)``, three ``(N_b,)``
        float arrays.

    Raises
    ------
    ValueError
        If ``driving_force`` and ``history`` are not 1D arrays of equal
        length holding finite nonnegative numbers, if
        ``critical_driving_force`` is not a finite positive number, or if
        ``kinematic_threshold`` is not a finite number in ``[0, 1)``.
    """
    return new_history, phase_field, kinematic_weight
```

### Step 7

07_run_separation_history

Goal
----
Drive a lattice block through a sequence of rigid mixed-mode separations of its two halves and return every bond's history, phase field and kinematic weight after the last state.

```python
def run_separation_history(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
    separation_layer: int,
    slip_angle: float,
    openings: "np.ndarray",
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> tuple:
    """Return every bond's history, phase field and kinematic weight after the last state.

    Nodes, positions, bonds and kernel weights follow ``build_bond_family``;
    node ``(i, j, k)`` lies above the separation plane when
    ``j > separation_layer``. In state ``m`` every node above the plane sits
    at ``X + (lam_m / 2) d`` and every node below at ``X - (lam_m / 2) d``,
    with ``lam_m = openings[m]`` and
    ``d = (cos theta, sin theta, 0)``, ``theta = slip_angle`` in degrees.
    Alternatively, ``openings`` can have shape ``(T, 3)``. Each row holds
    signed components in the orthonormal frame ``(d, t, e_z)``, where
    ``t = (-sin theta, cos theta, 0)``. Its global relative translation
    replaces ``lam_m * d``. Rows are absolute states relative to the
    reference geometry, not increments. This permits changing mode mix,
    unloading and reversed shear with an out-of-plane component.

    Initial bond histories are zero unless ``initial_history`` is supplied;
    derive their initial phase fields and kinematic weights using the same
    constitutive law before evaluating the first state's kinematics. The
    states are
    processed in order, each by one pass of:

    1. nodal gradients from ``assemble_nodal_deformation_gradients`` with
       the kinematic weights left by the previous state and nodal volume
       ``spacing**3 * volume_factors`` (unit factors by default) and the
       selected polynomial ``basis``;
    2. bond gradients from ``compute_bond_deformation_gradients``;
    3. driving forces from ``compute_crack_driving_force``;
    4. history, phase field and kinematic weight from
       ``update_bond_phase_field``, with ``Y_c`` taken from
       ``compute_critical_driving_force``.

    Parameters
    ----------
    grid_shape : tuple
        Lattice size ``(N_x, N_y, N_z)``.
    spacing, horizon : float
        Lattice spacing and horizon radius.
    profile_breaks, profile_coeffs : np.ndarray
        Piecewise-polynomial kernel profile, as in ``build_bond_family``.
    youngs_modulus, poisson_ratio : float
        Saint Venant-Kirchhoff elastic constants.
    fracture_energy : float
        Griffith energy release rate ``G_c``.
    kinematic_threshold : float
        Threshold phase field ``s_c`` in ``[0, 1)``.
    separation_layer : int
        Last layer index ``j`` below the separation plane.
    slip_angle : float
        Angle of the separation direction from the x axis, in degrees.
    openings : np.ndarray
        Nonempty nonnegative ``(T,)`` magnitudes, or finite signed
        ``(T, 3)`` local-frame components, one row per state.
    basis : str
        ``"C1"``, ``"RK1"``, or ``"RK2"`` as in the nodal reconstruction.
    volume_factors : np.ndarray or None
        Positive finite ``(N,)`` quadrature multipliers in lattice node
        order; defaults to one at every node.
    initial_history : np.ndarray or None
        Finite nonnegative ``(N_b,)`` history in the directed bond order
        of ``build_bond_family``. Enables restarting a loading path.
        All inputs are read-only; no state is retained between calls.

    Returns
    -------
    tuple
        ``(history, phase_field, kinematic_weight)``, three ``(N_b,)``
        float arrays in the bond order of ``build_bond_family``.

    Raises
    ------
    ValueError
        If ``openings`` has neither supported shape, is empty/nonfinite,
        or has negative entries in the 1D form; if the supplied volumes or
        history violate their contracts; if ``separation_layer`` is not an integer with
        ``0 <= separation_layer <= N_y - 2``, if ``slip_angle`` is not a
        finite number, or if any earlier step rejects its input.
    """
    return history, phase_field, kinematic_weight
```

### Step 8

08_estimate_flank_conditioning

Goal
----
Compose every earlier step to obtain the spectral condition number of the kinematically degraded shape tensor at a probe node after a separation history.

```python
def estimate_flank_conditioning(
    grid_shape: tuple = (10, 8, 5),
    spacing: float = 4.0e-4,
    horizon: float = 1.206e-3,
    profile_breaks: tuple = (0.0, 0.5, 1.0),
    profile_coeffs: tuple = ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)),
    youngs_modulus: float = 3.2e10,
    poisson_ratio: float = 0.25,
    fracture_energy: float = 3.0,
    kinematic_threshold: float = 0.95,
    separation_layer: int = 3,
    slip_angle: float = 60.0,
    openings: "tuple | np.ndarray" = (8.0e-7, 1.6e-6, 2.4e-6, 1.0e-6),
    probe_node: tuple = (5, 3, 0),
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> float:
    """Return the spectral condition number of the probe node's degraded shape tensor.

    The separation history is run as in ``run_separation_history``. The
    probe node's shape tensor is then assembled, as in
    ``assemble_nodal_deformation_gradients`` with neighbor volumes
    ``spacing**3 * volume_factors`` and the selected ``basis``, from the
    kinematic weights left by the last state. The requested tensor is the
    physical 3x3 shape tensor for every basis, not the enriched moment
    matrix. Return
    its 2-norm condition number (largest over smallest eigenvalue). The
    defaults reproduce the problem statement.

    Parameters
    ----------
    grid_shape : tuple
        Lattice size ``(N_x, N_y, N_z)``.
    spacing, horizon : float
        Lattice spacing and horizon radius.
    profile_breaks, profile_coeffs : tuple
        Piecewise-polynomial kernel profile, as in ``build_bond_family``.
    youngs_modulus, poisson_ratio : float
        Saint Venant-Kirchhoff elastic constants.
    fracture_energy : float
        Griffith energy release rate ``G_c``.
    kinematic_threshold : float
        Threshold phase field ``s_c`` in ``[0, 1)``.
    separation_layer : int
        Last layer index ``j`` below the separation plane.
    slip_angle : float
        Angle of the separation direction from the x axis, in degrees.
    openings : tuple
        Nonempty nonnegative ``(T,)`` magnitudes or signed ``(T, 3)``
        absolute relative translations in the local slip frame, as in
        ``run_separation_history``.
    probe_node : tuple
        Lattice indices ``(i, j, k)`` of the probe node.
    basis : str
        ``"C1"``, ``"RK1"``, or ``"RK2"`` for the nodal reconstruction.
    volume_factors : np.ndarray or None
        Positive finite ``(N,)`` quadrature multipliers, defaulting to one.
    initial_history : np.ndarray or None
        Initial bond histories in directed bond order, defaulting to zero.
        See ``run_separation_history``. Inputs must not be modified.

    Returns
    -------
    float
        The condition number of the probe node's degraded shape tensor.

    Raises
    ------
    ValueError
        If ``probe_node`` is not three integers inside ``grid_shape``, or if
        any earlier step rejects its input.
    """
    return 0.0
```
