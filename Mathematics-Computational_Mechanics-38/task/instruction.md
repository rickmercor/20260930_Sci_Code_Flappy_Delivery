# Mathematics-Computational_Mechanics-38

## Background

Peridynamics replaces the local balance of linear momentum by an integro-differential equation: each material point interacts with all neighbors inside a finite horizon, and no spatial derivatives of the displacement field appear. The governing equations therefore remain valid across discontinuities, which makes the theory naturally suited to fracture. The correspondence variant approximates a nonlocal deformation gradient by kernel-weighted averaging over the family, so that classical finite-strain constitutive models (here Saint Venant-Kirchhoff) can be used directly. Plain correspondence is polluted by zero-energy oscillation modes; computing deformation gradients bond-wise at quadrature points located on the bonds suppresses these modes and is currently the most accurate correspondence-type formulation, reproducing homogeneous deformations exactly even near boundaries.

Fracture in peridynamics is classically modeled by irreversibly deleting bonds that exceed a critical deformation. In correspondence-type formulations this is dangerous for a structural reason: the same kernel sums that carry the constitutive response also build the point-wise geometric moment operators whose inverses define the kinematic approximation. Progressive bond removal thins out those sums, can drive the moment operators toward singularity, and thereby destroys the accuracy and stability of the kinematics precisely where the crack needs them most. This motivates fracture treatments that avoid deleting bonds altogether and instead degrade bond contributions smoothly - while still being careful about what such degradation does to the kinematic operators, since energetic and kinematic roles of a bond need not be treated identically.

Phase-field fracture regularizes Griffith's theory: a continuous damage variable replaces the sharp crack, the stored energy of damaged material is reduced by a degradation function, and crack surface energy is represented by a damage-dependent density whose overall scale must be normalized so that total dissipation for a fully formed crack equals the critical energy release rate Gc times the crack area. In local models this normalization comes from the optimal one-dimensional damage profile of the regularized functional. In a bond-based nonlocal setting there is no such profile; the dissipation of a sharp crack is instead carried by the population of bonds that cross the crack surface, weighted by the kernel, so the correct normalization depends on the chosen kernel and must be derived within the nonlocal theory itself. Getting this constant right is what makes fracture behavior independent of the kernel choice and horizon-to-spacing ratio.

For damage initiation, stress-based criteria are attractive in engineering practice because design rules are formulated in terms of strength: a crack should start growing where a suitable measure of tensile stress reaches a critical value tied to Gc, while compressive states should not drive damage. Under impact-like loading the problem is genuinely dynamic: elastic waves emitted from the loaded boundaries interact with the notch, damage grows irreversibly along the loading history, and an explicit time integrator advances the coupled deformation-damage state. Small explicit steps keep the wave dynamics stable and resolve the progressive, history-dependent accumulation of damage that a quasi-static treatment would miss.

Boundary effects matter in nonlocal models: points near free surfaces have truncated families, so kernel sums that would be exact in the bulk lose mass near the boundary. Formulations that explicitly account for the per-point kernel content of truncated families avoid the systematic surface error that otherwise contaminates thin or small specimens - such as the deliberately small, thin slab used in this benchmark, where every point is boundary-affected in at least one direction.

## Problem

A recent line of work embeds a variational phase-field description of brittle fracture directly into the bond-associated (quadrature-point) correspondence formulation of peridynamics, replacing irreversible bond deletion by a continuous, thermodynamically consistent degradation of every bond. Reproduce that method's explicit dynamic-fracture pipeline on the bespoke miniature benchmark frozen below and return one scalar.

The following must be determined from the source and implemented exactly, none of it is restated here: (i) the normalization of the damage model that makes total crack dissipation consistent with Griffith's energy release rate for the kernel specified below, and the resulting critical crack-driving-force threshold; (ii) the crack-surface-density and stress-degradation functions the method adopts; (iii) how, and through which separate function, damage enters the kinematic (moment-matrix / shape-function) side of the correspondence formulation, including the role played by the critical phase-field value given below; (iv) the bond-level closed-form evolution of the bond phase-field from its history variable and how irreversibility is enforced; (v) the exact form of the stress-based crack driving force (which stress tensor, which part of it, how it is scaled); (vi) the bond-associated construction of the bond deformation gradient and the two-term assembly of the force state including the non-uniform stress contribution; (vii) the truncation-corrected bond-kernel weighting used in family averages; and (viii) the order in which kinematic quantities, driving force, history, phase-field, and degraded stress are updated within one internal-force evaluation.

Frozen benchmark configuration (exact, 0-based indices):
- Point cloud: uniform Cartesian grid, nx x ny x nz = 16 x 10 x 2 points, spacing dx = 1.0e-3 m; point (i,j,k) sits at X = ((i+1/2)dx, (j+1/2)dx, (k+1/2)dx); every point carries volume V = dx^3. Components x/y/z map to indices i/j/k.
- Horizon: delta = 2.015e-3 m. The family of a point contains every other point with 0 < |DX| <= delta, DX measured in the reference configuration.
- Influence function (cubic B-spline, q = |DX|/delta):
  omega(DX) = 8/(pi delta^3) * (1 - 6 q^2 + 6 q^3)   for q <= 1/2,
  omega(DX) = 8/(pi delta^3) * 2 (1 - q)^3           for 1/2 < q <= 1,
  omega(DX) = 0                                       for q > 1.
- Material: Saint Venant-Kirchhoff with lambda, mu from E = 32.0e9 Pa and nu = 0.25; rho0 = 2450.0 kg/m^3; Gc = 120.0 J/m^2; critical phase-field value s_c = 0.95.
- Pre-notch: every bond whose two endpoints lie in rows j <= ny/2 - 1 and j >= ny/2 respectively (here: j <= 4 and j >= 5, i.e. the bond crosses the mid-plane) AND whose endpoints both have i <= nx/2 - 1 (here: i <= 7) is fully pre-damaged: its bond phase-field equals 1 at t = 0 and must remain 1 for all time. Realize this by initializing that bond's damage-history variable to 1.0e30 J/m^3. No bond is removed from any family.
- Loading: rows j = 0 and j = ny-1 (here: j = 9) are Dirichlet velocity layers: v = (0, -v0, 0) on j = 0 and v = (0, +v0, 0) on j = ny-1 for all t >= 0, with v0 = 1.2 m/s; their displacement is exactly u_y = -v0 t resp. +v0 t (zero x,z components) and their acceleration is treated as zero. All other points start at rest with zero displacement. No external body force acts.
- No-fail zone: bonds with at least one endpoint in row j = 0 or j = ny-1 never accumulate damage (their crack driving force is treated as zero at all times); they transmit force normally.
- Time integration: velocity Verlet with dt = 2.0e-8 s for n_steps = 360 steps (T = 7.2e-6 s). Per step: positions update first (u <- u + dt v + dt^2 a / 2, then the prescribed layers are overwritten with their exact positions), then exactly one internal-force evaluation updates the bond state and yields the new acceleration, then velocities update (v <- v + dt (a_old + a_new)/2, prescribed layers overwritten with their prescribed velocity). One additional force evaluation of the initial state initializes the acceleration before the first step.

Benchmark conventions (fixed by this benchmark; these are not parameters of the source): float64 arithmetic throughout; no randomness anywhere; family sums use the plain one-point quadrature sum_n (.) V_n over the family as-is, with no partial-volume or surface correction beyond what the method itself prescribes; any one-dimensional kernel integrals over the unit interval must be evaluated exactly or with error below 1e-12 (e.g. high-order Gaussian quadrature); matrix inversions in double precision to machine accuracy; where the source offers both an energy-density-based and a maximum-principal-stress-based crack driving force, use the maximum-principal-stress-based one; report the final scalar to at least 6 significant figures.

Do not substitute normalization constants, surface-density functions, or evolution laws from other phase-field fracture formulations - derive the ones this method requires for the kernel specified above. Treat the pre-notch exactly as specified (a fully damaged initial bond state), not by any other mechanism. Do not add artificial damping, viscosity, contact, or any time-step adaptation. Do not hard-code intermediate results: every reported quantity must be produced by your run.

In your reasoning, report the following quantities from your run alongside the final answer:
1. the damage-model normalization constant for the specified kernel (dimensionless);
2. the critical crack-driving-force threshold (J/m^3);
3. the number of unordered bond pairs in the discretized body, and how many of them are pre-notched (two integers);
4. the mean over all points of the discrete kernel integral of the family (dimensionless);
5. the smallest eigenvalue, over the two point rows adjacent to the crack mid-plane (j = 4 and j = 5), of the kinematically degraded moment matrix (shape tensor) of the pre-notched initial state (m^2);
6. the total crack dissipation energy at t = 0 (J);
7. the total kinetic energy sum_k (1/2) rho0 |v_k|^2 V over all points at t = T (J);
8. the maximum point damage over all points at t = T, with point damage obtained from the bond phase-fields exactly as the method defines it (dimensionless);
9. the number of non-pre-notched unordered bond pairs whose bond phase-field exceeds 0.5 at t = T (integer);
10. the final answer itself: the crack dissipation growth dE = E_Gamma(T) - E_Gamma(0) in joules, where E_Gamma is the method's total crack dissipation energy of the body (so the pre-notch contribution present at t = 0 is excluded).

Return the crack dissipation growth dE (in J) produced by this deterministic benchmark.

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

step-01-bond-families

Goal
----
Constructs the geometric substrate that every later step consumes: the uniform Cartesian point cloud, the reference-configuration bond families within the horizon, the benchmark's normalized cubic B-spline influence function on every bond, the per-point discrete kernel integral, the method's truncation-corrected averaged bond kernel used in every family average, and the benchmark's pre-notch and no-fail bond masks. Returns one dictionary of arrays (points, bond pairs, kernel weights, masks) that steps 3, 4, 7, 8 and 9 take as input. Validates grid placement, the family radius rule, the kernel, the treatment of truncated families, and the bond bookkeeping. Deliberately excluded: damage, deformation, constitutive response, and dynamics.

```python
import numpy as np


def build_families(nx: int, ny: int, nz: int, dx: float, delta: float) -> dict:
    """Point cloud, bond families, kernel weights and bond masks of the benchmark.

    Builds the uniform Cartesian point cloud with points at
    ((i+1/2)*dx, (j+1/2)*dx, (k+1/2)*dx) for 0 <= i < nx, 0 <= j < ny,
    0 <= k < nz (0-based indices i/j/k along x/y/z), each carrying volume
    V = dx**3, forms every bond family {X' : 0 < |X'-X| <= delta} in the
    reference configuration, evaluates the benchmark's normalized cubic
    B-spline influence function on every bond, accumulates the per-point
    discrete kernel integral omega0(X_k) = sum_n omega(|DX_kn|) * V_n,
    forms the method's truncation-corrected averaged bond kernel omega_b
    of every bond pair (the weighting the source uses in every family
    average; its form must be taken from the source), and tags the
    benchmark's pre-notch bond pairs (endpoints in rows j <= ny/2-1 and
    j >= ny/2 with both endpoint columns i <= nx/2-1) and no-fail bond
    pairs (at least one endpoint in row j = 0 or j = ny-1). This
    dictionary is the single geometric substrate consumed by every later
    step.

    Parameters
    ----------
    nx : int
        Number of grid points along x (>= 1).
    ny : int
        Number of grid points along y (>= 1).
    nz : int
        Number of grid points along z (>= 1).
    dx : float
        Grid spacing in m (> 0).
    delta : float
        Horizon radius in m (> dx).

    Returns
    -------
    dict
        Keys (N = nx*ny*nz points, P = number of unordered bond pairs, each
        pair listed exactly once with endpoints pk[p] and pn[p]):
        'nx', 'ny', 'nz' (int), 'dx', 'delta', 'V' (float, V = dx**3);
        'ijk' : (N, 3) int array of grid indices (i, j, k) of every point;
        'X' : (N, 3) float array of reference positions in m;
        'pk', 'pn' : (P,) int arrays of the two endpoint indices of every pair;
        'dX' : (P, 3) float array X[pn] - X[pk] in m;
        'r' : (P,) float array of bond lengths |dX| in m;
        'omega' : (P,) float array of the cubic B-spline kernel omega(|dX|) in 1/m^3;
        'omega0' : (N,) float array of the discrete kernel integral (dimensionless);
        'omega_b' : (P,) float array of the truncation-corrected averaged bond kernel in 1/m^3;
        'notch' : (P,) bool array, True for pre-notch bond pairs;
        'nofail' : (P,) bool array, True for no-fail bond pairs.

    Raises
    ------
    ValueError
        If a grid dimension is not a positive integer, dx <= 0, delta <= dx,
        or some point has an empty family.
    """
    return {}
```

### Step 2

step-02-normalization-constant

Goal
----
Computes the Griffith-consistency normalization constant of the bond-based phase-field damage model for a named normalized spherical kernel ('cubic', 'linear', or 'constant'), following the source's derivation, with any one-dimensional kernel integrals evaluated to better than 1e-12. Validates the normalization that every later damage quantity inherits (the critical threshold of step 6 and the dissipation functional of step 9). Deliberately excluded: any mesh, deformation, or dynamics - the constant is a pure kernel functional.

```python
import numpy as np


def pfpd_normalization_constant(kernel: str) -> float:
    """Griffith-consistency normalization constant of the damage model.

    Evaluates, for a named normalized spherical kernel, the normalization
    constant of the bond-based phase-field damage model of the source: the
    constant that makes the total crack dissipation of a fully formed
    crack equal to the Griffith energy release rate per unit crack area
    in this nonlocal setting. The constant is a pure functional of the
    kernel shape; its construction must follow the source, and any
    one-dimensional kernel integrals over the unit interval must be
    evaluated exactly or with error below 1e-12. The returned value is
    consumed by step 6 (critical threshold), passed through steps 7 and
    8, and used by the orchestrator of step 9.

    Parameters
    ----------
    kernel : str
        One of 'cubic' (normalized cubic B-spline, shape
        f(q) = 1 - 6 q^2 + 6 q^3 for q <= 1/2 and 2 (1 - q)^3 for
        1/2 < q <= 1), 'linear' (normalized linear kernel, shape 1 - q on
        q <= 1), or 'constant' (normalized constant kernel, shape 1 on
        q <= 1), with q = |DX| / delta and zero beyond q = 1.

    Returns
    -------
    float
        The normalization constant c0 (dimensionless).

    Raises
    ------
    ValueError
        If the kernel name is not one of the three supported strings.
    """
    return 0.0
```

### Step 3

step-03-kinematic-operators

Goal
----
Builds, for a given bond phase-field state on the families of step 1, the kinematically degraded moment matrix (shape tensor) of every point and the shape-function derivatives of both endpoints of every bond pair. Validates the method's separate kinematic degradation function, the role of the critical phase-field value s_c in it, and the way it enters the kinematic operators; the returned operators are the kinematic input of the bond deformation gradients (step 4) and of every internal-force evaluation (step 7). Deliberately excluded: stress, driving force, and evolution.

```python
import numpy as np


def kinematic_operators(fam: dict, s: np.ndarray, s_c: float) -> dict:
    """Kinematically degraded moment matrices and shape-function derivatives.

    Given the bond families of step 1 and one bond phase-field value per
    bond pair, applies the method's kinematic degradation - the separate
    function through which damage enters the kinematic side of the
    correspondence formulation, in which the critical phase-field value
    s_c plays the role the source assigns to it - and builds the moment
    matrix (shape tensor) of every point and the shape-function
    derivatives of both endpoints of every bond pair exactly as the source
    prescribes. Which function is used, and how it enters the moment
    matrix and the shape-function derivatives, must be taken from the
    source. These operators are consumed by the bond deformation
    gradients of step 4 and by every internal-force evaluation of step 7.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    s : np.ndarray
        (P,) array of bond phase-field values in [0, 1], one per bond pair,
        in the pair order of fam['pk'] / fam['pn'].
    s_c : float
        Critical phase-field value in (0, 1).

    Returns
    -------
    dict
        'h' : (P,) float array, kinematic degradation factor of every bond
            pair as the method defines it;
        'M' : (N, 3, 3) float array, degraded moment matrix of every point in m^2;
        'dphi_kn' : (P, 3) float array, shape-function derivative of endpoint
            pn[p] with respect to the family of endpoint pk[p], in 1/m;
        'dphi_nk' : (P, 3) float array, shape-function derivative of endpoint
            pk[p] with respect to the family of endpoint pn[p], in 1/m.

    Raises
    ------
    ValueError
        If s_c is not in (0, 1), or s does not hold one value in [0, 1] per
        bond pair.
    """
    return {}
```

### Step 4

step-04-bond-deformation-gradients

Goal
----
Computes the nonlocal deformation gradient of every point from the shape-function derivatives of step 3 and from it the bond-associated (quadrature-point) deformation gradient of every bond pair for a given displacement field on the families of step 1. Validates the bond-level construction of the framework the source builds on, which reproduces homogeneous deformations exactly; the returned (P, 3, 3) array feeds the stress evaluation of step 5 inside every internal-force evaluation of step 7. Deliberately excluded: damage and constitutive response.

```python
import numpy as np


def bond_deformation_gradients(fam: dict, ops: dict, u: np.ndarray) -> np.ndarray:
    """Bond-associated deformation gradient of every bond pair.

    From the bond families of step 1, the kinematic operators of step 3
    and a displacement field u, computes the nonlocal deformation gradient
    of every point and from it the bond-associated (quadrature-point)
    deformation gradient of every bond pair, exactly as the bond-associated
    correspondence framework the source builds on prescribes; the
    construction must be taken from that framework. It reproduces
    homogeneous deformations exactly. The returned gradients are consumed
    by the stress evaluation of step 5 inside every internal-force
    evaluation of step 7.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    ops : dict
        Kinematic operators returned by kinematic_operators (step 3) for
        the current bond phase-field state.
    u : np.ndarray
        (N, 3) displacement of every point in m.

    Returns
    -------
    np.ndarray
        (P, 3, 3) array holding the bond-associated deformation gradient of
        every bond pair (dimensionless), in the pair order of fam['pk'] /
        fam['pn'].

    Raises
    ------
    ValueError
        If u is not an (N, 3) array.
    """
    return np.zeros((0, 3, 3))
```

### Step 5

step-05-bond-stress-driving-force

Goal
----
Evaluates the Saint Venant-Kirchhoff response of one or many bond deformation gradients and returns both the undamaged first Piola-Kirchhoff stress and the benchmark's stress-based crack driving force in the form the source adopts. Validates the constitutive law, the choice of stress measure for the driving force, and its scaling. Both outputs are consumed by every internal-force evaluation of step 7. Deliberately excluded: mesh, families, and evolution - this is a pure pointwise constitutive functional.

```python
import numpy as np


def bond_stress_and_driving_force(F: np.ndarray, E: float, nu: float) -> dict:
    """Undamaged bond stress and stress-based crack driving force.

    Evaluates the Saint Venant-Kirchhoff response of one or many bond
    deformation gradients: the undamaged first Piola-Kirchhoff stress
    P0 = F S with S = lambda tr(E_G) I + 2 mu E_G, E_G = (F^T F - I)/2
    (the benchmark material), and the benchmark's stress-based crack
    driving force in the exact form the source adopts for the
    maximum-principal-stress-based variant that the benchmark selects.
    Which stress tensor is used, which part of it, and how it is scaled
    to a driving force must be taken from the source. Both outputs are
    consumed by every internal-force evaluation of step 7.

    Parameters
    ----------
    F : np.ndarray
        Bond deformation gradient(s), shape (3, 3) for a single bond or
        (P, 3, 3) for a batch (dimensionless), with positive determinant.
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio, in (-1, 0.5).

    Returns
    -------
    dict
        'P0' : undamaged first Piola-Kirchhoff stress in Pa, shape (3, 3)
            for a single bond or (P, 3, 3) for a batch;
        'Y' : crack driving force in J/m^3, a float for a single bond or a
            (P,) array for a batch.

    Raises
    ------
    ValueError
        If F does not end in a 3x3 block, any det(F) <= 0, E <= 0, or nu
        is outside (-1, 0.5).
    """
    return {}
```

### Step 6

step-06-bond-phase-field

Goal
----
Applies the method's closed-form bond-level phase-field evolution to the damage-history variable, with the critical crack-driving-force threshold derived from Gc, the horizon, and the normalization constant produced by step 2. Works elementwise on arrays of bond histories. Validates the evolution law and the threshold it pivots on; the returned bond phase-field is consumed by every internal-force evaluation of step 7, by the initial state of the dynamics of step 8, and by the orchestrator of step 9. Deliberately excluded: how the history variable is accumulated in time.

```python
import numpy as np


def bond_phase_field(calY: "float | np.ndarray", Gc: float, delta: float,
                     c0: float) -> "float | np.ndarray":
    """Bond phase-field value(s) from the damage-history variable.

    Applies the method's closed-form bond-level evolution law to the bond
    damage-history variable, with the critical crack-driving-force
    threshold derived from Gc, the horizon delta and the Griffith-
    consistency normalization constant c0 of the kernel (the output of
    step 2) as the source prescribes. The law and the threshold must be
    taken from the source; irreversibility is carried by the history
    variable that the caller supplies. Works elementwise on arrays so that
    it can be applied to every bond pair at once; it is consumed by every
    internal-force evaluation of step 7, by the initial state of the
    dynamics of step 8, and by the orchestrator of step 9.

    Parameters
    ----------
    calY : float or np.ndarray
        Bond damage-history variable(s) in J/m^3 (>= 0).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    delta : float
        Horizon radius in m (> 0).
    c0 : float
        Griffith-consistency normalization constant of the damage model
        for the kernel in use (dimensionless, in (0, 1)).

    Returns
    -------
    float or np.ndarray
        Bond phase-field value(s) in [0, 1]: a float for scalar input, an
        array of the same shape for array input.

    Raises
    ------
    ValueError
        If any calY < 0, Gc <= 0, delta <= 0, or c0 is not in (0, 1).
    """
    return 0.0
```

### Step 7

step-07-internal-force-evaluation

Goal
----
Performs exactly one internal-force evaluation of the full method for a given displacement field and bond state on the families of step 1, by chaining steps 3 to 6 in the order the source prescribes and then assembling the method's two-term force state with its non-uniform stress contribution and the truncation-corrected bond weights. Returns the internal force density of every point together with the updated history and bond phase-field, which the dynamics of step 8 feed back into the next evaluation. Validates the within-evaluation update ordering and the force-state assembly. Deliberately excluded: time integration.

```python
import numpy as np


def internal_force_evaluation(fam: dict, u: np.ndarray, calY: np.ndarray,
                              s_prev: np.ndarray, E: float, nu: float, Gc: float,
                              s_c: float, c0: float) -> dict:
    """One internal-force evaluation of the method with its damage update.

    Performs exactly one internal-force evaluation of the method for a
    given displacement field and bond state by chaining the earlier steps:
    the kinematic operators of step 3, the bond-associated deformation
    gradients of step 4, the undamaged bond stress and crack driving force
    of step 5 (no-fail bond pairs never accumulate damage: their driving
    force is treated as zero), the history update and the closed-form
    phase-field of step 6, and finally the method's force-state assembly
    (the two-term bond force state with its non-uniform stress
    contribution) with the truncation-corrected bond weights of step 1 in
    every family sum, giving the internal force density of every point.
    The order in which kinematic quantities, driving force, history,
    phase-field and degraded stress are updated within the evaluation -
    in particular which of them use the phase-field carried over from the
    previous evaluation and which use the updated one - must be taken from
    the source. Called once per force evaluation by the dynamics of
    step 8.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    u : np.ndarray
        (N, 3) displacement of every point in m.
    calY : np.ndarray
        (P,) bond damage-history variable before this evaluation, in J/m^3.
    s_prev : np.ndarray
        (P,) bond phase-field carried over from the previous evaluation.
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio in (-1, 0.5).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    s_c : float
        Critical phase-field value in (0, 1).
    c0 : float
        Griffith-consistency normalization constant of the kernel (output
        of step 2), in (0, 1).

    Returns
    -------
    dict
        'B' : (N, 3) float array, internal force density of every point in N/m^3;
        'calY' : (P,) float array, updated bond damage-history variable in J/m^3;
        's' : (P,) float array, updated bond phase-field in [0, 1].

    Raises
    ------
    ValueError
        If a material, fracture or normalization parameter is invalid, or
        u, calY, s_prev have inconsistent shapes.
    """
    return {}
```

### Step 8

step-08-velocity-verlet-dynamics

Goal
----
Runs the frozen explicit dynamics of the pre-notched benchmark on the families of step 1: initial bond phase-field from the pre-notch history through step 6, velocity Dirichlet layers, one initializing internal-force evaluation of step 7, and n_steps velocity Verlet steps with exactly one internal-force evaluation of step 7 each. Returns the final kinematic state and the final bond history and phase-field, which the orchestrator of step 9 turns into the crack dissipation energy. Validates the boundary protocol, the integrator layout, and the irreversible accumulation of damage.

```python
import numpy as np


def velocity_verlet_run(fam: dict, E: float, nu: float, rho0: float, Gc: float,
                        s_c: float, v0: float, dt: float, n_steps: int,
                        c0: float) -> dict:
    """Explicit dynamics of the pre-notched benchmark; returns the final state.

    Runs the frozen benchmark protocol on the bond families of step 1:
    the pre-notch bond pairs start fully damaged through a damage history
    of 1.0e30 J/m^3 turned into the initial bond phase-field by step 6;
    rows j = 0 and j = ny-1 are velocity Dirichlet layers with
    v = (0, -v0, 0) and (0, +v0, 0) for all t >= 0, exact displacement
    -/+ v0 t (zero x, z components) and zero acceleration; all other
    points start at rest with zero displacement; no external body force.
    One internal-force evaluation of step 7 on the initial state
    initializes the acceleration; then n_steps velocity Verlet steps of
    size dt follow, each with the position update first (prescribed
    layers overwritten with their exact positions), then exactly one
    internal-force evaluation of step 7 that updates the bond state and
    yields the new acceleration, then the velocity update (prescribed
    layers overwritten with their prescribed velocity). The final state
    is consumed by the orchestrator of step 9.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1); needs
        ny >= 4.
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio in (-1, 0.5).
    rho0 : float
        Mass density in kg/m^3 (> 0).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    s_c : float
        Critical phase-field value in (0, 1).
    v0 : float
        Boundary layer speed in m/s (>= 0).
    dt : float
        Time step in s (> 0).
    n_steps : int
        Number of velocity Verlet steps (>= 0).
    c0 : float
        Griffith-consistency normalization constant of the kernel (output
        of step 2), in (0, 1).

    Returns
    -------
    dict
        'u' : (N, 3) displacement in m, 'v' : (N, 3) velocity in m/s,
        'a' : (N, 3) acceleration in m/s^2, all at t = n_steps * dt;
        'calY' : (P,) bond damage-history variable in J/m^3 and
        's' : (P,) bond phase-field in [0, 1], both after the last force
        evaluation.

    Raises
    ------
    ValueError
        If n_steps is not a non-negative integer, ny < 4, or a material,
        loading, integration or normalization parameter is invalid.
    """
    return {}
```

### Step 9

step-09-crack-dissipation-orchestrator

Goal
----
Final orchestrator. Chains the earlier steps end to end - families and bond kernel (step 1), damage-model normalization constant (step 2), initial bond phase-field of the pre-notched state (step 6), and the dynamics of step 8 (which invoke step 7 and through it steps 3-6 once per force evaluation) - then evaluates the method's total crack dissipation energy of the body on the final and the initial bond phase-field and returns their difference, the crack dissipation growth dE. Every earlier step is on the answer path; nothing is recomputed locally.

```python
import numpy as np


def pfpd_crack_dissipation(nx: int, ny: int, nz: int, dx: float, delta: float,
                           E: float, nu: float, rho0: float, Gc: float, s_c: float,
                           v0: float, dt: float, n_steps: int) -> float:
    """Crack dissipation growth of the full benchmark run (orchestrator).

    Chains the earlier steps end to end: build_families (step 1) for the
    point cloud, families, truncation-corrected bond kernel and bond masks;
    pfpd_normalization_constant('cubic') (step 2) for the damage-model
    normalization constant c0; bond_phase_field (step 6) for the bond
    phase-field of the pre-notched initial state (history 1.0e30 J/m^3 on
    the pre-notch pairs, zero elsewhere); velocity_verlet_run (step 8,
    which calls internal_force_evaluation of step 7 and through it steps
    3-6 once per force evaluation) for the final bond state. Then
    evaluates the method's total crack dissipation energy of the body -
    the source's nonlocal crack-surface functional, whose weighting and
    normalization must be taken from the source - on the final and on
    the initial bond phase-field and returns their difference, so that
    the pre-notch contribution present at t = 0 is excluded.

    Parameters
    ----------
    nx : int
        Number of grid points along x (>= 2).
    ny : int
        Number of grid points along y (>= 4).
    nz : int
        Number of grid points along z (>= 2).
    dx : float
        Grid spacing in m (> 0).
    delta : float
        Horizon radius in m (> dx).
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio in (-1, 0.5).
    rho0 : float
        Mass density in kg/m^3 (> 0).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    s_c : float
        Critical phase-field value in (0, 1).
    v0 : float
        Boundary layer speed in m/s (>= 0).
    dt : float
        Time step in s (> 0).
    n_steps : int
        Number of velocity Verlet steps (>= 0).

    Returns
    -------
    float
        Crack dissipation growth E_Gamma(T) - E_Gamma(0), in J.

    Raises
    ------
    ValueError
        If any parameter is invalid, or the crack dissipation decreased
        (irreversibility violated).
    """
    return 0.0
```
