# Material_Science-Semiconductor_Materials-23

## Background

Density-functional bandgap predictions depend strongly on the exchange-correlation functional. Meta-GGAs such as SCAN satisfy exact constraints but use fixed internal parameters that were not tuned per bonding environment. System-conditioned reparameterizations keep the SCAN exchange form while replacing selected internal exchange constants for a given crystal, with the goal of opening the gap without degrading the lattice constant.

## Problem

Semilocal meta-GGA functionals can underestimate semiconductor bandgaps. Recent system-conditioned reparameterizations retain the SCAN exchange construction but replace selected internal constants to account for the bonding environment of covalent semiconductors while constraining the equilibrium lattice constant.

For zincblende beta-SiC, quantify the exchange-strengthening effect of switching from the default SCAN exchange constants (kappa = 0.065, c1x = 0.667) to the system-conditioned kappa and c1x selected for beta-SiC by that reparameterization, which you must identify.

Evaluate the SCAN exchange enhancement Fx(s, alpha) on the grid s in {0.75, 1.00, 1.25} and alpha in {0.15, 0.30, 0.45}, with weights [[1, 2, 3], [2, 3, 4], [3, 4, 5]] where rows index s and columns index alpha. All auxiliary SCAN exchange constants follow the published SCAN definitions as functions of kappa.

From Pauling electronegativities X_Si = 1.90, X_C = 2.55 and cubic lattice constant a = 4.35 angstrom, compute the standard zincblende bonding descriptors (covalency C, nearest-neighbor bond length, bond strength B, and ionic density rho) used to condition the sampling weights.

The effective weight is W_ij = w_ij * exp(-B*(1 - alpha_j)^2 - rho*s_i^2 - lambda*s_i^2*(1 - alpha_j)^2), where B is the bond-strength descriptor, rho is the ionic density, and lambda is a running screening parameter. Starting from lambda = 0, perform exactly three sweeps over the grid, traversing alpha in the outer loop and s in the inner loop, accumulating each point into the running numerator sum_ij W_ij*(Fx_SD - Fx_def)/Fx_def and the running denominator sum_ij W_ij, and refreshing lambda = C * (running numerator)/(running denominator) after every point while leaving it unchanged whenever that running denominator is still zero, where C is the covalency descriptor; lambda carries over between sweeps while the running sums restart each sweep, and R = 100 * (numerator)/(denominator) from the third sweep.

In the reasoning, report the recovered beta-SiC exchange parameters, the reduced adjustable parameter set used in the source fit, and the source's tabulated SCAN/SD-SCAN/experimental bandgap and lattice comparisons supporting their constrained selection, the four descriptors, the SCAN intermediates b4, h1x, fx, gx, and Fx at (s = 1.00, alpha = 0.30) under both parameter sets, the single-point relative percent increase there, and the third-sweep numerator, denominator, and lambda used to form R.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

step_01

Goal
----
Compute the SCAN auxiliary exchange constants implied by kappa.

```python
def compute_scan_auxiliary_constants(
    kappa: float,
) -> "np.ndarray":
    """Compute the SCAN auxiliary exchange constants [b1, b2, b3, b4].

    Parameters
    ----------
    kappa : float
        SCAN exchange parameter kappa (positive).

    Returns
    -------
    constants : "np.ndarray"
        Shape (4,) array [b1, b2, b3, b4].
    """
    return constants  # placeholder
```

### Step 2

step_02

Goal
----
Compute the SCAN metallic-branch exchange factor h1x.

```python
def compute_h1x_metallic_factor(
    s: float,
    alpha: float,
    kappa: float,
) -> float:
    """Compute the SCAN metallic-branch exchange factor at one grid point.

    Parameters
    ----------
    s : float
        Dimensionless density gradient (positive).
    alpha : float
        Bond-strength indicator in [0, 1].
    kappa : float
        SCAN exchange parameter kappa (positive).

    Returns
    -------
    h1x : float
        SCAN metallic-branch exchange factor at (s, alpha). Assembled by
        calling the earlier sub-problem functions.
    """
    return 0.0
```

### Step 3

step_03

Goal
----
Compute the SCAN exchange enhancement factor

```python
def compute_fx_enhancement(
    s: float,
    alpha: float,
    kappa: float,
    c1x: float,
) -> float:
    """Compute the SCAN exchange enhancement factor at one grid point.

    Parameters
    ----------
    s : float
        Dimensionless density gradient (positive).
    alpha : float
        Bond-strength indicator in [0, 1].
    kappa : float
        SCAN exchange parameter kappa (positive).
    c1x : float
        SCAN exchange interpolation parameter c1x.

    Returns
    -------
    fx_enhancement : float
        SCAN exchange enhancement factor at (s, alpha).
    """
    return 0.0
```

### Step 4

step_04

Goal
----
Compute solid-state descriptors for system-dependent SCAN reparameterization.

```python
def compute_material_descriptors(
    electronegativity_a: float,
    electronegativity_b: float,
    lattice_a: float,
) -> "np.ndarray":
    """Compute [covalency, bond_length, bond_strength, ionic_density] for zincblende AB.

    Parameters
    ----------
    electronegativity_a : float
        Pauling electronegativity of species A.
    electronegativity_b : float
        Pauling electronegativity of species B.
    lattice_a : float
        Cubic lattice constant a in angstrom (must be positive).

    Returns
    -------
    descriptors : "np.ndarray"
        Shape (4,) array [covalency, bond_length, bond_strength, ionic_density].
    """
    return descriptors  # placeholder
```

### Step 5

step_05

Goal
----
Compute the task-defined descriptor-conditioned exchange-enhancement metric.

```python
def compute_weighted_fx_percent_increase(
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    """Descriptor-conditioned weighted mean percent increase in Fx over default SCAN.

    Parameters
    ----------
    s_grid : "np.ndarray"
        Shape (n_s,) dimensionless density gradients.
    alpha_grid : "np.ndarray"
        Shape (n_a,) bond-strength indicators in [0, 1].
    weight_matrix : "np.ndarray"
        Shape (n_s, n_a) nonnegative weights.
    descriptors : "np.ndarray"
        Shape (4,) material descriptors
        [covalency, bond_length, bond_strength, ionic_density].
    kappa_sd : float
        SD-SCAN kappa parameter.
    c1x_sd : float
        SD-SCAN c1x parameter.
    kappa_default : float
        Default SCAN kappa parameter.
    c1x_default : float
        Default SCAN c1x parameter.

    Returns
    -------
    percent_increase : float
        Descriptor-conditioned weighted mean relative percent increase in Fx
        over the (s, alpha) grid, relative to the default-SCAN reference.

    Notes
    -----
    Rows of weight_matrix index s_grid and columns index alpha_grid. Let
    C = descriptors[0], B = descriptors[2], rho = descriptors[3], and
    q_ij = (Fx_SD(s_i, alpha_j) - Fx_def(s_i, alpha_j)) / Fx_def(s_i, alpha_j).
    Initialize lambda = 0 and perform exactly three sweeps. At each sweep,
    reset numerator and denominator to zero while retaining lambda. Traverse
    alpha_grid in the outer loop and s_grid in the inner loop, preserving
    their supplied order. At each point use the current lambda to compute
    W_ij = weight_matrix[i,j] * exp(-B*(1-alpha_j)**2 - rho*s_i**2
                                  - lambda*s_i**2*(1-alpha_j)**2).
    Add W_ij*q_ij and W_ij to the running numerator and denominator, then
    update lambda = C*numerator/denominator if denominator is positive;
    otherwise retain lambda. Return 100*numerator/denominator from sweep 3.
    Inputs must give a positive final effective-weight sum and nonzero
    default enhancement at every grid point.
    """
    return 0.0
```

### Step 6

step_06

Goal
----
Calibrate the SD-SCAN kappa that reproduces a target weighted Fx metric.

```python
def calibrate_kappa_to_target(
    target_percent_increase: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
    kappa_low: float = 0.01,
    kappa_high: float = 0.5,
) -> float:
    """Solve for the SD-SCAN kappa reproducing a target weighted Fx metric.

    Parameters
    ----------
    target_percent_increase : float
        Target weighted mean relative percent increase in Fx.
    s_grid : "np.ndarray"
        Shape (n_s,) dimensionless density gradients.
    alpha_grid : "np.ndarray"
        Shape (n_a,) bond-strength indicators in [0, 1].
    weight_matrix : "np.ndarray"
        Shape (n_s, n_a) nonnegative weights.
    descriptors : "np.ndarray"
        Shape (4,) material descriptors conditioning the weighted metric.
    c1x_sd : float
        SD-SCAN c1x parameter, held fixed during calibration.
    kappa_default : float
        Default SCAN kappa parameter.
    c1x_default : float
        Default SCAN c1x parameter.
    kappa_low : float
        Lower end of the admissible kappa bracket.
    kappa_high : float
        Upper end of the admissible kappa bracket.

    Returns
    -------
    kappa_sd : float
        A kappa reproducing the target, converged to a bracket width of
        1e-12. The inputs must bracket a solution. When the metric is
        constant over the bracket, any kappa in the bracket is valid.
    """
    return 0.0
```

### Step 7

step_07

Goal
----
Orchestrator: call prior sub-problems and return the weighted Fx percent increase

```python
def orchestrate_sd_scan_exchange_proxy(
    electronegativity_si: float,
    electronegativity_c: float,
    lattice_a: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    """Run the SD-SCAN exchange enhancement pipeline and return percent increase.

    Parameters
    ----------
    electronegativity_si : float
        Pauling electronegativity of Si.
    electronegativity_c : float
        Pauling electronegativity of C.
    lattice_a : float
        Zincblende lattice constant a in angstrom.
    s_grid : "np.ndarray"
        Dimensionless density-gradient grid.
    alpha_grid : "np.ndarray"
        Bond-strength indicator grid.
    weight_matrix : "np.ndarray"
        Weight matrix over the (s, alpha) grid.
    kappa_sd : float
        SD-SCAN kappa parameter.
    c1x_sd : float
        SD-SCAN c1x parameter.
    kappa_default : float
        Default SCAN kappa parameter.
    c1x_default : float
        Default SCAN c1x parameter.

    Returns
    -------
    percent_increase : float
        Relaxed descriptor-conditioned weighted mean percent increase in Fx.
        Compute the forward metric at the supplied kappa, calibrate a kappa
        reproducing that target using the preceding calibration step, then
        recompute and return the metric at the calibrated kappa. A different
        calibrated kappa is acceptable when the inverse is nonunique.
        The forward target must be bracketed by kappa in [0.01, 0.5].
    """
    return 0.0
```
