# Mathematics-Computational_Finance-21

## Background

The Fokker-Planck equation transports a probability density, and its exact solution keeps two structural properties that numerical work must fight to retain: the density never becomes negative, and its total mass is conserved exactly. Both properties are fragile under discretization precisely in the regime this benchmark probes. When the diffusion tensor is anisotropic and carries a cross term, the mixed derivative couples the two coordinates and is the classical trouble spot of splitting methods; when the drift is strong relative to the diffusion on the scale of a grid cell (a large cell Peclet number), second-order spatial discretizations buy their accuracy with stencil weights of both signs, so the discrete generator is not an M-matrix generator and sign control of the evolved density is no longer automatic. A generator of this kind can still relax every initial density to a strictly positive stationary profile, but the road there may pass through states that dip below zero, and whether a given time discretization reproduces, suppresses, or amplifies such dips is a quantitative question about the time-stepping map, not only about the spatial stencil.

That question matters operationally. In computational finance and in kinetic modelling the density feeds later stages - quadratures for prices, conditional expectations, likelihoods - and negative pockets, even small ones, can poison those stages or make calibration loops fail. At the same time the cost profile matters: advancing each spatial direction by the exact matrix exponential of its one-dimensional generator is accurate but expensive, since Krylov approximations of the exponential action dominate the cost per step and their subspace dimension grows with the mesh. Replacing exponentials by rational functions of the generators turns every substep into one banded or sparse linear solve and drops the cost per step to linear in the number of unknowns. The price of that substitution is that sign behaviour must be re-established for the rational maps: different consistent, second-order rational functions of the same stable generator can behave very differently in how their entrywise sign depends on the step size, and a scheme assembled from such factors inherits its sign behaviour from the factors in a way that depends on how the composite step is built. Quantifying where a factor is entrywise nonnegative - and where it is not - therefore becomes part of certifying the scheme for a given problem, which is why this benchmark asks for computed positivity thresholds alongside the evolved density.

The benchmark datum combines a resolved Gaussian mass with a single-cell spike carrying a finite fraction of the total mass. The spike deliberately excites the stiffest resolved modes of the discretization, the regime in which time integrators of equal formal order differ most strongly in how much signed transient they inject into the density, while the smooth component keeps the run representative of a realistic density evolution. The run length is short enough that the recorded extremes are dominated by the transient response rather than by the approach to the stationary profile, so the reported quantities isolate what the benchmark is designed to measure: the interaction of one specific composite time-stepping construction with a sign-indefinite, advection-dominated spatial operator.

## Problem

A recent line of work develops alternating-direction-implicit time stepping for two-dimensional anisotropic Fokker-Planck equations in which every factor of the composite one-step map - the mixed-derivative block included - is a rational function of a discrete generator, applied by banded or sparse linear solves, and in which the entrywise sign behaviour of each factor at large steps is governed by a single scalar property of the rational function chosen. Reproduce that scheme's positivity benchmark on the bespoke configuration below and return the most negative nodal value the density attains over the run.

Four ingredients are deliberately not specified here and must be determined from the source: (i) the second-order rational function used for the two directional factors - the specific map, the scalar large-step property of the map by which the source selects it, and the single linear system by which one application of the factor is computed; (ii) the treatment of the mixed-derivative block inside the composite step - which factor advances it and how that factor is applied; (iii) the arrangement of one composite step - the order in which the factors are applied and the substep length each factor uses; (iv) the two per-direction positivity thresholds reported alongside the run - one attached to the first-order fallback factor's matrix function and one attached to the second-order directional map itself; which matrix function each threshold belongs to is fixed by the source, while the numerical location procedure is pinned in the conventions below. Do not invent any of these; they are fixed by the source.

FROZEN CONFIGURATION.
The equation is the two-dimensional Fokker-Planck equation in divergence form,
dp/dt = -d/dx[mu_x p] - d/dy[mu_y p] + d2/dx2[Sxx p] + d2/dy2[Syy p] + 2 d2/dxdy[Sxy p],
with linear drift mu_x(x) = -kx (x - mx), mu_y(y) = -ky (y - my), constant diffusion Sxx, Syy, and constant cross-diffusion Sxy = rho*sqrt(Sxx*Syy).

Grid: a cell-centred tensor mesh. Cells are indexed i = 1..nx, j = 1..ny with centres x_i = xL + (i - 1/2) hx, y_j = yL + (j - 1/2) hy, hx = (xR - xL)/nx, hy = (yR - yL)/ny. The density vector stores cell (i, j) at linear position ell = (i - 1) ny + j (positions numbered 1..nx*ny).

Directional generators: Lx (nx by nx) and Ly (ny by ny) are assembled in flux form. For a direction with coordinate a, drift u(a) = -kappa (a - m) and diffusion coefficient sigma:
(L p)_i = ( J_{i-1/2} - J_{i+1/2} ) / h, with fluxes only at the n-1 interior faces a_{i+1/2} = aL + i h (the two boundary faces carry zero flux), and
J_{i+1/2} = u(a_{i+1/2}) * phat_{i+1/2} - sigma * (p_{i+1} - p_i)/h.
The convective face value phat is the second-order linear-upwind value: if u(a_{i+1/2}) >= 0 then phat = (3 p_i - p_{i-1})/2, falling back to phat = p_i when cell i-1 does not exist; if u(a_{i+1/2}) < 0 then phat = (3 p_{i+1} - p_{i+2})/2, falling back to phat = p_{i+1} when cell i+2 does not exist. The Kronecker lifts are Ax = kron(Lx, I_ny) and Ay = kron(I_nx, Ly).

Mixed operator: Axy = 2 Sxy * kron(Dx, Dy), where each one-dimensional matrix D is in conservative face form (D p)_i = ( g_{i+1/2} - g_{i-1/2} ) / h with g = 0 at the two boundary faces. For Sxy >= 0: Dx uses the forward-biased face value g_{i+1/2} = (3 p_{i+1} - p_{i+2})/2 with fallback g = p_{i+1} when cell i+2 does not exist, and Dy uses the backward-biased face value g_{i+1/2} = (3 p_i - p_{i-1})/2 with fallback g = p_i when cell i-1 does not exist. For Sxy < 0 both directions use the forward-biased face value, with the same fallback rule.

Initial datum: evaluate q_ij = exp( -(x_i - x0)^2/(2 wx^2) - (y_j - y0)^2/(2 wy^2) ) at the cell centres, scale the smooth part as p0 = (1 - fsp) * q / ( sum(q) * hx * hy ), then add a spike of mass fsp by adding fsp/(hx*hy) to the single cell (isp, jsp). The discrete mass sum(p0)*hx*hy equals 1.

Run: advance nt composite steps of size dt from p0 with the source's composite arrangement of the two directional factors and the mixed-block factor, and record the minimum entry of the density vector after each complete composite step, n = 1..nt.

Parameters:
xL = -2.4        # left x boundary
xR = 2.4         # right x boundary
nx = 40          # cells in x
kx = 4.2         # x drift rate, 1/time
mx = 0.0         # x drift centre
Sxx = 1.0        # x diffusion coefficient
yL = -1.8        # left y boundary
yR = 1.8         # right y boundary
ny = 10          # cells in y
ky = 3.1         # y drift rate, 1/time
my = 0.0         # y drift centre
Syy = 0.25       # y diffusion coefficient
rho = 0.4        # correlation; Sxy = rho*sqrt(Sxx*Syy) = 0.2
x0 = 0.30        # datum centre, x
y0 = -0.25       # datum centre, y
wx = 0.45        # datum width, x
wy = 0.50        # datum width, y
isp = 21         # spike cell index, x (1-based)
jsp = 6          # spike cell index, y (1-based)
fsp = 0.35       # spike mass fraction
dt = 0.05        # time step
nt = 6           # number of composite steps
glo = 1e-6       # bisection bracket, lower endpoint
ghi = 64.0       # bisection bracket, upper endpoint
tolpos = -1e-12  # entrywise-nonnegativity tolerance
nbis = 200       # bisection iterations

BENCHMARK CONVENTIONS (fixed here for reproducibility; these are not parameters of the source method).
All arithmetic in IEEE double precision. Every implicit factor is applied by a direct linear solve at machine accuracy; no iterative solver tolerances enter anywhere. The matrix functions used for thresholds are formed as dense matrices and inverted, or solved against the identity, at machine accuracy. Threshold location: for a matrix function gamma -> M(gamma) with entrywise minimum theta(gamma), an entry counts as nonnegative when it is >= tolpos = -1e-12; if theta(glo) >= tolpos the threshold is 0.0; if theta(ghi) < tolpos the window is reported empty; otherwise run exactly nbis = 200 bisection steps on [glo, ghi], keeping the endpoint a violating (theta < tolpos) and the endpoint b satisfying (theta >= tolpos), replacing one endpoint by the midpoint each iteration and stopping early only if the midpoint coincides with an endpoint in double precision; report the final b. The entrywise minimum runs over all matrix entries, including entries outside the sparsity bands. "Off-diagonal entries" of a matrix means every entry with row index different from column index, structural zeros included. The stationary density of a directional generator is the kernel vector of L normalized so that its entries sum to 1 (no cell-width weighting), computed by a machine-accuracy direct method. The mass defect at final time is abs( sum(p)*hx*hy - 1 ). The recorded minimum is taken only after complete composite steps; intermediate stage states inside a step are not inspected. The run is fully deterministic; no randomness anywhere. Report every requested quantity to at least 6 significant figures.

Do not substitute a stage-based splitting of your own choosing for the source's composite arrangement; do not reorder, drop, or re-time any factor of that arrangement; do not hard-code any threshold or any run output; the two reported thresholds must come from the pinned bisection applied to the two factors the source prescribes, assembled for this configuration.

In your reasoning, report these quantities from your run alongside the final answer: (1) the minimum off-diagonal entry of Lx; (2) the smallest entry of the normalized stationary density of Lx and of Ly; (3) the first-order-factor threshold for Lx and for Ly; (4) the second-order-factor threshold for Lx and for Ly; (5) the most negative entry of the density at final time; (6) the density at cell (i, j) = (20, 7) at final time; (7) the maximum entry of the density at final time; (8) the final-time mass defect.

Return the most negative nodal value attained over the nt complete composite steps of this deterministic benchmark as the final answer.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction string, not a vector, and not prose.
Put only that one number between the tags. No units, words, or extra lines.
Keep short (a few hundred words).
In <reasoning>, identify the source-dependent choices needed to reproduce the experiment and report only the few intermediate quantities necessary to justify the final answer.
Do not paste matrices, full state vectors, probability tables, circuit dumps, per-iteration solver paths, or other large intermediate outputs.

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

step01_min_offdiagonal_entry

Goal
----
Assemble the 1-D flux-form directional Fokker-Planck generator on a cell-centred grid (zero-flux boundary faces, second-order linear-upwind convective face value with first-order upwind fallback at the boundary, central diffusive flux) and return the minimum off-diagonal entry of the assembled matrix, structural zeros included. Validates the conservative closure (column sums vanish to round-off) and the input parameters; the step deliberately excludes any time stepping.

```python
import numpy as np


def min_offdiagonal_entry(params: dict) -> float:
    """Minimum off-diagonal entry of the 1-D directional generator.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR' (domain bounds), 'n' (number of cells, integer
        >= 3), 'kappa' (drift rate, >= 0), 'm' (drift centre), 'sigma'
        (diffusion coefficient, > 0). The generator is the flux-form
        matrix pinned in the problem statement: (L p)_i =
        (J_{i-1/2} - J_{i+1/2})/h with zero boundary fluxes,
        J_{i+1/2} = u_{i+1/2}*phat - sigma*(p_{i+1}-p_i)/h, u(a) =
        -kappa*(a - m) at the face, and phat the second-order
        linear-upwind face value with first-order fallback where the
        second upwind cell does not exist.

    Returns
    -------
    float
        min over all entries L[i, j] with i != j (structural zeros
        included).

    Raises
    ------
    ValueError
        If a key is missing or non-finite, n < 3, aR <= aL, sigma <= 0,
        or kappa < 0.
    """
    return 0.0
```

### Step 2

step02_stationary_minimum_mass

Goal
----
Compute the discrete stationary density of the 1-D directional generator (the kernel vector of L normalized so its entries sum to 1, obtained by a machine-accuracy direct solve) and return its smallest entry. Validates the normalization; excludes any time stepping.

```python
import numpy as np


def stationary_minimum_mass(params: dict) -> float:
    """Smallest entry of the normalized discrete stationary density.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1; the
        generator is the same pinned flux-form matrix.

    Returns
    -------
    float
        min_k p_inf[k] where L p_inf = 0 and sum_k p_inf[k] = 1 (no
        cell-width weighting).

    Raises
    ------
    ValueError
        If a key is missing or non-finite, n < 3, aR <= aL, sigma <= 0,
        or kappa < 0.
    """
    return 0.0
```

### Step 3

step03_first_order_threshold

Goal
----
Locate the positivity threshold that the source attaches to the first-order fallback factor of the directional sweep: form that factor's matrix function for the 1-D directional generator and report the edge of the one-sided window on which it is entrywise nonnegative under the pinned tolerance, located by the pinned bisection contract (bracket [glo, ghi], nbis iterations, final upper endpoint reported, 0.0 when the condition already holds at glo). Which matrix function belongs to the first-order factor is one of the ingredients the problem statement withholds and is not restated here; the second-order directional map is step 4.

```python
import numpy as np


def first_order_threshold(params: dict) -> float:
    """Positivity threshold of the first-order factor's matrix function.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1, plus
        'glo', 'ghi' (bisection bracket endpoints, 0 < glo < ghi),
        'tol_pos' (entrywise-nonnegativity tolerance; an entry counts as
        nonnegative when >= tol_pos), and 'nbis' (bisection iterations).
        The matrix function is the one the source attaches to the
        first-order fallback factor of the directional sweep; it is
        formed densely and inverted at machine accuracy, and the
        entrywise minimum runs over all entries.

    Returns
    -------
    float
        0.0 when the entrywise minimum at glo already clears tol_pos;
        otherwise the final upper bisection endpoint after nbis
        iterations on [glo, ghi].

    Raises
    ------
    ValueError
        On invalid inputs, or when the entrywise minimum at ghi does not
        clear tol_pos (no window inside the bracket).
    """
    return 0.0
```

### Step 4

step04_second_order_threshold

Goal
----
Locate the positivity threshold of the source's second-order directional map itself, under exactly the same pinned bisection contract as step 3. Both the map and the single linear system by which one application of it is computed are ingredients the problem statement withholds; do not substitute a different second-order rational function of the generator. This step excludes the composite scheme.

```python
import numpy as np


def second_order_threshold(params: dict) -> float:
    """Positivity threshold of the source's second-order directional map.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma', 'glo', 'ghi',
        'tol_pos', 'nbis' with the same meanings and the same pinned
        bisection procedure as in step 3. Here the matrix function is
        the source's own second-order directional map evaluated at
        gamma*L, formed densely and inverted at machine accuracy.

    Returns
    -------
    float
        0.0 when the entrywise minimum at glo already clears tol_pos;
        otherwise the final upper bisection endpoint after nbis
        iterations on [glo, ghi].

    Raises
    ------
    ValueError
        On invalid inputs, or when the entrywise minimum at ghi does not
        clear tol_pos (no window inside the bracket).
    """
    return 0.0
```

### Step 5

step05_directional_factor_delta_action

Goal
----
Apply one directional factor of the source's scheme at substep s to a delta datum at cell k and return the minimum entry of the result. The factor must be applied by the single linear system the source prescribes for one application, at machine accuracy. Validates exact discrete mass conservation of the factor; excludes the 2-D composite scheme.

```python
import numpy as np


def directional_factor_delta_action(params: dict) -> float:
    """Minimum entry of one directional factor applied to a delta datum.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1, plus
        's' (substep length, > 0) and 'k' (1-based cell index of the
        delta datum, 1 <= k <= n). The factor is the source's
        second-order directional map at substep s; one application is
        computed by the single dense linear solve the source prescribes,
        applied to the unit vector e_k at machine accuracy.

    Returns
    -------
    float
        min entry of the resulting vector.

    Raises
    ------
    ValueError
        On invalid inputs, s <= 0, k out of range, or when the result
        does not sum to 1 within 1e-8 (mass conservation violated).
    """
    return 0.0
```

### Step 6

step06_mixed_action_at_node

Goal
----
Assemble the conservative one-sided mixed operator Axy = 2*Sxy*kron(Dx, Dy) (face-form one-sided differences with zero boundary faces; forward-biased in x and backward-biased in y for Sxy >= 0, forward-biased in both directions for Sxy < 0), build the pinned initial datum, and return the entry of Axy p0 at the query cell (iq, jq). Validates the exact zero column sums of the closure; excludes time stepping.

```python
import numpy as np


def mixed_action_at_node(params: dict) -> float:
    """Entry of Axy p0 at one query cell for the pinned mixed operator.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys 'xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
        'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho', 'x0', 'y0', 'wx',
        'wy', 'isp', 'jsp', 'fsp' as pinned in the problem statement,
        plus the query cell 'iq', 'jq' (1-based). The datum is the
        normalized Gaussian-plus-spike vector with x-major ordering
        ell = (i-1)*ny + j.

    Returns
    -------
    float
        (Axy @ p0)[(iq-1)*ny + (jq-1)] in 0-based storage.

    Raises
    ------
    ValueError
        On invalid inputs, a query cell out of range, or a violated
        conservative closure.
    """
    return 0.0
```

### Step 7

step07_composite_step_density

Goal
----
Advance a given density vector by exactly one complete composite step of the source's arrangement - the two directional factors and the mixed-derivative factor, each applied by a direct dense solve at the substep length the source assigns it - and return the new density vector. The arrangement, the directional map and the mixed-block factor are withheld ingredients of the problem statement and are not restated here; the step must conserve discrete mass exactly and be second-order accurate in dt. This is the single reusable building block of the benchmark: the runs of steps 8 and 9 are built by repeated calls to this step.

```python
import numpy as np


def composite_step_density(params: dict, p: np.ndarray) -> np.ndarray:
    """Density after one complete composite step applied to a given state.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys as in step 6 (without 'iq'/'jq'),
        plus 'dt' (> 0). One call advances the state by one complete
        composite step of the source's arrangement: every factor - the
        two directional ones and the mixed-derivative one - is applied
        by a direct dense solve, at the substep length the source
        assigns that factor. The arrangement itself, the directional map
        and the mixed-block factor are withheld ingredients of the
        problem statement and are not restated here.
    p : numpy.ndarray
        Density vector of length nx*ny (0-based, x-major ordering) to
        advance; it is not modified.

    Returns
    -------
    numpy.ndarray
        The density vector after the step, with the same length as p.

    Raises
    ------
    ValueError
        On invalid inputs, dt <= 0, or p of the wrong length or with
        non-finite entries.
    """
    return 0.0
```

### Step 8

step08_run_nodal_value

Goal
----
Build the pinned initial datum from the configuration, advance it by nt composite steps by repeatedly calling the step 7 oracle *oracle*composite_step_density, and return the density at the report cell (istar, jstar) at final time. nt = 0 is valid and returns the initial value at that cell. Excludes the running-minimum diagnostic, which belongs to the orchestrator.

```python
import numpy as np


def run_nodal_value(params: dict) -> float:
    """Density at one report cell after nt complete composite steps.

    The nt steps must be produced by repeated calls to the step 7
    oracle composite_step_density; this step contributes the
    pinned initial datum and the report-cell read-out.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys as in step 7, plus 'nt' (number of
        composite steps, integer >= 0) and the report cell 'istar',
        'jstar' (1-based).

    Returns
    -------
    float
        p[(istar-1)*ny + (jstar-1)] after nt steps (0-based storage,
        x-major ordering).

    Raises
    ------
    ValueError
        On invalid inputs, dt <= 0, nt < 0 or non-integer, or a report
        cell out of range.
    """
    return 0.0
```

### Step 9

step09_dfadi_positivity_benchmark

Goal
----
Orchestrator: run the complete positivity benchmark and return the most negative nodal value attained over the nt complete composite steps. The run itself is built from the earlier steps: the pinned datum is advanced by repeated calls to the step 7 oracle, and the minimum is taken over the states those calls return. Calls the oracles of steps 1-8 and uses every output: EM structure (step 1) and Perron positivity (step 2) as admissibility checks, the four thresholds (steps 3-4) as finiteness/positivity checks, the directional-factor action (step 5) and mixed action (step 6) as consistency checks against its own assembled factors, the step 7 oracle as the engine that produces every state in the run, and the step 8 oracle as a cross-check on the final-time report cell; it finally verifies discrete mass conservation.

```python
import numpy as np


def dfadi_positivity_benchmark(params: dict) -> float:
    """Most negative nodal value over the full composite-step run.

    Parameters
    ----------
    params : dict
        The complete frozen-configuration dictionary: the 2-D keys of
        step 6, the run keys 'dt', 'nt' (nt >= 1), the report cell
        'istar', 'jstar', and the threshold keys 'glo', 'ghi',
        'tol_pos', 'nbis'.

    Returns
    -------
    float
        min over n = 1..nt of the minimum entry of the density after
        complete composite step n.

    Raises
    ------
    ValueError
        On invalid inputs, or when any cross-check against the earlier
        steps fails (EM structure, stationary positivity, threshold
        finiteness, factor consistency, report-cell cross-check, mass
        conservation).
    """
    return 0.0
```
