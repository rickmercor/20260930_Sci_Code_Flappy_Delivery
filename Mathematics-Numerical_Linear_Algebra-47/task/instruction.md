# Mathematics-Numerical_Linear_Algebra-47

## Background

Advection-diffusion-reaction (ADR) systems are the computational workhorse of chemical kinetics, combustion, atmospheric transport, and plasma physics. Their semi-discretizations couple three operators with fundamentally different character: advection produces Jacobian spectra along the imaginary axis and is cheap to treat explicitly; diffusion produces large negative real eigenvalues that scale as $\mathcal{O}(\Delta x^{-2})$ and impose a severe step-size restriction on classical explicit methods; and local reaction terms can be arbitrarily stiff but couple nothing across the spatial grid. A time integrator for such systems must reconcile these regimes simultaneously, and each classical strategy compromises somewhere. Implicit-explicit additive Runge–Kutta (ARK) methods couple the operators tightly and carry embedded error estimators, but grouping diffusion with reaction in the implicit part produces algebraic systems coupled across the whole domain — increasingly a liability on accelerator hardware, where spatially local work is cheap and global solves are not. Operator splitting (Lie, Strang–Marchuk) grants complete freedom to treat each operator with its ideal method, but couples the operators only weakly, carries large error constants when the operators oppose one another, and offers no inexpensive temporal error estimate.

Super-time-stepping (STS) methods occupy a third position for the diffusive operator alone. Beginning with the Runge–Kutta–Chebyshev (RKC) schemes and continuing through Runge–Kutta–Legendre (RKL) and Runge–Kutta–Gegenbauer constructions, these are explicit Runge–Kutta methods with many internal stages chosen so that the stability interval along the negative real axis grows quadratically in the stage number. They retain the locality and simplicity of explicit stepping while taking steps competitive with implicit methods on parabolic problems; the price is that the stage count must be chosen from an estimate of the dominant eigenvalue of the diffusion Jacobian, and the schemes are second order at best. Hybrid integrators that embed STS methods inside a partitioned framework for full ADR systems exist — most prominently PIROCK, which welds a ROCK2 diffusion sweep to an ARK-type finishing procedure — but such constructions have been tied to one specific STS scheme and rely on intricate hand-derived stage patterns, which has limited their adoption and their ability to benefit from newer STS families.

A separate line of work on multirate infinitesimal (MRI) methods, originating in split-explicit numerical weather prediction and formalized through multirate infinitesimal step and multirate infinitesimal GARK theory, offers a systematic coupling mechanism: slow operators are sampled at a few stage times and folded into forcing terms for a sequence of modified fast initial-value problems, with coupling coefficients derived from order conditions of generalized-structure additive Runge–Kutta theory. Because the fast sub-problems may be solved by an arbitrary inner method, this framework decouples the choice of coupling from the choice of inner solver. Recent work exploits exactly this freedom to combine MRI-type coupling with modern STS methods for the diffusive partition and standard explicit or diagonally implicit treatment of the remaining partitions, yielding second-order, stiffly accurate, solve-decoupled integrators with embedded solutions, in which every implicit solve remains spatially local.

Deploying any STS-based integrator in production raises a practical question that is easy to underestimate: where does the dominant-eigenvalue estimate come from? Analytical bounds built from grid spacing and representative coefficient values are simple but can err badly when coefficients vary in space, and an underestimate is not a graceful failure — an explicit method run with too few stages is unstable, not merely inaccurate. In large kinetic and gyrokinetic simulations, where assembling or storing a Jacobian is infeasible, recent work advocates estimating the eigenvalue matrix-free: a power iteration whose matrix-vector products are difference quotients of the right-hand side, with perturbation sizes controlled through the same weighted root-mean-square norms used for temporal error control, a relative-change convergence test, and a multiplicative safety factor to absorb the residual estimation error. The interaction between estimator accuracy, safety margin, and integer stage selection then becomes part of the numerical method itself: it determines both whether the integration is stable and which member of the discrete family of stabilized schemes actually advances the solution.

## Problem

Multiphysics initial-value problems that couple advection, diffusion, and stiff local reactions strain every standard time-integration strategy: implicit-explicit additive Runge–Kutta methods require globally coupled implicit solves once diffusion is grouped with reaction, operator splitting weakens the coupling and forfeits an inexpensive error estimate, and earlier partitioned stabilized integrators are locked to one specific Runge–Kutta–Chebyshev construction. A recently introduced family of second-order integrators removes these constraints by advancing the diffusive part with a super-time-stepping method inside each stage of a solve-decoupled, stiffly accurate implicit-explicit pair, coupling the operators through forcing terms of multirate-infinitesimal type so that implicit solves remain spatially local and an embedded solution is available. A companion study by an overlapping group of authors supplies the practical machinery such integrators need: a Jacobian-free estimate of the dominant eigenvalue of the diffusion operator, obtained by a seeded power iteration whose matrix-vector products are difference quotients scaled through a component-wise weighted root-mean-square norm, together with a relative-change stopping rule and a multiplicative eigensafety factor.

Your task is to solve one concrete deterministic instance of this pipeline. On $x \in [0,1]$, integrate the advected Brusselator system
$$\partial_t u = -c\,\partial_x u + \partial_x\!\big(D(x)\,\partial_x u\big) + r\big(A - (w+1)u + vu^2\big),$$
$$\partial_t v = -c\,\partial_x v + \partial_x\!\big(D(x)\,\partial_x v\big) + r\big(wu - vu^2\big),$$
$$\partial_t w = -c\,\partial_x w + \partial_x\!\big(D(x)\,\partial_x w\big) + r\big(\tfrac{B-w}{\varepsilon} - wu\big),$$
with $D(x) = d\,(1 + 0.9\sin 2\pi x)$, discretized and integrated under the following configuration:

- $c = 0.45$, $d = 0.1$, $r = 1.1$, $\varepsilon = 0.02$, $A = 1.05$, $B = 2.9$
- grid: $N = 65$ nodes $x_j = j/64$, $j = 0,\dots,64$; unknowns stacked as $y = [u;\,v;\,w]$
- initial condition: $u = A + 0.1\sin 2\pi x$, $v = B/A + 0.1\sin 2\pi x$, $w = B + 0.1\sin 2\pi x$
- stationary boundary conditions: the advective, diffusive, and reactive right-hand-side components are each held at zero at both boundary nodes of every species
- advection: second-order centered differences at interior nodes
- diffusion: conservative flux form with face coefficients $D_{j+1/2} = (D_j + D_{j+1})/2$
- operator splitting: $f^A$ advection (explicit), $f^D$ diffusion (super-time-stepping), $f^R$ reaction (implicit, nodewise Newton iterated until the residual max-norm falls below $10^{-13}$)
- time stepping: 73 uniform steps of size $h = 1.5/73$ to $T = 1.5$, no step-size adaptivity
- eigenvalue estimation: performed once, from the initial state before the first step, on the diffusion operator alone, with initial vector drawn as numpy.random.default_rng(12).standard_normal(195), weighted-RMS tolerances $\mathrm{RTOL} = 10^{-4}$ and $\mathrm{ATOL} = 10^{-11}$, stopping tolerance $\tau = 0.003$, and eigensafety factor $q_\lambda = 1.1$

Advance the semi-discrete system with the integrator of this family built from the six-stage padded implicit-explicit Butcher pair with shared abscissae $\{0,\,2\gamma,\,2\gamma,\,1,\,1,\,1\}$, $\gamma = (2-\sqrt{2})/2$, exactly as printed in the work that introduced the family, using the second-order Runge–Kutta–Legendre method for every diffusion sub-step with its number of internal stages set, from the safeguarded eigenvalue estimate, to the minimum count that the family's stage-selection rule guarantees stable. Your final answer must be a single number: the value of $u$ at the zero-based node index 62 ($x = 0.96875$) at time $T$, reported to at least 10 significant digits. The reasoning should make clear how the eigenvalue estimate and the resulting sub-step stage counts, the construction of the inter-operator coupling, the integration itself, and the embedded-solution value at the same node contribute to the final result.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input state vectors, full solution vectors, or per-step traces.

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

01_compute_split_operators

Goal
----
Evaluate the three split right-hand-side components of the semi-discrete advected Brusselator system at a given state.

The state vector stacks the three species as y = [u; v; w], each on the same grid of n nodes with uniform spacing dx, so y has length 3n. The function returns the advective, diffusive, and reactive components as three separate rows.

Advection uses second-order centered differences at interior nodes: for a species s, the advective component at node j is -c (s_{j+1} - s_{j-1}) / (2 dx).

Diffusion uses conservative flux form with face coefficients formed as the arithmetic average of adjacent nodal values, D_{j+1/2} = (D_j + D_{j+1}) / 2. The diffusive component at interior node j is (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2.

Reaction is the Brusselator source, evaluated pointwise: r (A - (w + 1) u + v u^2) for the u species, r (w u - v u^2) for the v species, and r ((B - w) / eps - w u) for the w species.

Stationary boundary conditions are imposed by setting all three components to zero at both boundary nodes of every species. For the advective and diffusive components this follows from computing them at interior nodes only; for the reactive component the boundary entries are zeroed explicitly after evaluation.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y.size // 3; when y or D_nodes contains a non-finite value; when dx is not a positive scalar; when eps is not a positive scalar; or when any of c, r, A, B is not a finite scalar.

```python
def compute_split_operators(y: np.ndarray, D_nodes: np.ndarray, dx: float, c: float,
                            r: float, eps: float, A: float, B: float) -> np.ndarray:
    '''Evaluate the advective, diffusive, and reactive split components of the state.

    Parameters
    ----------
    y : np.ndarray
        State vector of shape (3n,), stacking the three species as [u; v; w],
        each sampled on the same grid of n nodes.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    c : float
        Advection speed.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.

    Returns
    -------
    components : np.ndarray
        Array of shape (3, 3n). Row 0 is the advective component, row 1 the
        diffusive component, row 2 the reactive component, each a vector of
        length 3n in the same species-stacked ordering as y.
    '''
    return components  # placeholder
```

### Step 2

02_estimate_dominant_eigenvalue

Goal
----
Estimate the dominant eigenvalue of the diffusion operator by a Jacobian-free power iteration, and return both the converged estimate and the safeguarded value used for stabilized-explicit stage selection.

The diffusion operator is the same conservative-flux operator used elsewhere in this problem: face coefficients D_{j+1/2} = (D_j + D_{j+1}) / 2, interior-node value (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2, and zero at both boundary nodes of every species. It is evaluated at the initial state y0 only.

Weights for the norm are formed once from the initial state as w_i = rtol |y0_i| + atol, and the weighted root-mean-square norm of a vector v is the square root of the mean of (v_i / w_i)^2.

The iteration starts from v drawn as numpy.random.default_rng(seed).standard_normal(y0.size). Each pass performs, in order: set sigma to the reciprocal of the weighted root-mean-square norm of the current v; form the matrix-vector product as the difference quotient (f_D(y0 + sigma v) - f_D(y0)) / sigma; update the eigenvalue estimate by the Rayleigh quotient (v dot Jv) / (v dot v); replace v by Jv divided by its Euclidean norm. From the second pass onward, the iteration stops as soon as the absolute difference between the current and previous eigenvalue estimates is less than tau times the absolute value of the current estimate.

The returned safeguarded value is q_lambda times the absolute value of the converged estimate.

The function raises ValueError when y0 is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y0.size // 3; when y0 or D_nodes contains a non-finite value; when any of dx, rtol, atol, tau, q_lambda is not a finite positive scalar; when max_iter is less than 2; when an iteration produces an image vector of zero Euclidean norm; or when the stopping criterion is not met within max_iter passes.

```python
def estimate_dominant_eigenvalue(y0: np.ndarray, D_nodes: np.ndarray, dx: float,
                                 seed: int, rtol: float, atol: float, tau: float,
                                 q_lambda: float, max_iter: int = 200) -> np.ndarray:
    '''Estimate the dominant eigenvalue of the diffusion operator, matrix-free.

    Parameters
    ----------
    y0 : np.ndarray
        Initial state vector of shape (3n,), stacking the three species as
        [u; v; w]. The operator is linearized and the weights are formed here.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    seed : int
        Seed for the initial iterate, drawn with numpy.random.default_rng.
    rtol : float
        Relative tolerance entering the weighted root-mean-square weights, positive.
    atol : float
        Absolute floor entering the weighted root-mean-square weights, positive.
    tau : float
        Relative-change stopping tolerance for successive eigenvalue estimates,
        positive.
    q_lambda : float
        Multiplicative safety factor applied to the converged estimate, positive.
    max_iter : int
        Maximum number of iteration passes permitted, at least 2.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,): the converged eigenvalue estimate (negative for a
        diffusion operator), the safeguarded magnitude q_lambda times its
        absolute value, and the number of passes performed, as a float.
    '''
    return result  # placeholder
```

### Step 3

03_select_rkl_stage_counts

Goal
----
Select the number of internal stages for each stabilized explicit diffusion sub-step.

For each supplied abscissa increment, the corresponding sub-step advances over an interval of length H = delta_c h, and the selected count is the smallest number of internal stages for which the second-order Runge-Kutta-Legendre method is linearly stable over the whole segment of the negative real axis reaching out to H lam_eff. The method requires at least two internal stages, so the selected count is never below 2.

The sub-step lengths are not the full step size: each entry of delta_c is the abscissa increment of one super-time-stepping stage, and the counts are returned in the order the increments are supplied.

The eigenvalue magnitude passed here is the safeguarded one; the estimate has already been multiplied by its safety factor before reaching this function.

The function raises ValueError when h is not a finite positive scalar; when lam_eff is not a finite scalar; when lam_eff is negative; when delta_c is not a non-empty one-dimensional array; when delta_c contains a non-finite value; or when any entry of delta_c is not positive.

```python
def select_rkl_stage_counts(h: float, delta_c: np.ndarray, lam_eff: float) -> np.ndarray:
    '''Select minimum stable Runge-Kutta-Legendre stage counts for diffusion sub-steps.

    Parameters
    ----------
    h : float
        Outer step size, positive.
    delta_c : np.ndarray
        Abscissa increments of the super-time-stepping stages, shape (m,), every
        entry positive. Stage k advances over an interval of length delta_c[k] * h.
    lam_eff : float
        Safeguarded dominant eigenvalue magnitude of the diffusion operator,
        non-negative.

    Returns
    -------
    counts : np.ndarray
        Array of shape (m,) holding the selected stage count for each sub-step,
        as floats, in the order the abscissa increments were supplied.
    '''
    return counts  # placeholder
```

### Step 4

04_rkl2_super_step

Goal
----
Advance the state over one diffusion sub-step of length H using a single second-order Runge-Kutta-Legendre super-step with s internal stages and a constant forcing term.

The slope evaluated at every internal stage is the conservative-flux diffusion operator applied to that stage value, plus the constant forcing vector. The diffusion operator uses face coefficients D_{j+1/2} = (D_j + D_{j+1}) / 2, interior-node value (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2, and zero at both boundary nodes of every species. The forcing vector is added unchanged at every internal stage; it does not vary across the sub-step.

The method is the standard low-storage form of this family: the first internal stage is a damped forward step from the incoming state, and each later internal stage is a linear combination of the two preceding internal stages and the incoming state, corrected by the slope at the immediately preceding stage and by the slope evaluated once at the incoming state and reused throughout. The coefficients are those of the second-order Legendre construction for the requested stage count. The sub-step result is the last internal stage.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when forcing does not have the same shape as y; when D_nodes is not a one-dimensional array of length y.size // 3; when y, forcing or D_nodes contains a non-finite value; when H is not a finite positive scalar; when dx is not a finite positive scalar; or when s is not an integer of at least 2.

```python
def rkl2_super_step(y: np.ndarray, H: float, s: int, forcing: np.ndarray,
                    D_nodes: np.ndarray, dx: float) -> np.ndarray:
    '''Advance one diffusion sub-step by a second-order Runge-Kutta-Legendre super-step.

    Parameters
    ----------
    y : np.ndarray
        Incoming state vector of shape (3n,), stacking the three species as
        [u; v; w].
    H : float
        Length of the sub-step, positive.
    s : int
        Number of internal stages, at least 2.
    forcing : np.ndarray
        Constant forcing vector of shape (3n,), added to the diffusion slope at
        every internal stage.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.

    Returns
    -------
    y_next : np.ndarray
        State at the end of the sub-step, shape (3n,).
    '''
    return y_next  # placeholder
```

### Step 5

05_solve_reaction_stage

Goal
----
Solve one implicit reaction stage of the partitioned step by nodewise Newton iteration.

The stage equation is z - h_gamma f_R(z) = rhs, where f_R is the Brusselator reaction source evaluated pointwise: r (A - (w + 1) u + v u^2) for the u species, r (w u - v u^2) for the v species, and r ((B - w) / eps - w u) for the w species, with the state stacked as [u; v; w] on a grid of n nodes. Stationary boundary conditions are imposed by zeroing the reaction source at both boundary nodes of every species, so the stage equation reduces to z = rhs at those nodes.

Because the reaction source couples only the three species values sharing a node, the Jacobian of the stage equation is block diagonal with one three by three block per node. The iteration is therefore performed nodewise: at each pass form the residual z - h_gamma f_R(z) - rhs, stop when its maximum absolute value is below tol, and otherwise solve the three by three block systems node by node and update z by subtracting the correction. The blocks are built from the exact derivative of the reaction source with respect to the three species values at that node, with the same boundary zeroing applied, so that boundary blocks reduce to the identity.

The iteration starts from guess and performs at most max_iter passes.

The function raises ValueError when rhs is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when guess does not have the same shape as rhs; when rhs or guess contains a non-finite value; when any of h_gamma, r, A, B is not a finite scalar; when h_gamma is negative; when eps is not a finite positive scalar; when tol is not a positive scalar; when max_iter is less than 1; when a nodewise block system is singular; when the iteration produces non-finite values; or when the residual criterion is not met within max_iter passes.

```python
def solve_reaction_stage(rhs: np.ndarray, h_gamma: float, guess: np.ndarray,
                         r: float, eps: float, A: float, B: float,
                         tol: float = 1e-13, max_iter: int = 60) -> np.ndarray:
    '''Solve the implicit reaction stage equation by nodewise Newton iteration.

    Parameters
    ----------
    rhs : np.ndarray
        Right-hand side of the stage equation, shape (3n,), stacking the three
        species as [u; v; w].
    h_gamma : float
        Product of the step size and the implicit diagonal coefficient,
        non-negative.
    guess : np.ndarray
        Initial iterate for the Newton solve, shape (3n,).
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.
    tol : float
        Residual tolerance in the maximum norm, positive.
    max_iter : int
        Maximum number of Newton passes permitted, at least 1.

    Returns
    -------
    z : np.ndarray
        Stage solution of shape (3n,) satisfying the stage equation to within tol.
    '''
    return z  # placeholder
```

### Step 6

06_assemble_coupling_vector

Goal
----
Assemble the multirate-infinitesimal coupling vector that carries the advective and reactive operators into a single stage of the partitioned step, and the analogous vector used by the embedded solution.

The tableau is an implicit-explicit pair of square arrays A_impl and A_expl of the same shape, with s stages and shared abscissae, together with embedding weight vectors d_impl and d_expl of length s. The stage values already computed in the current step are supplied as f_impl and f_expl, each of shape (s, m): row j holds the reactive right-hand side and the advective right-hand side respectively, evaluated at stage j. Rows beyond those already computed are not read.

The vector is the weighted combination of those stage values in which each stage value is weighted by the corresponding first-level coupling coefficient of the multirate-infinitesimal form of the tableau pair, the implicit table supplying the weights for the reactive values and the explicit table those for the advective values, with the whole sum scaled by h. These coefficients are not the tableau entries themselves; they are the quantities the multirate-infinitesimal construction derives from a pair of tableau rows for the stage in question.

For a stage row_index between 1 and s - 1, the two rows entering that construction are tableau rows row_index and row_index - 1, and the sum runs over the stage values with index strictly below row_index.

For row_index equal to s, the vector is the one used by the embedded solution. The upper row is then the embedding weight vector rather than a tableau row, the lower row is tableau row s - 2, and the sum runs over all s stage values.

The returned vector is scaled by h in both cases and is not divided by any abscissa increment.

The function raises ValueError when A_impl is not a square two-dimensional array; when A_expl does not have the same shape as A_impl; when the tableau has fewer than 3 stages; when d_impl or d_expl is not a one-dimensional array of length s; when f_impl is not a two-dimensional array with one row per tableau stage; when f_expl does not have the same shape as f_impl; when any tableau or stage-value input contains a non-finite value; when h is not a finite positive scalar; when row_index is not an integer; or when row_index lies outside the range from 1 to s.

```python
def assemble_coupling_vector(h: float, A_impl: np.ndarray, A_expl: np.ndarray,
                             d_impl: np.ndarray, d_expl: np.ndarray,
                             row_index: int, f_impl: np.ndarray,
                             f_expl: np.ndarray) -> np.ndarray:
    '''Assemble the multirate-infinitesimal coupling vector for one stage row.

    Parameters
    ----------
    h : float
        Step size, positive.
    A_impl : np.ndarray
        Implicit Butcher matrix, shape (s, s).
    A_expl : np.ndarray
        Explicit Butcher matrix, shape (s, s).
    d_impl : np.ndarray
        Implicit embedding weights, shape (s,).
    d_expl : np.ndarray
        Explicit embedding weights, shape (s,).
    row_index : int
        Stage row to assemble, between 1 and s. A value below s selects the
        stage coupling vector; the value s selects the embedding coupling vector.
    f_impl : np.ndarray
        Reactive right-hand-side values at the stages, shape (s, m). Only the
        rows entering the requested sum are read.
    f_expl : np.ndarray
        Advective right-hand-side values at the stages, shape (s, m).

    Returns
    -------
    g : np.ndarray
        Coupling vector of shape (m,), already scaled by h.
    '''
    return g  # placeholder
```

### Step 7

07_extsts_step

Goal
----
Advance the state by one full step of the partitioned integrator and return both the solution and the embedded solution.

The step uses the six-stage padded implicit-explicit Butcher pair of this integrator family, with shared abscissae c = (0, 2 gamma, 2 gamma, 1, 1, 1) where gamma = (2 - sqrt(2)) / 2, together with its embedding weight vectors, exactly as printed in the work that introduced the family. The pair is stiffly accurate and solve-decoupled.

Set the first stage value to the incoming state. For each stage i from 1 to 5, first assemble the coupling vector g for that stage row from the two tables against the stage values already computed, then branch on the tableau structure:

- if the implicit diagonal entry is zero and the abscissa increment is positive, advance from the previous stage value by one Runge-Kutta-Legendre super-step over an interval of length H = (c[i] - c[i-1]) h, with constant forcing g / H and with the stage count selected from H and lam_eff;

- if the implicit diagonal entry is zero and the abscissa increment is zero, set the stage value to the previous stage value plus g;

- otherwise solve the implicit reaction stage equation with coefficient h times the implicit diagonal entry, using the previous stage value as the initial iterate.

After each stage, evaluate the advective and reactive components at the new stage value; the diffusive component is not needed separately, since it enters through the super-step. The solution after the step is the sixth stage value. The embedded solution is the fifth stage value plus the coupling vector assembled from the embedding weights against the second-to-last tableau row, summed over all six stage values.

The reaction stage solves use a residual tolerance of 1e-13 in the maximum norm.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y.size // 3; when y or D_nodes contains a non-finite value; when h is not a finite positive scalar; when lam_eff is not a finite scalar; when lam_eff is negative; when dx is not a finite positive scalar; when eps is not a finite positive scalar; or when any of c, r, A, B is not a finite scalar.

```python
def extsts_step(y: np.ndarray, h: float, lam_eff: float, D_nodes: np.ndarray,
                dx: float, c: float, r: float, eps: float, A: float,
                B: float) -> np.ndarray:
    '''Advance one step of the partitioned integrator, returning solution and embedding.

    Parameters
    ----------
    y : np.ndarray
        Incoming state vector of shape (3n,), stacking the three species as
        [u; v; w].
    h : float
        Step size, positive.
    lam_eff : float
        Safeguarded dominant eigenvalue magnitude of the diffusion operator, used
        to select sub-step stage counts, non-negative.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    c : float
        Advection speed.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, 3n). Row 0 is the solution after the step, row 1 is
        the embedded solution, both in the same species-stacked ordering as y.
    '''
    return result  # placeholder
```

### Step 8

08_run_pipeline

Goal
----
Run the complete pipeline end to end and return the target scalar.

Build the grid as n_nodes equally spaced points on the unit interval, with spacing dx equal to the distance between neighbours. Form the nodal diffusion coefficients as d (1 + d_var sin(2 pi x)). Form the initial state by stacking A + 0.1 sin(2 pi x), B / A + 0.1 sin(2 pi x), and B + 0.1 sin(2 pi x).

Evaluate the split components at the initial state and check that the reactive component vanishes at both boundary nodes of every species, so that the stationary boundary condition holds before any stepping begins.

Estimate the dominant eigenvalue of the diffusion operator at the initial state using the given seed, weighted-root-mean-square tolerances, stopping tolerance, and safety factor. This estimate is formed once and reused for every step. Select the internal stage counts for the two super-time-stepping stages from the step size h = T / n_steps, the abscissa increments of those two stages, and the safeguarded magnitude.

Advance the first step by assembling it explicitly: set the first stage value to the initial state, and for each subsequent stage assemble the coupling vector from the tableau row differences against the stage values already computed, then either take one Runge-Kutta-Legendre super-step with the previously selected stage count, or copy the previous stage value plus the coupling vector when the abscissa increment vanishes and the implicit diagonal is zero, or solve the implicit reaction stage. Evaluate the split components after each stage to supply the advective and reactive stage values.

Advance the remaining n_steps - 1 steps with the composite single-step routine, using the same fixed step size and the same safeguarded eigenvalue magnitude throughout. No step-size adaptivity is performed and the embedded solution is not used to modify the step size.

Return the entry of the final state at position target_index.

The function raises ValueError when n_nodes is not an integer of at least 3; when n_steps is not a positive integer; when T is not a finite positive scalar; when d is not a finite positive scalar; when target_index is not an integer index into the state vector; when the reactive component fails the boundary check at the initial state; when the split operator returns an unexpected shape; when the eigenvalue estimate is not negative; when any selected stage count falls below 2; or when the number of super-time-stepping stages encountered does not match the number of selected stage counts. Errors raised by the called sub-problem functions propagate unchanged.

```python
def run_pipeline(n_nodes: int, n_steps: int, T: float, c: float, d: float,
                 r: float, eps: float, A: float, B: float, d_var: float,
                 seed: int, rtol: float, atol: float, tau: float,
                 q_lambda: float, target_index: int) -> float:
    '''Run the full integration and return the target state entry.

    Parameters
    ----------
    n_nodes : int
        Number of spatial nodes per species, at least 3.
    n_steps : int
        Number of uniform time steps, positive.
    T : float
        Final time, positive.
    c : float
        Advection speed.
    d : float
        Diffusion strength scaling, positive.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.
    d_var : float
        Amplitude of the sinusoidal variation of the diffusion coefficient.
    seed : int
        Seed for the eigenvalue estimator's initial iterate.
    rtol : float
        Relative tolerance entering the weighted root-mean-square weights.
    atol : float
        Absolute floor entering the weighted root-mean-square weights.
    tau : float
        Relative-change stopping tolerance of the eigenvalue estimator.
    q_lambda : float
        Safety factor applied to the eigenvalue estimate.
    target_index : int
        Zero-based index into the species-stacked final state vector.

    Returns
    -------
    value : float
        The entry of the final state at target_index, as a native Python float.
    '''
    return value  # placeholder
```
