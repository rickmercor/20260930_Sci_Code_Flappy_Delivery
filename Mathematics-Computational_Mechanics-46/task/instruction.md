# Mathematics-Computational_Mechanics-46

## Background

Dynamic fragmentation of brittle ceramics sits at the core of a leading computational solid-mechanics group's research on high-strain-rate fracture of alumina. Explicit cohesive-zone finite-element models are the tool of choice in that line of work because they produce sharp crack surfaces that can be tracked as a solid breaks into fragments which then collide. Contact is where purely explicit formulations struggle. Penalty regularization inflates the largest natural frequency of the system through artificial contact stiffness, shrinking the stable time step precisely when fragmentation generates dense impacts, and the nonsmooth loading paths it induces leave persistent energy artifacts. The nonsmooth contact dynamics (NSCD) framework, developed largely by the nonsmooth-mechanics community, takes the opposite view: impacts are velocity jumps governed by complementarity conditions rather than stiff forces. Against this backdrop, the source paper bridges the two traditions. It adapts the nonsmooth Newmark scheme to a finite-element cohesive-fracture setting, keeps bulk dynamics and cohesive softening explicit, and solves contact implicitly as a convex quadratic program built on a modified Delassus operator. A stiffness cap regularizes the extrinsic cohesive law so the stable time step stays practical. Validation on a bouncing ball, an impacting bar, and 1D fragmentation benchmarks shows machine-precision conservation of algorithmic energy, larger stable time steps than penalty-based alternatives, and fragment-size statistics consistent with analytical fragmentation models. The capped law and the QP contact solve are what make dense fragment-wall and fragment-fragment contact affordable.

## Problem

Simulations of dynamic fragmentation must couple explicit cohesive-zone fracture with unilateral contact between the faces of propagating cracks; penalty regularization of contact trades accuracy against the stable time step and can inject energy artifacts. A semi-explicit nonsmooth Newmark (NSN) time integrator resolves frictionless unilateral contact at the velocity level through a complementarity problem while keeping the bulk dynamics explicit. Your task is to simulate the free-expansion fragmentation of a pre-damaged one-dimensional elastic bar with the NSN scheme, including its capped traction-separation law, and to report the single deterministic output defined below.

The method splits the dynamics into a smooth part, integrated with the explicit Newmark scheme (β = 0, γ = 1/2), and a nonsmooth contact part collected in a velocity jump that follows from a constrained quadratic program over the active contact set, with the restitution coefficient e scaling the impact law. Bulk elasticity uses a lumped-mass finite-element bar, while cohesive interfaces follow a damage-regulated Camacho-Ortiz traction-separation law with a stiffness cap that holds the traction constant (zero stiffness) whenever the secant stiffness would exceed the cap, keeping the stable time step practical. The stable time step is fixed once at setup from a Gershgorin-circle estimate of the largest natural frequency of the assembled system, including the capped cohesive stiffness.

Set up the simulation with exactly the following configuration:

  - Bar: length L = 1.0×10⁻³ m, cross-section A = 6.45×10⁻⁴ m², linear elastic AD-995 alumina with E = 370 GPa and ρ = 3900 kg/m³.
  - Mesh: Ne = 500 uniform 2-node (P1) finite elements with lumped mass; Ncoh = 250 cohesive interfaces inserted at interior nodes 1, 3, 5, …, 499 (every odd-numbered interior node; duplicate nodes there); every cohesive interface has initial damage d₀ = 1×10⁻⁶.
  - Cohesive law: strength σ_c = 262 MPa, fracture energy G_c = 50 J/m², critical opening δ_c = 2G_c/σ_c; the mixed-mode parameter is irrelevant in 1D; stiffness cap k̃ = αE/h̄e with α = 10 and h̄e = L/Ne.
  - Loading: the bar models the 1D equivalent of an expanding Mott ring, centered at the origin and expanding freely with initial velocity field v(x) = ε̇·x (strain rate ε̇ = 5×10⁴ /s), both ends free; frictionless crack-face contact only; coefficient of restitution e = 0; zero initial displacement; no external forces afterward.
  - Time stepping: wave speed c = √(E/ρ); characteristic wave transit time t_b = 2L/c; the stable time step is computed once at setup as Δt = 0.99 × (2/ω_max), where ω²_max = max_i (Σ_j |K_ij| / M_ii) uses the lumped mass and the assembled stiffness matrix K, including the capped cohesive stiffness k̃ = αE/h̄e on every interface; this setup estimate is the contract, and the Δt it produces is the only admissible setup time step (any coarser estimate, such as one that under-counts the interface stiffness rows, is not the contract). The march then advances from the initial state to t_final = 0.42·t_b, before the first complete decohesion, using N = round(t_final/Δt) equal steps of duration Δt′ = t_final/N so that it lands exactly on t_final (Δt′ may differ from the setup Δt only at the 1/N rounding level); the contact QP is solved by an active-set method to a relative dual-feasibility tolerance of 1×10⁻¹² (inactive-set residual min_i (W′p + b)_i ≥ −10⁻¹² · max(1, max|b|)). Damage must saturate at 1 (complete decohesion: the secant stiffness and the traction both vanish at and beyond d = 1). Damage is measured from the signed opening history, δ_max = max(δ, 0): compression (δ ≤ 0) never damages. All per-step damage accounting and cap-traction selection are evaluated at the step's predicted configuration ũ_{n+1} — the same configuration the active contact set is formed from: the history δ_max is advanced with the predicted openings δ(ũ_{n+1}) at the start of the step, before reassembly, and the cap-regime traction acts only on interfaces with positive predicted opening (δ(ũ_{n+1}) > 0); a closed or compressed pair carries no cap traction, and the e = 0 contact impulse alone governs it. The active contact set is formed from the predicted gap H ũ_{n+1}.
  - Output: the axial displacement of the right-end node of the bar at t_final, normalized by L; report u_end(t_final)/L rounded to 4 significant figures as the single final numeric answer. In the reasoning, also report two run-level values produced by the same march: the damage sum over all 250 interfaces at t_final, Σ_i d_i, and the accumulated normal contact impulse, the sum over all N steps of the nonnegative interface contact impulses returned by that step's contact solve, in N·s.

Derive the scheme's update equations, the active-set definition, the QP (Delassus) operator, and the capped traction-separation law from the properties stated above and the structure of the configuration bullets; the prompt intentionally leaves these unstated beyond their names. In the reasoning, state the method-specific predictor--corrector state ordering that produces the two run-level values, identify the source expanding-ring benchmark's boundary and restitution configuration and explain which aspects are changed by this task, and explain the source's rationale for introducing the capped two-regime law and the role it assigns to the crossover in the stability and cohesive-work arguments; do not add extra reported outputs. One march of the scheme to t_final under these settings determines the requested quantity, so the result is fully deterministic.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_bar_model_setup

Goal
----
Builds the duplicated-node degree-of-freedom map, the connectivity, and the lumped nodal masses for a uniform two-node bar discretization with cohesive interfaces at every other interior node, plus the two derived elastic-wave scalars used to nondimensionalize the whole simulation.

```python
def bar_model_setup(n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4) -> dict:
    """Build the duplicated-node DOF map, connectivity, and lumped masses.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2;
        interfaces sit at every other interior node, which requires an even
        element count so the last node is never itself an interface node).
    length : float
        Total bar length L, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus E, in pascals (must be > 0).
    density : float
        Bulk mass density rho, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area A, in m^2 (must be > 0).

    Returns
    -------
    setup : dict
        n_dof : int, total degree-of-freedom count (regular nodes + 2 per
            interface node).
        h_e : float, element length L / n_e.
        wave_speed : float, elastic bar wave speed sqrt(E / rho).
        bar_period : float, 2 * length / wave_speed.
        interface_nodes : (n_if,) int array, node indices carrying an
            interface (the odd interior nodes).
        dof_minus, dof_plus : (n_if,) int arrays, the minus-face and
            plus-face degree-of-freedom index of each interface node.
        element_dofs : (n_e, 2) int array, the (left, right) degree-of-freedom
            index of every bulk element, in element order.
        mass : (n_dof,) float array, lumped mass at every degree of freedom.
        mass_regular : float, lumped mass of a regular (non-end, non-interface)
            node, rho * A * h_e.
        mass_face : float, lumped mass of an end node or an interface face,
            rho * A * h_e / 2.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if length, youngs_modulus,
        density, or area is not a positive real number.
    """
    return setup
```

### Step 2

02_initial_state_energy

Goal
----
Builds the initial free-expansion velocity field and its kinetic energy, the setup-time interface stiffness cap, and the full-row Gershgorin stability bound that fixes the march's fixed time step and step count.

```python
def initial_state_energy(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, t_star_over_tb: float = 0.42, safety: float = 0.99) -> dict:
    """Initial kinetic energy, interface cap, and the Gershgorin setup step.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0); the
        capped interface spring per unit area is alpha * youngs_modulus /
        h_e, with h_e = length / n_e.
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period 2*length/wave_speed
        (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    result : dict
        E0_discrete : float, initial kinetic energy from the lumped masses.
        E0_continuum : float, the continuum kinetic-energy estimate for the
            same velocity field.
        momentum : float, total initial momentum (nominally zero).
        k_tilde : float, capped interface spring per unit area.
        k_tilde_A : float, capped interface spring, k_tilde * area.
        k_bulk_element : float, bulk element spring youngs_modulus*area/h_e.
        omega2_max : float, full-row Gershgorin bound on the squared highest
            natural frequency.
        omega_max : float, sqrt(omega2_max).
        dt_setup : float, safety * 2 / omega_max.
        N_steps : int, round(t_final / dt_setup) with t_final =
            t_star_over_tb * bar_period.
        dt_prime : float, t_final / N_steps.
        convexity_margin : float, dt_prime * omega_max / 2 (must stay < 1
            for the contact solve of later steps to remain a convex QP).
        bar_period : float, 2 * length / wave_speed.
        wave_speed : float, sqrt(youngs_modulus / density).
        t_final : float, t_star_over_tb * bar_period.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if edot, alpha, length,
        youngs_modulus, density, area, t_star_over_tb, or safety is not a
        positive real number.
    """
    return result
```

### Step 3

03_cohesive_law_state

Goal
----
Evaluates the stiffness-capped Camacho-Ortiz traction-separation law for a set of interfaces given their current opening and opening history: it updates the monotone damage history, locates the crossover damage where the damage-secant stiffness first drops below the setup-time cap, and returns the traction each interface carries in whichever of the two regimes it currently occupies.

```python
def cohesive_law_state(delta, dmax_open, n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0) -> dict:
    """Evaluate the capped Camacho-Ortiz law for a set of interfaces.

    Parameters
    ----------
    delta : array_like
        Current signed opening of each interface, in metres.
    dmax_open : array_like
        Largest opening previously sustained by each interface, in metres
        (same shape as delta).
    n_e : int
        Number of bulk elements used only to size the element length that
        sets the interface stiffness cap (must be an even integer >= 2).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).

    Returns
    -------
    result : dict
        delta_c : float, critical opening 2*Gc/sigma_c.
        envelope_slope : float, sigma_c/delta_c.
        d_tilde : float, crossover damage where the secant stiffness equals
            the setup-time cap.
        k_d0 : float, secant stiffness evaluated at damage 1e-6.
        cap_ratio : float, k_d0 divided by the setup-time cap.
        damage : ndarray, updated damage of each interface, clamped to 1.
        dmax_open : ndarray, updated opening history of each interface.
        traction : ndarray, interface traction in whichever regime each
            interface currently occupies.

    Raises
    ------
    ValueError
        If delta and dmax_open do not share the same shape, if n_e is not
        an even integer >= 2, or if length, youngs_modulus, alpha, sigma_c,
        or Gc is not a positive real number.
    """
    return result
```

### Step 4

04_predictor_forces

Goal
----
Builds the smooth explicit predictor of one time step: it assembles the tridiagonal bulk-plus-secant-interface stiffness for the current damage state, predicts the displacement half a step ahead with no contact correction, assembles the cap-regime traction force on interfaces that are still below the crossover damage, and forms the contact-free velocity that later feeds the contact solve.

```python
def predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e: float, d_tilde: float, dt: float, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0, area: float = 6.45e-4) -> dict:
    """Assemble K(d), the smooth predictor, and the cap-regime force.

    Parameters
    ----------
    u, v, a : array_like
        Current displacement, velocity, and acceleration at every degree of
        freedom (same shape, length n_dof).
    damage : array_like
        Current damage of each interface, in [0, 1] (length n_if).
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        (length n_if, integer).
    element_dofs : array_like
        (n_bulk, 2) integer array, the (left, right) degree-of-freedom index
        of every bulk element.
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_e : float
        Bulk element spring constant (must be > 0).
    d_tilde : float
        Crossover damage separating the cap regime from the secant regime
        (must be in (0, 1)).
    dt : float
        Time step (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0); the
        assembled secant spring at damage d is (1-d)/d * sigma_c/delta_c *
        area, with delta_c = 2*Gc/sigma_c.
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).

    Returns
    -------
    result : dict
        u_tilde : ndarray, the predicted displacement (length n_dof).
        v_free : ndarray, the contact-free velocity (length n_dof).
        fcap : ndarray, the cap-regime traction force (length n_dof).
        k_diag : ndarray, diagonal of the assembled stiffness (length n_dof).
        k_up : ndarray, superdiagonal of the assembled stiffness (length
            n_dof - 1).
        delta_pred : ndarray, predicted opening of each interface (length
            n_if).

    Raises
    ------
    ValueError
        If u, v, and a do not share the same shape, if dt or k_e is not
        positive, or if alpha, sigma_c, or Gc is not a positive real number.
    """
    return result
```

### Step 5

05_contact_qp_solve

Goal
----
Solves the velocity-level contact problem for one time step: it forms the gap and active set from the predicted displacement, assembles the modified Delassus operator that already accounts for the smooth dynamics' own stiffness coupling, and returns the normal impulse each active interface pair needs to satisfy the prescribed restitution relation.

```python
def contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, u_tilde, v_free, v_n, dt: float, n_dof: int, e_restitution: float = 0.0) -> dict:
    """Solve the velocity-level contact QP for the active interface pairs.

    Parameters
    ----------
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        pair (length n_if, integer).
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_diag : array_like
        Diagonal of the assembled stiffness at the predicted configuration
        (length n_dof).
    k_up : array_like
        Superdiagonal of the assembled stiffness (length n_dof - 1).
    u_tilde : array_like
        Predicted displacement at every degree of freedom (length n_dof).
    v_free : array_like
        Contact-free velocity at every degree of freedom (length n_dof).
    v_n : array_like
        Velocity at the beginning of the current time step (length n_dof).
    dt : float
        Time step (must be > 0).
    n_dof : int
        Total number of degrees of freedom (must be >= 2).
    e_restitution : float
        Restitution coefficient, dimensionless (must be >= 0; 0 is
        perfectly inelastic contact).

    Returns
    -------
    result : dict
        p : ndarray, the normal impulse of every interface pair (length
            n_if; zero on pairs outside the active set).
        gap : ndarray, the signed opening of every interface pair at the
            predicted configuration (length n_if).
        active : ndarray, integer indices of the pairs in the active set.
        residual : float, the worst dual-feasibility residual among the
            inactive pairs at the solution (0.0 when the active set is
            empty).

    Raises
    ------
    ValueError
        If dt is not positive, if e_restitution is negative, or if n_dof is
        smaller than 2.
    """
    return result
```

### Step 6

06_nonsmooth_corrector

Goal
----
Folds the contact impulse into the state at the end of a time step: it turns the impulse into a velocity jump, corrects the predicted displacement with half of that jump, recomputes the acceleration at the corrected configuration, and forms the trapezoidal velocity update with the jump added on top.

```python
def nonsmooth_corrector(u_tilde, v, a, p, dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt: float) -> dict:
    """Fold the contact impulse into the end-of-step state.

    Parameters
    ----------
    u_tilde : array_like
        Predicted displacement at every degree of freedom (length n_dof).
    v, a : array_like
        Velocity and acceleration at the start of the step (length n_dof
        each, same shape as u_tilde).
    p : array_like
        Normal impulse of every interface pair (length n_if; zero outside
        the active set).
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        pair (length n_if, integer).
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_diag : array_like
        Diagonal of the assembled stiffness (length n_dof).
    k_up : array_like
        Superdiagonal of the assembled stiffness (length n_dof - 1).
    fcap : array_like
        Cap-regime traction force at every degree of freedom (length
        n_dof).
    dt : float
        Time step (must be > 0).

    Returns
    -------
    result : dict
        u_next : ndarray, the corrected end-of-step displacement (length
            n_dof).
        v_next : ndarray, the end-of-step velocity (length n_dof).
        a_next : ndarray, the end-of-step acceleration (length n_dof).
        bv : ndarray, the velocity jump produced by the impulse (length
            n_dof).

    Raises
    ------
    ValueError
        If dt is not positive, or if u_tilde, v, and a do not share the
        same shape.
    """
    return result
```

### Step 7

07_nsn_time_march

Goal
----
Runs the full semi-explicit nonsmooth Newmark march from the free-expansion initial state to the requested horizon, chaining the smooth predictor, the damage update, the contact QP, and the nonsmooth corrector every step, and reports the end-of-run displacement together with the damage, impulse, and event-census diagnostics that verify the run.

```python
def nsn_time_march(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> dict:
    """Run the full NSN march and report the end-of-run diagnostics.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    d0 : float
        Initial damage of every interface (must lie in (0, 1)).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period (must be > 0).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    result : dict
        u_end_over_L : float, the right-end displacement at the horizon,
            normalized by length.
        damage_sum : float, the sum of every interface's damage at the
            horizon.
        total_impulse : float, the accumulated normal impulse magnitude
            over the whole march.
        secant_predicted_max_ratio : float, the largest ratio of secant
            traction to sigma_c among predicted-open secant-regime
            interfaces over the whole march.
        N : int, the number of steps taken.
        dt_prime : float, the fixed step used by the march.
        first_spall_step : int, the 1-indexed step at which any interface
            first reaches damage 0.999, or -1 if this never happens.
        first_contact_step : int, the 1-indexed step at which the active
            set first becomes nonempty, or -1 if this never happens.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if edot, alpha, sigma_c, Gc,
        t_star_over_tb, length, youngs_modulus, density, area, or safety is
        not a positive real number, or if d0 does not lie in (0, 1).
    """
    return result
```

### Step 8

08_run_nsn_fragmentation

Goal
----
Chains the geometry, initial state, cohesive law, predictor, contact solve, and corrector physics of the preceding steps, step by step, into the complete free-expansion fragmentation run at the requested configuration, cross-checks the result against the independently-expressed full march, and extracts the single requested observable: the right-end displacement at the horizon, normalized by the bar length.

```python
def run_nsn_fragmentation(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> float:
    """Run the full NSN fragmentation pipeline and return the normalized
    end displacement.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    d0 : float
        Initial damage of every interface (must lie in (0, 1)).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period (must be > 0).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    u_end_over_L : float
        The right-end displacement at the horizon, normalized by length.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, if edot, alpha, sigma_c, Gc,
        t_star_over_tb, length, youngs_modulus, density, area, or safety is
        not a positive real number, if d0 does not lie in (0, 1), or if the
        step-by-step chain disagrees with the fused march beyond machine
        tolerance.
    """
    return u_end_over_L
```
