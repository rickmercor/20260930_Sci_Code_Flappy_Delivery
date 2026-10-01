# What sets the switching stimulus of a bistable epigenetic mark?

## Background

Epigenetic marks are heritable modifications of gene expression that leave the DNA sequence untouched, and two of their features are awkward for the kinetic models usually written for them. They are bistable: cells commit to discrete phenotypes separated by barriers, the picture Waddington drew as a landscape and that dynamical systems render as a regulatory switch. And they display memory: a mark laid down in response to a micro-environmental stimulus persists after the stimulus is withdrawn, and erasing it requires crossing a different threshold from the one that established it. Hypoxia is the canonical instance, suppressing the oxygen-dependent demethylation machinery and leaving a methylation imprint that outlives the episode. The two committed states are not in general iso-energetic, so the landscape that represents them is asymmetric, and the asymmetry is a free-energy difference an independent thermodynamic measurement can supply.

Threshold activation, path dependence and a residual state after unloading are exactly the signatures of rate-independent dissipative evolution, the class of processes whose response is invariant under reparametrisation of time. That class has a mature variational theory, developed largely in solid mechanics for plasticity, damage, fracture and phase transformation, in which a model is fixed by three ingredients and nothing else: a state space, a stored energy depending on the state and on an external loading, and a convex, one-homogeneous dissipation potential encoding the resistance to change. Transplanting it to chromatin makes the Waddington landscape the stored energy, the micro-environment the loading, and the resistance of the remodelling machinery the dissipation threshold. The appeal is that once the three are fixed, every governing equation follows, and the resulting model conserves energy and produces non-negative entropy by construction rather than by later repair.

The theory's own difficulty, however, is that the three ingredients do not by themselves fix the evolution. Postulating that the state is at every instant a global minimiser of the stored energy plus the dissipative cost of reaching it gives the energetic formulation, whose existence theory is strong precisely because the comparison ranges over the whole state space. That same global quantifier is its weakness on a non-convex landscape: it will transport the state into a basin that no continuous path of equilibria reaches, and it will do so as soon as the far basin wins the comparison, which can be well before the occupied basin has ceased to be locally stable. The alternative reading, obtained as the vanishing-viscosity limit of a regularised evolution, keeps the state on the branch it occupies until that branch runs out. The two agree wherever the landscape is convex and part company exactly at a switch, which for an epigenetic model is the event of interest.

Deciding between them is not a matter of numerical accuracy. Both are integrated by the same backward-Euler incremental minimisation, which reduces on a single branch to the return-mapping algorithm of computational plasticity and differs between the two readings only in whether the minimisation is taken over the whole state space or over a neighbourhood of the current state. That scheme carries a proven first-order bound on the defect of its discrete energy balance, and no general bound at all on the error of the state, so the honest way to certify a computed switching instant is to confirm the energy balance closes at the expected rate rather than to refine until the trajectory stops moving.

What this leaves open is an identifiability question with direct experimental consequences. A calibration exercise never starts from the constants of the landscape; it starts from what the assay returns over a stimulus cycle, and has to work backwards. If the observables an assay ordinarily reports are insensitive to which principle generated them, then a model fitted to a hysteresis record may reproduce that record perfectly and still be undetermined at the one instant the experiment was run to locate. Settling that requires recovering one landscape from one record, driving it through the same closed cycle under both principles on one and the same partition, and comparing not only where each puts the switch but what, if anything, about the two cycles could ever tell them apart.

## Problem

A chromatin mark at one locus shows threshold activation, path dependence and a residual level that outlives the stimulus, so I treat it as rate-independent dissipative evolution of a scalar state $q$: a stored energy $E(q,S) = F(q) - q\,\ell(S)$ in which the micro-environmental stimulus $S$ enters only through a saturating interaction potential, and a $1$-homogeneous dissipation potential $\Psi(\dot q) = \rho\,|\dot q|$ standing for the resistance of the remodelling machinery. The landscape $F$ is bistable and asymmetric: two wells of common curvature $k$, each quadratic in the departure from its own bottom, with bottoms at $-a$ and $+b$ and a corner between them at $q = 0$, where $F$ itself is continuous; the repressed well is the zero of energy. Calorimetry on the two marked states puts the active one $\Delta = 0.00603213$ above the repressed one. I have no independent handle on $k$, $a$, $b$ or $\rho$.

I drove a mark resting at the bottom of the repressed well through one closed hypoxia-like excursion, $S(t) = \tfrac12 S_{\max}\bigl(1 - \cos(2\pi t/T)\bigr)$ coupled through $\ell(S) = \ell_\infty\bigl(1 - e^{-\lambda S}\bigr)$, with $\ell_\infty = 0.5$, $\lambda = 1$, $S_{\max} = 5$ and $T = 1$. Over that excursion the assay returns a peak level of $0.448487$, a residual level of $0.165654$ once the stimulus is back at baseline, and an irreversible cost, the dissipation potential accumulated along the trajectory, of $0.0835480$.

Two solution concepts for this class of rate-independent evolution are compatible with the structure, and they disagree about when the mark flips: the energetic solution, and the balanced-viscosity solution reached as the vanishing-viscosity limit. Work out for yourself what each of the two selects at a barrier this landscape puts in the way, and integrate both by backward-Euler incremental minimization on one uniform partition of $[0, T]$ into $N = 4000$ intervals, locating each crossing at the first node whose state is strictly positive.

Tell me what this record implies about how the mark switches, and what it leaves undecided. Address in particular: which features of the landscape and of the trajectory the assay pins down whichever of the two principles produced it, and why a residual survives at baseline at all; which single observable of the loop would have to be measured to tell the two principles apart, how far apart it puts them, and where the energy they account for differently ends up; and what becomes of the disagreement in the two limits that matter here, a landscape whose wells merge and one whose wells sit at equal energy. Short assertions carrying their numbers are what is wanted, not derivations. Your final answer must be a single number: the interaction potential $\ell$ at the globally selected crossing divided by the interaction potential $\ell$ at the branch-tracking crossing.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_build_loading_path

Goal
----
Tabulate the micro-environmental stimulus protocol and the interaction potential it induces at every node of a uniform partition of the observation window.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_loading_path(n_steps: int, T: float = 1.0, S_max: float = 5.0,
                       ell_inf: float = 0.5, lam: float = 1.0) -> np.ndarray:
    """Tabulate the stimulus protocol and its interaction potential.

    The window [0, T] carries a uniform partition of n_steps intervals, so it
    has n_steps + 1 nodes, the first at t = 0 and the last at t = T. The
    stimulus follows one raised-cosine excursion of amplitude S_max over the
    whole window, and the interaction potential saturates exponentially in the
    stimulus towards the ceiling ell_inf at rate lam.

    Parameters
    ----------
    n_steps : int
        Number of intervals of the partition, n_steps >= 1.
    T : float
        Length of the observation window, T > 0.
    S_max : float
        Peak value of the stimulus, S_max > 0, attained at t = T / 2.
    ell_inf : float
        Saturation ceiling of the interaction potential, ell_inf > 0.
    lam : float
        Saturation rate of the interaction potential, lam > 0.

    Returns
    -------
    loading : np.ndarray
        Array of shape (n_steps + 1, 3) whose columns hold, in order, the
        nodal time, the stimulus at that time and the interaction potential
        at that stimulus.

    Raises
    ------
    ValueError
        If n_steps is not an integer, if n_steps is below one, or if T,
        S_max, ell_inf or lam is not a finite number strictly greater than
        zero.
    """
    return loading  # placeholder
```

### Step 2

02_evaluate_double_well

Goal
----
Evaluate the configurational free energy of a bistable chromatin landscape and the conjugate force it exerts, at an arbitrary set of states.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def evaluate_double_well(q, k: float = 1.0, a: float = 0.15,
                         b: float = None) -> np.ndarray:
    """Evaluate the bistable landscape and its conjugate force.

    The landscape holds two wells of curvature k whose bottoms sit at -a and
    at +b, meeting at the neutral state q = 0. The repressed well, the one
    centred at -a, is taken as the zero of energy, and the free energy itself
    is continuous at the corner; that continuity fixes the level of the far
    well, so no separate offset is supplied. The state q = 0 belongs to the
    well centred at -a.

    Parameters
    ----------
    q : array_like
        States at which the landscape is evaluated; any shape is accepted and
        the result is flattened in C order.
    k : float
        Curvature of each well, k > 0.
    a : float
        Distance from the corner to the bottom of the repressed well, a >= 0;
        a = 0 collapses the landscape to a single well.
    b : float or None
        Distance from the corner to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    landscape : np.ndarray
        Array of shape (2, q.size) whose first row holds the configurational
        free energy at each state and whose second row holds the conjugate
        force, the derivative of that energy with respect to the state.

    Raises
    ------
    ValueError
        If q holds no state or any non-finite state, if k is not a finite
        number strictly greater than zero, if a is not a finite non-negative
        number, or if b is neither None nor a finite non-negative number.
    """
    return landscape  # placeholder
```

### Step 3

03_solve_incremental_step

Goal
----
Advance a bistable chromatin state by one increment of the globally minimizing variational scheme, given the previous state and the interaction potential at the new node.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_incremental_step(q_prev: float, ell: float, k: float = 1.0,
                           a: float = 0.15, rho: float = 0.10,
                           b: float = None) -> float:
    """Advance the state by one globally minimising increment.

    The stored energy of the state q under the interaction potential ell is
    the configurational free energy of the bistable landscape, less q times
    ell; that free energy is k (q + a) ** 2 / 2 on q <= 0 and
    k (q - b) ** 2 / 2 + k (a ** 2 - b ** 2) / 2 on q > 0, so the two
    branches agree in value at the barrier. The dissipation cost of moving
    from q_prev to q is rho times the absolute increment. Ties are resolved
    in favour of the candidate closest to q_prev, and then in favour of the
    smaller state, so that a state which is only marginally displaced by the
    comparison stays where it is.

    Parameters
    ----------
    q_prev : float
        State at the previous node of the partition.
    ell : float
        Interaction potential at the new node.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    q_next : float
        The state at the new node, as a native Python float.

    Raises
    ------
    ValueError
        If q_prev, ell, k, a or rho is not a finite number, if k or rho is
        not strictly greater than zero, if a is negative, or if b is neither
        None nor a finite non-negative number.
    """
    return q_next  # placeholder
```

### Step 4

04_return_map_step

Goal
----
Advance a state by one increment along a single branch of the landscape, holding it fixed while the drive stays inside the elastic range and otherwise returning it to the yield surface.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def return_map_step(q_prev: float, ell: float, rho: float = 0.10,
                    stiffness: float = 1.0, offset: float = 0.15,
                    curvature: float = 0.0, tol: float = 1.0e-13,
                    max_iter: int = 100) -> float:
    """Advance the state by one increment along a single branch.

    The branch force is stiffness * (q + offset) + curvature * sinh(q), which
    is strictly increasing in q for the admitted parameters, so the corrected
    state is unique whenever a correction is called for.

    Parameters
    ----------
    q_prev : float
        State at the previous node of the partition.
    ell : float
        Interaction potential at the new node.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0; the
        elastic range of sustainable drives is [-rho, rho].
    stiffness : float
        Linear coefficient of the branch force, stiffness > 0.
    offset : float
        Displacement of the well bottom of this branch, so that the branch
        force vanishes at q = -offset.
    curvature : float
        Coefficient of the anharmonic part of the branch force,
        curvature >= 0; zero recovers a linear branch.
    tol : float
        Absolute tolerance on the residual of the corrector, tol > 0.
    max_iter : int
        Maximum number of corrector iterations, max_iter >= 1.

    Returns
    -------
    q_next : float
        The state at the new node, as a native Python float.

    Raises
    ------
    ValueError
        If q_prev, ell, rho, stiffness, offset, curvature or tol is not a
        finite number, if rho, stiffness or tol is not strictly greater than
        zero, if curvature is negative, if max_iter is not an integer of at
        least one, or if the branch force cannot be bracketed around the
        target within the doubling budget.
    """
    return q_next  # placeholder
```

### Step 5

05_integrate_energetic_evolution

Goal
----
Integrate the whole chromatin trajectory over a tabulated loading path by chaining the globally minimizing increment node by node.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def integrate_energetic_evolution(loading: np.ndarray, k: float = 1.0,
                                  a: float = 0.15, rho: float = 0.10,
                                  q_initial: float = -0.15,
                                  b: float = None) -> np.ndarray:
    """Integrate the trajectory selected by the global energetic principle.

    The trajectory starts at q_initial at the first node of the loading table
    and is advanced by one globally minimising increment per subsequent node,
    each increment anchored at the state returned by the previous one.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node; the first two columns are not used here.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node, taken as given.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at every node.

    Raises
    ------
    ValueError
        If loading is not a finite two-dimensional array of shape
        (n_nodes, 3) with at least one node, if q_initial, k, a or rho is not
        a finite number, if k or rho is not strictly greater than zero, if a
        is negative, or if b is neither None nor a finite non-negative
        number.
    """
    return trajectory  # placeholder
```

### Step 6

06_track_local_branch

Goal
----
Follow the state along the branch of equilibria of the initially occupied well alone, node by node, until that branch stops admitting a state inside its own well.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def track_local_branch(loading: np.ndarray, k: float = 1.0, a: float = 0.15,
                       rho: float = 0.10, q_initial: float = -0.15,
                       b: float = None) -> np.ndarray:
    """Follow the occupied branch, cross when it is exhausted, then follow the far one.

    The initial state must lie in the well centred at -a, that is q_initial
    must not be positive. The branch force of the occupied well is
    k * (q + a), and the state is advanced along it by one predictor-corrector
    increment per node. The branch is exhausted at the first node whose
    increment returns a state beyond the barrier. At that node the state is
    carried across to the far well by a passage that leaves its displacement
    from the well bottom unchanged, so the state recorded at that node is the
    exhausting one moved by exactly a + b. Every later node is advanced along
    the branch of the far well, whose branch force is k * (q - b), by the same
    increment.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node; the first two columns are not used here.
    k : float
        Curvature of the well, k > 0.
    a : float
        Displacement of the well bottom from the barrier, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node; must satisfy q_initial <= 0.
    b : float or None
        Displacement of the far well bottom from the barrier, b >= 0. None
        takes the two wells to be symmetric, b = a.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at every node, on the
        occupied branch up to the node that exhausts it and on the far branch
        from that node onwards.

    Raises
    ------
    ValueError
        If loading is not a finite two-dimensional array of shape
        (n_nodes, 3) with at least one node, if k or rho is not a finite
        number strictly greater than zero, if a is not finite and
        non-negative, if b is neither None nor a finite non-negative number,
        or if q_initial is not a finite number no greater than zero.
    """
    return trajectory  # placeholder
```

### Step 7

07_energy_balance_defect

Goal
----
Measure by how much a discrete trajectory fails to close the exact energy balance of the continuous evolution over the whole loading path.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def energy_balance_defect(loading: np.ndarray, trajectory: np.ndarray,
                          k: float = 1.0, a: float = 0.15,
                          rho: float = 0.10, b: float = None) -> float:
    """Measure the accumulated defect of the discrete energy balance.

    The stored energy of the state q under the interaction potential ell is
    the configurational free energy of the bistable landscape less q times
    ell. The work is credited at frozen state, so the increment between two
    consecutive nodes uses the state at the earlier node with the potential at
    both. The defect is the initial stored energy plus the accumulated work,
    less the final stored energy and the accumulated dissipation.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node.
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at each node.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    defect : float
        The accumulated energy-balance defect, as a native Python float.

    Raises
    ------
    ValueError
        If loading is not a two-dimensional array of shape (n_nodes, 3) with
        at least two nodes, if trajectory does not hold one state per node of
        that table, if either is not finite throughout, if k or rho is not a
        finite number strictly greater than zero, if a is not a finite
        non-negative number, or if b is neither None nor a finite
        non-negative number.
    """
    return defect  # placeholder
```

### Step 8

08_measure_energy_consistency_order

Goal
----
Refine the partition over a sequence of step counts and fit the observed decay order of the energy-balance defect against the step size.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def measure_energy_consistency_order(step_counts, k: float = 1.0, a: float = 0.15,
                                     rho: float = 0.10, q_initial: float = -0.15,
                                     T: float = 1.0, S_max: float = 5.0,
                                     ell_inf: float = 0.5, lam: float = 1.0,
                                     b: float = None) -> float:
    """Fit the decay order of the energy-balance defect under refinement.

    For every step count the loading path is rebuilt on the corresponding
    uniform partition, the globally minimising trajectory is integrated over
    it, and the accumulated energy-balance defect is recorded. The reported
    order is the slope of an ordinary least-squares fit of the natural
    logarithm of the defect against the natural logarithm of the step size.

    Parameters
    ----------
    step_counts : sequence of int
        Two or more distinct partition sizes, each at least one.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node of every partition.
    T, S_max, ell_inf, lam : float
        Settings of the stimulus protocol and its interaction potential.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    order : float
        The fitted decay order of the defect in the step size, as a native
        Python float.

    Raises
    ------
    ValueError
        If step_counts holds fewer than two partition sizes, holds a repeated
        size, or holds an entry below one; if T, S_max, ell_inf, lam, k or
        rho is not a finite number strictly greater than zero; if a is not a
        finite non-negative number; if b is neither None nor a finite
        non-negative number; if q_initial is not a finite number; or if the
        energy-balance defect fails to be positive at some partition of the
        sweep, leaving no order to fit.
    """
    return order  # placeholder
```

### Step 9

09_summarize_cycle

Goal
----
Reduce a discrete trajectory over a loading path to the small set of quantities an experiment on the mark could actually report.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def summarize_cycle(loading: np.ndarray, trajectory: np.ndarray,
                    rho: float = 0.10) -> np.ndarray:
    """Reduce one cycle of the trajectory to its reportable quantities.

    A crossing of the barrier is detected at the first node whose state is
    strictly positive; the two crossing entries are set to -1 when the state
    never becomes positive, a value the protocol cannot otherwise produce
    since both the time and the interaction potential are non-negative.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) holding the nodal time, the stimulus and
        the interaction potential; the stimulus column is not used here.
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at each node.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.

    Returns
    -------
    summary : np.ndarray
        Array of shape (6,) holding, in order, the peak state over the path,
        the residual state at the last node, the accumulated dissipation, the
        signed work supplied by the changing interaction potential, computed
        as minus the trapezoidal integral of state with respect to interaction
        potential along the nodal path, the interaction potential at the first
        node beyond the barrier and the time at that node. For a closed cycle
        in interaction potential, the fourth entry is the oriented hysteresis-
        loop area; it is not absolute-valued.

    Raises
    ------
    ValueError
        If loading is not a two-dimensional array of shape (n_nodes, 3) with
        at least two nodes, if trajectory does not hold one state per node of
        that table, if either is not finite throughout, or if rho is not a
        finite number strictly greater than zero.
    """
    return summary  # placeholder
```

### Step 10

10_identify_material_parameters

Goal
----
Recover the curvature, the two well positions and the dissipation threshold of a bistable rate-independent mark from the turning points of one closed hysteresis loop together with the free-energy difference between its two states.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def identify_material_parameters(q_peak: float, q_residual: float,
                                 dissipation: float, delta: float,
                                 ell_max: float) -> np.ndarray:
    """Recover the landscape and the dissipation threshold from a loop record.

    The record is produced by a mark that starts at the bottom of the
    repressed well, is carried once past the barrier onto the active branch
    while the interaction potential rises to ell_max, and is then unloaded to
    zero potential without returning across the barrier. Over such a loop the
    peak state is b + (ell_max - rho) / k, the residual state is b + rho / k,
    and the accumulated dissipation is rho times the total variation of the
    trajectory. The free energy of the active well relative to the repressed
    one is delta = k * (a ** 2 - b ** 2) / 2, which follows from insisting
    that the free energy be continuous at the barrier.

    Parameters
    ----------
    q_peak : float
        Largest state reached over the loop.
    q_residual : float
        State left at the end of the loop, at zero interaction potential;
        0 < q_residual < q_peak.
    dissipation : float
        Accumulated dissipation over the loop, dissipation > 0.
    delta : float
        Free energy of the active well above the repressed one.
    ell_max : float
        Ceiling of the interaction potential over the loop, ell_max > 0.

    Returns
    -------
    parameters : np.ndarray
        Array of shape (4,) holding, in order, the curvature k, the distance a
        from the barrier to the bottom of the repressed well, the distance b
        from the barrier to the bottom of the active well, and the dissipation
        threshold rho.

    Raises
    ------
    ValueError
        If q_peak, q_residual, dissipation, delta or ell_max is not a finite
        number, if the record does not satisfy 0 < q_residual < q_peak, if
        dissipation or ell_max is not strictly positive, or if the record
        admits no bistable landscape carrying the given offset.
    """
    return parameters  # placeholder
```

### Step 11

11_run_epigenetic_switch_comparison

Goal
----
Orchestrator of sub-problems 01-10, calling build_loading_path, evaluate_double_well, solve_incremental_step, return_map_step, integrate_energetic_evolution, track_local_branch, energy_balance_defect, measure_energy_consistency_order, summarize_cycle and identify_material_parameters to recover a bistable mark's landscape from one assay record, drive it through the same stimulus cycle under both selection principles, and return the ratio of the interaction potentials at which they carry it past the barrier.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_loading_path, evaluate_double_well, solve_incremental_step, return_map_step, integrate_energetic_evolution, track_local_branch, energy_balance_defect, measure_energy_consistency_order, summarize_cycle, identify_material_parameters) rather than reimplementing them.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_epigenetic_switch_comparison(q_peak: float = 0.448487,
                                     q_residual: float = 0.165654,
                                     dissipation: float = 0.0835480,
                                     delta: float = 0.00603213,
                                     ell_inf: float = 0.5,
                                     lam: float = 1.0,
                                     S_max: float = 5.0,
                                     T: float = 1.0,
                                     n_steps: int = 4000,
                                     refinement=None) -> float:
    """Compare the two selection principles on one stimulus cycle.

    The landscape and the dissipation threshold are recovered from the assay
    record before anything is integrated. Both arms are then integrated on the
    same uniform partition of [0, T] into n_steps intervals, from a mark
    resting at the bottom of the repressed well, under the same stimulus
    protocol. The crossing of each arm is the first node at which its state is
    strictly positive, and the reported ratio divides the interaction
    potential at the globally selected crossing by the interaction potential
    at the branch-tracking crossing.

    Parameters
    ----------
    q_peak, q_residual, dissipation : float
        The assay record of one closed stimulus cycle: the peak state, the
        residual state at baseline and the accumulated dissipation.
    delta : float
        Free energy of the active well above the repressed one, measured
        independently of the cycle.
    ell_inf, lam, S_max, T : float
        Settings of the stimulus protocol and its interaction potential.
    n_steps : int
        Number of intervals of the partition, n_steps >= 2.
    refinement : sequence of int or None
        Partition sizes of the consistency check; None uses the dyadic sweep
        100, 200, 400, 800, 1600, 3200 and 6400.

    Returns
    -------
    ratio : float
        The interaction potential at the globally selected crossing divided
        by the interaction potential at the branch-tracking crossing, as a
        native Python float.

    Raises
    ------
    ValueError
        If n_steps is not an integer of at least two, if ell_inf, lam, S_max
        or T is not a finite number strictly greater than zero, if the
        landscape recovered from the record carries no barrier or cannot
        retain the mark at baseline, if the stimulus is too weak for either
        principle to carry the mark across, if the discrete energy balance is
        violated or fails to close under refinement, if the identified
        landscape does not reproduce the recorded peak and residual, or if
        either arm leaves the mark short of the barrier.
    """
    return ratio  # placeholder
```
