# Biology-Biochemistry-6

## Background

When bacteria run short of a nutrient they reallocate their proteome and change the levels of their intracellular metabolites. Quantitative proteomics across growth conditions shows that enzyme levels change largely in step with growth rate, and it has become a central question how the kinetics of individual enzymes and the architecture of their pathways decide which enzymes are induced and which are repressed.

Resource-allocation models address this by treating the cell's state as an optimisation of growth rate, with metabolic reactions following enzyme kinetics and every intracellular metabolite diluted by growth, so that enzyme and metabolite levels are predicted together rather than separately.

The task below applies such a growth-optimal allocation model to a single linear pathway.

## Problem

I want to know how the growth-optimal proteome of one of my bacterial biosynthetic pathways reallocates as its carbon source runs short, starting from kinetic constants I have measured. I treat the pathway as the whole proteome: a transporter T followed by a linear chain of twelve irreversible single-substrate Michaelis-Menten enzymes E₁, …, E₁₂, where T converts the external nutrient, held by the medium at level ν and not diluted, into the internal metabolite m₁, enzyme Eᵢ converts mᵢ (level ρᵢ) into mᵢ₊₁, and E₁₂ converts m₁₂ into protein. All levels and enzyme amounts are mass fractions of total protein and all fluxes are per unit total protein, so the fluxes are j_T = κ_T φ_T ν / (K_T + ν) and jᵢ = κᵢ φᵢ ρᵢ / (Kᵢ + ρᵢ); in balanced growth at rate λ each internal metabolite is produced faster than it is consumed by exactly its dilution λρᵢ, protein is made at j₁₂ = λ, and φ_T + Σᵢ φᵢ = 1, with no mass budget on the metabolites. My measured constants for E₁ to E₁₂ are κ = (23.0, 9.3, 36.0, 5.9, 16.8, 12.6, 44.0, 7.3, 28.5, 10.7, 19.5, 8.1) per hour and K = (2.4, 6.1, 1.3, 9.0, 3.3, 19.0, 0.75, 4.4, 2.8, 12.0, 5.2, 8.3) × 10⁻⁴, and for the transporter κ_T = 22.5 per hour and K_T = 1.2 × 10⁻³. In the rich medium ν = 0.6 and all thirteen amounts take the values that maximise λ, which fixes the rich growth rate λ_hi and the rich transporter amount φ_T,hi. When carbon is scarce the transporter amount is no longer free but follows φ_T(ν) = φ₀ (1 + 2 / (1 + 3ν / K_T)), with φ₀ set so that φ_T(0.6) = φ_T,hi, while the twelve enzyme amounts still maximise λ at the prevailing ν; lowering ν in this regime lowers λ. For each enzyme, fit an ordinary least-squares straight line to its amount against λ over the states reached at λ / λ_hi = 1, 0.85, 0.7, 0.55 and 0.4, and take its zero-growth response factor qᵢ to be the line's value at λ = 0 divided by its value at λ_hi. Report, to ten significant figures, the root-mean-square spread of q₁, …, q₁₂ about their weighted mean, with both the mean and the spread weighted by the fitted enzyme amounts at λ_hi.

In `<reasoning>`, the scalars I need you to state are: λ_hi and the total internal metabolite level Σᵢ ρᵢ in the rich medium; the fraction of that rich metabolite total still present at λ = 0.85 λ_hi; the largest qᵢ together with the enzyme it belongs to, and the weighted mean of the qᵢ; which of my enzymes are induced as growth slows (qᵢ > 1) and what property of their measured constants sets that; the spread and the range of the qᵢ when the transporter amount instead also maximises λ at every ν; how my rich metabolite total, my qᵢ and my retained metabolite fraction compare with the closed-form estimates published for this kind of growth-optimal allocation model, and what accounts for the difference in the qᵢ; and the final spread. Those are the derived quantities that determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied constants, which this problem does not need.

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

01_compute_enzyme_demand

Goal
----
Evaluate the proteome fraction that each enzyme of a linear pathway needs, per unit growth rate, to hold its metabolites at given levels in balanced exponential growth.

```python
def compute_enzyme_demand(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Return each pathway enzyme's proteome fraction per unit growth rate.

    Enzyme ``i`` (``i = 0, ..., n - 1`` in pathway order) consumes internal
    metabolite ``i``, held at level ``levels[i]``, and produces metabolite
    ``i + 1``; enzyme ``n - 1`` produces protein. Levels and enzyme amounts
    are mass fractions of total protein and fluxes are per unit total
    protein. In balanced growth at rate ``lam``, enzyme ``n - 1`` makes
    protein at the flux ``lam``, each metabolite ``k >= 1`` is produced by
    enzyme ``k - 1`` faster than enzyme ``k`` consumes it by exactly its
    dilution ``lam * levels[k]``, and enzyme ``i`` of amount ``phi_i``
    carries its flux at the irreversible Michaelis-Menten rate
    ``kcat[i] * phi_i * levels[i] / (km[i] + levels[i])``. Return
    ``phi_i / lam`` for every enzyme.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat : np.ndarray
        Positive catalytic constants per unit enzyme mass fraction, shape
        ``(n,)``.
    km : np.ndarray
        Positive Michaelis constants in the units of ``levels``, shape
        ``(n,)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n,)``: entry ``i`` is ``phi_i / lam``.

    Raises
    ------
    ValueError
        If any argument is not a non-empty one-dimensional array of finite
        positive numbers, or if the three arrays differ in length.
    """
    return demand
```

### Step 2

02_compute_metabolite_costs

Goal
----
Evaluate, for every metabolite of a linear pathway, the proteome cost per unit metabolite level at which its current level would be the cheapest way to carry the enzyme's flux.

```python
def compute_metabolite_costs(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Return the cost per unit level at which each metabolite level is optimal.

    Use the pathway, units and balanced-growth fluxes of
    ``compute_enzyme_demand``: enzyme ``i`` consumes metabolite ``i`` at level
    ``levels[i]`` and its flux per unit growth rate is fixed by the levels of
    the metabolites downstream of it. For metabolite ``i``, hold that flux
    fixed and let ``g_i(x)`` be the proteome fraction per unit growth rate
    that enzyme ``i`` would then need if its substrate were at level ``x``.
    Return, for every ``i``, the cost ``c_i`` (proteome fraction per unit
    metabolite level, per unit growth rate) for which ``x = levels[i]``
    minimises ``c_i * x + g_i(x)`` over ``x > 0``.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat : np.ndarray
        Positive catalytic constants per unit enzyme mass fraction, shape
        ``(n,)``.
    km : np.ndarray
        Positive Michaelis constants in the units of ``levels``, shape
        ``(n,)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n,)``: entry ``i`` is ``c_i``.

    Raises
    ------
    ValueError
        If any argument is not a non-empty one-dimensional array of finite
        positive numbers, or if the three arrays differ in length.
    """
    return costs
```

### Step 3

03_propagate_optimal_levels

Goal
----
Construct the metabolite levels of a linear pathway at which its total enzyme demand is stationary, indexing the one-parameter family either by the last metabolite level or by the total metabolite pool.

```python
def propagate_optimal_levels(
    anchor: float,
    kcat: "np.ndarray",
    km: "np.ndarray",
    coordinate: str = "last",
    return_tangent: bool = False,
) -> "np.ndarray":
    """Return one stationary metabolite profile from its last level or total pool.

    Use the pathway, units and balanced-growth fluxes of
    ``compute_enzyme_demand``, and let ``D(levels)`` be the sum over the ``n``
    enzymes of their proteome fractions per unit growth rate. Return the
    positive levels, with ``levels[n - 1] = last_level``, at which the
    partial derivatives of ``D`` with respect to all ``n`` levels are equal.
    These levels need the least enzyme among all positive levels with the
    same total. If ``coordinate == "last"``, ``anchor`` is the prescribed
    value of ``levels[n - 1]``. If ``coordinate == "total"``, ``anchor`` is
    the prescribed value of ``sum(levels)``. There is at most one positive
    profile in either case. Levels, and the requested total in ``"total"``
    mode, must be accurate to a relative ``1e-12``. If ``return_tangent`` is
    true, also return the tangent of every level with respect to the chosen
    anchor along the same stationary branch. Tangent entries must have
    relative error at most ``1e-10`` where nonzero and absolute error at most
    ``1e-12`` near zero.

    Parameters
    ----------
    anchor : float
        Positive last-metabolite level when ``coordinate == "last"``, or
        positive total internal metabolite level when
        ``coordinate == "total"``.
    kcat : np.ndarray
        Positive catalytic constants per unit enzyme mass fraction, shape
        ``(n,)`` with ``n >= 1``.
    km : np.ndarray
        Positive Michaelis constants in the units of the levels, shape
        ``(n,)``.
    coordinate : str
        Either ``"last"`` or ``"total"``.
    return_tangent : bool
        If false, return only the profile. If true, return the profile and its
        derivative with respect to ``anchor`` as two rows.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n,)`` with the levels in pathway order, or
        shape ``(2, n)`` with rows ``[levels, dlevels/danchor]`` when
        ``return_tangent`` is true.

    Raises
    ------
    ValueError
        If ``anchor`` is not a finite positive number (booleans are rejected),
        if ``coordinate`` is not ``"last"`` or ``"total"``, if
        ``return_tangent`` is not boolean, if ``kcat`` or ``km`` is not a
        non-empty one-dimensional array of finite positive numbers, if they
        differ in length, or if no positive profile with the requested
        coordinate satisfies the conditions.
    """
    return levels
```

### Step 4

04_close_optimal_transporter

Goal
----
Close a growth-optimal set of internal metabolite levels at the uptake step when the transporter amount is itself growth-maximising, returning the external nutrient level, the transporter amount and the growth rate.

```python
def close_optimal_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    demand_fn: "Callable[..., np.ndarray]",
    cost_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the nutrient level, transporter amount and growth rate of an optimal state.

    The pathway, units and balanced-growth fluxes are those of
    ``compute_enzyme_demand``. A transporter of amount ``phi_t`` converts an
    external nutrient, held at level ``nu`` and not diluted, into metabolite
    ``0`` at the rate ``kcat_t * phi_t * nu / (km_t + nu)``, so that in
    balanced growth it supplies metabolite ``0`` faster than enzyme ``0``
    consumes it by exactly that metabolite's dilution; the transporter and
    the ``n`` enzymes make up the whole proteome, ``phi_t + sum(phi) = 1``,
    and metabolites are outside this budget. ``levels`` are positive internal
    levels satisfying the conditions of ``propagate_optimal_levels``. Return
    the external level ``nu`` at which the growth rate maximised over all
    ``n + 1`` amounts is attained at exactly these internal levels, together
    with the transporter amount and the growth rate of that state.
    ``demand_fn(levels, kcat, km)`` and ``cost_fn(levels, kcat, km)`` follow
    the contracts of ``compute_enzyme_demand`` and
    ``compute_metabolite_costs``; use them for those quantities.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat, km : np.ndarray
        Positive enzyme constants, each of shape ``(n,)``.
    kcat_t : float
        Positive catalytic constant of the transporter.
    km_t : float
        Positive Michaelis constant of the transporter for the external
        nutrient.
    demand_fn, cost_fn : callable
        Functions with the contracts named above.

    Returns
    -------
    np.ndarray
        Float array ``[nu, phi_t, lam]``.

    Raises
    ------
    ValueError
        If ``levels``, ``kcat`` or ``km`` is not a non-empty one-dimensional
        array of finite positive numbers or they differ in length, if
        ``kcat_t`` or ``km_t`` is not a finite positive number (booleans are
        rejected), if a callable is missing or returns anything other than
        ``n`` finite positive numbers, or if no finite positive ``nu`` makes
        these levels optimal.
    """
    return state
```

### Step 5

05_close_regulated_transporter

Goal
----
Close a growth-optimal set of internal metabolite levels at the uptake step when the transporter amount is set by nutrient-dependent regulation, returning the external nutrient level, the transporter amount and the growth rate.

```python
def close_regulated_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    basal_fraction: float,
    max_fold: float,
    demand_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the nutrient level, transporter amount and growth rate under regulated uptake.

    The pathway, units and balanced-growth fluxes are those of
    ``compute_enzyme_demand``, and the transporter, its rate law, the supply
    balance of metabolite ``0`` and the proteome budget
    ``phi_t + sum(phi) = 1`` are those of ``close_optimal_transporter``. The
    transporter amount is not chosen for growth but follows the external
    nutrient level ``nu`` as
    ``phi_t(nu) = basal_fraction * (1 + (max_fold - 1) / (1 + max_fold * nu / km_t))``.
    Given the internal levels, return the positive external level ``nu`` at
    which balanced growth holds with this transporter amount, together with
    ``phi_t(nu)`` and the growth rate. There is at most one such ``nu``.
    ``demand_fn(levels, kcat, km)`` follows the contract of
    ``compute_enzyme_demand``; use it for the enzyme demand.

    Parameters
    ----------
    levels : np.ndarray
        Positive internal metabolite levels, shape ``(n,)`` with ``n >= 1``.
    kcat, km : np.ndarray
        Positive enzyme constants, each of shape ``(n,)``.
    kcat_t : float
        Positive catalytic constant of the transporter.
    km_t : float
        Positive Michaelis constant of the transporter for the external
        nutrient.
    basal_fraction : float
        Transporter amount at saturating nutrient, in ``(0, 1)``.
    max_fold : float
        Largest fold increase of the transporter amount, at least 1, with
        ``basal_fraction * max_fold < 1``.
    demand_fn : callable
        Function with the contract of ``compute_enzyme_demand``.

    Returns
    -------
    np.ndarray
        Float array ``[nu, phi_t, lam]``.

    Raises
    ------
    ValueError
        If ``levels``, ``kcat`` or ``km`` is not a non-empty one-dimensional
        array of finite positive numbers or they differ in length, if a
        scalar argument is outside its stated domain (booleans are rejected),
        if ``demand_fn`` is not callable or returns anything other than ``n``
        finite positive numbers, or if no positive ``nu`` sustains these
        levels.
    """
    return state
```

### Step 6

06_solve_family_member

Goal
----
Select, by bisection on the level of the last metabolite, the member of the one-parameter family of growth-optimal states whose external nutrient level or growth rate equals a target.

```python
def solve_family_member(
    target: float,
    component: int,
    kcat: "np.ndarray",
    km: "np.ndarray",
    propagate_fn: "Callable[..., np.ndarray]",
    close_fn: "Callable[[np.ndarray], np.ndarray]",
    bracket: tuple = (1e-12, 1.0),
    tolerance: float = 1e-13,
) -> "np.ndarray":
    """Return the levels of the family member at which a closed quantity equals a target.

    ``propagate_fn(x, kcat, km)`` follows the contract of
    ``propagate_optimal_levels`` and returns the internal levels of the
    family member whose last level is ``x``; ``close_fn(levels)`` returns
    ``[nu, phi_t, lam]`` for those levels, as ``close_optimal_transporter``
    or ``close_regulated_transporter`` does with every other argument fixed.
    Both ``nu`` and ``lam`` increase with ``x``. A point ``x`` at which either
    function raises ``ValueError`` lies beyond the upper end of the family
    and counts as above the target; any other point is above the target
    when ``close_fn(levels)[component] >= target``. Starting from
    ``[ln(bracket[0]), ln(bracket[1])]``, halve the interval in ``ln x``,
    keeping the half whose lower end is below and whose upper end is above
    the target, until its width is at most ``tolerance``, and return the
    levels at the midpoint of the final interval.

    Parameters
    ----------
    target : float
        Positive target value of the closed quantity.
    component : int
        ``0`` to match the external nutrient level, ``2`` to match the growth
        rate.
    kcat, km : np.ndarray
        Enzyme constants passed unchanged to ``propagate_fn``.
    propagate_fn, close_fn : callable
        Functions with the contracts named above.
    bracket : tuple
        ``(x_low, x_high)`` with ``0 < x_low < x_high``.
    tolerance : float
        Positive width, in ``ln x``, of the final interval.

    Returns
    -------
    np.ndarray
        Internal levels of the selected member, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``component`` is not ``0`` or ``2``, if ``target`` or
        ``tolerance`` is not a finite positive number (booleans are
        rejected), if the bracket is not two finite numbers with
        ``0 < x_low < x_high``, if a callable is missing, if ``x_low`` is not
        below the target or ``x_high`` is not above it, if ``close_fn``
        returns anything other than three finite numbers, or if the upper end
        of the final interval still lies beyond the family, which means the
        target is not reached.
    """
    return levels
```

### Step 7

07_summarize_growth_response

Goal
----
Turn the enzyme amounts of a series of balanced-growth states into zero-growth response factors from straight-line fits and summarise them by their mass-weighted mean and spread.

```python
def summarize_growth_response(growth_rates: "np.ndarray", amounts: "np.ndarray", growth_hi: float) -> "np.ndarray":
    """Return the enzymes' zero-growth response factors with their weighted mean and spread.

    ``amounts[s, i]`` is the proteome fraction of enzyme ``i`` in state ``s``,
    whose growth rate is ``growth_rates[s]``. For every enzyme, fit the
    ordinary least-squares straight line of its amount against the growth
    rate over all states and let ``h_i`` be the line's value at
    ``growth_hi``; the response factor ``q_i`` is the line's value at zero
    growth divided by ``h_i``. With weights ``w_i = h_i / sum(h)``, the
    weighted mean is ``qbar = sum(w * q)`` and the spread is the weighted
    root-mean-square deviation ``sqrt(sum(w * (q - qbar) ** 2))``.

    Parameters
    ----------
    growth_rates : np.ndarray
        Positive growth rates of the ``m`` states, shape ``(m,)``, with at
        least two distinct values.
    amounts : np.ndarray
        Enzyme proteome fractions, shape ``(m, n)`` with ``n >= 1``.
    growth_hi : float
        Positive growth rate at which the lines are normalised.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n + 2,)``:
        ``[q_0, ..., q_{n-1}, qbar, spread]``.

    Raises
    ------
    ValueError
        If ``growth_rates`` is not a one-dimensional array of finite positive
        numbers with at least two distinct values, if ``amounts`` is not a
        finite array of shape ``(m, n)`` with ``n >= 1``, if ``growth_hi`` is
        not a finite positive number (booleans are rejected), or if any
        fitted amount ``h_i`` is not positive.
    """
    return summary
```

### Step 8

08_estimate_response_spread

Goal
----
Compose every earlier step to compute the growth-optimal states of the pathway across a nutrient-limited growth series, verify the stationary family's local response, and return the mass-weighted spread of the enzymes' zero-growth response factors.

```python
def estimate_response_spread(
    kcat: tuple = (23.0, 9.3, 36.0, 5.9, 16.8, 12.6, 44.0, 7.3, 28.5, 10.7, 19.5, 8.1),
    km: tuple = (2.4e-4, 6.1e-4, 1.3e-4, 9.0e-4, 3.3e-4, 1.9e-3,
                 7.5e-5, 4.4e-4, 2.8e-4, 1.2e-3, 5.2e-4, 8.3e-4),
    kcat_t: float = 22.5,
    km_t: float = 1.2e-3,
    nutrient_hi: float = 0.6,
    max_fold: float = 3.0,
    growth_fractions: tuple = (1.0, 0.85, 0.7, 0.55, 0.4),
    regulated: bool = True,
) -> float:
    """Return the mass-weighted spread of the zero-growth response factors.

    The pathway, the transporter and the balanced-growth model are those of
    ``compute_enzyme_demand`` and ``close_optimal_transporter``. The rich
    reference state is the state that maximises the growth rate over all
    ``n + 1`` amounts at external nutrient level ``nutrient_hi``; call its
    growth rate ``lam_hi`` and its transporter amount ``phi_t_hi``. The
    regulated transporter follows the law of
    ``close_regulated_transporter`` with ``max_fold`` and with the
    ``basal_fraction`` for which that law gives ``phi_t_hi`` at
    ``nutrient_hi``. For each ``f`` in ``growth_fractions``, the state of the
    growth series is the one whose ``n`` enzyme amounts maximise the growth
    rate at the prevailing nutrient level and whose growth rate is
    ``f * lam_hi``, with the transporter regulated when ``regulated`` is
    true and also growth-maximising when it is false. Return the spread
    rich stationary profile and its analytic tangent are first recovered
    through ``propagate_optimal_levels`` as a consistency check on the family
    used by the closures. Return the spread returned by
    ``summarize_growth_response`` for the enzyme amounts of
    these states, their growth rates and ``growth_hi = lam_hi``. Metabolite
    levels are located with ``solve_family_member`` and its default bracket
    and tolerance. The defaults reproduce the problem statement.

    Parameters
    ----------
    kcat, km : tuple
        Positive enzyme constants in pathway order, equal lengths ``n >= 1``.
    kcat_t, km_t : float
        Positive transporter constants.
    nutrient_hi : float
        Positive external nutrient level of the rich reference state.
    max_fold : float
        Largest fold increase of the regulated transporter, at least 1.
    growth_fractions : tuple
        Values of ``lam / lam_hi`` of the series, each in ``(0, 1]``, with at
        least two distinct values.
    regulated : bool
        Whether the transporter is regulated in the growth series.

    Returns
    -------
    float
        The mass-weighted root-mean-square spread of the response factors.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain or any stage rejects
        its input, including when a requested growth rate is not reached.
    """
    return 0.0
```
