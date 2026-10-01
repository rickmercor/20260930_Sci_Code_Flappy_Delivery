# Physics-Particle_Physics-25

## Background

Hadronic form factors are analytic functions of the invariant mass squared of the hadron pair, with a unitarity cut starting at the lowest open threshold and further cuts opening at each heavier channel. Because the data that constrain them are confined to limited kinematic windows, they are usually represented by a series in a conformal variable that carries the cut plane into the unit disc, so that the expansion converges wherever the parametrisation is meant to be used.

Resonances are not singularities of the first sheet, however: they are poles on the unphysical sheets reached by continuing through the cuts, and those sheets also carry the left-hand cut demanded by crossing. A conformal variable that reaches them turns resonance parameters into quantities that can be read directly off a fit performed on the physical axis.

## Problem

I am parametrising the isoscalar S-wave pion form factor F(s) of a two-channel problem whose elastic channel is two pions with M_π = 0.13957 GeV and whose inelastic channel is two kaons with M_K = 0.493677 GeV, so that the thresholds sit at s₊ = 4M_π² and s_in = 4M_K², and the left-hand cut carried by the sheet reached across the elastic cut opens at s = 0. I want a series in a conformal variable ψ that opens both two-particle cuts at once and leaves no cut inside the unit disc. Build ψ from the variable ϕ that inverts s = [s₊(1 + ϕ²)² − 4 s_in ϕ²]/(1 − ϕ²)², whose four branches ϕ, −ϕ, 1/ϕ and −1/ϕ are the four Riemann sheets opened by the two thresholds: the physical sheet (11) is the branch with 0 < ϕ < 1 for real s < s₊, sheet (21) is reached from it through the cut between the two thresholds, and sheet (22) through the cut above s_in. Then compose ϕ with the conformal map of the unit disc, cut along the real segment joining ϕ = −1 to the image of s = 0 on sheet (21), onto the unit disc that is real on the real axis, sends the image of the expansion point s₀ = −0.30 GeV² on the physical sheet to the origin and leaves ϕ = 1 fixed, continued to |ϕ| > 1.

Write F(s) = [Π_r 1/((ψ − ψ_r)(ψ − ψ_r*))] Σ_k a_k ψ^k with real coefficients a_k, the product running over the images ψ_r of my four poles: √s = 0.462 − 0.271i GeV and √s = 0.9935 − 0.0285i GeV on sheet (21), and √s = 0.9705 − 0.0655i GeV and √s = 0.700 − 0.350i GeV on sheet (22). My measured values of F are 0.45 at s = −2.00 GeV² and 0.80 at s = −0.40 GeV², and, on the physical axis above the two-pion threshold, 1.70 + 0.55i at s = 0.20 GeV², 2.30 + 1.30i at s = 0.50 GeV² and 2.90 + 3.60i at s = 0.80 GeV². Require in addition that F(s) fall off as 1/s at large |s|, and truncate the series at the order for which these conditions determine the coefficients uniquely. Report the modulus, in GeV², of the residue of F at the first of those poles, the one at √s = 0.462 − 0.271i GeV on sheet (21), to ten significant figures. For the numerical asymptotic check, use probe = -1e10 GeV² and ratio = 10000, so the second probe is -1e14 GeV². For the contour evaluation of the residue, use radius = 0.01 GeV² and nodes = 128 equally spaced points around the first pole (target = 0).

In the reasoning, state the analytic expression for the normalized slit-disc map and its continuation outside the unit disc, followed by these derived quantities: the two real reference points of the slit-disc map, namely the image of s = 0 on sheet (21) and the image of s₀ on the physical sheet; the exponent p and coefficient C in (1 − ψ(s))^p ∼ C/|s| as s → −∞ on the physical sheet, and the number of conditions the 1/s fall-off therefore imposes; the images ψ of my measurement at s = 0.80 GeV² and of my pole at √s = 0.700 − 0.350i GeV; the values of F at s = 0, at the two-pion threshold, and at s = 0.35, 0.65 and 0.90 GeV²; the limit of s F(s) at large |s|; and the final residue modulus. These map expressions and derived quantities determine the final number; the input data and full coefficient list need not be repeated.

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

01_map_two_threshold_variable

Goal
----
Map the invariant mass squared onto the uniformising variable that opens the elastic and the inelastic two-particle cut together, on a selected Riemann sheet.

```python
def map_two_threshold_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    sheet: str,
) -> complex:
    """Return the two-threshold uniformising variable on one Riemann sheet.

    The variable ``phi`` inverts

        ``s = (s_plus * (1 + phi**2)**2 - 4 * s_inelastic * phi**2)
              / (1 - phi**2)**2``

    whose four branches are the four Riemann sheets opened by the two
    thresholds. ``sheet`` selects one of them:

    * ``"11"`` -- the physical branch, fixed by ``0 < phi < 1`` for real
      ``s < s_plus``;
    * ``"21"`` -- reached from ``"11"`` by continuing through the cut
      between the two thresholds, where the elastic-channel square root
      changes sign;
    * ``"22"`` -- reached from ``"11"`` by continuing through the cut above
      the inelastic threshold, where both channel square roots change sign;
    * ``"12"`` -- the remaining branch, on which only the inelastic-channel
      square root changes sign.

    A real ``s`` lying on a cut is taken as the limit from the upper half of
    the complex ``s`` plane, which is the boundary value carrying physical
    amplitudes. The two branches inside the unit disc both take the value
    zero at ``s = s_plus``, where the other two are infinite.

    Parameters
    ----------
    s : complex
        Invariant mass squared; real values are accepted.
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    sheet : str
        One of ``"11"``, ``"21"``, ``"22"`` and ``"12"``.

    Returns
    -------
    complex
        The variable ``phi`` on the requested sheet.

    Raises
    ------
    ValueError
        If ``s`` is not a finite number, if ``s_plus`` or ``s_inelastic`` is
        not a finite real number with ``0 < s_plus < s_inelastic``, if
        ``sheet`` is not one of the four labels, or if the requested branch
        is not finite at ``s``.
    """
    return 0j
```

### Step 2

02_map_left_hand_cut

Goal
----
Return the image of point under the conformal map from the unit disc slit along [-1, cut_branch_point] onto the unit disc, normalized to preserve the real axis within its domain, send origin_point to zero, and fix +1.

```python
def map_left_hand_cut(
    point: complex,
    cut_branch_point: float,
    origin_point: float,
) -> complex:
    """Return the image of a point under the slit-disc conformal map.

    Let ``D`` be the open unit disc cut along the real segment that joins
    ``-1`` to ``cut_branch_point``. Return the image of ``point`` under the
    conformal map of ``D`` onto the unit disc that

    * takes real values on the real axis,
    * sends ``origin_point`` to ``0``,
    * leaves ``+1`` fixed.

    These three conditions fix the map uniquely. It carries the slit onto an
    arc of the unit circle, with ``cut_branch_point`` going to ``-1``. Only
    arguments of modulus at most one are accepted.

    Parameters
    ----------
    point : complex
        Point of the closed unit disc, ``abs(point) <= 1``.
    cut_branch_point : float
        Finite real number with ``-1 < cut_branch_point < origin_point``,
        where the slit opens.
    origin_point : float
        Finite real number with ``cut_branch_point < origin_point < 1``,
        mapped to the origin.

    Returns
    -------
    complex
        The image of ``point``.

    Raises
    ------
    ValueError
        If ``point`` is not a finite number of modulus at most one, if
        ``cut_branch_point`` and ``origin_point`` are not finite real
        numbers obeying ``-1 < cut_branch_point < origin_point < 1``, or if
        the image is not finite.
    """
    return 0j
```

### Step 3

03_map_four_sheet_variable

Goal
----
Compose the two-threshold variable with the slit-disc map to obtain the expansion variable of the four-sheet parametrisation.

```python
def map_four_sheet_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    s_lhc: float,
    s_origin: float,
    sheet: str,
    threshold_fn: "Callable[..., complex]",
    lhc_fn: "Callable[..., complex]",
) -> complex:
    """Return the four-sheet expansion variable at ``s`` on one Riemann sheet.

    ``threshold_fn(s, s_plus, s_inelastic, sheet)`` follows the contract of
    ``map_two_threshold_variable`` and ``lhc_fn(point, cut_branch_point,
    origin_point)`` that of ``map_left_hand_cut``.

    The slit is fixed by the two reference points
    ``threshold_fn(s_lhc, s_plus, s_inelastic, "21")``, the image of the
    left-hand branch point on the sheet reached across the elastic cut, and
    ``threshold_fn(s_origin, s_plus, s_inelastic, "11")``, the image of the
    expansion point on the physical sheet; both are real, and their real
    parts are the arguments handed to ``lhc_fn``.

    ``lhc_fn`` accepts only arguments of modulus at most one. Where the
    two-threshold variable leaves the closed unit disc, return instead the
    value of the analytic continuation of the composed map through the unit
    circle, which maps the exterior of the disc onto itself.

    Parameters
    ----------
    s : complex
        Invariant mass squared; real values are accepted.
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    s_lhc : float
        Finite real point where the left-hand cut opens.
    s_origin : float
        Finite real expansion point, mapped to the origin on the physical
        sheet.
    sheet : str
        One of ``"11"``, ``"21"``, ``"22"`` and ``"12"``.
    threshold_fn : callable
        Two-threshold uniformising map.
    lhc_fn : callable
        Slit-disc conformal map.

    Returns
    -------
    complex
        The expansion variable on the requested sheet.

    Raises
    ------
    ValueError
        If ``threshold_fn`` or ``lhc_fn`` is not callable, if either
        reference point is not a finite real number, or if any stage of the
        composition rejects its argument or fails to return a finite number.
    """
    return 0j
```

### Step 4

04_solve_asymptotic_behaviour

Goal
----
Measure how fast the expansion variable approaches its value at infinite invariant mass, and with what coefficient.

```python
def solve_asymptotic_behaviour(
    variable_fn: "Callable[[float], complex]",
    probe: float = -1.0e10,
    ratio: float = 1.0e4,
) -> "np.ndarray":
    """Return the exponent and coefficient of the variable's approach to unity.

    ``variable_fn(s)`` returns the physical-sheet expansion variable at a
    real spacelike ``s``, where it is real and below one, and tends to ``1``
    as ``s`` tends to ``-inf``. There is a positive exponent ``p`` and a
    finite positive ``c`` with

        ``(1 - variable_fn(s)) ** p -> c / abs(s)``   as ``s -> -inf``.

    Estimate ``p`` from the two probes ``s = probe`` and
    ``s = probe * ratio``, and evaluate ``c`` at the farther of the two.

    Parameters
    ----------
    variable_fn : callable
        Physical-sheet expansion variable as a function of real ``s``.
    probe : float
        Finite negative probe point.
    ratio : float
        Finite factor larger than one giving the second, farther probe.

    Returns
    -------
    np.ndarray
        Array ``[p, c]`` of two floats.

    Raises
    ------
    ValueError
        If ``variable_fn`` is not callable, if ``probe`` is not a finite
        negative number, if ``ratio`` is not a finite number greater than
        one, if either probe value is not a finite real number below one, or
        if the estimated exponent is not finite and positive.
    """
    return behaviour
```

### Step 5

05_evaluate_pole_factor

Goal
----
Evaluate the product of pole factors that carries the resonances of the parametrisation in the expansion variable.

```python
def evaluate_pole_factor(
    points: "np.ndarray",
    pole_points: "np.ndarray",
) -> "np.ndarray":
    """Return the resonance pole product at each requested point.

    With ``w_r`` the images of the resonance poles in the expansion
    variable, the product is

        ``P(w) = prod_r 1 / ((w - w_r) * (w - conj(w_r)))``

    so that every resonance contributes its pole and the mirror pole
    required for the parametrisation to be real where the variable is real.

    Parameters
    ----------
    points : np.ndarray
        One-dimensional array of finite complex points, at least one.
    pole_points : np.ndarray
        One-dimensional array of finite complex pole images, at least one,
        none of them real.

    Returns
    -------
    np.ndarray
        Complex array with the same length as ``points``.

    Raises
    ------
    ValueError
        If either argument is not a one-dimensional non-empty array of
        finite numbers, if any pole image is real, or if any point
        coincides with a pole or its mirror.
    """
    return product
```

### Step 6

06_solve_series_coefficients

Goal
----
Fix the real coefficients of the truncated series from measured values of the amplitude and a prescribed high-energy fall-off.

```python
def solve_series_coefficients(
    real_points: "np.ndarray",
    real_values: "np.ndarray",
    complex_points: "np.ndarray",
    complex_values: "np.ndarray",
    pole_points: "np.ndarray",
    vanishing_order: int,
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the real series coefficients fixed by the measurements and fall-off.

    The parametrisation is ``F(w) = factor_fn(w, pole_points) * g(w)`` with
    ``g(w) = sum_k a_k w**k`` and every ``a_k`` real. Impose

    * ``F`` equal to ``real_values[j]`` at ``real_points[j]``, where both the
      point and the value are real, so each pair is one real condition;
    * ``F`` equal to ``complex_values[j]`` at ``complex_points[j]``, where
      the point and the value are complex, so each pair is two real
      conditions;
    * ``g`` vanishing to order ``vanishing_order`` at ``w = 1``, that is
      ``g`` and its first ``vanishing_order - 1`` derivatives vanish there.

    Truncate ``g`` at the degree for which these conditions determine the
    coefficients uniquely, and return them in order of increasing power.

    Parameters
    ----------
    real_points : np.ndarray
        One-dimensional array of real points, possibly empty.
    real_values : np.ndarray
        Real values of ``F`` at ``real_points``, same length.
    complex_points : np.ndarray
        One-dimensional array of complex points, possibly empty, none of
        them real.
    complex_values : np.ndarray
        Values of ``F`` at ``complex_points``, same length.
    pole_points : np.ndarray
        One-dimensional array of pole images passed on to ``factor_fn``.
    vanishing_order : int
        Number of vanishing conditions imposed at ``w = 1``, at least one.
    factor_fn : callable
        Pole product following the contract of ``evaluate_pole_factor``.

    Returns
    -------
    np.ndarray
        Real coefficients ``[a_0, ..., a_n]``.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional with matching lengths and
        finite entries, if a real point or value is not real, if a complex
        point is real, if ``vanishing_order`` is not a positive integer, if
        ``factor_fn`` is not callable or returns the wrong shape, if no
        condition is supplied at all, or if the resulting linear system is
        singular.
    """
    return coefficients
```

### Step 7

07_evaluate_series_form_factor

Goal
----
Evaluate the parametrised amplitude from its series coefficients and its pole product.

```python
def evaluate_series_form_factor(
    points: "np.ndarray",
    coefficients: "np.ndarray",
    pole_points: "np.ndarray",
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the parametrised amplitude at each point of the expansion variable.

    The parametrisation is

        ``F(w) = factor_fn(w, pole_points) * sum_k coefficients[k] * w**k``

    evaluated at every entry of ``points``. Any point of the expansion
    variable is accepted, inside or outside the unit disc.

    Parameters
    ----------
    points : np.ndarray
        One-dimensional non-empty array of finite complex points.
    coefficients : np.ndarray
        One-dimensional non-empty array of finite real coefficients, in
        order of increasing power.
    pole_points : np.ndarray
        One-dimensional array of pole images passed on to ``factor_fn``.
    factor_fn : callable
        Pole product following the contract of ``evaluate_pole_factor``.

    Returns
    -------
    np.ndarray
        Complex array with the same length as ``points``.

    Raises
    ------
    ValueError
        If ``points`` or ``coefficients`` is not a one-dimensional
        non-empty array of finite numbers, if ``coefficients`` is not real,
        if ``factor_fn`` is not callable or returns the wrong shape, or if
        the result is not finite.
    """
    return amplitude
```

### Step 8

08_extract_pole_residue

Goal
----
Extract the residue of a function at a simple pole from its values on a small circle about that pole.

```python
def extract_pole_residue(
    pole_position: complex,
    function_fn: "Callable[[np.ndarray], np.ndarray]",
    radius: float,
    nodes: int,
) -> complex:
    """Return the residue of a function at a simple pole.

    ``function_fn`` maps a one-dimensional complex array of arguments to the
    array of function values. On the closed disc of radius ``radius`` about
    ``pole_position`` the function is analytic apart from a simple pole at
    the centre. Sample it at ``nodes`` equally spaced points on that circle,
    the first of them at ``pole_position + radius``, and return the residue.

    Parameters
    ----------
    pole_position : complex
        Location of the simple pole.
    function_fn : callable
        Function of a complex array returning an array of the same length.
    radius : float
        Finite positive radius of the sampling circle.
    nodes : int
        Number of sample points, at least two.

    Returns
    -------
    complex
        The residue at ``pole_position``.

    Raises
    ------
    ValueError
        If ``pole_position`` is not a finite number, if ``function_fn`` is
        not callable or returns the wrong shape, if ``radius`` is not a
        finite positive number, if ``nodes`` is not an integer of at least
        two, or if the sampled values are not finite.
    """
    return 0j
```

### Step 9

09_estimate_pole_residue

Goal
----
Compose every earlier step to obtain the modulus of the residue of the parametrised amplitude at a chosen resonance pole.

```python
def estimate_pole_residue(
    s_plus: float = 4.0 * 0.13957 ** 2,
    s_inelastic: float = 4.0 * 0.493677 ** 2,
    s_lhc: float = 0.0,
    s_origin: float = -0.30,
    pole_roots: tuple = (0.462 - 0.271j, 0.9935 - 0.0285j, 0.9705 - 0.0655j, 0.700 - 0.350j),
    pole_sheets: tuple = ("21", "21", "22", "22"),
    real_nodes: tuple = (-2.00, -0.40),
    real_values: tuple = (0.45, 0.80),
    timelike_nodes: tuple = (0.20, 0.50, 0.80),
    timelike_values: tuple = (1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j),
    target: int = 0,
    radius: float = 0.01,
    nodes: int = 128,
) -> float:
    """Return the modulus of the residue of the parametrised amplitude at one pole.

    The four-sheet expansion variable is built from the thresholds
    ``s_plus`` and ``s_inelastic``, from the left-hand branch point
    ``s_lhc`` on the sheet reached across the elastic cut, and from the
    expansion point ``s_origin``, which is sent to the origin on the
    physical sheet. Each entry of ``pole_roots`` is the square root of a
    pole position, carried by the sheet named in the matching entry of
    ``pole_sheets``, and these poles are implemented as explicit conjugate
    pairs of factors multiplying a truncated real power series.

    The coefficients are fixed by three kinds of condition: the values
    ``real_values`` taken at the real points ``real_nodes`` below the
    elastic threshold, the values ``timelike_values`` taken at the physical
    boundary values above it at ``timelike_nodes``, and the requirement
    that the parametrised amplitude fall off as one over the invariant mass
    squared at large spacelike argument. The series is truncated at the
    degree for which those conditions determine the coefficients uniquely.

    Return the modulus of the residue, in the units of ``s``, of the
    resulting amplitude continued to the sheet of the pole selected by
    ``target``, sampled on a circle of radius ``radius`` with ``nodes``
    points. The defaults reproduce the problem statement.

    Parameters
    ----------
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    s_lhc : float
        Finite real point where the left-hand cut opens.
    s_origin : float
        Finite real expansion point below ``s_plus``.
    pole_roots : tuple
        Non-empty tuple of finite complex square roots of pole positions.
    pole_sheets : tuple
        Sheet labels, same length as ``pole_roots``.
    real_nodes : tuple
        Real points below ``s_plus`` carrying real measurements.
    real_values : tuple
        Real measured values, same length as ``real_nodes``.
    timelike_nodes : tuple
        Real points above ``s_plus`` carrying complex measurements.
    timelike_values : tuple
        Measured values, same length as ``timelike_nodes``.
    target : int
        Index into ``pole_roots`` of the pole whose residue is returned.
    radius : float
        Finite positive radius of the sampling circle.
    nodes : int
        Number of sample points, at least two.

    Returns
    -------
    float
        The modulus of the residue at the selected pole.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain, if the tuple lengths
        do not match, if ``target`` is not a valid index, or if any stage
        rejects its input.
    """
    return 0.0
```
