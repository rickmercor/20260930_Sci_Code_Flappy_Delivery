# Mathematics-Computational_Mechanics-24

## Background

Fibre-reinforced polymer laminates fail predominantly by delamination, the separation of neighbouring plies along the thin, resin-dominated layer that bonds them. Progressive delamination is simulated with cohesive zone models, in which an interface element obeys a traction-separation law that rises elastically to an interlaminar strength, then softens while dissipating the interlaminar fracture toughness. The elastic branch of that law is governed by a penalty stiffness: it is meant to be large enough that the interface does not open artificially before damage starts, yet the value chosen has a direct effect on the through-thickness stress state the surrounding elements see, and hence on the predicted onset of matrix cracking and on the shape of the traction profile ahead of the crack front.

Solid-element models resolve the cohesive zone directly but are expensive, because the element length must stay below the cohesive zone length. Shell-based, or structural, cohesive elements avoid that restriction by placing the interface between higher-order plate elements built on Kirchhoff-Love kinematics, which cuts the cost of a delamination simulation by orders of magnitude. The price is that Kirchhoff-Love plates carry no transverse stresses of their own: the out-of-plane normal and transverse shear stresses that drive the interface have to be reconstructed from equilibrium rather than read off the element formulation. When the penalty stiffness is inherited unchanged from solid-element practice, a scaled ratio of the laminate out-of-plane modulus to the laminate thickness, shell models are found to overpredict the compression ahead of the crack tip and to lose mesh convergence as the mesh is coarsened, which is precisely the regime in which shell models are worth using.

A more physical route ties the penalty stiffness to the layer that actually deforms. The interlaminar tractions pass through the resin-rich layers between plies, which are far more compliant than the plies themselves, so the stiffness of a cohesive element placed between two individual plies is simply the resin modulus divided by the resin-layer thickness. That argument, however, presumes one cohesive element per ply interface. Production models homogenise runs of identically oriented plies into a single ply block carried by one shell element, and a cohesive element placed between two such blocks must then stand in for all the resin-rich layers buried inside them. Since those layers sit at different heights and therefore carry different fractions of the peak interlaminar stress, closing the gap between the layer-wise and the equivalent-single-layer picture requires the through-thickness stress distribution of each arm, and that distribution is not the same in opening as in shearing, nor is it the same when further plies are bonded outside the block whose resin layers the element represents. Once the interface stiffnesses are fixed this way, the remainder of the constitutive model is standard practice: a quadratic interaction of the tractions with the interlaminar strengths marks damage onset, the Benzeggagh-Kenane relation interpolates between the pure-mode toughnesses at the local mode ratio, and a bilinear law dissipates that toughness up to complete decohesion.

## Problem

A structural cohesive element lies at the mid-plane between a homogenised upper block of 15 unidirectional IM7/8552 plies and a lower block of 11 plies, and exactly 12 additional plies must be divided between co-cured reinforcements outside those two represented blocks: if the upper reinforcement contains the integer $p$, the lower contains $12-p$, with each attached through its own separate, undamaged cohesive layer. Use the resin-rich-layer equivalent-single-layer penalty-stiffness construction from the cited source to evaluate every feasible allocation $p=0,1,\ldots,12$.

Each ply is $0.1875\ \mathrm{mm}$ thick, every represented ply interface contains a resin-rich layer of thickness $0.02286\ \mathrm{mm}$ with tensile modulus $4700\ \mathrm{N/mm^2}$ and shear modulus $1715\ \mathrm{N/mm^2}$, and the mid-plane strengths are $30\ \mathrm{MPa}$ in opening and $60\ \mathrm{MPa}$ in shearing. Use mode-I and mode-II toughnesses $0.212\ \mathrm{N/mm}$ and $0.774\ \mathrm{N/mm}$, Benzeggagh-Kenane exponent $2.1$, and a proportional loading path whose tangential separation is $0.75$ times its normal separation. Damage initiates by the quadratic nominal-traction interaction and evolves with a bilinear cohesive law written in work-conjugate effective separation.

Determine the largest effective separation at complete decohesion attainable by distributing the 12 reinforcement plies, and report that maximum in millimetres to at least eight significant digits. In the reasoning, identify the maximizing allocation and give the source-specific stress boundary conditions, layer sampling and outer-surface anchoring rule, the two summed stress ratios and penalty stiffnesses at the optimum, and only the neighbouring allocation values needed to establish that the integer maximum is unique; do not tabulate all 13 allocations.

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

01_arm_ratio_sum

Goal
----
Reduce one arm of the joint to the single dimensionless number it contributes to the interfacial compliance of the mid-plane cohesive element for a given fracture mode.

```python
def arm_ratio_sum(mode: str, n_cohesive: int, n_total: int, t_ply: float) -> float:
    r"""Evaluate the source-defined dimensionless arm contribution.

    Parameters
    ----------
    mode : str
        Reconstruction mode, either ``"opening"`` or ``"shear"``.
    n_cohesive : int
        Number of plies in the arm block represented at the mid-plane
        interface (n_cohesive >= 1).
    n_total : int
        Total number of plies in the complete arm (n_total >= n_cohesive).
    t_ply : float
        Thickness of a single ply (t_ply > 0).

    Returns
    -------
    ratio_sum : float
        Dimensionless arm contribution prescribed by the source construction,
        as a native Python float.

    Raises ValueError if ``mode`` is not ``"opening"`` or ``"shear"``, if
    ``n_cohesive`` or ``n_total`` is not an integer, if ``n_cohesive`` < 1, if
    ``n_total`` < ``n_cohesive``, or if ``t_ply`` is not a finite real number
    greater than zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 2

02_resin_rich_penalty_stiffness

Goal
----
Convert the summed interface stress ratios of the two arms into the penalty stiffness of the structural cohesive element between them.

```python
def resin_rich_penalty_stiffness(ratio_sum_total: float, h_rr: float, modulus: float) -> float:
    """Evaluate the source-defined equivalent penalty stiffness.

    Parameters
    ----------
    ratio_sum_total : float
        Combined dimensionless arm result from step 01
        (ratio_sum_total > 0).
    h_rr : float
        Thickness of one resin-rich layer (h_rr > 0).
    modulus : float
        Mode-appropriate elastic modulus of the resin-rich layer
        (modulus > 0).

    Returns
    -------
    penalty_stiffness : float
        Equivalent penalty stiffness prescribed by the source construction,
        as a native Python float.

    Raises ValueError if any argument is not a finite real number greater than
    zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 3

03_mixed_mode_onset_separation

Goal
----
Locate damage onset along a proportional mixed-mode separation path through the quadratic interaction of the two interfacial strengths.

```python
def mixed_mode_onset_separation(k_n: float, k_s: float, tau_ic: float,
                                tau_iic: float, disp_ratio: float) -> float:
    """Compute the effective separation at mixed-mode damage onset.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    tau_ic : float
        Interlaminar strength in opening (tau_ic > 0).
    tau_iic : float
        Interlaminar strength in shearing (tau_iic > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).

    Returns
    -------
    delta_onset : float
        Effective separation at damage onset, as a native Python float.

    Raises ValueError if ``k_n``, ``k_s``, ``tau_ic`` or ``tau_iic`` is not a
    finite real number greater than zero, or if ``disp_ratio`` is not a finite
    real number greater than or equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 4

04_mixed_mode_energy_ratio

Goal
----
Express the local mode ratio of the interface as the shear share of the interfacial energy along a proportional separation path.

```python
def mixed_mode_energy_ratio(k_n: float, k_s: float, disp_ratio: float) -> float:
    """Compute the shear share of the interfacial energy.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).

    Returns
    -------
    b_ratio : float
        Local mode ratio, the shear fraction of the stored interfacial
        energy, in [0, 1) as a native Python float.

    Raises ValueError if ``k_n`` or ``k_s`` is not a finite real number greater
    than zero, or if ``disp_ratio`` is not a finite real number greater than or
    equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 5

05_benzeggagh_kenane_toughness

Goal
----
Interpolate the two pure-mode interlaminar toughnesses to the mixed-mode toughness at the local mode ratio of the interface.

```python
def benzeggagh_kenane_toughness(g_ic: float, g_iic: float, eta: float,
                                b_ratio: float) -> float:
    """Interpolate the mixed-mode interlaminar fracture toughness.

    Parameters
    ----------
    g_ic : float
        Interlaminar fracture toughness in opening (g_ic > 0).
    g_iic : float
        Interlaminar fracture toughness in shearing (g_iic > 0).
    eta : float
        Benzeggagh-Kenane material exponent (eta > 0).
    b_ratio : float
        Local mode ratio, the shear fraction of the interfacial energy, in
        [0, 1].

    Returns
    -------
    g_c : float
        Mixed-mode interlaminar fracture toughness, as a native Python float.

    Raises ValueError if ``g_ic``, ``g_iic`` or ``eta`` is not a finite real
    number greater than zero, or if ``b_ratio`` is not a finite real number in
    [0, 1].

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 6

06_bilinear_failure_separation

Goal
----
Close the bilinear traction-separation law by locating the effective separation at which the cohesive element loses all load-carrying capacity.

```python
def bilinear_failure_separation(k_n: float, k_s: float, disp_ratio: float,
                                delta_onset: float, g_c: float) -> float:
    """Compute the effective separation at complete decohesion.

    Parameters
    ----------
    k_n : float
        Normal penalty stiffness of the cohesive element (k_n > 0).
    k_s : float
        Shear penalty stiffness of the cohesive element (k_s > 0).
    disp_ratio : float
        Ratio of the tangential to the normal separation along the
        proportional path (disp_ratio >= 0).
    delta_onset : float
        Effective separation at damage onset (delta_onset > 0).
    g_c : float
        Mixed-mode interlaminar fracture toughness (g_c > 0).

    Returns
    -------
    delta_failure : float
        Effective separation at which the traction returns to zero, as a
        native Python float.

    Raises ValueError if ``k_n``, ``k_s``, ``delta_onset`` or ``g_c`` is not a
    finite real number greater than zero, if ``disp_ratio`` is not a finite real
    number greater than or equal to zero, or if the failure separation would
    fall below ``delta_onset``.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0
```

### Step 7

07_optimal_reinforcement_failure_separation

Goal
----
Search all integer allocations of a fixed reinforcement-ply budget between the two outer faces of an asymmetric joint and return the largest complete-decohesion separation.

```python
def optimal_reinforcement_failure_separation(
        n_cohesive: tuple = (15, 11), n_reinforcement: int = 12,
        t_ply: float = 0.1875, h_rr: float = 0.02286,
        e_rr: float = 4700.0, g_rr: float = 1715.0,
        tau_ic: float = 30.0, tau_iic: float = 60.0,
        g_ic: float = 0.212, g_iic: float = 0.774,
        eta: float = 2.1, disp_ratio: float = 0.75) -> float:
    """Return the best failure separation for a fixed reinforcement budget.

    Parameters
    ----------
    n_cohesive : tuple
        Two integers giving the represented upper and lower ply counts.
    n_reinforcement : int
        Total number of reinforcement plies to distribute between the upper
        and lower outer faces (n_reinforcement >= 0).
    t_ply, h_rr : float
        Ply thickness and resin-rich-layer thickness, both greater than zero.
    e_rr, g_rr : float
        Resin-rich-layer tensile and shear moduli, both greater than zero.
    tau_ic, tau_iic : float
        Opening and shearing strengths, both greater than zero.
    g_ic, g_iic : float
        Opening and shearing fracture toughnesses, both greater than zero.
    eta : float
        Benzeggagh-Kenane exponent (eta > 0).
    disp_ratio : float
        Tangential-to-normal separation ratio (disp_ratio >= 0).

    Returns
    -------
    delta_failure_max : float
        Maximum effective separation at complete decohesion over every integer
        allocation, as a native Python float.

    Raises ValueError if ``n_cohesive`` is not a sequence of exactly two
    integers, if ``n_reinforcement`` is not an integer greater than or equal to
    zero, if a scalar parameter other than ``disp_ratio`` is not a finite real
    number greater than zero, if ``disp_ratio`` is not a finite real number
    greater than or equal to zero, or if an earlier step rejects its inputs.

    Notes
    -----
    This is the final orchestrating step. For each integer upper allocation
    from zero through ``n_reinforcement``, construct the complete-arm ply
    counts and call the public functions of steps 01-06 in order. Return the
    largest failure separation without rounding. Include every import inside
    the function body.
    """
    return 0.0
```
