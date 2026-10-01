# Mathematics-Computational_Mechanics-52

## Background

Bond-based peridynamics represents a solid as material points that interact through bonds inside a finite horizon, which lets fracture emerge from bond failure without ever differentiating the displacement field. Realistic simulations refine the discretisation where a crack is expected, so the horizon varies in space. In that setting the classical single-horizon formulation makes the interaction between two points one-sided: one point reaches its neighbour while the neighbour does not reach back. The resulting asymmetry violates the balance of linear momentum and shows up as ghost forces and spurious wave reflections at the refinement interface. Deriving the formulation variationally instead yields a dual-horizon treatment that restores the balance exactly and, in turn, supports asynchronous time integration with different time steps in different regions. On a small fixed family of non-uniformly discretised bars the whole construction can be audited directly against the balance laws.

## Problem

Bond-based peridynamics models fracture without spatial derivatives by letting each material point interact with every neighbour inside its horizon. When the horizon varies in space and the discretisation is non-uniform, the classical single-horizon formulation makes those interactions asymmetric, which violates the balance of linear momentum and produces ghost forces and spurious wave reflections. A recent variational framework derives the dual-horizon formulation directly from the Lagrange-d'Alembert principle and, on that basis, an asynchronous time-integration strategy. Your job is to implement the source's dual-horizon formulation exactly as specified and audit it on the fixed family of non-uniformly discretised bars below. The load-bearing algorithmic choices - the dual-horizon assembly of the internal force, the horizon that enters the micro-modulus of a bond, the value of the one-dimensional dual-horizon micro-modulus, the weighting used in the local damage, and the critical-time-step estimate - are the source's; recover them from the paper. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

Each configuration is a small one-dimensional bar of exactly eight material points, a coarse region followed by a refined region, described by (nC, nF, m, a, dC). The coarse spacing is dC and the fine spacing is dF = dC/2. Build the reference coordinates X by starting at 0.0, taking nC-1 successive steps of dC, and then nF successive steps of dF, so N = nC + nF = 8. Build the per-point spacing array sp by setting the first nC entries to dC and the remaining nF entries to dF, and then overwriting the entry at index nC-1 (the last coarse point, which sits on the interface) with the average (dC + dF)/2. The point volumes are V = sp * A and the spatially varying horizons are delta = m * sp. The prescribed displacement field is u[i] = 0.010 * X[i] + 0.0015 * p[i], where p[i] = (((i+1)(i+2) + a(i+3)) mod 17) - 8 for i = 0..7. The three configurations use (nC, nF, m, a, dC) = (4, 4, 2.515, 3, 0.10), (3, 5, 2.515, 5, 0.08) and (5, 3, 3.015, 7, 0.10). Throughout, Young's modulus is E = 1.0, the cross-sectional area is A = 1.0, and the mass density is rho = 1.0. The declared base critical stretch is sc = 0.16125, and the audit multiplies it by sc_scale.

For each configuration, evaluate the source's dual-horizon model at the declared base critical stretch multiplied by sc_scale. Determine the family of every point under that point's own horizon (with spatially varying horizons this membership is not symmetric), the stretch of every ordered pair of points, which family bonds survive at that critical stretch, the local damage at each point under the source's weighting, the internal force density at each point under the source's dual-horizon assembly, and the critical time step of the whole point cloud, taken as the smallest per-point value. Summarise each configuration by four numbers - the internal virial, the critical time step, the mean local damage over the points, and the number of intact family bonds counted as ordered pairs - so that the full audit is a float64 array of shape (3, 4) whose rows follow the configuration order. The internal virial of a configuration is the sum over points of the internal force density times the reference coordinate times the point volume.

Evaluate the audit with sc_scale = 1.0. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. As the final answer, report the sum over the three configurations of the internal virial (the first column) to six significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 12.3456, -0.802, 45.9). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Budget your response so that the <final_answer> tag is always reached and closed. If you are running long, stop the intermediate work and emit the answer.
Keep <reasoning> under roughly 600 words. As numerical results, report only the three per-configuration virials and the running total; do NOT enumerate per-bond stretches, per-point forces, family tables, or any matrix.

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

micro_modulus

Goal
----
Return the source's one-dimensional bond micro-modulus c for each given horizon, for Young's modulus E and cross-sectional area A (the paper's Eq. 25, one-dimensional case). Note the paper states that the micro-modulus of the dual-horizon formulation differs by a constant factor from the classical single-horizon one; use the source's dual-horizon value. Recover the exact expression from the paper.

```python
import numpy as np


def micro_modulus(delta, E, A):
    """delta: horizon value(s) > 0; E: Young's modulus; A: cross-sectional area.
    Returns a float64 array shaped like delta with the source's 1D dual-horizon
    bond micro-modulus (paper Eq. 25). Raises ValueError on a non-positive or
    nonfinite horizon."""
    return np.zeros_like(np.asarray(delta, dtype=np.float64))
```

### Step 2

families

Goal
----
Build the family (neighbourhood) indicator of the point cloud (the paper's Eq. 1): entry (i, j) is 1.0 when point j belongs to the family of point i under that point's own horizon, and 0.0 otherwise. A point is never its own family member. With spatially varying horizons this relation is NOT symmetric, and the source relies on that asymmetry.

```python
import numpy as np


def families(X, delta):
    """X: (N,) reference coordinates; delta: (N,) per-point horizons.
    Returns (N, N) float64: 1.0 where point j lies in the family of point i
    under point i's own horizon, else 0.0; the diagonal is 0.0 (paper Eq. 1).
    The result is in general not symmetric."""
    n = np.asarray(X).size
    return np.zeros((n, n))
```

### Step 3

bond_stretch

Goal
----
Compute the bond stretch for every ordered pair of points from the reference coordinates and the displacement field (the paper's Eq. 22). The deformed position of a point is its reference coordinate plus its displacement. The diagonal is zero.

```python
import numpy as np


def bond_stretch(X, u):
    """X: (N,) reference coordinates; u: (N,) displacements. Returns (N, N)
    float64 with the bond stretch of every ordered pair (paper Eq. 22), zero on
    the diagonal."""
    n = np.asarray(X).size
    return np.zeros((n, n))
```

### Step 4

damage_state

Goal
----
Return the source's bond damage indicator for the given stretch state, family indicator and critical stretch (the paper's Eq. 27): a bond that belongs to a family is intact while its stretch stays below the critical stretch, and broken otherwise. Pairs that are not family members carry no bond. The comparison is strict.

```python
import numpy as np


def damage_state(s, H, sc):
    """s: (N, N) bond stretch; H: (N, N) family indicator; sc: critical stretch.
    Returns (N, N) float64: 1.0 for an intact family bond and 0.0 for a broken
    bond or a non-bond (paper Eq. 27)."""
    return np.zeros_like(np.asarray(s, dtype=np.float64))
```

### Step 5

point_damage

Goal
----
Return the source's local damage at each material point from the bond damage indicator, the family indicator and the point volumes (the paper's Eq. 30). The damage is zero when every bond of the point is intact and one when all of them are broken, and the source weights the bonds rather than merely counting them. Recover the exact weighting from the paper.

```python
import numpy as np


def point_damage(mu, H, V):
    """mu: (N, N) bond damage indicator; H: (N, N) family indicator; V: (N,)
    point volumes. Returns (N,) float64 with the source's local damage at each
    point (paper Eq. 30), between 0.0 and 1.0. The weighting used in the ratio
    is the source's convention."""
    return np.zeros(np.asarray(V).size)
```

### Step 6

internal_force

Goal
----
Assemble the internal force density at every material point for the dual-horizon formulation with spatially varying horizons (the paper's Eq. 17, with the pairwise bond force of Eq. 24 and the dual set of Eq. 11). Because family membership is asymmetric, a point receives contributions both from the bonds it owns and from the bonds owned by other points that reach it; the source's expression accounts for both, and the horizon that enters the micro-modulus of a bond is fixed by the source's convention in Eq. 24. Use E = 1.0 and A = 1.0. Recover the exact assembly and the horizon convention from the paper; the symmetric single-horizon expression is a different formula.

```python
import numpy as np


def internal_force(X, u, V, delta, H, mu):
    """X: (N,) reference coordinates; u: (N,) displacements; V: (N,) point
    volumes; delta: (N,) horizons; H: (N, N) family indicator; mu: (N, N) bond
    damage indicator. Returns (N,) float64 with the internal force density at
    each point for the source's dual-horizon formulation (paper Eq. 17 with
    Eq. 11 and Eq. 24), using E = 1.0 and A = 1.0."""
    return np.zeros(np.asarray(X).size)
```

### Step 7

critical_time_step

Goal
----
Return the source's estimate of the critical time step for explicit integration of this point cloud (the paper's Eq. 54), taken as the smallest per-point value over all points. The per-point value is built from the point's density and a sum over its family involving the bond micro-moduli, the neighbour volumes and the reference bond lengths, with the horizon convention of Eq. 24. Use E = 1.0 and A = 1.0. Recover the exact expression from the paper.

```python
import numpy as np


def critical_time_step(X, V, delta, H, rho):
    """X: (N,) reference coordinates; V: (N,) point volumes; delta: (N,)
    horizons; H: (N, N) family indicator; rho: mass density. Returns float:
    the source's critical time step for the whole point cloud (paper Eq. 54),
    using E = 1.0 and A = 1.0. Raises ValueError if a point has no family."""
    return 0.0
```

### Step 8

pd_fracture_audit

Goal
----
Run the full dual-horizon audit over the three declared bar configurations. For each configuration build its point cloud, families, stretch state and damage at the declared base critical stretch multiplied by sc_scale, then report a row of: the internal virial (the sum over points of the internal force density times the reference coordinate times the point volume), the critical time step, the mean local damage over the points, and the total number of intact bonds. Assemble by calling the earlier sub-problem functions. Use E = 1.0, A = 1.0 and rho = 1.0.

```python
import numpy as np


def pd_fracture_audit(sc_scale):
    """sc_scale: positive multiplier on the declared base critical stretch.
    Builds the three declared configurations and returns a float64 array (3, 4)
    whose rows are [internal virial, critical time step, mean local damage,
    number of intact bonds]. Assembled by calling the earlier sub-problem
    functions. Raises ValueError on a non-positive or nonfinite sc_scale."""
    return np.zeros((3, 4))
```
