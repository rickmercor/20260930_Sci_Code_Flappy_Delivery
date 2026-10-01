# Material_Science-Molecular_Modeling-53

## Background

Chain scission - the breaking of a covalent backbone bond under load - is the molecular event that underlies damage and fracture in polymer networks, and network-scale damage models need its rate as an input. Predicting that rate is not simply a matter of tilting a bond potential by the applied force: a bond sits in a three-dimensional chain, so its rupture kinetics are shaped by the configurational entropy of the surrounding conformations as well as by the stretching energy. When the chain has finite bending stiffness the surrounding conformations stop being independent - neighbouring bond orientations are correlated, the rest of the chain no longer factors out, and the free-energy landscape of a bond depends on where along the chain it sits. Propagating those orientational correlations with a transfer matrix, and fixing the rupture thresholds self-consistently from the resulting landscapes, makes the position dependence explicit and exposes which bond in a short chain is the weakest link.

## Problem

Chain scission - the breaking of a covalent backbone bond under load - is the molecular event underlying damage and fracture in polymer networks. The source formulates scission within transition-state theory as a multichannel first-rupture problem in bond-length space, with bond-resolved rates governed by potentials of mean force that retain the three-dimensional configurational freedom of the chain. When the chain has finite bending stiffness, neighbouring bond orientations become correlated and the surrounding chain no longer drops out: the source propagates those orientational correlations along the chain with a transfer-matrix construction, and determines the rupture thresholds self-consistently from the resulting free-energy landscapes. Your job is to implement that finite-bending construction exactly as the source specifies it and audit the resulting bond-resolved activation barriers. The load-bearing algorithmic choices - how the bending energy becomes an angular coupling kernel between successive bond orientations, how the local bond contribution is separated from that coupling, how the chain's orientational correlations are propagated along it and combined into the angular weight of an individual bond, the entropic content of the resulting potential of mean force, and the variational choice of rupture threshold - are the source's; recover them from the literature. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

Work throughout in the reduced units De = 1.0 and le = 1.0, with a = 2.15 and beta = 279.0, so that force is measured in units of De/le. The bending interaction uses an equilibrium bond angle of 69 degrees and a bending stiffness kphi fixed by beta*kphi*pi**2 = 1820. These correspond to the carbon-carbon parameters used in the source. The audit covers three configurations, each a chain of N bonds at base force f0, with (N, f0) = (5, 0.20), (4, 0.15) and (3, 0.25); each is evaluated at force f = f0 * force_scale.

Fix the following numerical conventions so the results are reproducible. Use a uniform grid of 121 polar angles spanning 0 to pi, and if any further angular integration enters, a uniform grid of 121 points spanning 0 to 2*pi; integrate on both with composite Simpson weights. Integrate over bond length with composite Simpson quadrature on 1201 uniform points from zero to the bond's rupture threshold. Locate the stationary points of a potential of mean force by scanning bond lengths on the closed interval [0.6, 6.0] with 24001 uniformly spaced points, taking the first sign change of the finite difference from decreasing to increasing as the bonded minimum and the next sign change from increasing to decreasing as the barrier top, then refining each by 80 iterations of ternary search on the starting half-width 0.02. Start every rupture threshold at 2.4 and run exactly 8 self-consistency sweeps, which is far past convergence here. Because beta is large, quantities such as the local bond weight span hundreds of orders of magnitude, so the computation has to be arranged to stay numerically stable rather than overflowing or underflowing; the reported numbers must not depend on how you achieve that. Where the configurational measure vanishes, at both polar-angle endpoints, use a finite floor rather than a negative infinity.

Implement eight functions. morse_potential(l, De, a, le) returns the source's Morse bond-stretching energy. bending_kernel(beta, kphi, phi_e) returns the (121, 121) angular coupling kernel between the polar angles of two successive bonds. local_intact_weight(lt, f, beta, De, a, le) returns the (121,) natural log of the source's local intact weight of one bond at each grid polar angle, raising ValueError on a non-positive threshold. tm_angular_weights(thresholds, f, beta, De, a, le, Q) returns the (N, 121) angular weight of every bond, obtained by propagating the source's messages along the chain. tm_pmf(l, w_i, f, beta, De, a, le) returns the source's bond-length potential of mean force for a bond of given angular weight. self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e) returns the (N,) converged rupture thresholds, raising ValueError unless N is an integer of at least 2. bond_barriers(N, f, beta, De, a, le, kphi, phi_e) returns the (N,) activation barriers in units of kT. scission_audit(force_scale), the final step, must be assembled by calling the earlier functions: it evaluates the three configurations at their scaled forces and returns a float64 array (3, 5) whose rows are [rupture threshold of the first bond, rupture threshold of the bond at index N//2, activation barrier of the first bond in kT, largest bond activation barrier in kT, sum of the bond activation barriers in kT].

Evaluate scission_audit with force_scale = 1.0. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. In your reasoning report, for each configuration, the full list of bond activation barriers, the rupture threshold of the first bond, and the summed barrier; then the running total across the three configurations. Also state how the finite-bending barriers compare with the corresponding freely-jointed barrier that the source obtains when the bending interaction is switched off, and what the shape of the barrier profile along the chain implies about which bond is the weakest link. Keep the reasoning under roughly 900 words, naming the source's conventions briefly rather than deriving them, and do not tabulate the angular grid, the coupling kernel, the messages, or any per-grid-point values. As the final answer, report the sum over the three configurations of the summed bond activation barriers (the fifth column) to six significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 12.3456, -0.802, 45.9). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Budget your response so that the <final_answer> tag is always reached and closed. If you are running long, stop the intermediate work and emit the answer.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 12.3456, -0.802, 45.9). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Budget your response so that the <final_answer> tag is always reached and closed. If you are running long, stop the intermediate work and emit the answer.
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

morse_potential

Goal
----
Evaluate the source's Morse bond-stretching potential at the given bond length(s), for dissociation energy De, inverse range parameter a and equilibrium length le (the paper's Eq. S8).

```python
import numpy as np


def morse_potential(l, De, a, le):
    """l: bond length(s); De, a, le: Morse parameters. Returns float64 array
    shaped like l with the source's Morse stretching energy (paper Eq. S8)."""
    return np.zeros_like(np.asarray(l, dtype=np.float64))
```

### Step 2

bending_kernel

Goal
----
Build the source's angular coupling kernel between the polar angles of two successive bonds (the paper's Eq. S61). The bending energy penalises departures of the local bond angle from its equilibrium value with stiffness kphi (Eq. S9); because only the RELATIVE azimuth of the two bonds is unconstrained, the source removes it by integrating over it, which is what turns a bending energy of the bond angle into a kernel in the two polar angles. Recover from the paper both how the bond angle follows from the two polar angles and that relative azimuth, and what is integrated. Use a uniform grid of 121 polar angles spanning 0 to pi and a uniform grid of 121 azimuths spanning 0 to 2*pi, both with composite Simpson weights.

```python
import numpy as np


def bending_kernel(beta, kphi, phi_e):
    """beta: inverse temperature; kphi: bending stiffness; phi_e: equilibrium
    bond angle. Returns (121, 121) float64: the source's angular coupling kernel
    on the declared polar-angle grid (paper Eq. S61)."""
    return np.zeros((121, 121))
```

### Step 3

local_intact_weight

Goal
----
Return the natural logarithm of the source's local intact weight for one bond held at each polar angle of the declared grid, given that bond's rupture threshold (the paper's Eq. S60). The weight integrates the Boltzmann factor of the bond over the allowed bond-length interval, carrying the configurational measure the source uses. Because beta is large the raw weight spans hundreds of orders of magnitude, so return its logarithm and build it stably. Integrate over bond length with composite Simpson quadrature on 1201 uniform points from zero to the threshold. Where the measure vanishes at the two grid endpoints the log is returned as the finite floor -1e300 rather than minus infinity, so the array stays comparable. Raise ValueError on a non-positive threshold.

```python
import numpy as np


def local_intact_weight(lt, f, beta, De, a, le):
    """lt: rupture threshold of this bond; f: applied force; beta: inverse
    temperature; De, a, le: Morse parameters. Returns (121,) float64 with the
    natural log of the source's local intact weight at each grid polar angle
    (paper Eq. S60); wherever the configurational measure vanishes -- which on
    this grid means BOTH polar-angle endpoints, judged by sin(theta) <= 1e-12 so
    that theta = pi is treated exactly like theta = 0 -- the returned value is the
    finite floor -1e300 rather than -inf. Raises ValueError on a non-positive
    threshold."""
    return np.zeros(121)
```

### Step 4

tm_angular_weights

Goal
----
Propagate the source's transfer-matrix messages along the chain and return the angular weight of every bond on the declared polar-angle grid (the paper's Eqs. S72 to S79). One message sweeps from the first bond toward the last and the other sweeps back, each step folding in the local intact weight of the bond just passed and the angular coupling kernel; the two boundary messages are uniform in the polar angle and normalised. Only the NORMALISED message shapes are needed, so renormalise each message to unit integral over the polar angle after every step and the accumulated magnitudes never have to be carried. The angular weight of a bond is then formed from the two messages meeting at that bond, in the source's way. Recover the propagation, the normalisation and the combination rule from the paper.

```python
import numpy as np


def tm_angular_weights(thresholds, f, beta, De, a, le, Q):
    """thresholds: (N,) rupture thresholds; f: applied force; beta: inverse
    temperature; De, a, le: Morse parameters; Q: (121, 121) angular coupling
    kernel. Returns (N, 121) float64 with the source's angular weight of each
    bond on the grid (paper Eqs. S72-S79)."""
    n = np.asarray(thresholds).size
    return np.zeros((n, 121))
```

### Step 5

tm_pmf

Goal
----
Evaluate the source's bond-length potential of mean force for a bond whose angular weight is given, at the requested bond length(s) (the paper's Eq. S82). Besides the bare stretching energy the PMF contains an entropic contribution: an angular average over the grid, weighted by that bond's angular weight, of the force coupling, together with the configurational measure the source uses. An l-independent additive constant is irrelevant and may be dropped. Recover the exact expression from the paper.

```python
import numpy as np


def tm_pmf(l, w_i, f, beta, De, a, le):
    """l: bond length(s); w_i: (121,) angular weight of the selected bond;
    f: applied force; beta: inverse temperature; De, a, le: Morse parameters.
    Returns a float64 array shaped like l with the source's bond-length PMF
    (paper Eq. S82)."""
    return np.zeros_like(np.atleast_1d(np.asarray(l, dtype=np.float64)))
```

### Step 6

self_consistent_thresholds

Goal
----
Determine the source's rupture thresholds for a chain of N bonds self-consistently (the numbered procedure in the paper's Sec. S2.2). Each bond's PMF depends parametrically on the thresholds of the OTHER bonds, so the source alternates between rebuilding the angular weights for the current threshold set and updating each threshold to the stationary point that the variational transition-state principle selects, until the set stops changing. Start every threshold at 2.4 and run exactly 8 sweeps, which is far past convergence here. Locate stationary points by scanning bond lengths on [0.6, 6.0] with 24001 uniform points, taking the first decreasing-to-increasing sign change and then the next increasing-to-decreasing one, refining each by 80 ternary-search iterations on half-width 0.02. Raise ValueError unless N is an integer of at least 2. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e):
    """N: number of bonds (>= 2); f: applied force; beta: inverse temperature;
    De, a, le: Morse parameters; kphi, phi_e: bending parameters. Returns (N,)
    float64 with the source's self-consistent rupture thresholds. Raises
    ValueError on an invalid N."""
    return np.zeros(int(N))
```

### Step 7

bond_barriers

Goal
----
Return the source's bond-resolved activation barriers, in units of kT, for a chain of N bonds at the given force. For each bond the barrier is the PMF difference between the two stationary points of that bond's converged PMF, taken in the source's sense. Because the chain ends truncate the orientational correlations, the barriers depend on bond position and the source's construction makes them symmetric about the chain midpoint. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def bond_barriers(N, f, beta, De, a, le, kphi, phi_e):
    """N: number of bonds; f: applied force; beta: inverse temperature;
    De, a, le: Morse parameters; kphi, phi_e: bending parameters. Returns (N,)
    float64 with the source's activation barrier of each bond in units of kT.
    Assembled by calling the earlier sub-problem functions."""
    return np.zeros(int(N))
```

### Step 8

scission_audit

Goal
----
Run the full finite-bending chain-scission audit over the three declared configurations. Each configuration is a chain of N bonds at base force f0 multiplied by force_scale, with (N, f0) = (5, 0.20), (4, 0.15) and (3, 0.25). Report a row of: the converged rupture threshold of the first bond, the converged threshold of the bond at index N//2, the activation barrier of the first bond in kT, the largest bond activation barrier in kT, and the sum of the bond activation barriers in kT. Use De = 1.0, a = 2.15, le = 1.0, beta = 279.0, equilibrium bond angle 69 degrees, and a bending stiffness such that beta*kphi*pi**2 = 1820. Assemble by calling the earlier sub-problem functions. Raise ValueError on a non-positive or nonfinite force_scale.

```python
import numpy as np


def scission_audit(force_scale):
    """force_scale: positive multiplier on each configuration's base force.
    Returns (3, 5) float64 whose rows are [first-bond threshold, threshold at
    index N//2, first-bond barrier in kT, largest bond barrier in kT, sum of
    bond barriers in kT]. Assembled by calling the earlier sub-problem
    functions. Raises ValueError on a non-positive or nonfinite force_scale."""
    return np.zeros((3, 5))
```
