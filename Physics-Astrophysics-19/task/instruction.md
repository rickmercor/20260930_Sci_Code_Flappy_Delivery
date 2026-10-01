# Physics-Astrophysics-19

## Background

Reconstructing the history of a gravitating system requires connecting observations of its evolved state to possible initial conditions. Softened gravitational interactions regularize close encounters, while hierarchical approximations reduce the cost of evaluating distant interactions. Differentiating a simulation requires accounting for how particles act both as sources of the field and as receivers of forces. Approximation choices and discrete integration can affect the inferred sensitivity even when forward trajectories appear similar.

## Problem

Gravitational reconstruction requires derivatives of a discrete approximate force solver as well as a time integrator; consider an isolated, softened eight-body system in dimensionless units and compute the directional derivative of the terminal loss below with respect to the scalar parameter \(s\) at \(s=0\).
Use a differentiable Cartesian fast multipole method, specialized to the fixed two-level interaction schedule below with total interaction order \(p=4\).
Within each parent group evaluate Plummer interactions pairwise and omit self interactions; between different parents use the two-sided Cartesian Taylor polynomial of the potential about the supplied centers, retaining total source-plus-destination degree at most \(p\).
Hold the tree assignments, centers, and interaction decisions fixed throughout the trajectory and differentiation, and obtain the reported derivative by reverse-mode (adjoint) differentiation through the frozen interaction operator and the integrator, contracting the resulting initial-position, initial-velocity and mass gradients with \(U\), \(W\) and \(\mu\).

All decimals below are exact input values, particle indices are zero-based, and the rows of \(X\) and \(V\) are listed in order:

| \(i\) | \(X_i\) | \(V_i\) |
| --- | --- | --- |
| 0 | \((-1.18,-.16,.08)\) | \((.04,.11,-.03)\) |
| 1 | \((-1.02,-.04,-.07)\) | \((-.08,.02,.06)\) |
| 2 | \((-.91,.18,.04)\) | \((.03,-.09,.04)\) |
| 3 | \((-1.13,.26,-.09)\) | \((.07,.01,-.05)\) |
| 4 | \((1.06,-.12,.13)\) | \((-.02,-.07,.05)\) |
| 5 | \((1.24,.02,-.08)\) | \((.06,.08,-.04)\) |
| 6 | \((.96,.19,.02)\) | \((-.05,.03,-.02)\) |
| 7 | \((1.17,.28,-.11)\) | \((.01,-.04,.07)\) |

\[
m=(.8,1.1,.9,1.2,1.05,.75,1.15,.95),\qquad
w=(1,1.3,.7,1.1,.9,1.4,.8,1.2).
\]

Leaf memberships are \((0,0,1,1,2,2,3,3)\), leaf-to-parent memberships are \((0,0,1,1)\), leaf centers are \((-1.1,-.1,0),(-1.1,.2,0),(1.1,-.1,0),(1.1,.2,0)\), and parent centers are \((-1.1,.05,0),(1.1,.05,0)\).
Take \(G=.7\), Plummer softening \(\epsilon=.18\), four Drift-Kick-Drift steps of length \(h=.025\), and kernel \(g(r)=-1/\sqrt{r^2+\epsilon^2}\), where acceleration is per unit target mass.
Define the initial conditions by \(x_i(0;s)=X_i+sU_i\), \(v_i(0;s)=V_i+sW_i\), and \(m_i(s)=m_i+s\mu_i\), where for \(i=0,\ldots,7\)
\[
U_i=.01(i-3.5,\;(-1)^i,\;(i\bmod3)-1).
\]
\[
W_i=.02((-1)^{i+1},\;(i-2)/3,\;1).
\]
\[
\mu_i=.03(-1)^i.
\]
The terminal target \(Y_i=X_i+.08V_i+(.015(-1)^i,-.01,.005(i-3))\) is fixed independently of \(s\); if \(x^{(4)}(s)\) is the position after four steps and \(\phi_i^{(4)}(s)\) is its potential evaluated with the same interaction schedule, the loss is
\[
J(s)=\frac12\sum_{i=0}^7w_i\|x_i^{(4)}(s)-Y_i\|^2
       +\frac{.05}{2}\sum_{i=0}^7m_i(s)\phi_i^{(4)}(s).
\]
Report \(J'(0)\), with absolute error at most \(10^{-8}\). In the reasoning, give those three contractions (position, velocity and mass), each with absolute error at most \(10^{-7}\), and state the relations that produce them: the forward relations of the frozen interaction operator and of the integrator, the terminal-loss seeds, and the adjoint relation of each operator and integrator step that carries those seeds back to the initial data.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning> . Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. 
Rules: 
- The tags are required. Do not omit them or leave them empty. 
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. 
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> concise (a few hundred words): give the relations the problem asks for and the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

kernel_derivatives

Goal
----
Evaluate Cartesian derivatives of the softened gravitational kernel.

```python
def kernel_derivatives(r: "np.ndarray", epsilon: float, p: int) -> "np.ndarray":
    """Evaluate derivatives of g(r)=-1/sqrt(r dot r+epsilon**2).

    Parameters
    ----------
    r : np.ndarray
        Finite displacement vector of shape (3,).
    epsilon : float
        Strictly positive softening length.
    p : int
        Maximum total derivative order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Float64 vector of length (p+1)(p+2)(p+3)/6 containing the raw
        derivatives D_n. Order by increasing total degree, then decreasing
        n_x, then decreasing n_y, with n_z fixed by that degree.
        Thus the first four indices are 000,100,010,001 when p >= 1.
    """
    return result
```

### Step 2

particle_moments

Goal
----
Aggregate monopole particles into raw Cartesian source moments.

```python
def particle_moments(x: "np.ndarray", masses: "np.ndarray", center: "np.ndarray", p: int) -> "np.ndarray":
    """Accumulate Q_n=sum_j masses_j (x_j-center)^n.

    Parameters
    ----------
    x : np.ndarray
        Finite positions, shape (N,3), including N=0.
    masses : np.ndarray
        Finite signed weights, shape (N,).
    center : np.ndarray
        Fixed expansion center, shape (3,).
    p : int
        Total order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Raw moment vector, ordered as kernel_derivatives. Empty input gives
        zeros. The zeroth component is the sum of the weights.
    """
    return result
```

### Step 3

translate_moments

Goal
----
Translate arbitrary raw multipoles between fixed expansion centers.

```python
def translate_moments(moments: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    """Translate raw source moments from old center to new center.

    Parameters
    ----------
    moments : np.ndarray
        Finite raw moments through total degree p, in kernel_derivatives order.
    displacement : np.ndarray
        Old center minus new center, shape (3,).
    p : int
        Total order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Raw moments of the same source distribution about the new center,
        with the same shape and kernel_derivatives order.
    """
    return result
```

### Step 4

multipole_to_local

Goal
----
Convert raw source multipoles to factorial-normalized local coefficients.

```python
def multipole_to_local(moments: "np.ndarray", displacement: "np.ndarray", epsilon: float, p: int, G: float) -> "np.ndarray":
    """Evaluate a Cartesian multipole-to-local interaction.

    Parameters
    ----------
    moments : np.ndarray
        Raw source multipoles through p, in kernel_derivatives order.
    displacement : np.ndarray
        Source center minus destination center, shape (3,).
    epsilon : float
        Positive Plummer softening.
    p : int
        Total interaction order, 0 <= p <= 6. Retain |n|+|k| <= p.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Normalized local coefficients L_k = (1/k!) d^k phi/dx^k at the
        destination center, where phi is G times the potential of the raw
        source moments under the two-sided Cartesian expansion of the kernel
        about the source and destination centers, keeping only terms with
        |n| + |k| <= p. Derivatives are with respect to destination position;
        ordered as kernel_derivatives.
    """
    return result
```

### Step 5

translate_locals

Goal
----
Shift factorial-normalized locals toward a destination particle.

```python
def translate_locals(locals_in: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    """Translate a local polynomial from an old center to a new center.

    Parameters
    ----------
    locals_in : np.ndarray
        Normalized local coefficients through p, in kernel_derivatives order.
    displacement : np.ndarray
        New center minus old center, shape (3,).
    p : int
        Total polynomial degree, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Same-shape normalized local coefficients L_n = (1/n!) d^n P at the
        new center, where P is the degree-p local polynomial that locals_in
        represents about the old center.
        Coefficient 000 is potential and minus the three first-degree
        coefficients is acceleration per unit target mass.
    """
    return result
```

### Step 6

evaluate_fixed_fmm

Goal
----
Evaluate the two-level frozen Cartesian interaction operator.

```python
def evaluate_fixed_fmm(x: "np.ndarray", moments: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    """Evaluate a prescribed fixed hierarchy on monopole or generalized sources.

    Parameters
    ----------
    x : np.ndarray
        Positions of N >= 1 particles, shape (N,3).
    moments : np.ndarray
        Particle-centered raw multipoles, shape (N,M), where
        M=(p+1)(p+2)(p+3)/6, in kernel_derivatives order. Signed moments
        are allowed. Ordinary particles have only the mass component.
    tree : dict
        Fixed geometry: leaf_ids is an integer (N,) array; parent_ids is an
        integer (B,) array assigning each of B leaves to a parent; leaf_centers
        and parent_centers have shapes (B,3) and (A,3). IDs are in range.
        All arrays are finite; empty leaves are allowed. A >= 1, B >= 1.
        Different parents interact at parent level; particles in the same
        parent interact directly, excluding identical particle indices.
    epsilon : float
        Positive softening length.
    p : int
        Total interaction order, 0 <= p <= 6.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Particle-centered normalized locals, shape (N,M). Inter-parent
        translations retain total source-plus-destination degree <= p.
        Within-parent pairs use the unexpanded softened kernel at the
        particle displacement, with the same |source|+|local| <= p rule
        for generalized moments. Self contributions are excluded.
        The fixed schedule applies even if particles move across centers.
    """
    return result
```

### Step 7

pullback_fixed_fmm

Goal
----
Pull back potential and acceleration sensitivities through the frozen FMM.

```python
def pullback_fixed_fmm(x: "np.ndarray", masses: "np.ndarray", local_bar: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    """Compute the VJP of monopole-particle locals with fixed hierarchy geometry.

    Parameters
    ----------
    x : np.ndarray
        Finite particle positions, shape (N,3), N >= 1.
    masses : np.ndarray
        Finite source masses, shape (N,), signed values supported.
    local_bar : np.ndarray
        Cotangents of normalized locals, shape (N,M). Only entries of total
        degree 0 or 1 may be nonzero.
    tree : dict
        Fixed geometry: leaf_ids is an integer (N,) array; parent_ids is an
        integer (B,) array assigning each of B leaves to a parent; leaf_centers
        and parent_centers have shapes (B,3) and (A,3). IDs are in range.
        All arrays are finite; empty leaves are allowed. A >= 1, B >= 1.
        Different parents interact at parent level; particles in the same
        parent interact directly, excluding identical particle indices.
    epsilon : float
        Positive softening.
    p : int
        Total interaction order, 2 <= p <= 6.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Shape (N,4); columns 0:3 are position cotangents and column 3 is
        mass cotangent for the contraction sum(local_bar * locals).
        Both source and destination position dependence is included.
        Centers, topology, softening and G are held fixed.
    """
    return result
```

### Step 8

trajectory_sensitivity

Goal
----
Compute the terminal-loss directional derivative through a DKD trajectory.

```python
def trajectory_sensitivity(config: dict) -> float:
    """Return the analytic derivative of the specified discrete terminal loss.

    Parameters
    ----------
    config : dict
        x, v: finite (N,3) initial positions and velocities, N >= 1.
        masses: positive (N,) masses; weights: nonnegative (N,) loss weights.
        target: fixed finite (N,3) terminal target independent of perturbation.
        dx, dv, dm: finite perturbation directions of shapes (N,3), (N,3), (N,).
        tree: fixed geometry dictionary with the evaluate_fixed_fmm contract.
        epsilon > 0, integer 2 <= p <= 6, finite G >= 0, finite h >= 0,
        integer steps >= 0, finite lam >= 0.
        For parameter s, initial x,v,masses are perturbed by s times dx,dv,dm.
        Advance the state with steps Drift-Kick-Drift updates of length h
        driven by the per-unit-mass field acceleration, reusing the fixed
        tree at every evaluation including the terminal potential.

    Returns
    -------
    result : float
        dJ/ds at s=0, where J is half the weighted sum of squared terminal
        position residuals plus lam/2 times sum_i masses_i phi_i at the
        terminal positions. Include explicit and force-mediated mass effects.
        Zero steps is the unevolved terminal-loss sensitivity.
        All calculations use the finite-order interaction model.
    """
    return result
```
