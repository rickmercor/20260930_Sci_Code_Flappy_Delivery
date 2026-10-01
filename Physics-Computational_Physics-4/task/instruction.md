# Physics-Computational_Physics-4

## Background

Multiphase lattice Boltzmann models represent fluids through populations moving on a discrete velocity lattice. Density-dependent interactions can support liquid and gas phases, while collision in moment space controls relaxation and transport. Near an interface, lattice discretization can produce velocity errors even when the continuum momentum equation is recovered at leading order. Studying higher-order source moments helps distinguish those numerical artifacts from physical viscous flow. A two-phase channel driven by a body force provides a useful setting for comparing the simulated velocity with a continuum momentum balance.

## Problem

A pseudopotential lattice Boltzmann model with a multiple-relaxation-time collision is used to simulate two-phase plane Poiseuille flow between two parallel no-slip walls, driven by a constant body force parallel to the walls; the fluid obeys a van der Waals equation of state and the third-order form of the model carries a discrete source term in moment space whose high-order entries the standard Chapman-Enskog analysis leaves undetermined, so they must be taken from the reference below. Measure the finite-time profile deviation produced by this prescribed numerical scheme, including its transient and boundary effects.

Configuration: $N=128$ nodes between the walls with the first and last held no-slip; van der Waals constants $a=9/49$, $b=2/21$, $R=1$, the whole equation of state multiplied by $K_{\mathrm{EOS}}=1/16$; unit lattice spacing and time step, $c_s^2=1/3$, interaction strength $G=-1$ with nearest-neighbour interaction weights $1/3$ axial and $1/12$ diagonal, the stencil closed periodically in the wall-normal direction so that a neighbour lying outside the channel takes the pseudopotential of the node it wraps to at the opposite wall; reduced temperature $T_r=0.54$, mechanical-stability parameter $\epsilon=2$, relaxation time $\tau=0.8$, driving force magnitude $2.0\times10^{-7}$; relaxation rates $s_0=s_j=1$, $s_e=s_{\epsilon}=s_p=1/\tau$, and $s_q$ fixed by

$$
\left(\frac{1}{s_q}-\frac{1}{2}\right)
\left(\frac{1}{s_p}-\frac{1}{2}\right)=\frac{1}{12}.
$$

Settle the interface by initialising the density as

$$
\rho_{\mathrm{cr}}+0.15\rho_{\mathrm{cr}}
\left[
\tanh\!\left(\frac{y-N/4}{8}\right)
-\tanh\!\left(\frac{y-3N/4}{8}\right)
\right]
$$

on nodes $y=0,\ldots,N-1$ with equilibrium distributions at zero velocity, then holding the system at each of $16$ reduced temperatures spaced linearly from $0.95$ down to $0.54$, both endpoints included, for $4000$ unforced steps each; from that state apply the driving force and run $80000$ further steps.

Four numerical conventions fix the comparison and are part of the specification. Both velocity components at the two wall nodes are reset to zero at collision time, before the moments are formed, not afterwards, and the profile reported at the end carries zero at those two nodes as well. The walls use same-node on-node bounce-back applied after streaming: at each wall node the three populations pointing into the fluid are replaced by the pre-streaming post-collision populations of their opposite directions at that same node. The reference profile is integrated with the density profile taken after the driven stage, not the settled pre-drive profile. The reference solve is a centered conservative discretisation of

$$
\frac{\mathrm d}{\mathrm dy}
\left(\rho\nu\frac{\mathrm du}{\mathrm dy}\right)=-|F|
$$

with no-slip walls, $\nu=c_s^2(\tau-1/2)$, in which the dynamic viscosity on each cell face is the arithmetic mean of its two neighbouring nodal values.

Compare the resulting wall-parallel velocity profile against that reference and report a single number: the relative two-norm deviation

$$
\frac{\lVert u-u_{\mathrm{ref}}\rVert_2}{\lVert u_{\mathrm{ref}}\rVert_2}.
$$

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

vdw_pressure

Goal
----
Evaluate the equation of state of the working fluid at the given densities and reduced temperature, using the scaled form adopted by the source.

```python
def vdw_pressure(rho: np.ndarray, Tr: float) -> np.ndarray:
    """Evaluate the equation of state of the working fluid at the given densities and reduced temperature, using the scaled form adopted by the source.

    Parameters
    ----------
    rho : np.ndarray
        Densities at each lattice node.
    Tr : float
        Reduced temperature T/T_cr.

    Returns
    -------
    np.ndarray, the pressure at each node as float64
    """
    return result  # placeholder
```

### Step 2

cohesion_field

Goal
----
Invert the model's pressure relation to obtain the pseudopotential at each node. Raise ValueError where the relation admits no real value.

```python
def cohesion_field(rho: np.ndarray, Tr: float) -> np.ndarray:
    """Invert the model's pressure relation to obtain the pseudopotential at each node. Raise ValueError where the relation admits no real value.

    Parameters
    ----------
    rho : np.ndarray
        Densities at each lattice node.
    Tr : float
        Reduced temperature T/T_cr.

    Returns
    -------
    np.ndarray, the pseudopotential at each node as float64
    """
    return result  # placeholder
```

### Step 3

pairwise_force

Goal
----
Compute the wall-normal component of the mesoscopic interaction force on a one-dimensional periodic lattice using the isotropy-weighted nearest-neighbour stencil of the source.

```python
def pairwise_force(psi: np.ndarray) -> np.ndarray:
    """Compute the wall-normal component of the mesoscopic interaction force on a one-dimensional periodic lattice using the isotropy-weighted nearest-neighbour stencil of the source.

    Parameters
    ----------
    psi : np.ndarray
        Pseudopotential on a 1-D periodic lattice.

    Returns
    -------
    np.ndarray, the wall-normal force component at each node as float64
    """
    return result  # placeholder
```

### Step 4

equilibrium_moments

Goal
----
Assemble the equilibrium moment vector of the multiple-relaxation-time model for the standard nine-velocity lattice.

```python
def equilibrium_moments(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Assemble the equilibrium moment vector of the multiple-relaxation-time model for the standard nine-velocity lattice.

    Parameters
    ----------
    rho, ux, uy : np.ndarray
        Density and the two velocity components at each node.

    Returns
    -------
    np.ndarray of shape (9, N), the equilibrium moments as float64
    """
    return result  # placeholder
```

### Step 5

discrete_force_moments

Goal
----
Assemble the discrete force moment vector entering the collision step.

```python
def discrete_force_moments(Fx: np.ndarray, Fy: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Assemble the discrete force moment vector entering the collision step.

    Parameters
    ----------
    Fx, Fy : np.ndarray
        Total force components at each node.
    ux, uy : np.ndarray
        Velocity components at each node.

    Returns
    -------
    np.ndarray of shape (9, N), the force moments as float64
    """
    return result  # placeholder
```

### Step 6

improved_source_moments

Goal
----
Assemble the third-order discrete source moment vector in the form the source paper prescribes for suppressing interfacial artefacts. Apply the source's own choice for the entry the analysis leaves undetermined.

```python
def improved_source_moments(Fix: np.ndarray, Fiy: np.ndarray, ux: np.ndarray, uy: np.ndarray, psi: np.ndarray, eps: float) -> np.ndarray:
    """Assemble the third-order discrete source moment vector in the form the source paper prescribes for suppressing interfacial artefacts. Apply the source's own choice for the entry the analysis leaves undetermined.

    Parameters
    ----------
    Fix, Fiy : np.ndarray
        Interaction force components at each node.
    ux, uy : np.ndarray
        Velocity components at each node.
    psi : np.ndarray
        Pseudopotential at each node.
    eps : float
        The parameter appearing in the mechanical stability condition.

    Returns
    -------
    np.ndarray of shape (9, N), the source moments as float64
    """
    return result  # placeholder
```

### Step 7

collide_stream

Goal
----
Advance the distribution functions by one collision and one streaming step, with no-slip walls at the first and last nodes.

```python
def collide_stream(f: np.ndarray, Tr: float, eps: float, tau: float, drive: float, improved: bool) -> np.ndarray:
    """Advance the distribution functions by one collision and one streaming step, with no-slip walls at the first and last nodes.

    Parameters
    ----------
    f : np.ndarray of shape (9, N)
        Distribution functions.
    Tr : float
        Reduced temperature.
    eps : float
        Mechanical-stability parameter.
    tau : float
        Dimensionless relaxation time.
    drive : float
        Constant driving force parallel to the walls.
    improved : bool
        Select the source paper's source term rather than the earlier one.

    Returns
    -------
    np.ndarray of shape (9, N), the post-streaming distributions as float64
    """
    return result  # placeholder
```

### Step 8

equilibrate_two_phase

Goal
----
Relax an initially smooth two-layer profile to a settled interface at the target temperature, approaching it through a sequence of intermediate temperatures rather than in one jump.

```python
def equilibrate_two_phase(N: int, Tr_target: float, eps: float, tau: float, n_stages: int, per_stage: int) -> np.ndarray:
    """Relax an initially smooth two-layer profile to a settled interface at the target temperature, approaching it through a sequence of intermediate temperatures rather than in one jump.

    Parameters
    ----------
    N : int
        Number of lattice nodes across the channel.
    Tr_target : float
        Final reduced temperature.
    eps : float
        Mechanical-stability parameter.
    tau : float
        Dimensionless relaxation time.
    n_stages : int
        Number of temperature stages.
    per_stage : int
        Steps held at each stage.

    Returns
    -------
    np.ndarray of shape (9, N), the settled distributions as float64
    """
    return result  # placeholder
```

### Step 9

theory_velocity

Goal
----
Solve the one-dimensional steady momentum balance for the velocity profile driven by a constant body force, using the given density profile and no-slip walls.

```python
def theory_velocity(rho: np.ndarray, tau: float, Fdri: float) -> np.ndarray:
    """Solve the one-dimensional steady momentum balance for the velocity profile driven by a constant body force, using the given density profile and no-slip walls.

    Parameters
    ----------
    rho : np.ndarray
        Settled density profile across the channel.
    tau : float
        Dimensionless relaxation time.
    Fdri : float
        Constant driving force.

    Returns
    -------
    np.ndarray, the reference velocity at each node as float64
    """
    return result  # placeholder
```

### Step 10

scheme_accuracy_audit

Goal
----
Settle the interface, apply the constant force for the given number of steps, and report how far the resulting velocity profile sits from the reference profile in a relative two-norm.

```python
def scheme_accuracy_audit(N: int, Tr: float, eps: float, tau: float, Fdri: float, n_stages: int, per_stage: int, drive_steps: int) -> float:
    """Settle the interface, apply the constant force for the given number of steps, and report how far the resulting velocity profile sits from the reference profile in a relative two-norm.

    Parameters
    ----------
    N, Tr, eps, tau, Fdri, n_stages, per_stage, drive_steps
        The full configuration of the run.

    Returns
    -------
    float, the relative two-norm deviation from the reference profile
    """
    return result  # placeholder
```
