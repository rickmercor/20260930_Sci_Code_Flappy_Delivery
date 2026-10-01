# Physics-Astrophysics-21

## Problem

Compute the pressure produced by the published stage-stabilization prescription for nodal polynomial approximations of axisymmetric relativistic flow that maintains conservation when the geometric measure vanishes at the axis and enforces a dimensionless margin from the physical-state boundary.

Use a flat stationary background in cylindrical coordinates (r,z), zero azimuthal momentum, c=1, and an ideal gas with adiabatic index 5/3; the stored state is W=r(D,m_r,m_z,E), where E includes rest-mass energy. The domain [0,2] by [0,2] consists of four distinct cells K_ab=[a,a+1] by [b,b+1], a,b in {0,1}, each with a tensor Q2 nodal polynomial at local coordinates 0,1/2,1 in both directions and mapped Gauss-Lobatto weights (1,4,1)/6 per direction. At each node (r,z) of K_ab the raw stage coefficients are

\[
W=r\bigl(1+0.1z+0.04a+0.01b,\;0.15r,\;0.2+0.05z+0.03a,\;3+0.2z+0.07a+0.11b\bigr),
\]

except in K_00, where W(0,1/2)=(0.01,0,0.005,0.03) and W(1/2,1/2)=(-0.05,0.8,0.1,0.65). The stabilization time interval is 0.02, the dimensionless oscillation-damping strength is 0.02, the absolute density and cone floors are both 1e-11, the relative-margin fraction is 0.1, and the machine precision for noise screening is 2^-52. The four cells constitute the entire normalization domain, and the patch's exterior faces do not participate in its interior-face variation measurements.

The requested result is the gas pressure on the K_00 side of (r,z)=(1,1/2) after this single stage stabilization of the supplied raw coefficients, within relative tolerance 1e-7 or absolute tolerance 1e-8, whichever is larger.

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

01_repair_axis

Goal
----
Restore an axis-compatible stored representation of a raw tensor nodal stage.

```python
def repair_axis(raw):
    """Return repaired stored nodal values for the four-cell patch.

    Parameters
    ----------
    raw : array_like, shape (2,2,3,3,4)
        Finite raw stored states, indexed by (a,b,radial_node,axial_node,component).

    Returns
    -------
    ndarray, shape (2,2,3,3,4)
        Conservative axis-compatible stored states. The input is not modified.
        Invalid shape or nonfinite entries raise ValueError.
    """
    return
```

### Step 2

02_regular_local_state

Goal
----
Recover the finite local orthonormal state associated with stored cell data.

```python
def regular_local_state(stored, radial_cell):
    """Return the regular local state on one cell, including any axis trace.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite stored states in radial-node, axial-node, component order.
        Axis-face stored values must already be zero.
    radial_cell : int
        Radial index a, either 0 or 1, for K_ab in the unit-width patch.

    Returns
    -------
    ndarray, shape (3,3,4)
        Finite local states in component order (D,m_r,m_z,E). The input is not
        modified. Invalid data or a nonzero stored axis face raise ValueError.
    """
    return
```

### Step 3

03_directional_jump_amplitudes

Goal
----
Measure interface irregularity of local Q2 states on a four-cell patch.

```python
def directional_jump_amplitudes(local_states):
    """Return the source's interior-face jump amplitudes.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite local states at nodes (0,1/2,1), with duplicated DG interfaces.

    Returns
    -------
    ndarray, shape (2,2,3,2,4)
        Entries indexed by (a,b,derivative_order,normal_direction,component).
        Direction 0 is radial and direction 1 is axial. The input is not
        modified. Invalid data raise ValueError.
    """
    return
```

### Step 4

04_normalized_damping_rates

Goal
----
Assign source-consistent damping rates to derivative orders on the patch.

```python
def normalized_damping_rates(local_states, jump_amplitudes):
    """Return order-dependent system damping rates for the four cells.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite local nodal states with the geometry and ordering of step 3.
    jump_amplitudes : array_like, shape (2,2,3,2,4)
        Nonnegative finite amplitudes in step 3's output ordering.

    Returns
    -------
    ndarray, shape (2,2,3)
        Rates indexed by (a,b,derivative_order), using binary64 precision for
        the source's noise screens. Neither input is modified. Invalid data
        raise ValueError.
    """
    return
```

### Step 5

05_conservative_local_filter

Goal
----
Apply local-state oscillation elimination while retaining stored cell means.

```python
def conservative_local_filter(local_states, rates, dt=0.02, strength=0.02):
    """Return the filtered stored states on the four unit-width cells.

    Parameters
    ----------
    local_states : array_like, shape (2,2,3,3,4)
        Finite regular local states at the patch's Q2 nodes.
    rates : array_like, shape (2,2,3)
        Finite nonnegative order-dependent rates from step 4.
    dt, strength : float
        Finite nonnegative timestep and oscillation-elimination strength.

    Returns
    -------
    ndarray, shape (2,2,3,3,4)
        Filtered stored states W, indexed as in step 1. Neither array input is
        modified. Invalid inputs raise ValueError. Zero dt or strength gives
        the undamped map to stored data.
    """
    return
```

### Step 6

06_absolute_physical_scaling

Goal
----
Impose the source's absolute physical constraints on a stored cell state.

```python
def absolute_physical_scaling(stored, radial_cell, density_floor=1e-11, cone_floor=1e-11):
    """Return stored states after the absolute density and cone constraints.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite axis-compatible stored nodal states with admissible mean.
    radial_cell : int
        Radial cell index a in {0,1}; nodes are a+(0,1/2,1).
    density_floor, cone_floor : float
        Positive finite requested absolute floors.

    Returns
    -------
    ndarray, shape (3,3,4)
        Conservatively scaled stored states, in (D,m_r,m_z,E) order. The input
        is not modified. Invalid inputs or an inadmissible cell mean raise
        ValueError.
    """
    return
```

### Step 7

07_relative_physical_scaling

Goal
----
Enforce the source's relative distance from the relativistic cone boundary.

```python
def relative_physical_scaling(stored, radial_cell, fraction=0.1):
    """Return the stored states after relative physical certification.

    Parameters
    ----------
    stored : array_like, shape (3,3,4)
        Finite axis-compatible stored states after absolute certification,
        with an admissible conservative cell mean.
    radial_cell : int
        Radial index a in {0,1} on the same unit-width patch.
    fraction : float
        Finite fraction in [0,1) specifying relative certification strength.

    Returns
    -------
    ndarray, shape (3,3,4)
        Conservatively certified stored state. The input is not modified.
        Invalid input, inadmissible mean, or negative local density/cone
        margin raise ValueError.
    """
    return
```

### Step 8

08_primitive_pressure

Goal
----
Recover ideal-gas pressure from an admissible relativistic local state.

```python
def primitive_pressure(local_state, gamma=5/3):
    """Return the physical pressure of one flat-frame conservative state.

    Parameters
    ----------
    local_state : array_like, shape (4,)
        Finite (D,m_r,m_z,E) with D>0 and E>sqrt(D^2+m_r^2+m_z^2).
    gamma : float
        Finite ideal-gas adiabatic index in (1,2].

    Returns
    -------
    float
        Positive pressure in the same units as E. The input is not modified.
        Invalid data raise ValueError.
    """
    return
```

### Step 9

09_stabilized_patch_pressure

Goal
----
FINAL ORCHESTRATOR: pressure of the finite prescribed stage on a Q2 patch.

```python
def stabilized_patch_pressure(raw=None, dt=0.02, strength=0.02):
    """Return the K00 pressure at (r,z)=(1,1/2) after the prescribed stage.

    Parameters
    ----------
    raw : array_like, shape (2,2,3,3,4), optional
        Stored patch values ordered (a,b,radial_node,axial_node,component).
        If omitted, the four cells K_ab=[a,a+1]x[b,b+1], a,b in {0,1}, use
        local nodes (0,1/2,1) and raw values
        r*(1+0.1*z+0.04*a+0.01*b, 0.15*r, 0.2+0.05*z+0.03*a,
        3+0.2*z+0.07*a+0.11*b). In K00 overwrite node [0,1] with
        (0.01,0,0.005,0.03) and [1,1] with (-0.05,0.8,0.1,0.65).
    dt, strength : float
        Nonnegative finite timestep and oscillation-elimination strength.
        Both requested absolute floors are 1e-11, relative fraction is 0.1,
        Gamma=5/3, c=1, chi=r; there is no PDE residual or boundary jump.

    Returns
    -------
    float
        Pressure at K00 radial node 2 and axial node 1. The supplied array is
        not modified. Invalid data or any inadmissible conservative cell mean
        raise ValueError. Assemble this result by calling the preceding
        sub-problem functions.
    """
    return
```
