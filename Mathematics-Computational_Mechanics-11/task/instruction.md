# Linear stability of the revised FMPM(k) material point update: the Courant limit of the full mass matrix expansion on the source's vibrating bar

## Background

A material point method time step moves information from particles to a background grid, updates the grid momenta with the internal forces, converts the momenta to grid velocities and maps the result back to the particles. Standard MPM divides grid momenta by a lumped mass matrix; the approximate full mass matrix method, FMPM(k), expands the inverse of the full mass matrix in a truncated series of k terms and thereby changes both the noise and the dynamics of the update.

The source of this task revises FMPM(k) by recasting the expansion as an incremental loop, in which every increment is the difference between the velocities of successive orders, so that lumped-mass features such as grid velocity conditions can be imposed on each increment instead of conflicting with the expansion; it adds a blend with the lumped mass matrix and with the FMPM(2) matrix, a dynamic order through an early exit, and it measures the temporal stability of the schemes by trial simulations of a freely vibrating bar, finding a Courant limit that falls with the order and plateaus at high order, limits raised by blending, and large energy dissipation for the lumped-mass PIC scheme.

Whether an explicit update is stable is a property of the linearised single-step operator: its eigenvalues decide whether any mode grows, and the Courant number at which an eigenvalue first leaves the unit circle is the linear stability limit. For the lumped-mass update such an analysis exists and gives a conservative band; for FMPM(k) the source attributes the loss of stability to a change of spectral properties but does not perform the analysis. The linear limit of a scheme, its dependence on the order of the expansion, on blending and on where the particles sit inside their cells, and the energy each scheme keeps over many periods are then questions about the spectrum of that operator rather than about trial simulations.

## Problem

The material point method (MPM) carries a body on particles and solves its equations of motion on a
background grid; each time step extrapolates particle momenta to the grid, updates the grid momenta
with the internal forces, converts them to grid velocities and maps the result back to the particles.
Standard MPM divides grid momenta by a lumped mass matrix. The source of this task revises its
approximate full mass matrix method, FMPM(k), which expands the inverse of the full mass matrix in
a truncated series of k terms: the expansion is recast as an incremental loop in which every increment
is the difference between the velocities of successive orders, so that lumped-mass features such as grid
velocity conditions can be imposed on each increment; the loop is given a blend with the lumped mass
matrix and with the FMPM(2) matrix, and an early exit that makes the order dynamic. The source
then measures the temporal stability of FMPM(k) by trial simulations of a freely vibrating bar,
searching the Courant number at which each scheme fails, and reports how that limit falls with the
order, how blending changes it and how much energy each scheme loses over five vibration periods.
It attributes the loss of stability at high order to a change in the spectral properties of the update,
cites a single-step spectral analysis of the lumped-mass update for the conservative Courant band of
standard MPM, and does not perform that analysis for FMPM(k): the linear stability limit of the
FMPM(k) update, its dependence on order, blending and particle placement, and what it says about
the conditioning of the full mass matrix, are the questions of this task.

Your task is to implement the revised loop with its generalised blend, write one time step of the
source's vibrating bar as a linear operator on the particle velocities and strains at frozen particle
positions, compute the spectral radius of that operator and the Courant number at which it first
exceeds one for the lumped-mass FLIP update, for FMPM(k) at several orders and blends (at the reference
order and at a high order), and for the closed-form limit of the loop, repeat the limits at a second particle placement, evaluate the energy
each scheme retains over five vibration periods, and report the linear stability limit of FMPM(4) on
the source's bar at its quarter-point particle placement.

## Conventions that fix every number

A one-dimensional uniform background grid with nodes x_i = i dx, i = 0..n_nodes - 1, carries N
material points at positions X_p with positive masses M_p and undeformed volumes Omega_p. The
particle-to-node interpolation uses linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it. The
lumped nodal masses are m_i = sum_p M_p S_pi; a node with m_i = 0 is inactive and carries nothing;
the lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active
nodes (zero rows at the inactive nodes), so that S+ S is n x n; the full mass matrix is
m~ = S^T M S = m (I - A) with A = I - S+ S. Where an operator array stores S+, it stores its
transpose, an N x n block indexed like S.

The revised FMPM(k) loop of the source: the grid velocity v+(k) is the sum of the increments
Delta v(l), l = 1..k, of the source's incremental form of the truncated expansion of the full mass
matrix inverse, with the source's treatment of the grid velocity conditions on the increments, its
generalised blend at blend fraction alpha_blend with the lumped mass matrix (blend period 1) or
with the FMPM(2) matrix (blend period 2), and its early exit on the Euclidean norm of an increment;
FMPM(1) is the lumped-mass velocity. The increment recursion, the placement of the velocity
condition within it, the blend rule for each increment and the exit rule are those of the source
and are not restated here. The source's original, non-incremental expansion must agree with the
loop when no node is controlled and no blend is applied.

Particle updates: the source's PIC-style FMPM(k) update of the particle velocities and positions
with its time-integration parameter alpha, its effective acceleration and its Lagrangian velocity
mismatch, and the source's FLIP alternative, all as the source defines them.

The loop as an operator: K(k) is the n x n matrix with v+(k) = K(k) p+ for the constrained, blended
loop, and K_inf is the closed-form limit of the same recursion continued to infinite order, which
exists when that recursion converges and equals the full mass matrix inverse on the active nodes
when no node is controlled. The convergence rate of the loop is the spectral rate per increment of
its blended recursion (the spectral radius of the map over one blend period, to the power 1/m for
blend period m), and the truncation error of order k is the 2-norm (largest singular value) of
K(k) - K_inf.

The source's vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx
for a placement fraction 0 < s < 0.5 (s = 0.25 puts them at the quarter points), volumes dx/2 and
masses rho dx/2, a linear-elastic small-strain material sigma = E eps with wave speed
v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and the loop
applies the source's velocity-condition treatment at node 0), the right end free, initial particle
velocities v0 sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, and time step dt = C dx/v_wave for the
Courant number C.

One USL time step of the bar, linearised at frozen particle positions, acts on the state
z = [V; eps] (particle velocities, then particle strains): momenta p = S^T M V, internal forces
f = -G^T Omega E eps with G the gradient matrix, p+ = p + f dt with p+_0 = 0, grid velocities v+
from the scheme (FLIP: m^-1 p+; FMPM: the loop to order k with the given blend; EXACT: the
closed-form limit of the loop), particle velocities from the source's PIC-style update for FMPM and
EXACT and from its FLIP update for FLIP, the reaction included in the grid force, and strains
eps(n+1) = eps + dt G v+ with the same grid velocities.
The amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin
of a scheme is the smallest Courant number at which the spectral radius of G(C) exceeds 1 + tol:
scan C = dC, 2 dC, ... up to C_max, take the first scan point over the threshold and the point
before it as the bracket, and locate the crossing by bisection to 1e-10. The energy retention after P
periods is (kinetic + strain energy)/(initial kinetic energy) after round(P T/dt) applications of
G(C) to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain energy
sum_p Omega_p E eps_p^2/2; it is defined only for a scheme that is linearly stable at that C.

Design settings: L_r = 40, dx = 1, s = 0.25, s_clustered = 0.4, E = 2, rho = 0.5, v0 = 0.16,
k_ref = 4, k_high = 40, alpha_blend = 0.8, C_ref = 0.5 (the Courant number of the energy
comparison), C_ret = 0.45 (the Courant number of the tabulated retentions), periods = 5, tol = 1e-8,
C_max = 1.5, dC = 0.01.

## Grid operators and the loop

Build the shape function matrix, the lumped masses and the lumped reverse map; advance one
increment of the loop with the blend and the per-increment conditions; run the loop to order k with
the generalised blend, the early exit and the achieved order; reproduce the original expansion; apply
the particle update with its effective acceleration, Lagrangian mismatch and FLIP alternative.

## The loop as an operator and the bar

Build K(k), its closed-form limit K_inf, the convergence rate and the truncation error. Assemble the
amplification matrix of the bar for FLIP, FMPM(k) with a blend, and EXACT; find the linear stability
limit by the scan-and-bisection rule; evaluate the energy retention.

## Audit

Run the complete chain and assemble the audit for the ten schemes FLIP, FMPM(1), FMPM(2),
FMPM(k_ref), FMPM(k_high), EXACT, FMPM(k_ref) blended with the lumped matrix (alpha_blend,
period 1), FMPM(k_ref) blended with the FMPM(2) matrix (alpha_blend, period 2), and FMPM(k_high)
blended in the same two ways: one row per scheme with the columns [scheme code (1 FLIP, 2 FMPM,
3 EXACT), k (1 in the FLIP and EXACT rows), alpha_blend (1 in the unblended rows), blend period (1 in
the unblended rows), C_lin at s, C_lin at s_clustered, energy retention over the given periods at
C_ret and placement s, convergence rate at s (0 for FLIP), truncation error at s (0 for FLIP and
EXACT), number of steps of the retention run, 0, 0], preceded by two head rows: row 0 = [C_lin of
FMPM(k_ref) at s, C_lin of FLIP at s, C_lin of FMPM(1) at s, C_lin of EXACT at s, C_lin of EXACT at
s_clustered, C_lin of FLIP at s_clustered, energy retention at C_ref of FMPM(1), of FMPM(k_ref), of
the lumped-matrix blend and of the FMPM(2) blend at k_ref, the spectral radius of T for the unblended
loop at s and at s_clustered]; row 1 = [N, n_nodes, v_wave, T, number of steps of the retention runs
at C_ret (the count tabulated in the scheme rows), C_lin of FMPM(2) at s, C_lin of FMPM(k_high) at
s, C_lin of the lumped-matrix blend at k_ref at s, C_lin of the FMPM(2) blend at k_ref at s,
truncation error of order k_ref at s and at s_clustered, largest absolute entry of the difference between
the operators of the source's original expansion and of the revised loop at order k_ref on the
design bar without controlled nodes and without blend (a consistency check, zero to rounding)].

## What to report

Report, as the final answer, the linear stability limit C_lin of FMPM(4) on the source's bar at
the design settings (placement s = 0.25), to six significant figures.

In the reasoning, report compactly, as a short list of labelled values, the linear limits at
s = 0.25 of FLIP, FMPM(1), FMPM(2), FMPM(40) and EXACT, of FMPM(4) blended with the lumped matrix
and with the FMPM(2) matrix, and of FMPM(40) blended in the same two ways; the limits of FLIP,
FMPM(4) and EXACT at s_clustered = 0.4; the energy retained after five periods at C_ref = 0.5 by
FMPM(1), FMPM(4) and the two order-4 blends; and the convergence rate of the unblended loop at both
placements. Give every limit to six significant figures. Explain, in a sentence or two each, why
the linear limit falls with the order and what sets the limit of the exact full mass matrix
inverse; why clustering the two particles of a cell collapses the exact limit while the
finite-order limits stay above it; and why FMPM(1) and the lumped-matrix blend lose energy while the FMPM(2) blend keeps it.
Report, with a citation, the source's measured stability limits for FMPM(4) and for its highest
orders, its measured limits for the two blends at the design blend fraction, its energy-loss figure
for FMPM(1) and its loss bound for the orders above one, its statement on the conditioning of the
full mass matrix at high order, and the conservative Courant band it cites for the lumped-mass
update; set each against the corresponding linear result.

These values, explanations and source comparisons are the reported result. A compact list of them
fits within the length the format note asks for and is not the per-iteration or per-candidate
output that note excludes.

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

fmpm_grid_operators

Goal
----
Builds the linear shape function matrix S (N x n_nodes), the lumped nodal masses m and the lumped reverse map S+ (n_nodes x N) of a particle set on the uniform grid, stacked as the rows [S; (S+)^T; m] of one (2N + 1) x n_nodes array, the transpose of S+ being indexed like S; nodes carrying no mass are inactive.

```python
def fmpm_grid_operators(Xp: "np.ndarray", Mp: "np.ndarray", n_nodes: int, dx: float) -> "np.ndarray":
    """Builds the linear shape function matrix S (N x n_nodes), the lumped nodal masses m and the lumped reverse map S+
    (n_nodes x N) of a particle set on the uniform grid, stacked as the rows [S; (S+)^T; m] of one (2N + 1) x n_nodes
    array, the transpose of S+ being indexed like S; nodes carrying no mass are inactive.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        Xp: array-like of shape (N,), particle positions inside the grid (0 <= X_p <= (n_nodes - 1) dx).
        Mp: array-like of shape (N,), positive particle masses.
        n_nodes: int, the number of grid nodes (at least 2) at x_i = i dx.
        dx: float, the positive cell size.

    Returns:
        A float64 array of shape (2N + 1, n_nodes): rows 0..N-1 hold S, rows N..2N-1 hold (S+)^T (entry [N + p, i] =
        (S+)_ip = M_p S_pi/m_i, zero at an inactive node) and row 2N holds m.

    Raises:
        ValueError: for empty or mismatched particle arrays, n_nodes < 2, a non-positive dx, a non-finite entry, a
        non-positive mass or a particle outside the grid.
    """
    return None
```

### Step 2

fmpm_velocity_increment

Goal
----
Advances one increment of the source's revised recursion: from the increment of order l - 1 it forms the increment of order l, scaled by alpha_blend, with the inactive nodes and then the controlled nodes of the result zeroed.

```python
def fmpm_velocity_increment(ops: "np.ndarray", N: int, dv_prev: "np.ndarray", bc_nodes: list, alpha_blend: float) -> "np.ndarray":
    """Advances one increment of the source's revised recursion: from the increment of order l - 1 it forms the increment
    of order l, scaled by alpha_blend, with the inactive nodes and then the controlled nodes of the result zeroed.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        dv_prev: array-like of shape (n,), the previous velocity increment Delta v(l - 1).
        bc_nodes: sequence of int node indices carrying a zero-velocity condition, or None.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).

    Returns:
        A float64 array of shape (n,), the next velocity increment.

    Raises:
        ValueError: for an ops array of the wrong shape or with non-finite entries, a dv_prev of the wrong length,
        alpha_blend outside (0, 1] or a controlled node index outside the grid.
    """
    return None
```

### Step 3

fmpm_loop

Goal
----
Runs the source's full FMPM loop to order k with the generalised blend and returns the FMPM(k) grid velocities, the final increment, and the order actually achieved together with a flag recording whether the loop exited early because the Euclidean norm of an increment fell below tol.

```python
def fmpm_loop(ops: "np.ndarray", N: int, p_plus: "np.ndarray", k: int, bc_nodes: list, alpha_blend: float, blend_period: int, tol: float) -> "np.ndarray":
    """Runs the source's full FMPM loop to order k with the generalised blend and returns the FMPM(k) grid velocities,
    the final increment, and the order actually achieved together with a flag recording whether the loop exited early
    because the Euclidean norm of an increment fell below tol.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        p_plus: array-like of shape (n,), the updated nodal momenta, already carrying the lumped-mass reaction of the
            velocity conditions.
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        bc_nodes: sequence of int node indices carrying a zero-velocity condition, or None.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.
        tol: float, the non-negative threshold on the Euclidean norm of an increment below which the loop exits early.

    Returns:
        A float64 array of shape (3, n): row 0 the FMPM(k) grid velocities, row 1 the final increment, row 2 the
        achieved order in entry 0 and the early-convergence flag (1 or 0) in entry 1 with the remaining entries zero.

    Raises:
        ValueError: for a malformed ops array, a p_plus of the wrong length, k < 1, a negative tol, alpha_blend
        outside (0, 1], blend_period < 1 or a controlled node index outside the grid.
    """
    return None
```

### Step 4

fmpm_prior_series

Goal
----
Reproduces the source's original, superseded expansion of the FMPM(k) velocities, the alternating sum sum_l (-1)^(l+1) v*_l of order k on the active nodes (no controlled nodes, no blend), returning the assembled velocities and the final term v*_k itself (unsigned, before the alternating sign (-1)^(k+1) of the sum is applied).

```python
def fmpm_prior_series(ops: "np.ndarray", N: int, p_plus: "np.ndarray", k: int) -> "np.ndarray":
    """Reproduces the source's original, superseded expansion of the FMPM(k) velocities, the alternating sum sum_l
    (-1)^(l+1) v*_l of order k on the active nodes (no controlled nodes, no blend), returning the assembled velocities
    and the final term v*_k itself (unsigned, before the alternating sign (-1)^(k+1) of the sum is applied).

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        p_plus: array-like of shape (n,), the updated nodal momenta, already carrying the lumped-mass reaction of the
            velocity conditions.
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.

    Returns:
        A float64 array of shape (2, n): row 0 the assembled velocities of the original expansion, row 1 the final
        term v*_k itself, without the sign (-1)^(k+1) it carries in the sum (so row 1 equals row 0 for k = 1).

    Raises:
        ValueError: for a malformed ops array, a p_plus of the wrong length or k < 1.
    """
    return None
```

### Step 5

fmpm_particle_update

Goal
----
Applies the source's PIC-style FMPM(k) particle velocity and position updates with time-integration parameter alpha and also reports the effective acceleration, the mismatch between the two Lagrangian velocities implied by the position and velocity updates, and the velocity a FLIP update would have produced.

```python
def fmpm_particle_update(ops: "np.ndarray", N: int, v_plus: "np.ndarray", Vn: "np.ndarray", Xn: "np.ndarray", f_grid: "np.ndarray", dt: float, alpha: float) -> "np.ndarray":
    """Applies the source's PIC-style FMPM(k) particle velocity and position updates with time-integration parameter
    alpha and also reports the effective acceleration, the mismatch between the two Lagrangian velocities implied by
    the position and velocity updates, and the velocity a FLIP update would have produced.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        v_plus: array-like of shape (n,), the grid velocities v+(k).
        Vn: array-like of shape (N,), the current particle velocities.
        Xn: array-like of shape (N,), the current particle positions.
        f_grid: array-like of shape (n,), the nodal forces.
        dt: float, the positive time step.
        alpha: float, the time-integration parameter of the position update.

    Returns:
        A float64 array of shape (5, N): rows hold the updated particle velocities, the updated particle positions,
        the effective acceleration, the Lagrangian velocity mismatch and the FLIP velocity.

    Raises:
        ValueError: for a malformed ops array, arrays of the wrong sizes or with non-finite entries, a non-positive dt
        or a non-finite alpha.
    """
    return None
```

### Step 6

fmpm_inverse_operator

Goal
----
Builds the n x n matrix K(k) that the constrained, blended FMPM loop applies to the momenta, v+(k) = K(k) p+ (the revised loop driven by one unit momentum per column, without early exit), the closed-form limit K_inf of the same recursion continued to infinite order (so K(k) and K_inf have zero rows at the controlled nodes), and the diagnostics of the series: the spectral radius of the unblended increment map T, the convergence rate per increment of the blended recursion, the 2-norm and the largest entry of K(k) - K_inf, and the number of active nodes.

```python
def fmpm_inverse_operator(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    """Builds the n x n matrix K(k) that the constrained, blended FMPM loop applies to the momenta, v+(k) = K(k) p+ (the
    revised loop driven by one unit momentum per column, without early exit), the closed-form limit K_inf of the same
    recursion continued to infinite order (so K(k) and K_inf have zero rows at the controlled nodes), and the
    diagnostics of the series: the spectral radius of the unblended increment map T, the convergence rate per
    increment of the blended recursion, the 2-norm and the largest entry of K(k) - K_inf, and the number of active
    nodes.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        ops: array of shape (2N + 1, n), the stacked rows [S; (S+)^T; m] returned by fmpm_grid_operators.
        N: int, the particle count that splits the rows of ops.
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        bc_nodes: sequence of int node indices carrying a zero-velocity condition, or None.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.

    Returns:
        A float64 array of shape (2n + 1, n): rows 0..n-1 hold K(k), rows n..2n-1 hold K_inf, and row 2n holds
        [spectral radius of T, convergence rate per increment, 2-norm (largest singular value) of K(k) - K_inf,
        largest absolute entry of K(k) - K_inf, number of active nodes] followed by zeros.

    Raises:
        ValueError: for a malformed ops array, k < 1, alpha_blend outside (0, 1], blend_period < 1, a controlled node
        index outside the grid, or a blended recursion that does not converge (spectral radius of its map over one
        blend period at or above 1), so that the series has no limit.
    """
    return None
```

### Step 7

fmpm_bar_operator

Goal
----
Assembles the (2N x 2N) single-step amplification matrix G(C) of one USL time step of the source's vibrating bar linearised at frozen particle positions, for the FLIP, FMPM(k) (with the given blend) or EXACT grid-velocity scheme, acting on the state [particle velocities; particle strains]; the particle-velocity block follows the scheme's particle update (PIC-style for FMPM and EXACT; FLIP with the reaction included in the grid force).

```python
def fmpm_bar_operator(L_r: float, dx: float, s: float, E: float, rho: float, C: float, scheme: str, k: int, alpha_blend: float, blend_period: int) -> "np.ndarray":
    """Assembles the (2N x 2N) single-step amplification matrix G(C) of one USL time step of the source's vibrating bar
    linearised at frozen particle positions, for the FLIP, FMPM(k) (with the given blend) or EXACT grid-velocity
    scheme, acting on the state [particle velocities; particle strains]; the particle-velocity block follows the
    scheme's particle update (PIC-style for FMPM and EXACT; FLIP with the reaction included in the grid force).

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        L_r: float, the bar length, a whole number of cells.
        dx: float, the positive cell size.
        s: float, the placement fraction of the two particles of each cell, at (i + s) dx and (i + 1 - s) dx, with 0 <
            s < 0.5.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        C: float, the positive Courant number, dt = C dx/v_wave.
        scheme: str, one of 'FLIP', 'FMPM' or 'EXACT' (the closed-form limit of the loop).
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.

    Returns:
        A float64 array of shape (2N, 2N), the amplification matrix on the state [V; eps].

    Raises:
        ValueError: for a bar length that is not a whole number of cells, a non-positive dx, E, rho or C, a placement
        fraction outside (0, 0.5), an unknown scheme, k < 1, alpha_blend outside (0, 1] or blend_period < 1.
    """
    return None
```

### Step 8

fmpm_stability_limit

Goal
----
Finds the linear stability limit C_lin of a scheme on the vibrating bar: the smallest Courant number at which the spectral radius of the amplification matrix exceeds 1 + tol, located by scanning C = dC, 2 dC, ... up to C_max for the first point over the threshold and bisecting the bracketing interval to 1e-10.

```python
def fmpm_stability_limit(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, tol: float, C_max: float, dC: float) -> float:
    """Finds the linear stability limit C_lin of a scheme on the vibrating bar: the smallest Courant number at which the
    spectral radius of the amplification matrix exceeds 1 + tol, located by scanning C = dC, 2 dC, ... up to C_max for
    the first point over the threshold and bisecting the bracketing interval to 1e-10.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        L_r: float, the bar length, a whole number of cells.
        dx: float, the positive cell size.
        s: float, the placement fraction of the two particles of each cell, at (i + s) dx and (i + 1 - s) dx, with 0 <
            s < 0.5.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        scheme: str, one of 'FLIP', 'FMPM' or 'EXACT' (the closed-form limit of the loop).
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.
        tol: float, the non-negative excess over 1 that the spectral radius must exceed for the operator to count as
            unstable.
        C_max: float, the largest Courant number of the scan.
        dC: float, the positive scan step.

    Returns:
        A float, the linear stability limit C_lin.

    Raises:
        ValueError: for invalid bar or scheme arguments, a negative tol, a non-positive dC, C_max not above dC, an
        operator unstable at every scanned Courant number, or no loss of stability up to C_max.
    """
    return None
```

### Step 9

fmpm_energy_retention

Goal
----
Advances the linearised bar from its initial velocity profile v0 sin(pi X_p/(2 L_r)) through round(periods T/dt) applications of the amplification matrix at Courant number C and returns the retained total energy, the kinetic and the strain parts, each relative to the initial kinetic energy, and the number of steps taken.

```python
def fmpm_energy_retention(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, C: float, v0: float, periods: float) -> "np.ndarray":
    """Advances the linearised bar from its initial velocity profile v0 sin(pi X_p/(2 L_r)) through round(periods T/dt)
    applications of the amplification matrix at Courant number C and returns the retained total energy, the kinetic
    and the strain parts, each relative to the initial kinetic energy, and the number of steps taken.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        L_r: float, the bar length, a whole number of cells.
        dx: float, the positive cell size.
        s: float, the placement fraction of the two particles of each cell, at (i + s) dx and (i + 1 - s) dx, with 0 <
            s < 0.5.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        scheme: str, one of 'FLIP', 'FMPM' or 'EXACT' (the closed-form limit of the loop).
        k: int, the FMPM order (at least 1); FMPM(k) sums exactly k increments, so k = 1 gives the lumped-mass
            velocities.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        blend_period: int, the blend period m of the source's generalised blend (at least 1): the increment of order l
            >= 2 is scaled by alpha_blend when m = 1 or l mod m = 1; the seed is never scaled.
        C: float, the positive Courant number, dt = C dx/v_wave.
        v0: float, the non-zero amplitude of the initial velocity v0 sin(pi X_p/(2 L_r)).
        periods: float, the positive number of fundamental periods over which the state is advanced.

    Returns:
        A float64 array of shape (4,): [retained energy fraction, kinetic fraction, strain fraction, number of steps].

    Raises:
        ValueError: for invalid bar or scheme arguments, a zero or non-finite v0, a non-positive periods, or a
        configuration whose spectral radius exceeds 1 + 1e-8 so that its energy has no finite retention.
    """
    return None
```

### Step 10

fmpm_stability_audit

Goal
----
Runs the complete chain and assembles the stability audit of the vibrating bar for ten schemes, in this row order: FLIP; FMPM(1); FMPM(2); FMPM(k_ref); FMPM(k_high); EXACT; FMPM(k_ref) blended with the lumped matrix (alpha_blend, period 1); FMPM(k_ref) blended with the FMPM(2) matrix (alpha_blend, period 2); FMPM(k_high) blended with the lumped matrix (alpha_blend, period 1); FMPM(k_high) blended with the FMPM(2) matrix (alpha_blend, period 2). Each scheme row holds the 12 columns [scheme code (1 FLIP, 2 FMPM, 3 EXACT), k (1 in the FLIP and EXACT rows), alpha_blend (1 in the unblended rows), blend period (1 in the unblended rows), C_lin at s, C_lin at s_clustered, energy retention after the given periods at C_ret and placement s, convergence rate of the loop at s (0 for FLIP), 2-norm of K(k) - K_inf at s (0 for FLIP and EXACT), number of steps of the retention run, 0, 0]. Head row 0 holds [C_lin of FMPM(k_ref) at s, C_lin of FLIP at s, C_lin of FMPM(1) at s, C_lin of EXACT at s, C_lin of EXACT at s_clustered, C_lin of FLIP at s_clustered, energy retention at C_ref of FMPM(1), of FMPM(k_ref), of the lumped-matrix blend and of the FMPM(2) blend, the spectral radius of T for the unblended loop at s and at s_clustered]; head row 1 holds [N, n_nodes, v_wave, T, number of steps of the retention runs at C_ret (the count tabulated in the scheme rows), C_lin of FMPM(2) at s, C_lin of FMPM(k_high) at s, C_lin of the lumped-matrix blend at s, C_lin of the FMPM(2) blend at s, 2-norm of K(k_ref) - K_inf at s and at s_clustered, and the largest absolute entry of the difference between the operators of the source's original expansion and of the revised loop at order k_ref on the design bar without controlled nodes and without blend (a consistency check, zero to rounding)]. The scheme rows follow.

```python
def fmpm_stability_audit(L_r: float, dx: float, s: float, s_clustered: float, E: float, rho: float, v0: float, k_ref: int, k_high: int, alpha_blend: float, C_ref: float, C_ret: float, periods: float, tol: float, C_max: float, dC: float) -> "np.ndarray":
    """Runs the complete chain and assembles the stability audit of the vibrating bar for ten schemes, in this row order:
    FLIP; FMPM(1); FMPM(2); FMPM(k_ref); FMPM(k_high); EXACT; FMPM(k_ref) blended with the lumped matrix (alpha_blend,
    period 1); FMPM(k_ref) blended with the FMPM(2) matrix (alpha_blend, period 2); FMPM(k_high) blended with the
    lumped matrix (alpha_blend, period 1); FMPM(k_high) blended with the FMPM(2) matrix (alpha_blend, period 2). Each
    scheme row holds the 12 columns [scheme code (1 FLIP, 2 FMPM, 3 EXACT), k (1 in the FLIP and EXACT rows),
    alpha_blend (1 in the unblended rows), blend period (1 in the unblended rows), C_lin at s, C_lin at s_clustered,
    energy retention after the given periods at C_ret and placement s, convergence rate of the loop at s (0 for FLIP),
    2-norm of K(k) - K_inf at s (0 for FLIP and EXACT), number of steps of the retention run, 0, 0]. Head row 0 holds
    [C_lin of FMPM(k_ref) at s, C_lin of FLIP at s, C_lin of FMPM(1) at s, C_lin of EXACT at s, C_lin of EXACT at
    s_clustered, C_lin of FLIP at s_clustered, energy retention at C_ref of FMPM(1), of FMPM(k_ref), of the
    lumped-matrix blend and of the FMPM(2) blend, the spectral radius of T for the unblended loop at s and at
    s_clustered]; head row 1 holds [N, n_nodes, v_wave, T, number of steps of the retention runs at C_ret (the count
    tabulated in the scheme rows), C_lin of FMPM(2) at s, C_lin of FMPM(k_high) at s, C_lin of the lumped-matrix blend
    at s, C_lin of the FMPM(2) blend at s, 2-norm of K(k_ref) - K_inf at s and at s_clustered, and the largest
    absolute entry of the difference between the operators of the source's original expansion and of the revised loop
    at order k_ref on the design bar without controlled nodes and without blend (a consistency check, zero to
    rounding)]. The scheme rows follow.

    Model and conventions (the source's revised FMPM(k) and its vibrating bar): a one-dimensional uniform background
    grid with nodes x_i = i dx, i = 0..n_nodes - 1; linear (tent) shape functions S_pi = max(0, 1 - |X_p - x_i|/dx)
    with gradients dS_pi/dx = -1/dx on the node below a particle and +1/dx on the node above it; N material points
    with positions X_p, positive masses M_p and undeformed volumes Omega_p; S is the N x n matrix of the S_pi (n =
    n_nodes); lumped nodal masses m_i = sum_p M_p S_pi, a node with m_i = 0 being inactive and carrying nothing; the
    lumped reverse map S+ is the n x N operator with entries (S+)_ip = M_p S_pi/m_i on the active nodes and zero rows
    at the inactive nodes, so that S+ S is n x n, the full mass matrix is m~ = S^T M S = m (I - A) and A = I - S+ S;
    the stacked operator array stores the transpose of S+, an N x n block indexed like S. The revised FMPM loop: the
    FMPM(k) grid velocities v+(k) are the sum of k increments Delta v(l), l = 1..k, the increment of order l being the
    difference between the velocities of the source's truncated expansions of the full mass matrix inverse at orders l
    and l - 1, seeded by the lumped-mass velocity Delta v(1) = m^-1 p+ on the active nodes; every increment, the seed
    included, is zeroed on the inactive nodes and on the controlled (zero-velocity) nodes, so a controlled node
    carries its prescribed zero velocity whatever momentum is supplied there; the generalised blend of the source
    scales the increment of order l >= 2 by alpha_blend when blend_period = 1 or l mod blend_period = 1 and by 1
    otherwise, and never scales the seed (blend_period = 1 blends with the lumped mass matrix, blend_period = 2 with
    the FMPM(2) matrix); the early exit stops the loop once the Euclidean norm of the increment just added falls below
    tol (tested after each increment beyond the seed). The source's original, non-incremental expansion writes the
    same order-k velocities as an alternating sum sum_l (-1)^(l+1) v*_l over l = 1..k whose terms are multiples of (S+
    S)^(l-1) m^-1 p+ by coefficients that depend on k and l; it must agree with the unconstrained, unblended loop.
    Particle updates: the source's FMPM(k) update is PIC-style, replacing every particle velocity by the extrapolated
    grid velocity, V(n+1) = S v+(k), and advancing the positions over the step with the velocity alpha V(n+1) + (1 -
    alpha) V(n) (time-integration parameter alpha); its effective acceleration is the change of the particle velocity
    over the step per unit time, and its Lagrangian velocity mismatch is the difference between the velocity that
    advances the positions and the mean of V(n) and V(n+1); the FLIP update adds the extrapolated lumped-mass
    acceleration increment, V(n+1) = V(n) + S m^-1 f dt. The loop as an operator: K(k) is the n x n matrix with v+(k)
    = K(k) p+ for the constrained, blended loop without early exit; K_inf is the closed-form limit of the same
    recursion continued to infinite order, which exists when the recursion converges and equals the full mass matrix
    inverse on the active nodes when no node is controlled (K(k) and K_inf have zero rows at the controlled nodes); T
    denotes the unblended increment map, the linear map that carries one increment to the next with alpha_blend = 1,
    the zeroing of the inactive and controlled nodes included; the convergence rate of the loop is the spectral rate
    per increment of its blended recursion, the spectral radius of the map that carries the increments over one blend
    period taken to the power 1/blend_period; the truncation error of order k is the 2-norm (largest singular value)
    of K(k) - K_inf. The vibrating bar: L_r = n_cells dx, two particles per cell at (i + s) dx and (i + 1 - s) dx for
    a placement fraction 0 < s < 0.5, volumes dx/2 and masses rho dx/2, linear-elastic small-strain material sigma = E
    eps with wave speed v_wave = sqrt(E/rho), the left node 0 held at zero velocity (the reaction sets p+_0 = 0 and
    the loop zeroes the controlled entry of every increment), the right end free, initial particle velocities v0
    sin(pi X_p/(2 L_r)), fundamental period T = 4 L_r/v_wave, time step dt = C dx/v_wave for the Courant number C. One
    USL time step linearised at frozen particle positions acts on the state z = [V; eps] (particle velocities, then
    particle strains): momenta p = S^T M V, internal forces f = -G^T Omega E eps with G the N x n matrix of the shape
    function gradients, p+ = p + f dt with the reaction p+_0 = 0, grid velocities v+ from the scheme (FLIP: m^-1 p+;
    FMPM: the loop to order k with the given blend; EXACT: the closed-form limit of the loop), particle velocities
    from the scheme's update (the PIC-style update for FMPM and EXACT; the FLIP update, whose grid force includes the
    reaction at the controlled node), strains eps(n+1) = eps + dt G v+ with the same grid velocities. The
    amplification matrix G(C) is the (2N x 2N) matrix of that map. The linear stability limit C_lin is the smallest
    Courant number at which the spectral radius of G(C) exceeds 1 + tol: C is scanned at dC, 2 dC, ... up to C_max,
    the first scan point over the threshold and the point before it bracket the crossing, and the crossing is located
    by bisection to 1e-10. The energy retention after P periods is (kinetic + strain energy)/(initial kinetic energy)
    after round(P T/dt) applications of G to the initial state, with kinetic energy sum_p M_p V_p^2/2 and strain
    energy sum_p Omega_p E eps_p^2/2.

    Args:
        L_r: float, the bar length, a whole number of cells.
        dx: float, the positive cell size.
        s: float, the placement fraction of the two particles of each cell, at (i + s) dx and (i + 1 - s) dx, with 0 <
            s < 0.5.
        s_clustered: float, a second placement fraction (0 < s_clustered < 0.5) at which the limits are repeated.
        E: float, the positive elastic modulus.
        rho: float, the positive density.
        v0: float, the non-zero amplitude of the initial velocity v0 sin(pi X_p/(2 L_r)).
        k_ref: int, the reference order (at least 1) whose limit is the head-row answer.
        k_high: int, a high order (at least 1) audited against the closed-form limit; the blends are audited at k_ref
            and at k_high.
        alpha_blend: float in (0, 1], the blend fraction applied to the increment (1 for no blending).
        C_ref: float, the positive Courant number of the source's dissipation comparison, at which the head row
            reports energy retention.
        C_ret: float, the positive Courant number at which every audited scheme's energy retention is tabulated.
        periods: float, the positive number of fundamental periods over which the state is advanced.
        tol: float, the non-negative excess over 1 that the spectral radius must exceed for the operator to count as
            unstable.
        C_max: float, the largest Courant number of the scan.
        dC: float, the positive scan step.

    Returns:
        A float64 array of shape (12, 12): two head rows followed by one row per scheme.

    Raises:
        ValueError: for invalid bar, order, blend or Courant arguments of the underlying steps, a retention
        configuration beyond its linear limit, or a scan that finds no loss of stability.
    """
    return None
```
