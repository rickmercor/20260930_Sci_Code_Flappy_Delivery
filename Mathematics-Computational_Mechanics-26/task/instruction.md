# Mathematics-Computational_Mechanics-26

## Background

Rubber, polymer foams and soft biological tissue do not spring back instantly. Push on them and part of the shape change happens at once, while the rest catches up over seconds or minutes, and when you let go the recovery is just as sluggish. Engineers capture that lag by giving the material a second, hidden state alongside its visible shape: an internal variable that trails the deformation and relaxes towards it at a rate set by the material's viscosity. The visible shape has to satisfy force balance; the hidden state has to satisfy its own evolution law. The two are locked together, because the stress depends on both.

Turning that picture into a computer simulation forces a choice about *how* to solve the locked pair. The traditional route is nested. At every integration point of the mesh you first freeze the current shape, solve a little nonlinear problem for the hidden state until it is consistent, and only then go back and ask whether the whole structure is in equilibrium. It is a loop inside a loop: a small local solve repeated at every point, every iteration, every time step. It works, and it has been the standard for decades, but the inner solves are expensive and they can fail on large load steps.

The alternative is to stop treating the hidden state as something to be resolved separately. Write both conditions force balance and the evolution law as one big system, linearise the whole thing at once, and then eliminate the hidden state's increment algebraically from the linear system rather than by iterating on it. The elimination is a Schur complement, and it has a neat geometric reading: the evolution law defines a surface of admissible states, and the reduction restricts each Newton step to the directions tangent to that surface. That is the null-space interpretation.

The difference between the two routes is subtle but real, and it shows up in a single term. In the nested scheme the local equation has already been driven to zero before the global step is taken, so it contributes nothing to the right-hand side. In the monolithic scheme it has not, and the leftover internal residual feeds back into the displacement correction. Away from convergence the two schemes therefore take genuinely different steps, even though they arrive at the same answer in the end. This piece of research works out that reduction, shows it is the same object viewed two ways, and demonstrates that it converges on load steps where the nested scheme gives up.

## Problem

Finite-strain viscoelasticity couples the deformation to a strain-like internal variable whose evolution is a constitutive ordinary differential equation. Once the problem is discretised in time and space that evolution law becomes an additional set of algebraic equations, so the coupled Newton system acquires a naturally non-symmetric block tangent in the displacement and internal-variable increments. A recent contribution treats the discrete evolution equation as an internal constraint and eliminates its increment at the level of the linearised system instead of by a nested local solve, which changes the Newton correction whenever the internal residual has not yet vanished. Your task is to run that monolithic scheme on one small, fully specified plane-strain Cook's membrane problem and report a single number characterising its first Newton correction in the final time step.

Here is the exact setup to use:

- Geometry and mesh:  Cook's membrane with corners $P_1=(0,0)$, $P_2=(480,440)$, $P_3=(480,600)$, $P_4=(0,440)$, all in mm, discretised by $2 \times 2$ bilinear Q1 elements. Node $(i,j)$, $i,j=0,1,2$, carries the id $3j+i$ and sits at the bilinear image of $(\xi,\eta)=(i/2,\,j/2)$. Unit thickness, $2\times 2$ Gauss quadrature per element, quadrature points at $\pm 1/\sqrt{3}$ with unit weights.
- Kinematics: Plane strain: the in-plane $2\times 2$ deformation gradient is embedded in $3\times 3$ with a unit out-of-plane entry, so the three-dimensional constitutive relations apply unchanged.
- Free energy: $\Psi(\mathbf{C},\mathbf{C}_i)=\Psi_{\mathrm{eq}}(\mathbf{C})+\Psi_{\mathrm{neq}}(\mathbf{C}_e)$ with $\mathbf{C}_e=\mathbf{C}\,\mathbf{C}_i^{-1}$. Both parts take the Simo–Taylor form
  $$\Psi(\mathbf{A}) = \tfrac{m}{2}\left(\operatorname{tr}\mathbf{A} - 3 - 2\log J_{\mathbf{A}}\right) + \tfrac{l}{2}\left((\log J_{\mathbf{A}})^2 + (J_{\mathbf{A}}-1)^2\right),$$
  where $J_{\mathbf{A}}$ is the determinant of the deformation gradient belonging to $\mathbf{A}$. Constants: $\lambda = \lambda_{\mathrm{visc}} = 30000$, $\mu = \mu_{\mathrm{visc}} = 7500$.
- Viscosity: The isotropic fourth-order viscosity operator is built from the volumetric and deviatoric projectors with $V_{\mathrm{vol}} = 50000$ and $V_{\mathrm{dev}} = 10000$, using the volumetric–deviatoric split written down in the source paper's numerical-experiments section. The volumetric projector uses $d = 3$.
- Boundary conditions and load: The left edge $X_1 = 0$ is fully clamped. A dead traction $\Lambda(t)\,p\,\bar{\mathbf{T}}$ with $\bar{\mathbf{T}} = (-750,\,1000)$, $p = 2$ and $\Lambda(t) = t/T_{\mathrm{end}}$ acts on the whole right edge $X_1 = 480$ and is integrated over the reference edge. Body forces and inertia are absent, so the problem is quasi-static.
- Internal variable. $\mathbf{C}_i$ is stored at every quadrature point and initialised to the identity. A symmetric $3\times3$ tensor is packed as the 6-vector $(A_{11},A_{22},A_{33},A_{12},A_{13},A_{23})$ and unpacked by mirroring; the finite-difference perturbations below act on these packed entries.
- Time discretisation: The midpoint scheme of the source paper, over $T_{\mathrm{end}} = 10\,\mathrm{s}$ in five equal steps. The load factor is evaluated at the midpoint of each step.
- Newton iteration: Exactly eight monolithic Newton iterations per time step, with no stopping test. All four tangent blocks are obtained by central finite differences of the two residuals with respect to the packed unknowns, with step $10^{-6}$.
- Condensation: Eliminate the internal-variable increment at the level of the linearised system using the reduction the source paper proposes, and reconstruct it after the reduced solve.

March the five time steps and report $\lVert \Delta \mathbf{q}^{(1)} \rVert_2$, the Euclidean norm of the first condensed displacement correction of the fifth time step, taken over the free degrees of freedom. The answer is graded to an absolute tolerance of $10^{-3}$. In your reasoning, report the number of free displacement degrees of freedom, the norms of the momentum and evolution residuals at that first iteration, the norm of the correction that the classical nested right-hand side would have produced instead, and the norm of the final tip displacement at node 8.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the mesh coordinates, the assembled tangent blocks, the per-iteration residual tables, or the full internal-variable field.

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

Build the Cook's membrane mesh

Goal
----
Produce the nodal coordinates of an n x n bilinear quadrilateral mesh of Cook's membrane. The domain is the quadrilateral with corners (0,0), (480,440), (480,600) and (0,440), and the mesh is its bilinear image: node (i, j) sits at parameter values xi = i/n and eta = j/n, and carries the id j*(n+1)+i. Everything downstream indexes into this array, so the node ordering is part of the contract rather than an implementation detail. The clamped boundary is identified later by the nodes whose first coordinate vanishes, and the loaded edge by the last node in each row, both of which only work if the ids follow the stated ordering.

```python
import numpy as np


def build_cooks_mesh(n):
    """Nodal coordinates of an n x n bilinear (Q1) mesh of Cook's membrane.

    Cook's membrane has corners P1=(0,0), P2=(480,440), P3=(480,600),
    P4=(0,440), all in mm.  Node (i, j), i, j = 0..n, carries the id
    j*(n+1)+i and sits at the bilinear image of (xi, eta) = (i/n, j/n).

    Args:
        n (int): number of elements per side.

    Raises:
        ZeroDivisionError: if n == 0.

    Expected return:
        np.ndarray of shape (2*(n+1)**2,), the node coordinates flattened as
        [x_0, y_0, x_1, y_1, ...] in ascending node id.
    """
    return np.zeros(2 * (n + 1) ** 2)
```

### Step 2

Bilinear shape function gradients

Goal
----
Given the four node coordinates of one element and a point in the parent square, return the determinant of the isoparametric Jacobian together with the gradients of the four shape functions taken with respect to the reference coordinates. Node ordering is counter-clockwise, matching the parent corners (-1,-1), (1,-1), (1,1) and (-1,1), which is the same order the connectivity produced by the previous step uses. The determinant is returned alongside the gradients because both are needed at every quadrature point and computing them together avoids inverting the Jacobian twice.

```python
import numpy as np


def q1_shape_gradients(xe, xi, eta):
    """Reference-configuration gradients of the Q1 shape functions.

    Nodes are ordered counter-clockwise at the parent corners
    (-1,-1), (1,-1), (1,1), (-1,1).

    Args:
        xe: (4, 2) array of element node coordinates.
        xi, eta (float): parent coordinates in [-1, 1].

    Expected return:
        np.ndarray of shape (9,) packed as
        [detJ, dN1/dX, dN1/dY, dN2/dX, dN2/dY, dN3/dX, dN3/dY, dN4/dX, dN4/dY].
    """
    return np.zeros(9)
```

### Step 3

Constitutive response and viscous driving force

Goal
----
Evaluate the second Piola-Kirchhoff stress and the mixed driving force at one material point, given the right Cauchy-Green tensor and the current internal variable. The free energy splits into an equilibrium part evaluated at C and a non-equilibrium part evaluated at the elastic measure C_e = C C_i^-1; both parts use the same Simo-Taylor form and, in this task, the same pair of constants. The stress is the sum of the two contributions. The driving force is the non-equilibrium stress post-multiplied by C, and it is deliberately returned unsymmetrised.

```python
import numpy as np


def constitutive_response(C, Ci):
    """Second Piola-Kirchhoff stress and the mixed viscous driving force.

    The free energy splits as Psi = Psi_eq(C) + Psi_neq(Ce) with
    Ce = C Ci^{-1}, and both parts take the same Simo-Taylor form
    Psi(A) = m/2 (tr A - d - 2 log Jd) + l/2 ((log Jd)^2 + (Jd - 1)^2)
    with Jd the determinant of the corresponding deformation gradient,
    i.e. the square root of the determinant of the argument.  Material constants are
    lambda = lambda_visc = 30000, mu = mu_visc = 7500.

    Args:
        C: (3, 3) right Cauchy-Green tensor.
        Ci: (3, 3) internal (viscous) strain-like variable.

    Expected return:
        np.ndarray of shape (18,) packed as [S.ravel(), M.ravel()], where
        S = S_eq + S_neq and M is the mixed driving force conjugate to the
        viscous rate.  M is in general non-symmetric.
    """
    return np.zeros(18)
```

### Step 4

Viscous flow term

Goal
----
Apply the inverse viscosity operator to the transpose of the driving force and pre-multiply by twice the internal variable. This is the whole right-hand side of the internal evolution law at a single material point. The viscosity operator is isotropic and built from the volumetric and deviatoric projectors with separate coefficients, so inverting it is a matter of dividing each projection by its own coefficient rather than inverting any matrix. The volumetric projector uses a factor of one third, corresponding to three dimensions, even in the plane-strain setting.

```python
import numpy as np


def viscous_flow(M, Ci):
    """The viscous flow term appearing in the internal evolution law.

    The isotropic viscosity operator is built from the volumetric and
    deviatoric projectors with V_vol = 50000 and V_dev = 10000, using the
    volumetric-deviatoric split stated in the source paper.  The volumetric
    part uses d = 3.

    Args:
        M: (3, 3) mixed driving force.
        Ci: (3, 3) internal variable.

    Raises:
        ValueError: if M does not have shape (3, 3).

    Expected return:
        np.ndarray of shape (9,): the flow tensor flattened row-major.
    """
    return np.zeros(9)
```

### Step 5

Midpoint evolution residual

Goal
----
Form the discrete residual of the internal evolution law at one quadrature point over one time step. The internal variable is supplied at both ends of the step in packed six-component form; its rate is the backward difference quotient, and the constitutive quantities are evaluated at the arithmetic midpoint of the two endpoint values. The residual is the rate minus the flow term, projected onto symmetric tensors before being packed back into six components.

```python
import numpy as np


def evolution_residual(C_mid, ci_new, ci_old, h):
    """Midpoint-discrete evolution residual at one quadrature point.

    A symmetric 3x3 tensor is packed as the 6-vector
    (A11, A22, A33, A12, A13, A23) and unpacked by mirroring.
    The internal variable is evaluated at the arithmetic midpoint of its
    endpoint values; the rate is the backward difference quotient.  The
    residual is tested with symmetric test functions.

    Args:
        C_mid: (3, 3) midpoint right Cauchy-Green tensor.
        ci_new, ci_old: (6,) packed internal variable at t_{n+1} and t_n.
        h (float): time-step size.

    Raises:
        numpy.linalg.LinAlgError: if the midpoint internal variable is singular.

    Expected return:
        np.ndarray of shape (6,): the packed residual.
    """
    return np.zeros(6)
```

### Step 6

Assemble the global momentum residual

Goal
----
Loop the elements and their quadrature points, evaluate the midpoint deformation gradient from the averaged displacement field, obtain the stress from the constitutive step, and scatter the internal force contributions into a global vector; then subtract the external contribution of the dead traction on the right edge. The residual is returned for every degree of freedom, before any Dirichlet reduction, so that the caller decides which rows to keep. Inertia and body forces are absent, so what remains is the quasi-static balance.

```python
import numpy as np


def assemble_momentum_residual(n, u, u_n, ci, ci_n, lam):
    """Global midpoint momentum residual, internal minus external.

    Plane strain: the 2x2 in-plane deformation gradient is embedded in 3x3
    with a unit out-of-plane entry.  All constitutive quantities are taken
    at the midpoint configuration.  Quadrature is 2x2 Gauss per element and
    unit thickness.  The dead traction lam * (-750, 1000) acts on the whole
    right edge X1 = 480, integrated over the reference edge.  Inertia and
    body forces are absent.

    Args:
        n (int): elements per side.
        u, u_n: (2*(n+1)**2,) nodal displacements at t_{n+1} and t_n.
        ci, ci_n: (24*n**2,) packed quadrature-point internal variables.
        lam (float): current load factor.

    Expected return:
        np.ndarray of shape (2*(n+1)**2,), the residual for every degree of
        freedom, before any Dirichlet reduction.
    """
    return np.zeros(2 * (n + 1) ** 2)
```

### Step 7

Schur-condensed Newton correction

Goal
----
Given the four blocks of the monolithic tangent and the two residuals, eliminate the internal-variable increment at the level of the linearised system and return the displacement increment. The elimination requires one solve against the internal block with the coupling matrix and the internal residual as simultaneous right-hand sides, followed by one solve of the reduced system. The internal increment itself is recovered afterwards by the caller from the same two factors.

```python
import numpy as np


def condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC):
    """Condensed Newton correction of the monolithic block system.

    The monolithic system couples the displacement increment and the
    internal-variable increment through a generally non-symmetric block
    tangent.  The internal increment is eliminated at the level of the
    linearised system; the reduction keeps whatever the internal residual
    contributes to the reduced right-hand side, which is what distinguishes
    this scheme from the classical nested Gauss-point condensation.

    Args:
        Kqq: (nq, nq); KqC: (nq, nc); KCq: (nc, nq); KCC: (nc, nc).
        Rq: (nq,); RC: (nc,).

    Raises:
        numpy.linalg.LinAlgError: if KCC or the Schur complement is singular.

    Expected return:
        np.ndarray of shape (nq,): the displacement increment.
    """
    return np.zeros(np.asarray(Rq).shape[0])
```

### Step 8

Run the monolithic pipeline

Goal
----
Chain the seven earlier steps into the full calculation and return the requested scalar. Build the mesh and cache the shape-function data, initialise the internal variable to the identity at every quadrature point, and march the load in equal time steps. Within each step, run a fixed number of Newton iterations with no stopping test: assemble both residuals, build all four tangent blocks by central finite differences of those residuals with respect to the packed unknowns, take the condensed correction, and update both fields. Capture the norm of the first correction of the final step and return it. This is the orchestrator step: its reference implementation calls the earlier public functions by name rather than reproducing their contents.

```python
import numpy as np


def run_monolithic_pipeline(n, n_steps, p_mult, fd_step, newton_iters):
    """Chain the seven earlier steps and report the requested scalar.

    Builds the mesh, marches the load with the midpoint scheme, runs a fixed
    number of monolithic Newton iterations per step with the internal
    variable condensed out at the linearised level, and returns the
    Euclidean norm of the first condensed displacement correction of the
    final time step.

    Args:
        n (int): elements per side.
        n_steps (int): number of equal time steps over T_end = 10 s.
        p_mult (float): load multiplier.
        fd_step (float): central finite-difference step for the tangents.
        newton_iters (int): Newton iterations performed per time step.

    Raises:
        ZeroDivisionError: if n_steps == 0.

    Expected return:
        float: || dq^(1) || in the final time step.
    """
    return 0.0
```
