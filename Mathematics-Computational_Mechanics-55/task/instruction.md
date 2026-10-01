# Mathematics-Computational_Mechanics-55

## Background

Ferronematic order couples a magnetic unit direction to a nematic orientation with twice its angle. A nodal constraint can therefore be enforced by normalizing a magnetic tangent update and reconstructing the nematic nodal values, but the energy of their affine interpolants must still be evaluated separately. Splitting the two tangent fields introduces a nonlinear compatibility relation and auxiliary variables whose boundary behavior differs from that of the magnetic increments.

A decrease in energy does not describe how a finite relaxation trajectory responds to a perturbation of its boundary data. That response requires differentiating the coupled inner solves, the geometry rebuilt after each projection, and the initial energy used to normalize the decrease. Boundary directions are constant during a run while varying across the family of perturbed runs. Holding the executed stopping counts fixed defines a local directional response without claiming that an equilibrium derivative or a derivative across a stopping-count transition is the same observable.

## Problem

Consider the boundary-angle response of a two-dimensional ferronematic relaxation whose nodal physical state is $\Psi_a=(Q_cR(n_a),M_cn_a)$, with $|n_a|=1$, $R(n)=(2nn^{\mathsf T}-I)e_1$, and harmonic energy $E(\Psi_h)=\frac12\int_\Omega|\nabla\Psi_h|_F^2\,dx$ evaluated on the continuous piecewise affine interpolant of these four-component nodal values.
On the dimensionless unit square, number vertices by $a=4j+i$ with coordinates $(i/3,j/3)$ for $i,j=0,1,2,3$, and for each $i,j=0,1,2$ use the ordered triangles $(a,a+1,a+5)$ and $(a,a+5,a+4)$, holding the magnetic directions fixed during relaxation at vertices with $i$ or $j$ equal to $0$ or $3$.
For a dimensionless perturbation parameter $\alpha$, initialize $n_a^0(\alpha)=(\cos\theta_a(\alpha),\sin\theta_a(\alpha))$ with angles in radians given by $\theta_a(\alpha)=0.1+0.8x_a+0.6y_a+\delta_a+\alpha v_a$, where $\delta_a$ is zero except for the ordered values $(1.6,-1.3,1.1,-1.5)$ at vertices $(5,6,9,10)$, and $v_a=\cos(\pi x_a)\sin(\pi(y_a+1/4))$ at fixed vertices and zero elsewhere.
Take $Q_c=1.0013$ and $M_c=1.0025$ as exact dimensionless constants and perform exactly six projected tangent relaxation updates with counterclockwise unit tangents, using the split integrals of the squared gradients of the unnormalized magnetic and nematic tangent fields without a factor of one half.
Within each update use the alternating augmented-Lagrangian iteration for the exact compatibility between normalized magnetic and nematic tangent increments: linearize compatibility only in the magnetic minimization at its preceding inner iterate, retain its nonlinear form in the auxiliary minimization and dual ascent, use residual $p-\varphi(r)$ with positive multiplier pairing, scaling $D=I_{16}$, augmentation $\zeta=4$, and dual step $\rho=1$, eliminate only the fixed-node magnetic coefficients, and retain auxiliary coefficients and multipliers at every node.
Initialize the three inner state vectors at zero for the first outer update and carry all three terminal vectors to the next update, solve the linear systems directly, stop each inner loop at the first completed cycle whose all-node root-mean-square compatibility residual is at most $10^{-10}$ with a cap of $20000$ cycles, and after each inner loop normalize the magnetic tangent update and reconstruct the nematic directions.
Define $S(\alpha)=[E(\Psi_h^0(\alpha))-E(\Psi_h^6(\alpha))]/E(\Psi_h^0(\alpha))$ and evaluate its derivative at zero along the executed local iteration branch, holding the base execution's inner stopping counts fixed when differentiating; the boundary values remain fixed within each run but their derivatives with respect to $\alpha$ are prescribed by $v$.
Using binary64 arithmetic without randomization, compute the one finite scalar $\chi=S'(0)/S(0)$ to absolute error at most $10^{-7}$, treating a magnetic coefficient outside $(-1,1)$, nonconvergence, or a vanishing normalization as an error without clipping or substituting an equilibrium solution.
In the reasoning, justify the compatibility and projection variations and the distinction between fixed boundary values and fixed boundary sensitivities, and report $E^0$, $E^6$, their two directional derivatives, $S(0)$, and $S'(0)$ needed to determine $\chi$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_prepare_geometry_jet

Goal
----
Construct the value and directional variation of the nodal geometry for the family of angles $\theta+\alpha v$ on a fixed triangular mesh.

Use the continuous affine nodal basis and its dimensionless Dirichlet bilinear form without boundary elimination or lumping.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

```python
import numpy as np


def prepare_geometry_jet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    vertices : np.ndarray
        Finite dimensionless planar coordinates, shape (N, 2), N >= 3.
    triangles : np.ndarray
        Integer connectivity, shape (M, 3), M >= 1, using every vertex; each
        triangle has distinct zero-based indices and nonzero area, with either
        orientation.
    angles : np.ndarray
        Finite initial angles in radians, shape (N,), in vertex order.
    direction : np.ndarray
        Finite angle derivatives with respect to alpha, shape (N,), in vertex order.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N, N + 8) containing the
        stiffness and coupled nodal frames in layer zero and their directional
        derivatives in layer one.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 2

02_build_quadratic_jet

Goal
----
Determine the affine coefficient-space gradients and their directional variations for the magnetic and nematic unnormalized tangent energies.

At coefficient vector $z$, the respective nodal values are $M_c(n_a+z_at_a)$ and $Q_c(\nu_a+z_a\tau_a)$, interpolated affinely; each functional is the integral of its squared spatial gradient without a factor of one half.

Represent each gradient as $Hz+q$ and differentiate its coefficients with $z$ held fixed.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

```python
import numpy as np


def build_quadratic_jet(geometry: np.ndarray, Qc: float, Mc: float) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    geometry : np.ndarray
        Finite geometric jet, shape (2, N, N + 8), with the layout stated in the
        description; magnetic rows are unit, their derivatives tangent, frames
        consistent, and stiffness has zero row sums and nonpositive off-diagonals,
        checked to absolute 1e-12 (frame comparison also uses relative 1e-12).
    Qc : float
        Positive finite dimensionless nematic length, constant with respect to
        alpha.
    Mc : float
        Positive finite dimensionless magnetic length, constant with respect to
        alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, 2, N, N + 1) containing
        magnetic and nematic Hessians with gradient offsets and their directional
        derivatives.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 3

03_solve_magnetic_jet

Goal
----
Determine the magnetic block minimizer and its total directional derivative for the supplied quadratic and state jets.

Use the Lagrangian with positive multiplier pairing against $p-\varphi(r)$ and identity-metric quadratic penalty of strength $\zeta$.

In this block only, replace compatibility by its first-order affine expansion at the old magnetic coefficients, freeze the supplied auxiliary and multiplier values during minimization, and eliminate boundary magnetic coefficients.

Differentiate the resulting minimizer with respect to the supplied jet family, including the moving expansion point, rather than freezing the affine model when taking the derivative.

For a unit magnetic vector $n$, let $R(n)=(2nn^{\mathsf T}-I)e_1$, let $J$ be counterclockwise quarter-turn rotation, and let $P$ normalize a nonzero vector.

The exact compatibility is the continuous scalar branch through zero satisfying

$$

R(P(n+rJn))=P(R(n)+\varphi(r)JR(n)),\qquad |r|<1.

$$

```python
import numpy as np


def solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    systems : np.ndarray
        Finite quadratic jet, shape (2, 2, N, N + 1); first axis is
        value/derivative, second is magnetic/nematic, last column is gradient offset
        and preceding columns are Hessian; both Hessian layers are symmetric and
        only the base layer must be positive semidefinite, checked to absolute
        1e-12.
    boundary : np.ndarray
        Boolean mask of shape (N,); True fixes a magnetic increment at zero in both
        jet layers, while prescribed direction derivatives may remain nonzero.
    state : np.ndarray
        Finite state jet, shape (2, 3, N), with rows r, p, multiplier within each
        layer; base r lies in (-1, 1), and boundary entries of both r layers are
        zero.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N) containing the new
        magnetic coefficients and their total directional derivatives, with boundary
        entries zero in both layers.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 4

04_solve_auxiliary_jet

Goal
----
Determine the full-node auxiliary minimizer, subsequent dual-ascent update, and total directional derivatives.

Use the nematic quadratic from the supplied jet, positive multiplier pairing against $p-\varphi(r)$, and an identity-metric quadratic penalty of strength $\zeta$.

Minimize over auxiliary coefficients at every node at the supplied magnetic state, then advance the multiplier with step $\rho$ evaluated at this minimizer.

Both operations retain exact nonlinear compatibility, including in their directional derivatives.

For a unit magnetic vector $n$, let $R(n)=(2nn^{\mathsf T}-I)e_1$, let $J$ be counterclockwise quarter-turn rotation, and let $P$ normalize a nonzero vector.

The exact compatibility is the continuous scalar branch through zero satisfying

$$

R(P(n+rJn))=P(R(n)+\varphi(r)JR(n)),\qquad |r|<1.

$$

```python
import numpy as np


def solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    systems : np.ndarray
        Finite quadratic jet, shape (2, 2, N, N + 1); first axis is
        value/derivative, second is magnetic/nematic, last column is gradient offset
        and preceding columns are Hessian; both Hessian layers are symmetric and
        only the base layer must be positive semidefinite, checked to absolute
        1e-12.
    r_jet : np.ndarray
        Finite magnetic coefficient jet, shape (2, N), with base coefficients
        strictly in (-1, 1).
    multiplier_jet : np.ndarray
        Finite old multiplier jet, shape (2, N), including all nodes.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.
    rho : float
        Positive finite dual-ascent step, constant with respect to alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, 2, N) containing the new
        auxiliary and multiplier rows in the value layer and their directional
        derivatives in the derivative layer.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 5

05_relax_tangent_jet

Goal
----
Return the terminal inner state and its directional derivative along the executed stopping branch.

Each cycle performs the selectively linearized magnetic minimization, exact nonlinear auxiliary minimization on all nodes, and dual ascent with residual $p-\varphi(r)$, propagating the supplied jets through those operations.

Use the first completed cycle whose base all-node RMS compatibility residual is at most the tolerance; derivatives do not enter the stopping test, and the base cycle count is held fixed when differentiating.

Retain all state rows and both layers for the next outer update.

The preceding magnetic and auxiliary contracts define the minimizations, compatibility branch, and boundary treatment.

```python
import numpy as np


def relax_tangent_jet(
    systems: np.ndarray,
    boundary: np.ndarray,
    state: np.ndarray,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    systems : np.ndarray
        Finite quadratic jet, shape (2, 2, N, N + 1); first axis is
        value/derivative, second is magnetic/nematic, last column is gradient offset
        and preceding columns are Hessian; both Hessian layers are symmetric and
        only the base layer must be positive semidefinite, checked to absolute
        1e-12.
    boundary : np.ndarray
        Boolean mask of shape (N,); True fixes a magnetic increment at zero in both
        jet layers, while prescribed direction derivatives may remain nonzero.
    state : np.ndarray
        Finite state jet, shape (2, 3, N), with rows r, p, multiplier within each
        layer; base r lies in (-1, 1), and boundary entries of both r layers are
        zero.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.
    rho : float
        Positive finite dual-ascent step, constant with respect to alpha.
    tolerance : float
        Positive finite all-node RMS tolerance applied only to the base
        compatibility residual.
    max_iterations : int
        Positive integer inner-cycle cap; Boolean values are rejected.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, 3, N) containing the
        terminal magnetic, auxiliary, and multiplier state and its derivative on the
        base stopping branch.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    RuntimeError
        If the inner iteration cap is reached before the base tolerance.
    """
    return result
```

### Step 6

06_project_geometry_jet

Goal
----
Determine the updated geometric jet after nodewise normalization of the magnetic tangent increment.

Differentiate both the increment and its normalization, then reconstruct the nematic representation and both tangent frames.

A zero magnetic increment at a prescribed node does not set that node's direction derivative to zero.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

```python
import numpy as np


def project_geometry_jet(geometry: np.ndarray, r_jet: np.ndarray) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    geometry : np.ndarray
        Finite geometric jet, shape (2, N, N + 8), with the layout stated in the
        description; magnetic rows are unit, their derivatives tangent, frames
        consistent, and stiffness has zero row sums and nonpositive off-diagonals,
        checked to absolute 1e-12 (frame comparison also uses relative 1e-12).
    r_jet : np.ndarray
        Finite magnetic coefficient jet, shape (2, N), with base coefficients
        strictly in (-1, 1).

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point array of shape (2, N, N + 8) containing the
        projected geometry and its directional derivative with the fixed stiffness
        block preserved.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 7

07_measure_energy_jet

Goal
----
Determine the physical harmonic energy and its directional derivative on the fixed mesh.

The physical nodal state concatenates the nematic direction scaled by $Q_c$ and magnetic direction scaled by $M_c$.

Use the energy of its four-component affine interpolant with the factor of one half in the Dirichlet integral.

A jet stores a value in layer 0 and its ordinary first derivative with respect to the dimensionless parameter $\alpha$ in layer 1, without factorial scaling.

The geometry array has shape $(2,N,N+8)$: the first $N$ columns hold the fixed stiffness matrix and its zero derivative; the remaining pairs hold $n,t,\nu,\tau$ and their derivatives, with counterclockwise tangents and $\nu=(2nn^{\mathsf T}-I)e_1$.

```python
import numpy as np


def measure_energy_jet(geometry: np.ndarray, Qc: float, Mc: float) -> np.ndarray:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    geometry : np.ndarray
        Finite geometric jet, shape (2, N, N + 8), with the layout stated in the
        description; magnetic rows are unit, their derivatives tangent, frames
        consistent, and stiffness has zero row sums and nonpositive off-diagonals,
        checked to absolute 1e-12 (frame comparison also uses relative 1e-12).
    Qc : float
        Positive finite dimensionless nematic length, constant with respect to
        alpha.
    Mc : float
        Positive finite dimensionless magnetic length, constant with respect to
        alpha.

    Returns
    -------
    result : np.ndarray
        A dimensionless floating-point vector of length two containing the discrete
        harmonic energy followed by its directional derivative.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
    """
    return result
```

### Step 8

08_compute_boundary_sensitivity

Goal
----
Compute the relative directional response of the finite-budget energy decrease for the initial-angle family $\theta+\alpha v$.

The mask fixes magnetic directions during each individual relaxation, including their prescribed dependence on $\alpha$.

Propagate both layers of the coupled geometric and inner-state jets through the preceding numerical contracts, initializing both inner layers at zero only once and retaining all terminal state rows across outer updates.

Use the base inner stopping counts for the derivative and execute exactly the given outer budget.

For $S(\alpha)=1-E^{J}(\alpha)/E^0(\alpha)$, return $S'(0)/S(0)$, including the dependence of the initial energy on $\alpha$.

```python
import numpy as np


def compute_boundary_sensitivity(
    vertices: np.ndarray,
    triangles: np.ndarray,
    boundary: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
    Qc: float = 1.0013,
    Mc: float = 1.0025,
    outer_steps: int = 6,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> float:
    r"""Evaluate the stated numerical value and directional-derivative contract.

    Parameters
    ----------
    vertices : np.ndarray
        Finite dimensionless planar coordinates, shape (N, 2), N >= 3.
    triangles : np.ndarray
        Integer connectivity, shape (M, 3), M >= 1, using every vertex; each
        triangle has distinct zero-based indices and nonzero area, with either
        orientation.
    boundary : np.ndarray
        Boolean mask of shape (N,); True fixes a magnetic increment at zero in both
        jet layers, while prescribed direction derivatives may remain nonzero.
    angles : np.ndarray
        Finite initial angles in radians, shape (N,), in vertex order.
    direction : np.ndarray
        Finite angle derivatives with respect to alpha, shape (N,), in vertex order.
    Qc : float
        Positive finite dimensionless nematic length, constant with respect to
        alpha.
    Mc : float
        Positive finite dimensionless magnetic length, constant with respect to
        alpha.
    outer_steps : int
        Positive integer outer-update budget; Boolean values are rejected.
    zeta : float
        Positive finite augmentation parameter in the identity metric, constant with
        respect to alpha.
    rho : float
        Positive finite dual-ascent step, constant with respect to alpha.
    tolerance : float
        Positive finite all-node RMS tolerance applied only to the base
        compatibility residual.
    max_iterations : int
        Positive integer inner-cycle cap; Boolean values are rejected.

    Returns
    -------
    result : float
        A finite native Python float containing the dimensionless ratio of the
        directional derivative of the fractional energy decrease to its base value.

    Raises
    ------
    ValueError
        If an input violates its shape, finiteness, domain, or stated convention.
        The initial energy must exceed 1e-14 and the magnitude of $S$ must
        exceed 1e-12.
    RuntimeError
        If the inner iteration cap is reached before the base tolerance.
    """
    return result
```
