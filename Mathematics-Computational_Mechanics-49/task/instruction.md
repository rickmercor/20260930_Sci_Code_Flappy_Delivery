# Mathematics-Computational_Mechanics-49

## Background

Projection-based model order reduction (MOR) reduces the computational cost of nonlinear solid-mechanics simulations by restricting the solution to a lower-dimensional space. Hyperreduction reduces this cost more by evaluating nonlinear constitutive behavior only at a small subset of weighted integration points. In strain-space MOR, the reduced representation is applied directly to strain or deformation quantities. The boundary-consistent lifting fields can enforce nonzero Dirichlet conditions, while the reduced basis represents only the remaining fluctuation field. For hyperelastic materials, the constitutive response is computed from a strain-energy density $W(F)$, with stress and tangent defined as $P=\partial W/\partial F$, and $A=\partial P/\partial F$. These determine the reduced residual and Jacobian used in Newton-type solutions. Consistency between the constitutive model and its tangent is important for the reduced update. The component-wise MOR applies this by assigning separate reduced bases, integration points, and weights to individual substructures that are later combined. This allows the reduced components to be reused in different assemblies.

## Problem

Hyperreduction can make nonlinear solid-mechanics reduced models practical by replacing the full integration with weighted local evaluations. Strain-space formulations also provide nonlinear constitutive behavior that can be evaluated without an element-level assembly. Consider a two-component reduced system where each component has independently selected hyperreduction points and weights. It is connected to a common global reduced state through a component-specific coordinate map. The material is compressible Mooney-Rivlin with

$$
W(F)=c_1(\bar I_1-3)+c_2(\bar I_2-3)+\frac{\kappa}{4}\left(J^2-1-2\ln J\right),
$$

where $J=\det(F)$, $C=F^TF$, $\bar I_1=J^{-2/3}\operatorname{tr}(C)$, $\bar I_2=J^{-4/3}\frac{1}{2}\left[(\operatorname{tr}(C))^2-\operatorname{tr}(C^2)\right]$, $c_1=1.3$, $c_2=0.7$, and $\kappa=250$. For component $i\in\{A,B\}$, let its local reduced coordinate be $z_i=C_i y$, and at the selected point $c$, define

$$
F_{ic}(y)=I+a_{ic}(z_i)_1B_1+b_{ic}(z_i)_2B_2+\ell_{ic1}d_1L_1+\ell_{ic2}d_2L_2
$$

and 

$$
\phi_{ic}=\begin{matrix}[
a_{ic}\operatorname{vec}(B_1) & b_{ic}\operatorname{vec}(B_2)
\end{matrix}].
$$

With $P(F)=\partial W/\partial F$, the common reduced equilibrium residual is defined by

$$
r(y)=\sum_{i\in{A,B}}C_i^T\left[\sum_{c\in H_i}\xi_{ic}\phi_{ic}^T\operatorname{vec}\!\left(P(F_{ic}(y)) \right) \right]-f_{\mathrm{ext}},
$$

where $\operatorname{vec}$ uses a row-major order of $(11,12,13,21,22,23,31,32,33)$ for every second-order tensor, and the exact Jacobian $J_r(y)=\partial r/\partial y$ follows the same convention. 

Determine the first coordinate of
$$
y^{(1)}=y^{(0)} - J_r(y^{(0)})^{-1}r(y^{(0)})
$$

for the data below, using the exact constitutive derivatives implied by $W$ and no constitutive approximation. The final answer is the scalar $(y^{(1)})_1$, rounded to ten decimal places. 

$$
B_1=\left[\begin{matrix}
0.60 & 0.15 & 0\\
0.15 & -0.20 & 0.05\\
0 & 0.05 & 0.10
\end{matrix}\right],\qquad

B_2=\left[\begin{matrix}
-0.10 & 0.20 & 0.04\\
0.20 & 0.50 & 0\\
0.04 & 0 & -0.30
\end{matrix}\right]
$$

$$
L_1=\left[\begin{matrix}
0.40 & 0.10 & 0\\
0.10 & -0.10 & 0.02\\
0 & 0.02 & -0.20
\end{matrix}\right],\qquad

L_2=\left[\begin{matrix}
-0.20 & 0 & 0.05\\
0 & 0.30 & 0.08\\
0.05 & 0.08 & 0.10
\end{matrix}\right]
$$

$$
C_A=\left[\begin{matrix}
1.00 & 0.35\\
-0.25 & 0.90
\end{matrix}\right],\qquad

C_B=\left[\begin{matrix}
0.80 & -0.40\\
0.30 & 1.10
\end{matrix}\right]
$$

$$
y^{(0)}=\left[\begin{matrix}
0.018\\
-0.012
\end{matrix}\right],\qquad

d=\left[\begin{matrix}
0.04\\
-0.03
\end{matrix}\right],\qquad

f_{\mathrm{ext}}=\left[\begin{matrix}
0.08\\
-0.03
\end{matrix}\right]
$$

Component $A$:
$$
\begin{array}{c|ccccc}
c & a_{Ac} & b_{Ac} & \ell_{Ac1} & \ell_{Ac2} & \xi_{Ac}\\
\hline
1 & 1.0 & 0.7 & 0.8 & -0.2 & 0.55\\
2 & -0.6 & 1.1 & 0.3 & 0.9 & 0.85
\end{array}
$$

Component $B$:
$$
\begin{array}{c|ccccc}
c & a_{Bc} & b_{Bc} & \ell_{Bc1} & \ell_{Bc2} & \xi_{Bc}\\
\hline
1 & 0.9 & -0.8 & 1.2 & 0.4 & 0.65\\
2 & 1.3 & 0.5 & -0.4 & 1.0 & 0.95
\end{array}
$$

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise, but include the scientific evidence needed to support the result. Include scientific assumptions, methodological context, and intermediate numerical results that materially support the calculation.
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

step1_component_point_kinematics

Goal
----
Creates the deformation gradients and reduced strain-mode matrices required by the constitutive stages.

```python
import numpy as np

def component_point_kinematics(C: np.ndarray, y: np.ndarray, d: np.ndarray,
                                     points: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                                     L1: np.ndarray, L2: np.ndarray):
    """Construct selected-point deformation gradients and reduced mode matrices for one component.

        Parameters
        ----------
        C : np.ndarray
            Component coordinate map with shape (2, 2).
        y : np.ndarray
            Global reduced state with shape (2,).
        d : np.ndarray
            Boundary parameters with shape (2,).
        points : np.ndarray
            Selected-point data with shape (m, 5), where each row is
            [a, b, ell1, ell2, weight].
        B1 : np.ndarray
            First reduced deformation-mode matrix with shape (3, 3).
        B2 : np.ndarray
            Second reduced deformation-mode matrix with shape (3, 3).
        L1 : np.ndarray
            First boundary-lifting matrix with shape (3, 3).
        L2 : np.ndarray
            Second boundary-lifting matrix with shape (3, 3).

        Returns
        -------
        F_points : np.ndarray
            Selected-point deformation gradients with shape (m, 3, 3).
        phis : np.ndarray
            Selected-point reduced mode matrices with shape (m, 9, 2).
        
        Raises
        -------
        ValueError : If the component map, reduced-state data, point data, or mode
            have incompatible dimensions, contain nonfinite values, or violate the
            positivity conditions.
    """
    return F_points, phis
```

### Step 2

step2_mooney_rivlin_state

Goal
----
Computes the Mooney-Rivlin constitutive state quantities and their directional variations.

```python
import numpy as np

def mooney_rivlin_state(F: np.ndarray, directions: np.ndarray):
    """Compute Mooney-Rivlin constitutive state quantities and their directional variations.

        Parameters
        ----------
        F : np.ndarray
            Deformation gradient with shape (3, 3).
        directions : np.ndarray
            Two deformation-gradient perturbation directions with shape (2, 3, 3).

        Returns
        -------
        state : np.ndarray
            Constitutive state vector with shape (5,).
        differential_state : np.ndarray
            Directional variations of the constitutive state with shape (2, 5).

        Raises
        -------
        ValueError : If the deformation gradient or perturbation directions
            have incompatible dimensions, contain nonfinite values, or
            place the deformation outside the admissible hyperelastic domain.
    """
    return state, differential_state
```

### Step 3

step3_mooney_rivlin_constitutive

Goal
----
Evaluates the first Piola-Kirchhoff stress and exact consistent tangent actions.

```python
import numpy as np

def mooney_rivlin_constitutive(F: np.ndarray, directions: np.ndarray,
                                     state: np.ndarray, differential_state: np.ndarray,
                                     c1: float, c2: float, kappa: float):
    """Evaluate the first Piola-Kirchhoff stress and exact consistent tangent actions.

        Parameters
        ----------
        state : np.ndarray
            Constitutive state vector with shape (5,).
        differential_state : np.ndarray
            Directional constitutive-state variations with shape (2, 5).
        c1 : float
            First positive Mooney-Rivlin material coefficient.
        c2 : float
            Second positive Mooney-Rivlin material coefficient.
        kappa : float
            Positive bulk-modulus parameter.

        Returns
        -------
        P : np.ndarray
            First Piola-Kirchhoff stress with shape (3, 3).
        tangent_actions : np.ndarray
            Exact consistent tangent actions with shape (2, 3, 3).
            
        Raises
        -------
        ValueError : If the deformation data, constitutive-state data, or material parameters
            have incompatible dimensions, contain nonfinite values, or are mutually inconsistent
            with the admissible contitutive state.
    """
    return P,dP
```

### Step 4

step4_component_hyperreduced_system

Goal
----
Assembles the component-local hyperreduced residual and Jacobian.

```python
import numpy as np

def component_hyperreduced_system(phis: np.ndarray, weights: np.ndarray, stresses: np.ndarray, tangent_actions: np.ndarray):
    """Assemble the component-local hyperreduced residual and Jacobian.

        Parameters
        ----------
        phis : np.ndarray
            Selected-point reduced mode matrices with shape (m, 9, 2).
        weights : np.ndarray
            Positive hyperreduction weights with shape (m,).
        stresses : np.ndarray
            Selected-point first Piola-Kirchhoff stresses with shape (m, 3, 3).
        tangent_actions : np.ndarray
            Selected-point tangent actions with shape (m, 2, 3, 3).

        Returns
        -------
        r_local : np.ndarray
            Component-local reduced residual with shape (2,).
        K_local : np.ndarray
            Component-local reduced Jacobian with shape (2, 2).

        Raises
        -------
        ValueError : If the selected-point mode, weight, stress, or tangent data
            have incompatible dimensions, contain nonfinite values, or invalid
            hyperreduction weights.
    """
    return r_local,K_local
```

### Step 5

step5_global_reduced_system

Goal
----
Assembles the coupled global reduced residual and Jacobian from component contributions.

```python
import numpy as np


def global_reduced_system(component_maps: np.ndarray, local_residuals: np.ndarray, local_jacobians: np.ndarray, f_ext: np.ndarray):
    """Assemble the coupled global reduced residual and Jacobian from component contributions.

        Parameters
        ----------
        component_maps : np.ndarray
            Component coordinate maps with shape (n, 2, 2).
        local_residuals : np.ndarray
            Component-local reduced residuals with shape (n, 2).
        local_jacobians : np.ndarray
            Component-local reduced Jacobians with shape (n, 2, 2).
        f_ext : np.ndarray
            External reduced load with shape (2,).

        Returns
        -------
        r : np.ndarray
            Coupled global reduced residual with shape (2,).
        K : np.ndarray
            Coupled global reduced Jacobian with shape (2, 2).

        Raises
        -------
        ValueError : If the component maps, local residuals, local Jacobians, or external load
            have incompatible dimensions, contain nonfinite values, or inconsistent 
            component counts.
    """
    return r,K
```

### Step 6

step6_newton_update_diagnostics

Goal
----
Applies one undamped Newton correction and computes numerical diagnostics.

```python
import numpy as np

def newton_update_diagnostics(y: np.ndarray, r: np.ndarray, K: np.ndarray):
    """Apply one undamped Newton correction and compute numerical diagnostics.

        Parameters
        ----------
        y : np.ndarray
            Current reduced state with shape (2,).
        r : np.ndarray
            Coupled reduced residual with shape (2,).
        K : np.ndarray
            Coupled reduced Jacobian with shape (2, 2).

        Returns
        -------
        y_new : np.ndarray
            Updated reduced state with shape (2,).
        diagnostics : np.ndarray
            Array containing the residual 2-norm, Jacobian 2-norm condition number,
            and Newton-correction 2-norm.

        Raises
        -------
        ValueError : If the reduced state, residual, or Jacobian
            have incompatible dimensions, contain nonfinite values, or 
            do not define a valid finite Newton update.
    """
    return y_new,diagnostics
```

### Step 7

step7_first_newton_coordinate

Goal
----
Runs the complete component-wise strain-space hyperreduction pipeline.

```python
import numpy as np

def first_newton_coordinate(y: np.ndarray, d: np.ndarray,
                            component_maps: np.ndarray, point_sets: list,
                            B1: np.ndarray, B2: np.ndarray,
                            L1: np.ndarray, L2: np.ndarray,
                            f_ext: np.ndarray,
                            c1: float, c2: float, kappa: float):
    """Run the complete component-wise strain-space hyperreduction pipeline.

    Parameters
    ----------
    y : np.ndarray
        Initial global reduced state with shape (2,).
    d : np.ndarray
        Boundary parameters with shape (2,).
    component_maps : np.ndarray
        Component coordinate maps with shape (n, 2, 2).
    point_sets : list[np.ndarray]
        One selected-point array per component, with rows
        [a, b, ell1, ell2, weight].
    B1 : np.ndarray
        First reduced deformation-mode matrix with shape (3, 3).
    B2 : np.ndarray
        Second reduced deformation-mode matrix with shape (3, 3).
    L1 : np.ndarray
        First boundary-lifting matrix with shape (3, 3).
    L2 : np.ndarray
        Second boundary-lifting matrix with shape (3, 3).
    f_ext : np.ndarray
        External reduced load with shape (2,).
    c1 : float
        First positive Mooney-Rivlin material coefficient.
    c2 : float
        Second positive Mooney-Rivlin material coefficient.
    kappa : float
        Positive bulk-modulus parameter.

    Returns
    -------
    value : float
        First coordinate of the reduced state after one exact undamped Newton update.

    Raises
    ------
    ValueError : If any supplied data violate the dimensional, finiteness, positivity,
        constitutive-admissibility, or solvability requirements of the
        end-to-end reduced-order pipeline.
    """
    return value
```
