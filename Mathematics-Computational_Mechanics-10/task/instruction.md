# Irreversible response of a cell-face fracture patch

## Background

# Scientific background

Diffuse fracture models replace an explicitly tracked crack surface by a scalar phase field. The elastic response is weakened as the phase approaches the fully fractured state, while a diffusion-reaction equation balances crack regularization against a history-dependent elastic driving force. A tensile-only history reduces spurious crack growth caused by compressive volumetric energy.

The selected cell-face discretization represents displacement and phase independently on mesh cells and faces. Its affine symmetric-strain reconstruction is defined by a discrete integration-by-parts identity, and a quadratic displacement reconstruction supplies a stabilization that is exact on quadratic fields. The phase equation has its own affine reconstruction and jump stabilization, so both local problems can eliminate cell unknowns before assembling a global face system.

The numerical patch isolates the conventions that distinguish this formulation from a standard low-order finite-element implementation. In particular, a two-dimensional plane-strain calculation still evaluates the volumetric-deviatoric split in three dimensions, the degradation multiplies the full local mechanics form, and the history carries the strain field associated with the largest tensile-energy maximum through subsequent unloading and reverse loading. The non-monotone load sequence makes those choices visible in the final reaction without requiring a large mesh or proprietary software.

## Problem

Compute the normalized left-boundary reaction after five signed load increments for the plane-strain unit-square cell/face discretization defined below, using exactly one mechanics-history-phase sweep per increment. The $1\,\mathrm{mm}\times1\,\mathrm{mm}$ domain contains $T_0=[0,0.5]\times[0,1]$ and $T_1=[0.5,1]\times[0,1]$ in millimetres, with oriented faces

| face | oriented endpoints in $\mathrm{mm}$ |
|---|---|
| $F_0$ | $(0,0)\to(0.5,0)$ |
| $F_1$ | $(0.5,0)\to(0.5,1)$ |
| $F_2$ | $(1,0)\to(1,1)$ |
| $F_3$ | $(1,1)\to(0.5,1)$ |
| $F_4$ | $(0.5,0)\to(1,0)$ |
| $F_5$ | $(0,1)\to(0,0)$ |
| $F_6$ | $(0.5,1)\to(0,1)$ |

and local face orders $(F_0,F_1,F_6,F_5)$ and $(F_4,F_2,F_3,F_1)$, each with outward normals in bottom-right-top-left order. Use the cell basis $[1,X,Y]=[1,(x-\bar x_T)/h_T,(y-\bar y_T)/h_T]$, order the six cell displacement coefficients as $(u_{T,x}:1,X,Y,u_{T,y}:1,X,Y)$, order each face block as $(u_{F,x}:1,s,u_{F,y}:1,s)$ in the stated local face order with $s\in[-1,1]$, order the phase coefficients as $(\phi_T,\phi_{F_0},\ldots,\phi_{F_3})$, set $h_T=\operatorname{diam}(T)$ and $h_F=\operatorname{diam}(F)$, and use tensor-product two-point Gauss quadrature in each cell direction and two-point Gauss quadrature on each face, ordering the four cell nodes by $(\xi,\eta)=(-1/\sqrt{3},-1/\sqrt{3}),(-1/\sqrt{3},1/\sqrt{3}),(1/\sqrt{3},-1/\sqrt{3}),(1/\sqrt{3},1/\sqrt{3})$. For every affine symmetric tensor $\boldsymbol{\tau}$, define the affine strain reconstruction $E_T^1\mathbf{v}_T$ and constrained quadratic displacement $p_T^2\mathbf{v}_T$ by

$$\int_T E_T^1\mathbf{v}_T:\boldsymbol{\tau}=-\int_T\mathbf{v}_T\cdot\nabla\cdot\boldsymbol{\tau}+\sum_{F\in\mathcal{F}_T}\int_F\mathbf{v}_F\cdot(\boldsymbol{\tau}\mathbf{n}_{TF}),$$

$$\int_T(\nabla_s p_T^2\mathbf{v}_T-E_T^1\mathbf{v}_T):\nabla_s\mathbf{w}=0\quad\text{for all }\mathbf{w}\in\mathbb{P}_2(T)^2,$$

$$\int_T p_T^2\mathbf{v}_T=\int_T\mathbf{v}_T,\qquad\int_T\nabla_{ss}p_T^2\mathbf{v}_T=\frac12\sum_{F\in\mathcal{F}_T}\int_F(\mathbf{v}_F\otimes\mathbf{n}_{TF}-\mathbf{n}_{TF}\otimes\mathbf{v}_F),$$

where $\nabla_{ss}\mathbf{w}=(\nabla\mathbf{w}-\nabla\mathbf{w}^{\mathsf T})/2$; then, with $\delta_T^1\mathbf{v}_T=\pi_T^1(p_T^2\mathbf{v}_T-\mathbf{v}_T)$ and $\delta_{TF}^1\mathbf{v}_T=\pi_F^1(p_T^2\mathbf{v}_T-\mathbf{v}_F)$ for the $L^2$ projections $\pi_T^1$ and $\pi_F^1$, use

$$s_T(\mathbf{u}_T,\mathbf{v}_T)=h_T^{-2}\int_T\delta_T^1\mathbf{u}_T\cdot\delta_T^1\mathbf{v}_T+\sum_{F\in\mathcal{F}_T}h_F^{-1}\int_F\delta_{TF}^1\mathbf{u}_T\cdot\delta_{TF}^1\mathbf{v}_T,$$

and multiply the complete local form

$$\int_T\left(2\mu E_T^1\mathbf{u}_T:E_T^1\mathbf{v}_T+\lambda\,\operatorname{tr}(E_T^1\mathbf{u}_T)\operatorname{tr}(E_T^1\mathbf{v}_T)\right)+2\mu s_T(\mathbf{u}_T,\mathbf{v}_T)$$

by $g(\phi_T)=(1-\phi_T)^2$, with $\lambda=121.15\,\mathrm{kN\,mm^{-2}}$ and $\mu=80.77\,\mathrm{kN\,mm^{-2}}$, storing strain as $(xx,yy,zz,xy)$ with $\varepsilon_{zz}=0$ and counting the shear component twice in tensor contractions; for the ordered coefficient vector $\widehat{\mathbf v}_T\in\mathbb R^{22}$, define $B_T\in\mathbb R^{4\times4\times22}$ by $B_{T,q}\widehat{\mathbf v}_T=(\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},\varepsilon_{xy})_q^{\mathsf T}$ at the four ordered cell nodes, and define the symmetric matrix $S_T\in\mathbb R^{22\times22}$ by $\widehat{\mathbf u}_T^{\mathsf T}S_T\widehat{\mathbf v}_T=s_T(\mathbf u_T,\mathbf v_T)$, using the physical cell and face integration weights. For each phase vector, define $p_T^1\boldsymbol{\phi}_T=\phi_T+(G_T\boldsymbol{\phi}_T)\cdot(\mathbf{x}-\bar{\mathbf{x}}_T)$ and

$$G_T\boldsymbol{\phi}_T=\frac{1}{|T|}\sum_{F\in\mathcal{F}_T}|F|\phi_F\mathbf{n}_{TF},\qquad j_T(\boldsymbol{\phi}_T,\boldsymbol{\chi}_T)=\sum_{F\in\mathcal{F}_T}\frac{1}{h_T|F|}\left(\int_F(p_T^1\boldsymbol{\phi}_T-\phi_F)\right)\left(\int_F(p_T^1\boldsymbol{\chi}_T-\chi_F)\right),$$

then, with $G_c=2.7\times10^{-3}\,\mathrm{kN\,mm^{-1}}$, $\ell=0.0075\,\mathrm{mm}$ and $\eta=0$, solve the local phase relation

$$|T|(G_T\boldsymbol{\phi}_T)\cdot(G_T\boldsymbol{\chi}_T)+j_T(\boldsymbol{\phi}_T,\boldsymbol{\chi}_T)+\sum_qw_q\left(\ell^{-2}+\frac{2H_q}{\ell G_c}\right)\phi_T\chi_T=\sum_qw_q\frac{2H_q}{\ell G_c}\chi_T$$

without clipping its solution. Starting from zero phase and zero history, impose $u_x=-d_n$ and $u_y=0$ on $F_5$, impose zero displacement on $F_2$, leave the remaining faces natural, and process $d=(0.002,0.020,-0.035,0.004,0.015)\,\mathrm{mm}$ in this order. At each increment, solve mechanics using the phase at the start of the increment, evaluate

$$\psi_0^+(\boldsymbol{\varepsilon})=\frac{K}{2}[\operatorname{tr}(\boldsymbol{\varepsilon})]_+^2+\mu\,\boldsymbol{\varepsilon}^{\mathrm{dev}}:\boldsymbol{\varepsilon}^{\mathrm{dev}},\qquad K=\lambda+\frac{2\mu}{3},\qquad \boldsymbol{\varepsilon}^{\mathrm{dev}}=\boldsymbol{\varepsilon}-\frac{\operatorname{tr}(\boldsymbol{\varepsilon})}{3}I,$$

with $[a]_+=\max(a,0)$ and the three-dimensional plane-strain embedding $\varepsilon_{zz}=0$, and replace the entire stored four-node history vector by the current one only when its maximum is strictly larger than the stored maximum. Then solve the phase relation once, use static condensation in both subproblems with local recovery of cell unknowns, and define $R_n$ as the magnitude of the condensed residual coefficient conjugate to the constant $u_x$ mode on $F_5$. In `<reasoning>`, first name the displacement and phase spaces you used, the strain reconstruction and the constrained quadratic reconstruction the stabilization is built from, the phase gradient and jump term, and the staggered order, history rule, and condensation of both subproblems; next state where $g(\phi_T)$ is applied relative to the condensation, the tensile-energy split and its shear convention, and the phase reaction and source coefficients, backing each of those three with the value your own run produced; then report $\lVert B_{T_0}\rVert_F$, the rank and Frobenius norm of $S_{T_0}$, the Frobenius norm of the condensed mechanics face matrix of $T_0$ at zero phase, the cell phase, history maximum, and normalized reaction after each increment, the final raw reaction $R_5$, and the final normalized value, and quantify the two near misses obtained by resetting history every increment and by replacing the three-dimensional $1/3$ deviator with a two-dimensional $1/2$ deviator. Report every such scalar to at least five significant figures. Report the single finite scalar $R_5/(\mu A)$, where $A=1\,\mathrm{mm}^2$, using double-precision arithmetic.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules: The tags are required. Do not omit them or leave them empty. The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05), not NaN, not Inf, not a fraction string, not a vector, not prose. Put only that one number between the tags, with no units, no words, and no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_build_affine_strain_reconstruction

Goal
----
Build the local affine symmetric-strain reconstruction.

```python
def build_affine_strain_reconstruction(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
) -> np.ndarray:
    r"""Return the quadrature-point strain reconstruction matrices.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        $[x_{\min},x_{\max},y_{\min},y_{\max}]$ for an axis-aligned rectangle.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four nondegenerate oriented face segments in local order.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding finite outward unit normals.

    Returns
    -------
    np.ndarray, shape (4, 4, 22)
        Strain matrices at the four cell Gauss nodes.

    Raises
    ------
    ValueError
        If shapes, finiteness, rectangle bounds, face geometry, or outward
        unit-normal conditions are invalid.
    """
    return result
```

### Step 2

02_build_quadratic_stabilization

Goal
----
Build the local quadratic-reconstruction stabilization matrix.

```python
def build_quadratic_stabilization(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
    strain_reconstruction: np.ndarray,
) -> np.ndarray:
    """Return the quadratic-exact local stabilization matrix.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        Axis-aligned rectangle bounds.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four oriented local face segments.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding outward unit normals.
    strain_reconstruction : np.ndarray, shape (4, 4, 22)
        Affine strain matrices at the four cell Gauss nodes.

    Returns
    -------
    np.ndarray, shape (22, 22)
        Symmetric positive-semidefinite stabilization matrix.

    Raises
    ------
    ValueError
        If any input has an invalid shape, nonfinite value, or degenerate
        geometry.
    """
    return result
```

### Step 3

03_build_phase_reconstruction

Goal
----
Build the local constant-phase affine reconstruction and jump stabilization.

```python
def build_phase_reconstruction(
    cell_bounds: np.ndarray,
    face_segments: np.ndarray,
    outward_normals: np.ndarray,
) -> np.ndarray:
    r"""Return the packed phase gradient and stabilization operators.

    Parameters
    ----------
    cell_bounds : np.ndarray, shape (4,)
        Axis-aligned rectangle bounds.
    face_segments : np.ndarray, shape (4, 2, 2)
        Four local face segments.
    outward_normals : np.ndarray, shape (4, 2)
        Corresponding outward unit normals.

    Returns
    -------
    np.ndarray, shape (7, 5)
        Rows $0{:}2$ are $G_T$ and rows $2{:}7$ are the symmetric jump
        matrix $J_T$.

    Raises
    ------
    ValueError
        If shapes, finiteness, bounds, or face geometry are invalid.
    """
    return result
```

### Step 4

04_update_volumetric_deviatoric_history

Goal
----
Update one cell's tensile history from its reconstructed strain field.

```python
def update_volumetric_deviatoric_history(
    strain_reconstruction: np.ndarray,
    displacement: np.ndarray,
    previous_history: np.ndarray,
    lame_lambda: float,
    shear_modulus: float,
) -> np.ndarray:
    r"""Return the selected quadrature-node tensile-energy field.

    Parameters
    ----------
    strain_reconstruction : np.ndarray, shape (n_q, 4, n_dof)
        Strain matrices in $(xx,yy,zz,xy)$ order.
    displacement : np.ndarray, shape (n_dof,)
        Local displacement coefficients.
    previous_history : np.ndarray, shape (n_q,)
        Finite nonnegative stored energy values.
    lame_lambda : float
        Finite Lamé first parameter.
    shear_modulus : float
        Finite positive shear modulus.

    Returns
    -------
    np.ndarray, shape (n_q,)
        Either all current tensile-energy values or the unchanged previous
        field, according to the strict maximum comparison.

    Raises
    ------
    ValueError
        If dimensions, finiteness, nonnegativity, or elastic stability are
        invalid.
    """
    return result
```

### Step 5

05_condense_degraded_mechanics

Goal
----
Assemble and statically condense one degraded mechanics cell.

```python
def condense_degraded_mechanics(
    strain_reconstruction: np.ndarray,
    quadrature_weights: np.ndarray,
    stabilization: np.ndarray,
    cell_phase: float,
    lame_lambda: float,
    shear_modulus: float,
    n_cell: int = 6,
) -> np.ndarray:
    r"""Return the condensed matrix stacked above the cell recovery matrix.

    Parameters
    ----------
    strain_reconstruction : np.ndarray, shape (n_q, 4, n_dof)
        Local strain matrices in $(xx,yy,zz,xy)$ order.
    quadrature_weights : np.ndarray, shape (n_q,)
        Finite positive cell quadrature weights.
    stabilization : np.ndarray, shape (n_dof, n_dof)
        Finite symmetric positive-semidefinite local stabilization.
    cell_phase : float
        Finite phase value in $[0,1)$.
    lame_lambda : float
        Finite Lamé first parameter.
    shear_modulus : float
        Finite positive shear modulus.
    n_cell : int, optional
        Number of leading cell coefficients.

    Returns
    -------
    np.ndarray, shape (n_dof, n_dof - n_cell)
        The face Schur matrix in the first $n_{\mathrm{face}}$ rows and the
        recovery map $\mathbf{u}_C=R\mathbf{u}_F$ in the last
        $n_{\mathrm{cell}}$ rows.

    Raises
    ------
    ValueError
        If input contracts, symmetry, positivity, or the cell solve fail.
    """
    return result
```

### Step 6

06_condense_phase_field

Goal
----
Assemble and statically condense one local phase-field equation.

```python
def condense_phase_field(
    phase_reconstruction: np.ndarray,
    quadrature_weights: np.ndarray,
    history: np.ndarray,
    previous_cell_phase: float,
    length_scale: float,
    fracture_toughness: float,
    viscosity: float = 0.0,
    time_step: float = 1.0,
) -> np.ndarray:
    r"""Return the packed condensed phase system and cell recovery rule.

    Parameters
    ----------
    phase_reconstruction : np.ndarray, shape (7, 5)
        Gradient rows followed by the five-by-five jump matrix.
    quadrature_weights : np.ndarray, shape (n_q,)
        Finite positive cell quadrature weights.
    history : np.ndarray, shape (n_q,)
        Finite nonnegative quadrature-node history values.
    previous_cell_phase : float
        Finite previous cell phase in $[0,1]$.
    length_scale : float
        Finite positive regularization length.
    fracture_toughness : float
        Finite positive critical energy-release rate.
    viscosity : float, optional
        Finite nonnegative viscous coefficient.
    time_step : float, optional
        Finite positive pseudo-time increment.

    Returns
    -------
    np.ndarray, shape (5, 5)
        Rows $0{:}4$ contain $[A^{\mathrm{sc}}\mid\mathbf{b}^{\mathrm{sc}}]$;
        the last row stores $\mathbf{r}$ followed by $c$ in the recovery rule
        $\phi_T=\mathbf{r}^{T}\boldsymbol{\phi}_F+c$.

    Raises
    ------
    ValueError
        If any shape, finiteness, sign, or symmetry contract is violated.
    """
    return result
```

### Step 7

07_advance_staggered_patch

Goal
----
Advance the fixed two-cell patch by one staggered load increment.

```python
def advance_staggered_patch(
    displacement_load: float,
    previous_phase: np.ndarray,
    previous_history: np.ndarray,
    material: np.ndarray,
) -> np.ndarray:
    r"""Advance one load increment on the prescribed two-cell plane-strain patch.

    Parameters
    ----------
    displacement_load : float
        Finite signed left-boundary displacement magnitude in millimetres;
        the imposed constant horizontal coefficient is its negative.
    previous_phase : np.ndarray, shape (9,)
        Two cell values followed by global face values $F_0$ through $F_6$.
    previous_history : np.ndarray, shape (2, 4)
        Cell-by-quadrature nonnegative tensile-history values.
    material : np.ndarray, shape (6,)
        $[\lambda,\mu,G_c,\ell,\eta,\Delta t]$ in the stated unit system.

    Returns
    -------
    np.ndarray, shape (18,)
        Updated phase values, flattened updated history, and the normalized
        mechanics reaction from this sweep, in that order.

    Raises
    ------
    ValueError
        If the state or material contracts fail, or a condensed global solve
        is singular or produces a nonfinite value.
    """
    return result
```

### Step 8

08_compute_irreversible_patch_response

Goal
----
Run the full load path and return the final normalized reaction.

```python
def compute_irreversible_patch_response(
    load_path: np.ndarray | None = None,
    material: np.ndarray | None = None,
) -> float:
    r"""Return the final normalized reaction for the fixed two-cell patch.

    Parameters
    ----------
    load_path : np.ndarray, shape (n_steps,), optional
        Nonempty finite signed displacement sequence in millimetres. ``None``
        selects $[0.002,0.020,-0.035,0.004,0.015]\,\mathrm{mm}$.
    material : np.ndarray, shape (6,), optional
        $[\lambda,\mu,G_c,\ell,\eta,\Delta t]$ in the stated unit system.
        ``None`` selects $[121.15,80.77,2.7\times10^{-3},0.0075,0,1]$.

    Returns
    -------
    float
        Finite final normalized reaction after all increments.

    Raises
    ------
    ValueError
        If the load path or material is invalid, or any staggered step fails.
    """
    return result
```
