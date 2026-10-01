# Physics-Condensed_Matter_Physics-39

## Background

Stationary semiconductor drift-diffusion models couple a Poisson equation for the electrostatic potential $\psi$ to conservation equations for the electron and hole densities $n$ and $p$. Electric-field drift and concentration diffusion appear simultaneously in the carrier currents, producing a nonlinear and potentially convection-dominated system.

Classical finite-volume Scharfetter-Gummel discretizations use exponential fitting to stabilize carrier transport, but their usual multidimensional construction can depend strongly on mesh geometry. Discrete-duality finite-volume methods instead represent unknowns on coupled primal and dual meshes connected by diamond cells, allowing discrete gradients and locally conservative fluxes on more general nonorthogonal meshes.

After spatial discretization, the coupled Poisson and carrier equations form a nonlinear algebraic system. Residual convergence and discrete conservation provide standard numerical consistency checks for a computed stationary solution.

## Problem

Consider the following stationary nondimensional semiconductor drift-diffusion problem on a nonorthogonal primal-dual diamond mesh. The potential $\psi$ is scaled by the thermal voltage. Discretize carrier transport using a harmonic-average carrier-flux construction that couples the two oriented diamond directions, specialized to the conventions and parameters below.

The vertices are

$$
v_0=(0,0),\quad v_1=(2,0.1),\quad v_2=(1.7,1.3),\quad v_3=(-0.2,0.9),\quad v_4=(1.27,0.53).
$$

The counterclockwise primal triangles are

$$
K_0=(v_0,v_1,v_4),\quad K_1=(v_1,v_2,v_4),\quad K_2=(v_2,v_0,v_4),\quad K_3=(v_0,v_2,v_3).
$$

Let $x_0,\ldots,x_3$ be their centroids. Let $x_4,\ldots,x_7$ be the midpoints of boundary edges $(v_0,v_1)$, $(v_1,v_2)$, $(v_2,v_3)$, and $(v_3,v_0)$, respectively.

The oriented diamonds $(K,L,K^*,L^*)$ are

$$
D_0=(0,2,0,4),\quad D_1=(0,1,1,4),\quad D_2=(1,2,2,4),\quad D_3=(2,3,0,2),
$$

$$
D_4=(0,4,0,1),\quad D_5=(1,5,1,2),\quad D_6=(3,6,2,3),\quad D_7=(3,7,3,0).
$$

For every diamond,

$$
\sigma=[v_{K^*},v_{L^*}],\qquad \sigma^*=[x_K,x_L],
$$

$$
|\sigma|=\|v_{L^*}-v_{K^*}\|,\qquad |\sigma^*|=\|x_L-x_K\|,
$$

$$
|D|=\frac12\left|\det(x_L-x_K,v_{L^*}-v_{K^*})\right|.
$$

The unit normal $\mathbf n_{KL}\perp\sigma$ is oriented so that $\mathbf n_{KL}\cdot(x_L-x_K)>0$, and $\mathbf n_{K^*L^*}\perp\sigma^*$ is oriented so that $\mathbf n_{K^*L^*}\cdot(v_{L^*}-v_{K^*})>0$. Define the signed quantity

$$
\eta_D=\mathbf n_{KL}\cdot\mathbf n_{K^*L^*}.
$$

For a scalar field $u$,

$$
\nabla^D u=\frac{|\sigma|(u_L-u_K)\mathbf n_{KL}+|\sigma^*|(u_{L^*}-u_{K^*})\mathbf n_{K^*L^*}}{2|D|}.
$$

The parameters are

$$
D_n=1,\qquad D_p=0.65,\qquad \gamma_{\mathrm P}=0.35,\qquad R_n=R_p=0.
$$

The active cells are $K_0,K_1,K_2,K_3$, and the dual cell centered at $v_4$. Their doping values are

$$
(N_{K_0},N_{K_1},N_{K_2},N_{K_3},N_{v_4})=(-0.6,0.6,-0.6,-0.6,0.6).
$$

For contact doping $N$ and voltage $V$, the Dirichlet state is

$$
n_{\mathrm D}=\frac{N+\sqrt{N^2+4}}{2},\qquad p_{\mathrm D}=\frac{-N+\sqrt{N^2+4}}{2},\qquad \psi_{\mathrm D}=V+\log n_{\mathrm D}.
$$

The left contact has $N_{\mathrm L}=-0.6$ and $V_{\mathrm L}=0$. The right contact has $N_{\mathrm R}=0.6$ and $V_{\mathrm R}=0.25$. Vertices $v_0,v_3$ and boundary cell $K_7$ carry the left-contact state. Vertices $v_1,v_2$ and boundary cell $K_5$ carry the right-contact state.

On boundary diamonds $D_4$ and $D_6$, the outward primal boundary fluxes satisfy

$$
\nabla^D\psi\cdot\mathbf n_{KL}=0,\qquad \mathcal F^n_{KL}=0,\qquad \mathcal F^p_{KL}=0.
$$

No ghost values are associated with $K_4$ or $K_6$. Dual-direction fluxes on boundary diamonds do not enter an active dual residual because $v_0,\ldots,v_3$ are Dirichlet dual cells.

For each oriented diamond, a primal flux contributes positively to $K$ and negatively to $L$, while a dual flux contributes positively to $K^*$ and negatively to $L^*$. The integrated Poisson fluxes are

$$
\mathcal F^\psi_{KL}=|\sigma|\nabla^D\psi\cdot\mathbf n_{KL},\qquad \mathcal F^\psi_{K^*L^*}=|\sigma^*|\nabla^D\psi\cdot\mathbf n_{K^*L^*}.
$$

Each accumulated flux sum is divided by the area of its primal or dual control volume. The primal control-volume areas are the triangle areas, and the active dual-cell area is the area of the polygon $(x_0,x_1,x_2)$.

For every active cell $A$, the stationary equations are

$$
0=(\operatorname{div}_{\mathrm{DDFV}}\nabla\psi)_A+\gamma_{\mathrm P}(p_A-n_A+N_A),
$$

$$
0=(\operatorname{div}_{\mathrm{DDFV}}\mathcal F^n)_A,\qquad 0=(\operatorname{div}_{\mathrm{DDFV}}\mathcal F^p)_A.
$$

Among stationary solutions at $V_{\mathrm R}=0.25$, select the branch that is continuous in $V_{\mathrm R}$ from the charge-neutral equilibrium solution at $V_{\mathrm R}=0$ as the right-contact voltage is increased from $0$ to $0.25$. The numerical solution must satisfy the stationary equations with maximum absolute residual below $10^{-12}$.

The required scalar is the outward primal carrier current through the right physical contact edge,

$$
I_{\mathrm{right}}=\mathcal F^n_{1,5}+\mathcal F^p_{1,5}.
$$

## Output format

```
## Output format
Round $I_{\mathrm{right}}$ to exactly 10 decimal places.
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

evaluate_bernoulli

Goal
----
Evaluate the Bernoulli transport factor for finite real input values.

```python
def evaluate_bernoulli(arguments: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Evaluate the Bernoulli function stably and elementwise.

    Parameters
    ----------
    arguments : np.ndarray
        Finite real scalar or array of Bernoulli arguments.

    Returns
    -------
    values : np.ndarray
        Array with the same shape as ``arguments`` containing the corresponding
        Bernoulli transport-factor values.

    Raises
    ------
    ValueError
        If any input value is nonfinite.
    """
    return None  # placeholder
```

### Step 2

construct_diamond_geometry

Goal
----
Construct orientation-aware geometric data for primal-dual diamonds.

```python
def construct_diamond_geometry(
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
) -> "NDArray[np.float64]":
    """Construct oriented DDFV geometry for every diamond.

    Parameters
    ----------
    vertices : np.ndarray
        Finite array with shape (nv,2) containing dual-mesh vertices.
    cell_centers : np.ndarray
        Finite array with shape (nc,2) containing primal cell centers and
        boundary-edge midpoint cells.
    diamonds : np.ndarray
        Integer array with shape (m,4). Each row is (K,L,Kstar,Lstar).

    Returns
    -------
    geometry : np.ndarray
        Float array with shape (m,11) and columns
        [sigma_length, dual_length, area, nKL_x, nKL_y, nstar_x, nstar_y,
        eta, a, b, c].

    Raises
    ------
    ValueError
        If shapes or indices are invalid or any diamond is degenerate.
    """
    return None  # placeholder
```

### Step 3

build_ddfv_poisson_flux_operator

Goal
----
Construct local linear maps for the two oriented electrostatic fluxes.

```python
def build_ddfv_poisson_flux_operator(
    geometry: "NDArray[np.float64]",
) -> "NDArray[np.float64]":
    """Build the local integrated Poisson-flux operator for each diamond.

    Parameters
    ----------
    geometry : np.ndarray
        Finite array with shape (m,11) produced by
        ``construct_diamond_geometry``. Columns are
        [|sigma|, |sigma*|, |D|, nKL_x, nKL_y, nstar_x, nstar_y,
        eta, a, b, c].

    Returns
    -------
    operators : np.ndarray
        Float array with shape (m,2,4). For local values
        q=[u_K,u_L,u_Kstar,u_Lstar], ``operators[i] @ q`` equals
        [F_KL,F_KstarLstar], the two integrated oriented Poisson fluxes.

    Raises
    ------
    ValueError
        If geometry is not finite with shape (m,11), m < 1, or contains
        nonpositive lengths/areas or coefficients inconsistent with a valid
        nondegenerate DDFV diamond.
    """
    return None  # placeholder
```

### Step 4

compute_ddfv_ha_flux_jets

Goal
----
Evaluate carrier fluxes and their local first-derivative data on each diamond.

```python
def compute_ddfv_ha_flux_jets(
    psi_cells: "NDArray[np.float64]", psi_vertices: "NDArray[np.float64]",
    n_cells: "NDArray[np.float64]", n_vertices: "NDArray[np.float64]",
    p_cells: "NDArray[np.float64]", p_vertices: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]", geometry: "NDArray[np.float64]",
    D_n: float, D_p: float,
) -> "NDArray[np.float64]":
    """Evaluate harmonic-average carrier fluxes and exact local sensitivities.
    
    Parameters
    ----------
    psi_cells : np.ndarray
        One-dimensional primal-cell potential array. It must have the same
        length as ``n_cells`` and ``p_cells``. The first two columns of each
        ``diamonds`` row index this array.
    psi_vertices : np.ndarray
        One-dimensional dual-vertex potential array. It must have the same
        length as ``n_vertices`` and ``p_vertices``. The last two columns of
        each ``diamonds`` row index this array.
    n_cells : np.ndarray
        One-dimensional primal-cell electron-density array, with the same
        length as ``psi_cells`` and ``p_cells``. Referenced values must be finite.
    n_vertices : np.ndarray
        One-dimensional dual-vertex electron-density array, with the same
        length as ``psi_vertices`` and ``p_vertices``. Referenced values must be finite.
    p_cells : np.ndarray
        One-dimensional primal-cell hole-density array, with the same length
        as ``psi_cells`` and ``n_cells``. Referenced values must be finite.
    p_vertices : np.ndarray
        One-dimensional dual-vertex hole-density array, with the same length
        as ``psi_vertices`` and ``n_vertices``. Referenced values must be finite.
    diamonds : np.ndarray
        Integer array of shape ``(m,4)``. Each row is ``(K,L,Kstar,Lstar)``
        and indexes the primal-cell arrays with ``K,L`` and the dual-vertex
        arrays with ``Kstar,Lstar``.
    geometry : np.ndarray
        Finite float array of shape ``(m,11)`` returned by
        ``construct_diamond_geometry`` for the same diamond rows.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    
    Returns
    -------
    jets : np.ndarray
        Float array with shape ``(m,4,9)``. Flux rows are
        ``[Fn_primal, Fn_dual, Fp_primal, Fp_dual]``. Channel 0 is the flux.
        Channels 1:9 are derivatives with respect to
        ``[psi_K,psi_L,psi_Kstar,psi_Lstar,carrier_K,carrier_L,``
        ``carrier_Kstar,carrier_Lstar]``, where ``carrier`` is ``n`` for
        electron rows and ``p`` for hole rows.
    
    Raises
    ------
    ValueError
        If a field array is not one-dimensional, the three primal-cell field
        arrays do not have equal lengths, the three dual-vertex field arrays
        do not have equal lengths, connectivity or geometry is inconsistent,
        a referenced field value is nonfinite, or either diffusion coefficient
        is not finite and strictly positive.
    """
    return None  # placeholder
```

### Step 5

assemble_stationary_residual

Goal
----
Assemble the stationary residual for the prescribed benchmark topology.

```python
def assemble_stationary_residual(
    state: "NDArray[np.float64]",
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]",
    doping: "NDArray[np.float64]",
    left_state: "NDArray[np.float64]",
    right_state: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]",
    D_n: float,
    D_p: float,
    gamma_p: float,
) -> "NDArray[np.float64]":
    """Assemble the stationary DDFV residual for the prescribed topology.

    Parameters
    ----------
    state : np.ndarray
        Finite length-15 active state: five potentials, five electron
        densities, then five hole densities; each block is K0,K1,K2,K3,v4.
    vertices : np.ndarray
        Finite array with shape (5,2).
    cell_centers : np.ndarray
        Finite array with shape (8,2), ordered K0..K7.
    diamonds : np.ndarray
        Integer array with shape (8,4), rows (K,L,Kstar,Lstar), in arbitrary
        row order but with the prescribed eight connectivities.
    control_volumes : np.ndarray
        Positive length-5 array for K0..K3 and v4.
    doping : np.ndarray
        Finite length-5 active doping array in K0,K1,K2,K3,v4 order.
    left_state, right_state : np.ndarray
        Finite [psi,n,p] contact states with positive carrier densities.
    neumann_diamonds : np.ndarray
        Distinct integer row indices identifying exactly the two insulating
        connectivities (0,4,0,1) and (3,6,2,3) in the supplied row order.
    D_n, D_p : float
        Finite strictly positive diffusion coefficients.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.

    Returns
    -------
    residual : np.ndarray
        Finite length-15 residual in the same block ordering as ``state``.

    Raises
    ------
    ValueError
        If state, topology, boundary labels, areas, contacts, or coefficients
        violate this contract.
    """
    return None  # placeholder
```

### Step 6

assemble_newton_system

Goal
----
Assemble the Newton linear system associated with the stationary residual.

```python
def assemble_newton_system(
    state: "NDArray[np.float64]",
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]",
    doping: "NDArray[np.float64]",
    left_state: "NDArray[np.float64]",
    right_state: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]",
    D_n: float,
    D_p: float,
    gamma_p: float,
    jacobian_check_step: float = 2.0e-6,
) -> "NDArray[np.float64]":
    """Assemble the Newton matrix and right-hand side.
    
    Parameters
    ----------
    state : np.ndarray
        Finite length-15 active state ordered as the five potentials
        ``[psi_K0,psi_K1,psi_K2,psi_K3,psi_v4]``, followed by the five
        electron densities in the same control-volume order and then the five
        hole densities. Active carrier densities must be strictly positive.
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` containing the prescribed benchmark
        connectivity ``(K,L,Kstar,Lstar)``. Rows may be permuted.
    control_volumes : np.ndarray
        Finite strictly positive length-5 array ordered as
        ``[|K0|,|K1|,|K2|,|K3|,|v4*|]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in the same control-volume order.
    left_state : np.ndarray
        Finite length-3 left Dirichlet state ``[psi,n,p]`` with positive carrier densities.
    right_state : np.ndarray
        Finite length-3 right Dirichlet state ``[psi,n,p]`` with positive carrier densities.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        two insulating boundary diamonds. Indices must follow any row permutation
        of ``diamonds`` and must identify connectivities ``(0,4,0,1)`` and
        ``(3,6,2,3)``.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    jacobian_check_step : float, optional
        Finite strictly positive scale controlling the deterministic Jacobian
        consistency validation.
    
    Returns
    -------
    system : np.ndarray
        Float array with shape ``(16,15)``. Rows 0:15 contain the
        Jacobian ``J=dR/dx``. Row 15 contains the Newton right-hand side
        ``-R(state)``.
    
    Raises
    ------
    ValueError
        If any input violates the stated benchmark/residual contract, the
        Jacobian is nonfinite, or its deterministic
        consistency check fails.
    """
    return None  # placeholder
```

### Step 7

run_voltage_continuation

Goal
----
Compute the final stationary state along a supplied contact-voltage path.

```python
def run_voltage_continuation(
    vertices: "NDArray[np.float64]", cell_centers: "NDArray[np.float64]", diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]", doping: "NDArray[np.float64]", left_doping: float,
    right_doping: float, left_voltage: float, voltages: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]", D_n: float, D_p: float, gamma_p: float,
    tolerance: float=1.0e-12, max_iterations: int=20, jacobian_check_step: float=2.0e-6,
) -> "NDArray[np.float64]":
    """Solve the stationary system along a strictly increasing voltage path.
    
    Parameters
    ----------
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` containing the prescribed benchmark
        ``(K,L,Kstar,Lstar)`` connectivities; rows may be permuted.
    control_volumes : np.ndarray
        Finite strictly positive length-5 control-volume array ordered as
        ``[K0,K1,K2,K3,v4]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in that same order.
    left_doping : float
        Finite left-contact doping used to construct its Maxwell-Boltzmann
        Dirichlet state.
    right_doping : float
        Finite right-contact doping. Its carrier densities remain fixed along
        the continuation path while its potential changes with voltage.
    left_voltage : float
        Finite fixed left-contact voltage.
    voltages : np.ndarray
        Nonempty finite one-dimensional right-contact voltage sequence. The first
        entry must be zero and all later entries must increase strictly.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        two insulating boundary diamonds, remapped consistently if ``diamonds``
        has been permuted.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    tolerance : float, optional
        Finite strictly positive infinity-norm residual tolerance required at
        every continuation voltage.
    max_iterations : int, optional
        Positive integer maximum number of nonlinear iterations allowed at each
        continuation voltage.
    jacobian_check_step : float, optional
        Finite strictly positive consistency-check scale passed to
        ``assemble_newton_system``.
    
    Returns
    -------
    state : np.ndarray
        Final converged length-15 active state ordered as five potentials, five
        electron densities, and five hole densities for ``K0,K1,K2,K3,v4``.
    
    Raises
    ------
    ValueError
        If any input is invalid, an update system is singular or nonfinite, a
        candidate update leaves the positive-density domain, or a continuation level
        does not reach ``tolerance`` within ``max_iterations``.
    """
    return None  # placeholder
```

### Step 8

solve_stationary_ddfv_ha

Goal
----
Return the outward right-contact current for the converged stationary branch.

```python
def solve_stationary_ddfv_ha(
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]",
    doping: "NDArray[np.float64]",
    left_doping: float,
    right_doping: float,
    left_voltage: float,
    voltages: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]",
    D_n: float,
    D_p: float,
    gamma_p: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 20,
    jacobian_check_step: float = 2.0e-6,
    conservation_tolerance: float = 1.0e-10,
) -> float:
    """Run the complete stationary DDFV harmonic-average calculation.
    
    Parameters
    ----------
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` with the prescribed benchmark
        ``(K,L,Kstar,Lstar)`` connectivities. Rows may be permuted; contact
        diamonds are identified from connectivity rather than row number.
    control_volumes : np.ndarray
        Finite strictly positive length-5 control-volume array ordered as
        ``[K0,K1,K2,K3,v4]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in the same order.
    left_doping : float
        Finite left-contact doping.
    right_doping : float
        Finite right-contact doping.
    left_voltage : float
        Finite fixed left-contact voltage.
    voltages : np.ndarray
        Nonempty finite one-dimensional right-contact continuation sequence that
        begins at zero and increases strictly. Its final value is the terminal
        voltage at which the current is returned.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        insulating boundary diamonds, remapped consistently under row permutation.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    tolerance : float, optional
        Finite strictly positive residual infinity-norm tolerance used by voltage
        continuation.
    max_iterations : int, optional
        Positive integer Newton-iteration limit per continuation voltage.
    jacobian_check_step : float, optional
        Finite strictly positive consistency-check scale passed to the
        Newton-system assembly.
    conservation_tolerance : float, optional
        Finite strictly positive upper bound on ``abs(I_right + I_left)`` at the
        final stationary solution.
    
    Returns
    -------
    current : float
        Outward right-contact primal carrier current at the final voltage, with
        no dual-direction contact contribution.
    
    Raises
    ------
    ValueError
        If continuation fails, the required physical contact diamonds are absent,
        a terminal current is nonfinite, or terminal-current imbalance exceeds
        ``conservation_tolerance``.
    """
    return None  # placeholder
```
