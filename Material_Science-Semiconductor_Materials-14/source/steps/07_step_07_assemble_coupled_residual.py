"""
Assemble the residual of the full coupled Poisson and continuity system, with the Dirichlet and Neumann rows substituted in.

The stationary drift-diffusion model is three equations in three fields. Poisson relates the divergence of the electric displacement to the net space charge, which is the hole density minus the electron density plus the net doping, and the two continuity equations state that the divergence of each carrier flux equals the net recombination rate, taken to vanish here so that both carrier currents are separately conserved. In the nondimensionalisation used throughout, lengths are measured in micrometres, potentials in thermal voltages and densities in the reference doping, and the whole of the material data collapses into a single dimensionless group multiplying the Laplacian: the permittivity times the thermal voltage divided by the elementary charge, the reference density and the square of the length unit. That group is the square of the Debye length in units of the device size, so it is small, of order 1e-2 here, and it is what makes the Poisson equation singularly perturbed and the depletion region thin.




Assembling the residual is then a matter of stacking three blocks and then replacing the rows where the balance equations do not apply. The Poisson block is the dimensionless group multiplying the discrete Laplacian applied to the potential, plus the local space charge. The two carrier blocks are the flux matrices of the previous step applied to their own density, which is exactly the discrete flux balance because the matrices already carry the division by the control volume measure. Note the asymmetry that this creates in the Jacobian later on: the carrier blocks are linear in their own density at frozen potential, but nonlinear in the potential through every Bernoulli function.




Three kinds of rows are then overwritten. On contact-free boundary edges the governing equation is homogeneous Neumann, so the Poisson row becomes the vanishing of the normal gradient rather than a charge balance, and the carrier rows become the vanishing of the normal flux; the matrices of the previous steps already hold those raw contraction coefficients on exactly those rows, so nothing has to be recomputed and only the space charge term has to be removed from the Poisson row. On Ohmic contacts, primal and dual alike, all three rows are replaced by the difference between the unknown and its prescribed equilibrium value, which is what makes the assembled system square and pins the solution. Interior primal and dual rows are left untouched.


The residual is returned as one stacked vector in the order potential, electrons, holes, and its infinity norm is the scaled residual against which convergence is measured. Measuring it that way rather than as a two-norm matters because the three blocks carry very different magnitudes: the Poisson rows are of order the doping, while the carrier rows are divided by cell measures of order 1e-3 and are therefore amplified, so a two-norm would let a badly unconverged carrier equation hide behind a well-converged Poisson equation.

Returns
-------
np.ndarray of shape (3 * n_nodes,), float: the stacked Poisson, electron and hole residuals with the boundary rows substituted in.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_coupled_residual(nodes: np.ndarray, state: np.ndarray,
                              laplacian: np.ndarray, electron_matrix: np.ndarray,
                              hole_matrix: np.ndarray,
                              unknowns: np.ndarray) -> np.ndarray:
    """Assemble the residual of the coupled drift-diffusion system.

    Parameters
    ----------
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    state : np.ndarray
        Doping and Ohmic contact values of shape (n_nodes, 4) as returned by
        build_junction_state.
    laplacian : np.ndarray
        Discrete duality Laplacian of shape (n_nodes, n_nodes).
    electron_matrix : np.ndarray
        Electron flux matrix of shape (n_nodes, n_nodes) assembled at the
        potential held in unknowns.
    hole_matrix : np.ndarray
        Hole flux matrix of shape (n_nodes, n_nodes) assembled at the same
        potential.
    unknowns : np.ndarray
        Current iterate of shape (n_nodes, 3) holding the potential, the
        electron density and the hole density.

    Returns
    -------
    residual : np.ndarray
        Array of shape (3 * n_nodes,) stacking the Poisson, electron and
        hole residuals in that order.

    Raises
    ------
    ValueError
        If nodes, state, unknowns or any operator has the wrong shape, or if
        unknowns contains a non-finite value.
    """
    return residual  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_assemble_coupled_residual(nodes: np.ndarray, state: np.ndarray,
                                      laplacian: np.ndarray,
                                      electron_matrix: np.ndarray,
                                      hole_matrix: np.ndarray,
                                      unknowns: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    state = np.asarray(state, dtype=float)
    laplacian = np.asarray(laplacian, dtype=float)
    electron_matrix = np.asarray(electron_matrix, dtype=float)
    hole_matrix = np.asarray(hole_matrix, dtype=float)
    unknowns = np.asarray(unknowns, dtype=float)

    if nodes.ndim != 2 or nodes.shape[1] != 5 or nodes.shape[0] < 1:
        raise ValueError("nodes must be a 2D array with shape (n_nodes, 5)")
    n_nodes = nodes.shape[0]
    if state.shape != (n_nodes, 4):
        raise ValueError("state must have shape (n_nodes, 4)")
    if unknowns.shape != (n_nodes, 3):
        raise ValueError("unknowns must have shape (n_nodes, 3)")
    for name, matrix in (("laplacian", laplacian),
                         ("electron_matrix", electron_matrix),
                         ("hole_matrix", hole_matrix)):
        if matrix.shape != (n_nodes, n_nodes):
            raise ValueError(f"{name} must have shape (n_nodes, n_nodes)")
    if not np.all(np.isfinite(unknowns)):
        raise ValueError("unknowns must contain only finite values")

    # Silicon at 300 K. Under the stated nondimensionalisation the whole
    # material data set collapses into this one dimensionless group, the
    # square of the Debye length in units of the length scale.
    elementary_charge = 1.602192e-19          # C
    permittivity = 1.035941e-12               # C / (V cm)
    thermal_voltage = 0.025852                # V
    reference_density = 1.0e15                # cm^-3
    length_unit = 1.0e-4                      # cm, that is 1 micrometre
    debye_group = (permittivity * thermal_voltage
                   / (elementary_charge * reference_density * length_unit ** 2))

    code = nodes[:, 3]
    doping, electrons_dir, holes_dir, potential_dir = state.T
    potential, electrons, holes = unknowns.T

    curvature = laplacian @ potential
    poisson = debye_group * curvature + (holes - electrons + doping)
    electron_balance = electron_matrix @ electrons
    hole_balance = hole_matrix @ holes

    # Contact-free boundary edges: the row already holds the raw normal
    # contraction, so only the space charge term is dropped from Poisson.
    neumann = code == 2.0
    poisson[neumann] = curvature[neumann]

    # Ohmic contacts, primal and dual alike: prescribe all three unknowns.
    dirichlet = np.isin(code, (1.0, 4.0))
    poisson[dirichlet] = potential[dirichlet] - potential_dir[dirichlet]
    electron_balance[dirichlet] = electrons[dirichlet] - electrons_dir[dirichlet]
    hole_balance[dirichlet] = holes[dirichlet] - holes_dir[dirichlet]

    return np.concatenate([poisson, electron_balance, hole_balance])

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    synthetic = """import numpy as np
def make_problem(seed, n_cells, n_bnd, n_dual):
    rng = np.random.default_rng(seed)
    n_nodes = n_cells + n_bnd + n_dual
    nodes = np.zeros((n_nodes, 5))
    nodes[:, 0] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:, 1] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:n_cells, 2] = rng.uniform(0.05, 0.20, n_cells)
    nodes[n_cells + n_bnd:, 2] = rng.uniform(0.05, 0.20, n_dual)
    nodes[n_cells:n_cells + n_bnd, 3] = rng.choice([1.0, 2.0], n_bnd)
    nodes[n_cells + n_bnd:, 3] = rng.choice([3.0, 4.0, 5.0], n_dual)
    state = np.zeros((n_nodes, 4))
    state[:, 0] = np.where(nodes[:, 1] < 0.5, 1.0, -1.0)
    root = np.sqrt(state[:, 0] ** 2 + 4.0 * 1.087386e-05 ** 2)
    state[:, 1] = 0.5 * (state[:, 0] + root)
    state[:, 2] = 0.5 * (-state[:, 0] + root)
    state[:, 3] = np.log(state[:, 1] / 1.087386e-05)
    laplacian = rng.uniform(-2.0, 2.0, (n_nodes, n_nodes))
    electron_matrix = rng.uniform(-1.0, 1.0, (n_nodes, n_nodes))
    hole_matrix = rng.uniform(-1.0, 1.0, (n_nodes, n_nodes))
    unknowns = np.column_stack([
        rng.uniform(4.0, 12.0, n_nodes),
        rng.uniform(1e-8, 1.0, n_nodes),
        rng.uniform(1e-8, 1.0, n_nodes)])
    return nodes, state, laplacian, electron_matrix, hole_matrix, unknowns
"""
    call = 'float(np.sum((_a := np.ravel(assemble_coupled_residual(nodes, state, laplacian, electron_matrix, hole_matrix, unknowns))) * np.arange(1, _a.size + 1)))'
    gold = 'float(np.sum((_a := np.ravel(_oracle_assemble_coupled_residual(nodes, state, laplacian, electron_matrix, hole_matrix, unknowns))) * np.arange(1, _a.size + 1)))'
    return [
        # --- Valid: a mixed configuration exercising every equation class ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(0, 8, 5, 10)
""",
            "call": call,
            "gold_call": gold,
        },
        # --- Valid: a larger configuration ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(4, 20, 9, 24)
""",
            "call": call,
            "gold_call": gold,
        },
        # --- Boundary: the iterate already equals the prescribed contact values ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(2, 10, 6, 12)
unknowns = np.column_stack([state[:, 3], state[:, 1], state[:, 2]])
""",
            "call": call,
            "gold_call": gold,
        },
        # --- Boundary: no boundary or dual nodes, so every row is an interior balance ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(6, 12, 4, 8)
nodes[:, 3] = 0.0
""",
            "call": call,
            "gold_call": gold,
        },
        # --- Edge: carrier densities at the positivity truncation floor ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(9, 8, 5, 10)
unknowns[:, 1] = 1.0e-20
unknowns[:, 2] = 1.0e-20
""",
            "call": call,
            "gold_call": gold,
        },
        # --- Invalid: unknowns with the wrong column count ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(1, 8, 5, 10)
bad = unknowns[:, :2]
def run_model():
    try:
        assemble_coupled_residual(nodes, state, laplacian, electron_matrix, hole_matrix, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupled_residual(nodes, state, laplacian, electron_matrix, hole_matrix, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a flux matrix of the wrong order ---
        {
            "setup": synthetic + """
nodes, state, laplacian, electron_matrix, hole_matrix, unknowns = make_problem(3, 8, 5, 10)
bad = hole_matrix[:-1, :-1]
def run_model():
    try:
        assemble_coupled_residual(nodes, state, laplacian, electron_matrix, bad, unknowns)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_coupled_residual(nodes, state, laplacian, electron_matrix, bad, unknowns)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
