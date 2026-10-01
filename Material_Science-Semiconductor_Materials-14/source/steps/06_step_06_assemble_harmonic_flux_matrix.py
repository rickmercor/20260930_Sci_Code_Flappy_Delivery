"""
Assemble the harmonic-average discrete duality flux matrix for one carrier at a frozen potential, the operator whose row sums are the discrete continuity residuals.

The continuity equation for electrons becomes self adjoint under the Slotboom change of variable, in which the electron density is replaced by itself multiplied by the exponential of minus the scaled potential. The flux then reads as the diffusion coefficient multiplied by the exponential of the potential multiplied by the gradient of the Slotboom variable, so the entire drift physics has been pushed into a scalar coefficient in front of a pure gradient. Discretising that gradient with the diamond reconstruction leaves one question: what single number should stand in for the exponential coefficient on an edge across which the potential varies by many thermal voltages. Averaging it arithmetically is what a centred scheme does and it oscillates. The answer used here is the harmonic average of the exponential of the linear potential projection along the edge, which is the choice that reproduces the exact solution of the one-dimensional problem, and evaluating that average in closed form gives the exponential of the potential at one endpoint multiplied by the Bernoulli function of the potential difference between the two endpoints.




The subtlety that makes this a genuinely two-dimensional construction is which endpoints appear. Integrating over a primal cell and freezing the coefficient on the primal edge produces a coefficient built from the two dual nodes, while integrating over a dual cell and freezing on the dual edge produces one built from the two primal nodes. Each frozen coefficient therefore multiplies a Slotboom difference taken along the other direction, and if the expressions are left in that form the exponential in the coefficient and the exponentials hidden in the Slotboom variables do not share an index and do not cancel. What is left is a factor of the exponential of a difference of potentials at unrelated nodes, which across a junction is the exponential of the built-in voltage in thermal units and overflows on the first Newton step. The construction therefore swaps the two frozen coefficients in the diagonal terms, so that each exponential factor carries the same nodal pair as the density difference it multiplies. The swap costs nothing at second order, because the two coefficients differ by the local mesh size times the exponential evaluated at the diamond centre while the Slotboom differences are themselves first order.




After the swap the exponentials telescope, and each direction collapses to the classical Scharfetter-Gummel flux, the Bernoulli function of the potential difference times the first density minus the Bernoulli function of the reversed difference times the second. Each of the two discrete fluxes then has two terms: a diagonal term along its own direction weighted by the square of that edge length, and a cross term along the other direction weighted by the product of the two edge lengths and by the scalar product of the two unit normals, all divided by twice the diamond area. The cross term is the entire discrete duality contribution. When the two normals are orthogonal it disappears and the primal and dual fluxes decouple into two independent finite volume Scharfetter-Gummel fluxes, which is the sense in which the scheme extends the classical method rather than replacing it.




Holes are not obtained by negating the electron flux. Their Slotboom variable carries the opposite sign in the exponent, so every Bernoulli argument reverses: where the electron flux weights the first node by the Bernoulli function of the forward potential difference, the hole flux weights it by the Bernoulli function of the reversed difference. That reversal is what makes holes drift down the potential gradient while electrons drift up it.




The assembled matrix maps a nodal carrier vector to the discrete flux balance of every unknown: the sum of primal fluxes divided by the primal cell measure on interior primal rows, the sum of dual fluxes divided by the dual cell measure on interior and contact-free dual rows, and the raw primal flux on contact-free boundary edge rows, where the governing equation is the vanishing of that flux. Dirichlet rows receive no contribution at all and are left as zeros, to be overwritten when the boundary values are imposed.

Returns
-------
np.ndarray of shape (n_nodes, n_nodes), float: the harmonic-average discrete duality flux matrix of the requested carrier at the given frozen potential.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_harmonic_flux_matrix(diamonds: np.ndarray, nodes: np.ndarray,
                                  potential: np.ndarray, diffusivity: float,
                                  is_hole: bool) -> np.ndarray:
    """Assemble the harmonic-average flux matrix for one carrier.

    The exponential fitting factor is supplied by evaluate_bernoulli, which
    this function calls rather than reimplementing.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    potential : np.ndarray
        Nodal electrostatic potential of shape (n_nodes,), in units of the
        thermal voltage.
    diffusivity : float
        Scaled diffusion coefficient of the carrier (diffusivity > 0).
    is_hole : bool
        True for the hole equation, which reverses every Bernoulli argument,
        and False for the electron equation.

    Returns
    -------
    flux_matrix : np.ndarray
        Array of shape (n_nodes, n_nodes) which, applied to a nodal carrier
        density vector, returns the discrete flux balance of every unknown.

    Raises
    ------
    ValueError
        If the geometry, node table or potential has the wrong shape, the
        potential is non-finite, diffusivity is not finite and positive,
        is_hole is not boolean, a diamond area is non-positive, or a diamond
        node index lies outside the node table.
    """
    return flux_matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_harmonic_flux_matrix(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    potential: np.ndarray,
    diffusivity: float,
    is_hole: bool,
) -> np.ndarray:
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    potential = np.asarray(potential, dtype=float)

    if (
        diamonds.ndim != 2
        or diamonds.shape[1] != 9
        or diamonds.shape[0] < 1
    ):
        raise ValueError(
            "diamonds must be a 2D array with shape (n_diamonds, 9)"
        )

    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or nodes.shape[0] < 1
    ):
        raise ValueError(
            "nodes must be a 2D array with shape (n_nodes, 5)"
        )

    if potential.shape != (nodes.shape[0],):
        raise ValueError(
            "potential must have shape (n_nodes,)"
        )

    if not np.all(np.isfinite(potential)):
        raise ValueError(
            "potential must contain only finite values"
        )

    if not (
        isinstance(diffusivity, (int, float))
        and np.isfinite(diffusivity)
        and float(diffusivity) > 0.0
    ):
        raise ValueError(
            "diffusivity must be a finite number > 0"
        )

    if not isinstance(is_hole, (bool, np.bool_)):
        raise ValueError(
            "is_hole must be a boolean"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    if (
        diamonds[:, :4].max() >= nodes.shape[0]
        or diamonds[:, :4].min() < 0.0
    ):
        raise ValueError(
            "diamond node indices must lie inside the node table"
        )

    bernoulli = _oracle_evaluate_bernoulli

    def _scatter(primal_block, dual_block, boundary_term):
        """Scatter per-diamond blocks into the global matrix."""
        n_nodes = nodes.shape[0]
        matrix = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        code = nodes[:, 3]
        measure = nodes[:, 2]
        inverse = np.where(
            measure > 0.0,
            1.0 / np.where(measure > 0.0, measure, 1.0),
            0.0,
        )

        k = diamonds[:, 0].astype(int)
        l = diamonds[:, 1].astype(int)
        ks = diamonds[:, 2].astype(int)
        ls = diamonds[:, 3].astype(int)

        columns = np.column_stack([
            k,
            l,
            ks,
            ls,
        ])

        def _push(rows, mask, block, scale):
            if not np.any(mask):
                return

            target = np.repeat(rows[mask], 4)
            source = columns[mask].ravel()
            values = (
                block[mask] * scale[mask][:, None]
            ).ravel()

            np.add.at(
                matrix,
                (target, source),
                values,
            )

        unit = np.ones(
            diamonds.shape[0],
            dtype=float,
        )

        _push(
            k,
            code[k] == 0.0,
            primal_block,
            inverse[k],
        )
        _push(
            l,
            code[l] == 0.0,
            -primal_block,
            inverse[l],
        )
        _push(
            ks,
            np.isin(code[ks], (3.0, 5.0)),
            dual_block,
            inverse[ks],
        )
        _push(
            ls,
            np.isin(code[ls], (3.0, 5.0)),
            -dual_block,
            inverse[ls],
        )
        _push(
            l,
            code[l] == 2.0,
            primal_block,
            unit,
        )

        if boundary_term:
            on_boundary = diamonds[:, 8] > 0.5

            _push(
                ks,
                on_boundary
                & np.isin(code[ks], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[ks],
            )
            _push(
                ls,
                on_boundary
                & np.isin(code[ls], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[ls],
            )

        return matrix

    diffusivity = float(diffusivity)
    area = diamonds[:, 4]
    len_sigma = diamonds[:, 5]
    len_sigma_star = diamonds[:, 6]
    normal_dot = diamonds[:, 7]

    coeff_primal = (
        diffusivity
        * len_sigma
        * len_sigma
        / (2.0 * area)
    )
    coeff_cross = (
        diffusivity
        * len_sigma
        * len_sigma_star
        * normal_dot
        / (2.0 * area)
    )
    coeff_dual = (
        diffusivity
        * len_sigma_star
        * len_sigma_star
        / (2.0 * area)
    )

    cell_k = diamonds[:, 0].astype(int)
    cell_l = diamonds[:, 1].astype(int)
    dual_k = diamonds[:, 2].astype(int)
    dual_l = diamonds[:, 3].astype(int)

    sign = -1.0 if bool(is_hole) else 1.0

    drop_primal = sign * (
        potential[cell_k] - potential[cell_l]
    )
    drop_dual = sign * (
        potential[dual_k] - potential[dual_l]
    )

    b_primal = bernoulli(drop_primal)
    b_primal_rev = bernoulli(-drop_primal)
    b_dual = bernoulli(drop_dual)
    b_dual_rev = bernoulli(-drop_dual)

    primal_block = np.column_stack([
        coeff_primal * b_primal,
        -coeff_primal * b_primal_rev,
        coeff_cross * b_dual,
        -coeff_cross * b_dual_rev,
    ])

    dual_block = np.column_stack([
        coeff_cross * b_primal,
        -coeff_cross * b_primal_rev,
        coeff_dual * b_dual,
        -coeff_dual * b_dual_rev,
    ])

    return _scatter(
        primal_block,
        dual_block,
        False,
    )

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
def make_tables(seed, n_cells, n_bnd, n_dual, n_diamonds, boundary_fraction, spread):
    rng = np.random.default_rng(seed)
    n_nodes = n_cells + n_bnd + n_dual
    nodes = np.zeros((n_nodes, 5))
    nodes[:, 0] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:, 1] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:n_cells, 2] = rng.uniform(0.05, 0.20, n_cells)
    nodes[n_cells + n_bnd:, 2] = rng.uniform(0.05, 0.20, n_dual)
    nodes[:n_cells, 3] = 0.0
    nodes[n_cells:n_cells + n_bnd, 3] = rng.choice([1.0, 2.0], n_bnd)
    nodes[n_cells:n_cells + n_bnd, 4] = np.where(
        nodes[n_cells:n_cells + n_bnd, 3] == 1.0,
        rng.choice([1.0, 2.0], n_bnd), 0.0)
    nodes[n_cells + n_bnd:, 3] = rng.choice([3.0, 4.0, 5.0], n_dual)
    nodes[n_cells + n_bnd:, 4] = np.where(
        nodes[n_cells + n_bnd:, 3] == 4.0, rng.choice([1.0, 2.0], n_dual), 0.0)
    diamonds = np.zeros((n_diamonds, 9))
    diamonds[:, 0] = rng.integers(0, n_cells, n_diamonds)
    diamonds[:, 1] = rng.integers(0, n_cells + n_bnd, n_diamonds)
    diamonds[:, 2] = rng.integers(n_cells + n_bnd, n_nodes, n_diamonds)
    diamonds[:, 3] = rng.integers(n_cells + n_bnd, n_nodes, n_diamonds)
    diamonds[:, 4] = rng.uniform(0.01, 0.10, n_diamonds)
    diamonds[:, 5] = rng.uniform(0.10, 0.50, n_diamonds)
    diamonds[:, 6] = rng.uniform(0.10, 0.50, n_diamonds)
    diamonds[:, 7] = rng.uniform(-0.9, 0.9, n_diamonds)
    diamonds[:, 8] = (rng.random(n_diamonds) < boundary_fraction).astype(float)
    potential = rng.uniform(-spread, spread, n_nodes)
    return diamonds, nodes, potential
"""
    return [
        # --- Valid: electrons at a potential spread typical of a biased junction ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(0, 8, 5, 10, 20, 0.3, 12.0)
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: holes on the same configuration, which reverses every argument ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(0, 8, 5, 10, 20, 0.3, 12.0)
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 0.3320394832207, True)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 0.3320394832207, True)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: a larger configuration with more boundary diamonds ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(7, 20, 9, 24, 60, 0.5, 8.0)
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: a constant potential, so every Bernoulli argument is zero ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(3, 10, 4, 12, 25, 0.3, 5.0)
potential = np.full(nodes.shape[0], 11.4291488)
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: orthogonal diamonds, so the scheme decouples into two SG fluxes ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(11, 9, 6, 11, 22, 0.4, 12.0)
diamonds[:, 7] = 0.0
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, False)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: potential drops large enough to saturate the exponential fitting ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(5, 8, 5, 10, 20, 0.3, 400.0)
""",
            "call": "float(1.0e12 * (_a := np.ravel(assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, True)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
            "gold_call": "float(1.0e12 * (_a := np.ravel(_oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 1.0, True)))[0] + np.sum(_a * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: non-positive diffusivity ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(1, 8, 5, 10, 20, 0.3, 12.0)
def run_model():
    try:
        assemble_harmonic_flux_matrix(diamonds, nodes, potential, 0.0, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_harmonic_flux_matrix(diamonds, nodes, potential, 0.0, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: potential vector of the wrong length ---
        {
            "setup": synthetic + """
diamonds, nodes, potential = make_tables(2, 8, 5, 10, 20, 0.3, 12.0)
short = potential[:-1]
def run_model():
    try:
        assemble_harmonic_flux_matrix(diamonds, nodes, short, 1.0, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_harmonic_flux_matrix(diamonds, nodes, short, 1.0, False)
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
