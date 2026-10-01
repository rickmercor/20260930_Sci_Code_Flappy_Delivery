"""
Assemble the discrete duality Laplacian, the matrix that maps the nodal potential to the divergence of its diamond-reconstructed gradient on both the primal and the dual mesh.

The Poisson equation of the drift-diffusion system needs a discrete Laplacian that works on the same doubly represented unknown vector as the rest of the scheme, and the discrete duality framework builds it in two stages. First a gradient is reconstructed on each diamond cell as a weighted combination of the primal difference and the dual difference across it, each weighted by the length of the corresponding edge and directed along the corresponding unit normal, the whole divided by twice the diamond area. The essential point is that this reconstruction uses both differences: a formula built from the primal difference alone would need the primal edge to be perpendicular to the line joining the two cell centres, which is exactly the orthogonality requirement that forces a Voronoi dual and hence a Delaunay primal mesh. Second the reconstructed gradient is contracted with the outward normal of each control volume and summed, once over the primal edges of a primal cell and once over the dual edges of a dual cell, and divided by the measure of that cell.




Two structural details of the divergence are easy to miss and both change the assembled matrix. A primal cell and its neighbour see the same diamond with opposite outward normals, so the two contributions differ by a sign, and the same holds for the two dual cells of a diamond; getting one of the four signs wrong destroys the conservation property that the framework exists to provide. And a dual cell touching the boundary is not closed by dual edges alone: half of the primal boundary edge belongs to it, so its divergence carries an extra term equal to the flux through that half edge, without which the dual mesh no longer tiles the domain and the discrete Green formula that makes the primal divergence and the dual gradient adjoint fails.




The rows attached to contact-free boundary edges are not balance equations at all. A boundary edge is a degenerate primal cell of zero measure, so the ordinary divergence cannot be formed on it, and the equation that closes the system there is the homogeneous Neumann condition, namely that the reconstructed gradient contracted with the outward normal of the domain vanishes on the single boundary diamond that edge belongs to. That row therefore holds the very same primal-edge contraction the divergence is built from, weighted by the primal edge length and oriented along n_KL, but divided by no cell measure. Rows belonging to Dirichlet nodes receive no contribution at all and are left as zeros, to be overwritten later when the boundary values are imposed.




Because the Neumann rows force the normal gradient to vanish on every contact-free boundary diamond, the extra half-edge term carried by the boundary dual cells evaluates to zero at any solution of the assembled system. It is retained regardless, both because it is what the divergence operator of the framework actually is and because omitting it changes the Jacobian and therefore the path the Newton iteration takes.

Returns
-------
np.ndarray of shape (n_nodes, n_nodes), float: the discrete duality Laplacian acting on the nodal potential vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_ddfv_laplacian(diamonds: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Assemble the discrete duality Laplacian of the potential.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.

    Returns
    -------
    laplacian : np.ndarray
        Array of shape (n_nodes, n_nodes). Applied to a nodal potential
        vector it returns the divergence of the diamond-reconstructed
        gradient on interior primal and dual rows, the outward normal
        gradient on contact-free boundary edge rows, and zeros on Dirichlet
        rows, which are overwritten downstream.

    Raises
    ------
    ValueError
        If either table has the wrong shape, a diamond has non-positive area,
        or a diamond node index lies outside the node table.
    """
    return laplacian  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_ddfv_laplacian(
    diamonds: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)

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

        cell_k = diamonds[:, 0].astype(int)
        cell_l = diamonds[:, 1].astype(int)
        dual_k = diamonds[:, 2].astype(int)
        dual_l = diamonds[:, 3].astype(int)

        columns = np.column_stack([
            cell_k,
            cell_l,
            dual_k,
            dual_l,
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
            cell_k,
            code[cell_k] == 0.0,
            primal_block,
            inverse[cell_k],
        )
        _push(
            cell_l,
            code[cell_l] == 0.0,
            -primal_block,
            inverse[cell_l],
        )
        _push(
            dual_k,
            np.isin(code[dual_k], (3.0, 5.0)),
            dual_block,
            inverse[dual_k],
        )
        _push(
            dual_l,
            np.isin(code[dual_l], (3.0, 5.0)),
            -dual_block,
            inverse[dual_l],
        )
        _push(
            cell_l,
            code[cell_l] == 2.0,
            primal_block,
            unit,
        )

        if boundary_term:
            on_boundary = diamonds[:, 8] > 0.5

            _push(
                dual_k,
                on_boundary
                & np.isin(code[dual_k], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[dual_k],
            )
            _push(
                dual_l,
                on_boundary
                & np.isin(code[dual_l], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[dual_l],
            )

        return matrix

    area = diamonds[:, 4]
    len_sigma = diamonds[:, 5]
    len_sigma_star = diamonds[:, 6]
    normal_dot = diamonds[:, 7]

    primal = (
        len_sigma * len_sigma
        / (2.0 * area)
    )
    cross = (
        len_sigma
        * len_sigma_star
        * normal_dot
        / (2.0 * area)
    )
    dual = (
        len_sigma_star * len_sigma_star
        / (2.0 * area)
    )

    primal_block = np.column_stack([
        -primal,
        primal,
        -cross,
        cross,
    ])
    dual_block = np.column_stack([
        -cross,
        cross,
        -dual,
        dual,
    ])

    return _scatter(
        primal_block,
        dual_block,
        True,
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
def make_tables(seed, n_cells, n_bnd, n_dual, n_diamonds, boundary_fraction):
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
    return diamonds, nodes
"""
    return [
        # --- Valid: a mixed configuration exercising every equation class ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(0, 8, 5, 10, 20, 0.3)
""",
            "call": "float(np.sum((_a := np.ravel(assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: a larger configuration with more boundary diamonds ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(7, 20, 9, 24, 60, 0.5)
""",
            "call": "float(np.sum((_a := np.ravel(assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: no boundary diamonds, so the half-edge term never fires ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(3, 10, 4, 12, 25, 0.0)
""",
            "call": "float(np.sum((_a := np.ravel(assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: orthogonal diamonds, so the cross coupling vanishes exactly ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(11, 9, 6, 11, 22, 0.4)
diamonds[:, 7] = 0.0
""",
            "call": "float(np.sum((_a := np.ravel(assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: a single diamond and the smallest possible node table ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(5, 2, 1, 2, 1, 1.0)
""",
            "call": "float(np.sum((_a := np.ravel(assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_assemble_ddfv_laplacian(diamonds, nodes))) * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: a diamond of zero area ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(1, 8, 5, 10, 20, 0.3)
diamonds[2, 4] = 0.0
def run_model():
    try:
        assemble_ddfv_laplacian(diamonds, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_ddfv_laplacian(diamonds, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: diamond table with the wrong column count ---
        {
            "setup": """import numpy as np
diamonds = np.ones((5, 7))
nodes = np.zeros((10, 5))
def run_model():
    try:
        assemble_ddfv_laplacian(diamonds, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_ddfv_laplacian(diamonds, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a diamond index outside the node table ---
        {
            "setup": synthetic + """
diamonds, nodes = make_tables(2, 8, 5, 10, 20, 0.3)
diamonds[0, 3] = float(nodes.shape[0])
def run_model():
    try:
        assemble_ddfv_laplacian(diamonds, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_ddfv_laplacian(diamonds, nodes)
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
