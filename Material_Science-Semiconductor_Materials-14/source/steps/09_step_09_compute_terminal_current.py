"""
Sum the converged harmonic-average fluxes over the boundary diamonds of one contact to obtain the terminal current there.

The terminal current is the observable that a device simulation exists to produce, because it is what a current-voltage curve is made of and what a designer compares against measurement. In a finite volume scheme it is not obtained by differentiating the computed fields but by summing the same discrete fluxes the scheme already balances, over the control volume faces that make up the contact. Doing it this way is what makes the computed current exactly consistent with the discrete conservation law rather than merely close to it.




The consistency has a precise consequence that is worth using as a check. Summing the discrete flux balance over every interior primal cell makes every interior face cancel against its neighbour, because the numerical flux is antisymmetric under exchanging the two cells, and leaves only the faces on the boundary. With the recombination rate set to zero the sum of all boundary fluxes must therefore vanish identically, and since the contact-free part of the boundary carries no flux by the Neumann condition, the current entering one contact must equal the current leaving the other to machine precision. A computed defect much larger than rounding is a reliable sign that a sign has been flipped somewhere in the assembly, or that the Neumann rows were not enforced, and it is a far sharper diagnostic than looking at the fields.




Both carriers contribute, with opposite signs. The electron particle flux and the electron current density point in opposite directions because the electron charge is negative, while for holes the two agree, so the total current through a face is the hole flux minus the electron flux in the convention where each flux is the one appearing in its own continuity equation. Reversing that relative sign gives many roughly the right magnitude at low injection, where one carrier dominates, and a badly wrong one at the injection levels a forward-biased junction actually reaches, which is a failure mode that leaves no other visible trace.




Only the primal boundary diamonds enter the sum. A contact is a set of boundary edges, each of which is a degenerate primal cell attached to exactly one diamond, and the flux across that diamond in the direction of the contact is the whole of the current through that edge; the dual cells sitting on the contact are Dirichlet nodes whose equations were replaced and which therefore carry no independent balance. The flux itself is the same harmonic-average expression used in the assembly, evaluated at the converged potential and densities, and it retains the cross term weighted by the scalar product of the two unit normals, so the reported current inherits the non-orthogonal coupling and is not simply the Scharfetter-Gummel flux of the primal edge.

Returns
-------
float, the total terminal current at the requested contact in scaled units, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_terminal_current(diamonds: np.ndarray, nodes: np.ndarray,
                             solution: np.ndarray, contact_tag: int) -> float:
    """Sum the converged fluxes over one contact to obtain its terminal current.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    solution : np.ndarray
        Converged state of shape (n_nodes, 3) holding the potential, the
        electron density and the hole density.
    contact_tag : int
        Which contact to sum over: 1 for the grounded cathode along y equal
        to zero, 2 for the biased anode along y equal to one.

    Returns
    -------
    current : float
        Total terminal current at that contact in scaled units, as a native
        Python float, positive when conventional current leaves the device
        through the contact.

    Raises
    ------
    ValueError
        If an input has the wrong shape, solution is non-finite, contact_tag
        is not 1 or 2, a diamond area is non-positive, or the requested
        contact has no boundary diamond.
    """
    return current  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_terminal_current(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    solution: np.ndarray,
    contact_tag: int,
) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    solution = np.asarray(solution, dtype=float)

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

    if solution.shape != (nodes.shape[0], 3):
        raise ValueError(
            "solution must have shape (n_nodes, 3)"
        )

    if not np.all(np.isfinite(solution)):
        raise ValueError(
            "solution must contain only finite values"
        )

    if not (
        isinstance(contact_tag, (int, np.integer))
        and not isinstance(contact_tag, bool)
        and int(contact_tag) in (1, 2)
    ):
        raise ValueError(
            "contact_tag must be the integer 1 or 2"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    def _bernoulli(t):
        """Bernoulli function B(t) = t / (exp(t) - 1)."""
        t = np.asarray(t, dtype=float)
        values = np.empty(t.shape, dtype=float)

        small = np.abs(t) < 1.0e-8
        large_pos = t > 500.0
        large_neg = t < -500.0
        middle = ~(
            small
            | large_pos
            | large_neg
        )

        ts = t[small]
        values[small] = (
            1.0
            - 0.5 * ts
            + ts * ts / 12.0
        )
        values[large_pos] = (
            t[large_pos]
            * np.exp(-t[large_pos])
        )
        values[large_neg] = -t[large_neg]

        tm = t[middle]
        values[middle] = (
            tm / np.expm1(tm)
        )

        return values

    scaled_hole_diffusivity = (
        12.16336 / 36.63227
    )

    contact_tag = int(contact_tag)
    potential, electrons, holes = solution.T

    cell_k = diamonds[:, 0].astype(int)
    cell_l = diamonds[:, 1].astype(int)
    dual_k = diamonds[:, 2].astype(int)
    dual_l = diamonds[:, 3].astype(int)

    selected = (
        (diamonds[:, 8] > 0.5)
        & (
            nodes[cell_l, 4]
            == float(contact_tag)
        )
    )

    if not np.any(selected):
        raise ValueError(
            "no boundary diamond belongs to the requested contact"
        )

    chosen = diamonds[selected]
    cell_k = cell_k[selected]
    cell_l = cell_l[selected]
    dual_k = dual_k[selected]
    dual_l = dual_l[selected]

    area = chosen[:, 4]
    len_sigma = chosen[:, 5]
    len_sigma_star = chosen[:, 6]
    normal_dot = chosen[:, 7]

    current = 0.0

    for diffusivity, is_hole, charge_sign in (
        (1.0, False, -1.0),
        (
            scaled_hole_diffusivity,
            True,
            1.0,
        ),
    ):
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

        density = (
            holes
            if is_hole
            else electrons
        )
        sign = (
            -1.0
            if is_hole
            else 1.0
        )

        drop_primal = sign * (
            potential[cell_k]
            - potential[cell_l]
        )
        drop_dual = sign * (
            potential[dual_k]
            - potential[dual_l]
        )

        flux = (
            coeff_primal
            * (
                _bernoulli(drop_primal)
                * density[cell_k]
                - _bernoulli(-drop_primal)
                * density[cell_l]
            )
            + coeff_cross
            * (
                _bernoulli(drop_dual)
                * density[dual_k]
                - _bernoulli(-drop_dual)
                * density[dual_l]
            )
        )

        current += (
            charge_sign
            * float(flux.sum())
        )

    return float(current)

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
def make_case(seed, n_cells, n_bnd, n_dual, n_diamonds):
    rng = np.random.default_rng(seed)
    n_nodes = n_cells + n_bnd + n_dual
    nodes = np.zeros((n_nodes, 5))
    nodes[:, 0] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:, 1] = rng.uniform(0.0, 1.0, n_nodes)
    nodes[:n_cells, 2] = rng.uniform(0.05, 0.20, n_cells)
    nodes[n_cells + n_bnd:, 2] = rng.uniform(0.05, 0.20, n_dual)
    nodes[n_cells:n_cells + n_bnd, 3] = 1.0
    nodes[n_cells:n_cells + n_bnd, 4] = np.where(
        np.arange(n_bnd) % 2 == 0, 1.0, 2.0)
    nodes[n_cells + n_bnd:, 3] = rng.choice([3.0, 4.0, 5.0], n_dual)
    diamonds = np.zeros((n_diamonds, 9))
    diamonds[:, 0] = rng.integers(0, n_cells, n_diamonds)
    diamonds[:, 1] = rng.integers(n_cells, n_cells + n_bnd, n_diamonds)
    diamonds[:, 2] = rng.integers(n_cells + n_bnd, n_nodes, n_diamonds)
    diamonds[:, 3] = rng.integers(n_cells + n_bnd, n_nodes, n_diamonds)
    diamonds[:, 4] = rng.uniform(0.01, 0.10, n_diamonds)
    diamonds[:, 5] = rng.uniform(0.10, 0.50, n_diamonds)
    diamonds[:, 6] = rng.uniform(0.10, 0.50, n_diamonds)
    diamonds[:, 7] = rng.uniform(-0.9, 0.9, n_diamonds)
    diamonds[:, 8] = 1.0
    solution = np.column_stack([
        rng.uniform(4.0, 12.0, n_nodes),
        rng.uniform(1.0e-9, 1.0, n_nodes),
        rng.uniform(1.0e-9, 1.0, n_nodes)])
    return diamonds, nodes, solution
"""
    return [
        # --- Valid: cathode current of a mixed configuration (normal scenario) ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(0, 8, 6, 10, 16)
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 1)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 1)",
        },
        # --- Valid: anode current of the same configuration ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(0, 8, 6, 10, 16)
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 2)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 2)",
        },
        # --- Valid: larger configuration with more contact faces ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(5, 20, 12, 24, 40)
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 1)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 1)",
        },
        # --- Boundary: interior diamonds present, which must be excluded from the sum ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(2, 10, 8, 12, 24)
diamonds[::3, 8] = 0.0
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 2)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 2)",
        },
        # --- Boundary: orthogonal diamonds, so the cross term drops out ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(3, 8, 6, 10, 16)
diamonds[:, 7] = 0.0
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 1)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 1)",
        },
        # --- Edge: a uniform potential, so every Bernoulli argument vanishes ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(4, 8, 6, 10, 16)
solution[:, 0] = 11.4291488
""",
            "call": "compute_terminal_current(diamonds, nodes, solution, 1)",
            "gold_call": "_oracle_compute_terminal_current(diamonds, nodes, solution, 1)",
        },
        # --- Invalid: contact tag outside the allowed pair ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(1, 8, 6, 10, 16)
def run_model():
    try:
        compute_terminal_current(diamonds, nodes, solution, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_terminal_current(diamonds, nodes, solution, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: solution array of the wrong width ---
        {
            "setup": synthetic + """
diamonds, nodes, solution = make_case(6, 8, 6, 10, 16)
bad = solution[:, :2]
def run_model():
    try:
        compute_terminal_current(diamonds, nodes, bad, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_terminal_current(diamonds, nodes, bad, 1)
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
