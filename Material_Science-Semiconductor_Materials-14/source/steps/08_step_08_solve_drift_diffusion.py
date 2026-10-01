"""
Solve the coupled nonlinear drift-diffusion system by voltage continuation and full Newton iteration, calling the assembly functions of sub-problems 04 to 07 and returning the converged potential and carrier densities.

The discrete system is nonlinear in two distinct ways. The space charge term of the Poisson equation is linear in the carrier densities, but the carrier densities themselves respond exponentially to the potential, and the flux matrices depend on the potential through a Bernoulli function on every edge. A fixed point iteration that alternates between the two, which is what a Gummel map does, converges but slowly and unreliably at the injection levels a forward-biased junction reaches. Full Newton on the coupled residual converges quadratically instead, at the price of an analytic Jacobian whose off-diagonal blocks are the derivatives of the fluxes with respect to the potential. Those derivatives need the derivative of the Bernoulli function, which has the same removable singularity at the origin as the function itself and is handled by the same branch structure, with the value minus one half at zero and the asymptotes one minus the argument times the decaying exponential for large positive argument and minus one for large negative argument.

Newton from an arbitrary starting point diverges on this problem, because a potential error of a few thermal voltages changes the carrier densities by orders of magnitude and the linear model becomes worthless. Two devices make it robust. The first is the initial guess: at zero applied bias, the charge-neutral equilibrium state, in which the densities satisfy neutrality and the mass action law and the potential is the logarithm of the electron density over the intrinsic density, is close enough to the true zero-bias solution that Newton reaches it in a handful of steps. The second is voltage continuation: the target bias is approached through a sequence of smaller biases, each solved from the previous converged state, so that the iteration never has to cross the region where the junction switches from depletion to injection in a single step. Because every intermediate problem is driven to convergence, the final answer is a property of the discrete system alone and does not depend on how many continuation steps were used.

Everything this step needs apart from the Jacobian already exists as an earlier sub-problem, so it composes them: the contact state and the charge-neutral initial guess come from sub-problem 04, the Laplacian from sub-problem 05, the two flux matrices at the frozen potential from sub-problem 06 and the stacked residual from sub-problem 07. Only the Jacobian is assembled here, because the derivative of the Bernoulli function has no sub-problem of its own.

Only the Dirichlet entries of the iterate are reset when the bias advances. Everything else is carried over, which is what makes continuation cheap. Within a step the Jacobian is assembled block by block: the dimensionless Debye group times the Laplacian and the two identity blocks of the space charge in the Poisson rows, the potential derivative of each flux contracted with the current density in the carrier rows, and the flux matrices themselves on the diagonal, with the Neumann rows carrying the raw normal contraction and the Dirichlet rows replaced by identity rows. Solving the resulting linear system gives a full Newton update which is applied without damping.

A Newton update can overshoot a carrier density into negative values, so the densities are truncated at a small positive floor after each update. On a converged solution of this device, the truncation never activates, the minority density bottoming out around 1e-10 in scaled units, ten orders of magnitude above the floor, so it is a safeguard rather than a part of the discretization. The convergence criterion is the maximum absolute entry of the scaled residual, and at the forward bias of interest the achievable floor is a few times 1e-12, so a tolerance of 1e-10 is reached comfortably and quadratically. That floor is not universal: under reverse bias, the minority densities collapse by several orders of magnitude while the flux coefficients do not, so the carrier balance becomes a difference of comparatively large terms, its attainable absolute residual rises into the 1e-10 range, and a tolerance suited to forward bias can no longer be met.

Returns
-------
np.ndarray of shape (n_nodes, 3), float: the converged nodal potential, electron density and hole density.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_drift_diffusion(diamonds: np.ndarray, nodes: np.ndarray,
                          anode_voltage: float, n_steps: int = 4,
                          tolerance: float = 1.0e-10,
                          max_iterations: int = 50) -> np.ndarray:
    """Solve the coupled drift-diffusion system to the requested residual.

    The contact state, the Laplacian, the two flux matrices and the residual
    are supplied by build_junction_state, assemble_ddfv_laplacian,
    assemble_harmonic_flux_matrix and assemble_coupled_residual, which this
    function calls rather than reimplementing. Only the Jacobian, which needs
    the derivative of the Bernoulli function and has no sub-problem of its
    own, is assembled here.

    Parameters
    ----------
    diamonds : np.ndarray
        Diamond geometry table of shape (n_diamonds, 9) as returned by
        build_ddfv_mesh.
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table.
    anode_voltage : float
        Target voltage applied at the anode contact, in volts.
    n_steps : int
        Number of equal voltage continuation steps used to reach the target
        bias from zero (n_steps >= 1).
    tolerance : float
        Convergence threshold on the maximum absolute scaled residual
        (tolerance > 0).
    max_iterations : int
        Maximum Newton iterations allowed per continuation step
        (max_iterations >= 1).

    Returns
    -------
    solution : np.ndarray
        Array of shape (n_nodes, 3) holding the converged potential in
        thermal voltages, the electron density and the hole density, both in
        units of the reference doping.

    Raises
    ------
    ValueError
        If an input shape or scalar bound is invalid, a diamond area is
        non-positive, the Newton system is singular or non-finite, or the
        requested tolerance is not reached within max_iterations.
    """
    return solution  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_drift_diffusion(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    anode_voltage: float,
    n_steps: int = 4,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
) -> np.ndarray:
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

    if not (
        isinstance(anode_voltage, (int, float))
        and np.isfinite(anode_voltage)
    ):
        raise ValueError(
            "anode_voltage must be a finite number"
        )

    if not (
        isinstance(n_steps, (int, np.integer))
        and not isinstance(n_steps, bool)
        and int(n_steps) >= 1
    ):
        raise ValueError(
            "n_steps must be an integer >= 1"
        )

    if not (
        isinstance(tolerance, (int, float))
        and np.isfinite(tolerance)
        and float(tolerance) > 0.0
    ):
        raise ValueError(
            "tolerance must be a finite number > 0"
        )

    if not (
        isinstance(max_iterations, (int, np.integer))
        and not isinstance(max_iterations, bool)
        and int(max_iterations) >= 1
    ):
        raise ValueError(
            "max_iterations must be an integer >= 1"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    anode_voltage = float(anode_voltage)
    n_steps = int(n_steps)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    junction_state = _oracle_build_junction_state
    build_laplacian = _oracle_assemble_ddfv_laplacian
    flux_matrix = _oracle_assemble_harmonic_flux_matrix
    coupled_residual = _oracle_assemble_coupled_residual

    elementary_charge = 1.602192e-19
    permittivity = 1.035941e-12
    thermal_voltage = 0.025852
    reference_density = 1.0e15
    length_unit = 1.0e-4
    scaled_hole_diffusivity = 12.16336 / 36.63227

    debye_group = (
        permittivity
        * thermal_voltage
        / (
            elementary_charge
            * reference_density
            * length_unit ** 2
        )
    )

    density_floor = 1.0e-20

    n_nodes = nodes.shape[0]
    code = nodes[:, 3]
    measure = nodes[:, 2]
    dirichlet = np.isin(code, (1.0, 4.0))
    neumann = code == 2.0
    identity = np.eye(n_nodes)

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

    inverse_measure = np.where(
        measure > 0.0,
        1.0 / np.where(
            measure > 0.0,
            measure,
            1.0,
        ),
        0.0,
    )

    unit = np.ones(
        diamonds.shape[0],
        dtype=float,
    )

    def _bernoulli_derivative(t):
        """First derivative of the Bernoulli function."""
        t = np.asarray(t, dtype=float)
        values = np.empty(t.shape, dtype=float)

        small = np.abs(t) < 1.0e-6
        large_pos = t > 500.0
        large_neg = t < -500.0
        middle = ~(
            small
            | large_pos
            | large_neg
        )

        values[small] = (
            -0.5
            + t[small] / 6.0
        )
        values[large_pos] = (
            1.0 - t[large_pos]
        ) * np.exp(-t[large_pos])
        values[large_neg] = -1.0

        tm = t[middle]
        expm = np.expm1(tm)
        values[middle] = (
            expm
            - tm * np.exp(tm)
        ) / (expm * expm)

        return values

    def _scatter(primal_block, dual_block):
        """Scatter per-diamond blocks into the global matrix."""
        matrix = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        def _push(rows, mask, block, scale):
            if not np.any(mask):
                return

            target = np.repeat(
                rows[mask],
                4,
            )
            source = columns[mask].ravel()
            values = (
                block[mask]
                * scale[mask][:, None]
            ).ravel()

            np.add.at(
                matrix,
                (target, source),
                values,
            )

        _push(
            cell_k,
            code[cell_k] == 0.0,
            primal_block,
            inverse_measure[cell_k],
        )
        _push(
            cell_l,
            code[cell_l] == 0.0,
            -primal_block,
            inverse_measure[cell_l],
        )
        _push(
            dual_k,
            np.isin(
                code[dual_k],
                (3.0, 5.0),
            ),
            dual_block,
            inverse_measure[dual_k],
        )
        _push(
            dual_l,
            np.isin(
                code[dual_l],
                (3.0, 5.0),
            ),
            -dual_block,
            inverse_measure[dual_l],
        )
        _push(
            cell_l,
            code[cell_l] == 2.0,
            primal_block,
            unit,
        )

        return matrix

    def _geometric_coefficients(diffusivity):
        """Return the primal, cross and dual coefficients."""
        area = diamonds[:, 4]
        len_sigma = diamonds[:, 5]
        len_sigma_star = diamonds[:, 6]
        normal_dot = diamonds[:, 7]

        return (
            diffusivity
            * len_sigma
            * len_sigma
            / (2.0 * area),
            diffusivity
            * len_sigma
            * len_sigma_star
            * normal_dot
            / (2.0 * area),
            diffusivity
            * len_sigma_star
            * len_sigma_star
            / (2.0 * area),
        )

    def _flux_potential_jacobian(
        potential,
        density,
        diffusivity,
        is_hole,
    ):
        """Derivative of the flux balance with respect to potential."""
        primal, cross, dual = _geometric_coefficients(
            diffusivity
        )

        sign = (
            -1.0
            if is_hole
            else 1.0
        )

        drop_p = sign * (
            potential[cell_k]
            - potential[cell_l]
        )
        drop_d = sign * (
            potential[dual_k]
            - potential[dual_l]
        )

        grad_p = sign * (
            _bernoulli_derivative(drop_p)
            * density[cell_k]
            + _bernoulli_derivative(-drop_p)
            * density[cell_l]
        )
        grad_d = sign * (
            _bernoulli_derivative(drop_d)
            * density[dual_k]
            + _bernoulli_derivative(-drop_d)
            * density[dual_l]
        )

        primal_block = np.column_stack([
            primal * grad_p,
            -primal * grad_p,
            cross * grad_d,
            -cross * grad_d,
        ])

        dual_block = np.column_stack([
            cross * grad_p,
            -cross * grad_p,
            dual * grad_d,
            -dual * grad_d,
        ])

        return _scatter(
            primal_block,
            dual_block,
        )

    laplacian = build_laplacian(
        diamonds,
        nodes,
    )

    initial = junction_state(
        nodes,
        0.0,
    )

    potential = initial[:, 3].copy()
    electrons = initial[:, 1].copy()
    holes = initial[:, 2].copy()

    for step in range(1, n_steps + 1):
        state = junction_state(
            nodes,
            anode_voltage * step / n_steps,
        )

        potential[dirichlet] = state[
            dirichlet,
            3,
        ]
        electrons[dirichlet] = state[
            dirichlet,
            1,
        ]
        holes[dirichlet] = state[
            dirichlet,
            2,
        ]

        converged = False

        for _ in range(max_iterations):
            electron_matrix = flux_matrix(
                diamonds,
                nodes,
                potential,
                1.0,
                False,
            )
            hole_matrix = flux_matrix(
                diamonds,
                nodes,
                potential,
                scaled_hole_diffusivity,
                True,
            )

            current = coupled_residual(
                nodes,
                state,
                laplacian,
                electron_matrix,
                hole_matrix,
                np.column_stack([
                    potential,
                    electrons,
                    holes,
                ]),
            )

            if np.max(
                np.abs(current)
            ) < tolerance:
                converged = True
                break

            jacobian = np.zeros(
                (
                    3 * n_nodes,
                    3 * n_nodes,
                ),
                dtype=float,
            )

            jacobian[
                :n_nodes,
                :n_nodes,
            ] = (
                debye_group
                * laplacian
            )

            jacobian[
                :n_nodes,
                n_nodes:2 * n_nodes,
            ] = -identity

            jacobian[
                :n_nodes,
                2 * n_nodes:,
            ] = identity

            jacobian[
                :n_nodes,
                :n_nodes,
            ][neumann] = laplacian[neumann]

            jacobian[
                :n_nodes,
                n_nodes:2 * n_nodes,
            ][neumann] = 0.0

            jacobian[
                :n_nodes,
                2 * n_nodes:,
            ][neumann] = 0.0

            jacobian[
                n_nodes:2 * n_nodes,
                :n_nodes,
            ] = _flux_potential_jacobian(
                potential,
                electrons,
                1.0,
                False,
            )

            jacobian[
                n_nodes:2 * n_nodes,
                n_nodes:2 * n_nodes,
            ] = electron_matrix

            jacobian[
                2 * n_nodes:,
                :n_nodes,
            ] = _flux_potential_jacobian(
                potential,
                holes,
                scaled_hole_diffusivity,
                True,
            )

            jacobian[
                2 * n_nodes:,
                2 * n_nodes:,
            ] = hole_matrix

            rows = np.where(
                dirichlet
            )[0]

            for offset in (
                0,
                n_nodes,
                2 * n_nodes,
            ):
                jacobian[
                    rows + offset,
                    :,
                ] = 0.0

                jacobian[
                    rows + offset,
                    rows + offset,
                ] = 1.0

            try:
                update = np.linalg.solve(
                    jacobian,
                    -current,
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "singular Newton Jacobian"
                ) from exc

            if not np.all(
                np.isfinite(update)
            ):
                raise ValueError(
                    "Newton update is not finite"
                )

            potential = (
                potential
                + update[:n_nodes]
            )

            electrons = np.maximum(
                electrons
                + update[
                    n_nodes:2 * n_nodes
                ],
                density_floor,
            )

            holes = np.maximum(
                holes
                + update[
                    2 * n_nodes:
                ],
                density_floor,
            )

        if not converged:
            raise ValueError(
                "Newton iteration failed to reach the tolerance"
            )

    return np.column_stack([
        potential,
        electrons,
        holes,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    mesh = """import numpy as np
def build_mesh(nx, ny, distortion):
    dx, dy = 1.0 / nx, 1.0 / ny
    xy = np.zeros(((nx + 1) * (ny + 1), 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = i * dx, j * dy
            if 0 < i < nx and 0 < j < ny:
                s = 1.0 if (i + j) % 2 == 0 else -1.0
                x += distortion * dx * s
                y += distortion * dy * s
            xy[j * (nx + 1) + i] = (x, y)
    tris = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = (j + 1) * (nx + 1) + i
            tris.append((a, a + 1, b + 1))
            tris.append((a, b + 1, b))
    tris = np.asarray(tris, dtype=int)
    emap = {}
    for t, (a, b, c) in enumerate(tris):
        for (u, v) in ((a, b), (b, c), (c, a)):
            emap.setdefault((min(u, v), max(u, v)), []).append(t)
    inner = sorted(e for e, ts in emap.items() if len(ts) == 2)
    outer = sorted(e for e, ts in emap.items() if len(ts) == 1)
    n_tri, n_bnd, n_v = len(tris), len(outer), len(xy)
    bary = xy[tris].mean(axis=1)
    bidx = {e: k for k, e in enumerate(outer)}
    nodes = np.zeros((n_tri + n_bnd + n_v, 5))
    nodes[:n_tri, :2] = bary
    for e, k in bidx.items():
        nodes[n_tri + k, :2] = 0.5 * (xy[e[0]] + xy[e[1]])
    nodes[n_tri + n_bnd:, :2] = xy
    def area3(p, q, r):
        return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))
    for t, (a, b, c) in enumerate(tris):
        nodes[t, 2] = area3(xy[a], xy[b], xy[c])
    rows = []
    for e in inner + outer:
        ks, ls = e
        ts = emap[e]
        ck = ts[0]
        xk = bary[ck]
        if len(ts) == 2:
            cl, xl, flag = ts[1], bary[ts[1]], 0.0
        else:
            cl, xl, flag = n_tri + bidx[e], 0.5 * (xy[ks] + xy[ls]), 1.0
        nodes[n_tri + n_bnd + ks, 2] += area3(xk, xy[ks], xl)
        nodes[n_tri + n_bnd + ls, 2] += area3(xk, xl, xy[ls])
        d1 = xl - xk
        d2 = xy[ls] - xy[ks]
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        if cr < 0.0:
            ks, ls = ls, ks
            d2, cr = -d2, -cr
        ls_ = float(np.hypot(d2[0], d2[1]))
        lss = float(np.hypot(d1[0], d1[1]))
        nkl = np.array([-d2[1], d2[0]]) / ls_
        if nkl @ d1 < 0.0:
            nkl = -nkl
        nks = np.array([-d1[1], d1[0]]) / lss
        if nks @ d2 < 0.0:
            nks = -nks
        rows.append([ck, cl, n_tri + n_bnd + ks, n_tri + n_bnd + ls,
                     0.5 * abs(cr), ls_, lss, float(nkl @ nks), flag])
    diamonds = np.asarray(rows, dtype=float)
    for i in range(n_tri, n_tri + n_bnd):
        y = nodes[i, 1]
        if y == 0.0:
            nodes[i, 3], nodes[i, 4] = 1.0, 1.0
        elif y == 1.0:
            nodes[i, 3], nodes[i, 4] = 1.0, 2.0
        else:
            nodes[i, 3] = 2.0
    for i in range(n_tri + n_bnd, nodes.shape[0]):
        x, y = nodes[i, 0], nodes[i, 1]
        if y == 0.0:
            nodes[i, 3], nodes[i, 4] = 4.0, 1.0
        elif y == 1.0:
            nodes[i, 3], nodes[i, 4] = 4.0, 2.0
        elif x == 0.0 or x == 1.0:
            nodes[i, 3] = 5.0
        else:
            nodes[i, 3] = 3.0
    return diamonds, nodes
"""
    return [
        # --- Valid: small distorted mesh at forward bias (normal scenario) ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(3, 4, 0.40)
""",
            "call": "float(np.sum((_a := np.ravel(solve_drift_diffusion(diamonds, nodes, 0.4))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_solve_drift_diffusion(diamonds, nodes, 0.4))) * np.arange(1, _a.size + 1)))",
        },
        # --- Valid: undistorted mesh at a lower bias with fewer continuation steps ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(2, 2, 0.0)
""",
            "call": "float(np.sum((_a := np.ravel(solve_drift_diffusion(diamonds, nodes, 0.2, 2))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_solve_drift_diffusion(diamonds, nodes, 0.2, 2))) * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: zero applied bias, so the device stays at equilibrium ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(3, 4, 0.40)
""",
            "call": "float(np.sum((_a := np.ravel(solve_drift_diffusion(diamonds, nodes, 0.0, 1))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_solve_drift_diffusion(diamonds, nodes, 0.0, 1))) * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: reverse bias, where the minority densities collapse and the
        #     attainable residual floor rises above the forward-bias tolerance ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(3, 4, 0.40)
""",
            "call": "float(np.sum((_a := np.ravel(solve_drift_diffusion(diamonds, nodes, -0.3, 3, 1.0e-8))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_solve_drift_diffusion(diamonds, nodes, -0.3, 3, 1.0e-8))) * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: a single continuation step straight to the target bias ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(2, 3, 0.45)
""",
            "call": "float(np.sum((_a := np.ravel(solve_drift_diffusion(diamonds, nodes, 0.4, 1, 1.0e-10, 40))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_solve_drift_diffusion(diamonds, nodes, 0.4, 1, 1.0e-10, 40))) * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: zero continuation steps ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(2, 2, 0.4)
def run_model():
    try:
        solve_drift_diffusion(diamonds, nodes, 0.4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_drift_diffusion(diamonds, nodes, 0.4, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: iteration budget too small to converge ---
        {
            "setup": mesh + """
diamonds, nodes = build_mesh(3, 4, 0.4)
def run_model():
    try:
        solve_drift_diffusion(diamonds, nodes, 0.4, 1, 1.0e-10, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_drift_diffusion(diamonds, nodes, 0.4, 1, 1.0e-10, 1)
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
