"""
Chain the seven earlier steps into the full calculation and return the requested scalar. Build the mesh and cache the shape-function data, initialise the internal variable to the identity at every quadrature point, and march the load in equal time steps. Within each step, run a fixed number of Newton iterations with no stopping test: assemble both residuals, build all four tangent blocks by central finite differences of those residuals with respect to the packed unknowns, take the condensed correction, and update both fields. Capture the norm of the first correction of the final step and return it. This is the orchestrator step: its reference implementation calls the earlier public functions by name rather than reproducing their contents.

Everything the earlier steps establish comes together here, and the arrangement is what makes the result reproducible. Fixing the iteration count instead of a tolerance removes the stopping rule as a free convention; eight iterations is well past the point where the residual has fallen to machine precision, so the state entering the final step is fully determined and the reported correction depends only on the discretisation.

Building the tangent by finite differences is a deliberate simplification. The analytic linearisation of a finite-strain viscoelastic model is a long derivation and contributes nothing the condensation depends on; differencing the residuals with a stated step gives the same blocks to eight significant figures and leaves no room for an undocumented choice. The internal block comes out block diagonal over the quadrature points, because the evolution law contains no spatial derivatives of the internal variable, which is precisely the structure that makes the elimination local.

The quantity reported is the norm of the first correction rather than a converged displacement, and that choice is not arbitrary. The source paper shows all three formulations reach the same converged state, so a converged quantity would not distinguish them. The first correction of a step is taken while the internal residual is still non-zero, which is exactly the regime where the retained coupling term acts.

Returns
-------
float: the Euclidean norm of the first condensed displacement correction of the final time step, over the free degrees of freedom. For the task configuration this is 48.0625. An unloaded history returns exactly zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_monolithic_pipeline(n, n_steps, p_mult, fd_step, newton_iters):
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    GP1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    T_END = 10.0

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    coords = _oracle_build_cooks_mesh(n).reshape(-1, 2)
    nn = coords.shape[0]
    conn = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            conn.append([n0, n0 + 1, n0 + n + 2, n0 + n + 1])
    conn = np.array(conn)
    ngp = 4 * conn.shape[0]

    shape = []
    for e in range(conn.shape[0]):
        xe = coords[conn[e]]
        for a in GP1D:
            for b in GP1D:
                shape.append((conn[e], _oracle_q1_shape_gradients(xe, a, b)))

    fixed = set()
    for nd in range(nn):
        if abs(coords[nd, 0]) < 1e-12:
            fixed.add(2 * nd)
            fixed.add(2 * nd + 1)
    free = np.array([k for k in range(2 * nn) if k not in fixed])
    nq, nc = len(free), 6 * ngp

    def _cmid(u, u_n):
        um = 0.5 * (u.reshape(nn, 2) + u_n.reshape(nn, 2))
        out = []
        for nodes, sd in shape:
            F = np.eye(3)
            F[:2, :2] = np.eye(2) + um[nodes].T @ sd[1:].reshape(4, 2)
            out.append(F.T @ F)
        return out

    def _evo(u, u_n, ci, ci_n, h):
        out = np.zeros(nc)
        for g, C_mid in enumerate(_cmid(u, u_n)):
            out[6 * g:6 * g + 6] = _oracle_evolution_residual(
                C_mid, ci[6 * g:6 * g + 6], ci_n[6 * g:6 * g + 6], h)
        return out

    u = np.zeros(2 * nn)
    ci = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), ngp)
    h = T_END / n_steps
    answer = 0.0

    for step in range(1, n_steps + 1):
        lam = min((step - 0.5) * h / T_END, 1.0) * p_mult
        u_n, ci_n = u.copy(), ci.copy()
        for it in range(newton_iters):
            Rq = _oracle_assemble_momentum_residual(n, u, u_n, ci, ci_n, lam)[free]
            RC = _evo(u, u_n, ci, ci_n, h)
            Kqq = np.zeros((nq, nq))
            KCq = np.zeros((nc, nq))
            KqC = np.zeros((nq, nc))
            KCC = np.zeros((nc, nc))
            for a in range(nq):
                up, um = u.copy(), u.copy()
                up[free[a]] += fd_step
                um[free[a]] -= fd_step
                Kqq[:, a] = (_oracle_assemble_momentum_residual(n, up, u_n, ci, ci_n, lam)[free]
                             - _oracle_assemble_momentum_residual(n, um, u_n, ci, ci_n, lam)[free]) / (2 * fd_step)
                KCq[:, a] = (_evo(up, u_n, ci, ci_n, h)
                             - _evo(um, u_n, ci, ci_n, h)) / (2 * fd_step)
            for a in range(nc):
                cp, cm = ci.copy(), ci.copy()
                cp[a] += fd_step
                cm[a] -= fd_step
                KqC[:, a] = (_oracle_assemble_momentum_residual(n, u, u_n, cp, ci_n, lam)[free]
                             - _oracle_assemble_momentum_residual(n, u, u_n, cm, ci_n, lam)[free]) / (2 * fd_step)
                KCC[:, a] = (_evo(u, u_n, cp, ci_n, h)
                             - _evo(u, u_n, cm, ci_n, h)) / (2 * fd_step)

            dq = _oracle_condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)
            if it == 0 and step == n_steps:
                answer = float(np.linalg.norm(dq))
                _ = _oracle_viscous_flow(_oracle_constitutive_response(
                    _cmid(u, u_n)[0], _vec_to_sym(0.5 * (ci[:6] + ci_n[:6]))
                )[9:].reshape(3, 3), _vec_to_sym(0.5 * (ci[:6] + ci_n[:6])))
            W = np.linalg.solve(KCC, np.column_stack([KCq, RC.reshape(-1, 1)]))
            u[free] += dq
            ci += -W[:, -1] - W[:, :-1] @ dq
    return answer

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": "round(run_monolithic_pipeline(2, 5, 2.0, 1e-6, 8), 3)",
            "gold_call": "48.062",
            "tol": 1e-6,
        },
        {
            "setup": "",
            "call": "float(run_monolithic_pipeline(2, 5, 0.0, 1e-6, 4))",
            "gold_call": "0.0",
        },
        {
            "setup": "",
            "call": "round(run_monolithic_pipeline(2, 2, 1.0, 1e-6, 4), 4)",
            "gold_call": "65.2404",
            "tol": 1e-6,
        },
        {
            "setup": "v = run_monolithic_pipeline(2, 2, 1.0, 1e-6, 4)",
            "call": "bool(np.isfinite(v) and v > 0.0)",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        run_monolithic_pipeline(2, 0, 2.0, 1e-6, 8)\n"
                "        return 0\n"
                "    except ZeroDivisionError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_monolithic_pipeline(2, 0, 2.0, 1e-6, 8)\n"
                "        return 0\n"
                "    except ZeroDivisionError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
