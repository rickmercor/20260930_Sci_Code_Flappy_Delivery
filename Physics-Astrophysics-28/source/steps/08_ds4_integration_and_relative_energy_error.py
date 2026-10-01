"""
Chain the earlier sub-problems: evaluate H(z_0) with pn_hamiltonian, build the schedule with ds4_schedule, advance the doubled state through the 13 sub-flows with kepler_flow, pn_perturbation_gradients, doubled_flow_A and doubled_flow_B, apply doubled_projection after each step, and return |H(z_N) - H(z_0)|/|H(z_0)|.

For a symplectic scheme the energy error stays bounded and oscillates with orbital phase instead of drifting. Its size at a given time measures the local error constant of the specific composition, splitting and projection.

Returns
-------
float: relative Hamiltonian error |H(z_N) - H(z_0)| / |H(z_0)| after n_steps steps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_ds4_pn_energy_error(beta, eps, P0, Q0, h, n_steps, lam0, mu0):
    """Integrate the non-spinning 2PN binary with DS4 and return the relative energy error.

    Chains pn_hamiltonian, pn_perturbation_gradients, kepler_flow, doubled_flow_A,
    doubled_flow_B, doubled_projection and ds4_schedule. Returns
    |H(z_N) - H(z_0)| / |H(z_0)| as a float after n_steps steps of size h.
    Raises ValueError for beta <= 0, h <= 0 or n_steps < 1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_ds4_pn_energy_error(beta, eps, P0, Q0, h, n_steps, lam0, mu0):
    # Oracle (orchestrator): chains the seven earlier sub-problem oracles (_oracle_-prefixed twins).
    if beta <= 0.0:
        raise ValueError("mass ratio beta must be positive")
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    if int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    eta = beta / (1.0 + beta) ** 2
    P0 = np.asarray(P0, dtype=float).ravel()
    Q0 = np.asarray(Q0, dtype=float).ravel()
    d = P0.size
    H_start = _oracle_pn_hamiltonian(P0, Q0, eta, eps)
    sched = _oracle_ds4_schedule(h, eps)
    ops = "CBABCBABCBABC"
    z = np.concatenate([P0, Q0, P0, Q0])
    for n in range(1, int(n_steps) + 1):
        for op, dt in zip(ops, sched):
            p, q, x, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
            if op == "C":
                a = _oracle_kepler_flow(p, q, dt)
                b = _oracle_kepler_flow(x, y, dt)
                z = np.concatenate([a[:d], a[d:], b[:d], b[d:]])
            elif op == "A":
                z = _oracle_doubled_flow_A(z, dt, _oracle_pn_perturbation_gradients(p, y, eta, eps))
            else:
                z = _oracle_doubled_flow_B(z, dt, _oracle_pn_perturbation_gradients(x, q, eta, eps))
        z = _oracle_doubled_projection(z, n, lam0, mu0)
    H_end = _oracle_pn_hamiltonian(z[:d], z[d:2 * d], eta, eps)
    return float(abs(H_end - H_start) / abs(H_start))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_task_instance_500_steps_log_relative"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "np.log(\n"
                "    run_ds4_pn_energy_error(\n"
                "        0.28,\n"
                "        0.6309573444801932,\n"
                "        np.array([0.0, 0.18]),\n"
                "        np.array([25.34, 0.0]),\n"
                "        2.0,\n"
                "        500,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "gold_call": (
                "np.log(\n"
                "    _oracle_run_ds4_pn_energy_error(\n"
                "        0.28,\n"
                "        0.6309573444801932,\n"
                "        np.array([0.0, 0.18]),\n"
                "        np.array([25.34, 0.0]),\n"
                "        2.0,\n"
                "        500,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "tol": 0.0001,
        },
        {
            "name": (
                "boundary_weak_field_eps_0p1_h4_250_steps_log_relative"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "np.log(\n"
                "    run_ds4_pn_energy_error(\n"
                "        0.28,\n"
                "        0.1,\n"
                "        np.array([0.0, 0.18]),\n"
                "        np.array([25.34, 0.0]),\n"
                "        4.0,\n"
                "        250,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "gold_call": (
                "np.log(\n"
                "    _oracle_run_ds4_pn_energy_error(\n"
                "        0.28,\n"
                "        0.1,\n"
                "        np.array([0.0, 0.18]),\n"
                "        np.array([25.34, 0.0]),\n"
                "        4.0,\n"
                "        250,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "tol": 0.001,
        },
        {
            "name": (
                "edge_newtonian_limit_eps_zero_round_off_only"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "run_ds4_pn_energy_error(\n"
                "    0.28,\n"
                "    0.0,\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    2.0,\n"
                "    100,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_run_ds4_pn_energy_error(\n"
                "    0.28,\n"
                "    0.0,\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    2.0,\n"
                "    100,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "tol": 1e-12,
        },
        {
            "name": (
                "edge_equal_mass_3d_orbit_large_step_log_relative"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "np.log(\n"
                "    run_ds4_pn_energy_error(\n"
                "        1.0,\n"
                "        0.1,\n"
                "        np.array([0.02, 0.2, 0.03]),\n"
                "        np.array([20.0, 0.0, 1.0]),\n"
                "        4.0,\n"
                "        100,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "gold_call": (
                "np.log(\n"
                "    _oracle_run_ds4_pn_energy_error(\n"
                "        1.0,\n"
                "        0.1,\n"
                "        np.array([0.02, 0.2, 0.03]),\n"
                "        np.array([20.0, 0.0, 1.0]),\n"
                "        4.0,\n"
                "        100,\n"
                "        (1.0 / np.e),\n"
                "        (1.0 / np.pi),\n"
                "    ),\n"
                ")\n"
            ),
            "tol": 0.005,
        },
        {
            "name": (
                "invalid_zero_steps_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        run_ds4_pn_energy_error(\n"
                "            0.28,\n"
                "            0.6309573444801932,\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([25.34, 0.0]),\n"
                "            2.0,\n"
                "            0,\n"
                "            (1.0 / np.e),\n"
                "            (1.0 / np.pi),\n"
                "        )\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_ds4_pn_energy_error(\n"
                "            0.28,\n"
                "            0.6309573444801932,\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([25.34, 0.0]),\n"
                "            2.0,\n"
                "            0,\n"
                "            (1.0 / np.e),\n"
                "            (1.0 / np.pi),\n"
                "        )\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": (
                "run_model()\n"
            ),
            "gold_call": (
                "run_gold()\n"
            ),
            "tol": 0,
        },
    ]
