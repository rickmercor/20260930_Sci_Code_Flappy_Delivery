"""
Compose every earlier step to obtain the smallest root-mean-square error, relative to the true derivative, of a jointly generated finite-difference estimator of a rate sensitivity of the processive phosphorylation network at a fixed path budget.
Orchestrator: yes - enumerates the reachable states (enumerate_processive_states), builds the channel generators (build_channel_generators), obtains the exact target derivative (compute_observable_derivatives), weights the stencil (build_stencil_weights), minimizes the exact mean square error over the perturbation size (minimize_stencil_rmse, which uses compute_stencil_moments) and re-evaluates the numerator moments at the optimum (compute_stencil_moments), consuming each output rather than reimplementing any step.

How accurately a sensitivity can be estimated for a given simulation effort depends jointly on the stencil, on how its paths are coupled and on how quickly the expected output bends as the perturbed rate constant moves.

Returns
-------
float: minimized exact root mean square error of the jointly generated stencil estimator divided by |D|.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_minimum_relative_rmse(
    n_sites: int = 5,
    initial_state: tuple = (3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    rates: tuple = (0.02, 0.2, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0,
                    1.5, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.15, 0.015),
    channel: int = 2,
    horizon: float = 30.0,
    offsets: tuple = (2.0, 1.0, -1.0, -2.0),
    derivative_order: int = 3,
    path_budget: int = 4096,
    eps_bounds: tuple = (0.01, 0.5),
) -> float:
    """Return the smallest root mean square error of the jointly generated stencil estimator divided by |D|.

    The network, species order and channel order are those of
    ``enumerate_processive_states``; the chain starts in ``initial_state``
    with rate vector ``rates`` and the observable is the copy number of
    ``S_0``. With ``g(theta) = E[S_0(T)]`` at ``T = horizon`` and
    ``D = d^k g / d theta[channel]^k`` at ``theta = rates``
    (``k = derivative_order``), each replication evaluates
    ``sum_r c_r S_0^{(r)}(T) / eps^k`` on ``J = len(offsets)`` paths generated
    jointly as in ``compute_stencil_moments``, with the weights ``c_r`` of
    ``build_stencil_weights``, and ``n = path_budget / J`` independent
    replications are averaged. Minimize the estimator's exact mean square
    error over ``eps`` in ``eps_bounds`` (as in ``minimize_stencil_rmse``) and
    return the square root of that minimum divided by ``|D|``. The defaults
    reproduce the problem statement.

    Parameters
    ----------
    n_sites : int
        Number of phosphorylation sites.
    initial_state : tuple
        Initial copy numbers, length ``2 * n_sites + 4``.
    rates : tuple
        Rate constants, length ``4 * n_sites + 2``.
    channel : int
        Zero-based index of the perturbed rate constant.
    horizon : float
        Observation time ``T``.
    offsets : tuple
        Stencil offsets ``a_r``.
    derivative_order : int
        Order ``k`` of the estimated derivative.
    path_budget : int
        Total number of simulated paths; a positive multiple of ``J``.
    eps_bounds : tuple
        Search interval ``(lo, hi)`` for ``eps``.

    Returns
    -------
    float
        Minimized root mean square error divided by ``|D|``.

    Raises
    ------
    ValueError
        For any invalid input of the earlier steps, if ``rates`` does not
        have ``4 * n_sites + 2`` entries, if ``path_budget`` is not a positive
        integer multiple of ``J``, or if ``D`` is zero.
    """
    return relative_rmse

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_minimum_relative_rmse(
    n_sites: int = 5,
    initial_state: tuple = (3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    rates: tuple = (0.02, 0.2, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0,
                    1.5, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8, 0.15, 0.015),
    channel: int = 2,
    horizon: float = 30.0,
    offsets: tuple = (2.0, 1.0, -1.0, -2.0),
    derivative_order: int = 3,
    path_budget: int = 4096,
    eps_bounds: tuple = (0.01, 0.5),
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    start_state = np.asarray(initial_state)
    states = _oracle_enumerate_processive_states(n_sites, start_state)
    generators = _oracle_build_channel_generators(n_sites, states)
    theta = np.asarray(rates, dtype=float)
    if theta.shape != (generators.shape[0],):
        raise ValueError("rates must have 4 * n_sites + 2 entries")
    start = int(np.flatnonzero(np.all(states == start_state.astype(np.int64), axis=1))[0])
    observable = states[:, 0].astype(float)
    k = derivative_order
    target = _oracle_compute_observable_derivatives(generators, theta, start, observable, horizon, channel, k)[k]
    if target == 0.0:
        raise ValueError("the target derivative is zero")
    paths = len(offsets)
    if isinstance(path_budget, bool) or not isinstance(path_budget, (int, np.integer)) or path_budget <= 0 \
            or path_budget % paths:
        raise ValueError("path_budget must be a positive multiple of the number of stencil paths")
    a = np.asarray(offsets, dtype=float)
    c = _oracle_build_stencil_weights(a, k)
    replications = float(path_budget // paths)
    eps_opt, _ = _oracle_minimize_stencil_rmse(generators, theta, start, observable, horizon, channel, a, c, k,
                                               replications, target, np.asarray(eps_bounds, dtype=float))
    # The error is re-assembled at the optimum from the numerator's own mean and second moment.
    mean, second = _oracle_compute_stencil_moments(generators, theta, start, observable, horizon, channel, a, c,
                                                   float(eps_opt))
    mse = (second - mean ** 2) / (replications * eps_opt ** (2 * k)) + (mean / eps_opt ** k - target) ** 2
    return float(np.sqrt(mse) / abs(target))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    pin = "import numpy as np\ndef _pin(value):\n    return float(20.0 + np.arcsinh(1.0e6 * float(value)))\n"
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        estimate_minimum_relative_rmse({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
        "def _reference():\n    try:\n        _oracle_estimate_minimum_relative_rmse({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n"
    )
    return [
        {   # Normal: one site, the third derivative from the four-point stencil.
            "setup": pin,
            "call": "_pin(estimate_minimum_relative_rmse(1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 2, 30.0, (2.0, 1.0, -1.0, -2.0), 3, 4096, (0.01, 1.0)))",
            "gold_call": "_pin(_oracle_estimate_minimum_relative_rmse(1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 2, 30.0, (2.0, 1.0, -1.0, -2.0), 3, 4096, (0.01, 1.0)))",
        },
        {   # Boundary: a two-path budget, so a single replication.
            "setup": pin,
            "call": "_pin(estimate_minimum_relative_rmse(1, (1, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 0, 30.0, (1.0, 0.0), 1, 2, (0.001, 0.5)))",
            "gold_call": "_pin(_oracle_estimate_minimum_relative_rmse(1, (1, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 0, 30.0, (1.0, 0.0), 1, 2, (0.001, 0.5)))",
        },
        {   # Edge: two sites and a four-point first-derivative stencil.
            "setup": pin,
            "call": "_pin(estimate_minimum_relative_rmse(2, (1, 0, 1, 1, 0, 0, 0, 0), (0.02, 0.2, 1.0, 0.5, 2.0, 1.5, 0.4, 0.8, 0.15, 0.015), 5, 10.0, (2.0, 1.0, -1.0, -2.0), 1, 1024, (0.01, 0.75)))",
            "gold_call": "_pin(_oracle_estimate_minimum_relative_rmse(2, (1, 0, 1, 1, 0, 0, 0, 0), (0.02, 0.2, 1.0, 0.5, 2.0, 1.5, 0.4, 0.8, 0.15, 0.015), 5, 10.0, (2.0, 1.0, -1.0, -2.0), 1, 1024, (0.01, 0.75)))",
        },
        {   # Edge: a second derivative from three copies over a short horizon.
            "setup": pin,
            "call": "_pin(estimate_minimum_relative_rmse(1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 3, 5.0, (0.0, 1.0, -1.0), 2, 300, (0.01, 1.5)))",
            "gold_call": "_pin(_oracle_estimate_minimum_relative_rmse(1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 3, 5.0, (0.0, 1.0, -1.0), 2, 300, (0.01, 1.5)))",
        },
        {   # Invalid: the budget is not a multiple of the number of stencil paths.
            "setup": raises.replace("{args}", "1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0, 1.5, 0.15, 0.015), 2, 30.0, (2.0, 1.0, -1.0, -2.0), 3, 4098, (0.01, 1.0)"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {   # Invalid: the rate vector has the wrong length.
            "setup": raises.replace("{args}", "1, (2, 0, 1, 1, 0, 0), (0.02, 0.2, 2.0), 2, 30.0, (2.0, 1.0, -1.0, -2.0), 3, 4096, (0.01, 1.0)"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
