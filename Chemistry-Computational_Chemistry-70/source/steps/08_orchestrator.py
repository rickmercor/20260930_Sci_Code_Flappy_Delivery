"""
Implement the function that ties every earlier step together into the single

scalar this problem asks for: build the shared Lennard-Jones initial

condition, generate the exact reference trajectory, independently generate

the RBL-approximated trajectory and train the DINaMo-style physics-only

trajectory, compute each approximate solver's whole-trajectory position RMSE

against the exact reference, and return the base-10 logarithm of the ratio

RMSE_RBL / RMSE_DINaMo.

RBMD 2.0's random batch list estimator and DINaMo's physics-informed neural

solver approximate the same microcanonical initial-value problem from

opposite directions -- RBL keeps the integrator exact and randomizes the

force, DINaMo keeps the force exact and replaces the integrator with a

learned function -- so the only way to compare them on equal footing is to

evaluate both against one shared exact reference trajectory generated from

one shared initial condition. log10(RMSE_RBL / RMSE_DINaMo) is negative when

RBL is the more accurate approximation on this system and positive when

DINaMo is; its magnitude reports how many orders of magnitude separate the

two approaches on this specific short-horizon Lennard-Jones argon problem.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def orchestrator(n_atoms: int = 50, density: float = 0.15,
                  temperature: float = 2.50487911765, pos_seed: int = 12345,
                  vel_seed: int = 987654, n_steps: int = 240, dt: float = 5e-4,
                  cutoff: float = 2.937, core_cutoff: float = 1.5,
                  batch_size: int = 10, rbl_seed: int = 20260911,
                  hidden_width: int = 64, n_epochs: int = 3000, lr: float = 1e-3,
                  dinamo_seed: int = 0, epsilon: float = 1.0, sigma: float = 1.0,
                  min_separation: float = 0.8, minimize_steps: int = 100,
                  burn_in_steps: int = 1000, rescale_interval: int = 20) -> float:
    """Orchestrate the full RBL-vs-DINaMo comparison and return log10 of the
    RMSE ratio.
    ...

    Raises
    ------
    ValueError
        If any parameter is invalid for the step it configures (see
        `generate_initial_condition`, `generate_reference_trajectory`,
        `generate_rbl_trajectory`, and `train_dinamo_trajectory` for the
        exact conditions on each shared parameter), or if the computed
        RMSE_DINaMo is 0 (e.g. because `n_steps` is 0), making
        log10(RMSE_RBL / RMSE_DINaMo) undefined.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_orchestrator(n_atoms: int = 50, density: float = 0.15,
                          temperature: float = 2.50487911765, pos_seed: int = 12345,
                          vel_seed: int = 987654, n_steps: int = 240, dt: float = 5e-4,
                          cutoff: float = 2.937, core_cutoff: float = 1.5,
                          batch_size: int = 10, rbl_seed: int = 20260911,
                          hidden_width: int = 64, n_epochs: int = 3000, lr: float = 1e-3,
                          dinamo_seed: int = 0, epsilon: float = 1.0, sigma: float = 1.0,
                          min_separation: float = 0.8, minimize_steps: int = 100,
                          burn_in_steps: int = 1000, rescale_interval: int = 20) -> float:
    """Reference implementation."""
    import numpy as np

    positions0, velocities0, box_length = _oracle_generate_initial_condition(
        n_atoms, density, temperature, pos_seed, vel_seed,
        min_separation=min_separation, cutoff=cutoff, epsilon=epsilon, sigma=sigma,
        minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, dt=dt,
        rescale_interval=rescale_interval,
    )

    reference_trajectory = _oracle_generate_reference_trajectory(
        positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff,
        epsilon=epsilon, sigma=sigma,
    )

    rbl_trajectory = _oracle_generate_rbl_trajectory(
        positions0, velocities0, box_length, n_steps, core_cutoff, cutoff,
        batch_size, rbl_seed, dt=dt, epsilon=epsilon, sigma=sigma,
    )

    dinamo_trajectory = _oracle_train_dinamo_trajectory(
        positions0, velocities0, box_length, n_steps, dt=dt, cutoff=cutoff,
        epsilon=epsilon, sigma=sigma, hidden_width=hidden_width, n_epochs=n_epochs,
        lr=lr, seed=dinamo_seed,
    )

    rmse_rbl = _oracle_trajectory_rmse(rbl_trajectory, reference_trajectory, box_length)
    rmse_dinamo = _oracle_trajectory_rmse(dinamo_trajectory, reference_trajectory, box_length)

    if rmse_dinamo <= 0.0:
        raise ValueError(
            "RMSE_DINaMo is 0 (e.g. because n_steps = 0), so log10(RMSE_RBL / "
            "RMSE_DINaMo) is undefined."
        )

    with np.errstate(divide="ignore"):
        return float(np.log10(rmse_rbl / rmse_dinamo))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: small system and short window, but every
            # stage (relaxation, burn-in, exact reference, RBL, DINaMo
            # training) is actually exercised. core_cutoff/cutoff are chosen
            # so that most particles have more shell neighbors than
            # batch_size, so RBL subsampling genuinely occurs (rather than
            # degenerating to the exact-sum branch every time).
            "setup": (
                "n_atoms = 8\n"
                "density = 0.5\n"
                "temperature = 1.5\n"
                "pos_seed = 1\n"
                "vel_seed = 2\n"
                "n_steps = 5\n"
                "dt = 1e-3\n"
                "cutoff = 1.25\n"
                "core_cutoff = 0.9\n"
                "batch_size = 1\n"
                "rbl_seed = 10\n"
                "hidden_width = 8\n"
                "n_epochs = 20\n"
                "lr = 1e-2\n"
                "dinamo_seed = 0\n"
                "min_separation = 0.4\n"
                "minimize_steps = 20\n"
                "burn_in_steps = 20\n"
                "rescale_interval = 5\n"
            ),
            "call": (
                "orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
            "gold_call": (
                "_oracle_orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
        },
        {
            # Boundary case: the minimum allowed n_atoms (2).
            "setup": (
                "n_atoms = 2\n"
                "density = 0.2\n"
                "temperature = 1.0\n"
                "pos_seed = 5\n"
                "vel_seed = 6\n"
                "n_steps = 3\n"
                "dt = 1e-3\n"
                "cutoff = 1.0\n"
                "core_cutoff = 0.5\n"
                "batch_size = 1\n"
                "rbl_seed = 11\n"
                "hidden_width = 4\n"
                "n_epochs = 10\n"
                "lr = 1e-2\n"
                "dinamo_seed = 1\n"
                "min_separation = 0.4\n"
                "minimize_steps = 10\n"
                "burn_in_steps = 10\n"
                "rescale_interval = 5\n"
            ),
            "call": (
                "orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
            "gold_call": (
                "_oracle_orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
        },
        {
            # Edge case: batch_size = 1 (maximal RBL subsampling, and most
            # particles here have more than 1 shell neighbor so subsampling
            # is genuinely exercised) combined with n_epochs = 0 (untrained
            # DINaMo, pure analytic ansatz) -- both solvers pushed to an
            # extreme but still well-defined.
            "setup": (
                "n_atoms = 5\n"
                "density = 0.25\n"
                "temperature = 1.2\n"
                "pos_seed = 3\n"
                "vel_seed = 4\n"
                "n_steps = 4\n"
                "dt = 1e-3\n"
                "cutoff = 1.3\n"
                "core_cutoff = 0.6\n"
                "batch_size = 1\n"
                "rbl_seed = 12\n"
                "hidden_width = 4\n"
                "n_epochs = 0\n"
                "lr = 1e-2\n"
                "dinamo_seed = 2\n"
                "min_separation = 0.4\n"
                "minimize_steps = 20\n"
                "burn_in_steps = 20\n"
                "rescale_interval = 5\n"
            ),
            "call": (
                "orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
            "gold_call": (
                "_oracle_orchestrator(n_atoms, density, temperature, pos_seed, vel_seed, "
                "n_steps, dt, cutoff, core_cutoff, batch_size, rbl_seed, hidden_width, "
                "n_epochs, lr, dinamo_seed, min_separation=min_separation, "
                "minimize_steps=minimize_steps, burn_in_steps=burn_in_steps, "
                "rescale_interval=rescale_interval)"
            ),
        },
    ]
