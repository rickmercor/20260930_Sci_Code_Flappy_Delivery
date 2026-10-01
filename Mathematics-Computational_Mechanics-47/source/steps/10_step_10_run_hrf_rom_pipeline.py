"""
Run the complete hyper-reduction-free reduced-order pipeline and return one requested scalar.

The workflow generates full-order training trajectories, extracts the

block-diagonal trial basis, precomputes the offline operator bank of the lifted

quadratic system, marches the reduced Newton solver over the longer online

horizon, and compares the reconstructed temperature with the unlifted

reference.

Returns
-------
float, the requested scalar diagnostic as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import math
import numpy as np


def run_hrf_rom_pipeline(
    quantity: str,
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    time_step: float,
    n_training_steps: int,
    n_online_steps: int,
    training_amplitudes: np.ndarray,
    test_amplitudes: np.ndarray,
    energy_tolerance: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> float:
    """Run the reduced-order pipeline end to end and report one scalar.

    The initial temperature on the interior nodes x_i = i * dx of
    (0, domain_length), with dx = domain_length / (n_nodes + 1), is

        q0(x) = s (1 - s) [6 (1 - s)**2 exp(-s) - 10 exp(s) sin(s / 6)] + s,
        s = x / domain_length.

    Call march_cubic_fom once per row of training_amplitudes over
    n_training_steps steps and concatenate the resulting trajectories
    column-wise into the temperature snapshot matrix. Pass that matrix to
    block_pod_basis to obtain the trial basis, build the operators of the
    lifted system with lifted_polynomial_operators, and precompute the offline
    bank with hrf_operator_gram. The initial reduced state is the projection of
    the lifted initial state onto the trial basis, the lifted initial state
    being the initial temperature followed by its pointwise square. Advance the
    reduced state with march_reduced_rom over n_online_steps steps at
    test_amplitudes, and obtain the reference trajectory by calling
    march_cubic_fom over the same horizon at test_amplitudes.

    Write Q for the reference temperature trajectory, Qt for the temperature
    block of the reconstructed reduced trajectory, and V for the temperature
    modes of the trial basis. The recognised values of quantity are

        "lspg_error_ratio"        squared Frobenius norm of Q - Qt under the
                                  least-squares scheme, divided by the squared
                                  Frobenius norm of (I - V V^T) Q
        "galerkin_error_ratio"    the same ratio under the Galerkin scheme
        "lspg_state_error"        squared Frobenius norm of Q - Qt under the
                                  least-squares scheme, divided by the squared
                                  Frobenius norm of Q
        "galerkin_state_error"    the same relative error under the Galerkin
                                  scheme
        "projection_error"        squared Frobenius norm of (I - V V^T) Q,
                                  divided by the squared Frobenius norm of Q
        "temperature_mode_count"  the number of retained temperature modes
        "auxiliary_mode_count"    the number of retained auxiliary modes
        "worst_excess_error_ratio" for a two-dimensional test_amplitudes array,
                                  compute both LSPG and Galerkin at every row
                                  and return the maximum of

                                  (E_lspg - E_projection)
                                  / (E_galerkin - E_projection),

                                  where each E is the corresponding squared
                                  Frobenius trajectory error

    Parameters
    ----------
    quantity : str
        One of the eight recognised names listed above.
    n_nodes : int
        Positive number of interior spatial nodes.
    domain_length : float
        Finite strictly positive length of the spatial domain.
    diffusivity : float
        Finite strictly positive diffusion coefficient.
    boundary_values : np.ndarray
        Finite real array of shape (2,) holding the Dirichlet values at
        x = 0 and x = domain_length, in that order.
    time_step : float
        Finite strictly positive step size shared by both horizons.
    n_training_steps : int
        Non-negative number of steps in each training trajectory.
    n_online_steps : int
        Non-negative number of steps in the online horizon.
    training_amplitudes : np.ndarray
        Finite real array of shape (n_train, 2) with n_train >= 1.
    test_amplitudes : np.ndarray
        Finite real array of shape (2,) for the first seven reporting names,
        or shape (n_test, 2) with n_test >= 1 for
        "worst_excess_error_ratio".
    energy_tolerance : float
        Finite tolerance in (0, 1) on the neglected fraction of squared
        singular values.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.
    residual_tolerance : float
        Finite strictly positive tolerance shared by both Newton solvers.
    max_newton_iterations : int
        Positive cap on iterations per time step.

    Returns
    -------
    value : float
        Native Python float equal to the requested scalar.

    Raises
    ------
    ValueError
        If quantity is not recognised, if any array has the wrong rank or
        shape, if any entry is not finite, or if a scalar control violates its
        stated sign or type requirement.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_hrf_rom_pipeline(
    quantity: str,
    n_nodes: int,
    domain_length: float,
    diffusivity: float,
    boundary_values: np.ndarray,
    time_step: float,
    n_training_steps: int,
    n_online_steps: int,
    training_amplitudes: np.ndarray,
    test_amplitudes: np.ndarray,
    energy_tolerance: float,
    alphas: np.ndarray,
    betas: np.ndarray,
    residual_tolerance: float,
    max_newton_iterations: int,
) -> float:
    """Reference implementation."""
    from numbers import Integral, Real

    import math
    import numpy as np

    def _is_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    recognised = ("lspg_error_ratio", "galerkin_error_ratio", "lspg_state_error",
                  "galerkin_state_error", "projection_error", "temperature_mode_count",
                  "auxiliary_mode_count", "worst_excess_error_ratio")
    if not isinstance(quantity, str) or quantity not in recognised:
        raise ValueError("quantity is not a recognised reporting name")
    if not _is_integer(n_nodes) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")

    training = np.asarray(training_amplitudes, dtype=float)
    if training.ndim != 2 or training.shape[0] < 1 or training.shape[1] != 2:
        raise ValueError("training_amplitudes must have shape (n_train, 2)")
    if not np.all(np.isfinite(training)):
        raise ValueError("training_amplitudes entries must be finite")
    testing = np.asarray(test_amplitudes, dtype=float)
    if quantity == "worst_excess_error_ratio":
        if (testing.ndim != 2 or testing.shape[0] < 1 or testing.shape[1] != 2
                or not np.all(np.isfinite(testing))):
            raise ValueError("test_amplitudes must have finite shape (n_test, 2)")
    elif testing.ndim != 1 or testing.shape[0] != 2 or not np.all(np.isfinite(testing)):
        raise ValueError("test_amplitudes must be a finite 1-D array of length 2")

    count = int(n_nodes)
    spacing = float(domain_length) / (count + 1.0)
    scaled = spacing * np.arange(1, count + 1, dtype=float) / float(domain_length)
    initial = (scaled * (1.0 - scaled)
               * (6.0 * (1.0 - scaled) ** 2 * np.exp(-scaled)
                  - 10.0 * np.exp(scaled) * np.sin(scaled / 6.0))
               + scaled)

    snapshots = np.concatenate(
        [np.asarray(_oracle_march_cubic_fom(initial, time_step, n_training_steps, row,
                                            domain_length, diffusivity, boundary_values,
                                            residual_tolerance, max_newton_iterations,
                                            rhs_function=_oracle_cubic_heat_rhs), dtype=float)
         for row in training],
        axis=1,
    )
    basis, n_temperature, n_auxiliary = _oracle_block_pod_basis(
        snapshots, energy_tolerance)
    if quantity == "temperature_mode_count":
        return float(n_temperature)
    if quantity == "auxiliary_mode_count":
        return float(n_auxiliary)

    basis = np.asarray(basis, dtype=float)
    modes = basis[:count, :int(n_temperature)]

    if quantity == "worst_excess_error_ratio":
        constant, linear, quadratic, inputs, bilinear = _oracle_lifted_polynomial_operators(
            n_nodes, domain_length, diffusivity, boundary_values)
        gram = _oracle_hrf_operator_gram(
            basis, constant, linear, quadratic, inputs, bilinear)
        lifted_initial = np.concatenate([initial, initial * initial])
        reduced_initial = basis.T @ lifted_initial
        excess_ratios = []

        for amplitudes in testing:
            reference = np.asarray(_oracle_march_cubic_fom(
                initial, time_step, n_online_steps, amplitudes, domain_length,
                diffusivity, boundary_values, residual_tolerance,
                max_newton_iterations, rhs_function=_oracle_cubic_heat_rhs),
                                   dtype=float)
            residual_field = reference - modes @ (modes.T @ reference)
            projection_energy = float(np.sum(residual_field ** 2))
            if not math.isfinite(projection_energy) or projection_energy <= 0.0:
                raise ValueError("the projection energy must be finite and positive")

            prediction_energies = {}
            for scheme in ("lspg", "galerkin"):
                reduced = np.asarray(_oracle_march_reduced_rom(
                    gram, reduced_initial, time_step, n_online_steps, amplitudes,
                    alphas, betas, scheme, residual_tolerance,
                    max_newton_iterations,
                    coefficient_function=_oracle_residual_coefficient_vector,
                    direction_function=_oracle_reduced_newton_direction,
                    kronecker_function=_oracle_kronecker_square_jacobian),
                                     dtype=float)
                reconstructed = modes @ reduced[:int(n_temperature), :]
                prediction_energies[scheme] = float(
                    np.sum((reference - reconstructed) ** 2))

            galerkin_excess = prediction_energies["galerkin"] - projection_energy
            lspg_excess = prediction_energies["lspg"] - projection_energy
            if (not math.isfinite(galerkin_excess) or galerkin_excess <= 0.0
                    or not math.isfinite(lspg_excess) or lspg_excess < 0.0):
                raise ValueError("the excess errors must be finite with a positive denominator")
            excess_ratios.append(lspg_excess / galerkin_excess)

        value = float(max(excess_ratios))
        if not math.isfinite(value):
            raise ValueError("the reported scalar must be finite")
        return value

    reference = np.asarray(_oracle_march_cubic_fom(
        initial, time_step, n_online_steps, testing, domain_length, diffusivity,
        boundary_values, residual_tolerance, max_newton_iterations,
        rhs_function=_oracle_cubic_heat_rhs),
                           dtype=float)
    residual_field = reference - modes @ (modes.T @ reference)
    projection_energy = float(np.sum(residual_field ** 2))
    reference_energy = float(np.sum(reference ** 2))
    if quantity == "projection_error":
        return float(projection_energy / reference_energy)

    constant, linear, quadratic, inputs, bilinear = _oracle_lifted_polynomial_operators(
        n_nodes, domain_length, diffusivity, boundary_values)
    gram = _oracle_hrf_operator_gram(
        basis, constant, linear, quadratic, inputs, bilinear)
    lifted_initial = np.concatenate([initial, initial * initial])
    reduced_initial = basis.T @ lifted_initial

    scheme = "lspg" if quantity.startswith("lspg") else "galerkin"
    reduced = np.asarray(_oracle_march_reduced_rom(
        gram, reduced_initial, time_step, n_online_steps, testing, alphas, betas,
        scheme, residual_tolerance, max_newton_iterations,
        coefficient_function=_oracle_residual_coefficient_vector,
        direction_function=_oracle_reduced_newton_direction,
        kronecker_function=_oracle_kronecker_square_jacobian),
                         dtype=float)
    reconstructed = modes @ reduced[:int(n_temperature), :]
    prediction_energy = float(np.sum((reference - reconstructed) ** 2))

    if quantity.endswith("state_error"):
        value = prediction_energy / reference_energy
    else:
        if not math.isfinite(projection_energy) or projection_energy <= 0.0:
            raise ValueError("the projection energy must be finite and positive")
        value = prediction_energy / projection_energy
    if not math.isfinite(value):
        raise ValueError("the reported scalar must be finite")
    return float(value)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    common = (
        "import numpy as np\n"
        "prod = dict(n_nodes=31, domain_length=1.0, diffusivity=0.005,\n"
        "            boundary_values=np.array([0.0, 1.0]), time_step=0.01,\n"
        "            n_training_steps=200, n_online_steps=500,\n"
        "            training_amplitudes=np.array([[-2.0, 0.0], [-1.0, -2.0], [0.0, 1.0],\n"
        "                                          [1.0, -1.0], [2.0, 2.0]]),\n"
        "            test_amplitudes=np.array([1.5, 0.5]), energy_tolerance=1e-4,\n"
        "            alphas=np.array([1.0, -1.0]), betas=np.array([1.0, 0.0]),\n"
        "            residual_tolerance=1e-12, max_newton_iterations=50)\n"
        "small = dict(prod)\n"
        "small['n_nodes'] = 15\n"
        "small['n_training_steps'] = 60\n"
        "small['n_online_steps'] = 120\n"
        "small['time_step'] = 0.02\n"
        "small['energy_tolerance'] = 1e-3\n"
        "benchmark = dict(prod)\n"
        "benchmark['test_amplitudes'] = np.array([[-1.65, 0.85], [0.85, 0.75],\n"
        "    [-0.35, -1.55], [1.15, -0.95], [0.45, 1.35]])\n"
    )
    return [
        # --- Valid: the headline instance of the prompt.
        {
            "setup": common,
            "call": "round(run_hrf_rom_pipeline(quantity='worst_excess_error_ratio', **benchmark), 6)",
            "gold_call": "round(_oracle_run_hrf_rom_pipeline(quantity='worst_excess_error_ratio', **benchmark), 6)",
        },
        # --- Valid: the Galerkin counterpart on a cheaper instance.
        {
            "setup": common,
            "call": "round(run_hrf_rom_pipeline(quantity='galerkin_error_ratio', **small), 9)",
            "gold_call": "round(_oracle_run_hrf_rom_pipeline(quantity='galerkin_error_ratio', **small), 9)",
        },
        # --- Valid: the least-squares relative state error on the cheaper
        #     instance, which exercises the reference-norm denominator.
        {
            "setup": common,
            "call": "round(run_hrf_rom_pipeline(quantity='lspg_state_error', **small), 12)",
            "gold_call": "round(_oracle_run_hrf_rom_pipeline(quantity='lspg_state_error', **small), 12)",
        },
        # --- Valid: the projection error, which never runs a reduced march.
        {
            "setup": common,
            "call": "round(run_hrf_rom_pipeline(quantity='projection_error', **small), 12)",
            "gold_call": "round(_oracle_run_hrf_rom_pipeline(quantity='projection_error', **small), 12)",
        },
        # --- Valid: the two mode counts, which stop before the operator stage.
        {
            "setup": common,
            "call": (
                "run_hrf_rom_pipeline(quantity='temperature_mode_count', **small) "
                "+ 100.0 * run_hrf_rom_pipeline(quantity='auxiliary_mode_count', **small)"
            ),
            "gold_call": (
                "_oracle_run_hrf_rom_pipeline(quantity='temperature_mode_count', **small) "
                "+ 100.0 * _oracle_run_hrf_rom_pipeline(quantity='auxiliary_mode_count', **small)"
            ),
        },
        # --- Decisive: the projection error is the attainable lower bound of
        #     the state prediction error for this subspace, so both ratios must
        #     be at least one and both state errors must exceed it.
        {
            "setup": common + (
                "def bounded():\n"
                "    pe = run_hrf_rom_pipeline(quantity='projection_error', **small)\n"
                "    ls = run_hrf_rom_pipeline(quantity='lspg_state_error', **small)\n"
                "    gs = run_hrf_rom_pipeline(quantity='galerkin_state_error', **small)\n"
                "    lr = run_hrf_rom_pipeline(quantity='lspg_error_ratio', **small)\n"
                "    gr = run_hrf_rom_pipeline(quantity='galerkin_error_ratio', **small)\n"
                "    return (int(ls >= pe) + 2 * int(gs >= pe) + 4 * int(lr >= 1.0)\n"
                "            + 8 * int(gr >= 1.0)\n"
                "            + 16 * int(abs(lr * pe - ls) < 1e-9 * max(1.0, ls)))\n"
            ),
            "call": "bounded()",
            "gold_call": "31",
        },
        # --- Decisive: the two projections must not coincide, since assembling
        #     the least-squares system from the trial-basis-projected residual
        #     would make them identical.
        {
            "setup": common + (
                "def distinct():\n"
                "    lr = run_hrf_rom_pipeline(quantity='lspg_error_ratio', **small)\n"
                "    gr = run_hrf_rom_pipeline(quantity='galerkin_error_ratio', **small)\n"
                "    return int(abs(lr - gr) > 1e-6 * max(1.0, abs(gr)))\n"
            ),
            "call": "distinct()",
            "gold_call": "1",
        },
        # --- Valid: a Crank-Nicolson online march on the cheaper instance,
        #     where the preceding level also enters the rate sum.
        {
            "setup": common + (
                "cn = dict(small)\n"
                "cn['betas'] = np.array([0.5, 0.5])\n"
            ),
            "call": "round(run_hrf_rom_pipeline(quantity='lspg_error_ratio', **cn), 9)",
            "gold_call": "round(_oracle_run_hrf_rom_pipeline(quantity='lspg_error_ratio', **cn), 9)",
        },
        # --- Invalid: an unrecognised reporting name.
        {
            "setup": common + (
                "def run_model():\n"
                "    try:\n"
                "        run_hrf_rom_pipeline(quantity='state_error', **small)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_hrf_rom_pipeline(quantity='state_error', **small)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: training amplitudes with the wrong second extent.
        {
            "setup": common + (
                "bad = dict(small)\n"
                "bad['training_amplitudes'] = np.array([[1.0, 2.0, 3.0]])\n"
                "def run_model():\n"
                "    try:\n"
                "        run_hrf_rom_pipeline(quantity='projection_error', **bad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_hrf_rom_pipeline(quantity='projection_error', **bad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
