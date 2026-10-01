#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_reference_frequency(mass: float, curvature: float) -> float:
    """Reference implementation with scale-safe positive-ratio evaluation."""
    import math
    import numbers

    import numpy as np

    if isinstance(mass, (bool, np.bool_)) or not isinstance(mass, numbers.Real):
        raise ValueError("mass must be a finite real scalar")
    if isinstance(curvature, (bool, np.bool_)) or not isinstance(
        curvature, numbers.Real
    ):
        raise ValueError("curvature must be a finite real scalar")
    mass_f = float(mass)
    curvature_f = float(curvature)
    if not math.isfinite(mass_f) or mass_f <= 0.0:
        raise ValueError("mass must be finite and > 0")
    if not math.isfinite(curvature_f) or curvature_f <= 0.0:
        raise ValueError("curvature must be finite and > 0")

    ratio = curvature_f / mass_f
    if math.isfinite(ratio) and ratio > 0.0:
        omega = math.sqrt(ratio)
    else:
        # For positive finite inputs, taking square roots before division
        # avoids an intermediate overflow or underflow in curvature / mass.
        omega = math.sqrt(curvature_f) / math.sqrt(mass_f)

    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("reference frequency must be finite and > 0")
    return float(omega)

def compute_bridge_statistics(
    left: float,
    right: float,
    minimum: float,
    tau: float,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    """Reference implementation with stable hyperbolic and affine evaluation."""
    import math
    import numbers

    import numpy as np

    values = {
        "left": left,
        "right": right,
        "minimum": minimum,
        "tau": tau,
        "mass": mass,
        "omega": omega,
        "hbar": hbar,
    }
    converted: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    left_f = converted["left"]
    right_f = converted["right"]
    minimum_f = converted["minimum"]
    tau_f = converted["tau"]
    mass_f = converted["mass"]
    omega_f = converted["omega"]
    hbar_f = converted["hbar"]
    if tau_f <= 0.0 or mass_f <= 0.0 or omega_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("tau, mass, omega, and hbar must be > 0")

    a = tau_f * hbar_f * omega_f
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("tau*hbar*omega must be finite and > 0")

    tanh_a = math.tanh(a)
    if a < 20.0:
        sech_a = 1.0 / math.cosh(a)
    else:
        exp_minus_a = math.exp(-a)
        sech_a = 2.0 * exp_minus_a / (1.0 + exp_minus_a * exp_minus_a)

    # Preserve the ordinary-regime operation order, but fall back to a
    # scaled convex combination if shifted coordinates overflow.
    q_left = left_f - minimum_f
    q_right = right_f - minimum_f
    q_sum = q_left + q_right
    harmonic_mean = minimum_f + 0.5 * q_sum * sech_a
    if not math.isfinite(harmonic_mean):
        scale = max(abs(minimum_f), abs(left_f), abs(right_f))
        if scale == 0.0:
            harmonic_mean = 0.0
        else:
            harmonic_mean = scale * math.fsum(
                [
                    (1.0 - sech_a) * (minimum_f / scale),
                    0.5 * sech_a * (left_f / scale),
                    0.5 * sech_a * (right_f / scale),
                ]
            )

    free_sum = left_f + right_f
    free_mean = 0.5 * free_sum
    if not math.isfinite(free_mean):
        scale = max(abs(left_f), abs(right_f))
        if scale == 0.0:
            free_mean = 0.0
        else:
            free_mean = 0.5 * scale * math.fsum(
                [left_f / scale, right_f / scale]
            )

    harmonic_variance = hbar_f * tanh_a / (2.0 * mass_f * omega_f)
    free_variance = hbar_f * hbar_f * tau_f / (2.0 * mass_f)

    result = np.array(
        [harmonic_mean, harmonic_variance, free_mean, free_variance],
        dtype=float,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("bridge statistics must be finite")
    if harmonic_variance <= 0.0 or free_variance <= 0.0:
        raise ValueError("bridge variances must be > 0")
    return result

def select_proposal_parameters(
    current: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> np.ndarray:
    """Reference implementation with function-local dependencies."""
    import math
    import numbers

    import numpy as np

    named_values = (
        ("current", current),
        ("minimum", minimum),
        ("domain_radius", domain_radius),
        ("harmonic_mean", harmonic_mean),
        ("harmonic_variance", harmonic_variance),
        ("free_mean", free_mean),
        ("free_variance", free_variance),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    current_f = converted["current"]
    minimum_f = converted["minimum"]
    radius_f = converted["domain_radius"]
    harmonic_mean_f = converted["harmonic_mean"]
    harmonic_variance_f = converted["harmonic_variance"]
    free_mean_f = converted["free_mean"]
    free_variance_f = converted["free_variance"]
    if radius_f < 0.0:
        raise ValueError("domain_radius must be >= 0")
    if harmonic_variance_f <= 0.0 or free_variance_f <= 0.0:
        raise ValueError("proposal variances must be > 0")

    if abs(current_f - minimum_f) <= radius_f:
        return np.array([harmonic_mean_f, harmonic_variance_f], dtype=float)
    return np.array([free_mean_f, free_variance_f], dtype=float)

def generate_trial_position(mean: float, variance: float, normal_draw: float) -> float:
    """Reference implementation with function-local dependencies."""
    import math
    import numbers

    import numpy as np

    named_values = (
        ("mean", mean),
        ("variance", variance),
        ("normal_draw", normal_draw),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    mean_f = converted["mean"]
    variance_f = converted["variance"]
    draw_f = converted["normal_draw"]
    if variance_f <= 0.0:
        raise ValueError("variance must be > 0")
    trial = mean_f + math.sqrt(variance_f) * draw_f
    if not math.isfinite(trial):
        raise ValueError("trial position must be finite")
    return float(trial)

def compute_hastings_factor(
    old: float,
    trial: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> float:
    """Reference implementation with stable paired Gaussian log ratios."""
    import math
    import numbers

    import numpy as np

    values = {
        "old": old,
        "trial": trial,
        "minimum": minimum,
        "domain_radius": domain_radius,
        "harmonic_mean": harmonic_mean,
        "harmonic_variance": harmonic_variance,
        "free_mean": free_mean,
        "free_variance": free_variance,
    }
    converted: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    old_f = converted["old"]
    trial_f = converted["trial"]
    minimum_f = converted["minimum"]
    radius_f = converted["domain_radius"]
    harmonic_mean_f = converted["harmonic_mean"]
    harmonic_variance_f = converted["harmonic_variance"]
    free_mean_f = converted["free_mean"]
    free_variance_f = converted["free_variance"]
    if radius_f < 0.0:
        raise ValueError("domain_radius must be >= 0")
    if harmonic_variance_f <= 0.0 or free_variance_f <= 0.0:
        raise ValueError("proposal variances must be > 0")

    old_inside = abs(old_f - minimum_f) <= radius_f
    trial_inside = abs(trial_f - minimum_f) <= radius_f
    forward_mean = harmonic_mean_f if old_inside else free_mean_f
    forward_variance = harmonic_variance_f if old_inside else free_variance_f
    reverse_mean = harmonic_mean_f if trial_inside else free_mean_f
    reverse_variance = harmonic_variance_f if trial_inside else free_variance_f

    def raw_log_normal(value: float, mean: float, variance: float) -> float:
        return -0.5 * math.log(2.0 * math.pi * variance) - (
            value - mean
        ) ** 2 / (2.0 * variance)

    def stable_log_ratio(
        value: float,
        numerator_mean: float,
        numerator_variance: float,
        denominator_mean: float,
        denominator_variance: float,
    ) -> float:
        """Return log N_num(value) - log N_den(value) without separate squares."""
        if (
            numerator_mean == denominator_mean
            and numerator_variance == denominator_variance
        ):
            return 0.0

        if numerator_variance == denominator_variance:
            # Difference of squares, ordered so the small scale is applied first.
            first = (numerator_mean - denominator_mean) / (
                2.0 * numerator_variance
            )
            second = 2.0 * value - numerator_mean - denominator_mean
            return first * second

        sqrt_num = math.sqrt(numerator_variance)
        sqrt_den = math.sqrt(denominator_variance)
        z_num = (value - numerator_mean) / sqrt_num
        z_den = (value - denominator_mean) / sqrt_den
        if not math.isfinite(z_num) or not math.isfinite(z_den):
            raise ValueError("paired standardized displacements must be finite")

        # Factor the difference of squares before either square is formed.
        quadratic_difference = (z_num - z_den) * (z_num + z_den)
        log_variance_ratio = math.log(denominator_variance) - math.log(
            numerator_variance
        )
        return 0.5 * log_variance_ratio - 0.5 * quadratic_difference

    # Preserve the previous binary64 route in ordinary regimes. If an
    # individual squared displacement overflows, switch to paired ratios.
    try:
        log_reverse_old = raw_log_normal(old_f, reverse_mean, reverse_variance)
        log_harmonic_old = raw_log_normal(
            old_f, harmonic_mean_f, harmonic_variance_f
        )
        log_forward_trial = raw_log_normal(
            trial_f, forward_mean, forward_variance
        )
        log_harmonic_trial = raw_log_normal(
            trial_f, harmonic_mean_f, harmonic_variance_f
        )
        log_factor = (
            log_reverse_old
            - log_harmonic_old
            - log_forward_trial
            + log_harmonic_trial
        )
        if not math.isfinite(log_factor):
            raise ArithmeticError("non-finite separately evaluated log ratio")
    except (ArithmeticError, OverflowError):
        log_factor = stable_log_ratio(
            old_f,
            reverse_mean,
            reverse_variance,
            harmonic_mean_f,
            harmonic_variance_f,
        ) + stable_log_ratio(
            trial_f,
            harmonic_mean_f,
            harmonic_variance_f,
            forward_mean,
            forward_variance,
        )

    try:
        factor = math.exp(log_factor)
    except OverflowError as exc:
        raise ValueError("Hastings factor must be finite") from exc
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Hastings factor must be finite and > 0")
    return float(factor)

def compute_mixed_acceptance(
    old: float,
    trial: float,
    tau: float,
    minimum: float,
    curvature: float,
    alpha_3: float,
    alpha_4: float,
    hastings_factor: float,
) -> float:
    """Reference implementation with a scale-safe residual difference."""
    import math
    import numbers
    from decimal import Decimal, InvalidOperation, Overflow, localcontext

    import numpy as np

    named_values = (
        ("old", old),
        ("trial", trial),
        ("tau", tau),
        ("minimum", minimum),
        ("curvature", curvature),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
        ("hastings_factor", hastings_factor),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    old_f = converted["old"]
    trial_f = converted["trial"]
    tau_f = converted["tau"]
    minimum_f = converted["minimum"]
    curvature_f = converted["curvature"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    factor_f = converted["hastings_factor"]
    if tau_f <= 0.0 or curvature_f <= 0.0 or factor_f <= 0.0:
        raise ValueError("tau, curvature, and hastings_factor must be > 0")

    q_trial = trial_f - minimum_f
    q_old = old_f - minimum_f

    # Preserve the ordinary binary64 route for the task instance and other
    # moderate regimes, then fall back only when separate powers or
    # residuals overflow or cancel to a non-finite value.
    try:
        residual_trial = 0.5 * curvature_f * (
            alpha_3_f * q_trial**3 + alpha_4_f * q_trial**4
        )
        residual_old = 0.5 * curvature_f * (
            alpha_3_f * q_old**3 + alpha_4_f * q_old**4
        )
        delta_residual = residual_trial - residual_old
        if not math.isfinite(delta_residual):
            raise ArithmeticError("non-finite separately evaluated residual change")
    except (ArithmeticError, OverflowError):
        try:
            with localcontext() as context:
                context.prec = 120
                q_trial_d = Decimal.from_float(trial_f) - Decimal.from_float(
                    minimum_f
                )
                q_old_d = Decimal.from_float(old_f) - Decimal.from_float(minimum_f)
                delta_d = (
                    Decimal("0.5")
                    * Decimal.from_float(curvature_f)
                    * (
                        Decimal.from_float(alpha_3_f)
                        * (q_trial_d**3 - q_old_d**3)
                        + Decimal.from_float(alpha_4_f)
                        * (q_trial_d**4 - q_old_d**4)
                    )
                )
                delta_residual = float(delta_d)
        except (InvalidOperation, Overflow, OverflowError) as exc:
            raise ValueError("residual-potential change must be finite") from exc
        if not math.isfinite(delta_residual):
            raise ValueError("residual-potential change must be finite")

    log_ratio = math.log(factor_f) - tau_f * delta_residual
    if math.isnan(log_ratio):
        raise ValueError("acceptance log-ratio must not be NaN")
    if log_ratio >= 0.0:
        return 1.0
    try:
        acceptance = math.exp(log_ratio)
    except OverflowError as exc:
        raise ValueError("acceptance probability must be finite") from exc
    if not math.isfinite(acceptance):
        raise ValueError("acceptance probability must be finite")
    return float(acceptance)

def compute_harmonic_energy(
    path: np.ndarray,
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    minimum: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Reference implementation with stable link and residual evaluation."""
    import math
    import numbers
    from decimal import Decimal, InvalidOperation, Overflow, localcontext

    import numpy as np

    try:
        path_array = np.asarray(path, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must be convertible to a finite float array") from exc
    if path_array.ndim != 1 or path_array.size < 2:
        raise ValueError("path must be one-dimensional with at least two beads")
    if not np.all(np.isfinite(path_array)):
        raise ValueError("path must contain only finite values")

    named_values = (
        ("beta", beta),
        ("mass", mass),
        ("omega", omega),
        ("hbar", hbar),
        ("minimum", minimum),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    beta_f = converted["beta"]
    mass_f = converted["mass"]
    omega_f = converted["omega"]
    hbar_f = converted["hbar"]
    minimum_f = converted["minimum"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    if beta_f <= 0.0 or mass_f <= 0.0 or omega_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("beta, mass, omega, and hbar must be > 0")

    bead_count = int(path_array.size)
    a = beta_f * hbar_f * omega_f / bead_count
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("beta*hbar*omega/P must be finite and > 0")

    tanh_a = math.tanh(a)
    if not math.isfinite(tanh_a) or tanh_a <= 0.0:
        raise ValueError("tanh(beta*hbar*omega/P) must be finite and > 0")

    if a < 350.0:
        sinh_a = math.sinh(a)
        cosh_a = math.cosh(a)
        inv_sinh = 1.0 / sinh_a
        inv_cosh_plus_one = 1.0 / (cosh_a + 1.0)
    else:
        # Both inverse factors are naturally zero once exp(-a) underflows.
        exp_minus_a = math.exp(-a)
        if exp_minus_a == 0.0:
            inv_sinh = 0.0
            inv_cosh_plus_one = 0.0
        else:
            inv_sinh = 2.0 * exp_minus_a / (1.0 - exp_minus_a * exp_minus_a)
            inv_cosh_plus_one = 2.0 * exp_minus_a / (
                (1.0 + exp_minus_a) * (1.0 + exp_minus_a)
            )

    q = path_array - minimum_f
    q_next = np.roll(q, -1)
    constant = hbar_f * omega_f / (2.0 * bead_count * tanh_a)
    link_scale = mass_f * omega_f * omega_f / (2.0 * bead_count)
    with np.errstate(over="ignore", invalid="ignore"):
        link = link_scale * (
            -(q - q_next) ** 2 * (inv_sinh * inv_sinh)
            + 2.0 * q * q_next * inv_cosh_plus_one
        )
    if not np.all(np.isfinite(link)):
        raise ValueError("harmonic-link contributions must be finite")

    # Preserve the original binary64 route in ordinary regimes. If the
    # unweighted powers overflow or separately weighted contributions
    # cancel through non-finite intermediates, recompute only the residual
    # in extended decimal range and convert the final weighted bead terms.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (
            0.5
            * mass_f
            * omega_f
            * omega_f
            * (alpha_3_f * q**3 + alpha_4_f * q**4)
            / bead_count
        )

    if not np.all(np.isfinite(residual)):
        residual_values: list[float] = []
        try:
            with localcontext() as context:
                context.prec = 150
                coefficient = (
                    Decimal("0.5")
                    * Decimal.from_float(mass_f)
                    * Decimal.from_float(omega_f)
                    * Decimal.from_float(omega_f)
                    / Decimal(bead_count)
                )
                alpha_3_d = Decimal.from_float(alpha_3_f)
                alpha_4_d = Decimal.from_float(alpha_4_f)
                minimum_d = Decimal.from_float(minimum_f)
                for position in path_array.tolist():
                    q_d = Decimal.from_float(float(position)) - minimum_d
                    term_d = coefficient * (
                        alpha_3_d * q_d**3 + alpha_4_d * q_d**4
                    )
                    residual_values.append(float(term_d))
        except (InvalidOperation, Overflow, OverflowError) as exc:
            raise ValueError("weighted residual contributions must be finite") from exc
        residual = np.asarray(residual_values, dtype=float)
        if not np.all(np.isfinite(residual)):
            raise ValueError("weighted residual contributions must be finite")

    energy = float(np.sum(constant + link + residual, dtype=float))
    if not math.isfinite(energy):
        raise ValueError("energy estimate must be finite")
    return energy

def run_mixed_pimc_sweep(
    path: np.ndarray,
    bead_indices: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    beta: float,
    mass: float,
    curvature: float,
    hbar: float,
    minimum: float,
    domain_radius: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Reference implementation composed exclusively from steps 1-7 oracles."""
    import math
    import numbers

    import numpy as np

    try:
        current_path = np.asarray(path, dtype=float).copy()
        raw_indices = np.asarray(bead_indices)
        normal_array = np.asarray(normal_draws, dtype=float)
        uniform_array = np.asarray(uniform_draws, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("path and draw inputs must be convertible to numeric arrays") from exc

    if current_path.ndim != 1 or current_path.size < 3 or not np.all(np.isfinite(current_path)):
        raise ValueError("path must be a finite one-dimensional array with at least three beads")
    if raw_indices.ndim != 1 or normal_array.ndim != 1 or uniform_array.ndim != 1:
        raise ValueError("bead_indices and draw arrays must be one-dimensional")
    if not (raw_indices.size == normal_array.size == uniform_array.size):
        raise ValueError("bead_indices, normal_draws, and uniform_draws must have equal lengths")
    if not np.all(np.isfinite(normal_array)) or not np.all(np.isfinite(uniform_array)):
        raise ValueError("draw arrays must be finite")
    if np.any((uniform_array < 0.0) | (uniform_array >= 1.0)):
        raise ValueError("uniform_draws must lie in [0, 1)")

    indices_list: list[int] = []
    for value in raw_indices.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            if not isinstance(value, numbers.Real) or not float(value).is_integer():
                raise ValueError("bead_indices must contain integers")
        index = int(value)
        if index < 0 or index >= current_path.size:
            raise ValueError("bead index out of range")
        indices_list.append(index)

    named_values = (
        ("beta", beta),
        ("hbar", hbar),
        ("minimum", minimum),
        ("domain_radius", domain_radius),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    beta_f = converted["beta"]
    hbar_f = converted["hbar"]
    minimum_f = converted["minimum"]
    radius_f = converted["domain_radius"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    if beta_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("beta and hbar must be > 0")
    if radius_f < 0.0:
        raise ValueError("domain_radius must be >= 0")

    # Step 1 oracle: validates mass and curvature and returns omega.
    omega = compute_reference_frequency(mass, curvature)
    mass_f = float(mass)
    curvature_f = float(curvature)
    tau = beta_f / current_path.size

    for index, normal_draw, uniform_draw in zip(
        indices_list,
        normal_array.tolist(),
        uniform_array.tolist(),
        strict=True,
    ):
        left = float(current_path[(index - 1) % current_path.size])
        old = float(current_path[index])
        right = float(current_path[(index + 1) % current_path.size])

        # Step 2 oracle: both fixed-endpoint Gaussian bridges.
        statistics = compute_bridge_statistics(
            left,
            right,
            minimum_f,
            tau,
            mass_f,
            omega,
            hbar_f,
        )
        harmonic_mean, harmonic_variance, free_mean, free_variance = statistics.tolist()

        # Step 3 oracle: forward family is selected from the current state.
        selected = select_proposal_parameters(
            old,
            minimum_f,
            radius_f,
            harmonic_mean,
            harmonic_variance,
            free_mean,
            free_variance,
        )

        # Step 4 oracle: deterministic proposal from the supplied normal deviate.
        trial = generate_trial_position(
            float(selected[0]),
            float(selected[1]),
            float(normal_draw),
        )

        # Step 5 oracle: reverse family is selected from the proposed state.
        factor = compute_hastings_factor(
            old,
            trial,
            minimum_f,
            radius_f,
            harmonic_mean,
            harmonic_variance,
            free_mean,
            free_variance,
        )

        # Step 6 oracle: residual-only Boltzmann factor under harmonic splitting.
        acceptance = compute_mixed_acceptance(
            old,
            trial,
            tau,
            minimum_f,
            curvature_f,
            alpha_3_f,
            alpha_4_f,
            factor,
        )
        if float(uniform_draw) < acceptance:
            current_path[index] = trial

    # Step 7 oracle: final periodic-path harmonic-reference energy.
    return compute_harmonic_energy(
        current_path,
        beta_f,
        mass_f,
        omega,
        hbar_f,
        minimum_f,
        alpha_3_f,
        alpha_4_f,
    )
SCICODE_GOLD_EOF
