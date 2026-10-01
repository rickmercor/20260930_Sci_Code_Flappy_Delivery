#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from numpy.polynomial import legendre as _pp_leg


def _pp_validate_reproduction_inputs(x, tau_star, g, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    for name, value in (("x", x), ("tau_star", tau_star), ("g", g)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if x < 0.0:
        raise ValueError("x must be >= 0")
    if g < 0.0:
        raise ValueError("g must be >= 0")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"params[{key!r}] must be a finite number")

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")
    if not (0.0 < tau_star < params["L"]):
        raise ValueError("tau_star must satisfy 0 < tau_star < params['L']")


def _pp_vital_rates(x, age, tau_star, g, p):
    age = np.atleast_1d(np.asarray(age, dtype=float))

    juvenile_fraction = 0.5 * (
        1.0 - np.tanh(0.5 * p["nu"] * (age - tau_star))
    )
    adult_fraction = 1.0 - juvenile_fraction

    pulse_birth = np.where(
        age >= tau_star,
        p["b_p"]
        * (np.exp(-p["b_ep"] * (age - tau_star)) + 1.0),
        0.0,
    )

    hunger = p["mu_M"] * np.exp(-p["rho"] * x)

    mortality = (
        g * x * juvenile_fraction
        + p["d_p"] * np.exp(p["d_ep"] * (age - p["L"]))
        + hunger
    )

    birth = (
        p["k"] * x * adult_fraction
        + pulse_birth * (1.0 - np.exp(-p["zeta"] * x))
    )

    mortality_derivative = g * juvenile_fraction - p["rho"] * hunger

    birth_derivative = (
        p["k"] * adult_fraction
        + p["zeta"] * np.exp(-p["zeta"] * x) * pulse_birth
    )

    return mortality, birth, mortality_derivative, birth_derivative


def _pp_quad(tau_star, p):
    L = p["L"]

    xg, wg = _pp_leg.leggauss(10)
    inverse_vandermonde = np.linalg.inv(_pp_leg.legvander(xg, 9))

    antiderivative_matrix = np.column_stack(
        [
            _pp_leg.legval(
                xg,
                _pp_leg.legint(inverse_vandermonde[:, index], lbnd=-1),
            )
            for index in range(10)
        ]
    )

    juvenile_left = max(0.0, tau_star - 0.3)
    adult_right = min(L, tau_star + 0.3)

    pieces = [
        (
            np.linspace(
                0.0,
                juvenile_left,
                max(1, int(np.ceil(juvenile_left / 0.1))) + 1,
            )
            if juvenile_left > 0.0
            else np.array([0.0])
        ),
        np.linspace(
            juvenile_left,
            tau_star,
            max(1, int(np.ceil((tau_star - juvenile_left) / 0.005))) + 1,
        ),
        np.linspace(
            tau_star,
            adult_right,
            max(1, int(np.ceil((adult_right - tau_star) / 0.005))) + 1,
        ),
        np.linspace(
            adult_right,
            L,
            max(1, int(np.ceil((L - adult_right) / 0.25))) + 1,
        ),
    ]

    edges = np.unique(np.concatenate(pieces))
    lower = edges[:-1, None]
    upper = edges[1:, None]
    half_width = 0.5 * (upper - lower)

    nodes = (half_width * xg + 0.5 * (upper + lower)).ravel()

    return {
        "T": nodes,
        "W": (half_width * wg).ravel(),
        "juv": nodes < tau_star,
        "half": half_width,
        "xg": xg,
        "wg": wg,
        "M": antiderivative_matrix,
    }


def _pp_survival(x, t, tau_star, g, p):
    nu = p["nu"]

    juvenile_integral = t - (
        np.logaddexp(0.0, nu * (t - tau_star))
        - np.logaddexp(0.0, -nu * tau_star)
    ) / nu

    cumulative_mortality = (
        g * x * juvenile_integral
        + p["d_p"]
        / p["d_ep"]
        * (
            np.exp(p["d_ep"] * (t - p["L"]))
            - np.exp(-p["d_ep"] * p["L"])
        )
        + p["mu_M"] * np.exp(-p["rho"] * x) * t
    )

    return np.exp(-cumulative_mortality)


def _pp_reproduction_on(q, x, tau_star, g, p):
    _, birth, _, _ = _pp_vital_rates(
        x,
        q["T"],
        tau_star,
        g,
        p,
    )
    return float(
        q["W"]
        @ (
            birth
            * _pp_survival(x, q["T"], tau_star, g, p)
        )
    )


def net_reproductive_number(
    x: float,
    tau_star: float,
    g: float,
    params: dict,
) -> float:
    """Return the validated net reproductive number."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_reproduction_inputs(x, tau_star, g, params)
    return _pp_reproduction_on(
        _pp_quad(tau_star, params),
        x,
        tau_star,
        g,
        params,
    )

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _pp_validate_common(tau_star, g, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    for name, value in (("tau_star", tau_star), ("g", g)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if g < 0.0:
        raise ValueError("g must be >= 0")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"params[{key!r}] must be a finite number")

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")
    if not (0.0 < tau_star < params["L"]):
        raise ValueError("tau_star must satisfy 0 < tau_star < params['L']")


def coexistence_steady_states(
    tau_star: float,
    g: float,
    params: dict,
) -> "np.ndarray":
    """Return all validated positive coexistence steady states."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_common(tau_star, g, params)

    p = params
    q = _pp_quad(tau_star, p)

    def excess(value):
        return _pp_reproduction_on(
            q,
            value,
            tau_star,
            g,
            p,
        ) - 1.0

    # Search to a prey size beyond which lifetime reproduction is below one.
    x_max = p["r"] / p["a"]
    if g > 0.0:
        while (
            x_max < 1e7
            and (p["k"] * x_max + 2.0 * p["b_p"])
            * p["L"]
            * np.exp(-g * x_max * tau_star)
            >= 1.0
        ):
            x_max *= 2.0
    else:
        # Without juvenile predation, reproduction increases with prey size.
        x_max = 1e3

    xs = np.unique(
        np.concatenate(
            [
                np.linspace(1e-9, p["r"] / p["a"], 201),
                np.geomspace(p["r"] / p["a"], x_max, 150),
            ]
        )
    )

    values = np.array([excess(value) for value in xs])
    brackets = [
        (xs[i], xs[i + 1])
        for i in range(len(xs) - 1)
        if values[i] * values[i + 1] < 0.0
    ]

    # A pair of roots can be hidden in one grid cell near an interior extremum.
    for i in range(1, len(xs) - 1):
        if (
            (values[i] - values[i - 1])
            * (values[i + 1] - values[i])
            < 0.0
        ):
            sign = 1.0 if values[i] > values[i - 1] else -1.0
            result = minimize_scalar(
                lambda value: -sign * excess(value),
                bounds=(xs[i - 1], xs[i + 1]),
                method="bounded",
                options={"xatol": 1e-13},
            )

            extremum_x = result.x
            extremum_value = excess(extremum_x)

            if (
                extremum_value * values[i - 1] < 0.0
                and extremum_value * values[i + 1] < 0.0
                and not any(
                    lower <= extremum_x <= upper
                    for lower, upper in brackets
                )
            ):
                brackets.extend(
                    [
                        (xs[i - 1], extremum_x),
                        (extremum_x, xs[i + 1]),
                    ]
                )

    rows = []
    for lower, upper in sorted(brackets):
        prey = brentq(
            excess,
            lower,
            upper,
            xtol=1e-15,
            rtol=1e-15,
        )

        survival = _pp_survival(
            prey,
            q["T"],
            tau_star,
            g,
            p,
        )

        juvenile_integral = q["W"] @ (survival * q["juv"])
        adult_integral = q["W"] @ (survival * ~q["juv"])

        newborn = (
            p["r"] - p["a"] * prey
        ) / (
            p["b"] * adult_integral
            - p["s"] * juvenile_integral
        )

        if newborn > 0.0:
            rows.append(
                [
                    prey,
                    newborn,
                    newborn * juvenile_integral,
                    newborn * adult_integral,
                ]
            )

    return np.array(rows, dtype=float).reshape(-1, 4)

import numpy as np
from numpy.polynomial import legendre as _pp_leg
from scipy.optimize import brentq


def _pp_validate_common(tau_star, g, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    for name, value in (("tau_star", tau_star), ("g", g)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if g < 0.0:
        raise ValueError("g must be >= 0")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"params[{key!r}] must be a finite number")

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")

    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")

    if not (0.0 < tau_star < params["L"]):
        raise ValueError("tau_star must satisfy 0 < tau_star < params['L']")


def _pp_validate_state(state):
    arr = np.asarray(state, dtype=float)

    if arr.ndim != 1 or arr.shape[0] != 4 or not np.all(np.isfinite(arr)):
        raise ValueError(
            "state must be a one-dimensional array of 4 finite numbers"
        )

    if arr[0] <= 0.0 or arr[1] <= 0.0:
        raise ValueError(
            "state[0] (x*) and state[1] (u*(0)) must be > 0"
        )


def _pp_vital_rates(x, age, tau_star, g, p):
    age = np.atleast_1d(np.asarray(age, dtype=float))

    juvenile_fraction = 0.5 * (
        1.0 - np.tanh(0.5 * p["nu"] * (age - tau_star))
    )
    adult_fraction = 1.0 - juvenile_fraction

    pulse_birth = np.where(
        age >= tau_star,
        p["b_p"]
        * (np.exp(-p["b_ep"] * (age - tau_star)) + 1.0),
        0.0,
    )

    hunger = p["mu_M"] * np.exp(-p["rho"] * x)

    mortality = (
        g * x * juvenile_fraction
        + p["d_p"] * np.exp(p["d_ep"] * (age - p["L"]))
        + hunger
    )

    birth = (
        p["k"] * x * adult_fraction
        + pulse_birth * (1.0 - np.exp(-p["zeta"] * x))
    )

    mortality_derivative = g * juvenile_fraction - p["rho"] * hunger

    birth_derivative = (
        p["k"] * adult_fraction
        + p["zeta"] * np.exp(-p["zeta"] * x) * pulse_birth
    )

    return mortality, birth, mortality_derivative, birth_derivative


def _pp_quad(tau_star, p):
    L = p["L"]

    xg, wg = _pp_leg.leggauss(10)
    inverse_vandermonde = np.linalg.inv(_pp_leg.legvander(xg, 9))

    antiderivative_matrix = np.column_stack(
        [
            _pp_leg.legval(
                xg,
                _pp_leg.legint(inverse_vandermonde[:, index], lbnd=-1),
            )
            for index in range(10)
        ]
    )

    juvenile_left = max(0.0, tau_star - 0.3)
    adult_right = min(L, tau_star + 0.3)

    pieces = [
        (
            np.linspace(
                0.0,
                juvenile_left,
                max(1, int(np.ceil(juvenile_left / 0.1))) + 1,
            )
            if juvenile_left > 0.0
            else np.array([0.0])
        ),
        np.linspace(
            juvenile_left,
            tau_star,
            max(1, int(np.ceil((tau_star - juvenile_left) / 0.005))) + 1,
        ),
        np.linspace(
            tau_star,
            adult_right,
            max(1, int(np.ceil((adult_right - tau_star) / 0.005))) + 1,
        ),
        np.linspace(
            adult_right,
            L,
            max(1, int(np.ceil((L - adult_right) / 0.25))) + 1,
        ),
    ]

    edges = np.unique(np.concatenate(pieces))
    lower = edges[:-1, None]
    upper = edges[1:, None]
    half_width = 0.5 * (upper - lower)

    nodes = (half_width * xg + 0.5 * (upper + lower)).ravel()

    return {
        "T": nodes,
        "W": (half_width * wg).ravel(),
        "juv": nodes < tau_star,
        "half": half_width,
        "xg": xg,
        "wg": wg,
        "M": antiderivative_matrix,
    }


def _pp_survival(x, t, tau_star, g, p):
    nu = p["nu"]

    juvenile_integral = t - (
        np.logaddexp(0.0, nu * (t - tau_star))
        - np.logaddexp(0.0, -nu * tau_star)
    ) / nu

    cumulative_mortality = (
        g * x * juvenile_integral
        + p["d_p"]
        / p["d_ep"]
        * (
            np.exp(p["d_ep"] * (t - p["L"]))
            - np.exp(-p["d_ep"] * p["L"])
        )
        + p["mu_M"] * np.exp(-p["rho"] * x) * t
    )

    return np.exp(-cumulative_mortality)


def _pp_discrete_exponents(x, u, h, tau_star, g, p, n):
    """Leading discretized exponents used to seed the root search."""
    n_age = int(round(p["L"] / h))
    maturation_index = int(round(tau_star / h))

    age = np.arange(n_age + 1) * h

    total_weights = np.full(n_age + 1, h)
    total_weights[[0, n_age]] = h / 2.0

    juvenile_weights = np.zeros(n_age + 1)
    juvenile_weights[: maturation_index + 1] = h
    juvenile_weights[[0, maturation_index]] = h / 2.0

    adult_weights = np.zeros(n_age + 1)
    adult_weights[maturation_index:] = h
    adult_weights[[maturation_index, n_age]] = h / 2.0

    mortality, birth, mortality_derivative, birth_derivative = _pp_vital_rates(
        x,
        age,
        tau_star,
        g,
        p,
    )

    jacobian = np.zeros((n_age + 2, n_age + 2))

    jacobian[0, 0] = 1.0 + h * (
        p["r"]
        - 2.0 * p["a"] * x
        + p["s"] * (juvenile_weights @ u)
        - p["b"] * (adult_weights @ u)
    )

    jacobian[0, 1:] = h * x * (
        p["s"] * juvenile_weights - p["b"] * adult_weights
    )

    jacobian[1, 0] = total_weights @ (birth_derivative * u)
    jacobian[1, 1:] = total_weights * birth

    indices = np.arange(1, n_age + 1)
    jacobian[indices + 1, 0] = (
        -h * mortality_derivative[1:] * u[:-1]
    )
    jacobian[indices + 1, indices] = 1.0 - h * mortality[1:]

    eigenvalues = np.linalg.eigvals(jacobian).astype(complex)
    exponents = np.log(eigenvalues[eigenvalues != 0]) / h
    exponents = exponents[exponents.imag >= 0.0]

    return exponents[
        np.lexsort((-exponents.imag, -exponents.real))
    ][:n]


def _pp_char_fun(lam, state, tau_star, g, p):
    """Characteristic determinant, vectorized over candidate roots."""
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))

    if lam.size > 48:
        return np.concatenate(
            [
                _pp_char_fun(
                    lam[index : index + 48],
                    state,
                    tau_star,
                    g,
                    p,
                )
                for index in range(0, lam.size, 48)
            ]
        )

    lam = lam[:, None]
    prey_size, newborn_density = state[0], state[1]

    quadrature = _pp_quad(tau_star, p)

    _, birth, mortality_derivative, birth_derivative = _pp_vital_rates(
        prey_size,
        quadrature["T"],
        tau_star,
        g,
        p,
    )

    survival = _pp_survival(
        prey_size,
        quadrature["T"],
        tau_star,
        g,
        p,
    )

    weights = quadrature["W"]
    juvenile = quadrature["juv"]
    adult = ~quadrature["juv"]

    phi_zero = np.exp(-lam * quadrature["T"]) * survival

    integrand = (
        np.exp(lam * quadrature["T"]) * mortality_derivative
    ).reshape(lam.shape[0], -1, len(quadrature["xg"]))

    within_panel = (
        integrand @ quadrature["M"].T
    ) * quadrature["half"][None, :, :]

    panel_integrals = (
        integrand * quadrature["wg"]
    ).sum(axis=2) * quadrature["half"][None, :, 0]

    starting_integrals = np.concatenate(
        [
            np.zeros((lam.shape[0], 1), dtype=complex),
            np.cumsum(panel_integrals, axis=1)[:, :-1],
        ],
        axis=1,
    )

    phi_one = -phi_zero * newborn_density * (
        starting_integrals[:, :, None] + within_panel
    ).reshape(phi_zero.shape)

    r00 = 1.0 - (birth * phi_zero) @ weights

    r01 = (
        -((birth * phi_one) @ weights)
        - newborn_density * (weights @ (birth_derivative * survival))
    )

    r10 = -prey_size * (
        p["s"] * ((phi_zero * juvenile) @ weights)
        - p["b"] * ((phi_zero * adult) @ weights)
    )

    r11 = (
        lam[:, 0]
        + p["a"] * prey_size
        - prey_size
        * (
            p["s"] * ((phi_one * juvenile) @ weights)
            - p["b"] * ((phi_one * adult) @ weights)
        )
    )

    return r00 * r11 - r01 * r10


def _pp_newton_many(function, roots, iterations=50, tolerance=1e-14):
    roots = np.asarray(roots, dtype=complex).copy()
    active = np.ones(roots.size, dtype=bool)

    for _ in range(iterations):
        active_indices = np.nonzero(active)[0]

        if active_indices.size == 0:
            break

        current = roots[active_indices]
        values = function(current)

        derivative = (
            function(current + 1e-6) - function(current - 1e-6)
        ) / (2e-6)

        with np.errstate(all="ignore"):
            step = np.where(
                np.abs(derivative) > 0.0,
                values / derivative,
                0.0,
            )

        large_steps = np.abs(step) > 0.05
        step[large_steps] = (
            0.05 * step[large_steps] / np.abs(step[large_steps])
        )

        roots[active_indices] = current - step

        complete = (
            (np.abs(step) < tolerance)
            | ~np.isfinite(roots[active_indices])
            | (np.abs(roots[active_indices]) > 5.0)
        )

        active[active_indices[complete]] = False

    return roots


def rightmost_characteristic_root(
    state: "np.ndarray",
    tau_star: float,
    g: float,
    params: dict,
) -> complex:
    """Return the rightmost characteristic root at a coexistence state."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_common(tau_star, g, params)
    _pp_validate_state(state)

    p = params
    state = np.asarray(state, dtype=float)

    def characteristic(values):
        return _pp_char_fun(values, state, tau_star, g, p)

    h = 0.1
    age = np.arange(int(round(p["L"] / h)) + 1) * h

    discrete = _pp_discrete_exponents(
        state[0],
        state[1] * _pp_survival(state[0], age, tau_star, g, p),
        h,
        tau_star,
        g,
        p,
        8,
    )

    seeds = list(discrete)

    seeds.extend(
        complex(root.real, max(root.imag, 0.0) + 0.01)
        for root in discrete
    )

    seeds.extend(
        complex(real, imaginary)
        for real in np.linspace(-0.3, 0.3, 5)
        for imaginary in (0.005, 0.08, 0.3, 0.6)
    )

    candidates = _pp_newton_many(
        characteristic,
        np.array(seeds),
    )

    roots = []

    for root in candidates[np.abs(characteristic(candidates)) < 1e-10]:
        root = (
            complex(root.real, abs(root.imag))
            if abs(root.imag) > 1e-9
            else complex(root.real, 0.0)
        )

        if (
            -0.35 <= root.real <= 0.35
            and root.imag <= 0.85
            and all(abs(root - known) > 1e-7 for known in roots)
        ):
            roots.append(root)

    real_grid = np.linspace(-0.3, 0.3, 301)
    real_values = characteristic(real_grid.astype(complex)).real

    for index in np.nonzero(
        real_values[:-1] * real_values[1:] < 0.0
    )[0]:
        root = brentq(
            lambda value: characteristic(
                np.array([complex(value, 0.0)])
            )[0].real,
            real_grid[index],
            real_grid[index + 1],
            xtol=1e-15,
        )

        if all(
            abs(complex(root, 0.0) - known) > 1e-7
            for known in roots
        ):
            roots.append(complex(root, 0.0))

    return max(roots, key=lambda root: (root.real, root.imag))

import numpy as np
from scipy.optimize import brentq


def _pp_validate_tau_params(tau_star, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    if (
        not isinstance(tau_star, (int, float))
        or isinstance(tau_star, bool)
        or not np.isfinite(tau_star)
    ):
        raise ValueError("tau_star must be a finite number")

    if not (1.0 <= tau_star <= 2.0):
        raise ValueError("tau_star must satisfy 1 <= tau_star <= 2")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"params[{key!r}] must be a finite number"
            )

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")


def invasion_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated juvenile-consumption invasion threshold."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    x_free = params["r"] / params["a"]
    quad = _pp_quad(tau_star, params)

    def invasion_residual(g):
        return (
            _pp_reproduction_on(
                quad,
                x_free,
                tau_star,
                g,
                params,
            )
            - 1.0
        )

    return float(
        brentq(
            invasion_residual,
            0.0,
            1.0,
            xtol=1e-13,
        )
    )

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def fold_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated fold threshold for coexistence."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    p = params

    if len(
        coexistence_steady_states(
            tau_star,
            1.0,
            p,
        )
    ) > 0:
        return 1.0

    quad = _pp_quad(tau_star, p)
    x_max = p["r"] / p["a"]

    while (
        x_max < 1e7
        and (p["k"] * x_max + 2.0 * p["b_p"])
        * p["L"]
        * np.exp(-x_max * tau_star)
        >= 1.0
    ):
        x_max *= 2.0

    xs = np.unique(
        np.concatenate(
            [
                np.linspace(1e-6, p["r"] / p["a"], 301),
                np.geomspace(p["r"] / p["a"], x_max, 120),
            ]
        )
    )

    def peak_excess(g):
        values = np.array(
            [
                _pp_reproduction_on(
                    quad,
                    x,
                    tau_star,
                    g,
                    p,
                )
                for x in xs
            ]
        )

        index = int(np.argmax(values))
        result = minimize_scalar(
            lambda x: -_pp_reproduction_on(
                quad,
                x,
                tau_star,
                g,
                p,
            ),
            bounds=(
                xs[max(index - 1, 0)],
                xs[min(index + 1, len(xs) - 1)],
            ),
            method="bounded",
            options={"xatol": 1e-10},
        )

        return max(values[index], -result.fun) - 1.0

    return float(
        brentq(
            peak_excess,
            0.0,
            1.0,
            xtol=1e-10,
        )
    )

import numpy as np
from scipy.optimize import brentq


def _pp_validate_tau_params(tau_star, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    if (
        not isinstance(tau_star, (int, float))
        or isinstance(tau_star, bool)
        or not np.isfinite(tau_star)
    ):
        raise ValueError("tau_star must be a finite number")

    if not (1.0 <= tau_star <= 2.0):
        raise ValueError("tau_star must satisfy 1 <= tau_star <= 2")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"params[{key!r}] must be a finite number"
            )

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")


def stability_switch_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated stability-switch threshold."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    p = params
    upper = fold_threshold(tau_star, p)
    upper = 1.0 if upper >= 1.0 else upper - 1e-4

    tracked = {}

    def growth_rate(g):
        lower_state = coexistence_steady_states(
            tau_star,
            g,
            p,
        )[0]

        if tracked:
            nearest_g = min(
                tracked,
                key=lambda known_g: abs(known_g - g),
            )

            def characteristic(values):
                return _pp_char_fun(
                    values,
                    lower_state,
                    tau_star,
                    g,
                    p,
                )

            root = complex(
                _pp_newton_many(
                    characteristic,
                    np.array([tracked[nearest_g]]),
                )[0]
            )

            if (
                abs(characteristic(np.array([root]))[0]) < 1e-10
                and abs(root - tracked[nearest_g]) < 0.05
            ):
                tracked[g] = root
                return root.real

        root = rightmost_characteristic_root(
            lower_state,
            tau_star,
            g,
            p,
        )
        tracked[g] = root
        return root.real

    growth_rate(0.0)
    growth_rate(upper)

    return float(
        brentq(
            growth_rate,
            0.0,
            upper,
            xtol=1e-9,
        )
    )

import numpy as np


def _pp_validate_tau_params(tau_star, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    if (
        not isinstance(tau_star, (int, float))
        or isinstance(tau_star, bool)
        or not np.isfinite(tau_star)
    ):
        raise ValueError("tau_star must be a finite number")

    if not (1.0 <= tau_star <= 2.0):
        raise ValueError("tau_star must satisfy 1 <= tau_star <= 2")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"params[{key!r}] must be a finite number"
            )

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")


def bistability_width(
    tau_star: float,
    params: dict,
) -> float:
    """Return the width of the validated bistable interval."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    invasion = invasion_threshold(tau_star, params)
    stability_switch = stability_switch_threshold(
        tau_star,
        params,
    )
    fold = fold_threshold(tau_star, params)

    lower_edge = max(invasion, stability_switch)

    if fold <= lower_edge:
        return 0.0

    # Verify the physical branch at the midpoint of the proposed bistable
    # interval using the earlier reproduction, coexistence, and root steps.
    probe_g = 0.5 * (lower_edge + fold)

    states = coexistence_steady_states(
        tau_star,
        probe_g,
        params,
    )
    if states.shape[0] == 0:
        return 0.0

    lower_state = states[0]

    reproductive_number = net_reproductive_number(
        float(lower_state[0]),
        tau_star,
        probe_g,
        params,
    )

    rightmost_root = rightmost_characteristic_root(
        lower_state,
        tau_star,
        probe_g,
        params,
    )

    # The selected coexistence state must satisfy the renewal condition and
    # be stable inside the claimed bistable interval.
    if (
        abs(reproductive_number - 1.0) > 1e-6
        or not np.isfinite(rightmost_root.real)
        or rightmost_root.real > 1e-6
    ):
        return 0.0

    return float(fold - lower_edge)
SCICODE_GOLD_EOF
