"""
Determine the leading growth rate of small perturbations around a coexistence steady state of the age-structured role-reversal model.

A steady state is linearly stable when every small perturbation decays. For an age-structured population coupled to a prey equation, perturbation growth rates are the roots of a characteristic equation, and the root with the largest real part decides stability and whether the approach is oscillatory.

Returns
-------
complex, the characteristic root with the largest real part at the given coexistence steady state (positive imaginary part for a conjugate pair)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rightmost_characteristic_root(state: "np.ndarray", tau_star: float, g: float, params: dict) -> complex:
    '''Characteristic root with the largest real part at a coexistence steady state.

    Parameters
    ----------
    state : np.ndarray
        One row [x*, u*(0), y1*, y2*] as returned by coexistence_steady_states for the same
        tau_star, g and params.
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    root : complex
        The root with the largest real part of the characteristic equation of the model defined
        in net_reproductive_number, linearised about the given steady state. If it belongs to a
        complex-conjugate pair, the member with positive imaginary part. Accurate to an absolute
        error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star or g is not a finite number, if g < 0, if params is not a dict containing
        finite numeric values for every required key, if L or nu in params is not finite and
        positive, if tau_star does not satisfy 0 < tau_star < L, or if state is not a
        one-dimensional array of 4 finite numbers with the first two entries positive.
    '''
    return root

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_rightmost_characteristic_root(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    params = """import numpy as np
params = dict(r=0.4, a=0.1, k=0.3, b=0.8, s=0.2, zeta=10.0, mu_M=1.0, rho=5.0,
              d_p=0.4, b_p=0.05, b_ep=0.1, d_ep=0.1, L=30.0, nu=100.0)
"""
    return [
        # --- Normal: weakly damped oscillatory mode, tau* = 1, g = 0.35 ---
        {
            "setup": params + "state = np.array([0.43360547741334027, 0.09846862199809107, 0.08560985893275563, 0.46720178005652135])\n",
            "call": "rightmost_characteristic_root(state.copy(), 1.0, 0.35, dict(params))",
            "gold_call": "_oracle_rightmost_characteristic_root(state.copy(), 1.0, 0.35, dict(params))",
            "tol": 1e-7,
        },
        # --- Normal: upper of two steady states, a saddle with a real positive root ---
        {
            "setup": params + "state = np.array([1.6747443739947518, 0.1911822146190545, 0.16238618997293736, 0.3312535007438904])\n",
            "call": "rightmost_characteristic_root(state.copy(), 2.0, 0.6, dict(params))",
            "gold_call": "_oracle_rightmost_characteristic_root(state.copy(), 2.0, 0.6, dict(params))",
            "tol": 1e-7,
        },
        # --- Edge: lower state near the merge, leading pair almost real (imaginary part 0.0054) ---
        {
            "setup": params + "state = np.array([0.849355973141835, 0.14302351961566773, 0.1553553913764681, 0.43266935120138766])\n",
            "call": "rightmost_characteristic_root(state.copy(), 2.0, 0.7685, dict(params))",
            "gold_call": "_oracle_rightmost_characteristic_root(state.copy(), 2.0, 0.7685, dict(params))",
            "tol": 1e-7,
        },
        # --- Edge: closer to the merge the pair has split into two real roots ---
        {
            "setup": params + "state = np.array([0.8569812894582145, 0.14368306025391175, 0.15539682601956772, 0.4317265453226151])\n",
            "call": "rightmost_characteristic_root(state.copy(), 2.0, 0.7688, dict(params))",
            "gold_call": "_oracle_rightmost_characteristic_root(state.copy(), 2.0, 0.7688, dict(params))",
            "tol": 1e-7,
        },
    ]
