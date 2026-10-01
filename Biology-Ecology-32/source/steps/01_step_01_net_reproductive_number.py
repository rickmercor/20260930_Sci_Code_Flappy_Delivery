"""
Compute the expected lifetime offspring of one newborn predator when the prey population is held fixed, in an age-structured predator-prey model with role reversal.

In this model prey eat juvenile predators while adult predators eat prey. The prey population size is $x(t)$ and the predator age density is $u(t,\tau)$ on ages $0 \le \tau \le L$; individuals reaching age $L$ leave the population. Juvenile and adult predator numbers are $y_1 = \int_0^{\tau^*} u\,d\tau$ and $y_2 = \int_{\tau^*}^{L} u\,d\tau$. The model is $\frac{dx}{dt} = x\,(r - a x + s y_1 - b y_2)$, $u_t + u_\tau = -\mu(x,\tau)\,u$, $u(t,0) = \int_0^L B(x,\tau)\,u(t,\tau)\,d\tau$, with $B(x,\tau) = k x\,\varphi_{\ge}(\tau) + \tilde{B}(\tau)\,(1 - e^{-\zeta x})$, $\mu(x,\tau) = g x\,\varphi_{<}(\tau) + \mu_B(\tau) + \mu_M e^{-\rho x}$, $\varphi_{\ge}(\tau) = \frac{1}{1 + e^{-\nu(\tau - \tau^*)}}$, $\varphi_{<}(\tau) = \frac{1}{1 + e^{-\nu(\tau^* - \tau)}}$, $\tilde{B}(\tau) = 0$ for $\tau < \tau^*$ and $\tilde{B}(\tau) = b_p\,(e^{-b_{ep}(\tau - \tau^*)} + 1)$ for $\tau \ge \tau^*$, and $\mu_B(\tau) = d_p\,e^{d_{ep}(\tau - L)}$. Here $\tau^*$ is the maturation age and $g$ the rate at which prey consume juvenile predators. The remaining constants are passed in a dict params with keys r, a, k, b, s, zeta, mu_M, rho, d_p, b_p, b_ep, d_ep, L and nu.

Returns
-------
float, the expected lifetime offspring of a newborn predator at fixed prey size x
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def net_reproductive_number(x: float, tau_star: float, g: float, params: dict) -> float:
    '''Expected number of offspring produced over its lifetime by one newborn predator at a fixed prey size.

    Parameters
    ----------
    x : float
        Prey population size, held constant, x >= 0. Values above the prey carrying
        capacity r / a are admissible.
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants with keys r, a, k, b, s, zeta, mu_M, rho, d_p, b_p, b_ep, d_ep, L, nu.

    Returns
    -------
    R : float
        Expected lifetime offspring of a newborn predator in the model with prey size fixed at x,
        accurate to a relative error of 1e-10.

    Raises
    ------
    ValueError
        If x, tau_star or g is not a finite number, if x < 0 or g < 0, if params is not a dict
        containing finite numeric values for every required key, if L or nu in params is not
        finite and positive, or if tau_star does not satisfy 0 < tau_star < L.
    '''
    return R

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_net_reproductive_number(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    params = """params = dict(r=0.4, a=0.1, k=0.3, b=0.8, s=0.2, zeta=10.0, mu_M=1.0, rho=5.0,
              d_p=0.4, b_p=0.05, b_ep=0.1, d_ep=0.1, L=30.0, nu=100.0)
"""
    return [
        # --- Normal: near the coexistence prey level, tau* = 1 ---
        {
            "setup": params,
            "call": "net_reproductive_number(0.43, 1.0, 0.35, dict(params))",
            "gold_call": "_oracle_net_reproductive_number(0.43, 1.0, 0.35, dict(params))",
            "tol": 1e-9,
        },
        # --- Boundary: prey at carrying capacity r / a, late maturation and strong juvenile predation ---
        {
            "setup": params,
            "call": "net_reproductive_number(4.0, 2.0, 0.6, dict(params))",
            "gold_call": "_oracle_net_reproductive_number(4.0, 2.0, 0.6, dict(params))",
            "tol": 1e-9,
        },
        # --- Edge: prey far above the carrying capacity r / a = 4, where juvenile survival is
        # the limiting factor ---
        {
            "setup": params,
            "call": "net_reproductive_number(53.404169, 2.0, 0.05, dict(params))",
            "gold_call": "_oracle_net_reproductive_number(53.404169, 2.0, 0.05, dict(params))",
            "tol": 1e-9,
        },
        # --- Edge: maturation age not on a round grid value, low prey density ---
        {
            "setup": params,
            "call": "net_reproductive_number(0.07, 1.37, 0.9, dict(params))",
            "gold_call": "_oracle_net_reproductive_number(0.07, 1.37, 0.9, dict(params))",
            "tol": 1e-9,
        },
    ]
