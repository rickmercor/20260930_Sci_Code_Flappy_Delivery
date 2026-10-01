"""
Evaluate derivative tensors, of orders zero to four, of the two-dimensional reaction surface of the problem statement at a set of points.

The surface is the sum of the Eckart barrier along the reaction coordinate x and the Morse vibration along y whose anharmonicity constant varies with x. All of its parameters arrive in the `params` mapping described below. The derivatives are needed to fourth order and must be analytic, since finite differences are not accurate enough at that order for the later steps.

Returns
-------
np.ndarray of shape (n,) + (2,)*order: the symmetric order-k derivative tensor of the surface at each point (hartree / bohr^k); shape (n,) of energies for order 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def potential_derivative_tensor(points: "np.ndarray", order: int, params: dict) -> "np.ndarray":
    '''Derivative tensor of the reaction surface at each point.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) holding the coordinates (x, y) in bohr of
        n >= 1 points. A single point may be given with shape (2,); it is
        treated as n = 1.
    order : int
        Derivative order k, an integer in 0..4.
    params : dict
        Surface parameters in atomic units, with keys "V0" (amplitude of the
        symmetric hyperbolic-secant-squared component of the Eckart barrier,
        hartree), "a" (the range parameter of that barrier, the length that
        scales x in it, bohr), "V_inf" (product asymptote relative to the
        reactant asymptote, hartree), "m" (mass shared by both coordinates,
        electron masses), "omega_e" (harmonic frequency of the Morse
        vibration, hartree), "chi_inf" and "chi_0" (its dimensionless
        anharmonicity constant in the asymptotic channels and at x = 0), and
        "sigma_e" (standard deviation of the Gaussian in x along which that
        constant varies, bohr).

    Returns
    -------
    tensor : np.ndarray
        Array of shape (n,) + (2,) * k. Entry [p, i_1, ..., i_k] is the
        k-th partial derivative of the surface with respect to
        q_{i_1}, ..., q_{i_k} at point p, where q_0 = x and q_1 = y
        (hartree / bohr^k). For k = 0 the shape is (n,) and the entries are
        energies measured from the reactant asymptote. The tensor is
        symmetric in its derivative indices.

    Raises
    ------
    ValueError
        If order is not an integer in 0..4 or points cannot be read as an
        (n, 2) array with n >= 1.
    '''
    return tensor

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb


def _jet_mul(f, g):
    """Leibniz product of two derivative jets [h, h', h'', h''', h''''']."""
    out = np.zeros(np.broadcast(f, g).shape)
    for n in range(5):
        out[n] = sum(comb(n, k) * f[k] * g[n - k] for k in range(n + 1))
    return out


def _jet_compose(f, g):
    """Jet of G(f(x)) from the jet f and outer derivatives g[k] = G^(k)(f(x))."""
    f1, f2, f3, f4 = f[1], f[2], f[3], f[4]
    h = np.empty(np.broadcast(f, g).shape)
    h[0] = g[0]
    h[1] = g[1] * f1
    h[2] = g[1] * f2 + g[2] * f1 ** 2
    h[3] = g[1] * f3 + 3.0 * g[2] * f1 * f2 + g[3] * f1 ** 3
    h[4] = (g[1] * f4 + g[2] * (4.0 * f1 * f3 + 3.0 * f2 ** 2)
            + 6.0 * g[3] * f1 ** 2 * f2 + g[4] * f1 ** 4)
    return h


def _jet_exp(f):
    e = np.exp(f[0])
    return _jet_compose(f, np.array([e, e, e, e, e]))


def _jet_sqrt(f):
    s = np.sqrt(f[0])
    return _jet_compose(f, np.array([s, 0.5 / s, -0.25 / s ** 3,
                                     0.375 / s ** 5, -0.9375 / s ** 7]))


def _jet_inv(f):
    r = 1.0 / f[0]
    return _jet_compose(f, np.array([r, -r ** 2, 2.0 * r ** 3,
                                     -6.0 * r ** 4, 24.0 * r ** 5]))


def _oracle_potential_derivative_tensor(points: "np.ndarray", order: int, params: dict) -> "np.ndarray":
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 4:
        raise ValueError("order must be an integer in 0..4")
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    x, y = pts[:, 0], pts[:, 1]
    a, V0, V_inf = float(params["a"]), float(params["V0"]), float(params["V_inf"])
    m, we = float(params["m"]), float(params["omega_e"])
    c_inf, c_0, sig = float(params["chi_inf"]), float(params["chi_0"]), float(params["sigma_e"])

    # Eckart part: sech^2(u) and the logistic step, differentiated in u = x/a.
    u = x / a
    t = np.tanh(u)
    s = 1.0 / np.cosh(u) ** 2
    sech2 = [s, -2.0 * s * t, s * (6.0 * t ** 2 - 2.0), s * t * (16.0 - 24.0 * t ** 2),
             s * (16.0 - 120.0 * t ** 2 + 120.0 * t ** 4)]
    step = [0.5 * (1.0 + t), 0.5 * s, -s * t, s * (3.0 * t ** 2 - 1.0),
            s * t * (8.0 - 12.0 * t ** 2)]
    v_eck = [(V0 * sech2[n] + V_inf * step[n]) / a ** n for n in range(5)]

    # Jets of chi(x), D(x) = omega_e / (4 chi) and alpha(x) = sqrt(2 m omega_e chi).
    gauss = np.exp(-x ** 2 / (2.0 * sig ** 2))
    z = x / sig
    hermite = [np.ones_like(x), -z / sig, (z ** 2 - 1.0) / sig ** 2,
               -(z ** 3 - 3.0 * z) / sig ** 3, (z ** 4 - 6.0 * z ** 2 + 3.0) / sig ** 4]
    chi = np.array([c_inf + (c_0 - c_inf) * gauss]
                   + [(c_0 - c_inf) * gauss * hermite[n] for n in range(1, 5)])
    depth = 0.25 * we * _jet_inv(chi)
    alpha = _jet_sqrt(2.0 * m * we * chi)

    # V_M = D - 2 D exp(-alpha y) + D exp(-2 alpha y); d^j/dy^j acts on the exponentials.
    mixed = {}
    for i in range(5):
        for j in range(5 - i):
            val = v_eck[i] if j == 0 else np.zeros_like(x)
            if j == 0:
                val = val + depth[i]
            for k, coef in ((1, -2.0), (2, 1.0)):
                amp = depth
                for _ in range(j):
                    amp = _jet_mul(amp, -k * alpha)
                val = val + coef * _jet_mul(amp, _jet_exp(-k * y * alpha))[i]
            mixed[(i, j)] = val

    out = np.empty((pts.shape[0],) + (2,) * order)
    for idx in np.ndindex(*((2,) * order)):
        n_y = sum(idx)
        out[(slice(None),) + idx] = mixed[(order - n_y, n_y)]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
params = {"V0": 0.01135, "V_inf": -0.00485, "a": 0.735, "m": 1836.15267, "omega_e": 0.00415,
          "chi_inf": 0.0138, "chi_0": 0.0735, "sigma_e": 0.565}
"""
    return [
        # Normal: off-axis points, fourth-order tensor (all mixed x/y derivatives).
        {
            "setup": base + """
points = np.array([[0.31, 0.23], [-0.54, -0.12], [1.17, 0.06], [-0.08, 0.00]])
""",
            "call": "potential_derivative_tensor(points.copy(), 4, dict(params))",
            "gold_call": "_oracle_potential_derivative_tensor(points.copy(), 4, dict(params))",
        },
        # Normal: third-order tensor along the line y = 0.
        {
            "setup": base + """
points = np.column_stack([np.linspace(-1.3, 0.72, 9), np.zeros(9)])
""",
            "call": "potential_derivative_tensor(points.copy(), 3, dict(params))",
            "gold_call": "_oracle_potential_derivative_tensor(points.copy(), 3, dict(params))",
        },
        # Boundary: energies deep in the reactant and product asymptotes and one displaced point.
        {
            "setup": base + """
points = np.array([[-30.0, 0.0], [30.0, 0.0], [0.0, 0.43]])
""",
            "call": "potential_derivative_tensor(points.copy(), 0, dict(params))",
            "gold_call": "_oracle_potential_derivative_tensor(points.copy(), 0, dict(params))",
        },
        # Edge: uniform anharmonicity (chi_0 = chi_inf), symmetric barrier, single point of shape (2,).
        {
            "setup": base + """
params.update({"V_inf": 0.0, "chi_0": 0.0138})
point = np.array([0.26, 0.17])
""",
            "call": "potential_derivative_tensor(point.copy(), 2, dict(params))",
            "gold_call": "_oracle_potential_derivative_tensor(point.copy(), 2, dict(params))",
        },
        # Normal: gradient in reduced units (m = 1) with a strongly varying anharmonicity.
        {
            "setup": """import numpy as np
params = {"V0": 13.5 / np.pi, "V_inf": -18.0 / np.pi, "a": 8.0 / np.sqrt(3.0 * np.pi), "m": 1.0,
          "omega_e": 0.8, "chi_inf": 0.02, "chi_0": 0.15, "sigma_e": 1.3}
points = np.array([[-2.0, 0.3], [0.5, -0.4], [2.5, 0.8]])
""",
            "call": "potential_derivative_tensor(points.copy(), 1, dict(params))",
            "gold_call": "_oracle_potential_derivative_tensor(points.copy(), 1, dict(params))",
        },
        # Invalid: derivative order outside 0..4.
        {
            "setup": base + """
points = np.array([[0.1, 0.0]])
def run_model():
    try:
        potential_derivative_tensor(points.copy(), 5, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_potential_derivative_tensor(points.copy(), 5, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
