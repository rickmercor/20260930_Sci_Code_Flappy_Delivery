"""
Evaluate a partial derivative with respect to the imaginary-time split of the discretized flux-side ring-polymer action on the coupled Eckart-Morse surface, together with its gradient, Hessian and bead-resolved third- and fourth-derivative tensors.

The flux-flux correlation function is a closed imaginary-time path divided into two halves of durations tau and beta - tau, and the dependence of the discretized action on that split is what the later time integration needs.

Returns
-------
np.ndarray: flat array [S_n, gradient (2N), Hessian (4N^2), bead third tensors (8N), bead fourth tensors (16N)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_action_derivatives(beads: "np.ndarray", surface: "np.ndarray", beta: float, tau: float,
                                order: int) -> "np.ndarray":
    """Return the order-th partial tau derivative of the discretized action and its coordinate derivatives.

    Atomic units with hbar = 1 are used throughout. ``surface`` holds
    ``(V0, a, m, omega, chi_inf, chi_0, sigma)`` and defines, for the two
    coordinates (x, y), both of mass ``m``, the potential

        V(x, y) = V0 sech^2(x / a) + D(x) [1 - exp(-b(x) y)]^2,
        D(x) = omega / (4 chi(x)),   b(x) = sqrt(2 m omega chi(x)),
        chi(x) = chi_inf + (chi_0 - chi_inf) exp(-x^2 / (2 sigma^2)),

    a symmetric Eckart barrier along x and a Morse stretch along y with
    x-independent harmonic frequency ``omega`` and anharmonicity constant
    ``chi(x)``. The ring polymer has ``N`` beads ``X_0, ..., X_{N-1}``
    (``N`` even, ``X_N = X_0``); segment ``i`` joins beads ``i`` and
    ``i + 1`` and has time step ``d_i = tau / (N/2)`` for ``i < N/2`` and
    ``d_i = (beta - tau) / (N/2)`` otherwise. The action is

        S(X; tau) = sum_i [ m |X_{i+1} - X_i|^2 / (2 d_i)
                            + d_i (V(X_i) + V(X_{i+1})) / 2 ].

    The derivative with respect to ``tau`` is partial: bead positions and the
    bead numbers of the two halves are held fixed. Coordinates are flattened
    bead-major, index ``2 i + c`` with ``c = 0`` for x and ``c = 1`` for y.

    Parameters
    ----------
    beads : np.ndarray
        Bead positions, shape ``(N, 2)``, with ``N`` even and ``N >= 4``.
        Every bead must lie on the line ``y = 0``.
    surface : np.ndarray
        The seven positive surface parameters described above.
    beta : float
        Total imaginary time (inverse temperature).
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    order : int
        Order of the partial tau derivative, from 0 (the action itself) to 4.

    Returns
    -------
    np.ndarray
        One-dimensional array of length ``1 + 2N + 4N^2 + 8N + 16N``, the
        concatenation of: the order-th partial tau derivative ``S_n`` of the
        action; the gradient of ``S_n`` (length ``2N``); the Hessian of
        ``S_n`` (``2N x 2N``, row-major); and, for every bead ``i``, the third
        (``2 x 2 x 2``) and then, after all third-derivative tensors, the
        fourth (``2 x 2 x 2 x 2``) derivative tensor of ``S_n`` with respect
        to the coordinates of bead ``i`` alone, in bead order.

    Raises
    ------
    ValueError
        If ``beads`` is not a finite ``(N, 2)`` array with even ``N >= 4``,
        if any bead has ``y != 0``, if ``surface`` does not hold seven finite
        positive numbers, if ``tau`` does not satisfy ``0 < tau < beta``, or
        if ``order`` is not an integer from 0 to 4.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_action_derivatives(beads: "np.ndarray", surface: "np.ndarray", beta: float, tau: float,
                                        order: int) -> "np.ndarray":
    """Reference implementation: closed-form surface derivatives on y = 0 and tau-differentiated coefficients."""
    import math
    import numpy as np

    X = np.asarray(beads, dtype=float)
    s = np.asarray(surface, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 4 or X.shape[0] % 2 or not np.all(np.isfinite(X)):
        raise ValueError("beads must be a finite (N, 2) array with even N >= 4")
    if np.any(X[:, 1] != 0.0):
        raise ValueError("every bead must lie on the line y = 0")
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    if not (math.isfinite(beta) and math.isfinite(tau) and 0.0 < tau < beta):
        raise ValueError("tau must satisfy 0 < tau < beta")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 4:
        raise ValueError("order must be an integer from 0 to 4")
    v0, a, m, omega, chi_inf, chi_0, sigma = s
    n, half, order = X.shape[0], X.shape[0] // 2, int(order)
    tail = beta - tau
    # Spring constants m / d scale as 1 / tau and 1 / (beta - tau); potential weights are linear in tau.
    k_a = math.factorial(order) * (-1) ** order * m * half / tau ** (order + 1)
    k_b = math.factorial(order) * m * half / tail ** (order + 1)
    springs = np.array([k_a] * half + [k_b] * half)
    w_a = (tau / half, 1.0 / half, 0.0, 0.0, 0.0)[order]
    w_b = (tail / half, -1.0 / half, 0.0, 0.0, 0.0)[order]
    weights = np.array([0.5 * (w_a + w_b)] + [w_a] * (half - 1) + [0.5 * (w_a + w_b)] + [w_b] * (half - 1))
    x = X[:, 0]
    e = np.exp(-2.0 * np.abs(x) / a)
    sech2 = 4.0 * e / (1.0 + e) ** 2
    t = np.tanh(x / a)
    eck = [sech2, -2.0 * t * sech2, sech2 * (6.0 * t**2 - 2.0), t * sech2 * (16.0 - 24.0 * t**2),
           sech2 * (16.0 - 120.0 * t**2 + 120.0 * t**4)]
    eck = [v0 * d / a**k for k, d in enumerate(eck)]
    bump = np.exp(-x**2 / (2.0 * sigma**2))
    chi = chi_inf + (chi_0 - chi_inf) * bump
    dchi = -(x / sigma**2) * (chi_0 - chi_inf) * bump
    # Morse Taylor coefficients on y = 0: -D b^3 y^3 and (7/12) D b^4 y^4 with D b^2 = m omega^2 / 2.
    cubic = -1.5 * omega * (2.0 * m * omega) ** 1.5
    third = np.zeros((n, 2, 2, 2))
    third[:, 0, 0, 0] = eck[3]
    third[:, 1, 1, 1] = cubic * np.sqrt(chi)
    fourth = np.zeros((n, 2, 2, 2, 2))
    fourth[:, 0, 0, 0, 0] = eck[4]
    fourth[:, 1, 1, 1, 1] = 14.0 * m**2 * omega**3 * chi
    for idx in ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)):
        fourth[(slice(None),) + idx] = cubic * dchi / (2.0 * np.sqrt(chi))
    step = np.roll(X, -1, axis=0) - X
    action = 0.5 * np.sum(springs * np.sum(step**2, axis=1)) + np.sum(weights * eck[0])
    grad = np.roll(springs, 1)[:, None] * (X - np.roll(X, 1, axis=0)) - springs[:, None] * step
    grad[:, 0] += weights * eck[1]
    hess = np.zeros((2 * n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        local = np.diag([eck[2][i], m * omega**2])
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] += (springs[i] + springs[i - 1]) * np.eye(2) + weights[i] * local
        hess[2 * i:2 * i + 2, 2 * j:2 * j + 2] -= springs[i] * np.eye(2)
        hess[2 * j:2 * j + 2, 2 * i:2 * i + 2] -= springs[i] * np.eye(2)
    return np.concatenate([[action], grad.ravel(), hess.ravel(),
                           (weights[:, None, None, None] * third).ravel(),
                           (weights[:, None, None, None, None] * fourth).ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import math\n"
        "import numpy as np\n"
        "def _surface(v0, wb, m, om, ci, c0, sg):\n"
        "    return np.array([v0, math.sqrt(2.0 * v0 / m) / wb, m, om, ci, c0, sg])\n"
        "def _beads(n, shift, amp, a):\n"
        "    x = shift + a * np.arcsinh(amp * np.sin(2.0 * np.pi * (np.arange(n) + 0.3) / n))\n"
        "    return np.column_stack([x, np.zeros(n)])\n"
        "def _digest(v):\n"
        "    return np.asarray(v, dtype=float)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": common,
            "call": "_digest(assemble_action_derivatives(_beads(8, 0.0, 3.0, 0.62), _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 0))",
            "gold_call": "_digest(_oracle_assemble_action_derivatives(_beads(8, 0.0, 3.0, 0.62), _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 0))",
        },
        {
            "setup": common,
            "call": "_digest(assemble_action_derivatives(_beads(10, 0.1, 2.0, 0.55), _surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1400.0, 1))",
            "gold_call": "_digest(_oracle_assemble_action_derivatives(_beads(10, 0.1, 2.0, 0.55), _surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1400.0, 1))",
        },
        {
            "setup": common,
            "call": "_digest(assemble_action_derivatives(_beads(6, 0.2, 1.5, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 3200.0, 1200.0, 3))",
            "gold_call": "_digest(_oracle_assemble_action_derivatives(_beads(6, 0.2, 1.5, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 3200.0, 1200.0, 3))",
        },
        {
            "setup": common,
            "call": "_digest(assemble_action_derivatives(_beads(4, -30.0, 1.0, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2800.0, 1400.0, 4))",
            "gold_call": "_digest(_oracle_assemble_action_derivatives(_beads(4, -30.0, 1.0, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2800.0, 1400.0, 4))",
            "tol": 1e-13,
        },
        {
            "setup": common,
            "call": "_digest(assemble_action_derivatives(_beads(8, -12.0, 0.5, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 3000.0, 1600.0, 2))",
            "gold_call": "_digest(_oracle_assemble_action_derivatives(_beads(8, -12.0, 0.5, 0.6), _surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 3000.0, 1600.0, 2))",
        },
        {
            "setup": common + status + "bad = _beads(8, 0.0, 3.0, 0.62)\nbad[3, 1] = 1.0e-3\n",
            "call": "_status(lambda: assemble_action_derivatives(bad, _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 0))",
            "gold_call": "_status(lambda: _oracle_assemble_action_derivatives(bad, _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 0))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: assemble_action_derivatives(_beads(8, 0.0, 3.0, 0.62), _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 3000.0, 1))",
            "gold_call": "_status(lambda: _oracle_assemble_action_derivatives(_beads(8, 0.0, 3.0, 0.62), _surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 3000.0, 1))",
        },
    ]
