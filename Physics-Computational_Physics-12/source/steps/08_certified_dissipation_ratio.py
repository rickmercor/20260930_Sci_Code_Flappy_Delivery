"""
Final orchestrator: certified dissipation expressed in units of junction dissipation.

Run the inference on the primary recording, with the tabulated bin centers as the time axis and the tabulated bin widths as the bin widths. Report the strongest lower bound on the total steady-state entropy production rate of the whole device that this recording certifies, expressed in units of the dissipation carried by the two resolved junctions together,



$$\mathcal{R} = \frac{\sigma_{\mathrm{certified}}}{\sigma_A + \sigma_B}.$$

Returns
-------
One finite Python float, the dimensionless ratio $\mathcal{R}$. Raise `ValueError` for a shape mismatch between the time axis, the bin widths and the density array, or for a non-positive total dissipation of the two resolved junctions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Final orchestrator: certified dissipation expressed in units of junction dissipation."""

import numpy as np


def certified_dissipation_ratio(node_times, bin_widths, recorded_density):
    """Return the certified entropy-production bound in units of junction dissipation.

    Parameters
    ----------
    node_times : array_like, shape (N,)
        Bin-center times of the primary recording in microseconds.
    bin_widths : array_like, shape (N,)
        Positive widths of the same bins in microseconds.
    recorded_density : array_like, shape (4, 4, N)
        Recorded waiting-time densities of the primary recording in inverse
        microseconds, indexed as ``[previous transition type, next transition
        type, bin]``.

    Returns
    -------
    float
        The strongest lower bound on the total steady-state entropy production
        rate of the device that the recording certifies, divided by the sum of
        the entropy production rates carried by the two resolved junctions,
        dimensionless.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _common_levels(detection):
    import numpy as np
    return np.array([min(detection[0, 0], detection[0, 1]),
                     min(detection[1, 0], detection[1, 1])])


def _keep_vector(detection, level):
    import numpy as np
    return np.array([level[0] / detection[0, 0], level[0] / detection[0, 1],
                     level[1] / detection[1, 0], level[1] / detection[1, 1]])


def _registration_constants(moments, detection):
    import numpy as np
    """Seven constants of the registration dependence of the joint cumulants."""
    reference = _common_levels(detection)
    points = [(reference[0], reference[1]),
              (0.5 * reference[0], reference[1]),
              (reference[0], 0.5 * reference[1])]
    cumulants = []
    for level in points:
        surviving = _oracle_discarded_record_moments(moments, _keep_vector(detection, level))
        cumulants.append(_oracle_counting_cumulants(surviving))
    alpha_a = cumulants[0][0] / points[0][0]
    alpha_b = cumulants[0][1] / points[0][1]
    system_a = np.array([[points[0][0], points[0][0] ** 2], [points[1][0], points[1][0] ** 2]])
    system_b = np.array([[points[0][1], points[0][1] ** 2], [points[2][1], points[2][1] ** 2]])
    if abs(np.linalg.det(system_a)) < 1e-300 or abs(np.linalg.det(system_b)) < 1e-300:
        raise ValueError("degenerate registration dependence")
    beta_a, gamma_a = np.linalg.solve(system_a, np.array([cumulants[0][2], cumulants[1][2]]))
    beta_b, gamma_b = np.linalg.solve(system_b, np.array([cumulants[0][3], cumulants[2][3]]))
    delta = cumulants[0][4] / (points[0][0] * points[0][1])
    return np.array([alpha_a, alpha_b, beta_a, gamma_a, beta_b, gamma_b, delta], dtype=float)


def _complete_registration_value(moments, detection):
    import numpy as np
    alpha_a, alpha_b, beta_a, gamma_a, beta_b, gamma_b, delta = _registration_constants(moments, detection)
    current = np.array([alpha_a, alpha_b])
    diffusion = np.array([[beta_a + gamma_a, delta], [delta, beta_b + gamma_b]])
    if diffusion[0, 0] <= 0.0 or diffusion[1, 1] <= 0.0:
        raise ValueError("the extended diffusion coefficients must be positive")
    if np.linalg.det(diffusion) <= 0.0:
        raise ValueError("the extended diffusion matrix must be positive definite")
    return float(current @ np.linalg.solve(diffusion, current))


def _oracle_certified_dissipation_ratio(node_times, bin_widths, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    widths = np.asarray(bin_widths, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or widths.shape != times.shape:
        raise ValueError("node_times and bin_widths must be one-dimensional arrays of equal length")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    short_time = _oracle_short_time_coefficients(times, density)
    detection = _oracle_detection_and_rate_constants(short_time)
    moments = _oracle_kernel_moments(times, widths, density)
    junction = _oracle_junction_dissipation(moments, detection)
    uncertainty = _complete_registration_value(moments, detection)
    keep = _keep_vector(detection, _common_levels(detection))
    waiting = _oracle_waiting_time_irreversibility(times, density, short_time, keep)
    resolved = junction[4] + junction[5]
    if resolved <= 0.0:
        raise ValueError("the resolved junctions must carry positive dissipation together")
    certified = max(uncertainty, waiting[4])
    return float(certified / resolved)

# =============================================================================
# TEST CASES
# =============================================================================

def _instance(rate_rows, etas):
    return (
        "import numpy as np\n"
        "K = np.array(%s, dtype=float)\n"
        "L = K - np.diag(K.sum(axis=1))\n"
        "eta = np.array(%r, dtype=float)\n"
        "pair = [(0, 1), (1, 0), (2, 3), (3, 2)]\n"
        "S = L.copy()\n"
        "for (a, b), e in zip(pair, eta):\n"
        "    S[a, b] -= e * L[a, b]\n"
        "w = np.concatenate([np.full(8, 0.003), np.full(4, 0.02), np.full(5, 0.09),\n"
        "                    np.full(6, 0.35), np.full(3, 1.3)])\n"
        "bounds = np.concatenate([[0.0], np.cumsum(w)])\n"
        "t = 0.5 * (bounds[:-1] + bounds[1:])\n"
        "vals, vecs = np.linalg.eig(S)\n"
        "back = np.linalg.inv(vecs)\n"
        "flow = np.array([(vecs @ np.diag(np.exp(vals * x)) @ back).real for x in bounds])\n"
        "block = (np.linalg.inv(S) @ (flow[1:] - flow[:-1])).real\n"
        "rate = np.array([e * L[a, b] for (a, b), e in zip(pair, eta)])\n"
        "psi = np.zeros((4, 4, t.size))\n"
        "for u, (au, bu) in enumerate(pair):\n"
        "    for v, (av, bv) in enumerate(pair):\n"
        "        psi[u, v] = block[:, bu, av] * rate[v]\n"
        "psi = psi / w\n"
        % (rate_rows, etas)
    )


_NET_ONE = [[0.0, 3.4, 3.8, 0.0, 2.6],
            [5.5, 0.0, 3.1, 0.0, 0.0],
            [2.7, 4.2, 0.0, 5.6, 0.0],
            [0.0, 0.0, 3.7, 0.0, 6.2],
            [7.4, 0.0, 0.0, 7.1, 0.0]]
_NET_TWO = [[0.0, 2.2, 5.0, 0.0, 3.6],
            [6.8, 0.0, 2.4, 0.0, 0.0],
            [3.6, 5.1, 0.0, 7.0, 0.0],
            [0.0, 0.0, 2.8, 0.0, 5.5],
            [6.0, 0.0, 0.0, 8.4, 0.0]]
_NET_THREE = [[0.0, 4.1, 3.2, 0.0, 4.4],
              [4.9, 0.0, 3.8, 0.0, 0.0],
              [2.5, 3.9, 0.0, 4.8, 0.0],
              [0.0, 0.0, 4.6, 0.0, 7.6],
              [5.2, 0.0, 0.0, 6.4, 0.0]]


def test_cases():
    """Differential cases for the complete inference chain."""
    return [
        {
            "setup": _instance(_NET_ONE, [0.45, 0.85, 0.70, 0.50]),
            "call": "certified_dissipation_ratio(t, w, psi)",
            "gold_call": "_oracle_certified_dissipation_ratio(t, w, psi)",
        },
        {
            "setup": _instance(_NET_TWO, [0.62, 0.80, 0.55, 0.90]),
            "call": "certified_dissipation_ratio(t, w, psi)",
            "gold_call": "_oracle_certified_dissipation_ratio(t, w, psi)",
        },
        {
            "setup": _instance(_NET_THREE, [0.999, 0.999, 0.999, 0.999]),
            "call": "certified_dissipation_ratio(t, w, psi)",
            "gold_call": "_oracle_certified_dissipation_ratio(t, w, psi)",
        },
    ]
