"""
Build the coupled Eckart-Morse surface from spectroscopic data and return the cumulant-resummed first-order corrected ring-polymer instanton rate constant per unit length.

This final quantity combines the preceding task outputs into the requested corrected thermal rate.

Returns
-------
float: 1e12 times the cumulant-resummed first-order corrected rate constant per unit length (atomic units).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_corrected_rate(barrier: "np.ndarray", stretch_lines: "np.ndarray", width: float, mass: float,
                           beta: float, n_beads: int) -> float:
    """Return the cumulant-resummed first-order instanton rate constant in units of 1e-12 a.u.

    Atomic units with hbar = 1 are used throughout.

    Parameters
    ----------
    barrier : np.ndarray
        ``(V0, omega_b)``, the positive Eckart barrier height and magnitude of
        its barrier-top imaginary frequency.
    stretch_lines : np.ndarray
        Shape ``(2, 2)``. Each row contains the Morse stretch's ``(E01, E02)``
        transition energies, first in the reactant asymptote and then at the
        barrier top. All entries must be positive, both rows must imply the
        same harmonic frequency, and both anharmonicity constants must be
        positive.
    width : float
        Positive standard deviation of the Gaussian interpolation of the
        stretch anharmonicity.
    mass : float
        Positive mass of both coordinates.
    beta : float
        Inverse temperature, larger than ``2 pi / omega_b``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    float
        The cumulant-resummed first-order rate constant per unit length,
        multiplied by ``1e12``.

    Raises
    ------
    ValueError
        If ``barrier``, ``stretch_lines``, ``width`` or ``mass`` do not have
        the stated shapes and positive finite values, if either site gives a
        non-positive anharmonicity constant, if the two sites' harmonic
        frequencies differ by more than 1e-9 relative, or if ``beta`` and
        ``n_beads`` do not satisfy the stated conditions.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_corrected_rate(barrier: "np.ndarray", stretch_lines: "np.ndarray", width: float, mass: float,
                                   beta: float, n_beads: int) -> float:
    """Reference implementation: leading-order rate, three first-order pieces, cumulant resummation."""
    import math
    import numpy as np

    top = np.asarray(barrier, dtype=float)
    lines = np.asarray(stretch_lines, dtype=float)
    if top.shape != (2,) or lines.shape != (2, 2) or not np.all(np.isfinite(top)) or not np.all(np.isfinite(lines)):
        raise ValueError("barrier must have shape (2,) and stretch_lines shape (2, 2), all finite")
    if np.any(top <= 0.0) or np.any(lines <= 0.0) or not (width > 0.0 and mass > 0.0):
        raise ValueError("all spectroscopic inputs, the width and the mass must be positive")
    harmonic = 3.0 * lines[:, 0] - lines[:, 1]
    if abs(harmonic[0] - harmonic[1]) > 1e-9 * abs(harmonic).max():
        raise ValueError("the two sites must share one harmonic frequency")
    omega = float(harmonic.mean())
    chi_inf, chi_0 = (2.0 * lines[:, 0] - lines[:, 1]) / (2.0 * omega)
    if not (omega > 0.0 and chi_inf > 0.0 and chi_0 > 0.0):
        raise ValueError("each site must give a positive harmonic frequency and anharmonicity constant")
    v0, omega_b = top
    surface = np.array([v0, math.sqrt(2.0 * v0 / mass) / omega_b, mass, omega, chi_inf, chi_0, width])
    tau = 0.5 * beta
    X = _oracle_locate_flux_instanton(surface, beta, tau, n_beads)[0]
    n, half = X.shape[0], X.shape[0] // 2
    spatial = _oracle_compute_spatial_correction(surface, beta, tau, n)
    w0, _, w2, w3, w4 = _oracle_compute_action_time_derivatives(surface, beta, tau, n)
    a1, l2 = _oracle_compute_prefactor_time_derivatives(surface, beta, tau, n)
    reactant = _oracle_compute_reactant_correction(surface, beta, n)
    # Real-time derivatives are imaginary-time ones up to powers of i; the signs are absorbed below.
    a2 = l2 + a1**2
    temporal = -w4 / (8.0 * w2**2) + 5.0 * w3**2 / (24.0 * w2**3) - a1 * w3 / (2.0 * w2**2) + a2 / (2.0 * w2)
    k = 1 + 2 * n + 4 * n * n
    hess = _oracle_assemble_action_derivatives(X, surface, beta, tau, 0)[1 + 2 * n:k].reshape(2 * n, 2 * n)
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    far = np.zeros((n, 2))
    far[:, 0] = -50.0 * max(surface[1], surface[6])
    flat_r = _oracle_assemble_action_derivatives(far, surface, beta, tau, 0)
    hess_r = flat_r[1 + 2 * n:k].reshape(2 * n, 2 * n)
    free_r = np.ones(2 * n, bool)
    free_r[0] = False
    logdet = np.linalg.slogdet(hess[np.ix_(free, free)])[1]
    logdet_r = np.linalg.slogdet(hess_r[np.ix_(free_r, free_r)])[1]
    d = beta / n
    gaps = np.array([X[1, 0] - X[0, 0], X[half, 0] - X[half - 1, 0],
                     X[half + 1, 0] - X[half, 0], X[0, 0] - X[n - 1, 0]]) * mass / d
    phi = gaps[0] * gaps[2] + gaps[0] * gaps[1] + gaps[2] * gaps[3] + gaps[3] * gaps[1]
    # The (m / 2 pi)^N and time-step factors of c_ff and Z_r cancel at tau = beta / 2.
    log_k0 = (math.log(abs(phi) / (4.0 * mass**2)) + 0.5 * (logdet_r - logdet) - 0.5 * math.log(-w2)
              - (w0 - flat_r[0]))
    return float(1e12 * math.exp(log_k0 + spatial + temporal - reactant))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import numpy as np\n"
        "def _lines(om, ci, c0):\n"
        "    return np.array([[om * (1 - 2 * ci), 2 * om * (1 - 3 * ci)], [om * (1 - 2 * c0), 2 * om * (1 - 3 * c0)]])\n"
        "def _wrap(value):\n"
        "    return float(value)\n"
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
            "call": "_wrap(compute_corrected_rate(np.array([0.011, 0.0058]), _lines(0.0031, 0.015, 0.08), 0.4, 1836.0, 3000.0, 12))",
            "gold_call": "_wrap(_oracle_compute_corrected_rate(np.array([0.011, 0.0058]), _lines(0.0031, 0.015, 0.08), 0.4, 1836.0, 3000.0, 12))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_corrected_rate(np.array([0.009, 0.0049]), _lines(0.0027, 0.02, 0.02), 0.3, 1836.0, 3300.0, 10))",
            "gold_call": "_wrap(_oracle_compute_corrected_rate(np.array([0.009, 0.0049]), _lines(0.0027, 0.02, 0.02), 0.3, 1836.0, 3300.0, 10))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_corrected_rate(np.array([0.01, 0.005]), _lines(0.003, 0.012, 0.09), 0.5, 1836.0, 1900.0, 4))",
            "gold_call": "_wrap(_oracle_compute_corrected_rate(np.array([0.01, 0.005]), _lines(0.003, 0.012, 0.09), 0.5, 1836.0, 1900.0, 4))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_corrected_rate(np.array([0.008, 0.0045]), _lines(0.0035, 0.01, 0.06), 0.25, 3671.5, 4200.0, 16))",
            "gold_call": "_wrap(_oracle_compute_corrected_rate(np.array([0.008, 0.0045]), _lines(0.0035, 0.01, 0.06), 0.25, 3671.5, 4200.0, 16))",
        },
        {
            "setup": common + status + "bad = _lines(0.003, 0.012, 0.09)\nbad[1, 0] += 1.0e-5\n",
            "call": "_status(lambda: compute_corrected_rate(np.array([0.01, 0.005]), bad, 0.5, 1836.0, 2600.0, 8))",
            "gold_call": "_status(lambda: _oracle_compute_corrected_rate(np.array([0.01, 0.005]), bad, 0.5, 1836.0, 2600.0, 8))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: compute_corrected_rate(np.array([0.01, 0.005]), np.array([[0.00306, 0.00618], [0.00246, 0.00438]]), 0.5, 1836.0, 2600.0, 8))",
            "gold_call": "_status(lambda: _oracle_compute_corrected_rate(np.array([0.01, 0.005]), np.array([[0.00306, 0.00618], [0.00246, 0.00438]]), 0.5, 1836.0, 2600.0, 8))",
        },
    ]
