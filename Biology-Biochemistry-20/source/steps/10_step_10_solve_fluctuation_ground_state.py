"""
Step name: Return the bound-state ground energy of the one-dimensional fluctuation operator for an even-lag potential, matching the sampled well to its constant long-lag plateau.

Linearising the mean-field response about a time-dependent collective state turns the growth of an infinitesimal perturbation into a Schrödinger-like eigenvalue problem in lag time, whose potential is the fluctuation potential evaluated along the autocorrelation.

Returns
-------
float: lowest even bound-state energy of the tail-matched fluctuation operator, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_fluctuation_ground_state(half_profile: "np.ndarray", spacing: float) -> float:
    """Return the even bound-state ground energy of the fluctuation operator.

    Entry ``k`` of ``half_profile`` is the fluctuation potential ``W`` at lag
    time ``k * spacing`` for ``k = 0 .. n - 1``. Between samples, use the
    continuous piecewise-linear potential; beyond the last sample the potential
    is the constant plateau ``W[-1]``. On the half-line, solve

    ``-psi''(tau) + W(tau) * psi(tau) = E * psi(tau)``

    for the smallest energy ``E < W[-1]`` having an even solution
    (``psi'(0)=0``) that decays at infinity. At the last sampled time ``T``, the
    exact constant-tail condition is
    ``psi'(T) + sqrt(W[-1] - E) * psi(T) = 0``. These two boundary conditions make
    ``E`` a nonlinear bound-state root. The normalization of ``psi`` is
    arbitrary, and the supplied array must not be modified. Any converged
    shooting, boundary-value, transfer-matrix, or equivalent method is valid.

    Parameters
    ----------
    half_profile : np.ndarray
        One-dimensional array of at least 3 finite samples on ``tau >= 0``.
    spacing : float
        Positive finite lag-time spacing of the samples.

    Returns
    -------
    float
        The lowest even bound-state energy, as a native Python float.

    Raises
    ------
    ValueError
        If ``half_profile`` is not a one-dimensional array of at least 3 finite
        numbers, if ``spacing`` is not positive and finite, or if the sampled
        potential admits no even bound state below its long-lag plateau.
    RuntimeError
        If the numerical boundary-value solve does not converge to a nodeless
        bound state.
    """
    return ground_energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eigh_tridiagonal

def _oracle_solve_fluctuation_ground_state(half_profile: "np.ndarray", spacing: float) -> float:
    """Solve the continuous piecewise-linear well with an exact Robin tail."""
    import numpy as np

    samples = _epi_require_vector(half_profile, "half_profile")
    if samples.size < 3:
        raise ValueError("half_profile must hold at least 3 samples")
    step = _epi_require_scalar(spacing, "spacing", 0.0, True)

    plateau = float(samples[-1])
    floor = float(np.min(samples))
    if not floor < plateau:
        raise ValueError("the potential has no well below its long-lag plateau")
    # An exactly constant suffix is analytically equivalent to the Robin tail.
    # Removing it avoids an arbitrarily long numerical interval, without
    # approximating any varying part of the supplied potential.
    last_change = int(np.flatnonzero(samples != plateau)[-1])
    values = samples[:last_change + 2]
    times = np.arange(values.size, dtype=float) * step

    # The positive finite-difference ground mode is only an initial guess.
    # Adaptive collocation below solves the continuous, interpolated potential.
    # Include a constant tail in the initial-guess eigenproblem: a Dirichlet
    # wall at a short sampling window can otherwise erase a shallow bound mode.
    padding = max(8, int(np.ceil(12.0 / (step * np.sqrt(plateau - floor)))))
    padded = np.concatenate([values, np.full(padding, plateau)])
    window = np.concatenate([padded[:0:-1], padded])
    coupling = 1.0 / step ** 2
    energies, modes = eigh_tridiagonal(
        window + 2.0 * coupling, np.full(window.size - 1, -coupling),
        select="i", select_range=(0, 0))
    estimate = min(float(energies[0]), plateau - 1e-5)
    mode = modes[padded.size - 1:padded.size - 1 + values.size, 0]
    mode = mode / mode[0]
    initial = np.vstack((mode, np.gradient(mode, times)))

    # Parameterizing plateau-E by exp(p) keeps the exact decay rate real.
    def _equation(tau, state, parameter):
        energy = plateau - np.exp(parameter[0])
        potential = np.interp(tau, times, values)
        return np.vstack((state[1], (potential - energy) * state[0]))

    def _boundary(left, right, parameter):
        decay = np.exp(0.5 * parameter[0])
        return np.array([left[0] - 1.0, left[1], right[1] + decay * right[0]])

    solution = solve_bvp(
        _equation, _boundary, times, initial,
        p=np.array([np.log(plateau - estimate)]), tol=1e-10,
        max_nodes=max(100000, 4 * times.size))
    if not solution.success:
        raise RuntimeError("the continuous bound-state solve did not converge")
    energy = float(plateau - np.exp(solution.p[0]))
    amplitude = solution.y[0]
    if (not floor < energy < plateau
            or float(np.min(amplitude)) < -1e-8 * float(np.max(np.abs(amplitude)))):
        raise RuntimeError("the converged eigenfunction is not a nodeless bound state")
    return energy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    header = "import numpy as np\n"
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: a Pöschl-Teller-like well on a deliberately short window ---
        {
            "setup": header + "tau = np.arange(0.0, 4.0 + 1e-9, 0.04)\nprofile = 0.16 - 1.4 / np.cosh(0.9 * tau) ** 2\n",
            "call": "solve_fluctuation_ground_state(profile, 0.04)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.04)",
            "tol": 2e-5,
        },
        # --- Normal: an asymmetric-on-the-half-line composite well ---
        {
            "setup": header + "tau = np.arange(0.0, 7.0 + 1e-9, 0.05)\nprofile = 0.31 - 1.8 / np.cosh(1.1 * tau) ** 2 + 0.12 * np.exp(-0.7 * tau)\n",
            "call": "solve_fluctuation_ground_state(profile, 0.05)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.05)",
            "tol": 2e-5,
        },
        # --- Boundary: a shallow broad well whose bound state lies close to the plateau ---
        {
            "setup": header + "tau = np.arange(0.0, 16.0 + 1e-9, 0.08)\nprofile = 0.22 - 0.075 / np.cosh(0.24 * tau) ** 2\n",
            "call": "solve_fluctuation_ground_state(profile, 0.08)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.08)",
            "tol": 2e-5,
        },
        # --- Edge: a narrow deep well tests inward shooting without overflow ---
        {
            "setup": header + "tau = np.arange(0.0, 9.0 + 1e-9, 0.03)\nprofile = 0.7 - 5.2 / np.cosh(2.4 * tau) ** 2\n",
            "call": "solve_fluctuation_ground_state(profile, 0.03)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.03)",
            "tol": 2e-5,
        },
        # --- Scale: the physical well on a long, finely sampled tail ---
        {
            "setup": header + "tau = np.arange(6001, dtype=float) * 0.006\nprofile = 0.16143 - 0.95 / np.cosh(0.72 * tau) ** 2 + 0.04 * np.exp(-0.35 * tau)\n",
            "call": "solve_fluctuation_ground_state(profile, 0.006)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.006)",
            "tol": 2e-5,
        },
        # --- Stability: a redundant far tail makes unscaled inward amplitudes overflow ---
        {
            "setup": header + "tau = np.arange(8001, dtype=float) * 0.1\nprofile = 0.4 - 1.25 * np.exp(-(0.9 * tau) ** 2)\n",
            "call": "solve_fluctuation_ground_state(profile, 0.1)",
            "gold_call": "_oracle_solve_fluctuation_ground_state(profile, 0.1)",
            "tol": 2e-5,
        },
        # --- Contract: the supplied samples are left unmodified ---
        {
            "setup": "import numpy as np\ndef _kept(fn):\n    profile = 0.3 - 0.2 * np.exp(-np.arange(40) * 0.1)\n    original = profile.copy()\n    fn(profile, 0.1)\n    return float(np.array_equal(profile, original))\n",
            "call": "_kept(solve_fluctuation_ground_state)",
            "gold_call": "_kept(_oracle_solve_fluctuation_ground_state)",
        },
        # --- Invalid: a constant profile has continuum states but no bound state ---
        {
            "setup": status,
            "call": "_status(lambda: solve_fluctuation_ground_state(np.full(20, 0.3), 0.1))",
            "gold_call": "_status(lambda: _oracle_solve_fluctuation_ground_state(np.full(20, 0.3), 0.1))",
        },
        # --- Invalid: zero spacing ---
        {
            "setup": status,
            "call": "_status(lambda: solve_fluctuation_ground_state(np.full(8, 0.3), 0.0))",
            "gold_call": "_status(lambda: _oracle_solve_fluctuation_ground_state(np.full(8, 0.3), 0.0))",
        },
    ]
