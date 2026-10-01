"""
Assemble the pair of two-by-two operators the accelerated inner iteration acts with, one on the new energy amplitudes and one on the old.

The accelerated loop replaces the plain angular integration by a solve of the macroscopic moment system, which carries four unknown error amplitudes: two energy densities and two heat fluxes. The two energy balances are exact and contain the divergence of the fluxes. The two flux equations are written by adding and subtracting the first-order Chapman-Enskog flux, so that an explicit diffusion part stands beside a higher-order remainder taken untruncated from the swept distribution rather than from a moment closure; that split, and not a Grad truncation, is what keeps the scheme faithful at every Knudsen number. Contracting the flux equations with the wave vector turns each flux divergence into the flux coefficients acting on the energy densities, so both flux amplitudes drop out and the system closes on the two energies alone: that closure, with the exchange rates entering the two rows with opposite signs because energy lost by one species is gained by the other, is the operator acting on the new amplitudes. The operator acting on the old amplitudes comes from the higher-order remainder, and it splits into two additive parts because the remainder is itself a difference. The exact stress-like moment of the swept error contributes the squared-projection integrals. The subtracted Chapman-Enskog expression is written in the plain integrals the unaccelerated loop itself produces, and therefore reintroduces the unaccelerated amplification matrix as a factor. Both parts are built from the same per-species scaling of the scattering-source coefficients, and it is the cancellation between them that decides how the accelerated scheme behaves in the diffusive limit.

Returns
-------
np.ndarray of shape (2, 2, 2), float: the operator on the new amplitudes, then the operator on the old amplitudes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_accelerated_operators(coefficients: np.ndarray, moments_e: np.ndarray,
                                moments_p: np.ndarray, unaccelerated: np.ndarray,
                                kn_ep: float, v_e: float = 1.0,
                                v_p: float = 1.0) -> np.ndarray:
    """Assemble the two operators of the accelerated inner iteration.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (11,) as returned by ``compute_system_coefficients``.
    moments_e, moments_p : np.ndarray
        Shape (2,) each, from ``compute_angular_moments``, electron then phonon.
    unaccelerated : np.ndarray
        Shape (2, 2) from ``build_unaccelerated_matrix``.
    kn_ep, v_e, v_p : float
        Electron-phonon Knudsen number and the two group speeds, all positive.

    Returns
    -------
    operators : np.ndarray
        Real shape (2, 2, 2): element 0 acts on the new energy amplitudes and
        element 1 on the old ones, at unit wave-vector magnitude.

    Raises
    ------
    ValueError
        If an array input has the wrong shape or contains a non-finite value,
        or if ``kn_ep``, ``v_e`` or ``v_p`` is non-finite or not strictly
        positive.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros((2, 2, 2), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_accelerated_operators(coefficients: np.ndarray, moments_e: np.ndarray,
                                        moments_p: np.ndarray, unaccelerated: np.ndarray,
                                        kn_ep: float, v_e: float = 1.0,
                                        v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    co, moments_e, moments_p = (np.asarray(x, float).ravel() for x in (coefficients, moments_e, moments_p))
    unaccelerated = np.asarray(unaccelerated, dtype=float)
    arrays = (co, unaccelerated, moments_e, moments_p)
    if (co.shape != (11,) or unaccelerated.shape != (2, 2)
            or moments_e.shape != (2,) or moments_p.shape != (2,)
            or not all(np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError("inputs must have shapes (11,), (2, 2), (2,) and (2,) and be finite")
    kn_ep, v_e, v_p = (float(v) for v in (kn_ep, v_e, v_p))
    if not all(np.isfinite(v) and v > 0.0 for v in (kn_ep, v_e, v_p)):
        raise ValueError("kn_ep, v_e and v_p must be finite and > 0")
    kn_p, G_e, G_p, K_ee, K_ep, K_pe, K_pp, c1, c2, c3, c4 = co

    left = np.array([[G_e + K_ee, -G_p + K_ep], [-G_e + K_pe, G_p + K_pp]], dtype=float)
    g1, g4 = kn_ep * v_e * v_e, kn_p * v_p * v_p
    scaled = np.array([[g1 * c1, g1 * c2], [g4 * c3, g4 * c4]], dtype=float)
    right = -scaled * np.array([[moments_e[1]], [moments_p[1]]]) + (4.0 * np.pi / 3.0) * (scaled @ unaccelerated)
    return np.stack([left, right]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the exact benchmark peak inputs and reported accelerated operators.
        {"setup": ("import numpy as np\nkn = 10.0 ** 0.5\n"
                   "co = _oracle_compute_system_coefficients(kn, kn, kn)\n"
                   "me = _oracle_compute_angular_moments(kn)\n"
                   "mp = _oracle_compute_angular_moments(float(co[0]))\n"
                   "U = _oracle_build_unaccelerated_matrix(co, me, mp)\n"),
         "call": "build_accelerated_operators(co, me, mp, U, kn)",
         "gold_call": "_oracle_build_accelerated_operators(co, me, mp, U, kn)"},
        # Normal: a transition-regime coefficient set with its integrals and plain matrix.
        {"setup": "import numpy as np\nco = np.array([1.25, 0.2, 0.2, 0.41, 0.41, 0.102, 0.305, 0.0311, 0.0311, 0.0204, 0.0512])\n"
                  "me = np.array([5.982, 1.052])\nmp = np.array([9.204, 2.413])\nU = np.array([[0.186, 0.186], [0.188, 0.471]])\n",
         "call": "build_accelerated_operators(co, me, mp, U, 2.5)",
         "gold_call": "_oracle_build_accelerated_operators(co, me, mp, U, 2.5)"},
        # Edge: anisotropic speeds, diffusive electrons against ballistic phonons.
        {"setup": "import numpy as np\nco = np.array([40.0, 0.02, 0.05, 3.1, 0.2, 0.9, 12.0, 0.07, 0.005, 0.002, 0.06])\n"
                  "me = np.array([12.0, 4.0])\nmp = np.array([0.05, 0.0002])\nU = np.array([[0.84, 0.06], [0.0001, 0.003]])\n",
         "call": "build_accelerated_operators(co, me, mp, U, 0.2, v_e=1.3, v_p=0.7)",
         "gold_call": "_oracle_build_accelerated_operators(co, me, mp, U, 0.2, v_e=1.3, v_p=0.7)"},
        # Edge: a strongly diffusive set, where the two additive blocks nearly cancel.
        {"setup": "import numpy as np\nco = np.array([0.008, 62.5, 62.5, 0.0021, 0.0021, 0.0005, 0.0016, 0.05, 0.0296, 0.02, 0.0596])\n"
                  "me = np.array([12.44, 4.11])\nmp = np.array([12.56, 4.187])\nU = np.array([[0.62, 0.37], [0.25, 0.74]])\n",
         "call": "build_accelerated_operators(co, me, mp, U, 0.01)",
         "gold_call": "_oracle_build_accelerated_operators(co, me, mp, U, 0.01)"},
        # Boundary: a vanishing unaccelerated matrix leaves one part standing alone.
        {"setup": "import numpy as np\nco = np.array([0.5, 0.3, 0.7, 0.2, 0.2, 0.1, 0.3, 0.04, 0.04, 0.02, 0.06])\n"
                  "me = np.array([9.0, 2.0])\nmp = np.array([7.0, 1.5])\nU = np.zeros((2, 2))\n",
         "call": "build_accelerated_operators(co, me, mp, U, 1.0)",
         "gold_call": "_oracle_build_accelerated_operators(co, me, mp, U, 1.0)"},
        # Invalid: non-finite upstream arrays must be rejected before assembly.
        {"setup": ("import numpy as np\n"
                   "co = np.ones(11)\nme = np.ones(2)\nmp = np.ones(2)\n"
                   "U = np.array([[1.0, np.inf], [0.0, 1.0]])\n"
                   "def run_model():\n"
                   "    try:\n        build_accelerated_operators(co, me, mp, U, 1.0)\n        return 0\n"
                   "    except ValueError:\n        return 1\n"
                   "    except Exception:\n        return 2\n"
                   "def run_gold():\n"
                   "    try:\n        _oracle_build_accelerated_operators(co, me, mp, U, 1.0)\n        return 0\n"
                   "    except ValueError:\n        return 1\n"
                   "    except Exception:\n        return 2\n"),
         "call": "run_model()",
         "gold_call": "run_gold()"},
    ]
