"""
Resolve the point-local return mapping of Eq. (23) for the smooth criterion: find the single eigenstrain magnitude that brings the state back onto the degraded strength surface, and report the resulting eigenstrain, stress and strength-potential value. Eq. (23b) is a two-case residual; use the source's own test to decide which case applies, and include the extra term the source adds to the active case, whose purpose it states immediately below the equation. The magnitude is non-negative.

Because the eigenstrain needs no spatial gradients its magnitude can be solved at a single point exactly as a plasticity model resolves a consistency condition, which is the whole point of the source's reformulation. Eq. (16) reduces the tensor-valued admissibility statement of Eq. (13) to one scalar condition per direction. Eq. (12) gives the stress from the strain that survives after the eigenstrain is removed.

Returns
-------
A (15,) float64 array [magnitude, active flag, potential value, eigenstrain (6), stress (6)], with the active flag 1.0 when the eigenstrain magnitude is non-zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smooth_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, eps_ref: float, kappa: float, kappa_t: float) -> "np.ndarray":
    """Resolve the point-local return mapping of Eq. (23) for the smooth criterion: find the
    single non-negative eigenstrain magnitude that brings the state back onto the degraded
    strength surface, and report the resulting eigenstrain, stress and strength-potential
    value.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        phi: Phase-field parameter carried into the increment, in [0, 1].
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        eps_ref: Reference strain of the pressure-sensitive criterion.
        kappa: The small floor parameter carried by the degradation function, in (0, 1).
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (15,) float64 array [magnitude, active flag, potential value, eigenstrain (6),
        stress (6)], with the active flag 1.0 when the eigenstrain magnitude is non-zero.

    Raises:
        ValueError: If eps is not a (6,) array, if kappa_t is negative or not finite, or if
            E, nu, phi, kappa, ft, fs or eps_ref lies outside the domain accepted by the
            earlier steps.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _solve_multiplier(residual: "Callable[[float], float]", hi0: float = 1.0e-3) -> "tuple[float, bool]":
    """Smallest lam >= 0 with residual(lam) = 0; lam = 0 when the state is inside."""
    probe = 1.0e-14
    if residual(probe) >= 0.0:
        return 0.0, False
    hi = hi0
    for _ in range(400):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def _oracle_smooth_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, eps_ref: float, kappa: float, kappa_t: float) -> "np.ndarray":
    if np.asarray(eps, dtype=float).shape != (6,):
        raise ValueError("eps must be a six-component array")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = _oracle_elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    dv = _oracle_degradation_state(phi, kappa)[0]
    G = _oracle_smooth_direction_gradient(eps, np.zeros(6), ft, fs, eps_ref)[0]

    def _residual(lam: float) -> float:
        eta = lam * G
        blk = _oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref)
        sig = D @ (eps - eta)
        return float(G @ (dv * blk[1] - sig) + kappa_t * K * lam)

    lam, active = _solve_multiplier(_residual)
    eta = lam * G
    blk = _oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref)
    sig = D @ (eps - eta)
    return np.concatenate([[lam, 1.0 if active else 0.0, float(blk[2][0])], eta, sig])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\nphi = 0.0\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\neps_ref = 0.01\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)',
         'gold_call': '_oracle_smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)'},
        {'setup': 'import numpy as np\neps = np.array([-1.2e-3, -1.3e-3, 0.0, 4.0e-4, 0.0, 2.1e-3])\nphi = 0.35\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\neps_ref = 0.01\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)',
         'gold_call': '_oracle_smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)'},
        {'setup': 'import numpy as np\neps = np.array([1.0e-5, 2.0e-5, 0.0, 0.0, 0.0, 1.0e-5])\nphi = 0.1\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\neps_ref = 0.01\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)',
         'gold_call': '_oracle_smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t)'},
        {'setup': 'import numpy as np\n# invalid input: a negative stabilising modulus factor must raise ValueError\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\nphi = 0.0\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\neps_ref = 0.01\nkappa = 1.0e-3\nkappa_t = -1.0e-9\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t))',
         'gold_call': '_catches_value_error(lambda: _oracle_smooth_return_map(eps, phi, E, nu, ft, fs, eps_ref, kappa, kappa_t))'},
    ]
