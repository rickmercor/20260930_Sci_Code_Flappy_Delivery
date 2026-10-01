"""
Resolve the point-local return mapping of Eq. (27) for the non-smooth criterion, where the eigenstrain can grow along either facet independently. Each facet carries its own two-case residual and its own admissibility test, and a facet whose test says the state is already admissible contributes nothing. Sweep the facets until both magnitudes stop moving, then report them together with their activity, the eigenstrain, the stress, and the degradable potential's value.

Eq. (15) writes the eigenstrain as a sum over facet directions with independent magnitudes, and Eq. (16) then gives one scalar admissibility condition per facet. The source stresses that the volumetric and shape-changing capacities are checked separately here, which is what distinguishes this criterion from the smooth one. Both potentials of Eq. (17a) enter the residual, only one of them scaled by the degradation.

Returns
-------
A (17,) float64 array [magnitude 1, magnitude 2, active flag 1, active flag 2, degradable potential value, eigenstrain (6), stress (6)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def faceted_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, kappa: float, kappa_t: float) -> "np.ndarray":
    """Resolve the point-local return mapping of Eq. (27) for the non-smooth criterion, where
    the eigenstrain can grow along either facet independently: sweep the two facets until
    both non-negative magnitudes stop moving, then report them together with their activity,
    the eigenstrain, the stress and the degradable potential's value.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        phi: Phase-field parameter carried into the increment, in [0, 1].
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        kappa: The small floor parameter carried by the degradation function, in (0, 1).
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (17,) float64 array [magnitude 1, magnitude 2, active flag 1, active flag 2,
        degradable potential value, eigenstrain (6), stress (6)], with an active flag 1.0
        when the corresponding magnitude is non-zero.

    Raises:
        ValueError: If eps is not a (6,) array, if kappa_t is negative or not finite, or if
            E, nu, phi, kappa, ft or fs lies outside the domain accepted by the earlier
            steps.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


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


def _oracle_faceted_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, kappa: float, kappa_t: float) -> "np.ndarray":
    if np.asarray(eps, dtype=float).shape != (6,):
        raise ValueError("eps must be a six-component array")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = _oracle_elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    dv = _oracle_degradation_state(phi, kappa)[0]
    Gs = _oracle_faceted_directions(eps)
    lam = np.zeros(2)

    def _facet_residual(i: int, value: float, other: float) -> float:
        lm = np.zeros(2)
        lm[i] = value
        lm[1 - i] = other
        eta = lm[0] * Gs[0] + lm[1] * Gs[1]
        gr = _oracle_faceted_potential_gradients(eta, ft, fs)
        sig = D @ (eps - eta)
        return float(Gs[i] @ (dv * gr[0] + gr[1] - sig) + kappa_t * K * value)

    active = np.zeros(2)
    for _ in range(60):
        new = lam.copy()
        for i in (0, 1):
            li, act = _solve_multiplier(lambda v, i=i: _facet_residual(i, v, lam[1 - i]))
            new[i] = li
            active[i] = 1.0 if act else 0.0
        if float(np.max(np.abs(new - lam))) < 1.0e-15:
            lam = new
            break
        lam = new
    eta = lam[0] * Gs[0] + lam[1] * Gs[1]
    sig = D @ (eps - eta)
    Fd = ft * max(_tr(eta), 0.0) + fs * float(np.linalg.norm(_dev(eta)))
    return np.concatenate([lam, active, [Fd], eta, sig])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\nphi = 0.0\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)',
         'gold_call': '_oracle_faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)'},
        {'setup': 'import numpy as np\neps = np.array([2.6e-3, 1.9e-3, 0.0, 3.0e-4, 0.0, 8.0e-4])\nphi = 0.3\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)',
         'gold_call': '_oracle_faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)'},
        {'setup': 'import numpy as np\neps = np.array([-2.0e-3, -1.5e-3, 0.0, 0.0, 0.0, 1.4e-3])\nphi = 0.55\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\nkappa = 1.0e-3\nkappa_t = 1.0e-9\n',
         'call': 'faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)',
         'gold_call': '_oracle_faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t)'},
        {'setup': 'import numpy as np\n# invalid input: a seven-component strain is not in the six-component layout and must raise ValueError\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4, 0.0])\nphi = 0.0\nE = 200.0\nnu = 0.3\nft = 0.15\nfs = 0.15\nkappa = 1.0e-3\nkappa_t = 1.0e-9\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t))',
         'gold_call': '_catches_value_error(lambda: _oracle_faceted_return_map(eps, phi, E, nu, ft, fs, kappa, kappa_t))'},
    ]
