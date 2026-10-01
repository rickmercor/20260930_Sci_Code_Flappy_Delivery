"""
Locate the discretized periodic tunnelling orbit of the reaction surface at a given inverse temperature, represented by N beads in imaginary time.

The orbit is the delocalized stationary configuration of the discretized imaginary-time action for a closed chain of N beads of total imaginary time beta hbar, and it exists only below the crossover temperature of the barrier. It is returned in the folded order fixed below, which removes the freedom to translate the beads around the chain, so that the result is unique and can be compared bead by bead.

Returns
-------
np.ndarray of shape (N, 2): bead coordinates (bohr) of the converged tunnelling orbit in folded order, bead N-1-i equal to bead i and beads 0..N/2-1 ordered by decreasing x
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ring_polymer_instanton(beta: float, n_beads: int, params: dict) -> "np.ndarray":
    '''Converged N-bead tunnelling orbit of the reaction surface.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1). The imaginary-time
        step of the chain is beta / n_beads.
    n_beads : int
        Number of beads N, an even integer >= 4. The chain is closed: bead
        N - 1 is a neighbour of bead 0.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    beads : np.ndarray
        Array of shape (N, 2) with the (x, y) coordinates in bohr of the
        beads, in folded order: bead N-1-i equals bead i, and beads
        0..N/2-1 run with strictly decreasing x from the product-side turning
        region to the reactant-side turning region. The configuration is
        converged to numerical precision, the final Newton step being below
        1e-10 times the barrier range parameter.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, or the
        search does not reach a delocalized, x-ordered stationary orbit, for
        example because the temperature is at or above the crossover
        temperature and the beads collapse onto the barrier top.
    '''
    return beads

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _half_ring_gradient_hessian(z, beta, n_beads, params):
    """Gradient and Hessian of the half-ring potential sum_i V + (k/2) sum |dz|^2."""
    half = z.shape[0]
    k_spring = float(params["m"]) * (n_beads / beta) ** 2
    grad = _oracle_potential_derivative_tensor(z, 1, params)
    grad[:-1] += k_spring * (z[:-1] - z[1:])
    grad[1:] += k_spring * (z[1:] - z[:-1])
    hess = np.zeros((2 * half, 2 * half))
    blocks = _oracle_potential_derivative_tensor(z, 2, params)
    for i in range(half):
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] = blocks[i]
    lap = np.diag(np.r_[1.0, np.full(half - 2, 2.0), 1.0])
    lap -= np.diag(np.ones(half - 1), 1) + np.diag(np.ones(half - 1), -1)
    hess += k_spring * np.kron(lap, np.eye(2))
    return grad.ravel(), hess


def _barrier_top(params):
    """Location of the maximum of V(x, 0), refined by Newton iterations."""
    a = float(params["a"])
    xs = np.linspace(-10.0 * a, 10.0 * a, 4001)
    energy = _oracle_potential_derivative_tensor(np.column_stack([xs, np.zeros_like(xs)]), 0, params)
    xb = xs[int(np.argmax(energy))]
    for _ in range(50):
        pt = np.array([[xb, 0.0]])
        g = _oracle_potential_derivative_tensor(pt, 1, params)[0, 0]
        h = _oracle_potential_derivative_tensor(pt, 2, params)[0, 0, 0]
        dx = -g / h
        xb += dx
        if abs(dx) < 1e-14 * max(1.0, abs(xb)):
            break
    return xb


def _oracle_ring_polymer_instanton(beta: float, n_beads: int, params: dict) -> "np.ndarray":
    if (isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer))
            or n_beads < 4 or n_beads % 2):
        raise ValueError("n_beads must be an even integer >= 4")
    beta = float(beta)
    if not beta > 0.0:
        raise ValueError("beta must be positive")
    a = float(params["a"])
    half = n_beads // 2
    xb = _barrier_top(params)
    theta = np.pi * (np.arange(half) + 0.5) / half
    z = np.column_stack([xb + a * np.cos(theta), np.zeros(half)])

    max_step = 0.3 * a
    # Characteristic force: spring stiffness times the barrier width plus the barrier force.
    f_ref = float(params["m"]) * (n_beads / beta) ** 2 * a + (abs(float(params["V0"])) + abs(float(params["V_inf"]))) / a
    for _ in range(500):
        grad, hess = _half_ring_gradient_hessian(z, beta, n_beads, params)
        w, v = np.linalg.eigh(hess)
        gt = v.T @ grad
        if np.max(np.abs(grad)) < 1e-6 * f_ref:
            coeff = -gt / w                      # Newton polish onto the nearby stationary point
        else:
            coeff = -gt / np.abs(w)              # eigenvector following: descend along all modes
            coeff[0] = gt[0] / abs(w[0])         # ... except ascend along the lowest one
        step = v @ coeff
        norm = np.linalg.norm(step)
        if norm > max_step:
            step *= max_step / norm
        z = z + step.reshape(half, 2)
        if norm < 1e-10 * a and np.max(np.abs(grad)) < 1e-10 * f_ref:
            break
    grad, _ = _half_ring_gradient_hessian(z, beta, n_beads, params)
    ordered = bool(np.all(np.diff(z[:, 0]) < 0.0))
    if z[0, 0] - z[-1, 0] < 1e-6 * a or not ordered or np.max(np.abs(grad)) > 1e-10 * f_ref:
        raise ValueError("no delocalized instanton found (beta at or below crossover?)")
    return np.vstack([z, z[::-1]])

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
        # Normal: asymmetric surface in the deep-tunnelling regime.
        {
            "setup": base,
            "call": "ring_polymer_instanton(2975.0, 32, dict(params))",
            "gold_call": "_oracle_ring_polymer_instanton(2975.0, 32, dict(params))",
            "tol": 1e-7,
        },
        # Normal: symmetric barrier (V_inf = 0) and more beads at a lower temperature.
        {
            "setup": base + """
params.update({"V_inf": 0.0})
""",
            "call": "ring_polymer_instanton(3900.0, 64, dict(params))",
            "gold_call": "_oracle_ring_polymer_instanton(3900.0, 64, dict(params))",
            "tol": 1e-7,
        },
        # Boundary: coarse but still resolved chain at the benchmark temperature.
        {
            "setup": base,
            "call": "ring_polymer_instanton(2975.0, 26, dict(params))",
            "gold_call": "_oracle_ring_polymer_instanton(2975.0, 26, dict(params))",
            "tol": 1e-7,
        },
        # Edge: close to the crossover temperature, where the orbit is short and compact.
        {
            "setup": base,
            "call": "ring_polymer_instanton(1750.0, 16, dict(params))",
            "gold_call": "_oracle_ring_polymer_instanton(1750.0, 16, dict(params))",
            "tol": 1e-7,
        },
        # Normal: reduced units (m = 1), a different barrier shape.
        {
            "setup": """import numpy as np
params = {"V0": 13.5 / np.pi, "V_inf": -18.0 / np.pi, "a": 8.0 / np.sqrt(3.0 * np.pi), "m": 1.0,
          "omega_e": 1.0, "chi_inf": 0.01, "chi_0": 0.01, "sigma_e": 1.0}
""",
            "call": "ring_polymer_instanton(4.0 * np.pi, 24, dict(params))",
            "gold_call": "_oracle_ring_polymer_instanton(4.0 * np.pi, 24, dict(params))",
            "tol": 1e-7,
        },
        # Invalid: odd number of beads cannot be folded.
        {
            "setup": base + """
def run_model():
    try:
        ring_polymer_instanton(2975.0, 15, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_ring_polymer_instanton(2975.0, 15, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # Invalid: above the crossover temperature the beads collapse onto the barrier top.
        {
            "setup": base + """
def run_model():
    try:
        ring_polymer_instanton(1000.0, 16, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_ring_polymer_instanton(1000.0, 16, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
