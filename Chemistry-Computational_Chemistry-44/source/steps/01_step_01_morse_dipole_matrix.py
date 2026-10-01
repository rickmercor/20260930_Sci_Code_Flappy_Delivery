"""
Step 01: Dipole matrix between exact Morse vibrational eigenstates.

Dipole-moment matrix of the lowest vibrational eigenstates of a Morse oscillator.

The stretching vibration of a diatomic molecule is modelled by the Morse Hamiltonian

  H0 = -(1 / (2 m)) d^2/dq^2 + D (1 - exp(-alpha q))^2,

written in atomic units (hbar = 1), where q = r - r_e is the displacement of the bond length from equilibrium in bohr,
m is the reduced mass in electron masses, D is the well depth in hartree and alpha is the range parameter in inverse
bohr. The bound eigenstates |v>, v = 0, 1, 2, ..., are the exact square-integrable eigenfunctions of H0 on the whole
real q axis (not of a truncated box or basis), normalised to one. The number of bound states is fixed by
lambda = sqrt(2 m D) / alpha: level v is bound while v < lambda - 1/2.

The molecular dipole moment along the bond is a polynomial in the displacement,
mu(q) = c_0 + c_1 q + c_2 q^2 + ..., with c_k in atomic units (e bohr^(1-k)). Its matrix in the vibrational basis,
M_vw = <v| mu(q) |w>, holds the permanent dipole moment of each level on the diagonal and the transition dipole
moments off the diagonal. The eigenfunctions are real, and the sign of each one is fixed so that it is positive at
large positive q, on the dissociation side beyond its outermost node. Every matrix element must be accurate to
1e-10 in absolute value, including for levels close to the dissociation limit, whose wavefunctions extend far out
along q.

Returns
-------
numpy.ndarray of shape (n_levels, n_levels), real symmetric dipole matrix <v|mu|w> in atomic units between exact Morse eigenfunctions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def morse_dipole_matrix(n_levels: int, mass: float, depth: float, alpha: float, dipole_coeffs: "np.ndarray") -> "np.ndarray":
    '''Matrix of a polynomial dipole function between the lowest bound Morse eigenstates.

    Parameters
    ----------
    n_levels : int
        Number of lowest bound levels v = 0 .. n_levels - 1 to include; at least 1.
    mass : float
        Reduced mass m in electron masses, positive.
    depth : float
        Well depth D in hartree, positive.
    alpha : float
        Range parameter alpha in inverse bohr, positive.
    dipole_coeffs : np.ndarray
        1-D array [c_0, c_1, ...] of the dipole polynomial mu(q) = sum_k c_k q^k in atomic units.

    Returns
    -------
    result : np.ndarray
        Real symmetric array of shape (n_levels, n_levels) with entries <v| mu(q) |w>, eigenfunctions normalised and
        positive at large positive q.

    Raises
    ------
    ValueError
        If n_levels < 1, if mass, depth or alpha is not positive, or if level n_levels - 1 is not bound
        (n_levels - 1 >= lambda - 1/2).
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import eval_genlaguerre, gammaln, roots_legendre


def _morse_wavefunctions(n_levels, mass, depth, alpha):
    """Exact normalised Morse eigenfunctions on a Gauss-Legendre panel grid in q; returns (q, weights, psi)."""
    import numpy as np
    from scipy.special import eval_genlaguerre, gammaln, roots_legendre
    lam = np.sqrt(2.0 * mass * depth) / alpha
    s_min = lam - (n_levels - 1) - 0.5
    # inner wall: z = 2 lam exp(-alpha q) far beyond the turning point of the ground state
    z_hi = 2.0 * lam + 40.0 * np.sqrt(2.0 * lam) + 150.0
    # outer tail of the least bound level: z^(2 s_min) e^(-z) negligible below z_lo
    z_lo = 2.0 * s_min * np.exp(-(110.0 + 2.0 * s_min) / (2.0 * s_min))
    q_lo = -np.log(z_hi / (2.0 * lam)) / alpha
    q_hi = -np.log(z_lo / (2.0 * lam)) / alpha
    n_pan = int(np.ceil((q_hi - q_lo) / 0.05))
    x, w = roots_legendre(16)
    edges = np.linspace(q_lo, q_hi, n_pan + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    half = 0.5 * (edges[1:] - edges[:-1])
    q = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    z = 2.0 * lam * np.exp(-alpha * q)
    psi = np.empty((n_levels, q.size))
    for v in range(n_levels):
        s = lam - v - 0.5
        log_norm = 0.5 * (np.log(alpha) + np.log(2.0 * s) + gammaln(v + 1.0) - gammaln(2.0 * lam - v))
        psi[v] = np.exp(log_norm + s * np.log(z) - 0.5 * z) * eval_genlaguerre(v, 2.0 * s, z)
    return q, weights, psi


def _oracle_morse_dipole_matrix(n_levels: int, mass: float, depth: float, alpha: float,
                                dipole_coeffs: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = int(n_levels)
    if n < 1:
        raise ValueError("n_levels must be at least 1")
    if not (mass > 0.0 and depth > 0.0 and alpha > 0.0):
        raise ValueError("mass, depth and alpha must be positive")
    lam = np.sqrt(2.0 * mass * depth) / alpha
    if n - 1 >= lam - 0.5:
        raise ValueError("level n_levels - 1 is not bound")
    coeffs = np.atleast_1d(np.asarray(dipole_coeffs, dtype=float))
    q, weights, psi = _morse_wavefunctions(n, mass, depth, alpha)
    mu = np.zeros_like(q)
    for k in range(coeffs.size - 1, -1, -1):
        mu = mu * q + coeffs[k]
    matrix = np.einsum("vi,i,wi->vw", psi, weights * mu, psi)
    return 0.5 * (matrix + matrix.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: hydrogen fluoride, five lowest levels, quadratic dipole function ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "morse_dipole_matrix(5, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_morse_dipole_matrix(5, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-9,
        },
        # --- Normal: heavier isotopologue-like oscillator with a cubic dipole function ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.43, 0.21, 0.05, -0.012])\n",
            "call": "morse_dipole_matrix(7, 3341.2, 0.1698, 1.0106, c)",
            "gold_call": "_oracle_morse_dipole_matrix(7, 3341.2, 0.1698, 1.0106, c)",
            "tol": 1e-9,
        },
        # --- Boundary: shallow well where the requested top level is the last bound one ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.0, 1.0])\n",
            "call": "morse_dipole_matrix(9, 1000.0, 0.1, 1.5, c)",
            "gold_call": "_oracle_morse_dipole_matrix(9, 1000.0, 0.1, 1.5, c)",
            "tol": 1e-8,
        },
        # --- Edge: a single level with a constant dipole returns a 1 x 1 matrix equal to that constant ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([1.25])\n",
            "call": "morse_dipole_matrix(1, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_morse_dipole_matrix(1, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-10,
        },
        # --- Boundary: very shallow well, top level just below dissociation with a far-reaching tail ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.1, 0.8, -0.05])\n",
            "call": "morse_dipole_matrix(6, 800.0, 0.02, 1.0, c)",
            "gold_call": "_oracle_morse_dipole_matrix(6, 800.0, 0.02, 1.0, c)",
            "tol": 1e-8,
        },
        # --- Normal: deep heavy well with fourteen levels and a quartic dipole function ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.2, 0.05, 0.01, -0.002, 0.0005])\n",
            "call": "morse_dipole_matrix(14, 25000.0, 0.06, 1.0, c)",
            "gold_call": "_oracle_morse_dipole_matrix(14, 25000.0, 0.06, 1.0, c)",
            "tol": 1e-9,
        },
        # --- Edge: every bound level of the HF well with a cubic dipole function ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165, 0.002])\n",
            "call": "morse_dipole_matrix(24, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_morse_dipole_matrix(24, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-8,
        },
        # --- Error: asking for more levels than the well binds must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(10, 1000.0, 0.1, 1.5, np.array([0.0, 1.0]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(morse_dipole_matrix)",
            "gold_call": "_probe(_oracle_morse_dipole_matrix)",
        },
    ]
