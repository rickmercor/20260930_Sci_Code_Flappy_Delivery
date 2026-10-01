"""
Step 02: Level populations of a vibrational ladder under a classical cosine field.

Level populations of a truncated vibrational ladder driven by a classical monochromatic field.

A molecule is described by n vibrational levels with energies E_v (hartree) and a real symmetric dipole matrix
M_vw = <v| mu |w> (atomic units) that includes the permanent dipole moments on its diagonal. At t = 0 a linearly
polarised classical field E(t) = F cos(omega t) along the bond is switched on, and the molecule is in level v = 0.
In the electric-dipole approximation the amplitudes c_v(t) of the state sum_v c_v(t) |v> obey the time-dependent
Schroedinger equation (atomic units, hbar = 1)

  i dc/dt = [diag(E) - F cos(omega t) M] c,     c(0) = (1, 0, ..., 0),

with every term of the coupling kept: the counter-rotating parts of the cosine and the diagonal (permanent dipole)
elements are not dropped. The level populations are rho_v(t) = |c_v(t)|^2.

The populations are sampled on the uniform grid t_k = k dt, k = 0, 1, ..., N with N = t_end / dt. The sampling interval
is only the spacing at which results are reported; the propagation itself has to resolve the field and the level
energies, and every returned population must be accurate to 1e-8 even when dt is several atomic time units.

Returns
-------
numpy.ndarray of shape (N + 1, n) with N = t_end / dt, level populations at t_k = k dt starting from (1, 0, ..., 0)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def driven_level_populations(energies: "np.ndarray", dipole: "np.ndarray", field_amplitude: float, omega: float,
                             t_end: float, dt: float) -> "np.ndarray":
    '''Populations of all levels on a uniform time grid for a ladder started in level 0 under F cos(omega t).

    Parameters
    ----------
    energies : np.ndarray
        Shape (n,), level energies E_v in hartree, n >= 2.
    dipole : np.ndarray
        Shape (n, n), real symmetric dipole matrix in atomic units, diagonal included.
    field_amplitude : float
        Peak field F in atomic units of field strength, non-negative.
    omega : float
        Angular frequency of the field in hartree, positive.
    t_end : float
        Final time in atomic time units, positive and an integer multiple of dt.
    dt : float
        Sampling interval in atomic time units, positive.

    Returns
    -------
    result : np.ndarray
        Shape (N + 1, n) with N = t_end / dt; row k holds rho_v(k dt), and row 0 is (1, 0, ..., 0).

    Raises
    ------
    ValueError
        If the shapes do not match, dt or t_end is not positive, or t_end is not an integer multiple of dt.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _apply_sequence(matrices, vector, block=32):
    """States v_k = A_(k-1) ... A_0 v for k = 0..K from a stack of K matrices, using batched block products."""
    import numpy as np
    count, n, _ = matrices.shape
    dtype = np.result_type(matrices.dtype, np.asarray(vector).dtype)
    n_blocks = -(-count // block)
    pad = n_blocks * block - count
    mats = matrices.astype(dtype)
    if pad:
        mats = np.concatenate([mats, np.broadcast_to(np.eye(n, dtype=dtype), (pad, n, n))], axis=0)
    mats = mats.reshape(n_blocks, block, n, n)
    prefix = np.empty_like(mats)
    prefix[:, 0] = mats[:, 0]
    for j in range(1, block):
        prefix[:, j] = mats[:, j] @ prefix[:, j - 1]
    starts = np.empty((n_blocks, n), dtype=dtype)
    current = np.asarray(vector, dtype=dtype)
    for b in range(n_blocks):
        starts[b] = current
        current = prefix[b, -1] @ current
    states = np.einsum("bjik,bk->bji", prefix, starts).reshape(n_blocks * block, n)[:count]
    return np.concatenate([np.asarray(vector, dtype=dtype)[None], states], axis=0)


def _oracle_driven_level_populations(energies: "np.ndarray", dipole: "np.ndarray", field_amplitude: float,
                                     omega: float, t_end: float, dt: float) -> "np.ndarray":
    """Reference implementation: fourth-order Magnus propagator (two Gauss points) on fine substeps."""
    import numpy as np
    e = np.asarray(energies, dtype=float).ravel()
    m = np.asarray(dipole, dtype=float)
    n = e.size
    if n < 2 or m.shape != (n, n):
        raise ValueError("energies must have shape (n,) with n >= 2 and dipole shape (n, n)")
    if not (dt > 0.0 and t_end > 0.0):
        raise ValueError("dt and t_end must be positive")
    n_samples = int(round(t_end / dt))
    if n_samples < 1 or abs(n_samples * dt - t_end) > 1e-9 * max(1.0, t_end):
        raise ValueError("t_end must be an integer multiple of dt")
    # substeps short against the field period and the coupling strength (H0 itself is treated exactly)
    rate = max(omega, abs(field_amplitude) * np.max(np.abs(m).sum(axis=1)))
    n_sub = max(1, int(np.ceil(dt * rate / 0.02)))
    h = dt / n_sub
    t = h * np.arange(n_samples * n_sub)
    g = np.sqrt(3.0) / 6.0
    f1 = -field_amplitude * np.cos(omega * (t + h * (0.5 - g)))
    f2 = -field_amplitude * np.cos(omega * (t + h * (0.5 + g)))
    h0 = np.diag(e)
    comm = m @ h0 - h0 @ m
    # exp(Omega) with Omega = -i h (H0 + fbar M) - (sqrt(3) h^2 / 12) (f2 - f1) [M, H0] = -i K, K Hermitian
    k_mat = (h * (h0[None] + (0.5 * (f1 + f2))[:, None, None] * m[None])
             - 1j * (np.sqrt(3.0) * h * h / 12.0) * (f2 - f1)[:, None, None] * comm[None])
    norm = float(np.max(np.abs(k_mat).sum(axis=-1)))
    squarings = 0
    while norm / 2 ** squarings > 0.1:
        squarings += 1
    a_mat = (-1j / 2 ** squarings) * k_mat
    identity = np.broadcast_to(np.eye(n, dtype=complex), k_mat.shape)
    prop = identity.copy()
    for j in range(18, 0, -1):
        prop = identity + (a_mat @ prop) / j
    for _ in range(squarings):
        prop = prop @ prop
    if n_sub > 1:
        prop = _apply_sequence_products(prop.reshape(n_samples, n_sub, n, n))
    c0 = np.zeros(n, dtype=complex)
    c0[0] = 1.0
    states = _apply_sequence(prop, c0)
    return states.real ** 2 + states.imag ** 2


def _apply_sequence_products(groups):
    """Ordered products A_(s-1) ... A_0 within each group of a stack of shape (K, s, n, n)."""
    total = groups[:, 0]
    for j in range(1, groups.shape[1]):
        total = groups[:, j] @ total
    return total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: five-level anharmonic ladder driven near its two-photon resonance, reported every 1 a.u. ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.0098, 0.0288, 0.0470, 0.0643, 0.0808])\n"
                     "d = np.array([[0.65, 0.042, -0.005, 0.001, -0.0002],\n"
                     "              [0.042, 0.67, 0.059, -0.008, 0.0015],\n"
                     "              [-0.005, 0.059, 0.69, 0.072, -0.012],\n"
                     "              [0.001, -0.008, 0.072, 0.71, 0.083],\n"
                     "              [-0.0002, 0.0015, -0.012, 0.083, 0.73]])\n",
            "call": "driven_level_populations(e, d, 0.0055, 0.0186, 3000.0, 1.0)[::50]",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.0055, 0.0186, 3000.0, 1.0)[::50]",
            "tol": 1e-7,
        },
        # --- Normal: two-level resonant drive sampled coarsely, so propagation must use internal substeps ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.0093389939, 0.0274231983])\n"
                     "d = np.array([[0.717431, 0.0392166], [0.0392166, 0.7346147]])\n",
            "call": "driven_level_populations(e, d, 0.01, 0.0180842044, 12500.0, 25.0)",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.01, 0.0180842044, 12500.0, 25.0)",
            "tol": 1e-7,
        },
        # --- Boundary: zero field leaves the molecule in level 0 for all times ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.01, 0.03, 0.048])\n"
                     "d = np.array([[0.7, 0.04, -0.005], [0.04, 0.72, 0.055], [-0.005, 0.055, 0.74]])\n",
            "call": "driven_level_populations(e, d, 0.0, 0.02, 400.0, 4.0)",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.0, 0.02, 400.0, 4.0)",
            "tol": 1e-10,
        },
        # --- Edge: strong field off resonance with large permanent-dipole difference, three levels ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.0, 0.02, 0.038])\n"
                     "d = np.array([[0.2, 0.3, 0.0], [0.3, 1.4, 0.4], [0.0, 0.4, -0.6]])\n",
            "call": "driven_level_populations(e, d, 0.02, 0.017, 900.0, 3.0)",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.02, 0.017, 900.0, 3.0)",
            "tol": 1e-7,
        },
        # --- Normal: six levels, strong low-frequency field, samples fifty atomic time units apart ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.0, 0.0181, 0.0355, 0.0521, 0.068, 0.083])\n"
                     "d = np.array([[1.1, 0.2, -0.03, 0.01, 0.0, 0.0], [0.2, 1.3, 0.28, -0.05, 0.01, 0.0],\n"
                     "              [-0.03, 0.28, 1.5, 0.33, -0.07, 0.01], [0.01, -0.05, 0.33, 1.7, 0.37, -0.09],\n"
                     "              [0.0, 0.01, -0.07, 0.37, 1.9, 0.4], [0.0, 0.0, 0.01, -0.09, 0.4, 2.1]])\n",
            "call": "driven_level_populations(e, d, 0.03, 0.006, 2000.0, 50.0)",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.03, 0.006, 2000.0, 50.0)",
            "tol": 1e-7,
        },
        # --- Boundary: two nearly degenerate upper levels resonantly driven from the ground state ---
        {
            "setup": "import numpy as np\n"
                     "e = np.array([0.0, 0.02, 0.020002])\n"
                     "d = np.array([[0.5, 0.1, 0.1], [0.1, 0.6, 0.0], [0.1, 0.0, 0.7]])\n",
            "call": "driven_level_populations(e, d, 0.004, 0.02, 6000.0, 20.0)",
            "gold_call": "_oracle_driven_level_populations(e, d, 0.004, 0.02, 6000.0, 20.0)",
            "tol": 1e-7,
        },
        # --- Error: t_end that is not a multiple of dt must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.0, 0.02]), np.array([[0.1, 0.2], [0.2, 0.3]]), 0.01, 0.02, 10.5, 2.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(driven_level_populations)",
            "gold_call": "_probe(_oracle_driven_level_populations)",
        },
    ]
