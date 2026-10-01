"""
Assemble the full confined-hole Hamiltonian for an arbitrary in-plane direction.

The confined hole Hamiltonian is the spin-orbit term plus the Luttinger-Kohn kinetic
terms on the product of envelope space and the six-dimensional internal space. The
argument k_inplane is the magnitude of the in-plane wave vector, while direction gives
its two Cartesian components before normalisation. Consequently, scaling direction or
reversing both of its components must not change the spectrum. The growth momentum is
the matrix from the preceding step and its square is diagonal. Retain every axial and
pair block, including the two terms linear in the growth momentum, multiply the kinetic
part by hbar^2/(2 m0) = 38.0998212 meV nm^2, and add the spin-orbit block. Use the product
basis with the envelope index slower and the internal index faster.

Returns
-------
numpy.ndarray of shape (6*num_modes, 6*num_modes), complex, in meV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hole_hamiltonian(width: float, k_inplane: float, direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    '''Full confined-hole Hamiltonian for one in-plane wave vector.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    k_inplane : float
        Magnitude of the in-plane wave vector, in 1/nm. Must be finite and
        non-negative.
    direction : tuple
        Two finite Cartesian components defining the in-plane direction. The vector
        need not be normalised but must have nonzero norm.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.

    Returns
    -------
    numpy.ndarray
        Hermitian complex array of shape (6*num_modes, 6*num_modes) in meV.

    Raises
    ------
    ValueError
        If an argument is out of range, direction is not a finite nonzero
        two-vector, or params is incomplete.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_hole_hamiltonian(width: float, k_inplane: float, direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    lw = float(width)
    if not np.isfinite(lw) or lw <= 0.0:
        raise ValueError("width must be finite and strictly positive")
    kk = float(k_inplane)
    if not np.isfinite(kk) or kk < 0.0:
        raise ValueError("k_inplane must be finite and non-negative")
    try:
        unit = np.asarray(direction, dtype=float)
    except Exception as exc:
        raise ValueError("direction must be a finite nonzero two-vector") from exc
    if unit.shape != (2,) or not np.all(np.isfinite(unit)):
        raise ValueError("direction must be a finite nonzero two-vector")
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be a finite nonzero two-vector")
    unit = unit / norm
    if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)) or int(num_modes) < 1:
        raise ValueError("num_modes must be an integer >= 1")
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for key in ("gamma1", "gamma2", "gamma3", "eta1", "eta2", "eta3", "delta"):
        if key not in params:
            raise ValueError("params is missing the key " + key)
        try:
            value = float(params[key])
        except Exception as exc:
            raise ValueError("params[" + key + "] must be finite") from exc
        if not np.isfinite(value):
            raise ValueError("params[" + key + "] must be finite")
    if float(params["delta"]) <= 0.0:
        raise ValueError("params['delta'] must be strictly positive")

    m = int(num_modes)
    lk = _oracle_luttinger_kohn_matrices(
        params["gamma1"], params["gamma2"], params["gamma3"],
        params["eta1"], params["eta2"], params["eta3"],
    )
    h_so = _oracle_spin_orbit_hamiltonian(float(params["delta"]))
    pz = _oracle_momentum_matrix_elements(lw, m)
    pz2 = np.diag([(n * np.pi / lw) ** 2 for n in range(1, m + 1)]).astype(complex)
    eye_m = np.eye(m, dtype=complex)
    kx, ky = kk * unit

    ham = np.kron((kx * kx + ky * ky) * eye_m + pz2, lk[0].astype(complex))
    ham += np.kron(kx * kx * eye_m, lk[1].astype(complex))
    ham += np.kron(ky * ky * eye_m, lk[2].astype(complex))
    ham += np.kron(pz2, lk[3].astype(complex))
    ham += np.kron(kx * ky * eye_m, lk[4].astype(complex))
    ham += np.kron(ky * pz, lk[5].astype(complex))
    ham += np.kron(kx * pz, lk[6].astype(complex))
    ham = 38.0998212 * ham + np.kron(eye_m, h_so.astype(complex))
    return 0.5 * (ham + ham.conj().T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = "import numpy as np\nCU2O = {'gamma1': 1.76, 'gamma2': 0.7532, 'gamma3': -0.3668, 'eta1': -0.020, 'eta2': -0.0037, 'eta3': -0.0337, 'delta': 131.0}"
    return [
        {
            "setup": setup,
            "call": "hole_hamiltonian(6.4, 0.35, (1.0, 1.0), 6, CU2O)",
            "gold_call": "_oracle_hole_hamiltonian(6.4, 0.35, (1.0, 1.0), 6, CU2O)",
        },
        {
            "setup": setup,
            "call": "hole_hamiltonian(6.4, 0.27, (1.0, 0.0), 4, CU2O)",
            "gold_call": "_oracle_hole_hamiltonian(6.4, 0.27, (1.0, 0.0), 4, CU2O)",
        },
        {
            "setup": setup,
            "call": "hole_hamiltonian(12.0, 0.2, (2.0, 1.0), 3, CU2O)",
            "gold_call": "_oracle_hole_hamiltonian(12.0, 0.2, (2.0, 1.0), 3, CU2O)",
        },
    ]
