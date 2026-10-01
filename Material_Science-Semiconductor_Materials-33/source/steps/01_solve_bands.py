"""
Diagonalize the specified two-orbital, spinless semiconductor model.

The orthonormal orbitals use the lattice Bloch gauge exp(i k.R). Positions

inside the cell are included in density vertices, not in this Hamiltonian.

The two columns of each eigenvector matrix label valence then conduction.

Eigenvector phases are unrestricted and are not graded.

Returns
-------
energies, vectors with shapes (K, 2) and (K, 2, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_bands(
    k_points: "np.ndarray", lattice: "np.ndarray", model: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Return band energies and orthonormal eigenvectors in the lattice gauge.

    Parameters
    ----------
    k_points : "np.ndarray"
        Finite real array (K, 2), K >= 1, of Cartesian momenta in inverse
        angstroms; momenta may lie outside the first Brillouin zone.
    lattice : "np.ndarray"
        Positive finite real array (2,), the rectangular lattice lengths
        (a_x, a_y) in angstroms.
    model : "np.ndarray"
        Finite real array (7,) containing (m, b_x, b_y, t_0, t_x, t_y,
        lambda), all in eV, with m > abs(b_x) + abs(b_y).
        Set x = a_x*k_x, y = a_y*k_y, h_z = m + b_x*cos(x) + b_y*cos(y),
        and f = t_0 + t_x*exp(-i*x) + t_y*exp(-i*y)
        + i*lambda*cos(x-y). The Hamiltonian is [[h_z, f], [f*, -h_z]].

    Returns
    -------
    energies : "np.ndarray"
        Real array (K, 2) in eV, ordered valence then conduction.
    vectors : "np.ndarray"
        Complex array (K, 2, 2), indexed by momentum, orbital, band.
        Columns are normalized eigenvectors; any column phases are valid.

    Raises
    ------
    ValueError
        If the shapes, finiteness, real-valuedness, positive lattice lengths,
        or strict mass bound above are violated.
    """
    return (0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value, dtype, name):
    """Convert numeric input without modifying it and reject invalid data."""
    try:
        if dtype is float and np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_scalar(value, name, minimum=0.0):
    """Validate a finite real scalar with an inclusive lower bound."""
    result = _finite_array(value, float, name)
    if result.ndim != 0 or result < minimum:
        raise ValueError(f"{name} must be a scalar >= {minimum}")
    return float(result)


def _integer_scalar(value, name, minimum):
    """Validate an integer parameter, excluding booleans."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _oracle_solve_bands(
    k_points: "np.ndarray", lattice: "np.ndarray", model: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference Bloch eigensystem; arbitrary normalized band phases."""
    k = _finite_array(k_points, float, "k_points")
    a = _finite_array(lattice, float, "lattice")
    pars = _finite_array(model, float, "model")
    if k.ndim != 2 or k.shape[1] != 2 or len(k) < 1:
        raise ValueError("k_points must have shape (K, 2), K >= 1")
    if a.shape != (2,) or np.any(a <= 0):
        raise ValueError("lattice must contain two positive lengths")
    if pars.shape != (7,) or pars[0] <= abs(pars[1]) + abs(pars[2]):
        raise ValueError("model must satisfy the documented mass bound")
    x, y = (k * a).T
    m, bx, by, t0, tx, ty, lam = pars
    hz = m + bx * np.cos(x) + by * np.cos(y)
    off = t0 + tx * np.exp(-1j * x) + ty * np.exp(-1j * y)
    off += 1j * lam * np.cos(x - y)
    h = np.zeros((len(k), 2, 2), dtype=complex)
    h[:, 0, 0], h[:, 1, 1] = hz, -hz
    h[:, 0, 1], h[:, 1, 0] = off, off.conj()
    return np.linalg.eigh(h)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Gauge-neutral band tests plus a declared invalid-input case."""
    pack = """import numpy as np


def pack(result):
    e, u = result
    e, u = np.asarray(e), np.asarray(u)
    assert e.shape == (len(k), 2)
    assert u.shape == (len(k), 2, 2)
    projectors = np.einsum("kan,kbn->knab", u, u.conj())
    return np.concatenate((e.ravel(), projectors.ravel()))


"""
    return [
        {
            "setup": pack
            + """k = np.array([[0.0, 0.0], [0.37, -0.28], [-0.52, 0.41]])
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
""",
            "call": ("pack(solve_bands(k.copy(), a.copy(), p.copy()))"),
            "gold_call": (
                "pack(_oracle_solve_bands(k.copy(), a.copy(), " "p.copy()))"
            ),
        },
        {
            "setup": pack
            + """k = np.array([[0.0, 0.0], [0.5, -0.2]])
a = np.array([2.0, 3.0])
p = np.array([1.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
""",
            "call": ("pack(solve_bands(k.copy(), a.copy(), p.copy()))"),
            "gold_call": (
                "pack(_oracle_solve_bands(k.copy(), a.copy(), " "p.copy()))"
            ),
        },
        {
            "setup": pack
            + """k = np.array([[0.4, -0.3], [-0.4, 0.3]])
a = np.array([3.0, 2.5])
p = np.array([0.9, 0.2, -0.1, -0.3, 0.7, 0.5, 0.0])
""",
            "call": ("pack(solve_bands(k.copy(), a.copy(), p.copy()))"),
            "gold_call": (
                "pack(_oracle_solve_bands(k.copy(), a.copy(), " "p.copy()))"
            ),
        },
        {
            "setup": pack
            + """k = np.array([[3.7, -4.1]])
a = np.array([1.2, 1.8])
p = np.array([0.301, 0.2, 0.1, 0.7, -0.4, 0.2, -0.35])
""",
            "call": ("pack(solve_bands(k.copy(), a.copy(), p.copy()))"),
            "gold_call": (
                "pack(_oracle_solve_bands(k.copy(), a.copy(), " "p.copy()))"
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(np.zeros((1, 2)), np.ones(2), np.zeros(7))
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(solve_bands)",
            "gold_call": "rejected(_oracle_solve_bands)",
        },
    ]
