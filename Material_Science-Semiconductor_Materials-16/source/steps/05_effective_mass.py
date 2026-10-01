"""
Return the effective mass of one band at one band extremum along one direction in momentum space, in units of the free electron mass, by fitting a parabola to that band on n_points samples spaced dk apart and centred on the extremum. The source fixes a convention here that the natural reading does not; follow the source.

Near an extremum a band is parabolic and its curvature defines a mass. The samples are placed symmetrically about the extremum, which makes the quadratic coefficient of the least-squares parabola available in closed form. The source quotes masses for a band maximum and a band minimum in the same table and in the same way.

Returns
-------
float, the effective mass in units of the free electron mass.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_mass(vectors: "np.ndarray", params: "np.ndarray", band: int, k_extremum: "np.ndarray", direction: "np.ndarray", dk: float, n_points: int) -> float:
    """Return the effective mass of one band at one band extremum along one direction in momentum space, in units of the free electron mass, by fitting a parabola to that band on n_points samples spaced dk apart and centred on the extremum. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the effective mass in units of the free electron mass.

    Raises
    ------
    ValueError: if direction or k_extremum is not a vector of length three, if direction is zero, if dk is not positive, if n_points is not an odd integer of at least three, or if the band is flat along the chosen direction.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_effective_mass(vectors: "np.ndarray", params: "np.ndarray", band: int, k_extremum: "np.ndarray", direction: "np.ndarray", dk: float, n_points: int) -> float:
    u = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(u)
    if u.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    if dk <= 0:
        raise ValueError("dk must be positive")
    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd integer of at least 3")
    k0 = np.asarray(k_extremum, dtype=float)
    if k0.shape != (3,):
        raise ValueError("k_extremum must be a vector of length 3")
    u = u / nrm
    half = (n_points - 1) // 2
    t = dk * np.arange(-half, half + 1, dtype=float)
    e = np.array([np.linalg.eigvalsh(
        _oracle_bloch_hamiltonian(k0 + s * u, vectors, params, -1))[band] for s in t])
    # symmetric nodes decouple the linear term, so the least-squares quadratic coefficient
    # is closed form; using it avoids BLAS-dependent drift from a least-squares solve.
    n = float(n_points)
    s2, s4 = (t ** 2).sum(), (t ** 4).sum()
    c2 = (n * (t ** 2 * e).sum() - s2 * e.sum()) / (n * s4 - s2 * s2)
    if c2 == 0.0:
        raise ValueError("the band is flat along this direction; no parabolic mass")
    # CONVENTION (source, Table V): masses quoted POSITIVE at both band edges.
    hbar2_over_me = 7.6199682  # hbar^2 / m_e in eV Angstrom^2
    return abs(hbar2_over_me / (2.0 * c2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,-0.57735,0.0]), 0.01, 7)",
         "gold_call": "_oracle_effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,-0.57735,0.0]), 0.01, 7)"},   # normal
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,0.0,0.0]), 0.01, 3)",
         "gold_call": "_oracle_effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,0.0,0.0]), 0.01, 3)"},   # boundary
        {"setup": "import numpy as np\nHF=np.array([-1.1472,-2.0304,1.6231,-5.6310,-4.7654,0.1490,0.1999,-0.1820,1.0371,-0.2387,-1.6199,0.9885,-0.2041,-0.0082,0.0909,0.0122,-0.0205])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "effective_mass(V, HF, 6, np.array([0.86,-0.4965,0.0]), np.array([-0.86,0.4965,0.0]), 0.005, 9)",
         "gold_call": "_oracle_effective_mass(V, HF, 6, np.array([0.86,-0.4965,0.0]), np.array([-0.86,0.4965,0.0]), 0.005, 9)"},   # edge
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\ndef _c():\n    try:\n        effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,0.0,0.0]), 0.01, 4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_effective_mass(V, ZR, 5, np.zeros(3), np.array([1.0,0.0,0.0]), 0.01, 4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
