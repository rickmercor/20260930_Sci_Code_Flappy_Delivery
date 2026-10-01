"""
Return the band gap of the monolayer, sampling the zone on a uniform grid of n_grid by n_grid points anchored at the zone centre and spanned by the two reciprocal lattice vectors, with the given number of occupied bands. The source fixes a convention here that the natural reading does not; follow the source.

The valence band maximum and the conduction band minimum of these compounds do not sit at the same wavevector, and the quantity the source reports is the one a measurement of the absorption edge would see.

Returns
-------
float, the band gap in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def indirect_gap(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int) -> float:
    """Return the band gap of the monolayer, sampling the zone on a uniform grid of n_grid by n_grid points anchored at the zone centre and spanned by the two reciprocal lattice vectors, with the given number of occupied bands. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the band gap in eV.

    Raises
    ------
    ValueError: if reciprocal does not have shape (5, 3), if n_grid is not positive, if n_occupied does not lie strictly between 0 and 11, or if any forwarded argument is invalid.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_indirect_gap(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int) -> float:
    r = np.asarray(reciprocal, dtype=float)
    if r.shape != (5, 3):
        raise ValueError("reciprocal must have shape (5, 3)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if not (0 < n_occupied < 11):
        raise ValueError("n_occupied must lie strictly between 0 and 11")
    b1, b2 = r[0], r[1]
    top, bot = -np.inf, np.inf
    for i in range(n_grid):
        for j in range(n_grid):
            e = np.linalg.eigvalsh(_oracle_bloch_hamiltonian(
                (i / n_grid) * b1 + (j / n_grid) * b2, vectors, params, -1))
            # extrema taken over the WHOLE grid independently: the gap is the INDIRECT one.
            top = max(top, e[n_occupied - 1])
            bot = min(bot, e[n_occupied])
    return bot - top

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "indirect_gap(V, R, ZR, 5, 6)",
         "gold_call": "_oracle_indirect_gap(V, R, ZR, 5, 6)"},   # normal
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "indirect_gap(V, R, ZR, 1, 6)",
         "gold_call": "_oracle_indirect_gap(V, R, ZR, 1, 6)"},   # boundary
        {"setup": "import numpy as np\nHF=np.array([-1.1472,-2.0304,1.6231,-5.6310,-4.7654,0.1490,0.1999,-0.1820,1.0371,-0.2387,-1.6199,0.9885,-0.2041,-0.0082,0.0909,0.0122,-0.0205])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "indirect_gap(V, R, HF, 4, 4)",
         "gold_call": "_oracle_indirect_gap(V, R, HF, 4, 4)"},   # edge
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])\ndef _c():\n    try:\n        indirect_gap(V, R, ZR, 5, 11)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_indirect_gap(V, R, ZR, 5, 11)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
