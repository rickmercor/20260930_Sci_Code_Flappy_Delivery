"""
Return one row for every interband transition available on the same uniform zone grid: its energy, and the squared magnitude of the velocity matrix element between the two states along the given axis. Rows run over grid points in the order the grid is generated, and within a grid point over the pairs of bands an optical transition can connect. The source fixes a convention here that the natural reading does not; follow the source.

Both the optical response and the geometry of the occupied states are built from the same list of transitions, so computing it once and passing it on keeps the two consistent. Which pairs of bands belong on the list is fixed by which states are filled: a pair of states that are both filled is not a transition, and neither is a pair that is both empty. The velocity operator is the wavevector derivative of the Hamiltonian, rotated into the band basis.

Returns
-------
ndarray of shape (N, 2): the transition energy in eV in the first column and the squared velocity matrix element in eV^2 Angstrom^2 in the second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interband_transitions(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int, axis: int) -> "np.ndarray":
    """Return one row for every interband transition available on the same uniform zone grid: its energy, and the squared magnitude of the velocity matrix element between the two states along the given axis. Rows run over grid points in the order the grid is generated, and within a grid point over the pairs of bands an optical transition can connect. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (N, 2): the transition energy in eV in the first column and the squared velocity matrix element in eV^2 Angstrom^2 in the second.

    Raises
    ------
    ValueError: if reciprocal does not have shape (5, 3), if n_grid is not positive, if n_occupied does not lie strictly between 0 and 11, if axis is not 0, 1 or 2, or if any forwarded argument is invalid.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_interband_transitions(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int, axis: int) -> "np.ndarray":
    r = np.asarray(reciprocal, dtype=float)
    if r.shape != (5, 3):
        raise ValueError("reciprocal must have shape (5, 3)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if not (0 < n_occupied < 11):
        raise ValueError("n_occupied must lie strictly between 0 and 11")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    b1, b2 = r[0], r[1]
    out = []
    for i in range(n_grid):
        for j in range(n_grid):
            k = (i / n_grid) * b1 + (j / n_grid) * b2
            w, u = np.linalg.eigh(_oracle_bloch_hamiltonian(k, vectors, params, -1))
            vel = u.conj().T @ _oracle_bloch_hamiltonian(k, vectors, params, axis) @ u
            # CONVENTION (source, eq. 17): only OCCUPIED -> EMPTY pairs appear; pairs inside
            # the occupied manifold destroy the gauge invariance.
            for n in range(n_occupied):
                for m in range(n_occupied, 11):
                    out.append((w[m] - w[n], abs(vel[n, m]) ** 2))
    return np.array(out, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "interband_transitions(V, R, ZR, 3, 6, 0)",
         "gold_call": "_oracle_interband_transitions(V, R, ZR, 3, 6, 0)"},   # normal
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "interband_transitions(V, R, ZR, 1, 6, 0)",
         "gold_call": "_oracle_interband_transitions(V, R, ZR, 1, 6, 0)"},   # boundary
        {"setup": "import numpy as np\nHF=np.array([-1.1472,-2.0304,1.6231,-5.6310,-4.7654,0.1490,0.1999,-0.1820,1.0371,-0.2387,-1.6199,0.9885,-0.2041,-0.0082,0.0909,0.0122,-0.0205])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])",
         "call": "interband_transitions(V, R, HF, 2, 4, 1)",
         "gold_call": "_oracle_interband_transitions(V, R, HF, 2, 4, 1)"},   # edge
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\nR=np.array([[1.703,-0.983,0.],[0.,1.966,0.],[0.,0.,0.],[0.851,-0.492,0.],[1.135,0.,0.]])\ndef _c():\n    try:\n        interband_transitions(V, R, ZR, 0, 6, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_interband_transitions(V, R, ZR, 0, 6, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
