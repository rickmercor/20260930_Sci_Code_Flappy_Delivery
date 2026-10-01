"""
Return the eleven-band Bloch Hamiltonian at one wavevector, built from the bond vectors of the first step and the parameter vector, when deriv is -1; when deriv is 0, 1 or 2 return instead its derivative with respect to that Cartesian component of the wavevector. Place every block so the result is Hermitian, and put the crystal-field-split on-site energies on the diagonal. The source fixes a convention here that the natural reading does not; follow the source.

Each family of bonds contributes a block weighted by a Bloch phase, and the two-centre Slater-Koster integrals give each block from the direction cosines of its bond. Crystal field splits the metal d shell into a singlet and two doublets and the chalcogen p shell into an in-plane doublet and an out-of-plane singlet, and the source labels the three d levels with a shorthand whose mapping onto the basis is stated in its text rather than implied by the order the levels are printed in. The two chalcogen planes are related by the symmetry of the structure, so the six metal-chalcogen bonds split into two groups of three reaching one plane or the other, and which group a bond belongs to is fixed by the structure rather than by the order it is listed in. Families whose opposite members were left out of the first step must have them restored here. One argument selects between the matrix and its derivative because returning both through the same assembly keeps the velocity operator consistent with the Hamiltonian; the part carrying no wavevector survives only in the former.

Returns
-------
ndarray of shape (11, 11), complex: in eV when deriv is -1, in eV Angstrom otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bloch_hamiltonian(k: "np.ndarray", vectors: "np.ndarray", params: "np.ndarray", deriv: int) -> "np.ndarray":
    """Return the eleven-band Bloch Hamiltonian at one wavevector, built from the bond vectors of the first step and the parameter vector, when deriv is -1; when deriv is 0, 1 or 2 return instead its derivative with respect to that Cartesian component of the wavevector. Place every block so the result is Hermitian, and put the crystal-field-split on-site energies on the diagonal. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (11, 11), complex: in eV when deriv is -1, in eV Angstrom otherwise.

    Raises
    ------
    ValueError: if k is not a vector of length three, if vectors does not have shape (15, 3), if params does not hold seventeen entries, or if deriv is not one of -1, 0, 1, 2.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _sk_pp(direction, v_sigma, v_pi):
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if d.shape != (3,) or n == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    u = d / n
    return np.outer(u, u) * (v_sigma - v_pi) + np.eye(3) * v_pi


def _sk_dd(direction, v_sigma, v_pi, v_delta):
    d = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if d.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    l, m, n = d / nrm
    r3 = np.sqrt(3.0)
    ll, mm, nn = l * l, m * m, n * n
    S, P, D = v_sigma, v_pi, v_delta
    E = np.zeros((5, 5))
    E[0, 0] = (nn - 0.5 * (ll + mm)) ** 2 * S + 3 * nn * (ll + mm) * P + 0.75 * (ll + mm) ** 2 * D
    E[0, 1] = r3 / 2 * (ll - mm) * (nn - 0.5 * (ll + mm)) * S - r3 * nn * (ll - mm) * P \
        + r3 / 4 * (1 + nn) * (ll - mm) * D
    E[0, 2] = r3 * l * m * (nn - 0.5 * (ll + mm)) * S - 2 * r3 * l * m * nn * P \
        + r3 / 2 * l * m * (1 + nn) * D
    E[0, 3] = r3 * m * n * (nn - 0.5 * (ll + mm)) * S + r3 * m * n * (ll + mm - nn) * P \
        - r3 / 2 * m * n * (ll + mm) * D
    E[0, 4] = r3 * l * n * (nn - 0.5 * (ll + mm)) * S + r3 * l * n * (ll + mm - nn) * P \
        - r3 / 2 * l * n * (ll + mm) * D
    E[1, 1] = 0.75 * (ll - mm) ** 2 * S + (ll + mm - (ll - mm) ** 2) * P \
        + (nn + 0.25 * (ll - mm) ** 2) * D
    E[1, 2] = 1.5 * l * m * (ll - mm) * S - 2 * l * m * (ll - mm) * P + 0.5 * l * m * (ll - mm) * D
    E[1, 3] = 1.5 * m * n * (ll - mm) * S - m * n * (1 + 2 * (ll - mm)) * P \
        + m * n * (1 + 0.5 * (ll - mm)) * D
    E[1, 4] = 1.5 * l * n * (ll - mm) * S + l * n * (1 - 2 * (ll - mm)) * P \
        - l * n * (1 - 0.5 * (ll - mm)) * D
    E[2, 2] = 3 * ll * mm * S + (ll + mm - 4 * ll * mm) * P + (nn + ll * mm) * D
    E[2, 3] = 3 * l * mm * n * S + l * n * (1 - 4 * mm) * P + l * n * (mm - 1) * D
    E[2, 4] = 3 * ll * m * n * S + m * n * (1 - 4 * ll) * P + m * n * (ll - 1) * D
    E[3, 3] = 3 * mm * nn * S + (mm + nn - 4 * mm * nn) * P + (ll + mm * nn) * D
    E[3, 4] = 3 * m * nn * l * S + m * l * (1 - 4 * nn) * P + m * l * (nn - 1) * D
    E[4, 4] = 3 * nn * ll * S + (nn + ll - 4 * nn * ll) * P + (mm + nn * ll) * D
    return E + np.triu(E, 1).T


def _sk_pd(direction, v_sigma, v_pi):
    d = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if d.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    l, m, n = d / nrm
    r3 = np.sqrt(3.0)
    ll, mm, nn = l * l, m * m, n * n
    S, P = v_sigma, v_pi
    E = np.zeros((3, 5))
    E[0, 0] = l * (nn - 0.5 * (ll + mm)) * S - r3 * l * nn * P
    E[1, 0] = m * (nn - 0.5 * (ll + mm)) * S - r3 * m * nn * P
    E[2, 0] = n * (nn - 0.5 * (ll + mm)) * S + r3 * n * (ll + mm) * P
    E[0, 1] = r3 / 2 * l * (ll - mm) * S + l * (1 - ll + mm) * P
    E[1, 1] = r3 / 2 * m * (ll - mm) * S - m * (1 + ll - mm) * P
    E[2, 1] = r3 / 2 * n * (ll - mm) * S - n * (ll - mm) * P
    E[0, 2] = r3 * ll * m * S + m * (1 - 2 * ll) * P
    E[1, 2] = r3 * mm * l * S + l * (1 - 2 * mm) * P
    E[2, 2] = r3 * l * m * n * S - 2 * l * m * n * P
    E[0, 3] = r3 * l * m * n * S - 2 * l * m * n * P
    E[1, 3] = r3 * mm * n * S + n * (1 - 2 * mm) * P
    E[2, 3] = r3 * nn * m * S + m * (1 - 2 * nn) * P
    E[0, 4] = r3 * ll * n * S + n * (1 - 2 * ll) * P
    E[1, 4] = r3 * l * m * n * S - 2 * l * m * n * P
    E[2, 4] = r3 * nn * l * S + l * (1 - 2 * nn) * P
    return E


def _onsite(p):
    # CONVENTION (source, shorthand after eq. 7, NOT eq. 7 read positionally): d1 is the
    # {dyz, dzx} doublet and d2 is {dx2-y2, dxy}. The positional reading is the reverse and
    # misses both published gaps.
    diag = np.empty(11)
    diag[0:3] = [p[3], p[3], p[4]]
    diag[3:8] = [p[0], p[2], p[2], p[1], p[1]]
    diag[8:11] = [p[3], p[3], p[4]]
    return np.diag(diag)


def _oracle_bloch_hamiltonian(k: "np.ndarray", vectors: "np.ndarray", params: "np.ndarray", deriv: int) -> "np.ndarray":
    kk = np.asarray(k, dtype=float)
    v = np.asarray(vectors, dtype=float)
    p = np.asarray(params, dtype=float)
    if kk.shape != (3,):
        raise ValueError("k must be a vector of length 3")
    if v.shape != (15, 3):
        raise ValueError("vectors must have shape (15, 3)")
    if p.shape != (17,):
        raise ValueError("params must have exactly 17 entries")
    if deriv not in (-1, 0, 1, 2):
        raise ValueError("deriv must be -1, 0, 1 or 2")
    Vdds, Vddp, Vddd, Vpps, Vppp, Vpds, Vpdp = p[5:12]
    Kdds, Kddp, Kddd, Kpps, Kppp = p[12:17]
    # the on-site block carries no k, so it survives only in the matrix itself.
    H = _onsite(p).astype(complex) if deriv < 0 else np.zeros((11, 11), dtype=complex)

    # CONVENTION (source, eq. 8): half of each bond family is listed; the opposite members
    # return as the FACTOR TWO on the cosine.
    for rows, sk in ((slice(6, 9), (Vdds, Vddp, Vddd, Vpps, Vppp)),
                     (slice(9, 12), (Kdds, Kddp, Kddd, Kpps, Kppp))):
        for d in v[rows]:
            f = 2.0 * np.cos(kk @ d) if deriv < 0 else -2.0 * d[deriv] * np.sin(kk @ d)
            H[3:8, 3:8] += f * _sk_dd(d, sk[0], sk[1], sk[2])
            blk = f * _sk_pp(d, sk[3], sk[4])
            H[0:3, 0:3] += blk
            H[8:11, 8:11] += blk

    top = np.zeros((5, 3), dtype=complex)
    bot = np.zeros((5, 3), dtype=complex)
    for i, d in enumerate(v[0:6], start=1):
        ph = np.exp(1j * (kk @ d))
        if deriv >= 0:
            ph = 1j * d[deriv] * ph
        # CONVENTION (source, eq. 9 / App. A, where the matrices are 5x3): the M-X block is
        # metal-index-first, so the p-d block enters TRANSPOSED and carries the phase there.
        blk = ph * _sk_pd(d, Vpds, Vpdp).T
        # CONVENTION (source, Fig. 2 / eq. 9): the six M-X bonds alternate between the two
        # chalcogen planes, and in the listing order of step 1 the FIRST one reaches the top
        # plane. (Getting this backwards is the same error as the orientation above.)
        if i % 2 == 1:
            top += blk
        else:
            bot += blk
    H[3:8, 0:3] += top
    H[0:3, 3:8] += top.conj().T
    H[3:8, 8:11] += bot
    H[8:11, 3:8] += bot.conj().T

    tb = np.zeros((3, 3), dtype=complex)
    for d in v[12:15]:
        ph = np.exp(1j * (kk @ d))
        if deriv >= 0:
            ph = 1j * d[deriv] * ph
        # CONVENTION (source, App. B, sentence introducing r0): the coupling ACROSS the metal
        # plane uses the NEAREST-neighbour Vpp integrals, not the next-nearest Kpp pair.
        tb += ph * _sk_pp(d, Vpps, Vppp)
    H[0:3, 8:11] += tb
    H[8:11, 0:3] += tb.conj().T
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "bloch_hamiltonian(np.array([0.31,-0.17,0.0]), V, ZR, -1)",
         "gold_call": "_oracle_bloch_hamiltonian(np.array([0.31,-0.17,0.0]), V, ZR, -1)"},   # normal
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "bloch_hamiltonian(np.zeros(3), V, ZR, -1)",
         "gold_call": "_oracle_bloch_hamiltonian(np.zeros(3), V, ZR, -1)"},   # boundary
        {"setup": "import numpy as np\nHF=np.array([-1.1472,-2.0304,1.6231,-5.6310,-4.7654,0.1490,0.1999,-0.1820,1.0371,-0.2387,-1.6199,0.9885,-0.2041,-0.0082,0.0909,0.0122,-0.0205])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6",
         "call": "bloch_hamiltonian(np.array([1.7,0.9,0.0]), V, HF, 1)",
         "gold_call": "_oracle_bloch_hamiltonian(np.array([1.7,0.9,0.0]), V, HF, 1)"},   # edge
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\nV=np.arange(45,dtype=float).reshape(15,3)*0.17-2.6\ndef _c():\n    try:\n        bloch_hamiltonian(np.array([0.31,-0.17,0.0]), V, ZR, 3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_bloch_hamiltonian(np.array([0.31,-0.17,0.0]), V, ZR, 3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
