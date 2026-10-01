"""
Two-electron integral of the twelve-Gaussian model in its Loewdin-orthonormal basis.

This step builds the four-centre electron-repulsion integrals over the primitive spherical

Gaussians of the model, carries them into the same symmetrically orthonormalised basis the

one-electron step uses, and returns the single physicists'-notation element

``g[p, q, r, s] = <pq|rs>``.



Scientific background.  The primitives are the helix of normalised s-type Gaussians with

alpha_p = 0.42 + 0.06 p centred at (1.35 cos(2.4 p), 1.35 sin(2.4 p), 0.62 p) bohr, in

atomic units.  For four primitives a, b, c, d write p = a_exp + b_exp, q = c_exp + d_exp,

P and Q for the two Gaussian product centres and K_ab, K_cd for the two product

prefactors.  The Coulomb repulsion integral in chemists' notation is the standard closed

form



(ab|cd) = N_a N_b N_c N_d * 2 pi^{5/2} / (p q sqrt(p + q)) * K_ab K_cd

              * F_0( p q / (p + q) * |P - Q|^2 ),



with F_0 the zeroth Boys function, F_0(0) = 1.  Nothing is screened or approximated: all

n_basis^4 elements are formed.



Two conventions have to be handled and both move the digits downstream.  First, the change

of basis.  With X = S^{-1/2} the symmetric orthonormaliser of the primitive overlap matrix

and the input ordering of the functions kept, the chemists'-notation array transforms as a

four-index congruence,



(pq|rs)_orth = sum_{abcd} X_ap X_bq X_cr X_ds (ab|cd),



i.e. each of the four primitive indices is contracted with the *first* index of X, exactly

matching the h = X^T (T + V) X convention of the one-electron step; mixing the two

conventions is invisible while the transform is symmetric and breaks the moment a general

orthogonal rotation is applied later in the pipeline.  Second, the notation.  Chemists'

(pq|rs) pairs the indices as (11|22), while the physicists' <pq|rs> used everywhere in the

second-quantised Hamiltonian H = h_pq p^+ q + (1/2) <pq|rs> p^+ q^+ s r pairs them as

<12|12>, so the two are related by the transposition of the middle two indices,

<pq|rs> = (pr|qs).  The returned array therefore satisfies <pq|rs> = <qp|sr> =

<rs|pq> = <rq|ps> for the real orbitals used here.



Raises ValueError when n_basis is not an integer in the range 1 to 16 inclusive, and when

any of p, q, r, s is not an integer in the range 0 to n_basis - 1 inclusive.

Returns
-------
float, the physicists'-notation two-electron integral <pq|rs> over the Loewdin-orthonormal basis of the n_basis-function model, in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orthonormal_two_electron_element(n_basis: int, p: int, q: int, r: int, s: int) -> float:
    """One physicists'-notation two-electron integral of the orthonormalised model.

    Parameters
    ----------
    n_basis : int
        Number of spherical Gaussian primitives in the helix, 1 <= n_basis <= 16.  The
        frozen model system uses n_basis = 12.
    p, q, r, s : int
        Indices of the requested element, each in the range 0 to n_basis - 1.

    Returns
    -------
    float
        <pq|rs> in hartree, i.e. the chemists' integral (pr|qs) over the Loewdin
        orthonormal basis X = S^{-1/2} of the primitive helix, with the four primitive
        indices contracted against the first index of X.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

from scipy.special import erf

def _check_index(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _boys_zero(x):
    """Zeroth Boys function F_0(x), with the removable singularity at x = 0 handled."""
    if x <= 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))

def _helix_basis(n_basis):
    """Exponents, centres and normalisation constants of the n_basis primitives."""
    angle_step = 2.4
    radius = 1.35
    pitch = 0.62
    exponent_base = 0.42
    exponent_step = 0.06
    idx = np.arange(n_basis, dtype=float)
    angle = angle_step * idx
    centres = np.stack([radius * np.cos(angle),
                        radius * np.sin(angle),
                        pitch * idx], axis=1)
    alpha = exponent_base + exponent_step * idx
    norm = (2.0 * alpha / np.pi) ** 0.75
    return alpha, centres, norm

def _primitive_repulsion(n_basis):
    """The n_basis^4 chemists'-notation (ab|cd) integrals and the overlap matrix."""
    alpha, centres, norm = _helix_basis(n_basis)
    ovlp = np.zeros((n_basis, n_basis))
    tot = np.zeros((n_basis, n_basis))
    pref = np.zeros((n_basis, n_basis))
    prod = np.zeros((n_basis, n_basis, 3))
    for a in range(n_basis):
        for b in range(n_basis):
            pa, pb = alpha[a], alpha[b]
            summ = pa + pb
            dist2 = float(np.sum((centres[a] - centres[b]) ** 2))
            kab = math.exp(-pa * pb / summ * dist2)
            tot[a, b] = summ
            pref[a, b] = kab
            prod[a, b] = (pa * centres[a] + pb * centres[b]) / summ
            base = (np.pi / summ) ** 1.5 * kab
            ovlp[a, b] = norm[a] * norm[b] * base
    eri = np.zeros((n_basis, n_basis, n_basis, n_basis))
    for a in range(n_basis):
        for b in range(n_basis):
            for c in range(n_basis):
                for d in range(n_basis):
                    s1, s2 = tot[a, b], tot[c, d]
                    arg = s1 * s2 / (s1 + s2) * float(np.sum((prod[a, b] - prod[c, d]) ** 2))
                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \
                        * pref[a, b] * pref[c, d] * _boys_zero(arg)
                    eri[a, b, c, d] = norm[a] * norm[b] * norm[c] * norm[d] * val
    return ovlp, eri

def _symmetric_orthonormaliser(ovlp):
    """Symmetric orthonormaliser X = S^{-1/2}, formed through the spectral resolution."""
    val, vec = np.linalg.eigh(ovlp)
    return vec @ np.diag(val ** -0.5) @ vec.T

def _orthonormal_two_electron(n_basis, *, _cache={}):
    """The full physicists'-notation <pq|rs> array of the model.

    The keyword-only ``_cache`` memoises the array per basis size; it is pure
    memoisation of a deterministic function and changes no result.
    """
    cached = _cache.get(n_basis)
    if cached is not None:
        return cached
    ovlp, eri = _primitive_repulsion(n_basis)
    x_mat = _symmetric_orthonormaliser(ovlp)
    chem = np.einsum('ap,bq,cr,ds,abcd->pqrs', x_mat, x_mat, x_mat, x_mat, eri,
                     optimize=True)
    phys = np.transpose(chem, (0, 2, 1, 3))
    _cache[n_basis] = phys
    return phys

def _oracle_orthonormal_two_electron_element(n_basis: int, p: int, q: int, r: int, s: int) -> float:
    """Reference implementation: Boys-function ERIs, four-index transform, transposition."""
    n_basis = _check_index("n_basis", n_basis, 1, 16)
    p = _check_index("p", p, 0, n_basis - 1)
    q = _check_index("q", q, 0, n_basis - 1)
    r = _check_index("r", r, 0, n_basis - 1)
    s = _check_index("s", s, 0, n_basis - 1)
    return float(_orthonormal_two_electron(n_basis)[p, q, r, s])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(1, 0, 0, 0, 0)',
            'gold_call': '_oracle_orthonormal_two_electron_element(1, 0, 0, 0, 0)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(2, 0, 1, 1, 0)',
            'gold_call': '_oracle_orthonormal_two_electron_element(2, 0, 1, 1, 0)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(3, 0, 1, 2, 0)',
            'gold_call': '_oracle_orthonormal_two_electron_element(3, 0, 1, 2, 0)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(3, 0, 1, 2, 1)',
            'gold_call': '_oracle_orthonormal_two_electron_element(3, 0, 1, 2, 1)',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return sum(orthonormal_two_electron_element(4, i, j, i, j)\n               for i in range(4) for j in range(4))\ndef run_gold():\n    return sum(_oracle_orthonormal_two_electron_element(4, i, j, i, j)\n               for i in range(4) for j in range(4))\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(12, 11, 10, 11, 10)',
            'gold_call': '_oracle_orthonormal_two_electron_element(12, 11, 10, 11, 10)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(12, 11, 10, 10, 11)',
            'gold_call': '_oracle_orthonormal_two_electron_element(12, 11, 10, 10, 11)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_two_electron_element(12, 1, 2, 3, 6)',
            'gold_call': '_oracle_orthonormal_two_electron_element(12, 1, 2, 3, 6)',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_two_electron_element(17, 0, 0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_two_electron_element(17, 0, 0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_two_electron_element(3, 0, -1, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_two_electron_element(3, 0, -1, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_two_electron_element(3.0, 0, 0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_two_electron_element(3.0, 0, 0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
