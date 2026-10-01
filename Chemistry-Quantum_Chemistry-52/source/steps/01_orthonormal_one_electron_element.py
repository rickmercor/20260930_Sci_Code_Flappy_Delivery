"""
One-electron integral of the twelve-Gaussian model in its Loewdin-orthonormal basis.

This step generates the primitive spherical Gaussian basis of the model system from the

rule that defines it, builds the overlap, kinetic-energy and nuclear-attraction matrices

over those primitives, symmetrically orthonormalises the basis, and returns the single

element ``h[p, q]`` of the resulting one-electron Hamiltonian.



Scientific background.  The model is a helix of `$n_basis$` normalised spherical (s-type)

Gaussian functions.  Function `$p$`, for p = 0, 1, ..., n_basis - 1, carries exponent

alpha_p = 0.42 + 0.06 p and sits at the centre

(1.35 cos(2.4 p), 1.35 sin(2.4 p), 0.62 p) in bohr, the angle being in radians, and is

normalised so that its self-overlap is one, which for an s-type primitive means the

prefactor (2 alpha / pi)^(3/4).  Everything is in atomic units.



For two s-type primitives with exponents a and b centred at A and B write

p = a + b, P = (a A + b B) / p and K = exp(-a b |A - B|^2 / p).  The three primitive

integrals needed here are the standard Gaussian product-theorem results



S_ab = N_a N_b (pi / p)^{3/2} K,

    T_ab = N_a N_b (a b / p) (3 - 2 (a b / p) |A - B|^2) (pi / p)^{3/2} K,

    V_ab = -N_a N_b sum_C Z_C (2 pi / p) K F_0(p |P - R_C|^2),



with F_0(x) = (1/2) sqrt(pi / x) erf(sqrt(x)) the zeroth Boys function, understood as its

limit F_0(0) = 1 at the origin.  The nuclear-attraction term places one point charge of

magnitude Z_C = 0.55 at *every* one of the n_basis Gaussian centres, so the sum over C runs

over all of them, including the two centres the pair of functions itself sits on.



The primitives are not orthogonal, so the one-electron operator must be carried into an

orthonormal basis before any second-quantised expression built on it means anything.  The

symmetric (Loewdin) choice X = S^{-1/2} is used, with the input ordering of the functions

kept: X is formed by diagonalising S, raising its eigenvalues to the power -1/2 and

transforming back, so X is symmetric and is the orthonormaliser closest to the identity in

the least-squares sense.  The orthonormal one-electron Hamiltonian is then the congruence

transform h = X^T (T + V) X, and the twelve resulting functions of the frozen model are

read as twelve spin orbitals of one general two-body Hamiltonian, with no spatial-orbital

pairing and no spin blocking.  Note that the transform acts on the *left* index through

X^T, which is inconsequential here because X is symmetric but fixes the convention that

the two-electron transform of the companion step must follow.



Raises ValueError when n_basis is not an integer in the range 1 to 24 inclusive, and when

p or q is not an integer in the range 0 to n_basis - 1 inclusive.

Returns
-------
float, the element h[p, q] of the one-electron Hamiltonian in the Loewdin-orthonormal basis of the n_basis-function model, in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orthonormal_one_electron_element(n_basis: int, p: int, q: int) -> float:
    """One element of the Loewdin-orthonormal one-electron Hamiltonian of the model.

    Parameters
    ----------
    n_basis : int
        Number of spherical Gaussian primitives in the helix, 1 <= n_basis <= 24.  The
        frozen model system uses n_basis = 12.
    p : int
        Row index of the requested element, 0 <= p < n_basis.
    q : int
        Column index of the requested element, 0 <= q < n_basis.

    Returns
    -------
    float
        h[p, q] = (X^T (T + V) X)[p, q] in hartree, with X = S^{-1/2} the symmetric
        orthonormaliser of the primitive overlap matrix, T the kinetic-energy matrix and
        V the attraction to point charges of magnitude 0.55 at all n_basis centres.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

from scipy.special import erf

def _check_int(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _boys0(x):
    """Zeroth Boys function F_0(x), with the removable singularity at x = 0 handled."""
    if x <= 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))

def _basis(n_basis):
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

def _primitive_one_electron(n_basis):
    """Overlap, kinetic and nuclear-attraction matrices over the raw primitives."""
    point_charge = 0.55
    alpha, centres, norm = _basis(n_basis)
    ovlp = np.zeros((n_basis, n_basis))
    kin = np.zeros((n_basis, n_basis))
    nuc = np.zeros((n_basis, n_basis))
    for a in range(n_basis):
        for b in range(n_basis):
            pa, pb = alpha[a], alpha[b]
            tot = pa + pb
            dist2 = float(np.sum((centres[a] - centres[b]) ** 2))
            pref = math.exp(-pa * pb / tot * dist2)
            prod = (pa * centres[a] + pb * centres[b]) / tot
            base = (np.pi / tot) ** 1.5 * pref
            ovlp[a, b] = norm[a] * norm[b] * base
            kin[a, b] = norm[a] * norm[b] * (
                pa * pb / tot * (3.0 - 2.0 * pa * pb / tot * dist2)) * base
            acc = 0.0
            for centre in centres:
                arg = tot * float(np.sum((prod - centre) ** 2))
                acc += -point_charge * 2 * np.pi / tot * pref * _boys0(arg)
            nuc[a, b] = norm[a] * norm[b] * acc
    return ovlp, kin, nuc

def _loewdin_transform(ovlp):
    """Symmetric orthonormaliser X = S^{-1/2}, formed through the spectral resolution."""
    val, vec = np.linalg.eigh(ovlp)
    return vec @ np.diag(val ** -0.5) @ vec.T

def _orthonormal_one_electron(n_basis, *, _cache={}):
    """The full Loewdin-orthonormal one-electron Hamiltonian of the model.

    The keyword-only ``_cache`` memoises the matrix per basis size; it is pure
    memoisation of a deterministic function and changes no result.
    """
    cached = _cache.get(n_basis)
    if cached is not None:
        return cached
    ovlp, kin, nuc = _primitive_one_electron(n_basis)
    x_mat = _loewdin_transform(ovlp)
    h_mat = x_mat.T @ (kin + nuc) @ x_mat
    _cache[n_basis] = h_mat
    return h_mat

def _oracle_orthonormal_one_electron_element(n_basis: int, p: int, q: int) -> float:
    """Reference implementation: primitive integrals, S^{-1/2}, congruence transform."""
    n_basis = _check_int("n_basis", n_basis, 1, 24)
    p = _check_int("p", p, 0, n_basis - 1)
    q = _check_int("q", q, 0, n_basis - 1)
    return float(_orthonormal_one_electron(n_basis)[p, q])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_one_electron_element(1, 0, 0)',
            'gold_call': '_oracle_orthonormal_one_electron_element(1, 0, 0)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_one_electron_element(2, 0, 1)',
            'gold_call': '_oracle_orthonormal_one_electron_element(2, 0, 1)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_one_electron_element(3, 0, 2)',
            'gold_call': '_oracle_orthonormal_one_electron_element(3, 0, 2)',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return [orthonormal_one_electron_element(4, i, j) for i in range(4) for j in range(4)]\ndef run_gold():\n    return [_oracle_orthonormal_one_electron_element(4, i, j) for i in range(4) for j in range(4)]\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return orthonormal_one_electron_element(12, 4, 4)\ndef run_gold():\n    value = _oracle_orthonormal_one_electron_element(12, 4, 4)\n    assert abs(value - (-1.675877147015951)) <= 1e-10 * max(1.0, abs(-1.675877147015951)),         "the frozen anchor -1.675877147015951 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_one_electron_element(12, 11, 11)',
            'gold_call': '_oracle_orthonormal_one_electron_element(12, 11, 11)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'orthonormal_one_electron_element(12, 0, 11)',
            'gold_call': '_oracle_orthonormal_one_electron_element(12, 0, 11)',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_one_electron_element(0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_one_electron_element(0, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_one_electron_element(4, 4, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_one_electron_element(4, 4, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    try:    orthonormal_one_electron_element(4, 1.5, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_orthonormal_one_electron_element(4, 1.5, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
