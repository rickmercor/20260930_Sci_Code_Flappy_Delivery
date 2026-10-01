"""
Project the factorized electronic Hamiltonian into the paired determinant basis.

Pair excitations preserve seniority zero. Single-electron hopping still contributes to the full walker dynamics, although it has no off-diagonal matrix element between paired basis determinants.

Returns
-------
return matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def doci_matrix(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray") -> "np.ndarray":
    """Project the factorized electronic Hamiltonian into the paired determinant basis.
    
    All inputs are real and finite and must not be modified. NumPy and SciPy are available.
    
    The spatial basis is orthonormal. For occupation tuple I, |D_I> has
    alpha creation operators in increasing spatial order followed by beta
    creation operators in the same order. The Hamiltonian is
    H = sum_{pq,s} h[p,q] a^dagger[p,s] a[q,s]
        + (1/2) sum_{l,pqrs,sigma,tau} L[l,p,q] L[l,r,s]
          a^dagger[p,sigma] a^dagger[r,tau] a[s,tau] a[q,sigma].
    No nuclear constant is included. Project this full Hamiltonian into
    the listed seniority-zero determinant subspace.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron integrals in Eh.
    factors : real ndarray, shape (r,n,n)
        Symmetric factors in sqrt(Eh); the two-electron tensor is their unweighted outer-product sum.
    occupations : integer ndarray, shape (nd,k)
        Distinct, strictly increasing orbital tuples from pair_occupations, or a nonempty subset in any row order.
    Returns
    -------
    matrix : real ndarray, shape (nd,nd)
        DOCI Hamiltonian in Eh, matrix[I,J] = <D_I|H|D_J>, with the
        same row and column order as occupations.
    Raises
    ------
    ValueError
        If h/factors have incompatible shapes or are not symmetric, or an occupation row is out of range or not strictly increasing.
    """
    return matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_doci_matrix(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray") -> "np.ndarray":
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    occupations = np.asarray(occupations, dtype=int)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T):
        raise ValueError('h must be real symmetric.')
    n = len(h)
    if factors.ndim != 3 or factors.shape[1:] != (n,n) or not np.allclose(factors, factors.transpose(0,2,1)):
        raise ValueError('factors must be real symmetric matrices.')
    if occupations.ndim != 2 or np.any(occupations < 0) or np.any(occupations >= n) or np.any(np.diff(occupations,axis=1) <= 0):
        raise ValueError('Invalid occupations.')
    d = len(occupations)
    result = np.zeros((d,d))
    for a, occ in enumerate(occupations):
        result[a,a] = 2 * np.trace(h[np.ix_(occ,occ)])
        for l in factors:
            block = l[np.ix_(occ,occ)]
            result[a,a] += 2*np.trace(block)**2 - np.trace(block@block)
        for b in range(a):
            other = occupations[b]
            removed = sorted(set(other)-set(occ))
            added = sorted(set(occ)-set(other))
            if len(removed) == len(added) == 1:
                value = np.sum(factors[:,added[0],removed[0]]**2)
                result[a,b] = result[b,a] = value
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge comparisons against the oracle."""
    checks = "\ndef _checked(fn, *args):\n    copied = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]\n    before = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in copied]\n    try:\n        result = fn(*copied)\n    finally:\n        for original, current in zip(before, copied):\n            if isinstance(original, np.ndarray) and not np.array_equal(original, current):\n                raise AssertionError('Input arrays must not be modified')\n    return result\n"
    setup_0 = 'import numpy as np\n\ndef pack(z):\n    z=np.asarray(z)\n    return np.stack((z.real,z.imag),axis=-1)\n\nH = np.array([[-1.084145,.12,-.08,.05],[.12,-.885553,.11,-.06],[-.08,.11,-.763931,.09],[.05,-.06,.09,-.742439]])\nL = np.array([\n [[.58,.07,-.03,.04],[.07,.43,.08,-.02],[-.03,.08,.36,.06],[.04,-.02,.06,.29]],\n [[.16,-.11,.04,.03],[-.11,-.22,.05,.07],[.04,.05,.19,-.08],[.03,.07,-.08,-.14]],\n [[-.12,.03,.09,-.05],[.03,.17,-.06,.02],[.09,-.06,.08,.11],[-.05,.02,.11,-.10]]])\nfor factor in L:\n    d = factor.diagonal().copy()\n    factor *= 2.8\n    np.fill_diagonal(factor,d)\nPHI_A=np.array([[1.,.07],[-.04,.92],[.18,-.14],[-.12,.21]])\nPHI_B=np.array([[.95,-.03],[.06,1.02],[-.13,.17],[.16,.09]])\nWALKERS=[]\nfor w in range(4):\n    spins=[]\n    for s,base in enumerate((PHI_A,PHI_B)):\n        p=np.empty((4,2),complex)\n        for pidx in range(4):\n            for j in range(2):\n                p[pidx,j]=base[pidx,j]+.035*w*np.cos((pidx+1)*(j+2)) + 1j*.02*(w+s)*np.sin((pidx+2)*(j+1))\n        spins.append(pack(p))\n    WALKERS.append(spins)\nWALKERS=np.array(WALKERS)\nWEIGHTS=np.array([1.,.8,1.2,.9])\nFIELDS=np.empty((7,4,3))\nfor t in range(7):\n    for w in range(4):\n        for l in range(3):\n            FIELDS[t,w,l]=.85*np.sin((t+1)*(w+2)*(l+1))+.35*np.cos((t+2)*(l+2)+w)\nDT=.08\nENERGY_SHIFT=-2.1\nNPAIR=2\nARGS=(H,L,NPAIR,WALKERS,WEIGHTS,FIELDS,DT,ENERGY_SHIFT)\n\nOCC=np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],dtype=int)\nC=pack(np.array([0.6641087358712602, 0.044491721035112175, -0.3682035944352817, -0.31448486148478577, 0.055587573647215255, 0.5651685414959503]))\nFORCE=np.array([[-0.0011784757701281626, 2.015555438600003], [0.0007484660377895938, -0.09440649469891317], [0.002555156579562141, 0.11223560577850836]])\nM_DOCI=np.array([[-2.8062679999999998, 0.09799999999999999, 0.044688000000000005, 0.083104, 0.03919999999999999, 0.0], [0.09799999999999999, -2.39546, 0.17326399999999997, 0.14033600000000002, 0.0, 0.03919999999999999], [0.044688000000000005, 0.17326399999999997, -2.6102680000000005, 0.0, 0.14033600000000002, 0.083104], [0.083104, 0.14033600000000002, 0.0, -2.5542680000000004, 0.17326399999999997, 0.044688000000000005], [0.03919999999999999, 0.0, 0.14033600000000002, 0.17326399999999997, -2.41546, 0.09799999999999999], [0.0, 0.03919999999999999, 0.083104, 0.044688000000000005, 0.09799999999999999, -2.794268]])\n\n'
    setup_1 = "import numpy as np\n\ndef pack(z):\n    z=np.asarray(z)\n    return np.stack((z.real,z.imag),axis=-1)\n\nH = np.array([[-1.084145,.12,-.08,.05],[.12,-.885553,.11,-.06],[-.08,.11,-.763931,.09],[.05,-.06,.09,-.742439]])\nL = np.array([\n [[.58,.07,-.03,.04],[.07,.43,.08,-.02],[-.03,.08,.36,.06],[.04,-.02,.06,.29]],\n [[.16,-.11,.04,.03],[-.11,-.22,.05,.07],[.04,.05,.19,-.08],[.03,.07,-.08,-.14]],\n [[-.12,.03,.09,-.05],[.03,.17,-.06,.02],[.09,-.06,.08,.11],[-.05,.02,.11,-.10]]])\nfor factor in L:\n    d = factor.diagonal().copy()\n    factor *= 2.8\n    np.fill_diagonal(factor,d)\nPHI_A=np.array([[1.,.07],[-.04,.92],[.18,-.14],[-.12,.21]])\nPHI_B=np.array([[.95,-.03],[.06,1.02],[-.13,.17],[.16,.09]])\nWALKERS=[]\nfor w in range(4):\n    spins=[]\n    for s,base in enumerate((PHI_A,PHI_B)):\n        p=np.empty((4,2),complex)\n        for pidx in range(4):\n            for j in range(2):\n                p[pidx,j]=base[pidx,j]+.035*w*np.cos((pidx+1)*(j+2)) + 1j*.02*(w+s)*np.sin((pidx+2)*(j+1))\n        spins.append(pack(p))\n    WALKERS.append(spins)\nWALKERS=np.array(WALKERS)\nWEIGHTS=np.array([1.,.8,1.2,.9])\nFIELDS=np.empty((7,4,3))\nfor t in range(7):\n    for w in range(4):\n        for l in range(3):\n            FIELDS[t,w,l]=.85*np.sin((t+1)*(w+2)*(l+1))+.35*np.cos((t+2)*(l+2)+w)\nDT=.08\nENERGY_SHIFT=-2.1\nNPAIR=2\nARGS=(H,L,NPAIR,WALKERS,WEIGHTS,FIELDS,DT,ENERGY_SHIFT)\n\nOCC=np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],dtype=int)\nC=pack(np.array([0.6641087358712602, 0.044491721035112175, -0.3682035944352817, -0.31448486148478577, 0.055587573647215255, 0.5651685414959503]))\nFORCE=np.array([[-0.0011784757701281626, 2.015555438600003], [0.0007484660377895938, -0.09440649469891317], [0.002555156579562141, 0.11223560577850836]])\nM_DOCI=np.array([[-2.8062679999999998, 0.09799999999999999, 0.044688000000000005, 0.083104, 0.03919999999999999, 0.0], [0.09799999999999999, -2.39546, 0.17326399999999997, 0.14033600000000002, 0.0, 0.03919999999999999], [0.044688000000000005, 0.17326399999999997, -2.6102680000000005, 0.0, 0.14033600000000002, 0.083104], [0.083104, 0.14033600000000002, 0.0, -2.5542680000000004, 0.17326399999999997, 0.044688000000000005], [0.03919999999999999, 0.0, 0.14033600000000002, 0.17326399999999997, -2.41546, 0.09799999999999999], [0.0, 0.03919999999999999, 0.083104, 0.044688000000000005, 0.09799999999999999, -2.794268]])\n\n\ndef _raises(fn,*args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError('Expected ValueError')\n"
    return [
        {
            "setup": setup_0 + checks,
            "call": '_checked(doci_matrix, H, L, OCC)',
            "gold_call": '_checked(_oracle_doci_matrix, H, L, OCC)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(doci_matrix, H, np.zeros_like(L), OCC)',
            "gold_call": '_checked(_oracle_doci_matrix, H, np.zeros_like(L), OCC)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(doci_matrix, H[:3, :3], L[:, :3, :3], np.array([[2], [0], [1]]))',
            "gold_call": '_checked(_oracle_doci_matrix, H[:3, :3], L[:, :3, :3], np.array([[2], [0], [1]]))',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(doci_matrix, H, L, OCC[[4, 0, 2]])',
            "gold_call": '_checked(_oracle_doci_matrix, H, L, OCC[[4, 0, 2]])',
            "tol": 2e-08,
        },
        {
            "setup": setup_1 + checks,
            "call": '_raises(_checked, doci_matrix, H, L, np.array([[0, 0]]))',
            "gold_call": '_raises(_checked, _oracle_doci_matrix, H, L, np.array([[0, 0]]))',
            "tol": 2e-08,
        },
    ]
