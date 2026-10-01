"""
Assemble the conditioning audit of the wave scheme with degree p and regularity r at parameter rho, calling the earlier steps. Row 0: the first three growth thresholds of (p, r) in increasing order (zero when fewer exist) and the number of thresholds. Row 1: the numbers of zeros of det S_rho inside, on and outside the unit circle, and the asymptotic growth factor per block, the reciprocal of the largest inside modulus (zero when there is none). Row 2: log10 of the spectral condition number of the section with nb_final blocks, its largest singular value, log10 of its smallest singular value, and the ratio of the condition numbers of the sections with nb_final and nb_final - 1 blocks. Row 3: the largest threshold of the (2, 0) scheme, the largest threshold of the (3, 0) scheme, the smallest threshold of the (3, 1) scheme, all recomputed by the same chain as a check against the source, and the (1, 1) entry of the section with two blocks of the (p, r) scheme at rho. Raise an error if the (2, 0) chain does not return exactly three thresholds ending at 60, or if any entry is not finite. Reject invalid p, r, a negative rho or nb_final < 3.

The audit ties the chain together where it is most sensitive: the thresholds test the exact symbol, the type and the growth factor test the zero classification, the condition number of a large section tests the exact arithmetic, and the recomputed thresholds of the source's own cases pin the basis, the index shift and the block extraction against the published values.

Returns
-------
A (4, 4) float64 array laid out as described, with the reported log10 condition number in row 2, column 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def iga_conditioning_audit(p, r, rho, nb_final):
    """Assemble the conditioning audit of the wave scheme with degree p and regularity r at
    parameter rho, calling the earlier steps.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nb_final (int): number of blocks of the reported section, at least 3.

    Returns:
        numpy.ndarray of shape (4, 4), the audit described in the docstring, with log10
        of the condition number of the reported section in row 2, column 0.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or nb_final < 3.
        RuntimeError: if the source's (2, 0) thresholds are not reproduced or the audit
            is not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction

def _check_pr(p, r):
    p = int(p)
    r = int(r)
    if p < 1 or r < 0 or r > p - 1:
        raise ValueError('degree p must be >= 1 and regularity r must satisfy 0 <= r <= p - 1')
    return (p, r)

def _check_rho(rho):
    if isinstance(rho, Fraction):
        rq = rho
    else:
        rq = Fraction(rho).limit_denominator(10 ** 12) if isinstance(rho, float) else Fraction(rho)
    if rq < 0:
        raise ValueError('rho must be non-negative')
    return rq

def _check_nb(nblocks):
    nb = int(nblocks)
    if nb < 1:
        raise ValueError('nblocks must be a positive integer')
    return nb

def _oracle_iga_conditioning_audit(p, r, rho, nb_final):
    p, r = _check_pr(p, r)
    rq = _check_rho(rho)
    nb = _check_nb(nb_final)
    if nb < 3:
        raise ValueError('nb_final must be at least 3')
    nmesh = 2 * p + 4
    vals = _oracle_spline_basis_values(p, r, nmesh, np.array([0.3, 1.7, nmesh - 0.25]))
    if np.max(np.abs(vals.sum(axis=0) - 1.0)) > 1e-12:
        raise RuntimeError('basis does not form a partition of unity')
    G = _oracle_petrov_galerkin_matrices(p, r, nmesh)
    blocks = _oracle_interior_symbol_blocks(p, r)
    N = p - r
    mid = nmesh // 2
    for d in range(-p, p + 1):
        bj = mid + d
        if 0 <= bj < (nmesh * N + r) // N:
            if np.max(np.abs(G[0, N * mid:N * mid + N, N * bj:N * bj + N] - blocks[d + p, 0])) > 1e-12 or np.max(np.abs(G[1, N * mid:N * mid + N, N * bj:N * bj + N] - blocks[d + p, 1])) > 1e-12:
                raise RuntimeError('interior blocks do not match the assembled matrices')
    th = _oracle_growth_thresholds(p, r)
    typ = _oracle_zero_location_type(p, r, rq)
    kap = _oracle_exact_condition_number(p, r, rq, nb)
    gf = _oracle_growth_factors(p, r, rq, [nb - 1])
    sec = _oracle_block_toeplitz_section(p, r, rq, 2)
    detc = _oracle_symbol_determinant(p, r, rq)
    t20 = _oracle_growth_thresholds(2, 0)
    t30 = _oracle_growth_thresholds(3, 0)
    t31 = _oracle_growth_thresholds(3, 1)
    if len(t20) != 3 or abs(t20[-1] - 60.0) > 1e-09:
        raise RuntimeError('the (2, 0) thresholds of the source are not reproduced')
    row0 = np.zeros(4)
    row0[:min(3, len(th))] = th[:3]
    row0[3] = float(len(th))
    row1 = np.array([typ[0], typ[1], typ[2], 1.0 / typ[3] if typ[3] > 0 else 0.0])
    row2 = np.array([kap[0], kap[1], kap[2], gf[0]])
    row3 = np.array([t20[-1], t30[-1], t31[0], float(sec[0, 0])])
    audit = np.vstack([row0, row1, row2, row3])
    nz = np.nonzero(detc)[0]
    core = detc[nz[0]:nz[-1] + 1]
    if not np.all(np.isfinite(audit)) or abs(core[0] - core[-1]) > 1e-09 * abs(core[0]):
        raise RuntimeError('audit not finite or symbol determinant not self-reciprocal')
    return audit

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\np, r, rho, nb_final = 4, 2, 60.0, 60\n',
         'call': 'iga_conditioning_audit(p, r, rho, nb_final)',
         'gold_call': '_oracle_iga_conditioning_audit(p, r, rho, nb_final)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nb_final = 2, 0, 70.0, 30\n',
         'call': 'iga_conditioning_audit(p, r, rho, nb_final)',
         'gold_call': '_oracle_iga_conditioning_audit(p, r, rho, nb_final)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nb_final = 3, 1, 50.0, 25\n',
         'call': 'iga_conditioning_audit(p, r, rho, nb_final)',
         'gold_call': '_oracle_iga_conditioning_audit(p, r, rho, nb_final)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\np, r, rho, nb_final = 3, 0, 50.0, 20\n',
         'call': 'iga_conditioning_audit(p, r, rho, nb_final)',
         'gold_call': '_oracle_iga_conditioning_audit(p, r, rho, nb_final)',
         'tol': 1e-08},
    ]
