"""
Run the whole audit over the three interface grid pairs of the configuration. For each pair assemble both blocks, both interface operators and the stabilising operator, then evaluate the inf-sup constant twice, once without the stabilising term and once with it, in the clamped setting of the configuration; and solve the loaded setting twice, once under the graded surface traction and once under the uniform one. Record one row per grid pair. The mesh size and the multiplier metric that enter the inf-sup characterisation are the ones the source's own derivation uses.

The source verifies its stabilisation in two ways: by checking that the inf-sup constant stays bounded away from zero as the interface is refined at a fixed grid ratio, and by checking that a constant traction still crosses the interface exactly. This audit performs both on the same configuration, so a single run exposes the stability behaviour and the accuracy behaviour together.

Returns
-------
A (3, 10) float64 array, one row per grid pair, with columns [inf-sup constant without stabilisation, inf-sup constant with stabilisation, Frobenius norm of the stabilising operator, mean normal multiplier under the graded load, smallest normal multiplier, largest normal multiplier, mean tangential multiplier magnitude, largest departure of the normal multiplier from its mean under the uniform load, number of non-mortar faces in the patch of the first internal mortar node, Frobenius norm of the local approximation formed on that same patch].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mortar_audit(depth):
    """Run the whole audit over the three interface grid pairs of the configuration. A (3, 10)
    float64 array, one row per grid pair, with columns [inf-sup constant without
    stabilisation, inf-sup constant with stabilisation, Frobenius norm of the stabilising
    operator, mean normal multiplier under the graded load, smallest normal multiplier,
    largest normal multiplier, mean tangential multiplier magnitude, largest departure of
    the normal multiplier from its mean under the uniform load, number of non-mortar faces
    in the patch of the first internal mortar node, Frobenius norm of the local
    approximation formed on that same patch]."""
    return np.zeros((3, 10))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

_CONFIGS = ((4, 2), (6, 3), (8, 4))

_PARS = dict(E=1.0, nu=0.0)

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def _top_load(n2, varying):
    f = np.zeros(3 * (n2 + 1) * (n2 + 1) * 2)
    h = 1.0 / n2
    a = h * h / 4.0
    for j in range(n2):
        for i in range(n2):
            for di, dj in _QUAD:
                p = _node_id(n2, 1, i + di, j + dj, 1)
                x, y = (i + di) * h, (j + dj) * h
                if varying:
                    f[3 * p + 0] += a * (0.30 * y)
                    f[3 * p + 1] += a * (0.20 * x)
                    f[3 * p + 2] += a * (-(1.0 + 0.5 * x + 0.25 * y))
                else:
                    f[3 * p + 2] += -a
    return f

def _oracle_mortar_audit(depth):
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be a positive finite block thickness")
    import scipy.linalg as _sla
    E, nu = _PARS["E"], _PARS["nu"]
    rows = []
    for (n1, n2) in _CONFIGS:
        Ke1 = _oracle_hex_stiffness(E, nu, 1.0 / n1, 1.0 / n1, depth)
        Ke2 = _oracle_hex_stiffness(E, nu, 1.0 / n2, 1.0 / n2, depth)
        A1 = _oracle_block_stiffness(n1, Ke1)
        A2 = _oracle_block_stiffness(n2, Ke2)
        D = _oracle_mortar_mass(n1)
        M = _oracle_mortar_coupling(n1, n2)
        H = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)
        Z = np.zeros_like(H)
        nt = n1 * n1
        Q = np.diag(np.repeat(1.0 / (n1 * n1), 3 * nt))
        h = 1.0 / n1
        # stability setting: both far faces clamped
        S1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))
        S2 = np.setdiff1d(np.arange(A2.shape[0]), _far_face(n2, 1))
        As = _sla.block_diag(A1[np.ix_(S1, S1)], A2[np.ix_(S2, S2)])
        Bs = np.hstack([D[:, S1], -M[:, S2]]).T
        b_un = _oracle_infsup_constant(As, Bs, Q, Z, h)
        b_st = _oracle_infsup_constant(As, Bs, Q, H, h)
        # load setting: bottom face clamped, top face loaded
        L1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))
        L2 = np.arange(A2.shape[0])
        Al = _sla.block_diag(A1[np.ix_(L1, L1)], A2[np.ix_(L2, L2)])
        Bl = np.hstack([D[:, L1], -M[:, L2]]).T
        fv = np.concatenate([np.zeros(len(L1)), _top_load(n2, True)[L2]])
        fc = np.concatenate([np.zeros(len(L1)), _top_load(n2, False)[L2]])
        tv = _oracle_interface_tractions(Al, Bl, H, fv)
        tc = _oracle_interface_tractions(Al, Bl, H, fc)
        tN = tv[2::3]
        tT = np.sqrt(tv[0::3] ** 2 + tv[1::3] ** 2)
        patch = float(np.max(np.abs(tc[2::3] - np.mean(tc[2::3]))))
        mask = _oracle_macroelement_masks(n1, n2, 1, 1)
        patch_faces = float(np.sum(mask[:n1 * n1]))
        scal = _oracle_local_scaling(n1, n2, 1, 1, A1, A2, D, M)
        rows.append([b_un, b_st, float(np.linalg.norm(H)), float(np.mean(tN)),
                     float(np.min(tN)), float(np.max(tN)), float(np.mean(tT)), patch,
                     patch_faces, float(np.linalg.norm(scal))])
    return np.array(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndepth = 1.0\n',
         'call': 'mortar_audit(depth)',
         'gold_call': '_oracle_mortar_audit(depth)'},
        {'setup': 'import numpy as np\ndepth = 0.5\n',
         'call': 'mortar_audit(depth)',
         'gold_call': '_oracle_mortar_audit(depth)'},
        {'setup': 'import numpy as np\ndepth = 1.5\n',
         'call': 'mortar_audit(depth)',
         'gold_call': '_oracle_mortar_audit(depth)'},
    ]
