"""
Schmidt gap lambda_0 - lambda_1 of the half chain B = [2, inf) for the infinite binary MERA
built from the given per-layer angles. Returns the gap.

Assemble the chain from the earlier steps, bottom layer first in both angle lists. For each
layer build the two block matrices (block_gate) and the tensors (layer_tensors). Every
renormalised site above the top layer is |0>, so the top two-copy operator has a single unit
entry Q[0, 0, 0, 0, 0, 0, 0, 0] = 1 and the top joint operator R has shape (2, 2, 1, 2, 2,
1) with R[0, 0, 0, 0, 0, 0] = 1. Apply the layers from the top layer down, with
doubled_layer_map on Q and edge_layer_map on R. Close Q with renyi2_from_doubled, reduce R
with edge_reduced_state and diagonalise it with entanglement_spectrum. The two routes must
agree, -log2(sum of lambda_i^2) = S2; raise ValueError if they differ by more than 1e-8.
Return lambda_0 - lambda_1 as a Python float.

Returns
-------
float: Schmidt gap lambda_0 - lambda_1.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def schmidt_gap(u_angles: list, w_angles: list) -> float:
    """Schmidt gap lambda_0 - lambda_1 of the half chain B = [2, inf) for the infinite binary
    MERA built from the given per-layer angles. Returns the gap.

    Args:
        u_angles: list of per-layer disentangler angle lists, bottom layer first.
        w_angles: list of per-layer isometry angle lists, bottom layer first.
        Uses: block_gate, layer_tensors, doubled_layer_map, renyi2_from_doubled,
        edge_layer_map, edge_reduced_state, entanglement_spectrum

    Returns:
        float: Schmidt gap lambda_0 - lambda_1.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _block_gate(angles):
    a = [float(v) for v in angles]
    x1, x2 = a[-2], a[-1]
    cp, sp = np.cos(np.pi * (x1 + x2) / 2.0), np.sin(np.pi * (x1 + x2) / 2.0)
    cm, sm = np.cos(np.pi * (x2 - x1) / 2.0), np.sin(np.pi * (x2 - x1) / 2.0)
    g = np.array([[cp, 0.0, 0.0, -sp],
                  [0.0, cm, sm, 0.0],
                  [0.0, -sm, cm, 0.0],
                  [sp, 0.0, 0.0, cp]])
    if len(a) == 4:
        c1, s1 = np.cos(np.pi * a[0] / 2.0), np.sin(np.pi * a[0] / 2.0)
        c2, s2 = np.cos(np.pi * a[1] / 2.0), np.sin(np.pi * a[1] / 2.0)
        g = np.kron(np.array([[c1, -s1], [s1, c1]]), np.array([[c2, -s2], [s2, c2]])) @ g
    return g

def _layer_tensors(gate_u, gate_w):
    gu = np.asarray(gate_u, dtype=float)
    gw = np.asarray(gate_w, dtype=float)
    return gu.reshape(2, 2, 2, 2), gw[:, [0, 2]].reshape(2, 2, 2)

def _doubled_layer_map(Q, U, W):
    # copy-1 ket W[i,j,A] W[k,l,B]   copy-1 bra W[i,n,C] W[o,t,D]
    # copy-2 ket W[q,r,E] W[s,t,F]   copy-2 bra W[q,v,G] W[x,l,H]
    # i, q: site -1 traced in each copy; l, t: site 2 contracted across the copies
    Wc, Uc = np.conj(W), np.conj(U)
    return np.einsum(
        "ABCDEFGH,ijA,klB,inC,otD,qrE,stF,qvG,xlH,abjk,cdno,efrs,ghvx->abcdefgh",
        Q, W, W, Wc, Wc, W, W, Wc, Wc, U, Uc, U, Uc, optimize=True)

def _renyi2_from_doubled(Q):
    return float(-np.log2(np.einsum("ababcdcd->", np.asarray(Q, dtype=float))))

def _edge_layer_map(R, U, W):
    # ket W[i,j,A] W[k,l,B], bra W[i,n,C] W[o,p,D]; i: site -1 traced; l, p: new edge qubit
    d = R.shape[2]
    Wc, Uc = np.conj(W), np.conj(U)
    X = np.einsum("ABeCDf,ijA,klB,inC,opD,abjk,cdno->abelcdfp",
                  R, W, W, Wc, Wc, U, Uc, optimize=True)
    return X.reshape(2, 2, 2 * d, 2, 2, 2 * d)

def _edge_reduced_state(R):
    return np.einsum("abeabf->ef", np.asarray(R, dtype=float))

def _entanglement_spectrum(rho):
    m = np.asarray(rho, dtype=float)
    return np.linalg.eigvalsh(0.5 * (m + m.T))[::-1].copy()

def _bind():
    fallbacks = (("_oracle_block_gate", _block_gate),
                 ("_oracle_layer_tensors", _layer_tensors),
                 ("_oracle_doubled_layer_map", _doubled_layer_map),
                 ("_oracle_renyi2_from_doubled", _renyi2_from_doubled),
                 ("_oracle_edge_layer_map", _edge_layer_map),
                 ("_oracle_edge_reduced_state", _edge_reduced_state),
                 ("_oracle_entanglement_spectrum", _entanglement_spectrum))
    g = globals()
    for nm, fb in fallbacks:
        g.setdefault(nm, fb)
    return dict(fallbacks)


def _oracle_schmidt_gap(u_angles: list, w_angles: list) -> float:
    _bind()
    layers = [_oracle_layer_tensors(_oracle_block_gate(ua), _oracle_block_gate(wa))
              for ua, wa in zip(u_angles, w_angles)]
    Q = np.zeros([2] * 8)
    Q[(0,) * 8] = 1.0
    R = np.zeros((2, 2, 1, 2, 2, 1))
    R[0, 0, 0, 0, 0, 0] = 1.0
    for U, W in reversed(layers):
        Q = _oracle_doubled_layer_map(Q, U, W)
        R = _oracle_edge_layer_map(R, U, W)
    s2 = _oracle_renyi2_from_doubled(Q)
    lam = _oracle_entanglement_spectrum(_oracle_edge_reduced_state(R))
    if abs(s2 + np.log2(np.sum(lam ** 2))) > 1e-8:
        raise ValueError("purity of the doubled map disagrees with the edge spectrum")
    return float(lam[0] - lam[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; u_angles = [[0.16148386263647466, -0.056145691508003415], [0.13118304250021381, -0.0383352598604525], [0.08035122878854976, -0.021415347769675463], [0.010069626069892758, 0.01006953360769033]]; w_angles = [[-0.02633623257988793, 0.13609039706878254], [-0.019513199676447934, 0.1145574729714855], [-0.0096600845365713, 0.06892662021737829], [-0.061861621487427465, 0.08198065488845328]]',
            'call': 'schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
            'gold_call': '_oracle_schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; u_angles = [[0.14128916837190672, -0.04855295372072077], [0.09485246552828838, -0.026780826052108297], [0.014283396219806037, 0.014283396218955706]]; w_angles = [[-0.022645095961054018, 0.11788128500836616], [-0.011366270004517398, 0.07993999252030186], [0.014254702701648131, 0.014254702701648131]]',
            'call': 'schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
            'gold_call': '_oracle_schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; u_angles = [[-0.08645507352591686, -0.08645536601575388, 0.06035718626700043, 0.06035714232872677], [0.4787965134747363, 0.4787965137485082, 0.00035538260020011894, 0.0003553837941574635]]; w_angles = [[-0.05538461872318612, 0.44461502277088877], [0.0003553828933530067, 0.0003553828933530067]]',
            'call': 'schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
            'gold_call': '_oracle_schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; u_angles = [[0.0488592928520786, 0.04885929285205401]]; w_angles = [[0.02945730049443655, 0.06589993124750528]]',
            'call': 'schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
            'gold_call': '_oracle_schmidt_gap(*copy.deepcopy((u_angles, w_angles)))',
        },
    ]
