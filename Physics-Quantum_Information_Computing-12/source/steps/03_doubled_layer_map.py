"""
Apply one layer of the doubled, SWAP-contracted transition map that carries the purity of
the half chain B = [2, inf) down the causal cone of the boundary. Returns the two-copy
boundary operator one level lower.

Geometry of every layer: isometry j turns site j of the level above into sites (2j-1, 2j) of
the level below, W[a, b, c] with a on 2j-1, b on 2j and c the upper site; the disentanglers
of the same layer then act on the pairs (2j, 2j+1), U[a, b, c, d] with outputs a, b on (2j,
2j+1) and inputs c, d. The bipartition is A = (-inf, 1], B = [2, inf) at every level, so the
two boundary sites (0, 1) of the level above are produced by isometries 0 and 1 and feed
disentangler (0, 1) of the level below. Q is a two-copy operator on the boundary sites (0,
1) of the upper level with legs [k1_0, k1_1, b1_0, b1_1, k2_0, k2_1, b2_0, b2_1]: ket and
bra legs of copy 1, then of copy 2, where an operator O has components O[ket, bra]. Descend
one layer. In each copy apply isometries 0 and 1 to the ket legs and their complex
conjugates to the bra legs, giving sites -1, 0, 1, 2. Site -1 lies in A and is traced inside
each copy. Site 2 lies in B and leaves the boundary causal cone for good, so it is
contracted across the two copies exactly the way the indices of Tr(rho_B^2) are. Then apply
the disentangler on (0, 1) to every ket (U) and bra (conjugate of U). Return the new
operator with the same leg order; do not normalise it.

Returns
-------
np.ndarray: float array of shape (2,)*8, legs [k1_0, k1_1, b1_0, b1_1, k2_0, k2_1, b2_0, b2_1].
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def doubled_layer_map(Q: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    """Apply one layer of the doubled, SWAP-contracted transition map that carries the purity
    of the half chain B = [2, inf) down the causal cone of the boundary. Returns the two-
    copy boundary operator one level lower.

    Args:
        Q: float array of shape (2,)*8, the two-copy boundary operator of the upper level.
        U: disentangler tensor U[a, b, c, d].
        W: isometry tensor W[a, b, c].

    Returns:
        np.ndarray: float array of shape (2,)*8, legs [k1_0, k1_1, b1_0, b1_1, k2_0, k2_1, b2_0, b2_1].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _doubled_layer_map(Q, U, W):
    # copy-1 ket W[i,j,A] W[k,l,B]   copy-1 bra W[i,n,C] W[o,t,D]
    # copy-2 ket W[q,r,E] W[s,t,F]   copy-2 bra W[q,v,G] W[x,l,H]
    # i, q: site -1 traced in each copy; l, t: site 2 contracted across the copies
    Wc, Uc = np.conj(W), np.conj(U)
    return np.einsum(
        "ABCDEFGH,ijA,klB,inC,otD,qrE,stF,qvG,xlH,abjk,cdno,efrs,ghvx->abcdefgh",
        Q, W, W, Wc, Wc, W, W, Wc, Wc, U, Uc, U, Uc, optimize=True)


def _oracle_doubled_layer_map(Q: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    return _doubled_layer_map(Q, U, W)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(5); Q = rng.normal(size=[2] * 8); U = rng.normal(size=(2, 2, 2, 2)); W = rng.normal(size=(2, 2, 2))',
            'call': 'doubled_layer_map(*copy.deepcopy((Q, U, W)))',
            'gold_call': '_oracle_doubled_layer_map(*copy.deepcopy((Q, U, W)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(6); Q = np.zeros([2] * 8); Q[(0,) * 8] = 1.0; U = np.linalg.qr(rng.normal(size=(4, 4)))[0].reshape(2, 2, 2, 2); W = np.linalg.qr(rng.normal(size=(4, 4)))[0][:, :2].reshape(2, 2, 2)',
            'call': 'doubled_layer_map(*copy.deepcopy((Q, U, W)))',
            'gold_call': '_oracle_doubled_layer_map(*copy.deepcopy((Q, U, W)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(9); Q = rng.normal(size=[2] * 8); U = np.eye(4).reshape(2, 2, 2, 2); W = rng.normal(size=(2, 2, 2))',
            'call': 'doubled_layer_map(*copy.deepcopy((Q, U, W)))',
            'gold_call': '_oracle_doubled_layer_map(*copy.deepcopy((Q, U, W)))',
        },
        {
            'setup': "import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(12); Q = rng.normal(size=[2] * 8) + 3.0 * np.einsum('ab,cd,ef,gh->acbdegfh', np.eye(2), np.eye(2), rng.normal(size=(2, 2)), rng.normal(size=(2, 2))); U = rng.normal(size=(2, 2, 2, 2)); W = rng.normal(size=(2, 2, 2))",
            'call': 'doubled_layer_map(*copy.deepcopy((Q, U, W)))',
            'gold_call': '_oracle_doubled_layer_map(*copy.deepcopy((Q, U, W)))',
        },
    ]
