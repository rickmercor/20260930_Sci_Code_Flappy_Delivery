"""
Apply one layer of the causal-cone circuit of the subsystem A = (-inf, 1] while keeping the
B-side qubit that leaves the cone in an edge register. Returns the joint state of the
boundary pair and the enlarged register one level lower.

Geometry of every layer: isometry j turns site j of the level above into sites (2j-1, 2j) of
the level below, W[a, b, c] with a on 2j-1, b on 2j and c the upper site; the disentanglers
of the same layer then act on the pairs (2j, 2j+1), U[a, b, c, d] with outputs a, b on (2j,
2j+1) and inputs c, d. The bipartition is A = (-inf, 1], B = [2, inf) at every level, so the
two boundary sites (0, 1) of the level above are produced by isometries 0 and 1 and feed
disentangler (0, 1) of the level below. R is the joint (unnormalised) operator of the
boundary sites (0, 1) of the upper level and an edge register of dimension d, with legs [k0,
k1, e, b0, b1, f]: ket sites, ket register index, bra sites, bra register index. Descend one
layer: apply isometries 0 and 1 to the ket legs and their conjugates to the bra legs, trace
site -1 (it belongs to A), and move site 2 into the register instead of tracing it, since
every later gate acting on it lies outside the causal cone of A and is a unitary on B alone.
The new register index is 2*e + s2, with s2 the value of site 2, so earlier (higher-layer)
edge qubits are the more significant digits. Then apply the disentangler on (0, 1) to kets
(U) and bras (conjugate of U). Return an array of shape (2, 2, 2d, 2, 2, 2d).

Returns
-------
np.ndarray: float array of shape (2, 2, 2d, 2, 2, 2d), legs [k0, k1, e, b0, b1, f].
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def edge_layer_map(R: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    """Apply one layer of the causal-cone circuit of the subsystem A = (-inf, 1] while keeping
    the B-side qubit that leaves the cone in an edge register. Returns the joint state of
    the boundary pair and the enlarged register one level lower.

    Args:
        R: float array of shape (2, 2, d, 2, 2, d).
        U: disentangler tensor U[a, b, c, d].
        W: isometry tensor W[a, b, c].

    Returns:
        np.ndarray: float array of shape (2, 2, 2d, 2, 2, 2d), legs [k0, k1, e, b0, b1, f].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _edge_layer_map(R, U, W):
    # ket W[i,j,A] W[k,l,B], bra W[i,n,C] W[o,p,D]; i: site -1 traced; l, p: new edge qubit
    d = R.shape[2]
    Wc, Uc = np.conj(W), np.conj(U)
    X = np.einsum("ABeCDf,ijA,klB,inC,opD,abjk,cdno->abelcdfp",
                  R, W, W, Wc, Wc, U, Uc, optimize=True)
    return X.reshape(2, 2, 2 * d, 2, 2, 2 * d)


def _oracle_edge_layer_map(R: "np.ndarray", U: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    return _edge_layer_map(R, U, W)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(4); R = rng.normal(size=(2, 2, 1, 2, 2, 1)); U = rng.normal(size=(2, 2, 2, 2)); W = rng.normal(size=(2, 2, 2))',
            'call': 'edge_layer_map(*copy.deepcopy((R, U, W)))',
            'gold_call': '_oracle_edge_layer_map(*copy.deepcopy((R, U, W)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(14); R = rng.normal(size=(2, 2, 4, 2, 2, 4)); U = rng.normal(size=(2, 2, 2, 2)); W = rng.normal(size=(2, 2, 2))',
            'call': 'edge_layer_map(*copy.deepcopy((R, U, W)))',
            'gold_call': '_oracle_edge_layer_map(*copy.deepcopy((R, U, W)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(21); R = np.zeros((2, 2, 1, 2, 2, 1)); R[0, 0, 0, 0, 0, 0] = 1.0; U = np.linalg.qr(rng.normal(size=(4, 4)))[0].reshape(2, 2, 2, 2); W = np.linalg.qr(rng.normal(size=(4, 4)))[0][:, :2].reshape(2, 2, 2)',
            'call': 'edge_layer_map(*copy.deepcopy((R, U, W)))',
            'gold_call': '_oracle_edge_layer_map(*copy.deepcopy((R, U, W)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(33); R = rng.normal(size=(2, 2, 2, 2, 2, 2)); U = rng.normal(size=(2, 2, 2, 2)); W = rng.normal(size=(2, 2, 2))',
            'call': 'edge_layer_map(*copy.deepcopy((R, U, W)))',
            'gold_call': '_oracle_edge_layer_map(*copy.deepcopy((R, U, W)))',
        },
    ]
