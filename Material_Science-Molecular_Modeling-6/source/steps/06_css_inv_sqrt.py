"""
Compute the CSS approximation of the overlap inverse square root: color the graph with the overlap cutoff, build the probing states, form the block-Krylov basis of the overlap from them, apply the projected inverse square root, assemble the projected block exactly as the source specifies, and extract the sparse result.

The source computes the inverse square root of the overlap on its own probing set with its own cutoff before any density quantity is touched.

Returns
-------
return (n, n) float64: CSS-reconstructed overlap inverse square root
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def css_inv_sqrt(S, D, dcut_s, v_s):
    """S: (n, n) SPD overlap; D: (n, n) distances; dcut_s: overlap cutoff;
    v_s: Krylov dimension for the overlap branch. Returns (n, n): the
    CSS-reconstructed inverse square root of S, assembled from the coloring,
    the probing states, the block-Krylov basis, the projected inverse square
    root and the extraction, exactly in the source's arrangement."""
    return np.eye(S.shape[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: CSS reconstruction of the overlap inverse square root."""

import numpy as np


def _oracle_css_inv_sqrt(S, D, dcut_s, v_s):
    colors_s = _oracle_greedy_multicoloring(D, dcut_s)
    O = _oracle_css_matrix(colors_s)
    Q = _oracle_block_krylov_basis(S, O, v_s)
    SKinv = _oracle_projected_inv_sqrt(S, Q)
    Y = Q @ (SKinv @ (Q.T @ O))
    return _oracle_css_extract(Y, O, colors_s, D, dcut_s)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_inv_sqrt(S, D, 5.0, 3)", "gold_call": "_oracle_css_inv_sqrt(S, D, 5.0, 3)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2', "call": "css_inv_sqrt(S, D, 5.0, 4)", "gold_call": "_oracle_css_inv_sqrt(S, D, 5.0, 4)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2', "call": "css_inv_sqrt(S, D, 4.0, 3)", "gold_call": "_oracle_css_inv_sqrt(S, D, 4.0, 3)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_inv_sqrt(S, D, 5.0, 1)", "gold_call": "_oracle_css_inv_sqrt(S, D, 5.0, 1)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_inv_sqrt(S, D, 1.0, 3)", "gold_call": "_oracle_css_inv_sqrt(S, D, 1.0, 3)", "tol": 1e-08},
    ]
