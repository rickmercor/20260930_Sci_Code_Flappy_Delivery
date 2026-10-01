"""
Recover the full sparse operator from its CSS-projected block: for each orbital, read the column of its color and undo the superposition exactly as the source's extraction prescribes, keep only entries the cutoff criterion admits, set all others to zero, and symmetrize the result.

Projection onto chromatic superposition states scrambles each colored group into one column; the extraction step must undo that scrambling and restore the sparsity pattern the cutoff defines.

Returns
-------
return (n, n) float64: extracted sparse operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def css_extract(Mtilde, R, colors, D, dcut):
    """Mtilde: (n, Nc) CSS-projected block; R: (n, Nc) the CSS matrix used
    for the projection; colors: (n,) color assignment; D: (n, n) distances;
    dcut: the cutoff. Returns (n, n): the extracted operator, with entries
    outside the cutoff criterion set to zero and the result symmetrized.
    Raises ValueError on inconsistent shapes."""
    return np.zeros((D.shape[0], D.shape[0]))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: extraction of the sparse operator from the CSS block."""

import numpy as np


def _oracle_css_extract(Mtilde, R, colors, D, dcut):
    Mtilde = np.asarray(Mtilde, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    colors = np.asarray(colors)
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n or Mtilde.shape[0] != n or R.shape != Mtilde.shape or colors.shape[0] != n:
        raise ValueError("inconsistent shapes")
    out = np.zeros((n, n))
    for nn in range(n):
        c = int(colors[nn])
        s = R[nn, c]
        for mm in range(n):
            if D[mm, nn] < dcut:
                out[mm, nn] = Mtilde[mm, c] * s
    return (out + out.T) / 2.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 5.0)\nR=css_matrix(colors)\nMt=S@R', "call": "css_extract(Mt, R, colors, D, 5.0)", "gold_call": "_oracle_css_extract(Mt, R, colors, D, 5.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 3.0)\nR=css_matrix(colors)\nMt=H@R', "call": "css_extract(Mt, R, colors, D, 3.0)", "gold_call": "_oracle_css_extract(Mt, R, colors, D, 3.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 4.0)\nR=css_matrix(colors)\nMt=(H+S)@R', "call": "css_extract(Mt, R, colors, D, 4.0)", "gold_call": "_oracle_css_extract(Mt, R, colors, D, 4.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 5.0)\nR=css_matrix(colors)\nMt=S@R', "call": "css_extract(Mt, R, colors, D, 1.0)", "gold_call": "_oracle_css_extract(Mt, R, colors, D, 1.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\ncolors=_n.zeros(20,dtype=_n.int64)\nR=css_matrix(colors)\nMt=S@R', "call": "css_extract(Mt, R, colors, D, 25.0)", "gold_call": "_oracle_css_extract(Mt, R, colors, D, 25.0)", "tol": 1e-09},
    ]
