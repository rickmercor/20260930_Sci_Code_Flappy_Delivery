"""
Assemble the chromatic superposition states as columns of an (n, Nc) matrix: column c carries the declared deterministic sign s_j on every orbital j of color c and zeros elsewhere, with s_j = +1 when ((3(j+1)) mod 7) >= 3 and -1 otherwise.

The source populates each colored state with plus-minus-one entries; this task pins the deterministic sign rule declared in the problem statement in place of the source's random signs.

Returns
-------
return (n, Nc) float64: CSS probing matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def css_matrix(colors):
    """colors: (n,) integer color assignment with colors in 0..Nc-1.
    Returns (n, Nc) float64: column c holds the declared deterministic sign
    s_j on each orbital j with colors[j] == c and zeros elsewhere, where
    s_j = +1 if ((3 * (j + 1)) % 7) >= 3 else -1, and columns are ordered by
    color index. Raises ValueError on negative colors or an empty input."""
    return np.zeros((colors.shape[0], int(colors.max()) + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: chromatic superposition state matrix with the declared signs."""

import numpy as np


def _oracle_css_matrix(colors):
    colors = np.asarray(colors)
    if colors.ndim != 1 or colors.size == 0:
        raise ValueError("colors must be a nonempty vector")
    if colors.min() < 0:
        raise ValueError("negative color index")
    n = colors.shape[0]
    nc = int(colors.max()) + 1
    R = np.zeros((n, nc))
    for j in range(n):
        s = 1.0 if ((3 * (j + 1)) % 7) >= 3 else -1.0
        R[j, int(colors[j])] = s
    return R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 3.0)', "call": "css_matrix(colors)", "gold_call": "_oracle_css_matrix(colors)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 5.0)', "call": "css_matrix(colors)", "gold_call": "_oracle_css_matrix(colors)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2\ncolors=greedy_multicoloring(D, 4.0)', "call": "css_matrix(colors)", "gold_call": "_oracle_css_matrix(colors)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\ncolors=_n.zeros(20,dtype=_n.int64)', "call": "css_matrix(colors)", "gold_call": "_oracle_css_matrix(colors)", "tol": 1e-09},
        {"setup": 'import numpy as _n\ncolors=_n.array([0,3,3,0,2,0],dtype=_n.int64)', "call": "css_matrix(colors)", "gold_call": "_oracle_css_matrix(colors)", "tol": 1e-09},
    ]
