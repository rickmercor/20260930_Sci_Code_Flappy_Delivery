"""
Color the orbital-interaction graph for a given distance matrix and cutoff, exactly by the source's greedy multicoloring criterion, processing orbitals in index order and always taking the smallest admissible color.

The chromatic superposition construction starts from a coloring in which orbitals the source's criterion flags for the cutoff never share a color.

Returns
-------
return (n,) int64: color index per orbital
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def greedy_multicoloring(D, dcut):
    """D: (n, n) symmetric nonnegative distance matrix; dcut: positive float.
    Returns (n,) int64: a greedy multicoloring processed in index order,
    always assigning the smallest admissible color, with the conflict
    criterion exactly as the source states it relative to the cutoff.
    Raises ValueError on a non-square or nonfinite D or a nonpositive
    cutoff."""
    return np.zeros(D.shape[0], dtype=np.int64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: greedy multicoloring of the orbital-interaction graph."""

import numpy as np


def _oracle_greedy_multicoloring(D, dcut):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(D)):
        raise ValueError("nonfinite distance matrix")
    if not np.isfinite(dcut) or dcut <= 0:
        raise ValueError("invalid cutoff")
    n = D.shape[0]
    colors = np.full(n, -1, dtype=np.int64)
    for j in range(n):
        forbidden = set()
        for i in range(j):
            if D[i, j] < dcut:
                forbidden.add(int(colors[i]))
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
    return colors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "greedy_multicoloring(D, 3.0)", "gold_call": "_oracle_greedy_multicoloring(D, 3.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2', "call": "greedy_multicoloring(D, 5.0)", "gold_call": "_oracle_greedy_multicoloring(D, 5.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2', "call": "greedy_multicoloring(D, 4.0)", "gold_call": "_oracle_greedy_multicoloring(D, 4.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "greedy_multicoloring(D, 1.0)", "gold_call": "_oracle_greedy_multicoloring(D, 1.0)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "greedy_multicoloring(D, 25.0)", "gold_call": "_oracle_greedy_multicoloring(D, 25.0)", "tol": 1e-09},
    ]
