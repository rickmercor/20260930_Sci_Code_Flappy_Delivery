"""
Build the transverse-field Ising chain and report one matrix element.
Returns the element, dimensionless.

The model is the open chain of $m$ sites



$$H=-\sum_{i=1}^{m}\sigma^z_i-J\sum_{i=1}^{m-1}\sigma^x_i\sigma^x_{i+1}$$



with $\sigma^x_i$ and $\sigma^z_i$ the Pauli matrices. Build it in the product basis of the $m$ spins, ordering the basis so that state index $b$ has site $i$ carrying the bit `(b >> (m - 1 - i)) & 1`, with bit $0$ meaning spin up ($\sigma^z=+1$) and bit $1$ meaning flipped. In that convention the field term is diagonal and contributes $-(m-2n_{\mathrm{flip}})$, with $n_{\mathrm{flip}}$ the number of set bits, and the exchange term connects states differing by a flip of two neighbouring sites. Return the element in row $a$ and column $b$.

Returns
-------
float: the matrix element $H_{ab}$.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hamiltonian_element(a: int, b: int, m: int, dev: dict) -> float:
    r"""Build the transverse-field Ising chain and report one matrix element.

    Returns the element, dimensionless.

    Parameters
    ----------
    a : int
        Row index in the product basis, $0\le a<2^m$.
    b : int
        Column index in the product basis, $0\le b<2^m$.
    m : int
        Number of sites in the chain, $m\ge1$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    element : float
        The matrix element $H_{ab}$.

    Raises
    ------
    ValueError
        If $a$, $b$, $m$ or $J$ is not finite, or if $a$ or $b$ lies outside
        $0\le a,b<2^m$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

def _oracle_hamiltonian_element(a: int, b: int, m: int, dev: dict) -> float:
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    return _nlce_do_hamiltonian_element(a, b, m, dev)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal: the fully polarised diagonal element
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'hamiltonian_element(0, 0, 6, dev)',
         "gold_call": '_oracle_hamiltonian_element(0, 0, 6, dev)'},
        # normal: an off-diagonal element connected by the exchange
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'hamiltonian_element(0, 48, 6, dev)',
         "gold_call": '_oracle_hamiltonian_element(0, 48, 6, dev)'},
        # edge: states differing by a single flip are not connected
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'hamiltonian_element(0, 32, 6, dev)',
         "gold_call": '_oracle_hamiltonian_element(0, 32, 6, dev)'},
        # boundary: a diagonal element in the one-magnon sector
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=0.6)',
         "call": 'hamiltonian_element(4, 4, 6, dev)',
         "gold_call": '_oracle_hamiltonian_element(4, 4, 6, dev)'},
        # boundary: the chain is open, so the two end sites (0 and m-1) are not coupled
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'hamiltonian_element(0, 33, 6, dev)',
         "gold_call": '_oracle_hamiltonian_element(0, 33, 6, dev)'},
    ]
