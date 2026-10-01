"""
Continue a local natural eigenframe through every supplied derivative order in a fixed phase convention.

For J=9, the Hermitian reduced-state jet specifies $A(s,u)=\sum_{a,b=0}^2 A_{ab}s^a u^b/(a!b!)$ through bidegree $(2,2)$. For J=3 use its one-parameter restriction. The equations below apply throughout this local continuation; both variables enter every continuous quantity.

The columns of the requested unitary frame obey



$$

A(s)v_j(s)=\lambda_j(s)v_j(s),\qquad V(s)^\dagger V(s)=I.

$$



Order the eigenvalues decreasingly at zero. If any adjacent eigenvalue gap is at most the supplied threshold, return an identity frame with zero derivatives of positive order.

Otherwise choose each base column's largest-magnitude component positive real, resolving an exact tie by its lowest index, and continue with that same component positive real for small $s$.

Differentiate the eigenproblem, normalization and phase condition; the output includes motion of the eigenvectors, not just their base values.

The fixed phase pivot is part of the coordinate convention. The base reduced state is positive semidefinite and trace one; each derivative matrix is Hermitian.

Hermiticity, base trace and positivity are checked to tolerance $10^{-9}$.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The eigenframe and all supplied ordinary partial derivatives in the fixed pivot gauge.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def natural_frame_jets(rdm: "np.ndarray", gap_tol: float = 1e-10) -> "np.ndarray":
    r"""Continue a local natural eigenframe through every supplied derivative order in a fixed phase convention.

    Parameters
    ----------
    rdm : np.ndarray, shape (J, D, D)
        Finite Hermitian jet, $D\in\{2,4\}$; base trace one and eigenvalues at least $-10^{-9}$.
    gap_tol : float, optional
        Real finite threshold in $[0,10^{-8}]$, default $10^{-10}$.

    Returns
    -------
    frames : np.ndarray, shape (J, D, D)
        The eigenframe and its supplied ordinary partial derivatives in the fixed pivot gauge.

    Raises
    ------
    ValueError
        If the shape, finiteness, Hermiticity, base density-matrix conditions or threshold are invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_natural_frame_jets(
    rdm: "np.ndarray", gap_tol: float = 1e-10
) -> "np.ndarray":
    rdm = np.asarray(rdm, dtype=complex)
    if rdm.shape not in ((3, 2, 2), (3, 4, 4), (9, 2, 2), (9, 4, 4)):
        raise ValueError("rdm must have J derivative rows and local dimension 2 or 4")
    rdm = _jet(rdm, rdm.shape, "rdm")
    _controls(1, gap_tol, 0.0)
    if np.max(abs(rdm - rdm.conj().swapaxes(-1, -2))) > 1e-9:
        raise ValueError("every rdm derivative must be Hermitian")
    values, vectors = np.linalg.eigh(rdm[0])
    if values[0] < -1e-9 or abs(np.trace(rdm[0]) - 1) > 1e-9:
        raise ValueError("base rdm must be positive semidefinite with unit trace")
    length, d = len(rdm), len(values)
    result = np.zeros_like(rdm)
    if np.min(np.diff(values)) <= gap_tol:
        result[0] = np.eye(d)
        return result
    values, vectors = values[::-1], vectors[:, ::-1]
    indices = _multi(length)
    factors = np.array([factorial(a) * factorial(b) for a, b in indices])
    coefficients = rdm / factors[:, None, None]
    for column in range(d):
        v = vectors[:, column].copy()
        pivot = int(np.argmax(abs(v)))
        v *= np.conj(v[pivot]) / abs(v[pivot])
        wave = np.zeros((length, d), complex)
        energy = np.zeros(length, complex)
        wave[0], energy[0] = v, values[column]
        for row in sorted(range(1, length), key=lambda r: sum(indices[r])):
            rhs = np.zeros(d, complex)
            for i, j, _ in _terms(length, row):
                if i:
                    rhs += coefficients[i] @ wave[j]
                    if j:
                        rhs -= energy[i] * wave[j]
            energy[row] = np.vdot(v, rhs)
            for other in range(d):
                if other != column:
                    u = vectors[:, other]
                    wave[row] += u * np.vdot(u, rhs) / (values[column] - values[other])
        wave *= factors[:, None]
        wave = _scale(wave, _power(_norm(wave), -0.5))
        pivot_jet = wave[:, pivot]
        phase = _scale(
            pivot_jet.conj(), _power(_scale(pivot_jet.conj(), pivot_jet).real, -0.5)
        )
        result[:, :, column] = _scale(wave, phase)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Legacy regressions and shape-checked mixed response cases."""
    return [
        {
            "setup": "import numpy as np\n"
            "\n"
            "rdm = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                (0.20208404998238821 + 6.491429844048785e-19j),\n"
            "                (0.177593136086219 + 0.05030380634886192j),\n"
            "                (-0.16331377491334365 + 0.11893979526754893j),\n"
            "                (-0.2229604223041484 + 0.09527865016613424j),\n"
            "            ],\n"
            "            [\n"
            "                (0.177593136086219 - 0.050303806348861925j),\n"
            "                (0.17296979274840857 + 1.0172772218580666e-18j),\n"
            "                (-0.1168250587786825 + 0.12956930262270142j),\n"
            "                (-0.17637569695052988 + 0.1254520549839041j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.16331377491334365 - 0.11893979526754891j),\n"
            "                (-0.1168250587786825 - 0.12956930262270142j),\n"
            "                (0.2650866119009369 + 9.473114263961738e-20j),\n"
            "                (0.2980596104470637 + 0.044841691411371044j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.2229604223041484 - 0.09527865016613422j),\n"
            "                (-0.17637569695052988 - 0.12545205498390408j),\n"
            "                (0.2980596104470637 - 0.04484169141137106j),\n"
            "                (0.3598595453682662 + 1.3985636206322989e-18j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (-0.006912220382459508 + 0j),\n"
            "                (-0.009117182208738988 - 0.01630261314122602j),\n"
            "                (-0.022992749657289716 - 0.05061604284668759j),\n"
            "                (-0.01588231688241489 - 0.042473213667895436j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.009117182208738988 + 0.01630261314122602j),\n"
            "                (-0.014444051950663358 + 0j),\n"
            "                (-0.027352921530719804 - 0.065271032733443j),\n"
            "                (-0.01802322868345922 - 0.06295351270775712j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.022992749657289716 + 0.050616042846687595j),\n"
            "                (-0.027352921530719804 + 0.065271032733443j),\n"
            "                (0.009711935285677946 - 6.938893903907228e-18j),\n"
            "                (0.013428526914020043 - 0.025176990119503476j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.01588231688241489 + 0.042473213667895436j),\n"
            "                (-0.01802322868345922 + 0.06295351270775712j),\n"
            "                (0.013428526914020043 + 0.025176990119503473j),\n"
            "                (0.011644337047444938 - 4.336808689942018e-19j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (0.017931769592489946 + 1.734723475976807e-18j),\n"
            "                (0.019798475688413537 - 0.010104595576530885j),\n"
            "                (-0.0017799868027730918 - 0.021632190128040185j),\n"
            "                (-0.0015625227678154937 - 0.008977900713910626j),\n"
            "            ],\n"
            "            [\n"
            "                (0.019798475688413537 + 0.010104595576530885j),\n"
            "                (0.023386820602307692 + 1.734723475976807e-18j),\n"
            "                (-0.006246453764602948 - 0.030694734558692045j),\n"
            "                (-0.0068928335798547615 - 0.02217917444346175j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.0017799868027730914 + 0.021632190128040185j),\n"
            "                (-0.006246453764602948 + 0.030694734558692042j),\n"
            "                (-0.017489917597481582 + 3.469446951953614e-18j),\n"
            "                (-0.01946954117414121 - 0.01873772372154609j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.0015625227678154937 + 0.008977900713910626j),\n"
            "                (-0.006892833579854762 + 0.02217917444346175j),\n"
            "                (-0.01946954117414121 + 0.018737723721546088j),\n"
            "                (-0.023828672597316053 - 1.734723475976807e-18j),\n"
            "            ],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n",
            "call": "natural_frame_jets(rdm=rdm.copy())",
            "gold_call": "_oracle_natural_frame_jets(rdm=rdm.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "rdm = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                (0.5379155189694336 + 6.728947566421383e-18j),\n"
            "                (0.43192000604256947 + 0.23451333595366972j),\n"
            "            ],\n"
            "            [\n"
            "                (0.43192000604256947 - 0.2345133359536697j),\n"
            "                (0.46208448103056626 + 7.711796975817171e-19j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (-0.06208763275899371 + 0j),\n"
            "                (-0.025843738129357943 + 0.08961813903416806j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.025843738129357943 - 0.08961813903416807j),\n"
            "                (0.06208763275899373 + 0j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (-0.03860651297331985 + 2.7755575615628914e-17j),\n"
            "                (0.01385774593882784 - 0.2368430959256571j),\n"
            "            ],\n"
            "            [(0.013857745938827833 + 0.2368430959256571j), (0.03860651297331974 + "
            "0j)],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n",
            "call": "natural_frame_jets(rdm=rdm.copy())",
            "gold_call": "_oracle_natural_frame_jets(rdm=rdm.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "rdm = np.array(\n"
            "    [\n"
            "        [\n"
            "            [(0.25 + 0j), 0j, 0j, 0j],\n"
            "            [0j, (0.25 + 0j), 0j, 0j],\n"
            "            [0j, 0j, (0.25 + 0j), 0j],\n"
            "            [0j, 0j, 0j, (0.25 + 0j)],\n"
            "        ],\n"
            "        [\n"
            "            [(0.2 + 0j), 0j, 0j, 0j],\n"
            "            [0j, (-0.2 + 0j), 0j, 0j],\n"
            "            [0j, 0j, (0.1 + 0j), 0j],\n"
            "            [0j, 0j, 0j, (-0.1 + 0j)],\n"
            "        ],\n"
            "        [[0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j]],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n",
            "call": "natural_frame_jets(rdm=rdm.copy())",
            "gold_call": "_oracle_natural_frame_jets(rdm=rdm.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "rdm = np.array(\n"
            "    [\n"
            "        [\n"
            "            [(0.41 + 0j), 0j, 0j, 0j],\n"
            "            [0j, (0.29 + 0j), 0j, 0j],\n"
            "            [0j, 0j, (0.15002 + 0j), 0j],\n"
            "            [0j, 0j, 0j, (0.14998 + 0j)],\n"
            "        ],\n"
            "        [\n"
            "            [(0.01 + 0j), 0.002j, (0.003 + 0j), 0j],\n"
            "            [(-0 - 0.002j), (-0.01 + 0j), (0.002 + 0j), 0.001j],\n"
            "            [(0.003 + 0j), (0.002 + 0j), 0j, 0.0003j],\n"
            "            [0j, (-0 - 0.001j), (-0 - 0.0003j), 0j],\n"
            "        ],\n"
            "        [\n"
            "            [(0.006999999999999999 + 0j), 0.0014j, (0.0021 + 0j), 0j],\n"
            "            [-0.0014j, (-0.006999999999999999 + 0j), (0.0014 + 0j), 0.0007j],\n"
            "            [(0.0021 + 0j), (0.0014 + 0j), 0j, 0.00020999999999999998j],\n"
            "            [0j, -0.0007j, -0.00020999999999999998j, 0j],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n",
            "call": "natural_frame_jets(rdm=rdm.copy())",
            "gold_call": "_oracle_natural_frame_jets(rdm=rdm.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "rdm = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                (-2 + 0j),\n"
            "                (0.177593136086219 + 0.05030380634886192j),\n"
            "                (-0.16331377491334365 + 0.11893979526754893j),\n"
            "                (-0.2229604223041484 + 0.09527865016613424j),\n"
            "            ],\n"
            "            [\n"
            "                (0.177593136086219 - 0.050303806348861925j),\n"
            "                (0.17296979274840857 + 1.0172772218580666e-18j),\n"
            "                (-0.1168250587786825 + 0.12956930262270142j),\n"
            "                (-0.17637569695052988 + 0.1254520549839041j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.16331377491334365 - 0.11893979526754891j),\n"
            "                (-0.1168250587786825 - 0.12956930262270142j),\n"
            "                (0.2650866119009369 + 9.473114263961738e-20j),\n"
            "                (0.2980596104470637 + 0.044841691411371044j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.2229604223041484 - 0.09527865016613422j),\n"
            "                (-0.17637569695052988 - 0.12545205498390408j),\n"
            "                (0.2980596104470637 - 0.04484169141137106j),\n"
            "                (0.3598595453682662 + 1.3985636206322989e-18j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (-0.006912220382459508 + 0j),\n"
            "                (-0.009117182208738988 - 0.01630261314122602j),\n"
            "                (-0.022992749657289716 - 0.05061604284668759j),\n"
            "                (-0.01588231688241489 - 0.042473213667895436j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.009117182208738988 + 0.01630261314122602j),\n"
            "                (-0.014444051950663358 + 0j),\n"
            "                (-0.027352921530719804 - 0.065271032733443j),\n"
            "                (-0.01802322868345922 - 0.06295351270775712j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.022992749657289716 + 0.050616042846687595j),\n"
            "                (-0.027352921530719804 + 0.065271032733443j),\n"
            "                (0.009711935285677946 - 6.938893903907228e-18j),\n"
            "                (0.013428526914020043 - 0.025176990119503476j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.01588231688241489 + 0.042473213667895436j),\n"
            "                (-0.01802322868345922 + 0.06295351270775712j),\n"
            "                (0.013428526914020043 + 0.025176990119503473j),\n"
            "                (0.011644337047444938 - 4.336808689942018e-19j),\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                (0.017931769592489946 + 1.734723475976807e-18j),\n"
            "                (0.019798475688413537 - 0.010104595576530885j),\n"
            "                (-0.0017799868027730918 - 0.021632190128040185j),\n"
            "                (-0.0015625227678154937 - 0.008977900713910626j),\n"
            "            ],\n"
            "            [\n"
            "                (0.019798475688413537 + 0.010104595576530885j),\n"
            "                (0.023386820602307692 + 1.734723475976807e-18j),\n"
            "                (-0.006246453764602948 - 0.030694734558692045j),\n"
            "                (-0.0068928335798547615 - 0.02217917444346175j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.0017799868027730914 + 0.021632190128040185j),\n"
            "                (-0.006246453764602948 + 0.030694734558692042j),\n"
            "                (-0.017489917597481582 + 3.469446951953614e-18j),\n"
            "                (-0.01946954117414121 - 0.01873772372154609j),\n"
            "            ],\n"
            "            [\n"
            "                (-0.0015625227678154937 + 0.008977900713910626j),\n"
            "                (-0.006892833579854762 + 0.02217917444346175j),\n"
            "                (-0.01946954117414121 + 0.018737723721546088j),\n"
            "                (-0.023828672597316053 - 1.734723475976807e-18j),\n"
            "            ],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: natural_frame_jets(rdm=rdm.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_natural_frame_jets(rdm=rdm.copy()))",
            "tol": 0,
        },
        {
            "setup": "import numpy as np\n"
            "from math import comb, factorial\n"
            "indices = [(a,b) for a in range(3) for b in range(3)]\n"
            "def _multiply(left, right):\n"
            "    out = np.zeros((9, left.shape[1], right.shape[2]), complex)\n"
            "    for row,(a,b) in enumerate(indices):\n"
            "        for c in range(a+1):\n"
            "            for d in range(b+1):\n"
            "                out[row] += comb(a,c)*comb(b,d)*left[3*c+d]@right[3*(a-c)+b-d]\n"
            "    return out\n"
            "def _curve(dimension):\n"
            "    h = "
            "rng.normal(size=(3,dimension,dimension))+1j*rng.normal(size=(3,dimension,dimension))\n"
            "    h = (h-h.conj().swapaxes(-1,-2))/(5*dimension)\n"
            "    q,_ = "
            "np.linalg.qr(rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension)))\n"
            "    left = np.zeros((9,dimension,dimension),complex)\n"
            "    right = left.copy()\n"
            "    mixed = left.copy()\n"
            "    for a in range(3):\n"
            "        left[3*a] = np.linalg.matrix_power(h[0],a)\n"
            "        right[a] = np.linalg.matrix_power(h[1],a)\n"
            "        mixed[4*a] = factorial(a)*np.linalg.matrix_power(h[2],a)\n"
            "    return _multiply(_multiply(left,right),mixed)@q\n"
            "def _pack(value):\n"
            "    if np.shape(value) != _expected_shape:\n"
            '        raise AssertionError("return has the wrong shape")\n'
            "    return np.asarray(value).reshape(-1)\n"
            "rng=np.random.default_rng(2081)\n"
            "n=4\n"
            "k=7\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "axes=[n-1-int(q) for q in qubits]\n"
            "order=axes+[j for j in range(n) if j not in axes]\n"
            "blocks=np.array([v.reshape((2,)*n).transpose(order).reshape(4,-1) for v in state])\n"
            "rdm=_multiply(blocks,blocks.conj().swapaxes(-1,-2))\n"
            "\n"
            "_expected_shape=rdm.shape\n",
            "call": "_pack(natural_frame_jets(rdm.copy()))",
            "gold_call": "_pack(_oracle_natural_frame_jets(rdm.copy()))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "from math import comb, factorial\n"
            "indices = [(a,b) for a in range(3) for b in range(3)]\n"
            "def _multiply(left, right):\n"
            "    out = np.zeros((9, left.shape[1], right.shape[2]), complex)\n"
            "    for row,(a,b) in enumerate(indices):\n"
            "        for c in range(a+1):\n"
            "            for d in range(b+1):\n"
            "                out[row] += comb(a,c)*comb(b,d)*left[3*c+d]@right[3*(a-c)+b-d]\n"
            "    return out\n"
            "def _curve(dimension):\n"
            "    h = "
            "rng.normal(size=(3,dimension,dimension))+1j*rng.normal(size=(3,dimension,dimension))\n"
            "    h = (h-h.conj().swapaxes(-1,-2))/(5*dimension)\n"
            "    q,_ = "
            "np.linalg.qr(rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension)))\n"
            "    left = np.zeros((9,dimension,dimension),complex)\n"
            "    right = left.copy()\n"
            "    mixed = left.copy()\n"
            "    for a in range(3):\n"
            "        left[3*a] = np.linalg.matrix_power(h[0],a)\n"
            "        right[a] = np.linalg.matrix_power(h[1],a)\n"
            "        mixed[4*a] = factorial(a)*np.linalg.matrix_power(h[2],a)\n"
            "    return _multiply(_multiply(left,right),mixed)@q\n"
            "def _pack(value):\n"
            "    if np.shape(value) != _expected_shape:\n"
            '        raise AssertionError("return has the wrong shape")\n'
            "    return np.asarray(value).reshape(-1)\n"
            "rng=np.random.default_rng(711)\n"
            "n=3\n"
            "k=3\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "axes=[n-1-int(q) for q in qubits]\n"
            "order=axes+[j for j in range(n) if j not in axes]\n"
            "blocks=np.array([v.reshape((2,)*n).transpose(order).reshape(4,-1) for v in state])\n"
            "rdm=_multiply(blocks,blocks.conj().swapaxes(-1,-2))\n"
            "rdm=np.zeros((9,4,4),complex)\n"
            "rdm[0]=np.eye(4)/4\n"
            "rdm[4]=np.diag([.1,-.1,.2,-.2])\n"
            "\n"
            "_expected_shape=rdm.shape\n",
            "call": "_pack(natural_frame_jets(rdm.copy()))",
            "gold_call": "_pack(_oracle_natural_frame_jets(rdm.copy()))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "from math import comb, factorial\n"
            "indices = [(a,b) for a in range(3) for b in range(3)]\n"
            "def _multiply(left, right):\n"
            "    out = np.zeros((9, left.shape[1], right.shape[2]), complex)\n"
            "    for row,(a,b) in enumerate(indices):\n"
            "        for c in range(a+1):\n"
            "            for d in range(b+1):\n"
            "                out[row] += comb(a,c)*comb(b,d)*left[3*c+d]@right[3*(a-c)+b-d]\n"
            "    return out\n"
            "def _curve(dimension):\n"
            "    h = "
            "rng.normal(size=(3,dimension,dimension))+1j*rng.normal(size=(3,dimension,dimension))\n"
            "    h = (h-h.conj().swapaxes(-1,-2))/(5*dimension)\n"
            "    q,_ = "
            "np.linalg.qr(rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension)))\n"
            "    left = np.zeros((9,dimension,dimension),complex)\n"
            "    right = left.copy()\n"
            "    mixed = left.copy()\n"
            "    for a in range(3):\n"
            "        left[3*a] = np.linalg.matrix_power(h[0],a)\n"
            "        right[a] = np.linalg.matrix_power(h[1],a)\n"
            "        mixed[4*a] = factorial(a)*np.linalg.matrix_power(h[2],a)\n"
            "    return _multiply(_multiply(left,right),mixed)@q\n"
            "def _pack(value):\n"
            "    if np.shape(value) != _expected_shape:\n"
            '        raise AssertionError("return has the wrong shape")\n'
            "    return np.asarray(value).reshape(-1)\n"
            "rng=np.random.default_rng(993)\n"
            "n=4\n"
            "k=16\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "axes=[n-1-int(q) for q in qubits]\n"
            "order=axes+[j for j in range(n) if j not in axes]\n"
            "blocks=np.array([v.reshape((2,)*n).transpose(order).reshape(4,-1) for v in state])\n"
            "rdm=_multiply(blocks,blocks.conj().swapaxes(-1,-2))\n"
            "\n"
            "_expected_shape=rdm.shape\n",
            "call": "_pack(natural_frame_jets(rdm.copy()))",
            "gold_call": "_pack(_oracle_natural_frame_jets(rdm.copy()))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "from math import comb, factorial\n"
            "indices = [(a,b) for a in range(3) for b in range(3)]\n"
            "def _multiply(left, right):\n"
            "    out = np.zeros((9, left.shape[1], right.shape[2]), complex)\n"
            "    for row,(a,b) in enumerate(indices):\n"
            "        for c in range(a+1):\n"
            "            for d in range(b+1):\n"
            "                out[row] += comb(a,c)*comb(b,d)*left[3*c+d]@right[3*(a-c)+b-d]\n"
            "    return out\n"
            "def _curve(dimension):\n"
            "    h = "
            "rng.normal(size=(3,dimension,dimension))+1j*rng.normal(size=(3,dimension,dimension))\n"
            "    h = (h-h.conj().swapaxes(-1,-2))/(5*dimension)\n"
            "    q,_ = "
            "np.linalg.qr(rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension)))\n"
            "    left = np.zeros((9,dimension,dimension),complex)\n"
            "    right = left.copy()\n"
            "    mixed = left.copy()\n"
            "    for a in range(3):\n"
            "        left[3*a] = np.linalg.matrix_power(h[0],a)\n"
            "        right[a] = np.linalg.matrix_power(h[1],a)\n"
            "        mixed[4*a] = factorial(a)*np.linalg.matrix_power(h[2],a)\n"
            "    return _multiply(_multiply(left,right),mixed)@q\n"
            "def _pack(value):\n"
            "    if np.shape(value) != _expected_shape:\n"
            '        raise AssertionError("return has the wrong shape")\n'
            "    return np.asarray(value).reshape(-1)\n"
            "rng=np.random.default_rng(2081)\n"
            "n=4\n"
            "k=7\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "axes=[n-1-int(q) for q in qubits]\n"
            "order=axes+[j for j in range(n) if j not in axes]\n"
            "blocks=np.array([v.reshape((2,)*n).transpose(order).reshape(4,-1) for v in state])\n"
            "rdm=_multiply(blocks,blocks.conj().swapaxes(-1,-2))\n"
            "\n"
            "known_frame=_curve(4)\n"
            "spectrum=np.zeros((9,4,4),complex)\n"
            "spectrum[0]=np.diag([.41,.29,.155,.145])\n"
            "spectrum[1]=np.diag([.011,-.017,.009,-.003])\n"
            "spectrum[3]=np.diag([.021,-.014,-.011,.004])\n"
            "spectrum[4]=np.diag([.013,-.005,.007,-.015])\n"
            "rdm=_multiply(_multiply(known_frame,spectrum),known_frame.conj().swapaxes(-1,-2))\n"
            "\n"
            "_expected_shape=rdm.shape\n",
            "call": "_pack(natural_frame_jets(rdm.copy()))",
            "gold_call": "_pack(_oracle_natural_frame_jets(rdm.copy()))",
            "tol": 1e-07,
        },
    ]
