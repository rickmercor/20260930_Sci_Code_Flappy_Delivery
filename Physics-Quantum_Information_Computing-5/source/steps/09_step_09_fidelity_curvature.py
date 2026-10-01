"""
Compute a pure or mixed curvature response of physical fidelity to coherent control errors.

The initial vector is constant in the dimensionless perturbations $s,u$.

With cross_slopes supplied, angles vary as $A+sB+uC$; otherwise they vary as $A+sB$. These angles determine each ordered two-qubit gate by



$$

G(s)=e^{-ic(s)(Z\otimes Z)/2}[R_Y(a(s))\otimes R_X(b(s))][R_Z(d(s))\otimes R_Y(e(s))],\qquad

R_P(\theta)=\cos(\theta/2)I-i\sin(\theta/2)P.

$$



Here $X,Y,Z$ are the conventional Pauli matrices, $Y=iXZ$, all angles are radians, and the rightmost factor acts first.

Compose the approximate response with gate compression, single-qubit snapshot passes and transient pair sweeps, and propagate the same perturbed gates exactly for the reference.

Restore the approximate state's full moving product frame before differentiating its overlap with the moving exact state:



$$

F(s)=|\langle\psi_{\mathrm{exact}}(s)|\psi_{\mathrm{approx}}(s)\rangle|^2.

$$



Return $\partial_s^2\partial_u^2 F(0,0)$ when cross_slopes is supplied, or $\partial_s^2 F(0,0)$ when it is None. The displayed gate and overlap formulas then depend on both $s$ and $u$. Include the motion of both states, all accumulated frames and every mixed term.

This is the derivative of the branch fixed at zero, with the conventions of the circuit-response step; it may be negative.

Unit base states and base unitaries use maximum residual tolerance $10^{-9}$; derivative rows are supplied ordinary derivatives, not Taylor coefficients.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The requested finite dimensionless ordinary (2,2) or (2,0) partial derivative of physical fidelity at zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fidelity_curvature(
    initial: "np.ndarray",
    angles: "np.ndarray",
    slopes: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
    cross_slopes: "np.ndarray | None" = None,
) -> float:
    r"""Compute a pure or mixed curvature response of physical fidelity to coherent control errors.

    Parameters
    ----------
    initial : np.ndarray, shape (2**N,)
        Finite normalized complex vector, independent of $s,u$, $2 \le N \le 6$.
    angles : np.ndarray, shape (M, 5)
        Finite real base angles in column order $(a,b,c,d,e)$, with $0 \le M \le 24$.
    slopes : np.ndarray, shape (M, 5)
        Finite real derivatives of angles with respect to dimensionless $s$.
    pairs : np.ndarray, shape (M, 2)
        Ordered distinct integer qubit indices.
    k : int
        Retained budget in $[1,2^N]$.
    max_passes : int, optional
        One to three single-qubit passes per gate, default three.

    cross_slopes : np.ndarray or None, shape (M, 5), optional
        Finite real angle derivatives with respect to u. An array requests the mixed (2,2) derivative; None requests the pure (2,0) derivative. A zero array therefore gives zero mixed response, not the pure response.

    Returns
    -------
    curvature : float
        The requested finite dimensionless ordinary (2,2) or (2,0) partial derivative of physical fidelity at zero.

    Raises
    ------
    ValueError
        If any shape, finiteness, base normalization, real-angle condition, index, budget or pass count is invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _rotation(angle, slope, pauli, cross_slope=None):
    unitary = np.cos(angle / 2) * np.eye(len(pauli)) - 1j * np.sin(angle / 2) * pauli
    if cross_slope is None:
        indices = _multi(3)
        cross_slope = 0.0
    else:
        indices = _multi(9)
    generator = -0.5j * pauli
    return np.array(
        [
            slope**a
            * cross_slope**b
            * np.linalg.matrix_power(generator, a + b)
            @ unitary
            for a, b in indices
        ]
    )


def _gate_jets(angles, slopes, cross_slopes=None):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    gates = []
    for row, (a, b) in enumerate(zip(angles, slopes)):
        c = [None] * 5 if cross_slopes is None else cross_slopes[row]
        left = _rotation(a[2], b[2], np.kron(z, z), c[2])
        middle = _kron(_rotation(a[0], b[0], y, c[0]), _rotation(a[1], b[1], x, c[1]))
        right = _kron(_rotation(a[3], b[3], z, c[3]), _rotation(a[4], b[4], y, c[4]))
        gates.append(_matmul(_matmul(left, middle), right))
    return np.array(gates).reshape((-1, 3 if cross_slopes is None else 9, 4, 4))


def _fidelity_jet(exact, physical):
    overlap = _product(exact, physical, np.vdot)
    return _scale(overlap.conj(), overlap).real


def _oracle_fidelity_curvature(
    initial: "np.ndarray",
    angles: "np.ndarray",
    slopes: "np.ndarray",
    pairs: "np.ndarray",
    k: int,
    max_passes: int = 3,
    cross_slopes: "np.ndarray | None" = None,
) -> float:
    initial = np.asarray(initial, dtype=complex)
    if initial.ndim != 1:
        raise ValueError("initial must be a vector")
    state = np.zeros((3 if cross_slopes is None else 9, len(initial)), dtype=complex)
    state[0] = initial
    state = _state(state)
    angles, slopes = np.asarray(angles), np.asarray(slopes)
    if (
        angles.ndim != 2
        or angles.shape[1] != 5
        or angles.shape != slopes.shape
        or np.iscomplexobj(angles)
        or np.iscomplexobj(slopes)
    ):
        raise ValueError("angles and slopes must be real arrays of shape (M, 5)")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(slopes)):
        raise ValueError("angles and slopes must be finite")
    if cross_slopes is not None:
        cross_slopes = np.asarray(cross_slopes)
        if (
            cross_slopes.shape != angles.shape
            or np.iscomplexobj(cross_slopes)
            or not np.all(np.isfinite(cross_slopes))
        ):
            raise ValueError("cross_slopes must be finite real and match angles")
    gates = _gate_jets(angles, slopes, cross_slopes)
    approximate, bases, _ = _oracle_circuit_response_jets(
        state, gates, pairs, k, max_passes
    )
    exact = state.copy()
    for gate, pair in zip(gates, pairs):
        exact = _apply_jet(exact, gate, tuple(pair))
    for j in range(bases.shape[1]):
        approximate = _apply_jet(approximate, bases[:, j], (j,))
    return float(_fidelity_jet(exact, approximate)[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Legacy regressions and shape-checked mixed response cases."""
    return [
        {
            "setup": "import numpy as np\n"
            "\n"
            "initial = np.array(\n"
            "    [\n"
            "        (0.12884541506963756 + 0.2153850572414665j),\n"
            "        (0.20109075171301716 + 0.17856149772479532j),\n"
            "        (0.2401581374848401 + 0.12851597748141133j),\n"
            "        (0.23407070193217744 + 0.07078022897756676j),\n"
            "        (0.19527438923510046 + 0.014988431675607747j),\n"
            "        (0.1652226290673494 - 0.027900938357853514j),\n"
            "        (0.1531374521639625 - 0.05560646739647707j),\n"
            "        (0.07386822605484025 - 0.08553166197015274j),\n"
            "        (-0.08141364821997293 - 0.14362830985826544j),\n"
            "        (-0.1284801433018531 - 0.21955767676038454j),\n"
            "        (-0.16404583290747524 - 0.2511344954152934j),\n"
            "        (-0.2748047062810264 - 0.22081443920860172j),\n"
            "        (-0.20178002558462418 - 0.22088754743078068j),\n"
            "        (-0.25427035615766924 - 0.26023903068176496j),\n"
            "        (-0.12421997488076805 - 0.2071537564602983j),\n"
            "        (-0.1272805030045476 - 0.14930535157170446j),\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "angles = np.array(\n"
            "    [\n"
            "        [0.73, -0.42, 0.61, 0.19, -0.37],\n"
            "        [-0.58, 0.91, -0.47, 0.36, 0.52],\n"
            "        [1.07, 0.33, 0.82, -0.29, 0.64],\n"
            "        [0.41, -1.13, 0.57, 0.83, -0.46],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "slopes = np.array(\n"
            "    [\n"
            "        [0.24, -0.17, 0.09, 0.32, -0.21],\n"
            "        [-0.11, 0.23, 0.35, -0.17, 0.09],\n"
            "        [0.13, 0.21, -0.31, 0.27, 0.16],\n"
            "        [-0.17, 0.13, 0.22, -0.29, 0.31],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "pairs = np.array([[0, 2], [3, 1], [1, 0], [2, 3]], dtype=int)\n"
            "k = 7\n",
            "call": "fidelity_curvature(initial=initial.copy(), angles=angles.copy(), slopes=slopes.copy(), "
            "pairs=pairs.copy(), k=k)",
            "gold_call": "_oracle_fidelity_curvature(initial=initial.copy(), angles=angles.copy(), "
            "slopes=slopes.copy(), pairs=pairs.copy(), k=k)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "initial = np.array(\n"
            "    [\n"
            "        (0.12884541506963756 + 0.2153850572414665j),\n"
            "        (0.20109075171301716 + 0.17856149772479532j),\n"
            "        (0.2401581374848401 + 0.12851597748141133j),\n"
            "        (0.23407070193217744 + 0.07078022897756676j),\n"
            "        (0.19527438923510046 + 0.014988431675607747j),\n"
            "        (0.1652226290673494 - 0.027900938357853514j),\n"
            "        (0.1531374521639625 - 0.05560646739647707j),\n"
            "        (0.07386822605484025 - 0.08553166197015274j),\n"
            "        (-0.08141364821997293 - 0.14362830985826544j),\n"
            "        (-0.1284801433018531 - 0.21955767676038454j),\n"
            "        (-0.16404583290747524 - 0.2511344954152934j),\n"
            "        (-0.2748047062810264 - 0.22081443920860172j),\n"
            "        (-0.20178002558462418 - 0.22088754743078068j),\n"
            "        (-0.25427035615766924 - 0.26023903068176496j),\n"
            "        (-0.12421997488076805 - 0.2071537564602983j),\n"
            "        (-0.1272805030045476 - 0.14930535157170446j),\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "angles = np.array(\n"
            "    [\n"
            "        [0.73, -0.42, 0.61, 0.19, -0.37],\n"
            "        [-0.58, 0.91, -0.47, 0.36, 0.52],\n"
            "        [1.07, 0.33, 0.82, -0.29, 0.64],\n"
            "        [0.41, -1.13, 0.57, 0.83, -0.46],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "slopes = np.array(\n"
            "    [\n"
            "        [0.0, 0.0, 0.0, 0.0, 0.0],\n"
            "        [0.0, 0.0, 0.0, 0.0, 0.0],\n"
            "        [0.0, 0.0, 0.0, 0.0, 0.0],\n"
            "        [0.0, 0.0, 0.0, 0.0, 0.0],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "pairs = np.array([[0, 2], [3, 1], [1, 0], [2, 3]], dtype=int)\n"
            "k = 5\n",
            "call": "fidelity_curvature(initial=initial.copy(), angles=angles.copy(), slopes=slopes.copy(), "
            "pairs=pairs.copy(), k=k)",
            "gold_call": "_oracle_fidelity_curvature(initial=initial.copy(), angles=angles.copy(), "
            "slopes=slopes.copy(), pairs=pairs.copy(), k=k)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "initial = np.array(\n"
            "    [\n"
            "        (0.12884541506963756 + 0.2153850572414665j),\n"
            "        (0.20109075171301716 + 0.17856149772479532j),\n"
            "        (0.2401581374848401 + 0.12851597748141133j),\n"
            "        (0.23407070193217744 + 0.07078022897756676j),\n"
            "        (0.19527438923510046 + 0.014988431675607747j),\n"
            "        (0.1652226290673494 - 0.027900938357853514j),\n"
            "        (0.1531374521639625 - 0.05560646739647707j),\n"
            "        (0.07386822605484025 - 0.08553166197015274j),\n"
            "        (-0.08141364821997293 - 0.14362830985826544j),\n"
            "        (-0.1284801433018531 - 0.21955767676038454j),\n"
            "        (-0.16404583290747524 - 0.2511344954152934j),\n"
            "        (-0.2748047062810264 - 0.22081443920860172j),\n"
            "        (-0.20178002558462418 - 0.22088754743078068j),\n"
            "        (-0.25427035615766924 - 0.26023903068176496j),\n"
            "        (-0.12421997488076805 - 0.2071537564602983j),\n"
            "        (-0.1272805030045476 - 0.14930535157170446j),\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "angles = np.array(\n"
            "    [\n"
            "        [0.41, -1.13, 0.57, 0.83, -0.46],\n"
            "        [1.07, 0.33, 0.82, -0.29, 0.64],\n"
            "        [-0.58, 0.91, -0.47, 0.36, 0.52],\n"
            "        [0.73, -0.42, 0.61, 0.19, -0.37],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "slopes = np.array(\n"
            "    [\n"
            "        [0.28900000000000003, -0.221, -0.374, 0.49299999999999994, -0.527],\n"
            "        [-0.221, -0.357, 0.527, -0.459, -0.272],\n"
            "        [0.187, -0.391, -0.595, 0.28900000000000003, -0.153],\n"
            "        [-0.408, 0.28900000000000003, -0.153, -0.544, 0.357],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "pairs = np.array([[2, 3], [1, 0], [3, 1], [0, 2]], dtype=int)\n"
            "k = 6\n"
            "max_passes = 2\n",
            "call": "fidelity_curvature(initial=initial.copy(), angles=angles.copy(), slopes=slopes.copy(), "
            "pairs=pairs.copy(), k=k, max_passes=max_passes)",
            "gold_call": "_oracle_fidelity_curvature(initial=initial.copy(), angles=angles.copy(), "
            "slopes=slopes.copy(), pairs=pairs.copy(), k=k, max_passes=max_passes)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "initial = np.array(\n"
            "    [\n"
            "        (0.12884541506963756 + 0.2153850572414665j),\n"
            "        (0.20109075171301716 + 0.17856149772479532j),\n"
            "        (0.2401581374848401 + 0.12851597748141133j),\n"
            "        (0.23407070193217744 + 0.07078022897756676j),\n"
            "        (0.19527438923510046 + 0.014988431675607747j),\n"
            "        (0.1652226290673494 - 0.027900938357853514j),\n"
            "        (0.1531374521639625 - 0.05560646739647707j),\n"
            "        (0.07386822605484025 - 0.08553166197015274j),\n"
            "        (-0.08141364821997293 - 0.14362830985826544j),\n"
            "        (-0.1284801433018531 - 0.21955767676038454j),\n"
            "        (-0.16404583290747524 - 0.2511344954152934j),\n"
            "        (-0.2748047062810264 - 0.22081443920860172j),\n"
            "        (-0.20178002558462418 - 0.22088754743078068j),\n"
            "        (-0.25427035615766924 - 0.26023903068176496j),\n"
            "        (-0.12421997488076805 - 0.2071537564602983j),\n"
            "        (-0.1272805030045476 - 0.14930535157170446j),\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "angles = np.array(\n"
            "    [\n"
            "        [np.nan, np.nan, np.nan, np.nan, np.nan],\n"
            "        [np.nan, np.nan, np.nan, np.nan, np.nan],\n"
            "        [np.nan, np.nan, np.nan, np.nan, np.nan],\n"
            "        [np.nan, np.nan, np.nan, np.nan, np.nan],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "slopes = np.array(\n"
            "    [\n"
            "        [0.24, -0.17, 0.09, 0.32, -0.21],\n"
            "        [-0.11, 0.23, 0.35, -0.17, 0.09],\n"
            "        [0.13, 0.21, -0.31, 0.27, 0.16],\n"
            "        [-0.17, 0.13, 0.22, -0.29, 0.31],\n"
            "    ],\n"
            "    dtype=float,\n"
            ")\n"
            "pairs = np.array([[0, 2], [3, 1], [1, 0], [2, 3]], dtype=int)\n"
            "k = 7\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: fidelity_curvature(initial=initial.copy(), "
            "angles=angles.copy(), slopes=slopes.copy(), pairs=pairs.copy(), k=k))",
            "gold_call": "_raises_value_error(lambda: _oracle_fidelity_curvature(initial=initial.copy(), "
            "angles=angles.copy(), slopes=slopes.copy(), pairs=pairs.copy(), k=k))",
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
            "initial=state[0].copy()\n"
            "angles=rng.uniform(-1.2,1.2,size=(3,5))\n"
            "slopes=rng.uniform(-.4,.4,size=(3,5))\n"
            "cross_slopes=rng.uniform(-.3,.3,size=(3,5))\n"
            "pairs=np.array([[n-1,0],[1,0],[0,n-1]],dtype=int)\n"
            "\n"
            "_expected_shape=()\n",
            "call": "_pack(fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), pairs.copy(), k, "
            "2, cross_slopes=cross_slopes.copy()))",
            "gold_call": "_pack(_oracle_fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), "
            "pairs.copy(), k, 2, cross_slopes=cross_slopes.copy()))",
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
            "initial=state[0].copy()\n"
            "angles=rng.uniform(-1.2,1.2,size=(3,5))\n"
            "slopes=rng.uniform(-.4,.4,size=(3,5))\n"
            "cross_slopes=rng.uniform(-.3,.3,size=(3,5))\n"
            "pairs=np.array([[n-1,0],[1,0],[0,n-1]],dtype=int)\n"
            "\n"
            "_expected_shape=()\n",
            "call": "_pack(fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), pairs.copy(), k, "
            "2, cross_slopes=cross_slopes.copy()))",
            "gold_call": "_pack(_oracle_fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), "
            "pairs.copy(), k, 2, cross_slopes=cross_slopes.copy()))",
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
            "initial=state[0].copy()\n"
            "angles=rng.uniform(-1.2,1.2,size=(3,5))\n"
            "slopes=rng.uniform(-.4,.4,size=(3,5))\n"
            "cross_slopes=rng.uniform(-.3,.3,size=(3,5))\n"
            "pairs=np.array([[n-1,0],[1,0],[0,n-1]],dtype=int)\n"
            "\n"
            "_expected_shape=()\n",
            "call": "_pack(fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), pairs.copy(), k, "
            "2, cross_slopes=cross_slopes.copy()))",
            "gold_call": "_pack(_oracle_fidelity_curvature(initial.copy(), angles.copy(), slopes.copy(), "
            "pairs.copy(), k, 2, cross_slopes=cross_slopes.copy()))",
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
            "initial=state[0].copy()\n"
            "angles=rng.uniform(-1.2,1.2,size=(3,5))\n"
            "slopes=rng.uniform(-.4,.4,size=(3,5))\n"
            "cross_slopes=rng.uniform(-.3,.3,size=(3,5))\n"
            "pairs=np.array([[n-1,0],[1,0],[0,n-1]],dtype=int)\n"
            "\n"
            "cross_slopes=cross_slopes[:,:4]\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n"
            "\n"
            "_expected_shape=()\n",
            "call": "_raises_value_error(lambda: fidelity_curvature(initial.copy(), angles.copy(), "
            "slopes.copy(), pairs.copy(), k, 2, cross_slopes=cross_slopes.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_fidelity_curvature(initial.copy(), "
            "angles.copy(), slopes.copy(), pairs.copy(), k, 2, "
            "cross_slopes=cross_slopes.copy()))",
            "tol": 0,
        },
    ]
