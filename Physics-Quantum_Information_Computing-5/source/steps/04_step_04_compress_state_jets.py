"""
Differentiate sparse projection, normalization and participation on a fixed retained support.

Select $S_k$ using the largest squared base amplitudes, with the smaller global index first on an exact tie, and hold this support fixed for all derivatives.

Projection, retained mass and the normalized vector are



$$

y(s)=P_{S_k}v(s),\qquad m(s)=y(s)^\dagger y(s),\qquad w(s)=\frac{y(s)}{\sqrt{m(s)}}.

$$



The participation ratio is



$$

\operatorname{PR}(w)=\frac{(\sum_x|w_x|^2)^2}{\sum_x|w_x|^4}.

$$



Return ordinary derivatives through every supplied derivative order of all supplied quantities, including normalization response.

The input need not be normalized but must have positive finite base mass.

When base weights tie, this definition is the derivative of the selected projection branch, even if a re-ranked finite perturbation would choose another support.

Unit base states and base unitaries use maximum residual tolerance $10^{-9}$; derivative rows are supplied ordinary derivatives, not Taylor coefficients.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The normalized state jet, retained-mass jet and participation-ratio jet in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compress_state_jets(state: "np.ndarray", k: int) -> tuple:
    r"""Differentiate sparse projection, normalization and participation on a fixed retained support.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite complex derivative rows with positive base mass, $1 \le N \le 6$.
    k : int
        Retained budget in $[1,2^N]$.

    Returns
    -------
    result : tuple
        Ordered tuple of normalized state jet, shape (J, 2**N), retained-mass jet, shape (J,), and participation-ratio jet, shape (J,).

    Raises
    ------
    ValueError
        If the shape, finiteness, positive base mass or integer budget is invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compress_state_jets(state: "np.ndarray", k: int) -> tuple:
    state = _state(state, normalized=False)
    k = _integer(k, 1, state.shape[1], "k")
    support = np.lexsort((np.arange(state.shape[1]), -(abs(state[0]) ** 2)))[:k]
    retained = np.zeros_like(state)
    retained[:, support] = state[:, support]
    mass = _norm(retained)
    result = _scale(retained, _power(mass, -0.5))
    return result, mass, _pr(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Legacy regressions and shape-checked mixed response cases."""
    return [
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.2735051519544164 + 0.45720620153770497j),\n"
            "            (0.4268631256624138 + 0.3790394058029379j),\n"
            "            (0.5097929782785058 + 0.2728058420288081j),\n"
            "            (0.49687094310213686 + 0.1502479329312175j),\n"
            "            (0.4145165077987919 + 0.03181652432707721j),\n"
            "            (0.35072447277188695 - 0.05922640228300827j),\n"
            "            (0.3250707997749054 - 0.11803800163709265j),\n"
            "            (0.15680294390617536 - 0.18156137097635108j),\n"
            "        ],\n"
            "        [\n"
            "            (0.09667904088542927 + 0.028717550711972437j),\n"
            "            (0.064770446866267 + 0.06923533107657157j),\n"
            "            (0.027508891077253253 + 0.10475308853525724j),\n"
            "            (-0.011069217606725096 + 0.1325854761914147j),\n"
            "            (-0.04798225314781576 + 0.1503960227903566j),\n"
            "            (-0.0819262045763191 + 0.1564884637289539j),\n"
            "            (-0.11003805885927362 + 0.15046982770949557j),\n"
            "            (-0.1245745959973257 + 0.133915024891705j),\n"
            "        ],\n"
            "        [\n"
            "            (0.06426911991006659 - 0.13279100344785003j),\n"
            "            (0.05356247183336736 - 0.1080060166337035j),\n"
            "            (0.033657079579839065 - 0.07467106542601022j),\n"
            "            (0.009360613542048278 - 0.03592797166153005j),\n"
            "            (-0.01802459197997399 + 0.0037667274750136505j),\n"
            "            (-0.052598191709417215 + 0.03950991804136658j),\n"
            "            (-0.08956151380882982 + 0.06864065392188672j),\n"
            "            (-0.10082485205282775 + 0.09367533575458656j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 4\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(compress_state_jets(state=state.copy(), k=k), ((3, 8), (3,), (3,)))",
            "gold_call": "_pack(_oracle_compress_state_jets(state=state.copy(), k=k), ((3, 8), (3,), (3,)))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.21038857842647413 + 0.3516970781059269j),\n"
            "            (0.3283562505095491 + 0.2915687736945676j),\n"
            "            (0.39214844482961986 + 0.20985064771446776j),\n"
            "            (0.3822084177708745 + 0.11557533302401345j),\n"
            "            (0.3188588521529168 + 0.024474249482367087j),\n"
            "            (0.2697880559783746 - 0.04555877098692943j),\n"
            "            (0.25005446136531184 - 0.09079846279776357j),\n"
            "            (0.12061764915859643 - 0.1396625930587316j),\n"
            "        ],\n"
            "        [\n"
            "            (0.05687002405025251 + 0.016892676889395553j),\n"
            "            (0.038100262862510004 + 0.040726665339159744j),\n"
            "            (0.016181700633678384 + 0.06161946384426896j),\n"
            "            (-0.006511304474544175 + 0.07799145658318513j),\n"
            "            (-0.0282248547928328 + 0.08846824870020976j),\n"
            "            (-0.04819188504489359 + 0.09205203748761993j),\n"
            "            (-0.06472826991721978 + 0.08851166335852681j),\n"
            "            (-0.07327917411607394 + 0.07877354405394413j),\n"
            "        ],\n"
            "        [\n"
            "            (0.030604342814317422 - 0.06323381116564286j),\n"
            "            (0.02550593896827017 - 0.05143143649223976j),\n"
            "            (0.016027180752304315 - 0.03555765020286201j),\n"
            "            (0.004457435020022989 - 0.01710855793406193j),\n"
            "            (-0.008583139038082853 + 0.0017936797500065001j),\n"
            "            (-0.02504675795686534 + 0.018814246686365037j),\n"
            "            (-0.042648339908966576 + 0.03268602567708891j),\n"
            "            (-0.048011834310870354 + 0.04460730274027931j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 8\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(compress_state_jets(state=state.copy(), k=k), ((3, 8), (3,), (3,)))",
            "gold_call": "_pack(_oracle_compress_state_jets(state=state.copy(), k=k), ((3, 8), (3,), (3,)))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [(1 + 0j), 1j, (-1 + 0j), (-0 - 1j)],\n"
            "        [0.1j, (0.2 + 0j), 0.3j, (0.4 + 0j)],\n"
            "        [(0.2 + 0j), (-0 - 0.1j), (0.1 + 0j), 0.2j],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 2\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(compress_state_jets(state=state.copy(), k=k), ((3, 4), (3,), (3,)))",
            "gold_call": "_pack(_oracle_compress_state_jets(state=state.copy(), k=k), ((3, 4), (3,), (3,)))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.2735051519544164 + 0.45720620153770497j),\n"
            "            (0.4268631256624138 + 0.3790394058029379j),\n"
            "            (0.5097929782785058 + 0.2728058420288081j),\n"
            "            (0.49687094310213686 + 0.1502479329312175j),\n"
            "            (0.4145165077987919 + 0.03181652432707721j),\n"
            "            (0.35072447277188695 - 0.05922640228300827j),\n"
            "            (0.3250707997749054 - 0.11803800163709265j),\n"
            "            (0.15680294390617536 - 0.18156137097635108j),\n"
            "        ],\n"
            "        [\n"
            "            (0.09667904088542927 + 0.028717550711972437j),\n"
            "            (0.064770446866267 + 0.06923533107657157j),\n"
            "            (0.027508891077253253 + 0.10475308853525724j),\n"
            "            (-0.011069217606725096 + 0.1325854761914147j),\n"
            "            (-0.04798225314781576 + 0.1503960227903566j),\n"
            "            (-0.0819262045763191 + 0.1564884637289539j),\n"
            "            (-0.11003805885927362 + 0.15046982770949557j),\n"
            "            (-0.1245745959973257 + 0.133915024891705j),\n"
            "        ],\n"
            "        [\n"
            "            (0.06426911991006659 - 0.13279100344785003j),\n"
            "            (0.05356247183336736 - 0.1080060166337035j),\n"
            "            (0.033657079579839065 - 0.07467106542601022j),\n"
            "            (0.009360613542048278 - 0.03592797166153005j),\n"
            "            (-0.01802459197997399 + 0.0037667274750136505j),\n"
            "            (-0.052598191709417215 + 0.03950991804136658j),\n"
            "            (-0.08956151380882982 + 0.06864065392188672j),\n"
            "            (-0.10082485205282775 + 0.09367533575458656j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 0\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: compress_state_jets(state=state.copy(), k=k))",
            "gold_call": "_raises_value_error(lambda: _oracle_compress_state_jets(state=state.copy(), k=k))",
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
            "    if not isinstance(value, tuple) or len(value) != len(_expected_shapes):\n"
            '        raise AssertionError("return must be the specified tuple")\n'
            "    if any(np.shape(part) != shape for part, shape in zip(value, _expected_shapes)):\n"
            '        raise AssertionError("a returned field has the wrong shape")\n'
            "    return np.concatenate([np.asarray(part).reshape(-1) for part in value])\n"
            "rng=np.random.default_rng(2081)\n"
            "n=4\n"
            "k=7\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "state *= 2.3\n"
            "\n"
            "_expected_shapes=(state.shape,(9,),(9,))\n",
            "call": "_pack(compress_state_jets(state.copy(), k))",
            "gold_call": "_pack(_oracle_compress_state_jets(state.copy(), k))",
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
            "    if not isinstance(value, tuple) or len(value) != len(_expected_shapes):\n"
            '        raise AssertionError("return must be the specified tuple")\n'
            "    if any(np.shape(part) != shape for part, shape in zip(value, _expected_shapes)):\n"
            '        raise AssertionError("a returned field has the wrong shape")\n'
            "    return np.concatenate([np.asarray(part).reshape(-1) for part in value])\n"
            "rng=np.random.default_rng(711)\n"
            "n=3\n"
            "k=3\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "state *= 2.3\n"
            "\n"
            "_expected_shapes=(state.shape,(9,),(9,))\n",
            "call": "_pack(compress_state_jets(state.copy(), k))",
            "gold_call": "_pack(_oracle_compress_state_jets(state.copy(), k))",
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
            "    if not isinstance(value, tuple) or len(value) != len(_expected_shapes):\n"
            '        raise AssertionError("return must be the specified tuple")\n'
            "    if any(np.shape(part) != shape for part, shape in zip(value, _expected_shapes)):\n"
            '        raise AssertionError("a returned field has the wrong shape")\n'
            "    return np.concatenate([np.asarray(part).reshape(-1) for part in value])\n"
            "rng=np.random.default_rng(993)\n"
            "n=4\n"
            "k=16\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "state *= 2.3\n"
            "\n"
            "_expected_shapes=(state.shape,(9,),(9,))\n",
            "call": "_pack(compress_state_jets(state.copy(), k))",
            "gold_call": "_pack(_oracle_compress_state_jets(state.copy(), k))",
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
            "    if not isinstance(value, tuple) or len(value) != len(_expected_shapes):\n"
            '        raise AssertionError("return must be the specified tuple")\n'
            "    if any(np.shape(part) != shape for part, shape in zip(value, _expected_shapes)):\n"
            '        raise AssertionError("a returned field has the wrong shape")\n'
            "    return np.concatenate([np.asarray(part).reshape(-1) for part in value])\n"
            "rng=np.random.default_rng(2081)\n"
            "n=4\n"
            "k=7\n"
            "state = _curve(2**n)[:,:,0]\n"
            "bases = np.stack([_curve(2) for _ in range(n)],axis=1)\n"
            "gate = _curve(4)\n"
            "qubits = np.array([n-1,0],dtype=int)\n"
            "state *= 2.3\n"
            "\n"
            "state=np.zeros((9,4),complex)\n"
            "state[0]=[.5,.5j,-.5,-.5j]\n"
            "state[1]=[.04j,.03,.02j,-.01]\n"
            "state[3]=[.07,-.06j,.05,.08j]\n"
            "state[4]=[.02,.03j,-.04,.01j]\n"
            "state[5]=[.009j,.013,-.017j,.021]\n"
            "state[7]=[.017,.011j,-.023,-.019j]\n"
            "state[8]=[.032,-.014j,.018,.027j]\n"
            "k=2\n"
            "\n"
            "_expected_shapes=(state.shape,(9,),(9,))\n",
            "call": "_pack(compress_state_jets(state.copy(), k))",
            "gold_call": "_pack(_oracle_compress_state_jets(state.copy(), k))",
            "tol": 1e-08,
        },
    ]
