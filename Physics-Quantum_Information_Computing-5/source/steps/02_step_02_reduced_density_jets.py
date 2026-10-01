"""
Compute local reduced-density responses including mixed orders.

For a chosen ordered subsystem, let $C(s)$ contain its amplitudes by local configuration and complementary configuration.

The partial trace is



$$

\rho(s)=C(s)C(s)^\dagger.

$$



Return its value and supplied ordinary partial derivatives, retaining the mixed first-derivative contribution and complex conjugation.

For one qubit the local index is its bit; for two ordered qubits it is $2b_{q_1}+b_{q_2}$.

All supplied input rows use the same global array coordinates; no basis rotation or support selection is performed.

Unit base states and base unitaries use maximum residual tolerance $10^{-9}$; derivative rows are supplied ordinary derivatives, not Taylor coefficients.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The reduced matrix and all supplied ordinary partial derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduced_density_jets(state: "np.ndarray", qubits: "np.ndarray") -> "np.ndarray":
    r"""Compute local reduced-density responses including mixed orders.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet with $1 \le N \le 6$.
    qubits : np.ndarray, shape (L,)
        One or two distinct integer qubit indices in their local order.

    Returns
    -------
    rdm : np.ndarray, shape (J, 2**L, 2**L)
        The reduced matrix and its supplied ordinary partial derivatives.

    Raises
    ------
    ValueError
        If the state shape, base normalization, finiteness or subsystem indices are invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_density_jets(
    state: "np.ndarray", qubits: "np.ndarray"
) -> "np.ndarray":
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    q = _indices(qubits, n)
    axes = [n - 1 - j for j in q]
    order = axes + [j for j in range(n) if j not in axes]
    blocks = np.array(
        [v.reshape((2,) * n).transpose(order).reshape(2 ** len(q), -1) for v in state]
    )
    return _matmul(blocks, blocks.conj().swapaxes(-1, -2))

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
            "qubits = np.array([2], dtype=int)\n",
            "call": "reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
            "gold_call": "_oracle_reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.12884541506963756 + 0.2153850572414665j),\n"
            "            (0.20109075171301716 + 0.17856149772479532j),\n"
            "            (0.2401581374848401 + 0.12851597748141133j),\n"
            "            (0.23407070193217744 + 0.07078022897756676j),\n"
            "            (0.19527438923510046 + 0.014988431675607747j),\n"
            "            (0.1652226290673494 - 0.027900938357853514j),\n"
            "            (0.1531374521639625 - 0.05560646739647707j),\n"
            "            (0.07386822605484025 - 0.08553166197015274j),\n"
            "            (-0.08141364821997293 - 0.14362830985826544j),\n"
            "            (-0.1284801433018531 - 0.21955767676038454j),\n"
            "            (-0.16404583290747524 - 0.2511344954152934j),\n"
            "            (-0.2748047062810264 - 0.22081443920860172j),\n"
            "            (-0.20178002558462418 - 0.22088754743078068j),\n"
            "            (-0.25427035615766924 - 0.26023903068176496j),\n"
            "            (-0.12421997488076805 - 0.2071537564602983j),\n"
            "            (-0.1272805030045476 - 0.14930535157170446j),\n"
            "        ],\n"
            "        [\n"
            "            (-0.062319569744509334 - 0.008426273713857842j),\n"
            "            (-0.03521596515955462 - 0.03971572720348922j),\n"
            "            (-0.006144820824085628 - 0.06815559814969237j),\n"
            "            (0.021303685957341546 - 0.091496134026502j),\n"
            "            (0.04569001931102527 - 0.1075053426165692j),\n"
            "            (0.06843508890792385 - 0.11433708617162693j),\n"
            "            (0.08800124263900769 - 0.11157832485391182j),\n"
            "            (0.09396093144960689 - 0.10135475393928992j),\n"
            "            (0.08412036000801008 - 0.08719700240357509j),\n"
            "            (0.07622657545320355 - 0.06952283095180786j),\n"
            "            (0.060991725259357754 - 0.043989199683652544j),\n"
            "            (0.03093073829086885 - 0.010932621949474244j),\n"
            "            (0.013555391550898747 + 0.01831338221522109j),\n"
            "            (-0.019473265622306002 + 0.04074765640719607j),\n"
            "            (-0.03494021246021535 + 0.06752946352487425j),\n"
            "            (-0.061457273640940044 + 0.08840159734334457j),\n"
            "        ],\n"
            "        [\n"
            "            (0.09500897001349844 - 0.11049969660369961j),\n"
            "            (0.10082162298589104 - 0.09290648741089172j),\n"
            "            (0.08914043327538645 - 0.06779979816744842j),\n"
            "            (0.06264230787377766 - 0.03742335524618521j),\n"
            "            (0.025303693155816855 - 0.0046925286058391545j),\n"
            "            (-0.018765485695668323 + 0.027191340325585345j),\n"
            "            (-0.06220043134704089 + 0.05565175848932375j),\n"
            "            (-0.09135514460082397 + 0.0797167426493628j),\n"
            "            (-0.09965208737028243 + 0.09941514055476462j),\n"
            "            (-0.09683307114620321 + 0.11275801905264574j),\n"
            "            (-0.07602791307352622 + 0.11474181089514561j),\n"
            "            (-0.034308773294136105 + 0.10421074484918022j),\n"
            "            (0.004146491874538612 + 0.08845026222577249j),\n"
            "            (0.05361188884196897 + 0.06946875888800956j),\n"
            "            (0.08435437336134229 + 0.03983761297032205j),\n"
            "            (0.11165508917571859 + 0.007958269152417322j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([3, 0], dtype=int)\n",
            "call": "reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
            "gold_call": "_oracle_reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.24828297138469638 + 0.4150434221882991j),\n"
            "            (0.38749853323306593 + 0.34408503559131415j),\n"
            "            (0.4627807356957453 + 0.24764815063266257j),\n"
            "            (0.45105034865549976 + 0.1363923237496766j),\n"
            "        ],\n"
            "        [0j, 0j, 0j, 0j],\n"
            "        [0j, 0j, 0j, 0j],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([1, 0], dtype=int)\n",
            "call": "reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
            "gold_call": "_oracle_reduced_density_jets(state=state.copy(), qubits=qubits.copy())",
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
            "qubits = np.array([3], dtype=int)\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: reduced_density_jets(state=state.copy(), "
            "qubits=qubits.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_reduced_density_jets(state=state.copy(), "
            "qubits=qubits.copy()))",
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
            "\n"
            "_expected_shape=(9,2**len(qubits),2**len(qubits))\n",
            "call": "_pack(reduced_density_jets(state.copy(), qubits.copy()))",
            "gold_call": "_pack(_oracle_reduced_density_jets(state.copy(), qubits.copy()))",
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
            "qubits = np.array([1],dtype=int)\n"
            "\n"
            "_expected_shape=(9,2**len(qubits),2**len(qubits))\n",
            "call": "_pack(reduced_density_jets(state.copy(), qubits.copy()))",
            "gold_call": "_pack(_oracle_reduced_density_jets(state.copy(), qubits.copy()))",
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
            "\n"
            "_expected_shape=(9,2**len(qubits),2**len(qubits))\n",
            "call": "_pack(reduced_density_jets(state.copy(), qubits.copy()))",
            "gold_call": "_pack(_oracle_reduced_density_jets(state.copy(), qubits.copy()))",
            "tol": 1e-08,
        },
    ]
