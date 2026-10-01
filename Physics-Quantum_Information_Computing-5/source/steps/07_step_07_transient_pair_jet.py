"""
Differentiate a transient two-qubit refinement with two distinct sparse projections.

Compute the ordered two-qubit reduced-state jet of the current state and its natural eigenframe $V(s)$.

The candidate state is formed by two projections separated by inverse coordinate transformations:



$$

v\;\longmapsto\;V^\dagger v\;\longmapsto\;\frac{P_1V^\dagger v}{\|P_1V^\dagger v\|}

\;\longmapsto\;V\frac{P_1V^\dagger v}{\|P_1V^\dagger v\|}

\;\longmapsto\;\frac{P_2V(P_1V^\dagger v/\|P_1V^\dagger v\|)}{\|P_2V(P_1V^\dagger v/\|P_1V^\dagger v\|)\|}.

$$



Each projector is selected independently from its own incoming base amplitudes and then fixed for differentiation.

Differentiate both occurrences of $V$ and both normalizations through every supplied derivative order.

Accept only if the final candidate's base participation ratio decreases by more than the supplied tolerance.

On acceptance, its retained-mass jet is the product of the two mass jets; on rejection restore the entire input jet and return mass jet with base one and every other row zero.

This transformation is transient and leaves all accumulated single-qubit frames unchanged.

Unit base states and base unitaries use maximum residual tolerance $10^{-9}$; derivative rows are supplied ordinary derivatives, not Taylor coefficients.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The committed state jet, two-projection retained-mass jet and numerical acceptance flag in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transient_pair_jet(
    state: "np.ndarray",
    qubits: "np.ndarray",
    k: int,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    r"""Differentiate a transient two-qubit refinement with two distinct sparse projections.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet, $2 \le N \le 6$.
    qubits : np.ndarray, shape (2,)
        Two distinct ordered integer qubit indices.
    k : int
        Retained budget in $[1,2^N]$.
    gap_tol : float, optional
        Finite real spectral threshold in $[0,10^{-8}]$, default $10^{-10}$.
    accept_tol : float, optional
        Finite real guard tolerance in $[0,10^{-8}]$, default $10^{-12}$.

    Returns
    -------
    result : tuple
        Ordered tuple of committed state jet (J, 2**N), retained-mass jet (J,), and numerical acceptance flag 0 or 1.

    Raises
    ------
    ValueError
        If the state, subsystem indices, budget or either tolerance violates the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transient_pair_jet(
    state: "np.ndarray",
    qubits: "np.ndarray",
    k: int,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    q = _indices(qubits, n, 2)
    k = _integer(k, 1, state.shape[1], "k")
    _controls(1, gap_tol, accept_tol)
    frame = _oracle_natural_frame_jets(
        _oracle_reduced_density_jets(state, np.array(q)), gap_tol
    )
    rotated = _apply_jet(state, frame.conj().swapaxes(-1, -2), q)
    compressed, mass1, _ = _oracle_compress_state_jets(rotated, k)
    undone = _apply_jet(compressed, frame, q)
    candidate, mass2, pr = _oracle_compress_state_jets(undone, k)
    if _pr(state)[0] - pr[0] > accept_tol:
        return candidate, _scale(mass1, mass2), 1
    return state, _one(len(state)), 0

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
            "            (0.031159784872254667 + 0.004213136856928921j),\n"
            "            (0.01760798257977731 + 0.01985786360174461j),\n"
            "            (0.003072410412042814 + 0.034077799074846184j),\n"
            "            (-0.010651842978670773 + 0.045748067013251j),\n"
            "            (-0.022845009655512637 + 0.0537526713082846j),\n"
            "            (-0.034217544453961926 + 0.05716854308581346j),\n"
            "            (-0.044000621319503844 + 0.05578916242695591j),\n"
            "            (-0.046980465724803445 + 0.05067737696964496j),\n"
            "            (-0.04206018000400504 + 0.04359850120178754j),\n"
            "            (-0.038113287726601774 + 0.03476141547590393j),\n"
            "            (-0.030495862629678877 + 0.021994599841826272j),\n"
            "            (-0.015465369145434425 + 0.005466310974737122j),\n"
            "            (-0.006777695775449373 - 0.009156691107610545j),\n"
            "            (0.009736632811153001 - 0.020373828203598034j),\n"
            "            (0.017470106230107675 - 0.03376473176243713j),\n"
            "            (0.030728636820470022 - 0.04420079867167229j),\n"
            "        ],\n"
            "        [\n"
            "            (0.02375224250337461 - 0.027624924150924903j),\n"
            "            (0.02520540574647276 - 0.02322662185272293j),\n"
            "            (0.022285108318846614 - 0.016949949541862106j),\n"
            "            (0.015660576968444414 - 0.009355838811546302j),\n"
            "            (0.006325923288954214 - 0.0011731321514597886j),\n"
            "            (-0.004691371423917081 + 0.006797835081396336j),\n"
            "            (-0.015550107836760222 + 0.013912939622330938j),\n"
            "            (-0.022838786150205993 + 0.0199291856623407j),\n"
            "            (-0.024913021842570608 + 0.024853785138691156j),\n"
            "            (-0.024208267786550802 + 0.028189504763161434j),\n"
            "            (-0.019006978268381554 + 0.028685452723786403j),\n"
            "            (-0.008577193323534026 + 0.026052686212295055j),\n"
            "            (0.001036622968634653 + 0.02211256555644312j),\n"
            "            (0.013402972210492243 + 0.01736718972200239j),\n"
            "            (0.02108859334033557 + 0.009959403242580513j),\n"
            "            (0.027913772293929648 + 0.0019895672881043306j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([3, 1], dtype=int)\n"
            "k = 7\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), ((3, 16), "
            "(3,), ()))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), "
            "((3, 16), (3,), ()))",
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
            "            (-0.0654355482317348 - 0.008847587399550737j),\n"
            "            (-0.03697676341753235 - 0.041701513563663674j),\n"
            "            (-0.006452061865289901 - 0.07156337805717697j),\n"
            "            (0.022368870255208628 - 0.0960709407278271j),\n"
            "            (0.047974520276576546 - 0.11288060974739766j),\n"
            "            (0.07185684335332004 - 0.12005394048020827j),\n"
            "            (0.09240130477095809 - 0.11715724109660743j),\n"
            "            (0.09865897802208723 - 0.10642249163625442j),\n"
            "            (0.08832637800841059 - 0.09155685252375384j),\n"
            "            (0.08003790422586372 - 0.07299897249939827j),\n"
            "            (0.06404131152232564 - 0.046188659667835176j),\n"
            "            (0.032477275205412284 - 0.011479253046947963j),\n"
            "            (0.014233161128443687 + 0.01922905132598214j),\n"
            "            (-0.020446928903421306 + 0.042785039227555874j),\n"
            "            (-0.036687223083226125 + 0.07090593670111796j),\n"
            "            (-0.06453013732298706 + 0.09282167721051181j),\n"
            "        ],\n"
            "        [\n"
            "            (0.10474738943988203 - 0.12182591550557882j),\n"
            "            (0.11115583934194488 - 0.10242940237050813j),\n"
            "            (0.09827732768611357 - 0.0747492774796119j),\n"
            "            (0.06906314443083988 - 0.041259249158919195j),\n"
            "            (0.02789732170428808 - 0.005173512787937678j),\n"
            "            (-0.02068894797947432 + 0.029978452708957836j),\n"
            "            (-0.06857597556011256 + 0.061356063734479424j),\n"
            "            (-0.1007190469224084 + 0.08788770877092249j),\n"
            "            (-0.10986642632573637 + 0.10960519246162799j),\n"
            "            (-0.10675846093868904 + 0.12431571600554195j),\n"
            "            (-0.08382077416356266 + 0.12650284651189803j),\n"
            "            (-0.037825422556785046 + 0.11489234619622121j),\n"
            "            (0.004571507291678817 + 0.09751641410391418j),\n"
            "            (0.05910710744827079 + 0.07658930667403055j),\n"
            "            (0.09300069663087987 + 0.04392096829978007j),\n"
            "            (0.12309973581622977 + 0.008773991740540102j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([0, 2], dtype=int)\n"
            "k = 5\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), ((3, 16), "
            "(3,), ()))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), "
            "((3, 16), (3,), ()))",
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
            "        [\n"
            "            (0.043568320549994585 - 0.019423651151384124j),\n"
            "            (0.008215873501129192 + 0.01543227799989675j),\n"
            "            (-0.02478966180156604 + 0.04923341210166851j),\n"
            "            (-0.05045764088284662 + 0.07910476477628109j),\n"
            "        ],\n"
            "        [\n"
            "            (0.032526644536516935 - 0.06312731337098683j),\n"
            "            (0.03583528414709558 - 0.05714557183925745j),\n"
            "            (0.032326114205456086 - 0.046638596787727755j),\n"
            "            (0.023319917816401258 - 0.032626055556855015j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([1, 0], dtype=int)\n"
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
            "call": "_pack(transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), ((3, 4), (3,), "
            "()))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), "
            "((3, 4), (3,), ()))",
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
            "            (0.031159784872254667 + 0.004213136856928921j),\n"
            "            (0.01760798257977731 + 0.01985786360174461j),\n"
            "            (0.003072410412042814 + 0.034077799074846184j),\n"
            "            (-0.010651842978670773 + 0.045748067013251j),\n"
            "            (-0.022845009655512637 + 0.0537526713082846j),\n"
            "            (-0.034217544453961926 + 0.05716854308581346j),\n"
            "            (-0.044000621319503844 + 0.05578916242695591j),\n"
            "            (-0.046980465724803445 + 0.05067737696964496j),\n"
            "            (-0.04206018000400504 + 0.04359850120178754j),\n"
            "            (-0.038113287726601774 + 0.03476141547590393j),\n"
            "            (-0.030495862629678877 + 0.021994599841826272j),\n"
            "            (-0.015465369145434425 + 0.005466310974737122j),\n"
            "            (-0.006777695775449373 - 0.009156691107610545j),\n"
            "            (0.009736632811153001 - 0.020373828203598034j),\n"
            "            (0.017470106230107675 - 0.03376473176243713j),\n"
            "            (0.030728636820470022 - 0.04420079867167229j),\n"
            "        ],\n"
            "        [\n"
            "            (0.02375224250337461 - 0.027624924150924903j),\n"
            "            (0.02520540574647276 - 0.02322662185272293j),\n"
            "            (0.022285108318846614 - 0.016949949541862106j),\n"
            "            (0.015660576968444414 - 0.009355838811546302j),\n"
            "            (0.006325923288954214 - 0.0011731321514597886j),\n"
            "            (-0.004691371423917081 + 0.006797835081396336j),\n"
            "            (-0.015550107836760222 + 0.013912939622330938j),\n"
            "            (-0.022838786150205993 + 0.0199291856623407j),\n"
            "            (-0.024913021842570608 + 0.024853785138691156j),\n"
            "            (-0.024208267786550802 + 0.028189504763161434j),\n"
            "            (-0.019006978268381554 + 0.028685452723786403j),\n"
            "            (-0.008577193323534026 + 0.026052686212295055j),\n"
            "            (0.001036622968634653 + 0.02211256555644312j),\n"
            "            (0.013402972210492243 + 0.01736718972200239j),\n"
            "            (0.02108859334033557 + 0.009959403242580513j),\n"
            "            (0.027913772293929648 + 0.0019895672881043306j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([3, 1], dtype=int)\n"
            "k = 7\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n"
            "\n"
            "\n"
            "state = np.array(\n"
            "    [\n"
            "        [\n"
            "            (0.3533907779977201 - 0.7637731221214289j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.23194740110322196 + 0.08888435557230812j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.012779744059624391 - 0.02489157084129433j),\n"
            "            0j,\n"
            "            (-0.1045032125241309 + 0.04055061837952851j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.012309459270130253 + 0.0008963454224854408j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.011271302808856196 + 0.29068107065120086j),\n"
            "            0j,\n"
            "            (-0.010585729541413912 + 0.10412855843848907j),\n"
            "            (0.007702855250478965 + 0.017876980090681296j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.243301108767855 + 0.1556349372488301j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.10839392265128162 - 0.14238097196702815j),\n"
            "            (-0.06237645785227027 - 0.03575900917999836j),\n"
            "            0j,\n"
            "            0j,\n"
            "        ],\n"
            "        [\n"
            "            (-0.6597764069516524 - 0.3069625661934311j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.07342658119343581 - 0.06856875908408958j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.017638683375669435 + 0.033294670441382956j),\n"
            "            0j,\n"
            "            (-0.015028319473452935 + 0.030490735523821248j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.0020231271860935616 + 0.010307836847821997j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.05918206917390047 - 0.030471152056391806j),\n"
            "            0j,\n"
            "            (0.05254937904063799 - 0.10550243430917272j),\n"
            "            (-0.00691764818717827 - 0.003988440178263268j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.1009129811667691 + 0.15629808363575348j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.07119466153363156 + 0.028093792696302952j),\n"
            "            (-0.026688811845045495 - 0.008020486348769399j),\n"
            "            0j,\n"
            "            0j,\n"
            "        ],\n"
            "        [\n"
            "            (-0.3696692541826353 + 0.283109992900065j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (-0.05255810006441172 - 0.0980532385584133j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.06828819836002492 + 0.08402428683905108j),\n"
            "            0j,\n"
            "            (0.2112761625015314 + 0.03318904913650171j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.03747154965098286 - 0.011075657540184178j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.06832286110461844 - 0.2546608451529013j),\n"
            "            0j,\n"
            "            (-0.1485701602344807 - 0.16581557757906662j),\n"
            "            (-0.010708977485731297 - 0.005894422699995725j),\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.2843246647767752 - 0.20286733829215153j),\n"
            "            0j,\n"
            "            0j,\n"
            "            0j,\n"
            "            (0.0633289323703872 + 0.0671424453439765j),\n"
            "            (0.07249503641959842 - 0.0232904900870494j),\n"
            "            0j,\n"
            "            0j,\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([1, 2], dtype=int)\n"
            "k = 11\n",
            "call": "_pack(transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), ((3, 32), "
            "(3,), ()))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state=state.copy(), qubits=qubits.copy(), k=k), "
            "((3, 32), (3,), ()))",
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
            "            (0.031159784872254667 + 0.004213136856928921j),\n"
            "            (0.01760798257977731 + 0.01985786360174461j),\n"
            "            (0.003072410412042814 + 0.034077799074846184j),\n"
            "            (-0.010651842978670773 + 0.045748067013251j),\n"
            "            (-0.022845009655512637 + 0.0537526713082846j),\n"
            "            (-0.034217544453961926 + 0.05716854308581346j),\n"
            "            (-0.044000621319503844 + 0.05578916242695591j),\n"
            "            (-0.046980465724803445 + 0.05067737696964496j),\n"
            "            (-0.04206018000400504 + 0.04359850120178754j),\n"
            "            (-0.038113287726601774 + 0.03476141547590393j),\n"
            "            (-0.030495862629678877 + 0.021994599841826272j),\n"
            "            (-0.015465369145434425 + 0.005466310974737122j),\n"
            "            (-0.006777695775449373 - 0.009156691107610545j),\n"
            "            (0.009736632811153001 - 0.020373828203598034j),\n"
            "            (0.017470106230107675 - 0.03376473176243713j),\n"
            "            (0.030728636820470022 - 0.04420079867167229j),\n"
            "        ],\n"
            "        [\n"
            "            (0.02375224250337461 - 0.027624924150924903j),\n"
            "            (0.02520540574647276 - 0.02322662185272293j),\n"
            "            (0.022285108318846614 - 0.016949949541862106j),\n"
            "            (0.015660576968444414 - 0.009355838811546302j),\n"
            "            (0.006325923288954214 - 0.0011731321514597886j),\n"
            "            (-0.004691371423917081 + 0.006797835081396336j),\n"
            "            (-0.015550107836760222 + 0.013912939622330938j),\n"
            "            (-0.022838786150205993 + 0.0199291856623407j),\n"
            "            (-0.024913021842570608 + 0.024853785138691156j),\n"
            "            (-0.024208267786550802 + 0.028189504763161434j),\n"
            "            (-0.019006978268381554 + 0.028685452723786403j),\n"
            "            (-0.008577193323534026 + 0.026052686212295055j),\n"
            "            (0.001036622968634653 + 0.02211256555644312j),\n"
            "            (0.013402972210492243 + 0.01736718972200239j),\n"
            "            (0.02108859334033557 + 0.009959403242580513j),\n"
            "            (0.027913772293929648 + 0.0019895672881043306j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "qubits = np.array([1, 1], dtype=int)\n"
            "k = 7\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: transient_pair_jet(state=state.copy(), "
            "qubits=qubits.copy(), k=k))",
            "gold_call": "_raises_value_error(lambda: _oracle_transient_pair_jet(state=state.copy(), "
            "qubits=qubits.copy(), k=k))",
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
            "\n"
            "_expected_shapes=(state.shape,(9,),())\n",
            "call": "_pack(transient_pair_jet(state.copy(), qubits.copy(), k))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state.copy(), qubits.copy(), k))",
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
            "\n"
            "_expected_shapes=(state.shape,(9,),())\n",
            "call": "_pack(transient_pair_jet(state.copy(), qubits.copy(), k))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state.copy(), qubits.copy(), k))",
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
            "\n"
            "_expected_shapes=(state.shape,(9,),())\n",
            "call": "_pack(transient_pair_jet(state.copy(), qubits.copy(), k))",
            "gold_call": "_pack(_oracle_transient_pair_jet(state.copy(), qubits.copy(), k))",
            "tol": 1e-08,
        },
    ]
