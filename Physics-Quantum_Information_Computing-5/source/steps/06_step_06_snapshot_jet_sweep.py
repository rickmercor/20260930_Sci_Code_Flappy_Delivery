"""
Propagate a state response through finite passes of snapshot-based local coordinate refinement.

At the start of each pass compute every single-qubit reduced-state jet and corresponding natural-frame jet from the same current state jet.

Keep this complete snapshot fixed while visiting qubits in increasing order; each trial acts on the latest committed state, which may include earlier accepted updates.

Accepted trials change both the state and accumulated frame derivatives, while rejected trials leave both unchanged.

The committed retained-mass response is the derivative jet of



$$

\Gamma(s)=\prod_{\ell\in\mathcal A}m_\ell(s),

$$



where $\mathcal A$ contains accepted trials only.

Stop after a pass with no accepted trial or after the supplied pass limit.

The base decisions and pass count are fixed for the local continuation; an eigengap at or below the threshold gives the constant identity proposal.

Unit base states and base unitaries use maximum residual tolerance $10^{-9}$; derivative rows are supplied ordinary derivatives, not Taylor coefficients.



A jet has J=3 rows for one real perturbation s, ordered (0,1,2), or J=9 rows for two real perturbations (s,u). For J=9, row 3*a+b is the ordinary partial derivative of orders (a,b), with 0 <= a,b <= 2; in particular rows 4,5,7,8 are (1,1),(1,2),(2,1),(2,2). The rectangular jet includes total orders three and four. All jet-valued arguments in one call use the same J. These are derivatives, not factorial-divided Taylor coefficients. All discrete decisions are fixed at (0,0); every continuous dependence is differentiated.

Returns
-------
The final state jet, frame jets, committed-mass jet and total acceptance count in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def snapshot_jet_sweep(
    state: "np.ndarray",
    bases: "np.ndarray",
    k: int,
    max_passes: int = 3,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    r"""Propagate a state response through finite passes of snapshot-based local coordinate refinement.

    Parameters
    ----------
    state : np.ndarray, shape (J, 2**N)
        Finite normalized-state jet, $1 \le N \le 6$.
    bases : np.ndarray, shape (J, N, 2, 2)
        Local physical-frame jets.
    k : int
        Retained budget in $[1,2^N]$.
    max_passes : int, optional
        One to three passes, default three.
    gap_tol : float, optional
        Finite real spectral threshold in $[0,10^{-8}]$, default $10^{-10}$.
    accept_tol : float, optional
        Finite real guard tolerance in $[0,10^{-8}]$, default $10^{-12}$.

    Returns
    -------
    result : tuple
        Ordered tuple of state jet (J, 2**N), frame jets (J, N, 2, 2), committed-mass jet (J,), and total number of accepted trials as an integer.

    Raises
    ------
    ValueError
        If any stated shape, base state/frame condition, budget, pass count, finiteness or tolerance is invalid.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_snapshot_jet_sweep(
    state: "np.ndarray",
    bases: "np.ndarray",
    k: int,
    max_passes: int = 3,
    gap_tol: float = 1e-10,
    accept_tol: float = 1e-12,
) -> tuple:
    state = _state(state)
    n = state.shape[1].bit_length() - 1
    bases = _unitary_jet(bases, (len(state), n, 2, 2), "bases")
    k = _integer(k, 1, state.shape[1], "k")
    max_passes, gap_tol, accept_tol = _controls(max_passes, gap_tol, accept_tol)
    retained, total = _one(len(state)), 0
    for _ in range(max_passes):
        frames = [
            _oracle_natural_frame_jets(
                _oracle_reduced_density_jets(state, np.array([j])), gap_tol
            )
            for j in range(n)
        ]
        count = 0
        for j, frame in enumerate(frames):
            state, bases, mass, accepted = _oracle_guarded_single_jet(
                state, bases, frame, j, k, accept_tol
            )
            retained = _scale(retained, mass)
            count += accepted
        total += count
        if count == 0:
            break
    return state, bases, retained, total

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
            "bases = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                [(0.9895059058723947 + 0j), (-0.1444924297105264 + 0j)],\n"
            "                [(0.1444924297105264 + 0j), (0.9895059058723947 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9582438755126972 + 0j), (-0.28595222510483553 + 0j)],\n"
            "                [(0.28595222510483553 + 0j), (0.9582438755126972 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9068700422993367 + 0j), (-0.421410401366648 + 0j)],\n"
            "                [(0.421410401366648 + 0j), (0.9068700422993367 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.010114470079736848 + 0j), (-0.06926541341106764 + 0j)],\n"
            "                [(0.06926541341106764 + 0j), (-0.010114470079736848 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.030024983636007734 + 0j), (-0.10061560692883321 + 0j)],\n"
            "                [(0.10061560692883321 + 0j), (-0.030024983636007734 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.058997456191330726 + 0j), (-0.12696180592190714 + 0j)],\n"
            "                [(0.12696180592190714 + 0j), (-0.058997456191330726 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.004848578938774735 + 0j), (0.0007080129055815794 - 0j)],\n"
            "                [(-0.0007080129055815794 + 0j), (-0.004848578938774735 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.010564638727527487 + 0j), (0.0031526232817808125 - 0j)],\n"
            "                [(-0.0031526232817808125 + 0j), (-0.010564638727527487 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.017774652829067 + 0j), (0.008259643866786301 - 0j)],\n"
            "                [(-0.008259643866786301 + 0j), (-0.017774652829067 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
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
            "call": "_pack(snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k), ((3, 8), (3, 3, "
            "2, 2), (3,), ()))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k), "
            "((3, 8), (3, 3, 2, 2), (3,), ()))",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "state = np.array(\n"
            "    [[(1 + 0j), 0j, 0j, 0j], [0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j]], dtype=complex\n"
            ")\n"
            "bases = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                [(0.9895059058723947 + 0j), (-0.1444924297105264 + 0j)],\n"
            "                [(0.1444924297105264 + 0j), (0.9895059058723947 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9582438755126972 + 0j), (-0.28595222510483553 + 0j)],\n"
            "                [(0.28595222510483553 + 0j), (0.9582438755126972 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.010114470079736848 + 0j), (-0.06926541341106764 + 0j)],\n"
            "                [(0.06926541341106764 + 0j), (-0.010114470079736848 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.030024983636007734 + 0j), (-0.10061560692883321 + 0j)],\n"
            "                [(0.10061560692883321 + 0j), (-0.030024983636007734 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.004848578938774735 + 0j), (0.0007080129055815794 - 0j)],\n"
            "                [(-0.0007080129055815794 + 0j), (-0.004848578938774735 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.010564638727527487 + 0j), (0.0031526232817808125 - 0j)],\n"
            "                [(-0.0031526232817808125 + 0j), (-0.010564638727527487 + 0j)],\n"
            "            ],\n"
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
            "call": "_pack(snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k), ((3, 4), (3, 2, "
            "2, 2), (3,), ()))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k), "
            "((3, 4), (3, 2, 2, 2), (3,), ()))",
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
            "            (-0.04050772033393107 - 0.005477077914007601j),\n"
            "            (-0.022890377353710502 - 0.02581522268226799j),\n"
            "            (-0.003994133535655661 - 0.044301138797300035j),\n"
            "            (0.013847395872272003 - 0.059472487117226296j),\n"
            "            (0.029698512552166427 - 0.06987847270077j),\n"
            "            (0.0444828077901505 - 0.0743191060115575j),\n"
            "            (0.057200807715355004 - 0.0725259111550427j),\n"
            "            (0.06107460544224448 - 0.06588059006053845j),\n"
            "            (0.05467823400520656 - 0.056678051562323806j),\n"
            "            (0.0495472740445823 - 0.04518984011867511j),\n"
            "            (0.03964462141858254 - 0.02859297979437415j),\n"
            "            (0.020104979889064758 - 0.007106204267158257j),\n"
            "            (0.008811004508084188 + 0.011903698439893712j),\n"
            "            (-0.0126576226544989 + 0.026485976664677446j),\n"
            "            (-0.02271113809913998 + 0.04389415129116826j),\n"
            "            (-0.039947227866611035 + 0.057461038273173975j),\n"
            "        ],\n"
            "        [\n"
            "            (0.04014128983070309 - 0.0466861218150631j),\n"
            "            (0.04259713571153896 - 0.03925299093110176j),\n"
            "            (0.03766183305885078 - 0.028645414725746964j),\n"
            "            (0.026466375076671066 - 0.015811367591513248j),\n"
            "            (0.010690810358332619 - 0.001982593335967041j),\n"
            "            (-0.007928417706419869 + 0.011488341287559814j),\n"
            "            (-0.026279682244124775 + 0.023512867961739292j),\n"
            "            (-0.03859754859384814 + 0.03368032376935579j),\n"
            "            (-0.04210300691394433 + 0.04200289688438806j),\n"
            "            (-0.040911972559270864 + 0.04764026304974284j),\n"
            "            (-0.03212179327356483 + 0.04847841510319902j),\n"
            "            (-0.014495456716772496 + 0.044029039698778645j),\n"
            "            (0.0017518928169925644 + 0.03737023579038888j),\n"
            "            (0.022651023035731896 + 0.029350550630184047j),\n"
            "            (0.035639722745167114 + 0.01683139147996107j),\n"
            "            (0.04717427517674111 + 0.0033623687168963193j),\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "bases = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                [(0.9895059058723947 + 0j), (-0.1444924297105264 + 0j)],\n"
            "                [(0.1444924297105264 + 0j), (0.9895059058723947 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9582438755126972 + 0j), (-0.28595222510483553 + 0j)],\n"
            "                [(0.28595222510483553 + 0j), (0.9582438755126972 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9068700422993367 + 0j), (-0.421410401366648 + 0j)],\n"
            "                [(0.421410401366648 + 0j), (0.9068700422993367 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.8364626499151869 + 0j), (-0.5480239367918736 + 0j)],\n"
            "                [(0.5480239367918736 + 0j), (0.8364626499151869 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(0.013148811103657903 + 0j), (0.09004503743438794 + 0j)],\n"
            "                [(-0.09004503743438794 + 0j), (0.013148811103657903 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.03903247872681005 + 0j), (0.13080028900748317 + 0j)],\n"
            "                [(-0.13080028900748317 + 0j), (0.03903247872681005 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.07669669304872995 + 0j), (0.1650503476984793 + 0j)],\n"
            "                [(-0.1650503476984793 + 0j), (0.07669669304872995 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.12467544562015126 + 0j), (0.19029525285570506 + 0j)],\n"
            "                [(-0.19029525285570506 + 0j), (0.12467544562015126 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.008194098406529302 + 0j), (0.0011965418104328694 - 0j)],\n"
            "                [(-0.0011965418104328694 + 0j), (-0.008194098406529302 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.017854239449521456 + 0j), (0.005327933346209573 - 0j)],\n"
            "                [(-0.005327933346209573 + 0j), (-0.017854239449521456 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.030039163281123233 + 0j), (0.01395879813486885 - 0j)],\n"
            "                [(-0.01395879813486885 + 0j), (-0.030039163281123233 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.04329217002467291 + 0j), (0.028363663878584414 - 0j)],\n"
            "                [(-0.028363663878584414 + 0j), (-0.04329217002467291 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 7\n"
            "max_passes = 2\n"
            "\n"
            "\n"
            "def _pack(result, shapes):\n"
            "    if not isinstance(result, tuple) or len(result) != len(shapes):\n"
            '        raise ValueError("wrong numerical tuple structure")\n'
            "    arrays = [np.asarray(value) for value in result]\n"
            "    if any(value.shape != shape for value, shape in zip(arrays, shapes)):\n"
            '        raise ValueError("wrong numerical return shape")\n'
            "    return np.concatenate([value.astype(complex).reshape(-1) for value in arrays])\n",
            "call": "_pack(snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k, "
            "max_passes=max_passes), ((3, 16), (3, 4, 2, 2), (3,), ()))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), k=k, "
            "max_passes=max_passes), ((3, 16), (3, 4, 2, 2), (3,), ()))",
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
            "bases = np.array(\n"
            "    [\n"
            "        [\n"
            "            [\n"
            "                [(0.9895059058723947 + 0j), (-0.1444924297105264 + 0j)],\n"
            "                [(0.1444924297105264 + 0j), (0.9895059058723947 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9582438755126972 + 0j), (-0.28595222510483553 + 0j)],\n"
            "                [(0.28595222510483553 + 0j), (0.9582438755126972 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(0.9068700422993367 + 0j), (-0.421410401366648 + 0j)],\n"
            "                [(0.421410401366648 + 0j), (0.9068700422993367 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.010114470079736848 + 0j), (-0.06926541341106764 + 0j)],\n"
            "                [(0.06926541341106764 + 0j), (-0.010114470079736848 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.030024983636007734 + 0j), (-0.10061560692883321 + 0j)],\n"
            "                [(0.10061560692883321 + 0j), (-0.030024983636007734 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.058997456191330726 + 0j), (-0.12696180592190714 + 0j)],\n"
            "                [(0.12696180592190714 + 0j), (-0.058997456191330726 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "        [\n"
            "            [\n"
            "                [(-0.004848578938774735 + 0j), (0.0007080129055815794 - 0j)],\n"
            "                [(-0.0007080129055815794 + 0j), (-0.004848578938774735 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.010564638727527487 + 0j), (0.0031526232817808125 - 0j)],\n"
            "                [(-0.0031526232817808125 + 0j), (-0.010564638727527487 + 0j)],\n"
            "            ],\n"
            "            [\n"
            "                [(-0.017774652829067 + 0j), (0.008259643866786301 - 0j)],\n"
            "                [(-0.008259643866786301 + 0j), (-0.017774652829067 + 0j)],\n"
            "            ],\n"
            "        ],\n"
            "    ],\n"
            "    dtype=complex,\n"
            ")\n"
            "k = 5\n"
            "max_passes = 0\n"
            "\n"
            "\n"
            "def _raises_value_error(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises_value_error(lambda: snapshot_jet_sweep(state=state.copy(), bases=bases.copy(), "
            "k=k, max_passes=max_passes))",
            "gold_call": "_raises_value_error(lambda: _oracle_snapshot_jet_sweep(state=state.copy(), "
            "bases=bases.copy(), k=k, max_passes=max_passes))",
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
            "_expected_shapes=(state.shape,bases.shape,(9,),())\n",
            "call": "_pack(snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
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
            "_expected_shapes=(state.shape,bases.shape,(9,),())\n",
            "call": "_pack(snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
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
            "_expected_shapes=(state.shape,bases.shape,(9,),())\n",
            "call": "_pack(snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
            "gold_call": "_pack(_oracle_snapshot_jet_sweep(state.copy(), bases.copy(), k, 2))",
            "tol": 1e-08,
        },
    ]
