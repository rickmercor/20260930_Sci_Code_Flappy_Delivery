"""
Initialize the deterministic benchmark PCKAN and evaluate its untrained forward map on the six benchmark-scaled input coordinates supplied by Step 02. The step applies tanh coordinatewise, evaluates the second-kind Chebyshev basis through order 3, combines the basis values through a width-8 CKAN hidden layer, and applies the scalar linear readout. Network parameters are initialized deterministically in PyTorch float64 using seed 0, with the CKAN coefficient tensor drawn before the readout weights and the output bias set to zero.



The step returns the untrained scalar predictions together with the initialized CKAN coefficient tensor, readout weights, and output bias. Training-data construction, FVSJ/COS pricing, physics-residual evaluation, parameter optimization, and differential-evolution calibration are excluded from this step.

This step implements the deterministic forward map of the Physics-Informed Chebyshev Kolmogorov–Arnold Network (PCKAN) before any training or physics-informed optimization.



The six benchmark-scaled input coordinates from Step 02 are first transformed coordinatewise using tanh. Each transformed coordinate is then expanded in the second-kind Chebyshev basis through maximum order 3:



U_0(z) = 1,

U_1(z) = 2z,

U_2(z) = 4z^2 - 1,

U_3(z) = 8z^3 - 4z.



For each of the eight hidden coordinates, the CKAN layer sums the trainable Chebyshev-edge contributions over all six input coordinates and all four expansion orders. The resulting eight-dimensional hidden representation is mapped to one scalar prediction by a linear readout.



For this deterministic benchmark, use PyTorch CPU with float64 arithmetic. Set torch.manual_seed(0) immediately before initialization. Draw the CKAN coefficient tensor first with shape (8, 6, 4), followed by the readout vector with shape (8,), using independent sequential torch.randn draws multiplied by 0.01. Set the scalar output bias to exactly zero.



The input array contains the already benchmark-scaled Step-02 features. Do not apply min-max normalization or any additional scaling. The tanh transformation belongs inside the PCKAN forward map and must occur before evaluation of the second-kind Chebyshev basis.



This step performs initialization and forward evaluation only. It must not train the network, compute the Black-Scholes physics residual, or perform differential-evolution calibration.

Returns
-------
return predictions, W_hat, W_out, float(b_out)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initialize_and_evaluate_pckan(
    inputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Initialize the deterministic PCKAN and evaluate its forward map.

    Parameters
    ----------
    inputs : np.ndarray
        Float64 array of shape (n, 6) containing the benchmark-scaled
        PCKAN coordinates before the tanh transformation.

    Returns
    -------
    predictions : np.ndarray
        Float64 array of shape (n,) containing the scalar PCKAN predictions.
    W_hat : np.ndarray
        Float64 CKAN coefficient tensor of shape (8, 6, 4).
    W_out : np.ndarray
        Float64 linear-readout weight vector of shape (8,).
    b_out : float
        Scalar output bias, initialized to exactly 0.0.
    """
    return (
        np.empty((inputs.shape[0],), dtype=np.float64),
        np.empty((8, 6, 4), dtype=np.float64),
        np.empty((8,), dtype=np.float64),
        0.0,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_initialize_and_evaluate_pckan(inputs):
    import numpy as np

    arr = np.asarray(inputs, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("inputs must have shape (n, 6)")

    if not np.all(np.isfinite(arr)):
        raise ValueError("inputs must be finite")

    # Frozen torch.manual_seed(0), float64 initialization.
    # W_hat is drawn first with shape (8, 6, 4), then W_out with shape (8,).
    W_hat = np.array(
        [
            -0.023104118002341766,
            -0.003732508612577643,
            -0.010608166785462862,
            0.009995093547811761,
            -0.008840250045832653,
            -0.012755469302148053,
            -0.00623224579866079,
            -0.008664416019125773,
            -0.012956271403277857,
            0.015236316231063501,
            0.0032366056649837383,
            0.020177260314861084,
            0.011357423400213507,
            -0.01226881339001083,
            0.0007138847120511694,
            0.003380174030507378,
            0.0015351883531544036,
            -0.006332746088675096,
            -0.012609246557194385,
            -0.00726951062065411,
            -0.00019964750907762652,
            0.002102998901016947,
            0.0017718935139366585,
            -0.00830510051008449,
            0.010111892074660187,
            -0.002426793824588577,
            -0.007730112772958181,
            -0.01595181488078307,
            -0.006870036940435463,
            0.014880743158106724,
            -0.004484158047114511,
            -0.008910030351707376,
            -0.0009174174090816388,
            0.005563227724483105,
            -9.446277892507904e-05,
            -0.012176877791068701,
            -0.012654653080284904,
            -0.011195351830271254,
            0.011664857205254855,
            0.009262324572968777,
            0.006539244967350809,
            0.0014821544548810084,
            -0.011461092231976014,
            0.018727763768553417,
            -0.0030991497424174676,
            0.0024018023403782536,
            -0.013487610270963771,
            0.002443990933828689,
            -0.03145296273123133,
            -0.0011373127757597257,
            0.01696202480514395,
            -0.006652034805381666,
            -0.00587222598359879,
            0.02832104167511146,
            0.009039973689601122,
            0.00947836532432321,
            -0.013809366518587675,
            -0.020547510174380234,
            0.0004757655385024031,
            -0.0038556879759337023,
            -0.0027503491919216158,
            0.00886589630091246,
            0.014857416299332184,
            -7.855617400547969e-05,
            -0.011287986397999446,
            -0.0009376858138313218,
            0.00998343579451126,
            -0.0008262167465143246,
            0.0018253194918016991,
            0.009509038124935927,
            -0.017631617835191397,
            0.010107316682629281,
            0.007494883300195011,
            -0.00560555214098722,
            0.0007424681473465156,
            0.019936457271364453,
            0.004498653398445938,
            0.015531453898519594,
            0.0005398594882285299,
            0.014744415042312725,
            -0.007012303945657874,
            0.00326255796287562,
            -0.008738640219900763,
            0.003941269998738486,
            0.008915170821381017,
            0.011731555158319225,
            -0.00394903405651056,
            0.004464225721184322,
            -0.001726536808458421,
            0.009751203451810781,
            0.004164541966117019,
            -0.004986513835421579,
            -0.0047546981991056095,
            0.0028634406331767815,
            -0.013181871188381728,
            0.008887144617352294,
            -0.005918823592164964,
            0.006195092629710962,
            -0.014872950971304264,
            0.013626823808042957,
            0.004188191254194787,
            0.003434148780975708,
            -0.003485998065004807,
            -0.012090699446332277,
            0.0020710026569896174,
            -0.0011727393443191467,
            -0.00017436740784596905,
            -0.0024317130976865137,
            -0.004394223636068548,
            0.02714466016601841,
            -0.014631736798004056,
            0.004357325629861231,
            -0.005312331855486973,
            -0.0013408449243234942,
            0.0136427245242356,
            0.002327705351743837,
            0.011064458833206278,
            0.008527354795918417,
            -0.006301719953003105,
            -0.006019872001874334,
            0.011149190519421119,
            -0.0055598980336551964,
            0.005331687087503058,
            -0.0018787474948239727,
            0.003993685890598303,
            -0.006760874461803967,
            0.01662168703751691,
            -0.0060368393719886625,
            0.000534191942795709,
            0.0001342750276927035,
            -0.015406177939388192,
            -0.004819087554044004,
            0.007623209434576803,
            0.0021679229361763427,
            -0.00478912061625386,
            -0.002578760451372613,
            0.002631423185835337,
            -0.004053288789025133,
            -0.0007271235379509131,
            -0.0013985617336865128,
            0.00785995459350369,
            -0.020568571650881627,
            -0.006711099455038105,
            -0.0040096823326862124,
            0.0009346980467841958,
            -0.0064879924513363985,
            -0.00679095884649028,
            -0.0008377947239717037,
            -0.013083177047571642,
            -0.0141739733683466,
            0.003237499278733888,
            -0.012819324564954783,
            -0.004117908858856197,
            -0.011128682991908617,
            -0.003920249549198018,
            -0.004522967443547727,
            0.010930261889419516,
            -0.001639182226412006,
            -0.0020150312169022844,
            -0.018609565968157826,
            0.008942415239486997,
            -0.019664187911247126,
            0.014864847606394494,
            -0.001982554880893235,
            -0.016795171874434075,
            0.013863200736888246,
            -0.030465827500791125,
            -0.00598281249533576,
            0.0011680004036673396,
            0.00029905566085561,
            0.008536195134028616,
            0.009084886688785414,
            -0.011090488038575444,
            -0.00624538324560581,
            0.00136098548491325,
            -0.004218880327975954,
            -0.0010303497974091277,
            -0.012868901361560705,
            0.014145046448860667,
            0.004472847452382745,
            -0.002069050342565235,
            0.011506583648125866,
            -0.005611667183722734,
            0.00042508018555383173,
            0.008043963540244156,
            0.009072091619347682,
            -0.0021521353161099454,
            0.005114427131934867,
            0.00807731564311178,
            -0.013327491869404214,
            -0.002090149318312048,
            -0.006750742601943351,
        ],
        dtype=np.float64,
    ).reshape(8, 6, 4)

    W_out = np.array(
        [
            -0.0017502054629307568,
            0.0035290023830520153,
            -0.008974748047588913,
            -0.00652325833004202,
            -0.0006868076558889425,
            -0.022581728431474914,
            0.008445572500753966,
            -0.01264403375116388,
        ],
        dtype=np.float64,
    )

    b_out = 0.0

    # Second-kind Chebyshev basis U_0,...,U_3 after tanh.
    z = np.tanh(arr)

    U = np.stack(
        (
            np.ones_like(z),
            2.0 * z,
            4.0 * z**2 - 1.0,
            8.0 * z**3 - 4.0 * z,
        ),
        axis=-1,
    )

    h = np.einsum("qpn,bpn->bq", W_hat, U)
    predictions = np.einsum("bq,q->b", h, W_out) + b_out

    return (
        predictions.astype(np.float64, copy=False),
        W_hat.astype(np.float64, copy=False),
        W_out.astype(np.float64, copy=False),
        float(b_out),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # CASE 1 — structure and dtype
        {
            "setup": (
                "import numpy as np\n"
                "X = np.zeros((3, 6), dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "result = (\n"
                "    preds.shape,\n"
                "    W_hat.shape,\n"
                "    W_out.shape,\n"
                "    float(b_out),\n"
                "    preds.dtype == np.float64,\n"
                "    W_hat.dtype == np.float64,\n"
                "    W_out.dtype == np.float64,\n"
                ")"
            ),
            "call": "result",
            "gold_call": "((3,), (8, 6, 4), (8,), 0.0, True, True, True)",
        },

        # CASE 2 — deterministic re-initialization
        {
            "setup": (
                "import numpy as np\n"
                "X = np.array([[1.0, 0.9, 0.25, 0.05, "
                "0.2, 0.16666666666666669]], dtype=np.float64)\n"
                "a = initialize_and_evaluate_pckan(X)\n"
                "b = initialize_and_evaluate_pckan(X)\n"
                "result = (\n"
                "    bool(np.array_equal(a[0], b[0])),\n"
                "    bool(np.array_equal(a[1], b[1])),\n"
                "    bool(np.array_equal(a[2], b[2])),\n"
                "    float(a[3]) == float(b[3]),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True, True)",
        },

        # CASE 3 — frozen initialization entries
        {
            "setup": (
                "import numpy as np\n"
                "X = np.zeros((1, 6), dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "result = (\n"
                "    float(W_hat[0, 0, 0]),\n"
                "    float(W_hat[7, 5, 3]),\n"
                "    float(W_out[0]),\n"
                "    float(W_out[7]),\n"
                "    float(b_out),\n"
                ")"
            ),
            "call": "result",
            "gold_call": (
                "(-0.023104118002341766, "
                "-0.006750742601943351, "
                "-0.0017502054629307568, "
                "-0.01264403375116388, 0.0)"
            ),
        },

        # CASE 4 — second-kind U_n reconstruction matches;
        # first-kind T_n does not
        {
            "setup": (
                "import numpy as np\n"
                "X = np.full((1, 6), 0.5, dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "z = np.tanh(X)\n"
                "U_u = np.stack([\n"
                "    np.ones_like(z),\n"
                "    2.0 * z,\n"
                "    4.0 * z**2 - 1.0,\n"
                "    8.0 * z**3 - 4.0 * z,\n"
                "], axis=-1)\n"
                "U_t = np.stack([\n"
                "    np.ones_like(z),\n"
                "    z,\n"
                "    2.0 * z**2 - 1.0,\n"
                "    4.0 * z**3 - 3.0 * z,\n"
                "], axis=-1)\n"
                "h_u = np.einsum('qpn,bpn->bq', W_hat, U_u)\n"
                "h_t = np.einsum('qpn,bpn->bq', W_hat, U_t)\n"
                "y_u = np.einsum('bq,q->b', h_u, W_out) + b_out\n"
                "y_t = np.einsum('bq,q->b', h_t, W_out) + b_out\n"
                "result = (\n"
                "    bool(np.allclose(preds, y_u, rtol=0.0, atol=1e-14)),\n"
                "    bool(np.allclose(preds, y_t, rtol=0.0, atol=1e-14)),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, False)",
        },

        # CASE 5 — tanh placement: basis uses tanh(x), not raw x
        {
            "setup": (
                "import numpy as np\n"
                "X = np.ones((1, 6), dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "z = np.tanh(X)\n"
                "U_tanh = np.stack([\n"
                "    np.ones_like(z),\n"
                "    2.0 * z,\n"
                "    4.0 * z**2 - 1.0,\n"
                "    8.0 * z**3 - 4.0 * z,\n"
                "], axis=-1)\n"
                "U_raw = np.stack([\n"
                "    np.ones_like(X),\n"
                "    2.0 * X,\n"
                "    4.0 * X**2 - 1.0,\n"
                "    8.0 * X**3 - 4.0 * X,\n"
                "], axis=-1)\n"
                "y_tanh = np.einsum(\n"
                "    'bq,q->b',\n"
                "    np.einsum('qpn,bpn->bq', W_hat, U_tanh),\n"
                "    W_out,\n"
                ") + b_out\n"
                "y_raw = np.einsum(\n"
                "    'bq,q->b',\n"
                "    np.einsum('qpn,bpn->bq', W_hat, U_raw),\n"
                "    W_out,\n"
                ") + b_out\n"
                "result = (\n"
                "    bool(np.allclose(preds, y_tanh, rtol=0.0, atol=1e-14)),\n"
                "    bool(np.allclose(preds, y_raw, rtol=0.0, atol=1e-14)),\n"
                "    float(np.tanh(1.0)),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, False, 0.7615941559557649)",
        },

        # CASE 6 — first Step-02 feature row prediction
        {
            "setup": (
                "import numpy as np\n"
                "X = np.array([[1.0, 0.9, 0.25, 0.05, "
                "0.2, 0.16666666666666669]], dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "result = float(preds[0])"
            ),
            "call": "result",
            "gold_call": "-0.0017843316853409753",
        },

        # CASE 7 — identical zero rows share the same prediction.
        # Round regression scalar to avoid sub-ULP torch/NumPy difference.
        {
            "setup": (
                "import numpy as np\n"
                "X = np.zeros((4, 6), dtype=np.float64)\n"
                "preds, W_hat, W_out, b_out = initialize_and_evaluate_pckan(X)\n"
                "result = (\n"
                "    bool(np.allclose(preds, preds[0], rtol=0.0, atol=0.0)),\n"
                "    round(float(preds[0]), 15),\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, 0.000100584358599)",
        },
    ]
