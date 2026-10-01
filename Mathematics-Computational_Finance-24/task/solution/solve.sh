#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_heston_cos_put_prices(
    spot,
    strikes,
    maturities,
    risk_free_rate,
    kappa,
    theta,
    volatility_of_variance,
    correlation,
    initial_variance,
    num_cos_terms=128,
    truncation_width=10.0,
):
    import math as _math
    import cmath as _cmath
    import numpy as _np

    def _to_1d(x, name):
        arr = _np.asarray(x, dtype=_np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1-D array")
        return arr

    K = _to_1d(strikes, "strikes")
    T = _to_1d(maturities, "maturities")

    scalars = {
        "spot": spot,
        "risk_free_rate": risk_free_rate,
        "kappa": kappa,
        "theta": theta,
        "volatility_of_variance": volatility_of_variance,
        "correlation": correlation,
        "initial_variance": initial_variance,
        "truncation_width": truncation_width,
    }

    for name, val in scalars.items():
        if not _np.isfinite(val):
            raise ValueError(f"{name} must be finite")

    if not _np.all(_np.isfinite(K)):
        raise ValueError("strikes must be finite")
    if not _np.all(_np.isfinite(T)):
        raise ValueError("maturities must be finite")
    if spot <= 0:
        raise ValueError("spot must be > 0")
    if _np.any(K <= 0):
        raise ValueError("strikes must be > 0")
    if _np.any(T <= 0):
        raise ValueError("maturities must be > 0")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if volatility_of_variance <= 0:
        raise ValueError("volatility_of_variance must be > 0")
    if initial_variance < 0:
        raise ValueError("initial_variance must be >= 0")
    if correlation < -1.0 or correlation > 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    if K.shape != T.shape:
        raise ValueError("strikes and maturities must have the same shape")
    if (
        not isinstance(num_cos_terms, (int, _np.integer))
        or isinstance(num_cos_terms, bool)
    ):
        raise ValueError("num_cos_terms must be a positive integer")
    if int(num_cos_terms) != num_cos_terms or num_cos_terms <= 0:
        raise ValueError("num_cos_terms must be a positive integer")
    if truncation_width <= 0:
        raise ValueError("truncation_width must be > 0")

    N = int(num_cos_terms)
    L = float(truncation_width)
    S0 = float(spot)
    r = float(risk_free_rate)
    kap = float(kappa)
    th = float(theta)
    sig = float(volatility_of_variance)
    rho = float(correlation)
    v0 = float(initial_variance)

    def cf(u, maturity, ln_sk):
        A = 1j * u * ln_sk + 1j * u * r * maturity
        Delta = sig

        gamma = (
            (kap - 1j * u * rho * Delta) ** 2
            + (1j * u) * (1.0 - 1j * u) * Delta**2
        ) ** 0.5

        denom = (
            2.0 * gamma
            + (kap - gamma - 1j * u * rho * Delta)
            * (1.0 - _cmath.exp(-gamma * maturity))
        )

        B = (
            (kap - gamma) * kap * th * maturity / (Delta**2)
            - 1j * u * rho * kap * th * maturity / Delta
            + (2.0 * kap * th / (Delta**2))
            * _cmath.log(2.0 * gamma / denom)
        )

        C = (
            (1j * u)
            * (1j * u - 1.0)
            * (1.0 - _cmath.exp(-gamma * maturity))
            / denom
        ) * v0

        return _cmath.exp(A + B + C)

    def cum1(strike, maturity):
        return (
            _math.log(S0 / strike)
            + (r - 0.5 * th) * maturity
            + (th - v0)
            * (1.0 - _math.exp(-kap * maturity))
            / (2.0 * kap)
        )

    def cum2(maturity):
        ek = _math.exp(-kap * maturity)
        e2k = _math.exp(-2.0 * kap * maturity)

        return (1.0 / (8.0 * kap**3)) * (
            sig**2
            * maturity
            * kap
            * ek
            * (v0 - th)
            * (8.0 * kap * rho - 4.0 * sig)
            + kap
            * rho
            * sig
            * (1.0 - ek)
            * (16.0 * th - 8.0 * v0)
            + 2.0
            * th
            * kap
            * maturity
            * (-4.0 * kap * rho * sig + sig**2 + 4.0 * kap**2)
            + sig**2
            * (
                (th - 2.0 * v0) * e2k
                + th * (6.0 * ek - 7.0)
                + 2.0 * v0
            )
            + 8.0 * kap**2 * (v0 - th) * (1.0 - ek)
        )

    def Uk(k, a, b):
        if k == 0:
            return _math.exp(a) - a - 1.0

        omega = k * _math.pi / (b - a)

        return (
            (1.0 / (1.0 + omega**2))
            * (
                omega * _math.sin(a * omega)
                - _math.cos(a * omega)
                + _math.exp(a)
            )
            + ((a - b) / (k * _math.pi))
            * _math.sin(a * omega)
        )

    def term_weight(k):
        return 0.5 if k == 0 else 1.0

    def c2_window(c2_raw):
        return max(abs(float(c2_raw)), 1e-10)

    out = _np.empty(K.shape[0], dtype=_np.float64)

    for i in range(K.shape[0]):
        strike = float(K[i])
        maturity = float(T[i])

        c1v = cum1(strike, maturity)
        c2v = c2_window(cum2(maturity))

        a = c1v - L * _math.sqrt(c2v)
        b = c1v + L * _math.sqrt(c2v)

        ln_sk = _math.log(S0 / strike)
        acc = 0.0

        for k in range(N):
            uk = k * _math.pi / (b - a)
            phi = cf(complex(uk), maturity, ln_sk)
            re_term = (
                phi * _cmath.exp(-1j * uk * a)
            ).real

            acc += (
                term_weight(k)
                * re_term
                * Uk(k, a, b)
            )

        price = (
            _math.exp(-r * maturity)
            * strike
            * (2.0 / (b - a))
            * acc
        )

        out[i] = float(max(price, 0.0))

    return out

def build_pckan_training_data():
    import numpy as np

    # Reuse the Step-01 oracle implementation inside the Golden solution.
    pricer = compute_heston_cos_put_prices

    spot = 100.0
    rate = 0.05
    sigma1 = 0.30
    rho1 = -0.5
    v0 = 0.04
    n_cos = 128
    trunc = 10.0

    kappas = np.linspace(0.8, 3.5, 4, dtype=np.float64)
    thetas = np.linspace(0.02, 0.10, 4, dtype=np.float64)
    contracts = np.array(
        [
            [90.0, 0.25],
            [95.0, 0.50],
            [100.0, 0.75],
            [105.0, 1.00],
            [110.0, 1.50],
        ],
        dtype=np.float64,
    )

    n_rows = int(kappas.size * thetas.size * contracts.shape[0])
    if n_rows != 80:
        raise RuntimeError("benchmark dataset must contain exactly 80 rows")

    train_inputs = np.empty((n_rows, 6), dtype=np.float64)
    train_prices = np.empty(n_rows, dtype=np.float64)
    physical_rows = np.empty((n_rows, 4), dtype=np.float64)

    row = 0
    for kappa in kappas:
        for theta in thetas:
            for strike, maturity in contracts:
                kappa_f = float(kappa)
                theta_f = float(theta)
                strike_f = float(strike)
                maturity_f = float(maturity)

                train_inputs[row, 0] = spot / 100.0
                train_inputs[row, 1] = strike_f / 100.0
                train_inputs[row, 2] = maturity_f
                train_inputs[row, 3] = rate
                train_inputs[row, 4] = kappa_f / 4.0
                train_inputs[row, 5] = theta_f / 0.12

                physical_rows[row, 0] = strike_f
                physical_rows[row, 1] = maturity_f
                physical_rows[row, 2] = kappa_f
                physical_rows[row, 3] = theta_f

                priced = pricer(
                    spot,
                    strike_f,
                    maturity_f,
                    rate,
                    kappa_f,
                    theta_f,
                    sigma1,
                    rho1,
                    v0,
                    n_cos,
                    trunc,
                )
                train_prices[row] = float(
                    np.asarray(priced, dtype=np.float64).reshape(-1)[0]
                )
                row += 1

    collocation_inputs = np.array(
        train_inputs,
        dtype=np.float64,
        copy=True,
    )

    return train_inputs, train_prices, collocation_inputs, physical_rows

def initialize_and_evaluate_pckan(inputs):
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

def compute_pckan_physics_residual(inputs, W_hat, W_out, b_out):
    import numpy as np

    x = np.asarray(inputs, dtype=np.float64)
    Wh = np.asarray(W_hat, dtype=np.float64)
    Wo = np.asarray(W_out, dtype=np.float64)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("inputs must have shape (n, 6)")

    if Wh.shape != (8, 6, 4):
        raise ValueError("W_hat must have shape (8, 6, 4)")

    if Wo.shape != (8,):
        raise ValueError("W_out must have shape (8,)")

    try:
        b = float(b_out)
    except (TypeError, ValueError):
        raise ValueError("b_out must be finite")

    if not np.isfinite(b):
        raise ValueError("b_out must be finite")

    if not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite")

    if not np.all(np.isfinite(Wh)):
        raise ValueError("W_hat must be finite")

    if not np.all(np.isfinite(Wo)):
        raise ValueError("W_out must be finite")

    # ------------------------------------------------------------
    # PCKAN forward map
    #
    # z = tanh(x)
    #
    # Second-kind Chebyshev basis:
    # U0 = 1
    # U1 = 2z
    # U2 = 4z^2 - 1
    # U3 = 8z^3 - 4z
    # ------------------------------------------------------------
    z = np.tanh(x)

    u0 = np.ones_like(z)
    u1 = 2.0 * z
    u2 = 4.0 * z**2 - 1.0
    u3 = 8.0 * z**3 - 4.0 * z

    U = np.stack(
        (u0, u1, u2, u3),
        axis=-1,
    )

    h = np.einsum("qpn,bpn->bq", Wh, U)
    V = np.einsum("bq,q->b", h, Wo) + b

    # ------------------------------------------------------------
    # Collapse hidden/readout weights.
    #
    # For coordinate p and Chebyshev order n:
    #
    # C[p,n] = sum_q W_out[q] * W_hat[q,p,n]
    #
    # Therefore
    #
    # V = b + sum_{p,n} C[p,n] U_n(tanh(x_p)).
    #
    # This makes the coordinate derivatives explicit.
    # ------------------------------------------------------------
    C = np.einsum("q,qpn->pn", Wo, Wh)

    # ------------------------------------------------------------
    # First derivatives of U_n(z) with respect to z
    #
    # dU0/dz = 0
    # dU1/dz = 2
    # dU2/dz = 8z
    # dU3/dz = 24z^2 - 4
    # ------------------------------------------------------------
    dU_dz = np.stack(
        (
            np.zeros_like(z),
            2.0 * np.ones_like(z),
            8.0 * z,
            24.0 * z**2 - 4.0,
        ),
        axis=-1,
    )

    # tanh derivative
    dz_dx = 1.0 - z**2

    # dU/dx = dU/dz * dz/dx
    dU_dx = dU_dz * dz_dx[..., None]

    # dV/dx_p
    dV_dx = np.einsum("pn,bpn->bp", C, dU_dx)

    dV_dx0 = dV_dx[:, 0]
    dV_dx2 = dV_dx[:, 2]

    # ------------------------------------------------------------
    # Second derivative with respect to x_0.
    #
    # d2U/dx2 =
    #     d2U/dz2 * (dz/dx)^2
    #     + dU/dz * d2z/dx2
    #
    # where
    #
    # d2z/dx2 = -2 z (1-z^2)
    #
    # and
    #
    # d2U0/dz2 = 0
    # d2U1/dz2 = 0
    # d2U2/dz2 = 8
    # d2U3/dz2 = 48z
    # ------------------------------------------------------------
    d2U_dz2 = np.stack(
        (
            np.zeros_like(z),
            np.zeros_like(z),
            8.0 * np.ones_like(z),
            48.0 * z,
        ),
        axis=-1,
    )

    d2z_dx2 = -2.0 * z * (1.0 - z**2)

    d2U_dx2 = (
        d2U_dz2 * (dz_dx[..., None] ** 2)
        + dU_dz * d2z_dx2[..., None]
    )

    d2V_dx2_all = np.einsum(
        "pn,bpn->bp",
        C,
        d2U_dx2,
    )

    d2V_dx0 = d2V_dx2_all[:, 0]

    # ------------------------------------------------------------
    # Convert benchmark-scaled derivatives to physical derivatives.
    #
    # x_0 = S / 100
    #
    # dV/dS = dV/dx_0 / 100
    # d2V/dS2 = d2V/dx_0^2 / 10000
    #
    # x_2 is time to maturity directly.
    # ------------------------------------------------------------
    dV_dS = dV_dx0 / 100.0
    d2V_dS2 = d2V_dx0 / 10000.0
    dV_dT = dV_dx2

    # ------------------------------------------------------------
    # Physical Black-Scholes residual
    #
    # R = r V
    #     - V_T
    #     - r S V_S
    #     - 0.5 sigma_phys^2 S^2 V_SS
    #
    # Benchmark physical volatility:
    # sigma_phys = sqrt(v0) = sqrt(0.04) = 0.2
    # ------------------------------------------------------------
    S = 100.0 * x[:, 0]
    r = x[:, 3]

    sigma_phys = 0.2

    residuals = (
        r * V
        - dV_dT
        - r * S * dV_dS
        - 0.5 * (sigma_phys**2) * (S**2) * d2V_dS2
    )

    return (
        np.asarray(V, dtype=np.float64),
        np.asarray(dV_dS, dtype=np.float64),
        np.asarray(d2V_dS2, dtype=np.float64),
        np.asarray(dV_dT, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
    )

def train_pckan(train_inputs, train_prices, collocation_inputs):
    import numpy as np

    X_tr = np.asarray(train_inputs, dtype=np.float64)
    y_tr = np.asarray(train_prices, dtype=np.float64)
    X_col = np.asarray(collocation_inputs, dtype=np.float64)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    if X_tr.ndim != 2 or X_tr.shape[1] != 6:
        raise ValueError("train_inputs must have shape (n, 6)")

    if X_col.ndim != 2 or X_col.shape[1] != 6:
        raise ValueError("collocation_inputs must have shape (n, 6)")

    if y_tr.ndim != 1:
        raise ValueError("train_prices must be 1-D")

    if X_tr.shape[0] != y_tr.shape[0]:
        raise ValueError(
            "train_inputs and train_prices length mismatch"
        )

    if X_col.shape[0] != X_tr.shape[0]:
        raise ValueError(
            "collocation_inputs and train_inputs length mismatch"
        )

    if not np.all(np.isfinite(X_tr)):
        raise ValueError("train_inputs must be finite")

    if not np.all(np.isfinite(y_tr)):
        raise ValueError("train_prices must be finite")

    if not np.all(np.isfinite(X_col)):
        raise ValueError("collocation_inputs must be finite")

    # ---------------------------------------------------------
    # Deterministic benchmark initialization
    #
    # Exact serialization of the Step-03 PyTorch float64 seed-0
    # initialization:
    #   torch.manual_seed(0)
    #   W_hat = torch.randn((8, 6, 4), dtype=torch.float64) * 0.01
    #   W_out = torch.randn((8,), dtype=torch.float64) * 0.01
    #   b_out = 0.0
    #
    # These constants make the Step-05 oracle self-contained while
    # preserving exactly the same initialization convention as Step 03.
    # ---------------------------------------------------------
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

    assert W_hat[0, 0, 0] == -0.023104118002341766
    assert W_hat[7, 5, 3] == -0.006750742601943351
    assert W_out[0] == -0.0017502054629307568
    assert W_out[7] == -0.01264403375116388

    # ---------------------------------------------------------
    # Chebyshev basis
    # ---------------------------------------------------------
    def basis(x):
        z = np.tanh(x)

        u0 = np.ones_like(z)
        u1 = 2.0 * z
        u2 = 4.0 * z * z - 1.0
        u3 = 8.0 * z * z * z - 4.0 * z

        return np.stack(
            (u0, u1, u2, u3),
            axis=-1,
        )

    # ---------------------------------------------------------
    # Basis + first/second x derivatives
    # ---------------------------------------------------------
    def basis_with_derivatives(x):
        z = np.tanh(x)

        zp = 1.0 - z * z
        zpp = -2.0 * z * zp

        u0 = np.ones_like(z)
        u1 = 2.0 * z
        u2 = 4.0 * z * z - 1.0
        u3 = 8.0 * z * z * z - 4.0 * z

        U = np.stack(
            (u0, u1, u2, u3),
            axis=-1,
        )

        # dU / dz
        uz0 = np.zeros_like(z)
        uz1 = np.full_like(z, 2.0)
        uz2 = 8.0 * z
        uz3 = 24.0 * z * z - 4.0

        Uz = np.stack(
            (uz0, uz1, uz2, uz3),
            axis=-1,
        )

        # d2U / dz2
        uzz0 = np.zeros_like(z)
        uzz1 = np.zeros_like(z)
        uzz2 = np.full_like(z, 8.0)
        uzz3 = 48.0 * z

        Uzz = np.stack(
            (uzz0, uzz1, uzz2, uzz3),
            axis=-1,
        )

        dU = Uz * zp[..., None]

        d2U = (
            Uzz * (zp[..., None] ** 2)
            + Uz * zpp[..., None]
        )

        return U, dU, d2U

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(x):
        U = basis(x)

        h = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U,
        )

        return (
            np.einsum("bq,q->b", h, W_out)
            + b_out
        )

    # ---------------------------------------------------------
    # Physics quantities and residual
    # ---------------------------------------------------------
    def physics_components(x):
        U, dU, d2U = basis_with_derivatives(x)

        h = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U,
        )

        V = (
            np.einsum("bq,q->b", h, W_out)
            + b_out
        )

        dV_dx0 = np.einsum(
            "qpn,bpn,q->b",
            W_hat,
            dU,
            W_out,
        )

        # Only feature 0 derivative is required here.
        # Mask all other feature derivatives.
        dU0 = np.zeros_like(dU)
        dU0[:, 0, :] = dU[:, 0, :]

        d2U0 = np.zeros_like(d2U)
        d2U0[:, 0, :] = d2U[:, 0, :]

        dV_dx0 = np.einsum(
            "qpn,bpn,q->b",
            W_hat,
            dU0,
            W_out,
        )

        d2V_dx0 = np.einsum(
            "qpn,bpn,q->b",
            W_hat,
            d2U0,
            W_out,
        )

        dU2 = np.zeros_like(dU)
        dU2[:, 2, :] = dU[:, 2, :]

        dV_dx2 = np.einsum(
            "qpn,bpn,q->b",
            W_hat,
            dU2,
            W_out,
        )

        dV_dS = dV_dx0 / 100.0
        d2V_dS2 = d2V_dx0 / 10000.0
        dV_dT = dV_dx2

        S = 100.0 * x[:, 0]
        r = x[:, 3]

        sigma_phys = 0.2

        residual = (
            r * V
            - dV_dT
            - r * S * dV_dS
            - 0.5
            * (sigma_phys ** 2)
            * (S ** 2)
            * d2V_dS2
        )

        return V, residual

    # ---------------------------------------------------------
    # Residual expressed as basis operator.
    #
    # This lets us calculate exact parameter gradients using
    # NumPy rather than autograd.
    # ---------------------------------------------------------
    def residual_basis(x):
        U, dU, d2U = basis_with_derivatives(x)

        r = x[:, 3]
        S = 100.0 * x[:, 0]

        A = r[:, None, None] * U

        # - dV/dT
        A[:, 2, :] -= dU[:, 2, :]

        # - r*S*dV/dS
        A[:, 0, :] -= (
            r[:, None]
            * S[:, None]
            * dU[:, 0, :]
            / 100.0
        )

        # - 0.5*sigma^2*S^2*d2V/dS2
        A[:, 0, :] -= (
            0.5
            * (0.2 ** 2)
            * (S[:, None] ** 2)
            * d2U[:, 0, :]
            / 10000.0
        )

        return U, A, r

    # ---------------------------------------------------------
    # Adam state
    # ---------------------------------------------------------
    m_Wh = np.zeros_like(W_hat)
    v_Wh = np.zeros_like(W_hat)

    m_Wo = np.zeros_like(W_out)
    v_Wo = np.zeros_like(W_out)

    m_b = np.float64(0.0)
    v_b = np.float64(0.0)

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    n_train = X_tr.shape[0]
    n_col = X_col.shape[0]

    # ---------------------------------------------------------
    # 500 Adam iterations
    # ---------------------------------------------------------
    for step in range(1, 501):

        # DATA LOSS
        U_tr = basis(X_tr)

        h_tr = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U_tr,
        )

        preds = (
            np.einsum(
                "bq,q->b",
                h_tr,
                W_out,
            )
            + b_out
        )

        err = preds - y_tr

        coeff_data = (
            2.0 * err / float(n_train)
        )

        grad_Wh_data = np.einsum(
            "b,q,bpn->qpn",
            coeff_data,
            W_out,
            U_tr,
        )

        grad_Wo_data = np.einsum(
            "b,bq->q",
            coeff_data,
            h_tr,
        )

        grad_b_data = np.sum(coeff_data)

        # PHYSICS LOSS
        U_col, A, r_col = residual_basis(X_col)

        h_col = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U_col,
        )

        V_col = (
            np.einsum(
                "bq,q->b",
                h_col,
                W_out,
            )
            + b_out
        )

        h_res = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            A,
        )

        residual = (
            np.einsum(
                "bq,q->b",
                h_res,
                W_out,
            )
            + r_col * b_out
        )

        coeff_phys = (
            2.0 * residual / float(n_col)
        )

        grad_Wh_phys = np.einsum(
            "b,q,bpn->qpn",
            coeff_phys,
            W_out,
            A,
        )

        grad_Wo_phys = np.einsum(
            "b,bq->q",
            coeff_phys,
            h_res,
        )

        grad_b_phys = np.sum(
            coeff_phys * r_col
        )

        # TOTAL GRADIENT
        grad_Wh = (
            grad_Wh_data
            + 0.1 * grad_Wh_phys
        )

        grad_Wo = (
            grad_Wo_data
            + 0.1 * grad_Wo_phys
        )

        grad_b = (
            grad_b_data
            + 0.1 * grad_b_phys
        )

        # ADAM — W_hat
        m_Wh = (
            beta1 * m_Wh
            + (1.0 - beta1) * grad_Wh
        )

        v_Wh = (
            beta2 * v_Wh
            + (1.0 - beta2)
            * (grad_Wh * grad_Wh)
        )

        m_Wh_hat = (
            m_Wh / (1.0 - beta1 ** step)
        )

        v_Wh_hat = (
            v_Wh / (1.0 - beta2 ** step)
        )

        W_hat -= (
            lr
            * m_Wh_hat
            / (np.sqrt(v_Wh_hat) + eps)
        )

        # ADAM — W_out
        m_Wo = (
            beta1 * m_Wo
            + (1.0 - beta1) * grad_Wo
        )

        v_Wo = (
            beta2 * v_Wo
            + (1.0 - beta2)
            * (grad_Wo * grad_Wo)
        )

        m_Wo_hat = (
            m_Wo / (1.0 - beta1 ** step)
        )

        v_Wo_hat = (
            v_Wo / (1.0 - beta2 ** step)
        )

        W_out -= (
            lr
            * m_Wo_hat
            / (np.sqrt(v_Wo_hat) + eps)
        )

        # ADAM — bias
        m_b = (
            beta1 * m_b
            + (1.0 - beta1) * grad_b
        )

        v_b = (
            beta2 * v_b
            + (1.0 - beta2)
            * (grad_b * grad_b)
        )

        m_b_hat = (
            m_b / (1.0 - beta1 ** step)
        )

        v_b_hat = (
            v_b / (1.0 - beta2 ** step)
        )

        b_out -= (
            lr
            * m_b_hat
            / (np.sqrt(v_b_hat) + eps)
        )

    # ---------------------------------------------------------
    # FINAL POST-UPDATE VALUES
    # ---------------------------------------------------------
    predictions = forward(X_tr)

    data_mse_f = np.mean(
        (predictions - y_tr) ** 2
    )

    _, residuals_f = physics_components(X_col)

    physics_mse_f = np.mean(
        residuals_f ** 2
    )

    total_loss_f = (
        data_mse_f
        + 0.1 * physics_mse_f
    )

    return (
        W_hat.astype(np.float64, copy=True),
        W_out.astype(np.float64, copy=True),
        float(b_out),
        predictions.astype(np.float64, copy=True),
        float(data_mse_f),
        float(physics_mse_f),
        float(total_loss_f),
    )

def calibrate_with_differential_evolution(
    W_hat,
    W_out,
    b_out,
    market_prices,
):
    """Run the deterministic frozen-PCKAN DE/rand/1/bin calibration."""
    import numpy as np

    W_hat = np.asarray(W_hat, dtype=np.float64)
    W_out = np.asarray(W_out, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)

    if W_hat.shape != (8, 6, 4):
        raise ValueError("W_hat must have shape (8, 6, 4)")

    if W_out.shape != (8,):
        raise ValueError("W_out must have shape (8,)")

    if market_prices.shape != (5,):
        raise ValueError("market_prices must have shape (5,)")

    if not np.all(np.isfinite(W_hat)):
        raise ValueError("W_hat must be finite")

    if not np.all(np.isfinite(W_out)):
        raise ValueError("W_out must be finite")

    if not np.all(np.isfinite(market_prices)):
        raise ValueError("market_prices must be finite")

    try:
        b_out = float(b_out)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("b_out must be a finite scalar") from exc

    if not np.isfinite(b_out):
        raise ValueError("b_out must be a finite scalar")

    contracts = (
        (90.0, 0.25),
        (95.0, 0.50),
        (100.0, 0.75),
        (105.0, 1.00),
        (110.0, 1.50),
    )

    lo = np.array(
        [0.5, 0.01],
        dtype=np.float64,
    )

    hi = np.array(
        [4.0, 0.12],
        dtype=np.float64,
    )

    def _predict(candidate):
        kappa_1 = float(candidate[0])
        theta_1 = float(candidate[1])

        inputs = np.empty(
            (5, 6),
            dtype=np.float64,
        )

        for i, (strike, maturity) in enumerate(contracts):
            inputs[i] = (
                1.0,
                strike / 100.0,
                maturity,
                0.05,
                kappa_1 / 4.0,
                theta_1 / 0.12,
            )

        z = np.tanh(inputs)

        u0 = np.ones_like(z)
        u1 = 2.0 * z
        u2 = 2.0 * z * u1 - u0
        u3 = 2.0 * z * u2 - u1

        U = np.stack(
            (u0, u1, u2, u3),
            axis=-1,
        )

        hidden = np.einsum(
            "qpn,bpn->bq",
            W_hat,
            U,
        )

        predictions = (
            np.einsum(
                "bq,q->b",
                hidden,
                W_out,
            )
            + b_out
        )

        return predictions

    def _objective(candidate):
        predictions = _predict(candidate)
        errors = market_prices - predictions

        return float(
            np.sum(errors * errors)
        )

    # Deterministic local RNG.
    rng = np.random.default_rng(0)

    population_size = 20
    mutation_factor = 0.8
    crossover_probability = 0.9

    # Generation 0.
    population = (
        lo
        + (hi - lo)
        * rng.random((population_size, 2))
    )

    objectives = np.array(
        [
            _objective(population[i])
            for i in range(population_size)
        ],
        dtype=np.float64,
    )

    all_indices = np.arange(population_size)

    # Exactly 60 evolution generations.
    for _ in range(60):
        for i in range(population_size):

            # Current target cannot be selected as a donor.
            eligible = all_indices[
                all_indices != i
            ]

            donor_indices = rng.choice(
                eligible,
                size=3,
                replace=False,
            )

            a = population[donor_indices[0]]
            bb = population[donor_indices[1]]
            c = population[donor_indices[2]]

            # DE/rand/1 mutation.
            mutant = (
                a
                + mutation_factor * (bb - c)
            ).copy()

            # Coordinate-order bound repair.
            for j in range(2):
                if (
                    mutant[j] < lo[j]
                    or mutant[j] > hi[j]
                ):
                    mutant[j] = (
                        lo[j]
                        + (hi[j] - lo[j])
                        * rng.random()
                    )

            # One forced crossover coordinate.
            jrand = int(
                rng.integers(0, 2)
            )

            trial = population[i].copy()

            # Binomial crossover.
            #
            # IMPORTANT:
            # Consume one rng.random() for EVERY coordinate,
            # including the forced jrand coordinate.
            for j in range(2):
                crossover_draw = rng.random()

                if (
                    crossover_draw < crossover_probability
                    or j == jrand
                ):
                    trial[j] = mutant[j]

            trial_objective = _objective(trial)

            # Asynchronous / in-place replacement.
            if trial_objective <= objectives[i]:
                population[i] = trial
                objectives[i] = trial_objective

    best_idx = int(
        np.argmin(objectives)
    )

    best_parameters = (
        population[best_idx]
        .astype(
            np.float64,
            copy=True,
        )
    )

    best_objective = float(
        objectives[best_idx]
    )

    return (
        best_parameters,
        best_objective,
    )

def run_pckan_calibration_orchestrator():
    """Independent composition of Steps 01–06 oracle APIs."""

    import numpy as np

    # Step 01 — fixed synthetic Heston-limit FVSJ COS market prices
    market_prices = compute_heston_cos_put_prices(
        100.0,
        [90.0, 95.0, 100.0, 105.0, 110.0],
        [0.25, 0.50, 0.75, 1.00, 1.50],
        0.05,
        1.5,
        0.04,
        0.30,
        -0.5,
        0.04,
        128,
        10.0,
    )

    # Step 02 — supervised training and collocation data
    train_inputs, train_prices, collocation_inputs, _physical_rows = (
        build_pckan_training_data()
    )

    # Step 03 — deterministic initialized PCKAN forward stage
    _init_preds, W_hat_init, W_out_init, b_out_init = (
        initialize_and_evaluate_pckan(
            train_inputs
        )
    )

    # Step 04 — required initialized-model physics residual stage
    _pred, _dV_dS, _d2V_dS2, _dV_dT, _residuals = (
        compute_pckan_physics_residual(
            train_inputs,
            W_hat_init,
            W_out_init,
            b_out_init,
        )
    )

    # Step 04 is a required dependency of the end-to-end pipeline.
    # Validate its returned residual before allowing training to proceed.
    _residuals_arr = np.asarray(_residuals, dtype=float)

    if _residuals_arr.shape != (80,):
        raise ValueError(
            "Step 04 must return exactly 80 physics residuals."
        )

    if not np.all(np.isfinite(_residuals_arr)):
        raise ValueError(
            "Step 04 physics residuals must all be finite."
        )

    # Step 05 — train the physics-informed PCKAN only after the
    # required Step-04 physics stage has completed successfully.
    (
        W_hat,
        W_out,
        b_out,
        _train_preds,
        _data_mse,
        _physics_mse,
        _total_loss,
    ) = train_pckan(
        train_inputs,
        train_prices,
        collocation_inputs,
    )

    # Step 06 — calibrate using only the frozen Step-05 surrogate
    _best_parameters, best_objective = (
        calibrate_with_differential_evolution(
            W_hat,
            W_out,
            b_out,
            market_prices,
        )
    )

    return float(best_objective)
SCICODE_GOLD_EOF
