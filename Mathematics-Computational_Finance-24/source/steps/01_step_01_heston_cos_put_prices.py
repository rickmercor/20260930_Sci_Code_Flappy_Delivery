"""
Compute deterministic European put prices using Zhang, Wu, Xiao, Gong, and Xu's FVSJ Fourier-cosine pricing construction specialized to the single-factor Heston limit. The reduction sets lambda = 0, kappa_2 = theta_2 = sigma_2 = 0 and H_1 = H_2 = 1/2, so only the first stochastic-variance factor remains and Delta_1 = sigma_1.



For each strike and maturity, construct the Heston-limit characteristic function of terminal log-moneyness, compute the first and second cumulants used by the COS truncation window, evaluate the paper's European-put COS payoff coefficients, and sum the truncated Fourier-cosine expansion. The benchmark uses the conventional half-weight on the k = 0 COS summand and constructs the truncation window from max(abs(c2), 1e-10).



This step is the deterministic offline reference pricer used to generate synthetic market prices and supervised PCKAN training labels. It does not construct or train the PCKAN, evaluate the physics-informed loss, or perform differential-evolution calibration.

Zhang, Wu, Xiao, Gong, and Xu formulate European put valuation under their FVSJ model through a Fourier-cosine expansion driven by the model characteristic function. For the SciCode benchmark, the FVSJ model is reduced to its paper-supported single-factor Heston limit by setting lambda = 0, kappa_2 = theta_2 = sigma_2 = 0 and H_1 = H_2 = 1/2. Consequently, the jump contribution and second variance factor vanish, while Delta_1 = sigma_1.



The remaining characteristic function has the paper's exponential-affine structure



    phi(u) = exp(A(u) + B(u) + C(u)),



where A contains the initial log-moneyness and risk-free drift, and B and C contain the active stochastic-variance contribution.



For each contract, the COS approximation is evaluated on a cumulant-based log-moneyness interval



    [a,b] =

    [c1 - L*sqrt(c2_window),

     c1 + L*sqrt(c2_window)],



with



    c2_window = max(abs(c2), 1e-10).



The absolute value follows the cumulant-window construction, while the 1e-10 floor is a deterministic benchmark convention protecting the window against numerical roundoff.



The COS frequencies are



    u_k = k*pi/(b-a),



for k = 0,...,N-1. The European-put payoff is represented by the paper's COS coefficient U_k. In particular,



    U_0 = exp(a) - a - 1.



The benchmark applies the conventional COS half-weight to the zeroth summand exactly once: the k = 0 summand receives weight 1/2, while U_0 itself is not halved.



The resulting discounted expansion produces the European put price used as the offline FVSJ reference value. In the complete benchmark these prices become both the fixed synthetic market observations at the ground-truth parameters and the supervised labels used to train the PCKAN surrogate. Black-Scholes pricing, PCKAN construction, physics-informed training, and differential-evolution calibration belong to later steps.

Returns
-------
return put_prices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_heston_cos_put_prices(
    spot: float,
    strikes,
    maturities,
    risk_free_rate: float,
    kappa: float,
    theta: float,
    volatility_of_variance: float,
    correlation: float,
    initial_variance: float,
    num_cos_terms: int = 128,
    truncation_width: float = 10.0,
):
    """Compute European put prices using the Heston-limit FVSJ COS construction.

    Parameters
    ----------
    spot
        Current underlying price S0.
    strikes
        Scalar or one-dimensional array of positive strikes.
    maturities
        Scalar or one-dimensional array of positive maturities matching strikes.
    risk_free_rate
        Continuously compounded risk-free rate.
    kappa
        Mean-reversion speed kappa_1.
    theta
        Long-run variance theta_1.
    volatility_of_variance
        Volatility-of-variance sigma_1, equal to Delta_1 in the Heston limit.
    correlation
        Spot/variance correlation rho_1.
    initial_variance
        Initial variance v0.
    num_cos_terms
        Number N of COS expansion terms.
    truncation_width
        Cumulant-window multiplier L.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of European put prices.

    Raises
    ------
    ValueError
        If spot is non-positive; any strike is non-positive; any maturity is
        non-positive; strikes and maturities have different lengths; kappa is
        non-positive; volatility_of_variance is non-positive; correlation is
        outside the interval [-1, 1]; or num_cos_terms is non-positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_heston_cos_put_prices(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # CASE 1 — normal ATM benchmark
        {
            "setup": (
                "import numpy as np\n"
                "S0 = 100.0\n"
                "strikes = [100.0]\n"
                "maturities = [0.75]"
            ),
            "call": (
                "compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
            "gold_call": (
                "_oracle_compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
        },

        # CASE 2 — lower calibration-domain boundary
        {
            "setup": (
                "import numpy as np\n"
                "S0 = 100.0\n"
                "strikes = [100.0]\n"
                "maturities = [0.75]"
            ),
            "call": (
                "compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 0.5, 0.01, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
            "gold_call": (
                "_oracle_compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 0.5, 0.01, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
        },

        # CASE 3 — complete five-contract synthetic market panel
        {
            "setup": (
                "import numpy as np\n"
                "S0 = 100.0\n"
                "strikes = [90.0, 95.0, 100.0, 105.0, 110.0]\n"
                "maturities = [0.25, 0.50, 0.75, 1.00, 1.50]"
            ),
            "call": (
                "compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
            "gold_call": (
                "_oracle_compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
        },

        # CASE 4 — public-API COS-sum regression
        {
            "setup": (
                "import numpy as np\n"
                "S0 = 100.0\n"
                "strikes = [90.0]\n"
                "maturities = [0.25]"
            ),
            "call": (
                "compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
            "gold_call": (
                "_oracle_compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
        },

        # CASE 5 — opposite valid calibration-domain edge
        {
            "setup": (
                "import numpy as np\n"
                "S0 = 100.0\n"
                "strikes = [110.0]\n"
                "maturities = [1.50]"
            ),
            "call": (
                "compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 4.0, 0.12, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
            "gold_call": (
                "_oracle_compute_heston_cos_put_prices("
                "S0, strikes, maturities, "
                "0.05, 4.0, 0.12, 0.30, -0.5, 0.04, 128, 10.0)"
            ),
        },

        # CASE 6 — invalid-input contract
        {
            "setup": (
                "import numpy as np\n"
                "cases = [\n"
                "    (0.0, [100.0], [0.75], "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [0.0], [0.75], "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0], [0.0], "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0, 105.0], [0.75], "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0], [0.75], "
                "0.05, 0.0, 0.04, 0.30, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0], [0.75], "
                "0.05, 1.5, 0.04, 0.0, -0.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0], [0.75], "
                "0.05, 1.5, 0.04, 0.30, -1.5, 0.04, 128, 10.0),\n"
                "    (100.0, [100.0], [0.75], "
                "0.05, 1.5, 0.04, 0.30, -0.5, 0.04, 0, 10.0),\n"
                "]\n"
                "statuses = []\n"
                "for args in cases:\n"
                "    try:\n"
                "        compute_heston_cos_put_prices(*args)\n"
                "        statuses.append(0)\n"
                "    except ValueError:\n"
                "        statuses.append(1)\n"
                "    except Exception:\n"
                "        statuses.append(2)"
            ),
            "call": "tuple(statuses)",
            "gold_call": "(1, 1, 1, 1, 1, 1, 1, 1)",
        },
    ]
