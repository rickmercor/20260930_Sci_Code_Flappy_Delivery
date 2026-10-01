"""
Generate resolution-limited draws of the empirical prior from supplied standard-normal and uniform variates.

The prior on the emitted intensity $\mu_j$ of a bin is hierarchical. A latent local mean $m_j$ carries a lognormal distribution around the prior centre $\mu_{\mathrm{RL},j}$, shifted so that the latent mean averages to the centre exactly, $\log m_j \sim \mathrm{Normal}(\log \mu_{\mathrm{RL},j} - \tfrac{1}{2} \sigma_j^2,\ \sigma_j^2)$, with the log-scale width $\sigma_j$ assigned to that bin. Conditioned on the latent mean, the emitted intensity carries a Gamma distribution with a fixed shape $\alpha$ and rate $\alpha / m_j$, $\mu_j \mid m_j \sim \mathrm{Gamma}(\alpha, \alpha / m_j)$, which again leaves the expectation at the centre. The two layers are independent across bins, so no correlation is imposed on the emitted spectrum by the prior. Whatever correlation the reported spectrum shows comes from the resolution operator $\mathbf{G}_\gamma$, through which every emitted draw is mapped before it is reported.

Reproducibility replaces the two random layers by supplied variates. The Gaussian layer of every bin takes its standardised variate $z_j$ from the normal-draw array, $\log m_j = \log \mu_{\mathrm{RL},j} - \tfrac{1}{2} \sigma_j^2 + \sigma_j z_j$, and the conditional Gamma variate is obtained by evaluating the inverse cumulative distribution function of $\mathrm{Gamma}(\alpha, \alpha / m_j)$ at the corresponding entry $u_j$ of the uniform-draw array. One row of each array is one draw of the whole spectrum.

Returns
-------
np.ndarray, shape (N, J). Row s is the s-th prior draw of the emitted spectrum mapped through the resolution operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def draw_prior_spectra(
    centre: np.ndarray,
    width: np.ndarray,
    gamma_shape: float,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    r"""Map supplied variates through the two-layer prior and the resolution operator.

    Parameters
    ----------
    centre : np.ndarray
        Prior centre of every bin in emitted space, shape (J,), finite and
        strictly positive.
    width : np.ndarray
        Log-scale width of every bin, shape (J,), finite and strictly
        positive.
    gamma_shape : float
        Shape parameter of the conditional Gamma distribution, finite and
        strictly positive. The rate is the shape divided by the latent mean
        of the bin.
    normal_draws : np.ndarray
        Standard-normal variates, shape (N, J), finite, feeding the Gaussian
        layer of the corresponding bin and draw.
    uniform_draws : np.ndarray
        Uniform variates, shape (N, J), strictly between 0 and 1, evaluated
        through the inverse cumulative distribution function of the
        conditional Gamma distribution of the corresponding bin and draw.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, every column summing to one to within 1e-6.

    Returns
    -------
    draws : np.ndarray
        Shape (N, J). Row s is the s-th prior draw of the emitted spectrum
        mapped through the resolution operator.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above or disagree on the number
        of bins or draws, if there are no draws or no bins, if a centre or
        width is not finite and strictly positive, if ``gamma_shape`` is not
        finite and strictly positive, if a normal variate is not finite, if a
        uniform variate lies outside the open unit interval, or if the
        resolution is negative, not finite or not column-normalised to within
        1e-6.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_draw_prior_spectra(
    centre: np.ndarray,
    width: np.ndarray,
    gamma_shape: float,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    from scipy.special import gammaincinv

    mu_c = np.asarray(centre, dtype=float)
    sig = np.asarray(width, dtype=float)
    z = np.asarray(normal_draws, dtype=float)
    u = np.asarray(uniform_draws, dtype=float)
    g = np.asarray(resolution, dtype=float)
    alpha = float(gamma_shape)

    if mu_c.ndim != 1 or mu_c.size == 0:
        raise ValueError("centre must be a non-empty 1D array")
    n_bins = mu_c.size
    if sig.shape != (n_bins,):
        raise ValueError("width must have shape (J,) matching centre")
    if z.ndim != 2 or z.shape[1] != n_bins or z.shape[0] == 0:
        raise ValueError("normal_draws must have shape (N, J) with at least one draw")
    if u.shape != z.shape:
        raise ValueError("uniform_draws must have the same shape as normal_draws")
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching centre")
    if not np.all(np.isfinite(mu_c)) or np.any(mu_c <= 0.0):
        raise ValueError("centre must be finite and strictly positive")
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("width must be finite and strictly positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("gamma_shape must be a finite positive number")
    if not np.all(np.isfinite(z)):
        raise ValueError("normal_draws must be finite")
    if not np.all(np.isfinite(u)) or np.any(u <= 0.0) or np.any(u >= 1.0):
        raise ValueError("uniform_draws must lie strictly inside the unit interval")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("resolution must be finite and non-negative")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")

    # Non-centred lognormal layer, mean-preserving around the centre.
    log_m = np.log(mu_c) - 0.5 * sig ** 2 + sig * z
    latent_mean = np.exp(log_m)
    # Conditional Gamma(alpha, alpha / m) by inverse-CDF at the supplied uniform.
    emitted = latent_mean * gammaincinv(alpha, u) / alpha
    # Report in the resolution-limited space.
    return emitted @ g.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    def _benchmark_setup():
        return """import numpy as np
resolution = np.array([
    [0.9220, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0780, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0780],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.9220],
])
centre = np.array([42.563007, 1811.339584, 408.571456, 115.808072, 451.533414, 9.288457, 0.1, 0.1])
width = np.array([2.591208, 1.463601, 2.037942, 2.614201, 2.159064, 2.940618, 2.999967, 2.999999])
normal_draws = np.array([
    [-1.3172, 1.2482, -0.0462, 0.5644, 0.4015, 0.8277, -0.5209, -1.1960],
    [0.2368, -0.2066, 0.6623, -1.0854, -0.1586, 0.0253, 0.3285, 0.9852],
    [1.7075, 0.8812, 0.7809, -0.2051, -0.9937, 1.5846, -0.9145, -0.7679],
    [-0.9799, -1.0106, -1.2303, 1.0254, 0.1357, -1.3587, -2.2404, -0.2426],
    [-0.6562, 1.3738, -0.1800, 1.7620, 2.2083, 0.1282, 0.5415, -0.2978],
    [-1.1052, -0.5546, 0.4613, 1.5446, -0.3304, -0.1789, -0.3454, -0.3855],
    [0.4446, -0.5375, 0.1057, 0.6215, -0.7146, -0.8187, 0.4618, 0.2888],
    [-0.7490, -1.3887, -2.0672, -1.0401, -0.8608, 0.0542, -0.0914, 0.7233],
    [0.1356, -1.7114, -0.2807, 1.2544, -0.0028, -0.2288, 0.0483, 1.6938],
    [0.6963, 0.3910, 1.9035, 0.3795, -0.9446, -0.6751, -0.3260, -0.1255],
    [-1.3818, 0.5132, 0.3204, -0.6988, -0.6736, 0.3571, 0.2233, -0.3653],
    [1.6635, -0.4442, 0.4165, 1.8852, -0.6552, -1.2765, -0.5137, 0.7405],
    [0.0872, 0.7130, 1.0826, 0.4673, -2.1087, -1.9236, 0.2541, -1.0312],
    [-1.1075, 0.8420, -0.3870, 0.1768, -0.0485, -1.0076, -0.4778, 1.6000],
    [0.4420, 0.5495, 0.0880, -0.1985, 1.5141, 1.5562, -0.3511, -1.1659],
    [0.3934, 0.5039, 0.7601, -0.9601, 1.1018, -1.0292, -0.4393, -0.4425],
])
uniform_draws = np.array([
    [0.0888, 0.9355, 0.0318, 0.8306, 0.0715, 0.0664, 0.8101, 0.9222],
    [0.2122, 0.5367, 0.5288, 0.9245, 0.9291, 0.5789, 0.3794, 0.9377],
    [0.1072, 0.6361, 0.0406, 0.5973, 0.2047, 0.2923, 0.4446, 0.3373],
    [0.7535, 0.0263, 0.1654, 0.4910, 0.9082, 0.1059, 0.2709, 0.8614],
    [0.4247, 0.7255, 0.0231, 0.1075, 0.4890, 0.0209, 0.5067, 0.9778],
    [0.3624, 0.5765, 0.7857, 0.5459, 0.3383, 0.1387, 0.2239, 0.4842],
    [0.8331, 0.4785, 0.7814, 0.3562, 0.6291, 0.6273, 0.4827, 0.7815],
    [0.2814, 0.8762, 0.3056, 0.7721, 0.8010, 0.3934, 0.2598, 0.5451],
    [0.1599, 0.9630, 0.5310, 0.0950, 0.0614, 0.6113, 0.1181, 0.2301],
    [0.9018, 0.1109, 0.0210, 0.1611, 0.8519, 0.3587, 0.8923, 0.1646],
    [0.4891, 0.5280, 0.1943, 0.8251, 0.0464, 0.3037, 0.9450, 0.5611],
    [0.7510, 0.7096, 0.5391, 0.4929, 0.1551, 0.2387, 0.2519, 0.5734],
    [0.9566, 0.7836, 0.6671, 0.4128, 0.9785, 0.1016, 0.6567, 0.0204],
    [0.7842, 0.5067, 0.8415, 0.6694, 0.8716, 0.1006, 0.0680, 0.7919],
    [0.2552, 0.0236, 0.8433, 0.1673, 0.7827, 0.7424, 0.9036, 0.2916],
    [0.7761, 0.2803, 0.7159, 0.5054, 0.0913, 0.2354, 0.1605, 0.2408],
])
"""

    args = "centre, width, gamma_shape, normal_draws, uniform_draws, resolution"
    call = f"draw_prior_spectra({args})"
    gold = f"_oracle_draw_prior_spectra({args})"
    err = ""
    for name, function in (("run_model", "draw_prior_spectra"),
                           ("run_gold", "_oracle_draw_prior_spectra")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        {
            "setup": _benchmark_setup() + "gamma_shape = 1.0\n",
            "call": call,
            "gold_call": gold,
        },
        # a shape parameter above one, where the conditional distribution is no longer exponential
        {
            "setup": _benchmark_setup() + """gamma_shape = 2.5
""",
            "call": call,
            "gold_call": gold,
        },
        # a shape parameter below one, where the conditional distribution piles up at zero
        {
            "setup": _benchmark_setup() + """gamma_shape = 0.5
""",
            "call": call,
            "gold_call": gold,
        },
        # a three-bin grid with a broad resolution and a handful of draws
        {
            "setup": """import numpy as np
resolution = np.array([
    [0.80, 0.15, 0.05],
    [0.20, 0.70, 0.25],
    [0.00, 0.15, 0.70],
])
centre = np.array([12.0, 350.0, 2.5])
width = np.array([2.2, 1.1, 2.9])
gamma_shape = 1.0
normal_draws = np.array([
    [0.31, -1.20, 2.05],
    [-0.77, 0.44, -0.10],
    [1.62, 0.08, -1.35],
    [-0.05, -0.96, 0.58],
    [0.90, 1.31, -2.20],
])
uniform_draws = np.array([
    [0.10, 0.55, 0.93],
    [0.62, 0.21, 0.47],
    [0.35, 0.88, 0.05],
    [0.71, 0.14, 0.66],
    [0.28, 0.79, 0.52],
])
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a single draw
        {
            "setup": """import numpy as np
resolution = np.eye(3)
centre = np.array([12.0, 350.0, 2.5])
width = np.array([2.2, 1.1, 2.9])
gamma_shape = 1.0
normal_draws = np.array([[0.31, -1.20, 2.05]])
uniform_draws = np.array([[0.10, 0.55, 0.93]])
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: zero normal variates and uniform variates at one minus the reciprocal of e, so
        # every latent mean is the shifted centre and every Gamma variate is its own mean
        {
            "setup": """import numpy as np
resolution = np.eye(4)
centre = np.array([5.0, 50.0, 500.0, 0.1])
width = np.array([0.5, 1.0, 1.5, 3.0])
gamma_shape = 1.0
normal_draws = np.zeros((2, 4))
uniform_draws = np.full((2, 4), 1.0 - np.exp(-1.0))
""",
            "call": call,
            "gold_call": gold,
        },
        # boundary: a diagonal resolution, so the draws are reported in emitted space unchanged
        {
            "setup": """import numpy as np
resolution = np.eye(3)
centre = np.array([12.0, 350.0, 2.5])
width = np.array([2.2, 1.1, 2.9])
gamma_shape = 1.0
normal_draws = np.array([
    [0.31, -1.20, 2.05],
    [-0.77, 0.44, -0.10],
    [1.62, 0.08, -1.35],
])
uniform_draws = np.array([
    [0.10, 0.55, 0.93],
    [0.62, 0.21, 0.47],
    [0.35, 0.88, 0.05],
])
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a very small width, so the latent mean sits at the centre and only the Gamma
        # layer spreads the draws
        {
            "setup": """import numpy as np
resolution = np.eye(3)
centre = np.array([12.0, 350.0, 2.5])
width = np.array([1e-6, 1e-6, 1e-6])
gamma_shape = 1.0
normal_draws = np.array([
    [0.31, -1.20, 2.05],
    [-0.77, 0.44, -0.10],
])
uniform_draws = np.array([
    [0.10, 0.55, 0.93],
    [0.62, 0.21, 0.47],
])
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: uniform variates close to the ends of the interval, which reach far into the tails
        {
            "setup": """import numpy as np
resolution = np.eye(2)
centre = np.array([40.0, 4.0])
width = np.array([1.5, 2.5])
gamma_shape = 1.0
normal_draws = np.array([
    [0.0, 0.0],
    [1.0, -1.0],
])
uniform_draws = np.array([
    [1e-6, 1.0 - 1e-6],
    [1.0 - 1e-6, 1e-6],
])
""",
            "call": call,
            "gold_call": gold,
        },
        # edge: a single bin whose draws are scalars mapped through the number one
        {
            "setup": """import numpy as np
resolution = np.array([[1.0]])
centre = np.array([75.0])
width = np.array([1.8])
gamma_shape = 3.0
normal_draws = np.array([[0.4], [-0.9], [1.7], [0.0]])
uniform_draws = np.array([[0.3], [0.6], [0.9], [0.5]])
""",
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": """import numpy as np
# invalid: a uniform variate of exactly one, at which the inverse distribution function diverges
resolution = np.eye(2)
centre = np.array([50.0, 400.0])
width = np.array([1.5, 1.0])
normal_draws = np.array([[0.3, -0.4], [1.1, 0.2]])
uniform_draws = np.array([[0.3, 1.0], [0.6, 0.5]])
args = (centre, width, 1.0, normal_draws, uniform_draws, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a prior centre of zero, whose logarithm is not defined
resolution = np.eye(2)
centre = np.array([50.0, 0.0])
width = np.array([1.5, 1.0])
normal_draws = np.array([[0.3, -0.4], [1.1, 0.2]])
uniform_draws = np.array([[0.3, 0.7], [0.6, 0.5]])
args = (centre, width, 1.0, normal_draws, uniform_draws, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a Gamma shape of zero
resolution = np.eye(2)
centre = np.array([50.0, 400.0])
width = np.array([1.5, 1.0])
normal_draws = np.array([[0.3, -0.4], [1.1, 0.2]])
uniform_draws = np.array([[0.3, 0.7], [0.6, 0.5]])
args = (centre, width, 0.0, normal_draws, uniform_draws, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a width of zero in one bin
resolution = np.eye(2)
centre = np.array([50.0, 400.0])
width = np.array([1.5, 0.0])
normal_draws = np.array([[0.3, -0.4], [1.1, 0.2]])
uniform_draws = np.array([[0.3, 0.7], [0.6, 0.5]])
args = (centre, width, 1.0, normal_draws, uniform_draws, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a resolution column 5e-6 above one, outside the stated 1e-6 tolerance
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.800005, 0.15],
    [0.00, 0.10, 0.85],
])
centre = np.array([50.0, 400.0, 20.0])
width = np.array([1.5, 1.0, 2.0])
normal_draws = np.array([[0.3, -0.4, 0.8], [1.1, 0.2, -0.6]])
uniform_draws = np.array([[0.3, 0.7, 0.2], [0.6, 0.5, 0.9]])
args = (centre, width, 1.0, normal_draws, uniform_draws, resolution)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # boundary: a resolution column 5e-7 above one, inside the stated tolerance
        {
            "setup": """import numpy as np
resolution = np.array([
    [0.85, 0.10, 0.00],
    [0.15, 0.8000005, 0.15],
    [0.00, 0.10, 0.85],
])
centre = np.array([50.0, 400.0, 20.0])
width = np.array([1.5, 1.0, 2.0])
gamma_shape = 1.0
normal_draws = np.array([[0.3, -0.4, 0.8], [1.1, 0.2, -0.6]])
uniform_draws = np.array([[0.3, 0.7, 0.2], [0.6, 0.5, 0.9]])
""",
            "call": call,
            "gold_call": gold,
        },
    ]
    # Both sides receive their own deep copy of every argument, so an in-place
    # implementation on either side cannot contaminate the other's inputs.
    isolate = (
        "\nimport copy as _test_copy\n\n"
        "def _independent_inputs(_function):\n"
        "    def _invoke(*_args, **_kwargs):\n"
        "        _args, _kwargs = _test_copy.deepcopy((_args, _kwargs))\n"
        "        return _function(*_args, **_kwargs)\n"
        "    return _invoke\n\n"
        "draw_prior_spectra = _independent_inputs(draw_prior_spectra)\n"
        "_oracle_draw_prior_spectra = _independent_inputs(_oracle_draw_prior_spectra)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
