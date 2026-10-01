"""
Construct the three real Fourier-symbol channels and their analytic bias derivatives on the specified midpoint grid.

For $q_l=-\pi+2\pi(l+1/2)/n_q$, use $\omega_q=\sqrt{\mu^2+4\kappa\sin^2(q/2)}$ and reservoir inverse temperatures $\beta_L=b-\delta$, $\beta_R=b+\delta$. With $c_a=\coth(\beta_a\omega_q/2)$, the three real symbol columns are $(c_L+c_R)/(4\omega_q)$, $\omega_q(c_L+c_R)/4$, and $\operatorname{sgn}(q)(c_L-c_R)/4$. The last column represents the imaginary mixed symbol. Differentiate with respect to $\delta$ at fixed $\mu,\kappa,b$; $c_L'=\omega_q\operatorname{csch}^2(\beta_L\omega_q/2)/2$ and $c_R'=-\omega_q\operatorname{csch}^2(\beta_R\omega_q/2)/2$.

Returns
-------
np.ndarray, shape (2, nq, 3), real qq, pp and imaginary-mixed symbols followed by their bias derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_ness_symbols(mass: float, coupling: float, beta_mean: float,
                       bias: float, nq: int) -> "np.ndarray":
    """Evaluate the covariance symbols and their first derivative in bias.

    Parameters
    ----------
    mass : float
        Positive finite mass parameter, with oscillator masses and hbar set to one.
    coupling : float
        Positive finite nearest-neighbor spring coupling.
    beta_mean : float
        Finite mean inverse temperature, strictly greater than abs(bias).
    bias : float
        Finite reservoir inverse-temperature bias. Positive bias makes the left reservoir hotter. The derivative is with respect to this value.
    nq : int
        Even integer at least eight, specifying the midpoint momentum grid. Boolean values are invalid.

    Returns
    -------
    symbols : np.ndarray
        Real array of shape (2, nq, 3). Axis zero contains the value and bias derivative, respectively; the final axis is (qq, pp, mixed_imag). Use stable thermal factors at both small and large beta times omega.

    Raises
    ------
    ValueError
        If any input violates its scalar domain, nq is invalid, or the result cannot be represented by finite double-precision numbers.
    """
    return symbols

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_ness_symbols(mass: float, coupling: float, beta_mean: float,
                               bias: float, nq: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    parameters = []
    for value in (mass, coupling, beta_mean, bias):
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Thermal and chain parameters must be real scalars") from exc
        if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
            raise ValueError("Thermal and chain parameters must be real scalars")
        number = float(scalar)
        if not np.isfinite(number):
            raise ValueError("Thermal and chain parameters must be finite")
        parameters.append(number)
    mass, coupling, beta_mean, bias = parameters
    if mass <= 0 or coupling <= 0 or beta_mean <= abs(bias):
        raise ValueError("Require positive mass, coupling and reservoir temperatures")
    if (isinstance(nq, (bool, np.bool_)) or
            not isinstance(nq, (int, np.integer)) or nq < 8 or nq % 2):
        raise ValueError("nq must be an even integer at least eight")

    q = -np.pi + (2.0 * np.pi / nq) * (np.arange(nq) + 0.5)
    omega = np.hypot(mass, 2.0 * np.sqrt(coupling) * np.sin(q / 2.0))
    # Negative exponentials avoid overflow for cold reservoirs; expm1 retains
    # the small thermal denominator for hot reservoirs.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        arguments = np.array([beta_mean - bias, beta_mean + bias])[:, None] * omega
        decays = np.exp(-arguments)
        denominators = -np.expm1(-arguments)
        coth = 1.0 + 2.0 * decays / denominators
        coth_tangent = (omega * (decays / denominators)) * (2.0 / denominators)
        coth_tangent[1] *= -1.0
        sums = np.stack((coth.sum(axis=0), coth_tangent.sum(axis=0)))
        differences = np.stack((coth[0] - coth[1],
                                coth_tangent[0] - coth_tangent[1]))
        symbols = np.empty((2, nq, 3), dtype=float)
        symbols[:, :, 0] = (sums / 4.0) / omega
        symbols[:, :, 1] = (omega / 4.0) * sums
        symbols[:, :, 2] = np.sign(q) * differences / 4.0
    if not np.all(np.isfinite(symbols)):
        raise ValueError("The requested symbols are not finite in double precision")
    return symbols

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests spanning equilibrium, bias and thermal limits."""
    return [
        {
            "setup": "import numpy as np\nargs = (0.45, 1.0, 3.2, 0.9, 128)",
            "call": "build_ness_symbols(*args)",
            "gold_call": "_oracle_build_ness_symbols(*args)",
        },
        {
            "setup": "import numpy as np\nargs = (0.8, 0.4, 2.0, 0.0, 8)",
            "call": "build_ness_symbols(*args)",
            "gold_call": "_oracle_build_ness_symbols(*args)",
        },
        {
            "setup": "import numpy as np\nargs = (0.23, 1.7, 2.3, -1.4, 30)",
            "call": "build_ness_symbols(*args)",
            "gold_call": "_oracle_build_ness_symbols(*args)",
        },
        {
            "setup": "import numpy as np\nargs = (0.2, 0.7, 1e-4, 9.9e-5, 16)",
            "call": "build_ness_symbols(*args)",
            "gold_call": "_oracle_build_ness_symbols(*args)",
        },
        {
            "setup": "import numpy as np\nargs = (1.3, 0.25, 1e6, 2e5, 12)",
            "call": "build_ness_symbols(*args)",
            "gold_call": "_oracle_build_ness_symbols(*args)",
        },
        {
            "setup": """import numpy as np
args = (0.4, 1.0, 2.0, 2.0, 16)
def _raises_value_error(fn):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(build_ness_symbols)",
            "gold_call": "_raises_value_error(_oracle_build_ness_symbols)",
        },
        {
            "setup": """import numpy as np
args = (0.4, 1.0, 2.0, 0.3, True)
def _raises_value_error(fn):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(build_ness_symbols)",
            "gold_call": "_raises_value_error(_oracle_build_ness_symbols)",
        },
        {
            "setup": """import numpy as np
args = (0.4, -1.0, 2.0, 0.3, 16)
def _raises_value_error(fn):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(build_ness_symbols)",
            "gold_call": "_raises_value_error(_oracle_build_ness_symbols)",
        },
    ]
