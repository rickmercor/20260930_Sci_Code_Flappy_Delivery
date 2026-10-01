"""
Corrupt a population density with seeded multiplicative log-normal noise whose strength is specified by the expected noise-to-signal ratio.

Additive Gaussian noise can make artificial population densities negative, so noisy histogram data are modelled multiplicatively: every observation is $n_j=\varepsilon_j\,n^\star_j$ with $\varepsilon_j=e^{z_j}$ and $z_j$ independent $\mathcal N(0,\sigma^2)$. The noise strength is reported as the expected noise-to-signal ratio

 

$$\sigma_{NR}=\frac{\mathbb E\big[(n-n^\star)^2\big]}{\|n^\star\|_{RMS}^2}=\mathbb E\big[(\varepsilon-1)^2\big],$$

 

a mean-square (not root-mean-square) ratio that depends only on $\sigma$. Because the log-normal factor has mean $\mathbb E[\varepsilon]=e^{\sigma^2/2}>1$, noisy data are biased upward on average; downstream steps use $\sigma$ to correct this bias.

Returns
-------
np.ndarray with the shape of clean_density: clean_density * exp(sigma Z)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def add_lognormal_noise(clean_density: "np.ndarray", noise_to_signal_ratio: float, seed: int) -> "np.ndarray":
    '''Return the density corrupted by seeded multiplicative log-normal noise.
 
    The noise level sigma >= 0 is the unique nonnegative solution of
 
        E[(exp(z) - 1)^2] = noise_to_signal_ratio,   z ~ N(0, sigma^2),
 
    accurate to within 1e-12 relative. The noisy density is
 
        noisy = clean_density * exp(sigma * Z),
 
    where Z = np.random.default_rng(seed).standard_normal(clean_density.shape) is drawn in
    a single call immediately after constructing the generator (so Z[i, j] is the
    (i, j) entry of that one draw in C order).
 
    Parameters
    ----------
    clean_density : np.ndarray
        Array of noise-free densities (any shape).
    noise_to_signal_ratio : float
        Expected mean-square noise-to-signal ratio sigma_NR >= 0.
    seed : int
        Seed for np.random.default_rng.
 
    Returns
    -------
    noisy : np.ndarray
        Noisy density with the shape of clean_density.
 
    Raises
    ------
    ValueError
        If noise_to_signal_ratio is negative or not finite.
    '''
    return noisy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq
 
def _lognormal_sigma(noise_to_signal_ratio: float) -> float:
    r = float(noise_to_signal_ratio)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("noise_to_signal_ratio must be a finite nonnegative number")
    # E[(e^z - 1)^2] = e^{2 s^2} - 2 e^{s^2/2} + 1 = x^4 - 2x + 1 with x = e^{s^2/2} >= 1
    if r == 0.0:
        return 0.0
    upper = 2.0
    while upper ** 4 - 2.0 * upper + 1.0 < r:
        upper *= 2.0
    x = brentq(lambda y: y ** 4 - 2.0 * y + 1.0 - r, 1.0, upper, xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    return float(np.sqrt(2.0 * np.log(x)))
 
def _oracle_add_lognormal_noise(clean_density: "np.ndarray", noise_to_signal_ratio: float, seed: int) -> "np.ndarray":
    sigma = _lognormal_sigma(noise_to_signal_ratio)
    clean = np.asarray(clean_density, dtype=float)
    z = np.random.default_rng(seed).standard_normal(clean.shape)
    return clean * np.exp(sigma * z)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
a = np.arange(501) * 0.05
clean = np.tile(np.where(a <= 15.0, 1.0 - np.cos(2.0 * np.pi * a / 15.0), 0.0), (101, 1))
""",
            "call": 'add_lognormal_noise(clean.copy(), 0.1, 17)',
            "gold_call": '_oracle_add_lognormal_noise(clean.copy(), 0.1, 17)',
        },
        {
            "setup": """import numpy as np
clean = np.linspace(0.5, 2.0, 12).reshape(3, 4)
""",
            "call": 'add_lognormal_noise(clean.copy(), 0.66, 3)',
            "gold_call": '_oracle_add_lognormal_noise(clean.copy(), 0.66, 3)',
        },
        {
            "setup": """import numpy as np
clean = np.ones(5)
""",
            "call": 'add_lognormal_noise(clean.copy(), 250.0, 0)',
            "gold_call": '_oracle_add_lognormal_noise(clean.copy(), 250.0, 0)',
        },
        {
            "setup": """import numpy as np
clean = np.array([[0.0, 1.0], [2.0, 3.0]])
""",
            "call": 'add_lognormal_noise(clean.copy(), 0.0, 5)',
            "gold_call": '_oracle_add_lognormal_noise(clean.copy(), 0.0, 5)',
        },
        {
            "setup": """import numpy as np
clean = np.ones(3)
def run_model():
    try:
        add_lognormal_noise(clean.copy(), -0.1, 0)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
