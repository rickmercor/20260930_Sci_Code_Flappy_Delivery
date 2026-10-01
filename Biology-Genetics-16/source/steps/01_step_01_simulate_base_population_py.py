"""
'''Materialise the phased base population under the reference construction.




    Draw Z = rng.standard_normal((2 * n_individuals, n_loci)) from

    numpy.random.default_rng(seed). Set X[:, 0] = Z[:, 0] and, for i >= 1,

    X[:, i] = rho * X[:, i-1] + sqrt(1 - rho**2) * Z[:, i]. Locus i carries the

    reference allele on a haplotype when X[:, i] exceeds the standard-normal

    quantile of (1 - t_i), where t_i is the i-th of n_loci values evenly spaced

    from p_low to p_high inclusive. Individual k owns rows 2k and 2k+1.




    Parameters

    ----------

    n_loci : int

        Number of biallelic loci, in physical order along one chromosome. >= 2.

    n_individuals : int

        Number of diploid individuals in the base population. >= 2.

    rho : float

        Lag-1 correlation of the latent AR(1) process. In [0, 1).

    p_low : float

        Target reference-allele frequency at the first locus. In (0, 1).

    p_high : float

        Target reference-allele frequency at the last locus. In (0, 1).

    seed : int

        Seed passed to numpy.random.default_rng.




    Returns

    -------

    haplotypes : np.ndarray

        (2 * n_individuals, n_loci) array of 0/1 integers.




    Raises

    ------

    ValueError

        If n_loci < 2, if n_individuals < 2, if rho is not in [0, 1),

        if p_low or p_high is not in (0, 1), or if seed is not an integer.

    '''

Every downstream quantity is conditional on the genotypes of the base population

from which the replicate populations were founded, so the analysis starts by

materialising that population. Because the method needs both the gametic-phase

and the non-gametic-phase components of the diversity matrix, the population must

be represented as phased haplotypes, not as genotype dosages.




The reference construction produces a chromosome with graded linkage

disequilibrium: a latent Gaussian AR(1) process is run along the locus order and

thresholded per locus. The AR(1) correlation controls how fast disequilibrium

decays with distance; the per-locus threshold sets the reference-allele frequency,

which is spread across a wide range so the frequency contrast p - q carries real

signal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_base_population(n_loci: int, n_individuals: int, rho: float,
                             p_low: float, p_high: float, seed: int) -> np.ndarray:
    '''Materialise the phased base population under the reference construction.

    Draw Z = rng.standard_normal((2 * n_individuals, n_loci)) from
    numpy.random.default_rng(seed). Set X[:, 0] = Z[:, 0] and, for i >= 1,
    X[:, i] = rho * X[:, i-1] + sqrt(1 - rho**2) * Z[:, i]. Locus i carries the
    reference allele on a haplotype when X[:, i] exceeds the standard-normal
    quantile of (1 - t_i), where t_i is the i-th of n_loci values evenly spaced
    from p_low to p_high inclusive. Individual k owns rows 2k and 2k+1.

    Parameters
    ----------
    n_loci : int
        Number of biallelic loci, in physical order along one chromosome. >= 2.
    n_individuals : int
        Number of diploid individuals in the base population. >= 2.
    rho : float
        Lag-1 correlation of the latent AR(1) process. In [0, 1).
    p_low : float
        Target reference-allele frequency at the first locus. In (0, 1).
    p_high : float
        Target reference-allele frequency at the last locus. In (0, 1).
    seed : int
        Seed passed to numpy.random.default_rng.

    Returns
    -------
    haplotypes : np.ndarray
        (2 * n_individuals, n_loci) array of 0/1 integers.

    Raises
    ------
    ValueError
        If n_loci < 2, if n_individuals < 2, if rho is not in [0, 1),
        if p_low or p_high is not in (0, 1), or if seed is not an integer.
    '''
    return haplotypes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _standard_normal_quantile(prob: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF, vectorised, without a SciPy dependency."""
    from math import sqrt
    try:
        from scipy.special import ndtri
        return np.asarray(ndtri(prob), dtype=float)
    except Exception:
        from math import erf
        p = np.asarray(prob, dtype=float)
        lo, hi = np.full(p.shape, -40.0), np.full(p.shape, 40.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * (1.0 + np.vectorize(erf)(mid / sqrt(2.0)))
            hi = np.where(cdf > p, mid, hi)
            lo = np.where(cdf > p, lo, mid)
        return 0.5 * (lo + hi)


# ORACLE FUNCTION

def _oracle_simulate_base_population(n_loci: int, n_individuals: int, rho: float,
                                     p_low: float, p_high: float,
                                     seed: int) -> np.ndarray:
    if not (isinstance(n_loci, (int, np.integer)) and int(n_loci) >= 2):
        raise ValueError("n_loci must be an integer >= 2")
    if not (isinstance(n_individuals, (int, np.integer)) and int(n_individuals) >= 2):
        raise ValueError("n_individuals must be an integer >= 2")
    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.floating, np.integer)):
        raise ValueError("rho must be a real number in [0, 1)")
    if not (0.0 <= float(rho) < 1.0):
        raise ValueError("rho must be in [0, 1)")
    for name, val in (("p_low", p_low), ("p_high", p_high)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a real number in (0, 1)")
        if not (0.0 < float(val) < 1.0):
            raise ValueError(f"{name} must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n_loci, n_individuals = int(n_loci), int(n_individuals)
    rho = float(rho)
    rng = np.random.default_rng(int(seed))
    Z = rng.standard_normal((2 * n_individuals, n_loci))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(1, n_loci):
        X[:, i] = rho * X[:, i - 1] + scale * Z[:, i]
    targets = np.linspace(float(p_low), float(p_high), n_loci)
    thresholds = _standard_normal_quantile(1.0 - targets)
    return (X > thresholds[None, :]).astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration ---
        {
            "setup": "import numpy as np\n",
            "call": "simulate_base_population(10, 2000, 0.90, 0.15, 0.85, 2043)",
            "gold_call": "_oracle_simulate_base_population(10, 2000, 0.90, 0.15, 0.85, 2043)",
        },
        # --- boundary: rho = 0 (linkage equilibrium), narrow frequency range ---
        {
            "setup": "import numpy as np\n",
            "call": "simulate_base_population(4, 50, 0.0, 0.45, 0.55, 7)",
            "gold_call": "_oracle_simulate_base_population(4, 50, 0.0, 0.45, 0.55, 7)",
        },
        # --- edge: smallest legal shape, rho close to 1, extreme frequencies ---
        {
            "setup": "import numpy as np\n",
            "call": "simulate_base_population(2, 2, 0.999, 0.02, 0.98, 0)",
            "gold_call": "_oracle_simulate_base_population(2, 2, 0.999, 0.02, 0.98, 0)",
        },
    ]
