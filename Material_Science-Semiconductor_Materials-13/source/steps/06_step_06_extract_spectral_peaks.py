"""
Return the emission rate and the enclosed spectral weight of each of the strongest features of an emission-rate spectral density, partitioning the rate grid at the minima that separate those features.

A regularized non-negative inversion returns each discrete level not as a single grid weight but as a short cluster of adjacent non-zero weights, because the true rate almost never coincides with a grid point and the penalty spreads the weight over its neighbours. Reading the level's rate off the grid point of largest weight therefore quantises the answer to the grid spacing, which over several decades resolved by a modest number of points is a systematic error of several percent, far larger than the precision an activation energy needs. Taking instead the weighted geometric mean of the grid rates over the whole cluster recovers the rate to well inside one grid spacing, and it is the natural estimator because the grid is geometric and the eventual regression is linear in the logarithm of the rate.




Which grid points belong to which cluster is decided by the minima of the density. Between two genuine features the density falls to a minimum, and this benchmark assigns the minimum grid point once, to the feature on its right. The resulting half-open cluster slices partition the axis without overlap or gap. A partition by a fixed fractional threshold below each peak does not have that property: when a weak feature sits beside a strong one, the threshold set by the weak peak lies below the flank of the strong one, so its integration window swallows part of its neighbour and inflates the weak amplitude.




The total weight enclosed by a cluster is the spectral estimate of that level's capacitance deflection, since the spectral density integrates to the total deflection of the transient. It is reported here because it is the classical route to the individual amplitudes, but it inherits whatever reshaping the regularisation imposed, which is exactly the reshaping that biases weak features sitting beside strong ones. Selecting the features by descending peak height, rather than by enclosed weight, keeps the selection stable when one level is far weaker than the other.

Returns
-------
np.ndarray of shape (n_found, 2), float: feature emission rate (1/s) and enclosed spectral weight (pF), one row per feature.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_spectral_peaks(density: np.ndarray, emission_grid: np.ndarray,
                           n_peaks: int) -> np.ndarray:
    """Return the rate and the enclosed weight of the strongest spectral features.

    Parameters
    ----------
    density : np.ndarray
        Non-negative spectral density sampled on the emission-rate grid.
    emission_grid : np.ndarray
        Strictly increasing, strictly positive emission rates in 1/s, same
        length as density.
    n_peaks : int
        Number of features to return (n_peaks >= 1).

    Returns
    -------
    features : np.ndarray
        Array of shape (n_found, 2) sorted by increasing emission rate, whose
        columns are the emission rate of the feature in 1/s and the spectral
        weight enclosed by it in pF. n_found is the smaller of n_peaks and the
        number of interior local maxima of the density.

    Notes
    -----
    A grid point that is the separating minimum between adjacent retained
    maxima belongs to the feature on its right. Every grid point is therefore
    included in exactly one cluster slice.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return features  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_extract_spectral_peaks(density: np.ndarray, emission_grid: np.ndarray,
                                   n_peaks: int) -> np.ndarray:
    import numpy as np

    density = np.asarray(density, dtype=float)
    emission_grid = np.asarray(emission_grid, dtype=float)

    if not (isinstance(n_peaks, (int, np.integer)) and not isinstance(n_peaks, bool)
            and int(n_peaks) >= 1):
        raise ValueError("n_peaks must be an integer >= 1")
    if density.ndim != 1 or density.shape != emission_grid.shape:
        raise ValueError("density and emission_grid must be one-dimensional and equal length")
    if density.size < 3:
        raise ValueError("density must hold at least 3 grid points")
    if np.any(emission_grid <= 0.0) or np.any(np.diff(emission_grid) <= 0.0):
        raise ValueError("emission_grid must be strictly positive and strictly increasing")
    if np.any(density < 0.0):
        raise ValueError("density must be non-negative")

    n_grid = density.size
    log_rate = np.log(emission_grid)

    # Interior local maxima, kept by descending height so that a weak feature
    # beside a strong one is still selected in the right order.
    maxima = [j for j in range(1, n_grid - 1)
              if density[j] > density[j - 1] and density[j] >= density[j + 1]
              and density[j] > 0.0]
    maxima.sort(key=lambda j: -density[j])
    maxima = sorted(maxima[:int(n_peaks)])

    if not maxima:
        return np.zeros((0, 2), dtype=float)

    # Treat bounds as half-open slice starts.  A separating minimum is the
    # first point of the cluster on its right, so no point is double counted.
    bounds = [0]
    for left, right in zip(maxima[:-1], maxima[1:]):
        bounds.append(left + int(np.argmin(density[left:right + 1])))
    bounds.append(n_grid)

    rows = []
    for m in range(len(maxima)):
        low, high = bounds[m], bounds[m + 1]
        segment = density[low:high]
        weight = segment.sum()
        rate = float(np.exp((segment * log_rate[low:high]).sum() / weight))
        rows.append((rate, float(weight)))

    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: two well separated clusters of very unequal weight ---
        {
            "setup": """import numpy as np
grid = np.geomspace(0.1, 1.0e5, 150)
density = np.zeros(150)
density[[70, 71]] = [0.56, 0.19]
density[[110, 111]] = [0.029, 0.021]
""",
            "call": "extract_spectral_peaks(density, grid, 2) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 2) + 1000.0",
        },
        # --- Valid: the recovered rate of the weak feature alone ---
        {
            "setup": """import numpy as np
grid = np.geomspace(0.1, 1.0e5, 150)
density = np.zeros(150)
density[[70, 71]] = [0.56, 0.19]
density[[110, 111]] = [0.029, 0.021]
""",
            "call": "extract_spectral_peaks(density, grid, 2) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 2) + 1000.0",
        },
        # --- Valid: broad overlapping features separated only by a shallow minimum ---
        {
            "setup": """import numpy as np
grid = np.geomspace(1.0, 1.0e4, 80)
x = np.arange(80.0)
density = 0.7 * np.exp(-0.5 * ((x - 30.0) / 3.0) ** 2) + 0.2 * np.exp(-0.5 * ((x - 45.0) / 4.0) ** 2)
""",
            "call": "extract_spectral_peaks(density, grid, 2) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 2) + 1000.0",
        },
        # --- Boundary: a single feature requested from a two-feature density ---
        {
            "setup": """import numpy as np
grid = np.geomspace(0.1, 1.0e5, 150)
density = np.zeros(150)
density[[70, 71]] = [0.56, 0.19]
density[[110, 111]] = [0.029, 0.021]
""",
            "call": "extract_spectral_peaks(density, grid, 1) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 1) + 1000.0",
        },
        # --- Edge: an all-zero density has no interior maximum ---
        {
            "setup": """import numpy as np
grid = np.geomspace(1.0, 1.0e4, 40)
density = np.zeros(40)
""",
            "call": "extract_spectral_peaks(density, grid, 2) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 2) + 1000.0",
        },
        # --- Edge: more features requested than the density contains ---
        {
            "setup": """import numpy as np
grid = np.geomspace(1.0, 1.0e4, 40)
density = np.zeros(40)
density[15] = 0.4
""",
            "call": "extract_spectral_peaks(density, grid, 5) + 1000.0",
            "gold_call": "_oracle_extract_spectral_peaks(density, grid, 5) + 1000.0",
        },
        # --- Invalid: a negative spectral density ---
        {
            "setup": """import numpy as np
grid = np.geomspace(1.0, 1.0e4, 10)
def run_model():
    try:
        extract_spectral_peaks(np.linspace(-1.0, 1.0, 10), grid, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_spectral_peaks(np.linspace(-1.0, 1.0, 10), grid, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero features requested ---
        {
            "setup": """import numpy as np
grid = np.geomspace(1.0, 1.0e4, 10)
density = np.zeros(10)
density[4] = 1.0
def run_model():
    try:
        extract_spectral_peaks(density, grid, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extract_spectral_peaks(density, grid, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
