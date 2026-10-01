"""
Return the exact first and second derivatives of the specialization-averaged weight of every given candidate prey pair of one predator (the weights of step 05 for the pairs of step 03 at sigma = sigma_max, in the same order) with respect to the trophic position of one species, all other positions, the pair list and sigma_max held fixed. The weights are those of step 05 evaluated at the displaced positions, with the centre c = positions[predator] - 1 moving with the predator's own position, the pair list kept as given and every quantity that step 05 derives from the positions (its offsets, the thresholds of step 04 and the candidate count of the prior at each sigma) taken as the functions of the positions that steps 04 and 05 define. Return the exact derivatives, analytically and to full double precision (finite differences of any order do not meet the accuracy required of this step), as an array of shape (n_pairs, 2) whose columns are the first and the second derivative, in the order of the input pairs, zero for every pair whose weight does not depend on the species' position. Raise ValueError if the derivative does not exist: a pair whose two prey offsets are equal to within a relative tolerance of 1e-12 (its threshold switches between the two prey), or pairs whose thresholds are equal to within an absolute tolerance of 1e-12 and would move apart under the perturbation.

Empirical trophic positions carry measurement error and the reconstruction responds to it; the response of the pair weights, to first and to second order, is the core of that sensitivity and fixes the bias of the reconstruction under symmetric errors.

Returns
-------
numpy.ndarray of float64 with shape (n_pairs, 2): the exact first and second derivatives of the sigma-averaged pair weights with respect to the species' position.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def marginal_weight_derivatives(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                sigma_max: float, species: int) -> "numpy.ndarray":
    """Return the exact first and second derivatives of the specialization-averaged weight of every given candidate prey pair of one predator (the weights of step 05 for the pairs of step 03 at sigma = sigma_max, in the same order) with respect to the trophic position of one species, all other positions, the pair list and sigma_max held fixed. The weights are those of step 05 evaluated at the displaced positions, with the centre c = positions[predator] - 1 moving with the predator's own position, the pair list kept as given and every quantity that step 05 derives from the positions (its offsets, the thresholds of step 04 and the candidate count of the prior at each sigma) taken as the functions of the positions that steps 04 and 05 define. Return the exact derivatives, analytically and to full double precision (finite differences of any order do not meet the accuracy required of this step), as an array of shape (n_pairs, 2) whose columns are the first and the second derivative, in the order of the input pairs, zero for every pair whose weight does not depend on the species' position. Raise ValueError if the derivative does not exist: a pair whose two prey offsets are equal to within a relative tolerance of 1e-12 (its threshold switches between the two prey), or pairs whose thresholds are equal to within an absolute tolerance of 1e-12 and would move apart under the perturbation.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): the candidate pairs of the predator at sigma_max (step 03), non-empty.
    sigma_max : float
        Positive upper bound of the trophic specialization; every pair's threshold lies below it.
    species : int
        Index of the species whose trophic position is displaced (the predator itself or any other species).

    Returns
    -------
    derivatives : numpy.ndarray
        Array of shape (n_pairs, 2): column 0 the first and column 1 the second derivative of each pair's averaged weight with respect to the position of the species (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions, predator or species are invalid, pairs is empty, malformed or does not bracket the centre, a pair's threshold is not below sigma_max, a pair has prey offsets equal to within a relative tolerance of 1e-12, or pairs with thresholds equal to within an absolute tolerance of 1e-12 would move apart under the perturbation.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _primitive_dA(a: float, s: float) -> float:
    """d/dA of the step-05 primitive int_0^s exp(-A/x^2)/(2 pi x^2) dx, i.e. -int_0^s exp(-A/x^2)/(2 pi x^4) dx (u = 1/x)."""
    ra = math.sqrt(a)
    return -(math.sqrt(math.pi) * math.erfc(ra / s) / (4.0 * a * ra) + math.exp(-a / s ** 2) / (2.0 * a * s)) / (2.0 * math.pi)


def _primitive_dAA(a: float, s: float) -> float:
    """d^2/dA^2 of the step-05 primitive: int_0^s exp(-A/x^2)/(2 pi x^6) dx = (1/2pi) int_{1/s}^inf u^4 exp(-A u^2) du."""
    ra = math.sqrt(a)
    return (3.0 * math.sqrt(math.pi) * math.erfc(ra / s) / (8.0 * a ** 2 * ra)
            + math.exp(-a / s ** 2) * (1.0 / (2.0 * a * s ** 3) + 3.0 / (4.0 * a ** 2 * s))) / (2.0 * math.pi)


def _pair_density(a: float, s: float) -> float:
    """The product of the two Gaussian densities of a pair at specialization s: exp(-A/s^2) / (2 pi s^2)."""
    return math.exp(-a / s ** 2) / (2.0 * math.pi * s ** 2)


def _oracle_marginal_weight_derivatives(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                        sigma_max: float, species: int) -> "numpy.ndarray":
    """Exact first and second derivatives of the Eq 29 averaged weights (step 05) with respect to one trophic position."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if int(species) != species or not (0 <= species < y.size):
        raise ValueError("species must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0 or P.min() < 0 or P.max() >= y.size:
        raise ValueError("pairs must be a non-empty integer array of shape (n_pairs, 2) with valid species indices")
    i, m = int(predator), int(species)
    c = y[i] - 1.0
    smax = float(sigma_max)
    d_b = c - y[P[:, 0]]
    d_a = y[P[:, 1]] - c
    if np.any(d_b <= 0.0) or np.any(d_a <= 0.0):
        raise ValueError("every pair must hold one prey below and one prey above the predator's centre")
    if np.any(np.abs(d_b - d_a) <= 1e-12 * np.maximum(d_b, d_a)):
        raise ValueError("a pair with equal prey offsets has no derivative: its threshold switches between the two prey")
    thr = np.maximum(d_b, d_a) / 3.0
    if np.any(thr >= smax):
        raise ValueError("every pair must be a candidate at sigma_max (threshold below sigma_max)")
    # how the centre and the two offsets of every pair move with the position of the species (linearly)
    dc = 1.0 if m == i else 0.0
    dd_b = dc - (P[:, 0] == m).astype(np.float64)
    dd_a = (P[:, 1] == m).astype(np.float64) - dc
    A = 0.5 * (d_b ** 2 + d_a ** 2)
    dA = d_b * dd_b + d_a * dd_a                                  # first derivative of A
    d2A = dd_b ** 2 + dd_a ** 2                                   # second derivative of A (the offsets are linear)
    dthr = np.where(d_b > d_a, dd_b, dd_a) / 3.0                  # the threshold follows the farther prey, linearly
    # knots of the piecewise-constant candidate count: equal thresholds form one knot that must move as one
    order = np.argsort(thr, kind="stable")
    knots, members = [], []
    for k in order:
        if knots and abs(thr[k] - knots[-1]) <= 1e-12 * max(1.0, knots[-1]):
            members[-1].append(k)
        else:
            knots.append(thr[k]); members.append([k])
    dknots = np.zeros(len(knots))
    for g, mem in enumerate(members):
        moves = dthr[mem]
        if np.max(moves) - np.min(moves) > 1e-12:
            raise ValueError("pairs with equal thresholds that move apart under the perturbation have no derivative")
        dknots[g] = moves[0]
    group_of = np.zeros(P.shape[0], dtype=np.int64)
    for g, mem in enumerate(members):
        group_of[mem] = g
    counts = np.cumsum([len(mem) for mem in members])           # candidate pairs above each knot
    edges = np.array(knots + [smax])
    out = np.zeros((P.shape[0], 2))
    for k in range(P.shape[0]):
        g0 = group_of[k]
        a = A[k]
        sum_FA = sum_FAA = 0.0                                    # sums over the intervals above the pair's threshold
        bnd_g = bnd_gA = bnd_gs = 0.0                              # boundary terms: density, its A-derivative, its sigma-derivative
        for g in range(g0, len(knots)):
            lo, hi = edges[g], edges[g + 1]
            sum_FA += (_primitive_dA(a, hi) - _primitive_dA(a, lo)) / counts[g]
            sum_FAA += (_primitive_dAA(a, hi) - _primitive_dAA(a, lo)) / counts[g]
            if g > g0:
                # another group's knot moves: the prior 1/n jumps there from 1/n_before to 1/n_after
                coef = (1.0 / counts[g - 1] - 1.0 / counts[g]) * dknots[g]
                dens = _pair_density(a, lo)
                bnd_g += dens * coef
                bnd_gA += -dens / lo ** 2 * coef                  # d/dA of the density
                bnd_gs += dens * (2.0 * a / lo ** 3 - 2.0 / lo) * coef * dknots[g]   # d/dsigma of the density, times the knot speed
        # the pair's own threshold moves: the lower limit of its contribution
        lo0 = edges[g0]
        coef0 = -dknots[g0] / counts[g0]
        dens0 = _pair_density(a, lo0)
        bnd_g += dens0 * coef0
        bnd_gA += -dens0 / lo0 ** 2 * coef0
        bnd_gs += dens0 * (2.0 * a / lo0 ** 3 - 2.0 / lo0) * coef0 * dknots[g0]
        first = dA[k] * sum_FA + bnd_g
        # second order: the likelihood term twice differentiated, the moving knots inside it (once more each), and the
        # boundary terms differentiated through the exponent and through the knot position
        second = d2A[k] * sum_FA + dA[k] ** 2 * sum_FAA + 2.0 * dA[k] * bnd_gA + bnd_gs
        out[k, 0] = first / smax
        out[k, 1] = second / smax
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nsigma_max = 1.0 / 3.0\nspecies = 8\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nsigma_max = 1.0 / 3.0\nspecies = 7\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nsigma_max = 1.0 / 3.0\nspecies = 5\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 6\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nsigma_max = 1.0 / 3.0\nspecies = 4\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 11\nsigma_max = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, sigma_max)\nspecies = 9\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\npositions = np.array([1.0, 2.0, 2.2, 2.7999999, 3.5])\npredator = 4\nsigma_max = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, sigma_max)\nspecies = 2\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\npositions = np.array([1.0, 2.0, 2.2, 2.79999999, 3.5])\npredator = 4\nsigma_max = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, sigma_max)\nspecies = 2\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nsigma_max = 1.0 / 3.0\nspecies = 11\n",
            "call": "marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "gold_call": "_oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\npositions = np.array([1.0, 2.0, 2.2, 2.8, 3.5])\npredator = 4\nsigma_max = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, sigma_max)\nspecies = 2\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_exception_code(lambda: marginal_weight_derivatives(positions, predator, pairs, sigma_max, species))",
            "gold_call": "_exception_code(lambda: _oracle_marginal_weight_derivatives(positions, predator, pairs, sigma_max, species))",
        },
    ]
