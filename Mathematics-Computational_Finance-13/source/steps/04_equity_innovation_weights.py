"""
Compute the three weights that express the equity innovation in terms of the two factor innovations and one independent residual. The equity innovation is a linear combination of the variance-factor innovation, the rate-factor innovation, and a third component independent of both. The weights are fixed by three requirements: the combination must have unit variance, and its correlation with each of the two factor innovations must equal the prescribed value. Return the three weights in the order variance, rate, residual, with the residual weight taken non-negative.

Advancing an equity or fund process on a lattice whose state is carried by two other factors requires the equity shock to be expressed in the same coordinates. The two factor innovations are already produced by their own transitions, so the equity innovation is written as a linear combination of those two plus a further component that is independent of both and supplies whatever variance the factors cannot account for:

xi_S = w_V xi_V + w_r xi_r + w_perp eta

Here xi_V and xi_r are the centred unit-variance innovations of the two factors, correlated with one another at rho_vr, and eta is independent with mean zero and unit variance.

Three conditions determine the three weights. The combination must have unit variance, so that the equity shock is correctly scaled; and it must exhibit the prescribed correlation with each of the two factor innovations. Because xi_V and xi_r are themselves correlated, the variance of the combination carries a cross term in w_V w_r, and the correlation of xi_S with one factor picks up a contribution through the other. The first two conditions are linear in w_V and w_r and can be solved together; the third then fixes w_perp up to sign, and the positive root is taken by convention since eta is symmetric.

Not every triple of correlations is realisable. The three pairwise correlations must form a positive semidefinite matrix, and the quantity governing this is the determinant of that matrix. When it is negative, no such decomposition exists and the residual variance required would be negative. When it is zero, the equity innovation lies entirely in the span of the two factors and the residual weight vanishes. The residual weight is therefore not a small correction to be neglected: it is a measure of how much of the equity risk the two factors fail to span, and for typical equity-variance and equity-rate correlations it carries a substantial share of the total variance.

Returns
-------
np.ndarray of shape (3,): the weights on the variance-factor innovation, the rate-factor innovation, and the independent residual, in that order, as native floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equity_innovation_weights(rho_sv: float, rho_sr: float, rho_vr: float) -> np.ndarray:
    '''Weights expressing the equity innovation in the two factor innovations.

    Parameters
    ----------
    rho_sv : float
        Correlation between the equity and variance-factor innovations, in [-1, 1].
    rho_sr : float
        Correlation between the equity and rate-factor innovations, in [-1, 1].
    rho_vr : float
        Correlation between the two factor innovations, in [-1, 1].

    Returns
    -------
    weights : np.ndarray
        Shape (3,): the weight on the variance-factor innovation, the weight on
        the rate-factor innovation, and the non-negative weight on the
        independent residual, in that order.

    Raises
    ------
    ValueError
        If the arguments are not admissible correlations, or if they do not
        jointly describe a realisable dependence structure.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_equity_innovation_weights(rho_sv: float, rho_sr: float, rho_vr: float) -> np.ndarray:
    for name, val in (("rho_sv", rho_sv), ("rho_sr", rho_sr), ("rho_vr", rho_vr)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
        if abs(float(val)) > 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    a, b, c = float(rho_sv), float(rho_sr), float(rho_vr)

    den = 1.0 - c * c
    if den <= 1e-14:
        raise ValueError("the two factor innovations must not be perfectly correlated")

    det = 1.0 + 2.0 * a * b * c - a * a - b * b - c * c
    if det < 0.0:
        raise ValueError("the three correlations do not form a positive semidefinite matrix")

    w_v = (a - b * c) / den
    w_r = (b - a * c) / den
    w_perp = np.sqrt(max(det / den, 0.0))
    return np.array([w_v, w_r, w_perp], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    err = """import numpy as np
def run_model(**kw):
    a = dict(rho_sv=-0.70, rho_sr=-0.20, rho_vr=0.02)
    a.update(kw)
    try:
        equity_innovation_weights(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(rho_sv=-0.70, rho_sr=-0.20, rho_vr=0.02)
    a.update(kw)
    try:
        _oracle_equity_innovation_weights(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(-0.70, -0.20, 0.02)",
         "gold_call": "_oracle_equity_innovation_weights(-0.70, -0.20, 0.02)"},
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(-0.55, 0.30, -0.25)",
         "gold_call": "_oracle_equity_innovation_weights(-0.55, 0.30, -0.25)"},
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(0.40, 0.35, 0.30)",
         "gold_call": "_oracle_equity_innovation_weights(0.40, 0.35, 0.30)"},
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(-0.70, -0.20, 0.0)",
         "gold_call": "_oracle_equity_innovation_weights(-0.70, -0.20, 0.0)"},
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(0.0, 0.0, 0.45)",
         "gold_call": "_oracle_equity_innovation_weights(0.0, 0.0, 0.45)"},
        {"setup": "import numpy as np\n",
         "call": "equity_innovation_weights(-0.90, -0.40, 0.35)",
         "gold_call": "_oracle_equity_innovation_weights(-0.90, -0.40, 0.35)"},
        {"setup": err, "call": "run_model(rho_sv=-1.4)", "gold_call": "run_gold(rho_sv=-1.4)"},
        {"setup": err, "call": "run_model(rho_vr=1.0)", "gold_call": "run_gold(rho_vr=1.0)"},
        {"setup": err, "call": "run_model(rho_sv=-0.95, rho_sr=0.90, rho_vr=0.10)",
         "gold_call": "run_gold(rho_sv=-0.95, rho_sr=0.90, rho_vr=0.10)"},
    ]
