"""
Step 5: the weight of one window piece under the two sensitivity curves.

The developmental window is cut into pieces at its two edges and at every reading inside it. Each piece carries one temperature, the held temperature of its left edge, and one span of standardised size, running from zero at the window start to one at the window end.

Two curves turn that pair into a weight. The first is a thermal reaction norm of sexualization, the same four-parameter form as the development rate of step 1 with its rate at the reference temperature fixed at 100. It measures how strongly the gonad is committed at that temperature. The second is a fitted beta density over the standardised size, and its sensitivity over the piece's span enters the weight.

Returns
-------
weight : np.ndarray, shape (n_pieces,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sexualization_weight(
    temperature: "numpy.typing.ArrayLike",
    standardised_edges: "numpy.typing.ArrayLike",
    DHA: float = -719.576256,
    DHH: float = 685.203574,
    T12H: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    Rho25: float = 100.0,
) -> "numpy.ndarray":
    """Weight of each window piece from the sexualization curve and the beta mass.

    Parameters
    ----------
    temperature : array-like
        Held temperature in degrees Celsius of each piece, one entry per piece.
    standardised_edges : array-like
        Standardised size at the piece edges, one more entry than there are
        pieces, strictly increasing inside the closed unit interval. The task
        pipeline supplies the full window, from 0 at the window start to 1 at the
        window end, and any increasing subinterval is also valid.
    DHA, DHH, T12H : float, optional
        Parameters of the sexualization reaction norm (-719.576256, 685.203574,
        552.204465).
    shape1, shape2 : float, optional
        Shapes of the fitted beta density over the standardised size (1.725323,
        0.802045).
    Rho25 : float, optional
        Rate of the sexualization norm at the reference temperature of 298 K
        (100.0, fixed by the fitted model).

    Returns
    -------
    weight : numpy.ndarray
        One weight per piece. Each entry is the sexualization norm at the piece
        temperature times the fitted beta sensitivity over the piece's span of
        standardised size.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional and finite, if the edges number
        fewer than two or are not strictly increasing inside the closed unit
        interval, if the edges do not carry one more entry than the
        temperatures, or if a shape or the fixed rate is not positive.
    """
    return weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import betainc


def _oracle_sexualization_weight(
    temperature: "numpy.typing.ArrayLike",
    standardised_edges: "numpy.typing.ArrayLike",
    DHA: float = -719.576256,
    DHH: float = 685.203574,
    T12H: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    Rho25: float = 100.0,
) -> "numpy.ndarray":
    """Reference implementation for sexualization_weight."""
    T = np.asarray(temperature, dtype=float)
    edges = np.asarray(standardised_edges, dtype=float)
    if T.ndim != 1 or edges.ndim != 1 or edges.size != T.size + 1:
        raise ValueError("standardised_edges must be one longer than temperature")
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(edges))):
        raise ValueError("temperature and standardised_edges must be finite")
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("standardised_edges must strictly increase")
    if not (0.0 <= float(edges[0]) and float(edges[-1]) <= 1.0):
        raise ValueError("standardised_edges must lie inside the closed unit interval")
    if not (float(shape1) > 0.0 and float(shape2) > 0.0 and float(Rho25) > 0.0):
        raise ValueError("shape1, shape2 and Rho25 must be positive")
    norm = _oracle_development_rate(T, DHA, DHH, T12H, Rho25)
    norm = np.where(norm <= 0.0, 0.001, norm)
    mass = betainc(float(shape1), float(shape2), edges[1:]) - betainc(float(shape1), float(shape2), edges[:-1])
    return norm * mass

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return list of test case specifications."""
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        sexualization_weight(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_sexualization_weight(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: five equal pieces of the window, the temperature rising across them.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([29.0, 29.5, 30.0, 30.5, 31.0], dtype=float)\n"
                  "EDGES = np.linspace(0.0, 1.0, 6)\n",
         "call": "sexualization_weight(TEMPS, EDGES)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES)"},
        # Boundary: the whole window is one piece.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([30.0], dtype=float)\n"
                  "EDGES = np.array([0.0, 1.0], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES)"},
        # Edge: one very narrow piece beside wide ones, where the exact mass over the
        # span is what keeps the weight independent of how the window is split.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([29.8, 30.1, 30.4, 30.7], dtype=float)\n"
                  "EDGES = np.array([0.0, 1e-6, 0.4, 0.9, 1.0], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES)"},
        # Edge: a cold piece, where the sexualization norm is small.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([12.0, 18.0, 25.0], dtype=float)\n"
                  "EDGES = np.array([0.0, 0.25, 0.75, 1.0], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES)"},
        # Edge: a subinterval of the standardised window with non-default beta
        # shapes and a halved reference rate, where the exact mass over each span
        # and the rate scale of step 1 both matter.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([26.0, 30.0], dtype=float)\n"
                  "EDGES = np.array([0.2, 0.55, 0.9], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES, shape1=2.5, shape2=1.5, Rho25=50.0)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES, shape1=2.5, shape2=1.5, Rho25=50.0)"},
        # Edge: a narrow interior span between wide ones, where a pointwise beta
        # density at one edge differs from the exact mass over the span.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([29.8, 30.1], dtype=float)\n"
                  "EDGES = np.array([0.4, 0.45, 0.9], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES, shape1=2.5, shape2=1.5)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES, shape1=2.5, shape2=1.5)"},
        # Edge: the whole window with a second beta shape pair, where a fixed
        # shape or a fixed reference rate departs from the fitted model.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([28.0, 31.0], dtype=float)\n"
                  "EDGES = np.array([0.0, 0.5, 1.0], dtype=float)\n",
         "call": "sexualization_weight(TEMPS, EDGES, shape1=0.8, shape2=2.5)",
         "gold_call": "_oracle_sexualization_weight(TEMPS, EDGES, shape1=0.8, shape2=2.5)"},
        # Invalid: edges that are not increasing.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([29.0, 30.0], dtype=float)\n"
                  "EDGES = np.array([0.0, 0.6, 0.4], dtype=float)\n" + invalid,
         "call": "run_model(temperature=TEMPS, standardised_edges=EDGES)",
         "gold_call": "run_gold(temperature=TEMPS, standardised_edges=EDGES)"},
        # Invalid: an edge outside the unit interval.
        {"setup": "import numpy as np\n"
                  "TEMPS = np.array([29.0, 30.0], dtype=float)\n"
                  "EDGES = np.array([0.0, 0.6, 1.2], dtype=float)\n" + invalid,
         "call": "run_model(temperature=TEMPS, standardised_edges=EDGES)",
         "gold_call": "run_gold(temperature=TEMPS, standardised_edges=EDGES)"},
    ]
