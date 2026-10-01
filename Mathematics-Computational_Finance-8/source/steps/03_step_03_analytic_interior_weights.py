"""
Evaluate the source's closed-form seven-node interior weights for the first and second derivatives on a uniform stencil.

On interior rows the scheme does not solve the integrated-kernel system; it uses the closed-form weights the source derives for the centred uniform stencil y_i + j h, j = -3, ..., 3. Each weight is the classical sixth-order finite-difference weight plus shape-parameter corrections. For the first derivative (antisymmetric, centre weight 0):

w_{-3} = -11192299/71680 h^3/c^4 + 3953/3360 h/c^2 - 1/(60 h),

w_{-2} = 12571299/17920 h^3/c^4 - 3953/840 h/c^2 + 3/(20 h),

w_{-1} = -13398699/14336 h^3/c^4 + 3953/672 h/c^2 - 3/(4 h),

and w_{+j} = -w_{-j}. For the second derivative (symmetric):

w_{+-3} = (149380759993 h^2 - 704267904 c^2)/(371589120 c^4) + 1/(90 h^2),

w_{+-2} = (704267904 c^2 - 162082276345 h^2)/(61931520 c^4) - 3/(20 h^2),

w_{+-1} = (848515930781 h^2 - 3521339520 c^2)/(123863040 c^4) + 3/(2 h^2),

w_0 = (3521339520 c^2 - 861217447133 h^2)/(92897280 c^4) - 49/(18 h^2).

The orientation is fixed by the leading terms, which must reproduce the classical centred stencils; the c-dependent corrections do not vanish when c is tied to h.

Returns
-------
np.ndarray, float, shape (2, 7): row 0 the first-derivative weights and row 1 the second-derivative weights for offsets j = -3, ..., 3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def analytic_interior_weights(h: float, c: float) -> np.ndarray:
    '''Closed-form interior weights of the integrated-kernel scheme.

    Parameters
    ----------
    h : float
        Uniform node spacing, h > 0.
    c : float
        Shape parameter of the kernel, c > 0.

    Returns
    -------
    weights : np.ndarray
        Shape (2, 7) float array. Row 0 holds the first-derivative weights and
        row 1 the second-derivative weights for the offsets -3h, ..., 3h.

    Raises
    ------
    ValueError
        If h or c is not a finite positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((2, 7), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_analytic_interior_weights(h: float, c: float) -> np.ndarray:
    import numpy as np

    for name, value in (("h", h), ("c", c)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    h = float(h)
    c = float(c)

    # First derivative, negative-side weights; the positive side is their
    # negative and the centre weight vanishes.
    b3 = -11192299.0 / 71680.0 * h ** 3 / c ** 4 + 3953.0 / 3360.0 * h / c ** 2 - 1.0 / (60.0 * h)
    b2 = 12571299.0 / 17920.0 * h ** 3 / c ** 4 - 3953.0 / 840.0 * h / c ** 2 + 3.0 / (20.0 * h)
    b1 = -13398699.0 / 14336.0 * h ** 3 / c ** 4 + 3953.0 / 672.0 * h / c ** 2 - 3.0 / (4.0 * h)
    first = np.array([b3, b2, b1, 0.0, -b1, -b2, -b3])

    # Second derivative, symmetric about the centre.
    s3 = (149380759993.0 * h ** 2 - 704267904.0 * c ** 2) / (371589120.0 * c ** 4) + 1.0 / (90.0 * h ** 2)
    s2 = (704267904.0 * c ** 2 - 162082276345.0 * h ** 2) / (61931520.0 * c ** 4) - 3.0 / (20.0 * h ** 2)
    s1 = (848515930781.0 * h ** 2 - 3521339520.0 * c ** 2) / (123863040.0 * c ** 4) + 3.0 / (2.0 * h ** 2)
    s0 = (3521339520.0 * c ** 2 - 861217447133.0 * h ** 2) / (92897280.0 * c ** 4) - 49.0 / (18.0 * h ** 2)
    second = np.array([s3, s2, s1, s0, s1, s2, s3])
    return np.vstack([first, second]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the pricing grid spacing h = 5 with c = 4h.
        {
            "setup": """import numpy as np
""",
            "call": "analytic_interior_weights(5.0, 20.0)",
            "gold_call": "_oracle_analytic_interior_weights(5.0, 20.0)",
        },
        # --- Pinned limit: for c far larger than h the corrections are below
        # the comparison tolerance and the classical sixth-order centred
        # stencils must appear, with the orientation of a forward derivative.
        {
            "setup": """import numpy as np
EXPECTED = np.array([[-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60],
                     [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]])
""",
            "call": "analytic_interior_weights(1.0, 1e7)",
            "gold_call": "EXPECTED",
        },
        # --- Pinned moments at c = 4h, h = 1: the closed forms reproduce
        # constants, linear and quadratic functions exactly (moments 0, 1, 2 of
        # the second-derivative row are 0, 0, 2 and moment 1 of the first row
        # is 1), while the fourth moment of the second row, a signature of the
        # source's correction terms, equals -19.22715861002604.
        {
            "setup": """import numpy as np
j = np.arange(-3.0, 4.0)
def moment_digest(fn):
    w = fn(1.0, 4.0)
    return np.round(np.array([w[1].sum(), (w[1] * j).sum(), (w[1] * j ** 2).sum(),
                              (w[0] * j).sum(), (w[1] * j ** 4).sum()]), 8)
EXPECTED = np.array([0.0, 0.0, 2.0, 1.0, -19.22715861])
""",
            "call": "moment_digest(analytic_interior_weights)",
            "gold_call": "EXPECTED",
        },
        # --- Boundary: a fine spacing with the same c/h ratio, where every
        # weight scales like 1/h or 1/h^2.
        {
            "setup": """import numpy as np
""",
            "call": "analytic_interior_weights(1e-3, 4e-3)",
            "gold_call": "_oracle_analytic_interior_weights(1e-3, 4e-3)",
        },
        # --- Edge: c smaller than h, where the h^2/c^4 corrections dominate.
        {
            "setup": """import numpy as np
""",
            "call": "analytic_interior_weights(2.0, 0.5)",
            "gold_call": "_oracle_analytic_interior_weights(2.0, 0.5)",
        },
        # --- Invalid: non-positive spacing ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        analytic_interior_weights(0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_analytic_interior_weights(0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: infinite shape parameter ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        analytic_interior_weights(1.0, float('inf'))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_analytic_interior_weights(1.0, float('inf'))
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
