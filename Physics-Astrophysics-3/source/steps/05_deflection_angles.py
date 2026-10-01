"""
Step 05: the deflection angle of the local field direction from the background.

Definition

----------

The background direction is xhat. At each grid site the deflection angle is



    theta = arccos( B_x / |B| ),



with |B| = sqrt(B_x**2 + B_y**2 + B_z**2) formed pointwise as the sum of

squares over the component axis. Dividing by |B| takes the overall field

strength out of the ratio, so theta reports geometry alone. The result is

returned in DEGREES, not radians, elementwise on the (N, N, N) spatial grid,

and takes values in [0, 180]: theta = 0 where the field is exactly aligned with

xhat, 90 degrees where it is purely transverse, and 180 degrees where it is

exactly anti-aligned.



Floating-point convention

-------------------------

The ratio B_x / |B| is mathematically confined to [-1, 1], and that bound is

attained: wherever the field is exactly axial the ratio is exactly +1 (aligned)

or exactly -1 (anti-aligned). Such sites are not exotic here, because the

background is uniform and the transverse part is a localised envelope that is

zero to machine precision over most of the cube. The consequence is that arccos

is routinely evaluated with zero margin, sitting precisely on the edge of its

domain, where a single ulp in the wrong direction is fatal: np.arccos returns

NaN for any argument outside [-1, 1], and one NaN poisons a maximum taken over

the cube. The magnitude is a square root of a sum of squares, so the quotient is

only guaranteed to stay inside the interval when that reduction is the exact one

used here; a different summation order, a reduced-precision or fused

accumulation, or a field handed over by an upstream step can leave the quotient

a few ulp beyond the endpoint. The ratio is therefore clipped to [-1.0, 1.0]

before the arccos is taken. The clip is purely a floating-point guard: it can

only fold an overshoot back onto the endpoint it already belongs on, and it

never alters a ratio that lies genuinely inside the interval, so it changes no

physically meaningful angle.



Inputs

------

B: (3, N, N, N) float array, the vector field on the periodic unit-cube grid,

   component-first, every grid site of strictly positive magnitude.



Returns

-------

theta: (N, N, N) float array, the deflection angle from xhat in DEGREES,

   elementwise, taking values in [0, 180].

Returns
-------
theta : numpy.ndarray of shape (N, N, N), deflection angles in degrees.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deflection_angles(B: np.ndarray) -> np.ndarray:
    '''Deflection angle in degrees between the local field and xhat.

    Parameters
    ----------
    B : np.ndarray
        (3, N, N, N) component-first vector field on the periodic unit-cube
        grid. The shape is validated as in step 03: four dimensions, a leading
        axis of length 3, and three equal trailing axes of length N >= 1.
        Every grid site must have strictly positive magnitude, because the
        field direction, and hence the angle, is undefined at a site where the
        vector vanishes.

    Returns
    -------
    theta : np.ndarray
        (N, N, N) array of deflection angles in DEGREES, computed elementwise
        as np.degrees(np.arccos(np.clip(B[0] / mag, -1.0, 1.0))) with
        mag the pointwise magnitude. The clip is a floating-point guard. The
        ratio B[0] / mag reaches exactly +/-1 at grid sites where the field is
        exactly axial, which is common here because the transverse envelope
        vanishes over most of the cube, so arccos is evaluated with no margin
        at the edge of its domain. Any floating-point excursion a few ulp
        beyond 1 in absolute value, from a differently ordered or
        differently rounded magnitude, makes np.arccos return NaN, and a single
        NaN would poison the maximum over the cube. Clipping folds such an
        excursion back onto the endpoint and leaves every interior ratio
        untouched.

    Raises
    ------
    ValueError
        If B does not have shape (3, N, N, N) with N >= 1, or if any grid site
        has exactly zero magnitude.
        N = 1 is admissible: the angle is evaluated pointwise and couples
        no neighbouring samples, unlike the spectral steps, which require
        N >= 2.
    '''
    return theta  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deflection_angles(B: np.ndarray) -> np.ndarray:
    F = np.asarray(B, dtype=float)
    if F.ndim != 4:
        raise ValueError("B must be a 4-dimensional array of shape (3, N, N, N)")
    if F.shape[0] != 3:
        raise ValueError("B must carry 3 components on the leading axis")
    n = int(F.shape[1])
    if F.shape[2] != n or F.shape[3] != n:
        raise ValueError("B must be cubic: the three trailing axes must be equal")
    if n < 1:
        raise ValueError("N must be at least 1")

    mag = np.sqrt(np.sum(F * F, axis=0))
    if np.any(mag == 0.0):
        raise ValueError(
            "B has a grid site of zero magnitude; the deflection angle is "
            "undefined there")

    # The clip guards arccos against ratios that float a few ulp past +/-1 at
    # exactly axial sites; it never moves a ratio interior to [-1, 1].
    return np.degrees(np.arccos(np.clip(F[0] / mag, -1.0, 1.0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the N = 8 seed field at the reference envelope ---
        {
            "setup": """import numpy as np
N = 8
A = 20.0
sigma = 1.0 / 30.0
kx = 4
c = np.arange(N) / N
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
phi = 2.0 * np.pi * kx * X
B = np.zeros((3, N, N, N))
B[0] = 1.0
B[1] = np.cos(phi) * env
B[2] = np.sin(phi) * env
""",
            "call": "deflection_angles(B)",
            "gold_call": "_oracle_deflection_angles(B)",
        },
        # --- Normal: a broad-envelope seed so most sites carry a real angle ---
        {
            "setup": """import numpy as np
N = 10
A = 3.0
sigma = 0.15
kx = 2
c = np.arange(N) / N
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
phi = 2.0 * np.pi * kx * X
B = np.zeros((3, N, N, N))
B[0] = 1.0
B[1] = np.cos(phi) * env
B[2] = np.sin(phi) * env
""",
            "call": "deflection_angles(B)",
            "gold_call": "_oracle_deflection_angles(B)",
        },
        # --- Boundary: a purely axial field, every angle exactly zero ---
        {
            "setup": """import numpy as np
N = 6
B = np.zeros((3, N, N, N))
B[0] = 2.5
""",
            "call": "deflection_angles(B)",
            "gold_call": "_oracle_deflection_angles(B)",
        },
        # --- Edge: field along -xhat everywhere, 180 degrees, clip at the -1 end ---
        {
            "setup": """import numpy as np
N = 5
B = np.zeros((3, N, N, N))
B[0] = -1.0
""",
            "call": "deflection_angles(B)",
            "gold_call": "_oracle_deflection_angles(B)",
        },
        # --- Edge: mixed field with both clip endpoints and a transverse site ---
        {
            "setup": """import numpy as np
N = 4
rng = np.random.default_rng(11)
B = rng.standard_normal((3, N, N, N))
B[:, 0, 0, 0] = np.array([-7.0, 0.0, 0.0])
B[:, 1, 1, 1] = np.array([9.0, 0.0, 0.0])
B[:, 2, 2, 2] = np.array([0.0, 4.0, 0.0])
B[:, 3, 3, 3] = np.array([0.0, 0.0, -4.0])
""",
            "call": "deflection_angles(B)",
            "gold_call": "_oracle_deflection_angles(B)",
        },
        # --- Invalid: only three dimensions ---
        {
            "setup": """import numpy as np
B = np.ones((3, 8, 8))
def run_model():
    try:
        deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of components on the leading axis ---
        {
            "setup": """import numpy as np
B = np.ones((4, 6, 6, 6))
def run_model():
    try:
        deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: trailing axes not a cube ---
        {
            "setup": """import numpy as np
B = np.ones((3, 6, 5, 6))
def run_model():
    try:
        deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a grid site of exactly zero magnitude ---
        {
            "setup": """import numpy as np
N = 6
B = np.zeros((3, N, N, N))
B[0] = 1.0
B[1] = 0.25
B[:, 2, 3, 4] = 0.0
def run_model():
    try:
        deflection_angles(B)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_deflection_angles(B)
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
