"""
Evaluate the covariant metric components of the spindle-deformed black hole at a point, in

the frame reached by applying both mandatory coordinate transformations.

The line element of the spindle-deformed black hole is written in the

coordinates (t, r, x, phi) with x = cos(theta), as a sum of two squared

one-forms plus diagonal r and x pieces:



    ds^2 = -(Q / (LL Om^4)) [ (a x^2 / P0) dphi + C dt ]^2

           + ((1 - x^2) P / (Om^4 LL)) [ (r^2 / P0) dphi + D dt ]^2

           + LL [ dr^2 / Q + dx^2 / ((1 - x^2) P) ]



with the two dt coefficients



    C = (a^2 x^2 MM + LL r^2 Om^2 (1 - x^2) P) / (r^4 (1 - x^2) P - a^2 x^4 Q)

    D = a (r^2 MM + LL Q Om^2 x^2)             / (r^4 (1 - x^2) P - a^2 x^4 Q)



and the metric functions (MM and LL are the source's blackboard-bold M and L,

not the physical mass or a length; Om = sqrt(Om^2) enters MM and LL linearly):



    Sigma = r^2 + a^2 x^2

    Delta = (1 - B^2 m^2 I2 / I1^2) r^2 - 2 m (I2 / I1) r + a^2

    P     = 1 + B^2 (m^2 I2 / I1^2 - a^2) x^2

    Q     = (1 + B^2 r^2) Delta

    Om^2  = (1 + B^2 r^2) - B^2 x^2 Delta

    MM = (1 / (2 I1^3 Om)) {

            Om [ I1 I2 m r x^2 ( B^2 m r (B^2 r^2 + x^2) + 2 I1 (B^2 r^2 + 1) )

                 - I1^3 Sigma (B^2 r^2 x^2 + 1) ]

          + (B^2 r^2 + 1) [ I1^3 Sigma (a^2 B^2 x^2 - 1)

                 + I2 m r x^2 ( I2 B^4 m^2 r^2 x^2 + 3 I1 I2 B^2 m r x^2

                                - I1^2 (3 a^2 B^2 x^2 + B^2 r^2 x^2 - 2) ) ] }

    LL = (1 / (4 I1^2 Om^4)) {

            Sigma [ B^4 r^2 m^2 I2 x^2

                    + I1^2 ( 2 + a^2 B^4 r^2 x^2 + B^2 (r^2 - a^2 x^2 - r^2 x^2) ) ]

          + B^2 r m I2 x^2 ( 2 a^2 I1 x^2 + B^2 r^3 m I2 x^2 + 2 r^2 I1 (2 - a^2 B^2 x^2) )

          + 2 Om I1 ( Sigma I1 + B^2 r^3 m I2 x^2 ) }



That original form is not usable as it stands.  Two linear coordinate

transformations must be applied, in order:



1. The azimuthal shift t = t' - k phi, with k = 2 a / ((1 + sqrt(I2)) P0),

   which removes naked closed timelike curves.

2. The Killing-frame normalisation t' = lam1 t'', phi = phi'' + lam2 b t''.



The returned components are those of the doubly-transformed coordinates

(t'', phi''), written (t, phi) from here on.



Conventions fixed for this step:

* The components are returned in the coordinate order (t, r, x, phi), with

  g_xx the coefficient of dx^2.

* The implementation must remain valid for complex r and x so that later

  steps can take derivatives by complex-step differentiation.

Returns
-------
np.ndarray of shape (5,), the covariant components [g_tt, g_rr, g_xx, g_phph, g_tph]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ml_metric_components(r: float, x: float, params: np.ndarray, b: float) -> np.ndarray:
    '''Covariant metric components of the spindle-deformed black hole.

    Parameters
    ----------
    r : float
        Boyer-Lindquist-like radial coordinate.
    x : float
        Polar coordinate x = cos(theta), with |x| < 1.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.

    Returns
    -------
    result : np.ndarray
        Array of shape (5,), the components [g_tt, g_rr, g_xx, g_phph, g_tph]
        in the doubly-transformed frame.

    Raises
    ------
    ValueError
        If r is not positive or |x| is not less than 1.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ml_metric_components(r: float, x: float, params: np.ndarray, b: float) -> np.ndarray:
    """Reference implementation."""
    if np.real(r) <= 0.0:
        raise ValueError("r must be positive")
    if abs(np.real(x)) >= 1.0:
        raise ValueError("|x| must be less than 1")
    m, a, i1, i2, p0 = params[0], params[1], params[2], params[3], params[4]
    lam1, lam2 = params[8], params[9]
    sig = r * r + a * a * x * x
    dl = (1.0 - b * b * m * m * i2 / (i1 * i1)) * r * r - 2.0 * m * i2 / i1 * r + a * a
    q = (1.0 + b * b * r * r) * dl
    om2 = (1.0 + b * b * r * r) - b * b * x * x * dl
    om = np.sqrt(om2)
    pf = 1.0 + b * b * (m * m * i2 / (i1 * i1) - a * a) * x * x
    mf = (1.0 / (2.0 * i1 ** 3 * om)) * (
        om * (i1 * i2 * m * r * x * x
              * (b * b * m * r * (b * b * r * r + x * x) + 2.0 * i1 * (b * b * r * r + 1.0))
              - i1 ** 3 * sig * (b * b * r * r * x * x + 1.0))
        + (b * b * r * r + 1.0)
        * (i1 ** 3 * sig * (a * a * b * b * x * x - 1.0)
           + i2 * m * r * x * x
           * (i2 * b ** 4 * m * m * r * r * x * x + 3.0 * i1 * i2 * b * b * m * r * x * x
              - i1 * i1 * (3.0 * a * a * b * b * x * x + b * b * r * r * x * x - 2.0))))
    lf = (1.0 / (4.0 * i1 * i1 * om2 * om2)) * (
        sig * (b ** 4 * r * r * m * m * i2 * x * x
               + i1 * i1 * (2.0 + a * a * b ** 4 * r * r * x * x
                            + b * b * (r * r - a * a * x * x - r * r * x * x)))
        + b * b * r * m * i2 * x * x
        * (2.0 * a * a * i1 * x * x + b * b * r ** 3 * m * i2 * x * x
           + 2.0 * r * r * i1 * (2.0 - a * a * b * b * x * x))
        + 2.0 * om * i1 * (sig * i1 + b * b * r ** 3 * m * i2 * x * x))
    den = r ** 4 * (1.0 - x * x) * pf - a * a * x ** 4 * q
    a_t = (a * a * x * x * mf + lf * r * r * om2 * (1.0 - x * x) * pf) / den
    b_t = a * (r * r * mf + lf * q * om2 * x * x) / den
    c1, c2 = a * x * x / p0, r * r / p0
    f1, f2 = -q / (lf * om2 * om2), (1.0 - x * x) * pf / (om2 * om2 * lf)
    g_tt = f1 * a_t * a_t + f2 * b_t * b_t
    g_tp = f1 * a_t * c1 + f2 * b_t * c2
    g_pp = f1 * c1 * c1 + f2 * c2 * c2
    k = 2.0 * a / ((1.0 + np.sqrt(i2)) * p0)
    j_tt = lam1 - k * lam2 * b
    j_tp = -k
    j_pt = lam2 * b
    return np.array([
        j_tt * j_tt * g_tt + 2.0 * j_tt * j_pt * g_tp + j_pt * j_pt * g_pp,
        lf / q,
        lf / ((1.0 - x * x) * pf),
        j_tp * j_tp * g_tt + 2.0 * j_tp * g_tp + g_pp,
        j_tt * j_tp * g_tt + (j_tt + j_tp * j_pt) * g_tp + j_pt * g_pp,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Candidate and oracle start from separate equivalent inputs."""
    exception_setup = 'def _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n'
    return [
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)',
            "call": 'ml_metric_components(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01)))',
            "gold_call": '_oracle_ml_metric_components(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par, 0.01)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar0 = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.0)',
            "call": 'ml_metric_components(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par0, 0.0)))',
            "gold_call": '_oracle_ml_metric_components(*_review_deepcopy((8.0, np.cos(np.deg2rad(80.0)), par0, 0.0)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\nparE = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\nrE = 1.0001 * parE[10]',
            "call": 'ml_metric_components(*_review_deepcopy((rE, 0.0, parE, 0.01)))',
            "gold_call": '_oracle_ml_metric_components(*_review_deepcopy((rE, 0.0, parE, 0.01)))',
        },
        {
            "setup": 'import numpy as np\nfrom copy import deepcopy as _review_deepcopy\npar = _oracle_ml_parameters_and_horizon(1.0, 0.94, 0.01)\n\n\n' + exception_setup,
            "call": '_exception_code(ml_metric_components, *_review_deepcopy((8.0, 1.5, par, 0.01)))',
            "gold_call": '_exception_code(_oracle_ml_metric_components, *_review_deepcopy((8.0, 1.5, par, 0.01)))',
        },
    ]
