"""
Register matched statistics on a barycentric canvas and coadd them.

For each output canvas pixel (c, r) and exposure offset (px, py), bilinearly sample the matched-statistic maps at (c plus px, r plus py), taking samples outside the map to be zero. For source k and exposure j, define w_kj as the mean of that source's seven calibrated-band responses. Register the numerator and precision separately. The registered significance nu_k(c, r) is the weighted sum of the sampled xi_kj maps divided by the square root of the larger of the weighted sampled zeta_kj sum and 10^-300. The numerator weights are w_kj and the precision weights are w_kj squared.Only (px, py) is used to register the matched-statistic maps after the exposure-averaged statistic has been formed. Require xi and zeta to be equal finite four-dimensional arrays, path shape (N, 4), exposure shape (N, 11), and injection shape (M, 9).

Returns
-------
`np.ndarray` of shape `(M, 25, 25)` containing the finite registered significance maps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Register matched statistics on a barycentric canvas and coadd them."""

import numpy as np


def registered_coadd(xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> np.ndarray:
    """Return source-response-weighted significance maps; reject non-finite inputs."""
    return np.empty((0, 0, 0), dtype=float)


def _bilinear(image: np.ndarray, y: float, x: float) -> float:
    if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
    fy, fx = y - y0, x - x0
    return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))


def _mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
    """Return the seven-node source response in the exposure's calibrated band."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    fractions = np.linspace(-0.5, 0.5, 7)
    samples = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
    return float(np.mean(samples) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_registered_coadd(xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> np.ndarray:
    """Resample ξ and ζ separately, with source-response likelihood weights."""
    import numpy as np

    def bilinear(image: np.ndarray, y: float, x: float) -> float:
        if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
            return 0.0
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
        fy, fx = y - y0, x - x0
        return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))

    def mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
        if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
            raise ValueError("invalid source color or band code")
        fractions = np.linspace(-0.5, 0.5, 7)
        samples = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
        return float(np.mean(samples) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

    xi, zeta, path = np.asarray(xi, dtype=float), np.asarray(zeta, dtype=float), np.asarray(path, dtype=float)
    exposures, injections = np.asarray(exposures, dtype=float), np.asarray(injections, dtype=float)
    if xi.ndim != 4 or zeta.shape != xi.shape or path.shape != (xi.shape[1], 4) or exposures.shape != (xi.shape[1], 11) or injections.shape != (xi.shape[0], 9):
        raise ValueError("incompatible coadd arrays")
    if not all(np.all(np.isfinite(value)) for value in (xi, zeta, path, exposures, injections)):
        raise ValueError("non-finite coadd input")
    count, _, height, width = xi.shape
    result = np.empty((count, height, width), dtype=float)
    for source in range(count):
        numerator = np.zeros((height, width), dtype=float)
        precision = np.zeros((height, width), dtype=float)
        for exposure in range(xi.shape[1]):
            dx, dy = path[exposure, :2]
            weight = mean_relative_response(injections[source, 6], injections[source, 7], injections[source, 8], exposures[exposure, 0], exposures[exposure, 9], exposures[exposure, 10])
            for row in range(height):
                for col in range(width):
                    numerator[row, col] += weight * bilinear(xi[source, exposure], row + dy, col + dx)
                    precision[row, col] += weight**2 * bilinear(zeta[source, exposure], row + dy, col + dx)
        result[source] = numerator / np.sqrt(np.maximum(precision, 1e-300))
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, zero-shift, and fractional-shift cases."""
    return [
        {"setup": "import numpy as np\nx=np.ones((1,2,25,25)); z=np.ones_like(x); p=np.array([[0.,0.,.2,-.1],[.5,-.25,-.2,.3]]); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0],[.4,0,0,0,0,0,0,.8,90,.004,1]],float); q=np.array([[28,11,0,0,0,0,.3,.2,1.4]],float)", "call": "registered_coadd(x,z,p,e,q)", "gold_call": "_oracle_registered_coadd(x,z,p,e,q)"},
        {"setup": "import numpy as np\nx=np.arange(1250.,dtype=float).reshape(1,2,25,25); z=np.full_like(x,2.); p=np.zeros((2,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,1],[.5,0,0,0,0,0,0,.8,90,.004,0]],float); q=np.array([[28,13,0,0,0,0,.4,1.5,.7]],float)", "call": "registered_coadd(x,z,p,e,q)", "gold_call": "_oracle_registered_coadd(x,z,p,e,q)"},
        {"setup": "import numpy as np\nx=np.arange(1875.,dtype=float).reshape(1,3,25,25); z=np.full_like(x,1.5); p=np.array([[.2,.1,.1,.2],[-.3,.4,-.2,.1],[0.,-.2,.3,-.1]]); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0],[.4,0,0,0,0,0,0,.8,90,.005,1],[.9,0,0,0,0,0,0,.8,90,.004,1]],float); q=np.array([[28,17,0,0,0,0,.5,2.7,1.8]],float)", "call": "registered_coadd(x,z,p,e,q)", "gold_call": "_oracle_registered_coadd(x,z,p,e,q)"},
        {"setup": "import numpy as np\nx=np.ones((1,1,25,25)); x[0,0,12,12]=np.nan; z=np.ones_like(x); p=np.zeros((1,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0]],float); q=np.array([[28,11,0,0,0,0,.3,.2,1.4]],float)\ndef rejected(fn):\n try: fn()\n except ValueError: return 1.0\n return 0.0", "call": "rejected(lambda: registered_coadd(x,z,p,e,q))", "gold_call": "rejected(lambda: _oracle_registered_coadd(x,z,p,e,q))"},
    ]
