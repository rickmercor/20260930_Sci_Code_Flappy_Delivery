"""
Measure robust forced fluxes at the retained candidate positions.

The likelihood-coadd peak supplies a discrete candidate location, not the final photometric measurement. At candidate coordinates (c_k, r_k), apply the same registered sampling convention as in step 05. For each exposure with positive sampled precision, call the registered numerator sample xi_tilde_kj and the registered precision sample zeta_tilde_kj.



With the same seven-node calibrated-band mean response w_kj from step 05, the canonical forced flux is xi_tilde_kj divided by w_kj times zeta_tilde_kj, and its uncertainty is one divided by w_kj times the square root of zeta_tilde_kj.



A zero-precision sample is outside the zero-padded support and contributes no measurement.  Start with all positive-precision measurements.  For at most ten passes, calculate the ordinary arithmetic mean $\bar f$ of retained $f_{kj}$ and retain exactly those measurements satisfying $|f_{kj}-\bar f|\le3\sigma_{kj}$; stop early if the retained set does not change.  A pass whose acceptance test would reject every remaining measurement is mutually inconsistent photometry rather than a valid clip, so that pass is not applied: stop instead and keep the previous retained set.  The retained set $K$ is therefore the last nonempty one, is never empty, and $|K|\ge1$ always holds, so no source is scored from an undefined mean.  Then use



$$

\sigma_{\bar f}=\frac{\sqrt{\sum_{j\in K}\sigma_{kj}^2}}{|K|},\qquad

\nu_{\rm forced}=\frac{\bar f}{\sigma_{\bar f}}.
$$

	



An injected source is recovered when $\nu_{\rm forced}$ is at least the supplied retention significance $\nu_{\rm ret}$.  Preserve injection ordering and pair each input magnitude with the numerical flag $I[\nu_{\rm forced}\ge\nu_{\rm ret}]$.  Reject incompatible candidate, statistic, path, exposure, or `(M,9)` Injection arrays; also reject nonfinite data, a nonpositive threshold, or a candidate with no positive-precision forced measurement.

Returns
-------
`np.ndarray` of shape `(M,2)`, with magnitude in column zero and a numerical binary recovery flag in column one.  Because the retained clipping set is never allowed to empty, every returned flag is a finite `0.0` or `1.0` and no source yields a nonfinite significance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Measure robust forced fluxes at the retained candidate positions."""

import numpy as np


def injection_recoveries(candidates: np.ndarray, xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray, threshold: float) -> np.ndarray:
    """Return robust recovery flags; reject non-finite inputs or nonpositive threshold."""
    return np.empty((0, 2), dtype=float)


def _bilinear(image: np.ndarray, y: float, x: float) -> float:
    """Return a zero-padded bilinear sample from one statistic map."""
    if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
    fy, fx = y - y0, x - x0
    return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))


def _mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
    """Return the canonical-flux response for one calibrated exposure."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    fractions = np.linspace(-0.5, 0.5, 7)
    rotation = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
    return float(np.mean(rotation) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_injection_recoveries(candidates: np.ndarray, xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray, threshold: float) -> np.ndarray:
    """Use ten-pass 3-sigma clipping that never empties the retained set."""
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
        rotation = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
        return float(np.mean(rotation) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))

    candidates, xi, zeta = np.asarray(candidates, dtype=float), np.asarray(xi, dtype=float), np.asarray(zeta, dtype=float)
    path, exposures, injections = np.asarray(path, dtype=float), np.asarray(exposures, dtype=float), np.asarray(injections, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or xi.ndim != 4 or zeta.shape != xi.shape:
        raise ValueError("invalid candidate or matched-statistic input")
    if (len(candidates) != xi.shape[0] or path.shape != (xi.shape[1], 4)
            or exposures.shape != (xi.shape[1], 11) or injections.shape != (xi.shape[0], 9)
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("incompatible forced-photometry arrays")
    if not all(np.all(np.isfinite(value)) for value in (candidates, xi, zeta, path, exposures, injections)):
        raise ValueError("non-finite forced-photometry input")
    recovered = np.empty(len(injections), dtype=float)
    for source in range(len(injections)):
        values, errors = [], []
        for exposure in range(xi.shape[1]):
            row = candidates[source, 1] + path[exposure, 1]
            col = candidates[source, 0] + path[exposure, 0]
            numerator = bilinear(xi[source, exposure], row, col)
            precision = bilinear(zeta[source, exposure], row, col)
            if precision <= 0.0:
                continue
            response = mean_relative_response(injections[source, 6], injections[source, 7], injections[source, 8], exposures[exposure, 0], exposures[exposure, 9], exposures[exposure, 10])
            values.append(numerator / (response * precision))
            errors.append(1.0 / (response * np.sqrt(precision)))
        values, errors = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
        if len(values) == 0:
            raise ValueError("candidate has no positive-precision forced measurement")
        keep = np.ones(len(values), dtype=bool)
        for _ in range(10):
            mean_flux = float(np.mean(values[keep]))
            updated = np.abs(values - mean_flux) <= 3.0 * errors
            if np.array_equal(updated, keep) or not updated.any():
                break
            keep = updated
        mean_flux = float(np.mean(values[keep]))
        mean_error = float(np.sqrt(np.sum(errors[keep] ** 2)) / np.count_nonzero(keep))
        recovered[source] = float(mean_flux / mean_error >= threshold)
    return np.column_stack([injections[:, 0], recovered])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return mixed-label, clipping-sensitive, inconsistent, and invalid cases."""
    return [
        {"setup": "import numpy as np\nc=np.array([[1.,1.,6.],[1.,1.,2.]]); x=np.empty((2,2,3,3)); x[0]=20.; x[1]=5.; z=np.ones_like(x); p=np.zeros((2,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0],[.4,0,0,0,0,0,0,.9,110,.005,0]],float); q=np.array([[27.8,1,0,0,0,0,0.,0.,1.],[29.2,2,0,0,0,0,0.,0.,1.]],float)", "call": "injection_recoveries(c,x,z,p,e,q,15.0)", "gold_call": "_oracle_injection_recoveries(c,x,z,p,e,q,15.0)"},
        {"setup": "import numpy as np\nc=np.array([[1.,1.,6.]]); x=np.ones((1,4,3,3))*np.array([20.,10.,24.,16.]).reshape(1,4,1,1); z=np.ones_like(x); p=np.zeros((4,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,1],[.4,0,0,0,0,0,0,.9,110,.005,0],[.8,0,0,0,0,0,0,.8,95,.004,1],[1.2,0,0,0,0,0,0,.9,105,.005,0]],float); q=np.array([[28.5,1,0,0,0,0,0.,1.,2.]],float)", "call": "injection_recoveries(c,x,z,p,e,q,28.0)", "gold_call": "_oracle_injection_recoveries(c,x,z,p,e,q,28.0)"},
        {"setup": "import numpy as np\nc=np.array([[1.,1.,6.]]); x=np.ones((1,2,3,3))*np.array([0.,100.]).reshape(1,2,1,1); z=np.ones_like(x); p=np.zeros((2,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0],[.4,0,0,0,0,0,0,.8,90,.004,0]],float); q=np.array([[29.,1,0,0,0,0,0.,0.,1.]],float)", "call": "injection_recoveries(c,x,z,p,e,q,60.0)", "gold_call": "_oracle_injection_recoveries(c,x,z,p,e,q,60.0)"},
        {"setup": "import numpy as np\nc=np.array([[1.,1.,6.]]); x=np.full((1,1,3,3),20.); z=np.ones_like(x); p=np.zeros((1,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0]],float); q=np.array([[28.,1,0,0,0,0,0.,0.,1.]],float)\ndef rejected(fn):\n try: fn()\n except ValueError: return 1.0\n return 0.0", "call": "rejected(lambda: injection_recoveries(c,x,z,p,e,q,np.nan))", "gold_call": "rejected(lambda: _oracle_injection_recoveries(c,x,z,p,e,q,np.nan))"},
    ]
