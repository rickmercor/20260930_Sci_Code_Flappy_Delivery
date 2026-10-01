"""
Construct per-exposure PSF-matched numerator and precision maps.

For every source and exposure, make a $25\times25$ zero-padded stamp with center pixel $(12,12)$.  Index that stamp, every correlation output, and every statistic map built from it as `array[row, column]`: axis 0 carries the $y$ pixel coordinate, axis 1 carries the $x$ pixel coordinate, and the pixel at column $x$ and row $y$ is `array[y, x]`.  Its reference flux is $F=455\,10^{-0.4(m-27)}$.  Let the source's pixel trail be



$$

\mathbf a=(q_x,q_y)+\Delta t(d_x,d_y).

$$



For the seven equally timed fractions $f\in\{-0.5,-1/3,-1/6,0,1/6,1/3,0.5\}$, use centers



$$

(x_s,y_s)=(12+x_0+d_xt+p_x,\;12+y_0+d_yt+p_y)+f\mathbf a.

$$



At each fraction use the calibrated-band response $S_f$ defined in `01_observation_plan.py`.  The source profile is $F/7$ times the sum of the seven continuous Gaussians $S_f\exp[-((x-x_s)^2+(y-y_s)^2)/(2s^2)]/(2\pi s^2)$.  Add independent Gaussian noise of standard deviation $\sqrt V$ and no spatially correlated residual.  Draw that noise for each source and exposure as one `(25, 25)` array from a freshly constructed `np.random.default_rng(int(seed + 1009*j))`, in the same `[row, column]` ordering, so a single `rng.normal(0.0, sqrt(V), size=(25, 25))` call supplies the whole stamp and its element `[y, x]` is added at column $x$ and row $y$.



Construct the exposure template from the same S_f-weighted seven Gaussian profiles centered at f*a on integer offsets -5 through 5, then normalize the sampled 11 by 11 template to unit sum. Correlate with zero padding and calculate the sufficient statistics xi = (D/V) * P and zeta = (1/V) * P^2.



The input path has shape `(N,4)`, exposures shape `(N,11)`, and injections shape `(M,9)`; reject incompatible shapes, nonfinite inputs, nonpositive seeing width, or nonpositive variance.

Returns
-------
`tuple[np.ndarray, np.ndarray]`: finite `xi` and `zeta` arrays, each of shape `(M, N, 25, 25)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Construct per-exposure PSF-matched numerator and precision maps."""

import numpy as np

_ROTATION_PERIOD_DAY = 0.73
_FRACTIONS = np.linspace(-0.5, 0.5, 7)


def matched_filter_statistics(path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return matched maps; reject non-finite inputs, nonpositive seeing, or variance."""
    return ()


def _gaussian_psf(sigma: float) -> np.ndarray:
    axis = np.arange(-5, 6, dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def _relative_flux(amplitude_mag: float, phase_rad: float, sample_times_day: np.ndarray) -> np.ndarray:
    """Return flux factors for a sinusoidal magnitude light curve."""
    return 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * sample_times_day / _ROTATION_PERIOD_DAY + phase_rad))


def _relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, band_code: float, sample_times_day: np.ndarray) -> np.ndarray:
    """Return the rotating source response in the specified calibrated band."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    return _relative_flux(amplitude_mag, phase_rad, sample_times_day) * (red_blue_flux_ratio if band_code == 1.0 else 1.0)


def _trailed_psf(sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
    """Return a flux-weighted, unit-sum finite-exposure template."""
    axis = np.arange(-5, 6, dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    template = np.zeros_like(xx)
    for fraction, weight in zip(_FRACTIONS, relative_flux):
        template += weight * np.exp(-((xx - fraction * trail[0]) ** 2 + (yy - fraction * trail[1]) ** 2) / (2 * sigma**2))
    return template / template.sum()


def _source_profile(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
    """Return the light-curve-weighted continuous source profile on a stamp."""
    profile = np.zeros_like(xx)
    for fraction, weight in zip(_FRACTIONS, relative_flux):
        profile += weight * np.exp(-((xx - x0 - fraction * trail[0]) ** 2 + (yy - y0 - fraction * trail[1]) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return profile / 7.0


def _correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode="constant")
    output = np.empty_like(image, dtype=float)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = float(np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel))
    return output

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_matched_filter_statistics(path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate difference stamps and correlate each with its own PSF."""
    import numpy as np

    fractions = np.linspace(-0.5, 0.5, 7)

    def relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, band_code: float, sample_times_day: np.ndarray) -> np.ndarray:
        if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
            raise ValueError("invalid source color or band code")
        relative_flux = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * sample_times_day / 0.73 + phase_rad))
        return relative_flux * (red_blue_flux_ratio if band_code == 1.0 else 1.0)

    def trailed_psf(sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
        axis = np.arange(-5, 6, dtype=float)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        template = np.zeros_like(xx)
        for fraction, weight in zip(fractions, relative_flux):
            template += weight * np.exp(-((xx - fraction * trail[0]) ** 2 + (yy - fraction * trail[1]) ** 2) / (2 * sigma**2))
        return template / template.sum()

    def source_profile(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
        profile = np.zeros_like(xx)
        for fraction, weight in zip(fractions, relative_flux):
            profile += weight * np.exp(-((xx - x0 - fraction * trail[0]) ** 2 + (yy - y0 - fraction * trail[1]) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        return profile / 7.0

    def correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        pad = kernel.shape[0] // 2
        padded = np.pad(image, pad, mode="constant")
        output = np.empty_like(image, dtype=float)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                output[y, x] = float(np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel))
        return output

    path = np.asarray(path, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    injections = np.asarray(injections, dtype=float)
    if path.shape != (len(exposures), 4) or exposures.ndim != 2 or exposures.shape[1] != 11 or injections.ndim != 2 or injections.shape[1] != 9:
        raise ValueError("incompatible path, exposure, or injection arrays")
    if not np.all(np.isfinite(path)) or not np.all(np.isfinite(exposures)) or not np.all(np.isfinite(injections)):
        raise ValueError("non-finite matched-filter input")
    if np.any(exposures[:, 7] <= 0.0) or np.any(exposures[:, 8] <= 0.0):
        raise ValueError("seeing widths and variances must be positive")
    grid = 25
    center = (grid - 1) / 2
    yy, xx = np.meshgrid(np.arange(grid, dtype=float), np.arange(grid, dtype=float), indexing="ij")
    xi = np.empty((len(injections), len(exposures), grid, grid), dtype=float)
    zeta = np.empty_like(xi)
    for source_index, source in enumerate(injections):
        magnitude, seed, base_x, base_y, drift_x, drift_y, amplitude_mag, phase_rad, red_blue_flux_ratio = source
        flux = 455.0 * 10.0 ** (-0.4 * (magnitude - 27.0))
        for exposure_index, exposure in enumerate(exposures):
            time, sigma, variance, duration, band_code = exposure[0], exposure[7], exposure[8], exposure[9], exposure[10]
            source_x = center + base_x + drift_x * time + path[exposure_index, 0]
            source_y = center + base_y + drift_y * time + path[exposure_index, 1]
            trail = path[exposure_index, 2:4] + duration * np.array([drift_x, drift_y])
            response = relative_response(float(amplitude_mag), float(phase_rad), float(red_blue_flux_ratio), float(band_code), time + fractions * duration)
            psf = trailed_psf(float(sigma), trail, response)
            model = flux * source_profile(xx, yy, source_x, source_y, sigma, trail, response)
            rng = np.random.default_rng(int(seed + 1009 * exposure_index))
            image = model + rng.normal(0.0, np.sqrt(variance), size=(grid, grid))
            xi[source_index, exposure_index] = correlate_same(image / variance, psf)
            zeta[source_index, exposure_index] = correlate_same(np.full((grid, grid), 1.0 / variance), psf**2)
    return xi, zeta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, one-source, and varied-seeing cases."""
    return [
        {"setup": "import numpy as np\np=np.zeros((3,4)); e=np.array([[0,0,0,0,0,0,0,.8,90,.004,0],[.4,0,0,0,0,0,0,1.,120,.005,1],[.9,0,0,0,0,0,0,.9,100,.004,0]],float); q=np.array([[28.,17,0.,0.,0.,0.,.4,.2,1.3]],float)", "call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(matched_filter_statistics(p,e,q))", "gold_call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(_oracle_matched_filter_statistics(p,e,q))"},
        {"setup": "import numpy as np\np=np.array([[0.,0.,.1,.2],[.2,-.1,-.2,.1],[0.,0.,.1,-.1]],float); e=np.array([[0,0,0,0,0,0,0,.7,80,.004,1],[.3,0,0,0,0,0,0,.7,80,.005,0],[.6,0,0,0,0,0,0,.7,80,.004,1]],float); q=np.array([[27.5,23,.2,-.2,.01,.01,.3,1.4,.7]],float)", "call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(matched_filter_statistics(p,e,q))", "gold_call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(_oracle_matched_filter_statistics(p,e,q))"},
        {"setup": "import numpy as np\np=np.array([[0.,0.,.3,-.2],[-.15,.25,-.2,.4],[.1,-.1,.1,.1]],float); e=np.array([[0,0,0,0,0,0,0,1.2,150,.004,0],[.5,0,0,0,0,0,0,.75,70,.005,1],[1.,0,0,0,0,0,0,1.05,105,.004,1]],float); q=np.array([[29.,31,-.3,.4,-.02,.03,.5,2.6,1.8]],float)", "call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(matched_filter_statistics(p,e,q))", "gold_call": "(lambda r: np.concatenate([r[0].ravel(),r[1].ravel()]))(_oracle_matched_filter_statistics(p,e,q))"},
        {"setup": "import numpy as np\np=np.zeros((1,4)); e=np.array([[0,0,0,0,0,0,0,.8,0.,.004,0]],float); q=np.array([[28.,17,0.,0.,0.,0.,.4,.2,1.3]],float)\ndef rejected(fn):\n try: fn()\n except ValueError: return 1.0\n return 0.0", "call": "rejected(lambda: matched_filter_statistics(p,e,q))", "gold_call": "rejected(lambda: _oracle_matched_filter_statistics(p,e,q))"},
    ]
