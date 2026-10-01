"""
Light-time-corrected apparent offsets along a two-body orbit.

Treat the supplied observer positions as residual barycentric positions after the common geocentric pointing drift, and advance them linearly with their tabulated velocity during each exposure.  Propagate the two-body state with $\mu_\odot=2.959122082855911\times10^{-4}\ {\rm AU^3\,day^{-2}}$ using classical fourth-order Runge--Kutta: for elapsed time $\tau$, use $N=\max(1,\lceil|\tau|/0.01\rceil)$ equal substeps of $h=\tau/N$.  Initialize the retarded epoch with $t_e=t_{obs}-\lVert\mathbf r_0-\mathbf x_{obs}\rVert/173.144632674240$, then make five fixed-point iterations of

$$
t_e=t_{obs}-\frac{\lVert\mathbf r(t_e)-\mathbf x_{obs}\rVert}{173.144632674240}.
$$

Let $\hat e=(-\hat r_{0,y},\hat r_{0,x},0)/\sqrt{\hat r_{0,x}^2+\hat r_{0,y}^2}$ and $\hat n=\hat r_0\times\hat e$.  From $\hat u=(\mathbf r(t_e)-\mathbf x_{obs})/\lVert\mathbf r(t_e)-\mathbf x_{obs}\rVert$, form the apparent pixel coordinates $206264.806247(\hat u\cdot\hat e,\hat u\cdot\hat n)/(s_{\rm pix}\,\hat u\cdot\hat r_0)$ using the supplied $pixel_scale_arcsec$ value $s_{\rm pix}$.  At each midpoint subtract the WCS line through the first midpoint coordinate with the supplied $wcs_rate_px_day$ two-vector.  With $\delta=\Delta t/2$, evaluate midpoint coordinates at $(t,\mathbf x)$ and endpoint coordinates at $(t\mp\delta,\mathbf x\mp\delta\mathbf v)$.  Compute the finite-exposure residual trail directly as $\mathbf q=\mathbf u_{\rm stop}-\mathbf u_{\rm start}-\Delta t\,\dot{\mathbf u}_{\rm WCS}$; do not obtain it by subtracting two large WCS-subtracted endpoint coordinates.  Return midpoint residuals $(p_x,p_y)$ and trail $(q_x,q_y)$ rounded to the nearest $10^{-6}$ pixel.  Validate a finite (7,) state, a finite (N,11) exposure table, a finite positive pixel scale, and a finite two-vector rate.

Returns
-------
`np.ndarray` of shape `(N, 4)` giving finite `(p_x, p_y, q_x, q_y)` values in pixels for the exposure order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Light-time-corrected apparent offsets along a two-body orbit."""

import numpy as np

_MU = 2.959122082855911e-4


def apparent_registration_path(state: np.ndarray, exposures: np.ndarray, pixel_scale_arcsec: float, wcs_rate_px_day: np.ndarray) -> np.ndarray:
    """Return paths rounded to 1e-6 pixel; reject invalid or non-finite inputs."""
    return np.empty((0, 4), dtype=float)


def _acceleration(position: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(position))
    return -_MU * position / radius**3


def _propagate_rk4(position: np.ndarray, velocity: np.ndarray, elapsed_day: float) -> np.ndarray:
    """Propagate a Kepler state with fixed, short RK4 substeps."""
    count = max(1, int(np.ceil(abs(elapsed_day) / 0.01)))
    h = elapsed_day / count
    r = position.astype(float).copy()
    v = velocity.astype(float).copy()
    for _ in range(count):
        k1r, k1v = v, _acceleration(r)
        k2r, k2v = v + 0.5 * h * k1v, _acceleration(r + 0.5 * h * k1r)
        k3r, k3v = v + 0.5 * h * k2v, _acceleration(r + 0.5 * h * k2r)
        k4r, k4v = v + h * k3v, _acceleration(r + h * k3r)
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return r

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apparent_registration_path(state: np.ndarray, exposures: np.ndarray, pixel_scale_arcsec: float, wcs_rate_px_day: np.ndarray) -> np.ndarray:
    """Solve retarded midpoint directions and finite-exposure trajectories."""
    import numpy as np

    def acceleration(position: np.ndarray) -> np.ndarray:
        radius = float(np.linalg.norm(position))
        return -2.959122082855911e-4 * position / radius**3

    def propagate_rk4(position: np.ndarray, velocity: np.ndarray, elapsed_day: float) -> np.ndarray:
        count = max(1, int(np.ceil(abs(elapsed_day) / 0.01)))
        h = elapsed_day / count
        r, v = position.astype(float).copy(), velocity.astype(float).copy()
        for _ in range(count):
            k1r, k1v = v, acceleration(r)
            k2r, k2v = v + 0.5 * h * k1v, acceleration(r + 0.5 * h * k1r)
            k3r, k3v = v + 0.5 * h * k2v, acceleration(r + 0.5 * h * k2r)
            k4r, k4v = v + h * k3v, acceleration(r + h * k3r)
            r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
            v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        return r

    state = np.asarray(state, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    wcs_rate_px_day = np.asarray(wcs_rate_px_day, dtype=float)
    if (state.shape != (7,) or exposures.ndim != 2 or exposures.shape[1] != 11
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(exposures))
            or not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0
            or wcs_rate_px_day.shape != (2,) or not np.all(np.isfinite(wcs_rate_px_day))):
        raise ValueError("invalid state, exposure table, or pixel scale")
    r0, v0 = state[:3], state[3:6]
    u0 = r0 / np.linalg.norm(r0)
    east = np.array([-u0[1], u0[0], 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(u0, east)
    def apparent_at(time: float, observer: np.ndarray) -> np.ndarray:
        emitted = time - np.linalg.norm(r0 - observer) / 173.144632674240
        for _ in range(5):
            position = propagate_rk4(r0, v0, emitted)
            emitted = time - np.linalg.norm(position - observer) / 173.144632674240
        sightline = position - observer
        direction = sightline / np.linalg.norm(sightline)
        denominator = float(np.dot(direction, u0))
        return np.array([206264.806247 * np.dot(direction, east) / denominator / pixel_scale_arcsec,
                         206264.806247 * np.dot(direction, north) / denominator / pixel_scale_arcsec])

    midpoint = []
    start = []
    stop = []
    for row in exposures:
        time, observer, velocity, duration = float(row[0]), row[1:4], row[4:7], float(row[9])
        half = 0.5 * duration
        midpoint.append(apparent_at(time, observer))
        start.append(apparent_at(time - half, observer - half * velocity))
        stop.append(apparent_at(time + half, observer + half * velocity))
    midpoint, start, stop = np.asarray(midpoint), np.asarray(start), np.asarray(stop)
    def residual(coordinates: np.ndarray, times: np.ndarray) -> np.ndarray:
        line = midpoint[0] + (times - exposures[0, 0])[:, None] * wcs_rate_px_day
        return coordinates - line
    middle_residual = residual(midpoint, exposures[:, 0])
    trail = stop - start - exposures[:, 9, None] * wcs_rate_px_day
    return np.round(np.column_stack([middle_residual, trail]), 6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained differential cases for the apparent path."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "s=np.array([40.11610798425745,12.40936639278486,0.8399440011199893,-0.001260307483569058,0.0037678688458907404,-0.0004741196125724092,-0.11832949214162757])\n"
                "e=np.array([[0.,1e-5,0.,0.,0.,.002,0.,.8,100.,.004,0],[.8,-1e-5,2e-5,0.,-.002,0.,0.,1.,110.,.005,1],[1.6,1e-5,-2e-5,0.,0.,-.002,0.,.9,90.,.004,0]])\n"
                "r=np.array([1.2,-.3])"
            ),
            "call": "apparent_registration_path(s,e,0.04,r)",
            "gold_call": "_oracle_apparent_registration_path(s,e,0.04,r)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "s=np.array([40.,0.,0.,0.,0.003781015828056498,0.00037936698325794486,0.09999999999999945])\n"
                "e=np.array([[0.,0.,0.,0.,.001,0.,0.,.8,80.,.003,1],[.2,1e-5,0.,0.,0.,.001,0.,.8,80.,.004,0],[.4,0.,1e-5,0.,-.001,0.,0.,.8,80.,.003,1]])\n"
                "r=np.array([.5,.2])"
            ),
            "call": "apparent_registration_path(s,e,0.05,r)",
            "gold_call": "_oracle_apparent_registration_path(s,e,0.05,r)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "s=np.array([40.33650711445504,-22.035934268228555,-1.8395093725851714,0.0017097598857428625,0.0028660452921630817,0.0006576515159067075,0.19601159609915417])\n"
                "e=np.array([[0.,2e-5,-1e-5,1e-5,.002,0.,0.,.7,90.,.004,0],[.5,-2e-5,3e-5,-1e-5,0.,.002,0.,1.1,120.,.005,1],[1.1,1e-5,-3e-5,2e-5,-.002,0.,0.,.9,100.,.004,1]])\n"
                "r=np.array([-.8,.6])"
            ),
            "call": "apparent_registration_path(s,e,0.03,r)",
            "gold_call": "_oracle_apparent_registration_path(s,e,0.03,r)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "s=np.array([40.,0.,0.,0.,.0038,0.,0.])\n"
                "e=np.array([[0.,0.,0.,0.,0.,0.,0.,.8,80.,.004,0.]])\n"
                "r=np.array([0.,0.])\n"
                "def rejected(fn):\n"
                " try: fn()\n"
                " except ValueError: return 1.0\n"
                " return 0.0"
            ),
            "call": "rejected(lambda: apparent_registration_path(s,e,np.nan,r))",
            "gold_call": "rejected(lambda: _oracle_apparent_registration_path(s,e,np.nan,r))",
        },
    ]
