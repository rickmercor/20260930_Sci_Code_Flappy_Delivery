"""
Return the nudged-elastic-band force on every image of a band, given the image coordinates, the image energies, the true forces of the calculator and one spring constant per adjacent pair, with an optional climbing image. The tangent is the energy-weighted upwind tangent of Henkelman and Jonsson (J. Chem. Phys. 113, 9978, 2000) and the climbing image is that of Henkelman, Uberuaga and Jonsson (J. Chem. Phys. 113, 9901, 2000).

The nudged elastic band turns the search for a minimum-energy path into an ordinary minimisation by decoupling two jobs that would otherwise fight one another. The true force perpendicular to the path pulls each image down onto the path, while the spring force along the path keeps the images from sliding into the two basins; keeping only the perpendicular part of the one and only the parallel part of the other is what the nudging means. Which direction counts as along the path matters more than it appears: a central difference of the neighbouring coordinates makes the tangent kink wherever the energy varies rapidly, and the band then develops artificial oscillations. The plain band never places an image exactly at the saddle, so the barrier is only bracketed; a climbing image converts one image into a saddle search that runs alongside the band and yields the transition-state energy directly.



Conventions fixed by this task. The two endpoints are held fixed and receive zero force. With one spring constant per adjacent pair, $k_i$ belonging to the spring between images $i$ and $i+1$, the spring force on an ordinary interior image is the difference of the two neighbouring spring extensions along the unit tangent, as in the formulas. The climbing image, when one is named, carries no spring force. A tangent of zero length is replaced by the zero vector.

With $\\boldsymbol{\\tau}^{+}_i=\\mathbf{R}_{i+1}-\\mathbf{R}_i$, $\\boldsymbol{\\tau}^{-}_i=\\mathbf{R}_i-\\mathbf{R}_{i-1}$ and $\\hat{\\boldsymbol{\\tau}}_i$ the unit tangent, the spring force on an ordinary interior image is



$$\\mathbf{F}^{\\mathrm{spring}}_i=\\Big(k_i\\lVert\\boldsymbol{\\tau}^{+}_i\\rVert-k_{i-1}\\lVert\\boldsymbol{\\tau}^{-}_i\\rVert\\Big)\\hat{\\boldsymbol{\\tau}}_i .$$

Returns
-------
A `numpy` array of the shape of `band` holding the nudged-elastic-band force on every image in eV/A, with zeros on the two endpoints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def neb_band_forces(band: "np.ndarray", energies: "np.ndarray",
                    forces: "np.ndarray", k: "np.ndarray",
                    climb: int = -1) -> "np.ndarray":
    """Nudged-elastic-band force of every image, with an optional climbing image.

    Parameters
    ----------
    band : numpy.ndarray
        Image coordinates in angstrom, shape (M, N, 3) with M >= 3.
    energies : numpy.ndarray
        True energy of each image in eV, shape (M,).
    forces : numpy.ndarray
        True force on every atom of every image in eV/A, shape of band.
    k : numpy.ndarray
        Positive spring constants of the M - 1 adjacent image pairs in
        eV/A^2.
    climb : int
        Index of the climbing image, an interior index between 1 and M - 2,
        or -1 for a band without a climbing image.

    Returns
    -------
    out : numpy.ndarray
        Band force of the shape of band, zero on both endpoints.

    Raises
    ------
    ValueError
        If band is not an (M, N, 3) array with M >= 3, if energies, forces or
        k has the wrong shape, if any spring constant is not positive, or if
        climb is neither -1 nor an interior image index.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_neb_band_forces(band: "np.ndarray", energies: "np.ndarray",
                            forces: "np.ndarray", k: "np.ndarray",
                            climb: int = -1) -> "np.ndarray":
    band = np.asarray(band, dtype=float)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    forces = np.asarray(forces, dtype=float)
    k = np.asarray(k, dtype=float).reshape(-1)
    if band.ndim != 3 or band.shape[2] != 3 or band.shape[0] < 3:
        raise ValueError("band must have shape (M, N, 3) with M >= 3")
    if energies.shape != (band.shape[0],):
        raise ValueError("energies must have one entry per image")
    if forces.shape != band.shape:
        raise ValueError("forces must have the shape of band")
    if k.shape != (band.shape[0] - 1,):
        raise ValueError("k must hold one spring constant per adjacent pair")
    if np.any(k <= 0.0):
        raise ValueError("spring constants must be positive")

    m = band.shape[0]
    if int(climb) != -1 and not (1 <= int(climb) <= m - 2):
        raise ValueError("climb must be -1 or an interior image index")
    out = np.zeros_like(band)
    top = int(climb)

    for i in range(1, m - 1):
        tp = band[i + 1] - band[i]
        tm = band[i] - band[i - 1]
        if energies[i + 1] > energies[i] > energies[i - 1]:
            tan = tp
        elif energies[i + 1] < energies[i] < energies[i - 1]:
            tan = tm
        else:
            de_p = abs(energies[i + 1] - energies[i])
            de_m = abs(energies[i - 1] - energies[i])
            hi, lo = max(de_p, de_m), min(de_p, de_m)
            if energies[i + 1] > energies[i - 1]:
                tan = tp * hi + tm * lo
            else:
                tan = tp * lo + tm * hi
        nrm = float(np.sqrt(np.sum(tan * tan)))
        tan = tan / nrm if nrm > 1e-12 else np.zeros_like(tan)

        proj = float(np.sum(forces[i] * tan))
        if i == top:
            out[i] = forces[i] - 2.0 * proj * tan
            continue
        spring = (k[i] * float(np.sqrt(np.sum(tp * tp)))
                  - k[i - 1] * float(np.sqrt(np.sum(tm * tm))))
        out[i] = forces[i] - proj * tan + spring * tan
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    shared = """import numpy as np

def band_of(n_img, n_at, amp):
    lam = np.linspace(0.0, 1.0, n_img)
    base = np.array([[float(i), 0.4 * i, 1.0 + 0.1 * i] for i in range(n_at)])
    move = np.zeros((n_at, 3))
    move[n_at // 2] = [amp, 0.5 * amp, -0.2 * amp]
    band = base[None, :, :] + lam[:, None, None] * move[None, :, :]
    band += 0.05 * np.sin(3.0 * lam)[:, None, None] * np.ones((1, n_at, 3))
    return band

def forces_of(band, scale):
    c = band.mean(axis=1, keepdims=True)
    return -scale * (band - c) / (1.0 + np.sum((band - c) ** 2, axis=-1,
                                               keepdims=True))
"""
    return [
        # normal: a single-peaked profile with the paper's variable springs and
        # no climbing image
        {"setup": shared + """
band = band_of(20, 6, 2.6)
x = np.linspace(0.0, 1.0, 20)
e = -33.8 + 1.4 * np.sin(np.pi * x) ** 2 + 0.24 * x
f = forces_of(band, 1.3)
k = np.linspace(0.1, 4.0, 19)
""",
         "call": "neb_band_forces(band, e, f, k)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k)"},
        # normal: the same band with the highest interior image climbing
        {"setup": shared + """
band = band_of(20, 6, 2.6)
x = np.linspace(0.0, 1.0, 20)
e = -33.8 + 1.4 * np.sin(np.pi * x) ** 2 + 0.24 * x
f = forces_of(band, 1.3)
k = np.linspace(0.1, 4.0, 19)
top = int(np.argmax(e[1:-1])) + 1
""",
         "call": "neb_band_forces(band, e, f, k, top)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k, top)"},
        # boundary: a strictly rising profile, so every interior tangent is the
        # forward difference and no image is at a local extremum
        {"setup": shared + """
band = band_of(11, 5, 1.8)
e = np.linspace(-2.0, 1.0, 11)
f = forces_of(band, 0.7)
k = np.full(10, 1.0)
""",
         "call": "neb_band_forces(band, e, f, k)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k)"},
        # boundary: a flat profile, where every interior image is at an extremum
        # and the tangent falls to the weighted combination with equal weights
        {"setup": shared + """
band = band_of(9, 4, 2.2)
e = np.zeros(9)
f = forces_of(band, 1.1)
k = np.full(8, 0.5)
""",
         "call": "neb_band_forces(band, e, f, k)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k)"},
        # boundary: a climbing image chosen away from the energy maximum, which
        # the routine must honour rather than re-derive
        {"setup": shared + """
band = band_of(12, 5, 2.0)
x = np.linspace(0.0, 1.0, 12)
e = -1.0 + np.sin(np.pi * x) ** 2
f = forces_of(band, 0.9)
k = np.full(11, 2.0)
""",
         "call": "neb_band_forces(band, e, f, k, 2)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k, 2)"},
        # edge: three images, the smallest band with a single interior image,
        # which is also the climbing image
        {"setup": shared + """
band = band_of(3, 4, 1.4)
e = np.array([-1.0, 0.4, -0.7])
f = forces_of(band, 1.0)
k = np.array([0.8, 1.6])
""",
         "call": "neb_band_forces(band, e, f, k, 1)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k, 1)"},
        # edge: two coincident neighbouring images, whose tangent has zero
        # length and is replaced by the zero vector
        {"setup": shared + """
band = band_of(7, 4, 1.6)
band[3] = band[4]
e = np.array([-1.0, -0.5, 0.2, 0.9, 0.9, 0.1, -0.6])
f = forces_of(band, 1.0)
k = np.full(6, 1.0)
""",
         "call": "neb_band_forces(band, e, f, k)",
         "gold_call": "_oracle_neb_band_forces(band, e, f, k)"},
        # invalid: a spring array of the wrong length
        {"setup": shared + """
band = band_of(8, 4, 1.5)
e = np.linspace(0.0, 1.0, 8)
f = forces_of(band, 1.0)
bad = np.full(8, 1.0)
def run_model():
    try:
        neb_band_forces(band, e, f, bad)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_neb_band_forces(band, e, f, bad)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a climbing index naming an endpoint
        {"setup": shared + """
band = band_of(8, 4, 1.5)
e = np.linspace(0.0, 1.0, 8)
f = forces_of(band, 1.0)
k = np.full(7, 1.0)
def run_model():
    try:
        neb_band_forces(band, e, f, k, 7)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_neb_band_forces(band, e, f, k, 7)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
