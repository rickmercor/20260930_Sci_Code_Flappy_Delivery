"""
Validate integer n_mechanisms >= 8 and odd n_images >= 7. Use default_rng(seed). For mechanism m set phase=0.17*m+0.003*(seed mod 101), endpoints [-1.18-0.015*(m mod 4), -0.28+0.025*(m mod 5), 0.11*sin(phase)] and [1.16+0.018*(m mod 3), 0.31-0.022*(m mod 4), -0.09*cos(phase)], and s=linspace(0,1,n_images). Add the bow [0.04*sin(pi*s+phase), (0.18+0.015*(m mod 3))*sin(pi*s), 0.13*sin(2*pi*s+0.3*phase)] plus Gaussian noise of standard deviation 0.004+0.0004*(m mod 4), with zero endpoint noise. Use V=0.25*(x^2-1)^2+0.34*y^2+0.21*z^2+0.075*x*y-0.052*y*z+0.018*sin(1.7*x+phase)*cos(1.3*y-0.4*phase) and its exact negative gradient. Set F0=(1.75+0.055*(m mod 5))*max internal force norm+0.18. Return paths, energies, forces, F0, and panel_checksum=dot(paths.ravel(),0.7+(k mod 29)/31)+dot(energies.ravel(),1+(k mod 17)/19).

This step constructs the deterministic audit panel only. It does not encode the paper's adaptive handover policy.

Returns
-------
tuple : paths, energies, forces, baselines, and panel checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_oci_neb_panel(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    """Return synthetic paths, energies, forces, force baselines, and checksum.

    Raises:
        ValueError: If panel sizes are nonintegral, too small, or n_images is even.
    """
    return None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _validate_panel(n_mechanisms, n_images):
    if not isinstance(n_mechanisms, int) or not isinstance(n_images, int):
        raise ValueError("panel sizes must be integers")
    if n_mechanisms < 8 or n_images < 7 or n_images % 2 == 0:
        raise ValueError("n_mechanisms >= 8 and odd n_images >= 7 are required")

def _potential(r, phase):
    x, y, z = np.moveaxis(np.asarray(r, dtype=float), -1, 0)
    return (0.25 * (x * x - 1.0) ** 2 + 0.34 * y * y + 0.21 * z * z
            + 0.075 * x * y - 0.052 * y * z
            + 0.018 * np.sin(1.7 * x + phase) * np.cos(1.3 * y - 0.4 * phase))

def _gradient(r, phase):
    x, y, z = np.asarray(r, dtype=float)
    q = 1.7 * x + phase
    p = 1.3 * y - 0.4 * phase
    return np.array([
        x * (x * x - 1.0) + 0.075 * y + 0.0306 * np.cos(q) * np.cos(p),
        0.68 * y + 0.075 * x - 0.052 * z - 0.0234 * np.sin(q) * np.sin(p),
        0.42 * z - 0.052 * y,
    ])

def _oracle_generate_oci_neb_panel(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    _validate_panel(n_mechanisms, n_images)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_mechanisms, n_images, 3), dtype=float)
    energies = np.empty((n_mechanisms, n_images), dtype=float)
    forces = np.empty_like(paths)
    baselines = np.empty(n_mechanisms, dtype=float)
    s = np.linspace(0.0, 1.0, n_images)
    for m in range(n_mechanisms):
        phase = 0.17 * m + 0.003 * (seed % 101)
        left = np.array([-1.18 - 0.015 * (m % 4), -0.28 + 0.025 * (m % 5), 0.11 * np.sin(phase)])
        right = np.array([1.16 + 0.018 * (m % 3), 0.31 - 0.022 * (m % 4), -0.09 * np.cos(phase)])
        line = (1.0 - s[:, None]) * left + s[:, None] * right
        bow = np.column_stack((
            0.04 * np.sin(np.pi * s + phase),
            (0.18 + 0.015 * (m % 3)) * np.sin(np.pi * s),
            0.13 * np.sin(2.0 * np.pi * s + 0.3 * phase),
        ))
        noise = rng.normal(0.0, 0.004 + 0.0004 * (m % 4), (n_images, 3))
        noise[[0, -1]] = 0.0
        paths[m] = line + bow + noise
        energies[m] = _potential(paths[m], phase)
        forces[m] = np.array([-_gradient(point, phase) for point in paths[m]])
        baselines[m] = (1.75 + 0.055 * (m % 5)) * np.max(np.linalg.norm(forces[m, 1:-1], axis=1)) + 0.18
    weights = 0.7 + (np.arange(paths.size) % 29) / 31.0
    checksum = float(np.dot(paths.ravel(), weights) + np.dot(energies.ravel(), 1.0 + (np.arange(energies.size) % 17) / 19.0))
    return paths, energies, forces, baselines, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(generate_oci_neb_panel())','gold_call':'_tree(_oracle_generate_oci_neb_panel())'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(generate_oci_neb_panel(260601,8,7))','gold_call':'_tree(_oracle_generate_oci_neb_panel(260601,8,7))'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(generate_oci_neb_panel(260777,24,11))','gold_call':'_tree(_oracle_generate_oci_neb_panel(260777,24,11))'},
      {'tol':1e-9,'setup':error_setup+'\na=(260529,18,8)','call':'_value_error(generate_oci_neb_panel,*a)','gold_call':'_value_error(_oracle_generate_oci_neb_panel,*a)'},
    ]
