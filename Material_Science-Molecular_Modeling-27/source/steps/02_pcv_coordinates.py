"""
Evaluate the progress coordinate and the orthogonal coordinate of the arithmetic path collective variable for a batch of configurations, exactly as the source defines the pair. Two choices inside that definition are load-bearing and the source states both: which quantity plays the role of the inter-configuration metric, and how the reference configurations are indexed in the numerator of the progress coordinate. Set the sharpness of the soft minimum from the reference set itself, as 2.3 divided by the mean of that metric between consecutive reference configurations. Guard the exponentials against underflow; the result must not depend on how that guarding is done.

This is the standard arithmetic construction of Branduardi and co-workers, which the source adopts unchanged. It returns two numbers per configuration, one measuring how far along the route the configuration sits and one measuring how far off it. The second is not a physical displacement, a point the source makes explicitly and which matters for every step that follows.

Returns
-------
An (n_points, 2) float64 array whose first column is the progress coordinate and whose second column is the orthogonal coordinate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pcv_coordinates(points: "np.ndarray", images: "np.ndarray") -> "np.ndarray":
    """Evaluate the progress coordinate and the orthogonal coordinate of the arithmetic path collective
    variable for a batch of configurations, exactly as the source defines the pair. Two choices
    inside that definition are load-bearing and the source states both: which quantity plays the
    role of the inter-configuration metric, and how the reference configurations are indexed in the
    numerator of the progress coordinate. Set the sharpness of the soft minimum from the reference
    set itself, as 2.3 divided by the mean of that metric between consecutive reference
    configurations. Guard the exponentials against underflow; the result must not depend on how that
    guarding is done.

    Args:
        points: array-like of shape (n_points, 2) (a single (2,) point is accepted), configurations
            in the collective-variable plane: migration coordinate, transverse distortion in
            Angstrom.
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.

    Returns:
        An (n_points, 2) float64 array whose first column is the progress coordinate and whose
        second column is the orthogonal coordinate.

    Raises:
        ValueError: if points or images do not have two columns, or if fewer than two reference
        configurations are given.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _lambda_sharpness(images):
    images = np.atleast_2d(np.asarray(images, dtype=float))
    if images.shape[0] < 2:
        raise ValueError("at least two reference images are needed")
    return 2.3/float(np.mean(np.sum(np.diff(images, axis=0)**2, axis=1)))

def _oracle_pcv_coordinates(points: "np.ndarray", images: "np.ndarray") -> "np.ndarray":
    points = np.atleast_2d(np.asarray(points, dtype=float))
    images = np.atleast_2d(np.asarray(images, dtype=float))
    lam_sharp = _lambda_sharpness(images)
    if points.shape[1] != 2 or images.shape[1] != 2:
        raise ValueError("points and images must have two columns")
    M = images.shape[0]
    idx = np.arange(M, dtype=float)
    out = np.empty((points.shape[0], 2))
    step = 60000
    for a in range(0, points.shape[0], step):
        p = points[a:a+step]
        d = (p[:, None, 0]-images[None, :, 0])**2 + (p[:, None, 1]-images[None, :, 1])**2
        dmin = d.min(1, keepdims=True)
        w = np.exp(-lam_sharp*(d-dmin))
        tot = w.sum(1)
        out[a:a+step, 0] = (w @ idx)/tot/(M-1)
        out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(24)\npoints = images + np.array([[0.0, 0.05]])\n',
         'call': 'pcv_coordinates(points, images)',
         'gold_call': '_oracle_pcv_coordinates(points, images)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nt = np.linspace(0.0, 1.0, 37)\npoints = np.stack([t, 0.15*np.sin(np.pi*t) + 0.12*np.cos(3.0*t)], 1)\n',
         'call': 'pcv_coordinates(points, images)',
         'gold_call': '_oracle_pcv_coordinates(points, images)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\npoints = np.array([[0.0, 0.0], [0.5, 0.15], [0.5, 0.45], [1.0, 0.0], [0.25, -0.2]])\n',
         'call': 'pcv_coordinates(points, images)',
         'gold_call': '_oracle_pcv_coordinates(points, images)'},
    ]
