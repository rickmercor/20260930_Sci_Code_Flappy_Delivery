"""
Gaussian probability and moment response of a moving compatible region

Let the vertices of a smooth compatible polytope have expansions $$\boldsymbol v(\varepsilon)=\boldsymbol v^{(0)}+\varepsilon\boldsymbol v^{(1)}+\tfrac12\varepsilon^2\boldsymbol v^{(2)}+o(\varepsilon^2)$$. For a fixed Gaussian component with mean $$\boldsymbol\mu$$ and covariance $$\Sigma$$, integrate $$\boldsymbol m(\varepsilon)=\int_{\mathcal P(\varepsilon)}(1,u_1,u_2,u_3,u_1^2,u_1u_2,u_1u_3,u_2^2,u_2u_3,u_3^2)\varphi(\boldsymbol u;\boldsymbol\mu,\Sigma)\,d^3u$$ and return its value and first two true derivatives. The region has locally fixed face incidence, and every facet remains planar through second order, as supplied by the preceding active-plane construction. Moments use the original latent coordinates and remain unnormalized, so component weights can be combined before posterior conditioning. Any sufficiently accurate deterministic method is acceptable.

Returns
-------
np.ndarray of shape (3,10), containing Gaussian raw moments and their first two true derivatives with respect to the same dimensionless perturbation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def gaussian_polytope_moment_jet(vertex_jet: np.ndarray, mean: np.ndarray,
                                covariance: np.ndarray,
                                tol: float = 1e-8) -> np.ndarray:
    r'''Integrate the response of Gaussian raw moments to region motion.

    Parameters
    ----------
    vertex_jet : np.ndarray
        Finite real shape (3,V,3). Axis 0 contains positions and true first and second derivatives, with matched vertex identities across orders. The base region is a convex hull; vertex order is unrestricted. Jets describe a locally fixed face-incidence branch whose facets remain planar through second order. A locally persistent empty or zero-volume region returns zeros.
    mean : np.ndarray
        Fixed finite real Gaussian mean of shape (3,), in the original latent coordinates.
    covariance : np.ndarray
        Fixed finite real covariance of shape (3,3), symmetric within absolute tolerance 1e-12 and positive definite, with spectral condition number <= 1e6. Base vertices must lie within Mahalanobis distance 12 of the mean; these bounds define the resolved numerical domain.
    tol : float
        Finite accuracy target in [1e-10,1e-5], default 1e-8. Each returned entry should have absolute error <= tol times one plus its magnitude.

    Returns
    -------
    moment_jet : np.ndarray
        Finite shape (3,10), with true derivative orders on axis 0. Columns are integrals of [1,u1,u2,u3,u1^2,u1*u2,u1*u3,u2^2,u2*u3,u3^2] against the fixed Gaussian density. No factorial division or probability normalization is applied. A locally persistent zero-volume region gives thirty zeros.

    Raises
    ------
    ValueError
        If finite real shapes, the resolved covariance domain or tolerance contracts fail, or numerical geometry or integration cannot produce finite responses at the requested accuracy. The fixed-incidence and facet-planarity conditions are input preconditions and need not be numerically certified.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull

def _oracle_gaussian_polytope_moment_jet(vertex_jet: np.ndarray, mean: np.ndarray,
                                         covariance: np.ndarray,
                                         tol: float = 1e-8) -> np.ndarray:
    if any(not np.isrealobj(value) for value in (vertex_jet, mean, covariance)):
        raise ValueError("vertex jets, mean and covariance must be real")
    try:
        vertices, mu, cov = [np.asarray(value, dtype=float) for value in (vertex_jet, mean, covariance)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("vertex jets, mean and covariance must be finite real arrays") from exc
    if vertices.ndim != 3 or vertices.shape[0] != 3 or vertices.shape[2] != 3 or mu.shape != (3,) or cov.shape != (3, 3):
        raise ValueError("expected shapes (3,V,3), (3,), and (3,3)")
    if not all(np.all(np.isfinite(value)) for value in (vertices, mu, cov)):
        raise ValueError("vertex jets, mean and covariance must be finite")
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    cov = (cov + cov.T) / 2.0
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[-1] / eigenvalues[0] > 1e6:
        raise ValueError("covariance condition number must be at most 1e6")
    inverse_chol = np.linalg.inv(chol)
    if vertices.shape[1] and np.any(np.sum(((vertices[0] - mu) @ inverse_chol.T) ** 2, axis=1) > 144.0):
        raise ValueError("base vertices must lie within Mahalanobis distance 12 of the mean")
    if not np.isscalar(tol) or not np.isrealobj(tol):
        raise ValueError("tol must be a finite real accuracy target")
    try:
        tolerance = float(tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tol must be a finite real accuracy target") from exc
    if not np.isfinite(tolerance) or not 1e-10 <= tolerance <= 1e-5:
        raise ValueError("tol must lie in [1e-10,1e-5]")
    if vertices.shape[1] < 4 or np.linalg.matrix_rank(vertices[0] - vertices[0].mean(axis=0)) < 3:
        return np.zeros((3, 10), dtype=float)
    try:
        hull = ConvexHull(vertices[0])
    except Exception as exc:
        raise ValueError("the base convex region could not be constructed") from exc
    center = vertices[:, hull.vertices].mean(axis=1)
    static_region = not np.any(vertices[1:])
    precision = inverse_chol.T @ inverse_chol
    normalization = (2.0 * np.pi) ** (-1.5) / np.prod(np.diag(chol))

    def _moving_gaussian_integral(order):
        nodes, weights = leggauss(order)
        nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
        a, b, c = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        wa, wb, wc = np.meshgrid(weights, weights, weights, indexing="ij")
        barycentric = np.column_stack((a.ravel(), ((1.0 - a) * b).ravel(),
                                       ((1.0 - a) * (1.0 - b) * c).ravel()))
        quadrature_weights = (wa * wb * wc * (1.0 - a) ** 2 * (1.0 - b)).ravel()
        total = np.zeros((3, 10), dtype=float)
        for face in hull.simplices:
            if static_region:
                base = (vertices[0, face] - center[0]).T
                samples = center[0] + barycentric @ base.T
                standardized = (samples - mu) @ inverse_chol.T
                weight = (quadrature_weights * abs(np.linalg.det(base)) * normalization
                          * np.exp(-0.5 * np.sum(standardized ** 2, axis=1)))
                x, y, z = samples.T
                total[0, 0] += np.sum(weight)
                total[0, 1:4] += weight @ samples
                total[0, 4:] += weight @ np.column_stack((x*x, x*y, x*z, y*y, y*z, z*z))
                continue
            basis = (vertices[:, face] - center[:, None, :]).transpose(0, 2, 1)
            inverse_basis = np.linalg.inv(basis[0])
            first_matrix = inverse_basis @ basis[1]
            jacobian_first = np.trace(first_matrix)
            jacobian_second = np.trace(inverse_basis @ basis[2] - first_matrix @ first_matrix)
            determinant = abs(np.linalg.det(basis[0]))
            samples = center[:, None, :] + np.einsum("pi,kji->kpj", barycentric, basis)
            displacement = samples[0] - mu
            gaussian_first = -np.einsum("pi,ij,pj->p", samples[1], precision, displacement)
            gaussian_second = (-np.einsum("pi,ij,pj->p", samples[2], precision, displacement)
                               -np.einsum("pi,ij,pj->p", samples[1], precision, samples[1]))
            logarithmic_first = jacobian_first + gaussian_first
            logarithmic_second = jacobian_second + gaussian_second
            standardized = displacement @ inverse_chol.T
            weight = quadrature_weights * determinant * normalization * np.exp(-0.5 * np.sum(standardized ** 2, axis=1))
            weight_jet = np.array([weight, weight * logarithmic_first,
                                   weight * (logarithmic_first ** 2 + logarithmic_second)])
            monomials = np.zeros((3, len(barycentric), 10), dtype=float)
            monomials[0, :, 0] = 1.0
            monomials[:, :, 1:4] = samples
            for column, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), 4):
                monomials[0, :, column] = samples[0, :, i] * samples[0, :, j]
                monomials[1, :, column] = samples[1, :, i] * samples[0, :, j] + samples[0, :, i] * samples[1, :, j]
                monomials[2, :, column] = (samples[2, :, i] * samples[0, :, j]
                                          + 2.0 * samples[1, :, i] * samples[1, :, j]
                                          + samples[0, :, i] * samples[2, :, j])
            total[0] += weight_jet[0] @ monomials[0]
            total[1] += weight_jet[1] @ monomials[0] + weight_jet[0] @ monomials[1]
            total[2] += (weight_jet[2] @ monomials[0] + 2.0 * weight_jet[1] @ monomials[1]
                         + weight_jet[0] @ monomials[2])
        return total

    previous = _moving_gaussian_integral(8)
    for order in (12, 16, 24, 32, 48, 64):
        current = _moving_gaussian_integral(order)
        if not np.all(np.isfinite(current)):
            raise ValueError("the Gaussian response is nonfinite")
        if np.all(np.abs(current - previous) <= 0.2 * tolerance * (1.0 + np.abs(current))):
            return current
        previous = current
    raise ValueError("moving Gaussian integration did not converge to the requested accuracy")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = "import numpy as np\nfrom numpy.polynomial.legendre import leggauss\nfrom scipy.spatial import ConvexHull\n"
    cube = "base=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])\n"
    return [
        {"setup": imports + cube + "v=np.zeros((3,len(base),3));v[0]=base;mu=np.zeros(3);cov=np.eye(3)\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + cube + "v=np.array([base,np.tile([.2,-.1,.15],(len(base),1)),np.tile([-.08,.04,.06],(len(base),1))]);mu=np.array([.25,-.2,.1]);cov=np.array([[.7,.21,-.12],[.21,.9,.18],[-.12,.18,.6]])\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + cube + "a=np.array([[0.,-.3,.2],[.3,0.,-.1],[-.2,.1,0.]]);v=np.array([base,base@a.T,base@(a@a).T]);mu=np.array([.4,-.2,.1]);cov=np.array([[.3,.08,-.04],[.08,.7,.1],[-.04,.1,.5]])\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + cube + "v=np.array([base,base*np.array([.2,-.1,.15]),base*np.array([.05,.03,-.04])]);mu=np.array([.1,.2,-.15]);cov=np.array([[.5,-.12,.1],[-.12,.4,.06],[.1,.06,.3]])\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy(),1e-9)", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy(),1e-9)", "tol": 2e-6},
        {"setup": imports + "base=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])+np.array([.2,-.3,.1]);a=np.array([[.1,.2,0.],[-.1,.05,.1],[.04,0.,-.1]]);b=np.array([[.03,-.02,.04],[.01,.05,0.],[0.,.02,-.01]]);v=np.array([base,base@a.T+np.array([.1,.03,-.06]),base@b.T+np.array([-.02,.04,.01])]);mu=np.array([.2,.1,.3]);cov=np.array([[.3,.08,-.04],[.08,.4,.1],[-.04,.1,.5]])\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + "v=np.array([[[-1.0, -1.0, -1.0], [-1.0, -1.0, 0.3], [-1.0, 0.3, -1.0], [-1.0, 0.7978723404255319, 0.6595744680851063], [-0.18509433962264152, 0.9972641509433963, 0.7809433962264151], [0.3425925925925926, 0.5685185185185185, -1.0], [0.6764705882352942, -1.0, 0.4676470588235294], [0.97, -1.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.065], [0.0, -0.031, 0.0], [0.0, 0.01998641919420552, 0.23080579447713898], [0.05395343538625846, 0.014521324314702742, 0.273426023495906], [0.14901577503429356, -0.04684499314128944, 0.0], [0.32006920415224915, 0.0, 0.13891868512110728], [0.3185, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, -0.0495], [0.0, -0.00486, 0.0], [0.0, 0.03562635446866301, 0.006169634859327895], [0.14109202712306132, 0.06828177232883521, 0.00946895411648542], [0.1231073642229335, 0.01763025199410659, 0.0], [0.10596037723047697, 0.0, -0.07235638442228103], [0.06245, 0.0, 0.0]]],dtype=float);mu=np.array([.1,.2,-.1]);cov=np.array([[.6,.15,.05],[.15,.5,-.1],[.05,-.1,.7]])\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + "v=np.empty((3,0,3));mu=np.zeros(3);cov=np.eye(3)\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
        {"setup": imports + "v=np.zeros((3,4,3));v[0]=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]);mu=np.zeros(3);cov=np.eye(3)\n",
         "call": "gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "gold_call": "_oracle_gaussian_polytope_moment_jet(v.copy(),mu.copy(),cov.copy())", "tol": 2e-6},
    ]
