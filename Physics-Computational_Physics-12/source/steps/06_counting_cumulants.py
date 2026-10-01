"""
Joint counting cumulants of the two registered currents.

Two currents are counted at once. Current $A$ assigns the increments $d^{(A)} = (+1,-1,0,0)$ to the four registered transition types and current $B$ assigns $d^{(B)} = (0,0,+1,-1)$. For a semi-Markov process the joint scaled cumulant generating function $\lambda(z)$ of the two accumulated increments, with $z = (z_A, z_B)$, is fixed implicitly by the tilted kernel



$$M_{uv}(z,\lambda) = e^{z_A d^{(A)}_v + z_B d^{(B)}_v}\left[P - \lambda M^{(1)} + \tfrac{1}{2}\lambda^2 M^{(2)}\right]_{uv}, \qquad G(z,\lambda) \equiv \det\left[M(z,\lambda) - \mathbb{1}\right] = 0,$$



with $\lambda(0) = 0$ because $P$ is stochastic. Differentiating $G(z, \lambda(z)) = 0$ at the origin gives the first two joint cumulants without any numerical differentiation,



$$\lambda_i = -\frac{G_{z_i}}{G_\lambda}, \qquad \lambda_{ij} = -\frac{G_{z_i z_j} + G_{z_i \lambda}\lambda_j + G_{z_j \lambda}\lambda_i + G_{\lambda\lambda}\lambda_i\lambda_j}{G_\lambda},$$



where every partial derivative of $G$ is evaluated at $z = 0$, $\lambda = 0$. Because a determinant is multilinear in the columns of its matrix, the derivatives of $G$ follow from the entrywise derivatives of $M$ without expanding the determinant symbolically: writing $A = M - \mathbb{1}$ and $A(k \leftarrow x)$ for $A$ with column $k$ replaced by the vector $x$,



$$\partial_a \det A = \sum_k \det A\!\left(k \leftarrow \partial_a A_{:,k}\right), \qquad \partial_a \partial_b \det A = \sum_k \det A\!\left(k \leftarrow \partial_a\partial_b A_{:,k}\right) + \sum_{k \neq m} \det A\!\left(k \leftarrow \partial_a A_{:,k},\, m \leftarrow \partial_b A_{:,m}\right).$$



The entrywise derivatives at the origin are $\partial_{z_i} M_{uv} = d^{(i)}_v P_{uv}$, $\partial_{z_i}\partial_{z_j} M_{uv} = d^{(i)}_v d^{(j)}_v P_{uv}$, $\partial_\lambda M_{uv} = -M^{(1)}_{uv}$, $\partial_\lambda^2 M_{uv} = M^{(2)}_{uv}$ and $\partial_{z_i}\partial_\lambda M_{uv} = -d^{(i)}_v M^{(1)}_{uv}$.



The mean registered currents are $j_i = \lambda_i$, the diffusion matrix is $D_{ij} = \lambda_{ij}/2$, and the multidimensional uncertainty ratio of the two currents taken together is $\mathcal{U} = j^{\mathsf T} D^{-1} j$.

Returns
-------
A NumPy array of shape `(6,)` holding `[current_a, current_b, diffusion_aa, diffusion_bb, diffusion_ab, uncertainty_ratio]`, the first five in inverse microseconds and the last in Boltzmann constants per microsecond. Raise `ValueError` for a shape mismatch, a non-finite entry, a stationary characteristic determinant, a non-positive diagonal diffusion coefficient, or a diffusion matrix that is not positive definite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Joint counting cumulants of the two registered currents."""

import numpy as np

_INCREMENTS = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])


def counting_cumulants(moments):
    """Return the joint current cumulants and their multidimensional uncertainty ratio.

    Parameters
    ----------
    moments : array_like, shape (3, 4, 4)
        ``[P, M1, M2]`` of a semi-Markov kernel whose four states are the
        forward and reverse registered transitions of the two junctions.

    Returns
    -------
    numpy.ndarray, shape (6,)
        ``[current_a, current_b, diffusion_aa, diffusion_bb, diffusion_ab,
        uncertainty_ratio]``: the two mean registered currents in inverse
        microseconds, the three independent entries of their diffusion matrix
        in inverse microseconds, and the multidimensional uncertainty ratio of
        the two currents together in Boltzmann constants per microsecond.
    """
    return np.zeros(6)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _swap_columns(base, replacements):
    import numpy as np
    matrix = base.copy()
    for column, values in replacements.items():
        matrix[:, column] = values
    return matrix


def _first_variation(base, derivative):
    import numpy as np
    return sum(np.linalg.det(_swap_columns(base, {k: derivative[:, k]})) for k in range(base.shape[0]))


def _second_variation(base, first, other, mixed):
    import numpy as np
    size = base.shape[0]
    total = sum(np.linalg.det(_swap_columns(base, {k: mixed[:, k]})) for k in range(size))
    for k in range(size):
        for m in range(size):
            if k == m:
                continue
            total += np.linalg.det(_swap_columns(base, {k: first[:, k], m: other[:, m]}))
    return total


def _oracle_counting_cumulants(moments):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    transition, first, second = stack
    base = transition - np.eye(4)
    tilt = [transition * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] for i in range(2)]
    tilt_two = [[transition * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[j] for j in range(2)] for i in range(2)]
    lap = -first
    lap_two = second
    cross = [-first * np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])[i] for i in range(2)]
    grad_lap = _first_variation(base, lap)
    if abs(grad_lap) < 1e-300:
        raise ValueError("degenerate kernel: the characteristic determinant is stationary")
    current = np.array([-_first_variation(base, tilt[i]) / grad_lap for i in range(2)])
    diffusion = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            curvature = (_second_variation(base, tilt[i], tilt[j], tilt_two[i][j])
                         + _second_variation(base, tilt[i], lap, cross[i]) * current[j]
                         + _second_variation(base, lap, tilt[j], cross[j]) * current[i]
                         + _second_variation(base, lap, lap, lap_two) * current[i] * current[j])
            diffusion[i, j] = -0.5 * curvature / grad_lap
    if diffusion[0, 0] <= 0.0 or diffusion[1, 1] <= 0.0:
        raise ValueError("both diffusion coefficients must be positive")
    determinant = diffusion[0, 0] * diffusion[1, 1] - diffusion[0, 1] * diffusion[1, 0]
    if determinant <= 0.0:
        raise ValueError("the diffusion matrix must be positive definite")
    ratio = float(current @ np.linalg.solve(diffusion, current))
    return np.array([current[0], current[1], diffusion[0, 0], diffusion[1, 1],
                     0.5 * (diffusion[0, 1] + diffusion[1, 0]), ratio], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def _kernel_setup(rate_rows, etas):
    return (
        "import numpy as np\n"
        "K = np.array(%s, dtype=float)\n"
        "L = K - np.diag(K.sum(axis=1))\n"
        "eta = np.array(%r, dtype=float)\n"
        "pair = [(0, 1), (1, 0), (2, 3), (3, 2)]\n"
        "S = L.copy()\n"
        "for (a, b), e in zip(pair, eta):\n"
        "    S[a, b] -= e * L[a, b]\n"
        "w = np.concatenate([np.full(6, 0.004), np.full(5, 0.03), np.full(5, 0.16), np.full(4, 0.9)])\n"
        "bounds = np.concatenate([[0.0], np.cumsum(w)])\n"
        "t = 0.5 * (bounds[:-1] + bounds[1:])\n"
        "vals, vecs = np.linalg.eig(S)\n"
        "back = np.linalg.inv(vecs)\n"
        "flow = np.array([(vecs @ np.diag(np.exp(vals * x)) @ back).real for x in bounds])\n"
        "block = (np.linalg.inv(S) @ (flow[1:] - flow[:-1])).real\n"
        "rate = np.array([e * L[a, b] for (a, b), e in zip(pair, eta)])\n"
        "psi = np.zeros((4, 4, t.size))\n"
        "for u, (au, bu) in enumerate(pair):\n"
        "    for v, (av, bv) in enumerate(pair):\n"
        "        psi[u, v] = block[:, bu, av] * rate[v]\n"
        "psi = psi / w\n"
        "weight = psi * w\n"
        "mass = weight.sum(axis=2)\n"
        "first = (weight * t).sum(axis=2)\n"
        "second = (weight * t ** 2).sum(axis=2)\n"
        "scale = mass.sum(axis=1).reshape(4, 1)\n"
        "moments = np.stack([mass / scale, first / scale, second / scale])\n"
        % (rate_rows, etas)
    )


_NET_ONE = [[0.0, 3.4, 3.8, 0.0, 2.6],
            [5.5, 0.0, 3.1, 0.0, 0.0],
            [2.7, 4.2, 0.0, 5.6, 0.0],
            [0.0, 0.0, 3.7, 0.0, 6.2],
            [7.4, 0.0, 0.0, 7.1, 0.0]]
_NET_TWO = [[0.0, 2.2, 5.0, 0.0, 3.6],
            [6.8, 0.0, 2.4, 0.0, 0.0],
            [3.6, 5.1, 0.0, 7.0, 0.0],
            [0.0, 0.0, 2.8, 0.0, 5.5],
            [6.0, 0.0, 0.0, 8.4, 0.0]]
_NET_THREE = [[0.0, 4.1, 3.2, 0.0, 4.4],
              [4.9, 0.0, 3.8, 0.0, 0.0],
              [2.5, 3.9, 0.0, 4.8, 0.0],
              [0.0, 0.0, 4.6, 0.0, 7.6],
              [5.2, 0.0, 0.0, 6.4, 0.0]]


def test_cases():
    """Differential cases for the joint semi-Markov counting cumulants."""
    return [
        {
            "setup": _kernel_setup(_NET_ONE, [0.45, 0.85, 0.70, 0.50]),
            "call": "counting_cumulants(moments)",
            "gold_call": "_oracle_counting_cumulants(moments)",
        },
        {
            "setup": _kernel_setup(_NET_TWO, [0.62, 0.80, 0.55, 0.90]),
            "call": "counting_cumulants(moments)",
            "gold_call": "_oracle_counting_cumulants(moments)",
        },
        {
            "setup": _kernel_setup(_NET_THREE, [1.0, 1.0, 1.0, 1.0]),
            "call": "counting_cumulants(moments)",
            "gold_call": "_oracle_counting_cumulants(moments)",
        },
    ]
