"""
Mixed-state stabilizer partition function of a single qubit (Definition 8).

Definition 8 of the source paper extends the stabilizer partition function to a mixed state rho as the supremum, over all convex decompositions of rho into pure states, of the decomposition average of the pure-state partition functions. For a single qubit the decompositions are the ways of writing the Bloch vector r as a probability mixture of unit vectors, so the mixed-state value is the concave roof of the pure-state function n -> Z_beta(n) over the Bloch sphere, evaluated at r. The eigendecomposition is only one admissible decomposition and in general does not attain the supremum. A state whose Bloch vector lies inside the stabilizer octahedron |r_x| + |r_y| + |r_z| <= 1 is a mixture of stabilizer states and attains the stabilizer value; a pure state attains its own Definition 3 value.

The supremum is over an uncountable set, but by Caratheodory's theorem it is attained by a decomposition with at most four pure components. A reliable evaluation therefore needs a global optimisation, for example a linear program over a dense candidate set of pure states followed by a continuous refinement of the resulting support.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def roof_spf(r: np.ndarray, beta: float, grid_points: int = 4000) -> float:
    """Mixed-state stabilizer partition function of a single qubit.

    Return Z_beta(rho) of Definition 8 of the source paper for the
    single-qubit state with Bloch vector ``r``: the supremum over all
    convex decompositions rho = sum_i p_i |phi_i><phi_i| into pure
    states of sum_i p_i Z_beta(phi_i), with Z_beta the pure-state
    quantity of ``stabilizer_partition_function`` (evaluate the
    components through ``decomposition_spf``). The result must equal
    the pure-state value when ``r`` is a unit vector, must equal the
    stabilizer value e^{-beta}(2 cosh beta + 2) when ``r`` lies inside
    the stabilizer octahedron |r_x| + |r_y| + |r_z| <= 1, and must be
    at least the eigendecomposition average for every ``r``.

    The supremum is attained by a decomposition with at most four
    components; locate it globally (for example a linear program over
    ``grid_points`` candidate pure states on the Bloch sphere, then a
    continuous refinement of the support) and return the value to an
    absolute accuracy of 1e-9 or better.

    Supported domain: ``r`` finite real of shape (3,) with Euclidean
    norm at most 1; ``beta`` finite and >= 0; ``grid_points`` integer
    >= 500. Raise ValueError otherwise.

    Parameters
    ----------
    r : np.ndarray
        Bloch vector of the state, shape (3,).
    beta : float
        Inverse-temperature parameter, finite and >= 0.
    grid_points : int, optional
        Size of the initial candidate set of pure states, default 4000.

    Returns
    -------
    z_roof : float
        Mixed-state stabilizer partition function Z_beta(rho).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_roof_spf(r: np.ndarray, beta: float, grid_points: int = 4000) -> float:
    """Reference oracle: concave roof via grid LP, support refinement, two-grid agreement."""
    import numpy as np
    from scipy.optimize import linprog, minimize

    rv = np.asarray(r, dtype=float).reshape(-1)
    b = float(beta)
    npts = int(grid_points)
    if rv.size != 3 or not np.all(np.isfinite(rv)) or float(np.linalg.norm(rv)) > 1.0 + 1e-9:
        raise ValueError("r must be a finite Bloch vector of shape (3,) with norm <= 1.")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")
    if npts < 500:
        raise ValueError("grid_points must be an integer >= 500.")

    # Step 06 is available in the concatenated Studio namespace; the vectorised formula
    # below is used for the candidate set and as a fallback for isolated execution.
    dec_fn = globals().get("_oracle_decomposition_spf")

    def _z_pure(vectors: np.ndarray) -> np.ndarray:
        # Definition 3 on the spectrum {1, n_x, n_y, n_z} of a pure qubit, overflow-free.
        a = np.abs(np.asarray(vectors, dtype=float))
        ident = 0.5 * (1.0 + np.exp(-2.0 * b))
        return ident + 0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a)), axis=-1)

    def _z_decomposition(weights: np.ndarray, vectors: np.ndarray) -> float:
        if callable(dec_fn):
            try:
                return float(dec_fn(rv, weights, vectors, b))
            except ValueError:
                pass
        return float(np.sum(weights * _z_pure(vectors)))

    nrm = float(np.linalg.norm(rv))
    if nrm >= 1.0 - 1e-12:
        unit = rv / nrm
        return _z_decomposition(np.array([1.0]), unit.reshape(1, 3))

    def _fibonacci_sphere(count: int) -> np.ndarray:
        idx = np.arange(count, dtype=float) + 0.5
        phi = np.arccos(1.0 - 2.0 * idx / count)
        theta = np.pi * (1.0 + 5.0**0.5) * idx
        return np.stack(
            [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
        )

    def _solve(count: int) -> float:
        grid = _fibonacci_sphere(count)
        # Include every stabilizer vertex and an exact feasible decomposition of rv.
        grid = np.vstack([grid, np.eye(3), -np.eye(3)])
        if nrm > 0.0:
            direction = rv / nrm
            grid = np.vstack([grid, direction, -direction])
        count = grid.shape[0]
        values = _z_pure(grid)
        a_eq = np.vstack([grid.T, np.ones(count)])
        b_eq = np.append(rv, 1.0)
        res = linprog(-values, A_eq=a_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
        if not res.success:
            raise RuntimeError("linear program over the candidate set failed: " + str(res.message))
        weights = np.asarray(res.x, dtype=float)
        keep = np.where(weights > 1e-10)[0]
        if keep.size == 0:
            raise RuntimeError("linear program returned an empty support.")
        sup_vecs = grid[keep]
        sup_w = weights[keep] / float(np.sum(weights[keep]))
        best = float(np.sum(sup_w * _z_pure(sup_vecs)))
        k = keep.size

        def _unpack(x: np.ndarray):
            vecs = x[: 3 * k].reshape(k, 3)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs, x[3 * k :]

        def _objective(x: np.ndarray) -> float:
            vecs, w = _unpack(x)
            return -float(np.sum(w * _z_pure(vecs)))

        constraints = [
            {"type": "eq", "fun": lambda x: (lambda v, w: w @ v - rv)(*_unpack(x))},
            {"type": "eq", "fun": lambda x: float(np.sum(_unpack(x)[1])) - 1.0},
        ]
        x0 = np.concatenate([sup_vecs.ravel(), sup_w])
        bounds = [(None, None)] * (3 * k) + [(0.0, 1.0)] * k
        try:
            pol = minimize(
                _objective,
                x0,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"ftol": 1e-16, "maxiter": 5000},
            )
            vecs, w = _unpack(pol.x)
            w = np.clip(w, 0.0, None)
            if float(np.sum(w)) > 0.0:
                w = w / float(np.sum(w))
                if float(np.max(np.abs(w @ vecs - rv))) <= 1e-8:
                    cand = _z_decomposition(w, vecs)
                    if np.isfinite(cand) and cand > best:
                        best = cand
        except Exception:
            pass
        return best

    z1 = _solve(npts)
    z2 = _solve(int(round(1.5 * npts)))
    z_roof = max(z1, z2)
    # Stabilizer value e^{-b}(2 cosh b + 2) = 1 + e^{-2b} + 2 e^{-b}, the maximum of any pure Z_b.
    z_stab = float(1.0 + np.exp(-2.0 * b) + 2.0 * np.exp(-b))
    # Every pure Z_beta is at most the stabilizer value, hence so is any decomposition average.
    if z_roof > z_stab + 1e-9:
        raise RuntimeError("roof exceeds the stabilizer value; numerical failure.")
    return float(min(z_roof, z_stab))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = "float(np.round({fn}(r, beta), 8))"
    return [
        {
            # Task state at beta = 2: roof 1.26827898 (eigendecomposition gives only 1.22498284).
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\nbeta = 2.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            # Free state inside the octahedron: stabilizer value e^{-2}(2 cosh 2 + 2) = 1.28898621.
            "setup": "import numpy as np\nr = np.array([0.3, 0.3, 0.3])\nbeta = 2.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            # Pure T state: the roof of a pure state is its Definition 3 value 1.23406328.
            "setup": "import numpy as np\nr = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\nbeta = 2.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            # Depolarised T state (Bloch length 0.8) at beta = 2: 1.28467158.
            "setup": "import numpy as np\nr = 0.8 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\nbeta = 2.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            # Task state at beta = 1: 1.86805092.
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\nbeta = 1.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            # beta = 0 boundary: every pure Z_0 equals 4, so the roof is 4.
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\nbeta = 0.0\n",
            "call": pair.format(fn="roof_spf"),
            "gold_call": pair.format(fn="_oracle_roof_spf"),
        },
        {
            "setup": "import numpy as np\nr = np.array([0.0, 0.0, 0.999999])\nbeta = 2.0\n",
            "call": "float(np.round(roof_spf(r, beta), 8))",
            "gold_call": "float(np.round(_oracle_roof_spf(r, beta), 8))",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.3, 0.3, 0.3])\nbeta = 1000000.0\n",
            "call": "float(np.round(roof_spf(r, beta), 8))",
            "gold_call": "float(np.round(_oracle_roof_spf(r, beta), 8))",
        },
    ]
