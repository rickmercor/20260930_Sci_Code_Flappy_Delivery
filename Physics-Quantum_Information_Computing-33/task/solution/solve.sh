#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def single_qubit_pauli(kind: int) -> np.ndarray:
    """Reference oracle for the four phase-free single-qubit Paulis."""
    import numpy as np

    tables = (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    k = int(kind)
    if k not in (0, 1, 2, 3):
        raise ValueError("kind must be 0, 1, 2, or 3 (I, X, Y, Z).")
    return tables[k].copy()

import numpy as np

def n_qubit_pauli(n: int, index: int) -> np.ndarray:
    """Reference oracle for the canonical Kronecker Pauli string."""
    import numpy as np

    n = int(n)
    if n < 1:
        raise ValueError("n must be an integer >= 1.")
    nmax = 4**n
    idx = int(index)
    if idx < 0 or idx >= nmax:
        raise ValueError("index must satisfy 0 <= index < 4**n.")

    pauli = globals().get("single_qubit_pauli")
    if not callable(pauli):
        try:
            import importlib.util
            from pathlib import Path

            path = Path(__file__).resolve().parent / "01_single_qubit_pauli.py"
            spec = importlib.util.spec_from_file_location("_o1", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pauli = getattr(mod, "single_qubit_pauli")
            globals()["single_qubit_pauli"] = pauli
        except Exception:
            tables = (
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
                np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
                np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
            )

            def pauli(kind: int) -> np.ndarray:
                k = int(kind)
                if k not in (0, 1, 2, 3):
                    raise ValueError("kind must be 0, 1, 2, or 3 (I, X, Y, Z).")
                return tables[k].copy()

    mat = pauli(idx % 4)
    idx //= 4
    for _ in range(1, n):
        mat = np.kron(pauli(idx % 4), mat)
        idx //= 4
    return mat

import numpy as np

def pauli_spectrum(psi: np.ndarray, n: int) -> np.ndarray:
    """Reference oracle for the signed Pauli spectrum."""
    import numpy as np

    n = int(n)
    if n < 1:
        raise ValueError("n must be an integer >= 1.")
    rho = np.asarray(psi, dtype=complex)
    dim = 2**n
    if rho.ndim != 2 or rho.shape != (dim, dim):
        raise ValueError("psi must have shape (2**n, 2**n).")
    if not np.all(np.isfinite(rho)):
        raise ValueError("psi must be finite.")

    # Step 02 is available in the concatenated Studio namespace; the inline builder is a
    # fallback only for isolated execution of this step.
    _npauli = globals().get("n_qubit_pauli")
    if not callable(_npauli):
        tables = (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        )

        def _npauli(n_loc: int, index: int) -> np.ndarray:
            n_loc = int(n_loc)
            idx = int(index)
            mat = tables[idx % 4]
            idx //= 4
            for _ in range(1, n_loc):
                mat = np.kron(tables[idx % 4], mat)
                idx //= 4
            return mat

    out = np.empty(4**n, dtype=float)
    for k in range(out.size):
        p = _npauli(n, k)
        out[k] = float(np.real(np.trace(rho @ p)))
    return out

import numpy as np

def bloch_density_matrix(r: np.ndarray) -> np.ndarray:
    """Reference oracle: rho = (I + r . sigma)/2 built from step 01."""
    import numpy as np

    vec = np.asarray(r, dtype=float).reshape(-1)
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        raise ValueError("r must be a finite real vector of shape (3,).")
    if float(np.linalg.norm(vec)) > 1.0 + 1e-9:
        raise ValueError("Bloch vector must have Euclidean norm at most 1.")

    # Step 01 is available in the concatenated Studio namespace; the inline table is a
    # fallback only for isolated execution of this step.
    _pauli = globals().get("single_qubit_pauli")
    if not callable(_pauli):
        tables = (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        )

        def _pauli(kind: int) -> np.ndarray:
            return tables[int(kind)].copy()

    rho = _pauli(0).astype(complex)
    for i in range(3):
        rho = rho + vec[i] * _pauli(i + 1)
    return 0.5 * rho

import numpy as np

def stabilizer_partition_function(spectrum: np.ndarray, beta: float) -> float:
    """Reference oracle: Z_beta = e^{-beta} sum_x cosh(beta x), overflow-free form."""
    import numpy as np

    spec = np.asarray(spectrum, dtype=float).reshape(-1)
    if spec.size == 0 or not np.all(np.isfinite(spec)):
        raise ValueError("spectrum must be a nonempty finite 1-D array.")
    size = int(spec.size)
    tmp = size
    n_ok = 0
    while tmp > 1:
        if tmp % 4 != 0:
            raise ValueError("spectrum length must equal 4**n for integer n >= 1.")
        tmp //= 4
        n_ok += 1
    if n_ok < 1:
        raise ValueError("spectrum length must equal 4**n for integer n >= 1.")
    if float(np.max(np.abs(spec))) > 1.0 + 1e-12:
        raise ValueError("Pauli expectation values must satisfy |x| <= 1.")
    b = float(beta)
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")
    a = np.abs(spec)
    # e^{-b} cosh(b a) = 0.5 (e^{-b(1-a)} + e^{-b(1+a)}); both exponents <= 0 for |a| <= 1.
    return float(0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a))))

import numpy as np

def decomposition_spf(
    r: np.ndarray, weights: np.ndarray, bloch_vectors: np.ndarray, beta: float
) -> float:
    """Reference oracle: validated decomposition average of pure-state Z_beta."""
    import numpy as np

    rv = np.asarray(r, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    vecs = np.asarray(bloch_vectors, dtype=float)
    b = float(beta)
    if rv.size != 3 or not np.all(np.isfinite(rv)) or float(np.linalg.norm(rv)) > 1.0 + 1e-9:
        raise ValueError("r must be a finite Bloch vector of shape (3,) with norm <= 1.")
    if w.size < 1 or not np.all(np.isfinite(w)) or np.any(w < -1e-12):
        raise ValueError("weights must be a nonempty finite nonnegative array.")
    if abs(float(np.sum(w)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.")
    if vecs.ndim != 2 or vecs.shape != (w.size, 3) or not np.all(np.isfinite(vecs)):
        raise ValueError("bloch_vectors must be a finite array of shape (m, 3).")
    if float(np.max(np.abs(np.linalg.norm(vecs, axis=1) - 1.0))) > 1e-9:
        raise ValueError("every Bloch vector of a pure component must have unit norm.")
    if float(np.max(np.abs(w @ vecs - rv))) > 1e-8:
        raise ValueError("the decomposition does not reproduce the Bloch vector r.")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")

    # Steps 03, 04 and 05 are available in the concatenated Studio namespace; the inline
    # formula below is a fallback only for isolated execution of this step.
    rho_fn = globals().get("bloch_density_matrix")
    spec_fn = globals().get("pauli_spectrum")
    z_fn = globals().get("stabilizer_partition_function")

    total = 0.0
    for p, n in zip(w, vecs):
        n = n / float(np.linalg.norm(n))
        if callable(rho_fn) and callable(spec_fn) and callable(z_fn):
            spec = np.asarray(spec_fn(rho_fn(n), 1), dtype=float)
            z = float(z_fn(spec, b))
        else:
            spec = np.concatenate(([1.0], n))
            a = np.abs(spec)
            z = float(0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a))))
        total += float(p) * z
    return float(total)

import numpy as np

def roof_spf(r: np.ndarray, beta: float, grid_points: int = 4000) -> float:
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
    dec_fn = globals().get("decomposition_spf")

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

import numpy as np

def noisy_magic_roof_work(r: np.ndarray = (0.6, 0.5, 0.3), beta: float = 2.0) -> float:
    """Reference oracle: W = log2(Z^c_STAB_1 / (Z_roof - 4 e^{-beta})) via step 07."""
    import numpy as np

    rv = np.asarray(r, dtype=float).reshape(-1)
    b = float(beta)
    if rv.size != 3 or not np.all(np.isfinite(rv)) or float(np.linalg.norm(rv)) > 1.0 + 1e-9:
        raise ValueError("r must be a finite Bloch vector of shape (3,) with norm <= 1.")
    if not np.isfinite(b) or b < 0.01 or b > 10.0:
        raise ValueError("beta must be finite with 0.01 <= beta <= 10.")

    # Step 07 is available in the concatenated Studio namespace.
    roof_fn = globals().get("roof_spf")
    if not callable(roof_fn):
        raise RuntimeError("the roof partition function of step 07 is required.")

    z_roof = float(roof_fn(rv, b))
    zc = z_roof - 4.0 * float(np.exp(-b))
    # Reference: 2 e^{-b}(cosh b - 1) = expm1(-b)**2, evaluated without cancellation.
    zc_ref = float(np.expm1(-b) ** 2)
    if not np.isfinite(zc) or zc <= 0.0 or zc_ref <= 0.0:
        raise RuntimeError("core partition functions must be positive before log2.")
    work = float(np.log2(zc_ref / zc))
    # Free states: the roof attains the stabilizer value, so the work is exactly 0.
    if float(np.sum(np.abs(rv))) <= 1.0 + 1e-12 or abs(work) < 1e-9:
        return 0.0
    if work < 0.0:
        raise RuntimeError("negative work indicates a roof above the stabilizer value.")
    # Consistency: the roof is at least the eigendecomposition average.
    unit = rv / float(np.linalg.norm(rv))
    a = np.abs(unit)
    z_eig = float(
        0.5 * (1.0 + np.exp(-2.0 * b))
        + 0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a)))
    )
    work_eig = float(np.log2(zc_ref / (z_eig - 4.0 * float(np.exp(-b)))))
    if work > work_eig + 1e-9:
        raise RuntimeError("roof work exceeds the eigendecomposition work; numerical failure.")
    return work
SCICODE_GOLD_EOF
