#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def assemble_effective_stiffness(n_bodies: int, n_chords: int, h: float,
                                         beta: float, k_scale: float) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _kinematic_edges(n_bodies, n_chords):
        """Return the edge list of the kinematic constraint graph."""
        edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
        seen = set(edges)
        stride = max(1, n_bodies // max(1, n_chords))
        for c in range(n_chords):
            i = (c * stride) % n_bodies
            j = (i + n_bodies // 3 + c) % n_bodies
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
        return edges

    if not (isinstance(n_bodies, (int, np.integer)) and not isinstance(n_bodies, bool)
            and n_bodies >= 4):
        raise ValueError("n_bodies must be an integer >= 4")
    if not (isinstance(n_chords, (int, np.integer)) and not isinstance(n_chords, bool)
            and n_chords >= 0):
        raise ValueError("n_chords must be an integer >= 0")
    if not (isinstance(h, (int, float)) and np.isfinite(h) and float(h) > 0.0):
        raise ValueError("h must be a finite number > 0")
    if not (isinstance(beta, (int, float)) and np.isfinite(beta)
            and 0.0 < float(beta) <= 1.0):
        raise ValueError("beta must be a finite number in the half-open interval (0, 1]")
    if not (isinstance(k_scale, (int, float)) and np.isfinite(k_scale)
            and float(k_scale) > 0.0):
        raise ValueError("k_scale must be a finite number > 0")

    n_bodies = int(n_bodies)
    n_chords = int(n_chords)
    h = float(h)
    beta = float(beta)
    k_scale = float(k_scale)

    edges = _kinematic_edges(n_bodies, n_chords)
    n_dof = 6 * n_bodies
    stiffness = np.zeros((n_dof, n_dof), dtype=float)

    # Inertial term: block-diagonal generalised mass scaled by 1 / (beta * h^2).
    for b in range(n_bodies):
        mass = 1.0 + 0.5 * (b % 5)
        block = np.array([mass, mass, mass,
                          0.100 * mass, 0.125 * mass, 0.150 * mass], dtype=float)
        rows = np.arange(6 * b, 6 * b + 6)
        stiffness[rows, rows] += block / (beta * h * h)

    # Elastic term: one graph-Laplacian contribution per joint, with the joint
    # stiffness spread log-periodically over six orders of magnitude.
    n_edges = len(edges)
    for k, (i, j) in enumerate(edges):
        k_edge = k_scale * 10.0 ** (3.0 * np.cos(2.0 * np.pi * k / n_edges))
        for d in range(6):
            ii, jj = 6 * i + d, 6 * j + d
            stiffness[ii, ii] += k_edge
            stiffness[jj, jj] += k_edge
            stiffness[ii, jj] -= k_edge
            stiffness[jj, ii] -= k_edge

    return stiffness

def assemble_constraint_jacobian(n_bodies: int, n_chords: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _kinematic_edges(n_bodies, n_chords):
        """Return the edge list of the kinematic constraint graph."""
        edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
        seen = set(edges)
        stride = max(1, n_bodies // max(1, n_chords))
        for c in range(n_chords):
            i = (c * stride) % n_bodies
            j = (i + n_bodies // 3 + c) % n_bodies
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
        return edges

    if not (isinstance(n_bodies, (int, np.integer)) and not isinstance(n_bodies, bool)
            and n_bodies >= 4):
        raise ValueError("n_bodies must be an integer >= 4")
    if not (isinstance(n_chords, (int, np.integer)) and not isinstance(n_chords, bool)
            and n_chords >= 0):
        raise ValueError("n_chords must be an integer >= 0")

    n_bodies = int(n_bodies)
    n_chords = int(n_chords)

    edges = _kinematic_edges(n_bodies, n_chords)
    n_edges = len(edges)
    n_dof = 6 * n_bodies
    jacobian = np.zeros((3 * n_edges, n_dof), dtype=float)

    for k, (i, j) in enumerate(edges):
        phi = 2.0 * np.pi * (k + 1) / n_edges
        for d in range(3):
            row = 3 * k + d
            # Coincidence of the two attachment points along direction d.
            jacobian[row, 6 * i + d] = 1.0
            jacobian[row, 6 * j + d] = -1.0
            # Moment arms coupling the rotational coordinates of both bodies.
            jacobian[row, 6 * i + 3 + (d + 1) % 3] = np.sin(phi + d)
            jacobian[row, 6 * j + 3 + (d + 2) % 3] = -np.cos(phi + d)

    return jacobian

def build_strength_aggregates(matrix: np.ndarray, theta: float) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _strength_graph(matrix, theta):
        """Return the boolean strength-of-connection graph of a level operator."""
        magnitude = np.abs(matrix).copy()
        np.fill_diagonal(magnitude, 0.0)
        row_max = magnitude.max(axis=1)
        row_max = np.where(row_max > 0.0, row_max, 1.0)
        return (magnitude >= theta * row_max[:, None]) & (magnitude > 0.0)

    if not (isinstance(theta, (int, float)) and np.isfinite(theta)
            and 0.0 < float(theta) <= 1.0):
        raise ValueError("theta must be a finite number in the half-open interval (0, 1]")
    level = np.asarray(matrix, dtype=float)
    if level.ndim != 2 or level.shape[0] != level.shape[1] or level.shape[0] < 1:
        raise ValueError("matrix must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(level)):
        raise ValueError("matrix must be finite")

    n_dof = level.shape[0]
    strong = _strength_graph(level, float(theta))
    # Symmetrise: a pair is tied when either direction is strong.
    strong = strong | strong.T

    aggregates = -np.ones(n_dof, dtype=int)
    next_id = 0
    for i in range(n_dof):
        if aggregates[i] >= 0:
            continue
        neighbours = np.flatnonzero(strong[i] & (aggregates < 0))
        aggregates[i] = next_id
        aggregates[neighbours] = next_id
        next_id += 1

    return aggregates.astype(int)

def build_smoothed_prolongator(matrix: np.ndarray, aggregates: np.ndarray,
                                       omega: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(omega, (int, float)) and np.isfinite(omega)
            and 0.0 < float(omega) <= 1.0):
        raise ValueError("omega must be a finite number in the half-open interval (0, 1]")
    level = np.asarray(matrix, dtype=float)
    if level.ndim != 2 or level.shape[0] != level.shape[1] or level.shape[0] < 1:
        raise ValueError("matrix must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(level)):
        raise ValueError("matrix must be finite")
    labels = np.asarray(aggregates).ravel()
    if labels.size != level.shape[0]:
        raise ValueError("aggregates must have one entry per unknown of matrix")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("aggregates must be an integer array")
    if labels.min() != 0 or not np.array_equal(np.unique(labels),
                                               np.arange(labels.max() + 1)):
        raise ValueError("aggregates must hold consecutive indices starting at zero")

    diagonal = np.diag(level).copy()
    if np.any(diagonal == 0.0):
        raise ValueError("matrix must have a nonzero diagonal")

    n_dof = level.shape[0]
    n_coarse = int(labels.max()) + 1

    # Piecewise-constant tentative prolongator: one indicator column per aggregate.
    tentative = np.zeros((n_dof, n_coarse), dtype=float)
    tentative[np.arange(n_dof), labels] = 1.0

    # One damped Jacobi sweep applied to every column of the tentative operator.
    prolongator = tentative - float(omega) * (level @ tentative) / diagonal[:, None]

    return prolongator

def assemble_vcycle_operator(stiffness: np.ndarray, aggregates: np.ndarray,
                                     prolongator: np.ndarray, theta: float, omega: float,
                                     block_size: int, n_min: int,
                                     max_levels: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the coarsening steps 03 and 04 so that the hierarchy keeps no
    #    private copy of the coarsening rule. Preference order: (1) already in
    #    the executing namespace, (2) loaded from a sibling sub-problem file.
    #    The gold path never falls back to a public (submitted) implementation:
    #    binding one would make the differential comparison vacuous, so an
    #    unresolvable step raises instead.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        roots = []
        if "__file__" in namespace:
            roots.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        roots.append(os.getcwd())
        if sys.argv and sys.argv[0]:
            roots.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        # Each root, its sub_problems/ child and its parent (and that parent's
        # sub_problems/) are searched, so the gold resolves whether the harness
        # runs from the task root, from sub_problems/, or from a copy of this
        # file placed one level away from its siblings.
        for root in roots:
            parent = os.path.dirname(root)
            search_dirs += [root, os.path.join(root, "sub_problems"),
                            parent, os.path.join(parent, "sub_problems")]
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    aggregate_of = _resolve_step(
        "build_strength_aggregates", "*build_strength_aggregates*.py")
    prolongator_of = _resolve_step(
        "build_smoothed_prolongator", "*build_smoothed_prolongator*.py")

    def _symmetric_smoother(matrix, size):
        """Return the symmetric smoother built from the two triangular splittings."""
        n_dof = matrix.shape[0]
        width = size if (size > 1 and n_dof % size == 0) else 1
        block_of = np.arange(n_dof) // width
        diagonal_part = np.where(block_of[:, None] == block_of[None, :], matrix, 0.0)
        lower_part = np.where(block_of[:, None] > block_of[None, :], matrix, 0.0)
        upper_part = np.where(block_of[:, None] < block_of[None, :], matrix, 0.0)
        return (np.linalg.inv(diagonal_part + lower_part)
                + np.linalg.inv(diagonal_part + upper_part)
                - np.linalg.inv(diagonal_part))

    def _vcycle(matrix, transfer, level):
        """Recursively assemble the V-cycle operator of a level."""
        n_dof = matrix.shape[0]
        if n_dof <= n_min or level >= max_levels:
            return np.linalg.inv(matrix)

        if transfer is None:
            labels = aggregate_of(matrix, theta)
            if int(np.asarray(labels).max()) + 1 >= n_dof:
                return np.linalg.inv(matrix)
            transfer = prolongator_of(matrix, labels, omega)
        transfer = np.asarray(transfer, dtype=float)
        if transfer.shape[1] >= n_dof:
            return np.linalg.inv(matrix)

        coarse = transfer.T @ matrix @ transfer
        smoother = _symmetric_smoother(matrix, block_size if level == 0 else 1)
        coarse_inverse = _vcycle(coarse, None, level + 1)

        identity = np.eye(n_dof)
        # Pre-smooth from a zero guess, correct on the coarse level, then
        # post-smooth the corrected iterate against the same right-hand side.
        cycle = smoother
        coarse_correction = transfer @ coarse_inverse @ transfer.T
        cycle = cycle + coarse_correction @ (identity - matrix @ cycle)
        cycle = cycle + smoother @ (identity - matrix @ cycle)
        return cycle

    if not (isinstance(theta, (int, float)) and np.isfinite(theta)
            and 0.0 < float(theta) <= 1.0):
        raise ValueError("theta must be a finite number in the half-open interval (0, 1]")
    if not (isinstance(omega, (int, float)) and np.isfinite(omega)
            and 0.0 < float(omega) <= 1.0):
        raise ValueError("omega must be a finite number in the half-open interval (0, 1]")
    for name, value, floor in (("block_size", block_size, 1), ("n_min", n_min, 1),
                               ("max_levels", max_levels, 0)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    matrix = np.asarray(stiffness, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("stiffness must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("stiffness must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0.0):
        raise ValueError("stiffness must be symmetric")

    labels = np.asarray(aggregates).ravel()
    if labels.size != matrix.shape[0]:
        raise ValueError("aggregates must have one entry per unknown of stiffness")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("aggregates must be an integer array")
    transfer = np.asarray(prolongator, dtype=float)
    if transfer.ndim != 2 or transfer.shape[0] != matrix.shape[0]:
        raise ValueError("prolongator must have one row per unknown of stiffness")
    if transfer.shape[1] != int(labels.max()) + 1:
        raise ValueError("prolongator must have one column per aggregate")
    if not np.all(np.isfinite(transfer)):
        raise ValueError("prolongator must be finite")

    theta = float(theta)
    omega = float(omega)
    block_size = int(block_size)
    n_min = int(n_min)
    max_levels = int(max_levels)

    try:
        return _vcycle(matrix, transfer, 0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("a level operator of the hierarchy is singular") from exc

def compute_spectral_equivalence(stiffness: np.ndarray,
                                         vcycle_operator: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(stiffness, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("stiffness must be a square 2D array of order n >= 1")
    if cycle.shape != matrix.shape:
        raise ValueError("vcycle_operator must have the same shape as stiffness")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(cycle))):
        raise ValueError("stiffness and vcycle_operator must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0.0):
        raise ValueError("stiffness must be symmetric")

    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("stiffness must be positive definite") from exc

    # L^T M L is symmetric and similar to M K, so its spectrum is the one sought.
    similar = factor.T @ (0.5 * (cycle + cycle.T)) @ factor
    eigenvalues = np.linalg.eigvalsh(0.5 * (similar + similar.T))

    return float(np.max(np.abs(eigenvalues - 1.0)))

def partition_constraint_graph(jacobian: np.ndarray, dofs_per_body: int,
                                       max_cluster: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _find_root(parent, node):
        """Return the representative of a node with path compression."""
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for name, value in (("dofs_per_body", dofs_per_body), ("max_cluster", max_cluster)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= 1):
            raise ValueError(f"{name} must be an integer >= 1")
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("jacobian must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be finite")

    dofs_per_body = int(dofs_per_body)
    max_cluster = int(max_cluster)
    n_rows, n_cols = matrix.shape
    if n_cols % dofs_per_body != 0:
        raise ValueError("the column count of jacobian must be a multiple of dofs_per_body")
    n_bodies = n_cols // dofs_per_body

    # Every constraint row must touch exactly two bodies; that pair is its joint.
    touched = []
    for row in range(n_rows):
        bodies = np.unique(np.flatnonzero(matrix[row] != 0.0) // dofs_per_body)
        if bodies.size != 2:
            raise ValueError("every row of jacobian must touch exactly two bodies")
        touched.append((int(bodies[0]), int(bodies[1])))

    # Joints in order of first appearance, with the constraints they carry.
    joint_index = {}
    joint_rows = []
    for row, pair in enumerate(touched):
        if pair not in joint_index:
            joint_index[pair] = len(joint_rows)
            joint_rows.append([])
        joint_rows[joint_index[pair]].append(row)

    # Feedback edge set: a joint is cut when both its bodies already share a component.
    parent = list(range(n_bodies))
    cut = np.zeros(n_rows, dtype=int)
    for pair, index in sorted(joint_index.items(), key=lambda item: item[1]):
        root_i, root_j = _find_root(parent, pair[0]), _find_root(parent, pair[1])
        if root_i == root_j:
            for row in joint_rows[index]:
                cut[row] = 1
        else:
            parent[root_i] = root_j

    # Constraint adjacency: two constraints are tied when they share a body.
    body_sets = [set(pair) for pair in touched]
    neighbours = [[s for s in range(n_rows) if s != r and (body_sets[r] & body_sets[s])]
                  for r in range(n_rows)]

    # Greedy clustering in ascending index, uncut constraints first, then cut ones.
    labels = -np.ones(n_rows, dtype=int)
    next_id = 0
    for group in (np.flatnonzero(cut == 0), np.flatnonzero(cut == 1)):
        members = set(group.tolist())
        for row in group:
            if labels[row] >= 0:
                continue
            labels[row] = next_id
            size = 1
            for other in neighbours[row]:
                if size >= max_cluster:
                    break
                if other in members and labels[other] < 0:
                    labels[other] = next_id
                    size += 1
            next_id += 1

    return np.column_stack([labels, cut]).astype(int)

def assemble_schur_preconditioner(jacobian: np.ndarray, vcycle_operator: np.ndarray,
                                          partition: np.ndarray,
                                          eps_rel: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(eps_rel, (int, float)) and np.isfinite(eps_rel)
            and float(eps_rel) >= 0.0):
        raise ValueError("eps_rel must be a finite number >= 0")
    matrix = np.asarray(jacobian, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("jacobian must be a non-empty 2D array")
    if cycle.ndim != 2 or cycle.shape[0] != cycle.shape[1]:
        raise ValueError("vcycle_operator must be a square 2D array")
    if cycle.shape[0] != matrix.shape[1]:
        raise ValueError("vcycle_operator order must match the column count of jacobian")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(cycle))):
        raise ValueError("jacobian and vcycle_operator must be finite")

    table = np.asarray(partition)
    if table.ndim != 2 or table.shape != (matrix.shape[0], 2):
        raise ValueError("partition must have shape (m, 2) with m rows of jacobian")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("partition must be an integer array")
    labels = table[:, 0]
    cut = table[:, 1]
    if labels.min() != 0 or not np.array_equal(np.unique(labels),
                                               np.arange(labels.max() + 1)):
        raise ValueError("cluster indices must be consecutive integers starting at zero")
    if not np.all(np.isin(cut, (0, 1))):
        raise ValueError("cut flags must be zero or one")

    n_rows = matrix.shape[0]
    schur_prec = np.zeros((n_rows, n_rows), dtype=float)

    # Block-diagonal part: one local Schur complement per cluster.
    for cluster in range(int(labels.max()) + 1):
        rows = np.flatnonzero(labels == cluster)
        block = matrix[rows]
        schur_prec[np.ix_(rows, rows)] = block @ cycle @ block.T

    # Dense low-rank correction reinstating the couplings across kinematic loops.
    cut_rows = np.flatnonzero(cut == 1)
    if cut_rows.size > 0:
        block = matrix[cut_rows]
        schur_prec[np.ix_(cut_rows, cut_rows)] += block @ cycle @ block.T

    # Tikhonov shift set relative to the one-norm of the assembled matrix.
    shift = float(eps_rel) * float(np.linalg.norm(schur_prec, 1))
    schur_prec = schur_prec + shift * np.eye(n_rows)

    return schur_prec

def compute_spectral_deviation(stiffness: np.ndarray, jacobian: np.ndarray,
                                       vcycle_operator: np.ndarray,
                                       schur_prec: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    mechanical = np.asarray(stiffness, dtype=float)
    constraint = np.asarray(jacobian, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    schur = np.asarray(schur_prec, dtype=float)

    if mechanical.ndim != 2 or mechanical.shape[0] != mechanical.shape[1]:
        raise ValueError("stiffness must be a square 2D array")
    if constraint.ndim != 2 or constraint.shape[1] != mechanical.shape[0]:
        raise ValueError("jacobian must have as many columns as the order of stiffness")
    if cycle.shape != mechanical.shape:
        raise ValueError("vcycle_operator must have the same shape as stiffness")
    if schur.ndim != 2 or schur.shape != (constraint.shape[0], constraint.shape[0]):
        raise ValueError("schur_prec must be square of order the row count of jacobian")
    for name, value in (("stiffness", mechanical), ("jacobian", constraint),
                        ("vcycle_operator", cycle), ("schur_prec", schur)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    n_dof, n_con = mechanical.shape[0], constraint.shape[0]

    system = np.zeros((n_dof + n_con, n_dof + n_con), dtype=float)
    system[:n_dof, :n_dof] = mechanical
    system[:n_dof, n_dof:] = constraint.T
    system[n_dof:, :n_dof] = constraint

    # Closed-form inverse of the block lower triangular preconditioner.
    try:
        schur_inverse = np.linalg.inv(schur)
    except np.linalg.LinAlgError as exc:
        raise ValueError("schur_prec must be nonsingular") from exc

    preconditioner_inverse = np.zeros_like(system)
    preconditioner_inverse[:n_dof, :n_dof] = cycle
    preconditioner_inverse[n_dof:, :n_dof] = schur_inverse @ constraint @ cycle
    preconditioner_inverse[n_dof:, n_dof:] = -schur_inverse

    eigenvalues = np.linalg.eigvals(preconditioner_inverse @ system)
    magnitude = np.abs(eigenvalues)
    largest = float(np.max(magnitude))
    if largest == 0.0:
        raise ValueError("the preconditioned operator vanishes identically")
    nonzero = eigenvalues[magnitude > 1.0e-10 * largest]
    if nonzero.size == 0:
        raise ValueError("the preconditioned operator has no nonzero eigenvalue")

    return float(np.max(np.abs(nonzero - 1.0)))

def compute_bound_sharpness(gamma: float, deviation: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(gamma, (int, float, np.floating, np.integer))
            and not isinstance(gamma, bool) and np.isfinite(gamma)
            and 0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be a finite number in the half-open interval [0, 1)")
    if not (isinstance(deviation, (int, float, np.floating, np.integer))
            and not isinstance(deviation, bool) and np.isfinite(deviation)
            and float(deviation) >= 0.0):
        raise ValueError("deviation must be a finite number >= 0")

    gamma = float(gamma)
    bound = 0.5 * (gamma + np.sqrt(gamma * (4.0 + gamma)))
    if bound <= 0.0:
        raise ValueError("the theoretical bound vanishes, so the ratio is undefined")

    return float(float(deviation) / bound)

def run_bound_sharpness_pipeline(n_bodies: int = 24, n_chords: int = 4,
                                         h: float = 1.0e-3, beta: float = 0.25,
                                         k_scale: float = 1.0e8, theta: float = 0.25,
                                         omega: float = 0.6666666666666666,
                                         block_size: int = 6, n_min: int = 24,
                                         max_levels: int = 4, max_cluster: int = 6,
                                         eps_rel: float = 1.0e-8) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-10. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary).
    #    The gold path never falls back to a public (submitted) implementation:
    #    binding one would put the same defect on both sides of the differential
    #    comparison and make it vacuous, so an unresolvable step raises instead.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        roots = []
        if "__file__" in namespace:
            roots.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        roots.append(os.getcwd())
        if sys.argv and sys.argv[0]:
            roots.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        # Each root, its sub_problems/ child and its parent (and that parent's
        # sub_problems/) are searched, so the gold resolves whether the harness
        # runs from the task root, from sub_problems/, or from a copy of this
        # file placed one level away from its siblings.
        for root in roots:
            parent = os.path.dirname(root)
            search_dirs += [root, os.path.join(root, "sub_problems"),
                            parent, os.path.join(parent, "sub_problems")]
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    build_stiffness = _resolve_step(
        "assemble_effective_stiffness", "*assemble_effective_stiffness*.py")
    build_jacobian = _resolve_step(
        "assemble_constraint_jacobian", "*assemble_constraint_jacobian*.py")
    build_aggregates = _resolve_step(
        "build_strength_aggregates", "*build_strength_aggregates*.py")
    build_prolongator = _resolve_step(
        "build_smoothed_prolongator", "*build_smoothed_prolongator*.py")
    build_vcycle = _resolve_step(
        "assemble_vcycle_operator", "*assemble_vcycle_operator*.py")
    spectral_equivalence = _resolve_step(
        "compute_spectral_equivalence", "*compute_spectral_equivalence*.py")
    partition_graph = _resolve_step(
        "partition_constraint_graph", "*partition_constraint_graph*.py")
    build_schur = _resolve_step(
        "assemble_schur_preconditioner", "*assemble_schur_preconditioner*.py")
    spectral_deviation = _resolve_step(
        "compute_spectral_deviation", "*compute_spectral_deviation*.py")
    bound_sharpness = _resolve_step(
        "compute_bound_sharpness", "*compute_bound_sharpness*.py")

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_bodies", n_bodies, 4), ("n_chords", n_chords, 0),
                               ("block_size", block_size, 1), ("n_min", n_min, 1),
                               ("max_levels", max_levels, 0),
                               ("max_cluster", max_cluster, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("h", h), ("k_scale", k_scale)):
        if not (isinstance(value, (int, float)) and np.isfinite(value)
                and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("beta", beta), ("theta", theta), ("omega", omega)):
        if not (isinstance(value, (int, float)) and np.isfinite(value)
                and 0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    if not (isinstance(eps_rel, (int, float)) and np.isfinite(eps_rel)
            and float(eps_rel) >= 0.0):
        raise ValueError("eps_rel must be a finite number >= 0")

    # -- Sub-problems 01-02: the two blocks of the saddle-point system.
    stiffness = build_stiffness(int(n_bodies), int(n_chords), float(h),
                                float(beta), float(k_scale))
    jacobian = build_jacobian(int(n_bodies), int(n_chords))

    # -- Sub-problems 03-04: the finest-level coarsening, then 05: the V-cycle.
    aggregates = build_aggregates(stiffness, float(theta))
    prolongator = build_prolongator(stiffness, aggregates, float(omega))
    vcycle_operator = build_vcycle(stiffness, aggregates, prolongator,
                                   float(theta), float(omega),
                                   int(block_size), int(n_min), int(max_levels))

    # -- Sub-problem 06: quality of that approximate inverse.
    gamma = spectral_equivalence(stiffness, vcycle_operator)

    # -- Sub-problems 07-08: the sparsified, loop-corrected, regularised Schur block.
    partition = partition_graph(jacobian, 6, int(max_cluster))
    schur_prec = build_schur(jacobian, vcycle_operator, partition, float(eps_rel))

    # -- Sub-problem 09: spectrum of the preconditioned saddle-point operator.
    deviation = spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)

    # -- Sub-problem 10: observed spread measured against the theoretical bound.
    return float(bound_sharpness(gamma, deviation))
SCICODE_GOLD_EOF
