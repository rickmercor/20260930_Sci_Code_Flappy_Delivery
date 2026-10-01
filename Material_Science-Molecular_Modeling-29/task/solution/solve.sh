#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_monomer_reference_geometry(d_r: float = 116.09, alpha_r: float = 2.287,
                                               r_eq: float = 0.9419, k_theta: float = 87.85,
                                               theta_eq_deg: float = 107.4) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("d_r", d_r), ("alpha_r", alpha_r), ("r_eq", r_eq),
                        ("k_theta", k_theta)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(theta_eq_deg, (int, float, np.floating, np.integer))
            and not isinstance(theta_eq_deg, bool) and np.isfinite(theta_eq_deg)
            and 0.0 < float(theta_eq_deg) < 180.0):
        raise ValueError("theta_eq_deg must be a finite number strictly between 0 and 180")

    d_r, alpha_r, r_eq = float(d_r), float(alpha_r), float(r_eq)
    k_theta = float(k_theta)
    theta_eq = np.deg2rad(float(theta_eq_deg))

    # Newton minimization of the separable isolated-monomer energy in the three
    # internal coordinates, using the first two derivatives of the quartic Morse
    # stretch. Started away from the analytic stationary point so that the
    # iteration is genuinely exercised.
    bonds = np.array([1.05 * r_eq, 0.95 * r_eq], dtype=float)
    angle = theta_eq + 0.15
    for _ in range(200):
        shift = 0.0
        for index in range(2):
            x = bonds[index] - r_eq
            first = d_r * (2.0 * alpha_r ** 2 * x - 3.0 * alpha_r ** 3 * x ** 2
                           + (7.0 / 3.0) * alpha_r ** 4 * x ** 3)
            second = d_r * (2.0 * alpha_r ** 2 - 6.0 * alpha_r ** 3 * x
                            + 7.0 * alpha_r ** 4 * x ** 2)
            if second <= 0.0:
                raise ValueError("stretch curvature is non-positive; cannot minimize")
            step = first / second
            bonds[index] -= step
            shift = max(shift, abs(step))
        step = (k_theta * (angle - theta_eq)) / k_theta
        angle -= step
        shift = max(shift, abs(step))
        if shift < 1.0e-15:
            break
    else:
        raise ValueError("isolated-monomer minimization failed to converge")

    if np.any(bonds <= 0.0) or not (0.0 < angle < np.pi):
        raise ValueError("minimization left the physical range of the internal coordinates")

    # Paper-I Jacobi body frame: O is the position, the H-O-H bisector is +y,
    # and H2-H1 is +x.  For the optimized symmetric monomer those axes are
    # exactly orthogonal.
    half = 0.5 * angle
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [-bonds[0] * np.sin(half), bonds[0] * np.cos(half), 0.0],
        [bonds[1] * np.sin(half), bonds[1] * np.cos(half), 0.0],
    ], dtype=float)
    return geometry

def compute_potential_energy(coords: np.ndarray, d_r: float = 116.09,
                                     alpha_r: float = 2.287, r_eq: float = 0.9419,
                                     k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                                     epsilon: float = 0.1852, sigma: float = 3.1589,
                                     q_m: float = 1.1128, gamma: float = 0.73612,
                                     coulomb_constant: float = 332.06371) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of O, H, H monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    for name, value in (("d_r", d_r), ("alpha_r", alpha_r), ("r_eq", r_eq),
                        ("k_theta", k_theta), ("epsilon", epsilon), ("sigma", sigma),
                        ("q_m", q_m), ("coulomb_constant", coulomb_constant)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(theta_eq_deg, (int, float, np.floating, np.integer))
            and not isinstance(theta_eq_deg, bool) and np.isfinite(theta_eq_deg)
            and 0.0 < float(theta_eq_deg) < 180.0):
        raise ValueError("theta_eq_deg must be a finite number strictly between 0 and 180")
    if not (isinstance(gamma, (int, float, np.floating, np.integer))
            and not isinstance(gamma, bool) and np.isfinite(gamma) and 0.0 < float(gamma) < 1.0):
        raise ValueError("gamma must be a finite number strictly between 0 and 1")

    d_r, alpha_r, r_eq = float(d_r), float(alpha_r), float(r_eq)
    k_theta, theta_eq = float(k_theta), np.deg2rad(float(theta_eq_deg))
    epsilon, sigma = float(epsilon), float(sigma)
    q_m, gamma, coulomb = float(q_m), float(gamma), float(coulomb_constant)

    n_monomers = coords.shape[0] // 3
    energy = 0.0

    # -- Intramolecular: two quartic-Morse stretches and one harmonic bend.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        for i_h in (i_h1, i_h2):
            bond = float(np.linalg.norm(coords[i_h] - coords[i_o]))
            if bond <= 0.0:
                raise ValueError("an O-H bond has zero length")
            x = bond - r_eq
            energy += d_r * (alpha_r ** 2 * x ** 2 - alpha_r ** 3 * x ** 3
                             + (7.0 / 12.0) * alpha_r ** 4 * x ** 4)

        arm_1 = coords[i_h1] - coords[i_o]
        arm_2 = coords[i_h2] - coords[i_o]
        cosine = float(np.clip(np.dot(arm_1, arm_2)
                               / (np.linalg.norm(arm_1) * np.linalg.norm(arm_2)), -1.0, 1.0))
        energy += 0.5 * k_theta * (np.arccos(cosine) - theta_eq) ** 2

    # -- Intermolecular dispersion/repulsion: oxygen pairs only.
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            distance = float(np.linalg.norm(coords[3 * mono_j] - coords[3 * mono_i]))
            if distance <= 0.0:
                raise ValueError("two oxygen atoms coincide")
            ratio6 = (sigma / distance) ** 6
            energy += 4.0 * epsilon * (ratio6 * ratio6 - ratio6)

    # -- Intermolecular electrostatics on the H, H and M sites of each monomer.
    site_charges = np.array([0.5 * q_m, 0.5 * q_m, -q_m], dtype=float)
    sites = np.zeros((n_monomers, 3, 3), dtype=float)
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        sites[mono, 0] = coords[i_h1]
        sites[mono, 1] = coords[i_h2]
        sites[mono, 2] = gamma * coords[i_o] + (1.0 - gamma) * 0.5 * (coords[i_h1] + coords[i_h2])

    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            for site_a in range(3):
                for site_b in range(3):
                    distance = float(np.linalg.norm(sites[mono_j, site_b] - sites[mono_i, site_a]))
                    if distance <= 0.0:
                        raise ValueError("two charge sites coincide")
                    energy += coulomb * site_charges[site_a] * site_charges[site_b] / distance

    return float(energy)

def compute_potential_gradient(coords: np.ndarray, d_r: float = 116.09,
                                       alpha_r: float = 2.287, r_eq: float = 0.9419,
                                       k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                                       epsilon: float = 0.1852, sigma: float = 3.1589,
                                       q_m: float = 1.1128, gamma: float = 0.73612,
                                       coulomb_constant: float = 332.06371) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of O, H, H monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    for name, value in (("d_r", d_r), ("alpha_r", alpha_r), ("r_eq", r_eq),
                        ("k_theta", k_theta), ("epsilon", epsilon), ("sigma", sigma),
                        ("q_m", q_m), ("coulomb_constant", coulomb_constant)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(theta_eq_deg, (int, float, np.floating, np.integer))
            and not isinstance(theta_eq_deg, bool) and np.isfinite(theta_eq_deg)
            and 0.0 < float(theta_eq_deg) < 180.0):
        raise ValueError("theta_eq_deg must be a finite number strictly between 0 and 180")
    if not (isinstance(gamma, (int, float, np.floating, np.integer))
            and not isinstance(gamma, bool) and np.isfinite(gamma) and 0.0 < float(gamma) < 1.0):
        raise ValueError("gamma must be a finite number strictly between 0 and 1")

    d_r, alpha_r, r_eq = float(d_r), float(alpha_r), float(r_eq)
    k_theta, theta_eq = float(k_theta), np.deg2rad(float(theta_eq_deg))
    epsilon, sigma = float(epsilon), float(sigma)
    q_m, gamma, coulomb = float(q_m), float(gamma), float(coulomb_constant)

    n_monomers = coords.shape[0] // 3
    gradient = np.zeros_like(coords)

    # -- Intramolecular: two quartic-Morse stretches and one harmonic bend.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        for i_h in (i_h1, i_h2):
            offset = coords[i_h] - coords[i_o]
            bond = float(np.linalg.norm(offset))
            if bond <= 0.0:
                raise ValueError("an O-H bond has zero length")
            x = bond - r_eq
            d_energy = d_r * (2.0 * alpha_r ** 2 * x - 3.0 * alpha_r ** 3 * x ** 2
                              + (7.0 / 3.0) * alpha_r ** 4 * x ** 3)
            unit = offset / bond
            gradient[i_h] += d_energy * unit
            gradient[i_o] -= d_energy * unit

        arm_1 = coords[i_h1] - coords[i_o]
        arm_2 = coords[i_h2] - coords[i_o]
        len_1, len_2 = np.linalg.norm(arm_1), np.linalg.norm(arm_2)
        unit_1, unit_2 = arm_1 / len_1, arm_2 / len_2
        cosine = float(np.clip(np.dot(unit_1, unit_2), -1.0, 1.0))
        angle = np.arccos(cosine)
        sine = np.sqrt(max(1.0 - cosine * cosine, 1.0e-300))
        d_energy = k_theta * (angle - theta_eq)
        d_angle_1 = -(unit_2 - cosine * unit_1) / (len_1 * sine)
        d_angle_2 = -(unit_1 - cosine * unit_2) / (len_2 * sine)
        gradient[i_h1] += d_energy * d_angle_1
        gradient[i_h2] += d_energy * d_angle_2
        gradient[i_o] -= d_energy * (d_angle_1 + d_angle_2)

    # -- Intermolecular dispersion/repulsion: oxygen pairs only.
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            i_o, j_o = 3 * mono_i, 3 * mono_j
            offset = coords[j_o] - coords[i_o]
            distance = float(np.linalg.norm(offset))
            if distance <= 0.0:
                raise ValueError("two oxygen atoms coincide")
            ratio6 = (sigma / distance) ** 6
            d_energy = 4.0 * epsilon * (-12.0 * ratio6 * ratio6 + 6.0 * ratio6) / distance
            unit = offset / distance
            gradient[j_o] += d_energy * unit
            gradient[i_o] -= d_energy * unit

    # -- Intermolecular electrostatics on the H, H and M sites of each monomer.
    site_charges = np.array([0.5 * q_m, 0.5 * q_m, -q_m], dtype=float)
    sites = np.zeros((n_monomers, 3, 3), dtype=float)
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        sites[mono, 0] = coords[i_h1]
        sites[mono, 1] = coords[i_h2]
        sites[mono, 2] = gamma * coords[i_o] + (1.0 - gamma) * 0.5 * (coords[i_h1] + coords[i_h2])

    site_gradient = np.zeros_like(sites)
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            for site_a in range(3):
                for site_b in range(3):
                    offset = sites[mono_j, site_b] - sites[mono_i, site_a]
                    distance = float(np.linalg.norm(offset))
                    if distance <= 0.0:
                        raise ValueError("two charge sites coincide")
                    scale = coulomb * site_charges[site_a] * site_charges[site_b]
                    d_energy = -scale / distance ** 2
                    unit = offset / distance
                    site_gradient[mono_j, site_b] += d_energy * unit
                    site_gradient[mono_i, site_a] -= d_energy * unit

    # The massless site is a fixed linear combination of the three atoms, so its
    # force is distributed by constant weights and carries no geometric factor.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        gradient[i_h1] += site_gradient[mono, 0] + (1.0 - gamma) * 0.5 * site_gradient[mono, 2]
        gradient[i_h2] += site_gradient[mono, 1] + (1.0 - gamma) * 0.5 * site_gradient[mono, 2]
        gradient[i_o] += gamma * site_gradient[mono, 2]

    return gradient

def build_reference_configuration(coords: np.ndarray,
                                          monomer_geometry: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    monomer_geometry = np.asarray(monomer_geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if monomer_geometry.shape != (3, 3):
        raise ValueError("monomer_geometry must have shape (3, 3)")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(monomer_geometry)):
        raise ValueError("coords and monomer_geometry must contain only finite entries")

    def jacobi_frame(block):
        bisector = 0.5 * (block[1] + block[2]) - block[0]
        bisector_norm = float(np.linalg.norm(bisector))
        if bisector_norm <= 1.0e-14:
            raise ValueError("the oxygen-to-hydrogen-midpoint vector must be nonzero")
        axis_y = bisector / bisector_norm

        difference = block[2] - block[1]
        projected = difference - float(np.dot(difference, axis_y)) * axis_y
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= 1.0e-14:
            raise ValueError("the projected H2-H1 vector must be nonzero")
        axis_x = projected / projected_norm
        axis_z = np.cross(axis_x, axis_y)
        return np.column_stack((axis_x, axis_y, axis_z))

    n_monomers = coords.shape[0] // 3
    reference_coords = np.zeros_like(coords)

    reference_origin = monomer_geometry[0]
    reference_relative = monomer_geometry - reference_origin
    reference_frame = jacobi_frame(monomer_geometry)

    for mono in range(n_monomers):
        block = coords[3 * mono: 3 * mono + 3]
        target_frame = jacobi_frame(block)
        body_coordinates = reference_relative @ reference_frame
        reference_coords[3 * mono: 3 * mono + 3] = (
            body_coordinates @ target_frame.T + block[0])

    return reference_coords

def compute_block_hessian(coords: np.ndarray, atom_masses: np.ndarray,
                                  gradient_fn, step: float = 1.0e-5) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not callable(gradient_fn):
        raise ValueError("gradient_fn must be callable")
    if not (isinstance(step, (int, float, np.floating, np.integer))
            and not isinstance(step, bool) and np.isfinite(step) and float(step) > 0.0):
        raise ValueError("step must be a finite number > 0")

    step = float(step)

    n_monomers = coords.shape[0] // 3
    blocks = np.zeros((n_monomers, 9, 9), dtype=float)

    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        block = np.zeros((9, 9), dtype=float)
        # Displace only this monomer's coordinates and read back only this
        # monomer's gradient components: that is exactly the diagonal block.
        for local_atom in range(3):
            for axis in range(3):
                column = 3 * local_atom + axis
                forward = coords.copy()
                forward[rows[local_atom], axis] += step
                backward = coords.copy()
                backward[rows[local_atom], axis] -= step
                grad_forward = gradient_fn(forward)
                grad_backward = gradient_fn(backward)
                difference = (grad_forward[rows] - grad_backward[rows]) / (2.0 * step)
                block[:, column] = difference.reshape(9)

        block = 0.5 * (block + block.T)
        inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
        blocks[mono] = block * np.outer(inverse_root, inverse_root)

    return blocks

def build_subspace_pseudoinverse(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2] or blocks.shape[1] < 1:
        raise ValueError("blocks must have shape (n_monomers, n_dim, n_dim) with n_dim >= 1")
    if blocks.shape[0] < 1:
        raise ValueError("blocks must contain at least one block")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks must contain only finite entries")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)):
        raise ValueError("n_stiff must be an integer")
    n_stiff = int(n_stiff)
    if not 1 <= n_stiff <= blocks.shape[1]:
        raise ValueError("n_stiff must lie between 1 and the block dimension")

    pseudoinverse = np.zeros_like(blocks)
    for index in range(blocks.shape[0]):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        # Rank by eigenvalue and keep the stiffest directions; a tolerance-based
        # selection is not equivalent, because the block is not positive
        # semi-definite.
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("the retained stiff subspace contains a non-positive eigenvalue")
        selected_values = eigenvalues[order]
        selected_vectors = eigenvectors[:, order]
        pseudoinverse[index] = (selected_vectors / selected_values) @ selected_vectors.T

    return pseudoinverse

def relax_subspace_newton_raphson(coords: np.ndarray, pseudoinverse: np.ndarray,
                                          atom_masses: np.ndarray, gradient_fn,
                                          n_iterations: int = 2) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    pseudoinverse = np.asarray(pseudoinverse, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    n_monomers = coords.shape[0] // 3
    if pseudoinverse.shape != (n_monomers, 9, 9):
        raise ValueError("pseudoinverse must have shape (n_monomers, 9, 9)")
    if not np.all(np.isfinite(pseudoinverse)):
        raise ValueError("pseudoinverse must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not callable(gradient_fn):
        raise ValueError("gradient_fn must be callable")
    if not (isinstance(n_iterations, (int, np.integer)) and not isinstance(n_iterations, bool)
            and int(n_iterations) >= 0):
        raise ValueError("n_iterations must be an integer >= 0")

    n_iterations = int(n_iterations)

    relaxed_coords = coords.copy()

    for _ in range(n_iterations):
        gradient = gradient_fn(relaxed_coords)
        displacement = np.zeros_like(relaxed_coords)
        for mono in range(n_monomers):
            rows = np.arange(3 * mono, 3 * mono + 3)
            inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
            scaled_gradient = inverse_root * gradient[rows].reshape(9)
            displacement[rows] = (inverse_root
                                  * (pseudoinverse[mono] @ scaled_gradient)).reshape(3, 3)
        relaxed_coords = relaxed_coords - displacement

    return relaxed_coords

def compute_stiff_mode_frequencies(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2] or blocks.shape[1] < 1:
        raise ValueError("blocks must have shape (n_monomers, n_dim, n_dim) with n_dim >= 1")
    if blocks.shape[0] < 1:
        raise ValueError("blocks must contain at least one block")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks must contain only finite entries")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)):
        raise ValueError("n_stiff must be an integer")
    n_stiff = int(n_stiff)
    if not 1 <= n_stiff <= blocks.shape[1]:
        raise ValueError("n_stiff must lie between 1 and the block dimension")

    n_monomers = blocks.shape[0]
    frequencies = np.zeros((n_monomers, n_stiff), dtype=float)

    for index in range(n_monomers):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues = np.linalg.eigvalsh(block)
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("a retained stiff mode has a non-positive eigenvalue")
        frequencies[index] = np.sqrt(eigenvalues[order])

    return frequencies

def compute_stiff_mode_vectors(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2] or blocks.shape[1] < 1:
        raise ValueError("blocks must have shape (n_monomers, n_dim, n_dim) with n_dim >= 1")
    if blocks.shape[0] < 1:
        raise ValueError("blocks must contain at least one block")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks must contain only finite entries")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)):
        raise ValueError("n_stiff must be an integer")
    n_stiff = int(n_stiff)
    if not 1 <= n_stiff <= blocks.shape[1]:
        raise ValueError("n_stiff must lie between 1 and the block dimension")

    n_monomers, n_dim = blocks.shape[0], blocks.shape[1]
    modes = np.zeros((n_monomers, n_dim, n_stiff), dtype=float)

    for index in range(n_monomers):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("a retained stiff mode has a non-positive eigenvalue")
        selected = eigenvectors[:, order]
        # Fix the arbitrary eigenvector sign so the modes are reproducible.
        dominant = np.argmax(np.abs(selected), axis=0)
        signs = np.sign(selected[dominant, np.arange(n_stiff)])
        signs[signs == 0.0] = 1.0
        modes[index] = selected * signs

    return modes

def compute_cg_free_energy(potential_energy: float, frequencies: np.ndarray,
                                   temperature: float = 250.0,
                                   quantum: bool = False) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(potential_energy, (int, float, np.floating, np.integer))
            and not isinstance(potential_energy, bool) and np.isfinite(potential_energy)):
        raise ValueError("potential_energy must be a finite number")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 2 or frequencies.size < 1:
        raise ValueError("frequencies must have shape (n_monomers, n_stiff)")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and strictly positive")
    if not (isinstance(temperature, (int, float, np.floating, np.integer))
            and not isinstance(temperature, bool) and np.isfinite(temperature)
            and float(temperature) > 0.0):
        raise ValueError("temperature must be a finite number > 0")
    if not isinstance(quantum, (bool, np.bool_)):
        raise ValueError("quantum must be a boolean")

    temperature = float(temperature)

    # Physical constants (CODATA 2018) and the unit conversions implied by
    # energies in kcal/mol, lengths in angstrom and masses in u.
    avogadro = 6.02214076e23
    kcal_per_mole_in_joule = 4184.0 / avogadro
    atomic_mass_unit = 1.66053906660e-27
    angstrom = 1.0e-10
    hbar = 1.054571817e-34
    boltzmann = 1.380649e-23

    # A frequency of one in working units, expressed in radians per second.
    frequency_unit = np.sqrt(kcal_per_mole_in_joule / (angstrom ** 2 * atomic_mass_unit))
    thermal_energy_joule = boltzmann * temperature
    thermal_energy_kcal = thermal_energy_joule / kcal_per_mole_in_joule

    angular = frequencies.ravel() * frequency_unit
    reduced = hbar * angular / thermal_energy_joule

    if bool(quantum):
        # Zero-point energy plus the free energy of the thermal occupation.
        zero_point = float(np.sum(0.5 * hbar * angular)) / kcal_per_mole_in_joule
        occupation = thermal_energy_kcal * float(np.sum(np.log1p(-np.exp(-reduced))))
        harmonic = zero_point + occupation
    else:
        harmonic = thermal_energy_kcal * float(np.sum(np.log(reduced)))

    return float(potential_energy) + harmonic

def backmap_thermal_configuration(coords: np.ndarray, frequencies: np.ndarray,
                                          modes: np.ndarray, atom_masses: np.ndarray,
                                          temperature: float = 250.0,
                                          seed: int = 0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    modes = np.asarray(modes, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    n_monomers = coords.shape[0] // 3
    if frequencies.ndim != 2 or frequencies.shape[0] != n_monomers or frequencies.shape[1] < 1:
        raise ValueError("frequencies must have shape (n_monomers, n_stiff)")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and strictly positive")
    n_stiff = frequencies.shape[1]
    if modes.shape != (n_monomers, 9, n_stiff):
        raise ValueError("modes must have shape (n_monomers, 9, n_stiff)")
    if not np.all(np.isfinite(modes)):
        raise ValueError("modes must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not (isinstance(temperature, (int, float, np.floating, np.integer))
            and not isinstance(temperature, bool) and np.isfinite(temperature)
            and float(temperature) > 0.0):
        raise ValueError("temperature must be a finite number > 0")
    if not (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and int(seed) >= 0):
        raise ValueError("seed must be a non-negative integer")

    temperature = float(temperature)
    boltzmann_kcal = 0.0019872042586408316  # kcal/(mol K), from CODATA 2018
    thermal_energy = boltzmann_kcal * temperature

    generator = np.random.default_rng(int(seed))
    sampled_coords = coords.copy()

    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        # Classical equipartition variance of each stiff mode.
        deviations = np.sqrt(thermal_energy) / frequencies[mono]
        amplitudes = generator.normal(size=n_stiff) * deviations
        scaled_displacement = modes[mono] @ amplitudes
        inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
        sampled_coords[rows] += (inverse_root * scaled_displacement).reshape(3, 3)

    return sampled_coords

def run_shr_pipeline(coords: np.ndarray = None, temperature: float = 250.0,
                             n_iterations: int = 2, n_stiff: int = 3,
                             mass_o: float = 15.9994, mass_h: float = 1.00794,
                             hessian_step: float = 1.0e-5) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-11. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the
    #    public function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
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
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    monomer_reference_geometry = _resolve_step(
        "compute_monomer_reference_geometry", "*compute_monomer_reference_geometry*.py")
    potential_energy = _resolve_step(
        "compute_potential_energy", "*compute_potential_energy*.py")
    potential_gradient = _resolve_step(
        "compute_potential_gradient", "*compute_potential_gradient*.py")
    reference_configuration = _resolve_step(
        "build_reference_configuration", "*build_reference_configuration*.py")
    block_hessian = _resolve_step(
        "compute_block_hessian", "*compute_block_hessian*.py")
    subspace_pseudoinverse = _resolve_step(
        "build_subspace_pseudoinverse", "*build_subspace_pseudoinverse*.py")
    relax_newton_raphson = _resolve_step(
        "relax_subspace_newton_raphson", "*relax_subspace_newton_raphson*.py")
    stiff_mode_frequencies = _resolve_step(
        "compute_stiff_mode_frequencies", "*compute_stiff_mode_frequencies*.py")
    stiff_mode_vectors = _resolve_step(
        "compute_stiff_mode_vectors", "*compute_stiff_mode_vectors*.py")
    cg_free_energy = _resolve_step(
        "compute_cg_free_energy", "*compute_cg_free_energy*.py")
    backmap_configuration = _resolve_step(
        "backmap_thermal_configuration", "*backmap_thermal_configuration*.py")

    # -- The benchmark cyclic tetramer of the problem statement.
    if coords is None:
        coords = np.array([
            [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
            [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
            [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
            [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
            [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
            [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]],
            dtype=float)
    coords = np.asarray(coords, dtype=float)

    # -- Validate the orchestrator inputs.
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    for name, value in (("temperature", temperature), ("mass_o", mass_o),
                        ("mass_h", mass_h), ("hessian_step", hessian_step)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_iterations, (int, np.integer)) and not isinstance(n_iterations, bool)
            and int(n_iterations) >= 0):
        raise ValueError("n_iterations must be an integer >= 0")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)
            and 1 <= int(n_stiff) <= 9):
        raise ValueError("n_stiff must be an integer between 1 and 9")

    temperature = float(temperature)
    n_iterations, n_stiff = int(n_iterations), int(n_stiff)
    hessian_step = float(hessian_step)
    n_monomers = coords.shape[0] // 3
    atom_masses = np.tile(np.array([float(mass_o), float(mass_h), float(mass_h)]), n_monomers)

    # -- Sub-problems 01 and 04: the zeroth-order reference configuration.
    reference_geometry = monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, 107.4)
    frozen_coords = reference_configuration(coords, reference_geometry)

    # -- Sub-problem 03: the one analytic gradient that both the curvature and
    #    the relaxation are built on. Passing it explicitly is what keeps the
    #    Newton step and the Hessian it is preconditioned by on one surface.
    def gradient_fn(positions):
        return potential_gradient(positions)

    # -- Sub-problems 05 and 06: curvature at the frozen point and the operator
    #    that projects a gradient onto the stiff subspace.
    frozen_blocks = block_hessian(frozen_coords, atom_masses, gradient_fn, hessian_step)
    pseudoinverse = subspace_pseudoinverse(frozen_blocks, n_stiff)

    # -- Sub-problem 07: relaxation with the pseudoinverse held fixed.
    relaxed_coords = relax_newton_raphson(frozen_coords, pseudoinverse, atom_masses,
                                          gradient_fn, n_iterations)

    # -- Sub-problems 05, 08 and 09: the two sets of stiff frequencies. The block
    #    Hessian is genuinely rebuilt at the relaxed point.
    relaxed_blocks = block_hessian(relaxed_coords, atom_masses, gradient_fn, hessian_step)
    frozen_frequencies = stiff_mode_frequencies(frozen_blocks, n_stiff)
    relaxed_frequencies = stiff_mode_frequencies(relaxed_blocks, n_stiff)
    relaxed_modes = stiff_mode_vectors(relaxed_blocks, n_stiff)

    # -- Sub-problems 02 and 10: four coarse-grained free energies, the two
    #    treatments evaluated in each of the two vibrational regimes.
    frozen_energy = potential_energy(frozen_coords)
    relaxed_energy = potential_energy(relaxed_coords)
    frozen_classical = cg_free_energy(frozen_energy, frozen_frequencies, temperature, False)
    relaxed_classical = cg_free_energy(relaxed_energy, relaxed_frequencies, temperature, False)
    frozen_quantum = cg_free_energy(frozen_energy, frozen_frequencies, temperature, True)
    relaxed_quantum = cg_free_energy(relaxed_energy, relaxed_frequencies, temperature, True)

    # -- Sub-problem 11: the all-atom reconstruction. Its return value cannot
    #    enter a free-energy ratio, which must not depend on a random draw, so
    #    it is consumed as a check instead. Only the stiff directions are
    #    displaced, so mass-weighting the displacement of each monomer must
    #    return a vector lying wholly in the span of that monomer's stiff mode
    #    vectors; a reconstruction that forgets the inverse-square-root mass
    #    factor, or that displaces along the rigid-body directions too, leaves a
    #    residual outside that span.
    reconstructed = backmap_configuration(relaxed_coords, relaxed_frequencies, relaxed_modes,
                                          atom_masses, temperature, 0)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if reconstructed.shape != relaxed_coords.shape or not np.all(np.isfinite(reconstructed)):
        raise ValueError("the all-atom reconstruction must have the shape of the cluster")
    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        root = np.sqrt(np.repeat(atom_masses[rows], 3))
        scaled = root * (reconstructed[rows] - relaxed_coords[rows]).reshape(9)
        basis = relaxed_modes[mono]
        residual = scaled - basis @ (basis.T @ scaled)
        if np.linalg.norm(residual) > 1.0e-8 * max(np.linalg.norm(scaled), 1.0):
            raise ValueError("the reconstruction displaced outside the stiff subspace")
        # At a finite temperature the draw is non-degenerate, so a reconstruction
        # that returns the relaxed configuration untouched has not sampled at all.
        if np.linalg.norm(scaled) <= 0.0:
            raise ValueError("the reconstruction left the configuration unchanged")

    # -- The two intensive penalties and the reported amplification factor.
    boltzmann_kcal = 0.0019872042586408316  # kcal/(mol K), from CODATA 2018
    scale = n_monomers * n_stiff * boltzmann_kcal * temperature
    penalty_classical = (frozen_classical - relaxed_classical) / scale
    penalty_quantum = (frozen_quantum - relaxed_quantum) / scale
    if penalty_classical == 0.0:
        raise ValueError("the classical rigidification penalty vanishes; the ratio is undefined")

    return float(penalty_quantum / penalty_classical)
SCICODE_GOLD_EOF
