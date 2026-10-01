"""
This step solves the three-by-three interface system for the shell coefficients of the exact radial field, and evaluates the fourth, unused, interface condition as a residual. The coefficients are the reference solution against which the discrete strain field is measured at the end of the chain, so they must be obtained from the interface conditions themselves rather than from any approximation of them.

The residual is reported alongside them. It is not an error estimate of this step: it is a consistency test on the pair of steps that precede the mesh, because it can vanish only when the inclusion radius handed in really does make the coated sphere neutral.

Under the macroscopic strain $\overline{\epsilon} = I$ the displacement of a spherically symmetric three-phase body is radial and takes the form $u_r = a r + b / r^2$ in every shell, where the second term is dropped in the inclusion because the field must stay bounded at the origin. Neutrality means the matrix is undisturbed, so there $u_r = r$ exactly. Writing $A$ and $B$ for the coating constants and $C$ for the inclusion constant, the field is $u_r = r$ for $r \ge r_c$, $u_r = A r + B / r^2$ for $r_i \le r < r_c$ and $u_r = C r$ for $r < r_i$.

For a shell with bulk modulus $K$ and shear modulus $\mu$ the volumetric strain is $3a$ and the radial traction is $\sigma_{rr} = 3 K a - 4 \mu b / r^3$. Continuity of $u_r$ and of $\sigma_{rr}$ at $r = r_c$ and continuity of $u_r$ at $r = r_i$ give three linear equations for the triple $(A, B, C)$, namely $A r_c + B / r_c^2 = r_c$, $3 K_c A - 4 \mu_c B / r_c^3 = 3 K_m$ and $C r_i = A r_i + B / r_i^2$.

Continuity of $\sigma_{rr}$ at $r = r_i$ is a fourth equation that the three unknowns cannot satisfy in general. It is satisfied precisely when the coated sphere is neutral, so its residual $3 K_i C - [3 K_c A - 4 \mu_c B / r_i^3]$ is the verification that the geometry and the moduli are mutually consistent.

Returns
-------
dict, the three shell coefficients with the residual of the unused traction condition.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_coated_sphere_exact_field(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Solve the three-shell interface system for the exact radial field.

    Parameters
    ----------
    bulk_matrix : float
        Bulk modulus of the matrix.
    bulk_coating : float
        Bulk modulus of the coating.
    bulk_inclusion : float
        Bulk modulus of the inclusion.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_inclusion : float
        Inner interface radius, strictly positive and smaller than radius_coating.
    radius_coating : float
        Outer interface radius.

    Returns
    -------
    dict
        Keys coefficient_a, coefficient_b, coefficient_c and traction_residual.

    Raises
    ------
    ValueError
        If the radii do not satisfy 0 < radius_inclusion < radius_coating, or if poisson_ratio lies outside the open interval (-1, 1/2).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_coated_sphere_exact_field(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    bulk_matrix = float(bulk_matrix)
    bulk_coating = float(bulk_coating)
    bulk_inclusion = float(bulk_inclusion)
    poisson_ratio = float(poisson_ratio)
    radius_inclusion = float(radius_inclusion)
    radius_coating = float(radius_coating)
    if not 0.0 < radius_inclusion < radius_coating:
        raise ValueError("the radii must satisfy 0 < radius_inclusion < radius_coating")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in the open interval (-1, 1/2)")
    shear_coating = 3.0 * bulk_coating * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))
    system = np.array([
        [radius_coating, 1.0 / radius_coating ** 2, 0.0],
        [3.0 * bulk_coating, -4.0 * shear_coating / radius_coating ** 3, 0.0],
        [radius_inclusion, 1.0 / radius_inclusion ** 2, -radius_inclusion],
    ])
    right = np.array([radius_coating, 3.0 * bulk_matrix, 0.0])
    coefficient_a, coefficient_b, coefficient_c = np.linalg.solve(system, right)
    residual = (3.0 * bulk_inclusion * coefficient_c
                - (3.0 * bulk_coating * coefficient_a
                   - 4.0 * shear_coating * coefficient_b / radius_inclusion ** 3))
    return {
        "coefficient_a": float(coefficient_a),
        "coefficient_b": float(coefficient_b),
        "coefficient_c": float(coefficient_c),
        "traction_residual": float(residual),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
RI = 2*np.pi * 0.1399225665006822 ** (1.0/3.0)
def summarize(data):
    return (round(data["coefficient_a"], 12),
            round(data["coefficient_b"], 10),
            round(data["coefficient_c"], 12),
            round(abs(data["traction_residual"]), 12))
""",
            "call": "summarize(solve_coated_sphere_exact_field(1.0, 0.808024, 8.080240, 0.25, RI, 2*np.pi))",
            "gold_call": "summarize(_oracle_solve_coated_sphere_exact_field(1.0, 0.808024, 8.080240, 0.25, RI, 2*np.pi))",
        },
        {
            "setup": """import numpy as np
def homogeneous(fn):
    data = fn(1.7, 1.7, 1.7, 0.25, 0.4, 1.0)
    return (round(data["coefficient_a"], 13), round(data["coefficient_b"], 13),
            round(data["coefficient_c"], 13), round(abs(data["traction_residual"]), 13))
""",
            "call": "homogeneous(solve_coated_sphere_exact_field)",
            "gold_call": "homogeneous(_oracle_solve_coated_sphere_exact_field)",
        },
        {
            "setup": """def run(fn):
    codes = []
    for args in [(1.0, 0.8, 8.0, 0.25, 2.0, 1.0), (1.0, 0.8, 8.0, 0.7, 0.4, 1.0)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(solve_coated_sphere_exact_field)",
            "gold_call": "run(_oracle_solve_coated_sphere_exact_field)",
        },
    ]
