"""
Return, for every backbone bond of a chain that is either tethered at one end or free, the frequency prefactor of its transition-state rupture rate when bond lengths are measured in units of the equilibrium bond length.

A transition-state rate is the thermal one-sided flux through a dividing surface in the bond-length coordinate, divided by the reactant population. The flux prefactor depends only on how the momenta of the atoms bounding a bond feed its length velocity, so it depends on which atoms are free to move.

Returns
-------
np.ndarray: float array of shape (n_bonds,) with the per-bond kinetic frequency prefactors in s^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_kinetic_prefactors(
    n_bonds: int,
    atom_mass_kg: float,
    bond_length_m: float,
    temperature_k: float,
    tethered: bool,
) -> "np.ndarray":
    """Return the kinetic frequency prefactor of every bond of a chain.

    The chain has atoms 0 to ``n_bonds`` of identical mass ``atom_mass_kg``,
    and bond ``i`` (1-based) joins atoms ``i - 1`` and ``i``. If ``tethered``
    is True, atom 0 is held fixed and atoms 1 to ``n_bonds`` carry
    Maxwell-Boltzmann momenta at ``temperature_k``; otherwise every atom
    carries them. For each bond, return the equilibrium average of the
    positive part of the rate of change of its length, divided by
    ``bond_length_m`` so that the value is a frequency in s^-1. Use
    k_B = 1.380649e-23 J/K.

    Parameters
    ----------
    n_bonds : int
        Number of backbone bonds, at least 1.
    atom_mass_kg : float
        Mass of every atom in kg.
    bond_length_m : float
        Equilibrium bond length in m, the unit of the bond-length coordinate.
    temperature_k : float
        Temperature in K.
    tethered : bool
        Whether atom 0 is held fixed.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds,)`` in s^-1; index 0 is the bond
        containing atom 0.

    Raises
    ------
    ValueError
        If ``n_bonds`` is not an integer of at least 1 (booleans are
        rejected), if any of ``atom_mass_kg``, ``bond_length_m`` and
        ``temperature_k`` is not a finite positive real number, or if
        ``tethered`` is not a bool.
    """
    return prefactors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_bond_kinetic_prefactors(
    n_bonds: int,
    atom_mass_kg: float,
    bond_length_m: float,
    temperature_k: float,
    tethered: bool,
) -> "np.ndarray":
    """Reference implementation from the effective bond-length masses."""
    import numpy as np

    if isinstance(n_bonds, bool) or not isinstance(n_bonds, (int, np.integer)) or n_bonds < 1:
        raise ValueError("n_bonds must be an integer of at least 1")
    for value in (atom_mass_kg, bond_length_m, temperature_k):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("physical parameters must be real numbers")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("physical parameters must be finite and positive")
    if not isinstance(tethered, (bool, np.bool_)):
        raise ValueError("tethered must be a bool")
    kbt = 1.380649e-23 * float(temperature_k)
    # A bond between two mobile atoms is their relative coordinate, with half the atomic
    # mass; a bond attached to the fixed atom moves with that of one atom.
    effective_mass = np.full(int(n_bonds), 0.5 * float(atom_mass_kg))
    if tethered:
        effective_mass[0] = float(atom_mass_kg)
    mean_positive_velocity = np.sqrt(kbt / (2.0 * np.pi * effective_mass))
    return mean_positive_velocity / float(bond_length_m)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(values):\n"
        "    arr = np.asarray(values, dtype=float)\n"
        "    flat = arr.ravel() / 1.0e12\n"
        "    weight = 1.0 + 0.5 * np.cos(0.7 * np.arange(flat.size))\n"
        "    return float(arr.ndim + flat.size + np.sum(weight * flat))\n"
    )
    raises = (
        "import numpy as np\n"
        "def _candidate():\n    try:\n        compute_bond_kinetic_prefactors({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
        "def _reference():\n    try:\n        _oracle_compute_bond_kinetic_prefactors({args})\n        return 0\n"
        "    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(compute_bond_kinetic_prefactors(10, 1.99e-26, 1.525e-10, 293.15, True))",
            "gold_call": "_sig(_oracle_compute_bond_kinetic_prefactors(10, 1.99e-26, 1.525e-10, 293.15, True))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_bond_kinetic_prefactors(10, 1.99e-26, 1.525e-10, 293.15, False))",
            "gold_call": "_sig(_oracle_compute_bond_kinetic_prefactors(10, 1.99e-26, 1.525e-10, 293.15, False))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_bond_kinetic_prefactors(1, 2.66e-26, 1.43e-10, 350.0, True))",
            "gold_call": "_sig(_oracle_compute_bond_kinetic_prefactors(1, 2.66e-26, 1.43e-10, 350.0, True))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_bond_kinetic_prefactors(1, 2.66e-26, 1.43e-10, 350.0, False))",
            "gold_call": "_sig(_oracle_compute_bond_kinetic_prefactors(1, 2.66e-26, 1.43e-10, 350.0, False))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_bond_kinetic_prefactors(3, 1.99e-26, 1.525e-10, 77.0, False))",
            "gold_call": "_sig(_oracle_compute_bond_kinetic_prefactors(3, 1.99e-26, 1.525e-10, 77.0, False))",
        },
        {
            "setup": raises.replace("{args}", "0, 1.99e-26, 1.525e-10, 293.15, True"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "True, 1.99e-26, 1.525e-10, 293.15, False"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "4, -1.99e-26, 1.525e-10, 293.15, True"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
        {
            "setup": raises.replace("{args}", "4, 1.99e-26, 1.525e-10, 293.15, 'yes'"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        },
    ]
