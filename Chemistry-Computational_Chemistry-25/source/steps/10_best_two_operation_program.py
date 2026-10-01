"""
Search every two-operation program and report the best one. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, bond_lengths the corresponding separations in angstrom, and e_ref the reference total energies in hartree; e_atom and basis_data are as in the earlier steps and charges maps an element symbol to its nuclear charge. At each geometry the first atom sits at the origin and the second on the positive z axis, with the separation converted to bohr by dividing by 0.52917721092; the number of occupied orbitals is half the sum of the two nuclear charges, and the nuclear repulsion is the product of the two charges divided by the separation in bohr. The total predicted energy at a geometry is the electronic energy of the workspace matrix plus that nuclear repulsion. The candidates are every ordered pair of operation ids whose first entry is one of the twenty two matrix products, that is ids 60 to 70 and 115 to 125, and whose second entry is any id from 0 to 130. A candidate is discarded if at any geometry an operation fails, the eigenvalue problem cannot be solved, or the energy is not finite. Among the candidates that survive at every geometry, choose the one with the smallest score, breaking a tie by the smaller first id and then by the smaller second id. Return that smallest score in kcal/mol.

Two operations is a small enough space to enumerate exactly, which matters because a stochastic search would make the answer depend on a random seed rather than on the physics. Restricting the opening move to a matrix product allows nonsymmetric workspace matrices, so the choice of symmetrisation can affect the result; it does not guarantee that every opening or final workspace is nonsymmetric. The failures are as informative as the successes, since a large part of the catalogue asks for the inverse of a matrix that is singular at some geometry, and a candidate that works at equilibrium but not at a stretched bond is no use.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def best_two_operation_program(species: "list", bond_lengths: "list", e_ref: "list", e_atom: "dict",
                               basis_data: "dict", charges: "dict") -> "float":
    """Search every two-operation program and report the best one. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, bond_lengths the corresponding separations in angstrom, and e_ref the reference total energies in hartree; e_atom and basis_data are as in the earlier steps and charges maps an element symbol to its nuclear charge. At each geometry the first atom sits at the origin and the second on the positive z axis, with the separation converted to bohr by dividing by 0.52917721092; the number of occupied orbitals is half the sum of the two nuclear charges, and the nuclear repulsion is the product of the two charges divided by the separation in bohr. The total predicted energy at a geometry is the electronic energy of the workspace matrix plus that nuclear repulsion. The candidates are every ordered pair of operation ids whose first entry is one of the twenty two matrix products, that is ids 60 to 70 and 115 to 125, and whose second entry is any id from 0 to 130. A candidate is discarded if at any geometry an operation fails, the eigenvalue problem cannot be solved, or the energy is not finite. Among the candidates that survive at every geometry, choose the one with the smallest score, breaking a tie by the smaller first id and then by the smaller second id. Return that smallest score in kcal/mol.

    Parameters
    ----------
    species : list of tuple of str
        One pair of element symbols per geometry.
    bond_lengths : list of float
        The separation at each geometry, in angstrom. Same length as species.
    e_ref : list of float
        The reference total energy at each geometry, in hartree. Same length as species.
    e_atom : dict
        Maps an element symbol to a float, used only to size the shared fit.
    basis_data : dict
        Maps an element symbol to its list of shells, as in the first step.
    charges : dict
        Maps an element symbol to its nuclear charge as a float.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if species, bond_lengths and e_ref do not share one non-zero length, if any bond length is not positive and finite, if a molecule does not have an even number of electrons, or if no candidate program survives at every geometry.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _oracle_best_two_operation_program(species: "list", bond_lengths: "list", e_ref: "list", e_atom: "dict",
                                       basis_data: "dict", charges: "dict") -> "float":
    species = [tuple(s) for s in species]
    bond_lengths = np.asarray(bond_lengths, dtype=float)
    e_ref = np.asarray(e_ref, dtype=float)
    if not (len(species) == bond_lengths.size == e_ref.size) or len(species) == 0:
        raise ValueError("species, bond_lengths and e_ref must have the same non-zero length")
    if not np.all(np.isfinite(bond_lengths)) or np.any(bond_lengths <= 0.0):
        raise ValueError("bond lengths must be positive and finite")

    geoms = []
    for (a, b), R in zip(species, bond_lengths):
        bohr = 1.0/0.52917721092
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(R)*bohr]])
        Z = [float(charges[a]), float(charges[b])]
        table = _oracle_basis_table([a, b], basis_data)
        core = _oracle_core_integrals(table, coords, Z)
        eri = _oracle_electron_repulsion(table, coords)
        desc = _oracle_descriptor_matrices(table, coords, eri)
        operands = _oracle_operand_table(core, desc)
        n_elec = int(round(Z[0] + Z[1]))
        if n_elec % 2:
            raise ValueError("the molecule does not have an even number of electrons")
        geoms.append(dict(S=core[0], hcore=core[1] + core[2], eri=eri,
                          enuc=Z[0]*Z[1]/(float(R)*bohr),
                          operands=operands, n_occ=n_elec//2))

    matmul_ids = list(range(60, 71)) + list(range(115, 126))
    best = None
    for op1 in matmul_ids:
        try:
            for g in geoms:
                _oracle_apply_operation(g["hcore"], op1, g["operands"])
        except ValueError:
            continue                  # this opening already fails: no need to try any second op
        for op2 in range(131):
            e_pips, ok = [], True
            for g in geoms:
                try:
                    M = _oracle_workspace_matrix(g["hcore"], (op1, op2), g["operands"])
                    e_pips.append(_oracle_program_energy(M, g["S"], g["hcore"], g["eri"],
                                                         g["n_occ"]) + g["enuc"])
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            try:
                F = _oracle_target_function(species, e_pips, e_ref, e_atom)
            except ValueError:
                continue
            if np.isfinite(F) and (best is None or (F, op1, op2) < best):
                best = (F, op1, op2)
    if best is None:
        raise ValueError("no candidate program produced a finite target function")
    return float(best[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base_0 = """import numpy as np
TSP=[('X','Y')]*4
TRR=[1.7,2.1,2.6,3.2]
TREF=[-6.21,-6.44,-6.30,-6.05]
USP=[('X','Y')]*3
URR=[2.0,2.5,3.1]
UREF=[-6.38,-6.33,-6.08]
VSP=[('X','Y')]*4
VRR=[1.6,2.2,2.9,3.5]
VREF=[-6.15,-6.45,-6.18,-5.98]
TAT={'X':-1.4,'Y':-3.9}
TCH={'X':2.,'Y':4.}
TOY={'X':[(0,[1.9,0.44,0.14],[0.16,0.53,0.45])],'Y':[(0,[4.1,0.95,0.31],[0.15,0.54,0.44]),(1,[0.83,0.29,0.11],[0.16,0.61,0.39])]}
from copy import deepcopy as _dc
"""
    base_1 = """from copy import deepcopy
SP=[('X', 'Y'), ('X', 'Y'), ('X', 'Y'), ('X', 'Y'), ('X', 'Z'), ('X', 'Z'), ('X', 'Z')]
RR=[1.7, 2.1, 2.6, 3.2, 2.0, 2.5, 3.1]
REF=[-6.21, -6.44, -6.3, -6.05, -6.38, -6.33, -6.08]
AT={'X': -1.4, 'Y': -3.9, 'Z': -3.9}
BD={'X': [(0, [1.9, 0.44, 0.14], [0.16, 0.53, 0.45])], 'Y': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])], 'Z': [(0, [4.1, 0.95, 0.31], [0.15, 0.54, 0.44]), (1, [0.83, 0.29, 0.11], [0.16, 0.61, 0.39])]}
CH={'X': 2.0, 'Y': 4.0, 'Z': 4.0}
SP_G,RR_G,REF_G,AT_G,BD_G,CH_G=deepcopy((SP,RR,REF,AT,BD,CH))"""
    return [
        {
            "setup": base_0 + """TSP_G=_dc(TSP)
TRR_G=_dc(TRR)
TREF_G=_dc(TREF)
TAT_G=_dc(TAT)
TCH_G=_dc(TCH)
TOY_G=_dc(TOY)""",
            'call': 'best_two_operation_program(TSP,TRR,TREF,TAT,TOY,TCH)',
            'gold_call': '_oracle_best_two_operation_program(TSP_G,TRR_G,TREF_G,TAT_G,TOY_G,TCH_G)',
        },
        {
            "setup": base_0 + """USP_G=_dc(USP)
URR_G=_dc(URR)
UREF_G=_dc(UREF)
TAT_G=_dc(TAT)
TCH_G=_dc(TCH)
TOY_G=_dc(TOY)""",
            'call': 'best_two_operation_program(USP,URR,UREF,TAT,TOY,TCH)',
            'gold_call': '_oracle_best_two_operation_program(USP_G,URR_G,UREF_G,TAT_G,TOY_G,TCH_G)',
        },
        {
            "setup": base_0 + """VSP_G=_dc(VSP)
VRR_G=_dc(VRR)
VREF_G=_dc(VREF)
TAT_G=_dc(TAT)
TCH_G=_dc(TCH)
TOY_G=_dc(TOY)""",
            'call': 'best_two_operation_program(VSP,VRR,VREF,TAT,TOY,TCH)',
            'gold_call': '_oracle_best_two_operation_program(VSP_G,VRR_G,VREF_G,TAT_G,TOY_G,TCH_G)',
        },
        {
            "setup": base_0 + """TSP_G=_dc(TSP)
TRR_G=_dc(TRR)
TREF_G=_dc(TREF)
TAT_G=_dc(TAT)
TCH_G=_dc(TCH)
TOY_G=_dc(TOY)
def _c():
    try:
        best_two_operation_program(TSP,TRR[:2],TREF,TAT,TOY,TCH)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def _g():
    try:
        _oracle_best_two_operation_program(TSP_G,TRR_G[:2],TREF_G,TAT_G,TOY_G,TCH_G)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call': '_c()',
            'gold_call': '_g()',
        },
        {
            "setup": base_1,
            'call': 'best_two_operation_program(SP,RR,REF,AT,BD,CH)',
            'gold_call': '_oracle_best_two_operation_program(SP_G,RR_G,REF_G,AT_G,BD_G,CH_G)',
        },
    ]
