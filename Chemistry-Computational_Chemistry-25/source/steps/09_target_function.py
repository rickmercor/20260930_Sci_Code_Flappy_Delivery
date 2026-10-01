"""
Score a set of predicted energies against the reference curve. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, e_pips the predicted total energies and e_ref the reference total energies, both in hartree and in the same order, and e_atom maps an element symbol to its reference atomic energy. For each distinct pair in species, the reference shift is the sum of the two atomic energies of that pair, and the predicted shift is that same sum plus the mean, over just the geometries of that pair, of the predicted energy minus the reference energy. The residual at a geometry is the predicted energy minus its predicted shift, minus the quantity the reference energy minus its reference shift. Return the square root of the mean of the squared residuals taken over every geometry of every pair, multiplied by 627.5094740631 to convert from hartree to kcal/mol.

The predicted energies do not come from a variational solution of the problem they are scored against, so nothing ties their zero of energy to the reference's. A workspace matrix is not a Fock matrix, the occupied block is chosen by assumption, and no self-consistent cycle runs, so a candidate can follow the shape of a curve closely while sitting at an absurd absolute energy. Note that a transformation which moved every eigenvalue of the workspace matrix by the same amount would not do this: it leaves the eigenvectors, and therefore the density and the energy, untouched. The offset is a property of the surrogate, not of the spectrum. Allowing one free constant per molecule removes that freedom from the score. Writing the constant as a sum over the elements present, rather than fitting each molecule independently, is what lets the constants transfer to a molecule that was never trained on; it also makes the fit rank deficient once there are more elements than molecules, so the individual element parameters are not determined even though every molecular shift, and the score itself, is.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def target_function(species: "list", e_pips: "list", e_ref: "list", e_atom: "dict") -> "float":
    """Score a set of predicted energies against the reference curve. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, e_pips the predicted total energies and e_ref the reference total energies, both in hartree and in the same order, and e_atom maps an element symbol to its reference atomic energy. For each distinct pair in species, the reference shift is the sum of the two atomic energies of that pair, and the predicted shift is that same sum plus the mean, over just the geometries of that pair, of the predicted energy minus the reference energy. The residual at a geometry is the predicted energy minus its predicted shift, minus the quantity the reference energy minus its reference shift. Return the square root of the mean of the squared residuals taken over every geometry of every pair, multiplied by 627.5094740631 to convert from hartree to kcal/mol.

    Parameters
    ----------
    species : list of tuple of str
        One pair of element symbols per geometry.
    e_pips : list of float
        The predicted total energy at each geometry, in hartree.
    e_ref : list of float
        The reference total energy at each geometry, in hartree. Same length as e_pips.
    e_atom : dict
        Maps an element symbol to a float, used only to size the shared fit.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if species, e_pips and e_ref do not share one non-zero length, if any energy is not finite, or if e_atom has no entry for an element that appears.
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


def _oracle_target_function(species: "list", e_pips: "list", e_ref: "list", e_atom: "dict") -> "float":
    e_pips = np.asarray(e_pips, dtype=float)
    e_ref = np.asarray(e_ref, dtype=float)
    species = [tuple(s) for s in species]
    if len(species) != e_pips.size or e_pips.size != e_ref.size or e_pips.size == 0:
        raise ValueError("species, e_pips and e_ref must have the same non-zero length")
    if not np.all(np.isfinite(e_pips)) or not np.all(np.isfinite(e_ref)):
        raise ValueError("non-finite energies")
    res = np.empty(e_pips.size)
    for s in dict.fromkeys(species):
        for a in s:
            if a not in e_atom:
                raise ValueError(f"no atomic energy for {a}")
        idx = [i for i, x in enumerate(species) if x == s]
        Di = float(e_atom[s[0]] + e_atom[s[1]])
        Dbar = float(np.mean(e_pips[idx] - e_ref[idx])) + Di
        for i in idx:
            res[i] = (e_pips[i] - Dbar) - (e_ref[i] - Di)
    return float(np.sqrt(np.mean(res*res)))*627.5094740631

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
      {"setup": "import numpy as np\nSP=[('Li','F'),('Li','F'),('Li','Cl'),('Li','Cl')]\nRR=[1.4,2.0,1.9,2.6]\nEREF=[-105.3730526572,-105.2953633858,-461.9920679602,-461.9387468321]\nAT={'Li':-7.315525754,'Na':-159.6682151,'Cl':-454.5421021,'F':-97.98651118}\nCH={'Li':3.,'Na':11.,'Cl':17.,'F':9.}\nfrom copy import deepcopy as _dc\nSP_G=_dc(SP)\nEREF_G=_dc(EREF)\nAT_G=_dc(AT)",
       "call": "target_function(SP,[-105.0,-104.9,-461.5,-461.4],EREF,AT)",
       "gold_call": "_oracle_target_function(SP_G,[-105.0,-104.9,-461.5,-461.4],EREF_G,AT_G)"},  # two molecules, four geometries
      {"setup": "import numpy as np\nSP=[('Li','F'),('Li','F'),('Li','Cl'),('Li','Cl')]\nRR=[1.4,2.0,1.9,2.6]\nEREF=[-105.3730526572,-105.2953633858,-461.9920679602,-461.9387468321]\nAT={'Li':-7.315525754,'Na':-159.6682151,'Cl':-454.5421021,'F':-97.98651118}\nCH={'Li':3.,'Na':11.,'Cl':17.,'F':9.}\nfrom copy import deepcopy as _dc\nSP_G=_dc(SP)\nEREF_G=_dc(EREF)\nAT_G=_dc(AT)",
       "call": "target_function(SP[:2],[-105.1,-104.8],EREF[:2],AT)",
       "gold_call": "_oracle_target_function(SP_G[:2],[-105.1,-104.8],EREF_G[:2],AT_G)"},  # one molecule only
      {"setup": "import numpy as np\nSP=[('Li','F'),('Li','F'),('Li','Cl'),('Li','Cl')]\nRR=[1.4,2.0,1.9,2.6]\nEREF=[-105.3730526572,-105.2953633858,-461.9920679602,-461.9387468321]\nAT={'Li':-7.315525754,'Na':-159.6682151,'Cl':-454.5421021,'F':-97.98651118}\nCH={'Li':3.,'Na':11.,'Cl':17.,'F':9.}\nfrom copy import deepcopy as _dc\nSP_G=_dc(SP)\nEREF_G=_dc(EREF)\nAT_G=_dc(AT)",
       "call": "target_function(SP,[-105.30,-105.33,-461.90,-461.99],EREF,AT)",
       "gold_call": "_oracle_target_function(SP_G,[-105.30,-105.33,-461.90,-461.99],EREF_G,AT_G)"},  # a curve that is wrong by a different amount at every geometry
        {"setup": "import numpy as np\nSP=[('Li','F'),('Li','F'),('Li','Cl'),('Li','Cl')]\nRR=[1.4,2.0,1.9,2.6]\nEREF=[-105.3730526572,-105.2953633858,-461.9920679602,-461.9387468321]\nAT={'Li':-7.315525754,'Na':-159.6682151,'Cl':-454.5421021,'F':-97.98651118}\nCH={'Li':3.,'Na':11.,'Cl':17.,'F':9.}\nfrom copy import deepcopy as _dc\nSP_G=_dc(SP)\nEREF_G=_dc(EREF)\nAT_G=_dc(AT)\ndef _c():\n    try:\n        target_function(SP,[-105.0,-104.9],EREF,AT)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_target_function(SP_G,[-105.0,-104.9],EREF_G,AT_G)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
