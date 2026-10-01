"""
Return major-allele and normal/risk-allele recodings as a (4, p) array.

Major-allele coding flips SNPs with eaf < 0.5. Normal/risk-allele coding flips SNPs with bx < 0. Import numpy inside the function; np is not predefined.

Returns
-------
Array of shape (4, p): g_major, G_major, g_normal, G_normal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def allele_orientations(bx: np.ndarray, by: np.ndarray, eaf: np.ndarray) -> np.ndarray:
    """Major-allele and normal/risk-allele recodings of SNP associations.

    Parameters
    ----------
    bx, by, eaf : np.ndarray
        Length-p exposure effects, outcome effects, and effect-allele frequencies.

    Returns
    -------
    np.ndarray
        Array of shape (4, p) with rows (g_major, G_major, g_normal, G_normal).
    """
    p = np.asarray(bx).reshape(-1).size
    return np.zeros((4, p), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_allele_orientations(bx, by, eaf):
    bx = bx.reshape(-1)
    by = by.reshape(-1)
    eaf = eaf.reshape(-1)
    sm = 1.0 - 2.0 * (eaf < 0.5)
    sn = 1.0 - 2.0 * (bx < 0.0)
    g_m = bx * sm
    G_m = by * sm
    g_n = bx * sn
    G_n = by * sn
    out = type(bx)((4, bx.shape[0]))
    out[0] = g_m
    out[1] = G_m
    out[2] = g_n
    out[3] = G_n
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nbx = np.array([0.2, -0.1])\nby = np.array([0.1, 0.05])\neaf = np.array([0.6, 0.4])",
            "call": "allele_orientations(bx, by, eaf)",
            "gold_call": "_oracle_allele_orientations(bx, by, eaf)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([-0.3])\nby = np.array([-0.06])\neaf = np.array([0.2])",
            "call": "allele_orientations(bx, by, eaf)",
            "gold_call": "_oracle_allele_orientations(bx, by, eaf)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.08, -0.12, 0.16, 0.20, -0.24, 0.28, 0.32, -0.36, 0.40, -0.44, 0.48, -0.52, 0.012, -0.010])\nby = np.array([0.0252, -0.0368, 0.0486, 0.0590, -0.0716, 0.0835, 0.0969, -0.1083, 0.6720, -0.7335, 0.6450, -0.7372, 0.1636, -0.1150])\neaf = np.array([0.62, 0.31, 0.71, 0.44, 0.28, 0.58, 0.67, 0.39, 0.55, 0.41, 0.73, 0.36, 0.52, 0.48])",
            "call": "allele_orientations(bx, by, eaf)",
            "gold_call": "_oracle_allele_orientations(bx, by, eaf)",
        },
        {
            "setup": "import numpy as np\nbx = np.array([0.1, 0.2, 0.3])\nby = np.array([0.02, 0.04, 0.06])\neaf = np.array([0.7, 0.6, 0.55])",
            "call": "allele_orientations(bx, by, eaf)",
            "gold_call": "_oracle_allele_orientations(bx, by, eaf)",
        },
    ]
