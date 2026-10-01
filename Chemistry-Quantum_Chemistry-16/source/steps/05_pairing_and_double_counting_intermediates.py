"""
At the stationary gaps, return the two-body intermediates

that the excitation energies of the valence use many times.

Excitation energies from a perfect-pairing reference

use a small number of integral combinations many times. One family gives the

energy of an open-shell singlet in a given pair of orbitals. A second family

contains the double-counting corrections that occur when an excitation changes

more than one VBS at the same time.

Returns
-------
a tuple of twelve floats     ``(t_01, t_23, t_02, t_03, t_12, t_13, g_dd, g_plain, g_a_at_2,     g_a_at_3, g_b_at_0, g_b_at_1)``
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pairing_intermediates(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    """Return the open-shell-singlet and double-counting intermediates.

    The orbital labels and the integral conventions are the same as in the
    preceding steps. The gaps are the stationary gaps for the given
    integrals. Give the open-shell-singlet intermediate for each orbital pair
    in the return list. Give the double-counting intermediates for the
    subsystem pair (A, B).

    Expected return: a tuple of twelve floats
    ``(t_01, t_23, t_02, t_03, t_12, t_13, g_dd, g_plain, g_a_at_2,
    g_a_at_3, g_b_at_0, g_b_at_1)``:

    - ``t_pq`` is the open-shell-singlet intermediate for orbitals ``p`` and
      ``q``.
    - ``g_dd`` is the double-counting intermediate for (A, B) with a gap
      weight for both subsystems.
    - ``g_plain`` is the double-counting intermediate for (A, B) with no gap
      weight.
    - ``g_a_at_2`` and ``g_a_at_3`` are the intermediates of VBS A with one
      gap weight, at orbitals 2 and 3.
    - ``g_b_at_0`` and ``g_b_at_1`` are the intermediates of VBS B with one
      gap weight, at orbitals 0 and 1.

    Definitions. Use ``J``, ``K``, ``L``, ``G``, ``eta`` and the gaps ``w``
    from the first step, at the stationary gaps. Write ``x_A = w_A / eta_A``
    and ``x_B = w_B / eta_B``.

    - Same VBS (``t_01`` and ``t_23``):
      ``t_pq = J_pq + K_pq - L_pp / 2 - L_qq / 2``.
    - Different VBS (``t_02``, ``t_03``, ``t_12``, ``t_13``):
      ``t_pq = J_pq + K_pq - L_pp / 2 - L_qq / 2 - G_pq / 2``.
    - ``g_dd = (1/2) * x_A * x_B * (G_02 - G_03 - G_12 + G_13)``.
    - ``g_plain = (1/2) * (G_02 + G_03 + G_12 + G_13)``. Each term has one
      orbital of A and one orbital of B. (One printed version of this
      definition has ``G_11`` as the last term. That is a misprint. Use
      ``G_13``.)
    - ``g_a_at_p = (1/2) * x_A * (G_0p - G_1p)`` for ``p`` = 2, 3.
    - ``g_b_at_p = (1/2) * x_B * (G_2p - G_3p)`` for ``p`` = 0, 1.

    Raise ``ValueError`` if ``h`` or ``v`` does not have the four-orbital
    shape.
    """
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pairing_intermediates(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    w_a, w_b, _energy = _oracle_stationary_vbs_gaps(h, v)
    packed = _oracle_pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    gaps = [w_a, w_b]

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]

    t_same = []
    for p, q in ((0, 1), (2, 3)):
        t_same.append(bigj[p][q] + bigk[p][q] - 0.5 * bigl[p][p] - 0.5 * bigl[q][q])
    t_cross = []
    for p, q in ((0, 2), (0, 3), (1, 2), (1, 3)):
        t_cross.append(bigj[p][q] + bigk[p][q] - 0.5 * bigl[p][p] - 0.5 * bigl[q][q]
                       - 0.5 * bigg[p][q])

    g_dd = 0.0
    for mu in (0, 1):
        for nu in (0, 1):
            g_dd += ((-1.0) ** (mu + nu)) * bigg[mu][2 + nu]
    g_dd *= 0.5 * (gaps[0] / eta[0]) * (gaps[1] / eta[1])

    g_plain = 0.5 * (bigg[0][2] + bigg[0][3] + bigg[1][2] + bigg[1][3])
    g_single = []
    for p in (2, 3):
        g_single.append(0.5 * (gaps[0] / eta[0]) * (bigg[0][p] - bigg[1][p]))
    for p in (0, 1):
        g_single.append(0.5 * (gaps[1] / eta[1]) * (bigg[2][p] - bigg[3][p]))
    return (t_same[0], t_same[1], t_cross[0], t_cross[1], t_cross[2], t_cross[3],
            g_dd, g_plain, g_single[0], g_single[1], g_single[2], g_single[3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return the canonical chain and four further chains."""
    return [
        # canonical fixture of the task
        {
            "setup": (
                'h = [[-1.1114074762, 0.0988634904, -0.0394997241, 0.0276626402], [0.0988634904, -1.0016712992, 0.0276223611, -0.0323783080], [-0.0394997241, 0.0276223611, -1.1566175707, 0.0914760332], [0.0276626402, -0.0323783080, 0.0914760332, -1.0021521651]]\n'
                'vu = [0.5086943900, -0.0007761260, 0.0039529102, 0.0240128942, 0.5193065477, -0.0190798421, -0.0107169118, 0.1659122216, -0.0448520232, 0.1705764863, 0.2588967397, -0.0146179220, -0.0027594462, -0.0016634844, -0.0005507431, 0.0173057438, -0.0497530130, 0.0227179415, -0.0528800831, 0.0021644787, 0.0003519033, 0.0043251683, -0.0011509767, -0.0014063364, -0.0027098040, -0.0154139624, -0.0023018922, 0.0034703926, 0.0243312565, -0.0020915440, -0.0018027444, -0.0193959754, -0.0055565149, -0.0201738426, 0.5356642695, -0.0198276449, -0.0111388257, 0.1694444472, -0.0469793745, 0.1742687361, 0.0031582927, 0.0002348938, 0.0231282630, 0.0044870130, 0.0232195225, 0.0030052323, -0.0009517968, 0.0170753978, -0.0011381956, 0.5231507162, -0.0021499109, 0.5338118321, 0.2476679354, -0.0039309934, 0.5535510966]\n'
                'v=[[[[0.0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]\n'
                'k=0\n'
                'for p in range(4):\n'
                '    for q in range(p, 4):\n'
                '        for r in range(4):\n'
                '            for s in range(r, 4):\n'
                '                if (p, q) <= (r, s):\n'
                '                    x = vu[k]; k += 1\n'
                '                    for a, b in ((p, q), (q, p)):\n'
                '                        for c, d in ((r, s), (s, r)):\n'
                '                            v[a][b][c][d] = x; v[c][d][a][b] = x\n'
            ),
            "call": "pairing_intermediates(h, v)",
            "gold_call": "_oracle_pairing_intermediates(h, v)",
        },
        # more compact chain, both gaps larger
        {
            "setup": (
                'h = [[-1.1539080099, 0.1053264815, 0.0463726202, -0.0280445760], [0.1053264815, -1.0241112172, -0.0284914983, 0.0379181832], [0.0463726202, -0.0284914983, -1.2334250804, 0.0928773226], [-0.0280445760, 0.0379181832, 0.0928773226, -1.0168415075]]\n'
                'vu = [0.5135964805, 0.0000892447, -0.0079090452, -0.0313735482, 0.5244457653, 0.0257707517, 0.0174065331, 0.1778097892, -0.0441056453, 0.1844507910, 0.2524470746, 0.0187905012, 0.0056492113, -0.0006336747, -0.0011005566, -0.0215122411, -0.0539167758, 0.0218439967, -0.0583325335, 0.0037263949, 0.0015359380, -0.0084463239, -0.0025197021, -0.0028913226, 0.0053189194, 0.0202661896, 0.0045722165, 0.0061017793, -0.0317597371, -0.0041505127, -0.0038201020, 0.0263076967, 0.0084316145, 0.0273321381, 0.5425205849, 0.0268563349, 0.0181457122, 0.1821408535, -0.0463349098, 0.1889619280, 0.0054338877, 0.0017608443, -0.0300202990, -0.0089042914, -0.0297997581, 0.0051149717, -0.0031942132, -0.0214587206, -0.0029085976, 0.5398603494, -0.0038396762, 0.5497664297, 0.2343275658, -0.0065940196, 0.5734393087]\n'
                'v=[[[[0.0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]\n'
                'k=0\n'
                'for p in range(4):\n'
                '    for q in range(p, 4):\n'
                '        for r in range(4):\n'
                '            for s in range(r, 4):\n'
                '                if (p, q) <= (r, s):\n'
                '                    x = vu[k]; k += 1\n'
                '                    for a, b in ((p, q), (q, p)):\n'
                '                        for c, d in ((r, s), (s, r)):\n'
                '                            v[a][b][c][d] = x; v[c][d][a][b] = x\n'
            ),
            "call": "pairing_intermediates(h, v)",
            "gold_call": "_oracle_pairing_intermediates(h, v)",
        },
        # more stretched chain, both gaps smaller
        {
            "setup": (
                'h = [[-1.0535202653, -0.0954520982, 0.0327626406, 0.0265280288], [-0.0954520982, -0.9759840803, 0.0263393113, 0.0280724003], [0.0327626406, 0.0263393113, -1.0902056113, -0.0890301331], [0.0265280288, 0.0280724003, -0.0890301331, -0.9805693427]]\n'
                'vu = [0.4986283044, 0.0009087394, -0.0021323747, 0.0166899654, 0.5083061756, -0.0122484208, 0.0074694178, 0.1547693614, 0.0442260244, 0.1580641944, 0.2695937803, -0.0098650100, 0.0023145094, 0.0015931063, 0.0006372500, 0.0125755343, 0.0478636710, 0.0234927737, 0.0501471972, 0.0011245576, 0.0000018112, -0.0023411609, 0.0006633810, -0.0004719413, 0.0023832762, -0.0104105536, 0.0021787454, 0.0017649433, 0.0169221144, -0.0007425978, 0.0010368727, -0.0123895202, 0.0041589443, -0.0128551185, 0.5214820302, -0.0126649335, 0.0076872831, 0.1572982574, 0.0458610277, 0.1607090075, 0.0015955557, 0.0001499212, 0.0160117668, -0.0024541802, 0.0161441641, 0.0016663076, 0.0014156192, 0.0123300493, 0.0014898955, 0.5098825986, 0.0014581033, 0.5203040699, 0.2595073207, 0.0026422172, 0.5364785234]\n'
                'v=[[[[0.0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]\n'
                'k=0\n'
                'for p in range(4):\n'
                '    for q in range(p, 4):\n'
                '        for r in range(4):\n'
                '            for s in range(r, 4):\n'
                '                if (p, q) <= (r, s):\n'
                '                    x = vu[k]; k += 1\n'
                '                    for a, b in ((p, q), (q, p)):\n'
                '                        for c, d in ((r, s), (s, r)):\n'
                '                            v[a][b][c][d] = x; v[c][d][a][b] = x\n'
            ),
            "call": "pairing_intermediates(h, v)",
            "gold_call": "_oracle_pairing_intermediates(h, v)",
        },
        # widely separated bond pairs, weak inter-pair coupling
        {
            "setup": (
                'h = [[-1.1232826142, 0.0623146321, 0.0120829492, 0.0089777657], [0.0623146321, -0.9491444243, -0.0103217947, -0.0130190754], [0.0120829492, -0.0103217947, -1.0343155466, -0.0730466665], [0.0089777657, -0.0130190754, -0.0730466665, -0.9449557118]]\n'
                'vu = [0.5321511317, -0.0004865026, 0.0032464522, 0.0096629480, 0.5414372864, 0.0072981733, 0.0011824489, 0.1365461997, 0.0362040698, 0.1382569910, 0.2420949449, 0.0050424889, 0.0019152935, -0.0007224492, 0.0027549616, 0.0061302529, -0.0306684134, -0.0147513343, -0.0316364304, 0.0003087128, 0.0002633785, 0.0031251033, 0.0001050697, 0.0001577093, -0.0017968104, -0.0046552649, -0.0019179916, 0.0004846669, 0.0096512534, 0.0002752480, 0.0000811914, -0.0067421847, -0.0005267815, -0.0070069178, 0.5619543721, 0.0076867094, 0.0013012770, 0.1390665404, 0.0378633375, 0.1408849290, 0.0006100779, 0.0004579518, -0.0101667093, -0.0030235070, -0.0102946316, 0.0005425315, -0.0060490547, -0.0067399855, -0.0061950158, 0.5034270177, 0.0001644813, 0.5130934485, 0.2643063386, 0.0000928786, 0.5272470158]\n'
                'v=[[[[0.0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]\n'
                'k=0\n'
                'for p in range(4):\n'
                '    for q in range(p, 4):\n'
                '        for r in range(4):\n'
                '            for s in range(r, 4):\n'
                '                if (p, q) <= (r, s):\n'
                '                    x = vu[k]; k += 1\n'
                '                    for a, b in ((p, q), (q, p)):\n'
                '                        for c, d in ((r, s), (s, r)):\n'
                '                            v[a][b][c][d] = x; v[c][d][a][b] = x\n'
            ),
            "call": "pairing_intermediates(h, v)",
            "gold_call": "_oracle_pairing_intermediates(h, v)",
        },
        # strongly asymmetric chain, gaps far apart
        {
            "setup": (
                'h = [[-1.0605544628, 0.1109691270, 0.0337953986, -0.0275028430], [0.1109691270, -0.9967473907, -0.0249449747, 0.0327649004], [0.0337953986, -0.0249449747, -1.2443160028, 0.0817597006], [-0.0275028430, 0.0327649004, 0.0817597006, -0.9939225027]]\n'
                'vu = [0.4915101055, 0.0012858267, -0.0118137925, -0.0260152430, 0.5004507960, 0.0193482618, 0.0219267318, 0.1678926497, -0.0379522320, 0.1745941693, 0.2725692608, 0.0140407881, 0.0124760173, 0.0012716803, -0.0062844160, -0.0181475107, -0.0572727507, 0.0213445339, -0.0620901255, 0.0031266637, 0.0024968552, -0.0121840541, -0.0031543541, -0.0026739899, 0.0124141862, 0.0166071354, 0.0116688005, 0.0051392202, -0.0264054146, -0.0031714666, -0.0047666345, 0.0212675392, 0.0111768473, 0.0220123806, 0.5120507989, 0.0199322596, 0.0224684171, 0.1703955944, -0.0390431361, 0.1772614454, 0.0038997859, 0.0028014634, -0.0234232646, -0.0126297720, -0.0228421846, 0.0048810097, -0.0125818048, -0.0167007494, -0.0128161153, 0.5512448921, -0.0040750853, 0.5593630982, 0.2280982660, -0.0069854706, 0.5841558878]\n'
                'v=[[[[0.0]*4 for _ in range(4)] for _ in range(4)] for _ in range(4)]\n'
                'k=0\n'
                'for p in range(4):\n'
                '    for q in range(p, 4):\n'
                '        for r in range(4):\n'
                '            for s in range(r, 4):\n'
                '                if (p, q) <= (r, s):\n'
                '                    x = vu[k]; k += 1\n'
                '                    for a, b in ((p, q), (q, p)):\n'
                '                        for c, d in ((r, s), (s, r)):\n'
                '                            v[a][b][c][d] = x; v[c][d][a][b] = x\n'
            ),
            "call": "pairing_intermediates(h, v)",
            "gold_call": "_oracle_pairing_intermediates(h, v)",
        },
    ]
