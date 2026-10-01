"""
Compute the positive nodal magnitudes of the physical characteristic speed relative to each stretched component grid.

Inverting a time-dependent coordinate map gives $\\xi_t=-x_\\tau\\xi_x$, hence the transformed transport coefficient is $(c-x_\\tau)\\xi_x$. Compute the inverse spatial metric from the discrete Jacobian produced by the sixth/third-order derivative, not from the analytic derivative of the stretching map:



$$

X_{\\kappa,j}=\\frac{|c-x_\\tau^{(\\kappa)}|}{(D_\\kappa x_\\kappa)_j}.

$$



The outer-grid velocities vanish and the middle velocity is supplied separately. All three relative characteristics must have the same nonzero sign as $c$, and every discrete Jacobian and returned magnitude must be positive and finite.

Returns
-------
np.ndarray of length counts.sum(), positive |c-x_tau|/(D x) values in L, M, R order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_relative_metric_diagonals(
    c: float, grid_velocity: float, counts: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    r"""Return nodal magnitudes $|c-x_\tau|/(D_\kappa x_\kappa)$.

    Parameters
    ----------
    c : float
        Finite nonzero physical advection speed. Its sign determines the
        common incoming boundary.
    grid_velocity : float
        Middle-grid velocity $x_\tau$; the outer-grid velocities are zero.
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    nodes : np.ndarray
        Concatenated physical nodes of the three grids in left-middle-right
        order, of length ``counts.sum()``.

    Returns
    -------
    np.ndarray
        Concatenated metric diagonals of length ``counts.sum()`` in
        left-middle-right order.

    Raises
    ------
    ValueError
        If ``counts`` is not an integer array of shape ``(3,)`` or any count is
        below eighteen; if ``nodes`` has the wrong length or is not finite; if
        ``c`` is zero or non-finite; if ``grid_velocity`` is not finite; if the
        middle relative characteristic has a different sign from ``c``; or if
        any discrete Jacobian or metric magnitude is not positive and finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_metric_inputs(c, grid_velocity, counts, nodes):
    counts = np.asarray(counts)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size != int(np.sum(counts)):
        raise ValueError("nodes must be a 1D array of length counts.sum()")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be finite")
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(grid_velocity):
        raise ValueError("grid_velocity must be finite")
    return float(c), float(grid_velocity), counts.astype(int), nodes


def _oracle_compute_relative_metric_diagonals(
    c: float, grid_velocity: float, counts: np.ndarray, nodes: np.ndarray
) -> np.ndarray:
    """Reference discrete-Jacobian evaluation of the nodal scales."""
    c, grid_velocity, counts, nodes = _validated_metric_inputs(
        c, grid_velocity, counts, nodes
    )
    starts = np.r_[0, np.cumsum(counts)]
    pieces = []
    for grid in range(3):
        derivative = _oracle_build_sbp63_operator(int(counts[grid]))[2]
        jacobian = derivative @ nodes[starts[grid] : starts[grid + 1]]
        relative = c - (grid_velocity if grid == 1 else 0.0)
        if relative == 0.0 or np.sign(relative) != np.sign(c):
            raise ValueError("all grid-relative characteristics must share c's sign")
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = abs(relative) / jacobian
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                "the discrete Jacobian and metric magnitude must stay positive"
            )
        pieces.append(scale)
    return np.concatenate(pieces)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return positive, negative, static, and reversed-characteristic cases."""
    return [
        {
            "setup": """import numpy as np
counts = np.array([41, 49, 45])
sigmas = np.array([0.25, -0.15, 0.20])
state = np.array([0.6283185307179586, -1.0, -0.983119218812023, -0.9662037932714901, -0.9492192926197991, -0.9321317119693779, -0.9149076819551246, -0.8975146744676389, -0.8799212031998337, -0.8620970177704887, -0.8440132902279098, -0.8256427927838372, -0.8069600656818148, -0.7879415741650544, -0.7685658535760379, -0.7488136416932683, -0.7286679974892702, -0.7081144055776591, -0.6871408657053233, -0.6657379667379764, -0.643898944682929, -0.6216197243913529, -0.5988989446829289, -0.5757379667379764, -0.5521408657053233, -0.528114405577659, -0.5036679974892703, -0.4788136416932681, -0.45356585357603785, -0.4279415741650544, -0.40196006568181464, -0.37564279278383716, -0.3490132902279097, -0.3220970177704887, -0.2949212031998336, -0.26751467446763877, -0.23990768195512469, -0.21213171196937797, -0.18421929261979897, -0.15620379327148992, -0.12811921881202293, -0.09999999999999998, -0.35, -0.33323072809142906, -0.31647081670742866, -0.29972958628936613, -0.28301627728384465, -0.2663400105736924, -0.2497097484209418, -0.23313425608904792, -0.2166220643086843, -0.20018143274784497, -0.18382031464267767, -0.16754632274050585, -0.15136669670087596, -0.13528827209422367, -0.11931745113091186, -0.10346017524598308, -0.0877218996570226, -0.07210757000407886, -0.05662160117167386, -0.04126785838358982, -0.026049640651387318, -0.010969666647531628, 0.00396993693638048, 0.01876764449644036, 0.03342253804929801, 0.047934311163107035, 0.06230327026971377, 0.0765303333524684, 0.09061702601527927, 0.10456547494974355, 0.11837839882832613, 0.13205909666258775, 0.1456114336763107, 0.1590398247540169, 0.17234921553575477, 0.18554506123910963, 0.198633303299124, 0.2116203439261608, 0.22451301869065565, 0.23731856725215505, 0.2500446023579823, 0.2626990772442853, 0.2752902515790582, 0.2878266560929742, 0.30031705604948855, 0.31277041371063385, 0.325195849959238, 0.3376026052419042, 0.35, 0.1, 0.11636711134331885, 0.1327550513139448, 0.14918454240127452, 0.16567609535973835, 0.18224990469069716, 0.19892574573589242, 0.2157228739068377, 0.23265992656365514, 0.2497548280433605, 0.2670246983215503, 0.2844857657729303, 0.30215328447523804, 0.32004145647795945, 0.3381633594319395, 0.3565308799486715, 0.37515465302884965, 0.3940440078688433, 0.4132069203212503, 0.4326499722517807, 0.45237831799957795, 0.4723956581118878, 0.49270422048691764, 0.5133047490209787, 0.5341964998177597, 0.5553772449790535, 0.576843283957614, 0.5985894624142979, 0.620609198483395, 0.6428945163123078, 0.6654360867046668, 0.6882232746597776, 0.7112441935661472, 0.7344857657729302, 0.7579337892306411, 0.7815730098615423, 0.8053871992909278, 0.8293592375432013, 0.853471200281347, 0.8777044501452426, 0.9020397317233748, 0.9264572696740019, 0.950936869495763, 0.975458020434228, 1.0], dtype=np.float64)
c = 1.0
grid_velocity = float(state[0])
nodes = state[1:]
""",
            "call": "compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
            "gold_call": "_oracle_compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
        },
        {
            "setup": """import numpy as np
counts = np.array([18, 19, 20])
sigmas = np.array([0.0, 0.6, -0.6])
state = np.array([0.3141592653589793, -1.0, -0.9470588235294117, -0.8941176470588236, -0.8411764705882353, -0.788235294117647, -0.7352941176470589, -0.6823529411764706, -0.6294117647058823, -0.5764705882352941, -0.5235294117647058, -0.47058823529411764, -0.41764705882352937, -0.3647058823529412, -0.31176470588235294, -0.2588235294117647, -0.20588235294117652, -0.15294117647058825, -0.09999999999999998, -0.35, -0.33432616241216806, -0.3179469472379363, -0.30017840943192936, -0.28037881781390717, -0.25796815374594495, -0.232445734705243, -0.20340542726922417, -0.17054798727406575, -0.13369015219719207, -0.09277020949628795, -0.04784987171366867, 0.0008875986280902914, 0.05314295736516611, 0.10851007107498162, 0.16648825723473726, 0.22649749720650802, 0.28789605981005406, 0.35, 0.1, 0.17566014711394473, 0.2505485696919624, 0.32391459383173027, 0.3950490726917245, 0.46330373016844917, 0.5281088441786098, 0.5889887871923913, 0.6455750001126024, 0.6976160456060855, 0.7449844666587171, 0.7876802632704971, 0.8258308924555491, 0.8596877915470309, 0.8896195196421334, 0.916101704270672, 0.9397040675159407, 0.961074885481436, 0.9809233050086815, 1.0], dtype=np.float64)
c = -1.0
grid_velocity = float(state[0])
nodes = state[1:]
""",
            "call": "compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
            "gold_call": "_oracle_compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
        },
        {
            "setup": """import numpy as np
counts = np.array([18, 19, 20])
sigmas = np.array([0.0, 0.6, -0.6])
state = np.array([1.9236706937217898e-17, -1.0, -0.9470588235294117, -0.8941176470588236, -0.8411764705882353, -0.788235294117647, -0.7352941176470589, -0.6823529411764706, -0.6294117647058823, -0.5764705882352941, -0.5235294117647058, -0.47058823529411764, -0.41764705882352937, -0.3647058823529412, -0.31176470588235294, -0.2588235294117647, -0.20588235294117652, -0.15294117647058825, -0.09999999999999998, -0.3, -0.28432616241216807, -0.26794694723793633, -0.25017840943192937, -0.23037881781390718, -0.20796815374594496, -0.182445734705243, -0.15340542726922418, -0.12054798727406577, -0.08369015219719209, -0.04277020949628796, 0.002150128286331321, 0.05088759862809028, 0.1031429573651661, 0.1585100710749816, 0.21648825723473725, 0.276497497206508, 0.33789605981005405, 0.39999999999999997, 0.1, 0.17566014711394473, 0.2505485696919624, 0.32391459383173027, 0.3950490726917245, 0.46330373016844917, 0.5281088441786098, 0.5889887871923913, 0.6455750001126024, 0.6976160456060855, 0.7449844666587171, 0.7876802632704971, 0.8258308924555491, 0.8596877915470309, 0.8896195196421334, 0.916101704270672, 0.9397040675159407, 0.961074885481436, 0.9809233050086815, 1.0], dtype=np.float64)
c = 2.0
grid_velocity = float(state[0])
nodes = state[1:]
""",
            "call": "compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
            "gold_call": "_oracle_compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)",
        },
        {
            "setup": """import numpy as np
counts = np.array([18, 18, 18])
sigmas = np.array([0.25, -0.15, 0.20])
state = np.array([0.6283185307179586, -1.0, -0.9602189133524786, -0.9199896756298205, -0.8788793970042553, -0.8364851904392686, -0.7924478915244212, -0.7464642893885765, -0.6982974521824554, -0.6477847955085954, -0.5948436190380071, -0.5394739227706906, -0.4817584070356352, -0.42185965623030364, -0.3600146022039745, -0.2965264558277848, -0.23175438151217342, -0.16610126629365496, -0.09999999999999998, -0.35, -0.3026821541610002, -0.25557344549039746, -0.20887588924115144, -0.16277749936363542, -0.1174458859160544, -0.07302254730493885, -0.029618051726618855, 0.012691728100089561, 0.05386819868832482, 0.09391136003808698, 0.13285980563623767, 0.1707894082015926, 0.20781073593048216, 0.24406528722943677, 0.2797206721566613, 0.31496490466252913, 0.35, 0.1, 0.14241310461213483, 0.18518473008437886, 0.2286611882789487, 0.27316478882505574, 0.3189828632510513, 0.3663579802538447, 0.41547968531285917, 0.466478045946065, 0.5194192224166532, 0.574303214724624, 0.6310638626067859, 0.6895710985451688, 0.7496353770603498, 0.8110141294554192, 0.8734200242020259, 0.9365307516709583, 1.0], dtype=np.float64)
c = 1.0
grid_velocity = 1.0
nodes = state[1:]
def run_model():
    try:
        compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_relative_metric_diagonals(c, grid_velocity, counts, nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
