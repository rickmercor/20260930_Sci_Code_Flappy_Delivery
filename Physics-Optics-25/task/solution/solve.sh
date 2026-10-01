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

def normalize_mueller_stack(mueller_stack: np.ndarray) -> np.ndarray:
    matrices = np.asarray(mueller_stack, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1:] != (4, 4):
        raise ValueError("mueller_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("mueller_stack must be finite")
    scales = matrices[:, 0, 0]
    if np.any(scales <= 1e-12):
        raise ValueError("every M[0,0] must exceed 1e-12")
    with np.errstate(over="ignore", invalid="ignore"):
        result = matrices / scales[:, None, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized stack must be finite")
    return result

import numpy as np

def principal_mueller_logarithms(normalized_stack: np.ndarray, imag_tol: float = 1e-8) -> np.ndarray:
    from scipy.linalg import logm

    matrices = np.asarray(normalized_stack, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1:] != (4, 4):
        raise ValueError("normalized_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("normalized_stack must be finite")
    if isinstance(imag_tol, (bool, np.bool_)) or not isinstance(imag_tol, (int, float, np.integer, np.floating)):
        raise ValueError("imag_tol must be a real number")
    if not np.isfinite(imag_tol) or imag_tol <= 0:
        raise ValueError("imag_tol must be finite and positive")
    if np.any(np.abs(matrices[:, 0, 0] - 1.0) > 1e-8):
        raise ValueError("matrices must be intensity-normalized")
    result = np.empty_like(matrices)
    for index in range(matrices.shape[0]):
        value = np.asarray(logm(matrices[index]))
        if np.max(np.abs(np.imag(value))) > imag_tol:
            raise ValueError("matrix logarithm is not real")
        result[index] = np.real(value)
    return result

import numpy as np

def lorentz_decomposition(logarithm_stack: np.ndarray) -> np.ndarray:
    generators = np.asarray(logarithm_stack, dtype=float)
    if generators.ndim != 3 or generators.shape[0] < 1 or generators.shape[1:] != (4, 4):
        raise ValueError("logarithm_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(generators)):
        raise ValueError("logarithm_stack must be finite")
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    adjoint = np.einsum("ab,ncb,cd->nad", metric, generators, metric)
    material = 0.5 * (generators - adjoint)
    depolarization = generators - material
    return np.stack((material, depolarization))

import numpy as np

def anisotropy_coordinates(material_stack: np.ndarray) -> np.ndarray:
    material = np.asarray(material_stack, dtype=float)
    if material.ndim != 3 or material.shape[0] < 1 or material.shape[1:] != (4, 4):
        raise ValueError("material_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(material)):
        raise ValueError("material_stack must be finite")
    coordinates = np.stack(
        (-material[:, 0, 1], -material[:, 0, 2],
         material[:, 3, 2], material[:, 1, 3],
         material[:, 0, 3], material[:, 1, 2]),
        axis=-1,
    )
    ld, ldp, lb, lbp, cd, cb = np.moveaxis(coordinates, -1, 0)
    rebuilt = np.zeros_like(material)
    rebuilt[:, 0, 1] = rebuilt[:, 1, 0] = -ld
    rebuilt[:, 0, 2] = rebuilt[:, 2, 0] = -ldp
    rebuilt[:, 0, 3] = rebuilt[:, 3, 0] = cd
    rebuilt[:, 1, 2] = cb
    rebuilt[:, 2, 1] = -cb
    rebuilt[:, 1, 3] = lbp
    rebuilt[:, 3, 1] = -lbp
    rebuilt[:, 2, 3] = -lb
    rebuilt[:, 3, 2] = lb
    if not np.allclose(material, rebuilt, rtol=0.0, atol=1e-8):
        raise ValueError("material_stack violates the public entry map")
    return coordinates

import numpy as np

def conventional_circular_reading(normalized_stack: np.ndarray) -> np.ndarray:
    matrices = np.asarray(normalized_stack, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1:] != (4, 4):
        raise ValueError("normalized_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("normalized_stack must be finite")
    if np.any(np.abs(matrices[:, 0, 0] - 1.0) > 1e-8):
        raise ValueError("matrices must be intensity-normalized")
    output = matrices @ np.array([1.0, 0.0, 1.0, 0.0])
    q = output[:, 1]
    u = output[:, 2]
    if np.any(np.hypot(q, u) <= 1e-12):
        raise ValueError("output linear polarization must be nonzero")
    values = np.arctan2(q, u)
    return np.where(values <= -np.pi, np.pi, values)

import numpy as np

def material_model_reading(material_stack: np.ndarray) -> np.ndarray:
    import warnings

    from scipy.linalg import expm

    material = np.asarray(material_stack, dtype=float)
    if material.ndim != 3 or material.shape[0] < 1 or material.shape[1:] != (4, 4):
        raise ValueError("material_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(material)):
        raise ValueError("material_stack must be finite")
    model = np.empty_like(material)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index in range(material.shape[0]):
            model[index] = expm(material[index])
    if not np.all(np.isfinite(model)):
        raise ValueError("model matrices must be finite")
    scales = model[:, 0, 0]
    if np.any(scales <= 1e-12):
        raise ValueError("model M[0,0] must exceed 1e-12")
    model = model / scales[:, None, None]
    return conventional_circular_reading(model)

import numpy as np

def parity_defects(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first)
    right = np.asarray(second)
    for value in (left, right):
        if value.dtype.kind not in "iuf":
            raise ValueError("inputs must be real numeric arrays")
    if left.shape != right.shape or left.size < 1:
        raise ValueError("inputs must share one nonempty shape")
    left = left.astype(float)
    right = right.astype(float)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("inputs must be finite")
    difference = float(np.linalg.norm((left - right).ravel()))
    total = float(np.linalg.norm((left + right).ravel()))
    return np.array([difference / max(total, 1e-12), total / max(difference, 1e-12)])

import numpy as np

def closed_vector_winding(planar_vectors: np.ndarray, magnitude_floor: float = 1e-10) -> float:
    vectors = np.asarray(planar_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] < 3:
        raise ValueError("planar_vectors must have shape (n, 2) with n >= 3")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("planar_vectors must be finite")
    if isinstance(magnitude_floor, (bool, np.bool_)) or not isinstance(magnitude_floor, (int, float, np.integer, np.floating)):
        raise ValueError("magnitude_floor must be a real number")
    if not np.isfinite(magnitude_floor) or magnitude_floor <= 0:
        raise ValueError("magnitude_floor must be finite and positive")
    z = vectors[:, 0] + 1j * vectors[:, 1]
    if np.any(np.abs(z) <= magnitude_floor):
        raise ValueError("vector magnitude is at or below magnitude_floor")
    increments = np.angle(np.roll(z, -1) / z)
    increments = np.where(increments <= -np.pi, np.pi, increments)
    return float(np.sum(increments) / (2.0 * np.pi))

import numpy as np

def periodic_ring_rms(values: np.ndarray, azimuths_rad: np.ndarray) -> float:
    response = np.asarray(values, dtype=float)
    azimuths = np.asarray(azimuths_rad, dtype=float)
    if response.ndim != 1 or azimuths.ndim != 1 or response.size != azimuths.size or response.size < 3:
        raise ValueError("values and azimuths_rad must be one-dimensional with equal length n >= 3")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(azimuths)):
        raise ValueError("inputs must be finite")
    if azimuths[0] < 0 or azimuths[-1] >= 2 * np.pi or np.any(np.diff(azimuths) <= 0):
        raise ValueError("azimuths_rad must increase strictly inside [0, 2*pi)")
    gaps = np.diff(np.append(azimuths, azimuths[0] + 2 * np.pi))
    weights = 0.5 * (np.roll(gaps, 1) + gaps)
    return float(np.sqrt(np.sum(weights * response ** 2) / (2 * np.pi)))

import numpy as np

def _problem_inputs():
    samples = np.array([
        [[0.8653071300, -0.0781494224, 0.0488998759, 0.0170688722], [-0.0781961616, 0.8014568415, 0.0266321865, -0.1575368577], [0.0467076555, -0.0915734338, 0.7307419235, -0.3088412607], [0.0221856238, 0.1311203789, 0.3211293276, 0.7066762949]],
        [[0.8369387404, -0.1019701827, -0.0265960428, -0.0007625336], [-0.0978828378, 0.7343722252, 0.0748843101, 0.2733262659], [-0.0257695196, 0.0047065636, 0.7380209591, -0.2080095968], [0.0292723376, -0.2847973616, 0.1943005605, 0.6936486943]],
        [[0.8839121960, -0.0174714253, -0.0755386764, 0.0225782172], [-0.0177267863, 0.7936682171, 0.0868161055, 0.2508542420], [-0.0727035705, -0.0937442341, 0.8019370618, 0.0123126447], [0.0304532733, -0.2482925601, -0.0424726363, 0.7903528110]],
        [[0.8149627860, -0.0691646849, 0.0514860506, 0.0152629721], [-0.0759144422, 0.7360078662, -0.0860014591, -0.1251641702], [0.0385940982, 0.0247740301, 0.6912961065, -0.3059531908], [0.0202632143, 0.1466912139, 0.2964736972, 0.6830655887]],
        [[0.7615189412, 0.0181308638, -0.0396764486, -0.0176111526], [0.0109318487, 0.6053042830, -0.0699467856, 0.3485889215], [-0.0372602967, 0.0813469017, 0.7119725197, -0.0025055924], [-0.0265408048, -0.3464158835, 0.0437542074, 0.6127781360]],
        [[0.7967607310, -0.0941083378, -0.0253649268, -0.0271613833], [-0.0976340481, 0.6925395521, 0.0021124603, 0.2670542101], [-0.0266257497, 0.0718611750, 0.7143589360, -0.1809770644], [0.0013548452, -0.2560528809, 0.1944486737, 0.6514171916]],
    ])
    rings = np.array([
        [
            [[0.8535232862, -0.1158290805, -0.2012661370, 0.0020499657], [-0.1283819905, 0.5579715731, 0.1901728107, 0.4969655511], [-0.1935538212, 0.0978817589, 0.6940975638, -0.3290046156], [0.0024794954, -0.5202989366, 0.2910261541, 0.4657611464]],
            [[0.8802477307, -0.2403765875, -0.0364326575, 0.0022367311], [-0.2421150091, 0.8138226721, 0.0878262008, 0.0554595304], [-0.0222656307, 0.0000337850, 0.4483903913, -0.6303353073], [0.0024004702, -0.1001652548, 0.6248495130, 0.4524275721]],
            [[0.9028237687, -0.1495738309, 0.1944771324, 0.0024642943], [-0.1393474140, 0.5962068523, -0.1373847002, -0.5456117611], [0.2018507666, -0.2157612555, 0.6970168848, -0.3635586761], [0.0016329352, 0.5220689430, 0.3962099669, 0.4674006854]],
            [[0.9171940436, 0.0696059207, 0.2320101493, 0.0013738928], [0.0798229507, 0.5432501866, 0.1349546085, -0.6121034402], [0.2287383301, 0.0658568241, 0.8047593587, 0.2106281677], [0.0019252209, 0.6222953117, -0.1788016355, 0.5093953663]],
            [[0.9241318056, 0.2353866016, 0.0224360895, 0.0011420849], [0.2360495031, 0.8679709221, 0.0530470748, -0.0358163059], [0.0139685633, -0.0049768886, 0.5408323668, 0.6136497002], [0.0012853783, 0.0616414219, -0.6116279164, 0.5538715720]],
            [[0.9236640020, 0.1164104133, -0.1912693909, 0.0013993844], [0.1102050677, 0.6379747934, -0.1086514849, 0.5484420316], [-0.1948631869, -0.1599983085, 0.7707672349, 0.2995430953], [0.0006575380, -0.5369138806, -0.3194485549, 0.5464615559]],
            [[0.9195593711, -0.1101176627, -0.1859859251, 0.0007136875], [-0.1160268308, 0.6165482684, 0.1773054465, 0.5530884062], [-0.1824006238, 0.1265224485, 0.7561204735, -0.3452038704], [0.0014024485, -0.5657455223, 0.3243428545, 0.5075983814]],
            [[0.9105142108, -0.2094225806, 0.0394446272, 0.0014011987], [-0.2078233341, 0.8433022168, -0.0466720995, -0.1531505652], [0.0470858566, -0.1042563817, 0.4842020751, -0.6521115224], [0.0011469981, 0.1241087454, 0.6581897366, 0.4693586406]],
            [[0.9055750170, -0.0876898083, 0.1943163899, 0.0016897873], [-0.0790935214, 0.5364921172, -0.0986020290, -0.6233605085], [0.1979453214, -0.1665465153, 0.7834335820, -0.2420681496], [0.0013345553, 0.6098398006, 0.2740962783, 0.4732303757]],
            [[0.8996872946, 0.1517535814, 0.1601306286, 0.0016052519], [0.1601494646, 0.6777992427, 0.2051388389, -0.4332563824], [0.1517796214, 0.1214982949, 0.6712370500, 0.4560241549], [0.0019595398, 0.4611075513, -0.4280233214, 0.5140738679]],
            [[0.9005665487, 0.2266126532, -0.0447635953, 0.0020660011], [0.2234578385, 0.8249030891, -0.0177690167, 0.1559866290], [-0.0584543060, -0.1131687701, 0.5484105452, 0.5793841642], [0.0019291268, -0.1147144361, -0.5888947093, 0.5403597570]],
            [[0.9067965616, 0.0813690507, -0.2268601038, 0.0023588773], [0.0665548219, 0.5608859205, -0.0399624348, 0.5876523683], [-0.2316140921, -0.1407792244, 0.8050521444, 0.1634897841], [0.0021396372, -0.5737301464, -0.2068737744, 0.5293115444]],
        ],
        [
            [[0.8877242023, -0.1988130124, -0.2011744990, -0.1232612224], [-0.1988130124, 0.3973942024, -0.0014131889, 0.6913403561], [-0.2011744990, -0.0014131889, 0.7864971881, 0.0572058382], [0.1232612224, -0.6913403561, -0.0572058382, 0.3579036364]],
            [[0.9129381505, -0.1475288592, -0.2443899242, -0.1295988961], [-0.1475288592, 0.3933425287, -0.0981424857, 0.7016632567], [-0.2443899242, -0.0981424857, 0.7851347390, 0.2139311134], [0.1295988961, -0.7016632567, -0.2139311134, 0.3322564880]],
            [[0.9338441205, -0.0955222766, -0.2693990360, -0.1261691547], [-0.0955222766, 0.4426690821, -0.1661195509, 0.6782722508], [-0.2693990360, -0.1661195509, 0.7596967389, 0.3325990300], [0.1261691547, -0.6782722508, -0.3325990300, 0.3402829847]],
            [[0.9471604919, -0.0780473036, -0.2748169703, -0.1155825789], [-0.0780473036, 0.4884497949, -0.1668075220, 0.6659030097], [-0.2748169703, -0.1668075220, 0.7698153460, 0.3485169833], [0.1155825789, -0.6659030097, -0.3485169833, 0.3840555453]],
            [[0.9530004893, -0.0963710601, -0.2647630058, -0.1008398197], [-0.0963710601, 0.5011269154, -0.1168726887, 0.6833847122], [-0.2647630058, -0.1168726887, 0.8188418793, 0.2623371938], [0.1008398197, -0.6833847122, -0.2623371938, 0.4347769422]],
            [[0.9527412686, -0.1412875293, -0.2373050274, -0.0940829776], [-0.1412875293, 0.4696369011, -0.0338591194, 0.7244226003], [-0.2373050274, -0.0338591194, 0.8577995602, 0.1058602326], [0.0940829776, -0.7244226003, -0.1058602326, 0.4338169670]],
            [[0.9493315555, -0.1894499576, -0.1952020845, -0.0954275693], [-0.1894499576, 0.4386081308, 0.0746620992, 0.7478383500], [-0.1952020845, 0.0746620992, 0.8492633436, -0.0808770909], [0.0954275693, -0.7478383500, 0.0808770909, 0.3913081457]],
            [[0.9417230190, -0.2331830483, -0.1380633630, -0.1020241848], [-0.2331830483, 0.4613246232, 0.1872386770, 0.7170113403], [-0.1380633630, 0.1872386770, 0.7722094078, -0.2774335930], [0.1020241848, -0.7170113403, 0.2774335930, 0.3432957188]],
            [[0.9386784516, -0.2564057227, -0.1009787877, -0.1079150290], [-0.2564057227, 0.5134463476, 0.2200844552, 0.6725964460], [-0.1009787877, 0.2200844552, 0.7162322343, -0.3646174240], [0.1079150290, -0.6725964460, 0.3646174240, 0.3456246137]],
            [[0.9350488437, -0.2734292579, -0.0833238153, -0.1127894092], [-0.2734292579, 0.5524808109, 0.1986406613, 0.6426516216], [-0.0833238153, 0.1986406613, 0.7148295551, -0.3584140674], [0.1127894092, -0.6426516216, 0.3584140674, 0.3918269862]],
            [[0.9373644441, -0.2768092225, -0.1059532849, -0.1168682867], [-0.2768092225, 0.5384335103, 0.1554937379, 0.6616108908], [-0.1059532849, 0.1554937379, 0.7632205385, -0.2641155619], [0.1168682867, -0.6616108908, 0.2641155619, 0.4264889165]],
            [[0.9438496723, -0.2558956990, -0.1621220689, -0.1243762293], [-0.2558956990, 0.4793413902, 0.0870458763, 0.7099302929], [-0.1621220689, 0.0870458763, 0.8196157207, -0.1037595407], [0.1243762293, -0.7099302929, 0.1037595407, 0.4188699463]],
        ],
        [
            [[0.8288417814, -0.1903625457, -0.0954212133, 0.0002455790], [-0.1904727936, 0.7153763973, 0.1053401815, 0.2399439196], [-0.0952012182, 0.1018428201, 0.5524932868, -0.4743713150], [0.0005552446, -0.2413708759, 0.4736525316, 0.5044356433]],
            [[0.8546118141, -0.0265670974, -0.2169646392, 0.0010929518], [-0.0258114851, 0.5034994714, 0.0351334105, 0.5711107027], [-0.2170542866, 0.0335918377, 0.7739160271, -0.0681796938], [0.0012230959, -0.5712372358, 0.0671241263, 0.4948406000]],
            [[0.8768335193, 0.1877499629, -0.1171004578, 0.0000511428], [0.1876937892, 0.7310014728, -0.1331105568, 0.3147531980], [-0.1171879141, -0.1323133857, 0.5875377335, 0.4927190964], [-0.0005929996, -0.3151250799, -0.4924847224, 0.5182580582]],
            [[0.8910425884, 0.2032319615, 0.0787914413, -0.0014897869], [0.2035384183, 0.7958176696, 0.0901081643, -0.2094544794], [0.0780155647, 0.0911433808, 0.5789382848, 0.5272725922], [-0.0009715827, 0.2087125792, -0.5275612375, 0.5613025090]],
            [[0.8975695075, 0.0162253219, 0.2051633516, 0.0002436245], [0.0161877099, 0.6048117565, 0.0189086779, -0.5462907266], [0.2051661132, 0.0175928342, 0.8153127201, 0.0421640890], [0.0003573839, 0.5463362261, -0.0415805598, 0.5959623051]],
            [[0.8979279912, -0.1612948071, 0.1156392492, 0.0015764874], [-0.1615510861, 0.7606754272, -0.1122763049, -0.3268597768], [0.1152932111, -0.1161066750, 0.6642526710, -0.4422680480], [0.0009227428, 0.3253902327, 0.4433321078, 0.5884548216]],
            [[0.8944152277, -0.1823943931, -0.0626039830, -0.0004115063], [-0.1826291448, 0.8126465833, 0.0858432344, 0.1909297987], [-0.0619261046, 0.0818928278, 0.5797600789, -0.5477826743], [-0.0000213445, -0.1924453644, 0.5472591592, 0.5516745503]],
            [[0.8857490511, -0.0209201804, -0.1877847517, -0.0011084215], [-0.0220328812, 0.5337927184, 0.0342844784, 0.6060708959], [-0.1876593853, 0.0330149941, 0.8124342909, -0.0680984648], [-0.0009926354, -0.6061035843, 0.0678083282, 0.5155919131]],
            [[0.8813965393, 0.1400646481, -0.1334349936, 0.0005029760], [0.1403388054, 0.6858026346, -0.1464262336, 0.4149409911], [-0.1331497652, -0.1455131585, 0.6620908627, 0.4299769098], [0.0001044210, -0.4151737227, -0.4297537715, 0.5250101264]],
            [[0.8757265794, 0.1844239980, 0.0830782999, 0.0009500481], [0.1840253650, 0.7736858056, 0.0948417942, -0.2310052986], [0.0839501810, 0.0956735679, 0.6000804116, 0.5039269342], [0.0011577953, 0.2309671344, -0.5039447418, 0.5623454035]],
            [[0.8763850678, 0.0129766572, 0.2110458220, -0.0005239701], [0.0136218842, 0.5842939588, 0.0152968880, -0.5282059340], [0.2110054253, 0.0135281925, 0.8061288041, 0.0332326356], [-0.0004941645, 0.5282387545, -0.0327090313, 0.5817697130]],
            [[0.8823732641, -0.1964656279, 0.1041542648, -0.0008550538], [-0.1958064964, 0.7631796066, -0.0997403577, -0.2565027065], [0.1053791790, -0.1037927092, 0.6203029982, -0.4735841798], [-0.0010995765, 0.2553712127, 0.4741922183, 0.5693787070]],
        ],
    ])
    azimuths = np.array([0.00, 0.47, 1.02, 1.49, 2.07, 2.61, 3.12, 3.71, 4.18, 4.77, 5.29, 5.83])
    return samples, rings, azimuths

def enantiomer_geometric_ratio(sample_mueller: np.ndarray | None = None, ring_mueller: np.ndarray | None = None, azimuths_rad: np.ndarray | None = None) -> float:
    supplied = (sample_mueller, ring_mueller, azimuths_rad)
    if all(value is None for value in supplied):
        samples, rings, azimuths = _problem_inputs()
    elif any(value is None for value in supplied):
        raise ValueError("supply all three arrays or none of them")
    else:
        samples = np.asarray(sample_mueller, dtype=float)
        rings = np.asarray(ring_mueller, dtype=float)
        azimuths = np.asarray(azimuths_rad, dtype=float)
    if samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1:] != (4, 4):
        raise ValueError("sample_mueller must have shape (s, 4, 4) with s >= 2")
    if rings.ndim != 4 or rings.shape[0] < 1 or rings.shape[1] < 3 or rings.shape[2:] != (4, 4):
        raise ValueError("ring_mueller must have shape (r, m, 4, 4) with r >= 1 and m >= 3")
    ring_count, point_count = rings.shape[:2]

    sample_normalized = normalize_mueller_stack(samples)
    sample_material = lorentz_decomposition(principal_mueller_logarithms(sample_normalized))[0]
    sample_coordinates = anisotropy_coordinates(sample_material)

    ring_normalized = normalize_mueller_stack(rings.reshape(-1, 4, 4))
    ring_material = lorentz_decomposition(principal_mueller_logarithms(ring_normalized))[0]
    ring_coordinates = anisotropy_coordinates(ring_material).reshape(ring_count, point_count, 6)
    model_readings = material_model_reading(ring_material).reshape(ring_count, point_count)

    pairs = []
    for first in range(samples.shape[0]):
        for second in range(first + 1, samples.shape[0]):
            linear_even = parity_defects(sample_coordinates[first, :4], sample_coordinates[second, :4])[0]
            circular_odd = parity_defects(sample_coordinates[first, 4:], sample_coordinates[second, 4:])[1]
            if linear_even <= 0.08 and circular_odd <= 0.08:
                pairs.append((first, second))
    if len(pairs) != 1:
        raise ValueError("exactly one sample pair must be opposite enantiomers")

    geometric = []
    for index in range(ring_count):
        circular_peak = float(np.max(np.abs(ring_coordinates[index, :, 4:])))
        winding = closed_vector_winding(ring_coordinates[index, :, :2])
        if circular_peak <= 0.006 and abs(round(winding)) >= 1:
            geometric.append(index)
    if len(geometric) != 1:
        raise ValueError("exactly one ring must show the topology-induced geometric regime")

    first, second = pairs[0]
    ring = geometric[0]
    enantiomeric_cb = abs(sample_coordinates[first, 5] - sample_coordinates[second, 5]) / 2.0
    if enantiomeric_cb <= 1e-12:
        raise ValueError("the selected pair must carry intrinsic circular birefringence")
    false_cb = periodic_ring_rms(model_readings[ring] - ring_coordinates[ring, :, 5], azimuths)
    return float(false_cb / enantiomeric_cb)
SCICODE_GOLD_EOF
