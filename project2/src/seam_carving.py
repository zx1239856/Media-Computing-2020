import numpy as np
import cv2
from tqdm import tqdm
from numba import jit
from scipy.ndimage import convolve
import scipy.sparse as sp
from scipy.sparse import linalg

PROTECT_MASK_ENERGY = 1E6


def remove_single_seam(src, seam_mask):
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.dstack([seam_mask] * c)
        dst = src[seam_mask].reshape(h, w - 1, c)
    else:
        h, w = src.shape
        dst = src[seam_mask].reshape(h, w - 1)
    return dst

def mask_from_seam(src, seam):
    return ~np.eye(src.shape[1], dtype=np.bool)[seam]

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


@jit(forceobj=True)
def get_entropy_energy(gray):
    assert gray.ndim == 2
    e = get_backward_energy(gray) / 255.
    w_size = 9
    pad = np.pad(gray, w_size // 2)
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            blk = pad[i:i+w_size, j:j+w_size]
            p = cv2.calcHist([blk], [0], None, [256], (0, 256)) / blk.size
            entropy = -(p * np.log(p + 1e-5)).sum()
            e[i, j] += entropy
    return e


@jit(forceobj=True)
def get_hog_energy(gray):
    assert gray.ndim == 2
    e = get_backward_energy(gray)
    w_size = 11
    pad = np.pad(e, w_size // 2)
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            blk = pad[i:i+w_size, j:j+w_size]
            max_v = int(blk.max()) + 1
            p = cv2.calcHist([blk], [0], None, [max_v], (0, max_v))
            e[i, j] /= p.max()
    return e


def get_backward_energy(gray):
    assert gray.ndim == 2
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    return np.abs(gx) + np.abs(gy)


@jit(forceobj=True)
def get_forward_energy(gray):
    """
    Forward energy defined in "Improved Seam Carving for Video Retargeting" by Rubinstein, Shamir, Avidan.
    """
    assert gray.ndim == 2
    h, w = gray.shape
    energy = np.zeros((h, w), dtype=np.float32)
    m = np.zeros_like(energy)

    U = np.roll(gray, 1, axis=0)  # i - 1
    L = np.roll(gray, 1, axis=1)  # j - 1
    R = np.roll(gray, -1, axis=1)  # j + 1

    cU = np.abs(R - L)
    cR = cU + np.abs(U - R)
    cL = cU + np.abs(U - L)

    # dp
    for i in range(1, h):
        mU = m[i - 1]  # M(i-1, j)
        mL = np.roll(mU, 1)  # M(i-1, j-1)
        mR = np.roll(mU, -1)  # M(i-1, j+1)

        m_all = np.array([mU, mL, mR])
        c_all = np.array([cU[i], cL[i], cR[i]])
        m_all += c_all

        argmins = np.argmin(m_all, axis=0)
        m[i] = np.choose(argmins, m_all)
        energy[i] = np.choose(argmins, c_all)

    return energy


ENERGY_FUNC = {
    "forward": get_forward_energy,
    "backward": get_backward_energy,
    "hog": get_hog_energy
}


@jit(forceobj=True)
def get_single_seam(energy):
    assert energy.ndim == 2
    h, w = energy.shape
    cost = energy[0]
    dp = np.zeros_like(energy, dtype=np.int32)
    offset = np.arange(-1, w - 1)

    # vectorize column access
    for i in range(1, h):
        bound = np.array([np.inf])
        left = np.hstack((bound, cost[:-1]))
        right = np.hstack((cost[1:], bound))
        min_indices = np.argmin([left, cost, right], axis=0) + offset
        dp[i] = min_indices
        cost = cost[min_indices] + energy[i]

    j = np.argmin(cost)
    min_cost = cost[j]
    seam = np.empty(h, dtype=np.int32)
    for i in range(h - 1, -1, -1):
        seam[i] = j
        j = dp[i, j]

    return seam, min_cost


def get_energy(gray, energy_type, keep_mask):
    energy = ENERGY_FUNC[energy_type](gray)
    if keep_mask is not None:
        energy[keep_mask] += PROTECT_MASK_ENERGY
    return energy


def get_seams(gray, num_seams, energy_type, keep_mask=None, progress=True):
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    row_indices = np.arange(0, h, dtype=np.int32)
    indices = np.arange(0, w, dtype=np.int32)[None, ...].repeat(h, axis=0)

    iter = tqdm(range(num_seams)) if progress else range(num_seams)
    for _ in iter:
        energy = get_energy(gray, energy_type, keep_mask)

        seam, _min_cost = get_single_seam(energy)
        seams_mask[row_indices, indices[row_indices, seam]] = True
        seam_mask = mask_from_seam(gray, seam)

        indices = remove_single_seam(indices, seam_mask)
        gray = remove_single_seam(gray, seam_mask)
        if keep_mask is not None:
            keep_mask = remove_single_seam(keep_mask, seam_mask)

    return seams_mask


@jit(forceobj=True)
def _expand_from_mask(src, mask, target_size):
    dst = np.zeros(target_size, dtype=src.dtype)
    h, w = src.shape[:2]
    for i in range(h):
        d_j = 0
        for j in range(w):
            if mask[i, j]:
                lo = max(0, j - 1)
                hi = j + 1
                dst[i, d_j] = src[i, lo:hi].mean(axis=0)
                d_j += 1
            dst[i, d_j] = src[i, j]
            d_j += 1
    return dst


def resize_width(src_list, dw, energy_type, keep_mask=None, progress=True):
    def single_forward(item):
        target_size = list(item.shape)
        target_size[1] += dw
        if dw > 0:
            seams_mask = get_seams(gray, dw, energy_type, keep_mask, progress)
            dst = _expand_from_mask(item, seams_mask, target_size)
            return dst
        elif dw < 0:
            seams_mask = get_seams(gray, -dw, energy_type, keep_mask, progress)
            return item[~seams_mask].reshape(target_size)
        else:
            return item

    if isinstance(src_list, list):
        assert len(src_list) > 0
        src = src_list[0]
        gray = bgr2gray(src)
        return [single_forward(item) for item in src_list]
    else:
        src = src_list
        gray = bgr2gray(src)
        return single_forward(src)


def resize_height(src_list, dh, energy_type, keep_mask=None, progress=True):
    def single_cvt(src):
        if src.ndim == 3:
            src = src.transpose(1, 0, 2)
        else:
            src = src.T
        return src
    
    if isinstance(src_list, list):
        assert len(src_list) > 0
        return [single_cvt(item) for item in resize_width([single_cvt(src) for src in src_list], dh, energy_type, keep_mask, progress)]
    else:
        return single_cvt(resize_width(single_cvt(src_list), dh, energy_type, keep_mask, progress))


@jit(forceobj=True)
def _resize_optimal_impl(src, dw, dh, energy_type, keep_mask=None):
    gray = bgr2gray(src)
    ddw, ddh = np.abs(dw), np.abs(dh)
    dp = np.zeros((ddh + 1, ddw + 1), dtype=np.float32)
    orient = np.zeros(dp.shape, dtype=np.bool)
    mem = {(0, 0): gray}

    HORIZ = 0
    VERT = 1

    for _ in tqdm(range((ddh + 1) * (ddw + 1))):
        h, w = _ // (ddw + 1), _ % (ddw + 1)
        if h == 0 and w == 0:
            continue
        
        cost_horiz, cost_vert = np.inf, np.inf
        seam_vert, seam_horiz = None, None
        if w > 0:
            energy = get_energy(mem[h, w - 1], energy_type, keep_mask)
            seam_vert, cost_vert = get_single_seam(energy)
            cost_vert += dp[h, w - 1]
        
        if h > 0:
            energy = get_energy(mem[h - 1, w].T, energy_type, keep_mask)
            seam_horiz, cost_horiz = get_single_seam(energy)
            cost_horiz += dp[h - 1, w]

        assert cost_horiz != np.inf or cost_vert != np.inf
        
        if cost_horiz < cost_vert:
            dp[h, w] = cost_horiz
            mem[h, w] = remove_single_seam(mem[h - 1, w].T, mask_from_seam(mem[h - 1, w].T, seam_horiz)).T
            orient[h, w] = HORIZ
        else:
            dp[h, w] = cost_vert
            mem[h, w] = remove_single_seam(mem[h, w - 1], mask_from_seam(mem[h, w - 1], seam_vert))
            orient[h, w] = VERT

    seam_path = []
    hh, ww = ddh, ddw
    while hh > 0 or ww > 0:
        seam_path.append(orient[hh, ww])
        if orient[hh, ww] == VERT:
            ww -= 1
        else:
            hh -= 1
    
    assert len(seam_path) == ddw + ddh
    return seam_path


def resize_optimal(src_list, dw, dh, energy_type, keep_mask=None):
    if isinstance(src_list, list):
        src = src_list[0]
    else:
        src = src_list

    seam_path = _resize_optimal_impl(src, dw, dh, energy_type, keep_mask)
    ddw, ddh = np.abs(dw), np.abs(dh)
    w_step, h_step = dw // ddw, dh // ddh
    VERT = 1
    
    dst = src_list
    for ori in tqdm(seam_path[::-1]):
        if ori == VERT:
            dst = resize_width(dst, w_step, energy_type, keep_mask, progress=False)
        else:
            dst = resize_height(dst, h_step, energy_type, keep_mask, progress=False)
    return dst



def resize(src, size, energy_type, keep_mask=None, order='width_first'):
    if isinstance(src, list):
        assert len(src) > 0
        src_ = src[0]
        assert len({item.shape[:2] for item in src}) == 1
    else:
        src_ = src

    src_h, src_w = src_.shape[:2]
    dst_h, dst_w = size
    assert dst_h > 0 and dst_w > 0 and dst_h < 2 * src_h and dst_w < 2 * src_w
    assert energy_type in ENERGY_FUNC.keys()
    assert order in ['width_first', 'height_first', 'optimal']
    if keep_mask is not None:
        assert keep_mask.shape[0] == src_h and keep_mask.shape[1] == src_w and keep_mask.ndim == 2 and keep_mask.dtype == np.bool

    if order == 'width_first':
        dst = resize_width(src, dst_w - src_w, energy_type, keep_mask)
        dst = resize_height(dst, dst_h - src_h, energy_type, keep_mask)
    elif order == 'height_first':
        dst = resize_height(src, dst_h - src_h, energy_type, keep_mask)
        dst = resize_width(dst, dst_w - src_w, energy_type, keep_mask)
    else:
        # optimal
        dst = resize_optimal(src, dst_w - src_w, dst_h - src_h, energy_type, keep_mask)
    return dst


@jit
def _get_coo_indices(size_y, size_x, mask):
    flat_idx = lambda i, j : i * size_x + j
    row = []
    col = []
    data = []
    for i in range(size_y):
        for j in range(size_x):
            m = flat_idx(i, j)
            row.append(m)
            col.append(m)
            data.append(4)
            if i - 1 >= 0 and mask[i - 1, j]:
                row.append(m)
                col.append(flat_idx(i - 1, j))
                data.append(-1)
            if i + 1 < size_y and mask[i + 1, j]:
                row.append(m)
                col.append(flat_idx(i + 1, j))
                data.append(-1)
            if j - 1 >= 0 and mask[i, j - 1]:
                row.append(m)
                col.append(flat_idx(i, j - 1))
                data.append(-1)
            if j + 1 < size_x and mask[i, j + 1]:
                row.append(m)
                col.append(flat_idx(i, j + 1))
                data.append(-1)
    return np.array(row), np.array(col), np.array(data)


def _build_sys_sparse(im_shape, mask):
    size_y, size_x = im_shape
    size = size_y * size_x
    row, col, data = _get_coo_indices(size_y, size_x, mask)
    A = sp.coo_matrix((data, (row, col)), shape=(size, size))
    A = A.tocsr()
    return A


def poisson_resize_width(gray, lap, ref, dw, energy_type):
    assert dw <= 0
    if dw == 0:
        return lap

    h, w = gray.shape
    row_indices = np.arange(0, h, dtype=np.int32)
    indices = np.arange(0, w, dtype=np.int32)[None, ...].repeat(h, axis=0)
    
    for _ in tqdm(range(abs(dw))):
        
        energy = get_energy(gray, energy_type, None)
        seam, _min_cost = get_single_seam(energy)

        [row_indices, indices[row_indices, seam]]

        seam_mask = mask_from_seam(gray, seam)
        gray = remove_single_seam(gray, seam_mask)
        lap = remove_single_seam(lap, seam_mask)
        ref = remove_single_seam(ref, seam_mask)
        l_bnd = ref[row_indices, seam - 1]
        r_bnd = ref[row_indices, seam + 1]
        lap[row_indices, seam - 1] += l_bnd
        lap[row_indices, seam] += r_bnd - l_bnd
        indices = remove_single_seam(indices, seam_mask)

        # !fix boundary conditions

    
    return gray, lap, ref

def poisson_resize(src, size, energy_type, keep_mask=None, order='width_first'):
    """Operates on gradient domain, and use poisson solver to rebuild image
    """
    assert size[0] <= src.shape[0] and size[1] <= src.shape[1], "Poisson solver only supports downscale"
    assert order in ['width_first', 'height_first'], "Poisson solver only supports naive order"

    gray = bgr2gray(src)

    def Laplacian_dim2(im):
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        conv =  convolve(im, laplacian_kernel, mode='constant')
        return conv

    def Laplacian(im):
        if im.ndim == 2:
            return Laplacian_dim2(im)
        else:
            res = []
            for ch in range(im.shape[2]):
                res.append(Laplacian_dim2(im[..., ch]))
            return np.dstack(res)

    lap = Laplacian(src / 255.)
    if order == 'width_first':
        gray, lap, src = poisson_resize_width(gray, lap, src, size[1] - src.shape[1], energy_type)
    else:
        pass
    
    mask = np.ones(lap.shape[:2], dtype=np.bool)
    A = _build_sys_sparse(lap.shape[:2], mask)
    all_res = []
    for ch in range(lap.shape[2]):
        b = lap[..., ch].flatten()
        res = linalg.spsolve(A, b)
        all_res.append(res.reshape(lap.shape[:2]))

    res = np.round(np.clip(np.dstack(all_res), 0, 1) * 255).astype(np.uint8)

    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()
    # target_shape = list(src.shape)
    # target_shape[0] = size[0]
    # target_shape[1] = size[1]
    # res = res[mask].reshape(target_shape)
    return res
    
