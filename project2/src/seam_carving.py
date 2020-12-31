import numpy as np
import cv2
from tqdm import tqdm
from numba import jit
from scipy.ndimage import convolve
import scipy.sparse as sp
from scipy.sparse import linalg
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hog

MASK_ENERGY = 1E3


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


def get_entropy_energy(gray, w_size=9):
    assert gray.ndim == 2
    e = get_backward_energy(gray) / 255
    ent = entropy(gray.astype(np.uint8), disk(w_size))
    return e + ent


def patchify(img, patch_shape):
    # Adapted from:
    # https://stackoverflow.com/a/16788733
    img = np.ascontiguousarray(img)
    X, Y = img.shape
    x, y = patch_shape
    shape = (X-x+1), (Y-y+1), x, y
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def get_hog_energy(gray, w_size=11, orient_bins=8):
    assert gray.ndim == 2
    energy = get_backward_energy(gray)
    gray_pad = np.pad(gray, w_size // 2, mode='edge')
    patches = patchify(gray_pad, (w_size, w_size))
    tot = np.concatenate(np.concatenate(patches, axis=1), axis=1)
    res = hog(tot, orientations=orient_bins, pixels_per_cell=(
        w_size, w_size), cells_per_block=(1, 1), feature_vector=False).squeeze()
    return np.divide(energy, res.max(-1))


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
    "hog": get_hog_energy,
    "entropy": get_entropy_energy
}


# @jit(forceobj=True)
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


def get_energy(gray, energy_type, keep_mask=None):
    energy = ENERGY_FUNC[energy_type](gray)
    if keep_mask is not None:
        energy[keep_mask] += MASK_ENERGY
    return energy


def accumulate_energy(M):
    M = M.copy()
    h, w = M.shape
    offset = np.arange(-1, w - 1)
    for i in range(1, h):
        bound = np.array([np.inf])
        left = np.hstack((bound, M[i-1, :-1]))
        right = np.hstack((M[i-1, 1:], bound))
        min_indices = np.argmin([left, M[i-1], right], axis=0) + offset
        M[i] += M[i-1, min_indices]
    return M


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
    if isinstance(src_list, list):
        assert len(src_list) > 0
        src = src_list[0]
        gray = bgr2gray(src)
    else:
        gray = bgr2gray(src_list)

    if dw > 0:
        seams_mask = get_seams(gray, dw, energy_type, keep_mask, progress)
    elif dw < 0:
        seams_mask = get_seams(gray, -dw, energy_type, keep_mask, progress)
    else:
        seams_mask = None

    def single_forward(item):
        target_size = list(item.shape)
        target_size[1] += dw
        if dw > 0:
            dst = _expand_from_mask(item, seams_mask, target_size)
            return dst
        elif dw < 0:
            return item[~seams_mask].reshape(target_size)
        else:
            return item

    if isinstance(src_list, list):
        return [single_forward(item) for item in src_list]
    else:
        return single_forward(src_list)


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
            mem[h, w] = remove_single_seam(
                mem[h - 1, w].T, mask_from_seam(mem[h - 1, w].T, seam_horiz)).T
            orient[h, w] = HORIZ
        else:
            dp[h, w] = cost_vert
            mem[h, w] = remove_single_seam(
                mem[h, w - 1], mask_from_seam(mem[h, w - 1], seam_vert))
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
    return seam_path, dp


def resize_optimal(src_list, dw, dh, energy_type, keep_mask=None):
    if isinstance(src_list, list):
        src = src_list[0]
    else:
        src = src_list

    seam_path = _resize_optimal_impl(src, dw, dh, energy_type, keep_mask)[0]
    ddw, ddh = np.abs(dw), np.abs(dh)
    w_step, h_step = dw // ddw if ddw > 0 else 0, dh // ddh if ddh > 0 else 0
    VERT = 1

    dst = src_list
    for ori in tqdm(seam_path[::-1]):
        if ori == VERT:
            dst = resize_width(dst, w_step, energy_type,
                               keep_mask, progress=False)
        else:
            dst = resize_height(dst, h_step, energy_type,
                                keep_mask, progress=False)
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
        dst = resize_optimal(src, dst_w - src_w, dst_h -
                             src_h, energy_type, keep_mask)
    return dst


def remove_object(src, energy_type, remove_mask, keep_mask=None):
    if isinstance(src, list):
        src_ = src[0]
    else:
        src_ = src

    h, w = src_.shape[:2]
    assert src_.shape[:2] == remove_mask.shape
    if keep_mask is not None:
        assert remove_mask.shape == keep_mask.shape

    gray = bgr2gray(src_)
    seams_mask = np.zeros((h, w), dtype=np.bool)
    row_indices = np.arange(0, h, dtype=np.int32)
    indices = np.arange(0, w, dtype=np.int32)[None, ...].repeat(h, axis=0)

    total = np.count_nonzero(remove_mask)
    pbar = tqdm(total=total)
    prev = total
    while remove_mask.any():
        energy = get_energy(gray, energy_type, keep_mask)
        energy[remove_mask] -= 100 * MASK_ENERGY

        seam, _min_cost = get_single_seam(energy)
        seams_mask[row_indices, indices[row_indices, seam]] = True
        seam_mask = mask_from_seam(gray, seam)
        indices = remove_single_seam(indices, seam_mask)
        gray = remove_single_seam(gray, seam_mask)
        remove_mask = remove_single_seam(remove_mask, seam_mask)
        if keep_mask is not None:
            keep_mask = remove_single_seam(keep_mask, seam_mask)

        cnt = np.count_nonzero(remove_mask)
        pbar.update(prev - cnt)
        prev = cnt

        w -= 1
    pbar.close()

    def reducer(img, seams_mask, target_size):
        size = list(img.shape)
        size[0] = target_size[0]
        size[1] = target_size[1]
        return img[~seams_mask].reshape(size)

    t_size = (h, w)

    if isinstance(src, list):
        return [reducer(item, seams_mask, t_size) for item in src]
    else:
        return reducer(src, seams_mask, t_size)


# Carving via Poisson Solver

@jit
def _get_coo_indices(size_y, size_x, mask):
    def flat_idx(i, j): return i * size_x + j
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


def _gen_idx_mat(size):
    h, w = size
    i = np.arange(0, h)[..., None].repeat(w, axis=1)
    j = np.arange(0, w)[None, ...].repeat(h, axis=0)
    return np.dstack([i, j])


def _mask_from_indices(size, indices):
    mask = np.zeros(size, dtype=np.bool)
    indices = np.array(indices).T.tolist()
    mask[tuple(indices)] = True
    return mask


@jit(forceobj=True)
def _poisson_boundary_cond(src, indices, lap):
    h, w = src.shape[:2]
    hh, ww = indices.shape[:2]
    res = lap.copy()
    for r in range(hh):
        for c in range(ww):
            curr = indices[r, c]
            color = src[curr[0], curr[1]]
            if r == 0 and curr[0] != 0:
                res[curr[0], curr[1]] += color
            if r + 1 == hh and curr[0] + 1 != h:
                res[curr[0], curr[1]] += color
            if c == 0 and curr[1] != 0:
                res[curr[0], curr[1]] += color
            if c + 1 == ww and curr[1] + 1 != w:
                res[curr[0], curr[1]] += color
            if r + 1 < hh:
                down = indices[r + 1, c]
                if curr[0] + 1 != down[0] or curr[1] != down[1]:
                    res[curr[0], curr[1]] += color
                    res[down[0], down[1]] += color
            if c + 1 < ww:
                right = indices[r, c + 1]
                if curr[0] != right[0] or curr[1] + 1 != right[1]:
                    res[curr[0], curr[1]] += color
                    res[right[0], right[1]] += color
    return res


def poisson_wrapper(resize_func):
    import inspect

    def inner(*args, **kwargs):
        bound_args = inspect.getcallargs(resize_func, *args, **kwargs)
        assert 'src' in bound_args.keys()
        src = bound_args['src']
        h, w = src.shape[:2]

        if 'size' in bound_args.keys():
            size = bound_args['size']
            assert size[0] <= h and size[1] <= w, "Poisson solver only supports downscale"
        else:
            assert 'remove_mask' in bound_args.keys()
            size = None

        def Laplacian_dim2(im):
            laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            conv = convolve(im, laplacian_kernel, mode='constant')
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
        indices = _gen_idx_mat((h, w))

        bound_args['src'] = [src, indices]
        _, indices = resize_func(**bound_args)

        mask = _mask_from_indices((h, w), indices)
        print("Fix boundary condition for Poisson")
        lap = _poisson_boundary_cond(src / 255., indices, lap)
        lap[~mask] = 0

        A = _build_sys_sparse(lap.shape[:2], mask)
        print("Build coefficient matrix done, start solving poisson equation")
        all_res = []
        for ch in range(lap.shape[2]):
            b = lap[..., ch].flatten()
            res = linalg.spsolve(A, b)
            all_res.append(res.reshape(lap.shape[:2]))

        res = np.clip(np.round(np.dstack(all_res) * 255),
                      0, 255).astype(np.uint8)
        print("Solve done!")

        if size is None:
            size = indices.shape[:2]

        target_shape = list(src.shape)
        target_shape[0] = size[0]
        target_shape[1] = size[1]
        res_collated = np.zeros(target_shape, dtype=np.uint8)
        target_indices = _gen_idx_mat(size)
        res_collated[tuple(target_indices.T.tolist())
                     ] = res[tuple(indices.T.tolist())]
        return res_collated

    return inner
