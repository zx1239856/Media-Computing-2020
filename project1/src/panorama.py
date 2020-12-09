from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
import maxflow
import matplotlib.pyplot as plt
from scipy.sparse import linalg
import scipy.sparse as sp
import scipy.ndimage as nd
from numba import jit

## Poisson matting
def laplacian(im):
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    conv =  nd.convolve(im, laplacian_kernel, mode='constant')    
    return conv

@jit
def make_rcd(im_shape, mask):
    sizey, sizex = im_shape
    r = []
    c = []
    d = []
    for j in range(sizey):
        for i in range(sizex):
            m = j * sizex + i
            r.append(m)
            c.append(m)
            d.append(4)
            if j - 1 >= 0 and mask[j - 1, i]:
                r.append(m)
                c.append(m - sizex)
                d.append(-1)
            if j + 1 < sizey and mask[j + 1, i]:
                r.append(m)
                c.append(m + sizex)
                d.append(-1)
            if i - 1 >= 0 and mask[j, i - 1]:
                r.append(m)
                c.append(m - 1)
                d.append(-1)
            if i + 1 < sizex and mask[j, i + 1]:
                r.append(m)
                c.append(m + 1)
                d.append(-1)
    return np.array(r), np.array(c), np.array(d)


def build_sys_sparse(im_shape, mask):
    sizey, sizex= im_shape
    size = sizey * sizex
    r, c, d = make_rcd(im_shape, mask)
    A = sp.coo_matrix((d, (r, c)), shape=(size, size))
    A = A.tocsr()
    return A

def poisson(L, R, L_mask, R_mask):
    A = build_sys_sparse(L.shape[:-1], L_mask)
    L = L / 255.
    R = R / 255.
    all_res = []
    for ch in range(L.shape[-1]):
        b = (laplacian(L[..., ch]) * L_mask).flatten()
        b_edge = (laplacian(R[..., ch] * R_mask) * L_mask).flatten()
        b -= b_edge
        res = linalg.spsolve(A, b)
        res = np.clip(res.reshape(L.shape[:2]), 0, 1)
        all_res.append(res)
    result = np.dstack(all_res)
    result *= L_mask[..., None]
    result += (R * R_mask[..., None])
    result = np.clip(np.round(result * 255.).astype(int), 0, 255).astype(np.uint8)
    return result

INFTY = float('inf')

@jit
def edge_contraint(in_mask, overlap_mask):
    res = np.zeros_like(in_mask)
    h, w = in_mask.shape[:2]
    coords = np.where(overlap_mask)
    for r, c in zip(*coords):
        neighbors = [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)]
        for y, x in neighbors:
            if 0 <= x < w and 0 <= y < h and in_mask[y, x] == 1 and overlap_mask[y, x] == 0:
                res[r, c] = 1
                break
    return res

def rgb_dist(src, dst):
    return np.linalg.norm((dst - src) / 255., axis=-1)

def get_gradient(img):
    img = ImageOps.grayscale(Image.fromarray(img.astype(np.uint8)))
    img = np.array(img)
    def rescale(arr, x_min, x_max):
        i_min = np.min(arr)
        i_max = np.max(arr)
        return x_min + (arr - i_min) * (x_max - x_min) / (i_max - i_min + 1e-5)
    return rescale(np.abs(np.gradient(img, axis=1)), 0, 255) / 255., rescale(np.abs(np.gradient(img, axis=0)), 0, 255) / 255.

def calc_flow(src, dst, src_cons, dst_cons, overlap):
    h, w = src.shape[:-1]
    G = maxflow.Graph[float](w * h, 2 * ((w - 1) * (h - 1) * 2 + w + h - 2))
    G_indices = G.add_grid_nodes((h, w))
    c_dif = rgb_dist(src, dst)
    s_grad_x, s_grad_y = get_gradient(src)
    d_grad_x, d_grad_y = get_gradient(dst)

    for j, i in zip(*np.where(overlap)):
        s = G_indices[j, i]
        if j + 1 < h and overlap[j + 1, i]: # down
            t = G_indices[j + 1, i]
            M = (c_dif[j, i] + c_dif[j + 1, i] + 1e-5) / (s_grad_y[j, i] + s_grad_y[j + 1, i] + d_grad_y[j, i] + d_grad_y[j + 1, i] + 1.)
            G.add_edge(s, t, M, M)
        if i + 1 < w and overlap[j, i + 1]: # right
            t = G_indices[j, i + 1]
            M = (c_dif[j, i] + c_dif[j, i + 1] + 1e-5) / (s_grad_x[j, i] + s_grad_x[j, i + 1] + d_grad_x[j, i] + d_grad_x[j, i + 1] + 1.)
            G.add_edge(s, t, M, M)
        if src_cons[j, i] and not dst_cons[j, i]:
            G.add_tedge(s, INFTY, 0.0)
        elif not src_cons[j, i] and dst_cons[j, i]:
            G.add_tedge(s, 0.0, INFTY)
    flow = G.maxflow()
    print(f"Maxflow: {flow}")
    return G, G_indices


def output(src, dst, src_mask, dst_mask, overlap, G, G_indices):
    out = np.zeros_like(src)
    src_only = np.tile((src_mask & ~overlap)[..., None], [1, 1, 3])
    dst_only = np.tile((dst_mask & ~overlap)[..., None], [1, 1, 3])
    out[src_only] = src[src_only]
    out[dst_only] = dst[dst_only]

    src_mask = src_mask & ~overlap
    dst_mask = dst_mask & ~overlap
    
    SOURCE, SINK = 0, 1
    h, w = src.shape[:-1]
    is_seam = np.zeros_like(src_mask)
    for j, i in zip(*np.where(overlap)):
        s = G_indices[j, i]
        s_seg = G.get_segment(s)
        if s_seg == SOURCE:
            out[j, i] = src[j, i]
            src_mask[j, i] = True
        elif s_seg == SINK:
            out[j, i] = dst[j, i]
            dst_mask[j, i] = True
        if j + 1 < h and overlap[j + 1, i]:
            t = G_indices[j + 1, i]
            t_seg = G.get_segment(t)
            if s_seg != t_seg:
                is_seam[j:j+2, i] = 1
        if i + 1 < w and overlap[j, i + 1]:
            t = G_indices[j, i + 1]
            t_seg = G.get_segment(t)
            if s_seg != t_seg:
                is_seam[j, i:i+2] = 1
    out_seam = out.copy()
    out_seam[is_seam, :] = [0, 255, 255]

    out_poisson = poisson(dst, src, dst_mask, src_mask)

    plt.figure(figsize=(12, 4))
    plt.subplot(2, 3, 1)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(out_seam, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 3)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(out_poisson, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 4)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow((src_mask * 255).astype(np.uint8))
    plt.subplot(2, 3, 5)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow((dst_mask * 255).astype(np.uint8))
    plt.show()
    return out, src_mask, dst_mask



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input', help='Input image dir', type=str, required=True)
    args = parser.parse_args()
    input_dir = Path(args.input)

    def is_num(fname):
        try:
            int(fname.split('.')[0])
        except:
            return False
        return True

    img_files = [file for file in input_dir.iterdir() if is_num(file.name)]
    N = len(img_files)
    ext_name = '.jpg'
    assert N >= 2
    src = cv2.imread(str(input_dir / f'00_trans{ext_name}'))
    src_mask = cv2.imread(str(input_dir / f'00_trans_mask{ext_name}'), cv2.IMREAD_GRAYSCALE)
    assert src is not None and src_mask is not None
    src_mask = src_mask.squeeze() > 0

    for i in range(1, N):
        dst = cv2.imread(str(input_dir / f'{i:02}_trans{ext_name}'))
        dst_mask = cv2.imread(str(input_dir / f'{i:02}_trans_mask{ext_name}'), cv2.IMREAD_GRAYSCALE)
        assert dst is not None and dst_mask is not None
        dst_mask = dst_mask.squeeze() > 0

        overlap_mask = src_mask & dst_mask
        union_mask = src_mask | dst_mask
        src_cons = edge_contraint(src_mask, overlap_mask)
        dst_cons = edge_contraint(dst_mask, overlap_mask)

        G, G_indices = calc_flow(src, dst, src_cons, dst_cons, overlap_mask)
        out = output(src, dst, src_mask, dst_mask, overlap_mask, G, G_indices)[0]
        src = out
        src_mask = union_mask
        

