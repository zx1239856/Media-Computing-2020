from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import numpy as np
import maxflow
import matplotlib.pyplot as plt

INFTY = float('inf')

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
            M = c_dif[j, i] + c_dif[j + 1, i] / (s_grad_y[j, i] + s_grad_y[j + 1, i] + d_grad_y[j, i] + d_grad_y[j + 1, i] + 1.)
            G.add_edge(s, t, M, M)
        if i + 1 < w and overlap[j, i + 1]: # right
            t = G_indices[j, i + 1]
            M = c_dif[j, i] + c_dif[j, i + 1] / (s_grad_x[j, i] + s_grad_x[j, i + 1] + d_grad_x[j, i] + d_grad_x[j, i + 1] + 1.)
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
    SOURCE, SINK = 0, 1
    h, w = src.shape[:-1]
    is_seam = np.zeros_like(src_mask)
    for j, i in zip(*np.where(overlap)):
        s = G_indices[j, i]
        s_seg = G.get_segment(s)
        if s_seg == SOURCE:
            out[j, i] = src[j, i]
        elif s_seg == SINK:
            out[j, i] = dst[j, i]
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
    is_seam_ = np.tile(is_seam[..., None], [1, 1, 3])
    out_ = out.copy()
    out_[is_seam_] = 255

    plt.figure(figsize=(20,10))
    plt.imshow(cv2.cvtColor(out_, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return out



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
        out = output(src, dst, src_mask, dst_mask, overlap_mask, G, G_indices)
        src = out
        src_mask = union_mask
        

