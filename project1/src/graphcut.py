from PIL import Image, ImageOps
import numpy as np
import maxflow
import warnings
import matplotlib.pyplot as plt
from numpy import fft
from numpy.lib.utils import deprecate
from scipy.signal import fftconvolve

class AttrCtrl:
    def __init__(self):
        super().__setattr__('attr_list', [])
    
    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        self.attr_list.append(name)

class PixelNodeSet(AttrCtrl):
    """
    Fixed global nodes corresponding to output pixel
    Optimized for parallelization
      x -- right -- o
      |
    bottom
      |
      o
    """
    class Wrapper:
        def __init__(self, parent, x, y):
            super().__setattr__('parent', parent)
            super().__setattr__('x', x)
            super().__setattr__('y', y)

        def __getattr__(self, name: str):
            return getattr(self.parent, name)[self.y, self.x]

        def __setattr__(self, name: str, value):
            getattr(self.parent, name)[self.y, self.x] = value
            
        def reset_seam(self, type=None):
            if type is None:
                self.on_seam_right = False
                self.on_seam_bottom = False
            else:
                assert type in ['right', 'bottom']
                setattr(self, f'on_seam_{type}', False)
        
        def set_color(self, color):
            self.color_A = color
            self.empty = False
        
        def set_seam(self, flow, type='right'):
            assert type in ['right', 'bottom']
            setattr(self, f'on_seam_{type}', True)
            self.max_flow = flow

    def __init__(self, height, width):
        super().__init__()
        self.color_A = np.zeros((height, width, 3))
        self.color_B = np.zeros((height, width, 3))
        self.empty = np.ones((height, width), dtype=np.bool)
        self.right_cost = np.zeros((height, width))
        self.bottom_cost = np.zeros((height, width))
        self.on_seam_right = np.zeros((height, width), dtype=np.bool)
        self.on_seam_bottom = np.zeros((height, width), dtype=np.bool)
        self.max_flow = np.zeros((height, width))
        self.on_new_seam = np.zeros((height, width), dtype=np.bool)  ## whether node is on the newest seam

    def __getitem__(self, idx):
        y, x = idx
        return self.Wrapper(self, x, y)


class SeamNode:
    """
    Node holding all old seams
    """
    def __init__(self, start, end, c1, c2, c3, orientation):
        assert orientation in ['right', 'bottom']
        self.start = start
        self.end = end
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.orientation = orientation
        self.seam = -1

class GraphCut:
    def __init__(self, im, out_size):
        self._input = np.copy(im)
        self._input_gradient_x, self._input_gradient_y = self.get_gradient(im)

        self._input_h, self._input_w, ch = im.shape
        self._output_h, self._output_w = out_size
        self._global_nodes = PixelNodeSet(*out_size)
        self._INFTY = float('inf')
        self._MIN_CAP = 1e-10
        self._patch_cnt = 0
        self._is_filled = False
        self._max_err_node_idx = -1

        # SSD computation step
        self._ssdStep = 3

        self._border_size = 1  # for max error calc

        self._is_refined = np.zeros((self._output_h, self._output_w), dtype=np.bool)

        # self.output = np.zeros(out_size + (ch,))
        self.refine_step = 1

        ## visualization params
        self.seam_size = 1

    @property
    def output(self):
        return self._global_nodes.color_A.copy()

    @staticmethod
    def get_gradient(img):
        img = ImageOps.grayscale(Image.fromarray(img.astype(np.uint8)))
        img = np.array(img)
        def rescale(arr, x_min, x_max):
            i_min = np.min(arr)
            i_max = np.max(arr)
            return x_min + (arr - i_min) * (x_max - x_min) / (i_max - i_min + 1e-5)
        return rescale(np.gradient(img, axis=1), 0, 255) / 255., rescale(np.gradient(img, axis=0), 0, 255) / 255.

    @staticmethod
    def rgb_l1_dist(src, dst):
        return np.linalg.norm((dst - src) / 255., axis=-1)

    
    def get_seam_max_error_idx(self):
        max_err = -1
        x, y = -1, -1
        for j in range(self._border_size, self._output_h - self._border_size):
            for i in range(self._border_size, self._output_w - self._border_size):
                if not self._global_nodes[j, i].empty and not self._is_refined[j, i]:
                    name = None
                    if self._global_nodes[j, i].on_seam_right:
                        name = 'right'
                    elif self._global_nodes[j, i].on_seam_bottom:
                        name = 'bottom'
                    if name is not None:
                        err = getattr(self._global_nodes[j, i], f'{name}_cost')
                        if err > max_err:
                            max_err = err
                            x, y = i, j
        return x, y

    @staticmethod
    def _get_ssd_impl(O, I, O_mask, I_mask, w_lo, w_hi, h_lo, h_hi):
        O_square = (O ** 2).sum(-1)
        O_square = fftconvolve(O_square, ~I_mask)
        I_square = (I ** 2).sum(-1)
        I_square = fftconvolve(I_square[::-1, ::-1], ~O_mask)
        Area = fftconvolve(~O_mask, ~I_mask)
        CR = []
        for ch in range(3):
            CR.append(fftconvolve(O[..., ch], I[::-1, ::-1, ch]))
        CR = np.dstack(CR).sum(-1)

        div_area = Area.copy()
        div_area[div_area < 1e-3] = 1
        COST = (O_square + I_square - 2 * CR) / div_area
        COST[Area < 1e-3] = 0
        
        def get_value(x, y):
            cost = 0
            if w_lo <= x < w_hi and h_lo <= y < h_hi:
                cost = COST[y - h_lo, x - w_lo]
            return cost
            
        return get_value

    
    def get_sub_patch_ssd(self, x, y, size_x, size_y):
        I = self._global_nodes.color_A[y:y+size_y, x:x+size_x] / 255.
        O = self._input / 255.
        I_mask = self._global_nodes.empty[y:y+size_y, x:x+size_x]
        O_mask = np.zeros(O.shape[:-1], dtype=np.bool)
        return self._get_ssd_impl(O, I, O_mask, I_mask, -size_x + 1, self._input_w, -size_y + 1, self._input_h)

    
    def get_entire_patch_ssd(self):
        """Compute sum-of-squared-differences (accelerated)
        \frac{1}{|A_t|} \sum_{p \in A_t} | I(p) - O(p+t) |^2
        = \sum_p I^2(p) + \sum_p O^2(p + t) - ***2 \sum_p I(p)O(p+t)*** --> 2D correlation
        """
        O = self._global_nodes.color_A / 255.
        I = self._input / 255.
        O_mask = self._global_nodes.empty
        I_mask = np.zeros(I.shape[:-1], dtype=np.bool)
        return self._get_ssd_impl(O, I, O_mask, I_mask, -self._input_w + 1, self._output_w, -self._input_h + 1, self._output_h)         
    
    @deprecate
    def get_ssd_slow(self, x, y):
        v1x, v1y = max(0, x) - x, max(0, y) - y
        v2x, v2y = min(self._output_w, x + self._input_w) - x, min(self._output_h, y + self._input_h) - y
        sx, sy = v2x - v1x, v2y - v1y
        x, y = max(0, x), max(0, y)
        cost = 0
        num_pix = 0
        for j, cj in zip(range(v1y, v2y), range(sy)):
            for i, ci in zip(range(v1x, v2x), range(sx)):
                if not self._global_nodes[y + cj, x + ci].empty:
                    cost += (((self._global_nodes[y + cj, x + ci].color_A - self._input[j, i]) / 255.) ** 2).sum()
                    num_pix += 1
        if num_pix == 0:
            return 0
        else:
            return cost / num_pix
            
            
    @deprecate
    def get_ssd_sub_patch_slow(self, x, y, input_x, input_y, size_x, size_y):
        cost = 0
        num_pix = 0
        for j in range(size_y):
            for i in range(size_x):
                if not self._global_nodes[y + j, x + i].empty:
                    cost += (((self._global_nodes[y + j, x + i].color_A - self._input[input_y + j, input_x + i]) / 255.) ** 2).sum()
                    num_pix += 1
        if num_pix == 0:
            return 0
        else:
            return cost / num_pix


    def draw_seams(self, out_img):
        indices = np.where(~self._global_nodes.empty & (self._global_nodes.on_seam_right | self._global_nodes.on_seam_bottom))
        for r, c in zip(*indices):
            for j in range(-self.seam_size, self.seam_size):
                for i in range(-self.seam_size, self.seam_size):
                    rr = r + j
                    cc = c + i
                    if 0 <= rr < self._output_h and 0 <= cc < self._output_w:
                        out_img[rr, cc] = [255, 255, 0]

    def on_output(self, output, is_final_output=False):
        plt.figure(figsize=(32, 10))
        plt.subplot(1, 3, 1)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(output.astype(np.uint8))
        self.draw_seams(output)
        plt.subplot(1, 3, 2)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(output.astype(np.uint8))
        plt.subplot(1, 3, 3)
        plt.tight_layout()
        plt.axis('off')
        err_map = np.zeros((self._output_h, self._output_w))
        for j in range(self._output_h):
            for i in range(self._output_w):
                if ~self._global_nodes[j, i].empty:
                    name = None
                    if self._global_nodes[j, i].on_seam_right:
                        name = 'right'
                    elif self._global_nodes[j, i].on_seam_bottom:
                        name = 'bottom'
                    if name is not None:
                        err = getattr(self._global_nodes[j, i], f'{name}_cost')
                        err_map[j, i] = err
        max_err = np.max(err_map)
        err_map /= max_err if max_err > 1e-2 else 1
        plt.imshow(err_map)
        plt.show()
    
    def fill_output(self, step_x, step_y, method, **kwargs):
        assert method in ['random', 'entire_match', 'fixed']
        
        K = kwargs.get('k', 1.0)

        self._patch_cnt = 0
        output = None
        print("==== Initializing fill ====")
        if method == 'random':
            offset_y = 0
            y = offset_y - (step_y + np.random.randint(0, step_y))
            while True:
                print("--- Fill incoming row ----")
                x = -(step_x + np.random.randint(0, step_x))
                
                while True:
                    print(f"x:{x}, y:{y}")
                    
                    if y < self._output_h and self.insert_patch(x, y, self._input_w, self._input_h):
                        self._patch_cnt += 1
                        print(f"Patch count: {self._patch_cnt}")
                    
                    x += (step_x + np.random.randint(0, step_x))
                    y = offset_y - (step_y + np.random.randint(0, step_y))

                    if x >= self._output_w:
                        break
                
                offset_y += step_y
                y = offset_y - (step_y + np.random.randint(0, step_y))

                if y >= self._output_h:
                    break
        elif method == 'entire_match':
            assert 0 < K <= 1.0
            offset_y = 0
            y = offset_y - (step_y + np.random.randint(0, step_y))
            x = -(step_x + np.random.randint(0, step_x))

            while True:
                print("--- Fill incoming row ----")
                while True:
                    print(f"x:{x}, y:{y}")
                    if y < self._output_h and self.insert_patch(x, y, self._input_w, self._input_h):
                        self._patch_cnt += 1
                        print(f"Patch count: {self._patch_cnt}")
                        # output = self._global_nodes.color_A.copy()
                        # self.on_output(output)
                    # sample min err
                    ssd = self.get_entire_patch_ssd()

                    min_x = x + (step_x + np.random.randint(step_x))
                    min_y = offset_y - (step_y + np.random.randint(step_y))
                    min_err = ssd(min_x, min_y)

                    # err_list = []
                    for j in range(step_y):
                        for i in range(step_x):
                            test_x = x + step_x + i
                            test_y = offset_y - step_y - j
                            err = ssd(test_x, test_y)
                            if err < min_err:
                                min_err = err
                                min_x, min_y = test_x, test_y
                            # err_list.append((err, test_x, test_y))
                    # sorted(err_list, key=lambda x : x[0])
                    # n = np.random.randint(0, len(err_list) * K)
                    # x, y = err_list[n][1:]
                    x, y = min_x, min_y

                    if x >= self._output_w:
                        break
                
                offset_y += step_y

                min_x = -(step_x + np.random.randint(step_x))
                min_y = offset_y - (step_y + np.random.randint(step_y))
                min_err = ssd(min_x, min_y)

                # err_list = []
                for j in range(step_y):
                    for i in range(step_x):
                        test_x = -step_x + i
                        test_y = offset_y - step_y - j
                        err = ssd(test_x, test_y)
                        if err < min_err:
                            min_err = err
                            min_x, min_y = test_x, test_y
                        # err_list.append((err, test_x, test_y))
                # sorted(err_list, key=lambda x : x[0])
                # n = np.random.randint(0, len(err_list) * K)
                # x, y = err_list[n][1:]
                x, y = min_x, min_y

                if y >= self._output_h:
                    break
        elif method == 'fixed':
            for y in range(0, self._output_h, step_y):
                for x in range(0, self._output_w, step_x):
                    print(f"x:{x}, y:{y}")
                    if self.insert_patch(x, y, self._input_w, self._input_h):
                        self._patch_cnt += 1
                        print(f"Patch count: {self._patch_cnt}")
                        output = self._global_nodes.color_A.copy()
                        self.on_output(output)

        self._is_filled = True
        output = self._global_nodes.color_A.copy()
        self.on_output(output)

    def refinement(self, diameter):
        # refine via sub-patch matching
        print("---- Texture refinement ----")

        max_err_x, max_err_y = self.get_seam_max_error_idx()
        if max_err_x != -1 and max_err_y != -1:
            dd = diameter
            print(f"diameter: {dd}")
            min_err = float('inf')
            min_err_i, min_err_j = -1, -1

            x = max_err_x - dd // 2
            y = max_err_y - dd // 2
            v1x, v1y = max(0, x), max(0, y)
            v2x, v2y = min(self._output_w, x + dd), min(self._output_h, y + dd)
            sx, sy = v2x - v1x, v2y - v1y

            print(f"Center for max err: ({max_err_x}, {max_err_y})")

            if sx <= 1 or sy <= 1:
                warnings.warn('Patch for improvement too small')
                return

            ssd = self.get_sub_patch_ssd(v1x, v1y, sx, sy)

            for j in range(sy, self._input_h - sy, self.refine_step):
                for i in range(sx, self._input_w - sx, self.refine_step):
                    err = ssd(i, j)
                    if err < min_err:
                        min_err = err
                        min_err_i = i
                        min_err_j = j
            
            min_err_x = v1x - min_err_i
            min_err_y = v1y - min_err_j
            
            # crop_x, crop_y = max(0, min_err_x), max(0, min_err_y)
            # input_x = max_err_x - dd // 2 - crop_x
            # input_y = max_err_y - dd // 2 - crop_y
            # min_err_x, min_err_y = max(0, min_err_x), max(0, min_err_y)
            print(f"Input ({min_err_j} : {min_err_j + dd}, {min_err_i} : {min_err_i + dd}) applied to output ({min_err_x}, {min_err_y})")
            mask = np.zeros((self._input_h, self._input_w), dtype=np.bool)
            mask[min_err_j: min_err_j + dd, min_err_i: min_err_i + dd] = True
            self.insert_patch(min_err_x, min_err_y, self._input_w, self._input_h, mask)
            self._is_refined[v1y:v2y, v1x:v2x] = True

            self._patch_cnt += 1
            print(f"Patch count: {self._patch_cnt}")


    def insert_patch(self, tx, ty, sx, sy, mask=None) -> bool:
        """[summary]
            Normally, we paste input texture to (tx : tx + sx, ty : ty + sy) of output (modulated by output size, though)
            In refinement case, we need to select a center region from which we add SINK edges in graph (corresponding to pixels that must be copied from B)
            this area is denoted by mask of input patch, translated by (tx, ty) to output
        """
        v1x, v1y = max(0, tx) - tx, max(0, ty) - ty
        v2x, v2y = min(self._output_w, tx + sx) - tx, min(self._output_h, ty + sy) - ty

        sx, sy = v2x - v1x, v2y - v1y
        if sx <= 1 or sy <= 1:
            warnings.warn("Patch outsize of output image. Do nothing.")
            return False
        
        B = self._input[v1y:v2y, v1x:v2x]
        B_gr_x = self._input_gradient_x[v1y:v2y, v1x:v2x]
        B_gr_y = self._input_gradient_y[v1y:v2y, v1x:v2x]

        if mask is not None:
            mask = mask[v1y:v2y, v1x:v2x]
        
        tx, ty = max(0, tx), max(0, ty)

        non_overlaps = np.count_nonzero(self._global_nodes.empty[ty:ty+sy, tx:tx+sx])
        if non_overlaps == sx * sy:
            # no overlap, just fill
            self._global_nodes.empty[ty:ty+sy, tx:tx+sx] = False
            self._global_nodes.color_A[ty:ty+sy, tx:tx+sx] = B
            return True
        elif non_overlaps == 0 and mask is None:
            # all overlap, but in filling mode (We only allow this case in refinement)
            warnings.warn("Trying to fill on an already filled area")
            return False

        # partially overlap, do as described in the paper
        A = self._global_nodes.color_A[ty: ty+sy, tx:tx+sx]
        A_gr_x, A_gr_y = self.get_gradient(A)
        A2 = self._global_nodes.color_B[ty: ty+sy, tx:tx+sx]  # A2 is another old patch (if present)
        A2_gr_x, A2_gr_y = self.get_gradient(A2)

        # build graph

        G = maxflow.Graph[float](sx * sy, 2 * ((sx - 1) * (sy - 1) * 2 + sx + sy - 2))
        G_indices = G.add_grid_nodes((sx, sy))

        color_diff_AB = self.rgb_l1_dist(A, B)
        color_diff_A2B = self.rgb_l1_dist(A2, B)

        seam_nodes = []

        to_graph_idx = lambda i, j : G_indices[i, j]


        for j in range(sy):
            for i in range(sx):
                if i < sx - 1:
                    s, t = to_graph_idx(i, j), to_graph_idx(i + 1, j)
                    if not self._global_nodes[j + ty, i + tx].empty and not self._global_nodes[j + ty, i + tx + 1].empty:
                        if self._global_nodes[j + ty, i + tx].on_seam_right:
                            # old seam present !!
                            ##             New Patch B
                            ##                  |
                            ##  (curr)        cap1
                            ##    ||            |
                            ##     A -- cap2 -- + -- cap3 -- A2     +: the seam node
                            cap1 = self._global_nodes[j + ty, i + tx].right_cost  # memorized old cut
                            
                            grad = A_gr_x[j, i] + A2_gr_x[j, i + 1] + B_gr_x[j, i] + B_gr_x[j, i + 1] + 1.
                            cap2 = (color_diff_AB[j, i] + color_diff_A2B[j, i + 1]) / grad
                            cap2 += self._MIN_CAP

                            grad = A2_gr_x[j, i] + A_gr_x[j, i + 1] + B_gr_x[j, i] + B_gr_x[j, i + 1] + 1.
                            cap3 = (color_diff_A2B[j, i] + color_diff_AB[j, i + 1]) / grad
                            cap3 += self._MIN_CAP

                            seam_nodes.append(SeamNode(s, t, cap1, cap2, cap3, 'right'))
                        else:
                            grad = A_gr_x[j, i] + A_gr_x[j, i + 1] + B_gr_x[j, i] + B_gr_x[j, i + 1] + 1.
                            cap = (color_diff_AB[j, i] + color_diff_AB[j, i + 1]) / grad
                            cap += self._MIN_CAP
                            G.add_edge(s, t, cap, cap)
                            self._global_nodes[j + ty, i + tx].right_cost = cap
                    else:
                        cap = 0.0
                        G.add_edge(s, t, cap, cap)
                        self._global_nodes[j + ty, i + tx].right_cost = cap
                if j < sy - 1:
                    s, t = to_graph_idx(i, j), to_graph_idx(i, j + 1)
                    if not self._global_nodes[j + ty, i + tx].empty and not self._global_nodes[j + ty + 1, i + tx].empty:
                        if self._global_nodes[j + ty, i + tx].on_seam_bottom:
                            # old seam present, similar to the above codes
                            cap1 = self._global_nodes[j + ty, i + tx].bottom_cost  # memorized old cut
                            
                            grad = A_gr_y[j, i] + A2_gr_y[j + 1, i] + B_gr_y[j, i] + B_gr_y[j + 1, i] + 1.
                            cap2 = (color_diff_AB[j, i] + color_diff_A2B[j + 1, i]) / grad
                            cap2 += self._MIN_CAP

                            grad = A2_gr_y[j, i] + A_gr_y[j + 1, i] + B_gr_y[j, i] + B_gr_y[j + 1, i] + 1.
                            cap3 = (color_diff_A2B[j, i] + color_diff_AB[j + 1, i]) / grad
                            cap3 += self._MIN_CAP

                            seam_nodes.append(SeamNode(s, t, cap1, cap2, cap3, 'bottom'))
                        else:
                            grad = A_gr_y[j, i] + A_gr_y[j + 1, i] + B_gr_y[j, i] + B_gr_y[j + 1, i] + 1.
                            cap = (color_diff_AB[j, i] + color_diff_AB[j + 1, i]) / grad
                            cap += self._MIN_CAP
                            G.add_edge(s, t, cap, cap)
                            self._global_nodes[j + ty, i + tx].bottom_cost = cap
                    else:
                        cap = 0.0
                        G.add_edge(s, t, cap, cap)
                        self._global_nodes[j + ty, i + tx].bottom_cost = cap

        ## add seam nodes to graph
        for seam_node in seam_nodes:
            node_idx = G.add_nodes(1)
            seam_node.seam = node_idx
            G.add_edge(seam_node.start, node_idx, seam_node.c2, seam_node.c2)
            G.add_edge(node_idx, seam_node.end, seam_node.c3, seam_node.c3)
            G.add_tedge(node_idx, 0.0, seam_node.c1)
        
        for i in range(sx):
            j = 0
            if not self._global_nodes[ty + j, tx + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
            j = sy - 1
            if not self._global_nodes[ty + j, tx + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
        
        for j in range(sy):
            i = 0
            if not self._global_nodes[ty + j, tx + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
            i = sx - 1
            if not self._global_nodes[ty + j, tx + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)

        if mask is None:
            for j in range(sy):
                for i in range(sx):
                    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for nb in neighbors:
                        ni, nj = nb
                        if 0 <= ty + nj < self._output_h and 0 <= tx + ni < self._output_w and self._global_nodes[ty + nj, tx + ni].empty:
                            G.add_tedge(to_graph_idx(i, j), 0.0, self._INFTY)
                            break
        else:
            # refinement case, add edge from center of the new patch B
            for j, i in zip(*np.nonzero(mask)):
                G.add_tedge(to_graph_idx(i, j), 0.0, self._INFTY)
        
        flow = G.maxflow()

        print(f"Maxflow: {flow}")

        k = 0
        SOURCE = 0
        SINK = 1

        for j in range(sy):
            for i in range(sx):
                idx = to_graph_idx(i, j)
                if i < sx - 1:
                    if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i + 1, j)):
                        self._global_nodes[j + ty, i + tx].on_new_seam = True
                if j < sy - 1:
                    if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i, j + 1)):
                        self._global_nodes[j + ty, i + tx].on_new_seam = True
                if len(seam_nodes) > 0 and k < len(seam_nodes) and idx == seam_nodes[k].start:
                    cur_seam = seam_nodes[k].seam
                    cur_end = seam_nodes[k].end
                    if G.get_segment(idx) ^ G.get_segment(cur_end):
                        # old seam remains with updated cost
                        ## idx == src, end == sink, seam_node == src --> c3
                        ## idx == src, end == sink, seam_node == sink --> c2
                        ## idx == sink, end == src, seam_node == src --> c2
                        ## idx == sink, end == src, seam_node == sink --> c3
                        attr_name = 'c2'
                        if G.get_segment(cur_end) ^ G.get_segment(cur_seam):
                            attr_name = 'c3'
                        if seam_nodes[k].orientation == 'right':
                            self._global_nodes[j + ty, i + tx].right_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'right')
                        else:
                            self._global_nodes[j + ty, i + tx].bottom_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'bottom')
                    elif G.get_segment(cur_seam) == SOURCE:
                        # old seam with old cost
                        attr_name = 'c1'
                        if seam_nodes[k].orientation == 'right':
                            self._global_nodes[j + ty, i + tx].right_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'right')
                        else:
                            self._global_nodes[j + ty, i + tx].bottom_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'bottom')
                    k += 1
                else:
                    # new seam
                    if i < sx - 1:
                        if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i + 1, j)):
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'right')
                    if j < sy - 1:
                        if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i, j + 1)):
                            self._global_nodes[j + ty, i + tx].set_seam(flow, 'bottom')

        for j in range(sy):
            for i in range(sx):
                if self._global_nodes[j + ty, i + tx].empty:
                    self._global_nodes[j + ty, i + tx].set_color(B[j, i])
                else:
                    segment = G.get_segment(to_graph_idx(i, j))
                    if segment == SOURCE:
                        # copy from old_patch
                        self._global_nodes[j + ty, i + tx].color_B = B[j, i]
                    elif segment == SINK:
                        # copy from new patch
                        self._global_nodes[j + ty, i + tx].color_B = self._global_nodes[j + ty, i + tx].color_A
                        self._global_nodes[j + ty, i + tx].set_color(B[j, i])

                        if not self._global_nodes[j + ty, i + tx].on_new_seam:
                            self._global_nodes[j + ty, i + tx].on_seam_right = False
                            self._global_nodes[j + ty, i + tx].on_seam_bottom = False
                
                self._global_nodes[j + ty, i + tx].on_new_seam = False
                    

        return True
