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

        self._border_size = 32  # for max error calc

        self.output = np.zeros(out_size + (ch,))

        ## visualization params
        self.seam_size = 1

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

    
    def get_seam_max_error(self):
        in_region = np.zeros_like(self._global_nodes.empty)
        in_region[self._border_size : -self._border_size, self._border_size : -self._border_size] = True
        r_max = np.amax(self._global_nodes.right_cost, where=in_region & ~self._global_nodes.empty & self._global_nodes.on_seam_right)
        b_max = np.amax(self._global_nodes.bottom_cost, where=in_region & ~self._global_nodes.empty & self._global_nodes.on_seam_bottom)
        return max(r_max, b_max)
    
    def get_entire_patch_ssd(self):
        """Compute sum-of-squared-differences (accelerated)
        \frac{1}{|A_t|} \sum_{p \in A_t} | I(p) - O(p+t) |^2
        = \sum_p I^2(p) + \sum_p O^2(p + t) - ***2 \sum_p I(p)O(p+t)*** --> 2D correlation
        """
        O = self._global_nodes.color_A / 255.
        I = self._input / 255.
        O_square = (O ** 2).sum(-1)
        O_square = fftconvolve(O_square, np.ones((self._input_h, self._input_w)))
        I_square = (I ** 2).sum(-1)
        I_square = fftconvolve(I_square[::-1, ::-1], ~self._global_nodes.empty)
        Area = fftconvolve(~self._global_nodes.empty, np.ones((self._input_h, self._input_w)))
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
            if -self._input_w + 1 <= x < self._output_w and -self._input_h + 1 <= y < self._output_h:
                cost = COST[y + self._input_h - 1, x + self._input_w - 1]
            return cost
            
        return get_value

    
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
        self.draw_seams(output)
        plt.imshow(output.astype(np.uint8))
        plt.show()
    
    def fill_output(self, step_x, step_y, method, **kwargs):
        assert method in ['random', 'entire_match']
        
        K = kwargs.get('k', 1.0)

        self._patch_cnt = 0
        output = None
        method = 'entire_match'  # TODO:
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
                        output = self._global_nodes.color_A.copy()
                        self.on_output(output)
                    
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
                        output = self._global_nodes.color_A.copy()
                        self.on_output(output)
                    # sample min err
                    ssd = self.get_entire_patch_ssd()
                    err_list = []
                    for j in range(step_y):
                        for i in range(step_x):
                            test_x = x + step_x + i
                            test_y = offset_y - step_y - j
                            err = ssd(test_x, test_y)
                            err_list.append((err, test_x, test_y))
                    sorted(err_list, key=lambda x : x[0])
                    n = np.random.randint(0, len(err_list) * K)
                    x, y = err_list[n][1:]

                    if x >= self._output_w:
                        break
                
                offset_y += step_y
                err_list = []
                for j in range(step_y):
                    for i in range(step_x):
                        test_x = -step_x + i
                        test_y = offset_y - step_y - j
                        err = ssd(test_x, test_y)
                        err_list.append((err, test_x, test_y))
                sorted(err_list, key=lambda x : x[0])
                n = np.random.randint(0, len(err_list) * K)
                x, y = err_list[n][1:]

                if y >= self._output_h:
                    break


        self._is_filled = True

    def insert_patch(self, x, y, sx, sy, is_filling=True, radius=0, tx=0, ty=0, input_x=0, input_y=0, px=0, py=0) -> bool:
        """[summary]

        Args:
            x (int): x coord in output image
            y (int): y coord in output image
            sx (int): patch width
            sy (int): patch height
            tx (int): [description]
            ty (int): [description]
            is_filling (bool, optional): is_filling. Defaults to True.
            px (int, optional): x offset in texture. Defaults to 0.
            py (int, optional): y offset in texture. Defaults to 0.
        """
        v1x, v1y = max(0, x) - x, max(0, y) - y
        v2x, v2y = min(self._output_w, x + sx) - x, min(self._output_h, y + sy) - y

        sx, sy = v2x - v1x, v2y - v1y
        if sx <= 1 or sy <= 1:
            warnings.warn("Patch outsize of output image. Do nothing.")
            return False
        
        B = self._input[v1y + py: v2y + py, v1x + px : v2x + px]
        B_gr_x = self._input_gradient_x[v1y + py : v2y + py, v1x + px : v2x + px]
        B_gr_y = self._input_gradient_y[v1y + py : v2y + py, v1x + px : v2x + px]
        
        x, y = max(0, x), max(0, y)

        non_overlaps = np.count_nonzero(self._global_nodes.empty[y:y+sy, x:x+sx])
        if non_overlaps == sx * sy:
            # no overlap, just fill
            self._global_nodes.empty[y:y+sy, x:x+sx] = False
            self._global_nodes.color_A[y:y+sy, x:x+sx] = B
            return True
        elif non_overlaps == 0 and is_filling:
            # all overlap, but in filling mode (We only allow this case in refinement)
            warnings.warn("Trying to fill on an already filled area")
            return False

        # partially overlap, do as described in the paper
        A = self._global_nodes.color_A[y: y+sy, x:x+sx]
        A_gr_x, A_gr_y = self.get_gradient(A)
        A2 = self._global_nodes.color_B[y: y+sy, x:x+sx]  # A2 is another old patch (if present)
        A2_gr_x, A2_gr_y = self.get_gradient(A2)

        # build graph

        G = maxflow.Graph[float](sx * sy, 2 * ((sx - 1) * (sy - 1) * 2 + sx + sy - 2))
        G.add_nodes(sx * sy)

        color_diff_AB = self.rgb_l1_dist(A, B)
        color_diff_A2B = self.rgb_l1_dist(A2, B)

        seam_nodes = []

        to_graph_idx = lambda i, j : i + j * sx


        for j in range(sy):
            for i in range(sx):
                if i < sx - 1:
                    s, t = to_graph_idx(i, j), to_graph_idx(i + 1, j)
                    if not self._global_nodes[j + y, i + x].empty and not self._global_nodes[j + y, i + x + 1].empty:
                        if self._global_nodes[j + y, i + x].on_seam_right:
                            # old seam present !!
                            ##             New Patch B
                            ##                  |
                            ##  (curr)        cap1
                            ##    ||            |
                            ##     A -- cap2 -- + -- cap3 -- A2     +: the seam node
                            cap1 = self._global_nodes[j + y, i + x].right_cost  # memorized old cut
                            
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
                            self._global_nodes[j + y, i + x].right_cost = cap
                    else:
                        cap = 0.0
                        G.add_edge(s, t, cap, cap)
                        self._global_nodes[j + y, i + x].right_cost = cap
                if j < sy - 1:
                    s, t = to_graph_idx(i, j), to_graph_idx(i, j + 1)
                    if not self._global_nodes[j + y, i + x].empty and not self._global_nodes[j + y + 1, i + x].empty:
                        if self._global_nodes[j + y, i + x].on_seam_bottom:
                            # old seam present, similar to the above codes
                            cap1 = self._global_nodes[j + y, i + x].bottom_cost  # memorized old cut
                            
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
                            self._global_nodes[j + y, i + x].bottom_cost = cap
                    else:
                        cap = 0.0
                        G.add_edge(s, t, cap, cap)
                        self._global_nodes[j + y, i + x].bottom_cost = cap

        ## add seam nodes to graph
        for seam_node in seam_nodes:
            node_idx = G.add_nodes(1)
            seam_node.seam = node_idx
            G.add_edge(seam_node.start, node_idx, seam_node.c2, seam_node.c2)
            G.add_edge(node_idx, seam_node.end, seam_node.c3, seam_node.c3)
            G.add_tedge(node_idx, 0.0, seam_node.c1)
        
        for i in range(sx):
            j = 0
            if not self._global_nodes[y + j, x + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
            j = sy - 1
            if not self._global_nodes[y + j, x + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
        
        for j in range(sy):
            i = 0
            if not self._global_nodes[y + j, x + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)
            i = sx - 1
            if not self._global_nodes[y + j, x + i].empty:
                G.add_tedge(to_graph_idx(i, j), self._INFTY, 0.0)

        if is_filling:
            for j in range(sy):
                for i in range(sx):
                    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for nb in neighbors:
                        ni, nj = nb
                        if 0 <= y + nj < self._output_h and 0 <= x + ni < self._output_w and self._global_nodes[y + nj, x + ni].empty:
                            G.add_tedge(to_graph_idx(i, j), 0.0, self._INFTY)
                            break
        else:
            # refinement case, add edge from center of the new patch B
            out_x, out_y = input_x + tx, input_y + ty
            v1x, v1y = max(0, out_x), max(0, out_y)
            v2x, v2y = min(self._output_w, out_x + radius), min(self._output_h, out_y + radius)
            out_x, out_y = v1x, v1y
            input_x, input_y = out_x - tx, out_y - ty

            for j in range(v2y - v1y):
                for i in range(v2x - v1x):
                    if input_x + i > 0 and input_y + j > 0:
                        G.add_tedge(to_graph_idx(input_x + i, input_y + j), 0.0, self._INFTY)
        
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
                        self._global_nodes[j + y, i + x].on_new_seam = True
                if j < sy - 1:
                    if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i, j + 1)):
                        self._global_nodes[j + y, i + x].on_new_seam = True
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
                            self._global_nodes[j + y, i + x].right_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + y, i + x].set_seam(flow, 'right')
                        else:
                            self._global_nodes[j + y, i + x].bottom_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + y, i + x].set_seam(flow, 'bottom')
                    elif G.get_segment(cur_seam) == SOURCE:
                        # old seam with old cost
                        attr_name = 'c1'
                        if seam_nodes[k].orientation == 'right':
                            self._global_nodes[j + y, i + x].right_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + y, i + x].set_seam(flow, 'right')
                        else:
                            self._global_nodes[j + y, i + x].bottom_cost = getattr(seam_nodes[k], attr_name)
                            self._global_nodes[j + y, i + x].set_seam(flow, 'bottom')
                    k += 1
                else:
                    # new seam
                    if i < sx - 1:
                        if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i + 1, j)):
                            self._global_nodes[j + y, i + x].set_seam(flow, 'right')
                    if j < sy - 1:
                        if G.get_segment(idx) ^ G.get_segment(to_graph_idx(i, j + 1)):
                            self._global_nodes[j + y, i + x].set_seam(flow, 'bottom')

        for j in range(sy):
            for i in range(sx):
                if self._global_nodes[j + y, i + x].empty:
                    self._global_nodes[j + y, i + x].set_color(B[j, i])
                else:
                    if G.get_segment(to_graph_idx(i, j)) == SOURCE:
                        # copy from old_patch
                        self._global_nodes[j + y, i + x].color_B = B[j, i]
                    else:
                        # copy from new patch
                        self._global_nodes[j + y, i + x].color_B = self._global_nodes[j + y, i + x].color_A
                        self._global_nodes[j + y, i + x].set_color(B[j, i])

                        if not self._global_nodes[j + y, i + x].on_new_seam:
                            self._global_nodes[j + y, i + x].on_seam_right = False
                            self._global_nodes[j + y, i + x].on_seam_bottom = False
                
                self._global_nodes[j + y, i + x].on_new_seam = False
                    

        return True


        