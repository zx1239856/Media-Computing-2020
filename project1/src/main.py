from argparse import ArgumentParser
from PIL import Image
import numpy as np
from numpy.testing._private.utils import assert_string_equal
from graphcut import GraphCut

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-im', help='Input texture', type=str, required=True)
    parser.add_argument('-h_ratio', help='Ratio of height enlargement', type=float, default=2.0)
    parser.add_argument('-w_ratio', help='Raito of width enlargement', type=float, default=2.0)
    parser.add_argument('-placer', help='Patch placement method', choices=('random', 'entire_match', 'fixed'), type=str, default='entire_match')
    parser.add_argument('-overlap_x', help='Overlap X for fill', type=int, default=-1)
    parser.add_argument('-overlap_y', help='Overlap Y for fill', type=int, default=-1)
    parser.add_argument('-refine', help='Refinement rounds', type=int, default=0)
    parser.add_argument('-radius', help='Refinement radius', type=int, default=3)
    args = parser.parse_args()

    im = np.array(Image.open(args.im).convert('RGB'))
    h, w, ch = im.shape
    assert args.h_ratio > 1 and args.w_ratio > 1
    assert args.refine >= 0
    res_h = int(round(h * args.h_ratio))
    res_w = int(round(w * args.w_ratio))
    blender = GraphCut(im, (res_h, res_w))

    step_x, step_y = w // 3, h // 3
    if args.overlap_x > 0 and args.overlap_y > 0:
        step_x = w - args.overlap_x
        step_y = h - args.overlap_y
        assert step_x > 0 and step_y > 0

    blender.fill_output(step_x, step_y, args.placer)

    cnt = args.refine
    while cnt > 0:
        cnt -= 1
        blender.refinement(args.radius)
    blender.on_output(blender.output)

