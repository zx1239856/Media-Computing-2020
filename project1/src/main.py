from argparse import ArgumentParser
from PIL import Image
import numpy as np
from graphcut import GraphCut

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-im', help='Input texture', type=str, required=True)
    parser.add_argument('-h_ratio', help='Ratio of height enlargement', type=float, default=2.0)
    parser.add_argument('-w_ratio', help='Raito of width enlargement', type=float, default=2.0)
    parser.add_argument('-placer', help='Patch placement method', choices=('random', 'entire_match'), type=str, default='random')
    args = parser.parse_args()

    im = np.array(Image.open(args.im).convert('RGB'))
    h, w, ch = im.shape
    assert args.h_ratio > 1 and args.w_ratio > 1
    res_h = int(round(h * args.h_ratio))
    res_w = int(round(w * args.w_ratio))
    blender = GraphCut(im, (res_h, res_w))
    blender.fill_output(w // 3, h // 3, args.placer)

