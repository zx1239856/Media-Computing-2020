from argparse import ArgumentParser
import argparse
from graphcut import GraphCut
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-patch', help='Path to patch image to be applied', required=True)
    parser.add_argument('-bg', help='Path to background image', required=True)
    args = parser.parse_args()

    im = np.array(Image.open(args.bg).convert('RGB'))
    blender = GraphCut(im, im.shape[:-1])