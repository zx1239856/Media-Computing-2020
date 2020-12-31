import argparse
import cv2
import numpy as np
from seam_carving import resize, remove_object, poisson_wrapper
try:
    import matplotlib.pyplot as plt
except:
    pass

MASK_THRES = 240

def face_detection(img):
    import face_recognition
    mask = np.zeros(img.shape[:2], dtype=np.bool)
    face_locs = face_recognition.face_locations(img, number_of_times_to_upsample=0, model='cnn')
    print(f'Found {len(face_locs)} face(s) in the img')
    for idx, face in enumerate(face_locs):
        top, right, bottom, left = face
        mask[top:bottom, left:right] = True
        print(f"Face {idx}: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Seam carve")
    parser.add_argument('--src', help="Source image", required=True)
    parser.add_argument('--op', help="Operation (resize, remove)", default="resize", choices=["resize", "remove"])
    parser.add_argument('--energy', help="Energy type", default="forward", choices=["forward", "backward", "hog", "entropy"])
    parser.add_argument('--order', help="Operation order", default="width_first", choices=["width_first", "height_first", "optimal"])
    parser.add_argument('--keep', help="Protective mask of the image", default="")
    parser.add_argument('--remove', help="Mask of the part to remove (only in REMOVE op)", default="")
    parser.add_argument('--out', help="Output file (optional)", default="")
    parser.add_argument('--vis', help="Visualization", action="store_true", default=False)
    parser.add_argument('--poisson', help="Carve in gradient domain using Poisson Solver", action="store_true", default=False)
    parser.add_argument('--face', help="Use face detector", action="store_true", default=False)

    args = parser.parse_args()

    if args.out == "":
        args.vis = True
    assert not args.face or not args.keep, "Face detector and protective mask cannot be used simultaneously"

    print(f"Using {args.energy} energy")

    src = cv2.imread(args.src)
    assert src is not None
    keep = cv2.imread(args.keep, 0) if args.keep else None
    if keep is not None:
        keep = keep > MASK_THRES
    if args.face:
        keep = face_detection(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    op_map = {'resize': resize, 'remove': remove_object}

    resize_func = op_map[args.op] 
    if args.poisson:
        resize_func = poisson_wrapper(resize_func)

    if args.op == 'resize':
        print(f"Current image size: H {src.shape[0]}, W {src.shape[1]}")
        d_height = int(input("Please input a new height: "))
        d_width = int(input("Please input a new width: "))
        output = resize_func(src, (d_height, d_width), args.energy, keep, args.order)
    else:
        remove = cv2.imread(args.remove, 0)
        assert remove is not None
        remove = remove > MASK_THRES
        output = resize_func(src, args.energy, remove, keep)
    if args.out:
        cv2.imwrite(args.out, output)
    if args.vis and 'plt' in globals():
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 1, 1)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.show()
