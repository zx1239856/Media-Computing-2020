import argparse
import cv2
from seam_carving import resize, poisson_resize
try:
    import matplotlib.pyplot as plt
except:
    pass

MASK_THRES = 240

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Seam carve")
    parser.add_argument('--src', help="Source image", required=True)
    parser.add_argument('--op', help="Operation (resize, remove)", default="resize", choices=["resize", "remove"])
    parser.add_argument('--energy', help="Energy type", default="forward", choices=["forward", "backward", "hog"])
    parser.add_argument('--order', help="Operation order", default="width_first", choices=["width_first", "height_first", "optimal"])
    parser.add_argument('--mask', help="Protective mask of the image", default="")
    parser.add_argument('--rm_mask', help="Mask of the part to remove (only in REMOVE op)", default="")
    parser.add_argument('--out', help="Output file (optional)", default="")
    parser.add_argument('--vis', help="Visualization", action="store_true", default=False)

    args = parser.parse_args()

    print(f"Using {args.energy} energy")

    src = cv2.imread(args.src)
    assert src is not None
    mask = cv2.imread(args.mask, 0) if args.mask else None
    if mask is not None:
        mask = mask > MASK_THRES

    if args.op == 'resize':
        print(f"Current image size: H {src.shape[0]}, W {src.shape[1]}")
        d_height = int(input("Please input a new height: "))
        d_width = int(input("Please input a new width: "))
        output = resize(src, (d_height, d_width), args.energy, mask, args.order)
    else:
        rm_mask = cv2.imread(args.rm_mask, 0)
        assert rm_mask is not None
        rm_mask = rm_mask > MASK_THRES
        output = None
    if args.out:
        cv2.imwrite(args.out, output)
    if args.vis and 'plt' in globals():
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 1, 1)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        # plt.subplot(1, 2, 2)
        # plt.tight_layout()
        # plt.axis('off')
        # plt.imshow(output.astype(np.uint8))
        plt.show()
