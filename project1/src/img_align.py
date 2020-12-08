import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser

def detect_feature(img, method):
    mappings = {'brisk': cv2.BRISK_create(), 'orb': cv2.ORB_create()}
    try:
        mappings['sift'] = cv2.xfeatures2d.SIFT_create()
        mappings['surf'] = cv2.xfeatures2d.SURF_create()
    except:
        pass
    return mappings[method].detectAndCompute(img, None)

def create_matcher(method, cross_check):
    if method in ['sift', 'surf']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    return bf

def match_keypoints_bf(feat_a, feat_b, method):
    bf = create_matcher(method, cross_check=True)
    matches = bf.match(feat_a, feat_b)
    return sorted(matches, key = lambda x : x.distance)

def compute_homography(kpts_a, kpts_b, matches, reproj_thres):
    k_a = np.float32([kp.pt for kp in kpts_a])
    k_b = np.float32([kp.pt for kp in kpts_b])
    if len(matches) > 4:
        pts_a = np.float32([k_a[m.queryIdx] for m in matches])
        pts_b = np.float32([k_b[m.trainIdx] for m in matches])
        H, status = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thres)
        return H, status
    else:
        raise RuntimeError("At least 4 matches are required to compute Homography")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input', help='Input image dir', type=str, required=True)
    args = parser.parse_args()

    def is_num(fname):
        try:
            int(fname.split('.')[0])
        except:
            return False
        return True

    imread = lambda x : cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)

    img_files = [file for file in Path(args.input).iterdir() if is_num(file.name)]
    assert len(img_files) >= 2
    ext_name = img_files[0].suffix
    print(f"Processing {len(img_files)} imgs")

    transformed = []
    Hs = []

    for i in range(len(img_files) - 1, 0, -1):
        trainImg = imread(str(Path(args.input) / f'{i:02}{ext_name}'))
        queryImg = imread(str(Path(args.input) / f'{i - 1:02}{ext_name}'))
        
        assert trainImg is not None and queryImg is not None

        h_file = Path(args.input) / f'H{i:02}to{i - 1:02}.txt'
        if h_file.exists():
            H = np.loadtxt(h_file.as_posix())
        else:
            kpts_a, feat_a = detect_feature(trainImg, 'orb')
            kpts_b, feat_b = detect_feature(queryImg, 'orb')
            matches = match_keypoints_bf(feat_a, feat_b, 'orb')
            H, status = compute_homography(kpts_a, kpts_b, matches, 4)

            print(f"Computed Homography from {i:02} to {i - 1:02}: \n{H}")
            
        Hs.append(H)
    
    Hs = Hs[::-1]
    
    new_Hs = []
    for i in range(len(Hs)):
        if i == 0:
            new_Hs.append(Hs[0])
        else:
            new_Hs.append(np.matmul(Hs[i], new_Hs[-1]))

    # calc output size
    imgs = []
    height, width = -1, -1
    for i in range(len(img_files)):
        img = imread(str(Path(args.input) / f'{i:02}{ext_name}'))
        if i == 0:
            height, width = img.shape[0], img.shape[1]
        else:
            h, w = img.shape[0], img.shape[1]
            in_pnts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            res = cv2.perspectiveTransform(np.array([in_pnts]), new_Hs[i - 1])
            w, h = np.max(res[0], axis=0)
            height = max(height, h)
            width = max(width, w)
        imgs.append(img)

    height = int(np.ceil(height))
    width = int(np.ceil(width))
    
    result = np.zeros((height, width, imgs[0].shape[-1]), dtype=imgs[0].dtype)
    result[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    transformed = []
    transformed_mask = []
    for i in range(len(img_files)):
        if i == 0:
            trans = result
            mask = np.zeros(result.shape[:-1] + (1,), dtype=result.dtype)
            mask[:imgs[0].shape[0], :imgs[0].shape[1]] = 255
        else:
            trans = cv2.warpPerspective(imgs[i], new_Hs[i - 1], (width, height))
            mask = np.ones(imgs[i].shape[:-1] + (1,), dtype=imgs[i].dtype) * 255
            mask = cv2.warpPerspective(mask, new_Hs[i - 1], (width, height))
        cv2.imwrite(str(Path(args.input) / f'{i:02}_trans.jpg'), cv2.cvtColor(trans, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(args.input) / f'{i:02}_trans_mask.jpg'), mask)
        transformed.append(trans)
    result = np.max(np.array(transformed), axis=0)
    plt.figure(figsize=(20,10))
    plt.imshow(result)
    plt.axis('off')
    plt.show()