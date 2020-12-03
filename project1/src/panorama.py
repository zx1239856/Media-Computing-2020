import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
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
    parser.add_argument('-output', help='Output image file', type=str)
    args = parser.parse_args()

    import imageio
    trainImg = imageio.imread('http://www.ic.unicamp.br/~helio/imagens_registro/foto1A.jpg')
    queryImg = imageio.imread('http://www.ic.unicamp.br/~helio/imagens_registro/foto1B.jpg')
    kpts_a, feat_a = detect_feature(trainImg, 'orb')
    kpts_b, feat_b = detect_feature(queryImg, 'orb')
    matches = match_keypoints_bf(feat_a, feat_b, 'orb')
    H, status = compute_homography(kpts_a, kpts_b, matches, 4)
    width, height = trainImg.shape[1] + queryImg.shape[1], trainImg.shape[0] + queryImg.shape[0]
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    result = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thres.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    # do mask erosion
    min_rect = mask.copy()
    sub = mask
    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)
        sub = cv2.subtract(min_rect, thres)
    cnts = cv2.findContours(min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    plt.figure(figsize=(20,10))
    plt.imshow(result[y:y+h,x:x+w])
    plt.axis('off')
    plt.show()

