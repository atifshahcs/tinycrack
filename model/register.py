import argparse, cv2, numpy as np

def register_h(img_ref, img_mov):
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_mov, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:500]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H,mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    return H

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--mov', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    r = cv2.imread(args.ref); m = cv2.imread(args.mov)
    H = register_h(r,m)
    out = cv2.warpPerspective(m, H, (r.shape[1], r.shape[0]))
    cv2.imwrite(args.out, out)

if __name__=='__main__':
    main()
