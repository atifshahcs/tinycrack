import argparse, os, glob, cv2, numpy as np
from skimage.morphology import skeletonize

def mask_metrics(mask):
    area = float(mask.sum()) / mask.size  # fraction
    sk = skeletonize(mask>0).astype(np.uint8)
    length_px = sk.sum()
    return area, length_px, sk*255

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)         # folder of probability maps or logits as png
    ap.add_argument('--metric_csv', required=True)
    ap.add_argument('--thr', type=float, default=0.5)
    args=ap.parse_args()

    import csv
    with open(args.metric_csv,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['name','area_frac','length_px'])
        for p in sorted(glob.glob(os.path.join(args.preds,'*.png'))):
            prob = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if prob.max()>1.5: prob/=255.0
            m=(prob>=args.thr).astype(np.uint8)
            area,length,sk = mask_metrics(m)
            w.writerow([os.path.basename(p), f"{area:.6f}", int(length)])
            cv2.imwrite(p.replace('.png','_mask.png'), m*255)
            cv2.imwrite(p.replace('.png','_skel.png'), sk)

if __name__=='__main__':
    main()
