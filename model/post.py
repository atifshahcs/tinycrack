import argparse, os, glob, cv2, numpy as np
from skimage.morphology import skeletonize
import csv

def mask_metrics(mask):
    area = float(mask.sum()) / mask.size  # fraction
    sk = skeletonize(mask>0).astype(np.uint8)
    length_px = sk.sum()
    return area, length_px, sk*255

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preds', default='D:/ml_projects/tinycrack/outputs/predict_samples')         # folder of probability maps or logits as png
    ap.add_argument('--metric_csv', default='D:/ml_projects/tinycrack/outputs/post_process_samples')
    ap.add_argument('--thr', type=float, default=0.2)
    args=ap.parse_args()
    os.makedirs(args.metric_csv, exist_ok=True)

    subdir = [f for f in os.listdir(args.preds) if os.path.isdir(os.path.join(args.preds, f))]
    paths = []
    for sub in subdir:
        for file in os.listdir(os.path.join(args.preds,sub)):
                paths.append(os.path.join(args.preds,sub,file))

    with open(os.path.join(args.metric_csv,'metric.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['name','area_frac','length_px'])

        for p in paths:
            prob = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if prob.max()>1.5: 
                 prob/=255.0

            m = (prob >= args.thr).astype(np.uint8) * 255


            # light morphology to de-speckle / bridge tiny gaps
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            m = cv2.dilate(m, k, iterations=1)   # thicken cracks a little
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
            
            area,length,sk = mask_metrics(m)
            w.writerow([os.path.basename(p), f"{area:.6f}", int(length)])
            temp_name = os.path.join(args.metric_csv,os.path.basename(os.path.dirname(p)))
            os.makedirs(temp_name,exist_ok=True)
            cv2.imwrite(os.path.join(temp_name,os.path.basename(p).replace('.png','_mask.png')), m)
            cv2.imwrite(os.path.join(temp_name,os.path.basename(p).replace('.png','_skel.png')), sk)

if __name__=='__main__':
    main()
