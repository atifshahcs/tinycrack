import argparse, time, os, glob, cv2, numpy as np
import torch
from model import TinyCrackNet

def preprocess(img, size=256):
    im = cv2.resize(img, (size,size))[:,:,::-1].astype(np.float32)/255.0
    im = (im - 0.0)/1.0
    return np.transpose(im, (2,0,1))[None]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', default="D:/ml_projects/tinycrack/outputs/runs/tc1/best.pt")
    ap.add_argument('--onnx')
    ap.add_argument('--test_imgs', default="D:/ml_projects/tinycrack/data/test_images")
    ap.add_argument('--out', default='D:/ml_projects/tinycrack/outputs/predict_samples')
    ap.add_argument('--cpu_bench', action='store_true')
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    paths = []
    subdir = [f for f in os.listdir(args.test_imgs) if os.path.isdir(os.path.join(args.test_imgs, f))]
    
    for sub in subdir:
        sub_path = os.path.join(args.test_imgs, sub)
        for file in os.listdir(sub_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                paths.append(os.path.join(sub_path, file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCrackNet().eval()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    def run(im):
        with torch.no_grad():
            t = torch.from_numpy(im)
            y = model(t).sigmoid().numpy()
        return y

    count=1
    for p in paths:
        img = cv2.imread(p); 
        x = preprocess(img)
        t0=time.time(); 
        y = run(x); 
        # prob = (y[0,0]*255).astype(np.uint8)
        prob = ((1.0 - y[0,0]) * 255).astype(np.uint8) # invert the prob
        
        
        sub_fold_name = os.path.basename(os.path.dirname(p))
        os.makedirs(os.path.join(args.out,sub_fold_name), exist_ok=True)
        cv2.imwrite(os.path.join(args.out, sub_fold_name,os.path.splitext(os.path.basename(p))[0]+'_prob.png'), prob)
        print(f"Processing image {count}/{len(paths)}")
        count+=1

if __name__=='__main__':
    main()
