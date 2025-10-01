import argparse, time, os, glob, cv2, numpy as np
import torch
from tinycrack.model import TinyCrackNet

def preprocess(img, size=512):
    im = cv2.resize(img, (size,size))[:,:,::-1].astype(np.float32)/255.0
    im = (im - 0.0)/1.0
    return np.transpose(im, (2,0,1))[None]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights')
    ap.add_argument('--onnx')
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', default='outputs/samples')
    ap.add_argument('--cpu_bench', action='store_true')
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = glob.glob(os.path.join(args.images, '*.png'))+glob.glob(os.path.join(args.images, '*.jpg'))

    if args.onnx:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        def run(im):
            return ort_sess.run(None, {'input': im.astype(np.float32)})[0]
    else:
        model = TinyCrackNet().eval()
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        def run(im):
            with torch.no_grad():
                t = torch.from_numpy(im)
                y = model(t).sigmoid().numpy()
            return y

    times=[]
    for p in paths:
        img = cv2.imread(p); x = preprocess(img)
        t0=time.time(); y = run(x); dt=(time.time()-t0)*1000
        times.append(dt)
        prob = (y[0,0]*255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out, os.path.splitext(os.path.basename(p))[0]+'_prob.png'), prob)
    if args.cpu_bench and times:
        print(f"Median CPU time @512x512: {np.median(times):.1f} ms over {len(times)} images")

if __name__=='__main__':
    main()
