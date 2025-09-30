import argparse, cv2, os, glob
from pathlib import Path


def crop_tile(img, size, stride):
    H,W = img.shape[:2]; crops=[]
    for y in range(0, H-size+1, stride):
        for x in range(0, W-size+1, stride):
            crops.append((x,y,img[y:y+size, x:x+size]))
    return crops

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw',  default='D:\\ml_projects\\tinycrack\\data\\raw')
    ap.add_argument('--out',  default='D:\\ml_projects\\tinycrack\\data\\images')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--max', type=int, default=500)  # per subfolder
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # print("Arg:", args)
    subdirs = [ p for p in Path(args.raw).rglob('*') if p.is_dir() and ("cracked" in p.name.lower()) ]
    for sd in subdirs:
        imgs = list(sd.rglob("*.jpg")) + list(sd.rglob("*.JPG"))
        n = 0
        
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            for (x, y, c) in crop_tile(img, args.size, args.stride):
                
                outdir = Path(args.out) / p.parent.parent.name
                
                outdir.mkdir(parents=True, exist_ok=True)
                outp = outdir / f"{p.stem}_{y}_{x}.png"
                cv2.imwrite(str(outp), c)
                n += 1
                if n >= args.max:  # stop for this subfolder
                    break
            if n >= args.max:
                break
            print('processing ', p)

if __name__ == '__main__':

    main()