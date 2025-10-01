import argparse, cv2, os, glob
import numpy as np
from pathlib import Path

# def auto_mask(im):
#     g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     g = cv2.GaussianBlur(g, (5,5), 0)
#     edges = cv2.Canny(g, 50, 150)
#     # thin cracks: dilate a touch, close small gaps
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     m = cv2.morphologyEx(edges, cv2.MORPH_DILATE, k, iterations=1)
#     m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
#     return (m>0).astype(np.uint8)*255

# def auto_mask(im, min_area=200):
#     g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     g = cv2.GaussianBlur(g, (5,5), 0)
#     edges = cv2.Canny(g, 50, 150)

#     # Thin cracks: dilate a touch, close small gaps
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     m = cv2.morphologyEx(edges, cv2.MORPH_DILATE, k, iterations=1)
#     m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

#     # --- Remove small specks ---
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     mask = np.zeros_like(m)
#     for i in range(1, num_labels):  # skip background
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             mask[labels == i] = 255

#     return mask

def auto_mask(im, min_area=200, min_aspect=2.0):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    edges = cv2.Canny(g, 50, 150)

    # Thin cracks: dilate a touch, close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    m = cv2.morphologyEx(edges, cv2.MORPH_DILATE, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # --- Remove small specks & prefer elongated shapes ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    mask = np.zeros_like(m)
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # aspect ratio check (elongated structures)
        aspect_ratio = max(w, h) / max(1, min(w, h))  # avoid divide by zero

        if area >= min_area and aspect_ratio >= min_aspect:
            mask[labels == i] = 255

    return mask


def main():
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    ap=argparse.ArgumentParser()
    ap.add_argument('--images',  default='D:\\ml_projects\\tinycrack\\data\\images')
    ap.add_argument('--out',  default='D:\\ml_projects\\tinycrack\\data\\masks')
    ap.add_argument('--num', type=int, default=500)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    imgs = []
    subdir = [f for f in os.listdir(args.images) if os.path.isdir(os.path.join(args.images, f))]
    for sub in subdir:
        sub_path = os.path.join(args.images, sub)
        for file in os.listdir(sub_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                imgs.append(os.path.join(sub_path, file))
    # imgs = glob.glob(os.path.join(args.images, '*.png')) + glob.glob(os.path.join(args.images, '*.jpg'))

    for i,p in enumerate(sorted(imgs)):
        im = cv2.imread(p); 
        m = auto_mask(im)
        save_path = os.path.join(args.out,os.path.basename(os.path.dirname(p)))
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, os.path.splitext(os.path.basename(p))[0]+'.png'), m)
        print('processing:', p)

if __name__ == '__main__':
    main()
