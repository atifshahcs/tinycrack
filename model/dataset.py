import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import glob

def get_transforms(imgsz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(0.1,0.1,p=0.5),
        A.Normalize()
    ])

class CrackSegDS(Dataset):
    def __init__(self, img_dir, mask_dir, imgsz=256, aug=True):

        image_extensions = ('.png', '.jpg', '.jpeg')
        self.paths = []

        subdir = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
        for sub in subdir:
            sub_path = os.path.join(img_dir, sub)
            for file in os.listdir(sub_path):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    self.paths.append(os.path.join(sub_path, file))

        self.img_dir, self.mask_dir, self.imgsz, self.aug = img_dir, mask_dir, imgsz, aug
        self.t = get_transforms(imgsz) if aug else A.Compose([A.Normalize()])

    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, i):
        name = self.paths[i]
        img = cv2.imread(os.path.join(self.img_dir, name))[:,:,::-1]
        mask = cv2.imread(os.path.join(self.mask_dir, os.path.splitext(name)[0]+'.png'), 0)
        if mask is None: mask = np.zeros(img.shape[:2], np.uint8)
        img = cv2.resize(img, (self.imgsz,self.imgsz))
        mask = cv2.resize(mask, (self.imgsz,self.imgsz), interpolation=cv2.INTER_NEAREST)
        a = self.t(image=img, mask=mask)
        img, mask = a['image'].transpose(2,0,1).astype(np.float32), (a['mask']>127).astype(np.float32)[None]
        return torch.from_numpy(img), torch.from_numpy(mask)
