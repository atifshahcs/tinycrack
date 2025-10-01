import argparse, torch, os
from torch.utils.data import DataLoader, random_split
from tinycrack.model import TinyCrackNet
from tinycrack.dataset import CrackSegDS
from tinycrack.losses import BCEDiceLoss
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--save', default='outputs/runs/tc1')
    args = ap.parse_args()
    os.makedirs(args.save, exist_ok=True)

    ds = CrackSegDS(args.images, args.masks, args.imgsz, aug=True)
    n_val = max(20, int(0.15*len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TinyCrackNet(alpha=args.alpha).to('cpu')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # CPU AMP still helps memory in PyTorch 2.x

    best = 1e9; patience, bad=10, 0
    for epoch in range(args.epochs):
        model.train(); tr=0.0
        for x,y in tqdm(dl, desc=f"epoch {epoch}"):
            opt.zero_grad()
            with torch.autocast(device_type='cpu', enabled=True, dtype=torch.bfloat16):
                logits = model(x)
                loss = lossf(logits, y)
            loss.backward(); opt.step()
            tr += loss.item()*x.size(0)
        tr/=len(dl.dataset)

        # val
        model.eval(); va=0.0
        with torch.no_grad():
            for x,y in vl:
                logits = model(x); va += lossf(logits,y).item()*x.size(0)
        va/=len(vl.dataset)
        print(f"train {tr:.4f} val {va:.4f}")
        if va < best:
            best=va; bad=0
            torch.save(model.state_dict(), os.path.join(args.save,'best.pt'))
        else:
            bad+=1
            if bad>=patience: break

if __name__=='__main__':
    main()
