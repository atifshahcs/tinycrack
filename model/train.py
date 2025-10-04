import argparse, torch, os
import datetime   ### NEW ###
from torch.utils.data import DataLoader, random_split
from model import TinyCrackNet
from dataset import CrackSegDS
from loss import BCEDiceLoss
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='D:\\ml_projects\\tinycrack\\data\\images')
    ap.add_argument('--masks', default='D:\\ml_projects\\tinycrack\\data\\masks')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=256)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--save', default='outputs/runs/tc1')
    args = ap.parse_args()
    os.makedirs(args.save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = CrackSegDS(args.images, args.masks, args.imgsz, aug=True)

    n_val = max(20, int(0.15*len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TinyCrackNet(alpha=args.alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    best = 1e9; patience, bad=10, 0

    # open log file (clear if exists)  ### NEW ###
    log_file = os.path.join(args.save, "training_log.txt")
    with open(log_file, "w") as f:
        f.write("Training and Validation Loss Log\n")
        f.write("="*50 + "\n")

        # log parameters
        f.write("Run Parameters:\n")
        for k,v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("="*60 + "\n\n")

    for epoch in range(args.epochs):
        model.train()
        tr_loss=0.0
        for x,y in tqdm(dl, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                logits = model(x)
                loss = lossf(logits, y)
            loss.backward(); 
            opt.step()
            tr_loss += loss.item()*x.size(0)
        tr_loss/=len(dl.dataset)

        # validation
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for x,y in vl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += lossf(logits,y).item()*x.size(0)
        va_loss/=len(vl.dataset)

        # timestamp + log string   ### NEW ###
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{now} | Epoch {epoch+1}: train_loss: {tr_loss:.4f}, val_loss: {va_loss:.4f}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        # checkpoint
        if va_loss < best:
            best=va_loss; 
            bad=0
            torch.save(model.state_dict(), os.path.join(args.save,'best.pt'))
        else:
            bad+=1
            if bad>=patience: break

if __name__=='__main__':
    main()
