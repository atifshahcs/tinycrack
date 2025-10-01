import torch, torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2*(probs*targets).sum(dim=(2,3)) + self.smooth
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.smooth
        dice = 1 - (num/den).mean()
        return 0.5*bce + 0.5*dice
