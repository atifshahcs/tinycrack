import torch, torch.nn as nn
import torch.nn.functional as F

from torchvision.models.mobilenetv2 import mobilenet_v2

def conv_bn_relu(in_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,1,1,bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class DecoderBlock(nn.Module):
    def __init__(self, out_ch):  # <-- only out_ch now
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.LazyConv2d(out_ch, kernel_size=3, stride=1, padding=1, bias=False),  # infers in_channels
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TinyCrackNet(nn.Module):
    def __init__(self, alpha=0.5, num_classes=1):
        super().__init__()
        mb = mobilenet_v2(width_mult=alpha).features
        # collect feature maps for skips
        self.stem = nn.Sequential(*mb[:2])    # 1/2
        self.enc2 = nn.Sequential(*mb[2:4])   # 1/4
        self.enc3 = nn.Sequential(*mb[4:7])   # 1/8
        self.enc4 = nn.Sequential(*mb[7:14])  # 1/16
        self.enc5 = nn.Sequential(*mb[14:])   # 1/32

        self.dec4 = DecoderBlock(256)
        self.dec3 = DecoderBlock(128)
        self.dec2 = DecoderBlock(64)
        self.dec1 = DecoderBlock(32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x  = self.enc5(s4)

        

        # print(f"x before dec4 up: {tuple(x.shape)}")
        # print(f"s4: {tuple(s4.shape)}")

        x  = self.dec4(x, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        
        logits =  self.head(x)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
