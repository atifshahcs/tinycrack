import torch, torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2

def conv_bn_relu(in_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,1,1,bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(conv_bn_relu(in_ch+skip_ch, out_ch),
                                  conv_bn_relu(out_ch, out_ch))
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

        chs = [16, 24, 32, 96, 1280]  # approx for α=0.5; tolerate minor variance
        self.dec4 = DecoderBlock(chs[4], chs[3], 256)
        self.dec3 = DecoderBlock(256, chs[2], 128)
        self.dec2 = DecoderBlock(128, chs[1], 64)
        self.dec1 = DecoderBlock(64,  chs[0], 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x  = self.enc5(s4)

        x  = self.dec4(x, s4)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        return self.head(x)
