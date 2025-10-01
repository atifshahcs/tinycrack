import argparse, torch
from tinycrack.model import TinyCrackNet

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--imgsz', type=int, default=512)
    args=ap.parse_args()
    model = TinyCrackNet().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    x = torch.randn(1,3,args.imgsz,args.imgsz)
    torch.onnx.export(model, x, args.out, input_names=['input'], output_names=['mask'],
                      opset_version=17, dynamic_axes=None)
    print("Saved", args.out)

if __name__=='__main__':
    main()
