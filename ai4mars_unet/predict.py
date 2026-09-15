"""Predict a single image using the best trained checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train import UNet, ROOT

PALETTE = np.array([[166,112,64],[120,140,160],[235,202,100],[190,65,75]],dtype=np.uint8)

def load_model():
    checkpoint=torch.load(ROOT/'run/best.pt',map_location='cpu',weights_only=False)
    model=UNet(); model.load_state_dict(checkpoint['model']); model.eval()
    return model,checkpoint['config']['size']

def main():
    p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--output',default=str(ROOT/'run/prediction.png')); args=p.parse_args()
    torch.set_num_threads(4); model,size=load_model()
    image=Image.open(args.image).convert('RGB')
    x=torch.from_numpy(np.array(image.resize((size,size),Image.Resampling.BILINEAR))).permute(2,0,1).float()[None]/255
    with torch.no_grad(): labels=model(x).argmax(1)[0].numpy().astype(np.uint8)
    labels=Image.fromarray(labels).resize(image.size,Image.Resampling.NEAREST)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    labels.save(output)
    Image.fromarray(PALETTE[np.array(labels)]).save(output.with_name(output.stem+'_color.png'))
    print(output)

if __name__=='__main__': main()
