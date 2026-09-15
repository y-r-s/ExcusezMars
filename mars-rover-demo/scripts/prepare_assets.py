"""Rebuild real DTM crop and measured validation confusion assets."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import torch

PROJECT=Path(__file__).resolve().parents[1]
BASE=PROJECT.parent
OUT=PROJECT/'dist/assets'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE/'ai4mars_unet'))
from train import UNet

def terrain():
    source=BASE/'ai4mars_unet/HiRISE_Gale.IMG'
    with rasterio.open(source) as ds:
        # Read windows spanning exactly 256 m; output has 1 m pixel spacing.
        w,h=256/abs(ds.res[0]),256/abs(ds.res[1])
        candidates=[]
        for r in np.linspace(ds.height*.25,ds.height*.75,9).astype(int):
            for c in np.linspace(ds.width*.25,ds.width*.75,7).astype(int):
                window=Window(int(c),int(r),w,h)
                a=ds.read(1,window=window,out_shape=(256,256),resampling=Resampling.bilinear,masked=True)
                if np.ma.getmaskarray(a).any() or not np.isfinite(a).all() or a.min() < -10000: continue
                gy,gx=np.gradient(a.filled(0)); slope=np.hypot(gx,gy)
                # Gentle, non-flat crop selected for the geometric rover demo.
                score=abs(float(np.median(slope))-.07)+float(np.percentile(slope,95))*.1
                candidates.append((score,window,a.filled(0)))
        if not candidates: raise ValueError('No fully valid crop found')
        _,window,a=min(candidates,key=lambda x:x[0]); lo=float(a.min()); hi=float(a.max())
        normalized=(a-lo)/(hi-lo)
        Image.fromarray(np.round(normalized*255).astype('uint8')).save(OUT/'heightmap.png')
        meta={'product_id':'DTEEC_009149_1750_009294_1750_U01','source_url':'https://www.uahirise.org/PDS/DTM/PSP/ORB_009100_009199/PSP_009149_1750_PSP_009294_1750/DTEEC_009149_1750_009294_1750_U01.IMG','source_page':'https://www.uahirise.org/dtm/dtm.php?ID=PSP_009149_1750','credit':'NASA/JPL-Caltech/University of Arizona','width':256,'height':256,'grid_spacing_m':1,'source_resolution_m':list(ds.res),'window':{'col_off':window.col_off,'row_off':window.row_off,'width':window.width,'height':window.height},'elevation_min_m':lo,'elevation_max_m':hi,'vertical_exaggeration':1,'normalization':'(elevation-min)/(max-min); recover physical meters before rendering','crs':ds.crs.to_string(),'values':np.round(normalized,7).ravel().tolist()}
        (OUT/'terrain.json').write_text(json.dumps(meta,separators=(',',':')))
        print('DTM',meta['window'],lo,hi,flush=True)

def perception():
    torch.set_num_threads(4)
    checkpoint=torch.load(BASE/'ai4mars_unet/run/best.pt',map_location='cpu',weights_only=False)
    config=checkpoint['config']; model=UNet();model.load_state_dict(checkpoint['model']);model.eval()
    data=np.load(BASE/f"ai4mars_unet/data/subset_{config['seed']}_{config['size']}.npz")
    lookup={int(v):i for i,v in enumerate(data['indices'])}; ids=[lookup[i] for i in config['validation_indices']]
    cm=np.zeros((4,4),dtype=np.int64); frames=[]
    palette=np.array([[180,105,65],[113,147,164],[231,191,103],[226,77,66]],dtype=np.uint8)
    with torch.no_grad():
        for start in range(0,len(ids),16):
            batch=ids[start:start+16]; images=data['images'][batch]; masks=data['masks'][batch]
            pred=model(torch.from_numpy(images).permute(0,3,1,2).float()/255).argmax(1).numpy()
            valid=masks!=255
            cm+=np.bincount((4*masks[valid].astype(np.int64)+pred[valid]).ravel(),minlength=16).reshape(4,4)
            for j,idx in enumerate(batch):
                if len(frames)>=10: break
                n=len(frames); image=images[j]; overlay=(.5*image+.5*palette[pred[j]]).astype(np.uint8)
                Image.fromarray(image).resize((384,384)).save(OUT/f'nav-{n}.jpg',quality=92)
                Image.fromarray(overlay).resize((384,384)).save(OUT/f'overlay-{n}.jpg',quality=92)
                frames.append({'row':int(data['indices'][idx]),'image':f'assets/nav-{n}.jpg','overlay':f'assets/overlay-{n}.jpg'})
    counts=cm.sum(1); assert np.all(counts>0)
    union=cm.sum(1)+cm.sum(0)-cm.diagonal(); iou=cm.diagonal()/union
    assert abs(float(iou.mean())-checkpoint['metrics']['miou'])<1e-4
    result={'classes':['Soil','Bedrock','Sand','Big rock'],'counts':cm.tolist(),'probabilities':(cm/counts[:,None]).tolist(),'row_semantics':'true class','column_semantics':'predicted class','support_pixels':counts.tolist(),'checkpoint_epoch':checkpoint['epoch'],'checkpoint_sha256':hashlib.sha256((BASE/'ai4mars_unet/run/best.pt').read_bytes()).hexdigest(),'validation_images':len(ids),'ignore_index':255,'miou':float(iou.mean()),'pixel_accuracy':float(cm.trace()/cm.sum()),'frames':frames,'frame_note':'Real AI4MARS validation images and actual U-Net predictions at 128x128, enlarged for display; not georegistered to the DTM.'}
    (OUT/'perception.json').write_text(json.dumps(result,indent=2)); print('Confusion matrix',cm.tolist(),flush=True)

if __name__=='__main__': terrain(); perception()
