import argparse, io, json, random, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
REPO = 'https://huggingface.co/datasets/hassanjbara/AI4MARS'

def prepare(seed, size):
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    target = out / f'subset_{seed}_{size}.npz'
    if target.exists(): return target
    info = requests.get('https://huggingface.co/api/datasets/hassanjbara/AI4MARS', timeout=60)
    info.raise_for_status(); revision = info.json()['sha']
    rng = random.Random(seed)
    cache = ROOT / 'cache'; cache.mkdir(exist_ok=True)
    def download(i):
        name = f'train-{i:05d}-of-00013.parquet'; path = cache / name
        if not path.exists():
            for attempt in range(5):
                try:
                    with requests.get(f'{REPO}/resolve/{revision}/data/{name}', stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with path.with_suffix('.part').open('wb') as f:
                            for chunk in r.iter_content(1024*1024): f.write(chunk)
                    pq.ParquetFile(path.with_suffix('.part'))
                    path.with_suffix('.part').replace(path); break
                except Exception:
                    if attempt == 4: raise
                    time.sleep(2 ** attempt)
        print(f'Downloaded shard {i+1}/13', flush=True)
        return path
    with ThreadPoolExecutor(max_workers=3) as pool: paths = list(pool.map(download, range(13)))
    flags = [flag for path in paths for flag in pq.read_table(path, columns=['has_labels']).column(0).to_pylist()]
    eligible = []; row_id = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['label_mask','has_labels']):
            for row in batch.to_pylist():
                if row['has_labels']:
                    mask = np.array(Image.open(io.BytesIO(row['label_mask']['bytes'])).resize((size,size), Image.Resampling.NEAREST))
                    if np.any(mask != 255): eligible.append(row_id)
                row_id += 1
        print(f'Checked label validity: {row_id}/{len(flags)}', flush=True)
    chosen = sorted(rng.sample(eligible, 1813)); selected = set(chosen)
    print(f'Sampling 1813 from {len(eligible)} labeled rows ({len(flags)} total)', flush=True)
    xs, ys, ids = [], [], []; offset = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image','label_mask','has_labels']):
            for row in batch.to_pylist():
                idx = offset; offset += 1
                if idx not in selected: continue
                if not row['has_labels']: raise ValueError(f'Selected row {idx} lacks labels')
                x = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                y = np.array(Image.open(io.BytesIO(row['label_mask']['bytes'])))
                if y.ndim == 3:
                    assert np.all(y[..., :3] == y[..., :1]), 'Unexpected mask colors'
                    y = y[..., 0]
                assert set(np.unique(y)).issubset({0,1,2,3,255})
                y = np.array(Image.fromarray(y).resize((size,size), Image.Resampling.NEAREST))
                assert np.any(y != 255), f'No valid pixels in row {idx}'
                xs.append(np.array(x.resize((size,size), Image.Resampling.BILINEAR)))
                ys.append(y); ids.append(idx)
        print(f'Extracted {len(ids)}/1813 images', flush=True)
    assert offset == 18130 and len(ids) == 1813
    np.savez_compressed(target, images=np.stack(xs), masks=np.stack(ys), indices=ids)
    (out / 'selection.json').write_text(json.dumps({'repository':REPO,'revision':revision,'seed':seed,'total_rows':len(flags),'eligible_rows':len(eligible),'sampling':'1813 uniformly sampled labeled rows, 10% of original split size','indices':ids}, indent=2))
    return target

class Terrain(Dataset):
    def __init__(self, data, indices, augment=False):
        self.x=data['images'][indices]; self.y=data['masks'][indices]; self.augment=augment
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        x=self.x[i]; y=self.y[i]
        if self.augment and random.random()<0.5: x=x[:,::-1]; y=y[:,::-1]
        return torch.from_numpy(x.copy()).permute(2,0,1).float()/255, torch.from_numpy(y.copy()).long()

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths=[16,32,64,128,256]
        self.down=nn.ModuleList([block(a,b) for a,b in zip([3]+widths[:-1],widths)])
        self.up=nn.ModuleList([block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.head=nn.Conv2d(16,4,1)
    def forward(self,x):
        skips=[]
        for i,layer in enumerate(self.down):
            if i: x=nn.functional.max_pool2d(x,2)
            x=layer(x); skips.append(x)
        for layer,skip in zip(self.up,skips[-2::-1]):
            x=nn.functional.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=layer(torch.cat([x,skip],1))
        return self.head(x)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=10); p.add_argument('--size',type=int,default=128); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    data=np.load(prepare(args.seed,args.size)); perm=np.random.default_rng(args.seed).permutation(len(data['indices']))
    val_ids=perm[:182]; train_ids=perm[182:]
    run=ROOT/'run'; run.mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config={**vars(args),'device':str(device),'train_count':len(train_ids),'validation_count':len(val_ids),'classes':['soil','bedrock','sand','big rock'],'ignore_index':255,'train_indices':data['indices'][train_ids].tolist(),'validation_indices':data['indices'][val_ids].tolist()}
    (run/'config.json').write_text(json.dumps(config,indent=2))
    train=DataLoader(Terrain(data,train_ids,True),batch_size=args.batch_size,shuffle=True)
    val=DataLoader(Terrain(data,val_ids),batch_size=args.batch_size)
    model=UNet().to(device); optim=torch.optim.AdamW(model.parameters(),lr=1e-3)
    criterion=nn.CrossEntropyLoss(ignore_index=255); history=[]; best=-1
    print(f'Training on {device}: {len(train_ids)} train, {len(val_ids)} validation',flush=True)
    for epoch in range(1,args.epochs+1):
        start=time.time(); model.train(); loss_sum=0; pixels=0
        for step,(x,y) in enumerate(train):
            x,y=x.to(device),y.to(device); optim.zero_grad(); pred=model(x); loss=criterion(pred,y); loss.backward(); optim.step()
            n=int((y!=255).sum()); loss_sum+=loss.item()*n; pixels+=n
            if step%25==0: print(f'Epoch {epoch} step {step}/{len(train)} loss={loss.item():.4f}',flush=True)
        model.eval(); cm=torch.zeros(4,4,dtype=torch.int64); vl=0; vp=0
        with torch.no_grad():
            for x,y in val:
                x,y=x.to(device),y.to(device); logits=model(x); n=int((y!=255).sum()); vl+=criterion(logits,y).item()*n; vp+=n
                pred=logits.argmax(1).cpu(); y=y.cpu(); valid=y!=255
                cm+=torch.bincount(4*y[valid]+pred[valid],minlength=16).reshape(4,4)
        c=cm.numpy(); union=c.sum(0)+c.sum(1)-c.diagonal(); iou=np.divide(c.diagonal(),union,out=np.full(4,np.nan),where=union>0)
        metric=float(np.nanmean(iou)); row={'epoch':epoch,'train_loss':loss_sum/pixels,'val_loss':vl/vp,'miou':metric,'iou':[float(v) if np.isfinite(v) else None for v in iou],'pixel_accuracy':float(c.trace()/c.sum()),'seconds':time.time()-start}
        history.append(row); (run/'metrics.json').write_text(json.dumps(history,indent=2)); print(json.dumps(row),flush=True)
        checkpoint={'model':model.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch,'config':config,'metrics':row}
        torch.save(checkpoint,run/'last.pt')
        if metric>best: best=metric; torch.save(checkpoint,run/'best.pt')
    print('Training complete',flush=True)

if __name__=='__main__': main()
