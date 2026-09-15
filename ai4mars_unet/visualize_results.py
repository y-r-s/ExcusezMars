"""Create an English summary of the completed training run."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ai4mars-matplotlib')
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from predict import load_model, PALETTE
from train import ROOT

torch.set_num_threads(4)
run = ROOT / 'run'
history = json.loads((run / 'metrics.json').read_text())
config = json.loads((run / 'config.json').read_text())
best = max(history, key=lambda row: row['miou'])
data = np.load(ROOT / 'data' / f"subset_{config['seed']}_{config['size']}.npz")
indices = [int(np.flatnonzero(data['indices'] == i)[0]) for i in config['validation_indices'][:3]]
model, _ = load_model()
with torch.no_grad():
    predictions = model(torch.from_numpy(data['images'][indices]).permute(0,3,1,2).float()/255).argmax(1).numpy()

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold', 'axes.labelcolor':'#425066',
                     'text.color':'#17243b', 'axes.edgecolor':'#ced6df', 'figure.facecolor':'#f7f9fc'})
fig = plt.figure(figsize=(13,13))
fig.text(.065,.965,'AI4MARS | U-Net training results',fontsize=24,weight='bold')
fig.text(.065,.939,'10% subset: 1,813 images  •  Train: 1,631  /  Validation: 182  •  128 × 128  •  10 epochs',fontsize=11)
fig.text(.065,.902,f"BEST EPOCH  {best['epoch']}       VALIDATION mIoU  {best['miou']:.2%}       PIXEL ACCURACY  {best['pixel_accuracy']:.2%}",fontsize=14,weight='bold',color='#176b78')
gs=fig.add_gridspec(4,3,left=.065,right=.96,bottom=.10,top=.85,hspace=.48,wspace=.27,height_ratios=[1.05,1,1,1])
epochs=[row['epoch'] for row in history]
ax=fig.add_subplot(gs[0,0])
for key,label,color in [('train_loss','Train','#176b78'),('val_loss','Validation','#dd8742')]:
    ax.plot(epochs,[row[key] for row in history],label=label,color=color,lw=2)
ax.set(title='Loss',xlabel='Epoch',ylabel='Cross-entropy'); ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.15)
ax=fig.add_subplot(gs[0,1]); ax.plot(epochs,[row['miou']*100 for row in history],color='#176b78',lw=2,marker='o',ms=4)
ax.scatter([best['epoch']],[best['miou']*100],color='#dd8742',s=60,zorder=5)
ax.set(title='Validation mIoU',xlabel='Epoch',ylabel='IoU (%)',ylim=(0,65)); ax.grid(alpha=.15)
ax=fig.add_subplot(gs[0,2]); vals=np.array(best['iou'])*100
bars=ax.bar(['Soil','Bedrock','Sand','Big rock'],vals,color=PALETTE/255)
ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=3,fontsize=9)
ax.set(title=f"Class IoU · epoch {best['epoch']}",ylabel='IoU (%)',ylim=(0,103)); ax.tick_params(axis='x',labelsize=9)

def colorize(mask):
    rgb=np.full((*mask.shape,3),225,dtype=np.uint8)
    valid=mask!=255; rgb[valid]=PALETTE[mask[valid]]
    return rgb

for row,(idx,pred) in enumerate(zip(indices,predictions),start=1):
    truth=data['masks'][idx]; masked=pred.copy(); masked[truth==255]=255
    for col,array in enumerate([data['images'][idx],colorize(truth),colorize(masked)]):
        ax=fig.add_subplot(gs[row,col]); ax.imshow(array,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if row==1: ax.set_title(['Input image','Ground truth','Prediction (best model)'][col],pad=10)
        if col==0: ax.set_ylabel(f"Row {data['indices'][idx]}",fontsize=10)
legend=[Patch(facecolor=c/255,label=n) for c,n in zip(PALETTE,['Soil','Bedrock','Sand','Big rock'])]
legend.append(Patch(facecolor=np.array([225]*3)/255,label='Ignored / unlabeled'))
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,frameon=False)
fig.text(.065,.037,'Examples: first 3 validation samples in saved split order. Ignored pixels are hidden in predictions.',fontsize=9,color='#566277')
fig.text(.065,.019,'Validation results only; no official test evaluation. Big rock IoU remains 0%, indicating a baseline limitation.',fontsize=9,color='#566277')
fig.savefig(run/'results_summary.png',dpi=170)
fig.savefig(run/'results_summary.pdf')
print(run/'results_summary.png')
