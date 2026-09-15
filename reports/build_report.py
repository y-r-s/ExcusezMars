"""Build the English sample report from saved training artifacts and measured statistics."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
stats=json.loads((ROOT/'reports/sample_statistics.json').read_text()); metrics=json.loads((ROOT/'ai4mars_unet/run/metrics.json').read_text());best=max(metrics,key=lambda x:x['miou']);cm=json.loads((ROOT/'mars-rover-demo/dist/assets/perception.json').read_text());selection=json.loads((ROOT/'ai4mars_unet/data/selection.json').read_text())
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#17273a'),spaceAfter=14));styles.add(ParagraphStyle(name='Sub',fontName='Helvetica',fontSize=11,leading=16,textColor=colors.HexColor('#596b7b'),spaceAfter=12));styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=12.5;styles['BodyText'].spaceAfter=6;styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=9,leading=11));styles['Heading2'].fontSize=14;styles['Heading2'].leading=18;styles['Heading2'].textColor=colors.HexColor('#176b78');styles['Heading2'].spaceBefore=12;styles['Heading2'].spaceAfter=8
story=[];md=['# AI4MARS: Trained Data Sample Report\n\nExcusezMars | 15 September 2026\n']
def para(t,style='BodyText'):
 story.append(Paragraph(t,styles[style]));md.append(t.replace('<b>','**').replace('</b>','**')+'\n')
def heading(t):para(t,'Heading2')
def table(rows,widths):
 wrapped=[[Paragraph(escape(str(c)),styles['Cell']) for c in row] for row in rows];t=Table(wrapped,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#aebfca')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce3e9')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,8));md.extend(['| '+' | '.join(map(str,r))+' |' for r in rows[:1]]+['| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows[1:]]+[''])
def page():story.append(PageBreak())
para('AI4MARS<br/>Trained Data Sample Report','TitleCustom');para('ExcusezMars  /  15 September 2026  /  Recorded U-Net baseline','Sub')
para('<b>Summary.</b> A reproducible sample of 1,813 images was selected from the original 18,130-row training split. Of these, 1,631 images were used to fit U-Net and 182 were held out for validation. Best validation mIoU was 51.88%, but big-rock recall and IoU were both 0%.')
heading('1. Sampling and data quality')
table([['Stage','Images','Meaning'],['Original train split','18,130','Hugging Face AI4MARS mirror'],['Missing labels excluded','2,066','No segmentation label available'],['Empty resized labels excluded','165','No valid pixels at 128 x 128'],['Eligible sample pool','15,899','Available for supervised sampling'],['Selected subset','1,813','10% of original split; 11.40% of eligible pool'],['Training / validation','1,631 / 182','Fixed split after sampling']],[162,70,267])
para('Sampling used Python random seed 42, without replacement, from eligible rows. A NumPy seed-42 permutation then assigned the split. No original row IDs overlap between training and validation; SHA-256 checks also found no identical resized RGB images across the split. Near-duplicate scenes were not audited.')
heading('2. Label distribution in the actual sample')
table([['Class','Train pixels (%)','Val pixels (%)','Train images','Val images'],['Soil','37.71','37.69','813','84'],['Bedrock','49.11','48.19','1,048','111'],['Sand','12.07','13.69','530','57'],['Big rock','1.11','0.43','207','20']],[91,110,110,94,94])
para('Percentages use valid pixels only, after resizing. Images can contain multiple classes, so image counts are not mutually exclusive. Unlabeled pixels (255) comprise 43.84% of the entire sample and are excluded from loss and metrics.')
story.append(Image(str(ROOT/'reports/tmp/pdfs/balance.png'),width=410,height=138))
page();para('Training and error analysis','TitleCustom')
heading('3. Recorded training configuration')
table([['Component','Configuration'],['Network','U-Net; encoder widths 16 / 32 / 64 / 128 / 256; four output classes; no pretrained weights'],['Input processing','128 x 128 RGB; bilinear image resize; divide intensities by 255; nearest-neighbor mask resize'],['Optimization','AdamW, learning rate 0.001, default weight decay 0.01; batch size 16; 10 epochs'],['Loss and augmentation','Unweighted cross-entropy; ignore_index=255; paired horizontal flips with probability 0.5'],['Device and selection','Apple MPS; best checkpoint selected by validation mIoU (epoch 9)']],[123,376])
para('Training loss decreased from 0.984 to 0.414. Best-epoch validation loss was 0.373; final-epoch validation loss was 0.454. Recorded epoch times total 112.2 seconds, excluding data download, preparation and startup. Seeds are recorded, but bitwise reproducibility across devices is not guaranteed.')
heading('4. Best-checkpoint validation performance')
table([['Metric','Result'],['Mean IoU / pixel accuracy','51.88% / 86.73%'],['Class IoU: soil / bedrock / sand / big rock','76.58% / 87.22% / 43.73% / 0.00%']],[255,244])
para('<b>Measured confusion matrix:</b> rows are ground truth; columns are predictions. Counts include only labeled validation pixels.')
table([['True / predicted','Soil','Bedrock','Sand','Big rock']]+[[n]+[f'{v:,}' for v in row] for n,row in zip(cm['classes'],cm['counts'])],[119,95,95,95,95])
heading('5. Why big-rock performance is poor')
para('<b>Observed failure:</b> all 7,184 big-rock pixels were missed: 5,777 (80.41%) became bedrock, 1,328 (18.49%) became soil, and 79 (1.10%) became sand. The checkpoint predicts no big-rock pixels at all. Pixel recall and IoU are 0%; precision is undefined because there are no positive predictions. Object-level detection recall was not measured.')
para('<b>Evidence-backed concern:</b> training contains 165,903 big-rock pixels versus 7,365,101 bedrock pixels, approximately 44:1. Unweighted cross-entropy gives the rare class relatively little aggregate influence. Only 20 validation images contain big rock, so scene-level generalization remains uncertain.')
para('<b>Hypotheses to test:</b> 128 x 128 resizing may remove small rock boundaries; limited training, no pretraining and visual similarity to bedrock may contribute. These explanations are plausible, but have not been isolated by controlled experiments. High overall pixel accuracy does not demonstrate reliable obstacle detection.')
page();para('Big-rock improvement plan','TitleCustom');para('Prioritized experiments, not claims of achieved improvement. Keep the existing checkpoint and validation split as a baseline.','Sub')
heading('1 / Increase useful rock evidence')
para('Re-extract the same selected source images at 256 x 256, then evaluate 512 x 512 crops if memory permits. Start with crops containing annotated big-rock pixels, while retaining ordinary terrain and difficult bedrock negatives. Inspect original masks before and after resizing to measure how much rock area is lost. Changing image size in the current script can change eligibility and resample rows, so lock the saved row IDs for a fair resolution comparison.')
heading('2 / Balance batches and the loss')
para('Try batches with about 50% rock-positive training crops as an initial experiment, sampling only from the training partition. Keep validation at its natural class distribution. Oversampling alone can overfit the 207 rock-positive training images; use varied crop locations and moderate paired augmentations.')
para('Compare class-weighted cross-entropy against the baseline, with weights estimated from training labels only. A useful starting family is inverse-square-root class frequency with capped ratios. Alternatively test a clearly documented trial vector such as [1, 1, 2, 5] for soil, bedrock, sand and big rock; this is a hyperparameter proposal, not an optimum. PyTorch supports class weights directly [3].')
para('As a separate ablation, test focal loss (for example gamma=2) to reduce the contribution of easy pixels [4]. Its original evidence is for dense detection; effectiveness on this segmentation task must be measured. Do not combine all loss and sampling changes at once. Continue masking label 255 in every loss term.')
heading('3 / Train and select for the actual failure')
para('Run a controlled 30-50 epoch trial with a learning-rate schedule and early stopping. If the balanced, higher-resolution model still fails, compare a pretrained encoder. Select a checkpoint using big-rock recall and precision alongside macro mIoU, rather than overall accuracy alone. Set any acceptable false-positive rate before tuning; none has yet been defined for this project.')
heading('4 / Validate without leakage')
para('First compare on the same 182-image validation split. Then build an independent scene- or sequence-grouped evaluation with adequate rock-positive images; reserve official expert-labeled test data for final evaluation. Never oversample validation or test data. Report per-class IoU, rock precision/recall, false alarms and image-level bootstrap uncertainty. If object detection matters, define rock instances and a matching rule before reporting object-level recall.')
table([['Experiment','Change from previous reference','Primary question'],['A','Weighted loss only','Does the model begin predicting rock?'],['B','Rock-aware sampling only','Does positive exposure improve recall?'],['C','Best A/B + fixed-ID 256 px data','Do retained details improve precision/IoU?'],['D','Longer training; optional pretrained encoder','Does performance generalize across scenes?']],[65,245,189])
para('<b>Recommended next run:</b> begin with A and B separately, then combine only justified changes. Record three seeds if feasible. Recompute the measured confusion matrix before updating the rover demo; do not manually invent improved error probabilities.')
page();para('Visual evidence and provenance','TitleCustom');para('Big-rock failure examples from the held-out validation set','Sub')
story.append(Image(str(ROOT/'reports/tmp/pdfs/rock_examples.png'),width=400,height=380))
para('These are the three validation images with the largest big-rock pixel counts at 128 x 128, deliberately selected to inspect the failure rather than represent average performance. Red marks annotated big rock; predictions contain none. Ignored areas are hidden in both label views.','BodyText')
heading('Sources and reproducibility')
para('[1] Dataset: <link href="https://huggingface.co/datasets/hassanjbara/AI4MARS" color="#176b78">hassanjbara/AI4MARS on Hugging Face</link>. Revision: '+selection['revision']+'.')
para('[2] Local evidence: data/selection.json; data/subset_42_128.npz; run/config.json; run/metrics.json; run/best.pt under ai4mars_unet/, plus mars-rover-demo/dist/assets/perception.json. Sample statistics were recalculated from the saved masks. No retraining was performed for this report.')
para('[3] <link href="https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html" color="#176b78">PyTorch CrossEntropyLoss documentation</link>: optional per-class weights and ignored targets.<br/>[4] <link href="https://arxiv.org/abs/1708.02002" color="#176b78">Lin et al. (2017), Focal Loss for Dense Object Detection</link>.')
para('Scope: this report concerns the AI4MARS camera training sample. The HiRISE terrain, synthetic terrain classes and logistics simulation are separate demo inputs and were not used to train this U-Net.')
def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#d6dfe7'));c.line(48,39,547,39);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#596b7b'));c.drawString(48,26,'EXCUSEZMARS / AI4MARS SAMPLE AUDIT / 15 SEP 2026');c.drawRightString(547,26,str(doc.page))
SimpleDocTemplate(str(OUT/'AI4MARS_Trained_Sample_Report.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=49,title='AI4MARS Trained Data Sample Report',author='ExcusezMars').build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'reports/AI4MARS_Trained_Sample_Report.md').write_text('\n'.join(md))
print(OUT/'AI4MARS_Trained_Sample_Report.pdf')
