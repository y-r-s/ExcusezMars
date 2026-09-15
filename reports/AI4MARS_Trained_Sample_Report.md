# AI4MARS: Trained Data Sample Report

ExcusezMars | 15 September 2026

AI4MARS<br/>Trained Data Sample Report

ExcusezMars  /  15 September 2026  /  Recorded U-Net baseline

**Summary.** A reproducible sample of 1,813 images was selected from the original 18,130-row training split. Of these, 1,631 images were used to fit U-Net and 182 were held out for validation. Best validation mIoU was 51.88%, but big-rock recall and IoU were both 0%.

1. Sampling and data quality

| Stage | Images | Meaning |
| --- | --- | --- |
| Original train split | 18,130 | Hugging Face AI4MARS mirror |
| Missing labels excluded | 2,066 | No segmentation label available |
| Empty resized labels excluded | 165 | No valid pixels at 128 x 128 |
| Eligible sample pool | 15,899 | Available for supervised sampling |
| Selected subset | 1,813 | 10% of original split; 11.40% of eligible pool |
| Training / validation | 1,631 / 182 | Fixed split after sampling |

Sampling used Python random seed 42, without replacement, from eligible rows. A NumPy seed-42 permutation then assigned the split. No original row IDs overlap between training and validation; SHA-256 checks also found no identical resized RGB images across the split. Near-duplicate scenes were not audited.

2. Label distribution in the actual sample

| Class | Train pixels (%) | Val pixels (%) | Train images | Val images |
| --- | --- | --- | --- | --- |
| Soil | 37.71 | 37.69 | 813 | 84 |
| Bedrock | 49.11 | 48.19 | 1,048 | 111 |
| Sand | 12.07 | 13.69 | 530 | 57 |
| Big rock | 1.11 | 0.43 | 207 | 20 |

Percentages use valid pixels only, after resizing. Images can contain multiple classes, so image counts are not mutually exclusive. Unlabeled pixels (255) comprise 43.84% of the entire sample and are excluded from loss and metrics.

Training and error analysis

3. Recorded training configuration

| Component | Configuration |
| --- | --- |
| Network | U-Net; encoder widths 16 / 32 / 64 / 128 / 256; four output classes; no pretrained weights |
| Input processing | 128 x 128 RGB; bilinear image resize; divide intensities by 255; nearest-neighbor mask resize |
| Optimization | AdamW, learning rate 0.001, default weight decay 0.01; batch size 16; 10 epochs |
| Loss and augmentation | Unweighted cross-entropy; ignore_index=255; paired horizontal flips with probability 0.5 |
| Device and selection | Apple MPS; best checkpoint selected by validation mIoU (epoch 9) |

Training loss decreased from 0.984 to 0.414. Best-epoch validation loss was 0.373; final-epoch validation loss was 0.454. Recorded epoch times total 112.2 seconds, excluding data download, preparation and startup. Seeds are recorded, but bitwise reproducibility across devices is not guaranteed.

4. Best-checkpoint validation performance

| Metric | Result |
| --- | --- |
| Mean IoU / pixel accuracy | 51.88% / 86.73% |
| Class IoU: soil / bedrock / sand / big rock | 76.58% / 87.22% / 43.73% / 0.00% |

**Measured confusion matrix:** rows are ground truth; columns are predictions. Counts include only labeled validation pixels.

| True / predicted | Soil | Bedrock | Sand | Big rock |
| --- | --- | --- | --- | --- |
| Soil | 586,904 | 31,267 | 15,904 | 0 |
| Bedrock | 35,962 | 756,359 | 18,334 | 0 |
| Sand | 95,067 | 19,513 | 115,738 | 0 |
| Big rock | 1,328 | 5,777 | 79 | 0 |

5. Why big-rock performance is poor

**Observed failure:** all 7,184 big-rock pixels were missed: 5,777 (80.41%) became bedrock, 1,328 (18.49%) became soil, and 79 (1.10%) became sand. The checkpoint predicts no big-rock pixels at all. Pixel recall and IoU are 0%; precision is undefined because there are no positive predictions. Object-level detection recall was not measured.

**Evidence-backed concern:** training contains 165,903 big-rock pixels versus 7,365,101 bedrock pixels, approximately 44:1. Unweighted cross-entropy gives the rare class relatively little aggregate influence. Only 20 validation images contain big rock, so scene-level generalization remains uncertain.

**Hypotheses to test:** 128 x 128 resizing may remove small rock boundaries; limited training, no pretraining and visual similarity to bedrock may contribute. These explanations are plausible, but have not been isolated by controlled experiments. High overall pixel accuracy does not demonstrate reliable obstacle detection.

Big-rock improvement plan

Prioritized experiments, not claims of achieved improvement. Keep the existing checkpoint and validation split as a baseline.

1 / Increase useful rock evidence

Re-extract the same selected source images at 256 x 256, then evaluate 512 x 512 crops if memory permits. Start with crops containing annotated big-rock pixels, while retaining ordinary terrain and difficult bedrock negatives. Inspect original masks before and after resizing to measure how much rock area is lost. Changing image size in the current script can change eligibility and resample rows, so lock the saved row IDs for a fair resolution comparison.

2 / Balance batches and the loss

Try batches with about 50% rock-positive training crops as an initial experiment, sampling only from the training partition. Keep validation at its natural class distribution. Oversampling alone can overfit the 207 rock-positive training images; use varied crop locations and moderate paired augmentations.

Compare class-weighted cross-entropy against the baseline, with weights estimated from training labels only. A useful starting family is inverse-square-root class frequency with capped ratios. Alternatively test a clearly documented trial vector such as [1, 1, 2, 5] for soil, bedrock, sand and big rock; this is a hyperparameter proposal, not an optimum. PyTorch supports class weights directly [3].

As a separate ablation, test focal loss (for example gamma=2) to reduce the contribution of easy pixels [4]. Its original evidence is for dense detection; effectiveness on this segmentation task must be measured. Do not combine all loss and sampling changes at once. Continue masking label 255 in every loss term.

3 / Train and select for the actual failure

Run a controlled 30-50 epoch trial with a learning-rate schedule and early stopping. If the balanced, higher-resolution model still fails, compare a pretrained encoder. Select a checkpoint using big-rock recall and precision alongside macro mIoU, rather than overall accuracy alone. Set any acceptable false-positive rate before tuning; none has yet been defined for this project.

4 / Validate without leakage

First compare on the same 182-image validation split. Then build an independent scene- or sequence-grouped evaluation with adequate rock-positive images; reserve official expert-labeled test data for final evaluation. Never oversample validation or test data. Report per-class IoU, rock precision/recall, false alarms and image-level bootstrap uncertainty. If object detection matters, define rock instances and a matching rule before reporting object-level recall.

| Experiment | Change from previous reference | Primary question |
| --- | --- | --- |
| A | Weighted loss only | Does the model begin predicting rock? |
| B | Rock-aware sampling only | Does positive exposure improve recall? |
| C | Best A/B + fixed-ID 256 px data | Do retained details improve precision/IoU? |
| D | Longer training; optional pretrained encoder | Does performance generalize across scenes? |

**Recommended next run:** begin with A and B separately, then combine only justified changes. Record three seeds if feasible. Recompute the measured confusion matrix before updating the rover demo; do not manually invent improved error probabilities.

Visual evidence and provenance

Big-rock failure examples from the held-out validation set

These are the three validation images with the largest big-rock pixel counts at 128 x 128, deliberately selected to inspect the failure rather than represent average performance. Red marks annotated big rock; predictions contain none. Ignored areas are hidden in both label views.

Sources and reproducibility

[1] Dataset: <link href="https://huggingface.co/datasets/hassanjbara/AI4MARS" color="#176b78">hassanjbara/AI4MARS on Hugging Face</link>. Revision: 93dbb79de6cd234799750b6129fbfb3b197582fa.

[2] Local evidence: data/selection.json; data/subset_42_128.npz; run/config.json; run/metrics.json; run/best.pt under ai4mars_unet/, plus mars-rover-demo/dist/assets/perception.json. Sample statistics were recalculated from the saved masks. No retraining was performed for this report.

[3] <link href="https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html" color="#176b78">PyTorch CrossEntropyLoss documentation</link>: optional per-class weights and ignored targets.<br/>[4] <link href="https://arxiv.org/abs/1708.02002" color="#176b78">Lin et al. (2017), Focal Loss for Dense Object Detection</link>.

Scope: this report concerns the AI4MARS camera training sample. The HiRISE terrain, synthetic terrain classes and logistics simulation are separate demo inputs and were not used to train this U-Net.
