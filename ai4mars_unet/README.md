# AI4MARS U-Net 基线

## 本次训练结果

已在 Apple MPS 上完成 10 轮。最佳模型为第 9 轮：验证集 mIoU **51.88%**、像素准确率 **86.73%**；各类 IoU 为 soil 76.58%、bedrock 87.22%、sand 43.73%、big rock 0%。大石块识别尚未学好，128×128、10 轮仅作基线。最佳权重已经重新加载并通过单图预测检查。

数据来源：https://huggingface.co/datasets/hassanjbara/AI4MARS

官方 train 有 18,130 个样本，其中 2,066 个没有标签；另外 165 个在 128×128 缩放后无有效标签像素。使用 Python 随机种子 42，从 15,899 个有效样本中无放回抽取 1,813 个（原始 train 数量的 10%），再固定分为 1,631 个训练样本和 182 个验证样本。未使用官方测试集；验证结果不等于官方测试成绩。同场景相近图像可能造成验证结果偏乐观。

## 运行

在项目根目录运行：

```sh
.venv/bin/python -u ai4mars_unet/train.py
```

依赖版本记录在 requirements.txt。默认 10 轮、128×128 输入、batch size 16、AdamW（学习率 0.001）、四层下采样 U-Net，初始通道数 16，无预训练权重。优先 CUDA，其次 Apple MPS，再次 CPU。可使用 `--epochs`、`--size`、`--batch-size`、`--seed` 调整配置。

图像双线性缩放到 [0,1]；标签最近邻缩放，类别 0 soil、1 bedrock、2 sand、3 big rock，255 在损失和指标中忽略。训练使用同步水平翻转增强。数据集说明指出标签已经排除了车体和 30 米以外区域。

首次运行下载所有训练分片约 6.1 GB，以保证在完整训练集上随机抽样。仅将选定样本转换为训练数组；重复运行复用缓存。128×128 是本机快速基线，细小石块可能因缩放丢失。

## 输出

- `data/selection.json`：数据集版本与抽样原始行号。
- `data/subset_42_128.npz`：抽样图像和标签。
- `run/config.json`：训练参数与训练/验证行号。
- `run/metrics.json`：每轮损失、各类 IoU、mIoU、像素准确率与耗时。
- `run/best.pt`：验证集 mIoU 最高的模型和优化器状态。
- `run/last.pt`：最后一轮模型和优化器状态。
- `training.log`：本次运行日志。

## 单图预测

```sh
.venv/bin/python ai4mars_unet/predict.py /absolute/path/to/image.jpg --output ai4mars_unet/run/prediction.png
```

输出原图尺寸的 0–3 类别索引 PNG 和 `_color.png` 彩色分割图。预测不会自动屏蔽车体或远距离区域。

再次训练会覆盖 run 中的输出。训练脚本先筛除缺失和空标签，再固定抽样；遇到非法标签值会报错。有效样本池大小和抽样结果保存在 selection.json 中。
