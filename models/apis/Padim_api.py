import os
import glob
import re
import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image
import csv
from datetime import datetime
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
from anomalib.metrics import (
    F1Score,
    AUROC,
    F1Max,
    Evaluator
)
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback

# ─── 1. PostProcessor ve Model ───────────────────────────────────────────
post_processor = PostProcessor(image_sensitivity=0.5, pixel_sensitivity=0.5)

model = Padim(
    backbone="resnet18",  # Feature extraction backbone
    layers=["layer1", "layer2", "layer3"],  # Layers to extract features from
    pre_trained=True,  # Use pretrained weights
    n_features=100,  # Number of features to retain
)

model.post_processor = post_processor

val_metrics = [
    F1Max(fields=["pred_score", "gt_label"], prefix="image_"),
    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
]

test_metrics = [
    F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
]

model.evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

class PrintMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print("\n📢 Validation Metrics:")
        for metric in pl_module.evaluator.val_metrics:
            value = metric.compute()
            if value is not None:
                name = getattr(metric, "prefix", metric.__class__.__name__).rstrip("_")
                print(f"  {name}: {value.item():.4f}")

logger = TensorBoardLogger("tb_logs", name="efficientad")

# ─── 2. DataModule ───────────────────────────────────────────────────────
datamodule = Folder(
    name="wood_dataset",
    root="/content/anomaly_detect/datasets/wood_dataset",
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    mask_dir="ground_truth/defect",
    train_batch_size=1,
    eval_batch_size=16,
    num_workers=8,
    normal_split_ratio=0.2,
    test_split_mode="from_dir",
    test_split_ratio=0.2,
    val_split_mode="same_as_test",
    val_split_ratio=0.5,
    seed=None,
)

# ─── 3. Engine ───────────────────────────────────────────────────────────
engine = Engine(
    default_root_dir="/content/anomaly_detect/models",
    max_epochs=5,
    accelerator="auto",
    devices=1,
    precision=32,
    logger=logger,
    callbacks=[PrintMetricsCallback()],
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    val_check_interval=1.0,
    enable_progress_bar=True,
)

# ─── 4. Eğitim & Test ───────────────────────────────────────────────────
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)

model_name = "Padim"

print("\n📊 Final Test Metrics (from callback logs):")

# Tarih ve saat bilgisi
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Metrikleri dict olarak topla
metric_dict = {}
for name, value in engine.trainer.callback_metrics.items():
    if isinstance(value, (float, int)) or (hasattr(value, 'item') and callable(value.item)):
        val = value.item()
        print(f"  {name}: {val:.4f}")
        metric_dict[name] = val

# Pixel F1'den tahmini IoU hesapla (IoU = F1 / (2 - F1))
estimated_iou = None
if "pixel_F1Score" in metric_dict:
    pixel_f1 = metric_dict["pixel_F1Score"]
    estimated_iou = pixel_f1 / (2 - pixel_f1)
    metric_dict["Estimated_IoU"] = estimated_iou
    print(f"  Estimated_IoU (from pixel_F1Score): {estimated_iou:.4f}")

# CSV dosyasının tam yolu
csv_path = "/content/anomaly_detect/models/Model_Metrics.csv"

# Eğer dosya yoksa başlık yaz (header control)
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        headers = ["Model", "Timestamp"] + list(metric_dict.keys())
        writer.writerow(headers)

    row = [model_name, timestamp] + [metric_dict.get(m, "") for m in metric_dict.keys()]
    writer.writerow(row)

print(f"\n📁 Test metrikleri şu dosyaya eklendi: {csv_path}")

print(f"\n📁 Test metrikleri şu dosyaya eklendi: {csv_path}")
# ─── 5. Predict & Gerçek IoU Hesaplama ──────────────────────────────────
results = engine.predict(model=model, datamodule=datamodule)

# ─── 7. Pixel-F1 → IoU Tahmini (Regex ile log'dan) ──────────────────────
pixel_f1_value = metric_dict.get("pixel_F1Score", None)

if pixel_f1_value is not None:
    iou_est = pixel_f1_value / (2 - pixel_f1_value)
    print(f"\n🔁 Tahmini IoU (Pixel-F1'den): {iou_est:.4f} (Pixel F1: {pixel_f1_value:.4f})")
else:
    print("\n⚠️ pixel_F1Score değeri bulunamadı, tahmini IoU hesaplanamadı.")