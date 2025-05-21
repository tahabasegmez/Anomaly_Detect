import os
import glob
import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image

from anomalib.data import Folder
from anomalib.models import EfficientAd
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

# ─── IoU hesaplayan fonksiyon ────────────────────────────────────────────
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.1) -> float:
    pred_flat = (pred_mask > threshold).astype(np.uint8).flatten()
    gt_flat   = (gt_mask > 0).astype(np.uint8).flatten()
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return 1.0
    if gt_flat.sum() == 0 or pred_flat.sum() == 0:
        return 0.0
    return jaccard_score(gt_flat, pred_flat, zero_division=1)

# ─── 1. PostProcessor ve Model ───────────────────────────────────────────
post_processor = PostProcessor(image_sensitivity=0.5, pixel_sensitivity=0.5)

model = EfficientAd(
    teacher_out_channels=384,
    model_size="medium",
    lr=1e-4,
    post_processor=False
)
model.post_processor = post_processor

# ✅ Metrikleri açık şekilde image & pixel olarak ayır
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

# ✅ Epoch sonunda yazdıran callback
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
    root="/content/drive/MyDrive/datasets/wood",
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

# ─── 3. Engine (Trainer Wrapper) ────────────────────────────────────────
engine = Engine(
    default_root_dir="/content/anomaly_detect/models",
    max_epochs=20,
    accelerator="auto",
    devices=1,
    precision=16,
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

# ✅ Test metriklerini ayrı ve son olarak yazdır

print("\n📊 Final Test Metrics (from callback logs):")
for name, value in engine.trainer.callback_metrics.items():
    if isinstance(value, (float, int)) or (hasattr(value, 'item') and callable(value.item)):
        print(f"  {name}: {value.item():.4f}")

# ─── 5. Predict & IoU Hesaplama ─────────────────────────────────────────
results = engine.predict(model=model, datamodule=datamodule)
print(f"\nToplam tahmin edilen görüntü sayısı: {len(results)}")

ious = []
skipped_no_mask = 0
skipped_good = 0
skipped_shape = 0

for result in results:
    img_path = result.image_path[0]

    if "/test/good/" in img_path:
        skipped_good += 1
        continue

    pred_mask = result.anomaly_map[0].detach().cpu().numpy()
    filename = os.path.basename(img_path)
    mask_filename = filename.replace(".jpg", "_mask.jpg")

    mask_pattern = os.path.join(
        "/content/drive/MyDrive/datasets/wood",
        "ground_truth/defect", "**", mask_filename
    )
    matches = glob.glob(mask_pattern, recursive=True)

    if not matches:
        print(f"GT mask bulunamadı: {mask_pattern}")
        skipped_no_mask += 1
        continue

    gt_path = matches[0]
    gt_mask = np.array(Image.open(gt_path).convert("L"))

    if pred_mask.shape != gt_mask.shape:
        print(f"Boyut uyumsuzluğu: pred {pred_mask.shape}, gt {gt_mask.shape}")
        skipped_shape += 1
        continue

    iou = compute_iou(pred_mask, gt_mask, threshold=0.1)
    print(f"{filename} → IoU: {iou:.4f}")
    ious.append(iou)

# ─── 6. Sonuç Özeti ──────────────────────────────────────────────────────
print(f"\n🧾 Özet:")
print(f"  Toplam tahmin edilen örnek      : {len(results)}")
print(f"  Atlanan good sınıfı görüntüler  : {skipped_good}")
print(f"  Eksik GT mask nedeniyle atlanan : {skipped_no_mask}")
print(f"  Boyut uyumsuzluğu nedeniyle atla: {skipped_shape}")
print(f"  IoU hesaplanan görüntü sayısı   : {len(ious)}")

if ious:
    print(f"\n✅ Ortalama IoU: {np.mean(ious):.4f}")
else:
    print("\n⚠️  Hiçbir defect görüntüsü için IoU hesaplanamadı!")
