import os
import glob
import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
from pytorch_lightning.loggers import TensorBoardLogger

# ─── IoU hesaplayan fonksiyon ────────────────────────────────────────────
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    pred_flat = (pred_mask > threshold).astype(np.uint8).flatten()
    gt_flat   = (gt_mask   > 0).astype(np.uint8).flatten()
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return 1.0
    if gt_flat.sum() == 0 or pred_flat.sum() == 0:
        return 0.0
    return jaccard_score(gt_flat, pred_flat, zero_division=1)

# ─── 1. PostProcessor ve Model (Callback olmadan) ────────────────────────
post_processor = PostProcessor(
    image_sensitivity=0.5,
    pixel_sensitivity=0.5
)

# PatchCore modelini yapılandırıyoruz
model = Patchcore(
    backbone="wide_resnet50_2",  # ResNet backbone (Default: "wide_resnet50_2")
    layers=["layer2", "layer3"],  # Feature extraction layers (Default: ["layer2", "layer3"])
    #coreset_sampling_ratio=0.1,   # Coreset sampling için oran (Default: 0.1)
    pre_trained=True,             # Pre-trained ağırlıkları kullanma (Default: True)
    post_processor=False,         # Callback olmadan post-processing (Default: True)
    evaluator=True,               # Evaluator callback (Default: True)
    visualizer=True,              # Visualizer callback (Default: True)
    num_neighbors=9,              # Nearest neighbors sayısı (Default: 9)
)

# Modelin post-processor'ını sadece forward kısmına ekliyoruz
model.post_processor = post_processor

logger = TensorBoardLogger("tb_logs", name="patchcore")

# ─── 2. DataModule ───────────────────────────────────────────────────────
datamodule = Folder(
    name="wood_dataset",  # Dataset adı (Default: None)
    root="/content/drive/MyDrive/datasets/wood",  # Root dizini (Default: None)
    normal_dir="train/good",  # Normal sınıfı için klasör (Default: None)
    abnormal_dir="test/defect",  # Anormal sınıfı için klasör (Default: None)
    normal_test_dir="test/good",  # Normal test verisi için klasör (Default: None)
    mask_dir="ground_truth/defect",  # Maskeler için klasör (Default: None)
    
    train_batch_size=16,  # Eğitim batch boyutu (Default: 32)
    eval_batch_size=16,   # Validation/test batch boyutu (Default: 32)
    num_workers=8,        # Data loading için worker sayısı (Default: 8)
    
    # PatchCore ile ilişkili parametreler (varsayılan değerlerle):
    #coreset_sampling_ratio=0.1,  # Coreset sampling oranı (Default: 0.1)
    #num_neighbors=9,             # Nearest neighbors sayısı (Default: 9)
    #pre_trained=True,            # Pre-trained ağırlıkları kullanma (Default: True)
    
    normal_split_ratio=0.2,      # Normal görüntülerin validation/test olarak ayrılma oranı (Default: 0.2)
    test_split_mode="from_dir",  # Test setinin hangi şekilde alınacağı (Default: "from_dir")
    test_split_ratio=0.2,        # Test setinin train veri setinden ayrılacak oranı (Default: 0.2)
    val_split_mode="same_as_test",  # Validation veri seti için test veri setinden ayrılacak oran (Default: "same_as_test")
    val_split_ratio=0.5,        # Validation setinin train veri setinden ayrılacak oranı (Default: 0.5)
    
    # Data augmentation parametreleri (Varsayılan olarak None):
    train_augmentations=None,  # Eğitim için augmentasyon (Default: None)
    val_augmentations=None,    # Validation için augmentasyon (Default: None)
    test_augmentations=None,   # Test için augmentasyon (Default: None)
    augmentations=None,        # Genel augmentasyonlar (Default: None)
    
    seed=None,  # Random seed (Default: None)
)

# ─── 3. Engine (Trainer Wrapper) ────────────────────────────────────────
engine = Engine(
    # The path where the checkpoints will be saved. (Default: "results")
    default_root_dir="/content/anomaly_detect/models",  

    # The number of epochs to train for. (Default: 1)
    max_epochs=1,  # The maximum number of epochs for training
    
    # The minimum number of epochs to train. (Default: 1)
    #min_epochs=5,  # The minimum number of epochs for training
    
    # Accelerator used for training, e.g., "auto", "cpu", "gpu". (Default: "auto")
    accelerator="auto",  # Automatically detects GPU or CPU
    
    # The number of devices (GPUs) to use for training. (Default: 1)
    devices=1,  # Use 1 GPU for training
    
    # Whether to use automatic mixed precision (AMP). (Default: False)
    precision=16,  # 16-bit precision for training
    
    # Whether to use gradient checkpointing to save memory. (Default: False)
    #gradient_checkpointing=False,  # Disabling gradient checkpointing
    
    # The number of workers to use for data loading. (Default: 8)
    #num_workers=8,  # Number of workers for loading data
    
    # A logger to log training information. (Default: None)
    logger=logger,  # TensorBoard Logger for monitoring the training
    
    # Callbacks are added to the model for events like early stopping, model checkpoints, etc. (Default: None)
    callbacks=None,  # No callbacks, but you can add early stopping or checkpoints
    
    # Path to the model weights checkpoint file. (Default: None)
    #ckpt_path="/media/taha/5E42B89C42B87A7B/LINUX/models/eff_ckpt",  # Will not load any pre-trained weights since it's not provided
    
    # Optional argument for saving checkpoints at specified intervals
    enable_checkpointing=True,  # Enable saving checkpoints during training
    
    # Validation interval in terms of epochs. (Default: 1)
    check_val_every_n_epoch=1,  # Validation check after every epoch
    
    # The validation step frequency. (Default: 1)
    val_check_interval=1.0,  # Perform validation after every epoch
    
    # Training progress bar visualization. (Default: True)
    enable_progress_bar=True,  # Enable progress bar
    
    # Whether to use automatic learning rate scheduling. (Default: True)
    #auto_lr_find=True  # Automatically find optimal learning rate
    
)

# ─── 4. Eğitim & Test ───────────────────────────────────────────────────
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)

# ─── 5. Predict & IoU Hesaplama ─────────────────────────────────────────
results = engine.predict(model=model, datamodule=datamodule)
print(f"\nToplam tahmin edilen görüntü sayısı: {len(results)}")

ious = []
skipped_no_mask = 0
skipped_good = 0
skipped_shape = 0

for result in results:
    img_path = result.image_path[0]

    # Normal (good) test görüntülerini atla
    if "/test/good/" in img_path:
        skipped_good += 1
        continue

    # Tahmin maskesini al (zaten normalize + threshold edilmiş)
    pred_mask = result.anomaly_map[0].detach().cpu().numpy()

    # Dosya adı ve GT mask adı
    filename = os.path.basename(img_path)
    mask_filename = filename.replace(".jpg", "_mask.jpg")

    # GT mask yolunu tüm alt klasörlerde ara
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

    # PIL ile grayscale olarak oku (H×W)
    gt_mask = np.array(Image.open(gt_path).convert("L"))

    # Boyut uyumsuzluğu kontrolü
    if pred_mask.shape != gt_mask.shape:
        print(f"Boyut uyumsuzluğu: pred {pred_mask.shape}, gt {gt_mask.shape}")
        skipped_shape += 1
        continue

    # IoU hesapla ve listeye ekle
    iou = compute_iou(pred_mask, gt_mask, threshold=0.5)
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

