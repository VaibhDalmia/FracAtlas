# FracAtlas Bone Fracture Segmentation Project Progress Tracker

> **Last Updated:** 2026-11-13  
> **Course / Assignment:** Project Assignment 1 — Medical X-ray Image Segmentation  
> **Repository Root:** `D:\FracAtlas`  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4)  
> **Python Environment:** Python 3.12 with PyTorch 2.6.0+cu124, torchvision, OpenCV, pandas, tqdm  

---

## 1. Project Overview & Problem Statement
The objective of this research is to investigate and implement deep image segmentation models for automated bone fracture localization in musculoskeletal X-ray images, as defined in **Problem Statement 1**:
- **Dataset:** FracAtlas (digital X-ray collection with anatomical labels and fracture segmentations).
- **Primary Challenges:**
  1. *Extreme Class Imbalance:* Fracture lines represent $\sim 0.2\% - 0.4\%$ of total pixels, while background and non-fracture tissue occupy $>99.6\%$.
  2. *Low Contrast & Soft-Tissue Occlusion:* Varied exposure, low radiographic contrast between cortical bone and soft tissue, and foreign metal markers ("L", "R", screws).
  3. *Morphological Variation:* Irregular, thin fracture geometries.
- **Initial Baseline Model:** Classic Convolutional Encoder–Decoder **U-Net** (Ronneberger et al.) evaluated using standard clinical metrics (Dice, IoU, Precision, Recall, Accuracy).

---

## 2. Dataset Setup & Why Binary Segmentation Was Adopted
Initially, an attempt was made to pseudo-generate 3-class masks (`Background`, `Soft Tissue`, `Bone`) using edge detection and foundation models (SAM). However:
- FracAtlas **does not provide native ground-truth bone or soft-tissue segmentations**; it exclusively provides radiologist-verified fracture polygon segmentations.
- Heuristic pseudo-mask generation produced artifacts and inconsistent tissue boundaries.
- **Strategic Decision:** In alignment with Problem Statement 1's core objective (computer-aided fracture diagnosis), the project pivoted to **binary fracture segmentation (`0: Background, 1: Fracture`)**, using the official radiologist-verified ground-truth annotations.

### Dataset Splits (Official FracAtlas Fracture Split):
- **Training Set (`Utilities/Fracture Split/train.csv`):** 574 images
- **Validation Set (`Utilities/Fracture Split/valid.csv`):** 82 images
- **Test Set (`Utilities/Fracture Split/test.csv`):** 63 images
- **Total Fractured Scans:** 719 high-resolution images
- **Pre-Rasterized Ground Truth:** All 719 COCO polygons were rasterized losslessly into `fracture_masks/*.png` (`0: Background, 255: Fracture`).

---

## 3. Preprocessing & Data Augmentation Pipeline
Implemented in `fracatlas_dataset.py`:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Tile grid size: $8 \times 8$, clip limit: $2.0$.
   - Enhances local bone boundary contrast and faint radiolucent fracture lines without over-amplifying background noise.
2. **Standardized Resizing:** $256 \times 256$ pixels (configurable).
3. **Intensity Normalization:** $[0, 255] \longrightarrow [0.0, 1.0]$.
4. **Data Augmentations (Training Split):**
   - Random horizontal flips ($p=0.5$).
   - Random vertical flips ($p=0.5$).
   - Random orthogonal 90° rotations ($p=0.5$).

---

## 4. Model Architecture & Loss Formulation
- **Architecture (`unet_model.py`):**
  - Standard U-Net with 4 downsampling stages (DoubleConv blocks: `Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU`), MaxPool downsampling, expansive path with skip connection concatenations, and bilinear upsampling.
  - Total Trainable Parameters: **4,317,825**.
- **Loss Formulation (`train_unet.py`):**
  - Severe class imbalance causes standard Binary Cross-Entropy (BCE) to collapse into predicting all zeros (achieving $99.6\%$ accuracy with $0.0$ Dice score).
  - To eliminate this degeneracy, we employ **Combined BCE + Soft Dice Loss**:
    $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
    where:
    $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$

---

## 5. Artifacts & Deliverables Generated in Workspace
- `fracture_masks/`: 719 rasterized binary ground-truth PNG masks.
- `fracatlas_dataset.py`: PyTorch `Dataset` with CLAHE and augmentation.
- `unet_model.py`: Symmetrical U-Net architecture (4.32M parameters).
- `train_unet.py`: GPU training, validation, auto-checkpointing, and test evaluation script (BCE + Dice).
- `attention_unet.py`: Full Attention U-Net architecture with additive attention gates (7.98M parameters).
- `train_attention_unet.py`: GPU training script with Focal Tversky Loss.
- `train_boundary_unet.py`: GPU training script with combined Focal Tversky + Signed Distance Boundary Loss.
- `FracAtlas_UNet_Fracture_Segmentation.ipynb`: Interactive Jupyter Notebook containing end-to-end code, models, metrics, and visual comparison.
- `Project_Assignment_1_Report.md`: Academic report detailing theory, mathematics, architectures, and experimental protocols.
- `NEXT_STEPS_WORKFLOW.md`: Technical roadmap and ablation decision tree.
- `checkpoints/best_unet.pth`: Saved model weights for Baseline U-Net (256x256).
- `checkpoints/best_attention_unet.pth`: Saved model weights for Attention U-Net (256x256).
- `checkpoints/best_attention_unet_512.pth`: Saved model weights for High-Resolution Attention U-Net (512x512).
- `checkpoints/best_boundary_unet_512.pth`: Saved model weights for Boundary-Aware Attention U-Net (512x512).
- `results/predictions/`: Baseline U-Net (256x256) visual comparison images.
- `results/attention_predictions/`: Attention U-Net (256x256) visual comparison images.
- `results/attention_predictions_512/`: Attention U-Net (512x512) high-resolution visual comparison images.
- `results/boundary_predictions_512/`: Boundary-Aware Attention U-Net (512x512) visual prediction overlays.
- `results/training_curves.png`: Baseline U-Net loss and Dice curves.
- `results/comparison_curves.png`: 4-way comparative performance curves (Dice, Precision, IoU) across all four experimental models.
- `training_log.csv`: Baseline U-Net (256x256) epoch logs.
- `training_log_attention.csv`: Attention U-Net (256x256) epoch logs.
- `training_log_attention_512.csv`: Attention U-Net (512x512) epoch logs.
- `training_log_boundary_512.csv`: Boundary-Aware Attention U-Net (512x512) epoch logs.
- `results/flexray_bone_segmentations/`: 20 multi-class (Raw, FleXray Anatomy, Bone Mask, Multi-Class Composite) 4-panel diagnostic overlays.
- `results/flexray_test_20_summary.csv`: Quantitative evaluation table of bone coverage, containment, and latency across 20 FracAtlas test images.
- `test_flexray_bones.py`: Inference pipeline integrating FleXray for multi-class bone/background segmentation.

---

## 6. Live Training & Experimentation Logs
*(Epoch logs and final test evaluation metrics are dynamically updated below during training)*

### Epoch 01 / 20 Progress Update
- **Train Loss:** `0.6642`
- **Validation Loss:** `0.6038`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0000`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `1.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### Epoch 02 / 20 Progress Update
- **Train Loss:** `0.5685`
- **Validation Loss:** `0.5457`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0000`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `0.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### Epoch 03 / 20 Progress Update
- **Train Loss:** `0.5340`
- **Validation Loss:** `0.5149`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0000`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `1.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### Epoch 04 / 20 Progress Update
- **Train Loss:** `0.5169`
- **Validation Loss:** `0.5122`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0000`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `1.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### Epoch 05 / 20 Progress Update
- **Train Loss:** `0.5045`
- **Validation Loss:** `0.4977`
- **Validation Dice (DSC):** `0.0198` *(Best so far: `0.0198`)*
- **Validation IoU:** `0.0100`
- **Validation Precision:** `0.0511`
- **Validation Recall:** `0.0123`
- **Validation Accuracy:** `0.9945`

### Epoch 06 / 20 Progress Update
- **Train Loss:** `0.4954`
- **Validation Loss:** `0.5265`
- **Validation Dice (DSC):** `0.0747` *(Best so far: `0.0747`)*
- **Validation IoU:** `0.0388`
- **Validation Precision:** `0.0397`
- **Validation Recall:** `0.6302`
- **Validation Accuracy:** `0.9294`

### Epoch 07 / 20 Progress Update
- **Train Loss:** `0.4865`
- **Validation Loss:** `0.5087`
- **Validation Dice (DSC):** `0.0039` *(Best so far: `0.0747`)*
- **Validation IoU:** `0.0019`
- **Validation Precision:** `0.0436`
- **Validation Recall:** `0.0020`
- **Validation Accuracy:** `0.9953`

### Epoch 08 / 20 Progress Update
- **Train Loss:** `0.4794`
- **Validation Loss:** `0.5085`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0747`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `1.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### Epoch 09 / 20 Progress Update
- **Train Loss:** `0.4687`
- **Validation Loss:** `0.5029`
- **Validation Dice (DSC):** `0.0143` *(Best so far: `0.0747`)*
- **Validation IoU:** `0.0072`
- **Validation Precision:** `0.0463`
- **Validation Recall:** `0.0085`
- **Validation Accuracy:** `0.9947`

### Epoch 10 / 20 Progress Update
- **Train Loss:** `0.4624`
- **Validation Loss:** `0.4956`
- **Validation Dice (DSC):** `0.0964` *(Best so far: `0.0964`)*
- **Validation IoU:** `0.0506`
- **Validation Precision:** `0.0565`
- **Validation Recall:** `0.3271`
- **Validation Accuracy:** `0.9722`

### Epoch 11 / 20 Progress Update
- **Train Loss:** `0.4673`
- **Validation Loss:** `0.5115`
- **Validation Dice (DSC):** `0.1133` *(Best so far: `0.1133`)*
- **Validation IoU:** `0.0601`
- **Validation Precision:** `0.0613`
- **Validation Recall:** `0.7523`
- **Validation Accuracy:** `0.9467`

### Epoch 12 / 20 Progress Update
- **Train Loss:** `0.4591`
- **Validation Loss:** `0.4562`
- **Validation Dice (DSC):** `0.1312` *(Best so far: `0.1312`)*
- **Validation IoU:** `0.0702`
- **Validation Precision:** `0.1173`
- **Validation Recall:** `0.1488`
- **Validation Accuracy:** `0.9911`

### Epoch 13 / 20 Progress Update
- **Train Loss:** `0.4461`
- **Validation Loss:** `0.4911`
- **Validation Dice (DSC):** `0.0948` *(Best so far: `0.1312`)*
- **Validation IoU:** `0.0498`
- **Validation Precision:** `0.0566`
- **Validation Recall:** `0.2915`
- **Validation Accuracy:** `0.9748`

### Epoch 14 / 20 Progress Update
- **Train Loss:** `0.4459`
- **Validation Loss:** `0.4535`
- **Validation Dice (DSC):** `0.1691` *(Best so far: `0.1691`)*
- **Validation IoU:** `0.0924`
- **Validation Precision:** `0.1001`
- **Validation Recall:** `0.5455`
- **Validation Accuracy:** `0.9758`

### Epoch 15 / 20 Progress Update
- **Train Loss:** `0.4475`
- **Validation Loss:** `0.4482`
- **Validation Dice (DSC):** `0.1692` *(Best so far: `0.1692`)*
- **Validation IoU:** `0.0924`
- **Validation Precision:** `0.1052`
- **Validation Recall:** `0.4322`
- **Validation Accuracy:** `0.9808`

### Epoch 16 / 20 Progress Update
- **Train Loss:** `0.4352`
- **Validation Loss:** `0.4329`
- **Validation Dice (DSC):** `0.1998` *(Best so far: `0.1998`)*
- **Validation IoU:** `0.1110`
- **Validation Precision:** `0.1499`
- **Validation Recall:** `0.2995`
- **Validation Accuracy:** `0.9891`

### Epoch 17 / 20 Progress Update
- **Train Loss:** `0.4312`
- **Validation Loss:** `0.4406`
- **Validation Dice (DSC):** `0.1702` *(Best so far: `0.1998`)*
- **Validation IoU:** `0.0930`
- **Validation Precision:** `0.1187`
- **Validation Recall:** `0.3003`
- **Validation Accuracy:** `0.9867`

### Epoch 18 / 20 Progress Update
- **Train Loss:** `0.4310`
- **Validation Loss:** `0.4273`
- **Validation Dice (DSC):** `0.1999` *(Best so far: `0.1999`)*
- **Validation IoU:** `0.1110`
- **Validation Precision:** `0.1398`
- **Validation Recall:** `0.3505`
- **Validation Accuracy:** `0.9873`

### Epoch 19 / 20 Progress Update
- **Train Loss:** `0.4274`
- **Validation Loss:** `0.4055`
- **Validation Dice (DSC):** `0.2415` *(Best so far: `0.2415`)*
- **Validation IoU:** `0.1373`
- **Validation Precision:** `0.1633`
- **Validation Recall:** `0.4628`
- **Validation Accuracy:** `0.9868`

### Epoch 20 / 20 Progress Update
- **Train Loss:** `0.4258`
- **Validation Loss:** `0.4612`
- **Validation Dice (DSC):** `0.1134` *(Best so far: `0.2415`)*
- **Validation IoU:** `0.0601`
- **Validation Precision:** `0.1480`
- **Validation Recall:** `0.0919`
- **Validation Accuracy:** `0.9935`

## Final Test Evaluation Results (Unseen 63 Test Images)
| Metric | Score | Clinical Description |
| :--- | :---: | :--- |
| **Dice Similarity Coefficient (DSC)** | **`0.1724`** | Spatial overlap between predicted fracture and ground truth |
| **Intersection over Union (IoU)** | **`0.0943`** | Jaccard index measuring area of overlap over area of union |
| **Precision** | **`0.1092`** | Proportion of predicted fracture pixels that are genuine lesions |
| **Recall (Sensitivity)** | **`0.4085`** | Proportion of true fracture lesions successfully detected |
| **Pixel Accuracy** | **`0.9863`** | Overall pixel classification accuracy |


## 7. Attention U-Net + Focal Tversky Loss Experimentation Logs

### [Attention U-Net] Epoch 01 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9855`
- **Validation Loss:** `0.9848`
- **Validation Dice (DSC):** `0.0428` *(Best so far: `0.0428`)*
- **Validation IoU:** `0.0218`
- **Validation Precision:** `0.0219`
- **Validation Recall:** `0.8785`
- **Validation Accuracy:** `0.8221`

### [Attention U-Net] Epoch 02 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9736`
- **Validation Loss:** `0.9833`
- **Validation Dice (DSC):** `0.0338` *(Best so far: `0.0428`)*
- **Validation IoU:** `0.0172`
- **Validation Precision:** `0.0172`
- **Validation Recall:** `0.9479`
- **Validation Accuracy:** `0.7545`

### [Attention U-Net] Epoch 03 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9554`
- **Validation Loss:** `0.9675`
- **Validation Dice (DSC):** `0.0560` *(Best so far: `0.0560`)*
- **Validation IoU:** `0.0288`
- **Validation Precision:** `0.0293`
- **Validation Recall:** `0.6546`
- **Validation Accuracy:** `0.9002`

### [Attention U-Net] Epoch 04 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9435`
- **Validation Loss:** `0.9982`
- **Validation Dice (DSC):** `0.0000` *(Best so far: `0.0560`)*
- **Validation IoU:** `0.0000`
- **Validation Precision:** `1.0000`
- **Validation Recall:** `0.0000`
- **Validation Accuracy:** `0.9955`

### [Attention U-Net] Epoch 05 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9505`
- **Validation Loss:** `0.9916`
- **Validation Dice (DSC):** `0.0076` *(Best so far: `0.0560`)*
- **Validation IoU:** `0.0038`
- **Validation Precision:** `0.0249`
- **Validation Recall:** `0.0045`
- **Validation Accuracy:** `0.9947`

### [Attention U-Net] Epoch 06 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9538`
- **Validation Loss:** `0.9715`
- **Validation Dice (DSC):** `0.0497` *(Best so far: `0.0560`)*
- **Validation IoU:** `0.0255`
- **Validation Precision:** `0.0325`
- **Validation Recall:** `0.1050`
- **Validation Accuracy:** `0.9818`

### [Attention U-Net] Epoch 07 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9526`
- **Validation Loss:** `0.9558`
- **Validation Dice (DSC):** `0.0792` *(Best so far: `0.0792`)*
- **Validation IoU:** `0.0412`
- **Validation Precision:** `0.0437`
- **Validation Recall:** `0.4208`
- **Validation Accuracy:** `0.9557`

### [Attention U-Net] Epoch 08 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9478`
- **Validation Loss:** `0.9567`
- **Validation Dice (DSC):** `0.0783` *(Best so far: `0.0792`)*
- **Validation IoU:** `0.0408`
- **Validation Precision:** `0.0418`
- **Validation Recall:** `0.6190`
- **Validation Accuracy:** `0.9341`

### [Attention U-Net] Epoch 09 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9375`
- **Validation Loss:** `0.9140`
- **Validation Dice (DSC):** `0.1310` *(Best so far: `0.1310`)*
- **Validation IoU:** `0.0701`
- **Validation Precision:** `0.1157`
- **Validation Recall:** `0.1509`
- **Validation Accuracy:** `0.9909`

### [Attention U-Net] Epoch 10 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9253`
- **Validation Loss:** `0.9492`
- **Validation Dice (DSC):** `0.0739` *(Best so far: `0.1310`)*
- **Validation IoU:** `0.0384`
- **Validation Precision:** `0.0629`
- **Validation Recall:** `0.0895`
- **Validation Accuracy:** `0.9898`

### [Attention U-Net] Epoch 11 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9105`
- **Validation Loss:** `0.9260`
- **Validation Dice (DSC):** `0.1081` *(Best so far: `0.1310`)*
- **Validation IoU:** `0.0571`
- **Validation Precision:** `0.0921`
- **Validation Recall:** `0.1307`
- **Validation Accuracy:** `0.9902`

### [Attention U-Net] Epoch 12 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9271`
- **Validation Loss:** `0.9217`
- **Validation Dice (DSC):** `0.1379` *(Best so far: `0.1379`)*
- **Validation IoU:** `0.0740`
- **Validation Precision:** `0.0883`
- **Validation Recall:** `0.3147`
- **Validation Accuracy:** `0.9822`

### [Attention U-Net] Epoch 13 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9045`
- **Validation Loss:** `0.8941`
- **Validation Dice (DSC):** `0.1672` *(Best so far: `0.1672`)*
- **Validation IoU:** `0.0912`
- **Validation Precision:** `0.1145`
- **Validation Recall:** `0.3101`
- **Validation Accuracy:** `0.9860`

### [Attention U-Net] Epoch 14 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9060`
- **Validation Loss:** `0.8843`
- **Validation Dice (DSC):** `0.1409` *(Best so far: `0.1672`)*
- **Validation IoU:** `0.0758`
- **Validation Precision:** `0.1909`
- **Validation Recall:** `0.1117`
- **Validation Accuracy:** `0.9938`

### [Attention U-Net] Epoch 15 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8825`
- **Validation Loss:** `0.8837`
- **Validation Dice (DSC):** `0.1617` *(Best so far: `0.1672`)*
- **Validation IoU:** `0.0880`
- **Validation Precision:** `0.1081`
- **Validation Recall:** `0.3205`
- **Validation Accuracy:** `0.9850`

### [Attention U-Net] Epoch 16 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8829`
- **Validation Loss:** `0.8999`
- **Validation Dice (DSC):** `0.1482` *(Best so far: `0.1672`)*
- **Validation IoU:** `0.0800`
- **Validation Precision:** `0.1426`
- **Validation Recall:** `0.1543`
- **Validation Accuracy:** `0.9920`

### [Attention U-Net] Epoch 17 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8894`
- **Validation Loss:** `0.9805`
- **Validation Dice (DSC):** `0.0207` *(Best so far: `0.1672`)*
- **Validation IoU:** `0.0104`
- **Validation Precision:** `0.3009`
- **Validation Recall:** `0.0107`
- **Validation Accuracy:** `0.9954`

### [Attention U-Net] Epoch 18 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9114`
- **Validation Loss:** `0.8711`
- **Validation Dice (DSC):** `0.2134` *(Best so far: `0.2134`)*
- **Validation IoU:** `0.1194`
- **Validation Precision:** `0.1560`
- **Validation Recall:** `0.3374`
- **Validation Accuracy:** `0.9887`

### [Attention U-Net] Epoch 19 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8824`
- **Validation Loss:** `0.8500`
- **Validation Dice (DSC):** `0.2275` *(Best so far: `0.2275`)*
- **Validation IoU:** `0.1283`
- **Validation Precision:** `0.1630`
- **Validation Recall:** `0.3766`
- **Validation Accuracy:** `0.9884`

### [Attention U-Net] Epoch 20 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8516`
- **Validation Loss:** `0.8368`
- **Validation Dice (DSC):** `0.2408` *(Best so far: `0.2408`)*
- **Validation IoU:** `0.1369`
- **Validation Precision:** `0.2158`
- **Validation Recall:** `0.2722`
- **Validation Accuracy:** `0.9922`

### Final Test Evaluation Results (Attention U-Net + Focal Tversky Loss)
| Metric | Baseline U-Net | Attention U-Net | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | **`0.1892`** | **`+0.0168`** |
| **Intersection over Union (IoU)** | `0.0943` | **`0.1045`** | **`+0.0102`** |
| **Precision** | `0.1092` | **`0.1597`** | **`+0.0505`** |
| **Recall (Sensitivity)** | `0.4085` | **`0.2320`** | **`-0.1765`** |
| **Pixel Accuracy** | `0.9863` | **`0.9930`** | **`+0.0067`** |

### [Attention U-Net 512x512] Epoch 01 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9859`
- **Validation Loss:** `0.9835`
- **Validation Dice (DSC):** `0.0797` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0415`
- **Validation Precision:** `0.0429`
- **Validation Recall:** `0.5583`
- **Validation Accuracy:** `0.9416`

### [Attention U-Net 512x512] Epoch 02 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9721`
- **Validation Loss:** `0.9950`
- **Validation Dice (DSC):** `0.0027` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0014`
- **Validation Precision:** `0.0581`
- **Validation Recall:** `0.0014`
- **Validation Accuracy:** `0.9954`

### [Attention U-Net 512x512] Epoch 03 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9558`
- **Validation Loss:** `0.9929`
- **Validation Dice (DSC):** `0.0084` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0042`
- **Validation Precision:** `0.0291`
- **Validation Recall:** `0.0049`
- **Validation Accuracy:** `0.9948`

### [Attention U-Net 512x512] Epoch 04 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9567`
- **Validation Loss:** `0.9598`
- **Validation Dice (DSC):** `0.0676` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0350`
- **Validation Precision:** `0.0372`
- **Validation Recall:** `0.3750`
- **Validation Accuracy:** `0.9532`

### [Attention U-Net 512x512] Epoch 05 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9466`
- **Validation Loss:** `0.9577`
- **Validation Dice (DSC):** `0.0751` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0390`
- **Validation Precision:** `0.0408`
- **Validation Recall:** `0.4741`
- **Validation Accuracy:** `0.9471`

### [Attention U-Net 512x512] Epoch 06 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9404`
- **Validation Loss:** `0.9629`
- **Validation Dice (DSC):** `0.0752` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0391`
- **Validation Precision:** `0.0519`
- **Validation Recall:** `0.1369`
- **Validation Accuracy:** `0.9848`

### [Attention U-Net 512x512] Epoch 07 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9362`
- **Validation Loss:** `0.9620`
- **Validation Dice (DSC):** `0.0794` *(Best so far: `0.0797`)*
- **Validation IoU:** `0.0414`
- **Validation Precision:** `0.0657`
- **Validation Recall:** `0.1003`
- **Validation Accuracy:** `0.9895`

### [Attention U-Net 512x512] Epoch 08 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9319`
- **Validation Loss:** `0.9507`
- **Validation Dice (DSC):** `0.0898` *(Best so far: `0.0898`)*
- **Validation IoU:** `0.0470`
- **Validation Precision:** `0.0592`
- **Validation Recall:** `0.1859`
- **Validation Accuracy:** `0.9829`

### [Attention U-Net 512x512] Epoch 09 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9366`
- **Validation Loss:** `0.9502`
- **Validation Dice (DSC):** `0.1014` *(Best so far: `0.1014`)*
- **Validation IoU:** `0.0534`
- **Validation Precision:** `0.0757`
- **Validation Recall:** `0.1534`
- **Validation Accuracy:** `0.9877`

### [Attention U-Net 512x512] Epoch 10 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9285`
- **Validation Loss:** `0.9686`
- **Validation Dice (DSC):** `0.0598` *(Best so far: `0.1014`)*
- **Validation IoU:** `0.0308`
- **Validation Precision:** `0.0861`
- **Validation Recall:** `0.0458`
- **Validation Accuracy:** `0.9935`

### [Attention U-Net 512x512] Epoch 11 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9308`
- **Validation Loss:** `0.9437`
- **Validation Dice (DSC):** `0.0941` *(Best so far: `0.1014`)*
- **Validation IoU:** `0.0494`
- **Validation Precision:** `0.0604`
- **Validation Recall:** `0.2131`
- **Validation Accuracy:** `0.9814`

### [Attention U-Net 512x512] Epoch 12 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9207`
- **Validation Loss:** `0.9422`
- **Validation Dice (DSC):** `0.0860` *(Best so far: `0.1014`)*
- **Validation IoU:** `0.0449`
- **Validation Precision:** `0.0886`
- **Validation Recall:** `0.0836`
- **Validation Accuracy:** `0.9920`

### [Attention U-Net 512x512] Epoch 13 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9165`
- **Validation Loss:** `0.9235`
- **Validation Dice (DSC):** `0.1260` *(Best so far: `0.1260`)*
- **Validation IoU:** `0.0672`
- **Validation Precision:** `0.0841`
- **Validation Recall:** `0.2507`
- **Validation Accuracy:** `0.9842`

### [Attention U-Net 512x512] Epoch 14 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.9090`
- **Validation Loss:** `0.9217`
- **Validation Dice (DSC):** `0.1156` *(Best so far: `0.1260`)*
- **Validation IoU:** `0.0613`
- **Validation Precision:** `0.1403`
- **Validation Recall:** `0.0983`
- **Validation Accuracy:** `0.9932`

### [Attention U-Net 512x512] Epoch 15 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8984`
- **Validation Loss:** `0.8786`
- **Validation Dice (DSC):** `0.1633` *(Best so far: `0.1633`)*
- **Validation IoU:** `0.0889`
- **Validation Precision:** `0.1737`
- **Validation Recall:** `0.1541`
- **Validation Accuracy:** `0.9928`

### [Attention U-Net 512x512] Epoch 16 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8754`
- **Validation Loss:** `0.9229`
- **Validation Dice (DSC):** `0.1099` *(Best so far: `0.1633`)*
- **Validation IoU:** `0.0581`
- **Validation Precision:** `0.1592`
- **Validation Recall:** `0.0839`
- **Validation Accuracy:** `0.9938`

### [Attention U-Net 512x512] Epoch 17 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8922`
- **Validation Loss:** `0.9560`
- **Validation Dice (DSC):** `0.0488` *(Best so far: `0.1633`)*
- **Validation IoU:** `0.0250`
- **Validation Precision:** `0.1191`
- **Validation Recall:** `0.0307`
- **Validation Accuracy:** `0.9946`

### [Attention U-Net 512x512] Epoch 18 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8735`
- **Validation Loss:** `0.8644`
- **Validation Dice (DSC):** `0.1648` *(Best so far: `0.1648`)*
- **Validation IoU:** `0.0898`
- **Validation Precision:** `0.2264`
- **Validation Recall:** `0.1295`
- **Validation Accuracy:** `0.9941`

### [Attention U-Net 512x512] Epoch 19 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8511`
- **Validation Loss:** `0.8761`
- **Validation Dice (DSC):** `0.1808` *(Best so far: `0.1808`)*
- **Validation IoU:** `0.0994`
- **Validation Precision:** `0.1750`
- **Validation Recall:** `0.1869`
- **Validation Accuracy:** `0.9923`

### [Attention U-Net 512x512] Epoch 20 / 20 Progress Update
- **Train Loss (Focal Tversky):** `0.8503`
- **Validation Loss:** `0.8417`
- **Validation Dice (DSC):** `0.2069` *(Best so far: `0.2069`)*
- **Validation IoU:** `0.1154`
- **Validation Precision:** `0.2100`
- **Validation Recall:** `0.2038`
- **Validation Accuracy:** `0.9929`

### Final Test Evaluation Results (Attention U-Net 512x512 vs 256x256 vs Baseline)
| Metric | Baseline U-Net (256x256) | Attention U-Net (256x256) | Attention U-Net (512x512) | Delta vs 256x256 |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | `0.1892` | **`0.1968`** | **`+0.0076`** |
| **Intersection over Union (IoU)** | `0.0943` | `0.1045` | **`0.1092`** | **`+0.0047`** |
| **Precision** | `0.1092` | `0.1597` | **`0.1855`** | **`+0.0258`** |
| **Recall (Sensitivity)** | `0.4085` | `0.2320` | **`0.2097`** | **`-0.0223`** |
| **Pixel Accuracy** | `0.9863` | `0.9930` | **`0.9940`** | **`+0.0010`** |


## 9. Boundary-Aware Attention U-Net (512x512) + Boundary Loss Experimentation Logs

### [Boundary-Aware Attention U-Net 512x512] Epoch 01 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `1.0226`
- **Validation Loss:** `1.0235`
- **Validation Dice (DSC):** `0.0352` *(Best so far: `0.0352`)*
- **Validation IoU:** `0.0179`
- **Validation Precision:** `0.0179`
- **Validation Recall:** `0.9808`
- **Validation Accuracy:** `0.7567`

### [Boundary-Aware Attention U-Net 512x512] Epoch 02 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9898`
- **Validation Loss:** `0.9816`
- **Validation Dice (DSC):** `0.0711` *(Best so far: `0.0711`)*
- **Validation IoU:** `0.0368`
- **Validation Precision:** `0.0403`
- **Validation Recall:** `0.3012`
- **Validation Accuracy:** `0.9643`

### [Boundary-Aware Attention U-Net 512x512] Epoch 03 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9699`
- **Validation Loss:** `0.9628`
- **Validation Dice (DSC):** `0.0862` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0450`
- **Validation Precision:** `0.0476`
- **Validation Recall:** `0.4550`
- **Validation Accuracy:** `0.9563`

### [Boundary-Aware Attention U-Net 512x512] Epoch 04 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9622`
- **Validation Loss:** `0.9645`
- **Validation Dice (DSC):** `0.0799` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0416`
- **Validation Precision:** `0.0537`
- **Validation Recall:** `0.1562`
- **Validation Accuracy:** `0.9837`

### [Boundary-Aware Attention U-Net 512x512] Epoch 05 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9539`
- **Validation Loss:** `0.9571`
- **Validation Dice (DSC):** `0.0831` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0433`
- **Validation Precision:** `0.0460`
- **Validation Recall:** `0.4256`
- **Validation Accuracy:** `0.9574`

### [Boundary-Aware Attention U-Net 512x512] Epoch 06 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9512`
- **Validation Loss:** `0.9620`
- **Validation Dice (DSC):** `0.0703` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0365`
- **Validation Precision:** `0.0649`
- **Validation Recall:** `0.0768`
- **Validation Accuracy:** `0.9908`

### [Boundary-Aware Attention U-Net 512x512] Epoch 07 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9489`
- **Validation Loss:** `0.9666`
- **Validation Dice (DSC):** `0.0706` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0366`
- **Validation Precision:** `0.0375`
- **Validation Recall:** `0.6025`
- **Validation Accuracy:** `0.9281`

### [Boundary-Aware Attention U-Net 512x512] Epoch 08 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9375`
- **Validation Loss:** `0.9670`
- **Validation Dice (DSC):** `0.0595` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0306`
- **Validation Precision:** `0.0577`
- **Validation Recall:** `0.0613`
- **Validation Accuracy:** `0.9912`

### [Boundary-Aware Attention U-Net 512x512] Epoch 09 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9409`
- **Validation Loss:** `0.9568`
- **Validation Dice (DSC):** `0.0840` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0438`
- **Validation Precision:** `0.0479`
- **Validation Recall:** `0.3400`
- **Validation Accuracy:** `0.9664`

### [Boundary-Aware Attention U-Net 512x512] Epoch 10 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9365`
- **Validation Loss:** `0.9554`
- **Validation Dice (DSC):** `0.0807` *(Best so far: `0.0862`)*
- **Validation IoU:** `0.0421`
- **Validation Precision:** `0.0484`
- **Validation Recall:** `0.2435`
- **Validation Accuracy:** `0.9749`

### [Boundary-Aware Attention U-Net 512x512] Epoch 11 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9393`
- **Validation Loss:** `0.9439`
- **Validation Dice (DSC):** `0.0960` *(Best so far: `0.0960`)*
- **Validation IoU:** `0.0504`
- **Validation Precision:** `0.0653`
- **Validation Recall:** `0.1806`
- **Validation Accuracy:** `0.9846`

### [Boundary-Aware Attention U-Net 512x512] Epoch 12 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9351`
- **Validation Loss:** `0.9441`
- **Validation Dice (DSC):** `0.0810` *(Best so far: `0.0960`)*
- **Validation IoU:** `0.0422`
- **Validation Precision:** `0.0847`
- **Validation Recall:** `0.0776`
- **Validation Accuracy:** `0.9920`

### [Boundary-Aware Attention U-Net 512x512] Epoch 13 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9265`
- **Validation Loss:** `0.9535`
- **Validation Dice (DSC):** `0.0865` *(Best so far: `0.0960`)*
- **Validation IoU:** `0.0452`
- **Validation Precision:** `0.0594`
- **Validation Recall:** `0.1596`
- **Validation Accuracy:** `0.9847`

### [Boundary-Aware Attention U-Net 512x512] Epoch 14 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9327`
- **Validation Loss:** `0.9559`
- **Validation Dice (DSC):** `0.0751` *(Best so far: `0.0960`)*
- **Validation IoU:** `0.0390`
- **Validation Precision:** `0.0655`
- **Validation Recall:** `0.0879`
- **Validation Accuracy:** `0.9902`

### [Boundary-Aware Attention U-Net 512x512] Epoch 15 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9266`
- **Validation Loss:** `0.9482`
- **Validation Dice (DSC):** `0.0956` *(Best so far: `0.0960`)*
- **Validation IoU:** `0.0502`
- **Validation Precision:** `0.0567`
- **Validation Recall:** `0.3051`
- **Validation Accuracy:** `0.9739`

### [Boundary-Aware Attention U-Net 512x512] Epoch 16 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9216`
- **Validation Loss:** `0.9268`
- **Validation Dice (DSC):** `0.1298` *(Best so far: `0.1298`)*
- **Validation IoU:** `0.0694`
- **Validation Precision:** `0.1016`
- **Validation Recall:** `0.1797`
- **Validation Accuracy:** `0.9891`

### [Boundary-Aware Attention U-Net 512x512] Epoch 17 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.9046`
- **Validation Loss:** `0.8720`
- **Validation Dice (DSC):** `0.1753` *(Best so far: `0.1753`)*
- **Validation IoU:** `0.0961`
- **Validation Precision:** `0.1905`
- **Validation Recall:** `0.1623`
- **Validation Accuracy:** `0.9931`

### [Boundary-Aware Attention U-Net 512x512] Epoch 18 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.8968`
- **Validation Loss:** `0.9536`
- **Validation Dice (DSC):** `0.0693` *(Best so far: `0.1753`)*
- **Validation IoU:** `0.0359`
- **Validation Precision:** `0.1056`
- **Validation Recall:** `0.0515`
- **Validation Accuracy:** `0.9937`

### [Boundary-Aware Attention U-Net 512x512] Epoch 19 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.8880`
- **Validation Loss:** `0.8983`
- **Validation Dice (DSC):** `0.1540` *(Best so far: `0.1753`)*
- **Validation IoU:** `0.0834`
- **Validation Precision:** `0.2758`
- **Validation Recall:** `0.1068`
- **Validation Accuracy:** `0.9947`

### [Boundary-Aware Attention U-Net 512x512] Epoch 20 / 20 Progress Update
- **Train Loss (Tversky + Boundary):** `0.8748`
- **Validation Loss:** `0.8573`
- **Validation Dice (DSC):** `0.1980` *(Best so far: `0.1980`)*
- **Validation IoU:** `0.1099`
- **Validation Precision:** `0.2483`
- **Validation Recall:** `0.1646`
- **Validation Accuracy:** `0.9940`

### Final Test Evaluation Results (Boundary-Aware Attention U-Net 512x512 vs Prior Stages)
| Metric | Baseline U-Net (256x256) | Attention U-Net (256x256) | Attention U-Net (512x512) | Boundary-Aware U-Net (512x512) |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | `0.1724` | `0.1892` | `0.1968` | **`0.2075`** |
| **Intersection over Union (IoU)** | `0.0943` | `0.1045` | `0.1092` | **`0.1157`** |
| **Precision** | `0.1092` | `0.1597` | `0.1855` | **`0.2660`** |
| **Recall (Sensitivity)** | `0.4085` | `0.2320` | `0.2097` | **`0.1701`** |
| **Pixel Accuracy** | `0.9863` | `0.9930` | `0.9940` | **`0.9954`** |
| **95% Hausdorff Distance (HD95)** | *--* | *--* | *--* | **`377.77 px`** |

---

## 10. Multi-Class Bone vs. Background Segmentation via FleXray Foundation Model

To satisfy the multi-class requirement in Problem Statement 1 (*"bone, fracture region, background, and other relevant classes"*), we integrated **FleXray** (MIT CSAIL, 2024 universal clinical X-ray segmentation foundation model).

### 10.1 Architecture & Method
- **Model:** FleXray flagship bundle (`VictorButoi/flexray`) pre-trained on CT-derived digitally reconstructed radiographs (DRRs) and multi-center clinical X-rays.
- **Label Protocol:** 61 anatomical structures, including 55 skeletal bone classes (`skull`, `scapulae`, `clavicles`, `humeri`, `phalanges`, `metacarpals`, `carpals`, `radii`, `ulnae`, `tibiae`, `femurs`, `patellae`, `fibulae`, `metatarsals`, `tarsals`, `toes`, ribs 1-12, vertebrae C1-L5, `hips`, `sacrum`).
- **Unified Bone Mask:** Aggregated across all 55 bone channels:
  $$M_{\text{bone}}(x) = \bigvee_{k \in \text{Bones}} M_k(x)$$
- **Hierarchical Multi-Class Fusion:**
  - **Class 0 (Background / Soft Tissue):** $1 - M_{\text{bone}}$
  - **Class 1 (Intact Bone):** $M_{\text{bone}} \setminus M_{\text{fracture}}$
  - **Class 2 (Bone Fracture):** $M_{\text{fracture}} \cap M_{\text{bone}}$

### 10.2 Experimental Results across 20 FracAtlas Test Images
- **Mean Inference Speed:** **$\sim 0.02 - 0.03\text{ seconds per image}$** ($>35\text{ FPS}$ on RTX 4050 GPU).
- **Fracture Containment in Bone:** Reached up to **`100.0%`** (e.g. `IMG0003312`: $100\%$, `IMG0003331`: $97.6\%$, `IMG0003332`: $95.2\%$, `IMG0003381`: $92.6\%$).
- **Anatomical Recognition Accuracy:** Zero-shot identification of limb anatomies:
  - Hands/Wrists: accurately isolated `phalanges`, `metacarpals`, `carpals`, `radii`, and `ulnae`.
  - Legs/Knees: accurately isolated `femurs`, `tibiae`, `patellae`, and `fibulae`.
  - Pelvis/Hips: accurately isolated `hips`, `femurs`, and `sacrum`.
- **Diagnostic Deliverables:** 20 4-panel diagnostic images (`Raw | FleXray Multi-Bone Anatomy | Unified Bone Mask | Multi-Class Composite`) saved in `results/flexray_bone_segmentations/`. Full metrics saved in `results/flexray_test_20_summary.csv`.


---

## 11. Dataset-Wide 3-Class Ground Truth Regeneration via Coordinate-Aligned FleXray

To enable end-to-end 3-class semantic segmentation (`0: Background, 1: Intact Bone, 2: Fracture`), the coordinate-aligned FleXray pipeline was executed across the entire FracAtlas dataset:
- **Total Scans Processed:** 719 fractured X-rays (Train: 574, Validation: 82, Test: 63).
- **Processing Time:** **64.94 seconds** total throughput ($\sim 0.090\text{ s/image}$, **11.1 FPS** on RTX 4050 GPU).
- **Execution Script:** `generate_all_bone_masks.py` (optimized via pre-aggregation of 55 bone channels at $256 \times 256$, followed by single-channel coordinate unpadding and nearest-neighbor upsampling).

### 11.1 Dataset Summary Statistics
- **Mean Bone Coverage:** **`14.21%`** of total image area across the dataset.
- **Mean Fracture-in-Bone Containment:** **`80.10%`** across all 719 scans.
- **Artifacts Saved:**
  - `bone_masks/*.png`: Lossless native-resolution binary bone masks (`0: Non-Bone, 255: Bone`).
  - `multiclass_masks/*.png`: Lossless native-resolution 3-class semantic masks (`0: Background, 1: Intact Bone, 2: Fracture`).
  - `results/dataset_3class_summary.csv`: Image-by-image coverage, containment, and pixel distribution ledger.
  - `fracatlas_3class_dataset.py`: PyTorch Dataset class providing synchronized CLAHE, data augmentations, and $(512 \times 512)$ multi-class tensor conversion.
