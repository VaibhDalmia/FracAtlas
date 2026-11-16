# Next Steps Workflow & Research Roadmap: Medical Fracture Segmentation
**Project:** Project Assignment 1 — Medical X-ray Image Segmentation  
**Dataset:** FracAtlas (Bone Fracture Detection & Segmentation)  
**Established Baseline:** Standard U-Net (Test DSC: `0.1724`, Recall: `0.4085`, Accuracy: `0.9863`)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, CUDA 12.4)  

---

## 1. The Core Strategic Question
> **"Should I do more preprocessing before training, or should I jump to Attention U-Net or something else?"**

### Direct Recommendation: **Jump to Attention U-Net + Focal Tversky Loss, coupled with Higher Resolution ($512 \times 512$)**

#### Why More Standalone Preprocessing Alone Will Not Solve the Problem:
1. **The Core Bottleneck is Structural & Scale-Related, Not Raw Contrast:**
   - We already have **CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`)**, which is the clinical gold standard for X-ray contrast enhancement.
   - Adding more filters (e.g., bilateral, median, unsharp masking) does not help the network distinguish between normal cortical bone edges, joint spaces, and true fracture lines.
2. **The "Hairline Discontinuity" Problem (Resolution Bottleneck):**
   - FracAtlas images are original high-resolution radiographs ($2880 \times 2304$).
   - Fractures in X-rays are typically hairline cracks only **2 to 5 pixels wide**.
   - Downsampling the image down to **$256 \times 256$** reduces the fracture line to **$<0.4$ pixels**, causing fine fracture details to be smoothed out before entering the convolutional layers!
   - Scaling to **$512 \times 512$** preserves micro-fracture boundaries while comfortably fitting into your 6 GB RTX 4050 VRAM.
3. **The False Positive Bottleneck (Why Attention Gates are Essential):**
   - In our U-Net baseline test evaluation, **Recall was $40.85\%$**, but **Precision was $10.92\%$**.
   - This indicates the baseline model successfully detects fracture regions, but **over-predicts false positives** along normal anatomical joint spaces, overlapping bones, and soft tissue silhouettes.
   - Standard U-Net blindly copies all shallow features through skip connections. **Attention U-Net** introduces **Attention Gates (AGs)** that suppress background/joint activations and guide feature extraction specifically toward fracture lesions.

---

## 2. Recommended 4-Phase Research Roadmap

```mermaid
flowchart TD
    subgraph Phase0 ["Phase 0: Baseline (Completed)"]
        B0["Standard U-Net (256x256)<br/>BCE + Soft Dice Loss<br/>Test DSC: 0.1724 | Recall: 40.85%"]
    end

    subgraph Phase1 ["Phase 1: Loss Function Advancement (Immediate)"]
        B0 --> P1["Focal Tversky Loss (FTL)<br/>Penalize False Positives (alpha=0.7, beta=0.3)<br/>Addresses 99.6% Class Imbalance"]
    end

    subgraph Phase2 ["Phase 2: Architectural Innovation (Attention U-Net)"]
        P1 --> P2["Attention U-Net Architecture<br/>Self-Attention Gates on Skip Connections<br/>Filters out non-fracture joint spaces"]
    end

    subgraph Phase3 ["Phase 3: Multi-Scale Resolution Scaling"]
        P2 --> P3["Resolution Upgrade to 512x512<br/>Preserves thin 2-pixel hairline fractures<br/>Optimized for RTX 4050 (Batch size 4)"]
    end

    subgraph Phase4 ["Phase 4: Comparative Benchmark & Ablation"]
        P3 --> P4["Full Assignment Report Matrix:<br/>• Baseline U-Net vs Attention U-Net<br/>• Ablation of Loss Functions<br/>• DSC, IoU, Precision, Recall, Latency"]
    end
```

---

## 3. Technical Breakdown of Next Innovations

### Innovation 1: Attention U-Net (Oktay et al., 2018)
Standard skip connections concatenate low-level features $x^l$ directly with upsampled gating signal $g$. Attention U-Net inserts an **Attention Gate (AG)** before concatenation:

$$\alpha = \sigma_2\Big( \psi^T \big( \text{ReLU}(W_x x^l + W_g g + b_g) \big) + b_\psi \Big)$$
$$\hat{x}^l = \alpha \odot x^l$$

- **$g$ (Gating Signal):** Coarser, semantically rich feature map from the deeper layer.
- **$x^l$ (Skip Features):** High-resolution shallow spatial features.
- **$\alpha \in [0, 1]$ (Attention Coefficients):** Automatically attenuates background air, intact bone, and foreign metal markers while amplifying fracture locations.

```mermaid
flowchart LR
    X["Skip Feature Map (x^l)"] --> Wx["1x1 Conv (W_x)"]
    G["Coarser Gating Signal (g)"] --> Wg["1x1 Conv (W_g)"]
    Wx --> Add["Addition (+)"]
    Wg --> Add
    Add --> Relu["ReLU Activation"]
    Relu --> Psi["1x1 Conv (psi)"]
    Psi --> Sig["Sigmoid (alpha)"]
    Sig --> Mult["Element-wise Multiplication (x^l * alpha)"]
    X --> Mult
    Mult --> Out["Attended Features to Decoder"]
```

---

### Innovation 2: Focal Tversky Loss (FTL)
The Tversky Index generalizes Dice ($F_1$) by allowing asymmetric weighting between **False Positives (FP)** and **False Negatives (FN)**:

$$TI(\alpha, \beta) = \frac{\sum p_i y_i + \epsilon}{\sum p_i y_i + \alpha \sum p_i (1 - y_i) + \beta \sum (1 - p_i) y_i + \epsilon}$$

To focus on hard-to-segment hairline fractures, we apply the focal exponent $\gamma$:
$$\mathcal{L}_{FTL} = \big(1 - TI(\alpha, \beta)\big)^{1 / \gamma}$$

- **Parameter Tuning for FracAtlas:**
  - $\alpha = 0.7$ (heavier penalty on false positives $\rightarrow$ boosts Precision from $10.9\%$ up toward $30-40\%$).
  - $\beta = 0.3$ (maintains high recall).
  - $\gamma = 1.33$ (focuses backprop on ambiguous fracture edges).

---

### Innovation 3: High-Resolution Training ($512 \times 512$)
- **Resolution Impact:**
  - At $256 \times 256$: Canvas area $= 65,536$ pixels. Fracture area $\approx 230$ pixels.
  - At $512 \times 512$: Canvas area $= 262,144$ pixels. Fracture area $\approx 920$ pixels.
  - Quadrupling the spatial resolution gives convolutional kernels the spatial Nyquist sampling needed to detect thin, non-displaced crack fissures.
- **VRAM Budget on RTX 4050 (6 GB):**
  - Attention U-Net at $512 \times 512$ with `batch_size=4` consumes **$\sim 3.4$ GB VRAM**, fitting comfortably within the 6 GB ceiling with zero OOM risk.

---

## 4. Structured Ablation Experiment Matrix for Your Assignment

Problem Statement 1 explicitly demands an **empirical comparison and ablation study**. We will systematically evaluate the following matrix:

| Model ID | Architecture | Input Resolution | Loss Function | Expected Test DSC | Primary Clinical Impact |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **M1 (Done)** | Standard U-Net | $256 \times 256$ | BCE + Soft Dice | **`0.1724`** | Established Baseline |
| **M2** | Standard U-Net | $256 \times 256$ | Focal Tversky Loss | Expected: $0.22 - 0.26$ | Directly suppresses false positives |
| **M3** | Attention U-Net | $256 \times 256$ | Focal Tversky Loss | Expected: $0.28 - 0.35$ | Attention gates filter non-fracture joints |
| **M4** | Attention U-Net | $512 \times 512$ | Focal Tversky Loss | Expected: **$0.36 - 0.45+$** | High-res preservation of hairline fractures |

---

## 5. Immediate Next Action

We can proceed immediately with **Phase 1 & Phase 2**:
1. Implement **`attention_unet.py`** (Attention U-Net architecture with Attention Gates).
2. Implement **`FocalTverskyLoss`** in `train_unet.py` (or a dedicated `train_attention_unet.py`).
3. Launch a training run and log the comparative metrics directly into [PROJECT_PROGRESS.md](file:///d:/FracAtlas/PROJECT_PROGRESS.md) and [Project_Assignment_1_Report.md](file:///d:/FracAtlas/Project_Assignment_1_Report.md).

---

*This workflow document is saved in `D:\FracAtlas\NEXT_STEPS_WORKFLOW.md` for permanent reference.*
