# COMP 653 Stock Direction Prediction

**Course**: COMP 653 Statistical Machine Learning, Summer 2026, Rice University  
**Author**: Zachary Powell (zp21@rice.edu)  
**Hardware**: RTX 5080 (zach-ai) via Tailscale SSH

Cross-sectional equity direction prediction using a unified multi-branch network built entirely from COMP 653 algorithms.

---

## Results

| Model | Accuracy | AUC |
|---|---|---|
| **UnifiedCourseNetwork (LR + NB + MLP + Sent)** | **0.5099** | **0.5163** |
| LightGBM-GPU (baseline) | 0.5103 | 0.5138 |

Both models evaluated on a 420-ticker x 2066-day S&P 500 panel. Course model trained on 20 MI-selected cross-sectional rank features.

---

## Architecture

```
Input x  (20 MI-selected cross-sectional rank features)
      |
+-----+--------------------------------------------+
| Branch A  Logistic Regression  (Lec 5-2)         |
|   Linear(20->2) -> Sigmoid                        |
+---------------------------------------------------+
| Branch B  Naive Bayes  (Lec 5-3)                 |
|   Learnable Gaussian norm -> Linear(20->2)->Sig   |
+---------------------------------------------------+
| Branch C  Deep MLP  (Lec 5-5)                    |
|   Linear(20->128)->ReLU->Dropout(0.4)             |
|   Linear(128->64)->ReLU->Dropout(0.4)             |
+---------------------------------------------------+
| Branch D  VADER Sentiment  (extension)            |
|   Linear(1->2)->Sigmoid                           |
+-----+--------------------------------------------+
      |  MetaDrop(0.2) on concat(a_lr, a_nb, a_mlp, a_sent)
Meta-layer: Linear(70->2) -> Softmax
      |
P(Up), P(Down)
```

All branches trained jointly end-to-end via backpropagation.  
Optimizer: **Adam** (Module 6, Lec 6-5) with cosine LR annealing.  
Best-checkpoint restoration over full 500-epoch schedule.

---

## Feature Engineering

32 raw features per ticker, cross-sectionally ranked (percentile) by date:

| Group | Features |
|---|---|
| Returns | ret1, ret2, ret3, ret5, ret10, ret20, ret60, ret120, ret252, ret756 |
| Volatility | vol5, vol10, vol20, vol60, vol120, vol252 |
| Momentum | mom5, mom10, mom20, mom60, mom120, mom252 |
| Structural | vol_ratio, ma50_ratio, ma200_ratio, ma50_200_cross, ret_accel |
| Oscillator | rsi14 |
| Distance | dist52h, dist52l, dist3yh, dist3yl |

Top features by mutual information (Module 2): `ret252`, `mom252`, `dist3yh`, `dist52h`, `ret756`.

---

## Setup

```bash
git clone https://github.com/Atedgx12/Comp653StockMLModel
cd Comp653StockMLModel

# Python 3.11+
pip install -e .

# Optional: sequence models
pip install -e ".[torch]"

# Dev tools
pip install -e ".[dev]"
```

---

## Running

```bash
# Full pipeline (run on RTX 5080 via SSH)
python scripts/pipeline_course.py

# Deploy from local and run remotely
scp scripts/pipeline_course.py zach-ai:D:/StockModel/pipeline_course.py
ssh zach-ai "C:\Users\kizzi\.venv\Scripts\python.exe -u D:\StockModel\pipeline_course.py 2>&1"

# Sync outputs and push to GitHub
.\sync_from_zach.ps1
```

---

## Project Structure

```
src/stockml/
  data/ingestion.py          yfinance bulk download + parquet cache
  features/technicals.py     32-feature multi-timeframe pyramid
  features/pipeline.py       cross-sectional rank pipeline
  features/pruning.py        MI feature selection (Module 2)
  labels/returns.py          cross-sectional top/bottom 30% label
  models/neural.py           UnifiedCourseNetwork (pure NumPy)
  models/lightgbm_models.py  LightGBM GPU baseline
  splits/walk_forward.py     expanding walk-forward CV
  training/metrics.py        Wilcoxon AUC, accuracy, IC
scripts/
  pipeline_course.py         main COMP 653 pipeline
configs/model/
  unified_course_network.yaml
outputs/
  loss_curve_UnifiedCourseNetwork_Adam_Sent.png
  final_results_unified.csv
notebooks/
  03_presentation.ipynb
sync_from_zach.ps1           one-command sync from RTX 5080 + push
```

---

## Course Alignment

| Component | Module |
|---|---|
| Entropy, mutual information, MI feature selection | Module 2 |
| Logistic Regression branch | Module 5, Lec 5-2 |
| Naive Bayes normalization branch | Module 5, Lec 5-3 |
| Deep MLP + backpropagation branch | Module 5, Lec 5-5 |
| Adam optimizer | Module 6, Lec 6-5 |
| Cross-sectional evaluation, walk-forward CV | Module 3 |
