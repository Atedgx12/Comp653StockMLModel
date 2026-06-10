# Literature Review

This review summarizes the papers and resources that motivated the methodological choices in [docs/proposal_revised.md](proposal_revised.md). Each entry states what the paper contributes, how it informs this project's design, and which course-feedback challenge it helps resolve.

## Walk forward backtesting and information leakage

### Lopez de Prado, *Advances in Financial Machine Learning*, Wiley, 2018
The book formalizes the failure modes of standard k-fold cross validation when applied to financial time series. It introduces purging and embargo as the two corrective steps that prevent information leakage between training and test sets. The walk forward implementation in [src/stockml/splits/walk_forward.py](../src/stockml/splits/walk_forward.py) follows this prescription. The `purge_and_embargo` helper drops training rows whose forward labels overlap the test window and applies a configurable embargo after the test window before the next fold begins.

This source addresses the look ahead bias challenge. Without these corrections, autocorrelated labels would leak information from the test fold into the training fold and inflate the reported metrics.

## Tabular gradient boosting for noisy financial features

### Ke and others, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017
Introduces histogram based gradient boosting with leaf wise tree growth. The combination of categorical handling, monotonic constraints, and gain based feature importance makes LightGBM the dominant tabular learner on financial data. The implementation in [src/stockml/models/lightgbm_models.py](../src/stockml/models/lightgbm_models.py) uses the trainer's early stopping callback to halt fitting when the validation log loss stops improving.

This source supports both the model selection rationale and the response to the correlated feature challenge: gradient boosting splits handle redundant features without losing predictive power because each split selects only one feature at a time.

## Random forests as a baseline

### Breiman, "Random Forests", Machine Learning 45(1), 2001
The original random forest paper. Random forests served as a baseline in the original course proposal but are not retained in the revised model list because LightGBM provides equivalent or stronger performance with a much smaller hyperparameter footprint. The paper still informs the design because the bagging argument explains why tree ensembles handle correlated features more gracefully than linear models.

## Transfer learning, domain adaptation, foundation models

### Pan and Yang, "A Survey on Transfer Learning", IEEE TKDE 22(10), 2010
The canonical taxonomy of transfer learning settings (inductive, transductive, unsupervised) and the formal definition of source and target domains. The pretrain then fine tune protocol in the revised proposal corresponds to inductive transfer with shared input space and different label distributions. The paper grounds the choice to keep a single union schema across pretraining and target sets.

### Rahimikia, Ni, and Wang, "Re(visiting) Time Series Foundation Models in Finance", arXiv 2511.18578, 2025
Recent benchmark of time series foundation models on financial forecasting tasks. The paper documents the failure modes of zero shot foundation models on finance specific data: distribution shift, scale heterogeneity, and the absence of explicit volatility regimes in pretraining corpora. It motivates the design choice in this project to add explicit regime features instead of relying on the model to discover the macro state on its own.

## Sequence and attention models for financial time series

### Nie, Nguyen, Sinthong, and Kalagnanam, "A Time Series is Worth 64 Words: Long Term Forecasting with Transformers" (PatchTST), ICLR 2023
Patch based transformer architecture that splits the input series into fixed length patches before attention. The architecture transfers cleanly across instruments because the attention head learns relations among patch embeddings that generalize beyond a single ticker. The transformer implementation in [src/stockml/models/neural.py](../src/stockml/models/neural.py) follows this design with a configurable patch length.

### Ghoshal and Roberts, "Thresholded ConvNet Ensembles", Neural Computing and Applications 32, 2020
Convolutional ensembles for technical forecasting on equities. The paper motivates the temporal CNN family in the project: dilated convolutions over fixed length input windows scale to many instruments without per-asset retraining. The CNN implementation in `src/stockml/models/neural.py` mirrors the dilated stack described in the paper.

### FinTime Decoder Dataset (Hugging Face)
A community curated benchmark for decoder transformers on financial time series. Used as a reference for the patching and tokenization conventions in the project's transformer implementation.

## Nonstationarity, regime change, online learning

### Lo, Mamaysky, and Wang, "Foundations of Technical Analysis", Journal of Finance 55(4), 2000
Provides the rigorous statistical framework for technical indicators and their information content. The paper documents the regime dependent nature of technical indicators, which is the empirical basis for the regime feature group in [src/stockml/features/regime.py](../src/stockml/features/regime.py).

### Cai and Bollerslev, "Modeling and Forecasting (Un)reliable Realized Volatilities", Journal of Econometrics, 2010 (and successor work on rolling estimators)
Provides the standard realized volatility estimator and its asymptotic properties. The estimator implemented in `add_realized_volatility` in `src/stockml/features/technicals.py` follows the rolling standard deviation of log returns scaled by the square root of the trading day count, which is the form used in this literature.

### Chen and others on rolling window retraining for nonstationary financial series
Several recent papers (sample: Wang and Caginalp 2020; Krauss, Do, and Huck 2017 on deep neural networks for daily stock prediction) document that periodic refit on a rolling window outperforms fixed weight models on financial returns. The online linear regressor in [src/stockml/models/online_linear.py](../src/stockml/models/online_linear.py) is the project's direct response to the nonstationarity challenge raised in the proposal feedback.

## Cryptocurrency cross sectional ML

### "Machine Learning and the Cross Section of Cryptocurrency Returns", International Review of Financial Analysis 94, 2024
Cross sectional study of crypto returns using ML predictors. Documents that features developed for equities transfer to crypto with material modifications: session boundaries differ, trading is continuous, and microstructure features must be replaced. Informs the design in this project to keep crypto as a transfer evaluation rather than an integrated training set, and to use only asset class agnostic features during pretraining.

## Information theoretic and statistical foundations

### Cover and Thomas, *Elements of Information Theory*, 2nd ed., Wiley
The reference text for information theoretic metrics used to interpret feature importance and prediction quality. Mutual information between predicted and realized returns is a richer quality measure than RMSE alone and is one of the metrics reported in [src/stockml/training/metrics.py](../src/stockml/training/metrics.py) through the information coefficient (rank correlation).

---

## How the literature maps to the response document

| Course feedback challenge | Primary references |
|---|---|
| Binary task is too narrow | Pan and Yang (transfer); Nie and others (sequence forecasting) |
| Dataset merging and transfer learning | Pan and Yang; Rahimikia and others |
| Missing values and outliers | Lopez de Prado (preprocessing chapter) |
| How features combine for forecasting | Nie and others; Ghoshal and Roberts; Lo and others |
| Nonstationarity and regime change | Cai and Bollerslev; rolling refit literature; Lo and others |
| Correlated indicators | Ke and others (LightGBM); Breiman (RF) |
| Cross asset transfer evaluation | "Cross Section of Cryptocurrency Returns" |
