# Revised Project Proposal

This document supersedes the Module 3 homework proposal. It addresses every written comment from Dr. Lan's feedback and tightens the project scope so the deliverable is realistic for a single course term.

The original proposal is preserved in the course homework PDF. Only the parts that change are restated here.

---

## Summary of feedback and the response in this revision

| Feedback comment | Where it is addressed |
|---|---|
| Binary up or down classification is too narrow; the model cannot learn enough from a single sign label. | Section 1 reformulates the task as four parallel learning targets that share a feature pipeline. Binary direction is retained only as a sanity check. |
| The proposal lists three datasets but does not explain how they are merged or how transfer learning actually works across them. | Section 2 specifies the pretrain then fine tune protocol and the union schema. |
| Missing values and outliers are not discussed. | Section 3 spells out the cleaning and outlier policy and points to the implementation. |
| The feature list is interesting but does not explain how features combine to produce a forecast. | Section 4 documents the per family architectures and the way features feed each head. |
| Boosting and random forest may not solve the listed challenges. More literature review is required to choose appropriate models. | Section 5 introduces a model-by-challenge coverage matrix and Section 7 points at the standalone literature review. |
| Objectives should be narrowed and the model list trimmed. | Section 6 reduces scope to equities first with a crypto transfer evaluation, and trims the model list to six families with clearly distinct roles. |

---

## 1. Task reformulation

The original binary up or down task is reformulated as four parallel learning targets that share the same feature pipeline. Each target captures a different aspect of the future return distribution.

1. **Multi horizon return regression.** The model predicts the signed log return $r_{t+h} = \log C_{t+h} - \log C_t$ at horizons of one, five, and twenty sessions. The regression target carries strictly more information than a sign label. It also lets the same predictions feed downstream ranking and risk adjusted scoring.
2. **Quantile regression.** Three quantile heads predict the 10th, 50th, and 90th percentiles of $r_{t+h}$ at the same horizons. The result is an explicit uncertainty estimate. Pinball loss directly trains the calibration of the prediction interval.
3. **Multi class regime classification.** A small number of categorical labels combine direction with realized volatility into states such as up calm, up turbulent, flat, down calm, and down turbulent. This target retains classification simplicity while encoding richer structure than a sign.
4. **Autoregressive sequence forecasting.** A sequence model consumes the most recent sixty four sessions of features and predicts the same forward returns at the same three horizons. This is the formulation that benefits most from transfer learning because the sequence model learns reusable temporal pattern representations.

The original binary up or down label is retained only as a sanity check baseline. It is the simplest possible target and a useful regression test for the data pipeline, but it is not the primary deliverable.

---

## 2. Dataset, merging, and transfer learning protocol

### Sources

The same three sources are used as in the original proposal but with explicit roles.

| Source | Role | Time span |
|---|---|---|
| Huge Stock Market Dataset on Kaggle | Pretraining universe for equities | 1990s through 2017 |
| Yahoo Finance through yfinance | Recent equities and crypto evaluation set | 2010 through present |
| Cryptocurrency OHLCV from Kaggle community datasets | Crypto transfer evaluation only | 2018 through present |

### Union schema

All three sources are mapped to a single panel schema indexed by date with a ticker column and the OHLCV fields: `open`, `high`, `low`, `close`, `volume`. Adjusted close prices are used wherever available. Equities use the exchange-reported adjusted close so splits and dividends do not introduce artificial discontinuities. Cryptocurrency prices are already split free and dividend free and are used as is. Records with zero reported volume are dropped during ingestion, which removes delisted day artifacts and exchange suspension days.

### Pretrain then fine tune protocol

Transfer learning in this project is concrete and operational:

1. **Pretraining stage.** Each model family is trained on the equities pretraining universe, which is the Kaggle dataset filtered to a top two hundred liquidity universe with at least fifteen hundred sessions of history. Pretraining uses only the asset class agnostic feature group. Feature columns that are equities specific are excluded so the resulting weights are reusable on cryptocurrency data.
2. **Fine tuning stage.** The pretrained weights are loaded and training continues on a target subset such as a single asset, a sector, or the cryptocurrency panel. The fine tuning step uses a smaller learning rate and a shorter walk forward window. For tree models the equivalent step is to start from the pretrained booster and continue training with `init_model=` set to the pretrained checkpoint.
3. **Zero shot evaluation.** Before fine tuning the pretrained model is evaluated on the target set with no parameter update. The metric delta between zero shot and fine tuned scores quantifies the value of fine tuning and isolates the signal that transferred from pretraining.

### Missing values and outliers

The handling policy is explicit and implemented in `src/stockml/data/preprocessing.py`:

- Forward fill is used for fundamental columns reported less frequently than daily, capped at one or two sessions so the model does not see stale fundamentals from an arbitrary number of sessions in the past.
- Median imputation by ticker fills any residual missing values left at the start of an asset's history where forward fill has nothing to copy from.
- Per ticker winsorization clips returns at the 0.1st and 99.9th percentile to neutralize hard-limit moves and one off data errors without distorting the cross sectional distribution.
- Rows with zero reported volume are dropped before feature construction.

A short notebook in `notebooks/01_data_quality.ipynb` reports the missing value and outlier counts on the pretraining universe so the policy can be inspected.

---

## 3. How features combine to produce a forecast

The feature vector groups described in the original proposal are unchanged. What is new is an explicit description of how each model family consumes them.

### Tabular models

Linear, ridge, online linear, and LightGBM consume one feature row per `(date, ticker)` pair. The row contains the trend and momentum group, the volatility group, the price structure group, and the regime group. The label is the realized forward log return at the target horizon. The model therefore learns a mapping from the local snapshot of indicators to the next horizon's signed return. Feature importance for tree models is averaged across folds and reported as both gain and split frequency. Linear coefficients are reported with their sign, which gives the most direct interpretation.

### Sequence models

The temporal CNN, the LSTM, and the transformer all consume a window of the previous sixty four feature rows for one ticker. The window is reshaped into a tensor of shape `(window, n_features)` and passed through the encoder. The output is a small head that predicts the same forward log returns at horizons one, five, and twenty. The transformer uses patch embedding so the sequence is split into eight session patches before attention. The CNN uses dilated convolutions so the receptive field grows exponentially with depth.

### Online linear

The online linear model is the project's direct response to the nonstationarity challenge. It is identical to the ridge regression baseline at any one moment, but it refits every twenty one sessions on the trailing three years of data. The regime feature group is required so the model can adapt its coefficients to the current volatility bucket without seeing future information.

---

## 4. Model coverage matrix

The original proposal proposed gradient boosting, logistic regression, and a random forest. The feedback noted that this trio may not address the listed challenges. The revised model set is justified by mapping each challenge to the families that address it.

| Challenge | Linear | LightGBM | Online linear | TCN | Transformer | LSTM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Look ahead bias | Avoided by walk forward CV in every family | | | | | |
| Nonstationarity | Partial | Partial | Direct | Partial | Partial | Partial |
| Correlated indicators | Hurts | Robust | Hurts | Robust | Robust | Robust |
| Cross asset transfer | Limited | Limited | Limited | Strong | Strongest | Moderate |
| Interpretability | Strongest | Strong | Strongest | Weak | Weak | Weak |

The revised list trims the redundant random forest and adds the three sequence families because the literature consistently shows that sequence models share temporal pattern representations across instruments better than tabular models do. The supporting evidence is documented in `docs/literature_review.md`.

---

## 5. Narrowed scope

The deliverable for the course term is now narrowed to the following scope.

- **Phase one:** Build the full pipeline on US equities from the Kaggle dataset. Train every model family on the same task suite. Compare metrics across families. Identify the strongest pretraining baseline.
- **Phase two:** Use the strongest pretraining baseline for a transfer evaluation on a five coin cryptocurrency panel. Report the zero shot and fine tuned numbers separately. Discuss which model family transferred best.
- **Excluded for this term:** Options data, FINRA short interest, fundamental features that are not consistently available across the equity universe. These were promising in the original proposal but each one introduces its own data engineering subproject and would crowd out the model comparison work.

---

## 6. Deliverables

- A reproducible Python package, `stockml`, scaffolded in this repository.
- The end to end walk forward training driver, with embargoed and purged folds, implemented in `src/stockml/training/trainer.py`.
- Six model families: linear regression, logistic regression, LightGBM regression, LightGBM classification, online linear regression, and three sequence families behind the optional `torch` extra.
- A literature review document at `docs/literature_review.md` that summarizes the references cited above.
- A results document with the cross family comparison table on equities and the transfer evaluation numbers on cryptocurrency.
- A short presentation summarizing the findings.

---

## 7. Literature pointers

The detailed literature review with one paragraph summaries lives in `docs/literature_review.md`. The headline references are:

- Lopez de Prado, *Advances in Financial Machine Learning*, on walk forward backtesting, purging, and embargo.
- Ke and others, LightGBM on histogram based gradient boosting.
- Pan and Yang, transfer learning survey.
- Patch time series transformer (Nie and others, 2023) and the FinTime decoder dataset for transformer baselines on financial series.
- Recent papers on regime detection and rolling window retraining for nonstationary financial series.
