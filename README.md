# IS-project-YuanGao
# Span‑Based Aspect‑based Sentiment Triplet Extraction (ASTE) with BERT

## 🎯 Project Overview

This repository implements a span‑based ASTE framework built upon **BERT-base**, as described in the paper *“Span‑Based Aspect‑based Sentiment Triplet Extraction with BERT”* (Yuan Gao, 2025).  
The model enumerates candidate spans, constructs span representations (boundary + internal features), and classifies them in an end‑to‑end manner to extract (aspect, opinion, sentiment) triplets from text.
The pepar:https://www.overleaf.com/read/xthgqmpnkzpr#10bac8
---
## 1. Requirements

Conduct experiments with CUDA version 11.6 and PyTorch v1.10.1. 

To reproduce experimental environment.
```
python -m pip install -r requirements.txt
```

## 2. Data

Use `ASTE-Data-V2-EMNLP2020` from https://github.com/xuuuluuu/SemEval-Triplet-data.git
(widely-used datasets in ASTE task)

The data dir should be  ``data/ASTE-Data-V2-EMNLP2020`` (*or* , set the correct ***dataset_dir*** parameter during training or predicting)


## 3. Train

```
python run.py
```



