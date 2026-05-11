<!--
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  GraphOracle · A Relation-Centric Foundation Model for KG Reasoning   │
  └─────────────────────────────────────────────────────────────────────────┘
-->

# GraphOracle

<p align="center">
  <img src="Figures/GraphOracle.jpg" alt="GraphOracle Architecture" width="55%"/>
</p>
GraphOracle is a **relation-centric foundation model** that unifies transductive, inductive,  
and cross-domain knowledge-graph (KG) reasoning by converting KGs into compact  
Relation-Dependency Graphs (RDGs) and performing query-conditioned multi-head message passing.  
The accompanying code provides:

* **Pre-training & fine-tuning pipelines** for fully-inductive KG reasoning.  
* **Out-of-the-box scripts** for classic benchmarks (WN18RR, FB15k-237, NELL-995, YAGO3-10).  
* **Cross-domain adapters** for biomedical, geographic and recommendation KGs.  
* A lightweight implementation that runs on a single GPU.

> 📄 Looking for theoretical details and experimental results?  
> See the paper carefully .

---

## 1  Installation

```bash
cd GraphOracle
python -m venv venv         # optional but recommended
source venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

## 2  Quick Start

### 2.1 Pre-train on General KGs

```
python train.py \
  --mode train \
  --datasets data/YAGO data/FB15k-237 data/WN18RR \
  --save_model \
  --gpu 0 \
  --max_epoch 200 \
  --early_stopping 10
```

The best checkpoint will be stored in **`./saved_models`**.

### 2.2 Zero-Shot Inference

```
python train.py \
  --mode zero-shot \
  --datasets data/YAGO \
  --gpu 0 \
  --model_path ./saved_models/latest_model.pt
```

### Few-Shot Fine-Tuning

```
python train.py \
  --mode finetune \
  --datasets data/YAGO \
  --gpu 0 \
  --model_path ./saved_models/latest_model.pt \
  --finetune_epochs 10
```

## 3 Cross-domain Dataset 

### Biomedical dataset (PrimeKG) 

For processing details, please refer to the **Section Processing Details of Biomedical Domain** in the appendix of the paper. For comprehensive guidelines on constructing Bio datasets, please refer to the  [PrimeKG_Creation README](./PrimeKG/build_dataset/README.md).

```
cd PrimeKG
do as the README.md said
```

### Geographic datasets (GeoKG) 

For GeoKG processing details, please refer to the **Section Processing Detail of Geographic Datasets (GeoKG)** in the appendix of the paper. For comprehensive guidenlines on constructing Geo_datasets, please refer to the [GeoKG_Creation_README](./data/README.md)

```
cd GeoKG
do as the README.md said
```

### Recommendation domain (Amazon-book) 

For Amazon-book processing details, please refer to the **Section Processing Detail of Recommendation Domain** in the appendix of the paper. For comprehensive guidenlines on constructing Geo_datasets, please refer to the [Amazon_Creation_README](./Recommendation_domain/dataset/README.md)

## 4  Repository Structure

```
GraphOracle/
├── KGsReasoning/              # Classic benchmark scripts
├── PrimeKG/                   # Biomedical domain
├── GeoKG/                     # Geographic domain
├── Recommendation_domain/     # Amazon-Book recommender KG
└── Figures/                   # Paper figures & logo
```

## 5  Dataset-Specifi

| Domain & Dataset                                          | Folder & README                           | One-Line Setup             |
| --------------------------------------------------------- | ----------------------------------------- | -------------------------- |
| **Classic KGs**<br/>WN18RR, FB15k-237, NELL-995, YAGO3-10 | `KGsReasoning/README.md`                  | `cd KGsReasoning`          |
| **Biomedical**<br/>PrimeKG                                | `PrimeKG/build_dataset/README.md`         | `cd PrimeKG`               |
| **Geographic**<br/>GeoKG                                  | `GeoKG/data/README.md`                    | `cd GeoKG`                 |
| **Recommendation**<br/>Amazon-Book                        | `Recommendation_domain/dataset/README.md` | `cd Recommendation_domain` |



> Each sub-folder contains detailed **data-creation scripts, preprocessing steps, and training commands**.
>  For additional implementation notes, consult the *Processing Details* sections in the paper appendix.
