# Usage of GraphOracle

## Pre-Training

```
python train.py --mode train --datasets data/YAGO data/fb15k-237 data/WN18RR --save_model --gpu 0 --max_epoch 200 --early_stopping 10
```

The best model will be saved into ./saved_models.

## Zero-Shot

```
python train.py --mode zero-shot --datasets data/YAGO --gpu 0 --model_path ./path/to/your/model/latest_model.pt
```

## Finetune

```
python train.py --mode fintune --datasets data/YAGO --gpu 0 --model_path ./path/to/your/model/latest_model.pt --finetune_epoches 10 
```
