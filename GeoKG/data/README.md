We download this dataset in [Geonames](https://www.geonames.org/), and use [allCountries.zip](https://download.geonames.org/export/dump/allCountries.zip) as raw datasets.

# Details Creation of GeoKG

## 01 Download dataset

We download this dataset in [Geonames](https://www.geonames.org/), and use [allCountries.zip](https://download.geonames.org/export/dump/allCountries.zip),  [featureCodes_en.txt](https://download.geonames.org/export/dump/featureCodes_en.txt), [hierarchy.zip](https://download.geonames.org/export/dump/hierarchy.zip),as raw datasets.

Please rename featureCodes_en.txt to featureCodes.txt.

## 02 Procss

```shell
python process.py
```

which will automatically input allCountries.txt and featureCodes.txt and output kg_triples.txt, entities.txt and relations.txt.

## 03 Split

```
python split.py
```

* PRUNE_RATIO: Exactly 7.5% of triples to keep
* SEEN_RATIO: 70% of entities/relations seen in train
* TRAIN_RATIO : 80% of pruned triples in training
* VALID_RATIO: 10% of pruned triples in validation
* TEST_RATIO: 10% of pruned triples in testing

which will create the inductive datasets, note that if the SEEN_RATIO is set to 0%, the datasets will be the most challenge inductive datasets, which means all relations and entities in the inference are not training and seen during the training time. 



## 04 Move

Move the entities.txt, relations.txt, test.txt, train.txt and valid.txt into ./GeoKG_ind
