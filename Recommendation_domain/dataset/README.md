# Detail Process of Recommendation Dataset

## 01 Download Dataset

The raw data is given in [http://jmcauley.ucsd.edu/data/amazon](http://jmcauley.ucsd.edu/data/amazon), or you can download it directly in the [github repo](https://github.com/leolouis14/KUCNet/tree/main/data/amazon-book).

## 02 Suild Dataset

Our core idea is treat the relation 'purchase' as a new relation of the KG, then use it to create the new data.

```
python recommendation_split.py
```

## 03 Shuffle Dataset

```
python shuffle.py
```



## 04 Move

Move train.txt, valid.txt, test.txt, relations.txt and entities.txt into ../data