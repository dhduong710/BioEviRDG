# Dataset Creation for PrimeKG

## Download Dataset

Please download the raw PrimeKG data from: [PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM).

Then, put it under the data folder and unzip 

```bash
path/to/the/PrimeKG/data/PrimeKG/dataverse_files.zip -d data/PrimeKG
```

The preprocessed node features are under the folder `data/Processed/`.

The pre-encoded node embeddings are under the folder `data/embeddings`.

## Build data configurations

```
python build_data_config.py
```

## 2. build the node embedding dictionary

```shell
python build_node_emb.py
```
Attention:
In our experiments, UniMol fails to encode the following drug nodes

```
node_index = [14186, 14736, 14737, 19929, 19948, 20103, 20271, 20832]
drugbank_id = ['DB00515', 'DB00526', 'DB00958', 'DB08276', 'DB01929', 'DB04156', 'DB04100', 'DB13145']
```
so `embedding_dict.pkl` won't have these nodes and we will drop these nodes from the training dataset in step \# 4.

It is encouraged to investigate the reason why UniMol fails to encode these nodes and add the encoded embeddings to `embedding_dict.pkl`.

UniMol representation can be found in the guidance: https://github.com/dptech-corp/Uni-Mol/tree/main/unimol_tools#unimol-molecule-and-atoms-level-representation 

## 3. build the full triplet set by filtering out from PrimeKG

```shell
python build_full_triplets.py
```

## 4. split the tiplet_full.csv

```
python split.py
```

The train: valid: test= 80%: 10% 10%, you can adjust the split ratio if needed.

## 5. transfer into txt data

```
python trasfer_csv_to_txt.py
```

This method will automatically generate txt data and other necessary txt files.

## Move into training file

Move all the generated .txt file into ../data/PeimeKG_ind