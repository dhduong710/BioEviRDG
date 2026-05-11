import pandas as pd

# all files
csv_files = ['../data/BindData/train.csv', '../data/BindData/valid.csv', '../data/BindData/test.csv']
txt_files = ['../data/BindData/train.txt', '../data/BindData/valid.txt', '../data/BindData/test.txt']

# store all unique entities and relations
entity_set = set()
relation_set = set()

# step 1: convert triples and collect entities and relations
for csv_file, txt_file in zip(csv_files, txt_files):
    df = pd.read_csv(csv_file)
    
    with open(txt_file, 'w', encoding='utf-8') as out_f:
        for _, row in df.iterrows():
            h = str(row['x_index'])
            r = str(row['relation']).replace('off-label use', 'off-label-use')  # replace space in relation name
            t = str(row['y_index'])
            out_f.write(f"{h}\t{r}\t{t}\n")
            
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)

# step 2: write entity mapping file
entity_list = sorted(entity_set)
entity2id = {entity: idx for idx, entity in enumerate(entity_list)}
with open('../data/BindData/entities.txt', 'w', encoding='utf-8') as f:
    for entity, idx in entity2id.items():
        f.write(f"{entity}\t{idx}\n")

# step 3: write relation mapping file
relation_list = sorted(relation_set)
relation2id = {relation: idx for idx, relation in enumerate(relation_list)}
with open('../data/BindData/relations.txt', 'w', encoding='utf-8') as f:
    for relation, idx in relation2id.items():
        f.write(f"{relation}\t{idx}\n")
