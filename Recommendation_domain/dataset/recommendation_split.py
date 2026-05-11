def process_relations():
    # Read relation_list.txt and get the last relation ID
    with open('relation_list.txt', 'r') as file:
        relations = file.readlines()

    last_relation = relations[-1].strip().split()
    last_id = int(last_relation[-1])
    purchase_id = last_id + 1

    # Create relations.txt with the new purchase relation
    with open('../data/relations.txt', 'w') as file:
        for relation in relations:
            file.write(relation)
        if not relations[-1].endswith('\n'):
            file.write('\n')
        file.write(f'purchase\t{purchase_id}\n')
    
    return purchase_id

def create_purchase_triples(purchase_id):
    # Convert user-item relations from old_train.txt to triples
    purchase_triples = []
    with open('old_train.txt', 'r') as file:
        for line in file:
            entities = line.strip().split()
            user = entities[0]
            items = entities[1:]
            
            for item in items:
                purchase_triples.append(f"{user}\t{purchase_id}\t{item}")
    
    return purchase_triples

def test(purchase_id):
    # Convert user-item relations from old_train.txt to triples
    purchase_triples = []
    with open('test.txt', 'r') as file:
        for line in file:
            entities = line.strip().split()
            user = entities[0]
            items = entities[1:]
            
            for item in items:
                purchase_triples.append(f"{user}\t{purchase_id}\t{item}")
    with open('../data/test.txt', 'w') as file:
        file.write('\n'.join(purchase_triples))
    
    return purchase_id
def valid(purchase_id):
    # Convert user-item relations from old_train.txt to triples
    purchase_triples = []
    with open('valid.txt', 'r') as file:
        for line in file:
            entities = line.strip().split()
            user = entities[0]
            items = entities[1:]
            
            for item in items:
                purchase_triples.append(f"{user}\t{purchase_id}\t{item}")
    with open('../data/valid.txt', 'w') as file:
        file.write('\n'.join(purchase_triples))

    return purchase_id

def merge_triples(purchase_triples):
    # Merge existing KG triples with purchase triples
    with open('kg.txt', 'r') as file:
        kg_content = file.read()

    with open('../data/train.txt', 'w') as file:
        file.write(kg_content)
        if not kg_content.endswith('\n'):
            file.write('\n')
        file.write('\n'.join(purchase_triples))

def generate_entity_mapping():
    # Get entities from train.txt (which contains triples)
    train_entities = set()
    with open('../data/train.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  # Ensure it's a valid triple
                train_entities.add(parts[0])  # Add head entity
                train_entities.add(parts[2])  # Add tail entity
    
    # Get entities from old_test.txt
    test_entities = set()
    with open('old_test.txt', 'r') as file:
        for line in file:
            entities = line.strip().split()
            for entity in entities:
                test_entities.add(entity)
    
    # Combine all unique entities
    all_entities = train_entities.union(test_entities)
    
    # Sort entities to ensure consistent numbering
    sorted_entities = sorted(all_entities, key=int)
    
    # Create entity to ID mapping
    entity_mapping = {entity: str(idx) for idx, entity in enumerate(sorted_entities)}
    
    # Save entity mapping to entities.txt
    with open('../data/entities.txt', 'w') as file:
        for entity, entity_id in entity_mapping.items():
            file.write(f"{entity}\t{entity_id}\n")
    
    return entity_mapping

def main():
    purchase_id = process_relations()
    purchase_triples = create_purchase_triples(purchase_id)
    merge_triples(purchase_triples)
    entity_mapping = generate_entity_mapping()

if __name__ == "__main__":
    main()