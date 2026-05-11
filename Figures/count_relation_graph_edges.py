def read_knowledge_graph(file_path):
    """Read a knowledge graph from a tab-separated text file."""
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            head, rel, tail = line.strip().split('\t')
            triples.append((head, rel, tail))
    return triples

def build_ingram_graph(triples):
    """Build a relationship graph using the INGRAM method."""
    # Create dictionaries to track entities for each relation
    rel_to_head_entities = {}
    rel_to_tail_entities = {}
    
    for head, rel, tail in triples:
        if rel not in rel_to_head_entities:
            rel_to_head_entities[rel] = set()
        if rel not in rel_to_tail_entities:
            rel_to_tail_entities[rel] = set()
        
        rel_to_head_entities[rel].add(head)
        rel_to_tail_entities[rel].add(tail)
    
    # Build the INGRAM graph (undirected)
    graph = {}
    relations = list(set(rel_to_head_entities.keys()))
    
    for i in range(len(relations)):
        r1 = relations[i]
        if r1 not in graph:
            graph[r1] = set()
        
        for j in range(i+1, len(relations)):  # Only check each pair once for undirected graph
            r2 = relations[j]
            if r2 not in graph:
                graph[r2] = set()
            
            # Check if r1 and r2 share any head or tail entities
            heads1 = rel_to_head_entities[r1]
            tails1 = rel_to_tail_entities[r1]
            heads2 = rel_to_head_entities[r2]
            tails2 = rel_to_tail_entities[r2]
            
            # Check all possible entity sharing combinations
            if (heads1.intersection(heads2) or 
                heads1.intersection(tails2) or 
                tails1.intersection(heads2) or 
                tails1.intersection(tails2)):
                graph[r1].add(r2)
                graph[r2].add(r1)  # Undirected edge
    
    return graph

def build_ultra_graph(triples):
    """Build a relationship graph using the ULTRA method."""
    # Create dictionaries for faster lookups
    rel_to_head_entities = {}
    rel_to_tail_entities = {}
    
    for head, rel, tail in triples:
        if rel not in rel_to_head_entities:
            rel_to_head_entities[rel] = set()
        if rel not in rel_to_tail_entities:
            rel_to_tail_entities[rel] = set()
        
        rel_to_head_entities[rel].add(head)
        rel_to_tail_entities[rel].add(tail)
    
    # Build the ULTRA graph (directed)
    graph = {}
    relations = list(set(rel_to_head_entities.keys()))
    
    for r1 in relations:
        if r1 not in graph:
            graph[r1] = set()
        
        for r2 in relations:
            if r1 == r2:
                continue
            
            heads1 = rel_to_head_entities[r1]
            tails1 = rel_to_tail_entities[r1]
            heads2 = rel_to_head_entities[r2]
            tails2 = rel_to_tail_entities[r2]
            
            # Four types of directed edges
            if heads1.intersection(heads2):  # head2head
                graph[r1].add(r2)
            if tails1.intersection(tails2):  # tail2tail
                graph[r1].add(r2)
            if heads1.intersection(tails2):  # head2tail
                graph[r1].add(r2)
            if tails1.intersection(heads2):  # tail2head
                graph[r1].add(r2)
    
    return graph

def build_graphoracle_graph(triples):
    """Build a relationship graph using the GraphOracle method."""
    # Create a dictionary to track head entities to (relation, tail) pairs
    head_to_rel_tail = {}
    
    for head, rel, tail in triples:
        if head not in head_to_rel_tail:
            head_to_rel_tail[head] = []
        head_to_rel_tail[head].append((rel, tail))
    
    # Build the GraphOracle graph (directed)
    graph = {}
    relations = set([rel for _, rel, _ in triples])
    for rel in relations:
        graph[rel] = set()
    
    for head1, rel1, tail1 in triples:
        # If tail1 is a head of another relation
        if tail1 in head_to_rel_tail:
            for rel2, _ in head_to_rel_tail[tail1]:
                graph[rel1].add(rel2)
    
    return graph

def count_edges(graph):
    """Count the total number of edges in a graph."""
    total_edges = 0
    for rel, neighbors in graph.items():
        total_edges += len(neighbors)
    return total_edges

def main(file_path):
    """Main function to read a knowledge graph and count edges."""
    triples = read_knowledge_graph(file_path)
    
    ingram_graph = build_ingram_graph(triples)
    ultra_graph = build_ultra_graph(triples)
    graphoracle_graph = build_graphoracle_graph(triples)
    
    ingram_edges = count_edges(ingram_graph)
    ultra_edges = count_edges(ultra_graph)
    graphoracle_edges = count_edges(graphoracle_graph)
    
    print("Edge counts for different relationship graph methods:")
    print(f"INGRAM: {ingram_edges} edges")
    print(f"ULTRA: {ultra_edges} edges")
    print(f"GraphOracle: {graphoracle_edges} edges")
    
    # Create a simple ASCII bar chart
    max_count = max(ingram_edges, ultra_edges, graphoracle_edges)
    scale = 50 / max_count if max_count > 0 else 1
    
    print("\nGraph Visualization:")
    print(f"INGRAM:      {'#' * int(ingram_edges * scale)} {ingram_edges}")
    print(f"ULTRA:       {'#' * int(ultra_edges * scale)} {ultra_edges}")
    print(f"GraphOracle: {'#' * int(graphoracle_edges * scale)} {graphoracle_edges}")

if __name__ == "__main__":

    main("train.txt")