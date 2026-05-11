import collections

def read_graph_from_txt(file_path):
    """
    Reads a graph from a .txt file.
    Each line in the file should consist of a triple: head, relation, tail, separated by a tab.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        list: A list of tuples, where each tuple is (head, relation, tail).
    """
    file_path1 = file_path + '\\train.txt'
    file_path2 = file_path + '\\valid.txt'
    file_path3 = file_path + '\\test.txt'
    triples = []
    try:
        with open(file_path1, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
                else:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path2, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    try:
        with open(file_path3, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    return triples

def count_ingram_edges(triples):
    """
    Counts edges in the relation graph based on the INGRAM method.
    INGRAM: If two relations share entities, including (R1 and R2 share head and tail entities,
    or the head entity of R1 is the tail entity of R2, or the tail entity of R1 is the head entity of R2),
    then the two relations have an (undirected) edge.

    This is interpreted as: an edge exists between relation types rA and rB if there exist
    two triples (hA, rA, tA) and (hB, rB, tB) such that:
    1. hA == hB AND tA == tB (share head and tail entities)
    2. hA == tB (head of rA's instance is tail of rB's instance)
    3. tA == hB (tail of rA's instance is head of rB's instance)
    Edges are between relation types and are undirected.
    """
    if not triples:
        return 0

    ingram_edges = set()
    n = len(triples)

    for i in range(n):
        h1, r1, t1 = triples[i]
        for j in range(n): # Iterate through all pairs, including with itself
            h2, r2, t2 = triples[j]

            # Condition 1: R1 and R2 share head and tail entities
            if h1 == h2 and t1 == t2:
                # Add undirected edge (store as frozenset of a sorted tuple for uniqueness)
                edge = tuple(sorted((r1, r2)))
                ingram_edges.add(frozenset({r1,r2}))


            # Condition 2: The head entity of R1's instance is the tail entity of R2's instance
            if h1 == t2:
                edge = tuple(sorted((r1, r2)))
                ingram_edges.add(frozenset({r1,r2}))

            # Condition 3: The tail entity of R1's instance is the head entity of R2's instance
            if t1 == h2:
                edge = tuple(sorted((r1, r2)))
                ingram_edges.add(frozenset({r1,r2}))
                
    return len(ingram_edges)

def count_ultra_edges(triples):
    """
    Counts edges in the relation graph based on the ULTRA method.
    ULTRA: There are four types of directed edges in ULTRA:
    - head2head: the head entity of R1's instance is the head entity of R2's instance (R1 -> R2)
    - tail2tail: the tail entity of R1's instance is the tail entity of R2's instance (R1 -> R2)
    - head2tail: the head entity of R1's instance is the tail entity of R2's instance (R1 -> R2)
    - tail2head: the tail entity of R1's instance is the head entity of R2's instance (R1 -> R2)

    For any two triples (h1, r1, t1) and (h2, r2, t2):
    - If h1 == h2, add directed edge (r1, r2)
    - If t1 == t2, add directed edge (r1, r2)
    - If h1 == t2, add directed edge (r1, r2)
    - If t1 == h2, add directed edge (r1, r2)
    Edges are between relation types and are directed.
    """
    if not triples:
        return 0

    ultra_edges = set()
    n = len(triples)

    for i in range(n):
        h1, r1, t1 = triples[i]
        for j in range(n):
            h2, r2, t2 = triples[j]

            # head2head: head of triple i's relation -> relation of triple j
            if h1 == h2:
                ultra_edges.add((r1, r2))

            # tail2tail: tail of triple i's relation -> relation of triple j
            if t1 == t2:
                ultra_edges.add((r1, r2))

            # head2tail: head of triple i's relation -> relation of triple j
            if h1 == t2:
                ultra_edges.add((r1, r2))

            # tail2head: tail of triple i's relation -> relation of triple j
            if t1 == h2:
                ultra_edges.add((r1, r2))
                
    return len(ultra_edges)

def count_graph_oracle_edges(triples):
    """
    Counts edges in the relation graph based on the GraphOracle method.
    GraphOracle: If in the knowledge graph, an entity e1 is connected to a second entity e2
    through a certain relationship R1 (triple: e1, R1, e2), and this second entity e2 is
    connected to a third entity e3 through another relationship R2 (triple: e2, R2, e3),
    then we connect a directed edge from R1 to R2 in the relationship graph.

    For any two triples (h1, r1, t1) and (h2, r2, t2):
    - If t1 == h2, add directed edge (r1, r2).
    Edges are between relation types and are directed.
    """
    if not triples:
        return 0
        
    graph_oracle_edges = set()
    n = len(triples)

    for i in range(n):
        h1, r1, t1 = triples[i] # This is the first relationship R1
        for j in range(n):
            h2, r2, t2 = triples[j] # This is the second relationship R2

            # If tail of the first triple's entity matches head of the second triple's entity
            if t1 == h2:
                graph_oracle_edges.add((r1, r2))
                
    return len(graph_oracle_edges)

def main(file_path):



    print(f"Reading graph data from: {file_path}\n")
    triples = read_graph_from_txt(file_path)

    if triples is None:
        print("Could not read graph data. Exiting.")
        return

    print(f"Number of triples read: {len(triples)}\n")

    # Calculate edge counts for each method
    ingram_edge_count = count_ingram_edges(triples) * 2 # *2 because it's undirected
    ultra_edge_count = count_ultra_edges(triples)
    graph_oracle_edge_count = count_graph_oracle_edges(triples)

    # Output the results
    print("--- Relation Graph Edge Counts ---")
    print(f"INGRAM method:      {ingram_edge_count} edges")
    print(f"ULTRA method:       {ultra_edge_count} edges")
    print(f"GraphOracle method: {graph_oracle_edge_count} edges")
    
    # For verification against the manual trace with the first 4 lines:
    # e1	R1	e2
    # e1	R2	e3
    # e2	R3	e4
    # e5	R2	e2
    # Expected INGRAM: 5 ( {R1,R1}, {R1,R3}, {R2,R2}, {R3,R3}, {R2,R3} )
    # Expected ULTRA: 9 ( (R1,R1), (R1,R2), (R1,R3), (R2,R1), (R2,R2), (R3,R1), (R3,R3), (R3,R2), (R2,R3) )
    # Expected GraphOracle: 2 ( (R1,R3), (R2,R3) )
    # print("\n--- Verification with first 4 triples from example ---")
    # verification_triples = [
    #     ('e1','R1','e2'), ('e1','R2','e3'), ('e2','R3','e4'), ('e5','R2','e2')
    # ]
    # print(f"INGRAM (manual example):      {count_ingram_edges(verification_triples)}")
    # print(f"ULTRA (manual example):       {count_ultra_edges(verification_triples)}")
    # print(f"GraphOracle (manual example): {count_graph_oracle_edges(verification_triples)}")


if __name__ == '__main__':
    file_path = ""
    main("file_path")