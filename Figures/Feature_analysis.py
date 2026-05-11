#!/usr/bin/env python3
import os
import argparse

def read_triples(file_path):
    """
    Read triples from a text file.
    Each non-empty line is split into (head, relation, tail).
    Returns a list of tuples.
    """
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples

def main(data_dir):
    files = ['train.txt', 'valid.txt', 'test.txt']
    counts = {}

    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            print(f"Warning: file dont find {path}")
            counts[fn] = 0
            continue
        triples = read_triples(path)
        counts[fn] = len(triples)
        print(f"{fn} tiplets number is: {counts[fn]}")

    all_triples = set()
    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        for trip in read_triples(path):
            all_triples.add(trip)

    total_triples = len(all_triples)
    print(f"\nmerge and unique tiplets number is: {total_triples}")

    edge_count = total_triples
    print(f"graph edges number is: {edge_count}")

    entities = set()
    relations = set()
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    print(f"entities number is: {len(entities)}")
    print(f"relations number is: {len(relations)}")

if __name__ == "__main__":

    data_dir = ""
    main(data_dir)
