#!/usr/bin/env python3
"""
Optimized version: Prune the GeoNames KG by selecting important triples
using relation-specific filtering with duplicate prevention.
"""

import random
from collections import defaultdict
import heapq
import time
import os

# ---------- Hyper‑parameters ----------
INPUT_FILE     = "kg_triples.txt"
TRAIN_FILE     = "train.txt"
VALID_FILE     = "valid.txt"
TEST_FILE      = "test.txt"
ENTITIES_FILE  = "entities.txt"
RELATIONS_FILE = "relations.txt"

PRUNE_RATIO  = 0.075    # Exactly 7.5% of triples to keep
SEEN_RATIO   = 0.70     # 70% of entities/relations seen in train
TRAIN_RATIO  = 0.96     # 80% of pruned triples in training
VALID_RATIO  = 0.02     # 10% of pruned triples in validation
TEST_RATIO   = 0.02     # 10% of pruned triples in testing
SEED         = 42
random.seed(SEED)

# Importance scoring weights
W_DEGREE = 0.6        # Weight for entity degree

def load_triples(path):
    """Load all triples from file, deduplicating as we go."""
    triples = set()  # Use a set to automatically deduplicate
    with open(path, encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.add((h, r, t))
    print(f"[*] Loaded {len(triples):,} unique triples")
    return list(triples)  # Convert back to list for further processing

def relation_balanced_pruning(all_triples, prune_ratio=0.075):
    """
    Prune the knowledge graph while maintaining relation balance:
    1. Group triples by relation
    2. Select top K% triples for each relation based on importance
    3. Combine selected triples from all relations
    """
    start_time = time.time()
    print(f"[*] Processing {len(all_triples):,} triples with relation-balanced pruning...")
    
    # Group triples by relation
    relation_groups = defaultdict(list)
    for idx, (h, r, t) in enumerate(all_triples):
        relation_groups[r].append((idx, h, t))
    
    print(f"[*] Found {len(relation_groups):,} distinct relations")
    
    # Calculate entity degrees - O(|T|)
    entity_degrees = defaultdict(int)
    for h, r, t in all_triples:
        entity_degrees[h] += 1
        entity_degrees[t] += 1
    
    # Normalize entity degrees
    max_degree = max(entity_degrees.values()) if entity_degrees else 1
    norm_degrees = {e: deg/max_degree for e, deg in entity_degrees.items()}
    
    # Process each relation group and select top triples
    selected_indices = set()  # Use a set to prevent duplicates
    
    for r, triples in relation_groups.items():
        # Score triples within this relation
        scored_triples = []
        for idx, h, t in triples:
            # Score based on entity degrees (relation frequency is constant within group)
            score = W_DEGREE * (norm_degrees[h] + norm_degrees[t])/2
            scored_triples.append((score, idx))
        
        # Select top k% from this relation
        k = max(1, int(len(triples) * prune_ratio))
        top_k = heapq.nlargest(k, scored_triples)
        selected_indices.update([idx for _, idx in top_k])
    
    # Get the final pruned triples
    pruned_triples = [all_triples[idx] for idx in selected_indices]
    
    # Double-check for duplicates
    unique_pruned = set(pruned_triples)
    if len(unique_pruned) != len(pruned_triples):
        print(f"[!] Warning: Found {len(pruned_triples) - len(unique_pruned)} duplicate triples after pruning")
        pruned_triples = list(unique_pruned)
    
    print(f"[✓] Relation-balanced pruning completed in {time.time()-start_time:.1f}s")
    print(f"[✓] Pruned to {len(pruned_triples):,} triples "
          f"({len(pruned_triples)/len(all_triples)*100:.2f}% of original)")
    
    return pruned_triples

def split_sets(full_set, seen_ratio):
    """Split a set into seen/unseen by random shuffle."""
    lst = list(full_set)
    random.shuffle(lst)
    cut = int(len(lst) * seen_ratio)
    return set(lst[:cut]), set(lst[cut:])

def controlled_split(triples):
    """
    Perform 70% seen / 30% unseen split on entities and relations,
    ensuring that 30% of entities and relations are unseen in train.
    Then allocate triples to train/valid/test (80/10/10).
    """
    print(f"[*] Performing controlled split with {SEEN_RATIO*100:.0f}% seen entities/relations...")
    
    # Ensure we're working with unique triples
    triples = list(set(triples))
    
    # Collect all entities & relations
    ents, rels = set(), set()
    for h, r, t in triples:
        ents.update([h, t])
        rels.add(r)
    
    # Assign 70% of entities and relations to be seen in training
    seen_ents, unseen_ents = split_sets(ents, SEEN_RATIO)
    seen_rels, unseen_rels = split_sets(rels, SEEN_RATIO)
    
    print(f"[*] Entity split: {len(seen_ents):,} seen, {len(unseen_ents):,} unseen")
    print(f"[*] Relation split: {len(seen_rels):,} seen, {len(unseen_rels):,} unseen")
    
    # Divide triples into train and evaluation sets
    train_candidates, eval_candidates = [], []
    
    for h, r, t in triples:
        # Only put in train if both entities and relation are in the seen sets
        if h in seen_ents and t in seen_ents and r in seen_rels:
            train_candidates.append((h, r, t))
        else:
            eval_candidates.append((h, r, t))
    
    print(f"[*] Initial split: {len(train_candidates):,} train candidates, {len(eval_candidates):,} eval candidates")
    
    # Ensure we have the exact 80/10/10 split overall
    total = len(triples)
    desired_train_size = int(total * TRAIN_RATIO)
    desired_valid_size = int(total * VALID_RATIO)
    desired_test_size = total - desired_train_size - desired_valid_size  # To ensure we get exactly 100%
    
    # Adjust train size if needed
    if len(train_candidates) > desired_train_size:
        # Move excess from train to eval
        random.shuffle(train_candidates)
        overflow = train_candidates[desired_train_size:]
        train_candidates = train_candidates[:desired_train_size]
        eval_candidates.extend(overflow)
    elif len(train_candidates) < desired_train_size:
        # Move some from eval to train (only those with seen entities if possible)
        # This slightly compromises the strict 70/30 seen/unseen split
        deficit = desired_train_size - len(train_candidates)
        eligible = [
            t for t in eval_candidates 
            if t[0] in seen_ents and t[2] in seen_ents and t[1] in seen_rels
        ]
        
        if len(eligible) >= deficit:
            random.shuffle(eligible)
            to_move = eligible[:deficit]
            train_candidates.extend(to_move)
            # Use set difference to avoid linear search
            eval_set = set(eval_candidates)
            move_set = set(to_move)
            eval_candidates = list(eval_set - move_set)
        else:
            # Not enough eligible triples, move any eval triples to meet ratio
            random.shuffle(eval_candidates)
            to_move = eval_candidates[:deficit]
            train_candidates.extend(to_move)
            eval_candidates = eval_candidates[deficit:]
    
    # Split eval into valid and test
    random.shuffle(eval_candidates)
    valid = eval_candidates[:desired_valid_size]
    test = eval_candidates[desired_valid_size:desired_valid_size+desired_test_size]
    
    # Final check for duplicates across splits
    train_set = set(train_candidates)
    valid_set = set(valid)
    test_set = set(test)
    
    # Check for size discrepancies due to duplicates
    if len(train_set) != len(train_candidates):
        print(f"[!] Removed {len(train_candidates) - len(train_set)} duplicate triples from training set")
        train_candidates = list(train_set)
    
    if len(valid_set) != len(valid):
        print(f"[!] Removed {len(valid) - len(valid_set)} duplicate triples from validation set")
        valid = list(valid_set)
        
    if len(test_set) != len(test):
        print(f"[!] Removed {len(test) - len(test_set)} duplicate triples from test set")
        test = list(test_set)
    
    # Check for overlap between sets
    train_valid_overlap = train_set.intersection(valid_set)
    train_test_overlap = train_set.intersection(test_set)
    valid_test_overlap = valid_set.intersection(test_set)
    
    if train_valid_overlap:
        print(f"[!] Found {len(train_valid_overlap)} triples in both train and valid - removing from valid")
        valid = list(valid_set - train_valid_overlap)
        
    if train_test_overlap:
        print(f"[!] Found {len(train_test_overlap)} triples in both train and test - removing from test")
        test = list(test_set - train_test_overlap)
        
    if valid_test_overlap:
        print(f"[!] Found {len(valid_test_overlap)} triples in both valid and test - removing from test")
        test = list(test_set - valid_test_overlap)
    
    print(f"[✓] Final split: {len(train_candidates):,} train ({len(train_candidates)/total*100:.1f}%), "
          f"{len(valid):,} valid ({len(valid)/total*100:.1f}%), "
          f"{len(test):,} test ({len(test)/total*100:.1f}%)")
    
    return train_candidates, valid, test

def save_triples(path, triples):
    """Write triples to a file, ensuring no duplicates."""
    # Convert to set to ensure uniqueness
    unique_triples = set(triples)
    if len(unique_triples) != len(triples):
        print(f"[!] Warning: Removed {len(triples) - len(unique_triples)} duplicate triples before saving to {path}")
    
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in unique_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"[✓] Saved {len(unique_triples):,} triples to {path}")

def build_mappings(split_files, ent_out, rel_out):
    """
    Read train/valid/test, collect unique entities & relations,
    assign integer IDs starting from 0, and save mappings.
    """
    print(f"[*] Building entity and relation mappings...")
    
    entity_set = set()
    relation_set = set()
    
    # Collect all unique entities and relations across splits
    for path in split_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                entity_set.update([h, t])
                relation_set.add(r)
    
    # Sort for deterministic ordering and assign IDs
    entity_list = sorted(entity_set)
    relation_list = sorted(relation_set)
    
    # Write entity mappings
    with open(ent_out, "w", encoding="utf-8") as fe:
        for idx, ent in enumerate(entity_list):
            fe.write(f"{ent}\t{idx}\n")
    
    # Write relation mappings
    with open(rel_out, "w", encoding="utf-8") as fr:
        for idx, rel in enumerate(relation_list):
            fr.write(f"{rel}\t{idx}\n")
    
    print(f"[✓] Mapped {len(entity_list):,} entities, {len(relation_list):,} relations")

def main():
    start_time = time.time()
    print(f"[*] Starting knowledge graph processing...")
    
    all_triples = load_triples(INPUT_FILE)
    print(f"[*] Loaded {len(all_triples):,} triples in {time.time()-start_time:.1f}s")
    
    # Prune to approximately 7.5% of original using relation-balanced approach
    pruned = relation_balanced_pruning(all_triples, PRUNE_RATIO)
    
    # Split with controlled visibility (70% seen, 30% unseen)
    train, valid, test = controlled_split(pruned)
    
    # Save splits to files
    save_triples(TRAIN_FILE, train)
    save_triples(VALID_FILE, valid)
    save_triples(TEST_FILE, test)
    
    # Build entity and relation mappings from the split files
    build_mappings(
        [TRAIN_FILE, VALID_FILE, TEST_FILE],
        ENTITIES_FILE,
        RELATIONS_FILE
    )
    
    print(f"[✓] Total processing completed in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()