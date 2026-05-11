#!/usr/bin/env python3
"""
Build KG triples from GeoNames allCountries.txt and featureCodes.txt.

Outputs:
- kg_triples.txt      : Each line: head <TAB> relation <TAB> tail (with underscores)
- entities.txt        : Each line: entity <TAB> id
- relations.txt       : Each line: relation <TAB> id
"""

import csv
import re

# ======= File paths =======
ALL_COUNTRIES_FILE = "allCountries.txt"
FEATURE_CODES_FILE = "featureCodes.txt"
TRIPLE_OUTPUT_FILE = "kg_triples.txt"
ENTITIES_FILE = "entities.txt"
RELATIONS_FILE = "relations.txt"

def normalize(text: str) -> str:
    """Normalize entity/relation name by replacing spaces with underscores."""
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    return text

def load_feature_dict(path: str) -> dict:
    """Return dict mapping 'T.PK' -> 'MountainPeak' (as underscored)."""
    feat2label = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            code, label = parts[0].strip(), parts[1].strip()
            camel = re.sub(r"[^A-Za-z0-9]+", " ", label).title().strip().replace(" ", "_")
            feat2label[code] = camel
    return feat2label  # e.g., {'T.PK': 'MountainPeak'}

def write_triples_and_collect_relations(feature_map):
    """Extract triples and collect all unique entities and relations."""
    out_triples = []
    entity_set = set()
    relation_set = set()

    with open(ALL_COUNTRIES_FILE, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 19:
                continue

            name = normalize(row[1])
            alt_names = [normalize(n) for n in row[3].split(",") if n.strip() and normalize(n) != name]
            feature_key = f"{row[6]}.{row[7]}"
            country = normalize(row[8]) if row[8].strip() else None

            # ---------- Triple: (entity, type, TypeName) ----------
            type_label = feature_map.get(feature_key, normalize(feature_key))
            out_triples.append((name, "type", type_label))
            entity_set.update([name, type_label])
            relation_set.add("type")

            # ---------- Triple: (entity, located_in, CountryCode) ----------
            if country:
                out_triples.append((name, "located_in", country))
                entity_set.update([name, country])
                relation_set.add("located_in")

            # ---------- Triple: (alias, sameAs, entity) ----------
            for alt in alt_names:
                out_triples.append((alt, "sameAs", name))
                entity_set.update([alt, name])
                relation_set.add("sameAs")

            # ---------- Optional: (entity, feature_code, feature_label) ----------
            # e.g., (Roc_Meler, A.ADM1, MountainPeak)
            # you can either use `feature_key` as relation, or `hasFeatureCode`
            if feature_key in feature_map:
                out_triples.append((name, feature_key, type_label))  # use featureKey as relation
                entity_set.update([name, type_label])
                relation_set.add(feature_key)

    # ---------- Write triples ----------
    with open(TRIPLE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for h, r, t in out_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print(f"[✓] Wrote {len(out_triples):,} triples to {TRIPLE_OUTPUT_FILE}")

    return entity_set, relation_set

def save_dict_to_file(items: set, path: str):
    """Write each item in set to file with a unique integer ID."""
    sorted_items = sorted(items)
    with open(path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(sorted_items):
            f.write(f"{item}\t{idx}\n")
    print(f"[✓] Wrote {len(sorted_items)} items to {path}")

def main():
    feature_map = load_feature_dict(FEATURE_CODES_FILE)

if __name__ == "__main__":
    main()
