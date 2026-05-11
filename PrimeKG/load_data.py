import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir, rel_single=False, rel_reverse_only=False, fact_ratio=0.9, remove_1hop_edges=False):
        # We only need the inductive directory
        self.ind_dir = task_dir + '_ind'
        self.rel_single = rel_single
        self.rel_reverse_only = rel_reverse_only
        self.fact_ratio = fact_ratio
        self.remove_1hop_edges = remove_1hop_edges

        # Read inductive entities and relations
        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)

        with open(os.path.join(self.ind_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            n_rel = 0
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)
                n_rel += 1

            # Add inverse and self-loop relations
            for i in range(n_rel):
                id2relation.append(id2relation[i] + '_inv')
            id2relation.append('idd')
            self.id2relation = id2relation

            # Total relations count = original + inverse + self-loop
            self.total_relations = 2 * n_rel + 1

        self.n_ent = len(self.entity2id)
        self.n_rel = n_rel
        self.n_ent_ind = self.n_ent  # Same as n_ent since we only use inductive mode

        # Read inductive data only
        self.ind_train_original = self.read_triples('train.txt')
        self.ind_valid = self.read_triples('valid.txt')
        self.ind_test = self.read_triples('test.txt')

        # Store all triples for shuffle_train
        self.all_triples = np.array(self.ind_train_original)

        # Initialize train data with original triples
        self.ind_train = self.ind_train_original

        # Create filters for evaluation
        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        # Convert sets to lists for easier handling
        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        # Build relation graphs BEFORE calling shuffle_train
        self.train_relation_edges = self._build_relation_graph(self.ind_train)
        self.valid_relation_edges = self._build_relation_graph(self.ind_train_original)
        self.test_relation_edges = self._build_relation_graph(self.ind_train_original + self.ind_valid)
        self.relation_edges = self.train_relation_edges

        # AFTER setting up relation_edges, call shuffle_train
        self.shuffle_train()

        # Load graphs for each mode
        self.train_KG, self.train_sub = self.load_graph(self.ind_train, 'train')
        self.valid_KG, self.valid_sub = self.load_graph(self.ind_train_original, 'valid')
        self.test_KG, self.test_sub = self.load_graph(self.ind_train_original + self.ind_valid, 'test')

        # Prepare validation and test data
        self.valid_q, self.valid_a = self.load_query(self.ind_valid)
        self.test_q, self.test_a = self.load_query(self.ind_test)

        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        """Read triples from the specified file in inductive directory"""
        triples = []
        with open(os.path.join(self.ind_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h, r, t])
                triples.append([t, r + self.n_rel, h])  # Add inverse relation
        return triples

    def _build_relation_graph(self, triples):
        """Build relation graph based on given triples"""
        entity_relations = defaultdict(lambda: {'in': set(), 'out': set()})

        # Collect relation patterns
        for h, r, t in triples:
            entity_relations[t]['in'].add(r)
            entity_relations[h]['out'].add(r)

        # Build relation edges
        edges = set()
        for e in entity_relations:
            for r_in in entity_relations[e]['in']:
                for r_out in entity_relations[e]['out']:
                    if not self.rel_reverse_only:
                        edges.add((r_in, r_out))
                    if (not self.rel_single) or self.rel_reverse_only:
                        edges.add((r_out, r_in))  # Bidirectional connection

        # Convert to tensor format and add self-loops
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
                                                                                                            dtype=torch.long)
        loop_index = torch.tensor([[i, i] for i in range(self.total_relations)], dtype=torch.long).t()

        # Combine edges and remove duplicates
        if edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            return edge_index.unique(dim=1)
        else:
            return loop_index

    def set_mode(self, mode):
        """Set the current mode and update relation edges accordingly"""
        if mode == 'train':
            self.relation_edges = self.train_relation_edges
        elif mode == 'valid':
            self.relation_edges = self.valid_relation_edges
        elif mode == 'test':
            self.relation_edges = self.test_relation_edges
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return self.relation_edges

    def load_graph(self, triples, mode):
        """Load graph for specified mode"""
        KG = np.array(triples)
        # Add self-loops for entities
        idd = np.concatenate([
            np.expand_dims(np.arange(self.n_ent), 1),
            2 * self.n_rel * np.ones((self.n_ent, 1)),
            np.expand_dims(np.arange(self.n_ent), 1)
        ], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]
        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])), shape=(n_fact, self.n_ent))
        return KG, M_sub

    def load_query(self, triples):
        """Prepare query-answer pairs from triples"""
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        """Get neighbor nodes based on current mode"""
        if mode == 'train':
            KG = self.train_KG
            M_sub = self.train_sub
        elif mode == 'valid':
            KG = self.valid_KG
            M_sub = self.valid_sub
        elif mode == 'test' or mode == 'inductive':  # Keep 'inductive' for compatibility
            KG = self.test_KG
            M_sub = self.test_sub
        else:
            raise ValueError(f"Unknown mode: {mode}")

        n_ent = self.n_ent

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # Index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, data='train'):
        """Get batch of data based on current mode"""
        if data == 'train':
            return self.train_data[batch_idx]

        if data == 'valid':
            query = np.array(self.valid_q)
            answer = self.valid_a
        elif data == 'test':
            query = np.array(self.test_q)
            answer = self.test_a
        else:
            raise ValueError(f"Unknown data mode: {data}")

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))

        # Process each answer separately since they may have different lengths
        for i, idx in enumerate(batch_idx):
            objs[i][answer[idx]] = 1

        return subs, rels, objs

    def shuffle_train(self):
        """
        Shuffle training data and split into fact and train data based on fact_ratio.
        - fact_data: Used to build the knowledge graph (KG)
        - train_data: Used as training queries for model training
        """
        # Shuffle all triples
        n_all = len(self.all_triples)
        rand_idx = np.random.permutation(n_all)
        self.all_triples = self.all_triples[rand_idx]

        # Split data based on fact_ratio
        bar = int(n_all * self.fact_ratio)
        fact_data = self.all_triples[:bar].tolist()
        train_data = self.all_triples[bar:].tolist()

        # Handle potential 1-hop edge removal if requested
        if self.remove_1hop_edges:
            print('==> removing 1-hop links...')
            tmp_index = np.ones((self.n_ent, self.n_ent))

            # Mark edges in training data
            for h, r, t in train_data:
                tmp_index[h, t] = 0

            # Filter fact data to remove edges that would create 1-hop paths
            filtered_fact_data = []
            for h, r, t in fact_data:
                if tmp_index[h, t] == 1:  # Only keep edges not present in training data
                    filtered_fact_data.append([h, r, t])

            fact_data = filtered_fact_data
            print('==> done removing 1-hop links')

        # Key fix: Use only fact_data for KG building and train_data for training
        self.ind_train = fact_data  # Using only fact_data to build the KG
        self.train_data = np.array(train_data)  # Using only train_data for training
        self.n_train = len(self.train_data)

        # Rebuild the training graph using only fact_data
        self.train_KG, self.train_sub = self.load_graph(self.ind_train, 'train')
        self.train_relation_edges = self._build_relation_graph(self.ind_train)
        if self.relation_edges is self.train_relation_edges:
            self.relation_edges = self.train_relation_edges


    def get_filter(self, data='valid'):
        """Get filters for evaluation"""
        filters = defaultdict(lambda: set())
        if data == 'valid':
            # For validation, add train and valid answers
            for triple in self.ind_train_original:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
        else:  # test
            # For testing, add train, valid, and test answers
            for triple in self.ind_train_original:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        return filters