# base_model.py

import torch
import numpy as np
import time
import logging

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import RED_GNN_induc
from utils import cal_ranks, cal_performance
from tqdm import tqdm


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = RED_GNN_induc(args, loader)
        # Don't move to CUDA here - will be done when training
        
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_batch = args.n_batch
        self.n_test_batch = args.n_test_batch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0 

    def train_batch(self, ddp_model=None):
        """
        Train for one epoch - supports both standard and DDP models
        """
        epoch_loss = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        t_time = time.time()
        
        # Use DDP model if provided, otherwise use regular model
        model_to_train = ddp_model if ddp_model is not None else self.model
        model_to_train.train()
        
        # Show progress bar
        iterator = tqdm(range(n_batch), desc="Training Batches", 
                      position=1, leave=False, dynamic_ncols=True)
        
        for i in iterator:
            start = i * batch_size
            end = min(self.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.optimizer.zero_grad()
            
            # Forward pass in train mode
            scores = model_to_train(triple[:, 0], triple[:, 1], mode='train')

            # Get device for tensor operations
            device = scores.device
            
            # Calculate loss
            pos_scores = scores[torch.arange(len(scores), device=device), 
                               torch.tensor(triple[:, 2], dtype=torch.long, device=device)]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + 
                           torch.log(torch.sum(torch.exp(scores - max_n), 1)))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Avoid NaNs in parameters
            for p in model_to_train.parameters():
                X = p.data.clone()
                flag = X != X  # Find NaN values
                X[flag] = np.random.random()
                p.data.copy_(X)
                
            epoch_loss += loss.item()
            
            # Update progress bar
            iterator.set_postfix({'loss': f"{loss.item():.4f}"})
        
        self.scheduler.step()
        self.t_time += time.time() - t_time

        # Run evaluation
        v_mrr, t_mrr, out_str = self.evaluate(model_to_train)
        return t_mrr, out_str

    def evaluate(self, model_to_eval=None):
        """
        Evaluate on validation and test sets
        """
        batch_size =  self.n_test_batch
        
        # Use provided model or default model
        model = model_to_eval if model_to_eval is not None else self.model
        
        # Switch to evaluation mode
        model.eval()
        i_time = time.time()

        # Validation phase
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []

        # Show progress bar for validation
        iterator = tqdm(range(n_batch), desc="Validating",
                      position=1, leave=False, dynamic_ncols=True)

        for i in iterator:
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')

            # Forward pass in valid mode with no gradient tracking
            with torch.no_grad():
                scores = model(subs, rels, mode='valid').cpu().numpy()

            # Prepare filters for evaluation
            filters = []
            for j in range(len(subs)):
                filt = self.loader.val_filters[(subs[j], rels[j])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks

        ranking = np.array(ranking)
        v_mrr, v_h1, v_h10 = cal_performance(ranking)

        # Testing phase
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []

        # Show progress bar for testing
        iterator = tqdm(range(n_batch), desc="Testing",
                      position=1, leave=False, dynamic_ncols=True)

        for i in iterator:
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')

            # Forward pass in test mode with no gradient tracking
            with torch.no_grad():
                scores = model(subs, rels, mode='test').cpu().numpy()

            # Prepare filters for evaluation
            filters = []
            for j in range(len(subs)):
                filt = self.loader.tst_filters[(subs[j], rels[j])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks

        ranking = np.array(ranking)
        t_mrr, t_h1, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n' % (
            v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)

        return v_mrr, t_mrr, out_str