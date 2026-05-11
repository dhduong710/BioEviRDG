# multi_dataset_train.py

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import logging
from load_data import DataLoader
from base_model import BaseModel
from models import RED_GNN_induc
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser(description="Multi-dataset training and inference for GraphOracle")
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'zero-shot', 'finetune'], 
                       help="Operation mode: train, zero-shot, or finetune")
    
    # Dataset arguments
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help="Paths to datasets (without _ind suffix)")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, 
                       help="Path to pre-trained model for inference modes")
    parser.add_argument('--num_layers', type=int, default=3, 
                       help="Number of layers in relation encoder")
    
    # Training settings
    parser.add_argument('--seed', type=int, default=42, 
                       help="Random seed")
    parser.add_argument('--num_gpus', type=int, default=1, 
                       help="Number of GPUs to use")
    parser.add_argument('--gpu', type=int, default=0, 
                       help="GPU ID for single-GPU operations")
    parser.add_argument('--gpu_ids', nargs='+', type=int,
                       help="Specific GPU IDs to use for multi-GPU training")
    parser.add_argument('--max_epochs', type=int, default=200, 
                       help="Maximum number of epochs")
    parser.add_argument('--early_stopping', type=int, default=10, 
                       help="Number of epochs without improvement before early stopping")
    parser.add_argument('--finetune_epochs', type=int, default=50, 
                       help="Number of epochs for fine-tuning")
    
    # Model saving
    parser.add_argument('--save_model', action='store_true', 
                       help="Save model after training each dataset")
    parser.add_argument('--model_dir', type=str, default='saved_models', 
                       help="Directory to save trained models")
    parser.add_argument('--keep_history', action='store_true',
                       help="Keep separate model files for each dataset in addition to the latest model")
    
    # Data loading options
    parser.add_argument('--rel_single', action='store_true', 
                       help="Use single relations instead of bidirectional")
    parser.add_argument('--rel_reverse_only', action='store_true', 
                       help="Use reverse relations only")
    parser.add_argument('--fact_ratio', type=float, default=0.9, 
                       help="Ratio of facts to training triples")
    parser.add_argument('--remove_1hop_edges', action='store_true', 
                       help="Remove 1-hop edges to prevent leakage")
    
    # Logging
    parser.add_argument('--log_file', type=str, default='red_gnn.log', 
                       help="Log file path")
    parser.add_argument('--num_heads', type=int, default=8, 
                       help="Number of attention heads")
    
    args = parser.parse_args()
    
    # Input validation
    if args.num_gpus > 1 and args.gpu_ids and len(args.gpu_ids) != args.num_gpus:
        parser.error(f"When specifying gpu_ids, please provide exactly {args.num_gpus} GPU IDs")
    
    # Setup logging
    setup_logging(args)
    
    # Run the requested mode
    if args.mode == 'train':
        multi_dataset_train(args)
    elif args.mode == 'zero-shot':
        if not args.model_path:
            logging.error("--model_path is required for zero-shot mode")
            return
        zero_shot_inference(args)
    elif args.mode == 'finetune':
        if not args.model_path:
            logging.error("--model_path is required for finetune mode")
            return
        finetune_inference(args)


def setup_logging(args):
    """Configure logging"""
    logging.basicConfig(
        filename=args.log_file,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def setup(rank, world_size, gpu_ids):
    """
    Setup distributed training environment using specific GPU IDs
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set the device ID based on the provided GPU IDs
    if gpu_ids and rank < len(gpu_ids):
        # Use the specific GPU ID for this rank
        gpu_id = gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
    else:
        # Fallback to using consecutive GPUs
        torch.cuda.set_device(rank)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed process group"""
    dist.destroy_process_group()


def save_model(model, optimizer, scheduler, epoch, dataset_name, args):
    """Save model checkpoint"""
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'last_dataset': dataset_name,  # Store the last dataset it was trained on
    }
    
    # Use a fixed filename - always save to the same file
    save_path = os.path.join(args.model_dir, "latest_model.pt")
    torch.save(checkpoint, save_path)
    
    # Optionally keep a history file if you want to preserve each version
    if args.keep_history:
        history_path = os.path.join(args.model_dir, f"model_{dataset_name}.pt")
        torch.save(checkpoint, history_path)
        logging.info(f"Model saved to {save_path} and history copy to {history_path}")
    else:
        logging.info(f"Model saved to {save_path} (overwriting previous version)")


def load_model(model, optimizer, scheduler, checkpoint_path, device):
    """Load model from checkpoint with security considerations"""
    if os.path.exists(checkpoint_path):
        # Use weights_only=True for better security
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)  # Use get with default to be more robust
        logging.info(f"Loaded model from {checkpoint_path}")
        return start_epoch + 1
    return 0


def get_dataset_hyperparams(dataset_name):
    """Get dataset-specific hyperparameters"""
    opts = argparse.Namespace()
    
    # Dataset-specific hyperparameters
    if dataset_name == 'WN18RR_v1' :
        opts.lr = 0.005
        opts.lamb = 0.0002
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.21
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 100
    elif dataset_name == 'fb237_v1':
        opts.lr = 0.0092
        opts.lamb = 0.0003
        opts.decay_rate = 0.994
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.23
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20
    elif dataset_name == 'nell_v1':
        opts.lr = 0.0021
        opts.lamb = 0.000189
        opts.decay_rate = 0.9937
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2460
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
    elif dataset_name == 'WN18RR_v2':
        opts.lr = 0.0016
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset_name == 'fb237_v2':
        opts.lr = 0.0077
        opts.lamb = 0.0002
        opts.decay_rate = 0.993
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.3
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10
    elif dataset_name == 'nell_v2':
        opts.lr = 0.0075
        opts.lamb = 0.000066
        opts.decay_rate = 0.9996
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.2881
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 100
    elif dataset_name == 'WN18RR_v3':
        opts.lr = 0.0014
        opts.lamb = 0.000034
        opts.decay_rate = 0.991
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.dropout = 0.28
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset_name == 'fb237_v3':
        opts.lr = 0.0006
        opts.lamb = 0.000023
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.27
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 20
    elif dataset_name == 'nell_v3':
        opts.lr = 0.0008
        opts.lamb = 0.0004
        opts.decay_rate = 0.995
        opts.hidden_dim = 16
        opts.attn_dim = 3
        opts.dropout = 0.06
        opts.act = 'relu'
        opts.n_layer = 3
        opts.n_batch = 10
    elif dataset_name == 'WN18RR_v4':
        opts.lr = 0.006
        opts.lamb = 0.000132
        opts.decay_rate = 0.991
        opts.hidden_dim = 32
        opts.attn_dim = 5
        opts.dropout = 0.11
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
    elif dataset_name == 'fb237_v4':
        opts.lr = 0.0052
        opts.lamb = 0.000018
        opts.decay_rate = 0.999
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.dropout = 0.07
        opts.act = 'idd'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset_name == 'nell_v4':
        opts.lr = 0.0005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset_name == 'WN18RR':
        opts.lr = 0.000005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 64
    elif dataset_name == 'fb15k-237':
        opts.lr = 0.000005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 32
    elif dataset_name == 'YAGO':
        opts.lr = 0.000005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 20
    elif dataset_name == 'FB-25':
        opts.lr = 0.0005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 16
        opts.attn_dim = 5
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 5
        opts.n_batch = 24 
    elif dataset_name == 'GeoKG':
        opts.lr = 0.0005
        opts.lamb = 0.000398
        opts.decay_rate = 1
        opts.hidden_dim = 32
        opts.attn_dim = 3
        opts.dropout = 0.1472
        opts.act = 'tanh'
        opts.n_layer = 4
        opts.n_batch = 5 
        opts.n_test_batch = 10
        opts.num_heads = 8
    else:
        # Default hyperparameters
        opts.lr = 0.00016
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 48
        opts.attn_dim = 3
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 5
        opts.n_batch = 10
    
    return opts


def transfer_compatible_params(target_model, source_state_dict, rank=0):
    """
    Transfer parameters that have compatible shapes from source to target model
    """
    # Get current parameters
    target_state_dict = target_model.state_dict()
    
    # Track statistics for logging
    total_params = len(target_state_dict)
    transferred = 0
    
    # Find compatible parameters
    compatible_params = {}
    for name, param in target_state_dict.items():
        if name in source_state_dict and param.shape == source_state_dict[name].shape:
            compatible_params[name] = source_state_dict[name]
            transferred += 1
    
    # Update model with compatible parameters
    target_state_dict.update(compatible_params)
    target_model.load_state_dict(target_state_dict)
    
    # Log transfer statistics
    if rank == 0:
        logging.info(f"Transferred {transferred}/{total_params} parameters from previous model")
        if transferred < total_params:
            logging.info(f"Model architectures differ - transferred only compatible parameters")
    
    return transferred > 0


def train_model(rank, world_size, args, dataset_path, previous_model_state=None):
    """Train model on a specific dataset with DDP support"""
    # Setup process group if using multiple GPUs
    if world_size > 1:
        setup(rank, world_size, args.gpu_ids)
    
    # Extract dataset name
    dataset_parts = dataset_path.split('/')
    dataset_name = dataset_parts[-1] if dataset_parts[-1] else dataset_parts[-2]
    
    # Set device for this process - use specific GPU if provided
    if world_size > 1 and args.gpu_ids and rank < len(args.gpu_ids):
        device = torch.device(f"cuda:{args.gpu_ids[rank]}")
    else:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create results directory
    results_dir = 'results'
    if rank == 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize data loader
    loader = DataLoader(
        dataset_path,
        args.rel_single,
        args.rel_reverse_only,
        args.fact_ratio,
        args.remove_1hop_edges
    )
    
    # Get hyperparameters for this dataset
    opts = get_dataset_hyperparams(dataset_name)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.num_layers = args.num_layers
    opts.perf_file = os.path.join(results_dir, f"{dataset_name}_perf.txt")
    
    # Initialize the model
    base_model = BaseModel(opts, loader)
    base_model.model.to(device)
    
    # Load previous model state if provided
    if previous_model_state is not None:
        # Transfer compatible parameters
        if rank == 0:
            logging.info(f"Attempting to transfer compatible parameters from previous model")
        
        transfer_compatible_params(base_model.model, previous_model_state, rank)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(base_model.model, device_ids=[args.gpu_ids[rank]] if args.gpu_ids and rank < len(args.gpu_ids) else None)
    else:
        model = base_model.model
    
    # Print configuration (only from rank 0)
    if rank == 0:
        config_str = (f'Dataset: {dataset_name}, LR: {opts.lr:.6f}, Decay: {opts.decay_rate:.6f}, '
                    f'Lambda: {opts.lamb:.6f}, Hidden: {opts.hidden_dim}, Attn: {opts.attn_dim}, '
                    f'Layers: {opts.n_layer}, Batch: {opts.n_batch}, Dropout: {opts.dropout:.4f}, '
                    f'Act: {opts.act}, Fact ratio: {args.fact_ratio:.2f}')
        logging.info(config_str)
        
        if world_size > 1:
            gpu_str = f"Using GPUs: {args.gpu_ids if args.gpu_ids else list(range(world_size))}"
            logging.info(gpu_str)
    
    # Create progress bar for epochs (only on rank 0)
    if rank == 0:
        epoch_pbar = tqdm(total=args.max_epochs, desc=f"Training on {dataset_name}", 
                         position=0, leave=True, dynamic_ncols=True)
    
    # Training loop with early stopping
    best_mrr = 0
    best_epoch = 0
    best_str = ""
    early_stop_counter = 0
    
    for epoch in range(args.max_epochs):
        # Train for one epoch - FIXED: removed the rank parameter
        t_mrr, out_str = base_model.train_batch(model if world_size > 1 else None)
        
        # Shuffle training data after each epoch
        loader.shuffle_train()
        
        # Log performance (only from rank 0)
        if rank == 0:
            with open(opts.perf_file, 'a+') as f:
                f.write(f"Epoch {epoch}: {out_str}")
        
        # Check for improvement and handle early stopping
        if t_mrr > best_mrr:
            best_mrr = t_mrr
            best_str = out_str
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save best model (only from rank 0)
            if rank == 0 and args.save_model:
                save_model(base_model.model, base_model.optimizer, 
                          base_model.scheduler, epoch, dataset_name, args)
            
            if rank == 0:
                logging.info(f"Epoch {epoch}: New best performance - {best_str.strip()}")
        else:
            early_stop_counter += 1
            if rank == 0:
                logging.info(f"Epoch {epoch}: No improvement, counter: {early_stop_counter}/{args.early_stopping}")
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'MRR': f"{t_mrr:.4f}", 
                'best_MRR': f"{best_mrr:.4f}",
                'counter': f"{early_stop_counter}/{args.early_stopping}"
            })
        
        # Synchronize early stopping across processes
        if world_size > 1:
            early_stop_tensor = torch.tensor([early_stop_counter], device=device)
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            early_stop_counter = early_stop_tensor.item()
        
        if early_stop_counter > args.early_stopping:
            if rank == 0:
                logging.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Close progress bar
    if rank == 0:
        epoch_pbar.close()
        logging.info(f"Best performance on {dataset_name} (epoch {best_epoch}): {best_str.strip()}")
    
    # Get best model state dict for next dataset
    best_model_state = None
    if args.save_model:
        checkpoint_path = os.path.join(args.model_dir, "latest_model.pt")
        if os.path.exists(checkpoint_path):
            # Use weights_only=True for security
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            best_model_state = checkpoint['model_state_dict']
    else:
        # If not saving, use the current model state
        best_model_state = base_model.model.state_dict()
    
    # Clean up process group
    if world_size > 1:
        cleanup()
    
    return best_model_state


def multi_dataset_train(args):
    """Train model sequentially on multiple datasets with multi-GPU support"""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create model directory if saving
    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    logging.info(f"Starting sequential training on {len(args.datasets)} datasets")
    
    # Fixed model path for sequential loading
    fixed_model_path = os.path.join(args.model_dir, "latest_model.pt")
    
    # Create overall progress bar for datasets
    dataset_pbar = tqdm(total=len(args.datasets), desc="Datasets Progress", 
                       position=0, leave=True, dynamic_ncols=True)
    
    # Train sequentially on each dataset
    previous_model_state = None
    for dataset_idx, dataset_path in enumerate(args.datasets):
        logging.info(f"Training on dataset {dataset_idx+1}/{len(args.datasets)}: {dataset_path}")
        
        # Load latest model if it exists and this isn't the first dataset
        if dataset_idx > 0 and os.path.exists(fixed_model_path) and args.save_model:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(fixed_model_path, map_location=device, weights_only=True)
            previous_model_state = checkpoint['model_state_dict']
            last_dataset = checkpoint.get('last_dataset', 'unknown')
            logging.info(f"Loaded model trained on {last_dataset} for continuing training")
        
        # Train on this dataset with multi-GPU support if requested
        if args.num_gpus > 1:
            # Use multiprocessing to launch multiple GPU processes
            mp.spawn(
                train_model,
                args=(args.num_gpus, args, dataset_path, previous_model_state),
                nprocs=args.num_gpus,
                join=True
            )
            
            # Load the updated model for the next dataset if we're saving models
            if os.path.exists(fixed_model_path) and args.save_model:
                device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(fixed_model_path, map_location=device, weights_only=True)
                previous_model_state = checkpoint['model_state_dict']
        else:
            # Single GPU training
            previous_model_state = train_model(args.gpu, 1, args, dataset_path, previous_model_state)
        
        # Update dataset progress bar
        dataset_pbar.update(1)
        dataset_name = dataset_path.split('/')[-1] if dataset_path.split('/')[-1] else dataset_path.split('/')[-2]
        dataset_pbar.set_description(f"Completed {dataset_idx+1}/{len(args.datasets)}: {dataset_name}")
    
    # Close progress bar
    dataset_pbar.close()
    logging.info("Sequential training complete!")


def zero_shot_inference(args):
    """Perform zero-shot inference using a trained model"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        return
    
    logging.info(f"Performing zero-shot inference with model: {args.model_path}")
    
    # Load the original model
    # Use weights_only=True for security
    orig_checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    orig_state_dict = orig_checkpoint['model_state_dict']
    
    # Create progress bar for datasets
    dataset_pbar = tqdm(total=len(args.datasets), desc="Zero-shot Inference", 
                       position=0, leave=True, dynamic_ncols=True)
    
    # Process each dataset
    for dataset_idx, dataset_path in enumerate(args.datasets):
        # Extract dataset name
        dataset_parts = dataset_path.split('/')
        dataset_name = dataset_parts[-1] if dataset_parts[-1] else dataset_parts[-2]
        
        logging.info(f"Zero-shot inference on dataset: {dataset_name}")
        dataset_pbar.set_description(f"Zero-shot on {dataset_name}")
        
        # Initialize data loader
        loader = DataLoader(
            dataset_path,
            args.rel_single,
            args.rel_reverse_only,
            args.fact_ratio,
            args.remove_1hop_edges
        )
        
        # Get hyperparameters for this dataset
        opts = get_dataset_hyperparams(dataset_name)
        opts.n_ent = loader.n_ent
        opts.n_rel = loader.n_rel
        opts.num_layers = args.num_layers
        
        # Initialize model
        base_model = BaseModel(opts, loader)
        base_model.model.to(device)
        
        # Try to transfer parameters from the original model
        success = transfer_compatible_params(base_model.model, orig_state_dict)
        
        if not success:
            logging.warning(f"Model architectures are too different for zero-shot inference on {dataset_name}")
            dataset_pbar.update(1)
            continue
        
        # Evaluate - FIXED: removed rank parameter
        v_mrr, t_mrr, out_str = base_model.evaluate()
        logging.info(f"Zero-shot results on {dataset_name}: {out_str.strip()}")
        dataset_pbar.update(1)
    
    dataset_pbar.close()


def finetune_inference(args):
    """Fine-tune a pre-trained model on new datasets without saving"""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found: {args.model_path}")
        return
    
    logging.info(f"Fine-tuning model from: {args.model_path}")
    
    # Load the model once to get its state
    # Use weights_only=True for security
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model_state = checkpoint['model_state_dict']
    
    # Create progress bar for datasets
    dataset_pbar = tqdm(total=len(args.datasets), desc="Fine-tuning Datasets", 
                       position=0, leave=True, dynamic_ncols=True)
    
    # Process each dataset
    for dataset_idx, dataset_path in enumerate(args.datasets):
        # Extract dataset name
        dataset_parts = dataset_path.split('/')
        dataset_name = dataset_parts[-1] if dataset_parts[-1] else dataset_parts[-2]
        
        logging.info(f"Fine-tuning on dataset: {dataset_name}")
        dataset_pbar.set_description(f"Fine-tuning on {dataset_name}")
        
        # Initialize data loader
        loader = DataLoader(
            dataset_path,
            args.rel_single,
            args.rel_reverse_only,
            args.fact_ratio,
            args.remove_1hop_edges
        )
        
        # Get hyperparameters for this dataset
        opts = get_dataset_hyperparams(dataset_name)
        opts.n_ent = loader.n_ent
        opts.n_rel = loader.n_rel
        opts.num_layers = args.num_layers
        
        # Initialize model
        base_model = BaseModel(opts, loader)
        base_model.model.to(device)
        
        # Try to transfer parameters
        success = transfer_compatible_params(base_model.model, model_state)
        
        if not success:
            logging.warning(f"Model architectures are too different for fine-tuning on {dataset_name}")
            dataset_pbar.update(1)
            continue
        
        # Create progress bar for fine-tuning epochs
        epoch_pbar = tqdm(total=args.finetune_epochs, desc=f"Fine-tuning epochs for {dataset_name}", 
                         position=1, leave=False, dynamic_ncols=True)
        
        # Fine-tune the model for specified epochs
        logging.info(f"Fine-tuning for {args.finetune_epochs} epochs...")
        best_mrr = 0
        best_str = ""
        
        for epoch in range(args.finetune_epochs):
            # FIXED: removed rank parameter
            t_mrr, out_str = base_model.train_batch()
            loader.shuffle_train()
            
            if t_mrr > best_mrr:
                best_mrr = t_mrr
                best_str = out_str
                epoch_pbar.set_postfix({'MRR': f"{t_mrr:.4f}", 'best_MRR': f"{best_mrr:.4f}"})
                logging.info(f"Epoch {epoch}: {best_str.strip()}")
            
            epoch_pbar.update(1)
        
        epoch_pbar.close()
        
        # Evaluate after fine-tuning (no rank parameter)
        v_mrr, t_mrr, out_str = base_model.evaluate()
        logging.info(f"Fine-tuned results on {dataset_name}: {out_str.strip()}")
        dataset_pbar.update(1)
    
    dataset_pbar.close()





if __name__ == "__main__":
    main()