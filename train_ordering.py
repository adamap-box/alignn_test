"""
Training script for mp_ordering prediction using ALIGNN.

This script trains a model to predict magnetic ordering (mp_ordering) 
from crystal structures using prepared train/val/test JSON files.

mp_ordering classes:
- 0: NM (Non-Magnetic)
- 1: FM (Ferromagnetic)  
- 2: AFM (Antiferromagnetic)
- 3: FiM (Ferrimagnetic)
"""

import os
import sys

# Add alignn package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))

import json
import time
import random
import argparse
from functools import partial
from typing import Any, Dict, Union

import torch
import numpy as np
from torch import nn
from sklearn.metrics import (
    mean_absolute_error, 
    accuracy_score, 
    classification_report,
    confusion_matrix,
    f1_score,
)

from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.config import TrainingConfig
from data_prepared import get_prepared_train_val_test_loaders


def dumpjson(data=[], filename=""):
    """Write data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def compute_metrics(targets, predictions, num_classes=4):
    """Compute classification metrics."""
    acc = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def train_ordering(
    config: Union[TrainingConfig, Dict[str, Any]],
    train_loader,
    val_loader,
    test_loader,
    prepare_batch,
    model: nn.Module = None,
    num_classes: int = 4,
):
    """Training function for mp_ordering classification."""
    
    print("=" * 60)
    print("Training mp_ordering classification model")
    print("=" * 60)
    
    # Process config - create a modified config for TrainingConfig validation
    if isinstance(config, dict):
        try:
            print(json.dumps(config, indent=2))
            # Keep original config for our use
            original_config = config.copy()
            # Modify config to pass TrainingConfig validation
            config_for_validation = config.copy()
            config_for_validation["dataset"] = "user_data"  # Valid dataset type
            config_for_validation["target"] = "target"  # Valid target type
            config = TrainingConfig(**config_for_validation)
            # Restore our custom values for reference
            config.dataset = original_config.get("dataset", "prepared")
            config.target = original_config.get("target", "mp_ordering")
        except Exception as exp:
            print("Config error:", exp)
            raise
    
    # Create output directory
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=4)
    
    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print(f"Using device: {device}")
        print("WARNING: CUDA not available, training will be slow!")
    
    # Line graph check
    line_graph = config.model.alignn_layers > 0
    
    # Set random seed for reproducibility
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    
    # Prepare batch function
    prepare_batch = partial(prepare_batch, device=device)
    
    # Model setup
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
    }
    
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    
    # Load existing model if available
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading model from: {best_model_path}")
        net.load_state_dict(torch.load(best_model_path, map_location=device))
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
    net.to(device)
    
    # Optimizer setup
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    
    # Scheduler setup
    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    
    # Loss function - L1Loss for regression (then round predictions to get classes)
    # ALIGNN outputs 1 value per sample, so we treat this as regression
    criterion = nn.L1Loss()
    
    # Training history
    history_train = []
    history_val = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"\nStarting training for {config.epochs} epochs")
    print("-" * 60)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Training phase
        net.train()
        running_loss = 0.0
        train_targets = []
        train_preds = []
        
        for batch_idx, (dats, jid) in enumerate(zip(train_loader, train_loader.dataset.ids)):
            optimizer.zero_grad()
            
            # Use prepare_batch to move data to device (device already bound via partial)
            batch = prepare_batch(dats)
            
            if line_graph:
                g, lg = batch[0]
                result = net([g, lg])
                targets = batch[1].float()  # Use float for regression
            else:
                g = batch[0]
                result = net(g)
                targets = batch[1].float()
            
            # Get predictions and targets
            # ALIGNN outputs shape [batch_size] with regression values
            outputs = result["out"].squeeze()
            
            # Debug shape on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"  DEBUG: outputs shape = {outputs.shape}, targets shape = {targets.shape}")
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Convert regression outputs to class predictions (0, 1, 2, 3)
            # Round and clamp to valid class range
            preds = torch.clamp(torch.round(outputs), 0, num_classes - 1).long().cpu().numpy()
            train_targets.extend(targets.long().cpu().numpy())
            train_preds.extend(preds)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_metrics = compute_metrics(train_targets, train_preds, num_classes)
        
        history_train.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_metrics['accuracy'],
            'f1_macro': train_metrics['f1_macro'],
        })
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        val_results = []
        
        with torch.no_grad():
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                # Use prepare_batch to move data to device (device already bound via partial)
                batch = prepare_batch(dats)
                
                if line_graph:
                    g, lg = batch[0]
                    result = net([g, lg])
                    targets = batch[1].float()
                else:
                    g = batch[0]
                    result = net(g)
                    targets = batch[1].float()
                
                outputs = result["out"].squeeze()
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Convert regression outputs to class predictions
                preds = torch.clamp(torch.round(outputs), 0, num_classes - 1).long().cpu().numpy()
                val_targets.extend(targets.long().cpu().numpy())
                val_preds.extend(preds)
                
                # Store individual results
                for t, p in zip(targets.long().cpu().numpy(), preds):
                    val_results.append({
                        'id': jid,
                        'target': int(t),
                        'prediction': int(p),
                    })
        
        val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(val_targets, val_preds, num_classes)
        
        history_val.append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_metrics['accuracy'],
            'f1_macro': val_metrics['f1_macro'],
        })
        
        # Save current model
        torch.save(
            net.state_dict(),
            os.path.join(config.output_dir, "current_model.pt"),
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, "best_model.pt"),
            )
            dumpjson(
                data=val_results,
                filename=os.path.join(config.output_dir, "Val_results.json"),
            )
            print(f"  ** New best model saved (val_loss: {val_loss:.4f}, "
                  f"val_acc: {val_metrics['accuracy']:.4f}) **")
        
        # Save training history
        dumpjson(
            data=history_train,
            filename=os.path.join(config.output_dir, "history_train.json"),
        )
        dumpjson(
            data=history_val,
            filename=os.path.join(config.output_dir, "history_val.json"),
        )
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f} | "
              f"Time: {epoch_time:.1f}s")
    
    # Load best model for testing
    print("\n" + "=" * 60)
    print("Evaluating on test set with best model")
    print("=" * 60)
    
    net.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    net.eval()
    
    test_targets = []
    test_preds = []
    test_results = []
    test_probs = []
    
    with torch.no_grad():
        for dats, jid in zip(test_loader, test_loader.dataset.ids):
            # Use prepare_batch to move data to device (device already bound via partial)
            batch = prepare_batch(dats)
            
            if line_graph:
                g, lg = batch[0]
                result = net([g, lg])
                targets = batch[1].float()
            else:
                g = batch[0]
                result = net(g)
                targets = batch[1].float()
            
            outputs = result["out"].squeeze()
            
            # Ensure proper shapes for single sample batches
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            
            # Convert regression outputs to class predictions
            preds = torch.clamp(torch.round(outputs), 0, num_classes - 1).long().cpu().numpy()
            raw_outputs = outputs.cpu().numpy()
            
            test_targets.extend(targets.long().cpu().numpy())
            test_preds.extend(preds)
            test_probs.extend(raw_outputs.tolist() if hasattr(raw_outputs, 'tolist') else [raw_outputs])
            
            for t, p, raw in zip(targets.long().cpu().numpy(), preds, raw_outputs):
                test_results.append({
                    'id': jid,
                    'target': int(t),
                    'prediction': int(p),
                    'raw_output': float(raw),
                })
    
    # Compute test metrics
    test_metrics = compute_metrics(test_targets, test_preds, num_classes)
    
    # Class names for better reporting
    class_names = ['NM', 'FM', 'AFM', 'FiM'][:num_classes]
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_preds, 
                                target_names=class_names, 
                                zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    # Save test results
    dumpjson(
        data=test_results,
        filename=os.path.join(config.output_dir, "Test_results.json"),
    )
    
    # Save test metrics summary
    test_summary = {
        'accuracy': test_metrics['accuracy'],
        'f1_macro': test_metrics['f1_macro'],
        'f1_weighted': test_metrics['f1_weighted'],
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            test_targets, test_preds, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),
    }
    dumpjson(
        data=test_summary,
        filename=os.path.join(config.output_dir, "test_metrics.json"),
    )
    
    # Save last model
    torch.save(
        net.state_dict(),
        os.path.join(config.output_dir, "last_model.pt"),
    )
    
    # Write CSV prediction results
    with open(os.path.join(config.output_dir, "prediction_results_test_set.csv"), "w") as f:
        f.write("id,target,prediction,correct\n")
        for r in test_results:
            correct = 1 if r['target'] == r['prediction'] else 0
            f.write(f"{r['id']},{r['target']},{r['prediction']},{correct}\n")
    
    print(f"\nResults saved to {config.output_dir}")
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
    }


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train ALIGNN model for mp_ordering prediction"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_ordering.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="prepared_data_merged/train_data.json",
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="prepared_data_merged/val_data.json",
        help="Path to validation data JSON"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="prepared_data_merged/test_data.json",
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_ordering",
        help="Output directory for results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of classification classes"
    )
    
    args = parser.parse_args()
    
    # Default configuration for mp_ordering classification
    config = {
        "version": "1.0",
        "dataset": "prepared",
        "target": "mp_ordering",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "id",
        "random_seed": 123,
        "classification_threshold": None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": 1e-05,
        "learning_rate": args.learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "save_dataloader": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "num_workers": 0,
        "cutoff": 8.0,
        "cutoff_extra": 3.0,
        "max_neighbors": 12,
        "keep_data_order": False,
        "output_dir": args.output_dir,
        "model": {
            "name": "alignn_atomwise",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,  # Single output for regression
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "atomwise_output_features": 0,
            "calculate_gradient": False,
            "classification": False,  # Use regression approach
        }
    }
    
    # Load config from file if provided
    if os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Override with command line arguments
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["output_dir"] = args.output_dir
    # Keep output_features as 1 for regression
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, prepare_batch = get_prepared_train_val_test_loaders(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        target=config["target"],
        atom_features=config["atom_features"],
        neighbor_strategy=config["neighbor_strategy"],
        batch_size=config["batch_size"],
        line_graph=config["model"]["alignn_layers"] > 0,
        id_tag=config["id_tag"],
        use_canonize=config.get("use_canonize", True),
        cutoff=config["cutoff"],
        cutoff_extra=config["cutoff_extra"],
        max_neighbors=config["max_neighbors"],
        classification=False,  # Use regression approach
        output_dir=config["output_dir"],
        workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    
    # Train model
    result = train_ordering(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        prepare_batch=prepare_batch,
        num_classes=args.num_classes,
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
