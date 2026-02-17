"""
Training script for mp_ordering classification using ALIGNN with CrossEntropyLoss.

This script trains a model to predict magnetic ordering (mp_ordering) 
from crystal structures using proper classification with class weights
to handle imbalanced data (especially AFM minority class).

mp_ordering classes:
- 0: NM (Non-Magnetic) - ~59.5%
- 1: FM (Ferromagnetic) - ~32.0%
- 2: AFM (Antiferromagnetic) - ~1.9% (minority class!)
- 3: FiM (Ferrimagnetic) - ~6.6%

Key improvements over regression approach:
1. CrossEntropyLoss with inverse frequency class weights
2. 4 output neurons (one per class) with softmax
3. Optional Focal Loss for hard examples
4. Class-balanced sampling option
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
from typing import Any, Dict, Union, Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from sklearn.metrics import (
    mean_absolute_error, 
    accuracy_score, 
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from collections import Counter

from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.config import TrainingConfig
from data_prepared import get_prepared_train_val_test_loaders


# Class labels
CLASS_NAMES = ['NM', 'FM', 'AFM', 'FiM']


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    This down-weights easy examples and focuses on hard negatives.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model outputs (logits) [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t for correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassificationHead(nn.Module):
    """
    Classification head to replace ALIGNN's regression output.
    Takes the embedding and outputs class logits.
    """
    def __init__(self, in_features: int, num_classes: int, hidden_features: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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


def compute_class_weights(train_loader, num_classes: int = 4, device: str = 'cpu') -> torch.Tensor:
    """
    Compute class weights based on inverse frequency.
    
    Returns: tensor of weights for each class
    """
    class_counts = Counter()
    
    # Access targets directly from the dataset's labels
    dataset = train_loader.dataset
    
    # Try different ways to access labels depending on dataset structure
    if hasattr(dataset, 'labels') and dataset.labels is not None:
        # StructureDataset stores labels as pandas Series or similar
        labels = dataset.labels
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif hasattr(labels, 'values'):
            # pandas Series
            labels = labels.values
        for t in labels:
            class_counts[int(t)] += 1
    elif hasattr(dataset, 'df') and dataset.target in dataset.df.columns:
        # Access from dataframe using target column name
        for t in dataset.df[dataset.target].values:
            class_counts[int(t)] += 1
    else:
        # Fallback: iterate through dataset indices
        for i in range(len(dataset)):
            item = dataset[i]
            # item is typically (graph_data, target, id)
            if isinstance(item, tuple) and len(item) >= 2:
                target = item[1]
                if isinstance(target, torch.Tensor):
                    target = target.item() if target.dim() == 0 else target[0].item()
                class_counts[int(target)] += 1
    
    total = sum(class_counts.values())
    
    if total == 0:
        print("WARNING: No class counts found, using uniform weights")
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    else:
        # Compute inverse frequency weights, normalized
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            # Inverse frequency, then normalize so mean weight is 1
            weight = total / (num_classes * count)
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    print(f"\nClass counts: {dict(class_counts)}")
    print(f"Class weights: {weights.cpu().numpy()}")
    
    return weights


def compute_metrics(targets, predictions, num_classes=4):
    """Compute classification metrics."""
    acc = accuracy_score(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist(),
    }


def train_ordering_classification(
    config: Union[TrainingConfig, Dict[str, Any]],
    train_loader,
    val_loader,
    test_loader,
    prepare_batch,
    model: nn.Module = None,
    num_classes: int = 4,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    class_weights: Optional[List[float]] = None,
):
    """Training function for mp_ordering classification with CrossEntropyLoss."""
    
    print("=" * 60)
    print("Training mp_ordering CLASSIFICATION model")
    print("Using CrossEntropyLoss with class weights")
    print("=" * 60)
    
    # Process config
    if isinstance(config, dict):
        try:
            print(json.dumps(config, indent=2))
            original_config = config.copy()
            config_for_validation = config.copy()
            config_for_validation["dataset"] = "user_data"
            config_for_validation["target"] = "target"
            config = TrainingConfig(**config_for_validation)
            config.dataset = original_config.get("dataset", "prepared")
            config.target = original_config.get("target", "mp_ordering")
            # Force output_features for classification
            # NOTE: classification=False so ALIGNN uses output_features instead of hardcoding 1
            config.model.output_features = num_classes
            config.model.classification = False
        except Exception as exp:
            print("Config error:", exp)
            raise
    
    # Ensure model has correct output features for classification
    # NOTE: ALIGNN's classification=True forces 1 output (binary), so we use False for multi-class
    config.model.output_features = num_classes
    config.model.classification = False
    print(f"Model output_features set to: {config.model.output_features}")
    
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
    
    # Set random seed
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
    
    # Prepare batch function with device
    prepare_batch = partial(prepare_batch, device=device)
    
    # Model setup - ALIGNN with 4 outputs for classification
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
    }
    
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    
    # Check if we need to load existing model
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading model from: {best_model_path}")
        try:
            net.load_state_dict(torch.load(best_model_path, map_location=device))
        except RuntimeError as e:
            print(f"Warning: Could not load model (architecture mismatch?): {e}")
            print("Starting with fresh model")
    
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
    net.to(device)
    
    # Compute class weights from training data
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"Using provided class weights: {weights.cpu().numpy()}")
    else:
        print("\nComputing class weights from training data...")
        weights = compute_class_weights(train_loader, num_classes, device)
    
    # Loss function
    if use_focal_loss:
        print(f"Using Focal Loss (gamma={focal_gamma})")
        criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
    else:
        print("Using CrossEntropyLoss with class weights")
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    
    # Scheduler
    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training history
    history_train = []
    history_val = []
    best_val_loss = float('inf')
    best_val_f1 = 0.0
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
            
            batch = prepare_batch(dats)
            
            if line_graph:
                g, lg = batch[0]
                result = net([g, lg])
                targets = batch[1].long()  # Long for CrossEntropy
            else:
                g = batch[0]
                result = net(g)
                targets = batch[1].long()
            
            # Get model outputs - should be [batch_size, num_classes]
            outputs = result["out"]
            
            # Ensure proper shape
            if outputs.dim() == 1:
                # Single output per sample - need 4 outputs for classification
                # If model has 1 output, we need to modify approach
                # For now, assume we configured model with output_features=4
                outputs = outputs.unsqueeze(-1)
            
            # Debug on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"  DEBUG: outputs shape = {outputs.shape}, targets shape = {targets.shape}")
                print(f"  DEBUG: targets unique = {torch.unique(targets)}")
            
            # CrossEntropyLoss expects [batch, num_classes] and [batch] targets
            if outputs.shape[-1] != num_classes:
                # Model doesn't have enough outputs - need to use the single output differently
                # Treat single output as regression and use different approach
                raise ValueError(
                    f"Model has {outputs.shape[-1]} outputs but need {num_classes}. "
                    f"Set model.output_features={num_classes} in config."
                )
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Get predictions from logits
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_targets.extend(targets.cpu().numpy())
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
            'balanced_accuracy': train_metrics['balanced_accuracy'],
            'f1_macro': train_metrics['f1_macro'],
            'f1_per_class': train_metrics['f1_per_class'],
        })
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        val_results = []
        
        with torch.no_grad():
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                batch = prepare_batch(dats)
                
                if line_graph:
                    g, lg = batch[0]
                    result = net([g, lg])
                    targets = batch[1].long()
                else:
                    g = batch[0]
                    result = net(g)
                    targets = batch[1].long()
                
                outputs = result["out"]
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(preds)
                
                for t, p, prob in zip(targets.cpu().numpy(), preds, probs):
                    val_results.append({
                        'id': jid,
                        'target': int(t),
                        'prediction': int(p),
                        'probabilities': prob.tolist(),
                    })
        
        val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(val_targets, val_preds, num_classes)
        
        history_val.append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_metrics['accuracy'],
            'balanced_accuracy': val_metrics['balanced_accuracy'],
            'f1_macro': val_metrics['f1_macro'],
            'f1_per_class': val_metrics['f1_per_class'],
        })
        
        # Save current model
        torch.save(net.state_dict(), os.path.join(config.output_dir, "current_model.pt"))
        
        # Save best model (by f1_macro to prioritize minority classes)
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_loss
            torch.save(net.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            dumpjson(data=val_results, filename=os.path.join(config.output_dir, "Val_results.json"))
            print(f"  ** New best model (F1 Macro: {val_metrics['f1_macro']:.4f}, "
                  f"AFM F1: {val_metrics['f1_per_class'][2]:.4f}) **")
        
        # Save training history
        dumpjson(data=history_train, filename=os.path.join(config.output_dir, "history_train.json"))
        dumpjson(data=history_val, filename=os.path.join(config.output_dir, "history_val.json"))
        
        epoch_time = time.time() - epoch_start
        
        # Print per-class F1 scores
        f1_str = " | ".join([f"{CLASS_NAMES[i]}: {val_metrics['f1_per_class'][i]:.3f}" 
                             for i in range(num_classes)])
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {f1_str} | Time: {epoch_time:.1f}s")
    
    # Load best model for testing
    print("\n" + "=" * 60)
    print("Evaluating on test set with best model")
    print("=" * 60)
    
    net.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    net.eval()
    
    test_targets = []
    test_preds = []
    test_results = []
    
    with torch.no_grad():
        for dats, jid in zip(test_loader, test_loader.dataset.ids):
            batch = prepare_batch(dats)
            
            if line_graph:
                g, lg = batch[0]
                result = net([g, lg])
                targets = batch[1].long()
            else:
                g = batch[0]
                result = net(g)
                targets = batch[1].long()
            
            outputs = result["out"]
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0).unsqueeze(0)
            elif outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            test_targets.extend(targets.cpu().numpy())
            test_preds.extend(preds)
            
            for t, p, prob in zip(targets.cpu().numpy(), preds, probs):
                test_results.append({
                    'id': jid,
                    'target': int(t),
                    'prediction': int(p),
                    'probabilities': prob.tolist(),
                })
    
    # Compute test metrics
    test_metrics = compute_metrics(test_targets, test_preds, num_classes)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    f1_per_class = test_metrics['f1_per_class']
    f1_str = ' | '.join([f'{CLASS_NAMES[i]}: {f1_per_class[i]:.4f}' for i in range(num_classes)])
    print(f"  F1 Per Class: {f1_str}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES, zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    # Save test results
    dumpjson(data=test_results, filename=os.path.join(config.output_dir, "Test_results.json"))
    
    # Save test metrics summary
    test_summary = {
        'accuracy': test_metrics['accuracy'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'f1_macro': test_metrics['f1_macro'],
        'f1_weighted': test_metrics['f1_weighted'],
        'f1_per_class': test_metrics['f1_per_class'],
        'confusion_matrix': cm.tolist(),
        'class_weights_used': weights.cpu().numpy().tolist(),
        'classification_report': classification_report(
            test_targets, test_preds, 
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        ),
    }
    dumpjson(data=test_summary, filename=os.path.join(config.output_dir, "test_metrics.json"))
    
    # Save last model
    torch.save(net.state_dict(), os.path.join(config.output_dir, "last_model.pt"))
    
    # Write CSV prediction results
    with open(os.path.join(config.output_dir, "prediction_results_test_set.csv"), "w") as f:
        f.write("id,target,target_name,prediction,prediction_name,correct,prob_NM,prob_FM,prob_AFM,prob_FiM\n")
        for r in test_results:
            correct = 1 if r['target'] == r['prediction'] else 0
            probs = r['probabilities']
            f.write(f"{r['id']},{r['target']},{CLASS_NAMES[r['target']]},"
                    f"{r['prediction']},{CLASS_NAMES[r['prediction']]},"
                    f"{correct},{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f},{probs[3]:.4f}\n")
    
    print(f"\nResults saved to {config.output_dir}")
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
    }


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train ALIGNN classification model for mp_ordering prediction"
    )
    parser.add_argument("--config", type=str, default="config_classification.json",
                        help="Path to config JSON file")
    parser.add_argument("--train_file", type=str, default="prepared_data_merged/train_data.json",
                        help="Path to training data JSON")
    parser.add_argument("--val_file", type=str, default="prepared_data_merged/val_data.json",
                        help="Path to validation data JSON")
    parser.add_argument("--test_file", type=str, default="prepared_data_merged/test_data.json",
                        help="Path to test data JSON")
    parser.add_argument("--output_dir", type=str, default="output_classification",
                        help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classification classes")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use Focal Loss instead of CrossEntropyLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal Loss")
    parser.add_argument("--class_weights", type=str, default=None,
                        help="Comma-separated class weights (e.g., '1.0,1.5,10.0,3.0')")
    
    args = parser.parse_args()
    
    # Parse class weights if provided
    class_weights = None
    if args.class_weights:
        class_weights = [float(w) for w in args.class_weights.split(',')]
        print(f"Using custom class weights: {class_weights}")
    
    # Configuration for classification
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
        "criterion": "cross_entropy",
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
            "output_features": args.num_classes,  # 4 outputs for 4 classes!
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "atomwise_output_features": 0,
            "calculate_gradient": False,
            "classification": True,  # Enable classification mode
        }
    }
    
    # Load config from file if provided
    if os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            # Deep merge for model config
            if 'model' in file_config:
                config['model'].update(file_config['model'])
                del file_config['model']
            config.update(file_config)
    
    # Override with command line arguments
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["output_dir"] = args.output_dir
    config["model"]["output_features"] = args.num_classes  # Force 4 outputs
    # NOTE: ALIGNN's classification=True forces 1 output (binary), so False for multi-class
    config["model"]["classification"] = False
    
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
        classification=True,  # Classification mode
        output_dir=config["output_dir"],
        workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    
    # Train model
    result = train_ordering_classification(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        prepare_batch=prepare_batch,
        num_classes=args.num_classes,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        class_weights=class_weights,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best validation F1 Macro: {result['best_val_f1']:.4f}")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test balanced accuracy: {result['test_metrics']['balanced_accuracy']:.4f}")
    print(f"Test F1 Macro: {result['test_metrics']['f1_macro']:.4f}")
    f1_per_class = result['test_metrics']['f1_per_class']
    f1_str = ' | '.join([f'{CLASS_NAMES[i]}: {f1_per_class[i]:.4f}' for i in range(args.num_classes)])
    print(f"Test F1 Per Class: {f1_str}")


if __name__ == "__main__":
    main()
