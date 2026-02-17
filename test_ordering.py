"""
Test script for mp_ordering prediction using a trained ALIGNN model.

This script evaluates a trained model on the test set and generates
detailed predictions and metrics.
"""

import os
import sys

# Add alignn package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))

import json
import argparse

import torch
import numpy as np
from sklearn.metrics import (
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


def test_ordering(
    config_path: str,
    model_path: str,
    test_file: str,
    output_dir: str,
    num_classes: int = 4,
):
    """Test function for mp_ordering classification."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Testing mp_ordering classification model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Test file: {test_file}")
    
    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader, prepare_batch = get_prepared_train_val_test_loaders(
        train_file=test_file,  # Use test file for all to avoid loading extra data
        val_file=test_file,
        test_file=test_file,
        target=config["target"],
        atom_features=config["atom_features"],
        neighbor_strategy=config["neighbor_strategy"],
        batch_size=1,
        line_graph=config["model"]["alignn_layers"] > 0,
        id_tag=config["id_tag"],
        use_canonize=config.get("use_canonize", True),
        cutoff=config["cutoff"],
        cutoff_extra=config["cutoff_extra"],
        max_neighbors=config["max_neighbors"],
        classification=True,
        output_dir=output_dir,
        workers=0,
        pin_memory=False,
    )
    
    # Model setup
    line_graph = config["model"]["alignn_layers"] > 0
    config_obj = TrainingConfig(**config)
    net = ALIGNNAtomWise(config_obj.model)
    
    # Load trained model
    print(f"Loading model from {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_targets = []
    test_preds = []
    test_results = []
    
    with torch.no_grad():
        for i, (dats, jid) in enumerate(zip(test_loader, test_loader.dataset.ids)):
            if line_graph:
                result = net([dats[0].to(device), dats[1].to(device)])
            else:
                result = net(dats[0].to(device))
            
            logits = result["out"]
            targets = dats[-1].to(device).long()
            probs = torch.softmax(logits, dim=1)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_targets.extend(targets.cpu().numpy())
            test_preds.extend(preds)
            
            for t, p, prob in zip(targets.cpu().numpy(), preds, probs.cpu().numpy()):
                test_results.append({
                    'id': jid,
                    'target': int(t),
                    'prediction': int(p),
                    'probabilities': prob.tolist(),
                })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_loader.dataset)} samples")
    
    # Compute metrics
    acc = accuracy_score(test_targets, test_preds)
    f1_macro = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(test_targets, test_preds, average='weighted', zero_division=0)
    
    # Class names
    class_names = ['NM', 'FM', 'AFM', 'FiM'][:num_classes]
    
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"{'=' * 60}")
    print(f"  Total samples: {len(test_targets)}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_preds, 
                                target_names=class_names,
                                zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        mask = np.array(test_targets) == i
        if mask.sum() > 0:
            class_acc = (np.array(test_preds)[mask] == i).sum() / mask.sum()
            print(f"  {name}: {class_acc:.4f} ({mask.sum()} samples)")
    
    # Save results
    dumpjson(
        data=test_results,
        filename=os.path.join(output_dir, "test_predictions.json"),
    )
    
    test_summary = {
        'model_path': model_path,
        'test_file': test_file,
        'num_samples': len(test_targets),
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
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
        filename=os.path.join(output_dir, "test_summary.json"),
    )
    
    # Save CSV
    with open(os.path.join(output_dir, "test_predictions.csv"), "w") as f:
        f.write("id,target,target_name,prediction,prediction_name,correct")
        for i in range(num_classes):
            f.write(f",prob_{class_names[i]}")
        f.write("\n")
        
        for r in test_results:
            correct = 1 if r['target'] == r['prediction'] else 0
            target_name = class_names[r['target']]
            pred_name = class_names[r['prediction']]
            f.write(f"{r['id']},{r['target']},{target_name},{r['prediction']},{pred_name},{correct}")
            for prob in r['probabilities']:
                f.write(f",{prob:.6f}")
            f.write("\n")
    
    print(f"\nResults saved to {output_dir}")
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(
        description="Test ALIGNN model for mp_ordering prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_ordering.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output_ordering/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="prepared_data/test_data.json",
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of classification classes"
    )
    
    args = parser.parse_args()
    
    result = test_ordering(
        config_path=args.config,
        model_path=args.model,
        test_file=args.test_file,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
    )
    
    print("\nTesting completed!")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1 Macro: {result['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
