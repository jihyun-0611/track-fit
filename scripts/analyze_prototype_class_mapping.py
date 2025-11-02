#!/usr/bin/env python
"""
Analyze prototype-class mapping from trained model.

This script analyzes which prototypes are most associated with each exercise class
by running the entire dataset through the model and collecting Response Signals.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pickle
from dotenv import load_dotenv
import os
from tqdm import tqdm
from collections import defaultdict

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quality_assessment import ResponseSignalExtractor
from protogcn.apis import init_recognizer


def analyze_prototype_class_mapping(
    model,
    dataset_path,
    device='cuda:0',
    num_joints=20
):
    """
    Analyze which prototypes are associated with each class.

    Args:
        model: ProtoGCN model
        dataset_path: Path to dataset pickle file
        device: Device to run on
        num_joints: Number of joints

    Returns:
        Dictionary with prototype-class mapping
    """
    # Load dataset
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    annotations = data['annotations']

    # Create extractor
    extractor = ResponseSignalExtractor(model)

    # Collect response signals by class
    class_responses = defaultdict(list)

    print(f"Analyzing {len(annotations)} samples...")

    for ann in tqdm(annotations):
        keypoint = ann['keypoint']  # (M, T, V, C)
        keypoint_score = ann['keypoint_score']  # (M, T, V)
        label = ann['label']

        # Add confidence dimension
        keypoint_score_expanded = keypoint_score[..., np.newaxis]
        keypoint_with_conf = np.concatenate([keypoint, keypoint_score_expanded], axis=-1)

        # Convert to model input format: (N, num_clips, M, T, V, C)
        keypoint_with_conf = keypoint_with_conf[np.newaxis, :, :, :, :]  # (1, M, T, V, C)
        keypoint_with_conf = keypoint_with_conf[np.newaxis, :, :, :, :, :]  # (1, 1, M, T, V, C)

        # Extract response signal
        keypoint_tensor = torch.FloatTensor(keypoint_with_conf).to(device)
        response_signal = extractor.extract(keypoint_tensor)  # (V*V, n_proto)

        # Average over all joint pairs to get prototype activations
        # Shape: (n_proto,)
        avg_response = response_signal.mean(dim=0).cpu().numpy()

        class_responses[label].append(avg_response)

    extractor.remove_hook()

    # Compute statistics per class
    print("\nComputing prototype-class statistics...")

    class_proto_stats = {}
    for class_idx, responses in class_responses.items():
        responses_array = np.array(responses)  # (num_samples, n_proto)

        # Mean response per prototype for this class
        mean_response = responses_array.mean(axis=0)  # (n_proto,)
        std_response = responses_array.std(axis=0)

        class_proto_stats[class_idx] = {
            'mean': mean_response,
            'std': std_response,
            'num_samples': len(responses)
        }

    # Assign each prototype to the class where it has highest mean response
    n_proto = len(class_proto_stats[0]['mean'])
    prototype_to_class = {}

    for proto_idx in range(n_proto):
        max_response = -1
        best_class = None

        for class_idx, stats in class_proto_stats.items():
            if stats['mean'][proto_idx] > max_response:
                max_response = stats['mean'][proto_idx]
                best_class = class_idx

        prototype_to_class[proto_idx] = best_class

    # Group prototypes by class
    class_to_prototypes = defaultdict(list)
    for proto_idx, class_idx in prototype_to_class.items():
        class_to_prototypes[class_idx].append(proto_idx)

    return {
        'prototype_to_class': prototype_to_class,
        'class_to_prototypes': dict(class_to_prototypes),
        'class_proto_stats': class_proto_stats,
        'num_prototypes': n_proto
    }


def main():
    print("=" * 70)
    print("Prototype-Class Mapping Analysis")
    print("=" * 70)

    # Configuration
    CONFIG_PATH = str(PROJECT_ROOT / 'configs' / 'exercise' / 'j.py')
    CHECKPOINT_PATH = os.getenv('PRETRAINED')
    DATASET_PATH = os.getenv('DATASET_PATH')

    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return

    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return

    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print()

    # Initialize model
    print("Loading model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    print(f"✓ Model loaded on {device}")
    print()

    # Analyze
    mapping = analyze_prototype_class_mapping(model, DATASET_PATH, device=device)

    # Print results
    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)

    label_map = {
        0: "barbell biceps curl",
        1: "bench press",
        2: "lat pulldown",
        3: "push-up",
        4: "tricep Pushdown"
    }

    print(f"\nTotal prototypes: {mapping['num_prototypes']}")
    print("\nPrototypes per class:")
    for class_idx in sorted(mapping['class_to_prototypes'].keys()):
        protos = mapping['class_to_prototypes'][class_idx]
        print(f"  {label_map[class_idx]}: {len(protos)} prototypes")
        print(f"    Indices: {protos[:10]}{'...' if len(protos) > 10 else ''}")

    # Save mapping
    output_path = PROJECT_ROOT / 'prototype_class_mapping.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"\n✓ Mapping saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
