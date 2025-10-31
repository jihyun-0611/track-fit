#!/usr/bin/env python
"""
Test script for quality assessment functionality.

This script tests the quality assessment module with ProtoGCN model.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pickle
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quality_assessment import QualityAssessment
from protogcn.apis import init_recognizer


def load_sample_data(dataset_path: str, num_samples: int = 3):
    """Load sample sequences from dataset."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Get annotations
    annotations = data['annotations']

    # Sample some test sequences
    samples = []
    for i, ann in enumerate(annotations[:num_samples]):
        keypoint = ann['keypoint']  # Shape: (M, T, V, C) = (1, 131, 20, 2)
        keypoint_score = ann['keypoint_score']  # Shape: (M, T, V) = (1, 131, 20)
        label = ann['label']

        # Add confidence dimension to keypoint: (M, T, V, 2) -> (M, T, V, 3)
        keypoint_score_expanded = keypoint_score[..., np.newaxis]  # (M, T, V, 1)
        keypoint_with_conf = np.concatenate([keypoint, keypoint_score_expanded], axis=-1)  # (M, T, V, 3)

        # Convert to model input format: (N, num_clips, M, T, V, C)
        # Dataset returns (num_clips, M, T, V, C), need to add batch dimension
        # keypoint is (M, T, V, C), need to add num_clips and batch dimensions
        keypoint_with_conf = keypoint_with_conf[np.newaxis, :, :, :, :]  # (num_clips, M, T, V, C) = (1, M, T, V, C)
        keypoint_with_conf = keypoint_with_conf[np.newaxis, :, :, :, :, :]  # (N, num_clips, M, T, V, C) = (1, 1, M, T, V, C)

        samples.append({
            'keypoint': keypoint_with_conf,
            'label': label,
            'frame_dir': ann.get('frame_dir', f'sample_{i}')
        })

    return samples


def main():
    print("=" * 70)
    print("Quality Assessment Test")
    print("=" * 70)

    # Configuration
    CONFIG_PATH = str(PROJECT_ROOT / 'configs' / 'exercise' / 'j.py')
    CHECKPOINT_PATH = os.getenv('PRETRAINED')
    DATASET_PATH = os.getenv('DATASET_PATH')

    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please set PRETRAINED in .env file")
        return

    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("Please set DATASET_PATH in .env file")
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

    # Load sample data
    print("Loading sample data...")
    samples = load_sample_data(DATASET_PATH, num_samples=3)
    print(f"✓ Loaded {len(samples)} samples")
    print()

    # Label map
    label_map = {
        0: "barbell biceps curl",
        1: "bench press",
        2: "lat pulldown",
        3: "push-up",
        4: "tricep Pushdown"
    }

    # Test quality assessment
    print("=" * 70)
    print("Running Quality Assessment")
    print("=" * 70)
    print()

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample['frame_dir']}")
        print(f"  Ground Truth: {label_map[sample['label']]}")
        print(f"  Sequence Shape: {sample['keypoint'].shape}")

        keypoint_tensor = torch.FloatTensor(sample['keypoint']).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(keypoint_tensor, return_loss=False)
            scores = torch.softmax(torch.FloatTensor(output[0]), dim=0)
            pred_idx = scores.argmax().item()
            confidence = scores[pred_idx].item()

        print(f"  Prediction: {label_map[pred_idx]} (confidence: {confidence:.4f})")

        # Perform quality assessment (with joint-wise scores and class-specific prototypes)
        assessor_temp = QualityAssessment(model, top_k=5, num_joints=20)
        result = assessor_temp.assess_quality(
            keypoint_tensor,
            compute_joint_scores=True,
            predicted_class=pred_idx  # Use predicted class for class-specific assessment
        )
        interpretation = assessor_temp.get_quality_interpretation(result['global_quality'])

        print(f"  Quality Assessment:")
        print(f"    Global Quality Score: {result['global_quality']:.4f}")
        print(f"    Mean Top-5 Response: {result['mean_top_k_response']:.4f}")
        print(f"    Std Top-5 Response: {result['std_top_k_response']:.4f}")
        print(f"    Level: {interpretation['level']} ({interpretation['color']})")
        print(f"    Message: {interpretation['message']}")

        # Show class-specific prototype info if available
        if 'used_prototypes' in result:
            print(f"    Used Prototypes: {result['num_prototypes_used']} prototypes for class '{label_map[pred_idx]}'")

        # Print joint-wise scores
        if 'joint_wise' in result:
            joint_metrics = result['joint_wise']
            print(f"  Joint-wise Quality:")
            print(f"    Mean Joint Quality: {joint_metrics['mean_joint_quality']:.4f}")
            print(f"    Std Joint Quality: {joint_metrics['std_joint_quality']:.4f}")
            print(f"    Min Joint Quality: {joint_metrics['min_joint_quality']:.4f} (weakest)")
            print(f"    Max Joint Quality: {joint_metrics['max_joint_quality']:.4f} (strongest)")
            print(f"    Weak Joints (< 0.3): {joint_metrics['weak_joints']} ({joint_metrics['num_weak_joints']} joints)")

            # Show top 5 best and worst joints
            joint_scores = joint_metrics['joint_scores']
            sorted_indices = np.argsort(joint_scores)
            print(f"    Top 3 Best Joints: {sorted_indices[-3:][::-1].tolist()} with scores {joint_scores[sorted_indices[-3:][::-1]]}")
            print(f"    Top 3 Worst Joints: {sorted_indices[:3].tolist()} with scores {joint_scores[sorted_indices[:3]]}")

        assessor_temp.cleanup()
        print()

    # Test with persistent QualityAssessment instance
    print("=" * 70)
    print("Testing Persistent Quality Assessor")
    print("=" * 70)
    print()

    assessor = QualityAssessment(model, top_k=5)

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample['frame_dir']}")

        keypoint_tensor = torch.FloatTensor(sample['keypoint']).to(device)
        metrics = assessor.assess_quality(keypoint_tensor, return_response_signal=True)

        # Check Response Signal shape
        R = metrics['response_signal']
        print(f"  Response Signal Shape: {R.shape}")
        print(f"  Expected: (V*V, n_proto) = (400, 100) for COCO 20-joint dataset")
        print(f"  Global Quality: {metrics['global_quality']:.4f}")
        print()

    assessor.cleanup()

    print("=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
