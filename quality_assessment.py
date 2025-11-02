"""
Quality Assessment Module for Track-Fit

This module provides functionality to evaluate exercise motion quality
using ProtoGCN's Response Signal (prototype similarity).

Key Features:
- Response Signal extraction from ProtoGCN model
- Global quality score based on Top-K prototype concentration
- Joint-wise quality score based on maximum response
- Class-specific quality assessment
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import pickle
from pathlib import Path


class ResponseSignalExtractor:
    """
    Extracts Response Signal from ProtoGCN's Prototype Reconstruction Network.

    The Response Signal R represents how much the input motion aligns with
    each learned prototype:
        R = softmax(X @ W_query^T) ∈ R^(V^2 × n_proto)

    where V is the number of joints and n_proto is the number of prototypes.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: ProtoGCN model (RecognizerGCN instance)
        """
        self.model = model
        self.response_signal = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture Response Signal from PRN."""
        def hook_fn(module, input, output):
            """
            Hook function to capture query (Response Signal) before memory lookup.

            In PRN forward:
                query = softmax(query_matrix(x), dim=-1)  <- This is R
                z = memory_matrix(query)

            We capture 'query' which is the Response Signal R.
            """
            # Input[0] is x (the graph features)
            x = input[0]
            # Apply query_matrix and softmax to get Response Signal
            query = torch.softmax(module.query_matrix(x), dim=-1)
            self.response_signal = query

        # Register hook on PRN's forward method
        try:
            prn_module = self.model.backbone.prn
            self.hook_handle = prn_module.register_forward_hook(hook_fn)
        except AttributeError as e:
            raise AttributeError(
                f"Failed to find PRN module in model. "
                f"Expected model.backbone.prn but got: {e}"
            )

    def extract(self, keypoint_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract Response Signal by running forward pass.

        Args:
            keypoint_tensor: Input keypoints tensor
                Shape: (N, M, T, V, C) for GCN models
                    N: batch size
                    M: number of persons
                    T: number of frames
                    V: number of joints
                    C: number of channels (x, y, confidence)

        Returns:
            Response Signal R with shape (N, V*V, n_proto)
        """
        self.response_signal = None

        with torch.no_grad():
            # Run forward pass (this triggers the hook)
            _ = self.model(keypoint_tensor, return_loss=False)

        if self.response_signal is None:
            raise RuntimeError("Failed to capture Response Signal. Hook may not have been triggered.")

        return self.response_signal

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __del__(self):
        """Cleanup hook on deletion."""
        self.remove_hook()


class QualityAssessment:
    """
    Evaluates exercise motion quality using prototype-based similarity.

    Metrics:
    1. Global Quality Score: Top-K prototype concentration
    2. Joint-wise Quality Score: Maximum response per joint (to be implemented)
    """

    def __init__(
        self,
        model: nn.Module,
        top_k: int = 5,
        num_joints: int = 20,
        mapping_path: Optional[str] = None
    ):
        """
        Args:
            model: ProtoGCN model instance
            top_k: Number of top prototypes to consider for global quality score
            num_joints: Number of joints in the skeleton (default: 20 for COCO)
            mapping_path: Path to prototype-class mapping file (optional)
        """
        self.extractor = ResponseSignalExtractor(model)
        self.top_k = top_k
        self.num_joints = num_joints

        # Load prototype-class mapping if available
        self.class_to_prototypes = None
        if mapping_path is None:
            # Try default location
            default_path = Path(__file__).parent / 'prototype_class_mapping.pkl'
            if default_path.exists():
                mapping_path = str(default_path)

        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
                self.class_to_prototypes = mapping['class_to_prototypes']

    def compute_global_quality(
        self,
        response_signal: torch.Tensor,
        predicted_class: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute global quality score based on Top-K prototype concentration.

        Formula:
            Q_global = (1 / V^2) * Σ_{i=1}^{V^2} Σ_{j=1}^{K} R_i^{top-K}_j

        High scores indicate the motion strongly matches a few key prototypes,
        suggesting good form. Low scores indicate scattered responses across
        many prototypes, suggesting poor or unusual form.

        Args:
            response_signal: Response Signal tensor
                Shape: (N, V*V, n_proto) or (V*V, n_proto)
            predicted_class: If provided, only use prototypes for this class (0-4)

        Returns:
            Dictionary containing:
                - 'global_quality': Overall quality score (0.0 ~ 1.0)
                - 'mean_top_k_response': Average of top-K responses per joint pair
                - 'std_top_k_response': Standard deviation of top-K responses
                - 'used_prototypes': List of prototype indices used (if class filtering applied)
        """
        # Handle batch dimension
        if response_signal.dim() == 3:
            # Shape: (N, V*V, n_proto)
            # Process first sample in batch
            R = response_signal[0]
        else:
            # Shape: (V*V, n_proto)
            R = response_signal

        # R shape: (V*V, n_proto)

        # Filter by class if specified
        used_prototypes = None
        if predicted_class is not None and self.class_to_prototypes is not None:
            # Get prototypes for this class
            class_prototypes = self.class_to_prototypes.get(predicted_class, None)
            if class_prototypes:
                # Filter response signal to only include class prototypes
                proto_indices = torch.tensor(class_prototypes, device=R.device)
                R = R[:, proto_indices]  # (V*V, n_class_proto)

                # NOTE: Do NOT re-normalize to preserve original differences
                # When model outputs are already very uniform (std ~0.0002),
                # re-normalization makes them even more uniform

                used_prototypes = class_prototypes

        # Get top-K values for each joint pair (row)
        # top_k_values shape: (V*V, K)
        actual_k = min(self.top_k, R.shape[1])  # Handle case where n_class_proto < top_k
        top_k_values, top_k_indices = torch.topk(R, k=actual_k, dim=1)

        # Compute global quality score
        # Q_global = mean of all top-K responses
        global_quality = top_k_values.mean().item()

        # Additional statistics
        mean_top_k = top_k_values.mean(dim=1)  # Mean top-K per joint pair
        std_top_k = mean_top_k.std().item()

        result = {
            'global_quality': global_quality,
            'mean_top_k_response': mean_top_k.mean().item(),
            'std_top_k_response': std_top_k,
            'top_k_values': top_k_values.cpu().numpy(),
            'top_k_indices': top_k_indices.cpu().numpy()
        }

        if used_prototypes is not None:
            result['used_prototypes'] = used_prototypes
            result['num_prototypes_used'] = len(used_prototypes)

        return result

    def compute_joint_wise_quality(
        self,
        response_signal: torch.Tensor,
        predicted_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute joint-wise quality score based on maximum response per joint.

        Algorithm:
        1. Reshape Response Signal: R ∈ R^(V²×n_proto) → R_mat ∈ R^(V×V×n_proto)
           where R_mat[i,j,k] is the k-th prototype response between joint i and j

        2. Average responses for each joint across all its connections:
           r̄_i = (1/V) * Σ_{j=1}^V R_mat[i,j,:] ∈ R^(n_proto)

        3. Joint quality score:
           Q_joint(i) = max_{k=1,...,n_proto} r̄_{i,k}

        High scores indicate that joint's motion strongly aligns with learned prototypes.
        Low scores suggest the joint's movement deviates from expected patterns.

        Args:
            response_signal: Response Signal tensor
                Shape: (N, V*V, n_proto) or (V*V, n_proto)
            predicted_class: If provided, only use prototypes for this class (0-4)

        Returns:
            Dictionary containing:
                - 'joint_scores': Quality score for each joint (V,)
                - 'mean_joint_quality': Average quality across all joints
                - 'std_joint_quality': Standard deviation of joint qualities
                - 'min_joint_quality': Minimum joint quality (weakest joint)
                - 'max_joint_quality': Maximum joint quality (strongest joint)
                - 'weak_joints': Indices of joints with quality < threshold
        """
        # Handle batch dimension
        if response_signal.dim() == 3:
            # Shape: (N, V*V, n_proto)
            # Process first sample in batch
            R = response_signal[0]
        else:
            # Shape: (V*V, n_proto)
            R = response_signal

        # R shape: (V*V, n_proto)
        V = self.num_joints

        # Filter by class if specified
        if predicted_class is not None and self.class_to_prototypes is not None:
            # Get prototypes for this class
            class_prototypes = self.class_to_prototypes.get(predicted_class, None)
            if class_prototypes:
                # Filter response signal to only include class prototypes
                proto_indices = torch.tensor(class_prototypes, device=R.device)
                R = R[:, proto_indices]  # (V*V, n_class_proto)

                # NOTE: Do NOT re-normalize to preserve original differences
                # Re-normalization makes already-similar values even more uniform

        n_proto = R.shape[1]

        # Step 1: Reshape to joint-pair matrix
        # R_mat[i, j, k] = response between joint i and j for prototype k
        R_mat = R.view(V, V, n_proto)  # (V, V, n_proto)

        # Step 2: Average responses for each joint across all connections
        # r̄_i = (1/V) * Σ_j R_mat[i,j,:]
        r_bar = R_mat.mean(dim=1)  # (V, n_proto) - average over j dimension

        # Step 3: Joint quality = maximum prototype response
        # Q_joint(i) = max_k r̄_{i,k}
        joint_scores, best_proto_indices = r_bar.max(dim=1)  # (V,)

        # Convert to numpy for return
        joint_scores_np = joint_scores.cpu().numpy()
        best_proto_indices_np = best_proto_indices.cpu().numpy()

        # Compute statistics
        mean_quality = joint_scores.mean().item()
        std_quality = joint_scores.std().item()
        min_quality = joint_scores.min().item()
        max_quality = joint_scores.max().item()

        # Identify weak joints (quality < 0.3 threshold)
        weak_threshold = 0.3
        weak_joints = (joint_scores < weak_threshold).nonzero(as_tuple=True)[0].cpu().numpy()

        return {
            'joint_scores': joint_scores_np,
            'best_prototype_per_joint': best_proto_indices_np,
            'mean_joint_quality': mean_quality,
            'std_joint_quality': std_quality,
            'min_joint_quality': min_quality,
            'max_joint_quality': max_quality,
            'weak_joints': weak_joints.tolist(),
            'num_weak_joints': len(weak_joints)
        }

    def assess_quality(
        self,
        keypoint_tensor: torch.Tensor,
        return_response_signal: bool = False,
        compute_joint_scores: bool = True,
        predicted_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform complete quality assessment on input motion.

        Args:
            keypoint_tensor: Input keypoints tensor
            return_response_signal: Whether to include raw Response Signal in output
            compute_joint_scores: Whether to compute joint-wise quality scores
            predicted_class: If provided, only use prototypes for this class (0-4)
                           0: barbell biceps curl
                           1: bench press
                           2: lat pulldown
                           3: push-up
                           4: tricep Pushdown

        Returns:
            Dictionary containing quality metrics:
                - Global quality metrics (global_quality, mean_top_k_response, etc.)
                - Joint-wise quality metrics (if compute_joint_scores=True)
                - Response signal (if return_response_signal=True)
                - used_prototypes: List of prototype indices (if class filtering applied)
        """
        # Extract Response Signal
        response_signal = self.extractor.extract(keypoint_tensor)

        # Compute global quality (with class filtering if specified)
        quality_metrics = self.compute_global_quality(response_signal, predicted_class)

        # Compute joint-wise quality (with class filtering if specified)
        if compute_joint_scores:
            joint_metrics = self.compute_joint_wise_quality(response_signal, predicted_class)
            quality_metrics['joint_wise'] = joint_metrics

        if return_response_signal:
            quality_metrics['response_signal'] = response_signal.cpu().numpy()

        return quality_metrics

    def get_quality_interpretation(self, global_quality: float) -> Dict[str, str]:
        """
        Provide human-readable interpretation of global quality score.

        Args:
            global_quality: Global quality score (0.0 ~ 1.0)

        Returns:
            Dictionary with interpretation and recommendations
        """
        if global_quality >= 0.7:
            level = "Excellent"
            color = "green"
            message = "Strong alignment with learned prototypes. Motion form is excellent."
        elif global_quality >= 0.5:
            level = "Good"
            color = "blue"
            message = "Good alignment with prototypes. Motion form is acceptable."
        elif global_quality >= 0.4:
            level = "Fair"
            color = "yellow"
            message = "Moderate alignment. Some aspects of form may need improvement."
        else:
            level = "Poor"
            color = "red"
            message = "Weak alignment with prototypes. Motion form needs significant improvement."

        return {
            'level': level,
            'color': color,
            'message': message,
            'score': global_quality
        }

    def get_joint_interpretation(self, joint_score: float) -> Dict[str, str]:
        """
        Provide human-readable interpretation of joint-wise quality score.

        Args:
            joint_score: Joint quality score (0.0 ~ 1.0)

        Returns:
            Dictionary with interpretation
        """
        if joint_score >= 0.5:
            level = "Good"
            color = "green"
            message = "Joint motion aligns well with learned patterns."
        elif joint_score >= 0.3:
            level = "Fair"
            color = "yellow"
            message = "Joint motion shows moderate alignment."
        else:
            level = "Poor"
            color = "red"
            message = "Joint motion deviates from expected patterns."

        return {
            'level': level,
            'color': color,
            'message': message,
            'score': joint_score
        }

    def cleanup(self):
        """Remove hooks and cleanup resources."""
        self.extractor.remove_hook()


# Utility functions for easy usage
def create_quality_assessor(model: nn.Module, top_k: int = 5) -> QualityAssessment:
    """
    Factory function to create a QualityAssessment instance.

    Args:
        model: ProtoGCN model instance
        top_k: Number of top prototypes for quality calculation

    Returns:
        QualityAssessment instance
    """
    return QualityAssessment(model, top_k=top_k)


def assess_motion_quality(
    model: nn.Module,
    keypoint_tensor: torch.Tensor,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    One-shot function to assess motion quality.

    Args:
        model: ProtoGCN model instance
        keypoint_tensor: Input keypoints tensor
        top_k: Number of top prototypes for quality calculation

    Returns:
        Dictionary containing quality metrics and interpretation
    """
    assessor = QualityAssessment(model, top_k=top_k)
    try:
        metrics = assessor.assess_quality(keypoint_tensor)
        interpretation = assessor.get_quality_interpretation(metrics['global_quality'])
        metrics['interpretation'] = interpretation
        return metrics
    finally:
        assessor.cleanup()
