import numpy as np
import random
import warnings
from collections.abc import Sequence


class Flip:
    """Flip the input images with a probability.
    
    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required key are "img_shape", "modality", "imgs" (optional), "keypoint" 
    (optional), added or modified keys are "imgs", "keypoint", "flip_direction".
    The Flip augmentation should be placed after any cropping / reshaping 
    augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are 
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the fliped image 
            with the specific label. Defalut: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[int]): Indexes of right keypoints, used to flip keypoints.
            Default: None.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs, modality):
        """Flip images. Replace mmcv.imflip_ and mmcv.iminvert."""
        axis = 1 if self.direction == 'horizontal' else 0
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=axis).copy()
        lt = len(imgs)
        if modality == "Flow":
            # The 1st Frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = (255 - imgs[i]).astype(imgs[i].dtype)
        return imgs
    
    def _flip_kps(self, kps, kpscores, img_width=None):
        kps = kps.copy()
        kp_x = kps[..., 0]
        nonzero = kp_x != 0
        if img_width is not None:
            kp_x[nonzero] = img_width - kp_x[nonzero]
        else:
            x_vals = kp_x[nonzero]
            if len(x_vals) > 0:
                kp_x[nonzero] = x_vals.min() + x_vals.max() - x_vals
        kps[..., 0] = kp_x
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores
    
    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.
        
        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_
    
    def __call__(self, results):
        """Performs the Flip augmentation.
        
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'keypoint' in results:
            assert self.direction == 'horizontal', {
                'Only horizontal flips are'
                'supported for human keypoints'
            }

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1] if 'img_shape' in results else None

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'], results['label'])

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        if 'gt_bboxes' in results and flip:
            assert self.direction == 'horizontal'
            width = results['img_shape'][1] if 'img_shape' in results else None
            if width is not None:
                results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
                if 'proposals' in results and results['proposals'] is not None:
                    assert results['proposals'].shape[1] == 4
                    results['proposals'] = self._box_flip(results['proposals'], width)
        
        return results
    

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map})')
        return repr_str


class RandomJointMask:
    """Randomly mask (set to 0) specific joints throughout the entire sequence.

    Args:
        chance (float | list[float]): the probability of setting a keypoint 
                                    to invalid or a list of probabilities for each keypoint.
            If it is a float, the same probability is applied to all joints;
            if it is a list, different probabilities are applied to each joint.
        p (flaot): probability of applying the transform. Defaults to 1.0.
    
    Required keys: 'keypoint', 'keypoint_score'.
    """
    def __init__(self, chance, p=1.0):
        self.chance = chance
        self.p = p
    
    def __call__(self, results):
        if np.random.rand() >= self.p:
            return results
        
        keypoint = results['keypoint'] # (M, T, V, 2)
        keypoint_score = results['keypoint_score'] # (M, T, V)
        num_joints = keypoint.shape[2]

        for i in range(num_joints):
            chance = self.chance if isinstance(self.chance, float) else self.chance[i]
            if np.random.rand() < chance:
                keypoint[:, :, i, :] = 0.0
                keypoint_score[:, :, i] = 0.0
        
        results['keypoint'] = keypoint
        results['keypoint_score'] = keypoint_score
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(chance={self.chance}, p={self.p})'
    

class SpecificJointMask:
    """Set the specified keypoints to invalid.

    Args:
        joints (list[int]): The indeces of joints that should be set to invalid.
        p (float): probability of applying the transform. Defaults to 1.0.
    
    Required keys: 'keypoint', 'keypoint_score'.
    
    """
    def __init__(self, joints, p=1.0):
        self.joints = joints
        self.p = p

    def __call__(self, results):
        if np.random.rand() > self.p:
            return results
        
        results['keypoint'][:, :, self.joints, :] = 0.0
        results['keypoint_score'][:, :, self.joints] = 0.0
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(joints={self.joints}, p={self.p})'
        

class InterpolateOcclusions:
    """Interpolates occluded keypoints with linear interpolation.
    Only interpolates between two valid keypoints (no extrapolating).

    Args:
        p (float): probability of applying the transform. Defaults to 1.0.

    Required keys: 'keypoint', 'keypoint_score'.

    """
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, results):
        from scipy import interpolate as sci_interp

        if np.random.rand() >= self.p:
            return results
        
        keypoint = results['keypoint'] # (M, T, V, 2)
        keypoint_score = results['keypoint_score'] # (M, T, V)
        M, T, V, _ = keypoint.shape
        frame_ids = np.arange(T)

        for m in range(M):
            for k in range(V):
                score = keypoint_score[m, :, k] # (T,)
                valid = score > 0
                valid_ids = frame_ids[valid]

                if valid_ids.size < 2:
                    continue

                kp = keypoint[m, :, k, :] # (T, 2)
                f = sci_interp.interp1d(valid_ids, kp[valid], axis=0)

                inter_mask = ~valid
                inter_mask[:valid_ids[0]] = False
                inter_mask[valid_ids[-1]:] = False
                interp_ids = frame_ids[inter_mask]

                if interp_ids.size == 0:
                    continue

                keypoint[m, interp_ids, k, :] = f(interp_ids)

        results['keypoint'] = keypoint
        results['keypoint_score'] = keypoint_score
        return results
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class TemporalOcclusion:
    """Randomly mask consecutive frames within a 100-frame window.
    
    Args:
        min_frames (int): Mininum number of consecutive frames to mask.
        max_frames (int): Maximum number of consecutive frames to mask. 
        num_segments (int): Number of occlusion segments to apply.
        p (float): Probability of applying the transform.
    
    Required keys: 'keypoint', 'keypoint_score'.
    """
    def __init__(self, min_frames=25, max_frames=100, num_segments=1, p=0.5):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.num_segments = num_segments
        self.p = p
    
    def __call__(self, results):
        if np.random.rand() >= self.p:
            return results
        
        keypoint = results['keypoint'] # (M, T, V, 2)
        keypoint_score = results['keypoint_score'] # (M, T, V)
        T = keypoint.shape[1]

        for _ in range(self.num_segments):
            length = np.random.randint(self.min_frames, min(T, self.max_frames) + 1)
            start = np.random.randint(0, T - length + 1)
            end = start + length

            keypoint[:, start:end, :, :] = 0.0
            keypoint_score[:, start:end, :] = 0.0
        
        results['keypoint'] = keypoint
        results['keypoint_score'] = keypoint_score
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'min_frames={self.min_frames}, '
                f'max_frames={self.max_frames}, '
                f'num_segments={self.num_segments}, '
                f'p={self.p})')
