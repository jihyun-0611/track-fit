import copy as cp
import numpy as np
import warnings


class UniformSampleFrames:
    """
    Uniformly sample frames from the video.

    Divide the video into n segments of equal length and randomly sample one
    frame from each segment.
    To make the testing results reproducible, a random seed is set during testing, 
    to make the sampling results deterministic.

    Required keys are "total_frames", "start_index", added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frame of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default : 1
        p_interval (tuple | float): Sampling interval ratio. Default : 1
        seed (int) : The random seed used during test time. Default : 255
    """
    def __init__(self,
                 clip_len,
                 num_clips=1,
                 p_interval=1,
                 seed=255,
                 **deprecated_kwargs):
        
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)
        
        if len(deprecated_kwargs):
            warnings.warn('[UniformSampleFrames] The following args has been deprecated: ')
            for k, v in deprecated_kwargs.items():
                warnings.warn(f'Arg {k}: {v}')
            

    def _get_train_clips(self, num_frames, clip_len):
        """
        Uniformly sample indices for training clips.
        
        Args: 
            num_frames (int): Number of total frames
            clip_len (int): Number of frames to sample

        Returns:
            np.ndrarray : index of sampled frames
        """
        all_offsets = []
        
        for _ in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.uniform(pi[0], pi[1])
            num_frames_used = int(num_frames * ratio)
            start = np.random.randint(num_frames - num_frames_used + 1)
            
            # sampling in uniform 
            if num_frames_used < clip_len:
                start_idx = np.random.randint(0, num_frames_used)
                offsets = np.arange(start_idx, start_idx + clip_len)
            elif clip_len <= num_frames_used < 2 *clip_len:
                idxes = np.random.choice(
                    clip_len+1, 
                    num_frames_used - clip_len, 
                    replace=False
                )
                offsets = np.zeros(clip_len + 1, dtype=np.int64)
                offsets[idxes] = 1
                offsets = np.cumsum(offsets)
                offsets = np.arange(clip_len) + offsets[:-1]
            else:
                idxes = np.array(
                    [i*num_frames_used // clip_len for i in range(clip_len + 1)]
                )
                bin_size = np.diff(idxes)
                idxes = idxes[:clip_len]
                offsets = np.random.randint(bin_size)
                offsets = idxes + offsets
            
            offsets = offsets + start
            all_offsets.append(offsets)
        return np.concatenate(all_offsets)

    
    def _get_test_clips(self, num_frames, clip_len):
        """
        Random sampling in each section for test
        
        Args: 
            num_frames (int): Number of total frames
            clip_len (int): Number of frames to sample

        Returns:
            np.ndrarray : index of sampled frames
        """
        np.random.seed(self.seed)

        all_offsets = []
        
        for _ in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.uniform(pi[0], pi[1])
            num_frames_used = int(num_frames * ratio)
            start = np.random.randint(num_frames - num_frames_used + 1)
            
            # sampling in uniform 
            if num_frames_used < clip_len:
                start_idx = np.random.randint(0, num_frames_used)
                offsets = np.arange(start_idx, start_idx + clip_len)
            elif clip_len <= num_frames_used < 2 *clip_len:
                idxes = np.random.choice(
                    clip_len+1, 
                    num_frames_used - clip_len, 
                    replace=False
                )
                offsets = np.zeros(clip_len + 1, dtype=np.int64)
                offsets[idxes] = 1
                offsets = np.cumsum(offsets)
                offsets = np.arange(clip_len) + offsets[:-1]
            else:
                idxes = np.array(
                    [i*num_frames_used // clip_len for i in range(clip_len + 1)]
                )
                bin_size = np.diff(idxes)
                idxes = idxes[:clip_len]
                offsets = np.random.randint(bin_size)
                offsets = idxes + offsets
            
            offsets = offsets + start
            all_offsets.append(offsets)

        return np.concatenate(all_offsets)
    
    def __call__(self, results):
        num_frames = results['total_frames']

        if results.get('test_mode', False):
            idxes = self._get_test_clips(num_frames, self.clip_len)
        else:
            idxes = self._get_train_clips(num_frames, self.clip_len)
        
        idxes = np.mod(idxes, num_frames)
        start_index = results['start_index']
        idxes = idxes + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            people_per_frames = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j,i]) < 1e-5):
                    j -= 1
                people_per_frames[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if people_per_frames[i] != people_per_frames[i-1]:
                    transitional[i] = transitional[i-1] = True
                if people_per_frames[i] != people_per_frames[i+1]:
                    transitional[i] = transitional[i+1] = True
            idxes_int = idxes.astype(np.int64)
            coef = np.array([transitional[i] for i in idxes_int])
            idxes = (coef * idxes_int + (1 - coef) * idxes).astype(np.float32)

        results['frame_inds'] = idxes.astype(np.int64)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'seed={self.seed})')
        return repr_str


class UniformSample(UniformSampleFrames):
    pass


class UniformSampleDecode:
    """Uniformly sample and decode keypoints directly."""
    def __init__(self, clip_len, num_clips=1, p_interval=1, seed=255):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

    def _get_clips(self, full_kp, clip_len):
        """
        Get clips from full keypoints.

        Args:
            full_kp (np.ndarray): Full keypoints with shape (M, T, V, C).
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: Sampled clips concatenated along temporal dimension.
        """

        M, T, V, C = full_kp.shape
        clips = []
        for _ in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.uniform(pi[0], pi[1])
            num_frames_used = int(T * ratio)
            start = np.random.randint(T - num_frames_used + 1)
            
            # sampling in uniform 
            if num_frames_used < clip_len:
                start_idx = np.random.randint(0, num_frames_used)
                offsets = (np.arange(start_idx, start_idx + clip_len)%num_frames_used) + start
            elif clip_len <= num_frames_used < 2 *clip_len:
                idxes = np.random.choice(
                    clip_len+1, 
                    num_frames_used - clip_len, 
                    replace=False
                )
                offsets = np.zeros(clip_len + 1, dtype=np.int64)
                offsets[idxes] = 1
                offsets = np.cumsum(offsets)
                offsets = np.arange(clip_len) + offsets[:-1] + start
            else:
                idxes = np.array(
                    [i*num_frames_used // clip_len for i in range(clip_len + 1)]
                )
                bin_size = np.diff(idxes)
                idxes = idxes[:clip_len]
                offsets = np.random.randint(bin_size)
                offsets = idxes + offsets + start
            
            clip = full_kp[:, offsets].copy()
            clips.append(clip)

        return np.concatenate(clips, 1)


    def _handle_dict(self, results):
        """
        Handle dict input 
        
        Args:
            results (dict): Result dict containing keypoint data.

        Returns:
            dict: Result dict with sampled keypoints.
        """
        assert 'keypoint' in results
        kp = results.pop('keypoint')
        if 'keypoint_score' in results:
            kp_score = results.pop('keypoint_score') 
            kp =  np.concatenate([kp, kp_score[..., None]], axis=-1) 

        kp = kp.astype(np.float32)
        kp = self._get_clips(kp, self.clip_len)

        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        results['keypoint'] = kp
        return results
    
    def _handle_list(self, results):
        """Handle list input.

        Args:
            results (list): List of result dicts.

        Returns:
            dict: Merged result dict with sampled keypoints.
        """
        assert len(results) == self.num_clips
        self.num_clips = 1 
        clips = []
        for res in results:
            assert 'keypoint' in res
            kp = res.pop('keypoint')
            if 'keypoint_score' in res:
                kp_score = res.pop('keypoint_score')
                kp = np.concatenate([kp, kp_score[..., None]], axis=-1)
            kp = kp.astype(np.float32)
            kp = self._get_clips(kp, self.clip_len)
            clips.append(kp) 

        ret = cp.deepcopy(results[0])
        ret['clip_len'] = self.clip_len
        ret['frame_interval'] = None
        ret['num_clips'] = len(results)
        ret['keypoint'] = np.concatenate(clips, 1)
        self.num_clips = len(results)
        return ret
    

    def __call__(self, results):
        test_mode = results.get('test_mode', False) if isinstance(results, dict) else results[0].get('test_mode', False)
        if test_mode is True:
            np.random.seed(self.seed)
        if isinstance(results, list):
            return self._handle_list(results)
        else:
            return self._handle_dict(results)
        
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'p_interval={self.p_interval}, '
                    f'seed={self.seed})')
        return repr_str
    
    