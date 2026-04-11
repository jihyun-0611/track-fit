import copy
import os.path as osp
import warnings
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset

from .pipelines.compose import Compose
from ..utils.utils import get_logger, load_file, dump_file, auto_mix2
from ..utils.evaluation import mean_average_precision, mean_class_accuracy, top_k_accuracy, draw_confusion_matrix
from ..utils.metrics import intra_class_similarity, expected_calibration_error


class PoseDataset(Dataset):
    """
    Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a dict
    containing pose information.

    The ann_file is a pickle file, the json file contains a list of annotations,
    the fields of an annotation include frame_dir(video_id), total_frames, label, kp, kpscore.
    
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[callable]): A sequence of data transforms (instantiated).
        data_prefix (str): Path to a directory where videos are held. Default: ''.
        test_mode (bool): Store True when building test or validation dataset. Default: False.
        split (str | None): The dataset split used. Default: None.
        valid_raio (float | None): The valid_raio for videos in KineticsPose. Default: None.
        box_thr (float): The threshold for human proposals. Default: None.
        class_prob (list | None): The class-specific multiplier for resampling. Default: None.
        multi_class (bool): Whether it's a multi-class dataset. Default: False.
        num_classes (int | None): Number of classes. Default: None.
        start_index (int): Specify a start index for frames. Default: 0.
        modality (str): Modality of data. Default: 'Pose'.
        memcached (bool): Whether keypoint is cached in memcached. Default: False.
        mc_cfg (tuple): The config for memcached client. Default: ('localhost', 22077).
    """
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix='',
                 test_mode=False,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='Pose',
                 memcached=False,
                 mc_cfg=('localhost', 22077)):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.split = split
        self.valid_ratio = valid_ratio
        self.box_thr = box_thr
        self.class_prob = class_prob
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.memcached = memcached
        self.mc_cfg = mc_cfg
        self.cli = None

        # Build pipeline
        if isinstance(pipeline, Compose):
            self.pipeline = pipeline
        else:
            self.pipeline = Compose(pipeline)
        
        # Load annotations
        self.video_infos = self.load_annotations()

        # Apply valid_ratio filtering (for Kinetics)
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        if self.valid_ratio is not None and isinstance(self.valid_ratio, float) and self.valid_ratio > 0: 
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                assert 'box_score' in item, 'if valid_ratio is a positive number, item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        
        # Clean up unnecessary fields
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']
        
        logger = get_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    
    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.pkl'):
            return self.load_pkl_annotations()
        elif self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        else:
            raise ValueError(f'Unsupported annotation file format: {self.ann_file}')
    
    def load_pkl_annotations(self):
        """Load annotations from pickle file"""
        data = load_file(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data
    
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = load_file(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos
    
    
    def parse_by_class(self):
        """Parse video_infos by class."""
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class
    
    @staticmethod
    def label2array(num, label):
        """Convert label to one-hot array."""
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])

        # Handle memcached
        if self.memcached and 'key' in results:
            results = self._load_from_memcached(results)
        
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # Handle multi-class labels
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        
        results['test_mode'] = self.test_mode

        return self.pipeline(results)
    
    def prepare_test_frames(self, idx):
        """Prepare the frame for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])

        # Handle memcached
        if self.memcached and 'key' in results:
            results = self._load_from_memcached(results)
        
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # Handle multi-class labels
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        
        results['test_mode'] = self.test_mode
        results['idx'] = idx

        return self.pipeline(results)
    
    def _load_from_memcached(self, results):
        """Load data from memcached."""
        from pymemcache import serde
        from pymemcache.client.base import Client

        if self.cli is None:
            self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)

        key = results.pop('key')
        try:
            pack = self.cli.get(key)
        except:
            self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            pack = self.cli.get(key)
        
        if not isinstance(pack, dict):
            raw_file = results['raw_file']
            data = load_file(raw_file)
            pack = data[key]
            for k in data:
                try:
                    self.cli.set(k, data[k])
                except:
                    self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                    self.cli.set(k, data[k])
        

        for k in pack:
            results[k] = pack[k]
        
        return results
    

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)
    
    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        return self.prepare_test_frames(idx) if self.test_mode else self.prepare_train_frames(idx)
    
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 features=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.
        
        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options.
            logger: Logger for recording. Defalut: None.

        Returns:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')
        
        # Handle tuple/list results (multi-output)
        if isinstance(results[0], (list, tuple)):
            num_results = len(results[0])
            eval_results = dict()
            for i in range(num_results):
                eval_results_cur = self.evaluate(
                    [x[i] for x in results], metrics, metric_options, logger
                )
                eval_results.update({f'{k}_{i}': v for k, v in eval_results_cur.items()})
            return eval_results
        
        # Handle dict results
        if isinstance(results[0], dict):
            eval_results = dict()
            for key in results[0]:
                results_cur = [x[key] for x in results]
                eval_results_cur = self.evaluate(results_cur, metrics, metric_options, logger, **deprecated_kwargs)
                eval_results.update({f'{key}_{k}': v for k, v in eval_results_cur.items()})
            
            # RGB-Pose
            if len(results[0]) == 2 and 'rgb' in results[0] and 'pose' in results[0]:
                rgb = [x['rgb'] for x in results]
                pose = [x['pose'] for x in results]
                preds = auto_mix2([rgb, pose])
                for k in preds:
                    eval_results_cur = self.evaluate(preds[k], metrics, metric_options, logger, **deprecated_kwargs)
                    eval_results.update({f'RGBPose_{k}_{key}': v for key, v in eval_results_cur.items()})
            return eval_results
        
        metric_options = copy.deepcopy(metric_options)
        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                '`metric_options`, for more details'
            )
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs
            )
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'confusion_matrix', 'ece', 'intra_class_similarity',
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
            
        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            log_msg = f'Evaluating {metric} ...'
            if logger is None:
                print(log_msg)
            else:
                logger.info(log_msg)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
                if isinstance(topk, int):
                    topk = (topk, )
                
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg_parts = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg_parts.append(f'top{k}_acc: {acc:.4f}')
                log_msg = ', '.join(log_msg_parts)
                if logger is None:
                    print(log_msg)
                else:
                    logger.info(log_msg)
            
            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'mean_class_accuracy: {mean_acc:.4f}'
                if logger is None:
                    print(log_msg)
                else:
                    logger.info(log_msg)
            
            if metric == 'mean_average_precision':
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                mAP = mean_average_precision(results, gt_labels_arrays)
                eval_results['mean_average_precision'] = mAP
                log_msg = f'mean_average_precision: {mAP:.4f}'
                if logger is None:
                    print(log_msg)
                else:
                    logger.info(log_msg)

            if metric == 'confusion_matrix':
                opts = metric_options.get('confusion_matrix', {})
                draw_confusion_matrix(
                    results, self,
                    work_dir=opts.get('work_dir', '.'),
                    logger=logger,
                    save_path=opts.get('save_path'),
                    label_map_file=opts.get('label_map_file'),
                )

            if metric == 'ece':
                ece_res = expected_calibration_error(np.array(results), np.array(gt_labels))
                eval_results['ece'] = ece_res['ece']
                eval_results['mce'] = ece_res['mce']
                log_msg = f'ece: {ece_res["ece"]:.4f}, mce: {ece_res["mce"]:.4f}'
                if logger is None:
                    print(log_msg)
                else:
                    logger.info(log_msg)

            if metric == 'intra_class_similarity':
                if features is None:
                    log_msg = 'intra_class_similarity skipped: features not provided'
                    if logger is None:
                        print(log_msg)
                    else:
                        logger.warning(log_msg)
                else:
                    sim_res = intra_class_similarity(features, np.array(gt_labels))
                    eval_results['intra_class_sim_mean'] = sim_res['mean']
                    eval_results['intra_class_sim_std'] = sim_res['std']
                    log_msg = f'intra_class_sim: mean={sim_res["mean"]:.4f}, std={sim_res["std"]:.4f}'
                    if logger is None:
                        print(log_msg)
                    else:
                        logger.info(log_msg)

        return eval_results
    

    def dump_results(self, results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return dump_file(results, out)







        

        

