# 5 classes, 227 videos (181 train, 46 val)
# 경로 설정..
from config import DATASET_PATH, PRETRAINED, WORK_DIR

modality = 'j'
graph = 'coco_new'
work_dir = WORK_DIR + '/exercise/j_phase2_2'

model = dict(
    type = 'RecognizerGCN',
    backbone = dict(
        type = 'ProtoGCN',
        num_prototype = 50,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg = dict(layout='coco_new', mode='random', num_filter=8,
                         init_off=0.04, init_std=0.02)
    ),
    cls_head = dict(
        type='SimpleHead',
        joint_cfg='coco_new',
        num_classes=5,
        in_channels=384,
        weight=0.2,
        dropout=0.5
    )
)

dataset_type = 'PoseDataset'
ann_file = DATASET_PATH

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val')
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 80

checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, 
    metrics=['top_k_accuracy'], 
    topk=(1,), 
    save_best='top1_acc',
    rule='greater'
)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

load_from = "/home/ahi0611/workspace/track-fit/work_dirs/exercise/j_freeze2/best_top1_acc_epoch_13.pth"

resume_from = None
auto_resume = True

workflow = [('train', 1)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = False
