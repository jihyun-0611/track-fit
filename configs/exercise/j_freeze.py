# phase1: head만 학습 20epoch -> phase2: 전체 fine-tuning 80epoch
from config import WORK_DIR

_base_='./j.py'

custom_imports = dict(
    imports=['freeze_backbone_hook'], 
    allow_failed_imports=False
)

custom_hooks = [
    dict(
        type='FreezeBackboneHook', 
        freeze_epochs=20,
    )
]


total_epochs = 20
work_dir = WORK_DIR + '/exercise/j_freeze'
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001,
    nesterov=True
)