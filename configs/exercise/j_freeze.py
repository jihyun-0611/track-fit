# phase1: head만 학습 20epoch -> phase2: 전체 fine-tuning 80epoch
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_base_='./j.py'

# Load work_dir from .env (no defaults)
_work_dir = Path(os.environ['WORK_DIR'])

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
work_dir = str(_work_dir / 'j_freeze')
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001,
    nesterov=True
)