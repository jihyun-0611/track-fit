import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_BASE_DIR = Path(os.getenv('BASE_DIR', Path(__file__).resolve().parent))

_DATA_DIR = Path(os.getenv('DATA_DIR', _BASE_DIR / 'data'))
_CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', _BASE_DIR / 'checkpoints'))
_WORK_DIR = Path(os.getenv('WORK_DIR', _BASE_DIR / 'work_dirs'))

BASE_DIR = str(_BASE_DIR)
DATA_DIR = str(_DATA_DIR)
CHECKPOINT_DIR = str(_CHECKPOINT_DIR)
WORK_DIR = str(_WORK_DIR)

PROTOGCN_DIR = str(_BASE_DIR / 'external' / 'ProtoGCN')
DATASET_PATH = str(_DATA_DIR / 'exercise_dataset_new.pkl')
LABEL_MAPPING_PATH = str(_DATA_DIR / 'exercise_dataset_label_mapping.json')
PRETRAINED = str(_CHECKPOINT_DIR / 'finegym_j' / 'best_top1_acc_epoch_141.pth')
