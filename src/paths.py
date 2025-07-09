import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
