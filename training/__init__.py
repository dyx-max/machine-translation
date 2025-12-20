"""
训练模块
"""
from training.trainer import Trainer
from training.loss import LabelSmoothingLoss
from training.scheduler import NoamOpt
from training.validator import run_validation

__all__ = ['Trainer', 'LabelSmoothingLoss', 'NoamOpt', 'run_validation']

