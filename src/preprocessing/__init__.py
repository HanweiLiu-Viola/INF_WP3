"""
Preprocessing Module for DBS-EEG-LFP Data

Supports multiple data formats: BrainVision, FIF, EDF, BDF, SET, CNT
==========================================

Modules:
    - lfp_io: LFP data loading and processing
    - eeg_io: EEG data loading and processing
    - alignment: EEG-LFP synchronization
    - bids_export: BIDS format export
    ......
"""

__version__ = "1.0.0"
__author__ = "Hanwei Liu"

from . import lfp_io
from . import eeg_io
from . import alignment
from . import bids_export
from . import visualization
from .align_headmodel import make_trans_from_coordinates
from .data_io import BIDSDataLoader, BIDSDataSaver
from .data_validation import DataValidator
from .signal_cleaning import SignalCleaner, EEGCleaner, LFPCleaner
from .eeg_preprocessing_v2 import EEGPreprocessor
from .lfp_preprocessing import LFPPreprocessor
from .joint_preprocessing import JointPreprocessor
from .quality_control import QualityControl, BIDSDerivativesSaver
# from .eeg_preprocessing import EEGPreprocessor, preprocess_eeg_complete




__all__ = [
    'lfp_io',
    'eeg_io', 
    'alignment',
    'visualization',
    'make_trans_from_coordinates',
    'bids_export',
    'BIDSDataLoader',
    'BIDSDataSaver',
    'DataValidator',
    'SignalCleaner',
    'EEGCleaner',
    'LFPCleaner',
    'EEGPreprocessor',
    'LFPPreprocessor',
    'JointPreprocessor',
    'QualityControl',
    'BIDSDerivativesSaver'    
]
