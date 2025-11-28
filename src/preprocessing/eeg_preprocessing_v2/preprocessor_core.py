import logging

logger = logging.getLogger(__name__)

class EEGPreprocessor:
    """EEG-specific preprocessing operations with advanced artifact removal"""

    def __init__(self):
        self.processing_history = []
        self.bad_channels = []
        self.ica_component_labels = None

    # ========== Import submodules as bound methods ==========

    # Bad channel detection
    from .bad_channels import (
        detect_bad_channels_pyprep,
        mark_bad_channels,
        interpolate_bad_channels,
    )

    # ICA-related processing
    from .ica_processing import (
        apply_ica,
        classify_ica_components_iclabel,
        select_ica_components_by_label,
        detect_ica_artifacts,
        apply_ica_cleaning,
    )

    # Referencing
    from .referencing import apply_average_reference

    # ========== Epoching + Autoreject：使用函数内部延迟导入，避免循环依赖问题 ==========

    def create_fixed_length_epochs(self, *args, **kwargs):
        """Wrapper around epoching.create_fixed_length_epochs(self, ...)"""
        from . import epoching
        return epoching.create_fixed_length_epochs(self, *args, **kwargs)

    def apply_autoreject(self, *args, **kwargs):
        """Wrapper around epoching.apply_autoreject(self, ...)"""
        from . import epoching
        return epoching.apply_autoreject(self, *args, **kwargs)


    # Summary
    def get_processing_summary(self):
        summary = "\nProcessing History:\n"
        summary += "="*60 + "\n"
        for i, step in enumerate(self.processing_history, 1):
            summary += f"  {i}. {step}\n"
        summary += "="*60
        return summary

