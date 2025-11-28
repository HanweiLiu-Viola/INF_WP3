import logging
from .preprocessor_core import EEGPreprocessor

logger = logging.getLogger(__name__)

def preprocess_eeg_complete(
        raw,
        detect_bad_channels=True, ransac=False,
        interpolate=True, apply_ica=True,
        n_ica_components=None, use_iclabel=True,
        apply_reference=True, ref_channel='REF CZ',
        reference_before_ica=True,
        create_epochs=True, epoch_duration=2.0,
        epoch_overlap=0.0, apply_autoreject=True,
        autoreject_n_jobs=1, autoreject_reject_mode='drop',
        **iclabel_kwargs):

    pre = EEGPreprocessor()
    ica = None
    ar = None
    reject_log = None
    bad_epochs_idx = None

    if detect_bad_channels:
        raw, bad_channels = pre.mark_bad_channels(raw, method='pyprep', ransac=ransac, copy=False)
    else:
        bad_channels = []

    if interpolate and bad_channels:
        raw = pre.interpolate_bad_channels(raw, copy=False)

    if apply_reference and reference_before_ica:
        raw = pre.apply_average_reference(raw, ref_channel=ref_channel, copy=False)

    if apply_ica:
        raw, ica = pre.apply_ica(raw, n_components=n_ica_components, copy=False)
        raw = pre.apply_ica_cleaning(raw, ica, auto_detect=True,
                                     use_iclabel=use_iclabel,
                                     copy=False, **iclabel_kwargs)

    if apply_reference and not reference_before_ica:
        raw = pre.apply_average_reference(raw, ref_channel=ref_channel, copy=False)

    epochs = None
    epochs_clean = None

    if create_epochs:
        epochs = pre.create_fixed_length_epochs(raw, duration=epoch_duration,
                                                overlap=epoch_overlap, copy=False)

        if apply_autoreject:
            result = pre.apply_autoreject(
                epochs, n_jobs=autoreject_n_jobs,
                reject_mode=autoreject_reject_mode
            )
            if autoreject_reject_mode in ['mark','return_all']:
                epochs_clean, ar, reject_log, bad_epochs_idx = result
            else:
                epochs_clean, ar, reject_log = result
        else:
            epochs_clean = epochs

    return {
        'raw': raw,
        'epochs': epochs,
        'epochs_clean': epochs_clean,
        'preprocessor': pre,
        'ica': ica,
        'autoreject': ar,
        'reject_log': reject_log,
        'bad_epochs_idx': bad_epochs_idx
    }
