import logging
import numpy as np
import mne
logger = logging.getLogger(__name__)

def create_fixed_length_epochs(self, raw, duration=2.0, overlap=0.0,
                               reject=None, flat=None, copy=True):
    if copy: raw = raw.copy()
    epochs = mne.make_fixed_length_epochs(
        raw, duration=duration, overlap=overlap, preload=True
    )

    if reject or flat:
        epochs.drop_bad(reject=reject, flat=flat)

    return epochs


def apply_autoreject(self, epochs, n_interpolate=None, consensus=None,
                     n_jobs=1, random_state=42, verbose=False,
                     reject_mode='drop'):
    from autoreject import AutoReject

    if n_interpolate is None:
        n_channels = len(epochs.ch_names)
        n_interpolate = np.array([1, 4, 8, min(16, n_channels//2)])

    if consensus is None:
        consensus = np.linspace(0.1, 1.0, 5)

    ar = AutoReject(
        n_interpolate=n_interpolate,
        consensus=consensus,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )

    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    if reject_mode == 'drop':
        return epochs_clean, ar, reject_log
    elif reject_mode in ['mark', 'return_all']:
        bad_idx = np.where(reject_log.bad_epochs)[0].tolist()
        return epochs_clean, ar, reject_log, bad_idx
