import logging
import mne

logger = logging.getLogger(__name__)


def detect_bad_channels_pyprep(self, raw, ransac=False):
    """Detect bad channels on EEG using PyPREP NoisyChannels."""
    try:
        from pyprep.find_noisy_channels import NoisyChannels
    except ImportError:
        raise ImportError("pyprep is required")

    logger.info("=" * 60)
    logger.info("DETECTING BAD CHANNELS WITH PYPREP")
    logger.info("=" * 60)

    eeg_picks = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        ecg=False,
        emg=False,
        misc=False,
        ref_meg=False,
        exclude=[],
    )

    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found")
        return []

    eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]

    ref_patterns = ["REF", "Ref", "ref", "VREF", "VRef"]
    eeg_channel_names_filtered = [
        ch for ch in eeg_channel_names
        if not any(pattern in ch for pattern in ref_patterns)
    ]

    logger.info(f"Total channels: {len(raw.ch_names)}")
    logger.info(f"EEG channels (by type): {len(eeg_channel_names)}")
    logger.info(f"EEG channels (filtered): {len(eeg_channel_names_filtered)}")

    if len(eeg_channel_names_filtered) == 0:
        logger.warning("No valid EEG channels after filtering")
        return []

    raw_eeg = raw.copy().pick_channels(eeg_channel_names_filtered)

    logger.info(f"Channels for pyprep: {len(raw_eeg.ch_names)}")

    if ransac:
        logger.warning("RANSAC disabled due to channel position issues")
        logger.warning("Using correlation + deviation + HF noise instead")
        ransac = False

    logger.info("Detection methods: correlation, deviation, HF noise")

    nc = NoisyChannels(raw_eeg, do_detrend=False)

    logger.info("[1] Correlation detection.")
    nc.find_bad_by_correlation()

    logger.info("[2] Deviation detection.")
    nc.find_bad_by_deviation()

    logger.info("[3] HF noise detection.")
    nc.find_bad_by_hfnoise()

    logger.info("[4] RANSAC: skipped")

    bad_channels = nc.get_bads()

    logger.info("\n" + "=" * 60)
    logger.info("DETECTION RESULTS")
    logger.info("=" * 60)

    if bad_channels:
        logger.info(f"Found {len(bad_channels)} bad channels:")
        logger.info(f"  {bad_channels}")

        if hasattr(nc, "bad_by_correlation") and nc.bad_by_correlation:
            logger.info(f"  Correlation: {nc.bad_by_correlation}")

        if hasattr(nc, "bad_by_deviation") and nc.bad_by_deviation:
            logger.info(f"  Deviation: {nc.bad_by_deviation}")

        if hasattr(nc, "bad_by_hfnoise") and nc.bad_by_hfnoise:
            logger.info(f"  HF noise: {nc.bad_by_hfnoise}")
        elif hasattr(nc, "bad_by_hf_noise") and nc.bad_by_hf_noise:
            logger.info(f"  HF noise: {nc.bad_by_hf_noise}")
    else:
        logger.info("No bad channels detected")

    logger.info("=" * 60)

    return bad_channels


def mark_bad_channels(self, raw, bad_channels=None,
                      method="pyprep", ransac=True, copy=True):
    """Mark bad channels in raw.info['bads']."""
    if copy:
        raw = raw.copy()

    if bad_channels is None:
        if method == "pyprep":
            bad_channels = self.detect_bad_channels_pyprep(raw, ransac=ransac)
        else:
            logger.warning(f"Unknown method '{method}'")
            bad_channels = []
    else:
        logger.info(f"Using manual bad channels: {bad_channels}")

    if bad_channels:
        raw.info["bads"] = bad_channels
        self.bad_channels = bad_channels
        if hasattr(self, "processing_history"):
            self.processing_history.append(f"marked_{len(bad_channels)}_bad")
        logger.info(f"\nMarked {len(bad_channels)} bad channels")
    else:
        raw.info["bads"] = []
        self.bad_channels = []
        logger.info("\nNo bad channels to mark")

    return raw, bad_channels


def interpolate_bad_channels(self, raw, reset_bads=True, copy=True):
    """Interpolate channels currently in raw.info['bads']."""
    if copy:
        raw = raw.copy()

    if not raw.info["bads"]:
        logger.info("No bad channels to interpolate")
        return raw

    logger.info("=" * 60)
    logger.info(f"Interpolating: {raw.info['bads']}")

    raw.interpolate_bads(reset_bads=reset_bads)

    n = len(self.bad_channels)
    if hasattr(self, "processing_history"):
        self.processing_history.append(f"interpolated_{n}")
    logger.info(f"Interpolated {n} channels")
    logger.info("=" * 60)

    return raw
