"""
EEG Data I/O and Processing Module
===================================

Functions for loading and processing EEG data from EGI systems.
"""

import mne
import numpy as np
from pathlib import Path
from scipy import signal

# ============================================================
# EEG Loading Functions
# ============================================================

def load_eeg_data(mff_dir, preload=True, verbose=True):
    """
    Load EEG data from EGI MFF format.
    
    Parameters
    ----------
    mff_dir : str or Path
        Path to .mff directory
    preload : bool, optional
        Load data into memory (default: True)
    verbose : bool, optional
        Print loading information (default: True)
        
    Returns
    -------
    mne.io.Raw
        MNE raw object
    """
    mff_dir = Path(mff_dir)
    
    if not mff_dir.exists():
        raise FileNotFoundError(f"MFF directory not found: {mff_dir}")
    
    raw = mne.io.read_raw_egi(mff_dir, preload=preload, verbose=verbose)
    
    return raw


def setup_reference_channels(raw, ref_channels=None):
    """
    Setup reference channels as 'misc' type.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    ref_channels : list, optional
        List of reference channel names. If None, assumes last channel.
        
    Returns
    -------
    mne.io.Raw
        Modified raw object
    """
    if ref_channels is None:
        # Assume last channel is reference
        ref_channels = [raw.ch_names[-1]]
    
    print(f"Setting reference channels: {ref_channels}")
    
    # Set channel types
    channel_types = {ch: 'misc' for ch in ref_channels}
    raw.set_channel_types(channel_types)
    
    return raw


# ============================================================
# Stimulation Detection in EEG
# ============================================================

def detect_stim_frequency(raw, freq_range=(40, 70), prominence_threshold=0.3):
    """
    Detect DBS stimulation frequency using frequency domain analysis.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    freq_range : tuple
        Frequency range to search (Hz)
    prominence_threshold : float
        Peak prominence threshold
        
    Returns
    -------
    tuple
        (dominant_freq, freqs, psd_mean)
    """
    # Select EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)
    sfreq = raw.info['sfreq']
    
    # Compute power spectrum for all EEG channels
    n_channels = data.shape[0]
    nperseg = min(int(sfreq * 10), data.shape[1])  # 10 seconds or full length
    
    freqs = None
    psd_sum = None
    
    for ch_idx in range(n_channels):
        f, psd = signal.welch(data[ch_idx], fs=sfreq, nperseg=nperseg)
        if freqs is None:
            freqs = f
            psd_sum = psd
        else:
            psd_sum += psd
    
    psd_mean = psd_sum / n_channels
    
    # Find peaks in specified frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    peaks, properties = signal.find_peaks(psd_mean[mask], 
                                         prominence=prominence_threshold,
                                         distance=5)
    
    if len(peaks) > 0:
        # Return most prominent peak
        peak_freqs = freqs[mask][peaks]
        peak_prominences = properties['prominences']
        dominant_idx = np.argmax(peak_prominences)
        dominant_freq = peak_freqs[dominant_idx]
    else:
        # Return frequency with maximum PSD
        idx = np.argmax(psd_mean[mask])
        dominant_freq = freqs[mask][idx]
    
    return dominant_freq, freqs, psd_mean


def extract_stim_power(raw, stim_freq, freq_tolerance=2, method='hilbert'):
    """
    Extract stimulation power envelope from EEG.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    stim_freq : float
        Stimulation frequency (Hz)
    freq_tolerance : float
        Frequency band around stim_freq (Hz)
    method : str
        Method for envelope extraction ('hilbert' or 'rms')
        
    Returns
    -------
    tuple
        (stim_power, times)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)
    sfreq = raw.info['sfreq']
    times = raw.times
    
    # Bandpass filter around stim frequency
    freq_low = max(1, stim_freq - freq_tolerance)
    freq_high = min(sfreq / 2 - 1, stim_freq + freq_tolerance)
    
    filtered = mne.filter.filter_data(data, sfreq=sfreq, 
                                      l_freq=freq_low, h_freq=freq_high,
                                      verbose=False)
    
    if method == 'hilbert':
        # Hilbert envelope
        analytic = signal.hilbert(filtered, axis=1)
        envelope = np.abs(analytic)
        power = np.mean(envelope, axis=0)
        
        # Smooth
        window = int(sfreq * 1.0)
        power = np.convolve(power, np.ones(window)/window, mode='same')
        
    elif method == 'rms':
        # RMS in sliding windows
        window = int(sfreq * 0.5)
        power = np.zeros_like(times)
        
        for i in range(len(times)):
            start = max(0, i - window // 2)
            end = min(len(times), i + window // 2)
            power[i] = np.sqrt(np.mean(filtered[:, start:end] ** 2))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return power, times


def detect_stim_temporal(raw, stim_freq, window_sec=1.0):
    """
    Detect DBS stimulation in time domain.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    stim_freq : float
        Stimulation frequency (Hz)
    window_sec : float
        Window size for energy calculation (seconds)
        
    Returns
    -------
    tuple
        (stim_envelope, times)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)
    sfreq = raw.info['sfreq']
    times = np.arange(data.shape[1]) / sfreq
    
    # Bandpass filter
    freq_low = max(stim_freq - 5, 1)
    freq_high = min(stim_freq + 5, sfreq / 2 - 1)
    
    filtered = mne.filter.filter_data(data, sfreq=sfreq, 
                                      l_freq=freq_low, h_freq=freq_high,
                                      verbose=False)
    
    # Calculate signal envelope (Hilbert transform)
    analytic = signal.hilbert(filtered, axis=1)
    envelope = np.abs(analytic)
    
    # Average across channels
    mean_envelope = np.mean(envelope, axis=0)
    
    # Calculate energy in sliding windows
    window_samples = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_samples
    
    energy = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_samples
        end = (i + 1) * window_samples
        energy[i] = np.sum(mean_envelope[start:end] ** 2)
    
    # Normalize and interpolate back to original time resolution
    energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
    stim_envelope = np.interp(times, 
                              np.linspace(0, times[-1], len(energy_normalized)),
                              energy_normalized)
    
    return stim_envelope, times


def compute_spectrogram(raw, freq_range=(40, 70), time_resolution=1.0, 
                       channel_selection='all'):
    """
    Compute time-frequency spectrogram to identify stable narrowband signals.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    freq_range : tuple
        Frequency range of interest (Hz)
    time_resolution : float
        Time resolution in seconds
    channel_selection : str or list
        'all' or list of channel indices
        
    Returns
    -------
    tuple
        (f, t, Sxx, f_freq, Sxx_freq)
    """
    if channel_selection == 'all':
        eeg_picks = mne.pick_types(raw.info, eeg=True)
    else:
        eeg_picks = channel_selection
    
    data = raw.get_data(picks=eeg_picks)
    sfreq = raw.info['sfreq']
    
    # Average across selected channels
    mean_signal = np.mean(data, axis=0)
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(mean_signal, fs=sfreq, 
                                   nperseg=int(sfreq * time_resolution),
                                   noverlap=int(sfreq * time_resolution * 0.5))
    
    # Limit to frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    Sxx_freq = Sxx[freq_mask]
    f_freq = f[freq_mask]
    
    return f, t, Sxx, f_freq, Sxx_freq


# ============================================================
# EEG Preprocessing
# ============================================================

def filter_eeg(raw, l_freq=None, h_freq=None, notch_freq=None):
    """
    Apply filters to EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    l_freq : float, optional
        High-pass filter cutoff (Hz)
    h_freq : float, optional
        Low-pass filter cutoff (Hz)
    notch_freq : float or list, optional
        Notch filter frequency/frequencies (Hz)
        
    Returns
    -------
    mne.io.Raw
        Filtered raw object
    """
    raw_filtered = raw.copy()
    
    # Bandpass filter
    if l_freq is not None or h_freq is not None:
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        print(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
    
    # Notch filter
    if notch_freq is not None:
        raw_filtered.notch_filter(freqs=notch_freq, verbose=False)
        print(f"Applied notch filter at: {notch_freq} Hz")
    
    return raw_filtered


def rereference_eeg(raw, ref_channels='average'):
    """
    Re-reference EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    ref_channels : str or list
        'average' for average reference, or list of channel names
        
    Returns
    -------
    mne.io.Raw
        Re-referenced raw object
    """
    raw_reref = raw.copy()
    
    if ref_channels == 'average':
        raw_reref.set_eeg_reference('average', projection=True)
        raw_reref.apply_proj()
        print("Applied average reference")
    else:
        raw_reref.set_eeg_reference(ref_channels=ref_channels)
        print(f"Applied reference to: {ref_channels}")
    
    return raw_reref



# ============================================================
# Montage from EGI coordinates.xml
# ============================================================

import re
import xml.etree.ElementTree as ET

def infer_scale_to_m(xyz_array):
    """
    Infer unit scale to meters from value magnitudes.
    - If max |coord| > 30  -> assume mm  -> 1e-3
    - If 2 < max |coord| <= 30 -> assume cm -> 1e-2
    - Else assume meters
    """
    xyz_array = np.asarray(xyz_array, float)
    if xyz_array.size == 0:
        return 1.0
    max_abs = float(np.nanmax(np.abs(xyz_array)))
    if max_abs > 30.0:
        return 1e-3  # mm -> m
    elif max_abs > 2.0:
        return 1e-2  # cm -> m
    else:
        return 1.0   # already meters

def apply_coordinates_xml(raw, xml_path, set_ref_misc=True, verbose=True):
    """
    Parse EGI MFF coordinates.xml and apply a DigMontage (3D positions) to 'raw'.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw object already loaded from .mff (read_raw_egi / your load_eeg_data).
        Must contain EEG channels to map positions onto.
    xml_path : str or Path
        Path to coordinates.xml (inside the .mff folder or exported alongside).
    set_ref_misc : bool
        If True, any channel marked as 'reference' in XML will be set to 'misc'
        to avoid participating in forward/inverse.
    verbose : bool
        Print summary.

    Returns
    -------
    montage : mne.channels.DigMontage
        The montage applied to raw.
    scale : float
        The scale factor applied to coordinates to convert to meters.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"coordinates.xml not found: {xml_path}")

    # Parse XML (with/without namespace)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"egi": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    def ftext(elem, tag):
        return elem.findtext(f"egi:{tag}", default=elem.findtext(tag), namespaces=ns)

    # Gather all sensors first to infer unit scale
    sensors = root.findall(".//egi:sensor", ns) if ns else root.findall(".//sensor")
    if len(sensors) == 0:
        raise RuntimeError("No <sensor> entries found in coordinates.xml")

    pts_all = []
    for s in sensors:
        x = float(ftext(s, "x")); y = float(ftext(s, "y")); z = float(ftext(s, "z"))
        pts_all.append([x, y, z])
    scale = infer_scale_to_m(np.asarray(pts_all))

    # Second pass: build maps
    ch_xyz_by_number = {}   # EEG sensors by numeric index
    nas = lpa = rpa = None
    ref_names = []

    for s in sensors:
        name = ftext(s, "name") or ""
        num  = ftext(s, "number")
        typ  = ftext(s, "type")
        try:
            typ = int(typ)
        except Exception:
            typ = None

        x = float(ftext(s, "x")) * scale
        y = float(ftext(s, "y")) * scale
        z = float(ftext(s, "z")) * scale
        xyz = np.array([x, y, z], float)

        # EGI convention (commonly): type 0=EEG, 1=reference, 2=fiducial
        if typ == 0:
            if num is not None:
                try:
                    ch_xyz_by_number[int(num)] = xyz
                except Exception:
                    pass
        elif typ == 1:
            ref_names.append(name)
        elif typ == 2:
            nm = (name or "").lower()
            if "nas" in nm or "nasion" in nm:
                nas = xyz
            elif "lpa" in nm or ("left" in nm and ("aur" in nm or "periaur" in nm)):
                lpa = xyz
            elif "rpa" in nm or ("right" in nm and ("aur" in nm or "periaur" in nm)):
                rpa = xyz

    # Map positions to raw EEG channel names: E1/E2/... or plain numbers at the end
    ch_pos = {}
    for ch in raw.ch_names:
        if raw.get_channel_types(picks=[ch])[0] != "eeg":
            continue
        # Try 'E123' or trailing digits
        m = re.search(r'(\d+)$', ch.replace('E', ''))
        if m:
            idx = int(m.group(1))
            if idx in ch_xyz_by_number:
                ch_pos[ch] = ch_xyz_by_number[idx]

    if verbose:
        print(f"Mapped {len(ch_pos)} EEG channels from coordinates.xml to raw channels.")
        if nas is None or lpa is None or rpa is None:
            print("Warning: Missing one or more fiducials (NAS/LPA/RPA) in XML; montage will still be applied.")

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=nas, lpa=lpa, rpa=rpa,
        coord_frame='head'
    )
    raw.set_montage(montage, on_missing='warn')

    # Optionally set reference channels to misc so they don't enter forward/inverse
    if set_ref_misc and ref_names:
        renames = {}
        for rn in ref_names:
            # Some datasets use exact same name as channel; if present, set to misc
            if rn in raw.ch_names:
                renames[rn] = 'misc'
        if renames:
            raw.set_channel_types(renames)
            if verbose:
                print(f"Set reference channels to misc: {sorted(renames.keys())}")

    # Quick report
    if verbose:
        eeg_digs = [d['r'] for d in raw.info.get('dig', []) if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG]
        if eeg_digs:
            eeg_digs = np.array(eeg_digs)
            xmin, ymin, zmin = eeg_digs.min(axis=0)
            xmax, ymax, zmax = eeg_digs.max(axis=0)
            print(f"EEG coordinate ranges (meters): "
                  f"X {xmin:.4f}→{xmax:.4f}, Y {ymin:.4f}→{ymax:.4f}, Z {zmin:.4f}→{zmax:.4f}")
            if np.allclose(eeg_digs[:, 2], eeg_digs[0, 2]):
                print("Note: Z nearly constant. Check XML and channel mapping if this is unexpected.")

    return montage, scale
