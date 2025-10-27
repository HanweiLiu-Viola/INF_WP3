"""
EEG-LFP Alignment and Synchronization Module
============================================

Functions for synchronizing EEG and LFP recordings based on stimulation artifacts.
"""

import numpy as np
import mne
from scipy import signal


# ============================================================
# Duration-Based Synchronization
# ============================================================

def find_threshold_from_duration(stim_power, times, target_duration, tolerance=5):
    """
    Find optimal threshold that gives segment matching target duration.
    
    Parameters
    ----------
    stim_power : np.ndarray
        EEG stimulation power array
    times : np.ndarray
        Time array in seconds
    target_duration : float
        Target duration from LFP (seconds)
    tolerance : float
        Acceptable difference (seconds)
        
    Returns
    -------
    tuple
        (best_threshold, best_segment, results, power_norm)
    """
    # Normalize power to 0-1
    power_norm = (stim_power - np.min(stim_power)) / (np.max(stim_power) - np.min(stim_power))
    
    # Try different threshold values
    thresholds = np.linspace(0.1, 0.9, 80)
    
    best_threshold = None
    best_segment = None
    best_diff = float('inf')
    
    results = []
    
    for thresh in thresholds:
        # Apply threshold
        stim_on = power_norm > thresh
        
        # Find segments
        transitions = np.diff(np.concatenate([[False], stim_on, [False]]).astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        # Check each segment
        for start, end in zip(starts, ends):
            duration = times[end-1] - times[start]
            
            # Only consider segments close to target
            if duration > 20:  # At least 20 seconds
                diff = abs(duration - target_duration)
                results.append((thresh, times[start], times[end-1], duration, diff))
                
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = thresh
                    best_segment = (times[start], times[end-1], duration)
    
    return best_threshold, best_segment, results, power_norm


def synchronize_eeg_lfp_simple(raw_eeg, stim_freq, lfp_duration):
    """
    Simple synchronization using LFP duration to find threshold.
    
    Parameters
    ----------
    raw_eeg : mne.io.Raw
        MNE raw object
    stim_freq : float
        Stimulation frequency (Hz)
    lfp_duration : float
        Known LFP stim segment duration (seconds)
        
    Returns
    -------
    tuple
        (eeg_segment, sync_info)
    """
    print("\n" + "="*60)
    print("EEG-LFP SYNCHRONIZATION (Duration-based)")
    print("="*60)
    
    # Step 1: Extract stim power from EEG
    print(f"\n[1] Extracting {stim_freq} Hz power from EEG...")
    eeg_picks = mne.pick_types(raw_eeg.info, eeg=True)
    data = raw_eeg.get_data(picks=eeg_picks)
    sfreq = raw_eeg.info['sfreq']
    times = raw_eeg.times
    
    # Bandpass filter
    filtered = mne.filter.filter_data(data, sfreq=sfreq,
                                      l_freq=stim_freq - 3,
                                      h_freq=stim_freq + 3,
                                      verbose=False)
    
    # Calculate envelope
    analytic = signal.hilbert(filtered, axis=1)
    envelope = np.abs(analytic)
    stim_power = np.mean(envelope, axis=0)
    
    # Smooth
    window = int(sfreq * 1.0)
    stim_power = np.convolve(stim_power, np.ones(window)/window, mode='same')
    
    print(f"    Power range: {np.min(stim_power):.2f} - {np.max(stim_power):.2f}")
    
    # Step 2: Find optimal threshold based on LFP duration
    print(f"\n[2] Finding threshold to match LFP duration ({lfp_duration:.1f}s)...")
    best_threshold, best_segment, results, power_norm = find_threshold_from_duration(
        stim_power, times, target_duration=lfp_duration, tolerance=5
    )
    
    if best_threshold is None:
        print("\n❌ ERROR: Could not find matching segment!")
        return None, None
    
    print(f"\n✅ Found optimal threshold: {best_threshold:.3f} (normalized)")
    print(f"    Segment: {best_segment[0]:.1f}s - {best_segment[1]:.1f}s")
    print(f"    Duration: {best_segment[2]:.1f}s")
    print(f"    Difference from LFP: {abs(best_segment[2] - lfp_duration):.2f}s")
    
    # Step 3: Crop EEG
    print(f"\n[3] Cropping EEG...")
    eeg_segment = raw_eeg.copy().crop(tmin=best_segment[0], tmax=best_segment[1])
    
    # Prepare output
    sync_info = {
        'threshold_normalized': best_threshold,
        'threshold_actual': best_threshold * (np.max(stim_power) - np.min(stim_power)) + np.min(stim_power),
        'eeg_start': best_segment[0],
        'eeg_end': best_segment[1],
        'eeg_duration': best_segment[2],
        'lfp_duration': lfp_duration,
        'duration_diff': abs(best_segment[2] - lfp_duration),
        'stim_power': stim_power,
        'times': times,
        'all_results': results
    }
    
    print("\n" + "="*60)
    print("✅ SYNCHRONIZATION COMPLETE")
    print("="*60)
    
    return eeg_segment, sync_info


# ============================================================
# Advanced Synchronization with Diagnostics
# ============================================================

def detect_stim_onsets_adaptive(stim_power, times, threshold_method='otsu',
                               min_duration_sec=60):
    """
    Adaptive threshold detection with multiple methods.
    
    Parameters
    ----------
    stim_power : np.ndarray
        Power envelope array
    times : np.ndarray
        Time array
    threshold_method : str
        'otsu', 'percentile', or 'manual'
    min_duration_sec : float
        Minimum duration for valid stim segment
        
    Returns
    -------
    tuple
        (segments, threshold, diagnostics)
    """
    # Normalize power
    power_norm = (stim_power - np.min(stim_power)) / (np.max(stim_power) - np.min(stim_power) + 1e-10)
    
    # Calculate threshold using different methods
    if threshold_method == 'otsu':
        # Otsu's method (binary segmentation)
        hist, bin_edges = np.histogram(power_norm, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate Otsu threshold
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2 + 1e-10))[::-1]
        
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        threshold_norm = bin_centers[np.argmax(variance)]
        
    elif threshold_method == 'percentile':
        threshold_norm = np.percentile(power_norm, 50)
        
    else:  # manual
        threshold_norm = 0.5
    
    # Binary mask
    stim_on = power_norm > threshold_norm
    
    # Find transitions
    transitions = np.diff(np.concatenate([[False], stim_on, [False]]).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Filter by minimum duration
    sfreq = 1 / (times[1] - times[0]) if len(times) > 1 else 250
    min_samples = int(min_duration_sec * sfreq)
    
    segments = []
    for start, end in zip(starts, ends):
        duration = end - start
        if duration >= min_samples:
            start_time = times[start]
            end_time = times[end - 1]
            duration_sec = end_time - start_time
            segments.append((start_time, end_time, duration_sec, start, end))
    
    # Diagnostic information
    threshold_actual = threshold_norm * (np.max(stim_power) - np.min(stim_power)) + np.min(stim_power)
    
    diagnostics = {
        'power_norm': power_norm,
        'threshold_norm': threshold_norm,
        'threshold_actual': threshold_actual,
        'stim_on': stim_on,
        'all_starts': starts,
        'all_ends': ends
    }
    
    return segments, threshold_actual, diagnostics


def match_eeg_to_lfp_segment(eeg_segments, lfp_duration, tolerance_sec=5):
    """
    Match EEG segment to LFP segment based on duration.
    
    Parameters
    ----------
    eeg_segments : list
        List of EEG segments (start, end, duration, start_idx, end_idx)
    lfp_duration : float
        Target LFP segment duration in seconds
    tolerance_sec : float
        Acceptable duration difference
        
    Returns
    -------
    tuple or None
        Best matching segment
    """
    if len(eeg_segments) == 0:
        return None
    
    # Find segment with duration closest to LFP
    duration_diffs = [abs(seg[2] - lfp_duration) for seg in eeg_segments]
    best_idx = np.argmin(duration_diffs)
    
    if duration_diffs[best_idx] > tolerance_sec:
        print(f"Warning: Best match differs by {duration_diffs[best_idx]:.1f} sec from LFP duration")
    
    return eeg_segments[best_idx]


def synchronize_eeg_lfp_advanced(raw_eeg, stim_freq, lfp_duration,
                                 detection_method='combined',
                                 threshold_method='otsu',
                                 min_duration_sec=30):
    """
    Advanced synchronization with diagnostics.
    
    Parameters
    ----------
    raw_eeg : mne.io.Raw
        MNE raw object
    stim_freq : float
        Stimulation frequency (Hz)
    lfp_duration : float
        Known LFP segment duration (seconds)
    detection_method : str
        'hilbert', 'rms', or 'combined'
    threshold_method : str
        'otsu', 'percentile', or 'manual'
    min_duration_sec : float
        Minimum segment duration
        
    Returns
    -------
    tuple
        (eeg_segment, sync_info)
    """
    from . import eeg_io
    
    print("\n" + "="*60)
    print("EEG-LFP SYNCHRONIZATION (Advanced)")
    print("="*60)
    
    # Step 1: Detect stim artifacts
    print(f"\n[1] Detecting stimulation artifacts (method: {detection_method})...")
    stim_power, eeg_times = eeg_io.extract_stim_power(
        raw_eeg, stim_freq, 
        freq_tolerance=3,
        method='hilbert'
    )
    
    # Step 2: Detect segments
    print(f"\n[2] Detecting segments (threshold: {threshold_method})...")
    segments, threshold, diagnostics = detect_stim_onsets_adaptive(
        stim_power, eeg_times,
        threshold_method=threshold_method,
        min_duration_sec=min_duration_sec
    )
    
    print(f"    Found {len(segments)} segments")
    
    # Step 3: Match to LFP
    print(f"\n[3] Matching to LFP segment (target: {lfp_duration:.1f}s)...")
    best_segment = match_eeg_to_lfp_segment(segments, lfp_duration)
    
    if best_segment is None:
        print("\n❌ ERROR: No matching EEG segment found!")
        return None, None
    
    start_time, end_time, duration, start_idx, end_idx = best_segment
    print(f"    Best match: {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s)")
    
    # Step 4: Crop EEG
    eeg_segment = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)
    
    sync_info = {
        'eeg_start_time': start_time,
        'eeg_end_time': end_time,
        'eeg_duration': duration,
        'lfp_duration': lfp_duration,
        'duration_diff': abs(duration - lfp_duration),
        'all_eeg_segments': segments,
        'stim_power': stim_power,
        'threshold': threshold,
        'diagnostics': diagnostics,
        'times': eeg_times
    }
    
    return eeg_segment, sync_info
