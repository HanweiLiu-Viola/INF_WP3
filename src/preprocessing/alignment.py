"""
EEG-LFP Alignment and Synchronization Module
============================================

Functions for synchronizing EEG and LFP recordings based on stimulation artifacts.

Key Features:
-------------
1. Duration-based synchronization using LFP segment as reference
2. Event-based synchronization with unified STIM_ON/STIM_OFF labels
3. Temporal mapping between LFP and EEG time axes
4. Full data preservation for source reconstruction and causality analysis

Workflow:
---------
1. Extract LFP stimulation events from df_ts_stim_aligned
2. Synchronize EEG with LFP using middle long stim-on segment
3. Map all LFP events to EEG time axis using sync anchor
4. Add unified events to both EEG and LFP for downstream analysis

Note:
-----
- EEG (500 Hz) and LFP (250 Hz) maintain their original sampling rates
- Event synchronization is based on physical time (seconds), not sample indices
- Resampling to common rate (e.g., 250 Hz) only needed for final causality analysis
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal


# ============================================================
# Event Detection from LFP Stimulation Time Series
# ============================================================

def detect_stim_events_from_lfp(df_stim, stim_column='stim_on', time_column='timestamp'):
    """
    Extract STIM_ON and STIM_OFF events from LFP stimulation time series.
    
    This function detects transitions in the stimulation state and creates
    a list of events with their timing in seconds.
    
    Parameters
    ----------
    df_stim : pd.DataFrame
        DataFrame containing stimulation time series
        Expected columns: 'timestamp' (in seconds), 'stim_on' (boolean or 0/1)
    stim_column : str
        Column name for stimulation state (default: 'stim_on')
    time_column : str
        Column name for timestamps in seconds (default: 'timestamp')
        
    Returns
    -------
    events_list : list of dict
        List of events with keys:
        - 'event_type': 'STIM_ON' or 'STIM_OFF'
        - 'time_sec': event time in seconds (LFP time axis)
        - 'sample_idx': sample index in original LFP data
        - 'segment_id': segment number (0, 1, 2, ...)
        
    Examples
    --------
    >>> events = detect_stim_events_from_lfp(df_ts_stim_aligned)
    >>> print(f"Found {len(events)} events")
    >>> print(f"First STIM_ON at {events[0]['time_sec']:.2f}s")
    """
    print("\n" + "="*70)
    print("DETECTING STIMULATION EVENTS FROM LFP")
    print("="*70)
    
    # Get stim state and times
    stim_state = df_stim[stim_column].values.astype(bool)
    times = df_stim[time_column].values
    
    # Detect transitions
    # Add False at boundaries to detect start/end events
    stim_padded = np.concatenate([[False], stim_state, [False]])
    transitions = np.diff(stim_padded.astype(int))
    
    # Find ON and OFF transitions
    on_indices = np.where(transitions == 1)[0]  # 0 -> 1 transitions
    off_indices = np.where(transitions == -1)[0]  # 1 -> 0 transitions
    
    print(f"\nFound {len(on_indices)} STIM_ON events")
    print(f"Found {len(off_indices)} STIM_OFF events")
    
    # Create events list
    events_list = []
    segment_id = 0
    
    # Interleave ON and OFF events
    for on_idx, off_idx in zip(on_indices, off_indices):
        # STIM_ON event
        events_list.append({
            'event_type': 'STIM_ON',
            'time_sec': times[on_idx],
            'sample_idx': on_idx,
            'segment_id': segment_id
        })
        
        # STIM_OFF event
        events_list.append({
            'event_type': 'STIM_OFF',
            'time_sec': times[off_idx - 1] if off_idx > 0 else times[off_idx],
            'sample_idx': off_idx - 1 if off_idx > 0 else off_idx,
            'segment_id': segment_id
        })
        
        segment_id += 1
    
    # Sort by time
    events_list.sort(key=lambda x: x['time_sec'])
    
    # Print summary
    print("\n" + "-"*70)
    print("EVENT SUMMARY:")
    print("-"*70)
    for i, event in enumerate(events_list[:10]):  # Show first 10 events
        print(f"  Event {i}: {event['event_type']:10s} at {event['time_sec']:8.2f}s "
              f"(segment {event['segment_id']})")
    if len(events_list) > 10:
        print(f"  ... and {len(events_list) - 10} more events")
    
    print("\n" + "="*70)
    print(f"✓ DETECTED {len(events_list)} TOTAL EVENTS")
    print("="*70)
    
    return events_list


def identify_stim_segments(events_list, min_off_duration=10.0):
    """
    Identify distinct stimulation segments from events.
    
    Parameters
    ----------
    events_list : list of dict
        Events from detect_stim_events_from_lfp
    min_off_duration : float
        Minimum OFF duration to consider separate segments (seconds)
        
    Returns
    -------
    segments : list of dict
        List of segments with keys:
        - 'segment_id': segment number
        - 'start_time': segment start time (sec)
        - 'end_time': segment end time (sec)
        - 'duration': segment duration (sec)
        - 'type': 'stim_on' or 'stim_off'
    """
    segments = []
    
    for i in range(len(events_list) - 1):
        event = events_list[i]
        next_event = events_list[i + 1]
        
        start_time = event['time_sec']
        end_time = next_event['time_sec']
        duration = end_time - start_time
        
        segment = {
            'segment_id': event['segment_id'],
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'type': 'stim_on' if event['event_type'] == 'STIM_ON' else 'stim_off'
        }
        
        segments.append(segment)
    
    return segments


# ============================================================
# Event Mapping from LFP to EEG Time Axis
# ============================================================

def map_lfp_events_to_eeg(lfp_events, raw_eeg, sync_anchor):
    """
    Map LFP events to EEG time axis using synchronization anchor.
    
    This function uses the synchronization anchor point (from the middle long
    stim-on segment) to calculate the time offset between LFP and EEG, then
    maps all LFP events to the EEG time axis.
    
    Parameters
    ----------
    lfp_events : list of dict
        Events from detect_stim_events_from_lfp
    raw_eeg : mne.io.Raw
        EEG raw data object
    sync_anchor : dict
        Synchronization anchor information with keys:
        - 'eeg_start': EEG time of sync point (seconds)
        - 'lfp_start': LFP time of sync point (seconds)
        
    Returns
    -------
    eeg_events : np.ndarray
        MNE events array (n_events, 3) with columns:
        [sample, prev_value, event_id]
    event_id : dict
        Event ID mapping: {'STIM_ON': 1, 'STIM_OFF': 2}
    events_info : list of dict
        Extended event information including LFP times
        
    Examples
    --------
    >>> sync_anchor = {
    ...     'eeg_start': sync_info['eeg_start'],
    ...     'lfp_start': lfp_segment_start_time
    ... }
    >>> eeg_events, event_id, events_info = map_lfp_events_to_eeg(
    ...     lfp_events, raw_eeg, sync_anchor
    ... )
    """
    print("\n" + "="*70)
    print("MAPPING LFP EVENTS TO EEG TIME AXIS")
    print("="*70)
    
    # Calculate time offset
    time_offset = sync_anchor['eeg_start'] - sync_anchor['lfp_start']
    
    print(f"\nSynchronization anchor:")
    print(f"  EEG time:  {sync_anchor['eeg_start']:.2f}s")
    print(f"  LFP time:  {sync_anchor['lfp_start']:.2f}s")
    print(f"  Offset:    {time_offset:.2f}s")
    print(f"\nFormula: EEG_time = LFP_time + {time_offset:.2f}s")
    
    # Get EEG info
    sfreq = raw_eeg.info['sfreq']
    eeg_duration = raw_eeg.times[-1]
    
    # Event ID mapping
    event_id = {'STIM_ON': 1, 'STIM_OFF': 2}
    
    # Map events
    eeg_events_list = []
    events_info = []
    n_skipped = 0
    
    print(f"\nMapping {len(lfp_events)} events...")
    
    for event in lfp_events:
        # Calculate EEG time
        eeg_time = event['time_sec'] + time_offset
        
        # Check if event is within EEG recording
        if eeg_time < 0 or eeg_time > eeg_duration:
            n_skipped += 1
            continue
        
        # Convert to sample index
        sample = int(np.round(eeg_time * sfreq))
        
        # Get event ID
        evt_id = event_id[event['event_type']]
        
        # Add to MNE events array [sample, prev_value, event_id]
        eeg_events_list.append([sample, 0, evt_id])
        
        # Store extended info
        events_info.append({
            'event_type': event['event_type'],
            'lfp_time_sec': event['time_sec'],
            'eeg_time_sec': eeg_time,
            'eeg_sample': sample,
            'segment_id': event['segment_id']
        })
    
    # Convert to numpy array
    eeg_events = np.array(eeg_events_list, dtype=int)
    
    # Print summary
    print("\n" + "-"*70)
    print("MAPPING RESULTS:")
    print("-"*70)
    print(f"  Total LFP events:     {len(lfp_events)}")
    print(f"  Mapped to EEG:        {len(eeg_events)}")
    print(f"  Skipped (out of EEG): {n_skipped}")
    print(f"\nFirst few mapped events:")
    for i, info in enumerate(events_info[:5]):
        print(f"  {info['event_type']:10s} | LFP: {info['lfp_time_sec']:8.2f}s → "
              f"EEG: {info['eeg_time_sec']:8.2f}s (sample {info['eeg_sample']})")
    
    print("\n" + "="*70)
    print(f"✓ MAPPED {len(eeg_events)} EVENTS TO EEG")
    print("="*70)
    
    return eeg_events, event_id, events_info


def add_events_to_raw(raw, events, event_id, description_prefix=''):
    """
    Add events to MNE Raw object as annotations.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw data object
    events : np.ndarray
        MNE events array (n_events, 3)
    event_id : dict
        Event ID mapping
    description_prefix : str
        Prefix to add to event descriptions (e.g., 'synced/')
        
    Returns
    -------
    raw : mne.io.Raw
        Raw object with added annotations (modified in place)
    annotations : mne.Annotations
        The annotations object that was added
    """
    # Create inverse event_id mapping
    id_to_name = {v: k for k, v in event_id.items()}
    
    # Get event times and descriptions
    sfreq = raw.info['sfreq']
    onset = events[:, 0] / sfreq  # Convert samples to seconds
    duration = np.zeros(len(events))  # Instantaneous events
    description = [description_prefix + id_to_name[evt_id] 
                   for evt_id in events[:, 2]]
    
    # Create annotations
    annotations = mne.Annotations(onset=onset,
                                  duration=duration,
                                  description=description,
                                  orig_time=raw.info['meas_date'])
    
    # Add to raw
    raw.set_annotations(raw.annotations + annotations)
    
    print(f"\n✓ Added {len(events)} annotations to Raw object")
    
    return raw, annotations


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
    
    This function finds the EEG segment that matches the LFP stimulation
    duration, establishing a synchronization anchor point.
    
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
    sync_info : dict
        Synchronization information with keys:
        - 'eeg_start': EEG segment start time (seconds)
        - 'eeg_end': EEG segment end time (seconds)
        - 'eeg_duration': EEG segment duration (seconds)
        - 'lfp_duration': LFP segment duration (seconds)
        - 'threshold_normalized': Normalized threshold used
        - 'threshold_actual': Actual power threshold used
        - 'duration_diff': Difference between EEG and LFP durations
        - 'stim_power': Full stimulation power array
        - 'times': EEG time array
        - 'all_results': All candidate segments found
        
    Note
    ----
    This function does NOT crop the EEG data. It only identifies the
    synchronization segment and returns the timing information. The full
    EEG data should be preserved for source reconstruction.
    """
    print("\n" + "="*70)
    print("EEG-LFP SYNCHRONIZATION (Duration-based)")
    print("="*70)
    
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
        print("\n✗ ERROR: Could not find matching segment!")
        return None
    
    print(f"\n✓ Found optimal threshold: {best_threshold:.3f} (normalized)")
    print(f"    Segment: {best_segment[0]:.1f}s - {best_segment[1]:.1f}s")
    print(f"    Duration: {best_segment[2]:.1f}s")
    print(f"    Difference from LFP: {abs(best_segment[2] - lfp_duration):.2f}s")
    
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
    
    print("\n" + "="*70)
    print("✓ SYNCHRONIZATION COMPLETE")
    print(f"  EEG data preserved: {len(raw_eeg.times)} samples ({raw_eeg.times[-1]:.1f}s)")
    print("="*70)
    
    return sync_info


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
    sync_info : dict
        Synchronization information dictionary
        
    Note
    ----
    This function does NOT crop the EEG data. It returns sync_info
    similar to synchronize_eeg_lfp_simple() but with additional diagnostics.
    """
    print("\n" + "="*70)
    print("EEG-LFP SYNCHRONIZATION (Advanced)")
    print("="*70)
    
    # Step 1: Extract stim power (simplified - no external dependency)
    print(f"\n[1] Detecting stimulation artifacts (method: {detection_method})...")
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
    
    # Step 2: Detect segments
    print(f"\n[2] Detecting segments (threshold: {threshold_method})...")
    segments, threshold, diagnostics = detect_stim_onsets_adaptive(
        stim_power, times,
        threshold_method=threshold_method,
        min_duration_sec=min_duration_sec
    )
    
    print(f"    Found {len(segments)} segments")
    
    # Step 3: Match to LFP
    print(f"\n[3] Matching to LFP segment (target: {lfp_duration:.1f}s)...")
    best_segment = match_eeg_to_lfp_segment(segments, lfp_duration)
    
    if best_segment is None:
        print("\n✗ ERROR: No matching EEG segment found!")
        return None
    
    start_time, end_time, duration, start_idx, end_idx = best_segment
    print(f"    Best match: {start_time:.1f}s - {end_time:.1f}s (duration: {duration:.1f}s)")
    
    sync_info = {
        'eeg_start': start_time,
        'eeg_end': end_time,
        'eeg_duration': duration,
        'lfp_duration': lfp_duration,
        'duration_diff': abs(duration - lfp_duration),
        'all_eeg_segments': segments,
        'stim_power': stim_power,
        'threshold': threshold,
        'diagnostics': diagnostics,
        'times': times
    }
    
    print("\n" + "="*70)
    print("✓ SYNCHRONIZATION COMPLETE")
    print(f"  EEG data preserved: {len(raw_eeg.times)} samples ({raw_eeg.times[-1]:.1f}s)")
    print("="*70)
    
    return sync_info


# ============================================================
# Artifact Removal Functions
# ============================================================

def remove_initial_artifacts(raw_lfp, segment_start, segment_end, 
                             artifact_threshold_std=5.0, channel='LFP_L'):
    """
    Remove initial artifacts from LFP segment (matching lfp_io.extract_middle_segment).
    
    Parameters
    ----------
    raw_lfp : mne.io.Raw
        LFP raw data
    segment_start : float
        Segment start time (seconds)
    segment_end : float
        Segment end time (seconds)
    artifact_threshold_std : float
        Threshold in standard deviations for artifact detection
    channel : str
        LFP channel to use for artifact detection
        
    Returns
    -------
    tuple
        (new_start_time, n_samples_removed)
    """
    # Extract segment
    segment = raw_lfp.copy().crop(tmin=segment_start, tmax=segment_end)
    
    # Get voltage data
    if channel not in segment.ch_names:
        print(f"Warning: Channel {channel} not found, skipping artifact removal")
        return segment_start, 0
    
    ch_idx = segment.ch_names.index(channel)
    voltage = segment.get_data()[ch_idx]
    
    # Calculate statistics
    median_voltage = np.median(voltage)
    std_voltage = np.std(voltage)
    
    # Detect extreme outliers in the first 10% of data
    initial_portion = int(len(voltage) * 0.1)
    if initial_portion <= 10:
        return segment_start, 0
    
    initial_data = voltage[:initial_portion]
    
    # Find samples that are extreme outliers
    outliers = np.abs(initial_data - median_voltage) > (artifact_threshold_std * std_voltage)
    
    if not np.any(outliers):
        return segment_start, 0
    
    # Find the last outlier position
    last_outlier_idx = np.where(outliers)[0][-1]
    
    # Skip past the artifacts (add 1 second buffer after last outlier)
    sfreq = segment.info['sfreq']
    samples_to_skip = last_outlier_idx + int(sfreq)  # 1 second buffer
    
    if samples_to_skip >= len(voltage):
        print("Warning: Artifact removal would remove entire segment")
        return segment_start, 0
    
    # Calculate new start time
    new_start_time = segment_start + (samples_to_skip / sfreq)
    
    return new_start_time, samples_to_skip


# ============================================================
# Complete Workflow Function
# ============================================================

def complete_eeg_lfp_synchronization(raw_eeg, raw_lfp, df_stim, 
                                     stim_freq=130.0,
                                     middle_segment_id=1,
                                     stim_column='stim_on',
                                     time_column='timestamp',
                                     remove_artifacts=False,
                                     artifact_threshold_std=5.0,
                                     lfp_channel='LFP_L'):
    """
    Complete EEG-LFP synchronization workflow with unified event labeling.
    
    This is the main function that orchestrates the entire synchronization
    process:
    1. Extract events from LFP stimulation time series
    2. Identify the middle long stim-on segment
    3. Optionally remove initial artifacts from middle segment (matches lfp_io.extract_middle_segment)
    4. Synchronize EEG with LFP using this segment as anchor
    5. Map all LFP events to EEG time axis
    6. Add unified events to both EEG and LFP
    
    Parameters
    ----------
    raw_eeg : mne.io.Raw
        EEG raw data (full recording, not cropped)
    raw_lfp : mne.io.Raw
        LFP raw data (full recording, not cropped)
    df_stim : pd.DataFrame
        LFP stimulation time series with columns:
        - 'timestamp': time in seconds (LFP time axis)
        - 'stim_on': boolean or 0/1 stimulation state
    stim_freq : float
        Stimulation frequency in Hz (default: 130.0)
    middle_segment_id : int
        Segment ID of the middle long stim-on segment (default: 1)
    stim_column : str
        Column name for stimulation state
    time_column : str
        Column name for timestamps
    remove_artifacts : bool
        Whether to remove initial artifacts from middle segment (default: False)
        Set to True to match lfp_io.extract_middle_segment behavior
    artifact_threshold_std : float
        Threshold in standard deviations for artifact detection (default: 5.0)
    lfp_channel : str
        LFP channel to use for artifact detection (default: 'LFP_L')
        
    Returns
    -------
    results : dict
        Complete synchronization results with keys:
        - 'raw_eeg': EEG with added events
        - 'raw_lfp': LFP with added events
        - 'eeg_events': MNE events array for EEG
        - 'lfp_events': Event info for LFP
        - 'event_id': Event ID dictionary
        - 'sync_info': Synchronization information
        - 'sync_anchor': Anchor point used for mapping
        - 'segments': Identified stimulation segments
        - 'artifact_info': Information about removed artifacts (if remove_artifacts=True)
        
    Examples
    --------
    >>> # Standard synchronization (no artifact removal)
    >>> results = complete_eeg_lfp_synchronization(
    ...     raw_eeg, raw_lfp, df_ts_stim_aligned,
    ...     stim_freq=130.0, middle_segment_id=1
    ... )
    >>> 
    >>> # Match lfp_io.extract_middle_segment behavior (with artifact removal)
    >>> results = complete_eeg_lfp_synchronization(
    ...     raw_eeg, raw_lfp, df_ts_stim_aligned,
    ...     stim_freq=130.0, middle_segment_id=1,
    ...     remove_artifacts=True,
    ...     artifact_threshold_std=5.0,
    ...     lfp_channel='LFP_L'
    ... )
    """
    print("\n" + "="*80)
    print(" "*20 + "COMPLETE EEG-LFP SYNCHRONIZATION")
    print("="*80)
    
    # Step 1: Extract LFP events
    print("\n" + "STEP 1: Extract events from LFP stimulation time series")
    print("-"*80)
    lfp_events = detect_stim_events_from_lfp(df_stim, stim_column, time_column)
    
    # Step 2: Identify segments
    print("\n" + "STEP 2: Identify stimulation segments")
    print("-"*80)
    segments = identify_stim_segments(lfp_events)
    
    # Find middle long stim-on segment
    stim_on_segments = [s for s in segments if s['type'] == 'stim_on']
    
    if middle_segment_id >= len(stim_on_segments):
        raise ValueError(f"Segment ID {middle_segment_id} not found. "
                        f"Only {len(stim_on_segments)} stim-on segments available.")
    
    middle_segment = stim_on_segments[middle_segment_id]
    
    print(f"\nUsing segment {middle_segment_id} as synchronization anchor:")
    print(f"  Start:    {middle_segment['start_time']:.2f}s")
    print(f"  End:      {middle_segment['end_time']:.2f}s")
    print(f"  Duration: {middle_segment['duration']:.2f}s")
    
    # Step 2.5: Remove initial artifacts if requested
    artifact_info = None
    lfp_segment_start = middle_segment['start_time']
    lfp_segment_duration = middle_segment['duration']
    
    if remove_artifacts:
        print("\n" + "STEP 2.5: Remove initial artifacts from LFP segment")
        print("-"*80)
        new_start, n_removed = remove_initial_artifacts(
            raw_lfp,
            middle_segment['start_time'],
            middle_segment['end_time'],
            artifact_threshold_std=artifact_threshold_std,
            channel=lfp_channel
        )
        
        if n_removed > 0:
            lfp_segment_start = new_start
            lfp_segment_duration = middle_segment['end_time'] - new_start
            
            artifact_info = {
                'n_samples_removed': n_removed,
                'time_removed_sec': new_start - middle_segment['start_time'],
                'original_start': middle_segment['start_time'],
                'new_start': new_start,
                'new_duration': lfp_segment_duration
            }
            
            print(f"  Removed {n_removed} samples containing artifacts")
            print(f"  Time removed: {artifact_info['time_removed_sec']:.2f}s")
            print(f"  New start: {new_start:.2f}s")
            print(f"  New duration: {lfp_segment_duration:.2f}s")
        else:
            print("  No artifacts detected")
    
    # Step 3: Synchronize EEG with LFP
    print("\n" + "STEP 3: Synchronize EEG with LFP using middle segment")
    print("-"*80)
    sync_info = synchronize_eeg_lfp_simple(
        raw_eeg, 
        stim_freq=stim_freq,
        lfp_duration=lfp_segment_duration
    )
    
    if sync_info is None:
        raise RuntimeError("Synchronization failed!")
    
    # Step 4: Create sync anchor
    print("\n" + "STEP 4: Create synchronization anchor")
    print("-"*80)
    sync_anchor = {
        'eeg_start': sync_info['eeg_start'],
        'lfp_start': lfp_segment_start
    }
    
    print(f"Anchor point established:")
    print(f"  EEG: {sync_anchor['eeg_start']:.2f}s")
    print(f"  LFP: {sync_anchor['lfp_start']:.2f}s")
    print(f"  Time offset: {sync_anchor['eeg_start'] - sync_anchor['lfp_start']:.2f}s")
    
    if remove_artifacts and artifact_info and artifact_info['n_samples_removed'] > 0:
        print(f"\n  Note: LFP anchor adjusted by {artifact_info['time_removed_sec']:.2f}s")
        print(f"        due to artifact removal")
    
    # Step 5: Map LFP events to EEG
    print("\n" + "STEP 5: Map all LFP events to EEG time axis")
    print("-"*80)
    eeg_events, event_id, eeg_events_info = map_lfp_events_to_eeg(
        lfp_events, raw_eeg, sync_anchor
    )
    
    # Step 6: Add events to Raw objects
    print("\n" + "STEP 6: Add events to Raw objects")
    print("-"*80)
    
    # Add to EEG
    print("\nAdding events to EEG...")
    raw_eeg, eeg_annotations = add_events_to_raw(
        raw_eeg, eeg_events, event_id, description_prefix='synced/'
    )
    
    # Create LFP events array (using LFP time axis)
    print("\nCreating LFP events array...")
    lfp_sfreq = raw_lfp.info['sfreq']
    lfp_events_array = []
    
    for event in lfp_events:
        sample = int(np.round(event['time_sec'] * lfp_sfreq))
        evt_id = event_id[event['event_type']]
        lfp_events_array.append([sample, 0, evt_id])
    
    lfp_events_array = np.array(lfp_events_array, dtype=int)
    
    # Add to LFP
    raw_lfp, lfp_annotations = add_events_to_raw(
        raw_lfp, lfp_events_array, event_id, description_prefix='synced/'
    )
    
    # Step 7: Prepare results
    print("\n" + "="*80)
    print(" "*25 + "SYNCHRONIZATION SUMMARY")
    print("="*80)
    print(f"\nTotal events detected:     {len(lfp_events)}")
    print(f"Events mapped to EEG:      {len(eeg_events)}")
    print(f"Events in LFP:             {len(lfp_events_array)}")
    print(f"Sync anchor (EEG):         {sync_anchor['eeg_start']:.2f}s")
    print(f"Sync anchor (LFP):         {sync_anchor['lfp_start']:.2f}s")
    print(f"Time offset:               {sync_anchor['eeg_start'] - sync_anchor['lfp_start']:.2f}s")
    
    if artifact_info:
        print(f"\nArtifact removal:")
        print(f"  Samples removed:         {artifact_info['n_samples_removed']}")
        print(f"  Time removed:            {artifact_info['time_removed_sec']:.2f}s")
    
    print(f"\nEEG recording duration:    {raw_eeg.times[-1]:.2f}s")
    print(f"LFP recording duration:    {raw_lfp.times[-1]:.2f}s")
    print(f"\nEvent types: {list(event_id.keys())}")
    
    print("\n" + "="*80)
    print(" "*25 + "✓ SYNCHRONIZATION COMPLETE")
    print("="*80)
    
    results = {
        'raw_eeg': raw_eeg,
        'raw_lfp': raw_lfp,
        'eeg_events': eeg_events,
        'eeg_events_info': eeg_events_info,
        'lfp_events': lfp_events,
        'lfp_events_array': lfp_events_array,
        'event_id': event_id,
        'sync_info': sync_info,
        'sync_anchor': sync_anchor,
        'segments': segments,
        'middle_segment': middle_segment,
        'artifact_info': artifact_info
    }
    
    return results


# ============================================================
# Utility Functions
# ============================================================

def get_baseline_segment(results, mode='first_stim_off', segment_index=0):
    """
    Get baseline segment information for noise covariance estimation.
    
    Parameters
    ----------
    results : dict
        Results from complete_eeg_lfp_synchronization
    mode : str
        Baseline selection mode:
        - 'before_first_stim': All time before first STIM_ON
        - 'first_stim_off': First STIM_OFF segment (default)
        - 'specific_stim_off': Specific STIM_OFF segment by index
    segment_index : int
        Index of STIM_OFF segment to use (for 'specific_stim_off' mode)
        
    Returns
    -------
    baseline_info : dict
        Baseline segment information with keys:
        - 'start_time': start time in seconds (LFP time)
        - 'end_time': end time in seconds (LFP time)
        - 'duration': duration in seconds
        - 'description': text description
        - 'eeg_start_time': start time in EEG time axis
        - 'eeg_end_time': end time in EEG time axis
        
    Examples
    --------
    >>> # Use first stim-off segment (typical for short baseline ~2s)
    >>> baseline = get_baseline_segment(results, mode='first_stim_off')
    >>> 
    >>> # Use time before first stimulation
    >>> baseline = get_baseline_segment(results, mode='before_first_stim')
    >>> 
    >>> # Use second stim-off segment
    >>> baseline = get_baseline_segment(results, mode='specific_stim_off', segment_index=1)
    """
    segments = results['segments']
    
    # Calculate time offset for EEG mapping
    time_offset = (results['sync_anchor']['eeg_start'] - 
                  results['sync_anchor']['lfp_start'])
    
    if mode == 'before_first_stim':
        # Use time before first STIM_ON
        first_on_event = next(e for e in results['lfp_events'] 
                             if e['event_type'] == 'STIM_ON')
        
        baseline_info = {
            'start_time': 0.0,
            'end_time': first_on_event['time_sec'],
            'duration': first_on_event['time_sec'],
            'description': 'pre-stimulation baseline',
            'eeg_start_time': 0.0 + time_offset,
            'eeg_end_time': first_on_event['time_sec'] + time_offset
        }
        
    elif mode == 'first_stim_off':
        # Use first STIM_OFF segment
        first_off_segment = next(s for s in segments if s['type'] == 'stim_off')
        
        baseline_info = {
            'start_time': first_off_segment['start_time'],
            'end_time': first_off_segment['end_time'],
            'duration': first_off_segment['duration'],
            'description': 'first stim-off segment',
            'eeg_start_time': first_off_segment['start_time'] + time_offset,
            'eeg_end_time': first_off_segment['end_time'] + time_offset
        }
        
    elif mode == 'specific_stim_off':
        # Use specific STIM_OFF segment
        stim_off_segments = [s for s in segments if s['type'] == 'stim_off']
        
        if segment_index >= len(stim_off_segments):
            raise ValueError(f"Segment index {segment_index} out of range. "
                           f"Only {len(stim_off_segments)} stim-off segments available.")
        
        target_segment = stim_off_segments[segment_index]
        
        baseline_info = {
            'start_time': target_segment['start_time'],
            'end_time': target_segment['end_time'],
            'duration': target_segment['duration'],
            'description': f'stim-off segment {segment_index}',
            'eeg_start_time': target_segment['start_time'] + time_offset,
            'eeg_end_time': target_segment['end_time'] + time_offset
        }
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'before_first_stim', "
                        f"'first_stim_off', or 'specific_stim_off'")
    
    return baseline_info


def get_analysis_segment(results, segment_type='middle_stim_on'):
    """
    Get the segment to use for causality analysis.
    
    Parameters
    ----------
    results : dict
        Results from complete_eeg_lfp_synchronization
    segment_type : str
        'middle_stim_on' (default) or 'all_stim_on'
        
    Returns
    -------
    analysis_info : dict
        Analysis segment information
    """
    if segment_type == 'middle_stim_on':
        middle_seg = results['middle_segment']
        
        # Convert to EEG time
        time_offset = (results['sync_anchor']['eeg_start'] - 
                      results['sync_anchor']['lfp_start'])
        
        analysis_info = {
            'eeg_start': middle_seg['start_time'] + time_offset,
            'eeg_end': middle_seg['end_time'] + time_offset,
            'lfp_start': middle_seg['start_time'],
            'lfp_end': middle_seg['end_time'],
            'duration': middle_seg['duration'],
            'description': 'middle long stim-on segment'
        }
    else:
        # Use all stim-on segments
        stim_on_segments = [s for s in results['segments'] 
                           if s['type'] == 'stim_on']
        
        analysis_info = {
            'segments': stim_on_segments,
            'description': 'all stim-on segments'
        }
    
    return analysis_info


def print_synchronization_report(results):
    """
    Print a comprehensive synchronization report.
    
    Parameters
    ----------
    results : dict
        Results from complete_eeg_lfp_synchronization
    """
    print("\n" + "="*80)
    print(" "*25 + "SYNCHRONIZATION REPORT")
    print("="*80)
    
    # Basic info
    print("\n" + "RECORDINGS:")
    print("-"*80)
    print(f"EEG:")
    print(f"  Duration:      {results['raw_eeg'].times[-1]:.2f}s")
    print(f"  Sampling rate: {results['raw_eeg'].info['sfreq']:.1f} Hz")
    print(f"  N channels:    {len(results['raw_eeg'].ch_names)}")
    
    print(f"\nLFP:")
    print(f"  Duration:      {results['raw_lfp'].times[-1]:.2f}s")
    print(f"  Sampling rate: {results['raw_lfp'].info['sfreq']:.1f} Hz")
    print(f"  N channels:    {len(results['raw_lfp'].ch_names)}")
    
    # Segments
    print("\n" + "STIMULATION SEGMENTS:")
    print("-"*80)
    for i, seg in enumerate(results['segments']):
        print(f"Segment {i} ({seg['type']:10s}): "
              f"{seg['start_time']:7.2f}s - {seg['end_time']:7.2f}s "
              f"(duration: {seg['duration']:6.2f}s)")
    
    # Synchronization
    print("\n" + "SYNCHRONIZATION:")
    print("-"*80)
    print(f"Anchor segment: {results['middle_segment']['segment_id']}")
    print(f"  LFP time: {results['sync_anchor']['lfp_start']:.2f}s")
    print(f"  EEG time: {results['sync_anchor']['eeg_start']:.2f}s")
    print(f"  Offset:   {results['sync_anchor']['eeg_start'] - results['sync_anchor']['lfp_start']:.2f}s")
    
    # Artifact removal info
    if results.get('artifact_info'):
        art = results['artifact_info']
        print(f"\nArtifact removal:")
        print(f"  Samples removed: {art['n_samples_removed']}")
        print(f"  Time removed:    {art['time_removed_sec']:.2f}s")
        print(f"  Original start:  {art['original_start']:.2f}s")
        print(f"  New start:       {art['new_start']:.2f}s")
        print(f"  New duration:    {art['new_duration']:.2f}s")
    
    # Events
    print("\n" + "EVENTS:")
    print("-"*80)
    print(f"Total events:  {len(results['lfp_events'])}")
    print(f"  STIM_ON:     {sum(1 for e in results['lfp_events'] if e['event_type'] == 'STIM_ON')}")
    print(f"  STIM_OFF:    {sum(1 for e in results['lfp_events'] if e['event_type'] == 'STIM_OFF')}")
    print(f"Mapped to EEG: {len(results['eeg_events'])}")
    
    # Baseline options
    print("\n" + "BASELINE OPTIONS (for noise covariance):")
    print("-"*80)
    
    # First stim-off segment
    baseline_off = get_baseline_segment(results, mode='first_stim_off')
    print(f"Option 1 - First stim-off segment (recommended for ~2s baseline):")
    print(f"  LFP time: {baseline_off['start_time']:.2f}s - {baseline_off['end_time']:.2f}s")
    print(f"  Duration: {baseline_off['duration']:.2f}s")
    
    # Before first stim
    baseline_pre = get_baseline_segment(results, mode='before_first_stim')
    print(f"\nOption 2 - Before first stimulation:")
    print(f"  LFP time: {baseline_pre['start_time']:.2f}s - {baseline_pre['end_time']:.2f}s")
    print(f"  Duration: {baseline_pre['duration']:.2f}s")
    
    # Analysis segment
    analysis = get_analysis_segment(results, segment_type='middle_stim_on')
    print("\n" + "ANALYSIS SEGMENT (for causality):")
    print("-"*80)
    print(f"  LFP:      {analysis['lfp_start']:.2f}s - {analysis['lfp_end']:.2f}s")
    print(f"  EEG:      {analysis['eeg_start']:.2f}s - {analysis['eeg_end']:.2f}s")
    print(f"  Duration: {analysis['duration']:.2f}s")
    print(f"  Type:     {analysis['description']}")
    
    print("\n" + "="*80)
    