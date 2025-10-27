"""
LFP Data I/O and Processing Module
===================================

Functions for loading, processing, and analyzing LFP (Local Field Potential) data
from Medtronic Percept PC device.
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import datetime
from pathlib import Path
from scipy import signal


# ============================================================
# Data Loading Functions
# ============================================================

def str2int(s):
    """
    Convert comma-separated string to numpy array of integers.
    
    Parameters
    ----------
    s : str
        Comma-separated string of integers
        
    Returns
    -------
    np.ndarray
        Array of integers
    """
    l = list(str(s).split(','))
    arr = []
    for val in l:
        try:
            arr.append(int(val))
        except ValueError:
            pass
    return np.asarray(arr)


def get_ms_arr(time_packets, packet_sizes, sr):
    """
    Generate millisecond timestamp array from packet information.
    
    Parameters
    ----------
    time_packets : array-like
        Timestamp for each packet
    packet_sizes : array-like
        Number of samples in each packet
    sr : float
        Sampling rate in Hz
        
    Returns
    -------
    np.ndarray
        Array of millisecond timestamps
    """
    step_size = 1000 / sr  # in ms
    
    ntimes = int(sum(packet_sizes))
    arr = np.empty(ntimes)
    
    stop_i = 0
    for i in range(len(packet_sizes)):
        start_i = stop_i
        stop_i = start_i + packet_sizes[i]
        
        endpoint = time_packets[i] + step_size * packet_sizes[i] - 0.1
        interval = np.arange(time_packets[i], endpoint, step_size).astype(int)
        arr[start_i:stop_i] = interval
    
    return arr


def get_dt_arr(arr_ms, dt_init, dt_offset_init):
    """
    Convert millisecond array to datetime array with timezone offset.
    
    Parameters
    ----------
    arr_ms : np.ndarray
        Array of millisecond timestamps
    dt_init : str
        Initial datetime string (ISO format)
    dt_offset_init : str
        Timezone offset string (e.g., '+01:00' or '-05:00')
        
    Returns
    -------
    list
        List of datetime objects
    """
    # Handle both positive and negative offsets
    if dt_offset_init.startswith('-'):
        dt_offset = datetime.datetime.strptime(dt_offset_init, '-%H:%M')
        offset_hours = -dt_offset.hour
    elif dt_offset_init.startswith('+'):
        dt_offset = datetime.datetime.strptime(dt_offset_init, '+%H:%M')
        offset_hours = dt_offset.hour
    else:
        raise ValueError("Offset format not recognized")
    
    dt_utc = datetime.datetime.strptime(dt_init, '%Y-%m-%dT%H:%M:%S.%fZ')
    dt = dt_utc - datetime.timedelta(hours=offset_hours)
    arr_dt = [dt + datetime.timedelta(milliseconds=i) for i in arr_ms]
    return arr_dt


def get_id(fp):
    """Extract subject ID from file path."""
    return os.path.basename(os.path.dirname(os.path.dirname(fp)))


def get_side(ch_init):
    """Extract laterality (left/right) from channel string."""
    return str.split(ch_init, sep='_')[-1].lower()


def get_ch(ch_init):
    """
    Convert channel string to numeric format.
    
    Parameters
    ----------
    ch_init : str
        Channel string (e.g., 'ZERO_ONE_LEFT')
        
    Returns
    -------
    str
        Numeric channel format (e.g., '+1-0')
    """
    num_dict = {
        'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3',
        'FOUR': '4', 'FIVE': '5', 'SIX': '6', 'SEVEN': '7',
        'EIGHT': '8', 'NINE': '9', 'TEN': '10', 'ELEVEN': '11',
        'TWELVE': '12'
    }
    
    ch_str = str.split(ch_init, sep='_')[:-1]
    ch_int = '-'.join(num_dict[i] for i in reversed(ch_str))
    return '+' + ch_int


def get_stim_contact(ch_sense_contacts):
    """
    Map sensing contacts to stimulation contacts.
    
    Parameters
    ----------
    ch_sense_contacts : str
        Sensing contact configuration
        
    Returns
    -------
    str
        Stimulation contact number
    """
    stim_contact_dict = {
        '+2-0': '1',
        '+3-1': '2',
        '+3-0': '1'
    }
    
    stim_contact = stim_contact_dict.get(ch_sense_contacts, 'unknown')
    return stim_contact


def get_ts_sense(data):
    """
    Extract time series sensing data from JSON.
    
    Parameters
    ----------
    data : dict
        Loaded JSON data
        
    Returns
    -------
    pd.DataFrame
        Time series voltage data with datetime index
    """
    nsides = len(data['BrainSenseTimeDomain'])
    
    for i in range(nsides):
        # Get laterality of sensor
        ch_init = data['BrainSenseTimeDomain'][i]['Channel']
        implant_side = get_side(ch_init)
        
        # Get array with timestamps in sequences of milliseconds
        time_packets = str2int(data['BrainSenseTimeDomain'][i]['TicksInMses'])
        packet_sizes = str2int(data['BrainSenseTimeDomain'][i]['GlobalPacketSizes'])
        sr = data['BrainSenseTimeDomain'][i]['SampleRateInHz']
        arr_ms = get_ms_arr(time_packets, packet_sizes, sr)
        
        # Convert timestamp array into datetime
        dt_init = data['BrainSenseTimeDomain'][i]['FirstPacketDateTime']
        dt_offset_init = data['ProgrammerUtcOffset']
        arr_dt = get_dt_arr(arr_ms, dt_init, dt_offset_init)
        
        # Get voltage timeseries data
        voltage = data['BrainSenseTimeDomain'][i]['TimeDomainData']
        column_name = 'voltage' + '_' + implant_side
        
        if i == 0:
            df = pd.DataFrame(data=voltage,
                            columns=[column_name],
                            index=arr_dt)
        if i > 0:
            if (column_name in df):
                df2 = pd.DataFrame(data=voltage,
                                 columns=[column_name],
                                 index=arr_dt)
                df = pd.concat([df, df2], axis=0)
            else:
                df[column_name] = voltage
    
    return df


def get_settings(data, fp):
    """
    Extract device settings and configuration.
    
    Parameters
    ----------
    data : dict
        Loaded JSON data
    fp : str or Path
        File path
        
    Returns
    -------
    pd.DataFrame
        Settings dataframe
    """
    subj_id = get_id(fp)
    df = pd.DataFrame(data=[subj_id], columns=['subj_id'])
    
    nsides = len(data['BrainSenseTimeDomain'])
    for i in range(nsides):
        ch_init = data['BrainSenseTimeDomain'][i]['Channel']
        implant_side = get_side(ch_init)
        df[implant_side + '_sr'] = data['BrainSenseTimeDomain'][i]['SampleRateInHz']
        
        ch_sense_contacts = get_ch(ch_init)
        df[implant_side + '_ch'] = ch_sense_contacts
        df[implant_side + '_stim_contact'] = get_stim_contact(ch_sense_contacts)
    
    return df


def get_ms_arr_stim(stim_data):
    """Extract millisecond timestamps for stimulation data."""
    ntimes = len(stim_data)
    arr_ms = [stim_data[i]['TicksInMs'] for i in range(ntimes)]
    return arr_ms


def get_ma_df(stim_data, arr_dt):
    """
    Extract stimulation amplitude data for both hemispheres.
    
    Parameters
    ----------
    stim_data : list
        Stimulation data from JSON
    arr_dt : list
        Datetime array
        
    Returns
    -------
    pd.DataFrame
        Stimulation amplitudes with datetime index
    """
    ntimes = len(arr_dt)
    
    ma_left = [stim_data[i]['Left']['mA'] for i in range(ntimes)]
    ma_right = [stim_data[i]['Right']['mA'] for i in range(ntimes)]
    
    data = np.column_stack((ma_left, ma_right))
    
    df = pd.DataFrame(data=data,
                     columns=['stim_amp_left', 'stim_amp_right'],
                     index=arr_dt)
    
    return df


def get_ts_stim(data):
    """
    Extract time series stimulation data from JSON.
    
    Parameters
    ----------
    data : dict
        Loaded JSON data
        
    Returns
    -------
    pd.DataFrame
        Stimulation amplitude time series
    """
    stim_data = data['BrainSenseLfp'][0]['LfpData']
    sr = data['BrainSenseLfp'][0]['SampleRateInHz']
    arr_ms = get_ms_arr_stim(stim_data)
    
    # Convert timestamp array into datetime
    dt_init = data['BrainSenseLfp'][0]['FirstPacketDateTime']
    dt_offset_init = data['ProgrammerUtcOffset']
    arr_dt = get_dt_arr(arr_ms, dt_init, dt_offset_init)
    
    # Get stimulation amplitude data
    df_stim = get_ma_df(stim_data, arr_dt)
    
    return df_stim


def load_lfp_data(jsonfile, load_sense=True, load_stim=True, load_settings=True):
    """
    Main function to load all LFP data from JSON file.
    
    Parameters
    ----------
    jsonfile : str or Path
        Path to JSON file
    load_sense : bool, optional
        Load sensing data (default: True)
    load_stim : bool, optional
        Load stimulation data (default: True)
    load_settings : bool, optional
        Load settings data (default: True)
        
    Returns
    -------
    tuple
        (df_ts_sense, df_ts_stim, df_settings)
    """
    df_ts_sense = None
    df_ts_stim = None
    df_settings = None
    
    with open(jsonfile) as f:
        data = json.load(f)
    
    if load_sense:
        df_ts_sense = get_ts_sense(data)
    
    if load_stim:
        df_ts_stim = get_ts_stim(data)
    
    if load_settings:
        df_settings = get_settings(data, jsonfile)
    
    return df_ts_sense, df_ts_stim, df_settings


# ============================================================
# Signal Processing Functions
# ============================================================

def get_sr_arr(df_settings):
    """
    Extract sampling rate array from settings.
    
    Parameters
    ----------
    df_settings : pd.DataFrame
        Settings dataframe
        
    Returns
    -------
    np.ndarray
        Sampling rates
    """
    fs = []
    if 'left_sr' in df_settings:
        fs.append(df_settings.left_sr.iloc[0])
        if 'right_sr' in df_settings:
            fs.append(df_settings.right_sr.iloc[0])
            if df_settings.right_sr.iloc[0] != df_settings.left_sr.iloc[0]:
                print('Warning: sampling rates differ across hemispheres')
    else:
        fs.append(df_settings.right_sr.iloc[0])
    return np.array(fs)


def compute_psd(df_ts_sense, df_settings):
    """
    Convert time series to power spectral density.
    
    Parameters
    ----------
    df_ts_sense : pd.DataFrame
        Time series sensing data
    df_settings : pd.DataFrame
        Settings dataframe
        
    Returns
    -------
    pd.DataFrame
        Power spectral density with frequency index
    """
    nchannel = df_ts_sense.shape[1]
    fs_arr = get_sr_arr(df_settings)
    
    for i in range(nchannel):
        [f, Pxx] = signal.welch(df_ts_sense.iloc[:, i].values, fs=fs_arr[i])
        Pxx_log = np.log10(Pxx)
        colname = df_ts_sense.columns[i].replace('voltage', 'psd')
        
        if i == 0:
            df_psd = pd.DataFrame(data=Pxx_log, columns=[colname],
                                index=np.round(f).astype(int))
        else:
            df_psd[colname] = Pxx_log
    return df_psd


def compute_spectrogram(df_ts_sense, df_settings):
    """
    Convert time series to spectrogram.
    
    Parameters
    ----------
    df_ts_sense : pd.DataFrame
        Time series sensing data
    df_settings : pd.DataFrame
        Settings dataframe
        
    Returns
    -------
    xr.DataArray
        Spectrogram data array
    """
    nchannel = df_ts_sense.shape[1]
    fs_arr = get_sr_arr(df_settings)
    sides = [df_ts_sense.columns[i].replace('voltage_', '') for i in range(nchannel)]
    
    for i in range(nchannel):
        [f, t, Sxx] = signal.spectrogram(df_ts_sense.iloc[:, i], fs=fs_arr[i])
        Sxx_log = np.log10(Sxx)
        t_dt = [df_ts_sense.index[0] + datetime.timedelta(seconds=s) for s in t]
        side = sides[i]
        if i == 0:
            da_spect = xr.DataArray(
                dims=['side', 'time', 'frequency'],
                coords=dict(
                    side=sides,
                    time=t_dt,
                    frequency=f
                )
            )
        da_spect.loc[dict(side=side)] = Sxx_log.T
    return da_spect


# ============================================================
# Stimulation Alignment Functions
# ============================================================

def align_stim_to_sense(df_ts_stim, df_ts_sense):
    """
    Align stimulation data to sensing data timeline.
    
    Parameters
    ----------
    df_ts_stim : pd.DataFrame
        Stimulation amplitudes with sparse timestamps
    df_ts_sense : pd.DataFrame
        Sensing voltage data with dense timestamps
        
    Returns
    -------
    pd.DataFrame
        Stimulation data aligned to sensing timeline
    """
    # Reindex stimulation data to match sensing data timeline
    # Use forward fill to propagate stimulation values between timestamps
    df_ts_stim_aligned = df_ts_stim.reindex(
        df_ts_sense.index, 
        method='ffill'
    )
    
    # Fill any leading NaN values with 0 (stim off)
    df_ts_stim_aligned = df_ts_stim_aligned.fillna(0)
    
    return df_ts_stim_aligned


def detect_stim_segments(df_ts_stim_aligned, side='left'):
    """
    Detect stimulation ON segments for a given hemisphere.
    
    Parameters
    ----------
    df_ts_stim_aligned : pd.DataFrame
        Aligned stimulation data
    side : str
        'left' or 'right'
        
    Returns
    -------
    list
        List of (start_idx, end_idx, stim_amplitude) tuples
    """
    col_name = f'stim_amp_{side}'
    stim_values = df_ts_stim_aligned[col_name].values
    
    # Find where stimulation changes (0 to non-zero or vice versa)
    stim_on = stim_values > 0
    
    # Find transitions
    transitions = np.diff(np.concatenate([[False], stim_on, [False]]).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    segments = []
    for start, end in zip(starts, ends):
        if end > start:  # Valid segment
            amp = stim_values[start]
            segments.append((start, end, amp))
    
    return segments


def extract_middle_segment(df_ts_sense, df_ts_stim_aligned, side='left',
                          start_buffer_sec=5.0, end_buffer_sec=2.0,
                          remove_artifacts=True, artifact_threshold_std=5.0):
    """
    Extract the middle stimulation segment (for analysis) with artifact removal.
    
    Parameters
    ----------
    df_ts_sense : pd.DataFrame
        Original sensing data
    df_ts_stim_aligned : pd.DataFrame
        Aligned stimulation data
    side : str
        'left' or 'right'
    start_buffer_sec : float, optional
        Seconds to skip at the beginning (to avoid onset artifacts), default: 5.0
    end_buffer_sec : float, optional
        Seconds to skip at the end (to avoid offset artifacts), default: 2.0
    remove_artifacts : bool, optional
        Whether to detect and remove initial artifacts, default: True
    artifact_threshold_std : float, optional
        Threshold in standard deviations for artifact detection, default: 5.0
        
    Returns
    -------
    tuple
        (df_trimmed, segment_info)
    """
    segments = detect_stim_segments(df_ts_stim_aligned, side)
    
    if len(segments) < 2:
        print(f"Warning: Less than 2 stimulation segments found for {side} side")
        return None, None
    
    # Get the middle segment (or second segment if there are multiple)
    middle_segment = segments[1] if len(segments) > 1 else segments[0]
    
    start_idx, end_idx, amp = middle_segment
    start_time = df_ts_stim_aligned.index[start_idx]
    end_time = df_ts_stim_aligned.index[end_idx - 1]
    
    # Apply start buffer (skip initial seconds)
    start_time_buffered = start_time + pd.Timedelta(seconds=start_buffer_sec)
    
    # Apply end buffer (skip final seconds)
    end_time_buffered = end_time - pd.Timedelta(seconds=end_buffer_sec)
    
    # Ensure buffered times are valid
    if start_time_buffered >= end_time_buffered:
        print(f"Warning: Buffers too large, using minimal buffer of 1 second")
        start_time_buffered = start_time + pd.Timedelta(seconds=1.0)
        end_time_buffered = end_time - pd.Timedelta(seconds=1.0)
    
    # Trim the sensing data
    df_trimmed = df_ts_sense.loc[start_time_buffered:end_time_buffered].copy()
    
    # Additional artifact removal if requested
    if remove_artifacts and len(df_trimmed) > 0:
        # Get voltage column for this side
        voltage_col = f'voltage_{side}'
        if voltage_col in df_trimmed.columns:
            voltage = df_trimmed[voltage_col].values
            
            # Calculate statistics on the data after initial buffer
            median_voltage = np.median(voltage)
            std_voltage = np.std(voltage)
            
            # Detect extreme outliers in the first 10% of data
            initial_portion = int(len(voltage) * 0.1)
            if initial_portion > 10:
                initial_data = voltage[:initial_portion]
                
                # Find samples that are extreme outliers
                outliers = np.abs(initial_data - median_voltage) > (artifact_threshold_std * std_voltage)
                
                if np.any(outliers):
                    # Find the last outlier position
                    last_outlier_idx = np.where(outliers)[0][-1]
                    
                    # Skip past the artifacts (add 1 second buffer after last outlier)
                    samples_to_skip = last_outlier_idx + int(250)  # Assuming ~250 Hz sampling
                    
                    if samples_to_skip < len(df_trimmed):
                        new_start_time = df_trimmed.index[samples_to_skip]
                        df_trimmed = df_trimmed.loc[new_start_time:].copy()
                        start_time_buffered = new_start_time
                        print(f"Removed {samples_to_skip} samples containing artifacts")
    
    segment_info = {
        'side': side,
        'start_time_original': start_time,
        'end_time_original': end_time,
        'start_time': start_time_buffered,
        'end_time': end_time_buffered,
        'start_buffer_sec': start_buffer_sec,
        'end_buffer_sec': end_buffer_sec,
        'duration_sec': (df_trimmed.index[-1] - df_trimmed.index[0]).total_seconds(),
        'n_samples': len(df_trimmed),
        'stim_amplitude': amp,
        'artifacts_removed': remove_artifacts
    }
    
    return df_trimmed, segment_info



