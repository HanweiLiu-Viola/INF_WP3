"""
Visualization Module
====================

Plotting functions for LFP and EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Scientific Color Palette
# ============================================================

# Nature/Science style colors
COLOR_RAW = '#2C3E50'        # Dark blue-gray for raw data
COLOR_SELECTED = '#E74C3C'   # Red for selected/cropped segment
COLOR_LEFT = '#3498DB'       # Blue for left hemisphere
COLOR_RIGHT = '#E67E22'      # Orange for right hemisphere
COLOR_STIM = '#27AE60'       # Green for stimulation
COLOR_THRESHOLD = '#9B59B6'  # Purple for thresholds


# ============================================================
# LFP Visualization
# ============================================================

def plot_lfp_timeseries(df_ts_sense, df_ts_cropped, df_settings, save_path=None):
    """
    Plot LFP voltage time series with raw and cropped segments.
    
    Displays two subplots (left and right hemisphere) showing the complete
    raw LFP data with the stimulation-ON cropped segment highlighted in
    a different color.
    
    Parameters
    ----------
    df_ts_sense : pd.DataFrame
        Complete raw LFP sensing data with datetime index
    df_ts_cropped : pd.DataFrame
        Cropped LFP segment (stimulation ON period)
    df_settings : pd.DataFrame
        Settings dataframe containing channel information
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    Examples
    --------
    >>> fig = plot_lfp_timeseries(df_ts_sense, df_left_middle, df_settings,
    ...                           save_path='lfp_timeseries.png')
    """
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Convert datetime index to relative time (seconds from start)
    start_time = df_ts_sense.index[0]
    raw_times = [(t - start_time).total_seconds() for t in df_ts_sense.index]
    
    if df_ts_cropped is not None:
        crop_start_time = df_ts_cropped.index[0]
        crop_times = [(t - start_time).total_seconds() for t in df_ts_cropped.index]
    
    # Get channel information
    left_ch = df_settings.left_ch.iloc[0] if 'left_ch' in df_settings.columns else 'N/A'
    right_ch = df_settings.right_ch.iloc[0] if 'right_ch' in df_settings.columns else 'N/A'
    subject_id = df_settings.subj_id.iloc[0] if 'subj_id' in df_settings.columns else ''
    
    # === Plot Left Hemisphere ===
    ax_left = axes[0]
    
    if 'voltage_left' in df_ts_sense.columns:
        # Plot raw data
        ax_left.plot(raw_times, df_ts_sense['voltage_left'].values,
                    color=COLOR_RAW, linewidth=0.5, alpha=0.6,
                    label='Raw LFP', zorder=1)
        
        # Plot cropped segment on top
        if df_ts_cropped is not None and 'voltage_left' in df_ts_cropped.columns:
            ax_left.plot(crop_times, df_ts_cropped['voltage_left'].values,
                        color=COLOR_SELECTED, linewidth=0.8, alpha=0.9,
                        label='Stim ON segment', zorder=2)
            
            # Add vertical lines to mark crop boundaries
            ax_left.axvline(crop_times[0], color=COLOR_STIM, linestyle='--',
                          linewidth=1.5, alpha=0.7, label='Crop boundary')
            ax_left.axvline(crop_times[-1], color=COLOR_STIM, linestyle='--',
                          linewidth=1.5, alpha=0.7)
    
    ax_left.set_ylabel('LFP Voltage (µV)', fontsize=11, fontweight='bold')
    ax_left.set_title(f'Left Hemisphere - Channel {left_ch}', 
                     fontsize=12, fontweight='bold', pad=10)
    ax_left.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_left.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    
    # === Plot Right Hemisphere ===
    ax_right = axes[1]
    
    if 'voltage_right' in df_ts_sense.columns:
        # Plot raw data
        ax_right.plot(raw_times, df_ts_sense['voltage_right'].values,
                     color=COLOR_RAW, linewidth=0.5, alpha=0.6,
                     label='Raw LFP', zorder=1)
        
        # Plot cropped segment on top
        if df_ts_cropped is not None and 'voltage_right' in df_ts_cropped.columns:
            ax_right.plot(crop_times, df_ts_cropped['voltage_right'].values,
                         color=COLOR_SELECTED, linewidth=0.8, alpha=0.9,
                         label='Stim ON segment', zorder=2)
            
            # Add vertical lines to mark crop boundaries
            ax_right.axvline(crop_times[0], color=COLOR_STIM, linestyle='--',
                           linewidth=1.5, alpha=0.7, label='Crop boundary')
            ax_right.axvline(crop_times[-1], color=COLOR_STIM, linestyle='--',
                           linewidth=1.5, alpha=0.7)
    elif 'voltage_left' in df_ts_sense.columns:
        # If no right data, show message
        ax_right.text(0.5, 0.5, 'Right hemisphere data not available',
                     ha='center', va='center', transform=ax_right.transAxes,
                     fontsize=12, style='italic', color='gray')
        ax_right.set_ylim([0, 1])
    
    ax_right.set_ylabel('LFP Voltage (µV)', fontsize=11, fontweight='bold')
    ax_right.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_right.set_title(f'Right Hemisphere - Channel {right_ch}', 
                      fontsize=12, fontweight='bold', pad=10)
    ax_right.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_right.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    # Add main title
    if df_ts_cropped is not None:
        raw_duration = raw_times[-1]
        crop_duration = crop_times[-1] - crop_times[0]
        fig.suptitle(f'LFP Time Series - {subject_id}\n'
                    f'Raw: {raw_duration:.1f}s | Cropped: {crop_duration:.1f}s',
                    fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'LFP Time Series - {subject_id}',
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_eeg_comparison(raw_eeg, crop_start_idx, crop_end_idx, 
                       channel_idx=0, sync_info=None, save_path=None):
    """
    Plot original EEG with cropped segment highlighted by index.
    
    Instead of overlaying two signals, this function plots the original EEG
    and highlights the cropped segment using a different color based on
    sample indices.
    
    Parameters
    ----------
    raw_eeg : mne.io.Raw
        Original raw EEG data
    crop_start_idx : int
        Starting sample index for cropping in the original EEG
    crop_end_idx : int
        Ending sample index for cropping in the original EEG
    channel_idx : int, optional
        Index of EEG channel to display (default: 0)
    sync_info : dict, optional
        Synchronization information for additional details
    save_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
        
    Examples
    --------
    >>> # In notebook, calculate indices
    >>> crop_start_idx = 4250  # Sample index where cropping starts
    >>> crop_end_idx = 52833   # Sample index where cropping ends
    >>> 
    >>> fig = plot_eeg_comparison(
    ...     raw_eeg=raw,
    ...     crop_start_idx=crop_start_idx,
    ...     crop_end_idx=crop_end_idx,
    ...     channel_idx=0,
    ...     save_path='eeg_comparison.png'
    ... )
    """
    import numpy as np
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Get data from selected channel
    picks = [channel_idx]
    channel_name = raw_eeg.ch_names[channel_idx]
    
    # Get raw EEG data
    raw_data = raw_eeg.get_data(picks=picks)[0]
    raw_times = raw_eeg.times
    sfreq = raw_eeg.info['sfreq']
    
    # Validate indices
    n_samples = len(raw_data)
    if crop_start_idx < 0 or crop_end_idx > n_samples:
        raise ValueError(f"Invalid crop indices: start={crop_start_idx}, "
                        f"end={crop_end_idx}, total samples={n_samples}")
    
    # Split data into three parts: before, cropped, after
    data_before = raw_data[:crop_start_idx]
    data_cropped = raw_data[crop_start_idx:crop_end_idx]
    data_after = raw_data[crop_end_idx:]
    
    times_before = raw_times[:crop_start_idx]
    times_cropped = raw_times[crop_start_idx:crop_end_idx]
    times_after = raw_times[crop_end_idx:]
    
    # === Plot original EEG in three parts with different colors ===
    
    # Before cropped region (gray, semi-transparent)
    if len(data_before) > 0:
        ax.plot(times_before, data_before, color=COLOR_RAW, 
               linewidth=0.5, alpha=0.4, label='Original EEG (excluded)', 
               zorder=1)
    
    # Cropped region (red, highlighted)
    ax.plot(times_cropped, data_cropped, color=COLOR_SELECTED, 
           linewidth=0.8, alpha=0.9, label='Cropped segment', zorder=2)
    
    # After cropped region (gray, semi-transparent)
    if len(data_after) > 0:
        ax.plot(times_after, data_after, color=COLOR_RAW, 
               linewidth=0.5, alpha=0.4, zorder=1)
    
    # === Mark crop boundaries ===
    crop_start_time = raw_times[crop_start_idx]
    crop_end_time = raw_times[crop_end_idx - 1] if crop_end_idx > 0 else raw_times[-1]
    
    ax.axvline(crop_start_time, color=COLOR_STIM, linestyle='--',
               linewidth=2, alpha=0.7, label='Crop start', zorder=3)
    ax.axvline(crop_end_time, color=COLOR_THRESHOLD, linestyle='--',
               linewidth=2, alpha=0.7, label='Crop end', zorder=3)
    
    # === Highlight cropped region with background ===
    ax.axvspan(crop_start_time, crop_end_time, alpha=0.1, 
              color=COLOR_SELECTED, zorder=0)
    
    # === Labels and title ===
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    
    # Calculate durations
    raw_duration = raw_times[-1]
    crop_duration = crop_end_time - crop_start_time
    n_crop_samples = crop_end_idx - crop_start_idx
    
    if sync_info is not None:
        lfp_duration = sync_info.get('lfp_duration', 0)
        time_diff = sync_info.get('duration_diff', abs(crop_duration - lfp_duration))
        
        ax.set_title(f'EEG Comparison - Channel: {channel_name}\n'
                    f'Original: {raw_duration:.1f}s ({n_samples:,} samples) → '
                    f'Cropped: {crop_duration:.2f}s ({n_crop_samples:,} samples) | '
                    f'Time diff with LFP: {time_diff:.2f}s',
                    fontsize=13, fontweight='bold', pad=15)
    else:
        ax.set_title(f'EEG Comparison - Channel: {channel_name}\n'
                    f'Original: {raw_duration:.1f}s ({n_samples:,} samples) → '
                    f'Cropped: {crop_duration:.2f}s ({n_crop_samples:,} samples)',
                    fontsize=13, fontweight='bold', pad=15)
    
    # === Legend positioned on the right side, outside the plot ===
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), 
             fontsize=10, framealpha=0.95, edgecolor='gray', 
             fancybox=True, shadow=True)
    
    # === Grid and spines ===
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # === Add info box with statistics ===
    info_text = (f'Sampling rate: {sfreq:.0f} Hz\n'
                f'Total samples: {n_samples:,}\n'
                f'Crop indices: {crop_start_idx:,} - {crop_end_idx:,}\n'
                f'Cropped samples: {n_crop_samples:,}\n'
                f'Crop time: {crop_start_time:.2f}s - {crop_end_time:.2f}s\n'
                f'Crop duration: {crop_duration:.2f}s')
    
    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
    #        fontsize=9, va='top', ha='left',
    #        bbox=dict(boxstyle='round', facecolor='white', 
    #                 alpha=0.9, edgecolor='gray'),
    #        family='monospace')

    ax.text(1.01, 0.05, info_text, transform=ax.transAxes,
           fontsize=9, va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', 
                    alpha=0.9, edgecolor='gray'),
           family='monospace')    
    
    # Adjust layout to make room for legend on the right
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Leave space for legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_stimulation_amplitudes(df_ts_stim, df_settings, save_path=None):
    """
    Plot stimulation amplitude time series.
    
    Parameters
    ----------
    df_ts_stim : pd.DataFrame
        Stimulation data
    df_settings : pd.DataFrame
        Settings dataframe
    save_path : str or Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    left_contact = df_settings.left_stim_contact.iloc[0] if 'left_stim_contact' in df_settings.columns else ""
    right_contact = df_settings.right_stim_contact.iloc[0] if 'right_stim_contact' in df_settings.columns else ""
    
    ax.plot(df_ts_stim.stim_amp_left, color=COLOR_LEFT, linewidth=1.5,
            label=f'Left contact {left_contact}')
    ax.plot(df_ts_stim.stim_amp_right, color=COLOR_RIGHT, linewidth=1.5,
            label=f'Right contact {right_contact}')
    
    ax.set_xlabel('Time (HH:MM:SS)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude (mA)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    subj_id = df_settings.subj_id.iloc[0] if 'subj_id' in df_settings.columns else ''
    ax.set_title(f'DBS Stimulation Amplitudes - {subj_id}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_lfp_spectrogram(da_spect, df_settings, save_path=None):
    """
    Plot LFP spectrogram for both hemispheres.
    
    Parameters
    ----------
    da_spect : xr.DataArray
        Spectrogram data
    df_settings : pd.DataFrame
        Settings dataframe
    save_path : str or Path, optional
        Path to save figure
    """
    import pandas as pd
    
    nsides = da_spect.shape[0]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    
    plt.setp(axs[0], ylabel='Frequency (Hz)')
    plt.setp(axs, xlabel='Time (HH:MM:SS)')
    
    subj_id = df_settings.subj_id.iloc[0] if 'subj_id' in df_settings.columns else ''
    fig.suptitle(f'LFP Spectrogram - {subj_id}', fontsize=14, fontweight='bold')
    
    if nsides == 1:
        axs[1].axis('off')
    
    for i in range(nsides):
        side = da_spect.side.values[i]
        ax = axs[i]
        
        ch = df_settings.left_ch.iloc[0] if i == 0 and 'left_ch' in df_settings.columns else ''
        if i == 1 and 'right_ch' in df_settings.columns:
            ch = df_settings.right_ch.iloc[0]
        
        df = pd.DataFrame(da_spect.loc[dict(side=side)].values,
                         columns=da_spect.frequency.values,
                         index=da_spect.time.values)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        c = ax.pcolormesh(df.index, df.columns, df.values.T, vmin=-6, vmax=1,
                         cmap='viridis')
        ax.set_title(f'{side.capitalize()} - Ch: {ch}', fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Colorbar
    axins = inset_axes(axs[-1],
                      width="5%",
                      height="100%",
                      loc='center right',
                      borderpad=-5)
    cbar = fig.colorbar(c, cax=axins, orientation="vertical")
    cbar.ax.set_ylabel('log(Power)', rotation=270, labelpad=15, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_lfp_psd(df_psd, df_settings, save_path=None):
    """
    Plot LFP power spectral density.
    
    Parameters
    ----------
    df_psd : pd.DataFrame
        PSD dataframe
    df_settings : pd.DataFrame
        Settings dataframe
    save_path : str or Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if 'psd_left' in df_psd:
        left_ch = df_settings.left_ch.iloc[0] if 'left_ch' in df_settings.columns else ''
        ax.plot(df_psd.psd_left, color=COLOR_LEFT, linewidth=2,
                label=f'Left Ch{left_ch}')
    
    if 'psd_right' in df_psd:
        right_ch = df_settings.right_ch.iloc[0] if 'right_ch' in df_settings.columns else ''
        ax.plot(df_psd.psd_right, color=COLOR_RIGHT, linewidth=2,
                label=f'Right Ch{right_ch}')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('log(Power)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    subj_id = df_settings.subj_id.iloc[0] if 'subj_id' in df_settings.columns else ''
    ax.set_title(f'LFP Power Spectral Density - {subj_id}',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


# ============================================================
# EEG-LFP Synchronization Visualization
# ============================================================

def plot_synchronization_result(sync_info, df_lfp=None, save_path=None):
    """
    Visualize EEG-LFP synchronization result.
    
    Parameters
    ----------
    sync_info : dict
        Synchronization information
    df_lfp : pd.DataFrame, optional
        LFP data for comparison
    save_path : str or Path, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    times = sync_info['times']
    stim_power = sync_info['stim_power']
    threshold = sync_info.get('threshold', 0)
    
    # Plot 1: EEG stim power
    axes[0].plot(times, stim_power, color=COLOR_RAW, alpha=0.7, linewidth=0.8)
    axes[0].axhline(threshold, color=COLOR_THRESHOLD, linestyle='--',
                   linewidth=1.5, alpha=0.7, label='Threshold')
    
    start = sync_info.get('eeg_start_time', sync_info.get('eeg_start', 0))
    end = sync_info.get('eeg_end_time', sync_info.get('eeg_end', times[-1]))
    
    axes[0].axvspan(start, end, alpha=0.2, color=COLOR_STIM,
                   label='Selected segment')
    
    axes[0].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Stimulation Power', fontsize=11, fontweight='bold')
    axes[0].set_title('EEG Stimulation Detection', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot 2: Zoomed comparison
    if df_lfp is not None:
        ax2a = axes[1]
        margin = 2
        mask_start = max(0, start - margin)
        mask_end = min(times[-1], end + margin)
        mask = (times >= mask_start) & (times <= mask_end)
        
        ax2a.plot(times[mask], stim_power[mask], color=COLOR_LEFT,
                 linewidth=1.2, label='EEG power')
        ax2a.axvline(start, color=COLOR_STIM, linestyle='--', linewidth=2)
        ax2a.axvline(end, color=COLOR_THRESHOLD, linestyle='--', linewidth=2)
        ax2a.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax2a.set_ylabel('EEG Power', color=COLOR_LEFT, fontsize=11, fontweight='bold')
        ax2a.tick_params(axis='y', labelcolor=COLOR_LEFT)
        ax2a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2a.spines['top'].set_visible(False)
        
        # LFP overlay
        ax2b = ax2a.twinx()
        lfp_times = (df_lfp.index - df_lfp.index[0]).total_seconds()
        lfp_col = 'voltage_left' if 'voltage_left' in df_lfp.columns else df_lfp.columns[0]
        ax2b.plot(lfp_times, df_lfp[lfp_col], color=COLOR_SELECTED,
                 alpha=0.5, linewidth=0.6, label='LFP')
        ax2b.set_ylabel('LFP Voltage (µV)', color=COLOR_SELECTED,
                       fontsize=11, fontweight='bold')
        ax2b.tick_params(axis='y', labelcolor=COLOR_SELECTED)
        ax2b.spines['top'].set_visible(False)
        
        eeg_dur = sync_info.get('eeg_duration', end - start)
        lfp_dur = sync_info.get('lfp_duration', lfp_times.values[-1])
        axes[1].set_title(f'Synchronization Comparison: EEG {eeg_dur:.2f}s vs LFP {lfp_dur:.2f}s',
                         fontsize=12, fontweight='bold')
        
        lines1, labels1 = ax2a.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2a.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    else:
        axes[1].plot(times, stim_power, color=COLOR_RAW, linewidth=0.8)
        axes[1].axvspan(start, end, alpha=0.2, color=COLOR_STIM)
        axes[1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Stimulation Power', fontsize=11, fontweight='bold')
        axes[1].set_title('Selected Segment', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_threshold_search(times, stim_power, power_norm, 
                          best_threshold, best_segment, results,
                          target_duration, save_path=None):
    """
    Visualize threshold search process.
    
    Parameters
    ----------
    times : np.ndarray
        Time array
    stim_power : np.ndarray
        Stimulation power
    power_norm : np.ndarray
        Normalized power
    best_threshold : float
        Best threshold found
    best_segment : tuple
        Best segment (start, end, duration)
    results : list
        All threshold results
    target_duration : float
        Target duration
    save_path : str or Path, optional
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Threshold vs Duration
    if len(results) > 0:
        results_array = np.array(results)
        thresholds = results_array[:, 0]
        durations = results_array[:, 3]
        diffs = results_array[:, 4]
        
        scatter = axes[0].scatter(thresholds, durations, c=diffs, 
                                 cmap='viridis_r', s=50, alpha=0.6)
        axes[0].axhline(target_duration, color=COLOR_THRESHOLD, linestyle='--', 
                       linewidth=2, label=f'Target: {target_duration:.1f}s')
        
        if best_threshold is not None:
            axes[0].scatter([best_threshold], [best_segment[2]], 
                          color=COLOR_SELECTED, s=200, marker='*', 
                          edgecolors='black', linewidth=2,
                          label=f'Best: {best_threshold:.3f}')
        
        axes[0].set_xlabel('Threshold (normalized)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Segment Duration (s)', fontsize=11, fontweight='bold')
        axes[0].set_title('Threshold vs Detected Segment Duration',
                         fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        plt.colorbar(scatter, ax=axes[0], label='Duration Difference (s)')
    
    # Plot 2: Power with threshold
    axes[1].plot(times, stim_power, color=COLOR_RAW, alpha=0.7, linewidth=0.8)
    
    if best_threshold is not None:
        threshold_actual = best_threshold * (np.max(stim_power) - np.min(stim_power)) + np.min(stim_power)
        axes[1].axhline(threshold_actual, color=COLOR_THRESHOLD, linestyle='--', 
                       linewidth=2, label=f'Threshold')
        
        stim_on = power_norm > best_threshold
        axes[1].fill_between(times, np.min(stim_power), np.max(stim_power),
                            where=stim_on, alpha=0.2, color=COLOR_STIM)
        
        axes[1].axvline(best_segment[0], color=COLOR_STIM, linestyle='--', linewidth=2)
        axes[1].axvline(best_segment[1], color=COLOR_SELECTED, linestyle='--', linewidth=2)
    
    axes[1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Stimulation Power', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Optimal Threshold (Duration: {best_segment[2]:.1f}s)',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Plot 3: Zoomed view
    if best_threshold is not None:
        margin = 5
        start = max(0, best_segment[0] - margin)
        end = min(times[-1], best_segment[1] + margin)
        mask = (times >= start) & (times <= end)
        
        axes[2].plot(times[mask], stim_power[mask], color=COLOR_RAW, linewidth=1.5)
        axes[2].axhline(threshold_actual, color=COLOR_THRESHOLD, linestyle='--',
                       linewidth=2, alpha=0.5)
        axes[2].axvline(best_segment[0], color=COLOR_STIM, linestyle='--',
                       linewidth=3, label='Start')
        axes[2].axvline(best_segment[1], color=COLOR_SELECTED, linestyle='--',
                       linewidth=3, label='End')
        axes[2].axvspan(best_segment[0], best_segment[1], alpha=0.2, color=COLOR_STIM)
        
        axes[2].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Stimulation Power', fontsize=11, fontweight='bold')
        axes[2].set_title(f'Zoomed View: {best_segment[0]:.1f}s - {best_segment[1]:.1f}s',
                         fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig

