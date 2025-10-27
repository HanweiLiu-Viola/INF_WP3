"""
Signal Cleaning Module - UPDATED VERSION
=========================================

Fixed issues:
1. apply_standard_cleaning now respects l_freq and h_freq parameters
2. Validates h_freq against Nyquist frequency
3. Resampling happens BEFORE filtering (correct order)

Replace your existing signal_cleaning.py with this file.
"""

import mne
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalCleaner:
    """Base class for signal cleaning operations"""
    
    def __init__(self):
        self.processing_history = []
    
    def apply_bandpass_filter(self, raw, l_freq=1.0, h_freq=100.0, 
                             filter_type='fir', copy=True):
        """
        Apply bandpass filter to raw data.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data to filter
        l_freq : float
            Lower frequency bound (Hz)
        h_freq : float
            Upper frequency bound (Hz)
        filter_type : str
            Filter type ('fir' or 'iir')
        copy : bool
            If True, copy data before filtering
            
        Returns
        -------
        raw : mne.io.Raw
            Filtered data
        """
        if copy:
            raw = raw.copy()
        
        # Apply bandpass filter
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method=filter_type,
            picks='all',
            verbose=False
        )
        
        self.processing_history.append(f'bandpass_{l_freq}-{h_freq}Hz')
        logger.info(f"✓ Applied bandpass filter: {l_freq}-{h_freq} Hz ({filter_type})")
        
        return raw
    
    def apply_notch_filter(self, raw, freqs, copy=True):
        """
        Apply notch filter to remove line noise.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data
        freqs : list or array
            Frequencies to notch filter (Hz)
        copy : bool
            If True, copy data before filtering
            
        Returns
        -------
        raw : mne.io.Raw
            Notch filtered data
        """
        if copy:
            raw = raw.copy()
        
        raw.notch_filter(
            freqs=freqs,
            picks='all',
            verbose=False
        )
        
        self.processing_history.append(f'notch_{freqs}Hz')
        logger.info(f"✓ Applied notch filter at: {freqs} Hz")
        
        return raw
    
    def apply_standard_cleaning(self, raw, signal_type='eeg', 
                               target_sfreq=None, line_freq=50.0,
                               l_freq=None, h_freq=None, copy=True):
        """
        Apply standard cleaning pipeline.
        
        FIXED: Now respects l_freq and h_freq parameters
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw data
        signal_type : str
            'eeg' or 'lfp'
        target_sfreq : float, optional
            Target sampling frequency for resampling
        line_freq : float
            Power line frequency (50 or 60 Hz)
        l_freq : float, optional
            Lower frequency for bandpass filter
            If None, uses default based on signal_type
        h_freq : float, optional
            Upper frequency for bandpass filter
            If None, uses default based on signal_type
        copy : bool
            If True, copy data before processing
            
        Returns
        -------
        raw : mne.io.Raw
            Cleaned data
        """
        if copy:
            raw = raw.copy()
        
        sfreq = raw.info['sfreq']
        nyquist_freq = sfreq / 2.0
        
        logger.info("="*60)
        logger.info(f"STANDARD CLEANING PIPELINE ({signal_type.upper()})")
        logger.info("="*60)
        logger.info(f"Original sampling rate: {sfreq} Hz")
        logger.info(f"Nyquist frequency: {nyquist_freq} Hz")
        
        # 1. Resample FIRST if needed (before filtering!)
        if target_sfreq is not None and target_sfreq != sfreq:
            logger.info(f"\n[1] Resampling: {sfreq} Hz → {target_sfreq} Hz")
            raw.resample(target_sfreq, npad='auto', verbose=False)
            sfreq = raw.info['sfreq']
            nyquist_freq = sfreq / 2.0
            self.processing_history.append(f'resample_{target_sfreq}Hz')
            logger.info(f"✓ Resampled to {sfreq} Hz (new Nyquist: {nyquist_freq} Hz)")
        else:
            logger.info(f"\n[1] No resampling needed")
        
        # 2. Set default frequency ranges based on signal type
        if l_freq is None:
            l_freq = 1.0  # Default lower bound for both EEG and LFP
        
        if h_freq is None:
            if signal_type.lower() == 'eeg':
                h_freq = 100.0  # Default for EEG
            elif signal_type.lower() == 'lfp':
                h_freq = 100.0  # Default for LFP (safe for 250 Hz sampling)
            else:
                h_freq = 100.0  # Safe default
        
        # 3. Validate h_freq against Nyquist frequency
        if h_freq >= nyquist_freq:
            original_h_freq = h_freq
            h_freq = nyquist_freq - 5  # Leave 5 Hz safety margin
            logger.warning(
                f"h_freq ({original_h_freq} Hz) exceeds Nyquist frequency ({nyquist_freq} Hz). "
                f"Adjusted to {h_freq} Hz"
            )
        
        # 4. Apply bandpass filter with validated parameters
        logger.info(f"\n[2] Applying bandpass filter: {l_freq}-{h_freq} Hz")
        raw = self.apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq, copy=False)
        
        # 5. Notch filter for power line noise
        notch_freqs = []
        for harmonic in [1, 2, 3]:
            freq = line_freq * harmonic
            if freq < nyquist_freq - 5:
                notch_freqs.append(freq)
        
        if notch_freqs:
            logger.info(f"\n[3] Applying notch filter at: {notch_freqs} Hz")
            raw = self.apply_notch_filter(raw, freqs=notch_freqs, copy=False)
        
        logger.info("="*60)
        logger.info(f"{signal_type.upper()} CLEANING COMPLETED")
        logger.info("="*60)
        
        return raw
    
    def get_processing_summary(self):
        """Get summary of processing steps applied"""
        summary = "Processing History:\n"
        for i, step in enumerate(self.processing_history, 1):
            summary += f"  {i}. {step}\n"
        return summary


class EEGCleaner(SignalCleaner):
    """EEG-specific cleaning operations"""
    
    def apply_eeg_cleaning(self, raw, target_sfreq=250.0, line_freq=50.0,
                          l_freq=1.0, h_freq=100.0, copy=True):
        """
        Apply standard EEG cleaning pipeline.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        target_sfreq : float
            Target sampling frequency (Hz)
        line_freq : float
            Power line frequency (50 or 60 Hz)
        l_freq : float
            Lower frequency for bandpass filter (Hz)
        h_freq : float
            Upper frequency for bandpass filter (Hz)
        copy : bool
            If True, copy data before processing
            
        Returns
        -------
        raw : mne.io.Raw
            Cleaned EEG data
        """
        logger.info("="*60)
        logger.info("STARTING EEG CLEANING PIPELINE")
        logger.info("="*60)
        
        # Apply standard cleaning with EEG-specific parameters
        raw = self.apply_standard_cleaning(
            raw,
            signal_type='eeg',
            target_sfreq=target_sfreq,
            line_freq=line_freq,
            l_freq=l_freq,
            h_freq=h_freq,
            copy=copy
        )
        
        return raw


class LFPCleaner(SignalCleaner):
    """LFP-specific cleaning operations"""
    
    def apply_lfp_cleaning(self, raw, target_sfreq=None, line_freq=50.0,
                          l_freq=1.0, h_freq=100.0, copy=True):
        """
        Apply standard LFP cleaning pipeline.
        
        FIXED: Now properly respects l_freq and h_freq parameters
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw LFP data
        target_sfreq : float, optional
            Target sampling frequency (Hz)
            If None, keeps original sampling rate
        line_freq : float
            Power line frequency (50 or 60 Hz)
        l_freq : float
            Lower frequency for bandpass filter (Hz)
            Must be > 0
        h_freq : float
            Upper frequency for bandpass filter (Hz)
            Must be < Nyquist frequency (sampling_rate / 2)
        copy : bool
            If True, copy data before processing
            
        Returns
        -------
        raw : mne.io.Raw
            Cleaned LFP data
            
        Notes
        -----
        For LFP at 250 Hz sampling:
        - Nyquist frequency = 125 Hz
        - Recommended: l_freq=1.0, h_freq=100.0 (safe)
        - DO NOT use h_freq > 120 Hz
        """
        logger.info("="*60)
        logger.info("STARTING LFP CLEANING PIPELINE")
        logger.info("="*60)
        
        # Apply standard cleaning with LFP-specific parameters
        raw = self.apply_standard_cleaning(
            raw,
            signal_type='lfp',
            target_sfreq=target_sfreq,
            line_freq=line_freq,
            l_freq=l_freq,
            h_freq=h_freq,
            copy=copy
        )
        
        return raw


