"""
EEG Preprocessing Module - UPDATED WITH ICLabel & Autoreject
============================================================

Enhanced preprocessing pipeline with:
1. MNE-ICLabel for automatic ICA component classification (no EOG/ECG needed)
2. Autoreject for automatic epoch-level artifact rejection
3. Fixed-length epoch creation for continuous data

All previous fixes maintained:
- pyprep only processes true EEG channels (excludes misc, stim, REF CZ)
- 3-step re-referencing: REF CZ -> drop -> average
- Proper channel type handling
"""

import mne
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """EEG-specific preprocessing operations with advanced artifact removal"""
    
    def __init__(self):
        self.processing_history = []
        self.bad_channels = []
        self.ica_component_labels = None
    
    def detect_bad_channels_pyprep(self, raw, ransac=False):
        """
        Detect bad channels using pyprep (only on EEG channels).
        
        CRITICAL FIX: 
        - Excludes REF CZ and other non-standard EEG channels
        - RANSAC disabled due to channel position mismatch issues
        - Uses correlation, deviation, and HF noise detection (sufficient for most cases)
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        ransac : bool
            True: try RANSAC method, if it fails, a warning will be issued and the process will automatically skip. (RANSAC has channel position issues)
            Using other methods: correlation + deviation + HF noise only.
            
        Returns
        -------
        bad_channels : list
            List of bad channel names
        """
        try:
            from pyprep.find_noisy_channels import NoisyChannels
        except ImportError:
            raise ImportError("Install pyprep: pip install pyprep")
        
        logger.info("="*60)
        logger.info("DETECTING BAD CHANNELS WITH PYPREP")
        logger.info("="*60)
        
        # Get EEG channel indices (exclude misc, stim, ref)
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
            exclude=[]
        )
        
        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found!")
            return []
        
        # Get channel names
        eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]
        
        # ADDITIONAL FIX: Remove reference channels by name pattern
        ref_patterns = ['REF', 'Ref', 'ref', 'VREF', 'VRef']
        eeg_channel_names_filtered = [
            ch for ch in eeg_channel_names 
            if not any(pattern in ch for pattern in ref_patterns)
        ]
        
        logger.info(f"Total channels: {len(raw.ch_names)}")
        logger.info(f"EEG channels (by type): {len(eeg_channel_names)}")
        logger.info(f"EEG channels (filtered): {len(eeg_channel_names_filtered)}")
        
        if len(eeg_channel_names_filtered) == 0:
            logger.warning("No valid EEG channels after filtering!")
            return []
        
        # Create subset with only filtered EEG channels
        raw_eeg = raw.copy().pick_channels(eeg_channel_names_filtered)
        
        logger.info(f"Channels for pyprep: {len(raw_eeg.ch_names)}")
        

        if ransac:
            logger.info(f"Detection methods: correlation, deviation, HF noise, RANSAC")
        else:
            logger.info(f"Detection methods: correlation, deviation, HF noise")
        
        # Run pyprep detection
        nc = NoisyChannels(raw_eeg, do_detrend=False)
        
        logger.info("[1] Correlation detection...")
        nc.find_bad_by_correlation()
        
        logger.info("[2] Deviation detection...")
        nc.find_bad_by_deviation()
        
        logger.info("[3] HF noise detection...")
        nc.find_bad_by_hfnoise()
        
                
        if ransac:
            logger.info("[4] RANSAC detection...")
            try:
                nc.find_bad_by_ransac()
            except Exception as e:
                logger.warning(f"RANSAC failed: {e}")               
                logger.warning("Using correlation + deviation + HF noise instead")
                ransac = False
        else:
            # Skip RANSAC to avoid index errors
            logger.info("[4] RANSAC: skipped (not needed for filtered channels)")
        
        bad_channels = nc.get_bads()
        bad_channels = [str(ch) for ch in bad_channels]
        
        logger.info("\n" + "="*60)
        logger.info("DETECTION RESULTS")
        logger.info("="*60)
        
        if bad_channels:
            logger.info(f"Found {len(bad_channels)} bad channels:")
            logger.info(f"  {bad_channels}")
            
            # Handle different pyprep versions with different attribute names
            if hasattr(nc, 'bad_by_correlation') and nc.bad_by_correlation:
                logger.info(f"  Correlation: {nc.bad_by_correlation}")
            
            if hasattr(nc, 'bad_by_deviation') and nc.bad_by_deviation:
                logger.info(f"  Deviation: {nc.bad_by_deviation}")
            
            # Try both attribute names for HF noise (different pyprep versions)
            if hasattr(nc, 'bad_by_hfnoise') and nc.bad_by_hfnoise:
                logger.info(f"  HF noise: {nc.bad_by_hfnoise}")
            elif hasattr(nc, 'bad_by_hf_noise') and nc.bad_by_hf_noise:
                logger.info(f"  HF noise: {nc.bad_by_hf_noise}")

            if ransac and hasattr(nc, 'bad_by_ransac') and nc.bad_by_ransac:
                logger.info(f"  RANSAC: {nc.bad_by_ransac}")
        else:
            logger.info("No bad channels detected - all channels look good!")
        
        logger.info("="*60)
        
        return bad_channels
    
    def mark_bad_channels(self, raw, bad_channels=None, method='pyprep', 
                         ransac=False, copy=True):
        """
        Mark bad channels in raw data

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        bad_channels : list, optional
            List of bad channel names to mark. If None, will detect using specified method.
        method : str
            Method to detect bad channels if bad_channels is None. Default: 'pyprep'
        ransac : bool
            True: try RANSAC method, if it fails, a warning will be issued and the process will automatically skip. (RANSAC has channel position issues)
            Using other methods: correlation + deviation + HF noise only.
        copy : bool
            If True, operate on a copy of raw data.
        Returns
        -------
        raw : mne.io.Raw
            Raw data with bad channels marked in raw.info['bads']
        bad_channels : list
            List of bad channel names
            
        """
        if copy:
            raw = raw.copy()
        
        if bad_channels is None:
            if method == 'pyprep':
                bad_channels = self.detect_bad_channels_pyprep(raw, ransac=ransac)
            else:
                logger.warning(f"Unknown method '{method}'")
                bad_channels = []
        else:
            logger.info(f"Using manual bad channels: {bad_channels}")
        
        if bad_channels:
            raw.info['bads'] = bad_channels
            self.bad_channels = bad_channels
            self.processing_history.append(f'marked_{len(bad_channels)}_bad')
            logger.info(f"\nMarked {len(bad_channels)} bad channels")
        else:
            raw.info['bads'] = []
            self.bad_channels = []
            logger.info("\nNo bad channels to mark")
        
        return raw, bad_channels
    
    def interpolate_bad_channels(self, raw, reset_bads=True, copy=True):
        """Interpolate bad channels"""
        if copy:
            raw = raw.copy()
        
        if not raw.info['bads']:
            logger.info("No bad channels to interpolate")
            return raw
        
        logger.info("="*60)
        logger.info(f"Interpolating: {raw.info['bads']}")
        
        raw.interpolate_bads(reset_bads=reset_bads)
        
        n = len(self.bad_channels)
        self.processing_history.append(f'interpolated_{n}')
        logger.info(f"✓ Interpolated {n} channels")
        logger.info("="*60)
        
        return raw
    
    def apply_ica(self, raw, n_components=None, method='fastica',
                  random_state=42, copy=True):
        """
        Apply ICA
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        n_components : int, optional
            Number of ICA components to compute. If None, uses min(n_channels, n_times // 2).
        method : 'fastica' | 'infomax' | 'picard' 
            ICA method. Default: 'fastica'
        random_state : int
            Random state for reproducibility
        copy : bool
            If True, operate on a copy of raw data. 

        Returns 
        -------
        raw : mne.io.Raw
            Raw data (unchanged)
        ica : mne.preprocessing.ICA
            Fitted ICA object

        """
        if copy:
            raw = raw.copy()
        
        logger.info("="*60)
        logger.info("APPLYING ICA")
        logger.info("="*60)
        
        if n_components is None:
            n_components = min(len(raw.ch_names), raw.n_times // 2)
        
        logger.info(f"Components: {n_components}")
        logger.info(f"Method: {method}")
        
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            max_iter='auto'
        )
        
        logger.info("Fitting ICA...")
        ica.fit(raw)
        
        logger.info(f"✓ Fitted {ica.n_components_} components")
        self.processing_history.append(f'ica_{n_components}')

        # ---- ① 所有组件对各类通道解释的总方差 ----
        try:
            explained_var_ratio = ica.get_explained_variance_ratio(raw)
            for ch_type, ratio in explained_var_ratio.items():
                logger.info(
                    f"Fraction of {ch_type} variance explained by all components: {ratio:.3f} ({ratio * 100:.1f}%)"                    
                )
        except Exception as e:
            logger.warning(f"Could not compute explained variance ratio (all components): {e!r}")
                    
        return raw, ica
    
    def classify_ica_components_iclabel(self, ica, raw):
        """
        Classify ICA components using MNE-ICLabel.
        
        This method does NOT require EOG/ECG channels and works with
        any EEG data. It classifies each component into 7 categories:
        - brain
        - eye (eye artifacts)
        - heart (cardiac artifacts)
        - muscle (muscle artifacts)
        - line_noise (line noise)
        - channel_noise (channel noise)
        - other (other artifacts)
        
        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        raw : mne.io.Raw
            Raw EEG data
            
        Returns
        -------
        labels_pred : np.ndarray
            Predicted labels for each component (0-6)
        labels_pred_proba : np.ndarray
            Probability matrix (n_components × 7)
        label_names : list
            Category names
        """
        try:
            from mne_icalabel import label_components
        except ImportError:
            raise ImportError(
                "Install mne-icalabel: pip install mne-icalabel\n"
                "See: https://mne.tools/mne-icalabel/"
            )
        
        logger.info("\n" + "="*60)
        logger.info("CLASSIFYING ICA COMPONENTS WITH ICLabel")
        logger.info("="*60)
        
        # Run ICLabel classification
        logger.info("\n[1] Running ICLabel neural network classifier...")
        ic_labels = label_components(raw, ica, method='iclabel')
        
        labels_pred = ic_labels['labels']
        labels_pred_proba = ic_labels['y_pred_proba']
        
        # Label names (in order)
        label_names = ['brain', 'eye', 'heart', 'muscle', 'line_noise', 'channel_noise', 'other']
        
        # Debug: Check shapes
        logger.info(f"Debug: labels_pred type: {type(labels_pred)}")
        logger.info(f"Debug: labels_pred length: {len(labels_pred)}")
        logger.info(f"Debug: labels_pred_proba shape: {labels_pred_proba.shape}")
        
        # UPDATED FIX: Handle new mne-icalabel format
        # Newer versions return 1D array with max probabilities only
        # But labels_pred contains the correct classifications
        
        if labels_pred_proba.ndim == 1:
            n_components = len(labels_pred)
            expected_flat_size = n_components * len(label_names)
            
            if len(labels_pred_proba) == expected_flat_size:
                # Flattened 2D matrix - reshape it
                logger.info("Detected flattened probability matrix - reshaping...")
                labels_pred_proba = labels_pred_proba.reshape(n_components, len(label_names))
                logger.info(f"✓ Reshaped to: {labels_pred_proba.shape}")
            elif len(labels_pred_proba) == n_components:
                # Max probabilities only (new format)
                logger.info("Detected 1D max probabilities (new mne-icalabel format)")
                logger.info("✓ Will use labels_pred directly for component selection")
                logger.info("  (This is the correct behavior for newer versions)")
                
                # Store as 1D - we'll handle this in select_ica_components_by_label
                # No need to error out - we have all the info we need in labels_pred
            else:
                logger.error(f"Unexpected labels_pred_proba shape: {labels_pred_proba.shape}")
                logger.error(f"Expected either ({n_components},) or ({n_components}, {len(label_names)})")
                raise ValueError(
                    f"Incompatible labels_pred_proba shape: {labels_pred_proba.shape}\n"
                    f"Try: pip install --upgrade mne-icalabel"
                )
        
        # Handle different data structures
        # labels_pred might be indices (integers) or strings
        if len(labels_pred) > 0:
            if isinstance(labels_pred[0], (int, np.integer)):
                # Convert indices to strings
                labels_pred_str = np.array([label_names[i] for i in labels_pred])
            else:
                labels_pred_str = labels_pred
        else:
            labels_pred_str = labels_pred
        
        # Store results
        self.ica_component_labels = {
            'labels': labels_pred_str,
            'probabilities': labels_pred_proba,
            'label_names': label_names
        }
        
        logger.info("\n[2] Classification results:")
        logger.info("="*60)
        
        # Count by category
        label_counts = {}
        for label_name in label_names:
            label_counts[label_name] = np.sum(labels_pred_str == label_name)
        
        for label_name, count in label_counts.items():
            if count > 0:
                logger.info(f"  {label_name:15s}: {count:2d} components")
        
        logger.info("="*60)
        
        # Show detailed component info
        logger.info("\n[3] Component details:")
        logger.info("="*60)
        
        # Check format
        is_1d_format = labels_pred_proba.ndim == 1 and len(labels_pred_proba) == len(labels_pred_str)
        
        if is_1d_format:
            # New format: show label and max probability
            for i in range(len(labels_pred_str)):
                label_str = labels_pred_str[i]
                max_prob = float(labels_pred_proba[i])
                logger.info(f"  IC{i:2d}: {label_str:20s} (p={max_prob:.3f})")
        else:
            # Old format: show label and its probability from full matrix
            for i in range(len(labels_pred_str)):
                label_str = labels_pred_str[i]
                
                if i < len(labels_pred_proba):
                    probs = labels_pred_proba[i]
                    
                    try:
                        # Get the index of the predicted label
                        if label_str in label_names:
                            label_idx = label_names.index(label_str)
                            label_prob = float(probs[label_idx])
                        else:
                            # If label not in standard names, use max probability
                            label_idx = np.argmax(probs)
                            label_prob = float(probs[label_idx])
                        
                        logger.info(f"  IC{i:2d}: {label_str:20s} (p={label_prob:.3f})")
                        
                    except (ValueError, IndexError, TypeError) as e:
                        logger.warning(f"  IC{i:2d}: {label_str:20s} (p=N/A) - Error: {e}")
                else:
                    logger.warning(f"  IC{i:2d}: {label_str:20s} (p=N/A) - No probability data")
        
        logger.info("="*60)
        
        return labels_pred_str, labels_pred_proba, label_names
    
    def select_ica_components_by_label(self, labels_pred, labels_pred_proba, 
                                       label_names, exclude_labels=None,
                                       brain_threshold=0.5, artifact_threshold=0.5):
        """
        Select ICA components to exclude based on ICLabel classification.
        
        Strategy:
        - Keep components with high 'brain' probability (>= brain_threshold)
        - Exclude components with high artifact probability (>= artifact_threshold)
        - For uncertain components, keep them (conservative approach)
        
        UPDATED: Uses partial matching for labels to handle detailed labels like
        'eye blink', 'muscle artifact', etc.
        
        Parameters
        ----------
        labels_pred : np.ndarray
            Predicted labels
        labels_pred_proba : np.ndarray
            Probability matrix
        label_names : list
            Category names
        exclude_labels : list, optional
            Labels to exclude. Default: ['eye', 'heart', 'muscle', 'line_noise', 'channel_noise']
            Now supports partial matching (e.g., 'eye' matches 'eye blink')
        brain_threshold : float
            Minimum probability to confidently classify as 'brain' (keep these)
        artifact_threshold : float
            Minimum probability to confidently classify as artifact (exclude these)
            
        Returns
        -------
        exclude_idx : list
            Component indices to exclude
        """
        if exclude_labels is None:
            exclude_labels = ['eye', 'heart', 'muscle', 'line_noise', 'channel_noise']
        
        logger.info("\n" + "="*60)
        logger.info("SELECTING COMPONENTS TO EXCLUDE")
        logger.info("="*60)
        logger.info(f"Artifact labels: {exclude_labels}")
        logger.info(f"Brain threshold: {brain_threshold:.2f}")
        logger.info(f"Artifact threshold: {artifact_threshold:.2f}")
        
        exclude_idx = []
        
        # Handle different probability formats
        is_1d_format = labels_pred_proba.ndim == 1 and len(labels_pred_proba) == len(labels_pred)
        
        if is_1d_format:
            # New format: 1D array with max probabilities only
            # Use labels_pred directly for classification
            logger.info("\nUsing new format (1D probabilities)")
            logger.info("Will use labels_pred directly for component selection")
            
            for i in range(len(labels_pred)):
                label = labels_pred[i]
                label_str = str(label) if not isinstance(label, str) else label
                max_prob = float(labels_pred_proba[i])
                
                # Check if label matches any exclude pattern (partial matching)
                label_matches_exclude = False
                for exclude_pattern in exclude_labels:
                    if exclude_pattern.lower() in label_str.lower():
                        label_matches_exclude = True
                        break
                
                # Decision logic (simplified for 1D format)
                should_exclude = False
                reason = ""
                
                if 'brain' in label_str.lower():
                    # Brain component
                    if max_prob >= brain_threshold:
                        should_exclude = False
                        reason = f"brain (p={max_prob:.3f}) - KEEP"
                    else:
                        # Low confidence brain - keep (conservative)
                        should_exclude = False
                        reason = f"brain (p={max_prob:.3f}) - low confidence, KEEP"
                
                elif label_matches_exclude:
                    # Artifact component
                    if max_prob >= artifact_threshold:
                        should_exclude = True
                        reason = f"{label_str} (p={max_prob:.3f}) - EXCLUDE"
                    else:
                        # Low confidence artifact - keep (conservative)
                        should_exclude = False
                        reason = f"{label_str} (p={max_prob:.3f}) - low confidence, KEEP"
                
                else:
                    # Other/unknown - keep
                    should_exclude = False
                    reason = f"{label_str} (p={max_prob:.3f}) - KEEP"
                
                if should_exclude:
                    exclude_idx.append(i)
                    logger.info(f"  IC{i:2d}: {reason}")
        
        else:
            # Old format: 2D array with full probability matrix
            logger.info("\nUsing old format (2D probabilities)")
            
            # Verify it's actually 2D
            if labels_pred_proba.ndim != 2:
                logger.error(f"labels_pred_proba should be 2D but got shape: {labels_pred_proba.shape}")
                return []
            
            if labels_pred_proba.shape[1] != len(label_names):
                logger.error(f"labels_pred_proba has {labels_pred_proba.shape[1]} classes, expected {len(label_names)}")
                return []
            
            brain_idx = label_names.index('brain')
            
            for i in range(len(labels_pred)):
                label = labels_pred[i]
                label_str = str(label) if not isinstance(label, str) else label
                
                # Get probability array for this component
                probs = labels_pred_proba[i]
                
                # Get probabilities
                brain_prob = float(probs[brain_idx])
                
                # Find the probability for the predicted label
                try:
                    if label_str in label_names:
                        label_idx = label_names.index(label_str)
                    else:
                        # For non-standard labels, find which category has highest probability
                        label_idx = np.argmax(probs)
                    
                    artifact_prob = float(probs[label_idx])
                except (ValueError, IndexError) as e:
                    logger.warning(f"  IC{i:2d}: Unknown label '{label_str}' - KEEP (error: {e})")
                    continue
                
                # Check if label matches any exclude pattern (partial matching)
                label_matches_exclude = False
                for exclude_pattern in exclude_labels:
                    if exclude_pattern.lower() in label_str.lower():
                        label_matches_exclude = True
                        break
                
                # Decision logic
                should_exclude = False
                reason = ""
                
                if 'brain' in label_str.lower() and brain_prob >= brain_threshold:
                    should_exclude = False
                    reason = f"brain (p={brain_prob:.3f}) - KEEP"
                elif label_matches_exclude and artifact_prob >= artifact_threshold:
                    should_exclude = True
                    reason = f"{label_str} (p={artifact_prob:.3f}) - EXCLUDE"
                else:
                    should_exclude = False
                    reason = f"{label_str} (p={artifact_prob:.3f}) - uncertain, KEEP"
                
                if should_exclude:
                    exclude_idx.append(i)
                    logger.info(f"  IC{i:2d}: {reason}")
        
        logger.info("="*60)
        logger.info(f"\nTotal components to exclude: {len(exclude_idx)}")
        if exclude_idx:
            logger.info(f"Component indices: {exclude_idx}")
        else:
            logger.info("No components to exclude (all look good)")
        logger.info("="*60)
        
        return exclude_idx
    
    def detect_ica_artifacts(self, ica, raw, artifact_types=None, 
                            use_iclabel=True, **iclabel_kwargs):
        """
        Detect ICA artifact components.
        
        UPDATED: Supports both traditional EOG/ECG detection and ICLabel classification.
        
        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Fitted ICA object
        raw : mne.io.Raw
            Raw EEG data
        artifact_types : list, optional
            Traditional methods: ['eog', 'ecg']
            Only used if use_iclabel=False
        use_iclabel : bool
            If True, use ICLabel for automatic classification (recommended)
            If False, use traditional EOG/ECG detection (requires these channels)
        **iclabel_kwargs
            Additional arguments for ICLabel selection:
            - exclude_labels: list of artifact types to exclude
            - brain_threshold: float (default 0.5)
            - artifact_threshold: float (default 0.5)
            
        Returns
        -------
        exclude_idx : list
            Component indices to exclude
        """
        if artifact_types is None:
            artifact_types = ['eog', 'ecg']
        
        exclude_idx = []
        
        logger.info("\n" + "="*60)
        logger.info("DETECTING ARTIFACTS")
        logger.info("="*60)
        
        # METHOD 1: ICLabel (recommended for data without EOG/ECG)
        if use_iclabel:
            logger.info("Method: ICLabel automatic classification")
            
            # Classify components
            labels_pred, labels_pred_proba, label_names = \
                self.classify_ica_components_iclabel(ica, raw)
            
            # Select components to exclude
            exclude_idx = self.select_ica_components_by_label(
                labels_pred, labels_pred_proba, label_names,
                **iclabel_kwargs
            )
        
        # METHOD 2: Traditional EOG/ECG detection (requires these channels)
        else:
            logger.info("Method: Traditional EOG/ECG detection")
            logger.info("  (Requires EOG/ECG channels in data)")
            
            if 'eog' in artifact_types:
                logger.info("\nEOG artifacts...")
                try:
                    eog_idx, _ = ica.find_bads_eog(raw, threshold=3.0)
                    exclude_idx.extend(eog_idx)
                    if eog_idx:
                        logger.info(f"✓ Found {len(eog_idx)} EOG: {eog_idx}")
                    else:
                        logger.info("✓ No EOG components")
                except Exception as e:
                    logger.warning(f"EOG detection failed: {e}")
            
            if 'ecg' in artifact_types:
                logger.info("\nECG artifacts...")
                try:
                    ecg_idx, _ = ica.find_bads_ecg(raw, threshold=3.0)
                    exclude_idx.extend(ecg_idx)
                    if ecg_idx:
                        logger.info(f"✓ Found {len(ecg_idx)} ECG: {ecg_idx}")
                    else:
                        logger.info("✓ No ECG components")
                except Exception as e:
                    logger.warning(f"ECG detection failed: {e}")
            
            exclude_idx = sorted(list(set(exclude_idx)))
            
            logger.info(f"\nTotal artifacts: {len(exclude_idx)}")
            if exclude_idx:
                logger.info(f"Components: {exclude_idx}")
        
        logger.info("="*60)
        
        return exclude_idx
    
    def apply_ica_cleaning(self, raw, ica, exclude_idx=None, 
                          auto_detect=True, use_iclabel=True, copy=True,
                          **iclabel_kwargs):
        """
        Remove artifact components.
        
        UPDATED: Supports ICLabel for automatic detection without EOG/ECG channels.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        ica : mne.preprocessing.ICA
            Fitted ICA object
        exclude_idx : list, optional
            Manual component indices to exclude
        auto_detect : bool
            Automatically detect artifacts
        use_iclabel : bool
            Use ICLabel for detection (recommended)
        copy : bool
            Copy data
        **iclabel_kwargs
            Additional arguments for ICLabel selection
            
        Returns
        -------
        raw : mne.io.Raw
            Cleaned data
        """
        if copy:
            raw = raw.copy()
        
        if exclude_idx is None and auto_detect:
            exclude_idx = self.detect_ica_artifacts(
                ica, raw, use_iclabel=use_iclabel, **iclabel_kwargs
            )
        elif exclude_idx is None:
            exclude_idx = []
        
        if exclude_idx:
            logger.info("\n" + "="*60)
            logger.info("APPLYING ICA CLEANING")
            logger.info("="*60)
            logger.info(f"Removing {len(exclude_idx)} components: {exclude_idx}")
            
            ica.exclude = exclude_idx
            raw = ica.apply(raw)
            
            self.processing_history.append(f'removed_{len(exclude_idx)}_ica')
            logger.info(f"✓ Removed {len(exclude_idx)} components")
            logger.info("="*60)
        else:
            logger.info("No components to remove")
        
        return raw
    
    def create_fixed_length_epochs(self, raw, duration=2.0, overlap=0.0,
                                   reject=None, flat=None, copy=True):
        """
        Create fixed-length epochs from continuous data.
        
        Useful for resting-state or continuous recordings without events.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Cleaned continuous EEG data
        duration : float
            Epoch duration in seconds (default: 2.0)
        overlap : float
            Overlap between epochs in seconds (default: 0.0)
        reject : dict, optional
            Rejection thresholds (e.g., {'eeg': 100e-6})
        flat : dict, optional
            Flat channel detection thresholds
        copy : bool
            Copy data
            
        Returns
        -------
        epochs : mne.Epochs
            Fixed-length epochs
        """
        if copy:
            raw = raw.copy()
        
        logger.info("\n" + "="*60)
        logger.info("CREATING FIXED-LENGTH EPOCHS")
        logger.info("="*60)
        logger.info(f"Duration: {duration} s")
        logger.info(f"Overlap: {overlap} s")
        
        # Create fixed-length epochs
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap,
            preload=True
        )
        
        n_epochs = len(epochs)
        logger.info(f"✓ Created {n_epochs} epochs")
        
        # Apply rejection criteria if provided
        if reject is not None or flat is not None:
            logger.info("\nApplying rejection criteria...")
            if reject:
                logger.info(f"  Reject: {reject}")
            if flat:
                logger.info(f"  Flat: {flat}")
            
            epochs_before = len(epochs)
            epochs.drop_bad(reject=reject, flat=flat)
            epochs_after = len(epochs)
            
            n_dropped = epochs_before - epochs_after
            logger.info(f"✓ Dropped {n_dropped} bad epochs")
            logger.info(f"✓ Remaining: {epochs_after} epochs")
        
        self.processing_history.append(f'epochs_{duration}s_{len(epochs)}')
        logger.info("="*60)
        
        return epochs
    
    def apply_autoreject(self, epochs, n_interpolate=None, consensus=None,
                        n_jobs=1, random_state=42, verbose=False, 
                        reject_mode='drop'):
        """
        Apply Autoreject for automatic epoch-level artifact rejection.
        
        Autoreject learns optimal rejection thresholds from the data and can:
        1. Interpolate bad channels within epochs
        2. Reject epochs that are too noisy
        
        This provides a second line of defense after ICA cleaning.
        
        UPDATED: Added reject_mode parameter to control epoch handling.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to clean
        n_interpolate : array-like, optional
            Number of channels to interpolate (default: [1, 4, 8, 16])
        consensus : array-like, optional
            Consensus percs for rejection (default: [0.1, 0.2, 0.5, 0.8, 1.0])
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random state for reproducibility
        verbose : bool
            Verbose output
        reject_mode : str
            'drop': Remove bad epochs (default)
            'mark': Keep all epochs but mark bad ones in epochs.drop_log
            'return_all': Return both clean and all epochs with bad indices
            
        Returns
        -------
        epochs_clean : mne.Epochs
            Cleaned epochs (interpolated, bad ones marked/dropped based on reject_mode)
        ar : autoreject.AutoReject
            Fitted AutoReject object
        reject_log : autoreject.RejectLog
            Log of rejected/interpolated epochs
        bad_epochs_idx : list (only if reject_mode='mark' or 'return_all')
            Indices of rejected epochs
        """
        try:
            from autoreject import AutoReject
        except ImportError:
            raise ImportError(
                "Install autoreject: pip install autoreject\n"
                "See: https://autoreject.github.io/"
            )
        
        logger.info("\n" + "="*60)
        logger.info("APPLYING AUTOREJECT")
        logger.info("="*60)
        logger.info("Autoreject will learn optimal thresholds and:")
        logger.info("  1. Interpolate bad channels within epochs")
        logger.info("  2. Reject epochs that are too noisy")
        logger.info(f"\nReject mode: {reject_mode}")
        
        # Set defaults if not provided
        if n_interpolate is None:
            n_channels = len(epochs.ch_names)
            n_interpolate = np.array([1, 4, 8, min(16, n_channels // 2)])
            n_interpolate = n_interpolate[n_interpolate < n_channels]
        
        if consensus is None:
            consensus = np.linspace(0.1, 1.0, 5)
        
        logger.info(f"\nn_interpolate: {n_interpolate}")
        logger.info(f"consensus: {consensus}")
        
        # Create and fit AutoReject
        logger.info("\n[1] Learning optimal thresholds (this may take a while)...")
        ar = AutoReject(
            n_interpolate=n_interpolate,
            consensus=consensus,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        
        ar.fit(epochs)
        logger.info("✓ AutoReject fitted")
        
        # Apply cleaning
        logger.info("\n[2] Applying cleaning...")
        epochs_clean, reject_log = ar.transform(epochs, return_log=True)
        
        # Get bad epoch indices
        bad_epochs_idx = np.where(reject_log.bad_epochs)[0].tolist()
        
        # Report results
        n_total = len(epochs)
        n_interpolated = np.sum(reject_log.bad_epochs == False)  # epochs with interpolation
        n_rejected = len(bad_epochs_idx)
        n_clean = len(epochs_clean)
        
        logger.info("\n" + "="*60)
        logger.info("AUTOREJECT RESULTS")
        logger.info("="*60)
        logger.info(f"Total epochs: {n_total}")
        logger.info(f"  - Good/Interpolated: {n_interpolated}")
        logger.info(f"  - Rejected: {n_rejected} ({100*n_rejected/n_total:.1f}%)")
        
        if bad_epochs_idx:
            logger.info(f"\nRejected epoch indices: {bad_epochs_idx}")
            logger.info(f"Rejected epoch numbers: {[i for i in bad_epochs_idx]}")
        
        # Handle different rejection modes
        if reject_mode == 'mark':
            # Keep all epochs but mark bad ones
            logger.info("\nMode: MARK - Keeping all epochs, marking bad ones")
            epochs_all = epochs.copy()
            
            # Store bad epochs info as custom attribute
            # (We can't modify drop_log as it's read-only in MNE)
            epochs_all._bad_epochs_autoreject = bad_epochs_idx
            epochs_all._autoreject_log = reject_log
            
            logger.info(f"✓ All {n_total} epochs retained with {n_rejected} marked as bad")
            logger.info(f"✓ Bad epochs stored in: epochs._bad_epochs_autoreject")
            logger.info("="*60)
            
            self.processing_history.append(f'autoreject_{n_rejected}marked')
            return epochs_all, ar, reject_log, bad_epochs_idx
            
        elif reject_mode == 'return_all':
            # Return both clean epochs and full info
            logger.info("\nMode: RETURN_ALL - Returning clean epochs + bad indices")
            logger.info(f"✓ Clean epochs: {n_clean}")
            logger.info(f"✓ Bad epoch indices saved for reference")
            logger.info("="*60)
            
            self.processing_history.append(f'autoreject_{n_rejected}dropped')
            return epochs_clean, ar, reject_log, bad_epochs_idx
            
        else:  # reject_mode == 'drop' (default)
            # Drop bad epochs (default behavior)
            logger.info("\nMode: DROP - Removing bad epochs")
            logger.info(f"✓ Final clean epochs: {n_clean}")
            logger.info("="*60)
            
            self.processing_history.append(f'autoreject_{n_rejected}dropped')
            return epochs_clean, ar, reject_log
    
    def apply_average_reference(self, raw, ref_channel='REF CZ', copy=True,
                                set_ref_channel=True):
        """
        Apply average reference using 3-step process.
        
        Steps:
        1. Re-reference to REF CZ
        2. Drop REF CZ (now zeros)
        3. Average reference remaining EEG
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        ref_channel : str or None
            Reference channel name (e.g., 'REF CZ')
            If None, direct average reference
        copy : bool
            Copy data
        set_ref_channel : bool
            Set the reference channel as 'misc'. Set to False to
            retain the physical reference channel in the data.
            
        Returns
        -------
        raw : mne.io.Raw
            Re-referenced data
        """
        if copy:
            raw = raw.copy()
        
        logger.info("="*60)
        logger.info("APPLYING AVERAGE REFERENCE")
        logger.info("="*60)
        
        if ref_channel is not None and ref_channel in raw.ch_names:
            logger.info(f"\n[Step 1] Re-reference to {ref_channel}")
            raw.set_eeg_reference(ref_channels=[ref_channel], projection=False)      # 重新参考到 REF CZ

            logger.info(f"✓ Re-referenced to {ref_channel}")

            if  set_ref_channel:
                logger.info(f"\n[Step 2] Set {ref_channel} as 'misc'")
                raw.set_channel_types({ref_channel: 'misc'})
                logger.info(f"✓ {ref_channel} has been set to 'misc'")
            else:
                logger.info(f"\n[Step 2] Keeping {ref_channel} in the data")

            logger.info("\n[Step 3] Average reference")
            raw.set_eeg_reference('average', projection=True)
            logger.info("✓ Applied average reference")
            
            self.processing_history.append(f'avg_ref_via_{ref_channel}')
        else:
            if ref_channel:
                logger.warning(f"'{ref_channel}' not found, using direct average")
            
            logger.info("Direct average reference")
            raw.set_eeg_reference('average', projection=True)
            logger.info("✓ Applied average reference")
            
            self.processing_history.append('avg_ref_direct')
        
        raw.apply_proj()    # 应用投影器 
        logger.info("="*60)
        
        return raw
    
    def get_processing_summary(self):
        """Get processing summary"""
        summary = "\nProcessing History:\n"
        summary += "="*60 + "\n"
        for i, step in enumerate(self.processing_history, 1):
            summary += f"  {i}. {step}\n"
        summary += "="*60
        return summary


def preprocess_eeg_complete(raw, detect_bad_channels=True, ransac=False,
                           interpolate=True, apply_ica=True,
                           n_ica_components=None, use_iclabel=True,
                           apply_reference=True, ref_channel='REF CZ',
                           drop_reference_channel=True,
                           reference_before_ica=True,
                           create_epochs=True, epoch_duration=2.0,
                           epoch_overlap=0.0, epoch_tmin=None,
                           epoch_tmax=None, epoch_baseline=None,
                           apply_autoreject=True,
                           autoreject_n_jobs=1, autoreject_reject_mode='drop',
                           **iclabel_kwargs):
    """
    Complete EEG preprocessing pipeline with ICLabel and Autoreject.
    
    Pipeline:
    1. Detect bad channels (pyprep)
    2. Interpolate bad channels
    3. Apply average reference (if reference_before_ica=True, RECOMMENDED)
    4. Apply ICA
    5. Remove artifacts with ICLabel (automatic, no EOG/ECG needed)
    6. Apply average reference (if reference_before_ica=False)
    7. Create fixed-length epochs (if create_epochs=True)
    8. Apply Autoreject for epoch-level cleaning (if apply_autoreject=True)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Filtered EEG data
    detect_bad_channels : bool
        Detect bad channels with pyprep
    ransac : bool
        DEPRECATED: Keep as False (RANSAC has issues)
    interpolate : bool
        Interpolate bad channels
    apply_ica : bool
        Apply ICA artifact removal
    n_ica_components : int, optional
        Number of ICA components
    use_iclabel : bool
        Use ICLabel for automatic ICA component classification (recommended)
        If False, uses traditional EOG/ECG detection (requires these channels)
    apply_reference : bool
        Apply average reference (3-step process)
    ref_channel : str or None
        Reference channel name (e.g., 'REF CZ')
    drop_reference_channel : bool
        Drop the reference channel after re-referencing. Set to False to keep
        the reference channel.
    reference_before_ica : bool
        If True, apply reference BEFORE ICA (RECOMMENDED for ICLabel)
        If False, apply reference AFTER ICA (old behavior)
        Default: True
    create_epochs : bool
        Create fixed-length epochs from continuous data
    epoch_duration : float
        Epoch duration in seconds (default: 2.0)
    epoch_overlap : float
        Overlap between epochs in seconds (default: 0.0)
    epoch_tmin : float, optional
        Start time of each epoch in seconds relative to its reference point.
        If provided together with ``epoch_tmax``, these values will be used to
        shift the fixed-length epochs so that they span ``epoch_tmin`` to
        ``epoch_tmax``.
    epoch_tmax : float, optional
        End time of each epoch in seconds relative to its reference point.
        Requires ``epoch_tmin`` to be set.
    epoch_baseline : tuple | None
        Baseline correction interval (in seconds) to apply to the epochs. Use
        the same convention as ``mne.Epochs``.
    apply_autoreject : bool
        Apply Autoreject for automatic epoch cleaning
    autoreject_n_jobs : int
        Number of parallel jobs for Autoreject
    autoreject_reject_mode : str
        'drop': Remove bad epochs (default)
        'mark': Keep all epochs but mark bad ones
        'return_all': Return both clean epochs and bad indices
    **iclabel_kwargs
        Additional arguments for ICLabel component selection:
        - exclude_labels: list of artifact types
        - brain_threshold: float
        - artifact_threshold: float
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'raw': Cleaned continuous data (if create_epochs=False)
        - 'epochs': Cleaned epochs (if create_epochs=True and apply_autoreject=False)
        - 'epochs_clean': Autoreject-cleaned epochs (if apply_autoreject=True)
        - 'preprocessor': EEGPreprocessor object
        - 'ica': ICA object (or None)
        - 'autoreject': AutoReject object (or None)
        - 'reject_log': Rejection log (or None)
        - 'bad_epochs_idx': Bad epoch indices (if autoreject_reject_mode='mark' or 'return_all')
        
    Examples
    --------
    # Standard pipeline with ICLabel and Autoreject (RECOMMENDED)
    >>> result = preprocess_eeg_complete(
    ...     raw,
    ...     use_iclabel=True,
    ...     reference_before_ica=True,  # ICLabel recommendation
    ...     apply_autoreject=True,
    ...     epoch_duration=2.0
    ... )
    >>> epochs_clean = result['epochs_clean']
    
    # Keep all epochs but mark bad ones (for alignment with LFP)
    >>> result = preprocess_eeg_complete(
    ...     raw,
    ...     use_iclabel=True,
    ...     reference_before_ica=True,
    ...     apply_autoreject=True,
    ...     autoreject_reject_mode='mark',
    ...     epoch_duration=2.0
    ... )
    >>> epochs_all = result['epochs_clean']  # Contains all epochs
    >>> bad_idx = result['bad_epochs_idx']   # Indices of bad epochs
    >>> # Bad epochs also stored in: epochs_all._bad_epochs_autoreject
    
    # Adjust ICLabel thresholds
    >>> result = preprocess_eeg_complete(
    ...     raw,
    ...     use_iclabel=True,
    ...     reference_before_ica=True,
    ...     brain_threshold=0.6,
    ...     artifact_threshold=0.4
    ... )
    """
    preprocessor = EEGPreprocessor()
    ica = None
    ar = None
    reject_log = None
    
    logger.info("\n" + "="*70)
    logger.info("EEG PREPROCESSING PIPELINE (WITH ICLabel & Autoreject)")
    logger.info("="*70)
    
    if ransac:
        logger.warning("RANSAC disabled - using correlation + deviation + HF noise")
        ransac = False
    
    # ========================================
    # CONTINUOUS DATA PREPROCESSING
    # ========================================
    
    # Detect bad channels
    if detect_bad_channels:
        logger.info("\n[STEP 1] Detecting bad channels")
        raw, bad_channels = preprocessor.mark_bad_channels(
            raw, method='pyprep', ransac=ransac, copy=False
        )
    else:
        logger.info("\n[STEP 1] Skipping detection")
        bad_channels = []
    
    # Interpolate
    if interpolate and bad_channels:
        logger.info("\n[STEP 2] Interpolating")
        raw = preprocessor.interpolate_bad_channels(raw, copy=False)
    else:
        logger.info("\n[STEP 2] No interpolation needed")
    
    # Apply reference BEFORE ICA (recommended for ICLabel)
    if apply_reference and reference_before_ica:
        logger.info("\n[STEP 3] Re-referencing (BEFORE ICA)")
        logger.info("✓ This is the RECOMMENDED approach for ICLabel")
        raw = preprocessor.apply_average_reference(
            raw,
            ref_channel=ref_channel,
            copy=False,
            drop_ref_channel=drop_reference_channel
        )
    elif not reference_before_ica:
        logger.info("\n[STEP 3] Skipping re-reference (will apply after ICA)")
    else:
        logger.info("\n[STEP 3] Skipping re-reference")
        if use_iclabel:
            logger.warning("⚠️  Warning: ICLabel works best with average reference before ICA")
    
    # ICA with ICLabel or traditional detection
    if apply_ica:
        logger.info("\n[STEP 4] ICA")
        raw, ica = preprocessor.apply_ica(raw, n_components=n_ica_components, copy=False)
        
        logger.info("\n[STEP 5] Artifact removal")
        if use_iclabel:
            logger.info("Using ICLabel (automatic, no EOG/ECG needed)")
        else:
            logger.info("Using traditional EOG/ECG detection")
        
        raw = preprocessor.apply_ica_cleaning(
            raw, ica, auto_detect=True, use_iclabel=use_iclabel,
            copy=False, **iclabel_kwargs
        )
    else:
        logger.info("\n[STEP 4-5] Skipping ICA")
    
    # Apply reference AFTER ICA (old behavior, not recommended for ICLabel)
    if apply_reference and not reference_before_ica:
        logger.info("\n[STEP 6] Re-referencing (AFTER ICA)")
        logger.info("Note: For ICLabel, it's better to reference BEFORE ICA")
        raw = preprocessor.apply_average_reference(
            raw,
            ref_channel=ref_channel,
            copy=False,
            drop_ref_channel=drop_reference_channel
        )
    elif not apply_reference:
        logger.info("\n[STEP 6] Skipping re-reference")
    else:
        logger.info("\n[STEP 6] Re-reference already applied before ICA")
    
    # ========================================
    # EPOCH-LEVEL PREPROCESSING (OPTIONAL)
    # ========================================
    
    epochs = None
    epochs_clean = None
    bad_epochs_idx = None
    
    if create_epochs:
        logger.info("\n[STEP 7] Creating epochs")

        if epoch_tmin is not None and epoch_tmax is not None:
            epoch_duration = epoch_tmax - epoch_tmin
            if epoch_duration <= 0:
                raise ValueError("epoch_tmax must be greater than epoch_tmin")

            logger.info(
                f"Using custom epoch window: tmin={epoch_tmin}s, tmax={epoch_tmax}s"
            )
        elif (epoch_tmin is None) != (epoch_tmax is None):
            raise ValueError("Both epoch_tmin and epoch_tmax must be provided together")

        epochs = preprocessor.create_fixed_length_epochs(
            raw, duration=epoch_duration, overlap=epoch_overlap, copy=False
        )

        if epoch_tmin is not None and epoch_tmax is not None:
            epochs = epochs.shift_time(epoch_tmin, relative=True)
            logger.info(
                f"Shifted epochs to start at {epochs.tmin:.3f}s and end at {epochs.tmax:.3f}s"
            )

        if epoch_baseline is not None:
            logger.info(f"Applying baseline correction: {epoch_baseline}")
            epochs.apply_baseline(epoch_baseline)

        if apply_autoreject:
            logger.info("\n[STEP 8] Autoreject cleaning")
            ar_result = preprocessor.apply_autoreject(
                epochs, n_jobs=autoreject_n_jobs, reject_mode=autoreject_reject_mode
            )
            
            # Handle different return values based on reject_mode
            if autoreject_reject_mode in ['mark', 'return_all']:
                epochs_clean, ar, reject_log, bad_epochs_idx = ar_result
            else:  # 'drop'
                epochs_clean, ar, reject_log = ar_result
        else:
            logger.info("\n[STEP 8] Skipping Autoreject")
            epochs_clean = epochs
    else:
        logger.info("\n[STEP 7-8] Skipping epoch creation and Autoreject")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    logger.info("\n" + "="*70)
    logger.info("✅ PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(preprocessor.get_processing_summary())
    
    # Return results
    result = {
        'raw': raw,
        'epochs': epochs,
        'epochs_clean': epochs_clean,
        'preprocessor': preprocessor,
        'ica': ica,
        'autoreject': ar,
        'reject_log': reject_log,
        'bad_epochs_idx': bad_epochs_idx  # Indices of bad epochs (if available)
    }
    
    return result

