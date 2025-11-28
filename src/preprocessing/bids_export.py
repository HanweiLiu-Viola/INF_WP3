"""
BIDS Format Export Module (Updated)
====================================

Functions for exporting synchronized EEG and LFP data to BIDS format.
Now includes FIF format export for MNE-Python compatibility.

UPDATED: FIF files are saved as derivatives for preprocessing, maintaining BIDS compliance.
- BIDS-compliant files: BrainVision (EEG), TSV (LFP) in main structure
- Derivative files: FIF format (EEG + LFP) in derivatives/ for MNE-Python workflows
"""

import json
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from datetime import datetime
import mne_bids
from mne_bids import BIDSPath, write_raw_bids


# ============================================================
# BIDS Directory Structure
# ============================================================

def create_bids_structure(bids_root, subject_id, session='stim01'):
    """
    Create BIDS-compliant directory structure with derivatives.
    
    Parameters
    ----------
    bids_root : str or Path
        Root directory for BIDS dataset
    subject_id : str
        Subject identifier
    session : str
        Session identifier
        
    Returns
    -------
    tuple
        (bids_root, sub_dir, eeg_dir, ieeg_dir, deriv_eeg_dir, deriv_ieeg_dir)
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    
    # Create subject/session directories (main BIDS structure)
    sub_dir = bids_root / f"sub-{subject_id}" / f"ses-{session}"
    eeg_dir = sub_dir / "eeg"
    ieeg_dir = sub_dir / "ieeg"
    
    eeg_dir.mkdir(parents=True, exist_ok=True)
    ieeg_dir.mkdir(parents=True, exist_ok=True)
    
    # Create derivatives directory for FIF files
    deriv_root = bids_root / "derivatives" / "mne-python"
    deriv_sub_dir = deriv_root / f"sub-{subject_id}" / f"ses-{session}"
    deriv_eeg_dir = deriv_sub_dir / "eeg"
    deriv_ieeg_dir = deriv_sub_dir / "ieeg"
    
    deriv_eeg_dir.mkdir(parents=True, exist_ok=True)
    deriv_ieeg_dir.mkdir(parents=True, exist_ok=True)
    
    return bids_root, sub_dir, eeg_dir, ieeg_dir, deriv_eeg_dir, deriv_ieeg_dir


# ============================================================
# EEG Export - UPDATED WITH DERIVATIVES
# ============================================================

def save_eeg_bids(eeg_segment, subject_id, bids_root, deriv_eeg_dir,
                  stim_freq=55, task='rest', session='stim01'):
    """
    Save EEG data in BIDS format.
    
    UPDATED: Now saves FIF files in derivatives/ directory for preprocessing.
    - Raw BIDS: BrainVision format (standard)
    - Derivatives: FIF format with complete digitization (MNE-Python)
    
    Parameters
    ----------
    eeg_segment : mne.io.Raw
        Synchronized EEG segment
    subject_id : str
        Subject identifier
    bids_root : str or Path
        Root directory for BIDS dataset
    deriv_eeg_dir : Path
        Derivatives directory for EEG FIF files
    stim_freq : float
        Stimulation frequency (Hz)
    task : str
        Task name
    session : str
        Session identifier
        
    Returns
    -------
    BIDSPath
        BIDS path object
    """
    print("\n" + "="*60)
    print("SAVING EEG DATA IN BIDS FORMAT")
    print("="*60)
    
    bids_root = Path(bids_root)
    
    # Check for digitization information
    has_dig = eeg_segment.info.get('dig') is not None
    n_dig = len(eeg_segment.info['dig']) if has_dig else 0
    
    print(f"\n[1] Checking digitization information...")
    print(f"    Digitization points: {n_dig}")
    
    if has_dig:
        # Count different types of digitization points
        dig_types = {}
        for d in eeg_segment.info['dig']:
            kind = d['kind']
            dig_types[kind] = dig_types.get(kind, 0) + 1
        
        type_names = {1: 'Fiducials', 2: 'HPI', 3: 'EEG', 4: 'Extra'}
        for kind, count in dig_types.items():
            name = type_names.get(kind, f'Type-{kind}')
            print(f"      - {name}: {count}")
    
    # Create BIDS path
    bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task,
        datatype='eeg',
        root=bids_root
    )
    
    # Add recording date if not present
    if eeg_segment.info['meas_date'] is None:
        eeg_segment.info['meas_date'] = datetime.now()
    
    # Write EEG data as BrainVision (BIDS standard for EEG)
    print(f"\n[2] Writing EEG data (BrainVision format - BIDS standard)...")
    write_raw_bids(
        eeg_segment,
        bids_path,
        format='BrainVision',
        allow_preload=True,
        overwrite=True,
        verbose=False
    )
    
    print(f"    OK EEG data saved to: {bids_path.directory.relative_to(bids_root)}")
    
    # Save FIF format in derivatives (always, whether or not digitization present)
    print(f"\n[3] Saving FIF format in derivatives...")
    
    base_name = f"sub-{subject_id}_ses-{session}_task-{task}"
    
    # Save complete raw data as FIF in derivatives
    fif_path = deriv_eeg_dir / f"{base_name}_eeg.fif"
    eeg_segment.save(fif_path, overwrite=True, verbose=False)
    
    print(f"    OK Saved FIF: derivatives/mne-python/{fif_path.relative_to(bids_root / 'derivatives' / 'mne-python')}")
    
    if has_dig:
        print(f"    OK Digitization preserved: {n_dig} points")
        
        # Create a README for the derivatives directory
        readme_path = deriv_eeg_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("EEG Derivatives - FIF Format\n")
            f.write("="*50 + "\n\n")
            f.write("This directory contains EEG data in MNE-Python FIF format.\n\n")
            f.write("The FIF format preserves:\n")
            f.write(f"  - All digitization information ({n_dig} points)\n")
            for kind, count in dig_types.items():
                name = type_names.get(kind, f'Type-{kind}')
                f.write(f"    * {name}: {count}\n")
            f.write(f"  - Complete MNE metadata\n")
            f.write(f"  - Binary efficient storage\n\n")
            f.write(f"Usage:\n")
            f.write(f"```python\n")
            f.write(f"import mne\n")
            f.write(f"raw = mne.io.read_raw_fif('{fif_path.name}')\n")
            f.write(f"print(raw.info['dig'])  # View digitization points\n")
            f.write(f"```\n\n")
            f.write(f"Raw BIDS data (BrainVision format) is in:\n")
            f.write(f"  {bids_path.directory.relative_to(bids_root)}/\n")
        
        print(f"    OK Saved README: {readme_path.name}")
    else:
        print(f"    OK No digitization to preserve")
    
    # Verify BIDS files
    print(f"\n[4] Verifying BIDS files...")
    try:
        from mne_bids import read_raw_bids
        raw_reloaded = read_raw_bids(bids_path, verbose=False)
        reloaded_has_dig = raw_reloaded.info.get('dig') is not None
        reloaded_n_dig = len(raw_reloaded.info['dig']) if reloaded_has_dig else 0
        
        if reloaded_has_dig and reloaded_n_dig == n_dig:
            print(f"    OK BrainVision preserved all digitization: {reloaded_n_dig} points")
        elif reloaded_has_dig:
            print(f"    ! BrainVision partial preservation: {reloaded_n_dig}/{n_dig} points")
            print(f"    > Use derivatives FIF file for complete data")
        else:
            if has_dig:
                print(f"    ! BrainVision did not preserve digitization")
                print(f"    OK But derivatives FIF has complete data: {n_dig} points")
            else:
                print(f"    OK No digitization to preserve")
    except Exception as e:
        print(f"    ! Could not verify: {e}")
    
    # Add custom metadata
    print(f"\n[5] Adding custom metadata...")
    json_bids_path = bids_path.copy().update(suffix='eeg', extension='.json')
    json_path = Path(str(json_bids_path.fpath))
    
    with open(str(json_path), 'r') as f:
        metadata = json.load(f)
    
    metadata['DBSStimulationFrequency'] = stim_freq
    metadata['DBSStimulationPresent'] = True
    metadata['TaskDescription'] = f'Resting state with DBS stimulation at {stim_freq} Hz'
    
    # Add digitization info to metadata
    if has_dig:
        metadata['DigitizationPoints'] = n_dig
        coord_frame = eeg_segment.info['dig'][0]['coord_frame']
        coord_names = {1: 'Device', 4: 'Head', 5: 'MEG'}
        metadata['OriginalCoordinateFrame'] = coord_names.get(coord_frame, f'Frame-{coord_frame}')
        metadata['DerivativeFile'] = f"derivatives/mne-python/{fif_path.relative_to(bids_root / 'derivatives' / 'mne-python')}"
        metadata['DerivativeNote'] = "Complete data with digitization preserved in FIF format in derivatives/"
    
    with open(str(json_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    OK Metadata updated")
    
    return bids_path


# ============================================================
# LFP Export (as iEEG) - WITH DERIVATIVES
# ============================================================

def save_lfp_bids(df_lfp_sense, df_stim_aligned, df_settings, subject_id, 
                  bids_root, deriv_ieeg_dir, stim_freq=55, task='rest', 
                  session='stim01', save_fif=True, save_tsv=True):
    """
    Save LFP data in BIDS iEEG format.
    
    UPDATED: Now saves FIF files in derivatives/ directory.
    - Raw BIDS: TSV format (text, archival)
    - Derivatives: FIF format (binary, for preprocessing)
    
    Parameters
    ----------
    df_lfp_sense : pd.DataFrame
        LFP sensing data with columns ['voltage_left', 'voltage_right']
    df_stim_aligned : pd.DataFrame
        Aligned stimulation data with columns ['stim_amp_left', 'stim_amp_right']
    df_settings : pd.DataFrame
        LFP settings with sampling rate info
    subject_id : str
        Subject identifier
    bids_root : str or Path
        Root directory
    deriv_ieeg_dir : Path
        Derivatives directory for iEEG FIF files
    stim_freq : float
        Stimulation frequency (Hz)
    task : str
        Task name
    session : str
        Session identifier
    save_fif : bool
        If True, save as MNE-Python FIF format in derivatives/ (default: True)
        This is the format expected by the preprocessing pipeline
    save_tsv : bool
        If True, save as TSV format in raw BIDS (default: True)
        
    Returns
    -------
    Path
        iEEG directory path
    """
    print("\n" + "="*60)
    print("SAVING LFP DATA IN BIDS FORMAT (iEEG)")
    print("="*60)
    
    bids_root = Path(bids_root)
    _, _, _, ieeg_dir, _, _ = create_bids_structure(bids_root, subject_id, session)
    
    base_name = f"sub-{subject_id}_ses-{session}_task-{task}"
    
    # Get sampling rate from settings
    sampling_rate = float(df_settings['left_sr'].iloc[0]) if 'left_sr' in df_settings.columns else 250.0
    
    # 1. Prepare data arrays
    print(f"\n[1] Preparing LFP data arrays...")
    
    time_sec = (df_lfp_sense.index - df_lfp_sense.index[0]).total_seconds().values
    
    # Extract LFP voltage data (in uV)
    lfp_left = df_lfp_sense['voltage_left'].values if 'voltage_left' in df_lfp_sense.columns else np.zeros(len(df_lfp_sense))
    lfp_right = df_lfp_sense['voltage_right'].values if 'voltage_right' in df_lfp_sense.columns else np.zeros(len(df_lfp_sense))
    
    # Extract stimulation amplitudes (in mA)
    stim_subset = df_stim_aligned.loc[df_lfp_sense.index]
    stim_left = stim_subset['stim_amp_left'].values
    stim_right = stim_subset['stim_amp_right'].values
    
    print(f"    OK Data shape: {len(time_sec)} samples")
    print(f"    OK Sampling rate: {sampling_rate} Hz")
    print(f"    OK Duration: {time_sec[-1]:.2f} seconds")
    
    # 2. Save as FIF in derivatives (MNE-Python native format - for preprocessing)
    if save_fif:
        print(f"\n[2] Writing LFP data as MNE FIF format (derivatives)...")
        
        # Prepare data matrix (channels x samples)
        # Convert uV to V for MNE (MNE uses SI units: Volts, Amperes)
        data_matrix = np.array([
            lfp_left * 1e-6,   # LFP_L: uV to V
            lfp_right * 1e-6,  # LFP_R: uV to V
            stim_left * 1e-3,  # STIM_L: mA to A
            stim_right * 1e-3  # STIM_R: mA to A
        ])
        
        # Create channel info
        ch_names = ['LFP_L', 'LFP_R', 'STIM_L', 'STIM_R']
        ch_types = ['seeg', 'seeg', 'stim', 'stim']  # SEEG for LFP, STIM for stimulation
        
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sampling_rate,
            ch_types=ch_types
        )
        
        # Create Raw object first
        raw = mne.io.RawArray(data_matrix, info)
        
        # Add recording date (must be UTC timezone-aware)
        from datetime import timezone
        raw.set_meas_date(datetime.now(timezone.utc))
        
        # Add device info
        raw.info['device_info'] = {
            'type': 'Medtronic Percept PC',
            'model': 'DBS system with sensing capability'
        }
        
        # Save as FIF in derivatives
        fif_path = deriv_ieeg_dir / f"{base_name}_ieeg.fif"
        raw.save(fif_path, overwrite=True, verbose=False)
        
        print(f"    OK Saved FIF: derivatives/mne-python/{fif_path.relative_to(bids_root / 'derivatives' / 'mne-python')}")
        print(f"    OK Format: MNE-Python FIF (binary, efficient)")
        print(f"    OK Channels: {len(ch_names)}")
        print(f"    OK Ready for preprocessing pipeline!")
    
    # 3. Save as TSV in raw BIDS (optional, for human readability and archival)
    if save_tsv:
        print(f"\n[3] Writing LFP time series (TSV format - raw BIDS)...")
        
        ieeg_data = pd.DataFrame({
            'time': time_sec,
            'LFP_L': lfp_left,
            'LFP_R': lfp_right,
            'STIM_L': stim_left,
            'STIM_R': stim_right
        })
        
        ieeg_tsv_path = ieeg_dir / f"{base_name}_ieeg.tsv"
        ieeg_data.to_csv(ieeg_tsv_path, sep='\t', index=False, float_format='%.6f')
        print(f"    OK Saved TSV: {ieeg_tsv_path.relative_to(bids_root)}")
        print(f"    OK Format: TSV (text, human-readable)")
    
    # 4. Save metadata (JSON)
    print(f"\n[4] Writing metadata...")
    
    ieeg_metadata = {
        "TaskName": task,
        "SamplingFrequency": sampling_rate,
        "PowerLineFrequency": 50,
        "Manufacturer": "Medtronic",
        "ManufacturersModelName": "Percept PC",
        "iEEGReference": "bipolar",
        "ElectricalStimulation": True,
        "ElectricalStimulationParameters": {
            "Frequency": stim_freq,
            "PulseWidth": "60 microseconds"
        },
        "RecordingDuration": float(time_sec[-1]),
        "DataFormats": []
    }
    
    if save_fif:
        ieeg_metadata["DataFormats"].append("FIF (MNE-Python native, in derivatives/ for signal processing)")
    if save_tsv:
        ieeg_metadata["DataFormats"].append("TSV (plain text, in raw BIDS for archival)")
    if save_fif:
        ieeg_metadata["DerivativeFile"] = f"derivatives/mne-python/{fif_path.relative_to(bids_root / 'derivatives' / 'mne-python')}"
    
    ieeg_json_path = ieeg_dir / f"{base_name}_ieeg.json"
    with open(ieeg_json_path, 'w') as f:
        json.dump(ieeg_metadata, f, indent=2)
    print(f"    OK Saved: {ieeg_json_path.name}")
    
    # 5. Save channel information
    print(f"\n[5] Writing channel information...")
    
    left_ch = df_settings['left_ch'].iloc[0] if 'left_ch' in df_settings.columns else 'unknown'
    right_ch = df_settings['right_ch'].iloc[0] if 'right_ch' in df_settings.columns else 'unknown'
    
    channels_data = pd.DataFrame({
        'name': ['LFP_L', 'LFP_R', 'STIM_L', 'STIM_R'],
        'type': ['ECOG', 'ECOG', 'TRIG', 'TRIG'],
        'units': ['uV', 'uV', 'mA', 'mA'],
        'low_cutoff': [0.5, 0.5, 'n/a', 'n/a'],
        'high_cutoff': [100, 100, 'n/a', 'n/a'],
        'description': [
            f'Left STN LFP, contacts {left_ch}',
            f'Right STN LFP, contacts {right_ch}',
            'Left stimulation amplitude',
            'Right stimulation amplitude'
        ],
        'sampling_frequency': [sampling_rate] * 4,
        'status': ['good'] * 4
    })
    
    channels_tsv_path = ieeg_dir / f"{base_name}_channels.tsv"
    channels_data.to_csv(channels_tsv_path, sep='\t', index=False)
    print(f"    OK Saved: {channels_tsv_path.name}")
    
    # 6. Save stimulation events
    print(f"\n[6] Writing stimulation events...")
    
    stim_l_on = stim_subset['stim_amp_left'].values > 0
    stim_r_on = stim_subset['stim_amp_right'].values > 0
    
    def find_transitions(stim_on, time_sec):
        """Find stimulation on/off transitions"""
        transitions = np.diff(np.concatenate([[False], stim_on, [False]]).astype(int))
        onsets = np.where(transitions == 1)[0]
        offsets = np.where(transitions == -1)[0]
        
        events = []
        for onset, offset in zip(onsets, offsets):
            events.append({
                'onset': time_sec[onset],
                'duration': time_sec[offset-1] - time_sec[onset]
            })
        return events
    
    events_l = find_transitions(stim_l_on, time_sec)
    events_r = find_transitions(stim_r_on, time_sec)
    
    events_list = []
    for event in events_l:
        events_list.append({
            'onset': event['onset'],
            'duration': event['duration'],
            'trial_type': 'stim_on_left',
            'value': 3.0
        })
    
    for event in events_r:
        events_list.append({
            'onset': event['onset'],
            'duration': event['duration'],
            'trial_type': 'stim_on_right',
            'value': 3.5
        })
    
    if events_list:
        events_df = pd.DataFrame(events_list).sort_values('onset')
        events_tsv_path = ieeg_dir / f"{base_name}_events.tsv"
        events_df.to_csv(events_tsv_path, sep='\t', index=False, float_format='%.6f')
        print(f"    OK Saved: {events_tsv_path.name}")
    
    print(f"\n>> LFP data export complete!")
    
    return ieeg_dir


# ============================================================
# Dataset-Level Files
# ============================================================

def create_dataset_description(bids_root, dataset_name="DBS-EEG-LFP"):
    """Create dataset_description.json for raw BIDS data."""
    dataset_description = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "Authors": ["Your Name"],
        "Acknowledgements": "University of Wurzburg"
    }
    
    desc_path = Path(bids_root) / "dataset_description.json"
    with open(desc_path, 'w') as f:
        json.dump(dataset_description, f, indent=2)
    
    print(f"OK Dataset description: {desc_path.name}")


def create_derivatives_description(bids_root, pipeline_name="mne-python"):
    """Create dataset_description.json for derivatives."""
    deriv_root = Path(bids_root) / "derivatives" / pipeline_name
    
    deriv_description = {
        "Name": f"{pipeline_name} preprocessed data",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": pipeline_name,
                "Version": mne.__version__,
                "Description": "MNE-Python FIF format for preprocessing"
            }
        ],
        "SourceDatasets": [
            {
                "URL": "../",
                "Version": "1.0"
            }
        ]
    }
    
    desc_path = deriv_root / "dataset_description.json"
    with open(desc_path, 'w') as f:
        json.dump(deriv_description, f, indent=2)
    
    print(f"OK Derivatives description: derivatives/{pipeline_name}/dataset_description.json")


def create_participants_file(bids_root, subject_id, age=None, sex=None, diagnosis="PD"):
    """Create participants.tsv and participants.json."""
    participants_data = pd.DataFrame({
        'participant_id': [f'sub-{subject_id}'],
        'age': [age if age is not None else 'n/a'],
        'sex': [sex if sex is not None else 'n/a'],
        'diagnosis': [diagnosis]
    })
    
    participants_tsv = Path(bids_root) / "participants.tsv"
    participants_data.to_csv(participants_tsv, sep='\t', index=False)
    
    participants_json_data = {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age of the participant", "Units": "years"},
        "sex": {"Description": "Sex of the participant"},
        "diagnosis": {"Description": "Primary diagnosis"}
    }
    
    participants_json = Path(bids_root) / "participants.json"
    with open(participants_json, 'w') as f:
        json.dump(participants_json_data, f, indent=2)
    
    print(f"OK Participants files: {participants_tsv.name}, {participants_json.name}")


# ============================================================
# Main Export Function - UPDATED WITH DERIVATIVES
# ============================================================

def export_to_bids(eeg_segment, df_lfp_sense, df_stim_aligned, df_settings,
                   subject_id, bids_root, stim_freq=55, 
                   age=None, sex=None, diagnosis="PD",
                   task='rest', session='stim01',
                   save_lfp_fif=True, save_lfp_tsv=True):
    """
    Export synchronized EEG and LFP data to BIDS format.
    
    UPDATED: FIF files now saved in derivatives/ for preprocessing.
    - Raw BIDS: BrainVision (EEG) and TSV (LFP)
    - Derivatives: FIF format for both EEG and LFP
    
    Parameters
    ----------
    eeg_segment : mne.io.Raw
        Synchronized EEG segment
    df_lfp_sense : pd.DataFrame
        LFP sensing data
    df_stim_aligned : pd.DataFrame
        Aligned stimulation data
    df_settings : pd.DataFrame
        LFP settings
    subject_id : str
        Subject identifier
    bids_root : str or Path
        Root directory for BIDS dataset
    stim_freq : float
        Stimulation frequency (Hz)
    age : int, optional
        Subject age
    sex : str, optional
        Subject sex
    diagnosis : str
        Primary diagnosis
    task : str
        Task name
    session : str
        Session identifier
    save_lfp_fif : bool
        Save LFP as FIF format in derivatives/ (recommended for preprocessing)
    save_lfp_tsv : bool
        Save LFP as TSV format in raw BIDS (recommended for archival)
        
    Returns
    -------
    Path
        BIDS root directory
    """
    print("\n" + "="*70)
    print("CREATING BIDS DATASET WITH DERIVATIVES")
    print("="*70)
    
    bids_root = Path(bids_root)
    
    # Create structure (including derivatives)
    print("\n[STEP 1] Creating BIDS directory structure...")
    bids_root, sub_dir, eeg_dir, ieeg_dir, deriv_eeg_dir, deriv_ieeg_dir = \
        create_bids_structure(bids_root, subject_id, session)
    print(f"    OK Raw BIDS structure")
    print(f"    OK Derivatives structure")
    
    # Create dataset-level files
    print("\n[STEP 2] Creating dataset-level metadata...")
    create_dataset_description(bids_root, dataset_name="DBS-Multimodal-Recording")
    create_derivatives_description(bids_root, pipeline_name="mne-python")
    create_participants_file(bids_root, subject_id, age=age, sex=sex, diagnosis=diagnosis)
    
    # Save EEG (BrainVision in raw BIDS, FIF in derivatives)
    print("\n[STEP 3] Saving EEG data...")
    save_eeg_bids(eeg_segment, subject_id, bids_root, deriv_eeg_dir, 
                  stim_freq, task, session)
    
    # Save LFP (TSV in raw BIDS, FIF in derivatives)
    print("\n[STEP 4] Saving LFP data...")
    save_lfp_bids(df_lfp_sense, df_stim_aligned, df_settings, 
                  subject_id, bids_root, deriv_ieeg_dir, stim_freq, task, session,
                  save_fif=save_lfp_fif, save_tsv=save_lfp_tsv)
    
    # Summary
    print("\n" + "="*70)
    print("SUCCESS - BIDS DATASET CREATED")
    print("="*70)
    print(f"\nDataset location: {bids_root}")
    print(f"\nStructure:")
    print(f"  {bids_root.name}/")
    print(f"  ├── sub-{subject_id}/ses-{session}/")
    print(f"  │   ├── eeg/     (BrainVision format)")
    print(f"  │   └── ieeg/    (TSV format + metadata)")
    print(f"  └── derivatives/mne-python/sub-{subject_id}/ses-{session}/")
    print(f"      ├── eeg/     (FIF format with digitization)")
    print(f"      └── ieeg/    (FIF format for preprocessing)")
    
    if save_lfp_fif:
        print(f"\n>> FIF files ready for MNE-Python preprocessing pipeline")
    if save_lfp_tsv:
        print(f">> TSV files available for human inspection")
    
    print(f"\n>> Dataset is BIDS-compliant and ready for analysis!")
    
    return bids_root

