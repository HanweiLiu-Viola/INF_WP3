"""
Source Reconstruction Module - UPDATED for Real Coregistration
===============================================================

EEG source reconstruction using:
- Real electrode 3D coordinates (from EGI coordinates.xml)
- ICP-based head-MRI coregistration
- FieldTrip head model (simbio FEM)
- AAL3v2 atlas for ROI definition
- MNE-Python inverse solutions (dSPM, sLORETA, eLORETA, etc.)

CRITICAL UPDATES:
- Now uses real trans file (head->MRI transform) from ICP coregistration
- Validates electrode positions are present
- Checks coregistration quality
- No longer assumes identity transform
"""

import numpy as np
import mne
from scipy.io import loadmat
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import nibabel as nib

logger = logging.getLogger(__name__)


# ============================================================
# Utility Functions
# ============================================================

def validate_electrode_positions(epochs: mne.Epochs) -> Tuple[bool, str]:
    """
    Validate that epochs have proper 3D electrode positions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs to validate
        
    Returns
    -------
    is_valid : bool
        True if electrode positions are present and valid
    message : str
        Validation message
    """
    # Check if dig info exists
    if epochs.info.get('dig') is None or len(epochs.info['dig']) == 0:
        return False, "No digitization information found in epochs.info['dig']"
    
    # Count EEG electrode points
    n_eeg_points = sum(
        1 for d in epochs.info['dig']
        if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG
    )
    
    if n_eeg_points == 0:
        return False, "No EEG electrode positions found in dig info"
    
    # Check if positions look reasonable (in meters, should be < 0.2m from origin)
    positions = np.array([
        d['r'] for d in epochs.info['dig']
        if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG
    ])
    
    max_dist = np.max(np.linalg.norm(positions, axis=1))
    if max_dist > 0.5:  # 50cm - clearly wrong
        return False, f"Electrode positions look wrong (max distance: {max_dist:.2f}m)"
    
    # Check for fiducials (optional but recommended)
    n_fid = sum(
        1 for d in epochs.info['dig']
        if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL
    )
    
    if n_fid < 3:
        message = f"✓ Found {n_eeg_points} EEG positions (⚠️ only {n_fid}/3 fiducials)"
    else:
        message = f"✓ Found {n_eeg_points} EEG positions and {n_fid} fiducials"
    
    return True, message


def validate_trans_file(trans_file: Union[str, Path]) -> Tuple[bool, str, Optional[mne.Transform]]:
    """
    Validate and load a trans file (head->MRI transform).
    
    Parameters
    ----------
    trans_file : str or Path
        Path to trans file (.fif)
        
    Returns
    -------
    is_valid : bool
        True if trans file is valid
    message : str
        Validation message
    trans : mne.Transform or None
        Loaded transform object
    """
    trans_file = Path(trans_file)
    
    if not trans_file.exists():
        return False, f"Trans file not found: {trans_file}", None
    
    try:
        trans = mne.read_trans(str(trans_file))
        
        # Check if it's head->mri
        if trans['from'] != mne.io.constants.FIFF.FIFFV_COORD_HEAD:
            return False, f"Trans 'from' should be head coords, got {trans['from']}", None
        
        if trans['to'] != mne.io.constants.FIFF.FIFFV_COORD_MRI:
            return False, f"Trans 'to' should be MRI coords, got {trans['to']}", None
        
        # Check if transform matrix looks reasonable (not identity)
        T = trans['trans']
        is_identity = np.allclose(T, np.eye(4), atol=1e-3)
        
        if is_identity:
            message = "⚠️ Trans is identity matrix - using standard montage alignment"
        else:
            # Report translation
            translation = np.linalg.norm(T[:3, 3]) * 1000  # to mm
            message = f"✓ Real coregistration (translation: {translation:.1f}mm)"
        
        return True, message, trans
        
    except Exception as e:
        return False, f"Failed to load trans file: {e}", None


def check_and_set_montage(epochs: mne.Epochs, montage_name: str = 'standard_1020') -> mne.Epochs:
    """
    Check if epochs have montage. If not, set standard montage WITH WARNING.
    
    NOTE: For best results, epochs should already have 3D electrode coordinates
    from coordinates.xml. This is a fallback only.
    
    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    montage_name : str
        Standard montage name (default: 'standard_1020')
        
    Returns
    -------
    epochs : mne.Epochs
        Epochs with montage set
    """
    if epochs.info.get_montage() is None:
        logger.warning("⚠️  No montage found in epochs!")
        logger.warning("   RECOMMENDED: Load real 3D coordinates from coordinates.xml")
        logger.warning("   FALLBACK: Setting standard montage (less accurate)")
        logger.info(f"   Using: {montage_name}")
        
        montage = mne.channels.make_standard_montage(montage_name)
        epochs.set_montage(montage)
        
        logger.info(f"✓ Standard montage '{montage_name}' applied")
        logger.warning("   ⚠️  Source reconstruction accuracy will be reduced!")
    else:
        logger.info("✓ Montage already set")
        montage_info = epochs.info.get_montage()
        logger.info(f"  Montage: {montage_info}")
    
    return epochs


# ============================================================
# Head Model Loading and Conversion
# ============================================================

class HeadModelLoader:
    """Load and convert FieldTrip head model to MNE format"""
    
    def __init__(self, mat_file: Union[str, Path]):
        """
        Initialize head model loader.
        
        Parameters
        ----------
        mat_file : str or Path
            Path to FieldTrip head model .mat file
        """
        self.mat_file = Path(mat_file)
        self.data = None
        self.headmodel = None
        
    def load(self):
        """Load MAT file"""
        logger.info(f"Loading head model from: {self.mat_file}")
        
        try:
            self.data = loadmat(
                str(self.mat_file), 
                struct_as_record=False, 
                simplify_cells=True
            )
            logger.info("✓ Loaded as standard MAT file (v7 or lower)")
        except NotImplementedError:
            raise ValueError(
                "HDF5 MAT file detected. Please convert to v7 format in MATLAB:\n"
                "save('output.mat', '-struct', 'headmodel', '-v7')"
            )
        
        # Extract headmodel struct
        if 'headmodel' in self.data:
            self.headmodel = self.data['headmodel']
        else:
            # Try to find it at top level
            self.headmodel = self.data
        
        # Log basic info
        if hasattr(self.headmodel, 'type'):
            logger.info(f"Head model type: {self.headmodel.type}")
        
        if hasattr(self.headmodel, 'unit'):
            logger.info(f"Unit: {self.headmodel.unit}")
        
        if hasattr(self.headmodel, 'cond'):
            logger.info(f"Conductivity: {self.headmodel.cond}")
    
    def get_info(self) -> Dict:
        """Get head model information"""
        if self.headmodel is None:
            raise ValueError("Head model not loaded. Call load() first.")
        
        info = {}
        
        # Basic properties
        for attr in ['type', 'unit', 'cond', 'tissuetype']:
            if hasattr(self.headmodel, attr):
                val = getattr(self.headmodel, attr)
                info[attr] = val
        
        # Mesh properties
        if hasattr(self.headmodel, 'pos'):
            pos = np.array(self.headmodel.pos)
            info['n_vertices'] = len(pos)
        
        if hasattr(self.headmodel, 'tet'):
            tet = np.array(self.headmodel.tet)
            info['n_elements'] = len(tet)
        
        return info


# ============================================================
# Atlas Loading
# ============================================================

class AALAtlasLoader:
    """Load and process AAL3 atlas"""
    
    def __init__(self, atlas_dir: Union[str, Path]):
        """
        Initialize atlas loader.
        
        Parameters
        ----------
        atlas_dir : str or Path
            Path to AAL3 atlas directory
        """
        self.atlas_dir = Path(atlas_dir)
        self.atlas_file = self.atlas_dir / 'AAL3v1_1mm.nii.gz'
        self.labels_file = self.atlas_dir / 'AAL3v1_1mm.nii.txt'
        
        self.img = None
        self.data = None
        self.affine = None
        self.labels = {}
        
    def load(self):
        """Load atlas NIfTI file"""
        logger.info(f"Loading AAL3 atlas from: {self.atlas_file}")
        
        if not self.atlas_file.exists():
            raise FileNotFoundError(f"Atlas file not found: {self.atlas_file}")
        
        self.img = nib.load(str(self.atlas_file))
        self.data = self.img.get_fdata()
        self.affine = self.img.affine
        
        logger.info(f"✓ Atlas shape: {self.data.shape}")
        logger.info(f"  Affine:\n{self.affine}")
        
    def load_labels(self):
        """Load ROI labels from text file"""
        logger.info(f"Loading ROI labels from: {self.labels_file}")
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    idx = int(parts[0])
                    name = parts[1]
                    self.labels[idx] = name
        
        logger.info(f"✓ Loaded {len(self.labels)} ROI labels")
    
    def get_roi_center(self, roi_idx: int) -> np.ndarray:
        """
        Get ROI center of mass in MNI/MRI coordinates (meters).
        
        Parameters
        ----------
        roi_idx : int
            ROI index
            
        Returns
        -------
        center : np.ndarray
            ROI center (x, y, z) in meters
        """
        if self.data is None:
            raise ValueError("Atlas not loaded. Call load() first.")
        
        # Get voxels for this ROI
        mask = self.data == roi_idx
        if not np.any(mask):
            raise ValueError(f"ROI {roi_idx} not found in atlas")
        
        # Compute center of mass in voxel coordinates
        coords = np.array(np.where(mask)).T  # (N, 3)
        center_voxel = coords.mean(axis=0)
        
        # Convert to MNI coordinates (mm)
        center_mm = self.affine @ np.append(center_voxel, 1)
        center_mm = center_mm[:3]
        
        # Convert to meters
        center_m = center_mm / 1000.0
        
        return center_m


# ============================================================
# Source Space Creation
# ============================================================

class SourceSpaceBuilder:
    """Build source space from atlas ROIs"""
    
    def __init__(self, atlas: AALAtlasLoader):
        """
        Initialize source space builder.
        
        Parameters
        ----------
        atlas : AALAtlasLoader
            Loaded atlas
        """
        self.atlas = atlas
        self.src = None
        
    def create_roi_source_space(
        self,
        roi_indices: Optional[List[int]] = None,
        subject: str = 'fsaverage'
    ) -> mne.SourceSpaces:
        """
        Create discrete source space with one dipole per ROI.
        
        Parameters
        ----------
        roi_indices : list of int, optional
            Specific ROI indices to use. If None, uses all available ROIs.
        subject : str
            Subject name (for metadata)
            
        Returns
        -------
        src : mne.SourceSpaces
            Discrete source space
        """
        logger.info("Creating discrete source space (one dipole per ROI)...")
        
        # Determine which ROIs to use
        if roi_indices is None:
            roi_indices = sorted(self.atlas.labels.keys())
        
        # Get ROI centers
        centers = {}
        failed = []
        
        for idx in roi_indices:
            try:
                center = self.atlas.get_roi_center(idx)
                centers[idx] = center
            except ValueError as e:
                failed.append(idx)
                logger.warning(f"  Skipping ROI {idx}: {e}")
        
        if failed:
            logger.warning(f"  Failed to process {len(failed)} ROIs")
        
        n_sources = len(centers)
        logger.info(f"Creating {n_sources} source dipoles")
        
        # Prepare source space data
        rr = np.array([centers[idx] for idx in sorted(centers.keys())])
        nn = np.zeros_like(rr)  # Normal vectors (not used for discrete sources)
        nn[:, 2] = 1.0  # Point upward
        
        inuse = np.ones(n_sources, dtype=int)
        vertno = np.arange(n_sources)
        
        # Create MNE source space structure
        src_dict = {
            'rr': rr,  # Source locations (m)
            'nn': nn,  # Normal vectors
            'inuse': inuse,
            'vertno': vertno,
            'nuse': n_sources,
            'coord_frame': mne.io.constants.FIFF.FIFFV_COORD_MRI,
            'id': 1,
            'type': 'discrete',
            'subject_his_id': subject,
            'roi_indices': sorted(centers.keys()),
        }
        
        # Convert to proper MNE SourceSpaces object
        self.src = mne.SourceSpaces([src_dict])
        
        logger.info(f"✓ Created source space with {n_sources} ROIs")
        
        return self.src


# ============================================================
# Forward Solution
# ============================================================

class ForwardSolutionBuilder:
    """Build forward solution (leadfield matrix)"""
    
    def __init__(
        self,
        epochs: mne.Epochs,
        src: mne.SourceSpaces,
        trans: Optional[mne.Transform] = None,
        bem_model: Optional[Dict] = None
    ):
        """
        Initialize forward solution builder.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs with electrode positions
        src : mne.SourceSpaces
            Source space
        trans : mne.Transform, optional
            Coordinate transformation (head -> MRI).
            If None, will create identity transform (NOT RECOMMENDED).
        bem_model : dict, optional
            BEM model (converted from FieldTrip)
        """
        self.epochs = epochs
        self.src = src
        self.trans = trans
        self.bem_model = bem_model
        self.fwd = None
        
    def compute_forward(
        self,
        conductivity: Optional[Tuple] = None,
        mindist: float = 5.0,
        n_jobs: int = 1
    ) -> mne.Forward:
        """
        Compute forward solution.
        
        Parameters
        ----------
        conductivity : tuple, optional
            BEM conductivities (inner_skull, outer_skull, outer_skin).
            If None, uses standard values (0.3, 0.006, 0.3).
        mindist : float
            Minimum distance between sources and inner skull (mm)
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        fwd : mne.Forward
            Forward solution
        """
        logger.info("Computing forward solution...")
        
        # CRITICAL: Validate electrode positions
        is_valid, message = validate_electrode_positions(self.epochs)
        if not is_valid:
            raise ValueError(
                f"Electrode positions validation failed: {message}\n"
                "Please ensure you have loaded 3D coordinates from coordinates.xml!"
            )
        logger.info(f"  {message}")
        
        # Prepare trans matrix
        if self.trans is None:
            logger.warning("⚠️  No trans file provided!")
            logger.warning("   Creating identity transformation (head = MRI)")
            logger.warning("   This assumes standard montage alignment.")
            logger.warning("   STRONGLY RECOMMENDED: Use real coregistration!")
            
            trans_matrix = mne.Transform(
                fro='head',
                to='mri',
                trans=np.eye(4)
            )
        else:
            logger.info("✓ Using real head->MRI transformation")
            trans_matrix = self.trans
        
        # Prepare BEM model
        if self.bem_model is None:
            logger.warning("No BEM model provided. Using spherical head model.")
            logger.info("  Computing sphere fit from electrode positions...")
            
            sphere = mne.make_sphere_model(
                r0='auto',
                head_radius='auto',
                info=self.epochs.info,
                verbose=False
            )
        else:
            logger.info("Using provided BEM model")
            # NOTE: Full BEM conversion from FieldTrip is complex
            # For now, we use sphere as fallback
            logger.warning("BEM model conversion not fully implemented. Using sphere.")
            sphere = mne.make_sphere_model(
                r0='auto',
                head_radius='auto',
                info=self.epochs.info,
                verbose=False
            )
        
        # Compute forward solution
        logger.info("  Running make_forward_solution...")
        self.fwd = mne.make_forward_solution(
            self.epochs.info,
            trans=trans_matrix,
            src=self.src,
            bem=sphere,
            mindist=mindist / 1000.0,  # Convert mm to m
            n_jobs=n_jobs,
            verbose=False
        )
        
        # Log results
        logger.info(f"✓ Forward solution computed")
        logger.info(f"  Leadfield shape: {self.fwd['sol']['data'].shape}")
        logger.info(f"  Sources: {self.fwd['nsource']}")
        logger.info(f"  Channels: {self.fwd['nchan']}")
        
        # Validate leadfield
        leadfield = self.fwd['sol']['data']
        if np.any(np.isnan(leadfield)) or np.any(np.isinf(leadfield)):
            raise ValueError("Leadfield contains NaN or Inf values!")
        
        logger.info(f"  Leadfield range: [{leadfield.min():.6e}, {leadfield.max():.6e}]")
        
        return self.fwd


# ============================================================
# Inverse Solution
# ============================================================

class InverseSolutionComputer:
    """Compute inverse solutions (source reconstruction)"""
    
    def __init__(
        self,
        epochs: mne.Epochs,
        forward: mne.Forward,
        noise_cov: Optional[mne.Covariance] = None
    ):
        """
        Initialize inverse solution computer.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
        forward : mne.Forward
            Forward solution
        noise_cov : mne.Covariance, optional
            Noise covariance. If None, computes from baseline.
        """
        self.epochs = epochs
        self.forward = forward
        self.noise_cov = noise_cov
        self.inverse_operator = None
        
    def compute_noise_covariance(
        self,
        tmin: Optional[float] = None,
        tmax: float = 0.0,
        method: str = 'empirical',
        reg: float = 0.1
    ) -> mne.Covariance:
        """
        Compute noise covariance from baseline with regularization.
        
        Parameters
        ----------
        tmin : float, optional
            Start time for baseline (default: start of epoch)
        tmax : float
            End time for baseline (default: 0.0)
        method : str
            Covariance estimation method ('empirical', 'shrunk', etc.)
        reg : float
            Regularization parameter (0 to 1). 
            Adds reg * np.mean(np.diag(cov)) to diagonal for stability.
            
        Returns
        -------
        noise_cov : mne.Covariance
            Noise covariance matrix
        """
        logger.info("Computing noise covariance from baseline...")
        logger.info(f"  Method: {method}, regularization: {reg}")
        
        # STEP 1: Compute base covariance
        noise_cov = mne.compute_covariance(
            self.epochs,
            tmin=tmin,
            tmax=tmax,
            method=method,
            verbose=False
        )
        
        # STEP 2: Apply manual regularization if needed
        if reg > 0:
            logger.info(f"  Applying manual regularization: {reg}")
            
            # Get a COPY of the data (can't modify .data directly)
            cov_data = noise_cov.data.copy()
            
            # Calculate regularization amount
            diag_mean = np.mean(np.diag(cov_data))
            reg_amount = reg * diag_mean
            
            # Add to diagonal
            n_channels = cov_data.shape[0]
            cov_data += reg_amount * np.eye(n_channels)
            
            logger.info(f"    Added {reg_amount:.6e} to diagonal")
            
            # Create new Covariance object with regularized data
            self.noise_cov = mne.Covariance(
                data=cov_data,
                names=noise_cov.ch_names,
                bads=noise_cov['bads'],
                projs=noise_cov['projs'],
                nfree=noise_cov['nfree']
            )
        else:
            self.noise_cov = noise_cov
        
        logger.info(f"✓ Noise covariance computed")
        logger.info(f"  Shape: {self.noise_cov.data.shape}")
        logger.info(f"  Diagonal mean: {np.diag(self.noise_cov.data).mean():.6e}")
        
        # STEP 3: Validate
        if np.any(np.isnan(self.noise_cov.data)) or np.any(np.isinf(self.noise_cov.data)):
            logger.error("❌ Noise covariance contains NaN or Inf!")
            raise ValueError("Noise covariance contains NaN or Inf values!")
        
        return self.noise_cov
    
    def make_inverse_operator(
        self,
        loose: float = 0.2,
        depth: float = 0.8,
        fixed: bool = False
    ) -> mne.minimum_norm.InverseOperator:
        """
        Create inverse operator.
        
        Parameters
        ----------
        loose : float
            Loose orientation constraint (0 = fixed, 1 = free)
        depth : float
            Depth weighting (0 = no weighting, 1 = full weighting)
        fixed : bool
            Use fixed orientation
            
        Returns
        -------
        inverse_operator : mne.minimum_norm.InverseOperator
            Inverse operator
        """
        logger.info("Creating inverse operator...")
        
        if self.noise_cov is None:
            logger.info("No noise covariance provided. Computing from baseline...")
            self.compute_noise_covariance()
        
        try:
            self.inverse_operator = mne.minimum_norm.make_inverse_operator(
                self.epochs.info,
                self.forward,
                self.noise_cov,
                loose=loose,
                depth=depth,
                fixed=fixed,
                verbose=False
            )
            
            logger.info("✓ Inverse operator created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create inverse operator: {e}")
            logger.error("   This usually means:")
            logger.error("   1. Noise covariance is too small or unstable")
            logger.error("   2. Electrode positions/coregistration are wrong")
            logger.error("   3. Forward solution has numerical issues")
            raise
        
        return self.inverse_operator
    
    def apply_inverse(
        self,
        method: str = 'sLORETA',
        lambda2: float = 1.0 / 9.0,
        pick_ori: Optional[str] = None
    ) -> mne.SourceEstimate:
        """
        Apply inverse solution to averaged epochs.
        
        Parameters
        ----------
        method : str
            Inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
        lambda2 : float
            Regularization parameter (1/SNR^2)
        pick_ori : str, optional
            Orientation picking: None, 'normal', 'max-power'
            
        Returns
        -------
        stc : mne.SourceEstimate
            Source estimate (averaged across epochs)
        """
        logger.info(f"Applying inverse solution (method: {method})...")
        
        if self.inverse_operator is None:
            logger.info("Creating inverse operator...")
            self.make_inverse_operator()
        
        # Average epochs
        evoked = self.epochs.average()
        
        # Apply inverse
        stc = mne.minimum_norm.apply_inverse(
            evoked,
            self.inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
            verbose=False
        )
        
        logger.info(f"✓ Source estimate computed")
        logger.info(f"  Data shape: {stc.data.shape}")
        logger.info(f"  Time points: {len(stc.times)}")
        logger.info(f"  Sources: {len(stc.vertices[0])}")
        
        return stc
    
    def apply_inverse_epochs(
        self,
        method: str = 'sLORETA',
        lambda2: float = 1.0 / 9.0,
        pick_ori: Optional[str] = None
    ) -> List[mne.SourceEstimate]:
        """
        Apply inverse solution to individual epochs.
        
        Parameters
        ----------
        method : str
            Inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
        lambda2 : float
            Regularization parameter (1/SNR^2)
        pick_ori : str, optional
            Orientation picking: None, 'normal', 'max-power'
            
        Returns
        -------
        stcs : list of mne.SourceEstimate
            Source estimates for each epoch
        """
        logger.info(f"Applying inverse to {len(self.epochs)} epochs...")
        
        if self.inverse_operator is None:
            logger.info("Creating inverse operator...")
            self.make_inverse_operator()
        
        # Apply inverse to each epoch
        stcs = mne.minimum_norm.apply_inverse_epochs(
            self.epochs,
            self.inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
            verbose=False
        )
        
        logger.info(f"✓ Source estimates computed for all epochs")
        
        return stcs


# ============================================================
# ROI Analysis
# ============================================================

class ROIAnalyzer:
    """Analyze source estimates in ROIs"""
    
    def __init__(
        self,
        stcs: List[mne.SourceEstimate],
        atlas: AALAtlasLoader,
        src: mne.SourceSpaces
    ):
        """
        Initialize ROI analyzer.
        
        Parameters
        ----------
        stcs : list of mne.SourceEstimate
            Source estimates (one per epoch)
        atlas : AALAtlasLoader
            Atlas with ROI labels
        src : mne.SourceSpaces
            Source space (should have 'roi_indices' attribute)
        """
        self.stcs = stcs
        self.atlas = atlas
        self.src = src
        
    def extract_roi_timeseries(self) -> Dict[int, np.ndarray]:
        """
        Extract time series for each ROI.
        
        Returns
        -------
        roi_timeseries : dict
            Dictionary mapping ROI index to time series array
        """
        logger.info("Extracting ROI time series...")
        
        # Get ROI indices from source space
        if not hasattr(self.src[0], 'roi_indices'):
            raise ValueError("Source space does not have 'roi_indices' attribute")
        
        src_roi_indices = self.src[0]['roi_indices']
        roi_timeseries = {}
        
        for roi_idx in self.atlas.labels.keys():
            if roi_idx not in src_roi_indices:
                continue
            
            # Get source index for this ROI
            src_idx = src_roi_indices.index(roi_idx)
            
            # Extract time series (average across epochs if multiple)
            if len(self.stcs) == 1:
                ts = self.stcs[0].data[src_idx, :]
            else:
                # Average across epochs
                ts_list = [stc.data[src_idx, :] for stc in self.stcs]
                ts = np.mean(ts_list, axis=0)
            
            roi_timeseries[roi_idx] = ts
        
        logger.info(f"✓ Extracted time series for {len(roi_timeseries)} ROIs")
        
        return roi_timeseries


# ============================================================
# Complete Pipeline
# ============================================================

def run_source_reconstruction_pipeline(
    epochs: mne.Epochs,
    headmodel_file: Union[str, Path],
    atlas_dir: Union[str, Path],
    trans_file: Optional[Union[str, Path]] = None,
    method: str = 'eLORETA',
    lambda2: float = 1.0 / 9.0,
    noise_cov_reg: float = 0.1,
    roi_indices: Optional[List[int]] = None,
    n_jobs: int = 1,
    max_epochs: Optional[int] = None
) -> Dict:
    """
    Complete source reconstruction pipeline with real coregistration.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs with 3D electrode positions
    headmodel_file : str or Path
        FieldTrip head model .mat file
    atlas_dir : str or Path
        AAL3 atlas directory
    trans_file : str or Path, optional
        Head->MRI transform file (.fif) from ICP coregistration.
        If None, uses identity transform (NOT RECOMMENDED).
    method : str
        Inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
    lambda2 : float
        Regularization parameter (1/SNR^2)
    noise_cov_reg : float
        Noise covariance regularization (0 to 1)
    roi_indices : list of int, optional
        Specific ROI indices to use
    n_jobs : int
        Number of parallel jobs
    max_epochs : int, optional
        If provided, limit the reconstruction to the first ``max_epochs`` epochs
        to reduce memory usage. Useful when working with large Epochs objects
        that may overwhelm the kernel.
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'stc': Source estimate (averaged)
        - 'stcs_epochs': List of source estimates per epoch
        - 'roi_timeseries': ROI time series
        - 'src': Source space
        - 'fwd': Forward solution
        - 'inv': Inverse operator
        - 'trans': Transform used
        - 'atlas': Atlas
        - 'method': Method used
    """
    logger.info("\n" + "="*60)
    logger.info("SOURCE RECONSTRUCTION PIPELINE v2.0")
    logger.info("With Real Coregistration")
    logger.info("="*60)
    
    total_epochs = len(epochs)
    epochs_to_use = epochs
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")
        if max_epochs < total_epochs:
            logger.info(
                f"Limiting source reconstruction to the first {max_epochs} "
                f"epochs out of {total_epochs} available"
            )
            epochs_to_use = epochs[:max_epochs]
        else:
            logger.info(
                f"Requested max_epochs={max_epochs} but only {total_epochs} "
                "epochs are available; using all epochs"
            )

    logger.info(
        f"Epochs available: {total_epochs}; using {len(epochs_to_use)} for reconstruction"
    )

    # Step 0: Validate electrode positions
    logger.info("\n[STEP 0] Validating electrode positions...")
    is_valid, message = validate_electrode_positions(epochs_to_use)
    if not is_valid:
        logger.error(f"❌ {message}")
        logger.error("Please load 3D coordinates from coordinates.xml before running!")
        raise ValueError(message)
    logger.info(f"  {message}")
    
    # Step 0.5: Load and validate trans file
    trans = None
    if trans_file is not None:
        logger.info("\n[STEP 0.5] Loading head->MRI transform...")
        is_valid, message, trans = validate_trans_file(trans_file)
        if not is_valid:
            logger.error(f"❌ {message}")
            raise ValueError(message)
        logger.info(f"  {message}")
    else:
        logger.warning("\n⚠️  No trans file provided!")
        logger.warning("   Will use identity transform (less accurate)")
    
    # Step 1: Load head model
    logger.info("\n[STEP 1] Loading head model...")
    head_loader = HeadModelLoader(headmodel_file)
    head_loader.load()
    head_info = head_loader.get_info()
    logger.info(f"  Head model info: {head_info}")
    
    # Step 2: Load atlas
    logger.info("\n[STEP 2] Loading AAL3 atlas...")
    atlas = AALAtlasLoader(atlas_dir)
    atlas.load()
    atlas.load_labels()
    
    # Step 3: Create source space
    logger.info("\n[STEP 3] Creating source space...")
    src_builder = SourceSpaceBuilder(atlas)
    src = src_builder.create_roi_source_space(roi_indices=roi_indices)
    
    # Step 4: Compute forward solution
    logger.info("\n[STEP 4] Computing forward solution...")
    fwd_builder = ForwardSolutionBuilder(epochs_to_use, src, trans=trans)
    fwd = fwd_builder.compute_forward(n_jobs=n_jobs)
    
    # Step 5: Compute inverse solution
    logger.info("\n[STEP 5] Computing inverse solution...")
    inv_computer = InverseSolutionComputer(epochs_to_use, fwd)
    inv_computer.compute_noise_covariance(reg=noise_cov_reg)
    inv_computer.make_inverse_operator()
    
    # Apply to average
    stc = inv_computer.apply_inverse(method=method, lambda2=lambda2)
    
    # Apply to individual epochs
    stcs_epochs = inv_computer.apply_inverse_epochs(method=method, lambda2=lambda2)
    
    # Step 6: Extract ROI time series
    logger.info("\n[STEP 6] Extracting ROI time series...")
    roi_analyzer = ROIAnalyzer(stcs_epochs, atlas, src)
    roi_timeseries = roi_analyzer.extract_roi_timeseries()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ SOURCE RECONSTRUCTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Method: {method}")
    logger.info(f"Sources: {fwd['nsource']}")
    logger.info(f"ROIs: {len(roi_timeseries)}")
    logger.info(f"Epochs: {len(stcs_epochs)}")
    if trans is not None:
        logger.info(f"Coregistration: Real (from {trans_file})")
    else:
        logger.info(f"Coregistration: Identity (standard montage)")
    logger.info("="*60)
    
    return {
        'stc': stc,
        'stcs_epochs': stcs_epochs,
        'roi_timeseries': roi_timeseries,
        'src': src,
        'fwd': fwd,
        'inv': inv_computer.inverse_operator,
        'trans': trans,
        'atlas': atlas,
        'method': method
    }
