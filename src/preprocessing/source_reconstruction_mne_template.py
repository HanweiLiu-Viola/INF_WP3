"""
Source Reconstruction Module - Using MNE Templates
===================================================

EEG source reconstruction using:
- MNE-Python built-in fsaverage template
- Standard parcellations (Desikan-Killiany, Destrieux, etc.)
- Simplified workflow without external dependencies

Author: Neural Signal Processing Lab
Date: 2025-11-04
"""

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================
# Initialization and Validation
# ============================================================

def ensure_fsaverage(verbose: bool = False) -> Path:
    """
    Ensure fsaverage template is available.
    
    Parameters
    ----------
    verbose : bool
        Print download progress
        
    Returns
    -------
    subjects_dir : Path
        Path to FreeSurfer subjects directory containing fsaverage
    """
    logger.info("Checking for fsaverage template...")
    
    # Fetch fsaverage (downloads if needed)
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = Path(fs_dir)
    
    # Handle different return formats
    if subjects_dir.name == 'fsaverage':
        subjects_dir = subjects_dir.parent
    
    # Verify fsaverage exists
    fsaverage_path = subjects_dir / 'fsaverage'
    if not fsaverage_path.exists():
        raise FileNotFoundError(
            f"fsaverage not found at {fsaverage_path}. "
            f"Download may have failed. Please try manually: "
            f"python -c 'import mne; mne.datasets.fetch_fsaverage()'"
        )
    
    # Verify critical files exist
    surf_dir = fsaverage_path / 'surf'
    if not surf_dir.exists():
        raise FileNotFoundError(
            f"fsaverage surf directory not found at {surf_dir}"
        )
    
    required_files = ['lh.white', 'rh.white', 'lh.sphere', 'rh.sphere']
    missing_files = [f for f in required_files if not (surf_dir / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"fsaverage is incomplete. Missing files: {missing_files}\n"
            f"Please re-download: python -c 'import mne; mne.datasets.fetch_fsaverage()'"
        )
    
    logger.info(f"✓ fsaverage found at {fsaverage_path}")
    return subjects_dir



# ============================================================
# Utility Functions
# ============================================================

def check_and_set_montage(epochs: mne.Epochs, montage_name: str = 'standard_1020') -> mne.Epochs:
    """
    Check if epochs have montage, and set standard montage if not.
    
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
        logger.warning("No montage found in epochs. Setting standard montage...")
        logger.info(f"Using: {montage_name}")
        
        montage = mne.channels.make_standard_montage(montage_name)
        epochs.set_montage(montage)
        
        logger.info(f"✓ Standard montage '{montage_name}' applied")
    else:
        logger.info("✓ Montage already set")
        montage_info = epochs.info.get_montage()
        logger.info(f"  Montage: {montage_info}")
    
    return epochs


# ============================================================
# Source Space Creation with MNE Template
# ============================================================

class TemplateSourceSpaceBuilder:
    """Create source space using MNE fsaverage template"""
    
    def __init__(
        self, 
        subjects_dir: Optional[Union[str, Path]] = None,
        spacing: str = 'oct6'
    ):
        """
        Initialize template source space builder.
        
        Parameters
        ----------
        subjects_dir : str or Path, optional
            FreeSurfer subjects directory. If None, uses MNE default.
        spacing : str
            Source space spacing: 'ico4', 'ico5', 'oct5', 'oct6'
            - 'oct6': ~4098 sources per hemisphere (recommended)
            - 'ico4': ~2562 sources per hemisphere (faster)
            - 'ico5': ~10242 sources per hemisphere (high-res)
        """
        # Ensure fsaverage is available
        if subjects_dir is None:
            logger.info("Fetching fsaverage template...")
            subjects_dir = ensure_fsaverage(verbose=False)
        
        self.subjects_dir = Path(subjects_dir)
        self.spacing = spacing
        self.src = None
        
        logger.info(f"Using subjects_dir: {self.subjects_dir}")
        logger.info(f"Source spacing: {spacing}")
    
    def create_surface_source_space(self) -> mne.SourceSpaces:
        """
        Create cortical surface source space.
        
        Returns
        -------
        src : mne.SourceSpaces
            Source space object
        """
        logger.info(f"Creating source space with spacing '{self.spacing}'...")
        
        # Create source space using fsaverage
        self.src = mne.setup_source_space(
            subject='fsaverage',
            spacing=self.spacing,
            subjects_dir=self.subjects_dir,
            add_dist=False,  # Don't compute distances (faster)
            verbose=False
        )
        
        n_sources = sum([s['nuse'] for s in self.src])
        logger.info(f"✓ Created source space:")
        logger.info(f"  Total sources: {n_sources}")
        logger.info(f"  Left hemisphere: {self.src[0]['nuse']}")
        logger.info(f"  Right hemisphere: {self.src[1]['nuse']}")
        
        return self.src
    
    def create_volume_source_space(
        self,
        pos: float = 5.0,
        bem: Optional[mne.bem.ConductorModel] = None
    ) -> mne.SourceSpaces:
        """
        Create volume source space (alternative to surface).
        
        Parameters
        ----------
        pos : float
            Grid spacing in mm (default: 5.0)
        bem : mne.bem.ConductorModel, optional
            BEM model for defining volume bounds
            
        Returns
        -------
        src : mne.SourceSpaces
            Volume source space
        """
        logger.info(f"Creating volume source space (grid: {pos}mm)...")
        
        self.src = mne.setup_volume_source_space(
            subject='fsaverage',
            pos=pos,
            subjects_dir=self.subjects_dir,
            bem=bem,
            verbose=False
        )
        
        logger.info(f"✓ Volume source space created: {self.src[0]['nuse']} sources")
        
        return self.src


# ============================================================
# Forward Solution with Template
# ============================================================

class TemplateForwardSolutionBuilder:
    """Compute forward solution using template head model"""
    
    def __init__(
        self,
        epochs: mne.Epochs,
        src: mne.SourceSpaces,
        subjects_dir: Union[str, Path]
    ):
        """
        Initialize forward solution builder.
        
        Parameters
        ----------
        epochs : mne.Epochs
            EEG epochs with montage set
        src : mne.SourceSpaces
            Source space
        subjects_dir : str or Path
            FreeSurfer subjects directory
        """
        self.epochs = epochs
        self.src = src
        self.subjects_dir = Path(subjects_dir)
        self.fwd = None
        self.bem = None
    
    def create_bem_model(
        self,
        conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3)
    ) -> mne.bem.ConductorModel:
        """
        Create BEM model using template.
        
        Parameters
        ----------
        conductivity : tuple of float
            Conductivity values for (brain, skull, scalp)
            Default: (0.3, 0.006, 0.3) S/m
            
        Returns
        -------
        bem : mne.bem.ConductorModel
            BEM model
        """
        logger.info("Creating BEM model from template...")
        logger.info(f"Conductivity (brain, skull, scalp): {conductivity}")
        
        # Create BEM model using fsaverage template
        # This uses the template surfaces
        self.bem = mne.make_bem_model(
            subject='fsaverage',
            ico=4,  # Decimation level for BEM surfaces
            conductivity=conductivity,
            subjects_dir=self.subjects_dir,
            verbose=False
        )
        
        # Create BEM solution
        bem_sol = mne.make_bem_solution(self.bem, verbose=False)
        
        logger.info("✓ BEM model created")
        logger.info(f"  Surfaces: {len(self.bem)}")
        
        return bem_sol
    
    def create_spherical_model(self) -> mne.bem.ConductorModel:
        """
        Create spherical head model (faster, less accurate).
        
        Returns
        -------
        sphere : mne.bem.ConductorModel
            Spherical conductor model
        """
        logger.info("Creating spherical head model...")
        
        # Fit sphere to electrode positions
        sphere = mne.make_sphere_model(
            r0='auto',
            head_radius='auto',
            info=self.epochs.info,
            verbose=False
        )
        
        logger.info("✓ Spherical model created")
        logger.info(f"  Origin: {sphere['r0']}")
        logger.info(f"  Radii: {sphere['layers']}")
        
        return sphere
    
    def compute_forward(
        self,
        bem: Optional[mne.bem.ConductorModel] = None,
        trans: str = 'fsaverage',
        n_jobs: int = 1
    ) -> mne.Forward:
        """
        Compute forward solution.
        
        Parameters
        ----------
        bem : mne.bem.ConductorModel, optional
            BEM or sphere model. If None, creates spherical model.
        trans : str or mne.transforms.Transform
            Coordinate transformation. Use 'fsaverage' for template.
        n_jobs : int
            Number of parallel jobs
            
        Returns
        -------
        fwd : mne.Forward
            Forward solution
        """
        logger.info("Computing forward solution...")
        
        # Create BEM if not provided
        if bem is None:
            logger.info("No BEM provided, using spherical model...")
            bem = self.create_spherical_model()
        
        # Compute forward solution
        self.fwd = mne.make_forward_solution(
            info=self.epochs.info,
            trans=trans,
            src=self.src,
            bem=bem,
            eeg=True,
            meg=False,
            n_jobs=n_jobs,
            verbose=False
        )
        
        logger.info("✓ Forward solution computed")
        logger.info(f"  Sources: {self.fwd['nsource']}")
        logger.info(f"  Channels: {self.fwd['nchan']}")
        logger.info(f"  Coordinate frame: {self.fwd['coord_frame']}")
        
        return self.fwd


# ============================================================
# Inverse Solution
# ============================================================

class InverseSolutionComputer:
    """Compute inverse solution and apply to data"""
    
    def __init__(
        self,
        epochs: mne.Epochs,
        fwd: mne.Forward
    ):
        """
        Initialize inverse solution computer.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
        fwd : mne.Forward
            Forward solution
        """
        self.epochs = epochs
        self.fwd = fwd
        self.noise_cov = None
        self.inverse_operator = None
        self.noise_cov_strategy = None
    
    def compute_noise_covariance(
        self,
        method: str = 'auto',
        tmin: Optional[float] = None,
        tmax: float = 0.0,
        reg: Union[str, float] = 0.1
    ) -> mne.Covariance:
        """
        Compute noise covariance from baseline or empty room data.
        
        Parameters
        ----------
        method : str
            Covariance estimation method: 'auto', 'empirical', 'diagonal', 'shrunk'
        tmin : float, optional
            Start time for baseline (if None, uses all pre-stimulus)
        tmax : float
            End time for baseline (default: 0.0, i.e., stimulus onset)
        reg : str or float
            Regularization parameter
            
        Returns
        -------
        noise_cov : mne.Covariance
            Noise covariance matrix
        """
        logger.info(f"Computing noise covariance (method: {method})...")
        
        # Try baseline period
        if self.epochs.tmin < tmax:
            logger.info(f"Using baseline period: [{self.epochs.tmin:.3f}, {tmax:.3f}] s")
            
            try:
                self.noise_cov = mne.compute_covariance(
                    self.epochs,
                    tmin=tmin,
                    tmax=tmax,
                    method=method,
                    rank='info',
                    verbose=False
                )
                self.noise_cov_strategy = f'baseline ({method})'
                logger.info(f"✓ Noise covariance computed from baseline")
                
            except Exception as e:
                logger.warning(f"Baseline covariance failed: {e}")
                logger.info("Falling back to diagonal covariance...")
                self.noise_cov = mne.make_ad_hoc_cov(self.epochs.info)
                self.noise_cov_strategy = 'ad-hoc diagonal'
        else:
            logger.info("No baseline period, using diagonal covariance...")
            self.noise_cov = mne.make_ad_hoc_cov(self.epochs.info)
            self.noise_cov_strategy = 'ad-hoc diagonal'
        
        # # Apply regularization if needed
        # if isinstance(reg, float) and reg > 0:
        #     logger.info(f"Applying regularization: {reg}")
        #     # 正确的代码
        #     self.noise_cov = mne.cov.regularize(
        #         self.noise_cov,
        #         self.epochs.info,
        #         mag=reg if isinstance(reg, float) else 0.1,  # ✅ 正确
        #         grad=reg if isinstance(reg, float) else 0.1,
        #         eeg=reg if isinstance(reg, float) else 0.1,
        #         verbose=False
        #     )
        
        logger.info(f"✓ Noise covariance ready (strategy: {self.noise_cov_strategy})")
        
        return self.noise_cov
    
    def make_inverse_operator(
        self,
        loose: float = 0.2,
        depth: float = 0.8
    ) -> mne.minimum_norm.InverseOperator:
        """
        Create inverse operator.
        
        Parameters
        ----------
        loose : float
            Loose orientation constraint (0=fixed, 1=free)
        depth : float
            Depth weighting (0=none, 0.8=default)
            
        Returns
        -------
        inverse_operator : mne.minimum_norm.InverseOperator
            Inverse operator
        """
        logger.info("Creating inverse operator...")
        logger.info(f"  Loose orientation: {loose}")
        logger.info(f"  Depth weighting: {depth}")
        
        if self.noise_cov is None:
            logger.warning("No noise covariance, computing default...")
            self.compute_noise_covariance()
        
        self.inverse_operator = mne.minimum_norm.make_inverse_operator(
            info=self.epochs.info,
            forward=self.fwd,
            noise_cov=self.noise_cov,
            loose=loose,
            depth=depth,
            verbose=False
        )
        
        logger.info("✓ Inverse operator created")
        
        return self.inverse_operator
    
    def apply_inverse(
        self,
        method: str = 'dSPM',
        lambda2: float = 1.0 / 9.0,
        pick_ori: Optional[str] = None
    ) -> mne.SourceEstimate:
        """
        Apply inverse operator to averaged epochs.
        
        Parameters
        ----------
        method : str
            Inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
        lambda2 : float
            Regularization parameter (default: 1/9)
        pick_ori : str, optional
            Orientation selection: None, 'normal', 'vector'
            
        Returns
        -------
        stc : mne.SourceEstimate
            Source estimate
        """
        logger.info(f"Applying inverse solution (method: {method})...")
        
        if self.inverse_operator is None:
            logger.warning("No inverse operator, creating default...")
            self.make_inverse_operator()
        
        # Average epochs
        evoked = self.epochs.average()
        
        # Apply inverse
        stc = mne.minimum_norm.apply_inverse(
            evoked=evoked,
            inverse_operator=self.inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
            verbose=False
        )
        
        logger.info("✓ Inverse solution applied to averaged data")
        logger.info(f"  Time points: {len(stc.times)}")
        logger.info(f"  Time range: [{stc.times[0]:.3f}, {stc.times[-1]:.3f}] s")
        
        return stc
    
    def apply_inverse_epochs(
        self,
        method: str = 'dSPM',
        lambda2: float = 1.0 / 9.0,
        pick_ori: Optional[str] = None,
        max_epochs: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> List[mne.SourceEstimate]:
        """
        Apply inverse operator to individual epochs.
        
        Parameters
        ----------
        method : str
            Inverse method
        lambda2 : float
            Regularization parameter
        pick_ori : str, optional
            Orientation selection
        max_epochs : int, optional
            Maximum number of epochs to process (for memory efficiency)
        random_state : int, optional
            Random seed for epoch selection
            
        Returns
        -------
        stcs : list of mne.SourceEstimate
            Source estimates for each epoch
        """
        logger.info(f"Applying inverse solution to epochs (method: {method})...")
        
        if self.inverse_operator is None:
            self.make_inverse_operator()
        
        # Select subset of epochs if needed
        n_epochs = len(self.epochs)
        if max_epochs is not None and n_epochs > max_epochs:
            logger.info(f"Selecting {max_epochs}/{n_epochs} epochs...")
            rng = np.random.RandomState(random_state)
            epoch_indices = rng.choice(n_epochs, size=max_epochs, replace=False)
            epoch_indices.sort()
            epochs_subset = self.epochs[epoch_indices]
        else:
            epochs_subset = self.epochs
            epoch_indices = np.arange(n_epochs)
        
        # Apply inverse to each epoch
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs=epochs_subset,
            inverse_operator=self.inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
            verbose=False
        )
        
        logger.info(f"✓ Inverse solution applied to {len(stcs)} epochs")
        
        return stcs, epoch_indices


# ============================================================
# ROI Analysis with Parcellations
# ============================================================

class ParcellationROIAnalyzer:
    """Analyze source estimates using brain parcellations"""
    
    def __init__(
        self,
        stcs: Union[mne.SourceEstimate, List[mne.SourceEstimate]],
        subjects_dir: Union[str, Path],
        parc: str = 'aparc'
    ):
        """
        Initialize ROI analyzer.
        
        Parameters
        ----------
        stcs : mne.SourceEstimate or list
            Source estimate(s)
        subjects_dir : str or Path
            FreeSurfer subjects directory
        parc : str
            Parcellation name: 'aparc' (Desikan-Killiany), 'aparc.a2009s' (Destrieux)
        """
        self.stcs = [stcs] if isinstance(stcs, mne.SourceEstimate) else stcs
        self.subjects_dir = Path(subjects_dir)
        self.parc = parc
        self.labels = None
        
        logger.info(f"Using parcellation: {parc}")
    
    def load_labels(self) -> List[mne.Label]:
        """
        Load parcellation labels.
        
        Returns
        -------
        labels : list of mne.Label
            Cortical labels
        """
        logger.info("Loading parcellation labels...")
        
        self.labels = mne.read_labels_from_annot(
            subject='fsaverage',
            parc=self.parc,
            subjects_dir=self.subjects_dir,
            verbose=False
        )
        
        logger.info(f"✓ Loaded {len(self.labels)} labels")
        
        return self.labels
    
    def extract_label_timeseries(
        self,
        mode: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        Extract time series for each label/ROI.
        
        Parameters
        ----------
        mode : str
            Extraction mode: 'mean', 'max', 'pca'
            
        Returns
        -------
        label_ts : dict
            Dictionary mapping label name to time series
        """
        logger.info(f"Extracting label time series (mode: {mode})...")
        
        if self.labels is None:
            self.load_labels()
        
        label_ts = {}
        
        # Extract for averaged stc
        stc_avg = self.stcs[0] if len(self.stcs) == 1 else mne.average_source_estimates(self.stcs)
        
        for label in self.labels:
            # Extract time series for this label
            ts = stc_avg.extract_label_time_course(
                labels=label,
                src=stc_avg.vertices,
                mode=mode
            )
            
            label_ts[label.name] = ts.squeeze()
        
        logger.info(f"✓ Extracted time series for {len(label_ts)} labels")
        
        return label_ts
    
    def extract_label_timeseries_epochs(
        self,
        mode: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        Extract epoch-wise label time series.
        
        Parameters
        ----------
        mode : str
            Extraction mode
            
        Returns
        -------
        label_ts_epochs : dict
            Dictionary mapping label name to (n_epochs, n_times) array
        """
        logger.info("Extracting epoch-wise label time series...")
        
        if self.labels is None:
            self.load_labels()
        
        label_ts_epochs = {}
        
        for label in self.labels:
            # Extract for all epochs
            ts_list = []
            for stc in self.stcs:
                ts = stc.extract_label_time_course(
                    labels=label,
                    src=stc.vertices,
                    mode=mode
                )
                ts_list.append(ts.squeeze())
            
            label_ts_epochs[label.name] = np.array(ts_list)
        
        logger.info(f"✓ Extracted epoch time series for {len(label_ts_epochs)} labels")
        
        return label_ts_epochs


# ============================================================
# Complete Pipeline with MNE Template
# ============================================================

def run_source_reconstruction_pipeline(
    epochs: mne.Epochs,
    method: str = 'dSPM',
    lambda2: float = 1.0 / 9.0,
    spacing: str = 'oct6',
    parc: str = 'aparc',
    subjects_dir: Optional[Union[str, Path]] = None,
    use_bem: bool = False,
    noise_cov_method: str = 'auto',
    noise_cov_reg: Union[str, float] = 0.1,
    max_epochs: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    trans: str = 'fsaverage'
) -> Dict:
    """
    Complete source reconstruction pipeline using MNE template.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs
    method : str
        Inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
    lambda2 : float
        Regularization parameter (default: 1/9)
    spacing : str
        Source space spacing: 'ico4', 'ico5', 'oct5', 'oct6'
    parc : str
        Parcellation: 'aparc', 'aparc.a2009s'
    subjects_dir : str or Path, optional
        FreeSurfer subjects directory (uses MNE default if None)
    use_bem : bool
        Use BEM model (True) or spherical model (False, faster)
    noise_cov_method : str
        Noise covariance method: 'auto', 'empirical', 'diagonal', 'shrunk'
    noise_cov_reg : str or float
        Regularization for noise covariance
    max_epochs : int, optional
        Maximum epochs for source reconstruction
    random_state : int, optional
        Random seed for epoch selection
    n_jobs : int
        Number of parallel jobs
    trans : str or mne.transforms.Transform
        Coordinate transformation (default: 'fsaverage' for template)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'stc': Average source estimate
        - 'stcs_epochs': Per-epoch source estimates
        - 'label_timeseries': Label/ROI time series
        - 'src': Source space
        - 'fwd': Forward solution
        - 'inv': Inverse operator
        - 'labels': Parcellation labels
        - 'subjects_dir': Subjects directory path
        - 'method': Inverse method used
        - 'epoch_indices': Indices of epochs used (if subsampled)
        - 'noise_cov_strategy': Noise covariance strategy
    """
    logger.info("\n" + "="*60)
    logger.info("SOURCE RECONSTRUCTION PIPELINE (MNE Template)")
    logger.info("="*60)
    
    # Step 0: Check and set montage
    logger.info("\n[STEP 0] Checking electrode montage...")
    epochs = check_and_set_montage(epochs)
    
    # Step 1: Fetch/setup fsaverage template
    logger.info("\n[STEP 1] Setting up fsaverage template...")
    if subjects_dir is None:
        subjects_dir = ensure_fsaverage(verbose=False)
    else:
        subjects_dir = Path(subjects_dir)
    
    logger.info(f"Subjects directory: {subjects_dir}")
    
    # Step 2: Create source space
    logger.info("\n[STEP 2] Creating source space...")
    src_builder = TemplateSourceSpaceBuilder(subjects_dir, spacing=spacing)
    src = src_builder.create_surface_source_space()
    
    # Step 3: Create forward solution
    logger.info("\n[STEP 3] Computing forward solution...")
    fwd_builder = TemplateForwardSolutionBuilder(epochs, src, subjects_dir)
    
    if use_bem:
        logger.info("Creating BEM model...")
        bem = fwd_builder.create_bem_model()
    else:
        logger.info("Using spherical head model (faster)...")
        bem = None
    
    fwd = fwd_builder.compute_forward(bem=bem, trans=trans, n_jobs=n_jobs)
    
    # Step 4: Compute inverse solution
    logger.info("\n[STEP 4] Computing inverse solution...")
    inv_computer = InverseSolutionComputer(epochs, fwd)
    inv_computer.compute_noise_covariance(
        method=noise_cov_method,
        reg=noise_cov_reg
    )
    inv_computer.make_inverse_operator()
    
    # Apply to average
    stc = inv_computer.apply_inverse(method=method, lambda2=lambda2)
    
    # Apply to epochs
    stcs_epochs, epoch_indices = inv_computer.apply_inverse_epochs(
        method=method,
        lambda2=lambda2,
        max_epochs=max_epochs,
        random_state=random_state
    )
    
    # Step 5: Extract ROI time series using parcellation
    logger.info("\n[STEP 5] Extracting ROI time series...")
    roi_analyzer = ParcellationROIAnalyzer(stcs_epochs, subjects_dir, parc=parc)
    roi_analyzer.load_labels()
    label_timeseries = roi_analyzer.extract_label_timeseries()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ SOURCE RECONSTRUCTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Method: {method}")
    logger.info(f"Sources: {fwd['nsource']}")
    logger.info(f"ROIs/Labels: {len(label_timeseries)}")
    logger.info(f"Epochs processed: {len(stcs_epochs)}/{len(epochs)}")
    logger.info(f"Noise covariance: {inv_computer.noise_cov_strategy}")
    logger.info("="*60)
    
    return {
        'stc': stc,
        'stcs_epochs': stcs_epochs,
        'label_timeseries': label_timeseries,
        'src': src,
        'fwd': fwd,
        'inv': inv_computer.inverse_operator,
        'noise_covariance': inv_computer.noise_cov,
        'labels': roi_analyzer.labels,
        'subjects_dir': subjects_dir,
        'method': method,
        'epoch_indices': epoch_indices,
        'noise_cov_strategy': inv_computer.noise_cov_strategy,
        'parc': parc
    }

