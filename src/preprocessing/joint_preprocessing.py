"""
EEG-LFP联合处理模块
包括时间对齐、频段分解、标准化等跨模态分析准备
"""

import numpy as np
import mne
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class JointPreprocessor:
    """EEG-LFP联合处理器"""
    
    def __init__(self):
        """初始化联合处理器"""
        self.processing_log = []
        
        # 定义频段
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_beta': (13, 20),
            'high_beta': (20, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100),
            'gamma': (30, 100)
        }
    
    def align_time_windows(self, eeg_raw: mne.io.Raw, lfp_raw: mne.io.Raw,
                          crop_to: str = 'shorter',
                          copy: bool = True) -> Tuple[mne.io.Raw, mne.io.Raw]:
        """
        对齐EEG和LFP的时间窗口
        
        Parameters
        ----------
        eeg_raw : mne.io.Raw
            EEG数据
        lfp_raw : mne.io.Raw
            LFP数据
        crop_to : str
            裁剪策略 ('shorter': 裁至较短, 'longer': 填充至较长)
        copy : bool
            是否复制数据
            
        Returns
        -------
        eeg_aligned : mne.io.Raw
            对齐后的EEG
        lfp_aligned : mne.io.Raw
            对齐后的LFP
        """
        if copy:
            eeg_raw = eeg_raw.copy()
            lfp_raw = lfp_raw.copy()
        
        logger.info("对齐时间窗口...")
        
        # 获取时间范围
        eeg_tmin, eeg_tmax = eeg_raw.times[0], eeg_raw.times[-1]
        lfp_tmin, lfp_tmax = lfp_raw.times[0], lfp_raw.times[-1]
        
        logger.info(f"  EEG: {eeg_tmin:.3f} - {eeg_tmax:.3f} s ({eeg_tmax - eeg_tmin:.3f} s)")
        logger.info(f"  LFP: {lfp_tmin:.3f} - {lfp_tmax:.3f} s ({lfp_tmax - lfp_tmin:.3f} s)")
        
        if crop_to == 'shorter':
            # 裁剪至重叠的较短区间
            tmin = max(eeg_tmin, lfp_tmin)
            tmax = min(eeg_tmax, lfp_tmax)
            
            eeg_raw.crop(tmin=tmin, tmax=tmax)
            lfp_raw.crop(tmin=tmin, tmax=tmax)
            
            logger.info(f"✓ 已裁剪至: {tmin:.3f} - {tmax:.3f} s ({tmax - tmin:.3f} s)")
        
        elif crop_to == 'longer':
            # 填充至较长区间（使用零填充）
            tmin = min(eeg_tmin, lfp_tmin)
            tmax = max(eeg_tmax, lfp_tmax)
            
            # 这里需要实现零填充逻辑
            logger.warning("⚠ 零填充功能尚未实现，使用裁剪策略代替")
            tmin = max(eeg_tmin, lfp_tmin)
            tmax = min(eeg_tmax, lfp_tmax)
            
            eeg_raw.crop(tmin=tmin, tmax=tmax)
            lfp_raw.crop(tmin=tmin, tmax=tmax)
        
        self.processing_log.append('align_time_windows')
        
        return eeg_raw, lfp_raw
    
    def synchronize_epochs(self, eeg_epochs: mne.Epochs, lfp_epochs: mne.Epochs,
                          tolerance: float = 0.001) -> Tuple[mne.Epochs, mne.Epochs]:
        """
        同步EEG和LFP的epochs
        
        Parameters
        ----------
        eeg_epochs : mne.Epochs
            EEG epochs
        lfp_epochs : mne.Epochs
            LFP epochs
        tolerance : float
            时间容差（秒）
            
        Returns
        -------
        eeg_sync : mne.Epochs
            同步后的EEG epochs
        lfp_sync : mne.Epochs
            同步后的LFP epochs
        """
        logger.info("同步epochs...")
        
        # 获取事件时间
        eeg_events = eeg_epochs.events[:, 0] / eeg_epochs.info['sfreq']
        lfp_events = lfp_epochs.events[:, 0] / lfp_epochs.info['sfreq']
        
        # 找出匹配的epochs
        matching_indices = []
        for i, eeg_t in enumerate(eeg_events):
            # 找到最接近的LFP事件
            diffs = np.abs(lfp_events - eeg_t)
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < tolerance:
                matching_indices.append((i, min_idx))
        
        if len(matching_indices) == 0:
            raise ValueError("未找到匹配的epochs")
        
        # 提取匹配的epochs
        eeg_indices, lfp_indices = zip(*matching_indices)
        eeg_sync = eeg_epochs[list(eeg_indices)]
        lfp_sync = lfp_epochs[list(lfp_indices)]
        
        logger.info(f"✓ 同步完成: {len(matching_indices)} 个匹配epochs")
        self.processing_log.append('synchronize_epochs')
        
        return eeg_sync, lfp_sync
    
    def extract_frequency_bands(self, raw: mne.io.Raw,
                               bands: Optional[Dict[str, Tuple]] = None,
                               method: str = 'filter',
                               copy: bool = True) -> Dict[str, mne.io.Raw]:
        """
        提取频段信号
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        bands : dict, optional
            频段字典 {'band_name': (fmin, fmax)}
        method : str
            提取方法 ('filter', 'hilbert')
        copy : bool
            是否复制数据
            
        Returns
        -------
        band_data : dict
            频段数据字典
        """
        if bands is None:
            bands = self.freq_bands
        
        logger.info(f"提取频段信号 (方法: {method})...")
        
        band_data = {}
        
        for band_name, (fmin, fmax) in bands.items():
            if copy:
                raw_band = raw.copy()
            else:
                raw_band = raw
            
            if method == 'filter':
                # 带通滤波
                raw_band.filter(
                    l_freq=fmin,
                    h_freq=fmax,
                    picks='all',
                    method='fir',
                    verbose=False
                )
                band_data[band_name] = raw_band
                
            elif method == 'hilbert':
                # 带通滤波 + Hilbert变换提取包络
                raw_band.filter(
                    l_freq=fmin,
                    h_freq=fmax,
                    picks='all',
                    method='fir',
                    verbose=False
                )
                raw_band.apply_hilbert(envelope=True, verbose=False)
                band_data[band_name] = raw_band
            
            logger.info(f"  {band_name}: {fmin}-{fmax} Hz")
        
        logger.info(f"✓ 提取了 {len(band_data)} 个频段")
        self.processing_log.append(f'extract_frequency_bands_{method}')
        
        return band_data
    
    def compute_time_frequency(self, epochs: mne.Epochs,
                              freqs: Optional[np.ndarray] = None,
                              n_cycles: Union[float, np.ndarray] = 7.0,
                              method: str = 'morlet') -> mne.time_frequency.AverageTFR:
        """
        计算时频表示
        
        Parameters
        ----------
        epochs : mne.Epochs
            分段数据
        freqs : np.ndarray, optional
            频率数组
        n_cycles : float or array
            小波周期数
        method : str
            方法 ('morlet', 'multitaper')
            
        Returns
        -------
        power : AverageTFR
            时频功率
        """
        logger.info(f"计算时频表示 (方法: {method})...")
        
        if freqs is None:
            freqs = np.logspace(np.log10(1), np.log10(100), 30)
        
        if method == 'morlet':
            # Morlet小波
            power = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=True,
                verbose=False
            )
        
        elif method == 'multitaper':
            # 多锥度方法
            power = mne.time_frequency.tfr_multitaper(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=True,
                verbose=False
            )
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        logger.info(f"✓ 时频分析完成")
        logger.info(f"  频率范围: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")
        logger.info(f"  时间范围: {power.times[0]:.3f}-{power.times[-1]:.3f} s")
        
        self.processing_log.append(f'compute_time_frequency_{method}')
        
        return power
    
    def normalize_signals(self, data: np.ndarray, method: str = 'zscore',
                         axis: int = -1) -> np.ndarray:
        """
        标准化信号
        
        Parameters
        ----------
        data : np.ndarray
            数据数组
        method : str
            标准化方法 ('zscore', 'minmax', 'robust')
        axis : int
            标准化轴
            
        Returns
        -------
        data_norm : np.ndarray
            标准化后的数据
        """
        logger.info(f"标准化信号 (方法: {method})...")
        
        if method == 'zscore':
            # Z-score标准化
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            data_norm = (data - mean) / (std + 1e-10)
        
        elif method == 'minmax':
            # Min-Max标准化到[0, 1]
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            data_norm = (data - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'robust':
            # 鲁棒标准化（使用中位数和四分位距）
            median = np.median(data, axis=axis, keepdims=True)
            q75 = np.percentile(data, 75, axis=axis, keepdims=True)
            q25 = np.percentile(data, 25, axis=axis, keepdims=True)
            iqr = q75 - q25
            data_norm = (data - median) / (iqr + 1e-10)
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        logger.info("✓ 标准化完成")
        self.processing_log.append(f'normalize_{method}')
        
        return data_norm
    
    def create_joint_epochs(self, eeg_raw: mne.io.Raw, lfp_raw: mne.io.Raw,
                           events: np.ndarray, event_id: Dict,
                           tmin: float = -0.5, tmax: float = 1.5,
                           baseline: Optional[Tuple] = (-0.2, 0)) -> Tuple[mne.Epochs, mne.Epochs]:
        """
        创建联合epochs
        
        Parameters
        ----------
        eeg_raw : mne.io.Raw
            EEG数据
        lfp_raw : mne.io.Raw
            LFP数据
        events : np.ndarray
            事件数组
        event_id : dict
            事件ID字典
        tmin : float
            Epoch起始时间
        tmax : float
            Epoch结束时间
        baseline : tuple, optional
            基线校正时间窗
            
        Returns
        -------
        eeg_epochs : mne.Epochs
            EEG epochs
        lfp_epochs : mne.Epochs
            LFP epochs
        """
        logger.info("创建联合epochs...")
        
        # 创建EEG epochs
        eeg_epochs = mne.Epochs(
            eeg_raw,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False
        )
        
        # 创建LFP epochs
        lfp_epochs = mne.Epochs(
            lfp_raw,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False
        )
        
        logger.info(f"✓ 创建完成:")
        logger.info(f"  EEG: {len(eeg_epochs)} epochs")
        logger.info(f"  LFP: {len(lfp_epochs)} epochs")
        
        self.processing_log.append('create_joint_epochs')

        return eeg_epochs, lfp_epochs

    @staticmethod
    def _bad_epoch_indices_from_eeg(eeg_epochs: mne.Epochs) -> np.ndarray:
        """Infer indices of EEG epochs marked as bad (e.g., by AutoReject)."""

        if hasattr(eeg_epochs, '_bad_epochs_autoreject'):
            bad_idx = np.asarray(eeg_epochs._bad_epochs_autoreject, dtype=int)
            bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < len(eeg_epochs))]
            return np.unique(bad_idx)

        return np.array([], dtype=int)

    def align_lfp_to_eeg_epochs(
        self,
        eeg_epochs: mne.Epochs,
        lfp_raw: mne.io.Raw,
        preload: bool = True,
        drop_bad_from_eeg: bool = True,
    ) -> Tuple[mne.Epochs, mne.Epochs, np.ndarray, np.ndarray]:
        """Create LFP epochs aligned to EEG epochs and drop rejected ones.

        Parameters
        ----------
        eeg_epochs : mne.Epochs
            EEG epochs (may contain AutoReject metadata with marked bad epochs).
        lfp_raw : mne.io.Raw
            Raw LFP recording to segment using the EEG events.
        preload : bool
            Whether to preload the created LFP epochs into memory.
        drop_bad_from_eeg : bool
            If ``True`` (default), remove epochs that were marked as bad in the
            EEG data from both modalities.

        Returns
        -------
        eeg_synced : mne.Epochs
            EEG epochs after optional bad-epoch removal (copy).
        lfp_synced : mne.Epochs
            LFP epochs cropped with the same time window and selection.
        kept_indices : np.ndarray
            Indices of the original EEG epochs that were retained.
        dropped_indices : np.ndarray
            Indices of the EEG epochs that were removed (empty if none).
        """

        logger.info("根据EEG epochs 创建对齐的LFP epochs...")

        events = eeg_epochs.events
        if events is None or len(events) == 0:
            raise ValueError("EEG epochs do not contain events for alignment")

        event_id = eeg_epochs.event_id
        tmin = eeg_epochs.tmin
        tmax = eeg_epochs.tmax
        baseline = eeg_epochs.baseline

        lfp_epochs = mne.Epochs(
            lfp_raw,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=preload,
            verbose=False,
        )

        if len(lfp_epochs) != len(eeg_epochs):
            raise ValueError(
                "LFP epoch count does not match EEG epoch count. Ensure the "
                "raw LFP data covers the entire EEG recording interval."
            )

        all_indices = np.arange(len(eeg_epochs))
        dropped_indices = np.array([], dtype=int)

        if drop_bad_from_eeg:
            dropped_indices = self._bad_epoch_indices_from_eeg(eeg_epochs)

        kept_mask = np.ones(len(eeg_epochs), dtype=bool)
        kept_mask[dropped_indices] = False
        kept_indices = all_indices[kept_mask]

        if kept_indices.size == 0:
            raise ValueError("All EEG epochs were marked as bad; nothing to align.")

        eeg_synced = eeg_epochs.copy()[kept_indices]
        lfp_synced = lfp_epochs.copy()[kept_indices]

        logger.info(
            "✓ LFP epochs 已与EEG对齐: 保留 %d/%d 个epochs",
            kept_indices.size,
            len(eeg_epochs),
        )

        if dropped_indices.size > 0:
            logger.info(
                "  丢弃的EEG epochs索引: %s",
                dropped_indices.tolist(),
            )

        self.processing_log.append('align_lfp_to_eeg_epochs')

        return eeg_synced, lfp_synced, kept_indices, dropped_indices
    
    def compute_band_power(self, epochs: mne.Epochs,
                          bands: Optional[Dict[str, Tuple]] = None,
                          method: str = 'welch') -> Dict[str, np.ndarray]:
        """
        计算频段功率
        
        Parameters
        ----------
        epochs : mne.Epochs
            分段数据
        bands : dict, optional
            频段字典
        method : str
            计算方法 ('welch', 'multitaper')
            
        Returns
        -------
        band_power : dict
            频段功率字典 {band_name: power_array}
        """
        if bands is None:
            bands = self.freq_bands
        
        logger.info(f"计算频段功率 (方法: {method})...")
        
        # 计算功率谱
        if method == 'welch':
            spectrum = epochs.compute_psd(
                method='welch',
                fmin=0.5,
                fmax=150,
                n_fft=int(epochs.info['sfreq'] * 2),
                verbose=False
            )
        elif method == 'multitaper':
            spectrum = epochs.compute_psd(
                method='multitaper',
                fmin=0.5,
                fmax=150,
                verbose=False
            )
        
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # 提取各频段功率
        band_power = {}
        for band_name, (fmin, fmax) in bands.items():
            # 找到频段索引
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            # 计算平均功率
            band_psds = psds[:, :, freq_mask]
            band_power[band_name] = np.mean(band_psds, axis=-1)  # (n_epochs, n_channels)
            
            logger.info(f"  {band_name}: {fmin}-{fmax} Hz")
        
        logger.info("✓ 频段功率计算完成")
        self.processing_log.append(f'compute_band_power_{method}')
        
        return band_power
    
    def prepare_connectivity_data(self, eeg_epochs: mne.Epochs,
                                 lfp_epochs: mne.Epochs,
                                 eeg_roi_names: Optional[List[str]] = None,
                                 lfp_roi_names: Optional[List[str]] = None) -> Dict:
        """
        准备连接性分析数据
        
        Parameters
        ----------
        eeg_epochs : mne.Epochs
            EEG epochs
        lfp_epochs : mne.Epochs
            LFP epochs
        eeg_roi_names : list, optional
            EEG ROI名称
        lfp_roi_names : list, optional
            LFP ROI名称
            
        Returns
        -------
        connectivity_data : dict
            连接性分析数据包
        """
        logger.info("准备连接性分析数据...")
        
        # 获取数据
        eeg_data = eeg_epochs.get_data()  # (n_epochs, n_eeg_channels, n_times)
        lfp_data = lfp_epochs.get_data()  # (n_epochs, n_lfp_channels, n_times)
        
        # 合并数据
        combined_data = np.concatenate([eeg_data, lfp_data], axis=1)
        
        # 准备通道名称
        if eeg_roi_names is None:
            eeg_roi_names = eeg_epochs.ch_names
        if lfp_roi_names is None:
            lfp_roi_names = lfp_epochs.ch_names
        
        all_roi_names = eeg_roi_names + lfp_roi_names
        
        # 创建索引映射
        n_eeg = len(eeg_roi_names)
        n_lfp = len(lfp_roi_names)
        
        connectivity_data = {
            'data': combined_data,
            'sfreq': eeg_epochs.info['sfreq'],
            'times': eeg_epochs.times,
            'roi_names': all_roi_names,
            'eeg_indices': list(range(n_eeg)),
            'lfp_indices': list(range(n_eeg, n_eeg + n_lfp)),
            'n_eeg': n_eeg,
            'n_lfp': n_lfp
        }
        
        logger.info("✓ 数据准备完成")
        logger.info(f"  形状: {combined_data.shape}")
        logger.info(f"  EEG通道: {n_eeg}")
        logger.info(f"  LFP通道: {n_lfp}")
        
        self.processing_log.append('prepare_connectivity_data')
        
        return connectivity_data
    
    def get_processing_summary(self) -> str:
        """获取处理摘要"""
        if not self.processing_log:
            return "未执行任何联合处理步骤"
        
        summary = "联合处理步骤:\n"
        for i, step in enumerate(self.processing_log, 1):
            summary += f"  {i}. {step}\n"
        
        return summary
    