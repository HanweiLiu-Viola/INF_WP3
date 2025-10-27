"""
通用清洗模块
适用于EEG和LFP的基础预处理：去趋势、去直流偏移、滤波、重采样
"""

import numpy as np
import mne
from scipy import signal
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SignalCleaner:
    """信号清洗处理器"""
    
    def __init__(self):
        """初始化信号清洗器"""
        self.processing_history = []
    
    def remove_dc_offset(self, raw: mne.io.Raw, copy: bool = True) -> mne.io.Raw:
        """
        去除直流偏移
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_clean : mne.io.Raw
            去除直流偏移后的数据
        """
        if copy:
            raw = raw.copy()
        
        # 计算并移除每个通道的均值
        data = raw.get_data()
        means = np.mean(data, axis=1, keepdims=True)
        raw._data = data - means
        
        self.processing_history.append('remove_dc_offset')
        logger.info("✓ 已去除直流偏移")
        
        return raw
    
    def detrend_data(self, raw: mne.io.Raw, order: int = 1, 
                    copy: bool = True) -> mne.io.Raw:
        """
        去趋势（消除慢漂移）
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        order : int
            多项式阶数 (1=线性, 0=仅去均值)
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_detrend : mne.io.Raw
            去趋势后的数据
        """
        if copy:
            raw = raw.copy()
        
        data = raw.get_data()
        n_channels, n_times = data.shape
        
        # 对每个通道进行去趋势
        for ch_idx in range(n_channels):
            if order == 0:
                # 仅去均值
                data[ch_idx] = data[ch_idx] - np.mean(data[ch_idx])
            else:
                # 多项式拟合去趋势
                x = np.arange(n_times)
                coeffs = np.polyfit(x, data[ch_idx], order)
                trend = np.polyval(coeffs, x)
                data[ch_idx] = data[ch_idx] - trend
        
        raw._data = data
        
        self.processing_history.append(f'detrend_order_{order}')
        logger.info(f"✓ 已执行 {order} 阶去趋势")
        
        return raw
    
    def apply_bandpass_filter(self, raw: mne.io.Raw, l_freq: float, 
                             h_freq: float, filter_type: str = 'fir',
                             copy: bool = True) -> mne.io.Raw:
        """
        应用带通滤波
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        l_freq : float
            低频截止频率 (Hz)
        h_freq : float
            高频截止频率 (Hz)
        filter_type : str
            滤波器类型 ('fir' 或 'iir')
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_filt : mne.io.Raw
            滤波后的数据
        """
        if copy:
            raw = raw.copy()
        
        # 应用带通滤波
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method=filter_type,
            picks='all',
            verbose=False
        )
        
        self.processing_history.append(f'bandpass_{l_freq}-{h_freq}Hz')
        logger.info(f"✓ 已应用带通滤波: {l_freq}-{h_freq} Hz ({filter_type})")
        
        return raw
    
    def apply_notch_filter(self, raw: mne.io.Raw, freqs: Union[float, list],
                          notch_widths: Optional[float] = None,
                          copy: bool = True) -> mne.io.Raw:
        """
        应用陷波滤波（去除工频干扰）
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        freqs : float or list
            陷波频率 (Hz)，如 50 或 [50, 100]
        notch_widths : float, optional
            陷波宽度
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_notch : mne.io.Raw
            陷波滤波后的数据
        """
        if copy:
            raw = raw.copy()
        
        if isinstance(freqs, (int, float)):
            freqs = [freqs]
        
        # 应用陷波滤波
        raw.notch_filter(
            freqs=freqs,
            notch_widths=notch_widths,
            picks='all',
            verbose=False
        )
        
        freqs_str = ', '.join([f"{f} Hz" for f in freqs])
        self.processing_history.append(f'notch_{freqs_str}')
        logger.info(f"✓ 已应用陷波滤波: {freqs_str}")
        
        return raw
    
    def resample_data(self, raw: mne.io.Raw, sfreq: float,
                     copy: bool = True) -> mne.io.Raw:
        """
        重采样至目标采样率
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        sfreq : float
            目标采样率 (Hz)
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_resampled : mne.io.Raw
            重采样后的数据
        """
        if copy:
            raw = raw.copy()
        
        original_sfreq = raw.info['sfreq']
        
        if np.isclose(original_sfreq, sfreq):
            logger.info(f"采样率已为 {sfreq} Hz，跳过重采样")
            return raw
        
        # 执行重采样
        raw.resample(sfreq, npad='auto', verbose=False)
        
        self.processing_history.append(f'resample_{original_sfreq}->{sfreq}Hz')
        logger.info(f"✓ 已重采样: {original_sfreq} Hz -> {sfreq} Hz")
        
        return raw
    
    def apply_standard_cleaning(self, raw: mne.io.Raw, 
                               signal_type: str = 'eeg',
                               target_sfreq: float = 1000.0,
                               line_freq: float = 50.0,
                               copy: bool = True) -> mne.io.Raw:
        """
        应用标准清洗流程
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        signal_type : str
            信号类型 ('eeg' 或 'lfp')
        target_sfreq : float
            目标采样率 (Hz)
        line_freq : float
            工频频率 (50 或 60 Hz)
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_clean : mne.io.Raw
            清洗后的数据
        """
        if copy:
            raw = raw.copy()
        
        logger.info(f"开始 {signal_type.upper()} 标准清洗流程...")
        
        # 1. 去除直流偏移
        raw = self.remove_dc_offset(raw, copy=False)
        
        # 2. 去趋势
        raw = self.detrend_data(raw, order=1, copy=False)
        
        # 3. 带通滤波
        if signal_type.lower() == 'eeg':
            # EEG: 1-100 Hz
            raw = self.apply_bandpass_filter(raw, l_freq=1.0, h_freq=100.0, copy=False)
        elif signal_type.lower() == 'lfp':
            # LFP: 1-200 Hz
            raw = self.apply_bandpass_filter(raw, l_freq=1.0, h_freq=200.0, copy=False)
        
        # 4. 陷波滤波
        notch_freqs = [line_freq, line_freq * 2]  # 基频和二次谐波
        raw = self.apply_notch_filter(raw, freqs=notch_freqs, copy=False)
        
        # 5. 重采样
        raw = self.resample_data(raw, sfreq=target_sfreq, copy=False)
        
        logger.info(f"✓ {signal_type.upper()} 标准清洗完成")
        
        return raw
    
    def compute_psd(self, raw: mne.io.Raw, fmin: float = 0.5, 
                   fmax: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度用于质量检查
        
        Parameters
        ----------
        raw : mne.io.Raw
            数据
        fmin : float
            最小频率 (Hz)
        fmax : float
            最大频率 (Hz)
            
        Returns
        -------
        psds : np.ndarray
            功率谱密度 (channels x frequencies)
        freqs : np.ndarray
            频率向量
        """
        spectrum = raw.compute_psd(
            fmin=fmin,
            fmax=fmax,
            n_fft=int(raw.info['sfreq'] * 2),  # 2秒窗口
            n_overlap=int(raw.info['sfreq']),   # 50% 重叠
            verbose=False
        )
        
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # 转换为 dB
        psds_db = 10 * np.log10(psds)
        
        return psds_db, freqs
    
    def get_processing_summary(self) -> str:
        """
        获取处理历史摘要
        
        Returns
        -------
        summary : str
            处理步骤摘要
        """
        if not self.processing_history:
            return "未执行任何处理步骤"
        
        summary = "处理步骤:\n"
        for i, step in enumerate(self.processing_history, 1):
            summary += f"  {i}. {step}\n"
        
        return summary


class EEGCleaner(SignalCleaner):
    """EEG专用清洗器"""
    
    def apply_eeg_cleaning(self, raw: mne.io.Raw, 
                          target_sfreq: float = 1000.0,
                          line_freq: float = 50.0,
                          l_freq: float = 1.0,
                          h_freq: float = 100.0,
                          copy: bool = True) -> mne.io.Raw:
        """
        应用EEG标准清洗流程
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始EEG数据
        target_sfreq : float
            目标采样率
        line_freq : float
            工频频率
        l_freq : float
            低频截止
        h_freq : float
            高频截止
        copy : bool
            是否复制
            
        Returns
        -------
        raw_clean : mne.io.Raw
            清洗后的EEG数据
        """
        if copy:
            raw = raw.copy()
        
        logger.info("="*60)
        logger.info("开始 EEG 清洗流程")
        logger.info("="*60)
        
        # 标准清洗
        raw = self.apply_standard_cleaning(
            raw, 
            signal_type='eeg',
            target_sfreq=target_sfreq,
            line_freq=line_freq,
            copy=False
        )
        
        logger.info("="*60)
        logger.info("EEG 清洗流程完成")
        logger.info("="*60)
        
        return raw


class LFPCleaner(SignalCleaner):
    """LFP专用清洗器"""
    
    def apply_lfp_cleaning(self, raw: mne.io.Raw,
                          target_sfreq: float = 1000.0,
                          line_freq: float = 50.0,
                          l_freq: float = 1.0,
                          h_freq: float = 200.0,
                          copy: bool = True) -> mne.io.Raw:
        """
        应用LFP标准清洗流程
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始LFP数据
        target_sfreq : float
            目标采样率
        line_freq : float
            工频频率
        l_freq : float
            低频截止
        h_freq : float
            高频截止
        copy : bool
            是否复制
            
        Returns
        -------
        raw_clean : mne.io.Raw
            清洗后的LFP数据
        """
        if copy:
            raw = raw.copy()
        
        logger.info("="*60)
        logger.info("开始 LFP 清洗流程")
        logger.info("="*60)
        
        # 标准清洗
        raw = self.apply_standard_cleaning(
            raw,
            signal_type='lfp',
            target_sfreq=target_sfreq,
            line_freq=line_freq,
            copy=False
        )
        
        logger.info("="*60)
        logger.info("LFP 清洗流程完成")
        logger.info("="*60)
        
        return raw
