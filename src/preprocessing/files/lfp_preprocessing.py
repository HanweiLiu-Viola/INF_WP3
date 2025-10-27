"""
LFP专用预处理模块
包括刺激伪迹去除、电极接触点管理、降噪处理
"""

import numpy as np
import mne
from scipy import signal, interpolate
from scipy.signal import butter, filtfilt
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LFPPreprocessor:
    """LFP预处理器"""
    
    def __init__(self):
        """初始化LFP预处理器"""
        self.processing_log = []
        self.electrode_info = {}
    
    def remove_stimulation_artifacts(self, raw: mne.io.Raw,
                                    stim_events: Optional[np.ndarray] = None,
                                    method: str = 'template',
                                    window: Tuple[float, float] = (-0.005, 0.01),
                                    copy: bool = True) -> mne.io.Raw:
        """
        去除刺激伪迹
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始LFP数据
        stim_events : np.ndarray, optional
            刺激事件数组 (n_events, 3)
        method : str
            去除方法 ('template', 'blanking', 'interpolate')
        window : tuple
            伪迹窗口 (tmin, tmax) 相对刺激时间（秒）
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_clean : mne.io.Raw
            去除伪迹后的数据
        """
        if copy:
            raw = raw.copy()
        
        if stim_events is None:
            # 尝试从注释中提取刺激事件
            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                # 假设刺激事件ID为特定值
                stim_events = events[events[:, 2] == 2]  # 可调整
            except:
                logger.warning("⚠ 未找到刺激事件，跳过伪迹去除")
                return raw
        
        if len(stim_events) == 0:
            logger.info("没有刺激事件，跳过伪迹去除")
            return raw
        
        logger.info(f"开始去除刺激伪迹 (方法: {method}, 事件数: {len(stim_events)})...")
        
        sfreq = raw.info['sfreq']
        data = raw.get_data()
        
        # 计算窗口样本数
        win_samples = [int(window[0] * sfreq), int(window[1] * sfreq)]
        
        if method == 'template':
            # 模板减法
            data_clean = self._template_subtraction(
                data, stim_events, win_samples, sfreq
            )
        
        elif method == 'blanking':
            # 置零法
            data_clean = self._blanking_method(
                data, stim_events, win_samples, sfreq
            )
        
        elif method == 'interpolate':
            # 插值法
            data_clean = self._interpolation_method(
                data, stim_events, win_samples, sfreq
            )
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        raw._data = data_clean
        
        logger.info(f"✓ 已去除 {len(stim_events)} 个刺激伪迹")
        self.processing_log.append(f'remove_stim_artifacts_{method}')
        
        return raw
    
    def _template_subtraction(self, data: np.ndarray, events: np.ndarray,
                             win_samples: List[int], sfreq: float) -> np.ndarray:
        """
        模板减法去除伪迹
        
        Parameters
        ----------
        data : np.ndarray
            数据 (n_channels, n_times)
        events : np.ndarray
            事件数组
        win_samples : list
            窗口样本 [start, end]
        sfreq : float
            采样率
            
        Returns
        -------
        data_clean : np.ndarray
            清洗后的数据
        """
        data_clean = data.copy()
        n_channels = data.shape[0]
        
        # 提取所有伪迹段
        artifacts = []
        for event_sample in events[:, 0]:
            start = event_sample + win_samples[0]
            end = event_sample + win_samples[1]
            
            if start >= 0 and end < data.shape[1]:
                artifact = data[:, start:end]
                artifacts.append(artifact)
        
        if len(artifacts) == 0:
            return data_clean
        
        # 计算平均模板
        template = np.mean(artifacts, axis=0)
        
        # 减去模板
        for event_sample in events[:, 0]:
            start = event_sample + win_samples[0]
            end = event_sample + win_samples[1]
            
            if start >= 0 and end < data.shape[1]:
                data_clean[:, start:end] -= template
        
        return data_clean
    
    def _blanking_method(self, data: np.ndarray, events: np.ndarray,
                        win_samples: List[int], sfreq: float) -> np.ndarray:
        """
        置零法去除伪迹
        
        Parameters
        ----------
        data : np.ndarray
            数据
        events : np.ndarray
            事件数组
        win_samples : list
            窗口样本
        sfreq : float
            采样率
            
        Returns
        -------
        data_clean : np.ndarray
            清洗后的数据
        """
        data_clean = data.copy()
        
        for event_sample in events[:, 0]:
            start = event_sample + win_samples[0]
            end = event_sample + win_samples[1]
            
            if start >= 0 and end < data.shape[1]:
                # 置零
                data_clean[:, start:end] = 0
        
        return data_clean
    
    def _interpolation_method(self, data: np.ndarray, events: np.ndarray,
                             win_samples: List[int], sfreq: float) -> np.ndarray:
        """
        插值法去除伪迹
        
        Parameters
        ----------
        data : np.ndarray
            数据
        events : np.ndarray
            事件数组
        win_samples : list
            窗口样本
        sfreq : float
            采样率
            
        Returns
        -------
        data_clean : np.ndarray
            清洗后的数据
        """
        data_clean = data.copy()
        n_channels = data.shape[0]
        
        for event_sample in events[:, 0]:
            start = event_sample + win_samples[0]
            end = event_sample + win_samples[1]
            
            if start >= 10 and end < data.shape[1] - 10:
                # 使用前后10个样本进行线性插值
                for ch in range(n_channels):
                    x_before = np.arange(start - 10, start)
                    y_before = data_clean[ch, start - 10:start]
                    
                    x_after = np.arange(end, end + 10)
                    y_after = data_clean[ch, end:end + 10]
                    
                    # 合并边界点
                    x_boundary = np.concatenate([x_before, x_after])
                    y_boundary = np.concatenate([y_before, y_after])
                    
                    # 插值
                    x_interp = np.arange(start, end)
                    f = interpolate.interp1d(x_boundary, y_boundary, kind='linear')
                    data_clean[ch, start:end] = f(x_interp)
        
        return data_clean
    
    def parse_electrode_contacts(self, raw: mne.io.Raw,
                                channels_info: Optional[Dict] = None) -> Dict:
        """
        解析电极接触点信息
        
        Parameters
        ----------
        raw : mne.io.Raw
            LFP数据
        channels_info : dict, optional
            通道信息字典
            
        Returns
        -------
        electrode_info : dict
            电极信息
        """
        logger.info("解析电极接触点信息...")
        
        electrode_info = {
            'left': [],
            'right': [],
            'contacts': {}
        }
        
        # 解析通道名称
        for ch_name in raw.ch_names:
            # 假设命名规则: STN_L_01, STN_R_02等
            if '_L_' in ch_name or ch_name.endswith('_L'):
                electrode_info['left'].append(ch_name)
                side = 'left'
            elif '_R_' in ch_name or ch_name.endswith('_R'):
                electrode_info['right'].append(ch_name)
                side = 'right'
            else:
                side = 'unknown'
            
            # 提取接触点编号
            contact_num = ch_name.split('_')[-1] if '_' in ch_name else 'unknown'
            
            electrode_info['contacts'][ch_name] = {
                'side': side,
                'contact': contact_num,
                'region': ch_name.split('_')[0] if '_' in ch_name else 'unknown'
            }
        
        logger.info(f"✓ 左侧电极: {len(electrode_info['left'])} 个接触点")
        logger.info(f"✓ 右侧电极: {len(electrode_info['right'])} 个接触点")
        
        self.electrode_info = electrode_info
        self.processing_log.append('parse_electrode_contacts')
        
        return electrode_info
    
    def apply_bipolar_reference(self, raw: mne.io.Raw,
                               anode: Optional[List[str]] = None,
                               cathode: Optional[List[str]] = None,
                               copy: bool = True) -> mne.io.Raw:
        """
        应用双极参考
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        anode : list, optional
            阳极通道（如果为None，自动配对相邻接触点）
        cathode : list, optional
            阴极通道
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_bipolar : mne.io.Raw
            双极参考数据
        """
        if copy:
            raw = raw.copy()
        
        if anode is None or cathode is None:
            # 自动生成相邻接触点双极配对
            anode, cathode = self._auto_bipolar_pairs(raw)
        
        # 应用双极参考
        raw_bipolar = mne.set_bipolar_reference(
            raw,
            anode=anode,
            cathode=cathode,
            copy=False,
            verbose=False
        )
        
        logger.info(f"✓ 已应用双极参考，生成 {len(anode)} 个双极通道")
        self.processing_log.append('apply_bipolar_reference')
        
        return raw_bipolar
    
    def _auto_bipolar_pairs(self, raw: mne.io.Raw) -> Tuple[List[str], List[str]]:
        """
        自动生成双极配对
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
            
        Returns
        -------
        anode : list
            阳极列表
        cathode : list
            阴极列表
        """
        anode = []
        cathode = []
        
        # 按前缀分组通道
        channel_groups = {}
        for ch_name in raw.ch_names:
            prefix = '_'.join(ch_name.split('_')[:-1])  # 去掉最后的数字
            if prefix not in channel_groups:
                channel_groups[prefix] = []
            channel_groups[prefix].append(ch_name)
        
        # 为每组生成相邻配对
        for prefix, channels in channel_groups.items():
            # 排序通道
            channels_sorted = sorted(channels)
            
            # 生成配对 (ch0-ch1, ch1-ch2, ...)
            for i in range(len(channels_sorted) - 1):
                anode.append(channels_sorted[i])
                cathode.append(channels_sorted[i + 1])
        
        return anode, cathode
    
    def apply_wavelet_denoising(self, raw: mne.io.Raw,
                               wavelet: str = 'db4',
                               level: int = 4,
                               threshold_mode: str = 'soft',
                               copy: bool = True) -> mne.io.Raw:
        """
        应用小波去噪
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        wavelet : str
            小波基函数
        level : int
            分解层数
        threshold_mode : str
            阈值模式 ('soft' 或 'hard')
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_denoised : mne.io.Raw
            去噪后的数据
        """
        if copy:
            raw = raw.copy()
        
        try:
            import pywt
        except ImportError:
            logger.warning("⚠ PyWavelets未安装，跳过小波去噪")
            return raw
        
        logger.info(f"应用小波去噪 (小波: {wavelet}, 层数: {level})...")
        
        data = raw.get_data()
        data_denoised = np.zeros_like(data)
        
        for ch_idx in range(data.shape[0]):
            # 小波分解
            coeffs = pywt.wavedec(data[ch_idx], wavelet, level=level)
            
            # 计算阈值（使用通用阈值）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data[ch_idx])))
            
            # 应用阈值
            coeffs_thresholded = [coeffs[0]]  # 保留近似系数
            for coeff in coeffs[1:]:
                if threshold_mode == 'soft':
                    coeff_thresh = pywt.threshold(coeff, threshold, mode='soft')
                else:
                    coeff_thresh = pywt.threshold(coeff, threshold, mode='hard')
                coeffs_thresholded.append(coeff_thresh)
            
            # 重构信号
            data_denoised[ch_idx] = pywt.waverec(coeffs_thresholded, wavelet)
        
        raw._data = data_denoised
        
        logger.info("✓ 小波去噪完成")
        self.processing_log.append(f'wavelet_denoising_{wavelet}')
        
        return raw
    
    def apply_smoothing(self, raw: mne.io.Raw, window_length: int = 11,
                       polyorder: int = 3, copy: bool = True) -> mne.io.Raw:
        """
        应用Savitzky-Golay平滑
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        window_length : int
            窗口长度（必须为奇数）
        polyorder : int
            多项式阶数
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_smooth : mne.io.Raw
            平滑后的数据
        """
        if copy:
            raw = raw.copy()
        
        from scipy.signal import savgol_filter
        
        logger.info(f"应用Savitzky-Golay平滑 (窗口: {window_length}, 阶数: {polyorder})...")
        
        data = raw.get_data()
        data_smooth = np.zeros_like(data)
        
        for ch_idx in range(data.shape[0]):
            data_smooth[ch_idx] = savgol_filter(
                data[ch_idx],
                window_length=window_length,
                polyorder=polyorder
            )
        
        raw._data = data_smooth
        
        logger.info("✓ 平滑完成")
        self.processing_log.append(f'smoothing_savgol')
        
        return raw
    
    def enhance_snr(self, raw: mne.io.Raw, method: str = 'car',
                   copy: bool = True) -> mne.io.Raw:
        """
        增强信噪比
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        method : str
            方法 ('car': 共平均参考, 'median': 中值参考)
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_enhanced : mne.io.Raw
            增强后的数据
        """
        if copy:
            raw = raw.copy()
        
        data = raw.get_data()
        
        if method == 'car':
            # 共平均参考 (Common Average Reference)
            car = np.mean(data, axis=0, keepdims=True)
            data_enhanced = data - car
            logger.info("✓ 已应用共平均参考(CAR)")
        
        elif method == 'median':
            # 中值参考
            median_ref = np.median(data, axis=0, keepdims=True)
            data_enhanced = data - median_ref
            logger.info("✓ 已应用中值参考")
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        raw._data = data_enhanced
        
        self.processing_log.append(f'enhance_snr_{method}')
        
        return raw
    
    def detect_outliers(self, raw: mne.io.Raw, threshold: float = 5.0) -> Dict:
        """
        检测异常值
        
        Parameters
        ----------
        raw : mne.io.Raw
            LFP数据
        threshold : float
            Z-score阈值
            
        Returns
        -------
        outlier_info : dict
            异常值信息
        """
        logger.info(f"检测异常值 (阈值: {threshold} SD)...")
        
        data = raw.get_data()
        outlier_info = {}
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ch_data = data[ch_idx]
            
            # 计算Z-score
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            z_scores = np.abs((ch_data - mean) / std)
            
            # 找出异常值
            outlier_mask = z_scores > threshold
            n_outliers = np.sum(outlier_mask)
            
            outlier_info[ch_name] = {
                'n_outliers': n_outliers,
                'percent': (n_outliers / len(ch_data)) * 100,
                'outlier_indices': np.where(outlier_mask)[0]
            }
            
            if n_outliers > 0:
                logger.info(f"  {ch_name}: {n_outliers} 个异常值 "
                          f"({outlier_info[ch_name]['percent']:.2f}%)")
        
        return outlier_info
    
    def get_processing_summary(self) -> str:
        """获取处理摘要"""
        if not self.processing_log:
            return "未执行任何处理步骤"
        
        summary = "LFP处理步骤:\n"
        for i, step in enumerate(self.processing_log, 1):
            summary += f"  {i}. {step}\n"
        
        return summary
