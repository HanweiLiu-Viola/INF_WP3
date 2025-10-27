"""
EEG专用预处理模块
包括坏导检测、重参考、ICA去伪迹、事件分段、源空间重建
"""

import numpy as np
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """EEG预处理器"""
    
    def __init__(self):
        """初始化EEG预处理器"""
        self.bad_channels = []
        self.ica = None
        self.processing_log = []
    
    def detect_bad_channels(self, raw: mne.io.Raw, 
                           method: str = 'correlation',
                           threshold: float = 0.4) -> List[str]:
        """
        检测坏导
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始EEG数据
        method : str
            检测方法 ('correlation', 'variance', 'ransac')
        threshold : float
            阈值（用于correlation方法）
            
        Returns
        -------
        bad_channels : list
            坏导列表
        """
        logger.info(f"开始检测坏导 (方法: {method})...")
        
        if method == 'correlation':
            # 基于相关性的坏导检测
            data = raw.get_data()
            n_channels = data.shape[0]
            
            # 计算通道间相关性
            corr_matrix = np.corrcoef(data)
            
            # 找出与其他通道平均相关性低的通道
            mean_corr = np.mean(corr_matrix, axis=1)
            bad_idx = np.where(mean_corr < threshold)[0]
            bad_channels = [raw.ch_names[i] for i in bad_idx]
            
        elif method == 'variance':
            # 基于方差的坏导检测
            data = raw.get_data()
            variances = np.var(data, axis=1)
            
            # 使用3倍标准差作为阈值
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            threshold_low = mean_var - 3 * std_var
            threshold_high = mean_var + 3 * std_var
            
            bad_idx = np.where((variances < threshold_low) | 
                              (variances > threshold_high))[0]
            bad_channels = [raw.ch_names[i] for i in bad_idx]
            
        elif method == 'ransac':
            # 使用MNE的RANSAC方法
            from mne.preprocessing import Ransac
            
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
            ransac.fit(raw)
            bad_channels = ransac.bad_chs_
            
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        self.bad_channels = bad_channels
        
        if bad_channels:
            logger.info(f"✓ 检测到 {len(bad_channels)} 个坏导: {', '.join(bad_channels)}")
        else:
            logger.info("✓ 未检测到坏导")
        
        self.processing_log.append(f'detect_bad_channels_{method}')
        
        return bad_channels
    
    def interpolate_bad_channels(self, raw: mne.io.Raw,
                                bad_channels: Optional[List[str]] = None,
                                copy: bool = True) -> mne.io.Raw:
        """
        插值坏导
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        bad_channels : list, optional
            坏导列表，如果为None则使用已检测的坏导
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_interp : mne.io.Raw
            插值后的数据
        """
        if copy:
            raw = raw.copy()
        
        if bad_channels is None:
            bad_channels = self.bad_channels
        
        if not bad_channels:
            logger.info("没有坏导需要插值")
            return raw
        
        # 标记坏导
        raw.info['bads'] = bad_channels
        
        # 执行插值
        raw.interpolate_bads(reset_bads=True, verbose=False)
        
        logger.info(f"✓ 已插值 {len(bad_channels)} 个坏导")
        self.processing_log.append('interpolate_bad_channels')
        
        return raw
    
    def set_reference(self, raw: mne.io.Raw, ref_type: str = 'average',
                     ref_channels: Optional[List[str]] = None,
                     copy: bool = True) -> mne.io.Raw:
        """
        设置参考电极
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        ref_type : str
            参考类型 ('average', 'mastoid', 'custom')
        ref_channels : list, optional
            自定义参考通道（当ref_type='custom'时）
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_ref : mne.io.Raw
            重参考后的数据
        """
        if copy:
            raw = raw.copy()
        
        if ref_type == 'average':
            # 平均参考
            raw.set_eeg_reference('average', projection=False, verbose=False)
            logger.info("✓ 已设置平均参考")
            
        elif ref_type == 'mastoid':
            # 双侧乳突参考（需要M1, M2通道）
            mastoid_channels = ['M1', 'M2']
            if all(ch in raw.ch_names for ch in mastoid_channels):
                raw.set_eeg_reference(mastoid_channels, verbose=False)
                logger.info("✓ 已设置双侧乳突参考")
            else:
                logger.warning("⚠ 未找到M1/M2通道，使用平均参考代替")
                raw.set_eeg_reference('average', projection=False, verbose=False)
        
        elif ref_type == 'custom':
            if ref_channels is None:
                raise ValueError("custom参考需要指定ref_channels")
            raw.set_eeg_reference(ref_channels, verbose=False)
            logger.info(f"✓ 已设置自定义参考: {', '.join(ref_channels)}")
        
        else:
            raise ValueError(f"不支持的参考类型: {ref_type}")
        
        self.processing_log.append(f'set_reference_{ref_type}')
        
        return raw
    
    def run_ica(self, raw: mne.io.Raw, n_components: Optional[int] = None,
               method: str = 'fastica', random_state: int = 42,
               copy: bool = True) -> Tuple[mne.io.Raw, ICA]:
        """
        运行ICA分解
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        n_components : int, optional
            ICA成分数量
        method : str
            ICA方法 ('fastica', 'infomax', 'picard')
        random_state : int
            随机种子
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_ica : mne.io.Raw
            应用ICA后的数据
        ica : ICA
            ICA对象
        """
        if copy:
            raw = raw.copy()
        
        logger.info(f"开始ICA分解 (方法: {method})...")
        
        # 为ICA创建高通滤波数据（1Hz）
        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        
        # 创建ICA对象
        if n_components is None:
            n_components = min(25, len(raw.ch_names))
        
        ica = ICA(
            n_components=n_components,
            method=method,
            random_state=random_state,
            max_iter='auto'
        )
        
        # 拟合ICA
        ica.fit(raw_filt, verbose=False)
        
        logger.info(f"✓ ICA分解完成，提取 {n_components} 个成分")
        
        self.ica = ica
        self.processing_log.append(f'run_ica_{method}')
        
        return raw, ica
    
    def detect_artifact_components(self, raw: mne.io.Raw, ica: ICA,
                                  detect_eog: bool = True,
                                  detect_ecg: bool = True,
                                  eog_channels: Optional[List[str]] = None,
                                  ecg_channel: Optional[str] = None) -> Dict:
        """
        自动检测伪迹成分
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        ica : ICA
            ICA对象
        detect_eog : bool
            是否检测眼电伪迹
        detect_ecg : bool
            是否检测心电伪迹
        eog_channels : list, optional
            EOG通道
        ecg_channel : str, optional
            ECG通道
            
        Returns
        -------
        artifact_comps : dict
            检测到的伪迹成分
        """
        artifact_comps = {
            'eog': [],
            'ecg': [],
            'all': []
        }
        
        # 检测EOG成分
        if detect_eog:
            try:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw, 
                    ch_name=eog_channels,
                    threshold=3.0,
                    verbose=False
                )
                artifact_comps['eog'] = eog_indices
                logger.info(f"✓ 检测到 {len(eog_indices)} 个EOG成分: {eog_indices}")
            except Exception as e:
                logger.warning(f"⚠ EOG检测失败: {e}")
        
        # 检测ECG成分
        if detect_ecg:
            try:
                ecg_indices, ecg_scores = ica.find_bads_ecg(
                    raw,
                    ch_name=ecg_channel,
                    threshold='auto',
                    verbose=False
                )
                artifact_comps['ecg'] = ecg_indices
                logger.info(f"✓ 检测到 {len(ecg_indices)} 个ECG成分: {ecg_indices}")
            except Exception as e:
                logger.warning(f"⚠ ECG检测失败: {e}")
        
        # 合并所有伪迹成分
        all_comps = list(set(artifact_comps['eog'] + artifact_comps['ecg']))
        artifact_comps['all'] = sorted(all_comps)
        
        logger.info(f"总共检测到 {len(artifact_comps['all'])} 个伪迹成分")
        
        return artifact_comps
    
    def remove_artifacts_ica(self, raw: mne.io.Raw, ica: ICA,
                            exclude_components: Optional[List[int]] = None,
                            copy: bool = True) -> mne.io.Raw:
        """
        使用ICA移除伪迹
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        ica : ICA
            ICA对象
        exclude_components : list, optional
            要排除的成分索引
        copy : bool
            是否复制数据
            
        Returns
        -------
        raw_clean : mne.io.Raw
            去除伪迹后的数据
        """
        if copy:
            raw = raw.copy()
        
        if exclude_components is not None:
            ica.exclude = exclude_components
        
        if not ica.exclude:
            logger.warning("⚠ 未指定要排除的成分")
            return raw
        
        # 应用ICA
        ica.apply(raw, verbose=False)
        
        logger.info(f"✓ 已移除 {len(ica.exclude)} 个ICA成分: {ica.exclude}")
        self.processing_log.append('remove_artifacts_ica')
        
        return raw
    
    def create_epochs(self, raw: mne.io.Raw, events: Optional[np.ndarray] = None,
                     event_id: Optional[Dict] = None,
                     tmin: float = -0.5, tmax: float = 1.5,
                     baseline: Optional[Tuple] = (-0.2, 0),
                     reject: Optional[Dict] = None,
                     copy: bool = True) -> mne.Epochs:
        """
        按事件分段
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        events : np.ndarray, optional
            事件数组
        event_id : dict, optional
            事件ID字典
        tmin : float
            Epoch起始时间（相对事件，秒）
        tmax : float
            Epoch结束时间（相对事件，秒）
        baseline : tuple, optional
            基线校正时间窗
        reject : dict, optional
            拒绝阈值
        copy : bool
            是否复制数据
            
        Returns
        -------
        epochs : mne.Epochs
            分段数据
        """
        if events is None:
            # 从注释中提取事件
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        if event_id is None:
            # 使用所有事件
            event_id = {str(i): i for i in np.unique(events[:, 2])}
        
        # 默认拒绝阈值
        if reject is None:
            reject = dict(eeg=150e-6)  # 150 μV
        
        # 创建epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=True,
            verbose=False
        )
        
        n_epochs = len(epochs)
        n_dropped = epochs.drop_log_stats()
        
        logger.info(f"✓ 创建 {n_epochs} 个epochs")
        logger.info(f"  丢弃: {n_dropped} 个epochs")
        
        self.processing_log.append('create_epochs')
        
        return epochs
    
    def compute_source_space(self, raw: mne.io.Raw, 
                            subjects_dir: str,
                            subject: str = 'fsaverage',
                            spacing: str = 'oct6') -> Dict:
        """
        计算源空间
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        subjects_dir : str
            FreeSurfer subjects目录
        subject : str
            受试者名称
        spacing : str
            源空间采样密度
            
        Returns
        -------
        source_space_info : dict
            源空间信息
        """
        logger.info("开始计算源空间...")
        
        # 设置源空间
        src = mne.setup_source_space(
            subject,
            spacing=spacing,
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose=False
        )
        
        # 计算BEM模型
        conductivity = (0.3,)  # 单层模型
        model = mne.make_bem_model(
            subject=subject,
            ico=4,
            conductivity=conductivity,
            subjects_dir=subjects_dir,
            verbose=False
        )
        bem = mne.make_bem_solution(model, verbose=False)
        
        # 设置转换矩阵（如果有数字化数据）
        trans = None  # 需要用户提供或计算
        
        # 创建正向解
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans,
            src=src,
            bem=bem,
            meg=False,
            eeg=True,
            mindist=5.0,
            n_jobs=1,
            verbose=False
        )
        
        logger.info(f"✓ 源空间计算完成")
        logger.info(f"  源点数: {fwd['nsource']}")
        
        source_space_info = {
            'src': src,
            'bem': bem,
            'fwd': fwd,
            'trans': trans
        }
        
        self.processing_log.append('compute_source_space')
        
        return source_space_info
    
    def compute_inverse_operator(self, epochs: mne.Epochs,
                                fwd: mne.Forward,
                                noise_cov: Optional[mne.Covariance] = None,
                                method: str = 'dSPM',
                                lambda2: float = 1.0 / 9.0) -> Dict:
        """
        计算逆算子并重建源信号
        
        Parameters
        ----------
        epochs : mne.Epochs
            分段数据
        fwd : mne.Forward
            正向解
        noise_cov : mne.Covariance, optional
            噪声协方差矩阵
        method : str
            逆算法 ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
        lambda2 : float
            正则化参数
            
        Returns
        -------
        inverse_solution : dict
            逆解信息
        """
        logger.info(f"开始计算逆解 (方法: {method})...")
        
        # 计算噪声协方差
        if noise_cov is None:
            noise_cov = mne.compute_covariance(
                epochs,
                tmax=0,  # 基线期
                method='shrunk',
                verbose=False
            )
        
        # 创建逆算子
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            fwd,
            noise_cov,
            loose=0.2,
            depth=0.8,
            verbose=False
        )
        
        # 应用逆算子
        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs,
            inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=None,
            verbose=False
        )
        
        logger.info(f"✓ 逆解计算完成")
        logger.info(f"  源时间序列数: {len(stcs)}")
        
        inverse_solution = {
            'inverse_operator': inverse_operator,
            'stcs': stcs,
            'method': method,
            'lambda2': lambda2
        }
        
        self.processing_log.append(f'compute_inverse_{method}')
        
        return inverse_solution
    
    def extract_roi_timeseries(self, stcs: List,
                               labels_dict: Dict[str, mne.Label],
                               mode: str = 'mean') -> Dict[str, np.ndarray]:
        """
        从源空间提取ROI时间序列
        
        Parameters
        ----------
        stcs : list of SourceEstimate
            源时间序列列表
        labels_dict : dict
            ROI标签字典 {'M1': label_obj, 'SMA': label_obj}
        mode : str
            提取模式 ('mean', 'max', 'pca')
            
        Returns
        -------
        roi_timeseries : dict
            ROI时间序列字典
        """
        logger.info(f"提取ROI时间序列 (模式: {mode})...")
        
        roi_timeseries = {}
        
        for roi_name, label in labels_dict.items():
            # 提取每个epoch的ROI时间序列
            ts_list = []
            for stc in stcs:
                ts = stc.extract_label_time_course(
                    label,
                    src=None,  # 从stc中获取
                    mode=mode
                )
                ts_list.append(ts)
            
            # 堆叠为 (n_epochs, n_times)
            roi_timeseries[roi_name] = np.vstack(ts_list)
            
            logger.info(f"  {roi_name}: {roi_timeseries[roi_name].shape}")
        
        self.processing_log.append('extract_roi_timeseries')
        
        return roi_timeseries
    
    def get_processing_summary(self) -> str:
        """获取处理摘要"""
        if not self.processing_log:
            return "未执行任何处理步骤"
        
        summary = "EEG处理步骤:\n"
        for i, step in enumerate(self.processing_log, 1):
            summary += f"  {i}. {step}\n"
        
        return summary
