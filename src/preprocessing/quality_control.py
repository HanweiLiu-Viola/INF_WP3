"""
质量控制与可视化模块
用于检查预处理效果、生成质量报告和可视化
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class QualityControl:
    """质量控制器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化质量控制器
        
        Parameters
        ----------
        output_dir : str, optional
            输出目录路径
        """
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.qc_results = {}
    
    def plot_raw_data(self, raw: mne.io.Raw, title: str = 'Raw Data',
                     duration: float = 10.0, n_channels: int = 20,
                     save_path: Optional[str] = None):
        """
        绘制原始数据
        
        Parameters
        ----------
        raw : mne.io.Raw
            原始数据
        title : str
            图标题
        duration : float
            显示时长（秒）
        n_channels : int
            显示通道数
        save_path : str, optional
            保存路径
        """
        fig = raw.plot(
            duration=duration,
            n_channels=n_channels,
            scalings='auto',
            title=title,
            show=False
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存图像: {save_path}")
        
        plt.close(fig)
    
    def plot_psd_comparison(self, raw_before: mne.io.Raw, 
                           raw_after: mne.io.Raw,
                           fmin: float = 0.5, fmax: float = 150.0,
                           save_path: Optional[str] = None):
        """
        绘制处理前后的功率谱对比
        
        Parameters
        ----------
        raw_before : mne.io.Raw
            处理前的数据
        raw_after : mne.io.Raw
            处理后的数据
        fmin : float
            最小频率
        fmax : float
            最大频率
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 计算功率谱
        spectrum_before = raw_before.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        spectrum_after = raw_after.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        
        # 绘制处理前
        spectrum_before.plot(axes=axes[0], show=False)
        axes[0].set_title('处理前功率谱')
        axes[0].set_xlabel('频率 (Hz)')
        axes[0].set_ylabel('功率 (dB)')
        
        # 绘制处理后
        spectrum_after.plot(axes=axes[1], show=False)
        axes[1].set_title('处理后功率谱')
        axes[1].set_xlabel('频率 (Hz)')
        axes[1].set_ylabel('功率 (dB)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存PSD对比图: {save_path}")
        
        plt.close(fig)
    
    def plot_signal_comparison(self, raw_before: mne.io.Raw,
                              raw_after: mne.io.Raw,
                              duration: float = 5.0,
                              channel_idx: int = 0,
                              save_path: Optional[str] = None):
        """
        绘制处理前后的信号对比
        
        Parameters
        ----------
        raw_before : mne.io.Raw
            处理前的数据
        raw_after : mne.io.Raw
            处理后的数据
        duration : float
            显示时长
        channel_idx : int
            通道索引
        save_path : str, optional
            保存路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # 获取数据
        sfreq = raw_before.info['sfreq']
        n_samples = int(duration * sfreq)
        
        data_before = raw_before.get_data()[channel_idx, :n_samples]
        data_after = raw_after.get_data()[channel_idx, :n_samples]
        times = np.arange(n_samples) / sfreq
        
        ch_name = raw_before.ch_names[channel_idx]
        
        # 绘制处理前
        axes[0].plot(times, data_before, 'b-', linewidth=0.5)
        axes[0].set_title(f'处理前 - {ch_name}')
        axes[0].set_ylabel('振幅')
        axes[0].grid(True, alpha=0.3)
        
        # 绘制处理后
        axes[1].plot(times, data_after, 'r-', linewidth=0.5)
        axes[1].set_title(f'处理后 - {ch_name}')
        axes[1].set_xlabel('时间 (s)')
        axes[1].set_ylabel('振幅')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存信号对比图: {save_path}")
        
        plt.close(fig)
    
    def plot_epochs_quality(self, epochs: mne.Epochs,
                           save_path: Optional[str] = None):
        """
        绘制epochs质量概览
        
        Parameters
        ----------
        epochs : mne.Epochs
            分段数据
        save_path : str, optional
            保存路径
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Epochs平均
        ax1 = fig.add_subplot(gs[0, :])
        epochs.average().plot(axes=ax1, show=False)
        ax1.set_title('Epochs平均')
        
        # 2. 图像图
        ax2 = fig.add_subplot(gs[1, 0])
        epochs.plot_image(picks=[0], axes=ax2, show=False)
        ax2.set_title('Epochs图像')
        
        # 3. PSD
        ax3 = fig.add_subplot(gs[1, 1])
        epochs.compute_psd().plot(axes=ax3, show=False)
        ax3.set_title('功率谱密度')
        
        # 4. Drop log
        ax4 = fig.add_subplot(gs[2, 0])
        n_good = len(epochs)
        n_bad = len(epochs.drop_log) - n_good
        ax4.bar(['保留', '丢弃'], [n_good, n_bad], color=['green', 'red'])
        ax4.set_ylabel('数量')
        ax4.set_title('Epochs质量统计')
        
        # 5. 拓扑图（如果有位置信息）
        if epochs.info.get('dig') is not None:
            ax5 = fig.add_subplot(gs[2, 1])
            epochs.average().plot_topomap(times=[0], axes=ax5, show=False)
            ax5.set_title('拓扑图')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存epochs质量图: {save_path}")
        
        plt.close(fig)
    
    def plot_ica_components(self, ica: mne.preprocessing.ICA,
                           raw: mne.io.Raw,
                           n_components: int = 10,
                           save_path: Optional[str] = None):
        """
        绘制ICA成分
        
        Parameters
        ----------
        ica : ICA
            ICA对象
        raw : mne.io.Raw
            原始数据
        n_components : int
            显示的成分数量
        save_path : str, optional
            保存路径
        """
        fig = ica.plot_components(
            picks=list(range(min(n_components, ica.n_components_))),
            show=False
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存ICA成分图: {save_path}")
        
        plt.close(fig)
    
    def plot_frequency_bands(self, band_data: Dict[str, mne.io.Raw],
                            channel_idx: int = 0,
                            duration: float = 5.0,
                            save_path: Optional[str] = None):
        """
        绘制频段分解结果
        
        Parameters
        ----------
        band_data : dict
            频段数据字典
        channel_idx : int
            通道索引
        duration : float
            显示时长
        save_path : str, optional
            保存路径
        """
        n_bands = len(band_data)
        fig, axes = plt.subplots(n_bands, 1, figsize=(12, 2*n_bands), sharex=True)
        
        if n_bands == 1:
            axes = [axes]
        
        for ax, (band_name, raw_band) in zip(axes, band_data.items()):
            sfreq = raw_band.info['sfreq']
            n_samples = int(duration * sfreq)
            
            data = raw_band.get_data()[channel_idx, :n_samples]
            times = np.arange(n_samples) / sfreq
            
            ax.plot(times, data, linewidth=0.5)
            ax.set_ylabel(f'{band_name}\n振幅')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间 (s)')
        axes[0].set_title(f'频段分解 - 通道 {channel_idx}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存频段分解图: {save_path}")
        
        plt.close(fig)
    
    def plot_time_frequency(self, tfr: mne.time_frequency.AverageTFR,
                           channel_idx: int = 0,
                           save_path: Optional[str] = None):
        """
        绘制时频图
        
        Parameters
        ----------
        tfr : AverageTFR
            时频数据
        channel_idx : int
            通道索引
        save_path : str, optional
            保存路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        tfr.plot(
            picks=[channel_idx],
            baseline=None,
            mode='mean',
            axes=ax,
            show=False,
            colorbar=True
        )
        
        ax.set_title('时频分析')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('频率 (Hz)')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ 已保存时频图: {save_path}")
        
        plt.close(fig)
    
    def compute_snr(self, raw: mne.io.Raw, 
                   signal_window: Tuple[float, float] = (0.0, 1.0),
                   noise_window: Tuple[float, float] = (-0.5, 0.0)) -> Dict:
        """
        计算信噪比
        
        Parameters
        ----------
        raw : mne.io.Raw
            数据
        signal_window : tuple
            信号窗口
        noise_window : tuple
            噪声窗口
            
        Returns
        -------
        snr_dict : dict
            各通道的SNR
        """
        logger.info("计算信噪比...")
        
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # 计算窗口索引
        sig_start = int(signal_window[0] * sfreq)
        sig_end = int(signal_window[1] * sfreq)
        noise_start = int(noise_window[0] * sfreq)
        noise_end = int(noise_window[1] * sfreq)
        
        snr_dict = {}
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            # 计算信号和噪声功率
            signal_power = np.mean(data[ch_idx, sig_start:sig_end] ** 2)
            noise_power = np.mean(data[ch_idx, noise_start:noise_end] ** 2)
            
            # 计算SNR (dB)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_dict[ch_name] = snr_db
        
        # 统计
        snr_values = list(snr_dict.values())
        logger.info(f"✓ SNR统计:")
        logger.info(f"  平均: {np.mean(snr_values):.2f} dB")
        logger.info(f"  范围: {np.min(snr_values):.2f} - {np.max(snr_values):.2f} dB")
        
        self.qc_results['snr'] = snr_dict
        
        return snr_dict
    
    def generate_qc_report(self, preprocessing_steps: List[str],
                          metrics: Optional[Dict] = None,
                          save_path: Optional[str] = None) -> str:
        """
        生成质量控制报告
        
        Parameters
        ----------
        preprocessing_steps : list
            预处理步骤列表
        metrics : dict, optional
            质量指标
        save_path : str, optional
            保存路径
            
        Returns
        -------
        report : str
            报告文本
        """
        report_lines = [
            "="*70,
            "EEG-LFP 预处理质量控制报告",
            "="*70,
            "",
            "1. 预处理步骤",
            "-" * 70
        ]
        
        for i, step in enumerate(preprocessing_steps, 1):
            report_lines.append(f"  {i}. {step}")
        
        report_lines.extend([
            "",
            "2. 质量指标",
            "-" * 70
        ])
        
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key}: {value:.3f}")
                else:
                    report_lines.append(f"  {key}: {value}")
        
        if 'snr' in self.qc_results:
            snr_values = list(self.qc_results['snr'].values())
            report_lines.extend([
                "",
                "3. 信噪比统计",
                "-" * 70,
                f"  平均SNR: {np.mean(snr_values):.2f} dB",
                f"  最小SNR: {np.min(snr_values):.2f} dB",
                f"  最大SNR: {np.max(snr_values):.2f} dB",
                f"  标准差: {np.std(snr_values):.2f} dB"
            ])
        
        report_lines.extend([
            "",
            "="*70,
            "报告生成完成",
            "="*70
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✓ 质量报告已保存: {save_path}")
        
        return report


class BIDSDerivativesSaver:
    """BIDS derivatives保存器"""
    
    def __init__(self, bids_root: str, derivatives_name: str = 'preprocessing'):
        """
        初始化保存器
        
        Parameters
        ----------
        bids_root : str
            BIDS根目录
        derivatives_name : str
            Derivatives名称
        """
        self.bids_root = Path(bids_root)
        self.derivatives_dir = self.bids_root / 'derivatives' / derivatives_name
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Derivatives目录: {self.derivatives_dir}")
    
    def save_preprocessed_raw(self, raw: mne.io.Raw,
                             subject: str, session: str, task: str,
                             datatype: str, suffix: str = 'eeg',
                             run: Optional[str] = None,
                             description: Optional[str] = None):
        """
        保存预处理后的原始数据
        
        Parameters
        ----------
        raw : mne.io.Raw
            预处理后的数据
        subject : str
            受试者ID
        session : str
            会话ID
        task : str
            任务名称
        datatype : str
            数据类型 ('eeg', 'ieeg')
        suffix : str
            文件后缀
        run : str, optional
            运行编号
        description : str, optional
            描述标签
        """
        # 创建目录结构
        output_dir = self.derivatives_dir / subject / session / datatype
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建文件名
        filename_parts = [subject, session, task]
        if run:
            filename_parts.append(run)
        if description:
            filename_parts.append(f'desc-{description}')
        filename_parts.append(suffix)
        
        filename = '_'.join(filename_parts) + '.fif'
        output_path = output_dir / filename
        
        # 保存数据
        raw.save(output_path, overwrite=True, verbose=False)
        
        logger.info(f"✓ 已保存预处理数据: {output_path}")
    
    def save_epochs(self, epochs: mne.Epochs,
                   subject: str, session: str, task: str,
                   datatype: str, suffix: str = 'epo',
                   run: Optional[str] = None,
                   description: Optional[str] = None):
        """
        保存epochs数据
        
        Parameters
        ----------
        epochs : mne.Epochs
            分段数据
        subject : str
            受试者ID
        session : str
            会话ID
        task : str
            任务名称
        datatype : str
            数据类型
        suffix : str
            文件后缀
        run : str, optional
            运行编号
        description : str, optional
            描述标签
        """
        # 创建目录
        output_dir = self.derivatives_dir / subject / session / datatype
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建文件名
        filename_parts = [subject, session, task]
        if run:
            filename_parts.append(run)
        if description:
            filename_parts.append(f'desc-{description}')
        filename_parts.append(suffix)
        
        filename = '_'.join(filename_parts) + '-epo.fif'
        output_path = output_dir / filename
        
        # 保存
        epochs.save(output_path, overwrite=True, verbose=False)
        
        logger.info(f"✓ 已保存epochs: {output_path}")
    
    def save_derivative_metadata(self, processing_info: Dict,
                                subject: str, session: str):
        """
        保存处理元数据
        
        Parameters
        ----------
        processing_info : dict
            处理信息
        subject : str
            受试者ID
        session : str
            会话ID
        """
        import json
        
        output_dir = self.derivatives_dir / subject / session
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_dir / 'preprocessing_info.json'
        
        with open(metadata_file, 'w') as f:
            json.dump(processing_info, f, indent=2)
        
        logger.info(f"✓ 已保存处理元数据: {metadata_file}")
