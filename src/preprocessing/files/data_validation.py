"""
数据检查与验证模块
用于验证EEG和LFP数据的时间戳、采样率和同步精度
支持多种数据格式：BrainVision, FIF, EDF等
"""

import numpy as np
import mne
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """验证BIDS格式的EEG和LFP数据，支持多种格式"""
    
    def __init__(self, bids_root: str):
        """
        初始化数据验证器
        
        Parameters
        ----------
        bids_root : str
            BIDS根目录路径
        """
        self.bids_root = Path(bids_root)
        self.validation_results = {}
        
        # 导入数据加载器
        from .data_io import BIDSDataLoader
        self.loader = BIDSDataLoader(bids_root)
        
    def load_eeg_data(self, subject: str, session: str, task: str, 
                      run: Optional[str] = None,
                      format_type: Optional[str] = None) -> Tuple[mne.io.Raw, Dict]:
        """
        加载EEG数据及元数据（自动检测格式）
        
        Parameters
        ----------
        subject : str
            受试者ID (如 'sub-01')
        session : str
            会话ID (如 'ses-01')
        task : str
            任务名称 (如 'task-rest')
        run : str, optional
            运行编号 (如 'run-01')
        format_type : str, optional
            强制指定格式类型 ('brainvision', 'fif', 'edf'等)
            如果为None则自动检测
            
        Returns
        -------
        raw : mne.io.Raw
            原始EEG数据
        metadata : dict
            元数据字典
        """
        return self.loader.load_eeg_data(subject, session, task, run, format_type)
    
    def load_lfp_data(self, subject: str, session: str, task: str,
                     run: Optional[str] = None,
                     format_type: Optional[str] = None) -> Tuple[mne.io.Raw, Dict]:
        """
        加载LFP数据及元数据（自动检测格式）
        
        Parameters
        ----------
        subject : str
            受试者ID
        session : str
            会话ID
        task : str
            任务名称
        run : str, optional
            运行编号
        format_type : str, optional
            强制指定格式类型
            如果为None则自动检测
            
        Returns
        -------
        raw : mne.io.Raw
            原始LFP数据
        metadata : dict
            元数据字典
        """
        return self.loader.load_lfp_data(subject, session, task, run, format_type)
    
    def check_sampling_rates(self, eeg_raw: mne.io.Raw, 
                            lfp_raw: mne.io.Raw) -> Dict:
        """
        检查采样率
        
        Parameters
        ----------
        eeg_raw : mne.io.Raw
            EEG数据
        lfp_raw : mne.io.Raw
            LFP数据
            
        Returns
        -------
        results : dict
            采样率检查结果
        """
        eeg_sfreq = eeg_raw.info['sfreq']
        lfp_sfreq = lfp_raw.info['sfreq']
        
        results = {
            'eeg_sfreq': eeg_sfreq,
            'lfp_sfreq': lfp_sfreq,
            'match': np.isclose(eeg_sfreq, lfp_sfreq),
            'ratio': eeg_sfreq / lfp_sfreq if lfp_sfreq != 0 else None
        }
        
        if results['match']:
            logger.info(f"✓ 采样率匹配: {eeg_sfreq} Hz")
        else:
            logger.warning(f"⚠ 采样率不匹配: EEG={eeg_sfreq} Hz, LFP={lfp_sfreq} Hz")
            logger.info(f"  建议重采样至统一采样率")
        
        return results
    
    def check_time_alignment(self, eeg_raw: mne.io.Raw, lfp_raw: mne.io.Raw,
                            tolerance: float = 0.001) -> Dict:
        """
        检查时间对齐
        
        Parameters
        ----------
        eeg_raw : mne.io.Raw
            EEG数据
        lfp_raw : mne.io.Raw
            LFP数据
        tolerance : float
            时间差容差（秒）
            
        Returns
        -------
        results : dict
            时间对齐检查结果
        """
        eeg_tmin = eeg_raw.times[0]
        eeg_tmax = eeg_raw.times[-1]
        lfp_tmin = lfp_raw.times[0]
        lfp_tmax = lfp_raw.times[-1]
        
        time_offset = eeg_tmin - lfp_tmin
        duration_diff = (eeg_tmax - eeg_tmin) - (lfp_tmax - lfp_tmin)
        
        results = {
            'eeg_duration': eeg_tmax - eeg_tmin,
            'lfp_duration': lfp_tmax - lfp_tmin,
            'time_offset': time_offset,
            'duration_diff': duration_diff,
            'aligned': abs(time_offset) < tolerance and abs(duration_diff) < tolerance
        }
        
        if results['aligned']:
            logger.info(f"✓ 时间对齐良好 (偏移 {time_offset*1000:.3f} ms)")
        else:
            logger.warning(f"⚠ 时间对齐存在问题:")
            logger.warning(f"  时间偏移: {time_offset*1000:.3f} ms")
            logger.warning(f"  时长差异: {duration_diff*1000:.3f} ms")
        
        return results
    
    def check_events_synchronization(self, eeg_metadata: Dict, 
                                    lfp_metadata: Dict) -> Dict:
        """
        检查事件同步
        
        Parameters
        ----------
        eeg_metadata : dict
            EEG元数据
        lfp_metadata : dict
            LFP元数据
            
        Returns
        -------
        results : dict
            事件同步检查结果
        """
        results = {
            'eeg_has_events': 'events' in eeg_metadata,
            'lfp_has_events': 'events' in lfp_metadata,
            'events_match': False,
            'event_count_diff': None
        }
        
        if results['eeg_has_events'] and results['lfp_has_events']:
            eeg_events = eeg_metadata['events']
            lfp_events = lfp_metadata.get('events', pd.DataFrame())
            
            # 检查事件数量
            eeg_event_count = len(eeg_events)
            lfp_event_count = len(lfp_events) if not lfp_events.empty else 0
            
            results['eeg_event_count'] = eeg_event_count
            results['lfp_event_count'] = lfp_event_count
            results['event_count_diff'] = abs(eeg_event_count - lfp_event_count)
            results['events_match'] = eeg_event_count == lfp_event_count
            
            if results['events_match']:
                logger.info(f"✓ 事件数量匹配: {eeg_event_count} 个事件")
                
                # 检查事件时间对齐
                if 'onset' in eeg_events.columns and 'onset' in lfp_events.columns:
                    time_diffs = np.abs(eeg_events['onset'].values - 
                                       lfp_events['onset'].values)
                    results['max_time_diff'] = np.max(time_diffs)
                    results['mean_time_diff'] = np.mean(time_diffs)
                    
                    logger.info(f"  事件时间差: 平均 {results['mean_time_diff']*1000:.3f} ms, "
                              f"最大 {results['max_time_diff']*1000:.3f} ms")
            else:
                logger.warning(f"⚠ 事件数量不匹配: EEG={eeg_event_count}, LFP={lfp_event_count}")
        else:
            logger.warning("⚠ 缺少事件数据")
        
        return results
    
    def validate_metadata_consistency(self, eeg_metadata: Dict,
                                     lfp_metadata: Dict) -> Dict:
        """
        验证元数据一致性
        
        Parameters
        ----------
        eeg_metadata : dict
            EEG元数据
        lfp_metadata : dict
            LFP元数据
            
        Returns
        -------
        results : dict
            元数据一致性检查结果
        """
        results = {
            'consistent': True,
            'issues': []
        }
        
        # 检查JSON元数据
        if 'json' in eeg_metadata and 'json' in lfp_metadata:
            eeg_json = eeg_metadata['json']
            lfp_json = lfp_metadata['json']
            
            # 检查任务名称
            if eeg_json.get('TaskName') != lfp_json.get('TaskName'):
                results['consistent'] = False
                results['issues'].append(
                    f"任务名称不一致: EEG={eeg_json.get('TaskName')}, "
                    f"LFP={lfp_json.get('TaskName')}"
                )
            
            # 检查采样率记录
            if 'SamplingFrequency' in eeg_json and 'SamplingFrequency' in lfp_json:
                if not np.isclose(eeg_json['SamplingFrequency'], 
                                 lfp_json['SamplingFrequency']):
                    results['issues'].append(
                        f"元数据中采样率不一致: EEG={eeg_json['SamplingFrequency']}, "
                        f"LFP={lfp_json['SamplingFrequency']}"
                    )
        
        if results['consistent']:
            logger.info("✓ 元数据一致性良好")
        else:
            logger.warning("⚠ 元数据存在不一致:")
            for issue in results['issues']:
                logger.warning(f"  - {issue}")
        
        return results
    
    def run_full_validation(self, subject: str, session: str, task: str,
                          run: Optional[str] = None) -> Dict:
        """
        运行完整验证流程
        
        Parameters
        ----------
        subject : str
            受试者ID
        session : str
            会话ID
        task : str
            任务名称
        run : str, optional
            运行编号
            
        Returns
        -------
        results : dict
            完整验证结果
        """
        logger.info("="*60)
        logger.info("开始数据验证")
        logger.info("="*60)
        
        # 加载数据
        eeg_raw, eeg_metadata = self.load_eeg_data(subject, session, task, run)
        lfp_raw, lfp_metadata = self.load_lfp_data(subject, session, task, run)
        
        # 执行各项检查
        results = {
            'sampling_rates': self.check_sampling_rates(eeg_raw, lfp_raw),
            'time_alignment': self.check_time_alignment(eeg_raw, lfp_raw),
            'events_sync': self.check_events_synchronization(eeg_metadata, lfp_metadata),
            'metadata_consistency': self.validate_metadata_consistency(eeg_metadata, lfp_metadata)
        }
        
        # 总结
        logger.info("="*60)
        logger.info("验证完成")
        logger.info("="*60)
        
        all_passed = (
            results['sampling_rates']['match'] and
            results['time_alignment']['aligned'] and
            results['events_sync']['events_match'] and
            results['metadata_consistency']['consistent']
        )
        
        if all_passed:
            logger.info("✓ 所有验证通过，数据质量良好")
        else:
            logger.warning("⚠ 存在需要注意的问题，请查看详细结果")
        
        self.validation_results = results
        return results
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        生成验证报告
        
        Parameters
        ----------
        output_path : str, optional
            报告输出路径
            
        Returns
        -------
        report : str
            验证报告文本
        """
        if not self.validation_results:
            return "未执行验证，请先运行 run_full_validation()"
        
        report_lines = [
            "EEG-LFP 数据验证报告",
            "="*60,
            "",
            "1. 采样率检查",
            f"   EEG采样率: {self.validation_results['sampling_rates']['eeg_sfreq']} Hz",
            f"   LFP采样率: {self.validation_results['sampling_rates']['lfp_sfreq']} Hz",
            f"   状态: {'通过' if self.validation_results['sampling_rates']['match'] else '不匹配'}",
            "",
            "2. 时间对齐检查",
            f"   时间偏移: {self.validation_results['time_alignment']['time_offset']*1000:.3f} ms",
            f"   时长差异: {self.validation_results['time_alignment']['duration_diff']*1000:.3f} ms",
            f"   状态: {'通过' if self.validation_results['time_alignment']['aligned'] else '需要对齐'}",
            "",
            "3. 事件同步检查",
        ]
        
        events = self.validation_results['events_sync']
        if events.get('events_match'):
            report_lines.append(f"   事件数量: {events.get('eeg_event_count')} (匹配)")
            if 'mean_time_diff' in events:
                report_lines.append(f"   平均时间差: {events['mean_time_diff']*1000:.3f} ms")
        else:
            report_lines.append(f"   EEG事件: {events.get('eeg_event_count', 'N/A')}")
            report_lines.append(f"   LFP事件: {events.get('lfp_event_count', 'N/A')}")
        
        report_lines.extend([
            "",
            "4. 元数据一致性",
            f"   状态: {'通过' if self.validation_results['metadata_consistency']['consistent'] else '存在问题'}",
        ])
        
        if self.validation_results['metadata_consistency']['issues']:
            report_lines.append("   问题:")
            for issue in self.validation_results['metadata_consistency']['issues']:
                report_lines.append(f"   - {issue}")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存至: {output_path}")
        
        return report
