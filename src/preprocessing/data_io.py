"""
数据输入输出模块
支持多种格式: BrainVision (.vhdr/.eeg), FIF (.fif), EDF (.edf), BDF (.bdf)
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BIDSDataLoader:
    """
    BIDS格式数据加载器，支持多种文件格式
    """
    
    # 支持的格式及其扩展名
    SUPPORTED_FORMATS = {
        'brainvision': ['.vhdr', '.eeg', '.vmrk'],
        'fif': ['.fif'],
        'edf': ['.edf'],
        'bdf': ['.bdf'],
        'set': ['.set'],
        'cnt': ['.cnt']
    }
    
    def __init__(self, bids_root: str):
        """
        初始化数据加载器
        
        Parameters
        ----------
        bids_root : str
            BIDS根目录路径
        """
        self.bids_root = Path(bids_root)
        
    def _detect_format(self, directory: Path, base_filename: str) -> Optional[str]:
        """
        自动检测文件格式
        
        Parameters
        ----------
        directory : Path
            文件所在目录
        base_filename : str
            基础文件名（不含扩展名）
            
        Returns
        -------
        format_type : str or None
            检测到的格式类型
        """
        for format_name, extensions in self.SUPPORTED_FORMATS.items():
            for ext in extensions:
                file_path = directory / f"{base_filename}{ext}"
                if file_path.exists():
                    logger.info(f"检测到文件格式: {format_name} ({ext})")
                    return format_name
        
        return None
    
    def load_eeg_data(self, subject: str, session: str, task: str,
                      run: Optional[str] = None,
                      format_type: Optional[str] = None,
                      preload: bool = True) -> Tuple[mne.io.Raw, Dict]:
        """
        加载EEG数据（自动检测格式）
        
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
            强制指定格式类型，如果为None则自动检测
        preload : bool
            是否预加载数据到内存
            
        Returns
        -------
        raw : mne.io.Raw
            原始EEG数据
        metadata : dict
            元数据字典
        """
        # 构建目录和文件名
        eeg_dir = self.bids_root / subject / session / 'eeg'
        
        filename_parts = [subject, session, task]
        if run:
            filename_parts.append(run)
        base_filename = '_'.join(filename_parts)
        
        # 自动检测格式
        if format_type is None:
            format_type = self._detect_format(eeg_dir, f"{base_filename}_eeg")
            if format_type is None:
                raise FileNotFoundError(
                    f"未找到EEG文件: {eeg_dir / base_filename}_eeg.*"
                )
        
        # 根据格式加载数据
        raw = self._load_by_format(
            eeg_dir,
            f"{base_filename}_eeg",
            format_type,
            preload=preload
        )
        
        # 加载元数据
        metadata = self._load_metadata(eeg_dir, base_filename, 'eeg')
        
        logger.info(f"✓ 已加载EEG数据: {format_type}格式")
        logger.info(f"  采样率: {raw.info['sfreq']} Hz")
        logger.info(f"  通道数: {len(raw.ch_names)}")
        logger.info(f"  时长: {raw.times[-1]:.2f} 秒")
        
        return raw, metadata
    
    def load_lfp_data(self, subject: str, session: str, task: str,
                     run: Optional[str] = None,
                     format_type: Optional[str] = None,
                     preload: bool = True) -> Tuple[mne.io.Raw, Dict]:
        """
        加载LFP/iEEG数据（自动检测格式）
        
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
        preload : bool
            是否预加载
            
        Returns
        -------
        raw : mne.io.Raw
            原始LFP数据
        metadata : dict
            元数据字典
        """
        # 构建目录和文件名
        ieeg_dir = self.bids_root / subject / session / 'ieeg'
        
        filename_parts = [subject, session, task]
        if run:
            filename_parts.append(run)
        base_filename = '_'.join(filename_parts)
        
        # 自动检测格式
        if format_type is None:
            format_type = self._detect_format(ieeg_dir, f"{base_filename}_ieeg")
            if format_type is None:
                raise FileNotFoundError(
                    f"未找到LFP文件: {ieeg_dir / base_filename}_ieeg.*"
                )
        
        # 根据格式加载数据
        raw = self._load_by_format(
            ieeg_dir,
            f"{base_filename}_ieeg",
            format_type,
            preload=preload
        )
        
        # 加载元数据
        metadata = self._load_metadata(ieeg_dir, base_filename, 'ieeg')
        
        logger.info(f"✓ 已加载LFP数据: {format_type}格式")
        logger.info(f"  采样率: {raw.info['sfreq']} Hz")
        logger.info(f"  通道数: {len(raw.ch_names)}")
        logger.info(f"  时长: {raw.times[-1]:.2f} 秒")
        
        return raw, metadata
    
    def _load_by_format(self, directory: Path, base_filename: str,
                       format_type: str, preload: bool = True) -> mne.io.Raw:
        """
        根据格式加载原始数据
        
        Parameters
        ----------
        directory : Path
            文件所在目录
        base_filename : str
            基础文件名
        format_type : str
            格式类型
        preload : bool
            是否预加载
            
        Returns
        -------
        raw : mne.io.Raw
            原始数据对象
        """
        if format_type == 'brainvision':
            # BrainVision格式 (.vhdr)
            vhdr_file = directory / f"{base_filename}.vhdr"
            if not vhdr_file.exists():
                raise FileNotFoundError(f"未找到文件: {vhdr_file}")
            raw = mne.io.read_raw_brainvision(vhdr_file, preload=preload, verbose=False)
        
        elif format_type == 'fif':
            # FIF格式
            fif_file = directory / f"{base_filename}.fif"
            if not fif_file.exists():
                raise FileNotFoundError(f"未找到文件: {fif_file}")
            raw = mne.io.read_raw_fif(fif_file, preload=preload, verbose=False)
        
        elif format_type == 'edf':
            # EDF格式
            edf_file = directory / f"{base_filename}.edf"
            if not edf_file.exists():
                raise FileNotFoundError(f"未找到文件: {edf_file}")
            raw = mne.io.read_raw_edf(edf_file, preload=preload, verbose=False)
        
        elif format_type == 'bdf':
            # BDF格式
            bdf_file = directory / f"{base_filename}.bdf"
            if not bdf_file.exists():
                raise FileNotFoundError(f"未找到文件: {bdf_file}")
            raw = mne.io.read_raw_bdf(bdf_file, preload=preload, verbose=False)
        
        elif format_type == 'set':
            # EEGLAB SET格式
            set_file = directory / f"{base_filename}.set"
            if not set_file.exists():
                raise FileNotFoundError(f"未找到文件: {set_file}")
            raw = mne.io.read_raw_eeglab(set_file, preload=preload, verbose=False)
        
        elif format_type == 'cnt':
            # Neuroscan CNT格式
            cnt_file = directory / f"{base_filename}.cnt"
            if not cnt_file.exists():
                raise FileNotFoundError(f"未找到文件: {cnt_file}")
            raw = mne.io.read_raw_cnt(cnt_file, preload=preload, verbose=False)
        
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        return raw
    
    def _load_metadata(self, directory: Path, base_filename: str,
                      datatype: str) -> Dict:
        """
        加载元数据
        
        Parameters
        ----------
        directory : Path
            目录路径
        base_filename : str
            基础文件名
        datatype : str
            数据类型 ('eeg' 或 'ieeg')
            
        Returns
        -------
        metadata : dict
            元数据字典
        """
        import json
        import pandas as pd
        
        metadata = {}
        
        # JSON元数据
        json_file = directory / f"{base_filename}_{datatype}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata['json'] = json.load(f)
        
        # Events文件
        events_file = directory / f"{base_filename}_events.tsv"
        if events_file.exists():
            metadata['events'] = pd.read_csv(events_file, sep='\t')
        
        # Channels文件
        channels_file = directory / f"{base_filename}_channels.tsv"
        if channels_file.exists():
            metadata['channels'] = pd.read_csv(channels_file, sep='\t')
        
        # Electrodes文件（iEEG特有）
        if datatype == 'ieeg':
            electrodes_file = directory / f"{base_filename}_electrodes.tsv"
            if electrodes_file.exists():
                metadata['electrodes'] = pd.read_csv(electrodes_file, sep='\t')
        
        return metadata


class BIDSDataSaver:
    """
    BIDS格式数据保存器，支持多种格式
    """
    
    def __init__(self, bids_root: str, derivatives_name: str = 'preprocessing'):
        """
        初始化数据保存器
        
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
                             datatype: str,
                             run: Optional[str] = None,
                             description: Optional[str] = None,
                             format_type: str = 'brainvision',
                             overwrite: bool = True):
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
            数据类型 ('eeg' 或 'ieeg')
        run : str, optional
            运行编号
        description : str, optional
            描述标签
        format_type : str
            保存格式 ('brainvision', 'fif', 'edf')
        overwrite : bool
            是否覆盖已存在的文件
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
        
        base_filename = '_'.join(filename_parts)
        suffix = 'eeg' if datatype == 'eeg' else 'ieeg'
        
        # 根据格式保存
        if format_type == 'brainvision':
            output_path = output_dir / f"{base_filename}_{suffix}.vhdr"
            mne.export.export_raw(
                output_path,
                raw,
                fmt='brainvision',
                overwrite=overwrite,
                verbose=False
            )
        
        elif format_type == 'fif':
            output_path = output_dir / f"{base_filename}_{suffix}.fif"
            raw.save(output_path, overwrite=overwrite, verbose=False)
        
        elif format_type == 'edf':
            output_path = output_dir / f"{base_filename}_{suffix}.edf"
            mne.export.export_raw(
                output_path,
                raw,
                fmt='edf',
                overwrite=overwrite,
                verbose=False
            )
        
        else:
            raise ValueError(f"不支持的保存格式: {format_type}")
        
        logger.info(f"✓ 已保存预处理数据: {output_path}")
    
    def save_epochs(self, epochs: mne.Epochs,
                   subject: str, session: str, task: str,
                   datatype: str,
                   run: Optional[str] = None,
                   description: Optional[str] = None,
                   format_type: str = 'fif',
                   overwrite: bool = True):
        """
        保存epochs数据（推荐使用FIF格式）
        
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
        run : str, optional
            运行编号
        description : str, optional
            描述标签
        format_type : str
            保存格式（epochs通常保存为fif）
        overwrite : bool
            是否覆盖
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
        
        suffix = 'eeg' if datatype == 'eeg' else 'ieeg'
        filename_parts.append(suffix)
        
        base_filename = '_'.join(filename_parts)
        
        # 保存（epochs通常用FIF格式）
        output_path = output_dir / f"{base_filename}-epo.fif"
        epochs.save(output_path, overwrite=overwrite, verbose=False)
        
        logger.info(f"✓ 已保存epochs: {output_path}")

