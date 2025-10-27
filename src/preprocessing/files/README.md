# EEG-LFP 预处理流程

完整的EEG与LFP同步数据预处理工具包，用于跨模态神经信号分析。

## 功能特性

### 1️⃣ 数据检查与验证
- 自动验证BIDS格式数据完整性
- 检查采样率、时间对齐和事件同步
- 生成详细的验证报告

### 2️⃣ 通用信号清洗
- 去趋势与去直流偏移
- 带通滤波（EEG: 1-100 Hz, LFP: 1-200 Hz）
- 陷波滤波（50/60 Hz工频及谐波）
- 重采样至统一采样率（默认1000 Hz）

### 3️⃣ EEG专用预处理
- **坏导检测与插值**
  - 基于相关性的坏导检测
  - 基于方差的坏导检测
  - RANSAC方法
- **重参考**
  - 平均参考
  - 乳突参考
  - 自定义参考
- **ICA去伪迹**
  - FastICA / Infomax / Picard算法
  - 自动检测眼电(EOG)和心电(ECG)伪迹
  - 可视化ICA成分
- **事件分段（Epoching）**
  - 基线校正
  - 伪迹拒绝
- **源空间重建（可选）**
  - 计算正向解（Forward solution）
  - 逆解算法（dSPM, sLORETA, eLORETA）
  - ROI时间序列提取

### 4️⃣ LFP专用预处理
- **刺激伪迹去除**
  - 模板减法
  - Blanking方法
  - 插值方法
- **电极管理**
  - 自动解析接触点和左右分区
  - 双极参考
- **信号增强**
  - 共平均参考(CAR)
  - 小波去噪
  - Savitzky-Golay平滑
  - 异常值检测

### 5️⃣ 联合处理
- **时间对齐**
  - 自动裁剪至重叠时间窗
  - Epochs同步
- **频段分解**
  - Delta (1-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-100 Hz)
- **信号标准化**
  - Z-score标准化
  - Min-Max标准化
  - 鲁棒标准化
- **连接性分析准备**
  - 合并EEG和LFP数据
  - 准备跨通道分析数据结构

### 6️⃣ 质量控制
- **可视化**
  - 原始数据波形
  - 功率谱密度(PSD)对比
  - 信号处理前后对比
  - Epochs质量检查
  - ICA成分可视化
  - 频段分解可视化
  - 时频分析图
- **质量指标**
  - 信噪比(SNR)计算
  - 坏导统计
  - Epochs保留率
- **报告生成**
  - 自动生成质量控制报告
  - 处理步骤记录
- **BIDS Derivatives**
  - 自动保存为BIDS derivatives格式
  - 元数据记录

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd eeg_lfp_preprocessing

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from preprocessing.data_validation import DataValidator
from preprocessing.signal_cleaning import EEGCleaner, LFPCleaner
from preprocessing.eeg_preprocessing import EEGPreprocessor
from preprocessing.lfp_preprocessing import LFPPreprocessor
from preprocessing.joint_preprocessing import JointPreprocessor
from preprocessing.quality_control import QualityControl, BIDSDerivativesSaver

# 1. 数据验证
validator = DataValidator('/path/to/bids/root')
results = validator.run_full_validation('sub-01', 'ses-01', 'task-rest')

# 2. 加载数据
eeg_raw, eeg_meta = validator.load_eeg_data('sub-01', 'ses-01', 'task-rest')
lfp_raw, lfp_meta = validator.load_lfp_data('sub-01', 'ses-01', 'task-rest')

# 3. 清洗
eeg_cleaner = EEGCleaner()
eeg_clean = eeg_cleaner.apply_eeg_cleaning(eeg_raw, target_sfreq=1000.0)

lfp_cleaner = LFPCleaner()
lfp_clean = lfp_cleaner.apply_lfp_cleaning(lfp_raw, target_sfreq=1000.0)

# 4. EEG预处理
eeg_prep = EEGPreprocessor()
bad_channels = eeg_prep.detect_bad_channels(eeg_clean)
eeg_clean = eeg_prep.interpolate_bad_channels(eeg_clean)
eeg_clean = eeg_prep.set_reference(eeg_clean, ref_type='average')
eeg_clean, ica = eeg_prep.run_ica(eeg_clean)
artifacts = eeg_prep.detect_artifact_components(eeg_clean, ica)
eeg_clean = eeg_prep.remove_artifacts_ica(eeg_clean, ica, artifacts['all'])

# 5. LFP预处理
lfp_prep = LFPPreprocessor()
electrode_info = lfp_prep.parse_electrode_contacts(lfp_clean)
lfp_clean = lfp_prep.apply_bipolar_reference(lfp_clean)
lfp_clean = lfp_prep.enhance_snr(lfp_clean, method='car')

# 6. 联合处理
joint_prep = JointPreprocessor()
eeg_aligned, lfp_aligned = joint_prep.align_time_windows(eeg_clean, lfp_clean)

# 7. 创建epochs
events, event_id = mne.events_from_annotations(eeg_aligned)
eeg_epochs = eeg_prep.create_epochs(eeg_aligned, events, event_id)
lfp_epochs = mne.Epochs(lfp_aligned, events, event_id, 
                        tmin=-0.5, tmax=1.5, preload=True)

# 8. 频段分解
freq_bands = {'theta': (4, 8), 'beta': (13, 30), 'gamma': (30, 100)}
eeg_bands = joint_prep.extract_frequency_bands(eeg_aligned, bands=freq_bands)
lfp_bands = joint_prep.extract_frequency_bands(lfp_aligned, bands=freq_bands)

# 9. 质量控制
qc = QualityControl(output_dir='./qc_outputs')
qc.plot_psd_comparison(eeg_raw, eeg_clean, save_path='eeg_psd.png')
qc.compute_snr(eeg_clean)
report = qc.generate_qc_report(eeg_prep.processing_log)

# 10. 保存结果
saver = BIDSDerivativesSaver(bids_root='/path/to/bids')
saver.save_preprocessed_raw(eeg_clean, 'sub-01', 'ses-01', 'task-rest', 
                            'eeg', description='clean')
saver.save_epochs(eeg_epochs, 'sub-01', 'ses-01', 'task-rest', 'eeg')
```

## 完整示例

详细的使用示例请参见：
- `notebooks/01_complete_preprocessing.ipynb` - 完整预处理流程

## 项目结构

```
eeg_lfp_preprocessing/
├── src/
│   └── preprocessing/
│       ├── data_validation.py      # 数据验证模块
│       ├── signal_cleaning.py      # 信号清洗模块
│       ├── eeg_preprocessing.py    # EEG预处理模块
│       ├── lfp_preprocessing.py    # LFP预处理模块
│       ├── joint_preprocessing.py  # 联合处理模块
│       └── quality_control.py      # 质量控制模块
├── notebooks/
│   └── 01_complete_preprocessing.ipynb  # 完整示例
├── tests/                          # 单元测试
├── requirements.txt                # 依赖列表
└── README.md                       # 本文件
```

## 依赖项

主要依赖：
- `mne >= 1.5.0` - 神经电生理数据分析
- `numpy >= 1.21.0` - 数值计算
- `scipy >= 1.7.0` - 科学计算
- `matplotlib >= 3.5.0` - 可视化
- `pandas >= 1.3.0` - 数据处理
- `pywavelets` - 小波分析（可选）

完整依赖列表见 `requirements.txt`

## 输入数据格式

期望的BIDS格式结构：

```
bids_root/
├── sub-01/
│   └── ses-01/
│       ├── eeg/
│       │   ├── sub-01_ses-01_task-rest_eeg.fif
│       │   ├── sub-01_ses-01_task-rest_eeg.json
│       │   ├── sub-01_ses-01_task-rest_events.tsv
│       │   └── sub-01_ses-01_task-rest_channels.tsv
│       └── ieeg/
│           ├── sub-01_ses-01_task-rest_ieeg.fif
│           ├── sub-01_ses-01_task-rest_ieeg.json
│           ├── sub-01_ses-01_task-rest_channels.tsv
│           └── sub-01_ses-01_task-rest_electrodes.tsv
└── derivatives/
    └── preprocessing/
        └── sub-01/
            └── ses-01/
                ├── eeg/
                │   ├── sub-01_ses-01_task-rest_desc-clean_eeg.fif
                │   └── sub-01_ses-01_task-rest_desc-clean_epo-epo.fif
                └── ieeg/
                    ├── sub-01_ses-01_task-rest_desc-clean_ieeg.fif
                    └── sub-01_ses-01_task-rest_desc-clean_epo-epo.fif
```

## 输出文件

预处理后的数据保存在 `derivatives/preprocessing/` 目录下：

1. **预处理后的原始数据**: `*_desc-clean_eeg.fif` / `*_desc-clean_ieeg.fif`
2. **Epochs数据**: `*_desc-clean_epo-epo.fif`
3. **处理元数据**: `preprocessing_info.json`
4. **质量控制报告**: `quality_control_report.txt`
5. **质量控制图像**: `qc_outputs/*.png`

## 下一步分析

预处理完成后，可以进行：

1. **连接性分析**
   - 相干性(Coherence)
   - 相位锁定值(PLV)
   - 格兰杰因果(Granger Causality)

2. **时频分析**
   - 短时傅里叶变换(STFT)
   - 小波变换
   - 希尔伯特-黄变换(HHT)

3. **相位-振幅耦合(PAC)**
   - 跨频段耦合
   - EEG-LFP耦合

4. **机器学习**
   - 特征提取
   - 分类/回归
   - 解码分析

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证

## 引用

如果您在研究中使用了本工具包，请引用：

```bibtex
@software{eeg_lfp_preprocessing,
  title = {EEG-LFP Preprocessing Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/eeg_lfp_preprocessing}
}
```

## 联系方式

如有问题或建议，请：
- 提交 Issue
- 发送邮件至: your.email@example.com

## 致谢

本工具包基于以下优秀项目：
- [MNE-Python](https://mne.tools/)
- [BIDS](https://bids.neuroimaging.io/)
- [SciPy](https://scipy.org/)
