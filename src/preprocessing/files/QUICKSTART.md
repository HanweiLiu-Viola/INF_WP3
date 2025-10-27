# EEG-LFP 预处理快速入门指南

## 📋 目录结构

```
eeg_lfp_preprocessing/
├── src/preprocessing/           # 核心模块
│   ├── __init__.py             # 包初始化
│   ├── data_validation.py      # 数据验证
│   ├── signal_cleaning.py      # 信号清洗
│   ├── eeg_preprocessing.py    # EEG预处理
│   ├── lfp_preprocessing.py    # LFP预处理
│   ├── joint_preprocessing.py  # 联合处理
│   └── quality_control.py      # 质量控制
├── notebooks/                   # 示例notebooks
│   └── 01_complete_preprocessing.ipynb
├── tests/                       # 单元测试
├── README.md                    # 详细文档
├── requirements.txt             # 依赖列表
└── QUICKSTART.md               # 本文件
```

## 🚀 快速开始（5分钟）

### 1. 安装

```bash
cd eeg_lfp_preprocessing
pip install -r requirements.txt
```

### 2. 最简单的使用示例

```python
import sys
sys.path.append('src')

from preprocessing import (
    DataValidator, EEGCleaner, LFPCleaner,
    EEGPreprocessor, LFPPreprocessor, 
    JointPreprocessor, QualityControl
)

# 设置路径
bids_root = '/path/to/your/bids/data'

# 1. 验证数据
validator = DataValidator(bids_root)
validator.run_full_validation('sub-01', 'ses-01', 'task-rest')

# 2. 加载
eeg_raw, _ = validator.load_eeg_data('sub-01', 'ses-01', 'task-rest')
lfp_raw, _ = validator.load_lfp_data('sub-01', 'ses-01', 'task-rest')

# 3. 一键清洗
eeg_clean = EEGCleaner().apply_eeg_cleaning(eeg_raw)
lfp_clean = LFPCleaner().apply_lfp_cleaning(lfp_raw)

# 4. 完成！
print("预处理完成！")
```

## 📊 常见使用场景

### 场景1: 仅需要清洗数据

```python
from preprocessing import EEGCleaner, LFPCleaner

# 清洗EEG
cleaner = EEGCleaner()
eeg_clean = cleaner.apply_eeg_cleaning(
    eeg_raw,
    target_sfreq=1000.0,  # 重采样到1000Hz
    line_freq=50.0,       # 去除50Hz工频
)

# 清洗LFP
lfp_clean = LFPCleaner().apply_lfp_cleaning(lfp_raw)
```

### 场景2: EEG去伪迹（眼电、心电）

```python
from preprocessing import EEGPreprocessor

prep = EEGPreprocessor()

# 检测坏导
bad_chs = prep.detect_bad_channels(eeg_raw)
eeg_raw = prep.interpolate_bad_channels(eeg_raw, bad_chs)

# 重参考
eeg_raw = prep.set_reference(eeg_raw, ref_type='average')

# ICA去伪迹
eeg_raw, ica = prep.run_ica(eeg_raw, n_components=25)
artifacts = prep.detect_artifact_components(eeg_raw, ica)
eeg_clean = prep.remove_artifacts_ica(eeg_raw, ica, artifacts['all'])
```

### 场景3: LFP去刺激伪迹

```python
from preprocessing import LFPPreprocessor

prep = LFPPreprocessor()

# 去除DBS刺激伪迹
lfp_clean = prep.remove_stimulation_artifacts(
    lfp_raw,
    method='template',  # 模板减法
    window=(-0.005, 0.01)
)

# 应用双极参考
lfp_bipolar = prep.apply_bipolar_reference(lfp_clean)

# 增强信噪比
lfp_enhanced = prep.enhance_snr(lfp_bipolar, method='car')
```

### 场景4: 频段分解

```python
from preprocessing import JointPreprocessor

joint = JointPreprocessor()

# 定义频段
bands = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# 提取频段
eeg_bands = joint.extract_frequency_bands(eeg_raw, bands=bands)
lfp_bands = joint.extract_frequency_bands(lfp_raw, bands=bands)

# 访问特定频段
theta_eeg = eeg_bands['theta']
beta_lfp = lfp_bands['beta']
```

### 场景5: 创建epochs并对齐

```python
import mne
from preprocessing import JointPreprocessor

# 提取事件
events, event_id = mne.events_from_annotations(eeg_raw)

# 创建epochs
eeg_epochs = mne.Epochs(eeg_raw, events, event_id, 
                        tmin=-0.5, tmax=1.5, preload=True)
lfp_epochs = mne.Epochs(lfp_raw, events, event_id,
                        tmin=-0.5, tmax=1.5, preload=True)

# 同步
joint = JointPreprocessor()
eeg_sync, lfp_sync = joint.synchronize_epochs(eeg_epochs, lfp_epochs)
```

### 场景6: 质量控制

```python
from preprocessing import QualityControl

qc = QualityControl(output_dir='./qc_results')

# 对比处理前后
qc.plot_psd_comparison(
    eeg_raw_before, 
    eeg_raw_after,
    save_path='qc_results/psd_comparison.png'
)

# 计算信噪比
snr = qc.compute_snr(eeg_clean)

# 生成报告
report = qc.generate_qc_report(
    preprocessing_steps=['filter', 'ica', 'epoch'],
    save_path='qc_results/report.txt'
)
```

### 场景7: 保存为BIDS derivatives

```python
from preprocessing import BIDSDerivativesSaver

saver = BIDSDerivativesSaver(bids_root, derivatives_name='preprocessing')

# 保存清洗后的数据
saver.save_preprocessed_raw(
    eeg_clean,
    subject='sub-01',
    session='ses-01',
    task='task-rest',
    datatype='eeg',
    description='clean'
)

# 保存epochs
saver.save_epochs(
    eeg_epochs,
    subject='sub-01',
    session='ses-01',
    task='task-rest',
    datatype='eeg'
)
```

## 🔧 模块说明

### DataValidator
- 验证BIDS数据完整性
- 检查采样率和时间对齐
- 检查事件同步

### EEGCleaner / LFPCleaner
- 去趋势和去直流偏移
- 带通滤波和陷波滤波
- 重采样

### EEGPreprocessor
- 坏导检测与插值
- 重参考
- ICA去伪迹
- Epochs创建
- 源空间重建（可选）

### LFPPreprocessor
- 刺激伪迹去除
- 电极接触点管理
- 双极参考
- 信号增强

### JointPreprocessor
- 时间对齐
- Epochs同步
- 频段分解
- 标准化
- 连接性分析准备

### QualityControl
- 可视化对比
- 信噪比计算
- 质量报告生成
- BIDS derivatives保存

## 📝 完整示例

查看 `notebooks/01_complete_preprocessing.ipynb` 获取完整的端到端示例。

启动Jupyter：

```bash
cd notebooks
jupyter notebook 01_complete_preprocessing.ipynb
```

## ⚙️ 自定义参数

### 常用参数调整

```python
# 1. 滤波器参数
eeg_clean = cleaner.apply_bandpass_filter(
    eeg_raw,
    l_freq=0.5,      # 更低的低频截止
    h_freq=120.0,    # 更高的高频截止
    filter_type='iir'  # 使用IIR而非FIR
)

# 2. ICA参数
eeg_raw, ica = prep.run_ica(
    eeg_raw,
    n_components=30,    # 更多成分
    method='infomax',   # 不同的算法
    random_state=42
)

# 3. Epochs参数
epochs = prep.create_epochs(
    eeg_raw,
    tmin=-1.0,          # 更长的pre-stimulus
    tmax=2.0,           # 更长的post-stimulus
    baseline=(-0.5, 0), # 更长的基线
    reject=dict(eeg=200e-6)  # 更宽松的拒绝阈值
)

# 4. 频段定义
custom_bands = {
    'slow': (0.5, 4),
    'fast': (30, 150)
}
bands = joint.extract_frequency_bands(raw, bands=custom_bands)
```

## 🐛 故障排除

### 问题1: 找不到模块

```python
# 确保添加了路径
import sys
sys.path.append('src')  # 或完整路径
```

### 问题2: 内存不足

```python
# 使用较小的数据块
raw_crop = raw.copy().crop(tmax=60)  # 只处理前60秒

# 或减少ICA成分数
ica = prep.run_ica(raw, n_components=15)  # 更少的成分
```

### 问题3: 找不到事件

```python
# 检查注释
print(raw.annotations)

# 手动添加事件
events = mne.make_fixed_length_events(raw, duration=1.0)
```

## 📚 进一步学习

1. **详细文档**: 阅读 `README.md`
2. **完整示例**: 运行 `notebooks/01_complete_preprocessing.ipynb`
3. **API文档**: 查看各模块的docstrings
4. **MNE教程**: https://mne.tools/stable/tutorials.html

## 💡 最佳实践

1. **总是先验证数据**
   ```python
   validator.run_full_validation(...)
   ```

2. **保存原始数据副本**
   ```python
   raw_orig = raw.copy()
   ```

3. **逐步检查结果**
   ```python
   qc.plot_psd_comparison(raw_before, raw_after)
   ```

4. **记录处理步骤**
   ```python
   print(preprocessor.get_processing_summary())
   ```

5. **使用BIDS derivatives**
   ```python
   saver.save_preprocessed_raw(...)
   ```

## 🎯 下一步

预处理完成后，您可以进行：

- **连接性分析**: 相干性、PLV、Granger因果
- **时频分析**: STFT、小波、HHT
- **相位-振幅耦合**: PAC分析
- **机器学习**: 特征提取、分类

## 📧 获取帮助

遇到问题？

1. 查看 `README.md` 详细文档
2. 检查 notebooks 中的示例
3. 提交 Issue

祝您分析顺利！🎉
