# 📢 重要更新：多格式支持

## 🎯 更新背景

针对您提到的格式不一致问题（之前保存为BrainVision .eeg格式，但示例中读取的是.fif格式），我们进行了重要更新。

## ✨ 新增功能

### 1. 新增 `data_io.py` 模块

**功能：**
- 统一的数据加载接口（`BIDSDataLoader`）
- 统一的数据保存接口（`BIDSDataSaver`）
- **自动格式检测**（无需手动指定）
- 支持6种常用格式

**支持的格式：**
```python
formats = {
    'brainvision': ['.vhdr', '.eeg', '.vmrk'],  # ✅ 您使用的格式
    'fif': ['.fif'],                             # ✅ MNE标准格式
    'edf': ['.edf'],                             # ✅ 临床标准
    'bdf': ['.bdf'],                             # ✅ Biosemi
    'set': ['.set'],                             # ✅ EEGLAB
    'cnt': ['.cnt']                              # ✅ Neuroscan
}
```

### 2. 更新 `data_validation.py`

**改进：**
- 现在使用 `BIDSDataLoader` 进行数据加载
- 自动检测BrainVision、FIF等所有支持的格式
- 保持原有API不变，向后兼容

**新方法签名：**
```python
def load_eeg_data(self, subject, session, task, 
                  run=None, 
                  format_type=None):  # 新参数：可选指定格式
```

### 3. 新增文档

- **FORMAT_SUPPORT.md** - 格式支持详细说明
- **02_format_conversion.ipynb** - 格式转换示例notebook

## 🔄 解决的问题

### 问题描述
之前的代码假设数据保存为 `.fif` 格式：
```python
# 旧代码
eeg_file = eeg_dir / f"{filename_base}_eeg.fif"
raw = mne.io.read_raw_fif(eeg_file, preload=True)
```

但实际数据可能是 **BrainVision** 格式 (`.vhdr/.eeg/.vmrk`)

### 解决方案
新代码自动检测格式：
```python
# 新代码
loader = BIDSDataLoader(bids_root)
raw, metadata = loader.load_eeg_data(subject, session, task)
# 自动检测：.vhdr, .fif, .edf 等
```

## 📝 迁移指南

### 旧代码
```python
from preprocessing import DataValidator

validator = DataValidator(bids_root)
eeg_raw, eeg_meta = validator.load_eeg_data('sub-01', 'ses-01', 'task-rest')
# 只支持 .fif 格式
```

### 新代码（完全向后兼容）
```python
from preprocessing import DataValidator

validator = DataValidator(bids_root)
eeg_raw, eeg_meta = validator.load_eeg_data('sub-01', 'ses-01', 'task-rest')
# 自动支持 .vhdr, .fif, .edf 等所有格式！
```

**✅ 无需修改现有代码！**

## 🚀 新功能使用

### 1. 自动格式检测（推荐）
```python
from preprocessing import BIDSDataLoader

loader = BIDSDataLoader(bids_root)

# 自动检测并加载（无需关心是 .vhdr 还是 .fif）
eeg_raw, metadata = loader.load_eeg_data(
    subject='sub-01',
    session='ses-01',
    task='task-rest'
)
```

### 2. 强制指定格式
```python
# 如果有多个格式的文件，可以强制指定
eeg_raw, _ = loader.load_eeg_data(
    'sub-01', 'ses-01', 'task-rest',
    format_type='brainvision'  # 明确使用BrainVision
)
```

### 3. 格式转换
```python
from preprocessing import BIDSDataLoader, BIDSDataSaver

loader = BIDSDataLoader(bids_root)
saver = BIDSDataSaver(bids_root)

# 加载BrainVision
raw, _ = loader.load_eeg_data(..., format_type='brainvision')

# 保存为FIF（更快、更小）
saver.save_preprocessed_raw(
    raw, ...,
    format_type='fif'
)
```

### 4. 保存为不同格式
```python
from preprocessing import BIDSDataSaver

saver = BIDSDataSaver(bids_root)

# 同时保存两种格式
for fmt in ['brainvision', 'fif']:
    saver.save_preprocessed_raw(
        preprocessed_data,
        subject='sub-01',
        session='ses-01',
        task='task-rest',
        datatype='eeg',
        description='clean',
        format_type=fmt
    )
```

## 💡 推荐工作流

### 针对您的情况（BrainVision数据）

```python
from preprocessing import (
    DataValidator, EEGCleaner, LFPCleaner,
    BIDSDataSaver
)

# 1. 验证（自动识别BrainVision格式）
validator = DataValidator(bids_root)
results = validator.run_full_validation('sub-01', 'ses-01', 'task-rest')

# 2. 加载（自动识别）
eeg_raw, _ = validator.load_eeg_data('sub-01', 'ses-01', 'task-rest')
lfp_raw, _ = validator.load_lfp_data('sub-01', 'ses-01', 'task-rest')

# 3. 预处理
eeg_clean = EEGCleaner().apply_eeg_cleaning(eeg_raw)
lfp_clean = LFPCleaner().apply_lfp_cleaning(lfp_raw)

# 4. 保存结果
saver = BIDSDataSaver(bids_root)

# 保存为BrainVision（保持兼容性）
saver.save_preprocessed_raw(
    eeg_clean,
    'sub-01', 'ses-01', 'task-rest', 'eeg',
    description='preprocessed',
    format_type='brainvision'  # 与原始格式一致
)

# 同时保存为FIF（加快后续分析）
saver.save_preprocessed_raw(
    eeg_clean,
    'sub-01', 'ses-01', 'task-rest', 'eeg',
    description='preprocessed',
    format_type='fif'  # 快速访问版本
)
```

## 📊 格式建议

### BrainVision (.vhdr/.eeg/.vmrk)
**最适合：**
- 原始数据存储
- 跨软件共享（EEGLAB, FieldTrip, SPM等）
- 发表数据集

**您的情况：✅ 推荐继续使用**

### FIF (.fif)
**最适合：**
- 预处理流程中间步骤（快速I/O）
- Epochs数据（必须用FIF）
- 纯MNE工作流

**建议：** 在BrainVision基础上，额外保存FIF版本

## 🔍 验证更新

您可以运行以下代码验证更新是否成功：

```python
from preprocessing import BIDSDataLoader

# 测试自动格式检测
loader = BIDSDataLoader('/path/to/your/bids')

try:
    # 应该能自动识别您的BrainVision文件
    raw, meta = loader.load_eeg_data(
        subject='sub-001',
        session='ses-01', 
        task='task-StimOn55HzFull2'
    )
    print(f"✓ 成功加载数据！")
    print(f"  格式：自动检测")
    print(f"  采样率：{raw.info['sfreq']} Hz")
    print(f"  通道数：{len(raw.ch_names)}")
except Exception as e:
    print(f"✗ 加载失败：{e}")
```

## 📚 相关文档

1. **FORMAT_SUPPORT.md** - 完整的格式支持说明
2. **02_format_conversion.ipynb** - 格式转换示例
3. **01_complete_preprocessing.ipynb** - 已更新，兼容所有格式

## ✅ 检查清单

- [x] 新增 `data_io.py` 模块
- [x] 更新 `data_validation.py` 使用新加载器
- [x] 更新 `__init__.py` 导出新类
- [x] 创建 `FORMAT_SUPPORT.md` 文档
- [x] 创建 `02_format_conversion.ipynb` 示例
- [x] 保持向后兼容性
- [x] 支持自动格式检测
- [x] 支持格式转换

## 🎉 总结

**主要改进：**
1. ✅ **解决了格式不一致问题**
2. ✅ **自动识别BrainVision、FIF等格式**
3. ✅ **支持灵活的格式转换**
4. ✅ **保持完全向后兼容**
5. ✅ **无需修改现有代码**

**您现在可以：**
- 直接加载BrainVision格式的数据
- 在不同格式间轻松转换
- 同时保存多种格式
- 使用统一的API处理所有格式

**记住：工具包现在会自动处理格式问题，您无需担心！** 🚀
