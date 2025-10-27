# 数据格式支持说明

## 📋 支持的格式

本预处理工具包支持多种常用的EEG/LFP数据格式：

| 格式 | 扩展名 | 读取 | 写入 | 推荐用途 |
|------|--------|------|------|----------|
| **BrainVision** | .vhdr/.eeg/.vmrk | ✅ | ✅ | 通用、共享 |
| **FIF** | .fif | ✅ | ✅ | MNE工作流 |
| **EDF** | .edf | ✅ | ✅ | 临床数据 |
| **BDF** | .bdf | ✅ | ❌ | Biosemi数据 |
| **EEGLAB SET** | .set | ✅ | ❌ | EEGLAB导入 |
| **Neuroscan CNT** | .cnt | ✅ | ❌ | Neuroscan系统 |

## 🔄 格式对比

### BrainVision (.vhdr/.eeg/.vmrk)

**优点：**
- 广泛的软件支持（Brain Products, EEGLAB, FieldTrip, SPM等）
- 文本header文件，便于检查和编辑
- 完整保留标记（markers）信息
- 工业标准格式

**缺点：**
- 三个文件组成（vhdr, eeg, vmrk），需要同时管理
- 文件较大（未压缩）

**BIDS结构示例：**
```
sub-01/ses-01/eeg/
├── sub-01_ses-01_task-rest_eeg.vhdr  # Header文件（文本）
├── sub-01_ses-01_task-rest_eeg.eeg   # 数据文件（二进制）
├── sub-01_ses-01_task-rest_eeg.vmrk  # Marker文件（文本）
├── sub-01_ses-01_task-rest_eeg.json  # BIDS元数据
└── sub-01_ses-01_task-rest_events.tsv # BIDS事件
```

**使用场景：**
- 原始数据存储
- 跨软件数据共享
- 发表数据集
- 需要手动检查header的情况

---

### FIF (.fif)

**优点：**
- MNE-Python原生格式
- 压缩存储，节省磁盘空间
- 读写速度快
- 完整保留MNE处理信息（投影、历史、坏导等）
- 单个文件

**缺点：**
- 主要限于MNE-Python生态
- 二进制格式，难以手动检查

**BIDS结构示例：**
```
sub-01/ses-01/eeg/
├── sub-01_ses-01_task-rest_eeg.fif      # 数据文件
├── sub-01_ses-01_task-rest_eeg.json     # BIDS元数据
└── sub-01_ses-01_task-rest_events.tsv   # BIDS事件
```

**使用场景：**
- MNE-Python预处理流程
- 中间处理步骤
- Epochs和Evoked数据存储
- 需要快速I/O的分析

---

### EDF (.edf)

**优点：**
- 临床EEG标准格式
- 非常广泛的软件支持
- 适合长时程记录
- 国际标准（欧洲数据格式）

**缺点：**
- 固定采样率限制
- 通道数限制（256）
- 数据精度有限（16-bit）

**使用场景：**
- 临床EEG数据
- 睡眠研究
- 长时程监测（>1小时）
- 需要临床软件兼容性

---

### BDF (.bdf)

**优点：**
- Biosemi系统原生格式
- 24-bit精度（高于EDF）
- 支持更多通道

**缺点：**
- 主要限于Biosemi系统
- 写入支持有限

**使用场景：**
- Biosemi设备采集的数据
- 需要高精度的数据

---

## 🚀 使用示例

### 自动格式检测（推荐）

```python
from preprocessing import BIDSDataLoader

loader = BIDSDataLoader('/path/to/bids')

# 自动检测并加载（支持所有格式）
eeg_raw, metadata = loader.load_eeg_data(
    subject='sub-01',
    session='ses-01',
    task='task-rest'
)
# 会自动检测是 .vhdr, .fif, .edf 等格式
```

### 指定格式加载

```python
# 强制使用BrainVision格式
eeg_raw, _ = loader.load_eeg_data(
    subject='sub-01',
    session='ses-01',
    task='task-rest',
    format_type='brainvision'
)

# 强制使用FIF格式
eeg_raw, _ = loader.load_eeg_data(
    subject='sub-01',
    session='ses-01',
    task='task-rest',
    format_type='fif'
)
```

### 格式转换

```python
from preprocessing import BIDSDataLoader, BIDSDataSaver

loader = BIDSDataLoader(bids_root)
saver = BIDSDataSaver(bids_root)

# 加载BrainVision格式
raw, _ = loader.load_eeg_data(
    'sub-01', 'ses-01', 'task-rest',
    format_type='brainvision'
)

# 保存为FIF格式
saver.save_preprocessed_raw(
    raw,
    subject='sub-01',
    session='ses-01',
    task='task-rest',
    datatype='eeg',
    description='converted',
    format_type='fif'
)
```

### 预处理后保存为不同格式

```python
from preprocessing import EEGCleaner, BIDSDataSaver

# 预处理
cleaner = EEGCleaner()
eeg_clean = cleaner.apply_eeg_cleaning(eeg_raw)

saver = BIDSDataSaver(bids_root)

# 保存为BrainVision（用于共享）
saver.save_preprocessed_raw(
    eeg_clean,
    'sub-01', 'ses-01', 'task-rest', 'eeg',
    description='clean',
    format_type='brainvision'
)

# 同时保存为FIF（用于快速分析）
saver.save_preprocessed_raw(
    eeg_clean,
    'sub-01', 'ses-01', 'task-rest', 'eeg',
    description='clean',
    format_type='fif'
)
```

## 💡 格式选择建议

### 原始数据存储

**推荐格式：BrainVision 或 EDF**

原因：
- 广泛的软件兼容性
- 长期存档的可靠性
- 易于数据共享
- 符合BIDS标准

```python
# 原始数据应该已经是这些格式之一
# 如果不是，先转换：
saver.save_preprocessed_raw(
    raw_original,
    ...,
    format_type='brainvision'  # 或 'edf'
)
```

---

### 预处理流程中间步骤

**推荐格式：FIF**

原因：
- 快速读写
- 节省空间
- 保留MNE处理信息
- 适合迭代处理

```python
# 每个预处理步骤后保存
saver.save_preprocessed_raw(
    raw_after_ica,
    ...,
    description='after-ica',
    format_type='fif'  # 快速I/O
)
```

---

### 最终结果/发布数据

**推荐格式：BrainVision + FIF（双格式）**

原因：
- BrainVision：通用性和兼容性
- FIF：便于MNE用户快速分析

```python
# 同时保存两种格式
for fmt in ['brainvision', 'fif']:
    saver.save_preprocessed_raw(
        final_clean_data,
        ...,
        description='final',
        format_type=fmt
    )
```

---

### Epochs数据

**推荐格式：FIF（强烈推荐）**

原因：
- Epochs结构复杂，FIF完整支持
- 其他格式可能丢失重要信息
- FIF是epochs的原生格式

```python
# Epochs通常只保存为FIF
saver.save_epochs(
    epochs,
    ...,
    format_type='fif'  # 几乎是唯一选择
)
```

---

## 📊 实际工作流推荐

### 方案A：兼容性优先

```python
# 1. 原始数据：BrainVision
raw, _ = loader.load_eeg_data(..., format_type='brainvision')

# 2. 预处理：在内存中处理
raw_clean = preprocess(raw)

# 3. 保存结果：BrainVision（发布）+ FIF（快速访问）
for fmt in ['brainvision', 'fif']:
    saver.save_preprocessed_raw(raw_clean, ..., format_type=fmt)
```

**优点：**
- 最大兼容性
- 适合数据共享
- 适合发表

---

### 方案B：效率优先（MNE工作流）

```python
# 1. 原始数据：任意格式（自动检测）
raw, _ = loader.load_eeg_data(...)

# 2. 立即转为FIF
saver.save_preprocessed_raw(raw, ..., format_type='fif')

# 3. 后续全部使用FIF
# - 快速读写
# - 节省空间
# - 完整信息

# 4. 最终发布时转为BrainVision
saver.save_preprocessed_raw(final, ..., format_type='brainvision')
```

**优点：**
- 最快处理速度
- 节省磁盘空间
- 完整保留MNE信息

---

### 方案C：临床数据流程

```python
# 1. 原始数据：EDF（临床标准）
raw, _ = loader.load_eeg_data(..., format_type='edf')

# 2. 预处理：转为FIF（效率）
saver.save_preprocessed_raw(raw, ..., format_type='fif')

# 3. 分析：使用FIF
# 快速分析...

# 4. 结果：转回EDF（临床归档）
saver.save_preprocessed_raw(final, ..., format_type='edf')
```

**优点：**
- 符合临床标准
- 处理效率高
- 便于归档

---

## ⚠️ 注意事项

### 格式限制

1. **EDF限制：**
   - 最多256通道
   - 16-bit精度
   - 固定采样率

2. **BrainVision注意：**
   - 三个文件必须同时存在
   - 移动文件时要保持相对路径

3. **FIF注意：**
   - 某些软件不支持
   - 压缩格式不适合手动检查

### 数据精度

格式精度对比：
- **BDF**: 24-bit（最高）
- **BrainVision**: 32-bit float（高）
- **FIF**: 32-bit float（高）
- **EDF**: 16-bit（中）

### 兼容性测试

建议在重要项目中测试格式转换：

```python
# 测试往返转换
raw_original, _ = loader.load_eeg_data(...)

# BV → FIF → BV
saver.save_preprocessed_raw(raw_original, ..., format_type='fif')
raw_fif, _ = loader.load_eeg_data(..., format_type='fif')
saver.save_preprocessed_raw(raw_fif, ..., format_type='brainvision')
raw_bv, _ = loader.load_eeg_data(..., format_type='brainvision')

# 检查数据一致性
assert np.allclose(raw_original.get_data(), raw_bv.get_data())
```

---

## 📚 更多资源

- **MNE格式文档**: https://mne.tools/stable/auto_tutorials/io/index.html
- **BIDS规范**: https://bids-specification.readthedocs.io/
- **BrainVision格式**: https://www.brainproducts.com/
- **EDF规范**: https://www.edfplus.info/

---

## 🆘 常见问题

**Q: 我应该用哪种格式？**
A: 
- 数据共享/发布 → BrainVision
- MNE分析 → FIF
- 临床数据 → EDF
- 不确定 → 同时保存BrainVision和FIF

**Q: 格式转换会丢失信息吗？**
A: 
- BrainVision ↔ FIF：不会（推荐）
- 任意格式 → EDF：可能（精度降低）
- 复杂结构（epochs, evoked）→ 仅FIF完整支持

**Q: 为什么同时保存多种格式？**
A: 
- 平衡兼容性和效率
- BrainVision用于共享
- FIF用于快速分析
- 磁盘空间便宜，时间宝贵

**Q: 可以混用不同格式吗？**
A: 完全可以！本工具包自动处理格式差异，无需担心。

---

## ✅ 总结

1. **自动检测** - 无需手动指定格式
2. **灵活转换** - 轻松在格式间转换
3. **统一API** - 相同的处理流程
4. **最佳实践** - 根据场景选择格式

**记住：工具包会自动处理格式细节，您只需专注于数据分析！**
