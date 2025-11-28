import logging
logger = logging.getLogger(__name__)

def apply_average_reference(self, raw, ref_channel='REF CZ', copy=True):
    if copy:
        raw = raw.copy()

    if ref_channel is not None and ref_channel in raw.ch_names:
        raw.set_eeg_reference(ref_channels=[ref_channel])   # 重新参考到 REF CZ
        raw.set_channel_types({ref_channel: 'misc'})
        raw.set_eeg_reference('average', projection=True)   # 添加平均参考投影器（用于源重建）
        raw.apply_proj()    # 应用投影器        
        self.processing_history.append(f'avg_ref_via_{ref_channel}')
    else:
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        self.processing_history.append('avg_ref_direct')

    return raw



