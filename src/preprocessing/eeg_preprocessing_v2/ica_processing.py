import logging
import numpy as np
import mne

logger = logging.getLogger(__name__)

# ---------------- ICA 计算 ----------------

def apply_ica(self, raw, n_components=None, method='fastica',
              random_state=42, copy=True):
    if copy: raw = raw.copy()

    if n_components is None:
        n_components = min(len(raw.ch_names), raw.n_times // 2)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter='auto'
    )
    ica.fit(raw)
    return raw, ica


# ---------------- ICLabel 分类 ----------------

def classify_ica_components_iclabel(self, ica, raw):
    from mne_icalabel import label_components

    ic_labels = label_components(raw, ica, method='iclabel')
    labels_pred = ic_labels['labels']
    labels_pred_proba = ic_labels['y_pred_proba']
    label_names = ['brain','eye','heart','muscle','line_noise','channel_noise','other']

    # 统一存储
    self.ica_component_labels = {
        'labels': labels_pred,
        'probabilities': labels_pred_proba,
        'label_names': label_names
    }
    return labels_pred, labels_pred_proba, label_names


# ---------------- ICLabel 组件选择逻辑 ----------------

def select_ica_components_by_label(self, labels_pred, labels_pred_proba,
                                   label_names, exclude_labels=None,
                                   brain_threshold=0.5, artifact_threshold=0.5):
    if exclude_labels is None:
        exclude_labels = ['eye', 'heart', 'muscle', 'line_noise', 'channel_noise']

    exclude_idx = []

    # 新格式 (1D)
    if labels_pred_proba.ndim == 1:
        for i in range(len(labels_pred)):
            label = labels_pred[i]
            prob = float(labels_pred_proba[i])

            if any(p in label for p in exclude_labels) and prob >= artifact_threshold:
                exclude_idx.append(i)
    else:
        # 旧格式 (2D)
        for i in range(len(labels_pred)):
            label = labels_pred[i]
            probs = labels_pred_proba[i]

            if any(p in label for p in exclude_labels):
                exclude_idx.append(i)

    return exclude_idx


# ---------------- 自动检测 + 剔除 ----------------

def detect_ica_artifacts(self, ica, raw, artifact_types=None,
                         use_iclabel=True, **iclabel_kwargs):

    if use_iclabel:
        labels_pred, labels_pred_proba, label_names = \
            self.classify_ica_components_iclabel(ica, raw)

        return self.select_ica_components_by_label(
            labels_pred, labels_pred_proba, label_names, **iclabel_kwargs
        )

    return []


def apply_ica_cleaning(self, raw, ica, exclude_idx=None,
                       auto_detect=True, use_iclabel=True, copy=True,
                       **iclabel_kwargs):
    if copy: raw = raw.copy()

    if exclude_idx is None and auto_detect:
        exclude_idx = self.detect_ica_artifacts(
            ica, raw, use_iclabel=use_iclabel, **iclabel_kwargs
        )

    if exclude_idx:
        ica.exclude = exclude_idx
        raw = ica.apply(raw)

    return raw
