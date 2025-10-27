"""
Utilities to (1) load EGI coordinates.xml into Raw as a DigMontage,
(2) coarsely align electrodes to a FieldTrip headmodel using a rigid ICP,
(3) build & save an MNE head->MRI Transform (.fif).
All coordinates are converted to meters.

Dependencies: numpy, scipy, mne
"""

from __future__ import annotations
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree, ConvexHull
import mne


# ---------------------------
# XML -> DigMontage helpers
# ---------------------------

def _infer_scale_to_m(xyz: np.ndarray) -> float:
    """
    Infer unit scale to meters from value magnitudes.
    Heuristic:
      - if max|coord| > 30:    assume mm  -> 1e-3
      - elif > 2:              assume cm  -> 1e-2
      - else:                  assume meters
    """
    xyz = np.asarray(xyz, float)
    if xyz.size == 0:
        return 1.0
    mx = float(np.nanmax(np.abs(xyz)))
    if mx > 30.0:
        return 1e-3
    elif mx > 2.0:
        return 1e-2
    else:
        return 1.0


def _parse_coordinates_xml(xml_path: Path):
    """
    Parse an EGI coordinates.xml and return:
      - eeg_by_number: dict[int -> np.array(3,)], electrode positions (meters)
      - fiducials: dict with keys 'NAS','LPA','RPA' -> np.array(3,) or None
      - unit_scale: float, the applied scale to convert to meters
    """
    root = ET.parse(str(xml_path)).getroot()
    ns = {"egi": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
    sensors = root.findall(".//egi:sensor", ns) if ns else root.findall(".//sensor")

    rows = []
    def ftext(elem, tag):
        return elem.findtext(f"egi:{tag}", default=elem.findtext(tag), namespaces=ns)

    for s in sensors:
        name = ftext(s, "name") or ""
        num  = ftext(s, "number")
        typ  = ftext(s, "type")
        x    = float(ftext(s, "x"))
        y    = float(ftext(s, "y"))
        z    = float(ftext(s, "z"))
        rows.append((name, num, int(typ) if typ is not None else None, x, y, z))

    arr = np.array([[r[3], r[4], r[5]] for r in rows], float)
    scale = _infer_scale_to_m(arr)

    eeg_by_number = {}
    fid = {"NAS": None, "LPA": None, "RPA": None}

    for name, num, typ, x, y, z in rows:
        p = np.array([x, y, z], float) * scale
        if typ == 0:  # EEG sensor
            if num is None:
                continue
            try:
                eeg_by_number[int(num)] = p
            except Exception:
                pass
        elif typ == 2:  # fiducial
            nm = (name or "").lower()
            if "nas" in nm or "nasion" in nm:
                fid["NAS"] = p
            elif "lpa" in nm or ("left" in nm and ("aur" in nm or "periaur" in nm)):
                fid["LPA"] = p
            elif "rpa" in nm or ("right" in nm and ("aur" in nm or "periaur" in nm)):
                fid["RPA"] = p

    return eeg_by_number, fid, scale


def _map_numbers_to_raw_channels(raw: mne.io.BaseRaw, eeg_by_number: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Map numeric EGI sensor IDs (e.g., 1..N) to raw EEG channel names.
    Tries common patterns: 'E123', trailing digits, etc.
    Returns: ch_pos dict suitable for make_dig_montage(ch_pos=...)
    """
    ch_pos = {}
    for ch in raw.ch_names:
        if raw.get_channel_types(picks=[ch])[0] != "eeg":
            continue
        # Accept names like 'E57', '57', 'EEG057' -> match trailing digits
        m = re.search(r'(\d+)$', ch.replace('E', ''))
        if m:
            idx = int(m.group(1))
            if idx in eeg_by_number:
                ch_pos[ch] = eeg_by_number[idx]
    return ch_pos


def apply_coordinates_xml_to_raw(raw: mne.io.BaseRaw, coordinates_xml: str | Path, set_ref_misc: bool = True) -> mne.channels.DigMontage:
    """
    Build and apply a DigMontage from an EGI coordinates.xml to the given Raw.
    This writes the 3D electrode/fiducial locations into raw.info['dig'] (head coords).
    """
    coordinates_xml = Path(coordinates_xml)
    eeg_by_number, fid, _ = _parse_coordinates_xml(coordinates_xml)
    ch_pos = _map_numbers_to_raw_channels(raw, eeg_by_number)

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=fid["NAS"], lpa=fid["LPA"], rpa=fid["RPA"],
        coord_frame="head"
    )
    raw.set_montage(montage, on_missing="warn")

    # Optionally set any reference channels (if present in raw) to 'misc'.
    # If you need this, you can extend the XML parser to collect type==1 names.
    if set_ref_misc:
        # Example: often 'Cz' or 'REF' might be the reference; adapt if needed.
        pass

    return montage


# ---------------------------
# Rigid ICP alignment helpers
# ---------------------------

def _rigid_fit_svd(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kabsch algorithm: find R, t that minimizes ||R*A + t - B||_F.
    A, B: Nx3
    Returns:
      R: (3,3) rotation matrix
      t: (3,)  translation vector
    """
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = B.mean(axis=0) - R @ A.mean(axis=0)
    return R, t


def _icp_point_to_surface(src_pts: np.ndarray, dst_pts: np.ndarray, max_iter: int = 60, tol: float = 1e-6):
    """
    Simple point-to-surface ICP:
      - For each src point, find the nearest dst point
      - Fit rigid transform via SVD
      - Iterate until convergence
    Returns:
      R_total, t_total, moved_src (after alignment)
    """
    R_total = np.eye(3)
    t_total = np.zeros(3)
    moved = src_pts.copy()
    tree = cKDTree(dst_pts)
    prev_err = np.inf
    for _ in range(max_iter):
        dists, idx = tree.query(moved)
        R, t = _rigid_fit_svd(moved, dst_pts[idx])
        moved = (R @ moved.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        err = float(np.mean(dists))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return R_total, t_total, moved


def _get_eeg_points_from_raw(raw: mne.io.BaseRaw) -> np.ndarray:
    """Extract EEG dig points (meters) from raw.info['dig']."""
    digs = [
        d["r"] for d in (raw.info.get("dig") or [])
        if d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_EEG
    ]
    return np.array(digs, float) if len(digs) else np.empty((0, 3), float)


# ---------------------------
# Main: make_trans_from_coordinates
# ---------------------------

def make_trans_from_coordinates(
    raw: mne.io.BaseRaw,
    coordinates_xml: str | Path,
    ft_headmodel_mat: str | Path,
    out_trans_path: str | Path,
    max_iter: int = 60
) -> tuple[mne.transforms.Transform, float, float]:
    """
    High-level convenience function:
    1) Load coordinates.xml and apply as DigMontage to 'raw'.
    2) Load FieldTrip headmodel (.mat), convert to meters, and take convex hull vertices as scalp proxy.
    3) Rigid ICP to align EEG dig points (head coords) to the scalp proxy.
    4) Build an MNE head->MRI Transform and write to out_trans_path.
       (We treat the headmodel space as 'MRI' frame for this coarse alignment.)
    Returns:
      - trans (mne.Transform, head->mri)
      - mean_distance_mm (float): mean nearest distance after ICP
      - p95_distance_mm (float): 95th percentile nearest distance after ICP
    """
    coordinates_xml = Path(coordinates_xml)
    ft_headmodel_mat = Path(ft_headmodel_mat)
    out_trans_path = Path(out_trans_path)

    # 1) Apply coordinates.xml -> raw (writes dig into raw.info['dig'])
    apply_coordinates_xml_to_raw(raw, coordinates_xml, set_ref_misc=True)

    # 2) Load FieldTrip headmodel and convert to meters if needed
    mat = loadmat(str(ft_headmodel_mat), squeeze_me=True, struct_as_record=False)
    headmodel = mat.get("headmodel", None)
    if headmodel is None:
        raise RuntimeError("FieldTrip .mat file has no 'headmodel' struct.")

    pos = np.array(getattr(headmodel, "pos", None), float)
    if pos.size == 0:
        raise RuntimeError("headmodel.pos is missing or empty.")
    # FieldTrip often uses mm; convert to meters if values look large
    if np.max(np.abs(pos)) > 1.0:
        pos = pos * 1e-3

    # Use convex hull vertices as a scalp proxy for coarse alignment
    hull = ConvexHull(pos)
    scalp = pos[hull.vertices]

    # 3) Extract EEG dig points (meters) from raw and run ICP
    eeg_xyz = _get_eeg_points_from_raw(raw)
    if eeg_xyz.size == 0:
        raise RuntimeError("No EEG dig points found in raw. Did the montage apply correctly?")

    R, t, moved = _icp_point_to_surface(eeg_xyz, scalp, max_iter=max_iter, tol=1e-6)

    # Evaluate distances after alignment for QA
    tree = cKDTree(scalp)
    dists, _ = tree.query((R @ eeg_xyz.T).T + t)
    mean_mm = float(np.mean(dists) * 1000.0)
    p95_mm  = float(np.percentile(dists, 95) * 1000.0)

    # 4) Build head->mri Transform (treating headmodel space as 'mri') and save
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    trans = mne.transforms.Transform('head', 'mri', T)
    mne.write_trans(str(out_trans_path), trans)

    return trans, mean_mm, p95_mm
