"""
源重建准备验证工具
===================

快速检查源重建所需的所有组件是否就绪

"""

import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat


def validate_preparation(
    epochs,
    trans_file=None,
    headmodel_file=None,
    coord_xml=None,
    plot=True
):
    """
    验证源重建所需的所有组件
    
    Parameters
    ----------
    epochs : mne.Epochs
        预处理后的epochs
    trans_file : str, optional
        Trans文件路径
    headmodel_file : str, optional
        头模型文件路径
    coord_xml : str, optional
        坐标XML文件路径（仅用于显示）
    plot : bool
        是否生成可视化图
    """
    print("\n" + "="*70)
    print("源重建准备验证工具 v2.0")
    print("="*70)
    
    all_checks_passed = True
    
    # ========================================
    # 检查1: Epochs基本信息
    # ========================================
    print("\n[检查 1/5] Epochs基本信息")
    print(f"  通道数: {len(epochs.ch_names)}")
    n_epochs = len(epochs)
    print(f"  Epochs数: {n_epochs}")
    print(f"  采样率: {epochs.info['sfreq']} Hz")
    print(f"  时间窗口: [{epochs.times[0]:.3f}, {epochs.times[-1]:.3f}] s")
    print("  ✓ Epochs信息正常")

    if n_epochs > 300:
        print("  ⚠️ Epoch数量较多，建议在源重建时使用 max_epochs 参数以避免内存压力")
    
    # ========================================
    # 检查2: 电极位置 (CRITICAL)
    # ========================================
    print("\n[检查 2/5] 电极位置 (关键!)")
    
    if epochs.info.get('dig') is None or len(epochs.info['dig']) == 0:
        print("  ❌ 未找到电极位置信息!")
        print("  → 请在预处理时加载coordinates.xml:")
        if coord_xml:
            print(f"     montage, scale = eeg_io.apply_coordinates_xml(raw, '{coord_xml}')")
        else:
            print(f"     montage, scale = eeg_io.apply_coordinates_xml(raw, COORD_XML)")
        all_checks_passed = False
    else:
        # 统计电极点
        n_eeg = sum(
            1 for d in epochs.info['dig']
            if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG
        )
        n_fid = sum(
            1 for d in epochs.info['dig']
            if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL
        )
        
        if n_eeg == 0:
            print("  ❌ 未找到EEG电极位置!")
            all_checks_passed = False
        else:
            print(f"  ✓ 找到 {n_eeg} 个EEG电极位置")
            print(f"  ✓ 找到 {n_fid} 个基准点 (fiducials)")
            
            # 检查位置合理性
            positions = np.array([
                d['r'] for d in epochs.info['dig']
                if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG
            ])
            
            center = positions.mean(axis=0)
            max_dist = np.max(np.linalg.norm(positions - center, axis=1))
            
            print(f"  位置统计:")
            print(f"    中心: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] m")
            print(f"    最大半径: {max_dist:.4f} m")
            
            if max_dist > 0.3:
                print(f"  ⚠️  警告: 电极分布范围较大 ({max_dist:.3f}m)")
            
            # 可视化
            if plot:
                fig = plt.figure(figsize=(15, 5))
                
                # 3D视图
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c='blue', s=50, alpha=0.6)
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_zlabel('Z (m)')
                ax1.set_title('Electrode Positions (3D)')
                
                # 顶视图
                ax2 = fig.add_subplot(132)
                ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=50)
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_title('Top View')
                ax2.axis('equal')
                ax2.grid(True)
                
                # 侧视图
                ax3 = fig.add_subplot(133)
                ax3.scatter(positions[:, 1], positions[:, 2], c='blue', s=50)
                ax3.set_xlabel('Y (m)')
                ax3.set_ylabel('Z (m)')
                ax3.set_title('Side View')
                ax3.axis('equal')
                ax3.grid(True)
                
                plt.tight_layout()
                plt.savefig('/home/claude/electrode_validation.png', dpi=150, bbox_inches='tight')
                print(f"  ✓ 电极位置图已保存")
                plt.close()
    
    # ========================================
    # 检查3: Trans文件 (CRITICAL)
    # ========================================
    print("\n[检查 3/5] 头-MRI变换 (Trans文件)")
    
    if trans_file is None:
        print("  ⚠️  未提供trans文件")
        print("  → 将使用identity transform (精度降低)")
        print("  → 建议运行ICP配准:")
        print("     from preprocessing.align_headmodel import make_trans_from_coordinates")
        print("     trans, mean_mm, p95_mm = make_trans_from_coordinates(...)")
    else:
        trans_file = Path(trans_file)
        
        if not trans_file.exists():
            print(f"  ❌ Trans文件不存在: {trans_file}")
            print("  → 请先运行ICP配准生成trans文件")
            all_checks_passed = False
        else:
            try:
                trans = mne.read_trans(str(trans_file))
                print(f"  ✓ Trans文件加载成功: {trans_file.name}")
                
                # 检查变换类型
                if trans['from'] == mne.io.constants.FIFF.FIFFV_COORD_HEAD:
                    print("  ✓ 变换类型: head → MRI")
                else:
                    print(f"  ⚠️  变换类型异常: {trans['from']} → {trans['to']}")
                
                # 检查是否是identity
                T = trans['trans']
                is_identity = np.allclose(T, np.eye(4), atol=1e-3)
                
                if is_identity:
                    print("  ⚠️  这是identity变换 (未真实配准)")
                else:
                    translation = np.linalg.norm(T[:3, 3]) * 1000
                    rotation_angle = np.arccos((np.trace(T[:3, :3]) - 1) / 2) * 180 / np.pi
                    
                    print(f"  ✓ 真实配准检测到")
                    print(f"    平移: {translation:.1f} mm")
                    print(f"    旋转: {rotation_angle:.1f}°")
                    
            except Exception as e:
                print(f"  ❌ 加载trans文件失败: {e}")
                all_checks_passed = False
    
    # ========================================
    # 检查4: 头模型
    # ========================================
    print("\n[检查 4/5] 头模型")
    
    if headmodel_file is None:
        print("  ⚠️  未提供头模型文件")
    else:
        headmodel_file = Path(headmodel_file)
        
        if not headmodel_file.exists():
            print(f"  ❌ 头模型文件不存在: {headmodel_file}")
            all_checks_passed = False
        else:
            try:
                data = loadmat(str(headmodel_file), simplify_cells=True)
                headmodel = data.get('headmodel', data)
                
                print(f"  ✓ 头模型加载成功")
                
                # 检查基本属性
                if hasattr(headmodel, 'type'):
                    print(f"    类型: {headmodel.type}")
                
                if hasattr(headmodel, 'unit'):
                    print(f"    单位: {headmodel.unit}")
                    if headmodel.unit == 'mm':
                        print(f"    ✓ 单位正确 (将自动转换为米)")
                
                if hasattr(headmodel, 'pos'):
                    pos = np.array(headmodel.pos)
                    print(f"    网格点数: {len(pos)}")
                    
                    # 检查坐标范围
                    if headmodel.unit == 'mm':
                        pos_m = pos / 1000.0
                    else:
                        pos_m = pos
                    
                    print(f"    坐标范围:")
                    print(f"      X: [{pos_m[:, 0].min():.3f}, {pos_m[:, 0].max():.3f}] m")
                    print(f"      Y: [{pos_m[:, 1].min():.3f}, {pos_m[:, 1].max():.3f}] m")
                    print(f"      Z: [{pos_m[:, 2].min():.3f}, {pos_m[:, 2].max():.3f}] m")
                
            except Exception as e:
                print(f"  ❌ 加载头模型失败: {e}")
                all_checks_passed = False
    
    # ========================================
    # 检查5: 数据质量
    # ========================================
    print("\n[检查 5/5] 数据质量")
    
    data = epochs.get_data()
    
    print(f"  数据形状: {data.shape}")
    print(f"  数据范围: [{data.min():.6e}, {data.max():.6e}]")
    print(f"  标准差: {data.std():.6e}")
    
    # 检查NaN/Inf
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    
    if has_nan or has_inf:
        print(f"  ❌ 数据包含异常值: NaN={has_nan}, Inf={has_inf}")
        all_checks_passed = False
    else:
        print(f"  ✓ 数据无NaN/Inf")
    
    # 检查幅度
    if data.std() < 1e-6:
        print(f"  ⚠️  数据幅度很小，可能需要调整单位")
    elif data.std() > 1e-3:
        print(f"  ⚠️  数据幅度较大，可能需要调整单位")
    else:
        print(f"  ✓ 数据幅度合理")
    
    # ========================================
    # 配准质量估计
    # ========================================
    if (epochs.info.get('dig') is not None and len(epochs.info['dig']) > 0 and
        trans_file is not None and Path(trans_file).exists() and
        headmodel_file is not None and Path(headmodel_file).exists()):
        
        print("\n[额外] 配准质量估计")
        
        try:
            # 加载trans
            trans = mne.read_trans(str(trans_file))
            T = trans['trans']
            
            # 加载头模型
            data = loadmat(str(headmodel_file), simplify_cells=True)
            headmodel = data.get('headmodel', data)
            head_pos = np.array(headmodel.pos)
            
            if hasattr(headmodel, 'unit') and headmodel.unit == 'mm':
                head_pos = head_pos / 1000.0
            
            # 提取电极位置
            eeg_pos = np.array([
                d['r'] for d in epochs.info['dig']
                if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG
            ])
            
            # 应用变换到电极
            eeg_pos_hom = np.column_stack([eeg_pos, np.ones(len(eeg_pos))])
            eeg_transformed = (T @ eeg_pos_hom.T).T[:, :3]
            
            # 计算到头模型的距离
            from scipy.spatial import cKDTree
            tree = cKDTree(head_pos)
            dists, _ = tree.query(eeg_transformed)
            
            mean_dist = np.mean(dists) * 1000
            p95_dist = np.percentile(dists, 95) * 1000
            
            print(f"  配准后电极到头皮距离:")
            print(f"    平均: {mean_dist:.1f} mm")
            print(f"    95分位: {p95_dist:.1f} mm")
            
            if mean_dist < 10:
                print(f"  ✓ 配准质量: 优秀")
            elif mean_dist < 20:
                print(f"  ✓ 配准质量: 良好")
            elif mean_dist < 30:
                print(f"  ⚠️  配准质量: 可接受 (ROI分析)")
            else:
                print(f"  ❌ 配准质量: 差 (需要重新配准)")
                all_checks_passed = False
                
        except Exception as e:
            print(f"  ⚠️  无法估计配准质量: {e}")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    if all_checks_passed:
        print("✅ 所有检查通过! 可以运行源重建了")
        print("\n建议的命令:")
        print("```python")
        print("from preprocessing.source_reconstruction import run_source_reconstruction_pipeline")
        print()
        print("results = run_source_reconstruction_pipeline(")
        print("    epochs=epochs_clean,")
        print("    headmodel_file=HEAD_MODEL,")
        print("    atlas_dir=ATLAS_DIR,")
        if trans_file:
            print(f"    trans_file='{trans_file}',")
        else:
            print("    trans_file=None,  # ⚠️ 建议提供trans文件")
        print("    method='sLORETA',")
        print("    lambda2=1.0/9.0,")
        print("    noise_cov_method='auto',")
        print("    noise_cov_reg=0.1,")
        print("    max_epochs=200,")
        print("    n_jobs=2")
        print(")")
        print("```")
    else:
        print("❌ 部分检查未通过")
        print("\n请按照上述提示修复问题后再运行源重建")
    
    print("="*70)
    
    return all_checks_passed


# if __name__ == "__main__":
#     print("请在notebook中使用以下代码运行验证:")
#     print("""
#     from source_reconstruction_validation import validate_preparation
    
#     # 基本验证
#     validate_preparation(epochs_clean)
    
#     # 完整验证
#     validate_preparation(
#         epochs=epochs_clean,
#         trans_file='/workspace/shared/data/bids_dataset/derivatives/mne-python/sub-001/sub-001-trans.fif',
#         headmodel_file='/workspace/shared/data/raw/Roessner_Gerhard/headmodel_ROESSNER.mat',
#         coord_xml='/workspace/shared/data/raw/Roessner_Gerhard/eeg/Stim_On_55Hz_Full2.mff/coordinates.xml',
#         plot=True
#     )
#     """)