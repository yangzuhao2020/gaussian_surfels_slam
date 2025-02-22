from tqdm import tqdm
from utils.keyframe_selection import keyframe_selection_distance
import numpy as np
import torch
import numpy as np
from utils.recon_helpers import energy_mask
from utils.slam_external import calc_ssim
from utils.slam_helpers import l1_loss_v1
def select_keyframe(time_idx, selected_keyframes, keyframe_list, color, depth, params, config, actural_keyframe_ids, num_iters_mapping):
    """ 
    Selects a keyframe for Mapping, either randomly or using distance-based selection.
    
    Returns:
        iter_time_idx: Selected frame index
        iter_color: Selected frame color
        iter_depth: Selected frame depth
        actural_keyframe_ids: Updated keyframe selection history
    """

    if not config['distance_keyframe_selection']:
        # ✅ Randomly select a keyframe from `selected_keyframes`
        rand_idx = np.random.randint(0, len(selected_keyframes))
        selected_rand_keyframe_idx = selected_keyframes[rand_idx]
        actural_keyframe_ids.append(selected_rand_keyframe_idx)

        if selected_rand_keyframe_idx == -1:
            # Use Current Frame Data
            return time_idx, color, depth, actural_keyframe_ids
        else:
            # Use Keyframe Data
            return keyframe_list[selected_rand_keyframe_idx]['id'], keyframe_list[selected_rand_keyframe_idx]['color'], keyframe_list[selected_rand_keyframe_idx]['depth'], actural_keyframe_ids

    else:
        # ✅ Distance-based keyframe selection
        if len(actural_keyframe_ids) == 0:
            if len(keyframe_list) > 0:
                curr_position = params['cam_trans'][..., time_idx].detach().cpu()
                actural_keyframe_ids = keyframe_selection_distance(time_idx, curr_position, keyframe_list, config['distance_current_frame_prob'], num_iters_mapping)
            else:
                actural_keyframe_ids = [0] * num_iters_mapping

            print(f"\nUsed Frames for mapping at Frame {time_idx}: "
                  f"{[keyframe_list[i]['id'] if i != len(keyframe_list) else 'curr' for i in actural_keyframe_ids]}")

        selected_keyframe_id = actural_keyframe_ids[-1]

        if selected_keyframe_id == len(keyframe_list):
            # Use Current Frame Data
            return time_idx, color, depth, actural_keyframe_ids
        else:
            # Use Keyframe Data
            return keyframe_list[selected_keyframe_id]['id'], keyframe_list[selected_keyframe_id]['color'], keyframe_list[selected_keyframe_id]['depth'], actural_keyframe_ids


def should_continue_tracking(iter, num_iters_tracking, losses, config, do_continue_slam, progress_bar, time_idx):
    """ 判断 Tracking 是否应该继续优化 """
    # 检查是否达到最大迭代次数
    if iter == num_iters_tracking:
        # 1️⃣ 如果深度误差足够小，提前终止
        if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
            return False, iter, num_iters_tracking, do_continue_slam, progress_bar

        # 2️⃣ 启用 do_continue_slam，扩展 Tracking 迭代次数
        elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
            do_continue_slam = True
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            num_iters_tracking = 2 * num_iters_tracking  # 扩展迭代次数
            return True, iter, num_iters_tracking, do_continue_slam, progress_bar

        # 3️⃣ 否则直接终止 Tracking
        else:
            return False, iter, num_iters_tracking, do_continue_slam, progress_bar

    return True, iter, num_iters_tracking, do_continue_slam, progress_bar

def compute_valid_depth_mask(depth, curr_data_depth, ignore_outlier_depth_loss=True):
    """ 计算有效的深度 Mask，过滤 NaN、背景区域和异常深度值 """
    # 1️⃣ 过滤无效像素（NaN & 背景区域）
    valid_depth_mask = curr_data_depth > 0
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(curr_data_depth))
    bg_mask = energy_mask(curr_data_depth)  # 背景过滤

    # 2️⃣ 处理异常深度值（±2σ 过滤）
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data_depth - depth)  # 计算深度误差
        mean_error = depth_error.mean()
        std_error = depth_error.std()
        outlier_threshold_low = mean_error - 2 * std_error
        outlier_threshold_high = mean_error + 2 * std_error
        outlier_mask = (depth_error > outlier_threshold_low) & (depth_error < outlier_threshold_high)
        valid_depth_mask &= outlier_mask  # 结合异常值过滤

    # 3️⃣ 合并所有 Mask
    return valid_depth_mask & nan_mask & bg_mask


def compute_depth_loss(tracking, depth, curr_data, mask):
    """ 计算深度损失（Tracking: sum, Mapping: mean）"""
    mask = mask.detach()  # 避免梯度影响
    loss = torch.abs(curr_data['depth'] - depth)[mask]
    return loss.sum() if tracking else loss.mean()

def compute_rgb_loss(im, curr_data, mask, tracking, use_sil_for_loss, ignore_outlier_depth_loss):
    """
    计算 RGB 颜色损失：
    - Tracking 阶段：使用 `L1 Loss`，可选 `mask` 过滤前景区域。
    - Mapping 阶段：使用 `0.8 * L1 + 0.2 * SSIM` 作为颜色损失。
    """
    if tracking:
        # 仅 Tracking 阶段可能使用 mask 过滤
        if use_sil_for_loss or ignore_outlier_depth_loss:
            color_mask = torch.tile(mask, (3, 1, 1)).detach()  # 扩展 Mask 适用于 RGB
            return torch.abs(curr_data['im'] - im)[color_mask].sum()
        return torch.abs(curr_data['im'] - im).sum()

    # Mapping 阶段，使用 L1 + SSIM
    return 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))


