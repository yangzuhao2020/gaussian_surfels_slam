from tqdm import tqdm
from utils.keyframe_selection import keyframe_selection_distance
import numpy as np
import time

import numpy as np

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
