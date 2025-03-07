import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.gaussians_modify import build_rotation

def load_checkpoint(config, dataset):
    """ 加载训练 Checkpoint 以恢复优化状态 """
    
    checkpoint_time_idx = config['checkpoint_time_idx']
    print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")

    # 1️⃣ 加载存储的参数
    ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}

    # 2️⃣ 初始化变量
    num_gaussians = params['means3D'].shape[0]
    variables = {
        'max_2D_radius': torch.zeros(num_gaussians).cuda().float(),
        'means2D_gradient_accum': torch.zeros(num_gaussians).cuda().float(),
        'denom': torch.zeros(num_gaussians).cuda().float(),
        'timestep': torch.zeros(num_gaussians).cuda().float()
    }

    # 3️⃣ 加载关键帧索引
    keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
    keyframe_time_indices = keyframe_time_indices.tolist()

    # 4️⃣ 重新构建 `gt_w2c_all_frames` 和 `keyframe_list`
    gt_w2c_all_frames = []
    keyframe_list = []
    
    for time_idx in range(checkpoint_time_idx):
        # 加载 RGB-D 数据
        color, depth, _, gt_pose = dataset[time_idx]
        
        # 计算 Ground Truth 世界到相机变换
        gt_w2c = torch.linalg.inv(gt_pose)
        gt_w2c_all_frames.append(gt_w2c)

        # 如果当前帧是关键帧，恢复关键帧信息
        if time_idx in keyframe_time_indices:
            curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = params['cam_trans'][..., time_idx].detach()
            
            # 计算估计的 w2c
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

            # 存储关键帧
            curr_keyframe = {
                'id': time_idx,
                'est_w2c': curr_w2c,
                'color': color.permute(2, 0, 1) / 255,
                'depth': depth.permute(2, 0, 1)
            }
            keyframe_list.append(curr_keyframe)

    return params, variables, checkpoint_time_idx, keyframe_list, gt_w2c_all_frames