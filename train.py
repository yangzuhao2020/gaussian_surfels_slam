import os
import time
from importlib.machinery import SourceFileLoader
from configs.configs import *
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.tracking_and_mapping import *
from datasets.gradslam_datasets import  EndoSLAMDataset, C3VDDataset
from configs import ModelParams
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save, compute_average_runtimes, save_final_parameters
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame_3d, add_new_gaussians, matrix_to_quaternion)
from utils.slam_external import build_rotation, prune_gaussians, densify
from scene import Scene
from scene. import GaussianModel
from gaussian_render import  render
import argparse


def train(config: dict, arg=None):
    gaussians = GaussianModel(arg)
    device = torch.device(config["primary_device"])
    scene = Scene(arg, gaussians=gaussians, resolution_scales=1)
    config = setup_config_defaults(config)
    output_dir, eval_dir = setup_directories(config) # 实验结果输出文件。
    dataset_config = config["data"]
    dataset_config, gradslam_data_cfg = setup_dataset_config(dataset_config)
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,  # 数据集的配置字典
        basedir=dataset_config["basedir"],  # 数据集的基本目录
        sequence=os.path.basename(dataset_config["sequence"]),  # 数据序列的名称
        start=dataset_config["start"],  # 开始的帧索引
        end=dataset_config["end"],  # 结束的帧索引
        stride=dataset_config["stride"],  # 采样步长（跳过的帧数）
        desired_height=dataset_config["desired_image_height"],  # 目标图像高度
        desired_width=dataset_config["desired_image_width"],  # 目标图像宽度
        device=device,  # 运行设备（如 CPU 或 GPU）
        relative_pose=True,  # 让位姿相对于第一帧
        ignore_bad=dataset_config["ignore_bad"],  # 是否忽略损坏的帧
        use_train_split=dataset_config["use_train_split"],  # 是否使用训练集划分
        train_or_test=dataset_config["train_or_test"]  # 选择训练模式或测试模式
    )
    total_num_frames = dataset_config["num_frames"]
    if total_num_frames == -1:
        total_num_frames = len(dataset)                  

    # 这里已经开始初始化高斯点了。
    # Initialize Parameters & Canoncial Camera parameters 
    variables, intrinsics, first_frame_w2c_gt, cam = gaussians.initialize_first_timestep(
                                                    dataset, 
                                                    total_num_frames,
                                                    config['scene_radius_depth_ratio'])
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    checkpoint_time_idx = 0
    keyframe_list = []
    gt_w2c_all_frames = []
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, total_num_frames)):
        gt_rgb, gt_depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose) # 把世界坐标系的点转换到当前相机坐标系的矩阵。
        # Process RGB-D Data
        gt_rgb = gt_rgb.permute(2, 0, 1) / 255
        gt_depth = gt_depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {
            'cam': cam,                 # 当前帧的相机信息
            'im': gt_rgb,                # 当前帧的 RGB 图像 (归一化)
            'depth': gt_depth,              # 当前帧的深度图
            'id': iter_time_idx,         # 当前帧索引 (time_idx)
            'intrinsics': intrinsics,    # 相机内参 (fx, fy, cx, cy)
            'w2c': first_frame_w2c_gt,      # 参考帧的相机位姿 (世界到相机)
            'iter_gt_w2c_list': curr_gt_w2c  # 记录所有帧的 gt_w2c（世界到相机的变换）也是所有真实位姿的情况。
        }
        # Optimization Iterations 
        num_iters_mapping = config['mapping']['num_iters'] # 高斯点的优化，每帧优化的次数。
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            gaussians.initialize_camera_pose(time_idx)
            # 这里表示初始化相机的位姿。
        # timer.lap("initialized data", 1)

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            """time_idx > 0 → 只有在 Tracking 第二帧及以后才会执行（第 0 帧不做 Tracking）。"""
            # Reset Optimizer & Learning Rates for tracking
            optimizer, scheduler= gaussians.initialize_optimizer(config['tracking']['lrs'], mode="tracking")
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = gaussians._cam_rots[..., time_idx].detach().clone()
            # 此时的旋转矩阵。
            candidate_cam_tran = gaussians._cam_trans[..., time_idx].detach().clone()
            # 此时的平移矩阵。
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking_cam = config['tracking']['num_iters'] # 相机优化的次数。
            progress_bar = tqdm(range(num_iters_tracking_cam), desc=f"Tracking Time Step: {time_idx}")
            while True:
                """计算当前帧的损失 (get_loss())，用于优化 相机位姿 和 3D 高斯点云。
                执行梯度更新 (backward() + step())，优化 相机位姿 使得重投影误差最小。
                保存最佳相机位姿 (candidate_cam_unnorm_rot, candidate_cam_tran)。
                检查是否应该停止 Tracking (should_continue_tracking())，如果满足停止条件，则退出 while True 循环。
                统计运行时间 (tracking_iter_time_sum)，用于评估优化性能。
                """
                iter_start_time = time.time()
                # Loss for current frame
                
                loss, loss_dict = new_get_loss(
                                    curr_data, # 所有（渲染出的）的高斯点和相机位姿
                                    iter_time_idx, # 帧数情况。
                                    config['tracking']['sil_thres'],
                                    gaussians,
                                    config['loss_weights'],
                                    intrinsics,
                                    tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                gaussians.apply_gradient_mask()  # 屏蔽 Z 轴梯度
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()  # 调整学习率
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = gaussians._cam_rots[..., time_idx].detach().clone()
                        candidate_cam_tran = gaussians._cam_trans[..., time_idx].detach().clone()
                    progress_bar.update(1)
                        
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                
                # Check if we should stop tracking
                iter += 1
                tracking_active, iter, num_iters_tracking_cam, do_continue_slam, progress_bar = should_continue_tracking(
                    iter, num_iters_tracking_cam, loss_dict, config, do_continue_slam, progress_bar, time_idx)

                if not tracking_active:
                    break  # 终止 Tracking 过程

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                gaussians._cam_rots[..., time_idx] = candidate_cam_unnorm_rot
                gaussians._cam_trans[..., time_idx] = candidate_cam_tran
                
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            """使用真实位姿，为了不优化位姿。"""
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot) # 将矩阵转换为四元数
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                gaussians._cam_rots[..., time_idx] = rel_w2c_rot_quat
                gaussians._cam_trans[..., time_idx] = rel_w2c_tran
                
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # timer.lap("tracking done", 2)

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # 只有第一个帧的时候和每隔一个config['map_every'] 执行。
            # Densification
            # delete floating gaussians
            # params, variables = remove_floating_gaussians(params, variables, densify_curr_data, time_idx)
            
            # Add new Gaussians to the scene based on the Silhouette
            variables = add_new_gaussians(gaussians, 
                                        curr_data, 
                                        config['mapping']['sil_thres'], 
                                        time_idx,
                                        variables)
            post_num_pts = gaussians._xyz.shape[0]
            
            if not config['distance_keyframe_selection']:
                with torch.no_grad():
                    # 1️⃣ 归一化当前帧的相机旋转
                    curr_cam_rot = F.normalize(gaussians._cam_rots[..., time_idx].detach())
                    curr_cam_tran = gaussians._cam_trans[..., time_idx].detach()
                    # 2️⃣ 构造当前帧的世界到相机变换矩阵 (w2c)
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # 3️⃣ 选择最相关的关键帧
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(gt_depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                    # 4️⃣ 获取关键帧时间索引
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    # 5️⃣ 确保最后一个关键帧被选择
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # 6️⃣ 添加当前帧到关键帧列表
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # 7️⃣ 打印最终选择的关键帧
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")

            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = gaussians.initialize_optimizer(config['tracking']['lrs']) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
                
            actural_keyframe_ids = []
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Replace the messy code in `training()`
                iter_time_idx, iter_color, iter_depth, actural_keyframe_ids = select_keyframe(
                                                                                time_idx, 
                                                                                selected_keyframes, 
                                                                                keyframe_list, 
                                                                                gt_rgb, 
                                                                                gt_depth, 
                                                                                params, 
                                                                                config, 
                                                                                actural_keyframe_ids, 
                                                                                num_iters_mapping)
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 
                             'im': iter_color, 
                             'depth': iter_depth, 
                             'id': iter_time_idx, 
                             'intrinsics': intrinsics, 
                             'w2c': first_frame_w2c_gt, 
                             'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, loss_dict = new_get_loss(
                                    curr_data, 
                                    iter_time_idx,
                                    config['mapping']['sil_thres'],
                                    gaussians, 
                                    config['loss_weights'],
                                    intrinsics,
                                    tracking=False)
                # Backprop
                loss.backward()
                gaussians.apply_gradient_mask()  # 屏蔽 Z 轴梯度
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    # if config['report_iter_progress']:
                    #     report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                    #                     mapping=True, online_time_idx=time_idx)
                    # else:
                    progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
                
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # timer.lap('Mapping Done.', 4)
        
        # Add frame to keyframe list
        # 1️⃣ 关键帧存储的基本条件
        is_first_frame = (time_idx == 0)
        is_keyframe_interval = ((time_idx + 1) % config['keyframe_every'] == 0)
        is_last_valid_frame = (time_idx == total_num_frames - 2)

        # 2️⃣ 确保 GT 位姿不是无效值
        gt_w2c_valid = not (torch.isinf(curr_gt_w2c[-1]).any() or torch.isnan(curr_gt_w2c[-1]).any())

        # 3️⃣ 组合最终的关键帧存储条件
        if (is_first_frame or is_keyframe_interval or is_last_valid_frame) and gt_w2c_valid:
            """第一帧 time_idx=0 强制存入关键帧列表。
            每 keyframe_every 帧存储一个关键帧。
            time_idx == num_frames-2 倒数第二帧 强制存入关键帧列表，确保 Tracking 结束时仍然有关键帧可用。"""
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(gaussians._cam_rots[..., time_idx].detach())
                curr_cam_tran = gaussians._cam_trans[..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': gt_rgb, 'depth': gt_depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        torch.cuda.empty_cache()

    # timer.end()

    # 1️⃣ 计算运行时间
    compute_average_runtimes(
        tracking_iter_time_sum, tracking_iter_time_count,
        tracking_frame_time_sum, tracking_frame_time_count,
        mapping_iter_time_sum, mapping_iter_time_count,
        mapping_frame_time_sum, mapping_frame_time_count,
        output_dir
    )
    # Evaluate Final Parameters
    dataset = [dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(dataset, params, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'])

    # 2️⃣ 评估最终参数
    params = save_final_parameters(dataset, config["data"])
    # Save Parameters
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)

def get_dataset(config_dict, basedir, sequence, **kwargs):
    """根据数据集名称, 返回相应的数据集类实例""" 
    if config_dict["dataset_name"].lower() == "endoslam_unity":
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() == "c3vd":
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def new_get_loss(curr_data, iter_time_idx, sil_thres, 
                 gaussians:GaussianModel, loss_weights, intrinsics,
                 use_sil_for_loss=True, do_ba=False, 
                 ignore_outlier_depth_loss=True, tracking=False):
    if tracking:
        transformed_pts = transform_to_frame_3d(gaussians, iter_time_idx, 
                                            gaussians_grad=False, 
                                            camera_grad=True)  # 仅优化相机位姿
        
    else:  # mapping 有两种情况 do_ba 会决定是否捆绑优化，默认为false.
        transformed_pts = transform_to_frame_3d(gaussians, iter_time_idx, 
                                            gaussians_grad=True, 
                                            camera_grad=do_ba)  # do_ba 决定是否优化相机

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(transformed_pts)
    # 将变换后的高斯参数转换为可渲染变量，即返回一个可渲染变量字典。
    depth_sil_rendervar = transformed_params2depthplussilhouette(curr_data['w2c'],transformed_pts)
    
    # tracking loss
    rendervar['means2D'].retain_grad()
    # im, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    render_pkg = render(curr_data['cam'], 
                        gaussians,
                        pipe, 
                        background, 
                        patch_size)
    # Depth & Silhouette Rendering
    # render_depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    render_depth = render_depth_sil[0, :, :].unsqueeze(0) # 渲染得到的 深度图（对应高斯点投影的深度值）。
    silhouette = render_depth_sil[1, :, :] # 渲染得到的 轮廓图（Silhouette）
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = render_depth_sil[2, :, :].unsqueeze(0) # depth_sil[2, :, :] 存储的是 depth²（深度平方的均值）。
    uncertainty = depth_sq - render_depth**2
    uncertainty = uncertainty.detach()
    
    # 计算 Mask
    mask = compute_valid_depth_mask(render_depth, curr_data['depth'], ignore_outlier_depth_loss)
    # 额外的 Tracking 过滤（只优化前景）
    if tracking and use_sil_for_loss:
        mask &= presence_sil_mask
        
    mask = mask.detach()  # 避免梯度影响
    depth_normal = depth_to_normal(render_depth, mask, intrinsics)
    # 计算 Depth Loss
    loss_depth = compute_depth_loss(tracking, render_depth, curr_data, mask)
    loss_opac = compute_opacity_loss(gaussians._opacity)
    loss_depth_normal = None
    # 计算 RGB Loss
    loss_rgb = compute_rgb_loss(im, curr_data, mask, tracking, use_sil_for_loss, ignore_outlier_depth_loss)
    
    losses = {
        'rgb': loss_weights['rgb'] * loss_rgb,
        'opac': loss_weights['opac'] * loss_opac,
        'depth': loss_weights['depth_normal'] * loss_depth_normal,
        'depth_normal': loss_weights['depth'] * loss_depth
    }
    # 计算加权总损失
    loss = sum(losses.values())

    return loss, losses
    
def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    load_params = ModelParams(parser)
    parser.add_argument("experiment", type=str, 
                        help="Path to experiment file", nargs="?",
                        default="configs/c3vd/c3vd_base.py")
    # parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")
    args = parser.parse_args()
    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    # if not experiment.config['load_checkpoint']:
    #     os.makedirs(results_dir, exist_ok=True)
    #     shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    lp_params = load_params.extract(args)
    print("lp.extract(args): ", lp_params.__dict__)  # 打印对象内部所有属性和值
    
    train(experiment.config, load_params.extract(args))
    print("\nTraining complete.")
    