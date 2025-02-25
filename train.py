import argparse
import os
import shutil
import time
from importlib.machinery import SourceFileLoader
from configs.configs import *
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.tracking_and_mapping import *
from utils.initial import *
from datasets.gradslam_datasets import  EndoSLAMDataset, C3VDDataset
from utils.common_utils import seed_everything, save_params_ckpt, save_params, save_means3D
from utils.eval_helpers import report_progress, eval_save, compute_average_runtimes, save_final_parameters
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.vis_utils import plot_video
from utils.time_helper import Timer
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    """根据数据集名称, 返回相应的数据集类实例""" 
    if config_dict["dataset_name"].lower() == "endoslam_unity":
        return EndoSLAMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() == "c3vd":
        return C3VDDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def initialize_optimizer(params, lrs_dict):
    """初始化 adam 优化器，传入待优化的参数以及学习率字典。"""
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k != 'feature_rest']
    if 'feature_rest' in params:
        param_groups.append({'params': [params['feature_rest']], 'name': 'feature_rest', 'lr': lrs['rgb_colors'] / 20.0})
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, 
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    """"# loss 总损失； weighted_losses 包含各种损失。
    loss 损失计算需要区分 tracking 和 mapping 
    由于 Tracking 只优化相机位姿，误差计算方式是 像素级别的 L1 误差 所以 会逐个像素求和。
    由于 mapping 优化整个3D 场景，所以会去平均而不是求和。
    """
    global w2cs, w2ci
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                         gaussians_grad=False, 
                                         camera_grad=True)  # 仅优化相机位姿
    else:  # mapping 有两种情况 do_ba 会决定是否捆绑优化，默认为false.
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                            gaussians_grad=True, 
                                            camera_grad=do_ba)  # do_ba 决定是否优化相机

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    # 将变换后的高斯参数转换为可渲染变量。
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    # 计算用于深度和轮廓渲染的变量。
    # Visualize the Rendered Images
    # online_render(curr_data, iter_time_idx, rendervar, dev_use_controller=False)
        
    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D'] # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    bg_mask = energy_mask(curr_data['im'])
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        # 计算 depth_error（深度误差）
        mask = (depth_error < 20 * depth_error.mean())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask & bg_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses

def new_get_loss(params, curr_data, iter_time_idx, sil_thres, use_sil_for_loss=True,
                 tracking=False, mapping=False, do_ba=False, ignore_outlier_depth_loss=True):
    if tracking:
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                         gaussians_grad=False, 
                                         camera_grad=True)  # 仅优化相机位姿
    else:  # mapping 有两种情况 do_ba 会决定是否捆绑优化，默认为false.
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                            gaussians_grad=True, 
                                            camera_grad=do_ba)  # do_ba 决定是否优化相机

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    # 将变换后的高斯参数转换为可渲染变量，即返回一个可渲染变量字典。
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],transformed_pts)
    
    # tracking loss
    rendervar['means2D'].retain_grad()
    im, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    
    # Depth & Silhouette Rendering
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0) # 渲染得到的 深度图（对应高斯点投影的深度值）。
    silhouette = depth_sil[1, :, :] # 渲染得到的 轮廓图（Silhouette）
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0) # depth_sil[2, :, :] 存储的是 depth²（深度平方的均值）。
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()
    
    # 计算 Mask
    mask = compute_valid_depth_mask(depth, curr_data['depth'], ignore_outlier_depth_loss)
    # 额外的 Tracking 过滤（只优化前景）
    if tracking and use_sil_for_loss:
        mask &= presence_sil_mask
        
    mask = mask.detach()  # 避免梯度影响

    # 计算 Depth Loss
    loss_depth = compute_depth_loss(tracking, depth, curr_data, mask)
    loss_rgb = None  # 先初始化，避免非 Tracking 时变量未定义
    loss_opac = compute_opacity_loss(params)
    loss_monoN = None
    loss_depth_normal = None
    # 计算 RGB Loss
    loss_rgb = compute_rgb_loss(im, curr_data, mask, tracking, use_sil_for_loss, ignore_outlier_depth_loss)
        
    loss_weights = {'rgb': 1.0,
                    'opac':1.0,
                    'monoN':1.0,
                    'depth':1.0,
                    'depth_normal':1.0}
    # 计算加权总损失
    loss = (loss_weights['rgb'] * loss_rgb + 
            loss_weights['opac'] * loss_opac + 
            loss_weights['monoN'] * loss_monoN + 
            loss_weights['depth_normal'] * loss_depth_normal+
            loss_weights['depth'] * loss_depth)
    
    return loss
    
def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def training(config: dict):
    print("Training starts.")
    
    # Get Device
    device = torch.device(config["primary_device"])

    config = setup_config_defaults(config)

    output_dir, eval_dir = setup_directories() # 实验结果输出文件。

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]

    dataset_config, gradslam_data_cfg, seperate_densification_res, seperate_tracking_res = setup_dataset_config(dataset_config)
    
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
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    if dataset_config["train_or_test"] == 'train': # kind of ill implementation here. train_or_test should be 'all' or 'train'. If 'test', you view test set as full dataset.
        eval_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"], # if you eval, you should keep reso as raw image.
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test='test'
        )# 用于评估结果。
    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset, 
                                                                        use_simplification=config['gaussian_simplification'])                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'], 
                                                                                        use_simplification=config['gaussian_simplification'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            preload = dataset_config["preload"],
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            train_or_test=dataset_config["train_or_test"]
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0] # 获取第一帧信息。
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(
            tracking_color.shape[2],  # 宽度 W
            tracking_color.shape[1],  # 高度 H
            tracking_intrinsics.cpu().numpy(),  # 相机内参 (转换为 NumPy)
            first_frame_w2c.detach().cpu().numpy(),  # 第一帧相机位姿 (世界坐标到相机坐标变换)
            use_simplification=config['gaussian_simplification']
        )
    
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

    
     # 1️⃣ 处理 Checkpoint 加载
    if config['load_checkpoint']:
        params, variables, checkpoint_time_idx, keyframe_list, gt_w2c_all_frames = load_checkpoint(config, dataset)
    else:
        checkpoint_time_idx = 0
        keyframe_list = []
        gt_w2c_all_frames = []
        
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # timer.lap("iterating over frame "+str(time_idx), 0)
        # always show global iteration
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose) # 把世界坐标系的点转换到当前相机坐标系的矩阵。
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {
            'cam': cam,                 # 当前帧的相机信息
            'im': color,                # 当前帧的 RGB 图像 (归一化)
            'depth': depth,              # 当前帧的深度图
            'id': iter_time_idx,         # 当前帧索引 (time_idx)
            'intrinsics': intrinsics,    # 相机内参 (fx, fy, cx, cy)
            'w2c': first_frame_w2c,      # 参考帧的相机位姿 (世界到相机)
            'iter_gt_w2c_list': curr_gt_w2c  # 记录所有帧的 gt_w2c（世界到相机的变换）也是所有真实位姿的情况。
        }

        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, # Tracking 任务的 相机对象，包含相机的投影参数
                                  'im': tracking_color, 
                                  'depth': tracking_depth, 
                                  'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 
                                  'w2c': first_frame_w2c, 
                                  'iter_gt_w2c_list': curr_gt_w2c
                                  } 
            # Tracking 过程中当前帧的关键信息
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations 
        num_iters_mapping = config['mapping']['num_iters'] # 高斯点的优化，每帧优化的次数。
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
            # 这里表示 相机的位姿。
        # timer.lap("initialized data", 1)

        # Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            """time_idx > 0 → 只有在 Tracking 第二帧及以后才会执行（第 0 帧不做 Tracking）。"""
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'])
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            # 此时的旋转矩阵。
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            # 此时的平移矩阵。
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters'] # 相机优化的次数。
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                """计算当前帧的损失 (get_loss())，用于优化 相机位姿 和 3D 高斯点云。
                执行梯度更新 (backward() + step())，优化 相机位姿 使得重投影误差最小。
                保存最佳相机位姿 (candidate_cam_unnorm_rot, candidate_cam_tran)。
                检查是否应该停止 Tracking (should_continue_tracking())，如果满足停止条件，则退出 while True 循环。
                统计运行时间 (tracking_iter_time_sum)，用于评估优化性能。"""
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, # 所有（渲染出的）的高斯点和相机位姿
                                                   tracking_curr_data, # 当前帧的情况
                                                   variables, # 辅助初始变量。
                                                   iter_time_idx, # 帧数情况。
                                                   config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], 
                                                   config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], 
                                                   config['tracking']['ignore_outlier_depth_loss'], 
                                                   tracking=True, 
                                                   plot_dir=eval_dir, 
                                                   visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                        # 记录当前最佳的相机位姿
                    # Report Progress
                    if config['report_iter_progress']:
                        report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                        
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                
                # Check if we should stop tracking
                iter += 1
                tracking_active, iter, num_iters_tracking, do_continue_slam, progress_bar = should_continue_tracking(
                    iter, num_iters_tracking, losses, config, do_continue_slam, progress_bar, time_idx)

                if not tracking_active:
                    break  # 终止 Tracking 过程

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
                
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            """使用真实位姿，为了不优化位姿。"""
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
                
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # timer.lap("tracking done", 2)

        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # 只有第一个帧的时候和每隔一个config['map_every'] 执行。
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res: # 控制是否使用更高分辨率的 densify_dataset
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 
                                         'im': densify_color, 
                                         'depth': densify_depth, 
                                         'id': time_idx, 
                                         'intrinsics': densify_intrinsics, 
                                         'w2c': first_frame_w2c, 
                                         'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # delete floating gaussians
                # params, variables = remove_floating_gaussians(params, variables, densify_curr_data, time_idx)
                
                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], 
                                                      config['gaussian_simplification'])
                post_num_pts = params['means3D'].shape[0]
            
            if not config['distance_keyframe_selection']:
                with torch.no_grad():
                    # 1️⃣ 归一化当前帧的相机旋转
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    # 2️⃣ 构造当前帧的世界到相机变换矩阵 (w2c)
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # 3️⃣ 选择最相关的关键帧
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
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
            optimizer = initialize_optimizer(params, config['mapping']['lrs']) 

            # timer.lap("Densification Done at frame "+str(time_idx), 3)

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
                                                                                color, 
                                                                                depth, 
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
                             'w2c': first_frame_w2c, 
                             'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, 
                                                iter_data, 
                                                variables, 
                                                iter_time_idx, 
                                                config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], 
                                                config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], 
                                                config['mapping']['ignore_outlier_depth_loss'], 
                                                mapping=True)
                # Backprop
                loss.backward()
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
                    if config['report_iter_progress']:
                        report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                        mapping=True, online_time_idx=time_idx)
                    else:
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
        is_last_valid_frame = (time_idx == num_frames - 2)

        # 2️⃣ 确保 GT 位姿不是无效值
        gt_w2c_valid = not (torch.isinf(curr_gt_w2c[-1]).any() or torch.isnan(curr_gt_w2c[-1]).any())

        # 3️⃣ 组合最终的关键帧存储条件
        if (is_first_frame or is_keyframe_interval or is_last_valid_frame) and gt_w2c_valid:
            """第一帧 time_idx=0 强制存入关键帧列表。
            每 keyframe_every 帧存储一个关键帧。
            time_idx == num_frames-2 倒数第二帧 强制存入关键帧列表，确保 Tracking 结束时仍然有关键帧可用。"""
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
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
    dataset = [dataset, eval_dataset, 'C3VD'] if dataset_config["train_or_test"] == 'train' else dataset
    with torch.no_grad():
        eval_save(dataset, params, eval_dir, sil_thres=config['mapping']['sil_thres'],
                mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'])

    # 2️⃣ 评估最终参数
    params = save_final_parameters(dataset, eval_dataset, config["data"])
    # Save Parameters
    save_params(params, output_dir)
    save_means3D(params['means3D'], output_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    # parser.add_argument("--online_vis", action="store_true", help="Visualize mapping renderings while running")

    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()

    # Prepare dir for visualization
    # if args.online_vis:
    #     vis_dir = './online_vis'
    #     os.makedirs(vis_dir, exist_ok=True)
    #     for filename in os.listdir(vis_dir):
    #         os.unlink(os.path.join(vis_dir, filename))

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    training(experiment.config)
    
    plot_video(os.path.join(results_dir, 'eval', 'plots'), os.path.join('./experiments/', experiment.group_name, experiment.scene_name, 'keyframes'))