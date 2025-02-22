import torch
import numpy as np
from utils.recon_helpers import *
import torch.nn.functional as F
from utils.slam_helpers import transform_to_frame,transformed_params2depthplussilhouette
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.slam_external import build_rotation
import os


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=True):
    """
    1. means3D (num_pts, 3) 3D 位置。
    2. rgb_colors (num_pts, 3) 高斯点云的颜色(RGB)
    3. unnorm_rotations (num_pts, 4) 未归一化的四元数旋转
    4. logit_opacities	(num_pts, 1) 不透明度
    5. log_scales	(num_pts, 1 or 3) 高斯点的尺度（大小）
    6. feature_rest	(num_pts, 45) (可选) 额外特征（如球谐函数）
    7. cam_unnorm_rots	(1, 4, num_frames)	相机旋转（四元数）
    8. cam_trans	(1, 3, num_frames)	相机位移
    建params 字典，记录下高斯点的各个参数情况、以及相机位姿的初始化。
    构建 variables 构建辅助化初始变量。
    """
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    # 相机旋转矩阵四元数，每一帧的初始方向为 [1, 0, 0, 0] 位置为 [0,0,0]
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(), # (num_pts,) 1D张量 表示某个高斯点的最大投影半径。
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, use_simplification=True):
    """ 初始化第一帧的 RGB-D 数据、
    初始化相机参数和变换矩阵，
    加载更高分辨率的 Densification 数据
    生成初始 3D 点云
    初始化神经表示的优化参数
    估算场景的尺度
    """
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3] # 得到相机内参
    w2c = torch.linalg.inv(pose) # 得到相机位姿 (世界坐标系到相机坐标系)

    # Setup Camera 对象
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy(), use_simplification=use_simplification)

    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)

    mask = (depth > 0) & energy_mask(color) # Mask out invalid depth values
    # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio # NOTE: change_here

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam
    
def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    }
    if not use_simplification:
        params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    """初始化当前帧的相机位姿，可以使用恒定模型，或者是直接的复制上面一帧率。"""
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach()) # 上一帧的旋转
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach()) # 上上帧的旋转
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            # 新的一帧的旋转
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            # 新的一帧的位置。
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
            # 直接复制上一帧的位姿。
            
    return params

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    depth_sil, _, _ = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)        
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
    return params, variables

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
