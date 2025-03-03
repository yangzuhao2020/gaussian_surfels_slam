#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from recon_utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, normal2rotation
from torch import nn
from utils.common_utils import slerp
from torch.utils.cpp_extension import load
from recon_utils.general_utils import quaternion2rotmat
from recon_utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from utils.recon_helpers import setup_camera, energy_mask


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args):
        self.active_sh_degree = 0
        # self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._cam_rots = torch.empty(0) # 3D camera rotations
        self._cam_trans = torch.empty(0) # 3D camera translations
        self.max_radii2D = torch.empty(0)
        # self.xyz_gradient_accum = torch.empty(0)
        # self.scale_gradient_accum = torch.empty(0)
        # self.rot_gradient_accum = torch.empty(0)
        # self.opac_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.setup_functions()
        # self.utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
        self.opac_reset_record = [0, 0]

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            # self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            # self.xyz_gradient_accum,
            # self.scale_gradient_accum,
            # self.rot_gradient_accum,
            # self.opac_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        # self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # print(self._scaling)
        return self.scaling_activation(self._scaling)
        # scaling_2d = torch.cat([self._scaling[..., :2], torch.full_like(self._scaling[..., 2:], -1e10)], -1)
        # return self.scaling_activation(scaling_2d)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    # @property
    # def _xyz(self):
    #     return self._xyz
    
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     # features_rest = self._features_rest
    #     return 
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    @property
    def get_normal(self):
        return quaternion2rotmat(self.get_rotation)[..., 2]


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def compute_normals_cross_product(self, pts, width, height):
        """ 通过 3D 叉乘计算法线 同时归一化"""
        pts = pts.view(height, width, 3)  # 变成图像形状 (H, W, 3)

        # 计算相邻像素的 3D 坐标差分
        v1 = pts[1:, :-1, :] - pts[:-1, :-1, :]  # y 方向的梯度
        v2 = pts[:-1, 1:, :] - pts[:-1, :-1, :]  # x 方向的梯度

        # 计算叉乘得到法线
        normals = torch.cross(v1, v2, dim=-1)
        normals = F.normalize(normals, dim=-1)  # 归一化

        # 填充边界（因为叉乘少了一行一列）
        padded_normals = torch.zeros((height, width, 3), device=pts.device)
        padded_normals[:-1, :-1, :] = normals  # 复制有效区域

        return padded_normals.reshape(-1, 3)  # 变回 (N, 3)

        
    def create_pcd(self, color, depth, intrinsics, w2c, mask=None):   
        """ 创建高斯点云 从点云中初始化，设置好初始的各个参数情况。设置梯度追踪。"""
        width, height = color.shape[2], color.shape[1]
        CX = intrinsics[0][2]  # 主点 x 坐标
        CY = intrinsics[1][2]  # 主点 y 坐标
        FX = intrinsics[0][0]  # 焦距 x
        FY = intrinsics[1][1]  # 焦距 y

        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0].reshape(-1)

        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)  # (X, Y, Z, 1)
        c2w = torch.inverse(w2c)  # 计算世界坐标变换矩阵
        pts = (c2w @ pts4.T).T[:, :3]  # 从相机坐标变换到世界坐标

        scale_gaussian = depth_z / ((FX + FY)/2)
        scales = scale_gaussian.square().unsqueeze(-1).repeat(1, 3)
        
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pts)).float().cuda()), 0.0000001)
        # 将点云坐标转换为 PyTorch 张量 限制输出的值最小为 0.0000001 返回 每个点的 KNN 平均距离。 注意这里初始化的时候取了对数。NOTE: 为什么要取对数？
        # scales = torch.log(torch.sqrt(dist2 / 4))[...,None].repeat(1, 3)
        # scales 是一个 形状为 [N, 3] 的张量, 用于缩放点云的 x, y, z 坐标, 也同时可以表达密度。
        # scales = torch.log(torch.ones((len(fused_point_cloud), 3)).cuda() * 0.02)
        # 计算点云法线
        
        normals = self.compute_normals_cross_product(pts, width, height)

        rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
        scales[..., -1] -= 1e10 # squeeze z scaling
        # 抑制Z方向的缩放。
        
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
        
        # Select points based on mask
        if mask is not None:
            mask = mask.reshape(-1).bool()  # 确保 mask 形状正确
            cols = cols[mask]
            pts = pts[mask]
            new_points_count = pts.shape[0]
            normals = normals[mask]
            scales = scales[mask]
            rots = rots[mask]

        opacities = inverse_sigmoid(0.1 * torch.ones((pts.shape[0], 1), dtype=torch.float, device="cuda"))
        # 初始化透明度。

         # **保留已有点云，并拼接新点云**
        if self._xyz.shape[0] > 0: 
            self._xyz = nn.Parameter(torch.cat([self._xyz, pts]).requires_grad_(True))
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, cols]).requires_grad_(True))
            self._scaling = nn.Parameter(torch.cat([self._scaling, scales]).requires_grad_(True))
            self._rotation = nn.Parameter(torch.cat([self._rotation, rots]).requires_grad_(True))
            self._opacity = nn.Parameter(torch.cat([self._opacity, opacities]).requires_grad_(True))
            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(pts.shape[0], device="cuda")])
            return new_points_count
        else:
            self._xyz = nn.Parameter(pts.requires_grad_(True))
            self._features_dc = nn.Parameter(cols.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros(pts.shape[0], device="cuda")
    
    def initialize_cams(self, total_num_frames):
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        device = torch.device("cuda")  # 指定计算设备为 GPU

        # 初始化单位四元数 (1, 4, num_frames)，表示无旋转
        cam_rots = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device).reshape(1, 4, 1)
        cam_rots = cam_rots.expand(1, 4, total_num_frames)  # 在最后一个维度复制 num_frames 次

        # 初始化相机平移 (1, 3, num_frames)，全部为 0
        cam_trans = torch.zeros((1, 3, total_num_frames), dtype=torch.float32, device=device)

        self._cam_rots = torch.nn.Parameter(cam_rots.contiguous().requires_grad_(True))
        self._cam_trans = torch.nn.Parameter(cam_trans.contiguous().requires_grad_(True))

        variables = {'max_2D_radius': torch.zeros(self._xyz.shape[0]).cuda().float(), # (num_pts,) 1D张量 表示某个高斯点的最大投影半径。
                    'means2D_gradient_accum': torch.zeros(self._xyz.shape[0]).cuda().float(),
                    'denom': torch.zeros(self._xyz.shape[0]).cuda().float(),
                    'timestep': torch.zeros(self._xyz.shape[0]).cuda().float()}

        return variables
    
    # variables 没有被记录到优化器上，所以不会被优化。
    def initialize_first_timestep(self, dataset, total_num_frames, scene_radius_depth_ratio):
        """ 
        初始化第一帧的 RGB-D 数据、
        初始化相机参数和变换矩阵，
        加载更高分辨率的 Densification 数据
        生成初始 3D 点云
        初始化神经表示的优化参数
        估算场景的尺度
        """
        # Get RGB-D Data & Camera Parameters
        gt_rgb, gt_depth, intrinsics, gt_pose = dataset[0]

        # Process RGB-D Data
        gt_rgb = gt_rgb.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        gt_depth = gt_depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        
        # Process Camera Parameters
        intrinsics = intrinsics[:3, :3] # 得到相机内参
        gt_w2c = torch.linalg.inv(gt_pose) # 得到相机位姿 (世界坐标系到相机坐标系)

        # Setup Camera 对象
        w = gt_rgb.shape[2]
        h = gt_rgb.shape[1]
        cam = setup_camera(w, h, intrinsics.cpu().numpy(), gt_w2c.detach().cpu().numpy())

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (gt_depth > 0) & energy_mask(gt_rgb) # Mask out invalid depth values
        # Image.fromarray(np.uint8(mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')
        mask = mask.reshape(-1)
        self.create_pcd(gt_rgb, gt_depth, gt_w2c, intrinsics, mask)

        # Initialize cams
        variables = self.initialize_cams(total_num_frames)

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        variables['scene_radius'] = torch.max(gt_depth)/scene_radius_depth_ratio # NOTE: change_here

        return variables, intrinsics, gt_w2c, cam

    def initialize_camera_pose(self, curr_time_idx, forward_prop = True):
        """初始化当前帧的相机位姿，可以使用恒定模型，或者是直接的复制上面一帧率。"""
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                # Rotation
                prev_rot1 = self._cam_rots[..., curr_time_idx-1].detach()  # 上一帧的旋转 (1, 4)
                prev_rot2 = self._cam_rots[..., curr_time_idx-2].detach()  # 上上帧的旋转 (1, 4)

                # 使用 Slerp 外推：从 prev_rot2 到 prev_rot1 的旋转速度，应用到 prev_rot1
                relative_rot = slerp(prev_rot2, prev_rot1, t=1.0)  # t=1 表示完整步长
                new_rot = slerp(prev_rot1, relative_rot, t=2.0)  # t=2 表示外推一步
                self._cam_rots.data[..., curr_time_idx] = new_rot
                
                # Translation
                prev_tran1 = self._cam_trans[..., curr_time_idx-1].detach()
                prev_tran2 = self._cam_trans[..., curr_time_idx-2].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                self._cam_trans.data[..., curr_time_idx] = new_tran
                # 新的一帧的位置。
            else:
                # Initialize the camera pose for the current frame
                self._cam_rots[..., curr_time_idx] = self._cam_rots[..., curr_time_idx-1].detach()
                self._cam_rots[..., curr_time_idx] = self._cam_rots[..., curr_time_idx-1].detach()
                # 直接复制上一帧的位姿。
                

    def initialize_optimizer(self, training_args, mode="mapping"):
        """
        设置训练参数和优化器。
        Args:
            training_args: 包含学习率等训练参数的对象
            mode: "mapping" 或 "tracking"，决定优化哪些参数
        """
        # 初始化参数组
        param_groups = []

        # Mapping 模式：优化点云参数
        if mode == "mapping":
            param_groups.extend([
                {'params': [self._xyz], 'lr': training_args.means3D, "name": "_xyz"},
                {'params': [self._features_dc], 'lr': training_args.rgb_colors, "name": "_features_dc"},  # 颜色参数
                {'params': [self._opacity], 'lr': training_args.logit_opacities, "name": "_opacity"},
                {'params': [self._scaling], 'lr': training_args.log_scales, "name": "_scaling"},
                {'params': [self._rotation], 'lr': training_args.unnorm_rotations, "name": "_rotation"}
            ])
        # Tracking 模式：优化相机位姿参数
        elif mode == "tracking":
            param_groups.extend([
                {'params': [self._cam_rots], 'lr': training_args.cam_unnorm_rots, "name": "_cam_rots"},
                {'params': [self._cam_trans], 'lr': training_args.cam_trans, "name": "_cam_trans"}
            ])
        else:
            raise ValueError("Mode must be 'mapping' or 'tracking'")

        # 使用 Adam 优化器，调整 eps 以提高数值稳定性
        self.optimizer = torch.optim.Adam(param_groups, lr=training_args.optimizer_lr, eps=1e-8)

        # 添加学习率调度器以适应少量迭代（<100 次）
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        return self.optimizer, self.scheduler

    def apply_gradient_mask(self):
        """
        应用梯度掩码，仅保留 _scaling 参数的 XY 轴梯度。
        """
        for group in self.optimizer.param_groups:
            if group["name"] == "_scaling" and group["params"][0].grad is not None:
                # 创建掩码，仅保留 XY 轴的梯度 (前两列)
                grad_mask = torch.ones_like(group["params"][0].grad)
                grad_mask[:, 2:] = 0  # Z 轴及后续维度梯度置 0
                group["params"][0].grad *= grad_mask
    