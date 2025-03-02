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
import numpy as np
from recon_utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, normal2rotation
from torch import nn
from utils.common_utils import slerp
from torch.utils.cpp_extension import load
from recon_utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from recon_utils.general_utils import quaternion2rotmat
from recon_utils.image_utils import world2scrn
from recon_utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from utils.recon_helpers import setup_camera, energy_mask
from utils.slam_helpers import transform_to_frame_3d, transformed_params2depthplussilhouette


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
        self.max_sh_degree = args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._cam_rots = torch.empty(0) # 3D camera rotations
        self._cam_trans = torch.empty(0) # 3D camera translations
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.rot_gradient_accum = torch.empty(0)
        self.opac_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.setup_functions()
        self.utils_mod = load(name="cuda_utils", sources=["utils/ext.cpp", "utils/cuda_utils.cu"])
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
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.rot_gradient_accum,
            self.opac_gradient_accum,
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
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup(training_args)
        self.denom = denom
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

    def initialize_camera_pose(self, curr_time_idx, forward_prop):
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

    # def initialize_new_params(self, new_pt_cld, mean3_sq_dist, use_simplification):
    #     num_pts = new_pt_cld.shape[0]
    #     means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    #     unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    #     logit_opacities = torch.ones((num_pts, 1), dtype=torch.float, device="cuda") * 0.5
    #     params = {
    #         'means3D': means3D,
    #         'rgb_colors': new_pt_cld[:, 3:6],
    #         'unnorm_rotations': unnorm_rots,
    #         'logit_opacities': logit_opacities,
    #         'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 3)),
    #     }
    #     # if not use_simplification:
    #     #     params['feature_rest'] = torch.zeros(num_pts, 45) # set SH degree 3 fixed
    #     for k, v in params.items():
    #         # Check if value is already a torch tensor
    #         if not isinstance(v, torch.Tensor):
    #             params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
    #         else:
    #             params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    #     return params

    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def reset_opacity(self, ratio, iteration):
        # if len(self._xyz) < self.opac_reset_record[0] * 1.05 and iteration < self.opac_reset_record[1] + 3000:
        #     print(len(self._xyz), self.opac_reset_record, 'notreset')
        #     return
        # print(len(self._xyz), self.opac_reset_record, 'reset')
        # self.opac_reset_record = [len(self._xyz), iteration]

        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * ratio))
        opacities_new = inverse_sigmoid(self.get_opacity * ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    # def _prune_optimizer(self, mask):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors
    

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.rot_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(selected_pts_mask.dtype, pre_mask.dtype)
        # selected_pts_mask *= pre_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if self.config[0] > 0:
            new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask += (grad_rot > grad_rot_thrsh).squeeze()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        selected_pts_mask *= pre_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_opacities, new_scaling, new_rotation)

    def adaptive_prune(self, min_opacity, extent):

        # print(sum(grad_rot > 1.2) / len(grad_rot))
        # print(sum(grad_pos > max_grad) / len(grad_pos), max_grad)

        n_ori = len(self._xyz)

        # prune
        # prune_mask = 
        # opac_thrsh = torch.tensor([min_opacity, 1])
        opac_temp = self.get_opacity
        prune_opac =  (opac_temp < min_opacity).squeeze()
        # prune_opac += (opac_temp > opac_thrsh[1]).squeeze()

        # scale_thrsh = torch.tensor([2e-4, 0.1]) * extent
        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2)
        # print(prune_scale.sum())
        
        prune_vis = (self.denom == 0).squeeze()
        prune = prune_opac + prune_vis + prune_scale
        self.prune_points(prune)
        # print(f'opac:{prune_opac.sum()}, scale:{prune_scale.sum()}, vis:{prune_vis.sum()} extend:{extent}')
        # print(f'prune: {n_ori}-->{len(self._xyz)}')

    def adaptive_densify(self, max_grad, extent):
        grad_pos = self.xyz_gradient_accum / self.denom
        grad_scale = self.scale_gradient_accum /self.denom
        grad_rot = self.rot_gradient_accum /self.denom
        grad_opac = self.opac_gradient_accum /self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        grad_scale[grad_scale.isnan()] = 0.0
        grad_rot[grad_rot.isnan()] = 0.0
        grad_opac[grad_opac.isnan()] = 0.0


        # densify
        # opac_lr = [i['lr'] for i in self.optimizer.param_groups if i['name'] == 'opacity'][0]
        larger = torch.le(grad_scale, 1e-7)[:, 0] #if opac_lr == 0 else True
        # print(grad_opac.min(), grad_opac.max(), grad_opac.mean())
        denser = torch.le(grad_opac, 2)[:, 0]
        pre_mask = denser * larger
        
        self.densify_and_clone(grad_pos, max_grad, extent, pre_mask=pre_mask)
        self.densify_and_split(grad_pos, max_grad, extent)


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(self.xyz_gradient_accum.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # print(self.xyz_gradient_accum.shape)
        # print(self._scaling.grad.shape)
        # exit()
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        # print(self._scaling.grad)
        self.rot_gradient_accum[update_filter] += torch.norm(self._rotation[update_filter], dim=-1, keepdim=True)
        self.opac_gradient_accum[update_filter] += self._opacity[update_filter]
        self.denom[update_filter] += 1

    def mask_prune(self, cams, pad=4):
        batch_size = 32
        batch_num = len(cams) // batch_size + int(len(cams) % batch_size != 0)
        cams_batch = [cams[i * batch_size : min(len(cams), (i + 1) * batch_size)] for i in range(batch_num)]
        # 使用列表推导式生成了一个新列表 cams_batch，其中每一个元素都是从 cams 中提取的一个子列表。
        for c in cams_batch:
            _, _, inMask, outView = world2scrn(self._xyz.detach(), c, pad)
            visible = inMask.all(0) * ~(outView.all(0))
            if list(visible.shape) != []:
                self.prune_points(~visible)
        # 移除在各个视角中都不可见的点。

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # print(xyz_min, xyz_max, xyz_len)
        
        # grid_dim_max = 1024
        grid_len = xyz_len.max() / grid_dim_max
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               self._xyz, self.get_rotation, self.get_scaling, self.get_opacity,
                                               torch.tensor([cutoff]).to(torch.float32).cuda())
        
        
        return grid, -xyz_min, 1 / grid_len, grid_dim