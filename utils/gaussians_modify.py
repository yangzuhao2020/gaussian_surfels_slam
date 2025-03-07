"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""
import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q):
    """四元数 q 转换为 3×3 旋转矩阵 rot"""
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    """累积 means2D 的梯度的 2D 范数，并更新计数，常用于优化过程中统计梯度信息。"""
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1) # 计算 2D 梯度的范数
    variables['denom'][variables['seen']] += 1 # 更新计数 用于记录每个点被累积的次数
    return variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(gaussians, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0): # 按周期执行裁剪
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(gaussians._opacity) < remove_threshold).squeeze()
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(gaussians._scaling).max(dim=1).values > 0.1 * variables['scene_radius'] # 最大维度超过场景半径 10% 的点
                to_remove = torch.logical_or(to_remove, big_points_ws)
            gaussians, variables = remove_points(to_remove, gaussians, variables, optimizer)
            torch.cuda.empty_cache()
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            # 创建新的不透明度参数
            new_opacity = inverse_sigmoid(torch.ones_like(gaussians._opacity) * 0.01)
            # 使用类似 update_params_and_optimizer 的逻辑更新 _opacity 和优化器状态
            group = [x for x in optimizer.param_groups if x["name"] == "opacity"][0]
            stored_state = optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                # 重置动量和平方梯度
                stored_state["exp_avg"] = torch.zeros_like(new_opacity, device=new_opacity.device)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_opacity, device=new_opacity.device)
                del optimizer.state[group['params'][0]]
            else:
                stored_state = {
                    "exp_avg": torch.zeros_like(new_opacity, device=new_opacity.device),
                    "exp_avg_sq": torch.zeros_like(new_opacity, device=new_opacity.device)
                }

            # 更新 _opacity 为新的参数
            group["params"][0] = torch.nn.Parameter(new_opacity.requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            gaussians._opacity = group["params"][0]  # 更新 gaussians 的 _opacity 属性
    
    return gaussians, variables


def update_params_and_optimizer(new_gaussians, gaussians, optimizer):
    for k, v in new_gaussians.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        setattr(gaussians, k, group["params"][0])  # 更新 gaussians 对象的属性
    return gaussians

def cat_gaussians_to_optimizer(new_gaussians, gaussians, optimizer):
    for attr in ['_xyz', '_features_dc', '_opacity', '_scaling', '_rotation']:
        new_val = new_gaussians[attr]  # 从字典中获取值
        old_val = getattr(gaussians, attr)  # 从对象中获取属性值

        group = [g for g in optimizer.param_groups if g['name'] == attr][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(new_val)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(new_val)), dim=0)
            del optimizer.state[group['params'][0]]

            group["params"][0] = torch.nn.Parameter(torch.cat((old_val, new_val), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            setattr(gaussians, attr, group["params"][0])  # 更新 gaussians 对象
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((old_val, new_val), dim=0).requires_grad_(True))
            setattr(gaussians, attr, group["params"][0])  # 直接更新 gaussians 对象
    
    return gaussians


def remove_points(to_remove, gaussians, variables, optimizer):
    """
    从高斯点云中移除指定点，并更新相关参数、优化器状态和辅助变量。
    Args:
        to_remove: 布尔张量，表示需要移除的点 (shape: (N,))
        gaussians: 高斯点云对象，包含点云属性
        variables: 辅助变量字典
        optimizer: Adam 优化器
    Returns:
        gaussians, variables: 更新后的对象和变量
    """
    # 检查 to_remove 是否为布尔张量且形状匹配
    if not isinstance(to_remove, torch.Tensor) or to_remove.dtype != torch.bool:
        raise TypeError("to_remove must be a boolean torch.Tensor")
    if to_remove.shape[0] != gaussians.xyz.shape[0]:
        raise ValueError(f"to_remove shape {to_remove.shape} must match point cloud size {gaussians.xyz.shape[0]}")

    to_keep = ~to_remove
    keys = ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0] # 提取第一个匹配的组
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(getattr(gaussians, k)[to_keep].requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            setattr(gaussians, k, group["params"][0])
        else:
            group["params"][0] = torch.nn.Parameter(getattr(gaussians, k)[to_keep].requires_grad_(True))
            setattr(gaussians, k, group["params"][0])
            
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    if 'timestep' in variables.keys():
        variables['timestep'] = variables['timestep'][to_keep]
    return gaussians, variables



def densify(gaussians, variables, optimizer, iter, densify_dict):
    if iter <= densify_dict['stop_after']:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict['grad_thresh']
        if (iter >= densify_dict['start_after']) and (iter % densify_dict['densify_every'] == 0): # 
            grads = variables['means2D_gradient_accum'] / variables['denom'] # 计算每个点的平均梯度范数
            grads[grads.isnan()] = 0.0

            # 计算需要克隆的点：梯度大于阈值，同时对应的尺度（经过 exp 后）较小。
            to_clone = torch.logical_and(
                grads >= grad_thresh,
                torch.max(torch.exp(gaussians._scaling), dim=1).values <= 0.01 * variables['scene_radius'])
            # 克隆这些点（排除不需要的属性，如相机姿态等，本例中直接使用类方法 clone）
            
            new_gaussians = {
            '_xyz': gaussians._xyz[to_clone],
            '_features_dc': gaussians._features_dc[to_clone],
            '_opacity': gaussians._opacity[to_clone],
            '_scaling': gaussians._scaling[to_clone],
            '_rotation': gaussians._rotation[to_clone]
            }
            # 将克隆得到的新点云“拼接”到优化器中（内部实现类似于 torch.cat）
            gaussians = cat_gaussians_to_optimizer(new_gaussians, gaussians, optimizer)
            num_pts = gaussians._xyz.shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads

            # 找出需要拆分的点，拆分条件与克隆条件互补
            to_split = torch.logical_and(
                padded_grad >= grad_thresh,
                torch.max(torch.exp(gaussians._scaling), dim=1).values > 0.01 * variables['scene_radius'])
            n = densify_dict['num_to_split_into']  # 拆分后的复制数量
            new_gaussians = {
            '_xyz': gaussians._xyz[to_split].repeat(n, 1),
            '_features_dc': gaussians._features_dc[to_split].repeat(n, 1),
            '_opacity': gaussians._opacity[to_split].repeat(n, 1),
            '_scaling': gaussians._scaling[to_split].repeat(n, 1),
            '_rotation': gaussians._rotation[to_split].repeat(n, 1)}
            # 克隆拆分点并重复 n 倍
            # 对拆分点计算标准差（由 log_scaling 得到）并生成正态噪声
            stds = torch.exp(gaussians._scaling)[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(gaussians._rotation[to_split]).repeat(n, 1, 1)
            # 对新点云的 3D 坐标进行扰动
            new_gaussians['_xyz'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            # 调整新点云的尺度：将原有尺度除以 (0.8 * n) 并取对数
            new_gaussians['_scaling'] = torch.log(torch.exp(new_gaussians['_scaling']) / (0.8 * n))
            gaussians = cat_gaussians_to_optimizer(new_gaussians, gaussians, optimizer)
            num_pts = gaussians._xyz.shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            gaussians, variables = remove_points(to_remove, gaussians, variables, optimizer)

            if iter == densify_dict['stop_after']:
                remove_threshold = densify_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = densify_dict['removal_opacity_threshold']
            # 根据透明度阈值确定需要删除的点
            to_remove = (torch.sigmoid(gaussians._opacity) < remove_threshold).squeeze()
            if iter >= densify_dict['remove_big_after']:
                big_points_ws = torch.exp(gaussians._scaling).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            gaussians, variables = remove_points(to_remove, gaussians, variables, optimizer)

            torch.cuda.empty_cache()

        # 重置所有高斯的透明度（这一步在只映射当前帧时可能不需要）
        # 重置所有高斯点的不透明度
        if iter > 0 and iter % densify_dict['reset_opacities_every'] == 0 and densify_dict['reset_opacities']:
            new_gaussians = {'logit_opacities': inverse_sigmoid(torch.ones_like(gaussians._opacity) * 0.01)}
            gaussians = update_params_and_optimizer(new_gaussians, gaussians, optimizer)

    return gaussians, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = means3D_scheduler(iteration)
                param_group['lr'] = lr
                return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper