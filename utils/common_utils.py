import os
import numpy as np
import random
import torch


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_means3D(output_means, output_dir):
    # Save the Parameters containing the Gaussian means
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving means3D to: {output_dir}")
    save_path = os.path.join(output_dir, "means3D.ply")
    assert output_means.shape[1] == 3, "Tensor must be of shape (N, 3)"

    points = output_means.detach().cpu().numpy()

    ply_header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
""".format(len(points))

    with open(save_path, "w") as f:
        f.write(ply_header)
        np.savetxt(f, points, fmt="%f %f %f")


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)
    
import torch
import torch.nn.functional as F

def slerp(q0, q1, t, eps=1e-8):
    """
    球面线性插值 (Slerp) 两个四元数，确保输出始终是单位四元数。
    Args:
        q0: 起始四元数，形状 (..., 4)
        q1: 目标四元数，形状 (..., 4)
        t: 插值比例，标量或张量（0 <= t <= 1 表示插值，t > 1 表示外推）
        eps: 避免除零的小值
    Returns:
        插值后的单位四元数，形状与 q0, q1 相同
    """
    # 确保输入四元数是单位向量
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    # 计算点积
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)  # (..., 1)

    # 如果点积为负，反转 q1 以走短路径
    mask = dot < 0
    q1 = torch.where(mask, -q1, q1)
    dot = torch.where(mask, -dot, dot)

    # 处理边界条件
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)  # 避免 acos 的数值问题
    theta_0 = torch.acos(dot)  # 夹角 (..., 1)
    sin_theta_0 = torch.sin(theta_0)  # sin(夹角)

    # 如果 sin_theta_0 接近零（即 q0 ≈ q1 或 q0 ≈ -q1），退化为线性插值
    if torch.all(sin_theta_0 < eps):
        qt = (1 - t) * q0 + t * q1
        return F.normalize(qt, dim=-1)

    # 计算插值角度和系数
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / (sin_theta_0 + eps)
    s1 = sin_theta_t / (sin_theta_0 + eps)

    # 插值
    qt = s0 * q0 + s1 * q1

    # 强制归一化，确保输出是单位四元数
    qt = F.normalize(qt, dim=-1)
    return qt