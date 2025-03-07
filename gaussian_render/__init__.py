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
import random
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from recon_utils.sh_utils import eval_sh

def render(viewpoint_camera, 
           gaussians : GaussianModel, 
        #  pipe, 
        #  bg_color : torch.Tensor, 
        #  patch_size: list, 
        #  scaling_modifier = 1.0, 
           override_color = None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # viewpoint_camera.to_device()
    # viewpoint_camera.update()
    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=viewpoint_camera.tanfovx,
        tanfovy=viewpoint_camera.tanfovy,
        bg=viewpoint_camera.bg,
        scale_modifier=viewpoint_camera.scale_modifier,
        viewmatrix=viewpoint_camera.viewmatrix,
        projmatrix=viewpoint_camera.projmatrix,
        patch_bbox=random_patch(viewpoint_camera.image_height, viewpoint_camera.image_width, viewpoint_camera.patch_size[0], viewpoint_camera.patch_size[1]),
        prcppoint=viewpoint_camera.prcppoint,
        sh_degree=gaussians.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=viewpoint_camera.debug,
        config=viewpoint_camera.config
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = gaussians.get_covariance(scaling_modifier)
    # else:
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    # print(pc._scaling)
    # print(scales)
    # # print(rotations)
    # exit()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # if pipe.convert_SHs_python:
        #     shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        #     dir_pp = (gaussians.get_xyz - viewpoint_camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        #     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #     sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        #     colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # else:
        shs = gaussians.get_features
    else:
        colors_precomp = override_color


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_normal, rendered_depth, rendered_opac, radii = rasterizer(
                                                                            means3D = means3D,
                                                                            means2D = means2D,
                                                                            shs = shs,
                                                                            colors_precomp = colors_precomp,
                                                                            opacities = opacity,
                                                                            scales = scales,
                                                                            rotations = rotations,
                                                                            cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "normal": rendered_normal, 
            "depth": rendered_depth,
            "opac": rendered_opac,
            "viewspace_points": screenspace_points, # 表示渲染过程中物体点投影到屏幕空间的坐标。3D 到 2D的坐标。
            "visibility_filter" : radii > 1, # 过滤不可见的点。
            "radii": radii} # 半径

def random_patch(h, w, h_size=float('inf'), w_size=float('inf')):
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).cuda()