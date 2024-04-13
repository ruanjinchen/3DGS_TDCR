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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from F_kinematic import GaussianDeformer

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, deformer : GaussianDeformer = None, render_mask = False, add_mlp = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    n_points = means3D.shape[0]
    joints = viewpoint_camera.joints.cuda() # [num_joints]
    fwd = None
    means2D = screenspace_points
    opacity = pc.get_opacity
    # if deformer is not None:
    #     # batchify points
    #     num_point_per_batch = deformer.num_point
    #     num_batch = int(math.ceil(n_points / num_point_per_batch))
    #     means3D_batch = torch.zeros(num_batch * num_point_per_batch, 3, device=means3D.device, dtype=means3D.dtype)
    #     means3D_batch[:n_points] = means3D
    #     means3D_batch = means3D_batch.view(num_batch, num_point_per_batch, 3)
    #     # compute 16 batches at a time
    #     batch_size = 8
    #     if num_batch > 10000:
    #         fwd_list = []
    #         for i in range(0, num_batch, batch_size):
    #             fwd_list.append(deformer(means3D_batch[i:i+batch_size], joints))
    #             # torch.cuda.empty_cache()
    #         fwd = torch.cat(fwd_list, dim=0)
    #     else:
    #         fwd = deformer(means3D_batch, joints)
    #     # fwd: [num_batch, num_point_per_batch, 4, 4]
    #     fwd = fwd.view(-1, 4, 4)
    #     # only keep points' fwd
    #     fwd = fwd[:n_points]
    #     homo_coord = torch.ones(n_points, 1, dtype=means3D.dtype, device=means3D.device)
    #     x_hat_homo = torch.cat([means3D, homo_coord], dim=-1).view(n_points, 4, 1)
    #     x_bar = torch.matmul(fwd, x_hat_homo)[:, :3, 0]
    #     means3D = x_bar

    if deformer is not None:
        xyz_can = means3D
        means3D, fwd, center_transformed, cycle_loss = deformer(means3D, joints, add_mlp)
        xyz_obs = means3D
        _, center_can = deformer.get_canonical()


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp, rotation_can, rotation_obs = pc.get_covariance(scaling_modifier, fwd)
        cov3D_can, _, _ = pc.get_covariance(scaling_modifier, None)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation



    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    if render_mask:
        mask, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        mask = mask[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "mask": mask if render_mask else None,
            "xyz_can": xyz_can if deformer is not None else None,
            "xyz_obs": xyz_obs if deformer is not None else None,
            "cov_can": cov3D_can if pipe.compute_cov3D_python else None,
            "cov_obs": cov3D_precomp if pipe.compute_cov3D_python else None,
            "center_can": center_can if deformer is not None else None,
            "center_obs": center_transformed if deformer is not None else None,
            "rotation_can": rotation_can if deformer is not None else None,
            "rotation_obs": rotation_obs if deformer is not None else None,
            "cycle_loss": cycle_loss if deformer is not None else None}

