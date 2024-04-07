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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.ops.knn import knn_points

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def aiap_loss(xyz_can, xyz_obs, cov_can, cov_obs, rotation_can, rotation_obs, n_neighbors=21):
    _, index, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=n_neighbors, return_sorted=True)
    index = index.squeeze(0)
    rigid_loss, rot_loss = _rigid_loss(xyz_can, xyz_obs, rotation_can, rotation_obs, n_neighbors, index)
    return _aiap_loss(xyz_can, xyz_obs, index=index), _aiap_loss(cov_can, cov_obs, index=index), rigid_loss, rot_loss

def _aiap_loss(x_canonical, x_deformed, n_neighbors=6, index=None):
    if index is None:
        _, index, _ = knn_points(x_canonical.unsqueeze(0), x_canonical.unsqueeze(0), K=n_neighbors, return_sorted=True)
        index = index.squeeze(0)
    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[index])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[index])[:,0,1:]
    return F.l1_loss(dists_canonical, dists_deformed)

def _rigid_loss(x_can, x_obs, rotation_can, rotation_obs, n_neighbors, index):
    # weights = exp(2000 * ||x_can - x_can[index]||_2^2)
    weights = torch.exp(-2000 * torch.norm(x_can.unsqueeze(1) - x_can[index], dim=-1) ** 2)[:, 1:]
    vec_can = x_can[index] - x_can.unsqueeze(1)
    vec_obs = x_obs[index] - x_obs.unsqueeze(1)
    vec_can = vec_can[:, 1:, :]
    vec_obs = vec_obs[:, 1:, :]
    r = rotation_obs @ rotation_can.transpose(-1, -2)
    vec_can = r.unsqueeze(1) @ vec_can.unsqueeze(-1)
    vec_can = vec_can.squeeze(-1)

    rigid_loss = torch.sum(weights * torch.norm(vec_obs - vec_can, dim=-1))
    rigid_loss = rigid_loss / ((n_neighbors - 1) * x_can.size(0))

    rot_j = r[index]
    rot_j = rot_j[:, 1:]
    rot_i = r.unsqueeze(1)
    rot_loss = torch.sum(weights * torch.norm(rot_j - rot_i, dim=(-1, -2)))
    rot_loss = rot_loss / ((n_neighbors - 1) * x_can.size(0))

    return rigid_loss, rot_loss






def center_loss(center_can, xyz_can, center_obs, xyz_obs, e_radii, n_neighbors=10):
    # k = max(n_neighbors, int(xyz_can.size(0)/40))
    k = 10
    _, index, _ = knn_points(center_can.unsqueeze(0), xyz_can.unsqueeze(0), K=k, return_sorted=True)
    index = index.squeeze(0)

    return _center_loss(center_can, xyz_can, center_obs, xyz_obs, index=index)

def _center_loss(center_can, xyz_can, center_obs, xyz_obs, n_neighbors=11, index=None):
    if index is None:
        _, index, _ = knn_points(center_can.unsqueeze(0), xyz_can.unsqueeze(0), K=n_neighbors, return_sorted=True)
        index = index.squeeze(0)
    dists_canonical = torch.cdist(center_can.unsqueeze(1), xyz_can[index])[:,0,:]
    dists_deformed = torch.cdist(center_obs.unsqueeze(1), xyz_obs[index])[:,0,:]

    return F.l1_loss(dists_canonical, dists_deformed)

def rig_loss(rotations):
    return _rig_loss(rotations)

def _rig_loss(rotations):
    # do svd on the rotation matrices
    U, _, V = torch.svd(rotations)
    loss = l1_loss(rotations, U @ V.transpose(-1, -2))
    return loss