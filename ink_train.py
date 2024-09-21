import os

from pytorch3d.loss import chamfer_distance
import torch
from random import randint
import random

import torchvision

from utils.loss_utils import l1_loss, ssim, aiap_loss, center_loss, sparse_loss
from gaussian_renderer import render, network_gui, render_obstacle
import sys
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from F_kinematic import GaussianDeformer

from utils.camera_utils import camera_from_camInfo
from lpipsPyTorch import lpips
import matplotlib.pyplot as plt

def inv_training(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.prune(min_opacity=0.005)
    deformer = GaussianDeformer(dataset.joints, use_mlp=False)
    deformer.cuda()
    # if iteration == -1:
    #     it = searchForMaxIteration(dataset.model_path)
    # path = os.path.join(dataset.model_path, "chkpnt_{}".format(it)+"pth")
    path = dataset.model_path + "/chkpnt_" + str(scene.loaded_iter) + ".pth"
    (model_params, transform_params, transform_opt_params, transform_sch_params, first_iter) = torch.load(
        path)
    deformer.load(transform_params)
    deformer.eval()
    # freeze the deformer
    for param in deformer.parameters():
        param.requires_grad = False
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTestCameras()
    # pick a pair of views
    length = len(views) // 2
    idx1 = randint(0, length - 1)
    idx1 = 34 // 2
    view = views[idx1 * 2: idx1 * 2 + 2]
    # view = camera_from_camInfo(view, 1.0, dataset)
    print("Rendering view: ", view[0].image_name)
    limit = 90 / 180 * 3.1415926
    print("Original joints: ", view[0].joints / limit)
    # use pytorch to automatically compute joints angles, initialize with random values between -1 and 1
    joints = torch.zeros((dataset.joints,), device="cuda")
    joints.requires_grad = True
    optimizer = torch.optim.AdamW([joints], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # torch.autograd.detect_anomaly(True)
    for i in range(10000):
        optimizer.zero_grad()
        loss = 0
        j = 0
        for v in view:
            v = camera_from_camInfo(v, 1.0, dataset)
            v.joints = torch.sin(joints)
            results = render(v, gaussians, pipeline, background, deformer=deformer, render_mask=True, add_mlp=True,
                         render_bone=True)
            rendering = results["renders"]
            torchvision.utils.save_image(rendering, "./inv" + str(j) + "/rendering" + str(i) + ".png")
            gt_image = v.original_image.cuda()
            Ll1 = l1_loss(rendering, gt_image)
            loss += 0.2 * (1.0 - ssim(rendering, gt_image)) + 0.8 * Ll1
            loss += lpips(rendering, gt_image, net_type='vgg').squeeze()
            j += 1
        # keep the joints within the limit
        # loss += 0.1 * torch.sum(torch.relu(torch.abs(joints) - 1))
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print("Loss: ", loss.item())
            print("Joints: ", torch.sin(joints) * limit)

    print("Joints: ", torch.sin(joints))


def end_control(dataset : ModelParams, iteration : int, pipeline : PipelineParams, end_part=4):
    target_positon = torch.tensor([0.3, 0.6, 0.6], device="cuda")
    avoid_position = torch.tensor([ 0.1888,  0.2026,  0.6085], device="cuda")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.prune(min_opacity=0.005)
    deformer = GaussianDeformer(dataset.joints, use_mlp=False)
    deformer.cuda()
    # if iteration == -1:
    #     it = searchForMaxIteration(dataset.model_path)
    # path = os.path.join(dataset.model_path, "chkpnt_{}".format(it)+"pth")
    path = dataset.model_path + "/chkpnt_" + str(scene.loaded_iter) + ".pth"
    (model_params, transform_params, transform_opt_params, transform_sch_params, first_iter) = torch.load(
        path)
    deformer.load(transform_params)
    deformer.eval()
    # freeze the deformer
    for param in deformer.parameters():
        param.requires_grad = False
    joints = torch.zeros((dataset.joints,), device="cuda")
    joints.requires_grad = True
    optimizer = torch.optim.AdamW([joints], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    pc = gaussians.get_xyz
    views = scene.getTrainCameras()
    # views = views[0:2]
    # view = views[0]
    # view = camera_from_camInfo(view, 1.0, dataset)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    dis_min = []
    for i in range(500):
        with torch.no_grad():
            j = 0
            view = views[0]

            view = camera_from_camInfo(view, 1.0, dataset)
            view.joints = torch.sin(joints)
            results = render(view, gaussians, pipeline, background, deformer=deformer, render_mask=True, add_mlp=True,
                             render_bone=True)
            rendering = results["renders"]
            position = avoid_position.unsqueeze(0)
            radius = torch.tensor([0.05, 0.05, 0.05], device="cuda").unsqueeze(0)
            color = [1, 0, 0]
            obstacle = render_obstacle(view, pipeline, background, position=position, radius=radius, color=color)
            target = render_obstacle(view, pipeline, background, position=target_positon.unsqueeze(0), radius=radius, color=[0, 1, 0])
            # put the obstacle in the rendering
            rendering = rendering + target + obstacle
            #save the rendering
            torchvision.utils.save_image(rendering, "./reach_target/renderingobs" + str(i) + ".png")
            j += 1
        optimizer.zero_grad()
        loss = 0
        pc_transformed, _, bone_translation, _, _, _, weights = deformer(pc, torch.sin(joints))

        loss += torch.functional.F.mse_loss(bone_translation[end_part], target_positon)
        # robot arm should avoid the avoid_position, the distance between the avoid_position and the point cloud of the robot arm should be larger than 0.1
        with torch.no_grad():
            dis = torch.norm(pc_transformed - avoid_position, dim=1) - 0.05

            dis_min.append(torch.min(dis).detach().cpu().numpy())

        test = torch.max(torch.zeros(pc_transformed.shape[0], device='cuda'), 0.06 - torch.norm(pc_transformed - avoid_position, dim=1))
        loss += torch.sum(test)



        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print("Loss: ", loss.item())
            print("End Position: ", bone_translation[end_part])
            print("joints: ", torch.sin(joints) * 90 / 180 * 3.1415926)
            # torchvision.utils.save_image(rendering, "renderingobs" + str(i) + ".png")

    # generate a video
    os.system("ffmpeg -r 10 -i ./reach_target_avoid/renderingobs%d.png -vcodec mpeg4 -y reach_target_avoid.mp4")
    # draw a picture of dis_min

    plt.plot(dis_min)
    plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    plt.savefig("dis_min.png")
    plt.show()






if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=600000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    end_control(args, args.iteration, pipeline)